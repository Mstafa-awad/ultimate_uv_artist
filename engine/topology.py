"""Mesh topology: exact welding, edge/face adjacency, diagnostics.

Nothing here changes the input geometry. Welding is an *analysis* proxy:
the output mesh always keeps the original vertices, triangle order and
per-corner attributes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np


@dataclass
class Topo:
    V: np.ndarray        # (n,3) float64 welded positions
    F: np.ndarray        # (m,3) int64 welded triangle indices
    wid: np.ndarray      # (N,) original vertex -> welded vertex
    E: np.ndarray        # (ne,2) edge endpoints (lo,hi)
    FE: np.ndarray       # (m,3) edge id of face edge k (F[k] -> F[k+1])
    EF: np.ndarray       # (ne,2) first two incident faces (-1 none)
    ec: np.ndarray       # (ne,) incident face count
    FF: np.ndarray       # (m,3) neighbour across edge k (-1 = boundary/cut)
    degenerate: np.ndarray   # (m,) bool, excluded from charting (incl. duplicate copies)
    edge_len: np.ndarray     # (ne,)
    face_area: np.ndarray    # (m,)
    face_normal: np.ndarray  # (m,3)
    diag: dict
    dup_of: np.ndarray = None    # (m,) primary triangle of an exact duplicate copy, else -1


def weld_exact(vertices):
    v = np.ascontiguousarray(vertices)
    # exact position weld (bitwise equal float32 triples)
    key = v.view(np.dtype((np.void, v.dtype.itemsize * 3))).reshape(-1)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return np.ascontiguousarray(v[first], np.float64), inv.reshape(-1).astype(np.int64), first


@nb.njit(cache=True)
def _face_adjacency(F, FE, EF, ec, consistent, deg):
    m = F.shape[0]
    FF = -np.ones((m, 3), np.int64)
    for f in range(m):
        if deg[f]:
            continue
        for k in range(3):
            e = FE[f, k]
            if ec[e] != 2 or not consistent[e]:
                continue
            g = EF[e, 0] if EF[e, 0] != f else EF[e, 1]
            if g < 0 or deg[g]:
                continue
            FF[f, k] = g
    return FF


GPU_MIN_FACES = 300_000


def build_topology(vertices, faces, split_edges_mask=None, use_gpu=True):
    """Weld + adjacency + diagnostics. Large meshes use the GPU (sorting on
    CUDA); results are identical in meaning, numbering may differ."""
    if use_gpu and len(faces) >= GPU_MIN_FACES:
        try:
            return _build_topology_gpu(vertices, faces)
        except Exception as exc:          # OOM or no CUDA: CPU path
            import logging
            logging.getLogger("UltimateUVArtist").info("GPU topology unavailable (%s); using CPU", exc)
    return _build_topology_cpu(vertices, faces)


def _build_topology_gpu(vertices, faces):
    import torch
    from .gpu import device
    dev = device()
    if dev.type != "cuda":
        raise RuntimeError("no CUDA device")
    try:
        vt = torch.from_numpy(np.ascontiguousarray(vertices, np.float32)).to(dev)
        Vu, wid = torch.unique(vt, dim=0, return_inverse=True)
        del vt
        F = wid[torch.from_numpy(np.ascontiguousarray(faces, np.int64)).to(dev)]
        m = F.shape[0]
        V64 = Vu.double()
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        rep_idx = (a == b) | (b == c) | (c == a)
        P0, P1, P2 = V64[a], V64[b], V64[c]
        nrm = torch.cross(P1 - P0, P2 - P0, dim=1)
        dbl = nrm.norm(dim=1)
        l2max = torch.maximum(torch.maximum(((P1 - P0) ** 2).sum(1), ((P2 - P1) ** 2).sum(1)), ((P0 - P2) ** 2).sum(1))
        del P0, P1, P2
        zero = dbl <= 1e-12 * l2max.clamp_min(1e-300)
        deg = rep_idx | zero
        sf, _ = torch.sort(F, dim=1)
        _, inv_dup = torch.unique(sf, dim=0, return_inverse=True)
        del sf
        ar = torch.arange(m, device=dev)
        first_idx = torch.full((int(inv_dup.max()) + 1,), m, device=dev, dtype=torch.int64)
        first_idx.scatter_reduce_(0, inv_dup, ar, reduce="amin")
        primary = first_idx[inv_dup]
        dup_of = torch.where((primary != ar) & ~deg & ~deg[primary], primary, torch.full_like(primary, -1))
        secondary = dup_of >= 0
        deg = deg | secondary
        normal = nrm / dbl.clamp_min(1e-300)[:, None]
        nV = Vu.shape[0]
        he_a = torch.cat([a, b, c]); he_b = torch.cat([b, c, a])
        lo = torch.minimum(he_a, he_b); hi = torch.maximum(he_a, he_b)
        key = lo * nV + hi
        uk, eid, _ = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
        ne = uk.shape[0]
        FE = eid.reshape(3, m).T.contiguous()
        face_of_he = ar.repeat(3)
        valid_he = ~deg[face_of_he]
        ec = torch.bincount(eid[valid_he], minlength=ne)
        order = torch.sort(eid, stable=True).indices
        ov = order[valid_he[order]]
        es = eid[ov]; fs = face_of_he[ov]
        first = torch.ones_like(es, dtype=torch.bool); first[1:] = es[1:] != es[:-1]
        EF = torch.full((ne, 2), -1, device=dev, dtype=torch.int64)
        EF[es[first], 0] = fs[first]
        sec = torch.zeros_like(first); sec[1:] = (es[1:] == es[:-1]) & first[:-1]
        EF[es[sec], 1] = fs[sec]
        E = torch.stack([uk // nV, uk % nV], 1)
        dir_pos = he_a < he_b
        sgn = torch.zeros(ne, device=dev, dtype=torch.int64)
        sgn.index_add_(0, eid[valid_he], torch.where(dir_pos[valid_he], 1, -1))
        two = ec == 2
        consistent = torch.ones(ne, device=dev, dtype=torch.bool)
        consistent[two] = sgn[two] == 0
        # face adjacency across manifold, consistently oriented edges
        e = FE
        g0 = EF[e, 0]; g1 = EF[e, 1]
        g = torch.where(g0 == ar[:, None], g1, g0)
        okf = (ec[e] == 2) & consistent[e] & ~deg[:, None] & (g >= 0)
        okf &= ~deg[g.clamp_min(0)]
        FF = torch.where(okf, g, torch.full_like(g, -1))
        edge_len = (V64[E[:, 0]] - V64[E[:, 1]]).norm(dim=1)
        cpu = lambda t: t.cpu().numpy()
        diag = {
            "input_vertices": int(len(vertices)), "welded_vertices": int(nV), "triangles": int(m),
            "degenerate_triangles": int((rep_idx | zero).sum()), "repeated_index_triangles": int(rep_idx.sum()),
            "zero_area_triangles": int((zero & ~rep_idx).sum()), "duplicate_triangles": int(secondary.sum()),
            "boundary_edges": int((ec == 1).sum()), "nonmanifold_edges": int((ec > 2).sum()),
            "inconsistent_orientation_edges": int((two & ~consistent).sum()),
            "surface_area": float((0.5 * dbl).sum()),
        }
        lmax = l2max.clamp_min(1e-300).sqrt()
        needles = int((((dbl / lmax) / lmax < 1e-3) & ~deg).sum())
        topo = Topo(cpu(V64), cpu(F), cpu(wid), cpu(E), cpu(FE), cpu(EF), cpu(ec), cpu(FF), cpu(deg),
                    cpu(edge_len), cpu(0.5 * dbl), cpu(normal), diag, cpu(dup_of))
    finally:
        torch.cuda.empty_cache()
    comp = face_components(topo.FF, topo.degenerate)
    diag["components"] = int(len(np.unique(comp[~topo.degenerate]))) if (~topo.degenerate).any() else 0
    diag["needle_triangles"] = needles
    return topo, comp


def _build_topology_cpu(vertices, faces):
    V, wid, _ = weld_exact(np.asarray(vertices, np.float32))
    F = wid[np.asarray(faces, np.int64)]
    m = len(F)
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    rep_idx = (a == b) | (b == c) | (c == a)
    P0, P1, P2 = V[a], V[b], V[c]
    n = np.cross(P1 - P0, P2 - P0)
    dbl = np.linalg.norm(n, axis=1)
    area = 0.5 * dbl
    # zero-area faces: relative to squared longest edge
    l2max = np.maximum.reduce([((P1 - P0) ** 2).sum(1), ((P2 - P1) ** 2).sum(1), ((P0 - P2) ** 2).sum(1)])
    zero = dbl <= 1e-12 * np.maximum(l2max, 1e-300)
    deg = rep_idx | zero
    # exact duplicate triangles (same welded vertex set, any winding): keep the
    # first as primary; secondaries are excluded from adjacency so they do not
    # make the primary's edges non-manifold. They get UVs later (shadow islands).
    sf = np.sort(F, axis=1)
    sk = np.ascontiguousarray(sf).view(np.dtype((np.void, 24))).reshape(-1)
    _, first_idx, inv_dup = np.unique(sk, return_index=True, return_inverse=True)
    inv_dup = inv_dup.reshape(-1)
    primary = first_idx[inv_dup]
    dup_of = np.where((primary != np.arange(m)) & ~deg & ~deg[primary], primary, -1)
    secondary = dup_of >= 0
    deg = deg | secondary
    normal = n / np.maximum(dbl, 1e-300)[:, None]
    # half edges
    he_a = np.concatenate([a, b, c]); he_b = np.concatenate([b, c, a])
    lo = np.minimum(he_a, he_b); hi = np.maximum(he_a, he_b)
    nV = len(V)
    key = lo * nV + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.r_[True, ks[1:] != ks[:-1]]
    eid_sorted = np.cumsum(start) - 1
    ne = int(eid_sorted[-1]) + 1
    eid = np.empty_like(eid_sorted); eid[order] = eid_sorted
    FE = eid.reshape(3, m).T.copy()
    face_of_he = np.tile(np.arange(m), 3)
    # count only non-degenerate incidences
    valid_he = ~deg[face_of_he]
    ec = np.bincount(eid[valid_he], minlength=ne)
    EF = -np.ones((ne, 2), np.int64)
    # first and second incident face per edge (in sorted order)
    he_sorted_valid = order[valid_he[order]]
    es = eid[he_sorted_valid]
    fs = face_of_he[he_sorted_valid]
    first = np.r_[True, es[1:] != es[:-1]]
    EF[es[first], 0] = fs[first]
    sec = np.r_[False, (es[1:] == es[:-1]) & first[:-1]]
    EF[es[sec], 1] = fs[sec]
    E = np.stack([np.zeros(ne, np.int64), np.zeros(ne, np.int64)], 1)
    E[eid, 0] = lo; E[eid, 1] = hi
    # orientation consistency for 2-face edges: directions must be opposite
    dir_pos = (he_a < he_b)
    consistent = np.ones(ne, bool)
    two = ec == 2
    sgn = np.zeros(ne, np.int64)
    np.add.at(sgn, eid[valid_he], np.where(dir_pos[valid_he], 1, -1))
    consistent[two] = sgn[two] == 0
    FF = _face_adjacency(F, FE, EF, ec, consistent, deg)
    edge_len = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    diag = {
        "input_vertices": int(len(vertices)), "welded_vertices": int(nV), "triangles": int(m),
        "degenerate_triangles": int((rep_idx | zero).sum()), "repeated_index_triangles": int(rep_idx.sum()),
        "zero_area_triangles": int((zero & ~rep_idx).sum()),
        "duplicate_triangles": int(secondary.sum()),
        "boundary_edges": int((ec == 1).sum()), "nonmanifold_edges": int((ec > 2).sum()),
        "inconsistent_orientation_edges": int((two & ~consistent).sum()),
        "surface_area": float(area.sum()),
    }
    topo = Topo(V, F, wid, E, FE, EF, ec, FF, deg, edge_len, area, normal, diag, dup_of)
    comp = face_components(FF, deg)
    diag["components"] = int(len(np.unique(comp[~deg]))) if (~deg).any() else 0
    # thin triangles: min altitude / max edge < 1e-3
    lmax = np.sqrt(np.maximum(l2max, 1e-300))
    diag["needle_triangles"] = int((((dbl / lmax) / lmax < 1e-3) & ~deg).sum())
    return topo, comp


@nb.njit(cache=True)
def _components(FF, deg):
    m = FF.shape[0]
    lab = -np.ones(m, np.int64)
    stack = np.empty(m, np.int64)
    nc = 0
    for s in range(m):
        if lab[s] >= 0:
            continue
        lab[s] = nc
        if deg[s]:
            nc += 1
            continue
        top = 0
        stack[top] = s; top += 1
        while top > 0:
            top -= 1
            f = stack[top]
            for k in range(3):
                g = FF[f, k]
                if g >= 0 and lab[g] < 0:
                    lab[g] = nc
                    stack[top] = g; top += 1
        nc += 1
    return lab


def face_components(FF, deg):
    return _components(FF, deg)


def dihedral(topo):
    """Signed dihedral angle per edge (radians); + convex, - concave, 0 boundary."""
    e = np.arange(len(topo.E))
    f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
    ok = (topo.ec == 2) & (f0 >= 0) & (f1 >= 0)
    ang = np.zeros(len(e))
    n0 = topo.face_normal[np.maximum(f0, 0)]; n1 = topo.face_normal[np.maximum(f1, 0)]
    cosv = np.clip((n0 * n1).sum(1), -1, 1)
    a = np.arccos(cosv)
    # convexity: centroid of f1 below plane of f0 => convex
    c1 = topo.V[topo.F[np.maximum(f1, 0)]].mean(1)
    p0 = topo.V[topo.E[:, 0]]
    s = ((c1 - p0) * n0).sum(1)
    ang[ok] = np.where(s[ok] <= 0, a[ok], -a[ok])
    return ang
