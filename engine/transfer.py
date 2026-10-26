"""Proxy -> full-resolution UV transfer for high-poly meshes.

Every full-resolution triangle is assigned to the nearest normal-compatible
proxy triangle. Its corners are mapped through that proxy triangle's affine
3D->UV map (positions projected onto the proxy triangle plane, barycentric
extrapolation allowed). Corners are welded into chart vertices across edges
whose two sides agree on the UV of both shared vertices; where they disagree
(the proxy's seams and relief cuts) the edge becomes a seam. The result keeps
the proxy's validated layout and seams while every original triangle gets its
own exact UVs.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, parallel=True)
def corner_uvs(V, F, faces, pick, pV, pF, pgl, Ucat):
    n = faces.shape[0]
    out = np.empty((n, 3, 2))
    for i in nb.prange(n):
        f = faces[i]
        t = pick[i]
        a = pV[pF[t, 0]]; b = pV[pF[t, 1]]; c = pV[pF[t, 2]]
        ua = Ucat[pgl[t, 0]]; ub = Ucat[pgl[t, 1]]; uc = Ucat[pgl[t, 2]]
        e1 = b - a; e2 = c - a
        d11 = (e1 * e1).sum(); d12 = (e1 * e2).sum(); d22 = (e2 * e2).sum()
        den = d11 * d22 - d12 * d12
        for k in range(3):
            p = V[F[f, k]] - a
            if den <= 1e-300:
                out[i, k, 0] = ua[0]; out[i, k, 1] = ua[1]
                continue
            q1 = (p * e1).sum(); q2 = (p * e2).sum()
            b1 = (d22 * q1 - d12 * q2) / den
            b2 = (d11 * q2 - d12 * q1) / den
            out[i, k, 0] = ua[0] + b1 * (ub[0] - ua[0]) + b2 * (uc[0] - ua[0])
            out[i, k, 1] = ua[1] + b1 * (ub[1] - ua[1]) + b2 * (uc[1] - ua[1])
    return out


@nb.njit(cache=True)
def _find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        nx = p[x]; p[x] = r; x = nx
    return r


@nb.njit(cache=True)
def weld_corners(F, FF, faces, loc, lab, cuv, tol):
    """Union corners across manifold edges inside a chart when both shared
    vertices agree on UV within ``tol``. Returns corner root ids (n,3)."""
    n = faces.shape[0]
    p = np.arange(3 * n)
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            j = loc[g]
            if j < 0 or j <= i or lab[j] != lab[i]:
                continue
            a = F[f, k]; b = F[f, (k + 1) % 3]
            ka = -1; kb = -1
            for kk in range(3):
                if F[g, kk] == a:
                    ka = kk
                elif F[g, kk] == b:
                    kb = kk
            if ka < 0 or kb < 0:
                continue
            kb_i = (k + 1) % 3
            da = np.sqrt((cuv[i, k, 0] - cuv[j, ka, 0]) ** 2 + (cuv[i, k, 1] - cuv[j, ka, 1]) ** 2)
            db = np.sqrt((cuv[i, kb_i, 0] - cuv[j, kb, 0]) ** 2 + (cuv[i, kb_i, 1] - cuv[j, kb, 1]) ** 2)
            if da > tol or db > tol:
                continue
            ra = _find(p, 3 * i + k); rb = _find(p, 3 * j + ka)
            if ra != rb:
                p[rb] = ra
            ra = _find(p, 3 * i + kb_i); rb = _find(p, 3 * j + kb)
            if ra != rb:
                p[rb] = ra
    root = np.empty((n, 3), np.int64)
    for i in range(n):
        for k in range(3):
            root[i, k] = _find(p, 3 * i + k)
    return root


@nb.njit(cache=True)
def _near(ta, tb, pgl, aptr, anbr):
    """Proxy triangles within one ring of each other in the cut-open proxy
    chart mesh (seams separate the two sides topologically)."""
    if ta == tb:
        return True
    for i in range(3):
        a = pgl[ta, i]
        for j in range(3):
            if a == pgl[tb, j]:
                return True
    for i in range(3):
        a = pgl[ta, i]
        for q in range(aptr[a], aptr[a + 1]):
            nbv = anbr[q]
            for j in range(3):
                if nbv == pgl[tb, j]:
                    return True
    return False


@nb.njit(cache=True)
def weld_corners_topo(F, FF, faces, loc, lab, pick, pgl, aptr, anbr):
    """Union corners across manifold edges inside a chart when the two faces'
    proxy triangles are topologically adjacent in the proxy chart mesh."""
    n = faces.shape[0]
    p = np.arange(3 * n)
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            j = loc[g]
            if j < 0 or j <= i or lab[j] != lab[i]:
                continue
            if not _near(pick[i], pick[j], pgl, aptr, anbr):
                continue
            a = F[f, k]; b = F[f, (k + 1) % 3]
            ka = -1; kb = -1
            for kk in range(3):
                if F[g, kk] == a:
                    ka = kk
                elif F[g, kk] == b:
                    kb = kk
            if ka < 0 or kb < 0:
                continue
            ra = _find(p, 3 * i + k); rb = _find(p, 3 * j + ka)
            if ra != rb:
                p[rb] = ra
            ra = _find(p, 3 * i + (k + 1) % 3); rb = _find(p, 3 * j + kb)
            if ra != rb:
                p[rb] = ra
    root = np.empty((n, 3), np.int64)
    for i in range(n):
        for k in range(3):
            root[i, k] = _find(p, 3 * i + k)
    return root


def _local_adjacency(pfinal, total):
    I = []; J = []
    off = 0
    for c in pfinal:
        T = c.T + off
        for a, b in ((0, 1), (1, 2), (2, 0)):
            I.append(T[:, a]); J.append(T[:, b]); I.append(T[:, b]); J.append(T[:, a])
        off += len(c.U)
    I = np.concatenate(I); J = np.concatenate(J)
    key = np.unique(I * total + J)
    I, J = key // total, key % total
    ptr = np.zeros(total + 1, np.int64)
    np.add.at(ptr, I + 1, 1)
    np.cumsum(ptr, out=ptr)
    return ptr, J.astype(np.int64)


@nb.njit(cache=True)
def _piece_cuts(F, FF, FE, faces, T, loc_buf):
    n = faces.shape[0]
    for i in range(n):
        loc_buf[faces[i]] = i
    out = np.empty(3 * n, np.int64)
    k_out = 0
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            j = loc_buf[g]
            if j < 0 or j <= i:
                continue
            a = F[f, k]
            for kk in range(3):
                if F[g, kk] == a:
                    if T[j, kk] != T[i, k]:
                        out[k_out] = FE[f, k]; k_out += 1
                    break
    for i in range(n):
        loc_buf[faces[i]] = -1
    return out[:k_out].copy()


def piece_cut_edges(topo, faces, T):
    """Internal edges of a transferred piece along which its UVs are split."""
    buf = -np.ones(len(topo.F), np.int64)
    return _piece_cuts(topo.F, topo.FF, topo.FE, np.asarray(faces, np.int64), np.asarray(T, np.int64), buf)


TINY_PIECE_FACES = 24


def _piece_sizes(root):
    """Face pieces connected through welded corners; returns piece id per face and sizes."""
    from .metrics import uv_islands
    uk, inv = np.unique(root.ravel(), return_inverse=True)
    comp, n = uv_islands(inv.reshape(-1, 3), len(uk))
    return comp, np.bincount(comp, minlength=n)


@nb.njit(cache=True)
def _repick(FF, faces, rows, loc, lab, pick, comp, size, tiny):
    moved = 0
    for q in range(rows.shape[0]):
        i = rows[q]; f = faces[q]
        best = -1
        bs = tiny
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            j = loc[g]
            if j < 0 or lab[j] != lab[i]:
                continue
            if size[comp[j]] > bs:
                bs = size[comp[j]]; best = j
        if best >= 0 and pick[best] != pick[i]:
            pick[i] = pick[best]
            moved += 1
    return moved


def _remap_like(a, b, inverse=False):
    """fix_fragments renumbers labels. Map each new label to the old label it
    mostly covers (or, with inverse, return the array new->old)."""
    nb_ = int(b.max()) + 1
    na_ = int(a.max()) + 1
    key = b * na_ + a
    uk, cnt = np.unique(key, return_counts=True)
    nb_lab = uk // na_; old = uk % na_
    best = np.full(nb_, -1, np.int64); bc = np.zeros(nb_, np.int64)
    for l, o, c in zip(nb_lab.tolist(), old.tolist(), cnt.tolist()):
        if c > bc[l]:
            bc[l] = c; best[l] = o
    if inverse:
        return best
    return best[b]


def transfer_charts(topo, faces_sel, pick, ptopo, pfinal, log=None):
    """Build full-resolution charts from finalized proxy charts.

    Returns list of (faces_global, T_local, U_local, welded_vertex_of_local)."""
    # concatenated proxy UVs and per proxy face the global corner index
    pgl = -np.ones((len(ptopo.F), 3), np.int64)
    pchart = -np.ones(len(ptopo.F), np.int64)
    Us = []
    off = 0
    for ci, c in enumerate(pfinal):
        pgl[c.faces] = c.T + off
        pchart[c.faces] = ci
        Us.append(c.U)
        off += len(c.U)
    Ucat = np.ascontiguousarray(np.concatenate(Us), np.float64)
    import time as _time
    _t = _time.perf_counter(); tim = {}
    ok = pchart[pick] >= 0
    faces_sel = faces_sel[ok]; pick = pick[ok]
    lab = pchart[pick]
    # label specks near chart borders (a face that picked the neighbouring
    # chart's proxy triangle) are given to the surrounding chart, and their
    # proxy triangle is re-picked inside that chart
    from .refine import fix_fragments
    full = -np.ones(len(topo.F), np.int64)
    full[faces_sel] = lab
    hard = np.zeros(len(topo.E), bool)
    full2 = fix_fragments(topo, full, hard)
    tim["fix_fragments"] = _time.perf_counter() - _t
    # fix_fragments compacts labels; map back to proxy chart ids by majority
    lab2 = full2[faces_sel]
    changed = np.nonzero(lab != _remap_like(lab, lab2))[0]
    if len(changed):
        from scipy.spatial import cKDTree
        pc = ptopo.V[ptopo.F].mean(1)
        target = _remap_like(lab, lab2, inverse=True)[lab2[changed]]
        C = topo.V[topo.F[faces_sel[changed]]].mean(1)
        for t in np.unique(target):
            sel = changed[target == t]
            pf = np.nonzero(pchart == t)[0]
            if len(pf) == 0:
                continue
            _, j = cKDTree(pc[pf]).query(C[target == t], k=1)
            pick[sel] = pf[j]
            lab[sel] = t
    tim["repick"] = _time.perf_counter() - _t
    cuv = corner_uvs(topo.V, topo.F, faces_sel, pick, ptopo.V, ptopo.F, pgl, Ucat)
    loc = -np.ones(len(topo.F), np.int64)
    loc[faces_sel] = np.arange(len(faces_sel))
    aptr, anbr = _local_adjacency(pfinal, len(Ucat))
    root = weld_corners_topo(topo.F, topo.FF, faces_sel, loc, lab, pick, pgl, aptr, anbr)
    tim["weld"] = _time.perf_counter() - _t
    # faces just across a proxy seam were mapped through the proxy triangle on
    # the wrong side and end up as tiny isolated pieces: re-pick them from a
    # neighbour in a large piece of the same chart, recompute, re-weld
    for _ in range(2):          # later rounds move only a few dozen faces (crumbs absorb them)
        comp, size = _piece_sizes(root)
        tiny = np.nonzero(size[comp] <= TINY_PIECE_FACES)[0]
        if len(tiny) == 0:
            break
        moved = _repick(topo.FF, faces_sel[tiny], tiny, loc, lab, pick, comp, size, TINY_PIECE_FACES)
        tim.setdefault("repick_moved", []).append(int(moved))
        if moved == 0:
            break
        cuv[tiny] = corner_uvs(topo.V, topo.F, faces_sel[tiny], pick[tiny], ptopo.V, ptopo.F, pgl, Ucat)
        root = weld_corners_topo(topo.F, topo.FF, faces_sel, loc, lab, pick, pgl, aptr, anbr)
    tim["tiny_repick"] = _time.perf_counter() - _t
    # every UV-connected piece becomes its own chart (welded corners never join
    # different proxy charts, so components can be found globally)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(faces_sel)
    uk, inv = np.unique(root.ravel(), return_inverse=True)
    nv = len(uk)
    rows = np.repeat(np.arange(n), 3)
    M = csr_matrix((np.ones(3 * n, np.int8), (rows, n + inv)), shape=(n + nv, n + nv))
    nc, cc = connected_components(M, directed=False)
    fcomp = cc[:n]
    vcomp = cc[n:]
    cnt = np.bincount(inv, minlength=nv).astype(np.float64)
    cf = cuv.reshape(-1, 2)
    Uall = np.stack([np.bincount(inv, cf[:, 0], nv), np.bincount(inv, cf[:, 1], nv)], 1) / cnt[:, None]
    vwall = np.empty(nv, np.int64)
    vwall[inv] = topo.F[faces_sel].ravel()
    # local vertex index = rank of the vertex inside its component
    vo = np.argsort(vcomp, kind="stable")
    vb = np.searchsorted(vcomp[vo], np.arange(nc + 1))
    vloc = np.empty(nv, np.int64)
    vloc[vo] = np.arange(nv) - vb[vcomp[vo]]
    Tl = vloc[inv].reshape(-1, 3)
    fo = np.argsort(fcomp, kind="stable")
    fb = np.searchsorted(fcomp[fo], np.arange(nc + 1))
    out = []
    for k in range(nc):
        ii = fo[fb[k]:fb[k + 1]]
        if len(ii) == 0:
            continue
        vv = vo[vb[k]:vb[k + 1]]
        out.append((faces_sel[ii], Tl[ii], Uall[vv], vwall[vv]))
    tim["grouping"] = _time.perf_counter() - _t
    if log:
        log("transfer timings (cumulative): " + ", ".join(f"{k} {v:.1f}s" if np.isscalar(v) else f"{k} {v}" for k, v in tim.items()))
    return out, faces_sel
