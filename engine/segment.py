"""Part-aware, distortion-bounded charting.

1. Atoms: compact face regions grown by Dijkstra that never cross hard edges.
2. Dendrogram: agglomerative merging of atoms. Merging is cheap across smooth,
   visible surface and expensive across creases, concave valleys and hidden
   areas (the natural seam locations) and where the union stops being
   developable (normal-cone covariance). The last merges are therefore the
   most meaningful part boundaries.
3. Top-down selection: starting from each root, a node is accepted as one UV
   island if it can be cut to a disk and flattened within the distortion
   budget (optionally after Seamster-style relief cuts through the worst
   region along hidden paths); otherwise it is split into its two children.
"""
from __future__ import annotations

import heapq
import time

import numba as nb
import numpy as np

from . import charts as ch


# ------------------------------------------------------------------ atoms
@nb.njit(cache=True, nogil=True)
def _grow(FF, FE, C, N, hard, labels, seeds, seed_labels, normal_w):
    m = FF.shape[0]
    INF = 1e300
    dist = np.full(m, INF)
    cap = 4 * m + 16
    hd = np.empty(cap); hv = np.empty(cap, np.int64)
    hn = 0
    for i in range(seeds.shape[0]):
        s = seeds[i]
        if labels[s] >= 0:
            continue
        labels[s] = seed_labels[i]
        dist[s] = 0.0
        hd[hn] = 0.0; hv[hn] = s; hn += 1
    while hn > 0:
        d = hd[0]; f = hv[0]
        hn -= 1; hd[0] = hd[hn]; hv[0] = hv[hn]
        i = 0
        while True:
            l = 2 * i + 1; r = l + 1; s = i
            if l < hn and hd[l] < hd[s]:
                s = l
            if r < hn and hd[r] < hd[s]:
                s = r
            if s == i:
                break
            hd[i], hd[s] = hd[s], hd[i]; hv[i], hv[s] = hv[s], hv[i]
            i = s
        if d > dist[f]:
            continue
        for k in range(3):
            g = FF[f, k]
            if g < 0 or hard[FE[f, k]]:
                continue
            if labels[g] < -1:
                continue          # outside the region being grown
            if labels[g] >= 0 and dist[g] == INF:
                continue          # frozen by an earlier growth pass
            dc = np.sqrt(((C[f] - C[g]) ** 2).sum())
            nd = d + dc * (1.0 + normal_w * (1.0 - (N[f] * N[g]).sum()))
            if nd < dist[g]:
                dist[g] = nd
                labels[g] = labels[f]
                if hn < cap:
                    hd[hn] = nd; hv[hn] = g
                    i = hn; hn += 1
                    while i > 0:
                        pp = (i - 1) // 2
                        if hd[pp] <= hd[i]:
                            break
                        hd[i], hd[pp] = hd[pp], hd[i]; hv[i], hv[pp] = hv[pp], hv[i]
                        i = pp
    return labels


def make_atoms(topo, hard, n_atoms=1500, normal_w=4.0, N=None):
    m = len(topo.F)
    C = topo.V[topo.F].mean(1)
    N = np.ascontiguousarray(topo.face_normal if N is None else N)
    valid = ~topo.degenerate
    n_atoms = max(1, min(n_atoms, int(valid.sum())))
    lo = C[valid].min(0); ext = np.maximum(C[valid].max(0) - lo, 1e-12)
    # choose a grid so that ~n_atoms cells are occupied (surface ~ 2D)
    labels = -np.ones(m, np.int64)
    labels[~valid] = -2
    h = np.sqrt(topo.face_area[valid].sum() / n_atoms)
    for _ in range(6):
        cell = np.floor((C[valid] - lo) / h).astype(np.int64)
        key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
        nocc = len(np.unique(key))
        if 0.7 * n_atoms <= nocc <= 1.4 * n_atoms:
            break
        h *= (nocc / n_atoms) ** 0.5
    vidx = np.nonzero(valid)[0]
    cell = np.floor((C[vidx] - lo) / h).astype(np.int64)
    key = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    # seed = face closest to its cell centre
    centre = (cell + 0.5) * h + lo
    dd = ((C[vidx] - centre) ** 2).sum(1)
    o = np.lexsort((dd, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    seeds = vidx[o][first]
    _grow(topo.FF, topo.FE, C, N, hard, labels, seeds, np.arange(len(seeds)), normal_w)
    nl = len(seeds)
    while True:
        rest = np.nonzero(labels == -1)[0]
        if len(rest) == 0:
            break
        # seed every unreached region (hard-edge pockets, tiny components)
        s = rest[:max(1, min(len(rest), 64))]
        _grow(topo.FF, topo.FE, C, N, hard, labels, s, np.arange(nl, nl + len(s)), normal_w)
        nl += len(s)
    # compact labels; degenerate faces -> -1
    ok = labels >= 0
    _, inv = np.unique(labels[ok], return_inverse=True)
    out = -np.ones(m, np.int64)
    out[ok] = inv
    return out


# ------------------------------------------------------------------ atom graph
class Cluster:
    __slots__ = ("id", "area", "N", "children", "cost", "atoms", "alive", "nbr", "perim")

    def __init__(self, cid, area, N, atoms, perim):
        self.id = cid; self.area = area; self.N = N; self.children = None
        self.cost = 0.0; self.atoms = atoms; self.alive = True; self.nbr = {}; self.perim = perim


def _dev(N, area):
    if area <= 0:
        return 0.0
    w = np.linalg.eigvalsh(N / area)
    return float(max(w[0], 0.0))


def build_dendrogram(topo, atoms, ef, seam_cost, beta=1.0, gamma=0.15, N=None):
    """Agglomerative clustering of atoms. Returns list of clusters (index = id)
    and root ids."""
    na = int(atoms.max()) + 1
    A = topo.face_area
    n = topo.face_normal if N is None else N
    ok = atoms >= 0
    area = np.bincount(atoms[ok], weights=A[ok], minlength=na)
    NN = np.zeros((na, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            v = np.bincount(atoms[ok], weights=(A * n[:, i] * n[:, j])[ok], minlength=na)
            NN[:, i, j] = v; NN[:, j, i] = v
    # adjacency across manifold links
    f = np.repeat(np.arange(len(topo.F)), 3)
    g = topo.FF.ravel(); e = topo.FE.ravel()
    sel = (g >= 0) & ok[f]
    sel[sel] &= ok[g[sel]]
    la = atoms[f[sel]]; lb = atoms[g[sel]]; ee = e[sel]
    diff = la != lb
    la, lb, ee = la[diff], lb[diff], ee[diff]
    lo = np.minimum(la, lb); hi = np.maximum(la, lb)
    key = lo * na + hi
    # each shared edge appears twice (from both faces)
    uk, first, inv = np.unique(key * 0 + key, return_index=True, return_inverse=True)
    L = np.bincount(inv, weights=topo.edge_len[ee], minlength=len(uk)) * 0.5
    good = 1.0 - np.clip(seam_cost[ee] / np.maximum(topo.edge_len[ee], 1e-30) - 0.06, 0, 1) ** 0.5
    Lg = np.bincount(inv, weights=topo.edge_len[ee] * good, minlength=len(uk)) * 0.5
    H = np.bincount(inv, weights=ef["hard"][ee].astype(float), minlength=len(uk)) > 0
    # boundary perimeter of atoms (all edges whose other side is a different atom or nothing)
    per = np.zeros(na)
    bnd_e = np.ones(len(topo.F) * 3, bool)
    bnd_e[np.nonzero(sel)[0][~diff]] = False
    fa = atoms[f]; okf = ok[f]
    np.add.at(per, fa[okf & bnd_e], topo.edge_len[e[okf & bnd_e]])
    clusters = [Cluster(i, float(area[i]), NN[i], [i], float(per[i])) for i in range(na)]
    a_ = (uk // na).astype(np.int64); b_ = (uk % na).astype(np.int64)
    for i in range(len(uk)):
        x, y = int(a_[i]), int(b_[i])
        rec = [float(L[i]), float(Lg[i]), bool(H[i])]
        clusters[x].nbr[y] = rec; clusters[y].nbr[x] = list(rec)
    ell = np.sqrt(max(area.sum(), 1e-30) / max(na, 1))
    heap = []

    def cost(ca, cb, rec):
        if rec[2]:
            return np.inf
        Aab = ca.area + cb.area
        d = Aab * _dev(ca.N + cb.N, Aab) - ca.area * _dev(ca.N, ca.area) - cb.area * _dev(cb.N, cb.area)
        # removing a good seam location costs; compactness of the union is rewarded
        pab = ca.perim + cb.perim - 2 * rec[0]
        comp = (pab ** 2 / max(Aab, 1e-30)) - (ca.perim ** 2 / max(ca.area, 1e-30)) - (cb.perim ** 2 / max(cb.area, 1e-30))
        return d + beta * rec[1] * ell + gamma * comp * ell * ell * 0.05

    for x in range(na):
        for y, rec in clusters[x].nbr.items():
            if x < y:
                c = cost(clusters[x], clusters[y], rec)
                if np.isfinite(c):
                    heapq.heappush(heap, (c, x, y))
    while heap:
        c, x, y = heapq.heappop(heap)
        cx, cy = clusters[x], clusters[y]
        if not (cx.alive and cy.alive) or y not in cx.nbr:
            continue
        rec = cx.nbr[y]
        nid = len(clusters)
        nc = Cluster(nid, cx.area + cy.area, cx.N + cy.N, None, cx.perim + cy.perim - 2 * rec[0])
        nc.children = (x, y); nc.cost = c
        cx.alive = cy.alive = False
        for src in (cx, cy):
            for z, r in src.nbr.items():
                if z in (x, y):
                    continue
                cur = nc.nbr.get(z)
                if cur is None:
                    nc.nbr[z] = list(r)
                else:
                    cur[0] += r[0]; cur[1] += r[1]; cur[2] = cur[2] or r[2]
                cz = clusters[z]
                cz.nbr.pop(src.id, None)
        for z, r in nc.nbr.items():
            clusters[z].nbr[nid] = list(r)
        clusters.append(nc)
        cx.nbr = {}; cy.nbr = {}
        for z, r in nc.nbr.items():
            cc = cost(nc, clusters[z], r)
            if np.isfinite(cc):
                heapq.heappush(heap, (cc, nid, z) if nid < z else (cc, z, nid))
    roots = [c.id for c in clusters if c.alive]
    return clusters, roots


def cluster_atoms(clusters, cid):
    out = []
    stack = [cid]
    while stack:
        c = clusters[stack.pop()]
        if c.children is None:
            out.extend(c.atoms)
        else:
            stack.extend(c.children)
    return out


# ------------------------------------------------------------------ top-down
def charts_topdown(topo, atoms, clusters, roots, costs, cfg, log=None):
    """Return list of accepted Chart objects (with flattened U)."""
    t0 = time.perf_counter()
    order = np.argsort(atoms, kind="stable")
    sa = atoms[order]
    valid = sa >= 0
    order = order[valid]; sa = sa[valid]
    na = int(atoms.max()) + 1
    starts = np.searchsorted(sa, np.arange(na + 1))
    faces_of_atom = lambda a: order[starts[a]:starts[a + 1]]
    in_chart = np.zeros(len(topo.F), np.bool_)
    accepted = []
    stack = list(roots)
    stats = {"flattened": 0, "relief_cuts": 0, "splits": 0, "over_budget": 0}
    max_a = cfg["max_aniso"]; max_ar = cfg["max_area"]
    while stack:
        cid = stack.pop()
        node = clusters[cid]
        faces = np.concatenate([faces_of_atom(a) for a in cluster_atoms(clusters, cid)])
        chart = ch.Chart(topo, faces)
        in_chart[faces] = True
        ok_disk = ch.cut_to_disk(chart, costs, in_chart)
        best = None
        if ok_disk:
            for attempt in range(cfg["relief_cuts"] + 1):
                P, T, vw = chart.mesh(in_chart)
                try:
                    U, q = ch.flatten(P, T, iters=cfg["plan_iters"])
                except Exception:
                    U, q = None, None
                stats["flattened"] += 1
                if q is None:
                    break
                good = (q["flips"] == 0 and q["overlap"] <= cfg["max_overlap"] and
                        q["aniso_p95"] <= max_a and q["area_p95"] <= max_ar)
                score = max(q["aniso_p95"] / max_a, q["area_p95"] / max_ar) + 10 * (q["flips"] > 0) + 10 * q["overlap"]
                if best is None or score < best[0]:
                    best = (score, set(chart.cut), U, q, good)
                if good:
                    break
                if attempt == cfg["relief_cuts"] or node.children is None and attempt >= 1:
                    break
                # relief cut from the worst-distorted interior vertex to the boundary
                path = _relief_path(topo, chart, in_chart, P, T, vw, q, costs)
                if not path:
                    break
                # do not accept relief seams that are long relative to the chart
                plen = topo.edge_len[path].sum()
                if plen > cfg["max_relief_len"] * np.sqrt(topo.face_area[faces].sum()):
                    break
                chart.cut.update(path)
                stats["relief_cuts"] += 1
        in_chart[faces] = False
        if best is not None and best[4]:
            chart.cut = best[1]; chart.U = best[2]; chart.quality = best[3]
            accepted.append(chart)
            continue
        if node.children is not None:
            stack.extend(node.children)
            stats["splits"] += 1
            continue
        # leaf atom that still fails: keep the best attempt (or split it geometrically)
        if best is not None and best[3]["flips"] == 0 and best[3]["overlap"] <= 0.02:
            chart.cut = best[1]; chart.U = best[2]; chart.quality = best[3]
            stats["over_budget"] += 1
            accepted.append(chart)
        else:
            for sub in _bisect_faces(topo, faces):
                c2 = ch.Chart(topo, sub)
                in_chart[sub] = True
                ch.cut_to_disk(c2, costs, in_chart)
                P, T, vw = c2.mesh(in_chart)
                try:
                    c2.U, c2.quality = ch.flatten(P, T, iters=cfg["plan_iters"])
                except Exception:
                    c2.U, c2.quality = None, None
                in_chart[sub] = False
                accepted.append(c2)
            stats["over_budget"] += 1
    stats["seconds"] = time.perf_counter() - t0
    if log:
        log(f"top-down: {len(accepted)} charts {stats}")
    return accepted, stats


def _relief_path(topo, chart, in_chart, P, T, vw, q, costs):
    """Seam from the most distorted interior vertex to the chart boundary."""
    per_tri = np.abs(np.log2(np.maximum(q["per_tri_area"], 1e-12))) + 0.5 * np.log2(np.maximum(q["per_tri_aniso"], 1.0))
    nv = len(P)
    vs = np.zeros(nv); vc = np.zeros(nv)
    for k in range(3):
        np.add.at(vs, T[:, k], per_tri); np.add.at(vc, T[:, k], 1)
    vs /= np.maximum(vc, 1)
    bverts = chart.boundary_welded_vertices(in_chart)
    bmask = np.isin(vw, bverts)
    vs[bmask] = -1
    if vs.max() <= 0:
        return []
    src = int(vw[int(np.argmax(vs))])
    internal = chart.internal_edges(in_chart)
    return ch.shortest_seam_path(topo, internal, costs, [src], bverts)


def split_two(topo, faces, N=None):
    """Seam-aware 2-way split of a connected face set: two regions grown by
    Dijkstra from a far-apart seed pair meet where the surface bends (creases),
    so the new seam follows natural boundaries. Falls back to _bisect_faces."""
    faces = np.asarray(faces, np.int64)
    if len(faces) < 4:
        return _bisect_faces(topo, faces)
    m = len(topo.F)
    C = topo.V[topo.F[faces]].mean(1)
    c0 = C.mean(0)
    a = int(np.argmax(((C - c0) ** 2).sum(1)))
    b = int(np.argmax(((C - C[a]) ** 2).sum(1)))
    labels = np.full(m, -3, np.int64)
    labels[faces] = -1
    hard = np.zeros(len(topo.E), np.bool_)
    Nn = np.ascontiguousarray(topo.face_normal if N is None else N)
    Cf = getattr(topo, "_centroids", None)
    if Cf is None:
        Cf = topo.V[topo.F].mean(1)
        try:
            topo._centroids = Cf
        except Exception:
            pass
    _grow(topo.FF, topo.FE, Cf, Nn, hard, labels, np.array([faces[a], faces[b]], np.int64),
          np.array([0, 1], np.int64), 8.0)
    lab = labels[faces]
    parts = []
    for l in (0, 1):
        part = faces[lab == l]
        if len(part):
            mask = np.zeros(m, bool); mask[part] = True
            cl = _sub_components(topo.FF, mask, part)
            for k in np.unique(cl):
                parts.append(part[cl == k])
    rest = faces[lab < 0]
    if len(rest):
        mask = np.zeros(m, bool); mask[rest] = True
        cl = _sub_components(topo.FF, mask, rest)
        for k in np.unique(cl):
            parts.append(rest[cl == k])
    if len(parts) < 2:
        return _bisect_faces(topo, faces)
    return parts


def _bisect_faces(topo, faces):
    """Split a face set in two along its principal axis (connected pieces)."""
    C = topo.V[topo.F[faces]].mean(1)
    c = C.mean(0)
    _, _, vt = np.linalg.svd(C - c, full_matrices=False)
    s = (C - c) @ vt[0]
    a = faces[s <= np.median(s)]; b = faces[s > np.median(s)]
    out = []
    for part in (a, b):
        if len(part) == 0:
            continue
        mask = np.zeros(len(topo.F), bool); mask[part] = True
        lab = _sub_components(topo.FF, mask, part)
        for l in np.unique(lab):
            out.append(part[lab == l])
    return out


def _sub_components(FF, mask, part):
    """Connected components (edge adjacency) of the faces ``part``."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    part = np.asarray(part, np.int64)
    loc = -np.ones(len(FF), np.int64); loc[part] = np.arange(len(part))
    g = FF[part]
    r = np.repeat(np.arange(len(part)), 3); c = g.ravel()
    ok = c >= 0
    r, c = r[ok], c[ok]
    ok = mask[c]
    r, c = r[ok], loc[c[ok]]
    M = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(len(part), len(part)))
    return connected_components(M, directed=False)[1].astype(np.int64)


def _split_cost(topo, faces, groups, N):
    """Seam cost of splitting ``faces`` into ``groups`` (list of face arrays):
    boundary length weighted by (1 - crease), so cuts along creases are cheap,
    normalised by sqrt(area); slivers (< 4 % of the area) are penalised."""
    lab = -np.ones(len(topo.F), np.int64)
    for k, g in enumerate(groups):
        lab[g] = k
    f = np.repeat(faces, 3); g = topo.FF[faces].ravel(); e = topo.FE[faces].ravel()
    ok = (g >= 0)
    ok[ok] &= (lab[g[ok]] >= 0) & (lab[g[ok]] != lab[f[ok]])
    f, g, e = f[ok], g[ok], e[ok]
    cosang = np.clip((N[f] * N[g]).sum(1), -1, 1)
    crease = np.clip((np.arccos(cosang) - np.radians(12)) / np.radians(48), 0, 1)
    cost = 0.5 * float((topo.edge_len[e] * (1 - crease)).sum())     # each edge seen from both sides
    area = topo.face_area[faces].sum()
    fr = min(topo.face_area[gg].sum() for gg in groups) / max(area, 1e-30)
    return cost / np.sqrt(max(area, 1e-30)) + (10.0 if fr < 0.04 else 0.0)


def split_normals(topo, faces, N):
    """Two-way split by face-normal direction (area-weighted 2-means), split
    into connected pieces. Follows creases on boxy shapes."""
    faces = np.asarray(faces, np.int64)
    n = N[faces]; w = topo.face_area[faces]
    c = (n * w[:, None]).sum(0)
    # seeds: the two normals farthest apart along the main spread direction
    d = n - c / max(np.linalg.norm(c), 1e-30)
    _, _, vt = np.linalg.svd(d * np.sqrt(w)[:, None], full_matrices=False)
    s = n @ vt[0]
    ca, cb = n[np.argmax(s)], n[np.argmin(s)]
    lab = s > 0
    for _ in range(8):
        lab = (n @ ca) >= (n @ cb)
        if lab.all() or not lab.any():
            return [faces]
        ca = (n[lab] * w[lab, None]).sum(0); ca /= max(np.linalg.norm(ca), 1e-30)
        cb = (n[~lab] * w[~lab, None]).sum(0); cb /= max(np.linalg.norm(cb), 1e-30)
    out = []
    m = len(topo.F)
    for part in (faces[lab], faces[~lab]):
        mask = np.zeros(m, bool); mask[part] = True
        cl = _sub_components(topo.FF, mask, part)
        for k in np.unique(cl):
            out.append(part[cl == k])
    return out


def split_best(topo, faces, N=None):
    """Best of several two-way splits by seam cost (see _split_cost)."""
    Nn = np.ascontiguousarray(topo.face_normal if N is None else N)
    cands = [split_two(topo, faces, Nn)]
    try:
        sn = split_normals(topo, faces, Nn)
        if len(sn) >= 2:
            cands.append(sn)
    except Exception:
        pass
    if len(cands) == 1:
        return cands[0]
    costs = [_split_cost(topo, np.asarray(faces, np.int64), c, Nn) for c in cands]
    return cands[int(np.argmin(costs))]
