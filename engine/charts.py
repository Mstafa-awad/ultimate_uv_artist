"""Chart construction on the welded topology: chart-local meshes (seam splits),
cut-to-disk seam graphs and curvature-relief cuts."""
from __future__ import annotations

import heapq

import numba as nb
import numpy as np

from . import param
from .metrics import _raster_counts


@nb.njit(cache=True, nogil=True)
def _uf_find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        nx = p[x]; p[x] = r; x = nx
    return r


@nb.njit(cache=True, nogil=True)
def _chart_corners(faces, F, FE, FF, in_chart, cut):
    """Union corners across internal, uncut edges. Returns local vertex id per
    corner (t,3) and the welded vertex of each local vertex."""
    t = faces.shape[0]
    loc = -np.ones(F.shape[0], np.int64)
    for i in range(t):
        loc[faces[i]] = i
    p = np.arange(3 * t)
    for i in range(t):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0 or not in_chart[g] or cut[FE[f, k]]:
                continue
            j = loc[g]
            if j < i:
                continue
            a = F[f, k]; b = F[f, (k + 1) % 3]
            for kk in range(3):
                if F[g, kk] == a:
                    ra = _uf_find(p, 3 * i + k); rb = _uf_find(p, 3 * j + kk)
                    if ra != rb:
                        p[rb] = ra
                elif F[g, kk] == b:
                    ra = _uf_find(p, 3 * i + (k + 1) % 3); rb = _uf_find(p, 3 * j + kk)
                    if ra != rb:
                        p[rb] = ra
    root = np.empty(3 * t, np.int64)
    for c in range(3 * t):
        root[c] = _uf_find(p, c)
    # compact
    remap = -np.ones(3 * t, np.int64)
    nv = 0
    T = np.empty((t, 3), np.int64)
    vw = np.empty(3 * t, np.int64)
    for i in range(t):
        for k in range(3):
            r = root[3 * i + k]
            if remap[r] < 0:
                remap[r] = nv
                vw[nv] = F[faces[i], k]
                nv += 1
            T[i, k] = remap[r]
    return T, vw[:nv].copy()


class Chart:
    """A set of welded faces plus a set of cut (seam) edges inside it."""

    def __init__(self, topo, faces, cut_edges=None):
        self.topo = topo
        self.faces = np.asarray(faces, np.int64)
        self.cut = set() if cut_edges is None else set(int(e) for e in cut_edges)
        self.U = None
        self.quality = None

    def mesh(self, in_chart_buf=None, cut_buf=None):
        """``in_chart_buf`` (caller-owned) must already be True on this chart's faces."""
        topo = self.topo
        if in_chart_buf is None:
            in_chart = np.zeros(len(topo.F), np.bool_)
            in_chart[self.faces] = True
        else:
            in_chart = in_chart_buf
        cut = np.zeros(len(topo.E), np.bool_) if cut_buf is None else cut_buf
        if self.cut:
            cut[np.fromiter(self.cut, np.int64)] = True
        T, vw = _chart_corners(self.faces, topo.F, topo.FE, topo.FF, in_chart, cut)
        if self.cut:
            cut[np.fromiter(self.cut, np.int64)] = False
        return topo.V[vw], T, vw

    def internal_edges(self, in_chart):
        """Global ids of edges with both faces in the chart (manifold link) and not cut."""
        topo = self.topo
        fe = topo.FE[self.faces].ravel()
        ff = topo.FF[self.faces].ravel()
        ok = (ff >= 0)
        ok[ok] = in_chart[ff[ok]]
        e = np.unique(fe[ok])
        if self.cut:
            e = e[~np.isin(e, np.fromiter(self.cut, np.int64))]
        return e

    def boundary_welded_vertices(self, in_chart):
        """Welded vertices on the chart boundary (incl. cuts)."""
        topo = self.topo
        F = topo.F[self.faces]
        ff = topo.FF[self.faces]
        fe = topo.FE[self.faces]
        bnd = ff < 0
        nb_in = np.zeros_like(bnd)
        nb_in[~bnd] = in_chart[ff[~bnd]]
        bnd = bnd | ~nb_in
        if self.cut:
            bnd |= np.isin(fe, np.fromiter(self.cut, np.int64))
        a = F[bnd]
        b = np.roll(F, -1, axis=1)[bnd]
        return np.unique(np.r_[a, b])


# ------------------------------------------------------------------ graph helpers
@nb.njit(cache=True, nogil=True)
def _dijkstra(nv, indptr, nbr, wts, src_mask, dst_mask):
    """Multi-source Dijkstra; stops at first destination. Returns (dst, prev)."""
    INF = 1e300
    dist = np.full(nv, INF)
    prev = -np.ones(nv, np.int64)
    cap = nbr.shape[0] + nv + 16
    heap_d = np.empty(cap)
    heap_v = np.empty(cap, np.int64)
    hn = 0
    for v in range(nv):
        if src_mask[v]:
            dist[v] = 0.0
            # push
            i = hn; hn += 1
            heap_d[i] = 0.0; heap_v[i] = v
    # heapify (all zeros -> already heap)
    while hn > 0:
        d = heap_d[0]; v = heap_v[0]
        hn -= 1
        heap_d[0] = heap_d[hn]; heap_v[0] = heap_v[hn]
        i = 0
        while True:
            l = 2 * i + 1; r = l + 1; s = i
            if l < hn and heap_d[l] < heap_d[s]:
                s = l
            if r < hn and heap_d[r] < heap_d[s]:
                s = r
            if s == i:
                break
            heap_d[i], heap_d[s] = heap_d[s], heap_d[i]
            heap_v[i], heap_v[s] = heap_v[s], heap_v[i]
            i = s
        if d > dist[v]:
            continue
        if dst_mask[v] and not src_mask[v]:
            return v, prev, dist
        for q in range(indptr[v], indptr[v + 1]):
            u = nbr[q]; nd = d + wts[q]
            if nd < dist[u]:
                dist[u] = nd; prev[u] = v
                if hn >= heap_d.shape[0]:
                    continue
                i = hn; hn += 1
                heap_d[i] = nd; heap_v[i] = u
                while i > 0:
                    pp = (i - 1) // 2
                    if heap_d[pp] <= heap_d[i]:
                        break
                    heap_d[i], heap_d[pp] = heap_d[pp], heap_d[i]
                    heap_v[i], heap_v[pp] = heap_v[pp], heap_v[i]
                    i = pp
    return -1, prev, dist


def _csr(nv, u, v, w, eid):
    a = np.r_[u, v]; b = np.r_[v, u]; ww = np.r_[w, w]; ee = np.r_[eid, eid]
    o = np.argsort(a, kind="stable")
    a = a[o]; b = b[o]; ww = ww[o]; ee = ee[o]
    indptr = np.zeros(nv + 1, np.int64)
    np.add.at(indptr, a + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, b, ww, ee


def shortest_seam_path(topo, edges, costs, sources, targets):
    """Cheapest path over welded vertices using only ``edges`` (global ids)."""
    if len(edges) == 0:
        return []
    u = topo.E[edges, 0]; v = topo.E[edges, 1]
    verts, inv = np.unique(np.r_[u, v], return_inverse=True)
    lu = inv[:len(u)]; lv = inv[len(u):]
    nv = len(verts)
    indptr, nbr, wts, ee = _csr(nv, lu, lv, costs[edges], edges)
    pos = {int(x): i for i, x in enumerate(verts.tolist())}
    src = np.zeros(nv, np.bool_); dst = np.zeros(nv, np.bool_)
    for s in sources:
        if int(s) in pos:
            src[pos[int(s)]] = True
    for t in targets:
        if int(t) in pos:
            dst[pos[int(t)]] = True
    if not src.any() or not dst.any():
        return []
    end, prev, _ = _dijkstra(nv, indptr, nbr, wts, src, dst)
    if end < 0:
        return []
    path = []
    x = end
    while prev[x] >= 0:
        p = prev[x]
        # find edge id between p and x
        for q in range(indptr[p], indptr[p + 1]):
            if nbr[q] == x:
                path.append(int(ee[q]))
                break
        x = p
    return path


def farthest_vertex(topo, edges, costs, sources):
    """Vertex farthest (in seam-cost metric is wrong for this; use length) from sources."""
    u = topo.E[edges, 0]; v = topo.E[edges, 1]
    verts, inv = np.unique(np.r_[u, v], return_inverse=True)
    lu = inv[:len(u)]; lv = inv[len(u):]
    indptr, nbr, wts, ee = _csr(len(verts), lu, lv, topo.edge_len[edges], edges)
    src = np.isin(verts, np.asarray(sources))
    dst = np.zeros(len(verts), np.bool_)
    _, _, dist = _dijkstra(len(verts), indptr, nbr, wts, src, dst)
    dist[dist > 1e299] = -1
    return int(verts[int(np.argmax(dist))])


# ------------------------------------------------------------------ cut to disk
@nb.njit(cache=True, nogil=True)
def _topo_counts(T, nv):
    """Euler characteristic and number of boundary loops (components of the
    boundary-edge graph) of a triangle list."""
    m = T.shape[0]
    keys = np.empty(3 * m, np.int64)
    for f in range(m):
        for k in range(3):
            a = T[f, k]; b = T[f, (k + 1) % 3]
            lo = a if a < b else b; hi = b if a < b else a
            keys[3 * f + k] = lo * nv + hi
    keys.sort()
    ne = 0
    p = np.arange(nv)
    nb_v = np.zeros(nv, np.bool_)
    i = 0
    while i < 3 * m:
        j = i
        while j + 1 < 3 * m and keys[j + 1] == keys[i]:
            j += 1
        ne += 1
        if j == i:      # boundary edge
            a = keys[i] // nv; b = keys[i] % nv
            nb_v[a] = True; nb_v[b] = True
            ra = _uf_find(p, a); rb = _uf_find(p, b)
            if ra != rb:
                p[rb] = ra
        i = j + 1
    loops = 0
    for v in range(nv):
        if nb_v[v] and _uf_find(p, v) == v:
            loops += 1
    return nv - ne + m, loops


def euler_and_loops(T):
    nv = int(T.max()) + 1 if len(T) else 0
    if nv == 0:
        return 0, 0
    return _topo_counts(np.ascontiguousarray(T, np.int64), nv)


def is_disk(T):
    chi, nl = euler_and_loops(T)
    return chi == 1 and nl == 1, chi, nl


def cut_to_disk(chart, costs, in_chart):
    """Add cut edges until the chart is a topological disk. Maximum dual spanning
    tree keeps expensive (visible, smooth) edges glued; the pruned complement
    is the cut graph, so seams run along cheap (hidden, creased) edges."""
    topo = chart.topo
    P, T, vw = chart.mesh(in_chart)
    disk, chi, nl = is_disk(T)
    if disk:
        return True
    internal = chart.internal_edges(in_chart)
    if nl == 0:
        # closed surface: open it along the cheapest path between two far points
        a = int(chart.faces[0])
        start = int(topo.F[a, 0])
        far1 = farthest_vertex(topo, internal, costs, [start])
        far2 = farthest_vertex(topo, internal, costs, [far1])
        path = shortest_seam_path(topo, internal, costs, [far1], [far2])
        if not path:
            return False
        chart.cut.update(path)
        P, T, vw = chart.mesh(in_chart)
        disk, chi, nl = is_disk(T)
        if disk:
            return True
        internal = chart.internal_edges(in_chart)
    # dual spanning tree (Kruskal, most expensive edges first)
    f0 = topo.EF[internal, 0]; f1 = topo.EF[internal, 1]
    order = np.argsort(-costs[internal], kind="stable")
    tree = _kruskal(f0[order], f1[order], len(topo.F))
    cand = internal[order][~tree]
    if len(cand) == 0:
        return False
    # prune dangling candidate edges whose free endpoint is not on the boundary
    bverts = chart.boundary_welded_vertices(in_chart)
    full = cand
    before = set(chart.cut)
    cand = _prune(topo.E[cand], cand, bverts)
    chart.cut.update(int(e) for e in cand)
    P, T, vw = chart.mesh(in_chart)
    disk, chi, nl = is_disk(T)
    if not disk:
        # the unpruned complement of a dual spanning tree is always a disk
        chart.cut = before | set(int(e) for e in full)
        P, T, vw = chart.mesh(in_chart)
        disk, chi, nl = is_disk(T)
    return disk


@nb.njit(cache=True, nogil=True)
def _kruskal(a, b, m):
    p = np.arange(m)
    used = np.zeros(a.shape[0], np.bool_)
    for i in range(a.shape[0]):
        ra = _uf_find(p, a[i]); rb = _uf_find(p, b[i])
        if ra != rb:
            p[rb] = ra
            used[i] = True
    return used


@nb.njit(cache=True, nogil=True)
def _prune_nb(E, keep):
    """Peel cut edges hanging from vertices of degree 1 that are not kept."""
    n = keep.shape[0]
    k = E.shape[0]
    deg = np.zeros(n, np.int64)
    for i in range(k):
        deg[E[i, 0]] += 1; deg[E[i, 1]] += 1
    # vertex -> incident edges (CSR)
    ptr = np.zeros(n + 1, np.int64)
    for i in range(k):
        ptr[E[i, 0] + 1] += 1; ptr[E[i, 1] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    fill = ptr[:-1].copy()
    inc = np.empty(2 * k, np.int64)
    for i in range(k):
        inc[fill[E[i, 0]]] = i; fill[E[i, 0]] += 1
        inc[fill[E[i, 1]]] = i; fill[E[i, 1]] += 1
    alive = np.ones(k, np.bool_)
    stack = np.empty(n, np.int64)
    top = 0
    for v in range(n):
        if deg[v] == 1 and not keep[v]:
            stack[top] = v; top += 1
    while top > 0:
        top -= 1
        v = stack[top]
        if deg[v] != 1 or keep[v]:
            continue
        for q in range(ptr[v], ptr[v + 1]):
            e = inc[q]
            if alive[e]:
                alive[e] = False
                u = E[e, 0] if E[e, 1] == v else E[e, 1]
                deg[v] -= 1; deg[u] -= 1
                if deg[u] == 1 and not keep[u]:
                    stack[top] = u; top += 1
                break
    return alive


def _prune(E, eids, keep_vertices):
    if len(eids) == 0:
        return eids
    verts, inv = np.unique(E.ravel(), return_inverse=True)
    El = inv.reshape(-1, 2).astype(np.int64)
    keep = np.isin(verts, np.fromiter(keep_vertices, np.int64) if not isinstance(keep_vertices, np.ndarray) else keep_vertices)
    alive = _prune_nb(El, keep)
    return np.asarray(eids)[alive]


# ------------------------------------------------------------------ flatten
def flatten(P, T, iters=8, method="arap", slim_iters=0):
    """LSCM -> ARAP (flip-safe when the start is flip-free) -> flip repair
    [-> SLIM]. Returns U and quality dict."""
    X, area = param.local_frames(P, T)
    cots = param.cot_weights(X)
    U = param.lscm(P, T, X, cots)
    sa = param.signed_areas(U, T)
    if (sa < 0).sum() > (sa > 0).sum():
        U[:, 1] = -U[:, 1]
        sa = -sa
    folded = bool((sa <= 0).any())
    if iters:
        # folded starts: unconstrained ARAP usually unfolds; flip-free starts keep a
        # flip-preventing line search so they stay locally injective
        U, _ = param.refine(P, T, U, iters=iters * (2 if folded else 1), method="arap", X=X, area=area,
                            cots=cots, line_search=not folded, tol=1e-4)
    if (param.signed_areas(U, T) <= 0).any():
        U = param.repair_flips(U, T)
        if not (param.signed_areas(U, T) <= 0).any() and iters:
            U, _ = param.refine(P, T, U, iters=max(2, iters // 2), method="arap", X=X, area=area, cots=cots,
                                line_search=True, tol=1e-4)
    if slim_iters and not (param.signed_areas(U, T) <= 0).any():
        U2, _ = param.refine(P, T, U, iters=slim_iters, method="slim", X=X, area=area, cots=cots)
        if np.isfinite(U2).all():
            U = U2
    hits = boundary_intersections(U, T)
    if hits and not (param.signed_areas(U, T) <= 0).any():
        # local crossings (boundary vertices whose 3D angle sum exceeds 2*pi wrap
        # their fan over itself): relax only the vertices around the crossing
        U2, ok = untangle_boundary(U, T)
        if ok:
            U = U2
            hits = 0
    q = param.chart_quality(P, T, U, X, area)
    # exact injectivity: no folds + non-self-intersecting boundary
    q["boundary_hits"] = hits
    q["overlap"] = 1.0 if q["boundary_hits"] else 0.0
    return U, q


@nb.njit(cache=True, nogil=True)
def _ring_region(seeds, ptr, J, n, rings):
    mark = np.zeros(n, np.bool_)
    cur = list(seeds)
    for v in seeds:
        mark[v] = True
    out = list(seeds)
    for _ in range(rings):
        nxt = []
        for v in cur:
            for q in range(ptr[v], ptr[v + 1]):
                w = J[q]
                if not mark[w]:
                    mark[w] = True
                    nxt.append(w); out.append(w)
        cur = nxt
    res = np.empty(len(out), np.int64)
    for k in range(len(out)):
        res[k] = out[k]
    return res


@nb.njit(cache=True, nogil=True)
def _region_tris(reg, vt_ptr, vt, nt):
    mark = np.zeros(nt, np.bool_)
    cnt = 0
    for v in reg:
        for q in range(vt_ptr[v], vt_ptr[v + 1]):
            t = vt[q]
            if not mark[t]:
                mark[t] = True
                cnt += 1
    res = np.empty(cnt, np.int64)
    k = 0
    for t in range(nt):
        if mark[t]:
            res[k] = t; k += 1
    return res


@nb.njit(cache=True, nogil=True)
def _smooth_region(U, reg, ptr, J, step):
    Un = U.copy()
    for v in reg:
        sx = 0.0; sy = 0.0; c = 0
        for q in range(ptr[v], ptr[v + 1]):
            sx += U[J[q], 0]; sy += U[J[q], 1]; c += 1
        if c:
            Un[v, 0] = (1 - step) * U[v, 0] + step * sx / c
            Un[v, 1] = (1 - step) * U[v, 1] + step * sy / c
    return Un


def untangle_boundary(U, T, rounds=40):
    """Remove boundary self-crossings by locally smoothing the vertices around
    each crossing toward their UV 1-ring centroid, never folding a triangle.
    Returns (U, success)."""
    from .overlapfix import _all_hits
    U = np.array(U, np.float64)
    bi, bj = param.boundary_edges_oriented(T)
    if len(bi) < 3:
        return U, True
    n = len(U)
    I = np.r_[T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]]
    J = np.r_[T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]]
    key = np.unique(I * n + J)
    I, J = key // n, key % n
    ptr = np.searchsorted(I, np.arange(n + 1))
    vt_ptr = np.zeros(n + 1, np.int64)
    np.add.at(vt_ptr, T.ravel() + 1, 1)
    np.cumsum(vt_ptr, out=vt_ptr)
    vt = (np.argsort(T.ravel(), kind="stable") // 3).astype(np.int64)
    J = J.astype(np.int64); ptr = ptr.astype(np.int64)
    for r in range(rounds):
        A = np.ascontiguousarray(U[bi]); B = np.ascontiguousarray(U[bj])
        lo = U.min(0); ext = U.max(0) - lo
        L = np.linalg.norm(B - A, axis=1)
        cell = max(float(np.median(L)) * 4, float(ext.max()) / 512, 1e-30)
        nx = int(ext[0] / cell) + 1; ny = int(ext[1] / cell) + 1
        hits = _all_hits(A, B, cell, lo, nx, ny, 512)
        if len(hits) == 0:
            return U, True
        moved = False
        # all crossings of this round at once; if that cannot be done without a
        # fold, only the first one
        for sel in (hits, hits[:1]):
            seeds = np.unique(np.r_[bi[sel[:, 0]], bj[sel[:, 0]], bi[sel[:, 1]], bj[sel[:, 1]]]).astype(np.int64)
            reg = _ring_region(seeds, ptr, J, n, 1 + r // 8)
            tris = _region_tris(reg, vt_ptr, vt, len(T))
            Tt = T[tris]
            sa0 = param.signed_areas(U, Tt)
            was_ok = sa0 > 0
            for step in (0.5, 0.25, 0.1):
                Un = _smooth_region(U, reg, ptr, J, step)
                # never create a fold, never squash a triangle below 20% of its area
                if ((param.signed_areas(Un, Tt) > 0.2 * sa0) | ~was_ok).all():
                    U = Un
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return U, False
    return U, False


@nb.njit(cache=True, nogil=True)
def _seg_intersections(A, B, cell, lo, nx, ny, max_report):
    """Count proper intersections between non-adjacent segments A[i]->B[i]
    using a uniform grid. Segments sharing an endpoint index are skipped by
    the caller encoding (adjacent segments share coordinates exactly)."""
    n = A.shape[0]
    # bucket counts
    cnt = np.zeros(nx * ny + 1, np.int64)
    for i in range(n):
        x0 = int((min(A[i, 0], B[i, 0]) - lo[0]) / cell); x1 = int((max(A[i, 0], B[i, 0]) - lo[0]) / cell)
        y0 = int((min(A[i, 1], B[i, 1]) - lo[1]) / cell); y1 = int((max(A[i, 1], B[i, 1]) - lo[1]) / cell)
        for gx in range(max(x0, 0), min(x1, nx - 1) + 1):
            for gy in range(max(y0, 0), min(y1, ny - 1) + 1):
                cnt[gy * nx + gx + 1] += 1
    for k in range(1, cnt.shape[0]):
        cnt[k] += cnt[k - 1]
    items = np.empty(cnt[-1], np.int64)
    fill = cnt[:-1].copy()
    for i in range(n):
        x0 = int((min(A[i, 0], B[i, 0]) - lo[0]) / cell); x1 = int((max(A[i, 0], B[i, 0]) - lo[0]) / cell)
        y0 = int((min(A[i, 1], B[i, 1]) - lo[1]) / cell); y1 = int((max(A[i, 1], B[i, 1]) - lo[1]) / cell)
        for gx in range(max(x0, 0), min(x1, nx - 1) + 1):
            for gy in range(max(y0, 0), min(y1, ny - 1) + 1):
                c = gy * nx + gx
                items[fill[c]] = i; fill[c] += 1
    hits = 0
    for c in range(nx * ny):
        for p in range(cnt[c], cnt[c + 1]):
            i = items[p]
            for q in range(p + 1, cnt[c + 1]):
                j = items[q]
                # skip segments touching at an endpoint
                if (A[i, 0] == A[j, 0] and A[i, 1] == A[j, 1]) or (A[i, 0] == B[j, 0] and A[i, 1] == B[j, 1]) or                    (B[i, 0] == A[j, 0] and B[i, 1] == A[j, 1]) or (B[i, 0] == B[j, 0] and B[i, 1] == B[j, 1]):
                    continue
                d1x = B[i, 0] - A[i, 0]; d1y = B[i, 1] - A[i, 1]
                d2x = B[j, 0] - A[j, 0]; d2y = B[j, 1] - A[j, 1]
                o1 = d1x * (A[j, 1] - A[i, 1]) - d1y * (A[j, 0] - A[i, 0])
                o2 = d1x * (B[j, 1] - A[i, 1]) - d1y * (B[j, 0] - A[i, 0])
                o3 = d2x * (A[i, 1] - A[j, 1]) - d2y * (A[i, 0] - A[j, 0])
                o4 = d2x * (B[i, 1] - A[j, 1]) - d2y * (B[i, 0] - A[j, 0])
                tol1 = 1e-9 * (d1x * d1x + d1y * d1y); tol2 = 1e-9 * (d2x * d2x + d2y * d2y)
                if ((o1 > tol1 and o2 < -tol1) or (o1 < -tol1 and o2 > tol1)) and \
                   ((o3 > tol2 and o4 < -tol2) or (o3 < -tol2 and o4 > tol2)):
                    hits += 1
                    if hits >= max_report:
                        return hits
    return hits


def boundary_intersections(U, T, max_report=1000):
    """Exact count of self-intersections of the chart's UV boundary. With no
    folded triangles, zero intersections means the disk chart is injective."""
    bi, bj = param.boundary_edges_oriented(T)
    if len(bi) < 3:
        return 0
    A = np.ascontiguousarray(U[bi]); B = np.ascontiguousarray(U[bj])
    lo = U.min(0)
    ext = U.max(0) - lo
    L = np.linalg.norm(B - A, axis=1)
    cell = max(float(np.median(L)) * 4, float(ext.max()) / 512, 1e-30)
    nx = int(ext[0] / cell) + 1; ny = int(ext[1] / cell) + 1
    return int(_seg_intersections(A, B, cell, lo, nx, ny, max_report))


def overlap_fraction(U, T, res=512):
    lo = U.min(0); ext = (U.max(0) - lo).max()
    if not np.isfinite(ext) or ext <= 0:
        return 1.0
    uv = (U - lo) / ext * 0.999
    counts = np.zeros((res, res), np.uint8)
    _raster_counts(np.ascontiguousarray(uv), np.ascontiguousarray(T, np.int64), res, counts)
    cov = np.count_nonzero(counts)
    return float(np.count_nonzero(counts > 1)) / max(cov, 1)
