"""Remove UV self-overlaps by targeted splitting (no re-solve).

A flattened chart whose boundary crosses itself is split by a seam-aware
two-region growth seeded at the two colliding boundary triangles, so the
colliding parts end up in different islands and keep their (already good)
UVs. Folded triangles that survive repair are detached the same way.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .segment import _grow


@nb.njit(cache=True, nogil=True)
def _boundary_with_faces(T):
    """Directed boundary half-edges and the triangle owning each."""
    t = T.shape[0]
    n = 0
    for f in range(t):
        for k in range(3):
            if T[f, k] > n:
                n = T[f, k]
    n += 1
    keys = np.empty(3 * t, np.int64)
    for f in range(t):
        for k in range(3):
            keys[3 * f + k] = T[f, k] * n + T[f, (k + 1) % 3]
    srt = np.sort(keys)
    a = np.empty(3 * t, np.int64); b = np.empty(3 * t, np.int64); fo = np.empty(3 * t, np.int64)
    m = 0
    for f in range(t):
        for k in range(3):
            i = T[f, k]; j = T[f, (k + 1) % 3]
            rev = j * n + i
            pos = np.searchsorted(srt, rev)
            if pos < srt.shape[0] and srt[pos] == rev:
                continue
            a[m] = i; b[m] = j; fo[m] = f; m += 1
    return a[:m], b[:m], fo[:m]


@nb.njit(cache=True, nogil=True)
def _first_hit(A, B, cell, lo, nx, ny):
    """First pair of non-adjacent intersecting segments (grid hashed), or (-1,-1)."""
    n = A.shape[0]
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
    for c in range(nx * ny):
        for p in range(cnt[c], cnt[c + 1]):
            i = items[p]
            for q in range(p + 1, cnt[c + 1]):
                j = items[q]
                if (A[i, 0] == A[j, 0] and A[i, 1] == A[j, 1]) or (A[i, 0] == B[j, 0] and A[i, 1] == B[j, 1]) or \
                   (B[i, 0] == A[j, 0] and B[i, 1] == A[j, 1]) or (B[i, 0] == B[j, 0] and B[i, 1] == B[j, 1]):
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
                    return i, j
    return -1, -1


@nb.njit(cache=True, nogil=True)
def _all_hits(A, B, cell, lo, nx, ny, maxk):
    """Up to ``maxk`` pairs of non-adjacent intersecting segments (grid hashed)."""
    n = A.shape[0]
    out = np.empty((maxk, 2), np.int64)
    k_ = 0
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
    for c in range(nx * ny):
        for p in range(cnt[c], cnt[c + 1]):
            i = items[p]
            for q in range(p + 1, cnt[c + 1]):
                j = items[q]
                if (A[i, 0] == A[j, 0] and A[i, 1] == A[j, 1]) or (A[i, 0] == B[j, 0] and A[i, 1] == B[j, 1]) or \
                   (B[i, 0] == A[j, 0] and B[i, 1] == A[j, 1]) or (B[i, 0] == B[j, 0] and B[i, 1] == B[j, 1]):
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
                    out[k_, 0] = i; out[k_, 1] = j
                    k_ += 1
                    if k_ >= maxk:
                        return out[:k_]
    return out[:k_]


@nb.njit(cache=True, nogil=True)
def _raster_pair(uv, T, res):
    """First pair of triangles covering the same texel centre (top-left rule),
    or (-1,-1). uv normalised to [0,1)."""
    owner = -np.ones((res, res), np.int32)
    for t in range(T.shape[0]):
        a, b, c = T[t, 0], T[t, 1], T[t, 2]
        x0, y0 = uv[a, 0] * res, uv[a, 1] * res
        x1, y1 = uv[b, 0] * res, uv[b, 1] * res
        x2, y2 = uv[c, 0] * res, uv[c, 1] * res
        ar = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if ar == 0.0:
            continue
        if ar < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
        minx = max(int(np.floor(min(x0, min(x1, x2)) - 0.5)), 0)
        maxx = min(int(np.ceil(max(x0, max(x1, x2)) - 0.5)), res - 1)
        miny = max(int(np.floor(min(y0, min(y1, y2)) - 0.5)), 0)
        maxy = min(int(np.ceil(max(y0, max(y1, y2)) - 0.5)), res - 1)
        ex = (x1 - x0, x2 - x1, x0 - x2); ey = (y1 - y0, y2 - y1, y0 - y2)
        ox = (x0, x1, x2); oy = (y0, y1, y2)
        for py in range(miny, maxy + 1):
            cy = py + 0.5
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                inside = True
                for k in range(3):
                    e = ex[k] * (cy - oy[k]) - ey[k] * (cx - ox[k])
                    if e < 0 or (e == 0 and not ((ey[k] == 0 and ex[k] < 0) or ey[k] > 0)):
                        inside = False
                        break
                if inside:
                    if owner[py, px] >= 0:
                        return owner[py, px], t
                    owner[py, px] = t
    return -1, -1


@nb.njit(cache=True, nogil=True)
def _raster_all(uv, T, res, bad):
    """Mark every triangle that shares a texel centre with another one."""
    owner = -np.ones((res, res), np.int32)
    for t in range(T.shape[0]):
        a, b, c = T[t, 0], T[t, 1], T[t, 2]
        x0, y0 = uv[a, 0] * res, uv[a, 1] * res
        x1, y1 = uv[b, 0] * res, uv[b, 1] * res
        x2, y2 = uv[c, 0] * res, uv[c, 1] * res
        ar = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if ar == 0.0:
            continue
        if ar < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
        minx = max(int(np.floor(min(x0, min(x1, x2)) - 0.5)), 0)
        maxx = min(int(np.ceil(max(x0, max(x1, x2)) - 0.5)), res - 1)
        miny = max(int(np.floor(min(y0, min(y1, y2)) - 0.5)), 0)
        maxy = min(int(np.ceil(max(y0, max(y1, y2)) - 0.5)), res - 1)
        ex = (x1 - x0, x2 - x1, x0 - x2); ey = (y1 - y0, y2 - y1, y0 - y2)
        ox = (x0, x1, x2); oy = (y0, y1, y2)
        for py in range(miny, maxy + 1):
            cy = py + 0.5
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                inside = True
                for k in range(3):
                    e = ex[k] * (cy - oy[k]) - ey[k] * (cx - ox[k])
                    if e < 0 or (e == 0 and not ((ey[k] == 0 and ex[k] < 0) or ey[k] > 0)):
                        inside = False
                        break
                if inside:
                    o = owner[py, px]
                    if o >= 0:
                        bad[o] = True
                        bad[t] = True
                    else:
                        owner[py, px] = t


def overlapping_triangles(U, T, res=None):
    """Bool per triangle: involved in a UV overlap (texel-centre raster)."""
    bad = np.zeros(len(T), np.bool_)
    if len(T) < 2:
        return bad
    lo = U.min(0); ext = float((U.max(0) - lo).max())
    if not np.isfinite(ext) or ext <= 0:
        return bad
    if res is None:
        res = int(np.clip(np.sqrt(len(T)) * 2.5, 64, 3072))
    uv = np.ascontiguousarray((U - lo) / ext * 0.999, np.float64)
    _raster_all(uv, np.ascontiguousarray(T, np.int64), res, bad)
    return bad


def peel_by_regions(topo, piece, region, max_rounds=40):
    """Detach whole regions (atoms) that carry folds or overlaps from a part's
    layout, worst first, so the rest of the part stays one island.
    ``region`` is a per-global-face label. Returns list of pieces."""
    faces, T, U = piece
    keep = np.ones(len(faces), bool)
    detached = []
    reg = region[faces]
    for _ in range(max_rounds):
        rows = np.nonzero(keep)[0]
        if len(rows) == 0:
            break
        sub = (faces[rows], T[rows], U)
        sa = _signed(U, T[rows])
        bad = (sa < 0) | overlapping_triangles(U, T[rows])
        if not bad.any():
            break
        r = reg[rows]
        ur, inv = np.unique(r, return_inverse=True)
        nb_ = np.bincount(inv, weights=bad.astype(float))
        tot = np.bincount(inv)
        score = nb_ / np.maximum(tot, 1)
        # detach every region that is mostly involved, else the single worst one
        worst = np.nonzero(score >= max(0.5 * score.max(), 0.05))[0]
        if len(worst) == len(ur):
            worst = np.array([int(np.argmax(score))])
        drop = np.isin(r, ur[worst])
        detached.append(rows[drop])
        keep[rows[drop]] = False
    out = []
    for rows in [np.nonzero(keep)[0]] + detached:
        if len(rows):
            out += _pieces(faces, T, U, rows)
    return out


def raster_overlap(U, T, res=None):
    """Triangles (ta, tb) overlapping in UV (texel-centre raster), or None.
    Resolution adapts to the chart size (about 2 texels per triangle)."""
    if len(T) < 2:
        return None
    lo = U.min(0); ext = float((U.max(0) - lo).max())
    if not np.isfinite(ext) or ext <= 0:
        return None
    if res is None:
        res = int(np.clip(np.sqrt(len(T)) * 2.0, 64, 2048))
    uv = np.ascontiguousarray((U - lo) / ext * 0.999, np.float64)
    a, b = _raster_pair(uv, np.ascontiguousarray(T, np.int64), res)
    return None if a < 0 else (int(a), int(b))


def is_disk(T):
    from .charts import is_disk as _d
    return _d(T)[0]


def collision(U, T):
    """Local triangle indices (fa, fb) owning two crossing boundary edges, or None."""
    a, b, fo = _boundary_with_faces(np.ascontiguousarray(T, np.int64))
    if len(a) < 3:
        return None
    A = np.ascontiguousarray(U[a], np.float64); B = np.ascontiguousarray(U[b], np.float64)
    lo = U.min(0); ext = U.max(0) - lo
    L = np.linalg.norm(B - A, axis=1)
    cell = max(float(np.median(L)) * 4, float(ext.max()) / 1024, 1e-30)
    nx = int(ext[0] / cell) + 1; ny = int(ext[1] / cell) + 1
    i, j = _first_hit(A, B, cell, lo, nx, ny)
    if i < 0:
        return None
    return int(fo[i]), int(fo[j])


@nb.njit(cache=True, nogil=True)
def _split_local_nb(FF, faces, loc, C, N, sa, sb, normal_w):
    """Two-source Dijkstra over the piece's own faces (local arrays only)."""
    n = faces.shape[0]
    INF = 1e300
    dist = np.full(n, INF)
    lab = -np.ones(n, np.int64)
    cap = 4 * n + 16
    hd = np.empty(cap); hv = np.empty(cap, np.int64)
    hn = 0
    for s_, l_ in ((sa, 0), (sb, 1)):
        dist[s_] = 0.0; lab[s_] = l_
        hd[hn] = 0.0; hv[hn] = s_; hn += 1
    while hn > 0:
        d = hd[0]; i = hv[0]
        hn -= 1; hd[0] = hd[hn]; hv[0] = hv[hn]
        q = 0
        while True:
            l = 2 * q + 1; r = l + 1; m_ = q
            if l < hn and hd[l] < hd[m_]:
                m_ = l
            if r < hn and hd[r] < hd[m_]:
                m_ = r
            if m_ == q:
                break
            hd[q], hd[m_] = hd[m_], hd[q]; hv[q], hv[m_] = hv[m_], hv[q]
            q = m_
        if d > dist[i]:
            continue
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            j = loc[g]
            if j < 0:
                continue
            dc = np.sqrt(((C[i] - C[j]) ** 2).sum())
            nd = d + dc * (1.0 + normal_w * (1.0 - (N[i] * N[j]).sum()))
            if nd < dist[j]:
                dist[j] = nd; lab[j] = lab[i]
                if hn < cap:
                    hd[hn] = nd; hv[hn] = j
                    q = hn; hn += 1
                    while q > 0:
                        pq = (q - 1) // 2
                        if hd[pq] <= hd[q]:
                            break
                        hd[pq], hd[q] = hd[q], hd[pq]; hv[pq], hv[q] = hv[q], hv[pq]
                        q = pq
    for i in range(n):
        if lab[i] < 0:
            lab[i] = 0
    return lab


def _split_local(topo, faces, fa, fb, N):
    """Seam-aware 2-region growth inside ``faces`` seeded at local fa, fb."""
    loc = topo._loc_buf
    loc[faces] = np.arange(len(faces))
    try:
        C = topo._centroids[faces]
        lab = _split_local_nb(topo.FF, faces, loc, C, np.ascontiguousarray(N[faces]), fa, fb, 8.0)
    finally:
        loc[faces] = -1
    return lab


def _pieces(faces, T, U, rows):
    """Sub-piece of a chart (rows of T), split into UV-connected components."""
    from .metrics import uv_islands
    Tr = T[rows]
    used, inv = np.unique(Tr, return_inverse=True)
    Ts = inv.reshape(-1, 3)
    comp, nc = uv_islands(Ts, len(used))
    out = []
    for k in range(nc):
        r = np.nonzero(comp == k)[0] if nc > 1 else np.arange(len(rows))
        u2, inv2 = np.unique(Ts[r], return_inverse=True)
        out.append((faces[rows[r]], inv2.reshape(-1, 3), U[used[u2]]))
    return out


def fix_overlaps(topo, pieces, N=None, max_splits=400, flips_ok=False):
    """pieces: list of (faces_global, T, U). Returns valid pieces and stats.
    Each colliding chart is split at its first collision until clean."""
    if N is None:
        N = np.ascontiguousarray(topo.face_normal)
    if not hasattr(topo, "_centroids"):
        topo._centroids = topo.V[topo.F].mean(1)
    if not hasattr(topo, "_loc_buf"):
        topo._loc_buf = -np.ones(len(topo.F), np.int64)
    from . import param
    from .charts import untangle_boundary
    out = []
    work = [(p, 0, 0) for p in pieces]
    splits = 0
    repaired = untangled = 0
    while work:
        (faces, T, U), depth, tried = work.pop()
        if not np.isfinite(U).all():
            # numerically broken layout: one flat (isometric) island per triangle
            for fi in faces:
                P = topo.V[topo.F[fi]]
                e1 = P[1] - P[0]; e2 = P[2] - P[0]
                l1 = max(np.linalg.norm(e1), 1e-30)
                x2 = float(e2 @ e1) / l1
                y2 = float(np.sqrt(max((e2 @ e2) - x2 * x2, 0.0)))
                out.append((np.array([fi]), np.array([[0, 1, 2]]), np.array([[0.0, 0.0], [l1, 0.0], [x2, y2]])))
            splits += 1
            continue
        sa = _signed(U, T)
        bad = sa < 0          # zero-area slivers cover nothing and are left in place
        if bad.any() and not flips_ok:
            if bad.all():
                # consistently mirrored piece: mirror back, then check it like any other
                work.append(((faces, T, U * np.array([-1.0, 1.0])), depth + 1, tried))
                continue
            if not tried & 1 and len(faces) > 2 and bad.sum() < 0.5 * len(faces):
                # first try to unfold in place (interior vertices to their ring centroid)
                U2 = param.repair_flips(U, T)
                if np.isfinite(U2).all() and (_signed(U2, T) < 0).sum() < bad.sum():
                    repaired += 1
                    work.append(((faces, T, U2), depth, tried | 1))
                    continue
                tried |= 1
            # folded triangles overlap their neighbours: detach them as their own
            # (mirrored, internally consistent) pieces
            splits += 1
            for rows in (np.nonzero(~bad)[0], np.nonzero(bad)[0]):
                for pc in _pieces(faces, T, U, rows):
                    work.append((pc, depth + 1, tried))
            continue
        hit = collision(U, T) if len(faces) > 1 else None
        if hit is None and len(faces) > 1 and not is_disk(T):
            # a ring can wrap over itself without any boundary crossing
            hit = raster_overlap(U, T)
        if hit is None or len(faces) <= 1:
            out.append((faces, T, U))
            continue
        if not tried & 2 and is_disk(T):
            # boundary crossings: untangle locally before cutting the piece
            U2, ok_ = untangle_boundary(U, T, rounds=30)
            if ok_:
                untangled += 1
                work.append(((faces, T, U2), depth, tried | 2))
                continue
            tried |= 2
        if depth >= max_splits:
            # last resort: one flat island per triangle (always valid)
            for r in range(len(faces)):
                Ur = U[T[r]]
                if _signed(Ur, np.array([[0, 1, 2]]))[0] < 0:
                    Ur = Ur * np.array([-1.0, 1.0])
                out.append((faces[r:r + 1], np.array([[0, 1, 2]]), Ur))
            continue
        lab = _split_local(topo, faces, hit[0], hit[1], N)
        splits += 1
        for l in (0, 1):
            rows = np.nonzero(lab == l)[0]
            if len(rows):
                for pc in _pieces(faces, T, U, rows):
                    work.append((pc, depth + 1, tried))
    return out, {"overlap_splits": splits, "overlap_repaired": repaired, "overlap_untangled": untangled}


def _signed(U, T):
    a, b, c = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
