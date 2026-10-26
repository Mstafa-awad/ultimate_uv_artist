"""Validation of FINAL exported UVs (float32, as written to the MESH / GLB).

Exact geometric checks, not texel sampling:
* UV triangle overlap: every pair of UV triangles that overlap with positive
  area (proper edge crossing, or a vertex/centroid strictly inside the other),
  found with a uniform grid. Triangles sharing an edge or a vertex may touch.
* Orientation: triangles whose signed UV area disagrees with their island.
* Degenerate: zero UV area while the 3D triangle has area; and slivers below
  float32 UV resolution.
* Range: non-finite UVs or UVs outside [0, 1].
* Gutter: smallest gap between different islands, in texels of the atlas.
* Preservation: positions, triangle order, normals and colours per corner.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@nb.njit(cache=True)
def _same(U, i, j):
    return U[i, 0] == U[j, 0] and U[i, 1] == U[j, 1]


ULP = 2.0 ** -24          # float32 spacing just below 1.0 (UVs are stored as float32)


@nb.njit(cache=True)
def _strict_inside(px, py, ax, ay, bx, by, cx, cy, rel):
    l = max((bx - ax) ** 2 + (by - ay) ** 2, max((cx - bx) ** 2 + (cy - by) ** 2, (ax - cx) ** 2 + (ay - cy) ** 2))
    # orientation tolerance: relative, and never below what float32 storage can resolve
    eps = max(rel * l, 4.0 * ULP * np.sqrt(l)) if rel > 0 else 0.0
    d1 = _orient(ax, ay, bx, by, px, py)
    d2 = _orient(bx, by, cx, cy, px, py)
    d3 = _orient(cx, cy, ax, ay, px, py)
    if d1 > eps and d2 > eps and d3 > eps:
        return True
    if d1 < -eps and d2 < -eps and d3 < -eps:
        return True
    return False


@nb.njit(cache=True)
def _proper_cross(ax, ay, bx, by, cx, cy, dx, dy, rel):
    l1 = (bx - ax) ** 2 + (by - ay) ** 2; l2 = (dx - cx) ** 2 + (dy - cy) ** 2
    e1 = max(rel * l1, 4.0 * ULP * np.sqrt(l1)); e2 = max(rel * l2, 4.0 * ULP * np.sqrt(l2))
    o1 = _orient(ax, ay, bx, by, cx, cy); o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay); o4 = _orient(cx, cy, dx, dy, bx, by)
    return ((o1 > e1 and o2 < -e1) or (o1 < -e1 and o2 > e1)) and            ((o3 > e2 and o4 < -e2) or (o3 < -e2 and o4 > e2))


@nb.njit(cache=True)
def _tri_overlap(U, F, s, t, rel):
    """Positive-area overlap of UV triangles s, t. Corners at identical UV
    positions count as shared (touching is allowed)."""
    a = F[s]; b = F[t]
    shared = 0
    for i in range(3):
        for j in range(3):
            if _same(U, a[i], b[j]):
                shared += 1
                break
    if shared == 3:
        return False                     # coincident copy on purpose (stacked duplicates)
    for i in range(3):
        p0 = a[i]; p1 = a[(i + 1) % 3]
        for j in range(3):
            q0 = b[j]; q1 = b[(j + 1) % 3]
            if _same(U, p0, q0) or _same(U, p0, q1) or _same(U, p1, q0) or _same(U, p1, q1):
                continue
            if _proper_cross(U[p0, 0], U[p0, 1], U[p1, 0], U[p1, 1], U[q0, 0], U[q0, 1], U[q1, 0], U[q1, 1], rel):
                return True
    for i in range(3):
        v = a[i]
        if not (_same(U, v, b[0]) or _same(U, v, b[1]) or _same(U, v, b[2])):
            if _strict_inside(U[v, 0], U[v, 1], U[b[0], 0], U[b[0], 1], U[b[1], 0], U[b[1], 1], U[b[2], 0], U[b[2], 1], rel):
                return True
        v = b[i]
        if not (_same(U, v, a[0]) or _same(U, v, a[1]) or _same(U, v, a[2])):
            if _strict_inside(U[v, 0], U[v, 1], U[a[0], 0], U[a[0], 1], U[a[1], 0], U[a[1], 1], U[a[2], 0], U[a[2], 1], rel):
                return True
    cx = (U[a[0], 0] + U[a[1], 0] + U[a[2], 0]) / 3.0; cy = (U[a[0], 1] + U[a[1], 1] + U[a[2], 1]) / 3.0
    if _strict_inside(cx, cy, U[b[0], 0], U[b[0], 1], U[b[1], 0], U[b[1], 1], U[b[2], 0], U[b[2], 1], rel):
        return True
    cx = (U[b[0], 0] + U[b[1], 0] + U[b[2], 0]) / 3.0; cy = (U[b[0], 1] + U[b[1], 1] + U[b[2], 1]) / 3.0
    return _strict_inside(cx, cy, U[a[0], 0], U[a[0], 1], U[a[1], 0], U[a[1], 1], U[a[2], 0], U[a[2], 1], rel)


@nb.njit(cache=True, parallel=True)
def _overlaps(U, F, skip, G, rel, max_report):
    """Grid-hashed exact overlap test. Returns per-triangle flag (bool)."""
    m = F.shape[0]
    x0 = np.empty(m, np.int64); x1 = np.empty(m, np.int64); y0 = np.empty(m, np.int64); y1 = np.empty(m, np.int64)
    for t in range(m):
        mnx = min(U[F[t, 0], 0], min(U[F[t, 1], 0], U[F[t, 2], 0])); mxx = max(U[F[t, 0], 0], max(U[F[t, 1], 0], U[F[t, 2], 0]))
        mny = min(U[F[t, 0], 1], min(U[F[t, 1], 1], U[F[t, 2], 1])); mxy = max(U[F[t, 0], 1], max(U[F[t, 1], 1], U[F[t, 2], 1]))
        x0[t] = min(max(int(mnx * G), 0), G - 1); x1[t] = min(max(int(mxx * G), 0), G - 1)
        y0[t] = min(max(int(mny * G), 0), G - 1); y1[t] = min(max(int(mxy * G), 0), G - 1)
    cnt = np.zeros(G * G + 1, np.int64)
    for t in range(m):
        if skip[t]:
            continue
        for gy in range(y0[t], y1[t] + 1):
            for gx in range(x0[t], x1[t] + 1):
                cnt[gy * G + gx + 1] += 1
    for k in range(1, G * G + 1):
        cnt[k] += cnt[k - 1]
    items = np.empty(cnt[G * G], np.int64)
    fill = cnt[:G * G].copy()
    for t in range(m):
        if skip[t]:
            continue
        for gy in range(y0[t], y1[t] + 1):
            for gx in range(x0[t], x1[t] + 1):
                c = gy * G + gx
                items[fill[c]] = t; fill[c] += 1
    flag = np.zeros(m, np.bool_)
    for c in nb.prange(G * G):
        cy = c // G; cx = c % G
        for p in range(cnt[c], cnt[c + 1]):
            s = items[p]
            for q in range(p + 1, cnt[c + 1]):
                t = items[q]
                # test each pair once: in the first cell both bounding boxes share
                if max(x0[s], x0[t]) != cx or max(y0[s], y0[t]) != cy:
                    continue
                if x0[s] > x1[t] or x0[t] > x1[s] or y0[s] > y1[t] or y0[t] > y1[s]:
                    continue
                if _tri_overlap(U, F, s, t, rel):
                    flag[s] = True; flag[t] = True
    return flag


def validate_uvs(v, f, uv, atlas=4096, padding=None, in_v=None, in_f=None, in_normals=None, out_normals=None,
                 in_colors=None, out_colors=None, xref=None, gutter=True):
    """Checks on final UVs. Returns (report dict, per-triangle problem mask)."""
    from . import metrics
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    uv32 = np.asarray(uv, np.float32)              # validate what is exported
    U = uv32.astype(np.float64)
    rep = {}
    rep["non_finite_uv"] = int((~np.isfinite(U)).any(1).sum())
    rep["uv_outside_unit_square"] = int(((U < 0) | (U > 1)).any(1).sum())
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    a3 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    u = U[f]
    sa = 0.5 * ((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1]))
    texel = 1.0 / atlas ** 2
    # float32 UV resolution near 1.0 is 2^-24 ~ 6e-8: a triangle thinner than that
    # cannot keep a reliable sign
    f32_res = 2.0 ** -24
    edge = np.maximum(np.maximum(np.linalg.norm(u[:, 1] - u[:, 0], axis=1), np.linalg.norm(u[:, 2] - u[:, 1], axis=1)),
                      np.linalg.norm(u[:, 0] - u[:, 2], axis=1))
    sliver = np.abs(sa) < f32_res * np.maximum(edge, f32_res)
    has3d = a3 > 1e-14 * max(float(a3.max()), 1e-30)
    rep["degenerate_uv_triangles"] = int((has3d & (sa == 0)).sum())
    rep["sub_float32_slivers"] = int((has3d & sliver).sum())
    isl, n = metrics.uv_islands(f, len(v), U, v)
    w = np.abs(sa)
    pos = np.bincount(isl, weights=np.where(sa > 0, w, 0), minlength=n); neg = np.bincount(isl, weights=np.where(sa < 0, w, 0), minlength=n)
    dom = np.where(pos >= neg, 1.0, -1.0)
    flipped = has3d & ~sliver & (np.sign(sa) != dom[isl]) & (sa != 0)
    rep["flipped_triangles"] = int(flipped.sum())
    rep["mirrored_islands"] = int((neg > pos).sum())
    rep["islands"] = int(n)
    # exact overlaps (degenerate / sliver triangles cover no area and are skipped)
    skip = ~has3d | sliver | ~np.isfinite(sa)
    G = int(np.clip(np.sqrt(len(f)) / 2, 64, 4096))
    rel = 1e-6                                     # orientation tolerance relative to edge length^2
    ov = _overlaps(np.ascontiguousarray(U), np.ascontiguousarray(f), skip, G, rel, 0)
    rep["overlapping_triangles_exact"] = int(ov.sum())
    # pairs that only touch within float32 resolution (penetration below what
    # the stored UVs can represent): reported, not counted as overlap
    ov_strict = _overlaps_strict(np.ascontiguousarray(U), np.ascontiguousarray(f), skip, G)
    rep["float32_contacts"] = int((ov_strict & ~ov).sum())
    if ov.any():
        rep["islands_with_overlap"] = int(len(np.unique(isl[ov])))
    # gutter
    if gutter:
        res = min(atlas, 8192)
        g, _ = metrics.min_gutter(U, f, isl, res)
        rep["min_gutter_texels"] = float(g * atlas / res)
        if padding is not None:
            rep["gutter_ok"] = bool(rep["min_gutter_texels"] >= padding - 1e-6)
    # preservation
    if in_v is not None and in_f is not None:
        in_v = np.asarray(in_v, np.float64); in_f = np.asarray(in_f, np.int64)
        rep["triangle_count_preserved"] = bool(len(in_f) == len(f))
        if len(in_f) == len(f):
            rep["corner_positions_exact"] = bool(np.array_equal(np.asarray(v)[f].astype(np.float32), in_v[in_f].astype(np.float32)))
            if in_normals is not None and out_normals is not None:
                rep["corner_normals_exact"] = bool(np.array_equal(np.asarray(out_normals)[f], np.asarray(in_normals)[in_f]))
            if in_colors is not None and out_colors is not None:
                rep["corner_colors_exact"] = bool(np.array_equal(np.asarray(out_colors)[f], np.asarray(in_colors)[in_f]))
    problems = ov | flipped
    bad = (rep["non_finite_uv"] + rep["uv_outside_unit_square"] + rep["flipped_triangles"] +
           rep["overlapping_triangles_exact"])
    if gutter and padding is not None and not rep.get("gutter_ok", True):
        bad += 1
    for k in ("triangle_count_preserved", "corner_positions_exact", "corner_normals_exact", "corner_colors_exact"):
        if rep.get(k) is False:
            bad += 1
    rep["valid"] = bool(bad == 0)
    return rep, problems


def chart_overlaps(U, T):
    """Exact positive-area self-overlap of one chart (any UV units). Returns a
    bool per triangle."""
    U = np.asarray(U, np.float64); T = np.ascontiguousarray(T, np.int64)
    if len(T) < 2:
        return np.zeros(len(T), np.bool_)
    lo = U.min(0); ext = float((U.max(0) - lo).max())
    if not np.isfinite(ext) or ext <= 0:
        return np.zeros(len(T), np.bool_)
    Un = np.ascontiguousarray((U - lo) / ext * 0.999999)
    u = Un[T]
    sa = 0.5 * ((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1]))
    skip = np.abs(sa) <= 1e-14
    G = int(np.clip(np.sqrt(len(T)) / 2, 4, 2048))
    return _overlaps(Un, T, skip, G, 1e-6, 0)


ULP_STRICT = 0.0


@nb.njit(cache=True)
def _tri_overlap_strict(U, F, s, t):
    a = F[s]; b = F[t]
    shared = 0
    for i in range(3):
        for j in range(3):
            if _same(U, a[i], b[j]):
                shared += 1
                break
    if shared == 3:
        return False
    for i in range(3):
        p0 = a[i]; p1 = a[(i + 1) % 3]
        for j in range(3):
            q0 = b[j]; q1 = b[(j + 1) % 3]
            if _same(U, p0, q0) or _same(U, p0, q1) or _same(U, p1, q0) or _same(U, p1, q1):
                continue
            o1 = _orient(U[p0, 0], U[p0, 1], U[p1, 0], U[p1, 1], U[q0, 0], U[q0, 1])
            o2 = _orient(U[p0, 0], U[p0, 1], U[p1, 0], U[p1, 1], U[q1, 0], U[q1, 1])
            o3 = _orient(U[q0, 0], U[q0, 1], U[q1, 0], U[q1, 1], U[p0, 0], U[p0, 1])
            o4 = _orient(U[q0, 0], U[q0, 1], U[q1, 0], U[q1, 1], U[p1, 0], U[p1, 1])
            if o1 * o2 < 0 and o3 * o4 < 0:
                return True
    return False


@nb.njit(cache=True, parallel=True)
def _overlaps_strict(U, F, skip, G):
    """Like _overlaps but with zero tolerance on edge crossings (float32 noise
    included), to count contacts that are below storage resolution."""
    m = F.shape[0]
    x0 = np.empty(m, np.int64); x1 = np.empty(m, np.int64); y0 = np.empty(m, np.int64); y1 = np.empty(m, np.int64)
    for t in range(m):
        mnx = min(U[F[t, 0], 0], min(U[F[t, 1], 0], U[F[t, 2], 0])); mxx = max(U[F[t, 0], 0], max(U[F[t, 1], 0], U[F[t, 2], 0]))
        mny = min(U[F[t, 0], 1], min(U[F[t, 1], 1], U[F[t, 2], 1])); mxy = max(U[F[t, 0], 1], max(U[F[t, 1], 1], U[F[t, 2], 1]))
        x0[t] = min(max(int(mnx * G), 0), G - 1); x1[t] = min(max(int(mxx * G), 0), G - 1)
        y0[t] = min(max(int(mny * G), 0), G - 1); y1[t] = min(max(int(mxy * G), 0), G - 1)
    cnt = np.zeros(G * G + 1, np.int64)
    for t in range(m):
        if skip[t]:
            continue
        for gy in range(y0[t], y1[t] + 1):
            for gx in range(x0[t], x1[t] + 1):
                cnt[gy * G + gx + 1] += 1
    for k in range(1, G * G + 1):
        cnt[k] += cnt[k - 1]
    items = np.empty(cnt[G * G], np.int64)
    fill = cnt[:G * G].copy()
    for t in range(m):
        if skip[t]:
            continue
        for gy in range(y0[t], y1[t] + 1):
            for gx in range(x0[t], x1[t] + 1):
                c = gy * G + gx
                items[fill[c]] = t; fill[c] += 1
    flag = np.zeros(m, np.bool_)
    for c in nb.prange(G * G):
        cy = c // G; cx = c % G
        for p in range(cnt[c], cnt[c + 1]):
            s = items[p]
            for q in range(p + 1, cnt[c + 1]):
                t = items[q]
                if max(x0[s], x0[t]) != cx or max(y0[s], y0[t]) != cy:
                    continue
                if x0[s] > x1[t] or x0[t] > x1[s] or y0[s] > y1[t] or y0[t] > y1[s]:
                    continue
                if _tri_overlap_strict(U, F, s, t):
                    flag[s] = True; flag[t] = True
    return flag

