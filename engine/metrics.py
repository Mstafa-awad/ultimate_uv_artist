"""Independent UV quality metrics (used by benchmarks and by the node report).

All metrics take the *output* mesh (vertices, faces, uvs) and, optionally, the
original mesh for preservation checks. They never modify their inputs.
"""
from __future__ import annotations

import numpy as np
import numba as nb


# ---------------------------------------------------------------- rasterizer
@nb.njit(cache=True)
def _raster_counts(uv, faces, res, counts):
    """Count, per texel centre, how many triangles cover it (top-left rule so
    texel centres on shared edges are counted once)."""
    for t in range(faces.shape[0]):
        a, b, c = faces[t, 0], faces[t, 1], faces[t, 2]
        x0, y0 = uv[a, 0] * res, uv[a, 1] * res
        x1, y1 = uv[b, 0] * res, uv[b, 1] * res
        x2, y2 = uv[c, 0] * res, uv[c, 1] * res
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            continue
        if area < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
        minx = max(int(np.floor(min(x0, min(x1, x2)) - 0.5)), 0)
        maxx = min(int(np.ceil(max(x0, max(x1, x2)) - 0.5)), res - 1)
        miny = max(int(np.floor(min(y0, min(y1, y2)) - 0.5)), 0)
        maxy = min(int(np.ceil(max(y0, max(y1, y2)) - 0.5)), res - 1)
        # edge setup (CCW): e(p) = (bx-ax)*(py-ay) - (by-ay)*(px-ax) >= 0 inside
        ex = (x1 - x0, x2 - x1, x0 - x2)
        ey = (y1 - y0, y2 - y1, y0 - y2)
        ox = (x0, x1, x2)
        oy = (y0, y1, y2)
        for py in range(miny, maxy + 1):
            cy = py + 0.5
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                inside = True
                for k in range(3):
                    e = ex[k] * (cy - oy[k]) - ey[k] * (cx - ox[k])
                    if e < 0:
                        inside = False
                        break
                    if e == 0:
                        # top-left rule: accept only top or left edges
                        top_left = (ey[k] == 0 and ex[k] < 0) or ey[k] > 0
                        if not top_left:
                            inside = False
                            break
                if inside:
                    if counts[py, px] < 255:
                        counts[py, px] += 1


def raster_coverage(uv, faces, res=4096):
    counts = np.zeros((res, res), np.uint8)
    _raster_counts(np.ascontiguousarray(uv, np.float64), np.ascontiguousarray(faces, np.int64), int(res), counts)
    covered = int(np.count_nonzero(counts))
    overlap = int(np.count_nonzero(counts > 1))
    return counts, covered / float(res * res), overlap


# ---------------------------------------------------------------- islands
@nb.njit(cache=True)
def _find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        n = p[x]; p[x] = r; x = n
    return r


@nb.njit(cache=True)
def _face_islands(faces, nv):
    p = np.arange(nv)
    for t in range(faces.shape[0]):
        a = _find(p, faces[t, 0])
        for k in range(1, 3):
            b = _find(p, faces[t, k])
            if a != b:
                p[b] = a
    lab = np.empty(faces.shape[0], np.int64)
    for t in range(faces.shape[0]):
        lab[t] = _find(p, faces[t, 0])
    return lab


def uv_islands(faces, n_vertices, uv=None, vertices=None):
    """Islands = faces connected through shared output vertices. With ``uv``
    and ``vertices`` given, output vertices with identical position and UV are
    treated as one (attribute splits such as hard normals are not UV seams)."""
    if uv is not None and vertices is not None:
        key = np.concatenate([np.asarray(vertices, np.float32), np.asarray(uv, np.float32)], 1)
        key = np.ascontiguousarray(key).view(np.dtype((np.void, 20))).reshape(-1)
        _, inv = np.unique(key, return_inverse=True)
        faces = inv.reshape(-1)[faces]
        n_vertices = int(inv.max()) + 1
    lab = _face_islands(np.ascontiguousarray(faces, np.int64), int(n_vertices))
    _, inv = np.unique(lab, return_inverse=True)
    return inv.astype(np.int32), int(inv.max()) + 1 if len(inv) else 0


# ---------------------------------------------------------------- distortion
def _wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float("nan")
    return float(x[o][np.searchsorted(cw, q * cw[-1])].item() if np.searchsorted(cw, q * cw[-1]) < len(o) else x[o][-1])


def triangle_jacobians(P, U):
    """P: (F,3,3) 3D corners, U: (F,3,2) UV corners -> singular values, signed uv area, 3D area."""
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    n = np.cross(e1, e2)
    A3 = 0.5 * np.linalg.norm(n, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    ex = e1 / np.maximum(l1, 1e-30)[:, None]
    ey = np.cross(n / np.maximum(2 * A3, 1e-30)[:, None], ex)
    # local 2D coordinates of the 3D triangle
    x1 = l1
    x2 = np.einsum("ij,ij->i", e2, ex)
    y2 = np.einsum("ij,ij->i", e2, ey)
    du1 = U[:, 1] - U[:, 0]
    du2 = U[:, 2] - U[:, 0]
    # J * [[x1,x2],[0,y2]] = [du1 du2]
    det = x1 * y2
    good = np.abs(det) > 1e-30
    inv_det = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
    # inverse of [[x1,x2],[0,y2]] = 1/det * [[y2,-x2],[0,x1]]
    J = np.empty((len(P), 2, 2))
    J[:, :, 0] = du1 * (y2 * inv_det)[:, None]
    J[:, :, 1] = (du1 * (-x2 * inv_det)[:, None] + du2 * (x1 * inv_det)[:, None])
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    signed = 0.5 * (du1[:, 0] * du2[:, 1] - du1[:, 1] * du2[:, 0])
    # closed form singular values of 2x2
    E = (a + d) / 2; Fv = (a - d) / 2; G = (c + b) / 2; H = (c - b) / 2
    Q = np.sqrt(E * E + H * H); R = np.sqrt(Fv * Fv + G * G)
    s1 = Q + R
    s2 = np.abs(Q - R)
    return s1, s2, signed, A3, good


def distortion_report(vertices, faces, uvs, islands=None, chunk=2_000_000):
    """Area-weighted distortion statistics. UVs in [0,1] atlas space."""
    F = len(faces)
    s1 = np.empty(F); s2 = np.empty(F); sa = np.empty(F); a3 = np.empty(F); gd = np.empty(F, bool)
    ang = np.empty(F)
    for b in range(0, F, chunk):
        f = faces[b:b + chunk]
        P = vertices[f].astype(np.float64)
        U = uvs[f].astype(np.float64)
        r = triangle_jacobians(P, U)
        s1[b:b + chunk], s2[b:b + chunk], sa[b:b + chunk], a3[b:b + chunk], gd[b:b + chunk] = r
        ang[b:b + chunk] = _angle_error(P, U)
    w = a3 / max(a3.sum(), 1e-30)
    uv_abs = np.abs(sa)
    scale2 = uv_abs.sum() / max(a3.sum(), 1e-30)          # global density
    valid = gd & (a3 > 0)
    # degenerate UV triangles: zero/near-zero UV area while 3D area is not
    px8k = 1.0 / 8192 ** 2
    # sub-texel slivers (below 1/100 of an 8K texel) are at float32 UV resolution: their sign
    # can flip just by storing UVs as float32, so they count as "collapsed", not "flipped"
    collapsed = valid & (uv_abs < 1e-2 * px8k)
    # anisotropy
    aniso = np.where(s2 > 0, s1 / np.maximum(s2, 1e-300), np.inf)
    an_ok = valid & np.isfinite(aniso) & ~collapsed
    # area ratio vs global density (texel density uniformity)
    ar_g = np.where(valid, (s1 * s2) / max(scale2, 1e-300), 1.0)
    out = {}
    out["triangles"] = int(F)
    out["collapsed_uv_triangles"] = int(collapsed.sum())
    out["collapsed_uv_area_fraction"] = float(w[collapsed].sum())
    out["anisotropy_mean"] = float((aniso[an_ok] * w[an_ok]).sum() / max(w[an_ok].sum(), 1e-30))
    out["anisotropy_p95"] = _wquantile(aniso[an_ok], w[an_ok], 0.95)
    out["angle_error_deg_mean"] = float((ang[valid] * w[valid]).sum() / max(w[valid].sum(), 1e-30))
    lg = np.abs(np.log2(np.maximum(ar_g, 1e-30)))
    ok = valid & ~collapsed
    out["density_log2_dev_mean"] = float((lg[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-30))
    out["density_ratio_p95"] = float(2 ** _wquantile(lg[ok], w[ok], 0.95))
    # Sander L2 stretch normalised by global density (1 = ideal)
    st = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0) / np.sqrt(max(scale2, 1e-300))
    out["stretch_L2"] = float(np.sqrt((st[ok] ** 2 * w[ok]).sum() / max(w[ok].sum(), 1e-30)))
    # per-island intrinsic area distortion (density normalised per island)
    if islands is not None:
        n_is = int(islands.max()) + 1
        ia3 = np.bincount(islands, weights=a3, minlength=n_is)
        iuv = np.bincount(islands, weights=uv_abs, minlength=n_is)
        isc = iuv / np.maximum(ia3, 1e-30)
        ar_c = np.where(valid, (s1 * s2) / np.maximum(isc[islands], 1e-300), 1.0)
        lc = np.abs(np.log2(np.maximum(ar_c, 1e-30)))
        out["chart_area_log2_dev_mean"] = float((lc[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-30))
        out["chart_area_ratio_p95"] = float(2 ** _wquantile(lc[ok], w[ok], 0.95))
        # flips relative to the island's dominant orientation
        pos = np.bincount(islands, weights=np.where(sa > 0, a3, 0.0), minlength=n_is)
        neg = np.bincount(islands, weights=np.where(sa < 0, a3, 0.0), minlength=n_is)
        dom = np.where(pos >= neg, 1.0, -1.0)
        flipped = valid & ~collapsed & (np.sign(sa) != dom[islands])
        out["flipped_triangles"] = int(flipped.sum())
        out["flipped_area_fraction"] = float(w[flipped].sum())
        out["mirrored_islands"] = int((dom < 0).sum())
        # texel density spread between islands (area weighted, uniform target)
        d = np.sqrt(isc / max(scale2, 1e-300))
        wi = ia3 / max(ia3.sum(), 1e-30)
        out["island_density_ratio_p95"] = float(2 ** _wquantile(np.abs(np.log2(np.maximum(d, 1e-30))), wi, 0.95))
    return out


def _angle_error(P, U):
    def angles(X):
        a = X[:, 1] - X[:, 0]; b = X[:, 2] - X[:, 0]; c = X[:, 2] - X[:, 1]
        def ang(u, v):
            nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
            cs = np.einsum("ij,ij->i", u, v) / np.maximum(nu * nv, 1e-300)
            return np.arccos(np.clip(cs, -1, 1))
        return np.stack([ang(a, b), ang(-a, c), ang(-b, -c)], 1)
    A = angles(P)
    B = angles(np.concatenate([U, np.zeros((len(U), 3, 1))], axis=2))
    return np.degrees(np.abs(A - B).mean(axis=1))


# ---------------------------------------------------------------- seams
def seam_report(vertices, faces, uvs, weld_keys=None):
    """UV seam edges = interior edges (in the position-welded mesh) whose two
    sides disagree on UVs. Returns 3D seam length and counts."""
    if weld_keys is None:
        _, weld_keys = np.unique(vertices, axis=0, return_inverse=True)
    weld_keys = weld_keys.reshape(-1).astype(np.int64)
    W = weld_keys[faces]                                      # (F,3)
    F = len(faces)
    e0 = np.concatenate([W[:, 0], W[:, 1], W[:, 2]])
    e1 = np.concatenate([W[:, 1], W[:, 2], W[:, 0]])
    c0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    c1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    swap = e0 > e1
    lo = np.where(swap, e1, e0); hi = np.where(swap, e0, e1)
    clo = np.where(swap, c1, c0); chi = np.where(swap, c0, c1)
    key = lo * (int(weld_keys.max()) + 1) + hi
    o = np.argsort(key, kind="stable")
    key = key[o]; clo = clo[o]; chi = chi[o]
    start = np.r_[True, key[1:] != key[:-1]]
    gid = np.cumsum(start) - 1
    cnt = np.bincount(gid)
    first = np.nonzero(start)[0]
    interior = cnt == 2
    i0 = first[interior]; i1 = i0 + 1
    u = uvs.astype(np.float64)
    mism = (np.any(u[clo[i0]] != u[clo[i1]], axis=1) | np.any(u[chi[i0]] != u[chi[i1]], axis=1))
    P = vertices.astype(np.float64)
    L = np.linalg.norm(P[clo[i0]] - P[chi[i0]], axis=1)
    bnd = cnt == 1
    Lb = np.linalg.norm(P[clo[first[bnd]]] - P[chi[first[bnd]]], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(P[faces[:, 1]] - P[faces[:, 0]], P[faces[:, 2]] - P[faces[:, 0]]), axis=1).sum()
    return {"seam_edges": int(mism.sum()), "seam_length": float(L[mism].sum()),
            "seam_length_per_sqrt_area": float(L[mism].sum() / np.sqrt(max(area, 1e-30))),
            "open_boundary_edges": int(bnd.sum()), "open_boundary_length": float(Lb.sum()),
            "nonmanifold_edge_groups": int((cnt > 2).sum()), "surface_area": float(area)}


def preservation_report(in_v, in_f, out_v, out_f, in_attrs=None, out_attrs=None, chunk=2_000_000):
    """Exact per-triangle-corner equality of positions (and optional attributes)."""
    rep = {"triangle_count_equal": len(in_f) == len(out_f)}
    if not rep["triangle_count_equal"]:
        rep["positions_exact"] = False
        return rep
    ok = True
    for b in range(0, len(in_f), chunk):
        if not np.array_equal(in_v[in_f[b:b + chunk]], out_v[out_f[b:b + chunk]]):
            ok = False; break
    rep["positions_exact"] = ok
    for name in (in_attrs or {}):
        a, o = in_attrs[name], (out_attrs or {}).get(name)
        if a is None:
            continue
        if o is None:
            rep[name + "_exact"] = False; continue
        same = True
        for b in range(0, len(in_f), chunk):
            if not np.array_equal(a[in_f[b:b + chunk]], o[out_f[b:b + chunk]]):
                same = False; break
        rep[name + "_exact"] = same
    return rep


def full_report(out_v, out_f, out_uv, raster_res=4096, in_v=None, in_f=None, in_attrs=None, out_attrs=None):
    rep = {}
    isl, n_is = uv_islands(out_f, len(out_v), out_uv, out_v)
    rep["islands"] = n_is
    uv_ok = bool(np.isfinite(out_uv).all())
    rep["uv_finite"] = uv_ok
    rep["uv_in_unit_square"] = bool(out_uv.min() >= -1e-6 and out_uv.max() <= 1 + 1e-6)
    _, occ, ov = raster_coverage(out_uv, out_f, raster_res)
    rep["raster_res"] = raster_res
    rep["occupancy"] = occ
    rep["overlap_texels"] = ov
    rep.update(distortion_report(out_v, out_f, out_uv, isl))
    rep.update(seam_report(out_v, out_f, out_uv))
    if in_v is not None:
        rep.update(preservation_report(in_v, in_f, out_v, out_f, in_attrs, out_attrs))
    # island size distribution
    a3 = 0.5 * np.linalg.norm(np.cross(out_v[out_f[:, 1]] - out_v[out_f[:, 0]], out_v[out_f[:, 2]] - out_v[out_f[:, 0]]), axis=1)
    ia = np.bincount(isl, weights=a3, minlength=n_is)
    ia = np.sort(ia)[::-1] / max(ia.sum(), 1e-30)
    rep["largest_island_area_frac"] = float(ia[0]) if n_is else 0.0
    rep["islands_for_90pct_area"] = int(np.searchsorted(np.cumsum(ia), 0.9) + 1) if n_is else 0
    rep["tiny_islands_lt_0.01pct"] = int((ia < 1e-4).sum())
    return rep, isl


# ---------------------------------------------------------------- gutters
@nb.njit(cache=True)
def _raster_labels(uv, faces, lab_of_face, res, img):
    for t in range(faces.shape[0]):
        a, b, c = faces[t, 0], faces[t, 1], faces[t, 2]
        x0, y0 = uv[a, 0] * res, uv[a, 1] * res
        x1, y1 = uv[b, 0] * res, uv[b, 1] * res
        x2, y2 = uv[c, 0] * res, uv[c, 1] * res
        # conservative: include any texel whose square touches the triangle bbox and centre-distance test
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
            area = -area
        minx = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        maxx = min(int(np.floor(max(x0, max(x1, x2)))), res - 1)
        miny = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        maxy = min(int(np.floor(max(y0, max(y1, y2)))), res - 1)
        ex = (x1 - x0, x2 - x1, x0 - x2); ey = (y1 - y0, y2 - y1, y0 - y2)
        ox = (x0, x1, x2); oy = (y0, y1, y2)
        L = (np.sqrt(ex[0] ** 2 + ey[0] ** 2), np.sqrt(ex[1] ** 2 + ey[1] ** 2), np.sqrt(ex[2] ** 2 + ey[2] ** 2))
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                cx = px + 0.5; cy = py + 0.5
                inside = True
                for k in range(3):
                    # distance-to-edge test with half-texel diagonal tolerance (conservative raster)
                    e = ex[k] * (cy - oy[k]) - ey[k] * (cx - ox[k])
                    if e < -0.7072 * L[k]:
                        inside = False
                        break
                if inside:
                    img[py, px] = lab_of_face[t]


def min_gutter(uv, faces, islands, res=4096):
    """Smallest texel gap between two different islands (conservative raster,
    so a reported gap of g means at least ~g texels of empty space)."""
    from scipy import ndimage
    img = np.full((res, res), -1, np.int32)
    _raster_labels(np.ascontiguousarray(uv, np.float64), np.ascontiguousarray(faces, np.int64),
                   np.ascontiguousarray(islands, np.int32), int(res), img)
    empty = img < 0
    if empty.all() or not empty.any():
        return float("nan"), img
    d, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
    near = img[iy, ix]
    # touching islands (no empty texel between) -> gap 0
    touch = ((img[:, 1:] >= 0) & (img[:, :-1] >= 0) & (img[:, 1:] != img[:, :-1])).any() or \
            ((img[1:, :] >= 0) & (img[:-1, :] >= 0) & (img[1:, :] != img[:-1, :])).any()
    if touch:
        return 0.0, img
    dif_h = near[:, 1:] != near[:, :-1]
    dif_v = near[1:, :] != near[:-1, :]
    g = np.inf
    if dif_h.any():
        g = min(g, float((d[:, 1:] + d[:, :-1])[dif_h].min()))
    if dif_v.any():
        g = min(g, float((d[1:, :] + d[:-1, :])[dif_v].min()))
    # d counts from texel centre of empty texel to nearest filled texel centre; gap in empty texels ~ g-1
    return max(g - 1.0, 0.0), img
