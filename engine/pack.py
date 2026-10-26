"""Shape-accurate raster atlas packer.

Charts are rasterised conservatively (every texel a triangle touches) on a
planning grid, dilated by the gutter, and placed with an exact first-fit
search over free-run tables (so islands nest into holes and concavities).
Four 90-degree rotations are tried per chart after aligning each chart to
its minimum-area bounding rectangle. A binary search finds the largest
uniform texel scale that fits one square page.
"""
from __future__ import annotations

import time

import numba as nb
import numpy as np


# ------------------------------------------------------------------ orientation
def min_area_rect_angle(U):
    from scipy.spatial import ConvexHull
    if len(U) < 3:
        return 0.0
    try:
        hull = U[ConvexHull(U).vertices]
    except Exception:
        return 0.0
    e = np.roll(hull, -1, 0) - hull
    ang = np.arctan2(e[:, 1], e[:, 0])
    best = (np.inf, 0.0)
    for a in np.unique(np.round(np.mod(ang, np.pi / 2), 9)):
        c, s = np.cos(-a), np.sin(-a)
        R = hull @ np.array([[c, -s], [s, c]]).T
        ext = R.max(0) - R.min(0)
        area = ext[0] * ext[1]
        if area < best[0] - 1e-12:
            best = (area, a)
    return best[1]


def orient(U, upright=None):
    """Rotate a chart to its minimum-area rectangle, landscape."""
    a = min_area_rect_angle(U)
    c, s = np.cos(-a), np.sin(-a)
    V = U @ np.array([[c, -s], [s, c]]).T
    ext = V.max(0) - V.min(0)
    if ext[1] > ext[0]:
        V = V[:, ::-1] * np.array([1, -1])
    return V - V.min(0)


# ------------------------------------------------------------------ raster
@nb.njit(cache=True, nogil=True)
def _conservative(U, T, h, w, mask):
    for t in range(T.shape[0]):
        x0, y0 = U[T[t, 0], 0], U[T[t, 0], 1]
        x1, y1 = U[T[t, 1], 0], U[T[t, 1], 1]
        x2, y2 = U[T[t, 2], 0], U[T[t, 2], 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
        minx = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        maxx = min(int(np.floor(max(x0, max(x1, x2)))), w - 1)
        miny = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        maxy = min(int(np.floor(max(y0, max(y1, y2)))), h - 1)
        ex = (x1 - x0, x2 - x1, x0 - x2); ey = (y1 - y0, y2 - y1, y0 - y2)
        ox = (x0, x1, x2); oy = (y0, y1, y2)
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                inside = True
                for k in range(3):
                    # conservative: test the cell corner that maximises the edge function
                    cx = px + (0.0 if ey[k] > 0 else 1.0)
                    cy = py + (1.0 if ex[k] > 0 else 0.0)
                    e = ex[k] * (cy - oy[k]) - ey[k] * (cx - ox[k])
                    if e < 0:
                        inside = False
                        break
                if inside:
                    mask[py, px] = 1


def chart_mask(U, T, scale, pad):
    """U oriented (min 0). Returns mask (h,w) incl. padding border, and offset."""
    V = U * scale + pad
    w = int(np.ceil(V[:, 0].max() + pad)) + 1
    h = int(np.ceil(V[:, 1].max() + pad)) + 1
    m = np.zeros((h, w), np.uint8)
    _conservative(np.ascontiguousarray(V), np.ascontiguousarray(T, np.int64), h, w, m)
    if pad > 0:
        from scipy.ndimage import maximum_filter
        m = maximum_filter(m, size=2 * pad + 1, mode="constant", cval=0)
    return m


@nb.njit(cache=True, nogil=True)
def _intervals_nb(mask):
    h, w = mask.shape
    ptr = np.zeros(h + 1, np.int64)
    a = np.empty(h * (w // 2 + 1), np.int64); b = np.empty(h * (w // 2 + 1), np.int64)
    need = np.zeros(h, np.int64)
    n = 0
    for j in range(h):
        i = 0
        while i < w:
            if mask[j, i]:
                s = i
                while i < w and mask[j, i]:
                    i += 1
                a[n] = s; b[n] = i - 1; n += 1
                if i - s > need[j]:
                    need[j] = i - s
            else:
                i += 1
        ptr[j + 1] = n
    return ptr, a[:n].copy(), b[:n].copy(), need


def _intervals(mask):
    return _intervals_nb(np.ascontiguousarray(mask, np.uint8))


# ------------------------------------------------------------------ placement
@nb.njit(cache=True, nogil=True)
def _find(R, O, rowmax, H, W, h, w, ptr, ia, ib, need, rows):
    """First-fit (top-left) exact placement. ``rows``: mask rows, most
    constraining first, so impossible positions are rejected early."""
    for y in range(0, H - h + 1):
        ok = True
        for jj in range(h):
            j = rows[jj]
            if need[j] > rowmax[y + j]:
                ok = False
                break
        if not ok:
            continue
        x = 0
        while x <= W - w:
            fits = True
            skip = 1
            for jj in range(h):
                j = rows[jj]
                for q in range(ptr[j], ptr[j + 1]):
                    L = ib[q] - ia[q] + 1
                    r = R[y + j, x + ia[q]]
                    if r < L:
                        fits = False
                        # cell x+ia+r is occupied: jump past that whole occupied run
                        c = x + ia[q] + r
                        skip = r + (O[y + j, c] if c < W else 1)
                        break
                if not fits:
                    break
            if fits:
                return x, y
            x += skip
    return -1, -1


@nb.njit(cache=True, nogil=True)
def _fits(R, y, x, h, ptr, ia, ib, rows):
    for jj in range(h):
        j = rows[jj]
        for q in range(ptr[j], ptr[j + 1]):
            if R[y + j, x + ia[q]] < ib[q] - ia[q] + 1:
                return False
    return True


@nb.njit(cache=True, nogil=True)
def _find_contact(R, rowmax, occ, G, h, w, ptr, ia, ib, need, rows, ry, rx, stride):
    """Maximal-contact placement: among feasible positions on a stride grid,
    pick the one whose surrounding ring touches the most occupied texels or
    page border (ties: lower, then further left)."""
    best = -1.0
    bx = -1; by = -1
    nr = ry.shape[0]
    for y in range(0, G - h + 1, stride):
        ok = True
        for jj in range(h):
            j = rows[jj]
            if need[j] > rowmax[y + j]:
                ok = False
                break
        if not ok:
            continue
        x = 0
        while x <= G - w:
            if _fits(R, y, x, h, ptr, ia, ib, rows):
                c = 0
                for q in range(nr):
                    yy = y + ry[q]; xx = x + rx[q]
                    if yy < 0 or xx < 0 or yy >= G or xx >= G:
                        c += 1
                    elif occ[yy, xx]:
                        c += 1
                score = c / nr - 1e-3 * (y + h) / G - 1e-4 * (x + w) / G
                if score > best:
                    best = score; bx = x; by = y
                x += stride
            else:
                x += 1
    if bx < 0 or stride == 1:
        return bx, by, best
    # local refinement at full precision around the best coarse position
    cx, cy = bx, by
    for y in range(max(0, cy - stride), min(G - h, cy + stride) + 1):
        for x in range(max(0, cx - stride), min(G - w, cx + stride) + 1):
            if not _fits(R, y, x, h, ptr, ia, ib, rows):
                continue
            c = 0
            for q in range(nr):
                yy = y + ry[q]; xx = x + rx[q]
                if yy < 0 or xx < 0 or yy >= G or xx >= G:
                    c += 1
                elif occ[yy, xx]:
                    c += 1
            score = c / nr - 1e-3 * (y + h) / G - 1e-4 * (x + w) / G
            if score > best:
                best = score; bx = x; by = y
    return bx, by, best


def _ring(mask):
    """Offsets of the 1-texel ring just outside a mask."""
    from scipy.ndimage import binary_dilation
    pm = np.pad(mask.astype(bool), 1)
    ring = binary_dilation(pm) & ~pm
    yy, xx = np.nonzero(ring)
    return (yy - 1).astype(np.int64), (xx - 1).astype(np.int64)


@nb.njit(cache=True, nogil=True)
def _place(occ, R, O, rowmax, x, y, mask):
    """Mark a mask occupied; refresh free-run (R), occupied-run (O) and row max."""
    h, w = mask.shape
    W = occ.shape[1]
    for j in range(h):
        for i in range(w):
            if mask[j, i]:
                occ[y + j, x + i] = 1
        yy = y + j
        run = 0
        orun = 0
        mx = 0
        for i in range(W - 1, -1, -1):
            if occ[yy, i]:
                run = 0
                orun += 1
            else:
                run += 1
                orun = 0
            R[yy, i] = run
            O[yy, i] = orun
            if run > mx:
                mx = run
        rowmax[yy] = mx


@nb.njit(cache=True, parallel=True)
def _find_rots(R, O, rowmax, G, hs, ws, ptr_all, poff, ia_all, ib_all, need_all, rows_all, roff):
    """Search every rotation of one chart in parallel. Arrays are concatenated
    per rotation; ptr entries are absolute offsets into ia_all/ib_all."""
    k = hs.shape[0]
    out = np.full((k, 2), -1, np.int64)
    for r in nb.prange(k):
        h = hs[r]; w = ws[r]
        if h <= 0 or h > G or w > G:
            continue
        x, y = _find(R, O, rowmax, G, G, h, w, ptr_all[poff[r]:poff[r] + h + 1], ia_all, ib_all,
                     need_all[roff[r]:roff[r] + h], rows_all[roff[r]:roff[r] + h])
        out[r, 0] = x; out[r, 1] = y
    return out


@nb.njit(cache=True, nogil=True)
def _find_rots_serial(R, O, rowmax, G, hs, ws, ptr_all, poff, ia_all, ib_all, need_all, rows_all, roff):
    k = hs.shape[0]
    out = np.full((k, 2), -1, np.int64)
    for r in range(k):
        h = hs[r]; w = ws[r]
        if h <= 0 or h > G or w > G:
            continue
        x, y = _find(R, O, rowmax, G, G, h, w, ptr_all[poff[r]:poff[r] + h + 1], ia_all, ib_all,
                     need_all[roff[r]:roff[r] + h], rows_all[roff[r]:roff[r] + h])
        out[r, 0] = x; out[r, 1] = y
    return out


def _bundle(ms):
    hs, ws, ptrs, ias, ibs, needs, rows = [], [], [], [], [], [], []
    poff, roff = [], []
    po = ro = io = 0
    for m, iv in ms:
        if m is None:
            hs.append(0); ws.append(0); poff.append(po); roff.append(ro)
            continue
        ptr, a, b, need = iv
        hs.append(m.shape[0]); ws.append(m.shape[1])
        poff.append(po); roff.append(ro)
        ptrs.append(ptr + io); ias.append(a); ibs.append(b); needs.append(need)
        rows.append(np.argsort(-need, kind="stable").astype(np.int64))
        po += len(ptr); ro += len(need); io += len(a)
    cat = lambda xs: np.concatenate(xs).astype(np.int64) if xs else np.zeros(1, np.int64)
    return (np.array(hs, np.int64), np.array(ws, np.int64), cat(ptrs), np.array(poff, np.int64),
            cat(ias), cat(ibs), cat(needs), cat(rows), np.array(roff, np.int64))


CONTACT_MIN_FRACTION = 0.004     # islands above this share of the page get contact placement
CONTACT_MAX_CHARTS = 0           # measured worse than first-fit on real charts (kept for experiments)


def _try_pack(masks, order, G, deadline=None, serial=False):
    occ = np.zeros((G, G), np.uint8)
    R = np.tile(np.arange(G, 0, -1, dtype=np.int32), (G, 1))
    O = np.zeros((G, G), np.int32)
    rowmax = np.full(G, G, np.int64)
    out = [None] * len(masks)
    big_area = CONTACT_MIN_FRACTION * G * G
    n_contact = 0
    for idx in order:
        ms, bundle = masks[idx]
        hs, ws = bundle[0], bundle[1]
        best = None
        area = ms[0][0].sum()
        if area >= big_area and n_contact < CONTACT_MAX_CHARTS:
            # large islands: maximal-contact placement over all rotations
            n_contact += 1
            for rot, (m, iv) in enumerate(ms):
                if m is None or m.shape[0] > G or m.shape[1] > G:
                    continue
                ptr, a, b, need = iv
                rows = np.argsort(-need, kind="stable").astype(np.int64)
                ry, rx = _ring(m)
                stride = max(1, min(m.shape) // 24, G // 512)
                x, y, sc = _find_contact(R, rowmax, occ, G, m.shape[0], m.shape[1], ptr, a, b, need, rows, ry, rx, stride)
                if x >= 0 and (best is None or sc > best[0]):
                    best = (sc, rot, x, y)
            if best is not None:
                _, rot, x, y = best
                _place(occ, R, O, rowmax, x, y, ms[rot][0])
                out[idx] = (rot, x, y)
                continue
        big = (not serial) and int(bundle[0].max()) * int(bundle[1].max()) > 4096
        pos = (_find_rots if big else _find_rots_serial)(R, O, rowmax, G, *bundle)
        for rot in range(len(hs)):
            x, y = int(pos[rot, 0]), int(pos[rot, 1])
            if x < 0:
                continue
            score = (y + hs[rot]) * G + (x + ws[rot])
            if best is None or score < best[0]:
                best = (score, rot, x, y)
        if best is None:
            return None
        _, rot, x, y = best
        _place(occ, R, O, rowmax, x, y, ms[rot][0])
        out[idx] = (rot, x, y)
        if deadline is not None and time.perf_counter() > deadline:
            return None
    return out, occ


ANGLES = (0.0,)


def _rotated(U, a):
    if a == 0.0:
        return U
    c, s_ = np.cos(a), np.sin(a)
    V = U @ np.array([[c, -s_], [s_, c]]).T
    return V - V.min(0)


def _masks_at(Us, Ts, dens, s, pad, rot_set):
    """Per chart: masks for every (pre-rotation angle, 90-degree turn); index
    k = angle_index * len(rot_set) + turn."""
    masks = []
    for U, T, d in zip(Us, Ts, dens):
        ms = []
        for a in ANGLES:
            m0 = chart_mask(_rotated(U, a), T, s * d, pad)
            tiny = m0.shape[0] * m0.shape[1] < 400
            for r in rot_set:
                if tiny and (r >= 2 or a != 0.0):
                    ms.append((None, None)); continue     # tiny islands: two orientations are enough
                m = np.ascontiguousarray(np.rot90(m0, r))
                ms.append((m, _intervals(m)))
        masks.append((ms, _bundle(ms)))
    return masks


ORDERINGS = {
    "area": lambda M: np.array([mm[0][0][0].sum() for mm in M]),
    "bbox": lambda M: np.array([mm[0][0][0].shape[0] * mm[0][0][0].shape[1] for mm in M]),
    "height": lambda M: np.array([min(mm[0][0][0].shape) for mm in M]),
}


def _attempt(Us, Ts, dens, G, pad, rot_set, s, ordering="area", serial=False):
    masks = _masks_at(Us, Ts, dens, s, pad, rot_set)
    order = np.argsort(-ORDERINGS[ordering](masks), kind="stable")
    res = _try_pack(masks, order, G, serial=serial)
    return None if res is None else (s, res, masks)


PROBES = 1          # scales tried concurrently per search round (k-ary search when > 1)
MANY_CHARTS = 1500  # from this many islands the k-ary (4 probes) search is used


def _search(Us, Ts, dens, G, pad, rot_set, s, iters, tol=0.004, lo=0.0, hi=None, best=None, ordering="area",
            deadline=None):
    # many islands: each attempt is slow and the search is time-bound, so try
    # several scales at once (threads); otherwise the plain bisection is cheaper
    if PROBES > 1 or len(Us) >= MANY_CHARTS:
        return _search_parallel(Us, Ts, dens, G, pad, rot_set, s, iters, tol, lo, hi, best, ordering, deadline)
    return _search_serial(Us, Ts, dens, G, pad, rot_set, s, iters, tol, lo, hi, best, ordering, deadline)


def _search_parallel(Us, Ts, dens, G, pad, rot_set, s, iters, tol, lo, hi, best, ordering, deadline):
    """k-ary bracketing search: PROBES scales packed concurrently per round
    (serial kernels, threads), bracket shrinks ~PROBES+1 times per round."""
    from concurrent.futures import ThreadPoolExecutor
    k = max(PROBES, 4)
    rounds = max(2, int(np.ceil(iters / np.log2(k + 1))))
    with ThreadPoolExecutor(max_workers=k) as ex:
        for _ in range(rounds):
            if hi is not None and lo > 0 and (hi - lo) / lo < tol:
                break
            if deadline is not None and best is not None and time.perf_counter() > deadline:
                break
            if lo == 0.0 and hi is None:
                cand = s * np.array([0.8, 0.9, 1.0, 1.1])[:k]
            elif hi is None:
                cand = lo * np.array([1.08, 1.17, 1.27, 1.4])[:k]
            elif lo == 0.0:
                cand = hi * np.array([0.55, 0.7, 0.8, 0.9])[:k]
            else:
                cand = np.linspace(lo, hi, k + 2)[1:-1]
            res = list(ex.map(lambda sc: _attempt(Us, Ts, dens, G, pad, rot_set, float(sc), ordering, True), cand))
            for sc, r in zip(cand, res):
                if r is not None and sc > lo:
                    best, lo = r, float(sc)
            fails = [float(sc) for sc, r in zip(cand, res) if r is None and sc > lo]
            if fails:
                hi = min(fails) if hi is None else min(hi, min(fails))
    return best, lo, hi


def _search_serial(Us, Ts, dens, G, pad, rot_set, s, iters, tol=0.004, lo=0.0, hi=None, best=None, ordering="area",
                   deadline=None):
    """Bracketing binary search for the largest scale whose packing fits. Past
    ``deadline`` it stops refining (only once a valid packing exists)."""
    for it in range(iters):
        if hi is not None and lo > 0 and (hi - lo) / lo < tol:
            break
        if deadline is not None and best is not None and time.perf_counter() > deadline:
            break
        r = _attempt(Us, Ts, dens, G, pad, rot_set, s, ordering)
        if r is not None:
            best = r; lo = s
        else:
            hi = s
        if hi is None:
            s *= 1.2
        elif lo == 0.0:
            s *= 0.8
        else:
            s = 0.5 * (lo + hi)
    return best, lo, hi


def pack(charts, atlas=4096, padding=4, grid=None, rotations=4, density=None, iters=10, log=None,
         time_budget=None, orderings=("area", "bbox", "height"), angles=(0.0,), fine_attempts=8):
    """charts: list of (U, T) in 3D units. density: per-chart linear texel
    multipliers (1 = uniform). Returns list of UV arrays in [0,1] and info.
    Scale search runs coarse-to-fine: most binary-search steps on a half-size
    grid, then a narrow bracket on the full planning grid."""
    t0 = time.perf_counter()
    global ANGLES
    ANGLES = tuple(float(a) for a in angles)      # extra pre-rotations (each adds four 90-degree turns)
    deadline = None if not time_budget else t0 + float(time_budget)
    grid = grid or min(atlas, 2048)
    pad = int(np.ceil(padding * grid / atlas))
    n = len(charts)
    dens = np.ones(n) if density is None else np.asarray(density, float)
    Us = [orient(U) for U, _ in charts]
    Ts = [np.ascontiguousarray(T, np.int64) for _, T in charts]
    area = np.array([_uv_area(U, T) * d * d for U, T, d in zip(Us, Ts, dens)])
    tot = max(area.sum(), 1e-30)
    rot_set = [0, 1, 2, 3][:max(1, rotations)]
    best = None
    if grid >= 1024:
        g2 = grid // 2
        pad2 = max(1, int(np.ceil(padding * g2 / atlas)))
        # multi-start: the island ordering that reaches the largest scale on the coarse grid wins
        b2 = None; ordering = orderings[0]
        for o in orderings:
            if deadline is not None and b2 is not None and time.perf_counter() > t0 + 0.4 * float(time_budget):
                break
            bo, _, _ = _search(Us, Ts, dens, g2, pad2, rot_set,
                               (b2[0] if b2 is not None else np.sqrt(0.62 * g2 * g2 / tot)), iters, tol=0.01, ordering=o,
                               deadline=None if deadline is None else t0 + 0.6 * float(time_budget))
            if bo is not None and (b2 is None or bo[0] > b2[0]):
                b2 = bo; ordering = o
        if b2 is not None:
            # full grid: start just under the coarse estimate, bracket upwards
            s_est = b2[0] * grid / g2
            s0 = s_est * 0.98
            r = None
            used = 0
            for _ in range(4):
                r = _attempt(Us, Ts, dens, grid, pad, rot_set, s0, ordering)
                used += 1
                if r is not None:
                    break
                s0 *= 0.96
            if r is not None:
                # the coarse grid's conservative masks underestimate: climb on the
                # full grid while packing still succeeds, then bisect the bracket
                lo_, hi_ = s0, s0 * 1.06
                for _ in range(8):
                    if (deadline is not None and time.perf_counter() > deadline) or used >= fine_attempts:
                        break
                    used += 1
                    r2 = _attempt(Us, Ts, dens, grid, pad, rot_set, hi_, ordering)
                    if r2 is None:
                        break
                    r, lo_, hi_ = r2, hi_, hi_ * 1.06
                best = r
                if fine_attempts - used > 0:
                    best, _, _ = _search(Us, Ts, dens, grid, pad, rot_set, 0.5 * (lo_ + hi_),
                                         min(3, fine_attempts - used), tol=0.006,
                                         lo=lo_, hi=hi_, best=r, ordering=ordering, deadline=deadline)
    if best is None:
        best, _, _ = _search(Us, Ts, dens, grid, pad, rot_set, np.sqrt(0.62 * grid * grid / tot), iters)
    if best is None:
        raise RuntimeError("packing failed: charts do not fit one page even at small scale")
    s, (placement, occ), masks = best
    masks = [mm[0] for mm in masks]
    uvs = []
    nr = len(rot_set)
    for i, (U, T, d) in enumerate(zip(Us, Ts, dens)):
        k, x, y = placement[i]
        ai, rot = k // nr, k % nr
        m0 = masks[i][ai * nr][0]
        h0, w0 = m0.shape
        V = _rotated(U, ANGLES[ai]) * s * d + pad          # coordinates inside unrotated mask (x right, y down)
        V = _rot_coords(V, rot, h0, w0)
        uv = (V + np.array([x, y])) / grid
        uvs.append(uv)
    info = {"texel_scale_grid": s, "texels_per_unit_atlas": s * atlas / grid, "grid": grid, "grid_padding": pad,
            "plan_occupancy": float(occ.mean()), "seconds": time.perf_counter() - t0}
    if log:
        log(f"pack: {n} charts grid {grid} pad {pad} scale {s:.4g} occ(dilated) {occ.mean():.3f} {info['seconds']:.1f}s")
    return uvs, info


def _rot_coords(V, rot, h, w):
    """Map continuous coords in an (h,w) mask to coords in np.rot90(mask, rot)."""
    x, y = V[:, 0], V[:, 1]
    rot %= 4
    if rot == 0:
        return np.stack([x, y], 1)
    if rot == 1:      # rot90 CCW: new[i,j] = old[j, w-1-i] -> (x,y) -> (y, w - x)
        return np.stack([y, w - x], 1)
    if rot == 2:
        return np.stack([w - x, h - y], 1)
    return np.stack([h - y, x], 1)


def _uv_area(U, T):
    a, b, c = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
    return float(np.abs(0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))).sum())
