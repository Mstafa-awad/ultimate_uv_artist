"""Part segmentation: which triangles belong to the same meaningful object part.

Pipeline (works on the original triangles, any resolution):
1. Superfaces: compact regions (~150-400 triangles) that never cross strong
   creases - the units that are labelled.
2. Evidence between neighbouring superfaces
   * geometry: concave grooves and sharp creases along the shared boundary
     (parts meet in concave valleys - the "minima rule" - or at hard edges);
     smooth convex continuation means the same part;
   * multi-view masks (optional, SAM 2): the mesh is rendered from many
     directions, each image is segmented, masks are lifted onto the triangles
     through exact triangle-ID buffers, and two superfaces get affinity from
     how often they fall in the same mask across the views where both are
     visible (contradictions between views are averaged out).
3. Average-linkage region merging over the superface graph (boundary-length
   weighted), so a single weak link cannot chain-merge unrelated parts.
   Disconnected components are never merged.
4. Refinement: tiny fragments are absorbed by the neighbour they share the
   longest boundary with; every part is one connected region.
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from . import refine, segment
from .features import dihedral_from_normals


def superfaces(topo, Ns, ef, target_faces=250, max_count=40000):
    m = len(topo.F)
    n = int(np.clip(m / target_faces, 1, max_count))
    # strong creases bound superfaces (they are candidate part boundaries)
    dih = dihedral_from_normals(topo, Ns)
    strong = (dih < -np.radians(28)) | (dih > np.radians(60)) | ef["hard"]
    sf = segment.make_atoms(topo, strong, n_atoms=n, N=Ns, normal_w=6.0)
    sf = refine.fix_fragments(topo, sf, strong)
    return sf, dih


def _pair_arrays(topo, lab):
    f = np.repeat(np.arange(len(topo.F)), 3)
    g = topo.FF.ravel(); e = topo.FE.ravel()
    sel = (g >= 0) & (lab[f] >= 0)
    sel[sel] &= lab[g[sel]] >= 0
    f, g, e = f[sel], g[sel], e[sel]
    a, b = lab[f], lab[g]
    s = a < b
    return a[s], b[s], e[s]


def geometric_affinity(topo, lab, dih):
    """Per superface pair: 1 = smooth continuation, 0 = groove / hard crease."""
    a, b, e = _pair_arrays(topo, lab)
    n = int(lab.max()) + 1
    key = a * n + b
    uk, inv = np.unique(key, return_inverse=True)
    L = topo.edge_len[e]
    conc = np.clip((-dih[e] - np.radians(8)) / np.radians(30), 0, 1)
    conv = np.clip((dih[e] - np.radians(40)) / np.radians(40), 0, 1)
    strength = np.maximum(conc, conv)
    Lt = np.bincount(inv, weights=L)
    Ls = np.bincount(inv, weights=L * strength)
    return uk // n, uk % n, Lt, 1.0 - Ls / np.maximum(Lt, 1e-30)


def view_memberships(topo, lab, renderer, views, mask_fn, log=print, min_pixels=6):
    """Per view: (superface ids visible, bool matrix superface x mask) - a
    superface belongs to a mask when the mask covers most of its visible pixels."""
    n = int(lab.max()) + 1
    out = []
    for vi, view in enumerate(views):
        az, el = view[0], view[1]
        zoom = view[2] if len(view) > 2 else 1.0
        img, fid = renderer.render(az, el, zoom=zoom)
        masks = mask_fn(img)
        if not masks:
            continue
        vis = fid >= 0
        sfl = lab[fid[vis]]
        okp = sfl >= 0
        tot = np.bincount(sfl[okp], minlength=n)
        seen = np.nonzero(tot >= min_pixels)[0]
        if len(seen) == 0:
            continue
        idx = -np.ones(n, np.int64); idx[seen] = np.arange(len(seen))
        M = np.zeros((len(seen), len(masks)), bool)
        flat_sf = np.full(fid.shape, -1, np.int64); flat_sf[vis] = sfl
        for mi, mk in enumerate(masks):
            s_ = flat_sf[mk]
            s_ = s_[s_ >= 0]
            if len(s_) == 0:
                continue
            c = np.bincount(s_, minlength=n)
            hit = np.nonzero((c >= 0.5 * tot) & (tot >= min_pixels))[0]
            M[idx[hit], mi] = True
        out.append((idx, M))
    return out


def view_affinity(a, b, memberships):
    """Mean over views of the Jaccard overlap of the mask sets covering a and b
    (masks come at several granularities: parts share part-level masks, while
    different parts only share masks that were dropped as whole-object)."""
    acc = np.zeros(len(a)); co = np.zeros(len(a))
    for idx, M in memberships:
        ia, ib = idx[a], idx[b]
        both = (ia >= 0) & (ib >= 0)
        if not both.any():
            continue
        A = M[ia[both]]; B = M[ib[both]]
        inter = (A & B).sum(1); union = (A | B).sum(1)
        has = union > 0
        j = np.where(has, inter / np.maximum(union, 1), 0.0)
        w = np.nonzero(both)[0]
        acc[w[has]] += j[has]; co[w[has]] += 1
    aff = np.where(co >= 2, acc / np.maximum(co, 1), np.nan)
    return aff, co


def merge_regions(topo, lab, a, b, L, aff, thresh=0.55, min_frac=0.0):
    """Average-linkage agglomeration: region-pair affinity is the boundary-length
    weighted mean of the superface-pair affinities along their shared border."""
    n = int(lab.max()) + 1
    parent = np.arange(n)
    S = {}                     # (ra, rb) -> [sum L*aff, sum L]
    nbr = {i: set() for i in range(n)}
    for x, y, l, w in zip(a.tolist(), b.tolist(), L.tolist(), aff.tolist()):
        S[(x, y)] = [l * w, l]
        nbr[x].add(y); nbr[y].add(x)
    heap = [(-(s / l), x, y) for (x, y), (s, l) in S.items() if l > 0]
    heapq.heapify(heap)
    alive = np.ones(n, bool)
    while heap:
        negv, x, y = heapq.heappop(heap)
        if not (alive[x] and alive[y]):
            continue
        k = (x, y) if (x, y) in S else (y, x)
        if k not in S:
            continue
        s, l = S[k]
        if abs(-negv - s / l) > 1e-12:
            continue                       # stale entry
        if -negv < thresh:
            break
        # merge y into x
        alive[y] = False
        parent[y] = x
        for z in list(nbr[y]):
            if z == x:
                continue
            ky = (y, z) if (y, z) in S else (z, y)
            sy, ly = S.pop(ky)
            kx = (x, z) if (x, z) in S else ((z, x) if (z, x) in S else None)
            if kx is None:
                S[(x, z)] = [sy, ly]; kx = (x, z)
                nbr[x].add(z); nbr[z].add(x)
            else:
                S[kx][0] += sy; S[kx][1] += ly
            nbr[z].discard(y)
            sx, lx = S[kx]
            heapq.heappush(heap, (-(sx / lx), kx[0], kx[1]))
        S.pop(k, None)
        nbr[x].discard(y)
        nbr[y] = set()
    # resolve parents
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    out = lab.copy()
    ok = lab >= 0
    out[ok] = parent[lab[ok]]
    return out


def absorb_small(topo, part, min_area_frac):
    """Parts below ``min_area_frac`` of their component's area join the
    neighbouring part they share the longest boundary with."""
    for _ in range(4):
        ok = part >= 0
        n = int(part.max()) + 1
        area = np.bincount(part[ok], weights=topo.face_area[ok], minlength=n)
        thr = min_area_frac * topo.face_area[ok].sum()
        small = area < thr
        if not small.any():
            break
        a, b, e = _pair_arrays(topo, part)
        if len(a) == 0:
            break
        L = topo.edge_len[e]
        src = np.r_[a, b]; dst = np.r_[b, a]; LL = np.r_[L, L]
        m = small[src]
        if not m.any():
            break
        key = src[m] * n + dst[m]
        uk, inv = np.unique(key, return_inverse=True)
        Ls = np.bincount(inv, weights=LL[m])
        s_ = uk // n; d_ = uk % n
        o = np.lexsort((-Ls, s_))
        first = np.r_[True, s_[o][1:] != s_[o][:-1]]
        target = np.arange(n)
        sel = o[first]
        # do not chain small into small in the same round
        target[s_[sel]] = np.where(small[d_[sel]] & (area[d_[sel]] < area[s_[sel]]), s_[sel], d_[sel])
        new = target[part[ok]]
        if np.array_equal(new, part[ok]):
            break
        part = part.copy(); part[ok] = new
    return part


def segment_parts(topo, Ns, ef, cfg, log=print, mask_fn=None, views=None):
    t0 = time.perf_counter()
    info = {}
    sf, dih = superfaces(topo, Ns, ef, cfg.get("seg_superface_faces", 40))
    info["superfaces"] = int(sf.max()) + 1
    a, b, L, geo = geometric_affinity(topo, sf, dih)
    aff = geo.copy()
    if mask_fn is not None:
        from .views import Renderer, default_views
        R = Renderer(topo.V, topo.F, cfg.get("seg_view_size", 1024))
        try:
            vlist = views or default_views(zoom=cfg.get("seg_zoom_views", True))
            nv_ = cfg.get("seg_views")
            if nv_ and nv_ < len(vlist):
                vlist = [vlist[i] for i in np.linspace(0, len(vlist) - 1, nv_).round().astype(int)]
            mem = view_memberships(topo, sf, R, vlist, mask_fn, log)
        finally:
            R.release()
        vaff, co = view_affinity(a, b, mem)
        have = np.isfinite(vaff)
        # Jaccard ~1/3 already means "same part at a coarser granularity": rescale
        vs = np.clip(vaff / cfg.get("seg_jaccard_same", 0.45), 0, 1)
        w = cfg.get("seg_view_weight", 0.6)
        aff[have] = w * vs[have] + (1 - w) * geo[have]
        info["pairs_with_view_evidence"] = int(have.sum())
        info["views"] = len(mem)
    part = merge_regions(topo, sf, a, b, L, aff, cfg.get("seg_merge_threshold", 0.8))
    part = absorb_small(topo, part, cfg.get("seg_min_part", 1.5e-4))
    part = refine.fix_fragments(topo, part, np.zeros(len(topo.E), bool))
    from .merge import part_edge_confidence
    cfg["_part_edge_conf"] = part_edge_confidence(topo, part, sf, a, b, aff, cfg.get("seg_merge_threshold", 0.8))
    info["parts"] = int(part.max()) + 1
    info["seconds"] = time.perf_counter() - t0
    log(f"parts: {info}")
    return part, info
