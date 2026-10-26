"""Ultimate UV Artist pipeline: analyse -> chart -> flatten -> pack -> assemble.

The output keeps every input triangle in its original order; only vertices on
UV seams are duplicated. ``xref[i]`` is the input vertex that output vertex
``i`` copies, so positions, normals, colours and any other per-vertex
attribute transfer exactly.

Low-poly meshes (and small disconnected pieces of high-poly meshes) are charted
directly. Large high-poly components are planned on a decimated proxy: the
proxy is charted with full validation, the chart labels and relief seams are
transferred back to the original triangles, seams are re-smoothed at full
resolution and every chart is cut and flattened on the original geometry.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import charts as ch
from . import features, gpu, grow, overlapfix, pack, param, refine, segment, transfer, validate
from .topology import build_topology

PROFILES = {
    # atom_faces: target faces per atom (atoms = faces / atom_faces); grow_seconds caps
    # the validated merging; proxy_faces is the high-poly planning proxy size
    "fast": dict(atom_faces=90, max_atoms=600, relief_cuts=1, relief_ratio=0.6, plan_iters=4, final_iters=6,
                 slim_iters=0, grid=2048, rotations=4, max_aniso=1.45, max_area=1.75, max_overlap=0.0,
                 part_max_aniso=1.45, part_max_area=1.75,
                 proxy_faces=20_000, pack_iters=9, orderings=("area",), pack_seconds=8, pack_fine_attempts=3,
                 openness_res=96, grow_seconds=8,
                 smooth_passes=8, window_faces=2000, fullres_iters=3,
                 hier_atom_faces=900, hier_seconds=25, hier_merge_iters=4, hier_atom_iters=5,
                 seg_views=12, seg_points=16, seg_proxy_faces=100_000),
    "balanced": dict(atom_faces=60, max_atoms=1000, relief_cuts=2, relief_ratio=0.8, plan_iters=6, final_iters=12,
                     slim_iters=0, grid=2048, rotations=4, max_aniso=1.3, max_area=1.6, max_overlap=0.0,
                     part_max_aniso=1.3, part_max_area=1.6,
                     proxy_faces=40_000, pack_iters=11, orderings=("area",), pack_seconds=20, pack_fine_attempts=8, pack_orderings_small_max=600,
                     openness_res=128,
                     grow_seconds=25, smooth_passes=12, window_faces=3000, fullres_iters=4, hier_atom_faces=600, hier_seconds=60, hier_merge_iters=5,
                     hier_atom_iters=6, seg_views=16, seg_points=24, seg_proxy_faces=150_000),
    "artist": dict(atom_faces=40, max_atoms=2000, relief_cuts=4, relief_ratio=1.2, plan_iters=10, final_iters=20,
                   slim_iters=15, grid=2048, rotations=4, max_aniso=1.2, max_area=1.45, max_overlap=0.0,
                   part_max_aniso=1.2, part_max_area=1.45,
                   proxy_faces=80_000, pack_iters=13, orderings=("area", "bbox", "height"), pack_seconds=60, pack_fine_attempts=8,
                   openness_res=160,
                   grow_seconds=90, smooth_passes=16, window_faces=6000, fullres_iters=10, hier_atom_faces=400, hier_seconds=180, hier_merge_iters=8,
                   hier_atom_iters=8, seg_views=24, seg_points=32, seg_proxy_faces=200_000),
}


@dataclass
class UnwrapResult:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    xref: np.ndarray
    chart_of_face: np.ndarray
    report: dict = field(default_factory=dict)
    part_of_face: np.ndarray = None


@dataclass
class FinalChart:
    faces: np.ndarray        # global triangle ids (order matches T)
    T: np.ndarray            # (t,3) chart-local vertex ids
    U: np.ndarray            # (k,2) UVs in 3D units
    quality: dict


# ------------------------------------------------------------------ analysis
def split_normal_edges(topo, faces, normals, cos_tol=0.9998):
    """Edges whose two sides use different normals at a shared corner (hard edges)."""
    hard = np.zeros(len(topo.E), bool)
    if normals is None:
        return hard
    nrm = np.asarray(normals, np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
    f = np.repeat(np.arange(len(topo.F)), 3)
    k = np.tile(np.arange(3), len(topo.F))
    g = topo.FF.ravel()
    sel = g >= 0
    f, k, g = f[sel], k[sel], g[sel]
    e = topo.FE[f, k]
    for corner in (0, 1):
        kk = (k + corner) % 3
        w = topo.F[f, kk]
        o1 = faces[f, kk]
        idx = np.argmax(topo.F[g] == w[:, None], axis=1)
        o2 = faces[g, idx]
        d = (nrm[o1] * nrm[o2]).sum(1)
        hard[e[d < cos_tol]] = True
    return hard


def label_edges(topo, face_labels):
    hard = np.zeros(len(topo.E), bool)
    if face_labels is None:
        return hard
    fm = np.asarray(face_labels)
    f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
    ok = (f0 >= 0) & (f1 >= 0)
    hard[ok] = fm[f0[ok]] != fm[f1[ok]]
    return hard


def analyse(topo, cfg, forced=None, preferred=None, openness=None, seam_hide=1.0, hard_angle=None):
    """Per-edge seam features and costs. ``forced``: edges that must be seams
    (hard normals, materials). ``preferred``: edges that are cheap seams
    (semantic part boundaries)."""
    if openness is None:
        openness = features.face_openness(topo.V, topo.F, topo.face_normal, res=cfg["openness_res"])
    Ns = features.smooth_normals(topo, iters=cfg.get("smooth_iters", 8))
    if hard_angle is None:
        hard_angle = cfg.get("hard_angle")
    ef = features.edge_features(topo, openness, hard_angle, forced, None, smooth_N=Ns)
    costs = features.seam_cost(topo, ef, hide_weight=seam_hide)
    if preferred is not None:
        w = np.clip(np.asarray(preferred, np.float64), 0, 1)
        costs = costs * (1.0 - 0.85 * w)
    return ef, costs, Ns, openness


def color_edge_strength(topo, face_colors):
    """0..1 per edge: how strongly the vertex colours change across it
    (painted part boundaries such as clothing edges, eyes, mouths)."""
    if face_colors is None:
        return None
    c = np.asarray(face_colors, np.float64)[:, :3]
    f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
    ok = (f0 >= 0) & (f1 >= 0)
    s = np.zeros(len(topo.E))
    d = np.linalg.norm(c[np.maximum(f0, 0)] - c[np.maximum(f1, 0)], axis=1)
    s[ok] = np.clip((d[ok] - 0.06) / 0.24, 0, 1)
    return s


def face_importance(topo, face_colors, openness):
    """Relative detail importance per face for adaptive texel density: colour
    variation, geometric detail and exposure. Area-weighted mean 1."""
    imp = np.ones(len(topo.F))
    f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
    ok = (f0 >= 0) & (f1 >= 0)
    if face_colors is not None:
        c = np.asarray(face_colors, np.float64)[:, :3]
        d = np.zeros(len(topo.E)); d[ok] = np.linalg.norm(c[f0[ok]] - c[f1[ok]], axis=1)
        g = np.zeros(len(topo.F)); np.add.at(g, f0[ok], d[ok]); np.add.at(g, f1[ok], d[ok])
        imp += 0.5 * np.clip(g / max(np.percentile(g, 90), 1e-6), 0, 2)
    n0 = topo.face_normal[np.maximum(f0, 0)]; n1 = topo.face_normal[np.maximum(f1, 0)]
    k = np.zeros(len(topo.E)); k[ok] = 1 - (n0[ok] * n1[ok]).sum(1)
    gk = np.zeros(len(topo.F)); np.add.at(gk, f0[ok], k[ok]); np.add.at(gk, f1[ok], k[ok])
    imp += 0.125 * np.clip(gk / max(np.percentile(gk, 90), 1e-9), 0, 2)
    if openness is not None:
        imp *= 0.7 + 0.6 * np.asarray(openness)
    w = topo.face_area
    return imp / max((imp * w).sum() / max(w.sum(), 1e-30), 1e-30)


# ------------------------------------------------------------------ charting
def chart_direct(topo, ef, costs, Ns, cfg, log, st, face_mask=None):
    """Atoms -> validated growing. Returns grown Chart objects."""
    m = len(topo.F)
    n_atoms = cfg.get("n_atoms") or int(np.clip(m / cfg["atom_faces"], 1, cfg["max_atoms"]))
    atoms = segment.make_atoms(topo, ef["hard"], n_atoms=n_atoms, N=Ns)
    if face_mask is not None:
        atoms[~face_mask] = -1
    atoms, moved = refine.smooth_labels(topo, atoms, costs, ef["hard"], passes=cfg["smooth_passes"])
    atoms = refine.fix_fragments(topo, atoms, ef["hard"])
    st["atoms"] = st.get("atoms", 0) + int(atoms.max()) + 1
    charts, gst = grow.grow_charts(topo, atoms, ef, costs, cfg, N=Ns, log=log)
    for k, v in gst.items():
        if isinstance(v, (int, float)):
            st["grow_" + k] = st.get("grow_" + k, 0) + v
    return charts


def finalize(topo, charts, costs, cfg, log, st):
    """Final flattening of every chart; invalid charts (folds/overlaps) are
    split recursively; hopeless slivers get one flat island per triangle."""
    in_chart = np.zeros(len(topo.F), np.bool_)
    out = []
    work = [(c, 0) for c in charts]
    cap = cfg.get("max_chart_vertices")
    while work:
        c, depth = work.pop()
        if cfg.get("_check"):
            cfg["_check"]()
        if cap and len(c.faces) > 2 * cap and depth < 8:
            st["memory_splits"] = st.get("memory_splits", 0) + 1
            for sub_ in segment._bisect_faces(topo, c.faces):
                work.append((ch.Chart(topo, sub_), depth + 1))
            continue
        res_ = _final_flatten(topo, c, costs, cfg, in_chart, log)
        if res_ is not None:
            cc, U, q, (P, T, vw), score, valid = res_
            if valid:
                if score > 1.0:
                    st["final_over_budget"] = st.get("final_over_budget", 0) + 1
                out.append(FinalChart(np.asarray(c.faces, np.int64), T, U, q))
                continue
        if len(c.faces) > 2 and depth < 12:
            st["final_splits"] = st.get("final_splits", 0) + 1
            kids = grow.kids_of(topo, c)
            if kids is not None and min(len(k.faces) for k in kids) >= 0.25 * len(c.faces):
                # undo the last merge: both parts were validated while growing
                st["final_unmerges"] = st.get("final_unmerges", 0) + 1
                for k in kids:
                    work.append((k, depth + 1))
            else:
                for sub in segment.split_two(topo, c.faces):
                    work.append((ch.Chart(topo, sub), depth + 1))
            continue
        for fi in c.faces:
            X1, _ = param.local_frames(topo.V[topo.F[fi]], np.array([[0, 1, 2]]))
            out.append(FinalChart(np.array([fi]), np.array([[0, 1, 2]]), X1[0].copy(), {"flips": 0}))
        st["triangle_fallback_faces"] = st.get("triangle_fallback_faces", 0) + len(c.faces)
    return out


def _final_flatten(topo, c, costs, cfg, in_chart, log):
    """Final UVs for one chart. A chart validated during growing keeps its cut
    and UVs; extra refinement is kept only if it stays fold/overlap free.
    Otherwise flatten with its cut, then with a fresh minimal cut."""
    in_chart[c.faces] = True
    best = None
    try:
        if c.U is not None and c.quality is not None:
            cc = ch.Chart(topo, c.faces, set(c.cut))
            P, T, vw = cc.mesh(in_chart)
            if len(P) == len(c.U) and ch.is_disk(T)[0]:
                U = c.U; q = c.quality
                if cfg["final_iters"]:
                    U2, _ = param.refine(P, T, U, iters=cfg["final_iters"], line_search=True, tol=1e-5)
                    if cfg["slim_iters"]:
                        U3, _ = param.refine(P, T, U2, iters=cfg["slim_iters"], method="slim")
                        if np.isfinite(U3).all():
                            U2 = U3
                    q2 = param.chart_quality(P, T, U2)
                    q2["boundary_hits"] = ch.boundary_intersections(U2, T)
                    q2["overlap"] = 1.0 if q2["boundary_hits"] else 0.0
                    if q2["flips"] == 0 and q2["overlap"] == 0:
                        U, q = U2, q2
                valid = q["flips"] == 0 and q["overlap"] <= cfg["max_overlap"]
                score = (not valid) * 100 + max(q["aniso_p95"] / cfg["max_aniso"], q["area_p95"] / cfg["max_area"])
                best = (cc, U, q, (P, T, vw), score, valid)
                if valid:
                    return best
        tried = []
        for cut in (set(c.cut), set()):
            if cut in tried:
                continue
            tried.append(cut)
            cc = ch.Chart(topo, c.faces, cut)
            if not ch.cut_to_disk(cc, costs, in_chart):
                continue
            P, T, vw = cc.mesh(in_chart)
            try:
                U, q = ch.flatten(P, T, iters=cfg["final_iters"], slim_iters=cfg["slim_iters"])
            except Exception as exc:
                log(f"flatten failed on chart with {len(c.faces)} faces: {exc!r}")
                continue
            valid = q["flips"] == 0 and q["overlap"] <= cfg["max_overlap"]
            score = (not valid) * 100 + max(q["aniso_p95"] / cfg["max_aniso"], q["area_p95"] / cfg["max_area"])
            if best is None or score < best[4] - 1e-9:
                best = (cc, U, q, (P, T, vw), score, valid)
            if score <= 1.0:
                break
    finally:
        in_chart[c.faces] = False
    return best


# ------------------------------------------------------------------ high-poly
def _decimate(V, F, target):
    """Planning proxy. CuMesh (MIT, GPU, optional) when installed; otherwise
    fast-simplification (MIT, CPU quadric decimation). GPU decimation is not
    bit-for-bit repeatable; set UVA_PROXY_CACHE to a folder to reuse proxies
    (development A/B tests)."""
    cache = os.environ.get("UVA_PROXY_CACHE")
    if cache:
        import hashlib
        h = hashlib.sha1(np.ascontiguousarray(V, np.float32).tobytes())
        h.update(np.ascontiguousarray(F, np.int64).tobytes()); h.update(str(int(target)).encode())
        fn = os.path.join(cache, h.hexdigest()[:20] + ".npz")
        if os.path.exists(fn):
            d = np.load(fn)
            return d["v"], d["f"]
        pv, pf = _decimate_raw(V, F, target)
        os.makedirs(cache, exist_ok=True)
        np.savez(fn, v=pv, f=pf)
        return pv, pf
    return _decimate_raw(V, F, target)


def _decimate_raw(V, F, target):
    try:
        import torch
        import cumesh
        from .gpu import device
        if device().type == "cuda":
            cm = cumesh.CuMesh()
            try:
                cm.init(torch.from_numpy(np.ascontiguousarray(V, np.float32)).cuda(),
                        torch.from_numpy(np.ascontiguousarray(F, np.int32)).cuda())
                cm.simplify(int(target), verbose=False)
                pv, pf = cm.read()
                pv = pv.detach().cpu().numpy().astype(np.float64)
                pf = pf.detach().cpu().numpy().astype(np.int64)
            finally:
                del cm
                torch.cuda.empty_cache()
            if len(pf) >= 0.3 * target:
                return pv, pf
    except Exception:
        pass
    import fast_simplification as fs
    red = float(np.clip(1.0 - target / max(len(F), 1), 0.0, 0.995))
    pv, pf = fs.simplify(np.ascontiguousarray(V, np.float32), np.ascontiguousarray(F, np.int32),
                         target_reduction=red, agg=7)
    return np.asarray(pv, np.float64), np.asarray(pf, np.int64)


def _relief_seeds(ptopo, pchart):
    """Interior dangling ends of a proxy chart's cut graph (relief seam starts)."""
    if not pchart.cut:
        return np.zeros((0, 3))
    e = np.fromiter(pchart.cut, np.int64)
    E = ptopo.E[e]
    deg = np.bincount(E.ravel(), minlength=len(ptopo.V))
    ends = np.nonzero(deg == 1)[0]
    in_chart = np.zeros(len(ptopo.F), np.bool_); in_chart[pchart.faces] = True
    bnd = set(ch.Chart(ptopo, pchart.faces).boundary_welded_vertices(in_chart).tolist())
    ends = [v for v in ends.tolist() if v not in bnd]
    return ptopo.V[ends] if ends else np.zeros((0, 3))


def chart_highpoly(topo, comp, cfg, log, st, forced=None, preferred=None, seam_hide=1.0, timings=None):
    from scipy.spatial import cKDTree
    m = len(topo.F)
    csize = np.bincount(comp, minlength=int(comp.max()) + 1)
    small_thr = max(3000, int(0.002 * m))
    large = (csize[comp] >= small_thr) & ~topo.degenerate
    st["highpoly_large_faces"] = int(large.sum())
    t = time.perf_counter()
    small = ~large & ~topo.degenerate
    fl = np.nonzero(large)[0]
    used = np.unique(topo.F[fl])
    remap = -np.ones(len(topo.V), np.int64); remap[used] = np.arange(len(used))
    pv, pf = _decimate(topo.V[used], remap[topo.F[fl]], cfg["proxy_faces"])
    ptopo, _ = build_topology(pv.astype(np.float32), pf)
    st["proxy_faces"] = int(len(pf))
    ppref = None
    if cfg.get("_face_col") is not None:
        # proxy face colours from the nearest original face
        ct = cKDTree(topo.V[topo.F[fl]].mean(1))
        _, near = ct.query(ptopo.V[ptopo.F].mean(1), k=1, workers=-1)
        ppref = color_edge_strength(ptopo, cfg["_face_col"][fl[near]])
        del ct
    pef, pcosts, pNs, popen = analyse(ptopo, cfg, None, ppref, None, seam_hide)
    if timings is not None:
        timings["proxy_build"] = time.perf_counter() - t
        log(f"stage proxy_build: {timings['proxy_build']:.1f}s")
    t = time.perf_counter()
    pst = {}
    pcharts = chart_direct(ptopo, pef, pcosts, pNs, cfg, log, pst)
    st["proxy_charts"] = len(pcharts)
    st.update({"proxy_" + k: v for k, v in pst.items()})
    if timings is not None:
        timings["proxy_charting"] = time.perf_counter() - t
        log(f"stage proxy_charting: {timings['proxy_charting']:.1f}s")
    # ---- finalize the proxy charts (validated UVs to transfer)
    t = time.perf_counter()
    pfinal = finalize(ptopo, pcharts, pcosts, cfg, log, pst)
    st["proxy_final_charts"] = len(pfinal)
    if timings is not None:
        timings["proxy_finalize"] = time.perf_counter() - t
        log(f"stage proxy_finalize: {timings['proxy_finalize']:.1f}s")
    t = time.perf_counter()
    # ---- nearest normal-compatible proxy triangle for every original triangle
    pc = ptopo.V[ptopo.F].mean(1)
    tree = cKDTree(pc)
    C = topo.V[topo.F[fl]].mean(1)
    d, idx = tree.query(C, k=6, workers=-1)
    nd = (ptopo.face_normal[idx] * topo.face_normal[fl][:, None, :]).sum(2)
    ell = np.sqrt(ptopo.face_area.mean())
    pick = idx[np.arange(len(fl)), np.argmin(d / ell + 2.0 * (1.0 - nd), axis=1)]
    open_hi = np.ones(m)
    open_hi[fl] = popen[pick]
    cfg["_openness"] = open_hi
    del d, idx, nd, C, tree
    tr, fl_ok = transfer.transfer_charts(topo, fl, pick, ptopo, pfinal)
    if timings is not None:
        timings["transfer"] = time.perf_counter() - t
        log(f"stage transfer: {timings['transfer']:.1f}s")
    # ---- full-resolution relaxation of all charts at once on the GPU, then
    # targeted splits wherever a chart folds or its boundary collides
    t = time.perf_counter()
    final = []
    if tr:
        items = [(topo.V[vw], T, U) for faces, T, U, vw in tr]
        b = gpu.Batch(items)
        b.orient_positive()
        b.repair_flips(cfg.get("gpu_repair_rounds", 80))
        it = cfg.get("fullres_iters", 4)
        if it:
            b.arap(max(1, it // 2), line_search=False)
            b.repair_flips(cfg.get("gpu_repair_rounds", 80) * 2)
            b.arap(it, line_search=True)
        Us = b.chart_uvs()
        del b, items
        if timings is not None:
            timings["gpu_relax"] = time.perf_counter() - t
            log(f"stage gpu_relax: {timings['gpu_relax']:.1f}s")
        t = time.perf_counter()
        pieces, fst = overlapfix.fix_overlaps(topo, [(faces, T, U) for (faces, T, _, vw), U in zip(tr, Us)])
        st.update(fst)
        for faces, T, U in pieces:
            final.append(FinalChart(np.asarray(faces, np.int64), T, U, {"src": "fullres"}))
        st["fullres_pieces"] = len(pieces)
        if timings is not None:
            timings["overlap_fix"] = time.perf_counter() - t
            log(f"stage overlap_fix: {timings['overlap_fix']:.1f}s")
    # faces the proxy could not reach are charted on their own sub-mesh
    lost = np.setdiff1d(fl, fl_ok)
    small = small.copy()
    small[lost] = True
    st["lost_faces"] = int(len(lost))
    # ---- small components: charted directly on their own sub-mesh
    small_final = []
    if small.any():
        t = time.perf_counter()
        sf = np.nonzero(small)[0]
        sused = np.unique(topo.F[sf])
        srem = -np.ones(len(topo.V), np.int64); srem[sused] = np.arange(len(sused))
        stopo, _ = build_topology(topo.V[sused].astype(np.float32), srem[topo.F[sf]])
        sef, scosts, sNs, _ = analyse(stopo, cfg, None, None, None, seam_hide)
        sst = {}
        ncomp = len(np.unique(comp[sf]))
        scfg = dict(cfg, n_atoms=int(max(ncomp, min(cfg["max_atoms"], len(sf) / cfg["atom_faces"]))))
        sch = chart_direct(stopo, sef, scosts, sNs, scfg, log, sst)
        sfin = finalize(stopo, sch, scosts, scfg, log, sst)
        for fc in sfin:
            fc.faces = sf[fc.faces]
            fc.quality = dict(fc.quality, src="smallcomp_" + str(fc.quality.get("src", "?")))
            fc.quality = dict(fc.quality or {}, src="small")
        small_final = sfin
        st["small_piece_charts"] = len(sfin)
        st["small_piece_faces"] = int(len(sf))
        if timings is not None:
            timings["small_pieces"] = time.perf_counter() - t
        log(f"stage small_pieces: {timings['small_pieces']:.1f}s")
    return final + small_final


def chart_hier(topo, comp, cfg, log, st, forced=None, preferred=None, seam_hide=1.0, timings=None):
    """Full-resolution GPU hierarchical charting (see hier.py)."""
    from scipy.spatial import cKDTree
    from . import hier
    m = len(topo.F)
    t = time.perf_counter()
    open_hi = None
    if m > 300_000:
        # ambient openness from a small GPU-decimated proxy, mapped to the nearest proxy face
        ok = ~topo.degenerate
        pv, pf = _decimate(topo.V, topo.F[ok], 20_000)
        ptopo, _ = build_topology(pv.astype(np.float32), pf, use_gpu=False)
        popen = features.face_openness(ptopo.V, ptopo.F, ptopo.face_normal, res=cfg["openness_res"])
        _, near = cKDTree(ptopo.V[ptopo.F].mean(1)).query(topo.V[topo.F].mean(1), k=1, workers=-1)
        open_hi = popen[near]
    ef, costs, Ns, openness = analyse(topo, cfg, forced, preferred, open_hi, seam_hide)
    cfg["_openness"] = openness
    if timings is not None:
        timings["features"] = time.perf_counter() - t
        log(f"stage features: {timings['features']:.1f}s")
    t = time.perf_counter()
    # ---- part segmentation (semantic/geometric parts): islands never cross parts
    part = None
    if cfg.get("parts", True):
        part, pinfo = segment_parts_any(topo, cfg, log, costs, ef, mask_fn=cfg.get("_mask_fn"))
        st["parts"] = pinfo
        cfg["_part"] = part
    if timings is not None:
        timings["parts"] = time.perf_counter() - t
        log(f"stage parts: {timings['parts']:.1f}s ({st.get('parts', {}).get('parts')} parts)")
        vram_mark(timings, "parts")
    t = time.perf_counter()
    hard = ef["hard"].copy()
    if part is not None:
        f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
        okp = (f0 >= 0) & (f1 >= 0)
        hard[okp] |= part[f0[okp]] != part[f1[okp]]
    af = cfg["hier_atom_faces"] if m > cfg.get("highpoly_threshold", 150_000) else cfg["atom_faces"]
    n_atoms = int(np.clip(m / af, 1, cfg.get("hier_max_atoms", 20000)))
    atoms = segment.make_atoms(topo, hard, n_atoms=n_atoms, N=Ns)
    atoms, moved = refine.smooth_labels(topo, atoms, costs, hard, passes=cfg["smooth_passes"])
    atoms = refine.fix_fragments(topo, atoms, hard)
    st["atoms"] = int(atoms.max()) + 1
    if timings is not None:
        timings["atoms"] = time.perf_counter() - t
        log(f"stage atoms: {timings['atoms']:.1f}s ({st['atoms']} atoms)")
    t = time.perf_counter()
    H = hier.Hier(topo, costs, ef, cfg, Ns, log, part=part)
    whole = set()
    peeled = []
    unrolled = []
    handled = set()          # parts split into several unrolled pieces (+ leftovers)
    leftover = []            # layouts of sub-pieces that still fail (peeled below)

    def proj_piece(pf_):
        # layout of a face set taken from the per-face projection solve
        loc_ = -np.ones(m, np.int64); loc_[pf_] = np.arange(len(pf_))
        CV_, nv_ = hier.corner_ids(topo.F, topo.FF, pf_, loc_, part)
        ids_ = CV_.reshape(-1); cu__ = H.cuv[pf_].reshape(-1, 2)
        cnt_ = np.bincount(ids_, minlength=nv_)
        U__ = np.stack([np.bincount(ids_, cu__[:, k], nv_) for k in range(2)], 1) / np.maximum(cnt_, 1)[:, None]
        return (np.asarray(pf_, np.int64), CV_, U__)
    if part is not None and cfg.get("part_islands", True):
        _t0 = time.perf_counter()
        whole = H.try_parts(part)
        st["parts_single_island"] = len(whole)
        st["t_try_parts"] = time.perf_counter() - _t0
        # parts that do not project cleanly (curved lids, arch bands, tubes): exact
        # cut-to-disk + LSCM + ARAP with relief seams (CPU; parts are small here)
        if m <= cfg.get("part_unroll_max_faces", 400_000):
            okp = part >= 0
            rest = np.nonzero(okp & ~np.isin(part, list(whole)))[0]
            o_ = np.argsort(part[rest], kind="stable"); sl_ = part[rest][o_]
            cb_ = np.r_[0, np.nonzero(sl_[1:] != sl_[:-1])[0] + 1, len(sl_)]
            pcfg = dict(cfg, max_aniso=cfg.get("part_max_aniso", cfg["max_aniso"] * 1.25),
                        max_area=cfg.get("part_max_area", cfg["max_area"] * 1.4),
                        relief_cuts=cfg.get("part_relief_cuts", 4), plan_iters=max(cfg["plan_iters"], 8),
                        allow_holes=True)
            n_ok = 0
            max_depth = cfg.get("part_split_depth", 4)
            min_faces = cfg.get("part_split_min_faces", 48)
            import threading
            tls = threading.local()

            def buf():
                # per-thread scratch mask over all faces
                if getattr(tls, "b", None) is None:
                    tls.b = np.zeros(m, np.bool_)
                return tls.b

            def intrinsic(pf_):
                # cut-to-disk + LSCM/ARAP layout of a face set (may fold; peeled later)
                in_chart = buf()
                try:
                    c2 = ch.Chart(topo, pf_)
                    in_chart[pf_] = True
                    P2, T2, _ = c2.mesh(in_chart)
                    if ch.euler_and_loops(T2)[1] == 0:
                        ch.cut_to_disk(c2, costs, in_chart)
                        P2, T2, _ = c2.mesh(in_chart)
                    U2, q2 = ch.flatten(P2, T2, iters=pcfg["plan_iters"])
                    return (np.asarray(pf_, np.int64), T2, U2) if np.isfinite(U2).all() else None
                except Exception:
                    return None
                finally:
                    in_chart[pf_] = False

            def unroll_rec(pf_, depth):
                """Valid islands for a face set: whole if it unrolls cleanly, else
                split in two along creases and retry each half."""
                budget = 2.5 * float(np.sqrt(topo.face_area[pf_].sum()))
                ok_, q_, U_, cc = grow.validate(topo, pf_, set(), costs, pcfg, buf(), want_uv=True,
                                                relief_budget=budget)
                if ok_ and U_ is not None and getattr(cc, "T_local", None) is not None:
                    return [FinalChart(np.asarray(pf_, np.int64), cc.T_local, U_, {"src": "unroll"})], []
                big = len(pf_) > cfg.get("part_split_big", 1500)
                if (depth >= max_depth and not big) or depth >= 2 * max_depth or len(pf_) < 2 * min_faces:
                    return [], [pf_]
                halves = segment.split_best(topo, pf_, Ns)
                if len(halves) < 2:
                    return [], [pf_]
                oks, fails = [], []
                for h in halves:
                    o_, f_ = unroll_rec(h, depth + 1)
                    oks += o_; fails += f_
                return oks, fails

            def work(pf_):
                oks, fails = unroll_rec(pf_, 0)
                lays = []
                for fp in fails:
                    lay = intrinsic(fp)
                    valid = (lay is not None and not (param.signed_areas(lay[2], lay[1]) <= 0).any()
                             and ch.is_disk(lay[1])[0] and ch.boundary_intersections(lay[2], lay[1]) == 0)
                    if valid:
                        # relaxed acceptance is capped: a valid but badly stretched leaf is peeled
                        qv = _stretch_stats(topo.V, topo.F, [(lay[0], lay[1], lay[2])])
                        valid = qv["p95"] <= cfg.get("relaxed_max_aniso", 1.3 * cfg["max_aniso"])
                    if valid:
                        # valid but over the stretch budget (small spikes): keep it whole
                        oks.append(FinalChart(lay[0], lay[1], lay[2], {"src": "unroll_relaxed"}))
                    else:
                        lays.append(lay if lay is not None else fp)
                return oks, lays

            jobs = []
            for c0, c1 in zip(cb_[:-1], cb_[1:]):
                pf_ = rest[o_[c0:c1]]
                if 0 < len(pf_) <= cfg.get("part_unroll_max_part", 60_000):
                    jobs.append(pf_)
            jobs.sort(key=len, reverse=True)
            from concurrent.futures import ThreadPoolExecutor
            nth = cfg.get("unroll_threads", min(8, os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=max(1, nth)) as ex:
                results = list(ex.map(work, jobs))
            for pf_, (oks, lays) in zip(jobs, results):
                pid_ = int(part[pf_[0]])
                unrolled += oks
                if not lays:
                    whole.add(pid_)
                    n_ok += 1
                    continue
                handled.add(pid_)
                for lay in lays:
                    leftover.append(lay if isinstance(lay, tuple) else proj_piece(lay))
            st["parts_unrolled"] = n_ok
            st["t_unroll"] = time.perf_counter() - _t0 - st["t_try_parts"]
            log(f"parts unrolled (cut + LSCM + ARAP): {n_ok} ({st['t_unroll']:.1f}s)")
        # failed parts: keep the part layout and peel off only the folded or
        # colliding bits (undercuts, overhangs) as small separate pieces
        okp = part >= 0
        bad_faces = np.nonzero(okp & ~np.isin(part, list(whole | handled)))[0]
        _t1 = time.perf_counter()
        if len(bad_faces) or leftover:
            cb = np.zeros(1, np.int64)
            if len(bad_faces):
                locb = -np.ones(m, np.int64); locb[bad_faces] = np.arange(len(bad_faces))
                cutm = getattr(H, "part_cut", None)
                if cutm is not None:
                    CVb, nvb = hier.corner_ids_cut(topo.F, topo.FF, topo.FE, bad_faces, locb, part, cutm)
                else:
                    CVb, nvb = hier.corner_ids(topo.F, topo.FF, bad_faces, locb, part)
                idsb = CVb.reshape(-1); cub = H.cuv[bad_faces].reshape(-1, 2)
                cntb = np.bincount(idsb, minlength=nvb)
                Ub = np.stack([np.bincount(idsb, cub[:, k], nvb) for k in range(2)], 1) / np.maximum(cntb, 1)[:, None]
                ob = np.argsort(part[bad_faces], kind="stable"); slb = part[bad_faces][ob]
                cb = np.r_[0, np.nonzero(slb[1:] != slb[:-1])[0] + 1, len(slb)]
            pieces_in = list(leftover)
            for c0, c1 in zip(cb[:-1], cb[1:]):
                rows = ob[c0:c1]
                used, inv = np.unique(CVb[rows], return_inverse=True)
                pieces_in.append((bad_faces[rows], inv.reshape(-1, 3), Ub[used]))
            # peel whole atoms that carry folds/overlaps (worst first), then make the
            # few remaining pieces strictly valid
            peeled_in = []
            for pc in pieces_in:
                peeled_in += overlapfix.peel_by_regions(topo, pc, atoms)
            pieces, fst = overlapfix.fix_overlaps(topo, peeled_in, N=Ns)
            st["peel_splits"] = fst.get("overlap_splits", 0)
            # parts that shattered (many tiny pieces) go to atom merging instead
            by_part = {}
            for pc in pieces:
                by_part.setdefault(int(part[pc[0][0]]), []).append(pc)
            shattered = []
            for p_, pcs in by_part.items():
                nf = sum(len(x[0]) for x in pcs)
                if p_ not in handled and len(pcs) > max(cfg.get("peel_max_pieces", 24), nf // 4000):
                    shattered.append(p_)
                else:
                    peeled += pcs
            st["parts_peeled"] = len(by_part) - len(shattered)
            st["parts_shattered"] = len(shattered)
            # peeling only removes folds/overlaps: re-check stretch and send stretched
            # pieces back through split-and-retry (coherent sub-islands, not crumbs)
            if peeled and "unroll_rec" in locals() and cfg.get("peel_restretch", True):
                lim = cfg.get("relaxed_max_aniso", 1.3 * cfg["max_aniso"])
                redo, keep_ = [], []
                for pc in peeled:
                    if len(pc[0]) >= 3 and _stretch_stats(topo.V, topo.F, [pc])["p95"] > lim:
                        redo.append(pc)
                    else:
                        keep_.append(pc)

                def better_alone(pc):
                    # the piece re-flattened on its own, if valid and less stretched
                    lay = intrinsic(np.asarray(pc[0], np.int64))
                    if lay is None or (param.signed_areas(lay[2], lay[1]) <= 0).any() or not ch.is_disk(lay[1])[0]                             or ch.boundary_intersections(lay[2], lay[1]) != 0:
                        return pc
                    q0 = _stretch_stats(topo.V, topo.F, [pc])["p95"]
                    q1 = _stretch_stats(topo.V, topo.F, [lay])["p95"]
                    return lay if q1 < q0 else pc

                def split_stretched(pc, depth=0):
                    # still stretched after re-flattening: halve it while that
                    # measurably helps (below the budget, or >= 15 % lower p95)
                    pc = better_alone(pc)
                    q0 = _stretch_stats(topo.V, topo.F, [pc])["p95"]
                    if q0 <= lim or len(pc[0]) < 4 or depth >= cfg.get("peel_split_depth", 5):
                        return [pc]
                    halves = segment.split_best(topo, np.asarray(pc[0], np.int64), Ns)
                    if len(halves) < 2:
                        return [pc]
                    pos = {int(f): r for r, f in enumerate(pc[0])}
                    subs = []
                    for h in halves:
                        rows = np.array([pos[int(f)] for f in h], np.int64)
                        for x in overlapfix._pieces(pc[0], pc[1], pc[2], rows):
                            subs += split_stretched(x, depth + 1)
                    q1 = _stretch_stats(topo.V, topo.F, subs)["p95"]
                    return subs if (q1 <= lim or q1 < 0.85 * q0) else [pc]

                def rework(pc):
                    if len(pc[0]) < 2 * cfg.get("part_split_min_faces", 48):
                        return [], split_stretched(pc)
                    oks, fails = unroll_rec(np.asarray(pc[0], np.int64), 1)
                    rest_ = []
                    if fails:
                        pos = {int(f): r for r, f in enumerate(pc[0])}
                        for fp in fails:
                            rows = np.array([pos[int(f)] for f in fp], np.int64)
                            for x in overlapfix._pieces(pc[0], pc[1], pc[2], rows):
                                rest_ += split_stretched(x)
                    return oks, rest_

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max(1, cfg.get("unroll_threads", min(8, os.cpu_count() or 1)))) as ex:
                    res_ = list(ex.map(rework, sorted(redo, key=lambda x: -len(x[0]))))
                peeled = keep_
                for oks, rest_ in res_:
                    unrolled += oks
                    peeled += rest_
                st["peel_restretched"] = len(redo)
            st["t_peel"] = time.perf_counter() - _t1
            log(f"parts peeled: {st['parts_peeled']}, shattered {len(shattered)} ({st['t_peel']:.1f}s)")
            whole_or_peeled = set(whole) | handled | (set(by_part) - set(shattered))
        else:
            whole_or_peeled = set(whole) | handled
    else:
        whole_or_peeled = set()
    if whole_or_peeled:
        done = np.isin(part, list(whole_or_peeled))
        atoms = atoms.copy()
        atoms[done] = -1
    lab, cuv, hst = H.run(atoms) if (atoms >= 0).any() else (atoms.copy(), H.cuv, {})
    unrolled_parts = {int(part[c.faces[0]]) for c in unrolled}
    if whole:
        lab = lab.copy()
        base = int(lab.max()) + 1
        wd = np.isin(part, list(whole - unrolled_parts))
        lab[wd] = base + part[wd]
        if unrolled_parts:
            lab[np.isin(part, list(unrolled_parts))] = -1
    st.update({"hier_" + k: v for k, v in hst.items()})
    if timings is not None:
        timings["hier"] = time.perf_counter() - t
        log(f"stage hier: {timings['hier']:.1f}s")
        vram_mark(timings, "hier")
    t = time.perf_counter()
    faces = np.nonzero(lab >= 0)[0]
    loc = -np.ones(m, np.int64); loc[faces] = np.arange(len(faces))
    if getattr(H, "part_cut", None) is not None:
        CV, nv = hier.corner_ids_cut(topo.F, topo.FF, topo.FE, faces, loc, lab, H.part_cut)
    else:
        CV, nv = hier.corner_ids(topo.F, topo.FF, faces, loc, lab)
    ids = CV.reshape(-1); cu_ = cuv[faces].reshape(-1, 2)
    cnt = np.bincount(ids, minlength=nv)
    U = np.stack([np.bincount(ids, cu_[:, k], nv) for k in range(2)], 1) / np.maximum(cnt, 1)[:, None]
    order = np.argsort(lab[faces], kind="stable")
    sl = lab[faces][order]
    bounds = np.r_[0, np.nonzero(sl[1:] != sl[:-1])[0] + 1, len(sl)] if len(sl) else np.zeros(1, np.int64)
    final = []
    suspect = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        rows = order[b0:b1]
        used, inv = np.unique(CV[rows], return_inverse=True)
        piece = (faces[rows], inv.reshape(-1, 3), U[used])
        if int(sl[b0]) in H.invalid:
            suspect.append(piece)
        else:
            final.append(FinalChart(piece[0], piece[1], piece[2], {"src": "hier"}))
    final += [FinalChart(np.asarray(f_, np.int64), T_, U_, {"src": "peel"}) for f_, T_, U_ in peeled]
    final += unrolled
    if suspect:
        pieces, fst = overlapfix.fix_overlaps(topo, suspect, N=Ns)
        st.update(fst)
        final += [FinalChart(np.asarray(f_, np.int64), T_, U_, {"src": "fix"}) for f_, T_, U_ in pieces]
    if cfg.get("soft_seams", True) and final:
        from .merge import merge_islands
        t_m = time.perf_counter()
        final = merge_islands(topo, final, FinalChart, costs, ef, cfg, part=part,
                              edge_conf=cfg.get("_part_edge_conf"), log=log, st=st)
        st["t_merge"] = time.perf_counter() - t_m
    if cfg.get("absorb_crumbs", True):
        from .crumbs import absorb_crumbs
        n0 = len(final)
        final, nab = absorb_crumbs(topo, final, FinalChart, part=part, max_tris=cfg.get("crumb_max_tris", 100))
        st["crumbs_absorbed"] = nab
        log(f"crumbs: {n0} -> {len(final)} islands")
    if timings is not None:
        timings["charts_out"] = time.perf_counter() - t
    return final


def chart_semantic(topo, comp, cfg, log, st, forced=None, preferred=None, seam_hide=1.0, timings=None):
    """High-poly path: plan parts and islands on a crease-preserving proxy
    (where GPU solves are instant and carving/fur noise is small), transfer the
    finished islands to the original triangles through topology-safe corner
    welding, relax everything at full resolution on the GPU in one batch and
    peel off only the remaining fine-detail folds/collisions."""
    from scipy.spatial import cKDTree
    m = len(topo.F)
    target = cfg.get("seg_proxy_faces", 150_000)
    if m <= 1.3 * target:
        return chart_hier(topo, comp, cfg, log, st, forced, preferred, seam_hide, timings)
    t = time.perf_counter()
    csize = np.bincount(comp, minlength=int(comp.max()) + 1)
    small_thr = max(2000, int(0.0015 * m))
    large = (csize[comp] >= small_thr) & ~topo.degenerate
    small = ~large & ~topo.degenerate
    fl = np.nonzero(large)[0]
    pv, pf = _decimate(topo.V, topo.F[fl], int(target * len(fl) / max(m, 1)))
    ptopo, pcomp = build_topology(pv.astype(np.float32), pf, use_gpu=False)
    st["proxy_faces"] = int(len(pf))
    if timings is not None:
        timings["proxy"] = time.perf_counter() - t
        log(f"stage proxy: {timings['proxy']:.1f}s ({len(pf)} faces)")
    # ---- parts + islands on the proxy
    t = time.perf_counter()
    pcfg = dict(cfg)
    pcfg["seg_proxy_faces"] = len(pf) + 1          # segment the proxy directly
    pcfg["highpoly_threshold"] = 10 ** 12           # proxy uses the low-poly atom size
    pst = {}
    ptim = {}
    pfinal = chart_hier(ptopo, pcomp, pcfg, log, pst, None, None, seam_hide, ptim)
    st["proxy"] = pst
    ppart = pcfg.get("_part")
    if timings is not None:
        timings.update({"proxy_" + k: v for k, v in ptim.items()})
        timings["proxy_charting"] = time.perf_counter() - t
        log(f"stage proxy charting: {timings['proxy_charting']:.1f}s ({len(pfinal)} proxy islands)")
        vram_mark(timings, "proxy_charting")
    # ---- nearest normal-compatible proxy face of every original triangle
    t = time.perf_counter()
    pc = ptopo.V[ptopo.F].mean(1)
    d, idx = cKDTree(pc).query(topo.V[topo.F[fl]].mean(1), k=6, workers=-1)
    st["t_knn"] = time.perf_counter() - t
    nd = (ptopo.face_normal[idx] * topo.face_normal[fl][:, None, :]).sum(2)
    ell = np.sqrt(ptopo.face_area.mean())
    pick = idx[np.arange(len(fl)), np.argmin(d / ell + 2.0 * (1.0 - nd), axis=1)]
    del d, idx, nd
    part_full = -np.ones(m, np.int64)
    if ppart is not None:
        part_full[fl] = ppart[pick]
    tr, fl_ok = transfer.transfer_charts(topo, fl, pick, ptopo, pfinal, log=log)
    if pcfg.get("_openness") is not None and len(pcfg["_openness"]) == len(ptopo.F):
        op_full = np.ones(m)                        # small pieces: treated as visible
        op_full[fl] = np.asarray(pcfg["_openness"])[pick]
        cfg["_openness"] = op_full
    if timings is not None:
        timings["transfer"] = time.perf_counter() - t
        log(f"stage transfer: {timings['transfer']:.1f}s ({len(tr)} pieces)")
        vram_mark(timings, "transfer")
    # ---- one GPU relaxation of every island at full resolution, then peel folds
    t = time.perf_counter()
    final = []
    if tr:
        items = [(topo.V[vw], T, U) for faces, T, U, vw in tr]
        b = gpu.Batch(items)
        b.orient_positive()
        b.repair_flips(cfg.get("gpu_repair_rounds", 80))
        it = cfg.get("fullres_iters", 4)
        if it:
            # a per-chart flip-safe step is limited by its single worst triangle, so
            # transfer-squashed detail never recovers: free ARAP first, then repair
            # and finish flip-safe
            free_it = cfg.get("fullres_free_iters", it)
            if free_it:
                b.arap(free_it, line_search=False)
                b.repair_flips(cfg.get("gpu_repair_rounds", 80))
            b.arap(max(2, it // 2), line_search=True)
            b.repair_flips(cfg.get("gpu_repair_rounds", 80))
        Us = b.chart_uvs()
        del b, items
        st["t_gpu_relax"] = time.perf_counter() - t
        if os.environ.get("UVA_DEBUG"):
            st["stretch_proxy"] = _stretch_stats(ptopo.V, ptopo.F, [(c.faces, c.T, c.U) for c in pfinal])
            bys = {}
            for c in pfinal:
                bys.setdefault(str(c.quality.get("src")), []).append(c)
            st["stretch_proxy_by_src"] = {k: dict(_stretch_stats(ptopo.V, ptopo.F, [(c.faces, c.T, c.U) for c in v]),
                                               charts=len(v), faces=int(sum(len(c.faces) for c in v))) for k, v in bys.items()}
            st["stretch_transfer"] = _stretch_stats(topo.V, topo.F, [(f_, T_, U_) for (f_, T_, U_, _) in tr])
            st["stretch_relax"] = _stretch_stats(topo.V, topo.F, [(f_, T_, U_) for (f_, T_, _, _), U_ in zip(tr, Us)])
            st["sq_after_relax"] = _squashed(topo, [(f_, T_, U_) for (f_, T_, _, _), U_ in zip(tr, Us)])
            st["sq_after_transfer"] = _squashed(topo, [(f_, T_, U_) for (f_, T_, U_, _) in tr])
        # boundary crossings at fine detail: untangle locally instead of splitting
        n_unt = 0
        for k, ((faces, T, _, vw), U) in enumerate(zip(tr, Us)):
            if ch.boundary_intersections(U, T, max_report=1):
                U2, ok_ = ch.untangle_boundary(U, T, rounds=cfg.get("untangle_rounds", 60))
                Us[k] = U2
                n_unt += int(ok_)
        st["fullres_untangled"] = n_unt
        st["t_untangle"] = time.perf_counter() - t - st["t_gpu_relax"]
        if os.environ.get("UVA_DEBUG"):
            st["sq_after_untangle"] = _squashed(topo, [(f_, T_, U_) for (f_, T_, _, _), U_ in zip(tr, Us)])
        if os.environ.get("UVA_DEBUG"):
            dbg = []
            for (faces, T, _, vw), U in zip(tr, Us):
                nf = int((param.signed_areas(U, T) <= 0).sum())
                bh = ch.boundary_intersections(U, T)
                dbg.append((len(faces), nf, bh, bool(ch.is_disk(T)[0])))
            st["fullres_dbg"] = dbg
        pieces, fst = overlapfix.fix_overlaps(topo, [(faces, T, U) for (faces, T, _, vw), U in zip(tr, Us)])
        st["fullres_peel_splits"] = fst.get("overlap_splits", 0)
        final = [FinalChart(np.asarray(f_, np.int64), T_, U_, {"src": "proxy"}) for f_, T_, U_ in pieces]
    if timings is not None:
        timings["fullres_relax"] = time.perf_counter() - t
        log(f"stage full-res relax + peel: {timings['fullres_relax']:.1f}s ({len(final)} islands)")
        vram_mark(timings, "fullres_relax")
    # ---- small disconnected pieces and faces the proxy lost: charted directly
    small = small.copy()
    small[np.setdiff1d(fl, fl_ok)] = True
    if small.any():
        t = time.perf_counter()
        sf = np.nonzero(small)[0]
        sused = np.unique(topo.F[sf])
        srem = -np.ones(len(topo.V), np.int64); srem[sused] = np.arange(len(sused))
        stopo, scomp = build_topology(topo.V[sused].astype(np.float32), srem[topo.F[sf]], use_gpu=False)
        scfg = dict(cfg, highpoly_threshold=10 ** 12, seg_proxy_faces=len(sf) + 1, _mask_fn=None)
        sst = {}
        sfin = chart_hier(stopo, scomp, scfg, log, sst, None, None, seam_hide, None)
        for fc in sfin:
            fc.faces = sf[fc.faces]
        final += sfin
        if scfg.get("_part") is not None:
            sp = scfg["_part"]
            part_full[sf] = np.where(sp >= 0, sp + int(part_full.max()) + 1, -1)
        st["small_piece_faces"] = int(len(sf))
        if timings is not None:
            timings["small_pieces"] = time.perf_counter() - t
    cfg["_part"] = part_full
    return final


def segment_parts_any(topo, cfg, log, costs, ef, mask_fn=None):
    """Part labels for every original triangle. Large meshes are segmented on a
    crease-preserving proxy (creases spread over many tiny triangles are
    invisible to per-edge measures) and the labels are mapped back."""
    from scipy.spatial import cKDTree
    from . import segment3d
    m = len(topo.F)
    target = cfg.get("seg_proxy_faces", 150_000)
    if m > 1.3 * target:
        ok = ~topo.degenerate
        pv, pf = _decimate(topo.V, topo.F[ok], target)
        ptopo, _ = build_topology(pv.astype(np.float32), pf, use_gpu=False)
    else:
        ptopo = topo
    pNs = features.smooth_normals(ptopo, iters=3)
    pef = features.edge_features(ptopo, None, None, None, None, smooth_N=pNs)
    if ptopo is topo:
        pef["hard"] = ef["hard"].copy()
    ppart, info = segment3d.segment_parts(ptopo, pNs, pef, cfg, log=log, mask_fn=mask_fn)
    if ptopo is not topo:
        cfg["_part_edge_conf"] = None        # confidence lives on the segmentation proxy's edges
    cfg["_seg_proxy"] = (ptopo, ppart)
    if ptopo is topo:
        return ppart, info
    pc = ptopo.V[ptopo.F].mean(1)
    d, idx = cKDTree(pc).query(topo.V[topo.F].mean(1), k=6, workers=-1)
    nd = (ptopo.face_normal[idx] * topo.face_normal[:, None, :]).sum(2)
    ell = np.sqrt(ptopo.face_area.mean())
    pick = idx[np.arange(m), np.argmin(d / ell + 2.0 * (1.0 - nd), axis=1)]
    part = ppart[pick]
    part[topo.degenerate] = -1
    part, _ = refine.smooth_labels(topo, part, costs, ef["hard"], passes=cfg.get("smooth_passes", 12))
    part = refine.fix_fragments(topo, part, np.zeros(len(topo.E), bool))
    info["proxy_faces"] = int(len(ptopo.F))
    info["parts_fullres"] = int(part.max()) + 1
    return part, info


def _fill_unlabelled(topo, lab, mask, rounds=50):
    lab = lab.copy()
    for _ in range(rounds):
        miss = np.nonzero(mask & (lab < 0))[0]
        if len(miss) == 0:
            break
        nb = topo.FF[miss]
        cand = np.where(nb >= 0, lab[np.maximum(nb, 0)], -1)
        best = cand.max(axis=1)
        lab[miss[best >= 0]] = best[best >= 0]
    # isolated leftovers become their own label
    miss = np.nonzero(mask & (lab < 0))[0]
    if len(miss):
        lab[miss] = lab.max() + 1 + np.arange(len(miss))
    return lab


# ------------------------------------------------------------------ main entry
@dataclass
class ChartSet:
    """Result of the charting stage: validated charts (UVs in 3D units) plus
    everything packing and assembly need. Packing can be re-run on it with any
    strategy without repeating segmentation or parameterization."""
    topo: object
    v: np.ndarray
    f: np.ndarray
    normals: object
    colors: object
    attr_arrays: object
    final: list
    cfg: dict
    st: dict
    timings: dict
    diag: dict
    profile: str
    mesh_type: str
    face_col: object = None
    openness: object = None
    seconds: float = 0.0


def unwrap(vertices, faces, normals=None, colors=None, uvs=None, face_material=None, face_labels=None,
           atlas=4096, padding=4, profile="balanced", hard_angle=None, mesh_type="auto",
           density="uniform", importance=None, seam_hide=1.0, log=print, overrides=None, attr_arrays=None,
           color_hints=True, check=None, semantic="auto", packer="auto"):
    kw = dict(normals=normals, colors=colors, face_material=face_material, face_labels=face_labels,
              hard_angle=hard_angle, mesh_type=mesh_type, seam_hide=seam_hide, log=log, overrides=overrides,
              attr_arrays=attr_arrays, color_hints=color_hints, check=check, semantic=semantic)
    cs = make_charts(vertices, faces, profile=profile, **kw)
    res = pack_charts(cs, atlas=atlas, padding=padding, density=density, importance=importance, log=log, packer=packer)
    return artist_candidate(res, vertices, faces, profile, cs, kw, atlas, padding, density, importance, log, packer)


def artist_candidate(res, vertices, faces, profile, cs, kw, atlas, padding, density, importance, log, packer):
    """Artist mode on low-poly meshes: also build the balanced layout and keep
    it when the artist layout (more islands, less stretch) packs to clearly
    lower texel density - measured bake quality follows texel density."""
    cfg = PROFILES.get(profile, {})
    if profile != "artist" or len(faces) > cfg.get("candidate_max_faces", 300_000):
        return res
    t = time.perf_counter()
    cs_b = make_charts(vertices, faces, profile="balanced", **kw)
    res_b = pack_charts(cs_b, atlas=atlas, padding=padding, density=density, importance=importance, log=log,
                        packer=packer)
    da = res.report["packing"].get("texels_per_unit_atlas", 0.0)
    db = res_b.report["packing"].get("texels_per_unit_atlas", 0.0)
    pick_b = db > da / cfg.get("candidate_density_ratio", 0.97)
    chosen = res_b if pick_b else res
    chosen.report["candidates"] = {"artist_texels_per_unit": da, "balanced_texels_per_unit": db,
                                   "chosen": "balanced" if pick_b else "artist",
                                   "seconds": time.perf_counter() - t}
    log(f"artist candidates: artist {da:.0f} vs balanced {db:.0f} texels/unit -> {'balanced' if pick_b else 'artist'}")
    return chosen


def make_charts(vertices, faces, normals=None, colors=None, face_material=None, face_labels=None,
                profile="balanced", hard_angle=None, mesh_type="auto", seam_hide=1.0, log=print, overrides=None,
                attr_arrays=None, color_hints=True, check=None, semantic="auto"):
    t_all = time.perf_counter()
    cfg = dict(PROFILES[profile]); cfg.update(overrides or {})
    timings = {}
    st = {}
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.int64)
    t = time.perf_counter()
    topo, comp = build_topology(v, f)
    timings["topology"] = time.perf_counter() - t
    diag = dict(topo.diag)
    m = len(f)
    if mesh_type == "auto":
        mesh_type = "lowpoly" if m <= cfg.get("highpoly_threshold", 2 * cfg["proxy_faces"]) else "highpoly"
    if mesh_type == "lowpoly":
        # low-poly: big, clear parts -> lighter SAM sampling (same masks, ~2x faster)
        cfg["seg_points"] = min(cfg.get("seg_points", 24), cfg.get("seg_points_lowpoly", 16))
        cfg["seg_views"] = min(cfg.get("seg_views", 16), cfg.get("seg_views_lowpoly", 12))
    forced = label_edges(topo, face_material)
    if normals is not None and mesh_type == "lowpoly":
        forced |= split_normal_edges(topo, f, normals)
    preferred = label_edges(topo, face_labels).astype(float) if face_labels is not None else None
    face_col = None
    if color_hints and colors is not None:
        face_col = np.asarray(colors, np.float64)[f].mean(1)
        cs = color_edge_strength(topo, face_col)
        preferred = cs if preferred is None else np.maximum(preferred, cs)
        diag["color_seam_edges"] = int((cs > 0.5).sum())
    diag["forced_seam_edges"] = int(forced.sum())
    cfg["_check"] = check
    cfg["_face_col"] = face_col
    # semantic parts: SAM 2 multi-view evidence when its checkpoint is installed
    mask_fn = None
    if semantic in ("auto", "sam2"):
        try:
            from . import sam_masks
            if sam_masks.find_checkpoint() is not None:
                mask_fn = sam_masks.SAM2Masks("base_plus", points_per_side=cfg.get("seg_points", 24))
        except Exception as exc:
            if semantic == "sam2":
                raise
            log(f"SAM 2 unavailable ({exc!r}); geometry-only parts")
    cfg["_mask_fn"] = mask_fn
    cfg["parts"] = semantic != "off"
    diag["semantic"] = "sam2+geometry" if mask_fn is not None else ("geometry" if semantic != "off" else "off")
    t = time.perf_counter()
    openness = None
    if mesh_type == "lowpoly" and cfg.get("lowpoly_method", "hier") == "hier":
        final = chart_hier(topo, comp, cfg, log, st, forced, preferred, seam_hide, timings)
        timings["charting_total"] = time.perf_counter() - t
    elif mesh_type == "lowpoly":
        ef, costs, Ns, openness = analyse(topo, cfg, forced, preferred, None, seam_hide, hard_angle)
        timings["features"] = time.perf_counter() - t
        t = time.perf_counter()
        charts = chart_direct(topo, ef, costs, Ns, cfg, log, st)
        timings["charting"] = time.perf_counter() - t
    elif cfg.get("highpoly_method", "semantic") == "semantic":
        final = chart_semantic(topo, comp, cfg, log, st, forced, preferred, seam_hide, timings)
        timings["charting_total"] = time.perf_counter() - t
    elif cfg.get("highpoly_method") == "hier":
        final = chart_hier(topo, comp, cfg, log, st, forced, preferred, seam_hide, timings)
        timings["charting_total"] = time.perf_counter() - t
    else:
        final = chart_highpoly(topo, comp, cfg, log, st, forced, preferred, seam_hide, timings)
        timings["charting_total"] = time.perf_counter() - t
    if cfg.get("_mask_fn") is not None:
        cfg["_mask_fn"].release()
        cfg["_mask_fn"] = None
    if mesh_type == "lowpoly" and cfg.get("lowpoly_method", "hier") != "hier":
        t = time.perf_counter()
        final = finalize(topo, charts, costs, cfg, log, st)
        timings["final_flatten"] = time.perf_counter() - t
    if os.environ.get("UVA_DEBUG"):
        st["sq_before_crumbs"] = _squashed(topo, [(c.faces, c.T, c.U) for c in final])
    if cfg.get("absorb_crumbs", True):
        from .crumbs import absorb_crumbs
        t = time.perf_counter()
        n0 = len(final)
        final, nab = absorb_crumbs(topo, final, FinalChart, part=cfg.get("_part"),
                                   max_tris=cfg.get("crumb_max_tris", 100), log=log)
        st["crumbs_absorbed"] = nab
        timings["crumbs"] = time.perf_counter() - t
        hist = {}
        for c in final:
            k = f'{c.quality.get("src", "?")}:{"tiny" if len(c.faces) <= 100 else ("small" if len(c.faces) <= 2000 else "big")}'
            hist[k] = hist.get(k, 0) + 1
        st["island_sources"] = hist
        if os.environ.get("UVA_DEBUG"):
            st["sq_after_crumbs"] = _squashed(topo, [(c.faces, c.T, c.U) for c in final])
        log(f"stage crumbs: {timings['crumbs']:.1f}s ({n0} -> {len(final)} islands)")
    final, nsh = shadow_duplicates(topo, final, stack=cfg.get("stack_duplicates", False))
    st["duplicate_copies_mapped"] = nsh
    cfg.pop("_check", None)
    return ChartSet(topo, v, f, normals, colors, attr_arrays, final, cfg, st, timings, diag, profile, mesh_type,
                    face_col=face_col, openness=cfg.get("_openness", openness), seconds=time.perf_counter() - t_all)


def pack_charts(cs, atlas=4096, padding=4, density="uniform", importance=None, log=print, packer="auto"):
    """Packing stage: texel-level checks, density, packing, atlas-resolution
    verification and assembly. Does not modify ``cs`` (packs a copy of the
    chart list), so several strategies can be compared on one ChartSet."""
    t_all = time.perf_counter() - cs.seconds
    topo, v, f, normals, colors, attr_arrays = cs.topo, cs.v, cs.f, cs.normals, cs.colors, cs.attr_arrays
    cfg, diag, profile, mesh_type = cs.cfg, cs.diag, cs.profile, cs.mesh_type
    st = dict(cs.st); timings = dict(cs.timings)
    face_col, openness = cs.face_col, cs.openness
    final = list(cs.final)
    # every island positively oriented (a mirrored island flips texture and tangent space)
    nmir = 0
    for i, c in enumerate(final):
        if float(param.signed_areas(c.U, c.T).sum()) < 0:
            final[i] = FinalChart(c.faces, c.T, c.U * np.array([-1.0, 1.0]), c.quality)
            nmir += 1
    st["mirrored_charts_fixed"] = nmir
    if cfg.get("density_splits", True):
        final, nds = _split_for_density(final, openness, cfg, log, face_normal=topo.face_normal, face_area=topo.face_area)
        st["density_splits"] = nds
    # ---- texel-level self-overlap check at the expected atlas density (before
    # packing, so fixing never needs a repack)
    if cfg.get("prepack_overlap_check", True) and final:
        t = time.perf_counter()
        tot = sum(float(np.abs(param.signed_areas(c.U, c.T)).sum()) for c in final)
        s_est = atlas * np.sqrt(0.85 / max(tot, 1e-30))       # expected texels per unit (the exact
        # check after packing, at atlas resolution, backs this up)
        out_, nfix = [], 0
        for c in final:
            ext = float((c.U.max(0) - c.U.min(0)).max()) if len(c.U) else 0.0
            res_ = int(np.clip(np.ceil(ext * s_est), 16, 6144))
            ov = overlapfix.overlapping_triangles(c.U, c.T, res=res_) if len(c.T) > 1 else np.zeros(len(c.T), bool)
            if len(c.T) > 1:
                ov = ov | validate.chart_overlaps(c.U, c.T)       # exact test (catches sub-texel overlaps)
            if not ov.any():
                out_.append(c)
                continue
            nfix += 1
            rows_ok = np.nonzero(~ov)[0]
            if len(rows_ok):
                out_ += [FinalChart(np.asarray(f_, np.int64), T_, U_, c.quality)
                         for f_, T_, U_ in overlapfix._pieces(c.faces, c.T, c.U, rows_ok)]
            for r in np.nonzero(ov)[0]:
                out_ += [FinalChart(np.asarray(f_, np.int64), T_, U_, {"src": "texel_fix"})
                         for f_, T_, U_ in overlapfix._pieces(c.faces, c.T, c.U, np.array([r]))]
        if nfix:
            final = out_
        st["prepack_overlap_charts"] = nfix
        timings["prepack_check"] = time.perf_counter() - t
    # ---- density
    dens = np.ones(len(final))
    if density == "adaptive":
        imp = (np.asarray(importance, float) if importance is not None
               else face_importance(topo, face_col, cfg.get("_openness", openness)))
        for i, c in enumerate(final):
            w = topo.face_area[c.faces]
            dens[i] = float(np.clip(np.sqrt((imp[c.faces] * w).sum() / max(w.sum(), 1e-30)), 0.62, 1.6))
    # ---- packing: candidate backends, the valid layout with the highest realised
    # texel density wins (chart shapes are identical in every candidate)
    t = time.perf_counter()
    uv_list, pinfo = _pack_candidates(final, dens, atlas, padding, cfg, log, packer, st)
    # ---- final guarantee: no island may overlap itself at atlas resolution
    if cfg.get("final_overlap_check", True):
        t2 = time.perf_counter()
        st["final_overlap_charts"] = 0
        for _round in range(3):
            bad = _self_overlapping(final, uv_list, min(atlas, 8192))
            st["final_overlap_charts"] += len(bad)
            if not bad:
                break
            log(f"final check: {len(bad)} island(s) overlap at atlas resolution -> split and repack")
            fixed = []
            for i in bad:
                c = final[i]
                # triangles that share a texel at (twice) atlas resolution
                ext = float((uv_list[i].max(0) - uv_list[i].min(0)).max())
                res_ = int(np.clip(np.ceil(ext * atlas * 2), 64, 16384))
                ov = overlapfix.overlapping_triangles(uv_list[i], c.T, res=res_)
                if not ov.any():
                    ov = overlapfix.overlapping_triangles(uv_list[i], c.T, res=min(2 * res_, 16384))
                rows_bad = np.nonzero(ov)[0]
                rows_ok = np.nonzero(~ov)[0]
                if len(rows_ok):
                    fixed += overlapfix._pieces(c.faces, c.T, c.U, rows_ok)
                for r in rows_bad:
                    fixed += overlapfix._pieces(c.faces, c.T, c.U, np.array([r]))
            keep = [c for i, c in enumerate(final) if i not in set(bad)]
            final = keep + [FinalChart(np.asarray(f_, np.int64), T_, U_, {"src": "final_fix"}) for f_, T_, U_ in fixed]
            dens = np.r_[dens[[i for i in range(len(dens)) if i not in set(bad)]], np.ones(len(fixed))]
            uv_list, pinfo = _pack_candidates(final, dens, atlas, padding, cfg, log, packer, st)
        timings["final_check"] = time.perf_counter() - t2
    timings["packing"] = time.perf_counter() - t
    # ---- assemble output
    t = time.perf_counter()
    attr_class = attribute_classes(v, normals, colors, attr_arrays)
    out_f, out_uv, xref, chart_of_face = _assemble(topo, f, final, uv_list, attr_class)
    out_v = v[xref]
    timings["assemble"] = time.perf_counter() - t
    # ---- exact validation of the exported (float32) UVs
    t = time.perf_counter()
    vrep, _ = validate.validate_uvs(out_v, out_f, out_uv, atlas=atlas, padding=padding, gutter=False)
    timings["validation"] = time.perf_counter() - t
    if not vrep["valid"]:
        log(f"VALIDATION: {vrep}")
    rep = {"engine": "Ultimate UV Artist", "profile": profile, "mesh_type": mesh_type, "atlas": atlas,
           "padding": padding, "charts": len(final), "topology": diag, "charting": st, "packing": pinfo,
           "timings": timings, "seconds_total": time.perf_counter() - t_all,
           "chart_src": [str(c.quality.get("src", "?")) for c in final], "validation": vrep}
    return UnwrapResult(out_v, out_f, out_uv, xref, chart_of_face, rep, cfg.get("_part"))


def vram_mark(timings, name):
    """Peak GPU memory (allocated, MiB) since the previous mark, stored as vram_<name>."""
    try:
        import torch
        if torch.cuda.is_available():
            timings["vram_" + name] = torch.cuda.max_memory_allocated() / 2 ** 20
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _stretch_stats(V, F, items):
    """Debug: area-weighted anisotropy p50/p95/p99 and stretched-area fraction of
    (faces, T, U) items, UVs in 3D units."""
    A_ = []; W_ = []
    for faces, T, U in items:
        P = V[F[faces]]
        e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]
        l1 = np.linalg.norm(e1, axis=1); a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        x2 = (e1 * e2).sum(1) / np.maximum(l1, 1e-30); y2 = 2 * a3 / np.maximum(l1, 1e-30)
        u = U[T]; d1 = u[:, 1] - u[:, 0]; d2 = u[:, 2] - u[:, 0]
        ok = (l1 > 1e-20) & (y2 > 1e-20)
        l1s = np.where(ok, l1, 1); y2s = np.where(ok, y2, 1)
        j00 = d1[:, 0] / l1s; j10 = d1[:, 1] / l1s
        j01 = (d2[:, 0] - j00 * x2) / y2s; j11 = (d2[:, 1] - j10 * x2) / y2s
        q = np.hypot(.5 * (j00 + j11), .5 * (j10 - j01)); r = np.hypot(.5 * (j00 - j11), .5 * (j10 + j01))
        s2 = np.abs(q - r); an = (q + r) / np.maximum(s2, 1e-300)
        ok &= s2 > 0
        A_.append(np.minimum(an[ok], 1e6)); W_.append(a3[ok])
    a = np.concatenate(A_); w = np.concatenate(W_)
    o = np.argsort(a); c = np.cumsum(w[o]) / w.sum()
    pick = lambda qq: round(float(a[o][min(np.searchsorted(c, qq), len(a) - 1)]), 3)
    return {"p50": pick(.5), "p95": pick(.95), "p99": pick(.99), "frac>1.3": round(float(w[a > 1.3].sum() / w.sum()), 4)}


def _squashed(topo, items):
    """Debug: triangles whose UV area is < 1/1000 of their 3D area (UV in 3D units)."""
    n = 0
    for faces, T, U in items:
        a2 = np.abs(param.signed_areas(U, T))
        n += int((a2 < 1e-3 * topo.face_area[faces]).sum())
    return n


def _split_for_density(final, openness, cfg, log, fill=0.80, max_splits=6, face_normal=None, face_area=None):
    """Two islands that are each wider AND taller than half the side of the
    square that would hold all UV area can never share a row or a column, so
    they cap the texel density of the whole atlas (e.g. a ground plane's top and
    underside). Cut the more hidden of them straight across its long axis
    (a subset of a valid layout stays valid: no re-flattening) until at most one
    such island is left."""
    final = list(final)
    tot = sum(float(np.abs(param.signed_areas(c.U, c.T)).sum()) for c in final)
    W = float(np.sqrt(tot / fill))
    n = 0
    for _ in range(max_splits):
        big = []
        for i, c in enumerate(final):
            V = pack.orient(c.U)
            e = V.max(0)
            if min(e) > 0.5 * W:
                vis = float(np.mean(openness[c.faces])) if openness is not None else 1.0
                if face_normal is not None:
                    # undersides (mostly facing down, Y-up) are rarely seen even when open to the sky below
                    w_ = face_area[c.faces]
                    down = float((w_ * (face_normal[c.faces, 1] < -0.7)).sum() / max(w_.sum(), 1e-30))
                    if down > 0.8:
                        vis = min(vis, 0.0)
                big.append((vis, -float(e.max()), i, V))
        if len(big) <= 1:
            break
        big.sort()                                  # most hidden first, then largest
        if big[0][0] > cfg.get("density_split_max_openness", 0.35):
            break                                   # never cut a visible island just to pack tighter
        _, _, i, V = big[0]
        c = final[i]
        cen = V[c.T].mean(1)
        ax = int(np.argmax(V.max(0)))
        cut = float(np.median(cen[:, ax]))
        rows_a = np.nonzero(cen[:, ax] <= cut)[0]; rows_b = np.nonzero(cen[:, ax] > cut)[0]
        if len(rows_a) == 0 or len(rows_b) == 0:
            break
        pieces = overlapfix._pieces(c.faces, c.T, c.U, rows_a) + overlapfix._pieces(c.faces, c.T, c.U, rows_b)
        final = final[:i] + final[i + 1:] + [FinalChart(np.asarray(f_, np.int64), T_, U_, dict(c.quality, density_split=True))
                                            for f_, T_, U_ in pieces]
        n += 1
    if n and log:
        log(f"density: {n} large island(s) split so they no longer cap the texel density")
    return final, n


def _texel_density(final, uv_list):
    a3 = sum(float(np.abs(param.signed_areas(c.U, c.T)).sum()) for c in final)
    a2 = sum(float(np.abs(param.signed_areas(uv, c.T)).sum()) for c, uv in zip(final, uv_list))
    return float(np.sqrt(a2 / max(a3, 1e-30)))


def _pack_candidates(final, dens, atlas, padding, cfg, log, packer, st):
    """Run the configured packers (see PROFILES 'packers'); return the valid
    candidate with the highest texel density. Validity = no texel shared by two
    triangles of one chart at atlas resolution (gutters are guaranteed by
    each backend's padding)."""
    order = cfg.get("packers", ("native",)) if packer == "auto" else (packer,)
    charts = [(c.U, c.T) for c in final]
    angles = cfg.get("pack_angles", (0.0,))
    orderings = cfg.get("orderings", ("area",))
    if len(charts) <= cfg.get("pack_orderings_small_max", 0):
        # few islands: trying every island ordering is cheap. (45-degree island
        # rotations were measured too: more occupancy, but worse baked texture on
        # organic meshes - islands lose their alignment with the texel grid.)
        orderings = ("area", "bbox", "height")
    cands = []
    for name in order:
        t0 = time.perf_counter()
        uvl, info = pack.pack(charts, atlas=atlas, padding=padding, grid=min(cfg["grid"], atlas),
                              rotations=cfg["rotations"], density=dens, iters=cfg["pack_iters"], log=log,
                              orderings=orderings,
                              # attempts get slower with island count: scale the budget (capped)
                              time_budget=min(cfg.get("pack_seconds", 20) + cfg.get("pack_seconds_per_chart", 0.005) * len(charts),
                                              cfg.get("pack_seconds_max", 3 * cfg.get("pack_seconds", 20))),
                              angles=angles, fine_attempts=cfg.get("pack_fine_attempts", 8))
        info = dict(info, backend=name, seconds=time.perf_counter() - t0)
        info["texels_per_unit_atlas"] = _texel_density(final, uvl) * atlas
        cands.append((info["texels_per_unit_atlas"], name, uvl, info))
    if not cands:
        raise RuntimeError("packing failed with every backend")
    cands.sort(key=lambda x: -x[0])
    dens_, name, uvl, info = cands[0]
    if log and len(cands) > 1:
        log("packers: " + ", ".join(f"{n} {d:.1f} tex/unit" for d, n, _, _ in cands) + f" -> {name}")
    return uvl, info


def _self_overlapping(final, uv_list, res):
    """Indices of charts that cover some texel centre twice at ``res``."""
    from .metrics import _raster_labels
    faces = []; lab = []; uvs = []; off = 0
    for i, (c, uv) in enumerate(zip(final, uv_list)):
        faces.append(c.T + off); lab.append(np.full(len(c.T), i, np.int32)); uvs.append(uv); off += len(uv)
    F = np.ascontiguousarray(np.concatenate(faces), np.int64)
    U = np.ascontiguousarray(np.concatenate(uvs), np.float64)
    L = np.concatenate(lab)
    cnt = np.zeros((res, res), np.uint8)
    from .metrics import _raster_counts
    _raster_counts(U, F, int(res), cnt)
    if not (cnt > 1).any():
        return []
    img = np.full((res, res), -1, np.int32)
    _raster_labels(U, F, L, int(res), img)
    o = img[cnt > 1]
    return sorted(set(int(x) for x in o[o >= 0]))


def shadow_duplicates(topo, final, stack=False):
    """UVs for exact duplicate triangle copies (excluded from charting).
    Default: the copies of each chart form their own island that repeats the
    primary layout (unique texels, nothing overlaps). ``stack=True`` puts the
    copies onto the primary UVs instead (intentional overlap)."""
    dup = topo.dup_of
    if dup is None or not (dup >= 0).any():
        return final, 0
    m = len(topo.F)
    chart_of = -np.ones(m, np.int64); row_of = -np.ones(m, np.int64)
    for ci, c in enumerate(final):
        chart_of[c.faces] = ci
        row_of[c.faces] = np.arange(len(c.faces))
    sec = np.nonzero(dup >= 0)[0]
    prim = dup[sec]
    ok = chart_of[prim] >= 0
    sec, prim = sec[ok], prim[ok]
    cid = chart_of[prim]
    # local vertex of every secondary corner = primary corner with the same welded vertex
    Tp = np.empty((len(sec), 3), np.int64)
    for i in range(len(sec)):
        c = final[cid[i]]
        rowT = c.T[row_of[prim[i]]]
        wp = topo.F[prim[i]]
        for k in range(3):
            Tp[i, k] = rowT[int(np.nonzero(wp == topo.F[sec[i], k])[0][0])]
    out = list(final)
    # k-th copy of a primary goes to the k-th shadow island of its chart
    o = np.argsort(prim, kind="stable")
    rank = np.empty(len(sec), np.int64)
    ps = prim[o]
    first = np.r_[True, ps[1:] != ps[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(ps)), 0))
    rank[o] = np.arange(len(ps)) - grp_start
    # same-winding and reversed copies never share an island (no mirrored triangles inside one)
    Fp = topo.F[prim]; Fs = topo.F[sec]
    k0 = np.argmax(Fp == Fs[:, :1], axis=1)
    same = Fp[np.arange(len(sec)), (k0 + 1) % 3] == Fs[:, 1]
    rank = rank * 2 + (~same).astype(np.int64)
    key = cid * (int(rank.max()) + 1) + rank
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1, len(ks)]
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        idx = order[b0:b1]
        if len(idx) == 0:
            continue
        ci = int(cid[idx[0]])
        c = out[ci] if stack else final[ci]
        if stack:
            out[ci] = FinalChart(np.r_[c.faces, sec[idx]], np.r_[c.T, Tp[idx]], c.U, c.quality)
        else:
            used, T = np.unique(Tp[idx], return_inverse=True)
            out.append(FinalChart(sec[idx], T.reshape(-1, 3), c.U[used].copy(), {"shadow_of": ci}))
    return out, int(len(sec))


def attribute_classes(vertices, normals=None, colors=None, extra=None):
    """Original vertices with identical non-UV attributes (position, normal,
    colour, extras) may share an output vertex; others never do."""
    cols = [np.asarray(vertices, np.float32)]
    for a in (normals, colors):
        if a is not None:
            cols.append(np.asarray(a, np.float32).reshape(len(vertices), -1))
    for a in (extra or []):
        cols.append(np.asarray(a, np.float32).reshape(len(vertices), -1))
    key = np.ascontiguousarray(np.concatenate(cols, 1))
    kv = key.view(np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).reshape(-1)
    _, first, inv = np.unique(kv, return_index=True, return_inverse=True)
    return first[inv.reshape(-1)]          # representative original vertex per vertex


def _assemble(topo, faces, final, uv_list, attr_class):
    m = len(faces)
    gl_corner = -np.ones((m, 3), np.int64)
    chart_of_face = -np.ones(m, np.int32)
    offs = 0
    uv_all = []
    for ci, (c, uv) in enumerate(zip(final, uv_list)):
        gl_corner[c.faces] = c.T + offs
        chart_of_face[c.faces] = ci
        uv_all.append(uv)
        offs += len(c.U)
    uv_all = np.concatenate(uv_all) if uv_all else np.zeros((0, 2))
    # degenerate / uncharted faces: borrow a UV of a charted corner sharing the welded vertex
    # (all three corners inside ONE chart: the chart's own corner at that vertex
    # if it has one, else collapsed onto the anchor corner -> zero-area, no sliver)
    miss = np.nonzero(gl_corner[:, 0] < 0)[0]
    if len(miss):
        nV = len(topo.V)
        ok = np.nonzero(gl_corner[:, 0] >= 0)[0]
        wv = topo.F[ok].ravel(); ch_ = np.repeat(chart_of_face[ok].astype(np.int64), 3); gc = gl_corner[ok].ravel()
        key = ch_ * nV + wv
        o = np.argsort(key, kind="stable"); key_s = key[o]; gc_s = gc[o]
        wv_to_gl = -np.ones(nV, np.int64); wv_to_gl[wv] = gc
        wv_to_ch = -np.ones(nV, np.int64); wv_to_ch[wv] = ch_
        for f_ in miss.tolist():
            vs = topo.F[f_]
            hit = [k for k in range(3) if wv_to_gl[vs[k]] >= 0]
            if not hit:
                gl_corner[f_] = 0
                continue
            a = vs[hit[0]]; ci = wv_to_ch[a]; anchor = wv_to_gl[a]
            for k in range(3):
                kk = ci * nV + vs[k]
                j = np.searchsorted(key_s, kk)
                gl_corner[f_, k] = gc_s[j] if j < len(key_s) and key_s[j] == kk else anchor
            u_ = uv_all[gl_corner[f_]]
            if abs((u_[1, 0] - u_[0, 0]) * (u_[2, 1] - u_[0, 1]) - (u_[2, 0] - u_[0, 0]) * (u_[1, 1] - u_[0, 1])) > 0:
                gl_corner[f_] = anchor          # zero 3D area must stay zero UV area
            chart_of_face[f_] = ci
    rep = attr_class[faces.reshape(-1)]
    nv = int(attr_class.max()) + 1
    key = gl_corner.reshape(-1) * nv + rep
    uk, inv = np.unique(key, return_inverse=True)
    out_f = inv.reshape(m, 3).astype(np.int32)
    xref = (uk % nv).astype(np.int64)
    gl = uk // nv
    out_uv = uv_all[gl].astype(np.float32)
    return out_f, out_uv, xref, chart_of_face
