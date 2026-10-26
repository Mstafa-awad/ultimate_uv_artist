"""Soft semantic seams: merge neighbouring islands when the union is a valid,
low-stretch chart.

Semantic parts suggest seams, they do not force them. Every pair of adjacent
islands is a candidate; its priority is the seam length the merge would
remove, its penalty the crease strength along the shared border and - only
across part boundaries - the segmentation's confidence in that boundary.
Unions are flattened (warm-started from both layouts) and kept only if they
pass the same validation as any other chart (disk, no folds, exact boundary
test, stretch budget)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def part_edge_confidence(topo, part, sf, a, b, aff, thresh):
    """Per mesh edge: confidence (0..1) that a part boundary there is real.
    0 for edges inside a part. Derived from the superface-pair affinity used
    by the segmentation (affinity just under the merge threshold = weak)."""
    E = len(topo.E)
    conf = np.zeros(E)
    f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
    ok = (f0 >= 0) & (f1 >= 0)
    ok[ok] &= part[f0[ok]] != part[f1[ok]]
    if not ok.any():
        return conf
    n = int(sf.max()) + 1
    key = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
    o = np.argsort(key); ks = key[o]; av = np.asarray(aff, float)[o]
    sa, sb = sf[f0[ok]], sf[f1[ok]]
    q = np.minimum(sa, sb).astype(np.int64) * n + np.maximum(sa, sb)
    j = np.clip(np.searchsorted(ks, q), 0, len(ks) - 1)
    found = ks[j] == q
    w = np.where(found, av[j], 0.0)
    conf[np.nonzero(ok)[0]] = np.clip((thresh - w) / max(thresh, 1e-6), 0.0, 1.0) * found + (~found) * 1.0
    return conf


def merge_islands(topo, final, FinalChart, costs, ef, cfg, part=None, edge_conf=None, log=None, st=None):
    from . import grow
    m = len(topo.F)
    crease = np.asarray(ef.get("crease", np.zeros(len(topo.E))), float)
    hard = np.asarray(ef.get("hard", np.zeros(len(topo.E), bool)), bool)
    max_crease = cfg.get("merge_max_crease", 0.35)
    max_conf = cfg.get("merge_max_part_conf", 0.35)
    max_faces = cfg.get("merge_max_faces", 60_000)
    vcfg = dict(cfg, max_aniso=cfg.get("part_max_aniso", cfg["max_aniso"]),
                max_area=cfg.get("part_max_area", cfg["max_area"]), relief_cuts=0, allow_holes=True)
    nth = max(1, cfg.get("unroll_threads", min(8, os.cpu_count() or 1)))
    merged = tried = cross = 0
    for _round in range(cfg.get("merge_rounds", 4)):
        chart_of = -np.ones(m, np.int64)
        for ci, c in enumerate(final):
            chart_of[c.faces] = ci
        f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
        ok = (f0 >= 0) & (f1 >= 0)
        c0 = np.where(ok, chart_of[np.maximum(f0, 0)], -1); c1 = np.where(ok, chart_of[np.maximum(f1, 0)], -1)
        ok &= (c0 >= 0) & (c1 >= 0) & (c0 != c1)
        e = np.nonzero(ok)[0]
        if len(e) == 0:
            break
        a_, b_ = np.minimum(c0[e], c1[e]), np.maximum(c0[e], c1[e])
        key = a_ * len(final) + b_
        uk, inv = np.unique(key, return_inverse=True)
        L = np.bincount(inv, weights=topo.edge_len[e])
        Lc = np.bincount(inv, weights=topo.edge_len[e] * crease[e])
        Lh = np.bincount(inv, weights=topo.edge_len[e] * hard[e])
        Lp = np.bincount(inv, weights=topo.edge_len[e] * (edge_conf[e] if edge_conf is not None else 0.0))
        Xp = np.bincount(inv, weights=(part[f0[e]] != part[f1[e]]).astype(float) * topo.edge_len[e]) if part is not None \
            else np.zeros(len(uk))
        pa, pb = uk // len(final), uk % len(final)
        size = np.array([len(c.faces) for c in final])
        area = np.array([float(topo.face_area[c.faces].sum()) for c in final])
        mc, mp = Lc / np.maximum(L, 1e-30), Lp / np.maximum(Xp, 1e-30)
        # candidates: weak crease, no hard edge, weak semantic boundary, border long
        # relative to the smaller island (a merge must remove a meaningful seam)
        rel = L / np.sqrt(np.maximum(np.minimum(area[pa], area[pb]), 1e-30))
        cand = (mc <= max_crease) & (Lh <= 0) & ((Xp <= 0) | (mp <= max_conf)) & \
               (size[pa] + size[pb] <= max_faces) & (rel >= cfg.get("merge_min_rel_border", 0.6))
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            break
        score = rel[idx] * (1.0 - mc[idx]) * (1.0 - 0.5 * np.where(Xp[idx] > 0, mp[idx], 0.0))
        idx = idx[np.argsort(-score)]
        used = np.zeros(len(final), bool)
        batch = []
        for i in idx.tolist():
            x, y = int(pa[i]), int(pb[i])
            if used[x] or used[y]:
                continue
            used[x] = used[y] = True
            batch.append((x, y, bool(Xp[i] > 0)))
        import threading
        tls = threading.local()

        def test(xy):
            x, y, _ = xy
            if getattr(tls, "b", None) is None:
                tls.b = np.zeros(m, np.bool_)
            A, B = final[x], final[y]
            faces = np.r_[A.faces, B.faces]
            try:
                ok_, q_, U_, cc = grow.validate(topo, faces, set(), costs, vcfg, tls.b, want_uv=True,
                                                parts=((len(A.faces), A.T, A.U), (len(B.faces), B.T, B.U)))
            except Exception:
                return None
            if ok_ and U_ is not None and getattr(cc, "T_local", None) is not None:
                return FinalChart(np.asarray(faces, np.int64), cc.T_local, U_, dict(A.quality, merged=True))
            return None

        with ThreadPoolExecutor(max_workers=nth) as ex:
            res = list(ex.map(test, batch))
        tried += len(batch)
        alive = np.ones(len(final), bool)
        new = []
        n_ok = 0
        for (x, y, xp), r in zip(batch, res):
            if r is None:
                continue
            alive[x] = alive[y] = False
            new.append(r)
            n_ok += 1
            cross += int(xp)
        if n_ok == 0:
            break
        final = [c for i, c in enumerate(final) if alive[i]] + new
        merged += n_ok
    if st is not None:
        st["merge_tried"] = tried; st["merged"] = merged; st["merged_cross_part"] = cross
    if log:
        log(f"soft-seam merge: {merged} merges ({cross} across part boundaries) of {tried} tried")
    return final
