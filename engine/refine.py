"""Island refinement after charting:

* boundary smoothing - greedy face relabelling that lowers the total seam cost
  (shorter seams, seams pulled into creases/hidden areas, no zig-zags);
* fragment repair - every island is one connected piece;
* validated merging - neighbouring islands are merged when the union still
  cuts to a disk and flattens inside the distortion budget.
"""
from __future__ import annotations

import heapq
import time

import numba as nb
import numpy as np

from . import charts as ch


@nb.njit(cache=True)
def _smooth(FF, FE, lab, cost, hard, count, passes, deg):
    m = FF.shape[0]
    moved_total = 0
    for p in range(passes):
        moved = 0
        for f in range(m):
            if deg[f]:
                continue
            a = lab[f]
            if a < 0 or count[a] <= 1:
                continue
            na = 0
            for k in range(3):
                g = FF[f, k]
                if g >= 0 and lab[g] == a:
                    na += 1
            if na == 3:
                continue
            best_b = -1
            best_d = -1e-12
            for kk in range(3):
                g0 = FF[f, kk]
                if g0 < 0:
                    continue
                b = lab[g0]
                if b == a or b < 0:
                    continue
                # hard edge between f and a b-neighbour would become interior
                bad = False
                d = 0.0
                for k in range(3):
                    g = FF[f, k]
                    if g < 0:
                        continue
                    e = FE[f, k]
                    lg = lab[g]
                    if lg == b and hard[e]:
                        bad = True
                        break
                    before = cost[e] if lg != a else 0.0
                    after = cost[e] if lg != b else 0.0
                    d += after - before
                if bad:
                    continue
                if d < best_d:
                    best_d = d
                    best_b = b
            if best_b >= 0 and na <= 2:
                count[a] -= 1
                count[best_b] += 1
                lab[f] = best_b
                moved += 1
        moved_total += moved
        if moved == 0:
            break
    return moved_total


@nb.njit(cache=True)
def _label_components(FF, lab, deg):
    m = FF.shape[0]
    comp = -np.ones(m, np.int64)
    stack = np.empty(m, np.int64)
    nc = 0
    for s in range(m):
        if comp[s] >= 0 or lab[s] < 0:
            continue
        comp[s] = nc
        top = 0
        stack[top] = s; top += 1
        while top > 0:
            top -= 1
            f = stack[top]
            for k in range(3):
                g = FF[f, k]
                if g >= 0 and comp[g] < 0 and lab[g] == lab[s]:
                    comp[g] = nc
                    stack[top] = g; top += 1
        nc += 1
    return comp, nc


def fix_fragments(topo, lab, hard, max_rounds=12):
    """Give every disconnected fragment of an island to its best neighbour
    (longest non-hard shared boundary); fragments that cannot join become
    islands. Fully vectorised (linear in the number of faces per round)."""
    lab = np.asarray(lab, np.int64).copy()
    m = len(topo.F)
    for _ in range(max_rounds):
        comp, nc = _label_components(topo.FF, lab, topo.degenerate)
        ok = comp >= 0
        if nc == 0:
            break
        size = np.bincount(comp[ok], weights=topo.face_area[ok], minlength=nc)
        comp_lab = np.full(nc, -1, np.int64)
        comp_lab[comp[ok]] = lab[ok]
        # keeper = largest component of each label
        o = np.lexsort((-size, comp_lab))
        first = np.r_[True, comp_lab[o][1:] != comp_lab[o][:-1]]
        frag = np.ones(nc, bool)
        frag[o[first]] = False
        if not frag.any():
            break
        # half-edges of fragment faces only (fragments are a small part of the mesh)
        ff = np.nonzero(ok & frag[np.maximum(comp, 0)])[0]
        f_ = np.repeat(ff, 3); g_ = topo.FF[ff].ravel(); e_ = topo.FE[ff].ravel()
        sel = g_ >= 0
        sel[sel] &= ok[g_[sel]]
        cf = comp[f_[sel]]; lg = lab[g_[sel]]; ee = e_[sel]
        cross = (lg != lab[f_[sel]]) & ~hard[ee]
        new_lab = comp_lab.copy()
        nl = int(lab.max()) + 1
        if cross.any():
            key = cf[cross] * (nl + 1) + lg[cross]
            uk, inv = np.unique(key, return_inverse=True)
            L = np.bincount(inv, weights=topo.edge_len[ee[cross]])
            kc = uk // (nl + 1); kl = uk % (nl + 1)
            oo = np.lexsort((-L, kc))
            best = np.r_[True, kc[oo][1:] != kc[oo][:-1]]
            new_lab[kc[oo][best]] = kl[oo][best]
        stuck = frag & (new_lab == comp_lab)
        # fragments with no joinable neighbour become islands of their own
        ids = np.nonzero(stuck)[0]
        new_lab[ids] = nl + np.arange(len(ids))
        lab_new = lab.copy()
        lab_new[ok] = new_lab[comp[ok]]
        if np.array_equal(lab_new, lab):
            break
        lab = lab_new
        if not cross.any():
            break
    # guarantee: every label is one connected piece (left-over fragments get new labels)
    comp, nc = _label_components(topo.FF, lab, topo.degenerate)
    ok = comp >= 0
    if nc:
        size = np.bincount(comp[ok], weights=topo.face_area[ok], minlength=nc)
        comp_lab = np.full(nc, -1, np.int64); comp_lab[comp[ok]] = lab[ok]
        o = np.lexsort((-size, comp_lab))
        first = np.r_[True, comp_lab[o][1:] != comp_lab[o][:-1]]
        extra = o[~first]
        if len(extra):
            new_lab = comp_lab.copy()
            new_lab[extra] = int(lab.max()) + 1 + np.arange(len(extra))
            lab[ok] = new_lab[comp[ok]]
    return _compact(lab)


def _compact(lab):
    out = lab.copy()
    ok = lab >= 0
    _, inv = np.unique(lab[ok], return_inverse=True)
    out[ok] = inv
    return out


def smooth_labels(topo, lab, costs, hard, passes=12):
    lab = np.ascontiguousarray(lab, np.int64).copy()
    n = int(lab.max()) + 1
    count = np.bincount(lab[lab >= 0], minlength=n).astype(np.int64)
    moved = _smooth(topo.FF, topo.FE, lab, costs, hard, count, passes, topo.degenerate)
    return lab, int(moved)


def labels_from_charts(m, charts_):
    lab = -np.ones(m, np.int64)
    for i, c in enumerate(charts_):
        lab[c.faces] = i
    return lab


def charts_from_labels(topo, lab, old_charts=None, old_lab=None):
    """Rebuild Chart objects; keep old cut edges that stay inside a chart."""
    order = np.argsort(lab, kind="stable")
    sl = lab[order]
    ok = sl >= 0
    order, sl = order[ok], sl[ok]
    n = int(sl.max()) + 1 if len(sl) else 0
    st = np.searchsorted(sl, np.arange(n + 1))
    out = []
    cut_all = set()
    if old_charts:
        for c in old_charts:
            cut_all |= c.cut
    cut_arr = np.array(sorted(cut_all), np.int64)
    for i in range(n):
        faces = order[st[i]:st[i + 1]]
        c = ch.Chart(topo, faces)
        if len(cut_arr):
            f0 = topo.EF[cut_arr, 0]; f1 = topo.EF[cut_arr, 1]
            inside = (f0 >= 0) & (f1 >= 0)
            inside[inside] = (lab[f0[inside]] == i) & (lab[f1[inside]] == i)
            c.cut = set(cut_arr[inside].tolist())
        out.append(c)
    return out


def chart_pairs(topo, lab, hard):
    f = np.repeat(np.arange(len(topo.F)), 3)
    g = topo.FF.ravel(); e = topo.FE.ravel()
    sel = g >= 0
    f, g, e = f[sel], g[sel], e[sel]
    a, b = lab[f], lab[g]
    s = (a >= 0) & (b >= 0) & (a < b)
    a, b, e = a[s], b[s], e[s]
    n = int(lab.max()) + 1
    key = a * n + b
    uk, inv = np.unique(key, return_inverse=True)
    L = np.bincount(inv, weights=topo.edge_len[e] * ~hard[e])
    Lh = np.bincount(inv, weights=topo.edge_len[e] * hard[e])
    return uk // n, uk % n, L, Lh


def merge_pass(topo, charts_, costs, hard, cfg, log=None, time_budget=60.0):
    """Greedy validated merging of neighbouring islands (smallest first)."""
    t0 = time.perf_counter()
    m = len(topo.F)
    lab = labels_from_charts(m, charts_)
    alive = {i: c for i, c in enumerate(charts_)}
    area = {i: float(topo.face_area[c.faces].sum()) for i, c in alive.items()}
    perim = {}
    a, b, L, Lh = chart_pairs(topo, lab, hard)
    nbr = {}
    for x, y, l in zip(a.tolist(), b.tolist(), L.tolist()):
        if l > 0:
            nbr.setdefault(x, {})[y] = l; nbr.setdefault(y, {})[x] = l
    # perimeter per chart
    f = np.repeat(np.arange(m), 3); g = topo.FF.ravel(); e = topo.FE.ravel()
    bnd = (g < 0) | (lab[np.maximum(g, 0)] != lab[f])
    per = np.bincount(lab[f][bnd & (lab[f] >= 0)], weights=topo.edge_len[e][bnd & (lab[f] >= 0)], minlength=len(charts_))
    for i in alive:
        perim[i] = float(per[i])
    heap = []

    def push(x, y):
        l = nbr[x][y]
        small = x if area[x] <= area[y] else y
        ratio = l / max(perim[small], 1e-30)
        if ratio < cfg.get("merge_min_contact", 0.12):
            return
        heapq.heappush(heap, (min(area[x], area[y]) / max(ratio, 1e-6), x, y))

    for x in nbr:
        for y in nbr[x]:
            if x < y:
                push(x, y)
    in_chart = np.zeros(m, np.bool_)
    attempts = merges = 0
    max_attempts = cfg.get("merge_attempts", 300)
    next_id = len(charts_)
    while heap and attempts < max_attempts and time.perf_counter() - t0 < time_budget:
        _, x, y = heapq.heappop(heap)
        if x not in alive or y not in alive or y not in nbr.get(x, {}):
            continue
        cx, cy = alive[x], alive[y]
        faces = np.concatenate([cx.faces, cy.faces])
        c = ch.Chart(topo, faces, cx.cut | cy.cut)
        in_chart[faces] = True
        attempts += 1
        ok = False
        try:
            if ch.cut_to_disk(c, costs, in_chart):
                P, T, vw = c.mesh(in_chart)
                U, q = ch.flatten(P, T, iters=cfg["plan_iters"])
                ok = (q["flips"] == 0 and q["overlap"] <= cfg["max_overlap"] and
                      q["aniso_p95"] <= cfg["max_aniso"] and q["area_p95"] <= cfg["max_area"])
                # merged seams must not get much longer than what they replace
                if ok:
                    newcut = sum(topo.edge_len[e] for e in (c.cut - cx.cut - cy.cut))
                    if newcut > 1.0 * nbr[x][y]:
                        ok = False
        except Exception:
            ok = False
        in_chart[faces] = False
        if not ok:
            # never retry this pair
            nbr[x].pop(y, None); nbr[y].pop(x, None)
            continue
        c.U, c.quality = U, q
        nid = next_id; next_id += 1
        alive.pop(x); alive.pop(y)
        alive[nid] = c
        area[nid] = area[x] + area[y]
        perim[nid] = perim[x] + perim[y] - 2 * nbr[x][y]
        nn = {}
        for src in (x, y):
            for z, l in nbr.pop(src, {}).items():
                if z in (x, y):
                    continue
                nn[z] = nn.get(z, 0.0) + l
                nbr[z].pop(src, None)
        nbr[nid] = nn
        for z, l in nn.items():
            nbr[z][nid] = l
            push(min(nid, z), max(nid, z)) if False else push(nid, z)
        merges += 1
    out = list(alive.values())
    if log:
        log(f"merge pass: {merges} merges / {attempts} attempts -> {len(out)} charts, {time.perf_counter() - t0:.1f}s")
    return out, {"merges": merges, "attempts": attempts}
