"""Bottom-up validated chart growing.

Starts from compact atoms and greedily merges neighbouring charts in an
artist-like priority order: smooth, visible boundaries and developable unions
merge first; creases, concave valleys and hidden lines stay seams. A merge is
accepted only when the union can be cut to a disk and flattened inside the
distortion budget without folds or overlaps, so every island is valid by
construction. Large charts are validated on a local window around the merge
and re-validated in full whenever they have grown substantially.
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from . import charts as ch


class _C:
    __slots__ = ("faces", "cut", "area", "perim", "N", "nf_full", "ver", "U", "q", "T", "kids")


def _dev(N, A):
    if A <= 0:
        return 0.0
    return float(max(np.linalg.eigvalsh(N / A)[0], 0.0))


def _good(q, cfg, scale=1.0):
    return (q["flips"] == 0 and q["overlap"] <= cfg["max_overlap"] and
            q["aniso_p95"] <= cfg["max_aniso"] * scale and q["area_p95"] <= cfg["max_area"] * scale)


def _warm_start(T, parts):
    """Initial UVs for a union chart from its parts' validated layouts: part B
    is rigidly aligned (Procrustes) onto part A over their shared vertices."""
    n = int(T.max()) + 1
    acc = []
    row = 0
    for nrows, Tp, Up in parts:
        vals = Up[Tp].reshape(-1, 2)
        ids = T[row:row + nrows].reshape(-1)
        s = np.zeros((n, 2)); c = np.zeros(n)
        np.add.at(s, ids, vals); np.add.at(c, ids, 1)
        acc.append((s, c))
        row += nrows
    (sa, ca), (sb, cb) = acc
    shared = (ca > 0) & (cb > 0)
    if shared.sum() < 2:
        return None
    A = sa[shared] / ca[shared][:, None]; B = sb[shared] / cb[shared][:, None]
    ma, mb = A.mean(0), B.mean(0)
    H = (B - mb).T @ (A - ma)
    u, _, vt = np.linalg.svd(H)
    R = (u @ vt).T
    if np.linalg.det(R) < 0:
        return None
    t = ma - mb @ R.T
    sb2 = sb @ R.T + cb[:, None] * t
    U = (sa + sb2) / np.maximum(ca + cb, 1)[:, None]
    if not ((ca + cb) > 0).all():
        return None
    return U


def validate(topo, faces, cut, costs, cfg, in_chart, want_uv=False, relief_budget=0.0, parts=None):
    """Cut to disk, flatten, check. If it fails on distortion, try up to
    ``cfg['relief_cuts']`` Seamster-style relief seams (worst vertex -> boundary
    along the cheapest hidden path) whose total length stays below
    ``relief_budget``. ``parts`` (rows, T, U of the two merged charts) enables a
    warm start from their layouts; the full LSCM solve runs only if it fails."""
    from . import param
    from .segment import _relief_path
    c = ch.Chart(topo, faces, cut)
    in_chart[faces] = True
    try:
        if cfg.get("allow_holes", False) and not cut:
            # discs with holes (genus 0, >= 1 boundary) flatten without seams
            P0, T0, vw0 = c.mesh(in_chart)
            chi, nl = ch.euler_and_loops(T0)
            if nl >= 1 and (2 - chi - nl) == 0 and nl > 1:
                try:
                    U0, q0 = ch.flatten(P0, T0, iters=cfg["plan_iters"])
                    if _good(q0, cfg):
                        from .overlapfix import raster_overlap
                        if raster_overlap(U0, T0) is None:
                            c.T_local = T0
                            return True, q0, (U0 if want_uv else None), c
                except Exception:
                    pass
        if not ch.cut_to_disk(c, costs, in_chart):
            return False, None, None, c
        used = 0.0
        for attempt in range(cfg.get("relief_cuts", 0) + 1):
            P, T, vw = c.mesh(in_chart)
            if attempt == 0 and parts is not None:
                U0 = _warm_start(T, parts)
                if U0 is not None:
                    X, area = param.local_frames(P, T)
                    cots = param.cot_weights(X)
                    U1, _ = param.refine(P, T, U0, iters=3, X=X, area=area, cots=cots, line_search=False, tol=1e-4)
                    if (param.signed_areas(U1, T) <= 0).any():
                        U1 = param.repair_flips(U1, T)
                    if not (param.signed_areas(U1, T) <= 0).any():
                        U1, _ = param.refine(P, T, U1, iters=2, X=X, area=area, cots=cots, line_search=True, tol=1e-4)
                        q1 = param.chart_quality(P, T, U1, X, area)
                        q1["boundary_hits"] = ch.boundary_intersections(U1, T)
                        q1["overlap"] = 1.0 if q1["boundary_hits"] else 0.0
                        if _good(q1, cfg) and ch.is_disk(T)[0]:
                            c.T_local = T
                            return True, q1, (U1 if want_uv else None), c
            U, q = ch.flatten(P, T, iters=cfg["plan_iters"])
            if _good(q, cfg) and ch.is_disk(T)[0]:
                c.T_local = T
                return True, q, (U if want_uv else None), c
            if attempt == cfg.get("relief_cuts", 0) or q["flips"] > 0 or q["overlap"] > 0:
                break
            path = _relief_path(topo, c, in_chart, P, T, vw, q, costs)
            if not path:
                break
            used += float(topo.edge_len[path].sum())
            if used > relief_budget:
                break
            c.cut.update(path)
        return False, q, None, c
    except Exception:
        return False, None, None, c
    finally:
        in_chart[faces] = False


def _window(topo, big_faces, small_faces, in_big, radius):
    """Faces of the big chart within ``radius`` face hops of the small chart."""
    seen = {}
    front = list(small_faces.tolist())
    for f in front:
        seen[f] = 0
    out = []
    d = 0
    while front and d < radius:
        nxt = []
        for f in front:
            for g in topo.FF[f]:
                g = int(g)
                if g >= 0 and in_big[g] and g not in seen:
                    seen[g] = d + 1
                    nxt.append(g); out.append(g)
        front = nxt
        d += 1
    return np.array(out, np.int64)


def grow_charts(topo, atoms, ef, costs, cfg, N=None, log=None):
    t0 = time.perf_counter()
    m = len(topo.F)
    Nf = topo.face_normal if N is None else N
    A = topo.face_area
    ok = atoms >= 0
    na = int(atoms.max()) + 1
    order = np.argsort(atoms, kind="stable")
    sa = atoms[order]; order = order[sa >= 0]; sa = sa[sa >= 0]
    st = np.searchsorted(sa, np.arange(na + 1))
    lab = atoms.copy()
    C = {}
    NN = np.zeros((na, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            v = np.bincount(atoms[ok], weights=(A * Nf[:, i] * Nf[:, j])[ok], minlength=na)
            NN[:, i, j] = v; NN[:, j, i] = v
    area = np.bincount(atoms[ok], weights=A[ok], minlength=na)
    # adjacency between atoms
    f = np.repeat(np.arange(m), 3); g = topo.FF.ravel(); e = topo.FE.ravel()
    sel = (g >= 0) & ok[f]
    sel[sel] &= ok[g[sel]]
    f, g, e = f[sel], g[sel], e[sel]
    la, lb = atoms[f], atoms[g]
    s2 = la < lb
    la, lb, e = la[s2], lb[s2], e[s2]
    key = la * na + lb
    uk, inv = np.unique(key, return_inverse=True)
    good_e = 1.0 - np.clip(costs[e] / np.maximum(topo.edge_len[e], 1e-30) - 0.06, 0, 1) ** 0.5
    L = np.bincount(inv, weights=topo.edge_len[e])
    Lg = np.bincount(inv, weights=topo.edge_len[e] * good_e)
    H = np.bincount(inv, weights=ef["hard"][e].astype(float)) > 0
    # perimeters
    per = np.zeros(na)
    ff = topo.FF.ravel(); fe = topo.FE.ravel(); fa = np.repeat(np.arange(m), 3)
    bnd = (ff < 0) | (atoms[np.maximum(ff, 0)] != atoms[fa])
    okb = bnd & (atoms[fa] >= 0)
    np.add.at(per, atoms[fa][okb], topo.edge_len[fe][okb])
    for i in range(na):
        c = _C()
        c.faces = order[st[i]:st[i + 1]]; c.cut = set(); c.area = float(area[i]); c.perim = float(per[i])
        c.N = NN[i]; c.nf_full = len(c.faces); c.ver = 0; c.U = None; c.q = None; c.T = None; c.kids = None
        C[i] = c
    nbr = {i: {} for i in range(na)}
    for k, l, lg, h in zip(uk.tolist(), L.tolist(), Lg.tolist(), H.tolist()):
        x, y = divmod(k, na)
        nbr[x][y] = [l, lg, h]; nbr[y][x] = [l, lg, h]
    tot_area = float(area.sum())
    heap = []
    failed = set()

    def prio(x, y):
        r = nbr[x][y]
        if r[2]:
            return None
        cx, cy = C[x], C[y]
        Aab = cx.area + cy.area
        small = min(cx.area, cy.area)
        g_ = r[1] / max(r[0], 1e-30)                    # 0..1, 1 = excellent seam
        contact = r[0] / max(min(cx.perim, cy.perim), 1e-30)
        d = _dev(cx.N + cy.N, Aab)
        pab = cx.perim + cy.perim - 2 * r[0]
        comp = pab * pab / (4 * np.pi * max(Aab, 1e-30))
        comp0 = max(cx.perim ** 2 / (4 * np.pi * max(cx.area, 1e-30)), cy.perim ** 2 / (4 * np.pi * max(cy.area, 1e-30)))
        p = (0.15 + g_) * (1 + 6 * d) * (1 + 0.3 * max(comp - comp0, 0)) / (0.2 + contact)
        return p * np.sqrt(small / tot_area)

    def push(x, y):
        if (min(x, y), max(x, y)) in failed:
            return
        p = prio(x, y)
        if p is not None:
            heapq.heappush(heap, (p, x, y, C[x].ver, C[y].ver))

    for x in range(na):
        for y in nbr[x]:
            if x < y:
                push(x, y)
    in_chart = np.zeros(m, np.bool_)
    in_big = np.zeros(m, np.bool_)
    stats = {"attempts": 0, "merges": 0, "window_checks": 0, "full_checks": 0, "full_fail": 0}
    next_id = na
    budget = cfg.get("grow_seconds", 120)
    local_lim = cfg.get("window_faces", 3000)
    while heap:
        if time.perf_counter() - t0 > budget:
            break
        p, x, y, vx, vy = heapq.heappop(heap)
        if cfg.get("_check") and stats["attempts"] % 64 == 0:
            cfg["_check"]()
        if x not in C or y not in C or C[x].ver != vx or C[y].ver != vy or y not in nbr[x]:
            continue
        cx, cy = C[x], C[y]
        big, small = (cx, cy) if len(cx.faces) >= len(cy.faces) else (cy, cx)
        faces = np.concatenate([cx.faces, cy.faces])
        cut = cx.cut | cy.cut
        stats["attempts"] += 1
        use_window = len(big.faces) > local_lim and len(big.faces) > 4 * len(small.faces) and \
            len(faces) < 1.3 * big.nf_full
        if use_window:
            in_big[big.faces] = True
            radius = int(max(4, 2.5 * np.sqrt(len(small.faces))))
            w = _window(topo, big.faces, small.faces, in_big, radius)
            in_big[big.faces] = False
            wf = np.concatenate([small.faces, w])
            wcut = {e for e in cut}
            Uv = None
            ok_, q, _, cc = validate(topo, wf, wcut, costs, cfg, in_chart, relief_budget=cfg.get("relief_ratio", 0.8) * nbr[x][y][0])
            stats["window_checks"] += 1
            new_cut = None
        else:
            parts = None
            if cx.U is not None and cy.U is not None and cx.T is not None and cy.T is not None and cfg.get("warm_start", True):
                parts = [(len(cx.faces), cx.T, cx.U), (len(cy.faces), cy.T, cy.U)]
            ok_, q, Uv, cc = validate(topo, faces, cut, costs, cfg, in_chart, want_uv=True,
                                      relief_budget=cfg.get("relief_ratio", 0.8) * nbr[x][y][0], parts=parts)
            stats["full_checks"] += 1
            new_cut = cc.cut
        if not ok_:
            failed.add((min(x, y), max(x, y)))
            if q is None:
                stats["fail_disk"] = stats.get("fail_disk", 0) + 1
            else:
                for key_, bad in (("fail_flip", q["flips"] > 0), ("fail_overlap", q["overlap"] > cfg["max_overlap"]),
                                  ("fail_aniso", q["aniso_p95"] > cfg["max_aniso"]), ("fail_area", q["area_p95"] > cfg["max_area"])):
                    if bad:
                        stats[key_] = stats.get(key_, 0) + 1
            continue
        nid = next_id; next_id += 1
        c = _C()
        c.faces = faces; c.area = cx.area + cy.area; c.N = cx.N + cy.N
        c.perim = cx.perim + cy.perim - 2 * nbr[x][y][0]
        c.cut = set(new_cut) if new_cut is not None else set(cut)
        c.nf_full = len(faces) if not use_window else big.nf_full
        c.ver = 0; c.U = Uv; c.q = q if Uv is not None else None
        c.T = getattr(cc, "T_local", None) if Uv is not None else None
        c.kids = (cx, cy)           # merge history: undo point if this chart later fails
        # merged seams: cut edges must stay inside the chart; drop edges no longer internal
        nn = {}
        for src in (x, y):
            for z, r in nbr.pop(src).items():
                if z in (x, y):
                    continue
                cur = nn.get(z)
                if cur is None:
                    nn[z] = list(r)
                else:
                    cur[0] += r[0]; cur[1] += r[1]; cur[2] = cur[2] or r[2]
                nbr[z].pop(src, None)
        del C[x]; del C[y]
        C[nid] = c
        nbr[nid] = nn
        for z, r in nn.items():
            nbr[z][nid] = list(r)
        stats["merges"] += 1
        # periodic full validation of big charts grown through windows
        if use_window and len(c.faces) >= 1.3 * c.nf_full:
            ok_full, qf, Uf, cc = validate(topo, c.faces, c.cut, costs, cfg, in_chart, want_uv=True)
            stats["full_checks"] += 1
            if ok_full:
                c.cut = set(cc.cut); c.nf_full = len(c.faces); c.U = Uf; c.q = qf
                c.T = getattr(cc, "T_local", None)
            else:
                # roll back the last merge
                stats["full_fail"] += 1
                del C[nid]
                for z in nn:
                    nbr[z].pop(nid, None)
                del nbr[nid]
                C[x] = cx; C[y] = cy
                cx.ver += 1; cy.ver += 1
                nbr[x] = {}; nbr[y] = {}
                _rebuild_pair_adj(topo, C, nbr, ef, costs, [x, y])
                failed.add((min(x, y), max(x, y)))
                for z in list(nbr[x]):
                    push(x, z)
                for z in list(nbr[y]):
                    push(y, z)
                continue
        for z in nn:
            push(nid, z)
    # final full validation of window-grown charts (cut sets refreshed)
    out = []
    for cid, c in C.items():
        out.append(as_chart(topo, c))
    stats["seconds"] = time.perf_counter() - t0
    stats["charts"] = len(out)
    if log:
        log(f"grow: {len(out)} charts {stats}")
    return out, stats


def as_chart(topo, c):
    """Grown node -> Chart; ``kids`` stays lazy (see ``kids_of``)."""
    cc = ch.Chart(topo, c.faces, c.cut)
    cc.U = c.U; cc.quality = c.q; cc.T_local = c.T
    cc.node = c
    return cc


def kids_of(topo, chart):
    """The two validated charts a grown chart was merged from (or None)."""
    node = getattr(chart, "node", None)
    if node is None or node.kids is None:
        return None
    return [as_chart(topo, k) for k in node.kids]


def _rebuild_pair_adj(topo, C, nbr, ef, costs, ids):
    """Recompute adjacency of charts ``ids`` against all live charts."""
    m = len(topo.F)
    lab = -np.ones(m, np.int64)
    for cid, c in C.items():
        lab[c.faces] = cid
    for x in ids:
        fx = C[x].faces
        g = topo.FF[fx].ravel(); e = topo.FE[fx].ravel()
        ok = g >= 0
        g, e = g[ok], e[ok]
        lz = lab[g]
        sel = (lz >= 0) & (lz != x)
        good_e = 1.0 - np.clip(costs[e[sel]] / np.maximum(topo.edge_len[e[sel]], 1e-30) - 0.06, 0, 1) ** 0.5
        for z, l, lg, h in zip(lz[sel].tolist(), topo.edge_len[e[sel]].tolist(), (topo.edge_len[e[sel]] * good_e).tolist(),
                               ef["hard"][e[sel]].tolist()):
            r = nbr[x].setdefault(z, [0.0, 0.0, False])
            r[0] += l; r[1] += lg; r[2] = r[2] or h
        for z, r in nbr[x].items():
            nbr[z][x] = list(r)
