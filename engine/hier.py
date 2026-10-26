"""Full-resolution, GPU-parallel hierarchical charting.

1. Atoms (compact face regions) are flattened all at once on the GPU
   (planar init + ARAP) and validated.
2. Merge levels: neighbouring charts are paired by a seam-aware priority
   (greedy matching, every chart merges at most once per level). Each union
   is warm-started by rigidly aligning the two children's layouts on their
   shared boundary, all unions are relaxed together on the GPU, and each is
   validated on the real triangles: no folded triangle, no boundary
   self-intersection (exact), distortion inside the budget. Valid unions are
   kept, invalid ones fall back to their children.

Charts are stored as a face label plus per-corner UVs, so accepting or
rejecting a union is a cheap array write.
"""
from __future__ import annotations

import time

import numba as nb
import numpy as np
import torch
from .gpu import csr32, spmm

from .gpu import device


# ------------------------------------------------------------------ numba kernels
@nb.njit(cache=True)
def _find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        nx = p[x]; p[x] = r; x = nx
    return r


@nb.njit(cache=True)
def corner_ids(F, FF, faces, loc, lab):
    """Chart-local vertex id of every corner of ``faces``: corners of the same
    welded vertex are united across manifold edges whose faces share a label."""
    n = faces.shape[0]
    p = np.arange(3 * n)
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            j = loc[g]
            if j < 0 or j <= i or lab[faces[j]] != lab[f]:
                continue
            a = F[f, k]; b = F[f, (k + 1) % 3]
            for kk in range(3):
                if F[g, kk] == a:
                    ra = _find(p, 3 * i + k); rb = _find(p, 3 * j + kk)
                    if ra != rb:
                        p[rb] = ra
                elif F[g, kk] == b:
                    ra = _find(p, 3 * i + (k + 1) % 3); rb = _find(p, 3 * j + kk)
                    if ra != rb:
                        p[rb] = ra
    remap = -np.ones(3 * n, np.int64)
    CV = np.empty((n, 3), np.int64)
    nv = 0
    for i in range(n):
        for k in range(3):
            r = _find(p, 3 * i + k)
            if remap[r] < 0:
                remap[r] = nv; nv += 1
            CV[i, k] = remap[r]
    return CV, nv


@nb.njit(cache=True)
def chart_topology(CV, chart, nchart, nv):
    """Euler characteristic and boundary-loop count of every chart."""
    n = CV.shape[0]
    keys = np.empty(3 * n, np.int64)
    for i in range(n):
        for k in range(3):
            a = CV[i, k]; b = CV[i, (k + 1) % 3]
            lo = a if a < b else b; hi = b if a < b else a
            keys[3 * i + k] = lo * nv + hi
    order = np.argsort(keys)
    vchart = -np.ones(nv, np.int64)
    for i in range(n):
        for k in range(3):
            vchart[CV[i, k]] = chart[i]
    V = np.zeros(nchart, np.int64); E = np.zeros(nchart, np.int64); Fc = np.zeros(nchart, np.int64)
    for v in range(nv):
        if vchart[v] >= 0:
            V[vchart[v]] += 1
    for i in range(n):
        Fc[chart[i]] += 1
    p = np.arange(nv)
    isb = np.zeros(nv, np.bool_)
    q = 0
    while q < 3 * n:
        r = q
        while r + 1 < 3 * n and keys[order[r + 1]] == keys[order[q]]:
            r += 1
        kk = keys[order[q]]
        a = kk // nv; b = kk % nv
        E[vchart[a]] += 1
        if r == q:
            isb[a] = True; isb[b] = True
            ra = _find(p, a); rb = _find(p, b)
            if ra != rb:
                p[rb] = ra
        q = r + 1
    loops = np.zeros(nchart, np.int64)
    for v in range(nv):
        if isb[v] and _find(p, v) == v:
            loops[vchart[v]] += 1
    return V - E + Fc, loops


@nb.njit(cache=True)
def corner_ids_cut(F, FF, FE, faces, loc, lab, cut):
    """corner_ids that also keeps corners apart across ``cut`` edges (seams)."""
    n = faces.shape[0]
    p = np.arange(3 * n)
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            if g < 0 or cut[FE[f, k]]:
                continue
            j = loc[g]
            if j < 0 or j <= i or lab[faces[j]] != lab[f]:
                continue
            a = F[f, k]; b = F[f, (k + 1) % 3]
            for kk in range(3):
                if F[g, kk] == a:
                    ra = _find(p, 3 * i + k); rb = _find(p, 3 * j + kk)
                    if ra != rb:
                        p[rb] = ra
                elif F[g, kk] == b:
                    ra = _find(p, 3 * i + (k + 1) % 3); rb = _find(p, 3 * j + kk)
                    if ra != rb:
                        p[rb] = ra
    remap = -np.ones(3 * n, np.int64)
    CV = np.empty((n, 3), np.int64)
    nv = 0
    for i in range(n):
        for k in range(3):
            r = _find(p, 3 * i + k)
            if remap[r] < 0:
                remap[r] = nv; nv += 1
            CV[i, k] = remap[r]
    return CV, nv


@nb.njit(cache=True)
def boundary_directed(F, FF, faces, loc, chart, CV):
    """Directed boundary edges (local vertex ids) of all charts."""
    n = faces.shape[0]
    ba = np.empty(3 * n, np.int64); bb = np.empty(3 * n, np.int64)
    m = 0
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            b = True
            if g >= 0:
                j = loc[g]
                if j >= 0 and chart[j] == chart[i]:
                    a = F[f, k]
                    for kk in range(3):
                        if F[g, kk] == a:
                            if CV[j, kk] == CV[i, k]:
                                b = False
                            break
            if b:
                ba[m] = CV[i, k]; bb[m] = CV[i, (k + 1) % 3]; m += 1
    return ba[:m].copy(), bb[:m].copy()


@nb.njit(cache=True)
def charts_collide(CV, UV, chart, nchart, nv):
    """Per chart: True if its UV boundary self-intersects (exact segment test)."""
    n = CV.shape[0]
    keys = np.empty(3 * n, np.int64)
    for i in range(n):
        for k in range(3):
            keys[3 * i + k] = CV[i, k] * nv + CV[i, (k + 1) % 3]
    srt = np.sort(keys)
    ba = np.empty(3 * n, np.int64); bb = np.empty(3 * n, np.int64); bc = np.empty(3 * n, np.int64)
    m = 0
    for i in range(n):
        for k in range(3):
            a = CV[i, k]; b = CV[i, (k + 1) % 3]
            rev = b * nv + a
            pos = np.searchsorted(srt, rev)
            if pos < srt.shape[0] and srt[pos] == rev:
                continue
            ba[m] = a; bb[m] = b; bc[m] = chart[i]; m += 1
    order = np.argsort(bc[:m], kind="mergesort")
    out = np.zeros(nchart, np.bool_)
    s = 0
    while s < m:
        c = bc[order[s]]
        e = s
        while e < m and bc[order[e]] == c:
            e += 1
        cnt = e - s
        if cnt >= 3:
            A = np.empty((cnt, 2)); B = np.empty((cnt, 2))
            lo0 = 1e300; lo1 = 1e300; hi0 = -1e300; hi1 = -1e300; Ls = 0.0
            for q in range(cnt):
                ii = order[s + q]
                A[q, 0] = UV[ba[ii], 0]; A[q, 1] = UV[ba[ii], 1]
                B[q, 0] = UV[bb[ii], 0]; B[q, 1] = UV[bb[ii], 1]
                lo0 = min(lo0, min(A[q, 0], B[q, 0])); lo1 = min(lo1, min(A[q, 1], B[q, 1]))
                hi0 = max(hi0, max(A[q, 0], B[q, 0])); hi1 = max(hi1, max(A[q, 1], B[q, 1]))
                Ls += np.sqrt((A[q, 0] - B[q, 0]) ** 2 + (A[q, 1] - B[q, 1]) ** 2)
            ext = max(hi0 - lo0, hi1 - lo1)
            cell = max(4.0 * Ls / cnt, ext / 1024.0, 1e-30)
            nx = int((hi0 - lo0) / cell) + 1; ny = int((hi1 - lo1) / cell) + 1
            out[c] = _hit(A, B, cell, lo0, lo1, nx, ny)
        s = e
    return out


@nb.njit(cache=True)
def _hit(A, B, cell, lo0, lo1, nx, ny):
    n = A.shape[0]
    cnt = np.zeros(nx * ny + 1, np.int64)
    for i in range(n):
        x0 = int((min(A[i, 0], B[i, 0]) - lo0) / cell); x1 = int((max(A[i, 0], B[i, 0]) - lo0) / cell)
        y0 = int((min(A[i, 1], B[i, 1]) - lo1) / cell); y1 = int((max(A[i, 1], B[i, 1]) - lo1) / cell)
        for gx in range(max(x0, 0), min(x1, nx - 1) + 1):
            for gy in range(max(y0, 0), min(y1, ny - 1) + 1):
                cnt[gy * nx + gx + 1] += 1
    for k in range(1, cnt.shape[0]):
        cnt[k] += cnt[k - 1]
    items = np.empty(cnt[-1], np.int64)
    fill = cnt[:-1].copy()
    for i in range(n):
        x0 = int((min(A[i, 0], B[i, 0]) - lo0) / cell); x1 = int((max(A[i, 0], B[i, 0]) - lo0) / cell)
        y0 = int((min(A[i, 1], B[i, 1]) - lo1) / cell); y1 = int((max(A[i, 1], B[i, 1]) - lo1) / cell)
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
                    return True
    return False


@nb.njit(cache=True, parallel=True)
def chart_checks(F, FF, faces, loc, chart, CV, UV, nchart, nv):
    """Linear-time per-chart checks. Boundary edges come from face adjacency
    (an edge is interior only if the neighbour is in the same chart and shares
    the corner vertices). Returns (collide, chi, loops)."""
    n = faces.shape[0]
    isb = np.zeros((n, 3), np.bool_)
    nb_c = np.zeros(nchart, np.int64)
    for i in range(n):
        f = faces[i]
        for k in range(3):
            g = FF[f, k]
            b = True
            if g >= 0:
                j = loc[g]
                if j >= 0 and chart[j] == chart[i]:
                    a = F[f, k]
                    for kk in range(3):
                        if F[g, kk] == a:
                            if CV[j, kk] == CV[i, k]:
                                b = False
                            break
            isb[i, k] = b
    # counts
    Fc = np.zeros(nchart, np.int64)
    for i in range(n):
        Fc[chart[i]] += 1
        for k in range(3):
            if isb[i, k]:
                nb_c[chart[i]] += 1
    vchart = -np.ones(nv, np.int64)
    for i in range(n):
        for k in range(3):
            vchart[CV[i, k]] = chart[i]
    Vc = np.zeros(nchart, np.int64)
    for v in range(nv):
        if vchart[v] >= 0:
            Vc[vchart[v]] += 1
    chi = Vc - (3 * Fc + nb_c) // 2 + Fc
    # boundary edges grouped by chart (counting sort)
    start = np.zeros(nchart + 1, np.int64)
    for c in range(nchart):
        start[c + 1] = start[c] + nb_c[c]
    fill = start[:-1].copy()
    ba = np.empty(start[-1], np.int64); bb = np.empty(start[-1], np.int64)
    for i in range(n):
        for k in range(3):
            if isb[i, k]:
                c = chart[i]
                ba[fill[c]] = CV[i, k]; bb[fill[c]] = CV[i, (k + 1) % 3]; fill[c] += 1
    # loops: union-find over boundary vertices
    p = np.arange(nv)
    for q in range(ba.shape[0]):
        ra = _find(p, ba[q]); rb = _find(p, bb[q])
        if ra != rb:
            p[rb] = ra
    loops = np.zeros(nchart, np.int64)
    seen = np.zeros(nv, np.bool_)
    for q in range(ba.shape[0]):
        for v in (ba[q], bb[q]):
            if not seen[v]:
                seen[v] = True
                if _find(p, v) == v:
                    loops[vchart[v]] += 1
    collide = np.zeros(nchart, np.bool_)
    for c in nb.prange(nchart):
        s0 = start[c]; cnt = start[c + 1] - s0
        if cnt < 3:
            continue
        A = np.empty((cnt, 2)); B = np.empty((cnt, 2))
        lo0 = 1e300; lo1 = 1e300; hi0 = -1e300; hi1 = -1e300; Ls = 0.0
        for q in range(cnt):
            a = ba[s0 + q]; b = bb[s0 + q]
            A[q, 0] = UV[a, 0]; A[q, 1] = UV[a, 1]; B[q, 0] = UV[b, 0]; B[q, 1] = UV[b, 1]
            lo0 = min(lo0, min(A[q, 0], B[q, 0])); lo1 = min(lo1, min(A[q, 1], B[q, 1]))
            hi0 = max(hi0, max(A[q, 0], B[q, 0])); hi1 = max(hi1, max(A[q, 1], B[q, 1]))
            Ls += np.sqrt((A[q, 0] - B[q, 0]) ** 2 + (A[q, 1] - B[q, 1]) ** 2)
        ext = max(hi0 - lo0, hi1 - lo1)
        if not (ext < 1e300):
            collide[c] = True
            continue
        cell = max(4.0 * Ls / cnt, ext / 1024.0, 1e-30)
        nx = int((hi0 - lo0) / cell) + 1; ny = int((hi1 - lo1) / cell) + 1
        collide[c] = _hit(A, B, cell, lo0, lo1, nx, ny)
    return collide, chi, loops


# ------------------------------------------------------------------ GPU relaxation of a face subset
class Solver:
    """Per-face isometric frames and cotangent weights for the whole mesh,
    computed once on the GPU; ``relax`` runs ARAP on any subset of charts."""

    def __init__(self, topo, dev=None):
        self.dev = dev or device()
        d = self.dev
        V = torch.from_numpy(topo.V).to(d, torch.float64)
        F = torch.from_numpy(topo.F).to(d)
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        e1 = p1 - p0; e2 = p2 - p0
        l1 = e1.norm(dim=1)
        nrm = torch.cross(e1, e2, dim=1)
        dbl = nrm.norm(dim=1)
        ok = (dbl > 1e-14 * (l1 * l1).clamp_min(1e-300)) & torch.isfinite(dbl) & (l1 > 0)
        ex = e1 / l1.clamp_min(1e-30)[:, None]
        ey = torch.cross(nrm / dbl.clamp_min(1e-30)[:, None], ex, dim=1)
        X = torch.zeros(len(F), 3, 2, device=d, dtype=torch.float64)
        X[:, 1, 0] = l1; X[:, 2, 0] = (e2 * ex).sum(1); X[:, 2, 1] = (e2 * ey).sum(1)
        X[~ok] = 0
        cots = []
        for k in range(3):
            i, j, o = (k + 1) % 3, (k + 2) % 3, k
            a = X[:, i] - X[:, o]; b = X[:, j] - X[:, o]
            cr = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).abs().clamp_min(1e-30)
            cots.append(((a * b).sum(1) / cr).clamp(-1e4, 1e4))
        w = 0.5 * torch.stack(cots, 1)
        w[~ok] = 0
        self.X = X.float(); self.w = w.float(); self.ok = ok
        self.area = (0.5 * dbl).float()
        self.x1 = X[:, 1, 0].clamp_min(1e-30).float(); self.x2 = X[:, 2, 0].float()
        y2 = X[:, 2, 1]
        self.y2 = torch.where(y2.abs() > 1e-30, y2, torch.full_like(y2, 1e-30)).float()
        del V, F, p0, p1, p2, e1, e2, nrm, X

    def relax(self, faces, CV, nv, chart, nchart, U0, iters=6, cg_iters=20, repair_rounds=24):
        """faces (n,) global ids, CV (n,3) local vertex ids, chart (n,) local chart
        ids, U0 (nv,2) initial UVs (numpy). Returns UVs (nv,2) numpy float64 and
        per-chart (flips, aniso_bad_frac, area_bad_frac, uv_scale)."""
        d = self.dev
        fi = torch.from_numpy(faces).to(d)
        T = torch.from_numpy(CV).to(d)
        ct = torch.from_numpy(chart).to(d)
        U = torch.from_numpy(U0).to(d, torch.float32)
        X = self.X[fi]; w = self.w[fi]; ok = self.ok[fi]
        x1, x2, y2 = self.x1[fi], self.x2[fi], self.y2[fi]
        cv = torch.zeros(nv, dtype=torch.int64, device=d)
        cv[T.reshape(-1)] = ct.repeat_interleave(3)
        # Laplacian with one soft pin per chart
        I = torch.cat([T[:, 1], T[:, 2], T[:, 0]]); J = torch.cat([T[:, 2], T[:, 0], T[:, 1]])
        W = torch.cat([w[:, 0], w[:, 1], w[:, 2]])
        pins = torch.full((nchart,), nv, dtype=torch.int64, device=d)
        pins.scatter_reduce_(0, cv, torch.arange(nv, device=d), reduce="amin")
        pins = pins[pins < nv]
        wpin = W.abs().max().clamp_min(1e-12)
        rows = torch.cat([I, J, I, J, pins]); cols = torch.cat([J, I, I, J, pins])
        vals = torch.cat([-W, -W, W, W, wpin.expand(len(pins))])
        L = csr32(torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (nv, nv)).coalesce())
        diag = torch.zeros(nv, device=d); m_ = rows == cols
        diag.index_add_(0, rows[m_], vals[m_])
        dinv = (1.0 / diag.clamp_min(1e-30))[:, None]

        def sa(Uv):
            a, b, c = Uv[T[:, 0]], Uv[T[:, 1]], Uv[T[:, 2]]
            return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

        # orient every chart positively
        s_ = torch.zeros(nchart, device=d)
        s_.index_add_(0, ct, torch.sign(sa(U)) * self.area[fi])
        U = torch.where((s_ < 0)[cv][:, None], U * torch.tensor([1.0, -1.0], device=d), U)
        # adjacency for flip repair
        A = torch.sparse_coo_tensor(torch.stack([torch.cat([I, J]), torch.cat([J, I])]),
                                    torch.ones(2 * len(I), device=d), (nv, nv)).coalesce()
        bnd = torch.zeros(nv, dtype=torch.bool, device=d)
        one = A.values() == 1
        bnd[A.indices()[0][one]] = True
        Ar = csr32(torch.sparse_coo_tensor(A.indices(), torch.ones_like(A.values()), (nv, nv)))
        deg = spmm(Ar, torch.ones(nv, 1, device=d)).squeeze(1).clamp_min(1)

        def repair(Uv, rounds):
            for _ in range(rounds):
                bad = (sa(Uv) < 0) & ok
                if not bool(bad.any()):
                    break
                mv = torch.zeros(nv, dtype=torch.bool, device=d)
                mv[T[bad].reshape(-1)] = True
                mv &= ~bnd
                Uv = torch.where(mv[:, None], spmm(Ar, Uv) / deg[:, None], Uv)
            return Uv

        def rhs(Uv):
            a0, a1, a2 = Uv[T[:, 0]], Uv[T[:, 1]], Uv[T[:, 2]]
            j00 = (a1[:, 0] - a0[:, 0]) / x1; j10 = (a1[:, 1] - a0[:, 1]) / x1
            j01 = ((a2[:, 0] - a0[:, 0]) - j00 * x2) / y2
            j11 = ((a2[:, 1] - a0[:, 1]) - j10 * x2) / y2
            th = torch.atan2(j10 - j01, j00 + j11)
            th = torch.where(ok & torch.isfinite(th), th, torch.zeros_like(th))
            c, s = torch.cos(th), torch.sin(th)
            b = torch.zeros(nv, 2, device=d)
            for k in range(3):
                i, j = (k + 1) % 3, (k + 2) % 3
                dx = X[:, i] - X[:, j]
                r = torch.stack([c * dx[:, 0] - s * dx[:, 1], s * dx[:, 0] + c * dx[:, 1]], 1) * w[:, k:k + 1]
                b.index_add_(0, T[:, i], r); b.index_add_(0, T[:, j], -r)
            b[pins] += wpin * Uv[pins]
            return b

        def cg(b, x):
            r = b - spmm(L, x)
            z = r * dinv; p = z.clone(); rz = (r * z).sum(0)
            for _ in range(cg_iters):
                Ap = spmm(L, p)
                al = rz / (p * Ap).sum(0).clamp_min(1e-30)
                x = x + p * al; r = r - Ap * al
                z = r * dinv; rz2 = (r * z).sum(0)
                p = z + p * (rz2 / rz.clamp_min(1e-30)); rz = rz2
            return x

        def max_step(Uv, D):
            u0, u1, u2 = Uv[T[:, 0]], Uv[T[:, 1]], Uv[T[:, 2]]
            d0, d1, d2 = D[T[:, 0]], D[T[:, 1]], D[T[:, 2]]
            ux1, uy1 = u1[:, 0] - u0[:, 0], u1[:, 1] - u0[:, 1]
            ux2, uy2 = u2[:, 0] - u0[:, 0], u2[:, 1] - u0[:, 1]
            dx1, dy1 = d1[:, 0] - d0[:, 0], d1[:, 1] - d0[:, 1]
            dx2, dy2 = d2[:, 0] - d0[:, 0], d2[:, 1] - d0[:, 1]
            C = ux1 * uy2 - ux2 * uy1
            Bq = ux1 * dy2 + dx1 * uy2 - ux2 * dy1 - dx2 * uy1
            Aq = dx1 * dy2 - dx2 * dy1
            big = torch.full_like(C, 1e30)
            lin = torch.where((Aq.abs() < 1e-30) & (Bq < 0), -C / torch.where(Bq != 0, Bq, torch.ones_like(Bq)), big)
            disc = Bq * Bq - 4 * Aq * C
            sq = disc.clamp_min(0).sqrt()
            A2 = torch.where(Aq.abs() < 1e-30, torch.ones_like(Aq), 2 * Aq)
            r1 = (-Bq - sq) / A2; r2 = (-Bq + sq) / A2
            quad = torch.minimum(torch.where(r1 > 0, r1, big), torch.where(r2 > 0, r2, big))
            quad = torch.where((disc >= 0) & (Aq.abs() >= 1e-30), quad, big)
            t = torch.where((C > 0) & ok, torch.minimum(lin, quad), big)
            out = torch.full((nchart,), 1e30, device=d)
            out.scatter_reduce_(0, ct, t, reduce="amin")
            return out

        U = repair(U, repair_rounds)
        for it in range(iters):
            Un = cg(rhs(U), U.clone())
            badv = ~torch.isfinite(Un).all(1)
            if bool(badv.any()):
                badc = torch.zeros(nchart, dtype=torch.bool, device=d); badc[cv[badv]] = True
                Un = torch.where(badc[cv][:, None], U, Un)
            flipped = (sa(U) < 0) & ok
            fl_c = torch.zeros(nchart, device=d); fl_c.index_add_(0, ct, flipped.float())
            D = Un - U
            step = torch.clamp(0.9 * max_step(U, D), max=1.0)
            step = torch.where(fl_c > 0, torch.ones_like(step), step)     # folded charts: plain ARAP step
            U = U + D * step[cv][:, None]
            if it == iters // 2:
                U = repair(U, repair_rounds)
        U = repair(U, repair_rounds)
        # quality per chart
        a0, a1, a2 = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
        j00 = (a1[:, 0] - a0[:, 0]) / x1; j10 = (a1[:, 1] - a0[:, 1]) / x1
        j01 = ((a2[:, 0] - a0[:, 0]) - j00 * x2) / y2
        j11 = ((a2[:, 1] - a0[:, 1]) - j10 * x2) / y2
        E = (j00 + j11) / 2; Fq = (j00 - j11) / 2; G = (j10 + j01) / 2; H = (j10 - j01) / 2
        Q = (E * E + H * H).sqrt(); R = (Fq * Fq + G * G).sqrt()
        s1 = Q + R; s2 = (Q - R).abs()
        ar3 = self.area[fi] * ok
        sgn = sa(U)
        uva = torch.zeros(nchart, device=d); uva.index_add_(0, ct, sgn.abs() * ok)
        a3 = torch.zeros(nchart, device=d); a3.index_add_(0, ct, ar3)
        scale = uva / a3.clamp_min(1e-30)
        aniso = s1 / s2.clamp_min(1e-12)
        arat = (s1 * s2) / scale[ct].clamp_min(1e-30)
        flips = torch.zeros(nchart, device=d); flips.index_add_(0, ct, ((sgn < 0) & ok).float())
        self._q = (aniso, arat, ar3, ct, nchart, a3)
        out = U.double().cpu().numpy()
        return out, flips.cpu().numpy(), scale.cpu().numpy()

    def lscm(self, faces, CV, nv, chart, nchart, U0, ba, bb, iters=600):
        """Free-boundary conformal map of every chart at once (GPU CG on the
        LSCM energy, two soft pins per chart taken from the initial layout)."""
        d = self.dev
        fi = torch.from_numpy(faces).to(d)
        T = torch.from_numpy(CV).to(d)
        w = self.w[fi]
        I = torch.cat([T[:, 1], T[:, 2], T[:, 0]]); J = torch.cat([T[:, 2], T[:, 0], T[:, 1]])
        Wl = torch.cat([w[:, 0], w[:, 1], w[:, 2]])
        bi = torch.from_numpy(ba).to(d); bj = torch.from_numpy(bb).to(d)
        hb = torch.full((len(bi),), 0.5, device=d)
        rows = torch.cat([I, J, I, J, I + nv, J + nv, I + nv, J + nv, bi, bj, bi + nv, bj + nv])
        cols = torch.cat([J, I, I, J, J + nv, I + nv, I + nv, J + nv, bj + nv, bi + nv, bj, bi])
        vals = torch.cat([-Wl, -Wl, Wl, Wl, -Wl, -Wl, Wl, Wl, -hb, hb, hb, -hb])
        # two pins per chart: boundary vertices extreme along the initial u axis
        U0t = torch.from_numpy(U0).to(d, torch.float32)
        cv = torch.zeros(nv, dtype=torch.int64, device=d)
        cv[T.reshape(-1)] = torch.from_numpy(chart).to(d).repeat_interleave(3)
        bv = torch.unique(bi)
        key = U0t[bv, 0]
        big = torch.full((nchart,), 1e30, device=d)
        mn = big.clone().scatter_reduce_(0, cv[bv], key, reduce="amin")
        mx = (-big).scatter_reduce_(0, cv[bv], key, reduce="amax")
        p0 = torch.full((nchart,), -1, dtype=torch.int64, device=d)
        p1 = p0.clone()
        is_mn = key == mn[cv[bv]]; is_mx = key == mx[cv[bv]]
        p0[cv[bv][is_mn]] = bv[is_mn]; p1[cv[bv][is_mx]] = bv[is_mx]
        pins = torch.cat([p0[p0 >= 0], p1[p1 >= 0]])
        wpin = Wl.abs().max().clamp_min(1e-12) * 10
        rows = torch.cat([rows, pins, pins + nv]); cols = torch.cat([cols, pins, pins + nv])
        vals = torch.cat([vals, wpin.expand(2 * len(pins))])
        H = csr32(torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (2 * nv, 2 * nv)).coalesce())
        diag = torch.zeros(2 * nv, device=d); mm_ = rows == cols
        diag.index_add_(0, rows[mm_], vals[mm_])
        dinv = 1.0 / diag.clamp_min(1e-30)
        b = torch.zeros(2 * nv, device=d)
        b[pins] += wpin * U0t[pins, 0]; b[pins + nv] += wpin * U0t[pins, 1]
        x = torch.cat([U0t[:, 0], U0t[:, 1]])
        r = b - spmm(H, x)
        z = r * dinv; p = z.clone(); rz = (r * z).sum()
        b2 = (b * b).sum().clamp_min(1e-30)
        for it in range(iters):
            Ap = spmm(H, p)
            al = rz / (p * Ap).sum().clamp_min(1e-30)
            x = x + al * p; r = r - al * Ap
            if it % 50 == 0 and float((r * r).sum()) < 1e-12 * float(b2):
                break
            z = r * dinv; rz2 = (r * z).sum()
            p = z + (rz2 / rz.clamp_min(1e-30)) * p; rz = rz2
        U = torch.stack([x[:nv], x[nv:]], 1)
        U = torch.where(torch.isfinite(U), U, U0t)
        return U.double().cpu().numpy()

    def bad_fraction(self, max_aniso, max_area):
        aniso, arat, ar3, ct, nchart, a3 = self._q
        badA = torch.zeros(nchart, device=self.dev)
        badA.index_add_(0, ct, ar3 * (aniso > max_aniso))
        badR = torch.zeros(nchart, device=self.dev)
        badR.index_add_(0, ct, ar3 * ((arat > max_area) | (arat < 1.0 / max_area)))
        return (badA / a3.clamp_min(1e-30)).cpu().numpy(), (badR / a3.clamp_min(1e-30)).cpu().numpy()


def _split_atom(topo, fa, N):
    """Seam-aware 2-way split of a face set using piece-local arrays only."""
    from .overlapfix import _split_local
    if not hasattr(topo, "_centroids"):
        topo._centroids = topo.V[topo.F].mean(1)
    if not hasattr(topo, "_loc_buf"):
        topo._loc_buf = -np.ones(len(topo.F), np.int64)
    C = topo._centroids[fa]
    c0 = C.mean(0)
    a = int(np.argmax(((C - c0) ** 2).sum(1)))
    b = int(np.argmax(((C - C[a]) ** 2).sum(1)))
    if a == b:
        return [fa]
    lab = _split_local(topo, fa, a, b, np.ascontiguousarray(N))
    parts = []
    for l in (0, 1):
        p_ = fa[lab == l]
        if len(p_):
            parts.append(p_)
    return parts


# ------------------------------------------------------------------ driver
def planar_init(topo, faces, CV, nv, chart, nchart):
    """Project every chart onto its least-squares plane (area-weighted normal)."""
    V = topo.V; F = topo.F
    P = np.zeros((nv, 3))
    P[CV.reshape(-1)] = V[F[faces].reshape(-1)]
    wn = topo.face_normal[faces] * topo.face_area[faces][:, None]
    n = np.stack([np.bincount(chart, wn[:, k], nchart) for k in range(3)], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    t = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t = t - n * (t * n).sum(1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-30)[:, None]
    b = np.cross(n, t)
    vc = np.zeros(nv, np.int64); vc[CV.reshape(-1)] = np.repeat(chart, 3)
    return np.stack([(P * t[vc]).sum(1), (P * b[vc]).sum(1)], 1)


class Hier:
    def __init__(self, topo, costs, ef, cfg, N=None, log=print, part=None):
        self.topo = topo; self.costs = costs; self.ef = ef; self.cfg = cfg; self.log = log
        self.part = part                         # per-face part label: islands never cross parts
        self.invalid = set()
        self.N = topo.face_normal if N is None else N
        self.solver = Solver(topo)
        m = len(topo.F)
        self.loc = -np.ones(m, np.int64)
        self.cuv = np.zeros((m, 3, 2))          # per-corner UVs of the current charts

    # -- one batched solve of a set of charts (labels in ``lab``)
    def solve(self, faces, lab, init_cuv=None, iters=6, cut=None, init="planar"):
        topo = self.topo
        loc = self.loc
        loc[faces] = np.arange(len(faces))
        try:
            if cut is None:
                CV, nv = corner_ids(topo.F, topo.FF, faces, loc, lab)
            else:
                CV, nv = corner_ids_cut(topo.F, topo.FF, topo.FE, faces, loc, lab, cut)
        finally:
            loc[faces] = -1
        ul, chart = np.unique(lab[faces], return_inverse=True)
        nchart = len(ul)
        if init_cuv is None:
            U0 = planar_init(topo, faces, CV, nv, chart, nchart)
            if init == "lscm":
                loc[faces] = np.arange(len(faces))
                try:
                    ba, bb = boundary_directed(topo.F, topo.FF, faces, loc, chart, CV)
                finally:
                    loc[faces] = -1
                U0 = self.solver.lscm(faces, CV, nv, chart, nchart, U0, ba, bb)
        else:
            ids = CV.reshape(-1); cu_ = init_cuv.reshape(-1, 2)
            cnt = np.bincount(ids, minlength=nv)
            U0 = np.stack([np.bincount(ids, cu_[:, k], nv) for k in range(2)], 1) / np.maximum(cnt, 1)[:, None]
        U, flips, scale = self.solver.relax(faces, CV, nv, chart, nchart, U0, iters=iters)
        bad_an, bad_ar = self.solver.bad_fraction(self.cfg["max_aniso"], self.cfg["max_area"])
        loc = self.loc
        loc[faces] = np.arange(len(faces))
        try:
            Uf = np.where(np.isfinite(U), U, 1e301)
            col, chi, loops = chart_checks(topo.F, topo.FF, faces, loc, chart, CV, Uf, nchart, nv)
        finally:
            loc[faces] = -1
        frac = self.cfg.get("bad_area_fraction", 0.05)
        # per chart validity (U finite per chart)
        fin = np.ones(nchart, bool)
        nonfin = ~np.isfinite(U).all(1)
        if nonfin.any():
            vc = np.zeros(nv, np.int64); vc[CV.reshape(-1)] = np.repeat(chart, 3)
            fin[np.unique(vc[nonfin])] = False
        valid = (flips == 0) & ~col & (bad_an <= frac) & (bad_ar <= frac) & fin
        # non-disk charts (rings, pinched vertices) can overlap without a boundary
        # crossing: raster-check those that are otherwise valid
        nd = np.nonzero(valid & ~((chi == 1) & (loops == 1)))[0]
        if len(nd):
            from .overlapfix import raster_overlap
            Uc = U[CV]
            for c in nd.tolist():
                rows = np.nonzero(chart == c)[0]
                used, inv = np.unique(CV[rows], return_inverse=True)
                if raster_overlap(U[used], inv.reshape(-1, 3)) is not None:
                    valid[c] = False
                    col[c] = True
        return ul, chart, U[CV], valid, {"collide": int(col.sum()), "flips": int((flips > 0).sum()),
                                           "distort": int(((bad_an > frac) | (bad_ar > frac)).sum())}

    def try_parts(self, part):
        """Whole-part unwrap: every part projected onto its plane and relaxed
        (flip-safe ARAP) in one GPU batch, validated on the real triangles with
        the part budget. Returns the set of part ids that are valid islands."""
        faces = np.nonzero(part >= 0)[0]
        if len(faces) == 0:
            return set()
        saved = dict(self.cfg)
        self.cfg = dict(self.cfg, max_aniso=self.cfg.get("part_max_aniso", self.cfg["max_aniso"] * 1.25),
                        max_area=self.cfg.get("part_max_area", self.cfg["max_area"] * 1.4),
                        bad_area_fraction=self.cfg.get("part_bad_fraction", 0.10))
        try:
            ul, chart, cu, ok_, info = self.solve(faces, part, None, iters=self.cfg.get("part_iters", 8))
        finally:
            self.cfg = saved
        good = set(ul[ok_].tolist())
        self.cuv[faces] = cu                 # failed parts keep their layout for targeted splitting
        self.log(f"hier parts: {len(good)}/{len(ul)} parts unwrap as one island by projection {info}")
        # second chance: cut every failed part to a disk along cheap (hidden/creased)
        # seams and unroll it conformally, then ARAP
        bad = ul[~ok_]
        if len(bad) and self.cfg.get("part_lscm", False):
            from . import charts as ch
            topo = self.topo
            cut = np.zeros(len(topo.E), np.bool_)
            in_chart = np.zeros(len(topo.F), np.bool_)
            bf = faces[np.isin(part[faces], bad)]
            o = np.argsort(part[bf], kind="stable"); sl = part[bf][o]
            cuts = np.r_[0, np.nonzero(sl[1:] != sl[:-1])[0] + 1, len(sl)]
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                pf = bf[o[c0:c1]]
                c = ch.Chart(topo, pf)
                in_chart[pf] = True
                try:
                    ch.cut_to_disk(c, self.costs, in_chart)
                except Exception:
                    pass
                in_chart[pf] = False
                if c.cut:
                    cut[np.fromiter(c.cut, np.int64)] = True
            self.cfg = dict(self.cfg, max_aniso=saved.get("part_max_aniso", saved["max_aniso"] * 1.25),
                            max_area=saved.get("part_max_area", saved["max_area"] * 1.4),
                            bad_area_fraction=saved.get("part_bad_fraction", 0.10))
            try:
                ul2, chart2, cu2, ok2, info2 = self.solve(bf, part, None, iters=self.cfg.get("part_iters", 8),
                                                          cut=cut, init="lscm")
            finally:
                self.cfg = saved
            good2 = set(ul2[ok2].tolist())
            sel2 = ok2[np.searchsorted(ul2, part[bf])]
            self.cuv[bf[sel2]] = cu2[sel2]
            ce = np.nonzero(cut)[0]
            if len(ce):
                keep = np.isin(part[np.maximum(topo.EF[ce, 0], 0)], list(good2))
                cut[ce[~keep]] = False
            self.part_cut = cut
            self.log(f"hier parts: +{len(good2)}/{len(ul2)} after cut + conformal unroll {info2}")
            good |= good2
        return good

    def run(self, atoms):
        t0 = time.perf_counter()
        topo, cfg = self.topo, self.cfg
        lab = atoms.astype(np.int64).copy()
        valid_face = (lab >= 0)
        faces = np.nonzero(valid_face)[0]
        stats = {"levels": 0, "merges": 0, "atom_splits": 0}
        # ---- level 0: atoms (invalid atoms are split in two and re-solved)
        from .segment import split_two
        todo = faces
        invalid = set()
        for rnd in range(cfg.get("hier_atom_rounds", 6)):
            ts = time.perf_counter()
            ul, chart, cu, ok_, info = self.solve(todo, lab, None, iters=cfg.get("hier_atom_iters", 6))
            self.cuv[todo] = cu
            bad = ul[~ok_]
            self.log(f"hier atom round {rnd}: {len(todo)} faces, {len(ul)} charts, {len(bad)} invalid {info}, "
                     f"{time.perf_counter() - ts:.1f}s")
            if len(bad) == 0:
                invalid = set()
                break
            invalid = set(bad.tolist())
            if rnd == cfg.get("hier_atom_rounds", 6) - 1:
                break
            nl = int(lab.max()) + 1
            redo = []
            # faces of every bad atom, grouped once (not one full scan per atom)
            sel = todo[np.isin(lab[todo], bad)]
            o = np.argsort(lab[sel], kind="stable")
            sl = lab[sel][o]
            cuts = np.r_[0, np.nonzero(sl[1:] != sl[:-1])[0] + 1, len(sl)]
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                fa = sel[o[c0:c1]]
                if len(fa) <= 1:
                    continue
                parts = _split_atom(topo, fa, self.N)
                for p_ in parts[1:]:
                    lab[p_] = nl; nl += 1
                redo.append(fa)
                stats["atom_splits"] += 1
            if not redo:
                break
            todo = np.concatenate(redo)
        self.invalid = invalid
        self.log(f"hier: atoms ready {len(np.unique(lab[faces]))} charts ({len(invalid)} invalid), "
                 f"{time.perf_counter() - t0:.1f}s")
        # ---- merge levels
        failed = {}
        size = np.bincount(lab[faces], minlength=lab.max() + 1)
        budget = cfg.get("hier_seconds", 60)
        for level in range(cfg.get("hier_levels", 40)):
            if time.perf_counter() - t0 > budget:
                break
            if cfg.get("_check"):
                cfg["_check"]()
            pairs = [pq for pq in self._pairs(lab, faces, failed, size)
                     if pq[0] not in self.invalid and pq[1] not in self.invalid]
            if not pairs:
                break
            # union labels: b -> a
            newlab = lab.copy()
            mapping = {b: a for a, b in pairs}
            ua = np.array([a for a, b in pairs]); ub = np.array([b for a, b in pairs])
            rel = np.arange(lab.max() + 1); rel[ub] = ua
            newlab[faces] = rel[lab[faces]]
            uf = faces[np.isin(lab[faces], np.r_[ua, ub])]
            init = self._warm(uf, lab, newlab, pairs)
            ul, chart, cu, ok_, info = self.solve(uf, newlab, init, iters=cfg.get("hier_merge_iters", 5))
            accept = set(ul[ok_].tolist())
            # commit accepted unions
            pos = np.searchsorted(ul, newlab[uf])
            okf = ok_[pos]
            self.cuv[uf[okf]] = cu[okf]
            lab[uf[okf]] = newlab[uf[okf]]
            nacc = len(accept)
            for a, b in pairs:
                if a not in accept:
                    failed[(a, b)] = (size[a], size[b])
            size = np.bincount(lab[faces], minlength=lab.max() + 1)
            stats["levels"] += 1; stats["merges"] += nacc
            self.log(f"hier level {level}: {len(pairs)} pairs, {nacc} merged, {info} "
                     f"-> {len(np.unique(lab[faces]))} charts, {time.perf_counter() - t0:.1f}s")
            if nacc == 0:
                break
        stats["seconds"] = time.perf_counter() - t0
        stats["charts"] = int(len(np.unique(lab[faces])))
        return lab, self.cuv, stats

    def _pairs(self, lab, faces, failed, size):
        """Greedy matching of neighbouring charts by seam-aware priority."""
        topo = self.topo
        f = np.repeat(faces, 3); g = topo.FF[faces].ravel(); e = topo.FE[faces].ravel()
        sel = g >= 0
        f, g, e = f[sel], g[sel], e[sel]
        la, lb = lab[f], lab[g]
        s2 = (la < lb) & (lb >= 0)
        la, lb, e = la[s2], lb[s2], e[s2]
        if len(la) == 0:
            return []
        n = lab.max() + 1
        key = la * n + lb
        uk, inv = np.unique(key, return_inverse=True)
        L = np.bincount(inv, weights=topo.edge_len[e])
        good = 1.0 - np.clip(self.costs[e] / np.maximum(topo.edge_len[e], 1e-30) - 0.06, 0, 1) ** 0.5
        Lg = np.bincount(inv, weights=topo.edge_len[e] * good)
        H = np.bincount(inv, weights=self.ef["hard"][e].astype(float)) > 0
        A = np.bincount(lab[faces], weights=topo.face_area[faces], minlength=n)
        # perimeter per chart
        fb = np.repeat(faces, 3); gb = topo.FF[faces].ravel(); eb = topo.FE[faces].ravel()
        isb = (gb < 0) | (lab[np.maximum(gb, 0)] != lab[fb])
        per = np.bincount(lab[fb][isb], weights=topo.edge_len[eb][isb], minlength=n)
        a_ = uk // n; b_ = uk % n
        if self.part is not None:
            pl = -np.ones(n, np.int64)
            pl[lab[faces]] = self.part[faces]
            cross = pl[a_] != pl[b_]
        else:
            cross = np.zeros(len(a_), bool)
        contact = L / np.maximum(np.minimum(per[a_], per[b_]), 1e-30)
        g_ = Lg / np.maximum(L, 1e-30)
        small = np.minimum(A[a_], A[b_])
        prio = (0.15 + g_) / (0.2 + contact) * np.sqrt(small / max(A.sum(), 1e-30))
        prio[H | cross] = np.inf
        prio[contact < self.cfg.get("merge_min_contact", 0.04)] = np.inf
        order = np.argsort(prio, kind="stable")
        used = set()
        pairs = []
        for i in order.tolist():
            if not np.isfinite(prio[i]):
                break
            x, y = int(a_[i]), int(b_[i])
            if x in used or y in used:
                continue
            fk = failed.get((x, y)) or failed.get((y, x))
            if fk is not None and fk == (size[x], size[y]):
                continue
            used.add(x); used.add(y)
            pairs.append((x, y) if A[x] >= A[y] else (y, x))
        return pairs

    def _warm(self, uf, lab, newlab, pairs):
        """Per-corner initial UVs: child b rigidly aligned onto child a using the
        corners they share across their common boundary."""
        topo = self.topo
        cuv = self.cuv[uf].copy()
        # shared corner pairs: face f (child a) and neighbour g (child b) across an edge
        loc = self.loc
        loc[uf] = np.arange(len(uf))
        try:
            fs = np.repeat(uf, 3); ks = np.tile(np.arange(3), len(uf))
            g = topo.FF[uf].ravel()
            ok = g >= 0
            fs, ks, g = fs[ok], ks[ok], g[ok]
            ok = loc[g] >= 0
            fs, ks, g = fs[ok], ks[ok], g[ok]
            ok = (newlab[fs] == newlab[g]) & (lab[fs] != lab[g])
            fs, ks, g = fs[ok], ks[ok], g[ok]
            pa = np.array([a for a, b in pairs], np.int64); pb_ = np.array([b for a, b in pairs], np.int64)
            amap = -np.ones(int(lab.max()) + 1, np.int64); amap[pa] = np.arange(len(pairs))
            bmap = -np.ones(int(lab.max()) + 1, np.int64); bmap[pb_] = np.arange(len(pairs))
            pid = amap[lab[fs]]
            is_a = pid >= 0                 # f in child a, g in child b
            fs, ks, g, pid = fs[is_a], ks[is_a], g[is_a], pid[is_a]
            wv = topo.F[fs, ks]
            kg = np.argmax(topo.F[g] == wv[:, None], axis=1)
            ua = self.cuv[fs, ks]; ub = self.cuv[g, kg]
            npair = len(pairs)
            R = np.zeros((npair, 2, 2)); R[:, 0, 0] = R[:, 1, 1] = 1.0
            tvec = np.zeros((npair, 2))
            if len(pid):
                cnt = np.bincount(pid, minlength=npair).astype(float)
                ma = np.stack([np.bincount(pid, ua[:, k], npair) for k in range(2)], 1)
                mb = np.stack([np.bincount(pid, ub[:, k], npair) for k in range(2)], 1)
                ma /= np.maximum(cnt, 1)[:, None]; mb /= np.maximum(cnt, 1)[:, None]
                da = ua - ma[pid]; db = ub - mb[pid]
                sxx = np.bincount(pid, weights=db[:, 0] * da[:, 0] + db[:, 1] * da[:, 1], minlength=npair)
                sxy = np.bincount(pid, weights=db[:, 0] * da[:, 1] - db[:, 1] * da[:, 0], minlength=npair)
                th = np.arctan2(sxy, sxx)
                c, s = np.cos(th), np.sin(th)
                R = np.stack([np.stack([c, -s], 1), np.stack([s, c], 1)], 1)
                tvec = ma - np.einsum("pij,pj->pi", R, mb)
                # mirrored child b (inconsistent orientation): try reflection fit too
                good = cnt >= 2
                R[~good] = np.eye(2); tvec[~good] = 0
            pb = bmap[lab[uf]]
            sel = pb >= 0
            cb = cuv[sel]
            cuv[sel] = np.einsum("fij,fkj->fki", R[pb[sel]], cb) + tvec[pb[sel]][:, None, :]
        finally:
            loc[uf] = -1
        return cuv
