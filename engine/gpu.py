"""GPU (PyTorch/CUDA) batched parameterization of many charts at once.

All charts are stacked into one block-diagonal problem: per-triangle local
step (closed-form 2x2 rotation), one sparse cotangent Laplacian with a soft
pin per chart, Jacobi-preconditioned CG warm-started from the previous
iterate, and a per-chart flip-preventing line search. Runs on CPU torch when
no CUDA device is available. Memory: ~150 bytes per triangle.
"""
from __future__ import annotations

import numpy as np
import torch



def csr32(A):
    """CSR with int32 indices (cuSPARSE is markedly faster with them)."""
    A = A.to_sparse_csr() if A.layout != torch.sparse_csr else A
    return torch.sparse_csr_tensor(A.crow_indices().to(torch.int32), A.col_indices().to(torch.int32),
                                   A.values(), A.shape)


def spmm(L, X):
    """L @ X for a few dense columns: one SpMV per column is ~4x faster than
    the generic sparse @ dense-matrix path."""
    if X.dim() == 1:
        return L @ X
    return torch.stack([L @ X[:, k].contiguous() for k in range(X.shape[1])], 1)

def device():
    try:
        import comfy.model_management as mm
        d = mm.get_torch_device()
        return d if d.type == "cuda" else torch.device("cpu")
    except Exception:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def unique_edges(T, n):
    """Undirected edges of a triangle list: (a, b) with a < b, the inverse map
    from the 3t triangle edges (edge k of triangle i -> i + k*t) and the
    number of triangles using each edge. Memory ~ O(t), no 6t-entry coalesce."""
    a = torch.cat([T[:, 1], T[:, 2], T[:, 0]]); b = torch.cat([T[:, 2], T[:, 0], T[:, 1]])
    key = torch.minimum(a, b) * n + torch.maximum(a, b)
    del a, b
    uk, inv = torch.unique(key, return_inverse=True)
    del key
    cnt = torch.bincount(inv, minlength=len(uk))
    return uk // n, uk % n, inv, cnt


def sym_csr(n, ea, eb, off, diag):
    """Symmetric CSR (int32 indices) from unique edges with off-diagonal values
    and a diagonal."""
    dev = ea.device
    rows = torch.cat([ea, eb, torch.arange(n, device=dev)])
    cols = torch.cat([eb, ea, torch.arange(n, device=dev)])
    vals = torch.cat([off, off, diag])
    order = torch.argsort(rows * n + cols)
    rows, cols, vals = rows[order], cols[order], vals[order]
    crow = torch.zeros(n + 1, dtype=torch.int64, device=dev)
    crow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    return torch.sparse_csr_tensor(crow.to(torch.int32), cols.to(torch.int32), vals, (n, n))


class Batch:
    """charts: list of (P (k,3), T (t,3), U (k,2))."""

    def __init__(self, charts, dev=None, dtype=torch.float32):
        self.dev = dev or device()
        self.dt = dtype
        P, T, U, cv, ct = [], [], [], [], []
        off = 0
        for i, (p, t, u) in enumerate(charts):
            P.append(np.asarray(p, np.float64)); U.append(np.asarray(u, np.float64))
            T.append(np.asarray(t, np.int64) + off)
            cv.append(np.full(len(p), i, np.int64)); ct.append(np.full(len(t), i, np.int64))
            off += len(p)
        self.nc = len(charts)
        self.offsets = np.cumsum([0] + [len(p) for p, _, _ in charts])
        self.n = off
        d, f = self.dev, self.dt
        Pn = np.concatenate(P); Tn = np.concatenate(T)
        self.T = torch.from_numpy(Tn).to(d)
        self.U = torch.from_numpy(np.concatenate(U)).to(d, f)
        self.cv = torch.from_numpy(np.concatenate(cv)).to(d)
        self.ct = torch.from_numpy(np.concatenate(ct)).to(d)
        # isometric local frames (float64 on the CPU; only float32 results go to the GPU)
        p0, p1, p2 = Pn[Tn[:, 0]], Pn[Tn[:, 1]], Pn[Tn[:, 2]]
        e1 = p1 - p0; e2 = p2 - p0
        del p0, p1, p2
        l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-30)
        nrm = np.cross(e1, e2)
        dbl = np.linalg.norm(nrm, axis=1)
        ex = e1 / l1[:, None]
        ey = np.cross(nrm / np.maximum(dbl, 1e-30)[:, None], ex)
        X = np.zeros((len(Tn), 3, 2))
        X[:, 1, 0] = l1
        X[:, 2, 0] = (e2 * ex).sum(1)
        X[:, 2, 1] = (e2 * ey).sum(1)
        del e1, e2, nrm, ex, ey
        ok = (dbl > 1e-14 * np.maximum(l1 * l1, 1e-300)) & np.isfinite(dbl)
        X[~ok] = 0.0
        self.area = torch.from_numpy(0.5 * dbl).to(d, f)
        # triangles too thin to have a frame take no part in the energy
        self.ok = torch.from_numpy(ok).to(d)
        cots = []
        for k in range(3):
            i, j, o = (k + 1) % 3, (k + 2) % 3, k
            a_ = X[:, i] - X[:, o]; b_ = X[:, j] - X[:, o]
            cr = np.maximum(np.abs(a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]), 1e-30)
            cots.append(np.clip((a_ * b_).sum(1) / cr, -1e4, 1e4))
        w = 0.5 * np.stack(cots, 1)
        w[~ok] = 0.0
        self.w = torch.from_numpy(w).to(d, f)                   # (t,3) weight of edge k
        # x1, x2, y2 inverse for Jacobians
        self.x1 = torch.from_numpy(np.maximum(X[:, 1, 0], 1e-30)).to(d, f)
        self.x2 = torch.from_numpy(X[:, 2, 0]).to(d, f)
        y2 = X[:, 2, 1]
        self.y2 = torch.from_numpy(np.where(np.abs(y2) > 1e-30, y2, 1e-30)).to(d, f)
        self.X = torch.from_numpy(X).to(d, f)
        del X, w, cots
        self.scale = torch.ones(len(Tn), device=d, dtype=f)            # per-triangle target size
        self._build_laplacian()

    def _build_laplacian(self):
        T, w = self.T, self.w
        ea, eb, inv, cnt = unique_edges(T, self.n)
        self._edges = (ea, eb, cnt)                   # reused by repair_flips
        W = torch.cat([w[:, 0], w[:, 1], w[:, 2]])
        wsum = torch.zeros(len(ea), device=self.dev, dtype=self.dt).index_add_(0, inv, W)
        del inv
        diag = torch.zeros(self.n, device=self.dev, dtype=self.dt)
        diag.index_add_(0, ea, wsum); diag.index_add_(0, eb, wsum)
        # soft pin: first vertex of every chart (ARAP rhs sums to zero, so exact)
        pins = torch.from_numpy(self.offsets[:-1]).to(self.dev)
        self.pins = pins
        wpin = W.abs().max().clamp_min(1e-12)
        self.wpin = wpin
        diag[pins] += wpin
        self.L = sym_csr(self.n, ea, eb, -wsum, diag)
        self.dinv = 1.0 / diag.clamp_min(1e-30)

    # ---------------------------------------------------------------- pieces
    def jacobians(self, U):
        T = self.T
        a0, a1, a2 = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
        j00 = (a1[:, 0] - a0[:, 0]) / self.x1; j10 = (a1[:, 1] - a0[:, 1]) / self.x1
        j01 = ((a2[:, 0] - a0[:, 0]) - j00 * self.x2) / self.y2
        j11 = ((a2[:, 1] - a0[:, 1]) - j10 * self.x2) / self.y2
        return j00, j01, j10, j11

    def signed_area(self, U):
        T = self.T
        a, b, c = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

    def flips_per_chart(self, U=None):
        U = self.U if U is None else U
        bad = ((self.signed_area(U) < 0) & self.ok).to(self.dt)
        out = torch.zeros(self.nc, device=self.dev, dtype=self.dt)
        out.index_add_(0, self.ct, bad)
        return out

    def _rhs(self, U):
        j00, j01, j10, j11 = self.jacobians(U)
        # closest rotation of a 2x2 matrix: angle of its conformal part
        th = torch.atan2(j10 - j01, j00 + j11)
        th = torch.where(self.ok & torch.isfinite(th), th, torch.zeros_like(th))
        c, s = torch.cos(th) * self.scale, torch.sin(th) * self.scale
        X, T, w = self.X, self.T, self.w
        b = torch.zeros(self.n, 2, device=self.dev, dtype=self.dt)
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            dx = X[:, i] - X[:, j]
            rx = c * dx[:, 0] - s * dx[:, 1]
            ry = s * dx[:, 0] + c * dx[:, 1]
            r = torch.stack([rx, ry], 1) * w[:, k:k + 1]
            b.index_add_(0, T[:, i], r)
            b.index_add_(0, T[:, j], -r)
        b[self.pins] += self.wpin * U[self.pins]
        return b

    def _cg(self, b, x, iters=40, tol=1e-6):
        L, Minv = self.L, self.dinv[:, None]
        r = b - spmm(L, x)
        z = r * Minv
        p = z.clone()
        rz = (r * z).sum(0)
        b2 = (b * b).sum(0).clamp_min(1e-30)
        for _ in range(iters):
            Ap = spmm(L, p)
            alpha = rz / (p * Ap).sum(0).clamp_min(1e-30)
            x = x + p * alpha
            r = r - Ap * alpha
            if bool(((r * r).sum(0) <= (tol * tol) * b2).all()):
                break
            z = r * Minv
            rz2 = (r * z).sum(0)
            p = z + p * (rz2 / rz.clamp_min(1e-30))
            rz = rz2
        return x

    def _max_step(self, U, D):
        T = self.T
        u0, u1, u2 = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
        d0, d1, d2 = D[T[:, 0]], D[T[:, 1]], D[T[:, 2]]
        ux1, uy1 = u1[:, 0] - u0[:, 0], u1[:, 1] - u0[:, 1]
        ux2, uy2 = u2[:, 0] - u0[:, 0], u2[:, 1] - u0[:, 1]
        dx1, dy1 = d1[:, 0] - d0[:, 0], d1[:, 1] - d0[:, 1]
        dx2, dy2 = d2[:, 0] - d0[:, 0], d2[:, 1] - d0[:, 1]
        C = ux1 * uy2 - ux2 * uy1
        B = ux1 * dy2 + dx1 * uy2 - ux2 * dy1 - dx2 * uy1
        A = dx1 * dy2 - dx2 * dy1
        big = torch.full_like(C, 1e30)
        lin = torch.where((A.abs() < 1e-30) & (B < 0), -C / B.where(B != 0, torch.ones_like(B)), big)
        disc = B * B - 4 * A * C
        sq = disc.clamp_min(0).sqrt()
        A2 = torch.where(A.abs() < 1e-30, torch.ones_like(A), 2 * A)
        r1 = (-B - sq) / A2; r2 = (-B + sq) / A2
        quad = torch.minimum(torch.where(r1 > 0, r1, big), torch.where(r2 > 0, r2, big))
        quad = torch.where((disc >= 0) & (A.abs() >= 1e-30), quad, big)
        t = torch.minimum(lin, quad)
        t = torch.where(C > 0, t, big)          # already-flipped triangles do not limit the step
        out = torch.full((self.nc,), 1e30, device=self.dev, dtype=self.dt)
        out.scatter_reduce_(0, self.ct, t, reduce="amin")
        return out

    # ---------------------------------------------------------------- solvers
    def arap(self, iters=8, line_search=True, cg_iters=40):
        U = self.U
        for _ in range(iters):
            b = self._rhs(U)
            Un = self._cg(b, U.clone(), cg_iters)
            # charts that went non-finite keep their previous layout
            badv = ~torch.isfinite(Un).all(1)
            if bool(badv.any()):
                badc = torch.zeros(self.nc, dtype=torch.bool, device=self.dev)
                badc[self.cv[badv]] = True
                Un = torch.where(badc[self.cv][:, None], U, Un)
            if line_search:
                D = Un - U
                tmax = self._max_step(U, D)
                step = torch.clamp(0.9 * tmax, max=1.0)
                U = U + D * step[self.cv][:, None]
            else:
                U = Un
        self.U = U
        return U

    def repair_flips(self, rounds=60):
        """Move interior vertices of folded triangles to their 1-ring centroid."""
        T = self.T
        ea, eb, cnt = self._edges
        # boundary vertices: endpoints of edges used by one triangle only
        bnd = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        one = cnt == 1
        bnd[ea[one]] = True; bnd[eb[one]] = True
        A = sym_csr(self.n, ea, eb, torch.ones(len(ea), device=self.dev, dtype=self.dt),
                    torch.zeros(self.n, device=self.dev, dtype=self.dt))
        deg = spmm(A, torch.ones(self.n, 1, device=self.dev, dtype=self.dt)).squeeze(1).clamp_min(1)
        U = self.U
        for _ in range(rounds):
            bad = (self.signed_area(U) < 0) & self.ok
            if not bool(bad.any()):
                break
            mv = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
            mv[T[bad].reshape(-1)] = True
            mv &= ~bnd
            mean = spmm(A, U) / deg[:, None]
            U = torch.where(mv[:, None], mean, U)
        self.U = U
        return U

    def orient_positive(self):
        """Mirror charts whose triangles are mostly negatively oriented."""
        sa = self.signed_area(self.U)
        s = torch.zeros(self.nc, device=self.dev, dtype=self.dt)
        s.index_add_(0, self.ct, torch.sign(sa) * self.area)
        flip = (s < 0)[self.cv]
        self.U = torch.where(flip[:, None], self.U * torch.tensor([1.0, -1.0], device=self.dev, dtype=self.dt), self.U)

    def shrink(self, tri_idx, factor=0.6, floor=0.05, rings=3):
        """Lower the target size of the given triangles and their ``rings``
        neighbourhood (collision relief: the region contracts in UV)."""
        if len(tri_idx) == 0:
            return
        T = self.T
        mark = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        sel = torch.zeros(T.shape[0], dtype=torch.bool, device=self.dev)
        sel[torch.as_tensor(tri_idx, device=self.dev)] = True
        for _ in range(rings):
            mark[T[sel].reshape(-1)] = True
            sel = mark[T].any(1)
        self.scale = torch.where(sel, (self.scale * factor).clamp_min(floor), self.scale)

    def chart_uvs(self):
        U = self.U.double().cpu().numpy()
        return [U[a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:])]
