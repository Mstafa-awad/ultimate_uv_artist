"""Chart parameterization: LSCM initialisation + ARAP / symmetric-Dirichlet
refinement with a flip-preventing line search.

Every chart passed here is a triangle list over chart-local vertices; it should
be a topological disk (the charting stage cuts it). Results are isometric in
scale (UV units = 3D units) so texel density is consistent across charts.
"""
from __future__ import annotations

import numpy as np
import numba as nb
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIRECT_LIMIT = 220_000      # vertices: above this use warm-started CG


# ------------------------------------------------------------- basic geometry
def local_frames(P, T):
    """Isometric 2D coordinates of each triangle: (t,3,2) with corner 0 at origin."""
    p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e1 = p1 - p0; e2 = p2 - p0
    l1 = np.linalg.norm(e1, axis=1)
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1)
    ex = e1 / np.maximum(l1, 1e-300)[:, None]
    ey = np.cross(n / np.maximum(nn, 1e-300)[:, None], ex)
    X = np.zeros((len(T), 3, 2))
    X[:, 1, 0] = l1
    X[:, 2, 0] = (e2 * ex).sum(1)
    X[:, 2, 1] = (e2 * ey).sum(1)
    return X, 0.5 * nn


def cot_weights(X):
    """Per-triangle cotangents of the angle opposite edge k (edge k = corner k+1 -> k+2)."""
    cots = np.empty((len(X), 3))
    for k in range(3):
        i, j, o = (k + 1) % 3, (k + 2) % 3, k
        a = X[:, i] - X[:, o]; b = X[:, j] - X[:, o]
        cr = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        dt = (a * b).sum(1)
        cots[:, k] = dt / np.maximum(cr, 1e-300)
    return np.clip(cots, -1e4, 1e4)


def _lap_triplets(T, cots):
    i = np.concatenate([T[:, 1], T[:, 2], T[:, 0]])
    j = np.concatenate([T[:, 2], T[:, 0], T[:, 1]])
    w = 0.5 * np.concatenate([cots[:, 0], cots[:, 1], cots[:, 2]])
    return np.concatenate([i, j, i, j]), np.concatenate([j, i, i, j]), np.concatenate([-w, -w, w, w])


def cot_laplacian(n, T, cots):
    I, J, W = _lap_triplets(T, cots)
    return sp.csc_matrix((W, (I, J)), shape=(n, n))


def boundary_edges_oriented(T):
    """Directed boundary half-edges (i->j) of a triangle list."""
    a = np.concatenate([T[:, 0], T[:, 1], T[:, 2]])
    b = np.concatenate([T[:, 1], T[:, 2], T[:, 0]])
    n = int(max(a.max(), b.max())) + 1
    fwd = a * n + b
    rev = b * n + a
    s = np.sort(fwd)
    has_twin = np.isin(rev, s)
    return a[~has_twin], b[~has_twin]


def boundary_loops(T):
    bi, bj = boundary_edges_oriented(T)
    if len(bi) == 0:
        return []
    nxt = {}
    for i, j in zip(bi.tolist(), bj.tolist()):
        nxt.setdefault(i, []).append(j)
    loops = []
    used = set()
    for s in bi.tolist():
        if s in used:
            continue
        loop = [s]; used.add(s)
        cur = s
        while True:
            cand = [x for x in nxt.get(cur, []) if x not in used or x == s]
            if not cand:
                break
            cur = cand[0]
            if cur == s:
                break
            loop.append(cur); used.add(cur)
        loops.append(np.array(loop, np.int64))
    return loops


def signed_areas(U, T):
    a, b, c = U[T[:, 0]], U[T[:, 1]], U[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


# ------------------------------------------------------------- solvers
class SPDSolver:
    """Reusable solver for a fixed SPD matrix (after removing pinned rows)."""

    def __init__(self, A):
        self.n = A.shape[0]
        self.A = A.tocsc()
        self.lu = None
        if self.n <= DIRECT_LIMIT:
            try:
                self.lu = spla.splu(self.A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
            except Exception:
                self.lu = None
        if self.lu is None:
            d = self.A.diagonal()
            self.M = sp.diags(1.0 / np.where(np.abs(d) > 1e-300, d, 1.0))

    def solve(self, b, x0=None, tol=1e-7, maxiter=None):
        if self.lu is not None:
            return self.lu.solve(b)
        out = np.empty_like(b)
        bb = b if b.ndim == 2 else b[:, None]
        xx = None if x0 is None else (x0 if x0.ndim == 2 else x0[:, None])
        for c in range(bb.shape[1]):
            x, _ = spla.cg(self.A, bb[:, c], x0=None if xx is None else xx[:, c], rtol=tol,
                           maxiter=maxiter or 2000, M=self.M)
            if b.ndim == 2:
                out[:, c] = x
            else:
                out = x
        return out


# ------------------------------------------------------------- LSCM
def lscm(P, T, X=None, cots=None, pins=None):
    n = len(P)
    if X is None:
        X, _ = local_frames(P, T)
    if cots is None:
        cots = cot_weights(X)
    bi, bj = boundary_edges_oriented(T)
    if len(bi) == 0:
        raise ValueError("chart has no boundary (closed surface): needs a cut")
    # conformal energy matrix H = [[L, -B], [B, L]] assembled as triplets
    I, J, W = _lap_triplets(T, cots)
    hb = np.full(len(bi), 0.5)
    rows = np.concatenate([I, I + n, bi, bj, bi + n, bj + n])
    cols = np.concatenate([J, J + n, bj + n, bi + n, bj, bi])
    vals = np.concatenate([W, W, -hb, hb, hb, -hb])
    if pins is None:
        pins = _far_pair(P, np.unique(bi))
    p0, p1 = pins
    d = np.linalg.norm(P[p1] - P[p0])
    pin_idx = np.array([p0, p1, p0 + n, p1 + n])
    pin_val = np.zeros(2 * n); pin_val[p1] = d
    newid = np.arange(2 * n); newid[pin_idx] = -1
    keep = np.ones(2 * n, bool); keep[pin_idx] = False
    newid[keep] = np.arange(int(keep.sum()))
    rf = newid[rows]; cf = newid[cols]
    both = (rf >= 0) & (cf >= 0)
    nf = int(keep.sum())
    Hff = sp.csc_matrix((vals[both], (rf[both], cf[both])), shape=(nf, nf))
    pc = (rf >= 0) & (cf < 0)
    rhs = -np.bincount(rf[pc], weights=vals[pc] * pin_val[cols[pc]], minlength=nf)
    sol = SPDSolver(Hff).solve(rhs)
    x = pin_val.copy(); x[keep] = sol
    U = np.stack([x[:n], x[n:]], 1)
    return U


def _far_pair(P, cand):
    Q = P[cand]
    c = Q.mean(0)
    i = cand[np.argmax(((Q - c) ** 2).sum(1))]
    j = cand[np.argmax(((Q - P[i]) ** 2).sum(1))]
    if i == j:
        j = cand[0] if cand[0] != i else cand[-1]
    return int(i), int(j)


def tutte(P, T):
    """Uniform-weight Tutte embedding onto a circle (flip-free for disks)."""
    loops = boundary_loops(T)
    loop = max(loops, key=len)
    n = len(P)
    seg = np.linalg.norm(P[np.roll(loop, -1)] - P[loop], axis=1)
    t = np.r_[0, np.cumsum(seg)[:-1]] / max(seg.sum(), 1e-300) * 2 * np.pi
    r = seg.sum() / (2 * np.pi)
    Ub = np.stack([np.cos(t), np.sin(t)], 1) * r
    I = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    J = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    A = sp.csr_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    Lu = sp.diags(deg) - A
    fixed = np.zeros(n, bool); fixed[loop] = True
    U = np.zeros((n, 2)); U[loop] = Ub
    fr = ~fixed
    rhs = -Lu[fr][:, fixed] @ Ub
    U[fr] = SPDSolver(Lu[fr][:, fr]).solve(rhs)
    return U


# ------------------------------------------------------------- local step kernels
def _arap_rotations_py(U, T, X, cots, mode, sd_weights):
    """Per-triangle target Jacobian. mode 0: ARAP rotation; mode 1: SLIM
    symmetric-Dirichlet reweighting (returns R and weight matrices)."""
    t = T.shape[0]
    R = np.empty((t, 2, 2))
    Wm = np.empty((t, 2, 2))
    for f in nb.prange(t):
        # Jacobian from local X to U
        x1 = X[f, 1, 0]; x2 = X[f, 2, 0]; y2 = X[f, 2, 1]
        i0 = T[f, 0]; i1 = T[f, 1]; i2 = T[f, 2]
        du1x = U[i1, 0] - U[i0, 0]; du1y = U[i1, 1] - U[i0, 1]
        du2x = U[i2, 0] - U[i0, 0]; du2y = U[i2, 1] - U[i0, 1]
        det = x1 * y2
        if abs(det) < 1e-300:
            R[f, 0, 0] = 1.0; R[f, 0, 1] = 0.0; R[f, 1, 0] = 0.0; R[f, 1, 1] = 1.0
            Wm[f, 0, 0] = 1.0; Wm[f, 0, 1] = 0.0; Wm[f, 1, 0] = 0.0; Wm[f, 1, 1] = 1.0
            continue
        j00 = du1x / x1; j10 = du1y / x1
        j01 = (du2x - j00 * x2) / y2; j11 = (du2y - j10 * x2) / y2
        if mode == 0:
            # closest rotation in closed form (no SVD): conformal part of J
            ca = j00 + j11; cb = j10 - j01
            h = np.sqrt(ca * ca + cb * cb)
            if h < 1e-300:
                ca = 1.0; cb = 0.0; h = 1.0
            R[f, 0, 0] = ca / h; R[f, 0, 1] = -cb / h; R[f, 1, 0] = cb / h; R[f, 1, 1] = ca / h
            Wm[f, 0, 0] = 1.0; Wm[f, 0, 1] = 0.0; Wm[f, 1, 0] = 0.0; Wm[f, 1, 1] = 1.0
            continue
        J = np.array([[j00, j01], [j10, j11]])
        Uu, S, Vt = np.linalg.svd(J)
        if np.linalg.det(Uu @ Vt) < 0:
            Uu[:, 1] = -Uu[:, 1]
            S[1] = -S[1]
        Rot = Uu @ Vt
        if mode == 0:
            R[f] = Rot
            Wm[f, 0, 0] = 1.0; Wm[f, 0, 1] = 0.0; Wm[f, 1, 0] = 0.0; Wm[f, 1, 1] = 1.0
        else:
            # SLIM weights for symmetric Dirichlet: w_i = sqrt((s - s^-3)/(s - 1)) ; s->1 limit 2
            w = np.empty(2)
            for k in range(2):
                s = abs(S[k])
                s = max(s, 1e-8)
                if abs(s - 1.0) < 1e-6:
                    w[k] = 2.0
                else:
                    w[k] = np.sqrt(max((s - s ** -3) / (s - 1.0), 1e-12))
            W = Uu @ np.diag(w) @ Uu.T
            R[f] = Rot
            Wm[f] = W
    return R, Wm


@nb.njit(cache=True, nogil=True)
def _max_step(U, D, T):
    """Largest t in (0, inf) before any triangle's signed area reaches zero
    (Smith & Schaefer flip-preventing line search)."""
    tmax = 1e30
    for f in range(T.shape[0]):
        a, b, c = T[f, 0], T[f, 1], T[f, 2]
        ux1 = U[b, 0] - U[a, 0]; uy1 = U[b, 1] - U[a, 1]
        ux2 = U[c, 0] - U[a, 0]; uy2 = U[c, 1] - U[a, 1]
        dx1 = D[b, 0] - D[a, 0]; dy1 = D[b, 1] - D[a, 1]
        dx2 = D[c, 0] - D[a, 0]; dy2 = D[c, 1] - D[a, 1]
        C = ux1 * uy2 - ux2 * uy1
        Bq = ux1 * dy2 + dx1 * uy2 - ux2 * dy1 - dx2 * uy1
        A = dx1 * dy2 - dx2 * dy1
        if C <= 0:
            continue
        # smallest positive root of A t^2 + B t + C
        if A == 0.0 or abs(A) < 1e-30 * max(abs(Bq), 1e-300):
            if Bq < 0:
                r = -C / Bq
                if r < tmax:
                    tmax = r
            continue
        disc = Bq * Bq - 4 * A * C
        if disc < 0:
            continue
        sq = np.sqrt(disc)
        r1 = (-Bq - sq) / (2 * A); r2 = (-Bq + sq) / (2 * A)
        if r1 > 0 and r1 < tmax:
            tmax = r1
        if r2 > 0 and r2 < tmax:
            tmax = r2
    return tmax


def _energy_py(U, T, X, area, kind):
    """kind 0: ARAP energy, 1: symmetric Dirichlet (inf if flipped)."""
    t = T.shape[0]
    e = np.zeros(t)
    for f in nb.prange(t):
        x1 = X[f, 1, 0]; x2 = X[f, 2, 0]; y2 = X[f, 2, 1]
        i0 = T[f, 0]; i1 = T[f, 1]; i2 = T[f, 2]
        det = x1 * y2
        if abs(det) < 1e-300:
            continue
        j00 = (U[i1, 0] - U[i0, 0]) / x1; j10 = (U[i1, 1] - U[i0, 1]) / x1
        j01 = ((U[i2, 0] - U[i0, 0]) - j00 * x2) / y2; j11 = ((U[i2, 1] - U[i0, 1]) - j10 * x2) / y2
        if kind == 0:
            # signed singular values in closed form: s1 = q + r, s2 = q - r
            qe = 0.5 * (j00 + j11); qh = 0.5 * (j10 - j01)
            rf = 0.5 * (j00 - j11); rg = 0.5 * (j10 + j01)
            q = np.sqrt(qe * qe + qh * qh); r = np.sqrt(rf * rf + rg * rg)
            e[f] = area[f] * ((q + r - 1) ** 2 + (q - r - 1) ** 2)
        else:
            dj = j00 * j11 - j01 * j10
            if dj <= 0:
                e[f] = np.inf
                continue
            fro = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11
            e[f] = area[f] * (fro + fro / (dj * dj) - 4.0)
    return e.sum()


# serial and multithreaded builds of the per-triangle kernels: thread launch
# overhead dominates on the thousands of small charts tried while growing
_arap_rot_s = nb.njit(cache=True, nogil=True)(_arap_rotations_py)
_arap_rot_p = nb.njit(cache=True, parallel=True)(_arap_rotations_py)
_energy_s = nb.njit(cache=True, nogil=True)(_energy_py)
_energy_p = nb.njit(cache=True, parallel=True)(_energy_py)
PARALLEL_MIN_TRIS = 20000


def _arap_rotations(U, T, X, cots, mode, sd_weights):
    f = _arap_rot_p if T.shape[0] >= PARALLEL_MIN_TRIS else _arap_rot_s
    return f(U, T, X, cots, mode, sd_weights)


def _energy(U, T, X, area, kind):
    f = _energy_p if T.shape[0] >= PARALLEL_MIN_TRIS else _energy_s
    return f(U, T, X, area, kind)


def _assemble_weighted(n, T, X, Wm, area):
    """Global matrix for SLIM: sum_f area |W_f (J_f - R_f)|^2 with J linear in U.
    Uses per-triangle gradient operator G (2x3) so J = [u;v] G^T."""
    t = len(T)
    x1 = X[:, 1, 0]; x2 = X[:, 2, 0]; y2 = X[:, 2, 1]
    # gradient of linear function with values (q0,q1,q2): grad = [(q1-q0)/x1 ; ((q2-q0) - (q1-q0) x2/x1)/y2]
    G = np.zeros((t, 2, 3))
    G[:, 0, 0] = -1 / x1; G[:, 0, 1] = 1 / x1
    G[:, 1, 0] = (-1 + x2 / x1) / y2; G[:, 1, 1] = (-x2 / x1) / y2; G[:, 1, 2] = 1 / y2
    return G


def _slim_system(n, T, G, Wm, R, area, U_shape_hint=None):
    """Build normal equations for min sum a |W (J - R)|_F^2, unknowns x=[u;v].
    J = [[gu],[gv]] where gu = G q_u (2-vector). W acts on rows (2x2)."""
    t = len(T)
    # J_rows: row r of J is grad of component r: J[r,:] = G @ q_r  -> W J: (W J)[s,:] = sum_r W[s,r] G q_r
    # Residual entries: for s in 0..1, c in 0..1: sum_r W[s,r] (G[c,:] . q_r) - (W R)[s,c]
    WR = np.einsum("tsr,trc->tsc", Wm, R)
    rows = []; cols = []; vals = []; rhs_rows = []
    sq = np.sqrt(area)
    ridx = 0
    I = []; Jc = []; Vv = []; Bv = []
    for s in range(2):
        for c in range(2):
            # coefficient for unknown q_r[corner k]: W[s,r]*G[c,k]
            row = np.arange(t) + ridx * t
            for r in range(2):
                for k in range(3):
                    I.append(row); Jc.append(T[:, k] + r * n)
                    Vv.append(sq * Wm[:, s, r] * G[:, c, k])
            Bv.append(sq * WR[:, s, c])
            ridx += 1
    A = sp.csr_matrix((np.concatenate(Vv), (np.concatenate(I), np.concatenate(Jc))), shape=(4 * t, 2 * n))
    b = np.concatenate(Bv)
    return A, b


def refine(P, T, U, iters=10, method="arap", X=None, area=None, cots=None, solver=None, line_search=True,
           tol=1e-5, pin=None):
    """Local-global refinement. ``method``: 'arap' or 'slim' (symmetric Dirichlet)."""
    n = len(P)
    if X is None:
        X, area = local_frames(P, T)
    if cots is None:
        cots = cot_weights(X)
    flips0 = int((signed_areas(U, T) <= 0).sum())
    use_ls = line_search and flips0 == 0
    pin = 0 if pin is None else pin
    if method == "arap":
        # soft pin: L + w e_p e_p^T. The ARAP right-hand side sums to zero, so the
        # solution keeps U[pin] exactly while the matrix stays SPD (no slicing).
        I, J, W = _lap_triplets(T, cots)
        wpin = float(np.abs(W).max()) if len(W) else 1.0
        if solver is None:
            L = sp.csc_matrix((np.r_[W, wpin], (np.r_[I, pin], np.r_[J, pin])), shape=(n, n))
            solver = SPDSolver(L)
        prev = _energy(U, T, X, area, 0)
        for it in range(iters):
            R, _ = _arap_rotations(U, T, X, cots, 0, np.empty(0))
            b = _arap_rhs(n, T, X, cots, R)
            b[pin] += wpin * U[pin]
            Un = solver.solve(b, x0=U)
            if use_ls:
                D = Un - U
                tm = _max_step(U, D, T)
                step = min(1.0, 0.9 * tm)
                Un = U + step * D
            e = _energy(Un, T, X, area, 0)
            U = Un
            if abs(prev - e) <= tol * max(prev, 1e-300):
                break
            prev = e
        return U, solver
    # SLIM
    G = _assemble_weighted(n, T, X, None, area)
    prev = _energy(U, T, X, area, 1)
    if not np.isfinite(prev):
        return U, None
    lam = 1e-6 * area.sum()
    for it in range(iters):
        R, Wm = _arap_rotations(U, T, X, cots, 1, np.empty(0))
        A, b = _slim_system(n, T, G, Wm, R, area)
        AtA = (A.T @ A).tocsr()
        Atb = A.T @ b
        # fix translation softly
        reg = sp.diags(np.full(2 * n, lam * 1e-6))
        M = (AtA + reg).tocsr()
        x0 = np.r_[U[:, 0], U[:, 1]]
        Atb = Atb + lam * 1e-6 * x0
        if 2 * n <= DIRECT_LIMIT:
            x = spla.spsolve(M.tocsc(), Atb)
        else:
            d = M.diagonal()
            x, _ = spla.cg(M, Atb, x0=x0, rtol=1e-6, maxiter=500, M=sp.diags(1 / d))
        Un = np.stack([x[:n], x[n:]], 1)
        D = Un - U
        tm = _max_step(U, D, T)
        step = min(1.0, 0.9 * tm)
        # backtracking on energy
        e = _energy(U + step * D, T, X, area, 1)
        k = 0
        while (not np.isfinite(e) or e > prev) and k < 12:
            step *= 0.5
            e = _energy(U + step * D, T, X, area, 1)
            k += 1
        if not np.isfinite(e) or e > prev:
            break
        U = U + step * D
        if abs(prev - e) <= tol * prev:
            prev = e
            break
        prev = e
    return U, None


@nb.njit(cache=True, nogil=True)
def _arap_rhs_kernel(n, T, X, cots, R, b):
    for f in range(T.shape[0]):
        for k in range(3):
            i = T[f, (k + 1) % 3]; j = T[f, (k + 2) % 3]
            ii = (k + 1) % 3; jj = (k + 2) % 3
            w = 0.5 * cots[f, k]
            dx0 = X[f, ii, 0] - X[f, jj, 0]; dx1 = X[f, ii, 1] - X[f, jj, 1]
            rx = R[f, 0, 0] * dx0 + R[f, 0, 1] * dx1
            ry = R[f, 1, 0] * dx0 + R[f, 1, 1] * dx1
            b[i, 0] += w * rx; b[i, 1] += w * ry
            b[j, 0] -= w * rx; b[j, 1] -= w * ry


def _arap_rhs(n, T, X, cots, R):
    b = np.zeros((n, 2))
    _arap_rhs_kernel(n, T, X, cots, R, b)
    return b


# ------------------------------------------------------------- flip repair
@nb.njit(cache=True, nogil=True)
def _repair_kernel(U, T, vt_ptr, vt_idx, nbr_ptr, nbr_idx, is_bnd, rounds):
    nv = U.shape[0]
    for r in range(rounds):
        bad = 0
        for f in range(T.shape[0]):
            a, b, c = T[f, 0], T[f, 1], T[f, 2]
            ar = (U[b, 0] - U[a, 0]) * (U[c, 1] - U[a, 1]) - (U[c, 0] - U[a, 0]) * (U[b, 1] - U[a, 1])
            if ar > 0:
                continue
            bad += 1
            for k in range(3):
                v = T[f, k]
                if is_bnd[v]:
                    continue
                sx = 0.0; sy = 0.0; n = 0
                for q in range(nbr_ptr[v], nbr_ptr[v + 1]):
                    w = nbr_idx[q]
                    sx += U[w, 0]; sy += U[w, 1]; n += 1
                if n > 0:
                    U[v, 0] = sx / n; U[v, 1] = sy / n
        if bad == 0:
            break
    return U


def repair_flips(U, T, rounds=60):
    """Move interior vertices of folded triangles to their 1-ring centroid
    (local Tutte relaxation) until no fold remains or ``rounds`` is reached."""
    n = len(U)
    I = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    J = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    key = np.unique(I * n + J)
    I, J = key // n, key % n
    ptr = np.zeros(n + 1, np.int64); np.add.at(ptr, I + 1, 1); np.cumsum(ptr, out=ptr)
    bi, bj = boundary_edges_oriented(T)
    is_bnd = np.zeros(n, np.bool_); is_bnd[bi] = True; is_bnd[bj] = True
    U = np.ascontiguousarray(U.copy())
    return _repair_kernel(U, np.ascontiguousarray(T, np.int64), ptr, J, ptr, J, is_bnd, rounds)


# ------------------------------------------------------------- chart quality
def chart_quality(P, T, U, X=None, area=None):
    """Area-weighted distortion of one chart (density normalised to the chart)."""
    if X is None:
        X, area = local_frames(P, T)
    sa = signed_areas(U, T)
    A = area.sum()
    flips = int((sa <= 0).sum()) if (sa > 0).sum() >= (sa < 0).sum() else int((sa >= 0).sum())
    # Jacobians
    x1 = X[:, 1, 0]; x2 = X[:, 2, 0]; y2 = X[:, 2, 1]
    a0 = U[T[:, 0]]; a1 = U[T[:, 1]]; a2 = U[T[:, 2]]
    ok = np.abs(x1 * y2) > 1e-300
    x1s = np.where(ok, x1, 1); y2s = np.where(ok, y2, 1)
    j00 = (a1[:, 0] - a0[:, 0]) / x1s; j10 = (a1[:, 1] - a0[:, 1]) / x1s
    j01 = ((a2[:, 0] - a0[:, 0]) - j00 * x2) / y2s; j11 = ((a2[:, 1] - a0[:, 1]) - j10 * x2) / y2s
    E = (j00 + j11) / 2; Fv = (j00 - j11) / 2; Gv = (j10 + j01) / 2; H = (j10 - j01) / 2
    Q = np.sqrt(E * E + H * H); Rr = np.sqrt(Fv * Fv + Gv * Gv)
    s1 = Q + Rr; s2 = np.abs(Q - Rr)
    uvA = np.abs(sa).sum()
    sc = uvA / max(A, 1e-300)
    w = area / max(A, 1e-300)
    an = s1 / np.maximum(s2, 1e-12)
    ar = (s1 * s2) / max(sc, 1e-300)
    lar = np.abs(np.log2(np.maximum(ar, 1e-12)))
    def wq(x, q):
        o = np.argsort(x); cw = np.cumsum(w[o])
        i = min(np.searchsorted(cw, q * cw[-1]), len(o) - 1)
        return float(x[o][i])
    st = np.sqrt((s1 ** 2 + s2 ** 2) / 2 / max(sc, 1e-300))
    return {"flips": flips, "aniso_p95": wq(an, 0.95), "aniso_mean": float((an * w).sum()),
            "area_p95": float(2 ** wq(lar, 0.95)), "area_logdev": float((lar * w).sum()),
            "stretch": float(np.sqrt((st ** 2 * w).sum())), "scale": float(np.sqrt(sc)),
            "per_tri_aniso": an, "per_tri_area": ar}
