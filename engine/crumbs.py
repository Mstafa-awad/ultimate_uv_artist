"""Absorb tiny islands ("crumbs" left by fold/overlap peeling) back into the
neighbouring island they share the most edges with, when the union stays
valid (no folded triangle, no boundary self-intersection)."""
from __future__ import annotations

import numpy as np

from .overlapfix import _signed, collision, overlapping_triangles


def arap_fill(P, T, U0, fixed, iters=12):
    """ARAP of a small chart with some vertices held fixed (Dirichlet).
    P: 3D positions, U0: start UVs (fixed rows are the targets)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    from . import param
    X, area = param.local_frames(P, T)
    cots = np.clip(param.cot_weights(X), -10.0, 1e3)
    n = len(U0)
    L = param.cot_laplacian(n, T, cots).tocsr()
    free = np.nonzero(~fixed)[0]; fx = np.nonzero(fixed)[0]
    if len(free) == 0:
        return U0
    Lff = L[free][:, free].tocsc()
    Lfc = L[free][:, fx]
    try:
        lu = splu(Lff + sp.identity(len(free)) * 1e-12)
    except Exception:
        return U0
    U = np.array(U0, np.float64)
    for _ in range(iters):
        R, _w = param._arap_rotations(U, T, X, cots, 0, False)
        b = np.zeros((n, 2))
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            w = 0.5 * cots[:, k]
            d = X[:, i] - X[:, j]
            Rd = np.einsum("tab,tb->ta", R, d) * w[:, None]
            np.add.at(b, T[:, i], Rd); np.add.at(b, T[:, j], -Rd)
        rhs = b[free] - Lfc @ U[fx]
        U[free] = lu.solve(rhs)
    return U


def absorb_crumbs(topo, final, FinalChart, part=None, max_tris=100, max_rounds=3, log=None):
    m = len(topo.F)
    absorbed = 0
    for _ in range(max_rounds):
        chart_of = -np.ones(m, np.int64); row_of = -np.ones(m, np.int64)
        for ci, c in enumerate(final):
            chart_of[c.faces] = ci; row_of[c.faces] = np.arange(len(c.faces))
        sizes = np.array([len(c.faces) for c in final])
        order = np.argsort(sizes)
        changed = False
        alive = np.ones(len(final), bool)
        for ci in order.tolist():
            if sizes[ci] > max_tris:
                break
            if not alive[ci]:
                continue
            c = final[ci]
            # shared edges with neighbouring islands
            f = np.repeat(c.faces, 3); k = np.tile(np.arange(3), len(c.faces))
            g = topo.FF[c.faces].ravel()
            ok = g >= 0
            f, k, g = f[ok], k[ok], g[ok]
            nb = chart_of[g]
            ok = (nb >= 0) & (nb != ci) & alive[np.maximum(nb, 0)]
            if part is not None:
                ok &= part[g] == part[f]          # never merge across semantic parts
            if not ok.any():
                continue
            f, k, g, nb = f[ok], k[ok], g[ok], nb[ok]
            tgt = int(np.bincount(nb).argmax())
            sel = nb == tgt
            f, k, g = f[sel], k[sel], g[sel]
            n = final[tgt]
            # corner correspondences across each shared edge (both endpoints)
            pairs_c = []; pairs_n = []
            for fi, ki, gi in zip(f.tolist(), k.tolist(), g.tolist()):
                rc = row_of[fi]; rn = row_of[gi]
                for kk in (ki, (ki + 1) % 3):
                    w = topo.F[fi, kk]
                    kn = int(np.nonzero(topo.F[gi] == w)[0][0])
                    pairs_c.append(c.T[rc, kk]); pairs_n.append(n.T[rn, kn])
            pc = np.array(pairs_c); pn = np.array(pairs_n)
            A = c.U[pc]; B = n.U[pn]
            if len(pc) < 2:
                continue
            ma, mb = A.mean(0), B.mean(0)
            da, db = A - ma, B - mb
            th = np.arctan2((da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0]).sum(), (da * db).sum())
            # crumb may be mirrored relative to its neighbour
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            Uc = (c.U - ma) @ R.T + mb
            free = np.ones(len(c.U), bool); free[pc] = False
            remap = np.empty(len(c.U), np.int64)
            remap[free] = len(n.U) + np.arange(int(free.sum()))
            remap[pc] = pn
            Tc = remap[c.T]
            sa_old = _signed(c.U, c.T)

            def union(Uc_):
                # crumb vertices on the shared edges snap to the neighbour's; the
                # others are appended after the neighbour's vertices
                U_new = np.r_[n.U, Uc_[free]]
                sa_new = _signed(U_new, Tc)
                ratio = sa_new / np.maximum(sa_old, 1e-30)
                if (sa_new <= 0).any() or ratio.min() < 0.3 or ratio.max() > 3.0:
                    return None
                # local validity: neighbour triangles near the crumb + the crumb itself
                lo = U_new[Tc].reshape(-1, 2).min(0); hi = U_new[Tc].reshape(-1, 2).max(0)
                pad = 0.05 * float((hi - lo).max()) + 1e-12
                tb = n.U[n.T]
                near = ((tb[:, :, 0].max(1) >= lo[0] - pad) & (tb[:, :, 0].min(1) <= hi[0] + pad) &
                        (tb[:, :, 1].max(1) >= lo[1] - pad) & (tb[:, :, 1].min(1) <= hi[1] + pad))
                Tl = np.r_[n.T[near], Tc]
                used, inv = np.unique(Tl, return_inverse=True)
                if collision(U_new[used], inv.reshape(-1, 3)) is not None:
                    return None
                if overlapping_triangles(U_new[used], inv.reshape(-1, 3), res=1024).any():
                    return None
                return U_new

            U_new = union(Uc)
            if U_new is None and len(np.unique(pc)) >= 2:
                # rigid fit failed: hold the shared vertices on the neighbour and
                # let the rest of the crumb follow by ARAP
                vw = np.empty(len(c.U), np.int64); vw[c.T.ravel()] = topo.F[c.faces].ravel()
                Ua = Uc.copy(); Ua[pc] = n.U[pn]
                fixed = np.zeros(len(c.U), bool); fixed[pc] = True
                Ua = arap_fill(topo.V[vw].astype(np.float64), c.T, Ua, fixed)
                if np.isfinite(Ua).all():
                    U_new = union(Ua)
            if U_new is None:
                continue
            T_new = np.r_[n.T, Tc]
            final[tgt] = FinalChart(np.r_[n.faces, c.faces], T_new, U_new, n.quality)
            chart_of[c.faces] = tgt
            row_of[c.faces] = len(n.faces) + np.arange(len(c.faces))
            alive[ci] = False
            sizes[tgt] += sizes[ci]
            absorbed += 1
            changed = True
        final = [c for i, c in enumerate(final) if alive[i]]
        if not changed:
            break
    if log:
        log(f"crumbs absorbed into neighbours: {absorbed}")
    return final, absorbed
