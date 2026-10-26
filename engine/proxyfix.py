"""Clean-up of decimated planning proxies.

Aggressive quadric decimation (40x and more) leaves fold-overs: pairs of
adjacent triangles whose normals oppose each other although the original
surface is smooth there. Every such fin makes the chart containing it fold in
UV and forces peeling. Most are fixed by flipping the shared edge."""
from __future__ import annotations

import numpy as np


def _normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-30)[:, None], 0.5 * a


def _quality(V, tri):
    a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n)
    s = ((b - a) ** 2).sum() + ((c - b) ** 2).sum() + ((a - c) ** 2).sum()
    return 4 * np.sqrt(3) * area / max(s, 1e-30), n / max(2 * area, 1e-30)


def flip_foldovers(V, F, fold_dot=-0.2, passes=4, log=None):
    """Flip edges between opposing triangles when the flipped pair agrees
    (and agrees with the surrounding faces). Returns the new faces and the
    number of flips."""
    V = np.asarray(V, np.float64)
    F = np.array(F, np.int64)
    total = 0
    for _ in range(passes):
        N, A = _normals(V, F)
        # half-edge table
        m = len(F)
        a = F.ravel(); b = F[:, [1, 2, 0]].ravel()
        fid = np.repeat(np.arange(m), 3); kid = np.tile(np.arange(3), m)
        key = np.minimum(a, b) * len(V) + np.maximum(a, b)
        o = np.argsort(key, kind="stable"); ks = key[o]
        cnt = np.bincount(np.unique(ks, return_inverse=True)[1])
        # manifold interior edges only: exactly two half-edges with opposite direction
        st = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1]
        two = st[cnt == 2]
        h0, h1 = o[two], o[two + 1]
        ok = a[h0] == b[h1]
        h0, h1 = h0[ok], h1[ok]
        f0, f1 = fid[h0], fid[h1]
        d = (N[f0] * N[f1]).sum(1)
        cand = np.nonzero(d < fold_dot)[0]
        if len(cand) == 0:
            break
        edges = set(np.unique(key).tolist())
        # per-face neighbour normal sum (for the "agrees with surroundings" test)
        Nsum = np.zeros_like(N)
        np.add.at(Nsum, f0, N[f1] * A[f1][:, None]); np.add.at(Nsum, f1, N[f0] * A[f0][:, None])
        touched = np.zeros(m, bool)
        flips = 0
        for i in cand[np.argsort(d[cand])].tolist():
            fa, fb = int(f0[i]), int(f1[i])
            if touched[fa] or touched[fb]:
                continue
            ka = int(kid[h0[i]])
            va, vb = int(F[fa, ka]), int(F[fa, (ka + 1) % 3])
            vc = int(F[fa, (ka + 2) % 3])
            kb = int(kid[h1[i]])
            vd = int(F[fb, (kb + 2) % 3])
            if vc == vd:
                continue
            ek = min(vc, vd) * len(V) + max(vc, vd)
            if ek in edges:
                continue
            t1 = (va, vd, vc); t2 = (vd, vb, vc)
            q1, n1 = _quality(V, t1); q2, n2 = _quality(V, t2)
            if min(q1, q2) < 0.02:
                continue
            dn = float(n1 @ n2)
            ref = Nsum[fa] + Nsum[fb]
            rn = np.linalg.norm(ref)
            if rn > 0:
                ref = ref / rn
                agree = min(float(n1 @ ref), float(n2 @ ref))
            else:
                agree = 1.0
            if dn < max(float(d[i]), 0.0) + 0.3 or agree < 0.0:
                continue
            F[fa] = t1; F[fb] = t2
            touched[fa] = touched[fb] = True
            edges.add(ek)
            edges.discard(min(va, vb) * len(V) + max(va, vb))
            flips += 1
        total += flips
        if flips == 0:
            break
    if log:
        log(f"proxy clean-up: {total} fold-over edges flipped")
    return F, total
