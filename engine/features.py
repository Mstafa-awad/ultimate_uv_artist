"""Per-face / per-edge surface features that drive seam and chart decisions."""
from __future__ import annotations

import numba as nb
import numpy as np

from .topology import dihedral


def fibonacci_dirs(k):
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


@nb.njit(cache=True)
def _voxelize(V, F, lo, inv, R, occ):
    for f in range(F.shape[0]):
        a = V[F[f, 0]]; b = V[F[f, 1]]; c = V[F[f, 2]]
        # sample the triangle densely enough for the voxel size
        l = max(np.sqrt(((a - b) ** 2).sum()), max(np.sqrt(((b - c) ** 2).sum()), np.sqrt(((c - a) ** 2).sum())))
        n = int(l * inv * 2) + 1
        for i in range(n + 1):
            for j in range(n + 1 - i):
                u = i / n; v = j / n
                p = a + (b - a) * u + (c - a) * v
                x = int((p[0] - lo[0]) * inv); y = int((p[1] - lo[1]) * inv); z = int((p[2] - lo[2]) * inv)
                if 0 <= x < R[0] and 0 <= y < R[1] and 0 <= z < R[2]:
                    occ[x, y, z] = 1


@nb.njit(cache=True, parallel=True)
def _openness(C, N, dirs, lo, inv, R, occ, max_steps):
    m = C.shape[0]
    out = np.zeros(m)
    k = dirs.shape[0]
    for f in nb.prange(m):
        n0 = N[f, 0]; n1 = N[f, 1]; n2 = N[f, 2]
        # tangent frame
        if abs(n0) < 0.9:
            t0, t1, t2 = 1.0, 0.0, 0.0
        else:
            t0, t1, t2 = 0.0, 1.0, 0.0
        dt = t0 * n0 + t1 * n1 + t2 * n2
        t0 -= n0 * dt; t1 -= n1 * dt; t2 -= n2 * dt
        tl = np.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
        t0 /= tl; t1 /= tl; t2 /= tl
        b0 = n1 * t2 - n2 * t1; b1 = n2 * t0 - n0 * t2; b2 = n0 * t1 - n1 * t0
        free = 0.0
        tot = 0.0
        for d in range(k):
            dz = abs(dirs[d, 2])
            w = dz  # cosine weighting
            dx = dirs[d, 0]; dy = dirs[d, 1]
            v0 = t0 * dx + b0 * dy + n0 * dz
            v1 = t1 * dx + b1 * dy + n1 * dz
            v2 = t2 * dx + b2 * dy + n2 * dz
            p0 = (C[f, 0] - lo[0]) * inv + n0 * 1.75
            p1 = (C[f, 1] - lo[1]) * inv + n1 * 1.75
            p2 = (C[f, 2] - lo[2]) * inv + n2 * 1.75
            blocked = False
            for s in range(max_steps):
                p0 += v0; p1 += v1; p2 += v2
                if p0 < 0 or p1 < 0 or p2 < 0:
                    break
                x = int(p0); y = int(p1); z = int(p2)
                if x < 0 or y < 0 or z < 0 or x >= R[0] or y >= R[1] or z >= R[2]:
                    break
                if occ[x, y, z]:
                    blocked = True
                    break
            tot += w
            if not blocked:
                free += w
        out[f] = free / tot if tot > 0 else 1.0
    return out


def face_openness(V, F, normals, res=128, ndirs=48):
    """Ambient-openness in [0,1] (1 = fully exposed), voxel ray-marched."""
    lo = V.min(0); hi = V.max(0)
    ext = (hi - lo).max()
    pad = ext * 0.02
    lo = lo - pad
    vox = (ext + 2 * pad) / res
    inv = 1.0 / vox
    R = np.ceil((hi + pad - lo) * inv).astype(np.int64) + 1
    occ = np.zeros(tuple(R), np.uint8)
    _voxelize(V, F, lo, inv, R, occ)
    C = V[F].mean(1)
    d = fibonacci_dirs(2 * ndirs)
    d = d[d[:, 2] > 0.02]
    return _openness(C, normals, d, lo, inv, R, occ, int(res * 0.6))


def edge_features(topo, openness=None, hard_angle_deg=None, split_normal_edges=None, label_edges=None,
                  smooth_N=None):
    """Returns dict of per-edge arrays: dihedral, crease(0..1), concave(0..1),
    hidden(0..1), hard(bool). Creases are measured on ``smooth_N`` (bilateral
    normals) when given so surface noise does not look like creases."""
    dih = dihedral_from_normals(topo, smooth_N) if smooth_N is not None else dihedral(topo)
    raw = dihedral(topo)
    ad = np.abs(dih)
    crease = np.clip((ad - np.radians(12)) / np.radians(48), 0, 1)
    concave = np.clip((-dih - np.radians(8)) / np.radians(40), 0, 1)
    E = topo.E
    f0 = np.maximum(topo.EF[:, 0], 0); f1 = np.maximum(topo.EF[:, 1], 0)
    if openness is not None:
        hidden = 1 - 0.5 * (openness[f0] + openness[f1])
    else:
        hidden = np.zeros(len(E))
    hard = np.zeros(len(E), bool)
    if hard_angle_deg is not None:
        hard |= np.abs(raw) > np.radians(hard_angle_deg)
    if split_normal_edges is not None:
        hard |= split_normal_edges
    if label_edges is not None:
        hard |= label_edges
    return {"dihedral": dih, "crease": crease, "concave": concave, "hidden": hidden, "hard": hard}


def seam_cost(topo, ef, hide_weight=1.0, crease_weight=1.0, up=None, front=None, dir_weight=0.0):
    """Cost per edge of placing a seam there (lower = better seam location)."""
    good = np.maximum(ef["crease"] * crease_weight, ef["concave"])
    good = np.maximum(good, ef["hidden"] * hide_weight)
    cost = topo.edge_len * (0.06 + (1 - np.clip(good, 0, 1)) ** 2)
    if dir_weight > 0 and (up is not None or front is not None):
        f0 = np.maximum(topo.EF[:, 0], 0); f1 = np.maximum(topo.EF[:, 1], 0)
        n = topo.face_normal[f0] + topo.face_normal[f1]
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        exposure = np.zeros(len(n))
        if front is not None:
            exposure = np.maximum(exposure, (n @ np.asarray(front, float)).clip(0, 1))
        if up is not None:
            exposure = np.maximum(exposure, 0.6 * (n @ np.asarray(up, float)).clip(0, 1))
        cost = cost * (1 + dir_weight * exposure)
    return cost


@nb.njit(cache=True)
def _bilateral_pass(FF, N, A, cos_t, out):
    m = FF.shape[0]
    for f in range(m):
        s0 = N[f, 0] * A[f]; s1 = N[f, 1] * A[f]; s2 = N[f, 2] * A[f]
        for k in range(3):
            g = FF[f, k]
            if g < 0:
                continue
            d = N[f, 0] * N[g, 0] + N[f, 1] * N[g, 1] + N[f, 2] * N[g, 2]
            if d > cos_t:
                w = A[g] * (d - cos_t) / (1.0 - cos_t)
                s0 += N[g, 0] * w; s1 += N[g, 1] * w; s2 += N[g, 2] * w
        l = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
        if l > 0:
            out[f, 0] = s0 / l; out[f, 1] = s1 / l; out[f, 2] = s2 / l
        else:
            out[f, 0] = N[f, 0]; out[f, 1] = N[f, 1]; out[f, 2] = N[f, 2]


def smooth_normals(topo, iters=8, max_angle_deg=40.0):
    """Feature-preserving (bilateral) face-normal smoothing: removes surface
    noise from AI meshes/decimation while keeping real creases."""
    N = np.ascontiguousarray(topo.face_normal.copy())
    out = np.empty_like(N)
    A = np.ascontiguousarray(topo.face_area / max(topo.face_area.mean(), 1e-300))
    c = float(np.cos(np.radians(max_angle_deg)))
    for _ in range(iters):
        _bilateral_pass(topo.FF, N, A, c, out)
        N, out = out, N
    return N


def dihedral_from_normals(topo, N):
    f0, f1 = topo.EF[:, 0], topo.EF[:, 1]
    ok = (topo.ec == 2) & (f0 >= 0) & (f1 >= 0)
    a = np.arccos(np.clip((N[np.maximum(f0, 0)] * N[np.maximum(f1, 0)]).sum(1), -1, 1))
    c1 = topo.V[topo.F[np.maximum(f1, 0)]].mean(1)
    s = ((c1 - topo.V[topo.E[:, 0]]) * N[np.maximum(f0, 0)]).sum(1)
    d = np.zeros(len(f0))
    d[ok] = np.where(s[ok] <= 0, a[ok], -a[ok])
    return d
