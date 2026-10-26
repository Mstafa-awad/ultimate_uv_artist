"""Repack existing UV islands: keep each input island's shape, rescale it to
uniform texel density and pack it with the Ultimate UV Artist packer."""
from __future__ import annotations

import time

import numpy as np

from . import metrics, pack
from .pipeline import PROFILES, UnwrapResult


def repack_existing(vertices, faces, uvs, normals=None, colors=None, atlas=4096, padding=8, profile="balanced",
                    log=print):
    t0 = time.perf_counter()
    cfg = PROFILES[profile]
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    uv = np.asarray(uvs, np.float64)
    isl, n = metrics.uv_islands(f, len(v))           # islands by shared vertex index
    charts = []
    dens = []
    mirrored = 0
    order = np.argsort(isl, kind="stable")
    st = np.searchsorted(isl[order], np.arange(n + 1))
    vert_maps = []
    for i in range(n):
        fi = order[st[i]:st[i + 1]]
        used, T = np.unique(f[fi], return_inverse=True)
        T = T.reshape(-1, 3)
        U = uv[used].copy()
        P = v[used]
        a3 = 0.5 * np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1).sum()
        sa = 0.5 * ((U[T[:, 1], 0] - U[T[:, 0], 0]) * (U[T[:, 2], 1] - U[T[:, 0], 1]) -
                    (U[T[:, 2], 0] - U[T[:, 0], 0]) * (U[T[:, 1], 1] - U[T[:, 0], 1]))
        auv = np.abs(sa).sum()
        if sa.sum() < 0:
            mirrored += 1
        s = np.sqrt(a3 / auv) if auv > 1e-30 and a3 > 0 else 1.0
        charts.append((U * s, T))
        vert_maps.append(used)
    uv_list, pinfo = pack.pack(charts, atlas=atlas, padding=padding, grid=min(cfg["grid"], atlas),
                               rotations=cfg["rotations"], iters=cfg["pack_iters"], log=log)
    out_uv = np.zeros((len(v), 2), np.float32)
    for used, u in zip(vert_maps, uv_list):
        out_uv[used] = u
    rep = {"engine": "Ultimate UV Artist (repack existing)", "charts": n, "mirrored_islands_kept": mirrored,
           "packing": pinfo, "seconds_total": time.perf_counter() - t0}
    return UnwrapResult(np.asarray(vertices, np.float32), f.astype(np.int32), out_uv,
                        np.arange(len(v), dtype=np.int64), isl.astype(np.int32), rep)
