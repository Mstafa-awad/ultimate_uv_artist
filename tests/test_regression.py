"""Regression suite: synthetic meshes covering the hard cases. Every result must
pass the exact validation of the exported UVs, keep geometry and triangle order
exactly, and stay within stretch / island bounds. Runs without ComfyUI and
without SAM (geometry parts), in about a minute.

usage: python -m pytest tests/test_regression.py   (or: python tests/test_regression.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import pipeline, validate  # noqa: E402


# ------------------------------------------------------------------ meshes
def grid(n, m):
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m))
    idx = np.arange(n * m).reshape(m, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    return u.ravel(), v.ravel(), F


def weld(V, F, tol=1e-9):
    key = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[idx], inv.reshape(-1)[F]


def sphere(n=48):
    u, v, F = grid(n, n // 2)
    th, ph = u * 2 * np.pi, v * np.pi
    V = np.c_[np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)]
    V, F = weld(V, F, 1e-7)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    return V.astype(np.float32), F


def torus(n=64, m=24, R=1.0, r=0.35):
    u, v, F = grid(n, m)
    a, b = u * 2 * np.pi, v * 2 * np.pi
    V = np.c_[(R + r * np.cos(b)) * np.cos(a), r * np.sin(b), (R + r * np.cos(b)) * np.sin(a)]
    V, F = weld(V, F, 1e-7)
    return V.astype(np.float32), F


def cylinder(n=48, m=12):
    u, v, F = grid(n, m)
    a = u * 2 * np.pi
    V = np.c_[np.cos(a), v * 3.0, np.sin(a)]
    V, F = weld(V, F, 1e-7)
    return V.astype(np.float32), F


def cube(k=8):
    faces = []
    Vs = []
    off = 0
    u, v, F = grid(k, k)
    for axis in range(3):
        for sgn in (-1, 1):
            P = np.zeros((len(u), 3)); P[:, axis] = sgn
            o = [i for i in range(3) if i != axis]
            P[:, o[0]] = u * 2 - 1; P[:, o[1]] = v * 2 - 1
            FF = F if sgn > 0 else F[:, ::-1]
            Vs.append(P); faces.append(FF + off); off += len(P)
    V, F2 = weld(np.concatenate(Vs), np.concatenate(faces), 1e-7)
    return V.astype(np.float32), F2


def plane_with_hole(n=40):
    u, v, F = grid(n, n)
    V = np.c_[u * 2 - 1, np.zeros_like(u), v * 2 - 1]
    c = V[F].mean(1)
    keep = np.linalg.norm(c[:, [0, 2]], axis=1) > 0.35
    F = F[keep]
    used, inv = np.unique(F, return_inverse=True)
    return V[used].astype(np.float32), inv.reshape(-1, 3)


def messy():
    """Two disconnected pieces + a duplicate triangle + a zero-area triangle +
    a non-manifold fin."""
    V1, F1 = cube(6)
    V2, F2 = sphere(24)
    V2 = V2 * 0.5 + np.array([3.0, 0, 0], np.float32)
    V = np.r_[V1, V2]; F = np.r_[F1, F2 + len(V1)]
    F = np.r_[F, F[:1]]                                  # exact duplicate triangle
    V = np.r_[V, V[:1] + 0.0, V[1:2]]                    # zero-area: collinear points
    i = len(V) - 2
    F = np.r_[F, [[0, i, i + 1]]]
    e = F1[0, :2]
    V = np.r_[V, [[0.0, 2.0, 0.0]]]
    F = np.r_[F, [[e[0], e[1], len(V) - 1]]]             # third face on an edge (non-manifold)
    return V.astype(np.float32), F.astype(np.int64)


# (mesh, max islands); stretch must stay within the profile's own budget (+5 %)
CASES = {
    "cube": (cube, 12),
    "sphere": (sphere, 12),
    "torus": (torus, 12),
    "cylinder": (cylinder, 8),
    "plane_with_hole": (plane_with_hole, 3),
    "messy": (messy, 30),
}


def _aniso_p95(V, F, U):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = p1 - p0; e2 = p2 - p0
    l1 = np.linalg.norm(e1, axis=1); a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    ok = (l1 > 1e-12) & (a3 > 1e-12)
    x2 = (e1 * e2).sum(1) / np.maximum(l1, 1e-30); y2 = 2 * a3 / np.maximum(l1, 1e-30)
    u = U[F]; d1 = u[:, 1] - u[:, 0]; d2 = u[:, 2] - u[:, 0]
    j00 = d1[:, 0] / np.where(ok, l1, 1); j10 = d1[:, 1] / np.where(ok, l1, 1)
    j01 = (d2[:, 0] - j00 * x2) / np.where(ok, y2, 1); j11 = (d2[:, 1] - j10 * x2) / np.where(ok, y2, 1)
    q = np.hypot(.5 * (j00 + j11), .5 * (j10 - j01)); r = np.hypot(.5 * (j00 - j11), .5 * (j10 + j01))
    an = (q + r) / np.maximum(np.abs(q - r), 1e-300)
    ok &= np.abs(q - r) > 0
    a, w = an[ok], a3[ok]
    o = np.argsort(a); c = np.cumsum(w[o]) / w.sum()
    return float(a[o][np.searchsorted(c, 0.95)])


def run_case(name, profile="fast"):
    make, max_islands = CASES[name]
    max_aniso = pipeline.PROFILES[profile]["max_aniso"] * 1.05
    V, F = make()
    res = pipeline.unwrap(V, F, atlas=1024, padding=4, profile=profile, semantic="geometry", log=lambda *a: None)
    rep, _ = validate.validate_uvs(res.vertices, res.faces, res.uvs, atlas=1024, padding=4, in_v=V, in_f=F)
    assert rep["valid"], f"{name}: {rep}"
    assert rep["corner_positions_exact"] and rep["triangle_count_preserved"], name
    n = rep["islands"]
    assert n <= max_islands, f"{name}: {n} islands > {max_islands}"
    an = _aniso_p95(np.asarray(res.vertices, np.float64), np.asarray(res.faces, np.int64), np.asarray(res.uvs, np.float64))
    assert an <= max_aniso, f"{name}: anisotropy p95 {an:.3f} > {max_aniso}"
    return {"islands": n, "aniso_p95": round(an, 3), "gutter": rep.get("min_gutter_texels")}


def test_cube():
    for profile in ("fast", "balanced"):
        run_case("cube", profile)


def test_sphere():
    for profile in ("fast", "balanced"):
        run_case("sphere", profile)


def test_torus():
    for profile in ("fast", "balanced"):
        run_case("torus", profile)


def test_cylinder():
    for profile in ("fast", "balanced"):
        run_case("cylinder", profile)


def test_plane_with_hole():
    for profile in ("fast", "balanced"):
        run_case("plane_with_hole", profile)


def test_messy():
    for profile in ("fast", "balanced"):
        run_case("messy", profile)


if __name__ == "__main__":
    ok = True
    for profile in ("fast", "balanced"):
        for k in CASES:
            try:
                print(profile, k, run_case(k, profile), "OK", flush=True)
            except AssertionError as exc:
                ok = False
                print(profile, k, "FAILED", exc, flush=True)
    sys.exit(0 if ok else 1)
