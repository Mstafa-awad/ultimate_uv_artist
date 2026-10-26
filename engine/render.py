"""Small CPU z-buffer renderer for previews (chart colours, checker textures)."""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _raster(Pc, faces, fcol, uv, tex, use_tex, light, W, H, img, zbuf, nrm):
    th, tw = tex.shape[0], tex.shape[1]
    for t in range(faces.shape[0]):
        i0, i1, i2 = faces[t, 0], faces[t, 1], faces[t, 2]
        x0, y0, z0 = Pc[i0, 0], Pc[i0, 1], Pc[i0, 2]
        x1, y1, z1 = Pc[i1, 0], Pc[i1, 1], Pc[i1, 2]
        x2, y2, z2 = Pc[i2, 0], Pc[i2, 1], Pc[i2, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0:
            continue
        shade = abs(nrm[t, 0] * light[0] + nrm[t, 1] * light[1] + nrm[t, 2] * light[2])
        shade = 0.25 + 0.75 * shade
        minx = max(int(min(x0, min(x1, x2))), 0); maxx = min(int(max(x0, max(x1, x2))) + 1, W - 1)
        miny = max(int(min(y0, min(y1, y2))), 0); maxy = min(int(max(y0, max(y1, y2))) + 1, H - 1)
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                cx = px + 0.5; cy = py + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / area
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z >= zbuf[py, px]:
                    continue
                zbuf[py, px] = z
                if use_tex:
                    u = w0 * uv[3 * t, 0] + w1 * uv[3 * t + 1, 0] + w2 * uv[3 * t + 2, 0]
                    v = w0 * uv[3 * t, 1] + w1 * uv[3 * t + 1, 1] + w2 * uv[3 * t + 2, 1]
                    tx = min(max(int(u * tw), 0), tw - 1); ty = min(max(int(v * th), 0), th - 1)
                    for c in range(3):
                        img[py, px, c] = tex[ty, tx, c] * (0.55 + 0.45 * shade)
                else:
                    for c in range(3):
                        img[py, px, c] = fcol[t, c] * shade


def look_dirs():
    return {"front": (0, 0, 1), "back": (0, 0, -1), "left": (-1, 0, 0), "right": (1, 0, 0),
            "top": (0, 1, 0.001), "bottom": (0, -1, 0.001), "front_left": (-0.7, 0.25, 0.7), "back_right": (0.7, 0.25, -0.7)}


def render(V, F, view=(0, 0, 1), size=768, face_colors=None, corner_uv=None, texture=None, up=(0, 1, 0)):
    """Orthographic render looking from direction ``view`` (camera at +view)."""
    V = np.asarray(V, np.float64)
    c = 0.5 * (V.min(0) + V.max(0))
    r = np.linalg.norm(V - c, axis=1).max()
    d = np.asarray(view, float); d /= np.linalg.norm(d)
    upv = np.asarray(up, float)
    xr = np.cross(upv, d)
    if np.linalg.norm(xr) < 1e-6:
        xr = np.array([1.0, 0, 0])
    xr /= np.linalg.norm(xr)
    yu = np.cross(d, xr)
    Q = V - c
    Pc = np.stack([(Q @ xr) / r * 0.48 * size + size / 2, -(Q @ yu) / r * 0.48 * size + size / 2, -(Q @ d)], 1)
    F = np.ascontiguousarray(F, np.int64)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    light = 0.6 * d + 0.4 * yu + 0.2 * xr
    light /= np.linalg.norm(light)
    img = np.full((size, size, 3), 40.0)
    zbuf = np.full((size, size), np.inf)
    use_tex = corner_uv is not None and texture is not None
    fcol = np.ascontiguousarray(face_colors if face_colors is not None else np.full((len(F), 3), 200.0), np.float64)
    uv = np.ascontiguousarray(corner_uv if use_tex else np.zeros((3, 2)), np.float64)
    tex = np.ascontiguousarray(texture if use_tex else np.zeros((1, 1, 3)), np.float64)
    _raster(Pc, F, fcol, uv, tex, use_tex, light, size, size, img, zbuf, n)
    return img.clip(0, 255).astype(np.uint8)


def grid(images, cols=4):
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(images):
        out[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
    return out
