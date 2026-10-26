"""UV layout previews and checker textures."""
from __future__ import annotations
import numpy as np


def island_colors(n, seed=7):
    rng = np.random.default_rng(seed)
    h = rng.random(n)
    s = 0.45 + 0.4 * rng.random(n)
    v = 0.75 + 0.25 * rng.random(n)
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v]); g = np.choose(i, [t, v, v, q, p, p]); b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def layout_image(uv, faces, islands, res=1024, wire=True, label_img=None):
    """Islands filled with distinct colours on dark background; edges darkened."""
    from .metrics import _raster_labels
    if label_img is None or label_img.shape[0] != res:
        label_img = np.full((res, res), -1, np.int32)
        _raster_labels(np.ascontiguousarray(uv, np.float64), np.ascontiguousarray(faces, np.int64),
                       np.ascontiguousarray(islands, np.int32), res, label_img)
    n = int(islands.max()) + 1 if len(islands) else 1
    pal = island_colors(n)
    img = np.full((res, res, 3), 28, np.uint8)
    m = label_img >= 0
    img[m] = pal[label_img[m]]
    if wire:
        b = np.zeros_like(m)
        b[1:, :] |= label_img[1:, :] != label_img[:-1, :]
        b[:, 1:] |= label_img[:, 1:] != label_img[:, :-1]
        img[b & m] = (img[b & m] * 0.35).astype(np.uint8)
    # light grid every 1/8
    for k in range(1, 8):
        p = k * res // 8
        img[p, :][~m[p, :]] = 60; img[:, p][~m[:, p]] = 60
    return img


def checker_texture(res=2048, cells=32):
    """High-contrast checker with coloured quadrants and orientation marks."""
    y, x = np.mgrid[0:res, 0:res]
    c = ((x * cells // res) + (y * cells // res)) % 2
    base = np.where(c[..., None] == 1, 235, 30).astype(np.uint8).repeat(3, 2)
    tint = np.zeros((res, res, 3), np.float32)
    tint[..., 0] = x / res; tint[..., 2] = y / res; tint[..., 1] = 1 - (x + y) / (2 * res)
    out = (base.astype(np.float32) * (0.55 + 0.45 * tint)).clip(0, 255).astype(np.uint8)
    # thin lines every cell to reveal shear
    line = ((x * cells) % res < res // (cells * 10) + 1) | ((y * cells) % res < res // (cells * 10) + 1)
    out[line] = (255, 60, 40)
    return out
