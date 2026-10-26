"""Optional SAM 2 automatic mask generation for multi-view part evidence.

Uses Meta's SAM 2.1 (Apache-2.0 code and weights, vendored code in
``third_party/sam2``). The checkpoint is looked up in ComfyUI's
``models/sam2`` folder; if it is missing the engine uses geometry only.
"""
from __future__ import annotations

import os
import sys

import numpy as np

CKPTS = {
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
}


def find_checkpoint(name="base_plus"):
    fn = CKPTS[name][0]
    cands = []
    try:
        import folder_paths
        cands.append(os.path.join(folder_paths.models_dir, "sam2", fn))
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    comfy = os.path.abspath(os.path.join(here, "..", "..", ".."))
    cands.append(os.path.join(comfy, "models", "sam2", fn))
    extra = os.environ.get("UVA_SAM2_DIR")
    if extra:
        cands.append(os.path.join(extra, fn))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


_CACHE = {}          # name -> model kept in system RAM between runs (no VRAM held)


class SAM2Masks:
    """Callable: RGB uint8 image -> list of bool masks (finest first)."""

    def __init__(self, name="base_plus", points_per_side=32, device=None, max_area_frac=0.35):
        tp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
        if tp not in sys.path:
            sys.path.insert(0, tp)
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        ckpt = find_checkpoint(name)
        if ckpt is None:
            raise FileNotFoundError("SAM 2.1 checkpoint not found in models/sam2")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        # embedded Python has no stdlib sources, so torch.jit.script of SAM 2's
        # small image transform fails; the eager module is equivalent
        script = torch.jit.script
        torch.jit.script = lambda m, *a, **k: m
        try:
            cached = _CACHE.pop(name, None) if os.environ.get("UVA_SAM2_CACHE", "1") != "0" else None
            if cached is not None:
                self.model = cached.to(self.device)
            else:
                self.model = build_sam2(CKPTS[name][1], ckpt, device=self.device, apply_postprocessing=False)
            self.gen = SAM2AutomaticMaskGenerator(self.model, points_per_side=points_per_side, points_per_batch=32,
                                                  pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                                  box_nms_thresh=0.7, crop_n_layers=0, min_mask_region_area=0,
                                                  output_mode="binary_mask", multimask_output=True)
        finally:
            torch.jit.script = script
        self.max_area_frac = max_area_frac
        self.pps = int(points_per_side)
        self.full_grid = self.gen.point_grids

    def __call__(self, img):
        import torch
        fg = img.max(2) > 0
        obj = max(int(fg.sum()), 1)
        # prompt points only on the object (same spacing as the full grid):
        # background points only produce background masks and cost time
        H, W = fg.shape
        g = (np.arange(self.pps) + 0.5) / self.pps
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        on = fg[(pts[:, 1] * H).astype(int), (pts[:, 0] * W).astype(int)]
        self.gen.point_grids = [pts[on]] if on.sum() >= 16 else self.full_grid
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            res = self.gen.generate(img)
        masks = []
        for r in res:
            m = r["segmentation"] & fg
            a = int(m.sum())
            if a < 12 or a > self.max_area_frac * obj:
                continue          # specks and whole-object / background masks carry no part evidence
            masks.append((a, m))
        masks.sort(key=lambda x: x[0])
        return [m for _, m in masks]

    def release(self):
        """Free VRAM. The model is parked in system RAM (~0.35 GB) so the next
        run skips loading the checkpoint (UVA_SAM2_CACHE=0 disables)."""
        import torch
        if os.environ.get("UVA_SAM2_CACHE", "1") != "0":
            _CACHE[self.name] = self.model.to("cpu")
        del self.gen, self.model
        torch.cuda.empty_cache()
