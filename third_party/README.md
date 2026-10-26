# Third-party code

## sam2 (Meta "Segment Anything Model 2")

* Source: https://github.com/facebookresearch/sam2 at commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4` (the `sam2/` Python package and
  configs only; no CUDA extension, demo, notebooks or training code).
* Licence: Apache-2.0 (`sam2/LICENSE`). Model checkpoints released by Meta are
  also Apache-2.0. Copyright (c) Meta Platforms, Inc. and affiliates.
* Unmodified except that `benchmark.py` and `sam2_video_predictor_legacy.py`
  were not included.
* Used optionally by `engine/segment3d.py` for multi-view part masks. The UV
  engine works without it.
