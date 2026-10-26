# Licensing audit — Ultimate UV Artist

Audit date: 2026-10-25, re-checked 2026-10-26 for phase 2 (no new dependency or model was added). Scope: everything the production node imports or ships,
plus every candidate that was evaluated and rejected. Versions are the ones
installed in the tested ComfyUI portable Python 3.12 environment.

**Result: the production path uses only permissive (MIT / BSD / Apache-2.0)
code and one Apache-2.0 model checkpoint (SAM 2.1 Base+, optional), makes no
network access at run time, and puts no condition on the meshes, UVs or
textures you make with it.** Nothing here requires credit in a
game, a product or on generated assets.

This is an engineering audit of declared licences, not legal advice.

## 1. Code shipped in this package

| Component | Licence | Notes |
|---|---|---|
| `comfyui_ultimate_uv_artist/` (all files except `third_party/`) | MIT (see `LICENSE`) | Written from scratch for this project. No code copied from the old addon, UltraTex, ComfyUI, xatlas, uvunwrap, libigl, Blender or any paper's reference implementation. |
| `third_party/sam2/` | Apache-2.0 (`third_party/sam2/LICENSE`), © Meta Platforms | Unmodified copy of the `sam2` Python package + configs from https://github.com/facebookresearch/sam2 at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` (`benchmark.py`, legacy video predictor, CUDA extension, demo and training code not included). Obligation if you redistribute this package: keep the LICENSE file and notices. No obligation on outputs. The PyPI package named `sam2` is an unrelated third-party upload and is **not** used. |

## 1b. Model checkpoint

| File | Licence | Source | Use |
|---|---|---|---|
| `ComfyUI/models/sam2/sam2.1_hiera_base_plus.pt` (323,606,802 bytes, SHA-256 `cca7edab9d6b97c3e8e3ba086c9e1f1baa136643020ac37f4de0cc4cfed86daa`) | Apache-2.0 (Meta's SAM 2 release: code and checkpoints) | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt (downloaded with your approval) | Optional. Automatic multi-view mask generation for part segmentation. Loaded with `torch.load(weights_only=True)`. If it is missing the node silently uses geometry-only parts. |

Algorithms are implemented from their published descriptions (ideas are not
copyrightable; no source was copied):

| Technique | Reference |
|---|---|
| LSCM | Lévy, Petitjean, Ray, Maillot, *Least Squares Conformal Maps*, SIGGRAPH 2002 |
| ARAP parameterization | Liu, Zhang, Xu, Gotsman, Gortler, *A Local/Global Approach to Mesh Parameterization*, SGP 2008 |
| Symmetric-Dirichlet reweighting (SLIM-style) | Rabinovich, Poranne, Panozzo, Sorkine-Hornung, *Scalable Locally Injective Mappings*, TOG 2017 |
| Flip-preventing line search | Smith, Schaefer, *Bijective Parameterization with Free Boundaries*, SIGGRAPH 2015 |
| Relief seams through distortion hot spots along hidden paths | Sheffer, Hart, *Seamster*, IEEE Vis 2002 |
| Normal-cone developability measure | in the spirit of Julius, Kraevoy, Sheffer, *D-Charts*, 2005 |
| Bilateral normal filtering | Zheng et al., *Bilateral Normal Filtering for Mesh Denoising*, TVCG 2011 |
| Cut graph = complement of a dual spanning tree | Erickson, Har-Peled, *Optimally Cutting a Surface into a Disk*, 2002 |
| Raster shape packing with free-run tables | standard bitmap packing; independent implementation |

## 2. Runtime dependencies (already present in the ComfyUI Python)

| Package | Version | Licence (declared) | Evidence | Obligation |
|---|---|---|---|---|
| NumPy | 1.26.4 | BSD-3-Clause | `numpy-1.26.4.dist-info`, https://numpy.org/doc/stable/license.html | keep notice if redistributed |
| SciPy (incl. SuperLU, qhull) | 1.17.1 | BSD-3-Clause (bundled libs BSD-style) | `scipy-1.17.1.dist-info/LICENSE.txt`, https://github.com/scipy/scipy/blob/main/LICENSE.txt | keep notices if redistributed |
| Numba | 0.65.1 | BSD-2-Clause | `numba-0.65.1.dist-info/licenses/LICENSE`, https://github.com/numba/numba/blob/main/LICENSE | keep notice if redistributed |
| llvmlite (LLVM) | 0.47.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | `llvmlite-0.47.0.dist-info/licenses/`, https://github.com/numba/llvmlite/blob/main/LICENSE | keep notices if redistributed |
| fast-simplification (Fast-Quadric-Mesh-Simplification) | 0.1.13 | MIT | `fast_simplification-0.1.13.dist-info/licenses/LICENSE`, https://github.com/pyvista/fast-simplification | keep notice if redistributed |
| Pillow | 12.2.0 | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow/blob/main/LICENSE | keep notice if redistributed |
| PyTorch | 2.8.0+cu128 | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE | GPU solves (CG/ARAP), GPU topology build, SAM 2 inference |
| moderngl / glcontext | 5.12.0 / 3.0.0 | MIT | `moderngl-5.12.0.dist-info/LICENSE`, https://github.com/moderngl/moderngl | offscreen triangle-ID renders for multi-view segmentation |
| CuMesh (optional) | 1.0 | MIT | `cumesh-1.0.dist-info/licenses/` | fast GPU decimation for the planning proxy; falls back to fast-simplification. Note: GPU decimation is not bit-for-bit deterministic, so islands can differ slightly between runs |
| hydra-core | 1.3.3 | MIT | `hydra_core-1.3.3.dist-info/licenses/` | needed by `sam2.build_sam` (config loading) |
| omegaconf | 2.3.1 | BSD-3-Clause | `omegaconf-2.3.1.dist-info/licenses/` | needed by hydra/sam2 |
| iopath | 0.1.10 | MIT | `iopath-0.1.10.dist-info/licenses/` | imported by sam2 |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | `tqdm-4.70.0.dist-info/licenses/` | imported by sam2 utilities (progress bars only). MPL-2.0 is file-level and only matters if you modify and ship tqdm's own files; unmodified use is unrestricted |

These obligations apply only if *you redistribute the software itself*. None
of them applies to meshes, UVs, baked textures or games made with it.

No dependency is installed by this package (all of the above were already in
your ComfyUI Python). The only download made during this project was the
SAM 2.1 Base+ checkpoint, with your approval.

## 3. Host application

ComfyUI is GPL-3.0. Ultimate UV Artist runs inside it as a separate custom
node package through the public `comfy_api` node interface, like many
MIT-licensed custom nodes. The UV engine (`engine/`) does not import ComfyUI at
all and runs in plain Python. GPL-3.0 places no condition on output data.

UltraTex Modular (your project) is GPL-3.0-only. This package does not import,
copy or modify UltraTex code; it only produces a standard ComfyUI `MESH` that
UltraTex already accepts.

## 4. The previous addon (`comfyui_ultratex_uvunwrap_compare`)

* Engine: archonic/uvunwrap 0.2.0 wheel, **MIT** (its SBOM lists only MIT /
  Apache-2.0 / Zlib / BSD-2 / Unicode-3.0 / LLVM-exception components).
* Adapter (`uvunwrap_node.py`, `core/rust_uv.py`): **declared GPL-3.0-only**
  (it ships the GPL text and its README says so). Commercial use is allowed,
  but redistribution of the adapter carries GPL source obligations.
* Decision: no line of that adapter was reused. The new package is MIT and
  does not need uvunwrap at runtime. The old addon is left installed and
  untouched for comparison/rollback.

## 5. Baselines used only for measurement (not part of the product)

| Tool | Licence | Use |
|---|---|---|
| ComfyUI `UnwrapMesh` (`comfy_extras/nodes_mesh_postprocess.py`) — UltraTex's `comfy_gpu` mode | GPL-3.0 (ComfyUI) | baseline only, called in a separate benchmark process |
| xatlas (python binding xatlas-python 0.0.11) | MIT | baseline only |
| uvunwrap 0.2.0 | MIT | baseline only |

## 5b. Phase 2 additions (2026-10-26)

All phase-2 code is new MIT code in this package: `engine/merge.py` (soft semantic seams),
`engine/validate.py` (exact UV validation), `engine/crumbs.py`, `engine/proxyfix.py`, the stretch
control and split-candidate code in `engine/pipeline.py` / `engine/segment.py`, the packer changes in
`engine/pack.py`, the regression suite `tests/test_regression.py` and the development tools in
`scripts/` (bake test, benchmark runners, report generator). No new package, model or download.

The `uvunwrap` Rust engine (MIT wheel, already installed by the old addon) was evaluated as an
alternative **packing backend** through an adapter written for this package. It was rejected and the
adapter deleted: uvunwrap re-parameterises existing islands (residuals of 1-47 texels on valid
charts), so it cannot pack our charts without changing them. It is used only as a benchmark baseline,
as before, through the old addon's code in a separate measurement process.

## 6. Candidates evaluated and rejected for production

| Candidate | Code licence | Weights licence | Decision |
|---|---|---|---|
| GeoSAM2 (VAST-AI-Research) | Apache-2.0 | Apache-2.0 (`geosam2.pt` 588 MB / bf16 294 MB on Hugging Face) | Licence acceptable. Not adopted: it is *interactive* (needs a click or mask prompt per part), needs Blender multi-view rendering and an optional Linux CUDA extension, and segmentation is not UV unwrapping. Would also be a new 300–600 MB model download (not done without your approval). The node accepts no prompt-free automatic mode from it. |
| SAMesh (gtangg12/samesh) | **no licence file** (all rights reserved by default) | uses SAM2 | Rejected: code cannot be used. |
| ComfyUI-Sam-Mesh (3dmindscapper) | MIT wrapper | bundles SAMesh code without a licence | Rejected: provenance of bundled code unclear. |
| uvunwrap as packing backend (phase 2) | MIT | — | Rejected on quality: reshapes existing islands (see 5b). |
| SAM 3 / SAM 3.1 (`models/checkpoints/sam3.1_multiplex_fp16.safetensors`, present on disk) | custom Meta "SAM License" | custom | Rejected: not MIT/Apache; non-transferable, termination and acceptable-use clauses. Not loaded by this package. |
| SAM 2 / 2.1 (Meta) | Apache-2.0 | Apache-2.0 | **Adopted** (Base+ checkpoint, optional) — prompt-free automatic masks from 12–24 rendered views, fused on the mesh. See §1 and §1b. |
| Microsoft UVAtlas | MIT | — | Not adopted: archived C++/DirectX library, no Python binding; isochart charting tends to many charts. |
| xatlas as engine | MIT | — | Not adopted: measured 795–48,864 charts on the test meshes and 566 s on the 1.7 M-triangle fox. |
| pymeshlab (installed) | GPL-3.0 | — | Not used. |
| Blender | GPL-2.0+ (application) | — | Not used by the node. Can be used by you to inspect the exported GLBs; Blender's licence does not apply to files you make with it. |

## 7. Outputs

The node outputs your own mesh with new UV coordinates and preview images. No
licence, watermark, attribution string or metadata claim is added to the
geometry, the UVs or the GLB. Rights to the *input* meshes are separate and
unchanged.
