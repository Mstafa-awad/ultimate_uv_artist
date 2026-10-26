# Ultimate UV Artist (ComfyUI custom node)

Automatic, artist-style UV unwrapping for AI-generated and hand-made meshes,
from low-poly (a few thousand triangles) to dense scans (millions of triangles)
on an 8 GB GPU.

## Visual comparisons

The following comparison sheets show UV part regions, UV islands, checker-texture
previews, and packed atlases. These are benchmark examples, not guarantees
for every mesh. See [BENCHMARK.md](docs/BENCHMARK.md) for the test settings
and measured metrics.

### High-poly chest — latest comparison

![High-poly treasure chest comparison of Rust uvunwrap, UltraTex comfy_gpu and Ultimate UV Artist, with UV islands, checker textures and atlas layouts](docs/images/chest_phase2_comparison.png)

### Fox character — latest comparison

![Fox character UV comparison showing old unwraps and Ultimate UV Artist, with part regions, checkerboard renders and packed atlases](docs/images/fox_phase2_comparison.jpg)

<details>
<summary>Earlier chest benchmark (before the latest refinements)</summary>

![Earlier 6-million-triangle chest comparison between Rust unwrap, ComfyUI GPU unwrap and Ultimate UV Artist](docs/images/chest_initial_comparison.jpg)

</details>

* **Part-aware, not part-bound.** The mesh is segmented into parts (lid,
  bands, rivets, handles, planks, limbs...) from geometry and, optionally,
  SAM 2.1 masks of 12–24 rendered views fused on the surface. Parts *suggest*
  seams: a part that flattens cleanly stays one island; a part that would
  stretch gets extra seams, chosen where they reduce stretch most and placed
  along creases; neighbouring islands are merged across weak part boundaries
  when the union stays valid and low-stretch.
* **Stretch budget per quality mode.** Every island must stay within the
  mode's anisotropy and area-distortion budget (fast 1.45, balanced 1.30,
  artist 1.20 at the 95th percentile of each island). Small leftovers that
  cannot meet it are split only where that measurably helps.
* **Valid by construction, checked on the final UVs.** Exact geometric
  validation of the exported float32 UVs: no overlapping triangles (exact
  triangle-triangle test, not texel sampling), no folds, no mirrored islands,
  UVs inside [0, 1], gutters of at least the requested padding. The result is
  in the report (`validation`).
* **Exact preservation.** Vertex positions, triangle order, normals and vertex
  colours are kept; vertices are only duplicated along seams.
* **GPU where it pays.** Topology, proxy decimation (CuMesh, if installed),
  batched ARAP and SAM 2 run on the GPU; charting runs multi-threaded.
* **Licence.** MIT. Uses only MIT / BSD / Apache-2.0 code and the Apache-2.0
  SAM 2.1 checkpoint. See `docs/LICENSING_AUDIT.md`. Nothing is added to your
  outputs and no credit is required.

## Install

1. Copy the folder `comfyui_ultimate_uv_artist` into `ComfyUI/custom_nodes/`
   **without** any `__pycache__` folders (they hold compiled-kernel caches
   for another module path).
2. Optional, for semantic parts: put `sam2.1_hiera_base_plus.pt` (Meta,
   Apache-2.0, 324 MB) in `ComfyUI/models/sam2/`. Without it the node uses
   geometry-only parts.
3. Restart ComfyUI. No `pip install` is needed: every dependency (numpy, scipy,
   numba, torch, moderngl, hydra-core, fast-simplification) is already part of
   the ComfyUI portable environment. The SAM 2 code ships in `third_party/sam2/`.

The first run compiles numba kernels (about 10–30 s, cached afterwards). The
SAM 2 model stays in system RAM between runs (≈0.35 GB, no VRAM held) so later
runs skip loading it.

## Nodes

**Ultimate UV Artist** (`UltraTex Modular/UV`)

| Input | Meaning |
|---|---|
| `mesh` | any ComfyUI `MESH` (batch of one) |
| `quality` | `fast` / `balanced` / `artist` (see below) |
| `atlas_resolution` | 1024–8192 |
| `padding` | empty texels around each island |
| `uv_mode` | `fresh_unwrap`, `keep_existing` (pass input UVs through), `repack_existing` (keep island shapes, repack) |
| `part_hints` | add vertex-colour edges as preferred seam places |
| `seam_placement` | `hide_seams` / `balanced` / `shortest_seams` |
| `texel_density` | `uniform` or `adaptive` (detail/visibility-weighted, 0.62–1.6× linear) |
| `distortion_tolerance` | scales the stretch budget (lower = less stretch, more islands) |
| `memory_mode` | `low` uses smaller proxies |
| `mesh_type` | `auto` / `lowpoly` / `highpoly` |
| `semantic_parts` | `sam2+geometry` (default) / `geometry` / `off` |

Outputs: `uv_mesh` (new UVs; old texture maps cleared because a new bake is
needed), `original_mesh` (untouched input), `uv_layout`, `checker_preview`,
`report` (JSON: islands, occupancy, stretch, exact validation, timings),
`parts_preview`.

**Charts are generated once.** If you re-run the node and change only
`atlas_resolution`, `padding` or `texel_density`, it reuses the charts of the
previous run and only repacks (seconds instead of a full unwrap).

**UV Artist Checker Texture**: puts a checker on a UV mesh so `SaveGLB`
writes an inspectable GLB. **UV Artist Validate UVs**: measures any mesh's UVs.

### Quality modes

| | fast | balanced (default) | artist |
|---|---|---|---|
| stretch budget (island p95 anisotropy / area) | 1.45 / 1.75 | 1.30 / 1.60 | 1.20 / 1.45 |
| SAM 2 views / prompt density (high-poly) | 12 / 16 | 16 / 24 | 24 / 32 |
| atoms / merge search | coarse | medium | fine, longer |
| packing | 1 ordering, 3 refinement steps, 8 s budget | full refinement, 20 s (all orderings below 600 islands) | 3 orderings, 60 s |

Low-poly meshes use lighter SAM sampling in every mode (their parts are large
and clear). Measured differences are in `docs/BENCHMARK.md`.

## Using it in UltraTex

In the UltraTex workflow replace the unwrap node (`UTXUnwrapMeshUVV25`,
`UTXRustUVUnwrap` or `ComfyUIMeshUVUnwrap`) with **Ultimate UV Artist** and
connect `uv_mesh` to `UTXPrepareTextureV25 → mesh`. Nothing else changes:
FLUX models, LoRAs, view generation and bake settings are untouched.
`example_workflows/` contains copies of your workflows with only that node
swapped; your original workflow files are not modified.

## Rollback

* Per workflow: open your original workflow (unchanged) — it still uses the old
  unwrap node, which is still installed.
* Remove completely: delete `ComfyUI/custom_nodes/comfyui_ultimate_uv_artist`
  and restart ComfyUI. Nothing else was changed; the SAM 2.1 checkpoint in
  `models/sam2/` can be deleted too if no other node uses it.

## Tests

* `python tests/test_regression.py` — synthetic hard cases (cube, sphere,
  torus, cylinder, plane with hole, a messy mesh with duplicate, zero-area and
  non-manifold faces) in fast and balanced mode; every result must pass the
  exact validation, keep geometry exactly and stay within the mode's stretch
  budget. About a minute, no ComfyUI or SAM needed.
* `python tests/test_node_direct.py <ComfyUI dir> <mesh.glb> <out dir>` — runs
  the node classes and ComfyUI's own GLB writer, checks preservation, exact
  validation and chart reuse.

## Documents

* `docs/PHASE2_REPORT.md`: what changed in phase 2, why, what improved, what got worse, with the numbers.
* `docs/BENCHMARK.md`: measured results against the old Rust unwrap (shape and
  fast packing), UltraTex comfy_gpu and xatlas on your meshes, including the
  deterministic bake test and the cases where the new system loses.
* `docs/LIMITATIONS.md`: known limitations.
* `docs/LICENSING_AUDIT.md`: code, dependency and checkpoint licences.
