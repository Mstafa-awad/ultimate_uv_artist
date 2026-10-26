# Ultimate UV Artist — Phase 2 report

Date: 2026-10-26. Code: git repository in `UltimateUV_Work/` (tag `v5-baseline` = the version you
reviewed; every change below is a separate commit). Full tables: `BENCHMARK.md`. Evidence images:
`UltimateUV_Work/results/compare/` (side-by-side sheets, bake crops) and `results/ut_compare/`
(real UltraTex bakes).

## Headline numbers (balanced mode unless noted, atlas 4096, padding 4)

| Mesh | Islands NEW bal / v5 / Rust shape | Stretch p95 NEW bal / NEW artist / v5 / Rust shape | Stretch p99 NEW bal / v5 / Rust | Occupancy NEW bal / v5 / Rust shape | Bake PSNR dB NEW bal / NEW artist / v5 / Rust shape / comfy_gpu | Valid NEW / comfy_gpu | Time NEW bal / Rust shape |
|---|---|---|---|---|---|---|---|
| Chest (6.0 M tris) | 785 / 863 / 4175 | 1.114 / 1.079 / 1.155 / 1.009 | 1.235 / 1.374 / 1.034 | 78.2% / 76.8% / 77.2% | 27.56 / 27.49 / 27.50 / 27.39 / 24.90 | yes / NO | 106 s / 46 s |
| Fox (1.7 M) | 590 / 597 / 3597 | 1.214 / 1.150 / 1.508 / 1.040 | 1.482 / 2.618 / 1.136 | 68.2% / 66.9% / 64.2% | 28.89 / 28.90 / 28.68 / 28.64 / 27.92 | yes / NO | 49 s / 26 s |
| Wolf, fur (5.9 M) | 1279 / 1310 / 11477 | 1.276 / 1.200 / 1.585 / 1.022 | 1.833 / 3.122 / 1.076 | 66.4% / 65.2% / 56.5% | 28.59 / 28.73 / 28.55 / 28.27 / 27.32 | yes / NO | 95 s / 230 s |
| Tree (6.0 M) | 4817 / 6034 / 51734 | 1.122 / 1.103 / 1.168 / 1.018 | 1.468 / 2.339 / 1.061 | 59.9% / 48.6% / 34.1% | 24.72 / 24.62 / 24.08 / 22.70 / 22.92 | yes / NO | 207 s / 1710 s |
| Deer (low-poly) | 84 / 88 / 202 | 1.159 / 1.159 / 1.210 / 1.074 | 1.413 / 1.778 / 1.133 | 53.9% / 57.5% / 59.0% | 25.81 / 25.81 / 25.54 / 25.59 / 25.01 | yes / NO | 13 s / 0 s |
| Chest low 12K | 436 / 430 / 766 | 1.133 / 1.117 / 1.203 / 1.064 | 1.394 / 1.677 / 1.236 | 82.3% / 83.1% / 84.8% | 24.16 / 24.17 / 24.32 / 24.38 / 23.58 | yes / NO | 18 s / 2 s |
| Fox low 30K | 662 / 614 / 1781 | 1.241 / 1.208 / 1.490 / 1.134 | 1.652 / 2.603 / 1.303 | 67.2% / 67.0% / 70.5% | 25.87 / 25.87 / 25.78 / 25.95 / 24.94 | yes / NO | 23 s / 7 s |
| Wolf low 25K | 710 / 634 / 1872 | 1.251 / 1.221 / 1.548 / 1.128 | 1.668 / 3.109 / 1.263 | 66.4% / 66.9% / 69.8% | 25.77 / 25.77 / 25.70 / 25.81 / 24.87 | yes / NO | 25 s / 10 s |
| Tree low 60K | 5038 / 4806 / 8302 | 1.172 / 1.156 / 1.723 / 1.080 | 1.629 / 5.247 / 1.182 | 60.0% / 54.5% / 60.7% | 20.55 / 20.96 / 20.26 / 20.56 / 20.18 | yes / NO | 90 s / 159 s |

Time = cold process (includes loading torch and the SAM checkpoint). Inside a running ComfyUI the SAM
model stays in RAM between runs, and changing only atlas size / padding / density reuses the charts
(chest_low12k: 27 s first run, 4 s repack).

Peak VRAM, balanced: 2.0–2.8 GB on low-poly meshes, 2.5 GB on the 1.7 M fox, 5.1–5.2 GB on the 6 M
meshes (was 8.2–14.2 GB in the first phase-2 benchmark, i.e. spilling into shared memory on the
8 GB card). Peak RAM 2.9–7.0 GB.

## What changed, and why

1. **Stretch control (the main complaint).** I measured where stretch came from instead of adding cuts
   everywhere: 99.99 % of the high-poly stretch was already in the planning layout, coming from
   leftover pieces of failed parts (p95 2.66) and from parts being accepted at 1.25× the stretch
   budget. Now: no extra slack for parts, leftovers are re-checked and split only where the split
   lowers stretch below the budget or by at least 15 %, and every mode has an explicit budget
   (fast 1.45, balanced 1.30, artist 1.20). Fox p95 1.51 → 1.21, p99 2.62 → 1.48; wolf p95 1.59 → 1.28;
   low-poly fox/wolf 1.49/1.55 → 1.24/1.25.
2. **Soft semantic seams.** Segmentation boundaries are no longer hard cuts. Each mesh edge gets a
   confidence from the segmentation; neighbouring islands merge (also across parts) when the boundary
   is weak, there is no crease, and the merged island passes the same validation. Splits of a part
   now pick, among several candidates, the seam with the lowest crease-weighted cost (cuts follow
   creases). Chest 913 → 785 islands, fox 698 → 590, with stretch inside budget.
3. **Charting and packing separated.** `make_charts` produces a ChartSet; `pack_charts` packs it. The
   node keeps the last ChartSet, so packing options can be changed without re-segmenting.
4. **Packing.** Full-resolution search above the coarse estimate; island-count-aware time budget and a
   4-probe parallel search for meshes with thousands of islands (trees 54 % → 60 %); a
   packing-aware split: two islands that are each larger than half the atlas square cannot share a row
   or column and cap the texel density of everything — the hidden/downward one is cut straight across
   (chest_low12k 78.8 % → 82.3 %, chest 77.1 % → 78.2 %); all island orderings tried below 600 islands.
5. **Validation on the exported UVs.** New exact validator (`engine/validate.py`): exact
   triangle–triangle overlap on float32 UVs, folds, mirrored islands, slivers below float32
   resolution, range, gutters, geometry/attribute preservation. It found 22 truly overlapping
   sub-texel triangles in the chest that the texel test had missed; now the same exact test runs per
   chart before packing and on the final UVs (result in the node report). Islands are always positively
   oriented.
6. **Low-poly speed.** SAM 2 model cached in RAM between runs, lighter SAM sampling on low-poly meshes
   (same masks, 40 % faster), faster flattening kernels, threaded part unrolling.
7. **Memory.** The GPU solver built its sparse matrices from 72 M raw entries; it now builds them from
   unique edges (peak 8.0 → 2.3 GB for 6 M triangles); SAM decodes 32 prompts per batch instead of 128
   (same masks, half the memory).
8. **Artist mode.** Lower stretch budget and longer search; on low-poly meshes it also builds the
   balanced layout and keeps it when the artist layout packs to >3 % lower texel density (the deer:
   more islands, 8 % less density, −1.2 dB bake — now the balanced layout is used).

## What improved (evidence)

* **Stretch** vs the version you reviewed: better on all 9 meshes (table above, p95 and p99).
* **Bake quality** (deterministic bake test: a fixed pattern baked through the UVs and compared with
  the exact surface pattern from 8 cameras, trilinear mipmapping): best of all methods on all four
  high-poly meshes (chest +0.17 dB, fox +0.25, wolf +0.32 (artist +0.46), tree +2.0 vs Rust shape
  packing) and on the deer; better than v5 on 8 of 9 meshes.
* **Validity**: 0 exact overlaps, 0 flips, exact geometry on every mesh in every mode. UltraTex
  comfy_gpu (the current UltraTex default) fails validation on every mesh (overlaps, flips, moved
  vertices); xatlas has exact overlaps.
* **Real UltraTex bake** (your workflow, p12, identical generation/bake settings, 8192 texture, only
  the unwrap differs): texture detail identical for the three unwraps (detail 0.161 / 0.161 / 0.162);
  the new UVs have 261 islands vs 1276 (comfy_gpu) and 2465 (Rust) and 2.3× fewer seam pixels
  (75 K vs 173 K / 141 K); per seam pixel the colour step is similar (1.10× vs 1.08× / 1.09× of the
  other bakes at the same pixels), so the total visible seam contrast is about 45 % lower. See
  `results/ut_compare/ultratex_p12_side_by_side.png` and `ultratex_p12_crop.png`.
* **Memory** fits the 8 GB card now (above).

## What is worse or still loses (evidence)

* **Stretch vs Rust uvunwrap**: Rust still has lower stretch everywhere (p95 1.01–1.13), because it
  cuts 2–10× more islands. The new system trades a little stretch for far fewer seams; artist mode
  narrows the gap.
* **Low-poly occupancy and speed vs Rust**: Rust shape packing packs 0.7–5.1 points denser on all five
  low-poly meshes. Rust is much faster there: 0.04–0.3 s with fast packing, 0.5–10 s with shape packing
  (159 s on the 5 000-leaf tree), vs 13–25 s for the new system (90 s on that tree).
* **Bake quality on three low-poly meshes**: 0.04–0.22 dB below Rust shape packing (chest_low12k
  24.16 vs 24.38, fox_low 25.87 vs 25.95, wolf_low 25.77 vs 25.81).
* **chest_low12k vs v5**: −0.16 dB (24.16 vs 24.32) and −0.8 points occupancy; the stricter stretch
  budget makes islands that pack slightly worse there.
* **Deer occupancy** 53.9 % vs 57.5 % in v5 and 59 % Rust; bake quality is still the best (25.81 dB).
* **High-poly time**: 49–207 s; Rust is faster on the fox and chest, slower on the wolf and tree.

## Things I tried and rejected (measured)

* uvunwrap (MIT) as a packing backend for our charts: it re-parameterises existing islands (1–47 texel
  changes), so it cannot pack our charts unchanged.
* 45° island rotations: +1–5 % occupancy, but worse baked texture on organic meshes (deer −0.2 dB) —
  islands lose their alignment with the texel grid.
* A cheap "effective resolution" score to choose between candidates: it did not predict the bake test.
* Flipping decimation fold-overs, forced seams at sharp angles, parallel scale search for normal meshes.

## How to reproduce

`scripts/final_bench.sh v7 "balanced fast artist"` (all meshes, own process per run, RAM/VRAM sampled),
`scripts/bake_new.sh` + `scripts/bake_baselines.sh` (bake test), `scripts/make_report.py v7`
(BENCHMARK.md), `scripts/compare_all.sh v7` (sheets), `tests/test_regression.py` (regression suite),
`scripts/start_test_comfy.sh` + `scripts/ut_api.py` + `scripts/ut_compare.py` (UltraTex on port 8191).
