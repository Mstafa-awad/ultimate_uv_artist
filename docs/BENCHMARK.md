# Benchmark — Ultimate UV Artist (phase 2)

All runs: atlas 4096, padding 4 texels, same input meshes, each method in its own process on the same laptop
(RTX 5050 Laptop 8 GB, i5-13450HX, 16 GB RAM). Metrics computed by the same code for every method:
* stretch = anisotropy sigma_max/sigma_min per triangle, area-weighted percentiles; area distortion = UV/3D area ratio vs the global density;
* exact overlaps / flips / mirrored = exact geometric test on the exported float32 UVs (`engine/validate.py`);
* bake PSNR = deterministic bake test (`scripts/bake_test.py`): a fixed procedural pattern baked through the UVs into a
  1024 (low-poly) or 2048 (high-poly) texture, rendered with trilinear mipmapping from 8 fixed cameras (incl. close-ups) and
  compared with the exact surface pattern; seam band = pixels next to a UV seam; detail fidelity = 1 - relative error of
  the detail (Laplacian) band. Higher is better.
* times are wall clock and vary up to ~2x on this laptop between runs (thermal state).

## Chest (6.0 M tris)

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 1076 | 63 | 76.2% | 6.0 | 1.007 / 1.124 / 1.291 / 2.21 | 0.9% | 1.13 | 42 | 0 / 0 / 0 | yes | yes | 27.42 / 25.46 | 0.270 | 87 s | 6824 | 5196 |
| **NEW balanced** | 785 | 66 | 78.2% | 6.0 | 1.015 / 1.114 / 1.235 / 1.94 | 0.6% | 1.17 | 44 | 0 / 0 / 0 | yes | yes | 27.56 / 25.60 | 0.274 | 106 s | 6847 | 5198 |
| **NEW artist** | 783 | 119 | 77.3% | 6.0 | 1.006 / 1.079 / 1.174 / 1.77 | 0.4% | 1.07 | 54 | 0 / 0 / 0 | yes | yes | 27.49 / 25.68 | 0.277 | 173 s | 5584 | 5352 |
| previous version (v5, balanced) | 863 | 96 | 76.8% | 6.0 | 1.007 / 1.155 / 1.374 / 2.12 | 1.6% | 1.13 | 50 | 0 / 0 / 0 | yes | yes | 27.50 / 25.81 | 0.273 | 94 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 4175 | 268 | 77.2% | 6.0 | 1.000 / 1.009 / 1.034 / 1.26 | 0.1% | 1.32 | 73 | 0 / 0 / 0 | yes | yes | 27.39 / 25.84 | 0.265 | 46 s | 2644 | - |
| Rust uvunwrap, fast packing (old default) | 4175 | 268 | 48.7% | 6.0 | 1.000 / 1.009 / 1.034 / 1.26 | 0.1% | 1.32 | 73 | 0 / 0 / 0 | yes | yes | 26.39 / 24.99 | 0.235 | 33 s | 2632 | - |
| UltraTex comfy_gpu (current UltraTex default) | 12643 | 1850 | 48.3% | 0.0 | 1.000 / 1.072 / 1.257 / 1.57 | 0.7% | 1.19 | 381 | 94915 / 82 / 5858 | NO | NO | 24.90 / 21.54 | 0.084 | 101 s | 5376 | - |

NEW balanced stage times: segmentation 12.2 s, mesh processing 3.8 s, chart generation 28.1 s, distortion refinement 33.1 s, packing 14.8 s, validation 5.3 s, total 106 s

## Fox (1.7 M)

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 501 | 114 | 63.3% | 6.0 | 1.048 / 1.261 / 1.592 / 3.21 | 3.7% | 1.25 | 56 | 0 / 0 / 0 | yes | yes | 28.76 / 26.31 | 0.310 | 36 s | 3973 | 2456 |
| **NEW balanced** | 590 | 140 | 68.2% | 6.0 | 1.041 / 1.214 / 1.482 / 2.78 | 2.5% | 1.20 | 60 | 0 / 0 / 0 | yes | yes | 28.89 / 26.52 | 0.315 | 49 s | 4086 | 2456 |
| **NEW artist** | 689 | 203 | 66.2% | 6.0 | 1.033 / 1.150 / 1.338 / 2.21 | 1.3% | 1.14 | 70 | 0 / 0 / 0 | yes | yes | 28.90 / 26.51 | 0.313 | 107 s | 4022 | 2598 |
| previous version (v5, balanced) | 597 | 124 | 66.9% | 6.0 | 1.052 / 1.508 / 2.618 / 10.82 | 9.7% | 1.46 | 60 | 0 / 0 / 0 | yes | yes | 28.68 / 26.41 | 0.309 | 37 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 3597 | 651 | 64.2% | 6.0 | 1.005 / 1.040 / 1.136 / 1.91 | 0.4% | 1.70 | 110 | 0 / 0 / 0 | yes | yes | 28.64 / 26.76 | 0.297 | 26 s | 688 | - |
| Rust uvunwrap, fast packing (old default) | 3597 | 651 | 47.5% | 6.0 | 1.005 / 1.040 / 1.136 / 1.91 | 0.4% | 1.70 | 110 | 0 / 0 / 0 | yes | yes | 28.09 / 26.41 | 0.279 | 6 s | 688 | - |
| UltraTex comfy_gpu (current UltraTex default) | 4083 | 1513 | 47.0% | 0.0 | 1.023 / 1.209 / 1.407 / 1.75 | 2.2% | 1.20 | 191 | 1077 / 78 / 1691 | NO | NO | 27.92 / 26.55 | 0.260 | 20 s | 2566 | - |
| xatlas | 54650 | 1546 | 34.2% | 4.0 | 1.005 / 1.124 / 1.500 / 4.11 | 1.8% | 1.12 | 396 | 254 / 0 / 24502 | yes | NO | 26.34 / 25.14 | 0.115 | 566 s | 775 | - |

NEW balanced stage times: segmentation 7.4 s, mesh processing 2.9 s, chart generation 19.8 s, distortion refinement 5.3 s, packing 7.5 s, validation 1.5 s, total 49 s

## Wolf, fur (5.9 M)

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 1485 | 218 | 65.6% | 6.0 | 1.056 / 1.367 / 2.162 / 16.57 | 6.8% | 1.30 | 74 | 0 / 0 / 0 | yes | yes | 28.65 / 26.14 | 0.305 | 94 s | 6865 | 5132 |
| **NEW balanced** | 1279 | 217 | 66.4% | 6.0 | 1.050 / 1.276 / 1.833 / 6.76 | 4.4% | 1.25 | 74 | 0 / 0 / 0 | yes | yes | 28.59 / 26.16 | 0.305 | 95 s | 6962 | 5134 |
| **NEW artist** | 1371 | 270 | 66.4% | 6.0 | 1.039 / 1.200 / 1.593 / 4.61 | 2.6% | 1.17 | 80 | 0 / 0 / 0 | yes | yes | 28.73 / 26.41 | 0.309 | 172 s | 6914 | 5142 |
| previous version (v5, balanced) | 1310 | 214 | 65.2% | 6.0 | 1.061 / 1.585 / 3.122 / 17.53 | 11.5% | 1.52 | 74 | 29 / 0 / 238 | yes | NO | 28.55 / 26.04 | 0.299 | 92 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 11477 | 1426 | 56.5% | 6.0 | 1.003 / 1.022 / 1.076 / 1.44 | 0.2% | 1.48 | 168 | 0 / 0 / 0 | yes | yes | 28.27 / 26.49 | 0.275 | 230 s | 2202 | - |
| Rust uvunwrap, fast packing (old default) | 11477 | 1426 | 41.1% | 6.0 | 1.003 / 1.022 / 1.076 / 1.44 | 0.2% | 1.48 | 168 | 0 / 0 / 0 | yes | yes | 27.62 / 26.10 | 0.251 | 31 s | 2201 | - |
| UltraTex comfy_gpu (current UltraTex default) | 8030 | 2608 | 43.7% | 0.0 | 1.002 / 1.095 / 1.262 / 1.59 | 0.7% | 1.25 | 267 | 8000 / 418 / 2191 | NO | NO | 27.32 / 25.85 | 0.216 | 70 s | 5516 | - |

NEW balanced stage times: segmentation 7.5 s, mesh processing 3.6 s, chart generation 26.9 s, distortion refinement 26.2 s, packing 16.6 s, validation 5.6 s, total 95 s

## Tree (6.0 M)

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 6566 | 764 | 54.9% | 6.0 | 1.017 / 1.144 / 1.896 / 40.14 | 2.4% | 1.12 | 148 | 0 / 0 / 2 | yes | yes | 24.32 / 22.45 | 0.197 | 190 s | 6951 | 5212 |
| **NEW balanced** | 4817 | 718 | 59.9% | 6.0 | 1.015 / 1.122 / 1.468 / 9.73 | 1.5% | 1.10 | 142 | 0 / 0 / 1 | yes | yes | 24.72 / 23.04 | 0.210 | 207 s | 6979 | 5220 |
| **NEW artist** | 3561 | 655 | 60.9% | 6.0 | 1.012 / 1.103 / 1.416 / 9.44 | 1.3% | 1.09 | 138 | 0 / 0 / 1 | yes | yes | 24.62 / 22.83 | 0.213 | 281 s | 7015 | 5228 |
| previous version (v5, balanced) | 6034 | 906 | 48.6% | 6.0 | 1.015 / 1.168 / 2.339 / 41.96 | 3.0% | 1.13 | 159 | 155 / 0 / 3 | yes | NO | 24.08 / 22.22 | 0.180 | 161 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 51734 | 3152 | 34.1% | 6.0 | 1.001 / 1.018 / 1.061 / 1.33 | 0.1% | 1.16 | 265 | 0 / 0 / 0 | yes | yes | 22.70 / 20.66 | 0.082 | 1710 s | 2419 | - |
| Rust uvunwrap, fast packing (old default) | 51734 | 3152 | 28.4% | 6.0 | 1.001 / 1.018 / 1.061 / 1.33 | 0.1% | 1.16 | 265 | 0 / 0 / 0 | yes | yes | 22.19 / 20.27 | 0.058 | 34 s | 2222 | - |
| UltraTex comfy_gpu (current UltraTex default) | 29057 | 2868 | 47.0% | 0.0 | 1.001 / 1.084 / 1.239 / 1.60 | 0.6% | 1.25 | 355 | 28090 / 329 / 9207 | yes | NO | 22.92 / 20.86 | 0.088 | 139 s | 6692 | - |

NEW balanced stage times: segmentation 12.7 s, mesh processing 3.7 s, chart generation 60.4 s, distortion refinement 50.8 s, packing 65.7 s, validation 6.3 s, total 207 s

## Deer (low-poly)

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 76 | 16 | 55.2% | 6.0 | 1.044 / 1.167 / 1.452 / 2.93 | 1.8% | 1.12 | 22 | 0 / 0 / 0 | yes | yes | 25.87 / 24.31 | 0.220 | 12 s | 2981 | 2658 |
| **NEW balanced** | 84 | 20 | 53.9% | 6.0 | 1.042 / 1.159 / 1.413 / 2.88 | 1.5% | 1.12 | 23 | 0 / 0 / 0 | yes | yes | 25.81 / 24.68 | 0.215 | 13 s | 2998 | 2658 |
| **NEW artist** | 84 | 20 | 53.9% | 6.0 | 1.042 / 1.159 / 1.413 / 2.88 | 1.5% | 1.12 | 23 | 0 / 0 / 0 | yes | yes | 25.81 / 24.69 | 0.216 | 26 s | 3045 | 2638 |
| previous version (v5, balanced) | 88 | 26 | 57.5% | 6.0 | 1.031 / 1.210 / 1.778 / 5.83 | 3.5% | 1.13 | 26 | 0 / 0 / 0 | yes | yes | 25.54 / 23.95 | 0.210 | 17 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 202 | 27 | 59.0% | 6.0 | 1.022 / 1.074 / 1.133 / 1.25 | 0.1% | 1.28 | 28 | 0 / 0 / 0 | yes | yes | 25.59 / 24.23 | 0.215 | 0 s | 55 | - |
| Rust uvunwrap, fast packing (old default) | 202 | 27 | 40.2% | 6.0 | 1.022 / 1.074 / 1.133 / 1.25 | 0.1% | 1.28 | 28 | 0 / 0 / 0 | yes | yes | 24.51 / 23.35 | 0.187 | 0 s | 46 | - |
| UltraTex comfy_gpu (current UltraTex default) | 203 | 73 | 48.2% | 0.0 | 1.029 / 1.306 / 1.574 / 2.00 | 5.1% | 1.24 | 44 | 54 / 0 / 72 | NO | NO | 25.01 / 23.41 | 0.196 | 6 s | 1850 | - |
| xatlas | 271 | 44 | 53.0% | 6.0 | 1.038 / 1.081 / 1.134 / 1.25 | 0.1% | 1.26 | 34 | 4 / 0 / 141 | yes | NO | 25.03 / 24.47 | 0.212 | 2 s | 60 | - |

NEW balanced stage times: segmentation 3.7 s, mesh processing 0.3 s, chart generation 2.0 s, distortion refinement 0.0 s, packing 2.9 s, validation 0.0 s, total 13 s

## Chest low 12K

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 426 | 55 | 78.8% | 6.0 | 1.004 / 1.161 / 1.469 / 2.89 | 2.1% | 1.08 | 26 | 0 / 0 / 0 | yes | yes | 24.05 / 22.68 | 0.197 | 15 s | 2797 | 2824 |
| **NEW balanced** | 436 | 60 | 82.3% | 5.2 | 1.003 / 1.133 / 1.394 / 2.85 | 1.5% | 1.06 | 27 | 0 / 0 / 0 | yes | yes | 24.16 / 22.70 | 0.197 | 18 s | 2878 | 2824 |
| **NEW artist** | 444 | 62 | 82.3% | 6.0 | 1.003 / 1.117 / 1.385 / 3.45 | 1.4% | 1.05 | 28 | 0 / 0 / 0 | yes | yes | 24.17 / 22.76 | 0.198 | 32 s | 2858 | 2804 |
| previous version (v5, balanced) | 430 | 60 | 83.1% | 6.0 | 1.004 / 1.203 / 1.677 / 3.67 | 3.2% | 1.10 | 25 | 0 / 0 / 99 | yes | yes | 24.32 / 22.97 | 0.198 | 24 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 766 | 79 | 84.8% | 5.0 | 1.001 / 1.064 / 1.236 / 1.47 | 0.5% | 1.16 | 33 | 0 / 0 / 0 | yes | yes | 24.38 / 22.85 | 0.200 | 2 s | 59 | - |
| Rust uvunwrap, fast packing (old default) | 766 | 79 | 53.8% | 6.0 | 1.001 / 1.064 / 1.236 / 1.47 | 0.5% | 1.16 | 33 | 0 / 0 / 0 | yes | yes | 23.22 / 21.60 | 0.165 | 0 s | 46 | - |
| UltraTex comfy_gpu (current UltraTex default) | 493 | 150 | 66.9% | 0.0 | 1.003 / 1.106 / 1.465 / 5.73 | 1.6% | 1.37 | 41 | 290 / 141 / 93 | NO | NO | 23.58 / 21.89 | 0.173 | 4 s | 1833 | - |
| xatlas | 801 | 120 | 71.1% | 6.0 | 1.007 / 1.060 / 1.208 / 1.68 | 0.4% | 1.08 | 41 | 47 / 0 / 434 | yes | NO | 23.72 / 22.49 | 0.166 | 1 s | 60 | - |

NEW balanced stage times: segmentation 5.1 s, mesh processing 0.3 s, chart generation 2.2 s, distortion refinement 0.0 s, packing 5.3 s, validation 0.0 s, total 18 s

## Fox low 30K

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 638 | 192 | 67.2% | 6.0 | 1.042 / 1.285 / 1.799 / 4.18 | 4.6% | 1.16 | 53 | 0 / 0 / 0 | yes | yes | 25.88 / 24.52 | 0.219 | 21 s | 2841 | 1990 |
| **NEW balanced** | 662 | 204 | 67.2% | 6.0 | 1.038 / 1.241 / 1.652 / 3.89 | 3.4% | 1.13 | 55 | 0 / 0 / 0 | yes | yes | 25.87 / 24.43 | 0.218 | 23 s | 2918 | 1990 |
| **NEW artist** | 708 | 238 | 66.4% | 6.0 | 1.034 / 1.208 / 1.573 / 3.14 | 3.0% | 1.12 | 58 | 0 / 0 / 0 | yes | yes | 25.87 / 24.47 | 0.216 | 53 s | 2909 | 1970 |
| previous version (v5, balanced) | 614 | 187 | 67.0% | 6.0 | 1.048 / 1.490 / 2.603 / 11.83 | 8.9% | 1.30 | 52 | 0 / 0 / 86 | yes | yes | 25.78 / 24.42 | 0.215 | 23 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 1781 | 388 | 70.5% | 5.0 | 1.022 / 1.134 / 1.303 / 2.16 | 1.0% | 1.33 | 77 | 0 / 0 / 0 | yes | yes | 25.95 / 24.60 | 0.211 | 7 s | 74 | - |
| Rust uvunwrap, fast packing (old default) | 1781 | 388 | 49.4% | 6.0 | 1.022 / 1.134 / 1.303 / 2.16 | 1.0% | 1.33 | 77 | 0 / 0 / 0 | yes | yes | 25.09 / 23.81 | 0.185 | 0 s | 54 | - |
| UltraTex comfy_gpu (current UltraTex default) | 1027 | 607 | 51.2% | 0.0 | 1.019 / 1.254 / 1.606 / 3.59 | 3.8% | 1.41 | 87 | 624 / 119 / 219 | yes | NO | 24.94 / 23.76 | 0.176 | 2 s | 1859 | - |
| xatlas | 1596 | 422 | 55.4% | 5.0 | 1.018 / 1.141 / 1.349 / 2.15 | 1.3% | 1.23 | 80 | 52 / 0 / 802 | yes | NO | 25.25 / 23.99 | 0.192 | 3 s | 64 | - |

NEW balanced stage times: segmentation 3.7 s, mesh processing 0.4 s, chart generation 7.2 s, distortion refinement 0.2 s, packing 7.3 s, validation 0.0 s, total 23 s

## Wolf low 25K

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 694 | 235 | 66.4% | 5.0 | 1.040 / 1.288 / 1.765 / 3.75 | 4.7% | 1.15 | 57 | 0 / 0 / 0 | yes | yes | 25.76 / 24.33 | 0.213 | 22 s | 2869 | 1990 |
| **NEW balanced** | 710 | 246 | 66.4% | 6.0 | 1.036 / 1.251 / 1.668 / 3.95 | 3.7% | 1.13 | 58 | 0 / 0 / 0 | yes | yes | 25.77 / 24.35 | 0.213 | 25 s | 2886 | 1990 |
| **NEW artist** | 739 | 267 | 65.7% | 5.0 | 1.031 / 1.221 / 1.679 / 3.63 | 3.3% | 1.11 | 61 | 0 / 0 / 0 | yes | yes | 25.77 / 24.38 | 0.214 | 64 s | 2913 | 1970 |
| previous version (v5, balanced) | 634 | 211 | 66.9% | 6.0 | 1.051 / 1.548 / 3.109 / 18.02 | 9.8% | 1.33 | 53 | 0 / 0 / 81 | yes | yes | 25.70 / 24.18 | 0.208 | 23 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 1872 | 468 | 69.8% | 5.0 | 1.020 / 1.128 / 1.263 / 1.85 | 0.7% | 1.27 | 83 | 0 / 0 / 0 | yes | yes | 25.81 / 24.44 | 0.204 | 10 s | 76 | - |
| Rust uvunwrap, fast packing (old default) | 1872 | 468 | 47.3% | 5.0 | 1.020 / 1.128 / 1.263 / 1.85 | 0.7% | 1.27 | 83 | 0 / 0 / 0 | yes | yes | 24.84 / 23.62 | 0.174 | 0 s | 54 | - |
| UltraTex comfy_gpu (current UltraTex default) | 946 | 567 | 50.6% | 0.0 | 1.020 / 1.263 / 1.692 / 4.46 | 4.1% | 1.46 | 86 | 818 / 113 / 179 | NO | NO | 24.87 / 23.69 | 0.170 | 2 s | 1876 | - |
| xatlas | 1743 | 509 | 54.7% | 5.0 | 1.016 / 1.146 / 1.352 / 2.10 | 1.3% | 1.20 | 87 | 97 / 0 / 878 | yes | NO | 25.14 / 23.91 | 0.184 | 3 s | 59 | - |

NEW balanced stage times: segmentation 3.6 s, mesh processing 0.4 s, chart generation 7.3 s, distortion refinement 0.2 s, packing 9.1 s, validation 0.0 s, total 25 s

## Tree low 60K

| method | islands | 90% area in | occupancy | min gutter (texels) | stretch p50 / p95 / p99 / p99.9 | area >1.3x stretch | area dist. p95 | seam len | exact overlaps / flips / mirrored | geometry exact | valid | bake PSNR all / seam band (dB) | detail fidelity | time | peak RAM MiB | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NEW fast** | 4962 | 605 | 60.0% | 5.0 | 1.019 / 1.201 / 1.749 / 7.47 | 3.1% | 1.13 | 82 | 0 / 0 / 0 | yes | yes | 20.62 / 19.79 | 0.057 | 74 s | 3168 | 2218 |
| **NEW balanced** | 5038 | 632 | 60.0% | 5.2 | 1.018 / 1.172 / 1.629 / 6.49 | 2.4% | 1.11 | 83 | 0 / 0 / 0 | yes | yes | 20.55 / 19.72 | 0.055 | 90 s | 3202 | 2218 |
| **NEW artist** | 5084 | 661 | 65.4% | 5.3 | 1.017 / 1.156 / 1.554 / 5.49 | 2.2% | 1.10 | 85 | 0 / 0 / 0 | yes | yes | 20.96 / 20.10 | 0.066 | 277 s | 3430 | 2198 |
| previous version (v5, balanced) | 4806 | 605 | 54.5% | 5.0 | 1.025 / 1.723 / 5.247 / 153.58 | 9.8% | 1.44 | 83 | 0 / 0 / 1649 | yes | yes | 20.26 / 19.36 | 0.039 | 61 s | - | - |
| Rust uvunwrap, shape packing (old, best) | 8302 | 964 | 60.7% | 5.0 | 1.007 / 1.080 / 1.182 / 1.69 | 0.5% | 1.20 | 113 | 0 / 0 / 0 | yes | yes | 20.56 / 19.50 | 0.041 | 159 s | 107 | - |
| Rust uvunwrap, fast packing (old default) | 8302 | 964 | 42.2% | 6.0 | 1.007 / 1.080 / 1.182 / 1.69 | 0.5% | 1.20 | 113 | 0 / 0 / 0 | yes | yes | 19.44 / 18.55 | -0.004 | 0 s | 69 | - |
| UltraTex comfy_gpu (current UltraTex default) | 4017 | 966 | 56.3% | 0.0 | 1.006 / 1.162 / 2.570 / 30.84 | 2.7% | 1.24 | 92 | 6707 / 2590 / 491 | yes | NO | 20.18 / 19.26 | 0.026 | 11 s | 1876 | - |
| xatlas | 6540 | 882 | 54.9% | 5.0 | 1.010 / 1.075 / 1.202 / 1.68 | 0.5% | 1.11 | 111 | 330 / 0 / 3242 | yes | NO | 20.28 / 19.47 | 0.034 | 11 s | 78 | - |

NEW balanced stage times: segmentation 5.6 s, mesh processing 0.6 s, chart generation 26.0 s, distortion refinement 4.1 s, packing 49.1 s, validation 0.1 s, total 90 s

## Where the new system loses (valid competitors only)

* Chest (6.0 M tris): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.01 vs 1.11)
* Chest (6.0 M tris): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.03 vs 1.23)
* Chest (6.0 M tris): Rust uvunwrap, shape packing (old, best) beats NEW balanced on seam PSNR (25.8 vs 25.6)
* Chest (6.0 M tris): Rust uvunwrap, shape packing (old, best) beats NEW balanced on time (46 vs 106)
* Chest (6.0 M tris): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.01 vs 1.11)
* Chest (6.0 M tris): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.03 vs 1.23)
* Chest (6.0 M tris): Rust uvunwrap, fast packing (old default) beats NEW balanced on time (32.8 vs 106)
* Fox (1.7 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.04 vs 1.21)
* Fox (1.7 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.14 vs 1.48)
* Fox (1.7 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on seam PSNR (26.8 vs 26.5)
* Fox (1.7 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on time (26.1 vs 48.8)
* Fox (1.7 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.04 vs 1.21)
* Fox (1.7 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.14 vs 1.48)
* Fox (1.7 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on time (6.09 vs 48.8)
* Wolf, fur (5.9 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.02 vs 1.28)
* Wolf, fur (5.9 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.08 vs 1.83)
* Wolf, fur (5.9 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on seam PSNR (26.5 vs 26.2)
* Wolf, fur (5.9 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.02 vs 1.28)
* Wolf, fur (5.9 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.08 vs 1.83)
* Wolf, fur (5.9 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on time (31.1 vs 94.7)
* Tree (6.0 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.02 vs 1.12)
* Tree (6.0 M): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.06 vs 1.47)
* Tree (6.0 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.02 vs 1.12)
* Tree (6.0 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.06 vs 1.47)
* Tree (6.0 M): Rust uvunwrap, fast packing (old default) beats NEW balanced on time (33.5 vs 207)
* Deer (low-poly): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.07 vs 1.16)
* Deer (low-poly): Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.13 vs 1.41)
* Deer (low-poly): Rust uvunwrap, shape packing (old, best) beats NEW balanced on occupancy (0.59 vs 0.539)
* Deer (low-poly): Rust uvunwrap, shape packing (old, best) beats NEW balanced on time (0.485 vs 13.4)
* Deer (low-poly): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.07 vs 1.16)
* Deer (low-poly): Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.13 vs 1.41)
* Deer (low-poly): Rust uvunwrap, fast packing (old default) beats NEW balanced on time (0.0388 vs 13.4)
* Chest low 12K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.06 vs 1.13)
* Chest low 12K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.24 vs 1.39)
* Chest low 12K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on occupancy (0.848 vs 0.823)
* Chest low 12K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on bake PSNR (24.4 vs 24.2)
* Chest low 12K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on seam PSNR (22.9 vs 22.7)
* Chest low 12K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on time (1.66 vs 18)
* Chest low 12K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.06 vs 1.13)
* Chest low 12K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.24 vs 1.39)
* Chest low 12K: Rust uvunwrap, fast packing (old default) beats NEW balanced on time (0.0946 vs 18)
* Fox low 30K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.13 vs 1.24)
* Fox low 30K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.3 vs 1.65)
* Fox low 30K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on occupancy (0.705 vs 0.672)
* Fox low 30K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on seam PSNR (24.6 vs 24.4)
* Fox low 30K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on time (7.14 vs 23.5)
* Fox low 30K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.13 vs 1.24)
* Fox low 30K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.3 vs 1.65)
* Fox low 30K: Rust uvunwrap, fast packing (old default) beats NEW balanced on time (0.103 vs 23.5)
* Wolf low 25K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.13 vs 1.25)
* Wolf low 25K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.26 vs 1.67)
* Wolf low 25K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on occupancy (0.698 vs 0.664)
* Wolf low 25K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on time (10.1 vs 25.2)
* Wolf low 25K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.13 vs 1.25)
* Wolf low 25K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.26 vs 1.67)
* Wolf low 25K: Rust uvunwrap, fast packing (old default) beats NEW balanced on time (0.106 vs 25.2)
* Tree low 60K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p95 (1.08 vs 1.17)
* Tree low 60K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on stretch p99 (1.18 vs 1.63)
* Tree low 60K: Rust uvunwrap, shape packing (old, best) beats NEW balanced on occupancy (0.607 vs 0.6)
* Tree low 60K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p95 (1.08 vs 1.17)
* Tree low 60K: Rust uvunwrap, fast packing (old default) beats NEW balanced on stretch p99 (1.18 vs 1.63)
* Tree low 60K: Rust uvunwrap, fast packing (old default) beats NEW balanced on time (0.29 vs 90)

