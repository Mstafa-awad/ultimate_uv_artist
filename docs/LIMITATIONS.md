# Known limitations — Ultimate UV Artist (phase 2)

Measured on the test meshes in `C:\Users\super\Downloads\ultratex\no texture`,
the low-poly fixtures, `deer.glb` and the UltraTex test mesh `p12.glb`. Numbers
are in `BENCHMARK.md`.

## Stretch versus island count

* Stretch is controlled per island (95th percentile within the mode's budget),
  so a few small regions can still be stretched more than the old Rust unwrap,
  which cuts many more, smaller islands. On organic meshes (fox, wolf) the new
  balanced mode reaches stretch p95 ≈1.2 with 3–6× fewer islands; Rust reaches
  ≈1.03–1.13 with far more seams. `distortion_tolerance` below 1.0 or the
  `artist` mode trade islands for lower stretch.
* Area distortion (texel density uniformity) is usually *better* than Rust's.

## Speed

* Balanced mode (cold process, RTX 5050 laptop): 13–25 s on low-poly meshes, 49 s on the
  1.7 M fox, 95–106 s on the 6 M chest and wolf, 207 s on the 6 M tree (thousands of leaf
  islands). SAM 2 segmentation is 5–25 s of that; `semantic_parts = geometry` skips it.
  The old Rust unwrap is much faster on low-poly meshes (well under a second with fast
  packing) and remains the right tool when speed matters more than seams.
* Foliage meshes with thousands of separate pieces spend most of their time packing
  (≈50 s for 5 000 islands).
* Timings on the test laptop vary up to 2× between identical runs.
* The first run after installation compiles numba kernels (10–30 s, cached).

## Packing

* The packer is a raster first-fit with a scale search. On low-poly meshes the old
  Rust shape packer fills the atlas 0.7–5 points better (fewer, larger islands are
  harder to pack). In the deterministic bake test this costs 0.04–0.22 dB on three of
  the five low-poly meshes.
* Two islands that are each larger than half the atlas square cap the texel density;
  the packer cuts the hidden or downward-facing one, but never a visible one (a cube
  keeps its six faces whole and packs less densely).
* 45° island rotations are not used: they raise occupancy but measurably lower baked
  texture quality on organic meshes.

## Artist mode

* Artist mode lowers stretch on every mesh, but its extra islands can pack less densely.
  On low-poly meshes it compares itself with the balanced layout and keeps the denser one
  if the difference exceeds 3 %; on high-poly meshes this check is skipped (it would cost
  a second full run) and artist is within ±0.15 dB of balanced in the bake test.

## Determinism

* CuMesh GPU decimation (planning proxy) is not bit-for-bit repeatable, so island
  layouts can differ slightly between runs. Validity guarantees hold either way.

## Proxy (high-poly)

* Dense meshes are planned on a ~150 K-triangle proxy and transferred back to the
  original triangles, then relaxed and validated at full resolution. Detail the
  proxy cannot represent (fur strands, thin spikes, deep engravings) can still end
  up as small islands or local stretch; the final geometry is always the
  original, never the proxy.

## Segmentation

* SAM 2 sees the mesh only through rendered views; fully enclosed parts are
  grouped by geometry only. Segmentation boundaries are soft: islands may cross a
  part boundary when the boundary is weak (low segmentation confidence, no
  crease) and the merged island stays within the stretch budget. A wrong but
  confident segmentation can still put a seam where an artist would not.

## Validation

* Exact validation runs on the exported float32 UVs. Triangles thinner than
  float32 resolution are reported as `sub_float32_slivers`, and contacts closer
  than float32 resolution as `float32_contacts`; they are not counted as overlaps
  because they cover no measurable texture area.

## Output

* The node outputs new UVs; it does not bake. Old texture maps are cleared on
  `uv_mesh`. Bake with UltraTex as usual.
* Tangents are not written into the MESH; engines and ComfyUI's `SaveGLB` compute
  MikkTSpace tangents from normals and UVs. Islands are never mirrored, so tangent
  handedness is consistent.
* `uv_mode = keep_existing` passes input UVs through unchanged and does not
  repair them.
