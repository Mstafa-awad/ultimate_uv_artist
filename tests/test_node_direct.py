"""Direct node test (no server): run the ComfyUI node classes on a GLB and
export through ComfyUI's own SaveGLB writer.

usage: python test_node_direct.py <ComfyUI dir> <mesh.glb> <out_dir> [quality] [uv_mode]
"""
import json
import os
import sys
import time

import numpy as np

comfy_dir, mesh_path, out_dir = sys.argv[1:4]
quality = sys.argv[4] if len(sys.argv) > 4 else "balanced"
uv_mode = sys.argv[5] if len(sys.argv) > 5 else "fresh_unwrap"
semantic = sys.argv[6] if len(sys.argv) > 6 else "sam2+geometry"
pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, comfy_dir)
sys.path.insert(0, pkg_parent)
sys.argv = [sys.argv[0]]
os.makedirs(out_dir, exist_ok=True)

import torch  # noqa: E402
from comfy_api.latest import Types  # noqa: E402
from comfyui_ultimate_uv_artist import nodes  # noqa: E402
from comfyui_ultimate_uv_artist.engine.meshio import read_glb  # noqa: E402
from comfy_extras.nodes_save_3d import save_glb  # noqa: E402

m = read_glb(mesh_path)
t = lambda a: None if a is None else torch.from_numpy(np.ascontiguousarray(a))[None]
mesh = Types.MESH(vertices=t(m.vertices), faces=t(m.faces.astype(np.int32)), uvs=t(m.uvs),
                  vertex_colors=t(m.colors), normals=t(m.normals))
t0 = time.perf_counter()
out = nodes.UltimateUVArtist.execute(mesh, quality, "4096", 8, uv_mode, "geometry+vertex_colors", "balanced",
                                     "uniform", 1.0, "normal", "auto", semantic)
dt = time.perf_counter() - t0
uv_mesh, orig, layout, checker, report, parts_img = out.result
rep = json.loads(report)
assert orig is mesh, "original_mesh output must be the untouched input object"
assert uv_mesh.uvs is not None and uv_mesh.uvs.shape[1] == uv_mesh.vertices.shape[1]
assert uv_mesh.faces.shape[1] == mesh.faces.shape[1], "triangle count changed"
val = rep["uv_metrics"]
exact = rep["validation"]
assert val["positions_exact"], "triangle corner positions changed"
assert exact["valid"], f"exact UV validation failed: {exact}"
assert exact["overlapping_triangles_exact"] == 0 and exact["flipped_triangles"] == 0
# attributes per corner preserved
of = uv_mesh.faces[0].numpy().astype(np.int64)
if m.colors is not None:
    assert np.array_equal(uv_mesh.vertex_colors[0].numpy()[of], m.colors[m.faces]), "vertex colours changed"
if m.normals is not None:
    assert np.array_equal(uv_mesh.normals[0].numpy()[of], m.normals[m.faces]), "normals changed"
# checker helper + ComfyUI's own GLB writer
chk_mesh, chk = nodes.UVArtistCheckerTexture.execute(uv_mesh, "2048", 32).result
from PIL import Image  # noqa: E402
tex = Image.fromarray((chk[0].numpy() * 255).astype(np.uint8))
base = os.path.join(out_dir, os.path.splitext(os.path.basename(mesh_path))[0] + f"_{quality}")
save_glb(uv_mesh.vertices[0], uv_mesh.faces[0], base + "_uv.glb", uvs=uv_mesh.uvs[0],
         vertex_colors=None if uv_mesh.vertex_colors is None else uv_mesh.vertex_colors[0],
         normals=None if uv_mesh.normals is None else uv_mesh.normals[0])
save_glb(uv_mesh.vertices[0], uv_mesh.faces[0], base + "_checker.glb", uvs=uv_mesh.uvs[0], texture_image=tex,
         normals=None if uv_mesh.normals is None else uv_mesh.normals[0])
Image.fromarray((layout[0].numpy() * 255).astype(np.uint8)).save(base + "_layout.png")
Image.fromarray((checker[0].numpy() * 255).astype(np.uint8)).save(base + "_checker_views.png")
Image.fromarray((parts_img[0].numpy() * 255).astype(np.uint8)).save(base + "_parts.png")
v_out = nodes.UVArtistValidate.execute(uv_mesh, "4096").result
# packing-only change must reuse the charts (no second segmentation)
t1 = time.perf_counter()
out2 = nodes.UltimateUVArtist.execute(mesh, quality, "2048", 8, uv_mode, "geometry+vertex_colors", "balanced",
                                      "uniform", 1.0, "normal", "auto", semantic)
rep2 = json.loads(out2.result[4])
assert uv_mode != "fresh_unwrap" or rep2.get("charting_cached") is True, "charts were not reused"
assert rep2["validation"]["valid"]
dt2 = time.perf_counter() - t1
json.dump(rep, open(base + "_report.json", "w"), indent=1)
print(json.dumps({"seconds": round(dt, 2), "repack_2048_seconds": round(dt2, 2), "islands": val["islands"], "occupancy": round(val["occupancy"], 4),
                  "overlap_texels": val["overlap_texels"], "flipped": val["flipped_triangles"],
                  "positions_exact": val["positions_exact"], "parts": rep.get("parts"),
                  "semantic": rep.get("topology", {}).get("semantic"), "glb": base + "_uv.glb"}))
print("NODE TEST PASSED")
