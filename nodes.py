"""ComfyUI nodes for Ultimate UV Artist.

Main node: ``Ultimate UV Artist`` - MESH in, UV-mapped MESH out (plus the
untouched input, a UV layout preview, a checker preview and a JSON report).
Helpers: ``UV Artist Checker Texture`` (puts a checker texture on a UV mesh so
SaveGLB writes an inspectable GLB) and ``UV Artist Validate UVs``.
"""
from __future__ import annotations

import copy
import json
import logging
import time

import numpy as np
import torch
from comfy_api.latest import io

LOG = logging.getLogger("UltimateUVArtist")


def _interrupt_check():
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _np(t, dtype=None):
    a = t.detach().cpu().numpy()
    return a.astype(dtype) if dtype is not None else a


def mesh_to_arrays(mesh):
    """Read one native ComfyUI MESH (batch of one, padded counts honoured)."""
    if mesh.vertices.ndim != 3 or mesh.vertices.shape[0] != 1 or mesh.faces.shape[0] != 1:
        raise ValueError("Ultimate UV Artist expects a single MESH (batch size 1)")
    vc, fc = getattr(mesh, "vertex_counts", None), getattr(mesh, "face_counts", None)
    n = int(vc[0]) if vc is not None else int(mesh.vertices.shape[1])
    m = int(fc[0]) if fc is not None else int(mesh.faces.shape[1])
    v = _np(mesh.vertices[0, :n], np.float32)
    f = _np(mesh.faces[0, :m]).astype(np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or (m and (f.min() < 0 or f.max() >= n)):
        raise ValueError("MESH faces must be valid triangles")
    if not np.isfinite(v).all():
        raise ValueError("MESH has non-finite vertex positions")

    def opt(name, width):
        t = getattr(mesh, name, None)
        if t is None:
            return None
        a = _np(t[0, :n], np.float32) if t.ndim == 3 else _np(t[:n], np.float32)
        if a.shape[0] != n or (width and a.shape[1] not in width) or not np.isfinite(a).all():
            LOG.warning("ignoring invalid per-vertex %s", name)
            return None
        return a

    return v, f, opt("normals", (3,)), opt("vertex_colors", (3, 4)), opt("uvs", (2,))


def build_mesh(src, v, f, uv, xref, normals, colors):
    """Copy of the input MESH with new topology. Every per-vertex attribute is
    gathered through ``xref``; UV-dependent maps are cleared (a new bake is
    required) while scalar material values are kept."""
    out = copy.copy(src)
    out.vertices = torch.from_numpy(np.ascontiguousarray(v, np.float32))[None]
    out.faces = torch.from_numpy(np.ascontiguousarray(f, np.int32))[None]
    out.uvs = torch.from_numpy(np.ascontiguousarray(uv, np.float32))[None]
    out.vertex_counts = torch.tensor([len(v)], dtype=torch.int32)
    out.face_counts = torch.tensor([len(f)], dtype=torch.int32)
    out.normals = None if normals is None else torch.from_numpy(np.ascontiguousarray(normals[xref]))[None]
    out.vertex_colors = None if colors is None else torch.from_numpy(np.ascontiguousarray(colors[xref]))[None]
    for name in ("texture", "metallic_roughness", "normal_map", "emissive", "tangents"):
        if hasattr(out, name):
            setattr(out, name, None)
    if hasattr(out, "occlusion_in_mr"):
        out.occlusion_in_mr = False
    mat = getattr(src, "material", None)
    if isinstance(mat, dict):
        out.material = {k: copy.deepcopy(x) for k, x in mat.items() if not torch.is_tensor(x)}
    return out


# last charting result: re-running the node with only packing options changed
# (atlas size, padding, texel density) repacks without re-segmenting
_CHART_CACHE = {}


def _chart_key(v, f, normals, colors, *opts):
    import hashlib
    h = hashlib.sha1()
    for a in (v, f, normals, colors):
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(opts).encode())
    return h.hexdigest()


def _budget_overrides(quality, distortion_tolerance, memory_mode):
    from .engine import pipeline
    cfg = dict(pipeline.PROFILES[quality])
    over = {k: 1 + (cfg[k] - 1) * distortion_tolerance
            for k in ("max_aniso", "max_area", "part_max_aniso", "part_max_area") if k in cfg}
    if memory_mode == "low":
        over["proxy_faces"] = int(cfg["proxy_faces"] * 0.6)
        over["seg_proxy_faces"] = int(cfg.get("seg_proxy_faces", 150_000) * 0.6)
        over["max_chart_vertices"] = 400_000
    return over


def _image(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, np.float32) / 255.0)[None]


class UltimateUVArtist(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UltimateUVArtist",
            display_name="Ultimate UV Artist",
            category="UltraTex Modular/UV",
            description=("Artist-style automatic UV unwrap: part-aware seams in hidden/creased areas, validated "
                         "low-distortion islands (no folds, no overlaps), shape-accurate packing. Geometry, triangle "
                         "order, normals and vertex colours are preserved exactly; vertices are only duplicated on "
                         "seams. New UVs need a new bake: old texture maps are cleared on the UV output."),
            inputs=[
                io.Mesh.Input("mesh"),
                io.Combo.Input("quality", options=["fast", "balanced", "artist"], default="balanced",
                               tooltip="fast: coarser planning, fewer checks. balanced: production default. "
                                       "artist: finer planning proxy, more relief seams, symmetric-Dirichlet "
                                       "refinement, longer merge search."),
                io.Combo.Input("atlas_resolution", options=["1024", "2048", "4096", "8192"], default="4096"),
                io.Int.Input("padding", default=8, min=1, max=64,
                             tooltip="Empty texels kept around every island at atlas resolution (gutter = 2x)."),
                io.Combo.Input("uv_mode", options=["fresh_unwrap", "keep_existing", "repack_existing"],
                               default="fresh_unwrap",
                               tooltip="keep_existing passes valid input UVs through; repack_existing keeps the "
                                       "input islands' shapes and repacks them."),
                io.Combo.Input("part_hints", options=["geometry+vertex_colors", "geometry"],
                               default="geometry+vertex_colors",
                               tooltip="Where natural part boundaries come from. Vertex-colour edges (clothing, "
                                       "eyes, painted regions of AI meshes) become preferred seam locations."),
                io.Combo.Input("seam_placement", options=["hide_seams", "balanced", "shortest_seams"],
                               default="balanced"),
                io.Combo.Input("texel_density", options=["uniform", "adaptive"], default="uniform",
                               tooltip="adaptive gives islands with more surface/colour detail and more exposure "
                                       "up to 1.6x linear density (and flat hidden areas less)."),
                io.Float.Input("distortion_tolerance", default=1.0, min=0.6, max=2.0, step=0.05,
                               tooltip="Scales the stretch budget of the chosen quality. Higher = fewer, larger "
                                       "islands with more stretch; lower = more islands, less stretch."),
                io.Combo.Input("memory_mode", options=["normal", "low"], default="normal",
                               tooltip="low splits very large charts before the direct solver and uses a "
                                       "smaller planning proxy."),
                io.Combo.Input("mesh_type", options=["auto", "lowpoly", "highpoly"], default="auto"),
                io.Combo.Input("semantic_parts", options=["sam2+geometry", "geometry", "off"],
                               default="sam2+geometry", optional=True,
                               tooltip="Part segmentation used to keep islands inside parts (lid, bands, rivets, "
                                       "handles...). sam2+geometry renders the mesh from many views and fuses "
                                       "SAM 2.1 masks (Apache-2.0; needs models/sam2/sam2.1_hiera_base_plus.pt, "
                                       "falls back to geometry if missing). geometry uses creases/curvature only."),
            ],
            outputs=[
                io.Mesh.Output(display_name="uv_mesh"),
                io.Mesh.Output(display_name="original_mesh"),
                io.Image.Output(display_name="uv_layout"),
                io.Image.Output(display_name="checker_preview"),
                io.String.Output(display_name="report"),
                io.Image.Output(display_name="parts_preview"),
            ],
        )

    @classmethod
    def execute(cls, mesh, quality, atlas_resolution, padding, uv_mode, part_hints, seam_placement,
                texel_density, distortion_tolerance, memory_mode, mesh_type, semantic_parts="sam2+geometry"):
        from .engine import pipeline, existing, metrics, preview, render
        t0 = time.perf_counter()
        v, f, normals, colors, old_uv = mesh_to_arrays(mesh)
        atlas = int(atlas_resolution)
        report = {"node": "Ultimate UV Artist", "uv_mode": uv_mode, "input_vertices": len(v), "input_triangles": len(f)}
        part = None
        if uv_mode == "keep_existing":
            if old_uv is None:
                raise ValueError("keep_existing: the input mesh has no UVs")
            out_mesh = mesh
            ov, of, ouv = v, f, old_uv
            report["note"] = "input UVs passed through unchanged"
        else:
            if uv_mode == "repack_existing":
                if old_uv is None:
                    raise ValueError("repack_existing: the input mesh has no UVs")
                res = existing.repack_existing(v, f, old_uv, normals, colors, atlas=atlas, padding=padding,
                                               profile=quality, log=LOG.info)
            else:
                over = _budget_overrides(quality, distortion_tolerance, memory_mode)
                hide = {"hide_seams": 1.6, "balanced": 1.0, "shortest_seams": 0.4}[seam_placement]
                sem = {"sam2+geometry": "auto", "geometry": "geometry", "off": "off"}[semantic_parts]
                key = _chart_key(v, f, normals, colors, quality, mesh_type, part_hints, hide, sorted(over.items()), sem)
                cs = _CHART_CACHE.get(key)
                if cs is None:
                    cs = pipeline.make_charts(v, f, normals=normals, colors=colors, profile=quality,
                                              mesh_type=mesh_type, color_hints=part_hints == "geometry+vertex_colors",
                                              seam_hide=hide, log=LOG.info, overrides=over, check=_interrupt_check,
                                              semantic=sem)
                    _CHART_CACHE.clear()
                    _CHART_CACHE[key] = cs
                    report["charting_cached"] = False
                else:
                    report["charting_cached"] = True
                    LOG.info("Ultimate UV Artist: reusing charts from the previous run (only packing changed)")
                res = pipeline.pack_charts(cs, atlas=atlas, padding=padding, density=texel_density, log=LOG.info)
                res = pipeline.artist_candidate(res, v, f, quality, cs, dict(
                    normals=normals, colors=colors, mesh_type=mesh_type, color_hints=part_hints == "geometry+vertex_colors",
                    seam_hide=hide, log=LOG.info, overrides=_budget_overrides("balanced", distortion_tolerance, memory_mode),
                    check=_interrupt_check, semantic=sem),
                    atlas, padding, texel_density, None, LOG.info, "auto")
                part = getattr(res, "part_of_face", None)
            ov, of, ouv = res.vertices, res.faces, res.uvs
            out_mesh = build_mesh(mesh, ov, of, ouv, res.xref, normals, colors)
            report.update(res.report)
        # validation + previews
        rep, isl = metrics.full_report(ov, of.astype(np.int64), ouv.astype(np.float64), raster_res=min(atlas, 4096),
                                       in_v=v, in_f=f)
        report["uv_metrics"] = rep
        layout = preview.layout_image(ouv.astype(np.float64), of.astype(np.int64), isl, 1024)
        tex = preview.checker_texture(1024, 32).astype(np.float64)
        cuv = ouv[of].reshape(-1, 2)
        views = [render.render(ov, of, render.look_dirs()[k], 384, corner_uv=cuv, texture=tex)
                 for k in ("front", "back", "left", "right")]
        checker = render.grid(views, 2)
        if part is not None and len(part) == len(of) and (part >= 0).any():
            pal = preview.island_colors(int(part.max()) + 1, seed=3).astype(np.float64)
            fc = np.where(part[:, None] >= 0, pal[np.maximum(part, 0)], 90.0)
            pv = [render.render(ov, of, d, 384, face_colors=fc)
                  for d in ((-0.7, 0.35, 0.7), (0.7, 0.35, -0.7), (0, 0.1, 1), (0, 1, 0.001))]
            parts_img = render.grid(pv, 2)
            report["parts"] = int(len(np.unique(part[part >= 0])))
        else:
            parts_img = np.full_like(checker, 30)
        report["new_uv_requires_rebake"] = uv_mode != "keep_existing"
        report["seconds_node"] = time.perf_counter() - t0
        text = json.dumps(report, indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))
        LOG.info("Ultimate UV Artist: %d islands, occupancy %.1f%%, %.1fs", rep["islands"], 100 * rep["occupancy"],
                 report["seconds_node"])
        return io.NodeOutput(out_mesh, mesh, _image(layout), _image(checker), text, _image(parts_img))


class UVArtistCheckerTexture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UVArtistCheckerTexture",
            display_name="UV Artist Checker Texture",
            category="UltraTex Modular/UV",
            description="Attach a high-contrast checker as base colour so SaveGLB writes a GLB that reveals "
                        "stretching and seams in Blender/Unreal. For inspection only.",
            inputs=[io.Mesh.Input("uv_mesh"),
                    io.Combo.Input("texture_size", options=["1024", "2048", "4096"], default="2048"),
                    io.Int.Input("cells", default=32, min=4, max=256)],
            outputs=[io.Mesh.Output(display_name="checker_mesh"), io.Image.Output(display_name="checker")],
        )

    @classmethod
    def execute(cls, uv_mesh, texture_size, cells):
        from .engine import preview
        if getattr(uv_mesh, "uvs", None) is None:
            raise ValueError("mesh has no UVs")
        tex = preview.checker_texture(int(texture_size), int(cells))
        out = copy.copy(uv_mesh)
        out.texture = torch.from_numpy(tex.astype(np.float32) / 255.0)[None]
        return io.NodeOutput(out, out.texture)


class UVArtistValidate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UVArtistValidate",
            display_name="UV Artist Validate UVs",
            category="UltraTex Modular/UV",
            description="Measure any MESH's UVs: islands, occupancy, overlaps, gutters, folds, distortion, seams.",
            inputs=[io.Mesh.Input("mesh"),
                    io.Combo.Input("raster_resolution", options=["1024", "2048", "4096", "8192"], default="4096")],
            outputs=[io.Image.Output(display_name="uv_layout"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, mesh, raster_resolution):
        from .engine import metrics, preview
        v, f, _, _, uv = mesh_to_arrays(mesh)
        if uv is None:
            raise ValueError("mesh has no UVs")
        res = int(raster_resolution)
        rep, isl = metrics.full_report(v, f, uv.astype(np.float64), raster_res=res)
        g, _ = metrics.min_gutter(uv.astype(np.float64), f, isl, min(res, 4096))
        rep["min_gutter_texels"] = g * res / min(res, 4096)
        img = preview.layout_image(uv.astype(np.float64), f, isl, 1024)
        return io.NodeOutput(_image(img), json.dumps(rep, indent=1, default=float))


NODES = [UltimateUVArtist, UVArtistCheckerTexture, UVArtistValidate]
