"""Minimal, dependency-free GLB reader/writer used for tests and exports.

Reads every triangle primitive of every mesh node (node transforms are applied
only when ``apply_transforms`` is set; the default keeps local coordinates and
reports the transform). Writes one primitive with positions, optional normals,
colours, TEXCOORD_0 and an optional embedded PNG base-colour texture.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field

import numpy as np

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass
class MeshData:
    vertices: np.ndarray                      # (N,3) float32
    faces: np.ndarray                         # (F,3) int32
    normals: np.ndarray | None = None         # (N,3)
    colors: np.ndarray | None = None          # (N,3|4) float32 0..1
    uvs: np.ndarray | None = None             # (N,2) glTF convention (v down)
    uvs1: np.ndarray | None = None            # TEXCOORD_1 if present
    face_material: np.ndarray | None = None   # (F,) primitive/material index
    materials: list = field(default_factory=list)
    node_matrix: list | None = None
    extras: dict = field(default_factory=dict)


def _accessor(js, binbuf, idx):
    acc = js["accessors"][idx]
    view = js["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_CT[acc["componentType"]])
    ncomp = _NC[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    stride = view.get("byteStride", 0) or dtype.itemsize * ncomp
    if stride == dtype.itemsize * ncomp:
        arr = np.frombuffer(binbuf, dtype=dtype, count=count * ncomp, offset=offset).reshape(count, ncomp)
    else:
        raw = np.frombuffer(binbuf, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * ncomp, offset=offset)
        arr = np.lib.stride_tricks.as_strided(raw, shape=(count, ncomp * dtype.itemsize), strides=(stride, 1))
        arr = np.ascontiguousarray(arr).view(dtype).reshape(count, ncomp)
    if acc.get("normalized") and dtype != np.float32:
        arr = arr.astype(np.float32) / float(np.iinfo(dtype).max)
    return arr


def read_glb(path, apply_transforms=False) -> MeshData:
    with open(path, "rb") as fh:
        data = fh.read()
    magic, _ver, _len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    jl, _ = struct.unpack_from("<II", data, 12)
    js = json.loads(data[20:20 + jl])
    off = 20 + jl
    binbuf = b""
    if off < len(data):
        bl, _ = struct.unpack_from("<II", data, off)
        binbuf = memoryview(data)[off + 8:off + 8 + bl]
    parts = []
    node_matrix = None
    for node in js.get("nodes", []):
        if "mesh" not in node:
            continue
        mat = node.get("matrix")
        if mat is None and any(k in node for k in ("translation", "rotation", "scale")):
            mat = _trs(node)
        node_matrix = mat
        for prim in js["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            a = prim["attributes"]
            v = _accessor(js, binbuf, a["POSITION"]).astype(np.float32)
            f = (_accessor(js, binbuf, prim["indices"]).reshape(-1, 3).astype(np.int64)
                 if "indices" in prim else np.arange(len(v)).reshape(-1, 3))
            get = lambda k: _accessor(js, binbuf, a[k]).astype(np.float32) if k in a else None
            if apply_transforms and mat is not None:
                M = np.asarray(mat, np.float64).reshape(4, 4).T
                v = (v @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
            parts.append((v, f, get("NORMAL"), get("COLOR_0"), get("TEXCOORD_0"), get("TEXCOORD_1"), prim.get("material", -1)))
    if not parts:
        raise ValueError("GLB contains no triangle primitives")
    vs, fs, ns, cs, us, u1, fm = [], [], [], [], [], [], []
    base = 0
    for v, f, n, c, u, uu, m in parts:
        vs.append(v); fs.append(f + base); base += len(v)
        ns.append(n); cs.append(c); us.append(u); u1.append(uu)
        fm.append(np.full(len(f), m, np.int32))
    cat = lambda xs: np.concatenate(xs) if all(x is not None for x in xs) else None
    return MeshData(np.concatenate(vs), np.concatenate(fs).astype(np.int32), cat(ns), cat(cs), cat(us), cat(u1),
                    np.concatenate(fm), js.get("materials", []), node_matrix, js.get("asset", {}).get("extras", {}))


def _trs(node):
    t = np.asarray(node.get("translation", [0, 0, 0]), np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.asarray(node.get("scale", [1, 1, 1]), np.float64)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4); M[:3, :3] = R * s; M[:3, 3] = t
    return M.T.reshape(-1).tolist()


def write_glb(path, vertices, faces, normals=None, colors=None, uvs=None, texture_png: bytes | None = None,
              material: dict | None = None, node_matrix=None, unlit=False, face_material=None, materials=None):
    """Write a GLB. ``face_material`` + ``materials`` split the triangles into one
    primitive per material while keeping a single shared vertex buffer."""
    vertices = np.ascontiguousarray(vertices, np.float32)
    faces = np.ascontiguousarray(faces, np.uint32)
    chunks, views, accessors = [], [], []

    def add(arr, target=None, kind="VEC3", minmax=False):
        arr = np.ascontiguousarray(arr)
        offset = sum(len(c) for c in chunks)
        raw = arr.tobytes()
        raw += b"\0" * ((4 - len(raw) % 4) % 4)
        chunks.append(raw)
        v = {"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes}
        if target:
            v["target"] = target
        views.append(v)
        ct = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125}[arr.dtype]
        acc = {"bufferView": len(views) - 1, "componentType": ct, "count": int(arr.shape[0]) if kind != "SCALAR" else int(arr.size), "type": kind}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist(); acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    attrs = {"POSITION": add(vertices, 34962, "VEC3", True)}
    if normals is not None:
        n = np.asarray(normals, np.float32)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        attrs["NORMAL"] = add(n.astype(np.float32), 34962, "VEC3")
    if colors is not None:
        c = np.asarray(colors, np.float32)
        attrs["COLOR_0"] = add(c, 34962, "VEC4" if c.shape[1] == 4 else "VEC3")
    if uvs is not None:
        attrs["TEXCOORD_0"] = add(np.asarray(uvs, np.float32), 34962, "VEC2")
    js = {"asset": {"version": "2.0", "generator": "Ultimate UV Artist"}, "scene": 0, "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0}], "meshes": [{"primitives": []}], "materials": [], "accessors": accessors,
          "bufferViews": views, "buffers": []}
    if node_matrix is not None:
        js["nodes"][0]["matrix"] = list(node_matrix)
    mat = dict(material) if material else {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}, "doubleSided": True}
    if texture_png is not None:
        offset = sum(len(c) for c in chunks)
        raw = texture_png + b"\0" * ((4 - len(texture_png) % 4) % 4)
        chunks.append(raw)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(texture_png)})
        js["images"] = [{"bufferView": len(views) - 1, "mimeType": "image/png"}]
        js["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        js["textures"] = [{"source": 0, "sampler": 0}]
        mat = json.loads(json.dumps(mat))
        mat.setdefault("pbrMetallicRoughness", {})["baseColorTexture"] = {"index": 0}
        mat["pbrMetallicRoughness"]["baseColorFactor"] = [1, 1, 1, 1]
    if unlit:
        mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
        js["extensionsUsed"] = ["KHR_materials_unlit"]
    if face_material is not None and materials:
        fm = np.asarray(face_material)
        js["materials"] = [json.loads(json.dumps(m)) for m in materials]
        for mid in np.unique(fm):
            idx = add(faces[fm == mid].reshape(-1), 34963, "SCALAR")
            prim = {"attributes": attrs, "indices": idx, "mode": 4}
            if mid >= 0:
                prim["material"] = int(mid)
            js["meshes"][0]["primitives"].append(prim)
    else:
        js["materials"] = [mat]
        idx = add(faces.reshape(-1), 34963, "SCALAR")
        js["meshes"][0]["primitives"].append({"attributes": attrs, "indices": idx, "material": 0, "mode": 4})
    binary = b"".join(chunks)
    js["buffers"] = [{"byteLength": len(binary)}]
    jraw = json.dumps(js, separators=(",", ":")).encode()
    jraw += b" " * ((4 - len(jraw) % 4) % 4)
    total = 12 + 8 + len(jraw) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(jraw), 0x4E4F534A)); fh.write(jraw)
        fh.write(struct.pack("<II", len(binary), 0x004E4942)); fh.write(binary)


def png_bytes(img_u8: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()
