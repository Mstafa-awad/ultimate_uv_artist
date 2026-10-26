"""Multi-view rendering with exact triangle-ID buffers (OpenGL via moderngl,
MIT). Used to lift 2D segmentation masks back onto the original triangles."""
from __future__ import annotations

import numpy as np

_VS = """
#version 430
uniform mat4 mvp;
in vec3 in_pos;
out vec3 wpos;
void main() { wpos = in_pos; gl_Position = mvp * vec4(in_pos, 1.0); }
"""
_FS = """
#version 430
uniform vec3 eye;
uniform vec3 key;
in vec3 wpos;
layout(location=0) out vec4 color;
layout(location=1) out int fid;
void main() {
    vec3 n = normalize(cross(dFdx(wpos), dFdy(wpos)));
    vec3 v = normalize(eye - wpos);
    if (dot(n, v) < 0.0) n = -n;
    float diff = max(dot(n, normalize(key)), 0.0);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
    float hemi = 0.5 + 0.5 * n.y;
    float c = 0.18 + 0.55 * diff + 0.22 * hemi - 0.25 * rim;
    color = vec4(vec3(clamp(c, 0.0, 1.0)), 1.0);
    fid = gl_PrimitiveID + 1;
}
"""


def look_at(eye, target, up):
    f = target - eye; f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect; M[1, 1] = f
    M[2, 2] = (far + near) / (near - far); M[2, 3] = 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


def default_views(zoom=True):
    """(azimuth, elevation[, zoom]): a ring, a high ring, top, bottom, and a
    closer ring for small details."""
    v = [(a, 12.0) for a in range(0, 360, 45)]
    v += [(a + 22.5, 48.0) for a in range(0, 360, 90)]
    v += [(0.0, 89.0), (0.0, -70.0), (45.0, -25.0), (225.0, -25.0)]
    if zoom:
        v += [(a + 22.5, 25.0, 1.9) for a in range(0, 360, 45)]
    return v


class Renderer:
    def __init__(self, V, F, size=1024, fov=30.0):
        import moderngl
        self.ctx = moderngl.create_standalone_context(require=430)
        self.size = size
        self.fov = fov
        V = np.ascontiguousarray(V, np.float32)
        F = np.ascontiguousarray(F, np.int32)
        self.center = 0.5 * (V.min(0) + V.max(0))
        self.radius = float(np.linalg.norm(V - self.center, axis=1).max())
        ctx = self.ctx
        self.prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        vbo = ctx.buffer(V.tobytes())
        ibo = ctx.buffer(F.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos")], ibo)
        self.col = ctx.texture((size, size), 4, dtype="f1")
        self.ids = ctx.texture((size, size), 1, dtype="i4")
        self.dep = ctx.depth_texture((size, size))
        self.fbo = ctx.framebuffer([self.col, self.ids], self.dep)

    def render(self, az, el, zoom=1.0):
        ctx = self.ctx
        a, e = np.radians(az), np.radians(el)
        d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        dist = self.radius / np.sin(np.radians(self.fov) / 2) * 1.02 / max(zoom, 1e-3)
        eye = self.center + d * dist
        up = np.array([0.0, 1.0, 0.0]) if abs(el) < 80 else np.array([0.0, 0.0, -1.0 if el > 0 else 1.0])
        Vm = look_at(eye, self.center, up)
        P = perspective(self.fov, 1.0, max(dist - self.radius * 1.5, self.radius * 1e-3), dist + self.radius * 1.5)
        mvp = (P @ Vm).astype(np.float32)
        self.prog["mvp"].write(mvp.T.tobytes())
        self.prog["eye"].value = tuple(eye.astype(float))
        key = d * 0.7 + np.array([0.3, 0.6, 0.2])
        self.prog["key"].value = tuple(key.astype(float))
        self.fbo.use()
        ctx.enable(ctx.DEPTH_TEST)
        ctx.disable(ctx.CULL_FACE)
        self.fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.ids.write(np.zeros((self.size, self.size), np.int32).tobytes())
        self.vao.render()
        img = np.frombuffer(self.col.read(), np.uint8).reshape(self.size, self.size, 4)[::-1, :, :3].copy()
        fid = np.frombuffer(self.ids.read(), np.int32).reshape(self.size, self.size)[::-1].copy() - 1
        return img, fid

    def release(self):
        for o in (self.vao, self.fbo, self.col, self.ids, self.dep, self.prog):
            o.release()
        self.ctx.release()
