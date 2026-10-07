"""Triangle meshes: parsing, welding, metrics, transforms and export.

Everything is float64 metres once a mesh has been through `Mesh.scaled()`. Parsing keeps
the file's own units; the caller applies the unit scale explicitly, once.
"""
from __future__ import annotations

import io
import math
import re
import struct

import numpy as np
from numba import njit

UNIT_SCALE = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "ft": 0.3048}
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


class MeshError(ValueError):
    pass


class Mesh:
    def __init__(self, vertices, faces):
        self.v = np.ascontiguousarray(vertices, dtype=np.float64)
        self.f = np.ascontiguousarray(faces, dtype=np.int64)
        if self.v.ndim != 2 or self.v.shape[1] != 3 or self.f.ndim != 2 or self.f.shape[1] != 3:
            raise MeshError("mesh arrays have the wrong shape")

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_triangles(cls, tri, tol_rel=1e-7):
        """Weld a triangle soup (T, 3, 3) into an indexed mesh, dropping degenerate faces."""
        tri = np.asarray(tri, dtype=np.float64).reshape(-1, 3, 3)
        if len(tri) == 0:
            raise MeshError("the file contains no triangles")
        if not np.isfinite(tri).all():
            raise MeshError("the file contains NaN or infinite coordinates")
        pts = tri.reshape(-1, 3)
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        if diag <= 0:
            raise MeshError("the mesh has zero size")
        q = np.round((pts - pts.min(0)) / (diag * tol_rel)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        v = pts[first]
        f = inv.reshape(-1, 3)
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[ok]
        if len(f) == 0:
            raise MeshError("all triangles are degenerate")
        return cls(v, f)

    def copy(self):
        return Mesh(self.v.copy(), self.f.copy())

    # ---- basic quantities ----------------------------------------------------
    def triangles(self):
        return self.v[self.f]

    @property
    def n_faces(self):
        return int(len(self.f))

    def bbox(self):
        return self.v.min(axis=0), self.v.max(axis=0)

    def face_areas_normals(self):
        t = self.triangles()
        cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        a2 = np.linalg.norm(cr, axis=1)
        n = cr / np.maximum(a2, 1e-300)[:, None]
        return 0.5 * a2, n

    def area(self):
        return float(self.face_areas_normals()[0].sum())

    def signed_volume(self, faces=None):
        f = self.f if faces is None else faces
        t = self.v[f]
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def edge_stats(self):
        """Return (boundary_edges, nonmanifold_edges) counts."""
        e = np.sort(np.concatenate([self.f[:, [0, 1]], self.f[:, [1, 2]], self.f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return int((counts == 1).sum()), int((counts > 2).sum())

    def components(self):
        """Label connected face components (shells) -> array of component ids per face."""
        nv = len(self.v)
        parent = np.arange(nv)
        _union_faces(parent, self.f)
        roots = _find_all(parent)
        _, comp = np.unique(roots[self.f[:, 0]], return_inverse=True)
        return comp

    def orient_outward(self):
        """Make winding consistent across shared edges, then flip each shell so its signed
        volume is positive (outward normals). Repairs the mixed winding common in STLs."""
        self.f = _consistent_winding(self.f)
        comp = self.components()
        f = self.f.copy()
        for c in np.unique(comp):
            sel = comp == c
            if self.signed_volume(f[sel]) < 0:
                f[sel] = f[sel][:, ::-1]
        self.f = f
        return self

    # ---- transforms ----------------------------------------------------------
    def transformed(self, R=None, t=None, s=1.0):
        v = self.v * s
        if R is not None:
            v = v @ np.asarray(R).T
        if t is not None:
            v = v + np.asarray(t)
        return Mesh(v, self.f.copy())

    def scaled(self, units):
        if units not in UNIT_SCALE:
            raise MeshError(f"unknown unit {units!r}")
        return self.transformed(s=UNIT_SCALE[units])

    # ---- export --------------------------------------------------------------
    def to_stl_bytes(self, header=b"AeroSculpt export"):
        a, n = self.face_areas_normals()
        t = self.triangles().astype(np.float32)
        rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        rec["n"] = n.astype(np.float32)
        rec["v"] = t
        buf = io.BytesIO()
        buf.write(header[:80].ljust(80, b" "))
        buf.write(struct.pack("<I", len(t)))
        buf.write(rec.tobytes())
        return buf.getvalue()


def _consistent_winding(f):
    """Flip faces so every manifold edge is traversed in opposite directions by its two faces."""
    nf = len(f)
    if nf == 0:
        return f
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    face = np.concatenate([np.arange(nf)] * 3)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    fwd = (a < b).astype(np.int8)
    order = np.lexsort((hi, lo))
    lo, hi, fwd, face = lo[order], hi[order], fwd[order], face[order]
    same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
    # keep only edges shared by exactly two faces
    start = np.flatnonzero(np.concatenate([[True], ~same]))
    count = np.diff(np.concatenate([start, [len(lo)]]))
    pair = start[count == 2]
    f1, f2 = face[pair], face[pair + 1]
    flip_rel = (fwd[pair] == fwd[pair + 1]).astype(np.int8)
    flip = _propagate(nf, f1, f2, flip_rel)
    out = f.copy()
    out[flip] = out[flip][:, ::-1]
    return out


@njit(cache=True)
def _propagate(nf, f1, f2, rel):
    deg = np.zeros(nf + 1, np.int64)
    for k in range(len(f1)):
        deg[f1[k] + 1] += 1
        deg[f2[k] + 1] += 1
    for i in range(nf):
        deg[i + 1] += deg[i]
    adj = np.empty(deg[nf], np.int64)
    adr = np.empty(deg[nf], np.int8)
    fill = deg[:-1].copy()
    for k in range(len(f1)):
        adj[fill[f1[k]]] = f2[k]; adr[fill[f1[k]]] = rel[k]; fill[f1[k]] += 1
        adj[fill[f2[k]]] = f1[k]; adr[fill[f2[k]]] = rel[k]; fill[f2[k]] += 1
    state = np.full(nf, -1, np.int8)      # -1 unvisited, 0 keep, 1 flip
    stack = np.empty(nf, np.int64)
    for s0 in range(nf):
        if state[s0] >= 0:
            continue
        state[s0] = 0
        top = 0
        stack[0] = s0
        while top >= 0:
            u = stack[top]
            top -= 1
            for p in range(deg[u], deg[u + 1]):
                v = adj[p]
                if state[v] < 0:
                    state[v] = state[u] ^ adr[p]
                    top += 1
                    stack[top] = v
    return state == 1


@njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _union_faces(parent, f):
    for k in range(f.shape[0]):
        a = _find(parent, f[k, 0])
        for j in (1, 2):
            b = _find(parent, f[k, j])
            if a != b:
                parent[b] = a


@njit(cache=True)
def _find_all(parent):
    out = np.empty_like(parent)
    for i in range(parent.shape[0]):
        out[i] = _find(parent, i)
    return out


# ---- parsing -------------------------------------------------------------------
def parse_stl(data: bytes):
    if len(data) < 15:
        raise MeshError("file is too small to be an STL")
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data) and n > 0:
            rec = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")],
                                count=n, offset=84)
            return rec["v"].astype(np.float64)
    head = data[:512].lstrip().lower()
    if head.startswith(b"solid"):
        txt = data.decode("latin-1")
        nums = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", txt)
        if not nums:
            raise MeshError("ASCII STL has no vertices")
        v = np.array(nums, dtype=np.float64)
        if len(v) % 3:
            raise MeshError("ASCII STL vertex count is not a multiple of 3")
        return v.reshape(-1, 3, 3)
    raise MeshError("not a valid binary or ASCII STL file")


def parse_obj(data: bytes):
    verts = []
    faces = []
    for line in data.decode("latin-1").splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                i = int(tok.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)
            for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        raise MeshError("OBJ file has no faces")
    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)
    if f.max() >= len(v) or f.min() < 0:
        raise MeshError("OBJ face references a missing vertex")
    return v[f]


def load_mesh(data: bytes, filename: str = "model.stl") -> Mesh:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "obj":
        tri = parse_obj(data)
    elif ext == "stl":
        tri = parse_stl(data)
    else:
        raise MeshError(f"unsupported file type .{ext} (convert STEP/IGES/3MF in the browser first)")
    return Mesh.from_triangles(tri)


# ---- orientation ----------------------------------------------------------------
def body_to_wind(forward="-x", up="+z", aoa_deg=0.0, yaw_deg=0.0):
    """Rotation taking model coordinates to wind axes.

    `forward` is the model axis the nose points along; it is turned to face upstream (-x)
    because the flow travels +x. `up` becomes +z. Positive angle of attack pitches the nose
    up; positive yaw turns the nose towards +y.
    """
    f = np.array(AXES[forward], float)
    u = np.array(AXES[up], float)
    if abs(f @ u) > 1e-9:
        raise MeshError("forward and up axes must be perpendicular")
    ex = -f
    ez = u
    ey = np.cross(ez, ex)
    R0 = np.vstack([ex, ey, ez])           # rows: new axes in model coordinates
    a = math.radians(aoa_deg)
    Ra = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    b = math.radians(yaw_deg)
    Rb = np.array([[math.cos(b), math.sin(b), 0], [-math.sin(b), math.cos(b), 0], [0, 0, 1]])
    return Rb @ Ra @ R0, R0


def guess_units(mesh: Mesh):
    lo, hi = mesh.bbox()
    ext = float((hi - lo).max())
    if ext > 30:
        return "mm"
    if ext > 3:
        return "cm"
    return "m"


def guess_forward(mesh: Mesh):
    """Longest bounding-box axis is the usual flow axis for aerospace bodies."""
    lo, hi = mesh.bbox()
    k = int(np.argmax(hi - lo))
    return ["-x", "-y", "-z"][k]


# ---- metrics ----------------------------------------------------------------------
@njit(cache=True)
def _raster_projected(tri2, y0, z0, h, ny, nz, img):
    for t in range(tri2.shape[0]):
        a1 = tri2[t, 0, 0]; a2 = tri2[t, 0, 1]
        b1 = tri2[t, 1, 0]; b2 = tri2[t, 1, 1]
        c1 = tri2[t, 2, 0]; c2 = tri2[t, 2, 1]
        det = (b1 - a1) * (c2 - a2) - (c1 - a1) * (b2 - a2)
        if det == 0.0:
            continue
        j0 = max(0, int((min(a1, b1, c1) - y0) / h)); j1 = min(ny - 1, int((max(a1, b1, c1) - y0) / h) + 1)
        k0 = max(0, int((min(a2, b2, c2) - z0) / h)); k1 = min(nz - 1, int((max(a2, b2, c2) - z0) / h) + 1)
        for j in range(j0, j1 + 1):
            py = y0 + (j + 0.5) * h
            for k in range(k0, k1 + 1):
                if img[j, k]:
                    continue
                pz = z0 + (k + 0.5) * h
                l1 = ((py - a1) * (c2 - a2) - (c1 - a1) * (pz - a2)) / det
                l2 = ((b1 - a1) * (pz - a2) - (py - a1) * (b2 - a2)) / det
                if l1 >= 0.0 and l2 >= 0.0 and l1 + l2 <= 1.0:
                    img[j, k] = 1


def projected_area(mesh: Mesh, axis=0, res=512):
    """Silhouette area seen along `axis` (0 = x gives frontal area in wind axes)."""
    other = [a for a in range(3) if a != axis]
    t = mesh.triangles()[:, :, other]
    lo = t.reshape(-1, 2).min(0)
    hi = t.reshape(-1, 2).max(0)
    h = float((hi - lo).max()) / res
    ny = int((hi[0] - lo[0]) / h) + 2
    nz = int((hi[1] - lo[1]) / h) + 2
    img = np.zeros((ny, nz), dtype=np.uint8)
    _raster_projected(np.ascontiguousarray(t), lo[0], lo[1], h, ny, nz, img)
    return float(img.sum()) * h * h


def metrics(mesh: Mesh):
    """Geometric summary of a mesh already in wind axes and metres."""
    lo, hi = mesh.bbox()
    ext = hi - lo
    b_edges, nm_edges = mesh.edge_stats()
    vol = abs(mesh.signed_volume()) if b_edges == 0 else None
    af = projected_area(mesh, 0)
    ap = projected_area(mesh, 2)
    L = float(ext[0])
    d_eq = math.sqrt(4 * af / math.pi) if af > 0 else 0.0
    return {
        "triangles": mesh.n_faces,
        "vertices": int(len(mesh.v)),
        "length": L, "width": float(ext[1]), "height": float(ext[2]),
        "wetted_area": mesh.area(),
        "volume": vol,
        "frontal_area": af,
        "planform_area": ap,
        "fineness": (L / d_eq) if d_eq > 0 else None,
        "watertight": b_edges == 0 and nm_edges == 0,
        "open_edges": b_edges,
        "nonmanifold_edges": nm_edges,
        "shells": int(mesh.components().max() + 1),
    }


def shell_info(mesh: Mesh):
    """Component id per face and whether each shell is closed (every edge shared twice)."""
    comp = mesh.components()
    ncomp = int(comp.max()) + 1
    e = np.concatenate([mesh.f[:, [0, 1]], mesh.f[:, [1, 2]], mesh.f[:, [2, 0]]])
    ec = np.concatenate([comp, comp, comp])
    e = np.sort(e, axis=1)
    _, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bad = counts[inv] != 2
    closed = np.ones(ncomp, dtype=bool)
    closed[np.unique(ec[bad])] = False
    return comp, closed
