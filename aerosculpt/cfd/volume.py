"""The averaged 3D flow field of a run: a coarse copy for export (ParaView / VTK) and vortex
surfaces (Q-criterion) for the 3D view.

Q = 1/2 (|Omega|^2 - |S|^2), the second invariant of the velocity gradient: positive where rotation
dominates strain, i.e. inside vortices (Hunt, Wray & Moin 1988). It is shown as an iso-surface at
a level normalised by (U / L)^2 with L the body's near-field size, so the same level means the same
thing on every body and grid.
"""
from __future__ import annotations

import base64
import io
import math
import struct
import zipfile

import numpy as np

MAX_CELLS_ALONG = 112          # the exported volume keeps at most this many cells along its longest side
MAX_VORTEX_FACES = 24_000
Q_LEVEL = 4.0                  # Q (L/U)^2 of the shown vortex surfaces


def coarse_volume(rho, ux, uy, uz, solid, domain, u_lat, k_blockage):
    """Down-sampled mean field: velocity / freestream, pressure coefficient, solid mask."""
    st = max(1, int(math.ceil(max(domain.shape) / MAX_CELLS_ALONG)))
    sl = np.s_[::st, ::st, ::st]
    q_lat = 0.5 * u_lat * u_lat
    u = np.stack([np.nan_to_num(a[sl]) for a in (ux, uy, uz)]) / u_lat
    cp = np.nan_to_num(((rho[sl] - 1.0) / 3.0) / q_lat / k_blockage)
    s = solid[sl]
    u[:, s] = 0.0
    cp[s] = 0.0
    return {"origin": domain.to_physical([0, 0, 0]).astype(np.float64), "spacing": float(domain.dx * st),
            "u": u.astype(np.float16), "cp": cp.astype(np.float16), "solid": s.astype(np.uint8)}


def q_criterion(u, spacing):
    """Q of a (3, nx, ny, nz) velocity field on a uniform grid (central differences)."""
    g = [[np.gradient(u[i].astype(np.float32), spacing, axis=j) for j in range(3)] for i in range(3)]
    q = np.zeros(u.shape[1:], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            s = 0.5 * (g[i][j] + g[j][i])
            w = 0.5 * (g[i][j] - g[j][i])
            q += 0.5 * (w * w - s * s)
    return q


def vortex_surface(vol, length_scale):
    """Iso-surface of the normalised Q-criterion as (points float32 (N, 3) in metres, faces int32)."""
    from skimage.measure import marching_cubes
    u = vol["u"].astype(np.float32)                       # velocity / U
    h = vol["spacing"]
    qn = q_criterion(u, h) * (length_scale ** 2)          # Q (L/U)^2, since u is already / U
    qn[vol["solid"].astype(bool)] = 0.0
    level = Q_LEVEL
    if not (qn > level).any():
        return None
    for _ in range(6):
        verts, faces, _, _ = marching_cubes(qn, level=level)
        if len(faces) <= MAX_VORTEX_FACES:
            break
        level *= 1.6                                       # too much: show only the stronger vortices
    if len(faces) > MAX_VORTEX_FACES:
        faces = faces[:: int(math.ceil(len(faces) / MAX_VORTEX_FACES))]
    pts = (vol["origin"] + verts * h).astype(np.float32)
    return pts, faces.astype(np.int32), float(level)


def vortex_payload(vol, length_scale):
    out = vortex_surface(vol, length_scale)
    if out is None:
        return None
    pts, faces, level = out
    return {"points": base64.b64encode(pts.tobytes()).decode("ascii"),
            "faces": base64.b64encode(faces.tobytes()).decode("ascii"), "level": level}


# ---- legacy VTK (ParaView, VisIt, Tecplot all read it) ----------------------------------------
def _be(a):
    return np.ascontiguousarray(a, dtype=">f4").tobytes()


def vtk_volume(vol, title="AeroSculpt mean flow"):
    """STRUCTURED_POINTS file: velocity (/ freestream), Cp and Q (L/U)^2 at the points."""
    u = vol["u"].astype(np.float32)
    _, nx, ny, nz = u.shape
    h = vol["spacing"]
    o = vol["origin"]
    q = q_criterion(u, h)
    head = (f"# vtk DataFile Version 3.0\n{title}\nBINARY\nDATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {nx} {ny} {nz}\nORIGIN {o[0]:.9g} {o[1]:.9g} {o[2]:.9g}\nSPACING {h:.9g} {h:.9g} {h:.9g}\n"
            f"POINT_DATA {nx * ny * nz}\n").encode()
    body = b"VECTORS velocity_over_U float\n" + _be(np.moveaxis(u, 0, -1).transpose(2, 1, 0, 3).ravel())
    body += b"\nSCALARS cp float 1\nLOOKUP_TABLE default\n" + _be(vol["cp"].astype(np.float32).transpose(2, 1, 0).ravel())
    body += b"\nSCALARS q_criterion float 1\nLOOKUP_TABLE default\n" + _be(q.transpose(2, 1, 0).ravel())
    body += b"\nSCALARS solid float 1\nLOOKUP_TABLE default\n" + _be(vol["solid"].astype(np.float32).transpose(2, 1, 0).ravel())
    return head + body + b"\n"


def vtk_surface(mesh, cp=None, title="AeroSculpt surface"):
    """POLYDATA of the body with the surface pressure coefficient per face."""
    v = mesh.v.astype(np.float32)
    f = mesh.f.astype(np.int64)
    out = io.BytesIO()
    out.write(f"# vtk DataFile Version 3.0\n{title}\nBINARY\nDATASET POLYDATA\nPOINTS {len(v)} float\n".encode())
    out.write(_be(v.ravel()))
    out.write(f"\nPOLYGONS {len(f)} {4 * len(f)}\n".encode())
    cells = np.hstack([np.full((len(f), 1), 3), f]).astype(">i4")
    out.write(cells.tobytes())
    if cp is not None and len(cp) == len(f):
        out.write(f"\nCELL_DATA {len(f)}\nSCALARS cp float 1\nLOOKUP_TABLE default\n".encode())
        out.write(_be(np.nan_to_num(np.asarray(cp, np.float32))))
    out.write(b"\n")
    return out.getvalue()


def vtk_zip(entries):
    """entries: [(name, bytes)] -> zip bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


_ = struct
