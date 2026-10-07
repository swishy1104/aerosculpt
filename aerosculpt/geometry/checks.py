"""Geometry validity checks run on every candidate before any CFD is spent on it.

  closed      every shell is watertight (each edge shared by exactly two faces)
  intersect   no shell passes through itself (separate shells may overlap: a wing through a
              fuselage is how multi-body CAD exports and the templates are built)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .mesh import Mesh


def closed_shells(mesh: Mesh):
    """(all shells closed, number of open or non-manifold edges)."""
    b, nm = mesh.edge_stats()
    return (b == 0 and nm == 0), int(b + nm)


@njit(cache=True)
def _seg_tri(p, q, a, b, c):
    """Segment pq strictly crosses triangle abc (Moller-Trumbore on the segment)."""
    e1x, e1y, e1z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    e2x, e2y, e2z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x
    det = e1x * hx + e1y * hy + e1z * hz
    scale = math.sqrt((e1x * e1x + e1y * e1y + e1z * e1z) * (e2x * e2x + e2y * e2y + e2z * e2z)) * \
        math.sqrt(dx * dx + dy * dy + dz * dz)
    if abs(det) <= 1e-12 * scale:
        return False
    inv = 1.0 / det
    sx, sy, sz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    u = (sx * hx + sy * hy + sz * hz) * inv
    if u <= 1e-9 or u >= 1 - 1e-9:
        return False
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v <= 1e-9 or u + v >= 1 - 1e-9:
        return False
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    return 1e-9 < t < 1 - 1e-9


@njit(cache=True)
def _count(tri, f, comp, order, lo, hi, limit):
    n = tri.shape[0]
    hits = 0
    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            if lo[j, 0] > hi[i, 0]:
                break
            if comp[i] != comp[j]:
                continue
            if lo[j, 1] > hi[i, 1] or hi[j, 1] < lo[i, 1] or lo[j, 2] > hi[i, 2] or hi[j, 2] < lo[i, 2]:
                continue
            shared = False
            for a in range(3):
                for b in range(3):
                    if f[i, a] == f[j, b]:
                        shared = True
            if shared:
                continue
            hit = False
            for k in range(3):
                if _seg_tri(tri[i, k], tri[i, (k + 1) % 3], tri[j, 0], tri[j, 1], tri[j, 2]) or \
                        _seg_tri(tri[j, k], tri[j, (k + 1) % 3], tri[i, 0], tri[i, 1], tri[i, 2]):
                    hit = True
                    break
            if hit:
                hits += 1
                if hits >= limit:
                    return hits
    return hits


def self_intersections(mesh: Mesh, limit=1):
    """Number of intersecting face pairs within the same shell (stops counting at `limit`)."""
    tri = np.ascontiguousarray(mesh.triangles())
    comp = mesh.components().astype(np.int64)
    lo = tri.min(axis=1)
    hi = tri.max(axis=1)
    order = np.argsort(lo[:, 0], kind="stable").astype(np.int64)
    return int(_count(tri, np.ascontiguousarray(mesh.f), comp, order, lo, hi, int(limit)))


def check(mesh: Mesh):
    """Reason the mesh must not be simulated, or None."""
    ok, n_open = closed_shells(mesh)
    if not ok:
        return f"not watertight ({n_open} open edges)"
    if self_intersections(mesh, 1):
        return "self-intersecting surface"
    return None
