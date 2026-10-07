"""Triangle mesh -> lattice solid mask and interpolated wall-link fractions.

All coordinates here are lattice units: cell (i, j, k) has its centre at (i, j, k).

Inside/outside is decided by ray parity along each of the three axes and a majority vote,
which tolerates a small hole or a duplicated face. Open (non-watertight) meshes and
sub-cell-thin sheets additionally get their surface cells marked solid so that flow cannot
leak through them.
"""
import math

import numpy as np
from numba import njit

from .kernels import CX, CY, CZ, Q

# Irrational offsets keep rays away from mesh edges and vertices.
_EPS_A = 1.2345678e-4
_EPS_B = 2.7182818e-4


@njit(cache=True)
def _parity_scan(tri, n1, n2, mode, counts, hits, fillp):
    T = tri.shape[0]
    for t in range(T):
        a0 = tri[t, 0, 0]; a1 = tri[t, 0, 1]; a2 = tri[t, 0, 2]
        b0 = tri[t, 1, 0]; b1 = tri[t, 1, 1]; b2 = tri[t, 1, 2]
        c0 = tri[t, 2, 0]; c1 = tri[t, 2, 1]; c2 = tri[t, 2, 2]
        lo1 = max(0, int(math.ceil(min(a1, b1, c1) - _EPS_A)))
        hi1 = min(n1 - 1, int(math.floor(max(a1, b1, c1) - _EPS_A)))
        lo2 = max(0, int(math.ceil(min(a2, b2, c2) - _EPS_B)))
        hi2 = min(n2 - 1, int(math.floor(max(a2, b2, c2) - _EPS_B)))
        if lo1 > hi1 or lo2 > hi2:
            continue
        det = (b1 - a1) * (c2 - a2) - (c1 - a1) * (b2 - a2)
        if det == 0.0:
            continue
        for j in range(lo1, hi1 + 1):
            pj = j + _EPS_A
            for k in range(lo2, hi2 + 1):
                pk = k + _EPS_B
                # barycentric coordinates of (pj, pk) in the projected triangle
                l1 = ((pj - a1) * (c2 - a2) - (c1 - a1) * (pk - a2)) / det
                l2 = ((b1 - a1) * (pk - a2) - (pj - a1) * (b2 - a2)) / det
                l0 = 1.0 - l1 - l2
                if l0 < 0.0 or l1 < 0.0 or l2 < 0.0:
                    continue
                idx = j * n2 + k
                if mode == 0:
                    counts[idx] += 1
                else:
                    hits[fillp[idx]] = l0 * a0 + l1 * b0 + l2 * c0
                    fillp[idx] += 1


@njit(cache=True)
def _parity_axis(tri, n0, n1, n2):
    """Parity fill along axis 0. tri is (T, 3, 3) already permuted so axis 0 is first."""
    counts = np.zeros(n1 * n2, dtype=np.int64)
    dummy_f = np.empty(0, dtype=np.float64)
    dummy_i = np.empty(0, dtype=np.int64)
    _parity_scan(tri, n1, n2, 0, counts, dummy_f, dummy_i)
    starts = np.zeros(n1 * n2 + 1, dtype=np.int64)
    for k in range(n1 * n2):
        starts[k + 1] = starts[k] + counts[k]
    hits = np.empty(starts[-1], dtype=np.float64)
    fillp = starts[:-1].copy()
    _parity_scan(tri, n1, n2, 1, counts, hits, fillp)
    inside = np.zeros((n0, n1, n2), dtype=np.uint8)
    for j in range(n1):
        for k in range(n2):
            idx = j * n2 + k
            s = starts[idx]
            e = starts[idx + 1]
            if e - s < 2:
                continue
            xs = np.sort(hits[s:e])
            p = 0
            m = e - s
            for i in range(n0):
                while p < m and xs[p] < i:
                    p += 1
                if p % 2 == 1:
                    inside[i, j, k] = 1
    return inside


def solid_mask(tri, shape, mark_surface=False):
    """Boolean solid mask of shape (nx, ny, nz) for triangles tri (T, 3, 3) in lattice units."""
    nx, ny, nz = shape
    tri = np.ascontiguousarray(tri, dtype=np.float64)
    px = _parity_axis(tri, nx, ny, nz)
    py = _parity_axis(np.ascontiguousarray(tri[:, :, [1, 0, 2]]), ny, nx, nz).transpose(1, 0, 2)
    pz = _parity_axis(np.ascontiguousarray(tri[:, :, [2, 0, 1]]), nz, nx, ny).transpose(1, 2, 0)
    votes = px.astype(np.int16) + py + pz
    solid = votes >= 2
    if mark_surface:
        solid |= surface_cells(tri, shape)
    return solid


@njit(cache=True)
def _surface_cells(tri, nx, ny, nz, out):
    T = tri.shape[0]
    for t in range(T):
        a = tri[t, 0]; b = tri[t, 1]; c = tri[t, 2]
        e1 = math.sqrt(((b - a) ** 2).sum())
        e2 = math.sqrt(((c - a) ** 2).sum())
        n = int(max(e1, e2) / 0.4) + 1
        for i in range(n + 1):
            for j in range(n + 1 - i):
                u = i / n
                v = j / n
                p0 = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])
                p1 = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1])
                p2 = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
                ix = int(math.floor(p0 + 0.5)); iy = int(math.floor(p1 + 0.5)); iz = int(math.floor(p2 + 0.5))
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    out[ix, iy, iz] = 1


def surface_cells(tri, shape):
    out = np.zeros(shape, dtype=np.uint8)
    _surface_cells(np.ascontiguousarray(tri, dtype=np.float64), shape[0], shape[1], shape[2], out)
    return out.astype(bool)


@njit(cache=True)
def _bin_scan(tri, nbx, nby, nbz, bs, mode, counts, items, fillp):
    T = tri.shape[0]
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    nb = np.array([nbx, nby, nbz], dtype=np.int64)
    for t in range(T):
        for a in range(3):
            mn = min(tri[t, 0, a], tri[t, 1, a], tri[t, 2, a])
            mx = max(tri[t, 0, a], tri[t, 1, a], tri[t, 2, a])
            lo[a] = max(0, int(math.floor(mn / bs)))
            hi[a] = min(nb[a] - 1, int(math.floor(mx / bs)))
        if lo[0] > hi[0] or lo[1] > hi[1] or lo[2] > hi[2]:
            continue
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    idx = (i * nby + j) * nbz + k
                    if mode == 0:
                        counts[idx] += 1
                    else:
                        items[fillp[idx]] = t
                        fillp[idx] += 1


@njit(cache=True)
def _bin_triangles(tri, nbx, nby, nbz, bs):
    n = nbx * nby * nbz
    counts = np.zeros(n, dtype=np.int64)
    dummy = np.empty(0, dtype=np.int64)
    _bin_scan(tri, nbx, nby, nbz, bs, 0, counts, dummy, dummy)
    starts = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        starts[k + 1] = starts[k] + counts[k]
    items = np.empty(starts[-1], dtype=np.int64)
    fillp = starts[:-1].copy()
    _bin_scan(tri, nbx, nby, nbz, bs, 1, counts, items, fillp)
    return starts, items


@njit(cache=True)
def _seg_tri(o, d, tri, t):
    """Moller-Trumbore: parameter s in [0, 1] where segment o + s*d hits triangle t, else -1."""
    v0 = tri[t, 0]; v1 = tri[t, 1]; v2 = tri[t, 2]
    e1x = v1[0] - v0[0]; e1y = v1[1] - v0[1]; e1z = v1[2] - v0[2]
    e2x = v2[0] - v0[0]; e2y = v2[1] - v0[1]; e2z = v2[2] - v0[2]
    px = d[1] * e2z - d[2] * e2y
    py = d[2] * e2x - d[0] * e2z
    pz = d[0] * e2y - d[1] * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-14:
        return -1.0
    inv = 1.0 / det
    tx = o[0] - v0[0]; ty = o[1] - v0[1]; tz = o[2] - v0[2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0:
        return -1.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return -1.0
    s = (e2x * qx + e2y * qy + e2z * qz) * inv
    if s < 0.0 or s > 1.0:
        return -1.0
    return s


@njit(cache=True)
def _build_links(solid, tri, starts, items, nby, nbz, bs, cx, cy, cz):
    nx, ny, nz = solid.shape
    bidx = np.full((nx, ny, nz), -1, dtype=np.int32)
    nb = 0
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if solid[x, y, z]:
                    continue
                for i in range(1, 19):
                    if solid[x - cx[i], y - cy[i], z - cz[i]]:
                        bidx[x, y, z] = nb
                        nb += 1
                        break
    qtab = np.full((nb, 19), -1.0, dtype=np.float32)
    nfound = 0
    nlinks = 0
    o = np.empty(3)
    d = np.empty(3)
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                b = bidx[x, y, z]
                if b < 0:
                    continue
                for i in range(1, 19):
                    sx = x - cx[i]; sy = y - cy[i]; sz = z - cz[i]
                    if not solid[sx, sy, sz]:
                        continue
                    nlinks += 1
                    o[0] = x; o[1] = y; o[2] = z
                    d[0] = sx - x; d[1] = sy - y; d[2] = sz - z
                    best = 2.0
                    bx0 = int(math.floor(min(x, sx) / bs)); bx1 = int(math.floor(max(x, sx) / bs))
                    by0 = int(math.floor(min(y, sy) / bs)); by1 = int(math.floor(max(y, sy) / bs))
                    bz0 = int(math.floor(min(z, sz) / bs)); bz1 = int(math.floor(max(z, sz) / bs))
                    for bi in range(bx0, bx1 + 1):
                        for bj in range(by0, by1 + 1):
                            for bk in range(bz0, bz1 + 1):
                                idx = (bi * nby + bj) * nbz + bk
                                for p in range(starts[idx], starts[idx + 1]):
                                    s = _seg_tri(o, d, tri, items[p])
                                    if s >= 0.0 and s < best:
                                        best = s
                    if best <= 1.0:
                        nfound += 1
                        qtab[b, i] = min(max(best, 0.02), 1.0)
                    else:
                        qtab[b, i] = 0.5
    return bidx, qtab, nlinks, nfound


def build_links(solid, tri, bin_size=2.0):
    """Wall-link table for interpolated bounce-back.

    Returns (bidx, qtab, stats): bidx[x, y, z] indexes a row of qtab for every fluid cell next
    to the body; qtab[b, i] is the fraction of the link (x -> x - c_i) at which the surface is
    crossed, or -1 if that neighbour is fluid.
    """
    nx, ny, nz = solid.shape
    tri = np.ascontiguousarray(tri, dtype=np.float64)
    nbx = int(nx / bin_size) + 1
    nby = int(ny / bin_size) + 1
    nbz = int(nz / bin_size) + 1
    starts, items = _bin_triangles(tri, nbx, nby, nbz, bin_size)
    bidx, qtab, nlinks, nfound = _build_links(solid, tri, starts, items, nby, nbz, bin_size, CX, CY, CZ)
    return bidx, qtab, {"wall_links": int(nlinks), "links_with_surface_hit": int(nfound),
                        "boundary_cells": int(qtab.shape[0])}


@njit(cache=True)
def _rescue_thin(tri, nrm, solid):
    """Mark the cell nearest to the surface solid wherever the shell is too thin for any
    cell centre to fall inside it, so sub-cell fins and trailing edges still block flow.

    Every test reads the mask as it was before any rescue. Testing the mask being written
    made the result depend on triangle order: the first side of a thin trailing edge marked
    its cell, the other side then saw it and marked nothing, so the edge was shifted to one
    side (a symmetric NACA 0012 wing gave CL = -0.1 at zero incidence)."""
    nx, ny, nz = solid.shape
    ref = solid.copy()
    added = 0
    for t in range(tri.shape[0]):
        a = tri[t, 0]; b = tri[t, 1]; c = tri[t, 2]
        e1 = math.sqrt(((b - a) ** 2).sum())
        e2 = math.sqrt(((c - a) ** 2).sum())
        n = int(max(e1, e2) / 0.5) + 1
        for i in range(n + 1):
            for j in range(n + 1 - i):
                u = i / n
                v = j / n
                p0 = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])
                p1 = a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1])
                p2 = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
                hit = False
                for depth in (0.0, 0.5, 1.0):
                    ix = int(math.floor(p0 - depth * nrm[t, 0] + 0.5))
                    iy = int(math.floor(p1 - depth * nrm[t, 1] + 0.5))
                    iz = int(math.floor(p2 - depth * nrm[t, 2] + 0.5))
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and ref[ix, iy, iz]:
                        hit = True
                        break
                if not hit:
                    ix = int(math.floor(p0 + 0.5)); iy = int(math.floor(p1 + 0.5)); iz = int(math.floor(p2 + 0.5))
                    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and not solid[ix, iy, iz]:
                        solid[ix, iy, iz] = True
                        added += 1
    return added


def voxelize_shells(tri, comp, closed, shape):
    """Solid mask from a mesh split into shells.

    tri: (T, 3, 3) lattice coordinates; comp: shell id per triangle; closed: bool per shell.
    Each closed shell is parity-filled on its own sub-grid and OR-ed in, so overlapping
    bodies (typical of multi-body CAD exports) do not cancel each other. Open shells are
    surface-voxelized. Returns (solid, stats).
    """
    shape = tuple(int(s) for s in shape)
    solid = np.zeros(shape, dtype=bool)
    gmax = np.array(shape) - 1
    for c in range(len(closed)):
        sel = comp == c
        if not sel.any():
            continue
        t = tri[sel]
        lo = np.maximum(np.floor(t.reshape(-1, 3).min(0)).astype(int) - 1, 0)
        hi = np.minimum(np.ceil(t.reshape(-1, 3).max(0)).astype(int) + 1, gmax)
        if (hi < lo).any():
            continue
        sub = tuple(int(v) for v in (hi - lo + 1))
        tl = t - lo
        s = solid_mask(tl, sub) if closed[c] else surface_cells(tl, sub)
        solid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] |= s
    # outward normals for the closed shells (open shells are already surface-marked)
    keep = np.array([closed[c] for c in comp], dtype=bool)
    added = 0
    if keep.any():
        tc = np.ascontiguousarray(tri[keep])
        cr = np.cross(tc[:, 1] - tc[:, 0], tc[:, 2] - tc[:, 0])
        nrm = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-300)[:, None]
        added = int(_rescue_thin(tc, nrm, solid))
    return solid, {"thin_rescue_cells": added, "shells": int(len(closed)),
                   "open_shells": int(sum(1 for c in closed if not c))}
