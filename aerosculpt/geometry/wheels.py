"""Wheels for the rolling road: find them on a vehicle mesh, and turn wheel zones into the spinning
rims the tunnel needs.

Detection is a proposal for the user to check, not a measurement: the lowest vertices (the
contact patches) are grouped left/right and front/back, and each group gets a box of a typical
tyre radius (0.21 x the vehicle height, as on cars from a hatchback to an SUV) resting on the road.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _clusters(x, gap):
    """1D groups of sorted values separated by more than `gap`."""
    order = np.argsort(x)
    xs = x[order]
    cuts = np.flatnonzero(np.diff(xs) > gap) + 1
    return [order[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(xs)])]


def detect_wheels(body: Mesh):
    """Body frame (nose -x, z up, centred) -> wheel boxes [{"center": [x, y, z], "size": [dx, dy, dz]}]."""
    v = body.v
    lo, hi = body.bbox()
    L, W, H = hi - lo
    if min(L, W, H) <= 0:
        return []
    zmin = lo[2]
    low = v[v[:, 2] < zmin + 0.03 * H]
    R = float(np.clip(0.21 * H, 0.02 * L, 0.12 * L))
    out = []
    for side in (-1.0, 1.0):
        pts = low[low[:, 1] * side > 0.1 * W / 2]
        if len(pts) < 2:
            continue
        for idx in _clusters(pts[:, 0], 0.15 * L):
            c = pts[idx]
            if len(c) < 2:
                continue
            xc = float(c[:, 0].mean())
            yc = float(c[:, 1].mean())
            width = float(np.clip(np.ptp(c[:, 1]) * 1.3, 0.04 * L, 0.3 * W))
            out.append({"center": [xc, yc, float(zmin + R)], "size": [2.0 * R, width, 2.0 * R]})
    # at most two axles a side (the closest groups to the ends): stray low points are not wheels
    out.sort(key=lambda w: w["center"][0])
    left = [w for w in out if w["center"][1] < 0]
    right = [w for w in out if w["center"][1] > 0]
    keep = []
    for group in (left, right):
        keep += group if len(group) <= 2 else [group[0], group[-1]]
    return keep


def wheel_rims(zones):
    """Wheel zones (body frame Zone objects) -> (x, y, z, radius, width) per wheel. The axle is the
    zone's y axis; the radius is half the larger of its x and z sizes."""
    out = []
    for z in zones:
        if z.type != "wheel":
            continue
        c = np.asarray(z.center, float)
        h = np.asarray(z.half, float)
        out.append((float(c[0]), float(c[1]), float(c[2]), float(max(h[0], h[2])), float(2.0 * h[1])))
    return out
