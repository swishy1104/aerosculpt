"""Design constraints tied to places on the model.

Three zone types, each a box or a cylinder along the flow axis:

  protect  geometry inside never moves (screw holes, mounting faces, fins, interfaces).
           The deformation fades out smoothly around the zone so the surface never tears.
  keep     the volume must stay enclosed by the body (cargo bay, battery, motor tube).
           Checked on every candidate design before any CFD is spent on it.
  design   if any design zones exist, only geometry inside them may change.

Users define zones in their own model coordinates and units; `from_spec` maps them into the
body frame the optimizer works in (metres, nose -x, z up, centred).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

TYPES = ("protect", "keep", "design")
SHAPES = ("box", "cylinder")


@dataclass
class Zone:
    type: str
    shape: str
    center: np.ndarray     # body frame, metres
    half: np.ndarray       # half extents along body x, y, z (cylinder: half length, radius, radius)
    name: str = ""

    def distance(self, p):
        """Distance from points (N, 3) to the zone (0 inside)."""
        d = np.abs(p - self.center)
        if self.shape == "cylinder":
            r = np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2)
            dr = np.maximum(r - self.half[1], 0.0)
            dx = np.maximum(d[:, 0] - self.half[0], 0.0)
            return np.sqrt(dr * dr + dx * dx)
        return np.linalg.norm(np.maximum(d - self.half, 0.0), axis=1)

    def surface_points(self, n=7):
        """Points covering the zone boundary, used for the keep-inside test."""
        c, h = self.center, self.half
        g = np.linspace(-1.0, 1.0, n)
        pts = []
        if self.shape == "cylinder":
            th = np.linspace(0, 2 * np.pi, 4 * n, endpoint=False)
            for x in g:
                for t in th:
                    pts.append([c[0] + x * h[0], c[1] + h[1] * math.cos(t), c[2] + h[1] * math.sin(t)])
            for x in (-1.0, 1.0):  # end caps
                for rr in (0.5, 0.0):
                    for t in th[::2]:
                        pts.append([c[0] + x * h[0], c[1] + rr * h[1] * math.cos(t), c[2] + rr * h[1] * math.sin(t)])
        else:
            for axis in range(3):
                o = [k for k in range(3) if k != axis]
                for sgn in (-1.0, 1.0):
                    for a in g:
                        for b in g:
                            p = np.empty(3)
                            p[axis] = c[axis] + sgn * h[axis]
                            p[o[0]] = c[o[0]] + a * h[o[0]]
                            p[o[1]] = c[o[1]] + b * h[o[1]]
                            pts.append(p)
        return np.unique(np.round(np.array(pts, float), 12), axis=0)

    def corners(self):
        c, h = self.center, self.half
        return np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def from_spec(items, scale, R0, center):
    """Zones as sent by the client (model coordinates in file units) -> body-frame Zones.

    R0 only permutes and flips axes, so axis-aligned boxes stay axis-aligned. Cylinders
    always run along the flow axis (body x)."""
    out = []
    A = np.abs(np.asarray(R0))
    for it in items or []:
        c = np.asarray(it["center"], float) * scale
        s = np.asarray(it["size"], float) * scale
        cb = np.asarray(R0) @ c - center
        hb = (A @ s) / 2.0
        if it.get("shape") == "cylinder":
            r = max(hb[1], hb[2])
            hb = np.array([hb[0], r, r])
        out.append(Zone(it["type"], it.get("shape", "box"), cb, np.maximum(hb, 1e-6), it.get("name", "")))
    return out


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def deformation_weights(v, zones, body_length):
    """Per-vertex multiplier in [0, 1] for the design displacement."""
    w = np.ones(len(v))
    design = [z for z in zones if z.type == "design"]
    for z in zones:
        if z.type == "protect":
            blend = float(np.clip(z.half.max(), 0.02 * body_length, 0.25 * body_length))
            w *= _smoothstep(z.distance(v) / blend)
    if design:
        wd = np.zeros(len(v))
        for z in design:
            blend = float(np.clip(z.half.max(), 0.02 * body_length, 0.25 * body_length))
            wd = np.maximum(wd, 1.0 - _smoothstep(z.distance(v) / blend))
        w *= wd
    return w


@njit(cache=True)
def _inside_counts(tri, comp, ncomp, pts):
    """Ray parity (+x rays, slightly perturbed) per closed shell; inside any shell = inside."""
    P = pts.shape[0]
    out = np.zeros(P, dtype=np.bool_)
    counts = np.zeros(ncomp, dtype=np.int64)
    for p in range(P):
        px = pts[p, 0]
        py = pts[p, 1] + 1.234567e-9
        pz = pts[p, 2] + 2.718281e-9
        counts[:] = 0
        for t in range(tri.shape[0]):
            a1 = tri[t, 0, 1]; a2 = tri[t, 0, 2]
            b1 = tri[t, 1, 1]; b2 = tri[t, 1, 2]
            c1 = tri[t, 2, 1]; c2 = tri[t, 2, 2]
            if py < min(a1, b1, c1) or py > max(a1, b1, c1) or pz < min(a2, b2, c2) or pz > max(a2, b2, c2):
                continue
            det = (b1 - a1) * (c2 - a2) - (c1 - a1) * (b2 - a2)
            if det == 0.0:
                continue
            l1 = ((py - a1) * (c2 - a2) - (c1 - a1) * (pz - a2)) / det
            l2 = ((b1 - a1) * (pz - a2) - (py - a1) * (b2 - a2)) / det
            l0 = 1.0 - l1 - l2
            if l0 < 0.0 or l1 < 0.0 or l2 < 0.0:
                continue
            x = l0 * tri[t, 0, 0] + l1 * tri[t, 1, 0] + l2 * tri[t, 2, 0]
            if x > px:
                counts[comp[t]] += 1
        for c in range(ncomp):
            if counts[c] % 2 == 1:
                out[p] = True
                break
    return out


class KeepInsideCheck:
    """Every keep zone must stay at least as enclosed as it was in the original design."""

    def __init__(self, base_mesh, zones, comp):
        self.zones = [z for z in zones if z.type == "keep"]
        self.comp = np.ascontiguousarray(comp, dtype=np.int64)
        self.ncomp = int(comp.max()) + 1 if len(comp) else 1
        self.points = [z.surface_points() for z in self.zones]
        self.base_frac = [self.fraction(base_mesh, i) for i in range(len(self.zones))]

    def fraction(self, mesh, i):
        inside = _inside_counts(np.ascontiguousarray(mesh.triangles()), self.comp, self.ncomp,
                                np.ascontiguousarray(self.points[i]))
        return float(inside.mean())

    def ok(self, mesh):
        for i in range(len(self.zones)):
            if self.fraction(mesh, i) < self.base_frac[i] - 1e-9:
                return False
        return True

    def report(self):
        return [{"name": z.name, "inside_fraction": f} for z, f in zip(self.zones, self.base_frac)]
