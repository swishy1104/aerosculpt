"""Smooth, low-dimensional shape deformation for arbitrary bodies.

The body lives in its own frame (nose towards -x, z up, centred). A design vector of a
few named variables bends the whole mesh through smooth station-wise fields along x:

  sy(s), sz(s)  cross-section scale about the local section centroid
  dz(s)         vertical shift (camber)
  x stretch     overall length / fineness
  fairing       blend towards a Taubin-smoothed copy (rounds sharp edges)

where s in [0, 1] runs nose to tail. Every field is C1-smooth, so a watertight input stays
watertight and the optimizer sees a smooth objective. Nothing here knows what the body
is, which is what makes it usable on any upload.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, shell_info
from .zones import KeepInsideCheck, deformation_weights


@dataclass(frozen=True)
class Variable:
    key: str
    label: str
    lo: float          # bound at "max change" = 100 %
    hi: float
    neutral: float = 0.0
    unit: str = "%"


ASPECTS = {
    "nose": {
        "label": "Nose / forebody",
        "help": "Sharpen or fill out the front third of the body.",
        "vars": [Variable("nose_radial", "Nose fullness", -0.45, 0.30)],
    },
    "tail": {
        "label": "Tail / aft-body",
        "help": "Boat-tail taper of the rear third: the biggest lever on base drag.",
        "vars": [Variable("tail_taper", "Tail taper", -0.55, 0.20)],
    },
    "section": {
        "label": "Cross-section",
        "help": "Width and height of the mid-body independently (frontal area).",
        "vars": [Variable("mid_width", "Mid-body width", -0.30, 0.30),
                 Variable("mid_height", "Mid-body height", -0.30, 0.30)],
    },
    "length": {
        "label": "Length / fineness",
        "help": "Stretch the body along the flow; trades wetted area against form drag.",
        "vars": [Variable("length", "Length", -0.10, 0.35)],
    },
    "camber": {
        "label": "Camber",
        "help": "Nose droop and tail upsweep: aligns the body with the flow at incidence.",
        "vars": [Variable("nose_droop", "Nose droop", -0.20, 0.20),
                 Variable("tail_sweep", "Tail upsweep", -0.20, 0.20)],
    },
    "fairing": {
        "label": "Edge fairing",
        "help": "Round off sharp edges and corners (smoothing blend).",
        "vars": [Variable("fairing", "Fairing", 0.0, 1.0)],
    },
}


def design_space(aspects, max_change=0.25):
    """Variables for the chosen aspects; bounds scale with max_change (0.25 = default)."""
    out = []
    k = max_change / 0.25
    for a in aspects:
        if a not in ASPECTS:
            raise ValueError(f"unknown design aspect {a!r}")
        for v in ASPECTS[a]["vars"]:
            if v.key == "fairing":
                out.append(Variable(v.key, v.label, 0.0, min(1.0, v.hi * k), 0.0, ""))
            else:
                out.append(Variable(v.key, v.label, v.lo * k, v.hi * k, 0.0))
    return out


# ---- station fields --------------------------------------------------------------------
def _bump_front(s, w=0.42):
    t = np.clip(s / w, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def _bump_back(s, w=0.42):
    return _bump_front(1.0 - s, w)


def _bump_mid(s):
    return np.sin(np.pi * np.clip(s, 0.0, 1.0)) ** 2


def taubin_smooth(v, f, iters=12, lam=0.5, mu=-0.53):
    """Taubin lambda|mu smoothing: rounds edges without the shrinkage of plain Laplacian."""
    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    deg = np.bincount(e.ravel(), minlength=n).astype(float)
    deg[deg == 0] = 1.0
    x = v.copy()
    for it in range(iters):
        for w in (lam, mu):
            acc = np.zeros_like(x)
            np.add.at(acc, e[:, 0], x[e[:, 1]])
            np.add.at(acc, e[:, 1], x[e[:, 0]])
            x = x + w * (acc / deg[:, None] - x)
    return x


def refine(mesh: Mesh, max_edge: float, max_faces: int = 300_000) -> Mesh:
    """Split every triangle 1-to-4 until the longest edge is below max_edge (or the face
    budget is hit). Needed before fairing CAD exports made of a few huge triangles."""
    m = mesh
    while m.n_faces * 4 <= max_faces:
        t = m.triangles()
        longest = np.max(np.linalg.norm(t - np.roll(t, 1, axis=1), axis=2))
        if longest <= max_edge:
            break
        v, f = m.v, m.f
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        es = np.sort(e, axis=1)
        uniq, inv = np.unique(es, axis=0, return_inverse=True)
        mids = (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2
        mid_idx = len(v) + inv.reshape(3, -1).T          # (F, 3): mid of edges 01, 12, 20
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        ab, bc, ca = mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]
        nf = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1),
                             np.stack([c, ca, bc], 1), np.stack([ab, bc, ca], 1)])
        m = Mesh(np.vstack([v, mids]), nf)
    return m


class Deformer:
    """Applies design vectors to one base mesh (body frame, metres, centred)."""

    def __init__(self, base: Mesh, variables, zones=(), n_bins=48):
        self.base = base
        self.zones = list(zones)
        self.variables = list(variables)
        lo, hi = base.bbox()
        self.lo, self.hi = lo, hi
        self.L = float(hi[0] - lo[0])
        self.H = float(hi[2] - lo[2])
        v = base.v
        self.s = (v[:, 0] - lo[0]) / max(self.L, 1e-12)
        # section centroid line yc(s), zc(s) from area-weighted face centroids per bin
        areas, _ = base.face_areas_normals()
        cen = base.triangles().mean(axis=1)
        sb = np.clip(((cen[:, 0] - lo[0]) / max(self.L, 1e-12) * n_bins).astype(int), 0, n_bins - 1)
        wsum = np.bincount(sb, weights=areas, minlength=n_bins)
        yc = np.bincount(sb, weights=areas * cen[:, 1], minlength=n_bins)
        zc = np.bincount(sb, weights=areas * cen[:, 2], minlength=n_bins)
        ok = wsum > 0
        centers = (np.arange(n_bins) + 0.5) / n_bins
        yc = np.interp(centers, centers[ok], yc[ok] / wsum[ok])
        zc = np.interp(centers, centers[ok], zc[ok] / wsum[ok])
        k = np.array([0.25, 0.5, 0.25])   # light smoothing so the axis line is not jagged
        for _ in range(3):
            yc = np.convolve(np.pad(yc, 1, mode="edge"), k, mode="valid")
            zc = np.convolve(np.pad(zc, 1, mode="edge"), k, mode="valid")
        self.yc = np.interp(self.s, centers, yc)
        self.zc = np.interp(self.s, centers, zc)
        self.fn = _bump_front(self.s)
        self.bk = _bump_back(self.s)
        self.md = _bump_mid(self.s)
        self._smooth = None
        comp, closed = shell_info(base)
        self.closed = bool(closed.all())
        self.base_volume = self.volume(base)
        # protected / design zones scale the displacement; keep zones are a feasibility test
        self.weight = deformation_weights(v, self.zones, self.L) if self.zones else None
        self.keep = KeepInsideCheck(base, self.zones, comp) if any(z.type == "keep" for z in self.zones) else None

    def neutral(self):
        return np.array([v.neutral for v in self.variables])

    def to_values(self, x01):
        """Map a unit-cube vector to variable values."""
        x01 = np.clip(np.asarray(x01, float), 0.0, 1.0)
        lo = np.array([v.lo for v in self.variables])
        hi = np.array([v.hi for v in self.variables])
        return lo + x01 * (hi - lo)

    def to_unit(self, values):
        lo = np.array([v.lo for v in self.variables])
        hi = np.array([v.hi for v in self.variables])
        return np.where(hi > lo, (np.asarray(values, float) - lo) / np.maximum(hi - lo, 1e-12), 0.0)

    def apply(self, values) -> Mesh:
        p = {v.key: float(x) for v, x in zip(self.variables, values)}
        v = self.base.v
        if p.get("fairing", 0.0) > 0:
            if self._smooth is None:
                self._smooth = taubin_smooth(v, self.base.f)
            v = v + p["fairing"] * (self._smooth - v)
        sy = 1.0 + p.get("nose_radial", 0.0) * self.fn + p.get("tail_taper", 0.0) * self.bk \
            + p.get("mid_width", 0.0) * self.md
        sz = 1.0 + p.get("nose_radial", 0.0) * self.fn + p.get("tail_taper", 0.0) * self.bk \
            + p.get("mid_height", 0.0) * self.md
        sy = np.maximum(sy, 0.05)
        sz = np.maximum(sz, 0.05)
        dz = self.H * (-p.get("nose_droop", 0.0) * self.fn + p.get("tail_sweep", 0.0) * self.bk)
        out = np.empty_like(v)
        out[:, 1] = self.yc + (v[:, 1] - self.yc) * sy
        out[:, 2] = self.zc + (v[:, 2] - self.zc) * sz + dz
        stretch = 1.0 + p.get("length", 0.0)
        xm = 0.5 * (self.lo[0] + self.hi[0])
        out[:, 0] = xm + (v[:, 0] - xm) * stretch
        if self.weight is not None:
            out = self.base.v + self.weight[:, None] * (out - self.base.v)
        return Mesh(out, self.base.f)

    def feasible(self, mesh: Mesh, min_volume=0.0, max_volume=float("inf")):
        """Volume limits and keep-inside zones; cheap enough to run before every CFD run."""
        if self.base_volume:
            r = self.volume_ratio(mesh)
            if not (min_volume <= r <= max_volume):
                return False
        return self.keep is None or self.keep.ok(mesh)

    def volume(self, mesh: Mesh):
        """Enclosed volume, or None for open meshes (no volume constraint is applied then)."""
        return abs(mesh.signed_volume()) if self.closed else None

    def volume_ratio(self, mesh: Mesh):
        if not self.closed or not self.base_volume:
            return 1.0
        return abs(mesh.signed_volume()) / self.base_volume

    def envelope(self, rotation=None):
        """Bounding box covering every shape the design space can produce (for a fixed grid)."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        nvar = len(self.variables)
        corners = [np.zeros(nvar), np.ones(nvar)]
        for i in range(nvar):
            for val in (0.0, 1.0):
                x = np.full(nvar, 0.5)
                x[i] = val
                corners.append(x)
        for x in corners:
            m = self.apply(self.to_values(x))
            if rotation is not None:
                m = m.transformed(R=rotation)
            a, b = m.bbox()
            lo = np.minimum(lo, a)
            hi = np.maximum(hi, b)
        base = self.base if rotation is None else self.base.transformed(R=rotation)
        a, b = base.bbox()
        return np.minimum(lo, a), np.maximum(hi, b)
