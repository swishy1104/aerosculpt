"""Design spaces the optimizer searches: named template parameters, or the smooth whole-body
deformation of an uploaded mesh. Both map a dict of variable values to a candidate in the body
frame (metres, nose -x, z up, centred on the baseline) and know which candidates are allowed.

Constraints, cheapest first, all checked before any CFD is spent on a candidate:
  template rules     (nose longer than the body, tip radius too large, ...)
  overall size       length / width / height limits
  volume             within the chosen fraction of the baseline (closed bodies)
  wall thickness     thinnest section (templates)
  wing area          held fixed when lift is not modelled (uploads)
  regions            keep-inside volumes stay enclosed; protected regions stay put
  validity           every shell watertight, no shell passing through itself
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import checks, templates as T
from ..geometry.mesh import Mesh, projected_area
from ..geometry.zones import KeepInsideCheck


@dataclass
class Var:
    key: str
    label: str
    unit: str
    lo: float
    hi: float
    base: float
    choices: list = None      # categorical (template profile family, airfoil family, fin count)
    group: str = ""

    @property
    def categorical(self):
        return self.choices is not None

    def public(self):
        return {"key": self.key, "label": self.label, "unit": self.unit, "lo": self.lo, "hi": self.hi,
                "base": self.base, "choices": self.choices, "group": self.group}


@dataclass
class Candidate:
    values: dict
    mesh: Mesh                       # body frame
    groups: np.ndarray = None
    wing: object = None              # Built (templates): surfaces for the wing model
    info: dict = field(default_factory=dict)
    reason: str = None               # why it may not be simulated (None = fine)


class TemplateSpace:
    """Unlocked template parameters within their bounds; locked ones keep the baseline value."""

    def __init__(self, tid, base_values, param_cfg, frame_center, constraints, zones=()):
        self.tid = tid
        self.base_values = T.clean(tid, base_values)
        self.center = np.asarray(frame_center, float)
        self.c = constraints
        self.cfg = param_cfg or {}
        self.vars = self._vars(self.base_values)
        self.base = self.build(self.base_values)
        self.base_volume = self.base.info.get("volume") or 0.0
        self.keep_zones = [z for z in zones if z.type == "keep"]
        self.protect = [z for z in zones if z.type == "protect"]
        self.keep_base = None
        if self.keep_zones:
            k = KeepInsideCheck(self.base.mesh, self.keep_zones, self.base.mesh.components())
            self.keep_base = k.base_frac
        self.protect_pts = []
        L = self.base.info.get("overall_length", 1.0)
        self.protect_tol = max(0.002 * L, 5e-4)
        for z in self.protect:
            v = self.base.mesh.v
            pts = v[z.distance(v) <= 0.0]
            if len(pts):
                self.protect_pts.append(pts[:: max(1, len(pts) // 400)])
        self.base_check = checks.check(self.base.mesh)

    def _vars(self, values):
        out = []
        for p in T.get(self.tid)["params"]:
            cfg = self.cfg.get(p.key, {})
            if cfg.get("locked", p.locked) or not T.active(p, values):
                continue
            base = values[p.key]
            if p.kind == "choice":
                out.append(Var(p.key, p.label, "choice", 0, 0, base, [c[0] for c in p.choices], p.group))
                continue
            lo, hi = cfg.get("bounds") or p.bounds or (p.lo, p.hi)
            lo, hi = max(float(lo), p.lo), min(float(hi), p.hi)
            lo, hi = min(lo, base), max(hi, base)
            if hi - lo < 1e-12:
                continue
            out.append(Var(p.key, p.label, p.unit, lo, hi, float(base), None, p.group))
        return out

    def rebase(self, values):
        """After a choice is adopted (e.g. a power-law nose), parameters that only matter for
        it (the exponent) join the search. Returns the newly active variables."""
        old = {v.key for v in self.vars}
        self.vars = self._vars(T.clean(self.tid, values))
        return [v for v in self.vars if v.key not in old]

    def build(self, values, preview=False) -> Candidate:
        vals = dict(self.base_values)
        vals.update(values or {})
        b = T.build(self.tid, vals, preview=preview)
        mesh = b.mesh.transformed(t=-self.center)
        return Candidate(T.clean(self.tid, vals), mesh, b.groups, b, dict(b.info), "; ".join(b.problems) or None)

    def fast_check(self, cand: Candidate):
        """Cheap constraints (milliseconds)."""
        if cand.reason:
            return cand.reason
        c = self.c
        info = cand.info
        for k, lim in (("overall_length", "max_length"), ("overall_width", "max_width"), ("overall_height", "max_height")):
            if c.get(lim) and info.get(k, 0) > c[lim] * (1 + 1e-9):
                return f"{k.split('_')[1]} over the {c[lim]:g} m limit"
        if self.base_volume and cand.info.get("volume"):
            r = cand.info["volume"] / self.base_volume
            if not (c.get("min_volume", 0.0) - 1e-9 <= r <= c.get("max_volume", 1e9) + 1e-9):
                return f"volume {r:.0%} of the original is outside the allowed range"
        if c.get("min_thickness") and info.get("min_thickness", 1e9) < c["min_thickness"] * (1 - 1e-9):
            return f"thinnest section {info['min_thickness'] * 1000:.1f} mm is under the {c['min_thickness'] * 1000:g} mm minimum"
        return None

    def full_check(self, cand: Candidate):
        """Regions and mesh validity (tens of milliseconds)."""
        r = self.fast_check(cand)
        if r:
            return r
        if self.keep_zones:
            k = KeepInsideCheck.__new__(KeepInsideCheck)
            k.zones = self.keep_zones
            k.comp = np.ascontiguousarray(cand.mesh.components(), dtype=np.int64)
            k.ncomp = int(k.comp.max()) + 1
            k.points = [z.surface_points() for z in self.keep_zones]
            for i, f0 in enumerate(self.keep_base):
                if k.fraction(cand.mesh, i) < f0 - 1e-9:
                    return f"keep-inside region '{self.keep_zones[i].name or i + 1}' is no longer enclosed"
        if self.protect_pts:
            from .runner import _surface_distance
            for z, pts in zip(self.protect, self.protect_pts):
                if _surface_distance(pts, cand.mesh).max() > self.protect_tol:
                    return f"protected region '{z.name or 'protected'}' would move"
        bad = checks.check(cand.mesh)
        if bad and not self.base_check:
            return bad
        return None

    def label(self, key):
        for v in self.vars:
            if v.key == key:
                return v.label
        return key


class DeformSpace:
    """The whole-body deformation of an uploaded mesh (geometry/ffd.py), as named variables."""

    def __init__(self, deformer, constraints):
        self.dfm = deformer
        self.c = constraints
        self.vars = [Var(v.key, v.label, "ratio" if v.unit == "%" else v.unit, v.lo, v.hi, v.neutral) for v in deformer.variables]
        self.base_values = {v.key: v.base for v in self.vars}
        self.base = self.build(self.base_values)
        self.base_planform = projected_area(self.base.mesh, 2)
        self.base_closed, _ = checks.closed_shells(self.base.mesh)
        self.base_check = checks.check(self.base.mesh)

    def build(self, values, preview=False) -> Candidate:
        vals = dict(self.base_values)
        vals.update(values or {})
        mesh = self.dfm.apply(np.array([vals[v.key] for v in self.vars]))
        lo, hi = mesh.bbox()
        info = {"overall_length": float(hi[0] - lo[0]), "overall_width": float(hi[1] - lo[1]),
                "overall_height": float(hi[2] - lo[2]), "volume_ratio": self.dfm.volume_ratio(mesh)}
        return Candidate(vals, mesh, None, None, info, None)

    def fast_check(self, cand: Candidate):
        c = self.c
        info = cand.info
        for k, lim in (("overall_length", "max_length"), ("overall_width", "max_width"), ("overall_height", "max_height")):
            if c.get(lim) and info[k] > c[lim] * (1 + 1e-9):
                return f"{k.split('_')[1]} over the {c[lim]:g} m limit"
        r = info["volume_ratio"]
        if not (c.get("min_volume", 0.0) - 1e-9 <= r <= c.get("max_volume", 1e9) + 1e-9):
            return f"volume {r:.0%} of the original is outside the allowed range"
        if c.get("hold_wing_area"):
            pa = projected_area(cand.mesh, 2)
            if pa < self.base_planform * (1 - c.get("wing_area_tol", 0.01)):
                return f"planform (wing) area would shrink by {1 - pa / self.base_planform:.1%}"
        return None

    def full_check(self, cand: Candidate):
        r = self.fast_check(cand)
        if r:
            return r
        if self.dfm.keep is not None and not self.dfm.keep.ok(cand.mesh):
            return "a keep-inside region is no longer enclosed"
        if self.base_check is None:
            bad = checks.check(cand.mesh)
            if bad:
                return bad
        return None

    def label(self, key):
        for v in self.vars:
            if v.key == key:
                return v.label
        return key


def unit_to_values(space, keys, x01):
    out = {}
    for k, x in zip(keys, x01):
        v = next(v for v in space.vars if v.key == k)
        out[k] = v.lo + float(np.clip(x, 0, 1)) * (v.hi - v.lo)
    return out


def values_to_unit(space, keys, values):
    out = []
    for k in keys:
        v = next(v for v in space.vars if v.key == k)
        out.append((values[k] - v.lo) / max(v.hi - v.lo, 1e-12))
    return np.array(out)
