"""Turn a job spec into CFD runs over a flight envelope, an optimization history and geometry.

A design is evaluated at every angle of attack in the envelope (one CFD run per angle and
tunnel Reynolds number); forces at every airspeed follow from those runs without further
simulation. Bodies that are mirror-symmetric top-to-bottom only need the non-negative angles.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..cfd.analysis import FIDELITY, RETIRED, Evaluator, at_speed, isa, mirror_run, reynolds_groups
from numba import njit
from ..geometry.ffd import refine
from ..geometry.mesh import UNIT_SCALE, Mesh, body_to_wind, load_mesh, projected_area
from ..geometry.zones import from_spec

OBJECTIVES = {
    "drag": "Minimize drag across the envelope",
    "ld": "Maximize best lift-to-drag ratio",
    "drag_keep_lift": "Minimize drag, keep lift",
}
# Lift-based goals are marked indicative until a lifting case passes validation (wing-ar6 on
# /validation; measured 2026-10-06: lift slope 50-66% low at every fidelity). An optimizer on
# unvalidated lift can find shapes that exploit solver error. Flip this once the wing case passes.
LIFT_VALIDATED = False
LIFT_OBJECTIVES = ("ld", "drag_keep_lift")


def objective_available(kind):
    return kind in OBJECTIVES and (LIFT_VALIDATED or kind not in LIFT_OBJECTIVES)


def live_run(r, label, aoa, speed, f, cd, trace):
    """Progress extra describing the CFD run in progress: which one, and its drag coefficient
    chunk by chunk as the flow settles (the workspace draws it live)."""
    if cd is not None and math.isfinite(cd):
        trace.append(round(float(cd), 5))
    return {"run": {"i": r.run_index + 1, "n": r.total_runs, "label": label, "aoa": aoa, "speed": speed,
                    "f": round(min(float(f), 1.0), 3), "trace": trace[-48:]}}


class JobCancelled(Exception):
    pass


class ModelFrame:
    """Maps between the uploaded file's frame/units and the body frame (metres, nose -x,
    z up, centred) the optimizer works in."""

    def __init__(self, raw: Mesh, units: str, forward: str, up: str, scale=None):
        # "custom": a scale model or odd units, sized from its real length (metres per file unit)
        self.scale = float(scale) if units == "custom" and scale else UNIT_SCALE.get(units, 1.0)
        self.units = units
        _, R0 = body_to_wind(forward, up)
        self.R0 = R0
        m = raw.transformed(s=self.scale).transformed(R=R0)
        lo, hi = m.bbox()
        self.center = (lo + hi) / 2
        self.body = m.transformed(t=-self.center)

    def to_file_frame(self, body_mesh: Mesh) -> Mesh:
        v = (body_mesh.v + self.center) @ self.R0       # inverse of v @ R0.T
        return Mesh(v / self.scale, body_mesh.f)


def load_model(path: Path, filename: str) -> Mesh:
    return load_mesh(Path(path).read_bytes(), filename).orient_outward()


def wind_rotation(aoa_deg, yaw_deg=0.0):
    a = math.radians(aoa_deg)
    b = math.radians(yaw_deg)
    Ra = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rb = np.array([[math.cos(b), math.sin(b), 0], [-math.sin(b), math.cos(b), 0], [0, 0, 1]])
    return Rb @ Ra


def is_z_symmetric(mesh: Mesh, tol=2e-3, max_points=20000):
    """True if the body is mirror-symmetric about its z = 0 plane (body frame, centred).

    Every vertex is mirrored (z -> -z) and its distance to the original surface measured;
    a symmetric CAD surface puts the mirrored vertices back on itself to within tessellation
    error. The largest miss must be below tol x the body's largest dimension."""
    from scipy.spatial import cKDTree
    lo, hi = mesh.bbox()
    size = float((hi - lo).max())
    # split large triangles so the nearest-centroid shortlist always holds the right one
    tri = refine(mesh, size / 60.0, max_faces=400_000).triangles()
    tree = cKDTree(tri.mean(axis=1))
    pts = mesh.v
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    mir = pts * np.array([1.0, 1.0, -1.0])
    k = min(16, len(tri))
    _, idx = tree.query(mir, k=k)
    idx = idx.reshape(len(mir), k)
    d = _point_tri_min_dist(np.ascontiguousarray(mir), np.ascontiguousarray(tri), np.ascontiguousarray(idx))
    return bool(d.max() < tol * size)


@njit(cache=True)
def _point_tri_min_dist(pts, tri, cand):
    """Distance from each point to the nearest of its candidate triangles (Ericson's
    closest-point-on-triangle)."""
    out = np.empty(pts.shape[0])
    for n in range(pts.shape[0]):
        best = 1e300
        px, py, pz = pts[n, 0], pts[n, 1], pts[n, 2]
        for c in range(cand.shape[1]):
            t = cand[n, c]
            ax, ay, az = tri[t, 0, 0], tri[t, 0, 1], tri[t, 0, 2]
            bx, by, bz = tri[t, 1, 0], tri[t, 1, 1], tri[t, 1, 2]
            cx, cy, cz = tri[t, 2, 0], tri[t, 2, 1], tri[t, 2, 2]
            abx, aby, abz = bx - ax, by - ay, bz - az
            acx, acy, acz = cx - ax, cy - ay, cz - az
            apx, apy, apz = px - ax, py - ay, pz - az
            d1 = abx * apx + aby * apy + abz * apz
            d2 = acx * apx + acy * apy + acz * apz
            if d1 <= 0.0 and d2 <= 0.0:
                qx, qy, qz = ax, ay, az
            else:
                bpx, bpy, bpz = px - bx, py - by, pz - bz
                d3 = abx * bpx + aby * bpy + abz * bpz
                d4 = acx * bpx + acy * bpy + acz * bpz
                cpx, cpy, cpz = px - cx, py - cy, pz - cz
                d5 = abx * cpx + aby * cpy + abz * cpz
                d6 = acx * cpx + acy * cpy + acz * cpz
                vc = d1 * d4 - d3 * d2
                vb = d5 * d2 - d1 * d6
                va = d3 * d6 - d5 * d4
                if d3 >= 0.0 and d4 <= d3:
                    qx, qy, qz = bx, by, bz
                elif d6 >= 0.0 and d5 <= d6:
                    qx, qy, qz = cx, cy, cz
                elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    v = d1 / (d1 - d3)
                    qx, qy, qz = ax + v * abx, ay + v * aby, az + v * abz
                elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                    w = d2 / (d2 - d6)
                    qx, qy, qz = ax + w * acx, ay + w * acy, az + w * acz
                elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                    qx, qy, qz = bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz)
                else:
                    den = 1.0 / (va + vb + vc)
                    v = vb * den
                    w = vc * den
                    qx, qy, qz = ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w
            dist = math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
            if dist < best:
                best = dist
        out[n] = best
    return out


def _surface_distance(pts, mesh, k=16):
    """Distance from each point to the nearest point of the mesh surface."""
    from scipy.spatial import cKDTree
    lo, hi = mesh.bbox()
    tri = refine(mesh, float((hi - lo).max()) / 60.0, max_faces=400_000).triangles()
    tree = cKDTree(tri.mean(axis=1))
    kk = min(k, len(tri))
    _, idx = tree.query(np.asarray(pts, float), k=kk)
    idx = np.asarray(idx).reshape(len(pts), kk)
    return _point_tri_min_dist(np.ascontiguousarray(pts, dtype=np.float64), np.ascontiguousarray(tri), np.ascontiguousarray(idx))


def polar_summary(polar):
    """Headline numbers of a polar: zero-lift-ish drag, best L/D, lift slope, mean drag."""
    rows = [(p["aoa"], s) for p in polar for s in p["speeds"]]
    aoas = np.array([p["aoa"] for p in polar])
    i0 = int(np.argmin(np.abs(aoas)))
    mid = len(polar[i0]["speeds"]) // 2
    best = max((r for r in rows if r[1]["ld"] is not None), key=lambda r: r[1]["ld"], default=None)
    out = {"cd_at_min_aoa": polar[i0]["speeds"][mid]["cd"], "aoa_ref": float(aoas[i0]),
           "speed_ref": polar[i0]["speeds"][mid]["v"],
           "drag_mean_N": float(np.mean([s["drag_N"] for _, s in rows])),
           "drag_max_N": float(np.max([s["drag_N"] for _, s in rows])),
           "ld_max": best[1]["ld"] if best else None,
           "ld_max_at": {"aoa": best[0], "v": best[1]["v"]} if best else None}
    if len(polar) >= 2:
        a = np.radians(aoas)
        out["cl_alpha"] = float(np.polyfit(a, [p["cl"] for p in polar], 1)[0])
        out["cm_alpha"] = float(np.polyfit(a, [p["cm_pitch"] for p in polar], 1)[0])
    if "net_N" in polar[0]["speeds"][0]:
        out["top_speed"] = top_speed(polar)
        out["thrust_max_N"] = float(max(s["thrust_N"] for _, s in rows))
    return out


def top_speed(polar):
    """Level-flight top speed estimate at the angle nearest zero: where thrust = drag."""
    row = min(polar, key=lambda p: abs(p["aoa"]))
    vs = [s["v"] for s in row["speeds"]]
    net = [s["net_N"] for s in row["speeds"]]
    for (v0, n0), (v1, n1) in zip(zip(vs, net), zip(vs[1:], net[1:])):
        if n0 > 0 >= n1:
            return {"v": v0 + (v1 - v0) * n0 / (n0 - n1), "aoa": row["aoa"], "bracketed": True}
    if net[-1] > 0:
        return {"v": vs[-1], "aoa": row["aoa"], "bracketed": False, "above": True}
    return {"v": vs[0], "aoa": row["aoa"], "bracketed": False, "above": False}


def objective(kind, polar, base_polar):
    """Scalar to minimize, comparing every (angle, speed) condition with the baseline so each
    condition counts equally whatever its dynamic pressure."""
    pairs = [(s, s0) for p, p0 in zip(polar, base_polar) for s, s0 in zip(p["speeds"], p0["speeds"])]
    drag_ratio = float(np.mean([s["drag_N"] / s0["drag_N"] for s, s0 in pairs if s0["drag_N"] > 0]))
    if kind == "drag":
        return drag_ratio
    if kind == "ld":
        lds = [s["ld"] for s, _ in pairs if s["ld"] is not None]
        return -max(lds) if lds else 0.0
    if kind == "drag_keep_lift":
        deficits = [max(0.0, (s0["lift_N"] - s["lift_N"]) / abs(s0["lift_N"]))
                    for s, s0 in pairs if s0["lift_N"] > 0.02 * s0["drag_N"]]
        return drag_ratio * (1.0 + 4.0 * (float(np.mean(deficits)) if deficits else 0.0))
    raise ValueError(f"unknown objective {kind!r}")


class Runner:
    def __init__(self, spec: dict, model_path: Path, filename: str, out_dir: Path,
                 progress=None, log=None, cancel=None):
        self.spec = spec
        self.model_path = Path(model_path)
        self.filename = filename
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._progress = progress or (lambda frac, stage, extra=None: None)
        self._log = log or (lambda msg: None)
        self._cancel = cancel or (lambda: False)
        self.cfd_runs = 0
        self.runs_by_fid = {}
        self.history = []
        self.t0 = time.time()
        self.run_index = 0
        self.total_runs = 1
        self.fields_wanted = True
        self.cache = None
        if spec.get("cache", True):
            from ..cfd.cache import default_cache
            try:
                self.cache = default_cache()
            except Exception:  # noqa: BLE001 - no cache directory: just run everything
                self.cache = None

    # ---- helpers ----------------------------------------------------------------------
    def log(self, msg):
        self._log(f"[{time.time() - self.t0:6.1f}s] {msg}")

    def check_cancel(self):
        if self._cancel():
            raise JobCancelled()

    def write_json(self, name, obj):
        if name == "result.json" and isinstance(obj, dict):
            from .. import __version__, build_id
            obj.setdefault("version", __version__)
            obj.setdefault("build", build_id())
        (self.out / name).write_text(json.dumps(obj, separators=(",", ":"), default=_json_default))

    def make_evaluator(self, lo, hi, fid):
        ev = Evaluator(lo, hi, self.frontal_plan, self.L_body, self.ref_area, self.ref_length, fid,
                       log=self.log, cancel=self._cancel, planform_area=self.planform_area, cache=self.cache,
                       ground=getattr(self, "ground", None))
        info = ev.info()
        self.log(f"tunnel {info['grid']} = {info['cells']:,} cells, dx {info['dx_m'] * 1000:.2f} mm, "
                 f"body {info['body_cells']} cells")
        return ev

    def setup(self):
        spec = self.spec
        m = spec.get("model", {})
        raw = load_model(self.model_path, self.filename)
        self.frame = ModelFrame(raw, m.get("units", "m"), m.get("forward", "-x"), m.get("up", "+z"), m.get("scale"))
        env = spec.get("envelope", {})
        self.speeds = sorted(float(v) for v in env.get("speeds", [20.0]))
        self.aoas = sorted(float(a) for a in env.get("aoas", [0.0]))
        self.yaw = float(env.get("yaw", 0.0))
        self.air = isa(float(env.get("altitude", 0.0)))
        mach = max(self.speeds) / self.air["a"]
        if mach > 1.2:
            raise ValueError(f"Mach {mach:.2f} is beyond this tunnel (max 1.2); use the Supersonic tunnel")
        self.mach_max = mach
        self.fid = FIDELITY[RETIRED.get(spec.get("fidelity", "draft"), spec.get("fidelity", "draft"))]
        zone_items = spec.get("zones", [])
        wheel_items = [z for z in zone_items if z.get("type") == "wheel"]
        # wheels are never reshaped: to the optimizer they are protected regions
        self.zones = from_spec([z for z in zone_items if z.get("type") in ("protect", "keep", "design")]
                               + [{**z, "type": "protect"} for z in wheel_items],
                               self.frame.scale, self.frame.R0, self.frame.center)
        self.apply_features()
        body = self.frame.body
        lo, hi = body.bbox()
        self.L_body = float(hi[0] - lo[0])
        self.ground = None
        g = spec.get("ground") or {}
        if g.get("enabled") and spec.get("tunnel", "general") != "supersonic":
            if any(abs(a) > 1e-9 for a in self.aoas) or abs(self.yaw) > 1e-9:
                self.log("on the rolling road the body sits level: angle of attack and yaw are fixed at 0")
            self.aoas, self.yaw = [0.0], 0.0
            from ..geometry.wheels import wheel_rims
            wheels = wheel_rims(from_spec(wheel_items, self.frame.scale, self.frame.R0, self.frame.center))
            self.ground = {"z": float(lo[2]) - max(float(g.get("clearance", 0.0) or 0.0), 0.0), "wheels": wheels}
            self.log(f"rolling road under the body, {len(wheels)} spinning wheel(s)")
        self.ref_area = float(env.get("ref_area") or projected_area(body, 0))
        self.planform_area = projected_area(body, 2)
        self.ref_length = float(env.get("ref_length") or self.L_body)
        self.re_by_speed, self.re_groups = reynolds_groups(self.speeds, self.L_body, self.air["nu"], self.fid.re_cap)

    def feature_frame(self):
        from ..geometry.features import Frame
        return Frame(self.frame.scale, self.frame.R0, self.frame.center)

    def apply_features(self):
        """Shape features (spec["features"]) on the model: the body every run sees. The model
        without them stays in base_body for the optimizer to build on."""
        self.features = [f for f in self.spec.get("features") or [] if f.get("enabled", True)]
        self.base_body = self.frame.body
        if self.spec.get("kind") == "optimize" and (self.spec.get("optimize") or {}).get("auto_ideas"):
            # "let AeroSculpt decide": its own ideas (boat-tail, rounded front, slants) join the search,
            # each optional, so only those that measurably help are kept
            from ..geometry.features import Frame, apply, propose
            body = apply(self.frame.body, self.features, self.feature_frame())[0] if self.features else self.frame.body
            have = {f["name"] for f in self.features}
            ideas = [f for f in propose(body, Frame(self.frame.scale, self.frame.R0, self.frame.center)) if f["name"] not in have]
            if ideas:
                self.log("generated ideas to try: " + ", ".join(f["name"].replace(" (proposed)", "").lower() for f in ideas))
            self.features = self.features + ideas
        if self.features:
            from ..geometry.features import apply
            body, problems, info = apply(self.frame.body, self.features, self.feature_frame())
            for pr in problems:
                self.log(f"shape feature: {pr}")
            self.log(f"{len(self.features)} shape feature(s) applied: {info['faces']:,} faces, "
                     f"{info['hidden_faces']:,} buried inside other parts")
            self.frame.body = body

    def runs_per_design(self):
        return len(self.sim_angles) * len(self.re_groups)

    def simulated_angles(self, body, can_mirror=True):
        sym = (can_mirror and self.yaw == 0.0 and any(a < 0 for a in self.aoas)
               and all(abs(z.center[2]) < 1e-3 * self.L_body and (z.R is None or z.z_symmetric) for z in self.zones)
               and is_z_symmetric(body))
        self.symmetric = bool(sym)
        if sym:
            self.log("body is symmetric top-to-bottom: negative angles are mirrored, not re-simulated")
            return sorted({abs(a) for a in self.aoas})
        return list(self.aoas)

    def count_run(self, ev, run):
        """Book a finished run (cached results cost nothing)."""
        if run.get("cached"):
            self.cached_runs = getattr(self, "cached_runs", 0) + 1
            return
        n = 2 if run.get("zoom") else 1                  # a zoom grid run is the outer run plus the fine box
        self.cfd_runs += n
        bill = getattr(ev, "bill_as", None) or ev.fid.name
        self.runs_by_fid[bill] = self.runs_by_fid.get(bill, 0) + n

    def run_for(self, aoa, i_speed):
        """The run behind envelope angle `aoa` and speed index `i_speed` of the last design."""
        mirror = self.symmetric and aoa < 0
        run = self.last_runs[(abs(aoa) if mirror else aoa, self.last_re[i_speed])]
        return mirror_run(run) if mirror else run

    def re_flight_for(self, re_sim, re_by_speed):
        """The flight Reynolds number a tunnel run stands for: the geometric mean over the speeds
        that share it (the wall model's log law depends on it only logarithmically)."""
        fl = [v * self.L_body / self.air["nu"] for v, r in zip(self.speeds, re_by_speed) if r == re_sim]
        return float(np.exp(np.mean(np.log(fl)))) if fl else None

    def re_for(self, ev):
        """Reynolds-number groups for an evaluator: the chosen level's, or a finer grid's own higher
        cap (the optimizer's confirmation one level up)."""
        if ev.fid.re_cap <= self.fid.re_cap:
            return self.re_by_speed, self.re_groups
        return reynolds_groups(self.speeds, self.L_body, self.air["nu"], ev.fid.re_cap)

    def evaluate_design(self, ev, body, label, angles, aoas=None, groups=None):
        """All CFD runs for one design -> (polar, fields per simulated angle). aoas: the envelope
        angles to report (default all); groups: part id per face for per-part forces."""
        aoas = list(self.aoas if aoas is None else aoas)
        re_by_speed, re_groups = self.re_for(ev)
        runs = {}
        for a in angles:
            wind = body.transformed(R=wind_rotation(a, self.yaw))
            for re in re_groups:
                self.check_cancel()
                frac0 = self.run_index / self.total_runs
                frac1 = (self.run_index + 1) / self.total_runs

                trace = []

                def prog(f, cd, frac0=frac0, frac1=frac1, a=a, trace=trace):
                    self._progress(0.02 + 0.96 * (frac0 + (frac1 - frac0) * min(f, 1.0)),
                                   f"{label} · {a:g}° · CFD run {self.run_index + 1} of {self.total_runs}",
                                   live_run(self, label, a, None, f, cd, trace))
                try:
                    runs[(a, re)] = ev.run(wind, re, key=(a, re), want_fields=(re == re_groups[0]) and self.fields_wanted,
                                           re_flight=self.re_flight_for(re, re_by_speed),
                                           progress=prog, groups=groups)
                except InterruptedError:
                    raise JobCancelled()
                self.run_index += 1
                self.count_run(ev, runs[(a, re)])
            self.transonic_anchor(wind, [runs[(a, re)] for re in re_groups])
        self.last_runs, self.last_re = runs, re_by_speed
        polar = []
        for aoa in aoas:
            mirror = self.symmetric and aoa < 0
            a_sim = abs(aoa) if mirror else aoa
            row = None
            speeds = []
            for v, re in zip(self.speeds, re_by_speed):
                run = runs[(a_sim, re)]
                if mirror:
                    run = mirror_run(run)
                speeds.append(at_speed(run, v, self.air, self.L_body))
                if re == re_groups[0]:
                    row = run
            mc = row.get("compress", {}).get("mach_crit") if row.get("compress") else None
            polar.append({"aoa": aoa, "mirror": mirror, "regime": row["regime"], "mach_crit": mc, "cl": row["cl"], "cs": row["cs"],
                          "cm_pitch": row["cm_pitch"], "cm_roll": row["cm_roll"], "cm_yaw": row["cm_yaw"],
                          "cd_pressure": row["cd_pressure"], "converged": row["converged"], "cd_std": row["cd_std"],
                          "blockage": row["blockage"], "seconds": row["seconds"], "speeds": speeds})
        fields = {a: runs[(a, re_groups[0])].get("fields") for a in angles}
        return polar, fields

    def transonic_anchor(self, wind, runs):
        """Supersonic surface-method pressure drag at Mach 1.2 and 1.3 for this shape and angle: the
        end point of the transonic estimate. Only needed when the envelope reaches past Mach 0.6."""
        if getattr(self, "mach_max", 0.0) <= 0.6:
            return
        from ..hyper.panel import Surface
        try:
            surf = Surface(wind, self.ref_area, self.ref_length)
            cdp = []
            for M in (1.2, 1.3):
                r = surf.evaluate(M, 0.0, 0.0, min(float(self.spec.get("envelope", {}).get("altitude", 0.0)), 80000.0),
                                  wall="fixed", t_wall_k=300.0)
                cdp.append(r["cd_wave"] + r["cd_base"])
            anchor = {"mach": [1.2, 1.3], "cdp": cdp}
        except Exception as e:  # noqa: BLE001 - fall back to the empirical drag rise
            self.log(f"transonic anchor unavailable ({type(e).__name__}); using the empirical drag-rise law")
            anchor = None
        for run in runs:
            run["transonic"] = anchor

    def save_fields(self, tag, fields, angles):
        """Visual products per simulated angle: JSON for the browser, and beside it (npz) the
        coarse 3D field and the surface pressure per face for the VTK export."""
        out = []
        for k, a in enumerate(angles):
            name = f"fields_{tag}_{k}.json"
            fj = dict(fields[a] or {})
            vol = fj.pop("_volume", None)
            cpf = fj.pop("_cp_face", None)
            self.write_json(name, fj)
            if vol is not None:
                np.savez_compressed(self.out / f"volume_{tag}_{k}.npz", cp_face=cpf if cpf is not None else np.zeros(0),
                                    **{key: np.asarray(val) for key, val in vol.items()})
            out.append({"aoa": a, "file": name})
        return out

    def field_index(self, saved):
        """For every envelope angle: which saved file to show and whether to mirror it."""
        by = {e["aoa"]: e["file"] for e in saved}
        return [{"aoa": a, "file": by[abs(a) if self.symmetric and a < 0 else a],
                 "mirror": bool(self.symmetric and a < 0)} for a in self.aoas]

    def common(self, ev):
        return {"setup": {**ev.info(), "re_groups": self.re_groups, "symmetric": self.symmetric,
                          "simulated_angles": self.sim_angles, "air": self.air,
                          "regimes": sorted({"direct" if r < self.fid.re_cap else "hybrid" for r in self.re_groups}),
                          "runs_per_design": self.runs_per_design()},
                "envelope": {"speeds": self.speeds, "aoas": self.aoas, "yaw": self.yaw,
                             "altitude": self.spec.get("envelope", {}).get("altitude", 0.0)},
                "model": self.spec.get("model", {})}

    # ---- entry points -------------------------------------------------------------------
    def run(self):
        self.setup()
        kind = self.spec.get("kind", "analyze")
        if kind == "analyze":
            return self.run_analyze()
        if kind == "optimize":
            return self.run_optimize()
        raise ValueError(f"unknown job kind {kind!r}")

    def run_analyze(self):
        body = self.frame.body
        self.sim_angles = self.simulated_angles(body)
        grid_study = bool(self.spec.get("grid_study"))
        self.total_runs = self.runs_per_design() + (2 * len(self.re_groups) if grid_study else 0)
        lo, hi, self.frontal_plan = self.envelope_box([body])
        self._progress(0.01, "Building virtual wind tunnel")
        ev = self.make_evaluator(lo, hi, self.fid)
        if self.spec.get("refine"):
            # local refinement: each run again on a nested grid about twice as fine around the body
            from ..cfd.zoom import ZoomedEvaluator
            self.total_runs *= 2
            ev = ZoomedEvaluator(ev, log=self.log)
        polar, fields = self.evaluate_design(ev, body, "Flight envelope", self.sim_angles)
        saved = self.save_fields("baseline", fields, self.sim_angles)
        (self.out / "baseline_body.stl").write_bytes(body.to_stl_bytes())
        result = {"kind": "analyze", **self.common(ev),
                  "baseline": {"polar": polar, "summary": polar_summary(polar)},
                  "fields": {"baseline": self.field_index(saved)},
                  "meshes": {"baseline": "baseline_body.stl"}}
        if grid_study:
            result["grid_study"] = self.grid_study(body, lo, hi, ev, polar)
        result.update({"cfd_runs": self.cfd_runs, "elapsed_s": time.time() - self.t0})
        self.write_json("result.json", result)
        self._progress(1.0, "Done")
        return result

    def grid_study(self, body, lo, hi, ev_fine, polar_fine):
        """The design again on two coarser grids (cells / 2.2 and / 4.84, refinement ratio 1.3) at the
        angle nearest zero; Richardson extrapolation and the GCI of drag (and lift, when there is
        lift to speak of) on the chosen grid. The coarse runs bill at the exploration rate."""
        import dataclasses
        from ..cfd.gridcheck import CELLS_STEP, gci
        from ..plans import EXPLORE_FID
        a0 = min(self.aoas, key=abs)
        sim = [abs(a0) if self.symmetric else a0]
        k0 = [q["aoa"] for q in polar_fine].index(a0)
        rows = [(ev_fine.domain.dx, polar_fine[k0])]
        for k, factor in enumerate((CELLS_STEP, CELLS_STEP ** 2), 1):
            fid = dataclasses.replace(self.fid, cells=int(self.fid.cells / factor))
            ev = self.make_evaluator(lo, hi, fid)
            ev.bill_as = EXPLORE_FID.get(self.fid.name, "coarse")
            self.fields_wanted = False
            pol, _ = self.evaluate_design(ev, body, f"Grid check {k}/2", sim, aoas=[a0])
            rows.append((ev.domain.dx, pol[0]))
        self.fields_wanted = True
        (h1, p1), (h2, p2), (h3, p3) = rows
        out = {"aoa": a0, "speed": self.speeds[0], "cells": [int(self.fid.cells / CELLS_STEP ** i) for i in range(3)],
               "dx_m": [h1, h2, h3]}
        cd = [q["speeds"][0]["cd"] for q in (p1, p2, p3)]
        out["cd"] = gci(*cd, h1, h2, h3)
        cl = [q["cl"] for q in (p1, p2, p3)]
        if max(abs(c) for c in cl) > 0.05:
            out["cl"] = gci(*cl, h1, h2, h3)
        g = out["cd"]
        self.log(f"grid check: Cd {cd[2]:.4f} / {cd[1]:.4f} / {cd[0]:.4f} (coarse to fine), "
                 f"extrapolated {g['extrapolated']:.4f}, GCI {g['gci_fine']:.1%}")
        return out

    def envelope_box(self, bodies, deformer=None):
        """Grid box covering every shape at every simulated angle (+3 % margin) and the
        largest frontal area seen, for domain sizing."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        af = 0.0
        for a in self.sim_angles:
            R = wind_rotation(a, self.yaw)
            if deformer is not None:
                a_lo, a_hi = deformer.envelope(R)
                lo, hi = np.minimum(lo, a_lo), np.maximum(hi, a_hi)
            for b in bodies:
                w = b.transformed(R=R)
                b_lo, b_hi = w.bbox()
                lo, hi = np.minimum(lo, b_lo), np.maximum(hi, b_hi)
                af = max(af, projected_area(w, 0))
        pad = 0.03 * (hi - lo)
        return lo - pad, hi + pad, af

    def run_optimize(self):
        from .driver import optimize
        return optimize(self)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")
