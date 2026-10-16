"""Aerodynamic evaluation: CFD runs and the forces derived from them across a flight envelope.

Physics model, stated plainly so it can be shown to users:

* The flow is a 3D lattice-Boltzmann large-eddy simulation (D3Q19, regularized collision,
  Smagorinsky sub-grid model, bulk-viscosity acoustic damping) on a uniform grid with
  interpolated bounce-back walls placed on the true triangle surface.
* Resolution limits the Reynolds number the grid can carry. When the flight Reynolds number
  is within that cap the simulated total force is used directly ("direct"). Above it the
  tunnel runs at the cap ("hybrid"): pressure (form) drag, lift and moments come from the
  simulation, and skin friction is re-scaled to the flight Reynolds number with the
  Prandtl-Schlichting flat-plate law, weighted by the simulated local edge velocity.
* Speed sweeps cost nothing extra in the hybrid regime: the simulated coefficients do not
  depend on speed there, so one run per angle gives the forces at every speed exactly.
* Forces are corrected for tunnel blockage with a Maskell-type factor (theta = 2.5),
  calibrated against the sphere drag curve (see validation.py).
* Wall model: in the hybrid regime with a turbulent flight Reynolds number (above
  WALL_MODEL_RE) the wall links slip tangentially so the shear the coarse grid feels matches
  the log law of the wall at the flight Reynolds number. A grid capped at Re 2e4-8e4 otherwise
  grows thick, laminar-like boundary layers that separate too early and over-reward shapes that
  avoid separation; the wall model gives the attached, thin turbulent layer of the real flow.
* Road: cars can run over a moving road (a rolling-road rig: the floor moves with the air, so
  there is no floor boundary layer) with spinning wheels (moving-wall bounce-back on the rims).
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..geometry.mesh import Mesh, metrics, projected_area
from .tunnel import FlowSettings, RunStats, Tunnel, converge, near_field_scale, plan_domain

MASKELL_THETA = 2.5
WALL_MODEL_RE = 5e5        # flight Reynolds number (body length) above which the boundary layer is turbulent
SNAPSHOT_BUDGET_BYTES = 1.2e9


# ---- atmosphere ------------------------------------------------------------------
def isa(altitude_m: float):
    """International Standard Atmosphere (troposphere + lower stratosphere)."""
    h = float(np.clip(altitude_m, -500.0, 20000.0))
    if h <= 11000.0:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** 5.255877
    else:
        T = 216.65
        p = 22632.06 * math.exp(-9.80665 * (h - 11000.0) / (287.053 * T))
    rho = p / (287.053 * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    a = math.sqrt(1.4 * 287.053 * T)
    return {"T": T, "p": p, "rho": rho, "mu": mu, "nu": mu / rho, "a": a}


def cf_flat_plate(re: float):
    """Average skin-friction coefficient of a flat plate (Prandtl-Schlichting, transition 5e5)."""
    re = max(re, 10.0)
    if re < 5e5:
        return 1.328 / math.sqrt(re)
    return 0.455 / math.log10(re) ** 2.58 - 1700.0 / re


# ---- fidelity ----------------------------------------------------------------------
@dataclass
class Fidelity:
    name: str
    cells: int
    re_cap: float
    u_lat: float = 0.08
    passes_per_chunk: float = 1.5
    avg_chunks: int = 4
    extra_chunks: int = 24       # room for unsteady wakes to average down (steady flows stop early)
    tol: float = 0.01
    se_tol: float = 0.006        # standard error of the time-averaged drag, relative
    internal: bool = False        # used by the optimizer only, never offered as a choice


FIDELITY = {
    # exploration grid for the optimizer: ranks candidates, never reports a final number
    "coarse": Fidelity("coarse", 120_000, 1.2e4, internal=True),
    "draft": Fidelity("draft", 300_000, 2.0e4),
    "standard": Fidelity("standard", 1_000_000, 4.0e4),
    "high": Fidelity("high", 2_500_000, 8.0e4),
}
# retired levels still found in saved projects and queued jobs
RETIRED = {"ultra": "high"}


def fidelity_name(name):
    """A user-facing fidelity name, with retired levels mapped onto their replacement."""
    name = RETIRED.get(name, name)
    if name not in FIDELITY or FIDELITY[name].internal:
        raise KeyError(name)
    return name


def throughput(cells=300_000, steps=150):
    """Measured lattice updates per second of the tunnel this machine uses (CPU or GPU), on a
    draft-sized grid around a sphere. Used for the run-time estimates shown before a job."""
    import time
    from ..geometry.samples import icosphere
    from .gpu import device_name, make_tunnel
    mesh = icosphere(0.5, 2)
    lo, hi = mesh.bbox()
    ev = Evaluator(lo, hi, projected_area(mesh, 0), 1.0, 1.0, 1.0, Fidelity("bench", cells, 1e3))
    t = make_tunnel(ev.domain, ev._flow_for(1e3))
    t.set_geometry(mesh)
    t.step(20)                       # compile / warm the kernels
    t0 = time.perf_counter()
    t.step(steps)
    dt = time.perf_counter() - t0
    return {"backend": t.backend, "device": device_name() if t.backend == "gpu" else None,
            "mlups": round(ev.domain.n_cells * steps / dt / 1e6, 1), "measured_at": time.time()}


def seconds_per_run(cells, mlups):
    """Expected wall time of one cold CFD run on a grid of `cells` at `mlups` (measured above).
    Step counts grow with the grid length: about 2,000 steps on the 300k-cell draft grid."""
    steps = 2000.0 * (cells / 300_000) ** (1 / 3)
    return steps * cells / (mlups * 1e6) + 2.0 + cells * 2e-6     # + voxelizing and post-processing


@dataclass
class Conditions:
    """A single flight condition (used by validation and one-off analyses)."""
    speed: float = 20.0
    altitude: float = 0.0
    ref_area: float | None = None
    ref_length: float | None = None

    def air(self):
        return isa(self.altitude)


def reynolds_groups(speeds, length, nu, re_cap):
    """Map each speed to the Reynolds number the tunnel must run at.

    Every speed above the grid's cap shares one hybrid run; speeds below it are resolved
    directly and each needs its own run. Returns (re_sim per speed, unique re_sim, high first)."""
    re_sim = [round(min(v * length / nu, re_cap), 6) for v in speeds]
    return re_sim, sorted(set(re_sim), reverse=True)


def at_speed(run, speed, air, length):
    """Forces at one airspeed from a CFD run (no extra simulation): skin friction re-scaled to the
    flight Reynolds number and, above low Mach numbers, compressibility (see compress.py)."""
    from . import compress as cx
    re = speed * length / air["nu"]
    M = speed / air["a"]
    cl, cs, cm_pitch = run["cl"], run["cs"], run["cm_pitch"]
    if run["regime"] == "direct":
        cd_p = run["cd_pressure"]
        cd_f = run["cd_tunnel"] - run["cd_pressure"]
        cf = None
    else:
        cf = cf_flat_plate(re)
        cd_f = cf * run["friction_integral"]
        cd_p = run["cd_pressure"]
    bins = run.get("compress")
    mc = bins["mach_crit"] if bins else None
    comp = "incompressible" if M < 0.3 else "corrected"
    if bins is not None and M > 0.05:
        sgn = np.array([1.0, 1.0, -1.0]) if run.get("mirror") else np.ones(3)   # mirrored runs flip lift
        msg = np.array([-1.0, -1.0, 1.0]) if run.get("mirror") else np.ones(3)  # ... and roll and pitch
        Mp = min(M, mc)
        dF, dM = cx.pressure_delta(bins, Mp)
        dF, dM = dF * sgn, dM * msg
        cl, cs, cm_pitch = cl + dF[2], cs + dF[1], cm_pitch + dM[1]
        if M <= mc:
            cd_p = cd_p + dF[0]
        else:
            comp = "transonic estimate"
            m0 = max(mc - 0.02, 0.05)
            cdp_mc = run["cd_pressure"] + dF[0]
            slope = (cdp_mc - (run["cd_pressure"] + cx.pressure_delta(bins, m0)[0][0])) / (mc - m0)
            cd_p = cx.transonic_cdp(M, mc, cdp_mc, max(slope, 0.0), run.get("transonic"))
        cd_f = cd_f * (1.0 - 0.1 * min(M, 1.2) ** 2)
    cd = cd_p + cd_f
    q = 0.5 * air["rho"] * speed ** 2
    a = run["ref_area"]
    drag = cd * q * a
    return {"v": speed, "re": re, "mach": M, "cd": cd, "cd_friction": cd_f, "cd_pressure": cd_p, "cf": cf,
            "q": q, "drag_N": drag, "lift_N": cl * q * a, "side_N": cs * q * a, "cl": cl, "cm_pitch": cm_pitch,
            "ld": cl / cd if cd > 1e-9 else None, "power_W": drag * speed,
            "compressibility": comp, "mach_crit": mc}


def mirror_run(run):
    """Coefficients at -alpha for a body that is symmetric about its z = 0 plane."""
    out = {k: v for k, v in run.items() if k != "fields"}
    for k in ("cl", "cl_pressure", "cm_roll", "cm_pitch"):
        out[k] = -run[k]
    out["mirror"] = True
    return out


def part_run(run, g):
    """A run restricted to one part (see Evaluator.run groups): its own pressure drag, lift,
    friction integral and compressibility record. Friction resolved directly by the tunnel (low
    Reynolds numbers) is shared out in proportion to the parts' friction integrals."""
    p = run["groups"][g]
    out = {k: v for k, v in run.items() if k not in ("fields", "groups")}
    tot = max(run["friction_integral"], 1e-12)
    out.update({"cd_pressure": p["cd_pressure"], "cl": p["cl_pressure"], "cl_pressure": p["cl_pressure"], "cs": 0.0,
                "cm_pitch": 0.0, "cm_roll": 0.0, "cm_yaw": 0.0, "friction_integral": p["friction_integral"],
                "compress": p["compress"],
                "cd_tunnel": p["cd_pressure"] + (run["cd_tunnel"] - run["cd_pressure"]) * p["friction_integral"] / tot})
    if run.get("mirror"):
        out["cl"] = -out["cl"]
    return out


# ---- sampling helpers -------------------------------------------------------------
@njit(cache=True)
def _sample_fluid(field, solid, pts):
    """Trilinear interpolation that ignores solid nodes (weights renormalised).
    Returns NaN where all eight neighbours are solid or the point is outside."""
    nx, ny, nz = solid.shape
    out = np.empty(pts.shape[0])
    for p in range(pts.shape[0]):
        x = pts[p, 0]; y = pts[p, 1]; z = pts[p, 2]
        i = int(math.floor(x)); j = int(math.floor(y)); k = int(math.floor(z))
        if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
            out[p] = np.nan
            continue
        fx = x - i; fy = y - j; fz = z - k
        acc = 0.0
        wsum = 0.0
        for di in range(2):
            wx = fx if di else 1.0 - fx
            for dj in range(2):
                wy = fy if dj else 1.0 - fy
                for dk in range(2):
                    wz = fz if dk else 1.0 - fz
                    if solid[i + di, j + dj, k + dk]:
                        continue
                    w = wx * wy * wz
                    acc += w * field[i + di, j + dj, k + dk]
                    wsum += w
        out[p] = acc / wsum if wsum > 1e-9 else np.nan
    return out


@njit(cache=True)
def _trace(ux, uy, uz, solid, seeds, h, max_steps, keep_every):
    """RK2 streamlines through the mean velocity field (lattice units).
    Returns (points (N, 4) = x, y, z, speed; line start offsets)."""
    nx, ny, nz = solid.shape
    nseed = seeds.shape[0]
    cap = nseed * (max_steps // keep_every + 2)
    pts = np.empty((cap, 4))
    starts = np.zeros(nseed + 1, dtype=np.int64)
    n = 0
    for s in range(nseed):
        x = seeds[s, 0]; y = seeds[s, 1]; z = seeds[s, 2]
        starts[s] = n
        for step in range(max_steps):
            i = int(math.floor(x)); j = int(math.floor(y)); k = int(math.floor(z))
            if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
                break
            if solid[i, j, k] and solid[i + 1, j + 1, k + 1]:
                break
            v = _interp3(ux, uy, uz, solid, x, y, z)
            sp = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            if sp < 1e-6:
                break
            if step % keep_every == 0 and n < cap:
                pts[n, 0] = x; pts[n, 1] = y; pts[n, 2] = z; pts[n, 3] = sp
                n += 1
            xm = x + 0.5 * h * v[0] / sp
            ym = y + 0.5 * h * v[1] / sp
            zm = z + 0.5 * h * v[2] / sp
            v2 = _interp3(ux, uy, uz, solid, xm, ym, zm)
            sp2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
            if sp2 < 1e-6:
                break
            x += h * v2[0] / sp2
            y += h * v2[1] / sp2
            z += h * v2[2] / sp2
    starts[nseed] = n
    return pts[:n], starts


@njit(cache=True)
def _interp3(ux, uy, uz, solid, x, y, z):
    nx, ny, nz = solid.shape
    out = np.zeros(3)
    i = int(math.floor(x)); j = int(math.floor(y)); k = int(math.floor(z))
    if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return out
    fx = x - i; fy = y - j; fz = z - k
    wsum = 0.0
    for di in range(2):
        wx = fx if di else 1.0 - fx
        for dj in range(2):
            wy = fy if dj else 1.0 - fy
            for dk in range(2):
                wz = fz if dk else 1.0 - fz
                if solid[i + di, j + dj, k + dk]:
                    continue
                w = wx * wy * wz
                out[0] += w * ux[i + di, j + dj, k + dk]
                out[1] += w * uy[i + di, j + dj, k + dk]
                out[2] += w * uz[i + di, j + dj, k + dk]
                wsum += w
    if wsum > 1e-9:
        out /= wsum
    return out


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _quantize(a, lo, hi):
    q = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 254.0
    finite = np.isfinite(a)
    out = np.round(np.where(finite, q, 0.0)).astype(np.uint8)
    out[~finite] = 255
    return out


# ---- the evaluation ------------------------------------------------------------------
class Evaluator:
    """Runs CFD for related shapes and flight conditions on one fixed grid.

    The grid is sized once for an envelope that covers every shape and angle that will be
    run, so runs can warm-start from each other. Each (angle, Reynolds number) condition keeps
    a half-precision snapshot of its last converged flow, so the next design at that condition
    starts from it rather than from a different angle's flow.
    """

    def __init__(self, envelope_min, envelope_max, frontal_area, body_length, ref_area, ref_length,
                 fidelity: Fidelity, log=None, cancel=None, planform_area=None, cache=None, ground=None, domain=None):
        """ground: {"z": road height (wind axes, m), "wheels": [(x, y, z, radius, width), ...] in m}
        puts the body on a moving road with spinning wheels (zero angles only)."""
        self.fid = fidelity
        self.ground = ground
        self.cache = cache
        self.cache_hits = 0
        self.log = log or (lambda msg: None)
        self.cancel = cancel or (lambda: False)
        L, W, H = (np.asarray(envelope_max) - np.asarray(envelope_min))
        self.nest = None          # (outer mean field, outer domain, outer cross-section m^2): a zoom grid
        self.domain = domain if domain is not None else plan_domain(
            envelope_min, envelope_max, fidelity.cells, frontal_area, planform_area=planform_area,
            ground_z=None if ground is None else float(ground["z"]))
        dx = self.domain.dx
        self.wheels = ()
        if ground is not None:
            lat = []
            for x, y, z, r, w in ground.get("wheels") or []:
                c = self.domain.to_lattice([x, y, z])
                lat.append((float(c[0]), float(c[1]), float(c[2]), float(r) / dx, 0.5 * float(w) / dx))
            self.wheels = tuple(lat)
        self.L_body = float(body_length)
        self.ref_area = float(ref_area)
        self.ref_length = float(ref_length)
        self.Ln_cells = near_field_scale(L, W, H, frontal_area, planform_area) / dx
        self.flow = None
        self.tunnel = None
        self.n_runs = 0
        self.snapshots = {}
        self.max_snapshots = int(SNAPSHOT_BUDGET_BYTES // (19 * 2 * self.domain.n_cells))
        nx = self.domain.shape[0]
        self.chunk = int(max(80, fidelity.passes_per_chunk * self.Ln_cells / fidelity.u_lat))
        flush = nx / fidelity.u_lat
        self.cold_min_chunks = int(math.ceil(0.8 * flush / self.chunk)) + fidelity.avg_chunks
        self.warm_min_chunks = fidelity.avg_chunks + 2

    def _flow_for(self, re_sim, re_flight=None):
        L_cells = self.L_body / self.domain.dx
        wm = wall_model_on(re_sim, re_flight)
        return FlowSettings(u_lat=self.fid.u_lat, re_sim=re_sim, l_ref_cells=L_cells,
                            smagorinsky=0.0 if re_sim < 500 else 0.14,
                            wall_model=wm, wall_nu=self.fid.u_lat * L_cells / re_flight if wm else 0.0,
                            ground=self.ground is not None, wheels=self.wheels,
                            ramp_steps=0 if self.nest is not None else 600)

    def reset(self):
        """Fresh lattice after a diverged run; the next run starts cold."""
        self.tunnel = None
        self.flow = None
        self.snapshots.clear()
        self.n_runs = 0

    def info(self):
        nx, ny, nz = self.domain.shape
        return {"grid": [nx, ny, nz], "cells": int(nx * ny * nz), "dx_m": self.domain.dx,
                "body_cells": [round(v, 1) for v in self.domain.body_cells], "u_lat": self.fid.u_lat,
                # near-field scale in cells: mean chord for flat airframes, else the body's size across
                "ln_cells": round(self.Ln_cells, 1),
                "fidelity": self.fid.name, "re_cap": self.fid.re_cap, "ground": self.ground is not None,
                "wheels": len(self.wheels), "ground_gap_cells": getattr(self, "ground_gap_cells", None),
                "backend": getattr(self.tunnel, "backend", "cpu") if self.tunnel is not None else None,
                "ref_area": self.ref_area, "ref_length": self.ref_length}

    def run(self, mesh: Mesh, re_sim: float, key=None, want_fields=True, progress=None, forcing=None, groups=None,
            re_flight=None):
        """One CFD run of `mesh` (wind axes, metres) at tunnel Reynolds number `re_sim`.
        forcing: optional (cells, lattice forces) for propeller actuator disks.
        groups: optional part id per face; pressure, friction and compressibility data are then
        also integrated part by part (run["groups"]), e.g. the fuselage alone of a winged body.
        re_flight: the flight Reynolds number this run stands for (turns the wall model on when the
        flight boundary layer is turbulent and the grid runs below it)."""
        flow = self._flow_for(re_sim, re_flight)
        ckey = None
        if self.cache is not None:
            from .cache import case_key
            from .gpu import available
            # float32 GPU and float64 CPU runs agree to ~0.1 %, but never mix them in one comparison
            backend = self.tunnel.backend if self.tunnel is not None else ("gpu" if available() else "cpu")
            ckey = case_key(mesh, re_sim, self.domain, flow, self.fid, forcing, groups, extra=backend)
            hit = self.cache.get(ckey, need_fields=want_fields)
            if hit is not None:
                self.cache_hits += 1
                if progress:
                    progress(1.0, hit.get("cd_tunnel"))
                return {**hit, "cached": True}
        out = self._run(mesh, re_sim, flow, key, want_fields, progress, forcing, groups)
        if ckey is not None:
            self.cache.put(ckey, out)
        return out

    def _run(self, mesh, re_sim, flow, key, want_fields, progress, forcing, groups):
        if self.tunnel is None:
            from .gpu import make_tunnel
            self.tunnel = make_tunnel(self.domain, flow, self.log)
            if self.nest is not None:
                self.tunnel.nest_from(self.nest[0], self.nest[1])
        elif self.flow is None or self.flow != flow:
            self.tunnel.set_flow(flow)
        self.flow = flow
        tun = self.tunnel
        restored = key is not None and key in self.snapshots
        if restored:
            tun.restore(self.snapshots[key])
        tun.set_geometry(mesh)
        tun.set_forcing(*(forcing if forcing is not None else (None, None)))
        if self.ground is not None:
            # cells between the road and the underbody: below about 3 the gap flow is not resolved
            self.ground_gap_cells = round(float((mesh.bbox()[0][2] - float(self.ground["z"])) / self.domain.dx), 2)
        warm = self.n_runs > 0
        self.n_runs += 1
        fid = self.fid
        dx = self.domain.dx
        m = metrics(mesh)
        a_lat = self.ref_area / dx ** 2
        af_lat = m["frontal_area"] / dx ** 2
        min_chunks = self.warm_min_chunks if warm else self.cold_min_chunks
        max_chunks = min_chunks + fid.extra_chunks
        stats = RunStats()
        start_avg_after = max(0, min_chunks - fid.avg_chunks)

        def on_chunk(c, cmax, cd):
            if c == start_avg_after:
                tun.start_averaging(every=4)
            if progress:
                progress(c / max(cmax, 1), cd)
            if self.cancel():
                raise InterruptedError("cancelled")

        if start_avg_after == 0:
            tun.start_averaging(every=4)
        ok, F, cd_std = converge(tun, stats, self.chunk, min_chunks, max_chunks, fid.avg_chunks,
                                 fid.tol, a_lat, progress=on_chunk, avg_from=start_avg_after, se_tol=fid.se_tol)
        if key is not None and (key in self.snapshots or len(self.snapshots) < self.max_snapshots):
            self.snapshots[key] = tun.snapshot()
        u = flow.u_lat
        q_lat = 0.5 * u * u
        cd_raw = F[0] / (q_lat * a_lat)
        cross = (self.nest[2] / dx ** 2) if self.nest is not None else (self.domain.shape[1] - 1) * (self.domain.shape[2] - 1)
        blockage = af_lat / cross
        k = 1.0 + MASKELL_THETA * max(cd_raw * (a_lat / af_lat if af_lat > 0 else 1.0), 0.0) * blockage
        C_total = F[:3] / (q_lat * a_lat) / k
        C_mom = F[3:6] / (q_lat * a_lat * (self.ref_length / dx)) / k

        # surface pressure and edge velocity, sampled just outside the wall
        fields = tun.mean_fields()
        solid = tun.flags == 1
        rho = np.nan_to_num(fields[0], nan=1.0)
        spd = np.sqrt(np.nan_to_num(fields[1]) ** 2 + np.nan_to_num(fields[2]) ** 2 + np.nan_to_num(fields[3]) ** 2)
        tri = self.domain.to_lattice(mesh.triangles().reshape(-1, 3)).reshape(-1, 3, 3)
        cen = tri.mean(axis=1)
        areas, nrm = mesh.face_areas_normals()
        if getattr(mesh, "hidden", None) is not None:    # buried inside another shell: not wetted
            areas = np.where(mesh.hidden, 0.0, areas)
        rho_s = _sample_fluid(rho, solid, cen + 1.5 * nrm)
        spd_s = _sample_fluid(spd, solid, cen + 1.5 * nrm)
        bad = ~np.isfinite(rho_s)
        if bad.any():
            pts2 = cen[bad] + 2.5 * nrm[bad]
            rho_s[bad] = _sample_fluid(rho, solid, pts2)
            spd_s[bad] = _sample_fluid(spd, solid, pts2)
        cp_face = ((rho_s - 1.0) / 3.0) / q_lat / k
        cp_f = np.where(np.isfinite(cp_face), cp_face, 0.0)
        C_press = -(cp_f[:, None] * nrm * (areas / dx ** 2)[:, None]).sum(axis=0) / a_lat
        # skin friction is re-scaled per speed: keep the geometric integral it multiplies. The local
        # dynamic pressure at the boundary-layer edge comes from the surface pressure (Bernoulli,
        # (u_e/U)^2 = 1 - Cp): pressure is nearly constant across the boundary layer, while a speed
        # sampled a cell or two off the wall lies inside it on a coarse grid (that under-counted
        # friction about threefold on a rocket).
        edge = np.clip(np.nan_to_num(spd_s / u, nan=0.0), 0.0, 1.6)
        edge2 = np.clip(1.0 - cp_f, 0.0, 2.56)
        tx = np.sqrt(np.clip(1.0 - nrm[:, 0] ** 2, 0.0, 1.0))
        friction_integral = float((areas * edge2 * tx).sum() / self.ref_area)
        # compact surface-pressure record for compressibility corrections at any flight Mach number
        from .compress import surface_bins
        cen_phys = mesh.triangles().mean(axis=1)
        ref_phys = self.domain.to_physical(tun.ref)
        bins = surface_bins(cp_f, nrm * (areas / self.ref_area)[:, None], (cen_phys - ref_phys) / self.ref_length)
        regime = "direct" if re_sim < self.fid.re_cap * 0.999999 else "hybrid"
        parts = None
        if groups is not None:
            groups = np.asarray(groups)
            parts = {}
            fp = -(cp_f[:, None] * nrm * (areas / self.ref_area)[:, None]) / k
            fi = areas * edge2 * tx / self.ref_area
            for g in np.unique(groups):
                sel = groups == g
                parts[int(g)] = {"cd_pressure": float(fp[sel, 0].sum()), "cl_pressure": float(fp[sel, 2].sum()),
                                 "friction_integral": float(fi[sel].sum()),
                                 "compress": surface_bins(cp_f[sel], (nrm * (areas / self.ref_area)[:, None])[sel],
                                                          ((cen_phys - ref_phys) / self.ref_length)[sel])}
        out = {
            "re_sim": re_sim, "regime": regime, "converged": bool(ok), "steps": stats.steps,
            "seconds": stats.seconds, "mlups": stats.mlups * self.domain.n_cells / 1e6,
            "warm_start": warm, "restored": restored,
            "cd_tunnel": float(C_total[0]), "cd_pressure": float(C_press[0]), "cd_std": cd_std / k,
            "cl": float(C_total[2]), "cl_pressure": float(C_press[2]), "cs": float(C_total[1]),
            "cm_roll": float(C_mom[0]), "cm_pitch": float(C_mom[1]), "cm_yaw": float(C_mom[2]),
            "friction_integral": friction_integral, "ref_area": self.ref_area, "ref_length": self.ref_length,
            "wall_model": bool(flow.wall_model),
            "blockage": blockage, "blockage_factor": k, "geometry": m, "compress": bins,
            "groups": parts, "cached": False,
        }
        if want_fields:
            out["fields"] = self._fields(fields, solid, cp_face, spd_s / u, mesh, stats, a_lat, k)
        return out

    # ---- visual products ---------------------------------------------------------
    def _fields(self, fields, solid, cp_face, edge_face, mesh, stats, a_lat, k):
        dom = self.domain
        u = self.flow.u_lat
        q_lat = 0.5 * u * u
        rho, ux, uy, uz = fields
        lo, hi = mesh.bbox()
        cl = dom.to_lattice((lo + hi) / 2)
        jy = int(round(np.clip(cl[1], 1, dom.shape[1] - 2)))
        kz = int(round(np.clip(cl[2], 1, dom.shape[2] - 2)))
        spd = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2) / u
        cp = ((rho - 1.0) / 3.0) / q_lat / k
        stride = max(1, int(math.ceil(max(dom.shape) / 400)))
        slices = {}
        for name, sl in (("xz", np.s_[::stride, jy, ::stride]), ("xy", np.s_[::stride, ::stride, kz])):
            s_spd = spd[sl]
            s_cp = cp[sl]
            slices[name] = {
                "shape": list(s_spd.shape),
                "speed": _b64(_quantize(s_spd, 0.0, 1.6)), "speed_range": [0.0, 1.6],
                "cp": _b64(_quantize(s_cp, -1.5, 1.0)), "cp_range": [-1.5, 1.0],
            }
        # physical placement of the slice rectangles
        nx, ny, nz = dom.shape
        p0 = dom.to_physical([0, 0, 0])
        p1 = dom.to_physical([nx - 1, ny - 1, nz - 1])
        slices["xz"]["y"] = float(dom.to_physical([0, jy, 0])[1])
        slices["xy"]["z"] = float(dom.to_physical([0, 0, kz])[2])
        slices["bounds"] = [p0.tolist(), p1.tolist()]
        # physical extent of the sampled cell centres: [axis0 min, axis0 max, axis1 min, axis1 max]
        n0, n1 = slices["xz"]["shape"]
        slices["xz"]["extent"] = [float(p0[0]), float(p0[0] + (n0 - 1) * stride * dom.dx),
                                  float(p0[2]), float(p0[2] + (n1 - 1) * stride * dom.dx)]
        n0, n1 = slices["xy"]["shape"]
        slices["xy"]["extent"] = [float(p0[0]), float(p0[0] + (n0 - 1) * stride * dom.dx),
                                  float(p0[1]), float(p0[1] + (n1 - 1) * stride * dom.dx)]
        # streamlines from an upstream rake covering the body silhouette
        lo_l = dom.to_lattice(lo)
        hi_l = dom.to_lattice(hi)
        span_y = hi_l[1] - lo_l[1]
        span_z = hi_l[2] - lo_l[2]
        ny_s = int(np.clip(round(11 * max(span_y / max(span_z, 1e-9), 1.0) ** 0.5), 7, 22))
        nz_s = int(np.clip(round(9 * max(span_z / max(span_y, 1e-9), 1.0) ** 0.5), 5, 22))
        ys = np.linspace(lo_l[1] - 0.12 * span_y - 1, hi_l[1] + 0.12 * span_y + 1, ny_s)
        zs = np.linspace(lo_l[2] - 0.12 * span_z - 1, hi_l[2] + 0.12 * span_z + 1, nz_s)
        x0 = max(2.0, lo_l[0] - 0.5 * (lo_l[0] - 2))
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        seeds = np.stack([np.full(Y.size, x0), Y.ravel(), Z.ravel()], axis=1)
        ux0 = np.nan_to_num(ux); uy0 = np.nan_to_num(uy); uz0 = np.nan_to_num(uz)
        pts, starts = _trace(ux0, uy0, uz0, solid, seeds, 0.5, int(4 * nx), 3)
        phys = dom.to_physical(pts[:, :3]).astype(np.float32)
        spd_line = (pts[:, 3] / u).astype(np.float32)
        hist = np.array(stats.history)
        if len(hist):
            step = max(1, len(hist) // 300)
            h = hist[::step]
            conv = {"step": h[:, 0].astype(int).tolist(),
                    "cd": (h[:, 1] / (q_lat * a_lat) / k).round(5).tolist()}
        else:
            conv = {"step": [], "cd": []}
        # a coarse copy of the whole mean field (VTK export) and its vortex surfaces (3D view)
        from .volume import coarse_volume, vortex_payload
        vol = coarse_volume(rho, ux, uy, uz, solid, dom, u, k)
        try:
            vortex = vortex_payload(vol, self.Ln_cells * dom.dx)
        except Exception:  # noqa: BLE001 - a picture must not fail the run
            vortex = None
        return {
            "surface_cp": _b64(_quantize(cp_face, -2.0, 1.0)), "surface_cp_range": [-2.0, 1.0],
            "surface_speed": _b64(_quantize(edge_face, 0.0, 1.6)),
            "slices": slices,
            "streamlines": {"points": _b64(phys), "speed": _b64(spd_line),
                            "starts": starts.tolist(), "speed_range": [0.0, 1.6]},
            "vortex": vortex,
            "convergence": conv,
            "_volume": vol, "_cp_face": np.asarray(cp_face, np.float16),
        }


def wall_model_on(re_sim, re_flight):
    """The wall model stands in for a turbulent boundary layer the grid cannot carry: on when the
    flight boundary layer is turbulent and the tunnel runs well below the flight Reynolds number."""
    return re_flight is not None and re_flight >= WALL_MODEL_RE and re_flight > 1.5 * re_sim


def analyze(mesh: Mesh, conditions: Conditions, fidelity="draft", log=None, progress=None, wall_model=True,
            ground=None, refine=False):
    """One cold run of a single mesh (wind axes, metres) at one condition. Returns the run,
    the forces at that speed and the setup, flattened (used by validation and tests)."""
    fid = FIDELITY[RETIRED.get(fidelity, fidelity)] if isinstance(fidelity, str) else fidelity
    lo, hi = mesh.bbox()
    af = projected_area(mesh, 0)
    L = float(hi[0] - lo[0])
    air = conditions.air()
    ev = Evaluator(lo, hi, af, L, conditions.ref_area or af, conditions.ref_length or L, fid, log=log,
                   planform_area=projected_area(mesh, 2), ground=ground)
    re_fl = conditions.speed * L / air["nu"]
    re = min(re_fl, fid.re_cap)
    if refine:
        from .zoom import ZoomedEvaluator
        ev = ZoomedEvaluator(ev)
    run = ev.run(mesh, re, progress=progress, re_flight=re_fl if wall_model else None)
    sp = at_speed(run, conditions.speed, air, L)
    out = dict(run)
    out.update(sp)
    out["setup"] = {**ev.info(), "re_flight": sp["re"], "re_tunnel": re, "regime": run["regime"], "air": air}
    return out
