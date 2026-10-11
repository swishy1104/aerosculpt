"""RC tunnel jobs: the general tunnel plus running propellers.

A propeller's loading depends on the advance ratio J = V / nD, so with propellers turning
every (angle, airspeed) pair is its own CFD run (without them the RC tunnel is the general
tunnel). Each run gets the blade-element thrust and torque of every propeller as an actuator
disk, so the airframe drag is measured in the slipstream it really flies in.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..cfd.analysis import at_speed
from ..geometry.zones import Zone
from ..optimize.runner import Runner, live_run, wind_rotation
from .actuator import disk_forcing
from .propeller import INCH, Propeller, bemt

MAX_SLIPSTREAM_LAT = 0.15     # lattice speed allowed in the fastest slipstream


class PropSpec:
    def __init__(self, item, frame):
        self.name = str(item.get("name") or "Propeller")[:40]
        c = np.asarray(item["center"], float) * frame.scale
        self.center = np.asarray(frame.R0) @ c - frame.center          # body frame, m
        self.prop = Propeller(float(item["diameter_in"]) * INCH, float(item["pitch_in"]) * INCH,
                              int(item.get("blades", 2)), float(item.get("width", 1.0)),
                              family=item.get("family", "thin-electric"))
        self.rpm = float(item["rpm"])
        # spinning clockwise seen from behind = right-handed about the forward (-x) axis
        self.spin_forward = 1.0 if item.get("rotation", "cw") == "cw" else -1.0
        self.count = 1

    @property
    def radius(self):
        return 0.5 * self.prop.diameter


class RCRunner(Runner):
    def setup(self):
        super().setup()
        self.props = [PropSpec(p, self.frame) for p in self.spec.get("propellers", [])]
        if not self.props:
            return
        # implicit protected region around each hub so a reshaped nose never runs into the prop
        for p in self.props:
            h = np.array([0.2 * p.prop.diameter, 0.18 * p.prop.diameter, 0.18 * p.prop.diameter])
            self.zones.append(Zone("protect", "cylinder", p.center.copy(), h, f"{p.name} hub"))
        # slow the lattice down if a heavily loaded propeller would push its slipstream past
        # the solver's low-Mach range (the lowest airspeed is the worst case)
        v0 = min(self.speeds)
        worst = 1.0
        for p in self.props:
            perf = bemt(p.prop, v0, p.rpm, self.air["rho"], self.air["mu"], self.air["a"])
            ctd = max(perf["thrust_N"], 0.0) / (0.5 * self.air["rho"] * v0 * v0 * math.pi * p.radius ** 2)
            worst = max(worst, math.sqrt(1.0 + ctd))
        u = min(self.fid.u_lat, MAX_SLIPSTREAM_LAT / worst)
        self.u_cap = u
        if u < self.fid.u_lat:
            self.log(f"slipstream {worst:.2f}x the airspeed at {v0:g} m/s: lattice speed lowered to {u:.3f}")
            self.fid = dataclasses.replace(self.fid, u_lat=u)

    def make_evaluator(self, lo, hi, fid):
        # every tunnel used with these propellers (including a higher-fidelity re-check) keeps the slipstream cap
        if getattr(self, "u_cap", None) and fid.u_lat > self.u_cap:
            fid = dataclasses.replace(fid, u_lat=self.u_cap)
        return super().make_evaluator(lo, hi, fid)

    def simulated_angles(self, body, can_mirror=True):
        if self.props:      # swirl breaks top-to-bottom symmetry
            self.symmetric = False
            return list(self.aoas)
        return super().simulated_angles(body, can_mirror)

    def runs_per_design(self):
        return len(self.sim_angles) * (len(self.speeds) if self.props else len(self.re_groups))

    def envelope_box(self, bodies, deformer=None):
        lo, hi, af = super().envelope_box(bodies, deformer)
        for a in self.sim_angles:
            R = wind_rotation(a, self.yaw)
            for p in self.props:
                c = R @ p.center
                ax = R @ np.array([1.0, 0.0, 0.0])
                rim = np.array([c + p.radius * (np.cos(t) * e1 + np.sin(t) * e2)
                                for e1, e2 in [self._perp(ax)] for t in np.linspace(0, 2 * np.pi, 24)])
                lo = np.minimum(lo, rim.min(0) - 0.05 * p.radius)
                hi = np.maximum(hi, rim.max(0) + 0.05 * p.radius)
        return lo, hi, af

    @staticmethod
    def _perp(ax):
        a = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, ax)
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(ax, e1)

    def prop_loads(self, ev, aoa, v):
        """Actuator-disk forcing for every propeller at one condition, and their performance."""
        R = wind_rotation(aoa, self.yaw)
        air_dir = R @ np.array([1.0, 0.0, 0.0])           # propellers push air aft (+x body)
        v_ax = v * max(air_dir[0], 0.05)
        idx_all, F_all, perfs = [], [], []
        for p in self.props:
            perf = bemt(p.prop, v_ax, p.rpm, self.air["rho"], self.air["mu"], self.air["a"])
            idx, F = disk_forcing(ev.domain, R @ p.center, air_dir, p.radius, p.prop.hub * p.radius,
                                  perf["r"], perf["dT_dr"], perf["dQ_dr"], -p.spin_forward,
                                  max(perf["thrust_N"], 0.0), max(perf["torque_Nm"], 0.0),
                                  self.air["rho"], v, ev.fid.u_lat)
            idx_all.append(idx)
            F_all.append(F)
            perfs.append({k: perf[k] for k in ("thrust_N", "torque_Nm", "power_W", "eta", "J", "CT", "CP",
                                               "rpm", "tip_mach")} | {"name": p.name})
        return (np.concatenate(idx_all), np.concatenate(F_all)), perfs

    def run_for(self, aoa, i_speed):
        if not self.props:
            return super().run_for(aoa, i_speed)
        return self.last_runs[(aoa, self.speeds[i_speed])]

    def evaluate_design(self, ev, body, label, angles, aoas=None, groups=None):
        if not self.props:
            return super().evaluate_design(ev, body, label, angles, aoas, groups)
        aoas = list(self.aoas if aoas is None else aoas)
        v_fields = self.speeds[len(self.speeds) // 2]
        runs, perfs = {}, {}
        for a in angles:
            wind = body.transformed(R=wind_rotation(a, self.yaw))
            for v, re in zip(self.speeds, self.re_by_speed):
                self.check_cancel()
                frac0 = self.run_index / self.total_runs
                frac1 = (self.run_index + 1) / self.total_runs

                trace = []

                def prog(f, cd, frac0=frac0, frac1=frac1, a=a, v=v, trace=trace):
                    self._progress(0.02 + 0.96 * (frac0 + (frac1 - frac0) * min(f, 1.0)),
                                   f"{label} · {a:g}° · {v:g} m/s · CFD run {self.run_index + 1} of {self.total_runs}",
                                   live_run(self, label, a, v, f, cd, trace))
                forcing, perf = self.prop_loads(ev, a, v)
                try:
                    runs[(a, v)] = ev.run(wind, re, key=(a, v), want_fields=(v == v_fields) and self.fields_wanted, progress=prog,
                                          forcing=forcing, groups=groups)
                except InterruptedError:
                    from ..optimize.runner import JobCancelled
                    raise JobCancelled()
                perfs[(a, v)] = perf
                if getattr(self, "mach_max", 0.0) > 0.6:
                    self.transonic_anchor(wind, [runs[(a, v)]])
                self.run_index += 1
                self.count_run(ev, runs[(a, v)])
        self.last_runs = runs
        polar = []
        for aoa in aoas:
            speeds = []
            for v in self.speeds:
                run = runs[(aoa, v)]
                s = at_speed(run, v, self.air, self.L_body)
                pp = perfs[(aoa, v)]
                thrust = sum(p["thrust_N"] for p in pp)
                power = sum(p["power_W"] for p in pp)
                s.update({"cl": run["cl"], "cm_pitch": run["cm_pitch"], "thrust_N": thrust, "shaft_power_W": power,
                          "prop_eta": thrust * v / power if power > 1e-9 else 0.0, "net_N": thrust - s["drag_N"],
                          "props": pp})
                speeds.append(s)
            row = runs[(aoa, v_fields)]
            polar.append({"aoa": aoa, "mirror": False, "regime": row["regime"], "cl": row["cl"], "cs": row["cs"],
                          "cm_pitch": row["cm_pitch"], "cm_roll": row["cm_roll"], "cm_yaw": row["cm_yaw"],
                          "cd_pressure": row["cd_pressure"], "converged": all(runs[(aoa, v)]["converged"] for v in self.speeds),
                          "cd_std": row["cd_std"], "blockage": row["blockage"],
                          "seconds": sum(runs[(aoa, v)]["seconds"] for v in self.speeds), "speeds": speeds})
        fields = {a: runs[(a, v_fields)].get("fields") for a in angles}
        return polar, fields

    def common(self, ev):
        out = super().common(ev)
        out["setup"]["runs_per_design"] = self.runs_per_design()
        if self.props:
            out["setup"]["fields_speed"] = self.speeds[len(self.speeds) // 2]
            out["propellers"] = [{"name": p.name, "center_body": p.center.tolist(), "diameter_m": p.prop.diameter,
                                  "pitch_m": p.prop.pitch, "blades": p.prop.blades, "rpm": p.rpm, "family": p.prop.family,
                                  "rotation": "cw" if p.spin_forward > 0 else "ccw"} for p in self.props]
        return out

