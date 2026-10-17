"""Rocket optimization on the flight model: apogee (or the drag build-up) over the template's
free dimensions, every design checked for static margin, fin flutter and rail-exit speed before
it is scored. Deterministic and fast (no CFD, no error probes): the same job contract as the
CFD Runner, so the workspace, reports and exports work unchanged.

The result carries CFD-format polars over the project's envelope (drag from the build-up, lift
from Barrowman's normal-force slope) and a "rocket" block with both designs' full evaluation:
stability through the flight, drag curve, flutter, the flight itself and the checks.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..cfd.analysis import isa
from ..geometry.mesh import projected_area
from ..geometry.zones import from_spec
from ..optimize.runner import ModelFrame, Runner, load_model, polar_summary
from ..optimize.spaces import TemplateSpace
from ..optimize.strategy import Search
from .model import RocketModel
from .motors import BY_ID, Motor

OBJECTIVES = {"apogee": "Maximize apogee (flight model)", "rocket_drag": "Minimize drag (rocket drag model)"}


def motor_from_spec(rk):
    """The motor a job spec carries: the full curve ("motor_data") or a built-in id."""
    md = (rk or {}).get("motor_data")
    if md and md.get("curve"):
        return Motor(md["id"], md["name"], md.get("manufacturer", ""), md.get("diameter_mm", 18), md.get("length_mm", 70),
                     md.get("delays") or [], md.get("prop_mass_g", 0) / 1000.0, md.get("total_mass_g", 0) / 1000.0,
                     md["curve"], md.get("source", "file"), md.get("note", ""))
    mid = ((rk or {}).get("motor") or {}).get("id")
    return BY_ID.get(mid)


def rocket_polar(model: RocketModel, aoas, speeds, air, ref_area, ref_length):
    """A polar in the CFD result format from the drag build-up and Barrowman's normal force."""
    polar = []
    for a in aoas:
        ar = math.radians(a)
        rows = []
        for v in speeds:
            mach = v / air["a"]
            re = air["rho"] * v * model.g.length / air["mu"]
            cp = model.barrowman.cp(max(mach, 0.05))
            b = model.drag.breakdown(mach, re, ar, cp["cn_alpha"])
            q = 0.5 * air["rho"] * v * v
            cn = cp["cn_alpha"] * ar
            cl = cn * math.cos(ar) - b["cd"] * math.sin(ar)
            cd = b["cd"] * math.cos(ar) + cn * math.sin(ar)
            S = model.g.ref_area
            drag = cd * q * S
            lift = cl * q * S
            rows.append({"v": v, "re": re, "mach": mach, "cd": cd * S / ref_area, "cd_friction": b["cd_friction"] * S / ref_area,
                         "cd_pressure": (b["cd_pressure"] + b["cd_base"]) * S / ref_area, "cf": b["cf"], "q": q,
                         "drag_N": drag, "lift_N": lift, "side_N": 0.0, "cl": cl * S / ref_area,
                         "cm_pitch": -cn * (cp["x_cp"] - 0.5 * model.g.length) / ref_length * S / ref_area,
                         "ld": lift / drag if drag > 0 and abs(cl) > 1e-6 else None, "power_W": drag * v,
                         "compressibility": "build-up", "mach_crit": None, "cd_base": b["cd_base"] * S / ref_area})
        mid = rows[len(rows) // 2]
        polar.append({"aoa": a, "mirror": False, "regime": "rocket model", "mach_crit": None, "cl": mid["cl"], "cs": 0.0,
                      "cm_pitch": mid["cm_pitch"], "cm_roll": 0.0, "cm_yaw": 0.0, "cd_pressure": mid["cd_pressure"], "converged": True,
                      "cd_std": 0.0, "blockage": 0.0, "seconds": 0.0, "speeds": rows})
    return polar


class RocketRunner(Runner):
    def setup(self):
        spec = self.spec
        m = spec.get("model", {})
        raw = load_model(self.model_path, self.filename)
        self.frame = ModelFrame(raw, m.get("units", "m"), m.get("forward", "-x"), m.get("up", "+z"), m.get("scale"))
        env = spec.get("envelope", {})
        self.speeds = sorted(float(v) for v in env.get("speeds", [50.0]))
        self.aoas = sorted(float(a) for a in env.get("aoas", [0.0]))
        self.yaw = float(env.get("yaw", 0.0))
        self.air = isa(float(env.get("altitude", 0.0)))
        self.zones = from_spec([z for z in spec.get("zones", []) if z.get("type") in ("protect", "keep", "design")],
                               self.frame.scale, self.frame.R0, self.frame.center)
        body = self.frame.body
        lo, hi = body.bbox()
        self.L_body = float(hi[0] - lo[0])
        self.ref_area = float(env.get("ref_area") or projected_area(body, 0))
        self.ref_length = float(env.get("ref_length") or self.L_body)
        self.features, self.base_body = [], body
        self.tmpl = self.spec.get("template") or {}
        if self.tmpl.get("id") != "rocket":
            raise ValueError("the flight model needs a rocket template project")
        self.rk = dict(self.spec.get("rocket") or {})
        self.motor = motor_from_spec(self.rk)
        self.evaluations = 0

    def model(self, values):
        return RocketModel(values, self.rk, self.motor)

    def common_light(self):
        return {"tunnel": self.spec.get("tunnel", "general"),
                "setup": {"method": "rocket model", "air": self.air, "regimes": ["rocket model"], "symmetric": False,
                          "simulated_angles": list(self.aoas), "runs_per_design": 0, "fidelity": "rocket model",
                          "ref_area": self.ref_area, "ref_length": self.ref_length, "re_groups": []},
                "envelope": {"speeds": self.speeds, "aoas": self.aoas, "yaw": self.yaw,
                             "altitude": self.spec.get("envelope", {}).get("altitude", 0.0)},
                "model": self.spec.get("model", {})}

    def run(self):
        self.setup()
        if self.spec.get("kind") != "optimize":
            raise ValueError("the flight model runs as an optimization; the flight itself is computed live in the workspace")
        return self.run_optimize()

    def run_optimize(self):
        from ..optimize.driver import constraints_from, _jsonable
        o = self.spec.get("optimize", {})
        obj = o.get("objective", "apogee")
        if obj not in OBJECTIVES:
            raise ValueError(f"objective {obj!r} is not a flight-model goal")
        space = TemplateSpace("rocket", self.tmpl.get("values", {}), o.get("params"), self.frame.center, constraints_from(o), self.zones)
        if not space.vars:
            raise ValueError("every dimension is locked: unlock at least one for the optimizer to change")
        own_fast = space.fast_check

        def fast_check(cand):
            why = own_fast(cand)
            if why:
                return why
            return self.model(cand.values).guard(fast=True)
        space.fast_check = fast_check
        base_q = {}

        def evaluate(cand, label):
            self.check_cancel()
            m = self.model(cand.values)
            q = m.quick()
            self.evaluations += 1
            cand._quick = q
            b = base_q.setdefault("q", q)
            if obj == "apogee":
                J = b["apogee_m"] / max(q["apogee_m"], 1e-6)
            else:
                J = q["cd_envelope"] / max(b["cd_envelope"], 1e-9)
            return {"J": J, "drag_ratio": q["cd_envelope"] / max(b["cd_envelope"], 1e-9), "apogee_m": round(q["apogee_m"], 2),
                    "margin": round(q["margin_liftoff"], 3), "max_speed": round(q["max_speed"], 2), "cd_ref": q["cd_envelope"],
                    "dry_mass_kg": round(q["dry_mass"], 4), "converged": True}
        budget = int(o.get("iterations", 30))
        top_k = int(o.get("top_k", 4))
        self.log(f"optimizing {len(space.vars)} dimension(s) on the flight model: {OBJECTIVES[obj]}; motor {self.motor.manufacturer} {self.motor.name}")
        search = Search(space, evaluate, budget=budget, top_k=top_k, n_confirm=0, eps_floor=0.002, noise=None, confirm=None,
                        log=self.log, progress=self._progress, cancel=self.check_cancel, seed=int(o.get("seed", 7)),
                        patience=max(6, top_k + 2))
        search.relevance = None
        search.run()
        base_c, best_c = space.base, search.best["cand"]
        self._progress(0.95, "Best design · flight and stability")
        mb, mo = self.model(base_c.values), self.model(best_c.values)
        eb, eo = mb.evaluate(), mo.evaluate()
        pb = rocket_polar(mb, self.aoas, self.speeds, self.air, self.ref_area, self.ref_length)
        po = rocket_polar(mo, self.aoas, self.speeds, self.air, self.ref_area, self.ref_length)
        (self.out / "baseline_body.stl").write_bytes(base_c.mesh.to_stl_bytes())
        (self.out / "best_body.stl").write_bytes(best_c.mesh.to_stl_bytes())
        (self.out / "optimized.stl").write_bytes(self.frame.to_file_frame(best_c.mesh).to_stl_bytes(b"AeroSculpt optimized design"))
        J = search.best["J"] if search.reliable else 1.0
        bv = base_c.info.get("volume")
        result = {
            "kind": "optimize", **self.common_light(), "mode": "parameters", "objective": obj, "objective_label": OBJECTIVES[obj],
            "objective_indicative": False, "auto": False, "variables": [v.public() for v in space.vars], "aspects": [],
            "best_values": {k: _jsonable(v) for k, v in best_c.values.items()},
            "improvement": 1.0 - J, "rounds": [{"round": 1, "change": J - 1.0, "reliable": search.reliable, "adopted": [], "stop": search.stop_reason}],
            "zones": [{"name": z.name, "type": z.type} for z in self.zones], "keep_zones": [],
            "history": search.history, "relevance": search.relevance,
            "strategy": {"noise": search.noise_info, "ranking": search.ranking, "selected": search.selected, "adopted": search.adopted,
                         "adopted_all": getattr(search, "adopted_all", []), "stop_reason": search.stop_reason, "confirmed": None,
                         "reliable": search.reliable, "epsilon": search.final_eps, "budget": budget, "top_k": top_k,
                         "designs_explored": search.evals, "rejected": search.rejected, "explore": "rocket flight model",
                         "confirm": None, "confirm_grid": None, "cache_hits": 0, "explore_cells": None, "confirm_cells": None},
            "baseline": {"polar": pb, "summary": polar_summary(pb)}, "best": {"polar": po, "summary": polar_summary(po)},
            "drag_change": float(np.mean([s["drag_N"] for p in po for s in p["speeds"]]) / max(np.mean([s["drag_N"] for p in pb for s in p["speeds"]]), 1e-9)) - 1.0,
            "apogee_change": eo["flight"]["apogee_m"] / max(eb["flight"]["apogee_m"], 1e-6) - 1.0,
            "volume_ratio": (best_c.info.get("volume") / bv) if bv else 1.0,
            "fields": None, "verified": None, "meshes": {"baseline": "baseline_body.stl", "best": "best_body.stl"},
            "template": {"id": "rocket", "base_values": {k: _jsonable(v) for k, v in base_c.values.items()},
                         "best_values": {k: _jsonable(v) for k, v in best_c.values.items()}, "info_base": base_c.info, "info_best": best_c.info},
            "rocket": {"base": eb, "best": eo, "objective": obj},
            "evaluations": self.evaluations, "cfd_runs": 0, "elapsed_s": time.time() - self.t0,
            "best_index": next((h["i"] for h in search.history if h.get("best")), 0),
        }
        self.history = search.history
        self.write_json("result.json", result)
        self.write_json("history.json", search.history)
        self._progress(1.0, "Done", {"history": search.history[-60:]})
        return result

    def credits(self):
        return 1 + max(0, self.evaluations - 1) // 10
