"""Supersonic / hypersonic tunnel jobs: envelope sweeps and shape optimization on the panel
methods, with the material temperature limits as a constraint."""
from __future__ import annotations

import time

import numpy as np

from ..cfd.analysis import _b64, _quantize
from ..geometry.ffd import ASPECTS, Deformer, design_space, refine
from ..geometry.mesh import projected_area
from ..geometry.zones import from_spec
from ..optimize.runner import ModelFrame, Runner, load_model
from . import materials as mat
from .atmosphere import us76
from .panel import Surface

OBJECTIVES = {
    "drag": "Minimize drag across the envelope",
    "peak_temp": "Minimize peak wall temperature",
}


class HyperRunner(Runner):
    """Same job contract as the CFD Runner (spec in, result.json + files out)."""

    def setup(self):
        spec = self.spec
        m = spec.get("model", {})
        raw = load_model(self.model_path, self.filename)
        self.frame = ModelFrame(raw, m.get("units", "m"), m.get("forward", "-x"), m.get("up", "+z"))
        env = spec.get("envelope", {})
        self.machs = sorted(float(v) for v in env.get("machs", [2.0]))
        self.aoas = sorted(float(a) for a in env.get("aoas", [0.0]))
        self.yaw = float(env.get("yaw", 0.0))
        self.altitude = float(env.get("altitude", 20000.0))
        self.air = us76(self.altitude)
        if min(self.machs) < 1.2:
            raise ValueError("the supersonic tunnel covers Mach 1.2 and above; use the General tunnel up to Mach 1.2")
        th = spec.get("thermal", {})
        self.thermal = {
            "transition": th.get("transition", "auto") if th.get("transition") in ("auto", "laminar", "turbulent") else "auto",
            "re_transition": float(th.get("re_transition", 1.0e6)),
            "wall": "fixed" if th.get("wall") == "fixed" else "radiative",
            "t_wall_k": float(th.get("t_wall_c", 25.0)) + 273.15,
            "nose_radius": (float(th["nose_radius_mm"]) / 1000.0) if th.get("nose_radius_mm") else None,
        }
        self.material = mat.resolve(th.get("material", mat.DEFAULT))
        items = spec.get("zones", [])
        self.zones = from_spec([z for z in items if z.get("type") in ("protect", "keep", "design")],
                               self.frame.scale, self.frame.R0, self.frame.center)
        mzones = [z for z in items if z.get("type") == "material"]
        self.material_zones = list(zip(from_spec(mzones, self.frame.scale, self.frame.R0, self.frame.center),
                                       [mat.resolve(z.get("material", mat.DEFAULT)) for z in mzones]))
        self.materials = [self.material] + [m_ for _, m_ in self.material_zones]
        self.apply_features()
        body = self.frame.body
        lo, hi = body.bbox()
        self.L_body = float(hi[0] - lo[0])
        self.x_nose = float(lo[0])
        self.ref_area = float(env.get("ref_area") or projected_area(body, 0))
        self.ref_length = float(env.get("ref_length") or self.L_body)
        self.symmetric = False
        self.sim_angles = list(self.aoas)
        self.speeds = [m_ * self.air["a"] for m_ in self.machs]
        self.re_groups = [0]
        self.fid = None
        self.evaluations = 0

    # ---- one design ---------------------------------------------------------------------
    def face_materials(self, body):
        c = body.triangles().mean(axis=1)
        idx = np.zeros(len(c), dtype=np.uint8)
        for k, (z, _m) in enumerate(self.material_zones, start=1):
            idx[z.distance(c) <= 0.0] = k
        t_max = np.array([m_["t_max_c"] + 273.15 for m_ in self.materials])[idx]
        eps = np.array([m_["emissivity"] for m_ in self.materials])[idx]
        return idx, t_max, eps

    def evaluate_design(self, body, label, want_faces=False):
        idx, t_max, eps = self.face_materials(body)
        surf = Surface(body, self.ref_area, self.ref_length, np.zeros(3), t_max, eps, idx)
        polar, faces = [], {}
        for a in self.aoas:
            self.check_cancel()
            speeds = []
            for M in self.machs:
                r = surf.evaluate(M, a, self.yaw, self.altitude, self.thermal["transition"],
                                  self.thermal["re_transition"], self.thermal["wall"], self.thermal["t_wall_k"],
                                  self.thermal["nose_radius"], want_faces=want_faces)
                self.evaluations += 1
                if want_faces:
                    faces[(a, M)] = r.pop("faces")
                r.pop("air", None)
                r["power_W"] = r["drag_N"] * r["v"]
                speeds.append(r)
            mid = speeds[len(speeds) // 2]
            polar.append({"aoa": a, "mirror": False, "regime": "panel", "cl": mid["cl"], "cs": mid["cs"],
                          "cm_pitch": mid["cm_pitch"], "cm_roll": 0.0, "cm_yaw": 0.0, "cd_pressure": mid["cd_wave"],
                          "converged": True, "cd_std": 0.0, "blockage": 0.0, "seconds": 0.0, "speeds": speeds})
        return polar, faces, idx

    def save_faces(self, tag, faces, mat_idx):
        """Surface fields per angle: Cp, wall temperature and heat flux for every Mach number."""
        tw_hi = max(float(f["tw"].max()) for f in faces.values())
        tw_lo = min(float(self.air["T"]), min(float(f["tw"].min()) for f in faces.values()))
        q_hi = max(float(f["q"].max()) for f in faces.values()) or 1.0
        out = []
        for k, a in enumerate(self.aoas):
            name = f"fields_{tag}_{k}.json"
            doc = {"tunnel": "supersonic", "aoa": a, "machs": self.machs,
                   "ranges": {"cp": [-0.3, 2.0], "tw": [tw_lo, tw_hi], "q": [0.0, q_hi]},
                   "material_index": _b64(mat_idx.astype(np.uint8)),
                   "surface": [{"cp": _b64(_quantize(faces[(a, M)]["cp"], -0.3, 2.0)),
                                "tw": _b64(_quantize(faces[(a, M)]["tw"], tw_lo, tw_hi)),
                                "q": _b64(_quantize(faces[(a, M)]["q"], 0.0, q_hi)),
                                "method": _b64(faces[(a, M)]["method"].astype(np.uint8))} for M in self.machs]}
            self.write_json(name, doc)
            out.append({"aoa": a, "file": name, "mirror": False})
        return out

    def common(self, ev=None):
        return {"tunnel": "supersonic",
                "setup": {"method": "panel", "altitude": self.altitude, "air": self.air, "machs": self.machs,
                          "ref_area": self.ref_area, "ref_length": self.ref_length, "symmetric": False,
                          "regimes": ["panel"], "runs_per_design": len(self.aoas) * len(self.machs),
                          "thermal": {k: v for k, v in self.thermal.items()},
                          "materials": self.materials, "body_length": self.L_body, "x_nose": self.x_nose},
                "envelope": {"speeds": self.speeds, "machs": self.machs, "aoas": self.aoas, "yaw": self.yaw,
                             "altitude": self.altitude},
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
        self._progress(0.05, "Supersonic tunnel · surface methods")
        polar, faces, idx = self.evaluate_design(body, "Envelope", want_faces=True)
        saved = self.save_faces("baseline", faces, idx)
        (self.out / "baseline_body.stl").write_bytes(body.to_stl_bytes())
        result = {"kind": "analyze", **self.common(), "baseline": {"polar": polar, "summary": hyper_summary(polar)},
                  "fields": {"baseline": saved}, "meshes": {"baseline": "baseline_body.stl"},
                  "evaluations": self.evaluations, "cfd_runs": 0, "elapsed_s": time.time() - self.t0}
        self.write_json("result.json", result)
        self._progress(1.0, "Done")
        return result

    def run_optimize(self):
        """Screening and Bayesian search over template parameters or the body deformation, on the
        surface methods (deterministic: no error probe or confirmation level needed)."""
        from ..optimize.driver import constraints_from
        from ..optimize.spaces import DeformSpace, FeatureSpace, FixedSpace, TemplateSpace
        from ..optimize.strategy import Search
        o = self.spec.get("optimize", {})
        obj_kind = o.get("objective", "drag")
        if obj_kind not in OBJECTIVES:
            raise ValueError(f"objective {obj_kind!r} is not available in the supersonic tunnel")
        respect = bool(o.get("respect_limits", True))
        tmpl = self.spec.get("template")
        constraints = constraints_from(o)
        if tmpl:
            space = TemplateSpace(tmpl["id"], tmpl.get("values", {}), o.get("params"), self.frame.center, constraints,
                                  self.zones)
            aspects, mode = [], "parameters"
        else:
            aspects = [a for a in o.get("aspects", ["nose", "tail"]) if a in ASPECTS]
            if not aspects and not self.features:
                raise ValueError("choose at least one design aspect to modify")
            body = self.base_body
            if aspects:
                if "fairing" in aspects:
                    lo, hi = body.bbox()
                    body = refine(body, max_edge=float((hi - lo).max()) / 45.0)
                    if not self.features:
                        self.frame.body = body
                space = DeformSpace(Deformer(body, design_space(aspects, float(o.get("max_change", 0.25))), self.zones),
                                    constraints)
                mode = "aspects"
            else:
                space, mode = FixedSpace(body), "features"
        if self.features:
            space = FeatureSpace(space, self.features, self.feature_frame(), constraints)
            if not space.vars:
                raise ValueError("nothing to optimize: unlock a shape-feature dimension or choose a design aspect")
        base_ref = {}

        def evaluate(cand, label):
            polar, _, _ = self.evaluate_design(cand.mesh, label)
            base = base_ref.setdefault("polar", polar)
            J = hyper_objective(obj_kind, polar, base, respect)
            s = hyper_summary(polar)
            return {"J": J, "drag_ratio": drag_ratio(polar, base), "tw_max": s["tw_max"], "margin_K": s["margin_K"],
                    "cd_ref": s["cd_at_min_aoa"], "converged": True}

        budget = int(o.get("iterations", 30))
        top_k = int(o.get("top_k", 4))
        search = Search(space, evaluate, budget=budget, top_k=top_k, n_confirm=0, eps_floor=0.002,
                        noise=None, confirm=None, log=self.log, progress=self._progress, cancel=self.check_cancel,
                        seed=int(o.get("seed", 7)), patience=max(6, top_k + 2))
        search.relevance = None
        search.run()
        base_c, best_c = space.base, search.best["cand"]
        self._progress(0.95, "Best design · surface fields")
        base_polar, base_faces, base_idx = self.evaluate_design(base_c.mesh, "Baseline", want_faces=True)
        best_polar, best_faces, best_idx = self.evaluate_design(best_c.mesh, "Best", want_faces=True)
        base_saved = self.save_faces("baseline", base_faces, base_idx)
        best_saved = self.save_faces("best", best_faces, best_idx)
        (self.out / "baseline_body.stl").write_bytes(base_c.mesh.to_stl_bytes())
        (self.out / "best_body.stl").write_bytes(best_c.mesh.to_stl_bytes())
        (self.out / "optimized.stl").write_bytes(
            self.frame.to_file_frame(best_c.mesh).to_stl_bytes(b"AeroSculpt optimized design"))
        J0 = hyper_objective(obj_kind, base_polar, base_polar, respect)
        J1 = hyper_objective(obj_kind, best_polar, base_polar, respect)
        bs, os_ = hyper_summary(base_polar), hyper_summary(best_polar)
        bv = base_c.info.get("volume")
        if self.features and mode != "aspects":
            vr = space.FT.union_volume(best_c.mesh) / space.base_union if space.base_union else 1.0
        else:
            vr = space.dfm.volume_ratio(best_c.mesh) if mode == "aspects" else (best_c.info.get("volume") / bv if bv else 1.0)
        num = lambda v: float(v) if isinstance(v, (int, float, np.floating)) else v
        result = {
            "kind": "optimize", **self.common(), "mode": mode, "objective": obj_kind, "objective_label": OBJECTIVES[obj_kind],
            "respect_limits": respect, "variables": [v.public() for v in space.vars], "aspects": aspects,
            "best_index": next((h["i"] for h in search.history if h.get("best")), 0),
            "best_values": {k: num(v) for k, v in best_c.values.items()},
            "baseline": {"polar": base_polar, "summary": bs}, "best": {"polar": best_polar, "summary": os_},
            "improvement": (J0 - J1) / abs(J0) if J0 else 0.0,
            "drag_change": drag_ratio(best_polar, base_polar) - 1.0, "tw_change_K": os_["tw_max"] - bs["tw_max"],
            "volume_ratio": vr, "zones": [{"name": z.name, "type": z.type} for z in self.zones],
            "keep_zones": space.dfm.keep.report() if mode == "aspects" and space.dfm.keep is not None else [],
            "history": search.history, "relevance": search.relevance, "verified": None,
            "strategy": {"noise": search.noise_info, "ranking": search.ranking, "selected": search.selected,
                         "adopted": search.adopted, "stop_reason": search.stop_reason, "confirmed": None,
                         "reliable": search.reliable, "epsilon": search.final_eps, "budget": budget, "top_k": top_k,
                         "designs_explored": search.evals, "rejected": search.rejected,
                         "explore": "surface methods", "confirm": None, "cache_hits": 0},
            "fields": {"baseline": base_saved, "best": best_saved},
            "meshes": {"baseline": "baseline_body.stl", "best": "best_body.stl"},
            "evaluations": self.evaluations, "cfd_runs": 0, "elapsed_s": time.time() - self.t0,
        }
        if tmpl:
            result["template"] = {"id": tmpl["id"], "base_values": {k: num(v) for k, v in base_c.values.items()},
                                  "best_values": result["best_values"], "info_base": base_c.info, "info_best": best_c.info}
        if self.features:
            result["features"] = {"changes": [{**c, "base": num(c["base"]), "best": num(c["best"])} for c in space.changes(best_c)],
                                  "list": [{"id": f["id"], "name": f["name"], "type": f["type"]} for f in space.features]}
        self.write_json("result.json", result)
        self.write_json("history.json", search.history)
        self._progress(1.0, "Done", {"history": search.history[-60:]})
        return result

    def credits(self):
        return panel_credits(self.evaluations // max(1, len(self.aoas) * len(self.machs)))


def panel_credits(designs):
    """Surface-method evaluations are cheap: one credit per ten designs, at least one."""
    return 1 + max(0, designs - 1) // 10


def drag_ratio(polar, base_polar):
    pairs = [(s, s0) for p, p0 in zip(polar, base_polar) for s, s0 in zip(p["speeds"], p0["speeds"])]
    return float(np.mean([s["drag_N"] / s0["drag_N"] for s, s0 in pairs if s0["drag_N"] > 0]))


def hyper_objective(kind, polar, base_polar, respect_limits=True):
    rows = [s for p in polar for s in p["speeds"]]
    dr = drag_ratio(polar, base_polar)
    if kind == "peak_temp":
        t0 = max(s["tw_max"] for p in base_polar for s in p["speeds"])
        J = max(s["tw_max"] for s in rows) / t0 * (1.0 + 2.0 * max(0.0, dr - 1.05))
    else:
        J = dr
    if respect_limits:
        excess = max(0.0, -min(s["margin_K"] for s in rows))
        over = max(s["over_limit_area"] for s in rows)
        J *= 1.0 + excess / 100.0 + 5.0 * over
    return float(J)


def hyper_summary(polar):
    rows = [(p["aoa"], s) for p in polar for s in p["speeds"]]
    aoas = np.array([p["aoa"] for p in polar])
    i0 = int(np.argmin(np.abs(aoas)))
    mid = len(polar[i0]["speeds"]) // 2
    ref = polar[i0]["speeds"][mid]
    best = max((r for r in rows if r[1]["ld"] is not None and r[1]["cl"] > 0), key=lambda r: r[1]["ld"], default=None)
    hot = max(rows, key=lambda r: r[1]["tw_max"])
    out = {"cd_at_min_aoa": ref["cd"], "aoa_ref": float(aoas[i0]), "speed_ref": ref["v"], "mach_ref": ref["mach"],
           "drag_mean_N": float(np.mean([s["drag_N"] for _, s in rows])),
           "drag_max_N": float(np.max([s["drag_N"] for _, s in rows])),
           "ld_max": best[1]["ld"] if best else None,
           "ld_max_at": {"aoa": best[0], "v": best[1]["v"], "mach": best[1]["mach"]} if best else None,
           "tw_max": hot[1]["tw_max"], "tw_max_at": {"aoa": hot[0], "mach": hot[1]["mach"]},
           "margin_K": float(min(s["margin_K"] for _, s in rows)),
           "over_limit": any(s["over_limit_area"] > 0 for _, s in rows),
           "q_max": float(max(s["q_max"] for _, s in rows)),
           "heat_load_max_W": float(max(s["heat_load_W"] for _, s in rows)),
           "nose_radius": ref["nose_radius"],
           "cd_vs_mach": [{"mach": s["mach"], "cd": s["cd"], "cd_wave": s["cd_wave"], "cd_base": s["cd_base"],
                           "cd_friction": s["cd_friction"]} for s in polar[i0]["speeds"]]}
    if len(polar) >= 2:
        a = np.radians(aoas)
        out["cl_alpha"] = float(np.polyfit(a, [p["speeds"][mid]["cl"] for p in polar], 1)[0])
        out["cm_alpha"] = float(np.polyfit(a, [p["speeds"][mid]["cm_pitch"] for p in polar], 1)[0])
        xs = [p["speeds"][mid]["x_cp"] for p in polar if p["aoa"] != 0 and p["speeds"][mid]["x_cp"] is not None]
        if xs:
            out["x_cp"] = float(np.median(xs))
    return out
