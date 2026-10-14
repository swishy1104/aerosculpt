"""Optimization jobs in the CFD tunnels (General, RC): a design space, the search strategy
(strategy.py) and the evaluation levels it climbs.

  bodies (rockets, pods, uploads)   explore: coarse CFD on up to three angles
                                    confirm: CFD at the chosen fidelity on the whole envelope
  winged templates, drag at fixed   explore: component model -- vortex lattice (24 strips, 4 chordwise)
  lift                                       + NeuralFoil "large" sections + Hoerner fuselage
                                    confirm: the same model at full resolution (64 strips, 8
                                             chordwise, NeuralFoil "xlarge")
    No CFD in the loop for winged designs: the tunnel grid that fits a 2 m span puts only 3-5
    cells across the fuselage, and its measured run-to-run error (about 5 % of the total drag on
    the UAV template at draft fidelity) is larger than the changes being judged; its thin-wing
    lift is not validated either. The optimized design can be run in any tunnel afterwards.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from ..aero.body import fuselage_drag
from ..aero.neuralfoil import section_for
from ..aero.vlm import VLM
from ..aero.wing import STALL_MARGIN, WingModel
from ..cfd.analysis import FIDELITY
from ..geometry.ffd import ASPECTS, Deformer, design_space, refine
from .runner import OBJECTIVES, objective, polar_summary, wind_rotation
from .spaces import DeformSpace, FeatureSpace, FixedSpace, TemplateSpace
from .strategy import Search

from ..plans import EXPLORE_FID

SHIFTS = [(0.37, 0.23, 0.29), (-0.31, 0.41, -0.17)]
MIN_EXPLORE_CELLS = 12      # smallest body size across, in cells, that exploration can resolve
# when exploration had to run on the chosen grid, confirmation goes one level up (finer grid, higher
# Reynolds cap), charged as the chosen level; from the top level, the same level with more cells.
# Measured on the Mazda (2026-10-13): a draft-explored design read 0.39 on draft, 0.64 on standard, 0.67 on high
CONFIRM_UP = {"draft": "standard", "standard": "high"}
CONFIRM_REFINE = 2.0
MAX_ROUNDS = 4              # automatic mode: rounds of new ideas on the improved design
MIN_ROUND_BUDGET = 8        # designs a further round needs (probe, screening, a little search)
G0 = 9.80665
OBJ_LABELS = {**OBJECTIVES, "drag_fixed_lift": "Minimize drag at fixed lift"}


def constraints_from(o):
    c = {"min_volume": float(o.get("min_volume", 0.95)), "max_volume": float(o.get("max_volume", 1.5)),
         "hold_wing_area": bool(o.get("hold_wing_area", False))}
    for k in ("max_length", "max_width", "max_height", "min_thickness"):
        v = o.get(k)
        c[k] = float(v) if v not in (None, "", 0) else None
    return c


def subset(vals, n=3):
    vals = list(vals)
    if len(vals) <= n:
        return vals
    return [vals[0], vals[len(vals) // 2], vals[-1]]


class FixedLift:
    """Drag at the lift the aircraft needs, per envelope speed: wing model + fuselage."""

    RES = {False: {"ns": 24, "nc": 4, "nf": "large"}, True: {"ns": 64, "nc": 8, "nf": "xlarge"}}

    def __init__(self, r, space, mass_kg=None, keep_stability=True, tol=0.05):
        self.r = r
        self.space = space
        b0 = space.base.wing
        self.ref = b0.ref
        self.keep_stability = keep_stability
        self.tol = tol
        self.mac0 = b0.info["mac"]
        self.has_body = "length" in b0.info
        wm0 = self.model(space.base)
        self.np0 = None
        if mass_kg:
            self.lift = [float(mass_kg) * G0] * len(r.speeds)
            self.basis = f"flight mass {float(mass_kg):g} kg"
        else:
            a_mid = float(np.median(r.aoas))
            self.lift = [wm0.evaluate(v, r.air["rho"], r.air["mu"], alpha=a_mid)["lift_N"] for v in r.speeds]
            self.basis = f"the original design's lift at {a_mid:g}°"

    def model(self, cand, fine=False):
        key = "_wm_fine" if fine else "_wm"
        if getattr(cand, key, None) is None:
            b = cand.wing
            res = self.RES[fine]
            surfs = [dataclasses.replace(s, ns=res["ns"], nc=res["nc"]) for s in b.surfaces]
            setattr(cand, key, WingModel(surfs, self.ref["ref_area"], self.ref["ref_length"], self.ref["x_ref"],
                                         b.drag_surfaces, nf_size=res["nf"]))
        return getattr(cand, key)

    def wing_rows(self, cand, fine=False):
        key = "_rows_fine" if fine else "_rows"
        if getattr(cand, key, None) is None:
            wm = self.model(cand, fine)
            setattr(cand, key, [wm.evaluate(v, self.r.air["rho"], self.r.air["mu"], lift=L) for v, L in zip(self.r.speeds, self.lift)])
        return getattr(cand, key)

    def quick(self, cand):
        """Stall and stability from a small lattice and cached section maximum lift (no drag):
        cheap enough to screen every candidate the search considers (~10 ms)."""
        if getattr(cand, "_quick", None) is None:
            surfs = [dataclasses.replace(s, ns=16, nc=3) for s in cand.wing.surfaces]
            v = VLM(surfs, self.ref["ref_area"], self.ref["ref_length"], self.ref["x_ref"])
            rho, mu = self.r.air["rho"], self.r.air["mu"]
            rows = []
            for V, L in zip(self.r.speeds, self.lift):
                cl_req = L / (0.5 * rho * V * V * self.ref["ref_area"])
                a = v.alpha_for_cl(cl_req)
                cl = v.solve(a, V, rho)["cl_strip"]
                stall = False
                for si, s in enumerate(surfs):
                    idx = np.flatnonzero(v.strip_owner == si)
                    exp = np.array([v.strips[k]["exposed"] for k in idx])
                    ch = np.median([v.strips[k]["chord"] for k in idx])
                    clmax = section_for(s.section, "large").cl_max(rho * V * ch / mu)[0]
                    peak = float(np.max(cl[idx][exp] if exp.any() else cl[idx]))
                    stall = stall or peak > STALL_MARGIN * clmax
                rows.append({"v": V, "alpha": a, "CL": cl_req, "stall": stall})
            cand._quick = (v.neutral_point(), rows)
        return cand._quick

    def check(self, cand):
        if cand.wing is None or not cand.wing.surfaces:
            return None
        if any(min(s.root_chord, s.tip_chord) <= 0 for s in cand.wing.surfaces):
            return "a lifting surface has no chord"
        npx, rows = self.quick(cand)
        for row in rows:
            if row["stall"]:
                return f"the wing would stall at {row['v']:g} m/s (CL {row['CL']:.2f} needed)"
            if not (-10.0 < row["alpha"] < 16.0):
                return f"cruise attitude {row['alpha']:.1f}° at {row['v']:g} m/s is out of range"
        if self.keep_stability:
            if self.np0 is None:
                self.np0 = self.quick(self.space.base)[0]
            if abs(npx - self.np0) > self.tol * self.mac0:
                return f"neutral point moves {(npx - self.np0) * 1000:+.0f} mm (stability kept within {self.tol:.0%} MAC)"
        return None

    def component(self, cand, fine=False):
        rows = self.wing_rows(cand, fine)
        out = []
        for v, row in zip(self.r.speeds, rows):
            body = fuselage_drag(cand.info, v, self.r.air["rho"], self.r.air["mu"], row["alpha"])["drag_N"] \
                if self.has_body else 0.0
            out.append(self._row(v, row, body, "component model"))
        return out

    @staticmethod
    def _row(v, row, body, how, outside=False):
        total = row["drag_N"] + body
        return {"v": v, "alpha": row["alpha"], "lift_N": row["lift_N"], "CL": row["CL"], "induced_N": row["induced_N"],
                "profile_N": row["profile_N"], "body_N": body, "drag_N": total, "ld": row["lift_N"] / total if total > 0 else None,
                "power_W": total * v, "cl_margin": row["cl_margin"], "body_from": how, "extrapolated": outside,
                "surfaces": row["surfaces"]}


def _ratio(rows, base):
    return float(np.mean([a["drag_N"] / b["drag_N"] for a, b in zip(rows, base)]))


def optimize(r):
    """Run an optimization job on a set-up CFD Runner (General or RC). Writes result.json."""
    spec = r.spec
    o = spec.get("optimize", {})
    tmpl = spec.get("template")
    constraints = constraints_from(o)
    budget = int(o.get("iterations", 12))
    top_k = int(o.get("top_k", 4))
    features = getattr(r, "features", None) or []
    body = getattr(r, "base_body", r.frame.body)
    aspects = []
    r._progress(0.002, "Preparing the shape controls")
    if tmpl:
        space = TemplateSpace(tmpl["id"], tmpl.get("values", {}), o.get("params"), r.frame.center, constraints, r.zones)
        winged = tmpl["id"] in ("uav", "wing")
        can_mirror = tmpl["id"] == "rocket" and not any(v.key == "fin_count" for v in space.vars)
        mode = "parameters"
    else:
        aspects = [a for a in o.get("aspects", ["nose", "tail"]) if a in ASPECTS]
        if not aspects and not features:
            raise ValueError("choose at least one design aspect to modify")
        if aspects:
            if "fairing" in aspects:
                lo, hi = body.bbox()
                body = refine(body, max_edge=float((hi - lo).max()) / 45.0)
                if not features:
                    r.frame.body = body
            dfm = Deformer(body, design_space(aspects, float(o.get("max_change", 0.25))), r.zones)
            space = DeformSpace(dfm, constraints)
            mode = "aspects"
        else:
            space = FixedSpace(body)
            mode = "features"
        winged = False
        can_mirror = "camber" not in aspects
    if features:
        if winged:
            raise ValueError("shape features need the CFD tunnel; the UAV and wing templates use the wing model")
        space = FeatureSpace(space, features, r.feature_frame(), constraints)
        can_mirror = can_mirror and all(f["type"] != "wing" or f["values"].get("sides") != "single" for f in features)
        if not space.vars:
            raise ValueError("nothing to optimize: unlock a shape-feature dimension or choose a design aspect")
    obj_kind = "drag_fixed_lift" if winged else o.get("objective", "drag")
    if obj_kind not in ("drag", "ld", "drag_keep_lift") and not winged:
        raise ValueError("this goal is not available for this design")

    def score(polar, base):
        """The objective against the baseline, lower is better, the baseline scoring 1."""
        if obj_kind == "ld":
            ld, ld0 = -objective("ld", polar, base), -objective("ld", base, base)
            return ld0 / ld if ld > 1e-9 and ld0 > 1e-9 else 2.0
        if obj_kind == "drag_keep_lift":
            return objective("drag_keep_lift", polar, base)
        return objective("drag", polar, base)
    r.sim_angles = r.simulated_angles(space.base.mesh, can_mirror=can_mirror)
    fid_full = r.fid
    fid_x = FIDELITY[EXPLORE_FID.get(fid_full.name, "coarse")]
    if r.__class__.__name__ == "RCRunner" and getattr(r, "u_cap", None) and fid_x.u_lat > r.u_cap:
        fid_x = dataclasses.replace(fid_x, u_lat=r.u_cap)
    lift = FixedLift(r, space, o.get("mass_kg"), bool(o.get("keep_stability", True))) if winged else None
    wing_only = winged          # winged designs never need the CFD in the loop (see the module notes)

    # grid sized once for every shape the bounds allow (extremes along each parameter); only their
    # extent matters, so feature unions skip the buried-face bookkeeping
    def outline(vals):
        return space.build(vals, hidden=False) if isinstance(space, FeatureSpace) else space.build(vals)

    r._progress(0.005, "Sizing the wind tunnel for every shape the limits allow")

    extremes = [space.base.mesh]
    for v in space.vars:
        for val in (v.choices or [v.lo, v.hi]):
            c = outline({v.key: val})
            if not c.reason:
                extremes.append(c.mesh)
    for side in ("lo", "hi"):
        c = outline({v.key: getattr(v, side) for v in space.vars if not v.categorical})
        if not c.reason:
            extremes.append(c.mesh)
    lo, hi, r.frontal_plan = r.envelope_box(extremes, space.dfm if mode == "aspects" else None)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    box = (lo, hi)
    r.L_body = float(space.base.mesh.bbox()[1][0] - space.base.mesh.bbox()[0][0]) if tmpl else r.L_body

    def inside_grid(cand):
        for a in r.sim_angles:
            w = cand.mesh.transformed(R=wind_rotation(a, r.yaw))
            a_lo, a_hi = w.bbox()
            if (a_lo < box[0] - 1e-9).any() or (a_hi > box[1] + 1e-9).any():
                return "outside the tunnel grid sized for these bounds (widen the bounds before starting)"
        return None

    def guard(sp):
        """The tunnel grid, stall and stability on top of a space's own constraints."""
        own_full, own_fast = sp.full_check, sp.fast_check

        def fast_check(cand):
            why = own_fast(cand)
            if why or lift is None:
                return why
            return lift.check(cand)      # stall and stability: a vortex lattice, ~0.1 s

        def full_check(cand):
            why = own_full(cand)
            if why:
                return why
            if lift is not None:
                why = lift.check(cand)
                if why:
                    return why
            if not wing_only:
                return inside_grid(cand)
            return None

        sp.fast_check, sp.full_check = fast_check, full_check

    guard(space)

    angles_x = subset(r.aoas)
    sim_x = sorted({abs(a) for a in angles_x}) if r.symmetric else list(angles_x)
    per_x = len(sim_x) * (len(r.speeds) if getattr(r, "props", None) else len(r.re_groups))
    per_full = r.runs_per_design()
    explore_cfd = not winged
    n_confirm = int(o.get("confirm", 2))
    # screening runs count against the budget (paired ones included); confirmation is paired too
    per_one = per_full // max(1, len(r.sim_angles))      # one angle: the confirmation's shifted runs
    r.total_runs = max(1, (budget + 3) * per_x * explore_cfd + (n_confirm + 1) * (per_full + len(SHIFTS) * per_one) * (not wing_only))
    r.run_index = 0
    ev_x = r.make_evaluator(lo, hi, fid_x) if explore_cfd else None
    # a body only a few cells across on the exploration grid changes drag by several percent with
    # every sub-cell shift: no realistic design change could be told apart. On a GPU, explore on the
    # chosen grid instead (the confirmation then re-measures its own error there).
    from ..cfd.gpu import available as gpu_available
    if ev_x is not None and fid_x.name != fid_full.name and min(ev_x.info()["body_cells"]) < MIN_EXPLORE_CELLS and gpu_available():
        thin = min(ev_x.info()["body_cells"])
        fid_x = dataclasses.replace(fid_full, u_lat=fid_x.u_lat) if fid_x.u_lat < fid_full.u_lat else fid_full
        r.log(f"the body is only {thin:.1f} cells across on the {EXPLORE_FID.get(fid_full.name)} grid: exploring on {fid_full.name}")
        ev_x = r.make_evaluator(lo, hi, fid_x)
        ev_x.bill_as = EXPLORE_FID.get(fid_full.name, "coarse")     # charged as the exploration level the quote used
    # confirmation runs on a different grid from exploration, so a shape that only wins through the
    # exploration grid's error is caught. When exploration had to move onto the chosen grid, confirm
    # on a finer version of it (charged as the chosen level).
    fid_c = fid_full
    if explore_cfd and fid_x.name == fid_full.name:
        up = CONFIRM_UP.get(fid_full.name)
        fid_c = (dataclasses.replace(FIDELITY[up], name=fid_full.name) if up
                 else dataclasses.replace(fid_full, cells=int(fid_full.cells * CONFIRM_REFINE)))
        r.log(f"exploration used the {fid_full.name} grid itself: confirming on the "
              + (f"{up} grid" if up else f"{fid_full.name} grid with {CONFIRM_REFINE:g}x the cells"))
    ev_full = None
    r.log(f"optimizing {len(space.vars)} parameter(s): explore with "
          f"{'the wing model' if winged else f'{fid_x.name} CFD on {len(sim_x)} angle(s)'}, budget {budget} designs")

    base_x = {}

    def explore(cand, label):
        if winged:
            rows = lift.component(cand)
            base = base_x.setdefault("rows", rows)
            J = _ratio(rows, base)
            return {"J": J, "drag_ratio": J, "fixed_lift": _short(rows), "candidate": cand}
        r.fields_wanted = False
        polar, _ = r.evaluate_design(ev_x, cand.mesh, label, sim_x, aoas=angles_x, groups=None)
        base = base_x.setdefault("polar", polar)
        J = score(polar, base)
        s = polar_summary(polar)
        return {"J": J, "drag_ratio": J, "cd_ref": s["cd_at_min_aoa"], "converged": all(p["converged"] for p in polar)}

    base_shift = {}

    def noise():
        if winged:
            return []
        out = []
        for k, sh in enumerate(SHIFTS):
            m = space.base.mesh.transformed(t=np.array(sh) * ev_x.domain.dx)
            r.fields_wanted = False
            polar, _ = r.evaluate_design(ev_x, m, f"Measuring solver error ({k + 1}/2)", sim_x, aoas=angles_x)
            base_shift.setdefault(k, polar)
            out.append(score(polar, base_x["polar"]))
        return out

    def paired(cand, label):
        """The candidate shifted like the first error probe, against that shifted baseline."""
        m = cand.mesh.transformed(t=np.array(SHIFTS[0]) * ev_x.domain.dx)
        r.fields_wanted = False
        polar, _ = r.evaluate_design(ev_x, m, label, sim_x, aoas=angles_x)
        return score(polar, base_shift[0])

    kept = {}

    def confirm(cands):
        nonlocal ev_full
        if winged:
            base_rows = lift.component(cands[0], fine=True)
            return [{"J": _ratio(lift.component(c, fine=True), base_rows), "fixed_lift": _short(lift.component(c, fine=True))}
                    for c in cands]
        if ev_full is None:
            ev_full = r.make_evaluator(lo, hi, fid_c)
        dx = ev_full.domain.dx
        # the baseline, then the baseline at each error-probe shift; the candidates, then each at every
        # shift against the baseline at that shift: a gain must hold however the grid lines up. The
        # shifted runs use one angle (the one nearest zero): grid alignment is not a matter of angle.
        a0 = min(r.aoas, key=abs)
        one = [abs(a0) if r.symmetric else a0]
        seq = [("Confirming · baseline", cands[0], cands[0].mesh, "base", None)]
        seq += [(f"Confirming · solver error ({k + 1}/{len(SHIFTS)})", cands[0], cands[0].mesh.transformed(t=np.array(sh) * dx), "probe", k)
                for k, sh in enumerate(SHIFTS)]
        seq += [(f"Confirming · candidate {i}", c, c.mesh, "cand", None) for i, c in enumerate(cands[1:], 1)]
        seq += [(f"Confirming · candidate {i} (shifted {k + 1})", c, c.mesh.transformed(t=np.array(sh) * dx), "shift", k)
                for k, sh in enumerate(SHIFTS) for i, c in enumerate(cands[1:], 1)]
        rows_out, probes, shifted, js = [], {}, {}, [1.0]
        base_polar = None
        for label, cand, mesh, kind, k in seq:
            r.fields_wanted = kind in ("base", "cand")
            shifted_run = kind in ("probe", "shift")
            polar, fields = r.evaluate_design(ev_full, mesh, label, one if shifted_run else r.sim_angles, aoas=[a0] if shifted_run else None)
            if kind == "probe":
                probes[k] = polar
                js.append(score(polar, [q for q in base_polar if q["aoa"] == a0]))
                continue
            if kind == "shift":
                shifted.setdefault(id(cand), {})[k] = score(polar, probes[k])
                continue
            base_polar = base_polar or polar
            kept[id(cand)] = (polar, fields, None)
            rows_out.append({"J": score(polar, base_polar), "drag_ratio": objective("drag", polar, base_polar)})
        for row, c in zip(rows_out[1:], cands[1:]):
            got = shifted.get(id(c), {})
            row["J_shift"] = [got[k] for k in sorted(got)] or None
        sigma = float(np.std(js, ddof=1))
        rows_out[0]["eps"] = 2.0 * sigma
        r.log(f"confirmation error estimate: {sigma:.4f}")
        return rows_out

    def progress(f, stage, extra=None):
        fr = max(f, r.run_index / max(r.total_runs, 1))
        r._progress(min(0.97, fr), stage, extra)

    search = Search(space, explore, budget=budget, top_k=top_k, n_confirm=n_confirm,
                    eps_floor=0.004 if winged else 0.003, noise=noise, confirm=confirm,
                    log=r.log, progress=progress, cancel=r.check_cancel, seed=int(o.get("seed", 7)),
                    paired=None if winged else paired)
    search.relevance = None
    search.run()

    # ---- further rounds: new ideas on the improved design, until a round brings no confirmed gain --
    first_space, base_c = space, space.base
    all_history = list(search.history)
    rounds = [{"round": 1, "change": search.best["J"] - 1.0 if search.reliable else 0.0, "reliable": search.reliable,
               "adopted": [a["label"] for a in getattr(search, "adopted_all", [])], "stop": search.stop_reason}]
    kept_rounds = [(space, search.best["cand"])] if search.reliable else []
    total_J = search.best["J"] if search.reliable else 1.0
    used = search.evals
    if o.get("auto_ideas") and not winged:
        from ..geometry.features import propose as generate_ideas
        rnd = 1
        while search.reliable and rnd < MAX_ROUNDS and budget - used >= MIN_ROUND_BUDGET:
            rnd += 1
            start = search.best["cand"]
            ideas = generate_ideas(start.mesh, r.feature_frame())
            for f in ideas:
                f["name"] = f["name"].replace("(proposed)", f"(round {rnd})")
            if not ideas:
                break
            sp2 = FeatureSpace(FixedSpace(start.mesh), ideas, r.feature_frame(), constraints)
            guard(sp2)
            space = sp2                      # the error probes run on this round's starting design
            base_x.clear()
            base_shift.clear()
            r.log(f"round {rnd}: the design improved by {1 - total_J:.1%} so far; trying new ideas on it: "
                  + ", ".join(f["name"].split(" (")[0].lower() for f in ideas))
            s2 = Search(sp2, explore, budget=budget - used, top_k=top_k, n_confirm=n_confirm,
                        eps_floor=0.003, noise=noise, confirm=confirm, log=r.log, progress=progress,
                        cancel=r.check_cancel, seed=int(o.get("seed", 7)) + rnd, paired=paired)
            s2.relevance = None
            s2.run()
            used += s2.evals
            for h in s2.history:
                all_history.append({**h, "i": len(all_history), "round": rnd})
            rounds.append({"round": rnd, "change": s2.best["J"] - 1.0 if s2.reliable else 0.0, "reliable": s2.reliable,
                           "adopted": [a["label"] for a in getattr(s2, "adopted_all", [])], "stop": s2.stop_reason})
            if not s2.reliable:
                r.log(f"round {rnd}: nothing more beat the measured error; the design is as good as this search can make it")
                break
            for h in all_history:
                if h.get("round", 1) != rnd:
                    h["best"] = False
            total_J *= s2.best["J"]
            kept_rounds.append((sp2, s2.best["cand"]))
            search = s2
        space = first_space
    search.history = all_history

    # ---- results -------------------------------------------------------------------------------
    best_c = search.best["cand"]
    result = {"kind": "optimize", "mode": mode, "objective": obj_kind, "objective_label": OBJ_LABELS[obj_kind],
              "objective_indicative": obj_kind in ("ld", "drag_keep_lift"), "auto": bool(o.get("auto_ideas")),
              "variables": [v.public() for v in space.vars], "aspects": aspects,
              "best_values": {k: _jsonable(v) for k, v in best_c.values.items()},
              "improvement": 1.0 - total_J, "rounds": rounds, "zones": [{"name": z.name, "type": z.type} for z in r.zones],
              "keep_zones": [], "history": search.history, "relevance": search.relevance,
              "strategy": {"noise": search.noise_info, "ranking": search.ranking, "selected": search.selected,
                           "adopted": search.adopted, "adopted_all": getattr(search, "adopted_all", []), "stop_reason": search.stop_reason, "confirmed": search.confirmed,
                           "reliable": search.reliable, "epsilon": search.final_eps, "budget": budget, "top_k": top_k,
                           "designs_explored": used, "rejected": search.rejected,
                           "explore": "wing model (coarse)" if winged else fid_x.name,
                           "confirm": "wing model (full resolution)" if winged else fid_full.name,
                           "confirm_grid": None if winged or fid_c is fid_full else (CONFIRM_UP.get(fid_full.name) or f"{fid_full.name}, finer"),
                           "cache_hits": (ev_x.cache_hits if ev_x else 0) + (ev_full.cache_hits if ev_full else 0),
                           "explore_cells": ev_x.info()["body_cells"] if ev_x else None,
                           "confirm_cells": ev_full.info()["body_cells"] if ev_full else None},
              "cfd_runs": r.cfd_runs, "elapsed_s": time.time() - r.t0, "best_index": next((h["i"] for h in search.history if h.get("best")), 0)}
    if tmpl:
        result["template"] = {"id": tmpl["id"], "base_values": {k: _jsonable(v) for k, v in base_c.values.items()},
                              "best_values": result["best_values"], "info_base": base_c.info, "info_best": best_c.info}
    feat_rounds = [(sp_r, b) for sp_r, b in (kept_rounds or [(space, best_c)]) if isinstance(sp_r, FeatureSpace)]
    if feat_rounds:
        changes = [c for sp_r, best_r in feat_rounds for c in sp_r.changes(best_r)]
        result["features"] = {"changes": [{**c, "base": _jsonable(c["base"]), "best": _jsonable(c["best"])} for c in changes],
                              "list": [{"id": f["id"], "name": f["name"], "type": f["type"]} for sp_r, _ in feat_rounds for f in sp_r.features]}
    if isinstance(space, FeatureSpace) and mode != "aspects":
        result["volume_ratio"] = space.FT.union_volume(best_c.mesh) / space.base_union if space.base_union else 1.0
    elif mode == "aspects":
        result["volume_ratio"] = space.dfm.volume_ratio(best_c.mesh)
        result["keep_zones"] = space.dfm.keep.report() if space.dfm.keep is not None else []
    else:
        bv = base_c.info.get("volume")
        result["volume_ratio"] = (best_c.info.get("volume") / bv) if bv else 1.0
    if lift is not None:
        result["fixed_lift"] = {"basis": lift.basis, "lift_N": lift.lift, "speeds": r.speeds,
                                "baseline": lift.component(base_c, fine=True), "best": lift.component(best_c, fine=True),
                                "body_from": "Hoerner" if lift.has_body else None,
                                "neutral_point_shift_mm": (lift.quick(best_c)[0] - lift.quick(base_c)[0]) * 1000}
    (r.out / "baseline_body.stl").write_bytes(base_c.mesh.to_stl_bytes())
    (r.out / "best_body.stl").write_bytes(best_c.mesh.to_stl_bytes())
    lead = search.promising
    if not search.reliable and lead is not None:
        # nothing beat the measured error: still show the most promising design, clearly labelled
        (r.out / "promising_body.stl").write_bytes(lead["cand"].mesh.to_stl_bytes())
        (r.out / "promising.stl").write_bytes(r.frame.to_file_frame(lead["cand"].mesh).to_stl_bytes(b"AeroSculpt unconfirmed lead"))
        result["promising"] = {"change": lead["J"] - 1.0, "eps": search.final_eps, "mesh": "promising_body.stl",
                               "values": {k: _jsonable(v) for k, v in lead["cand"].values.items()},
                               "features": [{**c, "base": _jsonable(c["base"]), "best": _jsonable(c["best"])} for c in space.changes(lead["cand"])]
                               if features else None}
    export = r.frame.to_file_frame(best_c.mesh)
    (r.out / "optimized.stl").write_bytes(export.to_stl_bytes(b"AeroSculpt optimized design"))
    result["meshes"] = {"baseline": "baseline_body.stl", "best": "best_body.stl"}
    if wing_only:
        pb, pt = _wing_polar(r, lift, base_c), _wing_polar(r, lift, best_c)
        result.update(r.common_light() if hasattr(r, "common_light") else _common_light(r))
        result["baseline"] = {"polar": pb, "summary": polar_summary(pb)}
        result["best"] = {"polar": pt, "summary": polar_summary(pt)}
        result["drag_change"] = search.best["J"] - 1.0
        result["fields"] = None
        result["verified"] = None
    else:
        if id(base_c) not in kept:
            confirm([base_c])
        pb, fb, rb = kept[id(base_c)]
        pt, ft, rt = kept.get(id(best_c), (pb, fb, rb))
        result.update(r.common(ev_full))
        result["baseline"] = {"polar": pb, "summary": polar_summary(pb)}
        result["best"] = {"polar": pt, "summary": polar_summary(pt)}
        result["drag_change"] = objective("drag", pt, pb) - 1.0
        sb = r.save_fields("baseline", fb, r.sim_angles)
        st = r.save_fields("best", ft, r.sim_angles)
        result["fields"] = {"baseline": r.field_index(sb), "best": r.field_index(st)}
        result["verified"] = {"fidelity": fid_full.name, "setup": ev_full.info(), "confirm_of": "explore"}
        if lift is not None and rb is not None:
            result["fixed_lift"]["baseline"] = rb
            result["fixed_lift"]["best"] = rt
            result["fixed_lift"]["body_from"] = "CFD"
        result["cfd_runs"] = r.cfd_runs
    result["cached_runs"] = getattr(r, "cached_runs", 0)
    if winged:
        r.model_credits = 1 + max(0, search.evals - 1) // 10
    r.history = search.history
    r.write_json("result.json", result)
    r.write_json("history.json", search.history)
    r._progress(1.0, "Done", {"history": search.history[-60:]})
    return result


def _short(rows):
    return [{"v": x["v"], "alpha": round(x["alpha"], 3), "drag_N": x["drag_N"], "body_N": x["body_N"],
             "induced_N": x["induced_N"], "profile_N": x["profile_N"]} for x in rows]


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _common_light(r):
    return {"tunnel": r.spec.get("tunnel", "general"),
            "setup": {"method": "wing model", "air": r.air, "regimes": ["wing model"], "symmetric": False,
                      "simulated_angles": list(r.aoas), "runs_per_design": 0, "fidelity": "wing model",
                      "ref_area": r.ref_area, "ref_length": r.ref_length, "re_groups": []},
            "envelope": {"speeds": r.speeds, "aoas": r.aoas, "yaw": r.yaw,
                         "altitude": r.spec.get("envelope", {}).get("altitude", 0.0)},
            "model": r.spec.get("model", {})}


def _wing_polar(r, lift, cand):
    """A polar in the CFD result format from the component model (lifting surfaces from the wing
    model at full resolution, fuselage from Hoerner's method)."""
    wm = lift.model(cand, fine=True)
    polar = []
    for a in r.aoas:
        speeds = []
        for v in r.speeds:
            w = dict(wm.evaluate(v, r.air["rho"], r.air["mu"], alpha=a))
            if lift.has_body:
                w["profile_N"] += fuselage_drag(cand.info, v, r.air["rho"], r.air["mu"], a)["drag_N"]
                w["drag_N"] = w["induced_N"] + w["profile_N"]
            q = 0.5 * r.air["rho"] * v * v
            S = r.ref_area
            cd = w["drag_N"] / (q * S)
            speeds.append({"v": v, "re": r.air["rho"] * v * r.ref_length / r.air["mu"], "mach": v / r.air["a"], "cd": cd,
                           "cd_friction": w["profile_N"] / (q * S), "cd_pressure": w["induced_N"] / (q * S), "cf": None, "q": q,
                           "drag_N": w["drag_N"], "lift_N": w["lift_N"], "side_N": 0.0, "cl": w["lift_N"] / (q * S),
                           "cm_pitch": 0.0, "ld": w["lift_N"] / w["drag_N"] if w["drag_N"] > 0 else None,
                           "power_W": w["drag_N"] * v, "compressibility": "incompressible", "mach_crit": None})
        mid = speeds[len(speeds) // 2]
        polar.append({"aoa": a, "mirror": False, "regime": "wing model", "mach_crit": None, "cl": mid["cl"], "cs": 0.0,
                      "cm_pitch": 0.0, "cm_roll": 0.0, "cm_yaw": 0.0, "cd_pressure": mid["cd_pressure"], "converged": True,
                      "cd_std": 0.0, "blockage": 0.0, "seconds": 0.0, "speeds": speeds})
    return polar
