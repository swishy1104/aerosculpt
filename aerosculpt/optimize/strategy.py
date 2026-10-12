"""How the optimizer spends simulation runs.

  1. Baseline, and the solver's measured error: the baseline re-run with the body shifted by a
     fraction of a grid cell. The scatter is the noise a shape change has to beat; a change is
     only accepted when it improves the objective by more than twice that (or a floor).
  2. Screening: every unlocked parameter is moved once (a quarter of its range), every
     alternative of a choice parameter (nose profile, airfoil family) is tried once. Parameters
     are ranked by their effect; only the top few with an effect above the error are optimized.
  3. Search: Bayesian optimization (Gaussian process, expected improvement) over the selected
     parameters, starting from the screening runs. Stops at the run budget, or when several
     designs in a row bring no gain larger than the error.
  4. Confirmation: the best candidates and the baseline at the next fidelity level (full-resolution
     CFD). The result is reported as an improvement only if it survives there by more than the
     error; otherwise "no reliable improvement".

Every candidate passes the design space's checks (constraints, regions, watertight, no
self-intersection) before it is simulated; a rejected candidate costs no run.
"""
from __future__ import annotations

import numpy as np

from .bo import GP, propose
from .spaces import unit_to_values, values_to_unit

SCREEN_STEP = 0.25


class Search:
    def __init__(self, space, evaluate, *, budget, top_k=4, eps_floor=0.003, patience=None, n_confirm=2,
                 noise=None, confirm=None, log=None, progress=None, cancel=None, seed=7, label_stage=None, paired=None):
        """evaluate(candidate, label) -> dict with "J" (objective, baseline = 1, lower is better);
        noise() -> list of J for shifted baselines (None: deterministic model);
        confirm(list of candidates incl. baseline first) -> list of dicts with "J"."""
        self.space = space
        self.evaluate = evaluate
        self.budget = int(budget)
        self.top_k = int(top_k)
        self.eps_floor = float(eps_floor)
        self.patience = patience
        self.n_confirm = int(n_confirm)
        self.noise = noise
        self.confirm = confirm
        self.log = log or (lambda m: None)
        self.progress = progress or (lambda f, s, extra=None: None)
        self.cancel = cancel or (lambda: None)
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.evals = 0
        self.rejected = 0
        self.stage_names = label_stage or {}
        # paired(cand, label) -> objective of the candidate shifted by part of a cell, against the
        # baseline shifted the same way. A whole-body shift moves both drags together; a design
        # change is compared at equal shifts, so its own error is the spread of that difference.
        self.paired = paired
        self.pairs = []
        self.screened_all = []          # every screened design: the most promising lead may be one of them
        self.promising = None

    # ---- bookkeeping ---------------------------------------------------------------------
    def _entry(self, stage, status, cand, res=None, note=None, **extra):
        e = {"i": len(self.history), "stage": stage, "status": status,
             "values": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in cand.values.items()
                        if k in self.keys_all},
             "objective": None if res is None else float(res["J"])}
        if note:
            e["note"] = note
        if res is not None:
            e.update({k: v for k, v in res.items() if k not in ("J", "polar", "fields", "candidate")})
        e.update(extra)
        self.history.append(e)
        return e

    def _tick(self, stage):
        total = self.budget + 4 + (self.n_confirm + 2 if self.confirm else 0)
        self.progress(min(0.97, 0.02 + 0.95 * self.evals / total), stage, {"history": self.history[-60:]})

    def _effect(self, cand, res, J0, label):
        """Relative change against the baseline: the mean of the plain and the shifted comparison
        when paired evaluation is available."""
        eff = res["J"] / J0 - 1.0
        if self.paired is None:
            return eff
        self.cancel()
        j1 = self.paired(cand, label + " (shifted)")
        self.evals += 1
        e1 = float(j1) - 1.0
        self.pairs.append((eff, e1))
        return 0.5 * (eff + e1)

    def _eval(self, cand, label):
        self.cancel()
        res = self.evaluate(cand, label)
        self.evals += 1
        return res

    # ---- the four steps -------------------------------------------------------------------
    def run(self):
        sp = self.space
        self.keys_all = [v.key for v in sp.vars]
        cont = [v for v in sp.vars if not v.categorical]
        cats = [v for v in sp.vars if v.categorical]
        if not sp.vars:
            raise ValueError("every parameter is locked: unlock at least one for the optimizer to change")
        base = sp.base
        reason = sp.full_check(base)
        if reason:
            raise ValueError(f"the starting design cannot be simulated: {reason}")
        r0 = self._eval(base, "Baseline")
        self._entry("baseline", "baseline", base, r0)
        J0 = r0["J"]

        # 1. measured error
        samples = [J0]
        if self.noise is not None:
            js = self.noise()
            self.evals += len(js)
            for k, j in enumerate(js):
                self.history.append({"i": len(self.history), "stage": "noise", "status": "error probe",
                                     "values": {}, "objective": float(j),
                                     "note": f"baseline shifted by part of a grid cell ({k + 1})"})
                samples.append(j)
        sigma = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        eps = max(2.0 * sigma, self.eps_floor)
        self.noise_info = {"sigma": sigma, "eps": eps, "samples": len(samples),
                           "method": "baseline re-run with sub-cell shifts" if len(samples) > 1 else "deterministic model",
                           "floor": self.eps_floor}
        self.log(f"measured error: sigma {sigma:.4f} from {len(samples)} sample(s); changes must beat {eps:.2%}")
        self._tick("Measuring the solver's error")

        # 2. screening
        ranking = []
        screened = {}
        best_cat = None
        for v in cats:
            for ch in v.choices:
                if ch == v.base:
                    continue
                cand = sp.build({v.key: ch})
                why = sp.full_check(cand)
                if why:
                    self._entry("screen", "rejected", cand, note=why, param=v.key)
                    ranking.append({"key": v.key, "label": v.label, "choice": ch, "effect": None, "note": why})
                    continue
                res = self._eval(cand, f"Screening · {v.label}: {ch}")
                eff = self._effect(cand, res, J0, f"Screening · {v.label}: {ch}")
                self._entry("screen", "screened", cand, res, param=v.key)
                self.screened_all.append({"J": J0 * (1.0 + eff), "cand": cand, "res": res})
                ranking.append({"key": v.key, "label": v.label, "choice": ch, "effect": eff, "categorical": True,
                                "significant": abs(eff) > eps, "_res": res, "_cand": cand})
                self._tick("Screening parameters")
        for v in cont:
            room_up, room_dn = v.hi - v.base, v.base - v.lo
            step = SCREEN_STEP * (v.hi - v.lo)
            tries = [min(v.hi, v.base + step), max(v.lo, v.base - step)] if room_up >= room_dn else \
                [max(v.lo, v.base - step), min(v.hi, v.base + step)]
            done = False
            for val in tries:
                if abs(val - v.base) < 1e-12:
                    continue
                cand = sp.build({v.key: val})
                why = sp.full_check(cand)
                if why:
                    self._entry("screen", "rejected", cand, note=why, param=v.key)
                    continue
                res = self._eval(cand, f"Screening · {v.label}")
                eff = self._effect(cand, res, J0, f"Screening · {v.label}")
                self._entry("screen", "screened", cand, res, param=v.key)
                self.screened_all.append({"J": J0 * (1.0 + eff), "cand": cand, "res": res})
                helps = (val > v.base) == (eff < 0)
                ranking.append({"key": v.key, "label": v.label, "unit": v.unit, "tried": val, "base": v.base,
                                "effect": eff, "per_range": eff * (v.hi - v.lo) / abs(val - v.base),
                                "direction": "increase" if helps else "decrease", "significant": abs(eff) > eps})
                screened[v.key] = (val, res, cand)
                done = True
                self._tick("Screening parameters")
                break
            if not done:
                ranking.append({"key": v.key, "label": v.label, "unit": v.unit, "effect": None,
                                "note": "every move broke a constraint"})
        if self.pairs:
            sd = float(np.sqrt(np.mean([(a - b) ** 2 for a, b in self.pairs]) / 2.0))
            eps_p = max(2.0 * sd, self.eps_floor, 0.3 * eps)
            self.log(f"paired error: {sd:.4f} from {len(self.pairs)} design(s) at two sub-cell shifts; "
                     f"changes must beat {eps_p:.2%} (whole-body shift: {eps:.2%})")
            self.noise_info.update({"paired_sigma": sd, "whole_body_eps": eps, "eps": eps_p, "method": "paired sub-cell shifts"})
            eps = eps_p
            for r in ranking:
                if r.get("effect") is not None:
                    r["significant"] = abs(r["effect"]) > eps
        for r in ranking:
            if r.get("categorical") and r.get("effect") is not None and r["effect"] < -eps and \
                    (best_cat is None or r["effect"] < best_cat[2]["J"] / J0 - 1.0):
                best_cat = (r["key"], r["choice"], r["_res"], r["_cand"])
        for r in ranking:
            r.pop("_res", None)
            r.pop("_cand", None)
        ranking.sort(key=lambda r: -abs(r["effect"]) if r.get("effect") is not None else 1.0)
        incumbent = {"J": J0, "cand": base, "res": r0}
        adopted = None
        self.adopted_all = []
        rescreened = {}
        helped = sorted([r for r in ranking if r.get("categorical") and r.get("effect") is not None and r["effect"] < -eps
                         and best_cat is not None and r["key"] != best_cat[0]], key=lambda r: r["effect"])
        if best_cat is not None:
            k, ch, res, cand = best_cat
            adopted = {"key": k, "choice": ch, "label": sp.label(k), "effect": res["J"] / J0 - 1.0}
            self.adopted_all = [adopted]
            incumbent = {"J": res["J"], "cand": cand, "res": res}
            self.log(f"adopting {sp.label(k)} = {ch} ({res['J'] / J0 - 1:+.2%})")
            # other choices that helped on their own (generative features), added on top one at a time
            for r in helped:
                c2 = sp.build({**incumbent["cand"].values, r["key"]: r["choice"]})
                if sp.full_check(c2):
                    continue
                r2 = self._eval(c2, f"Adding · {r['label']}")
                self._entry("screen", "screened", c2, r2, param=r["key"])
                self._tick("Screening parameters")
                if r2["J"] < incumbent["J"] - eps * J0:
                    self.adopted_all.append({"key": r["key"], "choice": r["choice"], "label": r["label"], "effect": r2["J"] / J0 - 1.0})
                    self.log(f"adopting {r['label']} = {r['choice']} too ({r2['J'] / J0 - 1:+.2%} overall)")
                    incumbent = {"J": r2["J"], "cand": c2, "res": r2}
            k, ch, res, cand = k, ch, incumbent["res"], incumbent["cand"]
            # parameters that only matter for the adopted choice (a power-law exponent) are screened now
            new = sp.rebase(cand.values) if hasattr(sp, "rebase") else []
            self.keys_all = [v.key for v in sp.vars]
            for v in new:
                if v.categorical:
                    continue
                step = SCREEN_STEP * (v.hi - v.lo)
                for val in ([min(v.hi, v.base + step), max(v.lo, v.base - step)] if v.hi - v.base >= v.base - v.lo
                            else [max(v.lo, v.base - step), min(v.hi, v.base + step)]):
                    c2 = sp.build({**cand.values, v.key: val})
                    if sp.full_check(c2):
                        continue
                    r2 = self._eval(c2, f"Screening · {v.label}")
                    eff = r2["J"] / res["J"] - 1.0
                    self._entry("screen", "screened", c2, r2, param=v.key)
                    ranking.append({"key": v.key, "label": v.label, "unit": v.unit, "tried": val, "base": v.base,
                                    "effect": eff, "per_range": eff * (v.hi - v.lo) / abs(val - v.base),
                                    "direction": "increase" if (val > v.base) == (eff < 0) else "decrease",
                                    "significant": abs(eff) > eps, "after": f"{sp.label(k)} = {ch}"})
                    rescreened[v.key] = (val, r2, c2)
                    break
            ranking.sort(key=lambda r: -abs(r["effect"]) if r.get("effect") is not None else 1.0)
        sel = [r["key"] for r in ranking if r.get("effect") is not None and not r.get("categorical")
               and abs(r["effect"]) >= 0.5 * eps][: self.top_k]
        for r in ranking:
            r["selected"] = r["key"] in sel
        self.ranking = ranking
        self.selected = sel
        self.adopted = adopted
        self.log("screening ranking: " + ", ".join(f"{r['label']} {r['effect']:+.2%}" for r in ranking if r.get("effect") is not None))

        # 3. search
        stop = None
        evaluated = [incumbent] + [{"J": res["J"], "cand": cand, "res": res}
                                   for k, (val, res, cand) in screened.items()]
        if not sel:
            stop = ("added " + ", ".join(a["label"].lower() for a in self.adopted_all) + "; no dimension changed it further by more than the measured error"
                    if self.adopted_all else "no parameter changes the objective by more than the measured error")
        elif self.evals >= self.budget:
            stop = "the run budget was used by the screening"
        else:
            base_vals = dict(incumbent["cand"].values)
            X = [values_to_unit(sp, sel, base_vals)]
            Y = [incumbent["J"]]
            pts = screened if adopted is None else rescreened
            for k in sel:
                if k in pts:
                    val, res, cand = pts[k]
                    X.append(values_to_unit(sp, sel, cand.values))
                    Y.append(res["J"])
            bad = []

            def build_at(x01, preview=False):
                vals = dict(base_vals)
                vals.update(unit_to_values(sp, sel, x01))
                return sp.build(vals, preview=preview)

            def feasible(x01):
                if any(np.linalg.norm(np.asarray(x01) - b) < 0.02 for b in bad):
                    return False
                return sp.fast_check(build_at(x01, preview=True)) is None

            patience = self.patience or max(4, len(sel) + 1)
            stall = 0
            n_rej = 0
            while self.evals < self.budget:
                try:
                    x, _, _ = propose(np.array(X), np.array(Y), feasible, self.rng)
                    how = "bayesian"
                except Exception as e:  # noqa: BLE001 - surrogate trouble: explore instead
                    self.log(f"surrogate fallback: {type(e).__name__}")
                    x, how = self.rng.random(len(sel)), "random"
                cand = build_at(x)
                why = sp.full_check(cand)
                if why:
                    bad.append(np.asarray(x))
                    n_rej += 1
                    self.rejected += 1
                    self._entry("search", "rejected", cand, note=why)
                    if n_rej > 60:
                        stop = "too many candidates broke the constraints"
                        break
                    continue
                try:
                    res = self._eval(cand, f"Design {self.evals + 1} of {self.budget}")
                except FloatingPointError:
                    bad.append(np.asarray(x))
                    self._entry("search", "diverged", cand, note="flow solution diverged")
                    continue
                X.append(np.asarray(x))
                Y.append(res["J"])
                evaluated.append({"J": res["J"], "cand": cand, "res": res})
                if res["J"] < incumbent["J"] * (1.0 - eps):
                    status, stall = "accepted", 0
                    incumbent = {"J": res["J"], "cand": cand, "res": res}
                elif res["J"] < incumbent["J"]:
                    status, stall = "within error", stall + 1
                else:
                    status, stall = "worse", stall + 1
                self._entry("search", status, cand, res, how=how)
                self.log(f"design {self.evals}: J {res['J']:.4f} ({status}); best {incumbent['J']:.4f}")
                self._tick(f"Design {self.evals} of {self.budget}")
                if stall >= patience:
                    stop = f"{patience} designs in a row brought no gain larger than the measured error"
                    break
            else:
                stop = "run budget reached"
            if len(X) >= 4:
                try:
                    self.relevance = {k: float(r) for k, r in zip(sel, GP(np.array(X), np.array(Y)).relevance())}
                except Exception:  # noqa: BLE001
                    self.relevance = None
        self.stop_reason = stop
        self.log(f"search stopped: {stop}")

        # the best design seen, significant or not: offered as an unconfirmed lead when nothing is confirmed
        leads = [e for e in evaluated + self.screened_all if e["cand"] is not base and e["J"] < J0]
        self.promising = min(leads, key=lambda e: e["J"]) if leads else None

        # 4. confirmation
        pool = sorted([e for e in evaluated if e["J"] < J0 * (1.0 - eps)], key=lambda e: e["J"])
        picks = []
        for e in pool:
            if all(_distinct(e["cand"], p["cand"], sp) for p in picks):
                picks.append(e)
            if len(picks) >= self.n_confirm:
                break
        self.confirmed = None
        best = incumbent if incumbent["J"] < J0 * (1.0 - eps) else {"J": J0, "cand": base, "res": r0}
        reliable = best["cand"] is not base
        if self.confirm is not None and picks:
            rows = self.confirm([base] + [p["cand"] for p in picks])
            eps_c = max(eps, rows[0].get("eps", 0.0))
            jb = rows[0]["J"]
            conf = []
            for p, r in zip(picks, rows[1:]):
                j = r["J"] / jb
                if r.get("J_shift") is not None:          # paired: the mean of the plain and shifted comparisons
                    j = 0.5 * (j + r["J_shift"])
                conf.append({"explore_J": p["J"], "J": j, "res": r, "cand": p["cand"], "pair": (r["J"] / jb, r.get("J_shift"))})
            pairs = [c["pair"] for c in conf if c["pair"][1] is not None]
            if pairs:
                sd = float(np.sqrt(np.mean([(a - b) ** 2 for a, b in pairs]) / 2.0))
                eps_c = max(2.0 * sd, self.eps_floor, min(eps_c, max(eps, 0.3 * eps_c)))
            for c in conf:
                self._entry("confirm", "confirmed" if c["J"] < 1.0 - eps_c else "not confirmed", c["cand"], {**c["res"], "J": c["J"]},
                            explore_objective=c["explore_J"])
            top = min(conf, key=lambda c: c["J"])
            self.confirmed = {"eps": eps_c, "baseline": rows[0], "rows": [{"explore": c["explore_J"], "confirmed": c["J"]} for c in conf]}
            if top["J"] < 1.0 - eps_c:
                best = {"J": top["J"], "cand": top["cand"], "res": top["res"], "baseline_res": rows[0]}
                reliable = True
            else:
                best = {"J": 1.0, "cand": base, "res": rows[0], "baseline_res": rows[0]}
                reliable = False
            eps = eps_c
        elif self.confirm is not None:
            reliable = False
        for h in self.history:
            h["best"] = reliable and h.get("values") == {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                         for k, v in best["cand"].values.items() if k in self.keys_all} \
                and h["stage"] in ("search", "screen", "confirm")
        self.best = best
        self.reliable = reliable
        self.final_eps = eps
        return self


def _distinct(a, b, sp, tol=0.03):
    for v in sp.vars:
        if v.categorical:
            if a.values.get(v.key) != b.values.get(v.key):
                return True
        elif abs(a.values[v.key] - b.values[v.key]) > tol * (v.hi - v.lo):
            return True
    return False
