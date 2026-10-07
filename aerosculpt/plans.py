"""Subscription plans and the limits each one enforces. The server is the source of truth;
the pricing page and the workspace render from GET /api/plans."""
from __future__ import annotations

import math

ALL_ASPECTS = ["nose", "tail", "section", "length", "camber", "fairing"]
ALL_OBJECTIVES = ["drag", "ld", "drag_keep_lift", "drag_fixed_lift"]
HYPER_OBJECTIVES = ["drag", "peak_temp"]
TUNNELS = ["general", "rc", "supersonic"]

# Compute credits charged per CFD run at each fidelity (roughly cells x steps). A run is one
# angle of attack; every airspeed in the envelope comes from the same run.
CREDITS_PER_RUN = {"coarse": 0.4, "draft": 1, "standard": 3, "high": 8, "ultra": 20}
# the optimizer explores on the next coarser grid and confirms at the fidelity the user chose
EXPLORE_FID = {"draft": "coarse", "standard": "draft", "high": "draft", "ultra": "standard"}
CONFIRM_DESIGNS = 4          # baseline, its error probe and two candidates

PLANS = {
    "lite": {
        "key": "lite", "name": "Lite", "price_month": 0, "tagline": "Free forever, for first looks",
        "monthly_credits": 40, "max_iterations": 12, "fidelities": ["draft"],
        "objectives": ["drag", "drag_fixed_lift"], "max_aspects": 2, "aspects": ALL_ASPECTS, "max_top_k": 2,
        "max_angles": 3, "max_speeds": 4, "max_zones": 2, "max_propellers": 1, "max_iterations_panel": 30,
        "max_projects": 3, "max_upload_mb": 25, "max_triangles": 300_000,
        "export": False, "verify": False, "reports": False, "api": False,
        "priority": 0, "concurrent_jobs": 1,
        "features": ["All three tunnels: General, RC, Supersonic", "Draft-fidelity wind tunnel", "Sweep 3 angles × 4 airspeeds",
                     "1 propeller in the RC tunnel", "Rocket, UAV and wing templates with named dimensions",
                     "12-design optimization of the 2 most influential parameters", "2 protected regions",
                     "3 projects · STL, OBJ, STEP upload", "40 compute credits a month"],
    },
    "pro": {
        "key": "pro", "name": "Pro", "price_month": 49, "tagline": "For engineers shipping airframes",
        "monthly_credits": 1500, "max_iterations": 60, "fidelities": ["draft", "standard", "high"],
        "objectives": ALL_OBJECTIVES, "max_aspects": len(ALL_ASPECTS), "aspects": ALL_ASPECTS, "max_top_k": 6,
        "max_angles": 9, "max_speeds": 12, "max_zones": 20, "max_propellers": 4, "max_iterations_panel": 200,
        "max_projects": None, "max_upload_mb": 150, "max_triangles": 2_000_000,
        "export": True, "verify": True, "reports": True, "api": False,
        "priority": 10, "concurrent_jobs": 2, "trial": True,
        "features": ["Standard and high fidelity", "Sweep 9 angles × 12 airspeeds or Mach numbers",
                     "60-design optimization of up to 6 parameters, all 6 aspects", "Up to 4 propellers · 200-design supersonic optimization",
                     "Heating, wall temperature and material limits to Mach 12",
                     "Unlimited protected regions and keep-inside volumes", "Export optimized STL + reports",
                     "High-fidelity verification of the winner", "1,500 compute credits a month"],
    },
    "team": {
        "key": "team", "name": "Team", "price_month": 199, "tagline": "For labs, teams and suppliers",
        "monthly_credits": 6000, "max_iterations": 150, "fidelities": ["draft", "standard", "high", "ultra"],
        "objectives": ALL_OBJECTIVES, "max_aspects": len(ALL_ASPECTS), "aspects": ALL_ASPECTS, "max_top_k": 8,
        "max_angles": 15, "max_speeds": 20, "max_zones": 40, "max_propellers": 8, "max_iterations_panel": 500,
        "max_projects": None, "max_upload_mb": 400, "max_triangles": 5_000_000,
        "export": True, "verify": True, "reports": True, "api": True,
        "priority": 20, "concurrent_jobs": 4,
        "features": ["Everything in Pro", "Ultra fidelity (6M-cell tunnel)", "Sweep 15 angles × 20 airspeeds",
                     "150-design optimization of up to 8 parameters", "6,000 compute credits a month", "REST API keys",
                     "Highest queue priority, 4 jobs at once"],
    },
}


def plan_for(user) -> dict:
    """Effective plan for a user row (a lapsed trial or cancelled subscription is Lite)."""
    key = user["plan"] if user and user["plan"] in PLANS else "lite"
    if key != "lite" and user["plan_status"] not in ("active", "trialing", "past_due"):
        key = "lite"
    return PLANS[key]


def estimate_runs(spec: dict, re_groups: int = 1) -> dict:
    """CFD runs a job will need, by fidelity (an upper bound: symmetric bodies need fewer).
    With propellers turning every (angle, airspeed) pair is a run of its own."""
    env = spec.get("envelope", {})
    if spec.get("tunnel") == "rc" and spec.get("propellers"):
        per_design = len(env.get("aoas", [0])) * len(env.get("speeds", [0]))
    else:
        per_design = len(env.get("aoas", [0])) * max(1, re_groups)
    fid = spec.get("fidelity", "draft")
    if spec.get("kind") != "optimize":
        return {fid: per_design}
    o = spec.get("optimize", {})
    tid = (spec.get("template") or {}).get("id")
    if tid in ("wing", "uav") and spec.get("tunnel") != "supersonic":
        return {}                                   # winged designs: the wing model, no CFD
    runs = {fid: per_design * CONFIRM_DESIGNS}
    if True:
        n_ang = min(3, len(env.get("aoas", [0])))
        with_props = spec.get("tunnel") == "rc" and spec.get("propellers")
        per_x = n_ang * (len(env.get("speeds", [0])) if with_props else max(1, re_groups))
        fx = EXPLORE_FID.get(fid, "coarse")
        runs[fx] = runs.get(fx, 0) + per_x * (int(o.get("iterations", 0)) + 3)
    return runs


def estimate_credits(spec: dict, re_groups: int = 1) -> int:
    if spec.get("tunnel") == "supersonic":
        designs = 1 + (int(spec.get("optimize", {}).get("iterations", 0)) + 1 if spec.get("kind") == "optimize" else 0)
        return 1 + max(0, designs - 1) // 10
    return max(1, math.ceil(sum(n * CREDITS_PER_RUN.get(f, 1) for f, n in estimate_runs(spec, re_groups).items())))


def check_spec(spec: dict, plan: dict):
    """Raise PermissionError / ValueError with a user-facing message if the plan does not
    allow this job."""
    supersonic = spec.get("tunnel") == "supersonic"
    fid = spec.get("fidelity", "draft")
    if not supersonic and fid not in plan["fidelities"]:
        raise PermissionError(f"{fid.title()} fidelity needs a higher plan (your plan: {plan['name']}).")
    if len(spec.get("propellers", [])) > plan["max_propellers"]:
        raise PermissionError(f"{plan['name']} simulates up to {plan['max_propellers']} propeller(s).")
    env = spec.get("envelope", {})
    if len(env.get("aoas", [])) > plan["max_angles"]:
        raise PermissionError(f"{plan['name']} sweeps up to {plan['max_angles']} angles of attack.")
    if len(env.get("machs") or env.get("speeds") or []) > plan["max_speeds"]:
        raise PermissionError(f"{plan['name']} sweeps up to {plan['max_speeds']} {'Mach numbers' if supersonic else 'airspeeds'}.")
    if len(spec.get("zones", [])) > plan["max_zones"]:
        raise PermissionError(f"{plan['name']} allows {plan['max_zones']} protected regions.")
    if spec.get("kind") == "optimize":
        o = spec.get("optimize", {})
        it = int(o.get("iterations", 0))
        if it < 1:
            raise ValueError("Choose at least one iteration.")
        model_only = supersonic or (spec.get("template") or {}).get("id") in ("uav", "wing")   # no CFD runs in the loop
        limit = plan["max_iterations_panel"] if model_only else plan["max_iterations"]
        if it > limit:
            raise PermissionError(f"{plan['name']} allows up to {limit} designs{' without CFD' if model_only else ''}.")
        if int(o.get("top_k", 1)) > plan["max_top_k"]:
            raise PermissionError(f"{plan['name']} optimizes up to {plan['max_top_k']} parameters at once (the most influential ones).")
        if len(o.get("aspects", [])) > plan["max_aspects"]:
            raise PermissionError(f"{plan['name']} can change up to {plan['max_aspects']} design aspects at once.")
        if supersonic:
            return
        if o.get("objective", "drag") not in plan["objectives"]:
            raise PermissionError("That objective is available on Pro and Team.")
        if o.get("verify") and not plan["verify"]:
            raise PermissionError("High-fidelity verification is available on Pro and Team.")
        if o.get("verify") and o["verify"] not in plan["fidelities"]:
            raise PermissionError(f"{o['verify'].title()} verification needs a higher plan.")
