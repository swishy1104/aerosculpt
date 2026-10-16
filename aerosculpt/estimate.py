"""Run-time estimate of a job before it starts, from the grids it will actually use.

The browser's quick estimate assumes the optimizer explores on the next coarser grid. A body too
small for that grid (under MIN_EXPLORE_CELLS across) explores on the chosen grid instead and
confirms one level up, which costs several times more: this works the grids out from the model's
size exactly as the job will, and prices the runs with the machine's measured throughput.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .cfd.analysis import FIDELITY, seconds_per_run
from .cfd.tunnel import plan_domain
from .plans import EXPLORE_FID

AXIS = {"x": 0, "y": 1, "z": 2}
WARM = 0.55                    # a warm-started run (next design, same grid) costs about this much of a cold one
WING_MODEL_S = 2.4             # one design on the wing model (UAV template, measured)


def _body_dims(metrics, settings):
    """Length (along the nose), width and height in metres from the stored model metrics."""
    from .geometry.mesh import UNIT_SCALE
    units = settings.get("units", "m")
    scale = float(settings.get("unit_scale") or 1.0) if units == "custom" else UNIT_SCALE.get(units, 1.0)
    lo = np.asarray(metrics.get("bbox_min", [0, 0, 0]), float)
    hi = np.asarray(metrics.get("bbox_max", [1, 1, 1]), float)
    ext = (hi - lo) * scale
    f = AXIS[settings.get("forward", "-x")[1]]
    u = AXIS[settings.get("up", "+z")[1]]
    w = 3 - f - u
    return float(ext[f]), float(ext[w]), float(ext[u])


def _mlups():
    from .config import settings
    try:
        d = json.loads((settings.data_dir / "compute.json").read_text())
        return float(d["mlups"]), d.get("backend", "cpu")
    except (OSError, ValueError, KeyError):
        return 40.0, "cpu"


def estimate(spec, metrics, settings_):
    """-> {"seconds", "lines": [str], "explore_on", "body_cells"}"""
    mlups, backend = _mlups()
    env = spec.get("envelope", {})
    n_ang = max(1, len(env.get("aoas", [0])))
    kind = spec.get("kind", "analyze")
    tm = spec.get("template") or {}
    o = spec.get("optimize", {})
    lines = []
    if kind == "optimize" and tm.get("id") in ("uav", "wing"):
        it = int(o.get("iterations", 30))
        s = 12.0 + it * WING_MODEL_S
        lines.append(f"{it} designs on the wing model (~{WING_MODEL_S:g} s each)")
        return {"seconds": s, "lines": lines, "explore_on": "wing model", "body_cells": None}
    L, W, H = _body_dims(metrics, settings_)
    af = 0.8 * W * H
    fid = FIDELITY[spec.get("fidelity", "draft")]
    faces = int(metrics.get("triangles", 20000))
    setup = 5.0 + faces / 25_000.0

    def cells_across(f):
        d = plan_domain(np.zeros(3), np.array([L, W, H]), f.cells, af)
        return min(d.body_cells), d.n_cells

    if kind != "optimize":
        _, n = cells_across(fid)
        per = seconds_per_run(n, mlups)
        s = setup + per * n_ang
        lines.append(f"{n_ang} run(s) on the {fid.name} grid ({n:,} cells, ~{per:.0f} s each)")
        if spec.get("grid_study"):
            g = seconds_per_run(n / 2.2, mlups) + seconds_per_run(n / 4.84, mlups)
            s += g
            lines.append(f"grid check: 2 coarser runs (~{g:.0f} s)")
        return {"seconds": s, "lines": lines, "explore_on": fid.name, "body_cells": None}

    from .cfd.gpu import available
    fx = FIDELITY[EXPLORE_FID.get(fid.name, "coarse")]
    across_x, n_x = cells_across(fx)
    bumped = across_x < 12 and fx.name != fid.name and available()
    explore = fid if bumped else fx
    _, n_e = cells_across(explore)
    up = {"draft": "standard", "standard": "high"}.get(fid.name) if bumped else None
    confirm = FIDELITY[up] if up else fid
    _, n_c = cells_across(confirm)
    it = int(o.get("iterations", 24))
    n_ax = min(3, n_ang)
    # about 1.4 shape controls per whole-body aspect (measured), plus the generated ideas
    n_params = round(1.4 * max(1, len(o.get("aspects", [])) or 2)) + (3 if o.get("auto_ideas") else 0)
    # per design beyond the CFD: building and checking the shape; above Mach 0.6 the supersonic
    # anchor of the transonic estimate
    mach = max(env.get("speeds") or [0.0]) / 340.0
    over = 1.5 + faces / 40_000.0 + (3.0 if mach > 0.6 else 0.0)
    per_x = 0.8 * seconds_per_run(n_e, mlups) + over
    per_c = 0.8 * seconds_per_run(n_c, mlups) + over
    runs_c = 3 * n_ang + 6
    # the budget counts every screened design, paired ones included; the search often stops early
    hi_runs = (it + 3) * n_ax
    lo_runs = min(it + 3, 3 + 2 * n_params) * n_ax
    rounds = 1.3 if o.get("auto_ideas") else 1.0
    s_lo = 3 * setup + lo_runs * per_x + runs_c * per_c
    s_hi = 3 * setup + (hi_runs * per_x + runs_c * per_c) * rounds
    lines.append(f"explore on the {explore.name} grid ({n_e:,} cells, ~{per_x:.0f} s per run): {lo_runs}-{hi_runs} runs"
                 + (" — the body is too small for the coarser grid" if bumped else ""))
    lines.append(f"confirm on the {confirm.name} grid: {runs_c} runs (~{per_c:.0f} s each)")
    return {"seconds": s_hi, "seconds_low": s_lo, "lines": lines, "explore_on": explore.name,
            "body_cells": round(across_x, 1), "backend": backend}


_ = math
