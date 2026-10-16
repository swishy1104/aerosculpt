"""Mission performance of a winged design: power required against airspeed at its flight weight,
and from it stall speed, the best-endurance and best-range speeds, endurance, range and climb.

Standard electric-aircraft energy method:
* P_req(V) = D(V) V, with D at the lift that holds the weight (wing model + fuselage drag);
* battery power = P_req / eta (propeller x motor x controller efficiency);
* endurance = usable energy / battery power at the minimum-power speed;
* range = V x usable energy / battery power at the speed of minimum drag per speed (D/V... i.e. P/V);
* rate of climb = (eta x motor power - P_req) / weight.
Propeller efficiency is taken as one number; the RC tunnel's propeller model gives it for a
real propeller when needed.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .body import fuselage_drag
from .wing import WingModel

G0 = 9.80665


@dataclasses.dataclass
class MissionInputs:
    battery_wh: float = 100.0         # e.g. 6S 4500 mAh
    usable: float = 0.8               # fraction of the battery used (the rest is reserve)
    efficiency: float = 0.55          # propeller x motor x controller
    motor_w: float = 600.0            # electrical power the motor can draw (for climb)

    @classmethod
    def from_spec(cls, d):
        out = cls()
        for k, lo, hi in (("battery_wh", 1.0, 1e5), ("usable", 0.1, 1.0), ("efficiency", 0.1, 0.95), ("motor_w", 0.0, 1e6)):
            if (d or {}).get(k) is not None:
                try:
                    setattr(out, k, float(np.clip(float(d[k]), lo, hi)))
                except (TypeError, ValueError):
                    pass
        return out


def mission(built, mass_kg, air, inputs: MissionInputs, v_max=None, n=36):
    """built: templates.Built of a winged design. Returns the speed sweep and the summary."""
    W = float(mass_kg) * G0
    ref = built.ref
    surfs = [dataclasses.replace(s, ns=40, nc=6) for s in built.surfaces]
    wm = WingModel(surfs, ref["ref_area"], ref["ref_length"], ref["x_ref"], built.drag_surfaces, nf_size="large")
    rho, mu = air["rho"], air["mu"]
    S = ref["ref_area"]
    main = surfs[0]
    re_mid = rho * 15.0 * main.root_chord / mu
    clmax = 0.9 * wm.cl_max(0, re_mid)                  # 3D maximum lift is about 90 % of the section's
    v_stall = math.sqrt(2 * W / (rho * S * max(clmax, 0.3)))
    v_hi = float(v_max) if v_max else 3.0 * v_stall
    speeds = np.linspace(1.15 * v_stall, max(v_hi, 1.5 * v_stall), n)
    rows = []
    for V in speeds:
        r = wm.evaluate(float(V), rho, mu, lift=W)
        body = fuselage_drag(built.info, float(V), rho, mu, r["alpha"])["drag_N"] if "length" in built.info else 0.0
        D = r["drag_N"] + body
        rows.append({"v": float(V), "drag_N": D, "power_W": D * float(V), "alpha": r["alpha"], "stall": r["stall"],
                     "ld": W / D if D > 0 else None})
    ok = [x for x in rows if not x["stall"]] or rows
    be = min(ok, key=lambda x: x["power_W"])                         # best endurance: least power
    br = min(ok, key=lambda x: x["power_W"] / x["v"])                # best range: least energy per metre
    e_use = inputs.battery_wh * 3600.0 * inputs.usable
    endurance_s = e_use / (be["power_W"] / inputs.efficiency)
    range_m = br["v"] * e_use / (br["power_W"] / inputs.efficiency)
    climb = (inputs.efficiency * inputs.motor_w - be["power_W"]) / W if inputs.motor_w else None
    return {
        "weight_N": W, "mass_kg": float(mass_kg), "stall_speed": v_stall, "cl_max": clmax,
        "best_endurance": {"v": be["v"], "power_W": be["power_W"], "endurance_min": endurance_s / 60.0},
        "best_range": {"v": br["v"], "power_W": br["power_W"], "range_km": range_m / 1000.0,
                       "ld": br["ld"]},
        "climb_m_s": climb, "inputs": dataclasses.asdict(inputs), "curve": rows,
    }
