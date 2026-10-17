"""Rocket methods against their published sources and independent theory (seconds).

  skin friction     Prandtl-Schlichting law (Schlichting, Boundary-Layer Theory)
  base drag         Hoerner's subsonic blunt-base value, 0.13 on the base area
  nose CP           exact slender-body centres: cone 2/3 L, tangent ogive 0.466 L (Barrowman)
  fin lift slope    Helmbold lifting-surface slope for the fin pair as one low-aspect-ratio wing
  integrator        closed-form Fehskens-Malewicki altitude (constant thrust and drag)
  model rocket Cd   Gregorek's wind-tunnel band for typical kits (Estes TR-11, 1970)
  Alpha III apogee  the manufacturer's published figure (optimum conditions): indicative only
"""
from __future__ import annotations

import math
import time

from ..cfd.analysis import isa
from .drag import DragModel, skin_friction
from .flight import G0, Launch, malewicki, simulate
from .geometry import Fin, RocketGeometry
from .model import RocketModel
from .motors import BY_ID, Motor
from .stability import Barrowman


def _row(id_, body, condition, method, quantity, value, ref, source, band=None, seconds=0.0):
    err = (value - ref) / ref
    inside = band is not None and band[0] <= value <= band[1]
    status = "pass" if abs(err) <= 0.10 or inside else ("warn" if abs(err) <= 0.25 else "fail")
    return {"id": id_, "group": "rocket", "body": body, "condition": condition, "method": method, "quantity": quantity,
            "value": value, "cd": value, "ref": ref, "ref_band": band, "ref_source": source, "error": err,
            "status": status, "seconds": seconds}


def rocket_rows(verbose=False):
    t0 = time.time()
    rows = []
    for re in (1e6, 1e7):
        cf, _ = skin_friction(re, 1e-9, 1.0, 0.0)
        rows.append(_row(f"rk-cf-{re:.0e}", "Flat plate, turbulent", f"Re {re:.0e}", "skin friction fit", "Cf", cf,
                         0.455 / math.log10(re) ** 2.58, "Prandtl-Schlichting law (Schlichting, Boundary-Layer Theory)"))
    from .drag import base_drag_coefficient
    rows.append(_row("rk-base-m0.3", "Blunt base", "Mach 0.3", "base drag", "Cd on base area", base_drag_coefficient(0.3), 0.13,
                     "Hoerner, Fluid-Dynamic Drag (1965) ch. XIII: subsonic base drag about 0.13", [0.11, 0.15]))
    # nose centres of pressure: the general volume formula against the exact constants
    for fam, ref, src in (("conical", 2.0 / 3.0, "slender-body theory: 2/3 of the nose length"),
                          ("ogive", 0.466, "Barrowman (1966): 0.466 L for a tangent ogive")):
        g = RocketGeometry(fam, 0.3, 0.1, 1.0, fins=Fin(4, 0.1, 0.05, 0.08, 0.03, 0.003, 0.85))
        cna, x = Barrowman(g).nose()
        rows.append(_row(f"rk-cp-{fam}", f"{fam.capitalize()} nose", "small angle", "Barrowman", "x_cp / L", x / 0.3, ref, src))
    # fin pair lift slope against Helmbold's lifting-surface formula
    fin = Fin(4, 0.204, 0.084, 0.13, 0.109, 0.0072, 0.9)
    g = RocketGeometry("ogive", 0.264, 0.1, 1.2, fins=fin)
    b = Barrowman(g)
    cna_f = b._fin_sub(1.0)                           # without the body interference factor
    pair_area = 2 * fin.area
    ar = (2 * fin.span) ** 2 / pair_area
    helm = 2 * math.pi * ar / (2 + math.sqrt(ar * ar + 4)) * pair_area / g.ref_area
    rows.append(_row("rk-fin-cna", "Four trapezoidal fins", "incompressible", "Barrowman fins", "CN_alpha /rad", cna_f, helm,
                     "Helmbold lifting-surface lift slope of the fin pair as one wing"))
    # the integrator against the closed form: constant thrust, constant mass and Cd, no rail, no wind
    T, m, cd, a_ref, burn = 20.0, 0.1, 0.5, 0.0005, 1.5
    motor = Motor("const", "T", "test", 18, 70, [], 0.0, 0.0, [(0.0, T), (burn, T), (burn + 1e-6, 0.0)])
    air = isa(0.0)
    fl = simulate(lambda mach, re: cd, a_ref, 0.5, m, motor, Launch(rail_m=0.1, wind_m_s=0.0, constant_air=True), None)
    ex = malewicki(T, m, cd, a_ref, air["rho"], burn)
    rows.append(_row("rk-integrator", "Point mass, constant thrust", "no wind, sea level", "RK4 flight", "apogee (m)", fl["apogee_m"],
                     ex["apogee"], "Fehskens-Malewicki closed form (Estes TR-10)"))
    # a typical 4-fin model rocket (Estes Alpha-like, BT-50, 1/8 in lug, painted)
    vals = {"nose_family": "ogive", "nose_length": 0.07, "tip_radius": 0.0, "length": 0.31, "diameter": 0.0248, "tail_length": 0.0,
            "tail_taper": 1.0, "fin_count": "3", "fin_root": 0.064, "fin_tip": 0.025, "fin_span": 0.045, "fin_sweep": 35.0,
            "fin_thickness": 0.0024, "fin_shape": "trapezoid", "fin_position": 0.0}
    rk = {"launch_lug": "1/8in", "finish": "painted", "fin_material": "balsa", "materials": {"nose": "plastic", "body": "cardboard", "fins": "balsa"},
          "masses": {"mode": "dry", "dry_mass_g": 34.0, "dry_cg_mm": 170.0}, "motor": {"id": "estes-c6", "delay": 5.0},
          "launch": {"rail_m": 0.9, "wind_m_s": 0.0}, "parachute": {"enabled": True, "diameter_m": 0.3, "cd": 0.75}}
    mdl = RocketModel(vals, rk, BY_ID["estes-c6"])
    v = 30.0
    re = air["rho"] * v * mdl.g.length / air["mu"]
    cd_typ = mdl.drag.cd(v / air["a"], re)
    rows.append(_row("rk-cd-typical", "Typical 3-fin model rocket with lug", "30 m/s", "drag build-up", "Cd", cd_typ, 0.55,
                     "Gregorek, Estes TR-11 (1970): wind-tunnel Cd of typical kits 0.45-0.75", [0.45, 0.75]))
    fl = mdl.fly()
    rows.append(_row("rk-alpha3-c6", "Estes Alpha III, C6-5 (34 g dry)", "no wind", "flight simulation", "apogee (m)", fl["apogee_m"], 335.0,
                     "Estes catalogue: 1,100 ft (335 m) projected, optimum conditions (indicative)", [250.0, 360.0]))
    for r_ in rows:
        r_["seconds"] = (time.time() - t0) / len(rows)
        if verbose:
            print(f"{r_['id']:22s} {r_['value']:.4f} ref {r_['ref']:.4f} err {r_['error'] * 100:+.1f}% [{r_['status']}]", flush=True)
    return rows
