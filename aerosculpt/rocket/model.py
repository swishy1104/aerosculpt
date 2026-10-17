"""The rocket model the workspace, optimizer and report use: geometry + masses + motor + launch
site -> drag curve, stability through the flight, fin flutter and the flight itself.

Masses come from the project's rocket settings: either estimated from the geometry and typical
materials (so a longer nose or thicker fins weigh more, which keeps the optimizer honest) or
entered part by part / as a measured dry mass and centre of gravity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..cfd.analysis import isa
from .drag import DragModel
from .flight import Launch, Parachute, air_at, simulate
from .flutter import FIN_MATERIALS, flutter_check
from .geometry import FINISHES, LAUNCH_LUGS, RocketGeometry
from .motors import BY_ID, Motor, motor_for_diameter
from .stability import Barrowman, centre_of_gravity, rate, static_margin

DEFAULT_ROCKET = {
    "masses": {"mode": "estimate", "dry_mass_g": None, "dry_cg_mm": None, "items": []},
    "materials": {"nose": "plastic", "body": "cardboard", "fins": "plywood"},
    "motor": {"id": None, "delay": None},
    "launch": {"rail_m": 1.2, "angle_deg": 0.0, "wind_m_s": 2.0, "elevation_m": 0.0, "temperature_c": None},
    "parachute": {"enabled": True, "diameter_m": None, "cd": 0.75},
    "finish": "painted", "launch_lug": "none", "fin_edge": "square", "fin_material": "plywood",
    "min_margin": 1.0, "flutter_margin": 1.2, "min_rail_speed": 15.0,
}
# typical construction, for the estimate: (density kg/m^3 or kg/m^2, wall or thickness m)
MATERIALS = {
    "nose": {"plastic": {"label": "Plastic (moulded)", "shell": 0.0025, "density": 1050.0},
             "balsa": {"label": "Balsa (solid)", "shell": None, "density": 160.0},
             "fibreglass": {"label": "Fibreglass", "shell": 0.0015, "density": 1800.0},
             "pla": {"label": "3D-printed PLA (shell)", "shell": 0.0012, "density": 1100.0}},
    "body": {"cardboard": {"label": "Kraft paper tube", "areal": 0.7},
             "thin_paper": {"label": "Thin paper tube (Estes BT)", "areal": 0.45},
             "phenolic": {"label": "Phenolic tube", "areal": 1.0},
             "fibreglass": {"label": "Fibreglass tube", "areal": 1.6},
             "carbon": {"label": "Carbon tube", "areal": 1.3},
             "blue_tube": {"label": "Blue Tube", "areal": 1.1}},
}
MOTOR_DIAMETERS = [13, 18, 24, 29, 38, 54, 75]


def estimate_masses(g: RocketGeometry, materials, motor: Motor):
    """Part masses (kg) and their positions from the nose tip for typical materials."""
    mats = {**DEFAULT_ROCKET["materials"], **(materials or {})}
    nose_m = MATERIALS["nose"].get(mats.get("nose"), MATERIALS["nose"]["plastic"])
    body_m = MATERIALS["body"].get(mats.get("body"), MATERIALS["body"]["cardboard"])
    fin_m = FIN_MATERIALS.get(mats.get("fins"), FIN_MATERIALS["plywood"])
    items = []
    if nose_m["shell"] is None:
        m_nose = nose_m["density"] * g.nose_volume()
        x_nose = 0.6 * g.nose_length
    else:
        m_nose = nose_m["density"] * nose_m["shell"] * g.nose_wetted() + 0.004 * g.diameter / 0.05
        x_nose = 0.5 * g.nose_length
    items.append(("Nose cone", m_nose, x_nose))
    tube_len = g.length - g.nose_length
    items.append(("Body tube", body_m["areal"] * math.pi * g.diameter * tube_len, g.nose_length + 0.5 * tube_len))
    f = g.fins
    if f:
        items.append(("Fins", fin_m["density"] * f.thickness * f.area * f.count, f.centroid_x))
    # motor mount, centering rings and engine hook: grows with the motor
    md = motor.diameter_mm / 1000.0
    mount = 0.005 * (motor.diameter_mm / 18.0) ** 2 + 0.3 * math.pi * md * (motor.length_mm / 1000.0)
    items.append(("Motor mount", mount, g.length - 0.5 * motor.length_mm / 1000.0))
    # recovery: parachute, shock cord, wadding; a launch lug
    items.append(("Recovery", max(0.006, 0.4 * items[1][1]), g.nose_length + 0.2 * tube_len))
    lg = LAUNCH_LUGS.get(g.lug, LAUNCH_LUGS["none"])
    if lg["d"] > 0:
        items.append(("Launch lug", 0.002 * lg["length"] / 0.05, g.nose_length + 0.5 * tube_len))
    return items


class RocketModel:
    def __init__(self, template_values, rocket=None, motor: Motor = None):
        self.cfg = _merge(DEFAULT_ROCKET, rocket or {})
        self.values = template_values
        self.g = RocketGeometry.from_template(template_values, self.cfg)
        self.motor = motor or self._default_motor()
        self.drag = DragModel(self.g)
        self.barrowman = Barrowman(self.g)
        self.launch = Launch(**{k: self.cfg["launch"].get(k) for k in ("rail_m", "angle_deg", "wind_m_s", "elevation_m", "temperature_c")},
                             delay_s=self.cfg["motor"].get("delay"))
        for k in ("rail_m", "angle_deg", "wind_m_s", "elevation_m"):
            setattr(self.launch, k, float(getattr(self.launch, k) or 0.0))
        self.launch.rail_m = max(self.launch.rail_m, 0.1)
        pc = self.cfg["parachute"]
        self.chute = Parachute(float(pc.get("diameter_m") or self._default_chute()), float(pc.get("cd") or 0.75)) if pc.get("enabled", True) else None
        self.items = self._masses()
        self.dry_mass, self.dry_cg = centre_of_gravity(self.items)

    # ---- inputs -------------------------------------------------------------------------
    def _default_motor(self):
        mid = (self.cfg.get("motor") or {}).get("id")
        return BY_ID.get(mid) or motor_for_diameter(self.g.diameter)

    def _default_chute(self):
        # a descent of about 4-5 m/s for the dry mass: d = sqrt(8 m g / (pi rho cd v^2))
        m = max(self.dry_mass if hasattr(self, "dry_mass") else 0.05, 0.02) + self.motor.case_mass_kg
        return math.sqrt(8 * m * 9.81 / (math.pi * 1.225 * 0.75 * 4.5 ** 2))

    def _masses(self):
        mc = self.cfg["masses"]
        mode = mc.get("mode", "estimate")
        if mode == "dry" and mc.get("dry_mass_g"):
            cg = mc.get("dry_cg_mm")
            x = float(cg) / 1000.0 if cg else 0.55 * self.g.length
            return [("Rocket without motor", float(mc["dry_mass_g"]) / 1000.0, x)]
        if mode == "items" and mc.get("items"):
            return [(str(i.get("name", "part"))[:40], float(i.get("g", 0)) / 1000.0, float(i.get("x_mm", 0)) / 1000.0) for i in mc["items"]]
        return estimate_masses(self.g, self.cfg.get("materials"), self.motor)

    def mass_at(self, t):
        return self.dry_mass + self.motor.mass(t)

    def cg_at(self, t):
        m = self.motor.mass(t)
        x_motor = self.g.length - 0.5 * self.motor.length_mm / 1000.0
        return (self.dry_mass * self.dry_cg + m * x_motor) / (self.dry_mass + m)

    # ---- outputs ------------------------------------------------------------------------
    def stability(self, mach, t):
        cp = self.barrowman.cp(mach)
        cg = self.cg_at(t)
        sm = static_margin(cp["x_cp"], cg, self.g.diameter)
        return {"mach": mach, "t": t, "x_cp": cp["x_cp"], "x_cg": cg, "cn_alpha": cp["cn_alpha"], "margin": sm,
                "rating": rate(sm), "mass": self.mass_at(t), "parts": cp["parts"]}

    def cd_fn(self):
        """Cd(Mach) interpolated from a table at the site's air: the flight integrator calls it
        thousands of times. The Reynolds number is taken at each Mach number's sea-level speed
        (its effect on Cf is a few per cent over a flight)."""
        if getattr(self, "_cd_table", None) is None:
            import numpy as np
            air = air_at(0.0, self.launch)
            machs = np.concatenate([np.linspace(0.02, 0.8, 27), np.linspace(0.82, 1.4, 30), np.linspace(1.5, 6.0, 19)])
            cds = np.array([self.drag.cd(m, air["rho"] * m * air["a"] * self.g.length / air["mu"]) for m in machs])
            self._cd_table = (machs, cds)
        machs, cds = self._cd_table
        import numpy as np
        return lambda mach, re: float(np.interp(mach, machs, cds))

    def fly(self):
        return simulate(self.cd_fn(), self.g.ref_area, self.g.length, self.dry_mass, self.motor, self.launch, self.chute)

    def evaluate(self, with_trajectory=True):
        fl = self.fly()
        m_exit = fl["rail_exit_speed"] / air_at(0.0, self.launch)["a"]
        st_liftoff = self.stability(max(m_exit, 0.05), 0.0)
        t_v = next((p["t"] for p in fl["trajectory"] if abs(p["v"] - fl["max_speed"]) < 1e-6), fl["burnout"]["t"] if fl["burnout"] else 0.0)
        st_max = self.stability(max(fl["max_mach"], 0.05), t_v)
        st_burnout = self.stability(max(fl["burnout"]["v"] / air_at(fl["burnout"]["z"], self.launch)["a"], 0.05) if fl["burnout"] else 0.05,
                                    self.motor.burn_time)
        fin = self.g.fins
        flut = flutter_check(fin, self.cfg.get("fin_material", "plywood"), fl["max_speed"], fl["air_at_max_speed"],
                             float(self.cfg.get("flutter_margin", 1.2))) if fin else None
        air0 = air_at(0.0, self.launch)
        machs = sorted({0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0}
                       | {round(min(max(fl["max_mach"], 0.05), 4.0), 3)})
        machs = [m for m in machs if m <= max(1.2, 1.3 * fl["max_mach"], 0.9)]
        curve = self.drag.curve(machs, air0)
        checks = self.checks(fl, st_liftoff, st_max, flut)
        out = {
            "geometry": self.g.summary(), "motor": self.motor.public(),
            "masses": {"items": [{"name": n, "g": round(m * 1000, 2), "x_mm": round(x * 1000, 1)} for n, m, x in self.items],
                       "dry_mass_g": round(self.dry_mass * 1000, 2), "dry_cg_mm": round(self.dry_cg * 1000, 1),
                       "liftoff_mass_g": round(fl["liftoff_mass"] * 1000, 2), "mode": self.cfg["masses"].get("mode", "estimate")},
            "stability": {"liftoff": st_liftoff, "max_speed": st_max, "burnout": st_burnout,
                          "cp_vs_mach": [{"mach": m, "x_cp": self.barrowman.cp(m)["x_cp"], "cn_alpha": self.barrowman.cp(m)["cn_alpha"]}
                                         for m in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0) if m <= max(0.5, 1.2 * fl["max_mach"])]},
            "flutter": flut, "drag_curve": curve,
            "flight": {k: v for k, v in fl.items() if k not in ("trajectory", "air_at_max_speed")} | ({"trajectory": fl["trajectory"]} if with_trajectory else {}),
            "launch": {"rail_m": self.launch.rail_m, "angle_deg": self.launch.angle_deg, "wind_m_s": self.launch.wind_m_s,
                       "elevation_m": self.launch.elevation_m, "temperature_c": self.launch.temperature_c, "delay_s": self.launch.delay_s},
            "parachute": {"diameter_m": self.chute.diameter_m, "cd": self.chute.cd} if self.chute else None,
            "checks": checks, "ok": all(c["ok"] for c in checks),
            "silhouette": _silhouette(self.g),
            "settings": {"finish": self.g.finish, "finish_label": FINISHES[self.g.finish]["label"], "launch_lug": self.g.lug,
                         "fin_material": self.cfg.get("fin_material"), "fin_edge": self.g.fins.edge if self.g.fins else None,
                         "min_margin": float(self.cfg.get("min_margin", 1.0))},
        }
        return out

    def checks(self, fl, st_liftoff, st_max, flut):
        mm = float(self.cfg.get("min_margin", 1.0))
        vr = float(self.cfg.get("min_rail_speed", 15.0))
        out = [
            {"id": "margin_liftoff", "label": "Static margin at liftoff", "value": st_liftoff["margin"], "limit": mm, "unit": "cal",
             "ok": st_liftoff["margin"] >= mm, "why": f"{st_liftoff['margin']:.2f} calibres at liftoff; at least {mm:g} needed"},
            {"id": "margin_max", "label": "Static margin at top speed", "value": st_max["margin"], "limit": 0.5, "unit": "cal",
             "ok": st_max["margin"] >= 0.5, "why": f"{st_max['margin']:.2f} calibres at Mach {st_max['mach']:.2f}"},
            {"id": "rail_speed", "label": "Speed leaving the rail", "value": fl["rail_exit_speed"], "limit": vr, "unit": "m/s",
             "ok": fl["left_rail"] and fl["rail_exit_speed"] >= vr,
             "why": (f"{fl['rail_exit_speed']:.1f} m/s off a {self.launch.rail_m:g} m rail; {vr:g} m/s needed for the fins to steer"
                     if fl["left_rail"] else "the motor cannot lift the rocket off the rail")},
            {"id": "thrust_weight", "label": "Thrust to weight", "value": fl["thrust_to_weight"], "limit": 5.0, "unit": "",
             "ok": fl["thrust_to_weight"] >= 5.0, "why": f"average thrust {fl['thrust_to_weight']:.1f} x the liftoff weight (5 or more recommended)"},
        ]
        if flut:
            out.append({"id": "flutter", "label": "Fin flutter", "value": flut["ratio"], "limit": flut["margin_required"], "unit": "x",
                        "ok": flut["ok"], "why": f"flutter at {flut['flutter_speed']:.0f} m/s against a top speed of {flut['v_max']:.0f} m/s ({flut['material_label']})"})
        if st_liftoff["margin"] > 4.0:
            out.append({"id": "overstable", "label": "Weathercocking", "value": st_liftoff["margin"], "limit": 4.0, "unit": "cal",
                        "ok": True, "warn": True, "why": "very stable: it will turn hard into any wind and lose altitude"})
        if fl.get("deploy_speed") is not None and fl["deploy_speed"] > 25.0:
            out.append({"id": "deploy", "label": "Deployment speed", "value": fl["deploy_speed"], "limit": 25.0, "unit": "m/s",
                        "ok": True, "warn": True, "why": f"the parachute opens at {fl['deploy_speed']:.0f} m/s (hard on the shock cord); the optimal delay is {fl['optimal_delay']:.1f} s"})
        return out

    # ---- the optimizer's quick checks ------------------------------------------------------
    def quick(self):
        """Apogee, margins and flutter without the trajectory: milliseconds per design."""
        fl = simulate(self.cd_fn(), self.g.ref_area, self.g.length, self.dry_mass, self.motor, self.launch, None)
        m_exit = fl["rail_exit_speed"] / air_at(0.0, self.launch)["a"]
        st0 = self.stability(max(m_exit, 0.05), 0.0)
        st1 = self.stability(max(fl["max_mach"], 0.05), fl["burnout"]["t"] if fl["burnout"] else 0.0)
        fin = self.g.fins
        flut = flutter_check(fin, self.cfg.get("fin_material", "plywood"), fl["max_speed"], fl["air_at_max_speed"],
                             float(self.cfg.get("flutter_margin", 1.2))) if fin else None
        speeds = self.cfg.get("envelope_speeds") or [fl["max_speed"] * 0.5, fl["max_speed"]]
        air = isa(0.0)
        cd_env = float(sum(self.drag.cd(max(v / air["a"], 0.05), air["rho"] * v * self.g.length / air["mu"]) for v in speeds) / len(speeds))
        return {"apogee_m": fl["apogee_m"], "max_speed": fl["max_speed"], "max_mach": fl["max_mach"], "cd_envelope": cd_env,
                "rail_exit_speed": fl["rail_exit_speed"], "left_rail": fl["left_rail"],
                "margin_liftoff": st0["margin"], "margin_max": st1["margin"], "flutter_ratio": flut["ratio"] if flut else None,
                "flutter_ok": flut["ok"] if flut else True, "dry_mass": self.dry_mass, "liftoff_mass": fl["liftoff_mass"],
                "cd_at_max": self.drag.cd(max(fl["max_mach"], 0.05), 1e6), "thrust_to_weight": fl["thrust_to_weight"]}

    def guard(self, fast=False):
        """Why this design must not be accepted, or None. fast: the static margin at liftoff
        and flutter at the envelope's top speed only (milliseconds, no flight)."""
        mm = float(self.cfg.get("min_margin", 1.0))
        if fast:
            st = self.stability(0.3, 0.0)
            if st["margin"] < mm:
                return f"static margin {st['margin']:.2f} calibres at liftoff is under the {mm:g} required"
            v = float(self.cfg.get("check_speed") or 0.0)
            if v > 0 and self.g.fins:
                fl = flutter_check(self.g.fins, self.cfg.get("fin_material", "plywood"), v, isa(0.0), float(self.cfg.get("flutter_margin", 1.2)))
                if not fl["ok"]:
                    return f"fins would flutter at {fl['flutter_speed']:.0f} m/s (top speed {v:.0f} m/s)"
            return None
        q = self.quick()
        if not q["left_rail"]:
            return "the motor cannot lift it off the rail"
        if q["margin_liftoff"] < mm:
            return f"static margin {q['margin_liftoff']:.2f} calibres at liftoff is under the {mm:g} required"
        if q["margin_max"] < 0.5:
            return f"static margin {q['margin_max']:.2f} calibres at top speed"
        if not q["flutter_ok"]:
            return f"fins would flutter (flutter speed {q['flutter_ratio'] * q['max_speed']:.0f} m/s against {q['max_speed']:.0f} m/s)"
        if q["rail_exit_speed"] < float(self.cfg.get("min_rail_speed", 15.0)):
            return f"only {q['rail_exit_speed']:.1f} m/s leaving the rail"
        return None


def _silhouette(g):
    """Side view for the stability diagram: body (x, r) and one fin's outline."""
    x, r = g.profile(48)
    body = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(x, r)]
    fin = None
    f = g.fins
    if f:
        R = g.radius
        if f.shape == "elliptical":
            th = [k * math.pi / 16 for k in range(17)]
            fin = [[round(f.x_le + 0.5 * f.root * (1 - math.cos(t)), 4), round(R + f.span * math.sin(t), 4)] for t in th]
        else:
            fin = [[round(f.x_le, 4), round(R, 4)], [round(f.x_le + f.sweep, 4), round(R + f.span, 4)],
                   [round(f.x_le + f.sweep + f.tip, 4), round(R + f.span, 4)], [round(f.x_te, 4), round(R, 4)]]
    return {"body": body, "fin": fin, "length": g.length, "radius": g.radius}


def _merge(base, over):
    out = {}
    for k, v in base.items():
        o = over.get(k) if isinstance(over, dict) else None
        if isinstance(v, dict) and isinstance(o, dict):
            out[k] = _merge(v, o)
        else:
            out[k] = o if o is not None else (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v))
    for k, v in (over or {}).items():
        if k not in out:
            out[k] = v
    return out


def catalog():
    """What the UI offers."""
    from .motors import BUILTIN
    return {"finishes": {k: v["label"] for k, v in FINISHES.items()}, "launch_lugs": {k: v["label"] for k, v in LAUNCH_LUGS.items()},
            "fin_materials": {k: v["label"] for k, v in FIN_MATERIALS.items()},
            "materials": {grp: {k: v["label"] for k, v in d.items()} for grp, d in MATERIALS.items()},
            "motors": [m.public() | {"curve": None} for m in BUILTIN], "defaults": DEFAULT_ROCKET}
