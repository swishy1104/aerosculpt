"""Rocket geometry for the component methods: a body of revolution (nose, tube, boat tail), a
fin set and a launch lug, in metres with the nose tip at x = 0 (the template's frame)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geometry.templates import nose_profile, tail_profile

# surface finish -> equivalent sand roughness height (m), OpenRocket's table
FINISHES = {
    "rough": {"label": "Rough (unfinished, uneven)", "roughness": 500e-6},
    "unfinished": {"label": "Unfinished (bare paper or wood)", "roughness": 150e-6},
    "painted": {"label": "Painted", "roughness": 60e-6},
    "smooth": {"label": "Smooth paint, sanded", "roughness": 20e-6},
    "polished": {"label": "Polished", "roughness": 2e-6},
}
LAUNCH_LUGS = {          # outer diameter, length (m)
    "none": {"label": "None (rail buttons)", "d": 0.0, "length": 0.0},
    "1/8in": {"label": "1/8 in lug", "d": 0.0045, "length": 0.05},
    "3/16in": {"label": "3/16 in lug", "d": 0.0063, "length": 0.05},
    "1/4in": {"label": "1/4 in lug", "d": 0.0078, "length": 0.06},
    "buttons": {"label": "Rail buttons (two)", "d": 0.011, "length": 0.012},
}
FIN_SHAPES = [("trapezoid", "Trapezoidal / swept"), ("elliptical", "Elliptical")]
FIN_EDGES = [("square", "Square (as cut)"), ("rounded", "Rounded"), ("airfoil", "Airfoiled / tapered")]


@dataclass
class Fin:
    count: int
    root: float                 # root chord (m)
    tip: float                  # tip chord (m); elliptical: equivalent trapezoid tip
    span: float                 # exposed semi-span (m)
    sweep: float                # leading-edge sweep distance (m): root LE to tip LE along x
    thickness: float
    x_le: float                 # root leading edge from the nose tip (m)
    shape: str = "trapezoid"
    edge: str = "square"        # leading / trailing edge profile

    @property
    def area(self):
        """Planform area of one fin."""
        if self.shape == "elliptical":
            return math.pi * self.root * self.span / 4.0
        return 0.5 * self.span * (self.root + self.tip)

    @property
    def equivalent(self):
        """(root, tip, sweep) of the trapezoid with the same area and span: Barrowman's and the
        flutter formulas are written for trapezoids."""
        if self.shape == "elliptical":
            ct = (math.pi / 2.0 - 1.0) * self.root
            return self.root, ct, 0.5 * (self.root - ct)
        return self.root, self.tip, self.sweep

    @property
    def mac(self):
        cr, ct, _ = self.equivalent
        if cr + ct < 1e-12:
            return 0.0
        return (2.0 / 3.0) * (cr + ct - cr * ct / (cr + ct))

    @property
    def x_te(self):
        return self.x_le + self.root

    @property
    def aspect_ratio(self):
        """Of the fin pair as one wing: (2s)^2 / (2 A)."""
        return 2.0 * self.span * self.span / max(self.area, 1e-12)

    @property
    def centroid_x(self):
        cr, ct, xr = self.equivalent
        if cr + ct < 1e-12:
            return self.x_le
        return self.x_le + xr * (cr + 2 * ct) / (3 * (cr + ct)) + (cr * cr + ct * ct + cr * ct) / (3 * (cr + ct))

    def chord_at(self, eta):
        """Chord at a fraction eta of the span (0 root, 1 tip)."""
        if self.shape == "elliptical":
            return self.root * math.sqrt(max(0.0, 1.0 - eta * eta))
        return self.root + (self.tip - self.root) * eta


@dataclass
class RocketGeometry:
    nose_family: str
    nose_length: float
    diameter: float
    length: float               # nose tip to the aft end of the body (m)
    tail_length: float = 0.0
    base_diameter: float = None  # None: no boat tail
    tip_radius: float = 0.0
    nose_power: float = 0.6
    haack_c: float = 0.0
    fins: Fin = None
    lug: str = "none"
    finish: str = "painted"
    extra_wetted: float = 0.0   # launch lug and similar (m^2)
    notes: list = field(default_factory=list)

    @property
    def radius(self):
        return 0.5 * self.diameter

    @property
    def ref_area(self):
        return math.pi * self.radius ** 2

    @property
    def base_radius(self):
        return 0.5 * (self.base_diameter if self.base_diameter else self.diameter)

    @property
    def roughness(self):
        return FINISHES.get(self.finish, FINISHES["painted"])["roughness"]

    @property
    def fineness(self):
        return self.length / max(self.diameter, 1e-9)

    def profile(self, n_nose=60):
        """(x, r) of the body of revolution from the tip to the base."""
        R = self.radius
        xs, rs, _ = nose_profile(self.nose_family, self.nose_length, R, self.tip_radius, n_nose, self.nose_power, self.haack_c)
        x_tail = self.length - self.tail_length
        X = [xs, np.array([x_tail])]
        Rr = [rs, np.array([R])]
        if self.tail_length > 1e-9 and self.base_radius < R - 1e-9:
            xt, rt = tail_profile(x_tail, self.tail_length, R, self.base_radius, 24)
            X.append(xt[1:])
            Rr.append(rt[1:])
        else:
            X.append(np.array([self.length]))
            Rr.append(np.array([R]))
        x = np.concatenate(X)
        r = np.concatenate(Rr)
        keep = np.concatenate([[True], np.diff(x) > 1e-12])
        return x[keep], r[keep]

    def nose_volume(self):
        x, r = self.profile()
        m = x <= self.nose_length + 1e-12
        return float(np.trapezoid(math.pi * r[m] ** 2, x[m]))

    def body_wetted(self):
        x, r = self.profile()
        ds = np.hypot(np.diff(x), np.diff(r))
        return float(np.sum(math.pi * (r[1:] + r[:-1]) * ds))

    def nose_wetted(self):
        x, r = self.profile()
        m = x <= self.nose_length + 1e-12
        xm, rm = x[m], r[m]
        ds = np.hypot(np.diff(xm), np.diff(rm))
        return float(np.sum(math.pi * (rm[1:] + rm[:-1]) * ds))

    @classmethod
    def from_template(cls, values, rocket=None):
        """From the rocket template's values (metres) plus the rocket settings (finish, lug, fin edge)."""
        rocket = rocket or {}
        v = values
        L, D = float(v["length"]), float(v["diameter"])
        Lt = float(v.get("tail_length", 0.0))
        taper = float(v.get("tail_taper", 1.0))
        base_d = D * taper if (Lt > 1e-9 and taper < 1.0 - 1e-9) else None
        if base_d is None:
            Lt = 0.0
        cr, ct, s, tf = (float(v[k]) for k in ("fin_root", "fin_tip", "fin_span", "fin_thickness"))
        sweep = s * math.tan(math.radians(float(v.get("fin_sweep", 0.0))))
        shape = str(v.get("fin_shape", "trapezoid"))
        x_te = L - Lt - float(v.get("fin_position", 0.0))
        fins = Fin(int(v.get("fin_count", 4)), cr, ct if shape == "trapezoid" else 0.0, s, sweep if shape == "trapezoid" else 0.0,
                   tf, x_te - cr, shape, str(rocket.get("fin_edge", "square")))
        lug = rocket.get("launch_lug", "none")
        g = cls(str(v["nose_family"]), float(v["nose_length"]), D, L, Lt, base_d, float(v.get("tip_radius", 0.0)),
                float(v.get("nose_power", 0.6)), float(v.get("haack_c", 0.0)), fins,
                lug if lug in LAUNCH_LUGS else "none", rocket.get("finish", "painted") if rocket.get("finish") in FINISHES else "painted")
        lg = LAUNCH_LUGS[g.lug]
        g.extra_wetted = math.pi * lg["d"] * lg["length"]
        return g

    def summary(self):
        f = self.fins
        return {"length": self.length, "diameter": self.diameter, "ref_area": self.ref_area, "fineness": self.fineness,
                "nose_fineness": self.nose_length / self.diameter, "base_diameter": 2 * self.base_radius,
                "body_wetted": self.body_wetted(), "fin_area_each": f.area if f else 0.0,
                "fin_count": f.count if f else 0, "fin_aspect_ratio": f.aspect_ratio if f else 0.0,
                "fin_mac": f.mac if f else 0.0, "finish": self.finish, "launch_lug": self.lug}
