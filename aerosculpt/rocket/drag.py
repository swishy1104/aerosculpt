"""Rocket drag by component build-up: the method behind OpenRocket, RASAero and the classic
hand calculations, from Hoerner (Fluid-Dynamic Drag, 1965), Barrowman (1967) and the OpenRocket
technical documentation (Niskanen 2013, section 3.4). Every coefficient is on the body's
cross-section area.

  skin friction   Prandtl-Schlichting flat plate, limited by the surface roughness of the finish
                  (OpenRocket's fit), compressibility-corrected; body form factor 1 + 1/(2 f),
                  fins 1 + 2 t/c; wetted areas of body, fins (both sides) and launch lug
  nose pressure   subsonic: Hoerner's cone data, 0.8 sin^2(phi), scaled down for rounded profiles;
                  supersonic (M >= 1.3): tangent-cone / Prandtl-Meyer integration along the actual
                  profile (the same Taylor-Maccoll tables the supersonic tunnel uses), blunt tips
                  by modified Newtonian; transonic by cubic interpolation between M 0.8 and 1.3
  boat tail       subsonic: the base drag it removes, discounted by its slenderness (Hoerner /
                  OpenRocket); supersonic: from the profile integration (expansion)
  base            Cd_b = 0.12 + 0.13 M^2 below Mach 1, 0.25 / M above, on the base area
  fins            leading-edge drag by edge profile and sweep (Hoerner's cylinder / plate data),
                  trailing-edge base drag for square edges
  launch lug      Cd 1.1 on its frontal area (a stubby cylinder in the body's boundary layer)
  angle of attack body cross-flow and fin induced drag, approximately CN_alpha alpha^2

Measured against the sources on the validation page; the absolute Cd of a typical model rocket
comes out in the 0.4-0.7 band of Gregorek's wind-tunnel data (Estes TR-11).
"""
from __future__ import annotations

import math

import numpy as np

from ..hyper.gasdyn import base_cp, cone_table, cp_max, expansion_cp
from .geometry import LAUNCH_LUGS, RocketGeometry

M_SUB, M_SUP = 0.8, 1.3        # transonic bridge for the pressure terms
# subsonic nose pressure drag relative to a cone of the same fineness (Hoerner ch. III: rounded
# profiles recover most of the pressure a cone loses)
NOSE_SUBSONIC_FACTOR = {"conical": 1.0, "ogive": 0.12, "elliptical": 0.30, "haack": 0.06}


def skin_friction(re, roughness, length, mach):
    """Average skin-friction coefficient with roughness limit and compressibility
    (OpenRocket 3.4.1). Returns (Cf, roughness_limited)."""
    re = max(float(re), 10.0)
    if re < 1e4:
        cf = 1.48e-2
    else:
        cf = 1.0 / (1.50 * math.log(re) - 5.6) ** 2
    rough = False
    rel = max(roughness, 1e-9) / max(length, 1e-9)
    re_crit = 51.0 * rel ** -1.039
    if re > re_crit:
        cf_r = 0.032 * rel ** 0.2
        if cf_r > cf:
            cf, rough = cf_r, True
    m = float(mach)
    if rough:
        cf /= 1.0 + 0.18 * m * m
    elif m < 1.0:
        cf *= 1.0 - 0.1 * m * m
    else:
        cf /= (1.0 + 0.15 * m * m) ** 0.58
    return cf, rough


def wave_drag_profile(x, r, mach, a_ref, x_split=None):
    """Supersonic pressure drag of a body of revolution by tangent-cone (compression) and
    Prandtl-Meyer expansion (where the profile turns inward), on a_ref. Returns the total and
    the part forward of x_split (the nose)."""
    M = float(mach)
    th_c, cps, _, th_max = cone_table(M)
    dx = np.diff(x)
    dr = np.diff(r)
    theta = np.arctan2(dr, np.maximum(dx, 1e-12))
    cp = np.zeros_like(theta)
    comp = theta > 1e-9
    # tangent cone up to the largest attached cone angle; beyond it the shock detaches and the
    # surface is near a blunt stagnation region: modified Newtonian
    tc = np.interp(theta[comp], th_c, cps)
    newt = cp_max(M) * np.sin(theta[comp]) ** 2
    cp[comp] = np.where(theta[comp] <= th_max, tc, np.maximum(newt, tc))
    exp = theta < -1e-9
    cp[exp] = expansion_cp(M, -theta[exp])
    dA = math.pi * (r[1:] ** 2 - r[:-1] ** 2)
    contrib = cp * dA / a_ref
    total = float(np.sum(contrib))
    if x_split is None:
        return total, total
    nose = float(np.sum(contrib[x[1:] <= x_split + 1e-12]))
    return total, nose


def _hermite(m, m0, m1, y0, y1, d0=0.0, d1=0.0):
    t = (m - m0) / (m1 - m0)
    h = m1 - m0
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y0 + (t ** 3 - 2 * t ** 2 + t) * h * d0
            + (-2 * t ** 3 + 3 * t ** 2) * y1 + (t ** 3 - t ** 2) * h * d1)


def base_drag_coefficient(mach):
    """On the base area (OpenRocket 3.4.3; Fleeman's -0.25/M above Mach 1)."""
    m = float(mach)
    if m < 1.0:
        return 0.12 + 0.13 * m * m
    return -base_cp(m)


def fin_edge_drag(mach, edge, sweep_rad):
    """Leading-edge pressure drag coefficient on the fins' frontal (thickness x span) area."""
    m = float(mach)
    if edge == "airfoil":
        return 0.0
    if m < 0.9:
        cd = (1.0 - m * m) ** -0.417 - 1.0 if edge == "square" else 0.0
    elif m < 1.0:
        cd = 1.0 - 1.785 * (m - 0.9)
    else:
        cd = 1.214 - 0.502 / (m * m) + 0.1095 / m ** 4
    return cd * math.cos(sweep_rad) ** 2


class DragModel:
    """Cd(Mach, Reynolds, angle) of a RocketGeometry, with the breakdown."""

    def __init__(self, g: RocketGeometry):
        self.g = g
        self.x, self.r = g.profile()
        self.a_ref = g.ref_area
        self.body_wetted = g.body_wetted()
        f = g.fins
        self.fin_wetted = 2.0 * f.area * f.count if f else 0.0
        self.sweep_rad = math.atan2(f.sweep, f.span) if f and f.span > 0 else 0.0
        self._sup_cache = {}

    # ---- pieces -------------------------------------------------------------------------
    def friction(self, mach, re):
        g = self.g
        cf, rough = skin_friction(re, g.roughness, g.length, mach)
        ff_body = 1.0 + 1.0 / (2.0 * max(g.fineness, 0.5))
        f = g.fins
        tc = f.thickness / max(f.mac, 1e-9) if f else 0.0
        ff_fin = 1.0 + 2.0 * tc
        body = cf * ff_body * self.body_wetted / self.a_ref
        fins = cf * ff_fin * self.fin_wetted / self.a_ref
        lug = cf * 1.2 * g.extra_wetted / self.a_ref
        return {"cd_friction_body": body, "cd_friction_fins": fins, "cd_friction_lug": lug, "cf": cf, "rough_limited": rough}

    def _nose_subsonic(self):
        g = self.g
        phi = math.atan2(g.radius, max(g.nose_length, 1e-9))
        cone = 0.8 * math.sin(phi) ** 2
        fam = g.nose_family
        if fam == "power":
            k = min(1.0, max(0.05, g.nose_power ** 2))
        else:
            k = NOSE_SUBSONIC_FACTOR.get(fam, 0.3)
        # a blunt tip exposes a stagnation region: Hoerner's hemisphere ~0.01 on its own area
        tip = 0.01 * (g.tip_radius / g.radius) ** 2 if g.tip_radius > 0 else 0.0
        return cone * k + tip

    def _tail_subsonic(self, mach):
        g = self.g
        if not g.base_diameter or g.tail_length <= 1e-9:
            return 0.0
        a_removed = (g.ref_area - math.pi * g.base_radius ** 2) / g.ref_area
        ratio = g.tail_length / max(g.diameter - 2 * g.base_radius, 1e-9)
        f = 1.0 if ratio <= 1.0 else (0.0 if ratio >= 3.0 else (3.0 - ratio) / 2.0)
        return base_drag_coefficient(mach) * a_removed * f

    def _supersonic(self, mach):
        key = round(float(mach), 3)
        if key not in self._sup_cache:
            self._sup_cache[key] = wave_drag_profile(self.x, self.r, mach, self.a_ref, self.g.nose_length)
        return self._sup_cache[key]

    def pressure(self, mach):
        """Nose and boat-tail pressure drag with the transonic bridge."""
        m = float(mach)
        if m <= M_SUB:
            return {"cd_nose": self._nose_subsonic(), "cd_tail": self._tail_subsonic(m)}
        if m >= M_SUP:
            total, nose = self._supersonic(m)
            return {"cd_nose": nose, "cd_tail": total - nose}
        n0, t0 = self._nose_subsonic(), self._tail_subsonic(M_SUB)
        tot1, n1 = self._supersonic(M_SUP)
        tot2, n2 = self._supersonic(M_SUP + 0.1)
        dn = (n2 - n1) / 0.1
        dt = ((tot2 - n2) - (tot1 - n1)) / 0.1
        return {"cd_nose": _hermite(m, M_SUB, M_SUP, n0, n1, 0.0, dn),
                "cd_tail": _hermite(m, M_SUB, M_SUP, t0, tot1 - n1, 0.0, dt)}

    def base(self, mach):
        g = self.g
        return base_drag_coefficient(mach) * (math.pi * g.base_radius ** 2) / self.a_ref

    def fins_pressure(self, mach):
        f = self.g.fins
        if not f:
            return {"cd_fin_le": 0.0, "cd_fin_te": 0.0}
        frontal = f.count * f.thickness * f.span / self.a_ref
        le = fin_edge_drag(mach, f.edge, self.sweep_rad) * frontal
        te = {"square": 1.0, "rounded": 0.5, "airfoil": 0.0}[f.edge] * base_drag_coefficient(mach) * frontal
        return {"cd_fin_le": le, "cd_fin_te": te}

    def lug(self, mach):
        lg = LAUNCH_LUGS[self.g.lug]
        if lg["d"] <= 0:
            return 0.0
        return 1.1 * (math.pi * lg["d"] ** 2 / 4.0) / self.a_ref

    # ---- totals -------------------------------------------------------------------------
    def breakdown(self, mach, re, alpha_rad=0.0, cn_alpha=None):
        out = self.friction(mach, re)
        out.update(self.pressure(mach))
        out["cd_base"] = self.base(mach)
        out.update(self.fins_pressure(mach))
        out["cd_lug"] = self.lug(mach)
        out["cd_friction"] = out["cd_friction_body"] + out["cd_friction_fins"] + out["cd_friction_lug"]
        out["cd_pressure"] = out["cd_nose"] + out["cd_tail"] + out["cd_fin_le"] + out["cd_fin_te"] + out["cd_lug"]
        out["cd0"] = out["cd_friction"] + out["cd_pressure"] + out["cd_base"]
        a = abs(float(alpha_rad))
        out["cd_alpha"] = (cn_alpha if cn_alpha is not None else 12.0) * a * a
        out["cd"] = out["cd0"] + out["cd_alpha"]
        out["mach"] = float(mach)
        out["re"] = float(re)
        return out

    def cd(self, mach, re, alpha_rad=0.0, cn_alpha=None):
        return self.breakdown(mach, re, alpha_rad, cn_alpha)["cd"]

    def curve(self, machs, air, length=None):
        """Cd against Mach number at the given air (rho, mu, a): the drag table."""
        L = length or self.g.length
        rows = []
        for m in machs:
            v = m * air["a"]
            re = air["rho"] * v * L / air["mu"]
            b = self.breakdown(m, re)
            rows.append({k: (round(float(b[k]), 5) if isinstance(b[k], (int, float)) else b[k]) for k in
                         ("mach", "re", "cd", "cd0", "cd_friction", "cd_friction_body", "cd_friction_fins", "cd_nose",
                          "cd_tail", "cd_base", "cd_fin_le", "cd_fin_te", "cd_lug", "cf", "rough_limited")})
        return rows
