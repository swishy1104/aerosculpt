"""Static stability: the centre of pressure by Barrowman's method (1966, Centuri TIR-33; the
method behind every model-rocket stability calculator), the centre of gravity from the parts and
the motor as it burns, and the static margin in calibres (body diameters).

  nose        CN_alpha = 2 on the body cross-section; x_cp = L_n - V_n / A_ref, exact for any
              profile at small angles (2/3 L for a cone, 0.466 L for a tangent ogive)
  boat tail   CN_alpha = 2 [(d_base / d)^2 - 1] (destabilising), Barrowman's transition centre
  fins        Barrowman's trapezoidal-fin formula with the body interference factor 1 + R/(s+R),
              Prandtl-Glauert below Mach 0.9, Busemann (supersonic thin-airfoil) above Mach 1.5,
              cubic between; the centre of pressure moves aft toward mid-chord at supersonic speed
              (OpenRocket 3.2.2). Fin counts above four lose effectiveness (0.95 for 5, 0.91 for 6).
  margin      (x_cp - x_cg) / d. Rules of thumb: at least 1 calibre at liftoff (NAR), 1.5-2 for
              high-power and windy launches; above ~3-4 the rocket weathercocks strongly.

Not modelled: body lift at larger angles of attack (which moves the CP forward on long rockets),
fin cant, and fins on the nose or a transition.
"""
from __future__ import annotations

import math

from .geometry import RocketGeometry

FIN_EFFICIENCY = {1: 0.5, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.948, 6: 0.913, 7: 0.854, 8: 0.810}


def _cubic(m, m0, m1, y0, y1, d0, d1):
    t = (m - m0) / (m1 - m0)
    h = m1 - m0
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y0 + (t ** 3 - 2 * t ** 2 + t) * h * d0
            + (-2 * t ** 3 + 3 * t ** 2) * y1 + (t ** 3 - t ** 2) * h * d1)


class Barrowman:
    def __init__(self, g: RocketGeometry):
        self.g = g
        self.d = g.diameter
        self.a_ref = g.ref_area
        self.nose_v = g.nose_volume()

    # ---- components ---------------------------------------------------------------------
    def nose(self):
        g = self.g
        return 2.0, g.nose_length - self.nose_v / self.a_ref

    def tail(self):
        g = self.g
        if not g.base_diameter or g.tail_length <= 1e-9:
            return 0.0, 0.0
        d1, d2 = g.diameter, 2 * g.base_radius
        cna = 2.0 * ((d2 / self.d) ** 2 - (d1 / self.d) ** 2)
        r = d1 / d2
        x = (g.length - g.tail_length) + g.tail_length / 3.0 * (1.0 + (1.0 - r) / (1.0 - r * r))
        return cna, x

    def _fin_sub(self, beta):
        f = self.g.fins
        cr, ct, xr = f.equivalent
        s = f.span
        lm = math.hypot(s, xr + 0.5 * ct - 0.5 * cr)
        return 4.0 * f.count * (s / self.d) ** 2 / (1.0 + math.sqrt(1.0 + (beta * 2.0 * lm / max(cr + ct, 1e-9)) ** 2))

    def _fin_sup(self, beta):
        f = self.g.fins
        return f.count * (f.area / self.a_ref) * 2.0 / beta

    def fins(self, mach=0.0):
        f = self.g.fins
        if not f or f.span <= 0:
            return 0.0, 0.0
        R = self.g.radius
        k_int = 1.0 + R / (f.span + R)
        eff = FIN_EFFICIENCY.get(f.count, 0.8)
        m = float(mach)
        if m < 0.9:
            cna = self._fin_sub(math.sqrt(1.0 - m * m))
        elif m > 1.5:
            cna = self._fin_sup(math.sqrt(m * m - 1.0))
        else:
            y0 = self._fin_sub(math.sqrt(1.0 - 0.81))
            d0 = (y0 - self._fin_sub(math.sqrt(1.0 - 0.89 ** 2))) / 0.01
            y1 = self._fin_sup(math.sqrt(1.5 ** 2 - 1.0))
            d1 = (self._fin_sup(math.sqrt(1.51 ** 2 - 1.0)) - y1) / 0.01
            cna = _cubic(m, 0.9, 1.5, y0, y1, d0, d1)
        cna *= k_int * eff
        # centre of pressure: Barrowman subsonic, toward mid-chord supersonic
        cr, ct, xr = f.equivalent
        x_sub = f.x_le + xr * (cr + 2 * ct) / (3 * max(cr + ct, 1e-9)) + (cr + ct - cr * ct / max(cr + ct, 1e-9)) / 6.0
        if m <= 0.5:
            x = x_sub
        else:
            mac = f.mac
            x_mac_le = f.x_le + xr * (cr + 2 * ct) / (3 * max(cr + ct, 1e-9))
            ar = f.aspect_ratio
            beta = math.sqrt(max(m * m - 1.0, 1e-6)) if m > 1 else 1e-3
            frac = (ar * beta - 0.67) / max(2.0 * ar * beta - 1.0, 1e-6) if m >= 2.0 else None
            x_sup = x_mac_le + (min(max(frac, 0.25), 0.6) if frac is not None else 0.5) * mac
            if m >= 2.0:
                x = x_sup
            else:
                t = (m - 0.5) / 1.5
                x = x_sub + (x_sup - x_sub) * (3 * t * t - 2 * t ** 3)
        return cna, x

    def body_lift(self, alpha_rad):
        """Normal force of the cylindrical body at an angle (Galejs' correction, as in
        OpenRocket): 1.1 x planform / A_ref x sin^2(alpha); reported, not in the Barrowman sum."""
        g = self.g
        plan = g.diameter * (g.length - g.nose_length) + 0.6 * g.diameter * g.nose_length
        cn = 1.1 * plan / self.a_ref * math.sin(alpha_rad) ** 2
        return cn, g.nose_length + 0.5 * (g.length - g.nose_length)

    # ---- totals -------------------------------------------------------------------------
    def cp(self, mach=0.0):
        parts = {"nose": self.nose(), "tail": self.tail(), "fins": self.fins(mach)}
        cna = sum(c for c, _ in parts.values())
        x = sum(c * xx for c, xx in parts.values()) / cna if abs(cna) > 1e-12 else self.g.length / 2
        return {"cn_alpha": cna, "x_cp": x, "mach": float(mach),
                "parts": {k: {"cn_alpha": c, "x": xx} for k, (c, xx) in parts.items()}}


def centre_of_gravity(items):
    """items: [(name, mass_kg, x_m)] -> (mass, x_cg)."""
    m = sum(i[1] for i in items)
    if m <= 0:
        return 0.0, 0.0
    return m, sum(i[1] * i[2] for i in items) / m


def static_margin(x_cp, x_cg, diameter):
    return (x_cp - x_cg) / diameter


def rate(margin):
    """A verdict on a static margin in calibres."""
    if margin < 0.5:
        return "unstable"
    if margin < 1.0:
        return "marginal"
    if margin <= 3.0:
        return "stable"
    return "overstable"
