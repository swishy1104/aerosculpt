"""Fuselage drag by component build-up: the fast model the optimizer explores winged designs with
before the CFD confirms the best ones.

  friction + form   Hoerner (Fluid-Dynamic Drag, 1965, ch. 6): Cf(Re_L) x (1 + 1.5 (d/l)^1.5
                    + 7 (d/l)^3) on the wetted area -- the same reference the CFD's streamlined-body
                    validation case uses
  base              Hoerner's base drag for a body with a cut-off base: CD_base = 0.029 / sqrt(CD_fore)
                    (both on the base area)
  cross-flow        attached slender body at incidence: 1.2 sin^3(alpha) on the planform area
"""
from __future__ import annotations

import math

from ..cfd.analysis import cf_flat_plate


def fuselage_drag(info, V, rho, mu, alpha_deg=0.0):
    L, D = info["length"], info["diameter"]
    q = 0.5 * rho * V * V
    re = rho * V * L / mu
    k = D / L
    ff = 1 + 1.5 * k ** 1.5 + 7 * k ** 3
    s_wet = info["body_wetted"]
    d_f = q * cf_flat_plate(re) * ff * s_wet
    db = info.get("base_diameter", 0.0)
    s_b = math.pi * db * db / 4
    d_b = 0.0
    if s_b > 1e-12:
        cd_fore = d_f / (q * s_b)
        d_b = q * s_b * 0.029 / math.sqrt(max(cd_fore, 1e-6))
    a = math.radians(alpha_deg)
    d_c = q * 0.8 * L * D * 1.2 * abs(math.sin(a)) ** 3
    return {"drag_N": d_f + d_b + d_c, "friction_form_N": d_f, "base_N": d_b, "crossflow_N": d_c}
