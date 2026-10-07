"""Blade-element momentum theory (BEMT) for fixed-pitch RC propellers.

Geometry: diameter and geometric pitch, as printed on the propeller (e.g. 10x7 inches), blade
count, and a blade family. Twist follows the geometric pitch, beta = atan(P / 2 pi r), capped
at 45 degrees near the hub as on real blades. Each family has a measured planform (UIUC
propeller database) and its own section zero-lift angle and profile-drag level, fitted on one
propeller and checked on others (validation suite):

  thin-electric  APC thin-electric planform (apce_10x7_geom). Fitted on the APC 10x7E; APC 9x6E
                 and 11x10E mean thrust and power within 5 %, the fitted 10x7E within 8 %.
  wide           constant-chord blade, c/R = 0.23 (DA4022 9x6.75 geometry). Fitted on the
                 2-blade DA4022; the same blade with 3 and 4 blades within 5 % (blade-count
                 effect checked: 3 blades +29 %, 4 blades +56 % thrust at J = 0.5, as measured).

Section aerodynamics: a thin cambered low-Reynolds-number airfoil (2D lift slope 0.9 x 2 pi,
drag rising as Re falls), flat-plate behaviour past stall, Prandtl-Glauert for the local Mach
number. Prandtl tip and hub losses (these carry the blade-count effect).

Each annulus is solved for its inflow angle with a bracketed one-variable residual (the
approach of Ning 2014, written for a propeller), so it always converges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INCH = 0.0254
_RR = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
FAMILIES = {
    "thin-electric": {
        "name": "Thin electric (APC-style)", "note": "Slim tapered blades: efficient sport and UAV props",
        # UIUC volume 1, apce_10x7_geom: c/R at _RR
        "cr": np.array([0.138, 0.154, 0.175, 0.190, 0.198, 0.202, 0.200, 0.195, 0.186, 0.174, 0.161, 0.145, 0.129,
                        0.112, 0.096, 0.081, 0.061, 0.040]),
        "alpha0_deg": -3.5, "cd0": 0.018,
    },
    "wide": {
        "name": "Wide blade (constant chord)", "note": "Broad, square-tipped blades: more thrust per diameter, scale and multirotor props",
        # UIUC volume 2, da4022_9x6.75_geom: c/R at _RR
        "cr": np.array([0.1193, 0.1286, 0.1828, 0.2233, 0.2332, 0.2301, 0.2288, 0.2302, 0.2310, 0.2304, 0.2299, 0.2300,
                        0.2301, 0.2303, 0.2307, 0.2308, 0.2179, 0.0282]),
        "alpha0_deg": -1.5, "cd0": 0.018,
    },
}
CLA = 0.9 * 2.0 * math.pi
CL_MAX = 1.15


@dataclass
class Propeller:
    diameter: float          # m
    pitch: float             # m (geometric, at 75 % radius)
    blades: int = 2
    width: float = 1.0       # chord scale on the family's planform
    hub: float = 0.15        # hub radius / tip radius
    family: str = "thin-electric"

    @classmethod
    def from_inches(cls, d_in, p_in, blades=2, width=1.0, family="thin-electric"):
        return cls(d_in * INCH, p_in * INCH, int(blades), float(width), family=family)

    @property
    def fam(self):
        return FAMILIES.get(self.family, FAMILIES["thin-electric"])

    def stations(self, n=40):
        R = 0.5 * self.diameter
        x = 0.5 * (1 - np.cos(np.linspace(0.0, math.pi, n + 2)[1:-1]))   # cosine spacing
        rr = self.hub + (1.0 - self.hub) * x
        r = rr * R
        c = np.interp(rr, _RR, self.fam["cr"]) * R * self.width
        beta = np.minimum(np.arctan(self.pitch / (2 * math.pi * r)), math.radians(45.0))
        return r, c, beta


def _polar(alpha, re, mach, alpha0, cd_ref):
    """Section lift and drag coefficients."""
    cd0 = float(np.clip(cd_ref * (max(re, 1e3) / 2e5) ** -0.5, 0.66 * cd_ref, 5 * cd_ref))
    pg = 1.0 / math.sqrt(max(1.0 - min(mach, 0.85) ** 2, 0.25))
    cl_lin = CLA * (alpha - alpha0) * pg
    if abs(cl_lin) <= CL_MAX:
        cl = cl_lin
        cd = cd0 + 0.02 * (cl - 0.3) ** 2
    else:  # past stall: blend to a flat plate
        cl = math.copysign(min(abs(cl_lin), max(CL_MAX * 0.85, abs(2.0 * math.sin(alpha) * math.cos(alpha)))), cl_lin)
        cd = cd0 + 2.0 * math.sin(alpha) ** 2
    return cl, cd


def _loss(B, r, R, r_hub, sphi):
    sphi = max(abs(sphi), 1e-6)
    ft = 0.5 * B * (R - r) / (r * sphi)
    fh = 0.5 * B * (r - r_hub) / (r_hub * sphi)
    F = (2 / math.pi) * math.acos(min(math.exp(-ft), 1.0)) * (2 / math.pi) * math.acos(min(math.exp(-fh), 1.0))
    return max(F, 1e-4)


def bemt(prop: Propeller, v_axial, rpm, rho=1.225, mu=1.81e-5, a_sound=340.3, n=40):
    """Thrust (N), torque (N m), power (W) and the radial loading of one propeller."""
    V = max(float(v_axial), 0.1)
    omega = rpm * 2 * math.pi / 60.0
    R = 0.5 * prop.diameter
    r_hub = prop.hub * R
    r, c, beta = prop.stations(n)
    B = prop.blades
    a0 = math.radians(prop.fam["alpha0_deg"])
    cdr = prop.fam["cd0"]
    dT = np.zeros(n)
    dQ = np.zeros(n)
    phis = np.zeros(n)
    for k in range(n):
        rk, ck, bk = r[k], c[k], beta[k]
        sig = B * ck / (2 * math.pi * rk)
        wr = math.hypot(V, omega * rk)
        re = rho * wr * ck / mu
        mach = wr / a_sound

        def resid(phi):
            sp, cp_ = math.sin(phi), math.cos(phi)
            cl, cd = _polar(bk - phi, re, mach, a0, cdr)
            cx = cl * cp_ - cd * sp
            cy = cl * sp + cd * cp_
            F = _loss(B, rk, R, r_hub, sp)
            kk = sig * cx / (4 * F * sp * sp)
            if kk >= 1.0:
                return -1.0, None
            a = kk / (1 - kk)
            kp = sig * cy / (4 * F * sp * cp_)
            ap = kp / (1 + kp)
            return sp / (1 + a) - V * cp_ / (omega * rk * (1 - ap)), (a, ap, cx, cy)

        grid = np.linspace(1e-3, 0.5 * math.pi - 1e-3, 160)
        prev_phi, prev_val = grid[0], resid(grid[0])[0]
        lo = hi = None
        for phi in grid[1:]:
            val = resid(phi)[0]
            if prev_val < 0 <= val:
                lo, hi = prev_phi, phi
                break
            prev_phi, prev_val = phi, val
        if lo is None:
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if resid(mid)[0] < 0:
                lo = mid
            else:
                hi = mid
        phi = 0.5 * (lo + hi)
        _, (a, ap, cx, cy) = resid(phi)
        W2 = (V * (1 + a)) ** 2 + (omega * rk * (1 - ap)) ** 2
        dT[k] = 0.5 * rho * W2 * B * ck * cx
        dQ[k] = 0.5 * rho * W2 * B * ck * cy * rk
        phis[k] = phi
    T = float(np.trapezoid(dT, r))
    Q = float(np.trapezoid(dQ, r))
    P = Q * omega
    nrev = rpm / 60.0
    D = prop.diameter
    return {"thrust_N": T, "torque_Nm": Q, "power_W": P, "eta": T * V / P if P > 1e-9 else 0.0,
            "J": V / (nrev * D) if nrev > 0 else 0.0, "CT": T / (rho * nrev ** 2 * D ** 4) if nrev > 0 else 0.0,
            "CP": P / (rho * nrev ** 3 * D ** 5) if nrev > 0 else 0.0, "rpm": rpm, "v_axial": V,
            "tip_mach": math.hypot(V, omega * R) / a_sound,
            "r": r, "dT_dr": dT, "dQ_dr": dQ}
