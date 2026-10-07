"""Wing aerodynamics at a required lift: vortex-lattice span loading plus section drag.

The vortex lattice gives the angle of attack for the lift needed, the induced drag and each
strip's lift coefficient; NeuralFoil gives each strip's profile drag at that lift coefficient and
its own Reynolds number (strip theory). Strips buried in the fuselage carry lift (carry-over) but
no profile drag. Fins with no lift at zero sideslip contribute profile drag at zero lift.

Not modelled: propeller slipstream on the wing, pitch trim (the tail is set by its incidence),
compressibility (fine below Mach 0.3), and stall beyond flagging it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .neuralfoil import section_for
from .vlm import VLM

STALL_MARGIN = 0.92          # a strip above 92 % of its section's maximum lift counts as stalling


@dataclass
class DragSurface:
    """A fin carrying no lift: profile drag only (root-to-tip strips)."""
    name: str
    span: float
    root_chord: float
    tip_chord: float
    section: dict = field(default_factory=lambda: {"family": "naca4", "m": 0.0, "p": 0.4, "t": 0.10})
    exposed_from: float = 0.0
    n: int = 12


class WingModel:
    def __init__(self, surfaces, s_ref, c_ref, x_ref=(0.0, 0.0, 0.0), drag_surfaces=(), nf_size="xlarge"):
        self.nf_size = nf_size
        self.surfaces = list(surfaces)
        self.drag_surfaces = list(drag_surfaces)
        self.vlm = VLM(self.surfaces, s_ref, c_ref, x_ref)
        self.s_ref = float(s_ref)
        self._a_guess = {}

    def cl_max(self, si, re):
        return section_for(self.surfaces[si].section, self.nf_size).cl_max(re)[0]

    def evaluate(self, V, rho, mu, lift=None, alpha=None):
        """Forces in newtons at speed V for a required lift (N) or a given angle (deg)."""
        q = 0.5 * rho * V * V
        if alpha is None:
            alpha = self.vlm.alpha_for_cl(lift / (q * self.s_ref))
        r = self.vlm.solve(alpha, V, rho)
        st = self.vlm.strips
        cl = r["cl_strip"]
        out_s = []
        dp_total = 0.0
        stall = False
        margin = math.inf
        conf = 1.0
        for si, s in enumerate(self.surfaces):
            idx = np.flatnonzero(self.vlm.strip_owner == si)
            ch = np.array([st[k]["chord"] for k in idx])
            ds = np.array([st[k]["ds"] for k in idx])
            exp = np.array([st[k]["exposed"] for k in idx])
            re = rho * V * ch / mu
            sec = section_for(s.section, self.nf_size)
            a2 = sec.alpha_for_cl(cl[idx], re, guess=self._a_guess.get(si))
            self._a_guess[si] = a2
            pol = sec.polar(a2, re)
            cd = pol["CD"]
            dp = float(np.sum(q * ch * cd * ds * exp))
            dp_total += dp
            re_mid = float(np.median(re))
            clm = self.cl_max(si, re_mid)
            m_ = float(clm * STALL_MARGIN - np.max(cl[idx][exp] if exp.any() else cl[idx]))
            margin = min(margin, m_)
            stall = stall or m_ < 0
            conf = min(conf, float(np.min(pol["confidence"])))
            out_s.append({"name": s.name, "lift_N": float(r["lift_by_surface"][si]), "profile_N": dp,
                          "cl_max_section": clm, "cl_peak": float(np.max(cl[idx])), "re_mid": re_mid})
        for d in self.drag_surfaces:
            eta = (np.arange(d.n) + 0.5) / d.n
            ch = d.root_chord + (d.tip_chord - d.root_chord) * eta
            re = rho * V * ch / mu
            sec = section_for(d.section, self.nf_size)
            a2 = sec.alpha_for_cl(np.zeros_like(ch), re)
            cd = sec.polar(a2, re)["CD"]
            y = eta * d.span
            dp = float(np.sum(q * ch * cd * (d.span / d.n) * (y >= d.exposed_from)))
            dp_total += dp
            out_s.append({"name": d.name, "lift_N": 0.0, "profile_N": dp})
        return {"alpha": float(alpha), "CL": r["CL"], "lift_N": r["L"], "induced_N": r["Di"], "profile_N": dp_total,
                "drag_N": r["Di"] + dp_total, "stall": bool(stall), "cl_margin": margin, "confidence": conf,
                "surfaces": out_s}
