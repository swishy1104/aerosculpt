"""Fin flutter: the speed at which a fin's bending and torsion couple into a divergent
oscillation, from NACA TN 4197 (Martin 1958) in the form used throughout model and high-power
rocketry (Apogee Peak of Flight 291, Bennett):

    V_f = a sqrt( G / ( 1.337 AR^3 P (lambda + 1) / (2 (AR + 2) (t/c)^3) ) )

a: speed of sound at the altitude, P: air pressure there, G: the fin material's shear modulus,
AR: (span)^2 / area of one fin, lambda: tip / root chord, t/c: thickness / root chord.
A fin flutters when the rocket's speed exceeds V_f where it is fastest; keep a margin of at
least 20 % (the formula is a thin-plate estimate; a fin that is not glued stiffly, or with a
rough root joint, flutters sooner).
"""
from __future__ import annotations

import math

# shear modulus (Pa), typical values; wood values across the grain (the weak direction)
FIN_MATERIALS = {
    "balsa": {"label": "Balsa", "G": 0.25e9, "density": 160.0},
    "basswood": {"label": "Basswood", "G": 0.70e9, "density": 420.0},
    "plywood": {"label": "Birch plywood", "G": 0.62e9, "density": 680.0},
    "g10": {"label": "G10 fibreglass", "G": 4.1e9, "density": 1800.0},
    "carbon": {"label": "Carbon fibre plate", "G": 5.0e9, "density": 1550.0},
    "aluminium": {"label": "Aluminium", "G": 26.0e9, "density": 2700.0},
    "polycarbonate": {"label": "Polycarbonate", "G": 0.8e9, "density": 1200.0},
    "pla": {"label": "3D-printed PLA", "G": 1.2e9, "density": 1100.0},
    "abs": {"label": "ABS", "G": 0.9e9, "density": 1040.0},
}


def flutter_speed(fin, material, air):
    """Flutter speed (m/s) of a fin at the given air (p, a)."""
    mat = FIN_MATERIALS.get(material, FIN_MATERIALS["plywood"])
    cr, ct, _ = fin.equivalent
    if cr <= 0 or fin.area <= 0 or fin.thickness <= 0:
        return 0.0
    ar = fin.span ** 2 / fin.area
    lam = ct / cr
    tc = fin.thickness / cr
    denom = 1.337 * ar ** 3 * air["p"] * (lam + 1.0) / (2.0 * (ar + 2.0) * tc ** 3)
    return air["a"] * math.sqrt(mat["G"] / denom)


def flutter_check(fin, material, v_max, air_at_vmax, margin=1.2):
    vf = flutter_speed(fin, material, air_at_vmax)
    return {"flutter_speed": vf, "v_max": v_max, "ratio": vf / v_max if v_max > 0 else None,
            "ok": vf >= margin * v_max, "margin_required": margin, "material": material,
            "material_label": FIN_MATERIALS.get(material, FIN_MATERIALS["plywood"])["label"]}
