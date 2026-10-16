"""Wing structure: a spar sized for the design load, the wing's mass, and its tip deflection.

Preliminary-design beam model (the standard first sizing step, before any finite-element model):

* Load: the computed span loading (vortex-lattice strip lift) scaled to the ultimate load,
  load factor x safety factor x the flight weight, carried by the lifting surfaces.
* Spar: two caps of the chosen material at the section's depth (0.9 x the airfoil thickness),
  each `cap_width` x chord wide; the caps' area at every station is what the bending moment
  there needs at the material's allowable stress (sigma = M / (A h)). Each cap may fill at most
  CAP_FILL of the half-depth: a wing too thin for its load fails that check.
* Stiffness: tip deflection at the limit load (n x weight, no safety factor) from the same caps,
  by integrating M / (E I) twice from the root (Euler-Bernoulli cantilever). When it exceeds the
  allowed fraction of the semi-span, the caps grow until it does not (stiffness-sized spar).
* Mass: spar caps + a shear web allowance + skin (areal density x wetted area, about 2.04 x the
  planform for a 12 % airfoil).

Not modelled: torsion, the root joint and carry-through, buckling of thin caps, flutter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G0 = 9.80665
CAP_FILL = 0.35          # each cap may be at most this fraction of the half-depth thick
WEB_FACTOR = 0.25        # shear webs, glue, and fittings as a fraction of the cap mass


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    allowable_mpa: float     # design stress (compression side governs for laminates)
    modulus_gpa: float
    density: float           # kg/m^3


MATERIALS = {
    "carbon": Material("carbon", "Carbon fibre (unidirectional)", 600.0, 120.0, 1550.0),
    "glass": Material("glass", "Glass fibre (unidirectional)", 250.0, 30.0, 1900.0),
    "aluminium": Material("aluminium", "Aluminium 6061-T6", 160.0, 69.0, 2700.0),
    "spruce": Material("spruce", "Spruce / plywood", 45.0, 10.0, 450.0),
}


@dataclass
class StructureSettings:
    material: str = "carbon"
    load_factor: float = 3.8          # limit load factor (FAR 23 normal category)
    safety_factor: float = 1.5
    skin_kg_m2: float = 0.45          # skin areal density (foam + glass sheeting on a small UAV)
    cap_width: float = 0.15           # spar cap width as a fraction of the local chord
    max_tip_deflection: float = 0.15  # at limit load, as a fraction of the semi-span

    @classmethod
    def from_spec(cls, d):
        d = d or {}
        out = cls()
        if d.get("material") in MATERIALS:
            out.material = d["material"]
        for k, lo, hi in (("load_factor", 1.0, 12.0), ("safety_factor", 1.0, 3.0), ("skin_kg_m2", 0.05, 5.0),
                          ("max_tip_deflection", 0.02, 0.5)):
            if d.get(k) is not None:
                try:
                    out.__setattr__(k, float(np.clip(float(d[k]), lo, hi)))
                except (TypeError, ValueError):
                    pass
        return out


def size_surface(y, chord, thick, lift_per_m, semi_span, settings: StructureSettings, weight_share):
    """One half-wing, root at y = 0. y: station positions (m, ascending), chord and thick
    (thickness ratio) at them, lift_per_m: the span loading shape (N/m, any scale). The loading
    is scaled so the half-wing carries weight_share x the ultimate (or limit) load."""
    mat = MATERIALS[settings.material]
    y = np.asarray(y, float)
    w = np.clip(np.asarray(lift_per_m, float), 0.0, None)
    order = np.argsort(y)
    y, chord, thick, w = y[order], np.asarray(chord, float)[order], np.asarray(thick, float)[order], w[order]
    # add the root and tip stations so the integrals span the whole half-wing
    y = np.concatenate([[0.0], y, [semi_span]])
    chord = np.concatenate([[chord[0]], chord, [chord[-1]]])
    thick = np.concatenate([[thick[0]], thick, [thick[-1]]])
    w = np.concatenate([[w[0]], w, [0.0]])
    total = np.trapezoid(w, y)
    if total <= 0:
        w = np.ones_like(y)
        total = np.trapezoid(w, y)
    out = {}
    for case, factor in (("ultimate", settings.load_factor * settings.safety_factor), ("limit", settings.load_factor)):
        q = w / total * weight_share * factor          # N/m on this half-wing
        # shear and bending moment, integrated from the tip inwards
        V = np.zeros_like(y)
        M = np.zeros_like(y)
        for k in range(len(y) - 2, -1, -1):
            dy = y[k + 1] - y[k]
            V[k] = V[k + 1] + 0.5 * (q[k] + q[k + 1]) * dy
            M[k] = M[k + 1] + 0.5 * (V[k] + V[k + 1]) * dy
        out[case] = (V, M)
    V_u, M_u = out["ultimate"]
    h = 0.9 * thick * chord                                  # cap centroid spacing
    sigma = mat.allowable_mpa * 1e6
    A = M_u / (np.maximum(h, 1e-6) * sigma)                  # area of each cap
    width = settings.cap_width * chord
    # tip deflection at limit load with the strength-sized caps (I = 2 A (h/2)^2)
    _, M_l = out["limit"]

    def deflection(area):
        EI = mat.modulus_gpa * 1e9 * 2.0 * area * (0.5 * h) ** 2
        curv = M_l / np.maximum(EI, 1e-12)
        slope = np.concatenate([[0.0], np.cumsum(0.5 * (curv[1:] + curv[:-1]) * np.diff(y))])
        return float(np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(y))])[-1])

    defl = deflection(A)
    allowed = settings.max_tip_deflection * semi_span
    # stiffness-sized when the strength-sized spar bends too far: scale the caps (deflection ~ 1/A)
    k = max(1.0, defl / max(allowed, 1e-9))
    A = A * k
    defl = defl / k
    t_cap = A / np.maximum(width, 1e-9)
    fill = t_cap / np.maximum(0.5 * h, 1e-9)                 # fraction of the half-depth each cap needs
    cap_mass = 2.0 * np.trapezoid(A, y) * mat.density
    spar_mass = cap_mass * (1.0 + WEB_FACTOR)
    planform = np.trapezoid(chord, y)
    skin_mass = settings.skin_kg_m2 * planform * 2.04       # wetted area of a ~12 % airfoil, both sides
    return {
        "root_moment_Nm": float(M_u[0]), "root_shear_N": float(V_u[0]),
        "spar_kg": float(spar_mass), "skin_kg": float(skin_mass), "mass_kg": float(spar_mass + skin_mass),
        "max_cap_fill": float(np.max(fill[:-1])), "tip_deflection_m": defl,
        "tip_deflection_frac": defl / max(semi_span, 1e-9), "sized_by": "stiffness" if k > 1.0 else "strength",
        "root_cap_mm2": float(A[0] * 1e6), "root_depth_mm": float(h[0] * 1000),
    }


def wing_structure(vlm, surfaces, weight_N, settings: StructureSettings, cl_strip):
    """Spar sizing and mass for every lifting surface of a vortex lattice at one flight state
    (cl_strip from VLM.solve). The main wing carries the weight; another surface is sized for its
    own share of the computed lift, at least 10 % of the weight."""
    strips = vlm.strips
    owner = np.asarray(vlm.strip_owner)
    cl_strip = np.asarray(cl_strip)
    lifts = []
    for si in range(len(surfaces)):
        idx = np.flatnonzero(owner == si)
        lifts.append(float(sum(cl_strip[k] * strips[k]["chord"] * strips[k]["ds"] for k in idx)))
    main = int(np.argmax(np.abs(lifts))) if lifts else 0
    total = sum(abs(x) for x in lifts) or 1.0
    out = {"surfaces": [], "mass_kg": 0.0, "ok": True, "why": None, "material": MATERIALS[settings.material].name,
           "load_factor": settings.load_factor, "safety_factor": settings.safety_factor}
    for si, s in enumerate(surfaces):
        idx = [k for k in np.flatnonzero(owner == si) if strips[k]["eta"] >= 0.0]
        if not idx:
            idx = list(np.flatnonzero(owner == si))
        yy = np.array([strips[k]["y_abs"] for k in idx])
        ch = np.array([strips[k]["chord"] for k in idx])
        load = cl_strip[idx] * ch
        share = 0.5 * weight_N * (1.0 if si == main else max(abs(lifts[si]) / total, 0.1))
        r = size_surface(yy, ch, np.full(len(idx), float(s.section.get("t", 0.12))), load, s.span / 2, settings, share)
        for k in ("mass_kg", "spar_kg", "skin_kg"):
            r[k] *= 2.0                                     # both halves
        r["name"] = s.name
        out["surfaces"].append(r)
        out["mass_kg"] += r["mass_kg"]
        if out["ok"] and r["max_cap_fill"] > CAP_FILL:
            out["ok"] = False
            out["why"] = (f"{s.name} too thin for its load: the spar caps would fill {r['max_cap_fill']:.0%} of its depth "
                          f"({MATERIALS[settings.material].name}, n = {settings.load_factor:g})")
    return out
