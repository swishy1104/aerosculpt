"""Skin materials for the thermal check: maximum service temperature and emissivity.

Service limits are typical figures for structural use (strength retention or coating limit),
not melting points; check your supplier's data for a real design. Users can define their own.
"""
from __future__ import annotations

MATERIALS = {
    "fiberglass": {"name": "Fibreglass / epoxy (G10)", "t_max_c": 120.0, "emissivity": 0.90,
                   "note": "Hobby and amateur rockets; epoxy softens above ~120 °C"},
    "cfrp": {"name": "Carbon fibre / epoxy", "t_max_c": 150.0, "emissivity": 0.85,
             "note": "Limited by the resin glass transition"},
    "aluminium": {"name": "Aluminium alloy (painted)", "t_max_c": 150.0, "emissivity": 0.80,
                  "note": "2024 / 7075 lose strength quickly above ~150 °C"},
    "titanium": {"name": "Titanium Ti-6Al-4V", "t_max_c": 400.0, "emissivity": 0.60,
                 "note": "Mach 3 airframe skins"},
    "steel": {"name": "Stainless steel (PH 15-7 Mo)", "t_max_c": 550.0, "emissivity": 0.60,
              "note": "Hot structures, fins and leading edges"},
    "inconel": {"name": "Inconel 718", "t_max_c": 700.0, "emissivity": 0.85,
                "note": "Nickel superalloy hot structure"},
    "tile": {"name": "Silica tile (LI-900, black coating)", "t_max_c": 1260.0, "emissivity": 0.85,
             "note": "Insulating thermal-protection tile (not structural)"},
    "rcc": {"name": "Reinforced carbon-carbon", "t_max_c": 1650.0, "emissivity": 0.85,
            "note": "Nose caps and wing leading edges"},
    "uhtc": {"name": "UHTC (ZrB2-SiC)", "t_max_c": 2000.0, "emissivity": 0.80,
             "note": "Sharp hypersonic leading edges"},
    "ablator": {"name": "Carbon-phenolic ablator", "t_max_c": 3000.0, "emissivity": 0.90,
                "note": "Surface recedes; radiative equilibrium overstates its temperature"},
}
DEFAULT = "aluminium"


def resolve(spec) -> dict:
    """A material key or a custom {name, t_max_c, emissivity} -> material dict."""
    if isinstance(spec, str):
        if spec not in MATERIALS:
            raise ValueError(f"unknown material {spec!r}")
        return {"key": spec, **MATERIALS[spec]}
    if isinstance(spec, dict):
        t = float(spec.get("t_max_c", 150.0))
        e = float(spec.get("emissivity", 0.8))
        if not (-50.0 < t < 4000.0) or not (0.02 <= e <= 1.0):
            raise ValueError("custom material needs a service limit between -50 and 4000 °C and an emissivity between 0.02 and 1")
        return {"key": "custom", "name": str(spec.get("name") or "Custom material")[:60], "t_max_c": t,
                "emissivity": e, "note": "User defined"}
    return {"key": DEFAULT, **MATERIALS[DEFAULT]}
