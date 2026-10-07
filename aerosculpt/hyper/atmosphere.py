"""U.S. Standard Atmosphere 1976, 0-86 km (geopotential layers, hydrostatic pressure)."""
from __future__ import annotations

import math

R_AIR = 287.05287          # J/(kg K)
GAMMA = 1.4
G0 = 9.80665
R_EARTH = 6356766.0        # m, US76 effective radius for geopotential altitude
# base geopotential altitude (m), temperature (K), lapse rate (K/m), pressure (Pa)
_LAYERS = [
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.001, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
    (51000.0, 270.65, -0.0028, 66.93887),
    (71000.0, 214.65, -0.002, 3.956420),
]
MAX_ALTITUDE = 86000.0


def sutherland(T):
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


def us76(z_m: float) -> dict:
    """Air properties at geometric altitude z (m)."""
    z = float(z_m)
    if not (-500.0 <= z <= MAX_ALTITUDE):
        raise ValueError(f"altitude {z / 1000:.1f} km is outside 0-86 km")
    h = R_EARTH * z / (R_EARTH + z)
    base = _LAYERS[0]
    for layer in _LAYERS:
        if h >= layer[0]:
            base = layer
    hb, Tb, L, pb = base
    T = Tb + L * (h - hb)
    if L == 0.0:
        p = pb * math.exp(-G0 * (h - hb) / (R_AIR * Tb))
    else:
        p = pb * (Tb / T) ** (G0 / (R_AIR * L))
    rho = p / (R_AIR * T)
    mu = sutherland(T)
    return {"T": T, "p": p, "rho": rho, "mu": mu, "nu": mu / rho, "a": math.sqrt(GAMMA * R_AIR * T)}
