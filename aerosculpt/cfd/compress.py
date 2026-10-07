"""Compressibility for the General tunnel: from the incompressible lattice-Boltzmann solution to
flight Mach numbers up to 1.2.

The lattice runs at a low lattice Mach number, so its surface pressures are the incompressible
(M -> 0) solution. For the flight Mach number:

  M <= M_crit   Karman-Tsien correction of every face's pressure coefficient (capped at the
                isentropic stagnation value), re-integrated into force and moment changes;
                turbulent skin friction scaled by (1 - 0.1 M^2) (subsonic compressible
                flat-plate correction, as in Barrowman / OpenRocket).
  M >  M_crit   transonic estimate: pressure drag follows a cubic from the corrected value at
                M_crit (matching its slope) to the supersonic surface-method value at Mach 1.2
                (matching its slope), where the Supersonic tunnel takes over. Lift and moments
                are held at their M_crit values. This band is an estimate, not a validated model.

M_crit is where the Karman-Tsien-corrected minimum surface pressure reaches the sonic pressure
coefficient Cp*. Faces are grouped into narrow Cp bins with their summed area vectors and
moment arms, so the correction at any Mach number is a cheap sum.
"""
from __future__ import annotations

import math

import numpy as np

G = 1.4
M_SUPERSONIC = 1.2


def kt(cp0, M):
    """Karman-Tsien compressibility correction of an incompressible pressure coefficient."""
    b = math.sqrt(max(1.0 - M * M, 1e-6))
    return cp0 / (b + (M * M / (1.0 + b)) * cp0 / 2.0)


def cp_star(M):
    """Pressure coefficient at which the local flow becomes sonic."""
    return 2.0 / (G * M * M) * (((2.0 + (G - 1.0) * M * M) / (G + 1.0)) ** (G / (G - 1.0)) - 1.0)


def cp_stagnation(M):
    return 2.0 / (G * M * M) * ((1.0 + 0.5 * (G - 1.0) * M * M) ** (G / (G - 1.0)) - 1.0) if M > 1e-6 else 1.0


def critical_mach(cp_min):
    """Freestream Mach number at which the minimum surface pressure becomes sonic."""
    if cp_min >= -1e-3:
        return 0.99
    lo, hi = 0.05, 0.999
    for _ in range(60):
        m = 0.5 * (lo + hi)
        den = math.sqrt(1 - m * m) + (m * m / (1 + math.sqrt(1 - m * m))) * cp_min / 2
        if den <= 0 or kt(cp_min, m) <= cp_star(m):
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def surface_bins(cp, dA, r, n_bins=96):
    """Group faces by pressure coefficient. dA: area vectors / reference area (F, 3);
    r: moment arms / reference length (F, 3). Returns compact arrays for compressibility()."""
    ok = np.isfinite(cp)
    cp, dA, r = cp[ok], dA[ok], r[ok]
    area = np.linalg.norm(dA, axis=1)
    if not len(cp) or area.sum() <= 0:
        return None
    order = np.argsort(cp)
    cum = np.cumsum(area[order]) / area.sum()
    cp_min = float(cp[order][np.searchsorted(cum, 0.005)])      # ignore the lowest 0.5 % of area (noise)
    lo, hi = float(cp.min()), float(cp.max())
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, cp, side="right") - 1, 0, n_bins - 1)
    w = np.bincount(idx, weights=area, minlength=n_bins)
    cpm = np.bincount(idx, weights=cp * area, minlength=n_bins) / np.maximum(w, 1e-30)
    sdA = np.stack([np.bincount(idx, weights=dA[:, k], minlength=n_bins) for k in range(3)], axis=1)
    rxdA = np.cross(r, dA)
    srx = np.stack([np.bincount(idx, weights=rxdA[:, k], minlength=n_bins) for k in range(3)], axis=1)
    keep = w > 0
    return {"cp": cpm[keep], "dA": sdA[keep], "rxdA": srx[keep], "cp_min": cp_min,
            "mach_crit": critical_mach(cp_min)}


def pressure_delta(bins, M):
    """Change in force (3,) and moment (3,) coefficients from incompressible to Mach M (<= M_crit)."""
    cp0 = bins["cp"]
    b = math.sqrt(max(1.0 - M * M, 1e-6))
    den = b + (M * M / (1.0 + b)) * cp0 / 2.0
    cpc = np.where(den > 0.05 * b, cp0 / np.maximum(den, 1e-9), cp0 / (0.05 * b))
    cpc = np.clip(cpc, -2.0 / (G * M * M) if M > 0.05 else -1e9, cp_stagnation(M))
    d = cpc - cp0
    return -(d[:, None] * bins["dA"]).sum(axis=0), -(d[:, None] * bins["rxdA"]).sum(axis=0)


def hermite(x, x0, y0, s0, x1, y1, s1):
    t = (x - x0) / (x1 - x0)
    h = x1 - x0
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y0 + (t ** 3 - 2 * t ** 2 + t) * h * s0
            + (-2 * t ** 3 + 3 * t ** 2) * y1 + (t ** 3 - t ** 2) * h * s1)


def transonic_cdp(M, mc, cdp_mc, slope_mc, anchor):
    """Pressure drag between M_crit and Mach 1.2: cubic to the supersonic surface-method value."""
    if anchor is None:            # no supersonic anchor: Lock's empirical drag-rise law
        return cdp_mc + slope_mc * (M - mc) + 20.0 * (M - mc) ** 4
    y1 = anchor["cdp"][0]
    s1 = (anchor["cdp"][1] - anchor["cdp"][0]) / (anchor["mach"][1] - anchor["mach"][0])
    return hermite(M, mc, cdp_mc, slope_mc, M_SUPERSONIC, y1, s1)
