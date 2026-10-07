"""Perfect-gas (gamma = 1.4) shock and expansion relations used by the panel methods.

Per freestream Mach number the oblique-shock (wedge) and Taylor-Maccoll (cone) solutions are
tabulated once against the surface inclination, then looked up for every panel. Tables are
cached by Mach number, so an envelope of a few Mach numbers costs a few hundred ODE solves.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit

G = 1.4
GM1 = G - 1.0
GP1 = G + 1.0


# ---- normal shock and stagnation -----------------------------------------------------
def normal_shock(M):
    """Ratios across a normal shock at upstream Mach M (array or scalar)."""
    M2 = np.asarray(M, float) ** 2
    p = 1.0 + 2.0 * G / GP1 * (M2 - 1.0)
    rho = GP1 * M2 / (GM1 * M2 + 2.0)
    T = p / rho
    Md2 = (1.0 + 0.5 * GM1 * M2) / (G * M2 - 0.5 * GM1)
    p0 = rho ** (G / GM1) * (1.0 / p) ** (1.0 / GM1)
    return {"p": p, "rho": rho, "T": T, "M": np.sqrt(Md2), "p0": p0}


def p0_over_p(M):
    return (1.0 + 0.5 * GM1 * np.asarray(M, float) ** 2) ** (G / GM1)


def cp_max(M):
    """Pressure coefficient at the stagnation point behind a normal shock (Rayleigh pitot)."""
    M = float(M)
    p02_p1 = normal_shock(M)["p0"] * p0_over_p(M)
    return 2.0 / (G * M * M) * (p02_p1 - 1.0)


def cp_from_ratio(M, p_ratio):
    return 2.0 / (G * M * M) * (np.asarray(p_ratio, float) - 1.0)


# ---- Prandtl-Meyer ---------------------------------------------------------------------
NU_MAX = 0.5 * math.pi * (math.sqrt(GP1 / GM1) - 1.0)


def pm_nu(M):
    M = np.maximum(np.asarray(M, float), 1.0)
    a = np.sqrt(GP1 / GM1)
    return a * np.arctan(np.sqrt((M * M - 1.0) / a ** 2)) - np.arctan(np.sqrt(M * M - 1.0))


_PM_M = np.concatenate([1.0 + np.logspace(-6, 0, 600), np.logspace(math.log10(2.0), 3, 2400)[1:]])
_PM_NU = pm_nu(_PM_M)


def pm_mach(nu):
    return np.interp(nu, _PM_NU, _PM_M)


def expansion_cp(M, delta):
    """Cp after an isentropic turn of delta (rad, >= 0) away from a freestream at Mach M."""
    nu2 = pm_nu(M) + np.asarray(delta, float)
    M2 = pm_mach(np.minimum(nu2, _PM_NU[-1]))
    ratio = ((1.0 + 0.5 * GM1 * M * M) / (1.0 + 0.5 * GM1 * M2 * M2)) ** (G / GM1)
    ratio = np.where(nu2 >= NU_MAX, 0.0, ratio)
    return cp_from_ratio(M, ratio)


# ---- oblique shock (wedge) --------------------------------------------------------------
def theta_from_beta(M, beta):
    s = np.sin(beta)
    num = 2.0 / np.tan(beta) * (M * M * s * s - 1.0)
    den = M * M * (G + np.cos(2.0 * beta)) + 2.0
    return np.arctan(num / den)


@lru_cache(maxsize=64)
def wedge_table(M):
    """Weak-branch oblique-shock solution vs deflection: (theta, cp, p0 ratio), theta_max."""
    mu = math.asin(1.0 / M)
    b = mu + (0.5 * math.pi - mu) * np.linspace(0.0, 1.0, 4001)[1:] ** 1.5
    th = theta_from_beta(M, b)
    k = int(np.argmax(th))
    b, th = b[:k + 1], th[:k + 1]
    th = np.maximum.accumulate(np.maximum(th, 0.0))
    ns = normal_shock(M * np.sin(b))
    cp = cp_from_ratio(M, ns["p"])
    th = np.concatenate([[0.0], th])
    cp = np.concatenate([[0.0], cp])
    p0 = np.concatenate([[1.0], ns["p0"]])
    return th, cp, p0, float(th[-1])


# ---- Taylor-Maccoll (cone) ---------------------------------------------------------------
@njit(cache=True)
def _tm_rhs(th, vr, vt):
    A = 0.5 * GM1 * (1.0 - vr * vr - vt * vt)
    return (vt * vt * vr - A * (2.0 * vr + vt / math.tan(th))) / (A - vt * vt)


@njit(cache=True)
def taylor_maccoll(M, beta):
    """Integrate the Taylor-Maccoll equation inward from a conical shock at angle beta.

    Returns (cone half-angle, surface Mach number); (-1, -1) if the solution is invalid."""
    s = math.sin(beta)
    tdel = 2.0 / math.tan(beta) * (M * M * s * s - 1.0) / (M * M * (G + math.cos(2.0 * beta)) + 2.0)
    delta = math.atan(tdel)
    mn1 = M * s
    mn2 = math.sqrt((1.0 + 0.5 * GM1 * mn1 * mn1) / (G * mn1 * mn1 - 0.5 * GM1))
    M2 = mn2 / math.sin(beta - delta)
    V = 1.0 / math.sqrt(2.0 / (GM1 * M2 * M2) + 1.0)
    vr = V * math.cos(beta - delta)
    vt = -V * math.sin(beta - delta)
    th = beta
    h = -2.0e-4
    while th > 1e-4:
        # classic RK4 on (vr, vt)
        k1r, k1t = vt, _tm_rhs(th, vr, vt)
        k2r, k2t = vt + 0.5 * h * k1t, _tm_rhs(th + 0.5 * h, vr + 0.5 * h * k1r, vt + 0.5 * h * k1t)
        k3r, k3t = vt + 0.5 * h * k2t, _tm_rhs(th + 0.5 * h, vr + 0.5 * h * k2r, vt + 0.5 * h * k2t)
        k4r, k4t = vt + h * k3t, _tm_rhs(th + h, vr + h * k3r, vt + h * k3t)
        nvr = vr + h / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        nvt = vt + h / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        if nvt >= 0.0:
            f = -vt / (nvt - vt)
            vr_c = vr + f * (nvr - vr)
            th_c = th + f * h
            if vr_c >= 1.0 or vr_c <= 0.0:
                return -1.0, -1.0
            Mc = math.sqrt(2.0 / GM1 * vr_c * vr_c / (1.0 - vr_c * vr_c))
            return th_c, Mc
        vr, vt, th = nvr, nvt, th + h
    return -1.0, -1.0


@lru_cache(maxsize=64)
def cone_table(M):
    """Weak-branch conical-shock solution vs cone half-angle: (theta_c, cp, p0 ratio), max."""
    mu = math.asin(1.0 / M)
    betas = mu + (0.5 * math.pi - 1e-3 - mu) * np.linspace(0.0, 1.0, 241)[1:] ** 2
    th_c, cps, p0s = [0.0], [0.0], [1.0]
    p0_inf = float(p0_over_p(M))
    for b in betas:
        tc, Mc = taylor_maccoll(M, b)
        if tc <= 0.0:
            continue
        if tc < th_c[-1]:   # past the maximum cone angle: strong branch, stop
            break
        r0 = float(normal_shock(M * math.sin(b))["p0"])
        pc = r0 * p0_inf / float(p0_over_p(Mc))
        th_c.append(tc)
        cps.append(float(cp_from_ratio(M, pc)))
        p0s.append(r0)
    th_c, cps, p0s = np.array(th_c), np.array(cps), np.array(p0s)
    return th_c, cps, p0s, float(th_c[-1])


def base_cp(M):
    """Turbulent base pressure behind a blunt base, Cp_b = -0.25/M for M > 1 (Fleeman's
    empirical fit, as used in rocket design tools), never below the vacuum limit."""
    return max(-0.25 / M, -1.0 / (M * M))
