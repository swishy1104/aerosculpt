"""NACA 4- and 5-digit sections with continuous parameters.

The formulas take any camber, camber position and thickness, not only catalogue values, so an
optimizer can move smoothly between them; `designation` names the nearest catalogue section.

  4-digit  m = max camber (fraction of chord), p = its position, t = thickness
  5-digit  cl_i = design lift coefficient, p = max-camber position (0.05-0.25), t = thickness
           (non-reflexed 2x0xx camber line; Abbott & von Doenhoff, Theory of Wing Sections)
"""
from __future__ import annotations

import math

import numpy as np

FAMILIES = ("naca4", "naca5")

# 5-digit mean line constants for design cl 0.3 (Abbott & von Doenhoff, 1959, sec. 6.5)
_P5 = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
_R5 = np.array([0.0580, 0.1260, 0.2025, 0.2900, 0.3910])
_K5 = np.array([361.40, 51.640, 15.957, 6.643, 3.230])


def thickness(x, t, closed_te=True):
    a4 = -0.1036 if closed_te else -0.1015
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)


def camber_line(x, family="naca4", m=0.02, p=0.4, cl_i=0.3):
    """Mean line y_c(x) and slope dy_c/dx on chord fractions x."""
    x = np.asarray(x, float)
    if family == "naca5":
        pp = float(np.clip(p, _P5[0], _P5[-1]))
        r = float(np.interp(pp, _P5, _R5))
        k1 = math.exp(float(np.interp(pp, _P5, np.log(_K5)))) * cl_i / 0.3
        yc = np.where(x < r, k1 / 6.0 * (x ** 3 - 3 * r * x ** 2 + r * r * (3 - r) * x), k1 * r ** 3 / 6.0 * (1 - x))
        dy = np.where(x < r, k1 / 6.0 * (3 * x ** 2 - 6 * r * x + r * r * (3 - r)), -k1 * r ** 3 / 6.0)
        return yc, dy
    if m <= 0 or p <= 0 or p >= 1:
        return np.zeros_like(x), np.zeros_like(x)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x * x), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x))
    dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dy


def coordinates(family="naca4", m=0.02, p=0.4, t=0.12, cl_i=0.3, n=60, closed_te=True, repeat_te=None):
    """Section loop in Selig order: TE -> upper surface -> LE -> lower surface, on unit chord.

    closed_te gives a sharp trailing edge (for meshing); repeat_te ends the loop on the lower TE
    point as airfoil files do (default: only when the TE is open)."""
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = thickness(x, t, closed_te)
    yc, dy = camber_line(x, family, m, p, cl_i)
    th = np.arctan(dy)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    end = None if (repeat_te if repeat_te is not None else not closed_te) else -1
    return np.concatenate([xu[::-1], xl[1:end]]), np.concatenate([yu[::-1], yl[1:end]])


def zero_lift_angle(family="naca4", m=0.02, p=0.4, cl_i=0.3, n=400):
    """Thin-airfoil-theory zero-lift angle (radians): -1/pi * int (dyc/dx)(cos th - 1) dth."""
    th = (np.arange(n) + 0.5) * math.pi / n
    x = 0.5 * (1.0 - np.cos(th))
    _, dy = camber_line(x, family, m, p, cl_i)
    return float(-np.sum(dy * (np.cos(th) - 1.0)) / n)


def designation(family="naca4", m=0.02, p=0.4, t=0.12, cl_i=0.3):
    """Nearest catalogue name and whether the section is exactly that one."""
    tt = int(round(t * 100))
    if family == "naca5":
        L = int(round(cl_i * 20 / 3))
        P = int(round(p * 20))
        name = f"NACA {L}{P}0{tt:02d}"
        exact = abs(L * 3 / 20 - cl_i) < 5e-4 and abs(P / 20 - p) < 5e-4 and abs(tt / 100 - t) < 5e-4
        return name, exact
    M = int(round(m * 100))
    P = int(round(p * 10)) if M > 0 else 0
    name = f"NACA {M}{P}{tt:02d}"
    exact = abs(M / 100 - m) < 5e-4 and (M == 0 or abs(P / 10 - p) < 5e-4) and abs(tt / 100 - t) < 5e-4
    return name, exact


def describe(family="naca4", m=0.02, p=0.4, t=0.12, cl_i=0.3):
    name, exact = designation(family, m, p, t, cl_i)
    if exact:
        return name
    if family == "naca5":
        return f"≈ {name} (design CL {cl_i:.2f}, camber at {p * 100:.0f}%, {t * 100:.1f}% thick)"
    if m <= 0:
        return f"≈ {name} (symmetric, {t * 100:.1f}% thick)"
    return f"≈ {name} ({m * 100:.1f}% camber at {p * 100:.0f}%, {t * 100:.1f}% thick)"
