"""Section aerodynamics (lift, drag, moment, transition) from NeuralFoil, in plain numpy.

NeuralFoil (Peter Sharpe, MIT licence, nf_weights/LICENSE-NeuralFoil.txt) is a neural network
trained on XFoil viscous analyses of tens of millions of airfoil cases; it agrees with XFoil to
within a few percent and evaluates in microseconds. The published package depends on AeroSandbox
and CasADi, which would pull a newer numpy than numba supports, so this module re-implements its
two small pieces of glue -- coordinate normalisation and the 8-weight Kulfan (CST) fit -- and the
network evaluation itself, from neuralfoil 0.3.3 / aerosandbox 4.2.10. Outputs are checked
against the original package in tests/test_aero.py.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import comb

from . import naca

_DIR = Path(__file__).resolve().parent / "nf_weights"
MODEL = "xlarge"
_N_W = 8


MODELS = ("large", "xlarge")      # the optimizer explores with the smaller, confirms with the larger


def _layers(size):
    p = dict(np.load(_DIR / f"nn-{size}.npz"))
    idx = sorted({int(k.split(".")[1]) for k in p})
    layers = [(p[f"net.{i}.weight"], p[f"net.{i}.bias"]) for i in idx]
    w, b = layers[-1]
    layers[-1] = (w[:6], b[:6])          # only the force outputs are used here
    return layers


_NETS = {m: _layers(m) for m in MODELS}
_d = dict(np.load(_DIR / "scaled_input_distribution.npz"))
_MEAN, _INV_COV = _d["mean_inputs_scaled"], _d["inv_cov_inputs_scaled"]
_N_IN = len(_MEAN)
_LN_EPS = math.log(10 / np.finfo(np.float64).max)


def _net(x, size=MODEL):
    h = x.T
    layers = _NETS[size]
    last = len(layers) - 1
    for i, (w, b) in enumerate(layers):
        h = w @ h + b[:, None]
        if i != last:
            h = h / (1.0 + np.exp(-h))       # swish
    y = h.T
    dm = x - _MEAN[None, :]
    y[:, 0] -= np.sum(dm @ _INV_COV * dm, axis=1) / (2 * _N_IN)
    return y


def normalize(x, y):
    """LE to the origin, TE to (1, 0), unit chord (aerosandbox Airfoil.normalize)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xte, yte = (x[0] + x[-1]) / 2, (y[0] + y[-1]) / 2
    dist = np.hypot(x - xte, y - yte)
    i = int(np.argmax(dist))
    tx, ty = -x[i], -y[i]
    s = 1.0 / dist[i]
    xs, ys = (x + tx) * s, (y + ty) * s
    xte, yte = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
    rot = -math.atan2(yte, xte)
    c, sn = math.cos(rot), math.sin(rot)
    xr, yr = c * xs - sn * ys, sn * xs + c * ys
    return xr, yr, {"x_translation": tx, "y_translation": ty, "scale": s, "rotation_deg": math.degrees(rot)}


def kulfan_fit(x, y, n=_N_W, N1=0.5, N2=1.0):
    """Least-squares CST weights with leading-edge modification on normalised coordinates
    (aerosandbox get_kulfan_parameters, method='least_squares')."""
    le = int(np.argmin(x))
    upper = np.arange(len(x)) <= le
    C = np.maximum(x, 0.0) ** N1 * np.maximum(1 - x, 0.0) ** N2
    k = np.arange(n)
    S = comb(n - 1, k)[:, None] * x[None, :] ** k[:, None] * (1 - x[None, :]) ** (n - 1 - k)[:, None]
    CS = (C[None, :] * S).T
    A = np.concatenate([np.where(upper[:, None], 0.0, CS), np.where(upper[:, None], CS, 0.0),
                        (x * np.maximum(1 - x, 0.0) ** (n + 0.5))[:, None],
                        np.where(upper, x / 2, -x / 2)[:, None]], axis=1)
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    te = sol[-1]
    if te < 0:
        sol = np.linalg.lstsq(A[:, :-1], y, rcond=None)[0]
        sol = np.append(sol, 0.0)
        te = 0.0
    return {"lower_weights": sol[:n], "upper_weights": sol[n:2 * n], "leading_edge_weight": float(sol[2 * n]),
            "TE_thickness": float(te)}


def from_kulfan(kp, alpha, re, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0, size=MODEL):
    """NeuralFoil on Kulfan parameters (unit chord, normalised). alpha in degrees; vectorised over
    alpha and re. Returns CL, CD, CM (quarter chord), transition locations and confidence."""
    alpha = np.atleast_1d(np.asarray(alpha, float))
    re = np.atleast_1d(np.asarray(re, float))
    n = max(len(alpha), len(re))
    alpha = np.broadcast_to(alpha, (n,))
    re = np.broadcast_to(re, (n,))
    a = np.radians(alpha)
    x = np.empty((n, 25))
    x[:, 0:8] = kp["upper_weights"]
    x[:, 8:16] = kp["lower_weights"]
    x[:, 16] = kp["leading_edge_weight"]
    x[:, 17] = kp["TE_thickness"] * 50
    x[:, 18] = np.sin(2 * a)
    x[:, 19] = np.cos(a)
    x[:, 20] = 1 - np.cos(a) ** 2
    x[:, 21] = (np.log(re) - 12.5) / 3.5
    x[:, 22] = (n_crit - 9) / 4.5
    x[:, 23] = xtr_upper
    x[:, 24] = xtr_lower
    y = _net(x, size)
    xf = x.copy()
    xf[:, :8] = -x[:, 8:16]
    xf[:, 8:16] = -x[:, :8]
    xf[:, 16] = -x[:, 16]
    xf[:, 18] = -x[:, 18]
    xf[:, 23] = x[:, 24]
    xf[:, 24] = x[:, 23]
    yf = _net(xf, size)
    yu = yf.copy()
    yu[:, 1] = -yf[:, 1]
    yu[:, 3] = -yf[:, 3]
    yu[:, 4] = yf[:, 5]
    yu[:, 5] = yf[:, 4]
    f = (y + yu) / 2
    conf = 1.0 / (1.0 + np.exp(-np.clip(f[:, 0], _LN_EPS, -_LN_EPS)))
    return {"confidence": conf, "CL": f[:, 1] / 2, "CD": np.exp((f[:, 2] - 2) * 2), "CM": f[:, 3] / 20,
            "xtr_upper": np.clip(f[:, 4], 0, 1), "xtr_lower": np.clip(f[:, 5], 0, 1)}


class Section:
    """A section shape prepared for NeuralFoil (normalised and Kulfan-fitted once)."""

    def __init__(self, x, y, size=MODEL):
        xn, yn, self.norm = normalize(x, y)
        self.kp = kulfan_fit(xn, yn)
        self.size = size

    def at_size(self, size):
        """The same section evaluated with another network size (shares the fit)."""
        if size == self.size:
            return self
        s = Section.__new__(Section)
        s.norm, s.kp, s.size = self.norm, self.kp, size
        return s

    def polar(self, alpha, re, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0):
        nm = self.norm
        r = from_kulfan(self.kp, np.asarray(alpha, float) + nm["rotation_deg"], np.asarray(re, float) / nm["scale"],
                        n_crit, xtr_upper, xtr_lower, self.size)
        d = math.radians(nm["rotation_deg"])
        xq = -nm["x_translation"] + 0.25 * (1.0 / nm["scale"] * math.cos(d)) - 0.25
        yq = -nm["y_translation"] + 0.25 * (1.0 / nm["scale"] * math.sin(-d))
        r["CM"] = r["CM"] - r["CL"] * xq + r["CD"] * yq
        return r

    def alpha_for_cl(self, cl, re, n_crit=9.0, iters=8, guess=None):
        """Section angle (deg) giving lift coefficient cl at Reynolds number re (vectorised secant)."""
        cl = np.atleast_1d(np.asarray(cl, float))
        re = np.broadcast_to(np.atleast_1d(np.asarray(re, float)), cl.shape)
        a0 = np.zeros_like(cl) if guess is None else np.broadcast_to(np.asarray(guess, float), cl.shape).copy()
        c0 = self.polar(a0, re, n_crit)["CL"]
        a1 = a0 + (cl - c0) / 0.105
        c1 = self.polar(a1, re, n_crit)["CL"]
        for _ in range(iters):
            d = c1 - c0
            step = np.where(np.abs(d) > 1e-9, (cl - c1) * (a1 - a0) / np.where(np.abs(d) > 1e-9, d, 1.0), 0.0)
            a0, c0 = a1, c1
            a1 = np.clip(a1 + np.clip(step, -4.0, 4.0), -12.0, 20.0)
            c1 = self.polar(a1, re, n_crit)["CL"]
            if np.max(np.abs(c1 - cl)) < 1e-4:
                break
        return a1

    def cl_max(self, re, n_crit=9.0):
        """Maximum section lift coefficient and its angle at Reynolds number re (scalar, cached)."""
        key = (float(f"{re:.2g}"), n_crit)
        cache = self.__dict__.setdefault("_clmax", {})
        if key not in cache:
            cache[key] = self._cl_max(key[0], n_crit)
        return cache[key]

    def _cl_max(self, re, n_crit):
        a = np.linspace(0.0, 22.0, 45)
        r = self.polar(a, np.full_like(a, float(re)), n_crit)
        i = int(np.argmax(r["CL"]))
        return float(r["CL"][i]), float(a[i])


@lru_cache(maxsize=512)
def naca_section(family="naca4", m=0.02, p=0.4, t=0.12, cl_i=0.3):
    """Section object for a NACA shape (cached on rounded parameters by the callers)."""
    x, y = naca.coordinates(family, m, p, t, cl_i, n=120, closed_te=False)
    return Section(x, y)


def section_for(sec: dict, size=MODEL) -> Section:
    return _sized(naca_section(sec.get("family", "naca4"), round(float(sec.get("m", 0.0)), 5), round(float(sec.get("p", 0.4)), 4),
                               round(float(sec.get("t", 0.12)), 5), round(float(sec.get("cl_i", 0.3)), 4)), size)


_SIZED = {}


def _sized(sec, size):
    if size == sec.size:
        return sec
    k = (id(sec), size)
    if k not in _SIZED:
        _SIZED[k] = sec.at_size(size)
    return _SIZED[k]


def _section_for_unused(sec: dict) -> Section:
    return naca_section(sec.get("family", "naca4"), round(float(sec.get("m", 0.0)), 5), round(float(sec.get("p", 0.4)), 4),
                        round(float(sec.get("t", 0.12)), 5), round(float(sec.get("cl_i", 0.3)), 4))
