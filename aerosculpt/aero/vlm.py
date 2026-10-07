"""Vortex-lattice method for lifting surfaces (wings, horizontal tails).

Horseshoe vortices on the cambered mean surface: bound leg on each panel's quarter chord,
trailing legs to downstream infinity along the body axis, flow tangency at the three-quarter-chord
control points (Katz & Plotkin, Low-Speed Aerodynamics, ch. 12). Chordwise panels are cosine
spaced to resolve the leading-edge camber; spanwise strips are cosine spaced to resolve the tips.

Lift comes from Kutta-Joukowski on the bound legs. Induced drag comes from the Trefftz plane:
each surface's own share from a sine-series fit of its span loading (exact for a planar wake,
Di = pi/8 rho sum n B_n^2; a discrete point-vortex Trefftz sum converges only like 1/n), the
mutual share between surfaces (wing on tail) from the discrete wake with a smoothed vortex core.

The solution is computed once per geometry for unit freestreams along body x and body z, so any
angle of attack is a linear combination: Gamma(alpha) = cos(alpha) G_x + sin(alpha) G_z.
Body frame: x aft (the flow direction), y to starboard, z up; alpha positive nose-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import naca

_FAR = 1.0e4


@dataclass
class LiftingSurface:
    """A planform with dihedral, sweep, taper and twist; one NACA section along the span.

    le_root: leading edge of the root chord (centre line), metres. span: tip to tip. Stations are
    cosine-spaced so tip loading is resolved."""
    name: str
    le_root: tuple
    span: float
    root_chord: float
    tip_chord: float
    sweep_le_deg: float = 0.0
    dihedral_deg: float = 0.0
    incidence_deg: float = 0.0
    twist_tip_deg: float = 0.0          # added at the tip, linear in span (negative = washout)
    section: dict = field(default_factory=lambda: {"family": "naca4", "m": 0.0, "p": 0.4, "t": 0.12})
    exposed_from: float = 0.0           # |y| inside which the surface is buried in a fuselage
    ns: int = 40                        # spanwise strips over the whole span (even)
    nc: int = 6                         # chordwise panels

    def area(self):
        return 0.5 * self.span * (self.root_chord + self.tip_chord)

    def mac(self):
        lam = self.tip_chord / self.root_chord
        return 2.0 / 3.0 * self.root_chord * (1 + lam + lam * lam) / (1 + lam)

    def aspect_ratio(self):
        return self.span ** 2 / self.area()

    def chord_at(self, a):
        return self.root_chord + (self.tip_chord - self.root_chord) * a

    def station(self, eta):
        """Geometry at signed span fraction eta in [-1, 1]: LE point, chord, twist (deg), and the
        section's chord and normal directions."""
        eta = np.asarray(eta, float)
        a = np.abs(eta)
        y = eta * self.span / 2
        g = math.radians(self.dihedral_deg)
        c = self.chord_at(a)
        x = self.le_root[0] + np.abs(y) * math.tan(math.radians(self.sweep_le_deg))
        yy = self.le_root[1] + y * math.cos(g)
        zz = self.le_root[2] + np.abs(y) * math.sin(g)
        tw = self.incidence_deg + self.twist_tip_deg * a
        th = np.radians(tw)
        phi = np.where(eta >= 0, g, -g)
        # chord and normal directions: pitch by the local twist, then roll by the dihedral
        ec = np.stack([np.cos(th), np.sin(phi) * np.sin(th), -np.cos(phi) * np.sin(th)], axis=-1)
        en = np.stack([np.sin(th), -np.sin(phi) * np.cos(th), np.cos(phi) * np.cos(th)], axis=-1)
        return np.stack([x, yy, zz], axis=-1), c, tw, ec, en


def _segments(P, A, B, core2=None):
    """Velocity at points P (N, 3) from unit horseshoes A->B (M, 3) with trailing legs to +x
    infinity, split into (bound, legs), each (N, M, 3). core2: squared core radius per vortex
    (M,) applied to the legs (Scully-type smoothing), or None."""
    P = P[:, None, :]
    A = A[None, :, :]
    B = B[None, :, :]
    ex = np.array([1.0, 0.0, 0.0])
    r1 = P - A
    r2 = P - B
    cr = np.cross(r1, r2)
    cr2 = np.sum(cr * cr, axis=-1)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    r0 = B - A
    ok = cr2 > 1e-10 * np.sum(r0 * r0, axis=-1) ** 2
    k = np.where(ok, (np.sum(r0 * r1, axis=-1) / np.maximum(n1, 1e-30) - np.sum(r0 * r2, axis=-1) / np.maximum(n2, 1e-30))
                 / (4 * math.pi * np.where(ok, cr2, 1.0)), 0.0)
    bound = k[..., None] * cr

    def semi(r, nr):
        # semi-infinite line from the vortex end to +x infinity
        c = np.cross(ex, r)
        d2 = np.sum(c * c, axis=-1)
        if core2 is not None:
            d2c = d2 + core2[None, :]
        else:
            d2c = d2
        kk = np.where(d2c > 1e-20, (1.0 + r[..., 0] / np.maximum(nr, 1e-30)) / (4 * math.pi * np.maximum(d2c, 1e-30)), 0.0)
        return kk[..., None] * c

    legs = semi(r2, n2) - semi(r1, n1)       # +inf -> A, A -> B, B -> +inf
    return bound, legs


class VLM:
    """Vortex lattice for several surfaces; geometry fixed, any alpha and speed afterwards."""

    def __init__(self, surfaces, s_ref, c_ref, x_ref=(0.0, 0.0, 0.0)):
        self.surfaces = list(surfaces)
        self.s_ref = float(s_ref)
        self.c_ref = float(c_ref)
        self.x_ref = np.asarray(x_ref, float)
        A, B, C, N, owner, core = [], [], [], [], [], []
        strips = []
        for si, s in enumerate(self.surfaces):
            ns = s.ns + (s.ns % 2)
            eta_e = np.sin(np.linspace(-math.pi / 2, math.pi / 2, ns + 1))   # cosine spacing, eta=0 on an edge
            eta_e[ns // 2] = 0.0
            le, ch, tw, ec, en = s.station(eta_e)
            sec = s.section
            # Lan's quasi-vortex-lattice chordwise layout: vortices and control points on interleaved
            # Chebyshev nodes, which integrates the thin-airfoil vorticity exactly for smooth camber
            k = np.arange(1, s.nc + 1)
            xa = 0.5 * (1.0 - np.cos((2 * k - 1) * math.pi / (2 * s.nc)))   # bound legs
            xc = 0.5 * (1.0 - np.cos(k * math.pi / s.nc))                  # control points
            xi_e = np.concatenate([[0.0], 0.5 * (xa[1:] + xa[:-1]), [1.0]])  # panel edges (normals)
            zc = lambda xi: naca.camber_line(xi, sec.get("family", "naca4"), sec.get("m", 0.0), sec.get("p", 0.4),
                                             sec.get("cl_i", 0.3))[0]

            def pts(xi):                                    # (stations, len(xi), 3)
                return (le[:, None, :] + xi[None, :, None] * ch[:, None, None] * ec[:, None, :]
                        + zc(xi)[None, :, None] * ch[:, None, None] * en[:, None, :])

            Pa, Pc, Pe = pts(xa), pts(xc), pts(xi_e)
            slope = naca.camber_line(xc, sec.get("family", "naca4"), sec.get("m", 0.0), sec.get("p", 0.4),
                                     sec.get("cl_i", 0.3))[1]
            for j in range(ns):
                width = float(np.linalg.norm((le[j + 1] - le[j])[1:]))
                e_c = 0.5 * (ec[j] + ec[j + 1])
                e_n = 0.5 * (en[j] + en[j + 1])
                ids = []
                for i in range(s.nc):
                    a, b = Pa[j, i], Pa[j + 1, i]
                    cp = 0.5 * (Pc[j, i] + Pc[j + 1, i])
                    n = e_n - slope[i] * e_c          # normal to the local camber line
                    n /= np.linalg.norm(n)
                    ids.append(len(A))
                    A.append(a)
                    B.append(b)
                    C.append(cp)
                    N.append(n)
                    owner.append(si)
                    core.append((0.5 * width) ** 2)
                y0 = abs(0.5 * (eta_e[j] + eta_e[j + 1])) * s.span / 2
                strips.append({"surface": si, "panels": ids, "eta": float(0.5 * (eta_e[j] + eta_e[j + 1])),
                               "y": float(0.5 * (le[j][1] + le[j + 1][1])), "y_abs": float(y0),
                               "chord": float(0.5 * (ch[j] + ch[j + 1])), "ds": width,
                               "te": 0.5 * (Pe[j, -1] + Pe[j + 1, -1]), "normal": np.array(N[ids[0]]),
                               "exposed": bool(y0 >= s.exposed_from)})
        self.A, self.B, self.C, self.N = map(np.array, (A, B, C, N))
        self.owner = np.array(owner)
        self.strips = strips
        self.strip_owner = np.array([st["surface"] for st in strips])
        core = np.array(core)
        bound, legs = _segments(self.C, self.A, self.B)
        # another surface's trailing legs passing a control point are smoothed (a tail in the wing wake)
        if len(self.surfaces) > 1:
            _, legs_c = _segments(self.C, self.A, self.B, core)
            other = self.owner[:, None] != self.owner[None, :]
            legs = np.where(other[..., None], legs_c, legs)
        aic = np.einsum("ijk,ik->ij", bound + legs, self.N)
        rhs = -np.stack([self.N[:, 0], self.N[:, 2]], axis=1)    # unit freestream along x and z
        self.G = np.linalg.solve(aic, rhs)                          # (panels, 2)
        self.l = self.B - self.A
        self.mid = 0.5 * (self.A + self.B)
        self.strip_mat = np.zeros((len(strips), len(self.A)))
        for k, st in enumerate(strips):
            self.strip_mat[k, st["panels"]] = 1.0
        # mutual induced drag between surfaces: far-wake normalwash of surface j at strips of surface i
        P = np.array([[st["te"][0] + _FAR, st["te"][1], st["te"][2]] for st in strips])
        _, legs_far = _segments(P, self.A, self.B, core)
        nrm = np.array([st["normal"] for st in strips])
        self.trefftz = np.einsum("ijk,ik->ij", legs_far, nrm)       # (strips, panels)
        self._fit = []
        for si, s in enumerate(self.surfaces):
            idx = np.flatnonzero(self.strip_owner == si)
            y = np.array([strips[k]["y"] for k in idx]) - s.le_root[1]
            b_proj = s.span * math.cos(math.radians(s.dihedral_deg))
            th = np.arccos(np.clip(-2.0 * y / b_proj, -1.0, 1.0))
            nmax = max(3, min(len(idx) // 2, 31))
            n = np.arange(1, nmax + 1)
            S = np.sin(np.outer(th, n))
            self._fit.append((idx, n, np.linalg.pinv(S)))

    def solve(self, alpha_deg, V=1.0, rho=1.0):
        """Forces at angle of attack alpha (deg) and speed V. Coefficients on s_ref."""
        a = math.radians(alpha_deg)
        vinf = V * np.array([math.cos(a), 0.0, math.sin(a)])
        gam = V * (math.cos(a) * self.G[:, 0] + math.sin(a) * self.G[:, 1])
        F = rho * gam[:, None] * np.cross(vinf[None, :], self.l)
        eL = np.array([-math.sin(a), 0.0, math.cos(a)])
        q = 0.5 * rho * V * V
        L = float(F.sum(0) @ eL)
        M = np.cross(self.mid - self.x_ref, F).sum(0)
        g_strip = self.strip_mat @ gam
        ds = np.array([st["ds"] for st in self.strips])
        # induced drag: own wake by sine series, other surfaces' wakes by the discrete Trefftz sum
        Di = 0.0
        for si, (idx, n, pinv) in enumerate(self._fit):
            Bn = pinv @ g_strip[idx]
            Di += math.pi / 8.0 * rho * float(np.sum(n * Bn * Bn))
            if len(self.surfaces) > 1:
                cols = self.owner != si
                w = self.trefftz[np.ix_(idx, np.flatnonzero(cols))] @ gam[cols]
                Di += float(-0.5 * rho * np.sum(g_strip[idx] * w * ds[idx]))
        ch = np.array([st["chord"] for st in self.strips])
        cl_strip = 2.0 * g_strip / (V * ch)
        lift_surf = np.bincount(self.owner, weights=F @ eL, minlength=len(self.surfaces))
        return {"alpha": alpha_deg, "CL": L / (q * self.s_ref), "CDi": Di / (q * self.s_ref),
                "Cm": float(M[1]) / (q * self.s_ref * self.c_ref), "L": L, "Di": Di,
                "cl_strip": cl_strip, "lift_by_surface": lift_surf}

    def lift_slope(self):
        """dCL/dalpha per radian at zero angle (linear model)."""
        return (self.solve(1.0)["CL"] - self.solve(-1.0)["CL"]) / math.radians(2.0)

    def alpha_for_cl(self, cl):
        slope = self.lift_slope()
        a = math.degrees((cl - self.solve(0.0)["CL"]) / slope)
        for _ in range(3):         # cos/sin non-linearity: a couple of Newton steps
            a += math.degrees((cl - self.solve(a)["CL"]) / slope)
        return a

    def neutral_point(self):
        """x of the aerodynamic centre (where dCm/dCL = 0), body frame."""
        r1, r2 = self.solve(-1.0), self.solve(1.0)
        dcm = (r2["Cm"] - r1["Cm"]) / (r2["CL"] - r1["CL"])
        return float(self.x_ref[0] - dcm * self.c_ref)


def helmbold_slope(ar, a0=2 * math.pi, sweep_half_chord_deg=0.0, mach=0.0):
    """DATCOM / Helmbold-Diederich finite-wing lift slope per radian."""
    beta2 = 1.0 - mach * mach
    k = a0 / (2 * math.pi)
    t = math.tan(math.radians(sweep_half_chord_deg))
    return 2 * math.pi * ar / (2 + math.sqrt(ar * ar * beta2 / (k * k) * (1 + t * t / beta2) + 4))
