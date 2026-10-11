"""Local-inclination aerodynamics and heating on a triangle mesh.

Every face is treated on its own, from its inclination to the flow:

  windward  body shells: second-order shock-expansion along each meridian (Syvertson &
            Dennis, NACA Report 1328): tangent-cone (exact Taylor-Maccoll) at the tip, then
            Prandtl-Meyer expansion at every change of slope and exponential relaxation
            towards the tangent-cone pressure along each conical frustum -- so a curved nose
            (ogive, von Karman) gets the expansion that a face-by-face tangent cone misses;
            thin, plate-like shells (wings, fins): tangent-wedge (exact oblique shock); past
            the attached-shock limit: modified Newtonian, Cp = Cp_max sin^2(theta)
  leeward   Prandtl-Meyer expansion from the freestream, floored at -1/M^2
  base      faces looking downstream: Cp_b = -0.25/M (empirical turbulent base pressure)

Edge conditions follow from the local pressure and the total pressure behind that face's
shock. Skin friction and convective heating use Eckert's reference temperature on a flat
plate with the face's running length; the nose cap uses Fay-Riddell stagnation heating
scaled by the local pressure (Lees). The wall temperature balances convection against
radiation (radiative equilibrium) unless a fixed wall temperature is given.

Not modelled: shock-shock and shock-boundary-layer interaction, shadowing of one part by
another, real-gas chemistry (perfect gas, gamma = 1.4), transonic flow (use Mach >= 1.2).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import gasdyn as gd
from .atmosphere import R_AIR, sutherland, us76

SIGMA = 5.670374e-8
CP_AIR = gd.G * R_AIR / gd.GM1
PR = 0.71
BASE_SIN = math.sin(math.radians(70.0))   # faces looking further downstream than this are base


def wind_rotation(aoa_deg, yaw_deg=0.0):
    a = math.radians(aoa_deg)
    b = math.radians(yaw_deg)
    Ra = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rb = np.array([[math.cos(b), math.sin(b), 0], [-math.sin(b), math.cos(b), 0], [0, 0, 1]])
    return Rb @ Ra


@njit(cache=True)
def _pm_nu(M):
    m2 = max(M * M - 1.0, 0.0)
    return math.sqrt(6.0) * math.atan(math.sqrt(m2 / 6.0)) - math.atan(math.sqrt(m2))


@njit(cache=True)
def _pm_mach(nu, guess):
    M = max(guess, 1.0001)
    for _ in range(40):
        m2 = M * M - 1.0
        f = _pm_nu(M) - nu
        d = math.sqrt(m2) / (M * (1.0 + 0.2 * M * M))
        step = f / max(d, 1e-9)
        M = max(M - step, 1.0001)
        if abs(step) < 1e-10:
            break
    return M


@njit(cache=True)
def _cone_state(d, th, cpt, p0t, q, p0inf):
    """Tangent-cone surface pressure (p / p_inf) and Mach number at inclination d."""
    pc = 1.0 + q * np.interp(d, th, cpt)
    p0l = p0inf * np.interp(d, th, p0t)
    Mc = math.sqrt(max(5.0 * ((max(p0l / pc, 1.0)) ** (2.0 / 7.0) - 1.0), 1.0e-6))
    return pc, Mc


@njit(cache=True)
def _B(p, M):
    return 1.4 * p * M * M / (2.0 * max(M * M - 1.0, 1e-6))


@njit(cache=True)
def _Om(M):
    return (1.0 / M) * ((1.0 + 0.2 * M * M) / 1.2) ** 3.0


@njit(cache=True)
def _relax(p, dpds, pc, h):
    """Eqs. (8)-(10): along a frustum the pressure relaxes exponentially towards pc."""
    if h <= 0.0:
        return p, dpds
    diff = pc - p
    if abs(diff) < 1e-12 or dpds / diff <= 0.0:
        return p, 0.0
    e = math.exp(-dpds * h / diff)
    return pc - diff * e, dpds * e


@njit(cache=True)
def _shock_expansion(order, starts, use, theta, r, xa, th, cpt, p0t, Minf, cp):
    """Second-order shock-expansion (Syvertson & Dennis 1958) along meridians. Faces outside
    `use` (leeward, Newtonian, base) end a meridian; the next usable face restarts it from the
    tangent cone. Overwrites cp on the faces it marches over."""
    q = 0.7 * Minf * Minf
    p0inf = (1.0 + 0.2 * Minf * Minf) ** 3.5
    for m in range(starts.shape[0] - 1):
        active = False
        p = 1.0
        M = Minf
        dpds = 0.0
        p0l = p0inf
        pc_f = 1.0
        d_prev = 0.0
        x_prev = 0.0
        r_prev = 0.0
        for k in range(starts[m], starts[m + 1]):
            i = order[k]
            if not use[i]:
                active = False
                continue
            d = theta[i]
            if not active:
                p, M = _cone_state(d, th, cpt, p0t, q, p0inf)
                p0l = p * (1.0 + 0.2 * M * M) ** 3.5
                dpds = 0.0
                pc_f = p
                d_prev, x_prev, r_prev = d, xa[i], r[i]
                active = True
                cp[i] = (p - 1.0) / q
                continue
            dx = abs(xa[i] - x_prev)
            # to the corner halfway between the two faces, along the previous frustum
            p, dpds = _relax(p, dpds, pc_f, 0.5 * dx / max(math.cos(d_prev), 0.05))
            M = math.sqrt(max(5.0 * ((max(p0l / p, 1.0)) ** (2.0 / 7.0) - 1.0), 1e-6))
            if M < 1.02:
                active = False
                continue
            pc_new, Mc_new = _cone_state(d, th, cpt, p0t, q, p0inf)
            if d < d_prev - 1e-7:
                # expansion corner: Prandtl-Meyer, then the gradient just downstream (eq. 7)
                M2 = _pm_mach(_pm_nu(M) + (d_prev - d), M)
                p2 = p * ((1.0 + 0.2 * M * M) / (1.0 + 0.2 * M2 * M2)) ** 3.5
                rc = max(0.5 * (r_prev + r[i]), 1e-9)
                B1 = _B(p, M)
                B2 = _B(p2, M2)
                O1 = _Om(M)
                O2 = _Om(M2)
                dpds = B2 / rc * (O1 / O2 * math.sin(d_prev) - math.sin(d)) + B2 / B1 * O1 / O2 * dpds
                p, M = p2, M2
            elif d > d_prev + 1e-7:
                # compression: a new tangent cone (first-order approximation)
                p, M = pc_new, Mc_new
                p0l = p * (1.0 + 0.2 * M * M) ** 3.5
                dpds = 0.0
            pc_f = pc_new
            p, dpds = _relax(p, dpds, pc_f, 0.5 * dx / max(math.cos(d), 0.05))
            cp[i] = (p - 1.0) / q
            d_prev, x_prev, r_prev = d, xa[i], r[i]


@njit(cache=True)
def _suth(T):
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


@njit(cache=True)
def _conv(Tw, pe, Te, Me2, Ve, s, turb, Taw):
    """Eckert reference-temperature flat plate: (heat flux W/m2, wall shear Pa)."""
    Ts = Te * (1.0 + 0.032 * Me2 + 0.58 * (Tw / Te - 1.0))
    Ts = max(Ts, 50.0)
    rhos = pe / (R_AIR * Ts)
    res = max(rhos * Ve * s / _suth(Ts), 1.0)
    if turb:
        cf = 0.0592 * res ** -0.2 if res < 1e7 else 0.370 * math.log10(res) ** -2.584
    else:
        cf = 0.664 / math.sqrt(res)
    q = 0.5 * cf * PR ** (-2.0 / 3.0) * rhos * Ve * CP_AIR * (Taw - Tw)
    return q, 0.5 * cf * rhos * Ve * Ve


@njit(parallel=True, cache=True)
def _wall(pe, p0e, s, cap, eps, base, T0, Tinf, stag_k, p02, fixed, tw_fixed, re_tr, trans,
          out_tw, out_q, out_tau):
    """Per face: wall temperature, convective heat flux and wall shear."""
    n = pe.shape[0]
    for i in prange(n):
        if base[i]:
            out_tw[i] = tw_fixed if fixed else Tinf
            out_q[i] = 0.0
            out_tau[i] = 0.0
            continue
        ratio = max(p0e[i] / pe[i], 1.0)
        Me2 = max(5.0 * (ratio ** (2.0 / 7.0) - 1.0), 0.0)
        Te = T0 / (1.0 + 0.2 * Me2)
        Ve = math.sqrt(Me2 * 1.4 * R_AIR * Te)
        rhoe = pe[i] / (R_AIR * Te)
        ree = rhoe * Ve * s[i] / _suth(Te)
        turb = trans == 2 or (trans == 0 and ree > re_tr)
        r = 0.892 if turb else 0.843
        Taw = Te + r * Ve * Ve / (2.0 * CP_AIR)
        hot = max(Taw, T0 if cap[i] > 0.0 else Taw)
        if fixed:
            Tw = tw_fixed
        else:
            lo, hi = min(Tinf, 200.0), hot
            for _ in range(48):          # radiative equilibrium by bisection
                Tw = 0.5 * (lo + hi)
                qc, _t = _conv(Tw, pe[i], Te, Me2, Ve, s[i], turb, Taw)
                if cap[i] > 0.0:
                    qc = stag_k * (p02 / (R_AIR * Tw) * _suth(Tw)) ** 0.1 * CP_AIR * (T0 - Tw) * cap[i]
                if qc - eps[i] * SIGMA * (Tw ** 4 - Tinf ** 4) > 0.0:
                    lo = Tw
                else:
                    hi = Tw
            Tw = 0.5 * (lo + hi)
        qc, tau = _conv(Tw, pe[i], Te, Me2, Ve, s[i], turb, Taw)
        if cap[i] > 0.0:
            qc = stag_k * (p02 / (R_AIR * Tw) * _suth(Tw)) ** 0.1 * CP_AIR * (T0 - Tw) * cap[i]
        out_tw[i] = Tw
        out_q[i] = qc
        out_tau[i] = tau


class Surface:
    """A body (wind-tunnel body frame: metres, nose towards -x, z up, centred) ready for
    repeated evaluation at different flight conditions."""

    shock_expansion = True

    def __init__(self, mesh, ref_area, ref_length, ref_point=None, t_max_k=None, emissivity=None, material_index=None):
        from ..geometry.mesh import shell_info
        tri = mesh.triangles()
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm = np.linalg.norm(cr, axis=1)
        keep = nrm > 1e-18
        if getattr(mesh, "hidden", None) is not None:    # buried inside another shell (shape features)
            keep = keep & ~mesh.hidden
        self.mesh = mesh
        self.valid = keep
        self.area = 0.5 * nrm
        self.n = cr / np.maximum(nrm, 1e-300)[:, None]
        self.c = tri.mean(axis=1)
        self.v = mesh.v
        self.tri = tri
        self.edge = np.sqrt(self.area * 2.3)          # typical edge length per face
        comp, _closed = shell_info(mesh)
        self.comp = comp
        self.n_shell = int(comp.max()) + 1 if len(comp) else 0
        self.wing = np.zeros(len(tri), bool)
        self.shell_kind = []
        for k in range(self.n_shell):
            sel = comp == k
            pts = self.c[sel]
            w = self.area[sel]
            if w.sum() <= 0:
                self.shell_kind.append("body")
                continue
            mu = (pts * w[:, None]).sum(0) / w.sum()
            cov = ((pts - mu).T * w) @ (pts - mu) / w.sum()
            evals, evecs = np.linalg.eigh(cov)
            proj = (pts - mu) @ evecs
            ext = np.sort(proj.max(0) - proj.min(0))[::-1]
            kind = "wing" if ext[2] < 0.3 * ext[1] else "body"
            self.shell_kind.append(kind)
            if kind == "wing":
                self.wing[sel] = True
        self._meridians()
        lo, hi = mesh.bbox()
        self.L = float(hi[0] - lo[0])
        self.ref_area = float(ref_area)
        self.ref_length = float(ref_length)
        self.ref_point = np.zeros(3) if ref_point is None else np.asarray(ref_point, float)
        nf = len(tri)
        self.t_max = np.full(nf, 423.15) if t_max_k is None else np.asarray(t_max_k, float)
        self.eps = np.full(nf, 0.8) if emissivity is None else np.asarray(emissivity, float)
        self.mat = np.zeros(nf, dtype=np.int64) if material_index is None else np.asarray(material_index, dtype=np.int64)

    def _meridians(self, n_az=48):
        """Faces of every body shell grouped by azimuth about the shell's long axis and ordered
        nose to tail: the meridian lines the shock-expansion method marches along."""
        order, starts = [], [0]
        n = len(self.c)
        self.r_ax = np.zeros(n)
        self.x_ax = np.zeros(n)
        for k in range(self.n_shell):
            if self.shell_kind[k] != "body":
                continue
            sel = np.nonzero(self.comp == k)[0]
            w = self.area[sel]
            if len(sel) < 8 or w.sum() <= 0:
                continue
            pts = self.c[sel]
            mu = (pts * w[:, None]).sum(0) / w.sum()
            cov = ((pts - mu).T * w) @ (pts - mu) / w.sum()
            evals, evecs = np.linalg.eigh(cov)
            ax = evecs[:, -1]
            if ax[0] < 0:
                ax = -ax
            rel = pts - mu
            xa = rel @ ax
            rho = rel - xa[:, None] * ax
            e1 = np.cross(ax, [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(ax, e1)
            phi = np.arctan2(rho @ e2, rho @ e1)
            b = np.floor((phi + math.pi) / (2 * math.pi) * n_az).astype(int) % n_az
            self.r_ax[sel] = np.linalg.norm(rho, axis=1)
            self.x_ax[sel] = xa
            for j in range(n_az):
                f = sel[b == j]
                if len(f) < 2:
                    continue
                order.append(f[np.argsort(xa[b == j], kind="stable")])
                starts.append(starts[-1] + len(f))
        self.mer_order = np.concatenate(order).astype(np.int64) if order else np.zeros(0, np.int64)
        self.mer_starts = np.array(starts, dtype=np.int64)

    # ---- geometry per flow direction ----------------------------------------------------
    def basis(self, vhat):
        a = np.array([0.0, 0.0, 1.0]) if abs(vhat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, vhat)
        e1 /= np.linalg.norm(e1)
        return e1, np.cross(vhat, e1)

    def running_length(self, vhat):
        """Distance along the flow from the leading edge of each face's shell (bodies: from the
        nose; plate-like shells: from the local leading edge of each streamwise strip)."""
        x = self.c @ vhat
        xv = self.tri @ vhat                      # (F, 3)
        xmin_f = xv.min(axis=1)
        s = np.empty(len(x))
        e1, e2 = self.basis(vhat)
        for k in range(self.n_shell):
            sel = np.nonzero(self.comp == k)[0]
            if not len(sel):
                continue
            if self.shell_kind[k] == "body":
                s[sel] = x[sel] - xmin_f[sel].min()
                continue
            y = self.c[sel] @ e1
            z = self.c[sel] @ e2
            span = max(np.ptp(y), np.ptp(z), 1e-9)
            h = span / 60.0
            iy = np.floor((y - y.min()) / h).astype(int)
            iz = np.floor((z - z.min()) / h).astype(int)
            grid = np.full((iy.max() + 3, iz.max() + 3), np.inf)
            np.minimum.at(grid, (iy + 1, iz + 1), xmin_f[sel])
            nb = grid.copy()
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = np.minimum(nb, np.roll(np.roll(grid, dy, 0), dz, 1))
            s[sel] = x[sel] - nb[iy + 1, iz + 1]
        return np.maximum(s, np.maximum(0.5 * self.edge, 1e-4 * self.L))

    def nose_radius(self, vhat):
        """Effective nose radius from the vertices nearest the most upstream point: a sphere
        cap through them gives R = (r^2 + d^2) / 2d. Sharp tips come out as small as the mesh
        allows, so they get the (very high) heating a sharp tip really sees."""
        x = self.v @ vhat
        order = np.argsort(x)
        tip = self.v[order[0]]
        best = math.inf
        for k in (8, 16, 32, 64):
            if k >= len(order):
                break
            pts = self.v[order[:k + 1]]
            d = float(x[order[k]] - x[order[0]])
            if d <= 1e-12:
                continue
            rel = pts - tip
            r = float(np.sqrt(np.maximum((rel ** 2).sum(1) - (rel @ vhat) ** 2, 0.0)).max())
            best = min(best, (r * r + d * d) / (2.0 * d))
        return max(best if math.isfinite(best) else 0.01 * self.L, 1e-4)

    # ---- one flight condition -------------------------------------------------------------
    def evaluate(self, mach, aoa, yaw=0.0, altitude=0.0, transition="auto", re_transition=1.0e6,
                 wall="radiative", t_wall_k=300.0, nose_radius=None, want_faces=False):
        M = float(mach)
        if M < 1.05:
            raise ValueError("the supersonic tunnel starts at Mach 1.05")
        atm = us76(altitude)
        Tinf, pinf, rhoinf = atm["T"], atm["p"], atm["rho"]
        V = M * atm["a"]
        qinf = 0.5 * rhoinf * V * V
        R = wind_rotation(aoa, yaw)
        vhat = R.T @ np.array([1.0, 0.0, 0.0])
        sin_t = -(self.n @ vhat)
        theta = np.arcsin(np.clip(sin_t, -1.0, 1.0))
        cpm = gd.cp_max(M)
        th_c, cp_c, p0_c, tmax_c = gd.cone_table(M)
        th_w, cp_w, p0_w, tmax_w = gd.wedge_table(M)
        p0_ns = float(gd.normal_shock(M)["p0"])

        s = self.running_length(vhat)
        Rn = float(nose_radius) if nose_radius else self.nose_radius(vhat)
        cp = np.zeros(len(theta))
        p0r = np.ones(len(theta))
        wind = sin_t > 0.0
        # a blunt nose sits behind a detached bow shock: Newtonian there, tangent-cone beyond
        nose = ~self.wing & (s <= Rn)
        cone = wind & ~self.wing & ~nose & (theta <= tmax_c)
        wedge = wind & self.wing & (theta <= tmax_w)
        newton = wind & ~cone & ~wedge
        cp[cone] = np.interp(theta[cone], th_c, cp_c)
        p0r[cone] = np.interp(theta[cone], th_c, p0_c)
        if self.shock_expansion and len(self.mer_order):
            _shock_expansion(self.mer_order, self.mer_starts, cone, theta, self.r_ax, self.x_ax,
                             np.ascontiguousarray(th_c, dtype=np.float64), np.ascontiguousarray(cp_c, dtype=np.float64),
                             np.ascontiguousarray(p0_c, dtype=np.float64), M, cp)
        cp[wedge] = np.interp(theta[wedge], th_w, cp_w)
        p0r[wedge] = np.interp(theta[wedge], th_w, p0_w)
        cp[newton] = cpm * sin_t[newton] ** 2
        p0r[newton] = p0_ns
        base = sin_t < -BASE_SIN
        lee = ~wind & ~base
        # strong expansions separate; the separated pressure is taken as the base pressure
        cp[lee] = np.maximum(gd.expansion_cp(M, -theta[lee]), gd.base_cp(M))
        cp[base] = gd.base_cp(M)
        method = np.where(cone, 0, np.where(wedge, 1, np.where(newton, 2, np.where(base, 4, 3)))).astype(np.int8)

        # edge conditions and the boundary layer
        p0inf = pinf * float(gd.p0_over_p(M))
        T0 = Tinf * (1.0 + 0.2 * M * M)
        pe = np.maximum(pinf * (1.0 + 0.5 * gd.G * M * M * cp), 1e-6 * pinf)
        p0e = np.maximum(p0inf * p0r, pe)
        p02 = p0inf * p0_ns
        rho0 = p02 / (R_AIR * T0)
        due_dx = math.sqrt(max(2.0 * (p02 - pinf) / rho0, 0.0)) / Rn
        stag_k = 0.763 * PR ** -0.6 * (rho0 * sutherland(T0)) ** 0.4 * math.sqrt(due_dx)
        cap = np.where(newton & nose, np.clip(cp / cpm, 0.0, 1.0), 0.0)
        trans = {"auto": 0, "laminar": 1, "turbulent": 2}[transition]
        fixed = wall == "fixed"
        tw = np.empty(len(cp))
        qw = np.empty(len(cp))
        tau = np.empty(len(cp))
        _wall(pe, p0e, s, cap, self.eps, base, T0, Tinf, stag_k, p02, fixed, float(t_wall_k),
              float(re_transition), trans, tw, qw, tau)
        tw[~self.valid] = Tinf
        qw[~self.valid] = 0.0

        # forces: pressure on every face, shear along the surface flow direction
        t = vhat[None, :] + sin_t[:, None] * self.n          # vhat minus its normal component
        tn = np.linalg.norm(t, axis=1)
        t = t / np.maximum(tn, 1e-12)[:, None]
        A = self.area * self.valid
        dF_p = -(cp * A)[:, None] * self.n
        dF_f = ((tau / qinf) * A)[:, None] * t
        S = self.ref_area
        Cp_b = dF_p.sum(0) / S                                # pressure force coefficient, body axes
        Cf_b = dF_f.sum(0) / S
        base_b = dF_p[base].sum(0) / S
        r = self.c - self.ref_point
        Cm_b = np.cross(r, dF_p + dF_f).sum(0) / (S * self.ref_length)
        Fw = R @ (Cp_b + Cf_b)
        Mw = R @ Cm_b
        cd_p = float((R @ Cp_b)[0])
        cd_base = float((R @ base_b)[0])
        cd_f = float((R @ Cf_b)[0])
        cd, cl, cs = float(Fw[0]), float(Fw[2]), float(Fw[1])
        cn = float((Cp_b + Cf_b)[2])
        cm = float(Mw[1])
        x_cp = None
        if abs(cn) > 1e-4:
            x_cp = float(self.ref_point[0] - Cm_b[1] * self.ref_length / cn)   # body x of the normal-force line
        tmax = self.t_max
        over = (tw > tmax) & self.valid
        out = {
            "mach": M, "aoa": float(aoa), "v": V, "q": qinf, "re": rhoinf * V * self.L / atm["mu"],
            "air": atm, "cd": cd, "cd_wave": cd_p - cd_base, "cd_base": cd_base, "cd_friction": cd_f,
            "cl": cl, "cs": cs, "cn": cn, "cm_pitch": cm, "x_cp": x_cp,
            "ld": cl / cd if cd > 1e-9 else None,
            "drag_N": cd * qinf * S, "lift_N": cl * qinf * S,
            "nose_radius": Rn, "q_stag": float(stag_k * (p02 / (R_AIR * 300.0) * sutherland(300.0)) ** 0.1
                                              * CP_AIR * (T0 - 300.0)),
            "t_stag": T0, "tw_max": float(tw[self.valid].max()) if self.valid.any() else Tinf,
            "q_max": float(qw.max()) if len(qw) else 0.0,
            "heat_load_W": float((qw * A).sum()),
            "over_limit_area": float(A[over].sum() / max(A.sum(), 1e-12)),
            "margin_K": float((tmax - tw)[self.valid].min()) if self.valid.any() else 0.0,
            "tw_max_by_material": [float(tw[self.valid & (self.mat == k)].max()) if (self.valid & (self.mat == k)).any() else None
                                   for k in range(int(self.mat.max()) + 1)],
            "methods": {k: float(A[method == i].sum() / max(A.sum(), 1e-12))
                        for i, k in enumerate(("tangent_cone", "tangent_wedge", "newtonian", "expansion", "base"))},
        }
        if want_faces:
            out["faces"] = {"cp": cp, "tw": tw, "q": qw, "method": method, "over": over}
        return out
