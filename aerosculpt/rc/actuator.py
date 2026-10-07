"""Actuator disk: a propeller's thrust and swirl as a body force in the lattice.

The radial loading from blade-element theory (dT/dr, dQ/dr) is spread over the cells of a
thin disk (a hat profile about two cells thick), then rescaled so the lattice receives exactly
the thrust and torque the propeller delivers. Forces are added with the exact-difference
method (Kupershtokh): each cell's equilibrium is shifted by the velocity change F / rho, which
adds momentum F per step and no mass.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..cfd.kernels import CX, CY, CZ, W

HALF_THICKNESS = 1.5      # cells


@njit(parallel=True, cache=True)
def apply_forcing(f, idx, F, scale):
    for k in prange(idx.shape[0]):
        x, y, z = idx[k, 0], idx[k, 1], idx[k, 2]
        rho = 0.0
        mx = 0.0
        my = 0.0
        mz = 0.0
        for i in range(19):
            v = f[i, x, y, z]
            rho += v
            mx += CX[i] * v
            my += CY[i] * v
            mz += CZ[i] * v
        ux, uy, uz = mx / rho, my / rho, mz / rho
        vx = ux + scale * F[k, 0] / rho
        vy = uy + scale * F[k, 1] / rho
        vz = uz + scale * F[k, 2] / rho
        u0 = 1.5 * (ux * ux + uy * uy + uz * uz)
        u1 = 1.5 * (vx * vx + vy * vy + vz * vz)
        for i in range(19):
            c0 = CX[i] * ux + CY[i] * uy + CZ[i] * uz
            c1 = CX[i] * vx + CY[i] * vy + CZ[i] * vz
            f[i, x, y, z] += W[i] * rho * (3.0 * (c1 - c0) + 4.5 * (c1 * c1 - c0 * c0) - (u1 - u0))


def disk_forcing(domain, center, axis, radius, hub, r_st, dT_dr, dQ_dr, spin, thrust, torque,
                 rho_phys, v_phys, u_lat):
    """Lattice cells and per-step lattice forces for one propeller.

    center: disk centre (wind frame, m); axis: unit vector the propeller pushes the air along;
    spin: +1 if the propeller turns right-handed about `axis`, -1 otherwise; thrust (N) and
    torque (N m) are the totals the disk must deliver."""
    dx = domain.dx
    c = domain.to_lattice(np.asarray(center, float))
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Rl = radius / dx
    hl = hub / dx
    ext = int(math.ceil(Rl + HALF_THICKNESS + 1))
    lo = np.maximum(np.floor(c).astype(int) - ext, 1)
    hi = np.minimum(np.floor(c).astype(int) + ext + 1, np.array(domain.shape) - 2)
    gx, gy, gz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                             np.arange(lo[2], hi[2] + 1), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    rel = pts - c
    s = rel @ axis
    radial = rel - s[:, None] * axis
    r = np.linalg.norm(radial, axis=1)
    keep = (np.abs(s) < HALF_THICKNESS) & (r <= Rl + 0.5) & (r >= max(hl - 0.5, 0.0))
    if not keep.any():
        return np.zeros((0, 3), np.int32), np.zeros((0, 3))
    pts, s, radial, r = pts[keep], s[keep], radial[keep], r[keep]
    w = 1.0 - np.abs(s) / HALF_THICKNESS
    w *= np.clip(Rl + 0.5 - r, 0.0, 1.0)                       # soft rim
    rm = np.maximum(r * dx, hub)
    area_load_T = np.interp(rm, r_st, dT_dr) / (2 * math.pi * rm)          # N / m^2
    area_load_Q = np.interp(rm, r_st, dQ_dr) / (2 * math.pi * rm * rm)     # N / m^2 (tangential)
    rhat = radial / np.maximum(r, 1e-9)[:, None]
    tdir = spin * np.cross(axis, rhat)
    Fa = area_load_T * w
    Ft = area_load_Q * w
    sa = Fa.sum()
    st = (Ft * rm).sum()
    Fa = Fa * (thrust / sa if sa > 0 else 0.0)
    Ft = Ft * (torque / st if st > 0 else 0.0)
    F = Fa[:, None] * axis + Ft[:, None] * tdir                            # N per cell
    F_lat = F / (rho_phys * v_phys * v_phys * dx * dx) * u_lat * u_lat
    return pts.astype(np.int32), F_lat
