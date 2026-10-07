"""Virtual wind tunnel: a 3D lattice-Boltzmann LES around a triangle mesh.

Conventions (wind axes): flow travels in +x, z is up, y = z cross x. Drag is +x force,
lift +z, side force +y. Lattice units inside, SI at the boundary of this module.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import kernels as K
from .voxelize import build_links, voxelize_shells

FLUID, SOLID, FARFIELD, OUTLET = 0, 1, 2, 3


@dataclass
class Domain:
    shape: tuple          # (nx, ny, nz)
    dx: float             # metres per cell
    origin: np.ndarray    # physical position of cell (0, 0, 0) centre
    body_cells: tuple     # body extent in cells (x, y, z)

    def to_lattice(self, pts):
        return (np.asarray(pts, dtype=np.float64) - self.origin) / self.dx

    def to_physical(self, pts):
        return np.asarray(pts, dtype=np.float64) * self.dx + self.origin

    @property
    def n_cells(self):
        nx, ny, nz = self.shape
        return nx * ny * nz


def near_field_scale(L, W, H, frontal_area, planform_area=None):
    """Length over which a body disturbs the flow around it; tunnel margins are multiples of it.

    A sphere -> its diameter; a slender body -> its diameter; a plate facing the flow -> its
    size. Flat, wing-like airframes (span well above height) use their mean chord, planform
    area / span: the wingspan would size the tunnel for the wrong scale and waste most cells.
    """
    if planform_area and W > 1.5 * H and W > 0:
        return max(math.sqrt(max(frontal_area, 0.0)), min(L, planform_area / W))
    return max(math.sqrt(max(frontal_area, 0.0)), min(L, max(W, H)))


def plan_domain(bmin, bmax, n_cells, frontal_area=None, upstream=None, downstream=None, lateral=None,
                planform_area=None):
    """Size a tunnel around a body bounding box so that it holds about n_cells cells.

    Margins scale with the body's near-field length so blockage stays near 5% for compact
    bodies (corrected afterwards) and the wake has room to develop before the outlet sponge.
    """
    bmin = np.asarray(bmin, float)
    bmax = np.asarray(bmax, float)
    L, W, H = (bmax - bmin)
    if frontal_area is None:
        frontal_area = W * H * math.pi / 4
    Ln = near_field_scale(L, W, H, frontal_area, planform_area)
    up = upstream if upstream is not None else max(1.5 * Ln, 0.3 * L)
    down = downstream if downstream is not None else max(4.0 * Ln, 1.5 * L)
    my = mz = lateral if lateral is not None else 1.5 * Ln
    ext = np.array([up + L + down, W + 2 * my, H + 2 * mz])
    dx = (ext.prod() / n_cells) ** (1.0 / 3.0)
    shape = tuple(int(math.ceil(e / dx)) + 1 for e in ext)
    origin = np.array([bmin[0] - up, bmin[1] - my, bmin[2] - mz])
    # centre the body laterally on the grid
    slack = (np.array(shape) - 1) * dx - ext
    origin[1] -= slack[1] / 2
    origin[2] -= slack[2] / 2
    # Put a node plane on y = 0 and z = 0 (bodies arrive centred, so these are the planes of
    # symmetry, fins and wings). A sub-cell sheet there becomes one centred layer of cells;
    # between two planes it would have to be two cells thick or lean to one side.
    origin[1:] = np.round(origin[1:] / dx) * dx
    body_cells = tuple(float(v) for v in (bmax - bmin) / dx)
    return Domain(shape=shape, dx=dx, origin=origin, body_cells=body_cells)


@dataclass
class FlowSettings:
    u_lat: float = 0.08           # inlet speed in lattice units (Mach ~0.14)
    re_sim: float = 100.0         # Reynolds number actually simulated
    l_ref_cells: float = 1.0      # length (cells) the Reynolds number is based on
    smagorinsky: float = 0.14     # Smagorinsky constant (0 disables the LES model)
    tau_bulk: float = 1.0         # relaxation time of the stress trace (bulk viscosity, damps sound)
    sponge_frac: float = 0.18     # outlet sponge length as a fraction of nx
    sponge_tau: float = 1.2       # relaxation time reached at the outlet
    ramp_steps: int = 600         # inlet velocity ramp to avoid an impulsive start
    side_sponge_frac: float = 0.10  # acoustic sponge width at inlet and side walls
    side_sponge_tau: float = 0.56   # relaxation time reached at the inlet / side walls

    @property
    def nu_lat(self):
        return self.u_lat * self.l_ref_cells / self.re_sim

    @property
    def tau0(self):
        return 3.0 * self.nu_lat + 0.5


@dataclass
class RunStats:
    steps: int = 0
    seconds: float = 0.0
    history: list = field(default_factory=list)   # (step, Fx, Fy, Fz) lattice units

    @property
    def mlups(self):
        return 0.0 if self.seconds == 0 else self.steps / self.seconds


class Tunnel:
    """Holds the lattice and runs the flow. Geometry can be swapped for warm starts."""

    def __init__(self, domain: Domain, flow: FlowSettings):
        self.domain = domain
        self.flow = flow
        nx, ny, nz = domain.shape
        self.f = np.empty((19, nx, ny, nz), dtype=np.float32)
        self.fn = np.empty_like(self.f)
        self.flags = np.zeros((nx, ny, nz), dtype=np.uint8)
        self._boundary_flags()
        self.bidx = np.full((nx, ny, nz), -1, dtype=np.int32)
        self.qtab = np.zeros((0, 19), dtype=np.float32)
        self.tau_x, self.tau_y, self.tau_z = self._tau_profile()
        self.step_count = 0
        feq0 = K.feq_vector(1.0, 0.0, 0.0, 0.0)
        K.fill_equilibrium(self.f, self.flags, feq0, False, self.flags)
        self.fn[:] = self.f
        self.fpart = np.zeros((nx, 6), dtype=np.float64)
        self.ref = np.zeros(3)
        self.acc = None
        self.acc_n = 0
        self.link_stats = {}
        self.force_idx = None
        self.force = None

    # ---- setup -----------------------------------------------------------------
    def _boundary_flags(self):
        fl = self.flags
        fl[:, 0, :] = FARFIELD
        fl[:, -1, :] = FARFIELD
        fl[:, :, 0] = FARFIELD
        fl[:, :, -1] = FARFIELD
        fl[-1, :, :] = OUTLET
        fl[0, :, :] = FARFIELD

    def _tau_profile(self):
        """Absorbing sponge layers: the relaxation time rises smoothly towards the outlet
        (to dissipate the wake) and, more gently, towards the inlet and the side walls (to
        absorb the acoustic waves that a nearly inviscid lattice otherwise keeps reflecting)."""
        f = self.flow
        nx, ny, nz = self.domain.shape

        def ramp(n, frac, tau_end, both_ends=False, only_start=False):
            t = np.full(n, f.tau0, dtype=np.float64)
            w = max(2, int(round(n * frac)))
            for k in range(w):
                s = (w - k) / w  # 1 at the boundary, 0 inside
                val = f.tau0 + (tau_end - f.tau0) * s * s
                if both_ends or only_start:
                    t[k] = max(t[k], val)
                if both_ends or not only_start:
                    t[n - 1 - k] = max(t[n - 1 - k], val)
            return t

        tx = ramp(nx, f.sponge_frac, f.sponge_tau)
        tx = np.maximum(tx, ramp(nx, f.side_sponge_frac, f.side_sponge_tau, only_start=True))
        ty = ramp(ny, f.side_sponge_frac, f.side_sponge_tau, both_ends=True)
        tz = ramp(nz, f.side_sponge_frac, f.side_sponge_tau, both_ends=True)
        return tx, ty, tz

    def set_geometry(self, mesh, ref_point=None):
        """Voxelize a Mesh given in wind axes and metres. Safe to call again for warm starts."""
        from ..geometry.mesh import shell_info
        comp, closed = shell_info(mesh)
        tri = self.domain.to_lattice(mesh.triangles().reshape(-1, 3)).reshape(-1, 3, 3)
        solid, vstats = voxelize_shells(tri, comp, closed, self.domain.shape)
        # never let the body touch the tunnel boundary planes
        solid[:2] = False
        solid[-2:] = False
        solid[:, :2] = False
        solid[:, -2:] = False
        solid[:, :, :2] = False
        solid[:, :, -2:] = False
        prev = self.flags.copy()
        interior = (self.flags == FLUID) | (self.flags == SOLID)
        self.flags[interior] = FLUID
        self.flags[solid & interior] = SOLID
        self.bidx, self.qtab, self.link_stats = build_links(self.flags == SOLID, tri)
        self.link_stats.update(vstats)
        # cells uncovered by a shape change start from rest
        feq0 = K.feq_vector(1.0, 0.0, 0.0, 0.0)
        K.fill_equilibrium(self.f, self.flags, feq0, True, prev)
        if ref_point is None:
            ref_point = tri.reshape(-1, 3).mean(axis=0)
        else:
            ref_point = self.domain.to_lattice(ref_point)
        self.ref = np.asarray(ref_point, dtype=np.float64)
        self.solid_cells = int((self.flags == SOLID).sum())
        return self.solid_cells

    def set_forcing(self, idx=None, F=None):
        """Body forces (lattice units, per step) on fluid cells, e.g. propeller actuator disks.
        Cells that fall inside the body are dropped and the rest scaled to keep the total."""
        if idx is None or len(idx) == 0:
            self.force_idx = self.force = None
            return
        ok = self.flags[idx[:, 0], idx[:, 1], idx[:, 2]] == FLUID
        if not ok.any():
            self.force_idx = self.force = None
            return
        F = np.asarray(F, dtype=np.float64)
        scale = np.abs(F).sum() / max(np.abs(F[ok]).sum(), 1e-30)
        self.force_idx = np.ascontiguousarray(idx[ok], dtype=np.int32)
        self.force = np.ascontiguousarray(F[ok] * scale)

    def set_flow(self, flow: FlowSettings):
        """Switch the simulated Reynolds number (or other settings) on the same grid."""
        self.flow = flow
        self.tau_x, self.tau_y, self.tau_z = self._tau_profile()

    def snapshot(self):
        """Compact copy of the flow state (half precision) for a later warm start."""
        return {"f": self.f.astype(np.float16), "flags": self.flags.copy()}

    def restore(self, snap):
        """Continue from a snapshot; call set_geometry afterwards for the new shape."""
        self.f[:] = snap["f"]
        self.fn[:] = self.f
        self.flags[:] = snap["flags"]

    # ---- time stepping ---------------------------------------------------------
    def inlet_speed(self):
        r = self.flow.ramp_steps
        if r <= 0 or self.step_count >= r:
            return self.flow.u_lat
        s = self.step_count / r
        return self.flow.u_lat * (3 * s * s - 2 * s * s * s)

    def step(self, n=1, record_every=10, stats: RunStats | None = None):
        smag = 18.0 * math.sqrt(2.0) * self.flow.smagorinsky ** 2
        t0 = time.perf_counter()
        for _ in range(n):
            feq = K.feq_vector(1.0, self.inlet_speed(), 0.0, 0.0)
            K.stream_collide(self.f, self.fn, self.flags, self.bidx, self.qtab,
                             self.tau_x, self.tau_y, self.tau_z, feq, smag, self.flow.tau_bulk,
                             self.ref, self.fpart)
            K.outlet_copy(self.fn, self.flags)
            if self.force_idx is not None:
                from ..rc.actuator import apply_forcing
                ramp = (self.inlet_speed() / self.flow.u_lat) ** 2
                apply_forcing(self.fn, self.force_idx, self.force, ramp)
            self.f, self.fn = self.fn, self.f
            self.step_count += 1
            if stats is not None and self.step_count % record_every == 0:
                F = self.fpart.sum(axis=0)
                stats.history.append((self.step_count, F[0], F[1], F[2], F[3], F[4], F[5]))
            if self.acc is not None and self.step_count % self.acc_every == 0:
                K.accumulate(self.f, self.flags, self.acc)
                self.acc_n += 1
        if stats is not None:
            stats.steps += n
            stats.seconds += time.perf_counter() - t0
        return self.fpart.sum(axis=0)

    def start_averaging(self, every=4):
        nx, ny, nz = self.domain.shape
        self.acc = np.zeros((4, nx, ny, nz), dtype=np.float32)
        self.acc_n = 0
        self.acc_every = every

    def mean_fields(self):
        """Time-averaged (rho, ux, uy, uz) in lattice units; solid cells are NaN."""
        if self.acc is None or self.acc_n == 0:
            raise RuntimeError("averaging was not started")
        m = self.acc / self.acc_n
        m[:, self.flags == SOLID] = np.nan
        return m

    def is_finite(self):
        return bool(np.isfinite(self.fpart).all()) and bool(np.isfinite(self.f[0, ::7, ::7, ::7]).all())


def converge(tunnel: Tunnel, stats: RunStats, chunk: int, min_chunks: int, max_chunks: int,
             avg_chunks: int, tol: float, a_ref_lat: float, progress=None):
    """Run in chunks until the windowed mean drag coefficient settles.

    Convergence is judged on the objective (Cd), not on residuals: the mean of the last
    `avg_chunks` chunks must move by less than `tol` (relative) between successive checks.
    Returns (converged, mean force vector, std of Cd over the window).
    """
    qA = 0.5 * tunnel.flow.u_lat ** 2 * a_ref_lat
    means = []
    chunk_means = []
    converged = False
    for c in range(max_chunks):
        n0 = len(stats.history)
        tunnel.step(chunk, stats=stats)
        if not tunnel.is_finite():
            raise FloatingPointError("flow solution diverged")
        h = np.array(stats.history[n0:])
        chunk_means.append(h[:, 1:7].mean(axis=0))
        if progress:
            progress(c + 1, max_chunks, chunk_means[-1][0] / qA)
        if c + 1 >= min_chunks and len(chunk_means) >= avg_chunks:
            m = np.mean(chunk_means[-avg_chunks:], axis=0)[0] / qA
            means.append(m)
            if len(means) >= 2 and abs(means[-1] - means[-2]) <= tol * max(abs(means[-1]), 1e-6):
                converged = True
                break
    window = np.array(chunk_means[-avg_chunks:])
    F = window.mean(axis=0)
    cd_std = float(window[:, 0].std() / qA) if len(window) > 1 else 0.0
    return converged, F, cd_std
