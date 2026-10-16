"""The wind tunnel on an NVIDIA GPU (CUDA), with the CPU tunnel as the fallback.

The lattice (populations, flags, wall links, sponge profiles, averages) lives in GPU memory;
the geometry is still voxelized on the CPU and uploaded once per shape. Only the force sums
(every `record_every` steps), snapshots for warm starts and the averaged fields at the end
come back. The kernels are generated from the same source as the CPU ones
(tools/gen_kernels.py -> kernels_cuda.py), in float32 arithmetic.

AEROSCULPT_GPU=0 forces the CPU tunnel; =1 requires the GPU; unset = use it when available.
"""
from __future__ import annotations

import logging
import math
import os
import time
import warnings

import numpy as np

from . import kernels as K
from .tunnel import SOLID, RunStats, Tunnel, wall_table

_STATE = {"checked": False, "ok": False, "why": ""}
BLOCK3 = (32, 4, 1)


def available():
    """Whether a usable CUDA GPU is present (checked once)."""
    if not _STATE["checked"]:
        _STATE["checked"] = True
        mode = os.environ.get("AEROSCULPT_GPU", "auto").strip().lower()
        if mode in ("0", "off", "no", "false"):
            _STATE["why"] = "disabled by AEROSCULPT_GPU"
            return False
        try:
            from numba import config, cuda
            config.CUDA_LOW_OCCUPANCY_WARNINGS = 0          # small helper grids are fine
            logging.getLogger("numba.cuda").setLevel(logging.WARNING)   # allocation chatter at INFO
            try:
                from numba.core.errors import NumbaPerformanceWarning
                warnings.simplefilter("ignore", NumbaPerformanceWarning)
            except ImportError:
                pass
            if not cuda.is_available():
                _STATE["why"] = "no CUDA GPU or CUDA libraries"
            else:
                from . import kernels_cuda  # noqa: F401 - compile/load once
                _STATE["ok"] = True
                _STATE["name"] = cuda.get_current_device().name.decode() if isinstance(
                    cuda.get_current_device().name, bytes) else str(cuda.get_current_device().name)
        except Exception as e:  # noqa: BLE001 - any CUDA problem means: use the CPU
            _STATE["why"] = f"{type(e).__name__}: {e}"
        if not _STATE["ok"] and mode in ("1", "on", "yes", "true"):
            raise RuntimeError(f"AEROSCULPT_GPU=1 but the GPU cannot be used ({_STATE['why']})")
    return _STATE["ok"]


def device_name():
    return _STATE.get("name") if available() else None


def make_tunnel(domain, flow, log=None):
    if available():
        try:
            return GpuTunnel(domain, flow)
        except Exception as e:  # noqa: BLE001 - out of GPU memory, driver trouble: carry on on the CPU
            if log:
                log(f"GPU tunnel unavailable ({type(e).__name__}); running on the CPU")
    return Tunnel(domain, flow)


class GpuTunnel(Tunnel):
    backend = "gpu"

    def __init__(self, domain, flow):
        from numba import cuda
        from . import kernels_cuda as G
        self.cuda, self.G = cuda, G
        super().__init__(domain, flow)
        nx, ny, nz = domain.shape
        self.grid3 = (math.ceil(nz / BLOCK3[0]), math.ceil(ny / BLOCK3[1]), nx)
        self.grid2 = (math.ceil(nz / 32), math.ceil(ny / 8))
        self.d_f = cuda.to_device(self.f)
        self.d_fn = cuda.to_device(self.fn)
        self.d_flags = cuda.to_device(self.flags)
        self.d_bidx = cuda.to_device(self.bidx)
        self.d_qtab = cuda.to_device(np.zeros((1, 19), np.float32))
        self.d_wall = cuda.to_device(self.wall)
        self.d_fpart = cuda.device_array((nx, 6), dtype=np.float32)
        self.d_ref = cuda.to_device(np.zeros(3, np.float32))
        self._upload_tau()
        self._feq_key = None
        self.d_feq = None
        self._ramp_key = None
        self.d_ramp = None
        self.d_acc = None
        self.d_force_idx = self.d_force = None

    # ---- host <-> device ----------------------------------------------------------
    def _upload_tau(self):
        c = self.cuda
        self.d_tau = tuple(c.to_device(np.ascontiguousarray(t, np.float32)) for t in (self.tau_x, self.tau_y, self.tau_z))

    def set_flow(self, flow):
        super().set_flow(flow)
        self._upload_tau()

    def set_geometry(self, mesh, ref_point=None):
        from ..geometry.mesh import shell_info
        from .voxelize import build_links
        comp, closed = shell_info(mesh)
        tri = self.domain.to_lattice(mesh.triangles().reshape(-1, 3)).reshape(-1, 3, 3)
        solid, vstats = self._voxelize(tri, comp, closed)
        prev = self.flags.copy()
        self._place_solid(solid)
        self.bidx, self.qtab, self.link_stats = build_links(self.flags == SOLID, tri)
        self.link_stats.update(vstats)
        self.wall = wall_table(self.bidx, self.qtab, self.flow, self.zg)
        c = self.cuda
        self.d_flags = c.to_device(self.flags)
        self.d_bidx = c.to_device(self.bidx)
        self.d_qtab = c.to_device(self.qtab if len(self.qtab) else np.zeros((1, 19), np.float32))
        self.d_wall = c.to_device(self.wall)
        feq0 = K.feq_vector(1.0, 0.0, 0.0, 0.0)
        self.G.fill_changed[self.grid3, BLOCK3](self.d_f, self.d_flags, c.to_device(prev), c.to_device(feq0))
        if ref_point is None:
            ref_point = tri.reshape(-1, 3).mean(axis=0)
        else:
            ref_point = self.domain.to_lattice(ref_point)
        self.ref = np.asarray(ref_point, dtype=np.float64)
        self.d_ref = c.to_device(self.ref.astype(np.float32))
        self.solid_cells = int((self.flags == SOLID).sum())
        return self.solid_cells

    def nest_from(self, mean, outer):
        super().nest_from(mean, outer)
        self.d_f = self.cuda.to_device(self.f)
        self.d_fn = self.cuda.to_device(self.fn)
        self.d_flags = self.cuda.to_device(self.flags)

    def set_forcing(self, idx=None, F=None):
        super().set_forcing(idx, F)
        if self.force_idx is None:
            self.d_force_idx = self.d_force = None
        else:
            self.d_force_idx = self.cuda.to_device(self.force_idx)
            self.d_force = self.cuda.to_device(self.force.astype(np.float32))

    def snapshot(self):
        return {"f": self.d_f.copy_to_host().astype(np.float16), "flags": self.flags.copy()}

    def restore(self, snap):
        f = np.ascontiguousarray(snap["f"], dtype=np.float32)
        self.d_f = self.cuda.to_device(f)
        self.d_fn = self.cuda.to_device(f)
        self.flags[:] = snap["flags"]
        self.d_flags = self.cuda.to_device(self.flags)

    # ---- time stepping ----------------------------------------------------------------
    def _feq(self):
        """Inlet equilibrium for this step. During the start-up ramp every step has its own, so
        the whole ramp is uploaded once and indexed (a per-step upload stalls the GPU queue)."""
        r = self.flow.ramp_steps
        if r > 0 and self.step_count < r:
            key = (r, self.flow.u_lat)
            if self._ramp_key != key:
                self._ramp_key = key
                saved = self.step_count
                table = np.empty((r, 19))
                for k in range(r):
                    self.step_count = k
                    table[k] = K.feq_vector(1.0, self.inlet_speed(), 0.0, 0.0)
                self.step_count = saved
                d = self.cuda.to_device(table)
                self.d_ramp = [d[k] for k in range(r)]     # slicing a device array costs ~60 us
            return self.d_ramp[self.step_count]
        u = self.inlet_speed()
        if self._feq_key != u:
            self._feq_key = u
            self.d_feq = self.cuda.to_device(K.feq_vector(1.0, u, 0.0, 0.0))
        return self.d_feq

    def step(self, n=1, record_every=10, stats: RunStats | None = None):
        G = self.G
        smag = np.float32(18.0 * math.sqrt(2.0) * self.flow.smagorinsky ** 2)
        tb = np.float32(self.flow.tau_bulk)
        nx = self.domain.shape[0]
        zgrid = (math.ceil(nx / 128),)
        t0 = time.perf_counter()
        for k in range(n):
            # wall forces are only read on recording steps and after the last one: zero just before
            # those, and let the atomics pile up harmlessly on the steps in between
            if k == n - 1 or (stats is not None and (self.step_count + 1) % record_every == 0):
                G.zero_rows[zgrid, 128](self.d_fpart)
            wmod, nu_w, zg, ug = self.wall_args()
            G.stream_collide[self.grid3, BLOCK3](self.d_f, self.d_fn, self.d_flags, self.d_bidx, self.d_qtab, self.d_wall,
                                                 self.d_tau[0], self.d_tau[1], self.d_tau[2], self._feq(), smag, tb,
                                                 self.d_ref, self.d_fpart, np.float32(wmod), np.float32(nu_w),
                                                 np.int32(zg), np.float32(ug))
            G.outlet_copy[self.grid2, (32, 8)](self.d_fn, self.d_flags)
            if self.d_force_idx is not None:
                ramp = np.float32((self.inlet_speed() / self.flow.u_lat) ** 2)
                G.apply_forcing[(math.ceil(len(self.force_idx) / 128),), 128](self.d_fn, self.d_force_idx, self.d_force, ramp)
            self.d_f, self.d_fn = self.d_fn, self.d_f
            self.step_count += 1
            if stats is not None and self.step_count % record_every == 0:
                self.fpart = self.d_fpart.copy_to_host().astype(np.float64)
                F = self.fpart.sum(axis=0)
                stats.history.append((self.step_count, F[0], F[1], F[2], F[3], F[4], F[5]))
            if self.d_acc is not None and self.step_count % self.acc_every == 0:
                G.accumulate[self.grid3, BLOCK3](self.d_f, self.d_flags, self.d_acc)
                self.acc_n += 1
        self.cuda.synchronize()
        self.fpart = self.d_fpart.copy_to_host().astype(np.float64)
        if stats is not None:
            stats.steps += n
            stats.seconds += time.perf_counter() - t0
        return self.fpart.sum(axis=0)

    def start_averaging(self, every=4):
        nx, ny, nz = self.domain.shape
        self.d_acc = self.cuda.to_device(np.zeros((4, nx, ny, nz), dtype=np.float32))
        self.acc = True
        self.acc_n = 0
        self.acc_every = every

    def mean_fields(self):
        if self.d_acc is None or self.acc_n == 0:
            raise RuntimeError("averaging was not started")
        m = self.d_acc.copy_to_host() / self.acc_n
        m[:, self.flags == SOLID] = np.nan
        return m

    def is_finite(self):
        if not np.isfinite(self.fpart).all():
            return False
        return bool(np.isfinite(self.d_f[0].copy_to_host()[::7, ::7, ::7]).all())
