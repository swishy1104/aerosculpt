"""Local grid refinement around the body: a nested ("zoom") grid.

The whole tunnel is run first on the chosen grid. A box around the body and its near wake is
then run again with the same number of cells, so about twice as fine, its faces held at the
outer run's time-averaged flow and its interior started from it; the downstream face is an
open outlet. The body's forces come from the fine grid. Refining the whole tunnel by two would
cost about 16 times as much (8x the cells, 2x the time steps); this costs about 3 times one run.

One-way coupling: the outer flow sets the box faces, nothing goes back. It is accurate when the
box faces lie in flow the body only disturbs gently, which the margins below aim for (0.6 of the
body's near-field size ahead and to the sides, 1.5 behind). Blockage is corrected with the outer
tunnel's cross-section: the box is not a tunnel.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .tunnel import GROUND_CELLS, Domain

UP, DOWN, SIDE = 0.6, 1.5, 0.6         # box margins in near-field lengths


def zoom_domain(lo, hi, ln, cells, ground_z=None):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    blo = lo - np.array([UP, SIDE, SIDE]) * ln
    bhi = hi + np.array([DOWN, SIDE, SIDE]) * ln
    if ground_z is not None:
        blo[2] = ground_z
    ext = bhi - blo
    dx = float((ext.prod() / cells) ** (1.0 / 3.0))
    if ground_z is not None:
        ext[2] += GROUND_CELLS * dx
        dx = float((ext.prod() / cells) ** (1.0 / 3.0))
        blo[2] = ground_z - (GROUND_CELLS - 0.5) * dx
    shape = tuple(int(math.ceil(e / dx)) + 1 for e in ext)
    return Domain(shape=shape, dx=dx, origin=blo, body_cells=tuple(float(v) for v in (hi - lo) / dx))


def interpolate_mean(mean, outer, inner):
    """The outer grid's mean (rho, ux, uy, uz) (lattice units, NaN in solids) at the inner grid's
    cells, by trilinear interpolation; solid cells count as rest (rho 1, u 0)."""
    from scipy.ndimage import map_coordinates
    nx, ny, nz = inner.shape
    ix = (inner.origin[0] + np.arange(nx) * inner.dx - outer.origin[0]) / outer.dx
    iy = (inner.origin[1] + np.arange(ny) * inner.dx - outer.origin[1]) / outer.dx
    iz = (inner.origin[2] + np.arange(nz) * inner.dx - outer.origin[2]) / outer.dx
    X, Y, Z = np.meshgrid(ix, iy, iz, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    out = []
    for k, rest in enumerate((1.0, 0.0, 0.0, 0.0)):
        f = np.where(np.isfinite(mean[k]), mean[k], rest).astype(np.float32)
        out.append(map_coordinates(f, coords, order=1, mode="nearest").reshape(inner.shape))
    return out


class ZoomedEvaluator:
    """Drop-in for Evaluator.run: the outer run on `outer`, then the fine box around the body."""

    def __init__(self, outer, cells=None, log=None):
        self.outer = outer
        self.fid = outer.fid
        self.cells = int(cells or outer.fid.cells)
        self.log = log or outer.log
        self.cache_hits = 0
        self.inner = None
        self.domain = outer.domain

    def info(self):
        out = dict(self.outer.info())
        if self.inner is not None:
            d = self.inner.domain
            out.update({"zoom": {"grid": list(d.shape), "cells": int(np.prod(d.shape)), "dx_m": d.dx,
                                 "body_cells": [round(v, 1) for v in d.body_cells],
                                 "refinement": round(self.outer.domain.dx / d.dx, 2)}})
        return out

    def run(self, mesh, re_sim, key=None, want_fields=True, progress=None, forcing=None, groups=None, re_flight=None):
        from .analysis import Evaluator
        ev = self.outer
        saved, ev.cache = ev.cache, None             # the outer run must really run: its field is needed
        try:
            outer = ev.run(mesh, re_sim, key=key, want_fields=False,
                           progress=(lambda f, cd: progress(0.45 * f, cd)) if progress else None,
                           forcing=forcing, groups=groups, re_flight=re_flight)
        finally:
            ev.cache = saved
        mean = ev.tunnel.mean_fields()
        lo, hi = mesh.bbox()
        ln = ev.Ln_cells * ev.domain.dx
        ground_z = None if ev.ground is None else float(ev.ground["z"])
        dom = zoom_domain(lo, hi, ln, self.cells, ground_z)
        cross = (ev.domain.shape[1] - 1) * (ev.domain.shape[2] - 1) * ev.domain.dx ** 2
        # a grid r times finer carries an r times higher Reynolds number (the fidelity levels scale
        # their caps the same way), so the box runs closer to the flight value when it was capped
        r = ev.domain.dx / dom.dx
        fid_in = dataclasses.replace(ev.fid, re_cap=ev.fid.re_cap * r)
        re_in = min(re_flight, re_sim * r) if (re_flight and re_flight > re_sim * 1.0001) else re_sim
        inner = Evaluator(lo, hi, float(np.prod((hi - lo)[1:]) * 0.8), ev.L_body,
                          ev.ref_area, ev.ref_length, fid_in, log=ev.log, cancel=ev.cancel,
                          ground=ev.ground, domain=dom)
        inner.Ln_cells = ln / dom.dx
        inner.nest = (mean, ev.domain, cross)
        inner.n_runs = 1                              # starts from the outer flow: a warm start
        self.inner = inner
        self.log(f"zoom grid {list(dom.shape)} = {int(np.prod(dom.shape)):,} cells, dx {dom.dx * 1000:.2f} mm "
                 f"({ev.domain.dx / dom.dx:.2f}x finer), body {[round(v, 1) for v in dom.body_cells]} cells")
        out = inner.run(mesh, re_in, key=None, want_fields=want_fields,
                        progress=(lambda f, cd: progress(0.45 + 0.55 * f, cd)) if progress else None,
                        forcing=forcing, groups=groups, re_flight=re_flight)
        out = dict(out)
        out["zoom"] = {"outer_cd_tunnel": outer["cd_tunnel"], "outer_cd_pressure": outer["cd_pressure"],
                       "refinement": r, "cells": int(np.prod(dom.shape)), "re_sim": re_in}
        self.domain = dom
        return out
