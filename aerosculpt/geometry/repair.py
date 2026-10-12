"""Turn a visual mesh into one watertight solid.

Models downloaded for rendering (cars, aircraft from asset sites, game exports) are made of open
panels: body shells with no inside, windows as single sheets, wheels as separate pieces, gaps at
every seam. The wind tunnel then sees a leaky shell and air flows through it, so tiny shifts of
the model change the drag by tens of percent, and no shape edit can be trusted.

`solidify` wraps such a model like shrink film:
  1. mark every grid cell a triangle passes through (all panels, open or closed)
  2. grow that surface by `gap` cells, so seams and holes narrower than about 2 x gap close
  3. flood the outside from the grid boundary: whatever the flood cannot reach is solid
     (cabin, wheel wells and any closed cavity are filled)
  4. shrink back by the same `gap` (a morphological closing: the outer shape is kept)
  5. extract the surface (marching cubes), drop stray specks and smooth it lightly
The result is a single closed shell at roughly `cells` cells along the longest side; details
smaller than a cell (antennas, panel lines) are lost, which the tunnel could not resolve anyway.
"""
from __future__ import annotations

import math

import numpy as np

from .ffd import taubin_smooth
from .mesh import Mesh


def solidify(mesh: Mesh, cells=240, gap=2, smooth=6, max_faces=None):
    """Mesh (any coordinates) -> (closed Mesh in the same coordinates, info). With max_faces, a
    result with more triangles is wrapped again on a coarser grid sized to fit."""
    out, info = _solidify(mesh, cells, gap, smooth)
    if max_faces and out.n_faces > max_faces:
        out, info = _solidify(mesh, max(40, int(cells * math.sqrt(max_faces / out.n_faces) * 0.95)), gap, smooth)
    return out, info


def _solidify(mesh, cells, gap, smooth):
    from scipy import ndimage
    from skimage.measure import marching_cubes
    from ..cfd.voxelize import surface_cells

    lo, hi = mesh.bbox()
    size = hi - lo
    h = float(size.max()) / cells
    pad = gap + 3
    shape = tuple(int(math.ceil(s / h)) + 2 * pad + 1 for s in size)
    origin = lo - pad * h
    tri = (mesh.triangles().reshape(-1, 3) - origin).reshape(-1, 3, 3) / h
    shell = surface_cells(tri, shape)
    ball = ndimage.generate_binary_structure(3, 1)
    grown = ndimage.binary_dilation(shell, structure=ball, iterations=gap) if gap > 0 else shell
    free, n = ndimage.label(~grown)
    outside = free == free[0, 0, 0]
    # shrink back from the outside only (the filled inside has no surface to erode from)
    solid = ~ndimage.binary_dilation(outside, structure=ball, iterations=gap) if gap > 0 else ~outside
    lab, nl = ndimage.label(solid)
    if nl == 0:
        raise ValueError("nothing solid was found in the model")
    kept = 1
    if nl > 1:
        vol = ndimage.sum(np.ones(lab.shape), lab, index=np.arange(1, nl + 1))
        big = np.where(vol >= 0.01 * vol.max())[0] + 1
        solid = np.isin(lab, big)
        kept = len(big)
    field = ndimage.gaussian_filter(solid.astype(np.float32), 0.6)
    verts, faces, _, _ = marching_cubes(field, level=0.5, spacing=(h, h, h), allow_degenerate=False)
    v = verts + origin
    f = faces[:, ::-1].astype(np.int64)          # outward
    if smooth > 0:
        v = taubin_smooth(v, f, iters=smooth)
    out = Mesh(v, f)
    if out.signed_volume() < 0:
        out = Mesh(v, f[:, ::-1])
    info = {"cell": h, "grid": list(shape), "faces": out.n_faces, "pieces_before": int(mesh.components().max() + 1),
            "parts_kept": kept, "volume": out.signed_volume()}
    return out, info
