"""Result cache: an identical CFD case (same geometry, grid, flow and settings) is never run twice.

The key hashes the triangle mesh as simulated (float32 vertices and faces), the tunnel grid
(shape, cell size, origin), the flow settings, fidelity, propeller forcing and per-part face
groups, plus a code version that is bumped whenever the solver changes its results.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

CODE_VERSION = "2026-10-06.1"
BUDGET_BYTES = 3 * 1024 ** 3


def _h(*parts):
    h = hashlib.sha256(CODE_VERSION.encode())
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.dtype).encode() + str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def case_key(mesh, re_sim, domain, flow, fid, forcing=None, groups=None, extra=None):
    f = ("none",) if forcing is None or forcing[0] is None else (
        np.asarray(forcing[0]), np.asarray(forcing[1], np.float32))
    return _h(mesh.v.astype(np.float32), mesh.f.astype(np.int32), round(float(re_sim), 6), tuple(domain.shape),
              round(float(domain.dx), 12), np.asarray(domain.origin, np.float64).round(12), repr(flow),
              (fid.name, fid.cells, fid.re_cap, fid.u_lat, fid.tol, fid.avg_chunks, fid.extra_chunks), *f,
              None if groups is None else np.asarray(groups, np.int8), extra)


class RunCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key):
        return self.root / key[:2] / f"{key}.pkl"

    def get(self, key, need_fields=False):
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with open(p, "rb") as fh:
                out = pickle.load(fh)
        except Exception:  # noqa: BLE001 - a damaged entry is simply recomputed
            self.misses += 1
            return None
        if need_fields and "fields" not in out:
            self.misses += 1
            return None
        os.utime(p, None)
        self.hits += 1
        return out

    def put(self, key, run):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(run, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        if np.random.default_rng().random() < 0.05:
            self.prune()

    def prune(self):
        files = sorted(self.root.rglob("*.pkl"), key=lambda q: q.stat().st_mtime)
        total = sum(q.stat().st_size for q in files)
        while files and total > BUDGET_BYTES:
            q = files.pop(0)
            total -= q.stat().st_size
            q.unlink(missing_ok=True)


def default_cache():
    from ..config import settings
    return RunCache(settings.data_dir / "cache" / "cfd")

