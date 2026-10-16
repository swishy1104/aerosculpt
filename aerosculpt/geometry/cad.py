"""CAD exchange through the OpenCASCADE kernel that ships with gmsh.

* STEP / IGES in: the B-rep is read and tessellated into a watertight triangle mesh, sized from
  the part's curvature (fine on a nose radius, coarse on a flat side), in metres.
* STEP out: smooth solids lofted through the template's own cross-sections (true B-spline
  surfaces a CAD system can fillet, shell and dimension), or, for a reshaped mesh, a faceted
  solid sewn from its triangles.

gmsh is a process-wide singleton and not thread-safe, so every call holds one lock.
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np

from .mesh import Mesh, MeshError

CAD_EXTENSIONS = ("step", "stp", "iges", "igs")
MAX_FACETED_FACES = 60_000
_LOCK = threading.Lock()


def available():
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


class _Gmsh:
    """gmsh session: initialised without signal handlers (works off the main thread), quiet."""

    def __enter__(self):
        import gmsh
        _LOCK.acquire()
        try:
            gmsh.initialize(readConfigFiles=False, interruptible=False)
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("aerosculpt")
        except Exception:
            _LOCK.release()
            raise
        self.g = gmsh
        return gmsh

    def __exit__(self, *exc):
        try:
            self.g.finalize()
        finally:
            _LOCK.release()


def cad_to_mesh(data: bytes, filename: str, timeout: float = 180.0) -> Mesh:
    """STEP/IGES bytes -> triangle mesh in metres (all solids and surfaces of the file).

    Meshing runs in a child process with a time limit: an awkward CAD file then fails with a
    message instead of tying up the server (gmsh cannot be interrupted from another thread)."""
    import subprocess
    import sys
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext not in CAD_EXTENSIONS:
        raise MeshError(f"not a CAD file: .{ext}")
    if not available():
        raise MeshError("STEP/IGES import needs gmsh on the server (pip install gmsh)")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"model.{ext}"
        dst = Path(d) / "model.stl"
        src.write_bytes(data)
        if getattr(sys, "frozen", False):           # the desktop build has no separate interpreter
            mesh_cad_file(src, dst)
        else:
            try:
                r = subprocess.run([sys.executable, "-m", "aerosculpt.geometry.cad", str(src), str(dst)],
                                   capture_output=True, text=True, timeout=timeout,
                                   cwd=str(Path(__file__).resolve().parents[2]))
            except subprocess.TimeoutExpired:
                raise MeshError(f"meshing the CAD file took longer than {timeout:.0f} s; export it as STL "
                                "from your CAD system instead") from None
            if r.returncode != 0 or not dst.exists():
                lines = [x for x in (r.stderr or "").strip().splitlines() if x.strip()]
                raise MeshError(lines[-1] if lines else "the CAD file could not be meshed")
        from .mesh import load_mesh
        return load_mesh(dst.read_bytes(), "model.stl").orient_outward()


def mesh_cad_file(src, dst, points_per_circle: int = 20, max_size_frac: float = 1 / 60):
    """Tessellate a STEP/IGES file (in metres) into an STL file."""
    ext = str(src).lower().rsplit(".", 1)[-1]
    with _Gmsh() as gmsh:
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        for opt in ("Geometry.OCCFixDegenerated", "Geometry.OCCFixSmallEdges", "Geometry.OCCFixSmallFaces"):
            gmsh.option.setNumber(opt, 1)
        try:
            gmsh.model.occ.importShapes(str(src))
        except Exception as e:  # noqa: BLE001 - a broken file is the user's to fix
            raise MeshError(f"could not read the CAD file ({e})") from None
        gmsh.model.occ.synchronize()
        if not gmsh.model.getEntities(2):
            raise MeshError("the CAD file has no surfaces")
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        if not np.isfinite(diag) or diag <= 0:
            raise MeshError("the CAD file has no size")
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", points_per_circle)
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag * max_size_frac)
        gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 500)       # thin trailing edges must not demand millions of faces
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        err = None
        # Frontal-Delaunay first; MeshAdapt and Delaunay cope with surfaces it cannot (closed
        # periodic lofts, degenerate seams)
        for algo in (6, 1, 5):
            gmsh.option.setNumber("Mesh.Algorithm", algo)
            try:
                gmsh.model.mesh.clear()
                gmsh.model.mesh.generate(2)
                break
            except Exception as e:  # noqa: BLE001 - try the next algorithm
                err = e
        else:
            raise MeshError(f"the CAD surfaces could not be meshed ({err})")
        tags, coords, _ = gmsh.model.mesh.getNodes()
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
    tri = [np.asarray(nodes, dtype=np.int64).reshape(-1, 3) for t, nodes in zip(etypes, enodes) if t == 2]
    if not tri:
        raise MeshError("the CAD surfaces could not be meshed")
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    v = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    m = Mesh.from_triangles(v[index[np.vstack(tri)]])        # welds the faces' shared edge nodes
    Path(dst).write_bytes(m.to_stl_bytes())
    _ = ext
    return dst


def _bspline_loop(gmsh, ring):
    """Closed periodic B-spline through a ring of points -> curve loop tag."""
    occ = gmsh.model.occ
    pts = [occ.addPoint(*p) for p in ring]
    c = occ.addBSpline(pts + [pts[0]])
    return occ.addWire([c])


def lofted_solids_to_step(bodies, path, name="AeroSculpt design"):
    """Write smooth solids to STEP. bodies: list of lists of rings (each ring an (M, 3) array of
    points around one cross-section, in metres; a ring of one point is a tip), or (kind, rings)
    with kind "ruled" for flat-sided parts (polygon sections, straight walls). Each body is a
    loft (ThruSections) through its rings; all bodies are fused into one solid."""
    if not available():
        raise RuntimeError("STEP export needs gmsh on the server")
    with _Gmsh() as gmsh:
        occ = gmsh.model.occ
        vols = []
        for body in bodies:
            kind, rings = body if isinstance(body, tuple) else ("smooth", body)
            if kind == "ruled":
                wires = []
                for ring in rings:
                    pts = [occ.addPoint(*p) for p in np.asarray(ring, dtype=np.float64)]
                    lines = [occ.addLine(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
                    wires.append(occ.addWire(lines))
                out = occ.addThruSections(wires, makeSolid=True, makeRuled=True)
                vols += [t for d, t in out if d == 3]
                continue
            rings = [np.asarray(r, dtype=np.float64) for r in rings]
            full = [r for r in rings if len(r) > 1 and np.ptp(r, axis=0).max() > 1e-9]
            if len(full) < 1:
                continue
            wires = []
            for k, ring in enumerate(rings):
                if len(ring) == 1 or np.ptp(ring, axis=0).max() < 1e-9:
                    # a pointed tip: OpenCASCADE lofts through wires only, so close it with the
                    # neighbouring section shrunk to 0.2 % about the tip point
                    nb = rings[k + 1] if k + 1 < len(rings) else rings[k - 1]
                    tip = ring.mean(axis=0)
                    ring = tip + 0.002 * (nb - nb.mean(axis=0))
                wires.append(_bspline_loop(gmsh, ring))
            out = occ.addThruSections(wires, makeSolid=True, makeRuled=False)
            vols += [t for d, t in out if d == 3]
        occ.synchronize()
        if len(vols) > 1:
            fused, _ = occ.fuse([(3, vols[0])], [(3, t) for t in vols[1:]])
            occ.synchronize()
            vols = [t for d, t in fused if d == 3]
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.model.setAttribute("Name", [name]) if hasattr(gmsh.model, "setAttribute") else None
        gmsh.write(str(path))
    return path


def template_to_step(tid, values, path):
    """A template design as smooth solids (its own sections, B-spline lofts), metres, nose at x = 0."""
    from .templates import cad_sections
    bodies = []
    for kind, rings in cad_sections(tid, values):
        rings = np.asarray(rings, dtype=np.float64)
        if kind == "smooth":
            # sections collapsed to a point are tips; drop exact duplicates of consecutive sections
            keep = [0] + [k for k in range(1, len(rings)) if np.abs(rings[k] - rings[k - 1]).max() > 1e-9]
            rings = rings[keep]
            bodies.append(("smooth", [r[:1] if np.ptp(r, axis=0).max() < 1e-12 else r for r in rings]))
        else:
            bodies.append(("ruled", list(rings)))
    return lofted_solids_to_step(bodies, path, name=f"AeroSculpt {tid}")


def mesh_to_step(mesh: Mesh, path):
    """A faceted solid: every triangle becomes a planar face, sewn into one shell. Exact to the
    triangles, so a CAD system can measure and section it, though not edit its curvature."""
    if mesh.n_faces > MAX_FACETED_FACES:
        raise ValueError(f"{mesh.n_faces:,} triangles is too many for a faceted STEP "
                         f"(limit {MAX_FACETED_FACES:,}); use the STL, or the template's smooth STEP")
    if not available():
        raise RuntimeError("STEP export needs gmsh on the server")
    with _Gmsh() as gmsh:
        occ = gmsh.model.occ
        pt = [occ.addPoint(*p) for p in mesh.v]
        edges = {}

        def edge(a, b):
            k = (min(a, b), max(a, b))
            if k not in edges:
                edges[k] = occ.addLine(pt[k[0]], pt[k[1]])
            return edges[k] if a == k[0] else -edges[k]

        faces = []
        for a, b, c in mesh.f:
            loop = occ.addCurveLoop([edge(a, b), edge(b, c), edge(c, a)])
            faces.append(occ.addPlaneSurface([loop]))
        sl = occ.addSurfaceLoop(faces, sewing=True)
        occ.addVolume([sl])
        occ.synchronize()
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.write(str(path))
    return path


if __name__ == "__main__":       # python -m aerosculpt.geometry.cad <in.step> <out.stl>
    import contextlib
    import os
    import sys
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        mesh_cad_file(sys.argv[1], sys.argv[2])
