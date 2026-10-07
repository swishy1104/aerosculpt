import math

import numpy as np
import pytest

from aerosculpt.cfd.voxelize import voxelize_shells
from aerosculpt.geometry.ffd import Deformer, design_space, refine
from aerosculpt.geometry.mesh import (Mesh, MeshError, body_to_wind, load_mesh, metrics, parse_obj, parse_stl,
                                      projected_area, shell_info)
from aerosculpt.geometry.samples import SAMPLES, box, icosphere, merge


def test_binary_and_ascii_stl_roundtrip():
    m = icosphere(0.5, 2)
    b = m.to_stl_bytes()
    tri = parse_stl(b)
    assert tri.shape == (m.n_faces, 3, 3)
    ascii_ = "solid t\n" + "".join(
        "facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t) + "endloop\nendfacet\n"
        for t in m.triangles()) + "endsolid t\n"
    tri2 = parse_stl(ascii_.encode())
    assert np.allclose(tri, tri2, atol=1e-6)


def test_obj_polygons_are_triangulated():
    obj = b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    assert parse_obj(obj).shape == (2, 3, 3)


def test_bad_files_are_rejected():
    with pytest.raises(MeshError):
        load_mesh(b"hello world, not a mesh", "x.stl")
    with pytest.raises(MeshError):
        load_mesh(b"anything", "x.step")


def test_sphere_metrics():
    r = 0.5
    m = icosphere(r, 4)
    mt = metrics(m)
    assert mt["watertight"] and mt["shells"] == 1
    assert mt["volume"] == pytest.approx(4 / 3 * math.pi * r ** 3, rel=0.01)
    assert mt["frontal_area"] == pytest.approx(math.pi * r ** 2, rel=0.02)
    assert mt["wetted_area"] == pytest.approx(4 * math.pi * r ** 2, rel=0.01)


def test_winding_repair_recovers_volume():
    m = box((1, 2, 3), 6)
    rng = np.random.default_rng(3)
    flip = rng.random(m.n_faces) < 0.4
    m.f[flip] = m.f[flip][:, ::-1]
    m.orient_outward()
    assert m.signed_volume() == pytest.approx(6.0, rel=1e-6)


def test_body_to_wind_axes():
    # nose along +y, up along +x: the nose must end up facing upstream (-x) and up along +z
    R, _ = body_to_wind("+y", "+x")
    assert np.allclose(R @ [0, 1, 0], [-1, 0, 0])
    assert np.allclose(R @ [1, 0, 0], [0, 0, 1])
    # positive angle of attack raises the nose
    R, _ = body_to_wind("-x", "+z", aoa_deg=10)
    nose = R @ [-1, 0, 0]
    assert nose[2] > 0 and nose[0] < 0


def test_voxelize_sphere_volume():
    m = icosphere(0.5, 4)
    comp, closed = shell_info(m)
    cells_per_unit = 36                      # sphere diameter in cells
    tri = (m.triangles() + 0.5) * cells_per_unit + 2
    solid, _ = voxelize_shells(tri, comp, closed, (40, 40, 40))
    expected = 4 / 3 * math.pi * (cells_per_unit / 2) ** 3
    assert solid.sum() == pytest.approx(expected, rel=0.03)


def test_overlapping_shells_do_not_cancel():
    a = box((1, 1, 1), 2)
    b = box((1, 1, 1), 2).transformed(t=[0.5, 0, 0])
    m = merge(a, b)
    comp, closed = shell_info(m)
    assert len(closed) == 2
    tri = (m.triangles() + [0.6, 0.6, 0.6]) * 20
    solid, _ = voxelize_shells(tri, comp, closed, (40, 26, 26))
    union = 1.5 * 1 * 1 * 20 ** 3
    assert solid.sum() == pytest.approx(union, rel=0.08)


def test_open_sheet_blocks_flow():
    # a single square sheet (open shell) must still produce solid cells
    v = np.array([[0, 0, 0], [0, 10, 0], [0, 10, 10], [0, 0, 10]], float) + [5, 2, 2]
    m = Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]))
    comp, closed = shell_info(m)
    assert not closed.any()
    solid, st = voxelize_shells(m.triangles(), comp, closed, (12, 16, 16))
    assert solid[5].sum() >= 80


@pytest.mark.parametrize("name", list(SAMPLES))
def test_samples_are_closed(name):
    m = SAMPLES[name]["build"]()
    comp, closed = shell_info(m)
    assert closed.all()
    assert m.signed_volume() > 0


def test_deformer_neutral_is_identity_and_stays_closed():
    m = SAMPLES["blunt-pod"]["build"]()
    vs = design_space(["nose", "tail", "section", "length", "camber", "fairing"])
    d = Deformer(m, vs)
    out = d.apply(d.neutral())
    assert np.allclose(out.v, m.v)
    rng = np.random.default_rng(0)
    for _ in range(4):
        x = rng.random(len(vs))
        mm = d.apply(d.to_values(x))
        assert shell_info(mm)[1].all()


def test_tail_taper_reduces_volume_and_envelope_contains_designs():
    m = SAMPLES["blunt-pod"]["build"]()
    vs = design_space(["tail"])
    d = Deformer(m, vs)
    thin = d.apply([-0.4])
    assert d.volume_ratio(thin) < 0.97
    lo, hi = d.envelope()
    for x in (0.0, 0.3, 1.0):
        a, b = d.apply(d.to_values([x])).bbox()
        assert (a >= lo - 1e-9).all() and (b <= hi + 1e-9).all()


def test_refine_keeps_shape():
    m = box((1, 1, 1), 1)
    r = refine(m, 0.2)
    assert r.n_faces > m.n_faces
    assert abs(r.signed_volume()) == pytest.approx(1.0, rel=1e-9)
    assert shell_info(r)[1].all()


def test_projected_area_of_box():
    m = box((2, 1, 0.5), 2)
    assert projected_area(m, 0) == pytest.approx(0.5, rel=0.02)


def test_protect_zone_freezes_geometry_and_keep_zone_is_enforced():
    from aerosculpt.geometry.zones import Zone
    m = SAMPLES["rocket"]["build"]()
    fins = Zone("protect", "box", np.array([0.5, 0, 0]), np.array([0.13, 0.2, 0.2]), "fins")
    tube = Zone("keep", "cylinder", np.array([0.3, 0, 0]), np.array([0.2, 0.045, 0.045]), "motor tube")
    d = Deformer(m, design_space(["tail", "section"]), [fins, tube])
    mm = d.apply(d.to_values([0.0, 0.0, 0.0]))          # strongest taper and narrowest section
    frozen = fins.distance(m.v) == 0
    assert frozen.any() and np.abs(mm.v[frozen] - m.v[frozen]).max() == 0.0
    assert np.abs(mm.v - m.v).max() > 1e-3
    assert not d.feasible(mm)                             # the narrow body cuts into the motor tube
    assert d.feasible(d.apply(d.neutral()))


def test_design_zone_limits_changes_to_its_region():
    from aerosculpt.geometry.zones import Zone
    m = SAMPLES["streamlined-pod"]["build"]()
    nose_only = Zone("design", "box", np.array([-0.4, 0, 0]), np.array([0.12, 0.2, 0.2]), "nose")
    d = Deformer(m, design_space(["section", "tail"]), [nose_only])
    mm = d.apply(d.to_values([1.0, 1.0, 0.0]))
    far = nose_only.distance(m.v) > 0.3
    assert np.abs(mm.v[far] - m.v[far]).max() < 1e-12


def test_symmetry_detection():
    from aerosculpt.optimize.runner import is_z_symmetric
    for name, expected in (("rocket", True), ("streamlined-pod", True), ("wing", False), ("fixed-wing", False)):
        m = SAMPLES[name]["build"]()
        lo, hi = m.bbox()
        assert is_z_symmetric(m.transformed(t=-(lo + hi) / 2)) is expected, name
