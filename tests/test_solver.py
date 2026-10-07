import math

import numpy as np
import pytest

from aerosculpt.cfd.analysis import Conditions, Fidelity, analyze, cf_flat_plate, isa
from aerosculpt.geometry.samples import icosphere
from aerosculpt.optimize.bo import GP, propose


def test_isa_sea_level_and_tropopause():
    a = isa(0)
    assert a["rho"] == pytest.approx(1.225, rel=1e-3)
    assert a["a"] == pytest.approx(340.3, rel=1e-3)
    t = isa(11000)
    assert t["T"] == pytest.approx(216.65, rel=1e-4)


def test_flat_plate_friction_laws():
    assert cf_flat_plate(1e5) == pytest.approx(1.328 / math.sqrt(1e5))
    assert 0.002 < cf_flat_plate(1e7) < 0.004


def test_sphere_smoke_run():
    """Small, fast run at Re = 60 through the full pipeline (direct regime)."""
    m = icosphere(0.5, 3)
    re = 60.0
    speed = re * isa(0)["nu"] / 1.0
    r = analyze(m, Conditions(speed=speed), Fidelity("test", 120_000, 1e3, extra_chunks=6))
    ref = 24 / re * (1 + 0.15 * re ** 0.687)
    assert r["setup"]["regime"] == "direct"
    assert np.isfinite(r["cd"])
    # coarse grid (about 11 cells across): within 25% of the reference correlation
    assert r["cd"] == pytest.approx(ref, rel=0.25)
    f = r["fields"]
    assert len(f["streamlines"]["starts"]) > 10
    assert f["slices"]["xz"]["shape"][0] > 10


def test_gp_interpolates_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.random((14, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    gp = GP(X, y)
    Xt = rng.random((30, 2))
    mu, sd = gp.predict(Xt)
    assert np.sqrt(np.mean((mu - (np.sin(3 * Xt[:, 0]) + Xt[:, 1] ** 2)) ** 2)) < 0.15


def test_bayesian_optimization_finds_minimum():
    f = lambda x: (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2
    rng = np.random.default_rng(1)
    X = list(rng.random((4, 2)))
    Y = [f(x) for x in X]
    for _ in range(12):
        x, _, _ = propose(np.array(X), np.array(Y), lambda x: True, rng, n_candidates=800)
        X.append(x)
        Y.append(f(x))
    assert min(Y) < 0.01


def test_feasibility_is_respected():
    rng = np.random.default_rng(2)
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.2]])
    Y = np.array([1.0, 0.2, 0.6])
    for _ in range(5):
        x, _, _ = propose(X, Y, lambda x: x[0] < 0.5, rng, n_candidates=500)
        assert x[0] < 0.5


def test_speed_scaling_and_mirroring():
    from aerosculpt.cfd.analysis import at_speed, mirror_run, reynolds_groups
    air = isa(0)
    re_sim, groups = reynolds_groups([5.0, 30.0, 60.0], 1.0, air["nu"], 2.0e4)
    assert groups == [2.0e4] and re_sim == [2.0e4] * 3          # all above the cap: one shared run
    re_sim, groups = reynolds_groups([0.1, 0.2, 60.0], 1.0, air["nu"], 2.0e4)
    assert len(groups) == 3                                      # low speeds resolved directly
    run = {"regime": "hybrid", "cd_tunnel": 0.5, "cd_pressure": 0.2, "friction_integral": 10.0, "cl": 0.3, "cs": 0.0,
           "cl_pressure": 0.3, "cm_roll": 0.0, "cm_pitch": -0.1, "ref_area": 0.01}
    a, b = at_speed(run, 10.0, air, 1.0), at_speed(run, 40.0, air, 1.0)
    # drag scales with dynamic pressure (x16) times the Cd at each speed's own Reynolds number
    assert b["drag_N"] / a["drag_N"] == pytest.approx(16 * b["cd"] / a["cd"])
    assert a["cd"] == pytest.approx(0.2 + a["cf"] * 10.0)
    m = mirror_run(run)
    assert m["cl"] == -0.3 and m["cm_pitch"] == 0.1 and m["cd_pressure"] == 0.2


def _thin_wing_lattice(z_mid, dx=0.02):
    """Symmetric NACA 0012 wing, 0.2 m chord, in lattice units with its mid-plane at z_mid."""
    from aerosculpt.geometry.samples import wing
    w = wing(0.4, 0.2, 1.0, 0.0, "0012", n_span=8, n_af=40)
    lo, hi = w.bbox()
    v = (w.v - (lo + hi) / 2) / dx + np.array([8.0, 14.0, z_mid])
    return v[w.f], (int(0.2 / dx) + 17, int(0.4 / dx) + 29, 2 * int(z_mid) + 2)


def test_thin_rescue_is_order_independent_and_symmetric():
    """A sub-cell trailing edge must not lean to whichever side's triangles come first
    (that bias gave a symmetric wing CL = -0.1 at zero incidence)."""
    from aerosculpt.cfd.voxelize import voxelize_shells
    tri, shape = _thin_wing_lattice(10.5)          # mid-plane between two node planes
    comp = np.zeros(len(tri), dtype=np.int64)
    a, sa = voxelize_shells(tri, comp, [True], shape)
    b, _ = voxelize_shells(tri[::-1].copy(), comp, [True], shape)
    assert sa["thin_rescue_cells"] > 0
    assert np.array_equal(a, b)
    tri, shape = _thin_wing_lattice(10.0)          # mid-plane on a node plane, as the tunnel places it
    a, _ = voxelize_shells(tri, np.zeros(len(tri), dtype=np.int64), [True], shape)
    assert np.array_equal(a[:, :, :21], a[:, :, :21][:, :, ::-1])


def test_tunnel_has_node_planes_through_body_centre():
    from aerosculpt.cfd.tunnel import plan_domain
    for n in (300_000, 1_000_000, 777_777):
        d = plan_domain([-0.7, -1.0, -0.22], [0.7, 1.0, 0.22], n, planform_area=0.6)
        for ax in (1, 2):
            k = -d.origin[ax] / d.dx
            assert abs(k - round(k)) < 1e-9
