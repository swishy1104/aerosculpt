"""RC and supersonic tunnels: gas dynamics, panel methods, propellers, actuator disk, API."""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aerosculpt.geometry.samples import SAMPLES, box, icosphere, revolve
from aerosculpt.hyper import gasdyn as gd
from aerosculpt.hyper.atmosphere import us76
from aerosculpt.hyper.panel import Surface
from aerosculpt.rc.propeller import Propeller, bemt
from aerosculpt.rc.reference import UIUC


def test_us76_layers():
    assert us76(0)["T"] == pytest.approx(288.15)
    assert us76(11019)["T"] == pytest.approx(216.65, abs=0.05)      # tropopause: 11 km geopotential
    assert us76(20000)["p"] == pytest.approx(5529.3, rel=1e-3)        # US76 table, geometric altitude
    assert us76(32000)["T"] == pytest.approx(228.5, abs=0.3)
    assert us76(50000)["T"] == pytest.approx(270.65, abs=0.3)


def test_shock_tables_match_textbook_values():
    th, cp, _, tmax = gd.wedge_table(2.0)
    assert np.interp(math.radians(10), th, cp) == pytest.approx(0.2523, abs=2e-3)   # p2/p1 = 1.706
    assert math.degrees(tmax) == pytest.approx(22.97, abs=0.1)
    th, cp, _, tmax = gd.cone_table(2.0)
    assert np.interp(math.radians(10), th, cp) == pytest.approx(0.1045, abs=2e-3)   # Taylor-Maccoll
    assert math.degrees(tmax) == pytest.approx(40.6, abs=0.3)
    assert gd.cp_max(2.0) == pytest.approx(1.657, abs=2e-3)
    assert gd.expansion_cp(2.0, math.radians(10)) == pytest.approx(-0.161, abs=2e-3)


def test_sphere_and_plate_against_references():
    S = Surface(icosphere(0.5, 4), math.pi * 0.25, 1.0)
    for M in (4.0, 8.0):
        assert 0.90 < S.evaluate(M, 0.0, altitude=20000)["cd"] < 0.97          # ~0.92 hypersonic
    plate = Surface(box((1.0, 4.0, 0.01), 4), 4.0, 1.0)
    for M in (2.0, 3.0):
        r0 = plate.evaluate(M, 0.0, altitude=20000)
        r2 = plate.evaluate(M, 2.0, altitude=20000)
        cla = (r2["cl"] - r0["cl"]) / math.radians(2)
        assert cla == pytest.approx(4 / math.sqrt(M * M - 1), rel=0.03)         # Ackeret


def test_stagnation_heating_close_to_sutton_graves():
    S = Surface(icosphere(0.1, 4), math.pi * 0.01, 0.2)
    a = us76(40000)
    r = S.evaluate(8.0, 0.0, altitude=40000)
    V = 8.0 * a["a"]
    T0 = a["T"] * (1 + 0.2 * 64)
    sg = 1.7415e-4 * math.sqrt(a["rho"] / 0.1) * V ** 3 * (1 - 300.0 / T0)
    assert r["q_stag"] == pytest.approx(sg, rel=0.15)


def test_cone_centre_of_pressure_at_two_thirds():
    xs = np.linspace(0, 1, 50)
    cone = revolve(xs, xs * math.tan(math.radians(10)), 64)
    lo, hi = cone.bbox()
    cone = cone.transformed(t=-(lo + hi) / 2)
    r = Surface(cone, 0.1, 1.0).evaluate(6.0, 4.0, altitude=20000)
    assert (r["x_cp"] - (-0.5)) == pytest.approx(2 / 3, abs=0.03)


def test_bemt_out_of_sample_propeller():
    ref = UIUC["apce_9x6_6015"]        # the model was fitted on the 10x7E only
    p = Propeller.from_inches(ref["d_in"], ref["p_in"])
    n = ref["rpm"] / 60
    ct = [(bemt(p, J * n * p.diameter, ref["rpm"])["CT"], CT) for J, CT, _ in ref["data"]]
    assert np.mean([abs(a - b) for a, b in ct]) < 0.006


def test_actuator_forcing_adds_exact_momentum():
    from aerosculpt.cfd.kernels import feq_vector
    from aerosculpt.rc.actuator import apply_forcing
    f = np.empty((19, 4, 4, 4), np.float32)
    f[:] = feq_vector(1.0, 0.05, 0.0, 0.0)[:, None, None, None]
    idx = np.array([[1, 1, 1], [2, 2, 2]], np.int32)
    F = np.array([[1e-3, 2e-4, 0.0], [5e-4, 0.0, -1e-4]])
    before = f.copy()
    apply_forcing(f, idx, F, 1.0)
    from aerosculpt.cfd.kernels import CX, CY, CZ
    for k, (x, y, z) in enumerate(idx):
        d = f[:, x, y, z].astype(float) - before[:, x, y, z]
        assert abs(d.sum()) < 1e-6                                   # no mass added
        assert np.allclose([(CX * d).sum(), (CY * d).sum(), (CZ * d).sum()], F[k], atol=2e-6)


def test_supersonic_runner_end_to_end():
    from aerosculpt.hyper.runner import HyperRunner
    d = Path(tempfile.mkdtemp())
    (d / "m.stl").write_bytes(SAMPLES["hypersonic-cone"]["build"]().to_stl_bytes())
    spec = {"kind": "analyze", "tunnel": "supersonic", "model": {"units": "m", "forward": "-x", "up": "+z"},
            "envelope": {"machs": [4.0, 6.0], "aoas": [0.0, 4.0], "altitude": 35000.0},
            "thermal": {"material": "inconel"},
            "zones": [{"type": "material", "shape": "cylinder", "center": [-0.94, 0, 0], "size": [0.14, 0.1, 0.1],
                       "material": "rcc", "name": "nose"}]}
    r = HyperRunner(spec, d / "m.stl", "m.stl", d / "out").run()
    s = r["baseline"]["summary"]
    assert len(r["baseline"]["polar"]) == 2 and len(r["baseline"]["polar"][0]["speeds"]) == 2
    assert 0.03 < s["cd_at_min_aoa"] < 0.2 and s["tw_max"] > 600
    assert [m["key"] for m in r["setup"]["materials"]] == ["inconel", "rcc"]
    assert (d / "out" / "fields_baseline_1.json").exists()


def test_compressibility_corrections():
    from aerosculpt.cfd import compress as cx
    # critical Mach from the minimum pressure coefficient: sphere (potential flow) and a thin airfoil
    assert 0.53 < cx.critical_mach(-1.25) < 0.58
    assert 0.70 < cx.critical_mach(-0.43) < 0.75
    # Karman-Tsien reduces to Prandtl-Glauert for small Cp
    assert cx.kt(-0.01, 0.6) == pytest.approx(-0.01 / 0.8, rel=0.01)
    # binned correction: a uniform suction face gets more negative with Mach, a stagnation face is capped
    cp = np.array([-0.3, 1.0])
    dA = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    bins = cx.surface_bins(cp, dA, np.zeros((2, 3)), n_bins=4)
    dF, _ = cx.pressure_delta(bins, 0.5)
    assert dF[2] > 0                                   # more suction on a downward-facing area pushes it up
    assert dF[0] == pytest.approx(cx.cp_stagnation(0.5) - 1.0)   # stagnation face: more drag, capped at isentropic
    # the transonic estimate meets the supersonic value at Mach 1.2
    anchor = {"mach": [1.2, 1.3], "cdp": [0.6, 0.55]}
    assert cx.transonic_cdp(1.2, 0.8, 0.3, 0.1, anchor) == pytest.approx(0.6)
    assert cx.transonic_cdp(0.8, 0.8, 0.3, 0.1, anchor) == pytest.approx(0.3)
