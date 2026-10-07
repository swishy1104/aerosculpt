"""Parametric templates, the wing model and the optimizer's search strategy."""
import math
from pathlib import Path

import numpy as np
import pytest

from aerosculpt.aero import naca
from aerosculpt.aero.neuralfoil import Section
from aerosculpt.aero.vlm import VLM, LiftingSurface
from aerosculpt.geometry import checks, templates as T
from aerosculpt.geometry.mesh import Mesh
from aerosculpt.geometry.samples import icosphere

FIX = Path(__file__).parent / "fixtures" / "neuralfoil_ref.npz"


# ---- NeuralFoil port ---------------------------------------------------------------------------
def test_neuralfoil_port_matches_original_package():
    """Outputs of the numpy port against neuralfoil 0.3.3 (+ aerosandbox 4.2.10) on the same
    coordinates (fixture made in an isolated environment, see aero/neuralfoil.py)."""
    d = np.load(FIX)
    alpha = np.array([-4.0, 0.0, 2.0, 5.0, 8.0, 12.0])
    for key in [k for k in d.files if k.startswith("ref_")]:
        case, re = key[4:].split("_")
        xy = d["coords_" + case]
        r = Section(xy[:, 0], xy[:, 1]).polar(alpha, np.full(6, float(re)))
        ref = d[key]
        assert np.abs(r["CL"] - ref[:, 0]).max() < 5e-4
        assert np.abs(r["CD"] - ref[:, 1]).max() < 2e-5
        assert np.abs(r["CM"] - ref[:, 2]).max() < 1e-4


def test_naca_designation_and_zero_lift():
    assert naca.designation("naca4", 0.02, 0.4, 0.12)[0] == "NACA 2412"
    assert naca.designation("naca5", 0.0, 0.15, 0.12, 0.3)[0] == "NACA 23012"
    assert naca.describe("naca4", 0.023, 0.38, 0.104).startswith("≈ NACA 2410")
    # thin-airfoil theory: NACA 2412 about -2.08 deg
    assert math.degrees(naca.zero_lift_angle("naca4", 0.02, 0.4)) == pytest.approx(-2.077, abs=0.01)


# ---- vortex lattice ------------------------------------------------------------------------------
def _elliptic(span, c0, **kw):
    class Elliptic(LiftingSurface):
        def station(self, eta):
            le, c, tw, ec, en = super().station(eta)
            cc = self.root_chord * np.sqrt(np.maximum(1 - np.asarray(eta) ** 2, 0.0))
            le = le.copy()
            le[..., 0] = self.root_chord / 2 - cc / 2
            return le, cc, tw, ec, en
    return Elliptic("e", (0, 0, 0), span, c0, c0, section={"family": "naca4", "m": 0.0, "p": 0.4, "t": 0.12}, **kw)


def test_vlm_elliptic_span_efficiency_and_circular_wing():
    s = _elliptic(6.0, 4 / math.pi, ns=40)
    v = VLM([s], 6.0, 4 / math.pi)
    r = v.solve(4.0)
    assert r["CL"] ** 2 / (math.pi * 6.0 * r["CDi"]) == pytest.approx(1.0, abs=0.01)
    circ = _elliptic(1.0, 1.0, ns=60, nc=8)
    assert VLM([circ], math.pi / 4, 1.0).lift_slope() == pytest.approx(1.790, rel=0.03)   # Kinner


def test_vlm_camber_matches_thin_airfoil_theory():
    w = LiftingSurface("w", (0, 0, 0), 20.0, 1.0, 1.0, section={"family": "naca4", "m": 0.04, "p": 0.4, "t": 0.12}, ns=30)
    a0 = VLM([w], w.area(), 1.0).alpha_for_cl(0.0)
    assert a0 == pytest.approx(math.degrees(naca.zero_lift_angle("naca4", 0.04, 0.4)), abs=0.1)


# ---- templates and checks ----------------------------------------------------------------------
@pytest.mark.parametrize("tid", list(T.TEMPLATES))
def test_templates_build_watertight_and_valid(tid):
    b = T.build(tid, T.defaults(tid))
    assert not b.problems
    assert checks.check(b.mesh) is None
    assert set(np.unique(b.groups)) <= {T.BODY, T.WING, T.HTAIL, T.VTAIL, T.FINS}
    assert b.info["overall_length"] > 0 and b.info["volume"] > 0


def test_nose_profiles_and_blunting():
    for fam, _ in T.NOSE_FAMILIES:
        x, r, prob = T.nose_profile(fam, 0.3, 0.05, 0.0)
        assert prob is None and r[0] == pytest.approx(0.0) and r[-1] == pytest.approx(0.05)
        assert np.all(np.diff(x) > 0)
    x, r, prob = T.nose_profile("ogive", 0.3, 0.05, 0.01)
    assert prob is None and x[0] > 0.0          # blunting moves the tip back
    _, _, prob = T.nose_profile("conical", 0.3, 0.05, 0.2)
    assert prob                                  # too big to fit


def test_template_problems_are_reported_not_raised():
    v = T.defaults("rocket")
    v.update(nose_length=0.9, tail_length=0.5)
    assert any("longer than the body" in p for p in T.build("rocket", v).problems)


def test_self_intersection_detected():
    m = icosphere(0.5, 3)
    assert checks.check(m) is None
    v = m.v.copy()
    s = v[:, 0] > 0.3
    v[s, 0] -= 1.2
    v[s, 1:] *= 0.5
    assert checks.check(Mesh(v, m.f)) == "self-intersecting surface"
    assert "watertight" in checks.check(Mesh(m.v, m.f[:-3]))


# ---- search strategy ----------------------------------------------------------------------------
class _Space:
    """A toy space: objective is a quadratic bowl in two of three parameters plus noise-free."""

    def __init__(self):
        from aerosculpt.optimize.spaces import Var
        self.vars = [Var("a", "A", "-", 0.0, 1.0, 0.2), Var("b", "B", "-", 0.0, 1.0, 0.2), Var("c", "C", "-", 0.0, 1.0, 0.5)]
        self.base_values = {"a": 0.2, "b": 0.2, "c": 0.5}
        self.base = self.build({})

    def build(self, values, preview=False):
        from aerosculpt.optimize.spaces import Candidate
        v = dict(self.base_values)
        v.update(values)
        return Candidate(v, Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]])), None, None, {}, None)

    def fast_check(self, cand):
        return "too far" if cand.values["a"] > 0.95 else None

    def full_check(self, cand):
        return self.fast_check(cand)

    def label(self, k):
        return k.upper()


def test_search_screens_ranks_and_finds_the_optimum():
    from aerosculpt.optimize.strategy import Search
    sp = _Space()
    f = lambda v: 1.0 + (v["a"] - 0.7) ** 2 + 0.5 * (v["b"] - 0.6) ** 2 + 1e-5 * v["c"]
    s = Search(sp, lambda c, label: {"J": f(c.values)}, budget=30, top_k=2, eps_floor=0.002).run()
    ranked = [r["key"] for r in s.ranking]
    assert ranked[:2] == ["a", "b"] and s.selected == ["a", "b"]       # c has no real effect
    assert not next(r for r in s.ranking if r["key"] == "c")["significant"]
    assert s.reliable and s.best["J"] < f(sp.base_values) * 0.9
    assert s.best["cand"].values["a"] == pytest.approx(0.7, abs=0.12)


def test_search_reports_no_reliable_improvement_below_the_error():
    from aerosculpt.optimize.strategy import Search
    sp = _Space()
    noise = lambda: [1.03, 0.97]           # a solver error of about 3 %
    f = lambda v: 1.0 + 0.001 * (v["a"] - 0.2)
    s = Search(sp, lambda c, label: {"J": f(c.values)}, budget=20, noise=noise).run()
    assert not s.reliable and s.selected == []
    assert "measured error" in s.stop_reason
