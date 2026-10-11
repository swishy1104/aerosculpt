import numpy as np
import pytest

from aerosculpt.geometry import checks
from aerosculpt.geometry import features as FT
from aerosculpt.geometry.samples import SAMPLES, box, icosphere

FR = FT.Frame(1.0, np.eye(3), np.zeros(3))


def centred(key):
    m = SAMPLES[key]["build"]()
    lo, hi = m.bbox()
    return m.transformed(t=-(lo + hi) / 2)


@pytest.mark.parametrize("ftype,where", [
    ("nosecone", None), ("tailcone", None), ("pod", (0.0, 0.0, -0.2)), ("wing", (-0.1, 0.0, 0.0)),
    ("fins", (0.2, 0.0, 0.0)), ("fairing", (0.45, 0.0, 0.08)), ("slant", (0.55, 0.0, -0.03)), ("part", (0.4, 0.0, 0.0)),
])
def test_every_feature_builds_a_valid_union(ftype, where):
    m = centred("rocket")
    f = FT.suggest(ftype, m, FR, where, [0.5, 0.0, -0.85] if ftype == "slant" else None)
    out, problems, info = FT.apply(m, [f], FR)
    assert problems == []
    assert checks.check(out) is None                     # every shell closed, none passing through itself
    if ftype not in ("slant", "part"):
        assert out.hidden is not None and out.hidden.any()   # the added part's buried faces are known


def test_feature_values_are_cleaned_and_locked_by_default():
    f = FT.clean({"type": "tailcone", "center": [0, 0, 0], "values": {"length": -5, "shape": "bogus"}})
    assert f["values"]["length"] == FT.get("tailcone")["params"][1].lo and f["values"]["shape"] == "smooth"
    free = {p.key for p in FT.get("tailcone")["params"] if not FT.locked(p, f)}
    assert free == {"length", "end_ratio"}
    with pytest.raises(ValueError):
        FT.clean({"type": "fairing", "center": [0, 0, 0]})       # a region feature needs a size
    with pytest.raises(ValueError):
        FT.clean({"type": "nope", "center": [0, 0, 0]})


def test_slant_cut_is_capped_flat_and_removes_volume():
    m = box((1.0, 0.5, 0.5), 4)
    lo, hi = m.bbox()
    # a 45 degree plane through the top aft edge, moved 0.25 / sqrt(2) into the body: it removes a
    # wedge with two 0.25 legs across the full 0.5 width
    f = FT.clean({"type": "slant", "center": [hi[0], 0.0, hi[2]], "values": {"elevation": 45.0, "azimuth": 0.0, "depth": 0.25 / np.sqrt(2)}})
    out, problems, _ = FT.apply(m, [f], FR)
    assert problems == [] and checks.check(out) is None
    removed = m.signed_volume() - out.signed_volume()
    assert removed == pytest.approx(0.5 * 0.25 * 0.25 * (hi[1] - lo[1]), rel=1e-6)


def test_hidden_faces_cover_a_part_inside_a_fairing():
    body = icosphere(0.2, 2)
    inner = icosphere(0.05, 1).transformed(t=[0.0, 0.0, 0.0])
    m = type(body)(np.vstack([body.v, inner.v]), np.vstack([body.f, inner.f + len(body.v)]))
    h = FT.mark_hidden(m)
    assert h is not None and h[len(body.f):].all() and not h[:len(body.f)].any()


def test_wing_panel_stretches_span_and_keeps_the_root():
    m = centred("fixed-wing")
    lo, hi = m.bbox()
    L = hi[0] - lo[0]
    f = FT.suggest("wingpanel", m, FR, [lo[0] + 0.35 * L, 0, 0])
    f["values"]["span_scale"] = 1.2
    out, problems, _ = FT.apply(m, [f], FR)
    assert problems == [] and checks.check(out) is None
    span0, span1 = hi[1] - lo[1], np.ptp(out.v[:, 1])
    assert span1 > 1.15 * span0
    assert np.ptp(out.v[:, 0]) == pytest.approx(L, rel=1e-6)     # fuselage untouched


def test_feature_space_exposes_free_dimensions():
    from aerosculpt.optimize.spaces import FeatureSpace, FixedSpace
    m = centred("rocket")
    tail = FT.suggest("tailcone", m, FR)
    sp = FeatureSpace(FixedSpace(m), [tail], FR, {"min_volume": 0.5, "max_volume": 3.0})
    keys = [v.key for v in sp.vars]
    assert keys == [f"F.{tail['id']}.length", f"F.{tail['id']}.end_ratio"]
    v = next(v for v in sp.vars if v.key.endswith("length"))
    c_lo, c_hi = sp.build({v.key: v.lo}), sp.build({v.key: v.hi})
    assert c_hi.info["overall_length"] > c_lo.info["overall_length"]
    assert sp.full_check(c_hi) is None
    assert [c["key"] for c in sp.changes(c_hi)] == keys
