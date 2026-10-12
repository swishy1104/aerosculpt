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
    assert f["values"]["length"] == next(p for p in FT.get("tailcone")["params"] if p.key == "length").lo and f["values"]["shape"] == "smooth"
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


def test_solidify_closes_a_leaky_visual_mesh():
    from aerosculpt.geometry.repair import solidify
    # a box made of six separate, unconnected panels with gaps between them: open, 6 pieces
    b = box((1.0, 0.6, 0.4), 2)
    tri = b.triangles()
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    from aerosculpt.geometry.mesh import Mesh
    panels = (tri * 0.98).reshape(-1, 3)                 # shrink: the panels no longer touch
    leaky = Mesh(panels, np.arange(len(panels)).reshape(-1, 3))
    assert checks.check(leaky) is not None
    solid, info = solidify(leaky, cells=80, gap=2)
    assert checks.check(solid) is None and solid.components().max() == 0
    assert solid.signed_volume() == pytest.approx(0.98 ** 3 * 0.24, rel=0.15)


def test_generative_proposals_are_optional_and_valid():
    m = centred("rocket")
    props = FT.propose(m, FR)
    names = [f["name"] for f in props]
    assert "Boat-tail (proposed)" in names and all(f["optional"] for f in props)
    for f in props:
        out, problems, _ = FT.apply(m, [f], FR)
        assert problems == [] and checks.check(out) is None


def test_feature_space_screens_optional_features_off_first():
    from aerosculpt.optimize.spaces import FeatureSpace, FixedSpace
    m = centred("rocket")
    bt = FT.idea("boat_tail", m, FR)
    sp = FeatureSpace(FixedSpace(m), [bt], FR, {})
    assert [v.key for v in sp.vars] == [f"F.{bt['id']}._use"]           # dimensions join once it is added
    assert sp.base.mesh.n_faces == m.n_faces                             # the baseline is the original
    on = sp.build({f"F.{bt['id']}._use": "on"})
    assert on.mesh.n_faces > m.n_faces
    new = sp.rebase(on.values)
    assert {v.key for v in new} == {f"F.{bt['id']}.length", f"F.{bt['id']}.end_ratio"}


def test_ai_assistant_reply_is_validated(monkeypatch):
    from aerosculpt import ai
    from aerosculpt.config import settings
    monkeypatch.setattr(settings, "ai_base_url", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(settings, "ai_model", "test-model")
    sent = {}

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": 'Sure! ```json {"ideas": [{"idea": "boat_tail", "why": "blunt base",'
                                                        ' "hints": {"length": 0.9, "bogus": 3}}, {"idea": "warp_drive"}],'
                                                        ' "notes": "try the tail"} ```'}}]}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json)
        return R()

    out = ai.ask("less drag", {"length_m": 1.2}, None, post=post)
    assert sent["url"].endswith("/v1/chat/completions") and sent["body"]["model"] == "test-model"
    assert out["ideas"] == [{"idea": "boat_tail", "why": "blunt base", "hints": {"length": 0.9}}]
    f = FT.idea("boat_tail", centred("rocket"), FR, out["ideas"][0]["hints"])
    assert f["optional"] and f["values"]["section"] == "match"


def test_search_adopts_an_idea_then_tunes_its_dimensions():
    from aerosculpt.optimize.spaces import FeatureSpace, FixedSpace
    from aerosculpt.optimize.strategy import Search
    m = centred("rocket")
    bt = FT.idea("boat_tail", m, FR)
    slant = FT.idea("rear_slant", m, FR)
    sp = FeatureSpace(FixedSpace(m), [bt, slant], FR, {})
    L0 = m.bbox()[1][0]

    def evaluate(cand, label):     # a stand-in objective: the boat-tail helps, more length helps a little
        tail = cand.values.get(f"F.{bt['id']}._use") == "on"
        extra = cand.mesh.bbox()[1][0] - L0
        return {"J": 1.0 - (0.1 if tail else 0.0) - 0.05 * extra, "converged": True}

    s = Search(sp, evaluate, budget=14, top_k=2, n_confirm=0, eps_floor=0.01, noise=None, confirm=None, seed=3)
    s.run()
    assert [a["key"] for a in s.adopted_all] == [f"F.{bt['id']}._use"]
    assert s.best["cand"].values[f"F.{bt['id']}._use"] == "on" and s.best["J"] < 0.9
    assert s.best["cand"].values[f"F.{slant['id']}._use"] == "off"
