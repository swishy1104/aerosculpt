"""Rocket methods (drag build-up, Barrowman stability, flutter, motors, flight) and the rocket
endpoints: live evaluation, motor files, OpenRocket import, apogee optimization."""
import io
import json
import zipfile

import pytest
from starlette.testclient import TestClient

from aerosculpt.geometry import templates as T
from aerosculpt.rocket.model import RocketModel
from aerosculpt.rocket.motors import BY_ID, parse_eng, parse_rse
from aerosculpt.web import app

H = {"X-Requested-With": "AeroSculpt"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _signup(c, email):
    from aerosculpt.security import auth_limiter
    auth_limiter.hits.clear()
    c.post("/api/auth/logout", headers=H)
    r = c.post("/api/auth/signup", json={"email": email, "password": "Correct-Horse-9", "name": "R", "accept_terms": True})
    assert r.status_code == 200, r.text


# ---- methods ----------------------------------------------------------------------------------
def test_rocket_methods_match_their_references():
    from aerosculpt.rocket.validation import rocket_rows
    rows = rocket_rows()
    bad = [r["id"] for r in rows if r["status"] == "fail"]
    assert not bad, bad
    by = {r["id"]: r for r in rows}
    assert abs(by["rk-integrator"]["error"]) < 0.005            # the integrator against the closed form
    assert abs(by["rk-cp-conical"]["error"]) < 0.002


def test_default_rocket_is_stable_and_its_fins_hold():
    m = RocketModel(T.defaults("rocket"))
    q = m.quick()
    assert q["margin_liftoff"] >= 1.0 and q["margin_max"] >= 1.0 and q["flutter_ok"] and q["left_rail"]
    assert m.guard() is None and m.guard(fast=True) is None
    ev = m.evaluate()
    assert ev["ok"] and ev["flight"]["apogee_m"] > 200 and ev["silhouette"]["fin"]
    assert ev["masses"]["mode"] == "estimate" and ev["masses"]["dry_mass_g"] > 100
    # heavier fins at the tail move the centre of gravity aft: the margin falls
    heavy = RocketModel({**T.defaults("rocket"), "fin_thickness": 0.012}, {"materials": {"fins": "g10"}}).quick()
    assert heavy["margin_liftoff"] < q["margin_liftoff"]
    # a motor too weak to leave the rail is caught
    weak = RocketModel(T.defaults("rocket"), {}, BY_ID["estes-a8"])
    assert "rail" in (weak.guard() or "")


def test_wind_turns_the_rocket_upwind_and_the_parachute_drifts_downwind():
    calm = RocketModel(T.defaults("rocket"), {"launch": {"wind_m_s": 0.0}}).evaluate()
    windy = RocketModel(T.defaults("rocket"), {"launch": {"wind_m_s": 6.0}}).evaluate()
    assert abs(calm["flight"]["apogee_x"]) < 1.0
    assert windy["flight"]["apogee_x"] < -20                      # weathercocks into the wind
    assert windy["flight"]["drift_m"] > windy["flight"]["apogee_x"]  # then drifts downwind under the parachute
    assert windy["flight"]["apogee_m"] < calm["flight"]["apogee_m"]
    ballistic = RocketModel(T.defaults("rocket"), {"parachute": {"enabled": False}}).evaluate()
    assert ballistic["flight"]["flight_time"] < calm["flight"]["flight_time"]


def test_motor_files_round_trip():
    c6 = BY_ID["estes-c6"]
    back = parse_eng(c6.to_eng())[0]
    assert abs(back.total_impulse - c6.total_impulse) < 1e-3 and back.diameter_mm == 18 and back.delays == [3, 5, 7]
    assert c6.cls == "C" and BY_ID["aerotech-h128"].cls == "H"
    rse = """<engine-database><engine-list><engine mfg="Test" code="F40" dia="29" len="98" delays="4,7" propWt="30" initWt="80">
      <data><eng-data t="0" f="0"/><eng-data t="0.1" f="60"/><eng-data t="1.0" f="40"/><eng-data t="1.5" f="0"/></data></engine></engine-list></engine-database>"""
    m = parse_rse(rse)[0]
    assert m.name == "F40" and abs(m.total_impulse - (0.5 * 0.1 * 60 + 0.9 * 50 + 0.5 * 0.5 * 40)) < 1e-6
    assert m.mass(0.0) == pytest.approx(0.080) and m.mass(5.0) == pytest.approx(0.050)


def _ork_zip():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<openrocket version="1.8" creator="test"><rocket><name>Test rocket</name><subcomponents><stage><name>Sustainer</name><subcomponents>
  <nosecone><name>Nose</name><material type="bulk" density="1050">Plastic</material><length>0.07</length><thickness>0.002</thickness>
    <shape>ogive</shape><shapeparameter>1.0</shapeparameter><aftradius>auto</aftradius></nosecone>
  <bodytube><name>Body</name><material type="bulk" density="680">Paper</material><length>0.3</length><thickness>0.0005</thickness><radius>0.0124</radius>
    <subcomponents>
      <trapezoidfinset><name>Fins</name><material type="bulk" density="160">Balsa</material><fincount>3</fincount><thickness>0.0024</thickness>
        <crosssection>square</crosssection><rootchord>0.06</rootchord><tipchord>0.03</tipchord><sweeplength>0.03</sweeplength><height>0.045</height>
        <position type="bottom">0.0</position></trapezoidfinset>
      <launchlug><name>Lug</name><material type="bulk" density="680">Paper</material><radius>0.0022</radius><length>0.05</length><thickness>0.0005</thickness>
        <position type="middle">0.0</position></launchlug>
      <parachute><name>Chute</name><material type="surface" density="0.067">Ripstop</material><cd>auto</cd><diameter>0.3</diameter><length>0.04</length>
        <position type="top">0.02</position></parachute>
      <masscomponent><name>Wadding</name><mass>0.004</mass><length>0.02</length><position type="top">0.08</position></masscomponent>
      <innertube><name>Mount</name><material type="bulk" density="680">Paper</material><length>0.07</length><outerradius>0.0095</outerradius><thickness>0.0005</thickness>
        <position type="bottom">0.0</position>
        <motormount><ignitionevent>automatic</ignitionevent><overhang>0.0</overhang>
          <motor configid="x"><type>single</type><manufacturer>Estes</manufacturer><designation>C6</designation><diameter>0.018</diameter><length>0.07</length><delay>5.0</delay></motor>
        </motormount></innertube>
    </subcomponents></bodytube>
  <transition><name>Tail</name><material type="bulk" density="1050">Plastic</material><length>0.02</length><thickness>0.002</thickness><shape>conical</shape>
    <foreradius>auto</foreradius><aftradius>0.009</aftradius></transition>
</subcomponents></stage></subcomponents></rocket></openrocket>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rocket.ork", xml)
    return buf.getvalue()


def test_openrocket_import_maps_geometry_masses_and_motor():
    from aerosculpt.rocket.ork import parse_ork
    imp = parse_ork(_ork_zip())
    v = imp["template_values"]
    assert imp["name"] == "Test rocket" and v["nose_family"] == "ogive" and v["nose_length"] == pytest.approx(0.07)
    assert v["diameter"] == pytest.approx(0.0248) and v["length"] == pytest.approx(0.39) and v["tail_length"] == pytest.approx(0.02)
    assert v["fin_count"] == "3" and v["fin_root"] == pytest.approx(0.06) and v["fin_span"] == pytest.approx(0.045)
    assert v["fin_position"] == pytest.approx(0.0, abs=1e-6)        # fins end where the tail starts
    rk = imp["rocket"]
    assert rk["masses"]["mode"] == "items" and {i["name"] for i in rk["masses"]["items"]} >= {"Nose", "Body", "Fins", "Chute", "Wadding", "Lug"}
    assert rk["motor"]["id"] == "estes-c6" and rk["motor"]["delay"] == 5.0 and rk["launch_lug"] == "1/8in"
    assert rk["parachute"]["diameter_m"] == pytest.approx(0.3)
    total = sum(i["g"] for i in rk["masses"]["items"])
    assert 15 < total < 80
    m = RocketModel(T.clean("rocket", v), rk, BY_ID["estes-c6"])
    ev = m.evaluate()
    assert 50 < ev["flight"]["apogee_m"] < 600 and ev["stability"]["liftoff"]["margin"] > 0.5


def test_rocket_endpoints_and_apogee_optimization(client):
    _signup(client, "rocketeer@example.com")
    cat = client.get("/api/rocket/catalog", headers=H).json()
    assert "estes-c6" in {m["id"] for m in cat["motors"]} and "plywood" in cat["fin_materials"]
    p = client.post("/api/projects", json={"template": "rocket"}, headers=H).json()
    pid = p["id"]
    assert p["settings"]["optimize"]["objective"] == "apogee"
    # the live evaluation, saving a windier site
    ev = client.post(f"/api/projects/{pid}/rocket", json={"rocket": {"launch": {"wind_m_s": 4.0}}, "save": True}, headers=H).json()
    assert ev["ok"] and ev["flight"]["apogee_m"] > 200 and ev["launch"]["wind_m_s"] == 4.0 and ev["silhouette"]["body"]
    st = client.get(f"/api/projects/{pid}", headers=H).json()["settings"]
    assert st["rocket"]["launch"]["wind_m_s"] == 4.0
    assert client.post(f"/api/projects/{pid}/rocket", json={"rocket": {"launch": {"rail_m": 99}}}, headers=H).status_code == 400
    # a motor file, then a flight with it
    up = client.post("/api/motors?filename=my.eng", content=BY_ID["estes-e12"].to_eng().replace("E12", "E12X"), headers=H)
    assert up.status_code == 201, up.text
    mid = up.json()[0]["id"]
    assert mid.startswith("u-") and any(m["id"] == mid for m in client.get("/api/motors?q=e12x", headers=H).json())
    assert client.get(f"/api/motors/{mid}", headers=H).json()["curve"]
    ev2 = client.post(f"/api/projects/{pid}/rocket", json={"rocket": {"motor": {"id": mid, "delay": 6}}, "save": True}, headers=H).json()
    assert ev2["motor"]["id"] == mid and ev2["launch"]["delay_s"] == 6
    csv_ = client.get(f"/api/projects/{pid}/rocket/drag.csv", headers=H)
    assert csv_.status_code == 200 and "mach,cd" in csv_.text
    # an altimeter log for comparison
    log = "time,altitude (ft)\n0,0\n1,300\n2,800\n3,1100\n4,1200\n5,1150\n"
    r = client.post(f"/api/projects/{pid}/rocket/flightlog?filename=flight.csv", content=log, headers=H)
    assert r.status_code == 201 and r.json()["apogee_m"] == pytest.approx(1200 * 0.3048, rel=1e-3)
    assert client.post(f"/api/projects/{pid}/rocket", json={}, headers=H).json()["flight_log"]["apogee_t"] == 4.0
    # an apogee optimization: no CFD, a flat credit, the motor travels with the job
    client.patch(f"/api/projects/{pid}", json={"settings": {"template": {"params": {"nose_length": {"locked": False}, "tail_taper": {"locked": False}}}}}, headers=H)
    r = client.post(f"/api/projects/{pid}/jobs", json={"kind": "optimize", "fidelity": "draft",
                    "optimize": {"objective": "apogee", "iterations": 8, "top_k": 2}}, headers=H)
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["spec"]["optimize"]["objective"] == "apogee" and j["estimated_credits"] == 1
    assert j["spec"]["rocket"]["motor_data"]["id"] == mid and j["spec"]["rocket"]["motor_data"]["curve"]
    est = client.post(f"/api/projects/{pid}/estimate", json={"kind": "optimize", "optimize": {"objective": "apogee", "iterations": 8, "top_k": 2}}, headers=H).json()
    assert "flight model" in est["lines"][0]
    from aerosculpt.db import db
    from aerosculpt.jobs import job_dir, run_job
    with db() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (j["id"],)).fetchone()
    run_job(row)
    done = client.get(f"/api/jobs/{j['id']}", headers=H).json()
    assert done["status"] == "done", done
    res = json.loads((job_dir(j["id"]) / "result.json").read_text())
    assert res["objective"] == "apogee" and res["rocket"]["best"]["flight"]["apogee_m"] >= 0.98 * res["rocket"]["base"]["flight"]["apogee_m"]
    assert res["best"]["polar"][0]["speeds"][0]["cd"] > 0 and res["setup"]["method"] == "rocket model"
    assert res["rocket"]["best"]["stability"]["liftoff"]["margin"] >= 1.0            # the guard held
    assert done["summary"]["apogee_m"] == pytest.approx(res["rocket"]["best"]["flight"]["apogee_m"])
    # a CFD drag optimization of a rocket also carries the flight-model inputs for its guard
    r = client.post(f"/api/projects/{pid}/jobs", json={"kind": "optimize", "fidelity": "draft", "optimize": {"objective": "drag", "iterations": 4, "top_k": 2}}, headers=H)
    assert r.status_code == 201 and r.json()["spec"]["rocket"]["motor_data"]["id"] == mid
    client.post(f"/api/jobs/{r.json()['id']}/cancel", headers=H)
    # apogee goals need a rocket
    q = client.post("/api/projects", json={"template": "wing"}, headers=H).json()
    r = client.post(f"/api/projects/{q['id']}/jobs", json={"kind": "optimize", "optimize": {"objective": "apogee", "iterations": 4, "top_k": 2}}, headers=H)
    assert r.status_code == 400


def test_openrocket_upload_becomes_a_rocket_project(client):
    _signup(client, "orkuser@example.com")
    r = client.post("/api/projects?filename=test.ork", content=_ork_zip(), headers=H)
    assert r.status_code == 201, r.text
    p = r.json()
    st = p["settings"]
    assert p["name"] == "Test rocket" and st["template"]["id"] == "rocket" and st["rocket"]["masses"]["mode"] == "items"
    assert st["rocket"]["motor"]["id"] == "estes-c6" and p["source_format"] == "ork"
    ev = client.post(f"/api/projects/{p['id']}/rocket", json={}, headers=H).json()
    assert ev["masses"]["mode"] == "items" and ev["motor"]["id"] == "estes-c6" and ev["import_notes"]
