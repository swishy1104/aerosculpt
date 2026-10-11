import hashlib
import hmac
import json
import time

import pytest
from starlette.testclient import TestClient

from aerosculpt import billing
from aerosculpt.config import settings
from aerosculpt.db import db
from aerosculpt.geometry.samples import icosphere
from aerosculpt.web import app

H = {"X-Requested-With": "AeroSculpt"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def signup(c, email, pw="Correct-Horse-9"):
    r = c.post("/api/auth/signup", json={"email": email, "password": pw, "name": "T", "accept_terms": True})
    assert r.status_code == 200, r.text
    return r


def test_health_and_pages(client):
    assert client.get("/api/health").json()["ok"]
    for path in ("/", "/pricing", "/validation", "/docs", "/login", "/signup", "/terms", "/privacy"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "content-security-policy" in r.headers
    assert client.get("/nope").status_code == 404
    assert client.get("/api/nope").json()["error"]


def test_importmap_hash_matches_csp(client):
    r = client.get("/app/project/x")
    body = r.text
    start = body.index('<script type="importmap">') + len('<script type="importmap">')
    content = body[start:body.index("</script>", start)]
    import base64
    digest = base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()
    assert f"sha256-{digest}" in r.headers["content-security-policy"]


def test_signup_validation(client):
    r = client.post("/api/auth/signup", json={"email": "bad", "password": "x", "accept_terms": True})
    assert r.status_code == 400
    r = client.post("/api/auth/signup", json={"email": "a@b.co", "password": "short", "accept_terms": True})
    assert r.status_code == 400 and r.json()["field"] == "password"
    r = client.post("/api/auth/signup", json={"email": "a@b.co", "password": "Long-enough-1"})
    assert r.json()["field"] == "accept_terms"


def test_full_flow(client):
    signup(client, "flow@example.com")
    me = client.get("/api/me").json()
    assert me["plan"] == "lite" and me["credits_left"] == 40 and me["is_admin"]
    # CSRF: cookie-authenticated writes need the header
    assert client.patch("/api/me", json={"name": "X"}).status_code == 403
    assert client.patch("/api/me", json={"name": "X"}, headers=H).json()["name"] == "X"
    # create from a sample and from an upload
    p = client.post("/api/projects", json={"sample": "sphere"}, headers=H).json()
    assert p["metrics"]["watertight"] and p["settings"]["units"] == "m"
    stl = icosphere(10.0, 2).to_stl_bytes()
    r = client.post("/api/projects?filename=ball.stl&name=Ball", content=stl,
                    headers={**H, "Content-Type": "application/octet-stream"})
    assert r.status_code == 201, r.text
    up = r.json()
    assert up["settings"]["units"] in ("mm", "cm")
    assert client.get(f"/api/projects/{up['id']}/model").content == stl
    # settings validation
    assert client.patch(f"/api/projects/{p['id']}", json={"settings": {"envelope": {"speed_max": 9999}}}, headers=H).status_code == 400
    assert client.patch(f"/api/projects/{p['id']}", json={"settings": {"forward": "+z", "up": "+z"}}, headers=H).status_code == 400
    ok = client.patch(f"/api/projects/{p['id']}", json={"settings": {"envelope": {"speed_max": 12, "aoa_min": 3}}}, headers=H).json()
    e = ok["settings"]["envelope"]
    assert e["speed_max"] == 12 and (e["aoa_min"], e["aoa_max"]) == (0, 3)   # reversed range is swapped
    # plan gating on Lite
    env = {"speed_min": 10, "speed_max": 30, "speed_count": 3, "aoa_min": 0, "aoa_max": 4, "aoa_count": 2, "altitude": 0, "yaw": 0}
    base = {"units": "m", "forward": "-x", "up": "+z", "envelope": env}
    # envelope limits per plan: Lite sweeps up to 3 angles and 4 speeds
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "envelope": {**env, "aoa_count": 5}, "kind": "analyze",
                    "fidelity": "draft"}, headers=H)
    assert r.status_code == 403 and "angles" in r.json()["error"]
    # protected regions are validated and limited (2 on Lite)
    zone = {"type": "protect", "shape": "box", "center": [0, 0, 0], "size": [0.1, 0.1, 0.1], "name": "mount"}
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "zones": [dict(zone, size=[0, 1, 1])], "kind": "analyze"}, headers=H)
    assert r.status_code == 400
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "zones": [zone] * 3, "kind": "analyze"}, headers=H)
    assert r.status_code == 403
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "zones": [dict(zone, rotation=[0, 400, 0])], "kind": "analyze"}, headers=H)
    assert r.status_code == 400 and "rotation" in r.json()["error"]
    # the internal exploration grid is not a choice; the retired Ultra level maps onto High (a paid level)
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "analyze", "fidelity": "coarse"}, headers=H)
    assert r.status_code == 400
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "analyze", "fidelity": "ultra"}, headers=H)
    assert r.status_code == 403
    # Mach limit: the General tunnel stops at Mach 1.2 (the Supersonic tunnel takes over)
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "envelope": {**env, "speed_max": 440}, "kind": "analyze"}, headers=H)
    assert r.status_code == 400 and "Mach" in r.json()["error"]
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "analyze", "fidelity": "high"}, headers=H)
    assert r.status_code == 403 and r.json()["upgrade"]
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "optimize", "fidelity": "draft",
                    "optimize": {"aspects": ["nose"], "iterations": 50}}, headers=H)
    assert r.status_code == 403
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "optimize", "fidelity": "draft",
                    "optimize": {"aspects": ["nose", "tail", "length"], "iterations": 3}}, headers=H)
    assert r.status_code == 403
    # lift-based goals are refused until lift passes validation
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "optimize", "fidelity": "draft",
                    "optimize": {"objective": "ld", "aspects": ["nose"], "iterations": 3}}, headers=H)
    assert r.status_code == 400 and "validation" in r.json()["error"]
    assert "ld" in client.get("/api/plans").json()["objectives_unvalidated"]
    # a valid job is queued and holds its estimated credits
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "optimize", "fidelity": "draft", "zones": [zone],
                    "optimize": {"aspects": ["nose", "tail"], "iterations": 4, "top_k": 3}}, headers=H)
    assert r.status_code == 403 and "parameters at once" in r.json()["error"]     # Lite optimizes the top 2
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "optimize", "fidelity": "draft", "zones": [zone],
                    "optimize": {"aspects": ["nose", "tail"], "iterations": 4, "top_k": 2}}, headers=H)
    assert r.status_code == 201, r.text
    job = r.json()
    # explore on the coarse grid: 2 angles x (4 designs + 3) x 0.4 credits; confirm: 4 designs x 2 angles at draft
    assert job["status"] == "queued" and job["estimated_credits"] == 14
    assert job["spec"]["envelope"]["speeds"] == [10.0, 20.0, 30.0] and job["spec"]["envelope"]["aoas"] == [0.0, 4.0]
    assert job["spec"]["zones"][0]["name"] == "mount"
    cat = client.get("/api/plans").json()
    assert "ultra" not in cat["fidelities"] and cat["compute"]["backend"] in ("cpu", "gpu")
    sp = cat["compute"]["seconds_per_run"]
    assert 0 < sp["coarse"] < sp["draft"] < sp["standard"] < sp["high"]
    # concurrency limit on Lite
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "analyze", "fidelity": "draft"}, headers=H)
    assert r.status_code == 429
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=H).json()["ok"]
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "canceled"
    # export is gated
    r = client.get(f"/api/jobs/{job['id']}/files/optimized.stl")
    assert r.status_code == 403
    # project limit on Lite (3)
    client.post("/api/projects", json={"sample": "cube"}, headers=H)
    r = client.post("/api/projects", json={"sample": "cube"}, headers=H)
    assert r.status_code == 402
    # demo upgrade to a Pro trial lifts the limits
    r = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=H)
    assert r.json()["mode"] == "demo"
    me = client.get("/api/me").json()
    assert me["plan"] == "pro" and me["plan_status"] == "trialing" and me["trial_used"]
    r = client.post(f"/api/projects/{p['id']}/jobs", json={**base, "kind": "analyze", "fidelity": "high"}, headers=H)
    assert r.status_code == 201
    # second trial is not offered: switching away and back is a paid activation
    client.post("/api/billing/demo", json={"action": "downgrade_now"}, headers=H)
    client.post("/api/billing/checkout", json={"plan": "pro"}, headers=H)
    assert client.get("/api/me").json()["plan_status"] == "active"


def test_isolation_between_users(client):
    signup(client, "alice@example.com")
    p = client.post("/api/projects", json={"sample": "sphere"}, headers=H).json()
    client.post("/api/auth/logout", headers=H)
    signup(client, "bob@example.com")
    assert client.get(f"/api/projects/{p['id']}").status_code == 404
    assert client.get(f"/api/projects/{p['id']}/model").status_code == 404
    assert client.delete(f"/api/projects/{p['id']}", headers=H).status_code == 404
    assert all(x["id"] != p["id"] for x in client.get("/api/projects").json())
    assert not client.get("/api/me").json()["is_admin"]
    assert client.get("/api/admin/stats").status_code == 403


def test_login_logout_and_password_reset(client):
    client.post("/api/auth/logout", headers=H)
    assert client.post("/api/auth/login", json={"email": "alice@example.com", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "alice@example.com", "password": "Correct-Horse-9"}).status_code == 200
    assert client.get("/api/me").json()["email"] == "alice@example.com"
    client.post("/api/auth/logout", headers=H)
    assert client.get("/api/me").status_code == 401
    # reset: the token is only ever emailed; read it from the DB the way the email would carry it
    from aerosculpt.security import new_token, token_hash
    tok = new_token()
    with db() as con:
        uid = con.execute("SELECT id FROM users WHERE email='alice@example.com'").fetchone()["id"]
        con.execute("INSERT INTO password_resets (token_hash, user_id, expires_at) VALUES (?,?,?)",
                    (token_hash(tok), uid, time.time() + 60))
    assert client.post("/api/auth/reset", json={"token": "nope", "password": "Another-pass-7"}).status_code == 400
    assert client.post("/api/auth/reset", json={"token": tok, "password": "Another-pass-7"}).status_code == 200
    assert client.post("/api/auth/reset", json={"token": tok, "password": "Another-pass-8"}).status_code == 400
    client.post("/api/auth/logout", headers=H)
    assert client.post("/api/auth/login", json={"email": "alice@example.com", "password": "Another-pass-7"}).status_code == 200


def _sig(payload: bytes, secret="whsec_test_secret", ts=None):
    ts = ts or int(time.time())
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def test_webhook_signature_and_subscription_sync(client, monkeypatch):
    monkeypatch.setattr(settings, "stripe_price_pro", "price_pro_123")
    with db() as con:
        uid = con.execute("SELECT id FROM users WHERE email='bob@example.com'").fetchone()["id"]
        con.execute("UPDATE users SET stripe_customer_id='cus_bob' WHERE id=?", (uid,))
    sub = {"id": "sub_1", "customer": "cus_bob", "status": "trialing", "trial_end": int(time.time()) + 86400,
           "cancel_at_period_end": False, "metadata": {},
           "items": {"data": [{"price": {"id": "price_pro_123"}, "current_period_end": int(time.time()) + 86400 * 14}]}}
    event = {"id": "evt_1", "type": "customer.subscription.updated", "data": {"object": sub}}
    payload = json.dumps(event).encode()
    bad = client.post("/api/billing/webhook", content=payload, headers={"Stripe-Signature": "t=1,v1=deadbeef"})
    assert bad.status_code == 400
    old = client.post("/api/billing/webhook", content=payload, headers={"Stripe-Signature": _sig(payload, ts=int(time.time()) - 3600)})
    assert old.status_code == 400
    ok = client.post("/api/billing/webhook", content=payload, headers={"Stripe-Signature": _sig(payload)})
    assert ok.status_code == 200 and ok.json()["outcome"] == "ok"
    dup = client.post("/api/billing/webhook", content=payload, headers={"Stripe-Signature": _sig(payload)})
    assert dup.json()["outcome"] == "duplicate"
    with db() as con:
        u = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    assert u["plan"] == "pro" and u["plan_status"] == "trialing" and u["trial_used"] == 1
    # cancellation drops the user to Lite
    sub2 = dict(sub, status="canceled")
    payload = json.dumps({"id": "evt_2", "type": "customer.subscription.deleted", "data": {"object": sub2}}).encode()
    assert client.post("/api/billing/webhook", content=payload, headers={"Stripe-Signature": _sig(payload)}).status_code == 200
    with db() as con:
        assert con.execute("SELECT plan FROM users WHERE id=?", (uid,)).fetchone()["plan"] == "lite"


def test_form_encoding_for_stripe():
    flat = billing._flatten({"line_items": [{"price": "p", "quantity": 1}], "subscription_data": {"trial_period_days": 14},
                             "allow_promotion_codes": True})
    assert ("line_items[0][price]", "p") in flat
    assert ("subscription_data[trial_period_days]", "14") in flat
    assert ("allow_promotion_codes", "true") in flat


def test_three_tunnels(client):
    client.post("/api/auth/logout", headers=H)
    signup(client, "tunnels@example.com")
    lim = client.get("/api/me").json()["limits"]
    assert lim["max_propellers"] == 1 and lim["max_iterations_panel"] == 30   # the workspace sizes its sliders from these
    samples = {s["key"]: s for s in client.get("/api/samples").json()}
    assert samples["capsule"]["tunnel"] == "supersonic" and samples["rc-trainer"]["propellers"] == 1

    # supersonic: Mach envelope, materials, panel-method credits
    p = client.post("/api/projects", json={"sample": "hypersonic-cone"}, headers=H).json()
    st = p["settings"]
    assert st["tunnel"] == "supersonic" and st["hyper"]["thermal"]["material"] == "inconel"
    assert [z["type"] for z in st["zones"]] == ["material"] and st["zones"][0]["material"] == "rcc"
    bad = client.patch(f"/api/projects/{p['id']}", json={"settings": {"hyper": {"envelope": {"mach_max": 20}}}}, headers=H)
    assert bad.status_code == 400
    bad = client.patch(f"/api/projects/{p['id']}", json={"settings": {"hyper": {"thermal": {"material": "unobtainium"}}}}, headers=H)
    assert bad.status_code == 400
    custom = {"name": "My skin", "t_max_c": 320, "emissivity": 0.7}
    ok = client.patch(f"/api/projects/{p['id']}", json={"settings": {"hyper": {"thermal": {"material": custom}}}}, headers=H)
    assert ok.json()["settings"]["hyper"]["thermal"]["material"]["t_max_c"] == 320
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "analyze"}, headers=H)
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["spec"]["tunnel"] == "supersonic" and j["spec"]["envelope"]["machs"] == [4.0, 5.0, 6.0, 7.0]
    assert j["estimated_credits"] == 1
    client.post(f"/api/jobs/{j['id']}/cancel", headers=H)
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "optimize", "optimize": {
        "objective": "peak_temp", "aspects": ["nose"], "iterations": 25, "top_k": 2}}, headers=H)
    assert r.status_code == 201, r.text
    assert r.json()["estimated_credits"] == 3          # 27 designs: one credit per ten
    client.post(f"/api/jobs/{r.json()['id']}/cancel", headers=H)
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "optimize", "optimize": {
        "objective": "drag", "aspects": ["nose"], "iterations": 31, "top_k": 2}}, headers=H)
    assert r.status_code == 403                         # Lite: 30 supersonic designs

    # RC: propellers validated, every (angle, speed) is a run
    q = client.post("/api/projects", json={"sample": "rc-trainer"}, headers=H).json()
    assert q["settings"]["tunnel"] == "rc" and q["settings"]["propellers"][0]["pitch_in"] == 7.0
    prop = dict(q["settings"]["propellers"][0])
    assert client.patch(f"/api/projects/{q['id']}", json={"settings": {"propellers": [dict(prop, blades=9)]}},
                        headers=H).status_code == 400
    r = client.post(f"/api/projects/{q['id']}/jobs", json={"kind": "analyze", "fidelity": "draft"}, headers=H)
    assert r.status_code == 201, r.text
    assert r.json()["estimated_credits"] == 6           # 2 angles x 3 speeds, draft
    client.post(f"/api/jobs/{r.json()['id']}/cancel", headers=H)
    slow = dict(q["settings"]["envelope"], speed_min=1.0)
    r = client.post(f"/api/projects/{q['id']}/jobs", json={"kind": "analyze", "envelope": slow}, headers=H)
    assert r.status_code == 400 and "2 m/s" in r.json()["error"]
    two = client.patch(f"/api/projects/{q['id']}", json={"settings": {"propellers": [prop, dict(prop, id="b")]}}, headers=H)
    assert two.status_code == 200
    r = client.post(f"/api/projects/{q['id']}/jobs", json={"kind": "analyze", "envelope": dict(slow, speed_min=10)}, headers=H)
    assert r.status_code == 403 and "propeller" in r.json()["error"]

    # live propeller estimate: more blades, more thrust and a faster slipstream
    two = dict(prop, id="p2", blades=2, family="wide")
    three = dict(prop, id="p3", blades=3, family="wide")
    est = client.post("/api/propellers/estimate", json={"propellers": [two, three], "speeds": [10, 20]}, headers=H).json()["propellers"]
    t2, t3 = est[0]["rows"][0], est[1]["rows"][0]
    assert t3["thrust_N"] > 1.2 * t2["thrust_N"] and t3["slipstream"] > t2["slipstream"] > 1.0
    assert client.post("/api/propellers/estimate", json={"propellers": [two], "speeds": [0]}, headers=H).status_code == 400

    # switching the tunnel of an existing project
    sw = client.patch(f"/api/projects/{q['id']}", json={"settings": {"tunnel": "supersonic"}}, headers=H).json()
    assert sw["settings"]["tunnel"] == "supersonic" and sw["settings"]["hyper"]["envelope"]["mach_min"] == 1.5
    assert client.patch(f"/api/projects/{q['id']}", json={"settings": {"tunnel": "warp"}}, headers=H).status_code == 400


def test_invite_only_signup(client, monkeypatch):
    from aerosculpt.config import settings
    client.post("/api/auth/logout", headers=H)
    monkeypatch.setattr(settings, "allowed_signup_emails", {"owner@example.com"})
    r = client.post("/api/auth/signup", json={"email": "stranger@example.com", "password": "correct horse battery 9",
                                              "name": "S", "accept_terms": True}, headers=H)
    assert r.status_code == 403 and "invitation" in r.json()["error"]
    r = client.post("/api/auth/signup", json={"email": "owner@example.com", "password": "correct horse battery 9",
                                              "name": "O", "accept_terms": True}, headers=H)
    assert r.status_code in (200, 201), r.text
    monkeypatch.setattr(settings, "owner_emails", {"owner@example.com"})
    client.post("/api/auth/logout", headers=H)
    r = client.post("/api/auth/login", json={"email": "owner@example.com", "password": "correct horse battery 9"}, headers=H)
    assert r.status_code == 200, r.text
    me = client.get("/api/me").json()
    assert me["plan"] == "team" and me["is_admin"]


def test_templates(client):
    client.post("/api/auth/logout", headers=H)
    signup(client, "templates@example.com")
    cat = client.get("/api/plans").json()
    assert set(cat["templates"]) == {"rocket", "uav", "wing"} and "coarse" not in cat["fidelities"]
    assert client.get("/api/me").json()["limits"]["max_top_k"] == 2
    # live geometry for the 3D view: no project, no simulation
    r = client.post("/api/templates/build", json={"template": "wing", "values": {"wing_span": 1.5, "wing_camber": 0.04}}, headers=H)
    assert r.status_code == 200 and r.json()["faces"] > 500 and r.json()["info"]["airfoil"].startswith("NACA 44")
    assert client.post("/api/templates/build", json={"template": "boat"}, headers=H).status_code == 400
    # a project from a template keeps its named dimensions; changing one rebuilds the model file
    p = client.post("/api/projects", json={"template": "uav"}, headers=H).json()
    st = p["settings"]
    assert st["template"]["id"] == "uav" and st["units"] == "m" and st["optimize"]["objective"] == "drag_fixed_lift"
    assert st["envelope"]["ref_area"] == pytest.approx(0.465, rel=1e-3)        # coefficients on the wing area
    size0 = p["file_size"]
    q = client.patch(f"/api/projects/{p['id']}", json={"settings": {"template": {"values": {"wing_span": 2.4},
                     "params": {"wing_span": {"bounds": [1.8, 2.6]}, "wing_sweep": {"locked": True}}}}}, headers=H).json()
    assert q["settings"]["template"]["values"]["wing_span"] == 2.4
    assert q["settings"]["template"]["params"]["wing_sweep"]["locked"] is True
    assert q["settings"]["envelope"]["ref_area"] == pytest.approx(0.558, rel=1e-3) and q["metrics"]["width"] == pytest.approx(2.4, rel=1e-3)
    assert q["file_size"] != size0 or q["metrics"]["template_info"]["span"] == 2.4
    # winged templates are optimized on the wing model: drag at fixed lift, no CFD credits, the larger design budget
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "optimize", "fidelity": "draft",
                    "optimize": {"objective": "drag", "iterations": 30, "top_k": 2, "mass_kg": 3}}, headers=H)
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["spec"]["optimize"]["objective"] == "drag_fixed_lift" and j["spec"]["template"]["id"] == "uav"
    assert j["estimated_credits"] == 1 and j["spec"]["optimize"]["mass_kg"] == 3
    client.post(f"/api/jobs/{j['id']}/cancel", headers=H)
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "optimize", "optimize": {"iterations": 31, "top_k": 2}}, headers=H)
    assert r.status_code == 403
    keys = [k["key"] for k in cat["templates"]["uav"]["params"]]
    locked = client.patch(f"/api/projects/{p['id']}", json={"settings": {"template": {"params": {k: {"locked": True} for k in keys}}}},
                          headers=H)
    assert locked.status_code == 200
    r = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "optimize", "optimize": {"iterations": 10, "top_k": 2}}, headers=H)
    assert r.status_code == 400 and "locked" in r.json()["error"]


def test_guest_demo(client, monkeypatch):
    from aerosculpt.jobs import purge_guests
    client.post("/api/auth/logout", headers=H)
    assert client.post("/api/auth/guest", headers=H).status_code == 404          # off unless GUEST_DEMO=1
    assert client.get("/api/plans").json()["guest_demo"] is False
    monkeypatch.setattr(settings, "guest_demo", True)
    cat = client.get("/api/plans").json()
    assert cat["guest_demo"] and "guest" not in [p["key"] for p in cat["plans"]]  # never offered on the pricing page
    r = client.post("/api/auth/guest", headers=H)
    assert r.status_code == 200, r.text
    me = client.get("/api/me").json()
    assert me["guest"] and me["plan"] == "guest" and me["limits"]["fidelities"] == ["draft", "standard"]
    assert me["guest_expires_at"] > time.time()
    assert client.post("/api/billing/checkout", json={"plan": "pro"}, headers=H).status_code == 403
    p = client.post("/api/projects", json={"sample": "sphere"}, headers=H)
    assert p.status_code in (200, 201), p.text
    # expired guests are deleted with everything they made; live ones are kept
    assert purge_guests() == 0
    assert purge_guests(now=time.time() + settings.guest_hours * 3600 + 60) >= 1
    assert client.get("/api/me").status_code == 401
