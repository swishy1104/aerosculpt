"""The professional features: grid convergence, structure, mission, CAD exchange, exports,
road and wheels, nested refinement, resume after a restart, teams, single sign-on, Fusion."""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from starlette.testclient import TestClient

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
    r = c.post("/api/auth/signup", json={"email": email, "password": "Correct-Horse-9", "name": email.split("@")[0],
                                          "accept_terms": True})
    assert r.status_code == 200, r.text


def _login(c, email):
    from aerosculpt.security import auth_limiter
    auth_limiter.hits.clear()
    c.post("/api/auth/logout", headers=H)
    assert c.post("/api/auth/login", json={"email": email, "password": "Correct-Horse-9"}, headers=H).status_code == 200


# ---- numerics ---------------------------------------------------------------------------------
def test_gci_recovers_order_and_the_exact_value():
    from aerosculpt.cfd.gridcheck import gci
    h = [1.0, 1.3, 1.69]
    f = [2.0 + 0.4 * x ** 2 for x in h]
    g = gci(*f, *h)
    assert abs(g["order"] - 2.0) < 1e-6 and abs(g["extrapolated"] - 2.0) < 1e-9 and g["behaviour"] == "monotonic"
    g = gci(1.0, 1.1, 0.95, *h)                     # oscillating: no reliable order, a conservative band
    assert g["behaviour"] == "oscillatory" and g["order_used"] == 1.0 and g["gci_fine"] > 0


def test_spar_sizing_grows_for_stiffness_and_rejects_impossible_wings():
    from aerosculpt.aero.structure import StructureSettings, wing_structure
    from aerosculpt.aero.vlm import VLM, LiftingSurface

    def size(t, span, root, tip, mat):
        w = LiftingSurface("wing", (0.0, 0.0, 0.0), span, root, tip, 3.0, 2.0, section={"family": "naca4", "m": 0.02, "p": 0.45, "t": t})
        v = VLM([w], 0.56, 0.24)
        return wing_structure(v, [w], 3 * 9.80665, StructureSettings(material=mat), v.solve(v.alpha_for_cl(0.5), 20.0, 1.225)["cl_strip"])
    ok = size(0.12, 2.4, 0.30, 0.165, "carbon")
    thin = size(0.08, 2.4, 0.255, 0.066, "carbon")
    bad = size(0.06, 3.2, 0.2, 0.05, "spruce")
    assert ok["ok"] and ok["surfaces"][0]["sized_by"] == "strength"
    assert thin["ok"] and thin["surfaces"][0]["sized_by"] == "stiffness"
    assert abs(thin["surfaces"][0]["tip_deflection_frac"] - 0.15) < 1e-6
    assert thin["surfaces"][0]["spar_kg"] > ok["surfaces"][0]["spar_kg"]
    assert not bad["ok"] and "too thin" in bad["why"]


def test_mission_speeds_and_endurance_are_consistent():
    from aerosculpt.aero.mission import MissionInputs, mission
    from aerosculpt.cfd.analysis import isa
    from aerosculpt.geometry import templates as T
    m = mission(T.build("uav", T.defaults("uav")), 3.0, isa(0.0), MissionInputs(battery_wh=100, efficiency=0.55))
    be, br = m["best_endurance"], m["best_range"]
    assert m["stall_speed"] < be["v"] <= br["v"]
    assert be["power_W"] <= br["power_W"]
    # endurance from energy and power, by hand
    assert abs(be["endurance_min"] - 100 * 3600 * 0.8 / (be["power_W"] / 0.55) / 60) < 1e-6
    assert 30 < be["endurance_min"] < 600 and br["range_km"] > 0


# ---- geometry, CAD, exports ------------------------------------------------------------------------
def test_step_round_trip_keeps_the_volume():
    from aerosculpt.geometry import cad
    from aerosculpt.geometry.samples import ahmed_body
    if not cad.available():
        pytest.skip("gmsh not installed")
    import tempfile
    from pathlib import Path
    m = ahmed_body(25)
    out = Path(tempfile.mkdtemp()) / "a.step"
    cad.mesh_to_step(m, out)
    back = cad.cad_to_mesh(out.read_bytes(), "a.step")
    assert abs(back.signed_volume() / m.signed_volume() - 1) < 1e-3


def test_vtk_files_are_well_formed():
    from aerosculpt.cfd.volume import q_criterion, vtk_surface, vtk_volume
    from aerosculpt.geometry.samples import icosphere
    n = 12
    x = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.stack([-Y, X, 0 * Z]).astype(np.float32)          # solid-body rotation: Q = 1 everywhere
    q = q_criterion(u, x[1] - x[0])
    assert abs(float(np.median(q)) - 1.0) < 1e-4
    vol = {"origin": np.zeros(3), "spacing": float(x[1] - x[0]), "u": u.astype(np.float16), "cp": np.zeros((n, n, n), np.float16),
           "solid": np.zeros((n, n, n), np.uint8)}
    data = vtk_volume(vol)
    assert data.startswith(b"# vtk DataFile Version 3.0") and b"DIMENSIONS 12 12 12" in data
    assert len(data) > 4 * n ** 3 * 6                          # velocity (3) + cp + Q + solid as float32
    s = vtk_surface(icosphere(0.5, 1))
    assert b"POLYGONS 80 320" in s


def test_wheels_are_found_on_a_car_shape():
    from aerosculpt.geometry.samples import box, merge, revolve
    from aerosculpt.geometry.wheels import detect_wheels
    body = box((4.0, 1.8, 1.0), 6).transformed(t=[0, 0, 0.55])
    wheels = []
    for x in (-1.3, 1.3):
        for y in (-0.8, 0.8):
            w = revolve(np.linspace(-0.12, 0.12, 4), np.full(4, 0.33), 24)        # along x: turn it to an axle along y
            w = w.transformed(R=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]))
            wheels.append(w.transformed(t=[x, y, 0.33]))
    car = merge(body, *wheels)
    lo, hi = car.bbox()
    car = car.transformed(t=-(lo + hi) / 2)
    found = detect_wheels(car)
    assert len(found) == 4
    xs = sorted(round(w["center"][0], 1) for w in found)
    assert xs[0] == xs[1] and xs[2] == xs[3] and abs((xs[2] - xs[0]) - 2.6) < 0.15


# ---- solver extensions -------------------------------------------------------------------------------
def test_road_and_wall_model_run_and_keep_the_floor_out_of_the_force():
    from aerosculpt.cfd.analysis import Evaluator, Fidelity
    from aerosculpt.cfd.tunnel import SOLID
    from aerosculpt.geometry.samples import box
    m = box((1.0, 0.4, 0.3), 6)
    lo, hi = m.bbox()
    fid = Fidelity("t", 60_000, 2e4, extra_chunks=2)
    ev = Evaluator(lo, hi, 0.12, 1.0, 0.12, 1.0, fid, ground={"z": float(lo[2] - 0.12), "wheels": []})
    r = ev.run(m, 2e4, re_flight=3e6, want_fields=False)
    t = ev.tunnel
    assert t.zg == 2 and (t.flags[:, :, :3] == SOLID).all()
    assert r["wall_model"] and math.isfinite(r["cd_tunnel"]) and r["cd_tunnel"] > 0
    assert abs(ev.info()["ground_gap_cells"] - 0.12 / ev.domain.dx) < 0.02


def test_zoom_grid_is_finer_and_sized_round_the_body():
    from aerosculpt.cfd.zoom import zoom_domain
    d = zoom_domain([-0.5, -0.1, -0.1], [0.5, 0.1, 0.1], 0.3, 200_000)
    assert d.n_cells > 150_000 and d.origin[0] < -0.5 - 0.17 and d.origin[0] + d.shape[0] * d.dx > 0.5 + 0.44


# ---- platform -------------------------------------------------------------------------------------------
def test_jobs_of_a_dead_process_resume_at_once(client):
    from aerosculpt.db import db
    from aerosculpt.jobs import recover_orphans, worker_id
    _signup(client, "resume@example.com")
    p = client.post("/api/projects", json={"sample": "sphere"}, headers=H).json()
    j = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "analyze", "fidelity": "draft"}, headers=H).json()
    host = worker_id().rsplit(":", 1)[0]
    with db() as con:
        con.execute("UPDATE jobs SET status='running', worker=? WHERE id=?", (f"{host}:999999", j["id"]))
    assert j["id"] in recover_orphans(worker_id())
    assert client.get(f"/api/jobs/{j['id']}").json()["status"] == "queued"
    client.post(f"/api/jobs/{j['id']}/cancel", headers=H)


def test_estimate_and_grid_check_requests(client):
    _signup(client, "estimate@example.com")
    p = client.post("/api/projects", json={"sample": "sphere"}, headers=H).json()
    r = client.post(f"/api/projects/{p['id']}/estimate", json={"kind": "analyze", "fidelity": "draft", "grid_study": True}, headers=H)
    assert r.status_code == 200 and r.json()["seconds"] > 0 and any("grid check" in x for x in r.json()["lines"])
    r = client.post(f"/api/projects/{p['id']}/estimate", json={"kind": "optimize", "fidelity": "draft", "optimize": {"iterations": 12, "top_k": 2}}, headers=H)
    assert r.status_code == 200 and r.json()["seconds"] >= r.json().get("seconds_low", 0)
    j = client.post(f"/api/projects/{p['id']}/jobs", json={"kind": "analyze", "fidelity": "draft", "grid_study": True,
                                                         "refine": True, "ground": {"enabled": True, "clearance": 0.01}}, headers=H).json()
    assert j["spec"]["grid_study"] and j["spec"]["refine"] and j["spec"]["ground"]["enabled"] and j["spec"]["envelope"]["aoas"] == [0.0]
    client.post(f"/api/jobs/{j['id']}/cancel", headers=H)


def test_teams_share_projects_by_role(client):
    _signup(client, "member@example.com")
    _signup(client, "lead@example.com")
    assert client.post("/api/teams", json={"name": "Aero"}, headers=H).status_code == 403          # Team plan only
    client.post("/api/billing/checkout", json={"plan": "team"}, headers=H)
    tid = client.post("/api/teams", json={"name": "Aero"}, headers=H).json()["id"]
    assert client.post(f"/api/teams/{tid}/members", json={"email": "member@example.com", "role": "viewer"}, headers=H).json()["status"] == "added"
    assert client.post(f"/api/teams/{tid}/members", json={"email": "later@example.com", "role": "editor"}, headers=H).json()["status"] == "invited"
    p = client.post("/api/projects", json={"sample": "sphere"}, headers=H).json()
    assert client.post(f"/api/projects/{p['id']}/share", json={"team_id": tid}, headers=H).status_code == 200
    _login(client, "member@example.com")
    mine = client.get("/api/projects").json()
    shared = [x for x in mine if x["id"] == p["id"]]
    assert shared and shared[0]["team"]["role"] == "viewer" and not shared[0]["mine"]
    assert client.patch(f"/api/projects/{p['id']}", json={"name": "x"}, headers=H).status_code == 403   # viewers read only
    assert client.get(f"/api/teams/{tid}/audit").status_code == 403
    _login(client, "lead@example.com")
    uid = [m for m in client.get("/api/teams").json()[0]["members"] if m["email"] == "member@example.com"][0]["id"]
    client.patch(f"/api/teams/{tid}/members", json={"user_id": uid, "role": "editor"}, headers=H)
    _login(client, "member@example.com")
    assert client.patch(f"/api/projects/{p['id']}", json={"name": "Renamed by editor"}, headers=H).status_code == 200
    assert client.delete(f"/api/projects/{p['id']}", headers=H).status_code == 403                  # not the owner
    _signup(client, "later@example.com")                                                             # the invitation is honoured
    assert any(x["id"] == p["id"] for x in client.get("/api/projects").json())
    _login(client, "lead@example.com")
    acts = [a["action"] for a in client.get(f"/api/teams/{tid}/audit").json()]
    assert {"team_created", "team_invite", "project_shared", "team_role"} <= set(acts)


def test_single_sign_on_verifies_the_id_token(client, monkeypatch):
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from aerosculpt import sso
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.com")
    monkeypatch.setenv("OIDC_CLIENT_ID", "aerosculpt")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "s3cret")
    monkeypatch.setenv("OIDC_NAME", "Example SSO")
    disc = {"issuer": "https://idp.example.com", "authorization_endpoint": "https://idp.example.com/authorize",
            "token_endpoint": "https://idp.example.com/token", "jwks_uri": "https://idp.example.com/jwks"}
    monkeypatch.setattr(sso, "discovery", lambda: disc)
    state = {}

    def fake_token(url, data=None, timeout=15):
        now = int(time.time())
        claims = {"iss": disc["issuer"], "aud": "aerosculpt", "sub": "u-42", "email": "sso.user@example.com",
                  "email_verified": True, "name": "SSO User", "iat": now, "exp": now + 300, "nonce": state["nonce"]}
        return {"id_token": jwt.encode(claims, key, algorithm="RS256")}
    monkeypatch.setattr(sso, "_get_json", fake_token)

    class FakeJWKS:
        def __init__(self, uri):
            pass

        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": key.public_key()})()
    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKS)
    client.post("/api/auth/logout", headers=H)
    assert client.get("/api/auth/sso").json() == {"enabled": True, "name": "Example SSO"}
    r = client.get("/auth/sso/start", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith(disc["authorization_endpoint"])
    from urllib.parse import parse_qs, urlparse
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["code_challenge_method"] == ["S256"]
    state["nonce"] = sso._PENDING[q["state"][0]][0]
    r = client.get(f"/auth/sso/callback?code=abc&state={q['state'][0]}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/app"
    assert client.get("/api/me").json()["email"] == "sso.user@example.com"
    # a replayed or forged state is refused, and no session is made
    client.post("/api/auth/logout", headers=H)
    r = client.get(f"/auth/sso/callback?code=abc&state={q['state'][0]}", follow_redirects=False)
    assert r.status_code == 302 and "sso_error" in r.headers["location"]


def test_fusion_parameters_reach_the_bridge(monkeypatch, tmp_path):
    from aerosculpt import fusion
    seen = {}

    class FakeParam:
        def __init__(self, expr):
            self.expression = expr

    class FakeParams:
        def __init__(self):
            self.p = {"wing_span": FakeParam("2.0 m")}

        def itemByName(self, n):
            return self.p.get(n)

        def add(self, n, vi, unit, comment):
            self.p[n] = FakeParam(vi)

    class Bridge(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            assert self.headers.get("Authorization") == "Bearer test-secret"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            import io
            import sys
            import types
            core = types.SimpleNamespace(ValueInput=types.SimpleNamespace(createByString=lambda s: s))
            adsk = types.SimpleNamespace(core=core, fusion=types.SimpleNamespace())
            ups = FakeParams()
            design = types.SimpleNamespace(userParameters=ups)
            app_ = types.SimpleNamespace(activeDocument=types.SimpleNamespace(name="Drone frame"))
            out = io.StringIO()
            old, sys.stdout = sys.stdout, out
            try:
                ns = {"adsk": adsk, "app": app_, "ui": None, "design": design}
                sys.modules.setdefault("adsk", adsk)
                sys.modules.setdefault("adsk.core", core)
                sys.modules.setdefault("adsk.fusion", adsk.fusion)
                exec(body["script"], ns)
                ns["run"](None)
            finally:
                sys.stdout = old
            seen["params"] = {k: v.expression for k, v in ups.p.items()}
            data = json.dumps({"result": out.getvalue()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), Bridge)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    (tmp_path / "secret").write_text("test-secret")
    monkeypatch.setattr(fusion, "PORT", srv.server_address[1])
    monkeypatch.setattr(fusion, "SECRET_FILE", tmp_path / "secret")
    try:
        res = fusion.push_parameters([("wing_span", 2.4, "m", "Wing span"), ("wing_sweep", 3.0, "deg", "Sweep")])
    finally:
        srv.shutdown()
    assert res["document"] == "Drone frame"
    assert seen["params"]["wing_span"] == "2.4 m" and seen["params"]["wing_sweep"] == "3.0 deg"
    with pytest.raises(fusion.FusionError):
        fusion.push_parameters([("bad name; import os", 1.0, "m", "")])
