"""Publishing the app: licence keys with offline grace, the local-app guard (Host header and
launch token), update checks, the diagnostic bundle, project files that round-trip, and the
version stamp on results."""
import io
import json
import time
import zipfile

import pytest
from starlette.testclient import TestClient

from aerosculpt import __version__, license as lic
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
    r = c.post("/api/auth/signup", json={"email": email, "password": "Correct-Horse-9", "name": "P", "accept_terms": True})
    assert r.status_code == 200, r.text


@pytest.fixture
def keypair():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    import base64
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = base64.urlsafe_b64encode(priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    return priv, pub


def test_licence_keys_verify_offline_and_expire_with_grace(keypair, monkeypatch):
    priv, pub = keypair
    monkeypatch.setattr(lic, "PUBLIC_KEY_B64", pub)
    key = lic.issue("Pilot@Example.com", "pro", days=30, private_key=priv)
    p = lic.verify(key)
    assert p["email"] == "pilot@example.com" and p["plan"] == "pro"
    now = time.time()
    assert lic.status(key, now)["valid"] and lic.status(key, now)["plan"] == "pro"
    g = lic.status(key, now + 40 * 86400)                      # expired, inside the 30-day grace
    assert g["valid"] and g.get("in_grace") and "renew" in g["why"]
    assert not lic.status(key, now + 70 * 86400)["valid"]       # past the grace period
    tampered = key[:-6] + ("A" if key[-6] != "A" else "B") + key[-5:]
    assert not lic.status(tampered, now)["valid"]
    with pytest.raises(lic.LicenseError):
        lic.verify("not.a.key")
    with pytest.raises(lic.LicenseError):
        lic.issue("x@y.z", "lite", private_key=priv)
    # a key signed by someone else's private key is rejected by our public key
    other, _ = keypair.__class__((None, None)) if False else (None, None)
    from cryptography.hazmat.primitives.asymmetric import ed25519
    forged = lic.issue("x@y.z", "team", private_key=ed25519.Ed25519PrivateKey.generate())
    assert "signature" in lic.status(forged, now)["why"]


def test_local_guard_blocks_rebinding_and_requires_the_launch_token(monkeypatch):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from aerosculpt.config import settings
    from aerosculpt.local import LocalGuard
    inner = Starlette(routes=[Route("/api/ping", lambda r: JSONResponse({"ok": True})), Route("/app", lambda r: JSONResponse({"page": True}))])
    guarded = LocalGuard(inner)
    monkeypatch.setattr(settings, "open_access", True)
    monkeypatch.setattr(settings, "local_token", "")
    with TestClient(guarded, base_url="http://localhost") as c:
        assert c.get("/api/ping").status_code == 200
        assert c.get("/api/ping", headers={"host": "evil.example.com"}).status_code == 403     # DNS rebinding
    monkeypatch.setattr(settings, "local_token", "tok123")
    with TestClient(guarded, base_url="http://127.0.0.1") as c:
        assert c.get("/api/ping").status_code == 401                 # no launch cookie
        assert c.get("/app").status_code == 403
        r = c.get("/app?t=tok123", follow_redirects=False)           # the launch link sets the cookie
        assert r.status_code == 302 and r.headers["location"] == "/app" and "as_local=tok123" in r.headers["set-cookie"]
        assert c.get("/api/ping").status_code == 200
        assert c.get("/app?t=wrong", follow_redirects=False).status_code == 200    # already allowed by the cookie; wrong token ignored
    monkeypatch.setattr(settings, "open_access", False)
    with TestClient(guarded, base_url="http://anything.example.com") as c:
        assert c.get("/api/ping").status_code == 200                 # a public server is not a local app


def test_update_check_reads_releases_and_survives_no_network(monkeypatch):
    from aerosculpt import update
    import urllib.request

    class R:
        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: R({"tag_name": "v9.9.0", "html_url": "https://x/rel",
                                                                             "assets": [{"name": "AeroSculpt-Setup-9.9.0.exe", "browser_download_url": "https://x/setup.exe"}]}))
    u = update.check("1.1.0", force=True)
    assert u["newer"] and u["latest"] == "9.9.0" and u["url"].endswith("setup.exe")

    def boom(req, timeout=0):
        raise OSError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    u = update.check("1.1.0", force=True)
    assert not u["newer"] and u["error"] == "OSError"


def test_project_files_round_trip_and_results_carry_the_version(client, keypair, monkeypatch):
    _signup(client, "platform@example.com")
    p = client.post("/api/projects", json={"template": "rocket", "name": "Round trip"}, headers=H).json()
    client.post(f"/api/projects/{p['id']}/rocket", json={"rocket": {"launch": {"wind_m_s": 3.5}, "units": "imperial"}, "save": True}, headers=H)
    log = "time,alt\n0,0\n1,50\n2,90\n3,100\n4,95\n"
    assert client.post(f"/api/projects/{p['id']}/rocket/flightlog", content=log, headers=H).status_code == 201
    r = client.get(f"/api/projects/{p['id']}/export", headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    meta = json.loads(z.read("project.json"))
    assert meta["format"] == "aerosculpt-project" and meta["app"] == __version__ and meta["settings"]["rocket"]["launch"]["wind_m_s"] == 3.5
    assert "model.stl" in z.namelist() and "flight_log.json" in z.namelist()
    q = client.post("/api/projects?filename=roundtrip.aerosculpt", content=r.content, headers=H)
    assert q.status_code == 201, q.text
    st = q.json()["settings"]
    assert q.json()["name"] == "Round trip" and st["template"]["id"] == "rocket" and st["rocket"]["launch"]["wind_m_s"] == 3.5 and st["rocket"]["units"] == "imperial"
    ev = client.post(f"/api/projects/{q.json()['id']}/rocket", json={}, headers=H).json()
    assert ev["flight_log"] and ev["launch"]["wind_m_s"] == 3.5
    assert client.post("/api/projects?filename=bad.aerosculpt", content=b"not a zip", headers=H).status_code == 400
    # a Pro subscriber gets a desktop key for their email; Lite does not
    assert client.get("/api/license", headers=H).json()["can_issue"] is False
    assert client.post("/api/license/issue", headers=H).status_code == 403
    from aerosculpt.db import db
    with db() as con:
        con.execute("UPDATE users SET plan='pro', plan_status='active' WHERE email='platform@example.com'")
    priv, pub = keypair
    monkeypatch.setattr(lic, "PUBLIC_KEY_B64", pub)
    monkeypatch.setattr(lic, "load_private_key", lambda source=None: priv)
    r = client.post("/api/license/issue", headers=H)
    assert r.status_code == 200, r.text
    st = lic.status(r.json()["key"])
    assert st["valid"] and st["plan"] == "pro" and st["email"] == "platform@example.com"
    # the version is stamped on every result and health answer
    assert client.get("/api/health").json()["version"] == __version__
    me = client.get("/api/me", headers=H).json()
    assert me["app_version"] == __version__ and me["build"]
    # diagnostics are for admins (and the local app), not plain users
    assert client.get("/api/diagnostics", headers=H).status_code == 403
    with db() as con:
        con.execute("UPDATE users SET is_admin=1 WHERE email='platform@example.com'")
    r = client.get("/api/diagnostics", headers=H)
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "about.json" in names and "jobs.json" in names
    about = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("about.json"))
    assert about["app"] == __version__ and "numpy" in about["versions"] and "gpu" in about


def test_result_files_are_stamped(tmp_path):
    from aerosculpt.optimize.runner import Runner
    r = Runner.__new__(Runner)
    r.out = tmp_path
    r.write_json("result.json", {"kind": "analyze"})
    d = json.loads((tmp_path / "result.json").read_text())
    assert d["version"] == __version__ and d["build"]
