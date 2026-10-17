"""Account-free public use: every visitor gets a private temporary workspace without any sign-in."""
import pytest
from starlette.testclient import TestClient

from aerosculpt.config import settings
from aerosculpt.web import app

H = {"X-Requested-With": "AeroSculpt"}


@pytest.fixture
def anon(monkeypatch):
    from aerosculpt.web import guest_limiter
    guest_limiter.hits.clear()
    monkeypatch.setattr(settings, "anonymous_access", True)
    yield


def test_visitors_get_their_own_workspace_without_signing_in(anon):
    with TestClient(app) as a, TestClient(app) as b:
        r = a.get("/app")
        assert r.status_code == 200 and "as_session" in r.headers.get("set-cookie", "")
        me = a.get("/api/me").json()
        assert me["anonymous"] and me["guest"] and me["plan"] == "public" and me["limits"]["export"] and "standard" in me["limits"]["fidelities"]
        p = a.post("/api/projects", json={"template": "rocket", "name": "mine"}, headers=H)
        assert p.status_code == 201, p.text
        assert [q["name"] for q in a.get("/api/projects").json()] == ["mine"]
        # another browser is another visitor: nothing shared
        assert b.get("/api/projects").status_code == 200 and b.get("/api/projects").json() == []
        assert b.get("/api/me").json()["id"] != me["id"]
        assert b.get(f"/api/projects/{p.json()['id']}").status_code == 404
        # the flight model works for a visitor
        ev = a.post(f"/api/projects/{p.json()['id']}/rocket", json={}, headers=H).json()
        assert ev["flight"]["apogee_m"] > 100
        # no sign-up page; the owner's sign-in page stays reachable; the landing page knows
        assert a.get("/signup", follow_redirects=False).status_code == 302
        assert a.get("/login").status_code == 200
        assert a.get("/api/plans").json()["anonymous"] is True
        # an API key still identifies a real account, not a visitor
        assert a.get("/api/me", headers={"Authorization": "Bearer as_nothing"}).status_code in (401, 403)


def test_anonymous_mode_is_off_by_default():
    assert settings.anonymous_access is False
    with TestClient(app) as c:
        assert c.get("/api/me").status_code == 401
