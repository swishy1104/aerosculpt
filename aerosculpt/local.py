"""Protection of a local, no-sign-in app (the desktop app and the local preview).

Two things a website the user happens to visit could otherwise do to an app on 127.0.0.1:

  DNS rebinding   a page resolves its own hostname to 127.0.0.1 and talks to the app as if it
                  were same-origin -> only requests whose Host header names this machine are
                  served (ALLOWED_HOSTS to add a name)
  driving the API the desktop app starts with a random launch token (LOCAL_TOKEN); the launch
                  link carries it once (?t=...), which sets a SameSite=Strict cookie; API and
                  app pages need that cookie. The custom-header requirement on writes (web.py)
                  already stops cross-site POSTs; the cookie also keeps other users of a shared
                  machine out.

Also here: the small per-machine settings of the desktop app (data/local_settings.json): Fusion
bridge on/off, crash files, update checks.
"""
from __future__ import annotations

import hmac
import json
import urllib.parse

from .config import settings

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
COOKIE = "as_local"
DEFAULTS = {"fusion_bridge": False, "crash_reports": True, "check_updates": True, "online_motors": True}


def allowed_host(host: str) -> bool:
    h = (host or "").split(":")[0].lower() if not host.startswith("[") else host.split("]")[0].lower() + "]"
    return h in LOCAL_HOSTS or h.endswith(".localhost") or h in settings.allowed_hosts


def local_settings() -> dict:
    f = settings.data_dir / "local_settings.json"
    try:
        return {**DEFAULTS, **json.loads(f.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_local_settings(changes: dict) -> dict:
    cur = local_settings()
    for k in DEFAULTS:
        if k in changes:
            cur[k] = bool(changes[k])
    (settings.data_dir / "local_settings.json").write_text(json.dumps(cur), encoding="utf-8")
    return cur


class LocalGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not settings.open_access:
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if not allowed_host(headers.get("host", "")):
            return await _plain(send, 403, "This AeroSculpt runs on this computer only and answers to localhost.")
        token = settings.local_token
        if not token:
            return await self.app(scope, receive, send)
        path = scope.get("path", "/")
        qs = urllib.parse.parse_qs(scope.get("query_string", b"").decode())
        given = (qs.get("t") or [""])[0]
        if given and hmac.compare_digest(given, token):
            # the launch link: set the cookie, drop the token from the address
            clean = urllib.parse.urlencode([(k, v) for k, vs in qs.items() if k != "t" for v in vs])
            target = path + (f"?{clean}" if clean else "")
            cookie = f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"
            return await _plain(send, 302, "", extra=[(b"location", target.encode()), (b"set-cookie", cookie.encode())])
        cookies = {}
        for part in headers.get("cookie", "").split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                cookies[k] = v
        if hmac.compare_digest(cookies.get(COOKIE, ""), token) or path.startswith("/static/"):
            return await self.app(scope, receive, send)
        if path.startswith("/api/"):
            return await _plain(send, 401, '{"error":"Open AeroSculpt from its launch link (shown in the app window) to use it from this browser."}',
                                ctype=b"application/json")
        return await _plain(send, 403, "<!doctype html><title>AeroSculpt</title><body style='font:15px system-ui;padding:40px;max-width:560px'>"
                            "<h2>Open AeroSculpt from its launch link</h2><p>This copy of AeroSculpt runs privately on this computer. "
                            "Use the link shown in the AeroSculpt window (it carries a one-time launch token), or restart the app to get a fresh one.</p>",
                            ctype=b"text/html; charset=utf-8")


async def _plain(send, status, body, ctype=b"text/plain; charset=utf-8", extra=()):
    data = body.encode() if isinstance(body, str) else body
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", ctype), (b"content-length", str(len(data)).encode()), (b"cache-control", b"no-store"), *extra]})
    await send({"type": "http.response.body", "body": data})
