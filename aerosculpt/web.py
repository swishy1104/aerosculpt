"""HTTP layer: JSON API, Stripe webhook, static pages."""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.parse
import re
import secrets
import logging
import math
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import __version__, billing, build_id, emailer
from .config import ensure_dirs, settings
from .db import add_usage, audit, db, init_db, transaction, usage_for
from .jobs import enqueue, job_dir, project_dir
from .geometry import templates as T
from .plans import (ALL_ASPECTS, ALL_OBJECTIVES, CREDITS_PER_RUN, EXPLORE_FID, PLANS, ROCKET_OBJECTIVES, check_spec, estimate_credits,
                    plan_for)
from .security import (DUMMY_HASH, RateLimiter, auth_limiter, check_password, hash_password, new_api_key, new_token,
                       normalize_email, password_problem, token_hash, upload_limiter, valid_email)

log = logging.getLogger("aerosculpt.web")
STATIC = Path(__file__).resolve().parent / "static"
PAGES = STATIC / "pages"
SESSION_COOKIE = "as_session"
CSRF_HEADER = "x-requested-with"
VERSION = __version__
JOB_FILES = {"result.json", "history.json", "baseline_body.stl", "best_body.stl", "optimized.stl", "promising_body.stl", "promising.stl"}
JOB_FIELD_FILE = re.compile(r"^fields_(baseline|best)_\d{1,3}\.json$")
AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


class ApiError(Exception):
    def __init__(self, status, message, **extra):
        self.status = status
        self.message = message
        self.extra = extra


def err(status, message, **extra):
    raise ApiError(status, message, **extra)


# ---- auth helpers -----------------------------------------------------------------------
def _client_ip(request: Request):
    # uvicorn already resolved X-Forwarded-For from trusted proxies (FORWARDED_ALLOW_IPS);
    # never read the raw header here, or clients could spoof their way past rate limits
    return request.client.host if request.client else "?"


def current_user(request: Request, con, required=True):
    auth = request.headers.get("authorization", "")
    user = None
    via = None
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        user = con.execute("SELECT * FROM users WHERE api_key_hash=?", (token_hash(key),)).fetchone()
        via = "api_key"
        if user is not None and not plan_for(user)["api"]:
            err(403, "API access requires the Team plan.")
    else:
        tok = request.cookies.get(SESSION_COOKIE)
        if tok:
            row = con.execute("SELECT user_id, expires_at FROM sessions WHERE token_hash=?", (token_hash(tok),)).fetchone()
            if row and row["expires_at"] > time.time():
                user = con.execute("SELECT * FROM users WHERE id=?", (row["user_id"],)).fetchone()
                via = "cookie"
    if user is None and settings.open_access and not auth:
        user = _preview_user(con)
        via = "preview"
    if user is None:
        if required:
            err(401, "Please sign in.")
        return None
    if via == "cookie" and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get(CSRF_HEADER) != "AeroSculpt":
            err(403, "Missing request header.")
    if billing.refresh_demo(con, user):
        user = con.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    return user


PREVIEW_EMAIL = "preview@aerosculpt.local"
GUEST_DOMAIN = "guest.aerosculpt.local"
guest_limiter = RateLimiter(limit=4, window_s=3600)
anon_limiter = RateLimiter(limit=40, window_s=3600)      # new visitor workspaces per address per hour (shared addresses are common)
ai_limiter = RateLimiter(limit=30, window_s=3600)


def _preview_user(con):
    """Local preview mode (OPEN_ACCESS): everyone is one built-in user with every feature
    unlocked, so the app can be opened and tweaked without signing in."""
    row = con.execute("SELECT * FROM users WHERE email=?", (PREVIEW_EMAIL,)).fetchone()
    if row is None:
        con.execute("""INSERT OR IGNORE INTO users (email, name, password_hash, created_at, plan, plan_status,
                       billing_source, is_admin) VALUES (?, 'Local preview', '!', ?, 'team', 'active', 'preview', 1)""",
                    (PREVIEW_EMAIL, time.time()))
        row = con.execute("SELECT * FROM users WHERE email=?", (PREVIEW_EMAIL,)).fetchone()
    # the desktop app: features by the installed licence key (the free desktop tier without one)
    want = settings.open_access_plan
    if want == "license":
        from . import license as lic
        st = lic.installed()
        want = st["plan"] if st.get("valid") else "desktop"
    if want in PLANS and row["plan"] != want:
        con.execute("UPDATE users SET plan=?, plan_status='active', billing_source=? WHERE id=?",
                    (want, "license" if settings.open_access_plan == "license" else "preview", row["id"]))
        row = con.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone()
    return row


def _insert_session(con, user_id, ip, user_agent, age):
    tok = new_token()
    now = time.time()
    con.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at, ip, user_agent) VALUES (?,?,?,?,?,?)",
                (token_hash(tok), user_id, now, now + age, ip, (user_agent or "")[:300]))
    con.execute("UPDATE users SET last_login_at=? WHERE id=?", (now, user_id))
    con.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    return tok


def start_session(con, request, user_id, response: Response, max_age_s=None):
    age = int(max_age_s or settings.session_days * 86400)
    tok = _insert_session(con, user_id, _client_ip(request), request.headers.get("user-agent", ""), age)
    response.set_cookie(SESSION_COOKIE, tok, max_age=age, httponly=True,
                        secure=settings.secure_cookies, samesite="lax", path="/")


def new_guest(con, ip, plan="guest"):
    """A temporary account (jobs.purge_guests deletes it, with its projects, after GUEST_HOURS)."""
    email = f"guest-{secrets.token_hex(6)}@{GUEST_DOMAIN}"
    cur = con.execute("""INSERT INTO users (email, name, password_hash, created_at, plan, plan_status, billing_source)
                         VALUES (?, ?, '!', ?, ?, 'active', 'guest')""", (email, "Visitor" if plan == "public" else "Guest", time.time(), plan))
    audit(con, cur.lastrowid, "guest_created", ip)
    return cur.lastrowid


class AnonymousAccess:
    """Account-free public use (ANONYMOUS_ACCESS=1): a visitor without a session gets a temporary
    account of their own, on the Public plan, the moment they open the app or call the API; the
    session cookie keeps their projects theirs and nobody else's. No sign-in or sign-up is shown
    (the owner can still reach /login). Too many new visitors from one address, or too many live
    visitors, are asked to come back later."""

    SKIP = ("/api/auth/", "/api/health", "/api/billing/webhook", "/api/plans", "/api/samples", "/api/validation", "/api/me",
            "/auth/", "/static/")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not settings.anonymous_access:
            return await self.app(scope, receive, send)
        path = scope.get("path", "/")
        if not (path.startswith("/app") or path.startswith("/api/")) or path.startswith(self.SKIP):
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if headers.get("authorization", "").lower().startswith("bearer "):
            return await self.app(scope, receive, send)
        cookies = {}
        for part in headers.get("cookie", "").split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                cookies[k] = v
        tok = cookies.get(SESSION_COOKIE)
        if tok:
            with db() as con:
                row = con.execute("SELECT expires_at FROM sessions WHERE token_hash=?", (token_hash(tok),)).fetchone()
            if row and row["expires_at"] > time.time():
                return await self.app(scope, receive, send)
        ip = scope.get("client")[0] if scope.get("client") else "?"
        if not anon_limiter.allow("anon:" + ip):
            return await _refuse(send, path, 429, "Too many new visitors from this network. Try again in an hour.")
        age = int(settings.guest_hours * 3600)
        with transaction() as con:
            live = con.execute("SELECT COUNT(*) AS n FROM users WHERE billing_source='guest' AND created_at > ?",
                               (time.time() - settings.guest_hours * 3600,)).fetchone()["n"]
            if live >= settings.anon_max_active:
                return await _refuse(send, path, 503, "AeroSculpt is full right now. Please try again later.")
            uid = new_guest(con, ip, "public")
            tok = _insert_session(con, uid, ip, headers.get("user-agent", ""), age)
        cookie = f"{SESSION_COOKIE}={tok}"
        scope["headers"] = [(k, v) for k, v in scope["headers"] if k.lower() != b"cookie"] + \
            [(b"cookie", ((headers.get("cookie", "") + "; ") if headers.get("cookie") else "").encode("latin-1") + cookie.encode())]
        set_cookie = f"{cookie}; Path=/; Max-Age={age}; HttpOnly; SameSite=Lax" + ("; Secure" if settings.secure_cookies else "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"set-cookie", set_cookie.encode()))
            await send(message)
        return await self.app(scope, receive, send_wrapper)


async def _refuse(send, path, status, text):
    api = path.startswith("/api/")
    body = json.dumps({"error": text}).encode() if api else f"<!doctype html><title>AeroSculpt</title><body style='font:15px system-ui;padding:40px'><h2>{text}</h2>".encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json" if api else b"text/html; charset=utf-8"), (b"content-length", str(len(body)).encode()),
                            (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


async def body_json(request: Request, limit=1_000_000):
    raw = await request.body()
    if len(raw) > limit:
        err(413, "Request too large.")
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        err(400, "Invalid JSON.")
    if not isinstance(data, dict):
        err(400, "Expected a JSON object.")
    return data


def user_public(con, user):
    plan = plan_for(user)
    usage = usage_for(con, user["id"])
    return {
        "id": user["id"], "email": user["email"], "name": user["name"], "is_admin": bool(user["is_admin"]),
        "plan": plan["key"], "plan_name": plan["name"], "plan_status": user["plan_status"],
        "subscribed_plan": user["plan"],
        "trial_ends_at": user["trial_ends_at"], "trial_used": bool(user["trial_used"]),
        "current_period_end": user["current_period_end"], "cancel_at_period_end": bool(user["cancel_at_period_end"]),
        "billing_source": user["billing_source"], "billing_mode": settings.billing_mode,
        "preview": user["email"] == PREVIEW_EMAIL and settings.open_access,
        "app_version": VERSION, "build": build_id(), "local": settings.open_access,
        "guest": user["billing_source"] == "guest",
        "anonymous": settings.anonymous_access and user["billing_source"] == "guest",
        "credits_off": plan["monthly_credits"] >= 10 ** 8,        # no credit meter on this site (the Public plan)
        "guest_expires_at": user["created_at"] + settings.guest_hours * 3600 if user["billing_source"] == "guest" else None,
        "has_api_key": bool(user["api_key_hash"]), "api_key_prefix": user["api_key_prefix"],
        "limits": {k: plan[k] for k in ("monthly_credits", "max_iterations", "fidelities", "objectives",
                                        "max_aspects", "aspects", "max_angles", "max_speeds", "max_zones",
                                        "max_propellers", "max_iterations_panel", "max_top_k", "max_features",
                                        "max_projects", "max_upload_mb", "max_triangles",
                                        "export", "verify", "reports", "api", "concurrent_jobs")},
        "usage": usage, "credits_left": max(0, plan["monthly_credits"] - usage["credits"]),
        "created_at": user["created_at"],
    }


# ---- auth endpoints -------------------------------------------------------------------
def _apply_owner(con, uid, email):
    """Owners of a private deployment get admin and the Team plan, outside billing."""
    if email in settings.owner_emails:
        con.execute("""UPDATE users SET is_admin=1, plan='team', plan_status='active', billing_source='owner'
                       WHERE id=? AND (plan!='team' OR plan_status!='active' OR is_admin=0)""", (uid,))


async def signup(request: Request):
    data = await body_json(request)
    email = normalize_email(data.get("email"))
    pw = data.get("password") or ""
    name = (data.get("name") or "").strip()[:80]
    if not auth_limiter.allow("signup:" + _client_ip(request)):
        err(429, "Too many attempts. Try again in a few minutes.")
    if not valid_email(email):
        err(400, "Enter a valid email address.", field="email")
    if settings.allowed_signup_emails and email not in settings.allowed_signup_emails:
        err(403, "Sign-up on this server is by invitation only.", field="email")
    p = password_problem(pw)
    if p:
        err(400, p, field="password")
    if not data.get("accept_terms"):
        err(400, "Please accept the Terms of Service.", field="accept_terms")
    hashed = await run_in_threadpool(hash_password, pw)
    resp = JSONResponse({"ok": True})
    with transaction() as con:
        if con.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            err(409, "An account with this email already exists. Sign in instead.", field="email")
        # the first real account is the admin; visitors, guests and the local preview user do not count
        first = con.execute("SELECT COUNT(*) AS n FROM users WHERE billing_source IS NULL OR billing_source NOT IN ('guest', 'preview')").fetchone()["n"] == 0
        cur = con.execute("INSERT INTO users (email, name, password_hash, created_at, is_admin) VALUES (?,?,?,?,?)",
                          (email, name, hashed, time.time(), 1 if first else 0))
        uid = cur.lastrowid
        _apply_owner(con, uid, email)
        audit(con, uid, "signup", email)
        from .teams import accept_invites
        accept_invites(con, uid, email)
        start_session(con, request, uid, resp)
    return resp


async def claim(request: Request):
    """Account-free site: turn this visitor's temporary workspace into a permanent account (its
    projects included) with an email and password, so it is not cleared after GUEST_HOURS."""
    if not settings.anonymous_access:
        err(404, "Not found.")
    with db() as con:
        user = current_user(request, con)
    if user["billing_source"] != "guest":
        err(400, "This workspace already belongs to an account.")
    data = await body_json(request)
    email = normalize_email(data.get("email"))
    pw = data.get("password") or ""
    name = (data.get("name") or "").strip()[:80]
    if not auth_limiter.allow("signup:" + _client_ip(request)):
        err(429, "Too many attempts. Try again in a few minutes.")
    if not valid_email(email):
        err(400, "Enter a valid email address.", field="email")
    p = password_problem(pw)
    if p:
        err(400, p, field="password")
    if not data.get("accept_terms"):
        err(400, "Please accept the Terms of Service.", field="accept_terms")
    hashed = await run_in_threadpool(hash_password, pw)
    resp = JSONResponse({"ok": True, "email": email})
    with transaction() as con:
        if con.execute("SELECT 1 FROM users WHERE email=? AND id!=?", (email, user["id"])).fetchone():
            err(409, "An account with this email already exists. Sign in instead.", field="email")
        con.execute("UPDATE users SET email=?, name=?, password_hash=?, billing_source=NULL, plan='public', plan_status='active' WHERE id=?",
                    (email, name, hashed, user["id"]))
        _apply_owner(con, user["id"], email)
        audit(con, user["id"], "claimed", email)
        from .teams import accept_invites
        accept_invites(con, user["id"], email)
        start_session(con, request, user["id"], resp)       # a full-length session instead of the visitor's
    return resp


async def guest(request: Request):
    """The public demo: a temporary account on the Guest plan, signed in at once, no email asked.
    jobs.purge_guests deletes it (projects, runs and files) after GUEST_HOURS."""
    if not settings.guest_demo:
        err(404, "The guest demo is not enabled on this server.")
    ip = _client_ip(request)
    if not guest_limiter.allow("guest:" + ip):
        err(429, "Too many demo accounts from this network. Try again in an hour.")
    resp = JSONResponse({"ok": True})
    with transaction() as con:
        live = con.execute("SELECT COUNT(*) AS n FROM users WHERE billing_source='guest' AND created_at > ?",
                           (time.time() - settings.guest_hours * 3600,)).fetchone()["n"]
        if live >= settings.guest_max_active:
            err(503, "The demo is full right now. Please try again later.")
        uid = new_guest(con, ip)
        start_session(con, request, uid, resp, max_age_s=settings.guest_hours * 3600)
    return resp


async def login(request: Request):
    data = await body_json(request)
    email = normalize_email(data.get("email"))
    pw = data.get("password") or ""
    if not auth_limiter.allow("login:" + _client_ip(request)) or not auth_limiter.allow("login:" + email):
        err(429, "Too many attempts. Try again in a few minutes.")
    with db() as con:
        user = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    ok = await run_in_threadpool(check_password, pw, user["password_hash"] if user else DUMMY_HASH)
    if not user or not ok:
        err(401, "Email or password is incorrect.")
    resp = JSONResponse({"ok": True})
    with db() as con:
        _apply_owner(con, user["id"], email)
        from .teams import accept_invites
        accept_invites(con, user["id"], email)
        start_session(con, request, user["id"], resp)
        audit(con, user["id"], "login", _client_ip(request))
    return resp


async def logout(request: Request):
    tok = request.cookies.get(SESSION_COOKIE)
    if tok:
        with db() as con:
            con.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(tok),))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


async def forgot(request: Request):
    data = await body_json(request)
    email = normalize_email(data.get("email"))
    if not auth_limiter.allow("forgot:" + _client_ip(request)):
        err(429, "Too many attempts. Try again in a few minutes.")
    with db() as con:
        user = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user:
            tok = new_token()
            con.execute("INSERT INTO password_resets (token_hash, user_id, expires_at) VALUES (?,?,?)",
                        (token_hash(tok), user["id"], time.time() + 3600))
            link = f"{settings.public_url}/reset?token={tok}"
            await run_in_threadpool(emailer.send, email, f"Reset your {settings.app_name} password",
                                    f"Someone asked to reset the password for this account.\n\n"
                                    f"Open this link within one hour to choose a new password:\n{link}\n\n"
                                    f"If it was not you, ignore this email.")
    # same answer either way so the endpoint cannot be used to discover accounts
    return JSONResponse({"ok": True})


async def reset(request: Request):
    data = await body_json(request)
    tok = data.get("token") or ""
    pw = data.get("password") or ""
    p = password_problem(pw)
    if p:
        err(400, p, field="password")
    hashed = await run_in_threadpool(hash_password, pw)
    resp = JSONResponse({"ok": True})
    with transaction() as con:
        row = con.execute("SELECT * FROM password_resets WHERE token_hash=?", (token_hash(tok),)).fetchone()
        if not row or row["used"] or row["expires_at"] < time.time():
            err(400, "This reset link is invalid or has expired.")
        con.execute("UPDATE password_resets SET used=1 WHERE token_hash=?", (row["token_hash"],))
        con.execute("UPDATE users SET password_hash=? WHERE id=?", (hashed, row["user_id"]))
        con.execute("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        audit(con, row["user_id"], "password_reset")
        start_session(con, request, row["user_id"], resp)
    return resp


async def me(request: Request):
    with db() as con:
        user = current_user(request, con)
        if request.method == "PATCH":
            data = await body_json(request)
            name = (data.get("name") or "").strip()[:80]
            con.execute("UPDATE users SET name=? WHERE id=?", (name, user["id"]))
            user = con.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        return JSONResponse(user_public(con, user))


async def change_password(request: Request):
    data = await body_json(request)
    with db() as con:
        user = current_user(request, con)
    if not await run_in_threadpool(check_password, data.get("current") or "", user["password_hash"]):
        err(400, "Current password is incorrect.", field="current")
    p = password_problem(data.get("new") or "")
    if p:
        err(400, p, field="new")
    hashed = await run_in_threadpool(hash_password, data["new"])
    with db() as con:
        con.execute("UPDATE users SET password_hash=? WHERE id=?", (hashed, user["id"]))
        tok = request.cookies.get(SESSION_COOKIE)
        con.execute("DELETE FROM sessions WHERE user_id=? AND token_hash != ?", (user["id"], token_hash(tok or "")))
        audit(con, user["id"], "password_change")
    return JSONResponse({"ok": True})


async def delete_account(request: Request):
    data = await body_json(request)
    with db() as con:
        user = current_user(request, con)
    if not await run_in_threadpool(check_password, data.get("password") or "", user["password_hash"]):
        err(400, "Password is incorrect.", field="password")
    if user["stripe_subscription_id"] and settings.billing_mode == "stripe":
        try:
            billing.stripe("DELETE", f"/subscriptions/{user['stripe_subscription_id']}")
        except billing.BillingError as e:
            err(502, f"Could not cancel the subscription first: {e}")
    with db() as con:
        pids = [r["id"] for r in con.execute("SELECT id FROM projects WHERE user_id=?", (user["id"],))]
        jids = [r["id"] for r in con.execute("SELECT id FROM jobs WHERE user_id=?", (user["id"],))]
        con.execute("DELETE FROM users WHERE id=?", (user["id"],))
        audit(con, None, "account_deleted", str(user["id"]))
    for pid in pids:
        shutil.rmtree(project_dir(pid), ignore_errors=True)
    for jid in jids:
        shutil.rmtree(job_dir(jid), ignore_errors=True)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


async def api_key(request: Request):
    with db() as con:
        user = current_user(request, con)
        if not plan_for(user)["api"]:
            err(403, "API keys are available on the Team plan.", upgrade=True)
        if request.method == "DELETE":
            con.execute("UPDATE users SET api_key_hash=NULL, api_key_prefix=NULL WHERE id=?", (user["id"],))
            return JSONResponse({"ok": True})
        raw, prefix = new_api_key()
        con.execute("UPDATE users SET api_key_hash=?, api_key_prefix=? WHERE id=?", (token_hash(raw), prefix, user["id"]))
        audit(con, user["id"], "api_key_created", prefix)
    return JSONResponse({"key": raw, "prefix": prefix})


# ---- catalog ------------------------------------------------------------------------------
def ai_info():
    from . import ai
    return ai.info()


def FT_catalog():
    from .geometry import features as FT
    return FT.catalog()


def compute_info():
    """What the job workers run on and how long a CFD run takes there, measured by the worker
    at start-up (jobs.warm_up). Before the first measurement: a laptop CPU's typical figure."""
    from .cfd.analysis import FIDELITY, seconds_per_run
    try:
        info = json.loads((settings.data_dir / "compute.json").read_text())
        mlups = float(info["mlups"])
    except (OSError, ValueError, KeyError, TypeError):
        info, mlups = {"backend": "cpu", "device": None}, 30.0
    return {"backend": info.get("backend", "cpu"), "device": info.get("device"), "mlups": mlups,
            "seconds_per_run": {k: round(seconds_per_run(f.cells, mlups), 1) for k, f in FIDELITY.items()}}


async def plans(request: Request):
    from .geometry.ffd import ASPECTS
    from .optimize.runner import OBJECTIVES, objective_available
    from .cfd.analysis import FIDELITY
    from .hyper.materials import MATERIALS
    from .hyper.runner import OBJECTIVES as HYPER_OBJECTIVES
    from .rc.propeller import FAMILIES as PROP_FAMILIES
    return JSONResponse({
        "plans": [{k: v for k, v in p.items()} for p in PLANS.values() if not p.get("hidden")],
        "guest_demo": settings.guest_demo, "signup_open": not settings.allowed_signup_emails,
        "trial_days": settings.trial_days, "billing_mode": settings.billing_mode,
        "credits_per_run": CREDITS_PER_RUN,
        "aspects": {k: {"label": a["label"], "help": a["help"],
                        "vars": [{"key": v.key, "label": v.label} for v in a["vars"]]} for k, a in ASPECTS.items()},
        "objectives": {**OBJECTIVES, "drag_fixed_lift": "Minimize drag at fixed lift"},
        "objectives_unvalidated": [k for k in OBJECTIVES if not objective_available(k)],
        "fidelities": {k: {"cells": f.cells, "re_cap": f.re_cap} for k, f in FIDELITY.items() if not f.internal},
        "compute": compute_info(),
        "explore_fidelity": EXPLORE_FID,
        "templates": T.catalog(),
        "features": FT_catalog(),
        "ai": ai_info(),
        "materials": MATERIALS,
        "hyper_objectives": dict(HYPER_OBJECTIVES),
        "anonymous": settings.anonymous_access,
        "rocket_objectives": dict(ROCKET_GOALS),
        "prop_families": {k: {"name": f["name"], "note": f["note"]} for k, f in PROP_FAMILIES.items()},
    })


async def propeller_estimate(request: Request):
    """Blade-element thrust, power and slipstream of each propeller across a set of airspeeds:
    instant feedback while choosing a propeller, before any CFD is run."""
    from .cfd.analysis import isa
    from .rc.propeller import INCH, Propeller, bemt
    with db() as con:
        current_user(request, con)
    data = await body_json(request)
    props = _clean_props(data.get("propellers") or [])
    speeds = [_num(v, "airspeed") for v in (data.get("speeds") or [])][:25]
    if not speeds or min(speeds) < 1.0 or max(speeds) > 450.0:
        err(400, "Give airspeeds between 1 and 450 m/s.")
    altitude = _num(data.get("altitude", 0.0), "altitude")
    air = isa(altitude)
    out = []
    for p in props:
        prop = Propeller(p["diameter_in"] * INCH, p["pitch_in"] * INCH, p["blades"], p["width"], family=p["family"])
        area = math.pi * (prop.diameter / 2) ** 2
        rows = []
        for v in speeds:
            r = bemt(prop, v, p["rpm"], air["rho"], air["mu"], air["a"], n=24)
            ctd = max(r["thrust_N"], 0.0) / (0.5 * air["rho"] * v * v * area)
            rows.append({"v": v, "thrust_N": r["thrust_N"], "power_W": r["power_W"], "eta": r["eta"], "J": r["J"],
                         "slipstream": math.sqrt(1.0 + ctd), "tip_mach": r["tip_mach"]})
        out.append({"id": p["id"], "rows": rows})
    return JSONResponse({"propellers": out})


async def templates_catalog(request: Request):
    return JSONResponse(T.catalog())


async def template_build(request: Request):
    """Geometry for a set of template values, for the live 3D view (no simulation). Triangles as
    base64 float32 (file frame: metres, nose tip at the origin, nose towards -x)."""
    with db() as con:
        current_user(request, con)
    data = await body_json(request, limit=100_000)
    tid = data.get("template")
    if tid not in T.TEMPLATES:
        err(400, "Unknown template.")
    vals = T.clean(tid, data.get("values") or {})
    b = await run_in_threadpool(T.build, tid, vals, bool(data.get("preview", True)))
    tri = b.mesh.triangles().astype("<f4")
    return JSONResponse({"positions": base64.b64encode(tri.tobytes()).decode("ascii"), "faces": int(len(tri)),
                         "info": b.info, "problems": b.problems, "ref": b.ref, "values": vals})


def _model_frame(project, st):
    """The project's model in the optimizer's body frame (optimize.runner.ModelFrame)."""
    from .optimize.runner import ModelFrame, load_model
    path = project_dir(project["id"]) / ("model." + project["filename"].rsplit(".", 1)[-1].lower())
    raw = load_model(path, project["filename"])
    return ModelFrame(raw, st.get("units", "m"), st.get("forward", "-x"), st.get("up", "+z"), st.get("unit_scale"))


def _features_body(project, st, features):
    from .geometry import features as FT
    fr = _model_frame(project, st)
    ff = FT.Frame(fr.scale, fr.R0, fr.center)
    body, problems, info = FT.apply(fr.body, features, ff) if features else (fr.body, [], {})
    return fr, ff, body, problems, info


async def feature_suggest(request: Request):
    """A new shape feature fitted to the model where the user clicked (file coordinates), on top
    of the features already there."""
    from .geometry import features as FT
    with db() as con:
        user = current_user(request, con)
        project = _get_project(con, user, request.path_params["pid"], need="write")
    data = await body_json(request, limit=200_000)
    if data.get("type") not in FT.FEATURES:
        err(400, "Unknown shape feature.")
    st = _migrate(json.loads(project["settings"] or "{}"))
    existing = [f for f in _clean_features(data.get("features", st.get("features") or [])) if f.get("enabled", True)]

    def work():
        fr, ff, body, _, _ = _features_body(project, st, existing)
        point = ff.point(_num3(data["point"], "point")) if data.get("point") is not None else None
        normal = _num3(data["normal"], "normal") if data.get("normal") is not None else None
        return FT.suggest(data["type"], body, ff, point, normal)

    try:
        f = await run_in_threadpool(work)
    except ValueError as e:
        err(400, str(e))
    return JSONResponse(f)


async def feature_propose(request: Request):
    """Generative ideas: features worth trying on this model (boat-tail, rounded front, rear slant,
    diffuser), each optional so the optimizer adds only those that beat the measured error."""
    from .geometry import features as FT
    with db() as con:
        user = current_user(request, con)
        project = _get_project(con, user, request.path_params["pid"], need="write")
    data = await body_json(request, limit=500_000)
    st = _migrate(json.loads(project["settings"] or "{}"))
    existing = [f for f in _clean_features(data.get("features", st.get("features") or [])) if f.get("enabled", True)]

    def work():
        fr, ff, body, _, _ = _features_body(project, st, [f for f in existing if not f.get("optional")])
        return FT.propose(body, ff)

    try:
        return JSONResponse(await run_in_threadpool(work))
    except ValueError as e:
        err(400, str(e))


async def feature_ai(request: Request):
    """The AI design assistant: ideas from the generative menu chosen and sized by a language model
    from the brief, the body and its last result, each built as an optional feature."""
    from . import ai
    from .geometry import features as FT
    with db() as con:
        user = current_user(request, con)
        project = _get_project(con, user, request.path_params["pid"], need="write")
        last = con.execute("""SELECT summary FROM jobs WHERE project_id=? AND status='done' AND summary IS NOT NULL
                              ORDER BY finished_at DESC LIMIT 1""", (project["id"],)).fetchone()
    if not ai.available():
        err(404, "The AI assistant is not set up on this server.")
    if not ai_limiter.allow(f"ai:{user['id']}"):
        err(429, "Too many AI requests this hour.")
    data = await body_json(request, limit=500_000)
    st = _migrate(json.loads(project["settings"] or "{}"))
    existing = [f for f in _clean_features(data.get("features", st.get("features") or [])) if f.get("enabled", True)]
    brief = str(data.get("brief") or "")[:1500]

    def work():
        fr, ff, body, _, _ = _features_body(project, st, [f for f in existing if not f.get("optional")])
        summary = FT.body_summary(body)
        answer = ai.ask(brief, summary, json.loads(last["summary"]) if last else None)
        out = []
        for it in answer["ideas"]:
            f = FT.idea(it["idea"], body, ff, it["hints"])
            if f is not None:
                f["name"] = f["name"].replace("(proposed)", "(AI)")
                f["why"] = it["why"]
                out.append(f)
        return {"ideas": out, "notes": answer["notes"], "model": ai.info()["model"]}

    try:
        return JSONResponse(await run_in_threadpool(work))
    except (ValueError, RuntimeError) as e:
        err(502, f"AI assistant: {e}")
    except Exception as e:  # noqa: BLE001 - network trouble, timeouts
        err(502, f"The AI assistant did not answer ({type(e).__name__}).")


def _num3(v, what):
    vals = [_num(x, what) for x in (v or [])]
    if len(vals) != 3:
        err(400, f"Invalid {what}.")
    return vals


async def feature_build(request: Request):
    """The model with its shape features applied, for the live 3D view (no simulation):
    triangles as base64 float32 in the body frame, plus what is buried and any problems."""
    from .geometry import checks
    with db() as con:
        user = current_user(request, con)
        project = _get_project(con, user, request.path_params["pid"], need="write")
    data = await body_json(request, limit=500_000)
    st = _migrate(json.loads(project["settings"] or "{}"))
    feats = [f for f in _clean_features(data.get("features") or []) if f.get("enabled", True)]

    def work():
        fr, ff, body, problems, info = _features_body(project, st, feats)
        bad = checks.check(body) if feats and body.n_faces <= 300_000 else None
        return body, problems, info, bad

    body, problems, info, bad = await run_in_threadpool(work)
    tri = body.triangles().astype("<f4")
    hidden = body.hidden if getattr(body, "hidden", None) is not None else np.zeros(body.n_faces, bool)
    return JSONResponse({"positions": base64.b64encode(tri.tobytes()).decode("ascii"), "faces": int(len(tri)),
                         "hidden": base64.b64encode(np.packbits(hidden).tobytes()).decode("ascii"),
                         "problems": problems + ([f"the result is {bad}"] if bad else []), "info": info})


TEMPLATE_ENVELOPES = {
    "rocket": {"speed_min": 40.0, "speed_max": 170.0, "speed_count": 4, "aoa_min": 0.0, "aoa_max": 6.0, "aoa_count": 3},
    "uav": {"speed_min": 14.0, "speed_max": 26.0, "speed_count": 3, "aoa_min": -2.0, "aoa_max": 6.0, "aoa_count": 3},
    "wing": {"speed_min": 15.0, "speed_max": 30.0, "speed_count": 3, "aoa_min": 0.0, "aoa_max": 8.0, "aoa_count": 3},
}


def _template_files(tid, values):
    """Build a template at full resolution: (Built, STL bytes)."""
    b = T.build(tid, values)
    return b, b.mesh.to_stl_bytes(f"AeroSculpt template {tid}".encode())


def _clean_template(data, old):
    tid = old["id"]
    values = T.clean(tid, {**(old.get("values") or {}), **(data.get("values") or {})})
    params = {k: dict(v) for k, v in (old.get("params") or {}).items()}
    by_key = {p.key: p for p in T.get(tid)["params"]}
    for k, cfg in (data.get("params") or {}).items():
        p = by_key.get(k)
        if p is None or not isinstance(cfg, dict):
            continue
        c = params.setdefault(k, {})
        if "locked" in cfg:
            c["locked"] = bool(cfg["locked"])
        if "bounds" in cfg and p.kind == "float":
            try:
                lo, hi = (float(x) for x in cfg["bounds"])
            except (TypeError, ValueError):
                err(400, f"Invalid bounds for {p.label}.")
            if not (math.isfinite(lo) and math.isfinite(hi)):
                err(400, f"Invalid bounds for {p.label}.")
            lo, hi = sorted((min(max(lo, p.lo), p.hi), min(max(hi, p.lo), p.hi)))
            c["bounds"] = [lo, hi]
    return {"id": tid, "values": values, "params": params, "auto_ref": bool(data.get("auto_ref", old.get("auto_ref", True)))}


def _apply_template_ref(st, b):
    """Winged templates: coefficients on the wing area and mean chord, kept in step with the design."""
    if st.get("template", {}).get("auto_ref", True) and st["template"]["id"] in ("uav", "wing"):
        st["envelope"]["ref_area"] = round(b.ref["ref_area"], 6)
        st["envelope"]["ref_length"] = round(b.ref["ref_length"], 6)
        st["hyper"]["envelope"]["ref_area"] = round(b.ref["ref_area"], 6)
        st["hyper"]["envelope"]["ref_length"] = round(b.ref["ref_length"], 6)


async def samples(request: Request):
    from .geometry.samples import SAMPLES
    return JSONResponse([{"key": k, "title": s["title"], "description": s["description"],
                          "category": s.get("category", "other"), "tunnel": s.get("tunnel", "general"),
                          "envelope": s["envelope"], "hyper": (s.get("hyper") or {}).get("envelope"),
                          "propellers": len(s.get("propellers", []))}
                         for k, s in SAMPLES.items() if not s.get("hidden")])


async def validation(request: Request):
    f = Path(__file__).resolve().parent / "cfd" / "validation_results.json"
    if not f.exists():
        return JSONResponse({"cases": [], "note": "Validation has not been run on this server."})
    return Response(f.read_bytes(), media_type="application/json")


async def health(request: Request):
    with db() as con:
        q = con.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='queued'").fetchone()["n"]
        r = con.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='running'").fetchone()["n"]
    return JSONResponse({"ok": True, "version": VERSION, "billing": settings.billing_mode, "queued": q, "running": r})


# ---- projects -------------------------------------------------------------------------------
def _project_out(p, con=None, user=None):
    out = {k: p[k] for k in ("id", "name", "created_at", "updated_at", "filename", "source_format", "file_size")}
    out["team"] = None
    out["mine"] = user is None or p["user_id"] == user["id"]
    if con is not None and p["team_id"] is not None:
        from . import teams
        t = con.execute("SELECT id, name FROM teams WHERE id=?", (p["team_id"],)).fetchone()
        if t:
            out["team"] = {"id": t["id"], "name": t["name"],
                           "role": "owner" if out["mine"] else teams.role_of(con, t["id"], user["id"]) if user else None}
        if not out["mine"]:
            o = con.execute("SELECT name, email FROM users WHERE id=?", (p["user_id"],)).fetchone()
            out["owner"] = (o["name"] or o["email"]) if o else None
    out["settings"] = _migrate(json.loads(p["settings"] or "{}"))
    out["metrics"] = json.loads(p["metrics"] or "{}")
    out["has_thumbnail"] = (project_dir(p["id"]) / "thumb.png").exists()
    if con is not None:
        j = con.execute("SELECT id, kind, status, summary, finished_at FROM jobs WHERE project_id=? "
                        "ORDER BY created_at DESC LIMIT 1", (p["id"],)).fetchone()
        out["last_job"] = ({"id": j["id"], "kind": j["kind"], "status": j["status"],
                            "summary": json.loads(j["summary"]) if j["summary"] else None,
                            "finished_at": j["finished_at"]} if j else None)
    return out


def _get_project(con, user, pid, need="read"):
    """The project if this user may `need` it: read (viewers up), write (editors up) or owner (its
    creator, or a team owner/admin). Anything else is a 404, so ids of other teams leak nothing."""
    from . import teams
    p = con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if p and p["user_id"] == user["id"]:
        return p
    if p and p["team_id"] is not None:
        role = teams.role_of(con, p["team_id"], user["id"])
        want = {"read": "viewer", "write": "editor", "owner": "admin"}[need]
        if teams.can(role, want):
            return p
        if teams.can(role, "viewer"):
            err(403, f"Your role in this team ({role}) cannot {'change' if need == 'write' else 'delete'} this project.")
    err(404, "Project not found.")


def _inspect_model(data: bytes, filename: str):
    from .geometry.mesh import MeshError, guess_forward, guess_units, load_mesh, metrics
    try:
        m = load_mesh(data, filename).orient_outward()
    except MeshError as e:
        raise ValueError(str(e))
    units = guess_units(m)
    forward = guess_forward(m)
    up = "+z" if forward[1] != "z" else "+y"
    met = metrics(m)
    lo, hi = m.bbox()
    met["bbox_min"] = lo.tolist()
    met["bbox_max"] = hi.tolist()
    return m.n_faces, units, forward, up, met


async def projects(request: Request):
    with db() as con:
        user = current_user(request, con)
        if request.method == "GET":
            rows = con.execute("""SELECT * FROM projects WHERE user_id=? OR team_id IN
                                  (SELECT team_id FROM team_members WHERE user_id=?) ORDER BY updated_at DESC""",
                               (user["id"], user["id"])).fetchall()
            return JSONResponse([_project_out(r, con, user) for r in rows])
        plan = plan_for(user)
        n = con.execute("SELECT COUNT(*) AS n FROM projects WHERE user_id=?", (user["id"],)).fetchone()["n"]
    if plan["max_projects"] is not None and n >= plan["max_projects"]:
        err(402, f"{plan['name']} includes {plan['max_projects']} projects. Delete one or upgrade for unlimited projects.",
            upgrade=True)
    if not upload_limiter.allow(f"up:{user['id']}"):
        err(429, "Upload limit reached for this hour.")
    ctype = request.headers.get("content-type", "")
    source_format = None
    tmpl_built = None
    cad_source = None
    imported_meta, others = None, {}
    if ctype.startswith("application/json"):
        data = await body_json(request)
    if ctype.startswith("application/json") and data.get("template"):
        tid = data.get("template")
        if tid not in T.TEMPLATES:
            err(400, "Unknown template.")
        vals = T.clean(tid, data.get("values") or {})
        tmpl_built, raw = await run_in_threadpool(_template_files, tid, vals)
        filename = f"{tid}.stl"
        name = (data.get("name") or f"{T.TEMPLATES[tid]['title']} (template)")[:120]
        tunnel = data.get("tunnel") if data.get("tunnel") in TUNNELS else "general"
        preset = {"units": "m", "forward": "-x", "up": "+z", "tunnel": tunnel,
                  "envelope": dict(TEMPLATE_ENVELOPES[tid]),
                  "template": {"id": tid, "values": vals, "params": {}, "auto_ref": True},
                  "optimize": {"objective": "drag_fixed_lift" if tid in ("uav", "wing") else ("apogee" if tid == "rocket" else "drag"),
                               "top_k": min(4, plan["max_top_k"]),
                               "iterations": min(60, plan["max_iterations_panel"]), "min_volume": 0.9, "max_volume": 1.3, "keep_stability": True}}
        source_format = "template"
    elif ctype.startswith("application/json"):
        from .geometry.samples import SAMPLES
        key = data.get("sample")
        if key not in SAMPLES:
            err(400, "Unknown sample.")
        cache = settings.data_dir / "samples" / f"{key}.stl"
        if not cache.exists():
            mesh = await run_in_threadpool(SAMPLES[key]["build"])
            cache.write_bytes(mesh.to_stl_bytes(f"AeroSculpt sample {key}".encode()))
        raw = cache.read_bytes()
        filename = f"{key}.stl"
        name = SAMPLES[key]["title"]
        smp = SAMPLES[key]
        preset = {"units": "m", "forward": "-x", "up": "+z", "envelope": dict(smp["envelope"]),
                  "tunnel": data.get("tunnel") if data.get("tunnel") in TUNNELS else smp.get("tunnel", "general"),
                  "propellers": _clean_props(smp.get("propellers", [])), "zones": _clean_zones(smp.get("zones", []))}
        if smp.get("hyper"):
            preset["hyper"] = _clean_hyper(smp["hyper"], {})
        source_format = "sample"
    else:
        filename = Path(request.query_params.get("filename", "model.stl")).name[:120]
        name = (request.query_params.get("name") or filename.rsplit(".", 1)[0])[:120]
        source_format = (request.query_params.get("source") or filename.rsplit(".", 1)[-1]).lower()[:10]
        ext = filename.lower().rsplit(".", 1)[-1]
        from .geometry import cad
        cad_ok = ext in cad.CAD_EXTENSIONS and cad.available()
        design = ext in ("ork", "rkt")
        packed = ext == "aerosculpt"
        if ext not in ("stl", "obj") and not cad_ok and not design and not packed:
            err(400, "Upload STL, OBJ, STEP, IGES, OpenRocket (.ork) or RockSim (.rkt) files (3MF, PLY and glTF are converted in your browser)."
                     + (" STEP/IGES need gmsh on this server." if ext in cad.CAD_EXTENSIONS else ""))
        limit = plan["max_upload_mb"] * 1024 * 1024
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                err(413, f"File is larger than the {plan['max_upload_mb']} MB limit of the {plan['name']} plan.", upgrade=True)
            chunks.append(chunk)
        raw = b"".join(chunks)
        imported_meta = None
        if packed:
            # a project file from this app (any version): its model, source and settings
            imported_meta, raw, model_name, others = _import_project_file(raw)
            filename = model_name
            name = (request.query_params.get("name") or imported_meta.get("name") or "Imported project")[:120]
            source_format = (imported_meta.get("source_format") or "import")[:10]
            src = next((n for n in others if n.startswith("source.")), None)
            if src:
                cad_source = (src.split(".", 1)[1], others[src])
            preset = _clean_settings({k: v for k, v in (imported_meta.get("settings") or {}).items()
                                      if k in ("units", "unit_scale", "forward", "up", "envelope", "zones", "tunnel", "propellers", "hyper",
                                               "features", "ground", "rocket", "optimize", "fidelity")}, {"template": (imported_meta.get("settings") or {}).get("template")})
            if preset.get("template") and preset["template"].get("id") in T.TEMPLATES:
                preset["template"] = {"id": preset["template"]["id"], "values": T.clean(preset["template"]["id"], preset["template"].get("values") or {}),
                                      "params": preset["template"].get("params") or {}, "auto_ref": True}
            else:
                preset.pop("template", None)
            preset["import_notes"] = (imported_meta.get("settings") or {}).get("import_notes") or []
            preset["imported_from"] = {"app": imported_meta.get("app"), "exported_at": imported_meta.get("exported_at")}
        if design:
            # an OpenRocket / RockSim design becomes a rocket template project with its masses and motor
            from .rocket.ork import parse_design
            from .web_rocket import clean_rocket
            try:
                imp = await run_in_threadpool(parse_design, raw, filename)
            except Exception as e:  # noqa: BLE001 - the parser's message is the user-facing one
                err(400, f"Could not read the rocket design: {e}")
            vals = T.clean("rocket", imp["template_values"])
            cad_source = (ext, raw)
            tmpl_built, raw = await run_in_threadpool(_template_files, "rocket", vals)
            name = (request.query_params.get("name") or imp.get("name") or filename.rsplit(".", 1)[0])[:120]
            filename = "rocket.stl"
            source_format = ext
            tunnel = request.query_params.get("tunnel")
            preset = {"units": "m", "forward": "-x", "up": "+z", "tunnel": tunnel if tunnel in TUNNELS else "general",
                      "envelope": dict(TEMPLATE_ENVELOPES["rocket"]),
                      "template": {"id": "rocket", "values": vals, "params": {}, "auto_ref": True},
                      "rocket": clean_rocket(imp.get("rocket") or {}, {}), "import_notes": imp.get("notes") or [],
                      "optimize": {"objective": "apogee", "top_k": min(4, plan["max_top_k"]),
                                   "iterations": min(60, plan["max_iterations_panel"]), "min_volume": 0.9, "max_volume": 1.3}}
        if cad_ok:
            # CAD: tessellate once on the server (OpenCASCADE via gmsh), keep the original beside it
            cad_source = (ext, raw)
            try:
                raw = (await run_in_threadpool(cad.cad_to_mesh, raw, filename)).to_stl_bytes()
            except ValueError as e:
                err(400, f"Could not read the CAD file: {e}")
            filename = filename.rsplit(".", 1)[0] + ".stl"
            source_format = ext
        if not design and not packed:
            preset = None
            tunnel = request.query_params.get("tunnel")
            if tunnel in TUNNELS:
                preset = {"tunnel": tunnel}
    try:
        n_faces, units, forward, up, met = await run_in_threadpool(_inspect_model, raw, filename)
    except ValueError as e:
        err(400, f"Could not read the model: {e}")
    if n_faces > plan["max_triangles"]:
        err(413, f"Model has {n_faces:,} triangles; {plan['name']} supports up to {plan['max_triangles']:,}. "
                 "Decimate it or upgrade.", upgrade=True)
    pid = uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    ext = filename.lower().rsplit(".", 1)[-1]
    (pdir / f"model.{ext}").write_bytes(raw)
    if cad_source is not None:
        (pdir / f"source.{cad_source[0]}").write_bytes(cad_source[1])
    if imported_meta is not None:
        for n, data_ in others.items():
            if n in ("flight_log.json", "thumb.png"):
                (pdir / n).write_bytes(data_)
    st = {"units": units, "forward": forward, "up": up, "envelope": dict(DEFAULT_ENVELOPE), "zones": []}
    if preset:
        st.update(preset)
        st["envelope"] = {**DEFAULT_ENVELOPE, **preset.get("envelope", {})}
    if tmpl_built is not None:
        st = _migrate(st)
        _apply_template_ref(st, tmpl_built)
        met["template_info"] = tmpl_built.info
    now = time.time()
    with db() as con:
        con.execute("""INSERT INTO projects (id, user_id, name, created_at, updated_at, filename, source_format,
                       file_size, settings, metrics) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (pid, user["id"], name, now, now, filename, source_format, len(raw), json.dumps(st), json.dumps(met)))
        audit(con, user["id"], "project_created", pid)
        p = con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return JSONResponse(_project_out(p, con), status_code=201)


async def project_solidify(request: Request):
    """A watertight copy of a visual mesh (open panels, gaps, many pieces) as a new project with the
    same settings; the original project is left as it is (geometry/repair.py)."""
    from .geometry.repair import solidify
    from .optimize.runner import load_model
    with db() as con:
        user = current_user(request, con)
        project = _get_project(con, user, request.path_params["pid"], need="write")
        plan = plan_for(user)
        n = con.execute("SELECT COUNT(*) AS n FROM projects WHERE user_id=?", (user["id"],)).fetchone()["n"]
    if plan["max_projects"] is not None and n >= plan["max_projects"]:
        err(402, f"{plan['name']} includes {plan['max_projects']} projects. Delete one to make a solid copy.", upgrade=True)
    if not upload_limiter.allow(f"up:{user['id']}"):
        err(429, "Too many new models this hour.")
    data = await body_json(request)
    cells = {"standard": 220, "fine": 300}.get(data.get("detail"), 220)
    path = project_dir(project["id"]) / ("model." + project["filename"].rsplit(".", 1)[-1].lower())

    def work():
        mesh, info = solidify(load_model(path, project["filename"]), cells=cells,
                              max_faces=int(min(plan["max_triangles"] * 0.9, 400_000)))
        return mesh.to_stl_bytes(b"AeroSculpt solid"), info

    try:
        stl, info = await run_in_threadpool(work)
    except ValueError as e:
        err(400, f"Could not make a solid: {e}")
    n_faces, _, _, _, met = await run_in_threadpool(_inspect_model, stl, "model.stl")
    if n_faces > plan["max_triangles"]:
        err(413, f"The solid has {n_faces:,} triangles; {plan['name']} supports up to {plan['max_triangles']:,}.", upgrade=True)
    met["solidified"] = {k: info[k] for k in ("cell", "pieces_before", "parts_kept")}
    st = json.loads(project["settings"] or "{}")
    pid = uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "model.stl").write_bytes(stl)
    base = project["filename"].rsplit(".", 1)[0]
    now = time.time()
    with db() as con:
        con.execute("""INSERT INTO projects (id, user_id, name, created_at, updated_at, filename, source_format,
                       file_size, settings, metrics) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (pid, user["id"], f"{project['name']} (solid)"[:120], now, now, f"{base}-solid.stl"[:120], "solidified",
                     len(stl), json.dumps(st), json.dumps(met)))
        audit(con, user["id"], "project_solidified", f"{project['id']} -> {pid}")
        p = con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return JSONResponse(_project_out(p, con), status_code=201)


# automatic optimization: the whole-body controls in order of how much they usually matter
AUTO_ASPECTS = ["tail", "nose", "section", "length", "camber", "fairing"]

DEFAULT_ENVELOPE = {"speed_min": 15.0, "speed_max": 40.0, "speed_count": 4, "aoa_min": 0.0, "aoa_max": 8.0,
                    "aoa_count": 3, "altitude": 0.0, "yaw": 0.0, "ref_area": None, "ref_length": None}
ENVELOPE_LIMITS = {"speed_min": (0.5, 450.0), "speed_max": (0.5, 450.0), "aoa_min": (-30.0, 30.0),
                   "aoa_max": (-30.0, 30.0), "altitude": (-500.0, 20000.0), "yaw": (-30.0, 30.0)}
ZONE_TYPES = ("protect", "keep", "design", "material", "wheel")
TUNNELS = ("general", "rc", "supersonic")
DEFAULT_HYPER = {"mach_min": 1.5, "mach_max": 4.0, "mach_count": 4, "aoa_min": 0.0, "aoa_max": 6.0, "aoa_count": 3,
                 "altitude": 15000.0, "yaw": 0.0, "ref_area": None, "ref_length": None}
HYPER_LIMITS = {"mach_min": (1.2, 12.0), "mach_max": (1.2, 12.0), "aoa_min": (-20.0, 20.0), "aoa_max": (-20.0, 20.0),
                "altitude": (0.0, 80000.0), "yaw": (-15.0, 15.0)}
DEFAULT_THERMAL = {"material": "aluminium", "wall": "radiative", "t_wall_c": 25.0, "transition": "auto",
                   "re_transition": 1.0e6, "nose_radius_mm": None}
PROP_LIMITS = {"diameter_in": (2.0, 40.0), "pitch_in": (1.0, 30.0), "rpm": (500.0, 40000.0), "width": (0.5, 2.0)}


def _num(v, name):
    try:
        x = float(v)
    except (TypeError, ValueError):
        err(400, f"Invalid {name}.")
    if not math.isfinite(x):
        err(400, f"Invalid {name}.")
    return x


def _migrate(st):
    """Projects created before flight envelopes stored one speed and angle under 'flow'."""
    if "envelope" not in st:
        fl = st.pop("flow", {}) or {}
        sp = float(fl.get("speed", 20.0))
        st["envelope"] = {**DEFAULT_ENVELOPE, "speed_min": round(max(0.5, sp * 0.5), 1), "speed_max": sp,
                          "aoa_min": float(fl.get("aoa", 0.0)), "aoa_max": float(fl.get("aoa", 0.0)) + 6.0,
                          "altitude": float(fl.get("altitude", 0.0)), "yaw": float(fl.get("yaw", 0.0)),
                          "ref_area": fl.get("ref_area"), "ref_length": fl.get("ref_length")}
    st.setdefault("zones", [])
    st.setdefault("tunnel", "general")
    st.setdefault("propellers", [])
    hy = st.setdefault("hyper", {})
    hy["envelope"] = {**DEFAULT_HYPER, **(hy.get("envelope") or {})}
    hy["thermal"] = {**DEFAULT_THERMAL, **(hy.get("thermal") or {})}
    return st


def _clean_zones(items):
    if not isinstance(items, list) or len(items) > 50:
        err(400, "Invalid protected regions.")
    out = []
    for z in items:
        if not isinstance(z, dict) or z.get("type") not in ZONE_TYPES or z.get("shape", "box") not in ("box", "cylinder"):
            err(400, "Invalid protected region.")
        center = [_num(v, "region position") for v in (z.get("center") or [])]
        size = [_num(v, "region size") for v in (z.get("size") or [])]
        if len(center) != 3 or len(size) != 3 or min(size) <= 0:
            err(400, "A protected region needs a position and a positive size on all three axes.")
        item = {"id": str(z.get("id") or uuid.uuid4().hex[:8])[:16], "type": z["type"],
                "shape": z.get("shape", "box"), "center": center, "size": size,
                "name": str(z.get("name") or "")[:60]}
        if z.get("rotation") is not None:
            rot = [_num(v, "region rotation") for v in (z.get("rotation") or [])]
            if len(rot) != 3 or max(abs(a) for a in rot) > 360:
                err(400, "A region rotation is three angles in degrees (roll, pitch, yaw).")
            if any(abs(a) > 1e-9 for a in rot):
                item["rotation"] = [round(((a + 180.0) % 360.0) - 180.0, 4) for a in rot]
        if z["type"] == "material":
            item["material"] = _clean_material(z.get("material"))
        out.append(item)
    return out


def _clean_features(items):
    from .geometry import features as FT
    if not isinstance(items, list) or len(items) > FT.MAX_FEATURES:
        err(400, f"Up to {FT.MAX_FEATURES} shape features.")
    out = []
    for f in items:
        try:
            out.append(FT.clean(f))
        except ValueError as e:
            err(400, f"Shape feature: {e}.")
    if len({f["id"] for f in out}) != len(out):
        err(400, "Two shape features share an id.")
    return out


def _free_features(features):
    """Whether any enabled feature has a dimension left free for the optimizer."""
    from .geometry import features as FT
    if any(f.get("optional") and f.get("enabled", True) for f in features):
        return True
    for f in features:
        for p in FT.get(f["type"])["params"]:
            if not FT.locked(p, f) and T.active(p, f["values"]):
                return True
    return False


def _job_features(st, tm, kind):
    feats = [f for f in st.get("features") or [] if f.get("enabled", True)]
    if feats and tm and tm["id"] in ("uav", "wing"):
        err(400, "Shape features need a CFD tunnel: the UAV and wing templates use the wing model. "
                 "Switch them off, or start from an uploaded model.")
    return feats


def _clean_material(m):
    from .hyper.materials import MATERIALS, resolve
    if isinstance(m, str) and m in MATERIALS:
        return m
    if isinstance(m, dict):
        try:
            r = resolve(m)
        except ValueError as e:
            err(400, str(e))
        return {"name": r["name"], "t_max_c": r["t_max_c"], "emissivity": r["emissivity"]}
    err(400, "Unknown material.")


def _clean_props(items):
    if not isinstance(items, list) or len(items) > 16:
        err(400, "Up to 16 propellers.")
    out = []
    for p in items:
        if not isinstance(p, dict):
            err(400, "Invalid propeller.")
        center = [_num(v, "propeller position") for v in (p.get("center") or [])]
        if len(center) != 3:
            err(400, "A propeller needs a position.")
        from .rc.propeller import FAMILIES
        item = {"id": str(p.get("id") or uuid.uuid4().hex[:8])[:16], "name": str(p.get("name") or "Propeller")[:40],
                "center": center, "blades": int(_num(p.get("blades", 2), "blade count")),
                "rotation": "ccw" if p.get("rotation") == "ccw" else "cw",
                "family": p.get("family") if p.get("family") in FAMILIES else "thin-electric"}
        if not (2 <= item["blades"] <= 6):
            err(400, "Propellers have 2 to 6 blades.")
        for k, (lo, hi) in PROP_LIMITS.items():
            v = _num(p.get(k, {"width": 1.0}.get(k)), k.replace("_in", "").replace("_", " "))
            if not (lo <= v <= hi):
                err(400, f"Propeller {k.replace('_in', ' (in)').replace('_', ' ')} must be between {lo:g} and {hi:g}.")
            item[k] = v
        out.append(item)
    return out


def _clean_hyper(data, old):
    hy = {"envelope": {**DEFAULT_HYPER, **(old.get("envelope") or {})}, "thermal": {**DEFAULT_THERMAL, **(old.get("thermal") or {})}}
    env_in = data.get("envelope") if isinstance(data.get("envelope"), dict) else {}
    env = hy["envelope"]
    for k, (lo, hi) in HYPER_LIMITS.items():
        if k in env_in:
            v = _num(env_in[k], k.replace("_", " "))
            if not (lo <= v <= hi):
                err(400, f"{k.replace('_', ' ').capitalize()} must be between {lo:g} and {hi:g}.")
            env[k] = v
    for k in ("mach_count", "aoa_count"):
        if k in env_in:
            v = int(_num(env_in[k], k))
            if not (1 <= v <= 25):
                err(400, "Use between 1 and 25 points.")
            env[k] = v
    for k in ("ref_area", "ref_length"):
        if k in env_in:
            v = env_in[k]
            env[k] = None if v in (None, "", 0) else _num(v, k)
            if env[k] is not None and not (0 < env[k] < 1e4):
                err(400, f"Invalid {k}.")
    if env["mach_min"] > env["mach_max"]:
        env["mach_min"], env["mach_max"] = env["mach_max"], env["mach_min"]
    if env["aoa_min"] > env["aoa_max"]:
        env["aoa_min"], env["aoa_max"] = env["aoa_max"], env["aoa_min"]
    th_in = data.get("thermal") if isinstance(data.get("thermal"), dict) else {}
    th = hy["thermal"]
    if "material" in th_in:
        th["material"] = _clean_material(th_in["material"])
    if "wall" in th_in:
        th["wall"] = "fixed" if th_in["wall"] == "fixed" else "radiative"
    if "transition" in th_in:
        if th_in["transition"] not in ("auto", "laminar", "turbulent"):
            err(400, "Transition must be auto, laminar or turbulent.")
        th["transition"] = th_in["transition"]
    if "t_wall_c" in th_in:
        th["t_wall_c"] = _num(th_in["t_wall_c"], "wall temperature")
        if not (-100 <= th["t_wall_c"] <= 3000):
            err(400, "Wall temperature must be between -100 and 3000 °C.")
    if "re_transition" in th_in:
        th["re_transition"] = _num(th_in["re_transition"], "transition Reynolds number")
        if not (1e4 <= th["re_transition"] <= 1e9):
            err(400, "Transition Reynolds number must be between 1e4 and 1e9.")
    if "nose_radius_mm" in th_in:
        v = th_in["nose_radius_mm"]
        th["nose_radius_mm"] = None if v in (None, "", 0) else _num(v, "nose radius")
        if th["nose_radius_mm"] is not None and not (0.01 <= th["nose_radius_mm"] <= 5000):
            err(400, "Nose radius must be between 0.01 and 5000 mm.")
    return hy


def _clean_settings(data, old):
    st = _migrate(dict(old))
    if "units" in data:
        if data["units"] not in ("mm", "cm", "m", "in", "ft", "custom"):
            err(400, "Unknown unit.")
        st["units"] = data["units"]
    if "size_ok" in data:
        st["size_ok"] = bool(data["size_ok"])           # the user confirmed the model's real size
    if "unit_scale" in data:
        v = _num(data["unit_scale"], "model scale")
        if not (1e-7 < v < 1e4):
            err(400, "The model scale is out of range.")
        st["unit_scale"] = v
    if st.get("units") == "custom" and not st.get("unit_scale"):
        err(400, "Give the model's real length to size it.")
    for k in ("forward", "up"):
        if k in data:
            if data[k] not in AXES:
                err(400, f"Invalid {k} axis.")
            st[k] = data[k]
    if st.get("forward", "-x")[1] == st.get("up", "+z")[1]:
        err(400, "Forward and up must be different axes.")
    if isinstance(data.get("ground"), dict):
        g = data["ground"]
        clearance = _num(g.get("clearance", 0.0) or 0.0, "ground clearance")
        if not (0.0 <= clearance <= 5.0):
            err(400, "Ground clearance must be between 0 and 5 m.")
        st["ground"] = {"enabled": bool(g.get("enabled")), "clearance": clearance}
    if isinstance(data.get("envelope"), dict):
        env = dict(st["envelope"])
        for k, (lo, hi) in ENVELOPE_LIMITS.items():
            if k in data["envelope"]:
                v = _num(data["envelope"][k], k.replace("_", " "))
                if not (lo <= v <= hi):
                    err(400, f"{k.replace('_', ' ').capitalize()} must be between {lo:g} and {hi:g}.")
                env[k] = v
        for k in ("speed_count", "aoa_count"):
            if k in data["envelope"]:
                v = int(_num(data["envelope"][k], k))
                if not (1 <= v <= 25):
                    err(400, "Use between 1 and 25 points.")
                env[k] = v
        for k in ("ref_area", "ref_length"):
            if k in data["envelope"]:
                v = data["envelope"][k]
                env[k] = None if v in (None, "", 0) else _num(v, k)
                if env[k] is not None and not (0 < env[k] < 1e4):
                    err(400, f"Invalid {k}.")
        if env["speed_min"] > env["speed_max"]:
            env["speed_min"], env["speed_max"] = env["speed_max"], env["speed_min"]
        if env["aoa_min"] > env["aoa_max"]:
            env["aoa_min"], env["aoa_max"] = env["aoa_max"], env["aoa_min"]
        st["envelope"] = env
    if "zones" in data:
        st["zones"] = _clean_zones(data["zones"])
    if "features" in data:
        st["features"] = _clean_features(data["features"])
    if "tunnel" in data:
        if data["tunnel"] not in TUNNELS:
            err(400, "Unknown tunnel.")
        st["tunnel"] = data["tunnel"]
    if "propellers" in data:
        st["propellers"] = _clean_props(data["propellers"])
    if isinstance(data.get("hyper"), dict):
        st["hyper"] = _clean_hyper(data["hyper"], st.get("hyper") or {})
    if isinstance(data.get("rocket"), dict):
        from .web_rocket import clean_rocket
        st["rocket"] = clean_rocket(data["rocket"], st.get("rocket") or {})
    if "optimize" in data and isinstance(data["optimize"], dict):
        st["optimize"] = data["optimize"]
    if isinstance(data.get("template"), dict):
        if not st.get("template"):
            err(400, "This project was not made from a template.")
        st["template"] = _clean_template(data["template"], st["template"])
    if data.get("fidelity") in ("draft", "standard", "high", "ultra"):
        st["fidelity"] = "high" if data["fidelity"] == "ultra" else data["fidelity"]
    return st


def _range(lo, hi, n, digits=1):
    n = int(n)
    if n <= 1 or hi - lo < 1e-9:
        return [round(lo, digits)]
    return sorted({round(lo + (hi - lo) * i / (n - 1), digits) for i in range(n)})


def envelope_points(env):
    """Range settings -> the speeds (m/s) and angles (deg) actually simulated."""
    return _range(env["speed_min"], env["speed_max"], env["speed_count"]), _range(env["aoa_min"], env["aoa_max"], env["aoa_count"])


def hyper_points(env):
    """Supersonic range settings -> Mach numbers and angles."""
    return _range(env["mach_min"], env["mach_max"], env["mach_count"], 2), _range(env["aoa_min"], env["aoa_max"], env["aoa_count"])


def _body_length(project, st):
    """Body length along the chosen flow axis, in metres, from the upload's bounding box."""
    from .geometry.mesh import UNIT_SCALE
    met = json.loads(project["metrics"] or "{}")
    lo, hi = met.get("bbox_min"), met.get("bbox_max")
    if not lo or not hi:
        return None
    axis = "xyz".index(st.get("forward", "-x")[1])
    return (hi[axis] - lo[axis]) * _unit_scale(st)


def _unit_scale(st):
    from .geometry.mesh import UNIT_SCALE
    if st.get("units") == "custom":
        return float(st.get("unit_scale") or 1.0)
    return UNIT_SCALE.get(st.get("units", "m"), 1.0)


async def project_item(request: Request):
    pid = request.path_params["pid"]
    with db() as con:
        user = current_user(request, con)
        p = _get_project(con, user, pid, {"GET": "read", "DELETE": "owner"}.get(request.method, "write"))
        if request.method == "GET":
            return JSONResponse(_project_out(p, con, user))
        if request.method == "DELETE":
            jids = [r["id"] for r in con.execute("SELECT id FROM jobs WHERE project_id=?", (pid,))]
            con.execute("UPDATE jobs SET cancel_requested=1 WHERE project_id=? AND status IN ('queued','running')", (pid,))
            con.execute("DELETE FROM projects WHERE id=?", (pid,))
            audit(con, user["id"], "project_deleted", pid)
            shutil.rmtree(project_dir(pid), ignore_errors=True)
            for j in jids:
                shutil.rmtree(job_dir(j), ignore_errors=True)
            return JSONResponse({"ok": True})
        data = await body_json(request)
        name = (data.get("name") or p["name"]).strip()[:120] or p["name"]
        old = json.loads(p["settings"] or "{}")
        st = _clean_settings(data.get("settings", {}), old)
        tm = st.get("template")
        if tm and (tm.get("values") != (old.get("template") or {}).get("values") or not (project_dir(pid) / p["filename"]).exists()):
            # the dimensions changed: rebuild the model file the tunnels read
            b, raw = _template_files(tm["id"], tm["values"])
            (project_dir(pid) / p["filename"]).write_bytes(raw)
            _, _u, _f, _up, met = _inspect_model(raw, p["filename"])
            met["template_info"] = b.info
            _apply_template_ref(st, b)
            con.execute("UPDATE projects SET metrics=?, file_size=? WHERE id=?", (json.dumps(met), len(raw), pid))
        con.execute("UPDATE projects SET name=?, settings=?, updated_at=? WHERE id=?",
                    (name, json.dumps(st), time.time(), pid))
        return JSONResponse(_project_out(con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone(), con))


async def project_model(request: Request):
    pid = request.path_params["pid"]
    with db() as con:
        user = current_user(request, con)
        p = _get_project(con, user, pid)
    ext = p["filename"].lower().rsplit(".", 1)[-1]
    f = project_dir(pid) / f"model.{ext}"
    if not f.exists():
        err(404, "Model file missing.")
    return FileResponse(f, media_type="application/octet-stream", filename=p["filename"])


async def project_thumb(request: Request):
    pid = request.path_params["pid"]
    with db() as con:
        user = current_user(request, con)
        _get_project(con, user, pid)
    f = project_dir(pid) / "thumb.png"
    if request.method == "GET":
        if not f.exists():
            err(404, "No thumbnail.")
        return FileResponse(f, media_type="image/png", headers={"Cache-Control": "private, max-age=60"})
    data = await body_json(request, limit=600_000)
    url = data.get("png") or ""
    if not url.startswith("data:image/png;base64,"):
        err(400, "Expected a PNG data URL.")
    png = base64.b64decode(url.split(",", 1)[1])
    if not png.startswith(b"\x89PNG"):
        err(400, "Not a PNG.")
    f.write_bytes(png)
    return JSONResponse({"ok": True})


# ---- jobs ------------------------------------------------------------------------------------
def _build_spec(data, project, plan):
    from .cfd.analysis import FIDELITY, fidelity_name
    st = json.loads(project["settings"] or "{}")
    kind = data.get("kind")
    if kind not in ("analyze", "optimize"):
        err(400, "kind must be analyze or optimize.")
    try:
        fid = fidelity_name(data.get("fidelity", "draft"))
    except KeyError:
        err(400, "Unknown fidelity.")
    st = _clean_settings({k: data[k] for k in ("units", "unit_scale", "forward", "up", "envelope", "zones", "tunnel", "propellers",
                                                "hyper", "features", "ground", "rocket") if k in data}, st)
    st = _migrate(st)
    tunnel = st.get("tunnel", "general")
    if tunnel == "supersonic":
        return _build_hyper_spec(data, kind, st, plan)
    env = st["envelope"]
    speeds, aoas = envelope_points(env)
    spec = {"kind": kind, "fidelity": fid, "tunnel": tunnel,
            "model": {"units": st["units"], "forward": st["forward"], "up": st["up"], "scale": st.get("unit_scale")},
            "envelope": {"speeds": speeds, "aoas": aoas, "altitude": env["altitude"], "yaw": env["yaw"],
                         "ref_area": env["ref_area"], "ref_length": env["ref_length"]},
            "zones": [z for z in st.get("zones", []) if z["type"] != "material"]}
    if kind == "analyze" and data.get("grid_study"):
        spec["grid_study"] = True
    if kind == "analyze" and data.get("refine") and tunnel != "supersonic":
        spec["refine"] = True
    if (st.get("ground") or {}).get("enabled") and not (st.get("template") or {}).get("id") in ("uav", "wing"):
        spec["ground"] = dict(st["ground"])
        spec["envelope"]["aoas"], spec["envelope"]["yaw"] = [0.0], 0.0      # level on the road
    if tunnel == "rc" and st.get("propellers"):
        spec["propellers"] = st["propellers"]
        if min(speeds) < 2.0:
            err(400, "With propellers turning, the lowest airspeed must be at least 2 m/s (static thrust is not simulated).")
    tm = st.get("template")
    if tm:
        spec["template"] = {"id": tm["id"], "values": tm["values"]}
    spec["features"] = _job_features(st, tm, kind)
    rocket_goal = False
    if tm and tm["id"] == "rocket":
        # the flight model's inputs travel with the job (the worker has no user context): motor
        # curve included, so a rocket reshaped for drag is still checked for stability and flutter
        from .web_rocket import clean_rocket, motor_from_settings
        from .rocket.motors import motor_for_diameter
        rk = clean_rocket({}, st.get("rocket") or {})
        motor = motor_from_settings(project["user_id"], rk) or motor_for_diameter(float(tm["values"].get("diameter", 0.1)))
        rk["motor_data"] = motor.public()
        rk["envelope_speeds"] = speeds
        spec["rocket"] = rk
    if kind == "optimize":
        o = data.get("optimize") or {}
        auto = not tm and o.get("mode", "auto") == "auto"
        aspects = [] if tm else [a for a in o.get("aspects", []) if a in ALL_ASPECTS]
        if auto:          # everything the plan allows, most effective first, plus generated ideas
            aspects = [a for a in AUTO_ASPECTS if a in plan["aspects"]][: plan["max_aspects"]]
        if not tm and not aspects and not _free_features(spec["features"]):
            err(400, "Choose at least one design aspect to improve, or unlock a shape-feature dimension.")
        objective = o.get("objective", "drag")
        if objective in ROCKET_OBJECTIVES and not (tm and tm["id"] == "rocket"):
            err(400, "Apogee and rocket-drag goals need a rocket template project (or an imported OpenRocket / RockSim design).")
        rocket_goal = objective in ROCKET_OBJECTIVES
        if tm and tm["id"] in ("uav", "wing"):
            objective = "drag_fixed_lift"
        elif objective == "drag_fixed_lift":
            objective = "drag"
        if objective not in ALL_OBJECTIVES:
            err(400, "Unknown objective.")
        # lift-based goals use the CFD's lift, which reads low on thin wings (see /validation): they
        # compare designs fairly but their absolute lift is indicative; the results say so
        try:
            iterations = int(o.get("iterations", 6))
            max_change = float(o.get("max_change", 0.25))
            min_volume = float(o.get("min_volume", 0.95))
            max_volume = float(o.get("max_volume", 1.5))
        except (TypeError, ValueError):
            err(400, "Invalid optimization settings.")
        if not (1 <= iterations <= 500):
            err(400, "Iterations must be between 1 and 500.")
        if not (0.05 <= max_change <= 0.5):
            err(400, "Maximum change must be between 5% and 50%.")
        if not (0.5 <= min_volume <= 1.0) or not (1.0 <= max_volume <= 3.0):
            err(400, "Volume limits are out of range.")
        verify = o.get("verify") or None
        if verify is not None:
            try:
                verify = fidelity_name(verify)
            except KeyError:
                err(400, "Unknown verification fidelity.")
        spec["optimize"] = {"objective": objective, "aspects": aspects, "iterations": iterations,
                            "max_change": max_change, "min_volume": min_volume, "max_volume": max_volume,
                            "verify": verify, "seed": int(o.get("seed", 7)) % 100000,
                            "mode": "auto" if auto else "custom", "auto_ideas": auto,
                            **_search_options(o, tm, plan, _free_features(spec["features"]))}
    try:
        check_spec(spec, plan)
    except PermissionError as e:
        err(403, str(e), upgrade=True)
    except ValueError as e:
        err(400, str(e))
    # tunnel Reynolds groups: every speed above the grid's cap shares one run per angle
    from .cfd.analysis import isa, reynolds_groups
    L = _body_length(project, st) or 1.0
    _, groups = reynolds_groups(speeds, L, isa(env["altitude"])["nu"], FIDELITY[fid].re_cap)
    mach = max(speeds) / isa(env["altitude"])["a"]
    if mach > 1.2 and not rocket_goal:
        err(400, f"{max(speeds):g} m/s is Mach {mach:.2f}; this tunnel stops at Mach 1.2. "
                 "Use the Supersonic tunnel above that (the rocket flight model has no speed limit).")
    return spec, st, len(groups)


def _search_options(o, tm, plan, free_features=False):
    """Settings of the screening / search / confirmation strategy and the design constraints."""
    out = {}
    try:
        out["top_k"] = int(o.get("top_k", 4))
        out["confirm"] = 2
        for k, lo, hi in (("max_length", 0.001, 1000.0), ("max_width", 0.001, 1000.0), ("max_height", 0.001, 1000.0),
                          ("min_thickness", 0.00001, 1.0), ("mass_kg", 0.001, 1.0e6)):
            v = o.get(k)
            if v not in (None, "", 0):
                v = float(v)
                if not (lo <= v <= hi) or not math.isfinite(v):
                    err(400, f"{k.replace('_', ' ').capitalize()} is out of range.")
                out[k] = v
    except (TypeError, ValueError):
        err(400, "Invalid optimization settings.")
    if not (1 <= out["top_k"] <= 8):
        err(400, "Optimize between 1 and 8 parameters.")
    out["hold_wing_area"] = bool(o.get("hold_wing_area", False))
    out["keep_stability"] = bool(o.get("keep_stability", True))
    if isinstance(o.get("structure"), dict):
        from .aero.structure import StructureSettings
        import dataclasses as _dc
        out["structure"] = _dc.asdict(StructureSettings.from_spec(o["structure"]))
    if tm:
        out["params"] = tm.get("params") or {}
        free = [p for p in T.get(tm["id"])["params"]
                if not out["params"].get(p.key, {}).get("locked", p.locked) and T.active(p, tm["values"])]
        if not free and not free_features:
            err(400, "Every parameter is locked. Unlock at least one for the optimizer to change.")
    return out


def _build_hyper_spec(data, kind, st, plan):
    from .hyper.runner import OBJECTIVES as HYPER_OBJECTIVES
    hy = st["hyper"]
    env = hy["envelope"]
    machs, aoas = hyper_points(env)
    spec = {"kind": kind, "tunnel": "supersonic", "fidelity": "panel",
            "model": {"units": st["units"], "forward": st["forward"], "up": st["up"], "scale": st.get("unit_scale")},
            "envelope": {"machs": machs, "aoas": aoas, "altitude": env["altitude"], "yaw": env["yaw"],
                         "ref_area": env["ref_area"], "ref_length": env["ref_length"]},
            "thermal": dict(hy["thermal"]), "zones": st.get("zones", [])}
    tm = st.get("template")
    if tm:
        spec["template"] = {"id": tm["id"], "values": tm["values"]}
    spec["features"] = _job_features(st, tm, kind)
    if kind == "optimize":
        o = data.get("optimize") or {}
        auto = not tm and o.get("mode", "auto") == "auto"
        aspects = [] if tm else [a for a in o.get("aspects", []) if a in ALL_ASPECTS]
        if auto:
            aspects = [a for a in AUTO_ASPECTS if a in plan["aspects"]][: plan["max_aspects"]]
        if not tm and not aspects and not _free_features(spec["features"]):
            err(400, "Choose at least one design aspect to improve, or unlock a shape-feature dimension.")
        objective = o.get("objective", "drag")
        if objective not in HYPER_OBJECTIVES:
            err(400, "Unknown objective for the supersonic tunnel.")
        try:
            iterations = int(o.get("iterations", 30))
            max_change = float(o.get("max_change", 0.25))
            min_volume = float(o.get("min_volume", 0.95))
            max_volume = float(o.get("max_volume", 1.5))
        except (TypeError, ValueError):
            err(400, "Invalid optimization settings.")
        if not (1 <= iterations <= 500):
            err(400, "Iterations must be between 1 and 500.")
        if not (0.05 <= max_change <= 0.5):
            err(400, "Maximum change must be between 5% and 50%.")
        if not (0.5 <= min_volume <= 1.0) or not (1.0 <= max_volume <= 3.0):
            err(400, "Volume limits are out of range.")
        spec["optimize"] = {"objective": objective, "aspects": aspects, "iterations": iterations,
                            "max_change": max_change, "min_volume": min_volume, "max_volume": max_volume,
                            "respect_limits": bool(o.get("respect_limits", True)), "seed": int(o.get("seed", 7)) % 100000,
                            "mode": "auto" if auto else "custom", "auto_ideas": auto,
                            **_search_options(o, tm, plan, _free_features(spec["features"]))}
    try:
        check_spec(spec, plan)
    except PermissionError as e:
        err(403, str(e), upgrade=True)
    except ValueError as e:
        err(400, str(e))
    return spec, st, 1


def _job_out(con, j, full=True):
    out = {k: j[k] for k in ("id", "project_id", "kind", "status", "progress", "stage", "created_at", "started_at",
                             "finished_at", "error", "estimated_credits", "credits_used", "cfd_runs")}
    out["summary"] = json.loads(j["summary"]) if j["summary"] else None
    out["spec"] = json.loads(j["spec"])
    if full:
        out["live"] = json.loads(j["live"] or "{}")
        out["log"] = (j["log"] or "")[-4000:]
        if j["status"] == "queued":
            out["queue_position"] = con.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status='queued' AND (priority > ? OR (priority = ? AND created_at < ?))",
                (j["priority"], j["priority"], j["created_at"])).fetchone()["n"] + 1
    return out


async def project_jobs(request: Request):
    pid = request.path_params["pid"]
    with db() as con:
        user = current_user(request, con)
        p = _get_project(con, user, pid, "read" if request.method == "GET" else "write")
        if request.method == "GET":
            rows = con.execute("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 50", (pid,)).fetchall()
            return JSONResponse([_job_out(con, r, full=False) for r in rows])
    data = await body_json(request)
    with transaction() as con:
        user = con.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        plan = plan_for(user)
        spec, st, n_groups = _build_spec(data, p, plan)
        active = con.execute("SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND status IN ('queued','running')",
                             (user["id"],)).fetchone()["n"]
        if active >= plan["concurrent_jobs"]:
            err(429, f"{plan['name']} runs {plan['concurrent_jobs']} job(s) at a time. Wait for the current one or cancel it.",
                upgrade=plan["key"] != "team")
        est = estimate_credits(spec, n_groups)
        used = usage_for(con, user["id"])["credits"]
        reserved = con.execute("SELECT COALESCE(SUM(estimated_credits),0) AS n FROM jobs WHERE user_id=? AND status IN ('queued','running')",
                               (user["id"],)).fetchone()["n"]
        left = plan["monthly_credits"] - used - reserved
        if est > left:
            err(402, f"This job needs about {est} credits and you have {max(left, 0)} left this month.", upgrade=True,
                credits_needed=est, credits_left=max(left, 0))
        con.execute("UPDATE projects SET settings=? WHERE id=?", (json.dumps(st), pid))
        jid = enqueue(con, user, p, spec, plan, est)
        if p["team_id"] is not None:
            audit(con, user["id"], f"job_{spec.get('kind')}", f"{p['name']} ({jid})", team_id=p["team_id"])
        j = con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return JSONResponse(_job_out(con, j), status_code=201)


def _get_job(con, user, jid, need="read"):
    j = con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if j and j["user_id"] == user["id"]:
        return j
    if j:
        p = con.execute("SELECT team_id FROM projects WHERE id=?", (j["project_id"],)).fetchone()
        if p and p["team_id"] is not None:
            from . import teams
            if teams.can(teams.role_of(con, p["team_id"], user["id"]), "viewer" if need == "read" else "editor"):
                return j
    err(404, "Job not found.")


async def job_item(request: Request):
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"])
        return JSONResponse(_job_out(con, j))


async def job_cancel(request: Request):
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"], "write")
        if j["status"] == "queued":
            con.execute("UPDATE jobs SET status='canceled', stage='Cancelled', finished_at=?, cancel_requested=1 WHERE id=?",
                        (time.time(), j["id"]))
        elif j["status"] == "running":
            con.execute("UPDATE jobs SET cancel_requested=1, stage='Cancelling' WHERE id=?", (j["id"],))
        return JSONResponse({"ok": True})


async def job_promote(request: Request):
    """Start a new project from an optimization's best design (in the original file frame), or,
    for a template, from any design it evaluated ({"index": history entry}, the trade-off chart)."""
    try:
        data = await body_json(request)
    except Exception:  # noqa: BLE001 - an empty body means: the best design
        data = {}
    pick = data.get("index") if isinstance(data, dict) else None
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"])
        plan = plan_for(user)
        n = con.execute("SELECT COUNT(*) AS n FROM projects WHERE user_id=?", (user["id"],)).fetchone()["n"]
        src = con.execute("SELECT * FROM projects WHERE id=?", (j["project_id"],)).fetchone()
    if j["kind"] != "optimize" or j["status"] != "done":
        err(400, "Only finished optimizations can be continued.")
    if plan["max_projects"] is not None and n >= plan["max_projects"]:
        err(402, f"{plan['name']} includes {plan['max_projects']} projects. Delete one or upgrade.", upgrade=True)
    f = job_dir(j["id"]) / "optimized.stl"
    if not f.exists() or src is None:
        err(404, "The optimized geometry is no longer available.")
    st = json.loads(src["settings"] or "{}")
    res_f = job_dir(j["id"]) / "result.json"
    tmpl_built = None
    if st.get("template") and res_f.exists():
        res = json.loads(res_f.read_text())
        if res.get("template"):
            values = res["template"]["best_values"]
            if pick is not None:
                hist = res.get("history") or []
                if not isinstance(pick, int) or not (0 <= pick < len(hist)) or not hist[pick].get("values"):
                    err(400, "No such design in this optimization.")
                values = {**st["template"]["values"], **hist[pick]["values"]}
            st["template"] = {**st["template"], "values": T.clean(st["template"]["id"], values)}
            tmpl_built, raw = await run_in_threadpool(_template_files, st["template"]["id"], st["template"]["values"])
            st = _migrate(st)
            _apply_template_ref(st, tmpl_built)
    if tmpl_built is None:
        raw = f.read_bytes()
    try:
        n_faces, units, forward, up, met = await run_in_threadpool(_inspect_model, raw, "optimized.stl")
    except ValueError as e:
        err(400, str(e))
    if tmpl_built is not None:
        met["template_info"] = tmpl_built.info
    pid = uuid.uuid4().hex[:12]
    project_dir(pid).mkdir(parents=True, exist_ok=True)
    fname = src["filename"] if tmpl_built is not None else "optimized.stl"
    (project_dir(pid) / fname).write_bytes(raw)
    now = time.time()
    base = src["name"].removesuffix(" (optimized)")
    with db() as con:
        con.execute("""INSERT INTO projects (id, user_id, name, created_at, updated_at, filename, source_format,
                       file_size, settings, metrics) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (pid, user["id"], f"{base} (optimized)"[:120], now, now, fname,
                     "template" if tmpl_built is not None else "optimized", len(raw), json.dumps(st), json.dumps(met)))
        audit(con, user["id"], "project_promoted", f"{j['id']} -> {pid}")
        return JSONResponse(_project_out(con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone(), con),
                            status_code=201)


async def job_file(request: Request):
    name = request.path_params["name"]
    if name not in JOB_FILES and not JOB_FIELD_FILE.match(name):
        err(404, "No such file.")
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"])
        plan = plan_for(user)
        project = con.execute("SELECT name FROM projects WHERE id=?", (j["project_id"],)).fetchone()
    if name in ("optimized.stl", "promising.stl") and not plan["export"]:
        err(403, "Exporting optimized geometry is available on Pro and Team.", upgrade=True)
    f = job_dir(j["id"]) / name
    if not f.exists():
        err(404, "Result not available yet.")
    if name.endswith(".json"):
        return FileResponse(f, media_type="application/json", headers={"Cache-Control": "private, max-age=3600"})
    dl = None
    if name in ("optimized.stl", "promising.stl"):
        base = "".join(c for c in (project["name"] if project else "design") if c.isalnum() or c in "-_ ").strip() or "design"
        dl = f"{base} - {'optimized' if name == 'optimized.stl' else 'most promising (unconfirmed)'}.stl"
    return FileResponse(f, media_type="model/stl", filename=dl)


def _safe_name(name, fallback="design"):
    return "".join(c for c in (name or fallback) if c.isalnum() or c in "-_ ").strip() or fallback


def _metres(spec_model):
    from .geometry.mesh import UNIT_SCALE
    units = (spec_model or {}).get("units", "m")
    if units == "custom":
        return float((spec_model or {}).get("scale") or 1.0)
    return UNIT_SCALE.get(units, 1.0)


def _step_for_job(j, out):
    """The optimized design as STEP: a template as smooth solids through its own sections at the
    optimized values; any other design as a faceted solid of its optimized mesh (in metres)."""
    from .geometry import cad
    from .geometry.mesh import load_mesh
    spec = json.loads(j["spec"])
    result = json.loads((job_dir(j["id"]) / "result.json").read_text())
    tm = spec.get("template")
    if tm and result.get("kind") == "optimize":
        best = (result.get("template") or {}).get("best_values") or result.get("best_values") or {}
        cad.template_to_step(tm["id"], {**tm["values"], **best}, out)
        return out
    if tm:
        cad.template_to_step(tm["id"], tm["values"], out)
        return out
    src = job_dir(j["id"]) / ("optimized.stl" if result.get("kind") == "optimize" else "baseline_body.stl")
    if not src.exists():
        raise ValueError("this job has no design to export")
    mesh = load_mesh(src.read_bytes(), src.name)
    if src.name == "optimized.stl":
        mesh = mesh.transformed(s=_metres(spec.get("model")))
    cad.mesh_to_step(mesh, out)
    return out


def _template_params(spec, result):
    """(name, value, unit, comment) of every numeric template dimension at the job's design."""
    tm = spec.get("template")
    if not tm:
        return []
    vals = {**tm["values"], **((result.get("template") or {}).get("best_values") or {})}
    out = []
    for p in T.get(tm["id"])["params"]:
        v = vals.get(p.key)
        if p.unit == "choice" or not isinstance(v, (int, float)):
            continue
        out.append((p.key, float(v), p.unit, p.label))
    return out


async def fusion_status(request: Request):
    from . import fusion
    with db() as con:
        user = current_user(request, con)
    from .local import local_settings
    on = settings.fusion_bridge or (settings.open_access and local_settings().get("fusion_bridge"))
    if not on or not user["is_admin"]:
        return JSONResponse({"enabled": False})
    return JSONResponse({"enabled": True, **await run_in_threadpool(fusion.status)})


async def job_fusion(request: Request):
    """Send the job's design to the Fusion running on this machine (owner only, local only)."""
    from . import fusion
    data = await body_json(request)
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"])
    if not settings.fusion_bridge or not user["is_admin"]:
        err(403, "Send to Fusion works only where AeroSculpt runs on the same computer as Fusion.")
    if j["status"] != "done":
        err(409, "The job has not finished.")
    spec = json.loads(j["spec"])
    result = json.loads((job_dir(j["id"]) / "result.json").read_text())
    action = data.get("action")
    try:
        if action == "parameters":
            params = _template_params(spec, result)
            if not params:
                err(400, "Only template designs have named dimensions to send; use Import instead.")
            res = await run_in_threadpool(fusion.push_parameters, params)
        elif action == "import":
            from .geometry import cad
            out = job_dir(j["id"]) / "design.step"
            if not out.exists():
                if not cad.available():
                    err(501, "STEP export needs gmsh on the server.")
                await run_in_threadpool(_step_for_job, j, out)
            res = await run_in_threadpool(fusion.import_step, out)
        else:
            err(400, "action must be parameters or import.")
    except fusion.FusionError as e:
        err(502, str(e))
    except ValueError as e:
        err(413, str(e))
    with db() as con:
        audit(con, user["id"], "fusion_" + action, j["id"])
    return JSONResponse(res)


async def project_wheels(request: Request):
    """Wheel boxes found on the model (a proposal to check), in the file's own axes and units."""
    from .geometry.wheels import detect_wheels
    from .optimize.runner import ModelFrame, load_model
    data = await body_json(request)
    with db() as con:
        user = current_user(request, con)
        p = _get_project(con, user, request.path_params["pid"], "write")
    st = _clean_settings({k: data[k] for k in ("units", "unit_scale", "forward", "up") if k in data},
                         json.loads(p["settings"] or "{}"))

    def find():
        fr = ModelFrame(load_model(project_dir(p["id"]) / p["filename"], p["filename"]), st.get("units", "m"),
                        st.get("forward", "-x"), st.get("up", "+z"), st.get("unit_scale"))
        A = np.abs(fr.R0)
        out = []
        for k, w in enumerate(detect_wheels(fr.body)):
            c = ((np.asarray(w["center"]) + fr.center) @ fr.R0) / fr.scale
            size = (A.T @ np.asarray(w["size"])) / fr.scale
            out.append({"id": uuid.uuid4().hex[:8], "type": "wheel", "shape": "box", "rotation": [0, 0, 0],
                        "center": [round(float(v), 6) for v in c], "size": [round(float(v), 6) for v in size],
                        "name": f"Wheel {k + 1}"})
        return out
    return JSONResponse({"zones": await run_in_threadpool(find)})


async def project_estimate(request: Request):
    """How long a job would take, from the grids it will actually use (no job is created)."""
    from .estimate import estimate
    data = await body_json(request)
    with db() as con:
        user = current_user(request, con)
        project = _get_project(con, user, request.path_params["pid"])
        plan = plan_for(user)
    spec, st, _ = _build_spec(data, project, plan)
    out = await run_in_threadpool(estimate, spec, json.loads(project["metrics"] or "{}"), st)
    return JSONResponse(out)


async def job_mission(request: Request):
    """Mission performance (endurance, range, climb, best speeds) of a winged template job's
    original and optimized designs, with the user's battery and propulsion numbers."""
    from .aero.mission import MissionInputs, mission
    from .cfd.analysis import isa
    data = await body_json(request)
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"])
    if j["status"] != "done":
        err(409, "The job has not finished.")
    spec = json.loads(j["spec"])
    tm = spec.get("template")
    if not tm or tm["id"] not in ("uav", "wing"):
        err(400, "Mission performance needs a winged template design (fixed-wing UAV or wing).")
    result = json.loads((job_dir(j["id"]) / "result.json").read_text())
    try:
        mass = float(data.get("mass_kg") or spec.get("optimize", {}).get("mass_kg") or 0)
    except (TypeError, ValueError):
        err(400, "Invalid mass.")
    if not (0.01 <= mass <= 1e5):
        err(400, "Give the flight mass (kg).")
    inputs = MissionInputs.from_spec(data)
    env = spec.get("envelope", {})
    air = isa(float(env.get("altitude", 0.0)))
    vmax = 1.3 * max(env.get("speeds") or [20.0])
    best_vals = {**tm["values"], **((result.get("template") or {}).get("best_values") or {})}

    def run():
        out = {"base": mission(T.build(tm["id"], tm["values"]), mass, air, inputs, v_max=vmax)}
        if result.get("kind") == "optimize":
            out["best"] = mission(T.build(tm["id"], best_vals), mass, air, inputs, v_max=vmax)
        return out
    return JSONResponse(await run_in_threadpool(run))


async def job_export(request: Request):
    """Design and report exports of a finished job: step (CAD), pdf (report), vtk (flow fields)."""
    fmt = request.path_params["fmt"]
    with db() as con:
        user = current_user(request, con)
        j = _get_job(con, user, request.path_params["jid"])
        plan = plan_for(user)
        project = con.execute("SELECT name FROM projects WHERE id=?", (j["project_id"],)).fetchone()
    if j["status"] != "done":
        err(409, "The job has not finished.")
    base = _safe_name(project["name"] if project else None)
    jd = job_dir(j["id"])
    if fmt == "step":
        from .geometry import cad
        if not plan["export"]:
            err(403, "Exporting geometry is available on Pro and Team.", upgrade=True)
        if not cad.available():
            err(501, "STEP export needs gmsh on the server.")
        out = jd / "design.step"
        if not out.exists():
            try:
                await run_in_threadpool(_step_for_job, j, out)
            except ValueError as e:
                err(413, str(e))
        return FileResponse(out, media_type="model/step", filename=f"{base} - {'optimized' if j['kind'] == 'optimize' else 'design'}.step")
    if fmt == "pdf":
        from .report import build_pdf
        out = jd / "report.pdf"
        if not out.exists() or out.stat().st_mtime < (jd / "result.json").stat().st_mtime:
            spec = json.loads(j["spec"])
            result = json.loads((jd / "result.json").read_text())
            await run_in_threadpool(build_pdf, jd, spec, result, project["name"] if project else "Design", out, j["id"])
        return FileResponse(out, media_type="application/pdf", filename=f"{base} - report.pdf")
    if fmt == "vtk":
        from .cfd.volume import vtk_surface, vtk_volume, vtk_zip
        from .geometry.mesh import load_mesh

        def build():
            result = json.loads((jd / "result.json").read_text())
            entries = []
            for tag, stl in (("baseline", "baseline_body.stl"), ("best", "best_body.stl")):
                if not (jd / stl).exists():
                    continue
                mesh = load_mesh((jd / stl).read_bytes(), stl)
                for e in ((result.get("fields") or {}).get(tag) or []):
                    k = e["file"].rsplit("_", 1)[-1].split(".")[0]
                    npz = jd / f"volume_{tag}_{k}.npz"
                    if not npz.exists() or e.get("mirror"):
                        continue
                    z = np.load(npz)
                    vol = {key: z[key] for key in ("origin", "u", "cp", "solid")}
                    vol["spacing"] = float(z["spacing"])
                    label = f"{'original' if tag == 'baseline' else 'optimized'}_aoa{e['aoa']:g}"
                    entries.append((f"{label}_flow.vtk", vtk_volume(vol, f"AeroSculpt {label} mean flow")))
                    from .optimize.runner import wind_rotation
                    wind = mesh.transformed(R=wind_rotation(e["aoa"], float((result.get("envelope") or {}).get("yaw", 0.0))))
                    entries.append((f"{label}_surface.vtk", vtk_surface(wind, z["cp_face"].astype(np.float32) if len(z["cp_face"]) else None)))
            if not entries:
                raise ValueError("this job has no saved flow field (run it again to export one)")
            readme = ("AeroSculpt flow export (legacy VTK, open in ParaView: File > Open).\n"
                      "*_flow.vtk: mean velocity / freestream (vector), pressure coefficient cp, Q-criterion and a\n"
                      "solid mask on a uniform grid, wind axes (flow along +x, z up), metres.\n"
                      "*_surface.vtk: the body in the same axes with cp per face.\n")
            return vtk_zip([("README.txt", readme.encode())] + entries)
        try:
            data = await run_in_threadpool(build)
        except ValueError as e:
            err(404, str(e))
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{base} - flow fields (VTK).zip"'})
    if fmt == "fusion-script":
        from . import fusion
        params = _template_params(json.loads(j["spec"]), json.loads((jd / "result.json").read_text()))
        if not params:
            err(400, "Only template designs have named dimensions for Fusion.")
        return Response(fusion.parameter_script(params), media_type="text/x-python",
                        headers={"Content-Disposition": f'attachment; filename="{base} - Fusion parameters.py"'})
    err(404, "Unknown export format.")


# ---- teams ----------------------------------------------------------------------------------------
async def teams_list(request: Request):
    from . import teams
    with db() as con:
        user = current_user(request, con)
        if request.method == "GET":
            out = []
            for t in teams.teams_of(con, user["id"]):
                mem, inv = teams.members(con, t["id"])
                out.append({**t, "members": mem, "invites": inv if teams.can(t["role"], "admin") else []})
            return JSONResponse(out)
        if plan_for(user)["key"] != "team" and not user["is_admin"]:
            err(403, "Teams are part of the Team plan.", upgrade=True)
    data = await body_json(request)
    name = (data.get("name") or "").strip()[:80]
    if not name:
        err(400, "Give the team a name.")
    with transaction() as con:
        tid = teams.create(con, user["id"], name)
        audit(con, user["id"], "team_created", name, team_id=tid)
    return JSONResponse({"id": tid, "name": name}, status_code=201)


async def team_members(request: Request):
    """POST {email, role}: invite; PATCH {user_id, role}: change a role; DELETE {user_id}: remove."""
    from . import teams
    tid = int(request.path_params["tid"])
    data = await body_json(request)
    with transaction() as con:
        user = current_user(request, con)
        me_role = teams.role_of(con, tid, user["id"])
        if not teams.can(me_role, "admin"):
            err(403, "Only team owners and admins manage members.")
        role = data.get("role", "editor")
        if request.method in ("POST", "PATCH") and role not in teams.ROLES:
            err(400, "Unknown role.")
        if role == "owner" and me_role != "owner" and request.method != "DELETE":
            err(403, "Only an owner can make someone an owner.")
        if request.method == "POST":
            email = normalize_email(data.get("email"))
            if not valid_email(email):
                err(400, "Enter a valid email address.")
            how = teams.invite(con, tid, email, role, user["id"])
            audit(con, user["id"], "team_invite", f"{email} as {role} ({how})", team_id=tid)
            return JSONResponse({"ok": True, "status": how})
        uid = int(data.get("user_id") or 0)
        target = teams.role_of(con, tid, uid)
        if target is None:
            err(404, "Not a member.")
        if target == "owner" and me_role != "owner":
            err(403, "Only an owner can change another owner.")
        owners = con.execute("SELECT COUNT(*) AS n FROM team_members WHERE team_id=? AND role='owner'", (tid,)).fetchone()["n"]
        if target == "owner" and owners <= 1 and (request.method == "DELETE" or role != "owner"):
            err(400, "A team needs at least one owner.")
        if request.method == "PATCH":
            con.execute("UPDATE team_members SET role=? WHERE team_id=? AND user_id=?", (role, tid, uid))
            audit(con, user["id"], "team_role", f"user {uid}: {target} -> {role}", team_id=tid)
        else:
            con.execute("DELETE FROM team_members WHERE team_id=? AND user_id=?", (tid, uid))
            audit(con, user["id"], "team_remove", f"user {uid} ({target})", team_id=tid)
    return JSONResponse({"ok": True})


async def team_audit(request: Request):
    from . import teams
    tid = int(request.path_params["tid"])
    with db() as con:
        user = current_user(request, con)
        if not teams.can(teams.role_of(con, tid, user["id"]), "admin"):
            err(403, "Only team owners and admins can read the audit trail.")
        rows = con.execute("""SELECT a.action, a.detail, a.at, u.email FROM audit a LEFT JOIN users u ON u.id = a.user_id
                              WHERE a.team_id=? ORDER BY a.at DESC LIMIT 500""", (tid,)).fetchall()
    return JSONResponse([dict(r) for r in rows])


async def project_share(request: Request):
    """{team_id} shares the project with a team you belong to (editor or above); null stops sharing."""
    from . import teams
    data = await body_json(request)
    with transaction() as con:
        user = current_user(request, con)
        p = _get_project(con, user, request.path_params["pid"], "owner")
        tid = data.get("team_id")
        if tid is not None:
            tid = int(tid)
            if not teams.can(teams.role_of(con, tid, user["id"]), "editor"):
                err(403, "You can only share with a team you edit in.")
        con.execute("UPDATE projects SET team_id=? WHERE id=?", (tid, p["id"]))
        audit(con, user["id"], "project_shared" if tid else "project_unshared", p["name"], team_id=tid or p["team_id"])
    return JSONResponse({"ok": True, "team_id": tid})


async def sso_info(request: Request):
    from . import sso
    return JSONResponse({"enabled": sso.enabled(), "name": sso.config()["name"] if sso.enabled() else None})


async def sso_start(request: Request):
    from . import sso
    if not sso.enabled():
        err(404, "Single sign-on is not configured on this server.")
    url = await run_in_threadpool(sso.start, settings.public_url + "/auth/sso/callback")
    return RedirectResponse(url, status_code=302)


async def sso_callback(request: Request):
    from . import sso, teams
    if not sso.enabled():
        err(404, "Single sign-on is not configured on this server.")
    q = request.query_params
    if q.get("error"):
        return RedirectResponse("/login?sso_error=" + urllib.parse.quote(q.get("error_description") or q["error"]), status_code=302)
    try:
        who = await run_in_threadpool(sso.finish, q.get("code", ""), q.get("state", ""), settings.public_url + "/auth/sso/callback")
    except Exception as e:  # noqa: BLE001 - any verification failure ends here, with no account touched
        msg = str(e) if isinstance(e, sso.SSOError) else "the sign-in could not be verified"
        return RedirectResponse("/login?sso_error=" + urllib.parse.quote(msg), status_code=302)
    resp = RedirectResponse("/app", status_code=302)
    with transaction() as con:
        user = con.execute("SELECT * FROM users WHERE sso_subject=?", (who["sub"],)).fetchone() or \
            con.execute("SELECT * FROM users WHERE email=?", (who["email"],)).fetchone()
        if user is None:
            if settings.allowed_signup_emails and who["email"] not in settings.allowed_signup_emails \
                    and not con.execute("SELECT 1 FROM team_invites WHERE email=?", (who["email"],)).fetchone():
                return RedirectResponse("/login?sso_error=" + urllib.parse.quote("sign-up on this server is by invitation only"),
                                        status_code=302)
            uid = con.execute("INSERT INTO users (email, name, password_hash, created_at, sso_subject) VALUES (?,?,?,?,?)",
                              (who["email"], who["name"], "!", time.time(), who["sub"])).lastrowid
            audit(con, uid, "signup_sso", who["email"])
        else:
            uid = user["id"]
            con.execute("UPDATE users SET sso_subject=? WHERE id=?", (who["sub"], uid))
        _apply_owner(con, uid, who["email"])
        teams.accept_invites(con, uid, who["email"])
        start_session(con, request, uid, resp)
        audit(con, uid, "login_sso", _client_ip(request))
    return resp


# ---- billing --------------------------------------------------------------------------------
async def billing_checkout(request: Request):
    data = await body_json(request)
    plan_key = data.get("plan")
    if plan_key not in ("pro", "team"):
        err(400, "Choose Pro or Team.")
    with transaction() as con:
        user = current_user(request, con)
        if user["billing_source"] == "guest":
            err(403, "Guest demo accounts cannot change plan.")
        if user["plan"] == plan_key and user["plan_status"] in billing.ACTIVE:
            err(400, f"You are already on {PLANS[plan_key]['name']}.")
        mode = settings.billing_mode
        if mode == "stripe":
            if user["stripe_subscription_id"] and user["plan_status"] in billing.ACTIVE:
                # plan changes and cancellations happen in the Stripe portal
                url = billing.create_portal(con, user)
                return JSONResponse({"url": url, "mode": mode})
            try:
                url = billing.create_checkout(con, user, plan_key)
            except billing.BillingError as e:
                err(502, str(e))
            return JSONResponse({"url": url, "mode": mode})
        if mode == "demo":
            billing.demo_checkout(con, user, plan_key)
            return JSONResponse({"url": "/app/billing?checkout=success&demo=1", "mode": mode})
    err(503, "Billing is not configured on this server.")


async def billing_portal(request: Request):
    with db() as con:
        user = current_user(request, con)
        if settings.billing_mode != "stripe":
            err(400, "The billing portal is only available with Stripe.")
        try:
            return JSONResponse({"url": billing.create_portal(con, user)})
        except billing.BillingError as e:
            err(400, str(e))


async def billing_demo(request: Request):
    """Demo-mode subscription management (cancel / resume / downgrade now)."""
    if settings.billing_mode != "demo":
        err(404, "Not found.")
    data = await body_json(request)
    action = data.get("action")
    with db() as con:
        user = current_user(request, con)
        if user["billing_source"] == "guest":
            err(403, "Guest demo accounts cannot change plan.")
        if action == "cancel":
            billing.demo_cancel(con, user)
        elif action == "resume":
            billing.demo_resume(con, user)
        elif action == "downgrade_now":
            con.execute("UPDATE users SET plan='lite', plan_status='active', cancel_at_period_end=0, "
                        "current_period_end=NULL, trial_ends_at=NULL WHERE id=?", (user["id"],))
        else:
            err(400, "Unknown action.")
        user = con.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        return JSONResponse(user_public(con, user))


async def billing_webhook(request: Request):
    payload = await request.body()
    try:
        event = billing.verify_webhook(payload, request.headers.get("stripe-signature", ""))
    except billing.BillingError as e:
        log.warning("rejected webhook: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)
    try:
        with transaction() as con:
            outcome = billing.handle_event(con, event)
    except billing.BillingError as e:
        log.error("webhook processing failed: %s", e)
        return JSONResponse({"error": "retry"}, status_code=500)
    return JSONResponse({"received": True, "outcome": outcome})


# ---- admin -------------------------------------------------------------------------------------
# ---- desktop: licence, local settings, updates, diagnostics, project files -------------------------
async def license_info(request: Request):
    from . import license as lic
    with db() as con:
        user = current_user(request, con)
    if settings.open_access:
        return JSONResponse({"local": True, **lic.installed()})
    plan = plan_for(user)
    return JSONResponse({"local": False, "can_issue": plan["key"] in lic.PLANS, "plan": plan["key"],
                         "signing": lic.load_private_key() is not None})


async def license_issue(request: Request):
    """A desktop licence key for a Pro or Team subscriber (bound to their email), valid to the end of
    the billing period plus the grace period, or a year for an owner."""
    from . import license as lic
    with db() as con:
        user = current_user(request, con)
        plan = plan_for(user)
        if plan["key"] not in lic.PLANS:
            err(403, "Desktop licence keys come with Pro and Team.", upgrade=True)
        now = time.time()
        end = user["current_period_end"]
        expires = end if end and end > now + 7 * 86400 else now + 365 * 86400
        try:
            key = lic.issue(user["email"], plan["key"], expires_at=expires)
        except lic.LicenseError as e:
            err(503, str(e))
        audit(con, user["id"], "license_issued", plan["key"])
    return JSONResponse({"key": key, "plan": plan["key"], "expires": expires, "email": user["email"]})


async def local_license(request: Request):
    from . import license as lic
    if not settings.open_access:
        err(404, "Not found.")
    with db() as con:
        current_user(request, con)
    if request.method == "DELETE":
        lic.remove()
        return JSONResponse(lic.installed())
    data = await body_json(request)
    try:
        st = lic.install(str(data.get("key") or ""))
    except lic.LicenseError as e:
        err(400, f"That key cannot be used: {e}")
    return JSONResponse(st)


async def local_settings_api(request: Request):
    from .local import local_settings, save_local_settings
    if not settings.open_access:
        err(404, "Not found.")
    with db() as con:
        current_user(request, con)
    if request.method == "GET":
        return JSONResponse(local_settings())
    data = await body_json(request)
    return JSONResponse(save_local_settings(data if isinstance(data, dict) else {}))


async def update_check(request: Request):
    from .local import local_settings
    from . import update
    with db() as con:
        current_user(request, con)
    enabled = settings.check_updates or (settings.open_access and local_settings().get("check_updates"))
    if not enabled:
        return JSONResponse({"enabled": False, "current": VERSION})
    force = request.query_params.get("force") == "1"
    return JSONResponse({"enabled": True, **await run_in_threadpool(update.check, VERSION, 6.0, force)})


async def diagnostics_bundle(request: Request):
    from .diagnostics import bundle
    with db() as con:
        user = current_user(request, con)
    if not (user["is_admin"] or settings.open_access):
        err(403, "Admins only.")
    data = await run_in_threadpool(bundle)
    name = f"aerosculpt-diagnostics-{time.strftime('%Y%m%d-%H%M')}.zip"
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


async def third_party_notices(request: Request):
    for f in (Path(__file__).resolve().parents[1] / "THIRD-PARTY-NOTICES.txt", Path(getattr(__import__("sys"), "_MEIPASS", ".")) / "THIRD-PARTY-NOTICES.txt"):
        if f.exists():
            return Response(f.read_bytes(), media_type="text/plain; charset=utf-8")
    return Response("Third-party notices are generated at build time (scripts/third_party_notices.py).", media_type="text/plain")


PROJECT_FILE_FORMAT = "aerosculpt-project"


async def project_export(request: Request):
    """A project as one file: the model, its source (CAD / OpenRocket), settings and metrics, in a
    format later versions keep opening."""
    import io
    import zipfile
    with db() as con:
        user = current_user(request, con)
        p = _get_project(con, user, request.path_params["pid"])
    pdir = project_dir(p["id"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps({"format": PROJECT_FILE_FORMAT, "version": 1, "app": VERSION, "build": build_id(),
                                               "name": p["name"], "filename": p["filename"], "source_format": p["source_format"],
                                               "settings": json.loads(p["settings"] or "{}"), "metrics": json.loads(p["metrics"] or "{}"),
                                               "exported_at": time.time()}, indent=1))
        for f in pdir.iterdir():
            if f.is_file() and (f.name.startswith(("model.", "source.")) or f.name in ("flight_log.json", "thumb.png")):
                z.write(f, f.name)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{_safe_name(p["name"])}.aerosculpt"'})


def _import_project_file(raw):
    """-> (project.json dict, model bytes, model filename, {other files})"""
    import io
    import zipfile
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        meta = json.loads(z.read("project.json").decode("utf-8"))
    except Exception:  # noqa: BLE001
        err(400, "Not an AeroSculpt project file.")
    if meta.get("format") != PROJECT_FILE_FORMAT:
        err(400, "Not an AeroSculpt project file.")
    model_name = next((n for n in z.namelist() if n.startswith("model.")), None)
    if not model_name:
        err(400, "The project file has no model.")
    others = {n: z.read(n) for n in z.namelist() if n != "project.json" and n != model_name and len(z.getinfo(n).filename) < 80}
    return meta, z.read(model_name), model_name, others


async def admin_stats(request: Request):
    with db() as con:
        user = current_user(request, con)
        if not user["is_admin"]:
            err(403, "Admins only.")
        by_plan = {r["plan"]: r["n"] for r in con.execute(
            "SELECT plan, COUNT(*) AS n FROM users WHERE plan_status IN ('active','trialing','past_due') GROUP BY plan")}
        trialing = con.execute("SELECT COUNT(*) AS n FROM users WHERE plan_status='trialing'").fetchone()["n"]
        paying = {k: con.execute("SELECT COUNT(*) AS n FROM users WHERE plan=? AND plan_status IN ('active','past_due')",
                                 (k,)).fetchone()["n"] for k in ("pro", "team")}
        mrr = sum(PLANS[k]["price_month"] * n for k, n in paying.items())
        users = con.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        jobs = {r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")}
        recent = [dict(r) for r in con.execute(
            "SELECT id, email, plan, plan_status, created_at, last_login_at FROM users ORDER BY created_at DESC LIMIT 25")]
        credits = con.execute("SELECT COALESCE(SUM(credits),0) AS c, COALESCE(SUM(cfd_runs),0) AS r FROM usage WHERE period=?",
                              (time.strftime("%Y-%m", time.gmtime()),)).fetchone()
    return JSONResponse({"users": users, "by_plan": by_plan, "trialing": trialing, "paying": paying, "mrr": mrr,
                         "jobs": jobs, "recent_users": recent, "credits_this_month": credits["c"],
                         "cfd_runs_this_month": credits["r"], "billing_mode": settings.billing_mode})


# ---- pages -------------------------------------------------------------------------------------
THREE_VERSION = "0.170.0"
IMPORTMAP = ('{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@%s/build/three.module.js",'
             '"three/addons/":"https://cdn.jsdelivr.net/npm/three@%s/examples/jsm/"}}' % (THREE_VERSION, THREE_VERSION))
IMPORTMAP_TAG = f'<script type="importmap">{IMPORTMAP}</script>'
IMPORTMAP_HASH = "sha256-" + base64.b64encode(hashlib.sha256(IMPORTMAP.encode()).digest()).decode()
_page_cache = {}


def page(name, allow_eval=False):
    """Serve a page; the <!--IMPORTMAP--> marker becomes the three.js import map, which the
    CSP allows by hash so no inline script ever needs 'unsafe-inline'."""
    headers = {"Cache-Control": "no-cache"}
    if allow_eval:
        headers["Content-Security-Policy"] = csp(allow_eval=True)

    async def handler(request: Request):
        f = PAGES / f"{name}.html"
        mtime = f.stat().st_mtime
        cached = _page_cache.get(name)
        if not cached or cached[0] != mtime:
            cached = (mtime, f.read_text(encoding="utf-8").replace("<!--IMPORTMAP-->", IMPORTMAP_TAG))
            _page_cache[name] = cached
        return HTMLResponse(cached[1], headers=headers)
    return handler


def auth_page(name):
    """Sign-in pages; in local preview mode there is nothing to sign in to, so go to the app."""
    inner = page(name)

    async def handler(request: Request):
        if settings.open_access or (settings.anonymous_access and name == "signup"):
            return RedirectResponse("/app", status_code=302)
        return await inner(request)
    return handler


async def not_found(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Not found."}, status_code=404)
    return HTMLResponse((PAGES / "404.html").read_text(encoding="utf-8"), status_code=404)


async def api_error(request: Request, exc: ApiError):
    return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)


async def server_error(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"error": "Something went wrong on our side."}, status_code=500)


def csp(allow_eval=False):
    """Content-Security-Policy. allow_eval is needed only where the in-browser CAD converter
    (OpenCascade compiled with embind) runs, i.e. the upload page."""
    script = "script-src 'self' 'wasm-unsafe-eval' https://cdn.jsdelivr.net '" + IMPORTMAP_HASH + "'"
    if allow_eval:
        script += " 'unsafe-eval'"
    return ("default-src 'self'; " + script + "; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; "
            "connect-src 'self' https://cdn.jsdelivr.net; worker-src 'self' blob:; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self' https://checkout.stripe.com")


CSP_DEFAULT = csp().encode()


class SecurityHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                h = message.setdefault("headers", [])
                h += [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                      (b"referrer-policy", b"strict-origin-when-cross-origin"),
                      (b"permissions-policy", b"camera=(), microphone=(), geolocation=()")]
                if not any(k.lower() == b"content-security-policy" for k, _ in h):
                    h.append((b"content-security-policy", CSP_DEFAULT))
                if settings.secure_cookies:
                    h.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
            await send(message)
        return await self.app(scope, receive, send_wrapper)


class RevalidatedStatic(StaticFiles):
    """Static files that browsers must revalidate (ETag -> cheap 304), so a release never
    leaves customers running yesterday's JavaScript."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app):
    ensure_dirs()
    init_db()
    if settings.billing_mode == "demo":
        log.warning("Billing is in DEMO mode: plans are granted without payment. Set STRIPE_SECRET_KEY for production.")
    yield


from .local import LocalGuard
from .web_rocket import ROUTES as ROCKET_ROUTES
from .rocket.runner import OBJECTIVES as ROCKET_GOALS

routes = [
    Route("/api/health", health),
    *ROCKET_ROUTES,
    Route("/api/license", license_info),
    Route("/api/license/issue", license_issue, methods=["POST"]),
    Route("/api/local/license", local_license, methods=["POST", "DELETE"]),
    Route("/api/local/settings", local_settings_api, methods=["GET", "POST"]),
    Route("/api/update", update_check),
    Route("/api/diagnostics", diagnostics_bundle),
    Route("/api/projects/{pid}/export", project_export),
    Route("/third-party-notices", third_party_notices),
    Route("/api/auth/signup", signup, methods=["POST"]),
    Route("/api/auth/login", login, methods=["POST"]),
    Route("/api/auth/guest", guest, methods=["POST"]),
    Route("/api/auth/claim", claim, methods=["POST"]),
    Route("/api/auth/logout", logout, methods=["POST"]),
    Route("/api/auth/forgot", forgot, methods=["POST"]),
    Route("/api/auth/reset", reset, methods=["POST"]),
    Route("/api/me", me, methods=["GET", "PATCH"]),
    Route("/api/me/password", change_password, methods=["POST"]),
    Route("/api/me/delete", delete_account, methods=["POST"]),
    Route("/api/me/api-key", api_key, methods=["POST", "DELETE"]),
    Route("/api/jobs/{jid}/export/{fmt}", job_export),
    Route("/api/jobs/{jid}/fusion", job_fusion, methods=["POST"]),
    Route("/api/jobs/{jid}/mission", job_mission, methods=["POST"]),
    Route("/api/projects/{pid}/estimate", project_estimate, methods=["POST"]),
    Route("/api/projects/{pid}/wheels", project_wheels, methods=["POST"]),
    Route("/api/projects/{pid}/share", project_share, methods=["POST"]),
    Route("/api/teams", teams_list, methods=["GET", "POST"]),
    Route("/api/teams/{tid:int}/members", team_members, methods=["POST", "PATCH", "DELETE"]),
    Route("/api/teams/{tid:int}/audit", team_audit),
    Route("/api/auth/sso", sso_info),
    Route("/auth/sso/start", sso_start),
    Route("/auth/sso/callback", sso_callback),
    Route("/api/fusion", fusion_status),
    Route("/api/plans", plans),
    Route("/api/samples", samples),
    Route("/api/templates", templates_catalog),
    Route("/api/templates/build", template_build, methods=["POST"]),
    Route("/api/propellers/estimate", propeller_estimate, methods=["POST"]),
    Route("/api/validation", validation),
    Route("/api/projects", projects, methods=["GET", "POST"]),
    Route("/api/projects/{pid}", project_item, methods=["GET", "PATCH", "DELETE"]),
    Route("/api/projects/{pid}/model", project_model),
    Route("/api/projects/{pid}/thumbnail", project_thumb, methods=["GET", "POST"]),
    Route("/api/projects/{pid}/jobs", project_jobs, methods=["GET", "POST"]),
    Route("/api/projects/{pid}/features/suggest", feature_suggest, methods=["POST"]),
    Route("/api/projects/{pid}/solidify", project_solidify, methods=["POST"]),
    Route("/api/projects/{pid}/features/propose", feature_propose, methods=["POST"]),
    Route("/api/projects/{pid}/features/ai", feature_ai, methods=["POST"]),
    Route("/api/projects/{pid}/features/build", feature_build, methods=["POST"]),
    Route("/api/jobs/{jid}", job_item),
    Route("/api/jobs/{jid}/cancel", job_cancel, methods=["POST"]),
    Route("/api/jobs/{jid}/files/{name}", job_file),
    Route("/api/jobs/{jid}/promote", job_promote, methods=["POST"]),
    Route("/api/billing/checkout", billing_checkout, methods=["POST"]),
    Route("/api/billing/portal", billing_portal, methods=["POST"]),
    Route("/api/billing/demo", billing_demo, methods=["POST"]),
    Route("/api/billing/webhook", billing_webhook, methods=["POST"]),
    Route("/api/admin/stats", admin_stats),
    Route("/", page("index")),
    Route("/pricing", page("pricing")),
    Route("/validation", page("validation")),
    Route("/docs", page("docs")),
    Route("/login", auth_page("login")),
    Route("/signup", auth_page("signup")),
    Route("/forgot", auth_page("forgot")),
    Route("/reset", auth_page("reset")),
    Route("/terms", page("terms")),
    Route("/privacy", page("privacy")),
    Route("/app", page("dashboard", allow_eval=True)),
    Route("/app/project/{pid}", page("workspace")),
    Route("/app/report/{jid}", page("report")),
    Route("/app/billing", page("billing")),
    Route("/app/account", page("account")),
    Route("/admin", page("admin")),
    Mount("/static", RevalidatedStatic(directory=STATIC), name="static"),
]

app = Starlette(
    routes=routes,
    middleware=[Middleware(LocalGuard), Middleware(AnonymousAccess), Middleware(SecurityHeaders), Middleware(GZipMiddleware, minimum_size=1024)],
    exception_handlers={404: not_found, ApiError: api_error, Exception: server_error},
    lifespan=lifespan,
)
