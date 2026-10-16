"""Single sign-on with OpenID Connect (authorization code flow with PKCE).

Works with any standard provider: Okta, Microsoft Entra ID (Azure AD), Google Workspace, Auth0,
Keycloak, OneLogin. Configure in .env:

    OIDC_ISSUER=https://login.example.com            (its /.well-known/openid-configuration is read)
    OIDC_CLIENT_ID=...
    OIDC_CLIENT_SECRET=...
    OIDC_NAME=Example SSO                              (the button label)
    OIDC_ALLOWED_DOMAINS=example.com,example.org       (optional: only these email domains)

The provider's ID token is verified (signature against its published keys, issuer, audience,
expiry, nonce) before any account is touched. A first sign-in creates the account (no password)
and joins any team that invited the email.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
import urllib.request

_DISCOVERY = {"at": 0.0, "doc": None}
_PENDING = {}          # state -> (nonce, verifier, created_at)
PENDING_TTL = 600


def config():
    return {"issuer": os.environ.get("OIDC_ISSUER", "").rstrip("/"), "client_id": os.environ.get("OIDC_CLIENT_ID", ""),
            "client_secret": os.environ.get("OIDC_CLIENT_SECRET", ""), "name": os.environ.get("OIDC_NAME", "Single sign-on"),
            "domains": [d.strip().lower() for d in os.environ.get("OIDC_ALLOWED_DOMAINS", "").split(",") if d.strip()]}


def enabled():
    c = config()
    return bool(c["issuer"] and c["client_id"])


def _get_json(url, data=None, timeout=15):
    req = urllib.request.Request(url, data=data, headers={"Accept": "application/json",
                                                           **({"Content-Type": "application/x-www-form-urlencoded"} if data else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def discovery():
    if _DISCOVERY["doc"] is None or time.time() - _DISCOVERY["at"] > 3600:
        _DISCOVERY["doc"] = _get_json(config()["issuer"] + "/.well-known/openid-configuration")
        _DISCOVERY["at"] = time.time()
    return _DISCOVERY["doc"]


def _b64url(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def start(redirect_uri):
    """-> URL of the provider's sign-in page."""
    for k in [k for k, v in _PENDING.items() if time.time() - v[2] > PENDING_TTL]:
        _PENDING.pop(k, None)
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    _PENDING[state] = (nonce, verifier, time.time())
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    q = {"response_type": "code", "client_id": config()["client_id"], "redirect_uri": redirect_uri,
         "scope": "openid email profile", "state": state, "nonce": nonce,
         "code_challenge": challenge, "code_challenge_method": "S256"}
    return discovery()["authorization_endpoint"] + "?" + urllib.parse.urlencode(q)


class SSOError(Exception):
    pass


def finish(code, state, redirect_uri):
    """Exchange the code, verify the ID token -> {"sub", "email", "name"}."""
    import jwt
    pending = _PENDING.pop(state or "", None)
    if pending is None or time.time() - pending[2] > PENDING_TTL:
        raise SSOError("the sign-in expired or did not start here; try again")
    nonce, verifier, _ = pending
    c = config()
    d = discovery()
    tok = _get_json(d["token_endpoint"], urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
        "client_id": c["client_id"], "client_secret": c["client_secret"], "code_verifier": verifier}).encode())
    raw = tok.get("id_token")
    if not raw:
        raise SSOError("the provider returned no identity")
    key = jwt.PyJWKClient(d["jwks_uri"]).get_signing_key_from_jwt(raw).key
    claims = jwt.decode(raw, key, algorithms=["RS256", "ES256", "PS256"], audience=c["client_id"], issuer=d.get("issuer", c["issuer"]),
                        options={"require": ["exp", "iat", "sub"]})
    if claims.get("nonce") != nonce:
        raise SSOError("the sign-in could not be verified (nonce)")
    email = (claims.get("email") or "").strip().lower()
    if not email or claims.get("email_verified") is False:
        raise SSOError("the provider did not confirm an email address")
    if c["domains"] and email.rsplit("@", 1)[-1] not in c["domains"]:
        raise SSOError("this email domain is not allowed to sign in here")
    return {"sub": f"{d.get('issuer', c['issuer'])}|{claims['sub']}", "email": email, "name": (claims.get("name") or "")[:80]}
