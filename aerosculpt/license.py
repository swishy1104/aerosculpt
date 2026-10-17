"""Desktop licence keys: Ed25519-signed, checked offline, with a grace period.

A key is "AS1.<payload>.<signature>" (base64url): the payload names the licensee's email, the
plan (pro or team), the issue and expiry dates and a key id. The app holds only the public key, so
a key can be verified without a network; the private key stays on the licensing server
(LICENSE_PRIVATE_KEY: PEM text or a path; by default data/license_private.pem).

Offline grace: a key is honoured until GRACE_DAYS after its expiry, so a subscriber whose renewal
could not be fetched keeps working for a month. Renewal is a fresh key from the account page.
"""
from __future__ import annotations

import base64
import json
import secrets
import time
from pathlib import Path

PUBLIC_KEY_B64 = "rzg6RsFjUwJktBRVilRdSpNDUlL3THEaCaFrk0gKCZg="
GRACE_DAYS = 30
PLANS = ("pro", "team")


class LicenseError(ValueError):
    pass


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _public_key(b64=None):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    return ed25519.Ed25519PublicKey.from_public_bytes(_b64d(b64 or PUBLIC_KEY_B64))


def load_private_key(source=None):
    """The signing key from PEM text, a path, or the default file; None when not configured."""
    from cryptography.hazmat.primitives import serialization
    if source is None:
        from .config import settings
        source = settings.license_private_key or str(settings.data_dir / "license_private.pem")
    pem = None
    if "-----BEGIN" in source:
        pem = source.encode()
    elif Path(source).exists():
        pem = Path(source).read_bytes()
    if not pem:
        return None
    return serialization.load_pem_private_key(pem, password=None)


def issue(email, plan, days=365, private_key=None, expires_at=None):
    if plan not in PLANS:
        raise LicenseError("plans with a desktop licence are Pro and Team")
    key = private_key or load_private_key()
    if key is None:
        raise LicenseError("licence signing is not configured on this server")
    now = int(time.time())
    payload = {"v": 1, "email": (email or "").strip().lower(), "plan": plan, "issued": now,
               "expires": int(expires_at) if expires_at else now + int(days) * 86400, "id": secrets.token_hex(6)}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return f"AS1.{_b64e(body)}.{_b64e(key.sign(body))}"


def verify(key_text, public_b64=None):
    """The payload of a well-signed key (expiry is judged by status(), not here)."""
    try:
        head, body, sig = (key_text or "").strip().split(".")
    except ValueError:
        raise LicenseError("not a licence key (expected three parts)")
    if head != "AS1":
        raise LicenseError("unknown licence format")
    try:
        _public_key(public_b64).verify(_b64d(sig), _b64d(body))
    except Exception:  # noqa: BLE001 - any failure is "invalid"
        raise LicenseError("the signature does not match: the key was altered or is not from AeroSculpt")
    payload = json.loads(_b64d(body))
    if payload.get("v") != 1 or payload.get("plan") not in PLANS:
        raise LicenseError("unsupported licence")
    return payload


def status(key_text, now=None, public_b64=None):
    now = now or time.time()
    if not key_text:
        return {"installed": False, "valid": False, "plan": None, "why": "no licence key"}
    try:
        p = verify(key_text, public_b64)
    except LicenseError as e:
        return {"installed": True, "valid": False, "plan": None, "why": str(e)}
    grace_until = p["expires"] + GRACE_DAYS * 86400
    out = {"installed": True, "email": p["email"], "plan": p["plan"], "issued": p["issued"], "expires": p["expires"],
           "grace_until": grace_until, "id": p["id"], "valid": now < grace_until}
    if now >= grace_until:
        out["why"] = "the licence expired (and its grace period has passed): renew it from your account page"
    elif now >= p["expires"]:
        out["why"] = f"the licence expired; it keeps working for {int((grace_until - now) / 86400)} more day(s) — renew it from your account page"
        out["in_grace"] = True
    return out


def key_file():
    from .config import settings
    return settings.data_dir / "license.key"


def installed(now=None):
    f = key_file()
    return status(f.read_text(encoding="utf-8").strip() if f.exists() else "", now)


def install(key_text):
    p = verify(key_text)
    key_file().write_text(key_text.strip(), encoding="utf-8")
    return status(key_text)


def remove():
    f = key_file()
    if f.exists():
        f.unlink()
