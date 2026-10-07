"""Passwords, session tokens, API keys and a small in-process rate limiter."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from collections import defaultdict, deque

import bcrypt

from .config import settings

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,63}$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or "")) and len(email) <= 254


def password_problem(pw: str):
    if not pw or len(pw) < 10:
        return "Use at least 10 characters."
    if len(pw.encode("utf-8")) > 72:
        return "Use at most 72 bytes."
    if pw.lower() == pw and pw.isalpha():
        return "Mix in a number, symbol or capital letter."
    return None


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("ascii")


def check_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw((pw or "").encode("utf-8"), hashed.encode("ascii"))
    except ValueError:
        return False


# A hash to compare against when the email does not exist, so timing does not leak accounts.
DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt(rounds=12)).decode("ascii")


def new_token(nbytes=32) -> str:
    return secrets.token_urlsafe(nbytes)


def token_hash(token: str) -> str:
    return hmac.new(settings.secret_key.encode(), token.encode(), hashlib.sha256).hexdigest()


def new_api_key():
    raw = "as_" + secrets.token_urlsafe(30)
    return raw, raw[:10]


class RateLimiter:
    """Sliding-window limiter keyed by an arbitrary string (IP, email, ...)."""

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window = window_s
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self.hits) > 50_000:  # bound memory under abuse
                self.hits.clear()
            return True


auth_limiter = RateLimiter(limit=12, window_s=600)
upload_limiter = RateLimiter(limit=40, window_s=3600)
