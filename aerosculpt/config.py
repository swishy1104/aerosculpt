"""Runtime configuration, read once from the environment (and an optional .env file)."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _bool(name, default=False):
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    app_name = os.environ.get("APP_NAME", "AeroSculpt")
    public_url = os.environ.get("PUBLIC_URL", "http://localhost:8000").rstrip("/")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    data_dir = Path(os.environ.get("DATA_DIR", str(ROOT / "data"))).resolve()
    secret_key = os.environ.get("SECRET_KEY", "")
    secure_cookies = _bool("SECURE_COOKIES", public_url.startswith("https://"))
    session_days = int(os.environ.get("SESSION_DAYS", "30"))
    workers = int(os.environ.get("WORKERS", "1"))
    support_email = os.environ.get("SUPPORT_EMAIL", "support@example.com")

    # Stripe. Without a secret key the app runs billing in clearly-labelled demo mode.
    stripe_secret_key = os.environ.get("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    stripe_price_pro = os.environ.get("STRIPE_PRICE_PRO", "")
    trial_days = int(os.environ.get("TRIAL_DAYS", "14"))
    # demo billing is a development convenience: off by default on any https deployment
    allow_demo_billing = _bool("ALLOW_DEMO_BILLING", not public_url.startswith("https://"))
    forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")
    # local preview: no sign-in, one built-in user with every feature unlocked. On by default
    # only when the app is served from this machine; never enable it on a public deployment.
    open_access = _bool("OPEN_ACCESS", public_url.startswith(("http://localhost", "http://127.0.0.1")))
    # invite-only sign-up for a private deployment: comma-separated emails (empty = anyone may sign up)
    allowed_signup_emails = {e.strip().lower() for e in os.environ.get("ALLOWED_SIGNUP_EMAILS", "").split(",") if e.strip()}
    # the site owner(s): admin with the Pro plan, without billing (for a private deployment)
    owner_emails = {e.strip().lower() for e in os.environ.get("OWNER_EMAILS", "").split(",") if e.strip()}
    # public guest demo: "Try it without an account" makes a temporary account on the Guest plan
    # (a few credits, draft and standard fidelity, lowest queue priority), deleted after GUEST_HOURS
    guest_demo = _bool("GUEST_DEMO", False)
    # "Send to Fusion" through the FusionMCPBridge add-in: only where the app and Fusion share a
    # machine (local preview by default), never on a public deployment unless forced
    fusion_bridge = _bool("FUSION_BRIDGE", open_access)
    # the desktop app and local preview (local.py): only localhost may talk to the app; with a launch
    # token only the browser opened through the launch link may use it
    local_token = os.environ.get("LOCAL_TOKEN", "").strip()
    allowed_hosts = {h.strip().lower() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()}
    # what the built-in local user may use: a plan name, or "license" (the installed desktop licence key,
    # the free desktop tier without one)
    open_access_plan = os.environ.get("OPEN_ACCESS_PLAN", "pro").strip().lower()
    # desktop licence signing (license.py): PEM text or a path; default data/license_private.pem
    license_private_key = os.environ.get("LICENSE_PRIVATE_KEY", "").strip()
    check_updates = _bool("CHECK_UPDATES", False)
    guest_hours = float(os.environ.get("GUEST_HOURS", "24"))
    guest_max_active = int(os.environ.get("GUEST_MAX_ACTIVE", "60"))
    # account-free public use: every visitor gets a temporary account of their own (the Public plan)
    # the moment they open the app; no sign-in or sign-up anywhere. Kept GUEST_HOURS, then deleted.
    anonymous_access = _bool("ANONYMOUS_ACCESS", False)
    anon_max_active = int(os.environ.get("ANON_MAX_ACTIVE", "400"))
    # optional AI design assistant (aerosculpt/ai.py): any OpenAI-compatible chat endpoint, e.g. a
    # free local model through Ollama (http://127.0.0.1:11434/v1). Unset = off, nothing is sent.
    ai_base_url = os.environ.get("AI_BASE_URL", "").strip()
    ai_model = os.environ.get("AI_MODEL", "").strip()
    ai_api_key = os.environ.get("AI_API_KEY", "").strip()
    ai_timeout_s = float(os.environ.get("AI_TIMEOUT_S", "90"))

    # Email (password resets). Without SMTP the link is written to the server log.
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_from = os.environ.get("SMTP_FROM", "")

    @property
    def billing_mode(self):
        if self.stripe_secret_key:
            return "stripe"
        return "demo" if self.allow_demo_billing else "disabled"

    @property
    def db_path(self):
        return self.data_dir / "aerosculpt.sqlite3"


settings = Settings()


def ensure_dirs():
    for sub in ("", "projects", "jobs", "samples"):
        (settings.data_dir / sub).mkdir(parents=True, exist_ok=True)
    if not settings.secret_key:
        key_file = settings.data_dir / ".secret_key"
        if not key_file.exists():
            key_file.write_text(secrets.token_hex(32))
        settings.secret_key = key_file.read_text().strip()
