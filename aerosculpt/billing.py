"""Subscriptions: Stripe Checkout + Billing Portal + webhooks, or a local demo mode.

Stripe is called over its REST API with `requests` (no SDK dependency). The webhook is the
source of truth for plan state; checkout success pages only display what the webhook wrote.

Demo mode (no STRIPE_SECRET_KEY) grants plans locally without charging anything so the full
product can be exercised in development. It is labelled as such everywhere in the UI.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import requests

from .config import settings
from .db import audit
from .plans import PLANS

API = "https://api.stripe.com/v1"
ACTIVE = ("active", "trialing", "past_due")


class BillingError(Exception):
    pass


# ---- Stripe REST ----------------------------------------------------------------------
def _flatten(params, prefix=""):
    out = []
    if isinstance(params, dict):
        for k, v in params.items():
            key = f"{prefix}[{k}]" if prefix else k
            out += _flatten(v, key)
    elif isinstance(params, (list, tuple)):
        for i, v in enumerate(params):
            out += _flatten(v, f"{prefix}[{i}]")
    elif isinstance(params, bool):
        out.append((prefix, "true" if params else "false"))
    elif params is not None:
        out.append((prefix, str(params)))
    return out


def stripe(method, path, params=None, idempotency_key=None):
    if not settings.stripe_secret_key:
        raise BillingError("Stripe is not configured")
    headers = {}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    r = requests.request(method, API + path, auth=(settings.stripe_secret_key, ""),
                         data=_flatten(params or {}) if method != "GET" else None,
                         params=_flatten(params or {}) if method == "GET" else None,
                         headers=headers, timeout=20)
    try:
        body = r.json()
    except ValueError:
        raise BillingError(f"Stripe returned HTTP {r.status_code}")
    if r.status_code >= 400:
        msg = body.get("error", {}).get("message", f"HTTP {r.status_code}")
        raise BillingError(f"Stripe: {msg}")
    return body


def price_for(plan_key):
    return {"pro": settings.stripe_price_pro, "team": settings.stripe_price_team}.get(plan_key) or None


def plan_for_price(price_id):
    if price_id and price_id == settings.stripe_price_pro:
        return "pro"
    if price_id and price_id == settings.stripe_price_team:
        return "team"
    return None


def trial_eligible(user, plan_key):
    return bool(PLANS[plan_key].get("trial")) and not user["trial_used"]


def ensure_customer(con, user):
    if user["stripe_customer_id"]:
        return user["stripe_customer_id"]
    c = stripe("POST", "/customers", {"email": user["email"], "name": user["name"] or None,
                                       "metadata": {"user_id": user["id"]}},
               idempotency_key=f"customer-{user['id']}")
    con.execute("UPDATE users SET stripe_customer_id=? WHERE id=?", (c["id"], user["id"]))
    return c["id"]


def create_checkout(con, user, plan_key):
    price = price_for(plan_key)
    if not price:
        raise BillingError(f"No Stripe price configured for the {plan_key} plan (set STRIPE_PRICE_{plan_key.upper()})")
    customer = ensure_customer(con, user)
    sub_data = {"metadata": {"user_id": user["id"], "plan": plan_key}}
    if trial_eligible(user, plan_key):
        sub_data["trial_period_days"] = settings.trial_days
    s = stripe("POST", "/checkout/sessions", {
        "mode": "subscription",
        "customer": customer,
        "client_reference_id": str(user["id"]),
        "line_items": [{"price": price, "quantity": 1}],
        "subscription_data": sub_data,
        "allow_promotion_codes": True,
        "success_url": f"{settings.public_url}/app/billing?checkout=success",
        "cancel_url": f"{settings.public_url}/app/billing?checkout=cancel",
        "metadata": {"user_id": user["id"], "plan": plan_key},
    })
    audit(con, user["id"], "checkout_started", plan_key)
    return s["url"]


def create_portal(con, user):
    if not user["stripe_customer_id"]:
        raise BillingError("No billing account yet: choose a plan first.")
    s = stripe("POST", "/billing_portal/sessions", {"customer": user["stripe_customer_id"],
                                                    "return_url": f"{settings.public_url}/app/billing"})
    return s["url"]


def verify_webhook(payload: bytes, header: str, tolerance=300):
    """Validate a Stripe-Signature header (HMAC-SHA256 over 't.payload'). Returns the event."""
    if not settings.stripe_webhook_secret:
        raise BillingError("STRIPE_WEBHOOK_SECRET is not set")
    parts = {}
    for item in (header or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            parts.setdefault(k.strip(), []).append(v.strip())
    try:
        ts = int(parts["t"][0])
    except (KeyError, ValueError):
        raise BillingError("malformed signature header")
    if abs(time.time() - ts) > tolerance:
        raise BillingError("signature timestamp outside tolerance")
    expected = hmac.new(settings.stripe_webhook_secret.encode(), f"{ts}.".encode() + payload,
                        hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in parts.get("v1", [])):
        raise BillingError("signature mismatch")
    return json.loads(payload)


def _period_end(sub):
    if sub.get("current_period_end"):
        return sub["current_period_end"]
    items = (sub.get("items") or {}).get("data") or []
    ends = [it.get("current_period_end") for it in items if it.get("current_period_end")]
    return max(ends) if ends else None


def apply_subscription(con, sub):
    """Write a Stripe subscription object onto its user. Returns the user id or None."""
    user = None
    uid = (sub.get("metadata") or {}).get("user_id")
    if uid:
        user = con.execute("SELECT * FROM users WHERE id=?", (int(uid),)).fetchone()
    if user is None and sub.get("customer"):
        user = con.execute("SELECT * FROM users WHERE stripe_customer_id=?", (sub["customer"],)).fetchone()
    if user is None:
        return None
    items = (sub.get("items") or {}).get("data") or []
    plan_key = None
    for it in items:
        plan_key = plan_for_price((it.get("price") or {}).get("id")) or plan_key
    plan_key = plan_key or (sub.get("metadata") or {}).get("plan")
    status = sub.get("status", "canceled")
    if status not in ACTIVE or plan_key not in PLANS:
        plan_key, plan_status = "lite", "active"
    else:
        plan_status = status
    trial_end = sub.get("trial_end")
    con.execute(
        """UPDATE users SET plan=?, plan_status=?, stripe_subscription_id=?, stripe_customer_id=COALESCE(stripe_customer_id, ?),
           current_period_end=?, trial_ends_at=?, cancel_at_period_end=?, billing_source='stripe',
           trial_used = CASE WHEN ? IS NOT NULL THEN 1 ELSE trial_used END WHERE id=?""",
        (plan_key, plan_status, sub.get("id") if status != "canceled" else None, sub.get("customer"),
         _period_end(sub), trial_end, 1 if sub.get("cancel_at_period_end") else 0, trial_end, user["id"]))
    audit(con, user["id"], "subscription_sync", f"{plan_key}/{status}")
    return user["id"]


def handle_event(con, event):
    """Process one verified webhook event idempotently."""
    eid = event.get("id")
    if eid:
        seen = con.execute("SELECT 1 FROM stripe_events WHERE id=?", (eid,)).fetchone()
        if seen:
            return "duplicate"
        con.execute("INSERT INTO stripe_events (id, type, received_at) VALUES (?, ?, ?)",
                    (eid, event.get("type", ""), time.time()))
    et = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}
    if et == "checkout.session.completed":
        uid = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("user_id")
        if uid and obj.get("customer"):
            con.execute("UPDATE users SET stripe_customer_id=? WHERE id=?", (obj["customer"], int(uid)))
        if obj.get("subscription"):
            sub = stripe("GET", f"/subscriptions/{obj['subscription']}")
            sub.setdefault("metadata", {}).setdefault("user_id", uid)
            apply_subscription(con, sub)
        return "ok"
    if et.startswith("customer.subscription."):
        apply_subscription(con, obj)
        return "ok"
    return "ignored"


# ---- demo mode --------------------------------------------------------------------------
def demo_checkout(con, user, plan_key):
    now = time.time()
    if trial_eligible(user, plan_key):
        con.execute("""UPDATE users SET plan=?, plan_status='trialing', trial_ends_at=?, trial_used=1,
                       current_period_end=?, cancel_at_period_end=0, billing_source='demo' WHERE id=?""",
                    (plan_key, now + settings.trial_days * 86400, now + settings.trial_days * 86400, user["id"]))
    else:
        con.execute("""UPDATE users SET plan=?, plan_status='active', trial_ends_at=NULL,
                       current_period_end=?, cancel_at_period_end=0, billing_source='demo' WHERE id=?""",
                    (plan_key, now + 30 * 86400, user["id"]))
    audit(con, user["id"], "demo_checkout", plan_key)


def demo_cancel(con, user):
    con.execute("UPDATE users SET cancel_at_period_end=1 WHERE id=?", (user["id"],))
    audit(con, user["id"], "demo_cancel", user["plan"])


def demo_resume(con, user):
    con.execute("UPDATE users SET cancel_at_period_end=0 WHERE id=?", (user["id"],))


def refresh_demo(con, user):
    """Expire demo trials/periods. Stripe-managed users are updated by webhooks instead."""
    if user["billing_source"] != "demo" or user["plan"] == "lite":
        return False
    now = time.time()
    end = user["current_period_end"] or 0
    if end and end < now:
        if user["cancel_at_period_end"] or user["plan_status"] == "trialing":
            con.execute("""UPDATE users SET plan='lite', plan_status='active', trial_ends_at=NULL,
                           current_period_end=NULL, cancel_at_period_end=0 WHERE id=?""", (user["id"],))
        else:
            con.execute("UPDATE users SET current_period_end=? WHERE id=?", (end + 30 * 86400, user["id"]))
        return True
    return False
