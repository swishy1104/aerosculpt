"""One-time Stripe setup: products, monthly prices, customer portal and webhook endpoint.

    STRIPE_SECRET_KEY=sk_test_... PUBLIC_URL=https://your.domain python scripts/stripe_setup.py

Use a test-mode key first (sk_test_...), check the whole flow, then repeat with the live key.
Prints the values to put in .env. Safe to re-run: it reuses products it created before.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerosculpt.billing import stripe  # noqa: E402
from aerosculpt.config import settings  # noqa: E402
from aerosculpt.plans import PLANS  # noqa: E402

EVENTS = ["checkout.session.completed", "customer.subscription.created", "customer.subscription.updated",
          "customer.subscription.deleted", "customer.subscription.trial_will_end", "invoice.payment_failed"]


def find_product(key):
    res = stripe("GET", "/products/search", {"query": f"metadata['aerosculpt_plan']:'{key}'"})
    return res["data"][0] if res.get("data") else None


def main():
    if not settings.stripe_secret_key:
        sys.exit("Set STRIPE_SECRET_KEY first.")
    mode = "TEST" if settings.stripe_secret_key.startswith("sk_test") else "LIVE"
    print(f"Stripe {mode} mode")
    out = {}
    for key in ("pro", "team"):
        plan = PLANS[key]
        prod = find_product(key) or stripe("POST", "/products", {
            "name": f"{settings.app_name} {plan['name']}", "description": plan["tagline"],
            "metadata": {"aerosculpt_plan": key}})
        prices = stripe("GET", "/prices", {"product": prod["id"], "active": True, "limit": 10})["data"]
        price = next((p for p in prices if p["unit_amount"] == plan["price_month"] * 100
                      and (p.get("recurring") or {}).get("interval") == "month"), None)
        if price is None:
            price = stripe("POST", "/prices", {"product": prod["id"], "currency": "usd",
                                               "unit_amount": plan["price_month"] * 100,
                                               "recurring": {"interval": "month"},
                                               "metadata": {"aerosculpt_plan": key}})
        out[f"STRIPE_PRICE_{key.upper()}"] = price["id"]
        print(f"  {plan['name']}: product {prod['id']}, price {price['id']} (${plan['price_month']}/month)")

    portal = stripe("POST", "/billing_portal/configurations", {
        "business_profile": {"headline": f"Manage your {settings.app_name} subscription"},
        "features": {"invoice_history": {"enabled": True}, "payment_method_update": {"enabled": True},
                     "subscription_cancel": {"enabled": True, "mode": "at_period_end"},
                     "customer_update": {"enabled": True, "allowed_updates": ["email", "address", "tax_id"]}}})
    print(f"  customer portal configuration {portal['id']}")

    url = os.environ.get("PUBLIC_URL", settings.public_url).rstrip("/") + "/api/billing/webhook"
    if url.startswith("http://localhost"):
        print("  PUBLIC_URL is localhost: skip the webhook and use `stripe listen --forward-to "
              "localhost:8000/api/billing/webhook` from the Stripe CLI instead.")
    else:
        wh = stripe("POST", "/webhook_endpoints", {"url": url, "enabled_events": EVENTS,
                                                   "description": f"{settings.app_name} subscriptions"})
        out["STRIPE_WEBHOOK_SECRET"] = wh["secret"]
        print(f"  webhook {wh['id']} -> {url}")
    print("\nAdd to .env:")
    for k, v in out.items():
        print(f"{k}={v}")
    print("ALLOW_DEMO_BILLING=0")


if __name__ == "__main__":
    main()
