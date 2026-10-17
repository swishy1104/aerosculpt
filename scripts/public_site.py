"""Serve AeroSculpt from this computer at your own domain, account-free, through a named
Cloudflare Tunnel (a permanent https address; no ports opened on your router).

One-time setup (your Cloudflare account must manage the domain's DNS):

    python scripts/public_site.py --setup app.example.com

      1. opens the browser for  cloudflared tunnel login  (pick the domain)
      2. creates the tunnel "aerosculpt" (credentials land in ~/.cloudflared/<id>.json)
      3. routes app.example.com to it (a CNAME record in your Cloudflare DNS)
      4. writes tools/public_site.json with the hostname

Every launch afterwards (or double-click "Start public site.bat"):

    python scripts/public_site.py

Starts the tunnel and the app on port 8080 with PUBLIC_URL=https://app.example.com,
ANONYMOUS_ACCESS=1 (every visitor gets a private temporary workspace, no sign-in anywhere),
demo billing off, secure cookies. Add --owner you@example.com to be able to sign in at /login
for the admin page. Ctrl+C stops both. The site is up only while this window is open.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "tools" / "public_site.json"
PORT = int(os.environ.get("PUBLIC_PORT", "8080"))
TUNNEL = "aerosculpt"


def find_cloudflared():
    for p in (ROOT / "tools" / "cloudflared.exe", ROOT / "tools" / "cloudflared"):
        if p.exists():
            return str(p)
    return shutil.which("cloudflared")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def setup(exe, hostname):
    cred_dir = Path.home() / ".cloudflared"
    if not (cred_dir / "cert.pem").exists():
        print("Signing in to Cloudflare: a browser window opens; choose the domain the site will use.")
        run([exe, "tunnel", "login"])
    out = subprocess.run([exe, "tunnel", "list", "--output", "json"], capture_output=True, text=True)
    existing = [t for t in (json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else []) if t.get("name") == TUNNEL]
    if not existing:
        run([exe, "tunnel", "create", TUNNEL])
    run([exe, "tunnel", "route", "dns", "--overwrite-dns", TUNNEL, hostname])
    CONF.write_text(json.dumps({"hostname": hostname, "tunnel": TUNNEL}, indent=2), encoding="utf-8")
    print(f"\nDone: {hostname} points at this computer's tunnel. Start the site with  python scripts/public_site.py")


def serve(exe, hostname, owner):
    env = dict(os.environ, PUBLIC_URL=f"https://{hostname}", OPEN_ACCESS="0", PORT=str(PORT), HOST="127.0.0.1",
               ANONYMOUS_ACCESS="1", GUEST_DEMO="0", FORWARDED_ALLOW_IPS="127.0.0.1",
               ALLOWED_SIGNUP_EMAILS=owner or "nobody@invalid", OWNER_EMAILS=owner or "")
    env.setdefault("GUEST_HOURS", "72")
    tunnel = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "run", "--url", f"http://127.0.0.1:{PORT}", TUNNEL])
    (ROOT / "tools" / "public_link.txt").write_text(f"https://{hostname}
", encoding="utf-8")
    time.sleep(2)
    print("\n" + "=" * 64)
    print(f"  AeroSculpt is public at   https://{hostname}")
    print(f"  Account-free              every visitor gets a private workspace, kept {env['GUEST_HOURS']} h")
    print(f"  Owner sign-in at /login   {owner or 'none (add --owner you@example.com)'}")
    print("  Keep this window open; Ctrl+C stops the site.")
    print("=" * 64 + "\n", flush=True)
    app = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "serve"], cwd=str(ROOT), env=env)
    try:
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (app, tunnel):
            if p.poll() is None:
                p.terminate()


def main():
    args = sys.argv[1:]
    exe = find_cloudflared()
    if not exe:
        sys.exit("cloudflared not found: put cloudflared.exe in aerosculpt/tools/ or on PATH (https://github.com/cloudflare/cloudflared/releases).")
    owner = None
    if "--owner" in args:
        owner = args[args.index("--owner") + 1].strip().lower()
    if "--setup" in args:
        host = next((a for a in args[args.index("--setup") + 1:] if "." in a and "@" not in a), None)
        if not host:
            sys.exit("Give the hostname, e.g.  python scripts/public_site.py --setup app.example.com")
        return setup(exe, host.lower())
    if not CONF.exists():
        sys.exit("Not set up yet: run  python scripts/public_site.py --setup app.example.com")
    conf = json.loads(CONF.read_text(encoding="utf-8"))
    serve(exe, conf["hostname"], owner or conf.get("owner"))


if __name__ == "__main__":
    main()
