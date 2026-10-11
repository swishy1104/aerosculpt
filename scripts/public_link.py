"""Serve AeroSculpt from this computer at a public https address (Cloudflare quick tunnel).

    python scripts/public_link.py you@example.com [other@example.com ...] [--demo]

Starts `cloudflared tunnel --url http://127.0.0.1:8080` (no Cloudflare account needed), waits for
the https://<random>.trycloudflare.com address it prints, then starts the app on port 8080 with
that address as PUBLIC_URL. On an https address sign-in is required, demo billing is off and
cookies are secure; only the listed emails may create accounts, and the first one is the owner
(admin, Team plan, no billing). --demo also turns on the guest demo: anyone with the link can
press "Try the live demo" and get a temporary account (Guest plan: 30 credits, draft and standard
fidelity, lowest queue priority, deleted after 24 h). The address changes every time
this is started, and the site is up only while this computer is on and this window is open.
Ctrl+C stops both.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("PUBLIC_PORT", "8080"))


def find_cloudflared():
    for p in (ROOT / "tools" / "cloudflared.exe", ROOT / "tools" / "cloudflared"):
        if p.exists():
            return str(p)
    return shutil.which("cloudflared")


def main():
    emails = [e.strip().lower() for e in sys.argv[1:] if "@" in e]
    demo = "--demo" in sys.argv[1:]
    if not emails:
        sys.exit("Give the email address(es) allowed to sign up, e.g.  python scripts/public_link.py you@example.com")
    exe = find_cloudflared()
    if not exe:
        sys.exit("cloudflared not found: put cloudflared.exe in aerosculpt/tools/ or on PATH.")
    tunnel = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    url = None
    t0 = time.time()
    for line in tunnel.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
        if time.time() - t0 > 60:
            break
    if not url:
        tunnel.kill()
        sys.exit("The tunnel did not start (no trycloudflare.com address within 60 s). Check the internet connection.")
    threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()   # keep draining its log
    env = dict(os.environ, PUBLIC_URL=url, OPEN_ACCESS="0", PORT=str(PORT), HOST="127.0.0.1",
               ALLOWED_SIGNUP_EMAILS=",".join(emails), OWNER_EMAILS=emails[0], FORWARDED_ALLOW_IPS="127.0.0.1",
               GUEST_DEMO="1" if demo else "0")
    print("\n" + "=" * 64)
    print(f"  AeroSculpt is public at   {url}")
    print(f"  Sign-up allowed for       {', '.join(emails)}")
    print(f"  Guest demo                {'ON: visitors can try it without an account' if demo else 'off (add --demo to allow visitors)'}")
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


if __name__ == "__main__":
    main()
