"""Update check for the desktop app: the latest GitHub release against this version, at most
once every few hours, only when the user has not switched it off. Nothing is installed by
itself: the app shows a notice with the download link."""
from __future__ import annotations

import json
import re
import time
import urllib.request

RELEASES_API = "https://api.github.com/repos/swishy1104/aerosculpt/releases/latest"
TTL_S = 6 * 3600
_cache = {"at": 0.0, "result": None}


def _tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3]) or (0,)


def check(current, timeout=6.0, force=False):
    now = time.time()
    if not force and _cache["result"] is not None and now - _cache["at"] < TTL_S:
        return _cache["result"]
    try:
        req = urllib.request.Request(RELEASES_API, headers={"User-Agent": f"AeroSculpt/{current}", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        latest = (d.get("tag_name") or "").lstrip("v")
        assets = [a for a in d.get("assets", []) if a.get("browser_download_url")]
        win = next((a["browser_download_url"] for a in assets if a["name"].lower().endswith((".exe", ".zip"))), None)
        out = {"current": current, "latest": latest, "newer": _tuple(latest) > _tuple(current), "url": win or d.get("html_url"),
               "notes": (d.get("body") or "")[:2000], "checked_at": now}
    except Exception as e:  # noqa: BLE001 - no network is not an error for the user
        out = {"current": current, "latest": None, "newer": False, "url": None, "error": type(e).__name__, "checked_at": now}
    _cache.update(at=now, result=out)
    return out
