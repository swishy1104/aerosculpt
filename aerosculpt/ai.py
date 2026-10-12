"""Optional AI design assistant.

A language model reads the designer's brief, a summary of the body and its latest wind-tunnel
result, and picks and sizes ideas from the generative menu (geometry/features.IDEAS). It never
writes geometry itself: each idea is built by the same code as the "Generate design ideas" button,
added as an optional feature, and kept only if the optimizer measures it beating the solver's error.

Any OpenAI-compatible chat endpoint works (set in the environment):
  free, on this computer  AI_BASE_URL=http://127.0.0.1:11434/v1  AI_MODEL=qwen3:4b   (Ollama)
  hosted free tiers       AI_BASE_URL=<provider>/v1  AI_MODEL=<model>  AI_API_KEY=<key>
With AI_BASE_URL unset the assistant is off and nothing is ever sent anywhere.
"""
from __future__ import annotations

import json
import re

from .config import settings

SYSTEM = """You are an aerodynamicist helping redesign a body for lower drag in a wind tunnel.
You may only choose from this menu of shape ideas (each is tested by CFD afterwards; you do not
need to be certain, but explain your reasoning briefly):
{menu}
Reply with JSON only, no prose outside it, in exactly this form:
{{"ideas": [{{"idea": "<menu name>", "why": "<one sentence>", "hints": {{"<hint>": <number>}}}}], "notes": "<one or two sentences>"}}
Choose 1 to 4 ideas. Hints are optional numbers within the stated ranges."""


def available():
    return bool(settings.ai_base_url and settings.ai_model)


def info():
    return {"available": available(), "model": settings.ai_model if available() else None,
            "local": available() and ("127.0.0.1" in settings.ai_base_url or "localhost" in settings.ai_base_url)}


def _menu():
    from .geometry.features import IDEAS
    return "\n".join(f"- {k}: {what}; hints: " + ", ".join(f"{h} ({rng})" for h, rng in hints.items())
                     for k, (what, hints) in IDEAS.items())


def _parse(text):
    """The first JSON object in the reply (models sometimes wrap it in prose or code fences)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("the assistant did not reply with JSON")
    data = json.loads(m.group(0))
    if not isinstance(data, dict) or not isinstance(data.get("ideas"), list):
        raise ValueError("the assistant's reply has no ideas list")
    return data


def ask(brief, summary, result=None, post=None):
    """-> {"ideas": [{"idea", "why", "hints"}], "notes"} (validated against the menu).
    post(url, json, headers, timeout) -> response with .json(), for tests."""
    import requests
    from .geometry.features import IDEAS
    if not available():
        raise RuntimeError("the AI assistant is not configured on this server")
    user = {"brief": (brief or "Lower the drag without changing what the body is for.")[:1500],
            "body": summary, "last_wind_tunnel_result": result or "not simulated yet"}
    body = {"model": settings.ai_model, "temperature": 0.3,
            "messages": [{"role": "system", "content": SYSTEM.format(menu=_menu())},
                         {"role": "user", "content": json.dumps(user)}]}
    headers = {"Content-Type": "application/json"}
    if settings.ai_api_key:
        headers["Authorization"] = f"Bearer {settings.ai_api_key}"
    url = settings.ai_base_url.rstrip("/") + "/chat/completions"
    r = (post or requests.post)(url, json=body, headers=headers, timeout=settings.ai_timeout_s)
    if getattr(r, "status_code", 200) >= 400:
        raise RuntimeError(f"the assistant answered with HTTP {r.status_code}")
    text = r.json()["choices"][0]["message"]["content"]
    data = _parse(text)
    ideas = []
    for it in data["ideas"][:4]:
        if not isinstance(it, dict) or it.get("idea") not in IDEAS:
            continue
        hints = {k: float(v) for k, v in (it.get("hints") or {}).items()
                 if k in IDEAS[it["idea"]][1] and isinstance(v, (int, float))}
        ideas.append({"idea": it["idea"], "why": str(it.get("why") or "")[:300], "hints": hints})
    return {"ideas": ideas, "notes": str(data.get("notes") or "")[:600]}
