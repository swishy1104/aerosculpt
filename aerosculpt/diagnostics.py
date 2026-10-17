"""The diagnostic bundle: a zip of what support needs to see and nothing it does not — versions,
the machine's compute profile, GPU state, the app's non-secret settings, recent jobs with the
logs of those that failed, and crash files. No models, results or account data."""
from __future__ import annotations

import io
import json
import platform
import sys
import time
import zipfile

from . import __version__, build_id
from .config import settings
from .db import db


def about():
    from .cfd import gpu
    from .local import local_settings
    vers = {}
    for mod in ("numpy", "scipy", "numba", "llvmlite", "starlette", "uvicorn", "matplotlib", "skimage", "numba_cuda", "gmsh"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            vers[mod] = f"not available ({type(e).__name__})"
    try:
        compute = json.loads((settings.data_dir / "compute.json").read_text())
    except (OSError, ValueError):
        compute = None
    gpu_state = dict(gpu._STATE)
    gpu_state.pop("trace", None)
    return {"app": __version__, "build": build_id(), "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "frozen": bool(getattr(sys, "frozen", False)), "versions": vers, "compute": compute, "gpu": gpu_state,
            "settings": {"open_access": settings.open_access, "billing_mode": settings.billing_mode, "workers": settings.workers,
                         "data_dir": str(settings.data_dir), "fusion_bridge": settings.fusion_bridge, "public_url": settings.public_url},
            "local_settings": local_settings(), "generated_at": time.time()}


def bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("about.json", json.dumps(about(), indent=1, default=str))
        with db() as con:
            rows = con.execute("SELECT id, kind, status, error, created_at, started_at, finished_at, cfd_runs, credits_used, spec, log "
                               "FROM jobs ORDER BY created_at DESC LIMIT 30").fetchall()
        jobs = []
        for r in rows:
            spec = json.loads(r["spec"] or "{}")
            jobs.append({"id": r["id"], "kind": r["kind"], "status": r["status"], "error": r["error"], "created_at": r["created_at"],
                         "seconds": (r["finished_at"] or 0) - (r["started_at"] or 0) if r["started_at"] else None, "cfd_runs": r["cfd_runs"],
                         "tunnel": spec.get("tunnel"), "fidelity": spec.get("fidelity"), "objective": (spec.get("optimize") or {}).get("objective")})
            if r["status"] == "failed" and r["log"]:
                z.writestr(f"job_logs/{r['id']}.txt", r["log"])
        z.writestr("jobs.json", json.dumps(jobs, indent=1))
        crashes = settings.data_dir / "crashes"
        if crashes.is_dir():
            for f in sorted(crashes.glob("*.txt"))[-20:]:
                z.writestr(f"crashes/{f.name}", f.read_text(encoding="utf-8", errors="replace"))
    return buf.getvalue()


def record_crash(job_id, spec, text):
    """A crash file for the bundle (when the local setting allows it)."""
    from .local import local_settings
    if not local_settings().get("crash_reports", True):
        return
    d = settings.data_dir / "crashes"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job_id}.txt").write_text(f"AeroSculpt {__version__} ({build_id()})\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n\nspec: "
                                     f"{json.dumps({k: v for k, v in (spec or {}).items() if k != 'rocket'}, default=str)[:4000]}\n\n{text}", encoding="utf-8")
