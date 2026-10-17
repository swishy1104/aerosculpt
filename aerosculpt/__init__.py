"""AeroSculpt: wind tunnels, a rocket flight model and a design optimizer for airframes."""
from __future__ import annotations

from pathlib import Path

__version__ = "1.1.0"


def build_id() -> str:
    """The solver build: the git commit written to aerosculpt/BUILD by scripts/build_desktop.py
    (or found from the checkout), else "dev". Stamped on every result and report."""
    here = Path(__file__).resolve().parent
    f = here / "BUILD"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()[:12] or "dev"
    head = here.parent / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (here.parent / ".git" / ref[5:].strip()).read_text().strip()[:12]
        return ref[:12]
    except OSError:
        return "dev"
