"""Draw the README's rocket figure from the app's own flight model (no screenshots to keep in
sync): the template rocket with its centre of pressure and gravity, the flight, the drag curve.

    python scripts/readme_images.py        -> docs/img/rocket-flight.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerosculpt.geometry import templates as T  # noqa: E402
from aerosculpt.rocket.model import RocketModel  # noqa: E402

BG, PANEL, INK, MUTED, BLUE, ORANGE, RED, GREEN = "#0b1220", "#111a2b", "#e6edf3", "#8b98a9", "#60a5fa", "#f59e0b", "#f87171", "#34d399"


def main():
    out = ROOT / "docs" / "img" / "rocket-flight.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev = RocketModel(T.defaults("rocket")).evaluate()
    fl, st, sil = ev["flight"], ev["stability"], ev["silhouette"]
    fig = plt.figure(figsize=(12, 6.4), dpi=150, facecolor=BG)
    fig.text(0.04, 0.94, "AeroSculpt · rocket flight model", color=INK, fontsize=15, weight="bold")
    fig.text(0.04, 0.905, f"{ev['motor']['manufacturer']} {ev['motor']['name']} · {ev['launch']['rail_m']:g} m rail · "
             f"{ev['launch']['wind_m_s']:g} m/s wind · masses estimated from the geometry", color=MUTED, fontsize=9)

    # the rocket with CP and CG
    ax = fig.add_axes([0.04, 0.56, 0.56, 0.3], facecolor=BG)
    body = np.array(sil["body"])
    x, r = body[:, 0], body[:, 1]
    ax.fill_between(x, -r, r, color="#1f2a3d", lw=0)
    ax.plot(x, r, color=MUTED, lw=1)
    ax.plot(x, -r, color=MUTED, lw=1)
    if sil["fin"]:
        f = np.array(sil["fin"])
        for sgn in (1, -1):
            ax.fill(f[:, 0], sgn * f[:, 1], color="#263450", ec=MUTED, lw=1)
    cp, cg0, cg1 = st["liftoff"]["x_cp"], st["liftoff"]["x_cg"], st["burnout"]["x_cg"]
    ax.plot([cg0], [0], "o", ms=11, color=BLUE, mec=BG, mew=1.5, zorder=5)
    ax.plot([cg1], [0], "o", ms=11, mfc="none", mec=BLUE, mew=2, zorder=5)
    ax.plot([cp], [0], "o", ms=11, color=RED, mec=BG, mew=1.5, zorder=6)
    ax.annotate("CG, full motor", (cg0, 0), (cg0 - 0.12, 0.17), color=BLUE, fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("CP (Barrowman)", (cp, 0), (cp + 0.1, -0.19), color=RED, fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.text(0.02, -0.24, f"static margin {st['liftoff']['margin']:.2f} calibres at liftoff, {st['max_speed']['margin']:.2f} at top speed  ·  "
            f"fins hold to {ev['flutter']['flutter_speed']:.0f} m/s against {fl['max_speed']:.0f} m/s", color=INK, fontsize=9, transform=ax.transData)
    ax.set_xlim(-0.05, sil["length"] + 0.08)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # KPI tiles
    tiles = [("Apogee", f"{fl['apogee_m']:.0f} m", f"in {fl['apogee_t']:.1f} s"), ("Top speed", f"{fl['max_speed']:.0f} m/s", f"Mach {fl['max_mach']:.2f}"),
             ("Off the rail", f"{fl['rail_exit_speed']:.1f} m/s", "stable from the start"), ("Ejection delay", f"{fl['optimal_delay']:.1f} s", "optimal"),
             ("Descent", f"{fl['descent_time']:.0f} s", f"drifts {abs(fl['drift_m']):.0f} m"), ("Liftoff mass", f"{fl['liftoff_mass'] * 1000:.0f} g", "motor included")]
    for i, (k, v, s) in enumerate(tiles):
        axt = fig.add_axes([0.63 + (i % 2) * 0.185, 0.72 - (i // 2) * 0.125, 0.17, 0.105], facecolor=PANEL)
        axt.set_xticks([])
        axt.set_yticks([])
        for sp in axt.spines.values():
            sp.set_edgecolor("#223049")
        axt.text(0.08, 0.72, k.upper(), color=MUTED, fontsize=7, transform=axt.transAxes)
        axt.text(0.08, 0.36, v, color=INK, fontsize=13, weight="bold", transform=axt.transAxes)
        axt.text(0.08, 0.1, s, color=MUTED, fontsize=7.5, transform=axt.transAxes)

    # altitude and speed
    tr = fl["trajectory"]
    t = [p["t"] for p in tr]
    ax1 = fig.add_axes([0.07, 0.1, 0.27, 0.31], facecolor=PANEL)
    ax1.plot(t, [p["z"] for p in tr], color=BLUE, lw=2)
    ax1.set_title("Altitude (m)", color=INK, fontsize=9, loc="left")
    ax2 = fig.add_axes([0.385, 0.1, 0.27, 0.31], facecolor=PANEL)
    ax2.plot(t, [p["v"] for p in tr], color=ORANGE, lw=2)
    ax2.set_title("Speed (m/s)", color=INK, fontsize=9, loc="left")
    dc = ev["drag_curve"]
    ax3 = fig.add_axes([0.70, 0.1, 0.27, 0.31], facecolor=PANEL)
    ax3.plot([q["mach"] for q in dc], [q["cd"] for q in dc], color=GREEN, lw=2, label="total")
    ax3.plot([q["mach"] for q in dc], [q["cd_friction"] for q in dc], color=MUTED, lw=1.2, ls="--", label="friction")
    ax3.plot([q["mach"] for q in dc], [q["cd_base"] for q in dc], color=RED, lw=1.2, ls=":", label="base")
    ax3.set_title("Drag coefficient vs Mach", color=INK, fontsize=9, loc="left")
    ax3.legend(frameon=False, fontsize=7, labelcolor=INK)
    for ax_ in (ax1, ax2, ax3):
        ax_.tick_params(colors=MUTED, labelsize=7)
        for sp in ax_.spines.values():
            sp.set_edgecolor("#223049")
        ax_.grid(color="#1c2740", lw=0.6)
        ax_.set_xlabel("time (s)" if ax_ is not ax3 else "Mach", color=MUTED, fontsize=8)
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
