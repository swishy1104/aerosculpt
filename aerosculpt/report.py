"""Engineering report of a finished job as a PDF (matplotlib, no extra dependencies).

Pages: summary (setup, conditions, headline numbers, grid convergence); force charts across the
envelope; convergence history; geometry (original vs optimized) and flow slices; for
optimizations the screening, search history, structure and mission; methods and limitations.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import textwrap
from pathlib import Path

import numpy as np

A4 = (8.27, 11.69)
INK = "#1f2328"
MUTED = "#6b7280"
C_BASE = "#2563eb"
C_BEST = "#ea580c"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.edgecolor": "#9ca3af",
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


def _dequant(b64, lo, hi):
    q = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).astype(float)
    out = lo + (hi - lo) * q / 254.0
    out[q == 255] = np.nan
    return out


def _page(plt, title, subtitle=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.955, title, fontsize=15, weight="bold", color=INK)
    if subtitle:
        fig.text(0.07, 0.935, subtitle, fontsize=9, color=MUTED)
    fig.text(0.07, 0.03, "AeroSculpt engineering report", fontsize=7, color=MUTED)
    return fig


def _table(fig, rect, rows, col_w=(0.45, 0.55), fs=8.5):
    ax = fig.add_axes(rect)
    ax.axis("off")
    y = 1.0
    step = 1.0 / max(len(rows), 1)
    for k, row in enumerate(rows):
        if row is None:
            y -= step * 0.5
            continue
        if isinstance(row, str):
            ax.text(0.0, y, row, fontsize=fs + 1, weight="bold", color=INK, va="top", transform=ax.transAxes)
        else:
            ax.text(0.0, y, str(row[0]), fontsize=fs, color=MUTED, va="top", transform=ax.transAxes)
            ax.text(col_w[0], y, str(row[1]), fontsize=fs, color=INK, va="top", transform=ax.transAxes)
        y -= step
    return ax


def _fmt(v, d=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{d}g}"
    return str(v)


def _silhouettes(fig, rect, meshes, title):
    """Side, top and front outlines of one or two meshes (STL bytes), stacked full width."""
    from matplotlib.collections import PolyCollection
    from .geometry.mesh import load_mesh
    x0, y0, w, h = rect
    boxes = [(x0, y0 + 0.62 * h, w, 0.34 * h), (x0, y0 + 0.24 * h, w, 0.34 * h), (x0 + 0.25 * w, y0, 0.5 * w, 0.2 * h)]
    loaded = [(label, load_mesh(data, "m.stl"), color) for label, data, color in meshes]
    for (name, a, b), box in zip((("side", 0, 2), ("top", 0, 1), ("front", 1, 2)), boxes):
        ax = fig.add_axes(box)
        for _, m, color in loaded:
            tri = m.triangles()
            step = max(1, len(tri) // 6000)
            ax.add_collection(PolyCollection(tri[::step][:, :, [a, b]], facecolors=color, edgecolors="none", alpha=0.35))
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name, fontsize=8, color=MUTED, loc="left")
        ax.tick_params(labelsize=6)
    fig.text(x0, y0 + h + 0.01, title, fontsize=10, weight="bold", color=INK)


def build_pdf(job_dir: Path, spec: dict, result: dict, project_name: str, out: Path, job_id: str = "", png_dir=None):
    """png_dir: also save every page as a PNG there (previews, checks)."""
    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    job_dir = Path(job_dir)
    kind = result.get("kind", "analyze")
    setup = result.get("setup", {})
    env = result.get("envelope") or spec.get("envelope", {})
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with PdfPages(out) as pdf:
        pages = []

        def save(fig):
            pdf.savefig(fig)
            if png_dir is not None:
                fig.savefig(Path(png_dir) / f"page{len(pages) + 1}.png", dpi=70)
            pages.append(1)

        info = pdf.infodict()
        info["Title"] = f"AeroSculpt report — {project_name}"
        info["Creator"] = "AeroSculpt"

        # ---- 1. summary --------------------------------------------------------------------
        fig = _page(plt, project_name, f"{'Optimization' if kind == 'optimize' else 'Analysis'} · job {job_id} · {when}")
        rows = ["Setup",
                ("Tunnel", spec.get("tunnel", "general").capitalize()),
                ("Method", setup.get("method") or "Lattice-Boltzmann LES (D3Q19), interpolated bounce-back walls"),
                ("Fidelity", setup.get("fidelity", spec.get("fidelity", "—"))),
                ("Grid", f"{setup['grid'][0]} x {setup['grid'][1]} x {setup['grid'][2]} = {setup['cells']:,} cells" if setup.get("grid") else "—"),
                ("Cell size", f"{setup['dx_m'] * 1000:.1f} mm" if setup.get("dx_m") else "—"),
                ("Body across (cells)", ", ".join(f"{c:g}" for c in setup.get("body_cells", [])) or "—"),
                ("Regime", ", ".join(setup.get("regimes", [])) or "—"),
                ("Road / wheels", ("moving road, %d wheel(s)" % setup.get("wheels", 0)) if setup.get("ground") else "free air"),
                None, "Flight conditions",
                ("Speeds (m/s)", ", ".join(f"{v:g}" for v in env.get("speeds", []))),
                ("Angles of attack (deg)", ", ".join(f"{a:g}" for a in env.get("aoas", []))),
                ("Altitude", f"{env.get('altitude', 0):g} m"),
                ("Reference area / length", f"{_fmt(setup.get('ref_area'))} m² / {_fmt(setup.get('ref_length'))} m")]
        base = (result.get("baseline") or {}).get("polar")
        if base:
            p0 = base[0]["speeds"][0]
            rows += [None, "Results (first condition)",
                     (f"Drag coefficient at {base[0]['aoa']:g}°, {p0['v']:g} m/s", _fmt(p0["cd"])),
                     ("Drag", f"{p0['drag_N']:.4g} N"), ("Lift", f"{p0['lift_N']:.4g} N (indicative)")]
        fx = result.get("fixed_lift")
        if fx and fx.get("baseline"):
            b0 = fx["baseline"][0]
            b1 = (fx.get("best") or fx["baseline"])[0]
            rows += [None, f"At the flight weight ({fx.get('basis')})",
                     (f"Drag at {b0['v']:g} m/s", f"{b0['drag_N']:.3f} N" + (f" → {b1['drag_N']:.3f} N" if kind == "optimize" else "")),
                     ("Lift-to-drag ratio", f"{b0['ld']:.1f}" + (f" → {b1['ld']:.1f}" if kind == "optimize" else ""))]
        if kind == "optimize":
            st = result.get("strategy", {})
            rows += [None, "Optimization",
                     ("Goal", result.get("objective_label") or result.get("objective")),
                     ("Change (confirmed)", f"{-result.get('improvement', 0) * 100:+.1f} %" if st.get("reliable") else "no reliable improvement"),
                     ("Measured solver error (2σ)", f"±{st.get('epsilon', 0) * 100:.1f} %"),
                     ("Designs explored / CFD runs", f"{st.get('designs_explored', '—')} / {result.get('cfd_runs', '—')}"),
                     ("Explored with / confirmed with", f"{st.get('explore', '—')} / {st.get('confirm_grid') or st.get('confirm', '—')}")]
        gs = result.get("grid_study")
        if gs:
            g = gs["cd"]
            rows += [None, "Grid convergence (Celik et al. 2008)",
                     ("Cd on 3 grids (coarse → fine)", " / ".join(f"{v:.4f}" for v in g["values"][::-1])),
                     ("Grid-independent estimate", f"{g['extrapolated']:.4f} (order {_fmt(g.get('order'), 3)})"),
                     ("GCI on the chosen grid", f"±{g['gci_fine'] * 100:.1f} %" + (" — " + g["note"] if g.get("note") else ""))]
        _table(fig, [0.07, 0.08, 0.86, 0.82], rows)
        save(fig)
        plt.close(fig)

        # ---- 2. forces across the envelope ------------------------------------------------------
        polars = [("original" if kind == "optimize" else "design", base, C_BASE)]
        if kind == "optimize" and (result.get("best") or {}).get("polar"):
            polars.append(("optimized", result["best"]["polar"], C_BEST))
        if base:
            fig = _page(plt, "Forces across the flight envelope", "Coefficients on the reference area; lift indicative (see methods)")
            axs = [fig.add_axes([0.1, 0.56, 0.36, 0.3]), fig.add_axes([0.58, 0.56, 0.36, 0.3]),
                   fig.add_axes([0.1, 0.13, 0.36, 0.3]), fig.add_axes([0.58, 0.13, 0.36, 0.3])]
            for label, pol, color in polars:
                a = [q["aoa"] for q in pol]
                axs[0].plot(a, [q["speeds"][0]["cd"] for q in pol], "o-", color=color, label=label)
                axs[1].plot(a, [q["cl"] for q in pol], "o-", color=color, label=label)
                axs[2].plot([q["speeds"][0]["cd"] for q in pol], [q["cl"] for q in pol], "o-", color=color, label=label)
                sp = pol[0]["speeds"]
                axs[3].plot([s["v"] for s in sp], [s["drag_N"] for s in sp], "o-", color=color, label=label)
            for ax, (t, xl, yl) in zip(axs, (("Drag coefficient", "angle of attack (°)", "Cd"),
                                              ("Lift coefficient", "angle of attack (°)", "CL"),
                                              ("Drag polar", "Cd", "CL"), ("Drag force", "airspeed (m/s)", "N"))):
                ax.set_title(t, loc="left")
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, fontsize=7)
            save(fig)
            plt.close(fig)

        # ---- 3. convergence and flow ----------------------------------------------------------------
        fl = (result.get("fields") or {}).get("baseline") or []
        if fl and (job_dir / fl[0]["file"]).exists():
            fj = json.loads((job_dir / fl[0]["file"]).read_text())
            fig = _page(plt, "Convergence and flow", f"Angle of attack {fl[0]['aoa']:g}°")
            ax = fig.add_axes([0.1, 0.67, 0.84, 0.2])
            conv = fj.get("convergence", {})
            ax.plot(conv.get("step", []), conv.get("cd", []), color=C_BASE, lw=0.8)
            ax.set_title("Drag coefficient during the run (tunnel, uncorrected for friction scaling)", loc="left")
            ax.set_xlabel("time step")
            ax.grid(alpha=0.25)
            sl = fj.get("slices", {})
            for k, (name, label) in enumerate((("xz", "Speed / freestream, vertical centre plane"), ("xy", "Speed / freestream, horizontal plane"))):
                s = sl.get(name)
                if not s:
                    continue
                arr = _dequant(s["speed"], *s["speed_range"]).reshape(s["shape"])
                ax = fig.add_axes([0.1, 0.36 - k * 0.27, 0.84, 0.22])
                im = ax.imshow(arr.T, origin="lower", extent=s["extent"], cmap="viridis", vmin=0, vmax=1.4, aspect="equal")
                ax.set_title(label, loc="left")
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
            save(fig)
            plt.close(fig)

        # ---- 4. geometry -----------------------------------------------------------------------------
        b_stl = job_dir / "baseline_body.stl"
        o_stl = job_dir / "best_body.stl"
        if b_stl.exists():
            fig = _page(plt, "Geometry", "Original (blue)" + (" and optimized (orange)" if o_stl.exists() else ""))
            meshes = [("original", b_stl.read_bytes(), C_BASE)]
            if o_stl.exists() and kind == "optimize":
                meshes.append(("optimized", o_stl.read_bytes(), C_BEST))
            _silhouettes(fig, [0.07, 0.45, 0.86, 0.43], meshes, "Views (m)")
            if kind == "optimize":
                changes = []
                for v in result.get("variables", []) or []:
                    b = (result.get("template") or {}).get("base_values", {}).get(v.get("key")) if result.get("template") else v.get("base")
                    x = (result.get("best_values") or {}).get(v.get("key"))
                    if isinstance(b, (int, float)) and isinstance(x, (int, float)) and abs(x - b) > 1e-9:
                        changes.append((v.get("label", v.get("key")), f"{b:.4g} → {x:.4g}"))
                if changes:
                    _table(fig, [0.07, 0.07, 0.86, 0.32], ["What changed"] + changes[:16])
            save(fig)
            plt.close(fig)

        # ---- 5. optimization detail ----------------------------------------------------------------------
        if kind == "optimize":
            st = result.get("strategy", {})
            hist = result.get("history", [])
            fig = _page(plt, "How the optimizer spent its runs")
            rank = st.get("ranking") or []
            if rank:
                ax = fig.add_axes([0.35, 0.58, 0.58, 0.3])
                names = [r_.get("label", r_.get("key")) for r_ in rank][:14]
                eff = [100 * (r_.get("effect") or 0.0) for r_ in rank][:14]
                ax.barh(range(len(names)), eff, color=[C_BEST if e < 0 else "#9ca3af" for e in eff])
                ax.set_yticks(range(len(names)), names, fontsize=7)
                ax.invert_yaxis()
                eps = 100 * (st.get("noise") or {}).get("eps", 0) if isinstance(st.get("noise"), dict) else 0
                if eps:
                    for x in (-eps, eps):
                        ax.axvline(x, color="#ef4444", lw=0.8, ls="--")
                ax.set_xlabel("effect on the objective (%)")
                ax.set_title("Screening: each parameter moved once", loc="left")
            ys = [h.get("objective") for h in hist if h.get("objective") is not None]
            if ys:
                ax = fig.add_axes([0.1, 0.12, 0.84, 0.33])
                xs = [h["i"] for h in hist if h.get("objective") is not None]
                ax.plot(xs, ys, "o", ms=3, color=C_BASE, alpha=0.7)
                ax.axhline(1.0, color=MUTED, lw=0.6)
                ax.set_xlabel("design #")
                ax.set_ylabel("objective (1 = original)")
                ax.set_title("Every design evaluated", loc="left")
                ax.grid(alpha=0.25)
            save(fig)
            plt.close(fig)

            if result.get("structure") or result.get("mission"):
                fig = _page(plt, "Structure and mission", "Preliminary sizing (beam model) and electric-flight performance")
                rows = []
                stc = result.get("structure")
                if stc:
                    s0, s1 = stc["base"], stc["best"]
                    rows += ["Structure", ("Material / load factor", f"{s0.get('material')} / n = {s0.get('load_factor')} × {s0.get('safety_factor')}")]
                    rows += [("Wing + tail mass", f"{s0['mass_kg'] * 1000:.0f} g → {s1['mass_kg'] * 1000:.0f} g")]
                    w0, w1 = s0["surfaces"][0], s1["surfaces"][0]
                    rows += [("Main spar", f"{w0['spar_kg'] * 1000:.0f} g ({w0['sized_by']}) → {w1['spar_kg'] * 1000:.0f} g ({w1['sized_by']})"),
                             ("Tip deflection at limit load", f"{w0['tip_deflection_frac']:.0%} → {w1['tip_deflection_frac']:.0%} of the semi-span"),
                             ("Root bending moment (ultimate)", f"{w0['root_moment_Nm']:.1f} → {w1['root_moment_Nm']:.1f} N·m")]
                ms = result.get("mission")
                if ms:
                    m0, m1 = ms["base"], ms.get("best", ms["base"])
                    rows += [None, "Mission (battery %.0f Wh, efficiency %.0f %%)" % (m0["inputs"]["battery_wh"], 100 * m0["inputs"]["efficiency"]),
                             ("Stall speed", f"{m0['stall_speed']:.1f} → {m1['stall_speed']:.1f} m/s"),
                             ("Endurance (best-endurance speed)", f"{m0['best_endurance']['endurance_min']:.0f} min at {m0['best_endurance']['v']:.1f} m/s → "
                                                                  f"{m1['best_endurance']['endurance_min']:.0f} min at {m1['best_endurance']['v']:.1f} m/s"),
                             ("Range (best-range speed)", f"{m0['best_range']['range_km']:.0f} km at {m0['best_range']['v']:.1f} m/s → "
                                                          f"{m1['best_range']['range_km']:.0f} km at {m1['best_range']['v']:.1f} m/s")]
                _table(fig, [0.07, 0.55, 0.86, 0.35], rows)
                if ms:
                    ax = fig.add_axes([0.1, 0.1, 0.84, 0.36])
                    for label, m, color in (("original", ms["base"], C_BASE), ("optimized", ms.get("best"), C_BEST)):
                        if m:
                            ax.plot([c["v"] for c in m["curve"]], [c["power_W"] for c in m["curve"]], color=color, label=label)
                    ax.set_xlabel("airspeed (m/s)")
                    ax.set_ylabel("power required (W)")
                    ax.set_title("Power required at the flight weight", loc="left")
                    ax.grid(alpha=0.25)
                    ax.legend(frameon=False)
                save(fig)
                plt.close(fig)

        # ---- 6. methods and limitations -----------------------------------------------------------------
        fig = _page(plt, "Methods and limitations")
        text = METHODS_TEXT
        y = 0.9
        for para in text.strip().split("\n\n"):
            for line in textwrap.wrap(" ".join(para.split()), 105):
                fig.text(0.07, y, line, fontsize=8.2, color=INK, va="top")
                y -= 0.016
            y -= 0.012
        save(fig)
        plt.close(fig)
    return out


METHODS_TEXT = """
Flow solver. Three-dimensional lattice-Boltzmann large-eddy simulation (D3Q19 lattice, regularized
collision, Smagorinsky sub-grid model, bulk-viscosity acoustic damping) on a uniform grid sized
around the body, with interpolated (Bouzidi) bounce-back walls on the true triangle surface and
momentum-exchange forces. Forces are corrected for tunnel blockage (Maskell, theta 2.5).

Reynolds number. A grid carries a limited Reynolds number. Above it the tunnel runs at its cap
("hybrid" regime): pressure drag, lift and moments come from the simulation and skin friction is
re-scaled to the flight Reynolds number. When the flight boundary layer is turbulent, a log-law
wall model makes the wall shear the grid feels match the flight Reynolds number, which keeps
boundary layers thin and attached as in the real flow.

Accuracy. Measured against reference data on the validation page: sphere drag within 1-7 %, a
face-on cube within about 13 %, the Ahmed car body on a road within 6-14 % with the nested zoom grid
(about 2x too high on the standard grid alone). Streamlined bodies are still over-predicted (about
2.5x): treat their absolute drag as an upper estimate and compare designs on the same grid. Lift on thin wings from the CFD reads 50-66 % low and is indicative only; winged designs
are optimized with a vortex-lattice wing model with section data instead (validated separately).

Optimization. Each design is scored against the original at every (angle, speed) condition. The
solver's own run-to-run error is measured by re-running the original shifted by part of a grid
cell; a change is accepted only when it beats twice that error, and the best candidates are
re-run on a different (finer) grid at both shifts and must win at every one.

Not modelled. Transition location (fully turbulent above Re 5e5), propeller slipstream, aeroelastic
deformation, and, for cars, cooling flow and the underbody detail of the real vehicle.
"""
