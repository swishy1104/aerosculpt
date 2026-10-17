"""Solver validation suite: known bodies against published reference data.

    python run.py validate

writes aerosculpt/cfd/validation_results.json, which the public /validation page shows.
Run it after any change to the solver. An optimizer on an unvalidated solver finds shapes
that exploit solver error, so this is a release gate, not a nicety.
"""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from .analysis import FIDELITY, Conditions, Evaluator, Fidelity, analyze, cf_flat_plate, isa
from ..geometry.mesh import metrics, projected_area
from ..geometry.samples import ahmed_body, box, icosphere, streamlined_pod, wing

OUT = Path(__file__).with_name("validation_results.json")
NU = isa(0.0)["nu"]


def schiller_naumann(re):
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def hoerner_streamlined(mesh, re):
    """Reference band for a streamlined body: Hoerner form factor on wetted area with the
    transitional (lower) and fully turbulent (upper) flat-plate friction laws."""
    m = metrics(mesh)
    d = math.sqrt(4 * m["frontal_area"] / math.pi)
    k = d / m["length"]
    ff = 1 + 1.5 * k ** 1.5 + 7 * k ** 3
    swet_af = m["wetted_area"] / m["frontal_area"]
    lo = cf_flat_plate(re) * ff * swet_af
    hi = 0.455 / math.log10(re) ** 2.58 * ff * swet_af
    return lo, hi


def lift_slope_ref(ar, k):
    """Finite-wing lift slope per radian (Helmbold / DATCOM, unswept, incompressible);
    k = section lift slope / 2 pi."""
    return 2 * math.pi * ar / (2 + math.sqrt(ar * ar / (k * k) + 4))


def lift_slope_run(mesh, angles, re, fid):
    """CL_alpha (per rad) from runs at two angles of attack, and the setup of the last run."""
    from ..optimize.runner import wind_rotation
    lo, hi = mesh.bbox()
    mesh = mesh.transformed(t=-(lo + hi) / 2)
    rot = [mesh.transformed(R=wind_rotation(a)) for a in angles]
    elo = np.min([m.bbox()[0] for m in rot], axis=0)
    ehi = np.max([m.bbox()[1] for m in rot], axis=0)
    L = float(hi[0] - lo[0])
    span = float(hi[1] - lo[1])
    area = projected_area(mesh, 2)
    ev = Evaluator(elo, ehi, projected_area(mesh, 0), L, area, area / span, fid, planform_area=area)
    runs = [ev.run(m, min(re, fid.re_cap), key=(a, 0), want_fields=False) for a, m in zip(angles, rot)]
    slope = (runs[1]["cl"] - runs[0]["cl"]) / math.radians(angles[1] - angles[0])
    return slope, runs[-1], {**ev.info(), "regime": runs[-1]["regime"]}


def cases():
    sph = icosphere(0.5, 4)
    yield {"id": "sphere-re100", "body": "Sphere", "mesh": sph, "re": 100, "fid": Fidelity("validation-400k", 400_000, 1e3),
           "ref": schiller_naumann(100), "ref_source": "Schiller–Naumann correlation (±5% vs. experiment)"}
    yield {"id": "sphere-re200", "body": "Sphere", "mesh": sph, "re": 200, "fid": Fidelity("validation-400k", 400_000, 1e3),
           "ref": schiller_naumann(200), "ref_source": "Schiller–Naumann correlation"}
    yield {"id": "sphere-re300", "body": "Sphere", "mesh": sph, "re": 300, "fid": Fidelity("validation-400k", 400_000, 1e3),
           "ref": schiller_naumann(300), "ref_source": "Schiller–Naumann correlation"}
    yield {"id": "sphere-re300-fine", "body": "Sphere (refined grid)", "mesh": sph, "re": 300,
           "fid": Fidelity("validation-1.6M", 1_600_000, 1e3), "ref": schiller_naumann(300),
           "ref_source": "Schiller–Naumann correlation"}
    for f in ("draft", "standard"):
        yield {"id": f"sphere-re1e5-{f}", "body": "Sphere, subcritical", "mesh": sph, "re": 1e5, "fid": FIDELITY[f],
               "ref": 0.47, "ref_source": "Standard sphere drag curve, subcritical plateau (Achenbach 1972)"}
    cube = box((1, 1, 1), 12)
    for f in ("draft", "standard"):
        yield {"id": f"cube-{f}", "body": "Cube, face-on", "mesh": cube, "re": 6.8e5, "fid": FIDELITY[f],
               "ref": 1.05, "ref_source": "Hoerner, Fluid-Dynamic Drag (1965): Re-independent sharp-edged cube"}
    pod = streamlined_pod(1.0, 0.2)
    lo, hi = hoerner_streamlined(pod, 2.0e6)
    for f in ("draft", "standard", "high"):
        yield {"id": f"pod-{f}", "body": "Streamlined body, fineness 5", "mesh": pod, "re": 2.0e6, "fid": FIDELITY[f],
               "ref": (lo + hi) / 2, "ref_band": [lo, hi],
               "ref_source": "Hoerner form factor on wetted area, transitional–turbulent friction"}
    # Car: the Ahmed body (Ahmed, Ramm & Faltin 1984, SAE 840300), 60 m/s, 50 mm above a road,
    # wall model on; plain and with the nested zoom grid around the body
    for slant, ref in ((25, 0.285), (35, 0.257)):
        m = ahmed_body(slant)
        for refine in (False, True):
            yield {"id": f"ahmed{slant}-standard{'-zoom' if refine else ''}", "body": f"Ahmed body, {slant}° slant, on a road"
                   + (" (zoom grid)" if refine else ""), "mesh": m, "re": 4.29e6, "fid": FIDELITY["standard"], "ref": ref,
                   "ground": {"z": float(m.bbox()[0][2] - 0.05), "wheels": []}, "refine": refine,
                   "ref_source": "Ahmed, Ramm & Faltin (1984), SAE 840300: drag at Re 4.29e6"}
    # Lift: rectangular NACA 0012 wing, aspect ratio 6, slope between 0 and 6 degrees.
    ar = 6.0
    w = wing(ar * 0.2, 0.2, 1.0, 0.0, "0012", n_span=30, n_af=60)
    band = [lift_slope_ref(ar, k) for k in (0.9, 1.0)]
    for f in ("draft", "standard"):
        yield {"id": f"wing-ar6-{f}", "kind": "lift_slope", "body": "Rectangular wing, NACA 0012, AR 6", "mesh": w,
               "re": 3.0e5, "fid": FIDELITY[f], "angles": (0.0, 6.0), "ref": sum(band) / 2, "ref_band": band,
               "ref_source": "Lifting-line (Helmbold/DATCOM) lift slope, 2D slope 0.9–1.0 × 2π"}


def _row(id_, group, body, condition, method, quantity, value, ref, source, band=None, seconds=0.0):
    err = (value - ref) / ref
    inside = band is not None and band[0] <= value <= band[1]
    status = "pass" if abs(err) <= 0.10 or inside else ("warn" if abs(err) <= 0.25 else "fail")
    return {"id": id_, "group": group, "body": body, "condition": condition, "method": method, "quantity": quantity,
            "value": value, "cd": value, "ref": ref, "ref_band": band, "ref_source": source, "error": err,
            "status": status, "seconds": seconds}


def supersonic_rows(verbose=False):
    """Supersonic tunnel (surface methods) against measurements and exact theory. Seconds."""
    from ..geometry.samples import box as box_mesh, icosphere as sphere_mesh, revolve
    from ..hyper.atmosphere import us76
    from ..hyper.gasdyn import cone_table
    from ..hyper.panel import Surface
    rows = []
    t = time.time()
    S = Surface(sphere_mesh(0.5, 5), math.pi * 0.25, 1.0)
    for M in (4.0, 6.0, 10.0):
        cd = S.evaluate(M, 0.0, altitude=20000.0)["cd"]
        rows.append(_row(f"hs-sphere-m{M:g}", "supersonic", "Sphere", f"Mach {M:g}, 20 km", "Newtonian + base", "Cd", cd, 0.92,
                         "Free-flight sphere drag, hypersonic plateau ≈ 0.92 (Bailey & Hiatt 1972)", [0.874, 0.966]))
    xs = np.linspace(0, 1, 60)
    cone = revolve(xs, xs * math.tan(math.radians(10)), 96)
    lo, hi = cone.bbox()
    cone = cone.transformed(t=-(lo + hi) / 2)
    R = math.tan(math.radians(10))
    Sc = Surface(cone, math.pi * R * R, 2 * R)
    for M in (2.0, 4.0, 6.0):
        r0 = Sc.evaluate(M, 0.0, altitude=20000.0)
        th, cp, _p0, _tm = cone_table(M)
        exact = float(np.interp(math.radians(10), th, cp))
        rows.append(_row(f"hs-cone10-m{M:g}", "supersonic", "Sharp 10° cone, wave drag", f"Mach {M:g}", "tangent cone",
                         "Cd wave", r0["cd_wave"], exact, "Taylor–Maccoll exact conical flow (checks the surface integration)"))
    for M in (2.0, 4.0, 6.0):
        r0 = Sc.evaluate(M, 0.0, altitude=20000.0)
        r2 = Sc.evaluate(M, 2.0, altitude=20000.0)
        cna = (r2["cn"] - r0["cn"]) / math.radians(2.0)
        rows.append(_row(f"hs-cone10-cna-m{M:g}", "supersonic", "Sharp 10° cone, normal force", f"Mach {M:g}, α 0–2°", "tangent cone",
                         "CNα /rad", cna, 2.0, "Slender-body theory: CNα = 2 per radian on base area"))
    # curved noses: the shock-expansion march against Bonney's ogive data (as fitted by Fleeman)
    from ..geometry.templates import nose_profile, revolve as revolve_t
    for f in (3.0,):
        Rb = 0.05
        Ln = 2 * Rb * f
        xs, rs, _ = nose_profile("ogive", Ln, Rb, 0.0, 80)
        og = revolve_t(np.concatenate([xs, [Ln + 6 * Rb]]), np.concatenate([rs, [Rb]]), 96)
        lo, hi = og.bbox()
        So = Surface(og.transformed(t=-(lo + hi) / 2), math.pi * Rb * Rb, 2 * Rb)
        for M in (1.5, 2.0, 3.0, 4.0):
            ref = (1.586 + 1.834 / M ** 2) * math.atan(0.5 / f) ** 1.69
            rows.append(_row(f"hs-ogive{f:g}-m{M:g}", "supersonic", f"Sharp tangent ogive nose, fineness {f:g}", f"Mach {M:g}",
                             "shock-expansion", "Cd wave", So.evaluate(M, 0.0, altitude=10000.0)["cd_wave"], ref,
                             "Bonney (1950) ogive wave-drag data as fitted in Fleeman, Tactical Missile Design (2006)"))
    plate = Surface(box_mesh((1.0, 4.0, 0.01), 4), 4.0, 1.0)
    for M in (2.0, 3.0, 5.0):
        r0 = plate.evaluate(M, 0.0, altitude=20000.0)
        r2 = plate.evaluate(M, 2.0, altitude=20000.0)
        cla = (r2["cl"] - r0["cl"]) / math.radians(2.0)
        rows.append(_row(f"hs-plate-m{M:g}", "supersonic", "Thin flat plate (fin, wing)", f"Mach {M:g}, α 0–2°", "tangent wedge",
                         "CLα /rad", cla, 4 / math.sqrt(M * M - 1), "Ackeret linear supersonic theory, 4/√(M²−1)"))
    S2 = Surface(sphere_mesh(0.1, 5), math.pi * 0.01, 0.2)
    for M, h in ((6.0, 30000.0), (8.0, 40000.0), (10.0, 50000.0)):
        r = S2.evaluate(M, 0.0, altitude=h)
        a = us76(h)
        V = M * a["a"]
        T0 = a["T"] * (1 + 0.2 * M * M)
        sg = 1.7415e-4 * math.sqrt(a["rho"] / 0.1) * V ** 3 * (1 - 300.0 / T0)
        rows.append(_row(f"hs-stag-m{M:g}", "supersonic", "Stagnation heating, 0.1 m nose", f"Mach {M:g}, {h / 1000:g} km",
                         "Fay–Riddell", "q kW/m²", r["q_stag"] / 1e3, sg / 1e3, "Sutton–Graves correlation (cold wall 300 K)"))
    for r_ in rows:
        r_["seconds"] = (time.time() - t) / len(rows)
        if verbose:
            print(f"{r_['id']:22s} {r_['value']:.4f} ref {r_['ref']:.4f} err {r_['error'] * 100:+.1f}% [{r_['status']}]", flush=True)
    return rows


def rc_rows(verbose=False):
    """RC tunnel: propeller model against UIUC wind-tunnel data, actuator disk against momentum theory."""
    from ..geometry.samples import icosphere as sphere_mesh
    from ..geometry.mesh import projected_area
    from ..rc.actuator import disk_forcing
    from ..rc.propeller import Propeller, bemt
    from ..rc.reference import UIUC
    from .tunnel import FlowSettings, RunStats, Tunnel, plan_domain
    rows = []
    for key, ref in UIUC.items():
        t = time.time()
        prop = Propeller.from_inches(ref["d_in"], ref["p_in"], ref.get("blades", 2), family=ref.get("family", "thin-electric"))
        n = ref["rpm"] / 60.0
        res = [bemt(prop, J * n * prop.diameter, ref["rpm"]) for J, _, _ in ref["data"]]
        ct = float(np.mean([r["CT"] for r in res]))
        cp = float(np.mean([r["CP"] for r in res]))
        ct_ref = float(np.mean([d[1] for d in ref["data"]]))
        cp_ref = float(np.mean([d[2] for d in ref["data"]]))
        fitted = " (family fitted on this propeller)" if key in ("apce_10x7_6015", "da4022_9x6.75_2b_5030") else ""
        cond = f"{ref['rpm']} rpm, J {ref['data'][0][0]:.2f}–{ref['data'][-1][0]:.2f}"
        src = f"UIUC Propeller Data Site, vol. {ref.get('volume', 1)} ({ref['file']}){fitted}"
        rows.append(_row(f"rc-{key}-ct", "rc", f"{ref['name']}: thrust", cond, "blade-element momentum", "mean CT", ct, ct_ref, src,
                         seconds=time.time() - t))
        rows.append(_row(f"rc-{key}-cp", "rc", f"{ref['name']}: power", cond, "blade-element momentum", "mean CP", cp, cp_ref, src,
                         seconds=time.time() - t))
    # actuator disk in the lattice: far-wake speed against momentum theory
    t = time.time()
    air = isa(0.0)
    V = 10.0
    prop = Propeller.from_inches(10, 7)
    R = prop.diameter / 2
    perf = bemt(prop, V, 6000, air["rho"], air["mu"], air["a"])
    T = perf["thrust_N"]
    dom = plan_domain([-0.3 * R, -R, -R], [0.3 * R, R, R], 600_000, upstream=3 * prop.diameter,
                      downstream=6 * prop.diameter, lateral=3 * prop.diameter)
    u = 0.08
    tun = Tunnel(dom, FlowSettings(u_lat=u, re_sim=2e4, l_ref_cells=prop.diameter / dom.dx, smagorinsky=0.14))
    tun.set_geometry(sphere_mesh(0.15 * R, 2))
    idx, F = disk_forcing(dom, [0, 0, 0], [1, 0, 0], R, 0.15 * R, perf["r"], perf["dT_dr"], perf["dQ_dr"], 1.0,
                          T, perf["torque_Nm"], air["rho"], V, u)
    tun.set_forcing(idx, F)
    nx = dom.shape[0]
    tun.step(int(3 * nx / u), stats=RunStats())
    tun.start_averaging(4)
    tun.step(int(nx / u), stats=RunStats())
    rho, ux, _, _ = tun.mean_fields()
    c = dom.to_lattice(np.zeros(3))
    jj, kk = np.meshgrid(np.arange(dom.shape[1]), np.arange(dom.shape[2]), indexing="ij")
    rr = np.hypot(jj - c[1], kk - c[2]) * dom.dx / R
    i = int(round(c[0] + 1.5 * prop.diameter / dom.dx))
    wake = float(np.nanmean(ux[i][rr < 0.6]) / u)
    theory = math.sqrt(1 + T / (0.5 * air["rho"] * V * V * math.pi * R * R))
    rows.append(_row("rc-disk-wake", "rc", "Actuator disk slipstream (10x7 at 6000 rpm)", f"{V:g} m/s, 1.5 D downstream",
                     "lattice-Boltzmann + body force", "V wake / V", wake, theory, "Momentum theory: V_wake = V √(1 + C_T,disk)",
                     seconds=time.time() - t))
    if verbose:
        for r_ in rows:
            print(f"{r_['id']:26s} {r_['value']:.4f} ref {r_['ref']:.4f} err {r_['error'] * 100:+.1f}% [{r_['status']}]", flush=True)
    return rows


# NACA 0012, Re 6 million, transition fixed near the leading edge (Ladson 1988, NASA TM-4074),
# as tabulated by the NASA Turbulence Modeling Resource for comparisons with fully turbulent flow
LADSON_0012 = {"cl": {5.0: 0.536, 10.0: 1.057}, "cd": {0.0: 0.00809, 5.0: 0.00852, 10.0: 0.0119}}


def wing_rows(verbose=False):
    """The wing model the optimizer uses for lifting surfaces: vortex lattice against exact
    lifting-surface results and theory, NeuralFoil sections against wind-tunnel data. Seconds."""
    from ..aero import naca
    from ..aero.neuralfoil import naca_section
    from ..aero.vlm import VLM, LiftingSurface
    rows = []
    t = time.time()
    flat = {"family": "naca4", "m": 0.0, "p": 0.4, "t": 0.12}

    class Elliptic(LiftingSurface):
        def station(self, eta):
            le, c, tw, ec, en = super().station(eta)
            cc = self.root_chord * np.sqrt(np.maximum(1 - np.asarray(eta) ** 2, 0.0))
            le = le.copy()
            le[..., 0] = self.root_chord / 2 - cc / 2
            return le, cc, tw, ec, en

    e6 = Elliptic("ellipse", (0, 0, 0), 6.0, 4 / math.pi, 4 / math.pi, section=flat, ns=40)
    v = VLM([e6], 6.0, 4 / math.pi)
    r = v.solve(4.0)
    rows.append(_row("wm-ellipse-e", "wing", "Elliptic wing, AR 6", "α 4°", "vortex lattice", "span efficiency e",
                     r["CL"] ** 2 / (math.pi * 6 * r["CDi"]), 1.0, "Prandtl: an elliptic loading has e = 1 exactly"))
    circ = Elliptic("circle", (0, 0, 0), 1.0, 1.0, 1.0, section=flat, ns=60, nc=8)
    rows.append(_row("wm-circular-cla", "wing", "Circular wing", "small α", "vortex lattice", "CLα /rad",
                     VLM([circ], math.pi / 4, 1.0).lift_slope(), 1.790, "Kinner (1937) exact lifting-surface solution"))
    rect = LiftingSurface("rect", (0, 0, 0), 1.2, 0.2, 0.2, section={**flat}, ns=40)
    band = [lift_slope_ref(6.0, k) for k in (0.9, 1.0)]
    rows.append(_row("wm-rect6-cla", "wing", "Rectangular wing, NACA 0012, AR 6", "small α", "vortex lattice", "CLα /rad",
                     VLM([rect], 0.24, 0.2).lift_slope(), sum(band) / 2,
                     "Lifting-line (Helmbold/DATCOM) lift slope, 2D slope 0.9–1.0 × 2π (same reference as the CFD case)", band))
    w = LiftingSurface("w", (0, 0, 0), 20.0, 1.0, 1.0, section={"family": "naca4", "m": 0.02, "p": 0.4, "t": 0.12}, ns=30)
    rows.append(_row("wm-2412-alpha0", "wing", "NACA 2412 section, camber", "zero lift", "vortex lattice", "α₀ deg",
                     VLM([w], w.area(), 1.0).alpha_for_cl(0.0), -2.1,
                     "Abbott & von Doenhoff, Theory of Wing Sections (1959): measured zero-lift angle about −2.1°"))
    sec = naca_section("naca4", 0.0, 0.4, 0.12, 0.3)
    for a, ref in LADSON_0012["cl"].items():
        val = float(sec.polar([a], [6e6], xtr_upper=0.05, xtr_lower=0.05)["CL"][0])
        rows.append(_row(f"wm-0012-cl{a:g}", "wing", "NACA 0012 section, Re 6M, tripped", f"α {a:g}°", "NeuralFoil", "cl",
                         val, ref, "Ladson (1988), NASA TM-4074"))
    for a, ref in LADSON_0012["cd"].items():
        val = float(sec.polar([a], [6e6], xtr_upper=0.05, xtr_lower=0.05)["CD"][0])
        rows.append(_row(f"wm-0012-cd{a:g}", "wing", "NACA 0012 section, Re 6M, tripped", f"α {a:g}°", "NeuralFoil", "cd",
                         val, ref, "Ladson (1988), NASA TM-4074"))
    for r_ in rows:
        r_["seconds"] = (time.time() - t) / len(rows)
        if verbose:
            print(f"{r_['id']:22s} {r_['value']:.4f} ref {r_['ref']:.4f} err {r_['error'] * 100:+.1f}% [{r_['status']}]", flush=True)
    return rows


def run_groups(groups, verbose=False):
    """Re-run the RC, supersonic and/or wing-model cases and merge them into the stored results."""
    rows = []
    if "supersonic" in groups:
        rows += supersonic_rows(verbose)
    if "wing" in groups:
        rows += wing_rows(verbose)
    if "rc" in groups:
        rows += rc_rows(verbose)
    if "rocket" in groups:
        from ..rocket.validation import rocket_rows
        rows += rocket_rows(verbose)
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"cases": []}
    ids = {r["id"] for r in rows}
    doc["cases"] = [c for c in doc.get("cases", []) if c["id"] not in ids] + rows
    doc["generated_at"] = time.time()
    OUT.write_text(json.dumps(doc, indent=1))
    return doc


def run_suite(verbose=False, only=None):
    results = []
    t_all = time.time()
    for c in cases():
        if only and c["id"] not in only:
            continue
        m = c["mesh"]
        L = m.bbox()[1][0] - m.bbox()[0][0]
        speed = c["re"] * NU / L
        t = time.time()
        if c.get("kind") == "lift_slope":
            val, run, setup = lift_slope_run(m, c["angles"], c["re"], c["fid"])
            r = {**run, "setup": setup, "cd": run["cd_tunnel"]}
            quantity = "CLα /rad"
        else:
            r = analyze(m, Conditions(speed=speed), c["fid"], ground=c.get("ground"), refine=c.get("refine", False))
            val = r["cd"]
            quantity = None
        cd = r["cd"]
        err = (val - c["ref"]) / c["ref"]
        band = c.get("ref_band")
        inside = band is not None and band[0] <= val <= band[1]
        status = "pass" if abs(err) <= 0.10 or inside else ("warn" if abs(err) <= 0.25 else "fail")
        row = {
            "id": c["id"], "group": "general", "body": c["body"], "re": c["re"], "quantity": quantity, "value": val, "fidelity": c["fid"].name, "cells": r["setup"]["cells"],
            "cells_across": round(min(v for v in r["setup"]["body_cells"] if v > 0), 1),
            "regime": r["setup"]["regime"], "cd": cd, "cd_tunnel": r["cd_tunnel"], "cd_pressure": r["cd_pressure"],
            "cd_friction": r.get("cd_friction"), "cd_std": r["cd_std"], "converged": r["converged"],
            "wall_model": bool(r.get("wall_model")), "zoom": bool(r.get("zoom")), "ground": c.get("ground") is not None,
            "ref": c["ref"], "ref_band": band, "ref_source": c["ref_source"], "error": err, "status": status,
            "seconds": time.time() - t,
        }
        results.append(row)
        if verbose:
            print(f"{row['id']:22s} Re={row['re']:.3g} value={val:.4f} ref={c['ref']:.4f} err={err * 100:+.1f}% "
                  f"[{status}] {row['regime']} {row['seconds']:.0f}s", flush=True)
    doc = {"generated_at": time.time(), "machine": platform.processor() or platform.machine(),
           "python": platform.python_version(), "total_seconds": time.time() - t_all, "cases": results}
    if only and OUT.exists():
        old = json.loads(OUT.read_text())
        keep = [x for x in old.get("cases", []) if x["id"] not in {r["id"] for r in results}]
        doc["cases"] = keep + results
    OUT.write_text(json.dumps(doc, indent=1))
    if not only:
        doc = run_groups(("supersonic", "rc", "wing", "rocket"), verbose)
    return doc
