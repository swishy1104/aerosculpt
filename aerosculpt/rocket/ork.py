"""Import OpenRocket (.ork: a zip holding rocket.ork XML, or the XML itself) and RockSim (.rkt
XML) designs into the rocket template: geometry, part masses, motor and parachute. Export the
drag table as CSV.

What maps: nose cone (shape family and length), the body tubes (merged; the largest diameter
sets the template), a final boat-tail transition, the largest fin set (elliptical and freeform
fins become a trapezoid of the same area and span), the launch lug, every part's mass (override
or material density x volume) and position, the parachute, the motor mount's motor.
Anything else is listed in the import notes.
"""
from __future__ import annotations

import csv
import io
import math
import zipfile
import xml.etree.ElementTree as ET

from .motors import find_builtin

NOSE_MAP = {"conical": "conical", "ogive": "ogive", "ellipsoid": "elliptical", "elliptical": "elliptical", "power": "power",
            "parabolic": "power", "haack": "haack"}


def _f(el, tag, default=None):
    x = el.find(tag)
    if x is None or x.text is None:
        return default
    t = x.text.strip()
    if t == "auto":
        return "auto"
    try:
        return float(t)
    except ValueError:
        return default


def read_ork_bytes(data: bytes):
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".ork")] or z.namelist()
            data = z.read(names[0])
    return data


class _Part:
    def __init__(self, kind, name, x0, length, r_fore, r_aft, mass, cg_x, extra=None):
        self.kind, self.name, self.x0, self.length = kind, name, x0, length
        self.r_fore, self.r_aft, self.mass, self.cg_x = r_fore, r_aft, mass, cg_x
        self.extra = extra or {}


def _density(el):
    m = el.find("material")
    if m is None:
        return 0.0, "bulk"
    try:
        return float(m.get("density", 0.0)), m.get("type", "bulk")
    except ValueError:
        return 0.0, "bulk"


def _override(el):
    om = _f(el, "overridemass")
    oc = _f(el, "overridecg")
    return (om if isinstance(om, float) else None), (oc if isinstance(oc, float) else None)


def _offset(el, parent_len, own_len):
    """Axial position of an inner component inside its parent (from the parent's top)."""
    p = el.find("axialoffset")
    if p is None:
        p = el.find("position")
    if p is None:
        return 0.0
    method = p.get("method") or p.get("type") or "top"
    try:
        v = float(p.text or 0.0)
    except ValueError:
        v = 0.0
    if method == "bottom":
        return parent_len - own_len + v
    if method == "middle":
        return 0.5 * (parent_len - own_len) + v
    if method == "absolute":
        return v
    return v


def parse_ork(data: bytes):
    """-> {"name", "template_values", "masses", "motor", "parachute", "notes", "parts"}"""
    xml = read_ork_bytes(data)
    root = ET.fromstring(xml)
    rocket = root.find(".//rocket")
    if rocket is None:
        raise ValueError("not an OpenRocket design (no <rocket>)")
    name = (rocket.findtext("name") or "OpenRocket design").strip()
    notes = []
    stages = rocket.findall("./subcomponents/stage")
    if len(stages) > 1:
        notes.append(f"{len(stages)} stages: only the first (sustainer) is imported")
    if not stages:
        raise ValueError("the design has no stage")
    stage = stages[0]
    parts = []
    masses = []
    fins = []
    lug = None
    chute = None
    motor = None
    x = 0.0
    prev_r = None
    subs = stage.find("subcomponents")
    ext = [c for c in (subs if subs is not None else []) if c.tag in ("nosecone", "bodytube", "transition")]
    for i, c in enumerate(ext):
        L = float(_f(c, "length", 0.0) or 0.0)
        thick = float(_f(c, "thickness", 0.0) or 0.0)
        dens, dtype = _density(c)
        om, oc = _override(c)
        if c.tag == "nosecone":
            r = _f(c, "aftradius", "auto")
            if r == "auto":
                nxt = ext[i + 1] if i + 1 < len(ext) else None
                r = float(_f(nxt, "radius", 0.0) or 0.0) if nxt is not None else 0.0
                if r == "auto" or not r:
                    r = 0.0
            r = float(r)
            shape = (c.findtext("shape") or "ogive").strip().lower()
            sp = _f(c, "shapeparameter", 1.0)
            lateral = math.pi * r * math.hypot(L, r)     # cone-ish lateral area
            filled = (c.findtext("filled") or "false").strip().lower() == "true"
            vol = math.pi * r * r * L / 3.0
            m = om if om is not None else (dens * (vol if filled else lateral * thick) if dtype == "bulk" else dens * lateral)
            parts.append(_Part("nose", c.findtext("name") or "Nose cone", x, L, 0.0, r, m, x + (oc if oc is not None else 0.5 * L),
                               {"shape": shape, "param": sp}))
            prev_r = r
        elif c.tag == "bodytube":
            r = _f(c, "radius", "auto")
            r = float(prev_r if r == "auto" or r is None else r)
            lateral = 2 * math.pi * r * L
            m = om if om is not None else (dens * lateral * thick if dtype == "bulk" else dens * lateral)
            parts.append(_Part("tube", c.findtext("name") or "Body tube", x, L, r, r, m, x + (oc if oc is not None else 0.5 * L)))
            prev_r = r
        else:
            rf = _f(c, "foreradius", "auto")
            ra = _f(c, "aftradius", "auto")
            rf = float(prev_r if rf == "auto" or rf is None else rf)
            ra = float(rf if ra == "auto" or ra is None else ra)
            lateral = math.pi * (rf + ra) * math.hypot(L, rf - ra)
            m = om if om is not None else (dens * lateral * thick if dtype == "bulk" else dens * lateral)
            parts.append(_Part("transition", c.findtext("name") or "Transition", x, L, rf, ra, m, x + (oc if oc is not None else 0.5 * L)))
            prev_r = ra
        # inner components
        inner_subs = c.find("subcomponents")
        for inner in (inner_subs if inner_subs is not None else []):
            tag = inner.tag
            Li = float(_f(inner, "length", 0.0) or 0.0)
            dens_i, dtype_i = _density(inner)
            om_i, oc_i = _override(inner)
            if tag in ("trapezoidfinset", "ellipticalfinset", "freeformfinset"):
                n = int(_f(inner, "fincount", 3) or 3)
                t = float(_f(inner, "thickness", 0.003) or 0.003)
                if tag == "trapezoidfinset":
                    cr, ct = float(_f(inner, "rootchord", 0.05) or 0.05), float(_f(inner, "tipchord", 0.0) or 0.0)
                    s = float(_f(inner, "height", 0.04) or 0.04)
                    sw = float(_f(inner, "sweeplength", 0.0) or 0.0)
                    shape = "trapezoid"
                    area = 0.5 * s * (cr + ct)
                elif tag == "ellipticalfinset":
                    cr, s = float(_f(inner, "rootchord", 0.05) or 0.05), float(_f(inner, "height", 0.04) or 0.04)
                    ct, sw, shape = 0.0, 0.0, "elliptical"
                    area = math.pi * cr * s / 4
                else:
                    pts = [(float(p.get("x")), float(p.get("y"))) for p in inner.iter("point")]
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    area = 0.5 * abs(sum(xs[k] * ys[(k + 1) % len(pts)] - xs[(k + 1) % len(pts)] * ys[k] for k in range(len(pts)))) if len(pts) >= 3 else 0.0
                    s = max(ys) - min(ys) if ys else 0.04
                    cr = max(xs) - min(xs) if xs else 0.05
                    ct = max(0.0, 2 * area / max(s, 1e-9) - cr)
                    sw = max(0.0, 0.5 * (cr - ct))
                    shape = "trapezoid"
                    notes.append(f"free-form fins '{inner.findtext('name') or ''}' approximated by a trapezoid of the same area and span")
                x_le = x + _offset(inner, L, cr)
                m = om_i if om_i is not None else dens_i * t * area * n
                fins.append({"count": n, "root": cr, "tip": ct, "span": s, "sweep": sw, "thickness": t, "x_le": x_le, "shape": shape,
                             "mass": m, "edge": {"square": "square", "rounded": "rounded", "airfoil": "airfoil", "wedge": "airfoil"}.get(
                                 (inner.findtext("crosssection") or "square").strip().lower(), "square")})
                masses.append((inner.findtext("name") or "Fins", m, x_le + 0.5 * cr))
            elif tag == "launchlug":
                r = float(_f(inner, "radius", 0.002) or 0.002)
                th = float(_f(inner, "thickness", 0.0005) or 0.0005)
                m = om_i if om_i is not None else dens_i * 2 * math.pi * r * Li * th
                lug = {"d": 2 * r, "length": Li}
                masses.append((inner.findtext("name") or "Launch lug", m, x + _offset(inner, L, Li) + 0.5 * Li))
            elif tag == "parachute":
                d = float(_f(inner, "diameter", 0.3) or 0.3)
                cd = _f(inner, "cd", "auto")
                m = om_i if om_i is not None else dens_i * math.pi * d * d / 4
                chute = {"diameter_m": d, "cd": 0.8 if cd == "auto" or cd is None else float(cd)}
                masses.append((inner.findtext("name") or "Parachute", m, x + _offset(inner, L, Li) + 0.5 * Li))
            elif tag == "masscomponent":
                m = float(_f(inner, "mass", 0.0) or 0.0)
                masses.append((inner.findtext("name") or "Mass", m, x + _offset(inner, L, Li) + 0.5 * Li))
            elif tag in ("innertube", "tubecoupler", "engineblock", "centeringring", "bulkhead", "shockcord", "streamer"):
                ro = float(_f(inner, "outerradius", prev_r or 0.01) or 0.01)
                ri = float(_f(inner, "innerradius", 0.0) or 0.0) if tag != "bulkhead" else 0.0
                if tag in ("innertube", "tubecoupler", "engineblock"):
                    th = float(_f(inner, "thickness", 0.0005) or 0.0005)
                    vol = 2 * math.pi * ro * Li * th
                elif tag == "shockcord":
                    vol = float(_f(inner, "cordlength", 0.5) or 0.5)
                elif tag == "streamer":
                    vol = float(_f(inner, "stripwidth", 0.03) or 0.03) * float(_f(inner, "striplength", 0.5) or 0.5)
                else:
                    vol = math.pi * (ro * ro - ri * ri) * Li
                m = om_i if om_i is not None else dens_i * vol
                masses.append((inner.findtext("name") or tag, m, x + _offset(inner, L, Li) + 0.5 * Li))
            for mount in ([inner.find("motormount")] if inner.find("motormount") is not None else []):
                mo = mount.find("motor")
                if mo is not None and motor is None:
                    motor = {"manufacturer": (mo.findtext("manufacturer") or "").strip(), "designation": (mo.findtext("designation") or "").strip(),
                             "delay": _f(mo, "delay"), "diameter_mm": (float(_f(mo, "diameter", 0.0) or 0.0)) * 1000,
                             "length_mm": (float(_f(mo, "length", 0.0) or 0.0)) * 1000}
        mount = c.find("motormount")
        if mount is not None and motor is None:
            mo = mount.find("motor")
            if mo is not None:
                motor = {"manufacturer": (mo.findtext("manufacturer") or "").strip(), "designation": (mo.findtext("designation") or "").strip(),
                         "delay": _f(mo, "delay"), "diameter_mm": (float(_f(mo, "diameter", 0.0) or 0.0)) * 1000,
                         "length_mm": (float(_f(mo, "length", 0.0) or 0.0)) * 1000}
        masses.append((parts[-1].name, parts[-1].mass, parts[-1].cg_x))
        x += L
    return _assemble(name, parts, fins, lug, chute, motor, masses, notes, x)


def _assemble(name, parts, fins, lug, chute, motor, masses, notes, total_len):
    nose = next((p for p in parts if p.kind == "nose"), None)
    if nose is None:
        raise ValueError("the design has no nose cone")
    R = max(p.r_aft for p in parts)
    tubes = [p for p in parts if p.kind == "tube"]
    radii = sorted({round(p.r_aft, 5) for p in tubes})
    if len(radii) > 1:
        notes.append(f"body sections of {len(radii)} diameters: the template uses the largest ({2 * R * 1000:.1f} mm)")
    last = parts[-1]
    tail_len, taper = 0.0, 1.0
    if last.kind == "transition" and last.r_aft < last.r_fore - 1e-6:
        tail_len = last.length
        taper = max(0.2, last.r_aft / max(last.r_fore, 1e-9))
    elif any(p.kind == "transition" for p in parts):
        notes.append("a transition between body sections is not reproduced (the template body is one diameter)")
    fam = NOSE_MAP.get(nose.extra.get("shape", "ogive"), "ogive")
    values = {"nose_family": fam, "nose_length": nose.length, "length": total_len, "diameter": 2 * R,
              "tail_length": tail_len, "tail_taper": taper, "tip_radius": 0.0}
    if fam == "power":
        values["nose_power"] = 0.5 if nose.extra.get("shape") == "parabolic" else float(nose.extra.get("param") or 0.5)
    if fam == "haack":
        values["haack_c"] = float(nose.extra.get("param") or 0.0)
    rocket = {}
    if fins:
        f = max(fins, key=lambda q: q["span"] * q["root"])
        if len(fins) > 1:
            notes.append(f"{len(fins)} fin sets: the largest is used")
        values.update({"fin_count": str(min(max(f["count"], 3), 6)), "fin_root": f["root"], "fin_tip": f["tip"], "fin_span": f["span"],
                       "fin_sweep": math.degrees(math.atan2(f["sweep"], max(f["span"], 1e-9))), "fin_thickness": f["thickness"],
                       "fin_shape": f["shape"], "fin_position": max(0.0, (total_len - tail_len) - (f["x_le"] + f["root"]))})
        rocket["fin_edge"] = f["edge"]
    else:
        notes.append("no fins found: the template's default fins were kept")
    if lug:
        rocket["launch_lug"] = "1/8in" if lug["d"] < 0.0055 else ("3/16in" if lug["d"] < 0.0071 else "1/4in")
    else:
        rocket["launch_lug"] = "none"
    items = [{"name": n, "g": round(m * 1000, 2), "x_mm": round(x * 1000, 1)} for n, m, x in masses if m > 0]
    rocket["masses"] = {"mode": "items", "items": items}
    if chute:
        rocket["parachute"] = {"enabled": True, **chute}
    mot = None
    if motor and motor.get("designation"):
        b = find_builtin(motor["designation"], motor.get("manufacturer"))
        if b is not None:
            mot = {"id": b.id, "delay": motor.get("delay")}
            notes.append(f"motor {motor['manufacturer']} {motor['designation']} matched to the built-in {b.manufacturer} {b.name} (approximate curve)")
        else:
            notes.append(f"motor {motor['manufacturer']} {motor['designation']} is not in the built-in library: pick one or upload its .eng file")
    if mot:
        rocket["motor"] = mot
    return {"name": name, "template_values": values, "rocket": rocket, "notes": notes, "source_motor": motor,
            "parts": [{"kind": p.kind, "name": p.name, "x_mm": round(p.x0 * 1000, 1), "length_mm": round(p.length * 1000, 1),
                       "diameter_mm": round(2 * p.r_aft * 1000, 2), "g": round(p.mass * 1000, 2)} for p in parts]}


# ---- RockSim ----------------------------------------------------------------------------------
def parse_rkt(data: bytes):
    root = ET.fromstring(data)
    design = root.find(".//RocketDesign")
    if design is None:
        raise ValueError("not a RockSim design (no <RocketDesign>)")
    name = (design.findtext("Name") or "RockSim design").strip()
    notes = []
    stage = design.find("Stage3Parts")
    if stage is None:
        raise ValueError("the design has no parts")
    mm = 0.001
    parts, fins, masses = [], [], []
    lug = chute = motor = None
    x = 0.0
    prev_r = 0.0

    def num(el, tag, d=0.0):
        try:
            return float(el.findtext(tag) or d)
        except ValueError:
            return d

    def mass_of(el, vol_m3):
        if (el.findtext("UseKnownMass") or "0").strip() == "1":
            return num(el, "KnownMass") * 0.001
        m = num(el, "Mass")
        return m * 0.001 if m else num(el, "Density") * vol_m3

    for c in stage:
        tag = c.tag
        L = num(c, "Len") * mm
        if tag == "NoseCone":
            r = 0.5 * num(c, "BaseDia") * mm
            code = int(num(c, "ShapeCode", 1))
            shape = {0: "conical", 1: "ogive", 2: "parabolic", 3: "ellipsoid", 4: "power", 5: "parabolic", 6: "haack"}.get(code, "ogive")
            thick = num(c, "WallThickness") * mm
            m = mass_of(c, math.pi * r * math.hypot(L, r) * thick)
            parts.append(_Part("nose", c.findtext("Name") or "Nose cone", x, L, 0.0, r, m, x + 0.5 * L,
                               {"shape": shape, "param": num(c, "ShapeParameter", 0.5)}))
            prev_r = r
        elif tag == "BodyTube":
            r = 0.5 * num(c, "OD") * mm
            thick = 0.5 * (num(c, "OD") - num(c, "ID")) * mm
            m = mass_of(c, 2 * math.pi * r * L * thick)
            parts.append(_Part("tube", c.findtext("Name") or "Body tube", x, L, r, r, m, x + 0.5 * L))
            prev_r = r
            if (c.findtext("IsMotorMount") or "0").strip() == "1":
                motor = motor or {"diameter_mm": num(c, "MotorDia"), "designation": "", "manufacturer": "", "delay": None, "length_mm": 0.0}
        elif tag == "Transition":
            rf, ra = 0.5 * num(c, "FrontDia") * mm, 0.5 * num(c, "RearDia") * mm
            thick = num(c, "WallThickness") * mm
            m = mass_of(c, math.pi * (rf + ra) * math.hypot(L, rf - ra) * thick)
            parts.append(_Part("transition", c.findtext("Name") or "Transition", x, L, rf, ra, m, x + 0.5 * L))
            prev_r = ra
        else:
            continue
        att = c.find("AttachedParts")
        for inner in (att if att is not None else []):
            it = inner.tag
            if it == "FinSet":
                n = int(num(inner, "FinCount", 3))
                cr, ct = num(inner, "RootChord") * mm, num(inner, "TipChord") * mm
                s, sw, t = num(inner, "SemiSpan") * mm, num(inner, "SweepDistance") * mm, num(inner, "Thickness") * mm
                shape = "elliptical" if int(num(inner, "ShapeCode", 0)) == 1 else "trapezoid"
                area = (math.pi * cr * s / 4) if shape == "elliptical" else 0.5 * s * (cr + ct)
                xb = num(inner, "Xb") * mm
                loc = int(num(inner, "LocationMode", 0))
                x_le = x + (L - xb - cr if loc == 2 else xb)   # 0: from top, 2: from the bottom
                m = mass_of(inner, t * area * n)
                fins.append({"count": n, "root": cr, "tip": ct if shape == "trapezoid" else 0.0, "span": s, "sweep": sw, "thickness": t,
                             "x_le": x_le, "shape": shape, "mass": m, "edge": "square"})
                masses.append(("Fins", m, x_le + 0.5 * cr))
            elif it == "LaunchLug":
                lug = {"d": num(inner, "OD") * mm, "length": num(inner, "Len") * mm}
                masses.append(("Launch lug", mass_of(inner, 0.0) or 0.001, x + num(inner, "Xb") * mm))
            elif it == "Parachute":
                chute = {"diameter_m": num(inner, "Dia") * mm, "cd": num(inner, "DragCoefficient", 0.75) or 0.75}
                masses.append(("Parachute", mass_of(inner, 0.0), x + num(inner, "Xb") * mm))
            elif it in ("MassObject", "Streamer", "RingTube", "Bulkhead", "BodyTube", "Transition", "TubeFinSet"):
                masses.append((inner.findtext("Name") or it, mass_of(inner, 0.0), x + num(inner, "Xb") * mm + 0.5 * num(inner, "Len") * mm))
        masses.append((parts[-1].name, parts[-1].mass, parts[-1].cg_x))
        x += L
    for mo in root.iter("Motor"):
        code = mo.get("Code") or mo.findtext("Code") or ""
        if code:
            motor = {**(motor or {}), "designation": code, "manufacturer": mo.get("Manufacturer") or mo.findtext("Manufacturer") or "",
                     "delay": None}
            break
    total = sum(m for _, m, _ in masses)
    if total > 50.0:    # grams read as kilograms
        masses = [(n, m * 0.001, xx) for n, m, xx in masses]
        notes.append("masses read in grams")
    return _assemble(name, parts, fins, lug, chute, motor, masses, notes, x)


def parse_design(data: bytes, filename: str):
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "rkt":
        return parse_rkt(data)
    return parse_ork(data)


def drag_table_csv(curve, name=""):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"# AeroSculpt drag table {name}".strip()])
    w.writerow(["mach", "cd", "cd_friction", "cd_pressure", "cd_base"])
    for r in curve:
        w.writerow([r["mach"], r["cd"], r["cd_friction"], round(r["cd_nose"] + r["cd_tail"] + r["cd_fin_le"] + r["cd_fin_te"] + r["cd_lug"], 5), r["cd_base"]])
    return buf.getvalue()
