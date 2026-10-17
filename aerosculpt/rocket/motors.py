"""Rocket motors: thrust curves from RASP (.eng) and RockSim (.rse) files, a small built-in
library of common motors (approximate curves: upload the manufacturer's certified file for
exact numbers), and ThrustCurve.org search when the user asks for it.

A motor's mass falls with the impulse delivered so far: m(t) = m_case + m_prop (1 - I(t)/I_total).
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

CLASSES = "ABCDEFGHIJKLMNOP"


def impulse_class(total_ns):
    """Letter class: A 1.26-2.5 N·s, B 2.51-5, ... (each doubles)."""
    if total_ns <= 0:
        return "?"
    if total_ns <= 0.625:
        return "1/4A" if total_ns > 0.3125 else "1/8A"
    if total_ns <= 1.25:
        return "1/2A"
    k = int(math.ceil(math.log2(total_ns / 1.25)))
    return CLASSES[min(k - 1, len(CLASSES) - 1)] if k >= 1 else "A"


@dataclass
class Motor:
    id: str
    name: str                       # designation, e.g. C6
    manufacturer: str
    diameter_mm: float
    length_mm: float
    delays: list
    prop_mass_kg: float
    total_mass_kg: float
    curve: list                     # [(t, F)] with the final point at zero thrust
    source: str = "file"
    note: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        pts = [(float(t), max(0.0, float(f))) for t, f in self.curve]
        if not pts or pts[0][0] > 1e-9:
            pts.insert(0, (0.0, 0.0))
        self.curve = pts
        t = np.array([p[0] for p in pts])
        f = np.array([p[1] for p in pts])
        self._t, self._f = t, f
        imp = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
        self._imp = imp
        self.total_impulse = float(imp[-1])
        self.burn_time = float(t[-1])
        self.peak_thrust = float(f.max()) if len(f) else 0.0
        self.avg_thrust = self.total_impulse / self.burn_time if self.burn_time > 0 else 0.0
        self.cls = impulse_class(self.total_impulse)

    @property
    def case_mass_kg(self):
        return max(self.total_mass_kg - self.prop_mass_kg, 0.0)

    def thrust(self, t):
        if t <= 0 or t >= self.burn_time:
            return 0.0
        return float(np.interp(t, self._t, self._f))

    def impulse(self, t):
        if t <= 0:
            return 0.0
        if t >= self.burn_time:
            return self.total_impulse
        return float(np.interp(t, self._t, self._imp))

    def mass(self, t):
        if self.total_impulse <= 0:
            return self.total_mass_kg
        return self.case_mass_kg + self.prop_mass_kg * (1.0 - self.impulse(t) / self.total_impulse)

    @property
    def designation(self):
        return f"{self.name}" if self.name[:1] in CLASSES or self.name.startswith("1/") else self.name

    def public(self):
        return {"id": self.id, "name": self.name, "manufacturer": self.manufacturer, "class": self.cls,
                "diameter_mm": self.diameter_mm, "length_mm": self.length_mm, "delays": self.delays,
                "prop_mass_g": round(self.prop_mass_kg * 1000, 2), "total_mass_g": round(self.total_mass_kg * 1000, 2),
                "total_impulse": round(self.total_impulse, 3), "avg_thrust": round(self.avg_thrust, 2),
                "peak_thrust": round(self.peak_thrust, 2), "burn_time": round(self.burn_time, 3),
                "source": self.source, "note": self.note, "curve": [[round(t, 4), round(f, 3)] for t, f in self.curve]}

    def to_eng(self):
        head = f"{self.name} {self.diameter_mm:g} {self.length_mm:g} {'-'.join(str(d) for d in self.delays) or '0'} " \
               f"{self.prop_mass_kg:.5f} {self.total_mass_kg:.5f} {self.manufacturer.replace(' ', '_')}"
        body = "\n".join(f"{t:.4f} {f:.3f}" for t, f in self.curve if t > 0)
        return f"; {self.note or 'AeroSculpt export'}\n{head}\n{body}\n"


# ---- files ----------------------------------------------------------------------------
def parse_eng(text, source="RASP file"):
    """RASP .eng: comment lines (;), one header 'name diam len delays propmass totalmass mfr',
    then 'time thrust' pairs. A file may hold several motors."""
    motors = []
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        if not ln or ln.startswith(";"):
            continue
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            name, diam, length, delays, pm, tm = parts[0], float(parts[1]), float(parts[2]), parts[3], float(parts[4]), float(parts[5])
        except ValueError:
            continue
        mfr = " ".join(parts[6:]).replace("_", " ")
        pts = []
        while i < len(lines):
            q = lines[i]
            if not q or q.startswith(";"):
                i += 1
                if pts and (not q):
                    break
                continue
            nums = q.split()
            try:
                t, f = float(nums[0]), float(nums[1])
            except (ValueError, IndexError):
                break
            pts.append((t, f))
            i += 1
            if f <= 0 and t > 0:
                break
        if not pts:
            continue
        ds = [float(d) for d in re.split(r"[-,]", delays) if d.replace(".", "", 1).isdigit()]
        mid = f"{mfr}-{name}".lower().replace(" ", "")
        motors.append(Motor(mid, name, mfr, diam, length, ds, pm, tm, pts, source))
    if not motors:
        raise ValueError("no motor found in the file (expected RASP .eng format)")
    return motors


def parse_rse(text, source="RockSim file"):
    """RockSim .rse XML: <engine-database><engine-list><engine ...><data><eng-data t= f= m=/>"""
    root = ET.fromstring(text)
    motors = []
    for e in root.iter("engine"):
        a = e.attrib
        pts = [(float(d.get("t")), float(d.get("f"))) for d in e.iter("eng-data")]
        if not pts:
            continue
        name = a.get("code") or a.get("name") or "motor"
        mfr = a.get("mfg", "")
        ds = [float(d) for d in a.get("delays", "").split(",") if d.strip().replace(".", "", 1).isdigit()]
        motors.append(Motor(f"{mfr}-{name}".lower().replace(" ", ""), name, mfr, float(a.get("dia", 0)), float(a.get("len", 0)),
                            ds, float(a.get("propWt", 0)) / 1000.0, float(a.get("initWt", 0)) / 1000.0, pts, source))
    if not motors:
        raise ValueError("no motor found in the file (expected RockSim .rse format)")
    return motors


def parse_motor_file(text, filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "rse" or text.lstrip().startswith("<"):
        return parse_rse(text)
    return parse_eng(text)


# ---- built-in library (approximate) --------------------------------------------------------
def _shape(peak, avg, burn, t_peak=0.2, tail=0.15):
    """A plausible black-powder / composite curve: spike at t_peak, a plateau, a short tail,
    scaled so the average thrust over the burn is exactly `avg`."""
    pts = [(0.0, 0.0), (t_peak * 0.5, peak * 0.9), (t_peak, peak), (t_peak * 1.6, avg * 1.05),
           (burn * (1 - tail), avg * 0.95), (burn, 0.0)]
    t = np.array([p[0] for p in pts])
    f = np.array([p[1] for p in pts])
    imp = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    f *= avg * burn / imp
    return [(float(a), float(b)) for a, b in zip(t, f)]


_NOTE = "Approximate built-in curve shaped to the published impulse, burn time and peak thrust; upload the certified .eng for exact data."
BUILTIN = [
    Motor("estes-a8", "A8", "Estes", 18, 70, [3, 5], 0.0033, 0.0162, _shape(9.7, 2.5 / 0.5, 0.5, 0.18), "built-in", _NOTE),
    Motor("estes-b6", "B6", "Estes", 18, 70, [2, 4, 6], 0.0056, 0.0180, _shape(12.1, 5.0 / 0.86, 0.86, 0.2), "built-in", _NOTE),
    Motor("estes-c6", "C6", "Estes", 18, 70, [3, 5, 7], 0.0125, 0.0240, _shape(14.1, 8.8 / 1.86, 1.86, 0.2), "built-in", _NOTE),
    Motor("estes-d12", "D12", "Estes", 24, 70, [3, 5, 7], 0.0210, 0.0422, _shape(32.0, 16.8 / 1.65, 1.65, 0.25), "built-in", _NOTE),
    Motor("estes-e12", "E12", "Estes", 24, 95, [4, 6, 8], 0.0358, 0.0600, _shape(32.0, 28.5 / 2.4, 2.4, 0.3), "built-in", _NOTE),
    Motor("estes-f15", "F15", "Estes", 29, 114, [4, 6, 8], 0.0600, 0.1020, _shape(25.0, 49.6 / 3.4, 3.4, 0.3), "built-in", _NOTE),
    Motor("aerotech-f32", "F32", "AeroTech", 24, 70, [4, 6, 8], 0.0270, 0.0530, _shape(50.0, 56.9 / 1.8, 1.8, 0.1, 0.25), "built-in", _NOTE),
    Motor("aerotech-g80", "G80", "AeroTech", 29, 124, [4, 7, 10], 0.0620, 0.1250, _shape(120.0, 136.6 / 1.7, 1.7, 0.1, 0.25), "built-in", _NOTE),
    Motor("aerotech-h128", "H128", "AeroTech", 29, 194, [6, 10, 14], 0.1100, 0.2200, _shape(190.0, 219.0 / 1.75, 1.75, 0.1, 0.25), "built-in", _NOTE),
    Motor("aerotech-i200", "I200", "AeroTech", 38, 240, [6, 10, 14], 0.2100, 0.4100, _shape(280.0, 404.0 / 2.1, 2.1, 0.1, 0.25), "built-in", _NOTE),
    Motor("cti-j357", "J357", "Cesaroni", 38, 367, [], 0.3600, 0.6500, _shape(430.0, 704.0 / 2.0, 2.0, 0.1, 0.25), "built-in", _NOTE),
]
BY_ID = {m.id: m for m in BUILTIN}


def find_builtin(designation, manufacturer=None):
    """Match 'C6-5', 'C6', 'Estes C6' to a built-in motor."""
    if not designation:
        return None
    d = designation.strip().upper().split("-")[0].replace(" ", "")
    for m in BUILTIN:
        if m.name.upper() == d and (not manufacturer or manufacturer.lower().split()[0] in m.manufacturer.lower()):
            return m
    for m in BUILTIN:
        if m.name.upper() == d:
            return m
    return None


def motor_for_diameter(body_d_m):
    """A sensible default motor for a body diameter."""
    d = body_d_m * 1000
    if d < 32:
        return BY_ID["estes-c6"]
    if d < 48:
        return BY_ID["estes-e12"]
    if d < 66:
        return BY_ID["aerotech-g80"]
    if d < 120:
        return BY_ID["aerotech-h128"]
    return BY_ID["aerotech-i200"]


# ---- ThrustCurve.org (only when asked: a network request) --------------------------------------
TC_API = "https://www.thrustcurve.org/api/v1/"


def thrustcurve_search(query, timeout=10.0):
    q = {"maxResults": 25}
    m = re.match(r"^\s*([A-P]|1/[248]A)\s*(\d+)?", query.strip().upper())
    if m and len(query.strip()) <= 8:
        q["impulseClass"] = m.group(1)
        q["commonName"] = query.strip().upper().split("-")[0]
    else:
        q["manufacturer"] = query.strip()
    url = TC_API + "search.json?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AeroSculpt"}), timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for r_ in data.get("results", []):
        out.append({"tc_id": r_.get("motorId"), "name": r_.get("commonName") or r_.get("designation"),
                    "designation": r_.get("designation"), "manufacturer": r_.get("manufacturer"),
                    "diameter_mm": r_.get("diameter"), "length_mm": r_.get("length"), "total_impulse": r_.get("totImpulseNs"),
                    "avg_thrust": r_.get("avgThrustN"), "burn_time": r_.get("burnTimeS"), "class": r_.get("impulseClass"),
                    "delays": r_.get("delays"), "availability": r_.get("availability")})
    return out


def thrustcurve_download(tc_id, timeout=10.0):
    url = TC_API + "download.json?" + urllib.parse.urlencode({"motorId": tc_id, "format": "RASP", "data": "samples"})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AeroSculpt"}), timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    results = data.get("results") or []
    if not results:
        raise ValueError("ThrustCurve.org has no data file for this motor")
    import base64
    best = results[0]
    text = base64.b64decode(best["data"]).decode("utf-8", "replace")
    motors = parse_eng(text, source="ThrustCurve.org")
    for m in motors:
        m.id = f"tc-{tc_id}"
        m.note = f"ThrustCurve.org, {best.get('source', 'certified')} data"
    return motors[0]
