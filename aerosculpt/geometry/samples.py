"""Parametric sample bodies, used for onboarding demos and for solver validation.

All builders return a Mesh in metres, nose towards -x and z up (already wind axes).
"""
from __future__ import annotations

import math

import numpy as np

from .mesh import Mesh


def loft(sections, cap_start=True, cap_end=True):
    """Skin a list of closed rings (each (M, 3)) into a closed mesh.

    A ring with (near) zero radius is collapsed into a single tip vertex.
    """
    M = len(sections[0])
    verts = []
    rings = []  # list of index arrays (len M) or a single int for a tip
    for s in sections:
        s = np.asarray(s, float)
        c = s.mean(axis=0)
        if np.linalg.norm(s - c, axis=1).max() < 1e-12:
            rings.append(len(verts))
            verts.append(c)
        else:
            rings.append(np.arange(len(verts), len(verts) + M))
            verts.extend(s)
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        if isinstance(a, int) and isinstance(b, int):
            continue
        for k in range(M):
            k1 = (k + 1) % M
            if isinstance(a, int):
                faces.append((a, b[k1], b[k]))
            elif isinstance(b, int):
                faces.append((a[k], a[k1], b))
            else:
                faces.append((a[k], a[k1], b[k1]))
                faces.append((a[k], b[k1], b[k]))
    for ring, cap, flip in ((rings[0], cap_start, True), (rings[-1], cap_end, False)):
        if cap and not isinstance(ring, int):
            ci = len(verts)
            verts.append(np.mean([verts[i] for i in ring], axis=0))
            for k in range(M):
                k1 = (k + 1) % M
                faces.append((ci, ring[k1], ring[k]) if flip else (ci, ring[k], ring[k1]))
    m = Mesh(np.array(verts), np.array(faces))
    return m.orient_outward()


def merge(*meshes):
    v = []
    f = []
    off = 0
    for m in meshes:
        v.append(m.v)
        f.append(m.f + off)
        off += len(m.v)
    return Mesh(np.vstack(v), np.vstack(f))


def icosphere(radius=0.5, subdiv=4):
    t = (1 + 5 ** 0.5) / 2
    v = [np.array(p, float) for p in
         [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
          [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]]
    v = [p / np.linalg.norm(p) for p in v]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
         [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    for _ in range(subdiv):
        cache = {}

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]

        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return Mesh(np.array(v) * radius, np.array(f)).orient_outward()


def box(size=(1, 1, 1), n=12):
    """Axis-aligned box centred at the origin with each face split into n x n quads."""
    sx, sy, sz = (np.array(size, float) / 2)
    verts = []
    faces = []
    g = np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sign in (-1, 1):
            base = len(verts)
            for a in g:
                for b in g:
                    p = [0.0, 0.0, 0.0]
                    o = [k for k in range(3) if k != axis]
                    p[axis] = sign
                    p[o[0]] = a
                    p[o[1]] = b
                    verts.append(p)
            for i in range(n):
                for j in range(n):
                    a0 = base + i * (n + 1) + j
                    faces.append((a0, a0 + n + 1, a0 + n + 2))
                    faces.append((a0, a0 + n + 2, a0 + 1))
    m = Mesh.from_triangles((np.array(verts) * [sx, sy, sz])[np.array(faces)])
    return m.orient_outward()


def revolve(xs, rs, m=48, center=(0.0, 0.0)):
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    secs = []
    for x, r in zip(xs, rs):
        secs.append(np.stack([np.full(m, x), center[0] + r * np.cos(th), center[1] + r * np.sin(th)], axis=1))
    return loft(secs)


def streamlined_pod(length=1.0, diameter=0.2, n=60, m=48):
    """Myring-type hull: elliptic nose, parallel mid-body, cubic tail."""
    a, b, c = 0.25 * length, 0.4 * length, 0.35 * length
    R = diameter / 2
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n))) * length
    rs = []
    for x in xs:
        if x < a:
            rs.append(R * (1 - ((x - a) / a) ** 2) ** 0.5)
        elif x < a + b:
            rs.append(R)
        else:
            s = (x - a - b) / c
            rs.append(R * max(0.0, 1 - s * s * (3 - 2 * s)) ** 0.9)
    rs[0] = 0.0
    rs[-1] = 0.0
    return revolve(xs - length / 2, rs, m)


def blunt_pod(length=1.0, width=0.3, height=0.25, n=40, m=56, nexp=6.0):
    """Superelliptic cross-section pod with a blunt, nearly flat base: lots of base drag."""
    xs = np.linspace(0, length, n)
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    secs = []
    for x in xs:
        s = x / length
        nose = min(1.0, (s / 0.18)) ** 0.5
        tail = 1.0 - 0.15 * max(0.0, (s - 0.85) / 0.15)
        k = nose * tail
        cy = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / nexp)
        cz = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / nexp)
        secs.append(np.stack([np.full(m, x - length / 2), k * width / 2 * cy, k * height / 2 * cz], axis=1))
    secs[0] = np.repeat(secs[0].mean(axis=0, keepdims=True), m, axis=0)
    return loft(secs)


def naca4(code="0012", n=60):
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0 and p > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    # closed loop: TE -> upper -> LE -> lower -> TE (drop duplicate points)
    px = np.concatenate([xu[::-1], xl[1:-1]])
    pz = np.concatenate([yu[::-1], yl[1:-1]])
    return px, pz


def wing(span=1.0, root_chord=0.25, taper=0.5, sweep_deg=5.0, code="2412", n_span=24, n_af=50):
    px, pz = naca4(code, n_af)
    secs = []
    etas = np.sin(np.linspace(-np.pi / 2, np.pi / 2, n_span)) * 1.0
    for eta in etas:
        y = eta * span / 2
        c = root_chord * (1 - (1 - taper) * abs(eta))
        tip_round = math.sqrt(max(0.0, 1 - max(0.0, (abs(eta) - 0.94) / 0.06) ** 2))
        c_eff = c * max(tip_round, 0.0)
        xle = abs(y) * math.tan(math.radians(sweep_deg)) + (c - c_eff) * 0.3
        if c_eff < 1e-6:
            secs.append(np.tile([[xle + 0.3 * c, y, 0.0]], (len(px), 1)))
        else:
            secs.append(np.stack([xle + px * c_eff, np.full(len(px), y), pz * c_eff], axis=1))
    m = loft(secs)
    lo, hi = m.bbox()
    return m.transformed(t=[-(lo[0] + hi[0]) / 2, 0, 0])


def fin(x0, root, tip, span, sweep, thick, angle_deg):
    """Thin trapezoidal plate fin, rotated about the x axis to angle_deg."""
    pts = np.array([[x0, 0, 0], [x0 + root, 0, 0], [x0 + sweep + tip, span, 0], [x0 + sweep, span, 0]], float)
    top = pts + [0, 0, thick / 2]
    bot = pts - [0, 0, thick / 2]
    v = np.vstack([bot, top])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    a = math.radians(angle_deg)
    R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    return Mesh(v @ R.T, f).orient_outward()


def rocket(length=1.0, diameter=0.1, m=40):
    R = diameter / 2
    ln = 0.22 * length
    lt = 0.06 * length
    xs, rs = [], []
    rho = (R * R + ln * ln) / (2 * R)                  # tangent ogive
    for s in np.linspace(0, 1, 30):
        x = s * ln
        xs.append(x)
        rs.append(math.sqrt(max(rho * rho - (ln - x) ** 2, 0)) + R - rho)
    for x in np.linspace(ln, length - lt, 12)[1:]:
        xs.append(x)
        rs.append(R)
    for s in np.linspace(0, 1, 8)[1:]:
        xs.append(length - lt + s * lt)
        rs.append(R * (1 - 0.3 * s))
    rs[0] = 0.0
    body = revolve(np.array(xs), np.array(rs), m)
    fins = [fin(length - 0.2 * length, 0.17 * length, 0.07 * length, 1.3 * diameter, 0.09 * length,
                0.006 * length, 45 + 90 * k) for k in range(4)]
    # push fins out so their root sits inside the body surface
    out = []
    for k, fm in enumerate(fins):
        a = math.radians(45 + 90 * k)
        shift = (R * 0.6) * np.array([0, math.cos(a), math.sin(a)])
        out.append(fm.transformed(t=shift))
    mesh = merge(body, *out)
    return mesh.transformed(t=[-length / 2, 0, 0])


def fixed_wing_uav(length=1.4, span=2.0):
    """Small fixed-wing UAV: payload pod tapering into a tail boom, high tapered wing
    (NACA 2412), horizontal tail and fin (NACA 0012). Wing and tail are separate shells
    that overlap the fuselage, exactly like a multi-body CAD export."""
    R = 0.07
    xs = np.linspace(0, length, 70)
    rs = []
    for x in xs:
        s = x / length
        if s < 0.18:
            rs.append(R * math.sqrt(max(0.0, 1 - ((s - 0.18) / 0.18) ** 2)))
        elif s < 0.45:
            rs.append(R)
        else:
            t = (s - 0.45) / 0.55
            rs.append(0.016 + (R - 0.016) * max(0.0, 1 - t) ** 1.6)
    rs[0] = 0.0
    rs[-1] = 0.0
    body = revolve(xs, np.array(rs), 36)
    w = wing(span, 0.30, 0.55, 3.0, "2412", n_span=26, n_af=44).transformed(t=[0.36 * length, 0.0, 0.035])
    ht = wing(0.62, 0.16, 0.7, 6.0, "0012", n_span=14, n_af=30).transformed(t=[0.93 * length, 0.0, 0.0])
    fin = wing(0.44, 0.18, 0.55, 18.0, "0012", n_span=12, n_af=30)
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])   # span along y -> along z
    fin = fin.transformed(R=Rx, t=[0.92 * length, 0.0, 0.15])
    m = merge(body, w, ht, fin)
    lo, hi = m.bbox()
    return m.transformed(t=-(lo + hi) / 2)


def sphere_cone(length=2.0, half_angle=7.0, nose_radius=0.02, m=56):
    """Blunted cone: spherical nose cap tangent to a straight cone, flat base."""
    th = math.radians(half_angle)
    xt = nose_radius * (1 - math.sin(th))                  # tangency point from the tip
    xs, rs = [], []
    for phi in np.linspace(0, 0.5 * math.pi - th, 14):
        xs.append(nose_radius * (1 - math.cos(phi)))
        rs.append(nose_radius * math.sin(phi))
    rt = nose_radius * math.cos(th)
    for x in np.linspace(xt, length, 40)[1:]:
        xs.append(x)
        rs.append(rt + (x - xt) * math.tan(th))
    mesh = revolve(np.array(xs), np.array(rs), m)
    return mesh.transformed(t=[-length / 2, 0, 0])


def capsule(diameter=2.0, m=56):
    """Apollo-like re-entry capsule: spherical heat shield (radius 1.2 D) leading, toroidal
    shoulder (radius 0.05 D) tangent to it, 33 deg conical back shell."""
    R = diameter / 2
    Rs = 1.2 * diameter
    rsh = 0.05 * diameter
    phi_end = math.asin((R - rsh) / (Rs - rsh))
    xs, rs = [], []
    for phi in np.linspace(0, phi_end, 18):
        xs.append(Rs * (1 - math.cos(phi)))
        rs.append(Rs * math.sin(phi))
    cx = Rs - (Rs - rsh) * math.cos(phi_end)
    cr = (Rs - rsh) * math.sin(phi_end)
    for psi in np.linspace(phi_end, math.radians(123.0), 10)[1:]:
        xs.append(cx - rsh * math.cos(psi))
        rs.append(cr + rsh * math.sin(psi))
    xb, rb = xs[-1], rs[-1]
    r_top = 0.3 * R
    for r in np.linspace(rb, r_top, 12)[1:]:
        xs.append(xb + (rb - r) / math.tan(math.radians(33.0)))
        rs.append(r)
    mesh = revolve(np.array(xs), np.array(rs), m)
    lo, hi = mesh.bbox()
    return mesh.transformed(t=-(lo + hi) / 2)

SAMPLES = {
    "rocket": {
        "title": "Sounding rocket", "category": "rocket", "build": lambda: rocket(1.2, 0.1),
        "description": "Tangent-ogive nose, boat-tail and four thin fins — try protecting the fins and optimizing the nose.",
        "envelope": {"speed_min": 40.0, "speed_max": 170.0, "speed_count": 4, "aoa_min": 0.0, "aoa_max": 6.0, "aoa_count": 3}},
    "fixed-wing": {
        "title": "Fixed-wing UAV", "category": "fixed-wing", "build": lambda: fixed_wing_uav(),
        "description": "2 m span, high tapered wing, boom tail. Sweep angle of attack for its drag polar and best L/D.",
        # Coefficients on the wing: trapezoidal planform area and mean aerodynamic chord.
        "envelope": {"speed_min": 12.0, "speed_max": 30.0, "speed_count": 4, "aoa_min": -2.0, "aoa_max": 8.0, "aoa_count": 3,
                     "ref_area": 0.465, "ref_length": 0.239}},
    "wing": {
        "title": "Tapered wing", "category": "fixed-wing", "build": lambda: wing(1.0, 0.25, 0.5, 5.0, "2412"),
        "description": "NACA 2412, taper 0.5, 5° sweep. Coefficients use the wing area, so CL and CD read like a wing polar.",
        "envelope": {"speed_min": 15.0, "speed_max": 35.0, "speed_count": 3, "aoa_min": -2.0, "aoa_max": 8.0, "aoa_count": 3,
                     "ref_area": 0.1875, "ref_length": 0.194}},
    "streamlined-pod": {
        "title": "Payload pod", "category": "component", "build": lambda: streamlined_pod(1.0, 0.2),
        "description": "Fineness-5 body of revolution, the kind of pod slung under a UAV wing.",
        "envelope": {"speed_min": 20.0, "speed_max": 60.0, "speed_count": 3, "aoa_min": -4.0, "aoa_max": 4.0, "aoa_count": 3}},
    "blunt-pod": {
        "title": "Blunt cargo pod", "category": "component", "build": lambda: blunt_pod(1.0, 0.32, 0.26),
        "description": "Box-like pod with a flat base: lots of recoverable drag. Add a keep-inside cargo box first.",
        "envelope": {"speed_min": 20.0, "speed_max": 50.0, "speed_count": 3, "aoa_min": 0.0, "aoa_max": 0.0, "aoa_count": 1}},
    "rc-trainer": {
        "title": "RC trainer with tractor prop", "category": "rc", "tunnel": "rc", "build": lambda: fixed_wing_uav(),
        "description": "The 2 m UAV airframe with an 11x7 propeller at 8,000 rpm on the nose: drag in the slipstream, thrust and top speed.",
        "envelope": {"speed_min": 10.0, "speed_max": 22.0, "speed_count": 3, "aoa_min": 0.0, "aoa_max": 4.0, "aoa_count": 2,
                     "ref_area": 0.465, "ref_length": 0.239},
        # hub on the fuselage axis, which sits 0.15 m below the centre of the model's bounding box
        "propellers": [{"name": "Nose prop", "center": [-0.735, 0.0, -0.15], "diameter_in": 11.0, "pitch_in": 7.0,
                        "blades": 2, "rpm": 8000.0, "rotation": "cw", "width": 1.0, "family": "thin-electric"}]},
    "rocket-supersonic": {
        "title": "Supersonic sounding rocket", "category": "supersonic", "tunnel": "supersonic", "build": lambda: rocket(1.2, 0.1),
        "description": "Mach 1.5 to 4 at 10 km: wave, base and friction drag, centre of pressure, and nose heating on a fibreglass airframe.",
        "envelope": {"speed_min": 40.0, "speed_max": 170.0, "speed_count": 4, "aoa_min": 0.0, "aoa_max": 6.0, "aoa_count": 3},
        "hyper": {"envelope": {"mach_min": 1.5, "mach_max": 4.0, "mach_count": 4, "aoa_min": 0.0, "aoa_max": 4.0, "aoa_count": 3,
                               "altitude": 10000.0},
                  "thermal": {"material": "fiberglass"}}},
    "hypersonic-cone": {
        "title": "Hypersonic blunted cone", "category": "supersonic", "tunnel": "supersonic", "build": lambda: sphere_cone(),
        "description": "7° cone, 20 mm nose radius, Mach 4 to 7 at 35 km. Inconel skin, carbon-carbon nose cap: the skin limit binds at Mach 7.",
        "envelope": {"speed_min": 40.0, "speed_max": 170.0, "speed_count": 3, "aoa_min": 0.0, "aoa_max": 4.0, "aoa_count": 2},
        "hyper": {"envelope": {"mach_min": 4.0, "mach_max": 7.0, "mach_count": 4, "aoa_min": 0.0, "aoa_max": 6.0, "aoa_count": 3,
                               "altitude": 35000.0},
                  "thermal": {"material": "inconel"}},
        "zones": [{"type": "material", "shape": "cylinder", "center": [-0.94, 0.0, 0.0], "size": [0.14, 0.1, 0.1],
                   "name": "Nose cap", "material": "rcc"}]},
    "capsule": {
        "title": "Re-entry capsule", "category": "supersonic", "tunnel": "supersonic", "build": lambda: capsule(),
        "description": "Apollo-like heat shield leading at Mach 6 to 10, 45 km: bow-shock drag, stagnation heating, ablator limits.",
        "envelope": {"speed_min": 20.0, "speed_max": 40.0, "speed_count": 2, "aoa_min": 0.0, "aoa_max": 0.0, "aoa_count": 1},
        "hyper": {"envelope": {"mach_min": 6.0, "mach_max": 10.0, "mach_count": 3, "aoa_min": -20.0, "aoa_max": 0.0,
                               "aoa_count": 3, "altitude": 45000.0},
                  "thermal": {"material": "ablator"}}},
    "sphere": {
        "title": "Sphere (validation)", "category": "validation", "hidden": True, "build": lambda: icosphere(0.5, 4),
        "description": "Unit-diameter sphere, the classic drag validation case.",
        "envelope": {"speed_min": 10.0, "speed_max": 10.0, "speed_count": 1, "aoa_min": 0.0, "aoa_max": 0.0, "aoa_count": 1}},
    "cube": {
        "title": "Cube (validation)", "category": "validation", "hidden": True, "build": lambda: box((1, 1, 1), 12),
        "description": "Face-on cube; Cd about 1.05 in the Reynolds-independent range.",
        "envelope": {"speed_min": 10.0, "speed_max": 10.0, "speed_count": 1, "aoa_min": 0.0, "aoa_max": 0.0, "aoa_count": 1}},
}
