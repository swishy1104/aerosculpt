"""Parametric templates: bodies built from named dimensions instead of an uploaded mesh.

Each template is a list of parameters (units, hard limits, default optimizer bounds, lock) and
a builder that turns values into a watertight mesh (one closed shell per part), the part each
face belongs to, derived quantities (wing area, fineness ratio, ...) and, for winged templates,
the lifting surfaces for the wing model.

Coordinates are metres with the nose tip (or the wing's root leading edge) at the origin, nose
towards -x, z up -- so regions and propellers placed on a template stay attached to the same
spot when its dimensions change.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..aero import naca
from ..aero.vlm import LiftingSurface
from ..aero.wing import DragSurface
from .mesh import Mesh

BODY, WING, HTAIL, VTAIL, FINS = 0, 1, 2, 3, 4
GROUP_NAMES = {BODY: "body", WING: "wing", HTAIL: "horizontal tail", VTAIL: "vertical tail", FINS: "fins"}

NOSE_FAMILIES = [("conical", "Conical"), ("ogive", "Tangent ogive"), ("elliptical", "Elliptical"),
                 ("power", "Power law"), ("haack", "Von Kármán (Haack)")]
AIRFOILS = [("naca4", "NACA 4-digit"), ("naca5", "NACA 5-digit")]


@dataclass
class Param:
    key: str
    label: str
    group: str
    unit: str                    # "m", "deg", "ratio", "-" or "choice"
    lo: float = 0.0              # hard limits
    hi: float = 1.0
    default: object = 0.0
    bounds: tuple = None         # default optimizer range (inside lo..hi)
    choices: list = None
    locked: bool = False         # default lock state
    show_if: dict = None         # {param: [values]} -- only meaningful when this holds
    help: str = ""
    step: float = None

    @property
    def kind(self):
        return "choice" if self.choices else "float"

    def public(self):
        d = asdict(self)
        d["kind"] = self.kind
        d["bounds"] = list(self.bounds) if self.bounds else [self.lo, self.hi]
        return d


def _nose_params(default_family, length, bounds, power=0.6, family_locked=False):
    return [
        Param("nose_family", "Nose profile", "nose", "choice", default=default_family, choices=NOSE_FAMILIES, locked=family_locked,
              help="Conical and ogive noses are simple to make; von Kármán has the least wave drag for its length "
                   "and diameter at supersonic speed; elliptical suits subsonic bodies; power law ~0.75 suits hypersonic."),
        Param("nose_length", "Nose length", "nose", "m", 0.005, 20.0, length, bounds,
              help="Tip to shoulder. Its ratio to the body diameter is the nose fineness ratio."),
        Param("tip_radius", "Nose tip radius", "nose", "m", 0.0, 1.0, 0.0, (0.0, 0.25 * length / 2.6),
              help="Spherical blunting tangent to the profile. A blunter tip lowers stagnation heating "
                   "(heating falls as 1/√radius) but adds drag."),
        Param("nose_power", "Power-law exponent", "nose", "-", 0.25, 1.0, power, (0.4, 0.9),
              show_if={"nose_family": ["power"]}, help="r ∝ x^n: 1 is a cone, 0.5 a paraboloid; about 0.75 has the least "
                                                         "Newtonian (hypersonic) drag."),
        Param("haack_c", "Haack C", "nose", "-", 0.0, 0.34, 0.0, (0.0, 0.3333), show_if={"nose_family": ["haack"]},
              help="0 = von Kármán (LD-Haack, least drag for length and diameter); 1/3 = LV-Haack (for length and volume)."),
    ]


def _wing_params(span, root, tip, sweep, x_frac=None, z_frac=None, dihedral=2.0, locked_pos=False):
    ps = [
        Param("wing_span", "Wing span", "wing", "m", 0.05, 30.0, span, (0.8 * span, 1.2 * span), help="Tip to tip."),
        Param("wing_root", "Wing root chord", "wing", "m", 0.01, 5.0, root, (0.75 * root, 1.3 * root)),
        Param("wing_tip", "Wing tip chord", "wing", "m", 0.005, 5.0, tip, (0.4 * tip, 1.5 * tip)),
        Param("wing_sweep", "Wing sweep (leading edge)", "wing", "deg", -20.0, 60.0, sweep, (0.0, 15.0)),
        Param("wing_thickness", "Airfoil thickness", "wing", "ratio", 0.04, 0.24, 0.12, (0.08, 0.16),
              help="Maximum thickness as a fraction of chord: the last two digits of the NACA number."),
        Param("airfoil", "Airfoil family", "wing", "choice", default="naca4", choices=AIRFOILS, locked=True),
        Param("wing_camber", "Airfoil camber", "wing", "ratio", 0.0, 0.09, 0.02, (0.0, 0.05), show_if={"airfoil": ["naca4"]},
              help="Maximum camber as a fraction of chord: the first digit of a NACA 4-digit number."),
        Param("wing_camber_pos", "Camber position", "wing", "ratio", 0.15, 0.7, 0.4, (0.25, 0.55), show_if={"airfoil": ["naca4"]},
              help="Where the maximum camber sits, as a fraction of chord: the second digit."),
        Param("wing_cl_design", "Design lift coefficient", "wing", "-", 0.05, 0.9, 0.3, (0.15, 0.6), show_if={"airfoil": ["naca5"]},
              help="First digit × 3/20 of a NACA 5-digit number (230xx: 0.3)."),
        Param("wing_camber_pos5", "Camber position", "wing", "ratio", 0.05, 0.25, 0.15, (0.05, 0.25), show_if={"airfoil": ["naca5"]},
              help="Second digit / 20 of a NACA 5-digit number (230xx: 0.15)."),
        Param("wing_dihedral", "Wing dihedral", "wing", "deg", -15.0, 15.0, dihedral, (0.0, 6.0), locked=True),
        Param("wing_twist", "Wing twist (tip)", "wing", "deg", -8.0, 4.0, 0.0, (-4.0, 1.0),
              help="Tip incidence relative to the root; negative is washout."),
        Param("wing_incidence", "Wing incidence", "wing", "deg", -6.0, 10.0, 0.0, (-2.0, 4.0),
              help="Root chord angle to the fuselage axis: sets the fuselage attitude in cruise."),
    ]
    if x_frac is not None:
        ps.append(Param("wing_x", "Wing position along body", "wing", "ratio", 0.05, 0.85, x_frac, (x_frac - 0.06, x_frac + 0.06),
                        locked=locked_pos, help="Root leading edge from the nose, as a fraction of body length."))
        ps.append(Param("wing_z", "Wing vertical position", "wing", "-", -1.0, 1.0, z_frac, (-1.0, 1.0), locked=True,
                        help="-1 low wing, 0 mid wing, 1 high wing (fraction of the fuselage radius)."))
    return ps


TEMPLATES = {
    "rocket": {
        "title": "Rocket", "category": "rocket", "help": "Nose, body tube, boat-tail and fins.",
        "params": _nose_params("ogive", 0.264, (0.15, 0.45)) + [
            Param("length", "Overall length", "body", "m", 0.05, 50.0, 1.2, (1.0, 1.4), locked=True),
            Param("diameter", "Body diameter", "body", "m", 0.005, 5.0, 0.1, (0.085, 0.115), locked=True),
            Param("tail_length", "Boat-tail length", "tail", "m", 0.0, 10.0, 0.072, (0.02, 0.16)),
            Param("tail_taper", "Base / body diameter", "tail", "ratio", 0.2, 1.0, 0.7, (0.5, 1.0),
                  help="1 is a flat base the size of the body; smaller is a boat-tail with less base drag."),
            Param("fin_count", "Number of fins", "fins", "choice", default="4", choices=[("3", "3"), ("4", "4")], locked=True),
            Param("fin_root", "Fin root chord", "fins", "m", 0.005, 5.0, 0.204, (0.15, 0.25), locked=True),
            Param("fin_tip", "Fin tip chord", "fins", "m", 0.0, 5.0, 0.084, (0.04, 0.12), locked=True),
            Param("fin_span", "Fin span (exposed)", "fins", "m", 0.005, 5.0, 0.13, (0.09, 0.16), locked=True),
            Param("fin_sweep", "Fin sweep (leading edge)", "fins", "deg", 0.0, 75.0, 40.0, (25.0, 55.0), locked=True),
            Param("fin_thickness", "Fin thickness", "fins", "m", 0.0005, 0.2, 0.0072, (0.004, 0.01), locked=True),
        ],
    },
    "uav": {
        "title": "Fixed-wing UAV", "category": "fixed-wing", "help": "Pod-and-boom fuselage, tapered wing, conventional tail.",
        # subsonic: the fuselage model (Hoerner) only sees wetted area and fineness, so it should not pick the profile
        "params": _nose_params("elliptical", 0.252, (0.15, 0.4), family_locked=True) + [
            Param("length", "Fuselage length", "body", "m", 0.1, 30.0, 1.4, (1.2, 1.6), locked=True),
            Param("diameter", "Max diameter", "body", "m", 0.01, 5.0, 0.14, (0.12, 0.16)),
            Param("tail_length", "Taper length", "tail", "m", 0.0, 20.0, 0.77, (0.6, 0.95),
                  help="Length over which the pod tapers into the tail boom."),
            Param("tail_taper", "Boom / max diameter", "tail", "ratio", 0.1, 1.0, 0.23, (0.18, 0.4), locked=True),
        ] + _wing_params(2.0, 0.30, 0.165, 3.0, x_frac=0.25, z_frac=0.5) + [
            Param("htail_span", "H-tail span", "tail surfaces", "m", 0.02, 10.0, 0.62, (0.5, 0.75), locked=True),
            Param("htail_chord", "H-tail root chord", "tail surfaces", "m", 0.01, 2.0, 0.16, (0.12, 0.2), locked=True),
            Param("htail_x", "H-tail position", "tail surfaces", "ratio", 0.5, 1.1, 0.873, (0.82, 0.9), locked=True,
                  help="Root leading edge from the nose, as a fraction of fuselage length."),
            Param("htail_incidence", "H-tail incidence", "tail surfaces", "deg", -8.0, 8.0, 0.0, (-3.0, 3.0), locked=True),
            Param("vtail_height", "Fin height", "tail surfaces", "m", 0.02, 5.0, 0.36, (0.25, 0.45), locked=True,
                  help="From the boom axis to the fin tip."),
            Param("vtail_chord", "Fin root chord", "tail surfaces", "m", 0.01, 2.0, 0.18, (0.14, 0.22), locked=True),
        ],
    },
    "wing": {
        "title": "Wing", "category": "fixed-wing", "help": "A tapered wing on its own: planform and NACA section.",
        "params": _wing_params(1.0, 0.25, 0.125, 5.0, dihedral=0.0),
    },
}


def get(tid):
    if tid not in TEMPLATES:
        raise ValueError(f"unknown template {tid!r}")
    return TEMPLATES[tid]


def catalog():
    return {k: {"title": t["title"], "category": t["category"], "help": t["help"],
                "params": [p.public() for p in t["params"]]} for k, t in TEMPLATES.items()}


def defaults(tid):
    return {p.key: p.default for p in get(tid)["params"]}


def clean(tid, values):
    """Values clipped to hard limits, choices validated, missing keys defaulted."""
    out = defaults(tid)
    for p in get(tid)["params"]:
        if p.key not in (values or {}):
            continue
        v = values[p.key]
        if p.kind == "choice":
            if str(v) in [c[0] for c in p.choices]:
                out[p.key] = str(v)
        else:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out[p.key] = min(max(f, p.lo), p.hi)
    return out


def active(p, values):
    """Whether a parameter matters for these values (e.g. Haack C only for Haack noses)."""
    return not p.show_if or all(str(values.get(k)) in [str(x) for x in vs] for k, vs in p.show_if.items())


# ---- profiles ------------------------------------------------------------------------------
def _nose_r(family, x, L, R, n=0.6, C=0.0):
    x = np.clip(np.asarray(x, float), 0.0, L)
    if family == "conical":
        return R * x / L
    if family == "ogive":
        rho = (R * R + L * L) / (2 * R)
        return np.sqrt(np.maximum(rho * rho - (L - x) ** 2, 0.0)) + R - rho
    if family == "elliptical":
        return R * np.sqrt(np.maximum(x * (2 * L - x), 0.0)) / L
    if family == "power":
        return R * (x / L) ** n
    if family == "haack":
        th = np.arccos(np.clip(1 - 2 * x / L, -1.0, 1.0))
        return R / math.sqrt(math.pi) * np.sqrt(np.maximum(th - np.sin(2 * th) / 2 + C * np.sin(th) ** 3, 0.0))
    raise ValueError(f"unknown nose profile {family!r}")


def nose_profile(family, L, R, tip_radius=0.0, n_pts=40, power=0.6, haack_c=0.0):
    """(x, r) from the tip to the shoulder, with an optional spherical tip tangent to the profile.
    Returns (x, r, problem) -- problem is a message when the tip radius cannot be fitted."""
    s = np.linspace(0.0, 1.0, n_pts)
    xs = L * (1.0 - np.cos(0.5 * math.pi * s))          # clustered at the tip
    r = _nose_r(family, xs, L, R, power, haack_c)
    if tip_radius <= 1e-9:
        return xs, r, None
    # tangency: the normal from (xt, r(xt)) meets the axis at xc = xt + r r'; sphere radius r sqrt(1 + r'^2)
    xg = L * (1.0 - np.cos(0.5 * math.pi * np.linspace(1e-4, 1.0, 600)))
    h = 1e-6 * L
    rg = _nose_r(family, xg, L, R, power, haack_c)
    dr = (_nose_r(family, np.minimum(xg + h, L), L, R, power, haack_c) - _nose_r(family, np.maximum(xg - h, 0), L, R, power, haack_c)) \
        / (np.minimum(xg + h, L) - np.maximum(xg - h, 0))
    rn = rg * np.sqrt(1 + dr * dr)
    if tip_radius <= rn[0]:
        return xs, r, None                              # the profile is already this blunt
    k = int(np.argmax(rn >= tip_radius)) if np.any(rn >= tip_radius) else -1
    if k <= 0:
        return xs, r, "tip radius too large for this nose"
    f = (tip_radius - rn[k - 1]) / max(rn[k] - rn[k - 1], 1e-30)
    xt = xg[k - 1] + f * (xg[k] - xg[k - 1])
    rt = float(_nose_r(family, xt, L, R, power, haack_c))
    drt = float(dr[k - 1] + f * (dr[k] - dr[k - 1]))
    xc = xt + rt * drt
    phi_t = math.acos(np.clip((xc - xt) / tip_radius, -1.0, 1.0))
    phi = np.linspace(0.0, phi_t, 12)
    cap_x = xc - tip_radius * np.cos(phi)
    cap_r = tip_radius * np.sin(phi)
    keep = xs > xt + 1e-9 * L
    return np.concatenate([cap_x, xs[keep]]), np.concatenate([cap_r, r[keep]]), None


def tail_profile(x0, length, R, Rb, n=16):
    """Smooth taper from R to Rb (zero slope at both ends)."""
    t = np.linspace(0.0, 1.0, n)
    return x0 + t * length, R - (R - Rb) * (3 * t * t - 2 * t ** 3)


# ---- meshing -----------------------------------------------------------------------------
_RECORD = None      # list while a smooth-CAD export records the sections every part is lofted through


def loft_rings(rings, cap_start=True, cap_end=True):
    """Closed shell through rings (S, M, 3); rings of zero size collapse to a point. Vectorised."""
    rings = np.asarray(rings, float)
    if _RECORD is not None:
        _RECORD.append(("smooth", rings.copy()))
    S, M, _ = rings.shape
    verts, idx = [], []
    n = 0
    for s in range(S):
        c = rings[s].mean(axis=0)
        if np.abs(rings[s] - c).max() < 1e-12:
            verts.append(c[None, :])
            idx.append(np.full(M, n))
            n += 1
        else:
            verts.append(rings[s])
            idx.append(np.arange(n, n + M))
            n += M
    faces = []
    k = np.arange(M)
    k1 = (k + 1) % M
    for s in range(S - 1):
        a, b = idx[s], idx[s + 1]
        pa, pb = a[0] == a[1], b[0] == b[1]
        if pa and pb:
            continue
        if pa:
            faces.append(np.stack([a[k], b[k1], b[k]], 1))
        elif pb:
            faces.append(np.stack([a[k], a[k1], b[k]], 1))
        else:
            faces.append(np.stack([a[k], a[k1], b[k1]], 1))
            faces.append(np.stack([a[k], b[k1], b[k]], 1))
    for s, cap, flip in ((0, cap_start, True), (S - 1, cap_end, False)):
        ring = idx[s]
        if cap and ring[0] != ring[1]:
            verts.append(rings[s].mean(axis=0)[None, :])
            c = n
            n += 1
            faces.append(np.stack([np.full(M, c), ring[k1], ring[k]], 1) if flip else np.stack([np.full(M, c), ring[k], ring[k1]], 1))
    v = np.vstack(verts)
    f = np.vstack(faces)
    m = Mesh(v, f)
    if m.signed_volume() < 0:
        m = Mesh(v, f[:, ::-1])
    return m


def revolve(xs, rs, M=48, yc=0.0, zc=0.0):
    th = np.linspace(0.0, 2 * math.pi, M, endpoint=False)
    rings = np.stack([np.stack([np.full(M, x), yc + r * np.cos(th), zc + r * np.sin(th)], 1) for x, r in zip(xs, rs)])
    return loft_rings(rings)


def wing_mesh(surf: LiftingSurface, n_af=40, n_span=14, half=False):
    """Solid lofted through NACA sections at the planform stations; tips closed."""
    sec = surf.section
    xa, ya = naca.coordinates(sec.get("family", "naca4"), sec.get("m", 0.0), sec.get("p", 0.4), sec.get("t", 0.12),
                              sec.get("cl_i", 0.3), n=n_af, closed_te=True)
    if half:
        eta = np.sin(np.linspace(0.0, math.pi / 2, n_span + 1))
    else:
        eta = np.sin(np.linspace(-math.pi / 2, math.pi / 2, 2 * n_span + 1))
    le, ch, _tw, ec, en = surf.station(eta)
    rings = le[:, None, :] + (xa[None, :, None] * ec[:, None, :] + ya[None, :, None] * en[:, None, :]) * ch[:, None, None]
    return loft_rings(rings)


def _prism(poly_xr, thickness, angle_deg):
    """Fin: planform polygon in (x, radial) extruded by `thickness`, rotated about x."""
    P = np.asarray(poly_xr, float)
    n = len(P)
    a = math.radians(angle_deg)
    er = np.array([0.0, math.cos(a), math.sin(a)])
    et = np.array([0.0, -math.sin(a), math.cos(a)])
    base = P[:, 0:1] * np.array([1.0, 0, 0]) + P[:, 1:2] * er
    top = base + 0.5 * thickness * et
    bot = base - 0.5 * thickness * et
    v = np.vstack([bot, top, bot.mean(0, keepdims=True), top.mean(0, keepdims=True)])
    cb, ct = 2 * n, 2 * n + 1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    m = Mesh(v, np.array(f))
    if _RECORD is not None:
        _RECORD.append(("ruled", np.stack([bot, top])))
    return m if m.signed_volume() > 0 else Mesh(v, np.array(f)[:, ::-1])


def _merge(parts):
    v, f, g = [], [], []
    off = 0
    for m, grp in parts:
        v.append(m.v)
        f.append(m.f + off)
        g.append(np.full(len(m.f), grp, dtype=np.int8))
        off += len(m.v)
    return Mesh(np.vstack(v), np.vstack(f)), np.concatenate(g)


@dataclass
class Built:
    mesh: Mesh
    groups: np.ndarray
    info: dict
    problems: list
    surfaces: list = field(default_factory=list)       # LiftingSurface for the wing model
    drag_surfaces: list = field(default_factory=list)  # DragSurface (fins with no lift)
    ref: dict = field(default_factory=dict)            # ref_area, ref_length, x_ref (file frame)


def _section(v):
    if v.get("airfoil", "naca4") == "naca5":
        return {"family": "naca5", "m": 0.0, "p": float(v["wing_camber_pos5"]), "t": float(v["wing_thickness"]),
                "cl_i": float(v["wing_cl_design"])}
    return {"family": "naca4", "m": float(v["wing_camber"]), "p": float(v["wing_camber_pos"]), "t": float(v["wing_thickness"]),
            "cl_i": 0.3}


def _fuselage(v, M, n_nose, problems):
    L, D = float(v["length"]), float(v["diameter"])
    R = D / 2
    Ln, Lt = float(v["nose_length"]), float(v["tail_length"])
    if Ln + Lt > 0.98 * L:
        problems.append("nose and tail are longer than the body")
        Lt = max(0.0, 0.98 * L - Ln)
    xs, rs, pr = nose_profile(v["nose_family"], Ln, R, float(v["tip_radius"]), n_nose, float(v["nose_power"]), float(v["haack_c"]))
    if pr:
        problems.append(pr)
    if float(v["tip_radius"]) > 0.9 * R:
        problems.append("tip radius is larger than the body")
    Rb = R * float(v["tail_taper"])
    xb = np.linspace(Ln, L - Lt, max(2, int(round((L - Lt - Ln) / max(D, 1e-9))) + 1))[1:]
    xt, rt = tail_profile(L - Lt, Lt, R, Rb) if Lt > 1e-9 else (np.array([L]), np.array([R]))
    X = np.concatenate([xs, xb, xt[1:] if Lt > 1e-9 else []])
    Rr = np.concatenate([rs, np.full(len(xb), R), rt[1:] if Lt > 1e-9 else []])
    keep = np.concatenate([[True], np.diff(X) > 1e-9 * L])
    X, Rr = X[keep], Rr[keep]
    body = revolve(X, Rr, M)
    info = {"length": L, "diameter": D, "nose_fineness": Ln / D, "body_fineness": L / D, "base_diameter": 2 * Rb,
            "x_tip": float(X[0]), "body_wetted": float(body.area() - math.pi * Rb * Rb)}
    return body, info, R


def cad_sections(tid, values):
    """The sections every part of the template is lofted through, for a smooth CAD export:
    [(kind, rings)], kind "smooth" (B-spline sections) or "ruled" (flat-sided prisms)."""
    global _RECORD
    _RECORD = []
    try:
        build(tid, values)
        return _RECORD
    finally:
        _RECORD = None


def build(tid, values, preview=False):
    """Values -> Built. Problems (impossible combinations) are reported, not raised."""
    v = clean(tid, values)
    problems = []
    M = 32 if preview else 48
    n_nose = 24 if preview else 40
    n_af = 24 if preview else 40
    n_span = 8 if preview else 14
    parts = []
    info = {}
    surfaces, drag_surfaces = [], []
    ref = {}
    if tid in ("rocket", "uav"):
        body, info, R = _fuselage(v, M, n_nose, problems)
        parts.append((body, BODY))
        L = info["length"]
    if tid == "rocket":
        Lt = float(v["tail_length"])
        x_te = L - Lt
        cr, ct, s, tf = (float(v[k]) for k in ("fin_root", "fin_tip", "fin_span", "fin_thickness"))
        sweep = math.tan(math.radians(float(v["fin_sweep"])))
        x_le = x_te - cr
        if x_le < float(v["nose_length"]):
            problems.append("fins reach onto the nose")
        r_in = 0.6 * R
        poly = [(x_le, r_in), (x_te, r_in), (x_te, R), (x_le + s * sweep + ct, R + s), (x_le + s * sweep, R + s), (x_le, R)]
        if ct < 1e-6:
            poly = [(x_le, r_in), (x_te, r_in), (x_te, R), (x_le + s * sweep, R + s), (x_le, R)]
        nf = int(v["fin_count"])
        a0 = 45.0 if nf == 4 else 90.0
        for k in range(nf):
            parts.append((_prism(poly, tf, a0 + k * 360.0 / nf), FINS))
        info.update({"fin_count": nf, "min_thickness": tf, "fin_area_each": 0.5 * s * (cr + ct)})
        ref = {"ref_area": math.pi * R * R, "ref_length": L, "x_ref": [0.5 * L, 0.0, 0.0]}
    if tid in ("uav", "wing"):
        sec = _section(v)
        b, cr, ct = float(v["wing_span"]), float(v["wing_root"]), float(v["wing_tip"])
        if ct > 1.5 * cr:
            problems.append("tip chord is much larger than the root chord")
        if tid == "uav":
            if b < 2.5 * R:
                problems.append("wing span is smaller than the fuselage")
            x_le = float(v["wing_x"]) * L
            z_le = float(v["wing_z"]) * R
            if x_le + cr > L:
                problems.append("wing overhangs the tail")
            exposed = R * math.sqrt(max(0.0, 1 - float(v["wing_z"]) ** 2))
        else:
            x_le, z_le, exposed = 0.0, 0.0, 0.0
        wing = LiftingSurface("wing", (x_le, 0.0, z_le), b, cr, ct, float(v["wing_sweep"]), float(v["wing_dihedral"]),
                              float(v["wing_incidence"]), float(v["wing_twist"]), sec, exposed_from=exposed)
        parts.append((wing_mesh(wing, n_af, n_span), WING))
        surfaces.append(wing)
        S, mac = wing.area(), wing.mac()
        lam = ct / cr
        y_mac = b / 6 * (1 + 2 * lam) / (1 + lam)
        x_ac = x_le + y_mac * math.tan(math.radians(float(v["wing_sweep"]))) + 0.25 * mac
        name = naca.describe(sec["family"], sec["m"], sec["p"], sec["t"], sec.get("cl_i", 0.3))
        info.update({"wing_area": S, "aspect_ratio": b * b / S, "mac": mac, "taper_ratio": lam, "span": b,
                     "airfoil": name, "airfoil_exact": naca.designation(sec["family"], sec["m"], sec["p"], sec["t"], sec.get("cl_i", 0.3))[1],
                     "wing_x_ac": x_ac, "min_thickness": sec["t"] * ct})
        ref = {"ref_area": S, "ref_length": mac, "x_ref": [x_ac, 0.0, z_le]}
    if tid == "uav":
        Rb = R * float(v["tail_taper"])
        hs, hc = float(v["htail_span"]), float(v["htail_chord"])
        hx = float(v["htail_x"]) * L
        ht = LiftingSurface("horizontal tail", (hx, 0.0, 0.0), hs, hc, 0.7 * hc, 6.0, 0.0, float(v["htail_incidence"]), 0.0,
                            {"family": "naca4", "m": 0.0, "p": 0.4, "t": 0.10}, exposed_from=Rb, ns=24)
        parts.append((wing_mesh(ht, max(16, n_af // 2), max(6, n_span // 2)), HTAIL))
        surfaces.append(ht)
        vh, vc = float(v["vtail_height"]), float(v["vtail_chord"])
        fin = LiftingSurface("vertical tail", (hx - 0.1 * vc, 0.0, 0.0), 2 * vh, vc, 0.55 * vc, 18.0, 0.0, 0.0, 0.0,
                             {"family": "naca4", "m": 0.0, "p": 0.4, "t": 0.10})
        fm = wing_mesh(fin, max(16, n_af // 2), max(6, n_span // 2), half=True)
        # the half wing grows along +y: turn it to stand on the boom (span along +z)
        Rx = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        fm = Mesh(fm.v @ Rx.T, fm.f)
        fm = fm.transformed(t=[0.0, 0.0, -0.5 * Rb])
        if _RECORD is not None:                       # the recorded fin sections move with it
            kind, rings = _RECORD[-1]
            _RECORD[-1] = (kind, rings @ Rx.T + np.array([0.0, 0.0, -0.5 * Rb]))
        parts.append((fm, VTAIL))
        drag_surfaces.append(DragSurface("vertical tail", vh, vc, 0.55 * vc, exposed_from=Rb))
        info["min_thickness"] = min(info["min_thickness"], 0.10 * 0.7 * hc, 0.10 * 0.55 * vc)
        info["tail_volume_h"] = (hs * hc * 0.85) * (hx + 0.25 * hc - info["wing_x_ac"]) / (info["wing_area"] * info["mac"])
    mesh, groups = _merge(parts)
    lo, hi = mesh.bbox()
    vol_body = abs(parts[0][0].signed_volume())
    info.update({"overall_length": float(hi[0] - lo[0]), "overall_width": float(hi[1] - lo[1]),
                 "overall_height": float(hi[2] - lo[2]), "volume": vol_body,
                 "wetted_area": float(sum(m.area() for m, _ in parts))})
    return Built(mesh, groups, info, problems, surfaces, drag_surfaces, ref)
