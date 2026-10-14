"""Shape features: named, adjustable edits stacked on any model, like a CAD timeline.

A triangle mesh on its own can only be bent as a whole (geometry/ffd.py). Features add the
operations a designer actually reaches for, each with named dimensions that the optimizer can
also drive:

  add       nose cone, tail cone, pod / nacelle, wing or tail surface, fin set,
            fairing over an exposed part (the convex hull of the part, drawn out into a
            rounded or conical nose and tail)
  cut       slant: a plane cut, capped flat (upswept tails, chamfered noses, wedge edges)
  modify    wing panel: span, chord, thickness, sweep and dihedral of a wing on an upload;
            part: move, scale or turn whatever lies inside a region, blended into the rest

Features are applied in order. Added parts are separate closed shells that overlap the body:
the wind tunnel fills the union of all shells, and faces buried inside another shell are
marked hidden (mesh.hidden) so they never count as wetted surface or carry surface pressure.

Placement follows the protected regions: `center` (and `size` for features that act on a
region) in the model file's own coordinates and units, `rotation` (roll, pitch, yaw in degrees
about the airframe axes). Every other value is SI (metres, degrees, ratios) in feature["values"].
"""
from __future__ import annotations

import math
import uuid

import numpy as np
from numba import njit

from . import templates as T
from .mesh import Mesh, shell_info
from .templates import AIRFOILS, NOSE_FAMILIES, Param
from .zones import Zone, _inside_counts, rotation_matrix

FINS_CHOICES = [("2", "2"), ("3", "3"), ("4", "4"), ("6", "6")]

# ---- the catalog --------------------------------------------------------------------------------
FEATURES = {
    "nosecone": {
        "title": "Nose cone", "kind": "add", "icon": "rocket", "region": False,
        "help": "A new nose on the front of a body: profile family, length and diameter. Click where its base sits.",
        "params": [
            Param("section", "Cross-section", "shape", "choice", default="round", choices=[("round", "Round"), ("match", "Follow the body's own section")],
                  help="Round, or the body's own cross-section where the feature starts (a car's rear, a box-shaped frame), tapered."),
            Param("family", "Profile", "shape", "choice", default="ogive", choices=NOSE_FAMILIES),
            Param("length", "Length", "shape", "m", 0.002, 50.0, 0.1, locked=False, help="Tip to base."),
            Param("diameter", "Base diameter", "shape", "m", 0.001, 20.0, 0.05, show_if={"section": ["round"]}),
            Param("tip_radius", "Tip radius", "shape", "m", 0.0, 5.0, 0.0, help="Spherical blunting tangent to the profile."),
            Param("power", "Power-law exponent", "shape", "-", 0.25, 1.0, 0.6, show_if={"family": ["power"]}),
            Param("haack_c", "Haack C", "shape", "-", 0.0, 0.34, 0.0, show_if={"family": ["haack"]}),
            Param("overlap", "Shoulder into the body", "fit", "m", 0.0, 10.0, 0.01,
                  help="Straight section behind the base that sinks into the body, so the two join without a gap."),
        ],
    },
    "tailcone": {
        "title": "Tail cone", "kind": "add", "icon": "rocket", "region": False,
        "help": "A boat-tail or cone on the back of a body, tapering from the base diameter. Click where it starts.",
        "params": [
            Param("section", "Cross-section", "shape", "choice", default="round", choices=[("round", "Round"), ("match", "Follow the body's own section")],
                  help="Round, or the body's own cross-section where the feature starts (a car's rear, a box-shaped frame), tapered."),
            Param("shape", "Shape", "shape", "choice", default="smooth", choices=[("smooth", "Smooth (tangent)"), ("conical", "Conical")]),
            Param("length", "Length", "shape", "m", 0.002, 50.0, 0.1, locked=False),
            Param("diameter", "Start diameter", "shape", "m", 0.001, 20.0, 0.05, show_if={"section": ["round"]}),
            Param("end_ratio", "End / start size", "shape", "ratio", 0.0, 1.0, 0.4, locked=False,
                  help="0 ends in a point; 1 is a straight extension."),
            Param("overlap", "Overlap into the body", "fit", "m", 0.0, 10.0, 0.01),
        ],
    },
    "fairing": {
        "title": "Fairing over a part", "kind": "add", "icon": "shieldlock", "region": True,
        "help": "Encloses whatever lies inside the box (a motor, camera, bolt heads, landing gear) in a smooth "
                "shell drawn out into a nose and a tail, so the flow sees one streamlined shape.",
        "params": [
            Param("profile", "Shape", "shape", "choice", default="ogive", choices=[("ogive", "Rounded nose and tail"), ("cone", "Straight cones")]),
            Param("front_length", "Nose length", "shape", "m", 0.0, 50.0, 0.05, locked=False, help="How far it reaches ahead of the part."),
            Param("rear_length", "Tail length", "shape", "m", 0.0, 50.0, 0.1, locked=False,
                  help="How far it reaches behind the part: a long tail cuts the separated wake."),
            Param("clearance", "Clearance", "fit", "m", 0.0, 1.0, 0.002, help="Gap kept around the part."),
            Param("tip_radius", "Tip radius", "shape", "m", 0.0, 5.0, 0.0, help="0 is a sharp point at each end."),
        ],
    },
    "pod": {
        "title": "Pod / nacelle", "kind": "add", "icon": "box", "region": False,
        "help": "A streamlined body of revolution: payload pod, motor nacelle, sensor pod or tank.",
        "params": [
            Param("length", "Length", "shape", "m", 0.005, 50.0, 0.2),
            Param("diameter", "Diameter", "shape", "m", 0.001, 20.0, 0.05),
            Param("nose_family", "Nose profile", "shape", "choice", default="elliptical", choices=NOSE_FAMILIES),
            Param("nose_length", "Nose length", "shape", "m", 0.0, 50.0, 0.05),
            Param("tail_length", "Tail length", "shape", "m", 0.0, 50.0, 0.08),
            Param("tail_ratio", "Tail end / diameter", "shape", "ratio", 0.0, 1.0, 0.2),
            Param("offset_x", "Move along the flow", "position", "m", -50.0, 50.0, 0.0),
            Param("offset_z", "Move up / down", "position", "m", -50.0, 50.0, 0.0),
        ],
    },
    "wing": {
        "title": "Wing / tail surface", "kind": "add", "icon": "plane", "region": False,
        "help": "A NACA-section lifting surface from its root leading edge: a wing, canard or stabiliser; "
                "one side and a 90° roll makes a vertical fin.",
        "params": [
            Param("sides", "Sides", "planform", "choice", default="pair", choices=[("pair", "Left and right"), ("single", "One side")]),
            Param("span", "Span", "planform", "m", 0.01, 50.0, 0.6, help="Tip to tip (one side: root to tip)."),
            Param("root", "Root chord", "planform", "m", 0.005, 10.0, 0.15),
            Param("tip", "Tip chord", "planform", "m", 0.002, 10.0, 0.08),
            Param("sweep", "Sweep (leading edge)", "planform", "deg", -20.0, 60.0, 5.0),
            Param("dihedral", "Dihedral", "planform", "deg", -30.0, 30.0, 0.0),
            Param("twist", "Twist (tip)", "planform", "deg", -8.0, 4.0, 0.0),
            Param("incidence", "Incidence", "planform", "deg", -10.0, 10.0, 0.0),
            Param("airfoil", "Airfoil family", "section", "choice", default="naca4", choices=AIRFOILS),
            Param("thickness", "Thickness", "section", "ratio", 0.03, 0.24, 0.12),
            Param("camber", "Camber", "section", "ratio", 0.0, 0.09, 0.02, show_if={"airfoil": ["naca4"]}),
            Param("camber_pos", "Camber position", "section", "ratio", 0.15, 0.7, 0.4, show_if={"airfoil": ["naca4"]}),
            Param("cl_design", "Design lift coefficient", "section", "-", 0.05, 0.9, 0.3, show_if={"airfoil": ["naca5"]}),
            Param("camber_pos5", "Camber position", "section", "ratio", 0.05, 0.25, 0.15, show_if={"airfoil": ["naca5"]}),
            Param("offset_x", "Move along the flow", "position", "m", -50.0, 50.0, 0.0),
            Param("offset_z", "Move up / down", "position", "m", -50.0, 50.0, 0.0),
        ],
    },
    "fins": {
        "title": "Fin set", "kind": "add", "icon": "rocket", "region": False,
        "help": "Flat fins spaced evenly around the body axis. Click the body where the fins' leading edge goes.",
        "params": [
            Param("count", "Number of fins", "planform", "choice", default="4", choices=FINS_CHOICES),
            Param("root", "Root chord", "planform", "m", 0.005, 10.0, 0.1),
            Param("tip", "Tip chord", "planform", "m", 0.0, 10.0, 0.04),
            Param("span", "Span (exposed)", "planform", "m", 0.005, 10.0, 0.06),
            Param("sweep", "Sweep (leading edge)", "planform", "deg", 0.0, 75.0, 40.0),
            Param("thickness", "Thickness", "planform", "m", 0.0005, 0.5, 0.004),
            Param("roll", "Roll angle", "position", "deg", -180.0, 180.0, 45.0),
            Param("root_radius", "Body radius at the root", "position", "m", 0.0, 20.0, 0.025,
                  help="Distance from the axis to where each fin meets the body."),
            Param("offset_x", "Move along the flow", "position", "m", -50.0, 50.0, 0.0),
        ],
    },
    "slant": {
        "title": "Slant cut", "kind": "cut", "icon": "sliders", "region": False,
        "help": "Cuts the body flat along a plane and closes it: an upswept tail, a chamfered nose, a wedge. "
                "Click the surface to cut; the cut starts square to the surface there.",
        "params": [
            Param("depth", "Depth", "cut", "m", -50.0, 50.0, 0.0, locked=False,
                  help="How far the plane moves into the body from the point you clicked."),
            Param("elevation", "Tilt (up / down)", "cut", "deg", -90.0, 90.0, 0.0,
                  help="Angle of the cut's outward direction above the horizontal."),
            Param("azimuth", "Heading (around)", "cut", "deg", -180.0, 180.0, 0.0,
                  help="Direction of the cut's outward side around the vertical axis: 0 aft, 90 to the left (+y), 180 forward."),
            Param("parts", "Cuts", "cut", "choice", default="nearest", choices=[("nearest", "Only the part clicked"), ("all", "Every part")]),
        ],
    },
    "wingpanel": {
        "title": "Adjust a wing", "kind": "modify", "icon": "plane", "region": True,
        "help": "Reshapes a wing that is part of your upload: put the box around both wings (or one panel) "
                "and change span, chord, thickness, sweep and dihedral. The root stays attached.",
        "params": [
            Param("sides", "Wings in the box", "fit", "choice", default="pair", choices=[("pair", "Both, mirrored"), ("single", "One panel (root at -y)")]),
            Param("span_scale", "Span", "planform", "ratio", 0.4, 2.5, 1.0, locked=False),
            Param("chord_scale", "Chord", "planform", "ratio", 0.4, 2.5, 1.0),
            Param("thickness_scale", "Thickness / chord", "planform", "ratio", 0.3, 2.5, 1.0),
            Param("sweep_add", "Extra sweep", "planform", "deg", -30.0, 30.0, 0.0),
            Param("dihedral_add", "Extra dihedral", "planform", "deg", -20.0, 20.0, 0.0),
            Param("root_offset", "Fuselage half-width", "fit", "m", 0.0, 20.0, 0.02,
                  help="Distance from the centre of the box where the wing leaves the body: nothing inside it moves."),
            Param("blend", "Blend distance", "fit", "m", 0.0, 10.0, 0.02, help="How far the change fades into the surrounding surface."),
        ],
    },
    "part": {
        "title": "Move / scale a part", "kind": "modify", "icon": "move", "region": True,
        "help": "Moves, scales or turns whatever lies inside the box (a canopy, a pod, a fin, a sensor), "
                "fading the change into the surface around it.",
        "params": [
            Param("move_x", "Move along x", "move", "m", -50.0, 50.0, 0.0),
            Param("move_y", "Move along y", "move", "m", -50.0, 50.0, 0.0),
            Param("move_z", "Move along z", "move", "m", -50.0, 50.0, 0.0),
            Param("scale_x", "Scale x", "scale", "ratio", 0.2, 4.0, 1.0),
            Param("scale_y", "Scale y", "scale", "ratio", 0.2, 4.0, 1.0),
            Param("scale_z", "Scale z", "scale", "ratio", 0.2, 4.0, 1.0),
            Param("turn_roll", "Roll", "turn", "deg", -180.0, 180.0, 0.0),
            Param("turn_pitch", "Pitch", "turn", "deg", -90.0, 90.0, 0.0),
            Param("turn_yaw", "Yaw", "turn", "deg", -180.0, 180.0, 0.0),
            Param("blend", "Blend distance", "fit", "m", 0.0, 10.0, 0.02),
        ],
    },
}
MAX_FEATURES = 24
# the dimensions the optimizer may change until the user says otherwise; everything else starts locked
FREE_BY_DEFAULT = {
    "nosecone": {"length"}, "tailcone": {"length", "end_ratio"}, "fairing": {"front_length", "rear_length"},
    "pod": set(), "wing": set(), "fins": set(), "slant": {"depth"}, "wingpanel": {"span_scale"}, "part": set(),
}
for _t, _d in FEATURES.items():
    for _p in _d["params"]:
        _p.locked = _p.key not in FREE_BY_DEFAULT[_t]


def catalog():
    return {k: {"title": d["title"], "kind": d["kind"], "icon": d["icon"], "help": d["help"], "region": d["region"],
                "params": [p.public() for p in d["params"]]} for k, d in FEATURES.items()}


def get(ftype):
    if ftype not in FEATURES:
        raise ValueError(f"unknown feature type {ftype!r}")
    return FEATURES[ftype]


def defaults(ftype):
    return {p.key: p.default for p in get(ftype)["params"]}


def _num3(v, what):
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ValueError(f"invalid {what}")
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise ValueError(f"invalid {what}")
    return out


def clean(f):
    """Validate one feature as sent by the client; raises ValueError."""
    if not isinstance(f, dict):
        raise ValueError("invalid feature")
    d = get(f.get("type"))
    vals = defaults(f["type"])
    for p in d["params"]:
        x = (f.get("values") or {}).get(p.key, vals[p.key])
        if p.kind == "choice":
            x = str(x)
            if x not in [c[0] for c in p.choices]:
                x = p.default
        else:
            try:
                x = float(x)
            except (TypeError, ValueError):
                raise ValueError(f"invalid value for {p.label}")
            if not math.isfinite(x):
                raise ValueError(f"invalid value for {p.label}")
            x = min(max(x, p.lo), p.hi)
        vals[p.key] = x
    params = {}
    for k, cfg in (f.get("params") or {}).items():
        p = next((q for q in d["params"] if q.key == k), None)
        if p is None or not isinstance(cfg, dict):
            continue
        c = {}
        if "locked" in cfg:
            c["locked"] = bool(cfg["locked"])
        if cfg.get("bounds") is not None and p.kind != "choice":
            lo, hi = sorted(float(b) for b in cfg["bounds"][:2])
            c["bounds"] = [max(lo, p.lo), min(hi, p.hi)]
        params[k] = c
    out = {"id": str(f.get("id") or uuid.uuid4().hex[:8])[:16], "type": f["type"], "name": str(f.get("name") or d["title"])[:60],
           "enabled": f.get("enabled", True) is not False, "center": _num3(f.get("center"), "feature position"),
           "values": vals, "params": params}
    if f.get("optional"):
        out["optional"] = True             # the optimizer tries the design with and without it
    if f.get("why"):
        out["why"] = str(f["why"])[:300]   # the reasoning behind an AI or generated idea
    if d["region"]:
        size = _num3(f.get("size"), "feature size")
        if min(size) <= 0:
            raise ValueError("a feature region needs a positive size on all three axes")
        out["size"] = size
    if f.get("rotation") is not None:
        rot = _num3(f["rotation"], "feature rotation")
        if max(abs(a) for a in rot) > 360:
            raise ValueError("a rotation is three angles in degrees")
        if any(abs(a) > 1e-9 for a in rot):
            out["rotation"] = [round(((a + 180.0) % 360.0) - 180.0, 4) for a in rot]
    return out


def locked(p, f):
    c = (f.get("params") or {}).get(p.key) or {}
    return c["locked"] if "locked" in c else p.locked


def bounds(p, f):
    c = (f.get("params") or {}).get(p.key) or {}
    if c.get("bounds"):
        return tuple(c["bounds"])
    v = f["values"][p.key]
    if p.unit == "deg":
        return max(p.lo, v - 10.0), min(p.hi, v + 10.0)
    if p.unit in ("m",) and p.key.startswith(("offset", "move", "depth")):
        span = max(abs(v), 0.02)
        return max(p.lo, v - span), min(p.hi, v + span)
    if v > 0:
        return max(p.lo, 0.75 * v), min(p.hi, 1.3 * v)
    return p.lo, p.hi


# ---- placement ------------------------------------------------------------------------------------
class Frame:
    """File coordinates and units -> the body frame (metres, nose -x, z up, centred), as
    optimize.runner.ModelFrame; built from (scale, R0, centre) so templates can share it."""

    def __init__(self, scale, R0, center):
        self.scale = float(scale)
        self.R0 = np.asarray(R0, float)
        self.center = np.asarray(center, float)

    def point(self, p):
        return self.R0 @ (np.asarray(p, float) * self.scale) - self.center

    def size(self, s):
        return np.abs(self.R0) @ (np.asarray(s, float) * self.scale)


def placement(f, frame: Frame):
    """Body-frame origin, axes (columns) and half extents of a feature."""
    c = frame.point(f["center"])
    R = rotation_matrix(f["rotation"]) if f.get("rotation") else np.eye(3)
    half = frame.size(f["size"]) / 2 if "size" in f else None
    return c, R, half


def _place(mesh, c, R):
    return Mesh(mesh.v @ R.T + c, mesh.f)


# ---- builders: added shells ---------------------------------------------------------------------------
def body_loop(mesh: Mesh, c, R, M=72):
    """The body's own cross-section through c, square to the feature's x axis: the largest loop of
    that slice, resampled to M points by arc length, counter-clockwise in local (y, z)."""
    local = Mesh((mesh.v - c) @ R, mesh.f)
    loops = section_loops(local, 0.0, points=True)
    if not loops:
        return None
    pts = max(loops, key=lambda L: L[1])[5]
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tgt = np.linspace(0.0, s[-1], M, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    out = np.stack([np.interp(tgt, s, closed[:, 0]), np.interp(tgt, s, closed[:, 1])], 1)
    area = 0.5 * np.sum(out[:, 0] * np.roll(out[:, 1], -1) - np.roll(out[:, 0], -1) * out[:, 1])
    return out if area > 0 else out[::-1]


def _section_loft(loop, xs, scales):
    """Closed shell through copies of a (y, z) loop at stations xs, scaled about its centre."""
    cen = loop.mean(0)
    rings = np.stack([np.column_stack([np.full(len(loop), x), cen + (loop - cen) * k]) for x, k in zip(xs, scales)])
    return T.loft_rings(rings)


def _extension(m, c, R, v, forward):
    """Nose or tail that continues the body's own section: forward (-x) or aft (+x)."""
    loop = body_loop(m, c, R)
    if loop is None:
        return None, ["the body has no closed section here to extend; use a round section"]
    L = v["length"]
    ov = max(v["overlap"], 1e-4 * L)
    t = np.linspace(0.0, 1.0, 24)
    if forward:                                   # rounded (elliptical) nose to a point
        k = np.sqrt(np.clip(1.0 - t ** 2, 0.0, 1.0))
        xs = np.concatenate([[ov], -t * L])
        ks = np.concatenate([[1.0], k])
    else:                                         # boat-tail taper to end_ratio
        f = t if v["shape"] == "conical" else 3 * t * t - 2 * t ** 3
        xs = np.concatenate([[-ov], t * L])
        ks = np.concatenate([[1.0], 1.0 - (1.0 - v["end_ratio"]) * f])
    return _section_loft(loop, xs, ks), []


def _nosecone(v, n=48):
    L, Rb = v["length"], v["diameter"] / 2
    xs, rs, pr = T.nose_profile(v["family"], L, Rb, v["tip_radius"], 36, v["power"], v["haack_c"])
    X = np.concatenate([xs - L, [max(v["overlap"], 1e-4 * L)]])
    Rr = np.concatenate([rs, [Rb]])
    return T.revolve(X, Rr, n), [pr] if pr else []


def _tailcone(v, n=48):
    L, R0 = v["length"], v["diameter"] / 2
    R1 = R0 * v["end_ratio"]
    t = np.linspace(0.0, 1.0, 20)
    f = t if v["shape"] == "conical" else 3 * t * t - 2 * t ** 3
    X = np.concatenate([[-max(v["overlap"], 1e-4 * L)], t * L])
    Rr = np.concatenate([[R0], R0 - (R0 - R1) * f])
    return T.revolve(X, Rr, n), []


def _pod(v, n=40):
    L, R = v["length"], v["diameter"] / 2
    Ln, Lt = v["nose_length"], v["tail_length"]
    probs = []
    if Ln + Lt > 0.98 * L:
        probs.append("pod nose and tail are longer than the pod")
        s = 0.98 * L / max(Ln + Lt, 1e-12)
        Ln, Lt = Ln * s, Lt * s
    xs, rs, pr = T.nose_profile(v["nose_family"], max(Ln, 1e-6 * L), R, 0.0, 24)
    xt, rt = T.tail_profile(L - Lt, Lt, R, R * v["tail_ratio"]) if Lt > 1e-9 else (np.array([L]), np.array([R]))
    X = np.concatenate([xs, xt])
    Rr = np.concatenate([rs, rt])
    keep = np.concatenate([[True], np.diff(X) > 1e-9 * L])
    m = T.revolve(X[keep] - L / 2 + v["offset_x"], Rr[keep], n, zc=v["offset_z"])
    return m, probs + ([pr] if pr else [])


def _wing(v):
    from ..aero.vlm import LiftingSurface
    if v["airfoil"] == "naca5":
        sec = {"family": "naca5", "m": 0.0, "p": v["camber_pos5"], "t": v["thickness"], "cl_i": v["cl_design"]}
    else:
        sec = {"family": "naca4", "m": v["camber"], "p": v["camber_pos"], "t": v["thickness"], "cl_i": 0.3}
    single = v["sides"] == "single"
    span = v["span"] * (2.0 if single else 1.0)          # a half wing is half of a pair twice as wide
    s = LiftingSurface("feature wing", (v["offset_x"], 0.0, v["offset_z"]), span, v["root"], v["tip"], v["sweep"],
                       v["dihedral"], v["incidence"], v["twist"], sec)
    probs = ["tip chord is much larger than the root chord"] if v["tip"] > 1.5 * v["root"] else []
    return T.wing_mesh(s, 32, 12, half=single), probs


def _fins(v):
    cr, ct, s = v["root"], v["tip"], v["span"]
    sw = math.tan(math.radians(v["sweep"]))
    r = v["root_radius"]
    r_in = r - min(0.4 * max(r, 1e-6), 0.5 * s)            # sink the root into the body
    x0 = v["offset_x"]
    poly = [(x0, r_in), (x0 + cr, r_in), (x0 + cr, r), (x0 + s * sw + ct, r + s), (x0 + s * sw, r + s), (x0, r)]
    if ct < 1e-6:
        poly = [(x0, r_in), (x0 + cr, r_in), (x0 + cr, r), (x0 + s * sw, r + s), (x0, r)]
    n = int(v["count"])
    parts = [T._prism(poly, v["thickness"], v["roll"] + k * 360.0 / n) for k in range(n)]
    v_all, f_all, off = [], [], 0
    for m in parts:
        v_all.append(m.v)
        f_all.append(m.f + off)
        off += len(m.v)
    return Mesh(np.vstack(v_all), np.vstack(f_all)), []


def _ring(x, cy, cz, ay, az, n=24):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([np.full(n, x), cy + ay * np.cos(th), cz + az * np.sin(th)], 1)


def _fairing(mesh, c, R, half, v):
    """Convex hull of the part inside the box (local coordinates), inflated by the clearance and
    drawn out into a nose ahead and a tail behind along the box's own x axis."""
    from scipy.spatial import ConvexHull, QhullError
    q = (mesh.v - c) @ R
    inside = np.all(np.abs(q) <= half * (1 + 1e-9), axis=1)
    pts = q[inside]
    if len(pts) < 4:
        return None, ["the fairing box holds no part of the model"]
    try:
        hv = pts[ConvexHull(pts).vertices]
    except QhullError:
        hv = pts
    cl = v["clearance"]
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] +
                    [[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    cloud = (hv[:, None, :] + cl * dirs[None]).reshape(-1, 3) if cl > 0 else hv
    lo, hi = cloud.min(0), cloud.max(0)
    cy, cz = (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2
    ay, az = (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2
    extra = []
    for x_end, length, sgn in ((lo[0], v["front_length"], -1.0), (hi[0], v["rear_length"], 1.0)):
        if length <= 1e-9:
            continue
        tip = x_end + sgn * length
        rt = v["tip_radius"]
        if v["profile"] == "ogive":
            # tangent-ogive radius fraction r/R at stations from the tip (t = 0) to the part (t = 1)
            for t in np.linspace(0.12, 0.9, 6):
                k = math.sqrt(max(0.0, 1.0 - (1.0 - t) ** 2))
                extra.append(_ring(tip - sgn * t * length, cy, cz, max(ay * k, rt), max(az * k, rt)))
        if rt > 1e-9:
            extra.append(_ring(tip, cy, cz, rt, rt, 12))
        else:
            extra.append(np.array([[tip, cy, cz]]))
    allp = np.vstack([cloud] + extra)
    try:
        h = ConvexHull(allp)
    except QhullError:
        return None, ["the part inside the fairing box is flat; make the box larger"]
    used = np.unique(h.simplices)
    remap = -np.ones(len(allp), np.int64)
    remap[used] = np.arange(len(used))
    m = Mesh(allp[used], remap[h.simplices])
    m.orient_outward()
    return _place(m, c, R), []


# ---- builders: modifications -------------------------------------------------------------------------
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _region_weight(q, half, blend):
    """1 inside the box (local coordinates), fading to 0 over `blend` outside it."""
    d = np.linalg.norm(np.maximum(np.abs(q) - half, 0.0), axis=1)
    if blend <= 1e-12:
        return (d <= 0).astype(float)
    return 1.0 - _smoothstep(d / blend)


def refine_where(mesh: Mesh, weight, max_edge, max_faces=400_000, passes=4):
    """Split edges longer than max_edge where an edit fades in or out (weight strictly between 0
    and 1 at an end, or very different at the two ends), conformingly: each split edge splits
    both its faces. Inside the edit the change is smooth, so long triangles there are fine."""
    m = mesh
    for _ in range(passes):
        w = weight(m.v)
        f = m.f
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        es = np.sort(e, axis=1)
        uniq, inv = np.unique(es, axis=0, return_inverse=True)
        wa, wb = w[uniq[:, 0]], w[uniq[:, 1]]
        L = np.linalg.norm(m.v[uniq[:, 0]] - m.v[uniq[:, 1]], axis=1)
        fading = ((wa > 1e-6) & (wa < 1 - 1e-6)) | ((wb > 1e-6) & (wb < 1 - 1e-6)) | (np.abs(wa - wb) > 0.25)
        split = (L > max_edge) & fading
        if not split.any() or m.n_faces + 3 * int(split.sum()) > max_faces:
            break
        mid_id = -np.ones(len(uniq), np.int64)
        mid_id[split] = len(m.v) + np.arange(int(split.sum()))
        mids = (m.v[uniq[split, 0]] + m.v[uniq[split, 1]]) / 2
        me = mid_id[inv].reshape(3, -1).T                     # mid of edges ab, bc, ca or -1
        m = Mesh(np.vstack([m.v, mids]), _split_faces(f, me))
    return m


def _split_faces(f, me):
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    ab, bc, ca = me[:, 0], me[:, 1], me[:, 2]
    s1, s2, s3 = ab >= 0, bc >= 0, ca >= 0
    out = []
    pick = lambda sel, *tris: out.extend(np.stack([t[0][sel], t[1][sel], t[2][sel]], 1) for t in tris)
    none = ~s1 & ~s2 & ~s3
    out.append(f[none])
    pick(s1 & ~s2 & ~s3, (a, ab, c), (ab, b, c))
    pick(~s1 & s2 & ~s3, (a, b, bc), (a, bc, c))
    pick(~s1 & ~s2 & s3, (a, b, ca), (ca, b, c))
    pick(s1 & s2 & ~s3, (ab, b, bc), (a, ab, bc), (a, bc, c))
    pick(~s1 & s2 & s3, (ca, bc, c), (a, b, bc), (a, bc, ca))
    pick(s1 & ~s2 & s3, (a, ab, ca), (ab, b, c), (ab, c, ca))
    pick(s1 & s2 & s3, (a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
    return np.vstack([o for o in out if len(o)])


def _part(mesh, c, R, half, v):
    blend = v["blend"]
    wfn = lambda x: _region_weight((x - c) @ R, half, blend)
    m = refine_where(mesh, wfn, max(min(half.min(), blend) / 3, 1e-4 * half.max()))
    q = (m.v - c) @ R
    w = wfn(m.v)
    S = np.array([v["scale_x"], v["scale_y"], v["scale_z"]])
    Rt = rotation_matrix([v["turn_roll"], v["turn_pitch"], v["turn_yaw"]])
    qn = (q * S) @ Rt.T + np.array([v["move_x"], v["move_y"], v["move_z"]])
    q2 = q + w[:, None] * (qn - q)
    return Mesh(q2 @ R.T + c, m.f), []


def _wingpanel(mesh, c, R, half, v):
    blend = v["blend"]
    y0 = v["root_offset"]
    pair = v["sides"] == "pair"
    span = max(half[1] * (1.0 if pair else 2.0) - y0, 1e-9)
    root_blend = max(0.08 * span, blend)

    def wfn(x):
        q = (x - c) @ R
        u = (np.abs(q[:, 1]) if pair else q[:, 1] + half[1]) - y0
        return _region_weight(q, half, blend) * _smoothstep(u / root_blend) * (u > 0)

    m = refine_where(mesh, wfn, max(min(half[0], root_blend, blend * 2) / 3, 1e-4 * half.max()))
    q = (m.v - c) @ R
    u = (np.abs(q[:, 1]) if pair else q[:, 1] + half[1]) - y0
    sgn = np.where(q[:, 1] < 0, -1.0, 1.0) if pair else np.ones(len(q))
    w = wfn(m.v)
    sel = (w > 0) & np.all(np.abs(q) <= half + blend, axis=1)
    if not sel.any():
        return m, ["no wing inside the box beyond the fuselage half-width"]
    # station-wise chord line and mid-thickness of the wing inside the box
    nb = 24
    inb = sel & np.all(np.abs(q) <= half, axis=1)
    ub = np.clip(u[inb] / span, 0, 1 - 1e-9)
    k = (ub * nb).astype(int)
    xmin, xmax, zmin, zmax = (np.full(nb, np.nan) for _ in range(4))
    for j in range(nb):
        s = k == j
        if s.any():
            xs, zs = q[inb][s, 0], q[inb][s, 2]
            xmin[j], xmax[j], zmin[j], zmax[j] = xs.min(), xs.max(), zs.min(), zs.max()
    ok = np.isfinite(xmin)
    if ok.sum() < 2:
        return m, ["too little of the wing inside the box to reshape it"]
    st = (np.arange(nb) + 0.5) / nb
    fill = lambda a: np.interp(st, st[ok], a[ok])
    xmin, xmax, zmin, zmax = fill(xmin), fill(xmax), fill(zmin), fill(zmax)
    x_qc = xmin + 0.25 * (xmax - xmin)
    z_mid = 0.5 * (zmin + zmax)
    un = np.clip(u / span, 0, 1)
    xq = np.interp(un, st, x_qc)
    zm = np.interp(un, st, z_mid)
    xq0, zm0 = x_qc[0], z_mid[0]
    ss, cs, ts = v["span_scale"], v["chord_scale"], v["thickness_scale"]
    u2 = u * ss
    x2 = xq + (xq - xq0) * (ss - 1) + (q[:, 0] - xq) * cs + u2 * math.tan(math.radians(v["sweep_add"]))
    z2 = zm + (zm - zm0) * (ss - 1) + (q[:, 2] - zm) * cs * ts + u2 * math.tan(math.radians(v["dihedral_add"]))
    y2 = sgn * (y0 + u2) if pair else u2 + y0 - half[1]
    qn = np.stack([x2, y2, z2], 1)
    q2 = q.copy()
    q2[sel] = q[sel] + w[sel, None] * (qn[sel] - q[sel])
    return Mesh(q2 @ R.T + c, m.f), []


# ---- the slant: plane cut with a flat cap ---------------------------------------------------------------
def _plane_dir(elev, azim):
    e, a = math.radians(elev), math.radians(azim)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def clip_plane(mesh: Mesh, p0, n, faces_mask=None):
    """Remove everything on the +n side of the plane through p0 from the selected faces' shells,
    and close each cut with flat caps. Returns (mesh, problems)."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    lo, hi = mesh.bbox()
    L = float(np.linalg.norm(hi - lo)) or 1.0
    d = (mesh.v - p0) @ n
    if np.any(np.abs(d) < 1e-9 * L):        # nudge the plane off any vertex it passes through
        d = d - 3.7e-8 * L
    f = mesh.f
    act = np.ones(len(f), bool) if faces_mask is None else faces_mask
    df = d[f]
    gone = act & np.all(df > 0, axis=1)
    cross = act & ~gone & np.any(df > 0, axis=1)
    keep_f = f[~gone & ~cross]
    if not cross.any() and not gone.any():
        return mesh, ["the cut does not reach the body"]
    # crossing faces: new vertices on the cut edges, shared by both faces of each edge
    cf = f[cross]
    cd = df[cross]
    e = np.concatenate([cf[:, [0, 1]], cf[:, [1, 2]], cf[:, [2, 0]]])
    de = np.concatenate([cd[:, [0, 1]], cd[:, [1, 2]], cd[:, [2, 0]]])
    cut = (de[:, 0] > 0) != (de[:, 1] > 0)
    es = np.sort(e, axis=1)
    uniq, inv = np.unique(es[cut], axis=0, return_inverse=True)
    da, db = d[uniq[:, 0]], d[uniq[:, 1]]
    t = da / (da - db)
    newv = mesh.v[uniq[:, 0]] + t[:, None] * (mesh.v[uniq[:, 1]] - mesh.v[uniq[:, 0]])
    eid = -np.ones(len(e), np.int64)
    eid[cut] = len(mesh.v) + inv
    nf = len(cf)
    mids = eid.reshape(3, nf).T                           # vertex on edge ab, bc, ca
    tris, cap_edges = [], []
    for i in range(nf):
        a, b, c_ = cf[i]
        ka, kb, kc = cd[i] <= 0
        mab, mbc, mca = mids[i]
        if ka and not kb and not kc:
            tris.append((a, mab, mca)); cap_edges.append((mca, mab))
        elif kb and not kc and not ka:
            tris.append((b, mbc, mab)); cap_edges.append((mab, mbc))
        elif kc and not ka and not kb:
            tris.append((c_, mca, mbc)); cap_edges.append((mbc, mca))
        elif ka and kb:
            tris += [(a, b, mbc), (a, mbc, mca)]; cap_edges.append((mca, mbc))
        elif kb and kc:
            tris += [(b, c_, mca), (b, mca, mab)]; cap_edges.append((mab, mca))
        else:                                              # kc and ka
            tris += [(c_, a, mab), (c_, mab, mbc)]; cap_edges.append((mbc, mab))
    V = np.vstack([mesh.v, newv])
    problems = []
    caps = _cap_loops(V, np.array(cap_edges, np.int64), n, problems)
    F = np.vstack([keep_f, np.array(tris, np.int64).reshape(-1, 3)] + ([caps] if len(caps) else []))
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    out = Mesh(V[used], remap[F])
    return out, problems


def _cap_loops(V, edges, n, problems):
    nxt = {}
    for a, b in edges:
        if a in nxt:
            problems.append("the cut crosses a non-manifold edge; its cap may be incomplete")
        nxt[a] = b
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen and v in nxt:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if v == start and len(loop) >= 3:
            loops.append(np.array(loop, np.int64))
        elif loop:
            problems.append("an open cut edge could not be closed")
    if not loops:
        return np.zeros((0, 3), np.int64)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    P = [np.stack([V[lp] @ u, V[lp] @ w], 1) for lp in loops]
    area = [0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) for p in P]
    outer = [i for i, a in enumerate(area) if a > 0]
    holes = [i for i, a in enumerate(area) if a <= 0]
    tris = []
    owner = {}
    for h in holes:
        pt = P[h][0]
        best = None
        for o in outer:
            if _point_in_poly(pt, P[o]) and (best is None or abs(area[o]) < abs(area[best])):
                best = o
        if best is not None:
            owner.setdefault(best, []).append(h)
    for o in outer:
        idx, pts = loops[o], P[o]
        for h in sorted(owner.get(o, []), key=lambda h: -P[h][:, 0].max()):
            idx, pts = _bridge(idx, pts, loops[h], P[h])
        tris.append(_ear_clip(np.ascontiguousarray(pts), np.ascontiguousarray(idx)))
    return np.vstack(tris) if tris else np.zeros((0, 3), np.int64)


def _point_in_poly(pt, poly):
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    c = ((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / (ys2 - ys + 1e-300) + xs)
    return bool(c.sum() % 2)


def _bridge(idx, pts, hidx, hpts):
    """Join a hole to its outer loop with a two-way slit from the hole's rightmost vertex to
    the nearest outer vertex the slit can reach without crossing an edge."""
    k = int(np.argmax(hpts[:, 0]))
    m = hpts[k]
    order = np.argsort(np.linalg.norm(pts - m, axis=1))
    segs = np.vstack([np.stack([pts, np.roll(pts, -1, 0)], 1), np.stack([hpts, np.roll(hpts, -1, 0)], 1)])
    j = int(order[0])
    for cand in order[: min(len(order), 64)]:
        if not _crosses(m, pts[cand], segs):
            j = int(cand)
            break
    hi_ = np.concatenate([hidx[k:], hidx[: k + 1]])
    hp = np.concatenate([hpts[k:], hpts[: k + 1]])
    idx2 = np.concatenate([idx[: j + 1], hi_, idx[j:]])
    pts2 = np.vstack([pts[: j + 1], hp, pts[j:]])
    return idx2, pts2


def _crosses(a, b, segs):
    p, r = a, b - a
    q, s = segs[:, 0], segs[:, 1] - segs[:, 0]
    rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / rxs
        uu = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / rxs
    hit = (np.abs(rxs) > 1e-30) & (t > 1e-9) & (t < 1 - 1e-9) & (uu > 1e-9) & (uu < 1 - 1e-9)
    return bool(hit.any())


@njit(cache=True)
def _ear_clip(p, idx):
    """Triangulate a simple polygon (counter-clockwise, may contain the doubled vertices of hole
    bridges) by ear clipping. Returns triangles of `idx` entries."""
    n = p.shape[0]
    out = np.empty((max(n - 2, 0), 3), np.int64)
    if n < 3:
        return out[:0]
    prv = np.empty(n, np.int64)
    nxt = np.empty(n, np.int64)
    for i in range(n):
        prv[i] = (i - 1) % n
        nxt[i] = (i + 1) % n
    remaining = n
    k = 0
    i = 0
    stall = 0
    while remaining > 3:
        a, b, c = prv[i], i, nxt[i]
        ax, ay = p[a, 0], p[a, 1]
        bx, by = p[b, 0], p[b, 1]
        cx, cy = p[c, 0], p[c, 1]
        cr = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        ear = cr > 0.0
        if ear:
            j = nxt[c]
            while j != a:
                px, py = p[j, 0], p[j, 1]
                same = (px == ax and py == ay) or (px == bx and py == by) or (px == cx and py == cy)
                if not same:
                    d1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                    d2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
                    d3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
                    if d1 >= 0.0 and d2 >= 0.0 and d3 >= 0.0:
                        ear = False
                        break
                j = nxt[j]
        if ear or stall > 2 * remaining:
            out[k, 0], out[k, 1], out[k, 2] = idx[a], idx[b], idx[c]
            k += 1
            nxt[a] = c
            prv[c] = a
            remaining -= 1
            i = c
            stall = 0
        else:
            i = nxt[i]
            stall += 1
    a, b, c = prv[i], i, nxt[i]
    out[k, 0], out[k, 1], out[k, 2] = idx[a], idx[b], idx[c]
    k += 1
    return out[:k]


def _slant(mesh, c, v):
    n = _plane_dir(v["elevation"], v["azimuth"])
    p0 = c - v["depth"] * n
    mask = None
    if v["parts"] == "nearest":
        comp = mesh.components()
        cen = mesh.triangles().mean(axis=1)
        mask = comp == comp[int(np.argmin(np.linalg.norm(cen - c, axis=1)))]
    return clip_plane(mesh, p0, n, mask)


# ---- union bookkeeping ---------------------------------------------------------------------------------
def mark_hidden(mesh: Mesh):
    """Faces whose centre lies inside another closed shell: buried in the union, so neither
    wetted surface nor surface pressure. None when nothing is buried."""
    comp, closed = shell_info(mesh)
    ncomp = int(comp.max()) + 1 if len(comp) else 0
    if ncomp < 2:
        return None
    tri = mesh.triangles()
    cen = tri.mean(axis=1)
    hidden = np.zeros(len(tri), bool)
    lo = np.array([tri[comp == k].reshape(-1, 3).min(0) for k in range(ncomp)])
    hi = np.array([tri[comp == k].reshape(-1, 3).max(0) for k in range(ncomp)])
    for k in range(ncomp):
        if not closed[k]:
            continue
        cand = np.where((comp != k) & ~hidden & np.all((cen >= lo[k]) & (cen <= hi[k]), axis=1))[0]
        if not len(cand):
            continue
        tk = np.ascontiguousarray(tri[comp == k])
        ins = _inside_counts(tk, np.zeros(len(tk), np.int64), 1, np.ascontiguousarray(cen[cand]))
        hidden[cand[ins]] = True
    return hidden if hidden.any() else None


# ---- applying the timeline ---------------------------------------------------------------------------------
def apply(mesh: Mesh, features, frame: Frame, overrides=None, hidden=True):
    """Body-frame mesh -> (mesh with every enabled feature applied, problems, info).
    overrides: {(feature id, param key): value} from the optimizer. hidden=False skips marking
    buried faces (seconds on a refined mesh): enough for a shape's size, never for a CFD run."""
    problems, added = [], []
    m = mesh
    for f in features or []:
        if not f.get("enabled", True):
            continue
        if (overrides or {}).get((f["id"], "_use")) == "off":
            continue
        d = get(f["type"])
        v = dict(f["values"])
        for (fid, key), val in (overrides or {}).items():
            if fid == f["id"]:
                v[key] = val
        c, R, half = placement(f, frame)
        name = f.get("name") or d["title"]
        try:
            if f["type"] in ("nosecone", "tailcone") and v.get("section") == "match":
                shell, pr = _extension(m, c, R, v, forward=f["type"] == "nosecone")
                if shell is not None:
                    shell = _place(shell, c, R)
            elif f["type"] == "nosecone":
                shell, pr = _nosecone(v)
                shell = _place(shell, c, R)
            elif f["type"] == "tailcone":
                shell, pr = _tailcone(v)
                shell = _place(shell, c, R)
            elif f["type"] == "pod":
                shell, pr = _pod(v)
                shell = _place(shell, c, R)
            elif f["type"] == "wing":
                shell, pr = _wing(v)
                shell = _place(shell, c, R)
            elif f["type"] == "fins":
                shell, pr = _fins(v)
                shell = _place(shell, c, R)
            elif f["type"] == "fairing":
                shell, pr = _fairing(m, c, R, half, v)
            elif f["type"] == "slant":
                m, pr = _slant(m, c, v)
                shell = None
            elif f["type"] == "wingpanel":
                m, pr = _wingpanel(m, c, R, half, v)
                shell = None
            elif f["type"] == "part":
                m, pr = _part(m, c, R, half, v)
                shell = None
            else:
                raise ValueError(f["type"])
        except (ValueError, FloatingPointError) as e:
            shell, pr = None, [str(e)]
        problems += [f"{name}: {p}" for p in pr]
        if shell is not None:
            m = Mesh(np.vstack([m.v, shell.v]), np.vstack([m.f, shell.f + len(m.v)]))
            added.append(f["id"])
    m.hidden = mark_hidden(m) if hidden else None
    lo, hi = m.bbox()
    info = {"overall_length": float(hi[0] - lo[0]), "overall_width": float(hi[1] - lo[1]),
            "overall_height": float(hi[2] - lo[2]), "faces": m.n_faces,
            "hidden_faces": int(m.hidden.sum()) if m.hidden is not None else 0}
    return m, problems, info


def union_volume(mesh: Mesh, cells=160):
    """Volume of the union of all shells (voxel count on a grid `cells` long)."""
    from ..cfd.voxelize import voxelize_shells
    lo, hi = mesh.bbox()
    size = hi - lo
    h = float(size.max()) / cells
    shape = tuple(int(math.ceil(s / h)) + 4 for s in size)
    tri = ((mesh.triangles().reshape(-1, 3) - lo) / h + 2.0).reshape(-1, 3, 3)
    comp, closed = shell_info(mesh)
    solid, _ = voxelize_shells(tri, comp, closed, shape)
    return float(solid.sum()) * h ** 3


# ---- sensible starting values where the user clicked ------------------------------------------------------
def section_loops(mesh: Mesh, x0, points=False):
    """Closed loops where the plane x = x0 cuts the surface: [(centre_yz, area, perimeter, lo_yz, hi_yz)]
    (with points=True, each loop's (y, z) points are appended)."""
    d = mesh.v[:, 0] - x0
    if np.any(np.abs(d) < 1e-12):
        d = d - 1e-9 * float(np.ptp(mesh.v[:, 0]) or 1.0)
    f = mesh.f
    df = d[f]
    cross = np.any(df > 0, axis=1) & np.any(df <= 0, axis=1)
    if not cross.any():
        return []
    cf = f[cross]
    e = np.concatenate([cf[:, [0, 1]], cf[:, [1, 2]], cf[:, [2, 0]]])
    de = np.concatenate([df[cross][:, [0, 1]], df[cross][:, [1, 2]], df[cross][:, [2, 0]]])
    cut = (de[:, 0] > 0) != (de[:, 1] > 0)
    es = np.sort(e, axis=1)
    uniq, inv = np.unique(es[cut], axis=0, return_inverse=True)
    t = d[uniq[:, 0]] / (d[uniq[:, 0]] - d[uniq[:, 1]])
    pts = mesh.v[uniq[:, 0]] + t[:, None] * (mesh.v[uniq[:, 1]] - mesh.v[uniq[:, 0]])
    nf = len(cf)
    eid = -np.ones(3 * nf, np.int64)
    eid[np.where(cut)[0]] = inv
    eid = eid.reshape(3, nf).T
    adj = {}
    for row in eid:
        a, b = [x for x in row if x >= 0][:2]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, out = set(), []
    for s in adj:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nb = [n for n in adj[cur] if n != prev]
            if not nb or nb[0] == s or nb[0] in seen:
                break
            prev, cur = cur, nb[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) < 3:
            continue
        p = pts[loop][:, 1:]
        area = abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
        per = float(np.linalg.norm(p - np.roll(p, -1, 0), axis=1).sum())
        out.append((p.mean(0), float(area), per, p.min(0), p.max(0)) + ((p,) if points else ()))
    return out


def _section_at(mesh: Mesh, x, band):
    """Centre and size (x, y, z) of the body's own cross-section near x: the roundest loop of a
    slice there, so a wing, fin or tail surface cut by the same plane does not count."""
    loops = section_loops(mesh, x)
    if loops:
        compact = [(4 * math.pi * a / (p * p) if p > 0 else 0.0, a, c, lo, hi) for c, a, p, lo, hi in loops]
        round_ = [c for c in compact if c[0] >= 0.45]
        best = max(round_ or compact, key=lambda c: c[1])
        _, _, c, lo, hi = best
        return np.array([x, c[0], c[1]]), np.array([0.0, hi[0] - lo[0], hi[1] - lo[1]])
    sel = np.abs(mesh.v[:, 0] - x) <= band
    if sel.sum() < 3:
        k = np.argsort(np.abs(mesh.v[:, 0] - x))[:12]
        sel = np.zeros(len(mesh.v), bool)
        sel[k] = True
    p = mesh.v[sel]
    lo, hi = p.min(0), p.max(0)
    return (lo + hi) / 2, hi - lo


def suggest(ftype, body: Mesh, frame: Frame, point_body=None, normal_body=None):
    """A new feature with dimensions fitted to the model near `point_body` (body frame)."""
    d = get(ftype)
    lo, hi = body.bbox()
    L = float(hi[0] - lo[0])
    size = hi - lo
    D = float(max(size[1], size[2]))
    v = defaults(ftype)
    p = np.asarray(point_body, float) if point_body is not None else (lo + hi) / 2
    rot = None
    region = None
    if ftype == "nosecone":
        # the shoulder: the first station from the front where the body is near its full diameter.
        # A blunt front qualifies at once; a pointed nose gets a new cone long enough to cover it.
        xs = lo[0] + L * np.linspace(0.02, 0.4, 20)
        secs = [_section_at(body, x, 0.01 * L) for x in xs]
        dias = np.array([max(s[1][1], s[1][2]) for s in secs])
        k = int(np.argmax(dias >= 0.85 * dias.max())) if dias.max() > 0 else 0
        x0 = float(xs[k])
        cen, sec = secs[k]
        dia = float(dias[k]) or 0.3 * D
        p = np.array([x0, cen[1], cen[2]])
        v.update(length=max(1.6 * dia, 1.1 * (x0 - lo[0])), diameter=dia, overlap=min(0.25 * dia, 0.05 * L))
    elif ftype == "tailcone":
        x0 = hi[0] - 0.02 * L
        cen, sec = _section_at(body, x0, 0.02 * L)
        dia = float(max(sec[1], sec[2])) or 0.3 * D
        p = np.array([x0, cen[1], cen[2]])
        v.update(length=1.5 * dia, diameter=dia, overlap=min(0.25 * dia, 0.05 * L))
    elif ftype == "pod":
        v.update(length=0.3 * L, diameter=0.1 * L, nose_length=0.08 * L, tail_length=0.12 * L)
    elif ftype == "wing":
        cen, sec = _section_at(body, p[0], 0.03 * L)
        p = np.array([p[0], cen[1], p[2]])
        v.update(span=max(1.2 * L, 3 * D), root=0.2 * L, tip=0.1 * L)
    elif ftype == "fins":
        cen, sec = _section_at(body, p[0], 0.02 * L)
        r = float(max(sec[1], sec[2])) / 2
        p = np.array([p[0], cen[1], cen[2]])
        v.update(root=0.15 * L, tip=0.06 * L, span=max(1.2 * r, 0.06 * L), thickness=max(0.004 * L, 5e-4), root_radius=r)
    elif ftype == "slant":
        nrm = np.asarray(normal_body, float) if normal_body is not None else np.array([1.0, 0, 0])
        nrm = nrm / (np.linalg.norm(nrm) or 1.0)
        v.update(elevation=round(math.degrees(math.asin(np.clip(nrm[2], -1, 1))), 1),
                 azimuth=round(math.degrees(math.atan2(nrm[1], nrm[0])), 1), depth=0.03 * L)
    elif ftype == "fairing":
        region = np.full(3, 0.12 * L)
        v.update(front_length=0.1 * L, rear_length=0.2 * L, clearance=0.003 * L)
    elif ftype == "part":
        region = np.full(3, 0.12 * L)
        v.update(blend=0.05 * L)
    elif ftype == "wingpanel":
        # the wing near the click: geometry well out to the side, around the clicked station
        half_w = size[1] / 2
        cy = (lo[1] + hi[1]) / 2
        near = np.abs(body.v[:, 0] - p[0]) < 0.3 * L
        outer = near & (np.abs(body.v[:, 1] - cy) > 0.35 * half_w)
        if outer.sum() >= 8:
            w = body.v[outer]
            wl, wh = w.min(0), w.max(0)
            # fuselage half-width: the body just ahead of the wing's leading edge
            cen, sec = _section_at(body, wl[0] - 0.03 * L, 0.02 * L)
            fus = float(sec[1]) / 2 if sec[1] < 0.6 * size[1] else 0.1 * half_w
            y0 = min(fus * 1.05, 0.5 * half_w)
            # the whole exposed wing, root to tip: a tapered wing's root chord is the longest, and a
            # box edge across a thin trailing edge would pull its two surfaces apart
            band = (wl[2] - 0.5 * (wh[2] - wl[2]), wh[2] + 0.5 * (wh[2] - wl[2]))
            span_pts = (np.abs(body.v[:, 1] - cy) > y0 * 1.05) & (body.v[:, 2] >= band[0]) & (body.v[:, 2] <= band[1]) & \
                (np.abs(body.v[:, 0] - (wl[0] + wh[0]) / 2) < 0.4 * L)
            if span_pts.sum() >= 8:
                w = body.v[span_pts]
                wl, wh = np.minimum(wl, w.min(0)), np.maximum(wh, w.max(0))
            chord = wh[0] - wl[0]
            blend = 0.03 * L
            p = np.array([(wl[0] + wh[0]) / 2, cy, (wl[2] + wh[2]) / 2])
            region = np.array([chord + 2.5 * blend, size[1] * 1.04, max((wh[2] - wl[2]) * 1.4, 0.01 * L) + blend])
            v.update(root_offset=y0, blend=blend)
        else:
            p = np.array([p[0], cy, p[2]])
            region = np.array([0.35 * L, size[1] * 1.04, max(0.12 * size[2], 0.03 * L)])
            v.update(root_offset=0.1 * half_w, blend=0.03 * L)
    for pr in d["params"]:
        if pr.kind != "choice":
            v[pr.key] = float(min(max(v[pr.key], pr.lo), pr.hi))
    # back to file coordinates (the frame's inverse: R0 is orthogonal)
    center_file = (frame.R0.T @ (p + frame.center)) / frame.scale
    f = {"type": ftype, "name": d["title"], "center": [round(float(x), 6) for x in center_file], "values": v, "params": {}}
    if region is not None:
        f["size"] = [round(float(x), 6) for x in (np.abs(frame.R0).T @ region) / frame.scale]
    if rot is not None:
        f["rotation"] = rot
    return clean(f)


# ---- generative proposals: features the optimizer may add -----------------------------------------------
IDEAS = {
    # name: (what it is, the hints a design assistant may give, as fractions of the body)
    "boat_tail": ("the body's own rear section continued and tapered", {"length": "x rear height, 0.3-1.2", "end": "end / start size, 0.45-0.9"}),
    "rounded_front": ("a rounded front continuing the front section", {"length": "x front height, 0.08-0.6"}),
    "rear_slant": ("the upper rear edge cut back (fastback / hatch angle)", {"angle": "degrees above horizontal, 20-70", "depth": "x body height, 0-0.25"}),
    "rear_diffuser": ("the lower rear edge cut up (underbody diffuser)", {"angle": "degrees below horizontal, 20-70", "depth": "x body height, 0-0.2"}),
}


def _ends(body):
    lo, hi = body.bbox()
    L = float(hi[0] - lo[0])
    secs = {}
    for frac in (0.03, 0.5, 0.97):
        loops = section_loops(body, lo[0] + frac * L)
        secs[frac] = max(loops, key=lambda q: q[1]) if loops else None
    full = max((q[1] for q in secs.values() if q), default=0.0)
    return lo, hi, L, secs, full


def idea(kind, body: Mesh, frame: Frame, hints=None, name=None):
    """One generative idea as an optional feature (None if the body has no place for it). hints:
    fractions of the body as in IDEAS, clamped to the ranges there."""
    h_ = hints or {}
    clamp = lambda k, d, lo_, hi_: float(min(max(float(h_.get(k, d)), lo_), hi_))
    lo, hi, L, secs, full = _ends(body)

    def make(ftype, label, center_body, values, free, normal=None):
        f = suggest(ftype, body, frame, center_body, normal)
        f["center"] = [round(float(x), 6) for x in (frame.R0.T @ (np.asarray(center_body, float) + frame.center)) / frame.scale]
        f["name"] = name or label
        f["values"].update(values)
        f["optional"] = True
        f["params"] = {p.key: {"locked": p.key not in free} for p in get(ftype)["params"]}
        for k, (blo, bhi) in free.items():
            f["params"][k]["bounds"] = [blo, bhi]
        return clean(f)

    if kind == "boat_tail":
        rear = secs[0.97]
        if not rear:
            return None
        h = float(max(rear[4] - rear[3]))
        return make("tailcone", "Boat-tail (proposed)", [lo[0] + 0.97 * L, rear[0][0], rear[0][1]],
                    {"section": "match", "shape": "smooth", "length": clamp("length", 0.8, 0.3, 1.2) * h,
                     "end_ratio": clamp("end", 0.7, 0.45, 0.9), "overlap": 0.02 * L},
                    {"length": (0.3 * h, 1.2 * h), "end_ratio": (0.45, 0.9)})
    if kind == "rounded_front":
        front = secs[0.03]
        if not front:
            return None
        h = float(max(front[4] - front[3]))
        return make("nosecone", "Rounded front (proposed)", [lo[0] + 0.03 * L, front[0][0], front[0][1]],
                    {"section": "match", "length": clamp("length", 0.25, 0.08, 0.6) * h, "overlap": 0.02 * L},
                    {"length": (0.08 * h, 0.6 * h)})
    if kind in ("rear_slant", "rear_diffuser"):
        hz = float(hi[2] - lo[2])
        aft = body.v[body.v[:, 0] > lo[0] + 0.5 * L]
        if len(aft) < 4:
            return None
        up = kind == "rear_slant"
        elev = clamp("angle", 45.0, 20.0, 70.0) * (1 if up else -1)
        dmax = 0.25 if up else 0.2
        n = _plane_dir(elev, 0.0)
        edge = aft[int(np.argmax(aft @ n))]
        return make("slant", "Rear slant (proposed)" if up else "Rear diffuser (proposed)", edge,
                    {"elevation": elev, "azimuth": 0.0, "depth": clamp("depth", 0.04, 0.0, dmax) * hz, "parts": "all"},
                    {"depth": (0.0, dmax * hz), "elevation": (20.0, 70.0) if up else (-70.0, -20.0)}, n)
    raise ValueError(f"unknown idea {kind!r}")


def body_summary(body: Mesh):
    """A few numbers that describe the body's shape, for a design assistant."""
    lo, hi, L, secs, full = _ends(body)
    size = hi - lo
    frac = lambda q: round(q[1] / full, 3) if q and full else 0.0
    return {"length_m": round(L, 4), "width_m": round(float(size[1]), 4), "height_m": round(float(size[2]), 4),
            "fineness": round(L / max(float(size[1]), float(size[2]), 1e-9), 2),
            "front_section_vs_max": frac(secs[0.03]), "rear_section_vs_max": frac(secs[0.97])}


def propose(body: Mesh, frame: Frame):
    """Shape features worth trying on this body, each optional (the optimizer screens the design with
    and without it and keeps only those that beat the measured error), with its key dimensions free:
    a boat-tail when the rear is blunt, a rounded front when the front is, and rear slant and
    diffuser angles."""
    lo, hi, L, secs, full = _ends(body)
    kinds = []
    if secs[0.97] and secs[0.97][1] > 0.05 * full:
        kinds.append("boat_tail")
    if secs[0.03] and secs[0.03][1] > 0.35 * full:
        kinds.append("rounded_front")
    kinds += ["rear_slant", "rear_diffuser"]
    return [f for f in (idea(k, body, frame) for k in kinds) if f is not None]
