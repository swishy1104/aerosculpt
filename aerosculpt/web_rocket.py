"""Rocket endpoints: the catalog (motors, materials, finishes), the motor library (built-in,
uploaded .eng/.rse files, ThrustCurve.org when asked), the live flight evaluation of a rocket
project, the drag-table export and altimeter-log comparison. Mounted by web.py."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .config import settings
from .db import db
from .geometry import templates as T
from .jobs import project_dir
from .rocket import model as RM
from .rocket.flutter import FIN_MATERIALS
from .rocket.geometry import FINISHES, LAUNCH_LUGS
from .rocket.motors import BY_ID, Motor, parse_motor_file, thrustcurve_download, thrustcurve_search

MAX_MOTOR_FILE = 400_000


def _w():
    from . import web
    return web


# ---- motors ---------------------------------------------------------------------------------
def motors_dir(uid):
    d = settings.data_dir / "motors" / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _motor_from_json(d):
    return Motor(d["id"], d["name"], d.get("manufacturer", ""), d.get("diameter_mm", 18), d.get("length_mm", 70), d.get("delays") or [],
                 d.get("prop_mass_g", 0) / 1000.0, d.get("total_mass_g", 0) / 1000.0, d["curve"], d.get("source", "file"), d.get("note", ""))


def user_motors(uid):
    out = []
    for f in sorted(motors_dir(uid).glob("*.json")):
        try:
            out.append(_motor_from_json(json.loads(f.read_text(encoding="utf-8"))))
        except (ValueError, KeyError):
            continue
    return out


def load_motor(uid, mid):
    if not mid:
        return None
    if mid in BY_ID:
        return BY_ID[mid]
    f = motors_dir(uid) / f"{mid}.json"
    if f.exists():
        try:
            return _motor_from_json(json.loads(f.read_text(encoding="utf-8")))
        except (ValueError, KeyError):
            return None
    return None


def motor_from_settings(uid, rk):
    return load_motor(uid, ((rk or {}).get("motor") or {}).get("id"))


def _store_motor(uid, m: Motor):
    key = hashlib.sha1(f"{m.manufacturer}|{m.name}|{m.total_impulse:.3f}|{m.burn_time:.3f}".encode()).hexdigest()[:10]
    m.id = f"u-{key}"
    (motors_dir(uid) / f"{m.id}.json").write_text(json.dumps(m.public()), encoding="utf-8")
    return m


# ---- settings -------------------------------------------------------------------------------
def clean_rocket(data, old):
    """Validated rocket settings on top of the stored ones (and the defaults)."""
    W = _w()
    err, num = W.err, W._num
    out = RM._merge(RM.DEFAULT_ROCKET, old or {})
    d = data if isinstance(data, dict) else {}

    def rng(v, name, lo, hi, none_ok=False):
        if v in (None, "") and none_ok:
            return None
        x = num(v, name)
        if not (lo <= x <= hi):
            err(400, f"{name.capitalize()} must be between {lo:g} and {hi:g}.")
        return x

    if isinstance(d.get("masses"), dict):
        m = d["masses"]
        mode = m.get("mode", out["masses"].get("mode", "estimate"))
        if mode not in ("estimate", "dry", "items"):
            err(400, "Unknown mass mode.")
        out["masses"]["mode"] = mode
        if "dry_mass_g" in m:
            out["masses"]["dry_mass_g"] = rng(m["dry_mass_g"], "dry mass", 0.1, 1e6, True)
        if "dry_cg_mm" in m:
            out["masses"]["dry_cg_mm"] = rng(m["dry_cg_mm"], "centre of gravity", -1e4, 1e5, True)
        if "items" in m:
            items = m["items"]
            if not isinstance(items, list) or len(items) > 60:
                err(400, "Up to 60 parts.")
            out["masses"]["items"] = [{"name": str(i.get("name", "part"))[:40], "g": rng(i.get("g", 0), "part mass", 0.0, 1e6),
                                       "x_mm": rng(i.get("x_mm", 0), "part position", -1e4, 1e5)} for i in items if isinstance(i, dict)]
    if isinstance(d.get("materials"), dict):
        for k, allowed in (("nose", RM.MATERIALS["nose"]), ("body", RM.MATERIALS["body"]), ("fins", FIN_MATERIALS)):
            if k in d["materials"]:
                if d["materials"][k] not in allowed:
                    err(400, f"Unknown {k} material.")
                out["materials"][k] = d["materials"][k]
    if isinstance(d.get("motor"), dict):
        mid = d["motor"].get("id")
        out["motor"]["id"] = str(mid)[:80] if mid else None
        if "delay" in d["motor"]:
            out["motor"]["delay"] = rng(d["motor"]["delay"], "delay", 0.0, 30.0, True)
    if isinstance(d.get("launch"), dict):
        L = d["launch"]
        for k, lo, hi, none_ok in (("rail_m", 0.1, 30.0, False), ("angle_deg", -30.0, 30.0, False), ("wind_m_s", 0.0, 40.0, False),
                                   ("elevation_m", -500.0, 6000.0, False), ("temperature_c", -60.0, 60.0, True)):
            if k in L:
                v = rng(L[k], k.replace("_", " "), lo, hi, none_ok)
                out["launch"][k] = v if v is not None else (None if none_ok else out["launch"][k])
    if isinstance(d.get("parachute"), dict):
        P = d["parachute"]
        if "enabled" in P:
            out["parachute"]["enabled"] = bool(P["enabled"])
        if "diameter_m" in P:
            out["parachute"]["diameter_m"] = rng(P["diameter_m"], "parachute diameter", 0.05, 30.0, True)
        if "cd" in P:
            out["parachute"]["cd"] = rng(P["cd"], "parachute drag coefficient", 0.3, 3.0)
    for k, allowed in (("finish", FINISHES), ("launch_lug", LAUNCH_LUGS), ("fin_material", FIN_MATERIALS),
                       ("fin_edge", {"square", "rounded", "airfoil"})):
        if k in d:
            if d[k] not in allowed:
                err(400, f"Unknown {k.replace('_', ' ')}.")
            out[k] = d[k]
    if "min_margin" in d:
        out["min_margin"] = rng(d["min_margin"], "minimum static margin", 0.0, 5.0)
    if "flutter_margin" in d:
        out["flutter_margin"] = rng(d["flutter_margin"], "flutter margin", 1.0, 3.0)
    if "min_rail_speed" in d:
        out["min_rail_speed"] = rng(d["min_rail_speed"], "minimum rail speed", 3.0, 60.0)
    if "units" in d:
        if d["units"] not in ("metric", "imperial"):
            err(400, "Units are metric or imperial.")
        out["units"] = d["units"]
    return out


# ---- handlers -------------------------------------------------------------------------------
async def rocket_catalog(request):
    W = _w()
    with db() as con:
        user = W.current_user(request, con)
    cat = RM.catalog()
    cat["user_motors"] = [m.public() | {"curve": None} for m in user_motors(user["id"])]
    cat["online_motors"] = True
    return JSONResponse(cat)


async def motors(request):
    W = _w()
    with db() as con:
        user = W.current_user(request, con)
    if request.method == "GET":
        q = (request.query_params.get("q") or "").strip().lower()
        allm = list(BY_ID.values()) + user_motors(user["id"])
        if q:
            allm = [m for m in allm if q in m.name.lower() or q in m.manufacturer.lower() or q == m.cls.lower()]
        return JSONResponse([m.public() | {"curve": None} for m in allm])
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        data = await W.body_json(request)
        text, filename = str(data.get("text") or ""), str(data.get("filename") or "motor.eng")
    else:
        filename = request.query_params.get("filename", "motor.eng")
        raw = await request.body()
        if len(raw) > MAX_MOTOR_FILE:
            W.err(413, "Motor files are small text files (up to 400 kB).")
        text = raw.decode("utf-8", "replace")
    if len(text) > MAX_MOTOR_FILE:
        W.err(413, "Motor files are small text files (up to 400 kB).")
    try:
        found = parse_motor_file(text, filename)
    except Exception as e:  # noqa: BLE001 - a parser message is the user-facing message
        W.err(400, f"Could not read the motor file: {e}")
    stored = [_store_motor(user["id"], m) for m in found[:20]]
    return JSONResponse([m.public() for m in stored], status_code=201)


async def motor_item(request):
    W = _w()
    mid = request.path_params["mid"]
    with db() as con:
        user = W.current_user(request, con)
    if request.method == "DELETE":
        f = motors_dir(user["id"]) / f"{mid}.json"
        if mid.startswith("u-") and f.exists():
            f.unlink()
            return JSONResponse({"ok": True})
        W.err(404, "Motor not found.")
    m = load_motor(user["id"], mid)
    if m is None:
        W.err(404, "Motor not found.")
    return JSONResponse(m.public())


async def motors_online(request):
    """ThrustCurve.org search: a network request made only because the user asked."""
    W = _w()
    with db() as con:
        W.current_user(request, con)
    q = (request.query_params.get("q") or "").strip()
    if not q:
        W.err(400, "Type a motor designation (C6, H128) or a manufacturer.")
    try:
        rows = await run_in_threadpool(thrustcurve_search, q)
    except Exception as e:  # noqa: BLE001 - network problems are reported, not raised
        W.err(502, f"ThrustCurve.org could not be reached ({type(e).__name__}).")
    return JSONResponse(rows)


async def motor_online_add(request):
    W = _w()
    with db() as con:
        user = W.current_user(request, con)
    data = await W.body_json(request)
    tc_id = str(data.get("tc_id") or "")[:40]
    if not tc_id:
        W.err(400, "Missing ThrustCurve motor id.")
    try:
        m = await run_in_threadpool(thrustcurve_download, tc_id)
    except Exception as e:  # noqa: BLE001
        W.err(502, f"ThrustCurve.org download failed ({type(e).__name__}: {e}).")
    return JSONResponse(_store_motor(user["id"], m).public(), status_code=201)


def _rocket_project(con, request, need="read"):
    W = _w()
    user = W.current_user(request, con)
    p = W._get_project(con, user, request.path_params["pid"], need)
    st = W._migrate(json.loads(p["settings"] or "{}"))
    tm = st.get("template") or {}
    if tm.get("id") != "rocket":
        W.err(400, "The flight model needs a rocket template project (or an imported OpenRocket / RockSim design).")
    return user, p, st, tm


def evaluate_project(uid, st, values=None, rocket=None, with_trajectory=True):
    """The live evaluation: template values and rocket settings -> everything the Flight tab shows."""
    tm = st["template"]
    vals = T.clean("rocket", {**tm["values"], **(values or {})})
    rk = clean_rocket(rocket or {}, st.get("rocket") or {})
    motor = motor_from_settings(uid, rk)
    m = RM.RocketModel(vals, rk, motor)
    ev = m.evaluate(with_trajectory)
    ev["values"] = vals
    return ev, rk


async def project_rocket(request):
    W = _w()
    data = await W.body_json(request)
    with db() as con:
        user, p, st, tm = _rocket_project(con, request, "write" if data.get("save") else "read")
    try:
        ev, rk = await run_in_threadpool(evaluate_project, user["id"], st, data.get("values"), data.get("rocket"), not data.get("light"))
    except (ValueError, KeyError, ZeroDivisionError) as e:
        W.err(400, f"The rocket could not be evaluated: {e}")
    if data.get("save"):
        st["rocket"] = rk
        with db() as con:
            con.execute("UPDATE projects SET settings=?, updated_at=? WHERE id=?", (json.dumps(st), time.time(), p["id"]))
    log = project_dir(p["id"]) / "flight_log.json"
    ev["flight_log"] = json.loads(log.read_text()) if log.exists() else None
    ev["import_notes"] = st.get("import_notes") or []
    return JSONResponse(ev)


async def project_rocket_csv(request):
    W = _w()
    with db() as con:
        user, p, st, tm = _rocket_project(con, request)
    from .rocket.ork import drag_table_csv
    ev, _ = await run_in_threadpool(evaluate_project, user["id"], st, None, None, False)
    text = drag_table_csv(ev["drag_curve"], p["name"])
    return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{W._safe_name(p["name"])}-drag.csv"'})


def parse_flight_log(text):
    """CSV of time and altitude (any header; seconds and metres, or 'ft' in the header)."""
    rows = []
    feet = False
    for i, line in enumerate(csv.reader(io.StringIO(text))):
        if not line:
            continue
        if i == 0 and any(not _isnum(c) for c in line[:2]):
            feet = any("ft" in c.lower() or "feet" in c.lower() for c in line)
            continue
        if len(line) < 2 or not (_isnum(line[0]) and _isnum(line[1])):
            continue
        t, z = float(line[0]), float(line[1])
        rows.append([t, z * 0.3048 if feet else z])
        if len(rows) > 20000:
            break
    if len(rows) < 3:
        raise ValueError("expected rows of time (s) and altitude (m or ft)")
    rows.sort()
    t0 = rows[0][0]
    z0 = rows[0][1]
    return [[round(t - t0, 3), round(z - z0, 2)] for t, z in rows[:: max(1, len(rows) // 1500)]]


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


async def project_flightlog(request):
    W = _w()
    with db() as con:
        user, p, st, tm = _rocket_project(con, request, "write")
    f = project_dir(p["id"]) / "flight_log.json"
    if request.method == "DELETE":
        if f.exists():
            f.unlink()
        return JSONResponse({"ok": True})
    raw = await request.body()
    if len(raw) > 2_000_000:
        W.err(413, "Flight logs up to 2 MB.")
    try:
        rows = parse_flight_log(raw.decode("utf-8", "replace"))
    except ValueError as e:
        W.err(400, f"Could not read the flight log: {e}")
    doc = {"name": request.query_params.get("filename", "flight.csv")[:80], "points": rows, "apogee_m": max(z for _, z in rows),
           "apogee_t": max(rows, key=lambda r: r[1])[0]}
    f.write_text(json.dumps(doc))
    return JSONResponse(doc, status_code=201)


ROUTES = [
    Route("/api/rocket/catalog", rocket_catalog),
    Route("/api/motors", motors, methods=["GET", "POST"]),
    Route("/api/motors/online", motors_online),
    Route("/api/motors/online/add", motor_online_add, methods=["POST"]),
    Route("/api/motors/{mid}", motor_item, methods=["GET", "DELETE"]),
    Route("/api/projects/{pid}/rocket", project_rocket, methods=["POST"]),
    Route("/api/projects/{pid}/rocket/drag.csv", project_rocket_csv),
    Route("/api/projects/{pid}/rocket/flightlog", project_flightlog, methods=["POST", "DELETE"]),
]
