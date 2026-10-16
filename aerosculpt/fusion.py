"""Send a design to Autodesk Fusion through the local FusionMCPBridge add-in.

The add-in (the fusion360-mcp-bridge repository) listens on 127.0.0.1 and runs a Python script
inside Fusion when called with the shared secret in ~/.fusion-mcp-secret. This module only ever
sends scripts it builds itself from numbers and identifier-checked parameter names: nothing a
user typed reaches Fusion as code. The web app enables it only when it runs on the same machine
as Fusion (local preview, or FUSION_BRIDGE=1) and only for the owner account.

Two actions:
* parameters - write the design's dimensions into Fusion user parameters (created if missing),
  so a parametric Fusion model built on the same names rebuilds to the optimized shape;
* import     - import the design's STEP file into the open design as a new component.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("FUSION_MCP_PORT", "7654"))
SECRET_FILE = Path.home() / ".fusion-mcp-secret"
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
UNITS = {"m": "m", "mm": "mm", "deg": "deg", "ratio": "", "-": "", "": ""}

_PARAM_SCRIPT = """import adsk.core, adsk.fusion, json
PARAMS = json.loads({payload!r})

def run(_context):
    if design is None:
        print(json.dumps({{"error": "Open a design in Fusion first."}}))
        return
    ups = design.userParameters
    out = []
    for name, value, unit, comment in PARAMS:
        expr = (repr(float(value)) + " " + unit).strip()
        p = ups.itemByName(name)
        if p is None:
            ups.add(name, adsk.core.ValueInput.createByString(expr), unit, comment)
            out.append([name, None, expr])
        else:
            old = p.expression
            p.expression = expr
            out.append([name, old, expr])
    print(json.dumps({{"updated": out, "document": app.activeDocument.name if app.activeDocument else None}}))
"""

_IMPORT_SCRIPT = """import adsk.core, adsk.fusion, json
PATH = {path!r}

def run(_context):
    if design is None:
        print(json.dumps({{"error": "Open a design in Fusion first."}}))
        return
    im = app.importManager
    opts = im.createSTEPImportOptions(PATH)
    opts.isViewFit = True
    ok = im.importToTarget(opts, design.rootComponent)
    print(json.dumps({{"imported": bool(ok), "document": app.activeDocument.name if app.activeDocument else None}}))
"""


class FusionError(Exception):
    pass


def _secret():
    try:
        return SECRET_FILE.read_text().strip()
    except OSError:
        return ""


def _call(path, body=None, timeout=40.0):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {_secret()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise FusionError(f"Fusion bridge answered {e.code}") from None
    except (urllib.error.URLError, OSError):
        raise FusionError("Fusion is not reachable: open Fusion with a design and start the FusionMCPBridge "
                          "add-in (Utilities > Add-Ins).") from None


def status():
    try:
        h = _call("/health", timeout=3.0)
        return {"ok": h.get("status") == "ok", "document": h.get("documentName"), "has_design": h.get("hasDesign")}
    except FusionError as e:
        return {"ok": False, "why": str(e)}


def _run(script):
    out = _call("/execute", {"script": script})
    text = (out.get("result") or "").strip()
    if text.startswith("Error:"):
        raise FusionError(text.splitlines()[0])
    try:
        res = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError):
        raise FusionError("Fusion returned no result") from None
    if res.get("error"):
        raise FusionError(res["error"])
    return res


def push_parameters(params):
    """params: [(name, value, unit, comment)] -> {"updated": [[name, old, new], ...]}"""
    clean = []
    for name, value, unit, comment in params:
        if not NAME_RE.match(str(name)):
            raise FusionError(f"invalid parameter name {name!r}")
        v = float(value)
        if v != v or abs(v) > 1e9:
            raise FusionError(f"invalid value for {name}")
        clean.append([str(name), v, UNITS.get(unit, ""), str(comment)[:120]])
    return _run(_PARAM_SCRIPT.format(payload=json.dumps(clean)))


def import_step(path):
    p = Path(path).resolve()
    if not p.exists() or p.suffix.lower() not in (".step", ".stp"):
        raise FusionError("no STEP file to import")
    return _run(_IMPORT_SCRIPT.format(path=str(p)))


def parameter_script(params):
    """The same parameter update as a standalone Fusion script (Utilities > Scripts and Add-Ins),
    for a Fusion on another computer."""
    clean = [[str(n), float(v), UNITS.get(u, ""), str(c)[:120]] for n, v, u, c in params if NAME_RE.match(str(n))]
    body = _PARAM_SCRIPT.format(payload=json.dumps(clean))
    header = ("# AeroSculpt: optimized dimensions -> Fusion user parameters.\n"
              "# Run from Fusion: Utilities > Scripts and Add-Ins > + (script) and pick this file.\n"
              "app = adsk.core.Application.get()\n"
              "design = adsk.fusion.Design.cast(app.activeProduct)\n")
    return body.replace("import adsk.core, adsk.fusion, json\n", "import adsk.core, adsk.fusion, json\n" + header, 1)
