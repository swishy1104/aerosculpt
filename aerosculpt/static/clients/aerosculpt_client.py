"""AeroSculpt Python client and command line (standard library only: copy this one file).

Python:

    from aerosculpt_client import AeroSculpt
    a = AeroSculpt("https://your-aerosculpt", api_key="as_...")          # or AEROSCULPT_URL / AEROSCULPT_KEY
    p = a.upload("wing.step", units="mm")                                 # STL, OBJ, STEP or IGES
    a.settings(p["id"], envelope={"speed_min": 15, "speed_max": 30, "speed_count": 3,
                                  "aoa_min": 0, "aoa_max": 6, "aoa_count": 3})
    job = a.analyze(p["id"], fidelity="standard", grid_study=True)
    job = a.wait(job["id"])                                               # prints progress
    a.download(job["id"], "report.pdf", "pdf")                            # pdf, step, vtk, stl

Command line:

    python aerosculpt_client.py projects
    python aerosculpt_client.py upload model.stl --units mm
    python aerosculpt_client.py analyze PROJECT --fidelity draft --grid-study --wait
    python aerosculpt_client.py optimize PROJECT --effort balanced --goal drag --wait
    python aerosculpt_client.py jobs PROJECT
    python aerosculpt_client.py download JOB pdf report.pdf
    python aerosculpt_client.py sweep PROJECT --param wing_span=2.0,2.4,2.8     # a batch of analyses

API keys are created on the Account page (Pro). Every call is a plain HTTPS request with
"Authorization: Bearer <key>", so any language can do the same.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

__version__ = "1.0"


class AeroSculptError(Exception):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class AeroSculpt:
    def __init__(self, url=None, api_key=None, timeout=120):
        self.url = (url or os.environ.get("AEROSCULPT_URL") or "http://localhost:8000").rstrip("/")
        self.key = api_key or os.environ.get("AEROSCULPT_KEY") or ""
        self.timeout = timeout

    # ---- transport ----------------------------------------------------------------------------
    def _req(self, method, path, body=None, raw=None, content_type=None, accept_bytes=False):
        headers = {"X-Requested-With": "AeroSculpt", "User-Agent": f"aerosculpt-client/{__version__}"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = None
        if raw is not None:
            data = raw
            headers["Content-Type"] = content_type or "application/octet-stream"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get("error", e.reason)
            except Exception:  # noqa: BLE001
                msg = e.reason
            raise AeroSculptError(e.code, msg) from None
        if accept_bytes:
            return payload
        return json.loads(payload.decode()) if payload else None

    # ---- projects -----------------------------------------------------------------------------------
    def me(self):
        return self._req("GET", "/api/me")

    def projects(self):
        return self._req("GET", "/api/projects")

    def upload(self, path, name=None, units=None, tunnel=None):
        path = Path(path)
        q = {"filename": path.name, "name": name or path.stem}
        if tunnel:
            q["tunnel"] = tunnel
        p = self._req("POST", "/api/projects?" + urllib.parse.urlencode(q), raw=path.read_bytes())
        if units:
            p = self.settings(p["id"], units=units)
        return p

    def from_template(self, template, name=None, values=None):
        return self._req("POST", "/api/projects", {"template": template, "name": name, "values": values or {}})

    def settings(self, project_id, **settings):
        return self._req("PATCH", f"/api/projects/{project_id}", {"settings": settings})

    def project(self, project_id):
        return self._req("GET", f"/api/projects/{project_id}")

    # ---- jobs ---------------------------------------------------------------------------------------
    def _job(self, project_id, body):
        return self._req("POST", f"/api/projects/{project_id}/jobs", body)

    def analyze(self, project_id, fidelity="draft", grid_study=False, **extra):
        return self._job(project_id, {"kind": "analyze", "fidelity": fidelity, "grid_study": grid_study, **extra})

    def optimize(self, project_id, fidelity="draft", goal="drag", iterations=None, **options):
        o = {"objective": goal, "mode": options.pop("mode", "auto"), **options}
        if iterations:
            o["iterations"] = int(iterations)
        return self._job(project_id, {"kind": "optimize", "fidelity": fidelity, "optimize": o})

    def estimate(self, project_id, body):
        return self._req("POST", f"/api/projects/{project_id}/estimate", body)

    def jobs(self, project_id):
        return self._req("GET", f"/api/projects/{project_id}/jobs")

    def job(self, job_id):
        return self._req("GET", f"/api/jobs/{job_id}")

    def result(self, job_id):
        return self._req("GET", f"/api/jobs/{job_id}/files/result.json")

    def cancel(self, job_id):
        return self._req("POST", f"/api/jobs/{job_id}/cancel", {})

    def wait(self, job_id, poll=5.0, quiet=False):
        last = None
        while True:
            j = self.job(job_id)
            line = f"{j['status']:9s} {100 * (j.get('progress') or 0):5.1f}%  {j.get('stage') or ''}"
            if not quiet and line != last:
                print(line, flush=True)
                last = line
            if j["status"] in ("done", "failed", "canceled"):
                if j["status"] == "failed":
                    raise AeroSculptError("job failed", j.get("error") or "")
                return j
            time.sleep(poll)

    def download(self, job_id, dest, fmt="pdf"):
        """fmt: pdf (report), step (CAD), vtk (flow fields, zip), stl (optimized mesh), json (result)."""
        if fmt == "stl":
            data = self._req("GET", f"/api/jobs/{job_id}/files/optimized.stl", accept_bytes=True)
        elif fmt == "json":
            data = self._req("GET", f"/api/jobs/{job_id}/files/result.json", accept_bytes=True)
        else:
            data = self._req("GET", f"/api/jobs/{job_id}/export/{fmt}", accept_bytes=True)
        Path(dest).write_bytes(data)
        return dest

    def mission(self, job_id, **inputs):
        return self._req("POST", f"/api/jobs/{job_id}/mission", inputs)

    def compare(self, job_a, job_b):
        """Side-by-side summary of two finished jobs (drag at every condition)."""
        out = []
        ra, rb = self.result(job_a), self.result(job_b)
        pa = (ra.get("best") or ra["baseline"])["polar"]
        pb = (rb.get("best") or rb["baseline"])["polar"]
        for qa, qb in zip(pa, pb):
            for sa, sb in zip(qa["speeds"], qb["speeds"]):
                out.append({"aoa": qa["aoa"], "v": sa["v"], "drag_a": sa["drag_N"], "drag_b": sb["drag_N"],
                            "change": sb["drag_N"] / sa["drag_N"] - 1.0})
        return out


# ---- command line --------------------------------------------------------------------------------
EFFORT = {"quick": 32, "balanced": 60, "thorough": 120}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aerosculpt", description="AeroSculpt command line")
    ap.add_argument("--url", help="server (default $AEROSCULPT_URL or http://localhost:8000)")
    ap.add_argument("--key", help="API key (default $AEROSCULPT_KEY)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("me")
    sub.add_parser("projects")
    u = sub.add_parser("upload")
    u.add_argument("file")
    u.add_argument("--name")
    u.add_argument("--units", choices=["mm", "cm", "m", "in", "ft"])
    u.add_argument("--tunnel", choices=["general", "rc", "supersonic"])
    for name in ("analyze", "optimize"):
        s = sub.add_parser(name)
        s.add_argument("project")
        s.add_argument("--fidelity", default="draft", choices=["draft", "standard", "high"])
        s.add_argument("--wait", action="store_true")
        if name == "analyze":
            s.add_argument("--grid-study", action="store_true")
        else:
            s.add_argument("--goal", default="drag", choices=["drag", "ld", "drag_keep_lift", "drag_fixed_lift"])
            s.add_argument("--effort", default="balanced", choices=list(EFFORT))
            s.add_argument("--iterations", type=int)
            s.add_argument("--mass-kg", type=float)
    j = sub.add_parser("jobs")
    j.add_argument("project")
    w = sub.add_parser("wait")
    w.add_argument("job")
    d = sub.add_parser("download")
    d.add_argument("job")
    d.add_argument("fmt", choices=["pdf", "step", "vtk", "stl", "json"])
    d.add_argument("dest")
    c = sub.add_parser("compare")
    c.add_argument("job_a")
    c.add_argument("job_b")
    sw = sub.add_parser("sweep", help="analyses of a template project over values of one dimension")
    sw.add_argument("project")
    sw.add_argument("--param", required=True, help="name=v1,v2,...")
    sw.add_argument("--fidelity", default="draft")
    a = ap.parse_args(argv)
    api = AeroSculpt(a.url, a.key)
    try:
        if a.cmd == "me":
            print(json.dumps(api.me(), indent=1))
        elif a.cmd == "projects":
            for p in api.projects():
                print(f"{p['id']}  {p['name']}")
        elif a.cmd == "upload":
            p = api.upload(a.file, a.name, a.units, a.tunnel)
            print(p["id"])
        elif a.cmd in ("analyze", "optimize"):
            if a.cmd == "analyze":
                job = api.analyze(a.project, a.fidelity, a.grid_study)
            else:
                extra = {"mass_kg": a.mass_kg} if a.mass_kg else {}
                job = api.optimize(a.project, a.fidelity, a.goal, a.iterations or EFFORT[a.effort], **extra)
            print(job["id"], f"(up to {job.get('estimated_credits')} credits)")
            if a.wait:
                api.wait(job["id"])
        elif a.cmd == "jobs":
            for x in api.jobs(a.project):
                print(f"{x['id']}  {x['kind']:8s} {x['status']:8s} {x.get('stage') or ''}")
        elif a.cmd == "wait":
            api.wait(a.job)
        elif a.cmd == "download":
            print(api.download(a.job, a.dest, a.fmt))
        elif a.cmd == "compare":
            for r in api.compare(a.job_a, a.job_b):
                print(f"{r['aoa']:6g}°  {r['v']:7g} m/s   {r['drag_a']:9.4g} N -> {r['drag_b']:9.4g} N   {100 * r['change']:+6.1f} %")
        elif a.cmd == "sweep":
            name, vals = a.param.split("=", 1)
            proj = api.project(a.project)
            tm = proj["settings"].get("template")
            if not tm:
                raise AeroSculptError(400, "sweeps change a template dimension: use a template project")
            for v in [float(x) for x in vals.split(",")]:
                api.settings(a.project, template={**tm, "values": {**tm["values"], name: v}})
                job = api.wait(api.analyze(a.project, a.fidelity)["id"], quiet=True)
                r = api.result(job["id"])
                s0 = r["baseline"]["polar"][0]["speeds"][0]
                print(f"{name} = {v:g}: Cd {s0['cd']:.4f}, drag {s0['drag_N']:.4g} N (job {job['id']})", flush=True)
            api.settings(a.project, template=tm)
    except AeroSculptError as e:
        print(f"error {e.status}: {e.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
