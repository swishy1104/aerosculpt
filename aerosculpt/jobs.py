"""Job queue on top of the jobs table, and the worker loop that runs CFD jobs."""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import socket
import time
import traceback
import uuid

from .config import settings
from .db import add_usage, audit, db, transaction
from .plans import CREDITS_PER_RUN

STALE_AFTER_S = 900


def job_dir(job_id):
    return settings.data_dir / "jobs" / job_id


def project_dir(project_id):
    return settings.data_dir / "projects" / project_id


def enqueue(con, user, project, spec, plan, estimated_credits):
    job_id = uuid.uuid4().hex[:16]
    con.execute(
        """INSERT INTO jobs (id, project_id, user_id, kind, status, spec, priority, created_at, estimated_credits)
           VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)""",
        (job_id, project["id"], user["id"], spec.get("kind", "analyze"), json.dumps(spec),
         plan["priority"], time.time(), estimated_credits))
    con.execute("UPDATE projects SET updated_at=? WHERE id=?", (time.time(), project["id"]))
    audit(con, user["id"], "job_queued", f"{job_id} {spec.get('kind')} est={estimated_credits}")
    return job_id


def summarize(result):
    """Small dictionary shown on dashboards and run lists."""
    if not result:
        return None
    env = result["envelope"]
    base = result["baseline"]["summary"]
    out = {"angles": len(env["aoas"]), "speeds": len(env["speeds"]), "regimes": result["setup"]["regimes"],
           "tunnel": result.get("tunnel", "rc" if result.get("propellers") else "general")}
    if out["tunnel"] == "supersonic":
        out["machs"] = len(env["machs"])
        out["tw_max"] = base["tw_max"] if result["kind"] == "analyze" else result["best"]["summary"]["tw_max"]
        out["over_limit"] = base["over_limit"] if result["kind"] == "analyze" else result["best"]["summary"]["over_limit"]
    if base.get("top_speed"):
        out["top_speed"] = base["top_speed"]
    if result["kind"] == "analyze":
        out.update({"cd": base["cd_at_min_aoa"], "ld_max": base["ld_max"], "drag_mean_N": base["drag_mean_N"]})
    else:
        out.update({"improvement": result["improvement"], "drag_change": result["drag_change"],
                    "objective": result["objective"], "verified": bool(result.get("verified")),
                    "ld_max": result["best"]["summary"]["ld_max"]})
    return out


def claim_next(worker_id):
    with transaction() as con:
        row = con.execute("""SELECT id FROM jobs WHERE status='queued' AND cancel_requested=0
                             ORDER BY priority DESC, created_at LIMIT 1""").fetchone()
        if not row:
            return None
        now = time.time()
        con.execute("""UPDATE jobs SET status='running', started_at=?, heartbeat=?, worker=?, stage='Starting'
                       WHERE id=? AND status='queued'""", (now, now, worker_id, row["id"]))
        return con.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()


def _pid_alive(pid):
    """Whether a process exists (without signalling it: on Windows os.kill(pid, 0) would send it
    CTRL+C)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def recover_orphans(worker_id):
    """At start: jobs left 'running' by a process on this machine that no longer exists (a server
    restart, a crash) go straight back to the queue, instead of waiting STALE_AFTER_S. The
    result cache then replays every CFD run they had finished, so they resume nearly where they
    stopped. Returns the requeued job ids."""
    host = worker_id.rsplit(":", 1)[0]
    out = []
    with db() as con:
        for row in con.execute("SELECT id, worker, log FROM jobs WHERE status='running'").fetchall():
            w = row["worker"] or ""
            h, _, pid = w.rpartition(":")
            if h != host or w == worker_id or not pid.isdigit() or _pid_alive(int(pid)):
                continue
            con.execute("UPDATE jobs SET status='queued', progress=0, stage='Queued (resuming after a restart)', "
                        "log=log || '[resumed after a restart: finished runs come from the cache]\n' WHERE id=?",
                        (row["id"],))
            out.append(row["id"])
    return out


def release_jobs(worker_id):
    """On a clean shutdown: this worker's running job goes back to the queue for the next start."""
    with db() as con:
        cur = con.execute("UPDATE jobs SET status='queued', progress=0, stage='Queued (resuming after a restart)' "
                          "WHERE status='running' AND worker=?", (worker_id,))
        return cur.rowcount


def recover_stale():
    """Jobs whose worker died mid-run: requeue once, then give up."""
    cutoff = time.time() - STALE_AFTER_S
    with db() as con:
        for row in con.execute("SELECT id, log FROM jobs WHERE status='running' AND heartbeat < ?", (cutoff,)).fetchall():
            if "[requeued]" in (row["log"] or ""):
                con.execute("UPDATE jobs SET status='failed', error='The worker stopped while running this job.', "
                            "finished_at=? WHERE id=?", (time.time(), row["id"]))
            else:
                con.execute("UPDATE jobs SET status='queued', progress=0, stage='Queued', log=log || '[requeued]\n' "
                            "WHERE id=?", (row["id"],))


class JobContext:
    """Callbacks handed to the Runner; batches database writes."""

    def __init__(self, job):
        self.id = job["id"]
        self.last_write = 0.0
        self.last_cancel_check = 0.0
        self.cancelled = False
        self.lines = []
        self.pending = None
        self.live = {}

    def progress(self, frac, stage, extra=None):
        now = time.time()
        if extra:
            self.live.update(extra)   # merge: a live Cd update must not erase the history
        self.pending = (frac, stage, dict(self.live) if self.live else None)
        if now - self.last_write < 1.0 and frac < 1.0:
            return
        self.flush()

    def flush(self):
        if self.pending is None and not self.lines:
            return
        frac, stage, extra = self.pending or (None, None, None)
        live = json.dumps(extra, default=float) if extra is not None else None
        with db() as con:
            if frac is not None:
                con.execute("""UPDATE jobs SET progress=?, stage=?, heartbeat=?,
                               live=COALESCE(?, live) WHERE id=?""",
                            (float(frac), stage, time.time(), live, self.id))
            if self.lines:
                con.execute("UPDATE jobs SET log = substr(log || ?, -20000), heartbeat=? WHERE id=?",
                            ("\n".join(self.lines) + "\n", time.time(), self.id))
        self.lines = []
        self.pending = None
        self.last_write = time.time()

    def log(self, msg):
        self.lines.append(msg)
        if len(self.lines) > 20:
            self.flush()

    def cancel(self):
        now = time.time()
        if now - self.last_cancel_check > 2.0:
            self.last_cancel_check = now
            with db() as con:
                r = con.execute("SELECT cancel_requested FROM jobs WHERE id=?", (self.id,)).fetchone()
                self.cancelled = bool(r and r["cancel_requested"])
        return self.cancelled


def run_job(job):
    from .optimize.runner import JobCancelled, Runner

    spec = json.loads(job["spec"])
    ctx = JobContext(job)
    with db() as con:
        project = con.execute("SELECT * FROM projects WHERE id=?", (job["project_id"],)).fetchone()
    if project is None:
        with db() as con:
            con.execute("UPDATE jobs SET status='failed', error='Project was deleted.', finished_at=? WHERE id=?",
                        (time.time(), job["id"]))
        return
    out = job_dir(job["id"])
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    model = project_dir(project["id"]) / ("model." + project["filename"].rsplit(".", 1)[-1].lower())
    tunnel = spec.get("tunnel", "general")
    if tunnel == "supersonic":
        from .hyper.runner import HyperRunner as runner_cls
    elif tunnel == "rc":
        from .rc.runner import RCRunner as runner_cls
    else:
        runner_cls = Runner
    runner = runner_cls(spec, model, project["filename"], out, progress=ctx.progress, log=ctx.log, cancel=ctx.cancel)
    status, error, result = "done", None, None
    try:
        result = runner.run()
    except JobCancelled:
        status = "canceled"
    except (ValueError, PermissionError) as e:
        status, error = "failed", str(e)
    except FloatingPointError:
        status, error = "failed", ("The flow solution became unstable for this geometry. Try a lower "
                                   "fidelity, a smaller angle of attack, or check the model for defects.")
    except MemoryError:
        status, error = "failed", "The server ran out of memory for this tunnel size. Try a lower fidelity."
    except Exception as e:  # noqa: BLE001 - report, never crash the worker
        status, error = "failed", f"Internal error: {type(e).__name__}. Our team has been notified."
        ctx.log(traceback.format_exc())
    ctx.flush()
    if hasattr(runner, "credits"):
        credits = runner.credits() if getattr(runner, "evaluations", 0) else 0
    else:
        credits = math.ceil(sum(n * CREDITS_PER_RUN.get(f, 1) for f, n in runner.runs_by_fid.items()))
        credits += getattr(runner, "model_credits", 0)     # wing-model designs: one credit per ten
    if job["estimated_credits"]:
        credits = min(credits, int(job["estimated_credits"]))   # the quote is "up to": never charge more
    with db() as con:
        con.execute("""UPDATE jobs SET status=?, error=?, finished_at=?, progress=CASE WHEN ?='done' THEN 1 ELSE progress END,
                       stage=?, summary=?, cfd_runs=?, credits_used=? WHERE id=?""",
                    (status, error, time.time(), status, {"done": "Done", "canceled": "Cancelled"}.get(status, "Failed"),
                     json.dumps(summarize(result)) if result else None, runner.cfd_runs, credits, job["id"]))
        add_usage(con, job["user_id"], credits=credits, cfd_runs=runner.cfd_runs, jobs=1)


def purge_guests(now=None):
    """Delete guest-demo accounts older than GUEST_HOURS with everything they made (rows cascade;
    files are removed here). A guest whose job is still running is kept until it ends."""
    cutoff = (now or time.time()) - settings.guest_hours * 3600
    with db() as con:
        rows = con.execute("""SELECT id FROM users WHERE billing_source='guest' AND created_at < ? AND id NOT IN
                              (SELECT user_id FROM jobs WHERE status IN ('queued', 'running'))""", (cutoff,)).fetchall()
        gone = []
        for r in rows:
            pids = [p["id"] for p in con.execute("SELECT id FROM projects WHERE user_id=?", (r["id"],))]
            jids = [j["id"] for j in con.execute("SELECT id FROM jobs WHERE user_id=?", (r["id"],))]
            con.execute("DELETE FROM users WHERE id=?", (r["id"],))
            gone.append((pids, jids))
    for pids, jids in gone:
        for pid in pids:
            shutil.rmtree(project_dir(pid), ignore_errors=True)
        for jid in jids:
            shutil.rmtree(job_dir(jid), ignore_errors=True)
    return len(gone)


def warm_up():
    """Compile the numba kernels once so the first customer job does not pay for it."""
    from .cfd.analysis import Conditions, Fidelity, analyze
    from .geometry.samples import icosphere
    from .geometry.zones import Zone, _inside_counts
    import numpy as np
    analyze(icosphere(0.5, 2), Conditions(speed=1.0), Fidelity("warmup", 20_000, 1e3, extra_chunks=0, avg_chunks=1))
    m = icosphere(0.5, 1)
    _inside_counts(np.ascontiguousarray(m.triangles()), np.zeros(m.n_faces, dtype=np.int64), 1, np.zeros((1, 3)))
    record_throughput()


def record_throughput():
    """Benchmark the tunnel once and store it for the run-time estimates (web.compute_info)."""
    from .cfd.analysis import throughput
    info = throughput()
    tmp = settings.data_dir / "compute.json.tmp"
    tmp.write_text(json.dumps(info))
    tmp.replace(settings.data_dir / "compute.json")
    logging.getLogger(__name__).info("tunnel: %s%s, %.0f million cell updates/s", info["backend"],
                                     f" ({info['device']})" if info.get("device") else "", info["mlups"])


def worker_id():
    return f"{socket.gethostname()}:{os.getpid()}"


def worker_loop(stop=None, poll_s=1.0):
    wid = worker_id()
    try:
        back = recover_orphans(wid)
        if back:
            logging.getLogger(__name__).info("resuming %d job(s) left running by a previous process", len(back))
    except Exception:  # noqa: BLE001 - housekeeping never stops the queue
        traceback.print_exc()
    try:
        warm_up()
    except Exception:  # never block the queue on warm-up
        traceback.print_exc()
    last_recover = 0.0
    while stop is None or not stop.is_set():
        if time.time() - last_recover > 60:
            recover_stale()
            try:
                purge_guests()
            except Exception:  # noqa: BLE001 - housekeeping never stops the queue
                traceback.print_exc()
            last_recover = time.time()
        job = claim_next(wid)
        if job is None:
            time.sleep(poll_s)
            continue
        run_job(job)
