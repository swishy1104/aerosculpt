"""AeroSculpt command line.

    python run.py serve              web server + job workers (default)
    python run.py web                web server only (run workers elsewhere)
    python run.py worker             one job worker
    python run.py init               create the database
    python run.py make-admin EMAIL   grant admin to an existing account
    python run.py validate           run the solver validation suite (writes validation_results.json)
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys


def worker_main():
    from aerosculpt.config import ensure_dirs
    from aerosculpt.db import init_db
    from aerosculpt.jobs import worker_loop
    logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(message)s")
    ensure_dirs()
    init_db()
    worker_loop()


def main(argv=None):
    ap = argparse.ArgumentParser(description="AeroSculpt server")
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "web", "worker", "init", "make-admin", "validate"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from aerosculpt.config import ensure_dirs, settings
    from aerosculpt.db import db, init_db
    ensure_dirs()
    init_db()

    if a.command == "init":
        print(f"database ready at {settings.db_path}")
        return
    if a.command == "make-admin":
        with db() as con:
            n = con.execute("UPDATE users SET is_admin=1 WHERE email=?", ((a.arg or "").lower(),)).rowcount
        print("ok" if n else "no such user")
        return
    if a.command == "validate":
        from aerosculpt.cfd.validation import run_groups, run_suite
        if a.arg:   # e.g. "validate supersonic,rc": just those tunnels, merged into the stored results
            run_groups(a.arg.split(","), verbose=True)
        else:
            run_suite(verbose=True)
        return
    if a.command == "worker":
        worker_main()
        return

    import uvicorn
    procs = []
    if a.command == "serve":
        n = a.workers if a.workers is not None else settings.workers
        for _ in range(max(0, n)):
            p = mp.Process(target=worker_main, daemon=True)
            p.start()
            procs.append(p)
    try:
        uvicorn.run("aerosculpt.web:app", host=a.host or settings.host, port=a.port or settings.port,
                    proxy_headers=True, forwarded_allow_ips=settings.forwarded_allow_ips, log_level="info")
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
