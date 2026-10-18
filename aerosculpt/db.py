"""SQLite persistence (WAL mode, so the web process and job workers share one file).

The SQL is deliberately plain so moving to Postgres later is a driver change plus the
few `INSERT OR IGNORE` / `ON CONFLICT` statements below.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

from .config import settings

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        password_hash TEXT NOT NULL,
        created_at REAL NOT NULL,
        plan TEXT NOT NULL DEFAULT 'lite',
        plan_status TEXT NOT NULL DEFAULT 'active',
        trial_ends_at REAL,
        trial_used INTEGER NOT NULL DEFAULT 0,
        current_period_end REAL,
        cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
        stripe_customer_id TEXT,
        stripe_subscription_id TEXT,
        billing_source TEXT,
        is_admin INTEGER NOT NULL DEFAULT 0,
        api_key_hash TEXT,
        api_key_prefix TEXT,
        last_login_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        ip TEXT, user_agent TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS password_resets (
        token_hash TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        expires_at REAL NOT NULL,
        used INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        filename TEXT NOT NULL,
        source_format TEXT,
        file_size INTEGER NOT NULL DEFAULT 0,
        settings TEXT NOT NULL DEFAULT '{}',
        metrics TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        spec TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        started_at REAL, finished_at REAL, heartbeat REAL,
        progress REAL NOT NULL DEFAULT 0,
        stage TEXT NOT NULL DEFAULT 'Queued',
        live TEXT NOT NULL DEFAULT '{}',
        log TEXT NOT NULL DEFAULT '',
        error TEXT,
        summary TEXT,
        estimated_credits INTEGER NOT NULL DEFAULT 0,
        credits_used INTEGER NOT NULL DEFAULT 0,
        cfd_runs INTEGER NOT NULL DEFAULT 0,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        worker TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, priority DESC, created_at)",
    "CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS projects_user ON projects(user_id, updated_at DESC)",
    """CREATE TABLE IF NOT EXISTS usage (
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        period TEXT NOT NULL,
        credits INTEGER NOT NULL DEFAULT 0,
        cfd_runs INTEGER NOT NULL DEFAULT 0,
        jobs INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (user_id, period)
    )""",
    """CREATE TABLE IF NOT EXISTS stripe_events (
        id TEXT PRIMARY KEY, type TEXT NOT NULL, received_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, action TEXT NOT NULL, detail TEXT, at REAL NOT NULL
    )""",
]


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    return con


@contextmanager
def db():
    con = connect()
    try:
        yield con
    finally:
        con.close()


@contextmanager
def transaction():
    con = connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _add_column(con, table, column, decl):
    cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db():
    from .teams import migrate as teams_migrate
    with db() as con:
        for stmt in SCHEMA:
            con.execute(stmt)
        # columns added after the first release (idempotent)
        _add_column(con, "projects", "team_id", "INTEGER")
        _add_column(con, "audit", "team_id", "INTEGER")
        _add_column(con, "users", "sso_subject", "TEXT")
        con.execute("UPDATE users SET plan='pro' WHERE plan='team'")      # the Team plan was folded into Pro
        teams_migrate(con)


def period_key(ts=None):
    return time.strftime("%Y-%m", time.gmtime(ts or time.time()))


def audit(con, user_id, action, detail="", team_id=None):
    con.execute("INSERT INTO audit (user_id, action, detail, at, team_id) VALUES (?, ?, ?, ?, ?)",
                (user_id, action, (detail or "")[:2000], time.time(), team_id))


def usage_for(con, user_id):
    row = con.execute("SELECT credits, cfd_runs, jobs FROM usage WHERE user_id=? AND period=?",
                      (user_id, period_key())).fetchone()
    return dict(row) if row else {"credits": 0, "cfd_runs": 0, "jobs": 0}


def add_usage(con, user_id, credits=0, cfd_runs=0, jobs=0):
    con.execute(
        """INSERT INTO usage (user_id, period, credits, cfd_runs, jobs) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(user_id, period) DO UPDATE SET credits = credits + excluded.credits,
           cfd_runs = cfd_runs + excluded.cfd_runs, jobs = jobs + excluded.jobs""",
        (user_id, period_key(), credits, cfd_runs, jobs))
