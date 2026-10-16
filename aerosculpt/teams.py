"""Teams: shared projects with roles, invitations by email, and a team audit trail.

Roles, strongest first:
* owner  - everything, including deleting the team and changing anyone's role
* admin  - invite and remove members, change roles below admin, read the audit trail
* editor - open, edit and run jobs on the team's projects (jobs bill the person who starts them)
* viewer - open the team's projects and their results, download reports

A project belongs to its creator; sharing it with a team gives the members access by role.
"""
from __future__ import annotations

import time

ROLES = ("viewer", "editor", "admin", "owner")
RANK = {r: k for k, r in enumerate(ROLES)}


def migrate(con):
    con.execute("""CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, owner_id INTEGER NOT NULL REFERENCES users(id),
        created_at REAL NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS team_members (
        team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        role TEXT NOT NULL, added_at REAL NOT NULL, PRIMARY KEY (team_id, user_id))""")
    con.execute("""CREATE TABLE IF NOT EXISTS team_invites (
        team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE, email TEXT NOT NULL, role TEXT NOT NULL,
        invited_by INTEGER, created_at REAL NOT NULL, PRIMARY KEY (team_id, email))""")
    con.execute("CREATE INDEX IF NOT EXISTS team_members_user ON team_members(user_id)")


def role_of(con, team_id, user_id):
    if team_id is None:
        return None
    row = con.execute("SELECT role FROM team_members WHERE team_id=? AND user_id=?", (team_id, user_id)).fetchone()
    return row["role"] if row else None


def can(role, need):
    return role is not None and RANK.get(role, -1) >= RANK[need]


def teams_of(con, user_id):
    rows = con.execute("""SELECT t.id, t.name, t.owner_id, m.role FROM teams t JOIN team_members m ON m.team_id = t.id
                          WHERE m.user_id=? ORDER BY t.name""", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def team_ids(con, user_id):
    return [r["team_id"] for r in con.execute("SELECT team_id FROM team_members WHERE user_id=?", (user_id,))]


def create(con, owner_id, name):
    now = time.time()
    tid = con.execute("INSERT INTO teams (name, owner_id, created_at) VALUES (?,?,?)", (name, owner_id, now)).lastrowid
    con.execute("INSERT INTO team_members (team_id, user_id, role, added_at) VALUES (?,?,?,?)", (tid, owner_id, "owner", now))
    return tid


def members(con, team_id):
    rows = con.execute("""SELECT u.id, u.email, u.name, m.role, m.added_at FROM team_members m JOIN users u ON u.id = m.user_id
                          WHERE m.team_id=? ORDER BY m.added_at""", (team_id,)).fetchall()
    inv = con.execute("SELECT email, role, created_at FROM team_invites WHERE team_id=? ORDER BY created_at", (team_id,)).fetchall()
    return [dict(r) for r in rows], [dict(r) for r in inv]


def invite(con, team_id, email, role, by):
    """Adds an existing account at once; otherwise the invitation waits for that email to sign in."""
    user = con.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if user:
        con.execute("""INSERT INTO team_members (team_id, user_id, role, added_at) VALUES (?,?,?,?)
                       ON CONFLICT(team_id, user_id) DO UPDATE SET role=excluded.role""", (team_id, user["id"], role, time.time()))
        return "added"
    con.execute("""INSERT INTO team_invites (team_id, email, role, invited_by, created_at) VALUES (?,?,?,?,?)
                   ON CONFLICT(team_id, email) DO UPDATE SET role=excluded.role""", (team_id, email, role, by, time.time()))
    return "invited"


def accept_invites(con, user_id, email):
    """On sign-up or sign-in: join every team that invited this email."""
    rows = con.execute("SELECT team_id, role FROM team_invites WHERE email=?", (email,)).fetchall()
    for r in rows:
        con.execute("""INSERT OR IGNORE INTO team_members (team_id, user_id, role, added_at) VALUES (?,?,?,?)""",
                    (r["team_id"], user_id, r["role"], time.time()))
    con.execute("DELETE FROM team_invites WHERE email=?", (email,))
    return len(rows)
