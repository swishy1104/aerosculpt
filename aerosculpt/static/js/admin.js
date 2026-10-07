import { api, appTop, esc, fmtDate, fmtInt, handleError, requireAuth, timeAgo, $ } from "/static/js/core.js";

const me = await requireAuth();
appTop(me, [{ label: "Admin" }]);
try {
  const s = await api("/api/admin/stats");
  const tile = (label, value, sub = "") => `<div class="card stat-tile"><div class="label">${label}</div><div class="value">${value}</div><div class="small muted">${sub}</div></div>`;
  $("#tiles").innerHTML = tile("Monthly recurring revenue", "$" + fmtInt(s.mrr), s.billing_mode === "demo" ? "demo billing — not real revenue" : "from active paid plans")
    + tile("Users", fmtInt(s.users), `${fmtInt(s.trialing)} on trial`)
    + tile("Paying", fmtInt((s.paying.pro || 0) + (s.paying.team || 0)), `Pro ${s.paying.pro || 0} · Team ${s.paying.team || 0}`)
    + tile("Compute this month", fmtInt(s.credits_this_month) + " credits", `${fmtInt(s.cfd_runs_this_month)} CFD runs`);
  const rows = (obj) => Object.entries(obj).map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${fmtInt(v)}</span>`).join("") || '<span class="k">none</span>';
  $("#jobs").innerHTML = `<div class="kv">${rows(s.jobs)}</div>`;
  $("#plans").innerHTML = `<div class="kv">${rows(s.by_plan)}</div>`;
  $("#recent").innerHTML = `<thead><tr><th>Email</th><th>Plan</th><th>Status</th><th>Joined</th><th>Last sign-in</th></tr></thead><tbody>
    ${s.recent_users.map((u) => `<tr><td>${esc(u.email)}</td><td>${esc(u.plan)}</td><td>${esc(u.plan_status)}</td>
      <td>${fmtDate(u.created_at)}</td><td>${u.last_login_at ? timeAgo(u.last_login_at) : "—"}</td></tr>`).join("")}</tbody>`;
} catch (e) {
  handleError(e);
  $("#tiles").innerHTML = `<div class="card muted">Admins only.</div>`;
}
