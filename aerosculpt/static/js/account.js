import { api, appTop, busy, esc, h, handleError, modal, requireAuth, timeAgo, toast, $ } from "/static/js/core.js";

let me = await requireAuth();
appTop(me, [{ label: "Account" }]);

const pf = $("#profile");
pf.name.value = me.name || "";
pf.email.value = me.email;
pf.addEventListener("submit", async (e) => {
  e.preventDefault();
  const b = pf.querySelector("button");
  busy(b, true);
  try { me = await api("/api/me", { method: "PATCH", body: { name: pf.name.value } }); toast("Profile saved.", "good"); }
  catch (err) { handleError(err); } finally { busy(b, false); }
});

const pw = $("#password");
pw.addEventListener("submit", async (e) => {
  e.preventDefault();
  const b = pw.querySelector("button");
  busy(b, true);
  try {
    await api("/api/me/password", { method: "POST", body: { current: pw.current.value, new: pw.new.value } });
    pw.reset();
    toast("Password changed.", "good");
  } catch (err) { handleError(err); } finally { busy(b, false); }
});

function renderApi() {
  const box = $("#api-body");
  if (!me.limits.api) {
    box.innerHTML = `<a class="btn" href="/app/billing?upgrade=team">Upgrade to Team for API access</a>`;
    return;
  }
  box.innerHTML = me.has_api_key
    ? `<div class="row wrap"><code>${esc(me.api_key_prefix)}…</code><button class="btn sm" id="rot" type="button">Generate a new key</button><button class="btn sm danger" id="rev" type="button">Revoke</button></div>`
    : `<button class="btn" id="rot" type="button">Generate API key</button>`;
  $("#rot").addEventListener("click", async () => {
    try {
      const r = await api("/api/me/api-key", { method: "POST" });
      me = await api("/api/me");
      renderApi();
      const body = h("div");
      body.innerHTML = `<p>Copy this key now — it is shown only once. Generating a new key revokes the old one.</p><pre style="white-space:pre-wrap;word-break:break-all">${esc(r.key)}</pre>`;
      modal({ title: "Your API key", body, actions: [{ label: "Copy", kind: "primary", onClick: () => { navigator.clipboard.writeText(r.key); toast("Copied."); return true; } }, { label: "Done" }] });
    } catch (err) { handleError(err); }
  });
  const rev = $("#rev");
  if (rev) rev.addEventListener("click", async () => {
    try { await api("/api/me/api-key", { method: "DELETE" }); me = await api("/api/me"); renderApi(); toast("API key revoked."); }
    catch (err) { handleError(err); }
  });
}
renderApi();

$("#delete").addEventListener("click", () => {
  const body = h("div");
  body.innerHTML = `<p>Type your password to confirm. Everything is deleted immediately.</p>
    <label class="field"><span>Password</span><input type="password" id="del-pw" autocomplete="current-password"></label>`;
  modal({
    title: "Delete your account?", body,
    actions: [{ label: "Cancel", kind: "ghost" }, {
      label: "Delete everything", kind: "danger",
      onClick: async (btn) => {
        busy(btn, true);
        try {
          await api("/api/me/delete", { method: "POST", body: { password: $("#del-pw").value } });
          location.href = "/";
        } catch (err) { handleError(err); busy(btn, false); return true; }
      },
    }],
  });
});

// ---- teams ----------------------------------------------------------------------------------------------------
const ROLES = ["viewer", "editor", "admin", "owner"];
async function renderTeams() {
  const box = $("#teams-body");
  let teams = [];
  try { teams = await api("/api/teams"); } catch (e) { box.innerHTML = `<p class="small muted">${esc(e.message)}</p>`; return; }
  const canCreate = me.plan === "team" || me.is_admin;
  box.innerHTML = teams.map((t) => {
    const admin = ["admin", "owner"].includes(t.role);
    return `<div class="team" data-id="${t.id}" style="border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:10px 0">
      <div class="row spread"><b>${esc(t.name)}</b><span class="badge">${esc(t.role)}</span></div>
      <table class="tbl compact" style="margin-top:6px"><tbody>
        ${t.members.map((m) => `<tr><td>${esc(m.name || m.email)}<div class="small muted">${esc(m.email)}</div></td>
          <td class="r">${admin && m.id !== me.id ? `<select data-user="${m.id}" class="role">${ROLES.map((r) => `<option ${r === m.role ? "selected" : ""}>${r}</option>`).join("")}</select>
            <button class="btn sm ghost rm" data-user="${m.id}" type="button">Remove</button>` : esc(m.role)}</td></tr>`).join("")}
        ${t.invites.map((i) => `<tr><td>${esc(i.email)}<div class="small muted">invited ${timeAgo(i.created_at)} · joins at first sign-in</div></td><td class="r">${esc(i.role)}</td></tr>`).join("")}
      </tbody></table>
      ${admin ? `<form class="row invite" style="gap:6px;margin-top:8px"><input type="email" name="email" placeholder="colleague@company.com" required style="flex:1">
        <select name="role">${ROLES.slice(0, 3).map((r) => `<option ${r === "editor" ? "selected" : ""}>${r}</option>`).join("")}</select>
        <button class="btn sm" type="submit">Invite</button></form>
        <button class="btn sm ghost audit" type="button" style="margin-top:6px">Audit trail</button><div class="audit-box"></div>` : ""}
    </div>`;
  }).join("") + (canCreate ? `<form id="new-team" class="row" style="gap:6px;margin-top:10px"><input name="name" placeholder="New team name" required style="flex:1">
    <button class="btn sm" type="submit">Create team</button></form>` : `<p class="small muted">${teams.length ? "" : "You are in no team yet. "}Creating teams is part of the Team plan.</p>`);
  const nt = $("#new-team");
  if (nt) nt.addEventListener("submit", async (e) => {
    e.preventDefault();
    try { await api("/api/teams", { method: "POST", body: { name: nt.name.value } }); toast("Team created.", "good"); renderTeams(); } catch (err) { handleError(err); }
  });
  box.querySelectorAll(".team").forEach((el) => {
    const tid = el.dataset.id;
    const inv = el.querySelector(".invite");
    if (inv) inv.addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const r = await api(`/api/teams/${tid}/members`, { method: "POST", body: { email: inv.email.value, role: inv.role.value } });
        toast(r.status === "added" ? "Added to the team." : "Invitation saved: they join when they first sign in.", "good");
        renderTeams();
      } catch (err) { handleError(err); }
    });
    el.querySelectorAll("select.role").forEach((sel) => sel.addEventListener("change", async () => {
      try { await api(`/api/teams/${tid}/members`, { method: "PATCH", body: { user_id: +sel.dataset.user, role: sel.value } }); toast("Role changed."); } catch (err) { handleError(err); renderTeams(); }
    }));
    el.querySelectorAll("button.rm").forEach((b) => b.addEventListener("click", async () => {
      try { await api(`/api/teams/${tid}/members`, { method: "DELETE", body: { user_id: +b.dataset.user } }); renderTeams(); } catch (err) { handleError(err); }
    }));
    const au = el.querySelector(".audit");
    if (au) au.addEventListener("click", async () => {
      try {
        const rows = await api(`/api/teams/${tid}/audit`);
        el.querySelector(".audit-box").innerHTML = rows.length ? `<table class="tbl compact" style="margin-top:6px"><tbody>${rows.slice(0, 100).map((r) =>
          `<tr><td class="small muted" style="white-space:nowrap">${new Date(r.at * 1000).toLocaleString()}</td><td class="small">${esc(r.email || "—")}</td>
           <td class="small">${esc(r.action.replace(/_/g, " "))}</td><td class="small muted">${esc(r.detail || "")}</td></tr>`).join("")}</tbody></table>`
          : `<p class="small muted">Nothing recorded yet.</p>`;
      } catch (err) { handleError(err); }
    });
  });
}
renderTeams();
