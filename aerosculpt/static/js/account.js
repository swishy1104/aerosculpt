import { api, appTop, busy, esc, h, handleError, icon, modal, requireAuth, timeAgo, toast, $ } from "/static/js/core.js";

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
    box.innerHTML = `<a class="btn" href="/app/billing?upgrade=pro">Upgrade to Pro for API access</a>`;
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
  const canCreate = me.limits.api || me.is_admin;
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
    <button class="btn sm" type="submit">Create team</button></form>` : `<p class="small muted">${teams.length ? "" : "You are in no team yet. "}Creating teams is part of Pro.</p>`);
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


// ---- desktop app: licence keys, local settings, diagnostics ------------------------------------------------------
async function renderDesktop() {
  const box = $("#desktop-body");
  let lic;
  try { lic = await api("/api/license"); } catch (e) { box.innerHTML = `<p class="small muted">${esc(e.message)}</p>`; return; }
  const when = (ts) => new Date(ts * 1000).toLocaleDateString();
  if (!lic.local) {
    box.innerHTML = `<p class="small">AeroSculpt also runs as a <b>desktop app</b> on your own computer and GPU: nothing leaves the machine.
        A licence key unlocks your plan's features there${lic.can_issue ? "" : " (Pro)"}; without one it runs the free desktop tier.</p>
      ${lic.can_issue ? (lic.signing ? `<button class="btn" id="lic-issue" type="button">Generate a desktop licence key</button>
        <p class="small muted" style="margin:8px 0 0">Bound to your email, valid to the end of your billing period plus a 30-day grace period. Paste it in the desktop app under Account → Desktop app, or run <code>AeroSculpt.exe --license KEY</code>.</p>`
        : `<p class="small muted">Licence signing is not configured on this server.</p>`)
        : `<a class="btn" href="/app/billing?upgrade=pro">Upgrade to Pro for a desktop licence</a>`}
      <p class="small muted" style="margin:10px 0 0">Downloads and the GPU pack: <a href="https://github.com/swishy1104/aerosculpt/releases" target="_blank" rel="noopener">releases</a>.</p>`;
    const b = $("#lic-issue");
    if (b) b.addEventListener("click", async () => {
      try {
        const r = await api("/api/license/issue", { method: "POST" });
        const body = h("div");
        body.innerHTML = `<p>Your ${esc(r.plan)} desktop key, valid until ${when(r.expires)}:</p><pre style="white-space:pre-wrap;word-break:break-all">${esc(r.key)}</pre>`;
        modal({ title: "Desktop licence key", body, actions: [{ label: "Copy", kind: "primary", onClick: () => { navigator.clipboard.writeText(r.key); toast("Copied."); return true; } }, { label: "Done" }] });
      } catch (err) { handleError(err); }
    });
    return;
  }
  let ls = {};
  try { ls = await api("/api/local/settings"); } catch (e) { /* older server */ }
  const status = lic.installed
    ? (lic.valid ? `<span class="badge good"><span class="dot"></span>${esc(lic.plan)} licence</span> for ${esc(lic.email || "")}, until ${when(lic.expires)}${lic.why ? ` — ${esc(lic.why)}` : ""}`
      : `<span class="badge bad"><span class="dot"></span>licence problem</span> ${esc(lic.why || "")}`)
    : `<span class="badge"><span class="dot"></span>free desktop tier</span> — a Pro key unlocks the rest.`;
  box.innerHTML = `<p class="small">This copy runs on this computer (version ${esc(me.app_version || "")}, build ${esc(me.build || "")}).</p>
    <p class="small" style="margin:6px 0 10px">${status}</p>
    <div class="row" style="gap:8px;flex-wrap:wrap"><input type="text" id="lic-key" placeholder="Paste a licence key" style="flex:1;min-width:220px">
      <button class="btn sm" id="lic-install" type="button">Install key</button>${lic.installed ? `<button class="btn sm ghost" id="lic-remove" type="button">Remove</button>` : ""}</div>
    <p class="small muted" style="margin:8px 0 14px">Keys come from the account page of your AeroSculpt Pro subscription.</p>
    <span class="label">On this computer</span>
    <label class="check"><input type="checkbox" id="ls-fusion" ${ls.fusion_bridge ? "checked" : ""}><span><b>Send to Fusion</b> <span class="muted">talk to the FusionMCPBridge add-in when Fusion runs here</span></span></label>
    <label class="check"><input type="checkbox" id="ls-crash" ${ls.crash_reports !== false ? "checked" : ""}><span><b>Keep crash files</b> <span class="muted">for the diagnostic bundle; never sent by themselves</span></span></label>
    <label class="check"><input type="checkbox" id="ls-update" ${ls.check_updates !== false ? "checked" : ""}><span><b>Check for updates</b> <span class="muted">asks GitHub for the latest release; nothing is installed by itself</span></span></label>
    <label class="check"><input type="checkbox" id="ls-online" ${ls.online_motors !== false ? "checked" : ""}><span><b>Online motor search</b> <span class="muted">ThrustCurve.org, only when you press Search</span></span></label>
    <div class="row" style="gap:8px;margin-top:12px;flex-wrap:wrap">
      <a class="btn sm ghost" href="/api/diagnostics" download>${icon("download")}Save a diagnostic bundle</a>
      <button class="btn sm ghost" id="upd-check" type="button">${icon("refresh")}Check for updates now</button>
      <a class="btn sm ghost" href="/third-party-notices" target="_blank">Third-party notices</a></div>
    <p class="small muted" id="upd-status" style="margin:8px 0 0"></p>`;
  $("#lic-install").addEventListener("click", async () => {
    try { await api("/api/local/license", { method: "POST", body: { key: $("#lic-key").value.trim() } }); toast("Licence installed. Reload to see your plan.", "good"); location.reload(); }
    catch (err) { handleError(err); }
  });
  const rm = $("#lic-remove");
  if (rm) rm.addEventListener("click", async () => { try { await api("/api/local/license", { method: "DELETE" }); location.reload(); } catch (err) { handleError(err); } });
  for (const [id, key] of [["ls-fusion", "fusion_bridge"], ["ls-crash", "crash_reports"], ["ls-update", "check_updates"], ["ls-online", "online_motors"]]) {
    $("#" + id).addEventListener("change", async (e) => {
      try { await api("/api/local/settings", { method: "POST", body: { [key]: e.target.checked } }); toast("Saved."); } catch (err) { handleError(err); }
    });
  }
  $("#upd-check").addEventListener("click", async () => {
    const st = $("#upd-status");
    st.textContent = "Checking…";
    try {
      const u = await api("/api/update?force=1");
      st.innerHTML = !u.enabled ? "Update checks are switched off." : u.error ? `Could not reach GitHub (${esc(u.error)}).`
        : u.newer ? `Version ${esc(u.latest)} is available: <a href="${esc(u.url)}" target="_blank" rel="noopener">download</a>.` : `You have the latest version (${esc(u.current)}).`;
    } catch (err) { st.textContent = err.message || "Check failed."; }
  });
}
renderDesktop();
