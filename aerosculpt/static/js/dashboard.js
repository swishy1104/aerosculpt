import { api, appTop, confirmDialog, esc, fmtInt, h, handleError, icon, qs, requireAuth, timeAgo, toast, $, sig, fmtPct } from "/static/js/core.js";
import { ACCEPT, prepareUpload } from "/static/js/loaders.js";

const me = await requireAuth();
appTop(me, [{ label: "Projects" }]);
const TUNNELS = { rc: { label: "RC", cls: "rc" }, general: { label: "General", cls: "general" }, supersonic: { label: "Supersonic", cls: "hyper" } };
let tunnel = TUNNELS[qs("tunnel")] ? qs("tunnel") : (() => { try { return localStorage.getItem("as-tunnel"); } catch (e) { return null; } })();
if (!TUNNELS[tunnel]) tunnel = "general";
function pickTunnel(t) {
  tunnel = t;
  try { localStorage.setItem("as-tunnel", t); } catch (e) { /* private mode */ }
  document.querySelectorAll("#tunnel-pick .tp").forEach((b) => { b.classList.toggle("on", b.dataset.t === t); b.setAttribute("aria-checked", b.dataset.t === t); });
  $("#drop-title").textContent = `Drop a CAD, mesh or rocket design file (.ork, .rkt) to open it in the ${TUNNELS[t].label} tunnel`;
  if (samples) renderSamples();
}
document.querySelectorAll("#tunnel-pick .tp").forEach((b) => b.addEventListener("click", () => pickTunnel(b.dataset.t)));
document.querySelectorAll("[data-i]").forEach((el) => { el.innerHTML = icon(el.dataset.i); });

function tiles(projects) {
  const used = me.usage.credits, total = me.limits.monthly_credits;
  const frac = Math.min(1, used / Math.max(1, total));
  if (me.anonymous) {
    const hours = Math.max(1, Math.round((me.guest_expires_at - Date.now() / 1000) / 3600));
    $("#tiles").innerHTML = `
    <div class="card stat-tile"><div class="label">Workspace</div><div class="value">Yours</div>
      <div class="small muted">cleared in about ${hours} h</div></div>
    <div class="card stat-tile"><div class="label">Fidelity</div><div class="value">${me.limits.fidelities.map((f) => f[0].toUpperCase() + f.slice(1)).join(" · ")}</div>
      <div class="small muted">CFD runs on the host's GPU, one at a time</div></div>
    <div class="card stat-tile"><div class="label">CFD runs</div><div class="value">${fmtInt(me.usage.cfd_runs)}</div>
      <div class="small muted">${fmtInt(me.usage.jobs)} jobs</div></div>
    <div class="card stat-tile"><div class="label">Projects</div><div class="value">${projects.length}${me.limits.max_projects ? ` / ${me.limits.max_projects}` : ""}</div>
      <div class="small muted">per workspace</div></div>`;
    return;
  }
  $("#tiles").innerHTML = `
    <div class="card stat-tile"><div class="label">Plan</div><div class="value">${esc(me.plan_name)}</div>
      <div class="small muted">${me.plan_status === "trialing" ? "Trial active" : me.plan === "lite" ? '<a href="/app/billing">Upgrade for more</a>' : "Active"}</div></div>
    <div class="card stat-tile"><div class="label">Credits left this month</div><div class="value">${fmtInt(me.credits_left)}</div>
      <div class="meter ${frac > 0.9 ? "bad" : frac > 0.7 ? "warn" : ""}" style="margin-top:8px"><div style="width:${(1 - frac) * 100}%"></div></div></div>
    <div class="card stat-tile"><div class="label">CFD runs this month</div><div class="value">${fmtInt(me.usage.cfd_runs)}</div>
      <div class="small muted">${fmtInt(me.usage.jobs)} jobs</div></div>
    <div class="card stat-tile"><div class="label">Projects</div><div class="value">${projects.length}${me.limits.max_projects ? ` / ${me.limits.max_projects}` : ""}</div>
      <div class="small muted">${me.limits.max_projects ? `${esc(me.plan_name)} limit` : "Unlimited"}</div></div>`;
}

function jobLine(j) {
  if (!j) return `<span class="muted small">No runs yet</span>`;
  const s = j.summary;
  if (j.status === "done" && s && j.kind === "optimize") {
    if (s.drag_change === undefined) return `<span class="badge">older run</span>`;
    return `<span class="badge ${s.drag_change < 0 ? "good" : ""}"><span class="dot"></span>${fmtPct(s.drag_change, 1, true)} drag</span>`;
  }
  if (j.status === "done" && s) return `<span class="badge"><span class="dot" style="color:var(--series-1)"></span>Cd ${sig(s.cd, 3)}${s.ld_max && s.ld_max > 1 ? ` · L/D ${sig(s.ld_max, 3)}` : ""}</span>`;
  const cls = { running: "warn", queued: "", failed: "bad", canceled: "" }[j.status] || "";
  return `<span class="badge ${cls}"><span class="dot"></span>${esc(j.status)}</span>`;
}

async function loadProjects() {
  const projects = await api("/api/projects");
  tiles(projects);
  const box = $("#projects");
  box.innerHTML = "";
  if (!projects.length) {
    box.append(h("div", { class: "card muted", style: "grid-column:1/-1" }, "No projects yet. Upload a model above or open a sample below."));
    return;
  }
  for (const p of projects) {
    const card = h("div", { class: "card proj-card", tabindex: 0 });
    const thumb = h("div", { class: "thumb" });
    if (p.has_thumbnail) thumb.append(h("img", { src: `/api/projects/${p.id}/thumbnail?t=${Math.round(p.updated_at)}`, alt: "", loading: "lazy" }));
    else thumb.innerHTML = icon("box");
    const m = p.metrics || {};
    const body = h("div", { class: "body" });
    const tn = TUNNELS[(p.settings && p.settings.tunnel) || "general"] || TUNNELS.general;
    body.innerHTML = `${p.team ? `<div class="small muted" style="margin-bottom:2px">${icon("users", "i sm")} ${esc(p.team.name)}${p.mine ? "" : ` · ${esc(p.owner || "")} · ${esc(p.team.role || "")}`}</div>` : ""}
      <div class="row spread"><div class="name grow" title="${esc(p.name)}">${esc(p.name)}</div>
      <span class="tchip ${tn.cls}"><span class="tdot"></span>${tn.label}</span>
      ${p.mine === false ? "" : `<button class="icon-btn" data-del aria-label="Delete project">${icon("trash")}</button>`}</div>
      <div class="small muted" style="margin:2px 0 10px">${esc(p.filename)} · ${fmtInt(m.triangles)} triangles · ${timeAgo(p.updated_at)}</div>
      ${jobLine(p.last_job)}`;
    card.append(thumb, body);
    const open = () => { location.href = `/app/project/${p.id}`; };
    card.addEventListener("click", (e) => { if (!e.target.closest("[data-del]")) open(); });
    card.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    body.querySelector("[data-del]").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!(await confirmDialog("Delete project?", `“${esc(p.name)}” and all of its runs will be permanently deleted.`, "Delete", true))) return;
      try {
        await api(`/api/projects/${p.id}`, { method: "DELETE" });
        toast("Project deleted.");
        loadProjects();
      } catch (err) { handleError(err); }
    });
    box.append(card);
  }
}

const GROUPS = { rocket: "Rockets", "fixed-wing": "Fixed-wing UAVs", component: "Pods and components", rc: "RC aircraft", supersonic: "Supersonic and hypersonic" };
const ICON = { rocket: "rocket", "fixed-wing": "plane", component: "wind", rc: "fan", supersonic: "flame" };
let samples = null;
async function loadSamples() {
  samples = await api("/api/samples");
  renderSamples();
}
function renderSamples() {
  const box = $("#samples");
  box.innerHTML = "";
  const order = [tunnel, ...Object.keys(TUNNELS).filter((t) => t !== tunnel)];
  for (const t of order) {
    const group = samples.filter((s) => (s.tunnel || "general") === t);
    if (!group.length) continue;
    box.append(h("div", { class: "sample-head" }, h("span", { class: `tchip ${TUNNELS[t].cls}`, html: `<span class="tdot"></span>${TUNNELS[t].label} tunnel` })));
    const grid = h("div", { class: "grid-3" });
    for (const s of group) grid.append(sampleCard(s));
    box.append(grid);
  }
}
function sampleCard(s) {
  const c = h("div", { class: "card sample-card", tabindex: 0, role: "button" });
  const e = s.envelope, hy = s.hyper;
  const range = s.tunnel === "supersonic" && hy ? `Mach ${hy.mach_min}–${hy.mach_max} · ${hy.altitude / 1000} km`
    : `${e.speed_min}–${e.speed_max} m/s · ${e.aoa_min === e.aoa_max ? e.aoa_min + "°" : e.aoa_min + "° to " + e.aoa_max + "°"}${s.propellers ? ` · ${s.propellers} prop` : ""}`;
  c.innerHTML = `<div class="tag">${esc(GROUPS[s.category] || "")}</div>
    <div class="row" style="margin:6px 0 6px"><span style="color:var(${{ rc: "--rc", supersonic: "--heat" }[s.tunnel] || "--accent-ink"})">${icon(ICON[s.category] || "wind")}</span>
    <b style="color:var(--text);font-size:15px">${esc(s.title)}</b></div><div class="small muted">${esc(s.description)}</div>
    <div class="small" style="margin-top:8px;color:var(--text-2)">${range}</div>`;
  const go = async () => {
    try {
      c.style.opacity = 0.6;
      const p = await api("/api/projects", { method: "POST", body: { sample: s.key } });
      location.href = `/app/project/${p.id}`;
    } catch (err) { c.style.opacity = 1; handleError(err); }
  };
  c.addEventListener("click", go);
  c.addEventListener("keydown", (ev) => { if (ev.key === "Enter") go(); });
  return c;
}

// ---- templates -------------------------------------------------------------------------
const TEMPLATE_CARDS = [
  { id: "rocket", icon: "rocket", title: "Rocket", text: "Nose, body, boat-tail and fins with motor, masses and launch site: stability, flutter, apogee and the whole flight, live. Or drop an OpenRocket / RockSim file above." },
  { id: "uav", icon: "plane", title: "Fixed-wing UAV", text: "Pod-and-boom fuselage, tapered wing with a NACA 4- or 5-digit section, sweep, dihedral, twist and position, conventional tail." },
  { id: "wing", icon: "plane", title: "Wing", text: "A wing on its own: span, chords, sweep, twist and the NACA section, optimized for drag at the lift you need." },
];
function renderTemplates() {
  const box = $("#templates");
  box.innerHTML = "";
  for (const t of TEMPLATE_CARDS) {
    const c = h("div", { class: "card sample-card", tabindex: 0, role: "button" });
    c.innerHTML = `<div class="tag">Template</div><div class="row" style="margin:6px 0 6px"><span style="color:var(--accent-ink)">${icon(t.icon)}</span>
      <b style="color:var(--text);font-size:15px">${esc(t.title)}</b></div><div class="small muted">${esc(t.text)}</div>`;
    const go = async () => {
      try {
        c.style.opacity = 0.6;
        const p = await api("/api/projects", { method: "POST", body: { template: t.id, tunnel } });
        location.href = `/app/project/${p.id}`;
      } catch (err) { c.style.opacity = 1; handleError(err); }
    };
    c.addEventListener("click", go);
    c.addEventListener("keydown", (ev) => { if (ev.key === "Enter") go(); });
    box.append(c);
  }
}
renderTemplates();

// ---- upload --------------------------------------------------------------------------
const input = $("#file");
input.accept = ACCEPT;
const drop = $("#drop");
const status = $("#drop-status");
async function upload(file) {
  if (!file) return;
  const limit = me.limits.max_upload_mb * 1024 * 1024;
  try {
    status.textContent = `Reading ${file.name}…`;
    const prep = await prepareUpload(file, (msg) => { status.textContent = msg; });
    if (prep.body.byteLength > limit) throw new Error(`File is larger than your plan's ${me.limits.max_upload_mb} MB limit.`);
    status.textContent = "Uploading and checking the mesh…";
    const q = new URLSearchParams({ filename: prep.filename, name: file.name.replace(/\.[^.]+$/, ""), source: prep.source, tunnel });
    const p = await api(`/api/projects?${q}`, { method: "POST", raw: prep.body });
    location.href = `/app/project/${p.id}`;
  } catch (e) {
    status.textContent = "";
    if (e.status) handleError(e); else toast(e.message || "Upload failed.", "bad");
  }
}
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
$("#upload-btn").addEventListener("click", () => input.click());
input.addEventListener("change", () => upload(input.files[0]));
for (const ev of ["dragenter", "dragover"]) document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) document.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "drop" || !e.relatedTarget) drop.classList.remove("over"); });
document.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

if (qs("welcome")) {
  const w = $("#welcome");
  w.classList.remove("hidden");
  w.innerHTML = `<div class="notice info">${icon("sparkles")}<div><b style="color:var(--text)">Welcome to AeroSculpt.</b>
    Start with a sample body below to see a full wind-tunnel run and optimization, or drop your own model. Lite includes
    ${me.limits.monthly_credits} credits a month; <a href="/app/billing">Pro starts with a free trial</a>.</div></div>`;
}

pickTunnel(tunnel);
try {
  await Promise.all([loadProjects(), loadSamples()]);
} catch (e) { handleError(e); }
