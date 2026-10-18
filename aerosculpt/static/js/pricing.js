import { api, esc, fmtInt, icon, siteFooter, siteNav, $ } from "/static/js/core.js";

const [me, catalog] = await Promise.all([siteNav(), api("/api/plans")]);
siteFooter();

$("#plans").innerHTML = catalog.plans.map((p) => {
  const href = me ? (p.key === "lite" ? "/app" : `/app/billing?upgrade=${p.key}`) : `/signup${p.key === "lite" ? "" : `?plan=${p.key}`}`;
  const label = p.key === "lite" ? "Start free" : p.trial ? `Start ${catalog.trial_days}-day free trial` : `Choose ${p.name}`;
  return `<div class="card price-card ${p.key === "pro" ? "featured" : ""}">
    <div class="row spread"><h3 style="margin:0">${esc(p.name)}</h3>${p.key === "pro" ? '<span class="badge accent">Everything included</span>' : ""}</div>
    <p class="small" style="margin:4px 0 12px">${esc(p.tagline)}</p>
    <div class="price">$${p.price_month}<small> / month</small></div>
    <ul>${p.features.map((f) => `<li>${icon("check")}<span>${esc(f)}</span></li>`).join("")}</ul>
    <a class="btn block ${p.key === "pro" ? "primary" : ""}" href="${href}">${label}</a></div>`;
}).join("");

const yes = `<span style="color:var(--good-text)">${icon("check")}</span>`;
const no = `<span class="muted">—</span>`;
const rows = [
  ["Monthly compute", (p) => (p.monthly_credits >= 1e8 ? "Unlimited" : fmtInt(p.monthly_credits) + " credits")],
  ["Tunnel fidelity", (p) => p.fidelities.map((f) => f[0].toUpperCase() + f.slice(1)).join(", ")],
  ["Iterations per optimization", (p) => p.max_iterations],
  ["Design aspects at once", (p) => p.max_aspects],
  ["Objectives", (p) => (p.objectives.length >= 6 ? "All: drag, L/D, lift-constrained, apogee" : "Drag, drag at fixed lift, apogee")],
  ["Projects", (p) => (p.max_projects ? p.max_projects : "Unlimited")],
  ["Upload size", (p) => p.max_upload_mb + " MB"],
  ["Triangles per model", (p) => fmtInt(p.max_triangles)],
  ["Export optimized STL", (p) => (p.export ? yes : no)],
  ["High-fidelity verification", (p) => (p.verify ? yes : no)],
  ["Printable reports", (p) => (p.reports ? yes : no)],
  ["Teams and REST API", (p) => (p.api ? yes : no)],
  ["Desktop licence key", (p) => (p.export ? yes : no)],
  ["Concurrent jobs", (p) => p.concurrent_jobs],
  ["Queue priority", (p) => ["Standard", "High", "Highest"][Math.min(2, p.priority / 10)]],
];
$("#compare").innerHTML = `<thead><tr><th></th>${catalog.plans.map((p) => `<th class="r">${esc(p.name)}</th>`).join("")}</tr></thead>
  <tbody>${rows.map(([label, f]) => `<tr><td>${label}</td>${catalog.plans.map((p) => `<td class="r">${f(p)}</td>`).join("")}</tr>`).join("")}</tbody>`;
