import { api, appTop, busy, confirmDialog, esc, fmtDuration, fmtForce, fmtInt, fmtPct, fmtPower, getMe, h, handleError,
  icon, requireAuth, sig, timeAgo, toast, upgradePrompt, $, $$ } from "/static/js/core.js";
import { ModelFrame, Viewer, applyMat3, parseModel, parseSTL, rotate, transposeMat3, windRotation, zoneFromPose, zoneMatrix } from "/static/js/viewer.js";
import { MAPS, b64bytes, cssGradient, dequantize } from "/static/js/colormap.js";
import { lineChart } from "/static/js/charts.js";
import { Propellers, rcCharts, rcResultsHTML } from "/static/js/tunnel-rc.js";
import { HyperPanel, degC, hyperCharts, hyperFaceFields, hyperHistoryHTML, hyperResultsHTML } from "/static/js/tunnel-hyper.js";
import { DesignParams, fixedLiftHTML, paramsTableHTML, strategyHTML } from "/static/js/design-params.js";
import { FeatureEditor, featureChangesHTML } from "/static/js/features.js";

let me = await requireAuth();
const pid = location.pathname.split("/").filter(Boolean).pop();
let project, catalog;
try {
  [project, catalog] = await Promise.all([api(`/api/projects/${pid}`), api("/api/plans")]);
} catch (e) {
  // a deleted project or someone else's link: say so plainly instead of leaving a half-built workspace
  const gone = e && e.status === 404;
  document.body.innerHTML = `<main class="center-page"><div class="card" style="max-width:440px;text-align:center;padding:28px">
    <h2 style="margin:0 0 8px">${gone ? "Project not found" : "Could not open this project"}</h2>
    <p class="muted" style="margin:0 0 18px">${gone ? "It may have been deleted, or it belongs to another account." : esc(e && e.message ? e.message : "Please try again.")}</p>
    <a class="btn primary" href="/app">Back to projects</a></div></main>`;
  throw e;
}
document.title = `${project.name} — AeroSculpt`;
$$("[data-i]").forEach((el) => { el.innerHTML = icon(el.dataset.i); });

const titleInput = h("input", { class: "title-edit", type: "text", value: project.name, "aria-label": "Project name", maxlength: 120 });
titleInput.addEventListener("change", async () => {
  try {
    project = await api(`/api/projects/${pid}`, { method: "PATCH", body: { name: titleInput.value } });
    document.title = `${project.name} — AeroSculpt`;
  } catch (e) { handleError(e); }
});
const top = appTop(me, [{ label: "Projects", href: "/app" }, titleInput]);

// ---- state ------------------------------------------------------------------------------------------
const S = project.settings;
S.envelope = Object.assign({ speed_min: 15, speed_max: 40, speed_count: 4, aoa_min: 0, aoa_max: 8, aoa_count: 3, altitude: 0, yaw: 0, ref_area: null }, S.envelope || {});
S.zones = S.zones || [];
const opt = Object.assign({ objective: "drag", aspects: ["nose", "tail"], max_change: 0.25, iterations: Math.min(12, me.limits.max_iterations),
  min_volume: 0.95, max_volume: 1.5, verify: null, top_k: Math.min(4, me.limits.max_top_k || 4), keep_stability: true, mass_kg: null,
  max_length: null, max_width: null, max_height: null, min_thickness: null, hold_wing_area: false }, S.optimize || {});
opt.top_k = Math.min(opt.top_k, me.limits.max_top_k || 8);
opt.verify = null;              // the optimizer now confirms its best candidates itself
const winged = () => !!S.template && (S.template.id === "uav" || S.template.id === "wing");
const unvalidated = catalog.objectives_unvalidated || [];
if (!(opt.objective in catalog.objectives) || unvalidated.includes(opt.objective)) opt.objective = "drag";
let fidelity = me.limits.fidelities.includes(S.fidelity) ? S.fidelity : me.limits.fidelities[0];
const AXES = ["+x", "-x", "+y", "-y", "+z", "-z"];
// measured on the job worker at start-up (GPU or CPU); fallback: a laptop CPU
const COMPUTE = catalog.compute || { backend: "cpu", seconds_per_run: { coarse: 8, draft: 20, standard: 95, high: 300 } };
const PER_RUN_S = COMPUTE.seconds_per_run;
const ZONE_TYPES = {
  protect: { label: "Protect", color: "--protect", help: "Geometry inside never moves", name: "Protected region" },
  keep: { label: "Keep inside", color: "--keep", help: "Volume stays enclosed (payload, motor)", name: "Payload bay" },
  design: { label: "Design only", color: "--design", help: "Only these areas may change", name: "Design region" },
  material: { label: "Material", color: "--material", help: "Its own skin material (nose cap, leading edge)", name: "Material region", only: "supersonic" },
};
const TUNNELS = {
  general: { label: "General", note: "3D CFD (lattice-Boltzmann LES) with compressibility corrections up to Mach 1.2: drag across your airspeed and angle range." },
  rc: { label: "RC", note: "The General tunnel plus spinning propellers: the airframe is tested in its own slipstream, with thrust, power and top speed." },
  supersonic: { label: "Supersonic", note: "Mach 1.2 to 12: shock, expansion and base drag, aerodynamic heating and wall temperature against your materials." },
};
let tunnel = TUNNELS[S.tunnel] ? S.tunnel : "general";
let props = null, hyper = null;          // tunnel panels, created once the helpers below exist
S.propellers = S.propellers || [];
let raw = null, frame = null;
let feats = null, designBody = null;      // shape features and the model with them applied (body frame)
let watchJob = null, shownJob = null, result = null;
const fileCache = {};
let display = "best";
let cond = { a: 0, v: 0 };                // indices into result.envelope
let vis = { mode: "shaded", stream: false, slice: null, sliceField: "speed", zones: true, shock: true };
let showingResults = false;
let selectedZone = null, picking = null;   // picking: { id, kind: "zone" | "prop" }
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---- viewer -----------------------------------------------------------------------------------------
const viewer = new Viewer($("#vp"));
// overlays sit just under the viewport toolbar, which wraps onto more rows on narrow screens
new ResizeObserver(() => $("#vp").style.setProperty("--tb-h", $(".vp-toolbar").offsetHeight + "px")).observe($(".vp-toolbar"));
let dparams = null;
addEventListener("themechange", () => { viewer.applyTheme(); renderZones3D(); colorbar(); renderResults(); renderHistoryTab(); });
const ranges = { cp: [-1, 1], speed: [0, 1.4] };
function percentile(arr, q) {
  const v = Array.from(arr).filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? v[Math.min(v.length - 1, Math.floor(q * v.length))] : 0;
}

async function loadModel() {
  const res = await fetch(`/api/projects/${pid}/model`, { credentials: "same-origin" });
  if (!res.ok) throw new Error("Could not load the model file.");
  raw = parseModel(await res.arrayBuffer(), project.filename);
  rebuildFrame();
  showPreview(true);
  $("#vp-empty").classList.add("hidden");
  if (!project.has_thumbnail) {
    setTimeout(async () => {
      try { await api(`/api/projects/${pid}/thumbnail`, { method: "POST", body: { png: viewer.snapshot(480, 300) } }); } catch (e) { /* cosmetic */ }
    }, 700);
  }
}

function rebuildFrame() { frame = new ModelFrame(raw, S); }
function previewAngle() {
  const e = tunnel === "supersonic" ? S.hyper.envelope : S.envelope;
  return e.aoa_min <= 0 && e.aoa_max >= 0 ? 0 : e.aoa_min;
}
function displayRotation() {
  if (showingResults && result) {
    const f = currentFieldEntry();
    return windRotation(f.mirror ? -f.aoa : f.aoa, result.envelope.yaw);
  }
  return windRotation(previewAngle(), S.envelope.yaw);
}

function showPreview(fit = false) {
  if (!frame) return;
  viewer.setModel(rotate(designBody || frame.body, windRotation(previewAngle(), S.envelope.yaw)), { fit });
  viewer.setMirror(false);
  viewer.setGhost(null);
  viewer.setSlice(null);
  viewer.setStreamlines(null);
  viewer.setTunnel(null);
  viewer.setShock(null);
  showingResults = false;
  if (props) props.render3D(windRotation(previewAngle(), S.envelope.yaw), tunnel === "rc");
  vis.mode = vis.mode === "wire" ? "wire" : "shaded";
  viewer.setMode(vis.mode);
  $("#vp-cond").classList.add("hidden");
  $("#vp-info").textContent = `Preview at ${previewAngle()}° angle of attack`;
  updateToolbar();
  renderZones3D();
  modelInfo();
  flowInfo();
}

// ---- 1. model -----------------------------------------------------------------------------------------
function fillAxis(sel, value) {
  sel.innerHTML = AXES.map((a) => `<option value="${a}">${a.toUpperCase()}</option>`).join("");
  sel.value = value;
}
fillAxis($("#forward"), S.forward);
fillAxis($("#up"), S.up);
$("#units").value = S.units;
// the model's extent along its nose axis, in file units (for "set the real length")
function rawLength() {
  if (!raw) return 0;
  const k = "xyz".indexOf(S.forward[1]);
  let lo = Infinity, hi = -Infinity;
  for (let i = k; i < raw.length; i += 3) { if (raw[i] < lo) lo = raw[i]; if (raw[i] > hi) hi = raw[i]; }
  return hi - lo;
}
function syncRealLen() {
  $("#real-len-row").classList.toggle("hidden", S.units !== "custom");
  if (S.units === "custom" && S.unit_scale && raw && document.activeElement !== $("#real_len")) $("#real_len").value = sig(rawLength() * S.unit_scale, 4);
}
const lenStr = (v) => (v >= 1 ? sig(v, 4) + " m" : sig(v * 1000, 4) + " mm");

function modelInfo() {
  const m = project.metrics || {};
  const L = frame ? frame.size : [0, 0, 0];
  $("#model-axes").classList.toggle("hidden", !!S.template);
  if (S.template) {
    const t = catalog.templates[S.template.id];
    $("#model-kv").innerHTML = `
      <span class="k">Template</span><span class="v">${esc(t.title)}</span>
      <span class="k">Triangles</span><span class="v">${fmtInt(m.triangles)} · ${fmtInt(m.shells)} parts</span>
      <span class="k">Mesh</span><span class="v">${m.watertight ? '<span class="badge good"><span class="dot"></span>watertight</span>' : '<span class="badge warn"><span class="dot"></span>check</span>'}</span>
      <span class="k" title="Length × span × height">Size</span><span class="v">${lenStr(L[0])} × ${lenStr(L[1])} × ${lenStr(L[2])}</span>`;
    $("#s-model").textContent = t.title;
    return;
  }
  $("#model-kv").innerHTML = `
    <span class="k">File</span><span class="v" title="${esc(project.filename)}">${esc(project.filename.length > 26 ? project.filename.slice(0, 24) + "…" : project.filename)}</span>
    <span class="k" title="Triangles · separate shells">Triangles</span><span class="v">${fmtInt(m.triangles)} · ${fmtInt(m.shells)}</span>
    <span class="k">Mesh</span><span class="v">${m.watertight ? '<span class="badge good"><span class="dot"></span>watertight</span>'
      : `<span class="badge warn"><span class="dot"></span>open (${fmtInt(m.open_edges)} edges)</span>`}</span>
    <span class="k" title="Length × span × height">Size</span><span class="v">${lenStr(L[0])} × ${lenStr(L[1])} × ${lenStr(L[2])}</span>`;
  if (Math.max(...L) > 50 || Math.max(...L) < 0.005) {
    $("#model-kv").insertAdjacentHTML("beforeend", `<span class="k" style="grid-column:1/-1;color:var(--warning-text)">That size looks unusual — check the file units, or choose “Set the real length”.</span>`);
  }
  const box = $("#solid-box");
  if (!m.watertight) {
    box.innerHTML = `<div class="notice warn" style="margin-top:10px">${icon("alert")}<div><b style="color:var(--text)">This is a visual mesh, not a solid.</b>
      ${fmtInt(m.shells)} separate pieces with ${fmtInt(m.open_edges)} open edges: air leaks through the gaps in the tunnel, so drag jumps between
      runs and the optimizer cannot trust any change. Make a watertight copy first — seams and holes are closed and the inside filled.
      <div class="row" style="margin-top:8px;gap:6px"><button class="btn sm primary" type="button" id="make-solid">${icon("box")}Make a solid copy</button>
      <button class="btn sm ghost" type="button" id="make-solid-fine" title="Finer, slower, more triangles">Fine detail</button></div></div></div>`;
    const go = async (detail, btn) => {
      busy(btn, true);
      try {
        const np = await api(`/api/projects/${pid}/solidify`, { method: "POST", body: { detail } });
        toast(`Solid copy made: ${fmtInt(np.metrics.triangles)} triangles, watertight. Opening it…`, "good");
        setTimeout(() => { location.href = `/app/project/${np.id}`; }, 700);
      } catch (e) { handleError(e); busy(btn, false); }
    };
    $("#make-solid").addEventListener("click", (e) => go("standard", e.currentTarget));
    $("#make-solid-fine").addEventListener("click", (e) => go("fine", e.currentTarget));
  } else box.innerHTML = m.solidified ? `<p class="small muted" style="margin:8px 0 0">${icon("check")} Made solid from ${fmtInt(m.solidified.pieces_before)} pieces
    (${sig(m.solidified.cell * 1000, 3)} file-unit grid).</p>` : "";
  $("#s-model").textContent = `${S.units === "custom" ? "real size" : S.units} · nose ${S.forward.toUpperCase()}`;
  syncRealLen();
}

let saveTimer;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      project = await api(`/api/projects/${pid}`, { method: "PATCH", body: { settings: {
        units: S.units, ...(S.units === "custom" ? { unit_scale: S.unit_scale } : {}), forward: S.forward, up: S.up, envelope: S.envelope, zones: S.zones, features: S.features || [],
        tunnel, propellers: S.propellers, hyper: S.hyper, optimize: opt, fidelity,
        ...(S.template ? { template: { values: S.template.values, params: S.template.params } } : {}) } } });
      if (S.template) {
        const ne = project.settings.envelope;
        S.envelope.ref_area = ne.ref_area;
        S.envelope.ref_length = ne.ref_length;
        if (document.activeElement !== $("#ref_area")) $("#ref_area").value = ne.ref_area || "";
        if (project.settings.hyper) S.hyper.envelope.ref_area = project.settings.hyper.envelope.ref_area;
        modelInfo();
      }
    } catch (e) { handleError(e); }
  }, 500);
}
function onGeomSetting() {
  if (S.forward[1] === S.up[1]) return toast("Nose and up must be different axes.", "bad");
  rebuildFrame();
  showPreview(false);
  renderZones();
  saveSettings();
  estimates();
}
$("#units").addEventListener("change", (e) => {
  if (e.target.value === "custom") {
    // start from the current size so nothing jumps until a real length is typed
    S.unit_scale = S.unit_scale || (frame ? frame.s : 1);
  }
  S.units = e.target.value;
  syncRealLen();
  onGeomSetting();
  viewer.view("iso");
  if (S.units === "custom") $("#real_len").focus();
});
$("#real_len").addEventListener("change", (e) => {
  const L = +e.target.value, r = rawLength();
  if (!(L > 0) || !(r > 0)) return toast("Type the real length in metres.", "bad");
  S.unit_scale = L / r;
  onGeomSetting();
  viewer.view("iso");
});
$("#forward").addEventListener("change", (e) => { S.forward = e.target.value; onGeomSetting(); });
$("#up").addEventListener("change", (e) => { S.up = e.target.value; onGeomSetting(); });

// ---- 2. protected regions ---------------------------------------------------------------------------
function fileSizeOfBody() {
  // body extents expressed along the file's own axes, in file units
  const out = [0, 0, 0];
  for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) out[j] += Math.abs(frame.R0[k][j]) * frame.size[k];
  return out.map((v) => v / frame.s);
}
const round = (v) => Math.round(v * 1000) / 1000;

function addZone(type) {
  if (S.zones.length >= me.limits.max_zones) {
    return upgradePrompt(`${me.plan_name} allows ${me.limits.max_zones} protected regions. Pro and Team allow many more.`);
  }
  const center = frame.bodyToFile([0, 0, 0]).map(round);
  const ext = fileSizeOfBody();
  const z = { id: Math.random().toString(36).slice(2, 10), type, shape: type === "keep" ? "cylinder" : "box",
    center, size: ext.map((v) => round(v * (type === "keep" ? 0.35 : 0.18))),
    name: `${ZONE_TYPES[type].name} ${S.zones.filter((q) => q.type === type).length + 1}` };
  if (type === "material") z.material = "rcc";
  S.zones.push(z);
  selectedZone = z.id;
  renderZones();
  renderZones3D();
  saveSettings();
  startPick(z.id);
}

function renderZoneButtons() {
  const box = $("#zone-add");
  box.innerHTML = "";
  for (const [t, d] of Object.entries(ZONE_TYPES)) {
    if (d.only && d.only !== tunnel) continue;
    const b = h("button", { class: "btn sm", type: "button", style: { "--zc": `var(${d.color})` }, title: d.help,
      html: `<span class="dot"></span><b>${d.label}</b><span class="muted" style="font-weight:500">${esc(d.help)}</span>` });
    b.addEventListener("click", () => addZone(t));
    box.append(b);
  }
}

function renderZones() {
  const box = $("#zones");
  box.innerHTML = "";
  const unit = S.units;
  for (const z of S.zones) {
    const d = ZONE_TYPES[z.type];
    if (d.only && d.only !== tunnel) continue;
    const card = h("div", { class: "zone" + (z.id === selectedZone ? " sel" : ""), style: { "--zc": `var(${d.color})` } });
    card.innerHTML = `<div class="zh"><span class="zone-type">${esc(d.label)}</span>
        <input type="text" value="${esc(z.name)}" maxlength="60" aria-label="Region name">
        <button class="icon-btn" data-pick title="Place on the model">${icon("crosshair", "i sm")}</button>
        <button class="icon-btn" data-del title="Remove">${icon("trash", "i sm")}</button></div>
      <div class="row" style="margin-top:6px;gap:6px"><div class="seg"><button data-shape="box" class="${z.shape === "box" ? "on" : ""}">Box</button>
        <button data-shape="cylinder" class="${z.shape === "cylinder" ? "on" : ""}" title="Cylinder: along the flow axis until you turn it">Cylinder</button></div>
        <span class="grow"></span><button class="btn sm ghost" data-edit type="button" title="Move, turn and resize it in the 3D view">${icon("move", "i sm")}Edit in 3D</button></div>
      ${z.type === "material" ? `<label class="field" style="margin:8px 0 0"><span>Material</span><select data-mat>${Object.entries(catalog.materials).map(([k, m]) =>
        `<option value="${k}" ${z.material === k ? "selected" : ""}>${esc(m.name)} · ${fmtInt(m.t_max_c)} °C</option>`).join("")}</select></label>` : ""}
      <div class="zgrid"><span></span><span class="ax">X</span><span class="ax">Y</span><span class="ax">Z</span>
        <span>Centre</span>${z.center.map((v, i) => `<input type="number" step="any" data-c="${i}" value="${v}" aria-label="Centre ${"XYZ"[i]} (${unit})">`).join("")}
        <span>Size</span>${z.size.map((v, i) => `<input type="number" step="any" min="0" data-s="${i}" value="${v}" aria-label="Size ${"XYZ"[i]} (${unit})">`).join("")}
        <span class="ax-l" title="Roll about the flow axis, pitch, yaw">Turn °</span>${(z.rotation || [0, 0, 0]).map((v, i) =>
          `<input type="number" step="5" data-r="${i}" value="${v}" title="${["Roll (about the flow axis)", "Pitch", "Yaw"][i]}" aria-label="${["Roll", "Pitch", "Yaw"][i]} in degrees">`).join("")}</div>
      <div class="zfoot small muted">${unit} · turns are roll, pitch, yaw about the airframe axes</div>`;
    card.addEventListener("click", (e) => {
      if (selectedZone !== z.id && !e.target.closest("input,button")) selectZone(z.id);
    });
    card.querySelector("[data-edit]").addEventListener("click", () => selectZone(z.id, zoneTool.mode || "translate"));
    card.querySelector(".zh input").addEventListener("change", (e) => { z.name = e.target.value; saveSettings(); summaries(); });
    card.querySelector("[data-del]").addEventListener("click", () => {
      S.zones = S.zones.filter((q) => q !== z);
      renderZones(); renderZones3D(); saveSettings();
    });
    card.querySelector("[data-pick]").addEventListener("click", () => startPick(z.id));
    const ms = card.querySelector("[data-mat]");
    if (ms) ms.addEventListener("change", () => { z.material = ms.value; saveSettings(); });
    card.querySelectorAll("[data-shape]").forEach((b) => b.addEventListener("click", () => {
      z.shape = b.dataset.shape; renderZones(); renderZones3D(); saveSettings();
    }));
    card.querySelectorAll("input[data-c],input[data-s],input[data-r]").forEach((inp) => inp.addEventListener("change", () => {
      const v = parseFloat(inp.value);
      if (!Number.isFinite(v) || (inp.dataset.s !== undefined && v <= 0)) { inp.classList.add("invalid"); return; }
      inp.classList.remove("invalid");
      if (inp.dataset.c !== undefined) z.center[+inp.dataset.c] = v;
      else if (inp.dataset.s !== undefined) z.size[+inp.dataset.s] = v;
      else {
        z.rotation = (z.rotation || [0, 0, 0]).slice();
        z.rotation[+inp.dataset.r] = ((v + 180) % 360 + 360) % 360 - 180;
        inp.value = z.rotation[+inp.dataset.r];
        if (z.rotation.every((a) => a === 0)) delete z.rotation;
      }
      selectedZone = z.id;
      renderZones3D(); saveSettings();
    }));
    box.append(card);
  }
  $("#zone-limit").textContent = S.zones.length ? "" : "No regions yet: the optimizer may change the whole body.";
  summaries();
}

function renderZones3D() {
  if (viewer.dragging) return;          // never rebuild the boxes under a gizmo drag (redrawn on release)
  if (!frame || !vis.zones || (showingResults && viewer.root.scale.z < 0)) { viewer.setZones([]); return; }
  const R = displayRotation();
  viewer.setZones(S.zones.filter((z) => !ZONE_TYPES[z.type].only || ZONE_TYPES[z.type].only === tunnel).map((z) => {
    const c = frame.fileToBody(z.center);
    let s = frame.sizeToBody(z.size);
    if (z.shape === "cylinder") { const r = Math.max(s[1], s[2]); s = [s[0], r, r]; }
    return { id: z.id, shape: z.shape, color: cssVar(ZONE_TYPES[z.type].color), selected: z.id === selectedZone, matrix: zoneMatrix(c, s, R, z.rotation) };
  }).concat(featureMarkers(R)));
  syncZoneTool();
}

// shape features in the 3D view: the box a region feature acts on, a small cube at a point feature's anchor,
// a thin plate showing a slant's cut plane (its outward side is the part removed)
function featureMarkers(R) {
  if (!S.features || !catalog.features) return [];
  const L = Math.max(...frame.size) || 1;
  return S.features.filter((f) => f.enabled !== false && (f.size || f.id === selectedZone)).map((f) => {
    const c = frame.fileToBody(f.center);
    let s, rot = f.rotation;
    if (f.size) s = frame.sizeToBody(f.size);
    else if (f.type === "slant") {
      s = [L * 0.004, L * 0.3, L * 0.3];
      const n = [Math.cos((f.values.elevation * Math.PI) / 180) * Math.cos((f.values.azimuth * Math.PI) / 180),
        Math.cos((f.values.elevation * Math.PI) / 180) * Math.sin((f.values.azimuth * Math.PI) / 180), Math.sin((f.values.elevation * Math.PI) / 180)];
      for (let k = 0; k < 3; k++) c[k] -= f.values.depth * n[k];
      rot = [0, -f.values.elevation, f.values.azimuth];
    } else s = [L * 0.05, L * 0.05, L * 0.05];
    return { id: f.id, shape: "box", color: cssVar("--accent"), selected: f.id === selectedZone, matrix: zoneMatrix(c, s, R, rot) };
  });
}

// ---- protected regions in 3D: select, move, turn, resize ---------------------------------------------------
const zoneTool = { mode: null, snap: false };     // mode: null (closed) | "translate" | "rotate" | "scale"
const TOOL_LABEL = { translate: "Move", rotate: "Turn", scale: "Resize" };
function zoneEditable() {
  return !!frame && vis.zones && !(showingResults && viewer.root.scale.z < 0);
}
function selectZone(id, mode = zoneTool.mode) {
  selectedZone = id;
  zoneTool.mode = id ? mode || "translate" : null;
  renderZones();
  renderZones3D();
  if (feats && isFeature(findEditable(id))) { feats.open = id; feats.renderList(); }
  const card = $("#zones .zone.sel") || $(".ft-card.sel");
  if (card) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
const findEditable = (id) => S.zones.find((q) => q.id === id) || (S.features || []).find((q) => q.id === id);
const isFeature = (z) => !!(z && catalog.features && catalog.features[z.type]);
function syncZoneTool() {
  const z = findEditable(selectedZone);
  const bar = $("#zone-tools");
  const zt = z && ZONE_TYPES[z.type];
  const on = !!(z && zoneTool.mode && !picking && zoneEditable() && (!zt || !zt.only || zt.only === tunnel) && z.enabled !== false);
  bar.classList.toggle("hidden", !on);
  if (!on) { viewer.detachZoneTool(); return; }
  // point features have no box to resize; a slant's angles are its own dimensions
  const canScale = !isFeature(z) || !!z.size, canTurn = z.type !== "slant";
  bar.querySelector('[data-tool="scale"]').classList.toggle("hidden", !canScale);
  bar.querySelector('[data-tool="rotate"]').classList.toggle("hidden", !canTurn);
  $("#zt-reset").classList.toggle("hidden", !canTurn);
  if ((zoneTool.mode === "scale" && !canScale) || (zoneTool.mode === "rotate" && !canTurn)) zoneTool.mode = "translate";
  bar.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === zoneTool.mode));
  $("#zt-snap").classList.toggle("on", zoneTool.snap);
  $("#zt-name").textContent = z.name;
  const cur = viewer._zoneToolArgs;
  if (viewer.zoneTool && cur && cur[0] === z.id && cur[1] === zoneTool.mode && cur[2].snap === zoneTool.snap) return;
  viewer.attachZoneTool(z.id, zoneTool.mode, { snap: zoneTool.snap, onChange: (pose) => zoneDragged(z, pose), onEnd: () => zoneDragEnd(z) });
}
function zoneDragged(z, pose) {
  const R = displayRotation();
  const b = zoneFromPose(pose, R);
  if (isFeature(z)) {
    if (z.type === "slant") {          // the plate sits on the cut plane, `depth` behind the anchor
      const e = (z.values.elevation * Math.PI) / 180, a = (z.values.azimuth * Math.PI) / 180;
      const n = [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
      for (let k = 0; k < 3; k++) b.center[k] += z.values.depth * n[k];
    }
    z.center = frame.bodyToFile(b.center).map(round);
    if (z.size) {
      const fs = [0, 0, 0];
      for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) fs[j] += Math.abs(frame.R0[k][j]) * b.size[k];
      z.size = fs.map((v) => Math.max(round(v / frame.s), 1e-3));
    }
    if (z.type !== "slant") { if (b.rotation.some((a) => Math.abs(a) > 0.05)) z.rotation = b.rotation; else delete z.rotation; }
    feats.rebuild(false);
    return;
  }
  if (z.shape === "cylinder") {     // one radius: follow whichever cross-section handle moved
    const prev = frame.sizeToBody(z.size);
    const r = Math.abs(b.size[1] - Math.max(prev[1], prev[2])) >= Math.abs(b.size[2] - Math.max(prev[1], prev[2])) ? b.size[1] : b.size[2];
    b.size[1] = b.size[2] = r;
    viewer.zoneTool.target.scale.set(b.size[0], r, r);
  }
  z.center = frame.bodyToFile(b.center).map(round);
  const fs = [0, 0, 0];      // body-axes size -> file axes and units
  for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) fs[j] += Math.abs(frame.R0[k][j]) * b.size[k];
  z.size = fs.map((v) => Math.max(round(v / frame.s), 1e-3));
  if (b.rotation.some((a) => Math.abs(a) > 0.05)) z.rotation = b.rotation; else delete z.rotation;
  const card = $("#zones .zone.sel");      // live numbers without rebuilding the card
  if (card) {
    card.querySelectorAll("input[data-c]").forEach((inp) => { inp.value = z.center[+inp.dataset.c]; });
    card.querySelectorAll("input[data-s]").forEach((inp) => { inp.value = z.size[+inp.dataset.s]; });
    card.querySelectorAll("input[data-r]").forEach((inp) => { inp.value = (z.rotation || [0, 0, 0])[+inp.dataset.r]; });
  }
}
function zoneDragEnd(z) {
  if (isFeature(z)) feats.placed(); else saveSettings();
  renderZones3D();
}
$("#zone-tools").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const z = findEditable(selectedZone);
  if (b.dataset.tool) zoneTool.mode = b.dataset.tool;
  else if (b.id === "zt-snap") { setSnap(!zoneTool.snap); return; }
  else if (b.id === "zt-reset" && z) {
    delete z.rotation;
    if (isFeature(z)) feats.placed(); else { renderZones(); saveSettings(); }
  } else if (b.id === "zt-done") zoneTool.mode = null;
  renderZones3D();
});
addEventListener("keydown", (e) => {
  if (!zoneTool.mode || (e.target.closest && e.target.closest("input,textarea,select")) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (viewer.dragging && k !== "shift") return;      // never rebuild the regions under a drag
  const mode = { w: "translate", g: "translate", e: "rotate", r: "scale", s: "scale" }[k];
  if (mode) { zoneTool.mode = mode; renderZones3D(); e.preventDefault(); }
  else if (k === "escape" && !picking) { zoneTool.mode = null; renderZones3D(); }
  else if (k === "shift" && !zoneTool.snap) setSnap(true);
});
addEventListener("keyup", (e) => { if (e.key === "Shift" && zoneTool.mode && zoneTool.snap) setSnap(false); });
function setSnap(on) {
  zoneTool.snap = on;
  viewer.setZoneSnap(on);
  $("#zt-snap").classList.toggle("on", on);
}

function startPick(id, kind = "zone") {
  picking = { id, kind };
  $("#vp").classList.add("picking");
  $("#pick-hint").textContent = kind === "prop" ? "Click the model where the propeller hub goes · Esc to cancel"
    : "Click the model where this region should go · Esc to cancel";
  $("#pick-hint").classList.remove("hidden");
  $(`[data-sec="${kind === "prop" ? "props" : "protect"}"]`).classList.remove("closed");
  syncZoneTool();
}
function pickForFeature(type, hint) {
  return new Promise((resolve) => {
    if (picking && picking.resolve) picking.resolve(null);
    picking = { kind: "feature", resolve };
    $("#vp").classList.add("picking");
    $("#pick-hint").textContent = `${hint} · Esc to cancel`;
    $("#pick-hint").classList.remove("hidden");
    syncZoneTool();
  });
}
function stopPick() {
  if (picking && picking.resolve) picking.resolve(null);
  picking = null;
  $("#vp").classList.remove("picking");
  $("#pick-hint").classList.add("hidden");
  syncZoneTool();
}
addEventListener("keydown", (e) => { if (e.key === "Escape" && picking) stopPick(); });
let downAt = null;
viewer.renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
viewer.renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5 || viewer.dragging || viewer.overGizmo) return;  // a drag orbits
  if (!picking) {
    const id = zoneEditable() ? viewer.pickZone(e.clientX, e.clientY) : null;
    if (id && (id !== selectedZone || !zoneTool.mode)) {
      $(`[data-sec="protect"]`).classList.remove("closed");
      selectZone(id, zoneTool.mode || "translate");
    }
    return;
  }
  const p = viewer.pick(e.clientX, e.clientY);
  if (!p) return toast("Click on the model surface.");
  const R = displayRotation();
  const body = applyMat3(transposeMat3(R), [p.x, p.y, p.z]);
  if (picking.kind === "feature") {
    const nrm = p.normal ? applyMat3(transposeMat3(R), [p.normal.x, p.normal.y, p.normal.z]) : null;
    const done = picking.resolve;
    picking.resolve = null;
    stopPick();
    done({ point: frame.bodyToFile(body), normal: nrm });
    return;
  }
  if (picking.kind === "prop") {
    const pr = props.placeAt(picking.id, body);
    if (pr) { props.render3D(R, true); toast(`${pr.name} placed. Set its size and rpm on the left.`, "good"); }
    stopPick();
    return;
  }
  const z = S.zones.find((q) => q.id === picking.id);
  if (z) {
    z.center = frame.bodyToFile(body).map(round);
    selectedZone = z.id;
    renderZones();
    renderZones3D();
    saveSettings();
    zoneTool.mode = zoneTool.mode || "translate";
    syncZoneTool();
    toast(`${z.name} placed. Drag the handles to move, turn (E) or resize (R) it.`, "good");
  }
  stopPick();
});

// ---- 3. flight envelope ------------------------------------------------------------------------------
function isa(hm) {
  const h_ = Math.min(Math.max(hm, -500), 20000);
  let T, p;
  if (h_ <= 11000) { T = 288.15 - 0.0065 * h_; p = 101325 * Math.pow(T / 288.15, 5.255877); }
  else { T = 216.65; p = 22632.06 * Math.exp(-9.80665 * (h_ - 11000) / (287.053 * T)); }
  const rho = p / (287.053 * T);
  const mu = 1.458e-6 * Math.pow(T, 1.5) / (T + 110.4);
  return { rho, nu: mu / rho, a: Math.sqrt(1.4 * 287.053 * T) };
}
function points(lo, hi, n) {
  n = Math.max(1, Math.round(n));
  if (n === 1 || hi - lo < 1e-9) return [Math.round(lo * 10) / 10];
  return [...new Set(Array.from({ length: n }, (_, i) => Math.round((lo + ((hi - lo) * i) / (n - 1)) * 10) / 10))];
}
function envelopePoints() {
  const e = S.envelope;
  return { speeds: points(e.speed_min, e.speed_max, e.speed_count), aoas: points(e.aoa_min, e.aoa_max, e.aoa_count) };
}
function reGroups(fid = fidelity) {
  const { speeds } = envelopePoints();
  const air = isa(S.envelope.altitude || 0);
  const L = frame ? frame.size[0] : 1;
  const cap = catalog.fidelities[fid].re_cap;
  const groups = new Set(speeds.map((v) => Math.round(Math.min((v * L) / air.nu, cap))));
  return { n: groups.size, direct: speeds.filter((v) => (v * L) / air.nu < cap).length, air, L, speeds };
}

for (const id of ["speed_min", "speed_max", "speed_count", "aoa_min", "aoa_max", "aoa_count", "altitude", "yaw"]) $("#" + id).value = S.envelope[id];
$("#ref_area").value = S.envelope.ref_area || "";
$("#speed_count").max = me.limits.max_speeds;
$("#aoa_count").max = me.limits.max_angles;
S.envelope.speed_count = Math.min(S.envelope.speed_count, me.limits.max_speeds);
S.envelope.aoa_count = Math.min(S.envelope.aoa_count, me.limits.max_angles);
$("#speed_count").value = S.envelope.speed_count;
$("#aoa_count").value = S.envelope.aoa_count;

const LIMITS = { speed_min: [0.5, 450], speed_max: [0.5, 450], aoa_min: [-30, 30], aoa_max: [-30, 30], altitude: [-500, 20000], yaw: [-30, 30],
  speed_count: [1, 25], aoa_count: [1, 25] };
for (const id of Object.keys(LIMITS)) {
  $("#" + id).addEventListener("change", (e) => {
    let v = parseFloat(e.target.value);
    const [lo, hi] = LIMITS[id];
    if (!Number.isFinite(v)) { e.target.value = S.envelope[id]; return; }
    if (id.endsWith("_count")) {
      v = Math.round(v);
      const max = id === "aoa_count" ? me.limits.max_angles : me.limits.max_speeds;
      if (v > max) {
        upgradePrompt(`${me.plan_name} sweeps up to ${max} ${id === "aoa_count" ? "angles" : "airspeeds"}.`);
        v = max;
      }
    }
    v = Math.min(hi, Math.max(lo, v));
    e.target.value = v;
    S.envelope[id] = v;
    if (id.startsWith("aoa") || id === "yaw") showPreview(false); else flowInfo();
    saveSettings();
    estimates();
  });
}
$("#ref_area").addEventListener("change", (e) => {
  const v = parseFloat(e.target.value);
  S.envelope.ref_area = Number.isFinite(v) && v > 0 ? v : null;
  saveSettings();
});

function flowInfo() {
  const { speeds, aoas } = envelopePoints();
  $("#speed-chips").innerHTML = speeds.map((v) => `<span>${v} m/s</span>`).join("");
  $("#aoa-chips").innerHTML = aoas.map((a) => `<span>${a}°</span>`).join("");
  const g = reGroups();
  const reLo = (Math.min(...speeds) * g.L) / g.air.nu, reHi = (Math.max(...speeds) * g.L) / g.air.nu;
  const mach = Math.max(...speeds) / g.air.a;
  $("#flow-kv").innerHTML = `
    <span class="k">Reynolds number</span><span class="v">${reLo.toExponential(1)} – ${reHi.toExponential(1)}</span>
    <span class="k">Highest Mach</span><span class="v" style="${mach > 0.3 ? "color:var(--warning-text)" : ""}">${mach.toFixed(3)}</span>
    <span class="k">CFD runs per design</span><span class="v">${tunnel === "rc" && S.propellers.length
      ? `${aoas.length * speeds.length} <span class="muted">(${aoas.length} angle${aoas.length > 1 ? "s" : ""} × ${speeds.length} speed${speeds.length > 1 ? "s" : ""}: props change with speed)</span>`
      : `${aoas.length * g.n} <span class="muted">(${aoas.length} angle${aoas.length > 1 ? "s" : ""}${g.n > 1 ? ` × ${g.n} Re groups` : ""})</span>`}</span>`;
  if (mach > 1.2) $("#flow-kv").insertAdjacentHTML("beforeend", `<span class="k" style="grid-column:1/-1;color:var(--critical-text)">Above Mach 1.2 — beyond this tunnel. Use the Supersonic tunnel for higher speeds.</span>`);
  else if (mach > 0.7) $("#flow-kv").insertAdjacentHTML("beforeend", `<span class="k" style="grid-column:1/-1;color:var(--warning-text)">Compressibility-corrected up to this body's critical Mach number
    (found from the run, typically 0.7–0.85); above it the transonic drag rise is an estimate that joins the Supersonic tunnel at Mach 1.2.</span>`);
  else if (mach > 0.3) $("#flow-kv").insertAdjacentHTML("beforeend", `<span class="k" style="grid-column:1/-1">Above Mach 0.3 the simulated surface pressures are corrected for compressibility (Kármán–Tsien).</span>`);
  if (tunnel === "rc" && props) props.refreshEstimates();
  if (g.direct && g.direct < speeds.length) $("#flow-kv").insertAdjacentHTML("beforeend", `<span class="k" style="grid-column:1/-1">${g.direct} low speed(s) run at their exact Reynolds number, adding runs.</span>`);
  summaries();
}

// ---- 4/5. simulate & optimize ---------------------------------------------------------------------------
const FID_LABEL = { draft: "Draft", standard: "Standard", high: "High" };
function renderFidelity() {
  if (tunnel === "supersonic") {
    $("#fid-block").classList.add("hidden");
    $("#panel-block").classList.remove("hidden");
    $("#panel-block").innerHTML = `${icon("flame")}<div class="small"><b style="color:var(--text)">Surface methods, results in seconds.</b>
      Shock and expansion pressures, skin friction and heating are computed face by face on your mesh, so there is no fidelity to pick:
      a finer mesh near the nose and leading edges gives finer heating maps.</div>`;
    return;
  }
  $("#fid-block").classList.remove("hidden");
  $("#panel-block").classList.add("hidden");
  const seg = $("#fidelity");
  seg.innerHTML = "";
  for (const f of Object.keys(catalog.fidelities)) {
    const allowed = me.limits.fidelities.includes(f);
    const b = h("button", { type: "button", class: f === fidelity ? "on" : "", style: { flex: 1 },
      title: allowed ? `${fmtInt(catalog.fidelities[f].cells)} cells` : "Upgrade to unlock", html: (allowed ? "" : icon("lock") + " ") + FID_LABEL[f] });
    b.addEventListener("click", () => {
      if (!allowed) return upgradePrompt(`${FID_LABEL[f]} fidelity (${fmtInt(catalog.fidelities[f].cells)}-cell tunnel) is part of a higher plan.`);
      fidelity = f;
      renderFidelity();
      flowInfo();
      estimates();
      saveSettings();
    });
    seg.append(b);
  }
  const fd = catalog.fidelities[fidelity];
  const where = COMPUTE.backend === "gpu" ? `GPU${COMPUTE.device ? " (" + COMPUTE.device.replace(/^NVIDIA (GeForce )?/, "").replace(/ with Max-Q Design/, "") + ")" : ""}` : "CPU";
  $("#fid-note").innerHTML = `${fmtInt(fd.cells)}-cell tunnel · ≈ ${fmtDuration(PER_RUN_S[fidelity])} per run on the ${esc(where)} · ${catalog.credits_per_run[fidelity]} credit${catalog.credits_per_run[fidelity] > 1 ? "s" : ""} per run`;
}

function renderObjective() {
  const sel = $("#objective");
  $("#respect-row").classList.toggle("hidden", tunnel !== "supersonic");
  $("#verify-row").classList.toggle("hidden", tunnel === "supersonic");
  $("#respect_limits").checked = opt.respect_limits !== false;
  if (tunnel === "supersonic") {
    if (!(opt.objective in catalog.hyper_objectives)) opt.objective = "drag";
    sel.innerHTML = Object.entries(catalog.hyper_objectives).map(([k, label]) => `<option value="${k}">${esc(label)}</option>`).join("");
    $("#objective-note").classList.add("hidden");
    sel.value = opt.objective;
    sel.onchange = () => { opt.objective = sel.value; saveSettings(); summaries(); };
    return;
  }
  if (winged()) {
    opt.objective = "drag_fixed_lift";
    sel.innerHTML = `<option value="drag_fixed_lift">${esc(catalog.objectives.drag_fixed_lift)}</option>`;
    sel.value = "drag_fixed_lift";
    $("#objective-note").classList.remove("hidden");
    $("#objective-note").innerHTML = `Wings are scored at the lift they must carry, so the optimizer cannot win by shrinking the wing. Lift, induced and
      section drag come from the validated wing model (vortex lattice + NeuralFoil, see <a href="/validation#wing" target="_blank">validation</a>);
      ${S.template.id === "uav" ? "the fuselage from Hoerner's method. The search runs on a coarse version of this model and the best designs are confirmed at full resolution — no CFD runs and almost no credits, so a large budget costs nothing. Run the CFD tunnel on the result afterwards to see its flow." : "the search runs on a coarse version of this model and the best designs are confirmed at full resolution — no CFD runs needed."}`;
    sel.onchange = null;
    return;
  }
  sel.innerHTML = Object.entries(catalog.objectives).filter(([k]) => k !== "drag_fixed_lift").map(([k, label]) => {
    const ok = me.limits.objectives.includes(k) && !unvalidated.includes(k);
    const why = unvalidated.includes(k) ? " — lift not yet validated" : " (Pro)";
    return `<option value="${k}" ${ok ? "" : "disabled"}>${esc(label)}${ok ? "" : why}</option>`;
  }).join("");
  if (!me.limits.objectives.includes(opt.objective) || unvalidated.includes(opt.objective) || opt.objective === "drag_fixed_lift") opt.objective = "drag";
  $("#objective-note").classList.toggle("hidden", !unvalidated.length);
  $("#objective-note").innerHTML = `Lift-based goals (best L/D, keep lift) are off until lift passes the
    <a href="/validation" target="_blank">solver validation</a>: on thin wings the simulated lift is currently 50–66% low.`;
  sel.value = opt.objective;
  sel.onchange = () => { opt.objective = sel.value; saveSettings(); summaries(); };
}

function renderAspects() {
  const box = $("#aspects");
  box.innerHTML = "";
  const max = me.limits.max_aspects;
  $("#aspect-limit").textContent = max < Object.keys(catalog.aspects).length ? `(up to ${max} on ${me.plan_name})` : "";
  opt.aspects = opt.aspects.filter((a) => a in catalog.aspects).slice(0, max);
  for (const [k, a] of Object.entries(catalog.aspects)) {
    const on = opt.aspects.includes(k);
    const full = !on && opt.aspects.length >= max;
    const card = h("div", { class: `aspect ${on ? "on" : ""} ${full ? "locked" : ""}`, role: "checkbox", "aria-checked": on, tabindex: 0 });
    card.innerHTML = `<input type="checkbox" ${on ? "checked" : ""} tabindex="-1" aria-hidden="true"><div><div class="t">${esc(a.label)}</div><div class="d">${esc(a.help)}</div></div>`;
    const toggle = () => {
      if (on) opt.aspects = opt.aspects.filter((x) => x !== k);
      else if (full) return upgradePrompt(`${me.plan_name} can change ${max} design aspects at once. Pro and Team can change all of them.`);
      else opt.aspects.push(k);
      renderAspects();
      estimates();
      saveSettings();
    };
    card.addEventListener("click", toggle);
    card.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
    box.append(card);
  }
}

function slider(id, value, fmtOut, onChange) {
  const el = $("#" + id), out = $("#" + id + "_o");
  el.value = value;
  out.textContent = fmtOut(+el.value);
  el.addEventListener("input", () => { out.textContent = fmtOut(+el.value); onChange(+el.value); estimates(); });
  el.addEventListener("change", saveSettings);
}
const maxIter = () => (tunnel === "supersonic" || winged() ? me.limits.max_iterations_panel : me.limits.max_iterations);
$("#iterations").max = maxIter();
opt.iterations = Math.min(opt.iterations, maxIter());
$("#respect_limits").addEventListener("change", (e) => { opt.respect_limits = e.target.checked; saveSettings(); });
slider("max_change", Math.round(opt.max_change * 100), (v) => v + "%", (v) => { opt.max_change = v / 100; });
slider("iterations", opt.iterations, (v) => String(v), (v) => { opt.iterations = v; });
$("#top_k").max = me.limits.max_top_k || 8;
$("#topk-limit").textContent = (me.limits.max_top_k || 8) < 8 ? `(up to ${me.limits.max_top_k} on ${me.plan_name})` : "";
slider("top_k", opt.top_k, (v) => `${v} parameter${v > 1 ? "s" : ""}`, (v) => { opt.top_k = v; });
for (const k of ["max_length", "max_width", "max_height", "min_thickness"]) {
  const el = $("#" + k);
  el.value = opt[k] ? +(opt[k] * 1000).toFixed(3) : "";
  el.addEventListener("change", () => { const v = parseFloat(el.value); opt[k] = Number.isFinite(v) && v > 0 ? v / 1000 : null; saveSettings(); });
}
$("#mass_kg").value = opt.mass_kg || "";
$("#mass_kg").addEventListener("change", (e) => { const v = parseFloat(e.target.value); opt.mass_kg = Number.isFinite(v) && v > 0 ? v : null; saveSettings(); });
$("#keep_stability").checked = opt.keep_stability !== false;
$("#keep_stability").addEventListener("change", (e) => { opt.keep_stability = e.target.checked; saveSettings(); });
$("#hold_wing_area").checked = !!opt.hold_wing_area;
$("#hold_wing_area").addEventListener("change", (e) => { opt.hold_wing_area = e.target.checked; saveSettings(); });

function renderOptMode() {
  const tm = S.template;
  $("#opt-aspects").classList.toggle("hidden", !!tm);
  $("#opt-params").classList.toggle("hidden", !tm);
  $("#opt-wing").classList.toggle("hidden", !(winged() && tunnel !== "supersonic"));
  $("#wingarea-row").classList.toggle("hidden", !!tm);
  $("#thick-row").classList.toggle("hidden", !tm);
  $("#verify-row").classList.add("hidden");
  $("#opt-lead").textContent = tm
    ? "Changes the named dimensions you left unlocked, within their ranges: a screening run ranks them, the most influential are optimized, and the best designs are confirmed at full resolution."
    : "Reshapes the body within your regions and limits: a screening run ranks the shape controls, the most influential are optimized, and the best designs are confirmed at full resolution.";
  if (tm && dparams) {
    const free = dparams.unlocked();
    $("#opt-params").innerHTML = `${icon("sliders")}<div class="small"><b style="color:var(--text)">${free.length} free parameter${free.length === 1 ? "" : "s"}</b>${free.length
      ? `: ${free.map((p) => esc(p.label)).join(", ")}.` : " — unlock some in Design parameters."} Lock anything that must stay as it is.</div>`;
  }
}
slider("min_volume", Math.round(opt.min_volume * 100), (v) => v + "%", (v) => { opt.min_volume = v / 100; });
slider("max_volume", Math.round(opt.max_volume * 100), (v) => v + "%", (v) => { opt.max_volume = v / 100; });

function renderVerify() {
  const sel = $("#verify");
  const order = Object.keys(catalog.fidelities);
  sel.innerHTML = `<option value="">No separate check</option>` + order.filter((f) => order.indexOf(f) > order.indexOf(fidelity)).map((f) => {
    const ok = me.limits.verify && me.limits.fidelities.includes(f);
    return `<option value="${f}" ${ok ? "" : "disabled"}>${FID_LABEL[f]} fidelity${ok ? "" : " (upgrade)"}</option>`;
  }).join("");
  sel.value = opt.verify && [...sel.options].some((o) => o.value === opt.verify && !o.disabled) ? opt.verify : "";
  opt.verify = sel.value || null;
  sel.onchange = () => { opt.verify = sel.value || null; estimates(); saveSettings(); };
}

function estimates() {
  if (tunnel === "supersonic") return hyperEstimates();
  const { aoas, speeds } = envelopePoints();
  const withProps = tunnel === "rc" && S.propellers.length > 0;
  const perDesign = withProps ? aoas.length * speeds.length : aoas.length * reGroups().n;
  const per = catalog.credits_per_run[fidelity];
  const time = (runs, fid) => PER_RUN_S[fid] * (1 + (runs - 1) * 0.55);
  $("#run-analyze").innerHTML = withProps ? `${icon("fan")}Run RC tunnel` : `${icon("wind")}Simulate envelope`;
  $("#analyze-est").innerHTML = `${perDesign} CFD run${perDesign > 1 ? "s" : ""} · ≈ ${fmtDuration(time(perDesign, fidelity))} · up to ${perDesign * per} credit${perDesign * per > 1 ? "s" : ""} · ${fmtInt(me.credits_left)} left`;
  const xf = catalog.explore_fidelity[fidelity] || "coarse";
  // optional ideas are screened on/off first (their dimensions join only if one is kept); with CFD every
  // screened design also runs at a sub-cell shift (paired)
  const nFeat = feats ? feats.freeParams().filter(([f, p]) => !f.optional || p.key === "_use")
    .reduce((n, [, p]) => n + (p.kind === "choice" ? p.choices.length - 1 : 1), 0) * (winged() ? 1 : 2) : 0;
  const nScreen = nFeat + (S.template && dparams ? dparams.unlocked().reduce((n, p) => n + (p.kind === "choice" ? p.choices.length - 1 : 1), 0) : opt.aspects.length * 1.5);
  const nA = Math.min(3, aoas.length);
  const perX = withProps ? nA * speeds.length : nA * reGroups().n;
  const exploreCfd = !winged();
  const xRuns = exploreCfd ? perX * (opt.iterations + 3) : 0;
  const cRuns = winged() ? 0 : 6 * perDesign;      // baseline, its error probe, 2 candidates, each again shifted
  const credits = Math.max(1, Math.ceil(xRuns * (catalog.credits_per_run[xf] || 1) + cRuns * per));
  const secs = (exploreCfd ? xRuns * PER_RUN_S[xf] * 0.6 : opt.iterations * 1.5) + (cRuns ? time(cRuns, fidelity) : 0);
  $("#run-optimize").innerHTML = `${icon("sparkles")}Optimize design`;
  $("#optimize-est").innerHTML = `≈ ${Math.round(nScreen)} screening + up to ${Math.max(0, opt.iterations - Math.round(nScreen))} search designs on ${exploreCfd ? `${xf} CFD (${nA} angle${nA > 1 ? "s" : ""})` : "the wing model"}${
    cRuns ? ` · 6 confirmation runs at ${FID_LABEL[fidelity].toLowerCase()} fidelity` : " · confirmed at full model resolution"} · ≈ ${fmtDuration(secs)} · up to ${credits} credit${credits > 1 ? "s" : ""}` +
    (opt.iterations < nScreen + 2 ? ` · <span style="color:var(--warning-text)">budget too small for the screening — lock some parameters or raise it</span>` : "") +
    (credits > me.credits_left ? ` · <span style="color:var(--critical-text)">needs more credits</span>` : "");
  renderOptMode();
  renderVerify();
  summaries();
}

function hyperEstimates() {
  const { machs, aoas } = hyper.points();
  const n = machs.length * aoas.length;
  const designs = opt.iterations + 2;
  const credits = 1 + Math.floor((designs - 1) / 10);
  $("#run-analyze").innerHTML = `${icon("flame")}Run supersonic sweep`;
  $("#analyze-est").innerHTML = `${n} condition${n > 1 ? "s" : ""} · a few seconds · 1 credit · ${fmtInt(me.credits_left)} left`;
  $("#run-optimize").innerHTML = `${icon("sparkles")}Optimize design`;
  $("#optimize-est").innerHTML = `up to ${opt.iterations} designs (screening first) × ${n} conditions · ≈ ${fmtDuration(Math.max(5, designs * n * 0.15))} · ${credits} credit${credits > 1 ? "s" : ""}`;
  renderOptMode();
  summaries();
}

function summaries() {
  if (tunnel === "supersonic") {
    $("#s-protect").textContent = S.zones.length ? `${S.zones.length} region${S.zones.length > 1 ? "s" : ""}` : "none";
    $("#s-simulate").textContent = "Panel";
    $("#s-optimize").textContent = S.template ? `top ${opt.top_k} · ${opt.iterations} designs` : `${opt.aspects.length} aspect${opt.aspects.length === 1 ? "" : "s"} · ${opt.iterations} it`;
    return;
  }
  const { speeds, aoas } = envelopePoints();
  const nCond = aoas.length * speeds.length;
  $("#s-envelope").textContent = `${nCond} condition${nCond > 1 ? "s" : ""}`;
  $("#s-envelope").title = `${aoas.length} angle${aoas.length > 1 ? "s" : ""} × ${speeds.length} speed${speeds.length > 1 ? "s" : ""}`;
  $("#s-protect").textContent = S.zones.length ? `${S.zones.length} region${S.zones.length > 1 ? "s" : ""}` : "none";
  $("#s-simulate").textContent = FID_LABEL[fidelity];
  $("#s-optimize").textContent = S.template ? `top ${opt.top_k} · ${opt.iterations} designs` : `${opt.aspects.length} aspect${opt.aspects.length === 1 ? "" : "s"} · ${opt.iterations} it`;
  if (S.template && dparams) $("#s-params").textContent = `${dparams.unlocked().length} free`;
}

$$(".sec-h").forEach((hd) => hd.addEventListener("click", () => hd.parentElement.classList.toggle("closed")));

// ---- jobs ---------------------------------------------------------------------------------------------------
function jobSpec(kind) {
  const spec = { kind, tunnel, units: S.units, ...(S.units === "custom" ? { unit_scale: S.unit_scale } : {}), forward: S.forward, up: S.up, zones: S.zones, features: S.features || [] };
  if (tunnel === "supersonic") spec.hyper = S.hyper;
  else Object.assign(spec, { fidelity, envelope: S.envelope, propellers: S.propellers });
  if (kind === "optimize") spec.optimize = { ...opt };
  return spec;
}
async function submit(kind, btn) {
  const freeFeat = feats ? feats.freeParams().length : 0;
  if (kind === "optimize" && !S.template && !opt.aspects.length && !freeFeat) return toast("Choose at least one aspect the optimizer may change, or unlock a shape-feature dimension.", "bad");
  if (kind === "optimize" && S.template && dparams && !dparams.unlocked().length && !freeFeat) return toast("Unlock at least one design parameter for the optimizer.", "bad");
  busy(btn, true, kind === "optimize" ? "Queuing optimization…" : "Queuing…");
  try {
    clearTimeout(saveTimer);
    const job = await api(`/api/projects/${pid}/jobs`, { method: "POST", body: jobSpec(kind) });
    watch(job);
    toast(kind === "optimize" ? "Optimization queued." : `${TUNNELS[tunnel].label} tunnel run queued.`, "good");
    loadRuns();
  } catch (e) {
    handleError(e);
  } finally {
    busy(btn, false);
    estimates();
  }
}
$("#run-analyze").addEventListener("click", (e) => submit("analyze", e.currentTarget));
$("#run-optimize").addEventListener("click", (e) => submit("optimize", e.currentTarget));

let pollTimer;
function watch(job) {
  watchJob = job;
  clearTimeout(pollTimer);
  renderStatus(job);
  if (job.status === "queued" || job.status === "running") pollTimer = setTimeout(poll, 1500);
}
async function poll() {
  if (!watchJob) return;
  try {
    const j = await api(`/api/jobs/${watchJob.id}`);
    const was = watchJob.status;
    watchJob = j;
    renderStatus(j);
    if (j.kind === "optimize" && j.live && j.live.history && !$("#tab-history").classList.contains("hidden")) renderHistoryTab();
    if (j.status === "queued" || j.status === "running") {
      pollTimer = setTimeout(poll, j.status === "running" ? 1500 : 3000);
    } else if (was !== j.status) {
      me = await getMe(true);
      top.update(me);
      estimates();
      loadRuns();
      if (j.status === "done") {
        toast(j.kind === "optimize" ? "Optimization finished." : "Envelope simulation finished.", "good");
        await openJob(j);
      } else if (j.status === "failed") toast(j.error || "The run failed.", "bad");
    }
  } catch (e) {
    pollTimer = setTimeout(poll, 5000);
  }
}

function renderStatus(j) {
  const badge = $("#st-badge"), meter = $("#st-meter"), stage = $("#st-stage");
  const cls = { running: "warn", done: "good", failed: "bad" }[j.status] || "";
  badge.className = "badge " + cls;
  badge.innerHTML = `<span class="dot"></span>${{ queued: "Queued", running: "Running", done: "Done", failed: "Failed", canceled: "Cancelled" }[j.status] || j.status}`;
  const active = j.status === "queued" || j.status === "running";
  meter.classList.toggle("hidden", !active);
  meter.firstElementChild.style.width = Math.round((j.progress || 0) * 100) + "%";
  $("#st-cancel").classList.toggle("hidden", !active);
  let txt = j.stage || "";
  if (j.status === "queued") txt = `Waiting in queue${j.queue_position ? ` (position ${j.queue_position})` : ""}…`;
  if (j.status === "failed") txt = j.error || "Failed.";
  if (j.status === "done") txt = `${j.kind === "optimize" ? "Optimization" : "Simulation"} finished ${timeAgo(j.finished_at)} · ${
    j.spec.tunnel === "supersonic" ? "surface methods" : `${j.cfd_runs} CFD run${j.cfd_runs === 1 ? "" : "s"}`} · ${j.credits_used} credit${j.credits_used === 1 ? "" : "s"}`;
  stage.textContent = txt;
  $("#log").textContent = j.log || "";
  renderLive(j);
}

// ---- live run card: what the tunnel is doing right now ----------------------------------------------------
const PHASES = {
  analyze: [["setup", "Tunnel setup"], ["solve", "Flow solve"], ["post", "Results"]],
  optimize: [["setup", "Setup"], ["baseline", "Baseline"], ["probe", "Solver error"], ["screen", "Screening"], ["search", "Search"], ["confirm", "Confirm"]],
};
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
function phaseOf(j) {
  const s = j.stage || "";
  if (j.status === "queued" || /^(Starting|Building)/.test(s)) return "setup";
  if (j.kind !== "optimize") return /^Done|Writing|fields$/.test(s) ? "post" : (j.live && j.live.run) || /surface methods/.test(s) ? "solve" : "setup";
  if (/^(Confirming|Best design)/.test(s)) return "confirm";
  if (/^Design \d+ of/.test(s)) return "search";
  if (/^Screening/.test(s)) return "screen";
  if (/error/i.test(s)) return "probe";
  if (/^Baseline/.test(s)) return "baseline";
  return "setup";
}
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function sparkline(vals, w = 236, hgt = 44) {
  if (!vals || vals.length < 2) return `<div class="rc-spark empty">Waiting for the first forces…</div>`;
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || Math.abs(hi) * 0.05 || 1;
  const pts = vals.map((v, i) => `${((i / (vals.length - 1)) * (w - 4) + 2).toFixed(1)},${(hgt - 4 - ((v - lo) / span) * (hgt - 10)).toFixed(1)}`).join(" ");
  const [lx, ly] = pts.split(" ").pop().split(",");
  return `<svg class="rc-spark" viewBox="0 0 ${w} ${hgt}" width="100%" height="${hgt}" aria-hidden="true">
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linejoin="round"/>
    <circle cx="${lx}" cy="${ly}" r="3.2" fill="var(--accent)"><animate attributeName="r" values="3;5;3" dur="1.2s" repeatCount="indefinite"/></circle></svg>`;
}

function designDots(history, w = 236, hgt = 44) {
  const js = (history || []).filter((e) => Number.isFinite(e.objective)).map((e) => e.objective);
  if (js.length < 2) return "";
  const lo = Math.min(...js, 0.97), hi = Math.max(...js, 1.03);
  const y = (v) => (2 + ((hi - v) / (hi - lo)) * (hgt - 6)).toFixed(1);
  let best = Infinity;
  const dots = js.map((v, i) => {
    const isBest = v < best;
    if (isBest) best = v;
    return `<circle cx="${((i / Math.max(js.length - 1, 1)) * (w - 8) + 4).toFixed(1)}" cy="${y(v)}" r="${isBest ? 3.2 : 2.2}" fill="${isBest ? "var(--good)" : "var(--muted)"}" opacity="${isBest ? 1 : 0.6}"/>`;
  }).join("");
  const gain = (1 - best) * 100;
  return `<div class="rc-row"><span class="k">Designs tried</span><span class="v">${js.length} · best ${gain > 0.05 ? `<b style="color:var(--good-text)">−${gain.toFixed(1)} %</b>` : "no gain yet"}</span></div>
    <svg class="rc-spark" viewBox="0 0 ${w} ${hgt}" width="100%" height="${hgt}" aria-hidden="true">
      <line x1="0" x2="${w}" y1="${y(1)}" y2="${y(1)}" stroke="var(--border-strong)" stroke-dasharray="3 3"/>${dots}</svg>`;
}

function renderLive(j) {
  const card = $("#run-card");
  const active = !!j && (j.status === "queued" || j.status === "running");
  card.classList.toggle("hidden", !active);
  viewer.setWind(active && j.status === "running" && !reducedMotion.matches, { fast: tunnel === "supersonic" });
  if (!active) return;
  const phases = PHASES[j.kind === "optimize" ? "optimize" : "analyze"];
  const ci = Math.max(0, phases.findIndex(([k]) => k === phaseOf(j)));
  const run = j.live && j.live.run;
  const pct = Math.round((j.progress || 0) * 100);
  const el = j.started_at ? Math.max(0, Date.now() / 1000 - j.started_at) : 0;
  const left = j.status === "running" && j.progress > 0.04 ? (el * (1 - j.progress)) / j.progress : null;
  const where = tunnel === "supersonic" ? "surface methods" : COMPUTE.backend === "gpu" ? "GPU" : "CPU";
  const tr = run && run.trace ? run.trace : [];
  const settle = tr.length >= 2 ? Math.abs(tr[tr.length - 1] - tr[tr.length - 2]) / Math.max(Math.abs(tr[tr.length - 1]), 1e-6) * 100 : null;
  const steps = phases.map(([, label], i) => `<span class="rc-step ${i < ci ? "done" : i === ci ? "now" : ""}">${i < ci ? icon("check") : `<i></i>`}${label}</span>`).join("");
  let runs = "";
  if (run && run.n > 1) {
    runs = run.n <= 48
      ? `<div class="rc-dots" title="CFD run ${run.i} of ${run.n}">${Array.from({ length: run.n }, (_, k) =>
        `<span class="${k + 1 < run.i ? "done" : k + 1 === run.i ? "now" : ""}"${k + 1 === run.i ? ` style="--f:${run.f}"` : ""}></span>`).join("")}</div>`
      : `<div class="meter rc-meter"><div style="width:${Math.round(((run.i - 1 + run.f) / run.n) * 100)}%"></div></div>`;
  }
  const cond = run ? `${esc(run.label)} · α ${run.aoa}°${run.speed != null ? ` · ${run.speed} m/s` : ""}` : esc(j.stage || "Preparing");
  card.innerHTML = `
    <div class="rc-head"><span class="rc-pulse"></span><b>${j.status === "queued" ? "Queued" : j.kind === "optimize" ? "Optimizing" : "Simulating"}</b>
      <span class="muted small">${esc(TUNNELS[tunnel].label)} tunnel · ${esc(where)}</span><span class="grow"></span><span class="rc-pct">${pct}%</span></div>
    <div class="rc-steps">${steps}</div>
    ${j.status === "queued" ? `<div class="rc-now">Waiting in queue${j.queue_position ? ` · position ${j.queue_position}` : ""}…</div>` : `
    <div class="rc-now">${cond}${run ? ` <span class="muted">· run ${run.i} of ${run.n}</span>` : ""}</div>
    ${runs}
    ${run ? `<div class="rc-row" title="The run ends when the averaged drag changes by less than 1 % between checks"><span class="k">Drag settling</span><span class="v">${
      settle == null ? "—" : `${settle < 1 ? icon("check") : ""}Δ ${settle < 10 ? settle.toFixed(1) : Math.round(settle)} %`}</span></div>${sparkline(tr)}` : ""}
    ${j.kind === "optimize" && j.live && j.live.history ? designDots(j.live.history) : ""}
    <div class="rc-foot"><span>${icon("clock")} ${clock(el)} elapsed</span>${left != null ? `<span>about ${fmtDuration(Math.max(1, left))} left</span>` : ""}</div>`}`;
}
// tick the elapsed clock between polls
setInterval(() => { if (watchJob && watchJob.status === "running" && !$("#run-card").classList.contains("hidden")) renderLive(watchJob); }, 1000);
$("#st-cancel").addEventListener("click", async () => {
  if (!watchJob) return;
  if (!(await confirmDialog("Cancel this run?", "Credits for CFD runs already finished are not refunded.", "Cancel run", true))) return;
  try { await api(`/api/jobs/${watchJob.id}/cancel`, { method: "POST" }); poll(); } catch (e) { handleError(e); }
});
$("#st-log").addEventListener("click", () => $("#log").classList.toggle("hidden"));
$("#st-results").addEventListener("click", () => $("#main").classList.toggle("show-right"));
const narrow = matchMedia("(max-width: 1050px)");
const syncNarrow = () => $("#st-results").classList.toggle("hidden", !narrow.matches);
narrow.addEventListener("change", syncNarrow);
syncNarrow();

// ---- results: files & geometry --------------------------------------------------------------------------------
async function getFile(jobId, name, kind = "json") {
  const key = jobId + "/" + name;
  if (fileCache[key]) return fileCache[key];
  const res = await fetch(`/api/jobs/${jobId}/files/${name}`, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`Could not load ${name}`);
  return (fileCache[key] = kind === "json" ? await res.json() : parseSTL(await res.arrayBuffer()));
}
const isOpt = () => result && result.kind === "optimize";
const designKey = () => (isOpt() && display !== "baseline" ? "best" : "baseline");
function currentFieldEntry() {
  if (!result.fields) return { aoa: result.envelope.aoas[cond.a], mirror: false, file: null };
  return result.fields[designKey()][cond.a];
}

async function openJob(j) {
  try {
    shownJob = j;
    if (!watchJob || watchJob.id === j.id || !["queued", "running"].includes(watchJob.status)) renderStatus(j);
    const r = await getFile(j.id, "result.json");
    if (!r.envelope) {  // a run from before flight envelopes: not displayable here
      toast("This run was made with an older version and cannot be displayed.", "bad");
      return;
    }
    result = r;
    display = isOpt() ? "best" : "baseline";
    const env = result.envelope;
    cond.a = env.aoas.reduce((bi, a, i) => (Math.abs(a) < Math.abs(env.aoas[bi]) ? i : bi), 0);
    cond.v = Math.floor((env.speeds.length - 1) / 2);
    renderCond();
    await showResultGeometry();
    renderResults();
    renderHistoryTab();
    switchTab("results");
    $$("#tab-runs .job").forEach((el) => el.classList.toggle("on", el.dataset.id === j.id));
  } catch (e) {
    handleError(e, "Could not load the results.");
  }
}

const isHyper = () => result && result.tunnel === "supersonic";
function setDeformField(body, base) {
  const d = new Float32Array(body.length / 3);
  let mx = 0;
  for (let i = 0; i < d.length; i++) {
    d[i] = Math.hypot(body[i * 3] - base[i * 3], body[i * 3 + 1] - base[i * 3 + 1], body[i * 3 + 2] - base[i * 3 + 2]);
    mx = Math.max(mx, d[i]);
  }
  viewer.setVertexField("deform", d, 0, mx || 1, "deform");
}
async function showHyperGeometry(entry, R) {
  const j = shownJob;
  const [body, fields] = await Promise.all([getFile(j.id, result.meshes[designKey()], "stl"), getFile(j.id, entry.file)]);
  const firstResults = !showingResults;
  const pos = rotate(body, R);
  viewer.setModel(pos, { fit: firstResults });
  viewer.setMirror(false);
  showingResults = true;
  const f = hyperFaceFields(fields, cond.v, result);
  ranges.cp = [Math.min(-0.1, percentile(f.cp, 0.02)), Math.max(0.1, percentile(f.cp, 0.99))];
  ranges.temp = [Math.max(f.twRange[0], percentile(f.tw, 0.02)), Math.max(f.twRange[0] + 1, percentile(f.tw, 0.999))];
  ranges.flux = [0, Math.max(1, percentile(f.q, 0.995))];
  const mlo = Math.min(-1, percentile(f.margin, 0)), mhi = Math.max(1, percentile(f.margin, 0.99));
  ranges.margin = [mlo, mhi];
  viewer.setFaceField("cp", f.cp, ranges.cp[0], ranges.cp[1], "cp");
  viewer.setFaceField("temp", f.tw, ranges.temp[0], ranges.temp[1], "temp");
  viewer.setFaceField("flux", f.q, 0, ranges.flux[1], "flux");
  viewer.setFaceField("margin", f.margin, mlo, mhi, "margin");
  if (isOpt()) {
    const base = await getFile(j.id, result.meshes.baseline, "stl");
    if (base.length === body.length && designKey() === "best") setDeformField(body, base);
    viewer.setGhost(display === "overlay" ? rotate(base, R) : null);
  } else viewer.setGhost(null);
  viewer.setTunnel(null);
  viewer.setStreamlines(null);
  viewer.setSlice(null);
  viewer.setProps([]);
  let ix = 0;
  for (let i = 0; i < pos.length; i += 3) if (pos[i] < pos[ix]) ix = i;
  let xmax = -Infinity;
  for (let i = 0; i < pos.length; i += 3) xmax = Math.max(xmax, pos[i]);
  const M = result.envelope.machs[cond.v];
  viewer.setShock(vis.shock ? [pos[ix], pos[ix + 1], pos[ix + 2]] : null, Math.asin(1 / M), 1.3 * (xmax - pos[ix]));
  if (firstResults || !["shaded", "wire", "cp", "temp", "flux", "margin", "deform"].includes(vis.mode)) vis.mode = "temp";
  viewer.setMode(vis.mode);
  updateToolbar();
  renderZones3D();
  $("#vp-info").textContent = `Surface methods · Mach ${M} · ${sig(result.envelope.altitude / 1000, 3)} km`;
}

async function showResultGeometry() {
  const j = shownJob;
  const entry = currentFieldEntry();
  if (isHyper()) return showHyperGeometry(entry, windRotation(entry.aoa, result.envelope.yaw));
  const simAoa = entry.mirror ? -entry.aoa : entry.aoa;
  const R = windRotation(simAoa, result.envelope.yaw);
  if (!entry.file) return showModelOnly(j, R);
  const [body, fields] = await Promise.all([getFile(j.id, result.meshes[designKey()], "stl"), getFile(j.id, entry.file)]);
  const firstResults = !showingResults;
  viewer.setModel(rotate(body, R), { fit: firstResults });
  viewer.setMirror(entry.mirror);
  showingResults = true;
  const [clo, chi] = fields.surface_cp_range;
  const cp = dequantize(b64bytes(fields.surface_cp), clo, chi);
  const sp = dequantize(b64bytes(fields.surface_speed), 0, 1.6);
  ranges.cp = [Math.min(-0.1, percentile(cp, 0.02)), Math.max(0.1, percentile(cp, 0.98))];
  ranges.speed = [0, Math.max(0.5, percentile(sp, 0.98))];
  viewer.setFaceField("cp", cp, ranges.cp[0], ranges.cp[1], "cp");
  viewer.setFaceField("speed", sp, ranges.speed[0], ranges.speed[1], "speed");
  if (isOpt()) {
    const base = await getFile(j.id, result.meshes.baseline, "stl");
    if (base.length === body.length && designKey() === "best") setDeformField(body, base);
    viewer.setGhost(display === "overlay" ? rotate(base, R) : null);
  } else viewer.setGhost(null);
  viewer.setShock(null);
  if (result.propellers) props.render3DFromResult(result, R); else viewer.setProps([]);
  viewer.setTunnel(fields.slices.bounds);
  viewer.setStreamlines(fields.streamlines);
  viewer.showStreamlines(vis.stream);
  viewer.setSlice(fields.slices, vis.slice, vis.sliceField);
  if (firstResults && vis.mode === "shaded") vis.mode = "cp";
  if (!(vis.mode in viewer.fields) && !["shaded", "wire"].includes(vis.mode)) vis.mode = "cp";
  viewer.setMode(vis.mode);
  updateToolbar();
  renderZones3D();
  const st = result.setup;
  $("#vp-info").textContent = `${st.grid.join(" × ")} cells · ${st.regimes.join(" + ")}${entry.mirror ? " · mirrored from +" + Math.abs(entry.aoa) + "°" : ""}${
    st.fields_speed ? ` · flow shown at ${st.fields_speed} m/s` : ""}`;
}

async function showModelOnly(j, R) {
  // results from the wing model: geometry, no flow fields
  const body = await getFile(j.id, result.meshes[designKey()], "stl");
  const firstResults = !showingResults;
  viewer.setModel(rotate(body, R), { fit: firstResults });
  viewer.setMirror(false);
  showingResults = true;
  if (isOpt()) {
    const base = await getFile(j.id, result.meshes.baseline, "stl");
    if (base.length === body.length && designKey() === "best") setDeformField(body, base);
    viewer.setGhost(display === "overlay" ? rotate(base, R) : null);
  } else viewer.setGhost(null);
  viewer.setShock(null);
  viewer.setProps([]);
  viewer.setTunnel(null);
  viewer.setStreamlines(null);
  viewer.setSlice(null);
  if (!["shaded", "wire", "deform"].includes(vis.mode)) vis.mode = "shaded";
  viewer.setMode(vis.mode);
  updateToolbar();
  renderZones3D();
  $("#vp-info").textContent = "Wing model (vortex lattice + NeuralFoil): no flow fields";
}

function renderCond() {
  const box = $("#vp-cond");
  if (!result) { box.classList.add("hidden"); return; }
  const env = result.envelope;
  box.classList.remove("hidden");
  box.innerHTML = `<span>α</span><div class="seg" id="cond-a">${env.aoas.map((a, i) => `<button data-i="${i}" class="${i === cond.a ? "on" : ""}">${a}°</button>`).join("")}</div>
    ${isHyper() ? `<span>M</span><div class="seg" id="cond-v">${env.machs.map((m, i) => `<button data-i="${i}" class="${i === cond.v ? "on" : ""}">${m}</button>`).join("")}</div>`
      : `<span>V</span><div class="seg" id="cond-v">${env.speeds.map((v, i) => `<button data-i="${i}" class="${i === cond.v ? "on" : ""}">${v}</button>`).join("")}</div><span class="muted">m/s</span>`}
    ${currentFieldEntry().mirror ? `<span class="badge" title="Symmetric body: this angle is the mirror image of +${Math.abs(env.aoas[cond.a])}°">mirrored</span>` : ""}`;
  box.querySelectorAll("#cond-a button").forEach((b) => b.addEventListener("click", async () => {
    cond.a = +b.dataset.i; renderCond(); await showResultGeometry(); renderResults(); renderHistoryTab();
  }));
  box.querySelectorAll("#cond-v button").forEach((b) => b.addEventListener("click", async () => {
    cond.v = +b.dataset.i; renderCond();
    if (isHyper()) await showResultGeometry();
    renderResults(); renderHistoryTab();
  }));
}

function updateToolbar() {
  const has = (m) => m in viewer.fields;
  $$("#g-surface .vbtn").forEach((b) => {
    const m = b.dataset.mode;
    b.disabled = !(m === "shaded" || m === "wire" || has(m));
    b.classList.toggle("on", vis.mode === m);
  });
  const res = showingResults;
  const flow = res && !!(result && result.fields) && !isHyper();     // wing-model results have no flow fields
  const hy = res && isHyper();
  $$(".hyper-only").forEach((b) => b.classList.toggle("hidden", !hy));
  $$(".cfd-only").forEach((b) => b.classList.toggle("hidden", hy));
  $("#g-shock").classList.toggle("hidden", !hy);
  $("#t-shock").disabled = !hy;
  $("#t-shock").classList.toggle("on", hy && vis.shock);
  $("#t-stream").disabled = !flow;
  $("#t-slice").disabled = !flow;
  $("#t-slicefield").disabled = !flow || !vis.slice;
  $("#t-stream").classList.toggle("on", flow && vis.stream);
  $("#t-slice").classList.toggle("on", flow && !!vis.slice);
  $("#t-slice").textContent = "Slice: " + ({ xz: "side", xy: "top" }[vis.slice] || "off");
  $("#t-slicefield").textContent = "Slice: " + (vis.sliceField === "cp" ? "pressure" : "speed");
  $("#t-zones").classList.toggle("on", vis.zones);
  $("#g-design").classList.toggle("hidden", !(res && isOpt()));
  $$("#g-design .vbtn").forEach((b) => b.classList.toggle("on", b.dataset.show === display));
  colorbar();
}

function colorbar() {
  const cb = $("#colorbar");
  let map = null, lo, hi;
  if (vis.mode === "cp" && "cp" in viewer.fields) { map = "cp"; [lo, hi] = ranges.cp; }
  else if (vis.mode === "speed" && "speed" in viewer.fields) { map = "speed"; [lo, hi] = ranges.speed; }
  else if (vis.mode === "deform" && "deform" in viewer.fields) { map = "deform"; lo = 0; hi = viewer.fields.deform.hi; }
  else if (["temp", "flux", "margin"].includes(vis.mode) && vis.mode in viewer.fields) { map = vis.mode; [lo, hi] = ranges[vis.mode]; }
  else if (showingResults && vis.slice) { map = vis.sliceField === "cp" ? "cp" : "speed"; [lo, hi] = map === "cp" ? [-1, 1] : [0, 1.4]; }
  if (!map) return cb.classList.add("hidden");
  cb.classList.remove("hidden");
  cb.querySelector(".t").textContent = map === "deform" ? "Shape change (displacement)" : MAPS[map].label;
  cb.querySelector(".bar").style.background = cssGradient(map, lo, hi);
  const f = (v) => (map === "deform" ? sig(v * 1000, 2) + " mm" : map === "temp" ? degC(v) : map === "flux" ? sig(v / 1000, 2) + " kW/m²"
    : map === "margin" ? `${v > 0 ? "+" : ""}${Math.round(v)} K` : sig(v, 2));
  const mid = map === "cp" || map === "margin" ? 0 : (lo + hi) / 2;
  cb.querySelector(".ticks").innerHTML = [lo, mid, hi].map((v) => `<span>${v === 0 ? "0" : f(v)}</span>`).join("");
}

$$("#g-surface .vbtn").forEach((b) => b.addEventListener("click", () => {
  vis.mode = b.dataset.mode;
  viewer.setMode(vis.mode);
  updateToolbar();
}));
$("#t-stream").addEventListener("click", () => { vis.stream = !vis.stream; viewer.showStreamlines(vis.stream); updateToolbar(); });
$("#t-slice").addEventListener("click", async () => { vis.slice = { null: "xz", xz: "xy", xy: null }[String(vis.slice)]; await refreshSlice(); });
$("#t-slicefield").addEventListener("click", async () => { vis.sliceField = vis.sliceField === "speed" ? "cp" : "speed"; await refreshSlice(); });
$("#t-zones").addEventListener("click", () => { vis.zones = !vis.zones; renderZones3D(); updateToolbar(); });
$("#t-shock").addEventListener("click", async () => { vis.shock = !vis.shock; await showResultGeometry(); });
async function refreshSlice() {
  if (!showingResults || !shownJob || !currentFieldEntry().file) { updateToolbar(); return; }
  const fields = await getFile(shownJob.id, currentFieldEntry().file);
  viewer.setSlice(fields.slices, vis.slice, vis.sliceField);
  updateToolbar();
}
$$("#g-design .vbtn").forEach((b) => b.addEventListener("click", async () => { display = b.dataset.show; await showResultGeometry(); }));
$$("#g-cam [data-view]").forEach((b) => b.addEventListener("click", () => viewer.view(b.dataset.view)));
$("#snap").addEventListener("click", () => {
  const a = h("a", { href: viewer.snapshot(1600, 1000), download: `${project.name} - view.png` });
  document.body.append(a);
  a.click();
  a.remove();
});

// ---- right panel ------------------------------------------------------------------------------------------------
function switchTab(name) {
  $$(".panel-head .tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  for (const n of ["results", "history", "runs"]) $("#tab-" + n).classList.toggle("hidden", n !== name);
  if (name === "history") renderHistoryTab();
}
$$(".panel-head .tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

const kpi = (label, value, sub = "") => `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
const ldStr = (v, cl) => (v === null || v === undefined ? "—" : Math.abs(cl) < 0.005 ? "≈ 0" : sig(v, 3));
function delta(a, b, lowerBetter = true) {
  if (a === null || b === null || a === undefined || b === undefined || Math.abs(a) < 1e-9) return "";
  const d = (b - a) / Math.abs(a);
  if (Math.abs(d) < 0.0005) return "";
  const good = lowerBetter ? d < 0 : d > 0;
  // From a near-zero baseline a percentage is meaningless (+4800%): show the absolute change.
  const txt = Math.abs(d) > 2 ? `${b > a ? "+" : "−"}${sig(Math.abs(b - a), 3)}` : fmtPct(d, 1, true);
  return `<span class="delta ${good ? "good" : "bad"}">${txt}</span>`;
}

function emptyResults() {
  if (tunnel === "supersonic") {
    $("#tab-results").innerHTML = `<div class="notice info">${icon("flame")}<div>
      <b style="color:var(--text)">No supersonic results yet.</b> Check the orientation, set the Mach range and altitude, pick the skin material
      (and material regions for hot spots), then <b>Run supersonic sweep</b>. It takes seconds.</div></div>
      <div class="section-title">What you will get</div>
      <ul class="small" style="padding-left:18px;color:var(--text-2);margin:0">
        <li>Wave, base and friction drag against Mach number</li>
        <li>Wall temperature and heat-flux maps, checked against each material's limit</li>
        <li>Stagnation heating, lift, pitching moment and centre of pressure</li></ul>`;
    return;
  }
  if (tunnel === "rc") {
    $("#tab-results").innerHTML = `<div class="notice info">${icon("fan")}<div>
      <b style="color:var(--text)">No RC results yet.</b> Add your propellers (size in inches and rpm), place each hub on the model,
      set the airspeed range, then <b>Run RC tunnel</b>.</div></div>
      <div class="section-title">What you will get</div>
      <ul class="small" style="padding-left:18px;color:var(--text-2);margin:0">
        <li>Airframe drag measured inside the propeller slipstream</li>
        <li>Thrust, shaft power and propeller efficiency at every airspeed</li>
        <li>Thrust against drag, and the estimated level top speed</li></ul>`;
    return;
  }
  $("#tab-results").innerHTML = `<div class="notice info">${icon("info")}<div>
    <b style="color:var(--text)">No results yet.</b> Work down the steps on the left: check the model's orientation,
    mark anything that must not change, set the airspeeds and angles to test, then <b>Simulate envelope</b>.</div></div>
    <div class="section-title">What you will get</div>
    <ul class="small" style="padding-left:18px;color:var(--text-2);margin:0">
      <li>Drag polar, lift curve and L/D across every angle and airspeed</li>
      <li>Pitching-moment slope (stability) and drag at each speed</li>
      <li>Surface pressure, streamlines and flow slices at each angle</li></ul>
    <p class="small muted" style="margin-top:12px">Physics, accuracy and limits are explained in the <a href="/docs#physics">guide</a> and on the <a href="/validation">validation page</a>.</p>`;
}

function polarSeries(polar, key, vIdx) {
  return polar.map((p) => [p.aoa, key === "cl" || key === "cm_pitch" ? p[key] : p.speeds[vIdx][key]]);
}

function renderResults() {
  const box = $("#tab-results");
  if (!result) return emptyResults();
  if (isHyper()) {
    box.innerHTML = hyperResultsHTML(result, cond, display, shownJob.id) + featureChangesHTML(result, catalog);
    const go = $("#go-opt");
    if (go) go.addEventListener("click", () => { $('[data-sec="optimize"]').classList.remove("closed"); $("#run-optimize").scrollIntoView({ behavior: "smooth" }); });
    hyperCharts(chartCard($("#charts")), result, cond, display);
    if (!$("#charts").children.length) $("#charts").innerHTML = `<p class="small muted">Sweep more Mach numbers or angles to see curves.</p>`;
    return;
  }
  const env = result.envelope;
  const base = result.baseline.polar, best = isOpt() ? result.best.polar : null;
  const pb = base[cond.a], sb = pb.speeds[cond.v];
  const po = best ? best[cond.a] : null, so = po ? po.speeds[cond.v] : null;
  const p = po || pb, s = so || sb;
  const sum = (best ? result.best : result.baseline).summary;
  const lifting = base.some((q) => Math.abs(q.cl) > 0.02);
  const clv = s.cl ?? p.cl, cmv = s.cm_pitch ?? p.cm_pitch;
  const mc = p.mach_crit;
  const bc = result.setup.body_cells || [99, 99, 99];
  const flat = bc[1] > 1.5 * bc[2];
  const resCells = result.setup.ln_cells ?? Math.min(...bc);
  box.innerHTML = `
    <div class="row spread"><div><div class="label" style="margin:0">At ${env.aoas[cond.a]}° and ${env.speeds[cond.v]} m/s</div>
      <div class="hero-num">${sig(s.cd, 3)} <span class="small muted" style="font-size:15px;font-weight:600">Cd</span></div></div>
      ${best ? `<span class="badge accent">${display === "baseline" ? "Baseline" : "Optimized"}</span>` : ""}</div>
    <div class="kpis" style="margin-top:12px">
      ${kpi("Drag", fmtForce(s.drag_N), best ? delta(sb.drag_N, so.drag_N) + " vs baseline" : "")}
      ${kpi("Lift", fmtForce(s.lift_N), `Cl ${sig(clv, 3)}`)}
      ${kpi("Lift / drag", ldStr(s.ld, clv), best ? delta(sb.ld, so.ld, false) : "")}
      ${result.setup.method === "wing model" ? kpi("Induced / profile drag", `${fmtForce(s.cd_pressure * s.q * result.setup.ref_area)} / ${fmtForce(s.cd_friction * s.q * result.setup.ref_area)}`, "profile includes the fuselage")
        : kpi("Pitching moment Cm", sig(cmv, 3), result.baseline.summary.cm_alpha !== undefined ? `Cm<sub>α</sub> ${sig(sum.cm_alpha, 3)} /rad ${sum.cm_alpha < 0 ? "· stable" : "· unstable"}` : "")}
      ${kpi("Power to overcome drag", fmtPower(s.power_W))}
      ${kpi("Reynolds number", s.re.toExponential(2), `Mach ${s.mach.toFixed(3)}${s.compressibility && s.compressibility !== "incompressible" ? ` · ${s.compressibility}` : ""}`)}
    </div>
    ${s.compressibility === "transonic estimate" ? `<div class="notice warn" style="margin-top:10px">${icon("alert")}<div><b style="color:var(--text)">Transonic estimate.</b>
      Mach ${s.mach.toFixed(2)} is above this body's critical Mach number (${sig(mc, 2)}), where shocks start to form. The drag here bridges the corrected CFD
      and the supersonic surface methods at Mach 1.2; it is an estimate, not a validated result. Lift and moment are held at their Mach ${sig(mc, 2)} values.</div></div>`
      : s.compressibility === "corrected" ? `<div class="notice" style="margin-top:10px">${icon("info")}<div class="small">Compressibility-corrected (Kármán–Tsien) from the simulated surface pressures;
      this body's critical Mach number is ${sig(mc, 2)}.</div></div>` : ""}
    ${resCells < 12 ? `<div class="notice warn" style="margin-top:10px">${icon("alert")}<div>Coarse for this airframe: its
      ${flat ? "mean chord" : "size across"} spans only ${Math.round(resCells)} cells at ${esc(result.setup.fidelity)} fidelity, so thin wings and fins are
      one-cell walls. Use a higher fidelity for final drag numbers.</div></div>` : ""}
    ${result.setup.method === "wing model" ? `<div class="notice good" style="margin-top:10px">${icon("check")}<div><b style="color:var(--text)">Wing model.</b>
      Lift, induced drag and section drag from the vortex lattice and NeuralFoil, both checked against published data on the
      <a href="/validation#wing" target="_blank">validation page</a>; the pitching moment is not computed here.</div></div>` : ""}
    ${lifting && result.setup.method !== "wing model" ? `<div class="notice warn" style="margin-top:10px">${icon("alert")}<div><b style="color:var(--text)">Lift, L/D and pitching moment are indicative only.</b>
      The solver's drag is validated; its lift is not yet — on thin wings it currently comes out 50–66% low
      (<a href="/validation" target="_blank">validation</a>). Use these values for trends, not for sizing.</div></div>` : ""}
    ${lifting && sum.ld_max ? `<div class="notice" style="margin-top:10px">${icon("target")}<div>Best L/D in the envelope: <b style="color:var(--text)">${sig(sum.ld_max, 3)}</b> at ${sum.ld_max_at.aoa}° and ${sum.ld_max_at.v} m/s${sum.cl_alpha !== undefined ? ` · lift slope ${sig(sum.cl_alpha, 3)} /rad` : ""}</div></div>` : ""}
    ${rcResultsHTML(result, s, sb, sum, best)}
    <div class="section-title">Drag breakdown at this condition</div>
    ${breakdown(s, p)}
    <div class="section-title">Across the envelope</div>
    <div class="charts" id="charts"></div>
    <div class="section-title">Polar at ${env.speeds[cond.v]} m/s</div>
    <div style="overflow-x:auto"><table class="tbl compact"><thead><tr><th class="r">α</th><th class="r">Cl</th><th class="r">Cd</th><th class="r">L/D</th><th class="r">Cm</th><th class="r">Drag</th></tr></thead><tbody>
      ${(best || base).map((q, i) => `<tr class="${i === cond.a ? "sel" : ""}"><td class="r">${q.aoa}°${q.mirror ? "*" : ""}</td><td class="r">${sig(q.cl, 3)}</td>
        <td class="r">${sig(q.speeds[cond.v].cd, 3)}</td><td class="r">${ldStr(q.speeds[cond.v].ld, q.cl)}</td><td class="r">${sig(q.cm_pitch, 3)}</td>
        <td class="r">${fmtForce(q.speeds[cond.v].drag_N)}</td></tr>`).join("")}</tbody></table></div>
    ${result.setup.symmetric ? `<p class="small muted" style="margin-top:6px">* Mirrored from the positive angle: the body is symmetric top-to-bottom, so it was not simulated twice.</p>` : ""}
    ${details()}
    <div class="row wrap" style="margin-top:16px">${!best ? `<button class="btn sm" id="go-opt">${icon("sparkles")}Optimize this design</button>` : ""}
      <a class="btn sm ghost" href="/app/report/${shownJob.id}" target="_blank">${icon("file")}Report</a></div>`;
  const go = $("#go-opt");
  if (go) go.addEventListener("click", () => { $('[data-sec="optimize"]').classList.remove("closed"); $("#run-optimize").scrollIntoView({ behavior: "smooth" }); });
  drawCharts(base, best);
}

function breakdown(s, p) {
  const wm = result && result.setup.method === "wing model";
  const pr = Math.max(wm ? s.cd_pressure : p.cd_pressure ?? (s.cd - s.cd_friction), 0), fr = Math.max(s.cd_friction, 0), tot = pr + fr || 1;
  const [l1, l2] = wm ? ["Induced", "Profile + fuselage"] : ["Pressure", "Friction"];
  return `<div class="split-bar" role="img" aria-label="Pressure ${fmtPct(pr / tot, 0)}, friction ${fmtPct(fr / tot, 0)}">
      <div style="width:${(pr / tot) * 100}%;background:var(--series-1)"></div><div style="width:${(fr / tot) * 100}%;background:var(--series-2)"></div></div>
    <div class="legend" style="margin-top:8px"><span><span class="sw" style="background:var(--series-1)"></span>${l1} Cd ${sig(pr, 3)} (${fmtPct(pr / tot, 0)})</span>
      <span><span class="sw" style="background:var(--series-2)"></span>${l2} Cd ${sig(fr, 3)} (${fmtPct(fr / tot, 0)})</span></div>`;
}

function details() {
  const st = result.setup;
  if (st.method === "wing model") return "";
  const p = result.baseline.polar[cond.a];
  return `<details style="margin-top:14px"><summary class="label" style="cursor:pointer">Run details</summary><div class="kv" style="margin-top:8px">
    <span class="k">Fidelity</span><span class="v">${esc(st.fidelity)}</span>
    <span class="k">Tunnel grid</span><span class="v">${st.grid.join(" × ")} (${fmtInt(st.cells)})</span>
    <span class="k">Cell size</span><span class="v">${sig(st.dx_m * 1000, 3)} mm</span>
    <span class="k">Cells across body</span><span class="v">${st.body_cells.map((v) => Math.round(v)).join(" × ")}</span>
    <span class="k">Regime</span><span class="v">${st.regimes.join(" + ")}</span>
    <span class="k">CFD runs per design</span><span class="v">${st.runs_per_design}${st.symmetric ? " (symmetry used)" : ""}</span>
    <span class="k">Reference area / length</span><span class="v">${sig(st.ref_area, 4)} m² / ${sig(st.ref_length, 4)} m</span>
    <span class="k">Blockage (corrected)</span><span class="v">${fmtPct(p.blockage, 1)}</span>
    <span class="k">Converged at all angles</span><span class="v">${result.baseline.polar.every((q) => q.converged) ? "yes" : "no — treat with care"}</span>
    <span class="k">Total time</span><span class="v">${fmtDuration(result.elapsed_s)} · ${result.cfd_runs} runs</span></div></details>`;
}

function chartCard(box) {
  box.innerHTML = "";
  return (title, sub) => {
    const c = h("div", { class: "chart-card" });
    c.innerHTML = `<div class="ct">${title}<span>${sub}</span></div>`;
    const area = h("div");
    c.append(area);
    box.append(c);
    return area;
  };
}

function drawCharts(base, best) {
  const env = result.envelope;
  const box = $("#charts");
  const card = chartCard(box);
  rcCharts(card, result, base, best, cond);
  const series = (key) => {
    const out = [{ name: best ? "Baseline" : "This design", color: "var(--series-1)", points: polarSeries(base, key, cond.v), dots: true, faded: !!best }];
    if (best) out.push({ name: "Optimized", color: "var(--series-3)", points: polarSeries(best, key, cond.v), dots: true });
    return out;
  };
  const sel = best || base;
  const fmtA = (v) => `${+v.toFixed(1)}°`;
  if (env.aoas.length > 1) {
    lineChart(card("Drag coefficient vs angle", `${env.speeds[cond.v]} m/s`), series("cd"),
      { height: 150, xName: "α", fmtX: fmtA, fmtY: (v) => sig(v, 3), highlight: { x: env.aoas[cond.a], y: sel[cond.a].speeds[cond.v].cd } });
    lineChart(card("Lift coefficient vs angle", "all speeds"), series("cl"),
      { height: 150, xName: "α", fmtX: fmtA, fmtY: (v) => sig(v, 3), highlight: { x: env.aoas[cond.a], y: sel[cond.a].cl } });
    if (base.some((q) => Math.abs(q.cl) > 0.02)) {
      lineChart(card("Lift-to-drag ratio vs angle", `${env.speeds[cond.v]} m/s`), series("ld"),
        { height: 150, xName: "α", fmtX: fmtA, fmtY: (v) => sig(v, 3) });
    }
    if (result.setup.method !== "wing model") lineChart(card("Pitching moment vs angle", "about the body centre"), series("cm_pitch"),
      { height: 150, xName: "α", fmtX: fmtA, fmtY: (v) => sig(v, 3) });
  }
  const hi = sel[cond.a].speeds[sel[cond.a].speeds.length - 1];
  if (env.speeds.length > 1 && hi.mach > 0.5 && hi.compressibility) {
    const cdm = (polar) => polar[cond.a].speeds.map((s) => [s.mach, s.cd]);
    const ser = [{ name: best ? "Baseline" : "This design", color: "var(--series-1)", points: cdm(base), dots: true, faded: !!best }];
    if (best) ser.push({ name: "Optimized", color: "var(--series-3)", points: cdm(best), dots: true });
    const mc = sel[cond.a].mach_crit;
    if (mc) {
      const ys = ser.flatMap((q) => q.points.map((pt) => pt[1]));
      ser.push({ name: `critical Mach ${sig(mc, 2)}`, color: "var(--warning)", points: [[mc, Math.min(...ys)], [mc, Math.max(...ys)]], dashed: true });
    }
    lineChart(card("Drag coefficient vs Mach", `${env.aoas[cond.a]}° · dashed: critical Mach`), ser,
      { height: 150, xName: "M", xLabel: "Mach number", fmtY: (v) => sig(v, 3) });
  }
  if (env.speeds.length > 1) {
    const drag = (polar) => polar[cond.a].speeds.map((s) => [s.v, s.drag_N]);
    const ser = [{ name: best ? "Baseline" : "This design", color: "var(--series-1)", points: drag(base), dots: true, faded: !!best }];
    if (best) ser.push({ name: "Optimized", color: "var(--series-3)", points: drag(best), dots: true });
    lineChart(card("Drag force vs airspeed", `${env.aoas[cond.a]}°`), ser,
      { height: 150, xName: "V", xLabel: "airspeed (m/s)", fmtY: (v) => fmtForce(v), highlight: { x: env.speeds[cond.v], y: sel[cond.a].speeds[cond.v].drag_N } });
  }
  if (!box.children.length) box.innerHTML = `<p class="small muted">Add more angles or airspeeds to the envelope to see curves.</p>`;
}

// ---- optimization tab ------------------------------------------------------------------------------------------------
function renderHistoryTab() {
  const box = $("#tab-history");
  const live = watchJob && watchJob.kind === "optimize" && ["queued", "running"].includes(watchJob.status) && watchJob.live && watchJob.live.history;
  if (live) return historyBlock(box, watchJob.live.history, watchJob.spec.optimize, null);
  if (!isOpt()) {
    box.innerHTML = `<p class="small muted">Run <b>Optimize design</b> to see the redesign, the per-iteration history and what mattered.</p>`;
    return;
  }
  const r = result;
  const b = r.baseline.polar, o = r.best.polar;
  const env = r.envelope;
  const vr = r.volume_ratio;
  const fixedLift = r.objective === "drag_fixed_lift";
  const headline = r.objective === "ld" ? fmtPct(r.improvement, 1, true) : fmtPct(fixedLift ? -r.improvement : r.drag_change, 1, true);
  const good = r.objective === "ld" ? r.improvement > 0 : fixedLift ? r.improvement > 0 : r.drag_change < 0;
  const st = r.strategy;
  const noReliable = st && !st.reliable;
  const head = isHyper() ? hyperHistoryHTML(r, cond) : `<div class="label">${esc(r.objective_label)}</div>
    <div class="hero-num">${headline}</div>
    <div class="small muted" style="margin:4px 0 12px">${r.objective === "ld" ? "change in best L/D" : fixedLift ? `drag at the lift needed, averaged over ${env.speeds.length} speed${env.speeds.length > 1 ? "s" : ""}`
      : "average change in drag across all " + env.aoas.length * env.speeds.length + " conditions"}
      ${st ? ` · ${st.designs_explored} designs explored` : ` · best design #${r.best_index} of ${r.history.length - 1}`}${r.verified ? ` · confirmed at ${esc(r.verified.fidelity)} fidelity` : ""}</div>
    ${noReliable ? `<div class="notice warn" style="margin-bottom:12px">${icon("alert")}<div><b style="color:var(--text)">No reliable improvement.</b> No candidate beat the
      original design by more than the solver's measured error (${fmtPct(st.epsilon, 2)})${st.confirmed ? " once checked at full resolution" : ""}, so the original is kept.
      Unlock other parameters, widen their ranges, or raise the budget.</div></div>
      ${r.promising ? `<div class="card lead-card" style="margin-bottom:12px"><div class="row" style="gap:8px;align-items:baseline"><b>Most promising design</b>
        <span class="badge warn">unconfirmed</span><span class="grow"></span><span class="hero-num" style="font-size:22px">${fmtPct(r.promising.change, 1, true)}</span></div>
        <p class="small muted" style="margin:6px 0 8px">The best design explored, with a drag change smaller than the solver's error (${fmtPct(r.promising.eps, 1)}): a lead
          to check at higher fidelity, not a proven result.${r.promising.features ? " " + r.promising.features.filter((c) => c.unit === "choice" && c.best !== c.base)
          .map((c) => esc(c.label)).join(", ") : ""}</p>
        <div class="row" style="gap:6px"><button class="btn sm" type="button" id="lead-view">${icon("eye")}View in 3D</button>
        <button class="btn sm ghost" type="button" id="lead-dl">${icon("download")}STL${me.limits.export ? "" : " " + icon("lock", "i sm")}</button></div></div>` : ""}`
      : !good ? `<div class="notice warn" style="margin-bottom:12px">${icon("alert")}<div>No design beat the baseline within the allowed changes. Try more iterations, a larger maximum change, or other aspects.</div></div>` : ""}
    <table class="tbl compact"><thead><tr><th>At ${env.aoas[cond.a]}°, ${env.speeds[cond.v]} m/s</th><th class="r">Baseline</th><th class="r">Optimized</th><th class="r">Change</th></tr></thead><tbody>
      ${row("Drag", b[cond.a].speeds[cond.v].drag_N, o[cond.a].speeds[cond.v].drag_N, fmtForce)}
      ${row("Cd", b[cond.a].speeds[cond.v].cd, o[cond.a].speeds[cond.v].cd, (v) => sig(v, 3))}
      ${row("Lift", b[cond.a].speeds[cond.v].lift_N, o[cond.a].speeds[cond.v].lift_N, fmtForce, false, 0.02 * Math.abs(b[cond.a].speeds[cond.v].drag_N))}
      ${row("Best L/D in envelope", r.baseline.summary.ld_max, r.best.summary.ld_max, (v) => (v ? sig(v, 3) : "—"), false, 0.05)}
      ${row("Mean drag in envelope", r.baseline.summary.drag_mean_N, r.best.summary.drag_mean_N, fmtForce)}
      <tr><td>Volume</td><td class="r">100%</td><td class="r">${fmtPct(vr, 1)}</td><td class="r">${fmtPct(vr - 1, 1, true)}</td></tr></tbody></table>
    ${fixedLiftHTML(r)}`;
  box.innerHTML = `${head}
    ${r.zones && r.zones.length ? `<p class="small" style="margin:10px 0 0">${icon("shieldlock", "i sm")} Respected ${r.zones.length} region${r.zones.length > 1 ? "s" : ""}: ${r.zones.map((z) => esc(z.name)).join(", ")}.</p>` : ""}
    ${r.template ? paramsTableHTML(r, catalog) : `<div class="section-title">Best design</div>
    <div class="kv">${r.variables.filter((v) => !v.key.startsWith("F.")).map((v) => `<span class="k">${esc(v.label)}</span><span class="v">${v.key === "fairing" ? fmtPct(r.best_values[v.key], 0) : fmtPct(r.best_values[v.key], 1, true)}</span>`).join("")}</div>`}
    ${featureChangesHTML(r, catalog)}
    ${strategyHTML(r)}
    ${r.relevance && !st ? `<div class="section-title">What mattered</div><div>${Object.entries(r.relevance).sort((a, c) => c[1] - a[1]).map(([k, rel]) => {
      const v = r.variables.find((x) => x.key === k);
      return `<div class="row" style="gap:8px;margin:6px 0"><span class="small" style="width:118px;color:var(--text-2)">${esc(v ? v.label : k)}</span>
        <div class="grow"><div class="rel-bar" style="width:${Math.max(3, rel * 100)}%"></div></div><span class="small num" style="width:38px;text-align:right">${fmtPct(rel, 0)}</span></div>`;
    }).join("")}</div>` : ""}
    <div class="row wrap" style="margin-top:16px">
      <button class="btn sm primary" id="dl">${icon("download")}Optimized STL${me.limits.export ? "" : " " + icon("lock", "i sm")}</button>
      <button class="btn sm" id="promote">${icon("rotate")}${r.template ? "Continue from these dimensions" : "Continue from this design"}</button>
      <a class="btn sm ghost" href="/app/report/${shownJob.id}" target="_blank">${icon("file")}Report</a></div>
    <div id="hist"></div>`;
  $("#dl").addEventListener("click", () => {
    if (!me.limits.export) return upgradePrompt("Exporting optimized geometry is part of Pro and Team. Your results stay here either way.");
    location.href = `/api/jobs/${shownJob.id}/files/optimized.stl`;
  });
  if ($("#lead-view")) {
    $("#lead-view").addEventListener("click", async () => {
      try {
        const body = await getFile(shownJob.id, r.promising.mesh, "stl");
        const R = displayRotation();
        viewer.setModel(rotate(body, R), { fit: false });
        viewer.setGhost(rotate(await getFile(shownJob.id, result.meshes.baseline, "stl"), R));
        viewer.setMode("shaded");
        $("#vp-info").textContent = "Most promising design (unconfirmed) · the original is the wireframe";
      } catch (e) { handleError(e); }
    });
    $("#lead-dl").addEventListener("click", () => {
      if (!me.limits.export) return upgradePrompt("Exporting geometry is part of Pro and Team.");
      location.href = `/api/jobs/${shownJob.id}/files/promising.stl`;
    });
  }
  $("#promote").addEventListener("click", async (e) => {
    busy(e.currentTarget, true);
    try { location.href = `/app/project/${(await api(`/api/jobs/${shownJob.id}/promote`, { method: "POST" })).id}`; }
    catch (err) { handleError(err); busy(e.currentTarget, false); }
  });
  historyBlock($("#hist"), r.history, null, r);
}

function row(label, a, c, fmtv, lowerBetter = true, minAbs = 0) {
  const ok = a !== null && a !== undefined && c !== null && c !== undefined && Math.abs(a) > minAbs;
  const d = ok ? (c - a) / Math.abs(a) : NaN;
  const good = lowerBetter ? d < 0 : d > 0;
  return `<tr><td>${label}</td><td class="r">${fmtv(a)}</td><td class="r">${fmtv(c)}</td>
    <td class="r ${!Number.isFinite(d) || Math.abs(d) < 0.0005 ? "" : good ? "delta good" : "delta bad"}">${Number.isFinite(d) ? fmtPct(d, 1, true) : "—"}</td></tr>`;
}

function historyBlock(box, hist, spec, res) {
  if (!hist || !hist.length) { box.innerHTML = `<p class="small muted">Waiting for the baseline…</p>`; return; }
  const obj = res ? res.objective : (spec && spec.objective) || "drag";
  const ld = obj === "ld";
  const val = (e) => (ld ? -e.objective : (e.objective ?? e.drag_ratio) * 100);
  const scored = (e) => e.objective !== null && e.objective !== undefined && !["diverged", "rejected", "error probe"].includes(e.status) && e.stage !== "confirm";
  const ok = hist.filter(scored);
  let best = Infinity;
  const bestPts = hist.filter(scored).map((e) => { best = Math.min(best, e.objective); return [e.i, ld ? -best : best * 100]; });
  const STAGE = { baseline: "original", noise: "error probe", screen: "screening", search: "search", confirm: "confirm" };
  const total = spec ? spec.iterations : hist.length - 1;
  const wrap = h("div");
  wrap.innerHTML = `<div class="section-title">${res ? "History" : `Live · ${hist.length - 1} of ${total} designs`}</div>
    <div class="chart-card"><div class="ct">${ld ? "Best L/D in envelope" : "Envelope drag, % of baseline"}<span>per design</span></div><div id="hist-chart"></div></div>
    <div style="overflow-x:auto;margin-top:8px"><table class="tbl compact"><thead><tr><th>#</th><th>Stage</th><th>Result</th><th class="r">${ld ? "L/D max" : "Objective"}</th></tr></thead><tbody>
    ${hist.map((e) => `<tr class="${e.best ? "best" : ""}"><td>${e.i}${e.best ? " ★" : ""}</td><td>${esc(STAGE[e.stage] || e.stage || "")}</td>
      <td title="${esc(e.note || "")}">${esc(e.status)}${e.note && e.status === "rejected" ? ` <span class="muted">· ${esc(e.note)}</span>` : ""}</td>
      <td class="r">${e.objective === null || e.objective === undefined ? "—" : ld ? sig(-e.objective, 3) : fmtPct(e.objective, 2)}</td></tr>`).join("")}</tbody></table></div>
    <p class="small muted" style="margin-top:6px">#0 is your design (100%). Error probes re-run it shifted by part of a grid cell to measure the solver's noise;
      screening moves one parameter at a time; the search follows a Gaussian-process model's expected improvement; confirmations are at full resolution.</p>`;
  box.innerHTML = "";
  box.append(wrap);
  lineChart(wrap.querySelector("#hist-chart"), [
    { name: "Each design", color: "var(--series-1)", points: ok.map((e) => [e.i, val(e)]), dots: true, faded: true },
    { name: "Best so far", color: "var(--series-3)", points: bestPts, step: true },
  ], { height: 160, integerX: true, xName: "design", xLabel: "design #", fmtY: (v) => (ld ? sig(v, 3) : sig(v, 3) + "%") });
}

// ---- runs tab ---------------------------------------------------------------------------------------------------------
async function loadRuns() {
  const jobs = await api(`/api/projects/${pid}/jobs`);
  const box = $("#tab-runs");
  if (!jobs.length) { box.innerHTML = `<p class="muted small">No runs yet for this project.</p>`; return jobs; }
  box.innerHTML = `<div class="joblist"></div>`;
  const list = box.firstChild;
  for (const j of jobs) {
    const s = j.summary;
    const env = j.spec.envelope;
    const tl = TUNNELS[j.spec.tunnel || (j.spec.propellers && j.spec.propellers.length ? "rc" : "general")].label;
    const fid = j.spec.tunnel === "supersonic" ? "surface methods" : j.spec.fidelity;
    const what = j.kind === "optimize" ? `${tl} · optimize · ${j.spec.optimize.iterations} designs · ${fid}` : `${tl} · envelope · ${fid}`;
    let res = "";
    if (j.status === "done" && s) res = j.kind === "optimize" ? `${fmtPct(s.drag_change, 1, true)} drag` : `Cd ${sig(s.cd, 3)}`;
    const cls = { running: "warn", done: "good", failed: "bad" }[j.status] || "";
    const sub = !env ? "single condition (older run)" : env.machs ? `${env.aoas.length} angles × ${env.machs.length} Mach` : `${env.aoas.length} angles × ${env.speeds.length} speeds`;
    const el = h("div", { class: "job" + (shownJob && shownJob.id === j.id ? " on" : ""), dataset: { id: j.id } });
    el.innerHTML = `<span class="badge ${cls}"><span class="dot"></span>${esc(j.status)}</span>
      <div class="grow"><div style="font-size:13.5px;color:var(--text)">${esc(what)}</div>
      <div class="small muted">${timeAgo(j.created_at)} · ${sub}</div></div><span class="small" style="color:var(--text-2)">${res}</span>`;
    el.addEventListener("click", async () => {
      if (j.status === "done") await openJob(await api(`/api/jobs/${j.id}`));
      else watch(await api(`/api/jobs/${j.id}`));
    });
    list.append(el);
  }
  return jobs;
}

// ---- tunnels ------------------------------------------------------------------------------------------------------------
const tctx = {
  S, me, catalog, viewer, round, frame: () => frame, save: () => saveSettings(),
  changed: () => { saveSettings(); estimates(); if (tunnel === "supersonic") hyper.info(frame ? frame.size[0] : null); if (tunnel === "rc") { props.render3D(null, !showingResults); flowInfo(); } },
  startPick: (id, kind) => startPick(id, kind), displayRotation: () => displayRotation(), preview: () => showPreview(false),
  speeds: () => envelopePoints().speeds, showing: () => showingResults,
};
props = new Propellers(tctx);
hyper = new HyperPanel(tctx);

function applyTunnel(t, user = false) {
  tunnel = t;
  document.body.dataset.tunnel = t;
  $$("#tunnel-switch button").forEach((b) => { b.classList.toggle("on", b.dataset.t === t); b.setAttribute("aria-selected", b.dataset.t === t); });
  $("#tunnel-note").textContent = TUNNELS[t].note;
  $('[data-sec="envelope"] .lead').textContent = t === "rc"
    ? "The airspeeds and angles of attack to test. With propellers turning, each angle and airspeed is its own CFD run, because the propeller loading changes with speed."
    : "The airspeeds and angles of attack to test. Each angle is one CFD run; extra airspeeds come from the same run, so they are free.";
  let n = 0;
  $$("#left .sec").forEach((sec) => {
    const show = (!sec.dataset.tunnels || sec.dataset.tunnels.split(" ").includes(t)) && (!sec.dataset.template || !!S.template);
    sec.classList.toggle("hidden", !show);
    if (show) sec.querySelector(".sec-h .n").textContent = ++n;
  });
  $("#iterations").max = maxIter();
  opt.iterations = Math.min(Math.max(Number(opt.iterations) || 8, 1), maxIter());
  $("#iterations").value = opt.iterations;
  $("#iterations_o").textContent = opt.iterations;
  renderZoneButtons();
  renderZones();
  props.render();
  renderFidelity();
  renderObjective();
  if (user) {
    result = null;
    shownJob = null;
    showPreview(false);
    emptyResults();
    renderHistoryTab();
    saveSettings();
  }
  estimates();
  if (t === "supersonic") hyper.info(frame ? frame.size[0] : null);
}
$$("#tunnel-switch button").forEach((b) => b.addEventListener("click", () => { if (b.dataset.t !== tunnel) applyTunnel(b.dataset.t, true); }));

// ---- design parameters (template projects) ----------------------------------------------------------------------------------
if (S.template) {
  dparams = new DesignParams({
    S, catalog, me,
    onGeometry: (pos) => {
      raw = pos;
      rebuildFrame();
      showPreview(false);
      $("#vp-empty").classList.add("hidden");
    },
    save: () => saveSettings(),
    changed: () => { renderOptMode(); estimates(); },
  });
  dparams.render($("#params-box"));
}

// ---- shape features -------------------------------------------------------------------------------------------------------
feats = new FeatureEditor({
  S, catalog, me, pid,
  pick: (type, hint) => pickForFeature(type, hint),
  onGeometry: (pos) => {
    designBody = pos;
    showPreview(false);
    $("#s-features").textContent = (S.features || []).length ? `${S.features.filter((f) => f.enabled !== false).length} on` : "none";
  },
  save: () => saveSettings(),
  changed: () => { renderOptMode(); estimates(); renderZones3D(); },
  select: (id, tool) => selectZone(id, tool ? zoneTool.mode || "translate" : zoneTool.mode),
  selected: () => selectedZone,
});
feats.render($("#features-box"));
$("#s-features").textContent = (S.features || []).length ? `${S.features.filter((f) => f.enabled !== false).length} on` : "none";
if (S.template && (S.template.id === "uav" || S.template.id === "wing")) $('[data-sec="features"]').classList.add("hidden");

// ---- boot ----------------------------------------------------------------------------------------------------------------
applyTunnel(tunnel);
renderZoneButtons();
renderZones();
renderFidelity();
renderObjective();
renderAspects();
emptyResults();
renderHistoryTab();
try {
  await loadModel();
  if ((S.features || []).some((f) => f.enabled !== false)) feats.rebuild(false);
} catch (e) {
  $("#vp-empty").innerHTML = `${icon("alert")}<span>${esc(e.message)}</span>`;
}
estimates();
try {
  const jobs = await loadRuns();
  const active = jobs.find((j) => j.status === "queued" || j.status === "running");
  const sameTunnel = (j) => (j.spec.tunnel || (j.spec.propellers && j.spec.propellers.length ? "rc" : "general")) === tunnel;
  const lastDone = jobs.find((j) => j.status === "done" && j.spec.envelope && sameTunnel(j));
  if (lastDone) await openJob(await api(`/api/jobs/${lastDone.id}`));
  if (active) watch(await api(`/api/jobs/${active.id}`));
} catch (e) { handleError(e); }
