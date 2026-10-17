// Rockets: masses, motor, launch site and recovery on the left; the Flight tab on the right —
// stability, drag curve, fin flutter and the flight itself, computed live by the component
// methods (rocket/*.py) every time the design or the setup changes.
import { api, esc, fmtInt, h, handleError, icon, sig, toast, $, $$ } from "/static/js/core.js";
import { lineChart } from "/static/js/charts.js";

export const ROCKET_GOALS = ["apogee", "rocket_drag"];

// display units: metric, or the imperial units US rocketry works in
export const UNITS = {
  metric: {
    key: "metric", len: (m) => `${sig(m * 1000, 4)} mm`, alt: (m) => `${fmtInt(Math.round(m))} m`, speed: (v) => `${sig(v, 3)} m/s`,
    mass: (g) => `${sig(g, 4)} g`, dist: (m) => `${fmtInt(Math.round(m))} m`, accel: (g) => `${sig(g, 3)} g`,
    lenIn: ["mm", 1000], massIn: ["g", 1], altIn: ["m", 1], speedIn: ["m/s", 1], railIn: ["m", 1], chuteIn: ["cm", 100],
  },
  imperial: {
    key: "imperial", len: (m) => `${sig(m / 0.0254, 3)} in`, alt: (m) => `${fmtInt(Math.round(m / 0.3048))} ft`, speed: (v) => `${sig(v * 2.23694, 3)} mph`,
    mass: (g) => `${sig(g / 28.3495, 3)} oz`, dist: (m) => `${fmtInt(Math.round(m / 0.3048))} ft`, accel: (g) => `${sig(g, 3)} g`,
    lenIn: ["in", 1 / 0.0254], massIn: ["oz", 1 / 28.3495], altIn: ["ft", 1 / 0.3048], speedIn: ["mph", 2.23694], railIn: ["ft", 1 / 0.3048], chuteIn: ["in", 1 / 0.0254],
  },
};
const toIn = (u, kind, v) => (v === null || v === undefined || v === "" ? "" : +(v * u[kind][1]).toPrecision(5));
const fromIn = (u, kind, v) => { const x = parseFloat(v); return Number.isFinite(x) ? x / u[kind][1] : null; };

export class RocketPanel {
  /** ctx: { S, me, pid, catalog, values(), save(), onEvaluated(ev) } */
  constructor(ctx) {
    this.ctx = ctx;
    const S = ctx.S;
    S.rocket = Object.assign(structuredClone(ctx.catalog.defaults), S.rocket || {});
    for (const k of ["masses", "materials", "motor", "launch", "parachute"]) S.rocket[k] = Object.assign(structuredClone(ctx.catalog.defaults[k]), S.rocket[k] || {});
    this.R = S.rocket;
    this.motors = [...ctx.catalog.motors, ...(ctx.catalog.user_motors || [])];
    this.ev = null;
    this.timer = null;
    this.seq = 0;
  }

  units() { return UNITS[this.R.units] || UNITS.metric; }
  motor(id) { return this.motors.find((m) => m.id === id); }

  render(box) {
    this.box = box;
    const R = this.R, u = this.units(), cat = this.ctx.catalog;
    const opt = (obj, cur) => Object.entries(obj).map(([k, l]) => `<option value="${k}" ${k === cur ? "selected" : ""}>${esc(l)}</option>`).join("");
    const groups = [["Built-in (approximate curves)", cat.motors], ["My motors", cat.user_motors || []]];
    box.innerHTML = `<p class="lead">Masses, motor, launch site and recovery. The <b>Flight</b> tab on the right updates as you change anything here or in the design.</p>
      <div class="row spread" style="margin-bottom:10px"><span class="label" style="margin:0">Units</span>
        <div class="seg" id="rk-units"><button type="button" data-v="metric">Metric</button><button type="button" data-v="imperial">Imperial</button></div></div>
      <span class="label">Masses</span>
      <div class="seg" id="rk-mass-mode" style="width:100%;display:flex"><button type="button" data-v="estimate" style="flex:1">Estimate</button>
        <button type="button" data-v="dry" style="flex:1">Measured</button><button type="button" data-v="items" style="flex:1">Part by part</button></div>
      <div id="rk-mass-estimate" style="margin-top:8px"><div class="three">
        <label class="field sm"><span>Nose</span><select id="rk-mat-nose">${opt(cat.materials.nose, R.materials.nose)}</select></label>
        <label class="field sm"><span>Body</span><select id="rk-mat-body">${opt(cat.materials.body, R.materials.body)}</select></label>
        <label class="field sm"><span>Fins</span><select id="rk-mat-fins">${opt(cat.fin_materials, R.materials.fins)}</select></label></div>
        <p class="small muted" style="margin:-6px 0 10px">Typical construction from the geometry: a longer nose or thicker fins weigh more, so the optimizer cannot cheat with mass.</p></div>
      <div id="rk-mass-dry" class="two" style="margin-top:8px">
        <label class="field sm"><span>Dry mass (no motor)</span><div class="input-unit"><input type="number" step="any" id="rk-dry-mass" value="${toIn(u, "massIn", R.masses.dry_mass_g)}"><span class="unit">${u.massIn[0]}</span></div></label>
        <label class="field sm"><span>CG from the nose tip</span><div class="input-unit"><input type="number" step="any" id="rk-dry-cg" value="${toIn(u, "lenIn", R.masses.dry_cg_mm === null || R.masses.dry_cg_mm === undefined ? null : R.masses.dry_cg_mm / 1000)}" placeholder="balance point"><span class="unit">${u.lenIn[0]}</span></div></label></div>
      <div id="rk-mass-items" style="margin-top:8px"></div>
      <span class="label" style="margin-top:14px">Motor</span>
      <select id="rk-motor">${groups.map(([g, ms]) => (ms.length ? `<optgroup label="${esc(g)}">${ms.map((m) => `<option value="${m.id}" ${m.id === R.motor.id ? "selected" : ""}>${esc(m.manufacturer)} ${esc(m.name)} · ${m.diameter_mm} mm · ${sig(m.total_impulse, 3)} N·s</option>`).join("")}</optgroup>` : "")).join("")}</select>
      <div class="row" style="gap:8px;margin-top:8px;align-items:flex-end;flex-wrap:wrap">
        <label class="field sm" style="width:130px;margin:0"><span>Ejection delay</span><select id="rk-delay"></select></label>
        <button class="btn sm ghost" type="button" id="rk-motor-upload">${icon("download")}Upload .eng / .rse</button>
        <input type="file" id="rk-motor-file" accept=".eng,.rse,.txt" class="hidden">
      </div>
      <div class="kv" id="rk-motor-info" style="margin-top:8px"></div>
      <details class="adv" style="margin-top:6px"><summary>Find a motor online <span class="muted">ThrustCurve.org</span></summary>
        <div class="row" style="gap:8px;margin-top:8px"><input type="text" id="rk-online-q" placeholder="C6, H128, AeroTech…" style="flex:1">
          <button class="btn sm" type="button" id="rk-online-go">Search</button></div>
        <p class="small muted" style="margin:6px 0 0">Searches thrustcurve.org when you press Search (a request leaves this computer); the certified curve is saved to <b>My motors</b>.</p>
        <div id="rk-online" style="margin-top:8px"></div></details>
      <span class="label" style="margin-top:14px">Launch</span>
      <div class="two">
        <label class="field sm"><span>Rail / rod length</span><div class="input-unit"><input type="number" step="any" min="0" id="rk-rail" value="${toIn(u, "railIn", R.launch.rail_m)}"><span class="unit">${u.railIn[0]}</span></div></label>
        <label class="field sm"><span>Launch angle (into the wind)</span><div class="input-unit"><input type="number" step="1" id="rk-angle" value="${R.launch.angle_deg}"><span class="unit">°</span></div></label>
        <label class="field sm"><span>Wind</span><div class="input-unit"><input type="number" step="any" min="0" id="rk-wind" value="${toIn(u, "speedIn", R.launch.wind_m_s)}"><span class="unit">${u.speedIn[0]}</span></div></label>
        <label class="field sm"><span>Site elevation</span><div class="input-unit"><input type="number" step="any" id="rk-elev" value="${toIn(u, "altIn", R.launch.elevation_m)}"><span class="unit">${u.altIn[0]}</span></div></label>
      </div>
      <span class="label" style="margin-top:6px">Recovery</span>
      <label class="check"><input type="checkbox" id="rk-chute-on" ${R.parachute.enabled !== false ? "checked" : ""}><span>Parachute</span></label>
      <div class="two" id="rk-chute-opts">
        <label class="field sm"><span>Diameter</span><div class="input-unit"><input type="number" step="any" min="0" id="rk-chute-d" value="${toIn(u, "chuteIn", R.parachute.diameter_m)}" placeholder="auto ≈ 4.5 m/s"><span class="unit">${u.chuteIn[0]}</span></div></label>
        <label class="field sm"><span>Parachute Cd</span><input type="number" step="0.05" min="0.3" max="3" id="rk-chute-cd" value="${R.parachute.cd}"></label></div>
      <details class="adv"><summary>Construction and limits <span class="muted">finish, lug, fin edges, margins</span></summary>
        <div class="two" style="margin-top:8px">
          <label class="field sm"><span>Surface finish</span><select id="rk-finish">${opt(cat.finishes, R.finish)}</select></label>
          <label class="field sm"><span>Launch lug</span><select id="rk-lug">${opt(cat.launch_lugs, R.launch_lug)}</select></label>
          <label class="field sm"><span>Fin edges</span><select id="rk-edge">${opt({ square: "Square (as cut)", rounded: "Rounded", airfoil: "Airfoiled / tapered" }, R.fin_edge)}</select></label>
          <label class="field sm"><span>Fin material (flutter)</span><select id="rk-fin-mat">${opt(cat.fin_materials, R.fin_material)}</select></label>
          <label class="field sm"><span>Minimum static margin</span><div class="input-unit"><input type="number" step="0.1" min="0" max="5" id="rk-min-margin" value="${R.min_margin}"><span class="unit">cal</span></div></label>
          <label class="field sm"><span>Minimum rail-exit speed</span><div class="input-unit"><input type="number" step="any" min="3" id="rk-min-rail" value="${toIn(u, "speedIn", R.min_rail_speed)}"><span class="unit">${u.speedIn[0]}</span></div></label>
        </div>
        <p class="small muted" style="margin:-4px 0 0">Designs under these limits are rejected by the optimizer. NAR guidance: at least 1 calibre of static margin, and about 15 m/s (50 ft/s) off the rail.</p>
      </details>
      <details class="adv"><summary>Compare with a real flight <span class="muted">altimeter log</span></summary>
        <div class="row" style="gap:8px;margin-top:8px;align-items:center"><button class="btn sm ghost" type="button" id="rk-log-upload">${icon("download")}Upload CSV (time, altitude)</button>
          <button class="btn sm ghost hidden" type="button" id="rk-log-clear">${icon("x")}Remove</button><span class="small muted" id="rk-log-status"></span></div>
        <input type="file" id="rk-log-file" accept=".csv,.txt" class="hidden"></details>`;
    this.wire();
    this.syncMode();
    this.syncMotor();
  }

  wire() {
    const box = this.box, R = this.R, u = this.units();
    const on = (id, ev, fn) => $(`#${id}`, box).addEventListener(ev, fn);
    const changed = (geometryToo = false) => { this.ctx.save(); this.refresh(); void geometryToo; };
    $$("#rk-units button", box).forEach((b) => {
      b.classList.toggle("on", b.dataset.v === (R.units || "metric"));
      b.addEventListener("click", () => { R.units = b.dataset.v; this.ctx.save(); this.render(box); this.ctx.onEvaluated(this.ev); });
    });
    $$("#rk-mass-mode button", box).forEach((b) => b.addEventListener("click", () => { R.masses.mode = b.dataset.v; this.syncMode(); changed(); }));
    for (const [id, key] of [["rk-mat-nose", "nose"], ["rk-mat-body", "body"], ["rk-mat-fins", "fins"]]) {
      on(id, "change", (e) => { R.materials[key] = e.target.value; if (key === "fins") { R.fin_material = e.target.value; $("#rk-fin-mat", box).value = e.target.value; } changed(); });
    }
    on("rk-dry-mass", "change", (e) => { R.masses.dry_mass_g = fromIn(u, "massIn", e.target.value); changed(); });
    on("rk-dry-cg", "change", (e) => { const v = fromIn(u, "lenIn", e.target.value); R.masses.dry_cg_mm = v === null ? null : v * 1000; changed(); });
    on("rk-motor", "change", (e) => { R.motor.id = e.target.value; R.motor.delay = null; this.syncMotor(); changed(); });
    on("rk-delay", "change", (e) => { R.motor.delay = e.target.value === "" ? null : +e.target.value; changed(); });
    on("rk-motor-upload", "click", () => $("#rk-motor-file", box).click());
    on("rk-motor-file", "change", async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        const added = await api(`/api/motors?filename=${encodeURIComponent(f.name)}`, { method: "POST", raw: await f.arrayBuffer() });
        this.addMotors(added);
        toast(`${added.length} motor${added.length > 1 ? "s" : ""} added to My motors.`, "good");
      } catch (err) { handleError(err); }
      e.target.value = "";
    });
    on("rk-online-go", "click", () => this.searchOnline());
    on("rk-online-q", "keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.searchOnline(); } });
    on("rk-rail", "change", (e) => { R.launch.rail_m = Math.max(0.1, fromIn(u, "railIn", e.target.value) ?? R.launch.rail_m); changed(); });
    on("rk-angle", "change", (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) R.launch.angle_deg = Math.min(30, Math.max(-30, v)); changed(); });
    on("rk-wind", "change", (e) => { R.launch.wind_m_s = Math.max(0, fromIn(u, "speedIn", e.target.value) ?? 0); changed(); });
    on("rk-elev", "change", (e) => { R.launch.elevation_m = fromIn(u, "altIn", e.target.value) ?? 0; changed(); });
    on("rk-chute-on", "change", (e) => { R.parachute.enabled = e.target.checked; $("#rk-chute-opts", box).classList.toggle("muted", !e.target.checked); changed(); });
    on("rk-chute-d", "change", (e) => { R.parachute.diameter_m = fromIn(u, "chuteIn", e.target.value); changed(); });
    on("rk-chute-cd", "change", (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) R.parachute.cd = Math.min(3, Math.max(0.3, v)); changed(); });
    for (const [id, key] of [["rk-finish", "finish"], ["rk-lug", "launch_lug"], ["rk-edge", "fin_edge"], ["rk-fin-mat", "fin_material"]]) {
      on(id, "change", (e) => { R[key] = e.target.value; if (key === "fin_material") { R.materials.fins = e.target.value; $("#rk-mat-fins", box).value = e.target.value; } changed(); });
    }
    on("rk-min-margin", "change", (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) R.min_margin = Math.min(5, Math.max(0, v)); changed(); });
    on("rk-min-rail", "change", (e) => { R.min_rail_speed = Math.min(60, Math.max(3, fromIn(u, "speedIn", e.target.value) ?? 15)); changed(); });
    on("rk-log-upload", "click", () => $("#rk-log-file", box).click());
    on("rk-log-file", "change", async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        const r = await api(`/api/projects/${this.ctx.pid}/rocket/flightlog?filename=${encodeURIComponent(f.name)}`, { method: "POST", raw: await f.arrayBuffer() });
        toast(`Flight log loaded: apogee ${this.units().alt(r.apogee_m)} at ${sig(r.apogee_t, 3)} s.`, "good");
        this.refresh(true);
      } catch (err) { handleError(err); }
      e.target.value = "";
    });
    on("rk-log-clear", "click", async () => {
      try { await api(`/api/projects/${this.ctx.pid}/rocket/flightlog`, { method: "DELETE" }); this.refresh(true); } catch (err) { handleError(err); }
    });
  }

  syncMode() {
    const mode = this.R.masses.mode || "estimate";
    $$("#rk-mass-mode button", this.box).forEach((b) => b.classList.toggle("on", b.dataset.v === mode));
    $("#rk-mass-estimate", this.box).classList.toggle("hidden", mode !== "estimate");
    $("#rk-mass-dry", this.box).classList.toggle("hidden", mode !== "dry");
    $("#rk-mass-items", this.box).classList.toggle("hidden", mode !== "items");
    if (mode === "items") this.renderItems();
  }

  renderItems() {
    const box = $("#rk-mass-items", this.box), u = this.units(), R = this.R;
    const items = R.masses.items || (R.masses.items = []);
    box.innerHTML = `<table class="tbl compact rk-items"><thead><tr><th>Part</th><th class="r">Mass (${u.massIn[0]})</th><th class="r">From tip (${u.lenIn[0]})</th><th></th></tr></thead><tbody>
      ${items.map((it, i) => `<tr><td><input type="text" data-k="name" data-i="${i}" value="${esc(it.name)}" maxlength="40"></td>
        <td class="r"><input type="number" step="any" data-k="g" data-i="${i}" value="${toIn(u, "massIn", it.g)}"></td>
        <td class="r"><input type="number" step="any" data-k="x" data-i="${i}" value="${toIn(u, "lenIn", it.x_mm / 1000)}"></td>
        <td><button class="icon-btn" type="button" data-del="${i}" aria-label="Remove">${icon("x", "i sm")}</button></td></tr>`).join("")}</tbody></table>
      <div class="row" style="gap:8px;margin-top:6px"><button class="btn sm ghost" type="button" id="rk-item-add">${icon("plus")}Add a part</button>
        <button class="btn sm ghost" type="button" id="rk-item-fill" title="Start from the estimate for this geometry">${icon("sparkles")}Fill from the estimate</button></div>`;
    const commit = () => { this.ctx.save(); this.refresh(); };
    $$("input[data-k]", box).forEach((inp) => inp.addEventListener("change", () => {
      const it = items[+inp.dataset.i];
      if (inp.dataset.k === "name") it.name = inp.value;
      else if (inp.dataset.k === "g") it.g = fromIn(u, "massIn", inp.value) ?? 0;
      else it.x_mm = (fromIn(u, "lenIn", inp.value) ?? 0) * 1000;
      commit();
    }));
    $$("[data-del]", box).forEach((b) => b.addEventListener("click", () => { items.splice(+b.dataset.del, 1); this.renderItems(); commit(); }));
    $("#rk-item-add", box).addEventListener("click", () => { items.push({ name: "Part", g: 10, x_mm: 100 }); this.renderItems(); commit(); });
    $("#rk-item-fill", box).addEventListener("click", () => {
      if (!this.ev) return;
      items.splice(0, items.length, ...this.ev.masses.items.map((it) => ({ name: it.name, g: it.g, x_mm: it.x_mm })));
      this.renderItems();
      commit();
    });
  }

  addMotors(added) {
    for (const m of added) {
      const i = this.motors.findIndex((q) => q.id === m.id);
      if (i >= 0) this.motors[i] = m; else { this.motors.push(m); (this.ctx.catalog.user_motors = this.ctx.catalog.user_motors || []).push(m); }
    }
    if (added.length) { this.R.motor.id = added[0].id; this.R.motor.delay = null; }
    this.render(this.box);
    this.ctx.save();
    this.refresh();
  }

  syncMotor(inUse = null) {
    const box = this.box, R = this.R;
    let m = this.motor(R.motor.id);
    if (!m) { R.motor.id = null; m = this.motor(inUse) || this.motor(this.ev && this.ev.motor.id) || this.motors.find((q) => q.id === "estes-c6"); }
    const sel = $("#rk-motor", box);
    if (m) sel.value = m.id;
    const ds = $("#rk-delay", box);
    ds.innerHTML = `<option value="">Optimal (at apogee)</option>` + (m ? m.delays.map((d) => `<option value="${d}">${d} s</option>`).join("") : "");
    ds.value = R.motor.delay === null || R.motor.delay === undefined ? "" : String(R.motor.delay);
    $("#rk-motor-info", box).innerHTML = m ? `<span class="k">Class · impulse</span><span class="v">${esc(m.class)} · ${sig(m.total_impulse, 3)} N·s</span>
      <span class="k">Average / peak thrust</span><span class="v">${sig(m.avg_thrust, 3)} / ${sig(m.peak_thrust, 3)} N</span>
      <span class="k">Burn · mass</span><span class="v">${sig(m.burn_time, 3)} s · ${sig(m.total_mass_g, 3)} g (${sig(m.prop_mass_g, 3)} g propellant)</span>
      ${m.note ? `<span class="k" style="grid-column:1/-1;color:var(--warning-text)">${esc(m.note)}</span>` : ""}` : "";
  }

  async searchOnline() {
    const box = this.box, q = $("#rk-online-q", box).value.trim();
    const out = $("#rk-online", box);
    if (!q) return;
    out.innerHTML = `<p class="small muted">Searching…</p>`;
    try {
      const rows = await api(`/api/motors/online?q=${encodeURIComponent(q)}`);
      out.innerHTML = rows.length ? `<table class="tbl compact"><tbody>${rows.map((r) => `<tr><td>${esc(r.manufacturer)} <b>${esc(r.name)}</b> <span class="muted small">${esc(r.designation || "")}</span></td>
        <td class="r small">${r.diameter_mm} mm · ${sig(r.total_impulse || 0, 3)} N·s</td><td><button class="btn sm" type="button" data-tc="${esc(r.tc_id)}">Add</button></td></tr>`).join("")}</tbody></table>`
        : `<p class="small muted">Nothing found. Try the designation alone (C6, H128) or a manufacturer.</p>`;
      $$("[data-tc]", out).forEach((b) => b.addEventListener("click", async () => {
        b.disabled = true;
        try { this.addMotors([await api("/api/motors/online/add", { method: "POST", body: { tc_id: b.dataset.tc } })]); toast("Certified curve added to My motors.", "good"); }
        catch (e) { handleError(e); b.disabled = false; }
      }));
    } catch (e) { out.innerHTML = `<p class="small" style="color:var(--critical-text)">${esc(e.message || "ThrustCurve.org could not be reached.")}</p>`; }
  }

  // the server computes the flight; debounced so slider drags cost one request
  refresh(now = false) {
    clearTimeout(this.timer);
    const run = async () => {
      const seq = ++this.seq;
      try {
        const ev = await api(`/api/projects/${this.ctx.pid}/rocket`, { method: "POST", body: { values: this.ctx.values(), rocket: this.R } });
        if (seq !== this.seq) return;
        this.ev = ev;
        if (!this.R.motor.id) this.syncMotor(ev.motor.id);
        const st = $("#rk-log-status", this.box);
        if (st) st.textContent = ev.flight_log ? `${ev.flight_log.name}: apogee ${this.units().alt(ev.flight_log.apogee_m)}` : "";
        const clr = $("#rk-log-clear", this.box);
        if (clr) clr.classList.toggle("hidden", !ev.flight_log);
        this.ctx.onEvaluated(ev);
      } catch (e) { if (seq === this.seq) handleError(e); }
    };
    if (now) run(); else this.timer = setTimeout(run, 350);
  }
}

// ---- the Flight tab ---------------------------------------------------------------------------------------
const kpi = (label, value, sub = "") => `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
const col = (ok, warn) => (ok ? (warn ? "var(--warning-text)" : "var(--good-text)") : "var(--critical-text)");

function stabilitySVG(ev) {
  const s = ev.silhouette, st = ev.stability;
  if (!s || !s.body.length) return "";
  const L = s.length, R = s.radius;
  const span = s.fin ? Math.max(...s.fin.map((p) => p[1])) : R;
  const W = 340, pad = 14;
  const k = (W - 2 * pad) / L;
  const Hh = span * k + 10;
  const Hgt = 2 * Hh + 2 * pad + 18;
  const X = (x) => pad + x * k, Y = (r) => pad + Hh + 10 - r * k, Y2 = (r) => pad + Hh + 10 + r * k;
  const body = s.body.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).concat([...s.body].reverse().map((p) => `${X(p[0]).toFixed(1)},${Y2(p[1]).toFixed(1)}`)).join(" ");
  const fin = s.fin ? [`<polygon points="${s.fin.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="var(--surface-3)" stroke="var(--text-2)" stroke-width="1"/>`,
    `<polygon points="${s.fin.map((p) => `${X(p[0]).toFixed(1)},${Y2(p[1]).toFixed(1)}`).join(" ")}" fill="var(--surface-3)" stroke="var(--text-2)" stroke-width="1"/>`].join("") : "";
  const ym = pad + Hh + 10;
  const cp = st.liftoff.x_cp, cg0 = st.liftoff.x_cg, cg1 = st.burnout.x_cg;
  return `<svg viewBox="0 0 ${W} ${Hgt}" width="100%" role="img" aria-label="Side view with the centre of pressure and gravity">
    ${fin}<polygon points="${body}" fill="var(--surface-2)" stroke="var(--text-2)" stroke-width="1.2"/>
    <line x1="${X(cg1)}" y1="${ym - 6}" x2="${X(cg1)}" y2="${ym + 6}" stroke="var(--series-1)" stroke-width="1.2"/>
    <circle cx="${X(cg0)}" cy="${ym}" r="6" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="1.5"><title>Centre of gravity at liftoff</title></circle>
    <circle cx="${X(cg1)}" cy="${ym}" r="6" fill="none" stroke="var(--series-1)" stroke-width="1.8"><title>Centre of gravity at burnout</title></circle>
    <circle cx="${X(cp)}" cy="${ym}" r="6" fill="var(--critical-text)" stroke="var(--surface-1)" stroke-width="1.5"><title>Centre of pressure (Barrowman)</title></circle>
    <text x="${X(cg0)}" y="${Hgt - 4}" text-anchor="middle" class="ax">CG</text><text x="${X(cp)}" y="${Hgt - 4}" text-anchor="middle" class="ax" fill="var(--critical-text)">CP</text>
    <text x="${pad}" y="${pad - 2}" class="ax">airflow →</text></svg>`;
}

export function flightTabHTML(ev, unitsKey) {
  if (!ev) return `<p class="small muted">Computing the flight…</p>`;
  const u = UNITS[unitsKey] || UNITS.metric;
  const fl = ev.flight, st = ev.stability, fu = ev.flutter, m = ev.motor;
  const bad = ev.checks.filter((c) => !c.ok), warns = ev.checks.filter((c) => c.ok && c.warn);
  const delayTxt = fl.optimal_delay === null ? "—" : `${sig(fl.optimal_delay, 2)} s`;
  const nearest = m.delays && m.delays.length && fl.optimal_delay !== null ? m.delays.reduce((b, d) => (Math.abs(d - fl.optimal_delay) < Math.abs(b - fl.optimal_delay) ? d : b), m.delays[0]) : null;
  const cdMax = ev.drag_curve.reduce((b, r) => (Math.abs(r.mach - fl.max_mach) < Math.abs(b.mach - fl.max_mach) ? r : b), ev.drag_curve[0]);
  return `
    <div class="notice ${bad.length ? "bad" : "good"}">${icon(bad.length ? "alert" : "check")}<div><b style="color:var(--text)">${bad.length ? "Not ready to fly" : "Ready to fly"}</b>
      ${bad.length ? `: ${bad.map((c) => esc(c.why)).join("; ")}.` : ` on the ${esc(m.manufacturer)} ${esc(m.name)}: stable, the fins hold, and it leaves the rail fast enough.`}
      ${warns.length ? `<div class="small" style="margin-top:4px">${warns.map((c) => esc(c.why)).join("; ")}.</div>` : ""}</div></div>
    <div class="kpis" style="margin-top:12px">
      ${kpi("Apogee", u.alt(fl.apogee_m), `in ${sig(fl.apogee_t, 3)} s · ${ev.flight_log ? `measured ${u.alt(ev.flight_log.apogee_m)}` : `${ev.launch.wind_m_s ? u.speed(ev.launch.wind_m_s) + " wind" : "no wind"}`}`)}
      ${kpi("Top speed", u.speed(fl.max_speed), `Mach ${sig(fl.max_mach, 3)} · ${u.accel(fl.max_acceleration_g)} peak`)}
      ${kpi("Static margin", `<span style="color:${col(st.liftoff.margin >= ev.settings.min_margin)}">${sig(st.liftoff.margin, 3)} cal</span>`, `at liftoff · ${sig(st.max_speed.margin, 3)} at top speed (${esc(st.liftoff.rating)})`)}
      ${kpi("Off the rail", `<span style="color:${col(fl.rail_exit_speed >= 15)}">${u.speed(fl.rail_exit_speed)}</span>`, `${sig(ev.launch.rail_m, 3)} m rail · thrust ${sig(fl.thrust_to_weight, 3)}× weight`)}
      ${kpi("Ejection delay", delayTxt, nearest !== null ? `optimal · nearest ${nearest} s${ev.launch.delay_s !== null && ev.launch.delay_s !== undefined ? ` · using ${ev.launch.delay_s} s` : ""}` : "optimal")}
      ${fu ? kpi("Fin flutter", `<span style="color:${col(fu.ok)}">${u.speed(fu.flutter_speed)}</span>`, `${esc(fu.material_label)} · ${sig(fu.ratio, 3)}× the top speed`) : ""}
      ${kpi("Descent", fl.descent_time === null ? "ballistic" : `${sig(fl.descent_time, 3)} s`, fl.drift_m === null ? "no parachute" : `lands ${u.dist(Math.abs(fl.drift_m))} ${fl.drift_m >= 0 ? "downwind" : "upwind"}${fl.landing ? ` at ${u.speed(fl.landing.v)}` : ""}`)}
      ${kpi("Liftoff mass", u.mass(fl.liftoff_mass * 1000), `${u.mass(ev.masses.dry_mass_g)} dry + motor`)}
    </div>
    <div class="section-title">Stability <span class="small muted">Barrowman · CP ${u.len(st.liftoff.x_cp)} · CG ${u.len(st.liftoff.x_cg)} from the tip</span></div>
    <div class="rk-sil">${stabilitySVG(ev)}</div>
    <p class="small muted" style="margin:4px 0 0">Filled blue: centre of gravity with a full motor; hollow: at burnout. Red: centre of pressure at small angles.
      Through the flight the margin goes from ${sig(st.liftoff.margin, 3)} to ${sig(st.burnout.margin, 3)} calibres${fl.max_mach > 0.9 ? " (the fins' centre moves aft near Mach 1)" : ""}.</p>
    <div class="section-title">Checks</div>
    <ul class="rk-checks">${ev.checks.map((c) => `<li class="${c.ok ? (c.warn ? "warn" : "ok") : "bad"}">${icon(c.ok ? (c.warn ? "alert" : "check") : "x")}<div><b>${esc(c.label)}</b> — ${esc(c.why)}</div></li>`).join("")}</ul>
    <div class="section-title">Flight</div>
    <div class="charts" id="rk-charts"></div>
    <table class="tbl compact"><thead><tr><th>Event</th><th class="r">Time</th><th class="r">Altitude</th><th class="r">Speed</th></tr></thead><tbody>
      ${[["Leaves the rail", fl.rail_exit_t, null, fl.rail_exit_speed], ["Burnout", fl.burnout && fl.burnout.t, fl.burnout && fl.burnout.z, fl.burnout && fl.burnout.v],
        ["Apogee", fl.apogee_t, fl.apogee_m, fl.speed_at_apogee], ["Parachute opens", fl.deploy && fl.deploy.t, fl.deploy && fl.deploy.z, fl.deploy_speed],
        ["Landing", fl.landing && fl.landing.t, 0, fl.landing && fl.landing.v]]
        .filter((r) => r[1] !== null && r[1] !== undefined && r[1] !== false)
        .map((r) => `<tr><td>${r[0]}</td><td class="r">${sig(r[1], 3)} s</td><td class="r">${r[2] === null || r[2] === undefined ? "—" : u.alt(r[2])}</td><td class="r">${r[3] === null || r[3] === undefined ? "—" : u.speed(r[3])}</td></tr>`).join("")}
    </tbody></table>
    <div class="section-title">Drag <span class="small muted">Cd ${sig(cdMax.cd, 3)} at Mach ${sig(cdMax.mach, 2)} on the body cross-section</span></div>
    <div class="split-bar" role="img" aria-label="drag breakdown">
      <div style="width:${(100 * cdMax.cd_friction) / cdMax.cd}%;background:var(--series-2)"></div>
      <div style="width:${(100 * (cdMax.cd_nose + cdMax.cd_tail)) / cdMax.cd}%;background:var(--series-1)"></div>
      <div style="width:${(100 * cdMax.cd_base) / cdMax.cd}%;background:var(--heat)"></div>
      <div style="width:${(100 * (cdMax.cd_fin_le + cdMax.cd_fin_te + cdMax.cd_lug)) / cdMax.cd}%;background:var(--series-3)"></div></div>
    <div class="legend" style="margin-top:8px"><span><span class="sw" style="background:var(--series-2)"></span>Friction ${sig(cdMax.cd_friction, 3)}${cdMax.rough_limited ? " (finish-limited)" : ""}</span>
      <span><span class="sw" style="background:var(--series-1)"></span>Nose + tail ${sig(cdMax.cd_nose + cdMax.cd_tail, 3)}</span>
      <span><span class="sw" style="background:var(--heat)"></span>Base ${sig(cdMax.cd_base, 3)}</span>
      <span><span class="sw" style="background:var(--series-3)"></span>Fins + lug ${sig(cdMax.cd_fin_le + cdMax.cd_fin_te + cdMax.cd_lug, 3)}</span></div>
    <div class="section-title">Masses <span class="small muted">${ev.masses.mode === "estimate" ? "estimated from the geometry" : ev.masses.mode === "dry" ? "measured" : "part by part"}</span></div>
    <table class="tbl compact"><thead><tr><th>Part</th><th class="r">Mass</th><th class="r">From tip</th></tr></thead><tbody>
      ${ev.masses.items.map((it) => `<tr><td>${esc(it.name)}</td><td class="r">${u.mass(it.g)}</td><td class="r">${u.len(it.x_mm / 1000)}</td></tr>`).join("")}
      <tr><td><b>Dry mass</b></td><td class="r"><b>${u.mass(ev.masses.dry_mass_g)}</b></td><td class="r">CG ${u.len(ev.masses.dry_cg_mm / 1000)}</td></tr>
      <tr><td>${esc(m.manufacturer)} ${esc(m.name)} motor</td><td class="r">${u.mass(m.total_mass_g)}</td><td class="r">${u.len(ev.geometry.length - m.length_mm / 2000)}</td></tr></tbody></table>
    ${ev.import_notes && ev.import_notes.length ? `<div class="notice" style="margin-top:12px">${icon("info")}<div class="small"><b style="color:var(--text)">Imported design.</b> ${ev.import_notes.map(esc).join("; ")}.</div></div>` : ""}
    <div class="row wrap" style="margin-top:12px;gap:8px"><a class="btn sm ghost" href="/api/projects/${location.pathname.split("/").pop()}/rocket/drag.csv" download>${icon("download")}Drag table (CSV)</a></div>
    <div class="notice" style="margin-top:12px">${icon("info")}<div class="small">Drag build-up (Hoerner, Barrowman, OpenRocket documentation), Barrowman centre of pressure with Mach corrections, NACA TN 4197
      flutter, point-mass flight on a rail with wind. Built-in motor curves are approximate: use the manufacturer's .eng file for the real thing. Checked against published data on the
      <a href="/validation#rocket" target="_blank">validation page</a>. Fly within the <a href="https://www.nar.org/safety-information/model-rocket-safety-code/" target="_blank" rel="noopener">NAR</a> /
      <a href="https://www.tripoli.org/safetycode" target="_blank" rel="noopener">Tripoli</a> safety codes: this is a simulation, not a safety certification.</div></div>`;
}

export function wireFlightTab(box, ev, unitsKey) {
  const holder = box.querySelector("#rk-charts");
  if (!holder || !ev) return;
  const u = UNITS[unitsKey] || UNITS.metric;
  holder.innerHTML = "";
  const card = (title, sub) => { const c = h("div", { class: "chart-card" }); c.innerHTML = `<div class="ct">${title}<span>${sub}</span></div>`; const a = h("div"); c.append(a); holder.append(c); return a; };
  const tr = ev.flight.trajectory || [];
  const alt = tr.map((p) => [p.t, unitsKey === "imperial" ? p.z / 0.3048 : p.z]);
  const altSer = [{ name: "Simulated", color: "var(--series-1)", points: alt }];
  if (ev.flight_log) altSer.push({ name: "Measured", color: "var(--series-3)", points: ev.flight_log.points.map((p) => [p[0], unitsKey === "imperial" ? p[1] / 0.3048 : p[1]]), dashed: true });
  lineChart(card("Altitude", unitsKey === "imperial" ? "ft" : "m"), altSer, { height: 150, xName: "t", xLabel: "time (s)", fmtY: (v) => fmtInt(Math.round(v)) });
  lineChart(card("Speed", unitsKey === "imperial" ? "mph" : "m/s"), [{ name: "Speed", color: "var(--series-2)", points: tr.map((p) => [p.t, unitsKey === "imperial" ? p.v * 2.23694 : p.v]) }],
    { height: 140, xName: "t", xLabel: "time (s)", fmtY: (v) => sig(v, 3) });
  const dc = ev.drag_curve;
  lineChart(card("Drag coefficient vs Mach", "total · friction · base"), [
    { name: "Total", color: "var(--series-1)", points: dc.map((r) => [r.mach, r.cd]), dots: true },
    { name: "Friction", color: "var(--series-2)", points: dc.map((r) => [r.mach, r.cd_friction]), faded: true },
    { name: "Base", color: "var(--heat)", points: dc.map((r) => [r.mach, r.cd_base]), faded: true }],
  { height: 150, xName: "M", xLabel: "Mach number", fmtY: (v) => sig(v, 3), highlight: { x: ev.flight.max_mach, y: ev.drag_curve.reduce((b, r) => (Math.abs(r.mach - ev.flight.max_mach) < Math.abs(b.mach - ev.flight.max_mach) ? r : b), dc[0]).cd } });
  const cpm = ev.stability.cp_vs_mach || [];
  if (cpm.length > 2) {
    const cg = ev.stability.burnout.x_cg, d = ev.geometry.diameter;
    lineChart(card("Static margin vs Mach", "calibres, at burnout mass"), [{ name: "Margin", color: "var(--series-3)", points: cpm.map((r) => [r.mach, (r.x_cp - cg) / d]), dots: true },
      { name: "minimum", color: "var(--warning)", points: [[cpm[0].mach, ev.settings.min_margin], [cpm[cpm.length - 1].mach, ev.settings.min_margin]], dashed: true }],
    { height: 130, xName: "M", xLabel: "Mach number", fmtY: (v) => sig(v, 2) });
  }
}

// ---- optimization results ------------------------------------------------------------------------------
export function rocketHistoryHTML(r, unitsKey) {
  const rk = r.rocket;
  if (!rk) return "";
  const u = UNITS[unitsKey] || UNITS.metric;
  const b = rk.base, o = rk.best;
  const tr = (label, a, c, f, lowerBetter = false) => {
    const d = Number.isFinite(a) && Number.isFinite(c) && Math.abs(a) > 1e-9 ? (c - a) / Math.abs(a) : NaN;
    const ok = lowerBetter ? d < 0 : d > 0;
    return `<tr><td>${label}</td><td class="r">${f(a)}</td><td class="r"><b>${f(c)}</b></td><td class="r ${!Number.isFinite(d) || Math.abs(d) < 0.0005 ? "" : ok ? "delta good" : "delta bad"}">${Number.isFinite(d) ? `${d > 0 ? "+" : ""}${(100 * d).toFixed(1)} %` : "—"}</td></tr>`;
  };
  return `<div class="section-title">Flight: original → optimized</div>
    <table class="tbl compact"><thead><tr><th></th><th class="r">Original</th><th class="r">Optimized</th><th class="r">Change</th></tr></thead><tbody>
      ${tr("Apogee", b.flight.apogee_m, o.flight.apogee_m, u.alt)}
      ${tr("Top speed", b.flight.max_speed, o.flight.max_speed, u.speed)}
      ${tr("Cd at top speed", b.drag_curve.reduce((q, x) => (Math.abs(x.mach - b.flight.max_mach) < Math.abs(q.mach - b.flight.max_mach) ? x : q), b.drag_curve[0]).cd,
        o.drag_curve.reduce((q, x) => (Math.abs(x.mach - o.flight.max_mach) < Math.abs(q.mach - o.flight.max_mach) ? x : q), o.drag_curve[0]).cd, (v) => sig(v, 3), true)}
      ${tr("Static margin at liftoff", b.stability.liftoff.margin, o.stability.liftoff.margin, (v) => `${sig(v, 3)} cal`)}
      ${tr("Dry mass", b.masses.dry_mass_g, o.masses.dry_mass_g, u.mass, true)}
      ${tr("Off the rail", b.flight.rail_exit_speed, o.flight.rail_exit_speed, u.speed)}
      ${b.flutter && o.flutter ? tr("Flutter speed / top speed", b.flutter.ratio, o.flutter.ratio, (v) => `${sig(v, 3)}×`) : ""}
    </tbody></table>
    <p class="small muted" style="margin-top:6px">Every design the optimizer tried was checked for static margin (≥ ${sig(o.settings.min_margin, 2)} cal), fin flutter and rail-exit speed before it was scored; the
      masses follow the geometry, so a longer nose or bigger fins pay their weight.</p>`;
}
