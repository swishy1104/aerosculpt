// RC tunnel: propellers (family, size, blades, rpm), placing and moving them, live performance
// estimates, and the propulsion side of the results.
import { api, esc, fmtForce, fmtPct, fmtPower, h, icon, sig, upgradePrompt } from "/static/js/core.js";
import { applyMat3, transposeMat3 } from "/static/js/viewer.js";
import { lineChart } from "/static/js/charts.js";

const IN = 0.0254;
export const PROP_PRESETS = {
  "thin-electric": [
    { label: "7×4 · 14,000 rpm · park flyer", d: 7, p: 4, rpm: 14000, b: 2 },
    { label: "9×6 · 10,000 rpm · sport", d: 9, p: 6, rpm: 10000, b: 2 },
    { label: "10×7 · 9,000 rpm · sport / UAV", d: 10, p: 7, rpm: 9000, b: 2 },
    { label: "11×7 · 8,000 rpm · trainer / mapping", d: 11, p: 7, rpm: 8000, b: 2 },
    { label: "13×8 · 6,500 rpm · long endurance", d: 13, p: 8, rpm: 6500, b: 2 },
    { label: "16×10 · 5,000 rpm · large UAV", d: 16, p: 10, rpm: 5000, b: 2 },
    { label: "20×10 · 3,800 rpm · heavy lift", d: 20, p: 10, rpm: 3800, b: 2 },
  ],
  wide: [
    { label: "5×3.75 · 3 blades · 20,000 rpm · FPV", d: 5, p: 3.75, rpm: 20000, b: 3 },
    { label: "9×6.75 · 3 blades · 8,000 rpm · scale", d: 9, p: 6.75, rpm: 8000, b: 3 },
    { label: "10×6 · 4 blades · 7,500 rpm · warbird", d: 10, p: 6, rpm: 7500, b: 4 },
    { label: "12×8 · 3 blades · 6,000 rpm · scale", d: 12, p: 8, rpm: 6000, b: 3 },
    { label: "15×10 · 2 blades · 5,000 rpm · glow sport", d: 15, p: 10, rpm: 5000, b: 2 },
  ],
};
const CHORD = { "thin-electric": 0.11, wide: 0.2 };          // drawn blade width / radius
const STEPS = [0.001, 0.005, 0.01, 0.05];                      // nudge steps, metres

// ---- geometry helpers on the body-frame triangle soup ---------------------------------------
function extentsX(body) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < body.length; i += 3) { const x = body[i]; if (x < lo) lo = x; if (x > hi) hi = x; }
  return [lo, hi];
}
/** Centre (y, z) of the body's cross-section near station x0; with `near` = [y, z, radius] only
 * the part around that point counts (a nacelle rather than the whole wing). */
function sliceCentre(body, x0, near = null) {
  const [lo, hi] = extentsX(body);
  const L = hi - lo;
  for (const frac of [0.008, 0.02, 0.05, 0.1]) {
    const d = frac * L;
    let ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity, n = 0;
    for (let i = 0; i < body.length; i += 3) {
      if (Math.abs(body[i] - x0) > d) continue;
      const y = body[i + 1], z = body[i + 2];
      if (near && Math.hypot(y - near[0], z - near[1]) > near[2]) continue;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y; if (z < zmin) zmin = z; if (z > zmax) zmax = z;
      n++;
    }
    if (n >= 9) return [(ymin + ymax) / 2, (zmin + zmax) / 2];
  }
  return null;
}

export class Propellers {
  /** ctx: { S, me, catalog, frame(), save(), changed(), startPick(id, kind), displayRotation(), viewer, round, speeds(), preview(), showing() } */
  constructor(ctx) {
    this.ctx = ctx;
    this.selected = null;
    this.moving = null;
    this.step = 0.005;
    this.snapOnPlace = true;
    this.estimates = {};
    ctx.S.propellers = ctx.S.propellers || [];
    for (const p of ctx.S.propellers) { p.family = p.family || "thin-electric"; p.blades = p.blades || 2; }
    const add = document.getElementById("add-prop");
    add.innerHTML = `${icon("plus")}Add propeller`;
    add.addEventListener("click", () => this.add());
    addEventListener("keydown", (e) => { if (e.key === "Escape" && this.moving) this.stopMove(); });
  }

  get list() { return this.ctx.S.propellers; }
  frame() { return this.ctx.frame(); }

  // ---- placement --------------------------------------------------------------------------------
  noseHub(p) {
    const f = this.frame();
    const [lo, hi] = extentsX(f.body);
    const c = sliceCentre(f.body, lo + 0.004 * (hi - lo)) || [0, 0];
    return [lo - 0.03 * (hi - lo), c[0], c[1]];
  }
  tailHub(p) {
    const f = this.frame();
    const [lo, hi] = extentsX(f.body);
    const c = sliceCentre(f.body, hi - 0.004 * (hi - lo)) || [0, 0];
    return [hi + 0.03 * (hi - lo), c[0], c[1]];
  }
  /** Same station, hub moved onto the centre line of the part it sits on. */
  centred(p, bodyPoint) {
    const f = this.frame();
    const [lo, hi] = extentsX(f.body);
    const L = hi - lo;
    const x0 = Math.min(hi - 0.004 * L, Math.max(lo + 0.004 * L, bodyPoint[0]));
    const radius = Math.max((p.diameter_in * IN) / 2, 0.08 * L);
    const c = sliceCentre(f.body, x0, [bodyPoint[1], bodyPoint[2], radius]) || sliceCentre(f.body, x0);
    return c ? [bodyPoint[0], c[0], c[1]] : bodyPoint;
  }
  setBody(p, b) {
    p.center = this.frame().bodyToFile(b).map(this.ctx.round);
  }

  add() {
    const { me } = this.ctx;
    if (this.list.length >= me.limits.max_propellers) {
      return upgradePrompt(`${me.plan_name} simulates up to ${me.limits.max_propellers} propeller${me.limits.max_propellers > 1 ? "s" : ""}.`);
    }
    const pre = PROP_PRESETS["thin-electric"][3];
    const p = { id: Math.random().toString(36).slice(2, 10), name: `Propeller ${this.list.length + 1}`, center: [0, 0, 0],
      diameter_in: pre.d, pitch_in: pre.p, rpm: pre.rpm, blades: pre.b, rotation: "cw", width: 1.0, family: "thin-electric" };
    this.setBody(p, this.list.length ? this.tailHub(p) : this.noseHub(p));
    this.list.push(p);
    this.selected = p.id;
    this.render();
    this.ctx.changed();
  }

  /** A click on the model while placing: hub there, optionally snapped onto the local centre line. */
  placeAt(id, bodyPoint) {
    const p = this.list.find((q) => q.id === id);
    if (!p) return null;
    this.setBody(p, this.snapOnPlace ? this.centred(p, bodyPoint) : bodyPoint);
    this.selected = id;
    this.render();
    this.ctx.changed();
    return p;
  }

  // ---- moving -------------------------------------------------------------------------------------
  startMove(p) {
    if (this.ctx.showing()) this.ctx.preview();
    this.moving = p.id;
    this.selected = p.id;
    this.render3D();
    const R = this.ctx.displayRotation();
    const Rt = transposeMat3(R);
    const ok = this.ctx.viewer.attachMover(p.id, (pos) => {
      p.center = this.frame().bodyToFile(applyMat3(Rt, pos)).map(this.ctx.round);
      this.syncInputs(p);
    }, () => { this.ctx.save(); this.refreshEstimates(); });
    if (!ok) this.moving = null;
    this.render();
  }
  stopMove() {
    this.moving = null;
    this.ctx.viewer.detachMover();
    this.render();
  }
  nudge(p, axis, sign) {
    const s = this.frame().s;                       // file units per metre is 1/s
    p.center[axis] = this.ctx.round(p.center[axis] + (sign * this.step) / s);
    this.syncInputs(p);
    this.render3D();
    this.ctx.save();
  }
  syncInputs(p) {
    const card = document.querySelector(`#props [data-id="${p.id}"]`);
    if (card) card.querySelectorAll("[data-c]").forEach((inp) => { inp.value = p.center[+inp.dataset.c]; });
  }

  // ---- live estimates -------------------------------------------------------------------------------
  refreshEstimates() {
    clearTimeout(this._est);
    this._est = setTimeout(async () => {
      const speeds = this.ctx.speeds().filter((v) => v >= 1);
      if (!this.list.length || !speeds.length) return;
      try {
        const r = await api("/api/propellers/estimate", { method: "POST", body: { propellers: this.list, speeds, altitude: this.ctx.S.envelope.altitude || 0 } });
        this.estimates = Object.fromEntries(r.propellers.map((q) => [q.id, q.rows]));
        this.renderEstimates();
      } catch (e) { /* the estimate is a convenience; the run validates everything */ }
    }, 350);
  }
  renderEstimates() {
    for (const p of this.list) {
      const box = document.querySelector(`#props [data-id="${p.id}"] .prop-est`);
      const rows = this.estimates[p.id];
      if (!box || !rows) continue;
      const pick = rows.length > 3 ? [rows[0], rows[Math.floor(rows.length / 2)], rows[rows.length - 1]] : rows;
      box.innerHTML = `<table class="tbl compact"><thead><tr><th class="r">V</th><th class="r">Thrust</th><th class="r">Power</th><th class="r">Eff.</th><th class="r">Slipstream</th></tr></thead><tbody>
        ${pick.map((r) => `<tr><td class="r">${sig(r.v, 3)} m/s</td><td class="r" style="color:var(${r.thrust_N <= 0 ? "--critical-text" : "--text"})">${fmtForce(r.thrust_N)}</td>
          <td class="r">${fmtPower(r.power_W)}</td><td class="r">${r.thrust_N > 0 ? fmtPct(r.eta, 0) : "—"}</td><td class="r">${sig(r.slipstream, 3)}×</td></tr>`).join("")}</tbody></table>
        <p class="small muted" style="margin:4px 0 0">Blade-element estimate in free air; the RC tunnel run adds the airframe. Slipstream = far-wake speed ÷ airspeed.</p>`;
    }
    const cmp = document.getElementById("prop-compare");
    if (this.list.length < 2) { cmp.innerHTML = ""; return; }
    const mid = (rows) => rows[Math.floor(rows.length / 2)];
    const ready = this.list.filter((p) => this.estimates[p.id]);
    if (!ready.length) return;
    const v = mid(this.estimates[ready[0].id]).v;
    cmp.innerHTML = `<div class="section-title" style="margin-top:6px">Compare at ${sig(v, 3)} m/s</div>
      <table class="tbl compact"><thead><tr><th>Propeller</th><th class="r">Thrust</th><th class="r">Power</th><th class="r">Eff.</th><th class="r">Slip</th></tr></thead><tbody>
      ${ready.map((p) => { const r = mid(this.estimates[p.id]); return `<tr><td>${esc(p.name)}<div class="small muted">${sig(p.diameter_in, 3)}×${sig(p.pitch_in, 3)} · ${p.blades} bl</div></td>
        <td class="r">${fmtForce(r.thrust_N)}</td><td class="r">${fmtPower(r.power_W)}</td><td class="r">${r.thrust_N > 0 ? fmtPct(r.eta, 0) : "—"}</td><td class="r">${sig(r.slipstream, 3)}×</td></tr>`; }).join("")}
      </tbody></table>`;
  }

  // ---- cards -----------------------------------------------------------------------------------------
  render() {
    const box = document.getElementById("props");
    box.innerHTML = "";
    const units = this.ctx.S.units;
    const fams = this.ctx.catalog.prop_families;
    for (const p of this.list) {
      const tipMach = (Math.PI * p.diameter_in * IN * (p.rpm / 60)) / 340.3;
      const presets = PROP_PRESETS[p.family] || [];
      const moving = this.moving === p.id;
      const card = h("div", { class: "zone prop" + (p.id === this.selected ? " sel" : ""), style: { "--zc": "var(--rc)" }, dataset: { id: p.id } });
      card.innerHTML = `<div class="zh"><span class="zone-type">Prop</span>
          <input type="text" value="${esc(p.name)}" maxlength="40" aria-label="Propeller name">
          <button class="icon-btn" data-dup title="Duplicate (for twins: mirror it with Y)">${icon("plus", "i sm")}</button>
          <button class="icon-btn" data-del title="Remove">${icon("trash", "i sm")}</button></div>
        <div class="prop-sec">Propeller</div>
        <div class="seg prop-fam" data-fam>${Object.entries(fams).map(([k, f]) => `<button type="button" data-f="${k}" class="${p.family === k ? "on" : ""}" title="${esc(f.note)}">${esc(f.name.split(" (")[0])}</button>`).join("")}</div>
        <label class="field" style="margin:8px 0 6px"><span>Preset</span><select data-preset>
          <option value="">Custom</option>${presets.map((q, i) => `<option value="${i}" ${q.d === p.diameter_in && q.p === p.pitch_in && q.rpm === p.rpm && q.b === p.blades ? "selected" : ""}>${esc(q.label)}</option>`).join("")}</select></label>
        <div class="prop-grid">
          <label><span>Diameter</span><div class="input-unit"><input type="number" data-k="diameter_in" min="2" max="40" step="0.5" value="${p.diameter_in}"><span class="unit">in</span></div></label>
          <label><span>Pitch</span><div class="input-unit"><input type="number" data-k="pitch_in" min="1" max="30" step="0.25" value="${p.pitch_in}"><span class="unit">in</span></div></label>
          <label><span>Speed</span><div class="input-unit"><input type="number" data-k="rpm" min="500" max="40000" step="100" value="${p.rpm}"><span class="unit">rpm</span></div></label>
        </div>
        <div class="row" style="gap:8px;margin-top:8px;flex-wrap:wrap">
          <div class="seg" data-blades title="Number of blades">${[2, 3, 4, 5, 6].map((b) => `<button type="button" data-b="${b}" class="${p.blades === b ? "on" : ""}">${b}</button>`).join("")}</div>
          <span class="small muted">blades</span>
          <div class="seg" data-rot title="Seen from behind, looking forward"><button type="button" data-r="cw" class="${p.rotation === "cw" ? "on" : ""}">CW</button><button type="button" data-r="ccw" class="${p.rotation === "ccw" ? "on" : ""}">CCW</button></div>
        </div>
        <p class="small muted" style="margin:6px 0 0">${esc(fams[p.family] ? fams[p.family].note : "")} · ${sig(p.diameter_in * 25.4, 3)} mm · tip Mach ${tipMach.toFixed(2)}${tipMach > 0.7 ? ` <span style="color:var(--warning-text)">(high: noise and losses)</span>` : ""}${p.blades > 4 ? ` · <span style="color:var(--warning-text)">5–6 blades: beyond the measured 2–4</span>` : ""}</p>
        <div class="prop-est small muted" style="margin-top:8px">Estimating…</div>
        <div class="prop-sec">Position <span class="muted" style="font-weight:500">· hub, ${units}</span></div>
        <div class="zgrid"><span></span><span class="ax">X</span><span class="ax">Y</span><span class="ax">Z</span>
          <span>Hub</span>${p.center.map((v, i) => `<input type="number" step="any" data-c="${i}" value="${v}">`).join("")}
          <span>Nudge</span>${[0, 1, 2].map((i) => `<div class="nudge"><button type="button" data-n="${i}" data-s="-1" title="Move −${"XYZ"[i]}">−</button><button type="button" data-n="${i}" data-s="1" title="Move +${"XYZ"[i]}">+</button></div>`).join("")}</div>
        <div class="row" style="gap:6px;margin-top:8px;flex-wrap:wrap">
          <button type="button" class="btn sm ${moving ? "primary" : ""}" data-move>${icon("crosshair", "i sm")}${moving ? "Done moving" : "Move in 3D"}</button>
          <button type="button" class="btn sm" data-pick title="Click the model to place the hub">${icon("target", "i sm")}Click to place</button>
          <label class="small muted step-pick">step <select data-step>${STEPS.map((s) => `<option value="${s}" ${s === this.step ? "selected" : ""}>${s >= 0.01 ? s * 100 + " cm" : s * 1000 + " mm"}</option>`).join("")}</select></label>
        </div>
        <div class="row" style="gap:6px;margin-top:6px;flex-wrap:wrap">
          <span class="small muted">Snap to</span>
          <button type="button" class="btn sm ghost" data-snap="centre" title="Keep the station, move onto the centre line of the part under the hub">Centre line</button>
          <button type="button" class="btn sm ghost" data-snap="nose" title="Tractor: just ahead of the nose, on its axis">Nose</button>
          <button type="button" class="btn sm ghost" data-snap="tail" title="Pusher: just behind the tail, on its axis">Tail</button>
        </div>`;
      card.addEventListener("click", (e) => {
        if (this.selected !== p.id && !e.target.closest("input,button,select")) { this.selected = p.id; this.render(); this.render3D(); }
      });
      card.querySelector(".zh input").addEventListener("change", (e) => { p.name = e.target.value || "Propeller"; this.ctx.save(); this.renderEstimates(); });
      card.querySelector("[data-del]").addEventListener("click", () => {
        if (this.moving === p.id) this.stopMove();
        this.ctx.S.propellers = this.list.filter((q) => q !== p);
        this.render();
        this.ctx.changed();
      });
      card.querySelector("[data-dup]").addEventListener("click", () => {
        const { me } = this.ctx;
        if (this.list.length >= me.limits.max_propellers) return upgradePrompt(`${me.plan_name} simulates up to ${me.limits.max_propellers} propeller${me.limits.max_propellers > 1 ? "s" : ""}.`);
        const q = { ...p, id: Math.random().toString(36).slice(2, 10), name: `${p.name} (copy)`, center: [...p.center],
          rotation: p.rotation === "cw" ? "ccw" : "cw" };
        const b = this.frame().fileToBody(q.center);
        this.setBody(q, [b[0], -b[1] || b[1] + 0.1 * this.frame().size[1], b[2]]);   // mirrored across the centre plane
        this.list.push(q);
        this.selected = q.id;
        this.render();
        this.ctx.changed();
      });
      card.querySelectorAll("[data-f]").forEach((b) => b.addEventListener("click", () => {
        if (p.family === b.dataset.f) return;
        p.family = b.dataset.f;
        const q = (PROP_PRESETS[p.family] || [])[1];
        if (q) Object.assign(p, { diameter_in: q.d, pitch_in: q.p, rpm: q.rpm, blades: q.b });
        this.render();
        this.ctx.changed();
      }));
      card.querySelector("[data-preset]").addEventListener("change", (e) => {
        if (e.target.value === "") return;
        const q = presets[+e.target.value];
        Object.assign(p, { diameter_in: q.d, pitch_in: q.p, rpm: q.rpm, blades: q.b });
        this.render();
        this.ctx.changed();
      });
      card.querySelectorAll("[data-k]").forEach((inp) => inp.addEventListener("change", () => {
        const v = parseFloat(inp.value);
        const lim = { diameter_in: [2, 40], pitch_in: [1, 30], rpm: [500, 40000] }[inp.dataset.k];
        if (!Number.isFinite(v) || v < lim[0] || v > lim[1]) { inp.classList.add("invalid"); return; }
        p[inp.dataset.k] = v;
        this.render();
        this.ctx.changed();
      }));
      card.querySelectorAll("[data-b]").forEach((b) => b.addEventListener("click", () => { p.blades = +b.dataset.b; this.render(); this.ctx.changed(); }));
      card.querySelectorAll("[data-r]").forEach((b) => b.addEventListener("click", () => { p.rotation = b.dataset.r; this.render(); this.ctx.changed(); }));
      card.querySelectorAll("[data-c]").forEach((inp) => inp.addEventListener("change", () => {
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) { inp.classList.add("invalid"); return; }
        inp.classList.remove("invalid");
        p.center[+inp.dataset.c] = v;
        this.render3D();
        this.ctx.save();
      }));
      card.querySelectorAll("[data-n]").forEach((b) => b.addEventListener("click", () => this.nudge(p, +b.dataset.n, +b.dataset.s)));
      card.querySelector("[data-step]").addEventListener("change", (e) => { this.step = +e.target.value; });
      card.querySelector("[data-move]").addEventListener("click", () => (this.moving === p.id ? this.stopMove() : this.startMove(p)));
      card.querySelector("[data-pick]").addEventListener("click", () => { if (this.moving) this.stopMove(); this.ctx.startPick(p.id, "prop"); });
      card.querySelectorAll("[data-snap]").forEach((b) => b.addEventListener("click", () => {
        const kind = b.dataset.snap;
        const body = this.frame().fileToBody(p.center);
        this.setBody(p, kind === "nose" ? this.noseHub(p) : kind === "tail" ? this.tailHub(p) : this.centred(p, body));
        this.syncInputs(p);
        this.render3D();
        this.ctx.save();
      }));
      box.append(card);
    }
    box.append(h("div", { id: "prop-compare" }));
    if (this.list.length) {
      const snap = h("label", { class: "check", style: "margin-top:8px" });
      snap.innerHTML = `<input type="checkbox" ${this.snapOnPlace ? "checked" : ""}><span class="small">When placing by clicking, put the hub on the centre line of the part clicked</span>`;
      snap.querySelector("input").addEventListener("change", (e) => { this.snapOnPlace = e.target.checked; });
      box.append(snap);
    }
    document.getElementById("prop-limit").textContent = this.list.length ? ""
      : "No propellers: the RC tunnel then runs like the General tunnel. Add one to test the airframe in its slipstream.";
    const s = document.getElementById("s-props");
    s.textContent = this.list.length ? `${this.list.length} prop${this.list.length > 1 ? "s" : ""}` : "none";
    this.renderEstimates();
    this.refreshEstimates();
  }

  /** Draw the propellers at the displayed rotation R (or clear them). */
  render3D(R = null, visible = true) {
    const { viewer } = this.ctx;
    const f = this.frame();
    if (!visible || !f) { viewer.detachMover(); return viewer.setProps([]); }
    R = R || this.ctx.displayRotation();
    viewer.setProps(this.list.map((p) => ({
      id: p.id, center: applyMat3(R, f.fileToBody(p.center)), axis: applyMat3(R, [-1, 0, 0]),
      radius: (p.diameter_in * IN) / 2, blades: p.blades, spin: p.rotation === "cw" ? 1 : -1, chord: CHORD[p.family] || 0.11,
    })));
  }

  /** Propellers as stored with a finished result (body-frame centres). */
  render3DFromResult(result, R) {
    this.ctx.viewer.detachMover();
    this.moving = null;
    const props = result.propellers || [];
    this.ctx.viewer.setProps(props.map((p) => ({
      id: p.name, center: applyMat3(R, p.center_body), axis: applyMat3(R, [-1, 0, 0]), radius: p.diameter_m / 2,
      blades: p.blades, spin: p.rotation === "cw" ? 1 : -1, chord: CHORD[p.family] || 0.11,
    })));
  }
}

/** Extra KPIs, notices and charts for an RC result at condition (a, v). */
export function rcResultsHTML(result, s, sb, sum, best) {
  if (!result.propellers || !s.props) return "";
  const net = s.net_N;
  const ts = sum.top_speed;
  const props = s.props;
  return `
    <div class="section-title">Propulsion at this condition</div>
    <div class="kpis">
      <div class="kpi"><div class="label">Thrust</div><div class="value">${fmtForce(s.thrust_N)}</div><div class="sub">${props.length} prop${props.length > 1 ? "s" : ""} · J ${sig(props[0].J, 2)}</div></div>
      <div class="kpi"><div class="label">${net >= 0 ? "Excess thrust" : "Thrust deficit"}</div><div class="value" style="color:var(${net >= 0 ? "--good-text" : "--critical-text"})">${fmtForce(Math.abs(net))}</div><div class="sub">thrust − airframe drag</div></div>
      <div class="kpi"><div class="label">Shaft power</div><div class="value">${fmtPower(s.shaft_power_W)}</div><div class="sub">motor output needed</div></div>
      <div class="kpi"><div class="label">Propeller efficiency</div><div class="value">${fmtPct(s.prop_eta, 0)}</div><div class="sub">thrust × V ÷ shaft power</div></div>
    </div>
    ${ts ? `<div class="notice ${ts.bracketed ? "good" : ""}" style="margin-top:10px">${icon("gauge")}<div>${ts.bracketed
      ? `Estimated level top speed <b style="color:var(--text)">${sig(ts.v, 3)} m/s</b> (${sig(ts.v * 3.6, 3)} km/h) at ${ts.aoa}°, where thrust equals drag.`
      : ts.above ? `Thrust still exceeds drag at ${sig(ts.v, 3)} m/s: the top speed is above the envelope. Extend the airspeed range to find it.`
      : `Drag exceeds thrust across the whole envelope at ${ts.aoa}°: more propeller (pitch, diameter, blades or rpm) is needed.`}</div></div>` : ""}
    <div style="overflow-x:auto;margin-top:10px"><table class="tbl compact"><thead><tr><th>Propeller</th><th class="r">Thrust</th><th class="r">Power</th><th class="r">Eff.</th><th class="r">J</th><th class="r">Tip M</th></tr></thead><tbody>
      ${props.map((p) => `<tr><td>${esc(p.name)}</td><td class="r">${fmtForce(p.thrust_N)}</td><td class="r">${fmtPower(p.power_W)}</td><td class="r">${fmtPct(p.eta, 0)}</td><td class="r">${sig(p.J, 2)}</td><td class="r">${sig(p.tip_mach, 2)}</td></tr>`).join("")}
    </tbody></table></div>
    <p class="small muted" style="margin-top:6px">Propellers: blade-element momentum theory per blade family, checked against UIUC wind-tunnel data
      (thrust and power within 8%, 2 to 4 blades). Drag is measured on the airframe inside the slipstream.</p>`;
}

export function rcCharts(card, result, base, best, cond) {
  if (!result.propellers || !base[cond.a].speeds[0].props) return;
  const env = result.envelope;
  if (env.speeds.length < 2) return;
  const row = (best || base)[cond.a];
  lineChart(card("Thrust and drag vs airspeed", `${env.aoas[cond.a]}°`), [
    { name: "Thrust", color: "var(--rc)", points: row.speeds.map((s) => [s.v, s.thrust_N]), dots: true },
    { name: "Drag", color: "var(--series-2)", points: row.speeds.map((s) => [s.v, s.drag_N]), dots: true },
  ], { height: 160, xName: "V", xLabel: "airspeed (m/s)", fmtY: (v) => fmtForce(v) });
  lineChart(card("Shaft power vs airspeed", `${env.aoas[cond.a]}°`), [
    { name: "Shaft power", color: "var(--rc)", points: row.speeds.map((s) => [s.v, s.shaft_power_W]), dots: true },
    { name: "Power to overcome drag", color: "var(--series-2)", points: row.speeds.map((s) => [s.v, s.power_W]), dots: true },
  ], { height: 150, xName: "V", xLabel: "airspeed (m/s)", fmtY: (v) => fmtPower(v) });
}
