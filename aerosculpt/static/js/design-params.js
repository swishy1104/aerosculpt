// Named design parameters of a template project (rocket, UAV, wing): a slider, a number box, a
// unit, a lock and an optimizer range per parameter. Moving one rebuilds the geometry on the
// server (milliseconds) and updates the 3D view without running a simulation.
import { api, esc, fmtPct, h, handleError, icon, sig, $, $$ } from "/static/js/core.js";

const GROUPS = { nose: "Nose", body: "Body", tail: "Tail", fins: "Fins", wing: "Wing", "tail surfaces": "Tail surfaces" };

// stored SI value <-> displayed value
export const UNIT = {
  m: { k: 1000, label: "mm", dp: 1 },
  deg: { k: 1, label: "°", dp: 1 },
  ratio: { k: 100, label: "%", dp: 1 },
  "-": { k: 1, label: "", dp: 3 },
};
export function showValue(p, v) {
  if (p.kind === "choice") return (p.choices.find((c) => c[0] === String(v)) || [v, v])[1];
  const u = UNIT[p.unit] || UNIT["-"];
  return `${(+v * u.k).toFixed(u.dp).replace(/\.0+$/, "")}${u.label ? (u.label === "°" ? "°" : " " + u.label) : ""}`;
}
export const toDisp = (p, v) => +(v * (UNIT[p.unit] || UNIT["-"]).k).toFixed(6);
export const fromDisp = (p, v) => v / (UNIT[p.unit] || UNIT["-"]).k;
export const stepOf = (p, lo, hi) => {
  const span = Math.abs(hi - lo) * (UNIT[p.unit] || UNIT["-"]).k;
  const raw = span / 200;
  const pow = 10 ** Math.floor(Math.log10(raw || 1));
  return [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) || pow;
};
export function isActive(p, values) {
  if (!p.show_if) return true;
  return Object.entries(p.show_if).every(([k, vs]) => vs.map(String).includes(String(values[k])));
}

export class DesignParams {
  constructor(ctx) {
    this.ctx = ctx;                 // { S, catalog, me, onGeometry(raw, info), save(), changed() }
    this.tm = ctx.S.template;
    this.def = ctx.catalog.templates[this.tm.id];
    this.tm.params = this.tm.params || {};
    this.timer = null;
    this.seq = 0;
    this.info = null;
    this.problems = [];
  }

  param(key) { return this.def.params.find((p) => p.key === key); }
  locked(p) { const c = this.tm.params[p.key]; return c && "locked" in c ? !!c.locked : !!p.locked; }
  bounds(p) { const c = this.tm.params[p.key]; return (c && c.bounds) || p.bounds; }
  unlocked() { return this.def.params.filter((p) => !this.locked(p) && isActive(p, this.tm.values)); }
  winged() { return this.tm.id === "uav" || this.tm.id === "wing"; }

  render(box) {
    this.box = box;
    const v = this.tm.values;
    const groups = [...new Set(this.def.params.map((p) => p.group))];
    box.innerHTML = `<p class="lead">${esc(this.def.help)} Each dimension has a range the optimizer may use and a lock;
        moving a slider updates the 3D view straight away — nothing is simulated until you run the tunnel.</p>
      <div class="dp-tools"><span class="small muted" id="dp-free"></span>
        <button class="btn sm ghost" type="button" id="dp-lock-all">${icon("lock")}Lock all</button>
        <button class="btn sm ghost" type="button" id="dp-reset">${icon("rotate")}Defaults</button></div>
      <div id="dp-problems"></div>
      ${groups.map((g) => `<div class="dp-group" data-g="${esc(g)}"><div class="dp-gh"><span>${esc(GROUPS[g] || g)}</span><span class="dp-derived" data-derived="${esc(g)}"></span></div>
        ${this.def.params.filter((p) => p.group === g).map((p) => this.rowHTML(p, v)).join("")}</div>`).join("")}
      <div class="kv" id="dp-overall" style="margin-top:10px"></div>`;
    for (const p of this.def.params) this.wire(p);
    $("#dp-lock-all", box).addEventListener("click", () => {
      const all = this.unlocked().length > 0;
      for (const p of this.def.params) (this.tm.params[p.key] = this.tm.params[p.key] || {}).locked = all;
      this.render(box);
      this.save();
    });
    $("#dp-reset", box).addEventListener("click", () => {
      for (const p of this.def.params) v[p.key] = p.default;
      this.render(box);
      this.update(true);
    });
    this.refreshRows();
    this.update(false, false);
  }

  rowHTML(p, v) {
    const lk = this.locked(p);
    const [lo, hi] = this.bounds(p);
    const lockBtn = `<button type="button" class="dp-lock ${lk ? "on" : ""}" data-lock="${p.key}" title="${lk ? "Locked: the optimizer keeps this value" : "Free: the optimizer may change this within its range"}"
      aria-pressed="${lk}">${icon(lk ? "lock" : "unlock")}</button>`;
    const help = p.help ? ` title="${esc(p.help)}"` : "";
    if (p.kind === "choice") {
      return `<div class="dp-row" data-row="${p.key}"><div class="dp-top"><span class="dp-l"${help}>${esc(p.label)}</span>${lockBtn}</div>
        <div class="seg dp-seg" data-choice="${p.key}">${p.choices.map(([k, l]) => `<button type="button" data-v="${esc(k)}" class="${String(v[p.key]) === k ? "on" : ""}">${esc(l)}</button>`).join("")}</div></div>`;
    }
    const u = UNIT[p.unit] || UNIT["-"];
    const st = stepOf(p, lo, hi);
    return `<div class="dp-row" data-row="${p.key}"><div class="dp-top"><span class="dp-l"${help}>${esc(p.label)}</span>
        <div class="input-unit dp-num"><input type="number" step="any" data-num="${p.key}" value="${toDisp(p, v[p.key])}" aria-label="${esc(p.label)}"><span class="unit">${esc(u.label || "–")}</span></div>${lockBtn}</div>
      <input type="range" data-range="${p.key}" min="${toDisp(p, Math.min(lo, v[p.key]))}" max="${toDisp(p, Math.max(hi, v[p.key]))}" step="${st}" value="${toDisp(p, v[p.key])}" aria-label="${esc(p.label)} slider">
      <div class="dp-bounds ${lk ? "dim" : ""}"><span>optimizer range</span><input type="number" step="any" data-lo="${p.key}" value="${toDisp(p, lo)}" aria-label="${esc(p.label)} minimum">
        <span>to</span><input type="number" step="any" data-hi="${p.key}" value="${toDisp(p, hi)}" aria-label="${esc(p.label)} maximum"><span>${esc(u.label)}</span></div></div>`;
  }

  wire(p) {
    const box = this.box;
    const v = this.tm.values;
    const lockB = $(`[data-lock="${p.key}"]`, box);
    lockB.addEventListener("click", () => {
      const c = (this.tm.params[p.key] = this.tm.params[p.key] || {});
      c.locked = !this.locked(p);
      const row = $(`[data-row="${p.key}"]`, box);
      row.outerHTML = this.rowHTML(p, v);
      this.wire(p);
      this.refreshRows();
      this.save();
    });
    if (p.kind === "choice") {
      $$(`[data-choice="${p.key}"] button`, box).forEach((b) => b.addEventListener("click", () => {
        v[p.key] = b.dataset.v;
        $$(`[data-choice="${p.key}"] button`, box).forEach((x) => x.classList.toggle("on", x === b));
        this.refreshRows();
        this.update(true);
      }));
      return;
    }
    const num = $(`[data-num="${p.key}"]`, box), rng = $(`[data-range="${p.key}"]`, box);
    const set = (disp, commit) => {
      let val = fromDisp(p, +disp);
      if (!Number.isFinite(val)) return;
      val = Math.min(Math.max(val, p.lo), p.hi);
      v[p.key] = val;
      if (document.activeElement !== num) num.value = toDisp(p, val);
      if (+rng.min > toDisp(p, val)) rng.min = toDisp(p, val);
      if (+rng.max < toDisp(p, val)) rng.max = toDisp(p, val);
      rng.value = toDisp(p, val);
      this.update(commit);
    };
    rng.addEventListener("input", () => set(rng.value, false));
    rng.addEventListener("change", () => set(rng.value, true));
    num.addEventListener("input", () => { if (num.value !== "") set(num.value, false); });
    num.addEventListener("change", () => { set(num.value, true); num.value = toDisp(p, v[p.key]); });
    const bset = () => {
      let lo = fromDisp(p, +$(`[data-lo="${p.key}"]`, box).value), hi = fromDisp(p, +$(`[data-hi="${p.key}"]`, box).value);
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
      lo = Math.min(Math.max(lo, p.lo), p.hi);
      hi = Math.min(Math.max(hi, p.lo), p.hi);
      if (lo > hi) [lo, hi] = [hi, lo];
      (this.tm.params[p.key] = this.tm.params[p.key] || {}).bounds = [lo, hi];
      rng.min = toDisp(p, Math.min(lo, v[p.key]));
      rng.max = toDisp(p, Math.max(hi, v[p.key]));
      rng.step = stepOf(p, lo, hi);
      this.save();
    };
    $(`[data-lo="${p.key}"]`, box).addEventListener("change", bset);
    $(`[data-hi="${p.key}"]`, box).addEventListener("change", bset);
  }

  refreshRows() {
    const v = this.tm.values;
    for (const p of this.def.params) {
      const row = $(`[data-row="${p.key}"]`, this.box);
      if (row) row.classList.toggle("hidden", !isActive(p, v));
    }
    const n = this.unlocked().length;
    $("#dp-free", this.box).textContent = n ? `${n} free parameter${n > 1 ? "s" : ""} for the optimizer` : "All parameters locked";
    $("#dp-lock-all", this.box).innerHTML = `${icon(n ? "lock" : "unlock")}${n ? "Lock all" : "Unlock all"}`;
    this.ctx.changed && this.ctx.changed();
  }

  // geometry: preview while dragging, full resolution and saved when released
  update(commit, geometry = true) {
    clearTimeout(this.timer);
    const run = async () => {
      const seq = ++this.seq;
      try {
        const r = await api("/api/templates/build", { method: "POST", body: { template: this.tm.id, values: this.tm.values, preview: !commit } });
        if (seq !== this.seq) return;
        const bin = atob(r.positions);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        this.info = r.info;
        this.problems = r.problems;
        if (geometry) this.ctx.onGeometry(new Float32Array(bytes.buffer), r);
        this.showInfo();
      } catch (e) { handleError(e); }
    };
    if (commit) { run(); this.save(); } else this.timer = setTimeout(run, geometry ? 40 : 0);
  }

  save() { this.ctx.save(); }

  showInfo() {
    const i = this.info || {};
    const set = (g, html) => { const el = $(`[data-derived="${g}"]`, this.box); if (el) el.innerHTML = html; };
    if (i.nose_fineness !== undefined) set("nose", `fineness ${sig(i.nose_fineness, 3)}`);
    if (i.body_fineness !== undefined) set("body", `fineness ${sig(i.body_fineness, 3)}`);
    if (i.base_diameter !== undefined) set("tail", `base Ø ${sig(i.base_diameter * 1000, 3)} mm`);
    if (i.fin_area_each !== undefined) set("fins", `${sig(i.fin_area_each * 1e4, 3)} cm² each`);
    if (i.wing_area !== undefined) set("wing", `${esc(i.airfoil)} · ${sig(i.wing_area, 3)} m² · AR ${sig(i.aspect_ratio, 3)}`);
    $("#dp-overall", this.box).innerHTML = `
      <span class="k">Overall size</span><span class="v">${sig(i.overall_length * 1000, 4)} × ${sig(i.overall_width * 1000, 4)} × ${sig(i.overall_height * 1000, 4)} mm</span>
      <span class="k">${this.tm.id === "wing" ? "Wing volume" : "Body volume"}</span><span class="v">${sig(i.volume * 1e3, 4)} L</span>
      <span class="k">Thinnest section</span><span class="v">${sig(i.min_thickness * 1000, 3)} mm</span>
      ${i.mac ? `<span class="k">Mean chord</span><span class="v">${sig(i.mac * 1000, 4)} mm</span>` : ""}`;
    $("#dp-problems", this.box).innerHTML = this.problems && this.problems.length
      ? `<div class="notice warn" style="margin:8px 0">${icon("alert")}<div>${this.problems.map(esc).join("; ")}.</div></div>` : "";
  }
}

// ---- results helpers ---------------------------------------------------------------------------------
export function paramsTableHTML(res, catalog) {
  const t = res.template;
  if (!t) return "";
  const def = catalog.templates[t.id];
  const keys = new Set((res.variables || []).map((v) => v.key));
  const rows = def.params.filter((p) => keys.has(p.key) || String(t.base_values[p.key]) !== String(t.best_values[p.key]));
  const fmtd = (p, a, b) => {
    if (p.kind === "choice" || String(a) === String(b)) return "";
    const d = (b - a) / (Math.abs(a) || 1);
    return Math.abs(a) > 1e-12 ? fmtPct(d, 1, true) : "";
  };
  const ib = t.info_base || {}, it = t.info_best || {};
  return `<div class="section-title">Named dimensions</div>
    <div style="overflow-x:auto"><table class="tbl compact"><thead><tr><th>Parameter</th><th class="r">Original</th><th class="r">Optimized</th><th class="r">Change</th></tr></thead><tbody>
    ${rows.map((p) => `<tr><td>${esc(p.label)}</td><td class="r">${esc(showValue(p, t.base_values[p.key]))}</td><td class="r"><b>${esc(showValue(p, t.best_values[p.key]))}</b></td>
      <td class="r">${fmtd(p, t.base_values[p.key], t.best_values[p.key])}</td></tr>`).join("")}
    ${ib.airfoil ? `<tr><td>Airfoil section</td><td class="r">${esc(ib.airfoil)}</td><td class="r"><b>${esc(it.airfoil)}</b></td><td></td></tr>` : ""}
    ${ib.nose_fineness !== undefined ? `<tr><td>Nose fineness ratio</td><td class="r">${sig(ib.nose_fineness, 3)}</td><td class="r">${sig(it.nose_fineness, 3)}</td><td></td></tr>` : ""}
    ${ib.aspect_ratio !== undefined ? `<tr><td>Wing aspect ratio</td><td class="r">${sig(ib.aspect_ratio, 3)}</td><td class="r">${sig(it.aspect_ratio, 3)}</td><td></td></tr>` : ""}
    ${ib.wing_area !== undefined ? `<tr><td>Wing area</td><td class="r">${sig(ib.wing_area, 4)} m²</td><td class="r">${sig(it.wing_area, 4)} m²</td><td></td></tr>` : ""}
    </tbody></table></div>`;
}

export function strategyHTML(res) {
  const s = res.strategy;
  if (!s) return "";
  const n = s.noise || {};
  const rows = (s.ranking || []).filter((r) => r.effect !== null && r.effect !== undefined);
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.effect)));
  const eps = s.epsilon;
  const conf = s.confirmed;
  const effects = rows.map((r) => Math.abs(r.effect));
  const across = s.explore_cells ? Math.round(Math.min(...s.explore_cells.filter((c) => c > 0))) : null;
  const coarse = eps > 0.04 && n.samples > 1 ? `<div class="notice warn" style="margin:8px 0">${icon("alert")}<div><b style="color:var(--text)">The tunnel grid was too coarse for these changes.</b>
      Its measured run-to-run error (±${fmtPct(n.sigma, 1)}${across ? `, with only about ${across} cells across the body` : ""}) is larger than the parameters' effects
      (largest ${fmtPct(Math.max(0, ...effects), 1)}), so no change could be accepted. Raise the fidelity, narrow the job to the parameters that matter,
      or — for nose shapes — use the Supersonic tunnel, whose methods are deterministic and validated for noses. The ranking above still shows which way each parameter pushes.</div></div>` : "";
  return `<div class="section-title">How the runs were spent</div>
    <div class="kv">
      <span class="k">Explored with</span><span class="v">${esc(s.explore)}${s.confirm ? (s.confirm.includes("model") ? ` · confirmed with the ${esc(s.confirm)}`
        : s.confirm_grid ? ` · confirmed on the ${esc(s.confirm_grid)} grid (one level up, charged as ${esc(s.confirm)})` : ` · confirmed at ${esc(s.confirm)} fidelity`) : ""}</span>
      <span class="k">Designs explored</span><span class="v">${s.designs_explored} of a ${s.budget}-design budget${s.rejected ? ` · ${s.rejected} rejected before simulating` : ""}</span>
      <span class="k">Solver error</span><span class="v">${n.samples > 1 ? `±${fmtPct(n.sigma, 2)} (measured)` : "deterministic model"} → changes must beat ${fmtPct(eps, 2)}</span>
      <span class="k">Stopped because</span><span class="v" style="text-align:right">${esc(s.stop_reason || "—")}</span>
      ${s.cache_hits ? `<span class="k">Reused from earlier runs</span><span class="v">${s.cache_hits} identical case${s.cache_hits > 1 ? "s" : ""}</span>` : ""}</div>
    <div class="section-title">Screening: effect of each parameter</div>
    <p class="small muted" style="margin:-4px 0 8px">Each free parameter was moved once by a quarter of its range (each alternative profile tried once).
      Bars show the change in the objective; the dashed line is the measured error. Only the top ${s.top_k} with a real effect were optimized.</p>
    <div class="rank">${rows.map((r) => {
      const w = Math.abs(r.effect) / max * 100, e = Math.min(100, eps / max * 100);
      const lab = r.choice ? `${r.label}: ${r.choice}` : r.label;
      const dir = r.choice ? (r.effect < 0 ? "better" : "worse") : (r.direction === "increase" ? "↑ helps" : "↓ helps");
      return `<div class="rank-row ${r.selected ? "sel" : ""} ${Math.abs(r.effect) < eps ? "noise" : ""}">
        <span class="rank-l" title="${esc(lab)}">${esc(lab)}</span>
        <div class="rank-bar"><div class="rank-fill ${r.effect < 0 ? "good" : "bad"}" style="width:${Math.max(2, w)}%"></div><div class="rank-eps" style="left:${e}%"></div></div>
        <span class="rank-v">${fmtPct(r.effect, 2, true)}</span><span class="rank-tag">${r.selected ? "optimized" : Math.abs(r.effect) < eps ? "noise" : esc(dir)}</span></div>`;
    }).join("")}</div>
    ${coarse}
    ${(s.ranking || []).filter((r) => r.effect === null || r.effect === undefined).map((r) => `<p class="small muted" style="margin:4px 0">${esc(r.label)}${r.choice ? ": " + esc(r.choice) : ""} — ${esc(r.note || "not tried")}</p>`).join("")}
    ${s.adopted ? `<p class="small" style="margin-top:8px">${icon("check", "i sm")} Adopted <b>${esc(s.adopted.label)}: ${esc(s.adopted.choice)}</b> (${fmtPct(s.adopted.effect, 2, true)}) before the search.</p>` : ""}
    ${conf ? `<div class="section-title">Confirmation at full resolution</div>
      <table class="tbl compact"><thead><tr><th>Candidate</th><th class="r">Explored</th><th class="r">Confirmed</th></tr></thead><tbody>
      ${conf.rows.map((r, i) => `<tr><td>#${i + 1}</td><td class="r">${fmtPct(r.explore - 1, 2, true)}</td><td class="r ${r.confirmed < 1 - conf.eps ? "delta good" : ""}">${fmtPct(r.confirmed - 1, 2, true)}</td></tr>`).join("")}
      </tbody></table><p class="small muted" style="margin-top:6px">Objective change against the original design (the full-resolution error estimate is ±${fmtPct(conf.eps / 2, 2)}).</p>` : ""}`;
}

export function fixedLiftHTML(res) {
  const f = res.fixed_lift;
  if (!f) return "";
  const rows = f.baseline.map((b, i) => ({ b, o: f.best[i] }));
  const cell = (b, o, k, fmt) => `<td class="r">${fmt(b[k])}</td><td class="r"><b>${fmt(o[k])}</b></td>`;
  const N = (v) => sig(v, 3);
  return `<div class="section-title">Drag at the lift needed (${esc(f.basis)})</div>
    <div style="overflow-x:auto"><table class="tbl compact"><thead><tr><th>Speed</th><th class="r" colspan="2">Drag (N)</th><th class="r" colspan="2">Attitude α</th><th class="r" colspan="2">L/D</th></tr>
      <tr><th></th><th class="r muted">orig.</th><th class="r muted">opt.</th><th class="r muted">orig.</th><th class="r muted">opt.</th><th class="r muted">orig.</th><th class="r muted">opt.</th></tr></thead><tbody>
    ${rows.map(({ b, o }) => `<tr><td>${b.v} m/s</td>${cell(b, o, "drag_N", N)}${cell(b, o, "alpha", (v) => sig(v, 2) + "°")}${cell(b, o, "ld", (v) => (v ? sig(v, 3) : "—"))}</tr>`).join("")}
    </tbody></table></div>
    <div class="legend small" style="margin-top:6px">Drag = induced + wing/tail profile + fuselage. At ${rows[Math.floor(rows.length / 2)].b.v} m/s:
      induced ${N(rows[Math.floor(rows.length / 2)].b.induced_N)} → ${N(rows[Math.floor(rows.length / 2)].o.induced_N)} N,
      profile ${N(rows[Math.floor(rows.length / 2)].b.profile_N)} → ${N(rows[Math.floor(rows.length / 2)].o.profile_N)} N,
      fuselage ${N(rows[Math.floor(rows.length / 2)].b.body_N)} → ${N(rows[Math.floor(rows.length / 2)].o.body_N)} N
      (${f.body_from === "CFD" ? "fuselage from the CFD" : "fuselage from Hoerner's method"}).
      ${f.neutral_point_shift_mm !== undefined ? ` Neutral point moved ${sig(f.neutral_point_shift_mm, 2)} mm.` : ""}</div>`;
}
