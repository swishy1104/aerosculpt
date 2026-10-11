// Shape features: named, adjustable edits stacked on the model like a CAD timeline (nose and
// tail cones, fairings over exposed parts, pods, wings, fins, slant cuts, wing and part edits).
// The geometry is built on the server (geometry/features.py); this panel edits the list, its
// dimensions (slider, number, lock, optimizer range) and shows the result in the 3D view.
import { api, esc, h, handleError, icon, sig, toast, $, $$ } from "/static/js/core.js";
import { UNIT, fromDisp, isActive, stepOf, toDisp } from "/static/js/design-params.js";

const KIND_LABEL = { add: "Add", cut: "Cut", modify: "Reshape" };
const GROUP_LABEL = { shape: "Shape", fit: "Fit", position: "Position", planform: "Planform", section: "Section",
  cut: "Cut", move: "Move", scale: "Scale", turn: "Turn" };
// features placed with a click on the model (nose and tail cones find the ends themselves)
const NEEDS_CLICK = { fairing: 1, pod: 1, wing: 1, fins: 1, slant: 1, wingpanel: 1, part: 1 };
const CLICK_HINT = {
  fairing: "Click the part to enclose (a motor, camera, bolt heads)",
  pod: "Click where the pod goes",
  wing: "Click where the wing's root leading edge goes",
  fins: "Click the body where the fins' leading edge goes",
  slant: "Click the surface to cut",
  wingpanel: "Click the wing to reshape",
  part: "Click the part to move or scale",
};

export class FeatureEditor {
  /** ctx: { S, catalog, me, pid, pick(type, hint) -> Promise<{point, normal}|null>, onGeometry(Float32Array|null, r),
   *        save(), changed(), select(id), selected() } */
  constructor(ctx) {
    this.ctx = ctx;
    this.S = ctx.S;
    this.S.features = this.S.features || [];
    this.defs = ctx.catalog.features || {};
    this.open = null;
    this.timer = null;
    this.seq = 0;
    this.problems = [];
    this.info = null;
  }

  get list() { return this.S.features; }
  def(f) { return this.defs[f.type]; }
  locked(p, f) { const c = (f.params || {})[p.key]; return c && "locked" in c ? !!c.locked : !!p.locked; }
  bounds(p, f) {
    const c = (f.params || {})[p.key];
    if (c && c.bounds) return c.bounds;
    const v = f.values[p.key];
    if (p.unit === "deg") return [Math.max(p.lo, v - 10), Math.min(p.hi, v + 10)];
    if (/^(offset|move|depth)/.test(p.key)) { const s = Math.max(Math.abs(v), 0.02); return [Math.max(p.lo, v - s), Math.min(p.hi, v + s)]; }
    return v > 0 ? [Math.max(p.lo, 0.75 * v), Math.min(p.hi, 1.3 * v)] : [p.lo, p.hi];
  }
  freeParams() {
    const out = [];
    for (const f of this.list) {
      if (f.enabled === false) continue;
      for (const p of this.def(f).params) if (!this.locked(p, f) && isActive(p, f.values)) out.push([f, p]);
    }
    return out;
  }

  render(box) {
    this.box = box || this.box;
    const limit = this.ctx.me.limits.max_features ?? 24;
    const kinds = ["add", "cut", "modify"];
    this.box.innerHTML = `<p class="lead">Add or reshape parts like a CAD timeline: features apply in order, each with named
        dimensions. Unlock a dimension to let the optimizer change it. Added parts merge with the body in the tunnel.</p>
      ${kinds.map((k) => `<div class="ft-kind"><span class="ft-kl">${KIND_LABEL[k]}</span><div class="ft-add">${Object.entries(this.defs)
        .filter(([, d]) => d.kind === k).map(([t, d]) => `<button class="btn sm" type="button" data-add="${t}" title="${esc(d.help)}">${icon(d.icon)}${esc(d.title)}</button>`).join("")}</div></div>`).join("")}
      <div id="ft-problems"></div>
      <div class="ft-list" id="ft-list"></div>
      <p class="small muted" id="ft-limit" style="margin:6px 0 0">${this.list.length ? "" : "No features yet: the model is simulated as uploaded."}
        ${this.list.length >= limit ? ` Your plan allows ${limit}.` : ""}</p>`;
    $$("[data-add]", this.box).forEach((b) => b.addEventListener("click", () => this.add(b.dataset.add)));
    this.renderList();
    this.showProblems();
  }

  renderList() {
    const box = $("#ft-list", this.box);
    box.innerHTML = "";
    this.list.forEach((f, i) => box.append(this.card(f, i)));
  }

  card(f, i) {
    const d = this.def(f);
    const sel = this.ctx.selected() === f.id;
    const isOpen = this.open === f.id;
    const card = h("div", { class: `ft-card${sel ? " sel" : ""}${f.enabled === false ? " off" : ""}${isOpen ? " open" : ""}` });
    const free = d.params.filter((p) => !this.locked(p, f) && isActive(p, f.values)).length;
    card.innerHTML = `<div class="ft-h"><span class="ft-n">${i + 1}</span>${icon(d.icon, "i sm")}
        <input type="text" value="${esc(f.name)}" maxlength="60" aria-label="Feature name">
        ${free ? `<span class="badge accent" title="Dimensions the optimizer may change">${free} free</span>` : ""}
        <button class="icon-btn" data-act="up" title="Earlier in the timeline" ${i === 0 ? "disabled" : ""}>${icon("chev", "i sm up")}</button>
        <button class="icon-btn" data-act="toggle" title="${f.enabled === false ? "Switched off: click to use it" : "Switch off (keep it)"}">${icon(f.enabled === false ? "x" : "eye", "i sm")}</button>
        ${d.region || NEEDS_CLICK[f.type] ? `<button class="icon-btn" data-act="edit" title="Move${d.region ? ", turn and resize" : " and turn"} it in the 3D view">${icon("move", "i sm")}</button>` : ""}
        <button class="icon-btn" data-act="del" title="Remove">${icon("trash", "i sm")}</button></div>
      <div class="ft-b">${this.paramsHTML(f)}</div>`;
    const inp = $(".ft-h input", card);
    inp.addEventListener("change", () => { f.name = inp.value.trim() || d.title; this.ctx.save(); });
    $(".ft-h", card).addEventListener("click", (e) => {
      if (e.target.closest("input,button")) return;
      this.open = this.open === f.id ? null : f.id;
      this.ctx.select(f.id);
      this.renderList();
    });
    $$("[data-act]", card).forEach((b) => b.addEventListener("click", () => {
      const a = b.dataset.act;
      if (a === "del") { this.S.features = this.list.filter((q) => q !== f); if (this.open === f.id) this.open = null; }
      else if (a === "toggle") f.enabled = f.enabled === false;
      else if (a === "up" && i > 0) { const l = this.list; [l[i - 1], l[i]] = [l[i], l[i - 1]]; }
      else if (a === "edit") { this.open = f.id; this.ctx.select(f.id, true); }
      this.render();
      this.rebuild(true);
    }));
    if (isOpen) this.wire(f, card);
    return card;
  }

  paramsHTML(f) {
    const d = this.def(f);
    const groups = [...new Set(d.params.map((p) => p.group))];
    return `<p class="small muted" style="margin:2px 0 8px">${esc(d.help)}</p>` + groups.map((g) => `<div class="dp-group"><div class="dp-gh"><span>${esc(GROUP_LABEL[g] || g)}</span></div>
      ${d.params.filter((p) => p.group === g).map((p) => this.rowHTML(f, p)).join("")}</div>`).join("");
  }

  rowHTML(f, p) {
    const v = f.values;
    const lk = this.locked(p, f);
    const [lo, hi] = this.bounds(p, f);
    const id = `${f.id}-${p.key}`;
    const hide = isActive(p, v) ? "" : " hidden";
    const lockBtn = `<button type="button" class="dp-lock ${lk ? "on" : ""}" data-lock="${p.key}" title="${lk ? "Locked: the optimizer keeps this value" : "Free: the optimizer may change this within its range"}"
      aria-pressed="${lk}">${icon(lk ? "lock" : "unlock")}</button>`;
    const help = p.help ? ` title="${esc(p.help)}"` : "";
    if (p.kind === "choice") {
      return `<div class="dp-row${hide}" data-row="${p.key}"><div class="dp-top"><span class="dp-l"${help}>${esc(p.label)}</span>${lockBtn}</div>
        <div class="seg dp-seg" data-choice="${p.key}">${p.choices.map(([k, l]) => `<button type="button" data-v="${esc(k)}" class="${String(v[p.key]) === k ? "on" : ""}">${esc(l)}</button>`).join("")}</div></div>`;
    }
    const u = UNIT[p.unit] || UNIT["-"];
    // the slider reaches half a range beyond the optimizer's range on either side
    const span = Math.max(hi - lo, 1e-9);
    const rlo = Math.max(p.lo, Math.min(lo, v[p.key]) - 0.5 * span), rhi = Math.min(p.hi, Math.max(hi, v[p.key]) + 0.5 * span);
    return `<div class="dp-row${hide}" data-row="${p.key}"><div class="dp-top"><label class="dp-l" for="${id}"${help}>${esc(p.label)}</label>
        <div class="input-unit dp-num"><input id="${id}" type="number" step="any" data-num="${p.key}" value="${toDisp(p, v[p.key])}"><span class="unit">${esc(u.label || "–")}</span></div>${lockBtn}</div>
      <input type="range" data-range="${p.key}" min="${toDisp(p, rlo)}" max="${toDisp(p, rhi)}" step="${stepOf(p, rlo, rhi)}" value="${toDisp(p, v[p.key])}" aria-label="${esc(p.label)} slider">
      <div class="dp-bounds ${lk ? "dim" : ""}"><span>optimizer range</span><input type="number" step="any" data-lo="${p.key}" value="${toDisp(p, lo)}" aria-label="${esc(p.label)} minimum">
        <span>to</span><input type="number" step="any" data-hi="${p.key}" value="${toDisp(p, hi)}" aria-label="${esc(p.label)} maximum"><span>${esc(u.label)}</span></div></div>`;
  }

  wire(f, card) {
    const d = this.def(f);
    for (const p of d.params) {
      const row = $(`[data-row="${p.key}"]`, card);
      $(`[data-lock="${p.key}"]`, row).addEventListener("click", () => {
        const c = ((f.params = f.params || {})[p.key] = f.params[p.key] || {});
        c.locked = !this.locked(p, f);
        this.renderList();
        this.ctx.save();
        this.ctx.changed();
      });
      if (p.kind === "choice") {
        $$(`[data-choice="${p.key}"] button`, row).forEach((b) => b.addEventListener("click", () => {
          f.values[p.key] = b.dataset.v;
          this.renderList();
          this.rebuild(true);
        }));
        continue;
      }
      const num = $(`[data-num="${p.key}"]`, row), rng = $(`[data-range="${p.key}"]`, row);
      const set = (disp, commit) => {
        let val = fromDisp(p, +disp);
        if (!Number.isFinite(val)) return;
        val = Math.min(Math.max(val, p.lo), p.hi);
        f.values[p.key] = val;
        if (document.activeElement !== num) num.value = toDisp(p, val);
        if (+rng.min > toDisp(p, val)) rng.min = toDisp(p, val);
        if (+rng.max < toDisp(p, val)) rng.max = toDisp(p, val);
        rng.value = toDisp(p, val);
        this.rebuild(commit);
      };
      rng.addEventListener("input", () => set(rng.value, false));
      rng.addEventListener("change", () => set(rng.value, true));
      num.addEventListener("change", () => { set(num.value, true); num.value = toDisp(p, f.values[p.key]); });
      const bset = () => {
        let lo = fromDisp(p, +$(`[data-lo="${p.key}"]`, row).value), hi = fromDisp(p, +$(`[data-hi="${p.key}"]`, row).value);
        if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
        lo = Math.min(Math.max(lo, p.lo), p.hi);
        hi = Math.min(Math.max(hi, p.lo), p.hi);
        if (lo > hi) [lo, hi] = [hi, lo];
        ((f.params = f.params || {})[p.key] = f.params[p.key] || {}).bounds = [lo, hi];
        this.ctx.save();
      };
      $(`[data-lo="${p.key}"]`, row).addEventListener("change", bset);
      $(`[data-hi="${p.key}"]`, row).addEventListener("change", bset);
    }
  }

  async add(type) {
    const limit = this.ctx.me.limits.max_features ?? 24;
    if (this.list.length >= limit) return toast(`Your plan allows ${limit} shape features.`, "bad");
    let where = null;
    if (NEEDS_CLICK[type]) {
      where = await this.ctx.pick(type, CLICK_HINT[type]);
      if (!where) return;
    }
    try {
      const f = await api(`/api/projects/${this.ctx.pid}/features/suggest`, { method: "POST",
        body: { type, features: this.list, ...(where ? { point: where.point, normal: where.normal } : {}) } });
      const n = this.list.filter((q) => q.type === type).length;
      if (n) f.name = `${f.name} ${n + 1}`;
      this.list.push(f);
      this.open = f.id;
      this.ctx.select(f.id, !!this.def(f).region);
      this.render();
      this.rebuild(true);
      toast(`${f.name} added. Adjust its dimensions on the left${this.def(f).region ? " or drag its box in the view" : ""}.`, "good");
    } catch (e) { handleError(e); }
  }

  /** after a gizmo drag changed a feature's placement */
  placed() { this.rebuild(true); }

  rebuild(commit) {
    clearTimeout(this.timer);
    const run = async () => {
      const seq = ++this.seq;
      const active = this.list.filter((f) => f.enabled !== false);
      if (!active.length) {
        this.problems = [];
        this.info = null;
        this.ctx.onGeometry(null, null);
        this.showProblems();
        return;
      }
      try {
        const r = await api(`/api/projects/${this.ctx.pid}/features/build`, { method: "POST", body: { features: this.list } });
        if (seq !== this.seq) return;
        const bin = atob(r.positions);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        this.problems = r.problems || [];
        this.info = r.info;
        this.ctx.onGeometry(new Float32Array(bytes.buffer), r);
        this.showProblems();
      } catch (e) { handleError(e); }
    };
    if (commit) this.ctx.save();
    this.timer = setTimeout(run, commit ? 0 : 120);
    this.ctx.changed();
  }

  showProblems() {
    const el = $("#ft-problems", this.box);
    if (!el) return;
    const i = this.info;
    el.innerHTML = (this.problems.length ? `<div class="notice warn" style="margin:10px 0 0">${icon("alert")}<div>${this.problems.map(esc).join("; ")}.</div></div>` : "") +
      (i && this.list.some((f) => f.enabled !== false) ? `<div class="kv" style="margin-top:10px"><span class="k">With features</span><span class="v">${sig(i.overall_length * 1000, 4)} × ${sig(i.overall_width * 1000, 4)} × ${sig(i.overall_height * 1000, 4)} mm</span>
        <span class="k">Faces</span><span class="v">${(i.faces || 0).toLocaleString()} (${(i.hidden_faces || 0).toLocaleString()} buried inside other parts)</span></div>` : "");
  }
}

/** Results: the feature dimensions the optimizer was free to change. */
export function featureChangesHTML(res, catalog) {
  const ch = res.features && res.features.changes;
  if (!ch || !ch.length) return "";
  const fmt = (c, v) => {
    if (c.unit === "choice") return esc(String(v));
    const u = UNIT[c.unit] || UNIT["-"];
    return `${sig(v * u.k, 4)}${u.label ? (u.label === "°" ? "°" : " " + u.label) : ""}`;
  };
  return `<div class="section-title">Shape features</div>
    <div style="overflow-x:auto"><table class="tbl compact"><thead><tr><th>Dimension</th><th class="r">Original</th><th class="r">Optimized</th></tr></thead><tbody>
    ${ch.map((c) => `<tr><td>${esc(c.label)}</td><td class="r">${fmt(c, c.base)}</td><td class="r"><b>${fmt(c, c.best)}</b></td></tr>`).join("")}
    </tbody></table></div>`;
}
