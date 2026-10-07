// Supersonic → hypersonic tunnel: Mach envelope, thermal settings and materials, results.
import { esc, fmtForce, fmtInt, fmtPct, icon, sig, upgradePrompt } from "/static/js/core.js";
import { lineChart } from "/static/js/charts.js";
import { b64bytes, dequantize } from "/static/js/colormap.js";

export const DEFAULT_HYPER = {
  envelope: { mach_min: 1.5, mach_max: 4, mach_count: 4, aoa_min: 0, aoa_max: 6, aoa_count: 3, altitude: 15000, yaw: 0, ref_area: null, ref_length: null },
  thermal: { material: "aluminium", wall: "radiative", t_wall_c: 25, transition: "auto", re_transition: 1e6, nose_radius_mm: null },
};
export const degC = (K) => `${Math.round(K - 273.15).toLocaleString()} °C`;
const kW = (w) => (w >= 1e6 ? `${sig(w / 1e6, 3)} MW/m²` : `${sig(w / 1e3, 3)} kW/m²`);

// U.S. Standard Atmosphere 1976, geometric altitude (mirror of aerosculpt/hyper/atmosphere.py)
const LAYERS = [[0, 288.15, -0.0065, 101325], [11000, 216.65, 0, 22632.06], [20000, 216.65, 0.001, 5474.889],
  [32000, 228.65, 0.0028, 868.0187], [47000, 270.65, 0, 110.9063], [51000, 270.65, -0.0028, 66.93887], [71000, 214.65, -0.002, 3.95642]];
export function us76(z) {
  const R = 287.05287, h = (6356766 * z) / (6356766 + z);
  let b = LAYERS[0];
  for (const l of LAYERS) if (h >= l[0]) b = l;
  const [hb, Tb, L, pb] = b;
  const T = Tb + L * (h - hb);
  const p = L === 0 ? pb * Math.exp((-9.80665 * (h - hb)) / (R * Tb)) : pb * Math.pow(Tb / T, 9.80665 / (R * L));
  const rho = p / (R * T), mu = (1.458e-6 * Math.pow(T, 1.5)) / (T + 110.4);
  return { T, p, rho, mu, a: Math.sqrt(1.4 * R * T) };
}

function pts(lo, hi, n, d = 1) {
  n = Math.max(1, Math.round(n));
  const f = 10 ** d;
  if (n === 1 || hi - lo < 1e-9) return [Math.round(lo * f) / f];
  return [...new Set(Array.from({ length: n }, (_, i) => Math.round((lo + ((hi - lo) * i) / (n - 1)) * f) / f))];
}

export function materialOf(spec, catalog) {
  if (spec && typeof spec === "object") return { key: "custom", ...spec };
  const k = spec in catalog.materials ? spec : "aluminium";
  return { key: k, ...catalog.materials[k] };
}

export class HyperPanel {
  /** ctx: { S, me, catalog, save(), changed(), preview() } */
  constructor(ctx) {
    this.ctx = ctx;
    const S = ctx.S;
    S.hyper = S.hyper || structuredClone(DEFAULT_HYPER);
    S.hyper.envelope = { ...DEFAULT_HYPER.envelope, ...(S.hyper.envelope || {}) };
    S.hyper.thermal = { ...DEFAULT_HYPER.thermal, ...(S.hyper.thermal || {}) };
    const E = S.hyper.envelope, T = S.hyper.thermal, me = ctx.me;
    E.mach_count = Math.min(E.mach_count, me.limits.max_speeds);
    E.aoa_count = Math.min(E.aoa_count, me.limits.max_angles);
    const $ = (id) => document.getElementById(id);
    const lim = { mach_min: [1.2, 12], mach_max: [1.2, 12], mach_count: [1, me.limits.max_speeds], aoa_min: [-20, 20],
      aoa_max: [-20, 20], aoa_count: [1, me.limits.max_angles] };
    for (const [k, [lo, hi]] of Object.entries(lim)) {
      const el = $("h_" + k);
      el.value = E[k];
      el.addEventListener("change", () => {
        let v = parseFloat(el.value);
        if (!Number.isFinite(v)) { el.value = E[k]; return; }
        if (k.endsWith("_count")) {
          v = Math.round(v);
          if (v > hi) upgradePrompt(`${me.plan_name} sweeps up to ${hi} ${k === "aoa_count" ? "angles" : "Mach numbers"}.`);
        }
        v = Math.min(hi, Math.max(lo, v));
        el.value = v;
        E[k] = v;
        if (k.startsWith("aoa")) ctx.preview();
        this.info();
        ctx.changed();
      });
    }
    $("h_altitude").value = E.altitude / 1000;
    $("h_altitude").addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      E.altitude = Number.isFinite(v) ? Math.min(80, Math.max(0, v)) * 1000 : E.altitude;
      e.target.value = E.altitude / 1000;
      this.info();
      ctx.changed();
    });
    $("h_ref_area").value = E.ref_area || "";
    $("h_ref_area").addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      E.ref_area = Number.isFinite(v) && v > 0 ? v : null;
      ctx.changed();
    });
    // thermal
    const sel = $("h_material");
    sel.innerHTML = Object.entries(ctx.catalog.materials).map(([k, m]) => `<option value="${k}">${esc(m.name)} · ${fmtInt(m.t_max_c)} °C</option>`).join("")
      + `<option value="custom">Custom material…</option>`;
    const syncMaterial = () => {
      const m = materialOf(T.material, ctx.catalog);
      sel.value = m.key;
      $("custom-material").classList.toggle("hidden", m.key !== "custom");
      if (m.key === "custom") {
        $("cm_name").value = m.name || "";
        $("cm_tmax").value = m.t_max_c;
        $("cm_eps").value = m.emissivity;
      }
      $("material-info").innerHTML = `<span class="k">Service limit</span><span class="v">${fmtInt(m.t_max_c)} °C</span>
        <span class="k">Emissivity</span><span class="v">${sig(m.emissivity, 2)}</span>
        ${m.note ? `<span class="k" style="grid-column:1/-1">${esc(m.note)}</span>` : ""}`;
      $("s-thermal").textContent = (m.name || "Custom").split(" ")[0];
    };
    sel.addEventListener("change", () => {
      T.material = sel.value === "custom" ? { name: "Custom material", t_max_c: 300, emissivity: 0.8 } : sel.value;
      syncMaterial();
      ctx.changed();
    });
    for (const id of ["cm_name", "cm_tmax", "cm_eps"]) {
      $(id).addEventListener("change", () => {
        const t = parseFloat($("cm_tmax").value), e = parseFloat($("cm_eps").value);
        if (!Number.isFinite(t) || !Number.isFinite(e) || e < 0.02 || e > 1) return;
        T.material = { name: $("cm_name").value || "Custom material", t_max_c: t, emissivity: e };
        syncMaterial();
        ctx.changed();
      });
    }
    const seg = (id, key, after) => {
      const box = $(id);
      const sync = () => box.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === T[key]));
      box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { T[key] = b.dataset.v; sync(); after && after(); ctx.changed(); }));
      sync();
    };
    const wallRow = () => $("twall-row").classList.toggle("hidden", T.wall !== "fixed");
    seg("h_wall", "wall", wallRow);
    wallRow();
    const trRow = () => $("retr-row").classList.toggle("muted", T.transition !== "auto");
    seg("h_transition", "transition", trRow);
    trRow();
    $("h_twall").value = T.t_wall_c;
    $("h_twall").addEventListener("change", (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) { T.t_wall_c = v; ctx.changed(); } });
    $("h_retr").value = T.re_transition;
    $("h_retr").addEventListener("change", (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v) && v >= 1e4) { T.re_transition = v; ctx.changed(); } });
    $("h_rn").value = T.nose_radius_mm || "";
    $("h_rn").addEventListener("change", (e) => { const v = parseFloat(e.target.value); T.nose_radius_mm = Number.isFinite(v) && v > 0 ? v : null; ctx.changed(); });
    syncMaterial();
    this.info();
  }

  points() {
    const E = this.ctx.S.hyper.envelope;
    return { machs: pts(E.mach_min, E.mach_max, E.mach_count, 2), aoas: pts(E.aoa_min, E.aoa_max, E.aoa_count) };
  }

  info(bodyLength = null) {
    const E = this.ctx.S.hyper.envelope;
    const { machs, aoas } = this.points();
    document.getElementById("mach-chips").innerHTML = machs.map((m) => `<span>M ${m}</span>`).join("");
    document.getElementById("haoa-chips").innerHTML = aoas.map((a) => `<span>${a}°</span>`).join("");
    const air = us76(E.altitude);
    const Mx = Math.max(...machs), Mn = Math.min(...machs);
    const T0 = air.T * (1 + 0.2 * Mx * Mx);
    const q = 0.5 * air.rho * (Mx * air.a) ** 2;
    const reL = (air.rho * Mx * air.a * (bodyLength || 1)) / air.mu;
    let html = `<span class="k">Airspeed</span><span class="v">${Math.round(Mn * air.a)} – ${Math.round(Mx * air.a)} m/s</span>
      <span class="k">Stagnation temperature</span><span class="v" style="${T0 > 1500 ? "color:var(--critical-text)" : T0 > 600 ? "color:var(--warning-text)" : ""}">${degC(T0)}</span>
      <span class="k">Dynamic pressure</span><span class="v">${q > 1e5 ? sig(q / 1e5, 3) + " bar" : sig(q / 1e3, 3) + " kPa"}</span>
      <span class="k">Reynolds number</span><span class="v">${reL.toExponential(1)}${bodyLength ? "" : " per m"}</span>
      <span class="k">Conditions</span><span class="v">${machs.length * aoas.length} <span class="muted">(${aoas.length} α × ${machs.length} Mach)</span></span>`;
    if (Mx > 8) html += `<span class="k" style="grid-column:1/-1;color:var(--critical-text)">Above Mach 8 real-gas chemistry (dissociation) is significant; perfect-gas temperatures are over-estimated and heating is approximate.</span>`;
    else if (Mx > 6) html += `<span class="k" style="grid-column:1/-1;color:var(--warning-text)">Above Mach 6 real-gas effects start to matter; treat wall temperatures as approximate.</span>`;
    if (E.altitude > 60000) html += `<span class="k" style="grid-column:1/-1;color:var(--warning-text)">Above ~60 km the air is thin enough for rarefaction effects the continuum methods ignore.</span>`;
    document.getElementById("hyper-kv").innerHTML = html;
    const n = machs.length * aoas.length;
    document.getElementById("s-hyper-env").textContent = `${n} condition${n > 1 ? "s" : ""}`;
    document.getElementById("s-hyper-env").title = `${aoas.length} angles × ${machs.length} Mach numbers at ${E.altitude / 1000} km`;
  }
}

// ---- results --------------------------------------------------------------------------------------
/** Decode one Mach number's surface fields and the margin to each face's material limit. */
export function hyperFaceFields(fields, k, result) {
  const s = fields.surface[k];
  const r = fields.ranges;
  const cp = dequantize(b64bytes(s.cp), r.cp[0], r.cp[1]);
  const tw = dequantize(b64bytes(s.tw), r.tw[0], r.tw[1]);
  const q = dequantize(b64bytes(s.q), r.q[0], r.q[1]);
  const mi = b64bytes(fields.material_index);
  const lim = result.setup.materials.map((m) => m.t_max_c + 273.15);
  const margin = new Float32Array(tw.length);
  for (let i = 0; i < tw.length; i++) margin[i] = lim[mi[i]] - tw[i];
  return { cp, tw, q, margin, twRange: r.tw, qMax: r.q[1] };
}

const kpi = (label, value, sub = "") => `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
function delta(a, b, lowerBetter = true) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a) < 1e-12) return "";
  const d = (b - a) / Math.abs(a);
  if (Math.abs(d) < 0.0005) return "";
  return `<span class="delta ${(lowerBetter ? d < 0 : d > 0) ? "good" : "bad"}">${fmtPct(d, 1, true)}</span>`;
}

export function hyperResultsHTML(result, cond, display, jobId) {
  const env = result.envelope, st = result.setup;
  const opt = result.kind === "optimize";
  const base = result.baseline.polar, best = opt ? result.best.polar : null;
  const sb = base[cond.a].speeds[cond.v], so = best ? best[cond.a].speeds[cond.v] : null;
  const s = so && display !== "baseline" ? so : sb;
  const sum = (best && display !== "baseline" ? result.best : result.baseline).summary;
  const mats = st.materials;
  const limTxt = mats.map((m) => `${esc(m.name)} ${fmtInt(m.t_max_c)} °C`).join(" · ");
  const L = st.body_length;
  const xcp = sum.x_cp !== undefined ? (sum.x_cp - st.x_nose) / L : null;
  const over = s.over_limit_area > 0;
  const cdTot = Math.max(s.cd_wave, 0) + Math.max(s.cd_base, 0) + Math.max(s.cd_friction, 0) || 1;
  return `
    <div class="row spread"><div><div class="label" style="margin:0">At ${env.aoas[cond.a]}° and Mach ${env.machs[cond.v]} · ${sig(env.altitude / 1000, 3)} km</div>
      <div class="hero-num">${sig(s.cd, 3)} <span class="small muted" style="font-size:15px;font-weight:600">Cd</span></div></div>
      ${opt ? `<span class="badge accent">${display === "baseline" ? "Baseline" : "Optimized"}</span>` : `<span class="badge heat">${icon("flame", "i sm")}Supersonic</span>`}</div>
    <div class="kpis" style="margin-top:12px">
      ${kpi("Drag", fmtForce(s.drag_N), so ? delta(sb.drag_N, so.drag_N) + " vs baseline" : `${sig(s.v, 4)} m/s`)}
      ${kpi("Peak wall temperature", `<span style="color:var(${over ? "--critical-text" : s.margin_K < 50 ? "--warning-text" : "--text"})">${degC(s.tw_max)}</span>`,
        over ? `a limit is exceeded by ${Math.round(-s.margin_K)} K somewhere` : `tightest spot ${Math.round(s.margin_K)} K below its limit`)}
      ${Math.abs(s.cl) < 1e-4 ? kpi("Lift", "≈ 0", "symmetric body at this angle")
        : kpi("Lift", fmtForce(s.lift_N), `CL ${sig(s.cl, 3)}${s.ld ? ` · L/D ${sig(s.ld, 3)}` : ""}`)}
      ${kpi("Stagnation heating", kW(s.q_stag), `cold wall · nose R ${sig(s.nose_radius * 1000, 3)} mm`)}
      ${kpi("Centre of pressure", xcp !== null ? `${fmtPct(xcp, 0)} of length` : "—", xcp !== null ? "from the nose (needs α ≠ 0)" : "sweep a non-zero angle")}
      ${kpi("Heat load", s.heat_load_W >= 1e6 ? sig(s.heat_load_W / 1e6, 3) + " MW" : sig(s.heat_load_W / 1e3, 3) + " kW", `peak flux ${kW(s.q_max)}`)}
    </div>
    <div class="notice ${over ? "bad" : "good"}" style="margin-top:10px">${icon(over ? "alert" : "thermo")}<div>${over
      ? `<b style="color:var(--text)">Over the material limit</b> on ${fmtPct(s.over_limit_area, 1)} of the surface at this condition (${limTxt}). Switch the Margin view on to see where.`
      : `Within the material limits at this condition (${limTxt}); tightest margin ${Math.round(s.margin_K)} K.`}
      ${sum.over_limit && !over ? ` Elsewhere in the envelope the limit is exceeded: hottest at ${sum.tw_max_at.aoa}°, Mach ${sum.tw_max_at.mach} (${degC(sum.tw_max)}).` : ""}</div></div>
    <div class="section-title">Drag breakdown at this condition</div>
    <div class="split-bar" role="img" aria-label="Wave, base and friction drag">
      <div style="width:${(Math.max(s.cd_wave, 0) / cdTot) * 100}%;background:var(--series-1)"></div>
      <div style="width:${(Math.max(s.cd_base, 0) / cdTot) * 100}%;background:var(--heat)"></div>
      <div style="width:${(Math.max(s.cd_friction, 0) / cdTot) * 100}%;background:var(--series-2)"></div></div>
    <div class="legend" style="margin-top:8px"><span><span class="sw" style="background:var(--series-1)"></span>Wave Cd ${sig(s.cd_wave, 3)}</span>
      <span><span class="sw" style="background:var(--heat)"></span>Base Cd ${sig(s.cd_base, 3)}</span>
      <span><span class="sw" style="background:var(--series-2)"></span>Friction Cd ${sig(s.cd_friction, 3)}</span></div>
    <div class="section-title">Across the envelope</div>
    <div class="charts" id="charts"></div>
    <div class="section-title">At ${env.aoas[cond.a]}° by Mach number</div>
    <div style="overflow-x:auto"><table class="tbl compact"><thead><tr><th class="r">Mach</th><th class="r">V</th><th class="r">Cd</th><th class="r">CL</th><th class="r">Peak wall</th><th class="r">Margin</th></tr></thead><tbody>
      ${((best && display !== "baseline") ? best : base)[cond.a].speeds.map((q, i) => `<tr class="${i === cond.v ? "sel" : ""}"><td class="r">${q.mach}</td><td class="r">${Math.round(q.v)} m/s</td>
        <td class="r">${sig(q.cd, 3)}</td><td class="r">${sig(q.cl, 3)}</td><td class="r">${degC(q.tw_max)}</td>
        <td class="r" style="color:var(${q.margin_K < 0 ? "--critical-text" : q.margin_K < 50 ? "--warning-text" : "--good-text"})">${q.margin_K < 0 ? "−" : "+"}${Math.round(Math.abs(q.margin_K))} K</td></tr>`).join("")}</tbody></table></div>
    <div class="notice" style="margin-top:12px">${icon("info")}<div class="small">Engineering surface methods (tangent-cone/wedge, modified Newtonian, Prandtl–Meyer,
      Eckert reference temperature, Fay–Riddell), not a flow solver. Checked against sphere drag, linear supersonic theory and stagnation-heating correlations;
      no shock interactions or real-gas chemistry. See the <a href="/validation#supersonic" target="_blank">validation page</a>.</div></div>
    <details style="margin-top:14px"><summary class="label" style="cursor:pointer">Run details</summary><div class="kv" style="margin-top:8px">
      <span class="k">Method</span><span class="v">surface panels (${fmtInt(result.baseline.polar.length * env.machs.length)} conditions)</span>
      <span class="k">Altitude</span><span class="v">${sig(env.altitude / 1000, 3)} km · ${sig(st.air.T - 273.15, 3)} °C · ${sig(st.air.p / 1000, 3)} kPa</span>
      <span class="k">Wall model</span><span class="v">${st.thermal.wall === "fixed" ? `fixed at ${degC(st.thermal.t_wall_k)}` : "radiative equilibrium"}</span>
      <span class="k">Boundary layer</span><span class="v">${st.thermal.transition === "auto" ? `transition at Re<sub>x</sub> ${sig(st.thermal.re_transition, 2)}` : st.thermal.transition}</span>
      <span class="k">Reference area / length</span><span class="v">${sig(st.ref_area, 4)} m² / ${sig(st.ref_length, 4)} m</span>
      <span class="k">Methods by area</span><span class="v">${Object.entries(s.methods).filter(([, v]) => v > 0.005).map(([k, v]) => `${k.replace("_", " ")} ${fmtPct(v, 0)}`).join(", ")}</span>
      <span class="k">Evaluations</span><span class="v">${fmtInt(result.evaluations)} in ${sig(result.elapsed_s, 2)} s</span></div></details>
    <div class="row wrap" style="margin-top:16px">${!opt ? `<button class="btn sm" id="go-opt">${icon("sparkles")}Optimize this design</button>` : ""}
      <a class="btn sm ghost" href="/app/report/${jobId}" target="_blank">${icon("file")}Report</a></div>`;
}

export function hyperCharts(card, result, cond, display) {
  const env = result.envelope;
  const opt = result.kind === "optimize";
  const base = result.baseline.polar, best = opt ? result.best.polar : null;
  const sel = best && display !== "baseline" ? best : base;
  const mx = (polar, key) => polar[cond.a].speeds.map((s) => [s.mach, s[key]]);
  if (env.machs.length > 1) {
    const ser = [{ name: "Total", color: "var(--series-1)", points: mx(sel, "cd"), dots: true },
      { name: "Wave", color: "var(--accent-ink)", points: mx(sel, "cd_wave"), faded: true },
      { name: "Base", color: "var(--heat)", points: mx(sel, "cd_base"), faded: true },
      { name: "Friction", color: "var(--series-2)", points: mx(sel, "cd_friction"), faded: true }];
    if (best) ser.push({ name: "Baseline total", color: "var(--muted)", points: mx(base, "cd"), faded: true });
    lineChart(card("Drag coefficient vs Mach", `${env.aoas[cond.a]}°`), ser,
      { height: 160, xName: "M", xLabel: "Mach number", fmtY: (v) => sig(v, 3), highlight: { x: env.machs[cond.v], y: sel[cond.a].speeds[cond.v].cd } });
    const colors = ["var(--heat)", "var(--material)", "var(--series-3)", "var(--accent)", "var(--protect)"];
    const tw = [];
    result.setup.materials.forEach((m, k) => {
      const pts = sel[cond.a].speeds.filter((s) => s.tw_max_by_material && s.tw_max_by_material[k] !== null && s.tw_max_by_material[k] !== undefined)
        .map((s) => [s.mach, s.tw_max_by_material[k] - 273.15]);
      if (!pts.length && k > 0) return;
      tw.push({ name: m.name, color: colors[k % colors.length], points: pts.length ? pts : sel[cond.a].speeds.map((s) => [s.mach, s.tw_max - 273.15]), dots: true });
      tw.push({ name: `limit ${fmtInt(m.t_max_c)} °C`, color: colors[k % colors.length], points: env.machs.map((M) => [M, m.t_max_c]), dashed: true });
    });
    lineChart(card("Peak wall temperature by material", "°C · dashed: each material's limit"), tw,
      { height: 160, xName: "M", xLabel: "Mach number", fmtY: (v) => `${Math.round(v)}°` });
  }
  if (env.aoas.length > 1) {
    const cl = [{ name: best ? "Optimized" : "This design", color: "var(--series-1)", points: sel.map((p) => [p.aoa, p.speeds[cond.v].cl]), dots: true }];
    if (best) cl.push({ name: "Baseline", color: "var(--muted)", points: base.map((p) => [p.aoa, p.speeds[cond.v].cl]), faded: true });
    lineChart(card("Lift coefficient vs angle", `Mach ${env.machs[cond.v]}`), cl,
      { height: 150, xName: "α", fmtX: (v) => `${+v.toFixed(1)}°`, fmtY: (v) => sig(v, 3), highlight: { x: env.aoas[cond.a], y: sel[cond.a].speeds[cond.v].cl } });
    lineChart(card("Pitching moment vs angle", `Mach ${env.machs[cond.v]} · about the body centre`),
      [{ name: "Cm", color: "var(--series-3)", points: sel.map((p) => [p.aoa, p.speeds[cond.v].cm_pitch]), dots: true }],
      { height: 150, xName: "α", fmtX: (v) => `${+v.toFixed(1)}°`, fmtY: (v) => sig(v, 3) });
  }
}

export function hyperHistoryHTML(r, cond) {
  const env = r.envelope;
  const b = r.baseline.polar[cond.a].speeds[cond.v], o = r.best.polar[cond.a].speeds[cond.v];
  const bs = r.baseline.summary, os = r.best.summary;
  const good = r.objective === "peak_temp" ? os.tw_max < bs.tw_max : r.drag_change < 0;
  const head = r.objective === "peak_temp" ? `${r.tw_change_K <= 0 ? "−" : "+"}${Math.round(Math.abs(r.tw_change_K))} K` : fmtPct(r.drag_change, 1, true);
  const tr = (label, a, c, f, lower = true) => {
    const d = Number.isFinite(a) && Number.isFinite(c) && Math.abs(a) > 1e-12 ? (c - a) / Math.abs(a) : NaN;
    const ok = lower ? d < 0 : d > 0;
    return `<tr><td>${label}</td><td class="r">${f(a)}</td><td class="r">${f(c)}</td><td class="r ${!Number.isFinite(d) || Math.abs(d) < 0.0005 ? "" : ok ? "delta good" : "delta bad"}">${Number.isFinite(d) ? fmtPct(d, 1, true) : "—"}</td></tr>`;
  };
  return `<div class="label">${esc(r.objective_label)}${r.respect_limits ? " · within material limits" : ""}</div>
    <div class="hero-num">${head}</div>
    <div class="small muted" style="margin:4px 0 12px">${r.objective === "peak_temp" ? "change in the hottest wall temperature in the envelope" : `average change in drag across all ${env.aoas.length * env.machs.length} conditions`}
      · best design #${r.best_index} of ${r.history.length - 1}</div>
    ${!good ? `<div class="notice warn" style="margin-bottom:12px">${icon("alert")}<div>No design beat the baseline within the allowed changes${r.respect_limits && bs.over_limit ? " and the material limits" : ""}. Try more iterations, a larger change, or other aspects.</div></div>` : ""}
    ${os.over_limit ? `<div class="notice bad" style="margin-bottom:12px">${icon("flame")}<div>The best design still exceeds a material limit somewhere in the envelope (${degC(os.tw_max)} at Mach ${os.tw_max_at.mach}). A blunter nose, a lower Mach number or a hotter material is needed.</div></div>` : ""}
    <table class="tbl compact"><thead><tr><th>At ${env.aoas[cond.a]}°, Mach ${env.machs[cond.v]}</th><th class="r">Baseline</th><th class="r">Optimized</th><th class="r">Change</th></tr></thead><tbody>
      ${tr("Drag", b.drag_N, o.drag_N, fmtForce)}
      ${tr("Cd", b.cd, o.cd, (v) => sig(v, 3))}
      ${tr("Peak wall", b.tw_max - 273.15, o.tw_max - 273.15, (v) => `${Math.round(v)} °C`)}
      ${tr("Hottest in envelope", bs.tw_max - 273.15, os.tw_max - 273.15, (v) => `${Math.round(v)} °C`)}
      ${tr("Mean drag in envelope", bs.drag_mean_N, os.drag_mean_N, fmtForce)}
      <tr><td>Volume</td><td class="r">100%</td><td class="r">${fmtPct(r.volume_ratio, 1)}</td><td class="r">${fmtPct(r.volume_ratio - 1, 1, true)}</td></tr></tbody></table>`;
}
