// Professional result cards: grid convergence, nested refinement, structure, mission, the
// trade-off chart, exports (PDF / STEP / VTK / Fusion) and run comparison.
import { api, confirmDialog, esc, fmtForce, icon, sig, toast } from "/static/js/core.js";
import { lineChart } from "/static/js/charts.js";

const pct = (v, d = 1) => `${(100 * v).toFixed(d)} %`;

// ---- grid convergence and the zoom grid -----------------------------------------------------------------
export function accuracyHTML(result) {
  const out = [];
  const gs = result.grid_study;
  if (gs && gs.cd) {
    const g = gs.cd;
    const ok = !g.note && g.gci_fine < 0.05;
    out.push(`<div class="notice ${ok ? "good" : "warn"}" style="margin-top:10px">${icon(ok ? "check" : "alert")}<div>
      <b style="color:var(--text)">Grid convergence</b> (at ${gs.aoa}°, ${gs.speed} m/s): drag coefficient
      ${g.values.slice().reverse().map((v) => sig(v, 4)).join(" → ")} on grids of ${gs.cells.slice().reverse().map((c) => (c / 1e6).toFixed(2) + "M").join(" / ")} cells.
      Grid-independent estimate <b style="color:var(--text)">${sig(g.extrapolated, 4)}</b>, uncertainty on this grid
      <b style="color:var(--text)">±${pct(g.gci_fine)}</b> (GCI${g.order ? `, observed order ${sig(g.order, 2)}` : ""}).
      ${g.note ? `<span class="small">${esc(g.note)}.</span>` : ""}</div></div>`);
  }
  const z = result.setup && result.setup.zoom;
  if (z) {
    out.push(`<div class="notice" style="margin-top:10px">${icon("info")}<div class="small"><b style="color:var(--text)">Refined near the body.</b>
      Forces come from a nested grid ${sig(z.refinement, 3)}× finer around the body (${(z.cells / 1e6).toFixed(2)}M cells, body
      ${z.body_cells.map((v) => Math.round(v)).join(" × ")} cells), driven by the full tunnel's mean flow.</div></div>`);
  }
  const gap = result.setup && (result.setup.zoom ? null : result.setup.ground_gap_cells);
  if (gap !== null && gap !== undefined && gap > 0 && gap < 3) {
    out.push(`<div class="notice warn" style="margin-top:10px">${icon("alert")}<div><b style="color:var(--text)">The gap under the body is only ${sig(gap, 2)} cells.</b>
      The flow between the road and the underbody is not resolved at this fidelity, which upsets the forces. Use
      <b>Refine near the body</b> or a higher fidelity (at least 3 cells of ground clearance).</div></div>`);
  }
  if (result.setup && result.setup.ground) {
    out.push(`<div class="notice" style="margin-top:10px">${icon("info")}<div class="small"><b style="color:var(--text)">Rolling road.</b>
      The floor moves with the air, as on a moving-belt rig${result.setup.wheels ? `, and ${result.setup.wheels} wheel(s) spin` : ""}.</div></div>`);
  }
  return out.join("");
}

// ---- downloads ------------------------------------------------------------------------------------------------
export function exportsHTML(result, job, me, fusion) {
  const winged = !!(job.spec.template && ["uav", "wing"].includes(job.spec.template.id));
  const hasFlow = !!result.fields && result.tunnel !== "supersonic";
  const canExport = me.limits.export;
  const tm = !!job.spec.template;
  return `<div class="section-title">Export</div>
    <div class="row wrap" style="gap:8px">
      <a class="btn sm" href="/api/jobs/${job.id}/export/pdf" download>${icon("file")}PDF report</a>
      ${canExport ? `<a class="btn sm ghost" href="/api/jobs/${job.id}/export/step" download title="${tm ? "Smooth CAD solids through the design's own sections" : "A faceted solid of the design"}">${icon("box")}STEP (CAD)</a>`
        : `<span class="btn sm ghost" aria-disabled="true" title="Pro and Team">${icon("lock")}STEP (Pro)</span>`}
      ${hasFlow && !winged ? `<a class="btn sm ghost" href="/api/jobs/${job.id}/export/vtk" download title="Mean flow and surface pressure for ParaView">${icon("wind")}Flow fields (VTK)</a>` : ""}
      ${tm ? `<a class="btn sm ghost" href="/api/jobs/${job.id}/export/fusion-script" download title="A Fusion script that writes the dimensions as user parameters">${icon("code")}Fusion script</a>` : ""}
    </div>
    ${fusion && fusion.enabled ? `<div class="row wrap" style="gap:8px;margin-top:8px">
      ${tm ? `<button class="btn sm" data-fusion="parameters" ${fusion.ok ? "" : "disabled"}>${icon("send")}Send dimensions to Fusion</button>` : ""}
      <button class="btn sm ghost" data-fusion="import" ${fusion.ok && canExport ? "" : "disabled"}>${icon("download")}Import into Fusion</button>
      <span class="small muted">${fusion.ok ? `Fusion: ${esc(fusion.document || "open")}` : esc(fusion.why || "Fusion is not running")}</span></div>` : ""}`;
}

export function wireExports(box, job) {
  box.querySelectorAll("[data-fusion]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    try {
      const r = await api(`/api/jobs/${job.id}/fusion`, { method: "POST", body: { action: b.dataset.fusion } });
      if (r.updated) toast(`Fusion: ${r.updated.length} user parameter(s) set in ${r.document || "the open design"}.`, "good");
      else toast(`Imported into ${r.document || "Fusion"}.`, "good");
    } catch (e) { toast(e.message || String(e), "bad"); }
    b.disabled = false;
  }));
}

// ---- structure and mission (winged designs) ----------------------------------------------------------------
export function structureHTML(result) {
  const st = result.structure;
  if (!st) return "";
  const b = st.base, o = st.best || st.base;
  const w0 = b.surfaces[0], w1 = o.surfaces[0];
  const g = (kg) => `${Math.round(kg * 1000)} g`;
  return `<div class="section-title">Structure <span class="small muted">beam model · ${esc(b.material)} · n = ${b.load_factor} × ${b.safety_factor}</span></div>
    <table class="tbl compact"><thead><tr><th></th><th class="r">Original</th><th class="r">Optimized</th></tr></thead><tbody>
      <tr><td>Wing + tail structure</td><td class="r">${g(b.mass_kg)}</td><td class="r">${g(o.mass_kg)}</td></tr>
      <tr><td>Main spar (sized by)</td><td class="r">${g(w0.spar_kg)} (${w0.sized_by})</td><td class="r">${g(w1.spar_kg)} (${w1.sized_by})</td></tr>
      <tr><td>Tip deflection at limit load</td><td class="r">${pct(w0.tip_deflection_frac, 0)}</td><td class="r">${pct(w1.tip_deflection_frac, 0)}</td></tr>
      <tr><td>Spar caps fill of the depth</td><td class="r">${pct(w0.max_cap_fill, 0)}</td><td class="r">${pct(w1.max_cap_fill, 0)}</td></tr>
      <tr><td>Root bending moment (ultimate)</td><td class="r">${sig(w0.root_moment_Nm, 3)} N·m</td><td class="r">${sig(w1.root_moment_Nm, 3)} N·m</td></tr>
    </tbody></table>
    <p class="small muted" style="margin-top:6px">Spar caps at the airfoil's depth sized for strength, then grown until the tip bends no more than allowed; skin by
      area. A preliminary sizing — not a substitute for a structural analysis of the real layup.</p>`;
}

export function missionHTML(result) {
  const ms = result.mission;
  if (!ms) return "";
  const b = ms.base, o = ms.best;
  const inp = b.inputs;
  const cell = (m, f) => (m ? f(m) : "—");
  const rows = [
    ["Stall speed", (m) => `${sig(m.stall_speed, 3)} m/s`],
    ["Best-endurance speed", (m) => `${sig(m.best_endurance.v, 3)} m/s · ${sig(m.best_endurance.power_W, 3)} W`],
    ["Endurance", (m) => `<b>${Math.round(m.best_endurance.endurance_min)} min</b>`],
    ["Best-range speed", (m) => `${sig(m.best_range.v, 3)} m/s · L/D ${sig(m.best_range.ld, 3)}`],
    ["Range", (m) => `<b>${sig(m.best_range.range_km, 3)} km</b>`],
    ["Climb rate", (m) => (m.climb_m_s === null ? "—" : `${sig(m.climb_m_s, 2)} m/s`)],
  ];
  return `<div class="section-title">Mission <span class="small muted">at ${sig(b.mass_kg, 3)} kg</span></div>
    <div class="row wrap mission-in" style="gap:8px;margin-bottom:8px">
      <label class="field sm"><span>Battery (Wh)</span><input type="number" id="ms-wh" value="${inp.battery_wh}" min="1" step="1"></label>
      <label class="field sm"><span>Usable</span><input type="number" id="ms-use" value="${inp.usable}" min="0.1" max="1" step="0.05"></label>
      <label class="field sm"><span>Efficiency</span><input type="number" id="ms-eff" value="${inp.efficiency}" min="0.1" max="0.95" step="0.05"></label>
      <label class="field sm"><span>Motor (W)</span><input type="number" id="ms-motor" value="${inp.motor_w}" min="0" step="10"></label>
      <button class="btn sm" id="ms-go" style="align-self:flex-end">${icon("refresh")}Recompute</button></div>
    <table class="tbl compact" id="ms-table"><thead><tr><th></th><th class="r">Original</th>${o ? `<th class="r">Optimized</th>` : ""}</tr></thead><tbody>
      ${rows.map(([l, f]) => `<tr><td>${l}</td><td class="r">${cell(b, f)}</td>${o ? `<td class="r">${cell(o, f)}</td>` : ""}</tr>`).join("")}</tbody></table>
    <div class="chart-card" id="ms-chart" style="margin-top:8px"></div>`;
}

function drawMission(box, ms) {
  const c = box.querySelector("#ms-chart");
  if (!c || !ms) return;
  c.innerHTML = `<div class="ct">Power required at the flight weight</div><div class="cc"></div>`;
  const ser = [{ name: "Original", color: "var(--series-1)", points: ms.base.curve.map((r) => [r.v, r.power_W]), faded: !!ms.best }];
  if (ms.best) ser.push({ name: "Optimized", color: "var(--series-3)", points: ms.best.curve.map((r) => [r.v, r.power_W]) });
  lineChart(c.querySelector(".cc"), ser, { height: 160, xName: "V", xLabel: "airspeed (m/s)", fmtY: (v) => `${sig(v, 3)} W` });
}

export function wireMission(box, result, job) {
  drawMission(box, result.mission);
  const go = box.querySelector("#ms-go");
  if (!go) return;
  go.addEventListener("click", async () => {
    go.disabled = true;
    try {
      const body = { battery_wh: +box.querySelector("#ms-wh").value, usable: +box.querySelector("#ms-use").value,
        efficiency: +box.querySelector("#ms-eff").value, motor_w: +box.querySelector("#ms-motor").value,
        mass_kg: result.mission.base.mass_kg };
      const ms = await api(`/api/jobs/${job.id}/mission`, { method: "POST", body });
      result.mission = ms;
      const holder = box.querySelector("#ms-table").parentElement;
      const tmp = document.createElement("div");
      tmp.innerHTML = missionHTML(result);
      box.querySelector("#ms-table").replaceWith(tmp.querySelector("#ms-table"));
      drawMission(box, ms);
      void holder;
    } catch (e) { toast(e.message || String(e), "bad"); }
    go.disabled = false;
  });
}

// ---- trade-off chart: drag against wing mass (and stall margin) of every design --------------------------
function paretoFront(pts) {
  // non-dominated in (objective down, mass down)
  return pts.filter((p) => !pts.some((q) => q !== p && q.y <= p.y && q.x <= p.x && (q.y < p.y || q.x < p.x)));
}

export function tradeoffHTML(result) {
  const h = (result.history || []).filter((e) => e.objective !== null && e.wing_mass_kg !== undefined && e.wing_mass_kg !== null);
  if (h.length < 4) return "";
  return `<div class="section-title">Trade-off: drag against structure mass</div>
    <div class="chart-card"><div class="ct">Every design evaluated · the front (filled) cannot be beaten on both · click a point to open it</div>
    <div id="tradeoff"></div></div>`;
}

export function wireTradeoff(box, result, job) {
  const holder = box.querySelector("#tradeoff");
  if (!holder) return;
  const pts = (result.history || []).filter((e) => e.objective !== null && e.wing_mass_kg !== undefined && e.wing_mass_kg !== null && e.status !== "rejected")
    .map((e) => ({ x: e.wing_mass_kg * 1000, y: e.objective, i: e.i, best: !!e.best, stall: e.stall_margin }));
  if (!pts.length) return;
  const front = new Set(paretoFront(pts));
  const W = 360, H = 200, m = { l: 44, r: 10, t: 8, b: 30 };
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const fr = [...front].sort((a, b) => a.x - b.x);
  holder.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Drag against wing mass">
    <line x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}" stroke="var(--border)"/>
    <line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="var(--border)"/>
    <text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle" class="ax">structure mass (g)</text>
    <text x="10" y="${(H - m.b) / 2}" transform="rotate(-90 10 ${(H - m.b) / 2})" text-anchor="middle" class="ax">drag (1 = original)</text>
    <text x="${m.l}" y="${H - m.b + 13}" class="ax">${Math.round(x0)}</text><text x="${W - m.r}" y="${H - m.b + 13}" text-anchor="end" class="ax">${Math.round(x1)}</text>
    <text x="${m.l - 4}" y="${H - m.b}" text-anchor="end" class="ax">${sig(y0, 3)}</text><text x="${m.l - 4}" y="${m.t + 8}" text-anchor="end" class="ax">${sig(y1, 3)}</text>
    <polyline fill="none" stroke="var(--series-3)" stroke-width="1.5" stroke-dasharray="4 3" points="${fr.map((p) => `${X(p.x)},${Y(p.y)}`).join(" ")}"/>
    ${pts.map((p) => `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.best ? 6 : 4}" data-i="${p.i}" style="cursor:pointer"
      fill="${front.has(p) ? "var(--series-3)" : "var(--surface-1)"}" stroke="${p.best ? "var(--text)" : "var(--series-3)"}" stroke-width="${p.best ? 2 : 1.2}">
      <title>Design #${p.i}: drag ${sig(p.y, 4)}, structure ${Math.round(p.x)} g${p.stall !== undefined ? `, stall margin ${sig(p.stall, 2)}` : ""}${p.best ? " (chosen)" : ""}</title></circle>`).join("")}
  </svg>`;
  holder.querySelectorAll("circle[data-i]").forEach((c) => c.addEventListener("click", async () => {
    const i = +c.dataset.i;
    if (!(await confirmDialog(`Open design #${i} as a new project?`, "It starts a new project from this design's dimensions, so you can analyze or refine it."))) return;
    try {
      const p = await api(`/api/jobs/${job.id}/promote`, { method: "POST", body: { index: i } });
      location.href = `/app/project/${p.id}`;
    } catch (e) { toast(e.message || String(e), "bad"); }
  }));
}

// ---- run comparison --------------------------------------------------------------------------------------------
export async function compareRuns(box, a, b) {
  const [ra, rb] = await Promise.all([a, b].map((j) => api(`/api/jobs/${j.id}/files/result.json`)));
  const pol = (r) => (r.best || r.baseline).polar;
  const pa = pol(ra), pb = pol(rb);
  const rows = [];
  pa.forEach((qa, k) => {
    const qb = pb.find((q) => q.aoa === qa.aoa);
    if (!qb) return;
    qa.speeds.forEach((sa, n) => {
      const sb = qb.speeds[n];
      if (sb && Math.abs(sb.v - sa.v) < 1e-6) rows.push({ aoa: qa.aoa, v: sa.v, a: sa, b: sb });
    });
    void k;
  });
  const name = (j) => `${j.kind === "optimize" ? "Optimized" : "Analysis"} · ${j.spec.fidelity} · ${new Date(j.created_at * 1000).toLocaleString()}`;
  box.innerHTML = `<div class="section-title">Comparing two runs</div>
    <p class="small muted">A: ${esc(name(a))}<br>B: ${esc(name(b))}</p>
    ${rows.length ? `<table class="tbl compact"><thead><tr><th class="r">α</th><th class="r">V</th><th class="r">Drag A</th><th class="r">Drag B</th><th class="r">Change</th><th class="r">L/D A → B</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td class="r">${r.aoa}°</td><td class="r">${r.v}</td><td class="r">${fmtForce(r.a.drag_N)}</td><td class="r">${fmtForce(r.b.drag_N)}</td>
        <td class="r" style="color:${r.b.drag_N < r.a.drag_N ? "var(--good-text)" : "var(--critical-text)"}">${(100 * (r.b.drag_N / r.a.drag_N - 1)).toFixed(1)} %</td>
        <td class="r">${r.a.ld === null ? "—" : sig(r.a.ld, 3)} → ${r.b.ld === null ? "—" : sig(r.b.ld, 3)}</td></tr>`).join("")}</tbody></table>
      <div class="chart-card" id="cmp-chart" style="margin-top:10px"></div>`
      : `<p class="small muted">The two runs share no flight condition (angle and speed) to compare.</p>`}
    <button class="btn sm ghost" id="cmp-close" style="margin-top:10px">${icon("x")}Close</button>`;
  if (rows.length) {
    const c = box.querySelector("#cmp-chart");
    c.innerHTML = `<div class="ct">Drag force at the first angle</div><div class="cc"></div>`;
    const a0 = rows[0].aoa;
    const sel = rows.filter((r) => r.aoa === a0);
    lineChart(c.querySelector(".cc"), [{ name: "A", color: "var(--series-1)", points: sel.map((r) => [r.v, r.a.drag_N]), dots: true },
      { name: "B", color: "var(--series-3)", points: sel.map((r) => [r.v, r.b.drag_N]), dots: true }],
    { height: 160, xName: "V", xLabel: "airspeed (m/s)", fmtY: (v) => fmtForce(v) });
  }
}
