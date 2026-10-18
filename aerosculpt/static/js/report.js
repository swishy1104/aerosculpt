import { api, esc, fmtDate, fmtDuration, fmtForce, fmtInt, fmtPct, handleError, icon, requireAuth, sig, $ } from "/static/js/core.js";
import { lineChart } from "/static/js/charts.js";
import { Viewer, parseSTL, rotate, windRotation } from "/static/js/viewer.js";
import { b64bytes, dequantize } from "/static/js/colormap.js";
import { degC, hyperFaceFields } from "/static/js/tunnel-hyper.js";

const me = await requireAuth();
const jid = location.pathname.split("/").filter(Boolean).pop();
const root = $("#report");

async function file(name, kind = "json") {
  const r = await fetch(`/api/jobs/${jid}/files/${name}`, { credentials: "same-origin" });
  if (!r.ok) throw new Error(`missing ${name}`);
  return kind === "json" ? r.json() : parseSTL(await r.arrayBuffer());
}

function polarTable(polar, env) {
  return `<div style="overflow-x:auto"><table class="tbl compact"><thead><tr><th class="r">α</th><th class="r">Cl</th><th class="r">Cm</th>
    ${env.speeds.map((v) => `<th class="r">Cd @ ${v}</th>`).join("")}${env.speeds.map((v) => `<th class="r">L/D @ ${v}</th>`).join("")}</tr></thead><tbody>
    ${polar.map((p) => `<tr><td class="r">${p.aoa}°${p.mirror ? "*" : ""}</td><td class="r">${sig(p.cl, 3)}</td><td class="r">${sig(p.cm_pitch, 3)}</td>
      ${p.speeds.map((s) => `<td class="r">${sig(s.cd, 3)}</td>`).join("")}${p.speeds.map((s) => `<td class="r">${s.ld === null ? "—" : sig(s.ld, 3)}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`;
}

async function render() {
  const job = await api(`/api/jobs/${jid}`);
  const project = await api(`/api/projects/${job.project_id}`);
  if (job.status !== "done") {
    root.innerHTML = `<h1>Report not ready</h1><p>This run is ${esc(job.status)}.</p>`;
    return;
  }
  const r = await file("result.json");
  if (!r.envelope) {
    root.innerHTML = `<h1>Older run</h1><p>This run predates flight-envelope reports. Re-run it to get a report.</p>`;
    return;
  }
  if (r.tunnel === "supersonic") return renderHyper(job, project, r);
  const st = r.setup, env = r.envelope;
  const opt = r.kind === "optimize";
  const bs = r.baseline.summary, os = opt ? r.best.summary : null;
  const locked = !me.limits.reports;
  const ia = env.aoas.reduce((bi, a, i) => (Math.abs(a) < Math.abs(env.aoas[bi]) ? i : bi), 0);
  const iv = Math.floor((env.speeds.length - 1) / 2);
  root.innerHTML = `
    <div class="row spread no-print" style="margin-bottom:18px">
      <a href="/app/project/${project.id}">&larr; Back to project</a>
      <button class="btn primary" id="print" ${locked ? "disabled" : ""}>${icon("download")}Print / save as PDF</button></div>
    ${locked ? `<div class="notice warn no-print" style="margin-bottom:18px">${icon("lock")}<div>Printable reports are part of Pro.
      <a href="/app/billing">Upgrade</a> to export this report; every number stays available in your workspace.</div></div>` : ""}
    <div class="row" style="gap:10px"><img src="/static/img/logo.svg" width="30" height="30" alt=""><span class="muted">AeroSculpt engineering report</span></div>
    <h1>${esc(project.name)}</h1>
    <p class="muted">${opt ? "Shape optimization" : "Flight-envelope analysis"} · ${fmtDate(job.finished_at)} · job ${esc(job.id)}</p>

    <h2>Summary</h2>
    ${opt ? `<p>${esc(r.objective_label)} over ${r.history.length - 1} designs changed the average drag across all ${env.aoas.length * env.speeds.length} flight
        conditions by <b style="color:var(--text)">${fmtPct(r.drag_change, 1, true)}</b>${bs.ld_max && os.ld_max ? `, and best L/D from ${sig(bs.ld_max, 3)} to ${sig(os.ld_max, 3)}` : ""},
        keeping ${fmtPct(r.volume_ratio, 1)} of the original volume${r.zones.length ? ` and respecting ${r.zones.length} protected region${r.zones.length > 1 ? "s" : ""}` : ""}.
        ${r.verified ? `Re-checked at ${esc(r.verified.fidelity)} fidelity.` : ""}</p>`
      : `<p>Swept ${env.aoas.length} angle${env.aoas.length > 1 ? "s" : ""} (${env.aoas[0]}° to ${env.aoas[env.aoas.length - 1]}°) and ${env.speeds.length} airspeed${env.speeds.length > 1 ? "s" : ""}
        (${env.speeds[0]} to ${env.speeds[env.speeds.length - 1]} m/s). Cd ${sig(bs.cd_at_min_aoa, 3)} at ${bs.aoa_ref}° and ${bs.speed_ref} m/s${bs.ld_max ? `; best L/D ${sig(bs.ld_max, 3)} at ${bs.ld_max_at.aoa}° and ${bs.ld_max_at.v} m/s` : ""}${bs.cm_alpha !== undefined ? `; Cm<sub>α</sub> ${sig(bs.cm_alpha, 3)} /rad (${bs.cm_alpha < 0 ? "statically stable" : "statically unstable"})` : ""}.</p>`}
    ${r.baseline.polar.some((q) => Math.abs(q.cl) > 0.02) ? `<p class="small muted">Lift, L/D and pitching-moment values are indicative only: the solver's
      drag is validated, its lift is not yet (on thin wings it currently comes out 50–66% low; see the validation page).</p>` : ""}
    <div class="shots" id="shots"><div class="skeleton" style="height:220px"></div>${opt ? '<div class="skeleton" style="height:220px"></div>' : ""}</div>

    <h2>Across the envelope</h2>
    <div class="grid-2" id="charts"></div>
    ${r.propellers ? propulsion(r, ia, opt) : ""}

    <h2>${opt ? "Baseline polar" : "Polar"}</h2>${polarTable(r.baseline.polar, env)}
    ${opt ? `<h2>Optimized polar</h2>${polarTable(r.best.polar, env)}
      <h2>Best design</h2><table class="tbl compact"><tbody>${r.variables.map((v) => `<tr><td>${esc(v.label)}</td>
        <td class="r">${v.key === "fairing" ? fmtPct(r.best_values[v.key], 0) : fmtPct(r.best_values[v.key], 1, true)}</td>
        <td class="r muted">range ${fmtPct(v.lo, 0, true)} … ${fmtPct(v.hi, 0, true)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${st.symmetric ? `<p class="small muted">* mirrored from the positive angle (body symmetric top-to-bottom).</p>` : ""}

    <h2>Setup</h2>
    <table class="tbl compact"><tbody>
      <tr><td>Model file</td><td class="r">${esc(project.filename)} (${fmtInt(project.metrics.triangles)} triangles)</td></tr>
      <tr><td>Units / orientation</td><td class="r">${esc(r.model.units)}, nose ${esc(r.model.forward)}, up ${esc(r.model.up)}</td></tr>
      <tr><td>Airspeeds</td><td class="r">${env.speeds.join(", ")} m/s at ${env.altitude} m (ISA, ρ = ${sig(st.air.rho, 4)} kg/m³)</td></tr>
      <tr><td>Angles of attack / sideslip</td><td class="r">${env.aoas.join("°, ")}° / ${env.yaw}°</td></tr>
      <tr><td>Solver regime</td><td class="r">${st.regimes.join(" + ")}</td></tr>
      <tr><td>Fidelity / grid</td><td class="r">${esc(st.fidelity)}, ${st.grid.join(" × ")} = ${fmtInt(st.cells)} cells, dx ${sig(st.dx_m * 1000, 3)} mm</td></tr>
      <tr><td>Reference area / length</td><td class="r">${sig(st.ref_area, 4)} m² / ${sig(st.ref_length, 4)} m</td></tr>
      ${opt ? `<tr><td>Goal / aspects</td><td class="r">${esc(r.objective_label)}; ${r.aspects.map(esc).join(", ")}</td></tr>
      <tr><td>Protected regions</td><td class="r">${r.zones.length ? r.zones.map((z) => `${esc(z.name)} (${esc(z.type)})`).join(", ") : "none"}</td></tr>` : ""}
      <tr><td>Compute</td><td class="r">${r.cfd_runs} CFD runs, ${fmtDuration(r.elapsed_s)}</td></tr>
    </tbody></table>

    <h2>Method and limitations</h2>
    <p class="small">3D lattice-Boltzmann large-eddy simulation (D3Q19, regularized collision with Smagorinsky sub-grid model and
      bulk-viscosity damping) on a uniform voxel grid with interpolated bounce-back walls; forces by momentum exchange, corrected for
      tunnel blockage (Maskell, θ = 2.5). Each angle of attack is one run; forces at each airspeed follow from it.
      ${st.regimes.includes("hybrid") ? "Where the flight Reynolds number exceeds the grid's resolvable range, pressure forces come from the simulation and skin friction is re-scaled to flight Reynolds number with the Prandtl–Schlichting law; this is conservative for highly streamlined bodies." : ""}
      ${opt ? "Optimization by Bayesian optimization (Gaussian-process surrogate, expected improvement) over smooth free-form deformations, with protected and keep-inside regions enforced on every candidate." : ""}
      Results are engineering estimates; validate safety-critical designs with independent analysis or testing.</p>`;
  const pr = $("#print");
  if (pr) pr.addEventListener("click", () => print());

  const charts = $("#charts");
  const card = (t) => { const c = document.createElement("div"); c.className = "chart-card"; c.innerHTML = `<div class="ct">${t}</div>`; const a = document.createElement("div"); c.append(a); charts.append(c); return a; };
  const ser = (fn) => {
    const out = [{ name: opt ? "Baseline" : "Design", color: "var(--series-1)", points: r.baseline.polar.map(fn), dots: true, faded: opt }];
    if (opt) out.push({ name: "Optimized", color: "var(--series-3)", points: r.best.polar.map(fn), dots: true });
    return out;
  };
  if (env.aoas.length > 1) {
    lineChart(card(`Cd vs angle (${env.speeds[iv]} m/s)`), ser((p) => [p.aoa, p.speeds[iv].cd]), { height: 180, xName: "α", fmtX: (v) => v + "°", fmtY: (v) => sig(v, 3) });
    lineChart(card("Cl vs angle"), ser((p) => [p.aoa, p.cl]), { height: 180, xName: "α", fmtX: (v) => v + "°", fmtY: (v) => sig(v, 3) });
  }
  if (env.speeds.length > 1) {
    const drag = (polar) => polar[ia].speeds.map((s) => [s.v, s.drag_N]);
    const s = [{ name: opt ? "Baseline" : "Design", color: "var(--series-1)", points: drag(r.baseline.polar), dots: true, faded: opt }];
    if (opt) s.push({ name: "Optimized", color: "var(--series-3)", points: drag(r.best.polar), dots: true });
    lineChart(card(`Drag vs airspeed (${env.aoas[ia]}°)`), s, { height: 180, xName: "V", xLabel: "m/s", fmtY: (v) => fmtForce(v) });
  }
  if (opt) {
    let best = Infinity;
    lineChart(card("Optimization history"), [
      { name: "Each design", color: "var(--series-1)", points: r.history.filter((e) => e.status !== "diverged").map((e) => [e.i, e.objective]), dots: true, faded: true },
      { name: "Best so far", color: "var(--series-3)", points: r.history.map((e) => { if (e.status !== "diverged") best = Math.min(best, e.objective); return [e.i, best]; }), step: true },
    ], { height: 180, integerX: true, xName: "design", fmtY: (v) => sig(v, 3) });
  }
  await shots(r, ia, opt);
}

function propulsion(r, ia, opt) {
  const row = (opt ? r.best : r.baseline).polar[ia];
  const ts = (opt ? r.best : r.baseline).summary.top_speed;
  return `<h2>Propulsion at ${row.aoa}°</h2>
    <p>${r.propellers.map((p) => `${esc(p.name)}: ${sig(p.diameter_m / 0.0254, 3)}×${sig(p.pitch_m / 0.0254, 3)} in, ${p.blades} blades, ${fmtInt(p.rpm)} rpm (${p.rotation.toUpperCase()})`).join("; ")}.
      ${ts ? (ts.bracketed ? `Estimated level top speed <b>${sig(ts.v, 3)} m/s</b>, where thrust equals drag.` : ts.above ? "Thrust exceeds drag at every airspeed tested." : "Drag exceeds thrust at every airspeed tested.") : ""}</p>
    <table class="tbl compact"><thead><tr><th class="r">V</th><th class="r">Thrust</th><th class="r">Drag</th><th class="r">Net</th><th class="r">Shaft power</th><th class="r">Prop η</th></tr></thead><tbody>
      ${row.speeds.map((s) => `<tr><td class="r">${s.v} m/s</td><td class="r">${fmtForce(s.thrust_N)}</td><td class="r">${fmtForce(s.drag_N)}</td>
        <td class="r">${fmtForce(s.net_N)}</td><td class="r">${sig(s.shaft_power_W, 3)} W</td><td class="r">${fmtPct(s.prop_eta, 0)}</td></tr>`).join("")}</tbody></table>
    <p class="small muted">Propellers: blade-element momentum theory (checked against UIUC data, thrust and power within 8%), applied to the flow as actuator disks;
      the airframe drag is measured inside the slipstream.</p>`;
}

async function renderHyper(job, project, r) {
  const st = r.setup, env = r.envelope;
  const opt = r.kind === "optimize";
  const bs = r.baseline.summary, os = opt ? r.best.summary : null;
  const sum = os || bs;
  const pol = (opt ? r.best : r.baseline).polar;
  const locked = !me.limits.reports;
  const ia = env.aoas.reduce((bi, a, i) => (Math.abs(a) < Math.abs(env.aoas[bi]) ? i : bi), 0);
  const im = env.machs.length - 1;
  const mats = st.materials.map((m) => `${esc(m.name)} (limit ${fmtInt(m.t_max_c)} °C, ε ${sig(m.emissivity, 2)})`).join("; ");
  root.innerHTML = `
    <div class="row spread no-print" style="margin-bottom:18px">
      <a href="/app/project/${project.id}">&larr; Back to project</a>
      <button class="btn primary" id="print" ${locked ? "disabled" : ""}>${icon("download")}Print / save as PDF</button></div>
    ${locked ? `<div class="notice warn no-print" style="margin-bottom:18px">${icon("lock")}<div>Printable reports are part of Pro.
      <a href="/app/billing">Upgrade</a> to export this report; every number stays available in your workspace.</div></div>` : ""}
    <div class="row" style="gap:10px"><img src="/static/img/logo.svg" width="30" height="30" alt=""><span class="muted">AeroSculpt engineering report · Supersonic tunnel</span></div>
    <h1>${esc(project.name)}</h1>
    <p class="muted">${opt ? "Shape optimization" : "Supersonic envelope analysis"} · ${fmtDate(job.finished_at)} · job ${esc(job.id)}</p>
    <h2>Summary</h2>
    <p>Mach ${env.machs[0]} to ${env.machs[im]} at ${sig(env.altitude / 1000, 3)} km, angles ${env.aoas[0]}° to ${env.aoas[env.aoas.length - 1]}°.
      ${opt ? `${esc(r.objective_label)} over ${r.history.length - 1} designs changed the average drag by <b>${fmtPct(r.drag_change, 1, true)}</b> and the hottest wall temperature by
        ${r.tw_change_K <= 0 ? "−" : "+"}${Math.round(Math.abs(r.tw_change_K))} K, keeping ${fmtPct(r.volume_ratio, 1)} of the volume. ` : ""}
      Cd ${sig(sum.cd_at_min_aoa, 3)} at ${sum.aoa_ref}° and Mach ${sum.mach_ref}. Hottest wall ${degC(sum.tw_max)} at ${sum.tw_max_at.aoa}°, Mach ${sum.tw_max_at.mach};
      ${sum.over_limit ? `<b>a material limit is exceeded</b> (worst margin ${Math.round(sum.margin_K)} K).` : `all materials stay within their limits (tightest margin ${Math.round(sum.margin_K)} K).`}
      ${sum.x_cp !== undefined ? ` Centre of pressure ${fmtPct((sum.x_cp - st.x_nose) / st.body_length, 0)} of the length from the nose.` : ""}</p>
    <div class="shots" id="shots"><div class="skeleton" style="height:220px"></div>${opt ? '<div class="skeleton" style="height:220px"></div>' : ""}</div>
    <h2>At ${env.aoas[ia]}° by Mach number</h2>
    <table class="tbl compact"><thead><tr><th class="r">Mach</th><th class="r">V</th><th class="r">Cd</th><th class="r">wave</th><th class="r">base</th><th class="r">friction</th>
      <th class="r">CL</th><th class="r">Peak wall</th><th class="r">Margin</th><th class="r">q stag (cold)</th></tr></thead><tbody>
      ${pol[ia].speeds.map((s) => `<tr><td class="r">${s.mach}</td><td class="r">${Math.round(s.v)} m/s</td><td class="r">${sig(s.cd, 3)}</td><td class="r">${sig(s.cd_wave, 3)}</td>
        <td class="r">${sig(s.cd_base, 3)}</td><td class="r">${sig(s.cd_friction, 3)}</td><td class="r">${sig(s.cl, 3)}</td><td class="r">${degC(s.tw_max)}</td>
        <td class="r">${Math.round(s.margin_K)} K</td><td class="r">${sig(s.q_stag / 1000, 3)} kW/m²</td></tr>`).join("")}</tbody></table>
    <h2>Setup</h2>
    <table class="tbl compact"><tbody>
      <tr><td>Model file</td><td class="r">${esc(project.filename)} (${fmtInt(project.metrics.triangles)} triangles)</td></tr>
      <tr><td>Units / orientation</td><td class="r">${esc(r.model.units)}, nose ${esc(r.model.forward)}, up ${esc(r.model.up)}</td></tr>
      <tr><td>Atmosphere</td><td class="r">US Standard 1976, ${sig(env.altitude / 1000, 3)} km: ${sig(st.air.T - 273.15, 3)} °C, ${sig(st.air.p / 1000, 3)} kPa</td></tr>
      <tr><td>Materials</td><td class="r">${mats}</td></tr>
      <tr><td>Wall / boundary layer</td><td class="r">${st.thermal.wall === "fixed" ? `fixed at ${degC(st.thermal.t_wall_k)}` : "radiative equilibrium"};
        ${st.thermal.transition === "auto" ? `transition at Re<sub>x</sub> ${sig(st.thermal.re_transition, 2)}` : st.thermal.transition}</td></tr>
      <tr><td>Reference area / length</td><td class="r">${sig(st.ref_area, 4)} m² / ${sig(st.ref_length, 4)} m</td></tr>
      <tr><td>Compute</td><td class="r">${fmtInt(r.evaluations)} surface evaluations, ${sig(r.elapsed_s, 2)} s</td></tr>
    </tbody></table>
    <h2>Method and limitations</h2>
    <p class="small">Local-inclination surface methods: tangent-cone (Taylor–Maccoll) and tangent-wedge (oblique shock) on windward faces, modified Newtonian
      past shock detachment and on blunt noses, Prandtl–Meyer expansion on leeward faces and an empirical base pressure (C<sub>p,b</sub> = −0.25/M).
      Skin friction and heating by Eckert's reference temperature with each face's running length; nose stagnation heating by Fay–Riddell; wall temperature
      from radiative equilibrium. Perfect gas (γ = 1.4): real-gas chemistry above Mach 6 and shock interactions are not modelled, and low-supersonic normal
      force on bodies is over-predicted. Results are engineering estimates; validate safety-critical designs independently.</p>`;
  const pr = $("#print");
  if (pr) pr.addEventListener("click", () => print());
  const host = $("#render-host");
  const v = new Viewer(host);
  const out = [];
  for (const which of opt ? ["baseline", "best"] : ["baseline"]) {
    const entry = r.fields[which][ia];
    const body = await file(r.meshes[which], "stl");
    const f = await file(entry.file);
    const ff = hyperFaceFields(f, im, r);
    v.setModel(rotate(body, windRotation(entry.aoa, env.yaw)), { fit: true });
    v.setFaceField("temp", ff.tw, ff.twRange[0], ff.twRange[1], "temp");
    v.setMode("temp");
    v.view("iso");
    await new Promise((res) => setTimeout(res, 150));
    out.push({ url: v.snapshot(960, 600), cap: `${which === "best" ? "Optimized design" : "Baseline"} at ${entry.aoa}°, Mach ${env.machs[im]} — wall temperature` });
  }
  $("#shots").innerHTML = out.map((s) => `<figure><img src="${s.url}" alt="${esc(s.cap)}"><figcaption>${esc(s.cap)}</figcaption></figure>`).join("");
  v.dispose();
}

async function shots(r, ia, opt) {
  const host = $("#render-host");
  const v = new Viewer(host);
  const out = [];
  for (const which of opt ? ["baseline", "best"] : ["baseline"]) {
    const entry = r.fields[which][ia];
    const body = await file(r.meshes[which], "stl");
    const f = await file(entry.file);
    v.setModel(rotate(body, windRotation(entry.mirror ? -entry.aoa : entry.aoa, r.envelope.yaw)), { fit: true });
    v.setMirror(entry.mirror);
    const [lo, hi] = f.surface_cp_range;
    v.setFaceField("cp", dequantize(b64bytes(f.surface_cp), lo, hi), -0.6, 0.6, "cp");
    v.setMode("cp");
    v.setStreamlines(f.streamlines);
    v.showStreamlines(true);
    if (v.particles) v.particles.visible = false;
    v.view("iso");
    await new Promise((res) => setTimeout(res, 150));
    out.push({ url: v.snapshot(960, 600), cap: `${which === "best" ? "Optimized design" : "Baseline"} at ${entry.aoa}° — surface pressure and streamlines` });
  }
  $("#shots").innerHTML = out.map((s) => `<figure><img src="${s.url}" alt="${esc(s.cap)}"><figcaption>${esc(s.cap)}</figcaption></figure>`).join("");
  v.dispose();
}

render().catch((e) => { handleError(e); root.innerHTML = `<h1>Could not build the report</h1><p>${esc(e.message)}</p>`; });
