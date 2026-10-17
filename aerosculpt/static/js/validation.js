import { api, esc, fmtDate, fmtInt, fmtPct, icon, sig, siteFooter, siteNav, $ } from "/static/js/core.js";
import { lineChart } from "/static/js/charts.js";

siteNav();
siteFooter();
const data = await api("/api/validation");
const cases = data.cases || [];

const sn = (re) => (24 / re) * (1 + 0.15 * Math.pow(re, 0.687));
const ref = [];
for (let x = Math.log10(60); x <= Math.log10(900) + 1e-9; x += 0.05) ref.push([x, sn(10 ** x)]);
const ours = cases.filter((c) => c.id.startsWith("sphere-re") && c.regime === "direct").map((c) => [Math.log10(c.re), c.cd]);
const fmtRe = (v) => { const r = 10 ** v; return r >= 1000 ? sig(r / 1000, 2) + "k" : String(Math.round(r)); };
if (ours.length) {
  lineChart($("#chart"), [
    { name: "Schiller–Naumann reference", color: "var(--series-2)", points: ref },
    { name: "AeroSculpt", color: "var(--series-1)", points: ours, dots: true },
  ], { height: 230, xName: "Re", xLabel: "Reynolds number (log scale)", fmtX: fmtRe, fmtY: (v) => sig(v, 3) });
} else {
  $("#chart").innerHTML = `<p class="muted small">Validation results have not been generated on this server yet.</p>`;
}

const STATUS = {
  pass: `<span class="badge good">${icon("check")}within tolerance</span>`,
  warn: `<span class="badge warn">${icon("alert")}10–25% off</span>`,
  fail: `<span class="badge bad">${icon("alert")}over 25% off</span>`,
};
const GROUPS = [
  ["general", "General tunnel", "3D lattice-Boltzmann CFD: drag of bluff and streamlined bodies, and lift on a finite wing."],
  ["rc", "RC tunnel", "Blade-element propeller model against UIUC wind-tunnel data; the actuator disk in the CFD against momentum theory."],
  ["supersonic", "Supersonic tunnel", "Surface methods against free-flight data, exact conical-flow theory, ogive wave-drag data, linear supersonic theory and heating correlations."],
  ["wing", "Wing model", "What the optimizer uses for wings and NACA sections: the vortex lattice against exact lifting-surface results and theory, NeuralFoil sections against NASA wind-tunnel data."],
  ["rocket", "Rocket methods", "The flight model: drag build-up, Barrowman stability and the flight integrator against their published sources (Schlichting, Hoerner, Barrowman, Helmbold, Estes TR-10/TR-11) and the manufacturer's apogee for a classic kit."],
];
const fmtVal = (v) => (Math.abs(v) >= 100 ? sig(v, 4) : sig(v, 3));
function caseRow(c) {
  const cond = c.condition || (c.re >= 1e4 ? "Re " + c.re.toExponential(1) : "Re " + fmtInt(c.re));
  const method = c.method ? esc(c.method) : `${esc(c.fidelity)}<div class="small muted">${fmtInt(c.cells)} cells, ${Math.round(c.cells_across)} across · ${esc(c.regime)}</div>`;
  const ref = c.ref_band ? `${sig(c.ref_band[0], 2)}–${sig(c.ref_band[1], 2)}` : fmtVal(c.ref);
  return `<tr><td><span style="color:var(--text)">${esc(c.body)}</span><div class="small muted">${esc(c.ref_source)}</div></td>
    <td>${cond}</td><td>${method}</td>
    <td class="r"><span class="muted small">${c.quantity ? esc(c.quantity) : "C<sub>d</sub>"}</span> ${fmtVal(c.value ?? c.cd)}</td>
    <td class="r">${ref}</td><td class="r">${fmtPct(c.error, 1, true)}</td><td>${STATUS[c.status] || esc(c.status)}</td></tr>`;
}
if (cases.length) {
  $("#case-groups").innerHTML = GROUPS.map(([g, title, sub]) => {
    const rows = cases.filter((c) => (c.group || "general") === g);
    if (!rows.length) return "";
    const ok = rows.filter((c) => c.status === "pass").length;
    return `<h3 id="${g}" class="val-group">${esc(title)} <span class="badge ${ok === rows.length ? "good" : "warn"}">${ok} of ${rows.length} within tolerance</span></h3>
      <p class="small muted" style="margin:0 0 10px">${esc(sub)}</p>
      <div class="card" style="overflow-x:auto;padding:8px 12px;margin-bottom:26px"><table class="tbl"><thead><tr><th>Case</th><th>Condition</th><th>Method</th>
        <th class="r">Result</th><th class="r">Reference</th><th class="r">Error</th><th>Status</th></tr></thead><tbody>${rows.map(caseRow).join("")}</tbody></table></div>`;
  }).join("");
  $("#meta").textContent = `Generated ${fmtDate(data.generated_at)} on ${data.machine || "the production server"}.`;
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
} else {
  $("#case-groups").innerHTML = `<div class="card muted">No results yet — run <code>python run.py validate</code> on the server.</div>`;
}
