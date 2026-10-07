// Minimal SVG line chart: 2px lines, hairline grid, crosshair + tooltip, legend for >= 2 series.
import { esc } from "/static/js/core.js";

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

function niceTicks(lo, hi, n = 4) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) { lo -= Math.abs(lo) * 0.1 || 1; hi += Math.abs(hi) * 0.1 || 1; }
  const span = hi - lo;
  const raw = span / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  const t0 = Math.floor(lo / step) * step;
  const out = [];
  for (let t = t0; t <= hi + step * 0.5; t += step) out.push(+t.toPrecision(12));
  return out;
}

/**
 * series: [{ name, color, points: [[x, y], ...], dashed?, step? }]
 * opts: { height, xLabel, yLabel, fmtX, fmtY, highlight: {x, y, label}, integerX }
 */
export function lineChart(container, series, opts = {}) {
  container.innerHTML = "";
  container.classList.add("chart");
  const W = container.clientWidth || 340;
  const H = opts.height || 190;
  const m = { l: 46, r: 14, t: 10, b: opts.xLabel ? 34 : 22 };
  const pts = series.flatMap((s) => s.points).filter((p) => isFinite(p[1]));
  if (!pts.length) {
    container.innerHTML = `<div class="muted small" style="padding:20px 0">No data yet.</div>`;
    return;
  }
  let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  if (x0 === x1) x1 = x0 + 1;
  let y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
  const pad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.05 || 1;
  y0 -= pad; y1 += pad;
  if (opts.zeroBased && y0 > 0) y0 = 0;
  const yt = niceTicks(y0, y1, 4);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const fx = opts.fmtX || ((v) => (opts.integerX ? Math.round(v) : +v.toPrecision(3)));
  const fy = opts.fmtY || ((v) => +v.toPrecision(3));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": opts.ariaLabel || "chart" });
  for (const t of yt) {
    if (t < y0 - 1e-12 || t > y1 + 1e-12) continue;
    svg.append(el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
    const tx = el("text", { class: "tick", x: m.l - 7, y: Y(t) + 3.5, "text-anchor": "end" });
    tx.textContent = fy(t);
    svg.append(tx);
  }
  let xt = niceTicks(x0, x1, Math.min(6, Math.max(2, Math.floor((W - 60) / 70))));
  if (opts.integerX) xt = [...new Set(xt.map(Math.round))];
  for (const t of xt) {
    if (t < x0 - 1e-9 || t > x1 + 1e-9) continue;
    const tx = el("text", { class: "tick", x: X(t), y: H - m.b + 15, "text-anchor": "middle" });
    tx.textContent = fx(t);
    svg.append(tx);
  }
  svg.append(el("line", { class: "axisline", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  if (opts.xLabel) {
    const t = el("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle" });
    t.textContent = opts.xLabel;
    svg.append(t);
  }
  if (opts.yLabel) {
    const t = el("text", { class: "axis-title", x: 2, y: m.t + 2, "text-anchor": "start" });
    t.textContent = opts.yLabel;
    svg.append(t);
  }
  for (const s of series) {
    const p = s.points.filter((q) => isFinite(q[1]));
    if (!p.length) continue;
    let d = "";
    p.forEach((q, i) => {
      if (i === 0) d += `M${X(q[0])},${Y(q[1])}`;
      else if (s.step) d += `H${X(q[0])}V${Y(q[1])}`;
      else d += `L${X(q[0])},${Y(q[1])}`;
    });
    svg.append(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
      "stroke-dasharray": s.dashed ? "5 4" : "none", opacity: s.faded ? 0.55 : 1 }));
    if (s.dots) for (const q of p) svg.append(el("circle", { cx: X(q[0]), cy: Y(q[1]), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
    else {
      const q = p[p.length - 1];
      svg.append(el("circle", { cx: X(q[0]), cy: Y(q[1]), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
    }
  }
  if (opts.highlight) {
    const { x, y } = opts.highlight;
    svg.append(el("circle", { cx: X(x), cy: Y(y), r: 6, fill: "none", stroke: "var(--text)", "stroke-width": 1.5 }));
  }
  // hover layer
  const cross = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" });
  svg.append(hit);
  container.append(svg);
  const tip = document.createElement("div");
  tip.className = "tooltip hidden";
  container.append(tip);
  const xs = [...new Set(pts.map((p) => p[0]))].sort((a, b) => a - b);
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
    let best = xs[0];
    for (const v of xs) if (Math.abs(v - xv) < Math.abs(best - xv)) best = v;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best));
    cross.setAttribute("visibility", "visible");
    const rows = series.map((s) => {
      const q = s.points.find((p) => p[0] === best);
      return q && isFinite(q[1]) ? `<div class="tt-row"><span class="key" style="background:${s.color}"></span>${esc(s.name)}<b>${esc(fy(q[1]))}</b></div>` : "";
    }).join("");
    tip.innerHTML = `<div style="color:var(--muted);margin-bottom:3px">${esc(opts.xName || "x")} ${esc(fx(best))}</div>${rows}`;
    tip.style.left = (X(best) / W) * r.width + "px";
    tip.style.top = "8px";
    tip.style.transform = X(best) > W * 0.65 ? "translate(calc(-100% - 10px), 0)" : "translate(10px, 0)";
    tip.classList.remove("hidden");
  });
  hit.addEventListener("mouseleave", () => { tip.classList.add("hidden"); cross.setAttribute("visibility", "hidden"); });
  if (series.length >= 2) {
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.style.marginTop = "6px";
    lg.innerHTML = series.map((s) => `<span><span class="ln" style="background:${s.color}"></span>${esc(s.name)}</span>`).join("");
    container.append(lg);
  }
}
