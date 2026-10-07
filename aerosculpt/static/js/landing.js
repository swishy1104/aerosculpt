import { siteNav, siteFooter, icon } from "/static/js/core.js";

// theme colours for the canvas, re-read when the theme switches
let C = {};
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const rgb = (v) => {
    const m = v.trim().match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
  };
  C = { bg: rgb(cs.getPropertyValue("--bg")), accent: rgb(cs.getPropertyValue("--accent")), body: rgb(cs.getPropertyValue("--surface-3")),
        light: cs.getPropertyValue("color-scheme").trim() === "light" };
}
readTheme();
addEventListener("themechange", () => { readTheme(); ctx.clearRect(0, 0, W, H); });
siteNav();
siteFooter();
document.querySelectorAll("[data-i]").forEach((el) => { el.innerHTML = icon(el.dataset.i); });

// Hero: potential flow around a Rankine oval, traced by drifting particles.
const cv = document.getElementById("flow");
const ctx = cv.getContext("2d");
let W, H, parts;
const m = 0.12, a = 1.6;           // source strength and half-spacing: a slender, pod-like Rankine oval
let withBody = true;               // the pod sits behind the hero text on narrow screens: plain stream there
function vel(x, y) {
  if (!withBody) return [1, 0];
  // uniform stream + source at -a + sink at +a  (body length ~ 2.6 units)
  const dx1 = x + a, dx2 = x - a;
  const r1 = dx1 * dx1 + y * y + 1e-6, r2 = dx2 * dx2 + y * y + 1e-6;
  return [1 + m * (dx1 / r1 - dx2 / r2), m * (y / r1 - y / r2)];
}
function resize() {
  W = cv.width = cv.clientWidth * devicePixelRatio;
  H = cv.height = cv.clientHeight * devicePixelRatio;
  withBody = cv.clientWidth >= 900;
  parts = Array.from({ length: Math.round(W * H / 4200) }, () => spawn(true));
}
const S = () => Math.min(W * 0.11, H / 4.2);
const cx = () => W * 0.78, cy = () => H * 0.6;
function spawn(anywhere = false) {
  const left = -cx() / S() - 0.5;
  return { x: anywhere ? left + Math.random() * (W / S()) : left - Math.random() * 2, y: (Math.random() - 0.5) * (H / S()) * 1.1, life: 0 };
}
const XMAX = a * Math.sqrt(1 + 2 * m / a);   // stagnation points of the oval
function bodyShape() {
  // body outline from the zero streamline, solved numerically along x
  const pts = [];
  for (let i = 0; i <= 80; i++) {
    const x = -XMAX + 2 * XMAX * i / 80;
    let lo = 0, hi = 1.5;
    for (let k = 0; k < 40; k++) {
      const y = (lo + hi) / 2;
      const psi = y + m * (Math.atan2(y, x + a) - Math.atan2(y, x - a));
      if (psi > 0) hi = y; else lo = y;  // psi < 0 inside the oval, > 0 outside
    }
    pts.push([x, (lo + hi) / 2]);
  }
  return pts;
}
let outline;
function frame() {
  ctx.fillStyle = `rgba(${C.bg[0]},${C.bg[1]},${C.bg[2]},0.18)`;
  ctx.fillRect(0, 0, W, H);
  const s = S(), X = cx(), Y = cy();
  ctx.lineWidth = 1.2 * devicePixelRatio;
  for (const p of parts) {
    const [u, v] = vel(p.x, p.y);
    const sp = Math.hypot(u, v);
    const nx = p.x + u * 0.035, ny = p.y + v * 0.035;
    const speed = Math.min(sp / 1.6, 1);
    const a = C.accent;
    ctx.strokeStyle = `rgba(${a[0]},${a[1]},${a[2]},${(C.light ? 0.18 : 0.22) + 0.45 * speed})`;
    ctx.beginPath();
    ctx.moveTo(X + p.x * s, Y + p.y * s);
    ctx.lineTo(X + nx * s, Y + ny * s);
    ctx.stroke();
    p.x = nx; p.y = ny; p.life++;
    if (p.x * s + X > W + 20 || p.life > 2000) Object.assign(p, spawn());
  }
  if (!withBody) return requestAnimationFrame(frame);
  ctx.fillStyle = `rgb(${C.body[0]},${C.body[1]},${C.body[2]})`;
  ctx.strokeStyle = `rgba(${C.accent[0]},${C.accent[1]},${C.accent[2]},0.6)`;
  ctx.beginPath();
  outline.forEach(([x, y], i) => (i ? ctx.lineTo(X + x * s, Y - y * s) : ctx.moveTo(X + x * s, Y - y * s)));
  for (let i = outline.length - 1; i >= 0; i--) ctx.lineTo(X + outline[i][0] * s, Y + outline[i][1] * s);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  requestAnimationFrame(frame);
}
resize();
outline = bodyShape();
addEventListener("resize", resize);
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) frame();
