// Shared front-end helpers: API client, DOM builder, icons, toasts, modals, nav bars.
export const APP_NAME = "AeroSculpt";

// ---- API ---------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(status, data) {
    super((data && data.error) || `Request failed (${status})`);
    this.status = status;
    this.data = data || {};
  }
}

export async function api(path, { method = "GET", body, raw, headers = {}, signal } = {}) {
  const opts = { method, headers: { "X-Requested-With": "AeroSculpt", ...headers }, credentials: "same-origin", signal };
  if (raw !== undefined) {
    opts.body = raw;
    opts.headers["Content-Type"] = "application/octet-stream";
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("application/json") ? await res.json().catch(() => ({})) : null;
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

let _me;
export async function getMe(force = false) {
  if (_me && !force) return _me;
  try {
    _me = await api("/api/me");
  } catch (e) {
    if (e.status === 401) _me = null;
    else throw e;
  }
  return _me;
}

export async function requireAuth() {
  const me = await getMe();
  if (!me) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname + location.search);
    throw new Error("redirecting");
  }
  return me;
}

// ---- DOM -----------------------------------------------------------------------------
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k === "style" && typeof v === "object") {
        // custom properties (--x) need setProperty; Object.assign silently drops them
        for (const [sk, sv] of Object.entries(v)) sk.startsWith("--") ? el.style.setProperty(sk, sv) : (el.style[sk] = sv);
      }
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else el.setAttribute(k, v === true ? "" : v);
    }
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- icons (stroke icons drawn on a 24px grid) ---------------------------------------
const P = {
  send: '<path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4z"/>',
  code: '<path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9"/><path d="M16 3.1a4 4 0 0 1 0 7.8"/>',
  road: '<path d="M4 19 8 5"/><path d="m16 5 4 14"/><path d="M12 6v2"/><path d="M12 11v2"/><path d="M12 16v2"/>',
  wind: '<path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 12h16a3 3 0 1 1-3 3"/><path d="M3 16h7"/>',
  upload: '<path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  box: '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5"/><path d="M12 13v8"/>',
  play: '<path d="M7 4v16l13-8z"/>',
  sparkles: '<path d="M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4z"/><path d="M19 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/>',
  chart: '<path d="M4 20V4"/><path d="M4 20h16"/><path d="m7 15 4-5 3 3 5-7"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  card: '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20"/>',
  check: '<path d="m5 12 5 5L20 7"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  move: '<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
  scale: '<path d="M4 14v6h6M20 10V4h-6M4 20 20 4"/>',
  lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  unlock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-2"/>',
  alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  download: '<path d="M12 4v12"/><path d="m7 11 5 5 5-5"/><path d="M4 20h16"/>',
  trash: '<path d="M4 7h16"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
  camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  file: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/>',
  chev: '<path d="m6 9 6 6 6-6"/>',
  ext: '<path d="M14 4h6v6"/><path d="M20 4 10 14"/><path d="M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  gauge: '<path d="M12 14l4-4"/><path d="M3.3 17a9 9 0 1 1 17.4 0"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
  zap: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  shield: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/><path d="m9 12 2 2 4-4"/>',
  flask: '<path d="M9 3h6"/><path d="M10 3v6L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-10V3"/><path d="M7 15h10"/>',
  rotate: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>',
  plane: '<path d="M2 16l20-6-20-6 4 6z"/><path d="M6 10h8"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  plus: '<path d="M12 5v14"/><path d="M5 12h14"/>',
  terminal: '<path d="m5 8 4 4-4 4"/><path d="M12 16h7"/>',
  eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  compare: '<path d="M8 3v18"/><path d="M16 3v18"/><path d="m4 8 4-4 4 4"/><path d="m12 16 4 4 4-4"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="m11 12 9-9"/><path d="m17 6 3 3"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  rocket: '<path d="M5 15c-1.5 1.3-2 5-2 5s3.7-.5 5-2c.7-.8.7-2.1-.1-2.9a2.1 2.1 0 0 0-2.9-.1z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.9A12.9 12.9 0 0 1 22 2c0 2.7-.8 7.5-6 11a22.4 22.4 0 0 1-4 2z"/><path d="M9 12H4s.6-3 2-4c1.6-1.1 5 0 5 0"/><path d="M12 15v5s3-.6 4-2c1.1-1.6 0-5 0-5"/>',
  shieldlock: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/><rect x="9" y="11" width="6" height="5" rx="1"/><path d="M10 11V9.5a2 2 0 0 1 4 0V11"/>',
  crosshair: '<circle cx="12" cy="12" r="8"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
  fan: '<circle cx="12" cy="12" r="1.6"/><path d="M12 10.4c-.6-3.6.4-7 3-7.6 2.2-.5 3.2 2.1 1.6 3.9-1.2 1.4-2.9 2.6-4.6 3.7"/><path d="M13.4 12.8c2.9 2.2 4.3 5.6 2.4 7.5-1.6 1.6-3.9-.1-3.6-2.5.2-1.8 1-3.6 1.2-5"/><path d="M10.6 12.8c-3.4 1.4-7 .9-7.8-1.7-.6-2.2 2.2-3 3.9-1.5 1.4 1.1 2.6 2.3 3.9 3.2"/>',
  flame: '<path d="M12 22c4 0 7-2.7 7-6.6 0-3.7-2.6-6.1-4.4-8.4-.5 2-1.5 3.3-2.8 3.9.4-3-1-6.5-3.8-8.9.3 3.6-1.6 5.8-3.1 7.8C3.9 11.4 5 13 5 15.4 5 19.3 8 22 12 22z"/>',
  thermo: '<path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0z"/><path d="M12 9v8"/>',
  sliders: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  book: '<path d="M4 4h6a3 3 0 0 1 3 3v13a2 2 0 0 0-2-2H4z"/><path d="M20 4h-6a3 3 0 0 0-3 3v13a2 2 0 0 1 2-2h7z"/>',
};
export function icon(name, cls = "i") {
  return `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${P[name] || ""}</svg>`;
}
export function iconEl(name, cls) {
  const s = document.createElement("span");
  s.style.display = "inline-flex";
  s.innerHTML = icon(name, cls);
  return s.firstChild;
}

// ---- format -------------------------------------------------------------------------
export function fmt(v, d = 3) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: Math.min(d, a < 10 ? d : 0) });
}
export function sig(v, n = 3) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  if (Math.abs(v) < 1e-9) return "0";   // symmetric bodies give round-off like -1e-12
  const d = Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(v))));
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: Math.min(d, 6), minimumFractionDigits: Math.min(d, 6) });
}
export function fmtForce(n) {
  if (!isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a < 1e-9) return "0 N";
  if (a >= 1000) return sig(n / 1000, 3) + " kN";
  if (a < 0.1) return sig(n * 1000, 3) + " mN";
  return sig(n, 3) + " N";
}
export function fmtPower(w) {
  if (!isFinite(w)) return "—";
  return Math.abs(w) >= 1000 ? sig(w / 1000, 3) + " kW" : sig(w, 3) + " W";
}
export function fmtPct(f, d = 1, signed = false) {
  if (f === null || f === undefined || !isFinite(f)) return "—";
  const s = (f * 100).toFixed(d) + "%";
  return signed && f > 0 ? "+" + s : s;
}
export function fmtInt(n) { return Number(n || 0).toLocaleString(); }
export function fmtDate(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
export function timeAgo(ts) {
  if (!ts) return "";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + " min ago";
  if (s < 86400) return Math.floor(s / 3600) + " h ago";
  if (s < 86400 * 30) return Math.floor(s / 86400) + " d ago";
  return fmtDate(ts);
}
export function fmtDuration(s) {
  if (!isFinite(s)) return "—";
  if (s < 60) return Math.round(s) + " s";
  if (s < 3600) return Math.round(s / 60) + " min";
  return (s / 3600).toFixed(1) + " h";
}
export const UNIT_SCALE = { mm: 1e-3, cm: 1e-2, m: 1, in: 0.0254, ft: 0.3048 };
export function fmtLen(m, units = "m") {
  if (!isFinite(m)) return "—";
  if (!UNIT_SCALE[units]) units = "m";          // custom-scale models are shown in metres
  const v = m / UNIT_SCALE[units];
  return sig(v, 4) + " " + units;
}

// ---- toasts & modals -----------------------------------------------------------------
let toastBox;
export function toast(msg, kind = "") {
  if (!toastBox) {
    toastBox = h("div", { class: "toasts", role: "status", "aria-live": "polite" });
    document.body.append(toastBox);
  }
  const ic = kind === "bad" ? "alert" : kind === "good" ? "check" : "info";
  const t = h("div", { class: "toast " + kind, html: icon(ic) + `<div>${esc(msg)}</div>` });
  toastBox.append(t);
  setTimeout(() => t.remove(), kind === "bad" ? 7000 : 4000);
}

export function modal({ title, body, actions = [], onClose }) {
  const back = h("div", { class: "modal-back" });
  const box = h("div", { class: "modal", role: "dialog", "aria-modal": "true" });
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey); onClose && onClose(); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  box.append(h("h3", null, title));
  if (typeof body === "string") box.append(h("p", { html: body }));
  else if (body) box.append(body);
  const act = h("div", { class: "actions" });
  for (const a of actions) {
    const b = h("button", { class: "btn " + (a.kind || ""), type: "button" }, a.label);
    b.addEventListener("click", async () => {
      if (a.onClick) {
        const keep = await a.onClick(b);
        if (keep === true) return;
      }
      close();
    });
    act.append(b);
  }
  if (actions.length) box.append(act);
  back.append(box);
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  document.addEventListener("keydown", onKey);
  document.body.append(back);
  return { close, box };
}

export function confirmDialog(title, text, okLabel = "Confirm", danger = false) {
  return new Promise((resolve) => {
    let done = false;
    modal({
      title, body: text,
      actions: [
        { label: "Cancel", kind: "ghost", onClick: () => { done = true; resolve(false); } },
        { label: okLabel, kind: danger ? "danger" : "primary", onClick: () => { done = true; resolve(true); } },
      ],
      onClose: () => { if (!done) resolve(false); },
    });
  });
}

export function upgradePrompt(message) {
  modal({
    title: "Upgrade to unlock this",
    body: h("div", null,
      h("p", null, message),
      h("p", { class: "muted small" }, "Pro starts with a free trial. Cancel any time.")),
    actions: [
      { label: "Not now", kind: "ghost" },
      { label: "See plans", kind: "primary", onClick: () => { location.href = "/app/billing"; } },
    ],
  });
}

export function handleError(e, fallback = "Something went wrong.") {
  if (e && e.message === "redirecting") return;
  if (e instanceof ApiError) {
    if (e.status === 401) { location.href = "/login?next=" + encodeURIComponent(location.pathname); return; }
    if (e.data && e.data.upgrade) { upgradePrompt(e.message); return; }
    toast(e.message, "bad");
    return;
  }
  console.error(e);
  toast(fallback, "bad");
}

export function busy(btn, on, label) {
  if (on) {
    btn.dataset.label = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spin"></span>${label ? esc(label) : ""}`;
  } else {
    btn.disabled = false;
    if (btn.dataset.label) btn.innerHTML = btn.dataset.label;
  }
}

// ---- theme -----------------------------------------------------------------------------
export function currentTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function themeButton() {
  const b = h("button", { class: "icon-btn", type: "button", title: "Switch light / dark theme", "aria-label": "Switch light or dark theme" });
  const paint = () => { b.innerHTML = icon(currentTheme() === "dark" ? "sun" : "moon"); };
  paint();
  b.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("as-theme", next); } catch (e) { /* private mode */ }
    paint();
    dispatchEvent(new Event("themechange"));
  });
  return b;
}

// ---- navigation ------------------------------------------------------------------------
const LOGO = '<img src="/static/img/logo.svg" alt="" width="28" height="28">';

export async function siteNav() {
  const nav = h("nav", { class: "nav" });
  const inner = h("div", { class: "container nav-inner" });
  inner.innerHTML = `<a class="brand" href="/">${LOGO}${APP_NAME}</a>
    <div class="nav-links"><a href="/#tunnels">Tunnels</a><a href="/#compare">Compare</a><a href="/#how">How it works</a>
    <a href="/pricing">Pricing</a><a href="/validation">Validation</a><a href="/docs">Docs</a></div>
    <div class="nav-right"></div>`;
  nav.append(inner);
  document.body.prepend(nav);
  const right = inner.querySelector(".nav-right");
  right.append(themeButton());
  let me = null;
  try { me = await getMe(); } catch (e) { /* offline: show signed-out nav */ }
  if (me) {
    right.append(h("a", { class: "btn primary sm", href: "/app" }, "Open app"));
  } else {
    right.append(h("a", { class: "btn ghost sm", href: "/login" }, "Sign in"),
      h("a", { class: "btn primary sm", href: "/signup" }, "Start free"));
  }
  return me;
}

export function siteFooter() {
  const f = h("footer", { class: "site" });
  f.innerHTML = `<div class="container"><div class="cols">
    <div style="max-width:320px"><a class="brand" href="/">${LOGO}${APP_NAME}</a>
      <p class="small" style="margin-top:10px">Three wind tunnels — RC, subsonic CFD and supersonic to hypersonic — with automated shape optimization.</p></div>
    <div><div class="label">Product</div><a href="/#tunnels">Tunnels</a><a href="/#features">Features</a><a href="/pricing">Pricing</a><a href="/validation">Validation</a><a href="/docs">Documentation</a></div>
    <div><div class="label">Company</div><a href="/terms">Terms</a><a href="/privacy">Privacy</a><a href="mailto:support@example.com">Support</a></div>
    </div><p class="small" style="margin-top:28px">&copy; ${new Date().getFullYear()} ${APP_NAME}. Simulation results are engineering estimates; validate critical designs physically.</p></div>`;
  document.body.append(f);
}

export function appTop(me, crumbs = []) {
  const top = h("header", { class: "app-top" });
  const left = h("div", { class: "crumbs" });
  left.append(h("a", { class: "brand", href: "/app", html: LOGO }));
  crumbs.forEach((c) => {
    left.append(h("span", { class: "muted" }, "/"));
    left.append(c instanceof Node ? c : (c.href ? h("a", { href: c.href }, c.label) : h("span", null, c.label)));
  });
  top.append(left);
  const right = h("div", { class: "top-right" });
  top.append(right);
  right.append(themeButton());
  const pill = h("a", { class: "credits-pill", href: "/app/billing", title: "Compute credits left this month" });
  right.append(pill);
  const planBadge = h("span", { class: "badge accent" });
  right.append(planBadge);
  const menu = h("div", { class: "menu" });
  const av = h("button", { class: "avatar", "aria-label": "Account menu" }, (me.name || me.email)[0].toUpperCase());
  const pop = h("div", { class: "menu-pop hidden" });
  pop.innerHTML = `<div class="who">${esc(me.name || "")}<br>${esc(me.email)}</div>
    <a href="/app">${icon("grid")}Projects</a>
    ${me.guest ? "" : `<a href="/app/billing">${icon("card")}Plan &amp; billing</a>
    <a href="/app/account">${icon("user")}Account</a>`}
    <a href="/docs">${icon("book")}Documentation</a>
    ${me.is_admin ? `<a href="/admin">${icon("shield")}Admin</a>` : ""}
    ${me.preview ? "" : `<button type="button" data-logout>${icon("logout")}${me.guest ? "Leave the demo" : "Sign out"}</button>`}`;
  av.addEventListener("click", (e) => { e.stopPropagation(); pop.classList.toggle("hidden"); });
  document.addEventListener("click", () => pop.classList.add("hidden"));
  pop.querySelector("[data-logout]")?.addEventListener("click", async () => {
    await api("/api/auth/logout", { method: "POST" }).catch(() => {});
    location.href = "/";
  });
  menu.append(av, pop);
  right.append(menu);
  const update = (m) => {
    const used = m.usage.credits, total = m.limits.monthly_credits;
    const frac = Math.min(1, used / Math.max(total, 1));
    pill.innerHTML = `${icon("zap")}<span class="num">${fmtInt(m.credits_left)}</span> credits
      <span class="meter ${frac > 0.9 ? "bad" : frac > 0.7 ? "warn" : ""}"><div style="width:${(1 - frac) * 100}%"></div></span>`;
    planBadge.textContent = m.plan_name + (m.plan_status === "trialing" ? " trial" : "");
  };
  update(me);
  document.body.prepend(top);
  banners(me, top);
  return { top, update };
}

function banners(me, after) {
  if (me.guest) {
    const hours = Math.max(1, Math.round((me.guest_expires_at - Date.now() / 1000) / 3600));
    const b = h("div", { class: "trial-banner", html: `${icon("sparkles")} Guest demo — open a sample or template and run it. This temporary account and its projects are deleted in about ${hours} hour${hours === 1 ? "" : "s"}.` });
    after.after(b);
    return;
  }
  if (me.preview) {
    // local preview: sign-in is off and every feature is unlocked; one quiet banner says so
    const b = h("div", { class: "trial-banner", html: `${icon("eye")} Local preview — sign-in is turned off and every feature is unlocked. Set <code>OPEN_ACCESS=0</code> to bring accounts back.` });
    after.after(b);
    return;
  }
  if (me.billing_mode === "demo") {
    const b = h("div", { class: "trial-banner demo-banner", html: `${icon("alert")} Billing is in demo mode on this server: plans are granted without payment. Configure Stripe keys before launch.` });
    after.after(b);
  }
  if (me.plan_status === "trialing" && me.trial_ends_at) {
    const days = Math.max(0, Math.ceil((me.trial_ends_at - Date.now() / 1000) / 86400));
    const b = h("div", { class: "trial-banner", html: `${icon("sparkles")} ${esc(me.plan_name)} trial: ${days} day${days === 1 ? "" : "s"} left. <a href="/app/billing">Manage plan</a>` });
    after.after(b);
  }
}

export function qs(name) {
  return new URLSearchParams(location.search).get(name);
}
