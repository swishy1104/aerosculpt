// Field colormaps for the dark viewport. Sequential = one hue, dark (low) to light (high) so
// near-zero recedes into the background; diverging = two poles around a neutral gray.
const hex = (s) => [parseInt(s.slice(1, 3), 16) / 255, parseInt(s.slice(3, 5), 16) / 255, parseInt(s.slice(5, 7), 16) / 255];
const BLUE = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"].map(hex);
const ORANGE = ["#3b1607", "#5a210b", "#7a2d0f", "#9c3a14", "#bd4718", "#d95926", "#e4733f", "#ec8d5f", "#f2a882", "#f7c4a8", "#fbdfcf"].map(hex);
let MID = hex("#3a3a3d");      // neutral midpoint of the diverging map; follows the theme

/** Re-read theme-dependent colours (call after the theme changes). */
export function refreshColormaps() {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--cmap-mid").trim();
  if (/^#[0-9a-f]{6}$/i.test(v)) MID = hex(v);
}
refreshColormaps();
const HEAT = ["#1b0b2e", "#3d0f4f", "#6a1b5e", "#952b5c", "#bd4446", "#dc6532", "#ee8e2a", "#f6b93b", "#fbe17a", "#fff7d6"].map(hex);
const RED = hex("#e5484d");
const GREEN = hex("#3fb68b");
const NEG = hex("#5598e7");   // suction
const POS = hex("#e66767");   // stagnation / overpressure

function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function mix(a, b, f) {
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export const MAPS = {
  speed: { label: "Flow speed  |u| / U∞", kind: "seq", fn: (t) => ramp(BLUE, t) },
  deform: { label: "Shape change", kind: "seq", fn: (t) => ramp(ORANGE, t) },
  temp: { label: "Wall temperature", kind: "seq", fn: (t) => ramp(HEAT, t) },
  flux: { label: "Heat flux", kind: "seq", fn: (t) => ramp(ORANGE, Math.sqrt(Math.max(t, 0))) },
  // margin to the material's service limit: red where it is exceeded, green with room to spare
  margin: {
    label: "Margin to material limit", kind: "div",
    fnValue: (v, lo, hi) => (v < 0 ? mix(MID, RED, Math.min(1, v / lo)) : mix(MID, GREEN, Math.min(1, v / hi))),
  },
  // value-space diverging map: needs the data range so 0 sits at the gray midpoint
  cp: {
    label: "Pressure coefficient  Cp", kind: "div",
    fnValue: (v, lo, hi) => (v < 0 ? mix(MID, NEG, Math.min(1, v / lo)) : mix(MID, POS, Math.min(1, v / hi))),
  },
};

export function colorFor(mapName, value, lo, hi) {
  const m = MAPS[mapName];
  if (m.kind === "div") return m.fnValue(value, lo, hi);
  return m.fn((value - lo) / (hi - lo || 1));
}

export function cssGradient(mapName, lo, hi, n = 16) {
  const stops = [];
  for (let i = 0; i <= n; i++) {
    const v = lo + (hi - lo) * (i / n);
    const [r, g, b] = colorFor(mapName, v, lo, hi);
    stops.push(`rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(i / n * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${stops.join(",")})`;
}

// decode server-quantized arrays: uint8 codes 0..254 over [lo, hi], 255 = no data
export function dequantize(codes, lo, hi) {
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i++) out[i] = codes[i] === 255 ? NaN : lo + (hi - lo) * (codes[i] / 254);
  return out;
}

export function b64bytes(b64) {
  const s = atob(b64);
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u;
}
export function b64f32(b64) {
  const u = b64bytes(b64);
  return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4);
}
