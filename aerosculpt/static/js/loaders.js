// Turn any supported CAD/mesh file into bytes the server accepts (STL or OBJ).
// STEP/IGES/BREP are tessellated in the browser with OpenCascade (occt-import-js, WebAssembly);
// 3MF, PLY and glTF go through three.js loaders. The native CAD file never leaves the browser.
const OCCT_VERSION = "0.0.23";
const OCCT_BASE = `https://cdn.jsdelivr.net/npm/occt-import-js@${OCCT_VERSION}/dist/`;

export const ACCEPT = ".stl,.obj,.step,.stp,.iges,.igs,.brep,.3mf,.ply,.glb,.gltf,.ork,.rkt";
export const NATIVE = ["stl", "obj", "ork", "rkt"];      // ork / rkt: rocket designs the server reads itself

export function extOf(name) {
  return (name.split(".").pop() || "").toLowerCase();
}

let occtPromise;
function loadOcct() {
  if (!occtPromise) {
    occtPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = OCCT_BASE + "occt-import-js.js";
      s.onload = () => window.occtimportjs({ locateFile: (f) => OCCT_BASE + f }).then(resolve, reject);
      s.onerror = () => reject(new Error("Could not load the CAD converter (check your connection)."));
      document.head.append(s);
    });
  }
  return occtPromise;
}

export function trianglesToSTL(tri) {
  // tri: Float32Array of xyz per vertex, 3 vertices per triangle
  const n = tri.length / 9;
  const buf = new ArrayBuffer(84 + n * 50);
  const dv = new DataView(buf);
  const head = "AeroSculpt browser conversion";
  for (let i = 0; i < head.length; i++) dv.setUint8(i, head.charCodeAt(i));
  dv.setUint32(80, n, true);
  for (let t = 0; t < n; t++) {
    const o = 84 + t * 50;
    const a = t * 9;
    const ux = tri[a + 3] - tri[a], uy = tri[a + 4] - tri[a + 1], uz = tri[a + 5] - tri[a + 2];
    const vx = tri[a + 6] - tri[a], vy = tri[a + 7] - tri[a + 1], vz = tri[a + 8] - tri[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    dv.setFloat32(o, nx / l, true); dv.setFloat32(o + 4, ny / l, true); dv.setFloat32(o + 8, nz / l, true);
    for (let k = 0; k < 9; k++) dv.setFloat32(o + 12 + k * 4, tri[a + k], true);
  }
  return buf;
}

async function fromOcct(buf, ext, onStatus) {
  onStatus && onStatus("Loading CAD converter…");
  const occt = await loadOcct();
  onStatus && onStatus("Tessellating CAD geometry…");
  const data = new Uint8Array(buf);
  const params = { linearUnit: "millimeter", linearDeflectionType: "bounding_box_ratio", linearDeflection: 0.0015, angularDeflection: 0.35 };
  const res = ext === "brep" ? occt.ReadBrepFile(data, params)
    : ext === "iges" || ext === "igs" ? occt.ReadIgesFile(data, params) : occt.ReadStepFile(data, params);
  if (!res || !res.success || !res.meshes || !res.meshes.length) throw new Error("The CAD file contains no solid geometry we could tessellate.");
  const parts = [];
  let total = 0;
  for (const m of res.meshes) {
    const pos = m.attributes.position.array;
    const idx = m.index.array;
    const tri = new Float32Array(idx.length * 3);
    for (let i = 0; i < idx.length; i++) {
      tri[i * 3] = pos[idx[i] * 3]; tri[i * 3 + 1] = pos[idx[i] * 3 + 1]; tri[i * 3 + 2] = pos[idx[i] * 3 + 2];
    }
    parts.push(tri);
    total += tri.length;
  }
  const all = new Float32Array(total);
  let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  return { tri: all, units: "mm" };
}

async function fromThree(file, ext, onStatus) {
  onStatus && onStatus("Converting mesh…");
  const THREE = await import("three");
  const url = URL.createObjectURL(file);
  try {
    let root;
    if (ext === "ply") {
      const { PLYLoader } = await import("three/addons/loaders/PLYLoader.js");
      const g = await new PLYLoader().loadAsync(url);
      root = new THREE.Mesh(g);
    } else if (ext === "3mf") {
      const { ThreeMFLoader } = await import("three/addons/loaders/3MFLoader.js");
      root = await new ThreeMFLoader().loadAsync(url);
    } else {
      const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
      root = (await new GLTFLoader().loadAsync(url)).scene;
    }
    root.updateMatrixWorld(true);
    const chunks = [];
    root.traverse((o) => {
      if (!o.isMesh || !o.geometry) return;
      let g = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
      g.applyMatrix4(o.matrixWorld);
      chunks.push(g.attributes.position.array);
    });
    if (!chunks.length) throw new Error("No meshes found in the file.");
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const all = new Float32Array(total);
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    return { tri: all, units: ext === "3mf" ? "mm" : null };
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** -> { body: ArrayBuffer, filename, source } ready for POST /api/projects */
export async function prepareUpload(file, onStatus) {
  const ext = extOf(file.name);
  const base = file.name.replace(/\.[^.]+$/, "");
  if (NATIVE.includes(ext)) return { body: await file.arrayBuffer(), filename: file.name, source: ext };
  let out;
  if (["step", "stp", "iges", "igs", "brep"].includes(ext)) out = await fromOcct(await file.arrayBuffer(), ext, onStatus);
  else if (["3mf", "ply", "glb", "gltf"].includes(ext)) out = await fromThree(file, ext, onStatus);
  else throw new Error(`.${ext} files are not supported. Use STL, OBJ, STEP, IGES, 3MF, PLY or glTF.`);
  return { body: trianglesToSTL(out.tri), filename: base + ".stl", source: ext === "stp" ? "step" : ext };
}
