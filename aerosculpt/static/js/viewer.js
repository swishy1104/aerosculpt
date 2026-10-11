// Three.js viewport: model, surface fields, slices, streamlines, protected regions.
// Display frame = wind axes in metres (flow travels +x, z is up) at the selected angle.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { b64bytes, b64f32, colorFor, dequantize, refreshColormaps } from "/static/js/colormap.js";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";

export class Viewer {
  constructor(container) {
    this.el = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.001, 1000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.hemi = new THREE.HemisphereLight(0xdfe9ff, 0x1a1f28, 1.1);
    const d1 = new THREE.DirectionalLight(0xffffff, 1.6);
    d1.position.set(-2, -3, 4);
    const d2 = new THREE.DirectionalLight(0x9fc4ff, 0.6);
    d2.position.set(3, 2, -1);
    this.scene.add(this.hemi, d1, d2);
    this.root = new THREE.Group();          // everything that mirrors with the body
    this.scene.add(this.root);
    this.helpers = new THREE.Group();       // ground grid, flow arrow, tunnel box
    this.scene.add(this.helpers);
    this.zoneGroup = new THREE.Group();
    this.root.add(this.zoneGroup);
    this.propGroup = new THREE.Group();     // RC tunnel: spinning propeller disks
    this.root.add(this.propGroup);
    this.spinners = [];
    this.mover = null;                      // drag gizmo for moving a propeller
    this.shock = null;                      // supersonic tunnel: Mach cone from the nose
    this.mesh = this.ghost = this.slice = this.stream = this.particles = this.tunnelBox = null;
    this.mode = "shaded";
    this.fields = {};
    this.radius = 1;
    this._fitted = false;
    this.raycaster = new THREE.Raycaster();
    this.clock = new THREE.Clock();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    requestAnimationFrame(this._loop);
    const dt = Math.min(this.clock.getDelta(), 0.05);
    this.controls.update();
    if (this.particles && this.particles.visible) this._advanceParticles(dt);
    if (this.wind) this._advanceWind(dt);
    for (const s of this.spinners) s.obj.rotateOnAxis(s.axis, s.rate * dt);
    this.renderer.render(this.scene, this.camera);
  }

  // ---- geometry ---------------------------------------------------------------------
  setModel(positions, { fit = !this._fitted } = {}) {
    if (this.mesh) {
      this.root.remove(this.mesh);
      this.mesh.geometry.dispose();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.computeVertexNormals();
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(positions.length).fill(0.8), 3));
    g.computeBoundingSphere();
    g.computeBoundingBox();
    const mat = new THREE.MeshStandardMaterial({ color: 0xc9d2e0, metalness: 0.15, roughness: 0.55, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    this.mesh = new THREE.Mesh(g, mat);
    this.wire = new THREE.LineSegments(new THREE.WireframeGeometry(g),
      new THREE.LineBasicMaterial({ color: 0x6d7a90, transparent: true, opacity: 0.35 }));
    this.wire.visible = false;
    this.mesh.add(this.wire);
    this.root.add(this.mesh);
    this.radius = g.boundingSphere.radius || 1;
    this.fields = {};
    this._bb = g.boundingBox.clone();
    this._ground();
    this._rewind();
    if (fit) this.view("iso");
    this._fitted = true;
    this.setMode(this.mode in { shaded: 1, wire: 1 } ? this.mode : "shaded");
  }

  setGhost(positions) {
    if (this.ghost) {
      this.root.remove(this.ghost);
      this.ghost.geometry.dispose();
      this.ghost = null;
    }
    if (!positions) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.ghost = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x86b6ef, wireframe: true, transparent: true, opacity: 0.2 }));
    this.root.add(this.ghost);
  }

  // mirror the displayed solution top-to-bottom (negative angles of a symmetric body)
  setMirror(on) {
    this.root.scale.z = on ? -1 : 1;
  }

  setFaceField(name, values, lo, hi, map) {
    this.fields[name] = { values, lo, hi, map };
    if (this.mode === name) this._applyField(name);
  }

  setVertexField(name, values, lo, hi, map) {
    this.fields[name] = { values, lo, hi, map, perVertex: true };
    if (this.mode === name) this._applyField(name);
  }

  _applyField(name) {
    const f = this.fields[name];
    const col = this.mesh.geometry.attributes.color;
    for (let v = 0; v < col.count; v++) {
      const val = f.perVertex ? f.values[v] : f.values[Math.floor(v / 3)];
      const c = Number.isFinite(val) ? colorFor(f.map, val, f.lo, f.hi) : [0.45, 0.47, 0.52];
      col.setXYZ(v, c[0], c[1], c[2]);
    }
    col.needsUpdate = true;
  }

  setMode(mode) {
    this.mode = mode;
    if (!this.mesh) return;
    const m = this.mesh.material;
    const isField = mode in this.fields;
    m.vertexColors = isField;
    m.color.set(isField ? 0xffffff : 0xc9d2e0);
    m.roughness = isField ? 0.75 : 0.55;
    m.metalness = isField ? 0.0 : 0.15;
    m.transparent = mode === "wire";
    m.opacity = mode === "wire" ? 0.25 : 1;
    m.needsUpdate = true;
    this.wire.visible = mode === "wire";
    if (isField) this._applyField(mode);
  }

  // theme switch: re-read colours that come from CSS
  applyTheme() {
    refreshColormaps();
    this._ground();
    this._rewind();
    if (this.mode in this.fields) this._applyField(this.mode);
  }

  // ---- helpers --------------------------------------------------------------------------
  _ground() {
    this.helpers.remove(...this.helpers.children.filter((c) => c !== this.tunnelBox));
    const bb = this._bb;
    if (!bb) return;
    const size = bb.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y) * 3 || 1;
    const step = niceStep(span / 12);
    const n = Math.ceil(span / step);
    const grid = new THREE.GridHelper(n * step, n, new THREE.Color(css("--vp-grid")), new THREE.Color(css("--vp-grid-2")));
    grid.rotation.x = Math.PI / 2;
    grid.position.z = bb.min.z - size.z * 0.15;
    this.helpers.add(grid);
    const len = Math.max(size.x * 0.35, this.radius * 0.4);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(bb.min.x - len * 1.6, 0, bb.max.z + size.z * 0.3),
      len, new THREE.Color(css("--accent")), len * 0.25, len * 0.12);
    this.helpers.add(arrow);
  }

  setTunnel(bounds) {
    if (this.tunnelBox) {
      this.helpers.remove(this.tunnelBox);
      this.tunnelBox = null;
    }
    if (!bounds) return;
    const [a, b] = bounds;
    const geo = new THREE.BoxGeometry(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: new THREE.Color(css("--vp-grid")), transparent: true, opacity: 0.7 }));
    box.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
    this.tunnelBox = box;
    this.helpers.add(box);
  }

  // ---- protected regions ---------------------------------------------------------------------
  /** zones: [{ id, color, matrix (THREE.Matrix4 placing a unit box/cylinder), shape, selected }] */
  setZones(zones) {
    const tool = this.zoneTool && !this.zoneTool.tc.dragging ? this.zoneTool : null;
    if (tool) this.detachZoneTool(true);
    for (const c of [...this.zoneGroup.children]) {
      this.zoneGroup.remove(c);
      c.traverse((o) => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); });
    }
    for (const z of zones || []) {
      const geo = z.shape === "cylinder" ? new THREE.CylinderGeometry(0.5, 0.5, 1, 32, 1) : new THREE.BoxGeometry(1, 1, 1);
      if (z.shape === "cylinder") geo.rotateZ(-Math.PI / 2);   // axis along x
      const color = new THREE.Color(z.color);
      const fill = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: z.selected ? 0.22 : 0.12,
        depthWrite: false, side: THREE.DoubleSide }));
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 25), new THREE.LineBasicMaterial({ color, transparent: true, opacity: z.selected ? 1 : 0.75 }));
      const g = new THREE.Group();
      g.add(fill, edges);
      g.applyMatrix4(z.matrix);
      g.renderOrder = 3;
      g.userData.id = z.id;
      fill.userData.zoneId = z.id;
      this.zoneGroup.add(g);
    }
    if (tool && this._zoneToolArgs) this.attachZoneTool(...this._zoneToolArgs);
  }

  /** Zone id under the cursor, or null. */
  pickZone(clientX, clientY) {
    if (!this.zoneGroup.children.length) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const fills = this.zoneGroup.children.map((g) => g.children[0]);
    const hit = this.raycaster.intersectObjects(fills, false)[0];
    return hit ? hit.object.userData.zoneId : null;
  }

  /** Move / rotate / resize gizmo on zone `id`. onChange({ position, quaternion, scale }) in display
   *  axes while dragging; onEnd() on release. mode: "translate" | "rotate" | "scale". */
  attachZoneTool(id, mode, { snap = false, onChange, onEnd } = {}) {
    this.detachZoneTool(true);
    const target = this.zoneGroup.children.find((g) => g.userData.id === id);
    if (!target) return false;
    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.setMode(mode);
    tc.setSpace(mode === "translate" ? "world" : "local");
    tc.setSize(0.85);
    tc.setRotationSnap(snap ? THREE.MathUtils.degToRad(15) : null);
    tc.setScaleSnap(snap ? 0.05 : null);
    tc.attach(target);
    tc.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
      if (!e.value && onEnd) onEnd();
    });
    tc.addEventListener("objectChange", () => onChange && onChange({ position: target.position.clone(),
      quaternion: target.quaternion.clone(), scale: target.scale.clone() }));
    const helper = tc.getHelper ? tc.getHelper() : tc;
    this.root.add(helper);
    this.zoneTool = { tc, helper, id, target };
    this._zoneToolArgs = [id, mode, { snap, onChange, onEnd }];
    return true;
  }

  detachZoneTool(keepArgs = false) {
    if (!this.zoneTool) return;
    this.zoneTool.tc.detach();
    this.root.remove(this.zoneTool.helper);
    this.zoneTool.tc.dispose();
    this.zoneTool = null;
    this.controls.enabled = true;
    if (!keepArgs) this._zoneToolArgs = null;
  }

  // ---- propellers (RC tunnel) ----------------------------------------------------------------
  /** props: [{ center: [x,y,z], axis: unit thrust direction, radius, blades, spin: +1/-1, color }] in display axes */
  setProps(props) {
    const moving = this.mover && this.mover.userId;
    if (moving) this.detachMover(true);
    for (const c of [...this.propGroup.children]) {
      this.propGroup.remove(c);
      c.traverse((o) => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); });
    }
    this.spinners = [];
    for (const p of props || []) {
      const color = new THREE.Color(p.color || css("--rc"));
      const g = new THREE.Group();
      g.userData.id = p.id;
      const axis = new THREE.Vector3(...p.axis).normalize();
      g.position.set(...p.center);
      g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
      const disk = new THREE.Mesh(new THREE.CircleGeometry(p.radius, 48),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }));
      const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
        Array.from({ length: 64 }, (_, i) => new THREE.Vector3(Math.cos((i / 64) * 2 * Math.PI) * p.radius, Math.sin((i / 64) * 2 * Math.PI) * p.radius, 0))),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
      const rotor = new THREE.Group();
      const bladeMat = new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 0.6, metalness: 0.1 });
      for (let k = 0; k < (p.blades || 2); k++) {
        const blade = new THREE.Mesh(new THREE.BoxGeometry(p.radius * 0.92, p.radius * (p.chord || 0.11), p.radius * 0.02), bladeMat);
        blade.position.x = p.radius * 0.5;
        const arm = new THREE.Group();
        arm.rotation.z = (k / (p.blades || 2)) * 2 * Math.PI;
        arm.add(blade);
        rotor.add(arm);
      }
      const hub = new THREE.Mesh(new THREE.SphereGeometry(p.radius * 0.1, 16, 12), bladeMat);
      rotor.add(hub);
      g.add(disk, rim, rotor);
      g.renderOrder = 4;
      this.propGroup.add(g);
      this.spinners.push({ obj: rotor, axis: new THREE.Vector3(0, 0, 1), rate: 9 * (p.spin || 1) });
    }
    if (moving && this._moverArgs) this.attachMover(...this._moverArgs);
  }

  /** Drag gizmo on propeller `id`: onMove(displayPosition) while dragging, onEnd() when released. */
  attachMover(id, onMove, onEnd) {
    this.detachMover(true);
    const target = this.propGroup.children.find((g) => g.userData.id === id);
    if (!target) return false;
    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.setMode("translate");
    tc.setSize(0.8);
    tc.attach(target);
    tc.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
      if (!e.value && onEnd) onEnd();
    });
    tc.addEventListener("objectChange", () => onMove && onMove([target.position.x, target.position.y, target.position.z]));
    const helper = tc.getHelper ? tc.getHelper() : tc;
    this.root.add(helper);
    this.mover = { tc, helper, userId: id };
    this._moverArgs = [id, onMove, onEnd];
    return true;
  }

  detachMover(keepArgs = false) {
    if (!this.mover) return;
    this.mover.tc.detach();
    this.root.remove(this.mover.helper);
    this.mover.tc.dispose();
    this.mover = null;
    this.controls.enabled = true;
    if (!keepArgs) this._moverArgs = null;
  }

  /** True while a gizmo is being dragged (so a click on it is not a model pick). */
  get dragging() { return !!((this.mover && this.mover.tc.dragging) || (this.zoneTool && this.zoneTool.tc.dragging)); }
  /** True while the pointer is over a gizmo handle. */
  get overGizmo() { return !!((this.mover && this.mover.tc.axis) || (this.zoneTool && this.zoneTool.tc.axis)); }

  // ---- Mach cone (supersonic tunnel) ------------------------------------------------------------
  /** Cone of half-angle `angle` (rad) opening downstream (+x) from `apex`, `length` long. */
  setShock(apex, angle, length) {
    if (this.shock) {
      this.root.remove(this.shock);
      this.shock.traverse((o) => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); });
      this.shock = null;
    }
    if (!apex) return;
    const r = Math.tan(angle) * length;
    const geo = new THREE.ConeGeometry(r, length, 64, 1, true);
    geo.rotateZ(Math.PI / 2);                 // axis along x, apex towards -x
    geo.translate(length / 2, 0, 0);
    const color = new THREE.Color(css("--heat"));
    const g = new THREE.Group();
    g.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false })));
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.ConeGeometry(r, length, 24, 1, true).rotateZ(Math.PI / 2).translate(length / 2, 0, 0), 1),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 })));
    g.position.set(...apex);
    g.renderOrder = 1;
    this.shock = g;
    this.root.add(g);
  }

  // ---- "tunnel running" animation ---------------------------------------------------------------
  /** Air streaks flowing past the model while a job runs: potential flow around the model's
   *  bounding ellipsoid. A picture of what the tunnel is doing, not a result. */
  setWind(on, { fast = false } = {}) {
    if (!on || !this._bb) {
      if (this.wind) {
        this.helpers.remove(this.wind.group);
        this.wind.group.traverse((o) => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); });
        this.wind = null;
      }
      return;
    }
    if (this.wind) { this.wind.fast = fast; return; }
    const bb = this._bb, c = bb.getCenter(new THREE.Vector3()), size = bb.getSize(new THREE.Vector3());
    const m = Math.max(size.x, size.y, size.z) / 2 || 1;
    const A = [size.x, size.y, size.z].map((s) => Math.max((s / 2) * 1.18, m * 0.06));   // ellipsoid half-axes
    const x0 = c.x - A[0] - 1.1 * m, x1 = c.x + A[0] + 1.5 * m;
    const ry = A[1] * 1.7 + 0.3 * m, rz = A[2] * 1.7 + 0.3 * m;
    const N = 520;
    const p = new Float32Array(N * 3), seg = new Float32Array(N * 6), col = new Float32Array(N * 8);
    const spawn = (i, anywhere) => {
      p[i * 3] = anywhere ? x0 + Math.random() * (x1 - x0) : x0 - Math.random() * 0.1 * (x1 - x0);
      const t = Math.random() * 2 * Math.PI, r = Math.sqrt(Math.random());
      p[i * 3 + 1] = c.y + Math.cos(t) * r * ry;
      p[i * 3 + 2] = c.z + Math.sin(t) * r * rz;
    };
    for (let i = 0; i < N; i++) spawn(i, true);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(seg, 3));
    lineGeo.setAttribute("color", new THREE.BufferAttribute(col, 4));
    const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute("position", new THREE.BufferAttribute(p, 3));
    const pts = new THREE.Points(ptGeo, new THREE.PointsMaterial({ color: new THREE.Color(css("--accent")), size: m * 0.022,
      sizeAttenuation: true, transparent: true, opacity: 0.7, depthWrite: false }));
    const group = new THREE.Group();
    group.add(lines, pts);
    group.renderOrder = 2;
    this.helpers.add(group);
    const accent = new THREE.Color(css("--accent"));
    this.wind = { group, lines, pts, p, seg, col, N, c, A, x0, x1, spawn, accent, fast, U: (x1 - x0) / 2.4 };
  }

  _rewind() {        // the model or theme changed: rebuild the streaks around it
    if (!this.wind) return;
    const fast = this.wind.fast;
    this.setWind(false);
    this.setWind(true, { fast });
  }

  _advanceWind(dt) {
    const w = this.wind, { p, seg, col, c, A, x0, x1, accent } = w;
    const U = w.U * (w.fast ? 2.2 : 1);
    const tail = 0.11;
    for (let i = 0; i < w.N; i++) {
      let x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      // flow past a unit sphere in ellipsoid-normalised coordinates
      let ex = (x - c.x) / A[0], ey = (y - c.y) / A[1], ez = (z - c.z) / A[2];
      let r2 = ex * ex + ey * ey + ez * ez;
      if (r2 < 1) {                       // never inside the body: step out to the surface
        const s = 1.001 / Math.sqrt(r2 || 1e-9);
        ex *= s; ey *= s; ez *= s; r2 = 1.002;
        x = c.x + ex * A[0]; y = c.y + ey * A[1]; z = c.z + ez * A[2];
      }
      const r3 = r2 * Math.sqrt(r2), r5 = r3 * r2;
      const un = Math.max(1 + 0.5 / r3 - (1.5 * ex * ex) / r5, 0.12);
      const vn = (-1.5 * ex * ey) / r5, wn = (-1.5 * ex * ez) / r5;
      const vx = U * un, vy = U * vn * (A[1] / A[0]), vz = U * wn * (A[2] / A[0]);
      x += vx * dt; y += vy * dt; z += vz * dt;
      p[i * 3] = x; p[i * 3 + 1] = y; p[i * 3 + 2] = z;
      if (x > x1) { w.spawn(i, false); continue; }
      seg[i * 6] = x; seg[i * 6 + 1] = y; seg[i * 6 + 2] = z;
      seg[i * 6 + 3] = x - vx * tail; seg[i * 6 + 4] = y - vy * tail; seg[i * 6 + 5] = z - vz * tail;
      const u = (x - x0) / (x1 - x0);
      const fade = Math.max(0, Math.min(1, u / 0.12, (1 - u) / 0.18));
      const glow = Math.min(1, 0.55 + 0.45 * (un - 0.6));        // brighter where the flow speeds up
      col[i * 8] = accent.r; col[i * 8 + 1] = accent.g; col[i * 8 + 2] = accent.b; col[i * 8 + 3] = 0.7 * fade * glow;
      col[i * 8 + 4] = accent.r; col[i * 8 + 5] = accent.g; col[i * 8 + 6] = accent.b; col[i * 8 + 7] = 0;
    }
    w.lines.geometry.attributes.position.needsUpdate = true;
    w.lines.geometry.attributes.color.needsUpdate = true;
    w.pts.geometry.attributes.position.needsUpdate = true;
  }

  /** Point on the model under the cursor, in display coordinates (before mirroring). */
  pick(clientX, clientY) {
    if (!this.mesh) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) return null;
    return this.root.worldToLocal(hit.point.clone());
  }

  // ---- slices ------------------------------------------------------------------------------
  setSlice(slices, which, field) {
    if (this.slice) {
      this.root.remove(this.slice);
      this.slice.geometry.dispose();
      this.slice.material.map.dispose();
      this.slice.material.dispose();
      this.slice = null;
    }
    if (!slices || !which || !slices[which]) return;
    const s = slices[which];
    const [n0, n1] = s.shape;
    const codes = b64bytes(field === "cp" ? s.cp : s.speed);
    const [lo, hi] = field === "cp" ? s.cp_range : s.speed_range;
    const vals = dequantize(codes, lo, hi);
    const data = new Uint8Array(n0 * n1 * 4);
    for (let i = 0; i < n0; i++) {
      for (let j = 0; j < n1; j++) {
        const v = vals[i * n1 + j];
        const o = (j * n0 + i) * 4;
        if (!Number.isFinite(v)) { data[o + 3] = 0; continue; }
        const c = colorFor(field === "cp" ? "cp" : "speed", v, field === "cp" ? -1.0 : 0, field === "cp" ? 1.0 : 1.4);
        data[o] = c[0] * 255; data[o + 1] = c[1] * 255; data[o + 2] = c[2] * 255; data[o + 3] = 235;
      }
    }
    const tex = new THREE.DataTexture(data, n0, n1, THREE.RGBAFormat);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    const [x0, x1, y0, y1] = s.extent;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    if (which === "xz") {
      plane.rotation.x = Math.PI / 2;
      plane.position.set((x0 + x1) / 2, s.y, (y0 + y1) / 2);
    } else {
      plane.position.set((x0 + x1) / 2, (y0 + y1) / 2, s.z);
    }
    plane.renderOrder = 1;
    this.slice = plane;
    this.root.add(plane);
  }

  // ---- streamlines ----------------------------------------------------------------------------
  setStreamlines(sl) {
    for (const o of [this.stream, this.particles]) {
      if (o) { this.root.remove(o); o.geometry.dispose(); o.material.dispose(); }
    }
    this.stream = this.particles = null;
    if (!sl) return;
    const pts = b64f32(sl.points);
    const spd = b64f32(sl.speed);
    const starts = sl.starts;
    const segs = [], cols = [];
    this.lines = [];
    for (let k = 0; k < starts.length - 1; k++) {
      const a = starts[k], b = starts[k + 1];
      if (b - a < 3) continue;
      const cum = [0];
      for (let i = a + 1; i < b; i++) {
        const dx = pts[i * 3] - pts[(i - 1) * 3], dy = pts[i * 3 + 1] - pts[(i - 1) * 3 + 1], dz = pts[i * 3 + 2] - pts[(i - 1) * 3 + 2];
        cum.push(cum[cum.length - 1] + Math.hypot(dx, dy, dz));
        segs.push(pts[(i - 1) * 3], pts[(i - 1) * 3 + 1], pts[(i - 1) * 3 + 2], pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
        for (const j of [i - 1, i]) {
          const c = colorFor("speed", spd[j], 0, 1.4);
          cols.push(c[0], c[1], c[2]);
        }
      }
      this.lines.push({ a, b, cum, len: cum[cum.length - 1] });
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    this.stream = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.35, depthWrite: false }));
    this.root.add(this.stream);
    this.streamPts = pts;
    const per = 3;
    const pg = new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.lines.length * per * 3), 3));
    this.phase = new Float32Array(this.lines.length * per).map(() => Math.random());
    const pm = new THREE.PointsMaterial({ color: 0x9fd0ff, size: this.radius * 0.016, sizeAttenuation: true, transparent: true,
      opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending });
    this.particles = new THREE.Points(pg, pm);
    this.particles.renderOrder = 2;
    this.root.add(this.particles);
    this.perLine = per;
  }

  showStreamlines(on) {
    if (this.stream) this.stream.visible = on;
    if (this.particles) this.particles.visible = on;
  }

  _advanceParticles(dt) {
    const pos = this.particles.geometry.attributes.position;
    const pts = this.streamPts;
    let idx = 0;
    for (const L of this.lines) {
      for (let p = 0; p < this.perLine; p++, idx++) {
        let ph = (this.phase[idx] += dt * 0.12);
        if (ph > 1) ph = this.phase[idx] -= 1;
        const target = ph * L.len;
        let lo = 0, hi = L.cum.length - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (L.cum[mid] < target) lo = mid; else hi = mid;
        }
        const f = (target - L.cum[lo]) / (L.cum[hi] - L.cum[lo] || 1);
        const i0 = (L.a + lo) * 3, i1 = (L.a + hi) * 3;
        pos.setXYZ(idx, pts[i0] + (pts[i1] - pts[i0]) * f, pts[i0 + 1] + (pts[i1 + 1] - pts[i0 + 1]) * f, pts[i0 + 2] + (pts[i1 + 2] - pts[i0 + 2]) * f);
      }
    }
    pos.needsUpdate = true;
  }

  // ---- camera --------------------------------------------------------------------------------
  view(name) {
    if (!this.mesh) return;
    const c = this._bb.getCenter(new THREE.Vector3());
    const r = this.radius;
    const dist = (r / Math.sin((this.camera.fov * Math.PI) / 360)) * 1.12;
    const dirs = { iso: [-1, -1.1, 0.75], front: [-1, 0, 0.0001], side: [0, -1, 0.0001], top: [0.0001, 0, 1], rear: [1, 0.0001, 0.2] };
    const d = new THREE.Vector3(...(dirs[name] || dirs.iso)).normalize();
    this.camera.position.copy(c).addScaledVector(d, dist);
    this.camera.near = r / 100;
    this.camera.far = r * 200;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(c);
    this.controls.update();
  }

  snapshot(w = 640, h = 400) {
    const old = new THREE.Vector2();
    this.renderer.getSize(old);
    const aspect = this.camera.aspect;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
    const url = this.renderer.domElement.toDataURL("image/png");
    this.renderer.setSize(old.x, old.y, false);
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    return url;
  }

  dispose() {
    this.ro.disconnect();
    this.renderer.dispose();
  }
}

function niceStep(x) {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const m = x / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

// ---- frames (mirror of aerosculpt/optimize/runner.py ModelFrame + wind_rotation) ------------
const AXES = { "+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1] };
export const UNIT_SCALE = { mm: 1e-3, cm: 1e-2, m: 1, in: 0.0254, ft: 0.3048 };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function windRotation(aoa = 0, yaw = 0) {
  const a = (aoa * Math.PI) / 180, b = (yaw * Math.PI) / 180;
  const Ra = [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
  const Rb = [[Math.cos(b), Math.sin(b), 0], [-Math.sin(b), Math.cos(b), 0], [0, 0, 1]];
  return Rb.map((r) => [0, 1, 2].map((j) => r[0] * Ra[0][j] + r[1] * Ra[1][j] + r[2] * Ra[2][j]));
}

/** File coordinates <-> body frame (metres, nose -x, z up, centred on the bounding box). */
export class ModelFrame {
  constructor(raw, { units, forward, up }) {
    const f = AXES[forward], u = AXES[up];
    this.R0 = [f.map((v) => -v), cross(u, f.map((v) => -v)), u];
    this.s = UNIT_SCALE[units] || 1;
    const n = raw.length / 3;
    const body = new Float32Array(raw.length);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      const x = raw[i * 3] * this.s, y = raw[i * 3 + 1] * this.s, z = raw[i * 3 + 2] * this.s;
      for (let k = 0; k < 3; k++) {
        const v = this.R0[k][0] * x + this.R0[k][1] * y + this.R0[k][2] * z;
        body[i * 3 + k] = v;
        if (v < lo[k]) lo[k] = v;
        if (v > hi[k]) hi[k] = v;
      }
    }
    this.c = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) body[i * 3 + k] -= this.c[k];
    this.body = body;
    this.size = [0, 1, 2].map((k) => hi[k] - lo[k]);
  }

  fileToBody(p) {
    const x = p[0] * this.s, y = p[1] * this.s, z = p[2] * this.s;
    return this.R0.map((r, k) => r[0] * x + r[1] * y + r[2] * z - this.c[k]);
  }

  bodyToFile(b) {
    const q = [b[0] + this.c[0], b[1] + this.c[1], b[2] + this.c[2]];
    return [0, 1, 2].map((j) => (this.R0[0][j] * q[0] + this.R0[1][j] * q[1] + this.R0[2][j] * q[2]) / this.s);
  }

  /** half extents of a file-frame size vector, in body axes */
  sizeToBody(size) {
    return this.R0.map((r) => (Math.abs(r[0]) * size[0] + Math.abs(r[1]) * size[1] + Math.abs(r[2]) * size[2]) * this.s);
  }
}

export function rotate(positions, R) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    out[i] = R[0][0] * x + R[0][1] * y + R[0][2] * z;
    out[i + 1] = R[1][0] * x + R[1][1] * y + R[1][2] * z;
    out[i + 2] = R[2][0] * x + R[2][1] * y + R[2][2] * z;
  }
  return out;
}

export function applyMat3(R, p) {
  return [0, 1, 2].map((k) => R[k][0] * p[0] + R[k][1] * p[1] + R[k][2] * p[2]);
}

export function transposeMat3(R) {
  return [0, 1, 2].map((i) => [R[0][i], R[1][i], R[2][i]]);
}

const mat4 = (R) => new THREE.Matrix4().set(R[0][0], R[0][1], R[0][2], 0, R[1][0], R[1][1], R[1][2], 0, R[2][0], R[2][1], R[2][2], 0, 0, 0, 0, 1);

/** Zone turn (roll, pitch, yaw in degrees about the body axes, applied x then y then z) as a
 *  quaternion; mirrors geometry/zones.py rotation_matrix. */
export function zoneQuaternion(rotDeg) {
  const r = (rotDeg || [0, 0, 0]).map((a) => THREE.MathUtils.degToRad(a));
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(r[0], r[1], r[2], "ZYX"));
}

/** THREE.Matrix4 that places a unit box/cylinder (centred, size 1) as a body-frame zone, turned by
 *  rotDeg about its centre, shown at display rotation R. */
export function zoneMatrix(centerBody, sizeBody, R, rotDeg) {
  const t = new THREE.Matrix4().makeTranslation(centerBody[0], centerBody[1], centerBody[2]);
  const q = new THREE.Matrix4().makeRotationFromQuaternion(zoneQuaternion(rotDeg));
  const s = new THREE.Matrix4().makeScale(Math.max(sizeBody[0], 1e-6), Math.max(sizeBody[1], 1e-6), Math.max(sizeBody[2], 1e-6));
  return mat4(R).multiply(t).multiply(q).multiply(s);
}

/** Inverse of zoneMatrix for a gizmo edit: display-axes pose -> body centre, body size, turn (deg). */
export function zoneFromPose({ position, quaternion, scale }, R) {
  const Rt = new THREE.Quaternion().setFromRotationMatrix(mat4(R)).invert();
  const c = position.clone().applyQuaternion(Rt);
  const e = new THREE.Euler().setFromQuaternion(Rt.multiply(quaternion.clone()), "ZYX");
  const deg = [e.x, e.y, e.z].map((a) => Math.round(THREE.MathUtils.radToDeg(a) * 10) / 10);
  return { center: [c.x, c.y, c.z], size: [Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z)], rotation: deg };
}

export function parseSTL(buf) {
  const dv = new DataView(buf);
  if (buf.byteLength >= 84) {
    const n = dv.getUint32(80, true);
    if (84 + n * 50 === buf.byteLength) {
      const out = new Float32Array(n * 9);
      for (let t = 0; t < n; t++) {
        const o = 84 + t * 50 + 12;
        for (let k = 0; k < 9; k++) out[t * 9 + k] = dv.getFloat32(o + k * 4, true);
      }
      return out;
    }
  }
  const txt = new TextDecoder().decode(buf);
  const re = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;
  const vals = [];
  let m;
  while ((m = re.exec(txt))) vals.push(+m[1], +m[2], +m[3]);
  return new Float32Array(vals);
}

export function parseOBJ(buf) {
  const lines = new TextDecoder().decode(buf).split(/\r?\n/);
  const v = [], out = [];
  for (const line of lines) {
    if (line.startsWith("v ")) {
      const p = line.trim().split(/\s+/);
      v.push([+p[1], +p[2], +p[3]]);
    } else if (line.startsWith("f ")) {
      const idx = line.trim().split(/\s+/).slice(1).map((t) => {
        const i = parseInt(t.split("/")[0], 10);
        return i > 0 ? i - 1 : v.length + i;
      });
      for (let k = 1; k < idx.length - 1; k++) for (const j of [idx[0], idx[k], idx[k + 1]]) out.push(...v[j]);
    }
  }
  return new Float32Array(out);
}

export function parseModel(buf, filename) {
  return filename.toLowerCase().endsWith(".obj") ? parseOBJ(buf) : parseSTL(buf);
}
