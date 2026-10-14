# AeroSculpt

A web application with three wind tunnels for rocket, UAV and RC engineers: put a CAD or mesh
model through the tunnel that matches how it flies, across a range of speeds and angles of
attack, then reshape it automatically to cut drag while protecting the parts that must not
change. Free **Lite** plan, **Pro** with a free trial, and **Team** — billed monthly through Stripe.

| Tunnel | Speeds | Physics |
|---|---|---|
| **RC** | 2–60 m/s | General tunnel + propellers: blade-element momentum theory with two measured blade families and 2–6 blades (checked against UIUC data, thrust/power within 8 %) as actuator disks in the CFD |
| **General** | to Mach 1.2 | 3D lattice-Boltzmann LES (below); Kármán–Tsien compressibility corrections to the critical Mach number, a transonic estimate above it |
| **Supersonic** | Mach 1.2–12, 0–80 km | Surface methods: second-order shock-expansion along body meridians (tangent cone at the tip), tangent wedge, Newtonian, Prandtl–Meyer; Eckert friction and heating, Fay–Riddell stagnation heating, radiative-equilibrium wall temperature against a material library |

Above each body's critical Mach number (typically 0.7–0.85) the General tunnel's drag is a labelled
transonic estimate that meets the Supersonic tunnel at Mach 1.2.

- Upload STL, OBJ, STEP, IGES, BREP, 3MF, PLY or glTF (CAD formats are tessellated in the browser).
- **Flight-envelope sweeps:** airspeed range × angle-of-attack range. One CFD run per angle;
  every airspeed comes from the same run (exact in the hybrid regime). Top-to-bottom symmetric
  bodies mirror negative angles instead of re-simulating them.
- **Protected regions:** protect (frozen geometry, smooth fade-out), keep-inside (payload /
  motor volume must stay enclosed) and design-only zones, placed by clicking the model.
- Drag polar, drag/power vs speed, pressure/friction breakdown, surface pressure, slices and
  animated streamlines at every angle. Lift curve, L/D and pitching moment are shown but
  marked indicative: lift on thin wings is not validated yet (50–66% low, see `/validation`).
- **Templates with named dimensions**: rocket, fixed-wing UAV and wing. Nose profile (conical,
  ogive, elliptical, power law, von Kármán), nose length and tip radius; body and boat-tail; wing
  span, chords, sweep, NACA 4/5-digit section (continuous), dihedral, twist, incidence, position.
  Each has a unit, slider, number box, optimizer range and lock; the 3D view updates live.
- **Optimizer that spends runs carefully**: measures the solver's run-to-run error, screens and
  ranks every free parameter, Bayesian-optimizes only the most influential, stops when gains fall
  inside the error, confirms the best at full resolution and says "no reliable improvement"
  when that is the truth. Candidates are checked (limits, volume, regions, watertight, no
  self-intersection, stall, stability) before any run; identical runs are cached.
- **Wings at fixed lift** with a validated wing model (vortex lattice + NeuralFoil sections +
  Hoerner fuselage), so the optimizer cannot win by shrinking the wing. CFD lift-based goals
  stay off (`runner.LIFT_VALIDATED`) until the CFD wing case passes.
- RC tunnel: propellers by blade family, size (inches), pitch, 2–6 blades, rpm and rotation; place
  them by clicking (snapped to the local centre line), then drag them in 3D, nudge them or snap them
  to the nose, tail or centre line; live free-air thrust/power/slipstream per option; thrust, shaft
  power, efficiency, drag in the slipstream and the level top speed.
- Supersonic tunnel: wave/base/friction drag vs Mach, centre of pressure, wall temperature, heat
  flux and margin maps, material regions (nose caps, leading edges); optimize for drag or peak
  temperature within material limits — hundreds of designs in seconds.
- Light and dark themes, plans, credits, trials, Stripe Checkout + Portal + webhooks, REST API,
  printable reports, admin overview.

## Run it locally

Requires Python 3.11+ (developed on 3.13 with numpy 2.1 and numba 0.61).

```bash
pip install -r requirements.txt
python run.py serve
```

Open http://localhost:8000 (or double-click `Start AeroSculpt.bat` on Windows). On localhost the app
runs in **local preview**: sign-in is skipped and every feature is unlocked. Set `OPEN_ACCESS=0`
to use real accounts; the first account becomes the admin.
Without Stripe keys billing runs in **demo mode**: upgrades are granted instantly with no
payment, and a banner says so on every page. The first CFD job compiles the solver
kernels (about 30 s once; cached afterwards).

### NVIDIA GPU (optional, about 12x faster)

```bash
pip install "numba-cuda[cu12]==0.30.4"
```

With an NVIDIA GPU and that package the wind tunnel runs on the GPU automatically; without them it
runs on the CPU. `AEROSCULPT_GPU=0` forces the CPU, `=1` refuses to start without the GPU. Both use
kernels generated from one source (`tools/gen_kernels.py`) and agree to 0.2 %. Measured on a laptop
(i7-10750H, GTX 1650 Ti): a 3-angle draft sweep of the rocket takes 9 s on the GPU against 56 s on
the CPU; a standard-fidelity sphere 10 s against 135 s. Each worker measures its speed at start-up
and the workspace shows the expected time per run from it.

```bash
python -m pytest -q          # 88 tests: geometry, templates, wing model, solver, optimizer, API, billing
python run.py validate       # full validation suite (~4 min on the GPU, ~60 min on the CPU), feeds /validation
python run.py validate supersonic,rc,wing   # the fast groups (~1 min)
```

## A public link from this computer

Double-click `Start public link.bat` (or `python scripts/public_link.py you@example.com`). It
starts a Cloudflare quick tunnel (`tools/cloudflared.exe`, no Cloudflare account needed) and the
app behind it, and prints an `https://….trycloudflare.com` address that works from any browser.
On that address sign-in is required, demo billing is off and only the listed emails may sign up
(`ALLOWED_SIGNUP_EMAILS`); the first is the owner (`OWNER_EMAILS`: admin, Team plan, no billing).
The site runs on this computer: it is online only while the window is open, and the address
changes every time it starts. Front-end edits show on reload; Python edits need a restart (and so
a new address). For a fixed address use a named Cloudflare tunnel (needs a Cloudflare account and
a domain) or a server (below).

`Start public demo.bat` (or `--demo`) also turns on the **guest demo** (`GUEST_DEMO=1`): visitors
press "Try the live demo" and get a temporary account on the hidden Guest plan (30 credits, draft
and standard fidelity, lowest queue priority, at most `GUEST_MAX_ACTIVE` at once, 4 per network an
hour). The account, its projects and its runs are deleted after `GUEST_HOURS` (24).

## Going live

1. **Server.** One machine with many cores is the sweet spot: each CFD worker uses every core.
   `docker compose up -d --build` runs the app behind Caddy with automatic HTTPS
   (set `DOMAIN=your.domain` and fill in `.env` from `.env.example`). Back up the `/data` volume.
2. **Stripe.** Create an account, then with a *test* key:
   `STRIPE_SECRET_KEY=sk_test_... PUBLIC_URL=https://your.domain python scripts/stripe_setup.py`.
   It creates the Pro ($49) and Team ($199) monthly prices, the customer-portal configuration
   and the webhook endpoint, and prints the values for `.env`. Test a full trial sign-up with
   Stripe's test card, then repeat with the live key. Set `ALLOW_DEMO_BILLING=0`.
3. **Email.** Set the `SMTP_*` variables so password-reset emails are delivered.
4. **Legal.** `/terms` and `/privacy` are templates with `[placeholders]`. Fill them in and have
   them reviewed by a lawyer for your jurisdiction before charging customers.
5. **Validation.** Run `python run.py validate` on the production machine and commit the result.

Prices, credits and limits live in `aerosculpt/plans.py`; the pricing page and the
enforcement both read from there.

## Shape features

Uploads (and the rocket template) take a timeline of shape features (`geometry/features.py`): nose
and tail cones, fairings over exposed parts (the convex hull of what is inside a box, drawn into a
nose and tail), pods, NACA wings, fin sets, slant cuts (plane cut with a flat cap), wing-panel
edits (span, chord, thickness, sweep, dihedral on an uploaded wing) and region move/scale/turn.
Added parts are extra closed shells: the tunnel fills their union, and faces buried inside another
shell are marked hidden so they are neither wetted area nor surface pressure. Each dimension has a
lock and an optimizer range; free ones become search variables (`optimize/spaces.FeatureSpace`).

## Automatic optimization and lift goals

- Press Optimize and the default mode (`optimize.mode: "auto"`) uses every whole-body change the plan
  allows plus generated ideas (boat-tail, rounded front, rear slant, rear diffuser). Each confirmed gain
  starts a new round of ideas on the improved shape (up to 4 rounds) until a round finds nothing above
  the measured error. `"mode": "custom"` keeps the hand-picked aspects.
- Goals: lowest drag, best L/D, lowest drag keeping the lift (Pro/Team). The lift-based goals are
  indicative until lift passes validation. Results show the drag polar (lift vs drag, one point per
  angle, needs 2+ angles) and lift and drag force vs airspeed.
- The workspace opens in a simple view; "Show advanced options" brings back shape features, the
  aspect list, the exact budget and the limits.

## Generative design, solid repair and the AI assistant

- `features.propose` / `features.idea`: boat-tail (body section lofted and tapered), rounded front,
  rear slant, rear diffuser, each optional. `FeatureSpace` gives each a `F.<id>._use` off/on choice
  (baseline off); `Search` adopts every choice that beats the error, one at a time, then rebases so
  the adopted ideas' dimensions are screened and searched.
- Paired screening (`Search(paired=...)`): each screened design also runs at the first sub-cell
  shift against the shifted baseline; the error is the spread of the two comparisons. Confirmation
  is paired the same way. Unsteady wakes average until the drag's standard error is under 0.6 %.
- Bodies thinner than 12 cells on the exploration grid are explored on the chosen grid (GPU only).
- `geometry/repair.solidify`: surface voxels -> close gaps -> flood the outside -> erode -> marching
  cubes -> Taubin smoothing; `POST /api/projects/{pid}/solidify` makes a solid copy.
- `units: "custom"` + `unit_scale` (metres per file unit) for scale models ("Set the real length").
- `aerosculpt/ai.py`: optional, OpenAI-compatible (`AI_BASE_URL`, `AI_MODEL`, `AI_API_KEY`); e.g.
  Ollama on this PC: `AI_BASE_URL=http://127.0.0.1:11434/v1 AI_MODEL=qwen3:4b`.

## How it works

```
browser (ES modules, three.js)          server (Starlette + uvicorn)          worker process(es)
 upload / convert CAD  ──POST──▶  /api/projects  ──▶ data/projects/…
 configure + submit    ──POST──▶  /api/projects/{id}/jobs ──▶ jobs table ──▶ claim by priority
 poll progress         ◀──GET───  /api/jobs/{id}            ◀── progress ── Runner:
 3D results            ◀──GET───  /api/jobs/{id}/files/…    ◀── files ───   mesh → voxels → LBM
                                                                           → BO loop → export
```

| Path | Role |
|---|---|
| `aerosculpt/cfd/kernels.py` | D3Q19 stream–collide kernel (generated by `tools/gen_kernels.py`, numba) |
| `aerosculpt/cfd/voxelize.py` | shell-wise parity voxelization, thin-feature rescue, wall-link fractions |
| `aerosculpt/cfd/tunnel.py` | domain sizing, boundary conditions, sponges, convergence on Cd |
| `aerosculpt/cfd/analysis.py` | one evaluation: regimes, blockage correction, surface pressure, friction, fields |
| `aerosculpt/cfd/validation.py` | reference cases and the published data they are checked against |
| `aerosculpt/geometry/` | mesh I/O and repair, free-form deformation, sample bodies, parametric templates (`templates.py`), validity checks (`checks.py`) |
| `aerosculpt/aero/` | NACA sections, NeuralFoil (numpy port, MIT), vortex lattice, wing model, Hoerner fuselage |
| `aerosculpt/rc/` | propeller BEMT (`propeller.py`), UIUC reference data, actuator-disk forcing, RC job runner |
| `aerosculpt/hyper/` | US76 atmosphere, shock/expansion tables, surface aerothermodynamics (`panel.py`), materials, supersonic job runner |
| `aerosculpt/optimize/` | Gaussian-process Bayesian optimization, design spaces, the screening/search/confirm strategy, the job runner |
| `aerosculpt/web.py`, `jobs.py`, `billing.py`, `plans.py` | API, queue, Stripe, plan limits |
| `aerosculpt/static/` | site, app and workspace front end |

### Solver in one paragraph

A uniform voxel tunnel is sized around the body (1.5 near-field lengths upstream and to the
sides, 4 downstream, absorbing sponges). The lattice-Boltzmann LES uses a regularized collision
with a Smagorinsky sub-grid model and a separately relaxed bulk viscosity that damps acoustics;
walls are Bouzidi interpolated bounce-back on the true triangle surface, so forces change
smoothly with shape (which the optimizer needs). Forces come from momentum exchange with a
Maskell blockage correction. Runs stop when the windowed mean Cd moves by less than 1%. Where
the grid cannot resolve the flight Reynolds number the *hybrid* model takes pressure forces
from the simulation and re-scales skin friction to flight conditions. See `/validation` for
accuracy per regime — including the known conservative bias on highly streamlined bodies and
the under-predicted lift on thin wings (both from boundary layers the uniform grid cannot resolve).

## Scaling notes

SQLite in WAL mode is fine for one server with a few workers. For several machines, move the
database to Postgres (the SQL is plain) and the `data/` directory to shared or object storage,
then run `python run.py web` on the front end and `python run.py worker` on compute nodes.
