# Wave engine contract

This document fixes the interfaces between the parts of the new wave engine so
they can be built in parallel. The reference implementation of the physics is
`src/boussinesq.js` (double precision, Node, tested by
`test/boussinesq.test.mjs`); everything numerical in the GPU port must
reproduce it.

## 1. Coordinates and grids

Terrain (`src/terrain.js`): `N = 256` nodes per axis, `DX = 2.5` m, node
`(i, j)` sits at grid coordinates `(i*DX, j*DX)`; `j = 0` is the offshore
row, `j` grows shoreward. World coordinates are `grid - EXT/2` with
`EXT = (N-1)*DX = 637.5` m (world x alongshore, world z shoreward, world y up).
`beach.z[j*N+i]` is the bed elevation (m) relative to datum 0; the still water
level `wl` (tide) is relative to the same datum; `beach.rock[j*N+i]` in [0,1]
marks rock.

The solver works on cells whose corners are terrain nodes (bilinear bed).
A tier `r` in {1, 2} sets the solver spacing `dxs = DX / r` (2.5 m or 1.25 m).

- Visible cells per axis: `V = (N-1)*r` (255 or 510).
- Margins (cells), fixed in metres: lateral `ML = round(100/dxs)` on both
  alongshore sides, offshore `MO = round(300/dxs)`, inland `MI = 2*r`.
- Solver size: `nx = V + 2*ML` (alongshore, index I), `ny = MO + V + MI`
  (cross-shore, index J, J = 0 offshore).
- Solver cell `(I, J)` centre is at grid coordinates
  `x = (I - ML + 0.5)*dxs`, `y = (J - MO + 0.5)*dxs`; its corner `(I, J)` is
  at `((I - ML)*dxs, (J - MO)*dxs)`.
- Corner bed `Bcorner(I, J)` = bilinear interpolation of `beach.z` at the grid
  position clamped to `[0, (N-1)*DX]` on both axes (so the margins continue the
  edge values: the offshore margin keeps row 0's depth, the sides keep the
  edge columns). Manning `n = 0.02` on sand, `0.035` where the nearest node's
  rock value is >= 0.5.
- Cell bed `B = mean of its 4 corners`; east face bed `Bx = mean of the NE
  and SE corners`; north face bed `By = mean of the NW and NE corners`
  (exactly as `setBed` in `src/boussinesq.js`).

All of this is implemented once in `src/engine/layout.js` (plain JS, no
three.js) with:

```js
export function layout(r)            // -> { r, dxs, V, ML, MO, MI, nx, ny }
export function cornerBed(beach, wl, r, out)   // Float32Array((nx+1)*(ny+1)) of corner elevations
export function cornerManning(beach, r, out)   // Float32Array((nx+1)*(ny+1)) 0.02 / 0.035
export function cellToGrid(lay, I, J)  // -> [x, y] grid coords (m) of the cell centre
export function gridToCell(lay, x, y)  // -> [I, J] (fractional)
```

## 2. Physics (what the GPU port must reproduce)

`src/boussinesq.js` is the specification. In one time step `step(dt)`:

1. SSP-RK2 (Heun): two Euler stages, then `h = (h0 + h2)/2` etc. Each
   stage: reconstruct (`reconstruct`), fluxes (`fluxes`, Kurganov–Petrova
   central-upwind with hydrostatic reconstruction of the face depths,
   mirrored wall states at the non-periodic edges), right-hand side (`rhs`:
   flux divergence, hydrostatic bed sources, explicit dispersive terms `E`
   from the surface and lagged cross terms `M` from `Pt`, `Qt`), Euler update
   of `h` and of the modified momenta `U = L_x P`, `V = L_y Q`, tridiagonal
   solve of `L_x P = U` per row and `L_y Q = V` per column (`solveL`),
   semi-implicit Manning friction, speed cap.
2. Relaxation toward the target field (`applyRelaxation`).
3. Breaking update (`updateBreaking`) and the dispersion mask (`updateChi`).

Constants and switches to copy: `BQ = 1/15`, `theta = 1.3`, `eps = 0.05`
(desingularisation), `hDry = 1e-4`, `dispDepth`, breaking parameters
`{ onsetRatio: 0.8, onsetSlope: 0.65, slopeRes: 0.18, keepSlope: 0.15,
memory: 2.0, ramp: 3 }` (`ramp` is in cells: the driver passes `round(7.5 m / dxs)`, so
both tiers dilate the broken region by the same 7.5 m), the slope clamp (+-0.2) inside the dispersive terms,
`chiBed` (dispersion fades out in very shallow water and on steep beds), the
`Wd` rule (dry cells contribute the still water level to the dispersive
stencils), the clamping of `d` and of `eta/d`. Read the file; where the GPU
code departs from it for a good reason (e.g. PCR instead of Thomas), the
result must still match to float precision.

The tridiagonal systems are diagonally dominant; solve them with parallel
cyclic reduction (`ceil(log2(n))` passes) or an equivalent exact method.
Identity rows wherever `chi = 0` (so broken, dry and very shallow cells
reduce to shallow-water equations).

Time step: `dt = 0.25 * dxs / amax` with `amax` the largest `|u| + sqrt(g h)`
expected; the app passes `dt` in (it estimates `amax` from the bathymetry
and the swell: `sqrt(g*(dmax + H)) + 2`), and the solver exposes `maxSpeed()`
(a GPU reduction plus a synchronous readback) and the pair `maxSpeedIssue()` /
`maxSpeedPoll()` (the same reduction read back behind a fence on a later
frame, so the check never drains the GPU queue) so the driver can halve `dt`
every few seconds if that estimate is exceeded.

## 3. GPU solver API (`src/engine/gpu-solver.js`)

```js
import { createGpuSolver } from './engine/gpu-solver.js';
const sol = createGpuSolver(renderer /* THREE.WebGLRenderer, WebGL2 */, { r });
sol.layout                      // the layout(r) object
sol.setBathymetry(beach, wl)    // (re)upload corner beds + Manning; keeps the water surface where possible
sol.setWaterLevel(wl)           // tide; same as setBathymetry with the new wl
sol.setTargets(res)             // res = result of waves.js solve(): S[8], A[8], omegas, phases, D, tau, setOmega, setDepth, h
sol.hotStart(t)                 // state := target field at time t everywhere, F := 0
sol.step(dt, t)                 // one SSP-RK2 step; t is the simulation time at the START of the step (for the targets)
sol.textures.state              // THREE.Texture nx*ny RGBA32F: (h, P, Q, F)   -- read by the renderer
sol.textures.bathy              // THREE.Texture nx*ny RGBA32F: (B, d, chiBed, n)   -- B cell bed, d = wl - B
sol.readCrests()                // Float32Array (see section 5): the crest-scan pass, read back through a pixel buffer one call behind (asyncCrests: false reads synchronously)
sol.maxSpeed()                  // number (m/s), from a reduction pass + small synchronous readback
sol.maxSpeedIssue()             // true: the reduction is queued and read back behind a fence (false: no reduction on this grid, use maxSpeed())
sol.maxSpeedPoll()              // the issued result once the GPU has finished it (-1 until then: at least the next frame)
sol.readState()                 // Float32Array(nx*ny*4) copy of the state texture (tests only)
sol.loadState(h, P, Q, F)       // Float64Array/Float32Array per field, nx*ny (tests only)
sol.passesPerStep               // number, for the README
sol.dispose()
```

Everything is three.js 0.160 (`RawShaderMaterial`, `glslVersion: THREE.GLSL3`,
`WebGLRenderTarget` with `type: THREE.FloatType`, `texelFetch`, ping-pong
targets; `WebGLMultipleRenderTargets` for multiple outputs). Textures are
sampled with `texelFetch` and explicit clamping; no filtering of state.
The visible window of the state texture, for a renderer sampling at grid
coordinate `(x, y)` (m), is texel `((x/dxs) + ML - 0.5, (y/dxs) + MO - 0.5)`
(bilinear between cell centres).

### Target field (relaxation zones)

The target is the linear refracted field already computed by
`src/waves.js` (`solve()`), which `src/render.js` currently sums on the GPU
(the `waveMat` pass: components `S[m]`, `A[m]` on the 256x256 node grid,
phases `uPh[m] = phases[m] - omegas[m]*t`, saturation `D = res.D` and the set
modulation `setMod = 1 + setDepth*cos(setOmega*(t - tau))`). The relaxation
target at a solver cell is

```
eta_T = D * setMod * sum_m A_m cos(S_m + phases[m] - omegas[m]*t)
P_T   = sum_m eta_m * c_m * kx_m / |k_m|,  Q_T = sum_m eta_m * c_m * ky_m / |k_m|
```

with `eta_m` the m-th term (including `D*setMod`), `k_m = grad S_m` from
central differences of the `S_m` texture on the node grid and
`c_m = omegas[m] / |k_m|`. In the margins (outside the node grid) sample `A`,
`D`, `tau`, `k_m` at the clamped position and extrapolate the phase linearly:
`S_m(x) = S_m(x_c) + k_m(x_c) . (x - x_c)` where `x_c` is the clamped grid
position. The relaxation weight field (`relaxField` in `src/boussinesq.js`,
Mayer et al. weights) covers the whole lateral and offshore margins: `s` runs
from 0 at the inner edge of a margin to 1 at the domain edge, combined by
`max` where margins meet; none inland.

`hotStart(t)` sets `h = max(0, wl + eta_T - B)`, `P = P_T`, `Q = Q_T`
(zero where dry), `F = 0` on every cell, including the visible area.

### Bathymetry updates

`setBathymetry` is called while the user sculpts (up to every frame). It
re-runs the bed pass (corner beds -> B, Bx, By, d, chiBed, n). The water
surface `W = B_old + h` is kept where the cell stays wet:
`h_new = max(0, W_old - B_new)`; momentum is zeroed where the cell becomes dry.

## 4. Breaking and foam

`F` in the state texture is the breaking field of `updateBreaking` (1 at a
fresh breaking front, decaying with a 2 s memory). The renderer uses it as
the whitewater source and to decide where the solver's crest is a bore.

## 5. Crest scan (GPU pass + readback)

A pass with output `V x K` (K = 6 slots per visible column, `I` in
`[ML, ML+V)`), two RGBA32F outputs, written every frame after the last
sub-step and read back by `readCrests()` (through a pixel buffer object and a
fence, so the records describe the previous call's scan; `asyncCrests: false`
reads synchronously) as one
`Float32Array(V*K*8)` laid out `[(col*K + slot)*8 + channel]`, channels:

```
0: J   solver row of the crest (float), or -1 if the slot is empty
1: etaC crest surface elevation above wl (m)
2: hC   total depth at the crest (m)
3: FC   breaking field at the crest, 0..1
4: etaT surface elevation of the trough in front of (shoreward of) the crest (m)
5: Jf   row of the steepest point of the front face (between crest and trough)
6: sF   steepest front slope dEta/dy (positive, m/m)
7: uC   cross-shore velocity Q/h at the crest (m/s)
```

The scan walks each column from the offshore margin edge (J = MO) shoreward
over the visible rows, records local maxima of `eta = h + B - wl` on wet
cells (`h > 0.15`) whose height above the next trough shoreward is at least
0.08 m, in order from offshore to shore, up to K per column.

## 6. Breakers: crest tracker, lip model, mesh, spray (`src/breakers/`)

CPU side, driven by the crest records each frame. Public API:

```js
import { createBreakers } from './breakers/index.js';
const brk = createBreakers({ layout, wl, beach });
brk.update(records, dt, { t, wl, beach, swell })  // records = Float32Array from readCrests()
brk.group        // THREE.Group with the lip/curtain meshes and spray points; add to the scene once
brk.splats       // array of { x, y, r, s } in world x/z (m) and strength 0..1: foam to inject this frame (impact lines, splash-up)
brk.active       // array of tracked breakers: { id, state, points: [{x, y, etaC, hC, c}], H, xi, ... } for the probe/verdict and tests
brk.setWaterLevel(wl)
brk.dispose()
```

Model (what "update" does):

1. Link the crest records across columns into crest lines (same slot, rows
   within +-3 cells of the neighbours, split where a column has no crest),
   and across frames into tracked crests by nearest position (the crest
   advances about `c*dt` shoreward per frame).
2. For each tracked crest sample: wave height `H = etaC - etaT`, depth `hC`,
   phase speed `c` from the tracked motion (fallback `sqrt(g*hC)*(1 + 0.5*etaC/hC)`),
   local bed slope `tanB` along the propagation direction from `beach.z`,
   deep-water length `L0 = g*T^2/(2*pi)` with `T` from the swell, Iribarren
   number `xi = tanB / sqrt(H/L0)`.
3. Breaking type by `xi`: spilling `xi < 0.4` (no lip, the solver bore and
   foam do the work), plunging `0.4 <= xi < 2.0`, collapsing/surging
   `xi >= 2.0` (no lip, a surge up the beach).
4. A plunging crest that turns on (`FC` rises above 0.5 along a stretch of at
   least 4 m) spawns a lip: a curtain of water thrown forward from the crest
   line, ballistic: horizontal speed `1.15*c`, vertical start `etaC`, free
   fall under g, until it meets the solver surface ahead (the trough level
   `etaT`); impact time `tau = sqrt(2*(etaC - etaT)/g)`. The curtain is a
   ribbon mesh (crest samples x 12 profile samples) whose profile at time
   `s` after launch is the family of trajectories launched between `0` and
   `s`, so the tube (barrel) between the curtain and the face is an elliptical
   void with height about `H` and width `H * (0.65 + 0.3*clamp((xi-0.4)/1.6, 0, 1))`
   (Mead & Black 2001: rounder tubes for larger xi). The lip advances with the
   crest; the crest line keeps being tracked while the lip exists.
5. Impact: when the lip reaches the surface, emit `splats` along the impact
   line (strength 1, radius about `H`), start a splash-up (a second, lower
   curtain rising to about `0.8*H` and falling within 0.5 s) and spray
   (points with a short life, launched upward/forward from the lip tip and
   the impact line). After splash-up the breaker is `collapsed`: the lip mesh
   fades over 0.4 s while the solver's bore (`F`) and foam take over.
6. Dispose and recycle geometry; at most 24 live breakers; a crest that
   loses its tracked records for 0.5 s is dropped.

The materials: curtain = translucent water with foam along its leading edge
and underside (a `ShaderMaterial` similar to the water shader's colour
model is fine); spray = `THREE.Points` with soft white sprites, additive, sized
by distance.

Tests: `test/breakers.test.mjs` (Node, no WebGL) must exercise the tracker
and the lip model with a synthetic crest-record generator (a sinusoidal crest
line advancing shoreward whose `FC` switches on at a given row), checking
that a breaker spawns at the right place and time, the lip lands at the
expected distance (`1.15*c*tau`), the tube width follows `xi`, splats appear
at impact, and the breaker is dropped when the records stop. three.js in
Node: `import * as THREE from '/home/claude/three-0.160/build/three.module.js'`
(geometry-only; no renderer).

## 7. Renderer integration (done by the integrator)

`src/render.js` keeps its M x M `waveRT` (eta, dispX, dispY, foamSource) and
foam ping-pong; a new "present" pass fills `waveRT` from the solver state
(eta = h + B - wl where wet, else the bed; foamSource from F) instead of the
linear wave pass when the engine is on. The breakers' group is added to the
scene and their splats go into the foam pass. The linear pipeline stays as
the fallback (no WebGL2 float render targets, or the user switches it off).

## 8. Testing infrastructure

- Static server: `cd /home/claude/wave-simulator && npx http-server -p 8099 -s -c-1 .`
  (check with `curl -s -o /dev/null -w '%{http_code}' http://localhost:8099/index.html`;
  start it with `nohup ... &` if it is not running).
- Headless browser: `test/headless.mjs` (`openPage(url)`, `result(page)`),
  SwiftShader WebGL2 with float render targets; three.js is served from
  `/home/claude/three-0.160` in place of the CDN. Test pages live in
  `test/pages/*.html` and use the same import map as `index.html`.
- GPU equivalence: `test/gpu-equivalence.mjs` runs the same case on
  `src/boussinesq.js` (Node) and on the GPU (page) by loading the same
  initial state into both, stepping N times with the same dt, and comparing
  `h`, `P`, `Q`, `F` (max abs differences; tolerance 2e-3 m on h and 2e-2 on
  P, Q for a case with ~1 m waves, F exact up to the 3-cell ramp). Cases: the
  solitary wave over a flat bed (dispersion), the plane beach with a
  relaxation target (breaking, run-up, relaxation), still water with an island
  (wet/dry balance: everything must stay at 1e-5).
