# Shorebreak Lab: wave dynamics specification (v1, final)

Lead-engineer specification for replacing the linear phase-resolved wave model
with real-time depth-integrated nonlinear dynamics plus a kinematic overturning
(lip/barrel) model. It merges the three designs ("numerics", "visuals",
"shipping") and applies both judgments. Where this document and
`docs/engine-contract.md` differ, this document wins; the contract is to be
updated to match (the differences are listed in section 0.3).

Everything an engineer needs is in the text: equations in the exact form to be
coded, discretisation, grid, texture layouts, boundaries, breaking, wet/dry,
friction, GPU pass list, the JavaScript reference solver and its API, the
crest/lip/barrel model, foam and spray, the render data path, module interfaces
and the validation battery with expected numbers.

Repository: `/home/claude/wave-simulator`. The reference (oracle) solver already
exists at `src/boussinesq.js` with `test/boussinesq.test.mjs` passing in Node
(lake at rest, volume, Madsen-Sorensen dispersion, solitary wave, plane beach,
Ritter dam break). The GPU port must reproduce it; the lip model and renderer
consume its outputs.

---

## 0. Decisions

### 0.1 Architecture in one paragraph

Madsen & Sorensen (1992) extended Boussinesq equations (B = 1/15) in the
conservative (h, P, Q) form and the "modified momentum" grouping of Celeris
(Tavakkol & Lynett 2017), solved on a uniform square grid whose cell corners are
the terrain nodes: Kurganov & Petrova (2007) central-upwind fluxes on a
generalised-minmod MUSCL reconstruction with the hydrostatic reconstruction of
Audusse et al. (2004) at the faces (well-balanced, positivity preserving,
wet/dry), explicit central differences for the third-derivative and lagged
cross terms, a tridiagonal solve per row/column for the implicit part
(Thomas on the CPU, parallel cyclic reduction on the GPU), SSP-RK2 with
dt = 0.25 dx / a_max. Breaking is the hybrid switch of Tonelli & Petti (2009) /
Kazolea et al. (2014): dispersion coefficients go to zero inside the operator
where a wave is flagged, so the shock-capturing scheme carries bores whose jump
conditions dissipate the energy. Swell enters and leaves through relaxation
zones (Mayer et al. 1998; Jacobsen et al. 2012) whose target is the existing
refracted linear field of `src/waves.js` on the extended grid. A depth-
integrated surface cannot overturn, so a GPU crest-scan pass feeds a CPU
tracker and a kinematic lip model (ground-frame ballistic jet calibrated to the
Smith & Kraus 1991 plunge distance, elliptical tube with the Mead & Black 2001
vortex ratio), rendered as a dynamic sheet mesh with a lip-mask lowering and a
concave-face push on the water surface, splash-up, spray and foam injection,
handing over to the solver's bore.

### 0.2 Resolution of every disagreement

| Topic | Decision | Reason (one line) |
|---|---|---|
| Bed layout | Corner-node: solver cells sit between terrain nodes; B_c = mean of 4 corners, face bed = mean of 2 corners (numerics, oracle). | Node-centred beds break the lake-at-rest proof at partially wet cells (both judges). |
| Face reconstruction | Hydrostatic reconstruction (Audusse 2004) as implemented in the oracle, not KP's 2h_bar fix. | Already implemented and exact to 1e-15 for fully wet lakes; identical flux formula otherwise. |
| Breaking inside the operator | chi enters the tridiagonal coefficients as identity rows (numerics/visuals); no post-hoc blend of P_tridiag and P_nswe (shipping). | The blend satisfies neither equation at the flag edge (both judges). |
| PCR coefficients | Rebuilt every stage inside the PCR passes; no precomputed per-level textures. | chi changes every step; precompute is dead weight (both judges). |
| PCR layout | x and y systems fused in one MRT pass per level. | Halves the pass count (graphics judge). |
| Time integrator | SSP-RK2 (Heun), nu = 0.25; SSP-RK3 is a compile-time switch if validation test V5 (20-period amplitude) fails. | KP 2D positivity bound applies per Euler stage; RK2 is 1.5x cheaper than RK3 (both judges). |
| Time step | Step at the CFL dt (about 0.042 s at 2.5 m), render-interpolate between states n and n+1; never "one step per frame". | Stepping at 1/60 s wastes 2x the budget (graphics judge). |
| Default resolution | 2.5 m (r = 1) default; 1.25 m (r = 2) auto-selected on GPUs that step it in < 9 ms; 5 m phone tier; legacy linear pipeline when float render targets are missing. | 1.25 m resolves the front but costs 8x; tier by measurement (physics judge). |
| Margins | Offshore 300 m, lateral 150 m, inland 2 cells, flat clamp-to-edge bathymetry. | >= 1 wavelength of a 20 s swell offshore; 100/50 m zones are too short; sloping extrapolation adds a shoaling mismatch (both judges). |
| Lateral relaxation | Active only where still-water depth d > 1 m; target = linear field + the 30 s running mean of the computed (eta, P, Q) at the zone's inner edge of the same row. | Keeps the linear target out of the swash and lets longshore currents and set-up survive the margins (both judges). |
| Target flux | Per component (omega_m/k_m) eta_m k_hat_m, saturation ratio D and the set envelope. | Exact linear flux; the dominant-direction shortcut is 3% off (graphics judge). |
| Breaking onset | Any of three criteria (section 7): eta_t >= gamma_b(h, dx) sqrt(gh), front slope >= s_b(dx), eta/d >= 0.8; calibrated by the Weggel grid and Ting & Kirby. | eta_t alone fires 40 m late at 2.5 m (both judges, prototype evidence). |
| Breaking persistence | F memory 2.0 s with sustain at 0.15 sqrt(gh), advected with the front; dilation ramp 7.5 m. | Covers T* = 5 sqrt(h/g) = 2.3 s at h = 2 m; gives the lip a stable clock. |
| chi hold while the lip flies | Not in v1: chi switches on onset. The lip mesh holds its own crest height H_b. Knob `chiHoldFactor` (default 0) kept for experiments. | Keeping explicit third derivatives alive on an under-resolved vertical front is the known instability (risk R2). |
| Lip speed frame | Ground frame; tip speed V clamped to [1.1, 1.6] c; plunge distance X_p = H_b (3.0 + 0.8 lambda) is the master calibration. | Relative-frame speeds either gave 0.5 H_b cavities or 2.2 c tips (both judges). |
| Lip stage durations | In units of tau_imp and sqrt(H_b/g), never T_p. | 0.1 T_p pitching gives 6.4 H_b plunge distances (graphics judge). |
| Tube ratio | Length/height r_v = 0.065 x + 0.82 with x = 1/tan(beta) (bed gradient 1:x), clamped to [1.3, 3.0]. | Steeper beds must give rounder tubes; the degree form is inverted (both judges); verify against Mead & Black 2001. |
| Surface under the barrel | Both: lip-mask lowering pass (shipping) and the crest-strip concave-face push (visuals), dead-reckoned by c x readback latency. | Otherwise the bore pokes through the roof (both judges). |
| Lip impact feedback | Impact splats force F = 1 over a 2-cell disc in the solver. | The bore must exist where the lip lands (shipping, both judges). |
| Readback | Crest-scan pass (tens of KB) through a PBO fence ring, with explicit `gl.bindBuffer(PIXEL_PACK_BUFFER, null)` afterwards; STATS at 2 Hz; no full-field diag readback. | Full-field readback at 30 Hz is 35-70 MB/s (both judges); `resetState()` does not unbind the PBO. |
| Precision | RGBA32F everywhere in the solver; no half-float state in any tier. | Half-float P at 45 m^2/s drifts (graphics judge). |
| Manning | n = 0.02 sand, 0.035 rock. | 0.015 is low, 0.025 over-damps run-up (both judges). |
| Dry threshold | h_dry = 1e-4 m (oracle); desingularisation eps = 0.05 m. | 5 cm truncates the swash tongue (physics judge). |
| Undertow | Not claimed anywhere in the UI. | Depth-integrated models have no undertow (physics judge). |
| Forward Euler tier | Rejected; the phone tier is RK2 at 5 m. | Euler plus centred dispersive terms is linearly unstable (graphics judge). |

### 0.3 Changes to `docs/engine-contract.md`

1. Lateral margin `ML = round(150/dxs)` (was 100 m).
2. Phone tier `r = 0.5` added (section 1.3).
3. Crest record channel 5 is `Jt` (row of the trough ahead), not `Jf`.
4. `readCrests()` is asynchronous (PBO fence ring); the synchronous path is the fallback.
5. `sol.textures.stats` (section 9.3) and `sol.stampImpacts(list)` (section 11.8) added.
6. `createBreakers(...).update()` returns a `CrestStrip` record for the renderer (section 13.3).
7. `relax` weights: the lateral zone is masked to zero where `d <= 1 m`.

---

## 1. Coordinates, grid, tiers

### 1.1 Terrain (unchanged)

`src/terrain.js`: `N = 256` nodes per axis, `DX = 2.5` m, `EXT = (N-1) DX = 637.5` m.
Node `(i, j)` is at grid coordinates `(i DX, j DX)`; `j = 0` is the offshore
row; `j` grows shoreward. World: `x_world = x_grid - EXT/2` (alongshore),
`z_world = y_grid - EXT/2` (shoreward), `y_world` up. `beach.z[j N + i]` is the
bed elevation (m) relative to datum 0, `beach.rock[j N + i]` in [0, 1],
`beach.waterLevel = wl` (tide) on the same datum.

### 1.2 Solver grid

Square cells of spacing `dxs = DX / r`. Cell corners coincide with terrain
nodes (for r = 2 every other corner is a node, the others are bilinear
interpolants; for r = 0.5 every node is a corner of every other cell). Cell
index `(I, J)`, `I` alongshore in `[0, nx)`, `J` cross-shore in `[0, ny)`,
`J = 0` offshore. Row-major texel/array index `c = J nx + I`.

```
V  = round((N-1) r)            visible cells per axis (255, 510, 128 for r = 1, 2, 0.5)
ML = round(150 / dxs)          lateral margin cells, both sides   (60, 120, 30)
MO = round(300 / dxs)          offshore margin cells               (120, 240, 60)
MI = max(1, 2 r)               inland cells                        (2, 4, 1)
nx = V + 2 ML                  (375, 750, 188)
ny = MO + V + MI               (377, 754, 189)
cell centre (I, J) at grid coords  x = (I - ML + 0.5) dxs,  y = (J - MO + 0.5) dxs
corner (I, J)        at grid coords  x = (I - ML) dxs,        y = (J - MO) dxs
```

For r = 0.5 the visible box is 128 cells of 4.98046875 m (`dxs = EXT / 128`),
corners are bilinear samples of the heightfield; nothing else changes.

Corner bed `Bcorner(I, J)` = bilinear interpolation of `beach.z` at the corner's
grid position **clamped** to `[0, EXT]` on both axes (margins continue the edge
values: flat offshore extension of row 0, flat lateral extension of the edge
columns). Corner Manning `n = 0.02`, or `0.035` where the bilinear rock value
at the corner is >= 0.5. Cell values:

```
B(I,J)  = 0.25 (c00 + c10 + c01 + c11)       c00 = Bcorner(I,J), c10 = Bcorner(I+1,J), c01 = Bcorner(I,J+1), c11 = Bcorner(I+1,J+1)
Bx(I,J) = 0.5 (c10 + c11)                     bed at the EAST face (between I and I+1)
By(I,J) = 0.5 (c01 + c11)                     bed at the NORTH face (between J and J+1)
west face bed of (I,J) = Bx(I-1,J); for I = 0: 0.5 (c00 + c01). south face bed = By(I,J-1); for J = 0: 0.5 (c00 + c10)
d = wl - B  (still-water depth, negative on land)
```

So `B = (Bx_east + Bx_west)/2 = (By_north + By_south)/2` exactly, which the
well-balancing proof needs.

### 1.3 Tiers

| Tier | r | dxs | nx x ny | PCR levels L | dt (a_max = 15 m/s) | Hardware |
|---|---|---|---|---|---|---|
| H (high) | 2 | 1.25 m | 750 x 754 | 10 | 0.021 s | discrete GPU (step < 9 ms measured) |
| S (standard) | 1 | 2.5 m | 375 x 377 | 9 | 0.042 s | laptop iGPU, default |
| P (phone) | 0.5 | 4.98 m | 188 x 189 | 8 | 0.083 s | phones; spray off, K = 4 crest slots |
| C (CPU) | - | 4.98 m | 188 x 189 | - | 0.083 s | float render targets present but step time > 40 ms at tier P: `src/boussinesq.js` in the worker, dispersion off |
| L (legacy) | - | - | - | - | - | no `EXT_color_buffer_float`: the existing linear pipeline, unchanged |

Tier selection: start at S (P on `navigator.userAgentData.mobile` or
`maxTouchPoints > 0 && innerWidth < 900`). Measure the mean GPU step time over
the first 2 s (Date.now() over batches of 24 steps with one `readPixels` sync
at the end of each batch; `performance.now()` is clamped in headless). If
S steps in < 3 ms, try H for 2 s; keep H if its step is < 9 ms. If the current
tier needs more than 4 substeps per frame on 3 consecutive frames, drop one
tier and show "simulation at 0.5x" in the status element. Tiers never change
while painting.

### 1.4 Mapping between the renderer and the solver

A renderer sample at grid coordinates `(x, y)` (m) reads solver texel
coordinates `tx = x / dxs + ML - 0.5`, `ty = y / dxs + MO - 0.5` (cell
centres at integer texel coordinates), normalised `uv = (tx + 0.5, ty + 0.5) /
(nx, ny)`. The present pass (section 13.1) uses bicubic Catmull-Rom on W; all
solver passes use `texelFetch` with explicit clamping.

---

## 2. Governing equations (exact coded form)

Symbols per cell: `h` total depth (m, >= 0), `W = B + h` water surface (datum),
`eta = W - wl`, `P = h u`, `Q = h v` (m^2/s), `g = 9.81`, `Bq = 1/15`,
`d = wl - B` still-water depth, `d+ = max(d, 0)`, `d_x`, `d_y` its slopes
clamped to [-0.2, 0.2], `chi` in [0, 1] the dispersion factor (section 7.4),
`n` Manning. Friction `tau = (tau_x, tau_y)`.

Continuity and momentum (Madsen & Sorensen 1992 with mild-slope terms, as
quoted by Kazolea, Delis, Nikolos & Synolakis 2012 and Tavakkol & Lynett 2017,
with `g h eta_x` split as `(g h^2/2)_x + g h B_x` for well-balancing):

```
(1)  h_t + P_x + Q_y = 0
(2)  (L_x P)_t + (P^2/h + g h^2/2)_x + (P Q/h)_y = -g h B_x - tau_x + E_x + M_x
(3)  (L_y Q)_t + (P Q/h)_x + (Q^2/h + g h^2/2)_y = -g h B_y - tau_y + E_y + M_y
```

Modified momenta (the implicit part; linear operators per row / per column):

```
L_x P = P - chi [ (Bq + 1/3) d^2 P_xx + (1/3) d d_x P_x ]
L_y Q = Q - chi [ (Bq + 1/3) d^2 Q_yy + (1/3) d d_y Q_y ]
```

Explicit dispersive surface terms (every term written out; `W` means the
dispersive surface `Wd` of section 3.4, so these are derivatives of `eta`):

```
E_x = chi [ Bq g d^3 W_xxx + Bq g d^3 W_xyy + 2 Bq g d^2 d_x W_xx + Bq g d^2 d_x W_yy + Bq g d^2 d_y W_xy ]
E_y = chi [ Bq g d^3 W_yyy + Bq g d^3 W_xxy + 2 Bq g d^2 d_y W_yy + Bq g d^2 d_y W_xx + Bq g d^2 d_x W_xy ]
```

Lagged cross terms with the other flux's time derivative from the previous
full step (`Pt = (P^n - P^(n-1)) / dt_prev`, `Qt` likewise; zero on the first
step; first order in time, as in Celeris):

```
M_x = chi [ (Bq + 1/3) d^2 (Qt)_xy + (1/6) d d_x (Qt)_y + (1/6) d d_y (Qt)_x ]
M_y = chi [ (Bq + 1/3) d^2 (Pt)_xy + (1/6) d d_y (Pt)_x + (1/6) d d_x (Pt)_y ]
```

Friction (Manning), semi-implicit (section 8):

```
tau_x = g n^2 P sqrt(u^2 + v^2) / h^(4/3),   tau_y likewise,   u = P/h, v = Q/h
```

Derivation check: MS92 eq. (24)-(25) give `P_t + ... + g h eta_x + psi_1 = 0`
with `psi_1 = -(Bq+1/3) d^2 (P_xxt + Q_xyt) - Bq g d^3 (eta_xxx + eta_xyy) -
d d_x [ (1/3) P_xt + (1/6) Q_yt + 2 Bq g d eta_xx + Bq g d eta_yy ] - d d_y
[ (1/6) Q_xt + Bq g d eta_xy ]`. Moving the `P_t` terms to the left gives
`L_x P`; the remaining `-psi_1` terms are `E_x + M_x` above. All three designs
transcribed the same coefficients (1/3, 1/6, 2Bq, Bq); **Engineer B verifies
them once against Kazolea et al. 2012 eq. (3)-(4) before the GPU port is
accepted** and records the check in `test/boussinesq.test.mjs` as a comment.
The oracle (`src/boussinesq.js`, function `rhs`) already codes exactly these.

Linear dispersion (flat bed): `omega^2 = g d k^2 (1 + Bq k^2 d^2) / (1 + (Bq +
1/3) k^2 d^2)`; phase speed within 2.5% of Airy for `kd <= 3` and bounded by
`sqrt(g d)`, so the shallow-water CFL governs.

---

## 3. Spatial discretisation

Uniform square cells, finite volumes on `(h, P, Q)`; everything below is per
cell `c = (I, J)` with neighbours `cE = (I+1, J)`, `cW`, `cN = (J+1)`, `cS`.
Index clamping at the domain edge (non-periodic): out-of-range reads return
the edge cell; the edge faces are walls (section 6.4). The oracle's optional
periodic mode exists for tests only.

### 3.1 Reconstruction (generalised minmod MUSCL, theta = 1.3)

For each of `phi in {W, P, Q}` and each direction, the limited slope

```
s = minmod3( theta (phi_c - phi_W), 0.5 (phi_E - phi_W), theta (phi_E - phi_c) )
phi_Eface = phi_c + s/2,   phi_Wface = phi_c - s/2     (same in y with N/S)
minmod3(a,b,c) = min(a,b,c) if all > 0; max(a,b,c) if all < 0; else 0
```

Dry cells (`h_c <= h_dry = 1e-4`): all four face surfaces are set to `B_c`,
face depths and velocities to zero, and the cell's momentum right-hand side is
zero. (A fourth-order MUSCL-TVD option `order = 4` exists in the oracle; the
GPU implements order 2 only in v1.)

### 3.2 Face depths and hydrostatic reconstruction (Audusse et al. 2004)

Per cell side, with the continuous face beds of section 1.2:

```
hE = max(0, W_Eface - Bx_c),  hW = max(0, W_Wface - Bx_(I-1)),  hN, hS likewise with By
u_side = vel(h_side, P_side),  v_side = vel(h_side, Q_side)
vel(h, p) = 0 if h <= h_dry else sqrt(2) h p / sqrt(h^4 + max(h^4, eps^4)),  eps = 0.05 m   (Kurganov & Petrova 2007)
```

At the face between `c` (east side) and `cE` (west side):

```
z_L = W_Eface(c) - hE(c),  z_R = W_Wface(cE) - hW(cE)           (the bed each side "sees")
B_f = max(z_L, z_R)
h*_L = max(0, W_Eface(c) - B_f),  h*_R = max(0, W_Wface(cE) - B_f)
```

The flux uses `(h*_L, u_E(c), v_E(c))` and `(h*_R, u_W(cE), v_W(cE))`. A dry
face (both h* = 0) gives zero flux. Store `hsE(c) = h*_L` and `hsW(cE) = h*_R`
(and `hsN`, `hsS` for y faces).

### 3.3 Central-upwind flux (Kurganov & Petrova 2007)

For an x face with left state `(hl, ul, vl)` and right `(hr, ur, vr)`:

```
a+ = max(ul + sqrt(g hl), ur + sqrt(g hr), 0),   a- = min(ul - sqrt(g hl), ur - sqrt(g hr), 0)
if a+ - a- < 1e-9: flux = 0
ml = hl ul, mr = hr ur;  k = a+ a- / (a+ - a-)
F_h = (a+ ml - a- mr)/(a+ - a-) + k (hr - hl)
F_P = (a+ (ml ul + g hl^2/2) - a- (mr ur + g hr^2/2))/(a+ - a-) + k (hr ur - hl ul)
F_Q = (a+ (ml vl) - a- (mr vr))/(a+ - a-) + k (hr vr - hl vl)
```

y faces: the normal velocity is `v`, the pressure term `g h^2/2` goes into
`G_Q`, and `G_P = (a+ (ml ul) - a- (mr ur))/(a+ - a-) + k (hr ur - hl ul)` with
`ml = hl vl`. Walls (section 6.4): the mirrored state has the normal velocity
negated and everything else equal.

### 3.4 Right-hand sides

```
R_h = -(F_h(E) - F_h(W))/dx - (G_h(N) - G_h(S))/dx
R_P = -(F_P(E) - F_P(W))/dx - (G_P(N) - G_P(S))/dx
      - (g/2) [ (hE^2 - hsE^2) - (hW^2 - hsW^2) ] / dx                       (face corrections)
      - g (hE + hW)/2 [ (W_Eface - hE) - (W_Wface - hW) ] / dx                 (in-cell bed slope)
      + chi (E_x + M_x)
R_Q = the mirror with N/S, hN, hS, hsN, hsS, and chi (E_y + M_y)
```

The two bed terms together with the fluxes are exactly balanced for still water
over any bed, wet, dry or partially wet.

Dispersive surface `Wd`: `Wd = W` where `h > h_dry`, else `Wd = wl` (dry cells
contribute the still-water level). Stencils (5x5), `idx = 1/dx`:

```
W_xx  = (Wd[I+1] - 2 Wd[I] + Wd[I-1]) idx^2
W_yy  = (Wd[J+1] - 2 Wd[J] + Wd[J-1]) idx^2
W_xy  = (Wd[I+1,J+1] - Wd[I-1,J+1] - Wd[I+1,J-1] + Wd[I-1,J-1]) idx^2 / 4
W_xxx = (Wd[I+2] - 2 Wd[I+1] + 2 Wd[I-1] - Wd[I-2]) idx^3 / 2
W_yyy = (Wd[J+2] - 2 Wd[J+1] + 2 Wd[J-1] - Wd[J-2]) idx^3 / 2
W_xyy = (W_yy[I+1] - W_yy[I-1]) idx / 2         (W_yy evaluated at the east and west neighbours)
W_xxy = (W_xx[J+1] - W_xx[J-1]) idx / 2
(Qt)_xy, (Pt)_xy as W_xy;  (Qt)_x, (Qt)_y, (Pt)_x, (Pt)_y central 2-point / (2 dx)
```

The dispersive terms are evaluated only where `chi > 0` and `2 <= I <= nx-3`,
`2 <= J <= ny-3` (`chi = 0` on the two outermost rings anyway).

### 3.5 Operators L_x, L_y and the tridiagonal systems

With `al = chi (Bq + 1/3) d+^2 / dx^2`, `bx = chi d+ d_x / (6 dx)`,
`by = chi d+ d_y / (6 dx)`:

```
(L_x P)_c = (1 + 2 al) P_c - (al - bx) P_W - (al + bx) P_E
(L_y Q)_c = (1 + 2 al) Q_c - (al - by) Q_S - (al + by) Q_N
Row system per J:   a_I P_(I-1) + b_I P_I + c_I P_(I+1) = U_I,  a = -(al - bx), b = 1 + 2 al, c = -(al + bx)
Column system per I: a_J Q_(J-1) + b_J Q_J + c_J Q_(J+1) = V_J,  a = -(al - by), b = 1 + 2 al, c = -(al + by)
Identity row (a = c = 0, b = 1) wherever chi = 0 (dry, land, broken, edge rings).
```

Diagonal dominance `|a| + |c| = 2 al <= b` always, strictly when `|bx| <= al`,
i.e. `|d_x| <= 2.4 d / dx`; guaranteed by the slope clamp (|d_x| <= 0.2) together
with `chi = 0` below `d = 0.25 m` (so `2.4 d / dx >= 0.24` at dx = 2.5 m and
`0.48` at 1.25 m). No pivoting.

CPU: Thomas algorithm (`solveL` in the oracle). GPU: parallel cyclic reduction,
`L = ceil(log2(max(nx, ny)))` passes, each eliminating neighbours at distance
`s = 2^(l-1)`:

```
level l, stride s:  for each row element i
   alpha = a_i / b_(i-s),  beta = c_i / b_(i+s)           (elements outside [0, n) are identity rows: a = c = 0, b = 1, r = 0)
   a' = -alpha a_(i-s),  c' = -beta c_(i+s)
   b' = b_i - alpha c_(i-s) - beta a_(i+s)
   r' = r_i - alpha r_(i-s) - beta r_(i+s)
after level L: P_i = r_i / b_i
```

Coefficients `(a, b, c)` are computed inside the level-1 pass from the bathy
and chi textures (for `i`, `i-s`, `i+s`), never stored between steps. Float32
is adequate: the system is diagonally dominant with `b / |a| >= 1 + 1/al` and
the oracle comparison (test V13) bounds the round-off.

---

## 4. Time stepping

### 4.1 SSP-RK2 (Heun), per step of dt

```
Pt = (P^n - P^(n-1))/dt_prev, Qt likewise (zero if dt_prev = 0)    [read on the fly from the state ring]
Stage 1:  R(h^n, P^n, Q^n);  U = L_x P^n + dt R_P;  V = L_y Q^n + dt R_Q;  h1 = max(0, h^n + dt R_h)
          solve L_x P1 = U (rows), L_y Q1 = V (columns);  friction + cap (section 8) on (h1, P1, Q1)
Stage 2:  R(h1, P1, Q1);  U = L_x P1 + dt R_P; ...;  h2, P2, Q2 as above
Average:  h^(n+1) = (h^n + h2)/2,  P^(n+1) = (P^n + P2)/2,  Q^(n+1) = (Q^n + Q2)/2;  P = Q = 0 where h <= h_dry
Then:     relaxation (section 6), breaking update and chi (section 7), t += dt
```

`L_x` is applied with the **current step's chi** (frozen for both stages), so
averaging the primitives equals averaging the modified momenta. The state is
stored as `(h, P, Q, F)`, never `U`, so bathymetry and tide can change between
steps. SSP-RK3 (Shu & Osher 1988: `U1 = U^n + dt L(U^n)`, `U2 = 3/4 U^n + 1/4
(U1 + dt L(U1))`, `U^(n+1) = 1/3 U^n + 2/3 (U2 + dt L(U2))`, convex combinations
formed in P space because chi is frozen) is implemented in the oracle behind
`options.rk = 3` and in the GPU pass scheduler as a third stage, enabled only if
test V5 fails with RK2.

### 4.2 Time step and CFL

```
a_max = sqrt(g (d_max + H_swell)) + 2        from the bathymetry (max still depth of wet cells) and the swell height
dt = nu dxs / a_max,  nu = 0.25              (KP07 2D positivity bound per Euler stage)
```

`sol.maxSpeed()` (a max-reduction pass + 4-byte readback, at most once per
second) returns the actual `max(|u| + sqrt(g h))`; if it exceeds `a_max`, the
app halves `dt` for the next second and recomputes. `nu` may be raised to 0.4
only after validation tests V1, V2, V10 pass at 0.4 in the oracle **and** on
the GPU.

### 4.3 Scheduler and render interpolation

Per frame the app accumulates `t_target += dt_frame x speed` (dt_frame clamped
to 0.05 s). It runs `n = floor((t_target - t_sim) / dt)` steps, capped at 4; if
capped on 3 consecutive frames the clock slows (`speed_eff = speed x 0.5`,
shown in the UI) until the cap stops binding for 2 s. The renderer receives
states `S^n`, `S^(n+1)` and `alpha = (t_target - t_n) / dt` and displays
`lerp(S^n, S^(n+1), alpha)` for h (W is derived), which removes 24 Hz stutter.

---

## 5. Bathymetry and tide updates

`setBathymetry(beach, wl)` runs the bed pass (section 9.2) on every call (while
painting, up to once per frame; the app throttles the heightfield upload to
every frame and the linear solve to 350 ms as now):

```
B_new, Bx, By, d, chiBed, n  from the corner bed texture
W_old = B_old + h_old   (wet cells);  h_new = max(0, W_old - B_new);  P, Q unchanged where h_new > h_dry, else 0
```

(The surface is kept, so sand rising through the water makes a dry patch
without a splash; mass is not conserved across edits, accepted.)

`setWaterLevel(wl)`: `d` is recomputed; wet cells keep their surface `W`
(the relaxation zones bring the new level in over ~2 T_p); dry cells with
`wl - B > 0` are flooded to `h = wl - B` with `P = Q = 0`. The relaxation
target uses the new `wl` immediately.

Both operations also rebuild `chiBed` and the `relax` mask (the lateral mask
depends on `d > 1 m`) and leave `F` untouched.

---

## 6. Generation and absorption

### 6.1 Relaxation zones (Mayer, Garapon & Sorensen 1998; Jacobsen et al. 2012)

After each full step, in every cell with weight `s > 0`:

```
w(s) = (exp(s^3.5) - 1) / (e - 1)                       0 at the zone's inner edge, 1 at the domain edge
s(I,J) = max( s_off, s_lat ),
  s_off = (MO - J) / MO                   for J < MO
  s_lat = (ML - I) / ML  or  (I - (nx - ML) + 1) / ML   for the two sides, multiplied by m_lat(J)
  m_lat(J) = 1 where the still depth at the first interior column of row J, d(ML, J) > 1.0 m, else 0 (smoothstep over 0.8..1.2 m)
h   <- (1 - w) h + w h_T,  h_T = max(0, wl + eta_T - B)
P,Q <- (1 - w) (P,Q) + w (P_T, Q_T)   if h_T > h_dry, else (1 - w) (P,Q)
P = Q = 0 where h <= h_dry after the blend
```

The outermost row/column has `w = 1` (Dirichlet to the target). Zones are 300 m
offshore (>= 1 wavelength of a 20 s swell in 16 m, L = 230 m) and 150 m lateral.

### 6.2 Target field

From the existing `src/waves.js` `solve()` run **on the extended grid** in the
worker (`N -> nx, ny` and `dx -> dxs` are already parameters; the bathymetry
passed is the extended cell-centre `d` of section 1.2 as an elevation
`z = wl - d`; the worker call becomes `solve({ nx, ny, dx: dxs, z, wl, swell })`
and `waves.js` accepts rectangular grids, section 14.2). Per component `m`
(8 components; `omega_m`, `phase_m`, amplitude field `A_m`, phase field `S_m`):

```
k_m   = grad S_m   (central differences on the extended grid; |k_m| floored at omega_m^2 / (2 g))
setMod = 1 + setDepth cos(setOmega (t - tau))
eta_m  = D setMod A_m cos(S_m + phase_m - omega_m t)
eta_T  = sum_m eta_m + mean_eta(row)
(P_T, Q_T) = sum_m (omega_m / |k_m|) eta_m k_m / |k_m|  +  (mean_P(row), mean_Q(row))
```

`D` is the dominant train's saturation ratio (no unbroken wave is injected in
the margins' surf zone), `tau` the group travel time. The row means are the
30 s exponential running means of the computed `(eta, P, Q)` **at the first
interior column of the same row** (column `ML` for the left zone, `ML + V - 1`
for the right), read from the STATS texture (section 9.3) with a clamped
column fetch, multiplied by `m_lat(J)`, and zero in the offshore zone. This
lets set-up and longshore currents pass through the lateral zones without the
zone's own state feeding back.

Target ramp: on start and on every swell change the target amplitude is
multiplied by `smoothstep(0, 2 T_p, t - t_change)` while the old field is
cross-faded out over the same window (two sets of component textures during
the fade).

### 6.3 Hot start

`hotStart(t)`: `h = max(0, wl + eta_T - B)`, `P = P_T`, `Q = Q_T` (zero where
dry), `F = 0` in every cell including the visible box, with `D` and the
saturation applied, so waves exist on the first frame; the nonlinear
adjustment takes ~1 T_p and is visible only as a slightly softer first set.

### 6.4 Walls and ghost cells

All four domain edges are walls: the flux at an edge face uses the mirrored
state (normal velocity negated). The two outermost rings have `chiBed = 0`
(identity rows), and dispersive stencils never read beyond them because they
are not evaluated there. Inland the wall is 2 cells past the last terrain row
(dunes at +4.5 m keep it dry on every preset; a flooded flat beach meets a
wall 5 m behind the last node, accepted). Lateral walls 150 m outside the box
reflect only swash and the longshore current that `m_lat` leaves in; friction
and the zones damp it.

Known limits (accepted): no second-order bound long waves in the target
(spurious free harmonics for steep swell in 15-20 m, barely visible);
infragravity reflection at the zones; small mismatch ripples where nonlinear
waves meet the linear target at the lateral inner edge, hidden in the margins.

---

## 7. Breaking

### 7.1 Principle

Hybrid approach (Tonelli & Petti 2009; Shi et al. 2012; Kazolea, Delis &
Synolakis 2014): where a wave is flagged as breaking the dispersion factor
`chi` is zero, the equations are the nonlinear shallow-water equations, and the
central-upwind scheme carries the front as a shock whose jump conditions
dissipate energy at the rate of a hydraulic jump of the same height. No eddy
viscosity in v1 (Kennedy et al. 2000 is the documented fallback, risk R2).

### 7.2 Field F and its update (once per step, after the RK average)

`F` in [0, 1] lives in the state alpha channel (1 = fresh front, decaying).
`eta_t = (h^(n+1) - h^n) / dt` is free from the two states. Per cell:

```
F_adv = F sampled (bilinear) at the foot of the characteristic x - c_f dt,  c_f = sqrt(g h) n_prop
        n_prop = normalised (-grad Wd) if |grad Wd| > 0.02, else (0, 1) (shoreward); F_adv = F where h < 0.1
F = F_adv exp(-dt / tau_F),  tau_F = 2.0 s
onset  (F = 1):      h > 0.1 m  and  relax s < 0.5  and  eta_t > 0  and  ( C1 or C2 or C3 )
sustain (F = max(F, 0.8)):  eta_t >= 0.15 sqrt(g h)  and  F > 0.05                (Kennedy et al. 2000 termination level)
F = 0 where h <= 0.1 m
F = max(F, stamp)   where `stamp` is the lip-impact stamp texture (section 11.8)
```

Onset criteria (any one):

```
C1  eta / max(d, 0.3) >= 0.8                                   (Tonelli & Petti 2009 ratio)
C2  eta_t >= gamma_b sqrt(g h),  gamma_b = clamp(0.18 h / dxs, 0.25, 0.65)   (Kennedy front steepness, capped by what the grid can hold)
C3  |grad Wd| >= s_b(dxs)  and  eta / max(d, 0.3) >= 0.4      (Kazolea 2014 front-slope criterion)
    s_b = 0.577 (tan 30 deg) for dxs <= 1.25 m,  0.40 for 2.5 m,  0.30 for 5 m
```

`gamma_b`'s cap: a grid of spacing dx cannot steepen a front beyond ~3 cells,
so a front of height ~0.8 h reaches at most `eta_t ~ 0.27 (h/dx) sqrt(g h)`;
the threshold is capped at 2/3 of that so the flag fires when the front is as
steep as the grid can hold (where the limiter would otherwise start
dissipating). The prototype's plane-beach run showed H falling from 1.66 m to
1.07 m before the uncapped flag fired; **thresholds `0.8, 0.18, 0.25, 0.65, s_b`
are calibration knobs in `src/engine/params.js` and are set by validation
tests V8 (Weggel) and V9 (Ting & Kirby) before any lip tuning.**

### 7.3 Dilation and chi

```
Fmax = max of F over the square of half-width R_c = round(7.5 / dxs) cells (3 at 2.5 m, 6 at 1.25 m, 2 at 5 m), separable (x then y)
chi  = chiBed (1 - smoothstep(0.2, 0.5, Fmax))
chiBed = smoothstep(0.25, 0.75, d) (1 - smoothstep(0.35, 0.6, |grad d|)),  0 on the two edge rings, 0 if dispersion is off
```

So a broken cell stays NSWE for `tau_F ln(1/0.2) ~ 3.2 s` after its last
sustain (covers Kennedy's `T* = 5 sqrt(h/g) = 2.3 s` at h = 2 m), waves re-form
in a trough behind a bar and re-break on the beach, and the switch is ramped
over 7.5 m to limit the oscillations Kazolea et al. report at the flag edge.

### 7.4 Classification (CPU, lip module, per crest metre at onset)

```
L0  = g T_p^2 / (2 pi)                    (T_p = swell period)
m   = bed slope along the propagation direction e, averaged over the 20 m seaward of the crest from beach.z (CPU)
xi_b = m / sqrt(H_b / L0)                 (Battjes 1974 breaker Iribarren number; H_b from the crest record)
s_f  = max front slope sF from the crest record
lambda = smoothstep(0.35, 1.0, xi_b) x smoothstep(0.30, 0.55, s_f) x (1 - smoothstep(2.0, 3.0, xi_b))
lambda smoothed alongshore with a 5-point box (12.5 m) per frame
spilling: lambda < 0.25 (foam cap, no lip);  plunging: lambda >= 0.25 (lip, intensity lambda);  surging: xi_b > 2.5 (no lip, surge foam)
```

Secondary cue: onset by C2/C3 before C1 at the same sample adds `+0.15` to
lambda (fronts that steepen before they reach the depth limit plunge).

---

## 8. Wet/dry, friction, speed cap

- Wet/dry: the hydrostatic reconstruction (section 3.2) gives zero flux at dry
  faces and `h >= 0` under the CFL of section 4.2 (`h = max(0, ...)` is a
  guard, not a correction, and is counted in test V1). Dry cells `h <= h_dry =
  1e-4 m` carry no momentum; the shoreline advances by the continuity flux of
  the wet neighbour (run-up, swash, backwash) over sand and rock alike (dry
  rock = wall). Desingularised velocity below 5 cm.
- Friction, semi-implicit per stage after the tridiagonal solve, on wet cells:
  `fr = 1 + dt g n^2 sqrt(u^2 + v^2) / max(h, eps)^(4/3)`; `P /= fr`, `Q /= fr`.
  `n = 0.02` sand, `0.035` rock, from the corner Manning texture averaged to the cell.
- Speed cap: `|V| <= 4 sqrt(g max(h, 0.01)) + 1` m/s (scale P, Q down), which
  only acts on films.
- Dispersion off below `d = 0.25 m` (chiBed), so the swash is pure NSWE.
- Expected: a 2 m, 12 s swell on the crescent preset runs up 1-2 m vertically;
  backwash 1-2 m/s; shoreline set-up 0.1-0.25 H_b (test V10).

---

## 9. GPU implementation

### 9.1 Conventions

three.js 0.160 `WebGLRenderer` owns the context. Every pass is a full-screen
triangle drawn with a `RawShaderMaterial` (`glslVersion: THREE.GLSL3`,
`depthTest: false`, `depthWrite: false`) into `WebGLRenderTarget(nx, ny, {
type: FloatType, format: RGBAFormat, minFilter/magFilter: NearestFilter,
depthBuffer: false, stencilBuffer: false })` or
`WebGLMultipleRenderTargets(nx, ny, count, sameOptions)` with
`layout(location = i) out vec4` outputs. Requires
`renderer.extensions.get('EXT_color_buffer_float')`; verified in this
environment (swiftshader: RGBA32F renderable, 6 draw buffers, float readback).
`renderer.autoClear = false` during the solver; state is sampled with
`texelFetch(tex, clamp(ivec2, 0, size-1), 0)` only. The pipeline is restored
with `renderer.setRenderTarget(null)` after the solver; raw GL calls are
limited to the readback (section 9.5).

### 9.2 Textures (all RGBA32F, nx x ny unless noted; r = 1 sizes in MB)

| Name | Channels | Written by | MB |
|---|---|---|---|
| `S[0..3]` ring | (h, P, Q, F) for n-1, n, n+1 and the stage temporary | finalize | 4 x 2.3 |
| `BATHY` | (B, d, chiBed, n_manning) | bed pass | 2.3 |
| `BATHY2` | (Bx, By, d_x clamped, d_y clamped) | bed pass | 2.3 |
| `CORNER` (nx+1)x(ny+1) RG32F | (Bcorner, n_corner) | uploaded `DataTexture` from `layout.cornerBed` | 1.1 |
| `CHI` | (chi, Fmax, relax weight s, m_lat) | chi passes (relax channels by the bed pass) | 2.3 |
| `FLUX` MRT3 | T0 (F_h, F_P, F_Q, hsE_L) ; T1 (G_h, G_P, G_Q, hsN_L) ; T2 (hsE_R, hsN_R, a_max_local, 0) | flux pass | 6.9 |
| `PROV` | (h*, U*, V*, eta_t_stage) | rhs pass | 2.3 |
| `PCR[0..1]` MRT2 ping-pong | T0 (a, b, c, r) x-system ; T1 (a, b, c, r) y-system | PCR passes | 4 x 2.3 |
| `PQ` | (P, Q, 0, 0) after the last PCR level | PCR last pass | 2.3 |
| `COMP[0..3]` | (S_m, A_m, S_m', A_m') for components 2m, 2m+1 | uploaded by the worker result | 4 x 2.3 |
| `KHAT[0..3]` | (kx_m, ky_m, kx_m', ky_m') | uploaded | 4 x 2.3 |
| `DIR` | (D, tau, 0, 0) | uploaded | 2.3 |
| `STATS[0..1]` ping-pong, MRT2 each | T0 (mean eta, mean eta^2, breaking fraction, wetness) ; T1 (mean P, mean Q, mean |u|, eta_max decaying) | stats pass | 4 x 2.3 |
| `STAMP` 256x1 RGBA32F | impact splats (I, J, radius cells, strength) | uploaded `DataTexture` per frame | - |
| `CREST` MRT2, V x K (K = 6) | section 11.1 | crest-scan pass | - |
| `MAXS` 1x1 (and a 16x16 intermediate) | max speed | reduction | - |
| Renderer-side: `waveRT` M x M RGBA16F (eta, n_x, n_y, foamSrc); foam ping-pong M x M RGBA16F; `LIPMASK` M x M R16F; `CRESTSTRIP` V x 4 RGBA32F `DataTexture` | | | |

Total ~45 MB at r = 1, ~180 MB at r = 2, ~11 MB at r = 0.5.

### 9.3 Pass list per step (tier S, L = 9)

Per RK stage (input state `S_in`, previous state `S_prev` for Pt/Qt):

| # | Pass | Reads | Writes | Fetches/texel |
|---|---|---|---|---|
| 1 | FLUX | S_in (5 cells in x and y: I-1..I+2), BATHY, BATHY2, CORNER (edge faces) | FLUX MRT3 | ~34 |
| 2 | RHS | S_in (5x5 for Wd), S_prev + S_n (Pt, Qt, 3x3), FLUX (c, cW, cS), BATHY, BATHY2, CHI | PROV (h*, U* = L_x P + dt R_P, V*, eta_t) | ~70 where chi > 0, ~22 elsewhere |
| 3..3+L-1 | PCR level l = 1..L | level 1: PROV, BATHY, BATHY2, CHI at i, i-s, i+s (both systems); later: PCR ping-pong at i, i-s, i+s | PCR MRT2; last level writes PQ = (r/b) | 6 x 2 systems |
| 3+L | FINALIZE | PROV, PQ, BATHY, S_n (stage 2 average), COMP/KHAT/DIR + STATS (relaxation, stage 2 only), S_n + STAMP (F update, stage 2 only) | S_out (h, P, Q, F) | ~12 (stage 1), ~45 in zones (stage 2) |

Per step: `2 (L + 3) = 24` passes, plus CHI-X and CHI-Y (separable dilation of
F into Fmax, chi) = 26, plus STATS (every step, EMA with `alpha = dt / 30 s`
for means, `dt / 60 s` decay for wetness `= max(prev x decay, h > 0.005)`,
`eta_max = max(prev - dt x 0.02 m/s, eta)`) = 27. Per frame (not per step):
CREST (1), LIPMASK raster (1, renderer), PRESENT (1), FOAM (1), MAXS reduction
(2, at <= 1 Hz). Bed pass (1) on sculpt/tide; hot-start pass (1) on reset.

Cost: ~141k texels x ~430 fetches/texel/step = 61 M fetches and 27 draw calls
per step; 24 steps per simulated second. Estimates: Iris Xe / M1-class iGPU
4-7 ms per step (10-17% of wall time at 1x, 40-70% at 4x, where the scheduler
slows the clock); discrete laptop GPU 1-2 ms. Tier H: 565k texels, L = 10,
48 steps/s: 5-9 ms per step on a discrete GPU, not viable on an iGPU (hence
the 9 ms gate). Measured here (swiftshader): 7.6 ms per 15-fetch 384^2 pass,
so a tier-S step is ~0.3 s; browser tests use `nx, ny <= 128` grids.

### 9.4 Pass details that are easy to get wrong

- FLUX writes the face between `c` and `cE` at `c` (east face) and between `c`
  and `cN` at `c`; for `I = nx-1` / `J = ny-1` the face is the wall (mirror).
  The west wall flux of column 0 and the south wall flux of row 0 are computed
  in RHS from the cell's own west/south face values (as the oracle's `FhW0`,
  `GhS0`).
- RHS reads `F_h(W)` as `FLUX[cW].x` (clamped: for I = 0 it computes the wall
  flux itself), likewise `hsW = FLUX.T2[cW].x`.
- RHS computes `L_x P_in` with the frozen chi (3 fetches of S_in per axis) and
  adds `dt R`; it also writes `eta_t_stage = R_h` (unused in v1, kept for
  debugging).
- PCR level 1 builds `(a, b, c)` from `BATHY.d`, `BATHY2.d_x/d_y`, `CHI.chi`
  for the three elements it touches; `r = PROV.U*` (x) and `PROV.V*` (y).
  Identity rows for `chi = 0`. Out-of-range elements: `a = c = 0, b = 1, r = 0`.
  The last level outputs `(r_x/b_x, r_y/b_y, 0, 0)` into `PQ`.
- FINALIZE stage 1: `h = PROV.h*`, `P = PQ.x`, `Q = PQ.y`, zero on dry,
  friction, cap, `F` copied from `S_n`. Stage 2: the same then
  `h = (h_n + h)/2` etc., relaxation blend (only where `CHI.s > 0`), F update
  (section 7.2) using `S_n.h` as `h^n`, STAMP applied, written to `S_(n+1)`.
- CHI-X writes `Fmax_x` into `CHI.y`; CHI-Y reads it and writes `chi` and the
  final `Fmax`. The relaxation channels `s`, `m_lat` are written by the bed
  pass and preserved (CHI passes copy them through).
- Every division has a floored denominator; `max(h, h_dry)` where `1/h`
  appears; the finalize pass replaces any non-finite value by 0 and sets a
  flag in `MAXS.w` which triggers `hotStart` from the app (NaN guard).

### 9.5 Readback (crest records, stats, max speed)

Asynchronous path (primary):

```
const gl = renderer.getContext();
renderer.setRenderTarget(crestRT);                       // binds the FBO
gl.readBuffer(gl.COLOR_ATTACHMENT0 + k);                 // per MRT attachment
gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo[k][ring]);
gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, 0);
gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);               // explicit: three.js never binds this target and resetState() does not clear it
gl.readBuffer(gl.COLOR_ATTACHMENT0);
fence[ring] = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
renderer.setRenderTarget(null);
// later frames: if gl.getSyncParameter(fence, gl.SYNC_STATUS) === gl.SIGNALED:
gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo); gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, dst); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null); gl.deleteSync(fence)
```

Ring of 3 PBOs per target; latency 1-2 frames, dead-reckoned by the lip module
(section 11.3). Fallback (`options.syncReadback` or when `fenceSync` is
missing): `renderer.readRenderTargetPixels` per attachment (stalls the pipeline;
used in headless tests). Sizes: CREST 2 x V x K x 16 B = 49 KB at r = 1 (98 KB at
r = 2); STATS downsampled 4x by a small pass to (nx/4) x (ny/4) x 2 targets =
285 KB at 2 Hz; MAXS 16 B at 1 Hz.

### 9.6 Tier-specific switches

Tier P: `K = 4` crest slots, spray off, STATS at 1 Hz, `R_c = 2`. Tier C
(CPU): `src/boussinesq.js` in `solver-worker.js` with `dispersion: false`
(NSWE only: no tridiagonal), 188 x 189 at 4.98 m, posting `Float32Array(h, P,
Q, F)` per step (~8-12 ms per step on a laptop core, 12 steps/s needed), uploaded
as a `DataTexture` into `S[n+1]`; the renderer path is identical from there.
Tier L: the current pipeline, untouched (`render.js` keeps `waveMat`).

---

## 10. JavaScript reference solver (oracle)

`src/boussinesq.js` (existing, Float64, no DOM). It is the specification of
the physics; the GPU must match it (tests V13-V14). Its API (already exported):

```js
import { createSolver, omegaMS, wavenumberMS, wavenumberAiry, relaxField, G, BQ } from './boussinesq.js';
const s = createSolver({
  nx, ny, dx,                 // cells and spacing (m)
  bed,                        // Float64Array((nx+1)*(ny+1)) corner elevations, stride nx+1
  wl = 0,                     // still water level
  manning = 0.02,             // number or Float32Array(nx*ny)
  periodicX = false, periodicY = false,   // tests only
  dispersion = true, breaking = true,
  order = 2,                  // 2 = minmod MUSCL (GPU), 4 = MUSCL-TVD (oracle experiments)
  rk = 2,                     // 2 = SSP-RK2 (default), 3 = SSP-RK3                      [to add]
  theta = 1.3, eps = 0.05, hDry = 1e-4, dispDepth = [0.25, 0.75],
  breakingParams = { onsetRatio: 0.8, onsetSlope: 0.65, slopeRes: 0.18, onsetGammaMin: 0.25, slopeCrit: 0.40, keepSlope: 0.15, memory: 2.0, ramp: 3 },  // [slopeCrit, onsetGammaMin, advection to add]
  relax,                      // Float32Array(nx*ny) weights s in [0,1] (relaxField(...) masked by m_lat)
  target,                     // (t, etaT, PT, QT) => void, fills the three Float64Arrays(nx*ny)
  eta,                        // optional initial surface
});
s.step(dt); s.setBed(bed, manning); s.setWaterLevel(wl); s.reset(eta); s.setTarget(fn);
s.h, s.P, s.Q, s.F, s.chi, s.B, s.d   // Float64Array(nx*ny), live
s.t, s.wl, s.eta(out), s.surface(out), s.maxSpeed(), s.volume(), s.cfl(nu)
s.stamp(list)                 // [{I, J, rCells, strength}] -> F = max(F, strength) in the disc        [to add]
s.loadState(h, P, Q, F) / s.readState()                                                                 [to add, tests]
```

Additions required (Engineer B, day 1): the C3 slope criterion and the
`gamma_b` floor, F advection with the front, the `m_lat` mask in `relaxField`,
the mean-field term in the target helper, `rk = 3`, `stamp`, `loadState`,
`readState`, and the shared constants moved to `src/engine/params.js` which
both the oracle and the shaders import (the shader sources embed them through
template strings so there is one source of truth).

Roles: (1) oracle for the GPU (pass-twin harness, section 15.2); (2) tier C
runtime in the worker; (3) the Node validation battery; (4) offline tools
(run-up curves, dispersion scans).

Performance: ~300 flops per cell per stage; 128 x 128 in ~8 ms per step; the 1D
channel cases (400-2000 cells) run in seconds.

---

## 11. Crest, lip and barrel model

### 11.1 Crest-scan pass (GPU, per frame after the last substep)

Output `V x K` texels (`K = 6`, tier P: 4), two RGBA32F attachments, one
fragment per `(visible column I = ML + col, slot k)`. The fragment walks
`J` from `MO` to `MO + V - 1` over the blended state (`lerp(S_n, S_(n+1),
alpha)`), on wet cells `h > 0.15 m`, with `eta = h + B - wl`:

```
crest at J:  eta[J] >= eta[J-1] and eta[J] > eta[J+1]
trough Jt:   first J' > J with eta[J'] <= eta[J'+1] (or the first dry cell);  prominence eta[J] - eta[Jt] >= 0.08 m
sub-cell row: Jc = J + 0.5 (eta[J-1] - eta[J+1]) / (eta[J-1] - 2 eta[J] + eta[J+1])  (clamped to J +- 0.5)
sF = max over J' in (J, Jt] of (eta[J'-1] - eta[J']) / dxs
```

Slots are filled in order from offshore; empty slots have `Jc = -1`. Record
layout `Float32Array(V K 8)`, index `(col K + k) 8 + ch`:

```
T0: 0 Jc (solver row, float)   1 etaC (m)   2 hC = h at the crest (m)   3 FC = F at the crest
T1: 4 etaT = eta at the trough ahead (m)   5 Jt (row of that trough)   6 sF (m/m)   7 uC = Q/h at the crest (m/s)
```

Read back through the PBO ring (section 9.5); `readCrests()` returns the
newest complete record or `null`, plus `latencyFrames`.

### 11.2 Tracker (CPU, `src/breakers/tracker.js`)

1. **Columns to lines.** Scan columns left to right; a record in column `col`
   joins the line that holds a record in column `col-1` with `|Jc - Jc'| <= 3`
   (any slot; nearest wins). A line ends where no column matches. Lines
   shorter than 4 samples (10 m at 2.5 m) are discarded. Sample positions in
   grid metres: `x = (col + 0.5) dxs`, `y = (Jc - MO + 0.5) dxs`.
2. **Frames to tracks.** Each line is matched to last frame's tracks by the
   overlap of column ranges and the predicted row `J_pred = J_prev + c dt /
   dxs` (match if `|Jc - J_pred| <= 2` for >= 60% of the overlapping columns);
   best overlap wins; unmatched lines start new tracks (`id` from a counter);
   a track without a match for 0.5 s is dropped; a track must persist 0.3 s
   before it may spawn lips (hysteresis). Per-sample crest speed `c` is an
   exponential average (`alpha = 0.3`) of `(Jc - J_prev) dxs / dt`, floored at
   `0.5 sqrt(g hC)` and capped at `1.5 sqrt(g hC)`; the fallback before two
   frames exist is `sqrt(g hC) (1 + 0.5 etaC / hC)`.
3. **Propagation direction** per sample: crest tangent `tau = normalize(p_(s+1)
   - p_(s-1))`, `e = (tau_y, -tau_x)` flipped so `e_y > 0` (shoreward).
4. **Per-sample state** `{ t0, H_b, h_b, c_b, e, xi, lambda, type, stage,
   kin }` set at onset: the first frame where `FC >= 0.5` and the sample's
   previous `FC < 0.5` (re-trigger allowed after `FC` has been below 0.2 for
   1 s: a reformed wave). `H_b = etaC - etaT`, `h_b = hC`, `L_front = (Jt - Jc)
   dxs`, `c_b = c`.
5. **Dead reckoning.** Every position used by the mesh, the lip mask and the
   crest strip is advanced by `c e x (latencyFrames x dt_frame)` so the
   sheet stays on the crest at 60 fps despite the 1-2 frame readback latency.

### 11.3 Lip kinematics (`src/breakers/lip.js`, per plunging sample)

Inputs: `H_b`, `h_b`, `c` (tracked), `lambda`, bed gradient `1:x` with
`x = 1 / max(tan beta, 0.02)` (tan beta = the 20 m averaged bed slope along
`e`), `e`, `wl`, `etaC0`, `etaT0`. Ground frame, `k` = up.

```
H_t     = 0.85 H_b                                         tube height
w0      = 0.15 c                                           vertical launch speed
tau_imp = (w0 + sqrt(w0^2 + 2 g H_t)) / g                  fall time from the apex to the floor
c_r     = 0.85 c                                           root (crest) speed during the throw and persist phases
r_v     = clamp(0.065 x + 0.82, 1.3, 3.0)                  vortex ratio length/height (Mead & Black 2001, gradient as 1:x; verify against the paper)
W_t     = r_v H_t                                          target tube length
V_h     = clamp(c_r + W_t / tau_imp, 1.1 c, 1.6 c)         horizontal tip speed, ground frame (Grilli et al. 1997; Perlin et al. 2013: 1.1-1.6 c)
X_p     = V_h tau_imp                                      ground plunge distance from the launch point
X_rel   = (V_h - c_r) tau_imp                              tube length actually drawn (= W_t unless the clamp bound)
u_rel   = V_h - c_r
tau_p   = 0.6 sqrt(H_b / g)                                pitching duration
z_apex  = wl + etaC0 + 0.10 H_b                            apex height (Peregrine 1983: ~10% crest rise while pitching)
z_floor = wl + etaT0 + 0.10 H_b                            tube floor
```

Worked numbers (test V17): `H_b = 2 m, c = 6.6 m/s, lambda = 1, bed 1:20`:
`H_t = 1.70, w0 = 0.99, tau_imp = 0.698 s, c_r = 5.61, r_v = 2.12, W_t = 3.60,
V_h = 10.77 -> clamped 10.56 (1.6 c), X_p = 7.37 m = 3.69 H_b, X_rel = 3.46 m`.
Bed 1:10: `r_v = 1.47, W_t = 2.50, V_h = 9.19 (1.39 c), X_p = 6.42 m = 3.2 H_b`.
`H_b = 1 m, c = 4.7`: `X_p = 3.71 m = 3.7 H_b`; `H_b = 3 m, c = 8.1`: `X_p = 11.05 m
= 3.7 H_b`. All inside the Smith & Kraus (1991) / Galvin (1969) range
`X_p / H_b in [3, 4]`, which is the single calibration target: tune `c_r`,
`w0` and the `V_h` clamp, in that order, never `H_t`.

Timeline per sample (`tau' = t - t0`, launch at `t1 = t0 + tau_p`, `tau = t - t1`):

| Stage | Window | What happens |
|---|---|---|
| pitching | `0 <= tau' < tau_p` | apex rises linearly to `z_apex`; face push ramps 0 -> 1; no sheet; root moves at `c` |
| throw | `0 <= tau < tau_imp` | sheet grows from the apex (11.4); lowering follows the tip; root at `c_r` |
| impact | `tau = tau_imp` | splats (strength 1, radius `0.6 H_b`) along the impact line at `root + e X_rel`; `stampImpacts` disc radius 2 cells, strength 1; splash-up spawned; spray burst 200 per sample (tier S/H) |
| persist | `tau_imp <= tau < 2.5 tau_imp` | sheet frozen at full length, rides with the root at `c_r`; splash-up sheet lives `t_s = 2 sqrt(1.8 H_b / g)` (1.21 s for 2 m) |
| collapse | `2.5 tau_imp <= tau < 2.5 tau_imp + 0.4 s` | sheet nodes lerp down onto the solver surface, thickness and opacity -> 0, lowering and push -> 0, foam coverage in the footprint -> 1 |
| broken | after | sample retires; the solver bore + foam are the only surface; may re-trigger (11.2 step 4) |

Lifetime for a 2 m wave: `0.27 + 0.70 + 1.05 + 0.40 = 2.4 s`.

Splash-up (Peregrine 1983): a second, lower sheet launched from the impact
line with ground velocity `0.8 c e + sqrt(1.8 g H_b) k` (peak height `0.9 H_b`
at `sqrt(1.8 H_b / g)` = 0.61 s for 2 m), 8 profile nodes, thickness `0.08 H_b`
tapering to `0.01 H_b`, foam strength 1, plus its own spray (40 per sample).

Spilling samples (`lambda < 0.25`): no sheet; foam cap source `0.6` over
`[root - 0.3 H_b, root + 0.5 H_b]` and crest-cap spray (tier S/H). Surging:
no sheet; foam source `0.4` in the swash tongue only.

### 11.4 Sheet geometry

In the frame of the root `R(t) = R(t1) + c_r e (t - t1)` (dead-reckoned,
section 11.2 step 5), a fluid element launched at time `t1 + tau - l` has flown
for `l`; the sheet at time `tau` is the curve

```
p(l) = R(t) + e (u_rel l) + k (z_apex - z_R + w0 l - g l^2 / 2),   l in [0, min(tau, tau_imp)]
```

where `z_R` is the root's own height (the water surface at `R`, sampled in the
vertex shader from `waveRT` so the mesh stitches). The tip is `l = tau`. The
cavity between the sheet, the pushed face and the floor is close to an ellipse
with horizontal extent `X_rel` and vertical extent `z_apex - z_floor = 0.9 H_b`
(Longuet-Higgins 1982 parametric breaker; New 1983 ellipse), tilted forward by
the parabola; `r_v` enters through `W_t`, so steeper beds give rounder tubes.

Thickness along the sheet: `th(l) = H_b (0.18 (1 - l / tau_imp) + 0.02) (1 +
0.2 sin(2 pi s / 8 m + 1.7 t))` (breathing along the crest coordinate `s`,
wavelength 8 m), clamped >= 0.02 m.

### 11.5 Mesh construction (`src/breakers/mesh.js`)

One `THREE.BufferGeometry` for all sheets: capacity `MAXL = 8` lines x `MAXS =
48` samples x `K = 20` profile nodes = 7680 vertices, indexed quads (2
triangles each), `DynamicDrawUsage`; rebuilt on the CPU every frame (< 1 ms).
Samples every 2.5 m along the crest line (resampled from the tracked line by
linear interpolation of the per-sample state; `t0` interpolated so peeling is
continuous). Profile nodes per sample:

```
k = 0..3   back of the crest: from the solver surface at xi = -0.6 H_b (height from waveRT in the vertex shader) up to the apex (k = 3 at z_apex),
           xi = -0.6 H_b, -0.4 H_b, -0.15 H_b, 0 ; heights lerp(surface, z_apex, (k/3)^0.7)
k = 4..19  sheet p(l_k), l_k = (k - 4) / 15 x min(tau, tau_imp)      (k = 19 is the tip)
```

Attributes: `position` (world), `normal` = normalize(cross(profile tangent,
crest tangent)) oriented so the outside normal points away from the cavity,
`uv = (s / 8 m, l / tau_imp)`, `aThick` (11.4), `aAux = (lambda, stage phase
0..1, age t - t0, foamEdge)`. The splash-up sheets use the same buffer family
(`MAXS x 8` nodes). `DoubleSide`, rendered after the water with depth test on
and depth write off.

Material (`src/breakers/material.js`): thin-sheet water: transmitted colour
`exp(-sigma aThick)` tinted towards the water body colour (green room: `sigma =
(2.2, 1.1, 0.9) /m`), reflected sky by Fresnel (Schlick, F0 = 0.02), back-face
normal for the inside of the tube, foam (white, from `aAux.w` and the solver
foam sampled at the root) along the lip edge and the underside, specular lip
line, alpha from thickness (1 at `th >= 0.1 m`, 0.5 at the tip), fading to 0
during collapse.

### 11.6 Surface under the barrel: lip mask and crest strip

**Lip mask** (`LIPMASK`, M x M R16F, cleared to 0 each frame): for every active
plunging sample the CPU emits a footprint quad in grid coordinates (`2.5 m`
wide along the crest, from `xi = -0.3 H_b` to `xi = X_rel + 0.5 H_b` along `e`);
the fragment shader writes the lowering

```
xi   = position along e from the root (m)
lower = max(0, W_s - z_floor) x smoothstep(0.15 X_rel, 0.45 X_rel, xi) x (1 - smoothstep(X_rel, X_rel + 0.5 H_b, xi)) x on(t)
```

(`W_s` is the blended solver surface sampled from `waveRT` before the mask;
`on(t)` ramps 0 -> 1 as the tip passes `xi` during the throw and 1 -> 0 during
collapse). The present pass subtracts `lower` from the surface so the solver
bore never pokes through the roof or floor. Blending `MAX`.

**Crest strip** (`CRESTSTRIP`, V x 4 RGBA32F `DataTexture`, rows = up to 4
overlapping breakers per column): per visible column `(y_root in grid m,
X_rel, H_t, push_on)` plus `(e_x, e_y, z_floor, z_apex)` in a second row pair.
The water vertex shader, for a vertex at `xi in [-0.3 H_b, 0.3 X_rel]` along
`e` from the root, with height fraction `zeta = clamp((W - z_floor) / (z_apex -
z_floor), 0, 1)`, moves the vertex horizontally by

```
delta = 0.35 X_rel zeta^1.5 x push_on x smoothstep(-0.3 H_b, 0, xi) x (1 - smoothstep(0.2 X_rel, 0.3 X_rel, xi))   along e
```

which turns the solver's 2-3 cell S-shaped front into a vertical / concave wall
whose top overhangs by `0.35 X_rel` and meets the sheet root (nodes 0-3). The
normal is recomputed from the displaced neighbours in the vertex shader.

### 11.7 Foam, splats and the solver stamp

`brk.update()` returns `splats: [{ x, y, r, s }]` (grid metres, radius m,
strength 0..1): impact lines (every 2.5 m sample, `r = 0.6 H_b`, `s = 1`),
splash-up (`r = 0.8 H_b`, `s = 0.7`, drifting with `0.8 c e`), collapse
coverage (`r = X_rel/2`, `s = 1`). The renderer draws them as additive
sprites into the foam target (section 12.1). The same list, converted to
solver cells, goes to `sol.stampImpacts(list)` (`rCells = 2`, `strength = 1`,
impact splats only) which uploads the `STAMP` texture (max 256 per frame,
oldest dropped) read by FINALIZE in the next step.

### 11.8 Spray (`src/breakers/spray.js`)

CPU pool of 4096 particles (tier P: 0), `THREE.Points` with a soft round
sprite, size by distance (`8 px x 60 m / dist`, clamped 2..24 px), additive
over the sky and alpha over the water, lit by the sun colour, fading to the
foam colour in the last 30% of life. Integration (semi-implicit Euler):

```
a = (0, -g, 0) - (v - w) |v - w| / L_d,  L_d = 3 m  (quadratic drag; spindrift feathering),  w = wind vector (UI: 0, or offshore 4-8 m/s along -e)
life 1.2-1.8 s uniformly; killed below z_floor of its emitter or at end of life
```

Emitters: lip tip (rate `60 lambda` per metre per second, velocity = tip
velocity `(u_rel + c_r) e + (w0 - g tau) k` +- 15% + 0.5 m/s isotropic noise),
impact burst (200 per sample, velocity `(0.4 c e + sqrt(1.8 g H_b) k)` +- 40%),
splash-up top (40 per sample), spilling crest cap (10 per metre per second,
`0.3 c e + 1.5 k`). Seeded RNG (mulberry32) so screenshots are deterministic.

---

## 12. Foam and the present pass

### 12.1 Foam (M x M RGBA16F ping-pong, as today)

Per frame: `foam' = foam(x - u dt) exp(-dt / tau_foam) + src dt + splats`, with
`u = (P/h, Q/h)` from the blended state sampled at the renderer's position
(section 1.4; so whitewater rides the bore, drifts seaward in rips and
alongshore in the longshore current, and runs up the sand with the swash),
`tau_foam` = the existing decay constant of `render.js`. Sources at a render
texel (all evaluated from the blended state and F):

```
bore:   F x clamp(eta_t / (0.6 sqrt(g h)), 0, 1) x (h > 0.1)                  eta_t = (h_(n+1) - h_n) / dt
swash:  0.5 x (h < 0.3 m) x smoothstep(0.5, 1.0, |u|)
reform: 0 (F decays by itself)
splats: additive sprites, strength s inside radius r with a smooth edge
```

Foam is read by the water fragment shader as now (whitewater colour), by the
sheet material at its root, and the shore foam line now comes from the swash
term only (the old hard-coded shore foam is removed).

### 12.2 Present pass (solver -> `waveRT`)

`waveRT` (M x M RGBA16F) = `(eta_r, n_x, n_y, foamSrc)` where `eta_r =
W_bicubic - wl - lower` with `W_bicubic` the Catmull-Rom bicubic resample of
`W = B + h` (dry cells contribute `B`, so the sand shows) of
`lerp(S_n, S_(n+1), alpha)` at the mapping of section 1.4; `n_x, n_y` the
analytic Catmull-Rom derivatives of `W` minus the derivatives of `lower`
(central differences on `LIPMASK`); `foamSrc` = bore + swash sources above.
The water vertex shader uses `eta_r` as the vertical displacement (the old
horizontal `fwd` lip offset is removed), the crest-strip push (11.6) as the
only horizontal displacement, and the fragment stage keeps the existing
procedural detail normal (wind ripples scaled by depth) on top.

---

## 13. Render integration and the per-frame data path

```
app.loop(t):
  dt_frame = min(0.05, elapsed)                    (paused: 0)
  painting -> brush(); R.setHeight(beach); sol.setBathymetry(beach, wl)    (every frame while painting)
             requestSolve() every 350 ms (linear field on the extended grid -> sol.setTargets(res) when it returns)
  t_target += dt_frame x speed_eff;  n = min(4, floor((t_target - t_sim)/dt));  for n: sol.step(dt, t_sim); t_sim += dt
  alpha = (t_target - t_sim + dt) / dt  in [0, 1)  (interpolation between S_n and S_(n+1))
  sol.scanCrests(alpha); records = sol.readCrests()            (async; null until ready)
  brk.update(records, dt_frame, { t: t_target, wl, beach, swell, latencyFrames })
      -> sheet/splash geometry, spray buffers, splats, lip-mask quads, crest strip
  sol.stampImpacts(brk.splats.filter(impact))
  R.setLipMask(brk.maskQuads); R.setCrestStrip(brk.crestStrip)
  R.frame({ alpha, states: sol.textures, foamSplats: brk.splats, dt: dt_frame })
      LIPMASK pass -> PRESENT pass -> FOAM pass -> scene (sky, terrain with wetness, water, sheets, splash, spray, walls)
  every 0.5 s: sol.readStats() -> probe, verdict, camera spots;  every 1 s: sol.maxSpeed() -> dt check;  every 4th frame: updateProbe()
```

### 13.1 `render.js` changes

- `setEngine(sol | null)`: when set, `frame()` skips `waveMat` (the 8-component
  sum) and runs LIPMASK, PRESENT and FOAM from the solver textures; when null
  the legacy path runs unchanged (tier L).
- `setLipMask(quads)`, `setCrestStrip(dataTexture)`, `addBreakers(group)`.
- Water material: vertex reads `waveRT.x` (vertical) and the crest strip
  (horizontal push); normals from `waveRT.yz` recomputed after the push;
  fragment adds a milky aeration term where `F > 0.3` (sampled from the
  blended state) and the foam as now. Terrain shader darkens sand by the
  STATS wetness channel (`mix(dry, wet, 0.6 wetness)`).
- Diorama walls read the blended state along their section (bores show in
  section).
- Debug modes (`R.debug = 'F' | 'chi' | 'foam' | 'mask' | 'current' | null`)
  draw the chosen field as a colour overlay on the water.

### 13.2 Probe, verdict, cameras

Probe (hover): depth from `beach.z` (CPU) as now; `H_rms = sqrt(8 (mean eta^2
- mean eta^2))`, `H_s = 1.416 H_rms`; set-up = mean eta; mean current `(mean P,
mean Q) / max(d + mean eta, 0.1)` as speed and bearing; "breaking here N% of
the time" from the breaking fraction; breaker kind from the nearest tracked
crest within 25 m (`plunging 68% of the last 20 s`, else `spilling`, `bore`,
`surging`). No "undertow" wording anywhere.

Verdict (replaces `updateVerdict(res.summary)` once STATS has 2 periods of
data; the linear summary shows before that): per visible column the first
row from offshore with breaking fraction > 0.3 gives the break distance (to
the shoreline = first dry cell of the column at mean water) and depth `d`
there; `H_s` there; the type from the energy-weighted mean of `lambda` over the
last 20 s of breakers (`< 0.25` spilling, `< 0.6` "spilling to plunging", else
plunging; surging if `xi > 2.5` dominates); barrel count = plunging samples
that reached impact in the last 20 s / 10; peel direction and speed from the
alongshore gradient of `t0` (`peel speed = 2.5 m / mean(dt0 per sample)`).

Cameras: Lineup / Impact / Beach stand at the column maximising `breaking
fraction x H_s^2` (same selection logic as today, fed by STATS instead of the
linear `D` field); Impact = 1.5 m + 0.5 H_s above the trough, 15 m inside the
mean break line, looking along the crest. New Barrel view: follows the longest
active plunging segment, eye at `root + 0.6 X_rel e + 0.4 H_t k` above the
floor, looking along the crest tangent toward the peel; falls back to Impact
when no barrel is active for 5 s.

### 13.3 Breakers output record (`brk.update` return value)

```js
{
  sheets: BufferGeometry (updated in place), splash: BufferGeometry, spray: Points,
  splats: [{ x, y, r, s, impact: bool }],                       // grid metres
  maskQuads: Float32Array(nQuads * 4 * 6),                       // 4 corners x (x, y, xi, X_rel, z_floor, on) per quad
  crestStrip: DataTexture (V x 4 RGBA32F) + Float32Array backing,
  active: [{ id, type, lambda, xi, H, h, c, stage, samples: [{ x, y, etaC, t0, stage }] }],
  stats: { plungingMetres, impactsLast20s, peelSpeed, peelDir }
}
```

---

## 14. Module interfaces

### 14.1 Files and owners

| Module | Files | Owner |
|---|---|---|
| Shared parameters and layout | `src/engine/params.js`, `src/engine/layout.js` | B (day 1 morning, agreed by all) |
| Oracle solver and target helpers | `src/boussinesq.js` (extend), `src/engine/targets.js`, `src/waves.js` (rectangular grids), `src/solver-worker.js` | B |
| GPU solver | `src/engine/gpu-solver.js`, `src/engine/readback.js`, `src/engine/shaders/{common,bed,flux,rhs,pcr,finalize,chi,stats,crest,reduce,hotstart}.glsl.js` | A |
| Breakers | `src/breakers/{index,tracker,lip,mesh,material,spray,strip}.js` | C |
| Renderer and app | `src/render.js`, `src/app.js`, `index.html`, `README.md` | C (render), B (app scheduler/tiers/probe) |
| Tests | `test/boussinesq.test.mjs` (extend), `test/breakers.test.mjs`, `test/gpu-equivalence.mjs`, `test/pass-twins.mjs`, `test/soak.mjs`, `test/screens.mjs`, `test/pages/*.html`, `test/headless.mjs` (existing) | B (solver), C (breakers, screens), A (pass twins) |

### 14.2 Signatures

```js
// src/engine/params.js  (single source of truth; shaders embed these via template strings)
export const G = 9.81, BQ = 1/15, THETA = 1.3, EPS = 0.05, H_DRY = 1e-4, DISP_DEPTH = [0.25, 0.75];
export const SLOPE_CLAMP = 0.2, BED_SLOPE_OFF = [0.35, 0.6], CFL_NU = 0.25, MANNING = { sand: 0.02, rock: 0.035 };
export const BREAK = { onsetRatio: 0.8, slopeRes: 0.18, gammaMin: 0.25, gammaMax: 0.65, slopeCrit: { 1.25: 0.577, 2.5: 0.40, 5: 0.30 }, keepSlope: 0.15, memory: 2.0, rampMetres: 7.5, chiRamp: [0.2, 0.5], minDepth: 0.1 };
export const RELAX = { power: 3.5, offshoreM: 300, lateralM: 150, lateralMinDepth: [0.8, 1.2], meanTau: 30 };
export const LIP = { tubeHeight: 0.85, w0: 0.15, rootSpeed: 0.85, vortex: [0.065, 0.82, 1.3, 3.0], tipClamp: [1.1, 1.6], pitch: 0.6, apexRise: 0.10, floorRise: 0.10, persist: 2.5, collapse: 0.4, lambdaMin: 0.25, splash: { fwd: 0.8, up2: 1.8 } };
export const TIERS = { H: { r: 2 }, S: { r: 1 }, P: { r: 0.5, K: 4, spray: false } };

// src/engine/layout.js
export function layout(r) // -> { r, dxs, V, ML, MO, MI, nx, ny, K }
export function cornerBed(beach, wl, lay, out?)      // Float32Array((nx+1)(ny+1)) corner elevations (datum), clamp-extended
export function cornerManning(beach, lay, out?)       // Float32Array((nx+1)(ny+1)) 0.02 / 0.035
export function cellBed(cornerBed, lay)               // { B, Bx, By } Float32Array(nx ny) (CPU twin of the bed pass)
export function relaxWeights(lay, dCell)              // Float32Array(nx ny): s with the m_lat mask applied
export function cellToGrid(lay, I, J) / gridToCell(lay, x, y) / texelOf(lay, x, y)

// src/engine/targets.js
export function extendedBathymetry(beach, wl, lay)    // { z: Float32Array(nx ny) cell-centre bed, nx, ny, dx: dxs } for waves.solve
export function buildTargetTextures(res, lay)         // { comp: DataTexture[4], khat: DataTexture[4], dir: DataTexture } (RGBA32F nx x ny)
export function targetField(res, lay, wl, means?)     // (t, etaT, PT, QT) => void   for the oracle; `means` = { eta, P, Q } per row or null

// src/waves.js
export function solve({ N?, nx?, ny?, dx, z, wl, swell })   // N keeps the legacy square call; nx, ny rectangular; returns S[8], A[8] on nx x ny, D, tau, omegas, phases, setOmega, setDepth, ux, uy, k, summary

// src/engine/gpu-solver.js
export function createGpuSolver(renderer, { r, syncReadback = false, dispersion = true, breaking = true })
// -> { layout, setBathymetry(beach, wl), setWaterLevel(wl), setTargets(res), hotStart(t), step(dt, t),
//      textures: { state: () => Texture /*S_n*/, next: () => Texture /*S_(n+1)*/, bathy, chi, stats0, stats1 },
//      scanCrests(alpha), readCrests() -> { data: Float32Array(V K 8), latencyFrames } | null,
//      readStats() -> { w, h, t0: Float32Array, t1: Float32Array } | null   (downsampled 4x, async, 2 Hz),
//      maxSpeed() -> number | null (async), stampImpacts(list), readState() -> Float32Array(nx ny 4) (sync, tests), loadState(h, P, Q, F),
//      passesPerStep, timing: { stepMs }, nanCount, dispose() }

// src/engine/readback.js
export function createAsyncReader(renderer, rt, { attachments: [0, 1], ring: 3, width, height })
// -> { request(), poll() -> Float32Array[] | null, latencyFrames, dispose() }  ; falls back to readRenderTargetPixels when fenceSync is missing

// src/breakers/index.js
export function createBreakers({ layout, wl, beach, swell, tier })
// -> { update(records, dt, { t, wl, beach, swell, latencyFrames }) -> record of 13.3, group: THREE.Group, active, splats, setWaterLevel(wl), setWind(vec), dispose() }
// src/breakers/tracker.js
export function linkColumns(records, lay, K)          // -> [{ cols: [c0, c1], samples: [{ col, Jc, etaC, hC, FC, etaT, Jt, sF, uC }] }]
export function trackLines(tracks, lines, dt, lay)    // -> updated tracks (ids, c estimates, onset detection)
// src/breakers/lip.js
export function classify({ H_b, sF, tanB, T }) -> { xi, lambda, type }
export function kinematics({ H_b, c, lambda, tanB, wl, etaC0, etaT0 }) -> { H_t, w0, tauImp, cRoot, rV, W_t, V_h, X_p, X_rel, uRel, tauP, zApex, zFloor }
export function sheetPoint(kin, l, root, e, out)      // p(l) of 11.4
export function stageAt(kin, tau) -> { stage, phase, on }
// src/breakers/mesh.js
export function createSheets(MAXL, MAXS, K) -> { geometry, update(tracks, t, kinOf) }
// src/breakers/spray.js
export function createSpray(n, seed) -> { points, emit(pos, vel, count, spread), update(dt, wind), dispose() }
// src/breakers/strip.js
export function buildCrestStrip(tracks, lay, out) -> DataTexture  ;  export function buildMaskQuads(tracks, lay, out) -> Float32Array

// src/render.js additions
R.setEngine(sol | null); R.setLipMask(quads); R.setCrestStrip(tex); R.addBreakers(group); R.setWetness(tex);
R.frame({ dt, alpha, splats }); R.setView('barrel', info); R.debug = null | 'F' | 'chi' | 'foam' | 'mask' | 'current';
```

### 14.3 Shader pass contract (for the pass-twin harness)

Each `src/engine/shaders/<name>.glsl.js` exports `{ name, inputs: [texture
names], outputs: [channel lists], source }`, and `src/boussinesq.js` exports
`passes.<name>(state, scratch)` computing the same outputs on CPU arrays for
the same inputs, so `test/pass-twins.mjs` can run one pass on both sides from a
random state and compare (section 15.2, V14).

---

## 15. Validation battery

### 15.1 Node (`node --test test/*.test.mjs`, oracle, each < 30 s)

| # | Test | Setup | Expected |
|---|---|---|---|
| V1 | Lake at rest, well-balancing | 48 x 40 at 2.5 m, sloping bed with island, 200 steps of 0.05 s | fully wet: `max|P|,|Q| < 1e-9`; with shoreline: `max|u| < 0.02 m/s`, `max|deta| < 0.01 m`, no NaN |
| V2 | Closed basin, mass and positivity | 60 x 50 at 2 m, Gaussian hump 0.8 m, 400 steps at `cfl(0.4)` | volume relative drift `< 1e-9`; `min h >= 0`; no NaN |
| V3 | Ritter dam break (dry bed) | 1D 1000 cells, `h0 = 2 m`, NSWE | front position `2 sqrt(g h0) t` within 5% at `t = 10 s`; `h` L2 error `< 3%` |
| V4 | Stoker dam break (wet bed) | 1D 1000 cells, `h_l = 10, h_r = 1`, `t = 10 s` | intermediate depth `h_m = 3.96 m`, shock speed `9.8 m/s` (from the exact implicit solution computed in the test) within 2%; L2 `< 3%` |
| V5 | Linear dispersion and amplitude stability | periodic 1D, `d = 10 m`, `a = 0.01 m`, 25 cells per wavelength, `kd = 0.5, 1, 2, 3` | phase speed vs `c^2 = g d (1 + Bq k^2 d^2)/(1 + (Bq + 1/3) k^2 d^2)` within 1% (c = 8.645 m/s at kd = 1, 5.84 at kd = 3; Airy 8.644, 5.70); amplitude change over 20 periods `< 1%` (RK2 acceptance; else switch to RK3) |
| V6 | Solitary wave | `a/d = 0.2`, `d = 1 m`, flat, 50 depths | amplitude within 5% after the initial adjustment; trailing noise `< 2%` of `a` |
| V7 | Synolakis 1987 run-up, 1:19.85 | `H/d = 0.0185` (non-breaking) and `0.28` (breaking), dx = d/20 | `R/d = 2.831 sqrt(19.85) (0.0185)^1.25 = 0.0860` within 10%; breaking case within 25% of 0.55 (lab value to be confirmed from the paper before the bound is tightened) |
| V8 | Weggel 1972 breaker index (calibrates 7.2) | plane beaches 1:80, 1:30, 1:12; `H0 in {1, 2, 3} m`, `T in {8, 12, 16} s`; at lab-like dx = 0.5 m and at 2.5 m | `gamma_b = H_b/h_b` vs `b - a H_b/(g T^2)`, `a = 43.75 (1 - e^(-19 m))`, `b = 1.56/(1 + e^(-19.5 m))`: expected 0.86 (1:80), 1.00 (1:30), 1.25 (1:12) for `H_b = 2 m, T = 12 s`; within +-20%; first-flag position `h_b` within 15% of the depth where the measured `H/h` reaches `gamma_b` |
| V9 | Ting & Kirby 1994 (calibrates 7.2 at lab scale) | 1:35 slope, `d = 0.4 m`, dx = 0.05 m; spilling `T = 2.0 s, H0 = 0.125 m`; plunging `T = 5.0 s, H0 = 0.128 m` | break point `x_b` within 10% of the paper's value (spilling `h_b ~ 0.20 m`, plunging `h_b ~ 0.16 m`; confirm from the paper's table before asserting); surf-zone `H(x)` decay within 15% |
| V10 | Surf-zone saturation and set-up | 1:30, `H0 = 1.5 m, T = 10 s`, 60 s average after spin-up | `H_rms / h in [0.35, 0.7]` from `0.5 h_b` shoreward (Battjes & Janssen 1978 band); shoreline set-up in `[0.1, 0.25] H_b`; swash excursion `> 0` |
| V11 | Shoaling and asymmetry | 1:50, `T = 10 s, H0 = 1 m` | `H(x)` within 5% of energy-flux shoaling until `H/h = 0.5`; crest `eta_c/H > 0.6` in the last 2 depths before breaking; first flag where `H/h in [0.7, 0.95]` |
| V12 | Longshore current and rip | plane 1:30, 20 deg, `H0 = 1.5 m, T = 10 s`, 60 s average; crescent preset | `V_max in [0.3, 1.2] m/s` with the Longuet-Higgins 1970 sign, peak inside the surf zone; crescent: seaward mean current `> 0.1 m/s` in a channel |
| V13 | Relaxation zones | 1D, 300 m zone, 12 s 1 m target; then an outgoing pulse | generated amplitude within 3% of the target; reflection coefficient `< 5%` |
| V15 | Thomas vs PCR coefficient path | random diagonally dominant systems, n = 377 and 754, the JS PCR twin | agreement `1e-9` relative |
| V16 | Live edits | default scene, a 2 m bump sculpted mid-run; tide +-1.5 m steps | no NaN; `min h >= 0`; volume bounded; mean `eta` in the box within 0.3 m |
| V17 | Lip unit tests | `H_b = 2, c = 6.6, lambda = 1, 1:20` and the table of 11.3; synthetic crest records (sinusoidal line advancing at `c`, `FC` switching on at a row; a merge; a split) | `tau_imp = 0.698 s`, `X_p/H_b in [2.8, 4.2]` for `H_b in {1, 2, 3}`; `r_v` monotone decreasing with slope and clamped; breaker spawns at the right column/time; tip at `p(tau_imp)` within 1e-6; peel phase monotone along a tilted onset; ids kept through merge/split; splats at impact; track dropped 0.5 s after records stop; mesh finite with unit normals; stages in order |

### 15.2 Headless Chromium (Playwright + swiftshader, `test/headless.mjs`, tier S grids with `nx, ny <= 128`, each < 2 min)

| # | Test | Expected |
|---|---|---|
| V14 | GPU vs oracle | identical initial states loaded on both; solitary wave 200 steps: `max|dh| < 1e-4 m` before breaking; plane beach with target 300 steps: L2 of `h` difference `< 2%`, `F` equal up to the dilation ramp; lake with island: `1e-5` |
| V14b | Pass twins | each shader pass on a random state vs its CPU twin: `max rel diff < 1e-5` (coefficient/bed pass included) |
| V18 | Capabilities and errors | `EXT_color_buffer_float` present, every shader compiles, `gl.getError() == NO_ERROR` after a step, MRT with 3 attachments works, PBO readback returns the same bytes as `readRenderTargetPixels` |
| V19 | Soak | default scene (crescent, 1.8 m 13 s), 600 steps: no NaN (`nanCount = 0`), `max|eta| < 3 H0`, mean `eta` in the box within 0.3 m of 0, `min h >= 0` |
| V20 | Screens and barrels | reef preset, 2.5 m 14 s swell: `brk.stats.plungingMetres > 0` within 60 s simulated; Impact and Beach screenshots at fixed phase (seeded): foam-white pixel fraction in the surf band `> 3%`, lip-mask pixels `> 0` while a barrel is active, perceptual-hash diff vs golden PNGs `< 2%` |
| V21 | Tiering and timing | tier selector picks L when the extension is faked off and C when the step time is faked high; swiftshader step time at `128^2` logged (`< 400 ms`, not asserted) |

Runtime budget: Node suite `< 2 min`; headless suite `3-5 min` (a 384^2 step
is ~0.3 s in swiftshader, so grids stay small and simulated times short).

---

## 16. Work plan

Three engineers in parallel from this text; `params.js` and `layout.js` are
written first (B, two hours) and frozen.

- **A - GPU solver** (day 1: bed, flux, rhs, PCR, finalize, chi passes and the
  pass-twin page; day 2: stats, crest scan, readback ring, tiering probe, hot
  start, NaN guard, V14/V14b/V18/V19). Develops against the oracle's
  `readState/loadState`.
- **B - oracle, targets, tests, app** (day 1: oracle additions of section 10,
  `targets.js`, rectangular `waves.js`, worker, V1-V13, V15-V16; day 2:
  `app.js` scheduler, tier selection, probe/verdict from STATS, tier C worker
  path, README).
- **C - breakers and renderer** (day 1: tracker, lip kinematics, sheet mesh,
  strip/mask builders, V17 against synthetic records; day 2: material, spray,
  splash-up, LIPMASK/PRESENT/FOAM passes in `render.js`, Barrel view, V20).

Cut list if time runs out, in order: 1 spray; 2 splash-up sheet (keep the
foam burst); 3 lateral mean-field term (plain linear target); 4 lagged cross
terms `M` -> 0; 5 tier P/C (desktop only, with a clear message); 6 async
readback -> synchronous; 7 Mead-Black ratio -> constant 2.0; 8 Barrel view.
Never cut: the oracle, V1-V5, V8, V14, V17's plunge-distance check.

---

## 17. Open risks

- R1 Breaking onset at 2.5 m: the central-upwind limiter dissipates the front
  before the flag unless the capped `gamma_b` and the slope criterion are
  calibrated (V8, V9); until then break positions can be 10-20% too far
  inshore and lips spawn late.
- R2 Hybrid switching oscillations at the flag edge (Kazolea 2014) and
  explicit third derivatives near partially flagged fronts; mitigations are the
  7.5 m ramp, hysteresis via the 2 s memory, the slope clamp and chiBed; the
  Kennedy eddy-viscosity term is the fallback.
- R3 RK2 with centred dispersive terms is only marginally stable on the
  imaginary axis; V5 decides between RK2 and RK3 (1.5x cost).
- R4 Coefficient or sign slip in the mild-slope cross terms (1/3, 1/6, 2Bq,
  Bq); one verification against Kazolea 2012 / Tavakkol & Lynett 2017 is
  required before acceptance.
- R5 iGPU cost: 27 passes x 24 steps/s at 141k texels is 4-7 ms per step;
  4x playback slows the clock; tier H is discrete-GPU only.
- R6 Relaxation to a linear target: free second harmonics, infragravity
  reflection, mismatch ripples at the lateral inner edge; the mean-field term
  could drift if the STATS mean itself is polluted (bounded by m_lat and V19's
  0.3 m mean-eta check).
- R7 Lip/bore timing: the solver bore exists from onset while the lip flies
  ~0.7 s; the mask and push hide it but the look needs a day of tuning; the
  readback latency dead-reckoning assumes a steady `c`.
- R8 Raw-GL readback beside three.js: PBO binding must be cleared explicitly;
  Safari/iOS float-target and `fenceSync` behaviour untested here (tier L and
  the sync fallback cover it).
- R9 PCR float32 round-off on 754-element rows at tier H (V15 covers the JS
  twin; V14 bounds the GPU).
- R10 `WebGLMultipleRenderTargets` with 3 attachments and `RawShaderMaterial`
  in r160 on Safari; swiftshader reports 6 draw buffers.
- R11 Mead & Black vortex ratio form (gradient as 1:x) and the Synolakis
  breaking run-up (0.55) and Ting & Kirby break points are quoted from memory;
  verify before the bounds are tightened.
- R12 Swiftshader tests are slow (0.3 s per 384^2 step); grids must stay at
  <= 128^2 or the CI budget breaks.
- R13 Sculpting rock faces with bed slopes > 1 stresses the dispersive terms;
  chiBed masks |grad d| > 0.6 but ringing is possible at the mask edge.
