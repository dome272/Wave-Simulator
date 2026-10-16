# Wave Simulator

**Shorebreak Lab** is a browser app for building a beach in 3D and watching a swell break on it.

- Set the incoming swell: height, period, direction, and how far away the storm was. Height is the typical wave as it arrives offshore; waves come in sets, and the set waves run about a third bigger.
- Load a real California beach (Davenport Landing, Lower Trestles) cut from surveyed seabed data, and set the swell by compass bearing as buoys report it.
- Build a beach from a starting design (crescent sandbars, a long bar, a point, a reef peak, a rocky cove, or a blank slope), with adjustable steepness, bars, rocks and randomness.
- Reshape it with sand tools (pile sand, dig, smooth, flatten, lay rock) and watch how the breaking changes.
- Watch from the overview, from the lineup just outside the break, from the impact zone facing the waves, or standing on the sand. The eye-level views stand where the breaking is heaviest, so on a barred beach they look at a bar rather than down a rip channel.
- Change the tide, play back faster or slower, and hover anywhere to see the depth, local wave height and whether waves are breaking there.

## Running it

It is a static page with no build step. Serve the folder and open `index.html`:

```
npx http-server -c-1 .
```

three.js is loaded from jsDelivr through the import map in `index.html`.

Tests: `node test/boussinesq.test.mjs` (reference solver physics), `node test/breakers.test.mjs` (crest tracker and lip model), and with the page served on port 8099 and Playwright's Chromium available, `node test/gpu-equivalence.mjs` (GPU solver against the reference), `node test/lip-compile.mjs` (the curtain, depth pre-pass and spray shaders compile and draw) and `node test/breakers-demo.mjs` (eye-level renders of a synthetic breaking wave).

## How the waves are computed

There are two wave engines. The **real-fluid engine** (default where the GPU allows it) solves the water motion itself; the **classic engine** is the original linear wave model and remains the fallback and the forecast that feeds the verdict and the wave maker.

### Real-fluid engine

The water is simulated with the Madsen–Sørensen (1992) extended Boussinesq equations, the depth-integrated nonlinear equations used by surf-zone research models (the same family as FUNWAVE and Celeris). They carry wave height, period and the full depth-integrated flow, so waves shoal, steepen, refract, break into bores and run up the sand because the seabed makes them.

- `src/boussinesq.js` is the reference solver in double precision (Node). It discretises the equations in conservative form with Kurganov–Petrova central-upwind finite volumes and Audusse hydrostatic reconstruction (exact still water over any bed, wet and dry cells), the dispersive terms with central differences and an implicit tridiagonal operator per row and column, SSP Runge–Kutta time stepping, Manning friction, and hybrid breaking: where a front gets too steep (Kennedy et al. 2000; Tonelli & Petti 2009) dispersion is switched off and the shock-capturing scheme forms a bore with the right energy loss. `test/boussinesq.test.mjs` checks it against known solutions: lake at rest over an island, volume conservation, the dispersion relation, a solitary wave, breaking and run-up on a plane beach, the Ritter dam break and Snell's law refraction.
- `src/engine/` is the same algorithm on the GPU (WebGL2, float render targets, three.js passes; tridiagonal systems by parallel cyclic reduction). It runs on a grid of cells between the terrain nodes with margins outside the visible beach: 300 m offshore and 100 m on each side hold relaxation zones that generate the swell and absorb outgoing waves. The wave maker's target is the classic engine's refracted field, so the same swell controls, sets and storm distance apply. Two resolutions: 2.5 m cells (one per terrain cell) and 1.25 m (four per cell), chosen automatically from a timing probe. `test/gpu-equivalence.mjs` checks the GPU port against the reference solver.
- A depth-integrated surface cannot overturn, so pitching lips are a separate model (`src/breakers/`). A GPU pass finds every wave crest in every column; the CPU links them into crest lines, measures height, depth, speed and the local Iribarren number, and when a plunging crest starts to break it throws a ballistic lip from the crest line: a curtain of water that falls under gravity onto the trough ahead, leaving a hollow tube whose shape follows the Iribarren number (Mead & Black 2001). Impact kicks up splash and spray and seeds whitewater, then the solver's bore takes over. Under the tube the solver's surface is pulled down to the trough so the lip forms the roof. The same crest scan gives the engine its break line (per column, the most offshore crest that is breaking), and the eye-level views stand by the stretch of it where the breaking is heaviest.

### Classic engine

`src/waves.js` solves for the wave field on the seabed grid (runs in a Web Worker, `src/solver-worker.js`):

1. The swell is split into 8 components spread in frequency and direction. A distant storm gives a narrow spread (clean lines, long gaps between sets); a close storm gives a wide one (choppy, short-crested sea).
2. For each component, the local wavenumber comes from the linear dispersion relation ω² = g k tanh(k h), and the wave phase solves the eikonal equation |∇S| = k (fast sweeping). This gives refraction: crests bend toward shallow water, and long-period swells bend more because they feel the bottom earlier.
3. Amplitudes follow conservation of energy flux along rays (shoaling from group velocity, plus ray convergence and spreading), with some lateral spreading as a stand-in for diffraction.
4. The typical wave is marched shoreward with a depth limit (H ≤ 0.68 h). Energy lost where it breaks stays lost, so a sandbar shelters the inside. Sets are modelled as a slow modulation of the wave height travelling at the group speed, so set waves break further out than the typical wave, and the verdict quotes both distances.
5. The breaker type comes from the surf similarity (Iribarren) number: the local seabed slope against the offshore wave steepness. Gentle slopes spill, steeper ones plunge, very steep ones surge.

On the GPU (`src/render.js`) the components are summed every frame. Each wave's instantaneous height is compared with the local depth: as the ratio climbs the front face steepens and the crest sharpens, at the breaking point plunging waves throw their lip forward, and broken waves continue as whitewater bores. A foam texture is advected with the waves so whitewater trails and lingers. The scene has its own sky and an afternoon sun behind the beach, so wave faces are lit as seen from the shore; from high up the water normals are exaggerated so long, low swells still read as lines, while the eye-level views use the true geometry.

This is a phase-resolved linear model with breaking rules on top. It is fast enough to recompute while you sculpt, and in the real-fluid engine it is what the wave maker drives the Boussinesq solver with.

## Seabed format

The beach is a 256 × 256 heightfield at 2.5 m spacing (637.5 m square): bed elevation in metres relative to the still water level, plus a rock mask. Row 0 is the offshore edge and the last row is inland. Real bathymetry can be loaded into the same grid: the California cutter (16-bit PNG plus JSON sidecar with `cell_m`, `z_min_m`, `z_max_m`, north-up) needs resampling to 2.5 m, rotating so the ocean is at row 0, and shifting from NAVD88 to mean sea level.

## Real beaches

`beaches/<id>.json` are real seabeds in the saved-beach format, plus a `real` block (name, the compass bearing the beach faces, the typical swell it loads with, sources). `tools/cut_beach.py <id>` makes them: it reads the USGS CoNED topobathymetric model and the USACE NCMP 2014 California topobathy lidar from NOAA's public S3 mirror (both public domain), keeps the lidar where it has data and blends its difference from CoNED back to zero across its gaps, resamples to the 2.5 m grid rotated so the beach faces row 0, shifts NAVD88 to mean sea level and estimates a rock mask from bed roughness and the reef shelves.

The 2014 lidar matters on rocky coasts: at Davenport Landing CoNED has no survey between the waterline and about 10 m depth and only interpolates a smooth slope there, while the lidar shows the reef shelves and the channel. Steamer Lane is left out because neither source surveyed its reef.

On a real beach the swell direction slider is still degrees from straight in (+ from the left as seen from the sand); the readout adds the compass bearing it comes from: bearing = facing − direction. Water level is relative to mean sea level; a tide-table height (feet above MLLW) × 0.3 − 0.9 gives it in metres on these beaches.
