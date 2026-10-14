# Wave Simulator

**Shorebreak Lab** is a browser app for building a beach in 3D and watching a swell break on it.

- Set the incoming swell: height, period, direction, and how far away the storm was. Height is the typical wave as it arrives offshore; waves come in sets, and the set waves run about a third bigger.
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

## How the waves are computed

`src/waves.js` solves for the wave field on the seabed grid (runs in a Web Worker, `src/solver-worker.js`):

1. The swell is split into 8 components spread in frequency and direction. A distant storm gives a narrow spread (clean lines, long gaps between sets); a close storm gives a wide one (choppy, short-crested sea).
2. For each component, the local wavenumber comes from the linear dispersion relation ω² = g k tanh(k h), and the wave phase solves the eikonal equation |∇S| = k (fast sweeping). This gives refraction: crests bend toward shallow water, and long-period swells bend more because they feel the bottom earlier.
3. Amplitudes follow conservation of energy flux along rays (shoaling from group velocity, plus ray convergence and spreading), with some lateral spreading as a stand-in for diffraction.
4. The typical wave is marched shoreward with a depth limit (H ≤ 0.68 h). Energy lost where it breaks stays lost, so a sandbar shelters the inside. Sets are modelled as a slow modulation of the wave height travelling at the group speed, so set waves break further out than the typical wave, and the verdict quotes both distances.
5. The breaker type comes from the surf similarity (Iribarren) number: the local seabed slope against the offshore wave steepness. Gentle slopes spill, steeper ones plunge, very steep ones surge.

On the GPU (`src/render.js`) the components are summed every frame. Each wave's instantaneous height is compared with the local depth: as the ratio climbs the front face steepens and the crest sharpens, at the breaking point plunging waves throw their lip forward, and broken waves continue as whitewater bores. A foam texture is advected with the waves so whitewater trails and lingers. The scene has its own sky and an afternoon sun behind the beach, so wave faces are lit as seen from the shore; from high up the water normals are exaggerated so long, low swells still read as lines, while the eye-level views use the true geometry.

This is a phase-resolved linear model with breaking rules on top, not a full Boussinesq or Navier-Stokes solver. It is fast enough to recompute while you sculpt, and height, period and direction each change where and how waves break.

## Seabed format

The beach is a 256 × 256 heightfield at 2.5 m spacing (637.5 m square): bed elevation in metres relative to the still water level, plus a rock mask. Row 0 is the offshore edge and the last row is inland. Real bathymetry can be loaded into the same grid: the California cutter (16-bit PNG plus JSON sidecar with `cell_m`, `z_min_m`, `z_max_m`, north-up) needs resampling to 2.5 m, rotating so the ocean is at row 0, and shifting from NAVD88 to mean sea level.
