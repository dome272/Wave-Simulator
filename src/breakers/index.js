// Breakers: crest tracker + plunging-lip model + curtain/spray meshes
// (docs/engine-contract.md section 6). Public API:
//
//   import { createBreakers } from './breakers/index.js';
//   const brk = createBreakers({ layout, wl, beach, ...options });
//   brk.update(records, dt, { t, wl, beach, swell, camera, viewportHeight })
//                 records = Float32Array from readCrests(); null = a dropped scan frame (treated as an empty scan: the crests age)
//   brk.group     THREE.Group: lip curtains + splash-up sheets (one mesh plus its depth pre-pass copy) and the spray points
//   brk.splats    [{ x, y, r, s }] foam to inject this frame: x = world x, y = world z (m), r radius (m), s strength 0..1
//   brk.tubes     [{ x, y, dx, dy, a0, a1, hw, eta, s, H, hold, h0, h1 }] the barrel voids this frame, one per lip sample: the present pass
//                 should lower the water surface to wl + eta (blended by s, 0..1) inside the strip that starts a0 m ahead of the
//                 crest root (x, y: world x/z) along the unit direction (dx, dy), ends a1 m ahead, and is 2 hw wide. Without
//                 this the solver's 2-3-cell front face fills the tube: the curtain is w = H (0.65..0.95) wide and H high
//                 (contract s6 item 4), far steeper than a bore face, so the sheet would be buried in the face below its root.
//                 hold = the lip's alpha while it is in the air: the renderer keeps whitewater off the band [h0, h1] (along the
//                 normal, from behind the crest to past the landing point) until the impact.
//   brk.active    live breakers (lips), a fresh snapshot per frame: { id, crestId, state, t0, n, impact, H, xi, c, tau,
//                 tubeWidth, throw (= tubeWidth: the tip lands that far ahead of the root), tipSpeed (world, m/s), launchZ,
//                 impactZ, x0, x1, points }  state: 'throwing' | 'splashup' | 'collapsed';
//                 points: [{ x, y (world z), etaC, hC, c, H, xi, w, R, tau, u, phase, p, s, tipX, tipY, tipZ, rootZ, ... }]
//   brk.crests    tracked crests: { id, state: 'tracking' | 'breaking', n, points: [{ x, y (world z), etaC, hC, c, H, xi, FC, type }], H, xi, type }
//   brk.setWaterLevel(wl)
//   brk.dispose()
//
// The objects in brk.active / brk.crests are snapshots of the frame (built once per frame; `points` on first read);
// the emitter and tube objects are pooled and reused every frame, so consumers copy what they keep.
//
// Options (defaults from the contract; see DEFAULTS in lip-model.js and SPRAY_DEFAULTS in spray.js):
//   tubeScale 1 (x the contract tube width), tubeBack 0.1, tubeRamp 0.15 s, lipLift 0, tipThickness 0.15, rootThickness 0.09,
//   spawnFC 0.5, armWindow 0.4 s, minStretch 4 m, minH 0.3 m, xiSpill 0.4, xiSurge 2.0, splashTime 0.5 s, splashHeight 0.8,
//   fadeTime 0.4 s, maxBreakers 24, maxSamples 512, rearm 1.0 s, rearmFC 0.3, splatRadius 1.0, endTaper 3, period 10 s (swell period fallback),
//   tracker: maxCrests 32, linkCells 3, dropAfter 0.5 s, fitWindow 1.0 s, minLineCols 2, matchDist 3*dxs+0.5 m,
//   mesh: meshSlots 4096, sun [x,y,z], renderOrder 4 (depth pre-pass 3.5, spray 5); spray: maxParticles 8000, wind [0.4,0,-1.2],
//   gravity 9.81, drag 0.9, pointScale 700.
import * as THREE from 'three';
import { createTracker, K, CH } from './crest-tracker.js';
import { createLipModel, DEFAULTS, tubeWidth } from './lip-model.js';
import { createLipMesh } from './lip-mesh.js';
import { createSpray, SPRAY_DEFAULTS } from './spray.js';

export { createTracker, createLipModel, createLipMesh, createSpray, DEFAULTS, SPRAY_DEFAULTS, tubeWidth };

export function createBreakers({ layout, wl = 0, beach = null, swell = null, ...opts } = {}) {
  if (!layout) throw new Error('createBreakers: layout is required ({ r, dxs, V, ML, MO, MI, nx, ny })');
  const tracker = createTracker(layout, opts);
  const model = createLipModel(layout, { ...opts, wl });
  const mesh = createLipMesh(opts);
  const spray = createSpray({ ...opts, renderOrder: (opts.renderOrder ?? 4) + 1 });
  model.onRelease = (b) => mesh.freeBreaker(b);

  const group = new THREE.Group();
  group.name = 'breakers';
  group.add(mesh.depthMesh, mesh.mesh, spray.points);

  let t = 0, frame = 0;
  const EXT2 = model.options.gridExt / 2;
  let emptyRecords = null; // an all-empty scan, for update(null)

  const api = {
    group, splats: [], tubes: [], active: [], crests: [],
    tracker, model, mesh, spray, layout,
    options: model.options,
    update, setWaterLevel, dispose,
    get wl() { return wl; },
    get time() { return t; },
  };

  function crestSummaries() {
    const out = [];
    for (const cr of tracker.crests) {
      let Hmax = 0, xiSum = 0, m = 0, breaking = 0, typeCount = [0, 0, 0];
      for (let col = cr.c0; col <= cr.c1; col++) {
        if (cr.age[col] < 0) continue;
        if (cr.H[col] > Hmax) Hmax = cr.H[col];
        xiSum += cr.xi[col]; m++;
        if (cr.FC[col] >= 0.5) breaking++;
        typeCount[cr.type[col]]++;
      }
      const type = typeCount[1] >= typeCount[0] && typeCount[1] >= typeCount[2] ? 'plunging' : typeCount[2] > typeCount[0] ? 'surging' : 'spilling';
      const self = {
        id: cr.id, state: breaking > 0 ? 'breaking' : 'tracking', n: cr.n, c0: cr.c0, c1: cr.c1, t0: cr.t0,
        H: Hmax, xi: m ? xiSum / m : 0, type, breakingColumns: breaking,
        get points() {
          if (!self._pts) {
            const pts = [];
            for (let col = cr.c0; col <= cr.c1; col++) {
              if (cr.age[col] < 0) continue;
              pts.push({ col, x: (col + 0.5) * layout.dxs - EXT2, y: cr.y[col] - EXT2, etaC: cr.etaC[col], hC: cr.hC[col], etaT: cr.etaT[col], c: cr.c[col], H: cr.H[col], xi: cr.xi[col], tanB: cr.tanB[col], FC: cr.FC[col], type: cr.type[col], broken: cr.broken[col], nx: cr.nx[col], ny: cr.ny[col], interp: cr.interp[col] });
            }
            self._pts = pts;
          }
          return self._pts;
        },
      };
      out.push(self);
    }
    return out;
  }

  function update(records, dt, ctx = {}) {
    frame++;
    t = ctx.t != null ? ctx.t : t + dt;
    if (ctx.wl != null && ctx.wl !== wl) setWaterLevel(ctx.wl);
    if (ctx.beach) beach = ctx.beach;
    if (ctx.swell) swell = ctx.swell;
    if (!records) {
      if (!emptyRecords) { emptyRecords = new Float32Array(layout.V * K * CH); for (let i = 0; i < emptyRecords.length; i += CH) emptyRecords[i] = -1; }
      records = emptyRecords;
    }
    tracker.update(records, t, dt);
    model.classify(tracker.crests, { beach, swell, wl });
    const out = model.update(tracker.crests, t, dt);
    api.splats = out.splats;
    api.tubes = out.tubes;
    api.active = out.active;
    api.crests = crestSummaries();
    mesh.update(model, model.breakers, { wl, t });
    spray.emit(out.emitters, dt);
    spray.step(dt);
    if (ctx.camera && ctx.viewportHeight) spray.setCamera(ctx.camera, ctx.viewportHeight);
    return api;
  }

  function setWaterLevel(v) { tracker.shiftElevations(wl - v); wl = v; model.setWaterLevel(v); mesh.material.uniforms.uWl.value = v; }

  function dispose() {
    group.remove(mesh.depthMesh, mesh.mesh, spray.points);
    mesh.dispose();
    spray.dispose();
  }

  return api;
}
