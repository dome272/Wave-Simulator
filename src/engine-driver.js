// The real-fluid engine: drives the GPU Boussinesq solver, reads the crest
// scan back every frame and feeds the breakers (plunging lips, spray, foam
// splats). Owns the simulation clock: the solver advances in fixed steps and
// the clock follows wall time while the GPU keeps up; when it cannot, time
// slows down instead of the frame rate collapsing.
import { G } from './waves.js';

// Loads the engine modules; rejects where they are missing or the browser
// cannot run them, so the app can fall back to the classic pipeline.
export async function loadEngine() {
  const [gpu, lay, brk] = await Promise.all([
    import('./engine/gpu-solver.js'),
    import('./engine/layout.js'),
    import('./breakers/index.js'),
  ]);
  return { createGpuSolver: gpu.createGpuSolver, layout: lay.layout, createBreakers: brk.createBreakers };
}

const swellKey = (s) => `${s.height}|${s.period}|${s.direction}|${s.distanceKm}`;

export function createEngine(renderer, mods, { r = 1, maxStepsPerFrame = 8 } = {}) {
  const lay = mods.layout(r);
  const solver = mods.createGpuSolver(renderer, { r });
  let breakers = null;
  let beach = null, wl = 0, swell = null, targets = null, swellSig = '';
  let t = 0;        // simulation time (s)
  let acc = 0;      // simulated time not yet stepped
  let dt = 0.03;    // solver time step (s)
  let needHot = true;
  const stats = { msStep: 0, stepsPerFrame: 0, overruns: 0, frames: 0, slow: 0 };
  let speedCheck = 0;

  function retime() {
    if (!beach) return;
    let dmax = 0;
    const z = beach.z;
    for (let c = 0; c < z.length; c++) { const d = wl - z[c]; if (d > dmax) dmax = d; }
    const H = swell ? swell.height * 1.6 : 3;
    const amax = Math.sqrt(G * (dmax + H)) + 2;
    dt = 0.25 * lay.dxs / amax;
  }

  function setBeach(b, waterLevel) {
    beach = b; wl = waterLevel;
    solver.setBathymetry(b, wl);
    if (breakers) breakers.setWaterLevel(wl);
    retime();
  }
  function setWaterLevel(waterLevel) {
    wl = waterLevel;
    solver.setWaterLevel(wl);
    if (breakers) breakers.setWaterLevel(wl);
    retime();
  }
  function setTargets(res, sw) {
    targets = res;
    const sig = swellKey(sw);
    if (sig !== swellSig) { needHot = true; swellSig = sig; }
    swell = { ...sw };
    solver.setTargets(res);
    retime();
  }
  function ensureBreakers() {
    if (!breakers && beach) breakers = mods.createBreakers({ layout: lay, wl, beach });
    return breakers;
  }

  // Advance the simulation by dtSim seconds of simulated time.
  function advance(dtSim, ctx = null) {
    if (!targets || !beach) return 0;
    if (needHot) { solver.hotStart(t); needHot = false; acc = 0; }
    acc += dtSim;
    let n = Math.floor(acc / dt);
    if (n > maxStepsPerFrame) { n = maxStepsPerFrame; acc = n * dt; stats.overruns++; }
    stats.frames++;
    const t0 = performance.now();
    for (let k = 0; k < n; k++) { solver.step(dt, t); t += dt; }
    acc -= n * dt;
    if (n > 0) {
      stats.msStep = 0.9 * stats.msStep + 0.1 * (performance.now() - t0) / n;
      stats.stepsPerFrame = n;
      const brk = ensureBreakers();
      if (brk) brk.update(solver.readCrests(), dtSim, { t, wl, beach, swell, ...(ctx || {}) });
      // Every few seconds make sure the time step still respects the fastest wave.
      if ((speedCheck += dtSim) > 3) {
        speedCheck = 0;
        const s = solver.maxSpeed();
        if (s * dt / lay.dxs > 0.4) { dt *= 0.5; stats.slow++; }
      }
    }
    return n;
  }

  // Time one solver step (ms), forcing the GPU to finish. Used once at start
  // to pick the resolution tier.
  function probe(steps = 12) {
    if (needHot) { solver.hotStart(t); needHot = false; }
    solver.step(dt, t); solver.maxSpeed();
    const t0 = performance.now();
    for (let k = 0; k < steps; k++) solver.step(dt, t);
    solver.maxSpeed();
    return (performance.now() - t0) / steps;
  }

  function reset() { needHot = true; }

  return {
    r, layout: lay, solver, stats,
    get time() { return t; },
    get dt() { return dt; },
    get breakers() { return breakers; },
    get group() { return ensureBreakers()?.group || null; },
    get splats() { return breakers ? breakers.splats : []; },
    get textures() { return solver.textures; },
    setBeach, setWaterLevel, setTargets, advance, probe, reset,
    dispose() { solver.dispose(); if (breakers) breakers.dispose(); },
  };
}
