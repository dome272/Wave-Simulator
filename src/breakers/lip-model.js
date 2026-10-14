// Kinematic plunging-lip / barrel model (contract section 6, items 2-6).
// Pure JS, no three.js: it classifies the tracked crests (wave height, phase
// speed, bed slope, Iribarren number, breaker type), spawns lips on plunging
// stretches whose breaking field has just switched on, advances them
// (ballistic curtain, impact, splash-up, collapse), and produces per frame the
// foam splats, the spray emitters and the tube descriptors (the void the
// renderer should carve under the curtain). The mesh layer asks it for the
// curtain and splash-up profiles (lipProfile / splashProfile) in the vertical
// plane of the crest normal: `a` metres ahead of the crest root, `y` world
// elevation.
//
// Geometry (per crest sample, frozen at launch). The contract fixes the tube:
// "an elliptical void with height about H and width
// w = H (0.65 + 0.3 clamp((xi - 0.4) / 1.6))" between the curtain and the
// face, and the curtain is the family of trajectories launched from the crest
// top between the launch and now. The lip tip leaves the crest top, falls
// freely and meets the trough level etaT after tau = sqrt(2 (etaC - etaT) / g)
// (lipLift = 0); its horizontal speed RELATIVE TO THE CREST ROOT is u = w / tau,
// so that it lands exactly w ahead of the root and the void under the sheet is
// w wide and H high. In the crest frame all parcels lie on the one parabola
// a = w t'/tau, y = etaC - H (t'/tau)^2, t' in [0, s]. The root keeps
// advancing with the tracked crest at c, so the tip's world speed is
// c + w / tau (about 1.3 c for a 2 m wave; measured plunging-jet tips move at
// 1.0-1.5 c) and it lands c tau + w from the launch point.
//
// Spawn rule (contract: "a plunging crest that turns on - FC rises above 0.5
// along a stretch of at least 4 m - spawns a lip"): edge-triggered. A column
// whose FC crosses spawnFC upward is "armed" for armWindow seconds; a stretch
// of armed plunging columns of at least minStretch metres throws a lip (or
// extends an adjacent lip of the same crest that is still throwing). A column
// first seen with FC already above spawnFC (a crest born broken, or one
// re-acquired after a record gap) is a bore: it never throws until it has been
// calm (FC < rearmFC) for `rearm` seconds. The breaker type of a column is
// frozen the moment its FC first rises (a wave that broke spilling stays
// spilling while its bore decays).
import { N, DX, EXT } from '../terrain.js';

export const G = 9.81;
export const NPROF = 12; // profile samples per crest sample

export const DEFAULTS = {
  lipLift: 0, // upward launch speed, x sqrt(2 g H). 0 = horizontal launch (tau = sqrt(2 dEta / g))
  tubeScale: 1.0, // multiplier on the contract tube width w = H (0.65 + 0.3 clamp((xi-0.4)/1.6))
  tubeBack: 0.1, // x w: where the tube's back wall meets the floor (the crest overhangs the face by this)
  tubeRamp: 0.15, // s over which the tube carve blends in after launch
  tipThickness: 0.2, // thickness of the bulbous leading edge, x H
  rootThickness: 0.1, // thickness where the sheet leaves the crest, x H
  spawnFC: 0.5, // FC threshold that a column must cross upward to arm
  armWindow: 0.4, // s: an armed column must be taken by a lip within this, or it is a bore
  minStretch: 4, // m of armed crest needed to spawn a new lip
  minH: 0.3, // m, smaller waves get no lip
  xiSpill: 0.4, // xi below: spilling (no lip)
  xiSurge: 2.0, // xi at or above: collapsing/surging (no lip)
  splashTime: 0.5, // s, the splash-up rises and falls within this
  splashHeight: 0.8, // x H
  fadeTime: 0.4, // s, the lip mesh fades after the splash-up
  maxBreakers: 24,
  maxSamples: 512, // crest samples (columns) per breaker
  rearm: 1.0, // s of FC < rearmFC before a column may arm again (a reformed wave)
  rearmFC: 0.3,
  splatRadius: 1.0, // x H
  sprayRate: 1.0, // multiplier on the spray emission rates
  endTaper: 7.5, // m over which the ribbon (and its void) tapers to the crest at its ends (at least 3 samples)
  tanBFallback: 0.05, // bed slope when no beach is given
  period: 10, // s, swell period fallback
  gridSize: N, gridDX: DX, gridExt: EXT,
};

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function hash2(a, b) { let h = (a * 374761393 + b * 668265263) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967295; }
// Smooth alongshore noise (value noise over the column index): adjacent samples stay correlated.
function snoise(id, col, scale) { const u = col / scale, i = Math.floor(u), f = u - i, w = f * f * (3 - 2 * f); return hash2(id, i) * (1 - w) + hash2(id, i + 1) * w; }
function cnoise(id, col) { return 0.55 * snoise(id, col, 4) + 0.45 * snoise(id + 101, col, 11); }

export function tubeWidth(H, xi, scale = 1) { return scale * H * (0.65 + 0.3 * clamp((xi - 0.4) / 1.6, 0, 1)); }

export function createLipModel(lay, userOpts = {}) {
  const o = { ...DEFAULTS, ...userOpts };
  const { V, dxs } = lay;
  const S = o.maxSamples;
  const taperN = Math.max(3, Math.round(o.endTaper / dxs)); // samples
  let nextId = 1;
  let wl = userOpts.wl ?? 0;

  // --- bed slope along the propagation direction -------------------------------
  function bedAt(z, x, y) {
    const n = o.gridSize, dx = o.gridDX;
    const gx = clamp(x / dx, 0, n - 1.001), gy = clamp(y / dx, 0, n - 1.001);
    const i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j, c = j * n + i;
    return (z[c] * (1 - fx) + z[c + 1] * fx) * (1 - fy) + (z[c + n] * (1 - fx) + z[c + n + 1] * fx) * fy;
  }

  // Fill H, tanB, xi, type per column of each tracked crest. The type of a
  // column that has broken (FC crossed spawnFC, or born broken) stays frozen.
  function classify(crests, ctx) {
    const beach = ctx.beach, z = beach && beach.z;
    const T = (ctx.swell && ctx.swell.period) || o.period;
    const L0 = (G * T * T) / (2 * Math.PI);
    const ds = 2 * o.gridDX;
    for (const cr of crests) {
      for (let col = cr.c0; col <= cr.c1; col++) {
        if (cr.age[col] < 0) continue;
        const H = Math.max(cr.etaC[col] - cr.etaT[col], 0);
        cr.H[col] = H;
        let tanB = o.tanBFallback;
        if (z) {
          const x = (col + 0.5) * dxs, y = cr.y[col], nx = cr.nx[col], ny = cr.ny[col];
          tanB = (bedAt(z, x + nx * ds, y + ny * ds) - bedAt(z, x - nx * ds, y - ny * ds)) / (2 * ds);
        }
        cr.scratch[col] = tanB;
      }
      // smooth the slope alongshore (+-2 columns)
      for (let col = cr.c0; col <= cr.c1; col++) {
        if (cr.age[col] < 0) continue;
        let s = 0, m = 0;
        for (let q = -2; q <= 2; q++) { const cc = col + q; if (cc < cr.c0 || cc > cr.c1 || cr.age[cc] < 0) continue; s += cr.scratch[cc]; m++; }
        const tanB = s / m;
        cr.tanB[col] = tanB;
        const H = cr.H[col];
        const xi = tanB > 0 ? tanB / Math.sqrt(Math.max(H, 0.05) / L0) : 0;
        cr.xi[col] = xi;
        const live = xi < o.xiSpill ? 0 : xi < o.xiSurge ? 1 : 2;
        cr.type[col] = cr.broken[col] ? cr.typeF[col] : live;
      }
    }
  }

  // --- breakers -------------------------------------------------------------------
  const SAMPLE_KEYS = ['t0s', 't0e', 'x', 'zRoot', 'yRoot', 'dirX', 'dirZ', 'launchZ', 'launchX', 'H', 'hC', 'c', 'xi', 'tanB', 'tau', 'w', 'R', 'u', 'v0', 'face', 'etaC', 'etaT', 'etaCNow', 's', 'p', 'phase', 'alpha', 'taper', 'hit', 'impactZ', 'impactX', 'tipX', 'tipY', 'tipZ', 'seedA', 'seedB', 'slotLip', 'slotSplash'];
  function newBreaker() {
    const b = { id: 0, crestId: 0, crest: null, c0: 0, c1: -1, n: 0, state: 'throwing', t0: 0, impact: false, orphan: -1, done: false, api: null };
    for (const key of SAMPLE_KEYS) b[key] = key === 'phase' ? new Int8Array(S) : key === 'hit' ? new Uint8Array(S) : key === 'slotLip' || key === 'slotSplash' ? new Int32Array(S) : new Float32Array(S);
    return b;
  }
  const pool = [];
  for (let i = 0; i < o.maxBreakers; i++) pool.push(newBreaker());
  const breakers = [];
  const splats = [];
  // Emitters and tube descriptors are rebuilt every frame from object pools
  // (hundreds per frame while a wide lip throws): the arrays hold the live
  // count and the objects are reused, so consumers must not keep them.
  const emitters = [], emitPool = [];
  const tubes = [], tubePool = [];
  function pushEmitter(x, y, z, vx, vy, vz, spread, rate, size, life) {
    const e = emitPool.length ? emitPool.pop() : {};
    e.x = x; e.y = y; e.z = z; e.vx = vx; e.vy = vy; e.vz = vz; e.spread = spread; e.rate = rate; e.size = size; e.life = life;
    emitters.push(e);
  }
  function pushTube(x, y, dx, dy, a0, a1, hw, eta, s, H) {
    const tb = tubePool.length ? tubePool.pop() : {};
    tb.x = x; tb.y = y; tb.dx = dx; tb.dy = dy; tb.a0 = a0; tb.a1 = a1; tb.hw = hw; tb.eta = eta; tb.s = s; tb.H = H;
    tubes.push(tb);
  }

  function launchSample(b, i, cr, col, t) {
    const H = cr.H[col], c = Math.max(cr.c[col], 0.5), etaC = cr.etaC[col], etaT = cr.etaT[col];
    b.t0s[i] = t; b.t0e[i] = t;
    b.x[i] = (col + 0.5) * dxs - o.gridExt / 2;
    b.zRoot[i] = cr.y[col] - o.gridExt / 2;
    b.yRoot[i] = wl + etaC;
    b.dirX[i] = cr.nx[col]; b.dirZ[i] = cr.ny[col];
    b.launchZ[i] = b.zRoot[i]; b.launchX[i] = b.x[i];
    b.H[i] = H; b.hC[i] = cr.hC[col]; b.c[i] = c; b.xi[i] = cr.xi[col]; b.tanB[i] = cr.tanB[col];
    b.etaC[i] = etaC; b.etaT[i] = etaT; b.etaCNow[i] = etaC;
    const d = Math.max(etaC - etaT, 0.05);
    const v0 = o.lipLift * Math.sqrt(2 * G * H);
    const tau = (v0 + Math.sqrt(v0 * v0 + 2 * G * d)) / G;
    b.v0[i] = v0; b.tau[i] = tau;
    const w = tubeWidth(H, cr.xi[col], o.tubeScale);
    b.w[i] = w;
    b.u[i] = w / tau; // crest-frame horizontal speed: the tip lands w ahead of the root
    b.R[i] = w;
    // length of the solver's front face (crest to toe), from the steepest-point row of the scan
    const face = Number.isFinite(cr.Jf[col]) && Number.isFinite(cr.J[col]) ? Math.max(0, cr.Jf[col] - cr.J[col]) * dxs * 2 : 0;
    b.face[i] = face;
    b.s[i] = 0; b.p[i] = 0; b.phase[i] = 0; b.alpha[i] = 1; b.hit[i] = 0;
    b.impactZ[i] = NaN; b.impactX[i] = NaN;
    b.seedA[i] = cnoise(cr.id * 7 + 1, col); b.seedB[i] = cnoise(cr.id * 13 + 5, col);
    b.slotLip[i] = -1; b.slotSplash[i] = -1;
    cr.lipped[col] = t; cr.armed[col] = -1;
  }

  // Shift the sample arrays of a breaker right by `k` (growth at the offshore-x end).
  function shiftRight(b, k) { for (const key of SAMPLE_KEYS) b[key].copyWithin(k, 0, b.n); }

  // A column may be taken by a lip while it is armed (its FC rose within armWindow), plunging and high enough.
  function eligible(cr, col, t) {
    return cr.age[col] >= 0 && cr.armed[col] >= 0 && t - cr.armed[col] <= o.armWindow && cr.lipped[col] < 0 && cr.type[col] === 1 && cr.H[col] >= o.minH;
  }

  function spawn(crests, t, dt) {
    // 1. edges: arm the columns whose FC just crossed spawnFC; mark the ones born broken; re-arm the calm ones
    for (const cr of crests) {
      for (let col = cr.c0; col <= cr.c1; col++) {
        if (cr.age[col] < 0) continue;
        const FC = cr.FC[col], FCp = cr.FCprev[col];
        if (!cr.broken[col]) {
          if (FCp < 0) { // first record of this column
            if (FC >= o.spawnFC) { cr.broken[col] = 1; cr.typeF[col] = cr.type[col]; cr.lipped[col] = t; cr.armed[col] = -1; }
          } else if (FCp < o.spawnFC && FC >= o.spawnFC) {
            cr.broken[col] = 1; cr.typeF[col] = cr.type[col]; cr.armed[col] = t;
          }
        }
        if (cr.armed[col] >= 0 && t - cr.armed[col] > o.armWindow) cr.armed[col] = -1; // not taken in time: a bore
        if (FC < o.rearmFC) {
          cr.calm[col] += dt;
          if (cr.calm[col] >= o.rearm) { cr.lipped[col] = -1; cr.broken[col] = 0; cr.armed[col] = -1; }
        } else cr.calm[col] = 0;
      }
    }
    // 2. runs of eligible columns -> grow an adjacent lip or throw a new one
    for (const cr of crests) {
      let col = cr.c0;
      while (col <= cr.c1) {
        if (!eligible(cr, col, t)) { col++; continue; }
        let e = col;
        while (e + 1 <= cr.c1 && eligible(cr, e + 1, t)) e++;
        const run = [col, e];
        col = e + 1;
        // grow an adjacent live breaker of this crest if it still has room and is still throwing at that end
        let grown = false;
        for (const b of breakers) {
          if (b.crest !== cr || b.done || b.orphan >= 0) continue;
          if (run[0] === b.c1 + 1 && b.phase[b.n - 1] === 0) {
            const add = Math.min(run[1] - run[0] + 1, S - b.n);
            for (let q = 0; q < add; q++) launchSample(b, b.n + q, cr, run[0] + q, t);
            b.n += add; b.c1 += add;
            run[0] += add; grown = run[0] > run[1];
          } else if (run[1] === b.c0 - 1 && b.phase[0] === 0) {
            const add = Math.min(run[1] - run[0] + 1, S - b.n);
            shiftRight(b, add);
            for (let q = 0; q < add; q++) launchSample(b, q, cr, run[1] - add + 1 + q, t);
            b.n += add; b.c0 -= add;
            run[1] -= add; grown = run[0] > run[1];
          }
          if (grown) break;
        }
        if (grown) continue;
        if ((run[1] - run[0] + 1) * dxs < o.minStretch) continue;
        while (run[0] <= run[1]) {
          if (!pool.length) return;
          const b = pool.pop();
          const n = Math.min(run[1] - run[0] + 1, S);
          b.id = nextId++; b.crestId = cr.id; b.crest = cr; b.c0 = run[0]; b.c1 = run[0] + n - 1; b.n = n;
          b.state = 'throwing'; b.t0 = t; b.impact = false; b.orphan = -1; b.done = false; b.api = null;
          for (let q = 0; q < n; q++) launchSample(b, q, cr, run[0] + q, t);
          breakers.push(b);
          run[0] += n;
        }
      }
    }
  }

  // Two lips of the same crest that have grown into contact become one ribbon
  // (the per-sample launch times keep the peel timing; no seam between them).
  function mergeAdjacent() {
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < breakers.length && !merged; i++) {
        const a = breakers[i];
        if (a.done || a.orphan >= 0) continue;
        for (let j = 0; j < breakers.length; j++) {
          const b = breakers[j];
          if (b === a || b.done || b.orphan >= 0 || b.crest !== a.crest || a.c1 + 1 !== b.c0 || a.n + b.n > S) continue;
          for (const key of SAMPLE_KEYS) a[key].set(b[key].subarray(0, b.n), a.n);
          a.n += b.n; a.c1 = b.c1; a.t0 = Math.min(a.t0, b.t0); a.api = null;
          b.n = 0; b.done = true; b.crest = null; b.api = null; // its slots moved to a; nothing to free
          pool.push(b); breakers[j] = breakers[breakers.length - 1]; breakers.pop();
          merged = true;
          break;
        }
      }
    }
  }

  function advance(b, t, dt, crestsAlive) {
    const cr = b.crest;
    const alive = cr && cr.live && crestsAlive.has(cr) && cr.id === b.crestId;
    if (!alive && b.orphan < 0) b.orphan = t;
    let anyThrow = false, anySplash = false, anyLive = false, impact = false;
    const orphanFade = b.orphan >= 0 ? clamp(1 - (t - b.orphan) / o.fadeTime, 0, 1) : 1;
    // Launch times are quantised to frames; along a peeling lip that makes the
    // phase/fade step every few samples. Smooth them alongshore (+-3 samples).
    for (let i = 0; i < b.n; i++) {
      let sum = 0, m = 0;
      for (let q = -3; q <= 3; q++) { const j = i + q; if (j < 0 || j >= b.n) continue; sum += b.t0s[j]; m++; }
      b.t0e[i] = Math.min(sum / m, t);
    }
    const hw = 0.5 * dxs + 0.05;
    for (let i = 0; i < b.n; i++) {
      const col = b.c0 + i;
      // root follows the tracked crest; extrapolate at c when the column is gone
      if (alive && cr.age[col] >= 0 && Number.isFinite(cr.y[col])) {
        const zr = cr.y[col] - o.gridExt / 2;
        // keep the root monotone: never let it step back more than a little (fit jitter)
        b.zRoot[i] = Math.max(zr, b.zRoot[i] - 0.3 * dxs);
        b.etaCNow[i] += (cr.etaC[col] - b.etaCNow[i]) * Math.min(1, dt / 0.15);
        b.dirX[i] += (cr.nx[col] - b.dirX[i]) * Math.min(1, dt / 0.3);
        b.dirZ[i] += (cr.ny[col] - b.dirZ[i]) * Math.min(1, dt / 0.3);
      } else {
        b.zRoot[i] += b.c[i] * dt * b.dirZ[i];
      }
      b.yRoot[i] = wl + b.etaCNow[i];
      const s = t - b.t0e[i];
      b.s[i] = s;
      const tau = b.tau[i];
      const p = clamp(s / tau, 0, 1);
      b.p[i] = p;
      let phase = s < tau ? 0 : s < tau + o.splashTime ? 1 : s < tau + o.splashTime + o.fadeTime ? 2 : 3;
      b.phase[i] = phase;
      b.alpha[i] = (phase <= 1 ? 1 : phase === 2 ? 1 - (s - tau - o.splashTime) / o.fadeTime : 0) * orphanFade;
      // end taper
      const e = Math.min(1, (i + 0.5) / taperN, (b.n - 0.5 - i) / taperN);
      b.taper[i] = e * e * (3 - 2 * e);
      // tip (nominal, untapered)
      const tp = Math.min(s, tau);
      const a = b.u[i] * tp;
      b.tipX[i] = b.x[i] + b.dirX[i] * a;
      b.tipZ[i] = b.zRoot[i] + b.dirZ[i] * a;
      b.tipY[i] = wl + b.etaC[i] + b.v0[i] * tp - 0.5 * G * tp * tp;
      if (phase >= 1 && !b.hit[i]) {
        b.hit[i] = 1;
        b.impactX[i] = b.tipX[i]; b.impactZ[i] = b.tipZ[i];
        if (b.orphan < 0 || t - b.orphan < 0.05) {
          splats.push({ x: b.tipX[i], y: b.tipZ[i], r: o.splatRadius * b.H[i], s: 1 });
          impact = true;
        }
      }
      if (phase === 0) anyThrow = true; else if (phase === 1) anySplash = true;
      if (phase < 3 && b.alpha[i] > 0) anyLive = true;
      const H = b.H[i], c = b.c[i], w = b.w[i];
      // tube: the void under the curtain, from the back wall (tubeBack w ahead of the root) to beyond the
      // solver's face toe; the renderer lowers its surface to etaT there (blended by s)
      if (phase < 3 && b.alpha[i] > 0) {
        const a0 = o.tubeBack * w, a1 = clamp(Math.max(w + 0.5 * H, 1.25 * b.face[i] + dxs), w + 0.5 * H, 4 * H + dxs);
        const eT = 0.3 + 0.7 * b.taper[i]; // the void closes with the ribbon at the ends (same factor as the curtain's throw)
        pushTube(b.x[i], b.zRoot[i], b.dirX[i], b.dirZ[i], a0, a0 + (a1 - a0) * eT, hw, b.etaT[i], clamp(s / o.tubeRamp, 0, 1) * b.alpha[i] * b.taper[i], H);
      }
      // spray emitters (rates per metre of crest, scaled by the sample spacing)
      if (orphanFade > 0.2 && b.taper[i] > 0.2) {
        if (phase === 0 && p > 0.15) {
          pushEmitter(b.tipX[i], b.tipY[i], b.tipZ[i], b.dirX[i] * (c + 0.5 * b.u[i]) * 0.7, -G * tp * 0.4 + 1.0, b.dirZ[i] * (c + 0.5 * b.u[i]) * 0.7, 1.2 + 0.8 * H, o.sprayRate * 8 * dxs * (0.5 + H), 0.06 + 0.04 * H, 0.7);
        } else if (phase === 1) {
          const ss = s - tau;
          if (ss < 0.35) {
            const up = Math.sqrt(2 * G * H) * (0.55 + 0.25 * b.seedA[i]);
            pushEmitter(b.impactX[i], wl + b.etaT[i], b.impactZ[i] + c * ss * 0.6 * b.dirZ[i], b.dirX[i] * c * 0.5, up, b.dirZ[i] * c * 0.5, 1.5 + 1.0 * H, o.sprayRate * 38 * dxs * (0.5 + H) * (1 - ss / 0.35), 0.08 + 0.06 * H, 1.0);
          }
          const hs = o.splashHeight * H * Math.sin(Math.PI * Math.min(ss / o.splashTime, 1));
          if (ss > 0.08 && ss < o.splashTime) pushEmitter(b.impactX[i] + b.dirX[i] * hs * 0.5, wl + b.etaT[i] + hs, b.impactZ[i] + b.dirZ[i] * (c * ss * 0.6 + hs * 0.5), b.dirX[i] * c * 0.4, 1.5 + 2.0 * (1 - ss / o.splashTime), b.dirZ[i] * c * 0.4, 1.6, o.sprayRate * 16 * dxs * (0.5 + H), 0.07 + 0.05 * H, 0.9);
        }
      }
    }
    b.impact = impact;
    b.state = anyThrow ? 'throwing' : anySplash ? 'splashup' : 'collapsed';
    if (impact && b.state !== 'throwing') b.state = 'splashup';
    b.done = !anyLive || (b.orphan >= 0 && t - b.orphan >= o.fadeTime);
  }

  // Build (once per frame, lazily) the public description of a breaker: a
  // snapshot of this frame (scalars; `points` is built on first read).
  function apiOf(b) {
    if (b.api && b.api._frame === frame) return b.api;
    let Hs = 0, xis = 0, cs = 0, taus = 0, ws = 0, Rs = 0, us = 0, lz = 0, iz = 0, ni = 0;
    for (let i = 0; i < b.n; i++) { Hs += b.H[i]; xis += b.xi[i]; cs += b.c[i]; taus += b.tau[i]; ws += b.w[i]; Rs += b.R[i]; us += b.u[i]; lz += b.launchZ[i]; if (b.hit[i]) { iz += b.impactZ[i]; ni++; } }
    const n = b.n;
    const self = {
      _frame: frame, id: b.id, crestId: b.crestId, state: b.state, t0: b.t0, n, impact: b.impact, orphan: b.orphan >= 0,
      H: Hs / n, xi: xis / n, c: cs / n, tau: taus / n, tubeWidth: ws / n, throw: Rs / n, tipSpeed: (cs + us) / n, launchZ: lz / n, impactZ: ni ? iz / ni : NaN,
      x0: b.x[0], x1: b.x[n - 1],
      get points() {
        if (!self._pts) {
          const pts = [];
          for (let i = 0; i < n; i++) pts.push({ x: b.x[i], y: b.zRoot[i], etaC: b.etaC[i], etaT: b.etaT[i], hC: b.hC[i], c: b.c[i], H: b.H[i], xi: b.xi[i], tanB: b.tanB[i], w: b.w[i], R: b.R[i], tau: b.tau[i], u: b.u[i], face: b.face[i], phase: b.phase[i], p: b.p[i], s: b.s[i], alpha: b.alpha[i], taper: b.taper[i], launchZ: b.launchZ[i], impactZ: b.impactZ[i], tipX: b.tipX[i], tipY: b.tipY[i], tipZ: b.tipZ[i], rootZ: b.zRoot[i], rootY: b.yRoot[i], dirX: b.dirX[i], dirZ: b.dirZ[i] });
          self._pts = pts;
        }
        return self._pts;
      },
    };
    b.api = self;
    return self;
  }

  let frame = 0;
  const alive = new Set();
  const activeOut = [];

  function update(crests, t, dt) {
    frame++;
    splats.length = 0;
    for (const e of emitters) emitPool.push(e);
    emitters.length = 0;
    for (const tb of tubes) tubePool.push(tb);
    tubes.length = 0;
    alive.clear();
    for (const cr of crests) alive.add(cr);
    spawn(crests, t, dt);
    mergeAdjacent();
    for (let k = breakers.length - 1; k >= 0; k--) {
      const b = breakers[k];
      advance(b, t, dt, alive);
      if (b.done) { if (self.onRelease) self.onRelease(b); b.crest = null; b.api = null; pool.push(b); breakers[k] = breakers[breakers.length - 1]; breakers.pop(); }
    }
    activeOut.length = 0;
    for (const b of breakers) activeOut.push(apiOf(b));
    return { breakers, splats, emitters, tubes, active: activeOut };
  }

  // --- profiles for the mesh layer --------------------------------------------------
  // lipProfile(b, i, out): the curtain cross-section of sample i, NPROF points.
  // out.a[k] metres ahead of the crest root along the crest normal, out.y[k]
  // world elevation, out.th[k] thickness (m), out.fu[k]/out.fl[k] foam on the
  // upper / lower face, out.alpha the ribbon alpha of this sample.
  function lipProfile(b, i, out) {
    const H = b.H[i], tau = b.tau[i], s = b.s[i], p = b.p[i], e = b.taper[i], phase = b.phase[i];
    const sE = Math.min(s, tau) * (0.3 + 0.7 * e);
    const thin = 1 + 0.3 * (b.seedB[i] - 0.5);
    // Whiteness. Throwing: a little aeration at the leading edge. Splash-up:
    // the lower ~0.3 H of the sheet (the part that hit) is white, the sheet
    // above stays green water. Collapse: the white creeps up the sheet as it
    // fades and the solver's bore takes over.
    const ss = s - tau;
    const white = phase >= 1 ? Math.min(1, 0.4 + ss / 0.3) : sstep(0.85, 1, p) * 0.3;
    // profile parameter q where the drop is 0.7 H: q = sqrt(0.7) = 0.84; the white starts a little above that
    const qW = phase === 1 ? 0.8 : phase >= 2 ? 0.8 - 0.9 * clamp((ss - o.splashTime) / o.fadeTime, 0, 1) : 1.1;
    out.alpha = b.alpha[i];
    out.al.fill(1);
    // k = 0: root, inside the crest
    out.a[0] = -0.3 * H;
    out.y[0] = b.yRoot[i] - 0.25 * H;
    out.th[0] = H * o.rootThickness * (0.5 + 0.5 * e) * thin;
    out.fu[0] = 0.05; out.fl[0] = 0.15;
    const yL = wl + b.etaC[i];
    for (let k = 1; k < NPROF; k++) {
      const q = (k - 1) / (NPROF - 2);
      const tp = sE * q;
      out.a[k] = b.u[i] * tp;
      out.y[k] = yL + b.v0[i] * tp - 0.5 * G * tp * tp;
      const bulb = Math.exp(-(((1 - q) / 0.2) ** 2));
      const sheet = (o.rootThickness + (o.tipThickness - 0.06 - o.rootThickness) * q) * (1 - 0.2 * p);
      out.th[k] = H * (sheet + 0.06 * bulb) * (0.4 + 0.6 * e) * thin;
      const lead = sstep(0.55, 1, q);
      const wq = sstep(qW - 0.12, qW + 0.06, q);
      out.fu[k] = 0.06 + 0.3 * lead * (0.3 + 0.7 * p) + white * wq;
      out.fl[k] = 0.12 + 0.4 * sstep(0.4, 1, q) * (0.5 + 0.5 * p) + white * Math.max(wq, sstep(0.7, 1, q) * 0.7);
    }
    return out;
  }

  // splashProfile(b, i, out): the splash-up sheet rising from the impact line.
  // Returns false when the sample has no splash-up at this time.
  function splashProfile(b, i, out) {
    const ss = b.s[i] - b.tau[i];
    if (!(ss >= 0) || ss > o.splashTime + 0.1 || b.alpha[i] <= 0) return false;
    const H = b.H[i], c = b.c[i], e = b.taper[i];
    const f = Math.min(ss / o.splashTime, 1);
    const rag = 0.75 + 0.5 * b.seedA[i] + 0.25 * (b.seedB[i] - 0.5); // ragged top
    const hs = o.splashHeight * H * Math.pow(Math.sin(Math.PI * f), 0.8) * (0.4 + 0.6 * e) * rag;
    const aBase = b.R[i] * (0.3 + 0.7 * e) + c * ss * 0.6 - 0.15 * H;
    const yBase = wl + b.etaT[i] - 0.15 * H;
    out.alpha = b.alpha[i] * (1 - sstep(0.7, 1, f)) * (ss > o.splashTime ? 0 : 1);
    for (let k = 0; k < NPROF; k++) {
      const q = k / (NPROF - 1);
      out.a[k] = aBase + hs * (0.5 * q + 0.25 * Math.sin(Math.PI * q)) + 0.3 * H * f * q;
      out.y[k] = yBase + hs * q;
      out.th[k] = H * (0.14 * (1 - q) * (1 - q) + 0.04) * (1 + 0.4 * (b.seedB[i] - 0.5)) * (0.5 + 0.5 * e) * (0.15 + 0.85 * Math.sin(Math.PI * f));
      out.fu[k] = 1; out.fl[k] = 1;
      out.al[k] = 1 - 0.55 * q * q; // misty toward the top
    }
    return true;
  }

  function newProfile() { return { a: new Float32Array(NPROF), y: new Float32Array(NPROF), th: new Float32Array(NPROF), fu: new Float32Array(NPROF), fl: new Float32Array(NPROF), al: new Float32Array(NPROF).fill(1), alpha: 1 }; }

  const self = {
    options: o, lay, breakers, classify, update, lipProfile, splashProfile, newProfile, tubeWidth,
    onRelease: null, // called with the breaker just before it is recycled (the mesh layer frees its vertex slots)
    setWaterLevel: (v) => { wl = v; },
    get wl() { return wl; },
    get active() { return activeOut; },
    get tubes() { return tubes; },
  };
  return self;
}
