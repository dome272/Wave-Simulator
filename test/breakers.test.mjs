// Node test battery for the breakers package (contract section 6 "Tests").
// Run: /opt/node22/bin/node test/breakers.test.mjs   (exit code 1 on any FAIL)
//
// No WebGL: the tracker and lip model are pure JS; the mesh and spray layers
// are exercised geometry-only with three.js 0.160 loaded from the local copy
// (THREE_VENDOR, default /home/claude/three-0.160) through a module resolve
// hook that maps the bare specifier 'three' used by src/breakers/*.js.
import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import { standardCase, makeScenario, planeBeach, phaseSpeed, G } from './synthetic-crests.mjs';
import { createTracker } from '../src/breakers/crest-tracker.js';
import { createLipModel, NPROF, tubeWidth } from '../src/breakers/lip-model.js';
import { EXT } from '../src/terrain.js';

const THREE_DIR = process.env.THREE_VENDOR || '/home/claude/three-0.160';
const THREE_URL = pathToFileURL(`${THREE_DIR}/build/three.module.js`).href;
register(`data:text/javascript,export async function resolve(s, c, n) { if (s === 'three') return { url: ${JSON.stringify(THREE_URL)}, shortCircuit: true }; return n(s, c); }`, import.meta.url);
const { createBreakers } = await import('../src/breakers/index.js');
const { NLOOP } = await import('../src/breakers/lip-mesh.js');
const { createSpray } = await import('../src/breakers/spray.js');

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  (' + detail + ')' : ''}`);
  if (!ok) failures++;
}
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : String(v));
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : String(v));
const DT = 1 / 30;
const snap = (a) => ({ ...a, points: a.points.map((p) => ({ ...p })) }); // brk.active entries are per-frame snapshots; copy the points too

// ---------------------------------------------------------------------------
console.log('\n# 1. Tracker: one sinusoidal crest advancing shoreward');
{
  const { sc, c } = standardCase({ amp: 2, lambda: 60, breakY: Infinity });
  const tr = createTracker(sc.lay);
  const ids = new Set();
  let posErr = 0, nPos = 0, spdErr = 0, nSpd = 0, maxCrests = 0;
  for (let f = 0, t = 0; f < 90; f++, t += DT) {
    tr.update(sc.records(t), t, DT);
    maxCrests = Math.max(maxCrests, tr.crests.length);
    for (const cr of tr.crests) ids.add(cr.id);
    const cr = tr.crests[0];
    if (!cr) continue;
    for (let col = cr.c0; col <= cr.c1; col += 5) {
      if (cr.age[col] < 0) continue;
      posErr += Math.abs(cr.y[col] - sc.crestY(0, tr.colX(col), t)); nPos++;
      if (t >= 1.0) {
        const slope = (sc.crestY(0, tr.colX(col) + 0.5, t) - sc.crestY(0, tr.colX(col) - 0.5, t));
        const cTrue = c / Math.hypot(1, slope); // normal speed of a line translating at c in y
        spdErr += Math.abs(cr.c[col] - cTrue) / cTrue; nSpd++;
      }
    }
  }
  const cr = tr.crests[0];
  check('one tracked crest, stable id, spanning every column', tr.crests.length === 1 && ids.size === 1 && cr.n === sc.lay.V && maxCrests === 1, `crests=${tr.crests.length} ids=${ids.size} columns=${cr && cr.n}/${sc.lay.V}`);
  check('filtered position within half a cell of the true crest line', posErr / nPos < 0.5 * sc.lay.dxs, `mean |dy| = ${f3(posErr / nPos)} m (dxs ${sc.lay.dxs})`);
  check('phase speed from the tracked motion within 10% after 1 s', spdErr / nSpd < 0.10, `mean rel error = ${(100 * spdErr / nSpd).toFixed(1)}% (true c ${f2(c)} m/s)`);
  // straight crest (every column steps rows at the same instant: the worst case for the fit)
  const cs2 = standardCase({ amp: 0, breakY: Infinity });
  const tr2 = createTracker(cs2.sc.lay);
  let e2 = 0, n2 = 0, m2 = 0;
  for (let f = 0, t = 0; f < 90; f++, t += DT) {
    tr2.update(cs2.sc.records(t), t, DT);
    const cr = tr2.crests[0]; if (!cr || t < 1) continue;
    for (let col = cr.c0; col <= cr.c1; col += 5) { const e = Math.abs(cr.c[col] - cs2.c) / cs2.c; e2 += e; n2++; m2 = Math.max(m2, e); }
  }
  check('straight crest: phase speed within 10% (max) after 1 s', m2 < 0.10, `mean ${(100 * e2 / n2).toFixed(1)}%, max ${(100 * m2).toFixed(1)}%`);
}

// ---------------------------------------------------------------------------
console.log('\n# 2. Tracker: two crests, one oblique, r = 2; the leading one vanishes (slot shift)');
{
  const beach = planeBeach();
  const sc = makeScenario({ r: 2, beach, crests: [{ y0: 440, c: 6.5, H: 2, obliq: 0.08, stop: 3.0 }, { y0: 380, c: 7.5, H: 1.5, amp: 3, lambda: 80 }] });
  const tr = createTracker(sc.lay);
  const idsByCrest = [new Set(), new Set()];
  let obliqErr = 0, nOb = 0, twoFrames = 0, dropT = -1, idAfter = -1;
  for (let f = 0, t = 0; f < 150; f++, t += DT) {
    tr.update(sc.records(t), t, DT);
    if (t <= 3.0 && tr.crests.length === 2) twoFrames++;
    for (const cr of tr.crests) {
      const col = 255, y = cr.y[col];
      const k = Math.abs(y - sc.crestY(0, tr.colX(col), Math.min(t, 3))) < Math.abs(y - sc.crestY(1, tr.colX(col), t)) ? 0 : 1;
      idsByCrest[k].add(cr.id);
      if (k === 0 && t >= 1 && t <= 3) { obliqErr += Math.abs(cr.c[col] - 6.5 * Math.cos(Math.atan(0.08))) / (6.5 * Math.cos(Math.atan(0.08))); nOb++; }
    }
    if (t > 3.0 && tr.crests.length === 1 && dropT < 0) { dropT = t; idAfter = tr.crests[0].id; }
  }
  check('both crests tracked with stable ids until the leading one stops', twoFrames >= 89 && idsByCrest[0].size === 1 && idsByCrest[1].size === 1, `frames with 2 crests = ${twoFrames}/91, ids: ${[...idsByCrest[0]]} / ${[...idsByCrest[1]]}`);
  check('oblique crest: speed along its normal = c cos(theta) within 5%', obliqErr / nOb < 0.05, `mean rel error ${(100 * obliqErr / nOb).toFixed(1)}%`);
  check('leading crest dropped 0.5 s after its records stop; the other keeps its id through the slot shift', dropT > 3.5 - 1e-6 && dropT <= 3.5 + 2 * DT + 1e-6 && idAfter === [...idsByCrest[1]][0], `dropped at t = ${f3(dropT)} (stop 3.0 + 0.5), surviving id ${idAfter}`);
}

// ---------------------------------------------------------------------------
console.log('\n# 3. Tracker: a permanent 2-column hole in the records is bridged (one crest, one lip)');
{
  const cs = standardCase({});
  const base = cs.sc.records;
  const rec = (t) => { const r = base(t); for (const col of [120, 121]) for (let s = 0; s < 6; s++) r[(col * 6 + s) * 8] = -1; return r; };
  const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
  const crestIds = new Set(), lipIds = new Set();
  let maxCrests = 0, maxLips = 0, spans = null, interp = null, dyHole = 0, nHole = 0;
  for (let f = 0, t = 0; f < 100; f++, t += DT) {
    brk.update(rec(t), DT, { t, swell: cs.swell });
    for (const c of brk.crests) crestIds.add(c.id);
    maxCrests = Math.max(maxCrests, brk.crests.length);
    for (const a of brk.active) lipIds.add(a.id);
    maxLips = Math.max(maxLips, brk.active.length);
    if (brk.active.length && !spans) { spans = brk.active.map((a) => [a.x0, a.x1, a.n]); interp = brk.crests[0].points.filter((p) => p.interp).map((p) => p.col); }
    const cr = brk.tracker.crests[0];
    if (cr && t > 1) for (const col of [120, 121]) if (cr.age[col] >= 0) { dyHole += Math.abs(cr.y[col] - cs.sc.crestY(0, brk.tracker.colX(col), t)); nHole++; }
  }
  check('one crest id, one lip spanning all 255 columns', crestIds.size === 1 && maxCrests === 1 && lipIds.size === 1 && maxLips === 1 && spans && spans[0][2] === cs.sc.lay.V, `crest ids ${[...crestIds]}, lips ${[...lipIds]}, spans ${JSON.stringify(spans)}`);
  check('the hole columns are interpolated and sit on the true crest line', interp && interp.length === 2 && interp[0] === 120 && interp[1] === 121 && nHole > 0 && dyHole / nHole < 0.5 * cs.sc.lay.dxs, `interpolated columns ${JSON.stringify(interp)}, mean |dy| at the hole ${f3(dyHole / Math.max(nHole, 1))} m`);
  brk.dispose();
}

// ---------------------------------------------------------------------------
console.log('\n# 4. Lip model: spawn place and time, kinematics (contract tube), splats');
{
  const cs = standardCase({});
  const { sc, c, H, beach, swell, breakY, slope } = cs;
  const brk = createBreakers({ layout: sc.lay, wl: 0, beach, swell });
  const tBreak = (breakY - cs.y0) / c;
  const L0 = (G * swell.period ** 2) / (2 * Math.PI);
  const xiExp = slope / Math.sqrt(H / L0);
  const wExp = tubeWidth(H, xiExp); // the contract tube width from the scenario's own H and xi
  const tau = Math.sqrt((2 * H) / G);
  let spawnT = -1, spawn = null, nActiveMax = 0, totalSplats = 0, impactFrame = null, impactSplats = null, firstSplatT = -1, lastState = '', removedT = -1, spawnCount = 0;
  const seenIds = new Set();
  for (let f = 0, t = 0; f < 150; f++, t += DT) {
    brk.update(sc.records(t), DT, { t, swell });
    nActiveMax = Math.max(nActiveMax, brk.active.length);
    for (const a of brk.active) seenIds.add(a.id);
    if (brk.active.length && spawnT < 0) { spawnT = t; spawn = snap(brk.active[0]); spawnCount = brk.active.length; }
    totalSplats += brk.splats.length;
    if (brk.splats.length && firstSplatT < 0) {
      firstSplatT = t; impactSplats = brk.splats.slice();
      const a = snap(brk.active[0]), pts = a.points;
      impactFrame = { t, s: t - a.t0, a, tip: pts.map((p) => p.tipZ - p.rootZ), tipY: pts.map((p) => p.tipY), world: pts.map((p) => p.impactZ - p.launchZ), taper: pts.map((p) => p.taper), R: pts.map((p) => p.R), w: pts.map((p) => p.w), c: pts.map((p) => p.c), tau: pts.map((p) => p.tau), etaT: pts.map((p) => p.etaT), state: a.state };
    }
    if (brk.active.length) lastState = brk.active[0].state;
    if (spawnT >= 0 && !brk.active.length && removedT < 0) removedT = t;
  }
  check('a breaker spawns within one frame of FC switching on', spawnT >= tBreak - 1e-6 && spawnT <= tBreak + DT + 1e-6, `spawn t = ${f3(spawnT)}, FC on at ${f3(tBreak)}`);
  check('it covers the whole breaking stretch as one lip (255 columns), state throwing', spawn && spawn.n === sc.lay.V && spawnCount === 1 && spawn.state === 'throwing', `n = ${spawn && spawn.n}, breakers at spawn = ${spawnCount}, state ${spawn && spawn.state}`);
  check('launch position = the crest line where FC switched on (within 1 m)', spawn && Math.abs(spawn.launchZ - (breakY - EXT / 2 + c * (spawnT - tBreak))) < 1.0, `launchZ ${f2(spawn && spawn.launchZ)} vs ${f2(breakY - EXT / 2 + c * (spawnT - tBreak))} (world z)`);
  check('H, xi and the breaker type come out right', spawn && Math.abs(spawn.H - H) < 0.02 && Math.abs(spawn.xi - xiExp) / xiExp < 0.05, `H ${f3(spawn && spawn.H)} (true ${H}), xi ${f3(spawn && spawn.xi)} (expected ${f3(xiExp)}: plunging)`);
  check('tracked phase speed at launch within 10% of the true c', spawn && Math.abs(spawn.c - c) / c < 0.10, `c ${f3(spawn && spawn.c)} vs ${f3(c)}`);
  check('tau = sqrt(2 (etaC - etaT) / g)', spawn && Math.abs(spawn.tau - tau) < 1e-3, `tau ${f3(spawn && spawn.tau)} vs ${f3(tau)} s`);
  check('only one lip is ever spawned on this stretch', seenIds.size === 1 && nActiveMax === 1, `distinct breaker ids = ${seenIds.size}, max simultaneous = ${nActiveMax}`);
  if (impactFrame) {
    const interior = impactFrame.taper.map((e, i) => (e >= 0.999 ? i : -1)).filter((i) => i >= 0);
    // crest-frame: the tip lands exactly the contract tube width ahead of the root (expectation from the scenario's H and xi, not the breaker's arrays)
    const tipErr = Math.max(...interior.map((i) => Math.abs(impactFrame.tip[i] - wExp) / wExp));
    const yErr = Math.max(...interior.map((i) => Math.abs(impactFrame.tipY[i] - impactFrame.etaT[i])));
    // world frame: launch point + c tau (the root follows the crest) + w; expectation from the TRUE c
    const worldExp = c * tau + wExp;
    const worldErr = Math.max(...interior.map((i) => Math.abs(impactFrame.world[i] - worldExp) / worldExp));
    const tipSpeed = interior.map((i) => impactFrame.world[i] / tau);
    const tipSpeedC = [Math.min(...tipSpeed) / c, Math.max(...tipSpeed) / c];
    check('impact happens at t0 + tau (within one frame)', impactFrame.s >= tau - 1e-6 && impactFrame.s <= tau + DT + 1e-6, `s = ${f3(impactFrame.s)} s, tau = ${f3(tau)} s`);
    check('crest frame: the lip lands w = H (0.65 + 0.3 clamp((xi-0.4)/1.6)) ahead of the crest root (the contract tube width)', tipErr < 0.02, `tip - root = ${f3(impactFrame.tip[120])} m, w(H, xi) = ${f3(wExp)} m = ${f2(wExp / H)} H, max rel error ${(100 * tipErr).toFixed(1)}%`);
    check('the lip tip meets the trough level etaT', yErr < 0.05, `max |tipY - etaT| = ${f3(yErr)} m`);
    check('world frame: the tip lands c tau + w from the launch point (true c), i.e. its mean speed is 1.0-1.5 c', worldErr < 0.10 && tipSpeedC[0] >= 1.0 && tipSpeedC[1] <= 1.5, `impactZ - launchZ = ${f2(impactFrame.world[120])} m, expected ${f2(worldExp)} m (1.15 c tau would be ${f2(1.15 * c * tau)} m), max rel error ${(100 * worldErr).toFixed(1)}%; tip speed ${f2(tipSpeedC[0])}-${f2(tipSpeedC[1])} c`);
    check('splats appear at impact: one per column, strength 1, radius H', impactSplats.length === sc.lay.V && impactSplats.every((s) => s.s === 1 && Math.abs(s.r - H) < 1e-6) && totalSplats === sc.lay.V, `${impactSplats.length} splats at t = ${f3(firstSplatT)}, r = ${f2(impactSplats[0].r)}, total over the run ${totalSplats}`);
    const sp = impactSplats[120], pt = impactFrame.a.points[120];
    check('splats sit on the impact line (world x/z of the tip)', Math.abs(sp.x - pt.tipX) < 1e-3 && Math.abs(sp.y - pt.tipZ) < 1e-3, `splat (${f2(sp.x)}, ${f2(sp.y)}) tip (${f2(pt.tipX)}, ${f2(pt.tipZ)})`);
    check('state goes throwing -> splashup -> collapsed and the breaker is removed after tau + 0.5 + 0.4 s', impactFrame.state === 'splashup' && lastState === 'collapsed' && removedT > 0 && Math.abs(removedT - (spawnT + tau + 0.9)) <= 2 * DT + 1e-6, `removed at t0 + ${f3(removedT - spawnT)} s (expected ${f3(tau + 0.9)})`);
  } else check('impact reached', false, 'no splats within 5 s');
  brk.dispose();
}

// ---------------------------------------------------------------------------
console.log('\n# 5. Spawn rule is edge-triggered: no lip on a crest born broken, after a record gap, or on a decaying spilling bore; re-arm after calm');
{
  // (a) crest first seen with FC = 1 (breakY offshore of y0): a bore, no lip
  {
    const cs = standardCase({ breakY: 400 });
    const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
    let spawnFrame = -1, type = '', state = '';
    for (let f = 0, t = 0; f < 60; f++, t += DT) { brk.update(cs.sc.records(t), DT, { t, swell: cs.swell }); if (brk.active.length && spawnFrame < 0) spawnFrame = f; if (brk.crests.length) { type = brk.crests[0].type; state = brk.crests[0].state; } }
    check('a crest first seen with FC = 1 throws no lip (FC never rose above 0.5 while tracked)', spawnFrame < 0 && state === 'breaking' && type === 'plunging', `lip spawned at frame ${spawnFrame} (-1 = never), crest ${state}/${type}`);
    brk.dispose();
  }
  // (b) the standard plunging case with the records blanked for 0.6 s (> dropAfter) after the lip: the re-acquired crest is born broken
  {
    const cs = standardCase({});
    const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
    const ids = new Set(), crestIds = new Set(); const spawns = [];
    for (let f = 0, t = 0; f < 150; f++, t += DT) {
      const gap = t > 2.3 && t < 2.9;
      brk.update(gap ? cs.sc.empty() : cs.sc.records(t), DT, { t, swell: cs.swell });
      for (const a of brk.active) if (!ids.has(a.id)) { ids.add(a.id); spawns.push({ id: a.id, t: +t.toFixed(3), crest: a.crestId, n: a.n }); }
      for (const c of brk.crests) crestIds.add(c.id);
    }
    check('one lip only across a 0.6 s record gap (the re-acquired crest, a new id, is born broken)', crestIds.size === 2 && ids.size === 1, `breaker spawns ${JSON.stringify(spawns)}; crest ids ${[...crestIds]}`);
    brk.dispose();
  }
  // (c) 1:40 beach, T 12 s, H 2 m: breaks spilling (xi 0.27); its bore decays to 0.8 m over 2.5 s while FC stays 1
  {
    const cs = standardCase({ slope: 1 / 40, period: 12, H: 2.0, y0: 440, breakY: 455 });
    const base = cs.sc.records, tBreak = (455 - 440) / cs.c;
    const rec = (t) => {
      const r = base(t);
      const k = Math.min(1, Math.max(0, (t - tBreak) / 2.5)), H = 2.0 - 1.2 * k;
      for (let col = 0; col < cs.sc.lay.V; col++) { const b = col * 48; if (r[b] >= 0) { const etaC = 0.7 * H; r[b + 1] = etaC; r[b + 4] = etaC - H; r[b + 2] = r[b + 2] - 0.7 * 2.0 + etaC; } }
      return r;
    };
    const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
    const spawns = []; let types = new Set(), xiLive = 0, Hend = 0;
    for (let f = 0, t = 0; f < 165; f++, t += DT) {
      brk.update(rec(t), DT, { t, swell: cs.swell });
      if (brk.crests.length && t > tBreak) { types.add(brk.crests[0].type); xiLive = brk.crests[0].xi; Hend = brk.crests[0].H; }
      for (const a of brk.active) if (!spawns.some((s) => s.id === a.id)) spawns.push({ id: a.id, t: +t.toFixed(2), H: +a.H.toFixed(2), xi: +a.xi.toFixed(2) });
    }
    const L0 = (G * 144) / (2 * Math.PI), xiEnd = (1 / 40) / Math.sqrt(Hend / L0);
    check('a wave that broke spilling stays spilling while its bore decays (type frozen at onset); no lip is thrown on the bore', spawns.length === 0 && types.size === 1 && types.has('spilling') && xiEnd > 0.4, `lips ${JSON.stringify(spawns)}, types seen after onset ${[...types]}, live xi at the end ${f2(xiEnd)} (H ${f2(Hend)} m) would read plunging`);
    brk.dispose();
  }
  // (d) re-arm: FC on for 0.3 s, off (0) for 1.2 s (> rearm 1.0 s), on again: a reformed wave throws a second lip
  {
    const cs = standardCase({});
    const base = cs.sc.records, tBreak = (cs.breakY - cs.y0) / cs.c;
    const rec = (t) => { const r = base(t); const on = t < tBreak + 0.3 || t > tBreak + 1.5 ? 1 : 0; for (let col = 0; col < cs.sc.lay.V; col++) { const b = col * 48; if (r[b] >= 0) r[b + 3] = r[b + 3] * on; } return r; };
    const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
    const spawns = [];
    for (let f = 0, t = 0; f < 150; f++, t += DT) {
      brk.update(rec(t), DT, { t, swell: cs.swell });
      for (const a of brk.active) if (!spawns.some((s) => s.id === a.id)) spawns.push({ id: a.id, t: +t.toFixed(3), n: a.n });
    }
    check('a column calm (FC < 0.3) for 1 s re-arms: the second rise of FC throws a second lip', spawns.length === 2 && Math.abs(spawns[0].t - tBreak) <= DT + 1e-6 && Math.abs(spawns[1].t - (tBreak + 1.5)) <= 2 * DT + 1e-6, `spawns ${JSON.stringify(spawns)} (FC on at ${f3(tBreak)}, off ${f3(tBreak + 0.3)}-${f3(tBreak + 1.5)})`);
    brk.dispose();
  }
}

// ---------------------------------------------------------------------------
console.log('\n# 6. Tube width follows xi (measured on the mesh); spilling and surging slopes throw no lip');
{
  const run = (over) => {
    const cs = standardCase(over);
    const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
    let first = null, crestType = 'none', voidW = NaN, voidH = NaN, thTip = NaN;
    for (let f = 0, t = 0; f < 110; f++, t += DT) {
      brk.update(cs.sc.records(t), DT, { t, swell: cs.swell });
      if (brk.active.length && !first) first = snap(brk.active[0]);
      if (brk.crests.length && brk.crests[0].state === 'breaking') crestType = brk.crests[0].type;
      // at the first splash-up frame measure the curtain loop of column 120 on the mesh: its horizontal extent ahead of the root (minus the cap radius) is the void width
      if (brk.active.length && brk.active[0].state === 'splashup' && !Number.isFinite(voidW)) {
        const b = brk.model.breakers[0], i = 120 - b.c0, slot = b.slotLip[i], pos = brk.mesh.geometry.attributes.position.array, base = slot * NLOOP;
        let aMax = -1e9, yMin = 1e9;
        for (let k = 0; k < NLOOP; k++) { const x = pos[(base + k) * 3], y = pos[(base + k) * 3 + 1], z = pos[(base + k) * 3 + 2]; const a = (x - b.x[i]) * b.dirX[i] + (z - b.zRoot[i]) * b.dirZ[i]; aMax = Math.max(aMax, a); yMin = Math.min(yMin, y); }
        const P = (k) => [pos[(base + k) * 3], pos[(base + k) * 3 + 1], pos[(base + k) * 3 + 2]];
        const u = P(NPROF - 1), l = P(NPROF + 4);
        thTip = Math.hypot(u[0] - l[0], u[1] - l[1], u[2] - l[2]);
        voidW = aMax - 0.5 * thTip; voidH = (b.yRoot[i]) - yMin;
      }
    }
    brk.dispose();
    return { first, crestType, cs, voidW, voidH, thTip };
  };
  const a = run({ slope: 1 / 15, period: 12 }), b = run({ slope: 1 / 8, period: 12 }), c = run({ slope: 1 / 8, period: 18 });
  const d = run({ slope: 1 / 40, period: 12 }), e = run({ slope: 1 / 3, period: 12, y0: 490, breakY: 495 });
  for (const [name, r] of [['1:15, T 12 s', a], ['1:8, T 12 s', b], ['1:8, T 18 s', c]]) {
    const fb = r.first, L0 = (G * r.cs.swell.period ** 2) / (2 * Math.PI), xiExp = r.cs.slope / Math.sqrt(r.cs.H / L0), w = tubeWidth(r.cs.H, xiExp);
    check(`mesh void at impact: width = H (0.65 + 0.3 clamp((xi-0.4)/1.6)), height = H on ${name}`, fb && Math.abs(r.voidW - w) < 0.05 && Math.abs(r.voidH - r.cs.H) < 0.12 * r.cs.H, fb ? `xi ${f3(fb.xi)} (expected ${f3(xiExp)}), mesh void ${f3(r.voidW)} m wide (formula ${f3(w)}) x ${f2(r.voidH)} m high (H ${f2(r.cs.H)}), tip cap ${f2(r.thTip)} m` : 'no breaker');
  }
  check('tube width increases with xi', a.first && b.first && c.first && a.first.xi < b.first.xi && b.first.xi < c.first.xi && a.voidW < b.voidW && b.voidW < c.voidW, `xi ${f3(a.first && a.first.xi)} -> ${f3(b.first && b.first.xi)} -> ${f3(c.first && c.first.xi)}, mesh void ${f3(a.voidW)} -> ${f3(b.voidW)} -> ${f3(c.voidW)} m`);
  check('1:40 beach (xi < 0.4) is spilling: no lip', !d.first && d.crestType === 'spilling', `type ${d.crestType}, breakers ${d.first ? 1 : 0}`);
  check('1:3 beach (xi >= 2) is surging: no lip', !e.first && e.crestType === 'surging', `type ${e.crestType}, breakers ${e.first ? 1 : 0}`);
}

// ---------------------------------------------------------------------------
console.log('\n# 7. Records stop mid-throw: the crest is dropped after 0.5 s and the lip fades out; update(null) ages the crests');
{
  const cs = standardCase({ stop: 2.2 });
  const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
  let crestGone = -1, lipGone = -1, hadLip = false;
  for (let f = 0, t = 0; f < 150; f++, t += DT) {
    brk.update(cs.sc.records(t), DT, { t, swell: cs.swell });
    if (brk.active.length) hadLip = true;
    if (t > 2.2 && !brk.crests.length && crestGone < 0) crestGone = t;
    if (hadLip && !brk.active.length && lipGone < 0) lipGone = t;
  }
  check('crest dropped 0.5 s after its records stop', hadLip && crestGone >= 2.7 - 1e-6 && crestGone <= 2.7 + 2 * DT + 1e-6, `crests empty at t = ${f3(crestGone)} (stop 2.2 + 0.5)`);
  check('orphaned lip faded out within 0.4 s of the drop', lipGone > 0 && lipGone <= crestGone + 0.4 + 2 * DT + 1e-6, `active empty at t = ${f3(lipGone)}`);
  brk.dispose();
  const cs2 = standardCase({ breakY: Infinity });
  const brk2 = createBreakers({ layout: cs2.sc.lay, wl: 0, beach: cs2.beach, swell: cs2.swell });
  let t = 0, nullGone = -1, tLast = 0;
  for (let f = 0; f < 30; f++, t += DT) { brk2.update(cs2.sc.records(t), DT, { t, swell: cs2.swell }); tLast = t; }
  for (let f = 0; f < 45; f++, t += DT) { brk2.update(null, DT, { t, swell: cs2.swell }); if (!brk2.crests.length && nullGone < 0) nullGone = t; }
  check('update(null records) is an empty scan: the crest is dropped 0.5 s after its last records', nullGone > 0 && nullGone - tLast > 0.5 - 1e-6 && nullGone - tLast <= 0.5 + 2 * DT + 1e-6, `crest gone ${f3(nullGone - tLast)} s after the last real records`);
  brk2.dispose();
}

// ---------------------------------------------------------------------------
console.log('\n# 8. Capacity and merging: at most 24 live breakers; adjacent lips grow into one ribbon');
{
  const cs = standardCase({ amp: 3, lambda: 20 }); // 32 bulges cross the break line first: 32 candidate lips
  const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
  let maxLive = 0, firstCount = 0;
  for (let f = 0, t = 0; f < 120; f++, t += DT) {
    brk.update(cs.sc.records(t), DT, { t, swell: cs.swell });
    if (brk.active.length && !firstCount) firstCount = brk.active.length;
    maxLive = Math.max(maxLive, brk.active.length);
  }
  check('never more than 24 live breakers', maxLive <= 24 && firstCount >= 20, `max simultaneous = ${maxLive}, at first spawn = ${firstCount}`);
  brk.dispose();
  const cs2 = standardCase({ amp: 3, lambda: 40 }); // 16 bulges: every stretch gets a lip, the lips peel along the flanks and meet
  const brk2 = createBreakers({ layout: cs2.sc.lay, wl: 0, beach: cs2.beach, swell: cs2.swell });
  let merged = 0, first2 = 0;
  for (let f = 0, t = 0; f < 120; f++, t += DT) {
    brk2.update(cs2.sc.records(t), DT, { t, swell: cs2.swell });
    if (brk2.active.length && !first2) first2 = brk2.active.length;
    if (brk2.active.length === 1 && brk2.active[0].n === cs2.sc.lay.V) merged++;
  }
  check('16 separate stretches grow along the peeling crest and merge into one 255-sample ribbon', first2 >= 14 && merged > 0, `lips at first spawn ${first2}, frames with a single 255-sample lip: ${merged}`);
  brk2.dispose();
}

// ---------------------------------------------------------------------------
console.log('\n# 9. Curtain mesh (three.js geometry, no renderer) and the tube descriptors');
{
  const cs = standardCase({});
  const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
  let mid = null, atImpact = null, tubeSpawn = null, tubeMid = null, tubeFrames = 0, tubeBad = 0;
  for (let f = 0, t = 0; f < 100; f++, t += DT) {
    brk.update(cs.sc.records(t), DT, { t, swell: cs.swell });
    const a = brk.active[0];
    if (a && !tubeSpawn) tubeSpawn = brk.tubes.map((tb) => ({ ...tb }));
    if (a && !mid && a.points[120].p >= 0.5) { mid = snapshot(brk, 120); tubeMid = { ...brk.tubes[120] }; }
    if (a && !atImpact && a.state === 'splashup') atImpact = snapshot(brk, 120);
    if (a) {
      tubeFrames++;
      const n = brk.model.breakers.reduce((s, b) => s + b.n, 0);
      if (brk.tubes.length !== n) tubeBad++;
      for (const tb of brk.tubes) if (!(tb.a0 >= 0 && tb.a1 > tb.a0 && tb.hw > 0 && tb.s >= 0 && tb.s <= 1 && Number.isFinite(tb.eta))) tubeBad++;
    }
  }
  function snapshot(brk, col) {
    const b = brk.model.breakers[0], i = col - b.c0, geo = brk.mesh.geometry, pos = geo.attributes.position.array, aux = geo.attributes.aux.array;
    const slot = b.slotLip[i], base = slot * NLOOP;
    const P = (k) => [pos[(base + k) * 3], pos[(base + k) * 3 + 1], pos[(base + k) * 3 + 2]];
    const upTip = P(NPROF - 1), loTip = P(NPROF + 4), upRoot = P(1), loRoot = P(NLOOP - 2);
    const thTip = Math.hypot(upTip[0] - loTip[0], upTip[1] - loTip[1], upTip[2] - loTip[2]);
    const thRoot = Math.hypot(upRoot[0] - loRoot[0], upRoot[1] - loRoot[1], upRoot[2] - loRoot[2]);
    let finite = true, zmax = -1e9, ymin = 1e9;
    for (let k = 0; k < NLOOP; k++) { const p = P(k); if (!p.every(Number.isFinite)) finite = false; zmax = Math.max(zmax, p[2]); ymin = Math.min(ymin, p[1]); }
    let allFinite = true; for (let q = 0; q < brk.mesh.watermark * NLOOP * 3; q++) if (!Number.isFinite(pos[q])) { allFinite = false; break; }
    const pt = brk.active[0].points[i];
    // foam along the upper face: root side (k = 2..4) must stay water at impact, the tip region (k = NPROF-1) white
    const foamUpRoot = Math.max(aux[(base + 2) * 4], aux[(base + 3) * 4], aux[(base + 4) * 4]);
    return { thTip, thRoot, finite: finite && allFinite, zmax, ymin, H: b.H[i], p: pt.p, tipZ: pt.tipZ, tipY: pt.tipY, rootZ: pt.rootZ, idx: brk.mesh.indexCount, n: b.n, slots: brk.mesh.usedSlots, alphaTip: aux[(base + NPROF - 1) * 4 + 1], foamTip: aux[(base + NPROF + 2) * 4], foamUpRoot, splashSlots: Array.from(b.slotSplash.subarray(0, b.n)).filter((s) => s >= 0).length, w: b.w[i], etaT: b.etaT[i] };
  }
  check('index count = (samples - 1) * NLOOP * 6 while only the curtain exists', mid && mid.idx === (mid.n - 1) * NLOOP * 6 && mid.slots === mid.n, mid ? `${mid.idx} indices, ${mid.n} samples, ${mid.slots} slots` : 'no mid-throw snapshot');
  check('all written positions finite', mid && atImpact && mid.finite && atImpact.finite);
  check('leading edge is bulbous, ~0.15 H thick, thicker than the sheet at the crest', mid && mid.thTip > 0.10 * mid.H && mid.thTip < 0.20 * mid.H && mid.thTip > mid.thRoot, mid ? `tip ${f3(mid.thTip)} m = ${f3(mid.thTip / mid.H)} H, root ${f3(mid.thRoot)} m = ${f3(mid.thRoot / mid.H)} H` : '');
  check('mid-throw: the curtain reaches the tip position of the model (profile = trajectory family)', mid && Math.abs(mid.zmax - mid.tipZ) < 0.25 && mid.ymin < mid.tipY + 0.05, mid ? `mesh z max ${f2(mid.zmax)} vs tip z ${f2(mid.tipZ)} (root ${f2(mid.rootZ)}), p = ${f2(mid.p)}` : '');
  check('at impact the splash-up sheet is added (second ribbon), the leading edge goes white and the sheet near the crest stays water', atImpact && atImpact.splashSlots === atImpact.n && atImpact.idx === 2 * (atImpact.n - 1) * NLOOP * 6 && atImpact.foamTip >= 0.9 && atImpact.foamUpRoot < 0.3, atImpact ? `splash slots ${atImpact.splashSlots}, indices ${atImpact.idx}, tip foam ${f2(atImpact.foamTip)}, upper-face foam near the root ${f2(atImpact.foamUpRoot)}` : '');
  const tb0 = tubeSpawn && tubeSpawn[120];
  check('brk.tubes: one void strip per lip sample, from 0.1 w ahead of the root to past the face toe, floor at etaT, blended in over 0.15 s', tubeBad === 0 && tubeFrames > 0 && tb0 && Math.abs(tb0.a0 - 0.1 * mid.w) < 1e-3 && tb0.a1 >= mid.w + 0.5 * mid.H && Math.abs(tb0.eta - mid.etaT) < 1e-6 && tb0.s < 0.01 && tubeMid.s > 0.99 && Math.abs(tubeMid.y - mid.rootZ) < 1e-3, `${tubeSpawn && tubeSpawn.length} tubes at spawn; col 120: a0 ${f2(tb0 && tb0.a0)} a1 ${f2(tb0 && tb0.a1)} eta ${f2(tb0 && tb0.eta)} s ${f2(tb0 && tb0.s)} -> mid s ${f2(tubeMid && tubeMid.s)}; bad frames/entries ${tubeBad}`);
  brk.dispose();
  // slot pool short of loops: every curtain sample keeps its loop, splash sheets are what gets dropped
  const brk3 = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell, meshSlots: 300, renderOrder: 7 });
  let curtainMissing = 0, splashMissing = 0, splashFrames = 0;
  for (let f = 0, t = 0; f < 100; f++, t += DT) {
    brk3.update(cs.sc.records(t), DT, { t, swell: cs.swell });
    for (const b of brk3.model.breakers) { let sp = 0; for (let i = 0; i < b.n; i++) { if (b.slotLip[i] < 0) curtainMissing++; if (b.phase[i] === 1) { sp++; if (b.slotSplash[i] < 0) splashMissing++; } } if (sp) splashFrames++; }
  }
  check('with a 300-loop pool a 255-sample lip keeps every curtain loop; the splash-up sheet is what runs short', curtainMissing === 0 && splashFrames > 0 && splashMissing > 0, `curtain samples without a loop ${curtainMissing}, splash samples without a loop ${splashMissing} over ${splashFrames} frames`);
  check('renderOrder option: curtain = given, depth pre-pass = given - 0.5, spray = given + 1', brk3.mesh.mesh.renderOrder === 7 && brk3.mesh.depthMesh.renderOrder === 6.5 && brk3.spray.points.renderOrder === 8, `curtain ${brk3.mesh.mesh.renderOrder}, depth ${brk3.mesh.depthMesh.renderOrder}, spray ${brk3.spray.points.renderOrder}`);
  brk3.dispose();
}

// ---------------------------------------------------------------------------
console.log('\n# 10. Spray particles: emission, gravity, wind drift, death');
{
  const sp = createSpray({ maxParticles: 500, wind: [0, 0, -2], drag: 0 });
  sp.emit([{ x: 0, y: 5, z: 0, vx: 0, vy: 0, vz: 0, spread: 0, rate: 300 / DT, size: 0.2, life: 1.0 }], DT);
  const n0 = sp.count;
  let vy0 = 0; for (let i = 0; i < n0; i++) vy0 += sp.velocities[i * 3 + 1]; vy0 /= n0;
  for (let k = 0; k < 15; k++) sp.step(DT);
  let vy = 0, y = 0; for (let i = 0; i < sp.count; i++) { vy += sp.velocities[i * 3 + 1]; y += sp.positions[i * 3 + 1]; } vy /= sp.count; y /= sp.count;
  check('emitter spawned particles (rate * dt, capped by the pool)', n0 === 300, `${n0} particles`);
  check('free fall: mean vy = -g t after 0.5 s', Math.abs(vy - vy0 + G * 0.5) < 0.02 * G, `mean vy ${f3(vy)} m/s (expected ${f3(vy0 - G * 0.5)})`);
  check('they fell: mean height dropped by ~g t^2 / 2', Math.abs(5 - y - 0.5 * G * 0.25) < 0.25, `mean y ${f3(y)} m`);
  const sp2 = createSpray({ maxParticles: 100, wind: [0, 0, -2], drag: 2.0, gravity: 0 });
  sp2.emit([{ x: 0, y: 5, z: 0, vx: 0, vy: 0, vz: 0, spread: 0, rate: 50 / DT, size: 0.2, life: 3 }], DT);
  for (let k = 0; k < 60; k++) sp2.step(DT);
  let vz = 0; for (let i = 0; i < sp2.count; i++) vz += sp2.velocities[i * 3 + 2]; vz /= sp2.count;
  check('wind drift: velocity relaxes toward the wind', Math.abs(vz + 2) < 0.1, `mean vz ${f3(vz)} m/s (wind -2)`);
  for (let k = 0; k < 40; k++) sp.step(DT);
  check('particles die at the end of their life', sp.count === 0, `${sp.count} left after 1.8 s (lives 0.6..1.4 s)`);
  check('Points geometry draw range follows the live count', sp.geometry.drawRange.count === 0 && sp2.geometry.drawRange.count === sp2.count);
  sp.dispose(); sp2.dispose();
}

// --- water level moves mid-throw: absolute elevations hold -----------------------------
{
  const cs = standardCase({});
  const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell });
  let t = 0, b = null, i = -1;
  for (let f = 0; f < 100 && !b; f++, t += DT) {
    brk.update(cs.sc.records(t), DT, { t, wl: 0, swell: cs.swell });
    const a = brk.active[0];
    if (a && a.points[120].p >= 0.3) { b = brk.model.breakers[0]; i = 120 - b.c0; }
  }
  const prof = brk.model.newProfile();
  const before = { launch: 0 + b.etaC[i], root: b.yRoot[i], floor: 0 + brk.tubes.find((tb) => Math.abs(tb.y - b.zRoot[i]) < 1e-6).eta, sheet: brk.model.lipProfile(b, i, prof).y[1] };
  // the solver keeps the absolute surface when the level moves, so the records' eta drop by 1 m
  const shifted = cs.sc.records(t); for (let k = 0; k < shifted.length; k += 8) if (shifted[k] >= 0) { shifted[k + 1] -= 1; shifted[k + 4] -= 1; }
  brk.update(shifted, DT, { t, wl: 1, swell: cs.swell });
  const after = { launch: 1 + b.etaC[i], root: b.yRoot[i], floor: 1 + brk.tubes.find((tb) => Math.abs(tb.y - b.zRoot[i]) < 1e-6).eta, sheet: brk.model.lipProfile(b, i, prof).y[1] };
  const ok = Math.abs(after.launch - before.launch) < 1e-6 && Math.abs(after.floor - before.floor) < 1e-6 && Math.abs(after.root - before.root) < 1e-3 && Math.abs(after.sheet - before.sheet) < 1e-3;
  check('tide +1 m mid-throw: launch height, tube floor, root and sheet keep their absolute elevation', ok, `launch ${f3(before.launch)} -> ${f3(after.launch)}, floor ${f3(before.floor)} -> ${f3(after.floor)}, root ${f3(before.root)} -> ${f3(after.root)}, sheet ${f3(before.sheet)} -> ${f3(after.sheet)}`);
  brk.dispose();
}

// --- a sample without a curtain loop gets no tube carve ---------------------------------
{
  const cs = standardCase({});
  const brk = createBreakers({ layout: cs.sc.lay, wl: 0, beach: cs.beach, swell: cs.swell, meshSlots: 100 });
  let t = 0, frames = 0, n = 0;
  for (let f = 0; f < 100 && frames < 3; f++, t += DT) {
    brk.update(cs.sc.records(t), DT, { t, swell: cs.swell });
    if (brk.active.length) { frames++; n = brk.model.breakers.reduce((s, b) => s + b.n, 0); }
  }
  check('tubes are only carved under samples that own a curtain loop (100-loop pool, wider lip)', n > 100 && brk.tubes.length === brk.mesh.usedSlots && brk.tubes.length <= 100, `${n} samples, ${brk.mesh.usedSlots} loops, ${brk.tubes.length} tubes`);
  brk.dispose();
}

console.log(`\n${failures} failure(s)`);
process.exit(failures ? 1 : 0);
