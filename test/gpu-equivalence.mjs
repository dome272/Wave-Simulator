// GPU-vs-reference equivalence (section 8 of docs/engine-contract.md).
// Runs the same cases on src/boussinesq.js (double precision, in the page) and
// on the GPU solver (test/pages/gpu-test.html, SwiftShader WebGL2), both from
// the same initial state and dt, and compares h, P, Q, F after 60 steps.
// Also checks the layout helpers in Node. Run: node test/gpu-equivalence.mjs
// (needs the static server on port 8099, see test/headless.mjs).
import { openPage, SERVER } from './headless.mjs';
import { layout, cornerBed, cornerManning, cellToGrid, gridToCell, sampleNodes } from '../src/engine/layout.js';
import { N, DX } from '../src/terrain.js';

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  (' + detail + ')' : ''}`);
  if (!ok) failures++;
}
const e = (v) => (typeof v === 'number' ? v.toExponential(2) : String(v));

// --- layout (Node, no WebGL) ---------------------------------------------------------
{
  const l1 = layout(1), l2 = layout(2);
  check('layout(1) = 335 x 377 (ML 40, MO 120, MI 2)', l1.nx === 335 && l1.ny === 377 && l1.ML === 40 && l1.MO === 120 && l1.MI === 2 && l1.dxs === 2.5);
  check('layout(2) = 670 x 754 (ML 80, MO 240, MI 4)', l2.nx === 670 && l2.ny === 754 && l2.ML === 80 && l2.MO === 240 && l2.MI === 4 && l2.dxs === 1.25);
  const z = new Float32Array(N * N), rock = new Float32Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { z[j * N + i] = 0.01 * i * DX - 0.02 * j * DX + 3 * Math.sin(i * 0.3) * Math.cos(j * 0.2); rock[j * N + i] = i > 100 && j > 150 ? 1 : 0; }
  const beach = { z, rock };
  let worst = 0, worstM = 0;
  for (const r of [1, 2]) {
    const lay = layout(r), b = cornerBed(beach, 0, r), m = cornerManning(beach, r), sx = lay.nx + 1;
    for (let J = 0; J <= lay.ny; J += 7) for (let I = 0; I <= lay.nx; I += 5) {
      const gx = Math.min(N - 1, Math.max(0, (I - lay.ML) * lay.dxs / DX)), gy = Math.min(N - 1, Math.max(0, (J - lay.MO) * lay.dxs / DX));
      const ex = sampleNodes(z, gx, gy);
      worst = Math.max(worst, Math.abs(b[J * sx + I] - ex));
      const i = Math.round(gx), j = Math.round(gy);
      const em = rock[j * N + i] >= 0.5 ? 0.035 : 0.02;
      worstM = Math.max(worstM, Math.abs(m[J * sx + I] - em));
    }
    // node values are hit exactly at r = 1 inside the grid
    if (r === 1) { const I = lay.ML + 17, J = lay.MO + 33; worst = Math.max(worst, Math.abs(b[J * sx + I] - z[33 * N + 17])); }
    // margins continue the edge values
    const Jo = 3, I0 = lay.ML + 50; worst = Math.max(worst, Math.abs(b[Jo * sx + I0] - b[lay.MO * sx + I0]));
  }
  check('cornerBed is the clamped bilinear interpolation of beach.z (r = 1, 2)', worst < 1e-6, `max err ${e(worst)}`);
  check('cornerManning: 0.035 on rock, 0.02 on sand', worstM < 1e-6, `max err ${e(worstM)}`);
  const lay = layout(1);
  const [x, y] = cellToGrid(lay, lay.ML, lay.MO), [I, J] = gridToCell(lay, 100, 200);
  check('cellToGrid / gridToCell', x === 1.25 && y === 1.25 && Math.abs(I - (100 / 2.5 + 40 - 0.5)) < 1e-12 && Math.abs(J - (200 / 2.5 + 120 - 0.5)) < 1e-12);
}

// --- GPU cases ----------------------------------------------------------------------------
const { browser, page } = await openPage(`${SERVER}/test/pages/gpu-test.html`, { log: false });
page.on('pageerror', (er) => console.log('pageerror:', er.message));
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
const run = (name, opts) => page.evaluate(([n, o]) => window.runCase(n, o), [name, opts]);
const fmt = (c) => `dh=${e(c.dh)} dP=${e(c.dP)} dQ=${e(c.dQ)} dF=${e(c.dF)}`;
const conv = (r) => '   steps -> ' + r.conv.map((c) => `${c.n}: ${fmt(c)}`).join(' | ');
const TOL = { dh: 2e-3, dP: 2e-2, dQ: 2e-2, dF: 1e-3 };
const within = (c) => c.dh <= TOL.dh && c.dP <= TOL.dP && c.dQ <= TOL.dQ && !c.nan;

// 1. still water with an island: wet/dry balance, everything stays at 1e-5
{
  const r = await run('still');
  if (r.error) check('still water (island)', false, r.error.split('\n')[0]);
  else {
    const c = r.final;
    console.log(conv(r));
    check(`still water (island, ${r.nx}x${r.ny}, ${r.steps} steps): GPU == reference`, c.dh <= 1e-5 && c.dP <= 5e-5 && c.dQ <= 5e-5 && !c.nan, fmt(c));
    check('still water (island): GPU stays still', r.selfDh <= 1e-5 && r.selfMaxPQ <= 5e-5, `drift |dh|=${e(r.selfDh)} max|P|,|Q|=${e(r.selfMaxPQ)}`);
  }
}
// 2. solitary wave on a flat bed
{
  const r = await run('solitary');
  if (r.error) check('solitary wave', false, r.error.split('\n')[0]);
  else {
    const c = r.final;
    console.log(conv(r));
    check(`solitary wave (a=1 m, d=10 m, ${r.nx} cells, ${r.steps} steps): |dh|<=2e-3, |dP|,|dQ|<=2e-2`, within(c), fmt(c));
    check('solitary wave: F identical (no breaking)', c.dF === 0 && c.maxF === 0, `peak ref ${r.refPeak.eta.toFixed(4)} m @ ${r.refPeak.x} m, gpu ${r.gpuPeak.eta.toFixed(4)} m @ ${r.gpuPeak.x} m`);
  }
}
// 3. plane beach with relaxation, breaking and run-up
{
  const r = await run('beach');
  if (r.error) check('plane beach', false, r.error.split('\n')[0]);
  else {
    const c = r.final;
    console.log(conv(r));
    check(`plane beach (1:30, relaxation + hot start at t=${r.t0} s, ${r.steps} steps): |dh|<=2e-3, |dP|,|dQ|<=2e-2`, within(c), fmt(c));
    check('plane beach: F equal where the reference has F = 0 or 1 away from the 3-cell ramp', c.dFclean <= TOL.dF && c.flips === 0, `max|dF| clean=${e(c.dFclean)} all=${e(c.dF)} decision flips=${c.flips}; reference F: ${r.refF.ones} cells =1, ${r.refF.mid} intermediate, ${r.refF.zeros} =0`);
    check('plane beach: crest scan matches the CPU scan', r.crest.slotMismatch === 0 && r.crest.maxDiff <= 1e-4, `${r.crest.records} records, max diff ${e(r.crest.maxDiff)}`);
    check('plane beach: maxSpeed reduction', Math.abs(r.maxSpeed.gpu - r.maxSpeed.cpu) <= 1e-4 * r.maxSpeed.cpu, `gpu ${r.maxSpeed.gpu.toFixed(4)} cpu ${r.maxSpeed.cpu.toFixed(4)} m/s`);
  }
}
// 3b. bathymetry update
{
  const r = await run('rebase');
  if (r.error) check('setBed rebase', false, r.error.split('\n')[0]);
  else check('setBed keeps the surface where wet, dries/floods correctly', r.surfaceKept <= 1e-5 && r.badDry === 0 && r.bad === 0 && r.floodErr <= 1e-5 && r.driedCells > 0 && r.floodedCells > 0, `|dW|=${e(r.surfaceKept)} dried ${r.driedCells} flooded ${r.floodedCells} floodErr=${e(r.floodErr)}`);
}
// 4. target field on the real layout
{
  const r = await run('target');
  if (r.error) check('target field', false, r.error.split('\n')[0]);
  else {
    for (const mode of ['single', 'all']) {
      const t = r[mode];
      check(`target field (r=1, flat bed, ${mode === 'single' ? '1 component' : '8 components'}): eta within 1e-4 m of the component sum`, t.dEta <= 1e-4 && t.dP <= 1e-3 && t.dQ <= 1e-3, `dEta=${e(t.dEta)} (visible ${e(t.dEtaVisible)}, margins ${e(t.dEtaMargin)}) dP=${e(t.dP)} dQ=${e(t.dQ)} max|eta|=${t.maxEta.toFixed(3)}`);
    }
    check('hotStart sets h = wl + eta - B, P = PT, Q = QT', r.single.hotStartDiff <= 1e-5 && r.all.hotStartDiff <= 1e-5, `diff ${e(r.all.hotStartDiff)}`);
    check('one step on the real layout (335x377) is finite', !r.stepNaN, `${r.passesPerStep} passes, ${r.stepMs.toFixed(0)} ms on SwiftShader incl. readback, maxSpeed ${r.maxSpeedAfterStep.toFixed(2)} m/s`);
  }
}
await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
