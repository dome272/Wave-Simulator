// GPU solver benchmark: the real layout (r = 1 by default) over a plane beach
// with a sand bar, hot start from a one-component target, 50 steps.
// SwiftShader timings are only a relative measure. Run: node test/gpu-bench.mjs [--r 2] [--steps 50]
import { openPage, SERVER } from './headless.mjs';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? +process.argv[i + 1] : d; };
const r = arg('--r', 1), steps = arg('--steps', 50);
const { browser, page } = await openPage(`${SERVER}/test/pages/gpu-test.html`, { log: false });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
const t0 = Date.now();
const b = await page.evaluate(([r, steps]) => window.runCase('bench', { r, steps }), [r, steps]);
await browser.close();
if (b.error) { console.log('bench ERROR', b.error); process.exit(1); }
console.log(`bench r=${b.r}: grid ${b.nx} x ${b.ny} cells (${(b.nx * b.ny / 1e3).toFixed(0)}k), dt=${b.dt.toFixed(4)} s, ${b.steps} steps`);
console.log(`  ${b.msPerStep.toFixed(1)} ms/step on SwiftShader (${(1000 / b.msPerStep).toFixed(2)} steps/s), ${b.passesPerStep} passes/step (${b.pcrLevels} PCR levels), ${(b.msPerStep / b.passesPerStep).toFixed(1)} ms/pass`);
console.log(`  texture memory ${b.textureMB.toFixed(1)} MB; crest readback ${b.crestReadMs.toFixed(1)} ms (${b.crestRecords} records)`);
console.log(`  state after ${b.steps} steps: NaN=${b.nan}, max h ${b.hMax.toFixed(2)} m, maxSpeed ${b.maxSpeed.toFixed(2)} m/s, ${b.wetCells} wet cells; waves.js solve ${b.waveSolveMs.toFixed(0)} ms; wall ${((Date.now() - t0) / 1000).toFixed(1)} s`);
process.exit(b.nan ? 1 : 0);
