// Headless eye-level render check of the breakers package.
// Run: /opt/node22/bin/node test/breakers-demo.mjs   (needs the static server on :8099)
// Steps the synthetic crest of test/pages/breakers-demo.html with dt = 1/30 and
// saves screenshots at four moments (just before the first lip throws, mid-throw with
// the tube open, impact with splash-up, collapsed) from the beach-level
// camera, from inside the tube and from the lineup, to test/out/breakers-*.png.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPage, SERVER } from './headless.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, 'out');
fs.mkdirSync(outDir, { recursive: true });
const query = process.argv[2] || '';
const views = (process.env.VIEWS || 'beach,crouch,low,paddle,impact,shoulder,lineup,side').split(',');
const moments = (process.env.MOMENTS || 'before,mid,impact,collapsed').split(',');

const { browser, page } = await openPage(`${SERVER}/test/pages/breakers-demo.html${query}`, { width: 1280, height: 720 });
await page.waitForFunction(() => window.__demo && window.__demo.ready, null, { timeout: 120000 });
const t0 = Date.now();
const report = [];
for (const m of moments) {
  const info = await page.evaluate((m) => window.__demo.stepUntil(m), m);
  console.log(JSON.stringify(info));
  report.push(info);
  for (const v of views) {
    const vi = await page.evaluate((v) => window.__demo.view(v), v);
    const file = path.join(outDir, `breakers-${m}-${v}.png`);
    await page.screenshot({ path: file });
    console.log(`  ${v}: ${file} cam ${JSON.stringify(vi.cam)}`);
  }
}
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
fs.writeFileSync(path.join(outDir, 'breakers-demo.json'), JSON.stringify(report, null, 1));
await browser.close();
