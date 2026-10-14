// Shared headless-browser harness for the GPU tests and the eye-level render
// checks. Launches Chromium with SwiftShader (software WebGL2 with float
// render targets), serves three.js 0.160 from a local copy in place of the
// jsDelivr URLs used by the import map, and blocks web fonts.
//
// Usage:
//   import { openPage } from './headless.mjs';
//   const { browser, page } = await openPage('http://localhost:8099/test/pages/gpu-test.html');
//   ...; await browser.close();
//
// The static server: `cd <repo> && npx http-server -p 8099 -s -c-1 .`
import fs from 'node:fs';
import path from 'node:path';

const PLAYWRIGHT = process.env.PLAYWRIGHT_MODULE || '/opt/node22/lib/node_modules/playwright/index.mjs';
const THREE_DIR = process.env.THREE_VENDOR || '/home/claude/three-0.160';
export const SERVER = process.env.TEST_SERVER || 'http://localhost:8099';

export async function openPage(url, { width = 1280, height = 720, log = true, timeout = 120000 } = {}) {
  const { chromium } = await import(PLAYWRIGHT);
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width, height } });
  if (log) {
    page.on('pageerror', (e) => console.log('pageerror:', e.message));
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('console:', m.text().slice(0, 300)); });
  }
  await page.route('https://cdn.jsdelivr.net/npm/three@0.160.0/**', (r) => {
    const rel = r.request().url().split('three@0.160.0/')[1].split('?')[0];
    const file = path.join(THREE_DIR, rel);
    if (!fs.existsSync(file)) return r.fulfill({ status: 404, body: '' });
    r.fulfill({ body: fs.readFileSync(file), contentType: 'application/javascript' });
  });
  await page.route('https://fonts.googleapis.com/**', (r) => r.fulfill({ body: '', contentType: 'text/css' }));
  await page.route('https://fonts.gstatic.com/**', (r) => r.fulfill({ body: '', contentType: 'font/woff2' }));
  await page.goto(url, { timeout });
  return { browser, page };
}

// Wait for `window.<name>` to be set by the page and return it.
export async function result(page, name = '__out', timeout = 300000) {
  await page.waitForFunction((n) => window[n] !== undefined, name, { timeout });
  return page.evaluate((n) => window[n], name);
}
