// Compiles and draws the curtain, its depth pre-pass and the spray materials
// under headless Chromium and prints the shader diagnostics (none expected).
// Needs the static server (see test/headless.mjs).
import { openPage, result } from './headless.mjs';
const { browser, page } = await openPage('http://localhost:8099/test/pages/lip-compile.html', { width: 320, height: 240, log: false });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
const out = await result(page, '__out', 120000);
const bad = out.errors.length || out.progs.some((p) => p.diag) || out.calls < 3;
console.log((bad ? "FAIL" : "PASS") + "  lip curtain, depth pre-pass and spray shaders compile and draw  " + JSON.stringify(out));
process.exitCode = bad ? 1 : 0;
await browser.close();
