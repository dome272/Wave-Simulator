// Node test battery for the reference Boussinesq solver. Run: node test/boussinesq.test.mjs
import { createSolver, omegaMS, wavenumberMS, wavenumberAiry, relaxField, G } from '../src/boussinesq.js';

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  (' + detail + ')' : ''}`);
  if (!ok) failures++;
}
const corners = (nx, ny, dx, fn) => { const b = new Float64Array((nx + 1) * (ny + 1)); for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) b[j * (nx + 1) + i] = fn(i * dx, j * dx, i, j); return b; };
const maxAbs = (a) => { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; };
const hasNaN = (a) => { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return true; return false; };

// 1. Lake at rest over an irregular bed with an island: nothing may move.
{
  const nx = 48, ny = 40, dx = 2.5;
  const bed = corners(nx, ny, dx, (x, y) => -8 + 3 * Math.sin(x / 17) * Math.cos(y / 23) + 7 * Math.exp(-((x - 60) ** 2 + (y - 50) ** 2) / 300) + 0.3 * y / 10);
  const s = createSolver({ nx, ny, dx, bed, wl: 0 });
  const eta0 = s.eta();
  for (let n = 0; n < 200; n++) s.step(0.05);
  const eta1 = s.eta();
  let de = 0, umax = 0; for (let c = 0; c < nx * ny; c++) if (s.h[c] > 1e-4) { de = Math.max(de, Math.abs(eta1[c] - eta0[c])); umax = Math.max(umax, Math.hypot(s.P[c], s.Q[c]) / s.h[c]); }
  // Fully wet cells must be exactly still; partially wet shoreline cells may carry tiny spurious currents.
  check('lake at rest stays still (island, sloping bed)', umax < 0.02 && de < 0.01 && !hasNaN(s.h), `max|u|=${umax.toExponential(2)} m/s max|deta|=${de.toExponential(2)} m`);
  const bedFlat = corners(nx, ny, dx, (x, y) => -8 + 3 * Math.sin(x / 17) * Math.cos(y / 23) + 0.3 * y / 10);
  const s2 = createSolver({ nx, ny, dx, bed: bedFlat, wl: 0 });
  for (let n = 0; n < 200; n++) s2.step(0.05);
  check('lake at rest is exact when fully wet', maxAbs(s2.P) < 1e-9 && maxAbs(s2.Q) < 1e-9, `max|P|=${maxAbs(s2.P).toExponential(2)}`);
}

// 2. Closed basin sloshing with wet/dry: volume conserved, no NaN.
{
  const nx = 60, ny = 50, dx = 2;
  const bed = corners(nx, ny, dx, (x, y, i, j) => -4 + 0.08 * Math.hypot(i - 30, j - 25) ** 1.5 * 0.1);
  const eta = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) eta[j * nx + i] = 0.8 * Math.exp(-((i - 20) ** 2 + (j - 25) ** 2) / 30);
  const s = createSolver({ nx, ny, dx, bed, wl: 0, eta });
  const v0 = s.volume();
  let dtMin = 1;
  for (let n = 0; n < 400; n++) { const dt = Math.min(0.05, s.cfl(0.4)); dtMin = Math.min(dtMin, dt); s.step(dt); }
  const v1 = s.volume();
  check('closed basin conserves volume', Math.abs(v1 - v0) / v0 < 1e-9 && !hasNaN(s.h) && !hasNaN(s.P), `rel dV=${((v1 - v0) / v0).toExponential(2)} dtMin=${dtMin.toFixed(3)}`);
}

// 3. Linear dispersion on flat depth: a periodic wave must travel at the Madsen-Sorensen speed.
{
  const d = 10, T = 8, w = (2 * Math.PI) / T;
  const kMS = wavenumberMS(w, d), kA = wavenumberAiry(w, d);
  const nWaves = 4, nx = 256, ny = 4;
  const Lx = (nWaves * 2 * Math.PI) / kMS, dx = Lx / nx;
  const bed = new Float64Array((nx + 1) * (ny + 1)).fill(-d);
  const a = 0.05;
  const eta = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) eta[j * nx + i] = a * Math.cos(kMS * (i + 0.5) * dx);
  const run = (dispersion) => {
    const s = createSolver({ nx, ny, dx, bed, wl: 0, eta, periodicX: true, periodicY: true, dispersion, breaking: false, manning: 0 });
    const c = w / kMS;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) s.P[j * nx + i] = c * eta[j * nx + i];
    const dt = 0.5 * dx / Math.sqrt(G * d);
    const tEnd = 2 * T; let t = 0;
    while (t < tEnd - 1e-9) { const h = Math.min(dt, tEnd - t); s.step(h); t += h; }
    // Fit the phase: project the middle row onto cos/sin.
    let cc = 0, ss = 0; const e = s.eta();
    for (let i = 0; i < nx; i++) { const x = (i + 0.5) * dx; cc += e[2 * nx + i] * Math.cos(kMS * x); ss += e[2 * nx + i] * Math.sin(kMS * x); }
    const phase = Math.atan2(ss, cc); // eta = a cos(kx - phase) => phase = k c t mod 2pi
    const amp = (2 / nx) * Math.hypot(cc, ss);
    const cMeasured = (phase + 2 * Math.PI * Math.round((w * tEnd - phase) / (2 * Math.PI))) / (kMS * tEnd);
    return { cMeasured, amp };
  };
  const cMS = w / kMS, cA = w / kA, cSW = Math.sqrt(G * d);
  const withD = run(true), without = run(false);
  console.log(`   kd=${(kMS * d).toFixed(2)} c_MS=${cMS.toFixed(3)} c_Airy=${cA.toFixed(3)} c_SWE=${cSW.toFixed(3)} | measured with dispersion ${withD.cMeasured.toFixed(3)} (amp ${withD.amp.toFixed(4)}), without ${without.cMeasured.toFixed(3)}`);
  check('dispersion: phase speed matches Madsen-Sorensen within 1.5%', Math.abs(withD.cMeasured - cMS) / cMS < 0.015 && Math.abs(withD.amp - a) / a < 0.08);
  check('dispersion off: phase speed is the shallow-water speed', Math.abs(without.cMeasured - cSW) / cSW < 0.03);
}

// 4. Solitary wave keeps its shape and speed on a flat bottom.
{
  const d = 10, a = 1.0, c = Math.sqrt(G * (d + a)), kap = Math.sqrt((3 * a) / (4 * d * d * d));
  const nx = 512, ny = 4, dx = 1.0;
  const bed = new Float64Array((nx + 1) * (ny + 1)).fill(-d);
  const eta = new Float64Array(nx * ny);
  const x0 = 100;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) eta[j * nx + i] = a / Math.cosh(kap * (i * dx - x0)) ** 2;
  const s = createSolver({ nx, ny, dx, bed, wl: 0, eta, periodicX: true, periodicY: true, breaking: false, manning: 0 });
  for (let c2 = 0; c2 < nx * ny; c2++) s.P[c2] = c * eta[c2] * d / (d + eta[c2]);
  const dt = 0.4 * dx / (Math.sqrt(G * d) + 5);
  const tEnd = 25; let t = 0;
  while (t < tEnd - 1e-9) { const h = Math.min(dt, tEnd - t); s.step(h); t += h; }
  const e = s.eta(); let pk = -1, ipk = 0;
  for (let i = 0; i < nx; i++) if (e[2 * nx + i] > pk) { pk = e[2 * nx + i]; ipk = i; }
  const cMeas = (ipk * dx - x0) / tEnd;
  console.log(`   solitary a/d=${a / d}: peak ${pk.toFixed(3)} m (init ${a}), speed ${cMeas.toFixed(3)} vs ${c.toFixed(3)}`);
  check('solitary wave keeps amplitude within 12% and speed within 4%', Math.abs(pk - a) / a < 0.12 && Math.abs(cMeas - c) / c < 0.04);
}

// 5. Shoaling and breaking on a 1:30 plane beach, with run-up.
{
  const nx = 4, ny = 480, dx = 1.25;
  const d0 = 8, slope = 1 / 30, flat = 120; // flat part then the slope
  const bed = corners(nx, ny, dx, (x, y) => (y < flat ? -d0 : -d0 + (y - flat) * slope));
  const relax = relaxField(nx, ny, { offshore: 48 });
  const H = 1.2, T = 10, w = (2 * Math.PI) / T, k = wavenumberMS(w, d0), c = w / k;
  const target = (t, etaT, PT, QT) => {
    const ramp = Math.min(1, t / (2 * T));
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const cc = j * nx + i; const y = (j + 0.5) * dx;
      const e = ramp * 0.5 * H * Math.cos(k * y - w * t);
      etaT[cc] = e; PT[cc] = 0; QT[cc] = c * e;
    }
  };
  const s = createSolver({ nx, ny, dx, bed, wl: 0, periodicX: true, relax, target, manning: 0.02 });
  const dt = 0.03;
  const nT = 10; const tEnd = nT * T;
  const hi = new Float64Array(ny).fill(-1e9), lo = new Float64Array(ny).fill(1e9);
  let shoreMax = 0, shoreMin = 1e9, fAny = false, nanFound = false;
  const fMax = new Float64Array(ny);
  let t = 0;
  const e = new Float64Array(nx * ny);
  while (t < tEnd - 1e-9) {
    s.step(dt); t += dt;
    if (hasNaN(s.h)) { nanFound = true; break; }
    if (t > (nT - 3) * T) {
      s.eta(e);
      for (let j = 0; j < ny; j++) { const v = e[j * nx + 1]; if (s.h[j * nx + 1] > 1e-3) { hi[j] = Math.max(hi[j], v); lo[j] = Math.min(lo[j], v); } }
      let js = ny - 1; for (let j = 0; j < ny; j++) if (s.h[j * nx + 1] <= 1e-3) { js = j; break; }
      shoreMax = Math.max(shoreMax, js); shoreMin = Math.min(shoreMin, js);
      for (let j = 0; j < ny; j++) { if (s.F[j * nx + 1] > 0.5) fAny = true; fMax[j] = Math.max(fMax[j], s.F[j * nx + 1]); }
    }
  }
  // Wave height profile and first breaking point.
  let jb = -1; for (let j = 0; j < ny; j++) if (fMax[j] > 0.99) { jb = j; break; }
  const Hprof = []; for (let j = 0; j < ny; j += 20) Hprof.push(hi[j] > -1e8 ? (hi[j] - lo[j]).toFixed(2) : '-');
  console.log(`   H(y) every 25 m: ${Hprof.join(' ')}`);
  // Find the break point as the first j (shoreward) where H starts to drop after its maximum.
  let jmax = 0, hmax = 0; for (let j = 0; j < ny; j++) if (hi[j] > -1e8 && hi[j] - lo[j] > hmax) { hmax = hi[j] - lo[j]; jmax = j; }
  const hb = -s.B[jmax * nx + 1];
  const hbf = jb >= 0 ? -s.B[jb * nx + 1] : NaN, Hbf = jb >= 0 ? hi[jb] - lo[jb] : NaN;
  const swash = (shoreMax - shoreMin) * dx * slope;
  console.log(`   max H ${hmax.toFixed(2)} m at depth ${hb.toFixed(2)} m (H/h=${(hmax / hb).toFixed(2)}); first flagged breaking at depth ${hbf.toFixed(2)} m with H=${Hbf.toFixed(2)} (H/h=${(Hbf / hbf).toFixed(2)}); shoreline excursion ${((shoreMax - shoreMin) * dx).toFixed(1)} m (${swash.toFixed(2)} m vertical)`);
  check('plane beach: stable, breaks at H/h in 0.55..1.1, runs up the sand', !nanFound && fAny && Hbf / hbf > 0.55 && Hbf / hbf < 1.1 && swash > 0.2 && hmax > 1.0);
}

// 6. Dam break on a dry bed (Ritter): front speed and depth at the dam.
{
  const nx = 400, ny = 4, dx = 0.5, h0 = 1;
  const bed = new Float64Array((nx + 1) * (ny + 1)).fill(-h0);
  const eta = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) eta[j * nx + i] = (i + 0.5) * dx < 100 ? 0 : -h0; // dry beyond x=100
  const s = createSolver({ nx, ny, dx, bed, wl: 0, eta, periodicY: true, dispersion: false, breaking: false, manning: 0 });
  const tEnd = 5; let t = 0;
  while (t < tEnd - 1e-9) { const dt = Math.min(0.02, s.cfl(0.4), tEnd - t); s.step(dt); t += dt; }
  // Compare the position of the 5 cm depth contour (the desingularisation depth) with Ritter's profile.
  let front = 0; for (let i = 0; i < nx; i++) if (s.h[1 * nx + i] > 0.05) front = (i + 0.5) * dx;
  const hDam = s.h[1 * nx + 200];
  const frontRitter = 100 + (2 * Math.sqrt(G * h0) - 3 * Math.sqrt(G * 0.05)) * tEnd; // Ritter: h = (2 sqrt(g h0) - x/t)^2 / 9g
  console.log(`   dam break: 5 cm front ${front.toFixed(1)} m (Ritter ${frontRitter.toFixed(1)}), depth at dam ${hDam.toFixed(3)} (Ritter ${(4 / 9 * h0).toFixed(3)})`);
  check('dam break on dry bed matches Ritter within 10%', Math.abs(front - frontRitter) / (frontRitter - 100) < 0.10 && Math.abs(hDam - 4 / 9 * h0) < 0.03);
}

console.log(failures ? `\n${failures} test(s) failed` : '\nall tests passed');
process.exit(failures ? 1 : 0);
