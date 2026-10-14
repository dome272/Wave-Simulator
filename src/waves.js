// Wave solver: turns a seabed heightfield plus a swell description into the
// fields the renderer animates.
//
// Model (phase-resolved linear waves on a slowly varying seabed):
//  * The swell is split into 8 components spread in frequency and direction.
//    How wide that spread is depends on how far away the storm was: a distant
//    storm delivers clean, narrow-banded lines with long sets; a close one
//    delivers a broad, choppy sea.
//  * For each component the local wavenumber comes from the linear dispersion
//    relation w^2 = g k tanh(k h). The wave phase S(x, y) solves the eikonal
//    equation |grad S| = k, so crests bend toward shallow water (refraction),
//    and period matters because long waves feel the bottom earlier.
//  * Amplitude follows conservation of energy flux along rays: shoaling from
//    the group velocity, refraction from how rays converge or spread.
//  * A representative wave is marched with depth-limited saturation
//    (H <= gamma_s * h) to get how much energy survives the surf zone, so a
//    sandbar takes energy out of the waves that reach the inside.
//  * Breaker type comes from the surf similarity (Iribarren) number of the
//    local seabed slope against the offshore wave steepness.

export const G = 9.81;

const ZQ = [-1.534, -0.887, -0.489, -0.157, 0.157, 0.489, 0.887, 1.534];
const PERM = [3, 6, 0, 5, 2, 7, 1, 4];
const PHASES = [0.0, 2.4, 4.1, 1.3, 5.6, 3.3, 0.7, 4.9];
export const NCOMP = 8;

export function waveNumber(omega, h) {
  const k0 = (omega * omega) / G;
  const x = k0 * h;
  if (x > 12) return k0;
  // Fenton & McKee approximation, then one Newton step on kh*tanh(kh) = x.
  let kh = x / Math.pow(Math.tanh(Math.pow(x, 0.75)), 2 / 3);
  const t = Math.tanh(kh);
  const f = kh * t - x;
  const df = t + kh * (1 - t * t);
  kh -= f / df;
  return kh / h;
}

export function groupVelocity(omega, k, h) {
  const kh2 = 2 * k * h;
  const n = kh2 > 40 ? 0.5 : 0.5 * (1 + kh2 / Math.sinh(kh2));
  return (n * omega) / k;
}

// Swell description -> components. H is the significant wave height of the
// swell as it arrives in deep water offshore of the beach.
export function swellSpectrum(sw) {
  const R = Math.max(sw.distanceKm, 1);
  const bw = 0.035 + 0.13 * (200 / (R + 200)); // relative frequency spread
  const spread = ((4 + 26 * (300 / (R + 300))) * Math.PI) / 180; // direction spread
  const fp = 1 / sw.period;
  const th0 = (sw.direction * Math.PI) / 180;
  const a = sw.height / 8; // 8 equal-energy components => Hs = 4*sqrt(sum a^2/2)
  const comps = [];
  for (let m = 0; m < NCOMP; m++) {
    const f = fp * (1 + bw * ZQ[m]);
    comps.push({
      omega: 2 * Math.PI * f,
      theta: th0 + spread * 0.8 * ZQ[PERM[m]],
      a0: a,
      phase: PHASES[m],
    });
  }
  // Waves per set: beat between neighbouring bands of the spectrum.
  const wavesPerSet = Math.max(3, Math.round(1 / (bw * 1.25)));
  return { comps, bw, spread, wavesPerSet, rep: { omega: 2 * Math.PI * fp, theta: th0, a0: sw.height / 2 } };
}

// ---------------------------------------------------------------------------

function eikonal(N, dx, kk, kx, S, fixed) {
  // Fast sweeping, first-order Godunov upwind.
  const INF = 1e20;
  const sweeps = [
    [1, 1], [-1, 1], [1, -1], [-1, -1],
  ];
  for (let iter = 0; iter < 2; iter++) {
    for (const [di, dj] of sweeps) {
      const i0 = di > 0 ? 0 : N - 1, i1 = di > 0 ? N : -1;
      const j0 = dj > 0 ? 0 : N - 1, j1 = dj > 0 ? N : -1;
      for (let j = j0; j !== j1; j += dj) {
        const row = j * N;
        for (let i = i0; i !== i1; i += di) {
          const c = row + i;
          if (fixed[c]) continue;
          const a = Math.min(i > 0 ? S[c - 1] : INF, i < N - 1 ? S[c + 1] : INF);
          const b = Math.min(j > 0 ? S[c - N] : INF, j < N - 1 ? S[c + N] : INF);
          const f = kk[c] * dx;
          let s;
          if (Math.abs(a - b) >= f) s = Math.min(a, b) + f;
          else s = 0.5 * (a + b + Math.sqrt(2 * f * f - (a - b) * (a - b)));
          if (s < S[c]) S[c] = s;
        }
      }
    }
  }
}

function directions(N, dx, S, ux, uy) {
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const c = j * N + i;
      const gx = i === 0 ? S[c + 1] - S[c] : i === N - 1 ? S[c] - S[c - 1] : 0.5 * (S[c + 1] - S[c - 1]);
      const gy = j === 0 ? S[c + N] - S[c] : j === N - 1 ? S[c] - S[c - N] : 0.5 * (S[c + N] - S[c - N]);
      const l = Math.hypot(gx, gy) || 1;
      ux[c] = gx / l;
      uy[c] = gy / l;
    }
  }
}

function blur(N, src, tmp, passes) {
  for (let p = 0; p < passes; p++) {
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const c = j * N + i;
        const l = i > 0 ? c - 1 : c, r = i < N - 1 ? c + 1 : c;
        tmp[c] = 0.25 * src[l] + 0.5 * src[c] + 0.25 * src[r];
      }
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const c = j * N + i;
        const d = j > 0 ? c - N : c, u = j < N - 1 ? c + N : c;
        src[c] = 0.25 * tmp[d] + 0.5 * tmp[c] + 0.25 * tmp[u];
      }
  }
}

// March energy flux shoreward row by row along the wave direction field.
// Returns the local amplitude relative to the boundary amplitude. When
// `sat` is given ({H0, h, gamma}), the wave height is capped at gamma*h and
// the lost energy stays lost (surf-zone dissipation).
function transport(N, dx, ux, uy, div, cg, land, lambda, out, sat) {
  const lnF = new Float64Array(N);
  const prev = new Float64Array(N);
  const F = new Float64Array(N);
  let cgB = 0;
  for (let i = 0; i < N; i++) cgB += cg[i];
  cgB /= N;
  for (let i = 0; i < N; i++) {
    prev[i] = 0;
    out[i] = Math.sqrt(cgB / cg[i]);
  }
  for (let j = 1; j < N; j++) {
    const row = j * N;
    for (let i = 0; i < N; i++) {
      const c = row + i;
      if (land[c]) { lnF[i] = -30; continue; }
      const vy = Math.max(uy[c], 0.2);
      let xs = i - ux[c] / vy;
      if (xs < 0) xs = 0; else if (xs > N - 1) xs = N - 1;
      const i0 = Math.min(Math.floor(xs), N - 2), f = xs - i0;
      const up = prev[i0] * (1 - f) + prev[i0 + 1] * f;
      let g = (-div[c] * dx) / vy;
      if (g > 0.25) g = 0.25; else if (g < -0.25) g = -0.25;
      lnF[i] = up + g;
    }
    // Lateral spreading of energy, a stand-in for diffraction into shadows.
    for (let i = 0; i < N; i++) F[i] = Math.exp(lnF[i]);
    const alpha = Math.min(0.45, lambda[row + (N >> 1)] / (dx * 30));
    for (let i = 0; i < N; i++) {
      const c = row + i;
      if (land[c]) { prev[i] = -30; continue; }
      const l = i > 0 && !land[c - 1] ? F[i - 1] : F[i];
      const r = i < N - 1 && !land[c + 1] ? F[i + 1] : F[i];
      let Fi = F[i] * (1 - alpha) + 0.5 * alpha * (l + r);
      let amp = Math.sqrt((Fi * cgB) / cg[c]);
      if (sat) {
        const Hmax = sat.gamma * sat.h[c];
        if (amp * sat.H0 > Hmax) {
          amp = Hmax / sat.H0;
          Fi = (amp * amp * cg[c]) / cgB;
        }
      }
      out[c] = amp;
      prev[i] = Math.log(Math.max(Fi, 1e-13));
    }
  }
}

// Main entry. z: bed elevation (m), wl: still water level (m).
export function solve({ N, dx, z, wl, swell }) {
  const NN = N * N;
  const spec = swellSpectrum(swell);
  const h = new Float32Array(NN);
  const heff = new Float32Array(NN);
  const land = new Uint8Array(NN);
  for (let c = 0; c < NN; c++) {
    h[c] = wl - z[c];
    heff[c] = Math.max(h[c], 0.08);
    land[c] = h[c] < 0.03 ? 1 : 0;
  }

  const kk = new Float32Array(NN);
  const cg = new Float32Array(NN);
  const lambda = new Float32Array(NN);
  const S = new Float32Array(NN);
  const fixed = new Uint8Array(NN);
  const ux = new Float32Array(NN), uy = new Float32Array(NN);
  const div = new Float32Array(NN), tmp = new Float32Array(NN);
  const amp = new Float32Array(NN);

  const outS = [], outA = [];
  let repUx, repUy, repAmp, repK, repCg;

  const one = (comp, isRep) => {
    const { omega, theta } = comp;
    const k0 = (omega * omega) / G;
    const kx = k0 * Math.sin(theta);
    for (let c = 0; c < NN; c++) {
      let k = waveNumber(omega, heff[c]);
      if (land[c]) k *= 6;
      kk[c] = k;
      cg[c] = groupVelocity(omega, k, heff[c]);
      lambda[c] = (2 * Math.PI) / k;
    }
    // Boundary phase: plane wave on the offshore row, Snell's law up the
    // upwind side edge.
    S.fill(1e20);
    fixed.fill(0);
    for (let i = 0; i < N; i++) { S[i] = kx * i * dx; fixed[i] = 1; }
    if (Math.abs(kx) > 1e-6) {
      const ie = kx > 0 ? 0 : N - 1;
      for (let j = 1; j < N; j++) {
        const c = j * N + ie, p = c - N;
        const ky1 = Math.sqrt(Math.max(kk[p] * kk[p] - kx * kx, 0.01 * kk[p] * kk[p]));
        const ky2 = Math.sqrt(Math.max(kk[c] * kk[c] - kx * kx, 0.01 * kk[c] * kk[c]));
        S[c] = S[p] + 0.5 * (ky1 + ky2) * dx;
        fixed[c] = 1;
      }
    }
    eikonal(N, dx, kk, kx, S, fixed);
    directions(N, dx, S, ux, uy);
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const c = j * N + i;
        const dxu = i === 0 ? ux[c + 1] - ux[c] : i === N - 1 ? ux[c] - ux[c - 1] : 0.5 * (ux[c + 1] - ux[c - 1]);
        const dyu = j === 0 ? uy[c + N] - uy[c] : j === N - 1 ? uy[c] - uy[c - N] : 0.5 * (uy[c + N] - uy[c - N]);
        div[c] = land[c] ? 0 : (dxu + dyu) / dx;
      }
    blur(N, div, tmp, 2);

    // Boundary amplitude: deep water -> offshore row (shoaling + refraction
    // over straight contours).
    let cgB = 0, kB = 0;
    for (let i = 0; i < N; i++) { cgB += cg[i]; kB += kk[i]; }
    cgB /= N; kB /= N;
    const cg0 = (0.5 * G) / omega;
    const sinB = Math.min(Math.abs(kx) / kB, 0.99);
    const Ks = Math.sqrt(cg0 / cgB);
    const Kr = Math.sqrt(Math.cos(theta) / Math.sqrt(1 - sinB * sinB));
    const aB = comp.a0 * Ks * Kr;

    if (isRep) {
      // Undissipated and depth-limited versions of the representative wave.
      const und = new Float32Array(NN);
      transport(N, dx, ux, uy, div, cg, land, lambda, und, null);
      const sat = new Float32Array(NN);
      transport(N, dx, ux, uy, div, cg, land, lambda, sat, { H0: 2 * aB, h: heff, gamma: 0.68 });
      repAmp = { und, sat, aB };
      repUx = ux.slice(); repUy = uy.slice();
      repK = kk.slice(); repCg = cg.slice();
    } else {
      transport(N, dx, ux, uy, div, cg, land, lambda, amp, null);
      const A = new Float32Array(NN);
      for (let c = 0; c < NN; c++) A[c] = land[c] ? 0 : amp[c] * aB;
      outA.push(A);
      outS.push(S.slice());
    }
  };

  one(spec.rep, true);
  for (const comp of spec.comps) one(comp, false);

  // Per-cell breaking parameters.
  const D = new Float32Array(NN), gamma = new Float32Array(NN);
  const plunge = new Float32Array(NN), surge = new Float32Array(NN);
  const slope = new Float32Array(NN), xi = new Float32Array(NN);
  const hs = new Float32Array(NN), speed = new Float32Array(NN);
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const c = j * N + i;
      const gx = i === 0 ? z[c + 1] - z[c] : i === N - 1 ? z[c] - z[c - 1] : 0.5 * (z[c + 1] - z[c - 1]);
      const gy = j === 0 ? z[c + N] - z[c] : j === N - 1 ? z[c] - z[c - N] : 0.5 * (z[c + N] - z[c - N]);
      slope[c] = Math.max(0, (gx * repUx[c] + gy * repUy[c]) / dx);
    }
  blur(N, slope, tmp, 4);
  const L0 = (2 * Math.PI * G) / (spec.rep.omega * spec.rep.omega);
  const steep0 = Math.sqrt(Math.max(swell.height, 0.05) / L0);
  for (let c = 0; c < NN; c++) {
    const u = repAmp.und[c], s = repAmp.sat[c];
    D[c] = u > 1e-6 ? Math.min(1, s / u) : 0;
    gamma[c] = Math.min(1.15, 0.78 + 1.6 * slope[c]);
    const x = slope[c] / steep0;
    xi[c] = x;
    plunge[c] = smooth(0.3, 0.75, x) * (1 - smooth(2.2, 3.6, x));
    surge[c] = smooth(2.2, 3.6, x);
    hs[c] = land[c] ? 0 : s * 2 * repAmp.aB;
    speed[c] = land[c] ? 0 : spec.rep.omega / repK[c];
  }

  // Summary for the readout: where and how the waves first break.
  const summary = summarize(N, dx, h, land, repAmp, gamma, xi, spec, L0, swell);

  return {
    S: outS, A: outA, h, D, gamma, plunge, surge, ux: repUx, uy: repUy, speed, hs, xi,
    omegas: spec.comps.map((c) => c.omega),
    phases: spec.comps.map((c) => c.phase),
    summary,
  };
}

function smooth(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function summarize(N, dx, h, land, rep, gamma, xi, spec, L0, swell) {
  // For each alongshore column, walk shoreward to the first cell where the
  // biggest waves of a set (about 1.6 Hs) reach the breaking limit.
  let nB = 0, sumDepth = 0, sumXi = 0, sumDist = 0;
  for (let i = 4; i < N - 4; i += 2) {
    let shore = -1;
    for (let j = 0; j < N; j++) if (land[j * N + i]) { shore = j; break; }
    for (let j = 0; j < N; j++) {
      const c = j * N + i;
      if (land[c]) break;
      const Hbig = 1.6 * rep.und[c] * 2 * rep.aB;
      if (Hbig >= gamma[c] * h[c]) {
        // Weight by wave energy so the main peak decides the verdict.
        const wgt = Hbig * Hbig;
        nB += wgt;
        sumDepth += wgt * h[c];
        sumXi += wgt * xi[c];
        if (shore > 0) sumDist += wgt * (shore - j) * dx;
        break;
      }
    }
  }
  const depth = nB ? sumDepth / nB : 0;
  const xiAvg = nB ? sumXi / nB : 0;
  const type = xiAvg < 0.4 ? 'Spilling' : xiAvg < 2.4 ? 'Plunging' : 'Surging';
  return {
    L0, c0: L0 / swell.period, breakDepth: depth, breakDist: nB ? sumDist / nB : 0,
    xi: xiAvg, type, wavesPerSet: spec.wavesPerSet,
    spreadDeg: (spec.spread * 180) / Math.PI, bw: spec.bw,
  };
}
