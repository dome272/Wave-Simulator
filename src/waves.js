// Wave solver: turns a seabed heightfield plus a swell description into the
// fields the renderer animates.
//
// Model (phase-resolved linear waves on a slowly varying seabed):
//  * The swell is one dominant wave train plus 7 minor components spread in
//    frequency and direction. How much energy sits in the minors, and how
//    wide their spread is, depends on how far away the storm was: a distant
//    storm delivers clean, narrow-banded lines; a close one a broad, choppy
//    sea. "Wave height" is the trough-to-crest height of the typical wave in
//    the dominant train; sets run about a third bigger.
//  * For each component the local wavenumber comes from the linear dispersion
//    relation w^2 = g k tanh(k h). The wave phase S(x, y) solves the eikonal
//    equation |grad S| = k, so crests bend toward shallow water (refraction),
//    and period matters because long waves feel the bottom earlier.
//  * Amplitude follows conservation of energy flux along rays: shoaling from
//    the group velocity, refraction from how rays converge or spread.
//  * Sets: a slow modulation of the wave height that travels shoreward at the
//    group velocity. The group travel time field solves |grad tau| = 1 / cg.
//  * The dominant wave is marched with depth-limited saturation
//    (H <= gamma_s * h) to get how much energy survives the surf zone, so a
//    sandbar takes energy out of the waves that reach the inside.
//  * Breaker type comes from the surf similarity (Iribarren) number of the
//    local seabed slope against the offshore wave steepness.

export const G = 9.81;

// Frequency and direction offsets of the 7 minor components, in units of the
// spread, and the fixed starting phases of all 8.
const ZF = [-1.4, -0.9, -0.45, 0, 0.45, 0.9, 1.4];
const ZD = [0.6, -1.3, 1.3, -0.5, 0.2, -0.9, 0.9];
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

// Swell description -> components. The first component is the dominant
// train; the rest add frequency and direction texture.
export function swellSpectrum(sw) {
  const R = Math.max(sw.distanceKm, 1);
  const near = 200 / (R + 200); // 0 for a distant storm, 0.8 for one 50 km away
  const bw = 0.03 + 0.09 * near; // relative frequency spread of the minors
  const spread = ((4 + 24 * near) * Math.PI) / 180; // direction spread
  const eDom = 0.78 - 0.42 * near; // share of the energy in the dominant train
  const wavesPerSet = Math.round(12 - 7 * near);
  const fp = 1 / sw.period;
  const th0 = (sw.direction * Math.PI) / 180;
  const a = sw.height / 2;
  const comps = [{ omega: 2 * Math.PI * fp, theta: th0, a0: a * Math.sqrt(eDom), phase: PHASES[0] }];
  const wm = Math.sqrt((1 - eDom) / 7);
  for (let m = 0; m < 7; m++) {
    comps.push({
      omega: 2 * Math.PI * fp * (1 + bw * ZF[m]),
      theta: th0 + spread * ZD[m],
      a0: a * wm,
      phase: PHASES[m + 1],
    });
  }
  return {
    comps, bw, spread, wavesPerSet, eDom,
    setOmega: (2 * Math.PI * fp) / wavesPerSet,
    setDepth: 0.3,
    rep: { omega: 2 * Math.PI * fp, theta: th0, a0: a },
  };
}

// ---------------------------------------------------------------------------

function eikonal(N, dx, kk, S, fixed) {
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

// Solve |grad S| = kk with a plane wave (S = px * x) on the offshore row and
// Snell's law up the upwind side edge.
function phaseField(N, dx, kk, px, S, fixed) {
  S.fill(1e20);
  fixed.fill(0);
  for (let i = 0; i < N; i++) { S[i] = px * i * dx; fixed[i] = 1; }
  if (Math.abs(px) > 1e-9) {
    const ie = px > 0 ? 0 : N - 1;
    for (let j = 1; j < N; j++) {
      const c = j * N + ie, p = c - N;
      const ky1 = Math.sqrt(Math.max(kk[p] * kk[p] - px * px, 0.01 * kk[p] * kk[p]));
      const ky2 = Math.sqrt(Math.max(kk[c] * kk[c] - px * px, 0.01 * kk[c] * kk[c]));
      S[c] = S[p] + 0.5 * (ky1 + ky2) * dx;
      fixed[c] = 1;
    }
  }
  eikonal(N, dx, kk, S, fixed);
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
  let rep = null; // fields of the dominant train

  spec.comps.forEach((comp, m) => {
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
    phaseField(N, dx, kk, kx, S, fixed);
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
    const KsKr = Ks * Kr;

    transport(N, dx, ux, uy, div, cg, land, lambda, amp, null);
    const A = new Float32Array(NN);
    for (let c = 0; c < NN; c++) A[c] = land[c] ? 0 : amp[c] * comp.a0 * KsKr;
    outA.push(A);
    outS.push(S.slice());

    if (m === 0) {
      // The dominant train also decides breaking, foam direction and sets.
      // Saturate the typical wave (the whole swell height, not just this
      // component's share of it).
      const aB = spec.rep.a0 * KsKr;
      const und = amp.slice();
      const sat = new Float32Array(NN);
      transport(N, dx, ux, uy, div, cg, land, lambda, sat, { H0: 2 * aB, h: heff, gamma: 0.68 });
      // Group travel time: |grad tau| = 1 / cg.
      const slow = new Float32Array(NN);
      for (let c = 0; c < NN; c++) slow[c] = (land[c] ? 6 : 1) / cg[c];
      const tau = new Float32Array(NN);
      phaseField(N, dx, slow, Math.sin(theta) / cg0, tau, fixed);
      rep = { und, sat, aB, ux: ux.slice(), uy: uy.slice(), k: kk.slice(), lambda: lambda.slice(), tau };
    }
  });

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
      slope[c] = Math.max(0, (gx * rep.ux[c] + gy * rep.uy[c]) / dx);
    }
  blur(N, slope, tmp, 4);
  const L0 = (2 * Math.PI * G) / (spec.rep.omega * spec.rep.omega);
  const steep0 = Math.sqrt(Math.max(swell.height, 0.05) / L0);
  for (let c = 0; c < NN; c++) {
    const u = rep.und[c], s = rep.sat[c];
    D[c] = u > 1e-6 ? Math.min(1, s / u) : 0;
    gamma[c] = Math.min(1.15, 0.78 + 1.6 * slope[c]);
    const x = slope[c] / steep0;
    xi[c] = x;
    plunge[c] = smooth(0.3, 0.75, x) * (1 - smooth(2.2, 3.6, x));
    surge[c] = smooth(2.2, 3.6, x);
    hs[c] = land[c] ? 0 : s * 2 * rep.aB;
    speed[c] = land[c] ? 0 : spec.rep.omega / rep.k[c];
  }

  // Summary for the readout: where and how the waves first break.
  const summary = summarize(N, dx, h, land, rep, D, gamma, xi, spec, L0, swell);

  return {
    S: outS, A: outA, h, D, gamma, plunge, surge, ux: rep.ux, uy: rep.uy, speed, hs, xi,
    tau: rep.tau, lambda: rep.lambda,
    omegas: spec.comps.map((c) => c.omega),
    phases: spec.comps.map((c) => c.phase),
    setOmega: spec.setOmega, setDepth: spec.setDepth,
    summary,
  };
}

function smooth(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function summarize(N, dx, h, land, rep, D, gamma, xi, spec, L0, swell) {
  // For each alongshore column, walk shoreward to the first cell where the
  // biggest waves of a set (about 1.6x the typical wave) reach the breaking
  // limit. Weighted by wave energy so the main peak decides the verdict.
  // Also note where the typical wave starts losing energy (its break point).
  let nB = 0, sumDepth = 0, sumXi = 0, sumDist = 0, nT = 0, sumTyp = 0;
  for (let i = 4; i < N - 4; i += 2) {
    let shore = -1;
    for (let j = 0; j < N; j++) if (land[j * N + i]) { shore = j; break; }
    if (shore > 0) {
      for (let j = 0; j < shore; j++) {
        const c = j * N + i;
        if (D[c] < 0.97) {
          const w = rep.und[c] * rep.und[c];
          nT += w;
          sumTyp += w * (shore - j) * dx;
          break;
        }
      }
    }
    for (let j = 0; j < N; j++) {
      const c = j * N + i;
      if (land[c]) break;
      const Hbig = 1.6 * rep.und[c] * 2 * rep.aB;
      if (Hbig >= gamma[c] * h[c]) {
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
    breakDistTypical: nT ? sumTyp / nT : 0,
    xi: xiAvg, type, wavesPerSet: spec.wavesPerSet,
    spreadDeg: (spec.spread * 180) / Math.PI, bw: spec.bw,
    spot: breakSpot(N, dx, h, land, rep, D, gamma, depth),
  };
}

// The place to watch from: the alongshore column where the typical wave
// sheds the most energy (its break line is longest and heaviest), smoothed
// over 20 m and kept away from the box edges. On a barred beach this picks
// a bar, not the rip channel next to it. World coordinates, grid-centred.
function breakSpot(N, dx, h, land, rep, D, gamma, depthAvg) {
  const half = ((N - 1) * dx) / 2;
  const score = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < N; j++) {
      const c = j * N + i;
      if (land[c]) break;
      const H = 2 * rep.und[c] * rep.aB;
      s += Math.max(0, 1 - D[c]) * H * H;
    }
    score[i] = s;
  }
  const w = Math.max(1, Math.round(10 / dx));
  let best = N >> 1, bestS = -1;
  for (let i = N >> 3; i < N - (N >> 3); i++) {
    let s = 0;
    for (let k = -w; k <= w; k++) s += score[Math.min(N - 1, Math.max(0, i + k))];
    if (s > bestS) { bestS = s; best = i; }
  }
  let shore = -1;
  for (let j = 0; j < N; j++) if (land[j * N + best]) { shore = j; break; }
  // Where the typical wave is clearly whitewater on screen: the renderer
  // ramps a bore in as the wave's height ratio, eased inside the surf zone,
  // climbs from 0.92 to 1.15 (see the wave pass in render.js).
  let jB = -1;
  for (let j = 0; j < (shore > 0 ? shore : N); j++) {
    const c = j * N + best;
    const r = (2 * rep.aB * rep.sat[c] * 0.96) / Math.max(gamma[c] * h[c], 1e-3);
    const sz = smooth(0.03, 0.2, 1 - D[c]);
    if (r / (1 - 0.4 * sz) >= 1.12) { jB = j; break; }
  }
  return {
    x: best * dx - half,
    shoreZ: shore > 0 ? shore * dx - half : null,
    breakZ: jB >= 0 ? jB * dx - half : null,
    depth: jB >= 0 ? h[jB * N + best] : depthAvg,
  };
}
