// The beach is a heightfield: bed elevation z (metres, relative to mean sea
// level) on a square grid, plus a rock mask (0 = sand, 1 = rock). Rows run
// from offshore (j = 0) to the land (j = N-1); columns run alongshore.
// Hand-built beaches and real bathymetry share this representation.

export const N = 256;
export const DX = 2.5; // metres per cell
export const EXT = (N - 1) * DX; // 637.5 m square

// --- seeded noise ----------------------------------------------------------
function hash(x, y, s) {
  let h = (x * 374761393 + y * 668265263 + s * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}
function vnoise(x, y, s) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi, s), b = hash(xi + 1, yi, s);
  const c = hash(xi, yi + 1, s), d = hash(xi + 1, yi + 1, s);
  return (a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v) * 2 - 1;
}
export function fbm(x, y, s, oct = 4) {
  let f = 0, amp = 0.5, fr = 1;
  for (let o = 0; o < oct; o++) {
    f += amp * vnoise(x * fr, y * fr, s + o * 17);
    amp *= 0.5;
    fr *= 2.03;
  }
  return f;
}
const sstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// --- generator ---------------------------------------------------------------
export const PRESETS = {
  crescent: { name: 'Crescent sandbars', shape: 'straight', slope: 0.45, bar: 'rhythmic', barDist: 115, rocks: 0, roughness: 0.35 },
  longbar: { name: 'Long sandbar', shape: 'straight', slope: 0.5, bar: 'straight', barDist: 150, rocks: 0, roughness: 0.25 },
  point: { name: 'Point break', shape: 'point', slope: 0.55, bar: 'none', barDist: 120, rocks: 0.15, roughness: 0.4 },
  reef: { name: 'Reef peak', shape: 'reef', slope: 0.5, bar: 'none', barDist: 120, rocks: 0.05, roughness: 0.3 },
  cove: { name: 'Rocky cove', shape: 'cove', slope: 0.6, bar: 'straight', barDist: 90, rocks: 0.45, roughness: 0.5 },
  blank: { name: 'Blank slope', shape: 'straight', slope: 0.4, bar: 'none', barDist: 120, rocks: 0, roughness: 0 },
};

export function generate(opt) {
  const z = new Float32Array(N * N);
  const rock = new Float32Array(N * N);
  const s = opt.seed | 0;
  const R = opt.roughness;
  const A = 0.11 + 0.17 * opt.slope; // Dean profile parameter (m^1/3)
  const beachSlope = 0.045 + 0.08 * opt.slope;
  const y0 = EXT * 0.84;
  const xc = EXT / 2;

  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const c = j * N + i;
      const x = i * DX, y = j * DX;
      // Shoreline position (y of the waterline) for this column.
      let ys = y0 + 18 * R * fbm(x / 160, 0.5, s + 3, 3);
      let reefZ = -1e9, rockAmt = 0;
      if (opt.shape === 'cove') {
        ys += 55 * Math.exp(-(((x - xc) / 170) ** 2)) - 60 * (sstep(0.62, 0.95, Math.abs(x - xc) / xc));
        rockAmt = sstep(0.6, 0.85, Math.abs(x - xc) / xc);
      } else if (opt.shape === 'point') {
        const xp = EXT * 0.22; // tip of the headland
        const head = 1 - sstep(xp - 30, xp + 50, x);
        ys -= 230 * head;
        // Rocky shelf off the point, sloping out and down the coast.
        const dTip = Math.hypot(Math.max(0, x - xp) * 0.55, Math.max(0, ys - y + 0));
        rockAmt = Math.max(head, 1 - sstep(60, 140, Math.hypot(x - xp, y - (y0 - 230))));
        reefZ = -1.2 - 0.025 * dTip - 0.012 * Math.max(0, x - xp) + 1.2 * head;
      } else if (opt.shape === 'reef') {
        // A ridge of reef running out to sea, highest near shore and sinking
        // offshore. Waves catch its tip first and peel off both sides: an
        // A-frame peak.
        const d = ys - y;
        if (d > 40 && d < 380) {
          const crest = -0.6 - 0.045 * (d - 40);
          const taper = sstep(40, 90, d) * (1 - sstep(300, 380, d));
          reefZ = crest - Math.abs(x - xc) * 0.07 - (1 - taper) * 4;
        }
      }

      const d = ys - y; // distance offshore of the waterline
      let zz;
      if (d > 0) {
        const d0 = 12;
        zz = -(A * Math.pow(d + d0, 2 / 3) - A * Math.pow(d0, 2 / 3)) - 0.04 * d * 0.12;
        // Sandbars.
        if (opt.bar !== 'none') {
          let bd = opt.barDist, bh = 1.4 + 0.8 * opt.slope;
          if (opt.bar === 'rhythmic') {
            const ph = (2 * Math.PI * x) / 210 + 0.6 * fbm(x / 300, 1.7, s + 9, 2);
            bd *= 1 + 0.28 * Math.cos(ph);
            bh *= 0.45 + 0.55 * sstep(-0.75, 0.2, Math.cos(ph));
          }
          const bw = 22 + bd * 0.12;
          zz += bh * Math.exp(-(((d - bd) / bw) ** 2));
          zz -= 0.45 * bh * Math.exp(-(((d - bd * 0.62) / (bw * 0.8)) ** 2));
        }
      } else {
        const up = -d;
        const berm = 2.4 + 0.4 * opt.slope;
        zz = Math.min(up * beachSlope, berm + (up - berm / beachSlope) * 0.012);
        zz += 4.5 * sstep(85, 140, up); // dune
      }
      // Randomness: broad undulations plus fine ripples.
      zz += R * (0.7 * fbm(x / 90, y / 90, s, 4) + 0.12 * fbm(x / 14, y / 14, s + 5, 2));
      if (reefZ > -1e8) {
        const rz = reefZ + 0.6 * fbm(x / 25, y / 25, s + 11, 4) * (0.6 + R);
        if (rz > zz) {
          rockAmt = Math.max(rockAmt, sstep(0, 0.6, rz - zz));
          zz = rz;
        }
      }
      // Scattered rocky outcrops.
      if (opt.rocks > 0) {
        const n = fbm(x / 55, y / 55, s + 21, 4) + 0.5 * (opt.rocks - 0.5);
        const r2 = sstep(0.12, 0.3, n) * sstep(-60, -30, d) * (1 - sstep(140, 200, d));
        if (r2 > 0) {
          zz += r2 * (0.8 + 1.4 * Math.abs(fbm(x / 9, y / 9, s + 23, 3)));
          rockAmt = Math.max(rockAmt, r2);
        }
      }
      if (rockAmt > 0.05 && d > -20) zz += rockAmt * 0.5 * fbm(x / 7, y / 7, s + 31, 3);
      z[c] = zz;
      rock[c] = Math.min(1, rockAmt);
    }
  }
  return { z, rock };
}

// Add fractal noise to an existing beach (the "add randomness" button).
export function roughen(beach, amount, seed) {
  const { z, rock } = beach;
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const c = j * N + i, x = i * DX, y = j * DX;
      const n = 0.5 * fbm(x / 70, y / 70, seed, 4) + 0.15 * fbm(x / 12, y / 12, seed + 3, 2);
      z[c] += amount * n * (rock[c] > 0.5 ? 1.6 : 1);
    }
}

// --- sculpting -------------------------------------------------------------
export function sampleZ(z, x, y) {
  const gx = Math.min(Math.max(x / DX, 0), N - 1.001);
  const gy = Math.min(Math.max(y / DX, 0), N - 1.001);
  const i = Math.floor(gx), j = Math.floor(gy);
  const fx = gx - i, fy = gy - j;
  const c = j * N + i;
  return (z[c] * (1 - fx) + z[c + 1] * fx) * (1 - fy) + (z[c + N] * (1 - fx) + z[c + N + 1] * fx) * fy;
}

// tool: sand | dig | smooth | flatten | rock. (x, y) in metres.
export function brush(beach, tool, x, y, radius, strength, dt, target) {
  const { z, rock } = beach;
  const r = radius / DX;
  const ci = x / DX, cj = y / DX;
  const i0 = Math.max(1, Math.floor(ci - r)), i1 = Math.min(N - 2, Math.ceil(ci + r));
  const j0 = Math.max(1, Math.floor(cj - r)), j1 = Math.min(N - 2, Math.ceil(cj + r));
  const rate = strength * dt;
  const seed = 77;
  for (let j = j0; j <= j1; j++)
    for (let i = i0; i <= i1; i++) {
      const q = ((i - ci) ** 2 + (j - cj) ** 2) / (r * r);
      if (q >= 1) continue;
      const w = (1 - q) * (1 - q);
      const c = j * N + i;
      if (tool === 'sand') {
        z[c] += 2.2 * rate * w;
        rock[c] = Math.max(0, rock[c] - 2 * rate * w);
      } else if (tool === 'dig') {
        z[c] -= 2.2 * rate * w;
      } else if (tool === 'smooth') {
        const avg = (z[c - 1] + z[c + 1] + z[c - N] + z[c + N] + z[c]) / 5;
        z[c] += (avg - z[c]) * Math.min(1, 6 * rate * w);
      } else if (tool === 'flatten') {
        z[c] += (target - z[c]) * Math.min(1, 3 * rate * w);
      } else if (tool === 'rock') {
        const n = 0.6 + 0.8 * Math.abs(fbm((i * DX) / 6, (j * DX) / 6, seed, 3));
        z[c] += 1.6 * rate * w * n;
        rock[c] = Math.min(1, rock[c] + 4 * rate * w);
      }
    }
  if (tool !== 'rock') slump(beach, i0 - 2, i1 + 2, j0 - 2, j1 + 2);
}

// Sand slides when piled steeper than its angle of repose (rock does not).
export function slump(beach, i0, i1, j0, j1, iters = 4) {
  const { z, rock } = beach;
  const wl = beach.waterLevel;
  i0 = Math.max(0, i0); j0 = Math.max(0, j0);
  i1 = Math.min(N - 1, i1); j1 = Math.min(N - 1, j1);
  for (let it = 0; it < iters; it++)
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const c = j * N + i;
        if (rock[c] > 0.5) continue;
        const maxD = (z[c] > wl ? 0.62 : 0.45) * DX;
        for (const n of [i < N - 1 ? c + 1 : -1, j < N - 1 ? c + N : -1]) {
          if (n < 0 || rock[n] > 0.5) continue;
          const d = z[c] - z[n];
          if (Math.abs(d) > maxD) {
            const m = 0.25 * (Math.abs(d) - maxD) * Math.sign(d);
            z[c] -= m;
            z[n] += m;
          }
        }
      }
}

// Raycast against the heightfield. o, d: ray in grid metres (x, y, elevation).
export function raycast(z, o, d) {
  // Clip to the box.
  let t0 = 0, t1 = 1e5;
  for (const [oo, dd, lo, hi] of [[o[0], d[0], 0, EXT], [o[1], d[1], 0, EXT], [o[2], d[2], -40, 60]]) {
    if (Math.abs(dd) < 1e-9) { if (oo < lo || oo > hi) return null; continue; }
    let a = (lo - oo) / dd, b = (hi - oo) / dd;
    if (a > b) [a, b] = [b, a];
    t0 = Math.max(t0, a); t1 = Math.min(t1, b);
  }
  if (t0 > t1) return null;
  const step = DX * 0.5;
  let prevT = t0;
  for (let t = t0; t <= t1; t += step) {
    const px = o[0] + d[0] * t, py = o[1] + d[1] * t, pz = o[2] + d[2] * t;
    if (pz <= sampleZ(z, px, py)) {
      let a = prevT, b = t;
      for (let k = 0; k < 12; k++) {
        const m = 0.5 * (a + b);
        const qz = o[2] + d[2] * m;
        if (qz <= sampleZ(z, o[0] + d[0] * m, o[1] + d[1] * m)) b = m; else a = m;
      }
      return [o[0] + d[0] * b, o[1] + d[1] * b, o[2] + d[2] * b];
    }
    prevT = t;
  }
  return null;
}

// --- saving ---------------------------------------------------------------
// Compact encoding: elevation in centimetres (Int16) and rock in bytes.
export function encodeBeach(beach) {
  const zi = new Int16Array(N * N);
  const r = new Uint8Array(N * N);
  for (let c = 0; c < N * N; c++) {
    zi[c] = Math.max(-32000, Math.min(32000, Math.round(beach.z[c] * 100)));
    r[c] = Math.round(beach.rock[c] * 255);
  }
  const o = { v: 1, n: N, dx: DX, wl: beach.waterLevel, z: b64(new Uint8Array(zi.buffer)), rock: b64(r) };
  if (beach.real) o.real = beach.real; // a real beach keeps its name and compass orientation
  return o;
}
export function decodeBeach(o) {
  if (!o || o.n !== N) throw new Error('This beach was saved at a different grid size.');
  const zi = new Int16Array(unb64(o.z).buffer);
  const r = unb64(o.rock);
  const z = new Float32Array(N * N), rock = new Float32Array(N * N);
  for (let c = 0; c < N * N; c++) { z[c] = zi[c] / 100; rock[c] = r[c] / 255; }
  return { z, rock, waterLevel: o.wl || 0, real: o.real || null };
}
function b64(u8) {
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
function unb64(s) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}
