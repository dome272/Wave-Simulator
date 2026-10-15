// Synthetic crest-record generator for the breakers package tests and demo.
//
// Produces Float32Array crest records in the exact layout of section 5 of
// docs/engine-contract.md (V columns x K = 6 slots x 8 channels) for an
// analytic wave field: one or more crest lines advancing shoreward over a
// plane beach, each a bump with a steep front face and a trough ahead of it.
// `FC` switches from 0 to 1 for a column once that column's crest has passed a
// given cross-shore position (`breakY`), and the trough channels (etaT, Jf,
// sF) describe the trough in front of the crest.
//
// Coordinates follow section 1: grid x alongshore, grid y cross-shore (y = 0
// offshore, growing shoreward), world = grid - EXT/2.
import { N, DX, EXT } from '../src/terrain.js';

export const G = 9.81;
export const K = 6; // slots per column
export const CH = 8; // channels per record

// LOCAL COPY of the layout numbers of contract section 1. src/engine/layout.js
// is written by another engineer in parallel; this copy exists so the tests do
// not depend on it. Keep the two in sync (dxs, V, ML, MO, MI, nx, ny).
export function layoutLocal(r) {
  const dxs = DX / r;
  const V = (N - 1) * r;
  const ML = Math.round(100 / dxs);
  const MO = Math.round(300 / dxs);
  const MI = 2 * r;
  return { r, dxs, V, ML, MO, MI, nx: V + 2 * ML, ny: MO + V + MI };
}

// A plane beach: bed elevation rises shoreward with the given slope and
// crosses the datum at grid y = shoreY (so the still waterline sits there
// when wl = 0). Returns a beach object in the terrain.js format.
export function planeBeach({ slope = 1 / 15, shoreY = 500 } = {}) {
  const z = new Float32Array(N * N);
  const rock = new Float32Array(N * N);
  for (let j = 0; j < N; j++) {
    const zz = (j * DX - shoreY) * slope;
    for (let i = 0; i < N; i++) z[j * N + i] = zz;
  }
  return { z, rock, waterLevel: 0, slope, shoreY };
}

export function bedAt(beach, x, y) {
  const gx = Math.min(Math.max(x / DX, 0), N - 1.001);
  const gy = Math.min(Math.max(y / DX, 0), N - 1.001);
  const i = Math.floor(gx), j = Math.floor(gy);
  const fx = gx - i, fy = gy - j;
  const c = j * N + i, z = beach.z;
  return (z[c] * (1 - fx) + z[c + 1] * fx) * (1 - fy) + (z[c + N] * (1 - fx) + z[c + N + 1] * fx) * fy;
}

// Cross-shore profile of a crest bump: d = y - yCrest (m), positive ahead
// (shoreward). prof = 1 at the crest, a steep cosine front face of length Lf
// ahead, a gentler gaussian back of length Lb behind.
function prof(d, Lf, Lb) {
  if (d >= 0) return d < Lf ? 0.5 * (1 + Math.cos((Math.PI * d) / Lf)) : 0;
  return Math.exp(-(d * d) / (Lb * Lb));
}

/**
 * makeScenario(opts) -> scenario
 *
 * opts.r        layout tier (1 or 2)
 * opts.beach    beach object (default planeBeach())
 * opts.wl       still water level (m)
 * opts.crests   array of crest descriptions:
 *   { y0, c, H, amp = 0, lambda = 60, obliq = 0, breakY = Infinity, etaCFrac = 0.7,
 *     Lf = 2.5*dxs, Lb = 12, phase = 0, x0 = EXT/2 }
 *   The crest line is yC(x, t) = y0 + c*t + amp*sin(2*pi*(x - x0)/lambda + phase) + obliq*(x - x0)
 *   (grid coordinates). It carries elevation etaC = etaCFrac*H above wl at the
 *   crest and etaT = etaC - H in the trough ahead. FC = 1 once yC(x, t) >= breakY.
 * opts.quantise rows are rounded to integer solver rows (as a GPU scan would
 *   produce); default true.
 * opts.minDepth crests whose total depth hC < minDepth are not recorded
 *   (the scan skips cells with h <= 0.15); default 0.15.
 *
 * scenario.records(t, out?)   -> Float32Array(V*K*8)
 * scenario.eta(x, y, t)       -> surface elevation above wl at grid (x, y)
 * scenario.crestY(k, x, t)    -> yC of crest k
 * scenario.lay, scenario.beach, scenario.wl, scenario.crests
 */
export function makeScenario(opts = {}) {
  const lay = layoutLocal(opts.r || 1);
  const { V, dxs, MO } = lay;
  const beach = opts.beach || planeBeach();
  const wl = opts.wl ?? 0;
  const quantise = opts.quantise ?? true;
  const minDepth = opts.minDepth ?? 0.15;
  const crests = (opts.crests || []).map((c) => ({
    y0: c.y0, c: c.c, H: c.H, amp: c.amp || 0, lambda: c.lambda || 60, obliq: c.obliq || 0,
    breakY: c.breakY ?? Infinity, etaCFrac: c.etaCFrac ?? 0.7,
    Lf: c.Lf ?? 2.5 * dxs, Lb: c.Lb ?? 12, phase: c.phase || 0, x0: c.x0 ?? EXT / 2,
    stop: c.stop ?? Infinity, // records of this crest vanish after this time (tests the drop rule)
  }));

  function crestY(k, x, t) {
    const c = crests[k];
    return c.y0 + c.c * t + c.amp * Math.sin((2 * Math.PI * (x - c.x0)) / c.lambda + c.phase) + c.obliq * (x - c.x0);
  }

  function eta(x, y, t) {
    let e = 0;
    for (let k = 0; k < crests.length; k++) {
      const c = crests[k];
      if (t > c.stop) continue;
      const d = y - crestY(k, x, t);
      const etaC = c.etaCFrac * c.H, etaT = etaC - c.H;
      const win = Math.exp(-(d * d) / (28 * 28));
      e += etaT * win + c.H * prof(d, c.Lf, c.Lb);
    }
    return e;
  }

  function records(t, out) {
    const rec = out || new Float32Array(V * K * CH);
    rec.fill(0);
    for (let i = 0; i < rec.length; i += CH) rec[i] = -1;
    const order = [];
    for (let col = 0; col < V; col++) {
      const x = (col + 0.5) * dxs;
      order.length = 0;
      for (let k = 0; k < crests.length; k++) {
        if (t > crests[k].stop) continue;
        order.push([crestY(k, x, t), k]);
      }
      order.sort((a, b) => a[0] - b[0]); // offshore first, as the scan walks shoreward
      let slot = 0;
      for (const [yC, k] of order) {
        if (slot >= K) break;
        const c = crests[k];
        const etaC = c.etaCFrac * c.H, etaT = etaC - c.H;
        const depthC = wl - bedAt(beach, x, yC);
        const hC = etaC + depthC;
        if (hC < minDepth || yC < 0 || yC > EXT) continue;
        let J = yC / dxs + MO - 0.5;
        if (quantise) J = Math.round(J);
        const Jf = quantise ? Math.round(J + (0.5 * c.Lf) / dxs) : J + (0.5 * c.Lf) / dxs;
        const sF = (Math.PI * c.H) / (2 * c.Lf);
        const b = (col * K + slot) * CH;
        rec[b] = J;
        rec[b + 1] = etaC;
        rec[b + 2] = hC;
        rec[b + 3] = yC >= c.breakY ? 1 : 0;
        rec[b + 4] = etaT;
        rec[b + 5] = Jf;
        rec[b + 6] = sF;
        rec[b + 7] = (c.c * etaC) / hC;
        slot++;
      }
    }
    return rec;
  }

  // Empty records (every slot J = -1): what the scan returns when nothing is left.
  function empty(out) {
    const rec = out || new Float32Array(V * K * CH);
    rec.fill(0);
    for (let i = 0; i < rec.length; i += CH) rec[i] = -1;
    return rec;
  }

  return { lay, beach, wl, crests, crestY, eta, records, empty, G };
}

// Linear-theory phase speed at the crest, the tracker's own fallback formula.
export function phaseSpeed(hC, etaC) {
  return Math.sqrt(G * hC) * (1 + (0.5 * etaC) / hC);
}

// Convenience: the standard single-crest case used by several tests. A 2 m
// wave on a 1:15 beach, crest at y = 440 m at t = 0, breaking once it passes
// y = 455 m; the trough sits 0.3 H below wl.
export function standardCase(over = {}) {
  const slope = over.slope ?? 1 / 15;
  const beach = planeBeach({ slope, shoreY: over.shoreY ?? 500 });
  const H = over.H ?? 2.0;
  const y0 = over.y0 ?? 440;
  const breakY = over.breakY ?? 455;
  const etaC = 0.7 * H;
  const yRef = Number.isFinite(breakY) ? breakY : y0; // speed set by the depth at the break line (or the start)
  const hC = etaC + (beach.shoreY - yRef) * slope;
  const c = over.c ?? phaseSpeed(hC, etaC);
  const swell = { height: H, period: over.period ?? 12, direction: 0, distanceKm: 2000 };
  const sc = makeScenario({
    r: over.r ?? 1, beach, wl: 0,
    crests: [{ y0, c, H, amp: over.amp ?? 0, lambda: over.lambda ?? 60, obliq: over.obliq ?? 0, breakY, stop: over.stop, Lf: over.Lf }],
  });
  return { sc, beach, swell, H, c, etaC, hC, breakY, slope, y0 };
}
