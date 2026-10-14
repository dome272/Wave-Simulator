// Solver grid layout (section 1 of docs/engine-contract.md). Plain JS, no
// three.js: the solver cells sit on the terrain's node grid (N x N nodes,
// DX apart) with margins around the visible area, and a tier r in {1, 2}
// refines the solver spacing to DX / r.
import { N, DX } from '../terrain.js';

/** Layout for tier r: { r, dxs, V, ML, MO, MI, nx, ny }. */
export function layout(r) {
  const dxs = DX / r;
  const V = (N - 1) * r;
  const ML = Math.round(100 / dxs), MO = Math.round(300 / dxs), MI = 2 * r;
  return { r, dxs, V, ML, MO, MI, nx: V + 2 * ML, ny: MO + V + MI };
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

/** Bilinear sample of a node field (N x N, row-major) at grid coordinates (gx, gy) in node units, clamped. */
export function sampleNodes(field, gx, gy) {
  const x = clamp(gx, 0, N - 1), y = clamp(gy, 0, N - 1);
  const i0 = Math.min(N - 2, Math.floor(x)), j0 = Math.min(N - 2, Math.floor(y));
  const fx = x - i0, fy = y - j0;
  const c = j0 * N + i0;
  const a = field[c], b = field[c + 1], d = field[c + N], e = field[c + N + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
}

/**
 * Corner bed elevations (m, relative to the datum) of the solver cells:
 * Float32Array((nx+1)*(ny+1)), stride nx+1; corner (I, J) is the bilinear
 * interpolation of beach.z at the grid position clamped to the node grid, so
 * the margins continue the edge values. `wl` is accepted for the contract's
 * signature; the corner elevations do not depend on it.
 */
export function cornerBed(beach, wl, r, out) {
  const lay = layout(r);
  const sx = lay.nx + 1, sy = lay.ny + 1;
  const o = out && out.length === sx * sy ? out : new Float32Array(sx * sy);
  const z = beach.z;
  for (let J = 0; J < sy; J++) {
    const gy = ((J - lay.MO) * lay.dxs) / DX;
    for (let I = 0; I < sx; I++) {
      const gx = ((I - lay.ML) * lay.dxs) / DX;
      o[J * sx + I] = sampleNodes(z, gx, gy);
    }
  }
  return o;
}

/** Manning n at the corners: 0.02 on sand, 0.035 where the nearest node's rock value is >= 0.5. */
export function cornerManning(beach, r, out) {
  const lay = layout(r);
  const sx = lay.nx + 1, sy = lay.ny + 1;
  const o = out && out.length === sx * sy ? out : new Float32Array(sx * sy);
  const rock = beach.rock;
  for (let J = 0; J < sy; J++) {
    const j = clamp(Math.round(((J - lay.MO) * lay.dxs) / DX), 0, N - 1);
    for (let I = 0; I < sx; I++) {
      const i = clamp(Math.round(((I - lay.ML) * lay.dxs) / DX), 0, N - 1);
      o[J * sx + I] = rock && rock[j * N + i] >= 0.5 ? 0.035 : 0.02;
    }
  }
  return o;
}

/** Grid coordinates (m) of the centre of solver cell (I, J). */
export function cellToGrid(lay, I, J) {
  return [(I - lay.ML + 0.5) * lay.dxs, (J - lay.MO + 0.5) * lay.dxs];
}

/** Fractional solver cell (I, J) whose centre sits at grid coordinates (x, y) m. */
export function gridToCell(lay, x, y) {
  return [x / lay.dxs + lay.ML - 0.5, y / lay.dxs + lay.MO - 0.5];
}

/**
 * Relaxation-zone weight s in [0, 1] of cell (I, J): 0 at the inner edge of a
 * margin, 1 at the domain edge, max where margins meet, none inland (the same
 * formula as relaxField in src/boussinesq.js with offshore = MO, lateral = ML).
 */
export function relaxWeight(lay, I, J, zones) {
  const off = zones ? zones.offshore : lay.MO, lat = zones ? zones.lateral : lay.ML, inl = zones ? zones.inland || 0 : 0;
  let s = 0;
  if (off > 0 && J < off) s = Math.max(s, (off - J) / off);
  if (inl > 0 && J >= lay.ny - inl) s = Math.max(s, (J - (lay.ny - inl) + 1) / inl);
  if (lat > 0) {
    if (I < lat) s = Math.max(s, (lat - I) / lat);
    if (I >= lay.nx - lat) s = Math.max(s, (I - (lay.nx - lat) + 1) / lat);
  }
  return Math.min(1, s);
}
