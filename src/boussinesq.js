// Depth-integrated nonlinear wave solver: the extended Boussinesq equations of
// Madsen & Sorensen (1992) in the conservative form used by Celeris (Tavakkol
// & Lynett 2017), with the hyperbolic part solved by the well-balanced,
// positivity-preserving central-upwind scheme of Kurganov & Petrova (2007).
//
// This is the reference implementation in plain JavaScript: it is the oracle
// for the GPU port (src/gpu-solver.js), the CPU fallback, and the thing the
// Node test battery exercises. Layout, variable names and the order of
// operations are kept simple so the GPU shaders can follow them line by line.
//
// Variables per cell (row-major, c = j * nx + i; i alongshore, j cross-shore,
// row 0 offshore). The bed is given at the cell CORNERS ((nx+1) x (ny+1)
// nodes, the heightfield's nodes): the face beds are the mean of the two
// corners of that face and the centre bed is the mean of the four corners,
// so B_c = (B_E + B_W)/2 = (B_N + B_S)/2 exactly, which the Kurganov-Petrova
// wet/dry reconstruction relies on.
//   B   bed elevation (m), cell centre; Bx/By bed at the east/north faces
//   wl  still water level (m); d = wl - B still-water depth (< 0 on land)
//   h   total water depth (m) >= 0; W = B + h water surface; eta = W - wl
//   P,Q depth-integrated fluxes h*u, h*v (m^2/s)
//   F   breaking flag/age in [0,1] (1 = just broke), chi = dispersion factor
//
// Equations (B = 1/15 below, g = 9.81):
//   h_t + P_x + Q_y = 0
//   (L_x P)_t + (P u + g h^2/2)_x + (P v)_y = -g h B_x - tau_x + E_x + M_x
//   (L_y Q)_t + (Q u)_x + (Q v + g h^2/2)_y = -g h B_y - tau_y + E_y + M_y
//   L_x P = P - chi[(B+1/3) d^2 P_xx + (1/3) d d_x P_x]   (tridiagonal per row)
//   E_x = chi[ B g d^3 (W_xxx + W_xyy) + d d_x (2 B g d W_xx + B g d W_yy) + d d_y B g d W_xy ]
//   M_x = chi[ (B+1/3) d^2 (Q_t)_xy + (1/6) d d_x (Q_t)_y + (1/6) d d_y (Q_t)_x ]  (lagged)
// and the mirror images for y. tau is Manning friction. Where chi = 0 the
// equations reduce to the nonlinear shallow water equations, which is how
// breaking is modelled: a broken wave front is carried as a shock (bore)
// whose jump conditions dissipate the right amount of energy (hybrid
// approach of Tonelli & Petti 2009, as in FUNWAVE-TVD and Celeris).

export const G = 9.81;
export const BQ = 1 / 15;

const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function minmod3(a, b, c) {
  if (a > 0 && b > 0 && c > 0) return Math.min(a, b, c);
  if (a < 0 && b < 0 && c < 0) return Math.max(a, b, c);
  return 0;
}

/**
 * Create a solver.
 * @param {object} o
 * @param {number} o.nx columns (alongshore), @param {number} o.ny rows (cross-shore)
 * @param {number} o.dx cell size (m), square cells
 * @param {Float64Array|Float32Array} o.bed bed elevation at the cell corners, length (nx+1)*(ny+1), stride nx+1
 * @param {number} [o.wl=0] still water level
 * @param {number|Float32Array} [o.manning=0.02] Manning n, scalar or per cell
 * @param {boolean} [o.periodicX=false] wrap the alongshore direction (tests)
 * @param {boolean} [o.dispersion=true] include the Boussinesq terms
 * @param {boolean} [o.breaking=true] hybrid breaking switch
 * @param {Float32Array} [o.relax] relaxation-zone weight per cell (0 interior .. 1 at the edge)
 * @param {function} [o.target] (t, etaT, PT, QT) => fills the target field for the relaxation zones
 */
export function createSolver(o) {
  const nx = o.nx, ny = o.ny, dx = o.dx, N = nx * ny;
  const g = o.g ?? G;
  const theta = o.theta ?? 1.3;
  const periodicX = !!o.periodicX;
  const periodicY = !!o.periodicY; // tests only
  const useDisp = o.dispersion !== false;
  const useBreak = o.breaking !== false;
  const hDry = o.hDry ?? 1e-4; // below this a cell carries no momentum
  const eps = o.eps ?? 0.05; // desingularisation depth (m)
  const dispDepth = o.dispDepth ?? [0.25, 0.75]; // chi ramps up over this still-water depth range
  // onsetSlope: Kennedy et al. (2000) front-steepness criterion eta_t > 0.65 sqrt(g h).
  // slopeRes: a grid of spacing dx cannot steepen a front beyond ~3 cells, so on coarse
  // grids the achievable eta_t at a front of height ~0.8 h is about 0.27 (h/dx) sqrt(g h);
  // the threshold is capped there so breaking is flagged when the front reaches the
  // steepest shape the grid can hold (where the limiter would otherwise start dissipating).
  const brk = Object.assign({ onsetRatio: 0.8, onsetSlope: 0.65, slopeRes: 0.18, keepSlope: 0.15, keepRatio: 0.35, memory: 2.0, ramp: 3 }, o.breakingParams || {});

  const B = new Float64Array(N), Bx = new Float64Array(N), By = new Float64Array(N);
  const d = new Float64Array(N), dxs = new Float64Array(N), dys = new Float64Array(N); // still depth and its slopes
  const chi = new Float64Array(N), chiBed = new Float64Array(N); // dispersion factor; bed-only part
  const nMan = new Float64Array(N);
  const h = new Float64Array(N), P = new Float64Array(N), Q = new Float64Array(N);
  const F = new Float64Array(N); // breaking age/flag, 1 = fresh
  const relax = o.relax || null;
  let target = o.target || null;
  let wl = o.wl ?? 0;

  // Work arrays.
  const h0 = new Float64Array(N), P0 = new Float64Array(N), Q0 = new Float64Array(N); // stage start
  const hPrev = new Float64Array(N), PPrev = new Float64Array(N), QPrev = new Float64Array(N); // previous step (eta_t, lagged cross terms)
  const Pt = new Float64Array(N), Qt = new Float64Array(N); // lagged time derivatives
  const hE = new Float64Array(N), hW = new Float64Array(N), hN = new Float64Array(N), hS = new Float64Array(N);
  const uE = new Float64Array(N), uW = new Float64Array(N), uN = new Float64Array(N), uS = new Float64Array(N);
  const vE = new Float64Array(N), vW = new Float64Array(N), vN = new Float64Array(N), vS = new Float64Array(N);
  const Fh = new Float64Array(N), FP = new Float64Array(N), FQ = new Float64Array(N); // x-face fluxes at i+1/2
  const Gh = new Float64Array(N), GP = new Float64Array(N), GQ = new Float64Array(N); // y-face fluxes at j+1/2
  const Rh = new Float64Array(N), RP = new Float64Array(N), RQ = new Float64Array(N); // right-hand sides
  const U = new Float64Array(N), V = new Float64Array(N); // L_x P, L_y Q
  const W = new Float64Array(N); // water surface
  const etaT = new Float64Array(N), PT = new Float64Array(N), QT = new Float64Array(N);
  const tmpA = new Float64Array(Math.max(nx, ny)), tmpC = new Float64Array(Math.max(nx, ny));
  let t = 0, dtPrev = 0;

  const wrapI = (i) => (periodicX ? (i + nx) % nx : Math.min(nx - 1, Math.max(0, i)));
  const wrapJ = (j) => (periodicY ? (j + ny) % ny : Math.min(ny - 1, Math.max(0, j)));
  const clampJ = wrapJ;

  let bedRef = null; // the current corner bed (the wall faces at i = 0 / j = 0 read it)
  function setBed(bed, manning) {
    const sx = nx + 1;
    if (bed.length !== sx * (ny + 1)) throw new Error(`bed must hold (nx+1)*(ny+1) corner values, got ${bed.length}`);
    bedRef = bed;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      const c00 = bed[j * sx + i], c10 = bed[j * sx + i + 1], c01 = bed[(j + 1) * sx + i], c11 = bed[(j + 1) * sx + i + 1];
      B[c] = 0.25 * (c00 + c10 + c01 + c11);
      Bx[c] = 0.5 * (c10 + c11); // east face
      By[c] = 0.5 * (c01 + c11); // north face
    }
    if (manning !== undefined) {
      if (typeof manning === 'number') nMan.fill(manning);
      else for (let c = 0; c < N; c++) nMan[c] = manning[c];
    }
    updateDepth();
  }
  // West face of cell (i,j) is the east face of (i-1,j); for a wall at i=0 use the cell's own west corners.
  const bedW = (i, j) => (i > 0 ? Bx[j * nx + i - 1] : periodicX ? Bx[j * nx + nx - 1] : 0.5 * (bedRef[j * (nx + 1)] + bedRef[(j + 1) * (nx + 1)]));
  const bedS = (i, j) => (j > 0 ? By[(j - 1) * nx + i] : periodicY ? By[(ny - 1) * nx + i] : 0.5 * (bedRef[i] + bedRef[i + 1]));

  function setWaterLevel(w) {
    const dw = w - wl;
    wl = w;
    // Wet cells keep their surface (the relaxation zones bring the new level
    // in); dry land below the new level floods to it.
    for (let c = 0; c < N; c++) {
      const dd = wl - B[c];
      if (h[c] <= hDry && dd > 0) h[c] = dd;
    }
    void dw;
    updateDepth();
  }

  function updateDepth() {
    for (let c = 0; c < N; c++) d[c] = wl - B[c];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      const dl = d[j * nx + wrapI(i - 1)], dr = d[j * nx + wrapI(i + 1)];
      const ds = d[clampJ(j - 1) * nx + i], dn = d[clampJ(j + 1) * nx + i];
      // Still-water depth slopes, clamped so the tridiagonal stays dominant.
      dxs[c] = Math.max(-0.2, Math.min(0.2, (dr - dl) / (2 * dx)));
      dys[c] = Math.max(-0.2, Math.min(0.2, (dn - ds) / (2 * dx)));
      const slope = Math.hypot((dr - dl) / (2 * dx), (dn - ds) / (2 * dx));
      chiBed[c] = useDisp ? smoothstep(dispDepth[0], dispDepth[1], d[c]) * (1 - smoothstep(0.35, 0.6, slope)) : 0;
      if ((!periodicY && (j === 0 || j === ny - 1)) || (!periodicX && (i === 0 || i === nx - 1))) chiBed[c] = 0;
    }
    updateChi();
  }

  // chi = bed factor x (1 - broken), with the broken region widened by `ramp` cells.
  function updateChi() {
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      let fm = 0;
      if (useBreak) {
        const r = brk.ramp;
        for (let dj = -r; dj <= r; dj++) {
          const jj = clampJ(j + dj);
          for (let di = -r; di <= r; di++) { const f = F[jj * nx + wrapI(i + di)]; if (f > fm) fm = f; }
        }
      }
      chi[c] = chiBed[c] * (1 - smoothstep(0.2, 0.5, fm));
    }
  }

  // Initial condition: still water (or a given surface / fluxes).
  function reset(etaInit) {
    for (let c = 0; c < N; c++) {
      const e = etaInit ? etaInit[c] : 0;
      h[c] = Math.max(0, wl + e - B[c]);
      P[c] = 0; Q[c] = 0; F[c] = 0;
    }
    hPrev.set(h); PPrev.set(P); QPrev.set(Q); Pt.fill(0); Qt.fill(0);
    t = 0; dtPrev = 0;
    updateChi();
  }

  // Desingularised velocity (Kurganov & Petrova 2007): u = sqrt2 h P / sqrt(h^4 + max(h^4, eps^4)).
  const e4 = eps * eps * eps * eps;
  const vel = (hh, pp) => (hh <= hDry ? 0 : (Math.SQRT2 * hh * pp) / Math.sqrt(hh * hh * hh * hh + Math.max(hh * hh * hh * hh, e4)));

  // Second-order reconstruction with hydrostatic reconstruction at the faces
  // (Audusse, Bouchut, Bristeau, Klein & Perthame 2004): the surface W and
  // the fluxes are reconstructed with a generalised minmod limiter, the face
  // depth is W minus the (continuous) face bed clipped at zero, and the bed a
  // cell "sees" at its face is z = W - h. At a face the two sides use
  // B_f = max(z_L, z_R) and h* = max(0, W - B_f), which gives zero flux at a
  // dry face and an exact lake at rest, including partially wet cells.
  const wEf = new Float64Array(N), wWf = new Float64Array(N), wNf = new Float64Array(N), wSf = new Float64Array(N);
  const hsE = new Float64Array(N), hsW = new Float64Array(N), hsN = new Float64Array(N), hsS = new Float64Array(N); // h* per cell side
  const Wd = new Float64Array(N); // surface for the dispersive stencils: still water level over dry cells
  // Face values of the reconstructed variables: [W, P, Q] x [E, W, N, S].
  const pEf = new Float64Array(N), pWf = new Float64Array(N), pNf = new Float64Array(N), pSf = new Float64Array(N);
  const qEf = new Float64Array(N), qWf = new Float64Array(N), qNf = new Float64Array(N), qSf = new Float64Array(N);
  const order = o.order ?? 2; // 2 = generalised minmod MUSCL, 4 = fourth-order MUSCL-TVD (Yamamoto & Daiguji 1993)
  const bComp = o.compression ?? 2;
  const minmod2 = (a, b) => (a > 0 && b > 0 ? Math.min(a, b) : a < 0 && b < 0 ? Math.max(a, b) : 0);
  const line = new Float64Array(Math.max(nx, ny) + 6), dl = new Float64Array(Math.max(nx, ny) + 6), dlb = new Float64Array(Math.max(nx, ny) + 6), dls = new Float64Array(Math.max(nx, ny) + 6);
  const faceL = new Float64Array(Math.max(nx, ny) + 6), faceR = new Float64Array(Math.max(nx, ny) + 6);

  // Reconstruct one line of n values (periodic or clamped) into the "left"
  // value at each cell's high face (faceL[i] = phi at i+1/2 seen from i) and
  // the "right" value at its low face (faceR[i] = phi at i-1/2 seen from i).
  function reconLine(n, periodic) {
    const at = (k) => line[periodic ? ((k % n) + n) % n : Math.min(n - 1, Math.max(0, k))];
    if (order === 2) {
      for (let i = 0; i < n; i++) {
        const sl = minmod3(theta * (at(i) - at(i - 1)), 0.5 * (at(i + 1) - at(i - 1)), theta * (at(i + 1) - at(i)));
        faceL[i] = at(i) + 0.5 * sl; faceR[i] = at(i) - 0.5 * sl;
      }
      return;
    }
    // differences at faces k+1/2 for k = -3 .. n+2, stored at dl[k+3]
    for (let k = -3; k <= n + 2; k++) dl[k + 3] = at(k + 1) - at(k);
    // limited differences
    for (let k = -2; k <= n + 1; k++) dlb[k + 3] = minmod3(dl[k + 3], bComp * dl[k + 2], bComp * dl[k + 4]);
    // fourth-order corrected differences, limited again
    for (let k = -1; k <= n; k++) dls[k + 3] = dl[k + 3] - (dlb[k + 4] - 2 * dlb[k + 3] + dlb[k + 2]) / 6;
    for (let i = 0; i < n; i++) {
      const dm = dls[i + 2], dp = dls[i + 3]; // corrected differences at i-1/2 and i+1/2
      const dmL = minmod2(dm, bComp * dp), dpL = minmod2(dp, bComp * dm);
      faceL[i] = at(i) + (dmL + 2 * dpL) / 6;
      faceR[i] = at(i) - (2 * dmL + dpL) / 6;
    }
  }

  function reconstruct(hh, pp, qq) {
    for (let c = 0; c < N; c++) { W[c] = B[c] + hh[c]; Wd[c] = hh[c] > hDry ? W[c] : wl; }
    // x direction, row by row
    for (let j = 0; j < ny; j++) {
      const r = j * nx;
      for (let i = 0; i < nx; i++) line[i] = W[r + i];
      reconLine(nx, periodicX); for (let i = 0; i < nx; i++) { wEf[r + i] = faceL[i]; wWf[r + i] = faceR[i]; }
      for (let i = 0; i < nx; i++) line[i] = pp[r + i];
      reconLine(nx, periodicX); for (let i = 0; i < nx; i++) { pEf[r + i] = faceL[i]; pWf[r + i] = faceR[i]; }
      for (let i = 0; i < nx; i++) line[i] = qq[r + i];
      reconLine(nx, periodicX); for (let i = 0; i < nx; i++) { qEf[r + i] = faceL[i]; qWf[r + i] = faceR[i]; }
    }
    // y direction, column by column
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) line[j] = W[j * nx + i];
      reconLine(ny, periodicY); for (let j = 0; j < ny; j++) { wNf[j * nx + i] = faceL[j]; wSf[j * nx + i] = faceR[j]; }
      for (let j = 0; j < ny; j++) line[j] = pp[j * nx + i];
      reconLine(ny, periodicY); for (let j = 0; j < ny; j++) { pNf[j * nx + i] = faceL[j]; pSf[j * nx + i] = faceR[j]; }
      for (let j = 0; j < ny; j++) line[j] = qq[j * nx + i];
      reconLine(ny, periodicY); for (let j = 0; j < ny; j++) { qNf[j * nx + i] = faceL[j]; qSf[j * nx + i] = faceR[j]; }
    }
    // Face depths (clipped at the continuous face bed) and velocities.
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      if (hh[c] <= hDry) {
        // Dry: no water at any face, the cell sees its own bed level.
        wEf[c] = wWf[c] = wNf[c] = wSf[c] = B[c];
        hE[c] = hW[c] = hN[c] = hS[c] = 0;
        uE[c] = uW[c] = uN[c] = uS[c] = 0; vE[c] = vW[c] = vN[c] = vS[c] = 0;
        continue;
      }
      const he = Math.max(0, wEf[c] - Bx[c]), hw = Math.max(0, wWf[c] - bedW(i, j));
      const hn = Math.max(0, wNf[c] - By[c]), hs = Math.max(0, wSf[c] - bedS(i, j));
      hE[c] = he; hW[c] = hw; hN[c] = hn; hS[c] = hs;
      uE[c] = vel(he, pEf[c]); uW[c] = vel(hw, pWf[c]); vE[c] = vel(he, qEf[c]); vW[c] = vel(hw, qWf[c]);
      uN[c] = vel(hn, pNf[c]); uS[c] = vel(hs, pSf[c]); vN[c] = vel(hn, qNf[c]); vS[c] = vel(hs, qSf[c]);
    }
  }

  // Central-upwind numerical flux (Kurganov & Petrova 2007) across a face
  // with left state (hl,ul,vl) and right state (hr,ur,vr); `n` picks the
  // normal direction: 0 = x face (flux F), 1 = y face (flux G). Result in fx.
  const fx = new Float64Array(3);
  function cuFlux(hl, ul, vl, hr, ur, vr, n) {
    const unl = n === 0 ? ul : vl, unr = n === 0 ? ur : vr;
    const cl_ = Math.sqrt(g * hl), cr_ = Math.sqrt(g * hr);
    const ap = Math.max(unl + cl_, unr + cr_, 0), am = Math.min(unl - cl_, unr - cr_, 0);
    const den = ap - am;
    if (den < 1e-9) { fx[0] = 0; fx[1] = 0; fx[2] = 0; return; }
    const ml = hl * unl, mr = hr * unr; // normal mass flux
    const pl = hl * ul, pr = hr * ur, ql = hl * vl, qr = hr * vr;
    const k = (ap * am) / den;
    fx[0] = (ap * ml - am * mr) / den + k * (hr - hl);
    const f2l = ml * ul + (n === 0 ? 0.5 * g * hl * hl : 0), f2r = mr * ur + (n === 0 ? 0.5 * g * hr * hr : 0);
    const f3l = ml * vl + (n === 1 ? 0.5 * g * hl * hl : 0), f3r = mr * vr + (n === 1 ? 0.5 * g * hr * hr : 0);
    fx[1] = (ap * f2l - am * f2r) / den + k * (pr - pl);
    fx[2] = (ap * f3l - am * f3r) / den + k * (qr - ql);
  }

  // Hydrostatic-reconstruction states and fluxes at every east face (stored
  // at the cell to the west) and north face (stored at the cell to the
  // south). Wall faces use a mirrored state.
  const FhW0 = new Float64Array(ny), FPW0 = new Float64Array(ny), FQW0 = new Float64Array(ny); // west wall faces
  const GhS0 = new Float64Array(nx), GPS0 = new Float64Array(nx), GQS0 = new Float64Array(nx); // south wall faces
  function fluxes() {
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      // x face between c and the cell to the east
      if (!periodicX && i === nx - 1) {
        hsE[c] = hE[c];
        cuFlux(hE[c], uE[c], vE[c], hE[c], -uE[c], vE[c], 0);
      } else {
        const cr = j * nx + wrapI(i + 1);
        const bf = Math.max(wEf[c] - hE[c], wWf[cr] - hW[cr]);
        const hl = Math.max(0, wEf[c] - bf), hr = Math.max(0, wWf[cr] - bf);
        hsE[c] = hl; hsW[cr] = hr;
        cuFlux(hl, uE[c], vE[c], hr, uW[cr], vW[cr], 0);
      }
      Fh[c] = fx[0]; FP[c] = fx[1]; FQ[c] = fx[2];
      // y face between c and the cell to the north
      if (!periodicY && j === ny - 1) {
        hsN[c] = hN[c];
        cuFlux(hN[c], uN[c], vN[c], hN[c], uN[c], -vN[c], 1);
      } else {
        const cn = wrapJ(j + 1) * nx + i;
        const bf = Math.max(wNf[c] - hN[c], wSf[cn] - hS[cn]);
        const hl = Math.max(0, wNf[c] - bf), hr = Math.max(0, wSf[cn] - bf);
        hsN[c] = hl; hsS[cn] = hr;
        cuFlux(hl, uN[c], vN[c], hr, uS[cn], vS[cn], 1);
      }
      Gh[c] = fx[0]; GP[c] = fx[1]; GQ[c] = fx[2];
    }
    if (!periodicX) for (let j = 0; j < ny; j++) {
      const c = j * nx; hsW[c] = hW[c];
      cuFlux(hW[c], -uW[c], vW[c], hW[c], uW[c], vW[c], 0);
      FhW0[j] = fx[0]; FPW0[j] = fx[1]; FQW0[j] = fx[2];
    }
    if (!periodicY) for (let i = 0; i < nx; i++) {
      hsS[i] = hS[i];
      cuFlux(hS[i], uS[i], -vS[i], hS[i], uS[i], vS[i], 1);
      GhS0[i] = fx[0]; GPS0[i] = fx[1]; GQS0[i] = fx[2];
    }
  }

  // Right-hand sides: flux divergence, bed slope, dispersive source terms.
  function rhs(hh, pp, qq) {
    reconstruct(hh, pp, qq);
    fluxes();
    const idx = 1 / dx, idx2 = idx * idx, idx3 = idx2 * idx;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      const cl = j * nx + wrapI(i - 1), cs = clampJ(j - 1) * nx + i;
      const dry = hh[c] <= hDry;
      // Flux divergence. Non-periodic edges act as walls (zero normal flux).
      const wallW = !periodicX && i === 0, wallS = !periodicY && j === 0;
      const fw = wallW ? FhW0[j] : Fh[cl], fe = Fh[c];
      const gs = wallS ? GhS0[i] : Gh[cs], gn = Gh[c];
      Rh[c] = -(fe - fw) * idx - (gn - gs) * idx;
      if (dry) { RP[c] = 0; RQ[c] = 0; continue; }
      const fpw = wallW ? FPW0[j] : FP[cl], gps = wallS ? GPS0[i] : GP[cs];
      const fqw = wallW ? FQW0[j] : FQ[cl], gqs = wallS ? GQS0[i] : GQ[cs];
      let rp = -(FP[c] - fpw) * idx - (GP[c] - gps) * idx;
      let rq = -(FQ[c] - fqw) * idx - (GQ[c] - gqs) * idx;
      // Hydrostatic-reconstruction sources: the face corrections g/2 (h^2 - h*^2)
      // and the in-cell bed slope -g h_bar (z_E - z_W)/dx with z = W - h the bed
      // the cell sees at its faces. Together with the fluxes this is exactly
      // balanced for still water over any bed, wet, dry or in between.
      const he = hE[c], hw = hW[c], hn = hN[c], hs = hS[c];
      rp += -0.5 * g * ((he * he - hsE[c] * hsE[c]) - (hw * hw - hsW[c] * hsW[c])) * idx;
      rp += -g * 0.5 * (he + hw) * ((wEf[c] - he) - (wWf[c] - hw)) * idx;
      rq += -0.5 * g * ((hn * hn - hsN[c] * hsN[c]) - (hs * hs - hsS[c] * hsS[c])) * idx;
      rq += -g * 0.5 * (hn + hs) * ((wNf[c] - hn) - (wSf[c] - hs)) * idx;
      // Dispersive terms (explicit part), only where chi > 0 and far enough from the edges.
      const ch = chi[c];
      const inX = periodicX || (i >= 2 && i <= nx - 3), inY = periodicY || (j >= 2 && j <= ny - 3);
      if (ch > 0 && inX && inY) {
        const dd = Math.max(d[c], 0);
        const i1 = wrapI(i + 1), i2 = wrapI(i + 2), im1 = wrapI(i - 1), im2 = wrapI(i - 2);
        const r = j * nx, rn = wrapJ(j + 1) * nx, rs = wrapJ(j - 1) * nx, rn2 = wrapJ(j + 2) * nx, rs2 = wrapJ(j - 2) * nx;
        // Surface derivatives (Wd = eta + wl on wet cells, so these are derivatives of eta).
        const Wxx = (Wd[r + i1] - 2 * Wd[c] + Wd[r + im1]) * idx2;
        const Wyy = (Wd[rn + i] - 2 * Wd[c] + Wd[rs + i]) * idx2;
        const Wxy = (Wd[rn + i1] - Wd[rn + im1] - Wd[rs + i1] + Wd[rs + im1]) * 0.25 * idx2;
        const Wxxx = (Wd[r + i2] - 2 * Wd[r + i1] + 2 * Wd[r + im1] - Wd[r + im2]) * 0.5 * idx3;
        const Wyyy = (Wd[rn2 + i] - 2 * Wd[rn + i] + 2 * Wd[rs + i] - Wd[rs2 + i]) * 0.5 * idx3;
        // W_xyy = d/dx (W_yy), W_xxy = d/dy (W_xx)
        const Wyy_e = (Wd[rn + i1] - 2 * Wd[r + i1] + Wd[rs + i1]) * idx2, Wyy_w = (Wd[rn + im1] - 2 * Wd[r + im1] + Wd[rs + im1]) * idx2;
        const Wxyy = (Wyy_e - Wyy_w) * 0.5 * idx;
        const Wxx_n = (Wd[rn + i1] - 2 * Wd[rn + i] + Wd[rn + im1]) * idx2, Wxx_s = (Wd[rs + i1] - 2 * Wd[rs + i] + Wd[rs + im1]) * idx2;
        const Wxxy = (Wxx_n - Wxx_s) * 0.5 * idx;
        const bgd = BQ * g * dd;
        const Ex = bgd * dd * dd * (Wxxx + Wxyy) + dd * dxs[c] * (2 * bgd * Wxx + bgd * Wyy) + dd * dys[c] * bgd * Wxy;
        const Ey = bgd * dd * dd * (Wyyy + Wxxy) + dd * dys[c] * (2 * bgd * Wyy + bgd * Wxx) + dd * dxs[c] * bgd * Wxy;
        // Lagged cross terms with the previous step's time derivatives of the other flux.
        const Qt_xy = (Qt[rn + i1] - Qt[rn + im1] - Qt[rs + i1] + Qt[rs + im1]) * 0.25 * idx2;
        const Qt_y = (Qt[rn + i] - Qt[rs + i]) * 0.5 * idx, Qt_x = (Qt[r + i1] - Qt[r + im1]) * 0.5 * idx;
        const Pt_xy = (Pt[rn + i1] - Pt[rn + im1] - Pt[rs + i1] + Pt[rs + im1]) * 0.25 * idx2;
        const Pt_x = (Pt[r + i1] - Pt[r + im1]) * 0.5 * idx, Pt_y = (Pt[rn + i] - Pt[rs + i]) * 0.5 * idx;
        const Mx = (BQ + 1 / 3) * dd * dd * Qt_xy + (1 / 6) * dd * dxs[c] * Qt_y + (1 / 6) * dd * dys[c] * Qt_x;
        const My = (BQ + 1 / 3) * dd * dd * Pt_xy + (1 / 6) * dd * dys[c] * Pt_x + (1 / 6) * dd * dxs[c] * Pt_y;
        rp += ch * (Ex + Mx);
        rq += ch * (Ey + My);
      }
      RP[c] = rp; RQ[c] = rq;
    }
  }

  // Apply L_x to P (into out) and L_y to Q (into out2).
  function applyL(pp, qq, outU, outV) {
    const idx2 = 1 / (dx * dx), idx = 1 / dx;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = j * nx + i;
      const ch = chi[c];
      if (ch <= 0) { outU[c] = pp[c]; outV[c] = qq[c]; continue; }
      const dd = Math.max(d[c], 0);
      const al = ch * (BQ + 1 / 3) * dd * dd * idx2;
      const bx = ch * dd * dxs[c] * idx / 6, by = ch * dd * dys[c] * idx / 6;
      const cl = j * nx + wrapI(i - 1), cr = j * nx + wrapI(i + 1), cs = clampJ(j - 1) * nx + i, cn = clampJ(j + 1) * nx + i;
      outU[c] = (1 + 2 * al) * pp[c] - (al - bx) * pp[cl] - (al + bx) * pp[cr];
      outV[c] = (1 + 2 * al) * qq[c] - (al - by) * qq[cs] - (al + by) * qq[cn];
    }
  }

  // Solve L_x P = U per row and L_y Q = V per column (Thomas algorithm; the
  // periodic case uses the Sherman-Morrison correction).
  function solveL(Uin, Vin, pp, qq) {
    const idx2 = 1 / (dx * dx), idx = 1 / dx;
    // rows (x)
    for (let j = 0; j < ny; j++) {
      const r = j * nx;
      // coefficients
      let anyDisp = false;
      for (let i = 0; i < nx; i++) {
        const c = r + i, ch = chi[c];
        if (ch > 0) anyDisp = true;
        const dd = Math.max(d[c], 0);
        const al = ch * (BQ + 1 / 3) * dd * dd * idx2, bx = ch * dd * dxs[c] * idx / 6;
        tmpA[i] = -(al - bx); // sub-diagonal (coefficient of P[i-1])
        tmpC[i] = -(al + bx); // super-diagonal (coefficient of P[i+1])
        diagRow[i] = 1 + 2 * al;
        rhsRow[i] = Uin[c];
      }
      if (!anyDisp) { for (let i = 0; i < nx; i++) pp[r + i] = Uin[r + i]; continue; }
      thomas(nx, tmpA, diagRow, tmpC, rhsRow, periodicX, solRow);
      for (let i = 0; i < nx; i++) pp[r + i] = solRow[i];
    }
    // columns (y)
    for (let i = 0; i < nx; i++) {
      let anyDisp = false;
      for (let j = 0; j < ny; j++) {
        const c = j * nx + i, ch = chi[c];
        if (ch > 0) anyDisp = true;
        const dd = Math.max(d[c], 0);
        const al = ch * (BQ + 1 / 3) * dd * dd * idx2, by = ch * dd * dys[c] * idx / 6;
        tmpA[j] = -(al - by); tmpC[j] = -(al + by); diagRow[j] = 1 + 2 * al; rhsRow[j] = Vin[c];
      }
      if (!anyDisp) { for (let j = 0; j < ny; j++) qq[j * nx + i] = Vin[j * nx + i]; continue; }
      thomas(ny, tmpA, diagRow, tmpC, rhsRow, periodicY, solRow);
      for (let j = 0; j < ny; j++) qq[j * nx + i] = solRow[j];
    }
  }
  const diagRow = new Float64Array(Math.max(nx, ny)), rhsRow = new Float64Array(Math.max(nx, ny)), solRow = new Float64Array(Math.max(nx, ny));
  const cp = new Float64Array(Math.max(nx, ny)), dp = new Float64Array(Math.max(nx, ny)), zz = new Float64Array(Math.max(nx, ny)), uu = new Float64Array(Math.max(nx, ny));

  function thomas(n, a, b, c, r, periodic, x) {
    if (!periodic) {
      cp[0] = c[0] / b[0]; dp[0] = r[0] / b[0];
      for (let i = 1; i < n; i++) { const m = b[i] - a[i] * cp[i - 1]; cp[i] = c[i] / m; dp[i] = (r[i] - a[i] * dp[i - 1]) / m; }
      x[n - 1] = dp[n - 1];
      for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
      return;
    }
    // Cyclic: Sherman-Morrison. Perturb b[0] and b[n-1], solve twice.
    const gamma = -b[0];
    const b0 = b[0], bn = b[n - 1];
    b[0] = b0 - gamma; b[n - 1] = bn - (c[n - 1] * a[0]) / gamma;
    // solve A' y = r
    cp[0] = c[0] / b[0]; dp[0] = r[0] / b[0];
    for (let i = 1; i < n; i++) { const m = b[i] - a[i] * cp[i - 1]; cp[i] = c[i] / m; dp[i] = (r[i] - a[i] * dp[i - 1]) / m; }
    zz[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) zz[i] = dp[i] - cp[i] * zz[i + 1];
    // solve A' q = u, u = (gamma, 0, ..., a[0])  (note a[0] couples x[0] to x[n-1])
    for (let i = 0; i < n; i++) uu[i] = 0;
    uu[0] = gamma; uu[n - 1] = a[0];
    cp[0] = c[0] / b[0]; dp[0] = uu[0] / b[0];
    for (let i = 1; i < n; i++) { const m = b[i] - a[i] * cp[i - 1]; cp[i] = c[i] / m; dp[i] = (uu[i] - a[i] * dp[i - 1]) / m; }
    uu[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) uu[i] = dp[i] - cp[i] * uu[i + 1];
    const fact = (zz[0] + (c[n - 1] * zz[n - 1]) / gamma) / (1 + uu[0] + (c[n - 1] * uu[n - 1]) / gamma);
    for (let i = 0; i < n; i++) x[i] = zz[i] - fact * uu[i];
    b[0] = b0; b[n - 1] = bn;
  }

  // One forward-Euler stage from (hh, pp, qq) by dt into (ho, po, qo), with
  // the implicit dispersion solve and semi-implicit friction.
  function stage(hh, pp, qq, dt, ho, po, qo) {
    rhs(hh, pp, qq);
    applyL(pp, qq, U, V);
    for (let c = 0; c < N; c++) {
      ho[c] = Math.max(0, hh[c] + dt * Rh[c]);
      U[c] += dt * RP[c];
      V[c] += dt * RQ[c];
    }
    solveL(U, V, po, qo);
    for (let c = 0; c < N; c++) {
      if (ho[c] <= hDry) { po[c] = 0; qo[c] = 0; continue; }
      // Manning friction, semi-implicit: P/(1 + dt g n^2 |u| / h^(4/3))
      const hc = ho[c];
      const u = po[c] / hc, v = qo[c] / hc;
      const sp = Math.hypot(u, v);
      const fr = 1 + (dt * g * nMan[c] * nMan[c] * sp) / Math.pow(Math.max(hc, eps), 4 / 3);
      po[c] /= fr; qo[c] /= fr;
      // Cap unphysical speeds in films.
      const umax = 4 * Math.sqrt(g * Math.max(hc, 0.01)) + 1;
      if (sp / fr > umax) { const s = umax / (sp / fr); po[c] *= s; qo[c] *= s; }
    }
  }

  // Breaking detection: flag cells whose surface is high relative to the depth
  // or rising too fast (Kennedy et al. 2000 front steepness), keep the flag
  // while the front is still steep, and let it fade with a short memory.
  function updateBreaking(dt) {
    if (!useBreak) return;
    const decay = Math.exp(-dt / brk.memory);
    for (let c = 0; c < N; c++) {
      let f = F[c] * decay;
      const hc = h[c];
      if (hc > 0.1 && relaxW(c) < 0.5) {
        const eta = hc + B[c] - wl;
        const etat = (hc - hPrev[c]) / dt;
        const cg = Math.sqrt(g * hc);
        const ratio = eta / Math.max(d[c], 0.3); // Tonelli & Petti: crest elevation over still-water depth
        const fo = Math.min(brk.onsetSlope, brk.slopeRes * hc / dx);
        if (etat > 0 && (ratio > brk.onsetRatio || etat > fo * cg)) f = 1;
        else if (etat > brk.keepSlope * cg && f > 0.05) f = Math.max(f, 0.8);
      } else if (hc <= 0.1) f = 0;
      F[c] = f;
    }
    updateChi();
  }
  const relaxW = (c) => (relax ? relax[c] : 0);

  // Relaxation zones: blend toward the target field (Mayer et al. 1998 weights).
  function applyRelaxation(tt) {
    if (!relax || !target) return;
    target(tt, etaT, PT, QT);
    for (let c = 0; c < N; c++) {
      const s = relax[c];
      if (s <= 0) continue;
      const w = (Math.exp(Math.pow(s, 3.5)) - 1) / (Math.E - 1); // 0 at the inner edge .. 1 at the domain edge
      const hT = Math.max(0, wl + etaT[c] - B[c]);
      h[c] = (1 - w) * h[c] + w * hT;
      if (hT <= hDry) { P[c] = (1 - w) * P[c]; Q[c] = (1 - w) * Q[c]; }
      else { P[c] = (1 - w) * P[c] + w * PT[c]; Q[c] = (1 - w) * Q[c] + w * QT[c]; }
      if (h[c] <= hDry) { P[c] = 0; Q[c] = 0; }
    }
  }

  // Largest wave speed on the grid, for the time step.
  function maxSpeed() {
    let m = 0;
    for (let c = 0; c < N; c++) {
      const hc = h[c];
      if (hc <= hDry) continue;
      const s = Math.hypot(P[c], Q[c]) / hc + Math.sqrt(g * hc);
      if (s > m) m = s;
    }
    return m;
  }

  // Advance one step of dt (seconds) with SSP-RK2 (Heun).
  function step(dt) {
    // Lagged time derivatives for the cross terms.
    if (dtPrev > 0) for (let c = 0; c < N; c++) { Pt[c] = (P[c] - PPrev[c]) / dtPrev; Qt[c] = (Q[c] - QPrev[c]) / dtPrev; }
    hPrev.set(h); PPrev.set(P); QPrev.set(Q);
    // stage 1
    stage(h, P, Q, dt, h0, P0, Q0);
    // stage 2 into h,P,Q (temporaries reuse Rh etc. inside)
    stage(h0, P0, Q0, dt, h, P, Q);
    for (let c = 0; c < N; c++) {
      h[c] = 0.5 * (hPrev[c] + h[c]);
      P[c] = 0.5 * (PPrev[c] + P[c]);
      Q[c] = 0.5 * (QPrev[c] + Q[c]);
      if (h[c] <= hDry) { P[c] = 0; Q[c] = 0; }
    }
    t += dt; dtPrev = dt;
    applyRelaxation(t);
    updateBreaking(dt);
  }

  // Volume of water in the domain (m^3).
  function volume() { let s = 0; for (let c = 0; c < N; c++) s += h[c]; return s * dx * dx; }

  setBed(o.bed, o.manning ?? 0.02);
  reset(o.eta);

  return {
    nx, ny, dx, g,
    h, P, Q, F, chi, B, d,
    get t() { return t; },
    get wl() { return wl; },
    setBed, setWaterLevel, reset, step, maxSpeed, volume,
    setTarget: (fn) => { target = fn; },
    eta: (out) => { const e = out || new Float64Array(N); for (let c = 0; c < N; c++) e[c] = h[c] > hDry ? h[c] + B[c] - wl : B[c] - wl; return e; },
    surface: (out) => { const e = out || new Float64Array(N); for (let c = 0; c < N; c++) e[c] = h[c] + B[c]; return e; },
    cfl: (nu) => (nu ?? 0.4) * dx / Math.max(maxSpeed(), 1e-6),
  };
}

/** Madsen-Sorensen linear dispersion relation: omega^2 for wavenumber k at depth d. */
export function omegaMS(k, d, g = G) {
  const kd2 = k * k * d * d;
  return Math.sqrt((g * d * k * k * (1 + BQ * kd2)) / (1 + (BQ + 1 / 3) * kd2));
}

/** Wavenumber for angular frequency w at depth d under the Madsen-Sorensen relation (Newton). */
export function wavenumberMS(w, d, g = G) {
  let k = w / Math.sqrt(g * d);
  for (let it = 0; it < 40; it++) {
    const f = omegaMS(k, d, g) - w;
    const dk = 1e-6 * k;
    const df = (omegaMS(k + dk, d, g) - omegaMS(k - dk, d, g)) / (2 * dk);
    const kn = k - f / df;
    if (Math.abs(kn - k) < 1e-12) { k = kn; break; }
    k = kn;
  }
  return k;
}

/** Airy wavenumber for angular frequency w at depth d (Newton on w^2 = g k tanh(k d)). */
export function wavenumberAiry(w, d, g = G) {
  let k = (w * w) / g;
  for (let it = 0; it < 40; it++) {
    const th = Math.tanh(k * d);
    const f = g * k * th - w * w;
    const df = g * th + g * k * d * (1 - th * th);
    const kn = k - f / df;
    if (Math.abs(kn - k) < 1e-12) { k = kn; break; }
    k = kn;
  }
  return k;
}

/** Relaxation-zone weight field: 1 at the domain edge, 0 beyond the zone widths (cells). */
export function relaxField(nx, ny, { offshore = 0, lateral = 0, inland = 0 }) {
  const r = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let s = 0;
    if (offshore > 0 && j < offshore) s = Math.max(s, (offshore - j) / offshore);
    if (inland > 0 && j >= ny - inland) s = Math.max(s, (j - (ny - inland) + 1) / inland);
    if (lateral > 0) {
      if (i < lateral) s = Math.max(s, (lateral - i) / lateral);
      if (i >= nx - lateral) s = Math.max(s, (i - (nx - lateral) + 1) / lateral);
    }
    r[j * nx + i] = Math.min(1, s);
  }
  return r;
}
