// CPU crest tracker: turns the per-frame crest-scan records (contract
// section 5) into tracked crest lines with a filtered cross-shore position
// and a phase speed per column.
//
// Pure JS, no three.js. Coordinates: "grid" metres (x alongshore, y
// cross-shore, y = 0 offshore) as in contract section 1. A column `col` of the
// records sits at grid x = (col + 0.5) * dxs; a solver row J at grid
// y = (J - MO + 0.5) * dxs.
//
// Model:
//  1. Link the records across columns into crest lines: a record continues
//     the line of the previous column when its row is within `linkCells`
//     (3) cells of that line's last row (same slot preferred, any slot
//     accepted so the line survives the slot shift that happens when a new
//     crest enters the scan offshore). A hole of up to `linkCells` columns
//     without a matching record is bridged: the line stays open and the hole
//     columns get records interpolated from the two sides (the scan skips a
//     column where the crest is too flat for its prominence test; one crest
//     must stay one line and one lip). A longer hole splits the line.
//  2. Match the lines to the tracked crests of the previous frames by
//     nearest predicted position (the crest advances about c*dt per frame).
//     Several lines may feed one tracked crest (a line split by a column
//     without a crest); a line that matches nothing starts a new crest.
//  3. Per column estimate the cross-shore speed from the instants the record
//     row steps: the crest crosses the boundary between two cells at each
//     step, so with two or more steps inside the last `fitWindow` seconds the
//     speed is dxs * (rows stepped) / (time between the first and last step)
//     and the position is the last boundary plus the travel since. Before the
//     second step (a column just acquired, or a slow crest) a least-squares
//     line through the (t, y) history of the last `fitWindow` seconds is
//     used instead, blended with the linear-theory speed sqrt(g hC)(1 + etaC/2hC)
//     while the history is short.
//  4. A column that is not refreshed for `dropAfter` seconds is cleared; a
//     crest with no refreshed column for `dropAfter` seconds is dropped.
//
// Per column the crest also keeps the previous refresh's FC (`FCprev`, -1
// before the first record) so the lip model can detect the frame FC crosses
// its threshold, plus the lip bookkeeping (`broken`, `typeF`, `armed`,
// `lipped`, `calm`) that the lip model writes.
export const K = 6; // slots per column (section 5)
export const CH = 8; // channels per record
const NH = 32; // history ring length per column (samples)
const NS = 8; // step ring length per column (row changes)

const G = 9.81;

export function createTracker(lay, opts = {}) {
  const { V, dxs, MO } = lay;
  const maxCrests = opts.maxCrests ?? 32;
  const linkCells = opts.linkCells ?? 3;
  const dropAfter = opts.dropAfter ?? 0.5;
  const fitWindow = opts.fitWindow ?? 1.2;
  const minLineCols = opts.minLineCols ?? 2;
  const matchDist = opts.matchDist ?? 3 * dxs + 0.5; // m, line-to-prediction tolerance
  const etaTau = opts.etaSmooth ?? 0.12; // s, time constant of the elevation smoothing

  let nextId = 1;
  let tNow = 0;

  function newCrest() {
    return {
      id: 0, t0: 0, lastSeen: 0, c0: 0, c1: -1, n: 0, live: false,
      age: new Float32Array(V), // s since the column was last refreshed (-1: never / cleared)
      seen: new Uint8Array(V), // refreshed this frame
      interp: new Uint8Array(V), // this frame's record of the column was interpolated across a hole
      J: new Float32Array(V), // raw row this frame
      yRaw: new Float32Array(V),
      etaC: new Float32Array(V), hC: new Float32Array(V), FC: new Float32Array(V), etaT: new Float32Array(V),
      FCprev: new Float32Array(V), // FC at the previous refresh of the column (-1: this is the first)
      Jf: new Float32Array(V), sF: new Float32Array(V), uC: new Float32Array(V),
      y: new Float32Array(V), // filtered cross-shore position (grid m)
      cy: new Float32Array(V), // cross-shore speed (m/s)
      c: new Float32Array(V), // phase speed along the crest normal (m/s)
      span: new Float32Array(V), // seconds of history behind the fit
      nx: new Float32Array(V), ny: new Float32Array(V), // crest normal (grid), shoreward
      ht: new Float32Array(V * NH * 2), hn: new Uint8Array(V), hp: new Uint8Array(V),
      st: new Float32Array(V * NS * 2), sn: new Uint8Array(V), sp: new Uint8Array(V), // step ring: (time, new row)
      lastJ: new Float32Array(V), lastT: new Float32Array(V), // row and time of the previous refresh
      // per-column user fields filled by the lip model (kept here so they persist per crest)
      H: new Float32Array(V), xi: new Float32Array(V), tanB: new Float32Array(V), type: new Int8Array(V),
      broken: new Uint8Array(V), // FC has crossed the lip threshold on this column (or it was born above it)
      typeF: new Int8Array(V), // breaker type frozen at that moment
      armed: new Float32Array(V), // time FC rose above the threshold (-1: not armed); a lip may take the column for a short while
      lipped: new Float32Array(V), // time a lip was last spawned on this column (-1: none)
      calm: new Float32Array(V), // seconds the column has had a low FC (re-arms the lip)
      scratch: new Float32Array(V),
    };
  }
  const pool = [];
  for (let i = 0; i < maxCrests; i++) pool.push(newCrest());
  const crests = [];

  function resetCrest(cr, t) {
    cr.id = nextId++;
    cr.t0 = t; cr.lastSeen = t; cr.c0 = 0; cr.c1 = -1; cr.n = 0; cr.live = true;
    cr.age.fill(-1); cr.seen.fill(0); cr.interp.fill(0); cr.hn.fill(0); cr.hp.fill(0); cr.sn.fill(0); cr.sp.fill(0); cr.span.fill(0);
    cr.y.fill(NaN); cr.cy.fill(0); cr.c.fill(0);
    cr.lipped.fill(-1); cr.calm.fill(0); cr.type.fill(0); cr.xi.fill(0); cr.H.fill(0);
    cr.broken.fill(0); cr.typeF.fill(0); cr.armed.fill(-1); cr.FCprev.fill(-1);
    cr.etaC.fill(0); cr.etaT.fill(0); cr.hC.fill(0); cr.FC.fill(0);
    cr.nx.fill(0); cr.ny.fill(1);
  }
  function clearColumn(cr, col) {
    cr.age[col] = -1; cr.hn[col] = 0; cr.sn[col] = 0; cr.y[col] = NaN;
    cr.FCprev[col] = -1; cr.broken[col] = 0; cr.armed[col] = -1; cr.lipped[col] = -1; cr.calm[col] = 0;
  }

  // --- 1. link records across columns into lines ---------------------------
  // A line: { c0, c1, n, last, gap, J: Float32Array(V) (J per column, NaN elsewhere), slot: Int8Array(V),
  //           srcL/srcR: Int16Array(V) (for an interpolated column: the real columns on both sides) }
  const linePool = [];
  function newLine() { return { c0: 0, c1: -1, n: 0, last: -1, gap: 0, J: new Float32Array(V).fill(NaN), slot: new Int8Array(V), srcL: new Int16Array(V), srcR: new Int16Array(V), assigned: false }; }
  const lines = [];
  const open = []; // lines open at the previous column: indices into lines
  const nextOpen = []; // scratch for the next column's open lines
  const colJ = new Float32Array(K), colSlot = new Int8Array(K);
  const used = new Uint8Array(K);

  function linkColumns(rec) {
    for (const l of lines) linePool.push(l);
    lines.length = 0;
    open.length = 0;
    for (let col = 0; col < V; col++) {
      let m = 0;
      for (let s = 0; s < K; s++) {
        const b = (col * K + s) * CH;
        if (rec[b] >= 0 && Number.isFinite(rec[b])) { colJ[m] = rec[b]; colSlot[m] = s; used[m] = 0; m++; }
      }
      // match the open lines (from col-1, or across a short hole) to this column's records: greedy by row distance, same slot first
      const next = nextOpen;
      next.length = 0;
      for (const li of open) {
        const L = lines[li];
        const Jp = L.J[L.last];
        let best = -1, bd = linkCells + 0.5 * L.gap + 1e-6;
        for (let q = 0; q < m; q++) {
          if (used[q]) continue;
          let d = Math.abs(colJ[q] - Jp);
          if (colSlot[q] !== L.slot[L.last]) d += 0.01; // prefer the same slot on ties
          if (d < bd) { bd = d; best = q; }
        }
        if (best >= 0) {
          used[best] = 1;
          if (L.gap) { // bridge the hole: interpolate the row, remember both sides for the record channels
            const cl = L.last, J0 = L.J[cl], J1 = colJ[best];
            for (let cc = cl + 1; cc < col; cc++) {
              L.J[cc] = J0 + ((J1 - J0) * (cc - cl)) / (col - cl);
              L.slot[cc] = L.slot[cl]; L.srcL[cc] = cl; L.srcR[cc] = col;
            }
          }
          L.J[col] = colJ[best]; L.slot[col] = colSlot[best]; L.srcL[col] = col; L.srcR[col] = col;
          L.c1 = col; L.n++; L.last = col; L.gap = 0;
          next.push(li);
        } else if (L.gap < linkCells && col < V - 1) {
          L.gap++;
          next.push(li); // stays open across the hole
        }
      }
      for (let q = 0; q < m; q++) {
        if (used[q]) continue;
        const L = linePool.length ? linePool.pop() : newLine();
        L.c0 = col; L.c1 = col; L.n = 1; L.last = col; L.gap = 0; L.assigned = false;
        L.J[col] = colJ[q]; L.slot[col] = colSlot[q]; L.srcL[col] = col; L.srcR[col] = col;
        lines.push(L);
        next.push(lines.length - 1);
      }
      open.length = 0;
      for (let q = 0; q < next.length; q++) open.push(next[q]);
    }
    // drop the stubs
    for (let i = lines.length - 1; i >= 0; i--) if (lines[i].n < minLineCols) { linePool.push(lines[i]); lines[i] = lines[lines.length - 1]; lines.pop(); }
  }

  // --- 3. per-column history, steps and fit ----------------------------------
  function pushHist(cr, col, t, y) {
    const base = col * NH * 2;
    const p = cr.hp[col];
    cr.ht[base + p * 2] = t; cr.ht[base + p * 2 + 1] = y;
    cr.hp[col] = (p + 1) % NH;
    if (cr.hn[col] < NH) cr.hn[col]++;
  }
  function pushStep(cr, col, t, J) {
    const base = col * NS * 2;
    const p = cr.sp[col];
    cr.st[base + p * 2] = t; cr.st[base + p * 2 + 1] = J;
    cr.sp[col] = (p + 1) % NS;
    if (cr.sn[col] < NS) cr.sn[col]++;
  }
  // Writes [y, cy, span, fromSteps] into fitOut.
  const fitOut = new Float64Array(4);
  function fitColumn(cr, col, t) {
    // steps inside the window
    const ns = cr.sn[col], sbase = col * NS * 2;
    let tLast = 0, JLast = 0, tFirst = 0, JFirst = 0, m = 0;
    for (let q = 0; q < ns; q++) {
      const idx = sbase + ((cr.sp[col] - 1 - q + 2 * NS) % NS) * 2;
      const ts = cr.st[idx];
      if (t - ts > fitWindow) break;
      if (q === 0) { tLast = ts; JLast = cr.st[idx + 1]; }
      tFirst = ts; JFirst = cr.st[idx + 1]; m++;
    }
    if (m >= 2 && tLast - tFirst > 0.1 && JLast - JFirst >= 1) {
      const cy = ((JLast - JFirst) * dxs) / (tLast - tFirst);
      const y = (JLast - MO) * dxs + cy * (t - tLast); // the crest crossed the lower boundary of row JLast at tLast
      const span = t - tFirst;
      cr.span[col] = span;
      fitOut[0] = y; fitOut[1] = cy; fitOut[2] = span; fitOut[3] = 1;
      return fitOut;
    }
    // least squares through the (t, y) history
    const n = cr.hn[col];
    const base = col * NH * 2;
    let St = 0, Sy = 0, Stt = 0, Sty = 0, k = 0, tmin = 0;
    for (let q = 0; q < n; q++) {
      const idx = base + ((cr.hp[col] - 1 - q + 2 * NH) % NH) * 2;
      const dtq = cr.ht[idx] - t;
      if (-dtq > fitWindow) break;
      const y = cr.ht[idx + 1];
      St += dtq; Sy += y; Stt += dtq * dtq; Sty += dtq * y; k++; tmin = dtq;
    }
    const span = -tmin;
    let b = 0, a = Sy / Math.max(k, 1);
    if (k >= 3 && span > 0.08) {
      const den = k * Stt - St * St;
      if (den > 1e-9) { b = (k * Sty - St * Sy) / den; a = (Sy - b * St) / k; }
    }
    cr.span[col] = span;
    fitOut[0] = a; fitOut[1] = b; fitOut[2] = span; fitOut[3] = 0;
    return fitOut;
  }

  // --- the frame update ---------------------------------------------------------
  const assignMap = []; // per line: crest index or -1
  const chan = new Float32Array(CH);

  function update(records, t, dt) {
    tNow = t;
    linkColumns(records);
    // predicted position per column for the live crests (y already holds the fit value at the last update; extrapolate)
    for (const cr of crests) {
      for (let col = cr.c0; col <= cr.c1; col++) if (cr.age[col] >= 0) cr.scratch[col] = cr.y[col] + cr.cy[col] * dt;
    }
    // sort lines by length (longest first) so the main pieces claim their crest first
    lines.sort((p, q) => q.n - p.n);
    assignMap.length = lines.length;
    for (let li = 0; li < lines.length; li++) {
      const L = lines[li];
      let best = -1, bestCost = Infinity;
      for (let ci = 0; ci < crests.length; ci++) {
        const cr = crests[ci];
        // overlap
        let sum = 0, m = 0;
        const a = Math.max(L.c0, cr.c0), b = Math.min(L.c1, cr.c1);
        for (let col = a; col <= b; col++) {
          if (cr.age[col] < 0 || !Number.isFinite(L.J[col])) continue;
          const y = (L.J[col] - MO + 0.5) * dxs;
          sum += Math.abs(y - cr.scratch[col]); m++;
        }
        let cost;
        if (m > 0) cost = sum / m;
        else {
          // no overlapping column: accept an adjacent extension of the crest within a few columns
          let ec = -1;
          if (L.c0 > cr.c1 && L.c0 - cr.c1 <= 4) ec = cr.c1;
          else if (L.c1 < cr.c0 && cr.c0 - L.c1 <= 4) ec = cr.c0;
          if (ec < 0 || cr.age[ec] < 0) continue;
          const lc = L.c0 > cr.c1 ? L.c0 : L.c1;
          cost = Math.abs((L.J[lc] - MO + 0.5) * dxs - cr.scratch[ec]) + 0.5 * dxs;
        }
        if (cost < bestCost && cost <= matchDist) { bestCost = cost; best = ci; }
      }
      assignMap[li] = best;
    }
    // apply the assignments; columns already claimed this frame by a longer line are skipped
    for (const cr of crests) cr.n = 0;
    for (let li = 0; li < lines.length; li++) {
      const L = lines[li];
      let cr;
      if (assignMap[li] >= 0) cr = crests[assignMap[li]];
      else {
        // new crest
        if (L.n < minLineCols || !pool.length) continue;
        cr = pool.pop();
        resetCrest(cr, t);
        crests.push(cr);
      }
      let wrote = 0;
      for (let col = L.c0; col <= L.c1; col++) {
        if (cr.seen[col]) continue; // a longer line already refreshed this column of this crest
        const J = L.J[col];
        if (!Number.isFinite(J)) continue;
        const cl = L.srcL[col], crr = L.srcR[col];
        if (cl === crr) {
          const b = (col * K + L.slot[col]) * CH;
          for (let ch = 1; ch < CH; ch++) chan[ch] = records[b + ch];
          cr.interp[col] = 0;
        } else { // interpolated across a hole
          const bl = (cl * K + L.slot[cl]) * CH, br = (crr * K + L.slot[crr]) * CH, wgt = (col - cl) / (crr - cl);
          for (let ch = 1; ch < CH; ch++) chan[ch] = records[bl + ch] * (1 - wgt) + records[br + ch] * wgt;
          cr.interp[col] = 1;
        }
        const y = (J - MO + 0.5) * dxs;
        cr.J[col] = J; cr.yRaw[col] = y;
        const first = cr.age[col] < 0;
        const k = first ? 1 : Math.min(1, dt / etaTau);
        cr.FCprev[col] = first ? -1 : cr.FC[col];
        cr.etaC[col] += (chan[1] - cr.etaC[col]) * k;
        cr.hC[col] += (chan[2] - cr.hC[col]) * k;
        cr.FC[col] = chan[3]; // raw: the spawn rule wants the frame FC crosses its threshold
        cr.etaT[col] += (chan[4] - cr.etaT[col]) * k;
        cr.Jf[col] = chan[5]; cr.sF[col] = chan[6]; cr.uC[col] = chan[7];
        if (first) { cr.hn[col] = 0; cr.hp[col] = 0; cr.sn[col] = 0; cr.sp[col] = 0; }
        else if (J !== cr.lastJ[col]) pushStep(cr, col, 0.5 * (t + cr.lastT[col]), J); // the row changed since the last refresh: the crest crossed a cell boundary in between
        cr.lastJ[col] = J; cr.lastT[col] = t;
        cr.age[col] = 0; cr.seen[col] = 1;
        pushHist(cr, col, t, y);
        wrote++;
      }
      if (wrote) cr.lastSeen = t;
      // the J values of the line are consumed; clear for reuse
      for (let col = L.c0; col <= L.c1; col++) L.J[col] = NaN;
    }
    // age the columns, clear the stale ones, drop the dead crests, refit
    for (let ci = crests.length - 1; ci >= 0; ci--) {
      const cr = crests[ci];
      let c0 = V, c1 = -1, n = 0;
      for (let col = 0; col < V; col++) {
        if (cr.age[col] < 0) continue;
        if (!cr.seen[col]) cr.age[col] += dt;
        cr.seen[col] = 0;
        if (cr.age[col] > dropAfter) { clearColumn(cr, col); continue; }
        n++;
        if (col < c0) c0 = col;
        if (col > c1) c1 = col;
      }
      cr.c0 = c0; cr.c1 = c1; cr.n = n;
      if (n === 0 || t - cr.lastSeen > dropAfter) {
        cr.live = false;
        pool.push(cr);
        crests[ci] = crests[crests.length - 1];
        crests.pop();
        continue;
      }
      // fits
      for (let col = c0; col <= c1; col++) {
        if (cr.age[col] < 0) continue;
        if (cr.age[col] > 0) { cr.y[col] += cr.cy[col] * dt; continue; } // not refreshed: coast
        const fo = fitColumn(cr, col, t), a = fo[0], b = fo[1], span = fo[2], fromSteps = fo[3] > 0;
        cr.y[col] = a;
        const fb = Math.sqrt(G * Math.max(cr.hC[col], 0.05)) * (1 + (0.5 * cr.etaC[col]) / Math.max(cr.hC[col], 0.05));
        const w = fromSteps ? Math.min(1, Math.max(0, (span - 0.15) / 0.4)) : Math.min(1, Math.max(0, (span - 0.25) / 0.6));
        const bc = Math.min(Math.max(b, 0.4 * fb), 1.8 * fb);
        cr.cy[col] = fb * (1 - w) + bc * w;
      }
      // alongshore smoothing of the speed (+-4 columns) and the position (+-1)
      smooth(cr, cr.cy, 4);
      smooth(cr, cr.y, 1);
      // crest normals and the speed along them
      for (let col = c0; col <= c1; col++) {
        if (cr.age[col] < 0) continue;
        const l = col > c0 && cr.age[col - 1] >= 0 ? col - 1 : col;
        const r = col < c1 && cr.age[col + 1] >= 0 ? col + 1 : col;
        let tx = (r - l) * dxs, ty = r === l ? 0 : cr.y[r] - cr.y[l];
        if (r === l) { tx = 1; ty = 0; }
        const inv = 1 / Math.hypot(tx, ty);
        tx *= inv; ty *= inv;
        // normal, shoreward
        let nx = -ty, ny = tx;
        if (ny < 0) { nx = -nx; ny = -ny; }
        cr.nx[col] = nx; cr.ny[col] = ny;
        cr.c[col] = cr.cy[col] * ny; // the line translates in y at cy; the normal speed is cy * cos(theta)
      }
    }
  }

  function smooth(cr, arr, w) {
    const s = cr.scratch;
    for (let col = cr.c0; col <= cr.c1; col++) {
      if (cr.age[col] < 0) continue;
      let sum = 0, m = 0;
      for (let q = -w; q <= w; q++) {
        const cc = col + q;
        if (cc < cr.c0 || cc > cr.c1 || cr.age[cc] < 0) continue;
        if (Math.abs(cc - col) > 0 && Math.abs(arr[cc] - arr[col]) > 4 * dxs) continue; // not the same crest piece
        sum += arr[cc]; m++;
      }
      s[col] = sum / m;
    }
    for (let col = cr.c0; col <= cr.c1; col++) if (cr.age[col] >= 0) arr[col] = s[col];
  }

  // Grid coordinates of a column / of a crest sample (for consumers)
  function colX(col) { return (col + 0.5) * dxs; }

  return {
    crests, update, colX, lay,
    get time() { return tNow; },
    options: { maxCrests, linkCells, dropAfter, fitWindow, minLineCols, matchDist },
    // Tests: the lines linked in the last frame
    get lines() { return lines; },
  };
}
