// WebGL2 / three.js 0.160 port of src/boussinesq.js (sections 2, 3, 4, 5 of
// docs/engine-contract.md). The reference's step() is reproduced pass by pass
// on RGBA32F render targets; the tridiagonal dispersive operator is solved
// with parallel cyclic reduction (both the row systems for P and the column
// systems for Q in the same passes).
//
// Passes per step (r = 1, nx = 335, ny = 377, 9 PCR levels):
//   1       target field at t + dt (relaxation zones)
//   2..14   stage 1: recon, flux, rhs, 9 x PCR, finish (friction, speed cap)
//   15..27  stage 2: recon, flux, rhs, 9 x PCR, final (Heun average, relaxation, breaking, Pt/Qt)
//   28..29  chi: max F along x, max along y + tridiagonal coefficients
import * as THREE from 'three';
import { N, DX } from '../terrain.js';
import { NCOMP } from '../waves.js';
import { layout as makeLayout, cornerBed, cornerManning } from './layout.js';
import * as SH from './shaders.js';

export const G = 9.81;
export const BQ = 1 / 15;
export const CREST_SLOTS = 6;

const RT_OPTS = { type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false, generateMipmaps: false };

/** Can this renderer run the solver? (WebGL2 with float colour buffers.) */
export function gpuSolverSupported(renderer) {
  const gl = renderer.getContext();
  return !!(renderer.capabilities.isWebGL2 && gl.getExtension('EXT_color_buffer_float'));
}

/**
 * @param {THREE.WebGLRenderer} renderer
 * @param {object} o  { r } for the real layout; tests may pass { grid: { nx, ny, dx }, relax: { offshore, lateral, inland } }
 *   plus the reference's switches: wl, dispersion, breaking, breakingParams, dispDepth, theta, hDry, eps.
 */
export function createGpuSolver(renderer, o = {}) {
  if (!gpuSolverSupported(renderer)) throw new Error('GPU solver needs WebGL2 with EXT_color_buffer_float');
  const r = o.r ?? 1;
  const lay = o.grid ? { r: 0, dxs: o.grid.dx, V: o.grid.nx, ML: 0, MO: 0, MI: 0, nx: o.grid.nx, ny: o.grid.ny, custom: true } : makeLayout(r);
  const { nx, ny, dxs: dx } = lay;
  const zones = o.grid ? o.relax || null : { offshore: lay.MO, lateral: lay.ML, inland: 0 };
  const brk = Object.assign({ onsetRatio: 0.8, onsetSlope: 0.65, slopeRes: 0.18, keepSlope: 0.15, memory: 2.0, ramp: 3 }, o.breakingParams || {});
  const dispDepth = o.dispDepth ?? [0.25, 0.75];
  const levels = Math.max(Math.ceil(Math.log2(nx)), Math.ceil(Math.log2(ny)), 1);
  const K = CREST_SLOTS;
  const crestCols = lay.custom ? nx : lay.V, crestRows = lay.custom ? ny : lay.V; // the crest scan covers the visible area

  // --- shared uniforms ---------------------------------------------------------
  const U = {
    uNx: { value: nx }, uNy: { value: ny }, uDx: { value: dx }, uG: { value: o.g ?? G }, uBQ: { value: BQ },
    uHDry: { value: o.hDry ?? 1e-4 }, uEps: { value: o.eps ?? 0.05 }, uWl: { value: o.wl ?? 0 }, uTheta: { value: o.theta ?? 1.3 },
    uDt: { value: 0 },
    uRelOff: { value: zones ? zones.offshore || 0 : 0 }, uRelLat: { value: zones ? zones.lateral || 0 : 0 }, uRelInl: { value: zones ? zones.inland || 0 : 0 },
    uUseDisp: { value: o.dispersion === false ? 0 : 1 }, uUseBreak: { value: o.breaking === false ? 0 : 1 }, uUseRelax: { value: 0 },
    uDispDepth: { value: new THREE.Vector2(dispDepth[0], dispDepth[1]) },
    uRamp: { value: brk.ramp }, uBrkMem: { value: brk.memory },
    uBrk: { value: new THREE.Vector4(brk.onsetRatio, brk.onsetSlope, brk.slopeRes, brk.keepSlope) },
  };

  // --- render targets ------------------------------------------------------------
  const allRT = [];
  const mkRT = (w, h) => { const t = new THREE.WebGLRenderTarget(w, h, RT_OPTS); allRT.push(t); return t; };
  const mkMRT = (w, h, n) => { const t = new THREE.WebGLMultipleRenderTargets(w, h, n, RT_OPTS); allRT.push(t); return t; };
  const bathyRT = [mkMRT(nx, ny, 2), mkMRT(nx, ny, 2)]; // (B, d, chiBed, n), (dxs, dys, slope, 0)
  let bcur = 0;
  const coefRT = mkRT(nx, ny), maxXRT = mkRT(nx, ny);
  const S = [mkMRT(nx, ny, 2), mkMRT(nx, ny, 2)]; // (h, P, Q, F), (Pt, Qt, 0, 0)
  let cur = 0;
  const reconRT = mkMRT(nx, ny, 4), fluxRT = mkMRT(nx, ny, 2), rhsRT = mkRT(nx, ny);
  const pcrRT = [mkMRT(nx, ny, 2), mkMRT(nx, ny, 2)];
  const stage1RT = mkRT(nx, ny), targetRT = mkRT(nx, ny), readRT = mkRT(nx, ny);
  const crestRT = mkRT(crestCols, 2 * K);
  // max-speed reduction chain (4x4 blocks down to <= 16 x 16)
  const red = [];
  { let w = nx, h = ny; while (w > 16 || h > 16) { w = Math.ceil(w / 4); h = Math.ceil(h / 4); red.push(mkRT(w, h)); } }
  const redBuf = red.length ? new Float32Array(red[red.length - 1].width * red[red.length - 1].height * 4) : null;

  // --- data textures ---------------------------------------------------------------
  const mkData = (w, h) => { const t = new THREE.DataTexture(new Float32Array(w * h * 4), w, h, THREE.RGBAFormat, THREE.FloatType); t.minFilter = t.magFilter = THREE.NearestFilter; t.needsUpdate = true; return t; };
  // Corner bed (x) and Manning n (y): two channels, so a bed rebuild uploads half of what RGBA would.
  const cornerTex = new THREE.DataTexture(new Float32Array((nx + 1) * (ny + 1) * 2), nx + 1, ny + 1, THREE.RGFormat, THREE.FloatType);
  cornerTex.minFilter = cornerTex.magFilter = THREE.NearestFilter; cornerTex.needsUpdate = true;
  let loadTex = null;
  const compTex = [], kTex = [];
  let auxTex = null;

  // --- materials -----------------------------------------------------------------------
  const materials = [];
  const mat = (fs, uniforms) => { const m = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: SH.VS, fragmentShader: fs, uniforms: { ...U, ...uniforms }, depthTest: false, depthWrite: false }); materials.push(m); return m; };
  const tex = (v = null) => ({ value: v });
  const bedMat = mat(SH.BED_FS, { uCorner: tex(cornerTex) });
  const maxXMat = mat(SH.MAXX_FS, { uState: tex() });
  const coefMat = mat(SH.COEF_FS, { uMaxX: tex(maxXRT.texture), uBathy: tex(), uSlope: tex() });
  const reconMat = mat(SH.RECON_FS, { uState: tex(), uBathy: tex(), uCorner: tex(cornerTex) });
  const fluxMat = mat(SH.FLUX_FS, { uRE: tex(reconRT.texture[0]), uRW: tex(reconRT.texture[1]), uRN: tex(reconRT.texture[2]), uRS: tex(reconRT.texture[3]) });
  const rhsMat = mat(SH.RHS_FS, {
    uState: tex(), uBathy: tex(), uSlope: tex(), uCoef: tex(coefRT.texture), uAux: tex(),
    uRE: tex(reconRT.texture[0]), uRW: tex(reconRT.texture[1]), uRN: tex(reconRT.texture[2]), uRS: tex(reconRT.texture[3]),
    uFx: tex(fluxRT.texture[0]), uFy: tex(fluxRT.texture[1]),
  });
  const pcrMat = mat(SH.PCR_FS, { uCoef: tex(coefRT.texture), uRhs: tex(rhsRT.texture), uX: tex(), uY: tex(), uStride: { value: 1 }, uFirst: { value: 1 } });
  const finishMat = mat(SH.FINISH_FS, { uState: tex(), uRhs: tex(rhsRT.texture), uX: tex(), uY: tex(), uBathy: tex() });
  const finalMat = mat(SH.FINAL_FS, { uState0: tex(), uRhs: tex(rhsRT.texture), uX: tex(), uY: tex(), uBathy: tex(), uTarget: tex(targetRT.texture) });
  const targetMat = mat(SH.TARGET_FS, {
    uC0: tex(), uC1: tex(), uC2: tex(), uC3: tex(), uK0: tex(), uK1: tex(), uK2: tex(), uK3: tex(), uAux: tex(),
    uPh: { value: new Array(NCOMP).fill(0) }, uOm: { value: new Array(NCOMP).fill(0) },
    uSetOmega: { value: 0 }, uSetDepth: { value: 0 }, uT: { value: 0 },
    uML: { value: lay.ML }, uMO: { value: lay.MO }, uDxs: { value: dx }, uDX: { value: DX }, uN: { value: N },
  });
  const planeMat = mat(SH.PLANE_TARGET_FS, { uH: { value: 0 }, uK: { value: 0 }, uPh0: { value: 0 }, uC: { value: 0 }, uT: { value: 0 }, uTRamp: { value: 0 } });
  let planeW = 0;
  const hotMat = mat(SH.HOT_FS, { uTarget: tex(targetRT.texture), uBathy: tex() });
  const loadMat = mat(SH.LOAD_FS, { uSrc: tex() });
  const copyMat = mat(SH.COPY_FS, { uSrc: tex() });
  const rebaseMat = mat(SH.REBASE_FS, { uState: tex(), uAux: tex(), uBathyOld: tex(), uBathyNew: tex() });
  const crestMat = mat(SH.CREST_FS, { uState: tex(), uBathy: tex(), uML: { value: lay.ML }, uMO: { value: lay.MO }, uV: { value: crestRows }, uK: { value: K }, uHWet: { value: 0.15 }, uMinH: { value: 0.08 } });
  const redMat = mat(SH.MAXRED_FS, { uSrc: tex(), uMode: { value: 0 }, uSw: { value: nx }, uSh: { value: ny } });

  // --- full-screen triangle ------------------------------------------------------------
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const quad = new THREE.Mesh(geo, copyMat);
  quad.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(quad);
  const cam = new THREE.Camera();

  let passCount = 0;
  function run(m, target) {
    quad.material = m;
    renderer.setRenderTarget(target);
    renderer.render(scene, cam);
    passCount++;
  }
  // Wrap a sequence of passes: no clears, restore the renderer's target afterwards.
  function begin() { const st = { target: renderer.getRenderTarget(), autoClear: renderer.autoClear }; renderer.autoClear = false; return st; }
  function end(st) { renderer.autoClear = st.autoClear; renderer.setRenderTarget(st.target); }

  const bathyTex = () => bathyRT[bcur].texture[0];
  const slopeTex = () => bathyRT[bcur].texture[1];
  const stateTex = () => S[cur].texture[0];
  const auxState = () => S[cur].texture[1];

  // --- chi and the tridiagonal coefficients ------------------------------------------
  function updateChi() {
    maxXMat.uniforms.uState.value = stateTex();
    run(maxXMat, maxXRT);
    coefMat.uniforms.uBathy.value = bathyTex(); coefMat.uniforms.uSlope.value = slopeTex();
    run(coefMat, coefRT);
  }

  // --- bed -------------------------------------------------------------------------------
  function uploadCorners(z, n) {
    const d = cornerTex.image.data, M = (nx + 1) * (ny + 1);
    for (let c = 0; c < M; c++) { d[c * 2] = z[c]; d[c * 2 + 1] = typeof n === 'number' ? n : n[c]; }
    cornerTex.needsUpdate = true;
  }
  // (Re)run the bed pass and rebase the water surface onto the new bed.
  function rebuildBed() {
    const st = begin();
    run(bedMat, bathyRT[1 - bcur]);
    rebaseMat.uniforms.uState.value = stateTex(); rebaseMat.uniforms.uAux.value = auxState();
    rebaseMat.uniforms.uBathyOld.value = bathyRT[bcur].texture[0]; rebaseMat.uniforms.uBathyNew.value = bathyRT[1 - bcur].texture[0];
    run(rebaseMat, S[1 - cur]);
    cur = 1 - cur; bcur = 1 - bcur;
    updateChi();
    end(st);
  }
  let cornerZ = null, cornerN = null;
  function setBathymetry(beach, wl) {
    if (lay.custom) throw new Error('setBathymetry needs the real layout; use setBed for custom grids');
    cornerZ = cornerBed(beach, wl, r, cornerZ);
    cornerN = cornerManning(beach, r, cornerN);
    uploadCorners(cornerZ, cornerN);
    U.uWl.value = wl;
    rebuildBed();
  }
  function setBed(z, manning = 0.02) {
    if (z.length !== (nx + 1) * (ny + 1)) throw new Error(`bed must hold (nx+1)*(ny+1) corner values, got ${z.length}`);
    uploadCorners(z, manning);
    rebuildBed();
  }
  function setWaterLevel(wl) { U.uWl.value = wl; rebuildBed(); }

  // --- target field --------------------------------------------------------------------
  let targetMode = 'none'; // 'none' | 'waves' | 'plane'
  let omegas = null, phases = null;
  function setTargets(res) {
    if (!compTex.length) for (let q = 0; q < 4; q++) { compTex.push(mkData(N, N)); kTex.push(mkData(N, N)); }
    if (!auxTex) auxTex = mkData(N, N);
    const kx = new Float32Array(N * N), ky = new Float32Array(N * N);
    for (let q = 0; q < 4; q++) {
      const d = compTex[q].image.data, kd = kTex[q].image.data;
      for (let half = 0; half < 2; half++) {
        const m = 2 * q + half, Sm = res.S[m], Am = res.A[m];
        // k = grad S by central differences on the node grid (one-sided at the edges, as waves.js directions()).
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
          const c = j * N + i;
          const gx = i === 0 ? Sm[c + 1] - Sm[c] : i === N - 1 ? Sm[c] - Sm[c - 1] : 0.5 * (Sm[c + 1] - Sm[c - 1]);
          const gy = j === 0 ? Sm[c + N] - Sm[c] : j === N - 1 ? Sm[c] - Sm[c - N] : 0.5 * (Sm[c + N] - Sm[c - N]);
          kx[c] = gx / DX; ky[c] = gy / DX;
        }
        for (let c = 0; c < N * N; c++) {
          d[c * 4 + 2 * half] = Sm[c]; d[c * 4 + 2 * half + 1] = Am[c];
          kd[c * 4 + 2 * half] = kx[c]; kd[c * 4 + 2 * half + 1] = ky[c];
        }
      }
      compTex[q].needsUpdate = true; kTex[q].needsUpdate = true;
    }
    const a = auxTex.image.data;
    for (let c = 0; c < N * N; c++) { a[c * 4] = res.D[c]; a[c * 4 + 1] = res.tau[c]; }
    auxTex.needsUpdate = true;
    const u = targetMat.uniforms;
    u.uC0.value = compTex[0]; u.uC1.value = compTex[1]; u.uC2.value = compTex[2]; u.uC3.value = compTex[3];
    u.uK0.value = kTex[0]; u.uK1.value = kTex[1]; u.uK2.value = kTex[2]; u.uK3.value = kTex[3];
    u.uAux.value = auxTex;
    omegas = res.omegas; phases = res.phases;
    for (let m = 0; m < NCOMP; m++) u.uOm.value[m] = omegas[m];
    u.uSetOmega.value = res.setOmega; u.uSetDepth.value = res.setDepth;
    targetMode = 'waves';
    U.uUseRelax.value = zones ? 1 : 0;
  }
  // Tests: the reference test's plane-wave target etaT = ramp 0.5 H cos(k y - w t), QT = c etaT.
  function setPlaneTarget({ H, k, w, c, tRamp = 0 }) {
    const u = planeMat.uniforms;
    u.uH.value = H; u.uK.value = k; u.uC.value = c; u.uTRamp.value = tRamp; planeW = w;
    targetMode = 'plane';
    U.uUseRelax.value = zones ? 1 : 0;
  }
  function targetAt(t) {
    if (targetMode === 'plane') { planeMat.uniforms.uT.value = t; planeMat.uniforms.uPh0.value = (-planeW * t) % (2 * Math.PI); run(planeMat, targetRT); return; }
    const u = targetMat.uniforms;
    for (let m = 0; m < NCOMP; m++) u.uPh.value[m] = (phases[m] - omegas[m] * t) % (2 * Math.PI);
    u.uT.value = t;
    run(targetMat, targetRT);
  }

  function hotStart(t) {
    if (targetMode === 'none') throw new Error('hotStart needs setTargets first');
    const st = begin();
    targetAt(t);
    hotMat.uniforms.uBathy.value = bathyTex();
    run(hotMat, S[cur]);
    updateChi();
    end(st);
    invalidateCrests();
  }

  // --- one Euler stage: recon, flux, rhs, PCR; returns the PCR target holding the solved systems ---
  function stage(stateIn) {
    reconMat.uniforms.uState.value = stateIn; reconMat.uniforms.uBathy.value = bathyTex();
    run(reconMat, reconRT);
    run(fluxMat, fluxRT);
    const ru = rhsMat.uniforms;
    ru.uState.value = stateIn; ru.uBathy.value = bathyTex(); ru.uSlope.value = slopeTex(); ru.uAux.value = auxState();
    run(rhsMat, rhsRT);
    let a = 0;
    for (let L = 0; L < levels; L++) {
      pcrMat.uniforms.uStride.value = 1 << L;
      pcrMat.uniforms.uFirst.value = L === 0 ? 1 : 0;
      pcrMat.uniforms.uX.value = pcrRT[a].texture[0]; pcrMat.uniforms.uY.value = pcrRT[a].texture[1];
      run(pcrMat, pcrRT[1 - a]);
      a = 1 - a;
    }
    return pcrRT[a];
  }

  // --- one SSP-RK2 step; t is the simulation time at the start of the step ---
  function step(dt, t = 0) {
    const st = begin();
    U.uDt.value = dt;
    if (targetMode !== 'none' && zones) targetAt(t + dt);
    const s0 = stateTex();
    // stage 1
    let p = stage(s0);
    const fu = finishMat.uniforms;
    fu.uState.value = s0; fu.uX.value = p.texture[0]; fu.uY.value = p.texture[1]; fu.uBathy.value = bathyTex();
    run(finishMat, stage1RT);
    // stage 2 + average + relaxation + breaking
    p = stage(stage1RT.texture);
    const fi = finalMat.uniforms;
    fi.uState0.value = s0; fi.uX.value = p.texture[0]; fi.uY.value = p.texture[1]; fi.uBathy.value = bathyTex();
    run(finalMat, S[1 - cur]);
    cur = 1 - cur;
    updateChi();
    end(st);
  }

  // --- readbacks ----------------------------------------------------------------------
  function readTexture(texture, w, h, out) {
    copyMat.uniforms.uSrc.value = texture;
    run(copyMat, readRT);
    renderer.readRenderTargetPixels(readRT, 0, 0, w, h, out);
    return out;
  }
  function readState() {
    const st = begin();
    const out = readTexture(stateTex(), nx, ny, new Float32Array(nx * ny * 4));
    end(st);
    return out;
  }
  function readField(name) {
    const st = begin();
    const t = { state: stateTex(), aux: auxState(), bathy: bathyTex(), slope: slopeTex(), coef: coefRT.texture, target: targetRT.texture, rhs: rhsRT.texture, stage1: stage1RT.texture }[name];
    const out = readTexture(t, nx, ny, new Float32Array(nx * ny * 4));
    end(st);
    return out;
  }
  function loadState(h, P, Q, F) {
    if (!loadTex) loadTex = mkData(nx, ny);
    const d = loadTex.image.data;
    for (let c = 0; c < nx * ny; c++) { d[c * 4] = h[c]; d[c * 4 + 1] = P ? P[c] : 0; d[c * 4 + 2] = Q ? Q[c] : 0; d[c * 4 + 3] = F ? F[c] : 0; }
    loadTex.needsUpdate = true;
    const st = begin();
    loadMat.uniforms.uSrc.value = loadTex;
    run(loadMat, S[cur]);
    updateChi();
    end(st);
    invalidateCrests();
  }
  const crestBuf = new Float32Array(crestRT.width * crestRT.height * 4);
  const crestOut = new Float32Array(crestCols * K * 8).fill(-1); // J = -1: no crest
  // Asynchronous readback (default): the scan is copied into a pixel buffer
  // object and read on the next call, after a fence, so the CPU never waits
  // for the solver passes queued ahead of it. The breakers then run one call
  // behind the solver. `asyncCrests: false` reads synchronously (the tests).
  const gl = renderer.getContext();
  const isGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  const useAsync = (o.asyncCrests ?? true) && isGL2;
  const NRING = 3; // scans in flight at most; the CPU waits only when all are
  const crestRTs = useAsync ? [crestRT, mkRT(crestCols, 2 * K), mkRT(crestCols, 2 * K)] : [crestRT];
  const crestPBO = useAsync ? [gl.createBuffer(), gl.createBuffer(), gl.createBuffer()] : null;
  const crestFence = [null, null, null];
  const crestQueue = []; // slots with a scan in flight, oldest first
  let crestHave = false, crestStale = 0; // scans in flight that predate a state reset: read and discarded
  if (useAsync) {
    for (const b of crestPBO) { gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b); gl.bufferData(gl.PIXEL_PACK_BUFFER, crestBuf.byteLength, gl.STREAM_READ); }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  }
  // A new state (hot start, load) makes the scan in flight stale: drop it, so
  // the next call reports no crests instead of the old field's.
  function invalidateCrests() {
    crestStale = crestQueue.length; // consumed in order and discarded (deleting the fences would leave the buffers unread)
    crestHave = false;
    crestOut.fill(-1);
  }
  // Asynchronous ring: the oldest scan in flight is read once its fence has
  // signalled (a GPU running a frame behind costs a frame of latency, not a
  // stall), then a new scan is queued into a free slot. Only when both slots
  // are in flight is the oldest waited for (a fence cannot be seen signalled
  // within the task that created it, so tight loops that step many times per
  // task, like the tests, take this path on every call).
  function readCrests() {
    const st = begin();
    crestMat.uniforms.uState.value = stateTex(); crestMat.uniforms.uBathy.value = bathyTex();
    if (useAsync) {
      crestHave = false;
      if (crestQueue.length && (crestQueue.length >= NRING || gl.getSyncParameter(crestFence[crestQueue[0]], gl.SYNC_STATUS) === gl.SIGNALED)) {
        const slot = crestQueue.shift();
        gl.deleteSync(crestFence[slot]); crestFence[slot] = null;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, crestPBO[slot]);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, crestBuf);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        if (crestStale > 0) crestStale--; else crestHave = true;
      }
      if (crestQueue.length < NRING) {
        let slot = 0; while (crestFence[slot]) slot++;
        const rt = crestRTs[slot];
        run(crestMat, rt);
        renderer.state.bindFramebuffer(gl.FRAMEBUFFER, renderer.properties.get(rt).__webglFramebuffer);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, crestPBO[slot]);
        gl.readPixels(0, 0, rt.width, rt.height, gl.RGBA, gl.FLOAT, 0);
        crestFence[slot] = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        crestQueue.push(slot);
      }
    } else {
      run(crestMat, crestRT);
      renderer.readRenderTargetPixels(crestRT, 0, 0, crestRT.width, crestRT.height, crestBuf);
      crestHave = true;
    }
    end(st);
    if (!crestHave) return crestOut; // nothing new yet: the last records (or none)
    const V = crestCols;
    for (let col = 0; col < V; col++) for (let s = 0; s < K; s++) {
      const o = (col * K + s) * 8, a = (s * V + col) * 4, b = ((K + s) * V + col) * 4;
      crestOut[o] = crestBuf[a]; crestOut[o + 1] = crestBuf[a + 1]; crestOut[o + 2] = crestBuf[a + 2]; crestOut[o + 3] = crestBuf[a + 3];
      crestOut[o + 4] = crestBuf[b]; crestOut[o + 5] = crestBuf[b + 1]; crestOut[o + 6] = crestBuf[b + 2]; crestOut[o + 7] = crestBuf[b + 3];
    }
    return crestOut;
  }
  // The reduction passes: max speed per block down to a <= 16 x 16 target (null on tiny grids).
  function reduceSpeed() {
    let src = stateTex(), w = nx, h = ny, last = null;
    for (let k = 0; k < red.length; k++) {
      const u = redMat.uniforms;
      u.uSrc.value = src; u.uMode.value = k === 0 ? 0 : 1; u.uSw.value = w; u.uSh.value = h;
      run(redMat, red[k]);
      src = red[k].texture; w = red[k].width; h = red[k].height; last = red[k];
    }
    return last;
  }
  function maxOfRed(w, h) { let m = 0; for (let c = 0; c < w * h; c++) if (redBuf[c * 4] > m) m = redBuf[c * 4]; return m; }
  function maxSpeed() {
    const st = begin();
    const last = reduceSpeed();
    let m = 0;
    if (last) {
      renderer.readRenderTargetPixels(last, 0, 0, last.width, last.height, redBuf);
      m = maxOfRed(last.width, last.height);
    } else {
      const buf = readTexture(stateTex(), nx, ny, new Float32Array(nx * ny * 4));
      for (let c = 0; c < nx * ny; c++) { const hc = buf[c * 4]; if (hc > U.uHDry.value) { const s = Math.hypot(buf[c * 4 + 1], buf[c * 4 + 2]) / hc + Math.sqrt(G * hc); if (s > m) m = s; } }
    }
    end(st);
    return m;
  }

  // Asynchronous speed check: maxSpeedIssue() queues the reduction and its
  // readback behind a fence and returns true; maxSpeedPoll() returns the
  // result once the GPU has finished it (-1 until then, which is at least the
  // next frame), so the check never drains the queue inside a frame. Grids too
  // small for a reduction (and WebGL1) report false: use maxSpeed().
  const speedPBO = isGL2 && red.length ? gl.createBuffer() : null;
  let speedFence = null;
  if (speedPBO) { gl.bindBuffer(gl.PIXEL_PACK_BUFFER, speedPBO); gl.bufferData(gl.PIXEL_PACK_BUFFER, redBuf.byteLength, gl.STREAM_READ); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null); }
  function maxSpeedIssue() {
    if (!speedPBO) return false;
    const st = begin();
    const last = reduceSpeed();
    if (speedFence) { gl.deleteSync(speedFence); speedFence = null; }
    renderer.state.bindFramebuffer(gl.FRAMEBUFFER, renderer.properties.get(last).__webglFramebuffer);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, speedPBO);
    gl.readPixels(0, 0, last.width, last.height, gl.RGBA, gl.FLOAT, 0);
    speedFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    end(st);
    return true;
  }
  function maxSpeedPoll() {
    if (!speedFence) return -1;
    if (gl.getSyncParameter(speedFence, gl.SYNC_STATUS) !== gl.SIGNALED) return -1;
    gl.deleteSync(speedFence); speedFence = null;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, speedPBO);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, redBuf);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    const last = red[red.length - 1];
    return maxOfRed(last.width, last.height);
  }

  function dispose() {
    if (useAsync) {
      for (const b of crestPBO) gl.deleteBuffer(b);
      for (let i = 0; i < NRING; i++) if (crestFence[i]) { gl.deleteSync(crestFence[i]); crestFence[i] = null; }
      for (let i = 1; i < NRING; i++) crestRTs[i].dispose();
    }
    if (speedPBO) { gl.deleteBuffer(speedPBO); if (speedFence) { gl.deleteSync(speedFence); speedFence = null; } }
    for (const t of allRT) t.dispose();
    for (const m of materials) m.dispose();
    geo.dispose();
    cornerTex.dispose();
    if (loadTex) loadTex.dispose();
    for (const t of compTex) t.dispose();
    for (const t of kTex) t.dispose();
    if (auxTex) auxTex.dispose();
  }

  // Texture memory (MB) of everything allocated on the GPU.
  function textureMemoryMB() {
    let bytes = 0;
    for (const t of allRT) bytes += t.width * t.height * 16 * (Array.isArray(t.texture) ? t.texture.length : 1);
    bytes += (nx + 1) * (ny + 1) * 16;
    if (loadTex) bytes += nx * ny * 16;
    bytes += (compTex.length + kTex.length + (auxTex ? 1 : 0)) * N * N * 16;
    return bytes / 1048576;
  }

  // Initial bed: flat at -10 m for custom grids until setBed; the real layout waits for setBathymetry.
  if (lay.custom && o.grid.bed) setBed(o.grid.bed, o.grid.manning ?? 0.02);
  else if (lay.custom) { const z = new Float32Array((nx + 1) * (ny + 1)).fill(-10); setBed(z, o.grid.manning ?? 0.02); }
  else { const st = begin(); updateChi(); end(st); }

  return {
    layout: lay, nx, ny, dx, levels, crestCols, crestRows, crestSlots: K,
    passesPerStep: 1 + 2 * (3 + levels + 1) + 2,
    get passCount() { return passCount; },
    textures: { get state() { return stateTex(); }, get bathy() { return bathyTex(); }, get aux() { return auxState(); }, get coef() { return coefRT.texture; }, get target() { return targetRT.texture; }, get slope() { return slopeTex(); } },
    get wl() { return U.uWl.value; },
    setBathymetry, setBed, setWaterLevel, setTargets, setPlaneTarget, hotStart, step,
    readCrests, maxSpeed, maxSpeedIssue, maxSpeedPoll, readState, readField, loadState, textureMemoryMB, dispose,
  };
}
