// three.js geometry and material for the lip curtains and the splash-up
// sheets. One BufferGeometry holds a pool of vertex loops (one loop = the
// closed cross-section of a ribbon at one crest sample: NPROF points along the
// upper face, a few around the bulbous leading edge, NPROF back along the
// underside). Each breaker sample owns one loop for its curtain and, while
// the splash-up lasts, one for the splash sheet. Consecutive samples of a
// ribbon are stitched by a dynamic index that is rewritten every frame, so
// ribbons of any length share the pool and nothing is drawn for free slots.
//
// Attributes: position, normal, uv (u = metres along the crest, v = loop
// parameter 0..1: 0 root upper, 0.5 leading edge, 1 root lower), aux =
// (foam 0..1, alpha, thickness m, profile q 0..1).
import * as THREE from 'three';
import { NPROF } from './lip-model.js';

export const NCAP = 4; // points around the leading edge
export const NLOOP = 2 * NPROF + NCAP; // 28 vertices per loop

const DEFAULT_SUN = new THREE.Vector3(-0.45, 0.42, 0.79).normalize();

const GLSL_COMMON = /* glsl */ `
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float fbm2(vec2 p) {
  return 0.5 * vnoise(p) + 0.25 * vnoise(p * 2.03 + 7.1) + 0.125 * vnoise(p * 4.1 + 3.3) + 0.0625 * vnoise(p * 8.3 + 1.7);
}
vec3 sky(vec3 d, vec3 sun) {
  float y = max(d.y, 0.0);
  vec3 c = mix(vec3(0.80, 0.86, 0.90), vec3(0.32, 0.55, 0.82), pow(y, 0.55));
  float s = max(dot(d, sun), 0.0);
  c += vec3(1.0, 0.86, 0.62) * (pow(s, 600.0) * 30.0 + pow(s, 12.0) * 0.25);
  return c;
}
`;

export function createCurtainMaterial(opts = {}) {
  const sun = (opts.sun ? new THREE.Vector3().fromArray(opts.sun) : DEFAULT_SUN.clone()).normalize();
  const mat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      uSun: { value: sun },
      uTime: { value: 0 },
      uWl: { value: 0 },
      uTint: { value: new THREE.Vector3(0.16, 0.62, 0.56) }, // thin-sheet colour
      uDeep: { value: new THREE.Vector3(0.02, 0.17, 0.24) }, // thick-sheet colour
    },
    vertexShader: /* glsl */ `
      in vec4 aux;
      out vec3 vP; out vec3 vN; out vec2 vUv; out vec4 vAux;
      void main() {
        vP = (modelMatrix * vec4(position, 1.0)).xyz;
        vN = normalize(mat3(modelMatrix) * normal);
        vUv = uv; vAux = aux;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: GLSL_COMMON + /* glsl */ `
      uniform vec3 uSun, uTint, uDeep;
      uniform float uTime, uWl;
      in vec3 vP; in vec3 vN; in vec2 vUv; in vec4 vAux;
      out vec4 o;
      void main() {
        if (vAux.y <= 0.003) discard;
        vec3 n = normalize(vN);
        if (!gl_FrontFacing) n = -n;
        vec3 v = normalize(cameraPosition - vP);
        float th = vAux.z;
        // Ribbing: the sheet stretches as it falls, so its surface is corrugated along the flow (fine alongshore,
        // long down the profile). Perturb the normal with the noise gradient so the glint breaks up.
        vec2 rp = vec2(vUv.x * 1.7, vUv.y * 0.9 - uTime * 0.5);
        float r0 = fbm2(rp), rx = fbm2(rp + vec2(0.07, 0.0)), ry = fbm2(rp + vec2(0.0, 0.07));
        vec3 tAlong = normalize(cross(n, vec3(0.0, 1.0, 0.0)) + vec3(1e-4));
        vec3 tDown = normalize(cross(tAlong, n));
        n = normalize(n + (rx - r0) * 0.6 * tAlong + (ry - r0) * 0.3 * tDown);
        // Body: a thin sheet glows turquoise (light scattered inside it), a thick one is deep teal.
        vec3 body = mix(uTint, uDeep, smoothstep(0.04, 0.45, th));
        float dif = max(dot(n, uSun), 0.0);
        // The underside of the sheet looks into the tube: no sky light there.
        float under = smoothstep(0.5, 0.62, vUv.y) * (1.0 - smoothstep(0.93, 1.0, vUv.y));
        float occl = 1.0 - 0.7 * under;
        vec3 c = body * (0.42 + 0.78 * dif) * occl;
        // Light through the sheet toward the viewer when the sun is behind it.
        float back = max(-dot(v, uSun), 0.0);
        float trans = pow(back, 1.6) * exp(-th * 3.5);
        c += vec3(0.18, 0.80, 0.62) * trans * 1.3;
        // Upward-facing parts near the water pick up the water body colour below.
        c += vec3(0.0, 0.05, 0.05) * max(-n.y, 0.0);
        // Sky reflection and sun glint.
        float fres = 0.06 + 0.94 * pow(1.0 - max(dot(n, v), 0.0), 4.0);
        c = mix(c, sky(reflect(-v, n), uSun) * occl, fres * 0.85);
        vec3 hv = normalize(uSun + v);
        c += vec3(1.0, 0.92, 0.75) * pow(max(dot(n, hv), 0.0), 90.0) * 0.9;
        // Foam and aeration: streaks running down the profile, lumpy along the crest.
        // Aeration streaks run with the flow (down the profile): fine alongshore, long along v.
        float tex = 0.6 * fbm2(vec2(vUv.x * 2.3 + 0.7 * vUv.y, vUv.y * 1.4 - uTime * 0.8)) + 0.4 * fbm2(vec2(vUv.x * 0.61 + 5.0, vUv.y * 3.7 - uTime * 0.35 + vUv.x * 0.1));
        float splash = step(1.5, vAux.x); // splash-up sheets carry foam = 2
        float foam = smoothstep(0.3, 0.85, min(vAux.x, 1.0) - 0.4 * (tex - 0.5));
        float lump = fbm2(vec2(vUv.x * 4.1, vUv.y * 5.3 - uTime * 0.6));
        vec3 fc = vec3(0.94, 0.96, 0.97) * (0.78 + 0.22 * dif) * (0.82 + 0.3 * lump);
        c = mix(c, fc, foam);
        // A thin sheet is translucent (the crest behind shows through); the thick leading edge and foam are not.
        float a = mix(0.6 + 0.3 * smoothstep(0.06, 0.3, th), 1.0, foam) * vAux.y;
        if (splash > 0.5) {
          // Splash-up: a burst of spray and aerated water, not a sheet. Misty,
          // ragged, see-through, thinning toward its top (vAux.y already fades there).
          float rag = fbm2(vec2(vUv.x * 2.7, vUv.y * 2.1 - uTime * 1.4)) * 0.7 + 0.3 * fbm2(vec2(vUv.x * 7.0 + 3.0, vUv.y * 5.0 - uTime * 2.2));
          c = mix(c, vec3(0.90, 0.94, 0.96) * (0.8 + 0.2 * dif), 0.6);
          a = vAux.y * smoothstep(0.2, 0.75, rag + 0.22) * 0.8;
        }
        o = vec4(c, a);
      }`,
    transparent: true,
    depthWrite: false, // depth comes from the pre-pass (see createLipMesh)
    depthFunc: THREE.LessEqualDepth,
    side: THREE.DoubleSide,
  });
  return mat;
}

export function createLipMesh(opts = {}) {
  const SLOTS = opts.meshSlots ?? 4096; // vertex loops in the pool
  const NV = SLOTS * NLOOP;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(NV * 3), nrm = new Float32Array(NV * 3), uv = new Float32Array(NV * 2), aux = new Float32Array(NV * 4);
  const posA = new THREE.BufferAttribute(pos, 3), nrmA = new THREE.BufferAttribute(nrm, 3), uvA = new THREE.BufferAttribute(uv, 2), auxA = new THREE.BufferAttribute(aux, 4);
  for (const a of [posA, nrmA, uvA, auxA]) a.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', posA);
  geo.setAttribute('normal', nrmA);
  geo.setAttribute('uv', uvA);
  geo.setAttribute('aux', auxA);
  const idx = new Uint32Array(SLOTS * NLOOP * 6);
  const idxA = new THREE.BufferAttribute(idx, 1);
  idxA.setUsage(THREE.DynamicDrawUsage);
  geo.setIndex(idxA);
  geo.setDrawRange(0, 0);
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2000);
  const material = createCurtainMaterial(opts);
  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  mesh.name = 'lip-curtains';
  mesh.renderOrder = opts.renderOrder ?? 4;
  // Depth pre-pass: the sheet is a closed, slightly translucent shell, and
  // blending its upper and lower faces in index order leaves slivers of the
  // underside showing along every cross-section. A colour-less copy writes
  // depth in the opaque pass; the shaded pass (depthWrite off, LessEqual)
  // then only colours the nearest layer. Both share the geometry.
  const depthMat = material.clone(); // same shaders => bit-identical depth
  depthMat.uniforms = material.uniforms;
  depthMat.colorWrite = false;
  depthMat.transparent = false;
  depthMat.depthWrite = true;
  depthMat.depthFunc = THREE.LessEqualDepth;
  const depthMesh = new THREE.Mesh(geo, depthMat);
  depthMesh.frustumCulled = false;
  depthMesh.name = 'lip-curtains-depth';
  depthMesh.renderOrder = (opts.renderOrder ?? 4) - 0.5;

  const free = [];
  for (let s = SLOTS - 1; s >= 0; s--) free.push(s);
  let watermark = 0; // highest slot index + 1 ever written (upload range)
  let nIdx = 0;
  const owners = new Map(); // breaker id -> breaker (for freeing slots of vanished breakers)

  // scratch for a loop
  const la = new Float32Array(NLOOP), ly = new Float32Array(NLOOP), lna = new Float32Array(NLOOP), lny = new Float32Array(NLOOP), lf = new Float32Array(NLOOP), lq = new Float32Array(NLOOP), lth = new Float32Array(NLOOP), lal = new Float32Array(NLOOP);
  const ta = new Float32Array(NPROF), ty = new Float32Array(NPROF);

  // Write one loop from a profile (a[k], y[k], th[k], fu[k], fl[k]) at world (x, zRoot) with horizontal direction (dx, dz).
  function writeLoop(slot, prof, x, zRoot, dx, dz, uAlong, alpha) {
    const a = prof.a, y = prof.y, th = prof.th;
    // in-plane tangents
    for (let k = 0; k < NPROF; k++) {
      const k0 = k > 0 ? k - 1 : 0, k1 = k < NPROF - 1 ? k + 1 : NPROF - 1;
      let tx = a[k1] - a[k0], tyy = y[k1] - y[k0];
      const l = Math.hypot(tx, tyy) || 1;
      ta[k] = tx / l; ty[k] = tyy / l;
    }
    // upper face: root -> tip, offset +N (N = (-T.y, T.x))
    for (let k = 0; k < NPROF; k++) {
      const nxA = -ty[k], nyA = ta[k], h = 0.5 * th[k];
      la[k] = a[k] + nxA * h; ly[k] = y[k] + nyA * h; lna[k] = nxA; lny[k] = nyA; lf[k] = prof.fu[k]; lq[k] = k / (NPROF - 1); lth[k] = th[k]; lal[k] = prof.al[k];
    }
    // leading-edge cap: from +N through +T to -N
    const kt = NPROF - 1, r = 0.5 * th[kt];
    for (let m = 0; m < NCAP; m++) {
      const ang = Math.PI / 2 - (Math.PI * (m + 1)) / (NCAP + 1);
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const nxA = -ty[kt] * sa + ta[kt] * ca, nyA = ta[kt] * sa + ty[kt] * ca;
      const kk = NPROF + m;
      la[kk] = a[kt] + nxA * r; ly[kk] = y[kt] + nyA * r; lna[kk] = nxA; lny[kk] = nyA; lf[kk] = Math.max(prof.fu[kt], prof.fl[kt]); lq[kk] = 1; lth[kk] = th[kt]; lal[kk] = prof.al[kt];
    }
    // underside: tip -> root, offset -N
    for (let k = NPROF - 1; k >= 0; k--) {
      const nxA = ty[k], nyA = -ta[k], h = 0.5 * th[k];
      const kk = NPROF + NCAP + (NPROF - 1 - k);
      la[kk] = a[k] + nxA * h; ly[kk] = y[k] + nyA * h; lna[kk] = nxA; lny[kk] = nyA; lf[kk] = prof.fl[k]; lq[kk] = k / (NPROF - 1); lth[kk] = th[k]; lal[kk] = prof.al[k];
    }
    const base = slot * NLOOP;
    for (let kk = 0; kk < NLOOP; kk++) {
      const vi = base + kk;
      pos[vi * 3] = x + dx * la[kk]; pos[vi * 3 + 1] = ly[kk]; pos[vi * 3 + 2] = zRoot + dz * la[kk];
      nrm[vi * 3] = dx * lna[kk]; nrm[vi * 3 + 1] = lny[kk]; nrm[vi * 3 + 2] = dz * lna[kk];
      uv[vi * 2] = uAlong; uv[vi * 2 + 1] = kk / NLOOP;
      aux[vi * 4] = Math.min(lf[kk], 2); aux[vi * 4 + 1] = alpha * lal[kk]; aux[vi * 4 + 2] = lth[kk]; aux[vi * 4 + 3] = lq[kk];
    }
    if (slot + 1 > watermark) watermark = slot + 1;
  }

  function stitch(sa, sb) {
    const A = sa * NLOOP, B = sb * NLOOP;
    for (let kk = 0; kk < NLOOP; kk++) {
      const k2 = (kk + 1) % NLOOP;
      idx[nIdx++] = A + kk; idx[nIdx++] = B + k2; idx[nIdx++] = B + kk;
      idx[nIdx++] = A + kk; idx[nIdx++] = A + k2; idx[nIdx++] = B + k2;
    }
  }

  function freeBreaker(b) {
    for (let i = 0; i < b.n; i++) {
      if (b.slotLip[i] >= 0) { free.push(b.slotLip[i]); b.slotLip[i] = -1; }
      if (b.slotSplash[i] >= 0) { free.push(b.slotSplash[i]); b.slotSplash[i] = -1; }
    }
    owners.delete(b.id);
  }

  // model: the lip model (for lipProfile / splashProfile); breakers: its live list.
  const prof = { a: new Float32Array(NPROF), y: new Float32Array(NPROF), th: new Float32Array(NPROF), fu: new Float32Array(NPROF), fl: new Float32Array(NPROF), al: new Float32Array(NPROF).fill(1), alpha: 1 };
  function update(model, breakers, { wl = 0, t = 0 } = {}) {
    // free the slots of breakers that vanished since the last frame (the model also calls release hooks; this is the safety net)
    for (const [id, b] of owners) if (b.id !== id || b.done || !breakers.includes(b)) freeBreaker(b);
    nIdx = 0;
    // Slots: every live sample gets its curtain loop before any splash-up sheet takes one, so a pool that
    // runs short (many wide lips at once) drops splash sheets, never pieces of a curtain.
    for (const b of breakers) {
      owners.set(b.id, b);
      for (let i = 0; i < b.n; i++) if (b.slotLip[i] < 0 && free.length) b.slotLip[i] = free.pop();
    }
    for (const b of breakers) {
      let prevLip = -1, prevSplash = -1;
      for (let i = 0; i < b.n; i++) {
        // curtain
        const sl = b.slotLip[i];
        if (sl >= 0) {
          model.lipProfile(b, i, prof);
          writeLoop(sl, prof, b.x[i], b.zRoot[i], b.dirX[i], b.dirZ[i], (b.c0 + i) * model.lay.dxs, prof.alpha);
          if (prevLip >= 0) stitch(prevLip, sl);
        }
        prevLip = sl;
        // splash-up
        const has = model.splashProfile(b, i, prof);
        if (has) {
          if (b.slotSplash[i] < 0 && free.length) b.slotSplash[i] = free.pop();
          const ss = b.slotSplash[i];
          if (ss >= 0) {
            writeLoop(ss, prof, b.x[i], b.zRoot[i], b.dirX[i], b.dirZ[i], (b.c0 + i) * model.lay.dxs + 37.0, prof.alpha);
            if (prevSplash >= 0) stitch(prevSplash, ss);
          }
          prevSplash = ss;
        } else {
          if (b.slotSplash[i] >= 0) { free.push(b.slotSplash[i]); b.slotSplash[i] = -1; }
          prevSplash = -1;
        }
      }
    }
    if (!breakers.length && free.length === SLOTS) watermark = 0;
    geo.setDrawRange(0, nIdx);
    const nv = watermark * NLOOP;
    for (const [att, sz] of [[posA, 3], [nrmA, 3], [uvA, 2], [auxA, 4]]) {
      att.clearUpdateRanges();
      att.addUpdateRange(0, nv * sz);
      att.needsUpdate = true;
    }
    idxA.clearUpdateRanges();
    idxA.addUpdateRange(0, nIdx);
    idxA.needsUpdate = true;
    material.uniforms.uTime.value = t;
    material.uniforms.uWl.value = wl;
  }

  function dispose() { geo.dispose(); material.dispose(); depthMat.dispose(); }

  return {
    mesh, depthMesh, geometry: geo, material, depthMaterial: depthMat, update, freeBreaker, dispose,
    get indexCount() { return nIdx; },
    get usedSlots() { return SLOTS - free.length; },
    get watermark() { return watermark; },
    SLOTS,
  };
}
