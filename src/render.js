// Three.js scene: the beach as a diorama block, the water surface driven by
// a GPU wave pass, and a foam pass that remembers where waves broke.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { N, EXT } from './terrain.js';
import { NCOMP } from './waves.js';

// Afternoon sun behind the beach, so wave faces are lit as seen from the shore.
const SUN = new THREE.Vector3(-0.45, 0.42, 0.79).normalize();

const COMMON = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform float uExt;
uniform float uWl;
const int NN = ${N};
const vec3 SUN = vec3(${SUN.x.toFixed(4)}, ${SUN.y.toFixed(4)}, ${SUN.z.toFixed(4)});

vec4 bl(sampler2D t, vec2 uv) {
  vec2 g = clamp(uv, 0.0, 1.0) * float(NN - 1);
  ivec2 i0 = clamp(ivec2(floor(g)), ivec2(0), ivec2(NN - 2));
  vec2 f = g - vec2(i0);
  vec4 a = texelFetch(t, i0, 0), b = texelFetch(t, i0 + ivec2(1, 0), 0);
  vec4 c = texelFetch(t, i0 + ivec2(0, 1), 0), d = texelFetch(t, i0 + ivec2(1, 1), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
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
vec3 bedColor(float z, float rock, vec2 p) {
  float n = fbm2(p * 0.08);
  vec3 dry = vec3(0.89, 0.81, 0.64) * (0.92 + 0.16 * n);
  vec3 wet = vec3(0.66, 0.58, 0.44);
  vec3 under = vec3(0.80, 0.73, 0.56) * (0.9 + 0.2 * n);
  vec3 grass = vec3(0.48, 0.53, 0.33) * (0.85 + 0.3 * n);
  vec3 c = z > uWl + 0.9 ? dry : z > uWl - 0.1 ? mix(wet, dry, smoothstep(uWl + 0.1, uWl + 0.9, z)) : under;
  c = mix(c, grass, smoothstep(4.6, 5.4, z + n));
  vec3 rk = vec3(0.40, 0.38, 0.36) * (0.7 + 0.6 * fbm2(p * 0.35));
  rk = mix(rk, vec3(0.23, 0.27, 0.22), smoothstep(uWl + 0.3, uWl - 0.6, z) * 0.6);
  return mix(c, rk, smoothstep(0.35, 0.65, rock));
}
vec3 sky(vec3 d) {
  float y = max(d.y, 0.0);
  vec3 c = mix(vec3(0.80, 0.86, 0.90), vec3(0.32, 0.55, 0.82), pow(y, 0.55));
  float s = max(dot(d, SUN), 0.0);
  c += vec3(1.0, 0.86, 0.62) * (pow(s, 600.0) * 30.0 + pow(s, 12.0) * 0.25);
  return c;
}
`;

export function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const coarse = matchMedia('(pointer: coarse)').matches;
  const M = coarse ? 320 : 512; // water grid resolution

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 9000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 4;
  controls.maxDistance = 1800;

  // --- data textures (written by the solver) -------------------------------
  const mk = () => {
    const t = new THREE.DataTexture(new Float32Array(N * N * 4), N, N, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  };
  const heightTex = mk();
  const compTex = [mk(), mk(), mk(), mk()];
  const bedTex = mk();
  const dirTex = mk();
  const grpTex = mk();

  const rtOpts = { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
  const waveRT = new THREE.WebGLRenderTarget(M, M, rtOpts);
  const foamOpts = { ...rtOpts, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
  let foamA = new THREE.WebGLRenderTarget(M, M, foamOpts);
  let foamB = new THREE.WebGLRenderTarget(M, M, foamOpts);

  const shared = {
    uExt: { value: EXT },
    uWl: { value: 0 },
    uTime: { value: 0 },
    uHeight: { value: heightTex },
    uBed: { value: bedTex },
    uDir: { value: dirTex },
    uGrp: { value: grpTex },
    uContours: { value: 0 },
    uBrush: { value: new THREE.Vector4(0, 0, 0, 0) }, // x, y, radius, visible
    uFoam: { value: foamA.texture },
    uEngine: { value: 0 },
    // Real-fluid engine: solver state (h, P, Q, F), bathymetry (B, d, chiBed, n)
    // and the layout (lateral margin, offshore margin, cell size) in solver cells.
    uState: { value: null },
    uBathy: { value: null },
    uLay: { value: new THREE.Vector4(0, 0, 2.5, 0) },
  };

  // Sampling the engine's state texture at a visible grid position (uv in
  // 0..1 across the terrain). Returns (eta above wl, u, v, F); the surface is
  // interpolated between wet cells only, so the waterline stays crisp.
  const ENGINE_GLSL = /* glsl */ `
  uniform sampler2D uState, uBathy, uHeight;
  uniform vec4 uLay;
  uniform float uEngine;
  vec4 engineSample(vec2 uv, out float wetness) {
    vec2 tc = uv * uExt / uLay.z + uLay.xy - 0.5;
    ivec2 sz = textureSize(uState, 0);
    ivec2 i0 = ivec2(floor(tc));
    vec2 f = tc - vec2(i0);
    float W = 0.0, ww = 0.0, F = 0.0, u = 0.0, v = 0.0;
    for (int k = 0; k < 4; k++) {
      ivec2 ij = clamp(i0 + ivec2(k & 1, k >> 1), ivec2(0), sz - 1);
      float w = ((k & 1) == 0 ? 1.0 - f.x : f.x) * ((k >> 1) == 0 ? 1.0 - f.y : f.y);
      vec4 st = texelFetch(uState, ij, 0);
      vec4 b = texelFetch(uBathy, ij, 0);
      float wet = smoothstep(0.005, 0.04, st.x);
      W += w * wet * (b.x + st.x); ww += w * wet;
      F += w * st.w;
      float hh = max(st.x, 0.05);
      u += w * wet * st.y / hh; v += w * wet * st.z / hh;
    }
    wetness = ww;
    if (ww < 0.35) return vec4(bl(uHeight, uv).x - uWl - 0.08, 0.0, 0.0, 0.0);
    return vec4(W / ww - uWl, u / ww, v / ww, F);
  }
  `;

  // --- wave pass -----------------------------------------------------------
  const quadGeo = new THREE.BufferGeometry();
  quadGeo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const quadCam = new THREE.Camera();
  const passVS = /* glsl */ `in vec3 position; void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }`;

  const waveMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      ...shared,
      uC0: { value: compTex[0] }, uC1: { value: compTex[1] }, uC2: { value: compTex[2] }, uC3: { value: compTex[3] },
      uPh: { value: new Array(NCOMP).fill(0) },
      uM: { value: M },
      uActive: { value: 0 },
      uSetOmega: { value: 0 },
      uSetDepth: { value: 0 },
    },
    vertexShader: passVS,
    fragmentShader: COMMON + /* glsl */ `
      uniform sampler2D uC0, uC1, uC2, uC3, uBed, uDir, uGrp;
      uniform float uPh[${NCOMP}];
      uniform float uM, uActive, uTime, uSetOmega, uSetDepth;
      out vec4 o;
      void acc(vec4 c, float p0, float p1, inout vec2 z) {
        float t0 = c.x + p0, t1 = c.z + p1;
        z += c.y * vec2(cos(t0), sin(t0)) + c.w * vec2(cos(t1), sin(t1));
      }
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5) / (uM - 1.0);
        vec4 bed = bl(uBed, uv);
        float h = bed.x;
        if (h < 0.03 || uActive < 0.5) { o = vec4(0.0); return; }
        vec2 zc = vec2(0.0);
        acc(bl(uC0, uv), uPh[0], uPh[1], zc);
        acc(bl(uC1, uv), uPh[2], uPh[3], zc);
        acc(bl(uC2, uv), uPh[4], uPh[5], zc);
        acc(bl(uC3, uv), uPh[6], uPh[7], zc);
        vec4 grp = bl(uGrp, uv);
        // Sets: a slow swell of the wave height travelling at the group velocity.
        float setMod = 1.0 + uSetDepth * cos(uSetOmega * (uTime - grp.x));
        float env = length(zc) * bed.y * setMod * 0.96;
        float L = max(grp.y, 4.0);
        float phi = atan(zc.y, zc.x);
        float gam = bed.z, plunge = bed.w;
        vec4 dir = bl(uDir, uv);
        float surge = dir.w;
        float r = 2.0 * env / (gam * h);
        // Inside the surf zone (where the solver took energy out) a wave that
        // has already broken keeps going as a bore at a lower height ratio.
        float sz = smoothstep(0.03, 0.2, 1.0 - bed.y);
        float rb = r / mix(1.0, 0.6, sz);
        float ev = min(env, 0.52 * gam * h);
        float Hc = 2.0 * ev;
        float s = clamp(rb, 0.0, 1.0);
        float br = smoothstep(0.92, 1.15, rb) * (1.0 - 0.7 * surge);
        // Wave profile: t = 0 is the crest, t in (0, ff) the front face ahead
        // of it. As the wave shoals the crest sharpens and the front face
        // takes up less of the wavelength, until near breaking it is only
        // about 1.6 wave heights wide (a wall); a bore has an even steeper front.
        float p = mix(1.0 + 2.4 * s * s, 1.7, br);
        float ffSteep = clamp(max(1.3 * Hc, 2.5) / L, 0.015, 0.5);
        float ffBore = clamp(max(1.1 * Hc, 2.0) / L, 0.012, 0.5);
        float ff = mix(mix(0.5, ffSteep, smoothstep(0.25, 0.95, s)), ffBore, br);
        float t = fract(phi / 6.28318);
        float th = t < ff ? 3.14159 * t / ff : 3.14159 * (1.0 + (t - ff) / (1.0 - ff));
        float q = 0.5 + 0.5 * cos(th);
        float m = inversesqrt(3.14159 * (p + 0.25));
        float eta = Hc * (pow(q, p) - m);
        // Whitewater is lumpy and aerated.
        vec2 wp = uv * uExt;
        eta += br * ev * 0.35 * (fbm2(wp * 0.18 + vec2(uTime * 0.6, 0.0)) - 0.45) * smoothstep(0.1, 0.7, q);
        // Plunging lip: throw the crest forward just as the wave breaks.
        float lip = plunge * smoothstep(0.8, 1.0, rb) * (1.0 - smoothstep(1.05, 1.4, rb)) * (1.0 - surge) * (1.0 - 0.8 * sz);
        float fwd = lip * Hc * 1.3 * pow(q, 8.0) + 0.25 * s * s * ev * pow(q, 3.0);
        vec2 u = dir.xy / max(length(dir.xy), 1e-4);
        // Whitewater covers the front face and the first stretch behind the crest.
        float tb = t < ff ? 0.0 : (t - ff) / (1.0 - ff);
        float foam = br * (t < ff ? smoothstep(0.2, 0.85, q) : smoothstep(0.32, 0.1, tb)) + lip * 0.8 * smoothstep(0.6, 1.0, q);
        o = vec4(eta, fwd * u.x, fwd * u.y, clamp(foam, 0.0, 1.0));
      }`,
    depthTest: false,
    depthWrite: false,
  });
  const waveQuad = new THREE.Mesh(quadGeo, waveMat);
  waveQuad.frustumCulled = false;
  const waveScene = new THREE.Scene();
  waveScene.add(waveQuad);

  // --- barrel mask -------------------------------------------------------------
  // Under a thrown lip the depth-integrated solver still carries its smeared
  // front; the breakers report the barrel voids (brk.tubes: one strip per lip
  // sample) and this mask pulls the surface down to the trough inside them so
  // the lip curtain forms the roof, and shades the water under the lip.
  const MAX_TUBES = 24 * 512;
  const maskRT = new THREE.WebGLRenderTarget(M, M, rtOpts);
  const footGeo = new THREE.BufferGeometry();
  const footPos = new Float32Array(MAX_TUBES * 4 * 3), footAttr = new Float32Array(MAX_TUBES * 4 * 4); // (a, eta, s, a1)
  footGeo.setAttribute('position', new THREE.BufferAttribute(footPos, 3).setUsage(THREE.DynamicDrawUsage));
  footGeo.setAttribute('attr', new THREE.BufferAttribute(footAttr, 4).setUsage(THREE.DynamicDrawUsage));
  const footIdx = new Uint32Array(MAX_TUBES * 6);
  for (let q = 0; q < MAX_TUBES; q++) {
    const v = q * 4, k = q * 6;
    footIdx[k] = v; footIdx[k + 1] = v + 2; footIdx[k + 2] = v + 1; footIdx[k + 3] = v + 1; footIdx[k + 4] = v + 2; footIdx[k + 5] = v + 3;
  }
  footGeo.setIndex(new THREE.BufferAttribute(footIdx, 1));
  footGeo.setDrawRange(0, 0);
  const maskMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { uExt: shared.uExt },
    vertexShader: /* glsl */ `
      in vec3 position; in vec4 attr;
      uniform float uExt;
      out vec4 vA;
      void main() { vA = attr; gl_Position = vec4(position.xz / (uExt * 0.5), 0.0, 1.0); }`,
    fragmentShader: /* glsl */ `
      precision highp float;
      in vec4 vA;
      out vec4 o;
      void main() {
        // a: metres ahead of the crest root inside the strip [a0, a1]: a wall at
        // the back, soft toward the trough where the lip lands.
        float a = vA.x, a1 = vA.w;
        float edge = clamp((a1 - a) / 0.6, 0.0, 1.0);
        float w = vA.z * edge;
        float shade = vA.z * max(0.0, 1.0 - a / max(a1 * 0.8, 0.1));
        o = vec4(w, vA.y, shade, max(w, shade * 0.999));
      }`,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.CustomBlending,
    blendEquation: THREE.MaxEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
  });
  const footMesh = new THREE.Mesh(footGeo, maskMat);
  footMesh.frustumCulled = false;
  const maskScene = new THREE.Scene();
  maskScene.add(footMesh);
  // Fill the strip geometry from the breakers' tubes.
  function updateFootprint(tubes) {
    let q = 0;
    for (const tb of tubes || []) {
      if (!tb || tb.s <= 0 || q >= MAX_TUBES) continue;
      const px = -tb.dy * tb.hw, py = tb.dx * tb.hw; // half-width along the crest
      const bx = tb.x + tb.dx * tb.a0, by = tb.y + tb.dy * tb.a0;
      const fx = tb.x + tb.dx * tb.a1, fy = tb.y + tb.dy * tb.a1;
      const v = q * 4;
      const put = (k, x, y, a) => {
        footPos[(v + k) * 3] = x; footPos[(v + k) * 3 + 1] = 0; footPos[(v + k) * 3 + 2] = y;
        const o = (v + k) * 4; footAttr[o] = a; footAttr[o + 1] = tb.eta; footAttr[o + 2] = tb.s; footAttr[o + 3] = tb.a1;
      };
      put(0, bx - px, by - py, tb.a0); put(1, bx + px, by + py, tb.a0);
      put(2, fx - px, fy - py, tb.a1); put(3, fx + px, fy + py, tb.a1);
      q++;
    }
    footGeo.attributes.position.needsUpdate = true;
    footGeo.attributes.attr.needsUpdate = true;
    footGeo.attributes.position.addUpdateRange(0, q * 12);
    footGeo.attributes.attr.addUpdateRange(0, q * 16);
    footGeo.setDrawRange(0, q * 6);
    return q > 0;
  }

  // --- present pass: engine state -> the M x M wave texture ----------------
  const presentMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { ...shared, uM: { value: M }, uMask: { value: maskRT.texture } },
    vertexShader: passVS,
    fragmentShader: COMMON + ENGINE_GLSL + /* glsl */ `
      uniform float uM;
      uniform sampler2D uMask;
      out vec4 o;
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5) / (uM - 1.0);
        float wet;
        vec4 e = engineSample(uv, wet);
        vec4 mk = texture(uMask, gl_FragCoord.xy / uM);
        float eta = mix(e.x, min(e.x, mk.y), mk.x);
        // Whitewater source: the breaking field, strongest where the front is fresh.
        // y carries the tube shadow (the water VS ignores displacement in engine mode).
        o = vec4(eta, mk.z, 0.0, smoothstep(0.15, 0.6, e.w) * wet);
      }`,
    depthTest: false,
    depthWrite: false,
  });
  const presentQuad = new THREE.Mesh(quadGeo, presentMat);
  presentQuad.frustumCulled = false;
  const presentScene = new THREE.Scene();
  presentScene.add(presentQuad);

  // --- foam pass -------------------------------------------------------------
  const MAX_SPLATS = 48;
  const splatArray = Array.from({ length: MAX_SPLATS }, () => new THREE.Vector4());
  const foamMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { ...shared, uPrev: { value: null }, uWave: { value: waveRT.texture }, uDt: { value: 0 }, uM: { value: M }, uSplat: { value: splatArray }, uNSplat: { value: 0 } },
    vertexShader: passVS,
    fragmentShader: COMMON + ENGINE_GLSL + /* glsl */ `
      uniform sampler2D uPrev, uWave, uBed, uDir;
      uniform float uDt, uM;
      uniform vec4 uSplat[${MAX_SPLATS}];
      uniform int uNSplat;
      out vec4 o;
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5) / (uM - 1.0);
        vec2 tuv = gl_FragCoord.xy / uM;
        vec2 vel; float h; float wetNow = 0.0;
        if (uEngine > 0.5) {
          float wet;
          vec4 e = engineSample(uv, wet);
          vel = e.yz;
          h = wet > 0.35 ? e.x + uWl - bl(uHeight, uv).x : 0.0;
          wetNow = h > 0.01 ? 1.0 : 0.0;
        } else {
          vec4 d = bl(uDir, uv);
          h = bl(uBed, uv).x;
          vel = d.xy * d.z;
          wetNow = h > 0.02 ? 1.0 : 0.0;
        }
        vec4 pf = texture(uPrev, tuv - vel * uDt / uExt);
        vec4 ps = texture(uPrev, tuv - vel * 0.12 * uDt / uExt);
        vec4 pw = texture(uPrev, tuv);
        float src = texelFetch(uWave, ivec2(gl_FragCoord.xy), 0).w;
        // Foam splats from the breakers (lip impacts, splash-up), in world metres.
        vec2 wp = uv * uExt - uExt * 0.5;
        for (int i = 0; i < ${MAX_SPLATS}; i++) {
          if (i >= uNSplat) break;
          vec4 sp = uSplat[i];
          src = max(src, sp.w * (1.0 - smoothstep(sp.z * 0.5, sp.z, length(wp - sp.xy))));
        }
        float bore = max(pf.r * exp(-uDt / 2.2), src);
        float resid = min(1.0, ps.g * exp(-uDt / 16.0) + bore * uDt * 0.15);
        if (h < 0.02) { bore = 0.0; resid *= 0.9; }
        // Wet sand: where water has been recently, drying out slowly.
        float wetness = max(pw.b * exp(-uDt / 28.0), wetNow);
        o = vec4(bore, resid, wetness, 1.0);
      }`,
    depthTest: false,
    depthWrite: false,
  });
  const foamQuad = new THREE.Mesh(quadGeo, foamMat);
  foamQuad.frustumCulled = false;
  const foamScene = new THREE.Scene();
  foamScene.add(foamQuad);

  // --- sky ---------------------------------------------------------------------
  const skyMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: shared,
    side: THREE.BackSide,
    depthWrite: false,
    vertexShader: COMMON + /* glsl */ `
      out vec3 vD;
      void main() {
        vD = position;
        gl_Position = projectionMatrix * viewMatrix * vec4(position + cameraPosition, 1.0);
      }`,
    fragmentShader: COMMON + /* glsl */ `
      in vec3 vD;
      out vec4 o;
      void main() {
        vec3 d = normalize(vD);
        vec3 c = sky(vec3(d.x, max(d.y, 0.0), d.z));
        // Below the horizon: a hazy distant sea.
        c = mix(c, vec3(0.30, 0.46, 0.56), smoothstep(0.0, -0.08, d.y));
        o = vec4(c, 1.0);
      }`,
  });
  const skyDome = new THREE.Mesh(new THREE.SphereGeometry(4000, 48, 24), skyMat);
  skyDome.frustumCulled = false;
  skyDome.name = 'sky';
  skyDome.renderOrder = -1;
  scene.add(skyDome);

  // --- terrain -----------------------------------------------------------------
  function gridGeometry(n) {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(n * n * 3);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const c = (j * n + i) * 3;
      pos[c] = i; pos[c + 1] = j; pos[c + 2] = 0;
    }
    const idx = new Uint32Array((n - 1) * (n - 1) * 6);
    let k = 0;
    for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
      const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = b;
      idx[k++] = b; idx[k++] = c; idx[k++] = d;
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), EXT * 2);
    return g;
  }

  const terrainMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: shared,
    vertexShader: COMMON + /* glsl */ `
      uniform sampler2D uHeight;
      out vec3 vP; out vec3 vN; out float vRock;
      void main() {
        ivec2 ij = ivec2(position.xy);
        vec4 hc = texelFetch(uHeight, ij, 0);
        float zl = texelFetch(uHeight, clamp(ij - ivec2(1, 0), ivec2(0), ivec2(NN - 1)), 0).x;
        float zr = texelFetch(uHeight, clamp(ij + ivec2(1, 0), ivec2(0), ivec2(NN - 1)), 0).x;
        float zd = texelFetch(uHeight, clamp(ij - ivec2(0, 1), ivec2(0), ivec2(NN - 1)), 0).x;
        float zu = texelFetch(uHeight, clamp(ij + ivec2(0, 1), ivec2(0), ivec2(NN - 1)), 0).x;
        float dx = uExt / float(NN - 1);
        vN = normalize(vec3(-(zr - zl) / (2.0 * dx), 1.0, -(zu - zd) / (2.0 * dx)));
        vec2 xy = position.xy * dx;
        vP = vec3(xy.x - uExt * 0.5, hc.x, xy.y - uExt * 0.5);
        vRock = hc.y;
        gl_Position = projectionMatrix * viewMatrix * vec4(vP, 1.0);
      }`,
    fragmentShader: COMMON + /* glsl */ `
      uniform float uContours, uEngine;
      uniform vec4 uBrush;
      uniform sampler2D uFoam;
      in vec3 vP; in vec3 vN; in float vRock;
      out vec4 o;
      void main() {
        vec2 p = vP.xz + uExt * 0.5;
        vec3 n = normalize(vN);
        vec3 base = bedColor(vP.y, vRock, p);
        if (uEngine > 0.5) {
          // Sand the swash has just left is dark and glossy.
          float wet = texture(uFoam, p / uExt).b * smoothstep(uWl - 0.1, uWl + 0.1, vP.y);
          base *= mix(1.0, 0.66, wet);
        }
        float dif = max(dot(n, SUN), 0.0);
        vec3 col = base * (0.42 * vec3(0.85, 0.92, 1.0) + 0.85 * dif * vec3(1.0, 0.95, 0.86));
        if (uContours > 0.5) {
          float z = vP.y - uWl;
          float l = abs(fract(z + 0.5) - 0.5) / fwidth(z);
          col = mix(col, col * 0.55, (1.0 - min(l, 1.0)) * 0.8);
        }
        if (uBrush.w > 0.5) {
          float d = length(p - uBrush.xy);
          float ring = 1.0 - smoothstep(0.0, 1.6, abs(d - uBrush.z));
          col = mix(col, vec3(1.0), ring * 0.85);
        }
        o = vec4(col, 1.0);
      }`,
  });
  const terrain = new THREE.Mesh(gridGeometry(N), terrainMat);
  terrain.frustumCulled = false;
  terrain.name = 'terrain';
  scene.add(terrain);

  // --- diorama walls ---------------------------------------------------------
  function wallGeometry(n) {
    // Strip around the perimeter: attribute position = (gx, gy, top?)
    const pts = [];
    for (let i = 0; i < n; i++) pts.push([i, 0]);
    for (let j = 1; j < n; j++) pts.push([n - 1, j]);
    for (let i = n - 2; i >= 0; i--) pts.push([i, n - 1]);
    for (let j = n - 2; j >= 0; j--) pts.push([0, j]);
    const pos = new Float32Array(pts.length * 2 * 3);
    pts.forEach(([x, y], k) => {
      pos.set([x, y, 0], k * 6);
      pos.set([x, y, 1], k * 6 + 3);
    });
    const idx = [];
    for (let k = 0; k < pts.length - 1; k++) {
      const a = k * 2, b = a + 1, c = a + 2, d = a + 3;
      idx.push(a, b, c, c, b, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), EXT * 2);
    return g;
  }
  const BASE = -26;
  const wallMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: shared,
    side: THREE.DoubleSide,
    vertexShader: COMMON + /* glsl */ `
      uniform sampler2D uHeight;
      out vec3 vP; out float vTop;
      void main() {
        ivec2 ij = ivec2(position.xy);
        float z = texelFetch(uHeight, ij, 0).x;
        float dx = uExt / float(NN - 1);
        vTop = z;
        vP = vec3(position.x * dx - uExt * 0.5, position.z > 0.5 ? z : ${BASE.toFixed(1)}, position.y * dx - uExt * 0.5);
        gl_Position = projectionMatrix * viewMatrix * vec4(vP, 1.0);
      }`,
    fragmentShader: COMMON + /* glsl */ `
      in vec3 vP; in float vTop;
      out vec4 o;
      void main() {
        float depth = vTop - vP.y;
        float band = fbm2(vec2(vP.x + vP.z, vP.y * 6.0) * 0.08);
        vec3 c = mix(vec3(0.76, 0.66, 0.49), vec3(0.47, 0.39, 0.30), smoothstep(0.5, 9.0, depth));
        c *= 0.82 + 0.25 * band;
        c = mix(c, vec3(0.20, 0.17, 0.15), smoothstep(14.0, 26.0, depth));
        float side = abs(vP.x) > abs(vP.z) ? 0.78 : 0.95;
        o = vec4(c * side, 1.0);
      }`,
  });
  const walls = new THREE.Mesh(wallGeometry(N), wallMat);
  walls.frustumCulled = false;
  walls.name = 'walls';
  scene.add(walls);

  // --- water surface -----------------------------------------------------------
  const waterUniforms = {
    ...shared,
    uWave: { value: waveRT.texture },
    uFoam: { value: foamA.texture },
    uM: { value: M },
    uCam: { value: camera.position },
    uAlpha: { value: 1 },
    uDebug: { value: 0 },
  };
  const waterVS = COMMON + /* glsl */ `
    uniform sampler2D uWave, uHeight;
    uniform float uM, uEngine;
    out vec3 vP; out vec3 vN; out float vBedZ; out float vRock; out vec2 vUv; out float vEta; out float vShade;
    vec3 wpos(ivec2 ij) {
      ij = clamp(ij, ivec2(0), ivec2(int(uM) - 1));
      vec4 w = texelFetch(uWave, ij, 0);
      vec2 uv = vec2(ij) / (uM - 1.0);
      vec2 disp = w.yz * (1.0 - uEngine);
      return vec3(uv.x * uExt - uExt * 0.5 + disp.x, uWl + w.x, uv.y * uExt - uExt * 0.5 + disp.y);
    }
    void main() {
      ivec2 ij = ivec2(position.xy);
      vec3 p = wpos(ij);
      vShade = uEngine * texelFetch(uWave, clamp(ij, ivec2(0), ivec2(int(uM) - 1)), 0).y;
      vec3 n = normalize(cross(wpos(ij + ivec2(0, 1)) - wpos(ij - ivec2(0, 1)), wpos(ij + ivec2(1, 0)) - wpos(ij - ivec2(1, 0))));
      vUv = vec2(ij) / (uM - 1.0);
      vec4 hc = bl(uHeight, vUv);
      vBedZ = hc.x; vRock = hc.y;
      vEta = p.y - uWl;
      vP = p; vN = n;
      gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
    }`;
  const waterFS = COMMON + /* glsl */ `
    uniform sampler2D uFoam;
    uniform vec3 uCam;
    uniform float uContours, uTime, uAlpha, uDebug;
    uniform vec4 uBrush;
    in vec3 vP; in vec3 vN; in float vBedZ; in float vRock; in vec2 vUv; in float vEta; in float vShade;
    out vec4 o;
    void main() {
      float col_h = vP.y - vBedZ;
      if (uDebug > 0.5) {
        vec4 fd = texture(uFoam, vUv);
        if (uDebug < 1.5) o = vec4(fd.rgb, 1.0);
        else if (uDebug < 2.5) o = vec4(vec3(col_h / 10.0), 1.0);
        else if (uDebug < 3.5) o = vec4(vUv, 0.0, 1.0);
        else if (uDebug < 4.5) o = vec4(normalize(vN) * 0.5 + 0.5, 1.0);
        else o = vec4(vec3(vEta * 0.25 + 0.5), 1.0);
        return;
      }
      if (col_h < 0.0) discard;
      vec2 p = vP.xz + uExt * 0.5;
      vec3 n = normalize(vN);
      if (!gl_FrontFacing) n = -n;
      vec3 v = normalize(uCam - vP);
      // Shading normal: exaggerate slopes when looking down from above so
      // long, low swells still read as lines; true normals at eye level.
      float ex = mix(1.0, 5.0, smoothstep(0.12, 0.55, v.y));
      vec3 ns = normalize(vec3(n.x * ex, n.y, n.z * ex));
      // Body colour: the bed seen through the water column.
      vec3 bed = bedColor(vBedZ, vRock, p) * (0.5 + 0.5 * max(SUN.y, 0.0));
      vec3 absorb = exp(-col_h * vec3(0.55, 0.16, 0.12));
      vec3 deep = vec3(0.015, 0.11, 0.17);
      vec3 body = bed * absorb + deep * (1.0 - exp(-col_h * 0.32)) + vec3(0.0, 0.07, 0.07) * (1.0 - absorb.g);
      body *= 0.5 + 0.8 * max(dot(ns, normalize(SUN + vec3(0.0, 0.25, 0.0))), 0.0);
      body += vec3(0.02, 0.07, 0.07) * clamp(vEta, -1.0, 2.0);
      // Light through thin crests and wave faces.
      float thick = smoothstep(0.2, 2.0, vEta);
      float sss = pow(max(dot(v, -normalize(SUN + n * 0.5)), 0.0), 3.0) * thick;
      body += vec3(0.05, 0.42, 0.36) * (sss * 0.6 + thick * (0.22 + 0.3 * (1.0 - n.y)));
      float fres = 0.02 + 0.98 * pow(1.0 - max(dot(ns, v), 0.0), 5.0);
      vec3 refl = sky(reflect(-v, ns));
      vec3 c = mix(body, refl, fres);
      // Inside a barrel the water is in the lip's shadow.
      c *= 1.0 - 0.55 * clamp(vShade, 0.0, 1.0);
      vec3 hv = normalize(SUN + v);
      c += vec3(1.0, 0.9, 0.7) * pow(max(dot(n, hv), 0.0), 400.0) * 3.0;
      // Foam: active whitewater plus the lacy residue it leaves behind.
      vec4 f = texture(uFoam, vUv);
      float tex = fbm2(p * 0.55 + vec2(0.0, uTime * 0.25)) * 0.7 + 0.3 * fbm2(p * 2.1 - uTime * 0.3);
      // Fine lumps so a sheet of whitewater is not a flat white field up close.
      float fine = fbm2(p * 1.3 + vec2(uTime * 0.35, -uTime * 0.2));
      float act = smoothstep(0.05, 0.45, f.r * 1.15 - tex * 0.6 + 0.05) * (0.8 + 0.2 * smoothstep(0.25, 0.75, fine));
      float lace = smoothstep(0.66, 0.9, tex * 1.1 + 0.3 * f.g) * smoothstep(0.02, 0.45, f.g);
      float shore = smoothstep(0.3, 0.0, col_h) * 0.6 * smoothstep(0.35, 0.65, fbm2(p * 0.25 + uTime * 0.2));
      float foam = clamp(max(act, max(lace * 0.85, shore)), 0.0, 1.0);
      vec3 fc = (vec3(0.93, 0.95, 0.96) * (0.72 + 0.28 * max(dot(n, SUN), 0.0)) + 0.06) * (0.9 + 0.12 * fine);
      c = mix(c, fc, foam);
      if (uContours > 0.5) {
        float z = vBedZ - uWl;
        float l = abs(fract(z + 0.5) - 0.5) / fwidth(z);
        c = mix(c, c * 0.6, (1.0 - min(l, 1.0)) * 0.55);
      }
      if (uBrush.w > 0.5) {
        float d = length(p - uBrush.xy);
        c = mix(c, vec3(1.0), (1.0 - smoothstep(0.0, 1.6, abs(d - uBrush.z))) * 0.9);
      }
      float a = mix(uAlpha, 1.0, foam);
      o = vec4(c, a);
    }`;
  const waterMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: waterUniforms,
    vertexShader: waterVS,
    fragmentShader: waterFS,
    side: THREE.DoubleSide,
  });
  const water = new THREE.Mesh(gridGeometry(M), waterMat);
  water.frustumCulled = false;
  water.name = 'water';
  water.renderOrder = 2;
  scene.add(water);

  // Water cross-section on the diorama walls.
  const wwallMat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: waterUniforms,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    vertexShader: COMMON + /* glsl */ `
      uniform sampler2D uWave, uHeight;
      uniform float uM;
      out vec3 vP; out float vBed; out float vTop;
      void main() {
        ivec2 ij = ivec2(position.xy);
        vec4 w = texelFetch(uWave, ij, 0);
        vec2 uv = vec2(ij) / (uM - 1.0);
        float bz = bl(uHeight, uv).x;
        float top = uWl + w.x;
        vBed = bz; vTop = top;
        vP = vec3(uv.x * uExt - uExt * 0.5, position.z > 0.5 ? max(top, bz) : bz, uv.y * uExt - uExt * 0.5);
        gl_Position = projectionMatrix * viewMatrix * vec4(vP, 1.0);
      }`,
    fragmentShader: COMMON + /* glsl */ `
      in vec3 vP; in float vBed; in float vTop;
      out vec4 o;
      void main() {
        if (vTop < vBed + 0.01) discard;
        float d = vTop - vP.y;
        vec3 c = mix(vec3(0.20, 0.62, 0.66), vec3(0.03, 0.18, 0.30), smoothstep(0.0, 9.0, d));
        o = vec4(c, 0.82);
      }`,
  });
  const wwalls = new THREE.Mesh(wallGeometry(M), wwallMat);
  wwalls.frustumCulled = false;
  wwalls.name = 'wwalls';
  wwalls.renderOrder = 3;
  scene.add(wwalls);

  // Swell direction arrows, floating just offshore of the block.
  const arrows = new THREE.Group();
  const arrowMat = new THREE.MeshBasicMaterial({ color: 0xf2f5f4, transparent: true, opacity: 0.85 });
  for (let k = -1; k <= 1; k++) {
    const g = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.BoxGeometry(3, 1.2, 46), arrowMat);
    shaft.position.z = -10;
    const head = new THREE.Mesh(new THREE.ConeGeometry(8, 18, 3), arrowMat);
    head.rotation.x = Math.PI / 2;
    head.position.z = 20;
    g.add(shaft, head);
    g.userData.k = k;
    arrows.add(g);
  }
  scene.add(arrows);
  function setSwellArrow(dirDeg) {
    const th = (dirDeg * Math.PI) / 180;
    arrows.children.forEach((g) => {
      const k = g.userData.k;
      g.position.set(k * 160 - Math.sin(th) * 60, 6, -EXT / 2 - 70 - Math.cos(th) * 10);
      g.rotation.y = th; // +z is shoreward, rotate toward +x for positive angles
    });
  }

  // --- API -------------------------------------------------------------------
  function setHeight(beach) {
    const d = heightTex.image.data;
    for (let c = 0; c < N * N; c++) { d[c * 4] = beach.z[c]; d[c * 4 + 1] = beach.rock[c]; }
    heightTex.needsUpdate = true;
  }

  let omegas = new Array(NCOMP).fill(0), phases = new Array(NCOMP).fill(0);
  function setWaves(res) {
    for (let q = 0; q < 4; q++) {
      const d = compTex[q].image.data;
      const S0 = res.S[2 * q], A0 = res.A[2 * q], S1 = res.S[2 * q + 1], A1 = res.A[2 * q + 1];
      for (let c = 0; c < N * N; c++) {
        d[c * 4] = S0[c]; d[c * 4 + 1] = A0[c]; d[c * 4 + 2] = S1[c]; d[c * 4 + 3] = A1[c];
      }
      compTex[q].needsUpdate = true;
    }
    const b = bedTex.image.data, r = dirTex.image.data, g = grpTex.image.data;
    for (let c = 0; c < N * N; c++) {
      b[c * 4] = res.h[c]; b[c * 4 + 1] = res.D[c]; b[c * 4 + 2] = res.gamma[c]; b[c * 4 + 3] = res.plunge[c];
      r[c * 4] = res.ux[c]; r[c * 4 + 1] = res.uy[c]; r[c * 4 + 2] = res.speed[c]; r[c * 4 + 3] = res.surge[c];
      g[c * 4] = res.tau[c]; g[c * 4 + 1] = res.lambda[c];
    }
    bedTex.needsUpdate = dirTex.needsUpdate = grpTex.needsUpdate = true;
    omegas = res.omegas; phases = res.phases;
    waveMat.uniforms.uSetOmega.value = res.setOmega;
    waveMat.uniforms.uSetDepth.value = res.setDepth;
    waveMat.uniforms.uActive.value = 1;
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    }
  }

  let simTime = 0;
  let engine = null;
  function setEngine(e) {
    if (engine && engine.group) scene.remove(engine.group);
    engine = e;
    shared.uEngine.value = e ? 1 : 0;
    if (e) {
      shared.uState.value = e.textures.state;
      shared.uBathy.value = e.textures.bathy;
      shared.uLay.value.set(e.layout.ML, e.layout.MO, e.layout.dxs, 0);
      if (e.group) scene.add(e.group);
    }
  }
  function frame(dt) {
    resize();
    controls.update();
    arrows.visible = camera.position.y > 80; // only from high up; they would sit on the horizon at eye level
    simTime += dt;
    shared.uTime.value = simTime;
    if (engine) {
      engine.advance(dt, { camera, viewportHeight: canvas.height });
      shared.uState.value = engine.textures.state; // ping-pong: the current state texture
      if (engine.group && engine.group.parent !== scene) scene.add(engine.group);
      const sp = engine.splats || [];
      const n = Math.min(sp.length, MAX_SPLATS);
      for (let i = 0; i < n; i++) splatArray[i].set(sp[i].x, sp[i].y, sp[i].r, sp[i].s);
      foamMat.uniforms.uNSplat.value = n;
      renderer.setRenderTarget(maskRT);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      if (engine.breakers && updateFootprint(engine.breakers.tubes)) renderer.render(maskScene, quadCam);
      renderer.setRenderTarget(waveRT);
      renderer.render(presentScene, quadCam);
    } else {
      const ph = waveMat.uniforms.uPh.value;
      for (let m = 0; m < NCOMP; m++) {
        let v = (phases[m] - omegas[m] * simTime) % (2 * Math.PI);
        ph[m] = v;
      }
      foamMat.uniforms.uNSplat.value = 0;
      renderer.setRenderTarget(waveRT);
      renderer.render(waveScene, quadCam);
    }
    foamMat.uniforms.uPrev.value = foamA.texture;
    foamMat.uniforms.uDt.value = Math.min(dt, 0.1);
    renderer.setRenderTarget(foamB);
    renderer.render(foamScene, quadCam);
    [foamA, foamB] = [foamB, foamA];
    waterUniforms.uFoam.value = foamA.texture;
    shared.uFoam.value = foamA.texture;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
  }

  // Camera presets. `info` describes the current beach: shoreZ (world z of
  // the waterline at the centre), breakDist (m from the waterline to where
  // the waves break) and ground(x, z) (sand elevation).
  function setView(name, info) {
    const wl = shared.uWl.value;
    const ground = info ? info.ground : () => wl;
    const hb = info ? info.breakHeight : 3; // how tall the breaking waves are
    // The eye-level views stand where the breaking is heaviest (a bar, not
    // the rip channel beside it); fall back to the centre line.
    const sp = info && info.spot && info.spot.breakZ != null ? info.spot : null;
    const sx = sp ? sp.x : 0;
    const shoreZ = sp && sp.shoreZ != null ? sp.shoreZ : info ? info.shoreZ : EXT * 0.34;
    const zt = sp ? sp.breakZ : shoreZ - Math.max(info ? info.breakDistTypical : 100, 40); // where the typical wave breaks
    let p, t, fov = 42;
    if (name === 'lineup') {
      // Floating just outside the break, looking along the line at the peak.
      p = [sx - 70, wl + 1.8 + 0.3 * hb, zt - 20];
      t = [sx + 30, wl + 0.6, zt + 20];
      fov = 30;
    } else if (name === 'impact') {
      // In the whitewater inside the break, facing the waves. Stands further
      // in, and rides higher, when the waves are big, so the bores form in
      // front of it and the whitewater passes underneath.
      const y = wl + Math.max(2.0, 0.55 * hb);
      const inset = 16 + 3 * (sp ? sp.depth : 3);
      p = [sx + 10, y, zt + inset];
      t = [sx - 4, y - 1.0, zt - 15];
      fov = 30;
    } else if (name === 'shore') {
      // Standing on the sand a few steps up from the water, at eye level.
      const zc = shoreZ + 12;
      p = [sx, Math.max(ground(sx, zc), wl) + 1.7, zc];
      t = [sx - 15, wl + 1.2, zt - 40];
      fov = 30;
    } else if (name === 'top') {
      p = [0, 760, 1]; t = [0, 0, 0];
    } else {
      p = [-520, 290, 430]; t = [-100, -6, 10];
    }
    camera.fov = fov;
    camera.updateProjectionMatrix();
    camera.position.set(...p);
    controls.target.set(...t);
    controls.update();
  }
  setView('overview');
  function lookAt(p, t) { camera.position.set(...p); controls.target.set(...t); controls.update(); }

  // Ray in grid coordinates (x alongshore, y cross-shore, elevation).
  const ndc = new THREE.Vector2();
  const ray = new THREE.Raycaster();
  function gridRay(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const o = ray.ray.origin, d = ray.ray.direction;
    return { o: [o.x + EXT / 2, o.z + EXT / 2, o.y], d: [d.x, d.z, d.y] };
  }

  return {
    renderer, camera, controls, shared, scene,
    lookAt, setHeight, setWaves, setSwellArrow, frame, setView, gridRay, setEngine,
    setWaterLevel: (wl) => { shared.uWl.value = wl; },
    setContours: (on) => { shared.uContours.value = on ? 1 : 0; },
    setSeeThrough: (on) => {
      waterMat.transparent = on; waterUniforms.uAlpha.value = on ? 0.45 : 1; waterMat.depthWrite = !on; waterMat.needsUpdate = true;
    },
    setBrush: (x, y, r, vis) => shared.uBrush.value.set(x, y, r, vis ? 1 : 0),
    resetFoam: () => {
      for (const t of [foamA, foamB]) { renderer.setRenderTarget(t); renderer.clear(); }
      renderer.setRenderTarget(null);
    },
    gridSize: M,
    setDebug: (v) => { waterUniforms.uDebug.value = v; },
    // Test hook: elevation statistics of the current wave field.
    waveStats: () => {
      const b = new Uint16Array(M * M * 4);
      renderer.readRenderTargetPixels(waveRT, 0, 0, M, M, b);
      let mx = -1e9, mn = 1e9, foam = 0;
      for (let i = 0; i < M * M; i++) { const v = THREE.DataUtils.fromHalfFloat(b[i * 4]); if (v > mx) mx = v; if (v < mn) mn = v; foam = Math.max(foam, THREE.DataUtils.fromHalfFloat(b[i * 4 + 3])); }
      renderer.readRenderTargetPixels(foamA, 0, 0, M, M, b);
      let bore = 0, resid = 0;
      for (let i = 0; i < M * M; i++) { bore = Math.max(bore, THREE.DataUtils.fromHalfFloat(b[i * 4])); resid = Math.max(resid, THREE.DataUtils.fromHalfFloat(b[i * 4 + 1])); }
      return { mx, mn, foam, bore, resid };
    },
  };
}
