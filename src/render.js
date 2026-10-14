// Three.js scene: the beach as a diorama block, the water surface driven by
// a GPU wave pass, and a foam pass that remembers where waves broke.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { N, EXT } from './terrain.js';
import { NCOMP } from './waves.js';

const SUN = new THREE.Vector3(-0.45, 0.42, -0.79).normalize();

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
  const camera = new THREE.PerspectiveCamera(42, 1, 1, 6000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 30;
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
    uContours: { value: 0 },
    uBrush: { value: new THREE.Vector4(0, 0, 0, 0) }, // x, y, radius, visible
  };

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
    },
    vertexShader: passVS,
    fragmentShader: COMMON + /* glsl */ `
      uniform sampler2D uC0, uC1, uC2, uC3, uBed, uDir;
      uniform float uPh[${NCOMP}];
      uniform float uM, uActive, uTime;
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
        float env = length(zc) * bed.y;
        float phi = atan(zc.y, zc.x);
        float gam = bed.z, plunge = bed.w;
        float surge = bl(uDir, uv).w;
        float r = 2.0 * env / (gam * h);
        // Inside the surf zone (where the solver took energy out) a wave that
        // has already broken keeps going as a bore at a lower height ratio.
        float sz = smoothstep(0.03, 0.2, 1.0 - bed.y);
        float rb = r / mix(1.0, 0.6, sz);
        float ev = min(env, 0.52 * gam * h);
        float s = clamp(r, 0.0, 1.0);
        float br = smoothstep(0.92, 1.15, rb) * (1.0 - 0.7 * surge);
        // Wave profile: phi > 0 is ahead of the crest. As the wave shoals the
        // front face takes up less of the wavelength (it steepens) and the
        // crest sharpens while the trough flattens.
        float p = mix(1.0 + 2.4 * s * s, 1.7, br);
        float ff = mix(mix(0.5, 0.12, s * s * s), 0.09, br);
        float t = fract(phi / 6.28318);
        float th = t < ff ? 3.14159 * t / ff : 3.14159 * (1.0 + (t - ff) / (1.0 - ff));
        float q = 0.5 + 0.5 * cos(th);
        float m = inversesqrt(3.14159 * (p + 0.25));
        float eta = 2.0 * ev * (pow(q, p) - m);
        // Whitewater is lumpy and aerated.
        vec2 wp = uv * uExt;
        eta += br * ev * 0.35 * (fbm2(wp * 0.18 + vec2(uTime * 0.6, 0.0)) - 0.45) * smoothstep(0.1, 0.7, q);
        // Plunging lip: throw the crest forward just as the wave breaks.
        float lip = plunge * smoothstep(0.8, 1.0, r) * (1.0 - smoothstep(1.05, 1.4, r)) * (1.0 - surge) * (1.0 - 0.8 * sz);
        float fwd = lip * ev * 2.6 * pow(q, 8.0) + 0.25 * s * s * ev * pow(q, 3.0);
        vec2 u = bl(uDir, uv).xy;
        u /= max(length(u), 1e-4);
        float foam = br * smoothstep(0.2, 0.85, q) * (t < 0.5 ? 1.0 : smoothstep(0.75, 0.95, t)) + lip * 0.8 * smoothstep(0.6, 1.0, q);
        o = vec4(eta, fwd * u.x, fwd * u.y, clamp(foam, 0.0, 1.0));
      }`,
    depthTest: false,
    depthWrite: false,
  });
  const waveQuad = new THREE.Mesh(quadGeo, waveMat);
  waveQuad.frustumCulled = false;
  const waveScene = new THREE.Scene();
  waveScene.add(waveQuad);

  // --- foam pass -------------------------------------------------------------
  const foamMat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { ...shared, uPrev: { value: null }, uWave: { value: waveRT.texture }, uDt: { value: 0 }, uM: { value: M } },
    vertexShader: passVS,
    fragmentShader: COMMON + /* glsl */ `
      uniform sampler2D uPrev, uWave, uBed, uDir;
      uniform float uDt, uM;
      out vec4 o;
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5) / (uM - 1.0);
        vec2 tuv = gl_FragCoord.xy / uM;
        vec4 d = bl(uDir, uv);
        float h = bl(uBed, uv).x;
        vec2 vel = d.xy * d.z;
        vec4 pf = texture(uPrev, tuv - vel * uDt / uExt);
        vec4 ps = texture(uPrev, tuv - vel * 0.12 * uDt / uExt);
        float src = texelFetch(uWave, ivec2(gl_FragCoord.xy), 0).w;
        float bore = max(pf.r * exp(-uDt / 2.2), src);
        float resid = min(1.0, ps.g * exp(-uDt / 16.0) + bore * uDt * 0.45);
        if (h < 0.02) { bore = 0.0; resid *= 0.9; }
        o = vec4(bore, resid, 0.0, 1.0);
      }`,
    depthTest: false,
    depthWrite: false,
  });
  const foamQuad = new THREE.Mesh(quadGeo, foamMat);
  foamQuad.frustumCulled = false;
  const foamScene = new THREE.Scene();
  foamScene.add(foamQuad);

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
      uniform float uContours;
      uniform vec4 uBrush;
      in vec3 vP; in vec3 vN; in float vRock;
      out vec4 o;
      void main() {
        vec2 p = vP.xz + uExt * 0.5;
        vec3 n = normalize(vN);
        vec3 base = bedColor(vP.y, vRock, p);
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
  scene.add(walls);

  // --- water surface -----------------------------------------------------------
  const waterUniforms = {
    ...shared,
    uWave: { value: waveRT.texture },
    uFoam: { value: foamA.texture },
    uM: { value: M },
    uCam: { value: camera.position },
    uAlpha: { value: 1 },
  };
  const waterVS = COMMON + /* glsl */ `
    uniform sampler2D uWave, uHeight;
    uniform float uM;
    out vec3 vP; out vec3 vN; out float vBedZ; out float vRock; out vec2 vUv; out float vEta;
    vec3 wpos(ivec2 ij) {
      ij = clamp(ij, ivec2(0), ivec2(int(uM) - 1));
      vec4 w = texelFetch(uWave, ij, 0);
      vec2 uv = vec2(ij) / (uM - 1.0);
      return vec3(uv.x * uExt - uExt * 0.5 + w.y, uWl + w.x, uv.y * uExt - uExt * 0.5 + w.z);
    }
    void main() {
      ivec2 ij = ivec2(position.xy);
      vec3 p = wpos(ij);
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
    uniform float uContours, uTime, uAlpha;
    uniform vec4 uBrush;
    in vec3 vP; in vec3 vN; in float vBedZ; in float vRock; in vec2 vUv; in float vEta;
    out vec4 o;
    void main() {
      float col_h = vP.y - vBedZ;
      if (col_h < 0.0) discard;
      vec2 p = vP.xz + uExt * 0.5;
      vec3 n = normalize(vN);
      if (!gl_FrontFacing) n = -n;
      vec3 v = normalize(uCam - vP);
      float still = max(uWl - vBedZ, 0.0);
      // Body colour: the bed seen through the water column.
      vec3 bed = bedColor(vBedZ, vRock, p) * (0.5 + 0.5 * max(SUN.y, 0.0));
      vec3 absorb = exp(-col_h * vec3(0.55, 0.16, 0.12));
      vec3 deep = vec3(0.015, 0.11, 0.17);
      vec3 body = bed * absorb + deep * (1.0 - exp(-col_h * 0.32)) + vec3(0.0, 0.07, 0.07) * (1.0 - absorb.g);
      // Light through thin crests and wave faces.
      float thick = smoothstep(0.2, 2.0, vEta);
      float sss = pow(max(dot(v, -normalize(SUN + n * 0.5)), 0.0), 3.0) * thick;
      body += vec3(0.05, 0.42, 0.36) * (sss * 0.9 + thick * 0.18 * (1.0 - n.y));
      // Exaggerate slopes for shading only, so long low swells read as lines.
      vec3 ns = normalize(vec3(n.x * 5.0, n.y, n.z * 5.0));
      body *= 0.5 + 0.8 * max(dot(ns, normalize(SUN + vec3(0.0, 0.25, 0.0))), 0.0);
      body += vec3(0.02, 0.07, 0.07) * clamp(vEta, -1.0, 2.0);
      float fres = 0.02 + 0.98 * pow(1.0 - max(dot(ns, v), 0.0), 5.0);
      vec3 refl = sky(reflect(-v, ns));
      vec3 c = mix(body, refl, fres);
      vec3 hv = normalize(SUN + v);
      c += vec3(1.0, 0.9, 0.7) * pow(max(dot(n, hv), 0.0), 400.0) * 3.0;
      // Foam: active whitewater plus the lacy residue it leaves behind.
      vec4 f = texture(uFoam, vUv);
      float tex = fbm2(p * 0.55 + vec2(0.0, uTime * 0.25)) * 0.7 + 0.3 * fbm2(p * 2.1 - uTime * 0.3);
      float act = smoothstep(0.05, 0.45, f.r * 1.15 - tex * 0.6 + 0.05);
      float lace = smoothstep(0.55, 0.75, tex + f.g * 0.55) * smoothstep(0.02, 0.35, f.g);
      float shore = smoothstep(0.3, 0.0, col_h) * 0.6 * smoothstep(0.35, 0.65, fbm2(p * 0.25 + uTime * 0.2));
      float foam = clamp(max(act, max(lace * 0.85, shore)), 0.0, 1.0);
      vec3 fc = vec3(0.93, 0.95, 0.96) * (0.72 + 0.28 * max(dot(n, SUN), 0.0)) + 0.06;
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
    const b = bedTex.image.data, r = dirTex.image.data;
    for (let c = 0; c < N * N; c++) {
      b[c * 4] = res.h[c]; b[c * 4 + 1] = res.D[c]; b[c * 4 + 2] = res.gamma[c]; b[c * 4 + 3] = res.plunge[c];
      r[c * 4] = res.ux[c]; r[c * 4 + 1] = res.uy[c]; r[c * 4 + 2] = res.speed[c]; r[c * 4 + 3] = res.surge[c];
    }
    bedTex.needsUpdate = dirTex.needsUpdate = true;
    omegas = res.omegas; phases = res.phases;
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
  function frame(dt) {
    resize();
    controls.update();
    simTime += dt;
    shared.uTime.value = simTime;
    const ph = waveMat.uniforms.uPh.value;
    for (let m = 0; m < NCOMP; m++) {
      let v = (phases[m] - omegas[m] * simTime) % (2 * Math.PI);
      ph[m] = v;
    }
    renderer.setRenderTarget(waveRT);
    renderer.render(waveScene, quadCam);
    foamMat.uniforms.uPrev.value = foamA.texture;
    foamMat.uniforms.uDt.value = Math.min(dt, 0.1);
    renderer.setRenderTarget(foamB);
    renderer.render(foamScene, quadCam);
    [foamA, foamB] = [foamB, foamA];
    waterUniforms.uFoam.value = foamA.texture;
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
  }

  function setView(name) {
    const views = {
      overview: [[-520, 290, 430], [-100, -6, 10]],
      lineup: [[-70, 6, -30], [60, 0, 150]],
      top: [[0, 760, 1], [0, 0, 0]],
      shore: [[40, 9, 250], [-20, 0, 0]],
    };
    const [p, t] = views[name] || views.overview;
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
    renderer, camera, controls, shared,
    lookAt, setHeight, setWaves, setSwellArrow, frame, setView, gridRay,
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
  };
}
