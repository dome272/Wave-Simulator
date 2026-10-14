// Spray: a CPU particle system rendered as THREE.Points with soft white
// sprites (additive, sized by distance). Particles are launched by the
// emitters the lip model produces each frame (lip tip, impact line, splash-up
// top), fall under gravity with air drag, and drift with the wind.
import * as THREE from 'three';

export const SPRAY_DEFAULTS = {
  maxParticles: 8000,
  wind: [0.4, 0, -1.2], // m/s, world x/y/z (a light offshore breeze blows the spray back over the lip)
  gravity: 9.81,
  drag: 0.9, // 1/s, relaxation of the velocity toward the wind
  pointScale: 700, // pixels * metres: gl_PointSize = size * pointScale / distance; set from the camera with setCamera()
  sizeJitter: 0.6,
};

let seed = 12345;
function rnd() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; }
function gauss() { return (rnd() + rnd() + rnd() - 1.5) * 1.6; }

export function createSprayMaterial(opts = {}) {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { uScale: { value: opts.pointScale ?? SPRAY_DEFAULTS.pointScale }, uSun: { value: new THREE.Vector3(-0.45, 0.42, 0.79).normalize() } },
    vertexShader: /* glsl */ `
      in float aSize; in float aLife; in float aSeed;
      out float vLife; out float vSeed;
      uniform float uScale;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = clamp(aSize * uScale / max(-mv.z, 0.3), 1.0, 96.0);
        vLife = aLife; vSeed = aSeed;
      }`,
    fragmentShader: /* glsl */ `
      in float vLife; in float vSeed;
      out vec4 o;
      void main() {
        vec2 d = gl_PointCoord - 0.5;
        float r2 = dot(d, d) * 4.0;
        if (r2 > 1.0) discard;
        float a = (1.0 - r2) * (1.0 - r2);
        float fade = smoothstep(0.0, 0.12, vLife) * (0.25 + 0.75 * vLife);
        vec3 c = mix(vec3(0.86, 0.93, 0.97), vec3(1.0, 0.98, 0.94), vSeed);
        o = vec4(c * a * fade * 0.45, a * fade * 0.45);
      }`,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
  });
}

export function createSpray(userOpts = {}) {
  const o = { ...SPRAY_DEFAULTS, ...userOpts };
  const MAXP = o.maxParticles;
  const px = new Float32Array(MAXP * 3); // positions (also the geometry attribute)
  const vel = new Float32Array(MAXP * 3);
  const age = new Float32Array(MAXP), life = new Float32Array(MAXP);
  const size = new Float32Array(MAXP), lifeN = new Float32Array(MAXP), seedA = new Float32Array(MAXP);
  let n = 0;

  const geo = new THREE.BufferGeometry();
  const posA = new THREE.BufferAttribute(px, 3), sizeA = new THREE.BufferAttribute(size, 1), lifeA = new THREE.BufferAttribute(lifeN, 1), seedAtt = new THREE.BufferAttribute(seedA, 1);
  for (const a of [posA, sizeA, lifeA, seedAtt]) a.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', posA);
  geo.setAttribute('aSize', sizeA);
  geo.setAttribute('aLife', lifeA);
  geo.setAttribute('aSeed', seedAtt);
  geo.setDrawRange(0, 0);
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 2000);
  const material = createSprayMaterial(o);
  const points = new THREE.Points(geo, material);
  points.frustumCulled = false;
  points.name = 'spray';
  points.renderOrder = o.renderOrder ?? 5; // index.js passes curtain renderOrder + 1

  const wind = new Float32Array(o.wind);

  function spawnOne(e) {
    if (n >= MAXP) return;
    const i = n++;
    const sp = e.spread || 1;
    px[i * 3] = e.x + gauss() * 0.15 * (e.jitter ?? 1); px[i * 3 + 1] = e.y + gauss() * 0.1; px[i * 3 + 2] = e.z + gauss() * 0.15 * (e.jitter ?? 1);
    vel[i * 3] = e.vx + gauss() * sp; vel[i * 3 + 1] = e.vy + gauss() * sp * 0.8; vel[i * 3 + 2] = e.vz + gauss() * sp;
    age[i] = 0;
    life[i] = (e.life || 0.8) * (0.6 + 0.8 * rnd());
    size[i] = (e.size || 0.2) * (1 - o.sizeJitter * 0.5 + o.sizeJitter * rnd());
    seedA[i] = rnd();
  }

  // emitters: [{ x, y, z, vx, vy, vz, spread, rate (particles/s), size (m), life (s) }]
  function emit(emitters, dt) {
    let want = 0;
    for (const e of emitters) want += e.rate * dt;
    const room = MAXP - n;
    const scale = want > room ? room / want : 1;
    for (const e of emitters) {
      const cnt = Math.floor(e.rate * dt * scale + rnd());
      for (let q = 0; q < cnt; q++) spawnOne(e);
    }
  }

  function step(dt) {
    const g = o.gravity, k = Math.min(1, o.drag * dt);
    for (let i = 0; i < n; i++) {
      age[i] += dt;
      if (age[i] >= life[i]) { // swap-remove
        n--;
        if (i !== n) {
          px[i * 3] = px[n * 3]; px[i * 3 + 1] = px[n * 3 + 1]; px[i * 3 + 2] = px[n * 3 + 2];
          vel[i * 3] = vel[n * 3]; vel[i * 3 + 1] = vel[n * 3 + 1]; vel[i * 3 + 2] = vel[n * 3 + 2];
          age[i] = age[n]; life[i] = life[n]; size[i] = size[n]; seedA[i] = seedA[n];
        }
        i--;
        continue;
      }
      vel[i * 3 + 1] -= g * dt;
      vel[i * 3] += (wind[0] - vel[i * 3]) * k;
      vel[i * 3 + 1] += (wind[1] - vel[i * 3 + 1]) * k;
      vel[i * 3 + 2] += (wind[2] - vel[i * 3 + 2]) * k;
      px[i * 3] += vel[i * 3] * dt; px[i * 3 + 1] += vel[i * 3 + 1] * dt; px[i * 3 + 2] += vel[i * 3 + 2] * dt;
      lifeN[i] = 1 - age[i] / life[i];
    }
    geo.setDrawRange(0, n);
    for (const [att, sz] of [[posA, 3], [sizeA, 1], [lifeA, 1], [seedAtt, 1]]) {
      att.clearUpdateRanges();
      att.addUpdateRange(0, n * sz);
      att.needsUpdate = true;
    }
  }

  // Point sizes in metres need the projection scale: viewport height / (2 tan(fov/2)).
  function setCamera(camera, viewportHeightPx) {
    const f = (camera.fov * Math.PI) / 360;
    material.uniforms.uScale.value = viewportHeightPx / (2 * Math.tan(f));
  }

  function clear() { n = 0; geo.setDrawRange(0, 0); }
  function dispose() { geo.dispose(); material.dispose(); }

  return {
    points, geometry: geo, material, emit, step, setCamera, clear, dispose, options: o,
    get count() { return n; },
    positions: px, velocities: vel, ages: age, lives: life,
    setWind: (x, y, z) => { wind[0] = x; wind[1] = y; wind[2] = z; },
  };
}
