// GLSL sources of the GPU Boussinesq solver (WebGL2 / GLSL ES 3.00, used with
// three.js RawShaderMaterial + glslVersion GLSL3). Every pass is a full-screen
// triangle over an nx x ny render target; the fragment at gl_FragCoord
// (i + 0.5, j + 0.5) is solver cell (i, j) (i alongshore, j cross-shore,
// j = 0 offshore). All state is sampled with texelFetch and explicit clamping
// (non-periodic edges: mirrored wall states, exactly as src/boussinesq.js).
//
// Textures (RGBA32F, nx x ny unless noted):
//   corner  (nx+1) x (ny+1): (z, n, 0, 0)          corner bed elevation, Manning n
//   bathy   (B, d, chiBed, n)                        cell bed, still depth wl - B, bed dispersion factor, Manning
//   slope   (dxs, dys, slopeMag, 0)                  clamped still-depth slopes
//   coef    (chi, al, bx, by)                        dispersion factor and tridiagonal coefficients
//   state   (h, P, Q, F)                             depth, fluxes, breaking
//   aux     (Pt, Qt, 0, 0)                           lagged time derivatives (previous step)
//   recon   4 x (w, h, u, v) at the E, W, N, S faces of each cell
//   flux    (Fh, FP, FQ, 0) east face, (Gh, GP, GQ, 0) north face
//   rhs     (ho, U, V, 0)                            Euler-updated depth and modified momenta
//   pcr     2 x (a, b, c, r)                         tridiagonal systems (rows for P, columns for Q)
//   target  (etaT, PT, QT, 0)                        relaxation target

export const VS = /* glsl */ `
in vec3 position;
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// Shared header: precision, grid uniforms, clamped fetches.
export const COMMON = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int uNx, uNy;
uniform float uDx, uG, uHDry, uEps, uWl, uTheta, uBQ;
const float SQRT2 = 1.41421356237309504880;
ivec2 cellId() { return ivec2(gl_FragCoord.xy); }
int ci(int i) { return clamp(i, 0, uNx - 1); }
int cj(int j) { return clamp(j, 0, uNy - 1); }
vec4 at(sampler2D t, int i, int j) { return texelFetch(t, ivec2(ci(i), cj(j)), 0); }
// cos of a phase with float32 accuracy for any argument: range reduction to [-pi, pi], then
// cos x = 1 - 2 sin^2(x/2) with a Taylor sine (to h^11; truncation < 6e-8 on |h| <= pi/2). The
// built-in cos of some GPUs (SwiftShader) is only accurate to ~1e-4, which the 1e-4 m target
// tolerance cannot afford.
float cosAcc(float ph) {
  float x = ph - 6.28318530717958647692 * floor(ph / 6.28318530717958647692 + 0.5);
  float h = 0.5 * x, h2 = h * h;
  float s = h * (1.0 + h2 * (-1.0 / 6.0 + h2 * (1.0 / 120.0 + h2 * (-1.0 / 5040.0 + h2 * (1.0 / 362880.0 - h2 / 39916800.0)))));
  return 1.0 - 2.0 * s * s;
}
float smoothstepJS(float a, float b, float x) {
  float t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
`;

// Relaxation weights (relaxField in src/boussinesq.js): zone widths in cells.
export const RELAX = /* glsl */ `
uniform float uRelOff, uRelLat, uRelInl;
float relaxS(int i, int j) {
  float s = 0.0, fi = float(i), fj = float(j), ny = float(uNy), nx = float(uNx);
  if (uRelOff > 0.0 && fj < uRelOff) s = max(s, (uRelOff - fj) / uRelOff);
  if (uRelInl > 0.0 && fj >= ny - uRelInl) s = max(s, (fj - (ny - uRelInl) + 1.0) / uRelInl);
  if (uRelLat > 0.0) {
    if (fi < uRelLat) s = max(s, (uRelLat - fi) / uRelLat);
    if (fi >= nx - uRelLat) s = max(s, (fi - (nx - uRelLat) + 1.0) / uRelLat);
  }
  return min(1.0, s);
}
`;

// Semi-implicit Manning friction and the speed cap (stage() in the reference).
export const FRICTION = /* glsl */ `
vec2 friction(float hc, float p, float q, float n, float dt) {
  if (hc <= uHDry) return vec2(0.0);
  float u = p / hc, v = q / hc;
  float sp = length(vec2(u, v));
  float fr = 1.0 + (dt * uG * n * n * sp) / pow(max(hc, uEps), 4.0 / 3.0);
  p /= fr; q /= fr;
  float umax = 4.0 * sqrt(uG * max(hc, 0.01)) + 1.0;
  if (sp / fr > umax) { float s = umax / (sp / fr); p *= s; q *= s; }
  return vec2(p, q);
}
`;

// --- bed pass: corner texture -> (B, d, chiBed, n) and (dxs, dys, slope, 0) ---
export const BED_FS = COMMON + /* glsl */ `
uniform sampler2D uCorner;
uniform vec2 uDispDepth;
uniform float uUseDisp;
layout(location = 0) out vec4 oBathy;
layout(location = 1) out vec4 oSlope;
vec4 corner(int i, int j) { return texelFetch(uCorner, ivec2(clamp(i, 0, uNx), clamp(j, 0, uNy)), 0); }
float cellB(int i, int j) {
  i = ci(i); j = cj(j);
  return 0.25 * (corner(i, j).x + corner(i + 1, j).x + corner(i, j + 1).x + corner(i + 1, j + 1).x);
}
void main() {
  ivec2 c = cellId(); int i = c.x, j = c.y;
  float B = cellB(i, j);
  float n = 0.25 * (corner(i, j).y + corner(i + 1, j).y + corner(i, j + 1).y + corner(i + 1, j + 1).y);
  float d = uWl - B;
  float dl = uWl - cellB(i - 1, j), dr = uWl - cellB(i + 1, j);
  float ds = uWl - cellB(i, j - 1), dn = uWl - cellB(i, j + 1);
  float sx = (dr - dl) / (2.0 * uDx), sy = (dn - ds) / (2.0 * uDx);
  float dxs = clamp(sx, -0.2, 0.2), dys = clamp(sy, -0.2, 0.2);
  float slope = length(vec2(sx, sy));
  float chiBed = uUseDisp > 0.5 ? smoothstepJS(uDispDepth.x, uDispDepth.y, d) * (1.0 - smoothstepJS(0.35, 0.6, slope)) : 0.0;
  if (j == 0 || j == uNy - 1 || i == 0 || i == uNx - 1) chiBed = 0.0;
  oBathy = vec4(B, d, chiBed, n);
  oSlope = vec4(dxs, dys, slope, 0.0);
}
`;

// --- chi, pass 1: max of F over +-ramp cells along x ---
export const MAXX_FS = COMMON + /* glsl */ `
uniform sampler2D uState;
uniform int uRamp;
out vec4 o;
void main() {
  ivec2 c = cellId();
  float fm = 0.0;
  for (int di = -uRamp; di <= uRamp; di++) fm = max(fm, at(uState, c.x + di, c.y).w);
  o = vec4(fm, 0.0, 0.0, 0.0);
}
`;

// --- chi, pass 2: max along y, chi = chiBed (1 - smoothstep(0.2, 0.5, maxF)), tridiagonal coefficients ---
export const COEF_FS = COMMON + /* glsl */ `
uniform sampler2D uMaxX, uBathy, uSlope;
uniform int uRamp;
uniform float uUseBreak;
out vec4 o;
void main() {
  ivec2 c = cellId();
  float fm = 0.0;
  if (uUseBreak > 0.5) for (int dj = -uRamp; dj <= uRamp; dj++) fm = max(fm, at(uMaxX, c.x, c.y + dj).x);
  vec4 b = texelFetch(uBathy, c, 0);
  vec4 s = texelFetch(uSlope, c, 0);
  float chi = b.z * (1.0 - smoothstepJS(0.2, 0.5, fm));
  float dd = max(b.y, 0.0);
  float idx = 1.0 / uDx;
  float al = chi * (uBQ + 1.0 / 3.0) * dd * dd * idx * idx;
  float bx = chi * dd * s.x * idx / 6.0, by = chi * dd * s.y * idx / 6.0;
  o = vec4(chi, al, bx, by);
}
`;

// --- reconstruction: generalised minmod MUSCL of W, P, Q in x and y, hydrostatic face depths, desingularised velocities ---
export const RECON_FS = COMMON + /* glsl */ `
uniform sampler2D uState, uBathy, uCorner;
layout(location = 0) out vec4 oE;
layout(location = 1) out vec4 oW;
layout(location = 2) out vec4 oN;
layout(location = 3) out vec4 oS;
float minmod3(float a, float b, float c) {
  if (a > 0.0 && b > 0.0 && c > 0.0) return min(a, min(b, c));
  if (a < 0.0 && b < 0.0 && c < 0.0) return max(a, max(b, c));
  return 0.0;
}
// (value at the high face seen from the cell, value at the low face)
vec2 recon(float am, float a0, float ap) {
  float sl = minmod3(uTheta * (a0 - am), 0.5 * (ap - am), uTheta * (ap - a0));
  return vec2(a0 + 0.5 * sl, a0 - 0.5 * sl);
}
float vel(float hh, float pp) {
  if (hh <= uHDry) return 0.0;
  float h4 = hh * hh * hh * hh;
  float e4 = uEps * uEps * uEps * uEps;
  return (SQRT2 * hh * pp) / sqrt(h4 + max(h4, e4));
}
float cornerZ(int i, int j) { return texelFetch(uCorner, ivec2(i, j), 0).x; }
void main() {
  ivec2 c = cellId(); int i = c.x, j = c.y;
  vec4 st = texelFetch(uState, c, 0);
  float B = texelFetch(uBathy, c, 0).x;
  float h = st.x;
  if (h <= uHDry) {
    oE = vec4(B, 0.0, 0.0, 0.0); oW = oE; oN = oE; oS = oE;
    return;
  }
  vec4 sW = at(uState, i - 1, j), sE = at(uState, i + 1, j), sS = at(uState, i, j - 1), sN = at(uState, i, j + 1);
  float W0 = B + h;
  float WW = at(uBathy, i - 1, j).x + sW.x, WE = at(uBathy, i + 1, j).x + sE.x;
  float WS = at(uBathy, i, j - 1).x + sS.x, WN = at(uBathy, i, j + 1).x + sN.x;
  vec2 wx = recon(WW, W0, WE), px = recon(sW.y, st.y, sE.y), qx = recon(sW.z, st.z, sE.z);
  vec2 wy = recon(WS, W0, WN), py = recon(sS.y, st.y, sN.y), qy = recon(sS.z, st.z, sN.z);
  // Face beds from the corners: east/west = mean of the two corners of that face.
  float c00 = cornerZ(i, j), c10 = cornerZ(i + 1, j), c01 = cornerZ(i, j + 1), c11 = cornerZ(i + 1, j + 1);
  float Bx = 0.5 * (c10 + c11), BxW = 0.5 * (c00 + c01), By = 0.5 * (c01 + c11), ByS = 0.5 * (c00 + c10);
  float he = max(0.0, wx.x - Bx), hw = max(0.0, wx.y - BxW);
  float hn = max(0.0, wy.x - By), hs = max(0.0, wy.y - ByS);
  oE = vec4(wx.x, he, vel(he, px.x), vel(he, qx.x));
  oW = vec4(wx.y, hw, vel(hw, px.y), vel(hw, qx.y));
  oN = vec4(wy.x, hn, vel(hn, py.x), vel(hn, qy.x));
  oS = vec4(wy.y, hs, vel(hs, py.y), vel(hs, qy.y));
}
`;

// Central-upwind flux (Kurganov & Petrova 2007), n = 0: x face, n = 1: y face.
export const CUFLUX = /* glsl */ `
vec3 cuFlux(float hl, float ul, float vl, float hr, float ur, float vr, int n) {
  float unl = n == 0 ? ul : vl, unr = n == 0 ? ur : vr;
  float cl_ = sqrt(uG * hl), cr_ = sqrt(uG * hr);
  float ap = max(max(unl + cl_, unr + cr_), 0.0), am = min(min(unl - cl_, unr - cr_), 0.0);
  float den = ap - am;
  if (den < 1e-9) return vec3(0.0);
  float ml = hl * unl, mr = hr * unr;
  float pl = hl * ul, pr = hr * ur, ql = hl * vl, qr = hr * vr;
  float k = (ap * am) / den;
  float f1 = (ap * ml - am * mr) / den + k * (hr - hl);
  float f2l = ml * ul + (n == 0 ? 0.5 * uG * hl * hl : 0.0), f2r = mr * ur + (n == 0 ? 0.5 * uG * hr * hr : 0.0);
  float f3l = ml * vl + (n == 1 ? 0.5 * uG * hl * hl : 0.0), f3r = mr * vr + (n == 1 ? 0.5 * uG * hr * hr : 0.0);
  float f2 = (ap * f2l - am * f2r) / den + k * (pr - pl);
  float f3 = (ap * f3l - am * f3r) / den + k * (qr - ql);
  return vec3(f1, f2, f3);
}
`;

// --- fluxes at the east and north faces of each cell (hydrostatic reconstruction; mirrored wall states at the far edges) ---
export const FLUX_FS = COMMON + CUFLUX + /* glsl */ `
uniform sampler2D uRE, uRW, uRN, uRS;
layout(location = 0) out vec4 oF;
layout(location = 1) out vec4 oG;
void main() {
  ivec2 c = cellId(); int i = c.x, j = c.y;
  vec4 E = texelFetch(uRE, c, 0), Nn = texelFetch(uRN, c, 0);
  vec3 f, g;
  if (i == uNx - 1) {
    f = cuFlux(E.y, E.z, E.w, E.y, -E.z, E.w, 0);
  } else {
    vec4 Wr = texelFetch(uRW, ivec2(i + 1, j), 0);
    float bf = max(E.x - E.y, Wr.x - Wr.y);
    float hl = max(0.0, E.x - bf), hr = max(0.0, Wr.x - bf);
    f = cuFlux(hl, E.z, E.w, hr, Wr.z, Wr.w, 0);
  }
  if (j == uNy - 1) {
    g = cuFlux(Nn.y, Nn.z, Nn.w, Nn.y, Nn.z, -Nn.w, 1);
  } else {
    vec4 Sn = texelFetch(uRS, ivec2(i, j + 1), 0);
    float bf = max(Nn.x - Nn.y, Sn.x - Sn.y);
    float hl = max(0.0, Nn.x - bf), hr = max(0.0, Sn.x - bf);
    g = cuFlux(hl, Nn.z, Nn.w, hr, Sn.z, Sn.w, 1);
  }
  oF = vec4(f, 0.0);
  oG = vec4(g, 0.0);
}
`;

// --- right-hand side, Euler update of h and of the modified momenta U = L_x P + dt RP, V = L_y Q + dt RQ ---
export const RHS_FS = COMMON + CUFLUX + /* glsl */ `
uniform sampler2D uState, uBathy, uSlope, uCoef, uRE, uRW, uRN, uRS, uFx, uFy, uAux;
uniform float uDt;
out vec4 o;
float wd(int i, int j) {
  vec4 s = at(uState, i, j);
  return s.x > uHDry ? at(uBathy, i, j).x + s.x : uWl;
}
void main() {
  ivec2 c = cellId(); int i = c.x, j = c.y;
  vec4 st = texelFetch(uState, c, 0);
  vec4 bt = texelFetch(uBathy, c, 0);
  vec4 cf = texelFetch(uCoef, c, 0);
  float hh = st.x, pp = st.y, qq = st.z;
  float idx = 1.0 / uDx, idx2 = idx * idx, idx3 = idx2 * idx;
  vec4 E = texelFetch(uRE, c, 0), Wc = texelFetch(uRW, c, 0), Nc = texelFetch(uRN, c, 0), Sc = texelFetch(uRS, c, 0);
  // Flux divergence; the west and south walls use mirrored states.
  vec4 fe = texelFetch(uFx, c, 0), gn = texelFetch(uFy, c, 0);
  vec4 fw, gs;
  if (i == 0) fw = vec4(cuFlux(Wc.y, -Wc.z, Wc.w, Wc.y, Wc.z, Wc.w, 0), 0.0); else fw = texelFetch(uFx, ivec2(i - 1, j), 0);
  if (j == 0) gs = vec4(cuFlux(Sc.y, Sc.z, -Sc.w, Sc.y, Sc.z, Sc.w, 1), 0.0); else gs = texelFetch(uFy, ivec2(i, j - 1), 0);
  float Rh = -(fe.x - fw.x) * idx - (gn.x - gs.x) * idx;
  float ho = max(0.0, hh + uDt * Rh);
  // L_x P and L_y Q with the current coefficients (identity where chi = 0).
  float al = cf.y, bx = cf.z, by = cf.w;
  float U = (1.0 + 2.0 * al) * pp - (al - bx) * at(uState, i - 1, j).y - (al + bx) * at(uState, i + 1, j).y;
  float V = (1.0 + 2.0 * al) * qq - (al - by) * at(uState, i, j - 1).z - (al + by) * at(uState, i, j + 1).z;
  if (hh <= uHDry) { o = vec4(ho, U, V, 0.0); return; }
  float rp = -(fe.y - fw.y) * idx - (gn.y - gs.y) * idx;
  float rq = -(fe.z - fw.z) * idx - (gn.z - gs.z) * idx;
  // Hydrostatic-reconstruction sources: face corrections and the in-cell bed slope.
  float he = E.y, hw = Wc.y, hn = Nc.y, hs = Sc.y;
  float hsE, hsW, hsN, hsS;
  if (i == uNx - 1) hsE = he; else { vec4 Wr = texelFetch(uRW, ivec2(i + 1, j), 0); float bf = max(E.x - he, Wr.x - Wr.y); hsE = max(0.0, E.x - bf); }
  if (i == 0) hsW = hw; else { vec4 El = texelFetch(uRE, ivec2(i - 1, j), 0); float bf = max(El.x - El.y, Wc.x - hw); hsW = max(0.0, Wc.x - bf); }
  if (j == uNy - 1) hsN = hn; else { vec4 Sn = texelFetch(uRS, ivec2(i, j + 1), 0); float bf = max(Nc.x - hn, Sn.x - Sn.y); hsN = max(0.0, Nc.x - bf); }
  if (j == 0) hsS = hs; else { vec4 Ns = texelFetch(uRN, ivec2(i, j - 1), 0); float bf = max(Ns.x - Ns.y, Sc.x - hs); hsS = max(0.0, Sc.x - bf); }
  rp += -0.5 * uG * ((he * he - hsE * hsE) - (hw * hw - hsW * hsW)) * idx;
  rp += -uG * 0.5 * (he + hw) * ((E.x - he) - (Wc.x - hw)) * idx;
  rq += -0.5 * uG * ((hn * hn - hsN * hsN) - (hs * hs - hsS * hsS)) * idx;
  rq += -uG * 0.5 * (hn + hs) * ((Nc.x - hn) - (Sc.x - hs)) * idx;
  // Dispersive terms (explicit part), only where chi > 0 and at least 2 cells from the edges.
  float ch = cf.x;
  bool inX = i >= 2 && i <= uNx - 3, inY = j >= 2 && j <= uNy - 3;
  if (ch > 0.0 && inX && inY) {
    float dd = max(bt.y, 0.0);
    vec2 sl = texelFetch(uSlope, c, 0).xy;
    float dxs = sl.x, dys = sl.y;
    float w00 = wd(i, j);
    float wE = wd(i + 1, j), wW = wd(i - 1, j), wE2 = wd(i + 2, j), wW2 = wd(i - 2, j);
    float wN = wd(i, j + 1), wS = wd(i, j - 1), wN2 = wd(i, j + 2), wS2 = wd(i, j - 2);
    float wNE = wd(i + 1, j + 1), wNW = wd(i - 1, j + 1), wSE = wd(i + 1, j - 1), wSW = wd(i - 1, j - 1);
    float Wxx = (wE - 2.0 * w00 + wW) * idx2;
    float Wyy = (wN - 2.0 * w00 + wS) * idx2;
    float Wxy = (wNE - wNW - wSE + wSW) * 0.25 * idx2;
    float Wxxx = (wE2 - 2.0 * wE + 2.0 * wW - wW2) * 0.5 * idx3;
    float Wyyy = (wN2 - 2.0 * wN + 2.0 * wS - wS2) * 0.5 * idx3;
    float Wyy_e = (wNE - 2.0 * wE + wSE) * idx2, Wyy_w = (wNW - 2.0 * wW + wSW) * idx2;
    float Wxyy = (Wyy_e - Wyy_w) * 0.5 * idx;
    float Wxx_n = (wNE - 2.0 * wN + wNW) * idx2, Wxx_s = (wSE - 2.0 * wS + wSW) * idx2;
    float Wxxy = (Wxx_n - Wxx_s) * 0.5 * idx;
    float bgd = uBQ * uG * dd;
    float Ex = bgd * dd * dd * (Wxxx + Wxyy) + dd * dxs * (2.0 * bgd * Wxx + bgd * Wyy) + dd * dys * bgd * Wxy;
    float Ey = bgd * dd * dd * (Wyyy + Wxxy) + dd * dys * (2.0 * bgd * Wyy + bgd * Wxx) + dd * dxs * bgd * Wxy;
    // Lagged cross terms from the previous step's time derivatives of the other flux.
    vec2 aNE = at(uAux, i + 1, j + 1).xy, aNW = at(uAux, i - 1, j + 1).xy, aSE = at(uAux, i + 1, j - 1).xy, aSW = at(uAux, i - 1, j - 1).xy;
    vec2 aN = at(uAux, i, j + 1).xy, aS = at(uAux, i, j - 1).xy, aE = at(uAux, i + 1, j).xy, aW = at(uAux, i - 1, j).xy;
    float Qt_xy = (aNE.y - aNW.y - aSE.y + aSW.y) * 0.25 * idx2;
    float Qt_y = (aN.y - aS.y) * 0.5 * idx, Qt_x = (aE.y - aW.y) * 0.5 * idx;
    float Pt_xy = (aNE.x - aNW.x - aSE.x + aSW.x) * 0.25 * idx2;
    float Pt_x = (aE.x - aW.x) * 0.5 * idx, Pt_y = (aN.x - aS.x) * 0.5 * idx;
    float Mx = (uBQ + 1.0 / 3.0) * dd * dd * Qt_xy + (1.0 / 6.0) * dd * dxs * Qt_y + (1.0 / 6.0) * dd * dys * Qt_x;
    float My = (uBQ + 1.0 / 3.0) * dd * dd * Pt_xy + (1.0 / 6.0) * dd * dys * Pt_x + (1.0 / 6.0) * dd * dxs * Pt_y;
    rp += ch * (Ex + Mx);
    rq += ch * (Ey + My);
  }
  o = vec4(ho, U + uDt * rp, V + uDt * rq, 0.0);
}
`;

// --- parallel cyclic reduction, one level for both systems (rows: L_x P = U, columns: L_y Q = V) ---
// Level 0 (uFirst) builds the equations from the coefficient texture and the rhs texture.
export const PCR_FS = COMMON + /* glsl */ `
uniform sampler2D uCoef, uRhs, uX, uY;
uniform int uStride;
uniform float uFirst;
layout(location = 0) out vec4 oX;
layout(location = 1) out vec4 oY;
vec4 eqX(int i, int j) {
  if (uFirst > 0.5) {
    vec4 cf = texelFetch(uCoef, ivec2(i, j), 0);
    float al = cf.y, bx = cf.z;
    float a = i == 0 ? 0.0 : -(al - bx), cc = i == uNx - 1 ? 0.0 : -(al + bx);
    return vec4(a, 1.0 + 2.0 * al, cc, texelFetch(uRhs, ivec2(i, j), 0).y);
  }
  return texelFetch(uX, ivec2(i, j), 0);
}
vec4 eqY(int i, int j) {
  if (uFirst > 0.5) {
    vec4 cf = texelFetch(uCoef, ivec2(i, j), 0);
    float al = cf.y, by = cf.w;
    float a = j == 0 ? 0.0 : -(al - by), cc = j == uNy - 1 ? 0.0 : -(al + by);
    return vec4(a, 1.0 + 2.0 * al, cc, texelFetch(uRhs, ivec2(i, j), 0).z);
  }
  return texelFetch(uY, ivec2(i, j), 0);
}
vec4 reduce(vec4 e, vec4 em, vec4 ep, bool hasM, bool hasP) {
  float k1 = hasM ? e.x / em.y : 0.0, k2 = hasP ? e.z / ep.y : 0.0;
  return vec4(-k1 * em.x, e.y - k1 * em.z - k2 * ep.x, -k2 * ep.z, e.w - k1 * em.w - k2 * ep.w);
}
void main() {
  ivec2 c = cellId(); int i = c.x, j = c.y, s = uStride;
  vec4 ex = eqX(i, j);
  bool hm = i - s >= 0, hp = i + s < uNx;
  vec4 exm = hm ? eqX(i - s, j) : vec4(0.0, 1.0, 0.0, 0.0), exp_ = hp ? eqX(i + s, j) : vec4(0.0, 1.0, 0.0, 0.0);
  oX = reduce(ex, exm, exp_, hm, hp);
  vec4 ey = eqY(i, j);
  bool hmy = j - s >= 0, hpy = j + s < uNy;
  vec4 eym = hmy ? eqY(i, j - s) : vec4(0.0, 1.0, 0.0, 0.0), eyp = hpy ? eqY(i, j + s) : vec4(0.0, 1.0, 0.0, 0.0);
  oY = reduce(ey, eym, eyp, hmy, hpy);
}
`;

// --- end of an Euler stage: P = r/b, Q = r/b, friction, speed cap -> (ho, P, Q, F) ---
export const FINISH_FS = COMMON + FRICTION + /* glsl */ `
uniform sampler2D uState, uRhs, uX, uY, uBathy;
uniform float uDt;
out vec4 o;
void main() {
  ivec2 c = cellId();
  float ho = texelFetch(uRhs, c, 0).x;
  vec4 ex = texelFetch(uX, c, 0), ey = texelFetch(uY, c, 0);
  float n = texelFetch(uBathy, c, 0).w;
  vec2 pq = friction(ho, ex.w / ex.y, ey.w / ey.y, n, uDt);
  o = vec4(ho, pq.x, pq.y, texelFetch(uState, c, 0).w);
}
`;

// --- end of the step: stage 2 result, SSP-RK2 average, relaxation, breaking, lagged derivatives ---
export const FINAL_FS = COMMON + FRICTION + RELAX + /* glsl */ `
uniform sampler2D uState0, uRhs, uX, uY, uBathy, uTarget;
uniform float uDt, uUseBreak, uUseRelax, uBrkMem;
uniform vec4 uBrk; // onsetRatio, onsetSlope, slopeRes, keepSlope
layout(location = 0) out vec4 oState;
layout(location = 1) out vec4 oAux;
void main() {
  ivec2 c = cellId(); int i = c.x, j = c.y;
  vec4 s0 = texelFetch(uState0, c, 0);
  vec4 bt = texelFetch(uBathy, c, 0);
  float B = bt.x, d = bt.y, n = bt.w;
  // stage 2
  float h2 = texelFetch(uRhs, c, 0).x;
  vec4 ex = texelFetch(uX, c, 0), ey = texelFetch(uY, c, 0);
  vec2 pq2 = friction(h2, ex.w / ex.y, ey.w / ey.y, n, uDt);
  // Heun average
  float h = 0.5 * (s0.x + h2), P = 0.5 * (s0.y + pq2.x), Q = 0.5 * (s0.z + pq2.y);
  if (h <= uHDry) { P = 0.0; Q = 0.0; }
  // relaxation toward the target field
  float s = uUseRelax > 0.5 ? relaxS(i, j) : 0.0;
  if (s > 0.0) {
    vec4 tg = texelFetch(uTarget, c, 0);
    float w = (exp(pow(s, 3.5)) - 1.0) / (2.71828182845904523536 - 1.0);
    float hT = max(0.0, uWl + tg.x - B);
    h = (1.0 - w) * h + w * hT;
    if (hT <= uHDry) { P = (1.0 - w) * P; Q = (1.0 - w) * Q; }
    else { P = (1.0 - w) * P + w * tg.y; Q = (1.0 - w) * Q + w * tg.z; }
    if (h <= uHDry) { P = 0.0; Q = 0.0; }
  }
  // breaking
  float F = s0.w;
  if (uUseBreak > 0.5) {
    float f = F * exp(-uDt / uBrkMem);
    float hc = h;
    if (hc > 0.1 && s < 0.5) {
      float eta = hc + B - uWl;
      float etat = (hc - s0.x) / uDt;
      float cg = sqrt(uG * hc);
      float ratio = eta / max(d, 0.3);
      float fo = min(uBrk.y, uBrk.z * hc / uDx);
      if (etat > 0.0 && (ratio > uBrk.x || etat > fo * cg)) f = 1.0;
      else if (etat > uBrk.w * cg && f > 0.05) f = max(f, 0.8);
    } else if (hc <= 0.1) f = 0.0;
    F = f;
  }
  oState = vec4(h, P, Q, F);
  oAux = vec4((P - s0.y) / uDt, (Q - s0.z) / uDt, 0.0, 0.0);
}
`;

// --- target field from the linear refracted wave field of src/waves.js (section 3 of the contract) ---
// Node-grid textures (N x N): uC0..uC3 = (S_2q, A_2q, S_2q+1, A_2q+1), uK0..uK3 = (kx_2q, ky_2q, kx_2q+1, ky_2q+1),
// uAux = (D, tau, 0, 0). The cell centre is mapped to node coordinates, clamped to the grid for A, D, tau, k and the
// phase is extrapolated linearly outside the grid.
export const TARGET_FS = COMMON + /* glsl */ `
uniform sampler2D uC0, uC1, uC2, uC3, uK0, uK1, uK2, uK3, uAux;
uniform float uPh[8], uOm[8];
uniform float uSetOmega, uSetDepth, uT, uML, uMO, uDxs, uDX;
uniform int uN;
out vec4 o;
vec4 bl(sampler2D t, vec2 g) {
  ivec2 i0 = clamp(ivec2(floor(g)), ivec2(0), ivec2(uN - 2));
  vec2 f = g - vec2(i0);
  vec4 a = texelFetch(t, i0, 0), b = texelFetch(t, i0 + ivec2(1, 0), 0);
  vec4 c = texelFetch(t, i0 + ivec2(0, 1), 0), d = texelFetch(t, i0 + ivec2(1, 1), 0);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
void comp(vec2 S_A, vec2 k, int m, vec2 dxy, float env, inout float eta, inout float P, inout float Q) {
  float kk = length(k);
  float ph = S_A.x + dot(k, dxy) + uPh[m];
  float e = env * S_A.y * cosAcc(ph);
  eta += e;
  if (kk > 1e-7) { float cm = uOm[m] / kk; P += e * cm * k.x / kk; Q += e * cm * k.y / kk; }
}
void main() {
  ivec2 c = cellId();
  vec2 xy = vec2((float(c.x) - uML + 0.5) * uDxs, (float(c.y) - uMO + 0.5) * uDxs);
  vec2 g = xy / uDX;
  vec2 gc = clamp(g, vec2(0.0), vec2(float(uN - 1)));
  vec2 dxy = (g - gc) * uDX;
  vec4 aux = bl(uAux, gc);
  float D = aux.x, tau = aux.y;
  float env = D * (1.0 + uSetDepth * cosAcc(uSetOmega * (uT - tau)));
  float eta = 0.0, P = 0.0, Q = 0.0;
  vec4 c0 = bl(uC0, gc), k0 = bl(uK0, gc);
  comp(c0.xy, k0.xy, 0, dxy, env, eta, P, Q); comp(c0.zw, k0.zw, 1, dxy, env, eta, P, Q);
  vec4 c1 = bl(uC1, gc), k1 = bl(uK1, gc);
  comp(c1.xy, k1.xy, 2, dxy, env, eta, P, Q); comp(c1.zw, k1.zw, 3, dxy, env, eta, P, Q);
  vec4 c2 = bl(uC2, gc), k2 = bl(uK2, gc);
  comp(c2.xy, k2.xy, 4, dxy, env, eta, P, Q); comp(c2.zw, k2.zw, 5, dxy, env, eta, P, Q);
  vec4 c3 = bl(uC3, gc), k3 = bl(uK3, gc);
  comp(c3.xy, k3.xy, 6, dxy, env, eta, P, Q); comp(c3.zw, k3.zw, 7, dxy, env, eta, P, Q);
  o = vec4(eta, P, Q, 0.0);
}
`;

// --- plane-wave target for the tests (the reference test's target function) ---
// etaT = ramp 0.5 H cos(k y - w t), PT = 0, QT = c etaT, ramp = min(1, t / tRamp)
export const PLANE_TARGET_FS = COMMON + /* glsl */ `
uniform float uH, uK, uPh0, uC, uT, uTRamp; // uPh0 = (-w t) reduced mod 2 pi on the CPU
out vec4 o;
void main() {
  ivec2 c = cellId();
  float y = (float(c.y) + 0.5) * uDx;
  float ramp = uTRamp > 0.0 ? min(1.0, uT / uTRamp) : 1.0;
  float e = ramp * 0.5 * uH * cosAcc(uK * y + uPh0);
  o = vec4(e, 0.0, uC * e, 0.0);
}
`;

// --- hot start: state := target field, F := 0, Pt = Qt = 0 ---
export const HOT_FS = COMMON + /* glsl */ `
uniform sampler2D uTarget, uBathy;
layout(location = 0) out vec4 oState;
layout(location = 1) out vec4 oAux;
void main() {
  ivec2 c = cellId();
  vec4 tg = texelFetch(uTarget, c, 0);
  float B = texelFetch(uBathy, c, 0).x;
  float h = max(0.0, uWl + tg.x - B);
  float P = h > uHDry ? tg.y : 0.0, Q = h > uHDry ? tg.z : 0.0;
  oState = vec4(h, P, Q, 0.0);
  oAux = vec4(0.0);
}
`;

// --- load a state texture (tests): copy (h, P, Q, F), clear the lagged derivatives ---
export const LOAD_FS = COMMON + /* glsl */ `
uniform sampler2D uSrc;
layout(location = 0) out vec4 oState;
layout(location = 1) out vec4 oAux;
void main() {
  oState = texelFetch(uSrc, cellId(), 0);
  oAux = vec4(0.0);
}
`;

// --- plain copy of one texture (readback of MRT attachments) ---
export const COPY_FS = COMMON + /* glsl */ `
uniform sampler2D uSrc;
out vec4 o;
void main() { o = texelFetch(uSrc, cellId(), 0); }
`;

// --- bathymetry update: keep the water surface where the cell stays wet, flood dry land below the new still water level ---
export const REBASE_FS = COMMON + /* glsl */ `
uniform sampler2D uState, uAux, uBathyOld, uBathyNew;
layout(location = 0) out vec4 oState;
layout(location = 1) out vec4 oAux;
void main() {
  ivec2 c = cellId();
  vec4 s = texelFetch(uState, c, 0);
  float Bo = texelFetch(uBathyOld, c, 0).x, Bn = texelFetch(uBathyNew, c, 0).x;
  float h = s.x > uHDry ? max(0.0, Bo + s.x - Bn) : max(0.0, uWl - Bn);
  vec4 a = texelFetch(uAux, c, 0);
  if (h <= uHDry) { s.y = 0.0; s.z = 0.0; a = vec4(0.0); }
  oState = vec4(h, s.y, s.z, s.w);
  oAux = a;
}
`;

// --- crest scan (section 5): output V x 2K; row = slot (part 0) or K + slot (part 1) ---
export const CREST_FS = COMMON + /* glsl */ `
uniform sampler2D uState, uBathy;
uniform int uML, uMO, uV, uK; // uV = number of visible rows walked
uniform float uHWet, uMinH;
out vec4 o;
void main() {
  ivec2 c = cellId();
  int col = c.x, slot = c.y % uK, part = c.y / uK;
  int I = uML + col;
  int J0 = uMO, J1 = uMO + uV;
  // state of the walk
  vec4 sPrev2 = at(uState, I, J0 - 2), sPrev = at(uState, I, J0 - 1);
  float ePrev2 = sPrev2.x > uHWet ? sPrev2.x + at(uBathy, I, J0 - 2).x - uWl : -1e9;
  float ePrev = sPrev.x > uHWet ? sPrev.x + at(uBathy, I, J0 - 1).x - uWl : -1e9;
  bool wetPrev = sPrev.x > uHWet;
  bool risingPrev = wetPrev && ePrev > ePrev2;
  bool have = false;
  int jc = -1, jS = -1, count = 0;
  float etaC = 0.0, hC = 0.0, FC = 0.0, uC = 0.0, minE = 0.0, maxS = 0.0;
  int jPrev = J0 - 1;
  for (int J = J0; J <= J1; J++) {
    bool last = J == J1;
    vec4 s = last ? vec4(0.0) : at(uState, I, J);
    bool wet = !last && s.x > uHWet;
    float e = wet ? s.x + at(uBathy, I, J).x - uWl : -1e9;
    bool rising = wet && e > ePrev;
    if (have) {
      if (wet) {
        minE = min(minE, e);
        float sl = (ePrev - e) / uDx;
        if (sl > maxS) { maxS = sl; jS = jPrev; }
      }
      if (!wet || rising) {
        if (etaC - minE >= uMinH) {
          count++;
          if (count == slot + 1) {
            o = part == 0 ? vec4(float(jc), etaC, hC, FC) : vec4(minE, float(jS), maxS, uC);
            return;
          }
        }
        have = false;
      }
    }
    if (!have && wetPrev && risingPrev && !rising) {
      have = true; jc = jPrev; etaC = ePrev;
      vec4 sc = at(uState, I, jc);
      hC = sc.x; FC = sc.w; uC = sc.z / max(sc.x, 1e-6);
      minE = wet ? e : ePrev; maxS = wet ? (ePrev - e) / uDx : 0.0; jS = jPrev;
    }
    ePrev = e; risingPrev = rising; wetPrev = wet; jPrev = J;
  }
  o = part == 0 ? vec4(-1.0, 0.0, 0.0, 0.0) : vec4(0.0);
}
`;

// --- max speed reduction: 4x4 block max; mode 0 reads |u| + sqrt(g h) from the state, mode 1 passes a reduced texture through ---
export const MAXRED_FS = COMMON + /* glsl */ `
uniform sampler2D uSrc;
uniform int uMode, uSw, uSh;
out vec4 o;
float val(int i, int j) {
  if (i >= uSw || j >= uSh) return 0.0;
  vec4 s = texelFetch(uSrc, ivec2(i, j), 0);
  if (uMode == 1) return s.x;
  return s.x <= uHDry ? 0.0 : length(s.yz) / s.x + sqrt(uG * s.x);
}
void main() {
  ivec2 c = cellId() * 4;
  float m = 0.0;
  for (int dj = 0; dj < 4; dj++) for (int di = 0; di < 4; di++) m = max(m, val(c.x + di, c.y + dj));
  o = vec4(m, 0.0, 0.0, 0.0);
}
`;
