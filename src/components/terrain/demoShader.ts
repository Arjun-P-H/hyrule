/**
 * W(x,y) — WebGL2 / GLSL ES 3.00 raymarcher.
 * ---------------------------------------------------------------------------
 * EXACT port of src/lib/terrainMath.ts (the double-precision CPU twin).
 * Parity rules (spec §27, worklog constants lock):
 *   - Hashing: strict uint32 integer math (Math.imul → GLSL uint mult).
 *   - All constants mirror src/lib/constants.ts verbatim.
 *   - The GPU evaluates the field on (chunk_local + u_chunk_origin) in fp32;
 *     the CPU twin evaluates globals in fp64. The dyadic frequency table
 *     (n / 2^20) keeps origin·f exactly representable, so lattice cells align
 *     across chunk borders; residual divergence (< 1 mm on a 700 m peak) is
 *     absorbed by GROUND_BIAS = 0.08 in the walk-mode physics twin.
 *   - Output transform: gamma 1/2.2 for sRGB monitor calibration.
 *
 * Coordinate convention: y-up. Terrain field is E(p.x, p.z). Spec z ↔ engine y.
 * Camera basis is LEFT-handed (x=right, z=forward) to match the CPU twin's
 * yaw convention: forward = (sin yaw, 0, cos yaw), right = (cos yaw, 0, −sin yaw).
 */

export const TERRAIN_VS = /* glsl */ `#version 300 es
void main() {
  // Fullscreen triangle from gl_VertexID — no vertex buffers.
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const TERRAIN_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;

// ---- per-frame uniforms ---------------------------------------------------
uniform vec2  u_res;          // render-target size (px)
uniform float u_time;         // seconds (water ripple / cloud drift only)
uniform vec3  u_cam_pos;      // camera, chunk-local fp32
uniform vec2  u_chunk_origin; // multiples of CHUNK_SIZE (fp32-exact)
uniform mat3  u_basis;        // view→world rotation, columns (right, up, forward)
uniform float u_tan_half;     // tan(FOV/2)
uniform int   u_seed;
uniform int   u_march_steps;  // quality preset (spec §26)
uniform int   u_shadow_steps;
uniform float u_footprint;    // 2·tan(FOV/2)/Res_y  (Nyquist Δp per unit t)

out vec4 fragColor;

// ---- contract constants (MIRROR src/lib/constants.ts — audit together) ----
const float MAXH      = 1300.0;              // early-exit ceiling (§26)
const float MAXDIST   = 6000.0;              // march cutoff (§16)
const float F_MACRO   = 512.0  / 1048576.0;  // 1/2048
const float F_RIDGE   = 341.0  / 1048576.0;  // ~1/3074.6
const float F_MASK    = 128.0  / 1048576.0;  // 1/8192
const float F_WARP    = 655.0  / 1048576.0;  // ~1/1600.9
const float F_VALLEY  = 1942.0 / 1048576.0;  // ~1/539.9
const float F_CONT    = 64.0   / 1048576.0;  // 1/16384
const float F_MOIST   = 512.0  / 1048576.0;  // 1/2048
const float F_DUNE    = 23.0   / 1048576.0;  // dune phase wavelength ~45.6 m
const float F_CLOUD   = 403.0  / 1048576.0;  // ~1/2601.9
const float VEG_WL    = 14.0;                // vegetation clump wavelength (m)

const int   OCT_MACRO = 9;
const float LAC_MACRO = 2.0,  GAIN_MACRO = 0.5,  H_BASE  = 90.0;
const int   OCT_RIDGE = 5;
const float LAC_RIDGE = 2.17, GAIN_RIDGE = 0.48, H_MOUNT = 880.0;
const float MASK_LO = 0.28,  MASK_HI = 0.60;
const int   OCT_WARP = 2;
const float S_WARP   = 620.0;
const float C_DEPTH  = 48.0,  W_VALLEY = 0.42, W_LEVEL = 5.0, W_WIDTH = 0.18;
const float CONT_LO  = 0.22,  CONT_HI  = 0.44, CONT_MIN = 0.2;
const float REGION_CELL = 8192.0;
const float REG_BLEND_LO = 0.04, REG_BLEND_HI = 0.18;
const float REG_HILLS = 0.45, REG_MTN = 1.3, REG_DUNE = 0.85, DUNE_H = 42.0;
const float ANOM_CELL = 2048.0, P_ANOM = 0.06;
const float ANOM_R_IN = 380.0,  ANOM_R_OUT = 620.0;
const float TERRACE_STEP = 70.0, TERRACE_H = 15.0;
const float SPIRE_H = 250.0, SPIRE_R = 150.0;
const float SPIRAL_AMP = 16.0, SPIRAL_TURNS = 6.0, SPIRAL_RAD = 0.05;
const float T_BASE = 14.0, L_LAT = 1.2e-5, L_ALT = 0.02;
const float M_BASE = 0.5,  M_AMP = 0.40, M_OCEAN = 0.35;

const float LAMBDA  = 0.8;    // Lipschitz step relax (§16)
const float DMIN    = 0.05;   // minimum step (§16)
const float GROWTH  = 0.0015; // adaptive step growth (§26)
const float START_T = 0.4;
const int   BISECT  = 6;

const vec3  SUN      = vec3(0.550744, 0.420568, 0.720974); // normalize(0.55,0.42,0.72)
const vec3  SUN_COL  = vec3(1.0, 0.955, 0.885) * 3.4;
const vec3  SKY_ZEN  = vec3(0.20, 0.36, 0.52);
const vec3  SKY_HOR  = vec3(0.78, 0.80, 0.82);
const vec3  SKY_TINT = (SKY_HOR + SKY_ZEN) * 0.5;
const vec3  SKY_HOR_104 = SKY_HOR * 1.04;
const vec3  SUN_COL_010 = SUN_COL * 0.10;
const vec3  SUN_COL_030 = SUN_COL * 0.30;
const vec3  SUN_COL_600 = SUN_COL * 6.0;
const float CLOUD_PLANE = 1500.0;
const int   CLOUD_OCT = 4;
const float SIGMA_FOG = 9.0e-4;   // 1/m extinction at sea level (§19)
const float FOG_H     = 900.0;    // fog scale height (§19)

// ---------------------------------------------------------------------------
// §4 — deterministic hashing (uint32, Murmur3 finalizer) — mirrors terrainMath
// ---------------------------------------------------------------------------
uint hashU(uint x) {
  x ^= x >> 16; x *= 0x85ebca6bu;
  x ^= x >> 13; x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}
uint hash3(int xi, int yi, int seed) {
  // h = xi ⊕ (yi·0x27D4EB2F) ⊕ (seed·0x9E3779B9), all uint32-wrapping
  uint h = uint(xi) ^ (uint(yi) * 0x27d4eb2fu) ^ (uint(seed) * 0x9e3779b9u);
  return hashU(h);
}
float hash01(int xi, int yi, int seed) {
  // GPU note: float(uint) rounds beyond 2^24 — quantisation ≤ 6e-8, absorbed
  // by GROUND_BIAS (see header). CPU twin divides in double.
  return float(hash3(xi, yi, seed)) * (1.0 / 4294967295.0);
}

// ---------------------------------------------------------------------------
// §5 — quintic value noise (C²) + Nyquist weights
// ---------------------------------------------------------------------------
float quintic(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

const float INV_U32 = 1.0 / 4294967295.0;
const float INV_U32_X2 = 2.0 / 4294967295.0;

float vnoise(float x, float y, int seed) {
  vec2 p = vec2(x, y);
  vec2 ip = floor(p);
  vec2 f = p - ip;
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  uint ux0 = uint(int(ip.x));
  uint ux1 = ux0 + 1u;
  uint uy0 = uint(int(ip.y)) * 0x27d4eb2fu;
  uint uy1 = uy0 + 0x27d4eb2fu;
  uint s_term = uint(seed) * 0x9e3779b9u;

  uint h0 = uy0 ^ s_term;
  uint h1 = uy1 ^ s_term;

  float a = float(hashU(ux0 ^ h0)) * INV_U32_X2;
  float b = float(hashU(ux1 ^ h0)) * INV_U32_X2;
  float c = float(hashU(ux0 ^ h1)) * INV_U32_X2;
  float d = float(hashU(ux1 ^ h1)) * INV_U32_X2;

  vec2 ab_cd = mix(vec2(a, c), vec2(b, d), u.x);
  return mix(ab_cd.x, ab_cd.y, u.y) - 1.0;
}

float nyqW(float f, float dp) {
  // spec §5: fade an octave out between 0.5 and 1 cycle per pixel footprint (2.0 = 1.0 / 0.5)
  return clamp(2.0 - (f * dp) * 2.0, 0.0, 1.0);
}

const float ASUM_MACRO = 1.99609375;
const float ASUM_RIDGE = 0.93703808;
const float ASUM_WARP  = 1.5;
const float ASUM_CLOUD = 1.875;

float fbm(float x, float y, int seed, int oct, float lac, float gain, float dp, float f0, float totalAsum) {
  float freq = 1.0, amp = 1.0, sum = 0.0, wsum = 0.0;
  for (int i = 0; i < 9; i++) {
    if (i >= oct) break;
    float w = nyqW(f0 * freq, dp);
    if (w <= 0.0) break;
    sum  += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
    wsum += amp * w;
    freq *= lac;
    amp  *= gain;
  }
  return sum * (totalAsum / max(wsum, 1e-6)); // amplitude renormalisation (§5)
}

float ridged(float x, float y, int seed, int oct, float lac, float gain, float dp, float f0, float totalAsum) {
  // (1 − |N|)² octaves; ridged features double bandwidth → Nyquist uses 2f (§5n)
  float freq = 1.0, amp = 0.5, sum = 0.0, wsum = 0.0;
  for (int i = 0; i < 5; i++) {
    if (i >= oct) break;
    float w = nyqW(2.0 * f0 * freq, dp);
    if (w <= 0.0) break;
    float n = 1.0 - abs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
    sum  += amp * w * (n * n);
    wsum += amp * w;
    freq *= lac;
    amp  *= gain;
  }
  return sum * (totalAsum / max(wsum, 1e-6));
}

// ---------------------------------------------------------------------------
// §7/§12/§13 — cellular (Worley) fields (orthogonal-first distance pruning)
// ---------------------------------------------------------------------------
const ivec2 VOR_OFFSETS[8] = ivec2[8](
  ivec2( 1,  0), ivec2( 0,  1), ivec2(-1,  0), ivec2( 0, -1),
  ivec2( 1,  1), ivec2(-1,  1), ivec2( 1, -1), ivec2(-1, -1)
);

float voronoiEdge(vec2 p_in, int seed) {
  vec2 ip = floor(p_in);
  vec2 f = p_in - ip;
  int cxi0 = int(ip.x);
  int cyi0 = int(ip.y);
  uint h0 = hash3(cxi0, cyi0, seed);
  vec2 p0 = vec2(float(h0 & 0xfffu), float((h0 >> 12u) & 0xfffu)) * (1.0 / 4095.0);
  vec2 d0 = p0 - f;
  float f1_sq = dot(d0, d0);
  float f2_sq = 64.0;

  for (int k = 0; k < 8; k++) {
    ivec2 off = VOR_OFFSETS[k];
    float minDx = off.x == 1 ? (1.0 - f.x) : (off.x == -1 ? f.x : 0.0);
    float minDy = off.y == 1 ? (1.0 - f.y) : (off.y == -1 ? f.y : 0.0);
    if (minDx * minDx + minDy * minDy >= f2_sq) continue;

    int cxi = cxi0 + off.x;
    int cyi = cyi0 + off.y;
    uint h = hash3(cxi, cyi, seed);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(float(off.x), float(off.y)) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) {
      f2_sq = f1_sq; f1_sq = d2;
    } else if (d2 < f2_sq) {
      f2_sq = d2;
    }
  }
  return sqrt(f2_sq) - sqrt(f1_sq);
}

// ---------------------------------------------------------------------------
// §14 — domain warp
// ---------------------------------------------------------------------------
void domainWarp(float x, float y, int seed, float dp, out vec2 q) {
  float wx = x * F_WARP, wy = y * F_WARP;
  float qx = fbm(wx, wy,         seed + 11, OCT_WARP, 2.0, 0.5, dp, F_WARP, ASUM_WARP);
  float qy = fbm(wx + 5.2, wy + 1.3, seed + 37, OCT_WARP, 2.0, 0.5, dp, F_WARP, ASUM_WARP);
  q = vec2(x + S_WARP * qx, y + S_WARP * qy);
}

// ---------------------------------------------------------------------------
// §12 — geological anomalies (Zonai terraces + spire + spiral)
// ---------------------------------------------------------------------------
float anomalyE(vec2 g, int seed) {
  vec2 p_in = g * (1.0 / ANOM_CELL);
  vec2 ip = floor(p_in);
  vec2 f = p_in - ip;
  int cxi0 = int(ip.x);
  int cyi0 = int(ip.y);
  uint h0 = hash3(cxi0, cyi0, seed + 400);
  vec2 p0 = vec2(float(h0 & 0xfffu), float((h0 >> 12u) & 0xfffu)) * (1.0 / 4095.0);
  vec2 d0 = p0 - f;
  float f1_sq = dot(d0, d0);
  vec2 bc = vec2(float(cxi0), float(cyi0)) + p0;
  float btype = float(h0 >> 24u) * (1.0 / 255.0);

  for (int k = 0; k < 8; k++) {
    ivec2 off = VOR_OFFSETS[k];
    float minDx = off.x == 1 ? (1.0 - f.x) : (off.x == -1 ? f.x : 0.0);
    float minDy = off.y == 1 ? (1.0 - f.y) : (off.y == -1 ? f.y : 0.0);
    if (minDx * minDx + minDy * minDy >= f1_sq) continue;

    int cxi = cxi0 + off.x;
    int cyi = cyi0 + off.y;
    uint h = hash3(cxi, cyi, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(float(off.x), float(off.y)) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) {
      f1_sq = d2;
      bc = vec2(float(cxi), float(cyi)) + p;
      btype = float(h >> 24u) * (1.0 / 255.0);
    }
  }
  if (btype >= P_ANOM) return 0.0;
  vec2 c = bc * ANOM_CELL;
  vec2 delta = g - c;
  float r2 = dot(delta, delta);
  if (r2 >= ANOM_R_OUT * ANOM_R_OUT) return 0.0;
  float r = sqrt(r2);
  float theta = atan(delta.y, delta.x);
  float fall = 1.0 - smoothstep(ANOM_R_IN, ANOM_R_OUT, r);
  float t = r * (1.0 / TERRACE_STEP);
  float ft = fract(t);
  float steps = (floor(t) + smoothstep(0.3, 0.7, ft)) * TERRACE_H;
  float spire = SPIRE_H * exp(-r2 * (1.0 / (SPIRE_R * SPIRE_R)));
  float spiral = SPIRAL_AMP * sin(SPIRAL_TURNS * theta + SPIRAL_RAD * r);
  return fall * (steps + spire + spiral);
}

// ---------------------------------------------------------------------------
// §13 — regional rules
// ---------------------------------------------------------------------------
struct Reg { float wRegion; float mountFactor; float duneAmount; };

Reg regionField(vec2 g, int seed) {
  vec2 p_in = g * (1.0 / REGION_CELL);
  vec2 ip = floor(p_in);
  vec2 f = p_in - ip;
  int cxi0 = int(ip.x);
  int cyi0 = int(ip.y);
  uint h0 = hash3(cxi0, cyi0, seed + 900);
  vec2 p0 = vec2(float(h0 & 0xfffu), float((h0 >> 12u) & 0xfffu)) * (1.0 / 4095.0);
  vec2 d0 = p0 - f;
  float f1_sq = dot(d0, d0);
  float f2_sq = 64.0;
  float btype = float(h0 >> 24u) * (1.0 / 255.0);

  for (int k = 0; k < 8; k++) {
    ivec2 off = VOR_OFFSETS[k];
    float minDx = off.x == 1 ? (1.0 - f.x) : (off.x == -1 ? f.x : 0.0);
    float minDy = off.y == 1 ? (1.0 - f.y) : (off.y == -1 ? f.y : 0.0);
    if (minDx * minDx + minDy * minDy >= f2_sq) continue;

    int cxi = cxi0 + off.x;
    int cyi = cyi0 + off.y;
    uint h = hash3(cxi, cyi, seed + 900);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(float(off.x), float(off.y)) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) {
      f2_sq = f1_sq; f1_sq = d2;
      btype = float(h >> 24u) * (1.0 / 255.0);
    } else if (d2 < f2_sq) {
      f2_sq = d2;
    }
  }
  float edge = sqrt(f2_sq) - sqrt(f1_sq);
  Reg r;
  r.wRegion = smoothstep(REG_BLEND_LO, REG_BLEND_HI, edge);
  r.mountFactor = btype < 0.33 ? REG_HILLS : (btype < 0.66 ? REG_MTN : REG_DUNE);
  r.duneAmount = smoothstep(0.66, 0.8, btype) * r.wRegion;
  return r;
}

float dunesE(float x, float y, int seed, float dp, float amount) {
  if (amount <= 0.0) return 0.0;
  float phase = x * F_DUNE * 6.28318530718
    + 3.0 * fbm(x * F_WARP, y * F_WARP, seed + 301, 2, 2.0, 0.5, dp, F_WARP, ASUM_WARP);
  float c = 1.0 - abs(sin(phase));
  return DUNE_H * amount * c * c * c; // crescent (barchan) profile
}

// ---------------------------------------------------------------------------
// Master elevation field — spec §2 composition (mirror terrainMath.elevation)
// ---------------------------------------------------------------------------
float elevation(vec2 g, float dp, int seed) {
  // g = (local + chunk_origin) fp32 globals; CPU twin uses fp64 globals.
  vec2 wq;
  domainWarp(g.x, g.y, seed, dp, wq);          // §14 warp

  Reg reg = regionField(g, seed);             // §13 on the RAW plane

  float cont = smoothstep(CONT_LO, CONT_HI,    // §9 continent field
    0.5 + 0.5 * vnoise(g.x * F_CONT, g.y * F_CONT, seed + 801));

  float base = H_BASE * (0.5 + 0.5 * fbm(      // §5 macro hills (warped)
    wq.x * F_MACRO, wq.y * F_MACRO, seed + 1,
    OCT_MACRO, LAC_MACRO, GAIN_MACRO, dp, F_MACRO, ASUM_MACRO));

  float maskVal = 0.5 + 0.5 * vnoise(wq.x * F_MASK, wq.y * F_MASK, seed + 77);
  float mask = smoothstep(MASK_LO, MASK_HI, maskVal);
  float hMount = H_MOUNT * (1.0 + (reg.mountFactor - 1.0) * reg.wRegion);

  float ridge = 0.0;
  if (mask > 0.0) {
    ridge = ridged(wq.x * F_RIDGE, wq.y * F_RIDGE, seed + 5,
      OCT_RIDGE, LAC_RIDGE, GAIN_RIDGE, dp, F_RIDGE, ASUM_RIDGE);
  }

  float de = voronoiEdge(wq * F_VALLEY, seed + 21); // §7 carve
  float sv = 1.0 - smoothstep(0.0, W_VALLEY, de);
  float valley = C_DEPTH * sv * sv;

  float dune = dunesE(g.x, g.y, seed, dp, reg.duneAmount); // §13 dunes (raw)
  float anom = anomalyE(g, seed);                         // §12 (raw)

  return base * (CONT_MIN + (1.0 - CONT_MIN) * cont)
       + hMount * mask * ridge
       + dune
       - valley
       + anom;
}

// §17 — analytic normal via central differences at the pixel footprint.
// Note: This must remain a central difference (4 taps) to match the CPU twin
// (Nyquist-consistent normals); the CPU twin queries near-field with ε = 0.25.
vec2 gradient(vec2 g, float eps, float dp, int seed) {
  float e = max(eps, 0.05);
  float inv2e = 0.5 / e;
  float gx = (elevation(g + vec2(e, 0.0), dp, seed) - elevation(g - vec2(e, 0.0), dp, seed)) * inv2e;
  float gz = (elevation(g + vec2(0.0, e), dp, seed) - elevation(g - vec2(0.0, e), dp, seed)) * inv2e;
  return vec2(gx, gz);
}

// ---------------------------------------------------------------------------
// §9 — climate fields
// ---------------------------------------------------------------------------
float temperature(vec2 g, float e, int seed) {
  return T_BASE - L_LAT * abs(g.y) - L_ALT * max(e, 0.0);
}
float moisture(vec2 g, int seed) {
  float n = fbm(g.x * F_MOIST, g.y * F_MOIST, seed + 61, 2, 2.0, 0.5, 0.0, F_MOIST, ASUM_WARP);
  float cont = smoothstep(CONT_LO, CONT_HI,
    0.5 + 0.5 * vnoise(g.x * F_CONT, g.y * F_CONT, seed + 801));
  return M_BASE + M_AMP * n + M_OCEAN * (1.0 - cont);
}

// ---------------------------------------------------------------------------
// §10/§11/§21 — biome RBF anchors in OKLab (mirror constants.ts BIOME_ANCHORS)
// ---------------------------------------------------------------------------
const mat3 OKLAB_TO_LINEAR_MAT = mat3(
  4.0767416621, -1.2684380046, -0.0041960863,
  -3.3077115913, 2.6097574011, -0.7034186147,
  0.2309699292, -0.3413193965, 1.7076147010
);

vec3 oklabToLinear(vec3 lab) {
  float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  return OKLAB_TO_LINEAR_MAT * vec3(l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
}

const vec4 C0_CTR = vec4(-4.0, 0.55, 800.0, 1.0);
const vec4 C0_INV = vec4(1.0 / 4.5, 1.0 / 0.45, 1.0 / 600.0, 1.0 / 1.1);

const vec4 C1_CTR = vec4(9.0, 0.45, 550.0, 1.7);
const vec4 C1_INV = vec4(1.0 / 22.0, 1.0 / 0.55, 1.0 / 600.0, 1.0 / 0.8);

const vec4 C2_CTR = vec4(13.0, 0.72, 90.0, 0.32);
const vec4 C2_INV = vec4(1.0 / 6.5, 1.0 / 0.22, 1.0 / 260.0, 1.0 / 0.5);

const vec4 C3_CTR = vec4(11.0, 0.90, 230.0, 0.45);
const vec4 C3_INV = vec4(1.0 / 7.0, 1.0 / 0.15, 1.0 / 380.0, 1.0 / 0.55);

const vec4 C4_CTR = vec4(16.0, 0.13, 55.0, 0.35);
const vec4 C4_INV = vec4(1.0 / 8.0, 1.0 / 0.17, 1.0 / 320.0, 1.0 / 0.5);

// Single-exp RBF over (T, M, E, slope); colours mix in OKLab, convert once (§21).
// Vectorized 4D SIMD distance evaluations save instruction cycles per shaded pixel!
vec3 biomeColor(float T, float M, float e, float slopeM, float vegNoise, out float vegOut) {
  vec4 clim = vec4(T, M, e, slopeM);

  vec4 v0 = (clim - C0_CTR) * C0_INV;
  float w0 = exp(-dot(v0, v0));

  vec4 v1 = (clim - C1_CTR) * C1_INV;
  float w1 = exp(-dot(v1, v1));

  vec4 v2 = (clim - C2_CTR) * C2_INV;
  float w2 = exp(-dot(v2, v2));

  vec4 v3 = (clim - C3_CTR) * C3_INV;
  float w3 = exp(-dot(v3, v3));

  vec4 v4 = (clim - C4_CTR) * C4_INV;
  float w4 = exp(-dot(v4, v4));

  float wsum = max(w0 + w1 + w2 + w3 + w4, 1e-6);
  float invWsum = 1.0 / wsum;
  vec3 lab = (
    w0 * vec3(0.93,  0.0,    -0.015) +
    w1 * vec3(0.44,  0.004,   0.008) +
    w2 * vec3(0.52, -0.058,   0.048) +
    w3 * vec3(0.32, -0.072,   0.032) +
    w4 * vec3(0.79,  0.012,   0.072)) * invWsum;

  vegOut = (w2 * 0.92 + w3 * 1.0 + w1 * 0.06 + w4 * 0.05) * invWsum;

  // §11 vegetation clumps: canopy darkening modulated at the 14 m wavelength
  float clump = smoothstep(0.2, 0.8, 0.5 + 0.5 * vegNoise);
  vec3 col = oklabToLinear(lab);
  return mix(col, col * 0.52 + vec3(0.002, 0.020, 0.004), vegOut * clump * 0.85);
}

// ---------------------------------------------------------------------------
// §18 — soft shadow ray (4× footprint → free penumbra)
// ---------------------------------------------------------------------------
float softShadow(vec3 p, int seed, float footprint) {
  float res = 1.0;
  float t = 0.6;
  float fp4 = 4.0 * footprint;
  for (int i = 0; i < 40; i++) {
    if (i >= u_shadow_steps) break;
    vec3 q = p + SUN * t;
    if (q.y > MAXH) break;
    float h = q.y - elevation(q.xz + u_chunk_origin, t * fp4, seed); // §18: 4×Δp
    if (h <= 0.0) return 0.0;
    res = min(res, 10.0 * h / t);
    if (res < 0.005) return 0.0;
    t += clamp(h * 0.8, 0.5, 60.0) + t * 0.04;
  }
  return clamp(res, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// §20–§22 — sky, clouds, Beer-Lambert aerial perspective
// ---------------------------------------------------------------------------
vec3 skyColor(vec3 rd) {
  float sd = max(dot(rd, SUN), 0.0);
  float k = pow(max(rd.y, 0.0), 0.55);
  vec3 sky = mix(SKY_HOR, SKY_ZEN, k);
  sky = mix(SKY_HOR_104, sky, smoothstep(-0.08, 0.06, rd.y));
  float sd2 = sd * sd;
  float sd4 = sd2 * sd2;
  float sd8 = sd4 * sd4;
  float sd90 = sd > 0.8 ? pow(sd, 90.0) : 0.0;
  sky += SUN_COL_010 * sd8 + SUN_COL_030 * sd90;
  sky += SUN_COL_600 * smoothstep(0.99955, 0.99985, sd); // disc
  return sky;
}

// Aerial perspective: T = exp(−σ·t·exp(−h/H)) — Beer-Lambert with a
// 900 m scale height (§19). Distant geometry collapses into the horizon.
vec3 applyFog(vec3 col, float t, float h, vec3 rd) {
  float od = SIGMA_FOG * t * exp(-max(h, 0.0) * (1.0 / FOG_H));
  if (od < 0.0005) return col; // near-field early exit: optical depth negligible
  vec3 inscatter = (skyColor(rd) + SKY_HOR) * 0.5;
  if (od > 5.82) return inscatter; // far-field complete fogging (exp(-0.95*od) < 0.004)
  vec3 T = exp(-vec3(0.58, 0.72, 0.95) * od); // per-channel extinction
  if (T.b < 0.004) return inscatter; // far-field complete fogging
  return col * T + inscatter * (1.0 - T);
}

// ---------------------------------------------------------------------------
// §16/§26 — Lipschitz-corrected march + bisection + water plane
// ---------------------------------------------------------------------------
void main() {
  vec2 ndc = (gl_FragCoord.xy / u_res) * 2.0 - 1.0;
  float aspect = u_res.x / u_res.y;
  vec3 dirView = vec3(ndc.x * aspect * u_tan_half, ndc.y * u_tan_half, 1.0);
  vec3 rd = normalize(u_basis * dirView);
  vec3 ro = u_cam_pos;

  // Underwater camera: cheap depth fog (rare gameplay state).
  if (ro.y < W_LEVEL) {
    vec3 deep = vec3(0.012, 0.055, 0.075);
    fragColor = vec4(pow(deep * (0.6 + 0.4 * max(rd.y, 0.0)), vec3(1.0 / 2.2)), 1.0);
    return;
  }

  // §26 early exit: above the ceiling and climbing → pure sky
  if (ro.y > MAXH && rd.y >= 0.0) {
    vec3 col = skyColor(rd);
    fragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
    return;
  }

  // Analytical water-plane intersection distance
  float tWaterPlane = (rd.y < -1e-5 && ro.y > W_LEVEL) ? (W_LEVEL - ro.y) / rd.y : -1.0;
  bool waterChecked = false;

  float t = START_T;
  float tPrev = t;
  float grad = 0.0;
  float tWater = -1.0;
  float eWater = 0.0;
  bool hit = false;
  float tHit = 0.0;

  for (int i = 0; i < 160; i++) {
    if (i >= u_march_steps) break;
    vec3 p = ro + rd * t;

    if (p.y > MAXH && rd.y > 0.0) break;

    float dp = max(t * u_footprint, 1e-4);
    float e = elevation(p.xz + u_chunk_origin, dp, u_seed);

    // Analytical water plane crossing (§8): test at most ONCE when crossing plane
    if (!waterChecked && tWaterPlane > 0.0 && t >= tWaterPlane) {
      waterChecked = true;
      vec2 cw = ro.xz + rd.xz * tWaterPlane + u_chunk_origin;
      float dpw = max(tWaterPlane * u_footprint, 1e-4);
      float ew = elevation(cw, dpw, u_seed);
      if (ew < W_LEVEL - 0.25) {
        tWater = tWaterPlane;
        eWater = ew;
        break; // water is above any later terrain hit — nothing closer exists
      }
    }

    if (p.y < e) { hit = true; break; }

    // Lipschitz ‖∇E‖ via forward differences, refreshed every 2nd sample
    // (§16/§34.1); fast bitwise test (i & 1 == 0).
    // If ray is far above terrain, step is bounded by horizon clamp (anti-tunneling),
    // so gradient evaluation is bypassed, saving 2 full elevations per step!
    if ((i & 1) == 0) {
      if ((p.y - e) <= (t * 0.06 + 2.0) * 4.0) {
        float eps = max(dp, 0.15);
        vec2 g = p.xz + u_chunk_origin;
        float ex = elevation(g + vec2(eps, 0.0), dp, u_seed);
        float ez = elevation(g + vec2(0.0, eps), dp, u_seed);
        grad = length(vec2((ex - e) / eps, (ez - e) / eps));
      } else {
        grad = 0.0;
      }
    }

    tPrev = t;
    float step = max(DMIN, (p.y - e) * inversesqrt(grad * grad + 1.0) * LAMBDA);
    step *= 1.0 + t * GROWTH;            // §26 adaptive growth
    step = min(step, t * 0.06 + 2.0);    // horizon clamp (anti-tunnel)
    t += step;

    if (t > MAXDIST) break;
  }

  if (hit) {
    // --- bisection refinement over [tPrev, t] (§16) ---
    float lo = tPrev, hi = t;
    for (int b = 0; b < BISECT; b++) {
      float tm = 0.5 * (lo + hi);
      vec3 m = ro + rd * tm;
      float gap = m.y - elevation(m.xz + u_chunk_origin, tm * u_footprint, u_seed);
      if (gap < 0.0) hi = tm; else lo = tm;
    }
    tHit = 0.5 * (lo + hi);
  }

  vec3 col;

  if (hit && (tWater < 0.0 || tHit < tWater)) {
    // ---------------- terrain shading ----------------
    vec3 p = ro + rd * tHit;
    vec2 g = p.xz + u_chunk_origin;
    float dp = max(tHit * u_footprint, 1e-4);

    vec2 grad2 = gradient(g, dp, dp, u_seed);   // §17: ε = pixel footprint
    vec3 n = normalize(vec3(-grad2.x, 1.0, -grad2.y));

    float e = p.y; // bisection-converged surface hit point elevation (§16/§30)
    float T = temperature(g, e, u_seed);
    float M = moisture(g, u_seed);

    float vegNoise = vnoise(g.x / VEG_WL, g.y / VEG_WL, u_seed + 171);
    float veg;
    vec3 albedo = biomeColor(T, M, e, length(grad2), vegNoise, veg);

    float ndl = max(dot(n, SUN), 0.0);
    float sh = 0.0;
    if (ndl > 0.0) {
      sh = softShadow(p + n * 0.15, u_seed, u_footprint);
    }

    vec3 sunTerm = albedo * SUN_COL * ndl * sh;
    vec3 ambTerm = albedo * SKY_TINT * (0.22 + 0.28 * max(n.y, 0.0));

    col = sunTerm + ambTerm;
    col = applyFog(col, tHit, p.y, rd);
  } else if (tWater > 0.0) {
    // ---------------- water plane (§8) ----------------
    vec3 p = ro + rd * tWater;
    vec2 g = p.xz + u_chunk_origin;
    float depth = max(W_LEVEL - eWater, 0.0);

    // ripple normal — dynamic surface, deterministic in (x, z, t)
    float r1 = vnoise(g.x / 5.5 + u_time * 0.35, g.y / 5.5 - u_time * 0.22, u_seed + 911);
    float r2 = vnoise(g.x / 9.7 - u_time * 0.20, g.y / 9.7 + u_time * 0.31, u_seed + 913);
    vec3 n = normalize(vec3(-0.22 * r1, 1.0, -0.22 * r2));

    float sh = softShadow(p + vec3(0.0, 0.15, 0.0), u_seed, u_footprint);
    vec3 shallow = vec3(0.110, 0.320, 0.300);
    vec3 deep    = vec3(0.016, 0.075, 0.100);
    vec3 body = mix(shallow, deep, smoothstep(1.5, 14.0, depth))
      * (0.35 + 0.65 * sh) * (0.6 + 0.4 * max(dot(n, SUN), 0.0));

    // shoreline foam band
    float foamN = vnoise(g.x / 3.1, g.y / 3.1, u_seed + 917);
    body += vec3(0.9) * (1.0 - smoothstep(0.05, 0.6, depth)) * (0.25 + 0.35 * foamN);

    vec3 refl = skyColor(reflect(rd, n));
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
    float spec = pow(max(dot(reflect(rd, n), SUN), 0.0), 240.0) * 3.0;

    col = mix(body, refl, fres) + SUN_COL * spec / 3.4;
    col = applyFog(col, tWater, W_LEVEL, rd);
  } else {
    // ---------------- sky + clouds (§20/§22) ----------------
    col = skyColor(rd);
    if (rd.y > 0.02 && ro.y < CLOUD_PLANE) {
      float tc = (CLOUD_PLANE - ro.y) / rd.y;
      vec2 cp = ro.xz + rd.xz * tc + u_chunk_origin
        + vec2(u_time * 6.0, u_time * 2.0); // slow drift
      float dpc = max(tc * u_footprint, 0.02);
      float c = fbm(cp.x * F_CLOUD, cp.y * F_CLOUD, u_seed + 501,
        CLOUD_OCT, 2.0, 0.5, dpc, F_CLOUD, ASUM_CLOUD);
      float cov = smoothstep(0.42, 0.78, 0.5 + 0.5 * c);
      float shade = 0.85 + 0.30 * (1.0 - smoothstep(0.2, 0.9, c));
      vec3 cloudCol = vec3(shade) * (0.62 + 0.38 * max(SUN.y, 0.0));
      float fade = exp(-tc * 6.0e-5) * smoothstep(0.02, 0.12, rd.y);
      col = mix(col, cloudCol, cov * fade);
    }
  }

  // output transform (gamma correction for sRGB display)
  fragColor = vec4(pow(max(col, 0.0), vec3(1.0 / 2.2)), 1.0);
}
`;
