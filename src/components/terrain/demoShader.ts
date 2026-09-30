/**
 * W(x,y) — WebGL2 / GLSL ES 3.00 raymarcher.
 * ---------------------------------------------------------------------------
 * EXACT port of src/lib/terrainMath.ts (the double-precision CPU twin).
 * Parity rules (spec §27, worklog constants lock):
 *   - Hashing: strict uint32 integer math (Math.imul → GLSL uint mult).
 *   - All constants mirror src/lib/constants.ts verbatim.
 *   - The GPU evaluates the field on the fp32 global coordinate
 *     (chunk_local + u_chunk_origin); the CPU twin evaluates fp64 globals.
 *     Rebasing is seamless because both floating origins round a world point
 *     to (nearly) the same fp32 value: GPU readback 0.000 mm at 8.6 / 98 /
 *     1000 km, 0.14 mm at the boundary through the world origin. Precision
 *     therefore follows distance from the origin: 0.6 mm at 3 km, 15 mm at
 *     108 km, 99 mm at 534 km (bench/baseline/parity.json, GROUND_BIAS).
 *   - Restructurings that leave E(x,z) bit-identical (octave unrolling,
 *     vectorized Worley, anomaly pruning) need no CPU mirror; any change to
 *     the value of E does (tests/mathCore.test.ts pins the shared constants).
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

// ---- third-person hero (packed by src/lib/hero/rig.ts) ----------------------
// Read only by the withHero() program (third person); first person and fly
// compile without them. Camera-relative, so the body keeps full fp32
// precision anywhere in the world. u_hero_bound.w == 0 disables the hero.
uniform vec4  u_hero_bound;   // bounding sphere: centre (camera-relative), radius
uniform vec3  u_hero_j[18];   // joints (rig.ts J.*), camera-relative
uniform mat3  u_hero_torso;   // torso frame, columns right / up / forward
uniform mat3  u_hero_head;    // head frame, columns right / up / forward
uniform vec4  u_hero_misc;    // x glider open 0..1, y ground under the hero (camera-relative), zw sin/cos facing

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
const float BISECT_TOL = 0.001;
// Horizon clamp (§26, MARCH.clamp* in constants.ts): tight near the surface,
// wider for samples comfortably above it where the clamp alone paces the march.
const float CLAMP_SLOPE = 0.06, CLAMP_SLOPE_FAR = 0.12, CLAMP_BASE = 2.0;
const float GRAD_GATE = 2.5,    GRAD_EPS_MIN = 0.15;
const int   GRAD_CADENCE = 3;
// Soft shadows (§18, SHADOW in constants.ts)
const float SH_START = 0.6, SH_FOOT = 4.0, SH_K = 10.0, SH_CUTOFF = 0.005;
const float SH_STEP_SCALE = 0.8, SH_STEP_MIN = 0.5, SH_STEP_MAX = 60.0, SH_GROWTH = 0.04;

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
// Hero extents (HERO_SHAPE in constants.ts): every primitive of heroMap() lies
// within HERO_LIMB_PAD of its limb joints, HERO_HEAD_REACH of the head joint,
// HERO_BACK_REACH of the pelvis / chest joints and HERO_GLIDER_REACH × open of
// the glider joint; the CPU builds u_hero_bound from the same numbers.
const float HERO_LIMB_PAD = 0.09, HERO_HEAD_REACH = 0.46, HERO_BACK_REACH = 0.52, HERO_GLIDER_REACH = 1.3;
const int   HERO_STEPS = 64;
// Hero palette (linear, pre-exposed for SUN_COL)
const vec3  HC_SKIN    = vec3(0.38, 0.23, 0.15);
const vec3  HC_TUNIC   = vec3(0.028, 0.115, 0.022);
const vec3  HC_CAP     = vec3(0.024, 0.10, 0.019);
const vec3  HC_HAIR    = vec3(0.40, 0.25, 0.055);
const vec3  HC_LEATHER = vec3(0.095, 0.045, 0.018);
const vec3  HC_CREAM   = vec3(0.33, 0.30, 0.23);
const vec3  HC_GOLD    = vec3(0.45, 0.30, 0.05);
const vec3  HC_SILVER  = vec3(0.34, 0.35, 0.37);
const vec3  HC_SHIELD  = vec3(0.014, 0.038, 0.17);
const vec3  HC_RED     = vec3(0.30, 0.022, 0.015);
const vec3  HC_HILT    = vec3(0.045, 0.032, 0.19);
const vec3  HC_SHEATH  = vec3(0.018, 0.025, 0.07);
const vec3  HC_GLIDER  = vec3(0.28, 0.05, 0.02);
const vec3  HC_GLIDER2 = vec3(0.40, 0.31, 0.19);
const vec3  HC_WOOD    = vec3(0.14, 0.075, 0.035);
const vec3  HC_EYE     = vec3(0.015, 0.05, 0.075);

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

  uvec4 v = uvec4(ux0 ^ h0, ux1 ^ h0, ux0 ^ h1, ux1 ^ h1);
  v ^= v >> 16u; v *= 0x85ebca6bu;
  v ^= v >> 13u; v *= 0xc2b2ae35u;
  v ^= v >> 16u;
  vec4 abcd = vec4(v) * INV_U32_X2;

  vec2 ab_cd = mix(abcd.xz, abcd.yw, u.x);
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

const float MACRO_FREQ[9] = float[9](1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0);
const float MACRO_AMP[9]  = float[9](1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625);
const float RIDGE_FREQ[5] = float[5](1.0, 2.17, 4.7089, 10.218313, 22.17373921);
const float RIDGE_AMP[5]  = float[5](0.5, 0.24, 0.1152, 0.055296, 0.02654208);
const float WARP_FREQ[2]  = float[2](1.0, 2.0);
const float WARP_AMP[2]   = float[2](1.0, 0.5);

float fbm(float x, float y, int seed, int oct, float lac, float gain, float dp, float f0, float totalAsum) {
  float k = 2.0 * f0 * dp;
  if (oct == 9 && lac == 2.0 && gain == 0.5) {
    if (k <= 0.0) {
      return vnoise(x, y, seed)
        + 0.5 * vnoise(x * 2.0, y * 2.0, seed + 40503)
        + 0.25 * vnoise(x * 4.0, y * 4.0, seed + 81006)
        + 0.125 * vnoise(x * 8.0, y * 8.0, seed + 121509)
        + 0.0625 * vnoise(x * 16.0, y * 16.0, seed + 162012)
        + 0.03125 * vnoise(x * 32.0, y * 32.0, seed + 202515)
        + 0.015625 * vnoise(x * 64.0, y * 64.0, seed + 243018)
        + 0.0078125 * vnoise(x * 128.0, y * 128.0, seed + 283521)
        + 0.00390625 * vnoise(x * 256.0, y * 256.0, seed + 324024);
    }
    float w0 = clamp(2.0 - k, 0.0, 1.0);
    if (w0 <= 0.0) return 0.0;
    float sum = w0 * vnoise(x, y, seed);
    float wsum = w0;

    float w1 = clamp(2.0 - 2.0 * k, 0.0, 1.0);
    if (w1 > 0.0) {
      sum += 0.5 * w1 * vnoise(x * 2.0, y * 2.0, seed + 40503);
      wsum += 0.5 * w1;
      float w2 = clamp(2.0 - 4.0 * k, 0.0, 1.0);
      if (w2 > 0.0) {
        sum += 0.25 * w2 * vnoise(x * 4.0, y * 4.0, seed + 81006);
        wsum += 0.25 * w2;
        float w3 = clamp(2.0 - 8.0 * k, 0.0, 1.0);
        if (w3 > 0.0) {
          sum += 0.125 * w3 * vnoise(x * 8.0, y * 8.0, seed + 121509);
          wsum += 0.125 * w3;
          float w4 = clamp(2.0 - 16.0 * k, 0.0, 1.0);
          if (w4 > 0.0) {
            sum += 0.0625 * w4 * vnoise(x * 16.0, y * 16.0, seed + 162012);
            wsum += 0.0625 * w4;
            float w5 = clamp(2.0 - 32.0 * k, 0.0, 1.0);
            if (w5 > 0.0) {
              sum += 0.03125 * w5 * vnoise(x * 32.0, y * 32.0, seed + 202515);
              wsum += 0.03125 * w5;
              float w6 = clamp(2.0 - 64.0 * k, 0.0, 1.0);
              if (w6 > 0.0) {
                sum += 0.015625 * w6 * vnoise(x * 64.0, y * 64.0, seed + 243018);
                wsum += 0.015625 * w6;
                float w7 = clamp(2.0 - 128.0 * k, 0.0, 1.0);
                if (w7 > 0.0) {
                  sum += 0.0078125 * w7 * vnoise(x * 128.0, y * 128.0, seed + 283521);
                  wsum += 0.0078125 * w7;
                  float w8 = clamp(2.0 - 256.0 * k, 0.0, 1.0);
                  if (w8 > 0.0) {
                    sum += 0.00390625 * w8 * vnoise(x * 256.0, y * 256.0, seed + 324024);
                    wsum += 0.00390625 * w8;
                  }
                }
              }
            }
          }
        }
      }
    }
    return sum * (totalAsum / max(wsum, 1e-6));
  }
  if (oct == 2 && lac == 2.0 && gain == 0.5) {
    if (k <= 0.0) {
      return vnoise(x, y, seed) + 0.5 * vnoise(x * 2.0, y * 2.0, seed + 40503);
    }
    float w0 = clamp(2.0 - k, 0.0, 1.0);
    if (w0 <= 0.0) return 0.0;
    float sum = w0 * vnoise(x, y, seed);
    float wsum = w0;
    float w1 = clamp(2.0 - 2.0 * k, 0.0, 1.0);
    if (w1 > 0.0) {
      sum += 0.5 * w1 * vnoise(x * 2.0, y * 2.0, seed + 40503);
      wsum += 0.5 * w1;
    }
    return sum * (totalAsum / max(wsum, 1e-6));
  }
  float freq = 1.0, amp = 1.0, sum = 0.0, wsum = 0.0;
  for (int i = 0; i < 9; i++) {
    if (i >= oct) break;
    float w = clamp(2.0 - freq * k, 0.0, 1.0);
    if (w <= 0.0) break;
    sum  += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
    wsum += amp * w;
    freq *= lac;
    amp  *= gain;
  }
  return sum * (totalAsum / max(wsum, 1e-6));
}

float ridged(float x, float y, int seed, int oct, float lac, float gain, float dp, float f0, float totalAsum) {
  float k = 4.0 * f0 * dp;
  if (oct == 5 && lac == 2.17 && gain == 0.48) {
    if (k <= 0.0) {
      float n0 = 1.0 - abs(vnoise(x, y, seed));
      float n1 = 1.0 - abs(vnoise(x * 2.17, y * 2.17, seed + 34283));
      float n2 = 1.0 - abs(vnoise(x * 4.7089, y * 4.7089, seed + 68566));
      float n3 = 1.0 - abs(vnoise(x * 10.218313, y * 10.218313, seed + 102849));
      float n4 = 1.0 - abs(vnoise(x * 22.17373921, y * 22.17373921, seed + 137132));
      return 0.5 * (n0 * n0) + 0.24 * (n1 * n1) + 0.1152 * (n2 * n2) + 0.055296 * (n3 * n3) + 0.02654208 * (n4 * n4);
    }
    float w0 = clamp(2.0 - k, 0.0, 1.0);
    if (w0 <= 0.0) return 0.0;
    float n0 = 1.0 - abs(vnoise(x, y, seed));
    float sum = 0.5 * w0 * (n0 * n0);
    float wsum = 0.5 * w0;

    float w1 = clamp(2.0 - 2.17 * k, 0.0, 1.0);
    if (w1 > 0.0) {
      float n1 = 1.0 - abs(vnoise(x * 2.17, y * 2.17, seed + 34283));
      sum += 0.24 * w1 * (n1 * n1);
      wsum += 0.24 * w1;

      float w2 = clamp(2.0 - 4.7089 * k, 0.0, 1.0);
      if (w2 > 0.0) {
        float n2 = 1.0 - abs(vnoise(x * 4.7089, y * 4.7089, seed + 68566));
        sum += 0.1152 * w2 * (n2 * n2);
        wsum += 0.1152 * w2;

        float w3 = clamp(2.0 - 10.218313 * k, 0.0, 1.0);
        if (w3 > 0.0) {
          float n3 = 1.0 - abs(vnoise(x * 10.218313, y * 10.218313, seed + 102849));
          sum += 0.055296 * w3 * (n3 * n3);
          wsum += 0.055296 * w3;

          float w4 = clamp(2.0 - 22.17373921 * k, 0.0, 1.0);
          if (w4 > 0.0) {
            float n4 = 1.0 - abs(vnoise(x * 22.17373921, y * 22.17373921, seed + 137132));
            sum += 0.02654208 * w4 * (n4 * n4);
            wsum += 0.02654208 * w4;
          }
        }
      }
    }
    return sum * (totalAsum / max(wsum, 1e-6));
  }
  float freq = 1.0, amp = 0.5, sum = 0.0, wsum = 0.0;
  for (int i = 0; i < 5; i++) {
    if (i >= oct) break;
    float w = clamp(2.0 - freq * k, 0.0, 1.0);
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
// §7/§12/§13 — cellular (Worley) fields: centre + 4 orthogonal neighbours hashed as one uvec4,
// diagonals as a second uvec4 only when a corner could hold F1/F2
// ---------------------------------------------------------------------------
float voronoiEdge(vec2 p_in, int seed) {
  vec2 ip = floor(p_in);
  vec2 f = p_in - ip;
  uint s_term = uint(seed) * 0x9e3779b9u;
  uint cy_term = uint(int(ip.y)) * 0x27d4eb2fu;
  uint row0  = cy_term ^ s_term;
  uint rowP1 = (cy_term + 0x27d4eb2fu) ^ s_term;
  uint rowM1 = (cy_term - 0x27d4eb2fu) ^ s_term;

  uint cx0 = uint(int(ip.x));
  uint cxP1 = cx0 + 1u;
  uint cxM1 = cx0 - 1u;

  // Center (0, 0)
  uint h0 = hashU(cx0 ^ row0);
  vec2 p0 = vec2(float(h0 & 0xfffu), float((h0 >> 12u) & 0xfffu)) * (1.0 / 4095.0);
  vec2 d0 = p0 - f;
  float f1_sq = dot(d0, d0);
  float f2_sq = 64.0;

  // 4 Orthogonal neighbors in a single uvec4: (+1, 0), (0, +1), (-1, 0), (0, -1)
  uvec4 h_ortho = uvec4(cxP1 ^ row0, cx0 ^ rowP1, cxM1 ^ row0, cx0 ^ rowM1);
  h_ortho ^= h_ortho >> 16u; h_ortho *= 0x85ebca6bu;
  h_ortho ^= h_ortho >> 13u; h_ortho *= 0xc2b2ae35u;
  h_ortho ^= h_ortho >> 16u;

  vec4 px_o = vec4(h_ortho & 0xfffu) * (1.0 / 4095.0);
  vec4 py_o = vec4((h_ortho >> 12u) & 0xfffu) * (1.0 / 4095.0);
  vec4 dx_o = px_o - f.x + vec4(1.0, 0.0, -1.0, 0.0);
  vec4 dy_o = py_o - f.y + vec4(0.0, 1.0, 0.0, -1.0);
  vec4 d2_o = dx_o * dx_o + dy_o * dy_o;

  for (int k = 0; k < 4; k++) {
    float d2 = d2_o[k];
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; }
    else if (d2 < f2_sq) { f2_sq = d2; }
  }

  // 4 Diagonals: (+1, +1), (-1, +1), (+1, -1), (-1, -1)
  float fx0_2 = f.x * f.x;
  float fx1_2 = (1.0 - f.x) * (1.0 - f.x);
  float fy0_2 = f.y * f.y;
  float fy1_2 = (1.0 - f.y) * (1.0 - f.y);
  float minDiag = min(min(fx1_2 + fy1_2, fx0_2 + fy1_2), min(fx1_2 + fy0_2, fx0_2 + fy0_2));

  if (minDiag < f2_sq) {
    uvec4 h_diag = uvec4(cxP1 ^ rowP1, cxM1 ^ rowP1, cxP1 ^ rowM1, cxM1 ^ rowM1);
    h_diag ^= h_diag >> 16u; h_diag *= 0x85ebca6bu;
    h_diag ^= h_diag >> 13u; h_diag *= 0xc2b2ae35u;
    h_diag ^= h_diag >> 16u;

    vec4 px_d = vec4(h_diag & 0xfffu) * (1.0 / 4095.0);
    vec4 py_d = vec4((h_diag >> 12u) & 0xfffu) * (1.0 / 4095.0);
    vec4 dx_d = px_d - f.x + vec4(1.0, -1.0, 1.0, -1.0);
    vec4 dy_d = py_d - f.y + vec4(1.0, 1.0, -1.0, -1.0);
    vec4 d2_d = dx_d * dx_d + dy_d * dy_d;

    for (int k = 0; k < 4; k++) {
      float d2 = d2_d[k];
      if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; }
      else if (d2 < f2_sq) { f2_sq = d2; }
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

  // A neighbour cell can only change the result if its border lies within
  // ANOM_R_OUT (beyond it the anomaly is 0 whichever feature owns the point)
  // and it could hold a nearer feature than the best so far. Pruning against
  // min(F1², R_out²) visits ~1.5 of 8 neighbours on average, and subsumes the
  // former cell-interior fast path. Mirrored in terrainMath.anomalyE.
  const float R_OUT_CELL_SQ = (ANOM_R_OUT / ANOM_CELL) * (ANOM_R_OUT / ANOM_CELL);
  float fx0_2 = f.x * f.x;
  float fx1_2 = (1.0 - f.x) * (1.0 - f.x);
  float fy0_2 = f.y * f.y;
  float fy1_2 = (1.0 - f.y) * (1.0 - f.y);

  if (fx1_2 < f1_sq && fx1_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0 + 1, cyi0, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(1.0, 0.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0 + 1), float(cyi0)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fy1_2 < f1_sq && fy1_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0, cyi0 + 1, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(0.0, 1.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0), float(cyi0 + 1)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fx0_2 < f1_sq && fx0_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0 - 1, cyi0, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(-1.0, 0.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0 - 1), float(cyi0)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fy0_2 < f1_sq && fy0_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0, cyi0 - 1, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(0.0, -1.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0), float(cyi0 - 1)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fx1_2 + fy1_2 < f1_sq && fx1_2 + fy1_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0 + 1, cyi0 + 1, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(1.0, 1.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0 + 1), float(cyi0 + 1)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fx0_2 + fy1_2 < f1_sq && fx0_2 + fy1_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0 - 1, cyi0 + 1, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(-1.0, 1.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0 - 1), float(cyi0 + 1)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fx1_2 + fy0_2 < f1_sq && fx1_2 + fy0_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0 + 1, cyi0 - 1, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(1.0, -1.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0 + 1), float(cyi0 - 1)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
  }
  if (fx0_2 + fy0_2 < f1_sq && fx0_2 + fy0_2 < R_OUT_CELL_SQ) {
    uint h = hash3(cxi0 - 1, cyi0 - 1, seed + 400);
    vec2 p = vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
    vec2 d = vec2(-1.0, -1.0) + p - f;
    float d2 = dot(d, d);
    if (d2 < f1_sq) { f1_sq = d2; bc = vec2(float(cxi0 - 1), float(cyi0 - 1)) + p; btype = float(h >> 24u) * (1.0 / 255.0); }
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
  uint sTerm = uint(seed + 900) * 0x9e3779b9u;
  uint cy_term = uint(cyi0) * 0x27d4eb2fu;
  uint row0  = cy_term ^ sTerm;
  uint rowP1 = (uint(cyi0 + 1) * 0x27d4eb2fu) ^ sTerm;
  uint rowM1 = (uint(cyi0 - 1) * 0x27d4eb2fu) ^ sTerm;
  uint cx0  = uint(cxi0);
  uint cxP1 = uint(cxi0 + 1);
  uint cxM1 = uint(cxi0 - 1);

  uint h0 = hashU(cx0 ^ row0);
  vec2 p0 = vec2(float(h0 & 0xfffu), float((h0 >> 12u) & 0xfffu)) * (1.0 / 4095.0);
  vec2 d0 = p0 - f;
  float f1_sq = dot(d0, d0);
  float f2_sq = 64.0;
  float btype = float(h0 >> 24u) * (1.0 / 255.0);

  // Vectorized 4 orthogonal neighbors: (+1, 0), (0, +1), (-1, 0), (0, -1)
  uvec4 h_ortho = uvec4(cxP1 ^ row0, cx0 ^ rowP1, cxM1 ^ row0, cx0 ^ rowM1);
  h_ortho ^= h_ortho >> 16u; h_ortho *= 0x85ebca6bu;
  h_ortho ^= h_ortho >> 13u; h_ortho *= 0xc2b2ae35u;
  h_ortho ^= h_ortho >> 16u;

  vec4 px_o = vec4(h_ortho & 0xfffu) * (1.0 / 4095.0);
  vec4 py_o = vec4((h_ortho >> 12u) & 0xfffu) * (1.0 / 4095.0);
  vec4 dx_o = px_o - f.x + vec4(1.0, 0.0, -1.0, 0.0);
  vec4 dy_o = py_o - f.y + vec4(0.0, 1.0, 0.0, -1.0);
  vec4 d2_o = dx_o * dx_o + dy_o * dy_o;

  for (int k = 0; k < 4; k++) {
    float d2 = d2_o[k];
    if (d2 < f1_sq) {
      f2_sq = f1_sq;
      f1_sq = d2;
      btype = float(h_ortho[k] >> 24u) * (1.0 / 255.0);
    } else if (d2 < f2_sq) {
      f2_sq = d2;
    }
  }

  // 4 Diagonals: (+1, +1), (-1, +1), (+1, -1), (-1, -1)
  float fx0_2 = f.x * f.x;
  float fx1_2 = (1.0 - f.x) * (1.0 - f.x);
  float fy0_2 = f.y * f.y;
  float fy1_2 = (1.0 - f.y) * (1.0 - f.y);
  float minDiag = min(min(fx1_2 + fy1_2, fx0_2 + fy1_2), min(fx1_2 + fy0_2, fx0_2 + fy0_2));

  if (minDiag < f2_sq) {
    uvec4 h_diag = uvec4(cxP1 ^ rowP1, cxM1 ^ rowP1, cxP1 ^ rowM1, cxM1 ^ rowM1);
    h_diag ^= h_diag >> 16u; h_diag *= 0x85ebca6bu;
    h_diag ^= h_diag >> 13u; h_diag *= 0xc2b2ae35u;
    h_diag ^= h_diag >> 16u;

    vec4 px_d = vec4(h_diag & 0xfffu) * (1.0 / 4095.0);
    vec4 py_d = vec4((h_diag >> 12u) & 0xfffu) * (1.0 / 4095.0);
    vec4 dx_d = px_d - f.x + vec4(1.0, -1.0, 1.0, -1.0);
    vec4 dy_d = py_d - f.y + vec4(1.0, 1.0, -1.0, -1.0);
    vec4 d2_d = dx_d * dx_d + dy_d * dy_d;

    for (int k = 0; k < 4; k++) {
      float d2 = d2_d[k];
      if (d2 < f1_sq) {
        f2_sq = f1_sq;
        f1_sq = d2;
        btype = float(h_diag[k] >> 24u) * (1.0 / 255.0);
      } else if (d2 < f2_sq) {
        f2_sq = d2;
      }
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

  float dune = reg.duneAmount > 0.0 ? dunesE(g.x, g.y, seed, dp, reg.duneAmount) : 0.0; // §13 dunes (raw)
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
  float t = SH_START;
  float fp4 = SH_FOOT * footprint;
  for (int i = 0; i < 40; i++) {
    if (i >= u_shadow_steps) break;
    vec3 q = p + SUN * t;
    if (q.y > MAXH) break;
    float h = q.y - elevation(q.xz + u_chunk_origin, t * fp4, seed); // §18: 4×Δp
    if (h <= 0.0) return 0.0;
    res = min(res, SH_K * h / t);
    if (res < SH_CUTOFF) return 0.0;
    t += clamp(h * SH_STEP_SCALE, SH_STEP_MIN, SH_STEP_MAX) + t * SH_GROWTH;
  }
  return clamp(res, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// §20–§22 — sky, clouds, Beer-Lambert aerial perspective
// ---------------------------------------------------------------------------
vec3 skyColor(vec3 rd) {
  float sd = max(dot(rd, SUN), 0.0);
  float k = pow(max(rd.y, 0.0), 0.55);
  mediump vec3 sky = mix(SKY_HOR, SKY_ZEN, k);
  sky = mix(SKY_HOR_104, sky, smoothstep(-0.08, 0.06, rd.y));
  float sd2 = sd * sd;
  float sd4 = sd2 * sd2;
  float sd8 = sd4 * sd4;
  float sd90 = 0.0;
  if (sd > 0.8) {
    float sd16 = sd8 * sd8;
    float sd32 = sd16 * sd16;
    float sd64 = sd32 * sd32;
    sd90 = sd64 * sd16 * sd8 * sd2;
  }
  sky += SUN_COL_010 * sd8 + SUN_COL_030 * sd90;
  sky += SUN_COL_600 * smoothstep(0.99955, 0.99985, sd); // disc
  return sky;
}

// Aerial perspective: T = exp(−σ·t·exp(−h/H)) — Beer-Lambert with a
// 900 m scale height (§19). Distant geometry collapses into the horizon.
vec3 applyFog(vec3 col, float t, float h, vec3 rd) {
  float od = SIGMA_FOG * t * exp(-max(h, 0.0) * (1.0 / FOG_H));
  if (od < 0.0005) return col; // near-field early exit: optical depth negligible
  mediump vec3 inscatter = (skyColor(rd) + SKY_HOR) * 0.5;
  if (od > 5.82) return inscatter; // far-field complete fogging (exp(-0.95*od) < 0.004)
  mediump vec3 T = exp(-vec3(0.58, 0.72, 0.95) * od); // per-channel extinction
  if (T.b < 0.004) return inscatter; // far-field complete fogging
  return col * T + inscatter * (1.0 - T);
}

#ifdef HERO
// ---------------------------------------------------------------------------
// Hero — a signed-distance body built from capsules around the rig's joints
// (src/lib/hero/rig.ts), sphere-traced inside its bounding sphere. Material
// ids: 0 skin, 1 tunic, 2 cap, 3 hair, 4 leather, 5 cream, 6 gold, 7 shield,
// 8 hilt, 9 sheath, 10 glider, 11 wood, 12/13 forearm L/R, 14/15 shin L/R.
// ---------------------------------------------------------------------------
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
// Tapered capsule: radius r1 at a → r2 at b (Lipschitz ≈ 1 for gentle tapers).
float sdLimb(vec3 p, vec3 a, vec3 b, float r1, float r2) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - mix(r1, r2, h);
}
float sdBox3(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}
// Capped cone along y: half-height h, radius r1 at −h and r2 at +h.
float sdCone(vec3 p, float h, float r1, float r2) {
  vec2 q = vec2(length(p.xz), p.y);
  vec2 k1 = vec2(r2, h);
  vec2 k2 = vec2(r2 - r1, 2.0 * h);
  vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
  vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
// Equilateral triangle, apex up, half-side r (shield emblem).
float sdTri(vec2 p, float r) {
  const float k = 1.7320508;
  p.x = abs(p.x) - r;
  p.y = p.y + r / k;
  if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) * 0.5;
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
float smin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}
vec2 hU(vec2 a, float d, float id) { return d < a.x ? vec2(d, id) : a; }
// Heater-shield outline in the torso's back plane: an ellipse cut flat at the
// top (0.33 m wide), rounding to its point 0.49 m below.
float heroShieldEdge(vec2 s) {
  return max((length((s - vec2(0.0, 0.25)) / vec2(0.19, 0.33)) - 1.0) * 0.19, s.y - 0.41);
}

// Distance to the hero (x) and the material id of the nearest part (y), at a
// camera-relative point. Groups beyond the current nearest are skipped.
vec2 heroMap(vec3 p) {
  vec3 pel = u_hero_j[0], che = u_hero_j[1], hd = u_hero_j[2];
  // torso, tunic skirt, belt and the gear on the back (torso frame)
  vec3 q = (p - pel) * u_hero_torso;
  float body = smin(sdLimb(p, pel, che, 0.125, 0.135), sdCapsule(p, u_hero_j[3], u_hero_j[6], 0.062), 0.06);
  body = min(body, sdCone(q - vec3(0.0, -0.1, -0.01), 0.16, 0.205, 0.132));
  vec2 r = vec2(body, 1.0);
  r = hU(r, sdCapsule(p, che, hd, 0.05), 0.0);
  vec2 bw = vec2(length(q.xz) - 0.142, abs(q.y - 0.07) - 0.026);
  r = hU(r, min(max(bw.x, bw.y), 0.0) + length(max(bw, 0.0)), 4.0);
  r = hU(r, sdBox3(q - vec3(0.0, 0.07, 0.143), vec3(0.032, 0.026, 0.01)), 6.0);
  r = hU(r, sdCapsule(q, vec3(-0.15, -0.14, -0.175), vec3(0.1, 0.34, -0.175), 0.028), 9.0);
  r = hU(r, sdCapsule(q, vec3(0.1, 0.34, -0.175), vec3(0.155, 0.46, -0.175), 0.018), 8.0);
  r = hU(r, min(sdCapsule(q, vec3(0.038, 0.372, -0.175), vec3(0.162, 0.308, -0.175), 0.016),
                length(q - vec3(0.164, 0.478, -0.175)) - 0.024), 6.0);
  vec2 sw = vec2(heroShieldEdge(q.xy), abs(q.z + 0.222) - 0.015);
  r = hU(r, min(max(sw.x, sw.y), 0.0) + length(max(sw, 0.0)), 7.0);

  // head: face, hair, cap, ears (head frame)
  if (length(p - hd) - HERO_HEAD_REACH < r.x) {
    vec3 h = (p - hd) * u_hero_head;
    float skin = smin(length(h) - 0.112, length(h - vec3(0.0, -0.045, 0.03)) - 0.085, 0.04);
    skin = min(skin, length(h - vec3(0.0, -0.02, 0.11)) - 0.018);
    vec3 ha = vec3(abs(h.x), h.yz);
    skin = min(skin, sdLimb(ha, vec3(0.1, -0.01, -0.005), vec3(0.2, 0.045, -0.05), 0.024, 0.004));
    r = hU(r, skin, 0.0);
    float hair = max(length(h - vec3(0.0, 0.02, -0.02)) - 0.121, -sdBox3(h - vec3(0.0, -0.075, 0.1), vec3(0.09, 0.1, 0.09)));
    hair = min(hair, sdCapsule(h, vec3(-0.055, 0.035, 0.1), vec3(0.055, 0.035, 0.1), 0.026));
    hair = min(hair, sdLimb(ha, vec3(0.085, 0.02, 0.07), vec3(0.095, -0.075, 0.075), 0.026, 0.012));
    r = hU(r, hair, 3.0);
    float cap = smin(sdLimb(h, vec3(0.0, 0.065, -0.02), vec3(0.0, 0.02, -0.21), 0.124, 0.074),
                     sdLimb(h, vec3(0.0, 0.02, -0.21), vec3(0.0, -0.09, -0.4), 0.074, 0.016), 0.03);
    r = hU(r, max(cap, (0.42 * h.z - 0.005) - h.y), 2.0); // brim follows the brow
  }
  // arms: sleeve, forearm, hand
  if (length(p - u_hero_j[4]) - 0.33 < r.x) {
    r = hU(r, sdLimb(p, u_hero_j[3], u_hero_j[4], 0.054, 0.047), 1.0);
    r = hU(r, sdLimb(p, u_hero_j[4], u_hero_j[5], 0.047, 0.04), 12.0);
    r = hU(r, length(p - u_hero_j[5]) - 0.046, 0.0);
  }
  if (length(p - u_hero_j[7]) - 0.33 < r.x) {
    r = hU(r, sdLimb(p, u_hero_j[6], u_hero_j[7], 0.054, 0.047), 1.0);
    r = hU(r, sdLimb(p, u_hero_j[7], u_hero_j[8], 0.047, 0.04), 13.0);
    r = hU(r, length(p - u_hero_j[8]) - 0.046, 0.0);
  }
  // legs: trousers, shin into the boot, foot
  if (length(p - u_hero_j[10]) - 0.62 < r.x) {
    r = hU(r, sdLimb(p, u_hero_j[9], u_hero_j[10], 0.078, 0.06), 5.0);
    r = hU(r, sdLimb(p, u_hero_j[10], u_hero_j[11], 0.058, 0.05), 14.0);
    r = hU(r, sdLimb(p, u_hero_j[11], u_hero_j[12], 0.056, 0.042), 4.0);
  }
  if (length(p - u_hero_j[14]) - 0.62 < r.x) {
    r = hU(r, sdLimb(p, u_hero_j[13], u_hero_j[14], 0.078, 0.06), 5.0);
    r = hU(r, sdLimb(p, u_hero_j[14], u_hero_j[15], 0.058, 0.05), 15.0);
    r = hU(r, sdLimb(p, u_hero_j[15], u_hero_j[16], 0.056, 0.042), 4.0);
  }
  // paraglider: a dome-shell canopy in the (level) facing frame, two lines
  float g = u_hero_misc.x;
  if (g > 0.01 && length(p - u_hero_j[17]) - HERO_GLIDER_REACH * g < r.x) {
    vec3 c = u_hero_j[17];
    vec3 right = vec3(u_hero_misc.w, 0.0, -u_hero_misc.z);
    vec3 v = (p - c) / g;
    vec3 cq = vec3(dot(v, right), v.y, v.x * u_hero_misc.z + v.z * u_hero_misc.w);
    // dome shell (radius 1.6, crown 0.25 above the joint) cut to an elliptic
    // wing planform, 2.1 m span × 0.9 m chord
    float shell = abs(length(cq - vec3(0.0, -1.35, 0.0)) - 1.6) - 0.014;
    float wing = (length(cq.xz / vec2(1.05, 0.45)) - 1.0) * 0.45;
    r = hU(r, max(shell, max(wing, -cq.y - 0.3)) * g, 10.0);
    r = hU(r, min(sdCapsule(p, u_hero_j[5], c + g * (vec3(0.0, -0.1, 0.0) - 0.42 * right), 0.009),
                  sdCapsule(p, u_hero_j[8], c + g * (vec3(0.0, -0.1, 0.0) + 0.42 * right), 0.009)), 11.0);
  }
  return r;
}

// Sphere-trace the camera ray (origin 0, camera-relative) inside the bounds.
float heroTrace(vec3 rd) {
  vec3 c = u_hero_bound.xyz;
  float b = dot(c, rd);
  float h = b * b - dot(c, c) + u_hero_bound.w * u_hero_bound.w;
  if (h < 0.0) return -1.0;
  h = sqrt(h);
  float t = max(b - h, 0.0), t1 = b + h;
  if (t1 < 0.0) return -1.0;
  for (int i = 0; i < HERO_STEPS; i++) {
    float d = heroMap(rd * t).x;
    if (d < max(0.0015, 0.4 * t * u_footprint)) return t;
    t += d * 0.9;
    if (t > t1) break;
  }
  return -1.0;
}

// Soft shadow the hero casts on a camera-relative point (itself included).
float heroShadow(vec3 pr) {
  vec3 oc = pr - u_hero_bound.xyz;
  float b = dot(oc, SUN);
  float c = dot(oc, oc) - u_hero_bound.w * u_hero_bound.w;
  float h = b * b - c;
  if (h < 0.0 || (b > 0.0 && c > 0.0)) return 1.0;
  h = sqrt(h);
  float s = max(-b - h, 0.02), s1 = -b + h;
  float res = 1.0;
  for (int k = 0; k < 24; k++) {
    float d = heroMap(pr + SUN * s).x;
    res = min(res, 8.0 * d / s);
    if (res < 0.02) return 0.0;
    s += clamp(d, 0.02, 0.25);
    if (s > s1) break;
  }
  return clamp(res, 0.0, 1.0);
}

// Ambient contact darkening on the ground around the hero's legs.
float heroContactAO(vec3 pr) {
  float d = sdCapsule(pr, u_hero_j[0], 0.5 * (u_hero_j[11] + u_hero_j[15]), 0.18);
  return 1.0 - 0.5 * exp(-18.0 * d * max(d, 0.0) - 2.0 * max(-d, 0.0));
}

vec3 heroAlbedo(int id, vec3 pr, out float spec) {
  spec = 0.0;
  if (id == 0) {
    vec3 h = (pr - u_hero_j[2]) * u_hero_head;
    vec2 e = vec2(abs(h.x) - 0.04, (h.y + 0.004) * 0.75);
    if (h.z > 0.07 && dot(e, e) < 0.017 * 0.017) { spec = 0.5; return HC_EYE; }
    return HC_SKIN;
  }
  if (id == 1) return HC_TUNIC;
  if (id == 2) return HC_CAP;
  if (id == 3) return HC_HAIR;
  if (id == 4) return HC_LEATHER;
  if (id == 5) return HC_CREAM;
  if (id == 6) { spec = 1.0; return HC_GOLD; }
  if (id == 7) {
    // shield face: silver rim, Triforce, red chevron crest
    vec2 s = ((pr - u_hero_j[0]) * u_hero_torso).xy;
    if (heroShieldEdge(s) > -0.022) { spec = 0.8; return HC_SILVER; }
    float tri = min(sdTri(s - vec2(0.0, 0.33), 0.034),
                    min(sdTri(s - vec2(-0.034, 0.271), 0.034), sdTri(s - vec2(0.034, 0.271), 0.034)));
    if (tri < 0.0) { spec = 0.9; return HC_GOLD; }
    if (abs(s.y - 0.15 + 0.55 * abs(s.x)) < 0.018 && abs(s.x) < 0.14) return HC_RED;
    return HC_SHIELD;
  }
  if (id == 8) return HC_HILT;
  if (id == 9) return HC_SHEATH;
  if (id == 10) {
    vec3 v = pr - u_hero_j[17];
    float x = (v.x * u_hero_misc.w - v.z * u_hero_misc.z) / max(u_hero_misc.x, 0.05);
    return fract(x * 2.2 + 0.5) < 0.18 ? HC_GLIDER2 : HC_GLIDER;
  }
  if (id == 11) return HC_WOOD;
  // forearms 12/13 (sleeve → gauntlet), shins 14/15 (trousers → boot).
  // Constant indices only: a dynamic u_hero_j[i] made ANGLE/Metal copy the
  // array and cost the whole program 8 % (bench, Apple M1).
  vec3 a = u_hero_j[14], b = u_hero_j[15];
  if (id == 12) { a = u_hero_j[4]; b = u_hero_j[5]; }
  else if (id == 13) { a = u_hero_j[7]; b = u_hero_j[8]; }
  else if (id == 14) { a = u_hero_j[10]; b = u_hero_j[11]; }
  vec3 ab = b - a;
  float t = clamp(dot(pr - a, ab) / dot(ab, ab), 0.0, 1.0);
  return t > (id < 14 ? 0.5 : 0.38) ? HC_LEATHER : HC_CREAM;
}

// Tetrahedral-difference normal at a camera-relative hero point.
vec3 heroNormal(vec3 pr, float tH) {
  float e = max(0.001, 0.3 * tH * u_footprint);
  vec3 n = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec3 k = 2.0 * vec3(float(((i + 3) >> 1) & 1), float((i >> 1) & 1), float(i & 1)) - 1.0;
    n += k * heroMap(pr + k * e).x;
  }
  return normalize(n);
}

// sh: sun visibility (terrain soft shadow × the hero's own shadow).
vec3 heroShade(vec3 rd, float tH, vec3 n, float sh) {
  vec3 pr = rd * tH;
  float spec;
  vec3 alb = heroAlbedo(int(heroMap(pr).y + 0.5), pr, spec);
  float ndl = dot(n, SUN);
  float occ = 0.0;
  for (int i = 1; i <= 3; i++) {
    float hp = 0.04 * float(i);
    occ += (hp - heroMap(pr + n * hp).x) / float(i);
  }
  float ao = clamp(1.0 - 3.0 * occ, 0.25, 1.0) * (0.55 + 0.45 * smoothstep(0.0, 0.8, pr.y - u_hero_misc.y));
  // Soft two-tone terminator, sky fill, warm ground bounce, sky rim.
  float lit = smoothstep(0.0, 0.25, ndl) * 0.7 + max(ndl, 0.0) * 0.3;
  vec3 col = alb * (SUN_COL * 0.85 * lit * sh
    + SKY_TINT * (0.38 + 0.3 * n.y) * ao
    + vec3(0.10, 0.085, 0.05) * max(-n.y, 0.0) * ao);
  float rim = 1.0 - max(dot(n, -rd), 0.0);
  col += SKY_HOR * 0.18 * rim * rim * rim * ao;
  if (spec > 0.0) {
    float sp = max(dot(n, normalize(SUN - rd)), 0.0);
    sp *= sp; sp *= sp; sp *= sp; sp *= sp; sp *= sp;
    col += SUN_COL * (0.5 * spec * sp * sh);
  }
  return applyFog(col, tH, u_cam_pos.y + pr.y, rd);
}
#endif // HERO

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

  // Hero first: its hit distance caps the terrain march.
#ifdef HERO
  float heroT = u_hero_bound.w > 0.0 ? heroTrace(rd) : -1.0;
#else
  const float heroT = -1.0;
#endif
  float tMax = heroT > 0.0 ? heroT : MAXDIST;

  // Analytical water-plane intersection distance
  float tWaterPlane = (rd.y < -1e-5 && ro.y > W_LEVEL) ? (W_LEVEL - ro.y) / rd.y : -1.0;
  bool waterChecked = false;

  float t = START_T;
  float tPrev = t;
  float gradSq = 0.0;
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

    // Lipschitz ‖∇E‖ via forward differences, refreshed every GRAD_CADENCE-th
    // sample (§16/§34.1) and only within GRAD_GATE clamp-lengths of the surface;
    // farther up the horizon clamp bounds the step and the 2 extra evaluations
    // are skipped.
    float clampT = t * CLAMP_SLOPE + CLAMP_BASE;   // horizon clamp (anti-tunnel)
    bool farAbove = (p.y - e) > clampT * GRAD_GATE;
    if ((i % GRAD_CADENCE) == 0) {
      if (!farAbove) {
        float eps = max(dp, GRAD_EPS_MIN);
        vec2 g = p.xz + u_chunk_origin;
        float ex = elevation(g + vec2(eps, 0.0), dp, u_seed);
        float ez = elevation(g + vec2(0.0, eps), dp, u_seed);
        float invEps = 1.0 / eps;
        float gx = (ex - e) * invEps;
        float gz = (ez - e) * invEps;
        gradSq = gx * gx + gz * gz;
      } else {
        gradSq = 0.0;
      }
    }

    tPrev = t;
    float step = max(DMIN, (p.y - e) * inversesqrt(gradSq + 1.0) * LAMBDA);
    step *= 1.0 + t * GROWTH;            // §26 adaptive growth
    // Far above the surface the gradient is not refreshed and the clamp alone
    // paces the march, so it may widen there without loosening the near cap.
    step = min(step, farAbove ? t * CLAMP_SLOPE_FAR + CLAMP_BASE : clampT);
    t += step;

    if (t > tMax) {
      // Past the hero: one last sample exactly at it resolves water or terrain
      // in front of the body, then stop. Without a hero tMax = MAXDIST.
      if (tMax >= MAXDIST || tPrev >= tMax) break;
      t = tMax;
    }
  }

  if (hit) {
    // --- bisection refinement over [tPrev, t] (§16) ---
    float lo = tPrev, hi = t;
    for (int b = 0; b < BISECT; b++) {
      float tm = 0.5 * (lo + hi);
      vec3 m = ro + rd * tm;
      float gap = m.y - elevation(m.xz + u_chunk_origin, tm * u_footprint, u_seed);
      if (gap < 0.0) hi = tm; else lo = tm;
      if ((hi - lo) < BISECT_TOL) break;
    }
    tHit = 0.5 * (lo + hi);
  }

  vec3 col;
  bool heroFront = heroT > 0.0 && !hit && tWater < 0.0;
  bool terrainFront = hit && (tWater < 0.0 || tHit < tWater);
  bool waterFront = !heroFront && !terrainFront && tWater > 0.0;

  if (heroFront || terrainFront || waterFront) {
    float tS = heroFront ? heroT : terrainFront ? tHit : tWater;
    vec3 p = ro + rd * tS;
    vec2 g = p.xz + u_chunk_origin;
    float dp = max(tS * u_footprint, 1e-4);

    // ---- surface normal ----
    vec3 n;
    float gradLenSq = 0.0;
#ifdef HERO
    if (heroFront) {
      n = heroNormal(rd * tS, tS);
    } else
#endif
    if (terrainFront) {
      vec2 grad2 = gradient(g, dp, dp, u_seed);   // §17: ε = pixel footprint
      gradLenSq = dot(grad2, grad2);
      float invLen = inversesqrt(gradLenSq + 1.0);
      n = vec3(-grad2.x, 1.0, -grad2.y) * invLen;
    } else {
      // ripple normal — dynamic surface, deterministic in (x, z, t)
      float r1 = vnoise(g.x / 5.5 + u_time * 0.35, g.y / 5.5 - u_time * 0.22, u_seed + 911);
      float r2 = vnoise(g.x / 9.7 - u_time * 0.20, g.y / 9.7 + u_time * 0.31, u_seed + 913);
      vec2 wr = vec2(-0.22 * r1, -0.22 * r2);
      float invWLen = inversesqrt(dot(wr, wr) + 1.0);
      n = vec3(wr.x, 1.0, wr.y) * invWLen;
    }

    // ---- sun visibility: one soft-shadow ray per pixel (§18), whatever the
    // surface, and one query of the hero's shadow; single call sites keep the
    // inlined shader small ----
    float ndl = max(dot(n, SUN), 0.0);
    float sh = 0.0;
    if (waterFront || ndl > 0.0) {
      vec3 so = heroFront ? n * 0.05 : terrainFront ? n * 0.15 : vec3(0.0, 0.15, 0.0);
      sh = softShadow(p + so, u_seed, u_footprint);
#ifdef HERO
      if (u_hero_bound.w > 0.0 && sh > 0.0) sh = min(sh, heroShadow(rd * tS + (heroFront ? n * 0.03 : vec3(0.0))));
#endif
    }

#ifdef HERO
    if (heroFront) {
      col = heroShade(rd, tS, n, sh);
    } else
#endif
    if (terrainFront) {
      // ---------------- terrain shading ----------------
      float e = p.y; // bisection-converged surface hit point elevation (§16/§30)
      float T = temperature(g, e, u_seed);
      float M = moisture(g, u_seed);

      float vegNoise = vnoise(g.x / VEG_WL, g.y / VEG_WL, u_seed + 171);
      float veg;
      vec3 albedo = biomeColor(T, M, e, sqrt(gradLenSq), vegNoise, veg);

      vec3 sunTerm = albedo * SUN_COL * ndl * sh;
      vec3 ambTerm = albedo * SKY_TINT * (0.22 + 0.28 * max(n.y, 0.0));
#ifdef HERO
      if (u_hero_bound.w > 0.0) ambTerm *= heroContactAO(rd * tHit);
#endif

      col = sunTerm + ambTerm;
      col = applyFog(col, tHit, p.y, rd);
    } else {
      // ---------------- water plane (§8) ----------------
      float depth = max(W_LEVEL - eWater, 0.0);
      vec3 shallow = vec3(0.110, 0.320, 0.300);
      vec3 deep    = vec3(0.016, 0.075, 0.100);
      vec3 body = mix(shallow, deep, smoothstep(1.5, 14.0, depth))
        * (0.35 + 0.65 * sh) * (0.6 + 0.4 * max(dot(n, SUN), 0.0));

      // shoreline foam band
      float foamN = vnoise(g.x / 3.1, g.y / 3.1, u_seed + 917);
      body += vec3(0.9) * (1.0 - smoothstep(0.05, 0.6, depth)) * (0.25 + 0.35 * foamN);

      vec3 refld = reflect(rd, n);
      vec3 refl = skyColor(refld);
      float f1 = 1.0 - max(dot(-rd, n), 0.0);
      float f2 = f1 * f1;
      float f4 = f2 * f2;
      float fres = 0.02 + 0.98 * (f4 * f1);

      float sdot = max(dot(refld, SUN), 0.0);
      float spec = 0.0;
      if (sdot > 0.8) {
        float sp2 = sdot * sdot;
        float sp4 = sp2 * sp2;
        float sp8 = sp4 * sp4;
        float sp16 = sp8 * sp8;
        float sp32 = sp16 * sp16;
        float sp64 = sp32 * sp32;
        float sp128 = sp64 * sp64;
        spec = (sp128 * sp64) * (sp32 * sp16) * 3.0;
      }

      col = mix(body, refl, fres) + SUN_COL * spec / 3.4;
      col = applyFog(col, tWater, W_LEVEL, rd);
    }
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

/**
 * The fragment shader with the third-person hero compiled in. Without the
 * define every hero function is preprocessed out and the terrain path is the
 * whole program, so first person and fly pay nothing for the hero.
 */
export function withHero(fs: string): string {
  const nl = fs.indexOf('\n');
  return `${fs.slice(0, nl + 1)}#define HERO 1\n${fs.slice(nl + 1)}`;
}

/**
 * Runtime GLSL comment/whitespace stripper, applied before handing the source
 * to the driver (~0.5 ms at startup). It does not shrink the download: the
 * bundle still ships the commented source (36 KB raw / 9.6 KB gzip); a
 * build-time strip would save 2.9 KB gzip (1.6 % of /game first-load JS).
 */
export function minifyGlsl(source: string): string {
  const lines = source.split('\n');
  const result: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('#')) {
      result.push(trimmed);
      continue;
    }
    const code = line.replace(/\/\/.*$/, '');
    if (code.trim().length > 0) {
      result.push(code);
    }
  }
  let joined = result.join('\n');
  joined = joined.replace(/\/\*[\s\S]*?\*\//g, '');
  joined = joined.replace(/[ \t]+/g, ' ');
  joined = joined.replace(/\n\s*\n/g, '\n');
  return joined.trim();
}

