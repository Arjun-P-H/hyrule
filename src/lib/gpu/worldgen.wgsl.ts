/**
 * W(x,y) world function in WGSL — the WebGPU twin of demoShader.ts / terrainMath.ts.
 * ---------------------------------------------------------------------------
 * A line-for-line port of the GLSL terrain functions (hashing, quintic value
 * noise, fBm / ridged octaves with Nyquist weights, Worley valleys, domain
 * warp, the lands' biome regions, landforms and water, climate,
 * the biome look, sky and fog). Constants are interpolated from src/lib/constants.ts, the single
 * source of truth, so this twin cannot drift from the CPU one. The GLSL's
 * rounded literals (sun direction, ridged amplitude sum) are kept verbatim so
 * WebGPU frames can be diffed against the WebGL2 renderer.
 *
 * Coordinates: y-up, the field is E(x, z). Callers pass fp32 globals
 * `g = chunk_local + chunk_origin`, exactly like the WebGL2 shader.
 */

import {
  F, WARP, VALLEY, CLIMATE, LIGHT, LANDS, GEO, HYDRO,
  MAX_WORLD_H, MAX_DIST,
} from '../constants';
import { BI, BIOMES, BIOME_COUNT, LANDFORM, LANDFORM_KEYS, LOOK, SNOW_ALBEDO } from '../biomes';
import { EROSION_WGSL } from '../erosion';
import { COAST_RAMP, CRATER_RIM, SINK } from '../hydro';
import { SIMPLEX_F2, SIMPLEX_G2, SIMPLEX_GRAD } from '../noise';
import { CANYON, CEIL_SPAN, CEIL_START, DRY_TABLE, ERO_STEP, GULLY_SLOPE, KARST_BANK, KARST_CEIL, KARST_NORMALS, KARST_TURNS, LAND_MEANS, MESA, RIDGE_AMP, RIDGE_DAMP, RIDGE_FREQ, RIDGE_SCALE, RIDGE_SPAN, SEA_CLIFF } from '../terrainMath';

/** A WGSL float literal for a JS number (parsed by WGSL to the nearest f32). */
export function wgslF32(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`non-finite WGSL literal: ${x}`);
  const s = String(x);
  return /[.eE]/.test(s) ? s : `${s}.0`;
}

const f = wgslF32;
const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;
const NB = BIOME_COUNT;
const NK = LANDFORM_KEYS.length;
const LF_ROWS = BIOMES.flatMap((b) => LANDFORM_KEYS.map((k) => f(LANDFORM[b][k])));
const L_LOOK = BIOMES.map((b) => LOOK[b]);

export const WORLD_WGSL = /* wgsl */ `
// The rivers' node table round the camera (hydro.ts hydroTable): bound at 63 by every pipeline that
// evaluates the land's surface.
@group(0) @binding(63) var<storage, read> hydroBuf: array<vec4f>;
// ---- contract constants (from src/lib/constants.ts) -------------------------
const MAXH: f32 = ${f(MAX_WORLD_H)};
const MAXDIST: f32 = ${f(MAX_DIST)};
${Object.entries(F).filter(([k]) => k !== 'veg').map(([k, v]) => `const F_${k.toUpperCase()}: f32 = ${f(v)};`).join('\n')}
const VEG_WL: f32 = ${f(F.veg)};
const S_WARP: f32 = ${f(WARP.strength)};
const W_VALLEY: f32 = ${f(VALLEY.wValley)};
const W_LEVEL: f32 = ${f(VALLEY.wLevel)};
const SEA: f32 = ${f(LANDS.sea)};
const COAST_LEVEL: f32 = ${f(LANDS.coast.level)};
const RIDGE_MEAN: f32 = ${f(LAND_MEANS.ridge)};
const TOWER_MEAN: f32 = ${f(LAND_MEANS.tower)};
const DUNE_MEAN: f32 = ${f(LAND_MEANS.dune)};
const CEIL_START: f32 = ${f(CEIL_START)};
const DRY_TABLE: f32 = ${f(DRY_TABLE)};
const CEIL_SPAN: f32 = ${f(CEIL_SPAN)};
const RIDGE_FREQS = array<f32, 5>(${RIDGE_FREQ.map(f).join(', ')});
const RIDGE_AMPS = array<f32, 5>(${RIDGE_AMP.map(f).join(', ')});
const RIDGE_SCALE: f32 = ${f(RIDGE_SCALE)};
const RIDGE_SPAN: f32 = ${f(RIDGE_SPAN)};
const RIDGE_DAMP: f32 = ${f(RIDGE_DAMP)};
const KARST_NORMALS = array<vec2f, ${KARST_NORMALS.length}>(${KARST_NORMALS.map((v) => `vec2f(${f(v[0])}, ${f(v[1])})`).join(', ')});
const KARST_TURNS = array<vec2f, ${KARST_TURNS.length}>(${KARST_TURNS.map((v) => `vec2f(${f(v[0])}, ${f(v[1])})`).join(', ')});
const KARST_CEIL: f32 = ${f(KARST_CEIL)};
const KARST_BANK: f32 = ${f(KARST_BANK)};
const GULLY_SLOPE = vec4f(${GULLY_SLOPE.map(f).join(', ')});
const T_BASE: f32 = ${f(CLIMATE.tBase)};
const T_AMP: f32 = ${f(CLIMATE.tAmp)};
const T_FINE: f32 = ${f(CLIMATE.tFine)};
const M_FINE: f32 = ${f(CLIMATE.mFine)};
const L_LAT: f32 = ${f(CLIMATE.lambdaLat)};
const L_ALT: f32 = ${f(CLIMATE.lambdaAlt)};
const M_BASE: f32 = ${f(CLIMATE.mBase)};
const M_AMP: f32 = ${f(CLIMATE.mNoiseAmp)};
const M_OCEAN: f32 = ${f(CLIMATE.mOceanLambda)};
const CLOUD_PLANE: f32 = ${f(LIGHT.cloudPlane)};
const CLOUD_OCT: i32 = ${LIGHT.cloudOctaves};
const SIGMA_FOG: f32 = ${f(LIGHT.sigmaFog)};
const FOG_H: f32 = ${f(LIGHT.fogScaleHeight)};
// The WebGL2 shader's literal (normalize(0.55, 0.42, 0.72) to 6 places) — kept
// verbatim so WebGPU and WebGL2 frames are comparable pixel for pixel.
const SUN: vec3f = vec3f(0.550744, 0.420568, 0.720974);
const SUN_COL: vec3f = ${v3(LIGHT.sunColor)} * ${f(LIGHT.sunEnergy)};
const SKY_ZEN: vec3f = ${v3(LIGHT.skyZenith)};
const SKY_HOR: vec3f = ${v3(LIGHT.skyHorizon)};
const SKY_TINT: vec3f = (SKY_HOR + SKY_ZEN) * 0.5;

const ASUM_MACRO: f32 = 1.99609375;
const ASUM_RIDGE: f32 = 0.93703808;
const ASUM_WARP: f32 = 1.5;
const ASUM_CLOUD: f32 = 1.875;
const INV_U32_X2: f32 = 2.0 / 4294967295.0;
const INV_4095: f32 = 1.0 / 4095.0;

// ---- §4 hashing (uint32, Murmur3 finalizer) ----------------------------------
fn hashU(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u); x = x * 0x85ebca6bu;
  x = x ^ (x >> 13u); x = x * 0xc2b2ae35u;
  x = x ^ (x >> 16u);
  return x;
}
fn hashU4(v0: vec4u) -> vec4u {
  var v = v0;
  v = v ^ (v >> vec4u(16u)); v = v * 0x85ebca6bu;
  v = v ^ (v >> vec4u(13u)); v = v * 0xc2b2ae35u;
  v = v ^ (v >> vec4u(16u));
  return v;
}
fn hash3(xi: i32, yi: i32, seed: i32) -> u32 {
  let h = bitcast<u32>(xi) ^ (bitcast<u32>(yi) * 0x27d4eb2fu) ^ (bitcast<u32>(seed) * 0x9e3779b9u);
  return hashU(h);
}

// ---- §5 quintic value noise ---------------------------------------------------
fn vnoise(x: f32, y: f32, seed: i32) -> f32 {
  let p = vec2f(x, y);
  let ip = floor(p);
  let fr = p - ip;
  let u = fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0);
  let ux0 = bitcast<u32>(i32(ip.x));
  let ux1 = ux0 + 1u;
  let uy0 = bitcast<u32>(i32(ip.y)) * 0x27d4eb2fu;
  let uy1 = uy0 + 0x27d4eb2fu;
  let s = bitcast<u32>(seed) * 0x9e3779b9u;
  let h0 = uy0 ^ s;
  let h1 = uy1 ^ s;
  let v = hashU4(vec4u(ux0 ^ h0, ux1 ^ h0, ux0 ^ h1, ux1 ^ h1));
  let abcd = vec4f(v) * INV_U32_X2;
  let ab_cd = mix(abcd.xz, abcd.yw, u.x);
  return mix(ab_cd.x, ab_cd.y, u.y) - 1.0;
}
// vnoise with its gradient (noise.ts vnoiseG): (value, d/dx, d/dy), the value vnoise's bit for bit.
fn vnoiseG(x: f32, y: f32, seed: i32) -> vec3f {
  let p = vec2f(x, y);
  let ip = floor(p);
  let fr = p - ip;
  let u = fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0);
  let du = 30.0 * fr * fr * (fr * (fr - 2.0) + 1.0);
  let ux0 = bitcast<u32>(i32(ip.x));
  let ux1 = ux0 + 1u;
  let uy0 = bitcast<u32>(i32(ip.y)) * 0x27d4eb2fu;
  let uy1 = uy0 + 0x27d4eb2fu;
  let s = bitcast<u32>(seed) * 0x9e3779b9u;
  let h0 = uy0 ^ s;
  let h1 = uy1 ^ s;
  let v = hashU4(vec4u(ux0 ^ h0, ux1 ^ h0, ux0 ^ h1, ux1 ^ h1));
  let abcd = vec4f(v) * INV_U32_X2;
  let ab_cd = mix(abcd.xz, abcd.yw, u.x);
  let k = abcd.x - abcd.y - abcd.z + abcd.w;
  return vec3f(mix(ab_cd.x, ab_cd.y, u.y) - 1.0, du.x * ((abcd.y - abcd.x) + k * u.y), du.y * ((abcd.z - abcd.x) + k * u.x));
}

// ---- 2-D simplex noise (noise.ts snoise): (value, d/dx, d/dy) -----------------
const SIMPLEX_F2: f32 = ${f(SIMPLEX_F2)};
const SIMPLEX_G2: f32 = ${f(SIMPLEX_G2)};
const SIMPLEX_GRAD = array<vec2f, 16>(${SIMPLEX_GRAD.map((g) => `vec2f(${f(g[0])}, ${f(g[1])})`).join(', ')});
fn snoiseCorner(d: vec2f, h: u32) -> vec3f {
  let r = 0.5 - dot(d, d);
  if (r <= 0.0) { return vec3f(0.0); }
  let g = SIMPLEX_GRAD[h >> 28u];
  let gd = dot(g, d);
  let r2 = r * r;
  let r4 = r2 * r2;
  return vec3f(r4 * gd, r4 * g - 8.0 * r2 * r * gd * d);
}
fn snoise(x: f32, y: f32, seed: i32) -> vec3f {
  let p = vec2f(x, y);
  let ip = floor(p + (x + y) * SIMPLEX_F2);
  let d0 = p - ip + (ip.x + ip.y) * SIMPLEX_G2;
  let o = select(vec2u(0u, 1u), vec2u(1u, 0u), d0.x > d0.y);
  let d1 = d0 - vec2f(o) + SIMPLEX_G2;
  let d2 = d0 - 1.0 + 2.0 * SIMPLEX_G2;
  let ux = bitcast<u32>(i32(ip.x));
  let uy = bitcast<u32>(i32(ip.y)) * 0x27d4eb2fu;
  let s = bitcast<u32>(seed) * 0x9e3779b9u;
  let h = hashU4(vec4u(ux ^ uy ^ s, (ux + o.x) ^ (uy + o.y * 0x27d4eb2fu) ^ s, (ux + 1u) ^ (uy + 0x27d4eb2fu) ^ s, 0u));
  return snoiseCorner(d0, h.x) + snoiseCorner(d1, h.y) + snoiseCorner(d2, h.z);
}

// 9-octave fBm (lacunarity 2, gain 1/2) with Nyquist octave weights.
fn fbm9(x: f32, y: f32, seed: i32, dp: f32, f0: f32, totalAsum: f32) -> f32 {
  let k = 2.0 * f0 * dp;
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
  let w0 = clamp(2.0 - k, 0.0, 1.0);
  if (w0 <= 0.0) { return 0.0; }
  var sum = w0 * vnoise(x, y, seed);
  var wsum = w0;
  let w1 = clamp(2.0 - 2.0 * k, 0.0, 1.0);
  if (w1 > 0.0) {
    sum += 0.5 * w1 * vnoise(x * 2.0, y * 2.0, seed + 40503);
    wsum += 0.5 * w1;
    let w2 = clamp(2.0 - 4.0 * k, 0.0, 1.0);
    if (w2 > 0.0) {
      sum += 0.25 * w2 * vnoise(x * 4.0, y * 4.0, seed + 81006);
      wsum += 0.25 * w2;
      let w3 = clamp(2.0 - 8.0 * k, 0.0, 1.0);
      if (w3 > 0.0) {
        sum += 0.125 * w3 * vnoise(x * 8.0, y * 8.0, seed + 121509);
        wsum += 0.125 * w3;
        let w4 = clamp(2.0 - 16.0 * k, 0.0, 1.0);
        if (w4 > 0.0) {
          sum += 0.0625 * w4 * vnoise(x * 16.0, y * 16.0, seed + 162012);
          wsum += 0.0625 * w4;
          let w5 = clamp(2.0 - 32.0 * k, 0.0, 1.0);
          if (w5 > 0.0) {
            sum += 0.03125 * w5 * vnoise(x * 32.0, y * 32.0, seed + 202515);
            wsum += 0.03125 * w5;
            let w6 = clamp(2.0 - 64.0 * k, 0.0, 1.0);
            if (w6 > 0.0) {
              sum += 0.015625 * w6 * vnoise(x * 64.0, y * 64.0, seed + 243018);
              wsum += 0.015625 * w6;
              let w7 = clamp(2.0 - 128.0 * k, 0.0, 1.0);
              if (w7 > 0.0) {
                sum += 0.0078125 * w7 * vnoise(x * 128.0, y * 128.0, seed + 283521);
                wsum += 0.0078125 * w7;
                let w8 = clamp(2.0 - 256.0 * k, 0.0, 1.0);
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

// 2-octave fBm (domain warp, dunes, moisture).
fn fbm2(x: f32, y: f32, seed: i32, dp: f32, f0: f32, totalAsum: f32) -> f32 {
  let k = 2.0 * f0 * dp;
  if (k <= 0.0) {
    return vnoise(x, y, seed) + 0.5 * vnoise(x * 2.0, y * 2.0, seed + 40503);
  }
  let w0 = clamp(2.0 - k, 0.0, 1.0);
  if (w0 <= 0.0) { return 0.0; }
  var sum = w0 * vnoise(x, y, seed);
  var wsum = w0;
  let w1 = clamp(2.0 - 2.0 * k, 0.0, 1.0);
  if (w1 > 0.0) {
    sum += 0.5 * w1 * vnoise(x * 2.0, y * 2.0, seed + 40503);
    wsum += 0.5 * w1;
  }
  return sum * (totalAsum / max(wsum, 1e-6));
}

// General fBm loop (clouds: 4 octaves).
fn fbmN(x: f32, y: f32, seed: i32, oct: i32, lac: f32, gain: f32, dp: f32, f0: f32, totalAsum: f32) -> f32 {
  let k = 2.0 * f0 * dp;
  var freq = 1.0;
  var amp = 1.0;
  var sum = 0.0;
  var wsum = 0.0;
  for (var i = 0; i < 9; i++) {
    if (i >= oct) { break; }
    let w = clamp(2.0 - freq * k, 0.0, 1.0);
    if (w <= 0.0) { break; }
    sum += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
    wsum += amp * w;
    freq *= lac;
    amp *= gain;
  }
  return sum * (totalAsum / max(wsum, 1e-6));
}

// 5-octave ridged multifractal (lacunarity 2.17, gain 0.48), Nyquist weight on 2f.
fn ridged5(x: f32, y: f32, seed: i32, dp: f32, f0: f32, totalAsum: f32) -> f32 {
  let k = 4.0 * f0 * dp;
  if (k <= 0.0) {
    let n0 = 1.0 - abs(vnoise(x, y, seed));
    let n1 = 1.0 - abs(vnoise(x * 2.17, y * 2.17, seed + 34283));
    let n2 = 1.0 - abs(vnoise(x * 4.7089, y * 4.7089, seed + 68566));
    let n3 = 1.0 - abs(vnoise(x * 10.218313, y * 10.218313, seed + 102849));
    let n4 = 1.0 - abs(vnoise(x * 22.17373921, y * 22.17373921, seed + 137132));
    return 0.5 * (n0 * n0) + 0.24 * (n1 * n1) + 0.1152 * (n2 * n2) + 0.055296 * (n3 * n3) + 0.02654208 * (n4 * n4);
  }
  let w0 = clamp(2.0 - k, 0.0, 1.0);
  if (w0 <= 0.0) { return 0.0; }
  let n0 = 1.0 - abs(vnoise(x, y, seed));
  var sum = 0.5 * w0 * (n0 * n0);
  var wsum = 0.5 * w0;
  let w1 = clamp(2.0 - 2.17 * k, 0.0, 1.0);
  if (w1 > 0.0) {
    let n1 = 1.0 - abs(vnoise(x * 2.17, y * 2.17, seed + 34283));
    sum += 0.24 * w1 * (n1 * n1);
    wsum += 0.24 * w1;
    let w2 = clamp(2.0 - 4.7089 * k, 0.0, 1.0);
    if (w2 > 0.0) {
      let n2 = 1.0 - abs(vnoise(x * 4.7089, y * 4.7089, seed + 68566));
      sum += 0.1152 * w2 * (n2 * n2);
      wsum += 0.1152 * w2;
      let w3 = clamp(2.0 - 10.218313 * k, 0.0, 1.0);
      if (w3 > 0.0) {
        let n3 = 1.0 - abs(vnoise(x * 10.218313, y * 10.218313, seed + 102849));
        sum += 0.055296 * w3 * (n3 * n3);
        wsum += 0.055296 * w3;
        let w4 = clamp(2.0 - 22.17373921 * k, 0.0, 1.0);
        if (w4 > 0.0) {
          let n4 = 1.0 - abs(vnoise(x * 22.17373921, y * 22.17373921, seed + 137132));
          sum += 0.02654208 * w4 * (n4 * n4);
          wsum += 0.02654208 * w4;
        }
      }
    }
  }
  return sum * (totalAsum / max(wsum, 1e-6));
}

// ---- §7 Worley F2 − F1 (valleys) --------------------------------------------
fn cellPts(h: vec4u) -> array<vec4f, 2> {
  return array<vec4f, 2>(
    vec4f(h & vec4u(0xfffu)) * INV_4095,
    vec4f((h >> vec4u(12u)) & vec4u(0xfffu)) * INV_4095);
}

fn voronoiEdge(p_in: vec2f, seed: i32) -> f32 {
  let ip = floor(p_in);
  let fr = p_in - ip;
  let s_term = bitcast<u32>(seed) * 0x9e3779b9u;
  let cy_term = bitcast<u32>(i32(ip.y)) * 0x27d4eb2fu;
  let row0 = cy_term ^ s_term;
  let rowP1 = (cy_term + 0x27d4eb2fu) ^ s_term;
  let rowM1 = (cy_term - 0x27d4eb2fu) ^ s_term;
  let cx0 = bitcast<u32>(i32(ip.x));
  let cxP1 = cx0 + 1u;
  let cxM1 = cx0 - 1u;

  let h0 = hashU(cx0 ^ row0);
  let p0 = vec2f(f32(h0 & 0xfffu), f32((h0 >> 12u) & 0xfffu)) * INV_4095;
  let d0 = p0 - fr;
  var f1_sq = dot(d0, d0);
  var f2_sq = 64.0;

  let po = cellPts(hashU4(vec4u(cxP1 ^ row0, cx0 ^ rowP1, cxM1 ^ row0, cx0 ^ rowM1)));
  let dx_o = po[0] - fr.x + vec4f(1.0, 0.0, -1.0, 0.0);
  let dy_o = po[1] - fr.y + vec4f(0.0, 1.0, 0.0, -1.0);
  let d2_o = dx_o * dx_o + dy_o * dy_o;
  for (var k = 0; k < 4; k++) {
    let d2 = d2_o[k];
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }

  let fx0_2 = fr.x * fr.x;
  let fx1_2 = (1.0 - fr.x) * (1.0 - fr.x);
  let fy0_2 = fr.y * fr.y;
  let fy1_2 = (1.0 - fr.y) * (1.0 - fr.y);
  let minDiag = min(min(fx1_2 + fy1_2, fx0_2 + fy1_2), min(fx1_2 + fy0_2, fx0_2 + fy0_2));
  if (minDiag < f2_sq) {
    let pd = cellPts(hashU4(vec4u(cxP1 ^ rowP1, cxM1 ^ rowP1, cxP1 ^ rowM1, cxM1 ^ rowM1)));
    let dx_d = pd[0] - fr.x + vec4f(1.0, -1.0, 1.0, -1.0);
    let dy_d = pd[1] - fr.y + vec4f(1.0, 1.0, -1.0, -1.0);
    let d2_d = dx_d * dx_d + dy_d * dy_d;
    for (var k = 0; k < 4; k++) {
      let d2 = d2_d[k];
      if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
    }
  }
  return sqrt(f2_sq) - sqrt(f1_sq);
}

// ---- §14 domain warp ----------------------------------------------------------
fn domainWarp(x: f32, y: f32, seed: i32, dp: f32) -> vec2f {
  let wx = x * F_WARP;
  let wy = y * F_WARP;
  let qx = fbm2(wx, wy, seed + 11, dp, F_WARP, ASUM_WARP);
  let qy = fbm2(wx + 5.2, wy + 1.3, seed + 37, dp, F_WARP, ASUM_WARP);
  return vec2f(x + S_WARP * qx, y + S_WARP * qy);
}
// domainWarp's Jacobian at world g (terrainMath.ts warpJ): (∂wx/∂x, ∂wx/∂z, ∂wz/∂x, ∂wz/∂z).
fn warpJ(g: vec2f, seed: i32) -> vec4f {
  let wx = g.x * F_WARP;
  let wy = g.y * F_WARP;
  let k = S_WARP * F_WARP;
  let a = vnoiseG(wx, wy, seed + 11).yz + vnoiseG(wx * 2.0, wy * 2.0, seed + 40514).yz;
  let b = vnoiseG(wx + 5.2, wy + 1.3, seed + 37).yz + vnoiseG((wx + 5.2) * 2.0, (wy + 1.3) * 2.0, seed + 40540).yz;
  return vec4f(1.0 + k * a.x, k * a.y, k * b.x, 1.0 + k * b.y);
}


// ---- the geography (mirror hydro.ts): the uplift surface ---------------------------
fn n01(g: vec2f, f: f32, seed: i32) -> f32 { return 0.5 + 0.5 * vnoise(g.x * f, g.y * f, seed); }

fn continentAt(w: vec2f, seed: i32) -> f32 {
  return 0.5 + 0.5 * (0.78 * vnoise(w.x * F_CONT, w.y * F_CONT, seed + 801) + 0.22 * vnoise(w.x * F_COAST, w.y * F_COAST, seed + 802));
}

// A lake basin's depth at w (m): one candidate bowl per basin cell, kept inside its cell.
fn basinAt(w: vec2f, seed: i32) -> f32 {
  let cell = ${f(GEO.basinCell)};
  let ix = i32(floor(w.x / cell));
  let iz = i32(floor(w.y / cell));
  let h = hash3(ix, iz, seed + 3151);
  if (f32(h >> 24u) / 255.0 >= ${f(GEO.basinP)}) { return 0.0; }
  let c = (vec2f(f32(ix), f32(iz)) + 0.3 + 0.4 * vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) * INV_4095) * cell;
  let h2 = hash3(ix, iz, seed + 3152);
  let R = ${f(GEO.basinR[0])} + ${f(GEO.basinR[1] - GEO.basinR[0])} * f32(h2 & 0xffffu) / 65535.0;
  let d = length(w - c) / R;
  if (d >= 1.0) { return 0.0; }
  let D = ${f(GEO.basinD[0])} + ${f(GEO.basinD[1] - GEO.basinD[0])} * f32(h2 >> 16u) / 65535.0;
  let q = 1.0 - smoothstep(0.0, 1.0, d);
  return D * q * sqrt(q);
}

// Mountain ranges at warped w (0–1) (mirror hydro.ts rangeAt).
fn rangeAt(w: vec2f, seed: i32) -> f32 {
  let rm = smoothstep(${f(GEO.rangeMask[0])}, ${f(GEO.rangeMask[1])}, n01(w, F_RANGEMASK, seed + 3121));
  if (rm <= 0.0) { return 0.0; }
    let b = w + ${f(GEO.rangeWarp)} * vec2f(vnoise(w.x * F_RANGEWARP, w.y * F_RANGEWARP, seed + 3113), vnoise(w.x * F_RANGEWARP + 3.7, w.y * F_RANGEWARP - 1.9, seed + 3114));
    let ca = ${f(Math.cos(GEO.rangeDir[0]))}; let sa = ${f(Math.sin(GEO.rangeDir[0]))};
    let cb = ${f(Math.cos(GEO.rangeDir[1]))}; let sb = ${f(Math.sin(GEO.rangeDir[1]))};
    let c1 = 1.0 - abs(vnoise((b.x * ca + b.y * sa) * F_RANGEU, (b.y * ca - b.x * sa) * F_RANGEV, seed + 3111));
    let c2 = 1.0 - abs(vnoise((b.x * cb + b.y * sb) * F_RANGEU, (b.y * cb - b.x * sb) * F_RANGEV, seed + 3112));
    let c = mix(c1, c2, smoothstep(0.38, 0.62, n01(w, F_RANGEBLEND, seed + 3115)));
    let pk = ${f(GEO.pass)} + ${f(1 - GEO.pass)} * n01(w, F_PEAKS, seed + 3131);
    return rm * pow(c, ${f(GEO.rangePow)}) * pk;
}

// The uplift's raw height over the shore at warped w without a volcano (mirror hydro.ts baseAt).
fn baseAt(w: vec2f, seed: i32) -> f32 {
  let low = ${f(GEO.low[0])} + ${f(GEO.low[1])} * n01(w, F_UPL, seed + 3101) + ${f(GEO.low[2])} * n01(w, F_UPL2, seed + 3102);
  let rng = rangeAt(w, seed);
  let ms = smoothstep(${f(GEO.massif[0])}, ${f(GEO.massif[1])}, n01(w, F_MASSIF, seed + 3141));
  let pn = 0.72 * n01(w, F_PLATCOARSE, seed + 1401) + 0.28 * n01(w, F_PLATFINE, seed + 1402);
  let pl = 0.82 * smoothstep(${f(GEO.plat)}, ${f(GEO.plat + GEO.platEdge)}, pn) + 0.18 * smoothstep(${f(GEO.plat - 0.05)}, ${f(GEO.plat)}, pn);
  let notRange = 1.0 - smoothstep(0.15, 0.45, rng);
  return low + max(${f(GEO.rangeH)} * rng, ${f(GEO.massifH)} * ms * ms) + ${f(GEO.platH)} * pl * notRange - basinAt(w, seed) * notRange;
}

// The land's height over the shore for raw inland (mirror hydro.ts coastHeight).
fn coastHeight(raw: f32, inland: f32, cliff: f32, w: vec2f, seed: i32) -> f32 {
  if (inland >= 1.0) { return raw; }
  let cap = ${f(GEO.cliffH[0])} + ${f(GEO.cliffH[1] - GEO.cliffH[0])} * n01(w, F_CLIFFS, seed + 2702);
  return raw * inland + (1.0 - inland) * cliff * min(raw, cap);
}

struct Uplift {
  U: f32, C: f32, range: f32, massif: f32, plat: f32, cliff: f32, taper: f32,
  vol: vec4f,   // the volcano: centre x, z (warped), radius of its region (0: none), cone height
  vW: f32,      // its region's weight
  cone: f32,    // its cone's height here
  crater: f32,  // round its crater: the lava lake's level on the cone's scale (else -1e9)
  lava: f32,    // …and absolute (else -1e9)
  rr: f32,      // the distance from its summit as a share of its cone's radius (1: none)
};

// A volcano's cone at warped w (m): one candidate per cell, kept in the middle of it, on land, off the
// ranges, in land no colder than minT at sea level (mirror hydro.ts volcanoAt).
fn volcanoAt(w: vec2f, seed: i32, o: ptr<function, Uplift>) {
  let VC = ${f(LANDS.volcano.cell)};
  let ix = i32(floor(w.x / VC));
  let iz = i32(floor(w.y / VC));
  let h = hash3(ix, iz, seed + 2101);
  if (f32(h >> 24u) / 255.0 >= ${f(LANDS.volcano.p)}) { return; }
  let c = vec2f((f32(ix) + 0.28 + 0.44 * f32(h & 0xfffu) * INV_4095) * VC, (f32(iz) + 0.28 + 0.44 * f32((h >> 12u) & 0xfffu) * INV_4095) * VC);
  let h2 = hash3(ix, iz, seed + 2102);
  let R = ${f(LANDS.volcano.radius[0])} + ${f(LANDS.volcano.radius[1] - LANDS.volcano.radius[0])} * f32(h2 & 0xffffu) / 65535.0;
  let r = length(w - c);
  if (r >= R || continentAt(c, seed) <= COAST_LEVEL + 0.08 || rangeAt(c, seed) > 0.08) { return; }
  let T0 = T_BASE + T_AMP * (vnoise(c.x * F_TEMP, c.y * F_TEMP, seed + 1201) + T_FINE * vnoise(c.x * F_TEMPFINE, c.y * F_TEMPFINE, seed + 1202)) - L_LAT * abs(c.y);
  if (T0 < ${f(LANDS.volcano.minT)}) { return; }
  let H = ${f(LANDS.volcano.height[0])} + ${f(LANDS.volcano.height[1] - LANDS.volcano.height[0])} * f32(h2 >> 16u) / 65535.0;
  (*o).vol = vec4f(c, R, H);
  (*o).vW = 1.0 - smoothstep(0.55 * R, R, r);
  let Rc = ${f(LANDS.volcano.cone)} * R;
  if (r >= Rc) { return; }
  let q = 1.0 - r / Rc;
  var cone = H * q * pow(q, 0.9);
  let top = H * ${f(LANDS.volcano.crater)};
  if (cone > top) { cone = top - (cone - top) * 0.6; }
  (*o).rr = r / Rc;
  if ((*o).rr < ${f(CRATER_RIM * 1.6)}) { (*o).crater = top - 0.05 * H; }
  (*o).cone = cone;
}

// The uplift surface at warped w (m, absolute; the sea floor offshore).
fn upliftAt(w: vec2f, seed: i32) -> Uplift {
  var o: Uplift;
  o.vol = vec4f(0.0); o.vW = 0.0; o.cone = 0.0; o.crater = -1e9; o.lava = -1e9; o.rr = 1.0;
  let C = continentAt(w, seed);
  o.C = C;
  let cliff = smoothstep(${f(LANDS.coast.cliffs[0])}, ${f(LANDS.coast.cliffs[1])}, n01(w, F_CLIFFS, seed + 2701));
  let inland = smoothstep(COAST_LEVEL, COAST_LEVEL + ${f(LANDS.coast.inland)}, C);
  let taper = cliff + (1.0 - cliff) * inland;
  o.cliff = cliff;
  o.taper = taper;
  let low = ${f(GEO.low[0])} + ${f(GEO.low[1])} * n01(w, F_UPL, seed + 3101) + ${f(GEO.low[2])} * n01(w, F_UPL2, seed + 3102);
  let rng = rangeAt(w, seed);
  let ms = smoothstep(${f(GEO.massif[0])}, ${f(GEO.massif[1])}, n01(w, F_MASSIF, seed + 3141));
  let massif = ms * ms;
  let pn = 0.72 * n01(w, F_PLATCOARSE, seed + 1401) + 0.28 * n01(w, F_PLATFINE, seed + 1402);
  let pl = 0.82 * smoothstep(${f(GEO.plat)}, ${f(GEO.plat + GEO.platEdge)}, pn) + 0.18 * smoothstep(${f(GEO.plat - 0.05)}, ${f(GEO.plat)}, pn);
  o.range = rng; o.massif = massif; o.plat = pl;
  let hi = max(${f(GEO.rangeH)} * rng, ${f(GEO.massifH)} * massif);
  let notRange = 1.0 - smoothstep(0.15, 0.45, rng);
  volcanoAt(w, seed, &o);
  let base = low + hi + ${f(GEO.platH)} * pl * notRange - basinAt(w, seed) * notRange;
  var raw = base + o.cone;
  if (o.rr < 1.0) {
    let bc = baseAt(o.vol.xy, seed);
    raw += (bc - base) * (1.0 - smoothstep(0.35, 1.0, o.rr));
    if (o.crater > -1e8) { o.lava = SEA + ${f(LANDS.coast.beach)} + max(bc + o.crater, 1.0); }
  }
  o.U = SEA - 4.0 - 110.0 * smoothstep(COAST_LEVEL, COAST_LEVEL - 0.1, C);
  if (C >= COAST_LEVEL - ${f(COAST_RAMP)}) {
    // (the land rises from the sea floor over a steep but continuous face: a sea cliff, a beach's foot)
    let land = SEA + ${f(LANDS.coast.beach)} + coastHeight(max(raw, 1.0), inland, cliff, w, seed);
    o.U += (land - o.U) * smoothstep(COAST_LEVEL - ${f(COAST_RAMP)}, COAST_LEVEL, C);
  }
  return o;
}

// ---- §13 the lands: biome regions (mirror terrainMath.ts regionsInto) ------------
// Per biome (biomes.ts order) its landform parameters, LANDFORM_KEYS order: LF_TABLE[b * LF_K + k].
const LF_TABLE = array<f32, ${LF_ROWS.length}>(${LF_ROWS.join(', ')});

// The regions at raw g / warped w over the uplift up: their weights, the climate and the rare features there.
struct Regions {
  w: array<f32, ${NB}>,
  T0: f32, M: f32, C: f32,
  vol: vec4f,   // the volcano: centre x, z (warped), radius (0: none), cone height
  vW: f32,      // its region's weight
  sW: f32,      // the sacred grove's weight
};

fn regionsFrom(g: vec2f, w: vec2f, up: Uplift, seed: i32) -> Regions {
  var o: Regions;
  let T0 = T_BASE + T_AMP * (vnoise(w.x * F_TEMP, w.y * F_TEMP, seed + 1201) + T_FINE * vnoise(w.x * F_TEMPFINE, w.y * F_TEMPFINE, seed + 1202))
    - L_LAT * abs(g.y);
  let C = up.C;
  let M = M_BASE + M_AMP * (vnoise(w.x * F_MOISTR, w.y * F_MOISTR, seed + 1301) + M_FINE * vnoise(w.x * F_MOISTFINE, w.y * F_MOISTFINE, seed + 1302))
    + M_OCEAN * (1.0 - smoothstep(COAST_LEVEL, COAST_LEVEL + 0.12, C));
  o.T0 = T0; o.M = M; o.C = C;
  let mtn = smoothstep(0.26, 0.5, max(up.range, 0.9 * up.massif));
  let plat = smoothstep(0.35, 0.75, up.plat) * (1.0 - mtn);
  let low = 1.0 - mtn - plat;
  let a = smoothstep(${f(LANDS.cold[0])}, ${f(LANDS.cold[1])}, T0);
  let b = smoothstep(${f(LANDS.cool[0])}, ${f(LANDS.cool[1])}, T0);
  let c = smoothstep(${f(LANDS.hot[0])}, ${f(LANDS.hot[1])}, T0);
  let cold = 1.0 - a;
  let cool = a * (1.0 - b);
  let mild = a * b * (1.0 - c);
  let hot = a * b * c;
  let dry = 1.0 - smoothstep(${f(LANDS.dry[0])}, ${f(LANDS.dry[1])}, M);
  let hdry = 1.0 - smoothstep(${f(LANDS.hotDry[0])}, ${f(LANDS.hotDry[1])}, M);
  let arid = 1.0 - smoothstep(${f(LANDS.arid[0])}, ${f(LANDS.arid[1])}, M);
  let wet = smoothstep(${f(LANDS.wet[0])}, ${f(LANDS.wet[1])}, M);
  let bad = smoothstep(${f(LANDS.rugged[0])}, ${f(LANDS.rugged[1])}, n01(w, F_RUGGED, seed + 1501));
  let lake = smoothstep(${f(LANDS.lakedist[0])}, ${f(LANDS.lakedist[1])}, n01(w, F_LAKEDIST, seed + 1601));
  let karst = smoothstep(${f(LANDS.karst[0])}, ${f(LANDS.karst[1])}, n01(w, F_KARST, seed + 1701));
  let marsh = wet * smoothstep(${f(LANDS.marsh[0])}, ${f(LANDS.marsh[1])}, n01(w, F_MARSH, seed + 1801));
  for (var k = 0; k < ${NB}; k++) { o.w[k] = 0.0; }
  o.w[${BI.tundra}] += low * cold;
  o.w[${BI.taiga}] += low * cool * (1.0 - lake);
  o.w[${BI.lakes}] += low * (cool + mild * (1.0 - dry)) * lake;
  let mildL = low * mild * (1.0 - lake * (1.0 - dry));
  let badWet = smoothstep(${f(LANDS.badlandsWet[0])}, ${f(LANDS.badlandsWet[1])}, M);
  o.w[${BI.plains}] += mildL * dry * (1.0 - bad * (1.0 - badWet));
  o.w[${BI.badlands}] += mildL * dry * bad * (1.0 - badWet);
  o.w[${BI.forest}] += mildL * (1.0 - dry) * (1.0 - marsh);
  o.w[${BI.wetland}] += mildL * (1.0 - dry) * marsh;
  let hotL = low * hot;
  o.w[${BI.desert}] += hotL * arid * (1.0 - bad);
  o.w[${BI.badlands}] += hotL * (hdry * bad + arid * bad * (1.0 - hdry));
  o.w[${BI.savanna}] += hotL * hdry * (1.0 - arid) * (1.0 - bad);
  let hotM = hotL * (1.0 - hdry);
  o.w[${BI.rainforest}] += hotM * (1.0 - karst) * (1.0 - marsh);
  o.w[${BI.jungle}] += hotM * karst * (1.0 - marsh);
  o.w[${BI.wetland}] += hotM * marsh;
  o.w[${BI.highland}] += plat * (cold + cool + mild);
  o.w[${BI.badlands}] += plat * hot * hdry;
  o.w[${BI.jungle}] += plat * hot * (1.0 - hdry);
  o.w[${BI.mountains}] += mtn;

  let v = up.vW;
  o.vol = up.vol;
  // A sacred grove: one candidate per cell of a lattice offset by half a cell, in mild moist land
  // (the lattice in the warped plane, as terrainMath.ts regionsInto).
  let GC = ${f(LANDS.sacred.cell)};
  let sx = i32(floor(w.x / GC + 0.5));
  let sz = i32(floor(w.y / GC + 0.5));
  let sh = hash3(sx, sz, seed + 2201);
  var sc = 0.0;
  if (f32(sh >> 24u) * (1.0 / 255.0) < ${f(LANDS.sacred.p)}) {
    let cx = (f32(sx) - 0.5 + 0.3 + 0.4 * f32(sh & 0xfffu) * INV_4095) * GC;
    let cz = (f32(sz) - 0.5 + 0.3 + 0.4 * f32((sh >> 12u) & 0xfffu) * INV_4095) * GC;
    let R = ${f(LANDS.sacred.radius[0])} + ${f(LANDS.sacred.radius[1] - LANDS.sacred.radius[0])} * f32(hash3(sx, sz, seed + 2202) & 0xffffu) / 65535.0;
    let d = w - vec2f(cx, cz);
    if (dot(d, d) < R * R) {
      sc = (1.0 - smoothstep(0.6 * R, R, length(d))) * smoothstep(2.0, 5.0, T0) * smoothstep(0.3, 0.4, M) * (1.0 - mtn);
    }
  }
  if (v > 0.0) {
    for (var k = 0; k < ${NB}; k++) { o.w[k] *= 1.0 - v; }
    o.w[${BI.volcanic}] += v;
  }
  if (sc > 0.0) {
    let s = sc * (1.0 - v);
    for (var k = 0; k < ${NB}; k++) { o.w[k] *= 1.0 - s; }
    o.w[${BI.sacred}] += s;
  }
  o.vW = v;
  o.sW = sc;
  return o;
}

fn landRegions(g: vec2f, w: vec2f, seed: i32) -> Regions {
  return regionsFrom(g, w, upliftAt(w, seed), seed);
}

// The regions' landforms blended by their weights (LANDFORM_KEYS order).
struct Landform {
  ${LANDFORM_KEYS.map((k) => `${k}: f32`).join(', ')},
};
fn blendLandform(r: Regions) -> Landform {
  var p: array<f32, ${NK}>;
  for (var k = 0; k < ${NK}; k++) { p[k] = 0.0; }
  for (var i = 0; i < ${NB}; i++) {
    let wi = r.w[i];
    if (wi <= 0.0) { continue; }
    for (var k = 0; k < ${NK}; k++) { p[k] += wi * LF_TABLE[i * ${NK} + k]; }
  }
  return Landform(${LANDFORM_KEYS.map((_, i) => `p[${i}]`).join(', ')});
}

// Polynomial smooth max / min of a and b, the crease between them rounded over k (> 0).
fn ksmax(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return max(a, b) + 0.25 * h * h * k;
}
fn ksmin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - 0.25 * h * h * k;
}
// Karst towers at world g (mirror terrainMath.ts karstTowers): the highest of the seven-sided prisms
// (cap, wall, talus apron) of the 3 × 3 cells round g (m), and how much of g is tower.
fn karstTowers(g: vec2f, seed: i32, dp: f32) -> vec2f {
  let p = g * F_TOWER;
  let ip = floor(p);
  let ws = clamp(dp, 1.0, 24.0);
  let k2 = 4.0 * ws;
  let k3 = 0.7 * ws + 1.5;
  var best = 0.0;
  var cov = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let h2 = hash3(i32(ip.x) + i, i32(ip.y) + j, seed + 2302);
      if ((h2 >> 27u) < 4u) { continue; }
      let h = hash3(i32(ip.x) + i, i32(ip.y) + j, seed + 2301);
      let c = vec2f(f32(i), f32(j)) + 0.2 + 0.6 * vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) * INV_4095;
      let H = 50.0 + 170.0 * f32((h >> 28u) & 15u) / 15.0;
      let Rw = H / (2.0 * (0.7 + 0.8 * f32((h >> 24u) & 15u) / 15.0));
      let Ht = (0.12 + 0.08 * f32((h2 >> 18u) & 7u) / 7.0) * H;
      let Sw = 3.5 + 2.8 * f32((h2 >> 21u) & 7u) / 7.0;
      let St = 0.62 + 0.13 * f32((h2 >> 24u) & 7u) / 7.0;
      let Rz = Rw + (Ht + k3) / St;
      let u = (p - ip - c) / F_TOWER;
      if (dot(u, u) * 0.5184 >= Rz * Rz) { continue; }
      let tr = KARST_TURNS[h2 & 15u];
      let v = vec2f(tr.x * u.x + tr.y * u.y, tr.x * u.y - tr.y * u.x);
      let kr = 0.12 * Rw;
      var s = 0.0;
      for (var k = 0u; k < 7u; k++) {
        let d = dot(v, KARST_NORMALS[k]) / (0.8 + 0.4 * f32((h2 >> (4u + 2u * k)) & 3u) / 3.0);
        s = select(ksmax(s, d, kr), d, k == 0u);
      }
      let wall = Ht + Sw * (Rw - s);
      let talus = Ht - St * (s - Rw);
      let Rtop = Rw - (H - Ht) / Sw;
      let cap = H - (0.6 * Rtop * s * s) / (4.0 * Rtop * Rtop + s * s);
      let pr = ksmin(cap, ksmax(wall, talus, k2), 5.0 * ws + 0.03 * H);
      best = max(best, ksmax(pr, 0.0, k3));
      cov = max(cov, smoothstep(0.5 * Ht, Ht + 0.1 * H, pr));
    }
  }
  return vec2f(best, cov);
}

// Dunes: transverse ridges across a wind that turns over the field, crests bent and broken into
// crescents, a long windward rise to a sharp brink and a steep slip face, cross ripples (m).
fn dunesE(g: vec2f, seed: i32, dp: f32) -> f32 {
  let wMain = clamp(2.0 - 2.0 * dp * (2.0 * F_DUNE), 0.0, 1.0);
  let wCross = clamp(2.0 - 2.0 * dp * (2.0 * F_DUNECROSS), 0.0, 1.0);
  let amp = 0.3 + 0.7 * n01(g, F_DUNEAMP, seed + 2403);
  if (wMain <= 0.0) { return DUNE_MEAN * amp; }
  let ang = 0.6 + 0.9 * vnoise(g.x * F_DUNEFIELD, g.y * F_DUNEFIELD, seed + 2401);
  let ca = cos(ang);
  let sa = sin(ang);
  let u = (g.x * ca + g.y * sa) * F_DUNE + 0.9 * vnoise(g.x * F_WARP, g.y * F_WARP, seed + 2402) + 0.35 * vnoise(g.x * F_DUNEBEND, g.y * F_DUNEBEND, seed + 2405);
  let s = fract(u);
  var prof = 1.0 - smoothstep(0.86, 1.0, s);
  if (s < 0.86) { prof = pow(smoothstep(0.0, 0.86, s), 1.3); }
  var d = 30.0 * prof;
  if (wCross > 0.0) {
    let u2 = (g.y * ca - g.x * sa) * F_DUNECROSS + 0.6 * vnoise(g.x * F_DUNEBEND, g.y * F_DUNEBEND, seed + 2404);
    let s2 = fract(u2);
    var p2 = 1.0 - smoothstep(0.75, 1.0, s2);
    if (s2 < 0.75) { p2 = smoothstep(0.0, 0.75, s2); }
    d += 5.0 * wCross * p2;
  }
  return amp * (wMain * d + (1.0 - wMain) * DUNE_MEAN);
}

// Mesas and buttes (terrainMath.ts MESA): the tiers (n's threshold, height m), and the walls' and aprons' make.
const MESA_TIERS = array<vec2f, 3>(${MESA.tiers.map((t) => `vec2f(${f(t[0])}, ${f(t[1])})`).join(', ')});
// The mesas' plan noise n at world g.
fn mesaPlan(g: vec2f, seed: i32) -> f32 {
  return 0.5 + 0.35 * vnoise(g.x * F_MESA, g.y * F_MESA, seed + 2501) + 0.15 * vnoise(g.x * F_MESAFINE, g.y * F_MESAFINE, seed + 2502);
}
// The mesas' region weight at world g (its own warp, uplift and regions).
fn mesaWeightAt(g: vec2f, seed: i32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let rg = regionsFrom(g, w, upliftAt(w, seed), seed);
  var s = 0.0;
  for (var b = 0; b < ${NB}; b++) { s += rg.w[b] * LF_TABLE[b * ${NK} + ${LANDFORM_KEYS.indexOf('mesa')}]; }
  return smoothstep(0.35, 0.75, s);
}
// The mesas at world g where their region weighs mw (from the blended landform's share lfm), the weight
// applied, and how much of g is mesa (mirror terrainMath.ts mesasE: each tier a cap, a wall and a talus
// apron laid out by the distance in from its contour, refined by a secant step).
fn mesasE(g: vec2f, seed: i32, dp: f32, mw: f32, lfm: f32) -> vec2f {
  let hm = smoothstep(0.001, ${f(MESA.rise)}, mw);
  if (hm <= 0.0) { return vec2f(0.0); }
  let a = vnoiseG(g.x * F_MESA, g.y * F_MESA, seed + 2501);
  let b = vnoiseG(g.x * F_MESAFINE, g.y * F_MESAFINE, seed + 2502);
  let n = 0.5 + 0.35 * a.x + 0.15 * b.x;
  var gr = 0.35 * F_MESA * a.yz + 0.15 * F_MESAFINE * b.yz;
  // (the tiers retreat with the region's weight, whose gradient is a forward difference (MESA.fdSpan) where it is not yet 1)
  var mg = vec2f(0.0);
  if (lfm < 0.85) {
    let h = max(1.0, length(g) / ${f(MESA.fdSpan)});
    // (one call in a loop, not two: each call site inlines its own copy of the uplift and the regions)
    for (var k = 0; k < 2; k++) { mg[k] = (mesaWeightAt(g + h * vec2f(f32(1 - k), f32(k)), seed) - mw) / h; }
    gr += ${f(MESA.retreat)} * mg;
  }
  let gl = max(length(gr), 1e-7);
  let u = gr / gl;
  let mu = ${f(MESA.retreat)} * dot(mg, u);
  let hd0 = n01(g, F_MESA, seed + 2503);
  let lift = ${f(MESA.retreat)} * (1.0 - mw);
  let ws = max(dp, 1.0);
  let k3 = (0.7 * ws + 1.5) * hm + 1e-3;
  var e = 0.0;
  var top = 0.0;
  var cov = 0.0;
  for (var i = 0; i < 3; i++) {
    let H = MESA_TIERS[i].y * hm;
    let Cp = top;
    let C = top + H;
    top = C;
    let ei = n - MESA_TIERS[i].x - lift;
    var d = ei / gl;
    if (d > -${f(MESA.reach[1])} && d < ${f(MESA.reach[0])}) {
      let nq = mesaPlan(g - d * u, seed);
      let ds = d / clamp((n - nq + d * mu) / select(1e-12, ei, abs(ei) > 1e-12), 0.5, 2.0);
      d = ds + (d - ds) * smoothstep(0.75, 1.0, select(-d / ${f(MESA.reach[1])}, d / ${f(MESA.reach[0])}, d > 0.0));
    }
    let hd = clamp(hd0 + ${f(MESA.harder)} * f32(i - 1), 0.0, 1.0);
    let Sw = ${f(MESA.cliff[0])} + ${f(MESA.cliff[1] - MESA.cliff[0])} * hd;
    let hw = (${f(MESA.wall[0])} + ${f(MESA.wall[1] - MESA.wall[0])} * hd) * H;
    let St = ${f(MESA.talus[0])} + ${f(MESA.talus[1] - MESA.talus[0])} * hd;
    let cr = ${f(MESA.caprock[1])} * hw;
    // (the wall no steeper than its footprint can show)
    let m = min(1.0, (cr / ${f(MESA.caprock[0])} + (hw - cr) / Sw) / (1.5 * ws));
    let sc = ${f(MESA.caprock[0])} * m;
    let sw = Sw * m;
    let foot = -(cr / sc + (hw - cr) / sw);
    let wall = ksmax(C + sc * d, C - cr + sw * (d + cr / sc), ws);
    let talus = C - hw + St * (d - foot);
    let T = ksmin(C, ksmax(wall, talus, 3.0 * ws), 3.0 * ws + 0.02 * H);
    e += ksmax(T - Cp, 0.0, k3);
    cov = max(cov, smoothstep(foot - 0.5 * (H - hw) / St, foot, d));
  }
  return vec2f(e, cov * hm);
}

// Canyon walls (terrainMath.ts CANYON): a layer's profile, bottom, thickness, and its three parts' runs
// (shares) and D at their ends.
struct CLayer { b: f32, H: f32, ac: f32, ab: f32, at: f32, d1: f32, d2: f32 };
// Layer i's boundary (m).
fn canyonB(i: i32, seed: i32) -> f32 {
  return ${f(CANYON.layer)} * (f32(i) + ${f(CANYON.jitter)} * (f32(hash3(i, 0, seed + 2611) & 0xfffu) / 4095.0 - 0.5));
}
// A layer from b, H thick, of hardness hd, on a wall of slope s.
fn canyonProfile(b: f32, H: f32, hd: f32, s: f32) -> CLayer {
  let sc = ${f(CANYON.cliff[0])} + ${f(CANYON.cliff[1] - CANYON.cliff[0])} * hd;
  let st = min(${f(CANYON.talus[0])} + ${f(CANYON.talus[1] - CANYON.talus[0])} * hd, s);
  let rc = ${f(CANYON.share[0])} + ${f(CANYON.share[1] - CANYON.share[0])} * hd;
  let sb = ${f(CANYON.bench)};
  let A = rc / sc + (1.0 - rc) / st;
  let pb = clamp((1.0 / s - A) / (1.0 / sb - A), 0.0, 0.9);
  let pc = rc * (1.0 - pb);
  let pt = (1.0 - rc) * (1.0 - pb);
  var r = vec3f(pc * s / sc, pb * s / sb, pt * s / st);
  r /= r.x + r.y + r.z;
  return CLayer(b, H, r.x, r.y, r.z, pc - r.x, pc + pb - r.x - r.y);
}
// Layer i's profile on a wall of slope s.
fn canyonLayer(i: i32, s: f32, seed: i32) -> CLayer {
  let h = hash3(i, 0, seed + 2611);
  let b = ${f(CANYON.layer)} * (f32(i) + ${f(CANYON.jitter)} * (f32(h & 0xfffu) / 4095.0 - 0.5));
  return canyonProfile(b, canyonB(i + 1, seed) - b, f32((h >> 12u) & 0xfffu) / 4095.0, s);
}
// ∫₀ᵗ D over layer c (t its share of the layer's run).
fn canyonJ(c: CLayer, t: f32) -> f32 {
  if (t <= c.ac) { return c.d1 * t * t / (2.0 * max(c.ac, 1e-6)); }
  let j1 = 0.5 * c.d1 * c.ac;
  if (t <= c.ac + c.ab) {
    let u = t - c.ac;
    return j1 + c.d1 * u + (c.d2 - c.d1) * u * u / (2.0 * max(c.ab, 1e-6));
  }
  let u = min(t, 1.0) - c.ac - c.ab;
  return j1 + 0.5 * (c.d1 + c.d2) * c.ab + c.d2 * u - c.d2 * u * u / (2.0 * max(c.at, 1e-6));
}
// The layer holding height h (m).
fn canyonAt(h: f32, seed: i32) -> i32 {
  let i = i32(floor(h / ${f(CANYON.layer)}));
  if (h < canyonB(i, seed)) { return i - 1; }
  return select(i, i + 1, h >= canyonB(i + 1, seed));
}
// The canyon's raise (m, ≥ 0) at height h on a wall of slope s, box-filtered over heights h ± w/2; over
// more than a layer it fades to the strata's mean (mirror terrainMath.ts canyonRaise).
fn canyonRaise(h: f32, s: f32, w: f32, seed: i32) -> f32 {
  let L = ${f(CANYON.layer)};
  let f = smoothstep(0.72 * L, 1.2 * L, w);
  var r = 0.0;
  if (f < 1.0) {
    if (w < 1e-3) {
      let c = canyonLayer(canyonAt(h, seed), s, seed);
      let t = (h - c.b) / c.H;
      var D = c.d2 * (1.0 - t) / max(c.at, 1e-6);
      if (t <= c.ac) { D = c.d1 * t / max(c.ac, 1e-6); }
      else if (t <= c.ac + c.ab) { D = c.d1 + (c.d2 - c.d1) * (t - c.ac) / max(c.ab, 1e-6); }
      r = c.H * D;
    } else {
      let h1 = h - 0.5 * w;
      let h2 = h + 0.5 * w;
      let i1 = canyonAt(h1, seed);
      let i2 = canyonAt(h2, seed);
      var c = canyonLayer(i1, s, seed);
      var I = -c.H * c.H * canyonJ(c, (h1 - c.b) / c.H);
      for (var j = 0; j < 5 && i1 + j < i2; j++) {
        I += c.H * c.H * canyonJ(c, 1.0);
        c = canyonLayer(i1 + j + 1, s, seed);
      }
      I += c.H * c.H * canyonJ(c, (h2 - c.b) / c.H);
      r = I / w;
    }
  }
  if (f > 0.0) { r += (L * canyonJ(canyonProfile(0.0, L, 0.5, s), 1.0) - r) * f; }
  return r;
}


// The rolling hills' fBm at warped macro coords m, centred on zero (× the region's hills).
fn hillsAt(m: vec2f, seed: i32, dp: f32) -> f32 {
  let hLow = vnoise(m.x, m.y, seed + 1) + 0.5 * vnoise(m.x * 2.0, m.y * 2.0, seed + 1 + 40503);
  var hHigh = 0.0;
  let k = 2.0 * F_MACRO * dp;
  for (var i = 2; i < 9; i++) {
    let fr = exp2(f32(i));
    let wgt = select(clamp(2.0 - fr * k, 0.0, 1.0), 1.0, dp <= 0.0);
    if (wgt <= 0.0) { break; }
    hHigh += exp2(-f32(i)) * wgt * vnoise(m.x * fr, m.y * fr, seed + 1 + i * 40503);
  }
  return 0.2833333 * hLow + 0.5 * hHigh;
}

// The mountains' ridged octaves at warped w, centred on their mean (× ridge × the range there): simplex,
// each octave damped by the slope of those under it (terrainMath.ts ridgesAt).
fn ridgesAt(w: vec2f, seed: i32, dp: f32) -> f32 {
  let rw = w * F_RIDGE * RIDGE_SPAN;
  let kr = 4.0 * F_RIDGE * dp;
  var rh = 0.0;
  var slope2 = 0.0;
  for (var i = 0; i < 5; i++) {
    let fr = RIDGE_FREQS[i];
    let wgt = select(clamp(2.0 - fr * kr, 0.0, 1.0), 1.0, dp <= 0.0);
    if (wgt > 0.0) {
      let sn = snoise(rw.x * fr, rw.y * fr, seed + 5 + i * 34283);
      let n = 1.0 - abs(RIDGE_SCALE * sn.x);
      let a = RIDGE_AMPS[i] * wgt / (1.0 + RIDGE_DAMP * slope2);
      rh += a * (n * n - RIDGE_MEAN);
      let g = 2.0 * n * a * fr * RIDGE_SPAN * RIDGE_SCALE * length(sn.yz);
      slope2 += g * g;
    }
  }
  return rh;
}

// The land the gullies follow at warped w: the uplift U there and the hills and ridges over it, prefiltered at ERO_STEP (m).
const ERO_STEP = ${f(ERO_STEP)};
fn erosionLand(w: vec2f, U: f32, seed: i32, hills: f32, ridge: f32) -> f32 {
  var b = U + hills * hillsAt(w * F_MACRO, seed, ERO_STEP);
  if (ridge > 0.0) { b += ridge * ridgesAt(w, seed, ERO_STEP); }
  return b;
}
// erosionLand at world g.
fn erosionBase(g: vec2f, seed: i32, hills: f32, ridge: f32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  return erosionLand(w, upliftAt(w, seed).U, seed, hills, ridge);
}
${EROSION_WGSL}

// ---- the rivers (mirror hydro.ts riverField), read from the node table (hydroBuf) ----
// hydroBuf[0]: (i0, j0, n, node size); per node 6 vec4 from index 1: (x, z rel. to (i0, j0)·node, U, W),
// (code, acc, lake level or -1e9, fed: the river shows), levels at 1/16 … 15/16, lake radius.
const HY_DI = array<i32, 8>(-1, 0, 1, -1, 1, -1, 0, 1);
const HY_DJ = array<i32, 8>(-1, -1, -1, 0, 0, 1, 1, 1);
fn hyIndex(i: i32, j: i32) -> i32 {
  let h = hydroBuf[0];
  let li = i - i32(h.x);
  let lj = j - i32(h.y);
  let n = i32(h.z);
  if (li < 0 || lj < 0 || li >= n || lj >= n) { return -1; }
  return 1 + (lj * n + li) * 6;
}
// A valley wall's rise x out from its floor's rim, walls of slope k, and its slope (mirror hydro.ts wallRise, wallSlope).
fn wallRise(k: f32, x: f32, fl: f32) -> f32 {
  let s0 = ${f(2 * (HYDRO.floorRise - 0.3))} / fl;
  return ${f(HYDRO.floorRise)} + s0 * x + (k - s0) * (x - ${f(HYDRO.wallRound)} * (1.0 - exp(-x / ${f(HYDRO.wallRound)}))) + ${f(HYDRO.wallCurve)} * x * x;
}
fn wallSlope(k: f32, x: f32, fl: f32) -> f32 {
  let s0 = ${f(2 * (HYDRO.floorRise - 0.3))} / fl;
  return s0 + (k - s0) * (1.0 - exp(-x / ${f(HYDRO.wallRound)})) + ${f(2 * HYDRO.wallCurve)} * x;
}
fn valleyShape(dd: f32, ch: f32, w: f32, depth: f32, floorW: f32, k: f32) -> f32 {
  if (ch < w) {
    let q = ch / w;
    return -depth * (1.0 - q * q);
  }
  let fr = ${f(HYDRO.floorRise)};
  var base = 0.0;
  if (dd < floorW) { base = 0.3 + (fr - 0.3) * (dd / floorW) * (dd / floorW); } else { base = wallRise(k, dd - floorW, floorW); }
  return base * smoothstep(w, w + 5.0, ch);
}
// How steeply valleyShape rises across the valley there (m per m, a bound).
fn valleySlope(dd: f32, ch: f32, w: f32, depth: f32, floorW: f32, k: f32) -> f32 {
  if (ch < w) { return 2.0 * depth * ch / (w * w); }
  let fr = ${f(HYDRO.floorRise)};
  let x = dd - floorW;
  var base = 0.0;
  var db = 0.0;
  if (dd >= floorW) { base = wallRise(k, x, floorW); db = wallSlope(k, x, floorW); }
  if (dd < floorW) { base = 0.3 + (fr - 0.3) * (dd / floorW) * (dd / floorW); db = 2.0 * (fr - 0.3) * dd / (floorW * floorW); }
  let q = clamp((ch - w) / 5.0, 0.0, 1.0);
  return db * q * q * (3.0 - 2.0 * q) + base * 1.2 * q * (1.0 - q);
}
// A reach's level at t with its falls and without (smooth), and how fast each falls (m per unit t), in
// one pass over its profile (reachLevel, reachFall).
fn reachLF(Wa: f32, p0: vec4f, p1: vec4f, p2: vec4f, p3: vec4f, Wb: f32, t: f32) -> vec4f {
  let s = clamp(t, 0.0, 1.0) * 16.0;
  let k = min(i32(floor(s)), 15);
  let u = s - f32(k);
  var lv = array<f32, 17>(Wa, p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, Wb);
  let a = lv[k];
  let b = lv[k + 1];
  let D = a - b;
  let even = a + (b - a) * u;
  if (D <= max(${f(HYDRO.fallDrop)}, 3.0 * (Wa - Wb) / 16.0)) { return vec4f(even, even, D * 16.0, D * 16.0); }
  let fallH = min(D, ${f(HYDRO.maxFall)});
  let qf = clamp((u - 0.46) / 0.08, 0.0, 1.0);
  return vec4f(a - fallH * smoothstep(0.46, 0.54, u) - (D - fallH) * u, even, (fallH * 75.0 * qf * (1.0 - qf) + D - fallH) * 16.0, D * 16.0);
}
struct River {
  R: f32, F: f32, xFloor: f32,
  W: f32, ch: f32, w: f32, depth: f32, acc: f32, Ua: f32,
  dir: vec2f, fall: f32,
  L: f32, lakeK: f32, lakeMask: f32,
  vis: f32, cap: f32, Lnear: f32,
  gt: vec3f,   // the walls' slope tensor in the warped plane: (xx, xz, zz)
};
// A channel's half width where the drainage is acc (m; mirror hydro.ts riverHalfW).
fn riverHalfW(acc: f32) -> f32 {
  return 0.5 * min(${f(HYDRO.widthMax)}, ${f(HYDRO.width[0])} + ${f(HYDRO.width[1])} * pow(max(acc - 1.0, 1e-6), ${f(HYDRO.width[2])}));
}
// The meander's offset (m) from a reach's line at along metres from its head, t its share of the reach.
fn meanderOff(along: f32, t: f32, amp: f32, sg: i32) -> f32 {
  if (amp <= 0.0) { return 0.0; }
  return amp * smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t) * vnoise(along * F_MEANDER, 0.5, sg);
}
fn riverField(w: vec2f, seed: i32, k: f32) -> River {
  var o: River;
  o.R = 1e9; o.F = -1e9; o.xFloor = 1e9; o.W = -1e9; o.ch = 1e9; o.w = 0.0; o.depth = 0.0; o.acc = 0.0; o.Ua = 0.0;
  o.dir = vec2f(0.0); o.fall = 0.0; o.L = -1e9; o.lakeK = 0.0; o.lakeMask = 0.0; o.vis = 0.0; o.cap = 1e9; o.Lnear = -1e9;
  o.gt = vec3f(0.0);
  // each reach's (and river lake's) surface here: (surface, slope across, slope along, curvature) and
  // (the way across x, z; how loosely that holds; the lowest it can fall to: its own river's level)
  var sA: array<vec4f, 34>;
  var sB: array<vec4f, 34>;
  var sN: array<vec2i, 34>;   // the reach's two nodes (window-local indices; lakes: none)
  var nR = 0;
  var best = -1;
  var BN = vec2f(0.0);
  let hd = hydroBuf[0];
  let NODE = hd.w;
  let q = w - vec2f(hd.x, hd.y) * NODE;   // relative to the window's origin
  let ci = i32(floor(w.x / NODE));
  let cj = i32(floor(w.y / NODE));
  let rL = ${f(HYDRO.lakeR)} * NODE;
  // the lakes at sinks, and the lakes the rivers widen into (their bowls are cut below)
  var lk: array<vec4f, 9>;   // (distance, lobed radius, level, depth)
  var lkN: array<vec2f, 9>;  // the way out from the centre
  var nLk = 0;
  for (var j = cj - 1; j <= cj + 1; j++) {
    for (var i = ci - 1; i <= ci + 1; i++) {
      let ix = hyIndex(i, j);
      if (ix < 0) { continue; }
      let n0 = hydroBuf[ix];
      let n1 = hydroBuf[ix + 1];
      if (n1.x == ${SINK}.0 && n1.z > -1e8) {
        let d = length(q - n0.xy) / rL;
        o.lakeMask = max(o.lakeMask, 1.0 - smoothstep(1.0, 1.2, d));
        if (d < 1.0 && n1.z > o.L) { o.L = n1.z; o.lakeK = 1.0 - smoothstep(0.85, 1.0, d); }
        if (d < 1.35) { o.Lnear = max(o.Lnear, n1.z); }
      }
      let rr = hydroBuf[ix + 5].w;
      if (rr > 0.0) {
        let dv = q - n0.xy;
        let dist = length(dv);
        if (dist > ${f(2.5 * Math.max(HYDRO.riverLakeR[1], HYDRO.reservoirR[1]))}) { continue; }
        let rl = rr * (1.0 + 0.28 * vnoise(dv.x / rr * 1.3, dv.y / rr * 1.3, seed + 3503));
        lk[nLk] = vec4f(dist, rl, n1.z, ${f(HYDRO.riverLakeD[0])} + ${f(HYDRO.riverLakeD[1])} * rr);
        lkN[nLk] = dv / max(dist, 1e-6);
        nLk++;
        let d = dist / rl;
        o.lakeMask = max(o.lakeMask, 1.0 - smoothstep(1.0, 1.35, d));
        if (d < 1.0 && n1.z > o.L) { o.L = n1.z; o.lakeK = 1.0 - smoothstep(0.85, 1.0, d); }
        if (d < 1.35) { o.Lnear = max(o.Lnear, n1.z); }
      }
    }
  }
  var bestCh = 1e9;
  for (var j = cj - 2; j <= cj + 2; j++) {
    for (var i = ci - 2; i <= ci + 2; i++) {
      let ia = hyIndex(i, j);
      if (ia < 0) { continue; }
      let a0 = hydroBuf[ia];
      let a1 = hydroBuf[ia + 1];
      let code = i32(a1.x);
      if (code > 7) { continue; }
      let ib = hyIndex(i + HY_DI[code], j + HY_DJ[code]);
      if (ib < 0) { continue; }
      let b0 = hydroBuf[ib];
      let Wa = a0.w;
      let Wb = b0.w;
      let acc = a1.y;
      let dv = b0.xy - a0.xy;
      let len = length(dv);
      let u = dv / len;
      let r = q - a0.xy;
      let along = dot(r, u);
      let t = along / len;
      let sg = i32(hash3(i, j, seed + 3301) & 0xffffu);
      // (the river widens as it goes: its head's drainage plus the share of the reach run)
      let wd = riverHalfW(acc + clamp(t, 0.0, 1.0))
        * (1.0 + ${f(HYDRO.widthVary)} * vnoise(along * F_MEANDER * 2.3, 7.5, i32(hash3(i, j, seed + 3302) & 0xffffu)));
      let vis = a1.w; // (the river shows only below a lake: hydro.ts fedOf)
      // (a river whose water does not show cuts no channel: a dry wash, not a ditch)
      let depth = (${f(HYDRO.depth[0])} + ${f(HYDRO.depth[1])} * sqrt(acc)) * vis;
      let slope = (Wa - Wb) / len;
      let amp = ${f(HYDRO.meanderMin)} + ${f(HYDRO.meander - HYDRO.meanderMin)} * (1.0 - smoothstep(0.012, 0.05, (a0.z - b0.z) / len));
      let floorW = select(0.0, amp + wd, vis > 0.0) + ${f(HYDRO.floor[0])} + ${f(HYDRO.floor[1])} * sqrt(acc);
      let p0 = hydroBuf[ia + 2];
      let p1 = hydroBuf[ia + 3];
      let p2 = hydroBuf[ia + 4];
      let p3 = hydroBuf[ia + 5];
      var dd: f32;
      var ch: f32;
      if (t <= 0.0) { dd = length(r); ch = dd; }
      else if (t >= 1.0) { dd = length(q - b0.xy); ch = dd; }
      else {
        let ds = -r.x * u.y + r.y * u.x;
        dd = abs(ds);
        if (dd - floorW > 400.0) {
          let x = dd - floorW;
          let lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
          let lvl = lf.y;
          let cone = lvl + wallRise(k, x, floorW);
          o.R = min(o.R, cone);
          o.F = max(o.F, lvl + ${f(HYDRO.floorRise)} - ${f(HYDRO.riser)} * x);
          o.xFloor = min(o.xFloor, dd - floorW);
          sA[nR] = vec4f(cone, wallSlope(k, x, floorW), lf.w / len, ${f(HYDRO.wallCurve)});
          sB[nR] = vec4f(-u.y, u.x, 0.1, lvl);
          sN[nR] = vec2i(ia, ib);
          nR++;
          continue;
        }
        ch = abs(ds - meanderOff(along, t, amp, sg));
      }
      let lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
      let Wt = lf.x;
      let Wv = select(lf.y, mix(Wt, lf.y, smoothstep(floorW, floorW + 60.0, dd)), vis > 0.0);
      // (a dry valley has no channel: hydro.ts riverField)
      let chV = select(1e9, ch, vis > 0.0);
      let wV = select(0.0, wd, vis > 0.0);
      let cone = Wv + valleyShape(dd, chV, wV, depth, floorW, k);
      o.R = min(o.R, cone);
      var fillH = ${f(HYDRO.floorRise)} - ${f(HYDRO.riser)} * (dd - floorW);
      if (dd < floorW) { fillH = 0.3 + ${f(HYDRO.floorRise - 0.3)} * (dd / floorW) * (dd / floorW); }
      o.F = max(o.F, Wv + fillH);
      o.xFloor = min(o.xFloor, dd - floorW);
      // this reach's surface (its water in its channel, else its ground), how fast it changes across its
      // valley and along it, and which way across is (round a node: outward from it)
      let isEnd = t <= 0.0 || t >= 1.0;
      var fallT = 0.0;
      if (!isEnd) { fallT = max(lf.z, lf.w) / len; }
      var nrm = vec2f(-u.y, u.x);
      if (isEnd) { nrm = select(q - b0.xy, r, t <= 0.0) / max(dd, 1e-6); }
      sN[nR] = vec2i(ia, ib);
      if (vis > 0.0 && ch < wd + 0.3) { sA[nR] = vec4f(max(cone, Wt), 0.0, fallT, 0.0); sB[nR] = vec4f(nrm, 0.1, min(Wt, Wv)); }
      else {
        sA[nR] = vec4f(cone, valleySlope(dd, chV, wV, depth, floorW, k), fallT, ${f(HYDRO.wallCurve)});
        sB[nR] = vec4f(nrm, select(0.1, 0.35, dd < floorW + 60.0), min(Wt, Wv));
      }
      if (vis > 0.0 && ch - wd < bestCh) {
        bestCh = ch - wd; best = nR;
        o.W = Wt; o.ch = ch; o.w = wd; o.depth = depth; o.acc = acc; o.Ua = a0.z; o.vis = vis;
        o.dir = u; o.fall = slope;
        // across this channel: across its meander's own heading (round a node: outward)
        if (isEnd) { BN = nrm; }
        else {
          let dm = 0.5 * (meanderOff(along + 1.0, t + 1.0 / len, amp, sg) - meanderOff(along - 1.0, t - 1.0 / len, amp, sg));
          let hdg = normalize(vec2f(u.x - u.y * dm, u.y + u.x * dm));
          BN = vec2f(-hdg.y, hdg.x);
        }
      }
      nR++;
    }
  }
  // the river lakes' bowls: below the water, a shore rising gently, then walls (they are valleys too)
  for (var m = 0; m < nLk; m++) {
    let L4 = lk[m];
    let fl = L4.y + ${f(HYDRO.floor[0] + 20)};
    let cone = L4.z + valleyShape(L4.x, L4.x, L4.y, L4.w, fl, k);
    o.R = min(o.R, cone);
    o.xFloor = min(o.xFloor, L4.x - fl);
    if (L4.x < L4.y) { sA[nR] = vec4f(max(cone, L4.z), 0.0, 0.0, ${f(HYDRO.wallCurve)}); }
    else { sA[nR] = vec4f(cone, valleySlope(L4.x, L4.x, L4.y, L4.w, fl, k), 0.0, ${f(HYDRO.wallCurve)}); }
    sB[nR] = vec4f(lkN[m], 0.3, L4.z);
    sN[nR] = vec2i(-1, -2);
    nR++;
  }
  // the walls' slope: each reach's slope across, weighted by a soft minimum over the surfaces (hydro.ts riverField)
  var gw = 0.0;
  var gt = vec3f(0.0);
  for (var r = 0; r < nR; r++) {
    let d = sA[r].x - o.R;
    if (d > ${f(10 * HYDRO.wallSoft)}) { continue; }
    let wt = exp(-d / ${f(HYDRO.wallSoft)});
    let nn = sB[r].xy;
    gw += wt;
    gt += wt * sA[r].y * sA[r].y * vec3f(nn.x * nn.x, nn.x * nn.y, nn.y * nn.y);
  }
  if (gw > 0.0) { o.gt = gt / gw; }
  // the nearest channel's water stays under every other reach's surface out to its far bank (that surface
  // falls toward the bank by its slope along the line across the channel, but never below its own river)
  if (best >= 0) {
    let D = o.w + o.ch + 1.0;
    var cap = 1e9;
    for (var r = 0; r < nR; r++) {
      if (r == best) { continue; }
      let A = sA[r];
      let Bv = sB[r];
      let cn = abs(dot(BN, Bv.xy));
      let cu = abs(BN.x * Bv.y - BN.y * Bv.x);
      let fall = (cn * A.y + cu * A.z + Bv.z * (A.y + A.z) + A.w * D) * D;
      // (reaches meeting this one at a node run on into it, level with it where they meet)
      let Nr = sN[r];
      let Nb = sN[best];
      let kin = Nr.x == Nb.x || Nr.x == Nb.y || Nr.y == Nb.x || Nr.y == Nb.y;
      cap = min(cap, max(A.x - fall, Bv.w - (select(cu, 0.0, kin) + 0.05) * A.z * D));
    }
    o.cap = cap - 0.1;
  }
  return o;
}

// The regions' detail over the uplift at world g, warped w, before any mask, and the towers' height
// (mirror terrainMath.ts landDetail: with river the towers are left out of the detail).
fn landDetail(g: vec2f, w: vec2f, up: Uplift, p: Landform, seed: i32, dp: f32, river: bool) -> vec2f {
  var detail = p.hills * hillsAt(w * F_MACRO, seed, dp);
  let rs = min(1.0, 2.0 * up.range + 1.6 * up.massif);
  if (p.ridge > 0.0) { detail += p.ridge * (0.3 + 0.7 * rs) * ridgesAt(w, seed, dp); }
  // (karst, dunes and mesas only where their region rules: a few tiers in a meadow would be cliffs)
  let kw = smoothstep(0.35, 0.75, p.karst);
  var kc = 0.0;
  var kt = 0.0;
  if (kw > 0.001) {
    let wK = clamp(2.0 - 2.0 * dp / 150.0, 0.0, 1.0);
    var t = TOWER_MEAN;
    if (wK > 0.0) {
      let kk = karstTowers(g, seed, dp);
      t = wK * kk.x + (1.0 - wK) * TOWER_MEAN;
      kc = kw * wK * kk.y;
    }
    kt = kw * t;
    if (!river) { detail += kt; }
  }
  let dw = smoothstep(0.35, 0.75, p.dune);
  if (dw > 0.001) { detail += dw * dunesE(g, seed, dp); }
  let mw = smoothstep(0.35, 0.75, p.mesa);
  if (mw > 0.001) { let me = mesasE(g, seed, dp, mw, p.mesa); detail += me.x; kc = max(kc, me.y); }
  if (p.erosion > 0.01) {
    // (down the land itself, laid out in the world: terrainMath.ts landDetail)
    let rk = select(0.0, p.ridge * (0.3 + 0.7 * rs), p.ridge > 0.0);
    let b0 = erosionLand(w, up.U, seed, p.hills, rk);
    let gr = vec2f(erosionBase(g + vec2f(ERO_STEP, 0.0), seed, p.hills, rk) - b0, erosionBase(g + vec2f(0.0, ERO_STEP), seed, p.hills, rk) - b0) / ERO_STEP;
    // (only on flanks and aprons: terrainMath.ts landDetail)
    let S = length(gr);
    let keep = smoothstep(GULLY_SLOPE.x, GULLY_SLOPE.y, S) * (1.0 - smoothstep(GULLY_SLOPE.z, GULLY_SLOPE.w, S)) * (1.0 - kc);
    if (keep > 0.0) { detail += p.erosion * keep * erosionDelta(g, up.U + detail, gr, 0.0, dp); } // (no fade target: terrainMath.ts landDetail)
  }
  return vec2f(detail, kt);
}
// The land at world g before its rivers, lakes, marshes, ponds and the sea (mirror terrainMath.ts landBare).
fn landBare(g: vec2f, dp: f32, seed: i32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let up = upliftAt(w, seed);
  let rg = regionsFrom(g, w, up, seed);
  return up.U + landDetail(g, w, up, blendLandform(rg), seed, dp, false).x * up.taper;
}
// A pond whose reach (pondReach radii) holds world g (mirror terrainMath.ts pondAt): (level or -1e9, distance
// from its centre over its radius, depth).
fn pondAt(g: vec2f, seed: i32) -> vec4f {
  let cell = 1.0 / F_POND;
  let ix = i32(floor(g.x * F_POND));
  let iz = i32(floor(g.y * F_POND));
  let h = hash3(ix, iz, seed + 3401);
  let u = f32(h >> 24u) / 255.0;
  let R = ${f(HYDRO.pondR[0])} + ${f(HYDRO.pondR[1] - HYDRO.pondR[0])} * u * u * u;
  let inset = ${f(HYDRO.pondReach * 1.3)} * R / cell;
  let c = (vec2f(f32(ix), f32(iz)) + inset + (1.0 - 2.0 * inset) * vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) / 4095.0) * cell;
  let q = (g - c) / R;
  let d = length(g - c) / (R * (1.0 + 0.3 * vnoise(q.x * 1.2, q.y * 1.2, seed + 3403)));
  if (d >= ${f(HYDRO.pondReach)}) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  let h2 = hash3(ix, iz, seed + 3402);
  let w = domainWarp(c.x, c.y, seed, 0.0);
  let up = upliftAt(w, seed);
  if (up.C < COAST_LEVEL + 0.02) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  let rg = regionsFrom(c, w, up, seed);
  var chance = 0.0;
  for (var i = 0; i < ${NB}; i++) {
    if (rg.w[i] > 0.0) { chance += rg.w[i] * LF_TABLE[i * ${NK} + ${LANDFORM_KEYS.indexOf('pond')}]; }
  }
  if (f32(h2 & 0xffffu) / 65535.0 >= chance) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  // the land itself at its centre (the pond lies just under it) and round its rim: a pond needs flat land
  let ec = up.U + landDetail(c, w, up, blendLandform(rg), seed, 4.0, false).x * up.taper;
  var ring = array<vec2f, 3>(vec2f(1.0, 0.0), vec2f(-0.5, 0.8660254037844386), vec2f(-0.5, -0.8660254037844386));
  for (var k = 0; k < 3; k++) {
    if (abs(landBare(c + 1.25 * R * ring[k], 4.0, seed) - ec) > 0.06 * R) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  }
  // away from rivers and lakes
  let rv = riverField(w, seed, 0.12);
  if (rv.xFloor < 2.0 * R + 40.0 || rv.lakeMask > 0.0) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  return vec4f(ec - ${f(HYDRO.pondSink)}, d, ${f(HYDRO.pondD[0])} + ${f(HYDRO.pondD[1] - HYDRO.pondD[0])} * f32(h2 >> 16u) / 65535.0, R);
}

// The soft ceiling: heights ease toward MAXH − 10 m above CEIL_START (ground and water alike).
fn ceilH(h: f32) -> f32 {
  if (h > CEIL_START) { return CEIL_START + CEIL_SPAN * tanh((h - CEIL_START) / CEIL_SPAN); }
  return h;
}
// The land's surface at world g (mirror terrainMath.ts landSurface).
struct Surface {
  e: f32,          // the ground (prefiltered for dp)
  w: f32,          // the water's (or lava's) surface (dp-free)
  riverK: f32,     // how much of it is a river's channel
  lakeK: f32,      // a lake's (1 deep inside its ring)
  flow: vec2f,     // the flow's velocity (m/s, world xz)
  lava: f32,       // the liquid is lava (a volcano's crater and its rivers)
  shore: f32,      // the level of the water beside this point (a river's bank, a lake, a pond, the coast, a marsh; -1e9: none)
};
fn landSurface(g: vec2f, dp: f32, seed: i32) -> Surface {
  return landCore(g, dp, seed, pondAt(g, seed));
}
// landSurface with the pond (level or -1e9, distance in radii, depth, radius) given.
fn landCore(g: vec2f, dp: f32, seed: i32, pond: vec4f) -> Surface {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let up = upliftAt(w, seed);
  let rg = regionsFrom(g, w, up, seed);
  let p = blendLandform(rg);
  let rv = riverField(w, seed, p.k);
  var crater = 1.0;
  if (up.lava > -1e8) { crater = smoothstep(${f(CRATER_RIM * 0.9)}, ${f(CRATER_RIM * 1.3)}, up.rr); }
  let m = smoothstep(0.0, 50.0, rv.xFloor) * (1.0 - rv.lakeMask) * crater;
  // where marshes lie (not in a lake's ring, nor in a river's channel; only on flat land: marsh pools
  // stepping up a hillside would stand at many levels)
  var flat = p.flat * (1.0 - rv.lakeMask);
  if (rv.vis > 0.0) { flat *= smoothstep(rv.w + 2.0, rv.w + 8.0, rv.ch); }
  if (flat > 0.0) {
    let gU = vec2f(upliftAt(w + vec2f(8.0, 0.0), seed).U - up.U, upliftAt(w + vec2f(0.0, 8.0), seed).U - up.U) / 8.0;
    flat *= 1.0 - smoothstep(0.012, 0.035, length(gU));
  }

  var detail = 0.0;
  var towers = 0.0;
  if (m > 0.0) {
    let ld = landDetail(g, w, up, p, seed, dp, true);
    detail = ld.x * m * up.taper;
    // (the towers under the ceiling a river's floor puts over them: terrainMath.ts landCore)
    if (ld.y > 0.0) { towers = max(0.0, min(ld.y, KARST_CEIL * (min(rv.xFloor, 1000.0) - KARST_BANK) * (1.0 - rv.lakeMask))) * up.taper * crater; }
    // (round a river the detail only rises: a hollow beside it would leave the river standing over it)
    let near = 1.0 - smoothstep(50.0, 450.0, rv.xFloor);
    if (near > 0.0) { detail += (0.5 * (detail + sqrt(detail * detail + 36.0)) - 3.0 - detail) * near; }
    // (and round a marsh: its pools lie just under the smooth land, and a hollow at its edge would spill them)
    let nearM = smoothstep(0.0, 0.3, flat);
    if (nearM > 0.0) { detail += (0.5 * (detail + sqrt(detail * detail + 0.36)) - 0.3 - detail) * nearM; }
  }
  var e0 = up.U + detail;
  // the canyons: where mesas rule, a valley's steep walls cut through the strata (terrainMath.ts landCore)
  var Rs = rv.R;
  let mwC = smoothstep(0.35, 0.75, p.mesa);
  if (mwC > 0.0 && rv.R < max(e0, rv.F)) {
    var cw = mwC * smoothstep(${f(CANYON.rim[0])}, ${f(CANYON.rim[1])}, rv.xFloor) * (1.0 - rv.lakeMask);
    if (cw > 0.0) {
      // (the wall's slope in the world: the valleys are cut in the warped plane)
      let J = warpJ(g, seed);
      let M = vec3f(J.x * J.x + J.y * J.y, J.x * J.z + J.y * J.w, J.z * J.z + J.w * J.w);
      let s = sqrt(max(0.0, M.x * rv.gt.x + 2.0 * M.y * rv.gt.y + M.z * rv.gt.z));
      cw *= smoothstep(${f(CANYON.steep[0])}, ${f(CANYON.steep[1])}, s);
      if (cw > 0.0) {
        let dip = ${f(CANYON.dip)} * vnoise(g.x * F_MESA, g.y * F_MESA, seed + 2613);
        Rs = rv.R + cw * canyonRaise(rv.R + dip, s, s * max(dp, 0.5), seed);
      }
    }
  }
  e0 = min(max(e0, rv.F), Rs) + towers;
  let vis = rv.vis;
  // a crater's rim held above its lava lake (valleys and gullies would breach it)
  if (up.lava > -1e8) {
    let band = smoothstep(${f(CRATER_RIM * 0.82)}, ${f(CRATER_RIM)}, up.rr) * (1.0 - smoothstep(${f(CRATER_RIM * 1.3)}, ${f(CRATER_RIM * 1.6)}, up.rr));
    if (up.lava + 4.0 > e0) { e0 += (up.lava + 4.0 - e0) * band; }
  }
  // the marshes
  var marshW = -1e9;
  if (flat > 0.0) {
    let wH = clamp(2.0 - 2.0 * dp * F_HUMMOCK, 0.0, 1.0);
    let wH2 = clamp(2.0 - 2.0 * dp * F_HUMMOCKFINE, 0.0, 1.0);
    let hum = 1.4 * (wH * vnoise(g.x * F_HUMMOCK, g.y * F_HUMMOCK, seed + 2801) + 0.5 * wH2 * vnoise(g.x * F_HUMMOCKFINE, g.y * F_HUMMOCKFINE, seed + 2802)) - 0.2;
    let lvl = max(up.U - 1.2, SEA);
    e0 += (lvl + hum - e0) * flat;
    marshW = lvl + 0.25;
    // pools lie only in the marsh's heart; round it the hummocks stand a little over them (a rim faded in
    // from the marsh's very edge), so no pool spills where the land beyond lies low
    e0 += max(0.0, marshW + 0.05 - e0) * smoothstep(0.0, ${f(0.05)}, flat) * (1.0 - smoothstep(${f(0.8)}, ${f(0.82)}, flat));
  }
  // a pond in a meadow hollow: its bowl, and a rim to hold it
  var pondL = -1e9;
  if (pond.x > -1e8) {
    if (pond.y < 1.0) {
      e0 = pond.x - pond.z * (1.0 - smoothstep(0.45, 0.92, pond.y)) + ${f(HYDRO.pondRim)} * smoothstep(0.8, 1.0, pond.y);
      pondL = pond.x;
    } else {
      e0 += (pond.x + ${f(HYDRO.pondRim)} - e0) * (1.0 - smoothstep(1.0, ${f(HYDRO.pondReach)}, pond.y));
    }
  }
  // the coast, the sea floor, the soft ceiling; a cliffed coast cut in one profile (terrainMath.ts SEA_CLIFF)
  var land = smoothstep(COAST_LEVEL - 0.004 - 0.03 * (1.0 - up.cliff), COAST_LEVEL + 0.002, up.C);
  let seaFloor = SEA - 6.0 - ${f(LANDS.coast.shelf)} * (1.0 - smoothstep(COAST_LEVEL - 0.12, COAST_LEVEL, up.C));
  var e = seaFloor + (e0 - seaFloor) * land;
  let cw = smoothstep(${f(SEA_CLIFF.weight[0])}, ${f(SEA_CLIFF.weight[1])}, up.cliff) * (1.0 - smoothstep(${f(0.6 * SEA_CLIFF.band)}, ${f(SEA_CLIFF.band)}, abs(up.C - COAST_LEVEL)));
  if (cw > 0.0) {
    // the distance in from the coastline: C's rise over its world gradient
    let cX = continentAt(domainWarp(g.x + ${f(SEA_CLIFF.h)}, g.y, seed, 0.0), seed);
    let cZ = continentAt(domainWarp(g.x, g.y + ${f(SEA_CLIFF.h)}, seed, 0.0), seed);
    let d = (up.C - COAST_LEVEL) * ${f(SEA_CLIFF.h)} / max(length(vec2f(cX - up.C, cZ - up.C)), 1e-9);
    let apron = ${f(SEA_CLIFF.apron)} * smoothstep(0.45, 0.85, n01(w, F_TOWER, seed + 2703));
    let d0 = d - apron - ${f(SEA_CLIFF.toe)};
    let base = SEA - ${f(SEA_CLIFF.platform)};
    let wall = base + ${f(SEA_CLIFF.wall)} * d0;
    let talus = base + ${f(SEA_CLIFF.talus)} * (d0 + apron);
    let flr = max(seaFloor, base + ${f(SEA_CLIFF.shelf)} * min(d, 0.0)) - ${f(SEA_CLIFF.wall)} * max(d0, 0.0);
    let kr = ${f(SEA_CLIFF.rim)} + ${f(SEA_CLIFF.wall)} * dp;
    let kt = 1.0 + ${f(SEA_CLIFF.wall)} * dp;
    let kf = 0.3 + ${f(SEA_CLIFF.talus)} * dp;
    let eC = ksmax(flr, ksmin(e0, ksmax(talus, wall, kt), kr), kf);
    e += (eC - e) * cw;
    land += (smoothstep(0.0, apron + ${f(SEA_CLIFF.toe)}, d) - land) * cw;
  }
  // the water
  var o: Surface;
  o.e = ceilH(e); o.riverK = 0.0; o.lakeK = 0.0; o.flow = vec2f(0.0); o.lava = 0.0;
  var wl = e - DRY_TABLE;
  var speed = 0.0;
  let hot = up.vol.z > 0.0 && length(w - up.vol.xy) < ${f(0.6 * LANDS.volcano.cone)} * up.vol.z;
  // the river's water: its level, never over a bank another valley has cut lower; toward the sea it
  // sinks with its banks (they ease down to the sea floor together)
  let riverW = seaFloor + (min(rv.W, rv.cap) - seaFloor) * land;
  if (vis > 0.0 && rv.ch < rv.w + 0.3 && riverW > e && !(hot && rv.fall > 0.3)) {
    wl = max(wl, riverW);
    o.riverK = vis;
    speed = clamp(0.35 + 26.0 * rv.fall, 0.3, 4.5);
  }
  let lakeL = select(-1e9, seaFloor + (rv.L - seaFloor) * land, rv.L > -1e8);
  if (lakeL > e) { wl = max(wl, lakeL); o.lakeK = rv.lakeK; speed = 0.0; }
  if (pondL > e) { wl = max(wl, pondL); speed = 0.0; }
  if (e < SEA && up.C < COAST_LEVEL + ${f(LANDS.coast.inland)}) { wl = max(wl, SEA); }
  if (up.lava > e && up.rr < ${f(CRATER_RIM)}) { wl = max(wl, up.lava); speed = 0.0; }
  let marshWc = select(-1e9, seaFloor + (marshW - seaFloor) * land, marshW > -1e8);
  if (flat > ${f(0.8)} && marshWc > e) { wl = max(wl, marshWc); }
  // (the sea fills only what lies below its level by the coast: water eased toward it would climb the shore)
  o.w = ceilH(wl);
  // the level of the water beside this point, if any: what shores are judged from
  var shore = -1e9;
  if (vis > 0.0 && rv.ch < rv.w + 14.0 && !(hot && rv.fall > 0.3)) { shore = riverW; }
  if (rv.Lnear > -1e8) { shore = max(shore, seaFloor + (rv.Lnear - seaFloor) * land); }
  if (pond.x > -1e8 && pond.y < ${f(HYDRO.pondReach)}) { shore = max(shore, pond.x); }
  if (up.C < COAST_LEVEL + ${f(LANDS.coast.inland)}) { shore = max(shore, SEA); }
  if (flat > ${f(0.8)}) { shore = max(shore, marshWc); }
  o.shore = select(-1e9, ceilH(shore), shore > -1e8);
  if (speed > 0.0) {
    // the reach's direction is in the warped plane: back to the world's through the warp's Jacobian
    let ex = domainWarp(g.x + 1.0, g.y, seed, 0.0) - w;
    let ez = domainWarp(g.x, g.y + 1.0, seed, 0.0) - w;
    let det = ex.x * ez.y - ez.x * ex.y;
    var wd = vec2f(ez.y * rv.dir.x - ez.x * rv.dir.y, -ex.y * rv.dir.x + ex.x * rv.dir.y) / det;
    wd = wd / max(length(wd), 1e-6);
    o.flow = wd * speed * land;
  }
  if (up.vol.z > 0.0 && wl > e - 0.01) {
    o.lava = 1.0 - smoothstep(0.45, 0.6, length(w - up.vol.xy) / (${f(LANDS.volcano.cone)} * up.vol.z));
  }
  return o;
}

// The land's broad shape at world g (m): uplift and rolling hills, prefiltered for dp (mirror terrainMath.ts landCoarse).
fn landCoarse(g: vec2f, dp: f32, seed: i32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let up = upliftAt(w, seed);
  let rg = regionsFrom(g, w, up, seed);
  var hills = 0.0;
  for (var i = 0; i < ${NB}; i++) { if (rg.w[i] > 0.0) { hills += rg.w[i] * LF_TABLE[i * ${NK}]; } }
  return up.U + hills * hillsAt(w * F_MACRO, seed, dp);
}

// ---- master elevation E(x, z) — spec §2 (mirror terrainMath.elevation) --------
fn elevation(g: vec2f, dp: f32, seed: i32) -> f32 {
  return landSurface(g, dp, seed).e;
}

// ---- §9 climate ----------------------------------------------------------------
fn seaTemperature(g: vec2f, seed: i32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  return T_BASE + T_AMP * (vnoise(w.x * F_TEMP, w.y * F_TEMP, seed + 1201) + T_FINE * vnoise(w.x * F_TEMPFINE, w.y * F_TEMPFINE, seed + 1202))
    - L_LAT * abs(g.y);
}
fn temperature(g: vec2f, e: f32, seed: i32) -> f32 {
  return seaTemperature(g, seed) - L_ALT * max(e, 0.0);
}
fn moisture(g: vec2f, seed: i32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let C = continentAt(w, seed);
  return M_BASE + M_AMP * (vnoise(w.x * F_MOISTR, w.y * F_MOISTR, seed + 1301) + M_FINE * vnoise(w.x * F_MOISTFINE, w.y * F_MOISTFINE, seed + 1302))
    + M_OCEAN * (1.0 - smoothstep(COAST_LEVEL, COAST_LEVEL + 0.12, C));
}

// ---- the ground's biome look (biomes.ts LOOK) ------------------------------------
// Per biome: soil, grass at its lushest and dried, rock; (cover, blade height, strata, wetness);
// (frost, flowers). The WebGPU raster bakes their blend per clipmap lattice point (terrain.wgsl.ts
// biomeShader); the ray-march renderer evaluates it per pixel (groundAt).
const LK_SOIL = array<vec3f, ${NB}>(${L_LOOK.map((l) => v3(l.soil)).join(', ')});
const LK_GRASS = array<vec3f, ${NB}>(${L_LOOK.map((l) => v3(l.grass)).join(', ')});
const LK_DRY = array<vec3f, ${NB}>(${L_LOOK.map((l) => v3(l.dry)).join(', ')});
const LK_ROCK = array<vec3f, ${NB}>(${L_LOOK.map((l) => v3(l.rock)).join(', ')});
const LK_A = array<vec4f, ${NB}>(${L_LOOK.map((l) => `vec4f(${f(l.cover)}, ${f(l.blade)}, ${f(l.strata)}, ${f(l.wet)})`).join(', ')});
const LK_B = array<vec2f, ${NB}>(${L_LOOK.map((l) => `vec2f(${f(l.frost)}, ${f(l.bloom)})`).join(', ')});
const SNOW_T: f32 = ${f(CLIMATE.snowT)};

struct Biome { col: vec3f, veg: f32 };

// The ground's colour at world g, per pixel (height e, slope, the water's level wl): grass over soil,
// rock on the steep (banded in strata), sand on low gentle shores, snow above the snow line.
fn groundAt(g: vec2f, e: f32, slope: f32, wl: f32, seed: i32) -> Biome {
  let r = landRegions(g, domainWarp(g.x, g.y, seed, 0.0), seed);
  var soil = vec3f(0.0);
  var grass = vec3f(0.0);
  var dry = vec3f(0.0);
  var rock = vec3f(0.0);
  var a = vec4f(0.0);
  var b = vec2f(0.0);
  for (var k = 0; k < ${NB}; k++) {
    let w = r.w[k];
    soil += w * LK_SOIL[k]; grass += w * LK_GRASS[k]; dry += w * LK_DRY[k]; rock += w * LK_ROCK[k];
    a += w * LK_A[k]; b += w * LK_B[k];
  }
  let patchN = 0.5 + 0.5 * vnoise(g.x * 0.045, g.y * 0.045, seed + 7777);
  let patchHue = 0.5 + 0.5 * vnoise(g.x * 0.033, g.y * 0.033, seed + 7707);
  let grassCol = mix(grass, dry, clamp(patchHue * 1.3 - 0.45 + (0.5 - r.M) * 0.8, 0.0, 1.0) * 0.75);
  let dens = a.x * smoothstep(0.15, 0.55, patchN + 0.25 * a.x) * (1.0 - smoothstep(0.7, 1.1, slope));
  var col = mix(soil, grassCol * 0.8, clamp(dens * 1.2, 0.0, 0.9));
  col = mix(col, rock, smoothstep(0.72, 1.15, slope + (1.0 - a.x) * 0.22));
  let above = e - wl;
  let seaShore = 1.0 - smoothstep(SEA + 0.5, SEA + 1.5, wl);
  let beach = (1.0 - smoothstep(0.3, 1.8, above)) * (1.0 - smoothstep(0.35, 0.7, slope)) * (1.0 - 0.75 * a.w) * max(seaShore, 0.8 * (1.0 - a.x));
  col = mix(col, mix(vec3f(0.6, 0.5, 0.34), soil, 0.3), beach * 0.85);
  col *= 1.0 - 0.4 * ((1.0 - smoothstep(0.0, 1.5, above)) * 0.55 + a.w * 0.35);
  let snowAlt = (r.T0 - SNOW_T - b.x) / L_ALT;
  col = mix(col, vec3f(${SNOW_ALBEDO.map(f).join(', ')}), smoothstep(snowAlt - 15.0, snowAlt + 15.0, e) * (1.0 - smoothstep(0.85, 1.35, slope)));
  return Biome(col, dens);
}

// ---- §20–§22 sky and Beer-Lambert aerial perspective -----------------------------
fn skyColor(rd: vec3f) -> vec3f {
  let sd = max(dot(rd, SUN), 0.0);
  let k = pow(max(rd.y, 0.0), 0.55);
  var sky = mix(SKY_HOR, SKY_ZEN, k);
  sky = mix(SKY_HOR * 1.04, sky, smoothstep(-0.08, 0.06, rd.y));
  let sd2 = sd * sd;
  let sd4 = sd2 * sd2;
  let sd8 = sd4 * sd4;
  var sd90 = 0.0;
  if (sd > 0.8) {
    let sd16 = sd8 * sd8;
    let sd32 = sd16 * sd16;
    let sd64 = sd32 * sd32;
    sd90 = sd64 * sd16 * sd8 * sd2;
  }
  sky += SUN_COL * 0.10 * sd8 + SUN_COL * 0.30 * sd90;
  sky += SUN_COL * 6.0 * smoothstep(0.99955, 0.99985, sd);
  return sky;
}

fn applyFog(col: vec3f, t: f32, h: f32, rd: vec3f) -> vec3f {
  let od = SIGMA_FOG * t * exp(-max(h, 0.0) * (1.0 / FOG_H));
  if (od < 0.0005) { return col; }
  let inscatter = (skyColor(rd) + SKY_HOR) * 0.5;
  if (od > 5.82) { return inscatter; }
  let T = exp(-vec3f(0.58, 0.72, 0.95) * od);
  if (T.b < 0.004) { return inscatter; }
  return col * T + inscatter * (1.0 - T);
}
`;
