/**
 * W(x,y) world function in WGSL — the WebGPU twin of demoShader.ts / terrainMath.ts.
 * ---------------------------------------------------------------------------
 * A line-for-line port of the GLSL terrain functions (hashing, quintic value
 * noise, fBm / ridged octaves with Nyquist weights, Worley valleys, domain
 * warp, region Voronoi, Zonai anomalies, dunes, climate, biome colours, sky and
 * fog). Constants are interpolated from src/lib/constants.ts, the single
 * source of truth, so this twin cannot drift from the CPU one. The GLSL's
 * rounded literals (sun direction, ridged amplitude sum) are kept verbatim so
 * WebGPU frames can be diffed against the WebGL2 renderer.
 *
 * Coordinates: y-up, the field is E(x, z). Callers pass fp32 globals
 * `g = chunk_local + chunk_origin`, exactly like the WebGL2 shader.
 */

import {
  F, MACRO, MOUNTAIN, WARP, VALLEY, CONTINENT, REGION, ANOMALY, CLIMATE, LIGHT,
  MAX_WORLD_H, MAX_DIST,
} from '../constants';

/** A WGSL float literal for a JS number (parsed by WGSL to the nearest f32). */
export function wgslF32(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`non-finite WGSL literal: ${x}`);
  const s = String(x);
  return /[.eE]/.test(s) ? s : `${s}.0`;
}

const f = wgslF32;
const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;

export const WORLD_WGSL = /* wgsl */ `
// ---- contract constants (from src/lib/constants.ts) -------------------------
const MAXH: f32 = ${f(MAX_WORLD_H)};
const MAXDIST: f32 = ${f(MAX_DIST)};
const F_MACRO: f32 = ${f(F.macro)};
const F_RIDGE: f32 = ${f(F.ridge)};
const F_MASK: f32 = ${f(F.mask)};
const F_WARP: f32 = ${f(F.warp)};
const F_VALLEY: f32 = ${f(F.valley)};
const F_CONT: f32 = ${f(F.cont)};
const F_MOIST: f32 = ${f(F.moist)};
const F_DUNE: f32 = ${f(F.dune)};
const F_CLOUD: f32 = ${f(F.cloud)};
const VEG_WL: f32 = ${f(F.veg)};
const H_BASE: f32 = ${f(MACRO.hBase)};
const H_MOUNT: f32 = ${f(MOUNTAIN.hMount)};
const MASK_LO: f32 = ${f(MOUNTAIN.maskLo)};
const MASK_HI: f32 = ${f(MOUNTAIN.maskHi)};
const S_WARP: f32 = ${f(WARP.strength)};
const C_DEPTH: f32 = ${f(VALLEY.cDepth)};
const W_VALLEY: f32 = ${f(VALLEY.wValley)};
const W_LEVEL: f32 = ${f(VALLEY.wLevel)};
const CONT_LO: f32 = ${f(CONTINENT.lo)};
const CONT_HI: f32 = ${f(CONTINENT.hi)};
const CONT_MIN: f32 = ${f(CONTINENT.minFactor)};
const REGION_CELL: f32 = ${f(REGION.cell)};
const REG_BLEND_LO: f32 = ${f(REGION.blendLo)};
const REG_BLEND_HI: f32 = ${f(REGION.blendHi)};
const REG_HILLS: f32 = ${f(REGION.hillsFactor)};
const REG_MTN: f32 = ${f(REGION.mountainFactor)};
const REG_DUNE: f32 = ${f(REGION.duneFactor)};
const DUNE_H: f32 = ${f(REGION.duneHeight)};
const ANOM_CELL: f32 = ${f(ANOMALY.cell)};
const P_ANOM: f32 = ${f(ANOMALY.probability)};
const ANOM_R_IN: f32 = ${f(ANOMALY.rIn)};
const ANOM_R_OUT: f32 = ${f(ANOMALY.rOut)};
const TERRACE_STEP: f32 = ${f(ANOMALY.terraceStep)};
const TERRACE_H: f32 = ${f(ANOMALY.terraceHeight)};
const SPIRE_H: f32 = ${f(ANOMALY.spireHeight)};
const SPIRE_R: f32 = ${f(ANOMALY.spireRadius)};
const SPIRAL_AMP: f32 = ${f(ANOMALY.spiralAmp)};
const SPIRAL_TURNS: f32 = ${f(ANOMALY.spiralTurns)};
const SPIRAL_RAD: f32 = ${f(ANOMALY.spiralRadial)};
const T_BASE: f32 = ${f(CLIMATE.tBase)};
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

// ---- §12 Zonai anomalies (terraces + spire + spiral), pruned neighbours ------
struct AnomBest { f1: f32, bc: vec2f, btype: f32 };

fn anomTry(best: AnomBest, cxi: i32, cyi: i32, off: vec2f, fr: vec2f, seed: i32) -> AnomBest {
  let h = hash3(cxi, cyi, seed + 400);
  let p = vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) * INV_4095;
  let d = off + p - fr;
  let d2 = dot(d, d);
  if (d2 < best.f1) {
    return AnomBest(d2, vec2f(f32(cxi), f32(cyi)) + p, f32(h >> 24u) * (1.0 / 255.0));
  }
  return best;
}

fn anomalyE(g: vec2f, seed: i32) -> f32 {
  let p_in = g * (1.0 / ANOM_CELL);
  let ip = floor(p_in);
  let fr = p_in - ip;
  let cxi0 = i32(ip.x);
  let cyi0 = i32(ip.y);
  let h0 = hash3(cxi0, cyi0, seed + 400);
  let p0 = vec2f(f32(h0 & 0xfffu), f32((h0 >> 12u) & 0xfffu)) * INV_4095;
  let d0 = p0 - fr;
  var b = AnomBest(dot(d0, d0), vec2f(f32(cxi0), f32(cyi0)) + p0, f32(h0 >> 24u) * (1.0 / 255.0));

  let R_OUT_CELL_SQ = (ANOM_R_OUT / ANOM_CELL) * (ANOM_R_OUT / ANOM_CELL);
  let fx0_2 = fr.x * fr.x;
  let fx1_2 = (1.0 - fr.x) * (1.0 - fr.x);
  let fy0_2 = fr.y * fr.y;
  let fy1_2 = (1.0 - fr.y) * (1.0 - fr.y);
  if (fx1_2 < b.f1 && fx1_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0 + 1, cyi0, vec2f(1.0, 0.0), fr, seed); }
  if (fy1_2 < b.f1 && fy1_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0, cyi0 + 1, vec2f(0.0, 1.0), fr, seed); }
  if (fx0_2 < b.f1 && fx0_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0 - 1, cyi0, vec2f(-1.0, 0.0), fr, seed); }
  if (fy0_2 < b.f1 && fy0_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0, cyi0 - 1, vec2f(0.0, -1.0), fr, seed); }
  if (fx1_2 + fy1_2 < b.f1 && fx1_2 + fy1_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0 + 1, cyi0 + 1, vec2f(1.0, 1.0), fr, seed); }
  if (fx0_2 + fy1_2 < b.f1 && fx0_2 + fy1_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0 - 1, cyi0 + 1, vec2f(-1.0, 1.0), fr, seed); }
  if (fx1_2 + fy0_2 < b.f1 && fx1_2 + fy0_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0 + 1, cyi0 - 1, vec2f(1.0, -1.0), fr, seed); }
  if (fx0_2 + fy0_2 < b.f1 && fx0_2 + fy0_2 < R_OUT_CELL_SQ) { b = anomTry(b, cxi0 - 1, cyi0 - 1, vec2f(-1.0, -1.0), fr, seed); }

  if (b.btype >= P_ANOM) { return 0.0; }
  let c = b.bc * ANOM_CELL;
  let delta = g - c;
  let r2 = dot(delta, delta);
  if (r2 >= ANOM_R_OUT * ANOM_R_OUT) { return 0.0; }
  let r = sqrt(r2);
  let theta = atan2(delta.y, delta.x);
  let fall = 1.0 - smoothstep(ANOM_R_IN, ANOM_R_OUT, r);
  let t = r * (1.0 / TERRACE_STEP);
  let ft = fract(t);
  let steps = (floor(t) + smoothstep(0.3, 0.7, ft)) * TERRACE_H;
  let spire = SPIRE_H * exp(-r2 * (1.0 / (SPIRE_R * SPIRE_R)));
  let spiral = SPIRAL_AMP * sin(SPIRAL_TURNS * theta + SPIRAL_RAD * r);
  return fall * (steps + spire + spiral);
}

// ---- §13 regional rules ------------------------------------------------------
struct Reg { wRegion: f32, mountFactor: f32, duneAmount: f32 };

fn regionField(g: vec2f, seed: i32) -> Reg {
  let p_in = g * (1.0 / REGION_CELL);
  let ip = floor(p_in);
  let fr = p_in - ip;
  let cxi0 = i32(ip.x);
  let cyi0 = i32(ip.y);
  let sTerm = bitcast<u32>(seed + 900) * 0x9e3779b9u;
  let row0 = (bitcast<u32>(cyi0) * 0x27d4eb2fu) ^ sTerm;
  let rowP1 = (bitcast<u32>(cyi0 + 1) * 0x27d4eb2fu) ^ sTerm;
  let rowM1 = (bitcast<u32>(cyi0 - 1) * 0x27d4eb2fu) ^ sTerm;
  let cx0 = bitcast<u32>(cxi0);
  let cxP1 = bitcast<u32>(cxi0 + 1);
  let cxM1 = bitcast<u32>(cxi0 - 1);

  let h0 = hashU(cx0 ^ row0);
  let p0 = vec2f(f32(h0 & 0xfffu), f32((h0 >> 12u) & 0xfffu)) * INV_4095;
  let d0 = p0 - fr;
  var f1_sq = dot(d0, d0);
  var f2_sq = 64.0;
  var btype = f32(h0 >> 24u) * (1.0 / 255.0);

  let ho = hashU4(vec4u(cxP1 ^ row0, cx0 ^ rowP1, cxM1 ^ row0, cx0 ^ rowM1));
  let po = cellPts(ho);
  let dx_o = po[0] - fr.x + vec4f(1.0, 0.0, -1.0, 0.0);
  let dy_o = po[1] - fr.y + vec4f(0.0, 1.0, 0.0, -1.0);
  let d2_o = dx_o * dx_o + dy_o * dy_o;
  for (var k = 0; k < 4; k++) {
    let d2 = d2_o[k];
    if (d2 < f1_sq) {
      f2_sq = f1_sq;
      f1_sq = d2;
      btype = f32(ho[k] >> 24u) * (1.0 / 255.0);
    } else if (d2 < f2_sq) {
      f2_sq = d2;
    }
  }

  let fx0_2 = fr.x * fr.x;
  let fx1_2 = (1.0 - fr.x) * (1.0 - fr.x);
  let fy0_2 = fr.y * fr.y;
  let fy1_2 = (1.0 - fr.y) * (1.0 - fr.y);
  let minDiag = min(min(fx1_2 + fy1_2, fx0_2 + fy1_2), min(fx1_2 + fy0_2, fx0_2 + fy0_2));
  if (minDiag < f2_sq) {
    let hd = hashU4(vec4u(cxP1 ^ rowP1, cxM1 ^ rowP1, cxP1 ^ rowM1, cxM1 ^ rowM1));
    let pd = cellPts(hd);
    let dx_d = pd[0] - fr.x + vec4f(1.0, -1.0, 1.0, -1.0);
    let dy_d = pd[1] - fr.y + vec4f(1.0, 1.0, -1.0, -1.0);
    let d2_d = dx_d * dx_d + dy_d * dy_d;
    for (var k = 0; k < 4; k++) {
      let d2 = d2_d[k];
      if (d2 < f1_sq) {
        f2_sq = f1_sq;
        f1_sq = d2;
        btype = f32(hd[k] >> 24u) * (1.0 / 255.0);
      } else if (d2 < f2_sq) {
        f2_sq = d2;
      }
    }
  }

  let edge = sqrt(f2_sq) - sqrt(f1_sq);
  let w = smoothstep(REG_BLEND_LO, REG_BLEND_HI, edge);
  var mf = REG_DUNE;
  if (btype < 0.33) { mf = REG_HILLS; } else if (btype < 0.66) { mf = REG_MTN; }
  return Reg(w, mf, smoothstep(0.66, 0.8, btype) * w);
}

fn dunesE(x: f32, y: f32, seed: i32, dp: f32, amount: f32) -> f32 {
  if (amount <= 0.0) { return 0.0; }
  let phase = x * F_DUNE * 6.28318530718
    + 3.0 * fbm2(x * F_WARP, y * F_WARP, seed + 301, dp, F_WARP, ASUM_WARP);
  let c = 1.0 - abs(sin(phase));
  return DUNE_H * amount * c * c * c;
}

// ---- master elevation E(x, z) — spec §2 (mirror terrainMath.elevation) --------
fn elevation(g: vec2f, dp: f32, seed: i32) -> f32 {
  let wq = domainWarp(g.x, g.y, seed, dp);
  let reg = regionField(g, seed);
  let cont = smoothstep(CONT_LO, CONT_HI, 0.5 + 0.5 * vnoise(g.x * F_CONT, g.y * F_CONT, seed + 801));
  let base = H_BASE * (0.5 + 0.5 * fbm9(wq.x * F_MACRO, wq.y * F_MACRO, seed + 1, dp, F_MACRO, ASUM_MACRO));
  let maskVal = 0.5 + 0.5 * vnoise(wq.x * F_MASK, wq.y * F_MASK, seed + 77);
  let mask = smoothstep(MASK_LO, MASK_HI, maskVal);
  let hMount = H_MOUNT * (1.0 + (reg.mountFactor - 1.0) * reg.wRegion);
  var ridge = 0.0;
  if (mask > 0.0) {
    ridge = ridged5(wq.x * F_RIDGE, wq.y * F_RIDGE, seed + 5, dp, F_RIDGE, ASUM_RIDGE);
  }
  let de = voronoiEdge(wq * F_VALLEY, seed + 21);
  let sv = 1.0 - smoothstep(0.0, W_VALLEY, de);
  let valley = C_DEPTH * sv * sv;
  var dune = 0.0;
  if (reg.duneAmount > 0.0) { dune = dunesE(g.x, g.y, seed, dp, reg.duneAmount); }
  let anom = anomalyE(g, seed);
  return base * (CONT_MIN + (1.0 - CONT_MIN) * cont)
       + hMount * mask * ridge
       + dune
       - valley
       + anom;
}

// ---- §9 climate ----------------------------------------------------------------
fn temperature(g: vec2f, e: f32) -> f32 {
  return T_BASE - L_LAT * abs(g.y) - L_ALT * max(e, 0.0);
}
fn moisture(g: vec2f, seed: i32) -> f32 {
  let n = fbm2(g.x * F_MOIST, g.y * F_MOIST, seed + 61, 0.0, F_MOIST, ASUM_WARP);
  let cont = smoothstep(CONT_LO, CONT_HI, 0.5 + 0.5 * vnoise(g.x * F_CONT, g.y * F_CONT, seed + 801));
  return M_BASE + M_AMP * n + M_OCEAN * (1.0 - cont);
}

// ---- §10/§11/§21 biome RBF anchors in OKLab (mirror constants.ts BIOME_ANCHORS)
fn oklabToLinear(lab: vec3f) -> vec3f {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  let M = mat3x3f(
    4.0767416621, -1.2684380046, -0.0041960863,
    -3.3077115913, 2.6097574011, -0.7034186147,
    0.2309699292, -0.3413193965, 1.7076147010);
  return M * vec3f(l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
}

struct Biome { col: vec3f, veg: f32 };

fn rbf(clim: vec4f, ctr: vec4f, inv: vec4f) -> f32 {
  let v = (clim - ctr) * inv;
  return exp(-dot(v, v));
}

fn biomeColor(T: f32, M: f32, e: f32, slopeM: f32, vegNoise: f32) -> Biome {
  let clim = vec4f(T, M, e, slopeM);
  let w0 = rbf(clim, vec4f(-4.0, 0.55, 800.0, 1.0), vec4f(1.0 / 4.5, 1.0 / 0.45, 1.0 / 600.0, 1.0 / 1.1));
  let w1 = rbf(clim, vec4f(9.0, 0.45, 550.0, 1.7), vec4f(1.0 / 22.0, 1.0 / 0.55, 1.0 / 600.0, 1.0 / 0.8));
  let w2 = rbf(clim, vec4f(13.0, 0.72, 90.0, 0.32), vec4f(1.0 / 6.5, 1.0 / 0.22, 1.0 / 260.0, 1.0 / 0.5));
  let w3 = rbf(clim, vec4f(11.0, 0.90, 230.0, 0.45), vec4f(1.0 / 7.0, 1.0 / 0.15, 1.0 / 380.0, 1.0 / 0.55));
  let w4 = rbf(clim, vec4f(16.0, 0.13, 55.0, 0.35), vec4f(1.0 / 8.0, 1.0 / 0.17, 1.0 / 320.0, 1.0 / 0.5));
  let invWsum = 1.0 / max(w0 + w1 + w2 + w3 + w4, 1e-6);
  let lab = (
    w0 * vec3f(0.93, 0.0, -0.015) +
    w1 * vec3f(0.44, 0.004, 0.008) +
    w2 * vec3f(0.52, -0.058, 0.048) +
    w3 * vec3f(0.32, -0.072, 0.032) +
    w4 * vec3f(0.79, 0.012, 0.072)) * invWsum;
  let veg = (w2 * 0.92 + w3 * 1.0 + w1 * 0.06 + w4 * 0.05) * invWsum;
  let clump = smoothstep(0.2, 0.8, 0.5 + 0.5 * vegNoise);
  let col = oklabToLinear(lab);
  return Biome(mix(col, col * 0.52 + vec3f(0.002, 0.020, 0.004), veg * clump * 0.85), veg);
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
