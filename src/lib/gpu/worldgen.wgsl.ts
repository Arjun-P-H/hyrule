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
  F, WARP, VALLEY, CLIMATE, LIGHT, LANDS, GEO, HYDRO, TOWER, CRAG,
  MAX_WORLD_H, MAX_DIST, CLOUDS,
} from '../constants';
import { BI, BIOMES, BIOME_COUNT, GROUND_USE, LANDFORM, LANDFORM_KEYS, LOOK, SNOW_ALBEDO } from '../biomes';
import { EROSION_WGSL } from '../erosion';
import { COAST_RAMP, CRATER_RIM, SINK } from '../hydro';
import type { World } from '../world';
import { CEIL_SPAN, CEIL_START, DRY_TABLE, ERO_STEP, LAND_MEANS, RIDGE_AMP, RIDGE_FREQ } from '../terrainMath';

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

/**
 * The world's surface in WGSL for `world`: the land, or the cloud world's ground (CLOUD_WORLD: a
 * compile-time constant, so each world's pipelines carry only their own branch).
 */
export function worldWGSL(world: World): string {
  return /* wgsl */ `
const CLOUD_WORLD: bool = ${world === 'clouds'};
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
    // (bent again at three finer scales: hydro.ts rangeAt)
    let b = w + ${f(GEO.rangeWarp)} * vec2f(vnoise(w.x * F_RANGEWARP, w.y * F_RANGEWARP, seed + 3113), vnoise(w.x * F_RANGEWARP + 3.7, w.y * F_RANGEWARP - 1.9, seed + 3114))
      + vec2f(
        ${f(GEO.rangeBend[0])} * vnoise(w.x * F_PEAKS, w.y * F_PEAKS, seed + 3116) + ${f(GEO.rangeBend[1])} * vnoise(w.x * F_FLOORWOB, w.y * F_FLOORWOB, seed + 3117) + ${f(GEO.rangeBend[2])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3118),
        ${f(GEO.rangeBend[0])} * vnoise(w.x * F_PEAKS, w.y * F_PEAKS, seed + 3119) + ${f(GEO.rangeBend[1])} * vnoise(w.x * F_FLOORWOB, w.y * F_FLOORWOB, seed + 3120) + ${f(GEO.rangeBend[2])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3122));
    let ca = ${f(Math.cos(GEO.rangeDir[0]))}; let sa = ${f(Math.sin(GEO.rangeDir[0]))};
    let cb = ${f(Math.cos(GEO.rangeDir[1]))}; let sb = ${f(Math.sin(GEO.rangeDir[1]))};
    let c1 = 1.0 - abs(vnoise((b.x * ca + b.y * sa) * F_RANGEU, (b.y * ca - b.x * sa) * F_RANGEV, seed + 3111));
    let c2 = 1.0 - abs(vnoise((b.x * cb + b.y * sb) * F_RANGEU, (b.y * cb - b.x * sb) * F_RANGEV, seed + 3112));
    let c = mix(c1, c2, smoothstep(0.38, 0.62, n01(w, F_RANGEBLEND, seed + 3115)));
    let pk = ${f(GEO.pass)} + ${f(1 - GEO.pass)} * n01(w, F_PEAKS, seed + 3131);
    return rm * pow(c, ${f(GEO.rangePow)}) * pk;
}

// A tableland's weight at warped w, 0–1 (mirror hydro.ts plateauAt): its outline bent, its wall's width varied.
fn plateauAt(w: vec2f, seed: i32) -> f32 {
  let pn0 = 0.72 * n01(w, F_PLATCOARSE, seed + 1401) + 0.28 * n01(w, F_PLATFINE, seed + 1402);
  if (pn0 < ${f(GEO.plat - 0.05 - GEO.platBand)}) { return 0.0; }
  if (pn0 > ${f(GEO.plat + GEO.platEdge + GEO.platRamp + GEO.platBand)}) { return 1.0; }
  let p = w + vec2f(
    ${f(GEO.platWarp[0])} * vnoise(w.x * F_PLATBAY, w.y * F_PLATBAY, seed + 1411) + ${f(GEO.platWarp[1])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 1412),
    ${f(GEO.platWarp[0])} * vnoise(w.x * F_PLATBAY + 5.3, w.y * F_PLATBAY - 2.1, seed + 1413) + ${f(GEO.platWarp[1])} * vnoise(w.x * F_PLATNOTCH + 3.7, w.y * F_PLATNOTCH - 9.1, seed + 1414));
  let pn = 0.72 * n01(p, F_PLATCOARSE, seed + 1401) + 0.28 * n01(p, F_PLATFINE, seed + 1402);
  let ew = ${f(GEO.platEdge)} + ${f(GEO.platRamp)} * smoothstep(0.35, 0.7, n01(w, F_PLATWIDTH, seed + 1415));
  return 0.82 * smoothstep(${f(GEO.plat)}, ${f(GEO.plat)} + ew, pn) + 0.18 * smoothstep(${f(GEO.plat - 0.05)}, ${f(GEO.plat)}, pn);
}

// The uplift's raw height over the shore at warped w without a volcano (mirror hydro.ts baseAt).
fn baseAt(w: vec2f, seed: i32) -> f32 {
  let low = ${f(GEO.low[0])} + ${f(GEO.low[1])} * n01(w, F_UPL, seed + 3101) + ${f(GEO.low[2])} * n01(w, F_UPL2, seed + 3102);
  let rng = rangeAt(w, seed);
  let ms = smoothstep(${f(GEO.massif[0])}, ${f(GEO.massif[1])}, n01(w, F_MASSIF, seed + 3141));
  let pl = plateauAt(w, seed);
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
  let pl = plateauAt(w, seed);
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
  // (the regions' edges wander: terrainMath.ts regionsInto)
  let r = w + vec2f(
    ${f(LANDS.edge[0])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 1211) + ${f(LANDS.edge[1])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 1212),
    ${f(LANDS.edge[0])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 1213) + ${f(LANDS.edge[1])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 1214));
  let T0 = T_BASE + T_AMP * (vnoise(r.x * F_TEMP, r.y * F_TEMP, seed + 1201) + T_FINE * vnoise(r.x * F_TEMPFINE, r.y * F_TEMPFINE, seed + 1202))
    - L_LAT * abs(g.y);
  let C = up.C;
  let M = M_BASE + M_AMP * (vnoise(r.x * F_MOISTR, r.y * F_MOISTR, seed + 1301) + M_FINE * vnoise(r.x * F_MOISTFINE, r.y * F_MOISTFINE, seed + 1302))
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
  let bad = smoothstep(${f(LANDS.rugged[0])}, ${f(LANDS.rugged[1])}, n01(r, F_RUGGED, seed + 1501));
  let lake = smoothstep(${f(LANDS.lakedist[0])}, ${f(LANDS.lakedist[1])}, n01(r, F_LAKEDIST, seed + 1601));
  let karst = smoothstep(${f(LANDS.karst[0])}, ${f(LANDS.karst[1])}, n01(r, F_KARST, seed + 1701));
  let marsh = wet * smoothstep(${f(LANDS.marsh[0])}, ${f(LANDS.marsh[1])}, n01(r, F_MARSH, seed + 1801));
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

// Karst towers: the highest of the towers of the 3 × 3 cells round w (m): elliptical, their outlines wandering and their walls
// fluted into buttresses and gullies, rising from a skirt through a ledged wall to a domed, lumpy crown (mirror terrainMath.ts karstTowers).
fn karstTowers(w: vec2f, seed: i32, dp: f32) -> f32 {
  let wW = clamp(2.0 - 2.0 * dp * F_TOWERWARP, 0.0, 1.0);
  let wB = clamp(2.0 - 2.0 * dp * F_TOWERBUMP, 0.0, 1.0);
  let wC = clamp(2.0 - 2.0 * dp * F_TOWERCROWN, 0.0, 1.0);
  let w1 = clamp(2.0 - dp / 15.0, 0.0, 1.0);
  let w2 = clamp(2.0 - dp / 7.0, 0.0, 1.0);
  let aw = ${f(TOWER.warp)} * wW;
  let wp = w + aw * vec2f(vnoise(w.x * F_TOWERWARP, w.y * F_TOWERWARP, seed + 2311), vnoise(w.x * F_TOWERWARP, w.y * F_TOWERWARP, seed + 2312));
  let crown = 1.0 + ${f(TOWER.crown)} * wC * vnoise(w.x * F_TOWERCROWN, w.y * F_TOWERCROWN, seed + 2313);
  var bump = 0.0;
  if (wB > 0.0) { bump = ${f(TOWER.bump)} * wB * vnoise(w.x * F_TOWERBUMP, w.y * F_TOWERBUMP, seed + 2314); }
  let p = wp * F_TOWER;
  let ip = floor(p);
  var best = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let ci = i32(ip.x) + i;
      let cj = i32(ip.y) + j;
      let h = hash3(ci, cj, seed + 2301);
      let d = p - (vec2f(f32(ci), f32(cj)) + 0.2 + 0.6 * vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) * INV_4095);
      let r = (75.0 + 70.0 * f32((h >> 24u) & 15u) / 15.0) * F_TOWER;
      if (dot(d, d) >= ${f(TOWER.reach)} * r * r) { continue; }
      let g = hash3(ci, cj, seed + 2302);
      if ((g >> 28u) < ${Math.round(TOWER.absent * 16)}u) { continue; }
      let al = 0.5 + 1.6 * (f32(g & 63u) / 63.0 - 0.5);
      let ca = cos(al);
      let sa = sin(al);
      let sq = sqrt(1.0 + ${f(TOWER.stretch)} * f32((g >> 6u) & 15u) / 15.0);
      let uv = vec2f(dot(d, vec2f(ca, sa)) / (r * sq), (d.y * ca - d.x * sa) * sq / r);
      let q0 = length(uv);
      var q = q0;
      var n1 = 0.0;
      if (w1 > 0.0 && q0 > 0.3) {
        let cs = uv / q0;
        let o = vec2f(f32((g >> 10u) & 63u), f32((g >> 16u) & 63u));
        n1 = vnoise(${f(TOWER.rho[0])} * cs.x + o.x + 0.9 * q0, ${f(TOWER.rho[0])} * cs.y + o.y - 0.6 * q0, seed + 2321);
        var D = ${f(TOWER.flute[0])} * w1 * (abs(n1) - 0.38);
        if (w2 > 0.0) { D += ${f(TOWER.flute[1])} * w2 * (abs(vnoise(${f(TOWER.rho[1])} * cs.x + o.x, ${f(TOWER.rho[1])} * cs.y + o.y, seed + 2322)) - 0.38); }
        q = q0 * (1.0 + D * smoothstep(0.3, 0.65, q0));
      }
      if (q >= 1.0) { continue; }
      let sx = min((1.0 - q) / ${f(TOWER.wall)}, 1.0);
      var y = sx * sx * sx * (sx * (sx * 6.0 - 15.0) + 10.0);
      let ns = 5.0 + 1.5 * f32((g >> 22u) & 3u);
      let ph = y * ns + f32((g >> 24u) & 15u) / 15.0 + 1.2 * n1;
      let fl = floor(ph);
      y += ${f(TOWER.ledge)} * (fl + smoothstep(0.3, 0.7, ph - fl) - ph) / ns * smoothstep(0.0, 0.12, y);
      let H = (50.0 + 170.0 * f32((h >> 28u) & 15u) / 15.0) * crown;
      best = max(best, (H * (1.0 - ${f(TOWER.dome)} * q * q) + bump) * y);
    }
  }
  return best;
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
  // (the crests' bends at three scales: terrainMath.ts dunesE)
  let u = (g.x * ca + g.y * sa) * F_DUNE + 0.9 * vnoise(g.x * F_WARP, g.y * F_WARP, seed + 2402) + 0.55 * vnoise(g.x * F_DUNEBEND, g.y * F_DUNEBEND, seed + 2405)
    + 0.16 * clamp(2.0 - 2.0 * dp * F_PLATNOTCH, 0.0, 1.0) * vnoise(g.x * F_PLATNOTCH, g.y * F_PLATNOTCH, seed + 2406);
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

// Mesas: three tiers of flat-topped buttes, each a cliff over a talus skirt (m).
fn mesasE(w: vec2f, seed: i32, dp: f32, region0: f32) -> f32 {
  // the region's edge wanders (terrainMath.ts mesasE)
  let region = clamp(region0 + 4.0 * region0 * (1.0 - region0) * (0.45 * clamp(2.0 - 2.0 * dp * F_TOWER, 0.0, 1.0) * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 2541)
    + 0.25 * clamp(2.0 - 2.0 * dp * F_PLATNOTCH, 0.0, 1.0) * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 2542)), 0.0, 1.0);
  // the outline bent (as the tablelands': bays and spurs, notches), the finer scale fading out where the footprint would alias it
  let kb = clamp(2.0 - 4.0 * dp * F_PLATBAY, 0.0, 1.0);
  let kn = clamp(2.0 - 4.0 * dp * F_PLATNOTCH, 0.0, 1.0);
  let A = ${f(GEO.mesaWarp[0])} * kb;
  let B = ${f(GEO.mesaWarp[1])} * kn;
  let m = w + vec2f(
    A * vnoise(w.x * F_PLATBAY, w.y * F_PLATBAY, seed + 2511) + B * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 2512),
    A * vnoise(w.x * F_PLATBAY + 5.3, w.y * F_PLATBAY - 2.1, seed + 2513) + B * vnoise(w.x * F_PLATNOTCH + 3.7, w.y * F_PLATNOTCH - 9.1, seed + 2514));
  // (toward the region's edge the field is lowered: buttes on its fringe, one plateau in its heart, no wall along its edge)
  var n = 0.7 * n01(m, F_MESA, seed + 2501) + 0.3 * n01(m, F_MESAFINE, seed + 2502) - (1.0 - region) * ${f(GEO.mesaFade)} * (0.6 + 0.8 * n01(w, F_PLATBAY, seed + 2531));
  // (the crags: ridged knobs in the field, which only the walls, where the field is steep in height, show)
  let c0 = clamp(2.0 - 2.0 * dp * F_CRAGBIG, 0.0, 1.0);
  let c1 = clamp(2.0 - 2.0 * dp * F_CRAG, 0.0, 1.0);
  let c2 = clamp(2.0 - 2.0 * dp * F_CRAGFINE, 0.0, 1.0);
  if (c0 > 0.0) { let a = 1.0 - abs(vnoise(w.x * F_CRAGBIG, w.y * F_CRAGBIG, seed + 2523)); n += ${f(CRAG.mesa[0])} * c0 * (a * a - RIDGE_MEAN); }
  if (c1 > 0.0) { let a = 1.0 - abs(vnoise(w.x * F_CRAG, w.y * F_CRAG, seed + 2521)); n += ${f(CRAG.mesa[1])} * c1 * (a * a - RIDGE_MEAN); }
  if (c2 > 0.0) { let a = 1.0 - abs(vnoise(w.x * F_CRAGFINE, w.y * F_CRAGFINE, seed + 2522)); n += ${f(CRAG.mesa[2])} * c2 * (a * a - RIDGE_MEAN); }
  // the walls: sheer, or a slope where the field says so
  let ramp = smoothstep(0.45, 0.78, n01(w, F_PLATWIDTH, seed + 2515));
  let cw = max(0.025, 0.02 * dp) * (1.0 + ${f(GEO.mesaRamp)} * ramp);
  // (scaled in over the region's first quarter: landDetail calls this only where the region's weight is above 0)
  return smoothstep(0.0, 0.25, region) * (80.0 * (0.62 * smoothstep(0.0, cw, n - 0.42) + 0.38 * smoothstep(-0.2, 0.0, n - 0.42))
    + 70.0 * (0.62 * smoothstep(0.0, cw, n - 0.52) + 0.38 * smoothstep(-0.2, 0.0, n - 0.52))
    + 60.0 * (0.62 * smoothstep(0.0, cw, n - 0.61) + 0.38 * smoothstep(-0.2, 0.0, n - 0.61)));
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

// The mountains' ridged octaves at warped w, centred on their mean (× ridge × the range there).
fn ridgesAt(w: vec2f, seed: i32, dp: f32) -> f32 {
  let rw = w * F_RIDGE;
  let kr = 4.0 * F_RIDGE * dp;
  var rh = 0.0;
  for (var i = 0; i < 5; i++) {
    let fr = RIDGE_FREQS[i];
    let wgt = select(clamp(2.0 - fr * kr, 0.0, 1.0), 1.0, dp <= 0.0);
    if (wgt > 0.0) {
      let n = 1.0 - abs(vnoise(rw.x * fr, rw.y * fr, seed + 5 + i * 34283));
      rh += RIDGE_AMPS[i] * wgt * (n * n - RIDGE_MEAN);
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
};
// A channel's half width where the drainage is acc (m; mirror hydro.ts riverHalfW).
fn riverHalfW(acc: f32) -> f32 {
  return 0.5 * min(${f(HYDRO.widthMax)}, ${f(HYDRO.width[0])} + ${f(HYDRO.width[1])} * pow(max(acc - 1.0, 1e-6), ${f(HYDRO.width[2])}));
}
// The meander's offset (m) from a reach's line at along metres from its head, t its share of the reach.
fn meanderOff(along: f32, t: f32, amp: f32, sg: i32) -> f32 {
  if (amp <= 0.0) { return 0.0; }
  return amp * smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t) * (0.74 * vnoise(along * F_MEANDER, 0.5, sg) + 0.26 * vnoise(along * F_MEANDER * 3.1, 4.5, sg));
}
// How much wider than its base a reach's valley floor is at along (m): a smooth noise, nothing at the nodes.
fn floorWobbleAt(along: f32, t: f32, sg: i32) -> f32 {
  return ${f(HYDRO.floorWobble)} * smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t) * (0.5 + 0.5 * (0.6 * vnoise(along * F_FLOORWOB, 11.5, sg) + 0.4 * vnoise(along * F_FLOORWOB * 2.7, 5.5, sg)));
}
// How much of its width one side of a reach's floor keeps, along the reach (mirror hydro.ts floorSideAt).
fn floorSideAt(along: f32, t: f32, sg: i32, side: f32) -> f32 {
  let taper = smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t);
  let o = side * 40.7;
  let s = 0.5 + 0.5 * (0.62 * vnoise(along * F_FLOORWOB * 1.9 + o, 21.5, sg) + 0.38 * vnoise(along * F_FLOORWOB * 5.3 + o, 9.5, sg));
  return 1.0 - taper * 0.75 * (1.0 - s);
}
fn riverField(w: vec2f, seed: i32, k: f32) -> River {
  var o: River;
  o.R = 1e9; o.F = -1e9; o.xFloor = 1e9; o.W = -1e9; o.ch = 1e9; o.w = 0.0; o.depth = 0.0; o.acc = 0.0; o.Ua = 0.0;
  o.dir = vec2f(0.0); o.fall = 0.0; o.L = -1e9; o.lakeK = 0.0; o.lakeMask = 0.0; o.vis = 0.0; o.cap = 1e9; o.Lnear = -1e9;
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
  // (the walls' run wanders: hydro.ts riverField)
  let wk = 1.0 + ${f(HYDRO.wallVary[0])} * vnoise(w.x * F_FLOORWOB, w.y * F_FLOORWOB, seed + 3306) + ${f(HYDRO.wallVary[1])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3307)
    + ${f(HYDRO.wallVary[2])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 3308);
  let lat = ${f(HYDRO.wallLat)} * (0.6 * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3310) + 0.4 * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 3311));
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
        // (its shore: the noise's own place and a stretch along one axis, from the node's hash: hydro.ts lakeLobe)
        let lq = dv / rr;
        let lid = hash3(i, j, seed + 3510);
        let lns = i32(lid & 0xffffu);
        let la2 = 6.283185307179586 * f32(lid >> 24u) / 255.0;
        let le = 0.06 + 0.16 * f32((lid >> 16u) & 255u) / 255.0;
        let ld2 = dot(dv, dv);
        let ell = 1.0 + select(0.0, le * ((dv.x * dv.x - dv.y * dv.y) * cos(la2) + 2.0 * dv.x * dv.y * sin(la2)) / ld2, ld2 > 1e-9);
        let rl = rr * ell * (1.0 + 0.28 * (0.5 * vnoise(lq.x * 1.3, lq.y * 1.3, seed + 3503 + lns) + 0.3 * vnoise(lq.x * 3.1 + 9.2, lq.y * 3.1 - 4.4, seed + 3504 + lns) + 0.2 * vnoise(lq.x * 7.3 - 5.1, lq.y * 7.3 + 2.7, seed + 3505 + lns)));
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
      let sw = i32(hash3(i, j, seed + 3302) & 0xffffu);
      let wd = riverHalfW(acc + clamp(t, 0.0, 1.0))
        * (1.0 + ${f(HYDRO.widthVary)} * (0.62 * vnoise(along * F_MEANDER * 2.3, 7.5, sw) + 0.38 * vnoise(along * F_MEANDER * 6.1, 3.5, sw)));
      let vis = a1.w; // (the river shows only below a lake: hydro.ts fedOf)
      // (a river whose water does not show cuts no channel: a dry wash, not a ditch)
      let depth = (${f(HYDRO.depth[0])} + ${f(HYDRO.depth[1])} * sqrt(acc)) * vis;
      let slope = (Wa - Wb) / len;
      let amp = ${f(HYDRO.meanderMin)} + ${f(HYDRO.meander - HYDRO.meanderMin)} * (1.0 - smoothstep(0.012, 0.05, (a0.z - b0.z) / len));
      // (the floor: hydro.ts riverField: its base width, of which each side keeps a share but not less than the channel and a margin)
      let baseW = select(0.0, amp + wd, vis > 0.0) + ${f(HYDRO.floor[0])} + ${f(HYDRO.floor[1])} * sqrt(acc) + floorWobbleAt(along, t, i32(hash3(i, j, seed + 3305) & 0xffffu));
      let least = select(${f(HYDRO.floor[0])}, wd + 8.0, vis > 0.0);
      var floorW = baseW;
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
        floorW = max(baseW * floorSideAt(along, t, i32(hash3(i, j, seed + 3309) & 0xffffu), select(1.0, 0.0, ds >= 0.0)), least);
        if (dd - floorW > 400.0) {
          let x = dd - floorW;
          let lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
          let lvl = lf.y;
          let cone = lvl + wallRise(k, x * wk + lat, floorW);
          o.R = min(o.R, cone);
          o.F = max(o.F, lvl + ${f(HYDRO.floorRise)} - ${f(HYDRO.riser)} * x);
          o.xFloor = min(o.xFloor, dd - floorW);
          sA[nR] = vec4f(cone, wallSlope(k, x * wk + lat, floorW) * wk, lf.w / len, ${f(HYDRO.wallCurve)});
          sB[nR] = vec4f(-u.y, u.x, 0.1, lvl);
          sN[nR] = vec2i(ia, ib);
          nR++;
          continue;
        }
        // (the banks of unequal reach: the channel's distance scaled so each bank stands where its own half width puts it)
        let sd = ds - meanderOff(along, t, amp, sg);
        let skew = ${f(HYDRO.bankSkew)} * vnoise(along * F_MEANDER * 1.7, 13.1, i32(hash3(i, j, seed + 3304) & 0xffffu));
        ch = abs(sd) / (1.0 + select(-skew, skew, sd >= 0.0));
      }
      let lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
      let Wt = lf.x;
      let Wv = select(lf.y, mix(Wt, lf.y, smoothstep(floorW, floorW + 60.0, dd)), vis > 0.0);
      // (a dry valley has no channel: hydro.ts riverField)
      let chV = select(1e9, ch, vis > 0.0);
      let wV = select(0.0, wd, vis > 0.0);
      let ddS = select(dd, floorW + (dd - floorW) * wk + lat * smoothstep(0.0, 40.0, dd - floorW), dd > floorW);
      let wkS = select(1.0, wk, dd > floorW);
      let cone = Wv + valleyShape(ddS, chV, wV, depth, floorW, k);
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
        sA[nR] = vec4f(cone, valleySlope(ddS, chV, wV, depth, floorW, k) * wkS, fallT, ${f(HYDRO.wallCurve)});
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
    let dS = select(L4.x, fl + (L4.x - fl) * wk + lat * smoothstep(0.0, 40.0, L4.x - fl), L4.x > fl);
    let cone = L4.z + valleyShape(dS, L4.x, L4.y, L4.w, fl, k);
    o.R = min(o.R, cone);
    o.xFloor = min(o.xFloor, L4.x - fl);
    if (L4.x < L4.y) { sA[nR] = vec4f(max(cone, L4.z), 0.0, 0.0, ${f(HYDRO.wallCurve)}); }
    else { sA[nR] = vec4f(cone, valleySlope(dS, L4.x, L4.y, L4.w, fl, k) * select(1.0, wk, L4.x > fl), 0.0, ${f(HYDRO.wallCurve)}); }
    sB[nR] = vec4f(lkN[m], 0.3, L4.z);
    sN[nR] = vec2i(-1, -2);
    nR++;
  }
  // the nearest channel's water stays under every other reach's surface out to its far bank (that surface
  // falls toward the bank by its slope along the line across the channel, but never below its own river)
  if (best >= 0) {
    let D = (o.w + o.ch) * ${f(1 + HYDRO.bankSkew)} + 1.0;
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

// The crags (m): rock faces break into knobs, ribs and ledges (mirror terrainMath.ts cragDelta).
fn cragDelta(g: vec2f, e: f32, slope: f32, seed: i32, dp: f32) -> f32 {
  let s = smoothstep(${f(CRAG.face[0])}, ${f(CRAG.face[1])}, slope);
  if (s <= 0.0) { return 0.0; }
  let sl = min(slope, 3.2);
  let w0 = clamp(2.0 - 2.0 * dp * F_CRAGBIG, 0.0, 1.0);
  let w1 = clamp(2.0 - 2.0 * dp * F_CRAG, 0.0, 1.0);
  let w2 = clamp(2.0 - 2.0 * dp * F_CRAGFINE, 0.0, 1.0);
  var d = 0.0;
  if (w0 > 0.0) { let a = 1.0 - abs(vnoise(g.x * F_CRAGBIG, g.y * F_CRAGBIG, seed + 5124)); d += ${f(CRAG.rough[0])} * w0 * (a * a - RIDGE_MEAN); }
  if (w1 > 0.0) { let a = 1.0 - abs(vnoise(g.x * F_CRAG, g.y * F_CRAG, seed + 5121)); d += ${f(CRAG.rough[1])} * w1 * (a * a - RIDGE_MEAN); }
  if (w2 > 0.0) { let a = 1.0 - abs(vnoise(g.x * F_CRAGFINE, g.y * F_CRAGFINE, seed + 5122)); d += ${f(CRAG.rough[2])} * w2 * (a * a - RIDGE_MEAN); }
  let wl = clamp(2.0 - 2.0 * dp * sl / ${f(CRAG.period)}, 0.0, 1.0);
  var led = 0.0;
  if (wl > 0.0) {
    let ph = e / ${f(CRAG.period)} + 1.6 * vnoise(g.x * F_CRAGSTRATA, g.y * F_CRAGSTRATA, seed + 5123);
    let fl = floor(ph);
    led = ${f(CRAG.ledge)} * (fl + smoothstep(0.3, 0.7, ph - fl) - ph) * ${f(CRAG.period)} * wl;
  }
  return s * (sl * d + led);
}
// The regions' detail over the uplift at world g, warped w, before any mask (mirror terrainMath.ts landDetail).
fn landDetail(g: vec2f, w: vec2f, up: Uplift, p: Landform, seed: i32, dp: f32) -> f32 {
  var detail = p.hills * hillsAt(w * F_MACRO, seed, dp);
  let rs = min(1.0, 2.0 * up.range + 1.6 * up.massif);
  if (p.ridge > 0.0) { detail += p.ridge * (0.3 + 0.7 * rs) * ridgesAt(w, seed, dp); }
  // (karst, dunes and mesas only where their region rules: a few tiers in a meadow would be cliffs)
  let kw = smoothstep(0.1, 0.9, p.karst);
  if (kw > 0.001) {
    let wK = clamp(2.0 - 2.0 * dp / 150.0, 0.0, 1.0);
    var t = TOWER_MEAN;
    if (wK > 0.0) { t = wK * karstTowers(w, seed, dp) + (1.0 - wK) * TOWER_MEAN; }
    detail += kw * t;
  }
  let dw = smoothstep(0.35, 0.75, p.dune);
  if (dw > 0.001) { detail += dw * dunesE(g, seed, dp); }
  let mw = smoothstep(0.05, 0.95, p.mesa);
  if (mw > 0.001) { detail += mesasE(w, seed, dp, mw); }
  if (p.erosion > 0.01) {
    // (down the land itself, laid out in the world: terrainMath.ts landDetail)
    let rk = select(0.0, p.ridge * (0.3 + 0.7 * rs), p.ridge > 0.0);
    let b0 = erosionLand(w, up.U, seed, p.hills, rk);
    let gr = vec2f(erosionBase(g + vec2f(ERO_STEP, 0.0), seed, p.hills, rk) - b0, erosionBase(g + vec2f(0.0, ERO_STEP), seed, p.hills, rk) - b0) / ERO_STEP;
    // (the gullies' strength wanders, so their flutes are not evenly spaced for kilometres; the crags break every steep face)
    detail += p.erosion * (0.55 + 0.9 * n01(g, F_PLATNOTCH, seed + 5101)) * erosionDelta(g, up.U + detail, gr, 0.0, dp); // (no fade target: terrainMath.ts landDetail)
    detail += smoothstep(0.05, 0.4, p.erosion) * cragDelta(g, up.U + detail, length(gr), seed, dp);
  }
  return detail;
}
// The land at world g before its rivers, lakes, marshes, ponds and the sea (mirror terrainMath.ts landBare).
fn landBare(g: vec2f, dp: f32, seed: i32) -> f32 {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let up = upliftAt(w, seed);
  let rg = regionsFrom(g, w, up, seed);
  return up.U + landDetail(g, w, up, blendLandform(rg), seed, dp) * up.taper;
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
  let inset = ${f(HYDRO.pondReach * 1.6)} * R / cell;
  let c = (vec2f(f32(ix), f32(iz)) + inset + (1.0 - 2.0 * inset) * vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) / 4095.0) * cell;
  // (a place in the noise of its own and a stretch along one axis: terrainMath.ts pondAt)
  let q = (g - c) / R;
  let g3 = hash3(ix, iz, seed + 3406);
  let pns = i32(g3 & 0xffffu);
  let pdv = g - c;
  let pd2 = dot(pdv, pdv);
  let pa2 = 6.283185307179586 * f32(g3 >> 24u) / 255.0;
  let pe = 0.06 + 0.16 * f32((g3 >> 16u) & 255u) / 255.0;
  let pell = 1.0 + select(0.0, pe * ((pdv.x * pdv.x - pdv.y * pdv.y) * cos(pa2) + 2.0 * pdv.x * pdv.y * sin(pa2)) / pd2, pd2 > 1e-9);
  let d = sqrt(pd2) / (R * pell * (1.0 + 0.3 * (0.6 * vnoise(q.x * 1.2, q.y * 1.2, seed + 3403 + pns) + 0.4 * vnoise(q.x * 3.3 + 7.7, q.y * 3.3 - 2.9, seed + 3404 + pns))));
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
  let ec = up.U + landDetail(c, w, up, blendLandform(rg), seed, 4.0) * up.taper;
  var ring = array<vec2f, 3>(vec2f(1.0, 0.0), vec2f(-0.5, 0.8660254037844386), vec2f(-0.5, -0.8660254037844386));
  for (var k = 0; k < 3; k++) {
    if (abs(landBare(c + 1.25 * R * ring[k], 4.0, seed) - ec) > 0.06 * R) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  }
  // away from rivers and lakes
  let rv = riverField(w, seed, 0.12);
  if (rv.xFloor < 2.0 * R + 40.0 || rv.lakeMask > 0.0) { return vec4f(-1e9, 1.0, 0.0, 0.0); }
  return vec4f(ec - ${f(HYDRO.pondSink)}, d, ${f(HYDRO.pondD[0])} + ${f(HYDRO.pondD[1] - HYDRO.pondD[0])} * f32(h2 >> 16u) / 65535.0, R);
}

// ---- the cloud world's ground (mirror skyworld/clouds.ts) ----------------------------
const CLOUD_ALBEDO = vec3f(${CLOUDS.albedo.map(f).join(', ')});
const CL_BMID: f32 = ${f((CLOUDS.base[0] + CLOUDS.base[1]) / 2)};
const CL_BHALF: f32 = ${f((CLOUDS.base[1] - CLOUDS.base[0]) / 2)};
// The cloud sea (mirror clouds.ts cloudSurface): heaped |noise| domes of four sizes on a broad swell.
fn cloudSurface(g: vec2f, dp: f32, seed: i32) -> f32 {
  let fine = 1.0 - min(1.0, dp / 35.0);
  let e = CL_BMID + CL_BHALF * vnoise(g.x / 2400.0, g.y / 2400.0, seed + 8101)
    + ${f(CLOUDS.domes[0])} * abs(vnoise(g.x / 1100.0, g.y / 1100.0, seed + 8102))
    + ${f(CLOUDS.domes[1])} * abs(vnoise(g.x / 380.0, g.y / 380.0, seed + 8103))
    + ${f(CLOUDS.domes[2])} * fine * abs(vnoise(g.x / 120.0, g.y / 120.0, seed + 8105))
    + ${f(CLOUDS.domes[3])} * fine * abs(vnoise(g.x / 36.0, g.y / 36.0, seed + 8106));
  return min(e, ${f(CLOUDS.ceiling)});
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
  if (CLOUD_WORLD) { return Surface(cloudSurface(g, dp, seed), -1e9, 0.0, 0.0, vec2f(0.0), 0.0, -1e9); }
  return landCore(g, dp, seed, pondAt(g, seed));
}
// landSurface with the pond (level or -1e9, distance in radii, depth, radius) given.
fn landCore(g: vec2f, dp: f32, seed: i32, pond: vec4f) -> Surface {
  let w = domainWarp(g.x, g.y, seed, 0.0);
  let up = upliftAt(w, seed);
  let rg = regionsFrom(g, w, up, seed);
  let p = blendLandform(rg);
  let rv = riverField(w, seed, p.k);
  var m = smoothstep(0.0, 50.0, rv.xFloor) * (1.0 - rv.lakeMask);
  if (up.lava > -1e8) { m *= smoothstep(${f(CRATER_RIM * 0.9)}, ${f(CRATER_RIM * 1.3)}, up.rr); }
  // where marshes lie (not in a lake's ring, nor in a river's channel; only on flat land: marsh pools
  // stepping up a hillside would stand at many levels)
  var flat = p.flat * (1.0 - rv.lakeMask);
  if (rv.vis > 0.0) { flat *= smoothstep(rv.w + 2.0, rv.w + 8.0, rv.ch); }
  if (flat > 0.0) {
    let gU = vec2f(upliftAt(w + vec2f(8.0, 0.0), seed).U - up.U, upliftAt(w + vec2f(0.0, 8.0), seed).U - up.U) / 8.0;
    flat *= 1.0 - smoothstep(0.012, 0.035, length(gU));
  }

  var detail = 0.0;
  if (m > 0.0) {
    detail = landDetail(g, w, up, p, seed, dp);
    detail *= m * up.taper;
    // (round a river the detail only rises: a hollow beside it would leave the river standing over it)
    let near = 1.0 - smoothstep(50.0, 450.0, rv.xFloor);
    if (near > 0.0) { detail += (0.5 * (detail + sqrt(detail * detail + 36.0)) - 3.0 - detail) * near; }
    // (and round a marsh: its pools lie just under the smooth land, and a hollow at its edge would spill them)
    let nearM = smoothstep(0.0, 0.3, flat);
    if (nearM > 0.0) { detail += (0.5 * (detail + sqrt(detail * detail + 0.36)) - 0.3 - detail) * nearM; }
  }
  var e0 = min(max(up.U + detail, rv.F), rv.R);
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
  // the coast, the sea floor, the soft ceiling
  let land = smoothstep(COAST_LEVEL - 0.004 - 0.03 * (1.0 - up.cliff), COAST_LEVEL + 0.002, up.C);
  let seaFloor = SEA - 6.0 - ${f(LANDS.coast.shelf)} * (1.0 - smoothstep(COAST_LEVEL - 0.12, COAST_LEVEL, up.C));
  let e = seaFloor + (e0 - seaFloor) * land;
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
// Per biome: (trails, bare patches, tall grass, rocks): GROUND_USE.
const LK_USE = array<vec4f, ${NB}>(${BIOMES.map((b) => `vec4f(${f(GROUND_USE[b].trail)}, ${f(GROUND_USE[b].bare)}, ${f(GROUND_USE[b].tall)}, ${f(GROUND_USE[b].rocks)})`).join(', ')});

struct Biome { col: vec3f, veg: f32 };

// The ground's colour at world g, per pixel (height e, slope, the water's level wl): grass over soil,
// rock on the steep (banded in strata), sand on low gentle shores, snow above the snow line.
fn groundAt(g: vec2f, e: f32, slope: f32, wl: f32, seed: i32) -> Biome {
  if (CLOUD_WORLD) { return Biome(CLOUD_ALBEDO, 0.0); }
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
}
