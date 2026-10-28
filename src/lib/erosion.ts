/*
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at
 * https://mozilla.org/MPL/2.0/.
 *
 * Gully-erosion filter (TypeScript, WGSL and GLSL), a derivative work of Rune Skovbo Johansen's
 * "Advanced Terrain Erosion Filter" (https://www.shadertoy.com/view/wXcfWn, MPL-2.0;
 * https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html), by way of its WGSL
 * port in the bevy_erosion_filter crate (MPL-2.0). Copyright 2025 Rune Skovbo Johansen; earlier
 * lineage Copyright 2023 Fewes, Copyright 2020 Clay John.
 *
 * Changes from upstream: the cell jitter uses this project's integer Murmur3 lattice hash (so the
 * CPU twin in fp64 and the GPU in fp32 place the same cells at any distance from the origin, where
 * IQ's fract-based hash diverges); each octave fades to its fade target as it shrinks below the
 * pixel footprint (spec §5 prefiltering); the gully direction bends by a smooth sign of each octave's
 * slope (upstream: sign(), a discontinuity on which fp32 and fp64 part ways); the ridge map is not computed;
 * each cell scales its stripe spacing by up to ±jitter/2 (upstream: one spacing per octave, which lines
 * every flank with a comb of gullies at one wavelength; the pivot jitter keeps 12 bits a axis to make room).
 *
 * What it does: given a height and its gradient, every octave lays stripes ("Phacelle" noise)
 * along the downhill direction, interpolated between jittered cell pivots, and adds them as
 * gullies and ridges; each octave's gully slope bends the next, so smaller gullies branch off
 * larger ones. Flat ground (and the peaks and valley floors, through the fade target) is left
 * alone. Evaluated per point: chunkable and infinite.
 */

import { hash3 } from './noise';

/** The filter's parameters (upstream defaults, for heights and positions in units of EROSION.unit metres). */
const EROSION = {
  /** Metres per unit of height and of distance the filter works in. */
  unit: 1200,
  scale: 0.42,
  strength: 0.16,
  gullyWeight: 0.25,
  detail: 1.5,
  rounding: [0.1, 0.0, 0.1, 2.0],
  onset: [1.25, 1.25],
  assumedSlope: [0.7, 1.0],
  cellScale: 0.7,
  normalization: 0.5,
  octaves: 4,
  lacunarity: 2.0,
  gain: 0.5,
  /** Spread of the stripe spacing between cells (0: upstream's one spacing per octave). */
  jitter: 0.8,
} as const;

const TAU = Math.PI * 2;
const SEED = 9137;

function easeOut(t: number): number {
  const v = 1 - Math.min(Math.max(t, 0), 1);
  return 1 - v * v;
}
function smoothStart(t: number, s: number): number {
  return t >= s ? t - 0.5 * s : 0.5 * t * t / s;
}
function powInv(t: number, p: number): number {
  return 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), p);
}

/** Phacelle noise result (module scratch): cos, sin, side direction × freq × TAU. */
const PH = { c: 0, s: 0, sx: 0, sz: 0 };

function phacelle(px: number, pz: number, dx: number, dz: number, freq: number, offset: number, norm: number, octave: number): void {
  const sx = -dz * freq * TAU, sz = dx * freq * TAU;
  const off = offset * TAU;
  const ix = Math.floor(px), iz = Math.floor(pz);
  const fx = px - ix, fz = pz - iz;
  let pc = 0, ps = 0, ws = 0;
  for (let j = -1; j <= 2; j++) {
    for (let i = -1; i <= 2; i++) {
      const h = hash3(ix + i, iz + j, SEED + octave);
      const vx = fx - i - ((h & 0xfff) / 4095 - 0.5);
      const vz = fz - j - (((h >>> 12) & 0xfff) / 4095 - 0.5);
      const w = Math.max(0, Math.exp(-(vx * vx + vz * vz) * 2) - 0.01111);
      ws += w;
      const fj = 1 + EROSION.jitter * ((h >>> 24) / 255 - 0.5);
      const wave = (vx * sx + vz * sz) * fj + off;
      pc += Math.cos(wave) * w;
      ps += Math.sin(wave) * w;
    }
  }
  pc /= ws; ps /= ws;
  const mag = Math.max(1 - norm, Math.sqrt(pc * pc + ps * ps));
  PH.c = pc / mag; PH.s = ps / mag; PH.sx = sx; PH.sz = sz;
}

/**
 * The filter's change to the height at (x, z) (metres): `h` the input height (m), (gx, gz) its
 * gradient, `fade` the target (−1 in valleys … +1 on peaks), `dp` the pixel footprint (m).
 */
export function erosionDelta(x: number, z: number, h: number, gx: number, gz: number, fade: number, dp: number): number {
  const E = EROSION;
  const U = E.unit;
  const px = x / U, pz = z / U;
  let strength = E.strength * E.scale;
  let fadeT = Math.min(Math.max(fade, -1), 1);
  let hh = h / U;
  const h0 = hh;
  let freq = 1 / (E.scale * E.cellScale);
  const slopeLen = Math.max(Math.hypot(gx, gz), 1e-10);
  let roundingMult = 1;
  const roundIn = (E.rounding[1] + (E.rounding[0] - E.rounding[1]) * Math.min(Math.max(fadeT + 0.5, 0), 1)) * E.rounding[2];
  let mask = easeOut(smoothStart(slopeLen * E.onset[0], roundIn * E.onset[0]));
  let gsx = gx + (gx / slopeLen * E.assumedSlope[0] - gx) * E.assumedSlope[1];
  let gsz = gz + (gz / slopeLen * E.assumedSlope[0] - gz) * E.assumedSlope[1];
  for (let o = 0; o < E.octaves; o++) {
    const cell = U / freq; // metres per cell of this octave
    const wNy = Math.min(Math.max(2 - 4 * dp / cell, 0), 1);
    if (wNy <= 0) break;
    const gl = Math.hypot(gsx, gsz);
    const nx = gl > 1e-10 ? gsx / gl : gsx, nz = gl > 1e-10 ? gsz / gl : gsz;
    phacelle(px * freq, pz * freq, nx, nz, E.cellScale, 0.25, E.normalization, o);
    const pzx = PH.sx * -freq, pzz = PH.sz * -freq;
    const sloping = Math.abs(PH.s);
    const sg = PH.s / Math.sqrt(PH.s * PH.s + 0.01); // a smooth sign: continuous, so fp32 and fp64 agree
    gsx += sg * pzx * strength * E.gullyWeight;
    gsz += sg * pzz * strength * E.gullyWeight;
    const oh = PH.c * E.gullyWeight;
    let faded = fadeT + (oh - fadeT) * mask;
    faded = fadeT + (faded - fadeT) * wNy;
    hh += faded * strength;
    fadeT = faded;
    const roundO = (E.rounding[1] + (E.rounding[0] - E.rounding[1]) * Math.min(Math.max(PH.c + 0.5, 0), 1)) * roundingMult;
    const nm = easeOut(smoothStart(sloping * E.onset[1], roundO * E.onset[1]));
    mask = powInv(mask, E.detail) * nm;
    strength *= E.gain;
    freq *= E.lacunarity;
    roundingMult *= E.rounding[3];
  }
  return (hh - h0) * U;
}

const f = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

/** The filter in WGSL (needs hash3(vec2i, i32) -> u32 from the world function). */
export const EROSION_WGSL = /* wgsl */ `
// ---- gully-erosion filter (MPL-2.0: src/lib/erosion.ts) ----
fn eroEaseOut(t: f32) -> f32 { let v = 1.0 - clamp(t, 0.0, 1.0); return 1.0 - v * v; }
fn eroSmoothStart(t: f32, s: f32) -> f32 { if (t >= s) { return t - 0.5 * s; } return 0.5 * t * t / s; }
fn eroPowInv(t: f32, p: f32) -> f32 { return 1.0 - pow(1.0 - clamp(t, 0.0, 1.0), p); }
// Phacelle noise: (cos, sin, side direction × freq × TAU)
fn eroPhacelle(p: vec2f, dir: vec2f, freq: f32, offset: f32, norm: f32, octave: i32) -> vec4f {
  let side = vec2f(-dir.y, dir.x) * freq * 6.283185307;
  let off = offset * 6.283185307;
  let pi = floor(p);
  let pf = p - pi;
  let ii = vec2i(pi);
  var pc = 0.0; var ps = 0.0; var ws = 0.0;
  for (var j = -1; j <= 2; j++) {
    for (var i = -1; i <= 2; i++) {
      let h = hash3(ii.x + i, ii.y + j, ${SEED} + octave);
      let v = pf - vec2f(f32(i), f32(j)) - (vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) / 4095.0 - 0.5);
      let w = max(0.0, exp(-dot(v, v) * 2.0) - 0.01111);
      ws += w;
      // (range-reduced: the GPU's fast sin and cos lose accuracy far from zero)
      let fj = 1.0 + ${f(EROSION.jitter)} * (f32(h >> 24u) / 255.0 - 0.5);
      let w0 = dot(v, side) * fj + off;
      let wave = w0 - 6.283185307 * round(w0 / 6.283185307);
      pc += cos(wave) * w;
      ps += sin(wave) * w;
    }
  }
  pc /= ws; ps /= ws;
  let mag = max(1.0 - norm, sqrt(pc * pc + ps * ps));
  return vec4f(pc / mag, ps / mag, side);
}
fn erosionDelta(x: vec2f, h: f32, grad: vec2f, fade: f32, dp: f32) -> f32 {
  let U = ${f(EROSION.unit)};
  let p = x / U;
  var strength = ${f(EROSION.strength * EROSION.scale)};
  var fadeT = clamp(fade, -1.0, 1.0);
  var hh = h / U;
  let h0 = hh;
  var freq = ${f(1 / (EROSION.scale * EROSION.cellScale))};
  let slopeLen = max(length(grad), 1e-10);
  var roundingMult = 1.0;
  let roundIn = mix(${f(EROSION.rounding[1])}, ${f(EROSION.rounding[0])}, clamp(fadeT + 0.5, 0.0, 1.0)) * ${f(EROSION.rounding[2])};
  var mask = eroEaseOut(eroSmoothStart(slopeLen * ${f(EROSION.onset[0])}, roundIn * ${f(EROSION.onset[0])}));
  var gs = mix(grad, grad / slopeLen * ${f(EROSION.assumedSlope[0])}, ${f(EROSION.assumedSlope[1])});
  for (var o = 0; o < ${EROSION.octaves}; o++) {
    let cell = U / freq;
    let wNy = clamp(2.0 - 4.0 * dp / cell, 0.0, 1.0);
    if (wNy <= 0.0) { break; }
    let gl = length(gs);
    let nd = select(gs, gs / gl, gl > 1e-10);
    let ph = eroPhacelle(p * freq, nd, ${f(EROSION.cellScale)}, 0.25, ${f(EROSION.normalization)}, o);
    let pzw = ph.zw * -freq;
    let sloping = abs(ph.y);
    gs += ph.y * inverseSqrt(ph.y * ph.y + 0.01) * pzw * strength * ${f(EROSION.gullyWeight)};
    let oh = ph.x * ${f(EROSION.gullyWeight)};
    var faded = mix(fadeT, oh, mask);
    faded = mix(fadeT, faded, wNy);
    hh += faded * strength;
    fadeT = faded;
    let roundO = mix(${f(EROSION.rounding[1])}, ${f(EROSION.rounding[0])}, clamp(ph.x + 0.5, 0.0, 1.0)) * roundingMult;
    let nm = eroEaseOut(eroSmoothStart(sloping * ${f(EROSION.onset[1])}, roundO * ${f(EROSION.onset[1])}));
    mask = eroPowInv(mask, ${f(EROSION.detail)}) * nm;
    strength *= ${f(EROSION.gain)};
    freq *= ${f(EROSION.lacunarity)};
    roundingMult *= ${f(EROSION.rounding[3])};
  }
  return (hh - h0) * U;
}
`;

/** The filter in GLSL ES 3.00 (needs uint hash3(int, int, int) from the world function). */
export const EROSION_GLSL = /* glsl */ `
// ---- gully-erosion filter (MPL-2.0: src/lib/erosion.ts) ----
float eroEaseOut(float t) { float v = 1.0 - clamp(t, 0.0, 1.0); return 1.0 - v * v; }
float eroSmoothStart(float t, float s) { return t >= s ? t - 0.5 * s : 0.5 * t * t / s; }
float eroPowInv(float t, float p) { return 1.0 - pow(1.0 - clamp(t, 0.0, 1.0), p); }
vec4 eroPhacelle(vec2 p, vec2 dir, float freq, float offset, float norm, int octave) {
  vec2 side = vec2(-dir.y, dir.x) * freq * 6.283185307;
  float off = offset * 6.283185307;
  vec2 pi = floor(p);
  vec2 pf = p - pi;
  ivec2 ii = ivec2(pi);
  float pc = 0.0, ps = 0.0, ws = 0.0;
  for (int j = -1; j <= 2; j++) {
    for (int i = -1; i <= 2; i++) {
      uint h = hash3(ii.x + i, ii.y + j, ${SEED} + octave);
      vec2 v = pf - vec2(float(i), float(j)) - (vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) / 4095.0 - 0.5);
      float w = max(0.0, exp(-dot(v, v) * 2.0) - 0.01111);
      ws += w;
      float fj = 1.0 + ${f(EROSION.jitter)} * (float(h >> 24u) / 255.0 - 0.5);
      float w0 = dot(v, side) * fj + off;
      float wave = w0 - 6.283185307 * floor(w0 / 6.283185307 + 0.5);
      pc += cos(wave) * w;
      ps += sin(wave) * w;
    }
  }
  pc /= ws; ps /= ws;
  float mag = max(1.0 - norm, sqrt(pc * pc + ps * ps));
  return vec4(pc / mag, ps / mag, side);
}
float erosionDelta(vec2 x, float h, vec2 grad, float fade, float dp) {
  float U = ${f(EROSION.unit)};
  vec2 p = x / U;
  float strength = ${f(EROSION.strength * EROSION.scale)};
  float fadeT = clamp(fade, -1.0, 1.0);
  float hh = h / U;
  float h0 = hh;
  float freq = ${f(1 / (EROSION.scale * EROSION.cellScale))};
  float slopeLen = max(length(grad), 1e-10);
  float roundingMult = 1.0;
  float roundIn = mix(${f(EROSION.rounding[1])}, ${f(EROSION.rounding[0])}, clamp(fadeT + 0.5, 0.0, 1.0)) * ${f(EROSION.rounding[2])};
  float mask = eroEaseOut(eroSmoothStart(slopeLen * ${f(EROSION.onset[0])}, roundIn * ${f(EROSION.onset[0])}));
  vec2 gs = mix(grad, grad / slopeLen * ${f(EROSION.assumedSlope[0])}, ${f(EROSION.assumedSlope[1])});
  for (int o = 0; o < ${EROSION.octaves}; o++) {
    float cell = U / freq;
    float wNy = clamp(2.0 - 4.0 * dp / cell, 0.0, 1.0);
    if (wNy <= 0.0) break;
    float gl = length(gs);
    vec2 nd = gl > 1e-10 ? gs / gl : gs;
    vec4 ph = eroPhacelle(p * freq, nd, ${f(EROSION.cellScale)}, 0.25, ${f(EROSION.normalization)}, o);
    vec2 pzw = ph.zw * -freq;
    float sloping = abs(ph.y);
    gs += ph.y * inversesqrt(ph.y * ph.y + 0.01) * pzw * strength * ${f(EROSION.gullyWeight)};
    float oh = ph.x * ${f(EROSION.gullyWeight)};
    float faded = mix(fadeT, oh, mask);
    faded = mix(fadeT, faded, wNy);
    hh += faded * strength;
    fadeT = faded;
    float roundO = mix(${f(EROSION.rounding[1])}, ${f(EROSION.rounding[0])}, clamp(ph.x + 0.5, 0.0, 1.0)) * roundingMult;
    float nm = eroEaseOut(eroSmoothStart(sloping * ${f(EROSION.onset[1])}, roundO * ${f(EROSION.onset[1])}));
    mask = eroPowInv(mask, ${f(EROSION.detail)}) * nm;
    strength *= ${f(EROSION.gain)};
    freq *= ${f(EROSION.lacunarity)};
    roundingMult *= ${f(EROSION.rounding[3])};
  }
  return (hh - h0) * U;
}
`;
