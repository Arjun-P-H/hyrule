/**
 * The ground's detail textures: the shaders that paint them, and the helpers the surfaces read them with.
 * ---------------------------------------------------------------------------
 * There are no image files (as the leaf sprays, leafAtlas.ts): groundTex.ts paints one tileable layer
 * per material into a texture array at start-up, on the GPU, and builds its mip chain. A texel holds
 *   r, g  the surface's slope across the layer, as a tangent-space normal's x and y (½ = flat), from the
 *         layer's own height in metres over its tile, so a tile's relief is the same on every layer's scale;
 *   b     the layer's tone, a multiplier on the biome's colour (½ = ×1, 1 = ×2);
 *   a     its cavity: 1 on the open ground, down toward 0 in cracks and between pebbles (ambient light;
 *         the height blends by).
 * The biome's colour (the clipmap's soil, grass and rock layers) stays the albedo: the texture only
 * breaks it up, so a badlands' rust and a mountain's grey use the same rock.
 *
 * Every layer tiles (all its noise is periodic over the tile), so the surfaces read it at a world scale
 * that divides the chunk (power of two metres): chunk-local coordinates then agree across a chunk boundary.
 */

import { wgslF32 as f } from './worldgen.wgsl';

/** Texels per side of a layer (mip chain: 512 … 1). */
export const GROUND_SIZE = 512;
export const GROUND_MIPS = Math.log2(GROUND_SIZE) + 1;

/** The materials, in layer order. */
export const GROUND_LAYERS = ['turf', 'soil', 'crack', 'sand', 'gravel', 'rockA', 'rockB', 'rockC', 'moss'] as const;
export type GroundLayer = (typeof GROUND_LAYERS)[number];

/** Each layer's tile in metres on the ground (a power of two) and its relief (m, peak to peak) over the tile. */
export const GROUND_TILE: Record<GroundLayer, number> = {
  turf: 4, soil: 4, crack: 8, sand: 8, gravel: 4, rockA: 16, rockB: 16, rockC: 16, moss: 2,
};
export const GROUND_RELIEF: Record<GroundLayer, number> = {
  turf: 0.07, soil: 0.10, crack: 0.16, sand: 0.03, gravel: 0.16, rockA: 1.5, rockB: 1.1, rockC: 0.8, moss: 0.18,
};

const L = Object.fromEntries(GROUND_LAYERS.map((n, i) => [n, i])) as Record<GroundLayer, number>;

/** The layers' constants, for every shader that reads them. */
const GROUND_CONSTS_WGSL = /* wgsl */ `
const GL_TURF: i32 = ${L.turf};
const GL_SOIL: i32 = ${L.soil};
const GL_CRACK: i32 = ${L.crack};
const GL_SAND: i32 = ${L.sand};
const GL_GRAVEL: i32 = ${L.gravel};
const GL_ROCK_A: i32 = ${L.rockA};
const GL_ROCK_B: i32 = ${L.rockB};
const GL_ROCK_C: i32 = ${L.rockC};
const GL_MOSS: i32 = ${L.moss};
// 1 / the tile's metres
const GT_INV = array<f32, ${GROUND_LAYERS.length}>(${GROUND_LAYERS.map((n) => f(1 / GROUND_TILE[n])).join(', ')});
`;

/**
 * Three kernels paint it (groundTex.ts): `paint` (a pipeline a layer) writes each layer's height, tone and
 * cavity into a float scratch texture, one texel an invocation; `shade` turns the heights into slopes
 * (from the exact floats, not 8-bit steps) and packs the layer; `mip` averages a level into the next
 * (slope, tone and cavity alike: a coarser level is a flatter, duller surface, which is what a distant
 * one is).
 */
export function groundPaintWGSL(): string {
  return /* wgsl */ `
const SIZE: u32 = ${GROUND_SIZE}u;
const TAU: f32 = 6.28318530718;
const INV_U32: f32 = 2.3283064365386963e-10;

@group(0) @binding(0) var dst: texture_storage_2d_array<rgba16float, write>;

fn hu(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u); x = x * 0x85ebca6bu;
  x = x ^ (x >> 13u); x = x * 0xc2b2ae35u;
  x = x ^ (x >> 16u);
  return x;
}
// The hash of a lattice point, the lattice wrapping with period per (so every layer tiles).
fn hl(ix: i32, iy: i32, per: vec2i, seed: u32) -> u32 {
  let x = u32(((ix % per.x) + per.x) % per.x);
  let y = u32(((iy % per.y) + per.y) % per.y);
  return hu(x ^ hu(y ^ (seed * 0x9e3779b9u + 0x7f4a7c15u)));
}
fn h01(ix: i32, iy: i32, per: vec2i, seed: u32) -> f32 { return f32(hl(ix, iy, per, seed)) * INV_U32; }

// Value noise on a lattice of period per (p in lattice units): 0 … 1.
fn tn(p: vec2f, per: vec2i, seed: u32) -> f32 {
  let ip = floor(p);
  let i = vec2i(ip);
  let fr = p - ip;
  let u = fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0);
  let a = h01(i.x, i.y, per, seed);
  let b = h01(i.x + 1, i.y, per, seed);
  let c = h01(i.x, i.y + 1, per, seed);
  let d = h01(i.x + 1, i.y + 1, per, seed);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn tn1(uv: vec2f, per: i32, seed: u32) -> f32 { return tn(uv * f32(per), vec2i(per), seed); }

// fBm of the tile (uv 0 … 1), per0 cells across at the first octave: 0 … 1.
fn tfbm(uv: vec2f, per0: i32, oct: i32, gain: f32, seed: u32) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var norm = 0.0;
  var per = per0;
  for (var o = 0; o < oct; o++) {
    sum += amp * tn1(uv, per, seed + u32(o) * 7919u);
    norm += amp;
    amp *= gain;
    per *= 2;
  }
  return sum / norm;
}

// Cellular noise on a lattice of period pv (p in lattice units): f1 the distance to the nearest feature
// point, f2 to the second nearest (in lattice units), id the nearest cell's own random 0 … 1, d the vector
// from p to that point (a cell's plane, or its dome, is measured from it).
struct Vor { f1: f32, f2: f32, id: f32, d: vec2f, id2: f32, d2: vec2f };
fn tv(p: vec2f, pv: vec2i, seed: u32) -> Vor {
  let ip = floor(p);
  let i0 = vec2i(ip);
  let fr = p - ip;
  var f1 = 16.0;
  var f2 = 16.0;
  var id = 0.0;
  var id2 = 0.0;
  var dn = vec2f(0.0);
  var dn2 = vec2f(0.0);
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let h = hl(i0.x + i, i0.y + j, pv, seed);
      let o = vec2f(f32(h & 0xffffu), f32(h >> 16u)) * (1.0 / 65535.0);
      let d = vec2f(f32(i), f32(j)) + o - fr;
      let dd = dot(d, d);
      let rid = f32(hu(h ^ 0x68bc21ebu) & 0xffffu) * (1.0 / 65535.0);
      if (dd < f1) { f2 = f1; id2 = id; dn2 = dn; f1 = dd; id = rid; dn = d; }
      else if (dd < f2) { f2 = dd; id2 = rid; dn2 = d; }
    }
  }
  return Vor(sqrt(f1), sqrt(f2), id, dn, id2, dn2);
}
fn tv1(uv: vec2f, per: i32, seed: u32) -> Vor { return tv(uv * f32(per), vec2i(per), seed); }

fn tunit(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
// A random 0 … 1 from a cell's id (the same cell, another quantity: k)
fn rid(id: f32, k: f32) -> f32 { return fract(id * k + 0.37 * k); }
// A periodic offset (±amp tiles) for bending a lattice, one noise per axis.
fn tbend(uv: vec2f, per: i32, amp: f32, seed: u32) -> vec2f {
  return vec2f(tn1(uv, per, seed) - 0.5, tn1(uv, per, seed + 1u) - 0.5) * (2.0 * amp);
}

// Facets: each cell a tilted plane (at a level of its own), blended smoothly across a border of width bw so
// the surface has no steps: x the height (about ±0.5), y the distance to the nearest border, z whether a
// crack lies there (about a third of the borders, the same one from either side).
fn tfac(uv: vec2f, per: i32, tilt: f32, bw: f32, seed: u32) -> vec3f {
  let v = tv1(uv, per, seed);
  let t1 = (vec2f(rid(v.id, 127.1), rid(v.id, 311.7)) - 0.5) * tilt;
  let t2 = (vec2f(rid(v.id2, 127.1), rid(v.id2, 311.7)) - 0.5) * tilt;
  let h1 = dot(v.d, t1) + (rid(v.id, 53.3) - 0.5) * 0.3;
  let h2 = dot(v.d2, t2) + (rid(v.id2, 53.3) - 0.5) * 0.3;
  let e = v.f2 - v.f1;
  let open = select(0.0, 1.0, rid(v.id + v.id2, 3.3) > 0.72);
  return vec3f(mix(h2, h1, 0.5 + 0.5 * smoothstep(0.0, bw, e)), e, open);
}

// The same facets with the borders' own random as well: x the height, y the distance to the nearest border, z the
// border's random (the same from either side), w the nearest cell's.
fn tfac2(uv: vec2f, per: i32, tilt: f32, bw: f32, seed: u32) -> vec4f {
  let v = tv1(uv, per, seed);
  let t1 = (vec2f(rid(v.id, 127.1), rid(v.id, 311.7)) - 0.5) * tilt;
  let t2 = (vec2f(rid(v.id2, 127.1), rid(v.id2, 311.7)) - 0.5) * tilt;
  let h1 = dot(v.d, t1) + (rid(v.id, 53.3) - 0.5) * 0.45;
  let h2 = dot(v.d2, t2) + (rid(v.id2, 53.3) - 0.5) * 0.45;
  let e = v.f2 - v.f1;
  return vec4f(mix(h2, h1, 0.5 + 0.5 * smoothstep(0.0, bw, e)), e, rid(v.id + v.id2, 3.3), v.id);
}

// A layer at uv: (height 0 … 1, tone 0 … 2, cavity 0 … 1).

// TURF: the ground under a lawn: tufts (domes with dark gaps between), a slow mottling, faint combed
// strokes, here and there a dark seed or a fallen leaf.
fn lTurf(uv: vec2f) -> vec3f {
  let tuft = tv1(uv + tbend(uv, 5, 0.012, 13u), 14, 11u);
  let dome = pow(tunit(1.0 - tuft.f1 / 0.78), 0.8);
  let mott = tfbm(uv, 4, 3, 0.55, 12u);
  let flow = smoothstep(0.35, 0.65, tfbm(uv, 2, 2, 0.5, 17u));
  let sx = tn(uv * vec2f(160.0, 28.0), vec2i(160, 28), 21u);
  let sy = tn(uv * vec2f(28.0, 160.0), vec2i(28, 160), 23u);
  let stroke = mix(sy, sx, flow);
  let sp = tv1(uv, 60, 15u);
  let speck = (1.0 - smoothstep(0.05, 0.16, sp.f1)) * select(0.0, 1.0, sp.id > 0.9);
  let h = tunit(0.56 * dome + 0.14 * stroke + 0.3 * mott);
  let tone = 0.74 + 0.46 * h + 0.34 * (mott - 0.5) + 0.2 * (rid(tuft.id, 9.7) - 0.5) - 0.3 * speck;
  return vec3f(h, tone, 0.3 + 0.7 * pow(dome, 0.55) - 0.25 * speck);
}

// SOIL: loam: crumbs and grit, pebbles of two sizes (lighter, with a dark rim where they sit), dark clods.
fn lSoil(uv: vec2f) -> vec3f {
  let g = tfbm(uv, 20, 5, 0.55, 51u);
  let crumb = tv1(uv + tbend(uv, 6, 0.01, 52u), 17, 54u);
  let dome = tunit(1.0 - crumb.f1 * 1.2);
  let clods = smoothstep(0.62, 0.8, tfbm(uv, 7, 3, 0.55, 57u));
  let pb = tv1(uv, 9, 55u);
  let pbR = 0.2 + 0.2 * rid(pb.id, 7.3);
  let pbOn = select(0.0, 1.0, pb.id > 0.8);
  let big = pbOn * smoothstep(pbR, pbR * 0.6, pb.f1);
  let bigRim = pbOn * (smoothstep(pbR * 1.7, pbR, pb.f1) - smoothstep(pbR, pbR * 0.8, pb.f1));
  let ps = tv1(uv + 0.31, 27, 56u);
  let psR = 0.14 + 0.16 * rid(ps.id, 5.9);
  let psOn = select(0.0, 1.0, ps.id > 0.62);
  let small = psOn * smoothstep(psR, psR * 0.55, ps.f1);
  let pebble = max(big, small);
  let tint = 1.1 + 0.5 * rid(pb.id * 3.0 + ps.id, 13.1);
  let h = tunit(0.3 * g + 0.2 * dome + 0.55 * big + 0.4 * small);
  let tone = (0.84 + 0.4 * g + 0.2 * (dome - 0.5) - 0.2 * clods) * mix(1.0, tint, pebble) * (1.0 - 0.2 * max(bigRim, 0.0));
  return vec3f(h, tone, tunit(0.55 + 0.45 * h) * (1.0 - 0.45 * bigRim));
}

// CRACK: baked clay: plates of two sizes (some split again) with lifted rims, divided by shallow cracks.
fn lCrack(uv: vec2f) -> vec3f {
  let uw = uv + tbend(uv, 7, 0.03, 61u);
  let v = tv1(uw, 5, 63u);
  let v2 = tv1(uw + 0.43, 11, 65u);
  let split = select(0.0, 1.0, rid(v.id, 5.7) > 0.55);
  let e1 = v.f2 - v.f1;
  let e2 = (v2.f2 - v2.f1) * 0.5;
  let edge = mix(e1, min(e1, e2 + 0.04), split);
  let crack = 1.0 - smoothstep(0.014, 0.085, edge);
  let plate = smoothstep(0.0, 0.5, edge);
  let curl = smoothstep(0.4, 0.08, edge) * (1.0 - crack);
  let g = tfbm(uv, 32, 3, 0.5, 64u);
  let h = tunit(0.5 * plate + 0.3 * curl + 0.2 * g - 0.45 * crack);
  let tone = (0.92 + 0.2 * (rid(mix(v.id, v2.id, split), 13.7) - 0.5) + 0.32 * (g - 0.5)) * mix(1.0, 0.5, crack);
  return vec3f(h, tone, mix(1.0, 0.3, crack) * (0.8 + 0.2 * plate));
}

// SAND: wind ripples (a steep lee face) in drifts, a second fainter set across them, over fine grain.
fn lSand(uv: vec2f) -> vec3f {
  let wnd = tfbm(uv, 3, 3, 0.5, 71u);
  let am = smoothstep(0.3, 0.62, tfbm(uv, 3, 2, 0.5, 74u));
  let am2 = smoothstep(0.35, 0.7, tfbm(uv + 0.4, 4, 2, 0.5, 75u));
  let ph = uv.x * 26.0 + uv.y * 9.0 + wnd * 3.6;
  let r = fract(ph);
  let prof = select(1.0 - smoothstep(0.8, 1.0, r), smoothstep(0.0, 0.8, r), r < 0.8);
  let r2 = fract(uv.x * 7.0 - uv.y * 19.0 + wnd * 2.0);
  let prof2 = 0.5 - 0.5 * cos(TAU * r2);
  let grain = tn1(uv, 256, 72u);
  let spot = tn1(uv, 40, 73u);
  let h = tunit(0.62 * prof * (0.2 + 0.8 * am) + 0.22 * prof2 * am2 + 0.12 * grain + 0.06 * spot);
  let tone = 0.92 + 0.1 * prof * am + 0.06 * prof2 * am2 + 0.2 * (grain - 0.5) + 0.14 * (spot - 0.5);
  return vec3f(h, tone, 0.85 + 0.15 * prof);
}

// GRAVEL: packed pebbles, each its own tone, dark in the gaps, small stones between the large.
fn lGravel(uv: vec2f) -> vec3f {
  let v = tv1(uv + tbend(uv, 9, 0.012, 81u), 11, 83u);
  let dome = tunit(1.0 - v.f1 / 0.66);
  let hh = pow(dome, 0.6);
  let sm = tv1(uv, 31, 85u);
  let smd = pow(tunit(1.0 - sm.f1 / 0.62), 0.6);
  let gap = 1.0 - smoothstep(0.1, 0.5, dome);
  let fine = tn1(uv, 128, 84u);
  let hz = mix(hh, smd * 0.55, gap);
  let tone = (0.66 + 0.75 * rid(mix(v.id, sm.id, gap), 9.1)) * (0.55 + 0.45 * hz) * (0.9 + 0.2 * fine);
  return vec3f(tunit(hz * 0.9 + 0.1 * fine), tone, 0.2 + 0.8 * hz);
}

// ROCK A: fractured stone: blocks, each a tilted plane at a level of its own and ledged against its neighbours, a
// finer set of facets on them, cracks that open along half the borders (wide and dark, or a hairline), the blocks'
// edges worn bright, a soft roughness over all, and weathering streaks running down.
fn lRockA(uv: vec2f) -> vec3f {
  let uw = uv + tbend(uv, 4, 0.03, 91u);
  let big = tfac2(uw, 4, 1.7, 0.12, 93u);
  let mid = tfac2(uw + 0.21, 9, 1.0, 0.14, 94u);
  let wide = 0.035 + 0.06 * rid(big.z, 7.1);
  let crack = select(0.0, 1.0, big.z > 0.5) * (1.0 - smoothstep(0.0, wide, big.y)) + select(0.0, 0.7, mid.z > 0.6) * (1.0 - smoothstep(0.0, 0.03, mid.y));
  let worn = (1.0 - smoothstep(0.0, 0.2, big.y)) * (1.0 - min(crack, 1.0));
  let rough = 1.0 - abs(2.0 * tfbm(uv, 9, 4, 0.55, 95u) - 1.0);
  let fine = tn1(uv, 80, 96u);
  let streak = tn(uv * vec2f(6.0, 48.0), vec2i(6, 48), 97u);
  let h = tunit(0.5 + 0.34 * big.x + 0.2 * mid.x + 0.18 * (rough - 0.5) + 0.06 * (fine - 0.5) - 0.3 * min(crack, 1.0));
  let tone = (0.9 + 0.3 * (rid(big.w, 21.7) - 0.5) + 0.1 * mid.x + 0.2 * (rough - 0.5) + 0.12 * worn) * (0.88 + 0.24 * streak) * (1.0 - 0.5 * min(crack, 1.0));
  return vec3f(h, tone, tunit(0.6 + 0.4 * h) * (1.0 - 0.65 * min(crack, 1.0)));
}

// ROCK B: strata: bands of harder and softer rock of every thickness and wavy edges, the hard ones standing out as
// ledges; joints that cut some of them, slanted and tapering, each band's its own.
fn lRockB(uv: vec2f) -> vec3f {
  let y = uv.y;
  let wob = tn(vec2f(uv.x * 4.0, y * 3.0), vec2i(4, 3), 101u) - 0.5;
  let wob2 = tn(vec2f(uv.x * 11.0, y * 6.0), vec2i(11, 6), 108u) - 0.5;
  let yy = 8.0 * y + 0.5 * sin(TAU * (2.0 * y + 0.13)) + 0.3 * sin(TAU * (3.0 * y + 0.61)) + 1.0 * wob + 0.4 * wob2;
  let band = floor(yy);
  let fr = fract(yy);
  let b = i32(band);
  let hard = f32(hl(b, 0, vec2i(8, 1), 103u) & 0xffffu) * (1.0 / 65535.0);
  let ledge = smoothstep(0.0, 0.12, fr) * (1.0 - smoothstep(0.6 + 0.3 * hard, 1.0, fr));
  // joints: up to three a band, each at a place of its own, slanting across the band and tapering toward its ends
  var joint = 0.0;
  for (var c = 0; c < 3; c++) {
    let hj = hl(b * 3 + c, 7, vec2i(24, 1), 105u);
    let on = select(0.0, 1.0, (hj & 0xffu) < 80u);
    let jx = (f32(c) + 0.1 + 0.8 * f32((hj >> 8u) & 0xffu) / 255.0) / 3.0 + (fr - 0.5) * 0.06 * (f32((hj >> 16u) & 0xffu) / 255.0 - 0.5);
    var dx = abs(uv.x - jx);
    dx = min(dx, 1.0 - dx);
    let wj = 0.004 + 0.012 * f32((hj >> 24u) & 0xffu) / 255.0;
    joint = max(joint, on * (1.0 - smoothstep(0.0, wj * sqrt(sin(3.14159 * clamp(fr, 0.0, 1.0))), dx)));
  }
  let chips = tfbm(uv, 16, 4, 0.55, 106u);
  let streak = tn(uv * vec2f(9.0, 80.0), vec2i(9, 80), 107u);
  let h = tunit(0.26 * hard + 0.5 * ledge * (0.4 + 0.6 * hard) + 0.2 * chips - 0.3 * joint);
  let tone = (0.76 + 0.42 * hard + 0.26 * (chips - 0.5)) * (1.0 - 0.3 * smoothstep(0.65, 1.0, fr)) * (1.0 - 0.45 * joint) * (0.88 + 0.24 * streak);
  return vec3f(h, tone, tunit(0.3 + 0.7 * ledge - 0.45 * joint));
}

// ROCK C: dark volcanic stone: slabs, a few vesicles (little pits), a faint grain.
fn lRockC(uv: vec2f) -> vec3f {
  let v = tv1(uv + tbend(uv, 6, 0.015, 111u), 5, 113u);
  let edge = v.f2 - v.f1;
  let plate = smoothstep(0.0, 0.3, edge);
  let tilt = (vec2f(rid(v.id, 41.3), rid(v.id, 17.9)) - 0.5) * 0.7;
  let crack = 1.0 - smoothstep(0.015, 0.06, edge);
  let ves = tv1(uv, 30, 114u);
  let vr = 0.1 + 0.12 * rid(ves.id, 3.7);
  let pit = (1.0 - smoothstep(vr * 0.6, vr, ves.f1)) * select(0.0, 1.0, ves.id > 0.72);
  let g = tfbm(uv, 20, 4, 0.55, 115u);
  let h = tunit(0.5 * plate + 0.3 * g + 0.5 * dot(v.d, tilt) - 0.35 * pit - 0.35 * crack + 0.1);
  let tone = (0.86 + 0.3 * (g - 0.5) + 0.16 * (rid(v.id, 11.0) - 0.5)) * (1.0 - 0.45 * pit) * (1.0 - 0.4 * crack);
  return vec3f(h, tone, tunit(1.0 - 0.6 * pit - 0.6 * crack));
}

// MOSS: cushions: soft domes of two sizes, dark where they meet.
fn lMoss(uv: vec2f) -> vec3f {
  let a = tv1(uv + tbend(uv, 4, 0.02, 120u), 8, 121u);
  let b = tv1(uv + 0.17, 20, 122u);
  let da = pow(tunit(1.0 - a.f1 / 0.72), 0.7);
  let db = pow(tunit(1.0 - b.f1 / 0.72), 0.7);
  let g = tfbm(uv, 24, 3, 0.55, 123u);
  let h = tunit(0.55 * da + 0.25 * db + 0.2 * g);
  let tone = 0.55 + 0.7 * h + 0.18 * (g - 0.5) + 0.12 * (rid(a.id, 7.7) - 0.5);
  return vec3f(h, tone, 0.3 + 0.7 * h);
}

fn layerAt(layer: u32, uv: vec2f) -> vec3f {
  switch (layer) {
    case 0u: { return lTurf(uv); }
    case 1u: { return lSoil(uv); }
    case 2u: { return lCrack(uv); }
    case 3u: { return lSand(uv); }
    case 4u: { return lGravel(uv); }
    case 5u: { return lRockA(uv); }
    case 6u: { return lRockB(uv); }
    case 7u: { return lRockC(uv); }
    default: { return lMoss(uv); }
  }
}

// (a pipeline a layer: its constant folds the switch away, and the nine compile side by side)
override LAYER: u32 = 0u;

@compute @workgroup_size(8, 8, 1)
fn paint(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= SIZE || id.y >= SIZE) { return; }
  let uv = (vec2f(id.xy) + 0.5) / f32(SIZE);
  textureStore(dst, id.xy, LAYER, vec4f(layerAt(LAYER, uv), 1.0));
}
`;
}

/** The shade pass: slope from the painted heights (a texel's neighbours wrap: the layer tiles), tone and cavity as they are. */
export function groundShadeWGSL(): string {
  const tile = GROUND_LAYERS.map((n) => f(GROUND_TILE[n])).join(', ');
  const relief = GROUND_LAYERS.map((n) => f(GROUND_RELIEF[n])).join(', ');
  return /* wgsl */ `
const SIZE: i32 = ${GROUND_SIZE};
const TILE_M = array<f32, ${GROUND_LAYERS.length}>(${tile});
const RELIEF_M = array<f32, ${GROUND_LAYERS.length}>(${relief});

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn shade(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(SIZE) || id.y >= u32(SIZE)) { return; }
  let p = vec2i(id.xy);
  let l = i32(id.z);
  let c = textureLoad(src, p, l, 0);
  let hx0 = textureLoad(src, vec2i((p.x + SIZE - 1) % SIZE, p.y), l, 0).x;
  let hx1 = textureLoad(src, vec2i((p.x + 1) % SIZE, p.y), l, 0).x;
  let hy0 = textureLoad(src, vec2i(p.x, (p.y + SIZE - 1) % SIZE), l, 0).x;
  let hy1 = textureLoad(src, vec2i(p.x, (p.y + 1) % SIZE), l, 0).x;
  // slope in metres per metre: the layer's height (0 … 1 of its relief) over the metres between the two samples
  let k = RELIEF_M[id.z] * f32(SIZE) / (2.0 * TILE_M[id.z]);
  let n = clamp(vec2f(hx0 - hx1, hy0 - hy1) * k, vec2f(-1.0), vec2f(1.0));
  textureStore(dst, id.xy, id.z, vec4f(0.5 + 0.5 * n, clamp(c.y * 0.5, 0.0, 1.0), clamp(c.z, 0.0, 1.0)));
}
`;
}

/** The mip pass: each texel of level m + 1 the mean of the four it covers in level m. */
export const GROUND_MIP_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn mip(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let o = vec2i(id.xy) * 2;
  let l = i32(id.z);
  let s = textureLoad(src, o, l, 0) + textureLoad(src, o + vec2i(1, 0), l, 0)
        + textureLoad(src, o + vec2i(0, 1), l, 0) + textureLoad(src, o + vec2i(1, 1), l, 0);
  textureStore(dst, id.xy, id.z, s * 0.25);
}
`;

/**
 * What the surfaces read the layers with (binding 41 the array, 42 its sampler: repeat, anisotropic).
 * A tap is a vec4 as painted: .rg the slope (½ = flat), .b the tone (½ = ×1), .a the cavity.
 * The gradients are the world-metre derivatives of the position (dpdx, dpdy), taken where the control
 * flow is uniform, so a tap may sit in a branch.
 */
export const GROUND_SAMPLE_WGSL = /* wgsl */ `
@group(0) @binding(41) var groundTex: texture_2d_array<f32>;
@group(0) @binding(42) var groundSamp: sampler;
${GROUND_CONSTS_WGSL}
fn gTap(layer: i32, uv: vec2f, dx: vec2f, dy: vec2f) -> vec4f {
  return textureSampleGrad(groundTex, groundSamp, uv, layer, dx, dy);
}
// A layer laid on the ground (seen from above), at its own tile, and again at a fifth of that scale
// turned 37°: their tones multiply, so the tile's repeat does not show.
fn gTop(layer: i32, p: vec2f, ddx: vec2f, ddy: vec2f) -> vec4f {
  let s = GT_INV[layer];
  let a = gTap(layer, p * s, ddx * s, ddy * s);
  let r = mat2x2f(0.7986, 0.6018, -0.6018, 0.7986);
  let s2 = s * 0.2;
  let b = gTap(layer, r * p * s2 + 0.37, r * ddx * s2, r * ddy * s2);
  return vec4f(a.rg + (b.rg - 0.5) * 0.2, a.b * (0.55 + 0.9 * b.b), a.a * (0.7 + 0.6 * b.a));
}
`;
