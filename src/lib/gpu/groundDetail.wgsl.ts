/**
 * The ground's detail, as the raster terrain's fragment shader finishes it (terrain.wgsl.ts shadeGround).
 * ---------------------------------------------------------------------------
 * The clipmap's biome layers say what the ground is made of at a lattice point (soil, grass, rock, their
 * colours, the cover, the strata); the textures (groundTex.wgsl.ts) say what its surface is like up close:
 * the turf's tufts, the soil's grit and pebbles, baked clay's plates, sand's ripples, scree, fractured
 * stone, strata, dark volcanic stone, moss. The colour stays the biome's; a texture only moves it (its
 * tone), tilts the shading normal (its slope) and darkens the crevices (its cavity).
 *
 *   turf    grass over the ground (where the cover is high), soil's loam where it is not
 *   soil    loam; baked clay in dry country; sand where a desert's cover gives out; scree on bare slopes
 *           and pebbles at a shore
 *   rock    on the steep and in outcrops, seen from the side it stands facing (two of the three axes:
 *           a biplanar projection, the strata always level): fractured stone, strata where the biome
 *           bands its rock, dark slabs where its rock is black; moss on its upper faces in wet lands
 *
 * Where a texture has a height (turf, which grass thins out of) it also decides where the grass gives way
 * to bare earth, so the edge of a lawn, a path or a bare patch is ragged and not a smooth gradient.
 * Past DETAIL_NEAR the detail fades out (its mips have long averaged it out), gone by DETAIL_FAR: the
 * far ground costs no taps.
 */

import { GROUND_LAYERS } from './groundTex.wgsl';
import { wgslF32 as f } from './worldgen.wgsl';

/** Per layer (GROUND_LAYERS order): how steeply its slope tilts the shading normal. */
const NORMAL_K: Record<(typeof GROUND_LAYERS)[number], number> = {
  turf: 0.2, soil: 0.3, crack: 0.4, sand: 0.3, gravel: 0.45, rockA: 0.35, rockB: 0.35, rockC: 0.35, moss: 0.3,
};
/** Per layer: 1 / its mean tone (so the average colour stays the biome's) and the cavity that counts as open ground. */
const TONE_GAIN: Record<(typeof GROUND_LAYERS)[number], number> = {
  turf: 1.04, soil: 0.98, crack: 1.19, sand: 1.03, gravel: 1.24, rockA: 1.02, rockB: 1.03, rockC: 1.21, moss: 1.1,
};
const CAV_REF: Record<(typeof GROUND_LAYERS)[number], number> = {
  turf: 0.82, soil: 0.7, crack: 0.9, sand: 0.95, gravel: 0.7, rockA: 0.9, rockB: 0.9, rockC: 0.95, moss: 0.75,
};

const tab = (t: Record<string, number>) => GROUND_LAYERS.map((n) => f(t[n])).join(', ');

export const GROUND_DETAIL_WGSL = /* wgsl */ `
// Past DETAIL_NEAR the detail fades; from DETAIL_FAR it is not read at all.
const DETAIL_NEAR: f32 = 50.0;
const DETAIL_FAR: f32 = 160.0;
override DETAIL_LEVEL: u32 = 3u;
const K_TONE: f32 = 0.5;
const K_CAV: f32 = 0.65;
const NORMAL_K = array<f32, ${GROUND_LAYERS.length}>(${tab(NORMAL_K)});
const TONE_GAIN = array<f32, ${GROUND_LAYERS.length}>(${tab(TONE_GAIN)});
const CAV_REF = array<f32, ${GROUND_LAYERS.length}>(${tab(CAV_REF)});

// A tap as the shading wants it: tone (×1 = 1), slope (the tilt of the normal), cavity (1 = open).
struct Tap { t: f32, s: vec2f, a: f32 };
fn tapOf(layer: i32, c: vec4f) -> Tap {
  return Tap(c.b * 2.0 * TONE_GAIN[layer], (c.rg * 2.0 - 1.0) * NORMAL_K[layer], min(c.a / CAV_REF[layer], 1.0));
}
// The same layer laid on the ground at its own tile only (no second scale).
fn gOne(layer: i32, xz: vec2f, dx: vec2f, dy: vec2f) -> Tap {
  let s = GT_INV[layer];
  return tapOf(layer, gTap(layer, xz * s, dx * s, dy * s));
}

// A rock layer on a surface facing n, from its two (three, on a diagonal) best-facing axes: the
// tone and cavity, and the tilt as an offset of the normal in world axes. The strata (v = height)
// stay level on every face.
struct Rock { t: f32, a: f32, o: vec3f };
fn rockOn(layer: i32, p: vec3f, n: vec3f, ddx: vec3f, ddy: vec3f) -> Rock {
  let an = abs(n);
  var w = an * an;
  w = w * w;
  w = w / (w.x + w.y + w.z);
  let s = GT_INV[layer];
  var r = Rock(0.0, 0.0, vec3f(0.0));
  var ws = 0.0;
  // (the lookup bent by a slow wave a few tiles long, so the tile's repeat does not stand in rows on a wall: each copy is bent differently)
  let bend = vec2f(sin(p.y * 0.0371 + p.x * 0.0173 + p.z * 0.0123), sin(p.x * 0.0287 - p.y * 0.0309 + p.z * 0.0241)) * 0.11;
  if (w.y > 0.07) {
    let c = tapOf(layer, gTap(layer, p.xz * s + bend, ddx.xz * s, ddy.xz * s));
    r.t += w.y * c.t; r.a += w.y * c.a; r.o += w.y * vec3f(c.s.x, 0.0, c.s.y); ws += w.y;
  }
  if (w.x > 0.07) {
    let c = tapOf(layer, gTap(layer, vec2f(p.z, p.y) * s + bend, vec2f(ddx.z, ddx.y) * s, vec2f(ddy.z, ddy.y) * s));
    r.t += w.x * c.t; r.a += w.x * c.a; r.o += w.x * vec3f(0.0, c.s.y, c.s.x); ws += w.x;
  }
  if (w.z > 0.07) {
    let c = tapOf(layer, gTap(layer, vec2f(p.x, p.y) * s + bend, vec2f(ddx.x, ddx.y) * s, vec2f(ddy.x, ddy.y) * s));
    r.t += w.z * c.t; r.a += w.z * c.a; r.o += w.z * vec3f(c.s.x, c.s.y, 0.0); ws += w.z;
  }
  let k = 1.0 / max(ws, 1e-3);
  return Rock(r.t * k, r.a * k, r.o * k);
}

// One tap of a rock layer at five times its scale, from the axis the surface faces most: the fine chips and
// grain a wall shows close up (the broad tile alone is a soft smear at arm's length).
fn rockFine(layer: i32, p: vec3f, n: vec3f, ddx: vec3f, ddy: vec3f) -> Rock {
  let an = abs(n);
  let s = GT_INV[layer] * 5.0;
  var c: Tap;
  var o: vec3f;
  if (an.y >= an.x && an.y >= an.z) {
    c = tapOf(layer, gTap(layer, p.xz * s, ddx.xz * s, ddy.xz * s));
    o = vec3f(c.s.x, 0.0, c.s.y);
  } else if (an.x >= an.z) {
    c = tapOf(layer, gTap(layer, vec2f(p.z, p.y) * s, vec2f(ddx.z, ddx.y) * s, vec2f(ddy.z, ddy.y) * s));
    o = vec3f(0.0, c.s.y, c.s.x);
  } else {
    c = tapOf(layer, gTap(layer, vec2f(p.x, p.y) * s, vec2f(ddx.x, ddx.y) * s, vec2f(ddy.x, ddy.y) * s));
    o = vec3f(c.s.x, c.s.y, 0.0);
  }
  return Rock(c.t, c.a, o * 0.2);
}

// The biome's rock at p (rc: its colour as the clipmap stores it, √rgb; strata): banded where it has strata,
// black where it is volcanic, else fractured stone; the two kinds with most of the weight, in one loop (the code is
// one path, and the second is read only where the biome's rock is part way between two). l0: the main kind.
struct RockM { r: Rock, l0: i32 };
fn rockMix(rc: vec3f, strata: f32, p: vec3f, n: vec3f, ddx: vec3f, ddy: vec3f) -> RockM {
  let dark = 1.0 - smoothstep(0.23, 0.31, dot(rc, vec3f(0.3, 0.59, 0.11)));
  let kB = smoothstep(0.2, 0.7, strata);
  let kC = smoothstep(0.3, 0.7, dark) * (1.0 - kB);
  let kA = 1.0 - kB - kC;
  var l0 = GL_ROCK_A;
  var w0 = kA;
  if (kB > w0) { l0 = GL_ROCK_B; w0 = kB; }
  if (kC > w0) { l0 = GL_ROCK_C; w0 = kC; }
  var l1 = GL_ROCK_A;
  var w1 = kA;
  if (l0 == GL_ROCK_A) {
    l1 = select(GL_ROCK_C, GL_ROCK_B, kB >= kC);
    w1 = max(kB, kC);
  } else {
    let other = select(GL_ROCK_C, GL_ROCK_B, l0 == GL_ROCK_C);
    let wo = select(kC, kB, l0 == GL_ROCK_C);
    if (wo > w1) { l1 = other; w1 = wo; }
  }
  var r = Rock(0.0, 0.0, vec3f(0.0));
  for (var k = 0; k < 2; k++) {
    let layer = select(l0, l1, k == 1);
    let wk = select(w0, w1, k == 1);
    if (wk > 0.02) {
      let q = rockOn(layer, p, n, ddx, ddy);
      r = Rock(r.t + wk * q.t, r.a + wk * q.a, r.o + wk * q.o);
    }
  }
  return RockM(r, l0);
}

// The soil's tap here: loam, or what the dry country, a bare slope or a shore make of it.
fn soilOn(gc: GroundCache, p: vec3f, slope: f32, ws: f32, dx: vec2f, dy: vec2f) -> Tap {
  let sc = gc.bioA.rgb;
  let luma = dot(sc * sc, vec3f(0.3, 0.59, 0.11));
  let cover = gc.bioA.a;
  let strata = gc.bioC.a;
  // baked clay where the soil is the dry country's, sand where its cover gives out (but not in banded badlands)
  let arid = smoothstep(0.2, 0.26, luma);
  let sandK = arid * (1.0 - smoothstep(0.1, 0.4, cover)) * (1.0 - smoothstep(0.4, 0.8, strata));
  let crackK = arid - sandK;
  // scree on a bare slope, pebbles where water meets the bank
  let above = p.y - gc.wt;
  let shore = (1.0 - smoothstep(0.1, 1.6, above)) * (1.0 - smoothstep(0.3, 0.7, slope)) * step(SEA + 1.5, gc.wt);
  let gravK = max(smoothstep(0.32, 0.6, slope + (1.0 - cover) * 0.25) * (1.0 - arid), shore);
  var altL = GL_CRACK;
  var altK = crackK;
  if (sandK > altK) { altL = GL_SAND; altK = sandK; }
  if (gravK > altK) { altL = GL_GRAVEL; altK = gravK; }
  altK = min(altK, 1.0);
  var t = Tap(1.0, vec2f(0.0), 1.0);
  if (altK < 0.98) {
    let s = GT_INV[GL_SOIL];
    let a = tapOf(GL_SOIL, gTap(GL_SOIL, p.xz * s, dx * s, dy * s));
    t = a;
    // (a second scale, turned, so the tile's repeat does not show: where the soil is a share of the ground only, the blades and the turf cover it)
    let k2 = smoothstep(0.2, 0.5, ws);
    if (k2 > 0.0) {
      let r = mat2x2f(0.7986, 0.6018, -0.6018, 0.7986);
      let s2 = s * 0.23;
      let b = gTap(GL_SOIL, r * p.xz * s2 + 0.37, r * dx * s2, r * dy * s2);
      t = Tap(a.t * mix(1.0, 0.6 + 0.8 * b.b, k2), a.s, a.a * mix(1.0, 0.75 + 0.5 * b.a, k2));
    }
  }
  if (altK > 0.02) {
    let c = gOne(altL, p.xz, dx, dy);
    t = Tap(mix(t.t, c.t, altK), mix(t.s, c.s, altK), mix(t.a, c.a, altK));
  }
  return t;
}

// What the detail did at a pixel: the colour's multiplier, the shading normal, the cavity (ambient), and
// how much of the surface is moss (the caller colours it).
struct Detail { tone: f32, n: vec3f, cav: f32, moss: f32, grey: f32 };

// w: the shares of grass, soil, rock and sand in the colour (look.w; the snow's share is what is left of 1).
// DETAIL_LEVEL (bench: what each part costs): 1 the turf only, 2 and the soil and sand, 3 all.
fn groundDetail(gc: GroundCache, p: vec3f, n: vec3f, slope: f32, w: vec4f, turf: vec4f, ddx: vec3f, ddy: vec3f, dk: f32, tS: f32) -> Detail {
  let dx = ddx.xz;
  let dy = ddy.xz;
  var tone = 1.0 - (w.x + w.y + w.z + w.w);
  var cav = tone;
  var off = vec3f(0.0);
  var moss = 0.0;
  var grey = 0.0;
  if (w.x > 0.01) {
    let c = tapOf(GL_TURF, turf);
    // (the lawn's tone moves twice as far as the others': it is the ground the eye reads between the tufts)
    tone += w.x * (1.0 + (c.t - 1.0) * 2.0); cav += w.x * c.a; off += w.x * vec3f(c.s.x, 0.0, c.s.y);
  }
  if (w.y > 0.01 && DETAIL_LEVEL >= 2u) {
    let c = soilOn(gc, p, slope, w.y, dx, dy);
    tone += w.y * c.t; cav += w.y * c.a; off += w.y * vec3f(c.s.x, 0.0, c.s.y);
    // (the stones in the soil, its bright outliers, are grey where the earth is brown)
    grey = w.y * smoothstep(1.2, 1.6, c.t);
  }
  if (w.w > 0.01 && DETAIL_LEVEL >= 2u) {
    let c = gOne(GL_SAND, p.xz, dx, dy);
    tone += w.w * c.t; cav += w.w * c.a; off += w.w * vec3f(c.s.x, 0.0, c.s.y);
  }
  if (w.z > 0.01 && DETAIL_LEVEL >= 3u) {
    // the biome's rock: banded where it has strata, black where it is volcanic, else fractured stone
    let rm = rockMix(gc.bioC.rgb, gc.bioC.a, p, n, ddx, ddy);
    var r = rm.r;
    let l0 = rm.l0;
    // close up, the fine chips of the main kind over the broad forms
    let nearK = 1.0 - smoothstep(2.5, 14.0, tS);
    if (nearK > 0.02) {
      let q = rockFine(l0, p, n, ddx, ddy);
      r = Rock(r.t * mix(1.0, q.t, nearK), r.a * mix(1.0, q.a, 0.6 * nearK), r.o + q.o * nearK * 8.0);
    }
    tone += w.z * r.t; cav += w.z * r.a; off += w.z * r.o;
    // moss on the upper faces of rock in lush country
    let lush = clamp(max(gc.mat.y, 0.0) * 1.2 + 0.22 * gc.bioA.a, 0.0, 1.0);
    let up = smoothstep(0.55, 0.88, n.y);
    if (lush * up > 0.02) {
      let m = gOne(GL_MOSS, p.xz, dx, dy);
      moss = w.z * smoothstep(0.22, 0.55, lush * up * (0.55 + 0.9 * m.a) * (0.7 + 0.6 * r.a));
      tone = mix(tone, tone * 0.85 + w.z * (m.t - 0.85) * 0.5, moss);
      off = mix(off, w.z * vec3f(m.s.x, 0.0, m.s.y), moss);
    }
  }
  var d: Detail;
  d.tone = mix(1.0, tone, dk * K_TONE);
  d.cav = mix(1.0, cav, dk * K_CAV);
  d.n = normalize(n + off * dk);
  d.moss = moss * dk;
  d.grey = grey * dk;
  return d;
}
`;
