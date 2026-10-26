/**
 * Combat in WGSL: the Bokoblins and what flies and flares (combat/world.ts,
 * drawn by gpu/combatGpu.ts).
 * ---------------------------------------------------------------------------
 * A Bokoblin is a signed-distance body round its rig's joints, as the hero is
 * (hero.wgsl.ts), sphere-traced in its own scissored full-screen pass (one
 * uniform block a foe, at a dynamic offset): a pot-bellied trunk, a big round
 * head with a pig's snout, long ears, one horn and two tusks, thick limbs, a
 * loincloth, and its club or bow (the hero's gear code: gearWeapon / gearBow).
 * Its rig is the hero's, so its body is traced in its own unit space — the
 * ray taken to it, about its root, by 1 / its scale — and the hit and depth
 * brought back. A blow taken flashes it white.
 *
 * Parts: lit boxes, each one instance of 36 vertices made here (arrows stuck
 * and in flight, thrown and dropped weapons, a camp's logs and stones).
 * Sprites: camera-facing quads blended premultiplied (alpha 0 adds light):
 * glows, smoke puffs, streaks between two points (sparks), a slash's trail
 * (a quad of four corners), rings, flames and star-shaped glints.
 *
 * Needs, from the raster shader: u (Frame), the Hero block and its helpers
 * (sdCapsule, sdLimb, sdBox3, sdCone, smin, hU, the gear's functions,
 * heroShadow), project(), cameraRay(), display(), surfaceFog, keyDir, keyCol,
 * skyAmb, bounceCol, skyHorCol, sunVisibility, vnoise, CAMERA_NEAR.
 */

import { HERO_JOINTS, HERO_UNIFORM, GEAR, jointPad, packedGearPoints } from '../hero/rig';

/** Bytes of one foe's uniform block, and the stride between blocks (a dynamic offset's alignment). */
export const FOE_BYTES = 18 * 16 + 48 + 48 + 5 * 16 + GEAR.length * 4 + 16;
export const FOE_STRIDE = 768;
/** Bytes per part and per sprite in the GPU's lists. */
export const PART_BYTES = 64;
export const SPRITE_BYTES = 80;
/** Sprite kinds (combat/fx.ts). */
export const SPRITE = { glow: 0, puff: 1, streak: 2, trail: 3, ring: 4, flame: 5, star: 6 } as const;

// Offsets (floats) in a foe's block.
const F_TORSO = 72, F_HEAD = 84, F_BOUND = 96, F_MISC = 100, F_LO = 104, F_HI = 108, F_SKIN = 112, F_GEAR = 116, F_ROOT = F_GEAR + GEAR.length;

/**
 * A foe's block (into `out` at float offset `o`) from its rig packing (rig.ts
 * HERO_UNIFORM, camera-relative, in its unit space: as if of the hero's size),
 * its root (camera-relative), its scale, the flash of a blow (0..1) and its kind.
 */
export function packFoeWgsl(src: Float32Array, rx: number, ry: number, rz: number, scale: number, flash: number, kind: number, out: Float32Array, o: number): void {
  for (let j = 0; j < HERO_JOINTS; j++) {
    out[o + 4 * j] = src[3 * j]; out[o + 4 * j + 1] = src[3 * j + 1]; out[o + 4 * j + 2] = src[3 * j + 2]; out[o + 4 * j + 3] = 0;
  }
  for (let c = 0; c < 3; c++) {
    for (let k = 0; k < 3; k++) {
      out[o + F_TORSO + 4 * c + k] = src[HERO_UNIFORM.torso + 3 * c + k];
      out[o + F_HEAD + 4 * c + k] = src[HERO_UNIFORM.head + 3 * c + k];
    }
    out[o + F_TORSO + 4 * c + 3] = 0;
    out[o + F_HEAD + 4 * c + 3] = 0;
  }
  for (let k = 0; k < 4; k++) out[o + F_BOUND + k] = src[HERO_UNIFORM.bound + k];
  out[o + F_MISC] = scale; out[o + F_MISC + 1] = src[HERO_UNIFORM.misc + 1]; out[o + F_MISC + 2] = flash; out[o + F_MISC + 3] = kind;
  // the box round every part, in its unit space (as packHeroWgsl's, with a Bokoblin's bulk)
  for (let c = 0; c < 3; c++) { out[o + F_LO + c] = Infinity; out[o + F_HI + c] = -Infinity; }
  for (let j = 0; j < HERO_JOINTS - 1; j++) {
    const pad = jointPad(j, 0) + 0.06;
    for (let c = 0; c < 3; c++) {
      const x = src[3 * j + c];
      if (x - pad < out[o + F_LO + c]) out[o + F_LO + c] = x - pad;
      if (x + pad > out[o + F_HI + c]) out[o + F_HI + c] = x + pad;
    }
  }
  const gp = packedGearPoints(src);
  for (let k = 0; k < gp.n; k++) {
    const pad = gp.pts[4 * k + 3];
    for (let c = 0; c < 3; c++) {
      const x = gp.pts[4 * k + c];
      if (x - pad < out[o + F_LO + c]) out[o + F_LO + c] = x - pad;
      if (x + pad > out[o + F_HI + c]) out[o + F_HI + c] = x + pad;
    }
  }
  out[o + F_LO + 3] = 0; out[o + F_HI + 3] = 0;
  // (its bound grows by the same bulk)
  out[o + F_BOUND + 3] += 0.06;
  for (let k = 0; k < 3; k++) out[o + F_SKIN + k] = src[HERO_UNIFORM.dress + k];
  out[o + F_SKIN + 3] = 0;
  for (let k = 0; k < GEAR.length; k++) out[o + F_GEAR + k] = src[HERO_UNIFORM.gear + k];
  out[o + F_ROOT] = rx; out[o + F_ROOT + 1] = ry; out[o + F_ROOT + 2] = rz; out[o + F_ROOT + 3] = 0;
}

/** A foe block's box (unit space) taken to the world about its root: [min xyz, max xyz] into `box`. */
export function foeWorldBox(b: Float32Array, o: number, box: Float64Array): void {
  const s = b[o + F_MISC];
  for (let c = 0; c < 3; c++) {
    const r = b[o + F_ROOT + c];
    box[c] = r + (b[o + F_LO + c] - r) * s;
    box[3 + c] = r + (b[o + F_HI + c] - r) * s;
  }
}

/** A foe block's bounding sphere in the world: centre xyz, radius into `sp`. */
export function foeWorldSphere(b: Float32Array, o: number, sp: Float64Array): void {
  const s = b[o + F_MISC];
  for (let c = 0; c < 3; c++) {
    const r = b[o + F_ROOT + c];
    sp[c] = r + (b[o + F_BOUND + c] - r) * s;
  }
  sp[3] = b[o + F_BOUND + 3] * s;
}

export function combatWGSL(): string {
  return /* wgsl */ `
// ---- foes (combatGpu.ts: one block a foe, at a dynamic offset) ---------------------------------
struct Foe {
  j: array<vec4f, ${HERO_JOINTS}>, // joints in the foe's unit space (camera-relative)
  torso: mat3x3f,
  head: mat3x3f,
  bound: vec4f,                // bounding sphere (unit space)
  misc: vec4f,                 // x scale, y ground under it (camera-relative), z a blow's flash 0..1, w kind
  lo: vec4f,                   // box round every part (unit space)
  hi: vec4f,
  skin: vec4f,                 // rgb its skin's albedo
  gear: array<vec4f, 9>,       // rig.ts GEAR: its club (or bow)
  root: vec4f,                 // its root (camera-relative): the scale is about it
};
@group(1) @binding(50) var<uniform> foe: Foe;

const FOE_STEPS: i32 = 56;
const FC_CLOTH: vec3f = vec3f(0.07, 0.04, 0.022);
const FC_HORN: vec3f = vec3f(0.42, 0.37, 0.27);
const FC_EYE: vec3f = vec3f(0.55, 0.42, 0.04);
const FC_MOUTH: vec3f = vec3f(0.08, 0.015, 0.02);

fn foeBox(o: vec3f, d: vec3f) -> vec2f {
  let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
  let a = (foe.lo.xyz - o) * inv;
  let b = (foe.hi.xyz - o) * inv;
  let n = min(a, b);
  let x = max(a, b);
  return vec2f(max(max(n.x, n.y), n.z), min(min(x.x, x.y), x.z));
}

// Distance to the Bokoblin (unit space) and the material id: 1 skin, 2 cloth, 3 horn and tusks,
// then the gear's (hero.wgsl.ts).
fn foeMap(p: vec3f) -> vec2f {
  let pel = foe.j[0].xyz;
  let che = foe.j[1].xyz;
  let hd = foe.j[2].xyz;
  let q = (p - pel) * foe.torso;
  // a pot belly on a stooped trunk, heavy shoulders
  var body = smin(sdLimb(p, pel, che, 0.135, 0.125), length(q - vec3f(0.0, 0.14, 0.075)) - 0.165, 0.09);
  body = smin(body, sdCapsule(p, foe.j[3].xyz, foe.j[6].xyz, 0.072), 0.07);
  body = smin(body, sdCapsule(p, che, hd, 0.065), 0.05);
  var r = vec2f(body, 1.0);
  // a loincloth round the hips
  r = hU(r, sdCone(q - vec3f(0.0, -0.11, 0.0), 0.085, 0.175, 0.15), 2.0);
  if (length(p - hd) - 0.5 < r.x) {
    let h = (p - hd) * foe.head;
    let ha = vec3f(abs(h.x), h.yz);
    // a big round skull over a heavy jaw, the snout jutting from it
    var skin = smin(length(h - vec3f(0.0, 0.03, -0.01)) - 0.145, length(h - vec3f(0.0, -0.05, 0.07)) - 0.105, 0.07);
    skin = smin(skin, sdCapsule(h, vec3f(0.0, -0.03, 0.12), vec3f(0.0, -0.035, 0.205), 0.052), 0.035);
    // long pointed ears, out and back
    skin = min(skin, sdLimb(ha, vec3f(0.115, 0.04, -0.02), vec3f(0.31, 0.11, -0.1), 0.042, 0.005));
    // brow ridges
    skin = smin(skin, sdCapsule(ha, vec3f(0.035, 0.065, 0.125), vec3f(0.085, 0.07, 0.105), 0.022), 0.02);
    r = hU(r, skin, 1.0);
    // one horn from the brow, two tusks from the jaw
    let horn = sdLimb(h, vec3f(0.0, 0.135, 0.07), vec3f(0.0, 0.27, 0.035), 0.032, 0.004);
    let tusk = sdLimb(ha, vec3f(0.04, -0.075, 0.165), vec3f(0.06, -0.012, 0.2), 0.013, 0.003);
    r = hU(r, min(horn, tusk), 3.0);
  }
  if (length(p - foe.j[4].xyz) - 0.36 < r.x) {
    r = hU(r, sdLimb(p, foe.j[3].xyz, foe.j[4].xyz, 0.068, 0.058), 1.0);
    r = hU(r, sdLimb(p, foe.j[4].xyz, foe.j[5].xyz, 0.058, 0.05), 1.0);
    r = hU(r, length(p - foe.j[5].xyz) - 0.062, 1.0);
  }
  if (length(p - foe.j[7].xyz) - 0.36 < r.x) {
    r = hU(r, sdLimb(p, foe.j[6].xyz, foe.j[7].xyz, 0.068, 0.058), 1.0);
    r = hU(r, sdLimb(p, foe.j[7].xyz, foe.j[8].xyz, 0.058, 0.05), 1.0);
    r = hU(r, length(p - foe.j[8].xyz) - 0.062, 1.0);
  }
  if (length(p - foe.j[10].xyz) - 0.66 < r.x) {
    r = hU(r, sdLimb(p, foe.j[9].xyz, foe.j[10].xyz, 0.088, 0.072), 1.0);
    r = hU(r, sdLimb(p, foe.j[10].xyz, foe.j[11].xyz, 0.07, 0.06), 1.0);
    r = hU(r, sdLimb(p, foe.j[11].xyz, foe.j[12].xyz, 0.064, 0.05), 1.0);
  }
  if (length(p - foe.j[14].xyz) - 0.66 < r.x) {
    r = hU(r, sdLimb(p, foe.j[13].xyz, foe.j[14].xyz, 0.088, 0.072), 1.0);
    r = hU(r, sdLimb(p, foe.j[14].xyz, foe.j[15].xyz, 0.07, 0.06), 1.0);
    r = hU(r, sdLimb(p, foe.j[15].xyz, foe.j[16].xyz, 0.064, 0.05), 1.0);
  }
  r = gearWeapon(p, foe.gear[0], foe.gear[1], foe.gear[2], r);
  r = gearBow(p, foe.gear[6], foe.gear[7], foe.gear[8], foe.j[8].xyz, r);
  return r;
}

// The camera (camera-relative 0) in the foe's unit space, and the ray from it: p = o + rd·t there is
// rd·(t·scale) in the world.
fn foeOrigin() -> vec3f { return foe.root.xyz * (1.0 - 1.0 / foe.misc.x); }

// The hit along camera ray rd: the world distance (negative: none) and the material.
fn foeTrace(rd: vec3f) -> vec2f {
  let ro = foeOrigin();
  let c = foe.bound.xyz - ro;
  let b = dot(c, rd);
  var h = b * b - dot(c, c) + foe.bound.w * foe.bound.w;
  if (h < 0.0) { return vec2f(-1.0, 0.0); }
  h = sqrt(h);
  let bx = foeBox(ro, rd);
  var t = max(max(b - h, 0.0), bx.x);
  let t1 = min(b + h, bx.y);
  if (t1 < t) { return vec2f(-1.0, 0.0); }
  for (var i = 0; i < FOE_STEPS; i++) {
    let d = foeMap(ro + rd * t);
    if (d.x < max(0.0015, 0.4 * t * u.footprint)) { return vec2f(t * foe.misc.x, d.y); }
    t += d.x * 0.9;
    if (t > t1) { break; }
  }
  return vec2f(-1.0, 0.0);
}

fn foeNormal(pu: vec3f, tU: f32) -> vec3f {
  let e = max(0.001, 0.3 * tU * u.footprint);
  var n = vec3f(0.0);
  for (var i = 0; i < 4; i++) {
    let k = 2.0 * vec3f(f32(((i + 3) >> 1u) & 1), f32((i >> 1u) & 1), f32(i & 1)) - 1.0;
    n += k * foeMap(pu + k * e).x;
  }
  return normalize(n);
}

// Albedo and specular strength of material id at pu (unit space).
fn foeAlbedo(id: i32, pu: vec3f) -> vec4f {
  if (id == 1) {
    let h = (pu - foe.j[2].xyz) * foe.head;
    // yellow eyes under the brow, a dark mouth line across the jaw
    let e = vec2f(abs(h.x) - 0.06, h.y - 0.035);
    if (h.z > 0.08 && dot(e, e) < 0.022 * 0.022) {
      return vec4f(select(FC_EYE, vec3f(0.01), dot(e, e) < 0.009 * 0.009), 0.6);
    }
    if (h.z > 0.11 && abs(h.y + 0.085 - 0.02 * h.x * h.x * 40.0) < 0.008 && abs(h.x) < 0.07) { return vec4f(FC_MOUTH, 0.0); }
    // the belly paler
    let q = (pu - foe.j[0].xyz) * foe.torso;
    let belly = smoothstep(0.05, 0.13, q.z) * (1.0 - smoothstep(0.24, 0.32, q.y));
    return vec4f(mix(foe.skin.rgb, foe.skin.rgb * 0.55 + vec3f(0.17, 0.13, 0.09), belly * 0.7), 0.15);
  }
  if (id == 2) { return vec4f(FC_CLOTH, 0.0); }
  if (id == 3) { return vec4f(FC_HORN, 0.35); }
  if (id == 4) { return vec4f(HC_LEATHER, 0.0); }
  if (id == 5) { return vec4f(HC_CREAM, 0.0); }
  if (id == 11) { return vec4f(HC_WOOD, 0.0); }
  return gearAlbedo(id);
}

fn foeShade(rd: vec3f, tW: f32, pu: vec3f, id: f32, n: vec3f, sh: f32) -> vec3f {
  let pr = rd * tW;
  var alb = foeAlbedo(i32(id + 0.5), pu);
  alb = vec4f(mix(alb.rgb, vec3f(0.85), foe.misc.z), alb.a);
  let L = keyDir();
  let kc = keyCol(u.camPos.y + pr.y);
  let ndl = dot(n, L);
  var occ = 0.0;
  for (var i = 1; i <= 3; i++) {
    let hp = 0.04 * f32(i);
    occ += (hp - foeMap(pu + n * hp).x) / f32(i);
  }
  let ao = clamp(1.0 - 3.0 * occ, 0.25, 1.0) * (0.55 + 0.45 * smoothstep(0.0, 0.7, pr.y - foe.misc.y));
  let lit = smoothstep(0.0, 0.3, ndl) * 0.7 + max(ndl, 0.0) * 0.3;
  var col = alb.rgb * (kc * 0.85 * lit * sh
    + skyAmb(n, 0.38 + 0.3 * n.y) * ao
    + bounceCol(vec3f(0.10, 0.085, 0.05)) * max(-n.y, 0.0) * ao);
  let rim = 1.0 - max(dot(n, -rd), 0.0);
  col += skyHorCol() * 0.16 * rim * rim * rim * ao;
  if (alb.a > 0.0) {
    var sp = max(dot(n, normalize(L - rd)), 0.0);
    sp *= sp; sp *= sp; sp *= sp; sp *= sp;
    col += kc * (0.35 * alb.a * sp * sh);
  }
  col += vec3f(1.0, 0.9, 0.8) * foe.misc.z * 0.6;
  return surfaceFog(col, tW, u.camPos.y + pr.y, rd);
}

@fragment
fn fsFoe(@builtin(position) fc: vec4f) -> HeroOut {
  if (u.camPos.y < u.camWater) { discard; }
  let rd = cameraRay(fc.xy);
  let hit = foeTrace(rd);
  let tW = hit.x;
  if (tW < 0.0) { discard; }
  let pr = rd * tW;
  let tU = tW / foe.misc.x;
  let pu = foeOrigin() + rd * tU;
  let n = foeNormal(pu, tU);
  var sh = 0.0;
  if (dot(n, keyDir()) > 0.0) {
    sh = sunVisibility(u.camPos + pr, vec3f(0.0));
    if (sh > 0.0 && hero.bound.w > 0.0) { sh = min(sh, heroShadow(pr + n * 0.03)); }
  }
  var o: HeroOut;
  o.color = display(foeShade(rd, tW, pu, hit.y, n, sh));
  o.depth = CAMERA_NEAR / dot(pr, u.basis[2]);
  return o;
}

// ---- parts: lit boxes ---------------------------------------------------------------------------
struct CbPart {
  c: vec4f, // centre (camera-relative), how much it glows
  x: vec4f, // half-extent along its first axis; red
  y: vec4f, // … second; green
  z: vec4f, // … third; blue
};
@group(1) @binding(51) var<storage, read> cbParts: array<CbPart>;

// A box's 36 corners: face f (0–2 the + faces of x, y, z; 3–5 the − faces), two triangles a face.
const CB_QUAD = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));

struct PartOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) col: vec4f,
};

@vertex
fn vsPart(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PartOut {
  let P = cbParts[ii];
  let f = vi / 6u;
  let q = CB_QUAD[vi % 6u];
  let s = select(-1.0, 1.0, f < 3u);
  let a = f % 3u;
  var l = vec3f(q.x, q.y, s);
  if (a == 0u) { l = vec3f(s, q.x, q.y); } else if (a == 1u) { l = vec3f(q.y, s, q.x); }
  var nl = vec3f(0.0, 0.0, s);
  if (a == 0u) { nl = vec3f(s, 0.0, 0.0); } else if (a == 1u) { nl = vec3f(0.0, s, 0.0); }
  var o: PartOut;
  o.rel = P.c.xyz + P.x.xyz * l.x + P.y.xyz * l.y + P.z.xyz * l.z;
  o.pos = project(o.rel);
  o.n = normalize(P.x.xyz * nl.x / max(dot(P.x.xyz, P.x.xyz), 1e-8) + P.y.xyz * nl.y / max(dot(P.y.xyz, P.y.xyz), 1e-8) + P.z.xyz * nl.z / max(dot(P.z.xyz, P.z.xyz), 1e-8));
  o.col = vec4f(P.x.w, P.y.w, P.z.w, P.c.w);
  return o;
}

@fragment
fn fsPart(i: PartOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  var n = normalize(i.n);
  if (dot(n, rd) > 0.0) { n = -n; }
  let p = u.camPos + i.rel;
  let L = keyDir();
  let sun = select(0.0, sunVisibility(p, n * 0.05), dot(n, L) > 0.0);
  var col = i.col.rgb * (keyCol(p.y) * max(dot(n, L), 0.0) * 0.9 * sun + skyAmb(n, 0.45 + 0.3 * n.y));
  col += i.col.rgb * i.col.a * 3.0;
  return display(surfaceFog(col, dist, p.y, rd));
}

// ---- sprites: blended quads ---------------------------------------------------------------------
struct CbSprite {
  a: vec4f,   // first point (camera-relative); kind
  b: vec4f,   // second point (a streak's end, a trail's tip); size (m)
  c: vec4f,   // third (a trail's next base); fade at the first edge (trails), spin (puffs)
  d: vec4f,   // fourth (a trail's next tip); fade at the second edge (trails), age 0..1
  col: vec4f, // linear rgb; alpha (0: the sprite adds its light)
};
@group(1) @binding(52) var<storage, read> cbSprites: array<CbSprite>;

struct SpriteOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) uv: vec2f,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vsSprite(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SpriteOut {
  let S = cbSprites[ii];
  let kind = u32(S.a.w + 0.5);
  let q = CB_QUAD[vi];
  var rel: vec3f;
  var uv = q;
  if (kind == ${SPRITE.trail}u) {
    // a slash's trail: base and tip now (a, b), a moment ago (c, d)
    let e0 = select(S.a.xyz, S.b.xyz, q.x > 0.0);
    let e1 = select(S.c.xyz, S.d.xyz, q.x > 0.0);
    rel = select(e0, e1, q.y > 0.0);
    uv = vec2f(q.x * 0.5 + 0.5, q.y * 0.5 + 0.5);
  } else if (kind == ${SPRITE.streak}u) {
    // a streak from a to b, its width turned to the camera
    let ab = S.b.xyz - S.a.xyz;
    let mid = 0.5 * (S.a.xyz + S.b.xyz);
    let side = normalize(cross(ab, mid) + vec3f(0.0, 1e-6, 0.0)) * S.b.w;
    rel = mix(S.a.xyz, S.b.xyz, q.y * 0.5 + 0.5) + side * q.x - normalize(mid) * 0.02;
  } else {
    // a camera-facing quad round a (puffs turned their own way)
    let ang = S.c.w;
    let qr = vec2f(q.x * cos(ang) - q.y * sin(ang), q.x * sin(ang) + q.y * cos(ang));
    let up = select(u.basis[1], vec3f(0.0, 1.0, 0.0), kind == ${SPRITE.flame}u);
    rel = S.a.xyz + (u.basis[0] * qr.x + up * qr.y) * S.b.w;
  }
  var o: SpriteOut;
  o.rel = rel;
  o.pos = project(rel);
  o.uv = uv;
  o.id = ii;
  return o;
}

// Light added, as the display shows it over black, and what the haze lets through of it.
fn cbAdded(col: vec3f, rel: vec3f) -> vec3f {
  let dist = max(length(rel), 1e-3);
  let rd = rel / dist;
  let p = u.camPos + rel;
  let seen = surfaceFog(col, dist, p.y, rd) - surfaceFog(vec3f(0.0), dist, p.y, rd);
  return max(display(seen).rgb - display(vec3f(0.0)).rgb, vec3f(0.0));
}

@fragment
fn fsSprite(i: SpriteOut) -> @location(0) vec4f {
  let S = cbSprites[i.id];
  let kind = u32(S.a.w + 0.5);
  let q = i.uv;
  let r = length(q);
  var m = 0.0;
  if (kind == ${SPRITE.glow}u) {
    m = exp(-4.5 * r * r) * (1.0 - smoothstep(0.85, 1.0, r));
  } else if (kind == ${SPRITE.puff}u) {
    let n = 0.5 + 0.5 * vnoise(q.x * 2.3 + S.c.w * 3.0, q.y * 2.3 + S.d.w * 2.0, 4411);
    m = (1.0 - smoothstep(0.25, 1.0, r + 0.35 * (n - 0.5)));
  } else if (kind == ${SPRITE.streak}u) {
    m = (1.0 - smoothstep(0.0, 1.0, abs(q.x))) * smoothstep(-1.0, -0.6, q.y) * (0.4 + 0.6 * smoothstep(-1.0, 1.0, q.y));
  } else if (kind == ${SPRITE.trail}u) {
    // bright at the blade's tip edge, fading toward the base and along the way back
    m = mix(S.c.w, S.d.w, q.y) * smoothstep(0.0, 0.9, q.x) * (1.0 - smoothstep(0.92, 1.0, q.x));
  } else if (kind == ${SPRITE.ring}u) {
    m = exp(-60.0 * (r - 0.8) * (r - 0.8)) * (1.0 - smoothstep(0.95, 1.0, r));
  } else if (kind == ${SPRITE.flame}u) {
    // a tongue of flame: wide at the root, licking up to a point, flickering
    let y = q.y * 0.5 + 0.5;
    let w = (1.0 - y) * (0.75 + 0.25 * sin(u.time * 13.0 + S.c.w * 7.0 + y * 6.0));
    m = (1.0 - smoothstep(w * 0.6, w, abs(q.x + 0.12 * sin(u.time * 9.0 + y * 5.0 + S.c.w) * y))) * smoothstep(0.0, 0.15, y);
  } else {
    // a four-pointed glint
    let a = abs(q);
    m = max(exp(-14.0 * a.x) * (1.0 - a.y), exp(-14.0 * a.y) * (1.0 - a.x)) + exp(-9.0 * r * r) * 0.6;
    m *= 1.0 - smoothstep(0.9, 1.0, max(a.x, a.y));
  }
  if (S.col.a <= 0.0) {
    return vec4f(cbAdded(S.col.rgb * m, i.rel), 0.0);
  }
  let alpha = clamp(S.col.a * m, 0.0, 1.0);
  let dist = length(i.rel);
  let p = u.camPos + i.rel;
  let L = keyDir();
  // (smoke lit from above and the sun's side, the rest as it is)
  var col = S.col.rgb;
  if (kind == ${SPRITE.puff}u) {
    let lit = 0.6 + 0.4 * clamp(dot(normalize(vec3f(q.x, q.y, 0.6)), normalize(vec3f(0.0, 0.7, 0.3) + L * 0.5)), 0.0, 1.0);
    col = S.col.rgb * (keyCol(p.y) * 0.5 * lit + skyAmb(vec3f(0.0, 1.0, 0.0), 0.9));
  }
  let shown = display(surfaceFog(col, dist, p.y, i.rel / dist));
  return vec4f(shown.rgb * alpha, alpha);
}
`;
}
