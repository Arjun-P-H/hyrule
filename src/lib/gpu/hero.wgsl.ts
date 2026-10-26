/**
 * The third-person hero in WGSL — a port of the `#ifdef HERO` block of
 * demoShader.ts (signed-distance body from capsules around the rig joints,
 * sphere-traced inside its bounding sphere, soft shadow, contact darkening,
 * stylized shading). Uniforms are the rig's camera-relative packing
 * (rig.ts packHeroUniforms), repacked to WGSL layout by packHeroWgsl(),
 * which adds the box round every part: the camera's rays are traced only
 * where they cross it, and the renderer scissors the hero's pass to its
 * outline on screen. (The soft shadow keeps the sphere: rays passing near the
 * hero outside the box still darken its penumbra.)
 */

import { HERO_SHAPE } from '../constants';
import { GEAR, HERO_JOINTS, HERO_UNIFORM, J, jointPad, packedGearPoints } from '../hero/rig';
import { MODEL } from '../combat/weapons';
import { wgslF32 as f } from './worldgen.wgsl';

/** Bodies whose shadows the world's shading casts besides the player's own hero: other players, then foes. */
export const OTHERS = 8;
/**
 * WGSL Hero block: 18 joints as vec4, torso and head mat3x3f, bound, misc, the box's corners, the dress,
 * the gear (rig.ts GEAR), and the other bodies (three points each) for their shadows.
 */
export const HERO_BYTES = 18 * 16 + 48 + 48 + 16 + 16 + 32 + 16 + 9 * 16 + OTHERS * 3 * 16;
/** Where the box's min and max corners sit in the packed block (floats). */
export const HERO_BOX = 104;
/** Where the dress (characters.ts: the tunic's linear rgb) sits in the packed block (floats). */
const HERO_DRESS = 112;
/** Where the gear (rig.ts GEAR) sits in the packed block (floats). */
export const HERO_GEAR = 116;
/** Where the other bodies start (floats). */
export const HERO_OTHERS = HERO_GEAR + GEAR.length;

/** Repack rig.ts HERO_UNIFORM floats into the WGSL Hero block. */
export function packHeroWgsl(src: Float32Array, out: Float32Array): void {
  for (let j = 0; j < 18; j++) {
    out[4 * j] = src[3 * j];
    out[4 * j + 1] = src[3 * j + 1];
    out[4 * j + 2] = src[3 * j + 2];
    out[4 * j + 3] = 0;
  }
  for (let c = 0; c < 3; c++) {
    for (let k = 0; k < 3; k++) {
      out[72 + 4 * c + k] = src[HERO_UNIFORM.torso + 3 * c + k];
      out[84 + 4 * c + k] = src[HERO_UNIFORM.head + 3 * c + k];
    }
    out[72 + 4 * c + 3] = 0;
    out[84 + 4 * c + 3] = 0;
  }
  for (let k = 0; k < 4; k++) {
    out[96 + k] = src[HERO_UNIFORM.bound + k];
    out[100 + k] = src[HERO_UNIFORM.misc + k];
  }
  // The box round every part: each joint padded by its parts' reach (a limb's capsule lies in the
  // hull of its joints' padded spheres; the box holds them all), the glider's only while it is open,
  // and the gear's points padded likewise (a blade's or a bow's capsules lie in their hull).
  const glide = src[HERO_UNIFORM.misc];
  const nj = glide > 0 ? HERO_JOINTS : HERO_JOINTS - 1;
  for (let c = 0; c < 3; c++) { out[HERO_BOX + c] = Infinity; out[HERO_BOX + 4 + c] = -Infinity; }
  for (let j = 0; j < nj; j++) {
    const pad = jointPad(j, glide);
    for (let c = 0; c < 3; c++) {
      const x = src[3 * j + c];
      if (x - pad < out[HERO_BOX + c]) out[HERO_BOX + c] = x - pad;
      if (x + pad > out[HERO_BOX + 4 + c]) out[HERO_BOX + 4 + c] = x + pad;
    }
  }
  const gp = packedGearPoints(src);
  for (let k = 0; k < gp.n; k++) {
    const pad = gp.pts[4 * k + 3];
    for (let c = 0; c < 3; c++) {
      const x = gp.pts[4 * k + c];
      if (x - pad < out[HERO_BOX + c]) out[HERO_BOX + c] = x - pad;
      if (x + pad > out[HERO_BOX + 4 + c]) out[HERO_BOX + 4 + c] = x + pad;
    }
  }
  out[HERO_BOX + 3] = 0;
  out[HERO_BOX + 7] = 0;
  for (let k = 0; k < 3; k++) out[HERO_DRESS + k] = src[HERO_UNIFORM.dress + k];
  out[HERO_DRESS + 3] = 0;
  for (let k = 0; k < GEAR.length; k++) out[HERO_GEAR + k] = src[HERO_UNIFORM.gear + k];
}

/**
 * The other bodies into the player's own block: other players' heroes (rig.ts HERO_UNIFORM packings,
 * camera-relative), then foes' bodies (`bodies`: nine floats each, the same three points, camera-
 * relative) — for each, its pelvis (w 1), head and the point between its ankles: two capsules the
 * world's shading darkens the ground with. The list ends at the first w of 0.
 */
export function packHeroOthers(others: readonly Float32Array[] | null | undefined, out: Float32Array, bodies?: Float32Array | null, nBodies = 0): void {
  let n = 0;
  for (const src of others ?? []) {
    if (n === OTHERS) break;
    if (!(src[HERO_UNIFORM.bound + 3] > 0)) continue;
    const o = HERO_OTHERS + 12 * n++;
    for (let c = 0; c < 3; c++) {
      out[o + c] = src[3 * J.pelvis + c];
      out[o + 4 + c] = src[3 * J.head + c];
      out[o + 8 + c] = 0.5 * (src[3 * J.ankleL + c] + src[3 * J.ankleR + c]);
    }
    out[o + 3] = 1;
    out[o + 7] = 0;
    out[o + 11] = 0;
  }
  for (let k = 0; k < nBodies && bodies && n < OTHERS; k++) {
    const o = HERO_OTHERS + 12 * n++;
    for (let c = 0; c < 3; c++) {
      out[o + c] = bodies[9 * k + c];
      out[o + 4 + c] = bodies[9 * k + 3 + c];
      out[o + 8 + c] = bodies[9 * k + 6 + c];
    }
    out[o + 3] = 1;
    out[o + 7] = 0;
    out[o + 11] = 0;
  }
  if (n < OTHERS) out[HERO_OTHERS + 12 * n + 3] = 0;
}

/** Needs FRAME_WGSL (u), the light API (keyDir, keyCol, skyAmb, bounceCol, skyHorCol: sky.wgsl.ts) and surfaceFog. */
export const HERO_WGSL = /* wgsl */ `
struct Hero {
  j: array<vec4f, 18>, // joints (rig.ts J.*), camera-relative
  torso: mat3x3f,      // torso frame, columns right / up / forward
  head: mat3x3f,
  bound: vec4f,        // bounding sphere: centre (camera-relative), radius; radius 0 = no hero
  misc: vec4f,         // x glider open 0..1, y ground under the hero (camera-relative), zw sin/cos facing
  lo: vec4f,           // box round every part (camera-relative): min corner
  hi: vec4f,           //   and max corner
  dress: vec4f,        // rgb the tunic's albedo (characters.ts); the cap is a shade darker
  gear: array<vec4f, 9>, // rig.ts GEAR: the weapon (grip, axis, flat), the shield (centre, right, up), the bow (grip, up, aim)
  others: array<vec4f, ${OTHERS * 3}>, // other bodies: pelvis (w 1; 0 ends the list), head, between the ankles
};
@group(0) @binding(5) var<uniform> hero: Hero;

// Where a ray from o along d crosses the hero's box: t in [x, y] (x > y: it misses).
fn heroBox(o: vec3f, d: vec3f) -> vec2f {
  let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
  let a = (hero.lo.xyz - o) * inv;
  let b = (hero.hi.xyz - o) * inv;
  let n = min(a, b);
  let x = max(a, b);
  return vec2f(max(max(n.x, n.y), n.z), min(min(x.x, x.y), x.z));
}

const HERO_HEAD_REACH: f32 = ${f(HERO_SHAPE.headReach)};
const HERO_GLIDER_REACH: f32 = ${f(HERO_SHAPE.gliderReach)};
const HERO_STEPS: i32 = 64;
const HC_SKIN: vec3f = vec3f(0.38, 0.23, 0.15);
const HC_HAIR: vec3f = vec3f(0.40, 0.25, 0.055);
const HC_LEATHER: vec3f = vec3f(0.095, 0.045, 0.018);
const HC_CREAM: vec3f = vec3f(0.33, 0.30, 0.23);
const HC_GOLD: vec3f = vec3f(0.45, 0.30, 0.05);
const HC_SILVER: vec3f = vec3f(0.34, 0.35, 0.37);
const HC_SHIELD: vec3f = vec3f(0.014, 0.038, 0.17);
const HC_RED: vec3f = vec3f(0.30, 0.022, 0.015);
const HC_HILT: vec3f = vec3f(0.045, 0.032, 0.19);
const HC_SHEATH: vec3f = vec3f(0.018, 0.025, 0.07);
const HC_GLIDER: vec3f = vec3f(0.28, 0.05, 0.02);
const HC_GLIDER2: vec3f = vec3f(0.40, 0.31, 0.19);
const HC_WOOD: vec3f = vec3f(0.14, 0.075, 0.035);
const HC_EYE: vec3f = vec3f(0.015, 0.05, 0.075);
const HC_STEEL: vec3f = vec3f(0.5, 0.53, 0.58);
const HC_CLUB: vec3f = vec3f(0.17, 0.095, 0.045);
const HC_BONE: vec3f = vec3f(0.5, 0.46, 0.37);
const HC_BOW: vec3f = vec3f(0.11, 0.05, 0.022);
const HC_FLETCH: vec3f = vec3f(0.36, 0.05, 0.03);
const HC_GLOW: vec3f = vec3f(0.55, 0.85, 1.6);
const HC_PLANK: vec3f = vec3f(0.20, 0.10, 0.04);
const HC_PLANK_D: vec3f = vec3f(0.07, 0.032, 0.012);
const HC_IRON: vec3f = vec3f(0.12, 0.125, 0.14);
const HC_TRAV: vec3f = vec3f(0.15, 0.072, 0.03);
const HC_LID: vec3f = vec3f(0.22, 0.235, 0.26);
const HC_HIDE: vec3f = vec3f(0.30, 0.25, 0.16);
const GEAR_SWORD_LEN: f32 = ${f(HERO_SHAPE.swordLen)};
const GEAR_CLUB_LEN: f32 = ${f(HERO_SHAPE.clubLen)};
const GEAR_BOW_HALF: f32 = ${f(HERO_SHAPE.bowHalf)};
const GEAR_ARROW_LEN: f32 = ${f(HERO_SHAPE.arrowLen)};
const MODEL_SWORD: f32 = ${f(MODEL.sword)};
const MODEL_CLUB: f32 = ${f(MODEL.club)};
const MODEL_SPIKED: f32 = ${f(MODEL.spiked)};

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
// Tapered capsule: radius r1 at a → r2 at b.
fn sdLimb(p: vec3f, a: vec3f, b: vec3f, r1: f32, r2: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - mix(r1, r2, h);
}
fn sdBox3(p: vec3f, b: vec3f) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
// Capped cone along y: half-height h, radius r1 at −h and r2 at +h.
fn sdCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(r2, h);
  let k2 = vec2f(r2 - r1, 2.0 * h);
  let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
  let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
// Equilateral triangle, apex up, half-side r (shield emblem).
fn sdTri(p0: vec2f, r: f32) -> f32 {
  let k = 1.7320508;
  var p = vec2f(abs(p0.x) - r, p0.y + r / k);
  if (p.x + k * p.y > 0.0) { p = vec2f(p.x - k * p.y, -k * p.x - p.y) * 0.5; }
  p.x -= clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}
fn hU(a: vec2f, d: f32, id: f32) -> vec2f { return select(a, vec2f(d, id), d < a.x); }
// A shield's outline round its centre, by its model m (combat/weapons.ts SHIELD_MODEL): the Hylian
// Shield's heater, an ellipse cut flat at the top (0.33 m wide), rounding to its point 0.33 m below
// the centre; the Boko shield's, narrower and longer; discs for the wooden shield (0.27 m), the
// Traveler's (0.28) and the pot lid (0.22).
fn heroShieldEdge(s: vec2f, m: f32) -> f32 {
  if (m > 1.5 && m < 4.5) { return length(s) - select(select(0.27, 0.28, m > 2.5), 0.22, m > 3.5); }
  let k = select(vec2f(0.19, 0.33), vec2f(0.17, 0.36), m > 4.5);
  return max((length(s / k) - 1.0) * k.x, s.y - 0.16);
}

// The gear (rig.ts GEAR) — shared by the hero and the foes (foe.wgsl.ts), each passing its own
// frames: a weapon (g0 grip, model; g1 grip → tip, glow; g2 the blade's flat, sheathed), the shield
// (g3 centre, carried; g4 / g5 its face's right and up), the bow (g6 grip, in hand; g7 along the
// limbs, draw; g8 toward the aim, an arrow on; nock: the string hand). Each is skipped where it
// cannot be nearer than r already is. Material ids: 6 gold, 4 leather, 7 shield, 8 hilt, 16 steel,
// 17 club wood, 18 bone, 19 bow, 5 cream (string), 11 wood (shaft), 22 fletching.
fn gearWeapon(p: vec3f, g0: vec4f, g1: vec4f, g2: vec4f, r0: vec2f) -> vec2f {
  var r = r0;
  let m = g0.w;
  let len = select(GEAR_CLUB_LEN, GEAR_SWORD_LEN, m < MODEL_SWORD + 0.5);
  if (m < 0.5 || sdCapsule(p, g0.xyz - g1.xyz * 0.13, g0.xyz + g1.xyz * len, 0.09) > r.x) { return r; }
  // weapon-local: x across the edges, y through the flat, z from the grip to the tip
  let v = p - g0.xyz;
  let w = cross(g1.xyz, g2.xyz);
  let l = vec3f(dot(v, w), dot(v, g2.xyz), dot(v, g1.xyz));
  if (m < MODEL_SWORD + 0.5) {
    // the Master Sword: blue grip and wings, gold pommel and gem, a long blade (hidden in its sheath)
    r = hU(r, sdCapsule(l, vec3f(0.0, 0.0, -0.09), vec3f(0.0, 0.0, 0.07), 0.016), 8.0);
    r = hU(r, min(length(l - vec3f(0.0, 0.0, -0.112)) - 0.023, length(l - vec3f(0.0, 0.0, 0.086)) - 0.018), 6.0);
    let wing = vec3f(abs(l.x), l.y, l.z);
    r = hU(r, sdLimb(wing, vec3f(0.0, 0.0, 0.085), vec3f(0.092, 0.0, 0.112), 0.019, 0.009), 8.0);
    if (g2.w < 0.5) {
      // flat and double-edged, narrowing to its point over the last fifth
      let taper = clamp((l.z - 0.76) / 0.19, 0.0, 1.0);
      let hw = 0.024 * (1.0 - 0.94 * taper);
      let b = sdBox3(vec3f(l.x, l.y, l.z - 0.525), vec3f(hw, 0.0035 * (1.0 - 0.5 * taper), 0.425)) - 0.0015;
      r = hU(r, b, 16.0);
    }
    return r;
  }
  // a Boko club: a wrapped grip, a knotted wooden head (bone on the dragonbone one), spikes on the
  // spiked and the dragonbone ones
  r = hU(r, sdCapsule(l, vec3f(0.0, 0.0, -0.08), vec3f(0.0, 0.0, 0.13), 0.021), 4.0);
  let head = sdLimb(l, vec3f(0.0, 0.0, 0.12), vec3f(0.0, 0.0, 0.66), 0.03, 0.062) + 0.004 * sin(l.z * 41.0 + l.x * 37.0) * sin(l.y * 53.0);
  r = hU(r, head, select(17.0, 18.0, m > MODEL_SPIKED + 0.5));
  if (m > MODEL_CLUB + 0.5) {
    // spikes: four round the head, two rings
    let a = vec3f(abs(l.x), abs(l.y), l.z);
    let s1 = sdLimb(a, vec3f(0.035, 0.035, 0.5), vec3f(0.075, 0.075, 0.53), 0.013, 0.002);
    let s2 = sdLimb(a, vec3f(0.045, 0.0, 0.36), vec3f(0.098, 0.0, 0.38), 0.012, 0.002);
    let s3 = sdLimb(a, vec3f(0.0, 0.047, 0.6), vec3f(0.0, 0.1, 0.63), 0.012, 0.002);
    r = hU(r, min(s1, min(s2, s3)), select(18.0, 4.0, m > MODEL_SPIKED + 0.5));
  }
  return r;
}

fn gearShield(p: vec3f, g3: vec4f, g4: vec4f, g5: vec4f, r0: vec2f) -> vec2f {
  if (g3.w < 0.5 || length(p - g3.xyz) - 0.37 > r0.x) { return r0; }
  let v = p - g3.xyz;
  let sw = vec2f(heroShieldEdge(vec2f(dot(v, g4.xyz), dot(v, g5.xyz)), g3.w), abs(dot(v, cross(g4.xyz, g5.xyz))) - 0.015);
  return hU(r0, min(max(sw.x, sw.y), 0.0) + length(max(sw, vec2f(0.0))), 7.0);
}

fn gearBow(p: vec3f, g6: vec4f, g7: vec4f, g8: vec4f, nock: vec3f, r0: vec2f) -> vec2f {
  var r = r0;
  if (g6.w < 0.5 || length(p - g6.xyz) - (GEAR_BOW_HALF + 0.12) > r.x) { return r; }
  let c = g6.xyz;
  let up = g7.xyz;
  let fw = g8.xyz;
  // limbs bending back toward the archer, more as it draws; tips at ± GEAR_BOW_HALF
  let bend = 0.07 + 0.09 * g7.w;
  let m1 = c + up * 0.32 - fw * bend * 0.45;
  let t1 = c + up * GEAR_BOW_HALF - fw * bend * 1.5;
  let m2 = c - up * 0.32 - fw * bend * 0.45;
  let t2 = c - up * GEAR_BOW_HALF - fw * bend * 1.5;
  var bow = sdCapsule(p, c - up * 0.075, c + up * 0.075, 0.019);
  bow = min(bow, min(sdLimb(p, c + up * 0.06, m1, 0.016, 0.012), sdLimb(p, m1, t1, 0.012, 0.006)));
  bow = min(bow, min(sdLimb(p, c - up * 0.06, m2, 0.016, 0.012), sdLimb(p, m2, t2, 0.012, 0.006)));
  r = hU(r, bow, 19.0);
  // the string: straight between the tips, drawn back to the hand
  let n = select(c - fw * bend * 1.5, nock, g7.w > 0.01);
  r = hU(r, min(sdCapsule(p, t1, n, 0.0035), sdCapsule(p, n, t2, 0.0035)), 5.0);
  if (g8.w > 0.5) {
    // an arrow on the string: shaft, steel head, red fletching by the nock
    let ad = normalize(c + fw * 0.05 - n);
    let tip = n + ad * GEAR_ARROW_LEN;
    r = hU(r, sdCapsule(p, n, tip - ad * 0.05, 0.0048), 11.0);
    r = hU(r, sdLimb(p, tip - ad * 0.065, tip, 0.012, 0.0015), 16.0);
    r = hU(r, sdLimb(p, n + ad * 0.025, n + ad * 0.13, 0.013, 0.0065), 22.0);
  }
  return r;
}

// Distance to the hero (x) and the material id of the nearest part (y).
fn heroMap(p: vec3f) -> vec2f {
  let pel = hero.j[0].xyz;
  let che = hero.j[1].xyz;
  let hd = hero.j[2].xyz;
  let q = (p - pel) * hero.torso;
  var body = smin(sdLimb(p, pel, che, 0.125, 0.135), sdCapsule(p, hero.j[3].xyz, hero.j[6].xyz, 0.062), 0.06);
  body = min(body, sdCone(q - vec3f(0.0, -0.1, -0.01), 0.16, 0.205, 0.132));
  var r = vec2f(body, 1.0);
  r = hU(r, sdCapsule(p, che, hd, 0.05), 0.0);
  // The belt and its buckle (in a sphere of 0.19 m round the belt line), the sword, its sheath and
  // the shield on the back (0.42 m round their middle): each group only where it could be nearest.
  if (length(q - vec3f(0.0, 0.07, 0.0)) - 0.19 < r.x) {
    let bw = vec2f(length(q.xz) - 0.142, abs(q.y - 0.07) - 0.026);
    r = hU(r, min(max(bw.x, bw.y), 0.0) + length(max(bw, vec2f(0.0))), 4.0);
    r = hU(r, sdBox3(q - vec3f(0.0, 0.07, 0.143), vec3f(0.032, 0.026, 0.01)), 6.0);
  }
  // the sheath on the back (the sword, the shield and the bow are the gear: on the back or in hand)
  if (length(q - vec3f(-0.025, 0.1, -0.175)) - 0.31 < r.x) {
    r = hU(r, sdCapsule(q, vec3f(-0.15, -0.14, -0.175), vec3f(0.1, 0.34, -0.175), 0.028), 9.0);
  }
  r = gearWeapon(p, hero.gear[0], hero.gear[1], hero.gear[2], r);
  r = gearShield(p, hero.gear[3], hero.gear[4], hero.gear[5], r);
  r = gearBow(p, hero.gear[6], hero.gear[7], hero.gear[8], hero.j[8].xyz, r);

  if (length(p - hd) - HERO_HEAD_REACH < r.x) {
    let h = (p - hd) * hero.head;
    var skin = smin(length(h) - 0.112, length(h - vec3f(0.0, -0.045, 0.03)) - 0.085, 0.04);
    skin = min(skin, length(h - vec3f(0.0, -0.02, 0.11)) - 0.018);
    let ha = vec3f(abs(h.x), h.yz);
    skin = min(skin, sdLimb(ha, vec3f(0.1, -0.01, -0.005), vec3f(0.2, 0.045, -0.05), 0.024, 0.004));
    r = hU(r, skin, 0.0);
    var hair = max(length(h - vec3f(0.0, 0.02, -0.02)) - 0.121, -sdBox3(h - vec3f(0.0, -0.075, 0.1), vec3f(0.09, 0.1, 0.09)));
    hair = min(hair, sdCapsule(h, vec3f(-0.055, 0.035, 0.1), vec3f(0.055, 0.035, 0.1), 0.026));
    hair = min(hair, sdLimb(ha, vec3f(0.085, 0.02, 0.07), vec3f(0.095, -0.075, 0.075), 0.026, 0.012));
    r = hU(r, hair, 3.0);
    let cap = smin(sdLimb(h, vec3f(0.0, 0.065, -0.02), vec3f(0.0, 0.02, -0.21), 0.124, 0.074),
                   sdLimb(h, vec3f(0.0, 0.02, -0.21), vec3f(0.0, -0.09, -0.4), 0.074, 0.016), 0.03);
    r = hU(r, max(cap, (0.42 * h.z - 0.005) - h.y), 2.0);
  }
  if (length(p - hero.j[4].xyz) - 0.33 < r.x) {
    r = hU(r, sdLimb(p, hero.j[3].xyz, hero.j[4].xyz, 0.054, 0.047), 1.0);
    r = hU(r, sdLimb(p, hero.j[4].xyz, hero.j[5].xyz, 0.047, 0.04), 12.0);
    r = hU(r, length(p - hero.j[5].xyz) - 0.046, 0.0);
  }
  if (length(p - hero.j[7].xyz) - 0.33 < r.x) {
    r = hU(r, sdLimb(p, hero.j[6].xyz, hero.j[7].xyz, 0.054, 0.047), 1.0);
    r = hU(r, sdLimb(p, hero.j[7].xyz, hero.j[8].xyz, 0.047, 0.04), 13.0);
    r = hU(r, length(p - hero.j[8].xyz) - 0.046, 0.0);
  }
  if (length(p - hero.j[10].xyz) - 0.62 < r.x) {
    r = hU(r, sdLimb(p, hero.j[9].xyz, hero.j[10].xyz, 0.078, 0.06), 5.0);
    r = hU(r, sdLimb(p, hero.j[10].xyz, hero.j[11].xyz, 0.058, 0.05), 14.0);
    r = hU(r, sdLimb(p, hero.j[11].xyz, hero.j[12].xyz, 0.056, 0.042), 4.0);
  }
  if (length(p - hero.j[14].xyz) - 0.62 < r.x) {
    r = hU(r, sdLimb(p, hero.j[13].xyz, hero.j[14].xyz, 0.078, 0.06), 5.0);
    r = hU(r, sdLimb(p, hero.j[14].xyz, hero.j[15].xyz, 0.058, 0.05), 15.0);
    r = hU(r, sdLimb(p, hero.j[15].xyz, hero.j[16].xyz, 0.056, 0.042), 4.0);
  }
  let g = hero.misc.x;
  if (g > 0.01 && length(p - hero.j[17].xyz) - HERO_GLIDER_REACH * g < r.x) {
    let c = hero.j[17].xyz;
    let right = vec3f(hero.misc.w, 0.0, -hero.misc.z);
    let v = (p - c) / g;
    let cq = vec3f(dot(v, right), v.y, v.x * hero.misc.z + v.z * hero.misc.w);
    let shell = abs(length(cq - vec3f(0.0, -1.35, 0.0)) - 1.6) - 0.014;
    let wing = (length(cq.xz / vec2f(1.05, 0.45)) - 1.0) * 0.45;
    r = hU(r, max(shell, max(wing, -cq.y - 0.3)) * g, 10.0);
    r = hU(r, min(sdCapsule(p, hero.j[5].xyz, c + g * (vec3f(0.0, -0.1, 0.0) - 0.42 * right), 0.009),
                  sdCapsule(p, hero.j[8].xyz, c + g * (vec3f(0.0, -0.1, 0.0) + 0.42 * right), 0.009)), 11.0);
  }
  return r;
}

// Sphere-trace the camera ray (origin 0, camera-relative) inside the bounds: the hit's distance
// (negative: none) and the material id there.
fn heroTrace(rd: vec3f) -> vec2f {
  let c = hero.bound.xyz;
  let b = dot(c, rd);
  var h = b * b - dot(c, c) + hero.bound.w * hero.bound.w;
  if (h < 0.0) { return vec2f(-1.0, 0.0); }
  h = sqrt(h);
  // Only where the ray is in both the sphere and the box (nothing of the hero lies outside the box).
  let bx = heroBox(vec3f(0.0), rd);
  var t = max(max(b - h, 0.0), bx.x);
  let t1 = min(b + h, bx.y);
  if (t1 < t) { return vec2f(-1.0, 0.0); }
  for (var i = 0; i < HERO_STEPS; i++) {
    let d = heroMap(rd * t);
    if (d.x < max(0.0015, 0.4 * t * u.footprint)) { return vec2f(t, d.y); }
    t += d.x * 0.9;
    if (t > t1) { break; }
  }
  return vec2f(-1.0, 0.0);
}

// Soft shadow the hero casts on a camera-relative point (itself included).
fn heroShadow(pr: vec3f) -> f32 {
  let oc = pr - hero.bound.xyz;
  let L = keyDir();
  let b = dot(oc, L);
  let c = dot(oc, oc) - hero.bound.w * hero.bound.w;
  var h = b * b - c;
  if (h < 0.0 || (b > 0.0 && c > 0.0)) { return 1.0; }
  h = sqrt(h);
  var s = max(-b - h, 0.02);
  let s1 = -b + h;
  var res = 1.0;
  for (var k = 0; k < 24; k++) {
    let d = heroMap(pr + L * s).x;
    res = min(res, 8.0 * d / s);
    if (res < 0.02) { return 0.0; }
    s += clamp(d, 0.02, 0.25);
    if (s > s1) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

// Ambient contact darkening on the ground around the hero's legs.
fn heroContactAO(pr: vec3f) -> f32 {
  // (the legs' capsule reaches 1.3 m from the pelvis at most: 3 m away the darkening is below 1e-9)
  if (dot(pr - hero.j[0].xyz, pr - hero.j[0].xyz) > 9.0) { return 1.0; }
  let d = sdCapsule(pr, hero.j[0].xyz, 0.5 * (hero.j[11].xyz + hero.j[15].xyz), 0.18);
  return 1.0 - 0.5 * exp(-18.0 * d * max(d, 0.0) - 2.0 * max(-d, 0.0));
}

// The shield's face at pr (its frame g3–g5, g3.w its model): the Hylian Shield's silver rim, Triforce
// and red bird's crest; planks and iron straps on the wooden one, a leather face in a steel rim with a
// boss on the Traveler's, dull steel on the pot lid, hide and bone on the Boko shield.
fn shieldAlbedo(pr: vec3f, g3: vec4f, g4: vec4f, g5: vec4f) -> vec4f {
  let v = pr - g3.xyz;
  let s = vec2f(dot(v, g4.xyz), dot(v, g5.xyz));
  let m = g3.w;
  let e = heroShieldEdge(s, m);
  if (m > 1.5) {
    let rr = length(s);
    if (m < 2.5) {
      if (e > -0.02 || abs(abs(s.y) - 0.12) < 0.022) { return vec4f(HC_IRON, 0.6); }
      return vec4f(select(HC_PLANK, HC_PLANK_D, abs(fract((s.x + 0.27) / 0.108) - 0.5) > 0.455), 0.0);
    }
    if (m < 3.5) {
      if (e > -0.025 || rr < 0.065) { return vec4f(HC_STEEL, 0.8); }
      if (abs(rr - 0.19) < 0.008) { return vec4f(HC_PLANK_D, 0.0); }
      return vec4f(HC_TRAV, 0.0);
    }
    if (m < 4.5) {
      if (rr < 0.04) { return vec4f(HC_STEEL, 1.0); }
      if (e > -0.02) { return vec4f(HC_STEEL, 0.9); }
      if (abs(rr - 0.135) < 0.006) { return vec4f(HC_IRON, 0.5); }
      return vec4f(HC_LID, 0.7);
    }
    if (e > -0.025 || abs(s.x) < 0.018) { return vec4f(HC_BONE, 0.3); }
    if (abs(fract(s.y * 7.5) - 0.5) > 0.42) { return vec4f(HC_PLANK_D, 0.0); }
    return vec4f(HC_HIDE, 0.0);
  }
  if (e > -0.022) { return vec4f(HC_SILVER, 0.8); }
  let tri = min(sdTri(s - vec2f(0.0, 0.08), 0.034),
                min(sdTri(s - vec2f(-0.034, 0.021), 0.034), sdTri(s - vec2f(0.034, 0.021), 0.034)));
  if (tri < 0.0) { return vec4f(HC_GOLD, 0.9); }
  if (abs(s.y + 0.1 + 0.55 * abs(s.x)) < 0.018 && abs(s.x) < 0.14) { return vec4f(HC_RED, 0.0); }
  return vec4f(HC_SHIELD, 0.0);
}

// The gear's own materials (ids 16 and up): steel, club wood, bone, the bow, the fletching.
fn gearAlbedo(id: i32) -> vec4f {
  if (id == 16) { return vec4f(HC_STEEL, 1.0); }
  if (id == 17) { return vec4f(HC_CLUB, 0.0); }
  if (id == 18) { return vec4f(HC_BONE, 0.3); }
  if (id == 19) { return vec4f(HC_BOW, 0.25); }
  return vec4f(HC_FLETCH, 0.0);
}

// The bow's wood by its model (g6.w, combat/weapons.ts BOW_MODEL): the Traveler's plain brown, the
// Boko bow's pale wood with bone tips, the Soldier's dark lacquer with a gold wrap and steel tips.
fn bowAlbedo(pr: vec3f) -> vec4f {
  let m = hero.gear[6].w;
  if (m < 1.5) { return vec4f(HC_BOW, 0.25); }
  let along = abs(dot(pr - hero.gear[6].xyz, hero.gear[7].xyz));
  let tip = along > GEAR_BOW_HALF - 0.12;
  if (m < 2.5) { return select(vec4f(0.19, 0.12, 0.065, 0.1), vec4f(HC_BONE, 0.3), tip); }
  if (tip) { return vec4f(HC_STEEL, 1.0); }
  return select(vec4f(0.04, 0.022, 0.016, 0.5), vec4f(HC_GOLD, 0.9), along < 0.09);
}

// Albedo (rgb) and specular strength (a) of material id at pr.
fn heroAlbedo(id: i32, pr: vec3f) -> vec4f {
  if (id == 0) {
    let h = (pr - hero.j[2].xyz) * hero.head;
    let e = vec2f(abs(h.x) - 0.04, (h.y + 0.004) * 0.75);
    if (h.z > 0.07 && dot(e, e) < 0.017 * 0.017) { return vec4f(HC_EYE, 0.5); }
    return vec4f(HC_SKIN, 0.0);
  }
  if (id == 1) { return vec4f(hero.dress.rgb, 0.0); }
  if (id == 2) { return vec4f(hero.dress.rgb * 0.87, 0.0); }
  if (id == 3) { return vec4f(HC_HAIR, 0.0); }
  if (id == 4) { return vec4f(HC_LEATHER, 0.0); }
  if (id == 5) { return vec4f(HC_CREAM, 0.0); }
  if (id == 6) { return vec4f(HC_GOLD, 1.0); }
  if (id == 7) { return shieldAlbedo(pr, hero.gear[3], hero.gear[4], hero.gear[5]); }
  if (id == 19) { return bowAlbedo(pr); }
  if (id >= 16) { return gearAlbedo(id); }
  if (id == 8) { return vec4f(HC_HILT, 0.0); }
  if (id == 9) { return vec4f(HC_SHEATH, 0.0); }
  if (id == 10) {
    let v = pr - hero.j[17].xyz;
    let x = (v.x * hero.misc.w - v.z * hero.misc.z) / max(hero.misc.x, 0.05);
    return vec4f(select(HC_GLIDER, HC_GLIDER2, fract(x * 2.2 + 0.5) < 0.18), 0.0);
  }
  if (id == 11) { return vec4f(HC_WOOD, 0.0); }
  // forearms 12/13 (sleeve → gauntlet), shins 14/15 (trousers → boot); constant indices only
  var a = hero.j[14].xyz;
  var b = hero.j[15].xyz;
  if (id == 12) { a = hero.j[4].xyz; b = hero.j[5].xyz; }
  else if (id == 13) { a = hero.j[7].xyz; b = hero.j[8].xyz; }
  else if (id == 14) { a = hero.j[10].xyz; b = hero.j[11].xyz; }
  let ab = b - a;
  let t = clamp(dot(pr - a, ab) / dot(ab, ab), 0.0, 1.0);
  return vec4f(select(HC_CREAM, HC_LEATHER, t > select(0.38, 0.5, id < 14)), 0.0);
}

// Tetrahedral-difference normal at a camera-relative hero point.
fn heroNormal(pr: vec3f, tH: f32) -> vec3f {
  let e = max(0.001, 0.3 * tH * u.footprint);
  var n = vec3f(0.0);
  for (var i = 0; i < 4; i++) {
    let k = 2.0 * vec3f(f32(((i + 3) >> 1u) & 1), f32((i >> 1u) & 1), f32(i & 1)) - 1.0;
    n += k * heroMap(pr + k * e).x;
  }
  return normalize(n);
}

// sh: sun visibility (terrain soft shadow × the hero's own shadow); id: the material there (heroTrace).
fn heroShade(rd: vec3f, tH: f32, id: f32, n: vec3f, sh: f32) -> vec3f {
  let pr = rd * tH;
  let alb = heroAlbedo(i32(id + 0.5), pr);
  let spec = alb.a;
  let L = keyDir();
  let kc = keyCol(u.camPos.y + pr.y);
  let ndl = dot(n, L);
  var occ = 0.0;
  for (var i = 1; i <= 3; i++) {
    let hp = 0.04 * f32(i);
    occ += (hp - heroMap(pr + n * hp).x) / f32(i);
  }
  let ao = clamp(1.0 - 3.0 * occ, 0.25, 1.0) * (0.55 + 0.45 * smoothstep(0.0, 0.8, pr.y - hero.misc.y));
  let lit = smoothstep(0.0, 0.25, ndl) * 0.7 + max(ndl, 0.0) * 0.3;
  var col = alb.rgb * (kc * 0.85 * lit * sh
    + skyAmb(n, 0.38 + 0.3 * n.y) * ao
    + bounceCol(vec3f(0.10, 0.085, 0.05)) * max(-n.y, 0.0) * ao);
  let rim = 1.0 - max(dot(n, -rd), 0.0);
  col += skyHorCol() * 0.18 * rim * rim * rim * ao;
  if (spec > 0.0) {
    var sp = max(dot(n, normalize(L - rd)), 0.0);
    sp *= sp; sp *= sp; sp *= sp; sp *= sp; sp *= sp;
    col += kc * (0.5 * spec * sp * sh);
  }
  // a charged blade glows
  if (i32(id + 0.5) == 16) { col += HC_GLOW * hero.gear[1].w; }
  return surfaceFog(col, tH, u.camPos.y + pr.y, rd);
}

// Soft shadow of capsule a–b (radius r) on p toward L, its penumbra like heroShadow's: the ray's
// closest approach to the segment, analytically.
fn capsuleShadow(p: vec3f, L: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let ba = b - a;
  let w = p - a;
  let B = dot(L, ba);
  let D = dot(L, w);
  let h = clamp((dot(ba, w) - B * D) / max(dot(ba, ba) - B * B, 1e-4), 0.0, 1.0);
  let s = B * h - D; // along the ray to the closest approach
  if (s <= 0.0) { return 1.0; }
  let d = length(w + L * s - ba * h) - r;
  return clamp(8.0 * d / s, 0.0, 1.0);
}

// The other players' shadows on a camera-relative point: torso and head, and legs, as two capsules each.
fn othersShadow(rel: vec3f) -> f32 {
  let L = keyDir();
  var res = 1.0;
  for (var k = 0; k < ${OTHERS}; k++) {
    let pel = hero.others[3 * k];
    if (pel.w <= 0.0) { break; }
    let oc = rel - pel.xyz;
    let b = dot(oc, L);
    let c = dot(oc, oc) - 2.56; // (every part lies within 1.6 m of the pelvis)
    if (b * b - c < 0.0 || (b > 0.0 && c > 0.0)) { continue; }
    res = min(res, min(capsuleShadow(rel, L, pel.xyz, hero.others[3 * k + 1].xyz, 0.2),
                       capsuleShadow(rel, L, pel.xyz, hero.others[3 * k + 2].xyz, 0.15)));
  }
  return res;
}

fn heroShadowAt(rel: vec3f) -> f32 {
  var s = 1.0;
  if (hero.bound.w > 0.0) { s = heroShadow(rel); }
  if (hero.others[0].w > 0.0) { s = min(s, othersShadow(rel)); }
  return s;
}
fn heroAOAt(rel: vec3f) -> f32 {
  if (hero.bound.w <= 0.0) { return 1.0; }
  return heroContactAO(rel);
}
`;

/** No hero: the hooks shadeSurface() calls, as constants. */
export const NO_HERO_WGSL = /* wgsl */ `
fn heroShadowAt(rel: vec3f) -> f32 { return 1.0; }
fn heroAOAt(rel: vec3f) -> f32 { return 1.0; }
`;
