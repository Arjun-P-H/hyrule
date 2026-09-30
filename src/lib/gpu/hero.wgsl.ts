/**
 * The third-person hero in WGSL — a port of the `#ifdef HERO` block of
 * demoShader.ts (signed-distance body from capsules around the rig joints,
 * sphere-traced inside its bounding sphere, soft shadow, contact darkening,
 * stylized shading). Uniforms are the rig's camera-relative packing
 * (rig.ts packHeroUniforms), repacked to WGSL layout by packHeroWgsl().
 */

import { HERO_SHAPE } from '../constants';
import { HERO_UNIFORM } from '../hero/rig';
import { wgslF32 as f } from './worldgen.wgsl';

/** WGSL Hero block: 18 joints as vec4, torso and head mat3x3f, bound, misc. */
export const HERO_BYTES = 18 * 16 + 48 + 48 + 16 + 16;

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
}

/** Needs FRAME_WGSL (u), SUN, SUN_COL, SKY_TINT, SKY_HOR and surfaceFog. */
export const HERO_WGSL = /* wgsl */ `
struct Hero {
  j: array<vec4f, 18>, // joints (rig.ts J.*), camera-relative
  torso: mat3x3f,      // torso frame, columns right / up / forward
  head: mat3x3f,
  bound: vec4f,        // bounding sphere: centre (camera-relative), radius; radius 0 = no hero
  misc: vec4f,         // x glider open 0..1, y ground under the hero (camera-relative), zw sin/cos facing
};
@group(0) @binding(5) var<uniform> hero: Hero;

const HERO_HEAD_REACH: f32 = ${f(HERO_SHAPE.headReach)};
const HERO_GLIDER_REACH: f32 = ${f(HERO_SHAPE.gliderReach)};
const HERO_STEPS: i32 = 64;
const HC_SKIN: vec3f = vec3f(0.38, 0.23, 0.15);
const HC_TUNIC: vec3f = vec3f(0.028, 0.115, 0.022);
const HC_CAP: vec3f = vec3f(0.024, 0.10, 0.019);
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
fn heroShieldEdge(s: vec2f) -> f32 {
  return max((length((s - vec2f(0.0, 0.25)) / vec2f(0.19, 0.33)) - 1.0) * 0.19, s.y - 0.41);
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
  let bw = vec2f(length(q.xz) - 0.142, abs(q.y - 0.07) - 0.026);
  r = hU(r, min(max(bw.x, bw.y), 0.0) + length(max(bw, vec2f(0.0))), 4.0);
  r = hU(r, sdBox3(q - vec3f(0.0, 0.07, 0.143), vec3f(0.032, 0.026, 0.01)), 6.0);
  r = hU(r, sdCapsule(q, vec3f(-0.15, -0.14, -0.175), vec3f(0.1, 0.34, -0.175), 0.028), 9.0);
  r = hU(r, sdCapsule(q, vec3f(0.1, 0.34, -0.175), vec3f(0.155, 0.46, -0.175), 0.018), 8.0);
  r = hU(r, min(sdCapsule(q, vec3f(0.038, 0.372, -0.175), vec3f(0.162, 0.308, -0.175), 0.016),
                length(q - vec3f(0.164, 0.478, -0.175)) - 0.024), 6.0);
  let sw = vec2f(heroShieldEdge(q.xy), abs(q.z + 0.222) - 0.015);
  r = hU(r, min(max(sw.x, sw.y), 0.0) + length(max(sw, vec2f(0.0))), 7.0);

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

// Sphere-trace the camera ray (origin 0, camera-relative) inside the bounds.
fn heroTrace(rd: vec3f) -> f32 {
  let c = hero.bound.xyz;
  let b = dot(c, rd);
  var h = b * b - dot(c, c) + hero.bound.w * hero.bound.w;
  if (h < 0.0) { return -1.0; }
  h = sqrt(h);
  var t = max(b - h, 0.0);
  let t1 = b + h;
  if (t1 < 0.0) { return -1.0; }
  for (var i = 0; i < HERO_STEPS; i++) {
    let d = heroMap(rd * t).x;
    if (d < max(0.0015, 0.4 * t * u.footprint)) { return t; }
    t += d * 0.9;
    if (t > t1) { break; }
  }
  return -1.0;
}

// Soft shadow the hero casts on a camera-relative point (itself included).
fn heroShadow(pr: vec3f) -> f32 {
  let oc = pr - hero.bound.xyz;
  let b = dot(oc, SUN);
  let c = dot(oc, oc) - hero.bound.w * hero.bound.w;
  var h = b * b - c;
  if (h < 0.0 || (b > 0.0 && c > 0.0)) { return 1.0; }
  h = sqrt(h);
  var s = max(-b - h, 0.02);
  let s1 = -b + h;
  var res = 1.0;
  for (var k = 0; k < 24; k++) {
    let d = heroMap(pr + SUN * s).x;
    res = min(res, 8.0 * d / s);
    if (res < 0.02) { return 0.0; }
    s += clamp(d, 0.02, 0.25);
    if (s > s1) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

// Ambient contact darkening on the ground around the hero's legs.
fn heroContactAO(pr: vec3f) -> f32 {
  let d = sdCapsule(pr, hero.j[0].xyz, 0.5 * (hero.j[11].xyz + hero.j[15].xyz), 0.18);
  return 1.0 - 0.5 * exp(-18.0 * d * max(d, 0.0) - 2.0 * max(-d, 0.0));
}

// Albedo (rgb) and specular strength (a) of material id at pr.
fn heroAlbedo(id: i32, pr: vec3f) -> vec4f {
  if (id == 0) {
    let h = (pr - hero.j[2].xyz) * hero.head;
    let e = vec2f(abs(h.x) - 0.04, (h.y + 0.004) * 0.75);
    if (h.z > 0.07 && dot(e, e) < 0.017 * 0.017) { return vec4f(HC_EYE, 0.5); }
    return vec4f(HC_SKIN, 0.0);
  }
  if (id == 1) { return vec4f(HC_TUNIC, 0.0); }
  if (id == 2) { return vec4f(HC_CAP, 0.0); }
  if (id == 3) { return vec4f(HC_HAIR, 0.0); }
  if (id == 4) { return vec4f(HC_LEATHER, 0.0); }
  if (id == 5) { return vec4f(HC_CREAM, 0.0); }
  if (id == 6) { return vec4f(HC_GOLD, 1.0); }
  if (id == 7) {
    let s = ((pr - hero.j[0].xyz) * hero.torso).xy;
    if (heroShieldEdge(s) > -0.022) { return vec4f(HC_SILVER, 0.8); }
    let tri = min(sdTri(s - vec2f(0.0, 0.33), 0.034),
                  min(sdTri(s - vec2f(-0.034, 0.271), 0.034), sdTri(s - vec2f(0.034, 0.271), 0.034)));
    if (tri < 0.0) { return vec4f(HC_GOLD, 0.9); }
    if (abs(s.y - 0.15 + 0.55 * abs(s.x)) < 0.018 && abs(s.x) < 0.14) { return vec4f(HC_RED, 0.0); }
    return vec4f(HC_SHIELD, 0.0);
  }
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

// sh: sun visibility (terrain soft shadow × the hero's own shadow).
fn heroShade(rd: vec3f, tH: f32, n: vec3f, sh: f32) -> vec3f {
  let pr = rd * tH;
  let alb = heroAlbedo(i32(heroMap(pr).y + 0.5), pr);
  let spec = alb.a;
  let ndl = dot(n, SUN);
  var occ = 0.0;
  for (var i = 1; i <= 3; i++) {
    let hp = 0.04 * f32(i);
    occ += (hp - heroMap(pr + n * hp).x) / f32(i);
  }
  let ao = clamp(1.0 - 3.0 * occ, 0.25, 1.0) * (0.55 + 0.45 * smoothstep(0.0, 0.8, pr.y - hero.misc.y));
  let lit = smoothstep(0.0, 0.25, ndl) * 0.7 + max(ndl, 0.0) * 0.3;
  var col = alb.rgb * (SUN_COL * 0.85 * lit * sh
    + SKY_TINT * (0.38 + 0.3 * n.y) * ao
    + vec3f(0.10, 0.085, 0.05) * max(-n.y, 0.0) * ao);
  let rim = 1.0 - max(dot(n, -rd), 0.0);
  col += SKY_HOR * 0.18 * rim * rim * rim * ao;
  if (spec > 0.0) {
    var sp = max(dot(n, normalize(SUN - rd)), 0.0);
    sp *= sp; sp *= sp; sp *= sp; sp *= sp; sp *= sp;
    col += SUN_COL * (0.5 * spec * sp * sh);
  }
  return surfaceFog(col, tH, u.camPos.y + pr.y, rd);
}

fn heroShadowAt(rel: vec3f) -> f32 {
  if (hero.bound.w <= 0.0) { return 1.0; }
  return heroShadow(rel);
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
