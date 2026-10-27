/**
 * The horses in WGSL (horse/horses.ts poses them; gpu/horses.ts draws them): the horse mesh (horse/model.ts), one
 * instance a horse, skinned in the vertex shader by its joints' matrices (already camera-relative world from the
 * CPU), lit as the rocks are (the sun by the ground's visibility of it at its place, the sky's ambient) and fogged.
 * With the trees' shadow map, the horses cast into it.
 *
 * Every horse's coat is its own (hsCoat): made here from its coat seed (the colour: bay, chestnut, black, grey,
 * palomino, buckskin, dun, cream; mostly its herd's) and its id (its shade, its pattern: solid, pinto or appaloosa;
 * the dapples of a grey, a dun's stripe, socks on each leg, a star or a blaze), laid on the horse where it stands in
 * its bind pose, so the markings stay put on it as it moves; the mane and the tail (the mesh's hair tag) its hair
 * colour. The painted texture gives only its brushwork (its light and shade over the colour); its darkest parts
 * (the eyes, the nostrils, the hooves) are kept as painted. A vanishing horse (fsHorseFade, drawn apart: discard
 * would cost every horse the GPU's hidden-surface removal) dissolves in noise along a glowing edge; a blow flashes.
 *
 * Needs, from the raster shader: u (Frame), project(), display(), surfaceFog, keyDir, keyCol, skyAmb, sunAt,
 * treeShadowAt, skyShadowAt, heroShadowAt, underwater, rlFogged, hash3; with `shadows`, forest (the shadow map).
 */

import { HORSE_JOINTS, HORSE_STRIDE } from '../horse/horses';

export function horsesWGSL(shadows: boolean): string {
  return /* wgsl */ `
const HS_STRIDE: u32 = ${HORSE_STRIDE / 4}u; // vec4s a horse
const HS_JOINTS: u32 = ${HORSE_JOINTS}u;
@group(1) @binding(70) var<storage, read> horseData: array<vec4f>;
@group(1) @binding(71) var horseTex: texture_2d<f32>;
@group(1) @binding(72) var horseSamp: sampler;

struct HsIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
  @location(3) j: vec4u,
  @location(4) w: vec4f,
  @location(5) tag: vec4f,
};

/** The skinning matrix of the vertex (its four joints, weighted), rows. */
fn hsSkin(ii: u32, j: vec4u, w: vec4f) -> array<vec4f, 3> {
  let b = ii * HS_STRIDE + 1u;
  var m = array<vec4f, 3>(vec4f(0.0), vec4f(0.0), vec4f(0.0));
  for (var k = 0u; k < 4u; k++) {
    let o = b + min(j[k], HS_JOINTS - 1u) * 3u;
    m[0] += horseData[o] * w[k];
    m[1] += horseData[o + 1u] * w[k];
    m[2] += horseData[o + 2u] * w[k];
  }
  return m;
}

fn hsPoint(m: array<vec4f, 3>, p: vec3f) -> vec3f {
  return vec3f(dot(m[0].xyz, p) + m[0].w, dot(m[1].xyz, p) + m[1].w, dot(m[2].xyz, p) + m[2].w);
}

struct HsOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(3) bp: vec3f,  // where it is on the horse in the bind pose (+y up, +z ahead, +x right)
  @location(4) bn: vec3f,
  @location(5) hair: f32,
  @location(6) @interpolate(flat) seeds: vec2u, // its id, its coat seed
  @location(7) @interpolate(flat) fx: vec2f,    // how far it has vanished, a blow's flash
};

@vertex
fn vsHorse(@builtin(instance_index) ii: u32, v: HsIn) -> HsOut {
  let m = hsSkin(ii, v.j, v.w);
  let h = horseData[ii * HS_STRIDE];
  var o: HsOut;
  o.rel = hsPoint(m, v.pos);
  o.pos = project(o.rel);
  o.n = vec3f(dot(m[0].xyz, v.nrm), dot(m[1].xyz, v.nrm), dot(m[2].xyz, v.nrm));
  o.uv = v.uv;
  o.bp = v.pos;
  o.bn = v.nrm;
  o.hair = v.tag.x;
  o.seeds = vec2u(bitcast<u32>(h.x), bitcast<u32>(h.y));
  o.fx = h.zw;
  return o;
}
${shadows ? `
@vertex
fn vsHorseShadow(@builtin(instance_index) ii: u32, v: HsIn) -> @builtin(position) vec4f {
  let q = hsPoint(hsSkin(ii, v.j, v.w), v.pos) - forest.shN.xyz;
  let Rr = forest.shC.w;
  return vec4f(dot(q, forest.shR.xyz) / Rr, dot(q, forest.shU.xyz) / Rr, 0.5 + 0.5 * dot(q, forest.shF.xyz) / forest.shR.w, 1.0);
}
` : ''}
/** A number 0..1 for seed h and channel k. */
fn hsU(h: u32, k: i32) -> f32 {
  return f32(hash3(i32(h & 0x7fffffffu), k, 9127) & 0xffffu) / 65535.0;
}

/** Smooth 3D value noise 0..1. */
fn hsNoise(p: vec3f, s: u32) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let sd = i32(s & 0xffffu);
  var c: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let q = i + vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    c[k] = f32(hash3(q.x + q.z * 7919, q.y + q.z * 104729, sd) & 0xffffu) / 65535.0;
  }
  let x0 = mix(mix(c[0], c[1], u.x), mix(c[2], c[3], u.x), u.y);
  let x1 = mix(mix(c[4], c[5], u.x), mix(c[6], c[7], u.x), u.y);
  return mix(x0, x1, u.z);
}

const HS_WHITE = vec3f(0.82, 0.8, 0.76);
const HS_BLACK = vec3f(0.018, 0.016, 0.016);

/** The coat's colour at bind-pose place bp (normal bn) of the horse with id h, coat seed c; its hair there 0..1. */
fn hsCoat(bp: vec3f, bn: vec3f, hairTag: f32, h: u32, c: u32) -> vec3f {
  // the colour (the coat seed): body, hair, how dark the lower legs (the points), a stripe down the back, dapples
  let f = hsU(c, 1);
  var body: vec3f; var hair: vec3f; var points = 0.0; var stripe = 0.0; var dapple = 0.0;
  if (f < 0.26) { body = vec3f(0.28, 0.11, 0.045); hair = HS_BLACK; points = 1.0; }                        // bay
  else if (f < 0.44) { body = vec3f(0.4, 0.14, 0.045); hair = select(body * 0.8, vec3f(0.72, 0.56, 0.32), hsU(h, 2) < 0.4); } // chestnut
  else if (f < 0.52) { body = vec3f(0.028, 0.024, 0.024); hair = HS_BLACK; }                               // black
  else if (f < 0.64) { body = vec3f(0.48, 0.48, 0.5) * (0.6 + 0.6 * hsU(c, 2)); hair = vec3f(0.62, 0.62, 0.64) * (0.4 + hsU(c, 3)); dapple = 1.0; } // grey
  else if (f < 0.72) { body = vec3f(0.7, 0.46, 0.17); hair = vec3f(0.86, 0.8, 0.66); }                     // palomino
  else if (f < 0.8) { body = vec3f(0.58, 0.41, 0.17); hair = HS_BLACK; points = 1.0; }                     // buckskin
  else if (f < 0.88) { body = vec3f(0.44, 0.35, 0.23); hair = vec3f(0.05, 0.04, 0.035); points = 0.7; stripe = 1.0; } // dun
  else { body = vec3f(0.78, 0.72, 0.6); hair = vec3f(0.9, 0.88, 0.82); }                                  // cream
  // its own shade of it (the id)
  let sh = hsU(h, 3) - 0.5;
  body *= (1.0 + 0.6 * sh) * vec3f(1.0 + 0.2 * (hsU(h, 4) - 0.5), 1.0, 1.0 - 0.2 * (hsU(h, 4) - 0.5));
  var col = body;
  let n1 = hsNoise(bp * 2.3 + vec3f(f32(h & 255u)), h);
  // the points: the lower legs (and a little of the muzzle) dark, with a soft ragged edge
  col = mix(col, HS_BLACK * 1.5, points * smoothstep(0.62, 0.38, bp.y + 0.08 * n1));
  // a dun's stripe down the middle of its back
  col = mix(col, hair, stripe * smoothstep(0.05, 0.02, abs(bp.x)) * step(1.15, bp.y) * step(-0.85, bp.z) * step(bp.z, 0.75));
  // a grey's dapples on its body
  if (dapple > 0.0) { col *= mix(0.72, 1.12, smoothstep(0.38, 0.62, hsNoise(bp * 7.5, h ^ 77u))) ; }
  // the pattern: pinto blotches (more of them low down), or an appaloosa's spotted blanket over the rump
  let pat = hsU(h, 5);
  if (pat < 0.27) {
    let amount = 0.25 + 0.35 * hsU(h, 6);
    let b = hsNoise(bp * 1.7 + vec3f(3.1, 0.0, f32(h & 63u)), h ^ 1234u) + 0.3 * (0.9 - bp.y);
    col = mix(col, HS_WHITE, smoothstep(1.0 - amount, 1.04 - amount, b));
  } else if (pat < 0.4) {
    let blanket = smoothstep(0.1, -0.35, bp.z) * smoothstep(0.85, 1.1, bp.y + 0.15 * n1);
    let spot = smoothstep(0.68, 0.72, hsNoise(bp * 9.0, h ^ 4321u));
    col = mix(col, mix(HS_WHITE, body * 0.7, spot), blanket);
  }
  // socks: each leg white to its own height (none for most), the edge ragged
  if (bp.y < 0.62) {
    let leg = select(0, 1, bp.x > 0.0) + select(0, 2, bp.z > 0.0);
    let sock = select(0.0, 0.1 + 0.42 * hsU(h, 11 + leg), hsU(h, 10 + 4 * leg) < 0.5);
    col = mix(col, HS_WHITE, smoothstep(sock + 0.02, sock - 0.02, bp.y + 0.04 * n1));
  }
  // the face: a star on the forehead, or a blaze down the front of the face
  let face = hsU(h, 20);
  let front = smoothstep(0.1, 0.35, bn.z + 0.4 * bn.y) * step(1.02, bp.z);
  if (face > 0.25 && face < 0.55) {
    col = mix(col, HS_WHITE, smoothstep(0.075, 0.055, length(bp - vec3f(0.0, 1.56, 1.16))) * front);
  } else if (face >= 0.55) {
    let w = select(0.04, 0.075, face > 0.86);
    col = mix(col, HS_WHITE, smoothstep(w, w - 0.015, abs(bp.x)) * front * step(1.06, bp.z));
  }
  // the mane and the tail
  col = mix(col, hair, smoothstep(0.3, 0.7, hairTag));
  // the hooves
  return mix(col, vec3f(0.07, 0.06, 0.05), smoothstep(0.085, 0.06, bp.y));
}

/** The coat as lit at the pixel (the texture's brushwork over it), its distance and direction from the eye. */
fn hsShade(i: HsOut, tex: vec3f) -> vec3f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let p = u.camPos + i.rel;
  var n = normalize(i.n);
  if (dot(n, rd) > 0.0) { n = -n; }
  // the painted texture's light and shade, over the colour of each of its own parts (the cream and the tan)
  let lum = dot(tex, vec3f(0.2126, 0.7152, 0.0722));
  let sat = max(tex.r, max(tex.g, tex.b)) - min(tex.r, min(tex.g, tex.b));
  let brush = clamp(lum / mix(0.66, 0.24, smoothstep(0.04, 0.16, sat)), 0.65, 1.35);
  var c = hsCoat(i.bp, i.bn, i.hair, i.seeds.x, i.seeds.y) * brush;
  c = mix(c, tex, smoothstep(0.09, 0.03, lum)); // (the eyes, the nostrils as painted)
  let L = keyDir();
  let ndl = max(dot(n, L), 0.0);
  var sh = 0.0;
  if (ndl > 0.0) {
    sh = sunAt(p.xz) * treeShadowAt(p) * skyShadowAt(p);
    if (sh > 0.0) { sh = min(sh, heroShadowAt(i.rel)); }
  }
  // (a soft wrap of the light round the coat, and the belly a little shaded from the sky)
  let wrapL = max(dot(n, L) * 0.8 + 0.2, 0.0);
  var col = c * (keyCol(p.y) * mix(ndl, wrapL, 0.35) * sh + skyAmb(n, 0.3 + 0.35 * max(n.y, 0.0)) * (0.75 + 0.25 * n.y));
  col += vec3f(0.9, 0.22, 0.16) * (0.7 * i.fx.y); // (a blow's flash)
  return surfaceFog(col, dist, p.y, rd);
}

@fragment
fn fsHorse(i: HsOut) -> @location(0) vec4f {
  // (sampled where the control flow is uniform: before the early returns)
  let tex = textureSample(horseTex, horseSamp, i.uv).rgb;
  if (u.camPos.y < u.camWater) { return underwater(normalize(i.rel)); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  return display(hsShade(i, tex));
}

/** A vanishing horse: dissolving in noise (gone where it is below the vanish), its edge glowing pale. */
@fragment
fn fsHorseFade(i: HsOut) -> @location(0) vec4f {
  let tex = textureSample(horseTex, horseSamp, i.uv).rgb;
  let edge = i.fx.x * 1.2 - 0.1;
  let nz = hsNoise(i.bp * 6.0, i.seeds.x ^ 999u) * 0.8 + hsNoise(i.bp * 17.0, i.seeds.x) * 0.2;
  if (nz < edge) { discard; }
  if (u.camPos.y < u.camWater) { return underwater(normalize(i.rel)); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  let glow = smoothstep(edge + 0.09, edge, nz) * step(0.001, i.fx.x);
  return display(hsShade(i, tex) + vec3f(0.9, 0.85, 0.65) * glow);
}
`;
}
