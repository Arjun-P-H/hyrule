/**
 * The colossi in WGSL (colossus/colossi.ts poses them; gpu/horses.ts SkinnedGpu draws them): the colossus mesh, one
 * instance a colossus, skinned in the vertex shader by its joints' matrices (camera-relative world from the CPU),
 * its painted stone lit as the horses are (the sun by the ground's visibility of it, the sky's ambient) and fogged.
 * Its core (the eye in its face) and the cracks of its stone glow when it lies felled (and a little when enraged);
 * a blow flashes; slain, it dissolves in noise along a glowing edge (fsColossusFade, drawn apart).
 *
 * Needs, from the raster shader: u (Frame), project(), display(), surfaceFog, keyDir, keyCol, skyAmb, sunAt,
 * treeShadowAt, skyShadowAt, underwater, rlFogged, hash3; with `shadows`, forest (the shadow map).
 */

import { COLOSSUS_JOINTS, COLOSSUS_STRIDE } from '../colossus/colossi';

export function colossusWGSL(shadows: boolean): string {
  return /* wgsl */ `
const CS_STRIDE: u32 = ${COLOSSUS_STRIDE / 4}u; // vec4s a colossus
const CS_JOINTS: u32 = ${COLOSSUS_JOINTS}u;
const CS_CORE = vec3f(0.6, 16.3, 6.3);           // the eye in its face, in its bind pose
@group(1) @binding(73) var<storage, read> colossusData: array<vec4f>;
@group(1) @binding(74) var colossusTex: texture_2d<f32>;
@group(1) @binding(75) var colossusSamp: sampler;

struct CsIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
  @location(3) j: vec4u,
  @location(4) w: vec4f,
};

fn csSkin(ii: u32, j: vec4u, w: vec4f) -> array<vec4f, 3> {
  let b = ii * CS_STRIDE + 1u;
  var m = array<vec4f, 3>(vec4f(0.0), vec4f(0.0), vec4f(0.0));
  for (var k = 0u; k < 4u; k++) {
    let o = b + min(j[k], CS_JOINTS - 1u) * 3u;
    m[0] += colossusData[o] * w[k];
    m[1] += colossusData[o + 1u] * w[k];
    m[2] += colossusData[o + 2u] * w[k];
  }
  return m;
}

fn csPoint(m: array<vec4f, 3>, p: vec3f) -> vec3f {
  return vec3f(dot(m[0].xyz, p) + m[0].w, dot(m[1].xyz, p) + m[1].w, dot(m[2].xyz, p) + m[2].w);
}

struct CsOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(3) bp: vec3f,                          // where it is on the colossus in the bind pose
  @location(4) @interpolate(flat) fx: vec4f,       // its id (bits), core glow, how far it has vanished, a blow's flash
};

@vertex
fn vsColossus(@builtin(instance_index) ii: u32, v: CsIn) -> CsOut {
  let m = csSkin(ii, v.j, v.w);
  var o: CsOut;
  o.rel = csPoint(m, v.pos);
  o.pos = project(o.rel);
  o.n = vec3f(dot(m[0].xyz, v.nrm), dot(m[1].xyz, v.nrm), dot(m[2].xyz, v.nrm));
  o.uv = v.uv;
  o.bp = v.pos;
  o.fx = colossusData[ii * CS_STRIDE];
  return o;
}
${shadows ? `
@vertex
fn vsColossusShadow(@builtin(instance_index) ii: u32, v: CsIn) -> @builtin(position) vec4f {
  let q = csPoint(csSkin(ii, v.j, v.w), v.pos) - forest.shN.xyz;
  let Rr = forest.shC.w;
  return vec4f(dot(q, forest.shR.xyz) / Rr, dot(q, forest.shU.xyz) / Rr, 0.5 + 0.5 * dot(q, forest.shF.xyz) / forest.shR.w, 1.0);
}
` : ''}
/** Smooth 3D value noise 0..1. */
fn csNoise(p: vec3f, s: i32) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  var c: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let q = i + vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    c[k] = f32(hash3(q.x + q.z * 7919, q.y + q.z * 104729, s) & 0xffffu) / 65535.0;
  }
  let x0 = mix(mix(c[0], c[1], u.x), mix(c[2], c[3], u.x), u.y);
  let x1 = mix(mix(c[4], c[5], u.x), mix(c[6], c[7], u.x), u.y);
  return mix(x0, x1, u.z);
}

/** The stone as lit at the pixel: the painted colour, the sun and the sky, its glow and a blow's flash, fogged. */
fn csShade(i: CsOut, tex: vec3f) -> vec3f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let p = u.camPos + i.rel;
  var n = normalize(i.n);
  if (dot(n, rd) > 0.0) { n = -n; }
  let L = keyDir();
  let ndl = max(dot(n, L), 0.0);
  var sh = 0.0;
  if (ndl > 0.0) { sh = sunAt(p.xz) * treeShadowAt(p) * skyShadowAt(p); }
  // (a little wrap of the light, and its underside darker: a hulk standing in its own shade)
  let wrapL = max(dot(n, L) * 0.85 + 0.15, 0.0);
  let under = 0.55 + 0.45 * smoothstep(-0.2, 0.9, n.y + i.bp.y / 30.0);
  var col = tex * (keyCol(p.y) * mix(ndl, wrapL, 0.3) * sh + skyAmb(n, 0.35 + 0.35 * max(n.y, 0.0)) * under);
  // its core and the cracks of its stone (the painting's darkest seams) glowing
  let glow = i.fx.y;
  if (glow > 0.0) {
    let lum = dot(tex, vec3f(0.2126, 0.7152, 0.0722));
    let core = smoothstep(2.6, 0.3, length(i.bp - CS_CORE));
    let seam = smoothstep(0.035, 0.012, lum) * (0.6 + 0.4 * sin(u.time * 3.0 + i.bp.y * 0.4));
    col += vec3f(1.0, 0.5, 0.15) * glow * (2.2 * core + 0.35 * seam);
  }
  col += vec3f(0.9, 0.35, 0.2) * (0.35 * i.fx.w);
  return surfaceFog(col, dist, p.y, rd);
}

@fragment
fn fsColossus(i: CsOut) -> @location(0) vec4f {
  let tex = textureSample(colossusTex, colossusSamp, i.uv).rgb;
  if (u.camPos.y < u.camWater) { return underwater(normalize(i.rel)); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  return display(csShade(i, tex));
}

/** A slain colossus going: dissolving in noise (gone below the vanish), its edge glowing. */
@fragment
fn fsColossusFade(i: CsOut) -> @location(0) vec4f {
  let tex = textureSample(colossusTex, colossusSamp, i.uv).rgb;
  let edge = i.fx.z * 1.2 - 0.1;
  let s = i32(bitcast<u32>(i.fx.x) & 0xffffu);
  let nz = csNoise(i.bp * 0.35, s) * 0.75 + csNoise(i.bp * 1.3, s + 7) * 0.25;
  if (nz < edge) { discard; }
  if (u.camPos.y < u.camWater) { return underwater(normalize(i.rel)); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  let rim = smoothstep(edge + 0.07, edge, nz) * step(0.001, i.fx.z);
  return display(csShade(i, tex) + vec3f(1.0, 0.7, 0.35) * 2.0 * rim);
}
`;
}
