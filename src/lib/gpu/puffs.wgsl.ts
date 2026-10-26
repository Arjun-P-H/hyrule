/**
 * Puffs in WGSL: the cloud world's stepping clouds and its stacks' billows (skyworld/puffs.ts) drawn
 * as instanced squashed spheres (vertex pulling from a storage buffer, the
 * sphere made from the vertex index). The top matches what the hero stands on
 * (the puff's half height); the underside hangs deeper, as a cloud's does.
 * A springy puff glows warm round its edge.
 *
 * Needs, from the raster shader: u (Frame), vnoise, the light API (sky.wgsl.ts),
 * sunVisibility, surfaceFog, display, project().
 */

import { CLOUDS } from '../constants';
import { wgslF32 as f } from './worldgen.wgsl';

export const MAX_PUFFS = 4096;
export const PUFF_BYTES = 32;
const SLICES = 16, RINGS = 8;
/** Vertices per puff: a quad (two triangles) per slice and ring. */
export const PUFF_VERTS = SLICES * RINGS * 6;

export const PUFFS_WGSL = /* wgsl */ `
struct PuffG {
  c: vec4f, // centre (chunk-local), spring lift (m; 0: plain; −1: a stack's billow, round underneath)
  r: vec4f, // radii; a per-puff hash in [0, 1)
};
@group(0) @binding(30) var<storage, read> puffs: array<PuffG>;

struct PuffOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) spring: f32,
};

@vertex
fn vsPuff(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PuffOut {
  let p = puffs[ii];
  let corners = array<vec2u, 6>(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(1u, 1u), vec2u(0u, 0u), vec2u(1u, 1u), vec2u(0u, 1u));
  let q = vi / 6u;
  let c = corners[vi % 6u];
  let th = f32(q % ${SLICES}u + c.x) / ${f(SLICES)} * 6.2831853;
  let ph = f32(q / ${SLICES}u + c.y) / ${f(RINGS)} * 3.1415927; // 0 the top … π the bottom
  let d = vec3f(sin(ph) * cos(th), cos(ph), sin(ph) * sin(th));
  // billows round the sides (none at the top: it is what is stood on); the underside hangs deeper
  let lump = 1.0 + 0.14 * sin(ph) * vnoise(th * 1.6 + p.r.w * 37.0, ph * 1.8, 8401);
  let down = select(1.0, 0.55 * max(p.r.x, p.r.z) / p.r.y, d.y < 0.0 && p.c.w >= 0.0);
  let local = d * p.r.xyz * vec3f(lump, down, lump);
  var o: PuffOut;
  o.rel = p.c.xyz + local - u.camPos;
  o.pos = project(o.rel);
  o.n = normalize(d / (p.r.xyz * vec3f(1.0, down, 1.0)));
  o.spring = p.c.w;
  return o;
}

@fragment
fn fsPuff(i: PuffOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let n = normalize(i.n);
  let p = u.camPos + i.rel;
  let albedo = vec3f(${CLOUDS.albedo.map(f).join(', ')}) * mix(0.74, 1.0, smoothstep(900.0, 1250.0, p.y));
  // cloud wraps the light round itself (lit as the cloud ground is: terrain.wgsl.ts CLOUD_WORLD)
  let ramp = clamp((dot(n, keyDir()) + 0.05) / 1.05, 0.0, 1.0);
  let sun = sunVisibility(p, n * 0.1);
  let amb = skyAmb(n, 0.22 + 0.28 * max(n.y, 0.0)) * 0.55;
  // cumulus: the underside in its own shadow, cool and dark; the top bright; a silver lining toward the sun
  let under = smoothstep(0.15, -0.65, n.y);
  var col = albedo * (keyCol(p.y) * ramp * sun + amb * mix(1.0, 0.8, under)) * mix(vec3f(1.0), vec3f(0.68, 0.72, 0.82), under);
  col += albedo * keyCol(p.y) * (0.3 * smoothstep(0.3, 1.0, n.y) * sun);
  col += keyCol(p.y) * (0.45 * pow(1.0 - abs(dot(n, rd)), 4.0) * pow(max(dot(rd, keyDir()), 0.0), 3.0) * sun);
  // a springy puff: a warm glow round its edge
  let rim = pow(1.0 - abs(dot(n, rd)), 3.0);
  col += vec3f(1.0, 0.72, 0.32) * rim * 0.5 * step(0.5, i.spring);
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}
`;
