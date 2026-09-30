/**
 * Sky islands in WGSL: the twin of src/lib/skyIslands.ts, a polar-mesh draw
 * (one instance per island surface: even = top, odd = underside) and an
 * analytic shadow the islands cast on everything below them.
 *
 * The CPU picks the islands near the camera (islandsNear) and uploads them,
 * chunk-local, into IslandU; the vertex shader places the shared polar grid
 * (ring i, bearing j) on each island's outline with ρ = i / rings exactly, so
 * top and underside share their rim vertices bit for bit.
 *
 * Needs, from the raster shader: u (Frame), vnoise, the light API (sky.wgsl.ts),
 * surfaceFog, display, underwater, heroShadowAt, project().
 */

import { SKY_ISLANDS as S } from '../constants';
import { wgslF32 as f } from './worldgen.wgsl';

export const MAX_ISLANDS = 16;
/** count vec4, then three vec4 per island. */
export const ISLANDS_BYTES = 16 + MAX_ISLANDS * 48;
export const ISLAND_RINGS = 40;
export const ISLAND_SEGMENTS = 96;

/** Triangles of the polar grid: rings 0..rings (ring 0 = the centre, repeated), bearings wrap. */
export function polarIndices(rings: number, segments: number): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(rings * segments * 6);
  let o = 0;
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const j1 = (j + 1) % segments;
      const a = i * segments + j, b = (i + 1) * segments + j, c = (i + 1) * segments + j1, d = i * segments + j1;
      out[o++] = a; out[o++] = b; out[o++] = c;
      out[o++] = a; out[o++] = c; out[o++] = d;
    }
  }
  return out;
}

export const SKY_ISLANDS_WGSL = /* wgsl */ `
struct IslandU {
  count: vec4u,
  a: array<vec4f, ${MAX_ISLANDS}>, // centre x, z (chunk-local), radius, top base height
  b: array<vec4f, ${MAX_ISLANDS}>, // underside depth, outline phases a1, a2, a3
  s: array<vec4i, ${MAX_ISLANDS}>, // noise seed
};
@group(0) @binding(10) var<uniform> isl: IslandU;

const ISL_RINGS: u32 = ${ISLAND_RINGS}u;
const ISL_SEGS: u32 = ${ISLAND_SEGMENTS}u;
const ISL_HILL: f32 = ${f(S.hill)};
const ISL_CLIFF: f32 = ${f(S.cliff)};
const ISL_RIM: f32 = ${f(S.rimDrop)};
const ISL_OUTLINE: f32 = ${f(S.outline)};

fn islOutline(k: u32, th: f32) -> f32 {
  let b = isl.b[k];
  return 1.0 + ISL_OUTLINE * (0.6 * sin(3.0 * th + b.y) + 0.4 * sin(5.0 * th + b.z) + 0.25 * sin(8.0 * th + b.w));
}
fn islRho(k: u32, xz: vec2f) -> f32 {
  let d = xz - isl.a[k].xy;
  return length(d) / (isl.a[k].z * islOutline(k, atan2(d.y, d.x)));
}
fn islTopR(k: u32, xz: vec2f, rho: f32) -> f32 {
  let g = xz + u.origin;
  let sd = isl.s[k].x;
  let n = 0.65 * vnoise(g.x / 90.0, g.y / 90.0, sd) + 0.35 * vnoise(g.x / 37.0, g.y / 37.0, sd + 1);
  let relief = ISL_HILL * (0.35 * (1.0 - rho * rho) + 0.65 * (0.5 + 0.5 * n));
  return isl.a[k].w + relief * (1.0 - smoothstep(0.78, 1.0, rho)) - ISL_RIM * smoothstep(0.85, 1.0, rho);
}
fn islBottomR(k: u32, xz: vec2f, rho: f32) -> f32 {
  let g = xz + u.origin;
  let sd = isl.s[k].x;
  let wall = ISL_CLIFF * (1.0 - smoothstep(0.86, 1.0, rho));
  let v = max(0.0, 1.0 - rho / 0.86);
  let cone = isl.b[k].x * select(pow(v, 0.75), 0.0, v <= 0.0) * (0.8 + 0.35 * (0.5 + 0.5 * vnoise(g.x / 60.0, g.y / 60.0, sd + 3)));
  let rock = 10.0 * (0.5 + 0.5 * vnoise(g.x / 28.0, g.y / 28.0, sd + 2)) * (1.0 - smoothstep(0.8, 1.0, rho));
  let dn = max(0.0, vnoise(g.x / 26.0, g.y / 26.0, sd + 4));
  let drip = 38.0 * dn * dn * dn * (1.0 - smoothstep(0.55, 0.9, rho));
  return isl.a[k].w - ISL_RIM - wall - cone - rock - drip;
}
fn islSurf(k: u32, xz: vec2f, bottom: bool) -> f32 {
  let rho = islRho(k, xz);
  return select(islTopR(k, xz, rho), islBottomR(k, xz, rho), bottom);
}

// Top of the island under chunk-local xz (well inside its outline), or -1e9.
fn islandTopAt(xz: vec2f, rhoMax: f32) -> f32 {
  for (var k = 0u; k < isl.count.x; k++) {
    let rho = islRho(k, xz);
    if (rho < rhoMax) { return islTopR(k, xz, rho); }
  }
  return -1e9;
}

// Sun visibility under the islands (all but island \`skip\`): the sun ray tested
// against each island's top disc and a narrower disc half way down its cone.
fn islandShadowAt(p: vec3f, skip: u32) -> f32 {
  let L = keyDir();
  if (L.y < 0.01) { return 1.0; } // a light on the horizon: no island stands between
  var vis = 1.0;
  for (var k = 0u; k < isl.count.x; k++) {
    if (k == skip) { continue; }
    let A = isl.a[k];
    let t1 = (A.w - p.y) / L.y;
    if (t1 <= 0.0) { continue; }
    let d1 = length(p.xz + L.xz * t1 - A.xy) / A.z;
    var occ = 1.0 - smoothstep(0.8, 1.05, d1);
    let ym = A.w - ISL_RIM - ISL_CLIFF - 0.45 * isl.b[k].x;
    let t2 = (ym - p.y) / L.y;
    if (t2 > 0.0) {
      let d2 = length(p.xz + L.xz * t2 - A.xy) / (A.z * 0.55);
      occ = max(occ, 1.0 - smoothstep(0.7, 1.0, d2));
    }
    vis = min(vis, 1.0 - 0.88 * occ);
  }
  return vis;
}

struct IOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) rho: f32,
  @location(3) @interpolate(flat) kb: u32, // island index · 2 + underside
};

@vertex
fn vsIsland(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> IOut {
  let k = ii / 2u;
  let bottom = (ii & 1u) == 1u;
  let i = vi / ISL_SEGS;
  let j = vi % ISL_SEGS;
  let rho = f32(i) / f32(ISL_RINGS);
  let th = f32(j) * (6.2831853 / f32(ISL_SEGS));
  let A = isl.a[k];
  let xz = A.xy + (A.z * islOutline(k, th) * rho) * vec2f(cos(th), sin(th));
  let y = select(islTopR(k, xz, rho), islBottomR(k, xz, rho), bottom);
  let e = 0.75;
  let gx = (islSurf(k, xz + vec2f(e, 0.0), bottom) - islSurf(k, xz - vec2f(e, 0.0), bottom)) / (2.0 * e);
  let gz = (islSurf(k, xz + vec2f(0.0, e), bottom) - islSurf(k, xz - vec2f(0.0, e), bottom)) / (2.0 * e);
  var n = normalize(vec3f(-gx, 1.0, -gz));
  if (bottom) { n = normalize(vec3f(gx, -1.0, gz)); }
  let rel = vec3f(xz.x, y, xz.y) - u.camPos;
  var o: IOut;
  o.pos = project(rel);
  o.rel = rel;
  o.n = n;
  o.rho = rho;
  o.kb = k * 2u + select(0u, 1u, bottom);
  return o;
}

@fragment
fn fsIsland(i: IOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < W_LEVEL) { return underwater(rd); }
  let n = normalize(i.n);
  let k = i.kb / 2u;
  let bottom = (i.kb & 1u) == 1u;
  let p = u.camPos + i.rel;
  let g = p.xz + u.origin;
  var albedo: vec3f;
  if (!bottom) {
    let patchN = 0.5 + 0.5 * vnoise(g.x / 23.0, g.y / 23.0, isl.s[k].x + 5);
    // The grass blades' average colour (as groundTint does for the terrain), a little darker.
    let grass = mix(vec3f(0.04, 0.085, 0.022), vec3f(0.068, 0.11, 0.024), patchN);
    let rock = vec3f(0.16, 0.14, 0.12);
    albedo = mix(grass, rock, smoothstep(0.9, 0.985, i.rho) + smoothstep(0.55, 0.9, 1.0 - n.y));
  } else {
    // Layered rock (noise-warped strata with dark seams), mossy near the rim, darker toward the tip.
    let layer = fract(p.y / 6.5 + 1.6 * vnoise(g.x / 45.0, g.y / 45.0, isl.s[k].x + 7));
    let tone = 0.5 + 0.5 * vnoise(floor(p.y / 6.5 + 1.6 * vnoise(g.x / 45.0, g.y / 45.0, isl.s[k].x + 7)) * 3.1, 0.5, isl.s[k].x + 9);
    albedo = mix(vec3f(0.12, 0.105, 0.095), vec3f(0.21, 0.185, 0.16), tone) * (1.0 - 0.35 * smoothstep(0.9, 1.0, layer));
    albedo = mix(albedo, vec3f(0.07, 0.11, 0.05), smoothstep(0.93, 1.0, i.rho) * 0.7);
    albedo *= mix(0.6, 1.0, smoothstep(0.0, 0.7, i.rho));
  }
  var sun = islandShadowAt(p, k);
  if (sun > 0.0) { sun = min(sun, heroShadowAt(i.rel)); }
  let ndl = dot(n, keyDir());
  let ramp = smoothstep(-0.12, 0.45, ndl);
  // Sky from above, the green world's bounce from below.
  let amb = skyAmb(n, 0.3 + 0.28 * max(n.y, 0.0)) + bounceCol(vec3f(0.06, 0.08, 0.04)) * max(-n.y, 0.0);
  var col = albedo * (keyCol(p.y) * 0.62 * ramp * sun + amb);
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}
`;
