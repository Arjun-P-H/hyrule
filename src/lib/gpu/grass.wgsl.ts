/**
 * Grass: GPU-scattered blades on world-anchored lattices around the camera.
 * ---------------------------------------------------------------------------
 * Ring r is an n × n lattice of cells of spacing spacing0 · 2^r around the
 * camera; each cell holds one blade candidate at a hashed position, so blades
 * stay put in the world while the rings follow the camera. Spacing doubles
 * with distance, which keeps the blade count per screen area roughly
 * constant; neighbouring rings cross-fade (blades grow and shrink, never
 * pop) over the outer `blend` fraction of each ring's radius.
 *
 * `grassScatter` (compute, one invocation per candidate) drops candidates
 * outside the view frustum, under water, on steep rock, or where the biome
 * grows little (biomeColor().veg, clumped), and appends the rest to a blade
 * buffer; the instance count of an indirect draw is the append counter.
 * `vsGrass` bends each blade (lean, travelling wind gusts, pushed aside
 * around the hero) as a 7-vertex strip; `fsGrass` shades it with a soft
 * wrap-lit, translucent, tip-brightened stylized look. Sun visibility is the
 * clipmap's cached ground visibility at the blade's root.
 *
 * Needs, from the raster shader: u (Frame), heightAt, gradientAt, sunAt,
 * surfaceFog, display, heroShadowAt, hero, mesh (projection), project(), and
 * the light API (keyDir, keyCol, skyAmb: sky.wgsl.ts).
 */

import { wgslF32 as f } from './worldgen.wgsl';

export type GrassSpec = {
  rings: number;
  /** Cells per ring side (multiple of 8). */
  n: number;
  /** Cell size of ring 0 in metres (power of two). */
  spacing0: number;
  /** Cross-fade width as a fraction of a ring's radius. */
  blend: number;
  /** Blade height range, metres. */
  height: [number, number];
  /** Blade width in ring 0, metres (grows with each ring). */
  width: number;
  /** Global density multiplier (0..1). */
  density: number;
};

/** Grass uniform block: per ring two vec4, then params and wind. */
export const grassBytes = (rings: number) => 16 * (2 * rings + 2);
/** One blade in the scatter buffer. */
export const BLADE_BYTES = 32;
/** Vertices per blade (triangle strip). */
export const BLADE_VERTS = 7;

export function grassWGSL(spec: GrassSpec): string {
  const R = spec.rings;
  return /* wgsl */ `
struct GrassU {
  ringF: array<vec4f, ${R}>,  // per ring: camera position in cells mod 1 (x, z), spacing, radius
  ringI: array<vec4i, ${R}>,  // per ring: absolute lattice index of the first cell (x, z)
  params: vec4f,              // blend fraction, density, time, unused
  wind: vec4f,                // direction x, z; strength; gust scale
};
@group(0) @binding(6) var<uniform> grass: GrassU;

struct Blade {
  p: vec4f,       // root (chunk-local xyz), height
  dir: vec2f,     // lean direction (unit, xz)
  width: f32,
  packed: u32,    // sqrt(colour) rgb, sun visibility (unorm8 × 4)
};
struct DrawArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};
@group(1) @binding(0) var<storage, read_write> bladesOut: array<Blade>;
@group(1) @binding(1) var<storage, read_write> drawArgs: DrawArgs;
@group(1) @binding(2) var<storage, read> bladesIn: array<Blade>;

const GRASS_N: u32 = ${spec.n}u;
const GRASS_RINGS: u32 = ${R}u;
const GRASS_H0: f32 = ${f(spec.height[0])};
const GRASS_H1: f32 = ${f(spec.height[1])};
const GRASS_W0: f32 = ${f(spec.width)};

// Grass palette (linear): the biome's ground colour pulled toward a vivid
// yellow-green, varied between warmer and cooler greens by a 30 m patch field.
fn grassColor(bioCol: vec3f, g: vec2f, v: f32) -> vec3f {
  let patchHue = 0.5 + 0.5 * vnoise(g.x * 0.033, g.y * 0.033, u.seed + 7707);
  let warm = vec3f(0.135, 0.205, 0.035);
  let cool = vec3f(0.06, 0.16, 0.05);
  let vivid = mix(cool, warm, clamp(patchHue * 1.2 - 0.1 + (v - 0.5) * 0.35, 0.0, 1.0));
  return mix(bioCol, vivid, 0.62);
}

// How densely grass grows here (0..1): the biome's vegetation weight, broken
// into meadows by a 22 m patch field, and none on steep ground.
fn grassDensity(veg: f32, slope: f32, g: vec2f) -> f32 {
  let patchN = 0.5 + 0.5 * vnoise(g.x * 0.045, g.y * 0.045, u.seed + 7777);
  return veg * smoothstep(0.15, 0.55, patchN + 0.25 * veg) * (1.0 - smoothstep(0.7, 1.1, slope));
}

// Ground under and beyond the blades takes the grass's average colour, so a
// field reads as one surface where the blades thin out with distance.
fn groundTint(col: vec3f, veg: f32, slope: f32, g: vec2f) -> vec3f {
  let cover = grassDensity(veg, slope, g);
  return mix(col, grassColor(col, g, 0.5) * 0.78, clamp(cover * 1.2, 0.0, 0.85));
}

fn grassHash(ix: i32, iz: i32, s: i32) -> vec4f {
  let h = hashU4(vec4u(
    hash3(ix, iz, s),
    hash3(ix, iz, s + 1),
    hash3(ix, iz, s + 2),
    hash3(ix, iz, s + 3)));
  return vec4f(h >> vec4u(8u)) * (1.0 / 16777216.0);
}

// Frustum test of a sphere at camera-relative c.
fn inFrustum(c: vec3f, r: f32) -> bool {
  let v = c * u.basis;
  if (v.z < -r) { return false; }
  let tx = u.tanHalf * u.res.x / u.res.y;
  let ty = u.tanHalf;
  if (abs(v.x) - r * sqrt(1.0 + tx * tx) > v.z * tx) { return false; }
  if (abs(v.y) - r * sqrt(1.0 + ty * ty) > v.z * ty) { return false; }
  return true;
}

@compute @workgroup_size(8, 8, 1)
fn grassScatter(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= GRASS_N || id.y >= GRASS_N || id.z >= GRASS_RINGS) { return; }
  let r = id.z;
  let RF = grass.ringF[r];
  let RI = grass.ringI[r];
  let ix = RI.x + i32(id.x);
  let iz = RI.y + i32(id.y);
  let hs = grassHash(ix, iz, u.seed + 7001 + i32(r) * 131);
  // Root relative to the camera: cells from the camera's cell, minus its position inside it.
  let half = f32(GRASS_N / 2u);
  let relXZ = (vec2f(f32(id.x), f32(id.y)) - half + hs.xy - RF.xy) * RF.z;
  let d = length(relXZ);
  // Cross-fade: grow in over the inner band (ring r > 0), shrink out over the outer band.
  let blend = grass.params.x;
  var w = 1.0 - smoothstep(RF.w * (1.0 - blend), RF.w, d);
  if (r > 0u) {
    let inner = grass.ringF[r - 1u].w;
    w *= smoothstep(inner * (1.0 - blend), inner, d);
  }
  if (w <= 0.02) { return; }

  let xz = u.camPos.xz + relXZ;
  if (inRuinSite(xz)) { return; }
  // On a sky island's top the blade grows there instead of on the ground below.
  let islandTop = islandTopAt(xz, 0.93);
  let onIsland = islandTop > -1e8;
  let e = select(heightAt(xz, vec2f(0.0), RF.z), islandTop, onIsland);
  if (!onIsland && e < W_LEVEL + 0.05) { return; }
  let hMax = GRASS_H1 * (1.0 + 0.35 * f32(r));
  let rel = vec3f(relXZ.x, e - u.camPos.y, relXZ.y);
  if (!inFrustum(rel + vec3f(0.0, 0.5 * hMax, 0.0), hMax)) { return; }

  // Biome: the terrain shading's vegetation weight, clumped at the 14 m scale.
  let g = xz + u.origin;
  var slope = 0.2;
  var bio = Biome(vec3f(0.075, 0.14, 0.032), 0.95); // island tops: lush meadow
  if (!onIsland) {
    slope = length(gradientAt(xz, max(RF.z, 0.25), RF.z));
    if (slope > 1.1) { return; }
    let T = temperature(g, e);
    let M = moisture(g, u.seed);
    let vegNoise = vnoise(g.x / VEG_WL, g.y / VEG_WL, u.seed + 171);
    bio = biomeColor(T, M, e, slope, vegNoise);
  }
  // Island tops are meadows throughout; the ground follows the biome and its patches.
  let dens = select(grassDensity(bio.veg, slope, g), max(grassDensity(bio.veg, slope, g), 0.85), onIsland) * grass.params.y;
  if (hs.z > dens) { return; }

  let height = mix(GRASS_H0, hMax, hs.w) * (0.55 + 0.45 * smoothstep(0.2, 0.9, dens)) * w;
  let a = hs.w * 37.0 + hs.z * 11.0;
  let root = vec3f(xz.x, e, xz.y);
  let bladeMid = vec3f(xz.x, e + 0.25 * height, xz.y);
  let sun = select(sunAt(xz), 1.0, onIsland) * treeShadowAt(bladeMid) * islandShadowAt(bladeMid, 0xffffffffu);
  // Blade colour: the grass palette at this spot, varied per blade.
  let col = clamp(grassColor(bio.col, g, fract(hs.x * 7.13 + hs.y * 3.7)) * (0.85 + 0.3 * fract(hs.y * 5.3)), vec3f(0.0), vec3f(1.0));
  let idx = atomicAdd(&drawArgs.instanceCount, 1u);
  var b: Blade;
  b.p = vec4f(root, height);
  b.dir = vec2f(cos(a), sin(a));
  b.width = GRASS_W0 * exp2(0.8 * f32(r)) * (0.8 + 0.4 * fract(hs.x * 13.1));
  b.packed = pack4x8unorm(vec4f(sqrt(col), sun));
  bladesOut[idx] = b;
}

struct GOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) packed: u32,
  @location(3) t: f32,
};

// Gust field: a slow travelling value-noise pattern plus a fine flutter.
fn windAt(xz: vec2f, time: f32) -> f32 {
  let wd = grass.wind.xy;
  let q = xz * grass.wind.w - wd * time * 0.35;
  let gust = 0.5 + 0.5 * vnoise(q.x, q.y, 991);
  let flutter = sin(dot(xz, wd) * 1.7 - time * 3.1 + xz.x * 0.3) * 0.12;
  return grass.wind.z * (gust * gust * 1.4 + flutter);
}

@vertex
fn vsGrass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GOut {
  let b = bladesIn[ii];
  let h = b.p.w;
  let t = select(f32(vi / 2u) / 3.0, 1.0, vi == 6u);
  let side = select(select(-1.0, 1.0, (vi & 1u) == 1u), 0.0, vi == 6u);
  let root = b.p.xyz - u.camPos;
  let f = vec3f(b.dir.x, 0.0, b.dir.y);
  // Turn the blade's face part-way toward the camera so it is never seen edge-on.
  let toCam = normalize(vec3f(-root.x, 0.0, -root.z) + vec3f(1e-4, 0.0, 0.0));
  let sv = normalize(mix(vec3f(-f.z, 0.0, f.x), vec3f(-toCam.z, 0.0, toCam.x), 0.55));
  // Bend: natural lean, wind gusts, pushed aside around the hero.
  var bend = f * 0.25 + vec3f(grass.wind.x, 0.0, grass.wind.y) * windAt(b.p.xz + u.origin, u.time);
  if (hero.bound.w > 0.0) {
    let dh = root.xz - hero.j[0].xz;
    let dist = length(dh);
    let feet = hero.misc.y + 0.3 - root.y; // hero ground (camera-relative) near the blade root
    if (dist < 0.9 && abs(feet) < 1.2) {
      let push = (0.9 - dist) / 0.9;
      bend += vec3f(dh.x, 0.0, dh.y) / max(dist, 0.05) * push * 1.6;
    }
  }
  let bl = length(bend);
  let up = 1.0 - 0.45 * min(bl, 1.2) * t;
  let pos = root + vec3f(0.0, h * t * up, 0.0) + bend * (h * t * t) + sv * (side * 0.5 * b.width * pow(1.0 - t, 0.8));
  // Normal: the strip's face, rounded across the width.
  let dPdt = vec3f(0.0, h * up, 0.0) + bend * (2.0 * h * t);
  var n = normalize(cross(sv, dPdt));
  n = normalize(n + sv * side * 0.45);
  var o: GOut;
  o.pos = project(pos);
  o.rel = pos;
  o.n = n;
  o.packed = b.packed;
  o.t = t;
  return o;
}

@fragment
fn fsGrass(i: GOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < W_LEVEL) { return underwater(rd); }
  var n = normalize(i.n);
  if (dot(n, rd) > 0.0) { n = -n; }
  let c = unpack4x8unorm(i.packed);
  let base = c.rgb * c.rgb;
  var sun = c.a;
  if (sun > 0.0) { sun = min(sun, heroShadowAt(i.rel)); }
  let t = i.t;
  // Dark roots, sun-bleached tips.
  let albedo = mix(base * 0.4, base * 1.35 + vec3f(0.035, 0.03, 0.0), t * t);
  // Soft wrap lighting (stylized), translucency toward the sun, sky fill, a faint sheen.
  let L = keyDir();
  let kc = keyCol(u.camPos.y + i.rel.y);
  let wrap = clamp((dot(n, L) + 0.5) / 1.5, 0.0, 1.0);
  let trans = pow(max(dot(rd, L), 0.0), 3.0) * 0.6 * t;
  // (no sheen under a light at the horizon: the blades' glints would line up into a glitter path)
  let spec = pow(max(dot(reflect(rd, n), L), 0.0), 24.0) * 0.06 * t * smoothstep(0.0, 0.2, L.y);
  var col = albedo * (kc * (0.62 * wrap + trans) * sun + skyAmb(n, 0.34 + 0.3 * t))
          + kc * spec * sun;
  col = surfaceFog(col, dist, u.camPos.y + i.rel.y, rd);
  return display(col);
}
`;
}
