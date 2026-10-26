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
 * grows little (its baked cover, in meadow patches), and appends the rest to a blade
 * buffer; the instance count of an indirect draw is the append counter.
 * `vsGrass` bends each blade (lean, travelling wind gusts, pushed aside
 * around the hero) as a strip of three segments, two out past
 * GRASS_FAR_RING; `fsGrass` shades it with a soft
 * wrap-lit, translucent, tip-brightened stylized look. Sun visibility is the
 * clipmap's cached ground visibility at the blade's root.
 *
 * Up on the sky islands (when the camera is up there with them) blades grow
 * on an island's top instead of the land below it, in its biome's colours,
 * height and density, a share of them flowering (skyRealm.wgsl.ts).
 *
 * Needs, from the raster shader: u (Frame), heightAt, gradientAt, sunAt,
 * surfaceFog, display, heroShadowAt, hero, mesh (projection), project(), the
 * light API (keyDir, keyCol, skyAmb: sky.wgsl.ts), and with the sky realm
 * its lookups (rlFind, rlTop, …) and skyShadowAt.
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
/**
 * Level of detail: blades of rings below GRASS_FAR_RING bend in three segments, those of the rings
 * from it out (63 m and more away, 3 … 17 pixels tall) in two — the curve within a third of a pixel.
 * Each set has its own buffer and indirect draw (DrawArgs pair), drawn with segs() vertices a blade.
 */
export const GRASS_FAR_RING = 3;
export const BLADE_SEGS = [3, 2] as const;
/** Vertices per blade (triangle strip) for a set's segments. */
export const bladeVerts = (segs: number) => 2 * segs + 1;

/** How densely grass grows (0..1): the biome's cover, broken into meadows by a 22 m patch field, none on the steep. */
const GROUND_TINT_WGSL = /* wgsl */ `
fn grassDensityFrom(veg: f32, slope: f32, patchN: f32) -> f32 {
  return veg * smoothstep(0.15, 0.55, patchN + 0.25 * veg) * (1.0 - smoothstep(0.7, 1.1, slope));
}
`;

export function grassWGSL(spec: GrassSpec, islands = false): string {
  const R = spec.rings;
  return /* wgsl */ `${islands ? SKY_GRASS_WGSL : NO_SKY_GRASS_WGSL}
${GROUND_TINT_WGSL}
struct GrassU {
  ringF: array<vec4f, ${R}>,  // per ring: camera position in cells mod 1 (x, z), spacing, radius
  ringI: array<vec4i, ${R}>,  // per ring: absolute lattice index of the first cell (x, z)
  params: vec4f,              // blend fraction, density, time, the Eyries' district maps' version (sky blades are made again when it changes)
  wind: vec4f,                // direction x, z; strength; gust scale
};
@group(0) @binding(6) var<uniform> grass: GrassU;

struct Blade {
  p: vec4f,       // root (chunk-local xyz), height
  dir: vec2f,     // lean direction (unit, xz)
  width: f32,     // negative: a flowering blade (its colour is the flower's)
  packed: u32,    // sqrt(colour) rgb, sun visibility (unorm8 × 4)
};
struct DrawArgs {
  vertexCount: u32,
  instanceCount: atomic<u32>,
  firstVertex: u32,
  firstInstance: u32,
};
// The near and far blade sets (GRASS_FAR_RING): their buffers and indirect draws.
@group(1) @binding(0) var<storage, read_write> bladesOut: array<Blade>;
@group(1) @binding(1) var<storage, read_write> drawArgs: array<DrawArgs, 2>;
@group(1) @binding(3) var<storage, read_write> bladesOutFar: array<Blade>;
// The sky's blades per cell of every ring (skyBlade), GRASS_N² a ring.
@group(1) @binding(4) var<storage, read_write> bladeCache: array<vec4u>;
@group(1) @binding(2) var<storage, read> bladesIn: array<Blade>;
// Segments a blade of the set drawn bends in (a pipeline per set).
override BLADE_SEGS: u32 = ${BLADE_SEGS[0]}u;
const GRASS_FAR_RING: u32 = ${GRASS_FAR_RING}u;

const GRASS_N: u32 = ${spec.n}u;
const GRASS_RINGS: u32 = ${R}u;
const GRASS_H0: f32 = ${f(spec.height[0])};
const GRASS_H1: f32 = ${f(spec.height[1])};
const GRASS_W0: f32 = ${f(spec.width)};

fn grassHash(ix: i32, iz: i32, s: i32) -> vec4f {
  let h = hashU4(vec4u(
    hash3(ix, iz, s),
    hash3(ix, iz, s + 1),
    hash3(ix, iz, s + 2),
    hash3(ix, iz, s + 3)));
  return vec4f(h >> vec4u(8u)) * (1.0 / 16777216.0);
}

// The tuft a point belongs to: the nearest root of a 1 m lattice (global, so the tufts stay put in the world):
// x the distance to it (cells), y its own random 0 … 1, zw the vector from the point to the root (cells).
fn tuftAt(g: vec2f, seed: i32) -> vec4f {
  let ip = floor(g);
  let fr = g - ip;
  let i0 = vec2i(ip);
  var m = 64.0;
  var dn = vec2f(0.0);
  var hn = 0u;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let h = hash3(i0.x + i, i0.y + j, seed);
      let o = vec2f(f32(h & 0xfffu), f32((h >> 12u) & 0xfffu)) * INV_4095;
      let d = vec2f(f32(i), f32(j)) + o - fr;
      let dd = dot(d, d);
      if (dd < m) { m = dd; dn = d; hn = h; }
    }
  }
  return vec4f(sqrt(m), f32(hashU(hn ^ 0x68bc21ebu) >> 8u) * (1.0 / 16777216.0), dn);
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

// The blade the sky grows at a cell (camera-independent, so cached: bladeCache), as a record:
// x its tag (the cell, its ring, the seed); y the top's height (−1e9: no top here, the land's grass
// grows); z its colour (√ rgb, unorm8) and whether it flowers (a); w its height before the rings'
// cross-fade (0: none grows) and its density (f16 pair).
fn skyBladeRecord(xz: vec2f, hs: vec4f, r: u32) -> vec4u {
  var rec = vec4u(0u, bitcast<u32>(-1e9), 0u, 0u);
  let sky = skyGrassTop(xz);
  if (sky.x <= -1e8) { return rec; }
  rec.y = bitcast<u32>(sky.x);
  let g = xz + u.origin;
  let skyLook = skyGrassLook(sky.y, xz, sky);
  let bio = Biome(skyGrassColour(sky.y, xz, sky), skyLook.x);
  // Island tops grow as their biome says (patchy at its edges).
  let dens = bio.veg * (0.75 + 0.25 * smoothstep(0.2, 0.6, 0.5 + 0.5 * vnoise(g.x * 0.045, g.y * 0.045, u.seed + 7777))) * grass.params.y;
  if (hs.z > dens) { return rec; }
  let hMax = GRASS_H1 * (1.0 + 0.35 * f32(r));
  let height = mix(GRASS_H0, hMax, hs.w) * (0.55 + 0.45 * smoothstep(0.2, 0.9, dens)) * skyLook.y;
  // (the sky biomes keep more of their own palette: golden, mossy, frosted; a flowering blade, its flower's)
  let vary = fract(hs.x * 7.13 + hs.y * 3.7);
  var col = clamp(bio.col * (0.85 + 0.3 * vary) * (0.85 + 0.3 * fract(hs.y * 5.3)), vec3f(0.0), vec3f(1.0));
  let flower = fract(hs.x * 91.7 + hs.w * 13.3) < skyLook.z;
  if (flower) { col = skyFlower(u32(skyLook.w), fract(hs.y * 41.3 + hs.z * 7.7)); }
  rec.z = pack4x8unorm(vec4f(sqrt(col), select(0.0, 1.0, flower)));
  rec.w = pack2x16float(vec2f(height, dens));
  return rec;
}

// The record of the sky's blade at a cell: from the cache when it was made for this cell, else made
// now (cells entering a ring as the camera moves; a new world).
// Bench only: make every sky blade afresh each frame (as before the cache), to compare.
override NO_BLADE_CACHE: bool = false;
fn skyBlade(id: vec3u, ix: i32, iz: i32, xz: vec2f, hs: vec4f) -> vec4u {
  if (NO_BLADE_CACHE) { return skyBladeRecord(xz, hs, id.z); }
  let slot = (id.z * GRASS_N + u32(iz) % GRASS_N) * GRASS_N + u32(ix) % GRASS_N;
  // (the districts' version in the tag: when an Eyrie's map arrives its blades are made again)
  let tag = hash3(ix, iz, u.seed * 8 + i32(id.z) + i32(grass.params.w) * 977) | 1u;
  var rec = bladeCache[slot];
  if (rec.x != tag) {
    rec = skyBladeRecord(xz, hs, id.z);
    rec.x = tag;
    bladeCache[slot] = rec;
  }
  return rec;
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
  // On a sky island's top the blade grows there instead of on the land below, when the top is the nearer.
  let land = heightAt(xz, vec2f(0.0), RF.z);
  var rec = vec4u(0u, bitcast<u32>(-1e9), 0u, 0u);
  if (skyAloft()) { rec = skyBlade(id, ix, iz, xz, hs); }
  let onIsland = skyGrassNear(bitcast<f32>(rec.y), land);
  let e = select(land, bitcast<f32>(rec.y), onIsland);
  // The cheap tests first: what the camera cannot see, and the gaps between the tufts, are dropped before the
  // ground's biome layers (a dozen fetches) are read; most candidates are behind the camera or in a gap.
  let hMax = GRASS_H1 * (1.0 + 0.35 * f32(r));
  let rel = vec3f(relXZ.x, e - u.camPos.y, relXZ.y);
  if (!inFrustum(rel + vec3f(0.0, 0.5 * hMax, 0.0), hMax)) { return; }
  // Tufts (the near rings; out in the far ones the blades are single and the lawn is level): blades gather
  // round roots with bare ground between, some roots sparse, some tall; they lean away from their root.
  let tk = 1.0 - smoothstep(1.0, 3.0, f32(r));
  var tuft = vec4f(1.0, 0.5, 0.0, 0.0);
  var tuftK = 1.0;
  if (!onIsland && tk > 0.0) {
    tuft = tuftAt(xz + u.origin, u.seed + 7211);
    let tuftRoot = smoothstep(0.78, 0.12, tuft.x) * (0.45 + 1.1 * tuft.y);
    tuftK = mix(1.0, clamp(2.4 * tuftRoot, 0.0, 1.35), tk);
    // (the cover is at most 1, so a candidate over this grows nothing whatever the ground is)
    if (hs.z > grass.params.y * tuftK) { return; }
  }
  // The ground's biome layers at the root (terrain.wgsl.ts groundCacheAt): its water, cover, grass,
  // blade height and flowers. Reeds (tall blades) stand in the shallows; other grass stays on land.
  var gc: GroundCache;
  if (!onIsland) {
    gc = groundCacheAt(xz);
    // (reeds: tall blades in a wet land: long grass in a meadow is not one, and keeps off the water)
    let reeds = gc.bioB.a * BLADE_SCALE > 1.6 && gc.mat.y > 0.55;
    if (e < gc.wt - select(-0.05, 0.6, reeds)) { return; }
  }

  var height: f32;
  var packedCol: vec4f;
  var flower = false;
  var tuftV = vec4f(0.0);
  if (onIsland) {
    let hd = unpack2x16float(rec.w);
    if (hd.x <= 0.0) { return; }
    height = hd.x * w;
    packedCol = unpack4x8unorm(rec.z);
    flower = packedCol.a > 0.5;
  } else {
    // The biome's grass: its cover broken into meadows, none on steep rock, under snow or on lava.
    tuftV = tuft;
    // (the most the density can be, level ground, no snow, beach or lava: a candidate over it grows nothing,
    // and the slope, the snow and the beach, which are dear, are not asked: bare earth, roads, the gaps between tufts)
    if (hs.z > grassDensityFrom(gc.bioA.a, 0.0, gc.mat.z) * grass.params.y * tuftK) { return; }
    let slope = length(gradientAt(xz, max(RF.z, 0.25), RF.z));
    if (slope > 1.1) { return; }
    let p = vec3f(xz.x, e, xz.y);
    var dens = grassDensityFrom(gc.bioA.a, slope, gc.mat.z) * (1.0 - snowAt(gc, p, slope)) * (1.0 - beachAt(gc, p, slope)) * select(1.0, 0.0, gc.mat.y < -0.05) * grass.params.y;
    dens = dens * tuftK;
    if (hs.z > dens) { return; }
    let blade = gc.bioB.a * BLADE_SCALE;
    // (a blade is within a third of its tuft's height: the height varies from tuft to tuft, and with the ground's
    // long and short grass (blade), not blade by blade)
    let tuftH = mix(1.0, 0.7 + 0.6 * fract(tuft.y * 17.31), tk);
    height = mix(GRASS_H0, hMax, mix(hs.w, 0.35 + 0.65 * hs.w, tk)) * blade * tuftH * (0.55 + 0.45 * smoothstep(0.2, 0.9, dens)) * w;
    // Blade colour: the biome's grass here, varied per blade (and by tuft); a share flowering where the biome blooms.
    let vary = fract(hs.x * 7.13 + hs.y * 3.7);
    var col = clamp(gc.bioB.rgb * gc.bioB.rgb * (0.8 + 0.4 * vary) * (0.85 + 0.3 * fract(hs.y * 5.3)) * mix(1.0, 0.88 + 0.24 * tuft.y, tk), vec3f(0.0), vec3f(1.0));
    flower = fract(hs.x * 91.7 + hs.w * 13.3) < gc.mat.w;
    if (flower) {
      let fv = fract(hs.y * 41.3 + hs.z * 7.7);
      col = select(select(vec3f(0.85, 0.75, 0.15), vec3f(0.55, 0.3, 0.75), fv > 0.45), vec3f(0.9, 0.88, 0.85), fv > 0.8);
    }
    packedCol = vec4f(sqrt(col), select(0.0, 1.0, flower));
  }

  let a = hs.w * 37.0 + hs.z * 11.0;
  let root = vec3f(xz.x, e, xz.y);
  let bladeMid = vec3f(xz.x, e + 0.25 * height, xz.y);
  let sun = select(sunAt(xz), 1.0, onIsland) * treeShadowAt(bladeMid) * skyShadowAt(bladeMid);
  let far = r >= GRASS_FAR_RING;
  let idx = atomicAdd(&drawArgs[select(0u, 1u, far)].instanceCount, 1u);
  var b: Blade;
  b.p = vec4f(root, height * select(1.0, 1.15, flower));
  b.dir = vec2f(cos(a), sin(a));
  var narrow = 1.0;
  if (!onIsland) {
    // away from its tuft's root (near rings), and tall grass thin, short grass broad
    let tk2 = 1.0 - smoothstep(1.0, 3.0, f32(r));
    let out = -tuftV.zw;
    b.dir = normalize(mix(b.dir, out / max(length(out), 1e-3), 0.65 * tk2) + vec2f(1e-4, 0.0));
    narrow = clamp(1.0 / sqrt(max(gc.bioB.a * BLADE_SCALE, 0.25)), 0.7, 1.35);
  }
  b.width = GRASS_W0 * exp2(0.8 * f32(r)) * narrow * (0.8 + 0.4 * fract(hs.x * 13.1)) * select(1.0, -1.3, flower);
  b.packed = pack4x8unorm(vec4f(packedCol.rgb, sun));
  if (far) { bladesOutFar[idx] = b; } else { bladesOut[idx] = b; }
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
  let tip = vi == 2u * BLADE_SEGS;
  let t = select(f32(vi / 2u) / f32(BLADE_SEGS), 1.0, tip);
  let side = select(select(-1.0, 1.0, (vi & 1u) == 1u), 0.0, tip);
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
  let pos = root + vec3f(0.0, h * t * up, 0.0) + bend * (h * t * t) + sv * (side * 0.5 * abs(b.width) * pow(1.0 - t, 0.8));
  // Normal: the strip's face, rounded across the width.
  let dPdt = vec3f(0.0, h * up, 0.0) + bend * (2.0 * h * t);
  var n = normalize(cross(sv, dPdt));
  n = normalize(n + sv * side * 0.45);
  var o: GOut;
  o.pos = project(pos);
  o.rel = pos;
  o.n = n;
  o.packed = b.packed;
  o.t = select(t, -t - 0.001, b.width < 0.0); // (a flowering blade carries its t negated)
  return o;
}

@fragment
fn fsGrass(i: GOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  var n = normalize(i.n);
  if (dot(n, rd) > 0.0) { n = -n; }
  let c = unpack4x8unorm(i.packed);
  let base = c.rgb * c.rgb;
  var sun = c.a;
  if (sun > 0.0) { sun = min(sun, heroShadowAt(i.rel)); }
  let flower = i.t < 0.0;
  let t = abs(i.t);
  // Dark roots, sun-bleached tips; a flowering blade green below its bloom.
  var albedo = mix(base * 0.4, base * 1.35 + vec3f(0.035, 0.03, 0.0), t * t);
  if (flower) { albedo = mix(vec3f(0.035, 0.07, 0.015) * (0.6 + 0.6 * t), base, smoothstep(0.62, 0.8, t)); }
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

/**
 * Sky-island grass (with the sky realm): where a blade grows up there, and
 * how — skyGrassAt: the island's top or an arch's deck at xz (vec4f: height,
 * island (an arch a as −1 − a), wetness, ρ (on a deck: t along it); height
 * −1e9 where none, or where the land below is nearer the camera, or in the
 * water, at the rim, on a floating rock); its biome's density, height,
 * flowering share (skyGrassLook) and colours (skyGrassColour, skyFlower).
 */
const SKY_GRASS_WGSL = /* wgsl */ `
// Where a blade could grow up here at chunk-local xz, whatever the camera: an island's top or an
// arch's deck (vec4f as skyGrassAt; height −1e9 where none: no top, the water, the rim, the steep).
fn skyGrassTop(xz: vec2f) -> vec4f {
  let none = vec4f(-1e9, 0.0, 0.0, 0.0);
  let k = rlFind(xz, true);
  if (k < 0) {
    // An arch's deck (its island's grass; the arch's index is passed as −1 − a).
    let deck = rlDeckAt(xz);
    if (deck.x < -1e8) { return none; }
    return vec4f(deck.x, -1.0 - deck.y, 0.0, deck.z);
  }
  let ku = u32(k);
  let q = xz - realm.isl[ku].c.xy;
  let rho = rlRho(ku, q);
  if (rho > 0.97) { return none; }
  // Nothing in an Eyrie's streets, nor against its walls.
  let dd = rlDistrict(ku, q);
  if (dd.x < 0.25 || dd.y < 1.0) { return none; }
  // (one walk along the stream serves the water, the top and the damp here)
  let sq = rlStreamAt(ku, q);
  if (rlWaterS(ku, q, sq) > -1e8) { return none; }
  let y = rlTopS(ku, q, sq);
  // Nothing on the steep (a ravine's walls, a crag's flanks).
  if (rlInStream(ku, q) || realm.isl[ku].r.w > 0.0) {
    let sl = length(vec2f(rlTop(ku, q + vec2f(0.75, 0.0)) - y, rlTop(ku, q + vec2f(0.0, 0.75)) - y)) / 0.75;
    if (sl > 0.9) { return none; }
  }
  return vec4f(y, f32(ku), rlWetS(ku, q, sq), rho);
}
// Is the camera up among the islands (else their tops grow no grass: the land's does)?
fn skyAloft() -> bool { return realmU.morph.w != 0.0; }
// The camera's part: only where the top is nearer it than the land below (else the land's grass grows there).
fn skyGrassNear(y: f32, land: f32) -> bool {
  return skyAloft() && y > -1e8 && abs(u.camPos.y - y) <= abs(u.camPos.y - land);
}
// The biome a grass spot takes (an arch's: the island it springs from, or ends on past its middle).
fn skyGrassBiome(k: f32, t: f32) -> u32 {
  if (k < 0.0) {
    let A = realm.arch[u32(-1.0 - k)];
    return u32(select(A.e.y, A.e.z, t > 0.5));
  }
  return realm.isl[u32(k)].m.y;
}
fn skyGrassLook(k: f32, xz: vec2f, sky: vec4f) -> vec4f {
  let b = skyGrassBiome(k, sky.w);
  let G = BI_GRASS[b];
  let wet = sky.z;
  var dens = min(1.0, G.x * (1.0 + 0.4 * wet));
  // Little grows through the snow, nothing on bare crag.
  if (k >= 0.0) {
    let q = xz - realm.isl[u32(k)].c.xy;
    let fine = 0.5 + 0.5 * vnoise(q.x / 6.5, q.y / 6.5, rlSeed(u32(k)) + 22);
    dens *= (1.0 - smoothstep(0.05, 0.3, rlSnow(u32(k), q, sky.x, 1.0))) * (1.0 - smoothstep(0.1, 0.4, rlCrag(u32(k), q, fine)));
  }
  return vec4f(dens, G.y * (1.0 + 0.25 * wet), G.z * (1.0 - wet), f32(b));
}
fn skyGrassColour(k: f32, xz: vec2f, sky: vec4f) -> vec3f {
  if (k < 0.0) {
    let b = skyGrassBiome(k, sky.w);
    let patchN = 0.5 + 0.5 * vnoise(xz.x / 23.0, xz.y / 23.0, 4119);
    return mix(BI_DRY[b], BI_LUSH[b], clamp(patchN * 1.2 - 0.1, 0.0, 1.0));
  }
  let ku = u32(k);
  let b = realm.isl[ku].m.y;
  // The same tone the ground under it takes (rlGround), so blades and ground read as one sward.
  var c = rlGrassTone(ku, xz - realm.isl[ku].c.xy, sky.z) * 1.1;
  // Frost-bound grass: pale with rime.
  if (BI_SNOW[b] > 0.0) { c = mix(c, vec3f(0.55, 0.6, 0.63), 0.7); }
  return c;
}
fn skyFlower(b: u32, v: f32) -> vec3f {
  if (v < 0.4) { return BI_FLOWER0[b]; }
  if (v < 0.75) { return BI_FLOWER1[b]; }
  return BI_FLOWER2[b];
}
`;

const NO_SKY_GRASS_WGSL = /* wgsl */ `
fn skyGrassTop(xz: vec2f) -> vec4f { return vec4f(-1e9, 0.0, 0.0, 0.0); }
fn skyAloft() -> bool { return false; }
fn skyGrassNear(y: f32, land: f32) -> bool { return false; }
fn skyGrassLook(k: f32, xz: vec2f, sky: vec4f) -> vec4f { return vec4f(1.0, 1.0, 0.0, 0.0); }
fn skyGrassColour(k: f32, xz: vec2f, sky: vec4f) -> vec3f { return vec3f(0.075, 0.14, 0.032); }
fn skyFlower(b: u32, v: f32) -> vec3f { return vec3f(1.0); }
`;
