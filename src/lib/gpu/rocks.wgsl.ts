/**
 * Rocks: GPU-scattered stones on world-anchored lattices round the camera, drawn as instanced meshes.
 * ---------------------------------------------------------------------------
 * Four tiers, a lattice each (as the grass rings): pebbles and small stones, rocks, boulders, and the
 * landmarks' big ones, their cells 0.5 / 2 / 8 / 32 m, their sizes growing with them. A cell holds at most
 * one rock at a hashed place, so rocks stay put while the tiers follow the camera. `rockScatter` (compute,
 * a thread a cell) keeps a cell's rock where the biome has rocks (biomes.ts GROUND_USE) thickened by drifts
 * of stone, on slopes and on bare ground, out of the water, the ruins and the view, picks its kind by the
 * tier, sets it a little into the ground, tilts it toward the slope, and appends it to the bucket of its
 * kind and level of detail (chosen by the pixels it spans: a rock under 2.5 px is not drawn). A tier's
 * rocks sink into the ground over the last of its radius, so none pops. In the box of cells round the hero
 * the CPU owns the rocks (rocks/field.ts: what is drawn there is what is solid): the scatter leaves those cells
 * and the cells whose rock was taken, and `rockList` (a thread a rock) puts the CPU's list into the same
 * buckets — the box's rocks set on the land here, and the free ones (in the hands, flying, rolling, lying
 * where they came to rest) where they are.
 *
 * `vsRock` / `fsRock` draw a bucket (its instances by a dynamic offset into one buffer, its vertices by
 * the indirect draw's first vertex). The rock's surface is the terrain's own rock (groundDetail.wgsl.ts
 * rockMix: fractured stone, strata or black stone by the biome, at the same scale and in the same
 * world axes, so a boulder at a cliff's foot is of a piece with the cliff), over the biome's colour at its
 * place; with the dirt it sits in at its foot (the ground's soil colour), moss on its upper faces in lush
 * country, snow on them where snow lies.
 *
 * Needs, from the raster shader: u (Frame), heightAt, waterAt, gradientAt, groundCacheAt, snowAt, inRuinSite,
 * landRegions + domainWarp (worldgen.wgsl.ts), sunAt, treeShadowAt, skyShadowAt, heroShadowAt, heroAOAt, the light
 * API (keyDir, keyCol, skyAmb), surfaceFog, display, underwater, project(), the ground detail (rockMix, rockFine,
 * gOne), SNOW.
 */

import { BIOME_COUNT } from '../biomes';
import { ROCK_KINDS, ROCK_LODS, buildRockModels } from '../rocks/models';
import { ROCK_CAP, ROCK_CAST_SIZE, ROCK_MESHES, ROCK_N, ROCK_SOLID_FROM, ROCK_TIERS } from '../rocks/tiers';
import { wgslF32 as f } from './worldgen.wgsl';

let models: ReturnType<typeof buildRockModels> | null = null;
/** The rock meshes (made once). */
export function rockModels(): ReturnType<typeof buildRockModels> {
  return (models ??= buildRockModels());
}

/** shadows: the trees' shadow map exists (forest.wgsl.ts), and rocks cast into it. */
export function rocksWGSL(shadows: boolean): string {
  const m = rockModels();
  return /* wgsl */ `
const RK_TIERS: u32 = ${ROCK_TIERS.length}u;
const RK_N: u32 = ${ROCK_N}u;
const RK_LODS: u32 = ${ROCK_LODS}u;
const RK_CAP: u32 = ${ROCK_CAP}u;
// (the solid tiers stand whatever the density: rocks/field.ts makes them on the CPU, where the hero meets them)
const RK_SOLID: u32 = ${ROCK_SOLID_FROM}u;
// the parameters past the tiers': the frame's (density, cells taken, rocks listed), the CPU's box a tier, the cells taken
const RK_FRAME: u32 = ${ROCK_TIERS.length * 3}u;
const RK_ZONE0: u32 = ${ROCK_TIERS.length * 3 + 1}u;
const RK_TAKEN0: u32 = ${ROCK_TIERS.length * 4 + 1}u;
// the rockiest land (1) × a full drift (1.7) × the steepest slope's thickening (2.6)
const RK_MOST: f32 = 4.42;
const RK_MESHES: u32 = ${ROCK_MESHES}u;
const RK_CAST: f32 = ${f(ROCK_CAST_SIZE)};
const RK_KIND_TABLE = array<u32, ${ROCK_TIERS.length * 6}>(${ROCK_TIERS.flatMap((t) => t.kinds).map((k) => `${k}u`).join(', ')});
// Per kind: how far under its middle the finest mesh's flat base lies (a share of its size), for setting it on the ground.
const RK_BOTTOM = array<f32, ${ROCK_KINDS.length}>(${m.extent.map((e) => f(e.bottom)).join(', ')});

struct RockInst {
  p: vec4f,     // the mesh's origin (chunk-local), its size
  q: vec4f,     // its turn (a quaternion: tilt toward the slope, then yaw)
  s: vec4f,     // the scale of its axes (× size)
  c: vec4u,     // rock √rgb + strata; soil √rgb + snow; moss √rgb + moss; a random number
};
@group(1) @binding(30) var<storage, read> rkIn: array<RockInst>;
@group(1) @binding(31) var<storage, read_write> rkOut: array<RockInst>;
// A bucket's draw arguments: vertex count, instance count (the scatter's counter), first vertex, first instance.
@group(1) @binding(32) var<storage, read_write> rkArgs: array<atomic<u32>>;
@group(1) @binding(33) var<storage, read> rkp: array<vec4f>;
// The CPU's rocks (rocks/field.ts packRocks): the box round the hero and the free ones (lifted, thrown, rolled).
struct RockSrc {
  p: vec4f,     // chunk-local x, y (m: on its cell, the lift over the land), z; its size
  q: vec4f,     // its turn
  s: vec4f,     // the scale of its axes
  m: vec4u,     // tier, kind, a random number, 1: on its cell (set on the land here) or 0: where it is
};
@group(1) @binding(34) var<storage, read> rkSrc: array<RockSrc>;

fn rkHash(ix: i32, iz: i32, s: i32) -> vec4f {
  let h = hashU4(vec4u(hash3(ix, iz, s), hash3(ix, iz, s + 1), hash3(ix, iz, s + 2), hash3(ix, iz, s + 3)));
  return vec4f(h >> vec4u(8u)) * (1.0 / 16777216.0);
}

fn rkQuatMul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn rkRot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn rkInFrustum(c: vec3f, r: f32) -> bool {
  let v = c * u.basis;
  if (v.z < -r) { return false; }
  let tx = u.tanHalf * u.res.x / u.res.y;
  let ty = u.tanHalf;
  if (abs(v.x) - r * sqrt(1.0 + tx * tx) > v.z * tx) { return false; }
  if (abs(v.y) - r * sqrt(1.0 + ty * ty) > v.z * ty) { return false; }
  return true;
}

// How rocky the land is at world g: the biome's (GROUND_USE) blended by its regions' weights.
fn rkRockiness(g: vec2f) -> f32 {
  let r = landRegions(g, domainWarp(g.x, g.y, u.seed, 0.0), u.seed);
  var rocky = 0.0;
  for (var k = 0; k < ${BIOME_COUNT}; k++) { rocky += r.w[k] * LK_USE[k].w; }
  return rocky;
}

@compute @workgroup_size(8, 8, 1)
fn rockScatter(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= RK_N || id.y >= RK_N || id.z >= RK_TIERS) { return; }
  let t = id.z;
  let CAM = rkp[t * 3u];                       // camera's share of a cell (x, z), the cell (m), the radius (m)
  let IDX = bitcast<vec4i>(rkp[t * 3u + 1u]);  // the first cell's absolute index (x, z)
  let SZ = rkp[t * 3u + 2u];                   // size from, to; the chance; the kinds' count
  let dens = rkp[RK_FRAME].x;
  let ix = IDX.x + i32(id.x);
  let iz = IDX.y + i32(id.y);
  // (the CPU's box round the hero: its rocks come from the list, rockList)
  let Z = bitcast<vec4i>(rkp[RK_ZONE0 + t]);
  if (ix >= Z.x && iz >= Z.y && ix <= Z.z && iz <= Z.w) { return; }
  let hs = rkHash(ix, iz, u.seed + 9001 + i32(t) * 131);
  // a cell sits at a hashed place in the middle four fifths of it
  let half = f32(RK_N / 2u);
  let relXZ = (vec2f(f32(id.x), f32(id.y)) - half + 0.1 + 0.8 * hs.xy - CAM.xy) * CAM.z;
  let d = length(relXZ);
  if (d > CAM.w) { return; }
  let hs2 = rkHash(ix, iz, u.seed + 9501 + i32(t) * 131);
  let size = SZ.x * pow(SZ.y / SZ.x, hs2.x * hs2.x);
  // too small to see (under 2.5 pixels across) is not drawn
  let px = size / (max(d, 1.0) * u.footprint);
  if (px < 2.5) { return; }
  // (the most the chance can be: terrainMath-side rocks/collision.ts MOST. Over it, no rock stands here whatever the land is)
  if (hs.z > SZ.z * select(dens, 1.0, t >= RK_SOLID) * RK_MOST * select(1.0, 4.0, t == 0u)) { return; }
  // a cell whose rock was taken (lifted, rolled away) stays empty
  if (t >= RK_SOLID) {
    let nt = u32(rkp[RK_FRAME].y);
    for (var k = 0u; k < nt; k++) {
      let T = bitcast<vec4i>(rkp[RK_TAKEN0 + k]);
      if (T.x == i32(t) && T.y == ix && T.z == iz) { return; }
    }
  }
  let xz = u.camPos.xz + relXZ;
  if (inRuinSite(xz)) { return; }
  let land = heightAt(xz, vec2f(0.0), CAM.z);
  // in view
  if (!rkInFrustum(vec3f(relXZ.x, land + 0.4 * size - u.camPos.y, relXZ.y), size * 1.9)) { return; }
  if (land < waterAt(xz) + 0.1) { return; }
  // where the land is rocky, thickened by drifts of stone, on slopes and on bare ground
  let g = xz + u.origin;
  // (the ground's cover first: the pebbles' bare ground thickens them, and a second look at the most the chance can be, now with it,
  // leaves most of a lawn's cells before the biomes' regions are asked)
  let gc = groundCacheAt(xz);
  let bare = 1.0 + select(0.0, 3.0, t == 0u) * (1.0 - gc.bioA.a);
  if (hs.z > SZ.z * select(dens, 1.0, t >= RK_SOLID) * RK_MOST * bare) { return; }
  let rocky = rkRockiness(g);
  let fld = 0.5 + 0.5 * vnoise(g.x / 83.0, g.y / 83.0, u.seed + 9011);
  let drift = mix(0.15, 1.7, smoothstep(0.4, 0.8, fld));
  let grad = gradientAt(xz, max(0.4, 0.4 * size), CAM.z);
  let slope = length(grad);
  // (the solid tiers stand whatever the density: the hero meets them, as rocks/field.ts places them by the biomes alone)
  let chance = SZ.z * select(dens, 1.0, t >= RK_SOLID) * rocky * drift * bare * (1.0 + 1.6 * smoothstep(0.25, 0.75, slope)) * (1.0 - smoothstep(1.1, 1.5, slope));
  if (hs.z > chance) { return; }
  let kind = RK_KIND_TABLE[t * 6u + min(u32(hs2.y * 6.0), 5u)];
  // its level of detail by the pixels it spans
  let lod = select(select(2u, 1u, px > 18.0), 0u, px > 70.0);
  // aspect, set a little into the ground, sinking away over the last of the tier's radius
  let asp = vec3f(0.82 + 0.36 * hs.w, 0.85 + 0.3 * hs2.z, 0.82 + 0.36 * hs2.y);
  let edge = smoothstep(0.8, 1.0, d / CAM.w);
  let sink = size * asp.y * (0.14 + 0.16 * hs2.z) + edge * size * asp.y * 1.7;
  let y0 = land - sink - RK_BOTTOM[kind] * size * asp.y;
  // tilted toward the slope (two thirds of the way), then turned about its own axis
  let nrm = normalize(vec3f(-grad.x, 1.0, -grad.y));
  let tilt = normalize(mix(vec3f(0.0, 1.0, 0.0), nrm, 0.66));
  let ax = cross(vec3f(0.0, 1.0, 0.0), tilt);
  let al = length(ax);
  var q1 = vec4f(0.0, 0.0, 0.0, 1.0);
  if (al > 1e-4) {
    let cs = sqrt(0.5 * (1.0 + tilt.y));
    let sn = sqrt(max(0.5 * (1.0 - tilt.y), 0.0));
    q1 = vec4f(ax / al * sn, cs);
  }
  let yaw = hs2.w * 6.2831853;
  let q = rkQuatMul(q1, vec4f(0.0, sin(0.5 * yaw), 0.0, cos(0.5 * yaw)));
  rkEmit(xz, land, y0, size, q, asp, kind, lod, slope, hash3(ix, iz, u.seed + 9777), gc);
}

// A rock into its bucket: what it is made of and sits in (the ground's colours at its place), its kind and level of detail.
fn rkEmit(xz: vec2f, land: f32, y0: f32, size: f32, q: vec4f, asp: vec3f, kind: u32, lod: u32, slope: f32, rnd: u32, gc: GroundCache) {
  let p3 = vec3f(xz.x, land, xz.y);
  let snow = snowAt(gc, p3, slope);
  let lush = clamp(max(gc.mat.y, 0.0) * 1.2 + 0.22 * gc.bioA.a, 0.0, 1.0);
  var inst: RockInst;
  inst.p = vec4f(xz.x, y0, xz.y, size);
  inst.q = q;
  inst.s = vec4f(asp, 0.0);
  inst.c = vec4u(
    pack4x8unorm(vec4f(gc.bioC.rgb, gc.bioC.a)),
    pack4x8unorm(vec4f(gc.bioA.rgb, snow)),
    pack4x8unorm(vec4f(gc.bioB.rgb, lush)),
    rnd);
  // (the bucket: a rock too small to cast a shadow is in the first set, the others in the second)
  let b = select(0u, RK_MESHES, size >= RK_CAST) + kind * RK_LODS + lod;
  let slot = atomicAdd(&rkArgs[b * 4u + 1u], 1u);
  if (slot >= RK_CAP) { return; }
  rkOut[b * RK_CAP + slot] = inst;
}

// The CPU's rocks (a thread a rock): those of its box round the hero, set on the land here as the scatter sets them, and the
// free ones where they are (in the hands, in the air, rolling, lying where they came to rest).
@compute @workgroup_size(64, 1, 1)
fn rockList(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(rkp[RK_FRAME].z)) { return; }
  let R = rkSrc[id.x];
  let t = min(R.m.x, RK_TIERS - 1u);
  let CAM = rkp[t * 3u];
  let xz = R.p.xz;
  let relXZ = xz - u.camPos.xz;
  let d = length(relXZ);
  if (d > CAM.w) { return; }
  let size = R.p.w;
  let px = size / (max(d, 1.0) * u.footprint);
  if (px < 2.5) { return; }
  let asp = R.s.xyz;
  var land = R.p.y;
  var y0 = R.p.y;
  var slope = 0.0;
  if (R.m.w == 1u) {
    land = heightAt(xz, vec2f(0.0), CAM.z);
    let edge = smoothstep(0.8, 1.0, d / CAM.w);
    y0 = land + R.p.y - edge * size * asp.y * 1.7;
    slope = length(gradientAt(xz, max(0.4, 0.4 * size), CAM.z));
  }
  if (!rkInFrustum(vec3f(relXZ.x, y0 + 0.2 * size - u.camPos.y, relXZ.y), size * 1.9)) { return; }
  let lod = select(select(2u, 1u, px > 18.0), 0u, px > 70.0);
  rkEmit(xz, land, y0, size, R.q, asp, min(R.m.y, ${ROCK_KINDS.length - 1}u), lod, slope, R.m.z, groundCacheAt(xz));
}

${shadows ? `
// The same transform into the sun's orthographic shadow map (as forest.wgsl.ts vsTreeShadow).
@vertex
fn vsRockShadow(@builtin(instance_index) ii: u32, @location(0) pos: vec3f) -> @builtin(position) vec4f {
  if (ii >= RK_CAP) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  let R = rkIn[ii];
  let wp = rkRot(R.q, pos * R.s.xyz * R.p.w) + (R.p.xyz - u.camPos);
  let q = wp - forest.shN.xyz;
  let Rr = forest.shC.w;
  return vec4f(dot(q, forest.shR.xyz) / Rr, dot(q, forest.shU.xyz) / Rr, 0.5 + 0.5 * dot(q, forest.shF.xyz) / forest.shR.w, 1.0);
}
` : ''}
struct RkOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) c: vec4u,
  @location(3) attr: vec3f,
  // the rock's own axes (its texture is laid on it, and turns with it): where on it, which way the face looks, its turn
  @location(4) loc: vec3f,
  @location(5) nl: vec3f,
  @location(6) @interpolate(flat) q: vec4f,
};

@vertex
fn vsRock(@builtin(instance_index) ii: u32, @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) attr: vec4f) -> RkOut {
  var o: RkOut;
  if (ii >= RK_CAP) {
    // (a bucket that overflowed: the instances past its end are not there)
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    o.rel = vec3f(0.0); o.n = vec3f(0.0, 1.0, 0.0); o.c = vec4u(0u); o.attr = vec3f(0.0);
    o.loc = vec3f(0.0); o.nl = vec3f(0.0, 1.0, 0.0); o.q = vec4f(0.0, 0.0, 0.0, 1.0);
    return o;
  }
  let R = rkIn[ii];
  let sc = R.s.xyz * R.p.w;
  let wp = rkRot(R.q, pos * sc) + (R.p.xyz - u.camPos);
  o.rel = wp;
  o.pos = project(wp);
  o.n = rkRot(R.q, normalize(nrm.xyz / max(R.s.xyz, vec3f(0.2))));
  o.c = R.c;
  o.attr = vec3f(attr.xy, R.p.w);
  // (a window of the texture of its own: each rock's random number moves it over the tile)
  let rh = vec3f(f32(R.c.w & 1023u), f32((R.c.w >> 10u) & 1023u), f32((R.c.w >> 20u) & 1023u)) * (16.0 / 1024.0);
  o.loc = pos * sc + rh;
  o.nl = normalize(nrm.xyz / max(R.s.xyz, vec3f(0.2)));
  o.q = R.q;
  return o;
}

@fragment
fn fsRock(i: RkOut) -> @location(0) vec4f {
  // (derivatives for the textures' filtering, taken where the control flow is uniform)
  let ddx = dpdx(i.rel);
  let ddy = dpdy(i.rel);
  let ddxL = dpdx(i.loc);
  let ddyL = dpdy(i.loc);
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let p = u.camPos + i.rel;
  var n = normalize(i.n);
  if (dot(n, rd) > 0.0) { n = -n; }
  let c0 = unpack4x8unorm(i.c.x);
  let c1 = unpack4x8unorm(i.c.y);
  let c2 = unpack4x8unorm(i.c.z);
  let tv = f32(i.c.w & 255u) / 255.0;
  var albedo = c0.rgb * c0.rgb * (0.98 + 0.3 * tv);
  // the rock's own surface: the terrain's rock, laid on the rock in its own axes (so it stays on the rock as the rock
  // is carried, thrown and rolled), blurred and toned down: weathered stone, soft marks, no cuts
  var tone = 1.0;
  var cav = 1.0;
  var tilt = vec3f(0.0);
  // (a rock is small: its texture is gone by a hundred metres, much sooner for a small one, and the taps are not made past it)
  let dk = (1.0 - smoothstep(0.5 * DETAIL_NEAR, 0.6 * DETAIL_FAR, dist)) * (1.0 - smoothstep(14.0 * i.attr.z, 36.0 * i.attr.z, dist));
  if (!STUB_DETAIL && dk > 0.0) {
    let blur = 3.0;
    let rm = rockMix(c0.rgb, c0.a, i.loc, i.nl, ddxL * blur, ddyL * blur);
    var r = rm.r;
    let nearK = 1.0 - smoothstep(2.5, 14.0, dist);
    if (nearK > 0.02) {
      let q = rockFine(rm.l0, i.loc, i.nl, ddxL * blur, ddyL * blur);
      r = Rock(r.t * mix(1.0, q.t, 0.45 * nearK), r.a * mix(1.0, q.a, 0.25 * nearK), r.o + q.o * nearK * 2.0);
    }
    tone = mix(1.0, r.t, 0.8 * K_TONE * dk);
    cav = mix(1.0, r.a, 0.4 * K_CAV * dk);
    tilt = rkRot(i.q, r.o) * (0.35 * dk);
    n = normalize(n + tilt);
  }
  // weathering: broad soft mottling and long faint streaks (rain runs, down the rock's own up), on the rock's own axes,
  // gone with distance (a speck of rock has none to show)
  let wk = 1.0 - smoothstep(30.0 * i.attr.z, 120.0 * i.attr.z, dist);
  if (wk > 0.0) {
    let sz = 1.4 / sqrt(max(i.attr.z, 0.15));
    let ql = i.loc * sz;
    let sd = i32(i.c.w & 255u) + 41;
    let mot = vnoise(ql.x + 0.6 * ql.z, ql.y * 1.1 + 0.4 * ql.z, sd) * 0.6 + vnoise(ql.x * 2.3 - ql.z, ql.y * 2.1 + ql.z * 1.7, sd + 7) * 0.4;
    let stk = vnoise(ql.x * 4.0 + ql.z * 2.5, ql.y * 0.7 + ql.z * 0.3, sd + 19);
    tone *= 1.0 + wk * (0.13 * mot + 0.09 * stk);
    albedo = mix(albedo, albedo * vec3f(0.92, 0.9, 0.88), wk * 0.5 * smoothstep(0.1, 0.7, -stk * 0.5 - mot * 0.3 + 0.25));
  }
  albedo *= tone;
  // the dirt it sits in, climbing its foot
  let h = i.attr.y;
  let soil = c1.rgb * c1.rgb;
  albedo = mix(albedo, soil * 0.85, (1.0 - smoothstep(0.02, 0.3, h)) * 0.8);
  // moss on its upper faces in lush country
  let up = smoothstep(0.5, 0.85, n.y);
  if (c2.a * up > 0.02) {
    let m = gOne(GL_MOSS, i.loc.xz * 1.7, ddxL.xz * 1.7, ddyL.xz * 1.7);
    let mk = smoothstep(0.2, 0.5, c2.a * up * (0.5 + 0.9 * m.a) * (0.7 + 0.6 * cav));
    albedo = mix(albedo, c2.rgb * c2.rgb * 0.75 * m.t, mk);
  }
  // and snow where it lies
  albedo = mix(albedo, SNOW, c1.a * smoothstep(0.3, 0.78, n.y));
  // light: the sun by the ground's visibility of it at the rock's place (the terrain's, the trees' shadows)
  let L = keyDir();
  let ndl = max(dot(n, L), 0.0);
  var sh = 0.0;
  if (ndl > 0.0) {
    sh = sunAt(p.xz) * treeShadowAt(p) * skyShadowAt(p);
    if (sh > 0.0) { sh = min(sh, heroShadowAt(i.rel)); }
  }
  let kc = keyCol(p.y);
  let ao = mix(1.0, i.attr.x, 0.55) * cav;
  var col = albedo * (kc * ndl * sh * mix(1.0, ao, 0.4) + skyAmb(n, 0.22 + 0.28 * max(n.y, 0.0)) * ao * heroAOAt(i.rel));
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}
`;
}

