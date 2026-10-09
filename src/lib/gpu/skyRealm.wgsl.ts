/**
 * The sky realm in WGSL: the twin of src/lib/skyworld/ (island.ts, flora.ts,
 * arches.ts) evaluating the same surfaces in fp32 on what the CPU uploads
 * (skyRealm.ts), and the shaders that draw them.
 * ---------------------------------------------------------------------------
 * realm (storage, binding 10): the archipelagos and isles round the camera —
 * clusters (an archipelago, or a lone isle, with its bounds and its islands),
 * islands (shape, biome, lake), stream points, arches and falls, all
 * chunk-local. realmU (uniform, binding 11): per frame, the island shadow
 * maps' boxes, the sky rings and the polar draws' instance list.
 *
 * Drawn:
 *   polar meshes  an island's underside always, and its top unless the sky
 *                 rings draw it: rings × bearings with the rings bunched
 *                 toward the rim (its cliff), three levels of detail;
 *   sky rings     near the camera, when it is up among the islands: nested
 *                 grid rings (geometry clipmap, 1 m cells first) whose
 *                 vertices take the top of the island they fall on and are
 *                 clipped along its outline (clip distance = signed distance
 *                 inside it), so a walkable top is as fine as the land below;
 *                 polar tops are clipped out of the rings' square;
 *   arches        a deck of grass over a rock hull, along each arch's axis;
 *   lakes         a disc at the water line, blended (ice: opaque);
 *   streams       a ribbon down the channel, its water flowing to the lip;
 *   shadow map    the undersides and arches seen from the sun (orthographic,
 *                 double-buffered like the trees'), which every surface below
 *                 samples (skyShadowAt).
 * Surfaces are shaded by biome (the ground's palette, wet near water, rock on
 * the steep and at the rim, snow up high; the underside's strata, mossy or
 * iced rim, a wet streak under the fall's lip), lit by the light API with a
 * soft bounce off the cloud deck from below, and fogged through the cloud deck
 * (surfaceFog's cloud veil).
 *
 * Needs, from the raster shader: u (Frame), vnoise, the light API (sky.wgsl.ts),
 * surfaceFog, display, underwater, heroShadowAt, treeShadowAt, project().
 */

import { LIGHT, SKY_REALM as S } from '../constants';
import { SKY_BIOMES, SKY_BIOME_INFO, SKY_FLORA_COLOURS } from '../skyworld/biomes';
import { SKY_PLACE } from '../skyworld/flora';
import { CHANNEL_REACH, STREAM_DEPTH, UNDERSIDES } from '../skyworld/island';
import { wgslF32 as f } from './worldgen.wgsl';
import { DISTRICT_RANGE } from '../skyworld/eyrie/district';

export const REALM_MAX = { clusters: 48, islands: 768, points: 4096, arches: 96, falls: 64, instances: 1024 } as const;
/**
 * The realm is drawn out to this distance (m): whole within REALM_NEAR (clusters, arches, floating
 * rocks nearby, streams and lakes), beyond it only the islands and their falls (skyRealm.ts packRealm).
 */
export const REALM_NEAR = 16000;
export const REALM_FAR = 32000;
/** Bytes per record in the realm buffer (see the structs below). */
const ISL_BYTES = 14 * 16, ARCH_BYTES = 5 * 16, FALL_BYTES = 3 * 16, CLUSTER_BYTES = 2 * 16;
/** Offsets (bytes) of each array in the realm buffer, and its size. */
export const REALM_LAYOUT = (() => {
  const cl = 16;
  const isl = cl + REALM_MAX.clusters * CLUSTER_BYTES;
  const pts = isl + REALM_MAX.islands * ISL_BYTES;
  const arch = pts + REALM_MAX.points * 16;
  const fall = arch + REALM_MAX.arches * ARCH_BYTES;
  return { cl, isl, pts, arch, fall, bytes: fall + REALM_MAX.falls * FALL_BYTES, islStride: ISL_BYTES, archStride: ARCH_BYTES, fallStride: FALL_BYTES, clStride: CLUSTER_BYTES };
})();

/** Polar meshes' levels of detail: rings (bunched toward the rim) × bearings. */
export const POLAR_LODS: ReadonlyArray<readonly [number, number]> = [[8, 32], [24, 96], [64, 192]];
/**
 * And for an island a few pixels across (most of those out to REALM_FAR): each vertex works out the
 * island's surface three times (its height and normal), so this level has a quarter of LOD 0's.
 */
export const POLAR_FAR: readonly [number, number] = [4, 16];
/** Sky rings: cells per half ring, rings, the first ring's cell (m). */
export const SKY_RINGS = { k: 64, rings: 6, spacing0: 1 } as const;
/** Arch mesh: sections along it × points round its cross-section. */
export const ARCH_SEGS = 48;
export const ARCH_SIDES = 16;
/** Lake disc: rings × bearings. */
export const LAKE_RINGS = 6;
export const LAKE_SEGS = 48;
/** Stream ribbon: sections per stretch between two stream points. */
export const STREAM_SUB = 4;

/** Per-frame uniform: shadow maps (sampled, drawn), the rings, the polar instance list. */
export const REALM_FRAME_BYTES = 16 * (10 + 2 * SKY_RINGS.rings + 2 + REALM_MAX.instances / 4);

/** Island flags (SkyIsl.m.z); bits 8–10 its underside (island.ts UNDERSIDES), 11 a lagoon's tint, 12–14 its lobes, 16–20 its Eyrie's palette. */
export const RL_FLAGS = { lake: 1, stream: 2, frozen: 4, debris: 8, fall: 16 } as const;
export const RL_UNDER_SHIFT = 8, RL_TINT_BIT = 11, RL_LOBES_SHIFT = 12, RL_PALETTE_SHIFT = 16;

const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;
const arr = (type: string, xs: string[]) => `array<${type}, ${xs.length}>(${xs.join(', ')})`;
const B = SKY_BIOMES.map((b) => SKY_BIOME_INFO[b]);

export function skyRealmWGSL(clip: boolean): string {
  return /* wgsl */ `
struct SkyIsl {
  c: vec4f,   // centre x, z (chunk-local), outline radius R, top base height y
  o: vec4f,   // outline phases
  r: vec4f,   // outline wobble, hill height, hill wavelength, crags
  u: vec4f,   // summit, cliff, underside depth, district map's layer (−1: none)
  m: vec4u,   // noise seed (i32 bits), biome, flags (RL_FLAGS), stream: first point | count << 16
  l0: vec4f,  // lake: middle x, z (island-local), radius, level
  l1: vec4f,  // lake: aspect, cos, sin (major axis), depth
  l2: vec4f,  // lake: shore wobble phases; fall index (−1: none); 1 / its district map's side (m)
  sb: vec4f,  // stream bounds: middle x, z (island-local), radius (its carve included); unused
  sh: vec4f,  // the Eyries' shaping: terrace height (m), bowl, tilt x, z
  lb: array<vec4f, 4>, // extra hanging spikes: centre x, z (island-local), radius, depth
};
struct SkyArch {
  a: vec4f,   // start x, z (chunk-local), deck height there, length
  b: vec4f,   // end x, z, deck height there, sideways bow
  c: vec4f,   // direction x, z (unit), hump, deck width in the middle
  d: vec4f,   // deck width at the ends, rock under the deck in the middle and at the ends, unused
  e: vec4f,   // noise seed, biome of the island it starts on, of the one it ends on, unused
};
struct SkyFall {
  a: vec4f,   // lip x, y, z (chunk-local; y: the water's surface), width at the lip
  b: vec4f,   // direction x, z (unit), speed over the lip, reach (m fallen when it has all become mist)
  c: vec4f,   // seed, island, biome, unused
};
struct SkyCluster {
  c: vec4f,   // centre x, z (chunk-local), radius holding all of it, unused
  r: vec4u,   // first island, islands, first arch, arches
};
struct SkyWorld {
  n: vec4u,   // clusters, islands, arches, falls
  cl: array<SkyCluster, ${REALM_MAX.clusters}>,
  isl: array<SkyIsl, ${REALM_MAX.islands}>,
  pts: array<vec4f, ${REALM_MAX.points}>,  // stream points: x, z (island-local), bed, half-width
  arch: array<SkyArch, ${REALM_MAX.arches}>,
  fall: array<SkyFall, ${REALM_MAX.falls}>,
};
@group(0) @binding(10) var<storage, read> realm: SkyWorld;

struct RealmU {
  shC: vec4f,  // shadow map sampled now (drawn last frame): centre (camera-relative), half width (m)
  sR: vec4f,   // its light axes: right, up, forward (away from the light); sF.w: half depth (m)
  sU: vec4f,
  sF: vec4f,
  shN: vec4f,  // map drawn this frame: centre (camera-relative); texel (m)
  nR: vec4f,   // its axes
  nU: vec4f,
  nF: vec4f,
  ring: vec4f, // sky rings: unused x, z; cell of ring 0 (m); rings drawn (0: none)
  sq: vec4f,   // the square they cover (camera-relative min x, z, max x, z); polar tops are clipped out of it
  rings: array<vec4i, ${SKY_RINGS.rings}>,   // per ring: unused x, z; cells of ring 0 per cell; unused
  center: array<vec4f, ${SKY_RINGS.rings}>,  // per ring: its centre in its own cells from its first vertex (x, z); its first vertex, chunk-local (m)
  morph: vec4f, // morph start (cells from the ring's centre), 1 / its width, last ring; 1 when the camera is up among the islands (their grass grows)
  mask: vec4u,  // which islands' tops the rings draw: bits of islands 0–127 (four words) …
  inst: array<vec4u, ${REALM_MAX.instances / 4}>, // polar draws: island | surface << 16 (top 0, underside 1), four to a word
};
@group(0) @binding(11) var<uniform> realmU: RealmU;
// The Eyries' district maps (eyrie/district.ts): per settled island near the camera, the distance to
// its paving and to its footprints.
@group(0) @binding(33) var districtTex: texture_2d_array<f32>;
@group(0) @binding(34) var districtSamp: sampler;
@group(0) @binding(26) var realmShadow: texture_depth_2d;
@group(0) @binding(27) var realmShadowSamp: sampler_comparison;
// Its caster mask (skyRealm.ts shadowMask): a bit for each of 64 × 64 cells, set where a caster lies in reach.
@group(0) @binding(29) var<uniform> realmMask: array<vec4u, 32>;

const RL_RIM: f32 = ${f(S.rimDrop)};
const RL_STREAM_DEPTH: f32 = ${f(STREAM_DEPTH)};
const RL_DECK: f32 = ${f(LIGHT.cloudPlane)};
const RL_LAKE: u32 = ${RL_FLAGS.lake}u;
const RL_STREAM: u32 = ${RL_FLAGS.stream}u;
const RL_FROZEN: u32 = ${RL_FLAGS.frozen}u;
const RL_DEBRIS: u32 = ${RL_FLAGS.debris}u;
const BI_LUSH: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.lush)))};
const BI_DRY: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.dry)))};
const BI_SOIL: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.soil)))};
const BI_ROCK: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.rock)))};
const BI_SNOW: array<f32, ${B.length}> = ${arr('f32', B.map((b) => f(b.snow)))};
const BI_STRATA0: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.strata[0])))};
const BI_STRATA1: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.strata[1])))};
const BI_RIM: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.rimCover)))};
const BI_GRASS: array<vec4f, ${B.length}> = ${arr('vec4f', B.map((b) => `vec4f(${f(b.grass.density)}, ${f(b.grass.height)}, ${f(b.grass.flowers)}, 0.0)`))};
const BI_FLOWER0: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.grass.flower[0])))};
const BI_FLOWER1: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.grass.flower[1])))};
const BI_FLOWER2: array<vec3f, ${B.length}> = ${arr('vec3f', B.map((b) => v3(b.grass.flower[2])))};
// Far canopy per biome: the colour of its trees' leaves (first kind's palette), and how much of an island they cover.
const BI_CANOPY: array<vec4f, ${B.length}> = ${arr('vec4f', B.map((b) => {
  const pal = SKY_FLORA_COLOURS[b.trees.kinds[0].palette].leaf;
  return `vec4f(${[0, 1, 2].map((c) => f(0.5 * (pal[0][c] + pal[1][c]))).join(', ')}, ${f(Math.min(0.85, b.trees.forest * 1.4 + b.trees.lone * 0.25))})`;
}))};
const BI_FROST: u32 = ${SKY_BIOMES.indexOf('frost')}u;
const BI_MIST: u32 = ${SKY_BIOMES.indexOf('mistwood')}u;
const BI_BLOSSOM: u32 = ${SKY_BIOMES.indexOf('blossom')}u;

// ---- the island's shape (island.ts) -----------------------------------------------------------

fn rlSeed(k: u32) -> i32 { return bitcast<i32>(realm.isl[k].m.x); }

// atan2 that is 0 at the origin, as the CPU's (the GPU's is undefined there: NaN on Metal,
// which turned an island's very middle into a spire where a lake's shore curve took over).
fn rlAtan2(y: f32, x: f32) -> f32 {
  if (x == 0.0 && y == 0.0) { return 0.0; }
  return atan2(y, x);
}

fn rlOutline(k: u32, th: f32) -> f32 {
  let a = realm.isl[k].o;
  return 1.0 + realm.isl[k].r.x * (0.5 * sin(3.0 * th + a.x) + 0.3 * sin(5.0 * th + a.y) + 0.2 * sin(8.0 * th + a.z) + 0.12 * sin(13.0 * th + a.w));
}

// Normalised radius of island-local q.
fn rlRho(k: u32, q: vec2f) -> f32 {
  return length(q) / (realm.isl[k].c.z * rlOutline(k, rlAtan2(q.y, q.x)));
}

// island.ts terraced(): a height folded into level steps T high, each rising to the next over its last quarter.
fn rlTerraced(h: f32, T: f32) -> f32 {
  if (T <= 0.0) { return h; }
  let k = floor(h / T);
  return T * (k + smoothstep(0.75, 1.0, h / T - k));
}

fn rlRelief(k: u32, q: vec2f, rho: f32) -> f32 {
  let r = realm.isl[k].r;
  let wl = r.z;
  let s = rlSeed(k);
  let sh = realm.isl[k].sh;
  let n = 0.5 + 0.5 * (0.6 * vnoise(q.x / wl, q.y / wl, s) + 0.28 * vnoise(q.x / (0.45 * wl), q.y / (0.45 * wl), s + 1)
    + 0.12 * vnoise(q.x / (0.2 * wl), q.y / (0.2 * wl), s + 2));
  var h = r.y * (0.35 * (1.0 - rho * rho) + 0.65 * n);
  if (r.w > 0.0) {
    let c = 1.0 - abs(vnoise(q.x / (0.8 * wl), q.y / (0.8 * wl), s + 3));
    h += r.y * r.w * 1.1 * c * c;
  }
  let peak = realm.isl[k].u.x;
  if (peak > 0.0) {
    let pr = rho / 0.32;
    h += r.y * peak * 2.2 * exp(-pr * pr);
  }
  if (sh.y > 0.0) { h += r.y * sh.y * smoothstep(0.25, 0.85, rho); }
  h = rlTerraced(h, sh.x);
  return realm.isl[k].c.w + h * (1.0 - smoothstep(0.72, 1.0, rho)) - RL_RIM * smoothstep(0.84, 1.0, rho) + sh.z * q.x + sh.w * q.y;
}

fn rlLakeDist(k: u32, q: vec2f) -> f32 {
  let L0 = realm.isl[k].l0;
  let L1 = realm.isl[k].l1;
  let L2 = realm.isl[k].l2;
  let p = q - L0.xy;
  let a = p.x * L1.y + p.y * L1.z;
  let b = (-p.x * L1.z + p.y * L1.y) / L1.x;
  let th = rlAtan2(b, a);
  return length(vec2f(a, b)) / (L0.z * (1.0 + 0.12 * sin(3.0 * th + L2.x) + 0.07 * sin(5.0 * th + L2.y)));
}

fn rlLakeBed(k: u32, d: f32) -> f32 {
  let L0 = realm.isl[k].l0;
  let c = min(d, 1.0);
  let o = max(d - 1.5, 0.0);
  return L0.w - realm.isl[k].l1.w * pow(max(1.0 - c * c, 0.0), 0.7) + L0.z * (0.12 * max(d - 1.0, 0.0) + 6.0 * o * o);
}

fn rlInStream(k: u32, q: vec2f) -> bool {
  let b = realm.isl[k].sb;
  let d = q - b.xy;
  return (realm.isl[k].m.z & RL_STREAM) != 0u && dot(d, d) < b.z * b.z;
}

// Nearest point of the island's stream to island-local q: distance from its axis, bed, half-width, how far along (0..1).
fn rlStream(k: u32, q: vec2f) -> vec4f {
  let m = realm.isl[k].m.w;
  let first = m & 0xffffu;
  let cnt = m >> 16u;
  var best = 1e20;
  var bed = 0.0;
  var hw = 1.0;
  var t = 0.0;
  for (var i = 0u; i + 1u < cnt; i++) {
    let a = realm.pts[first + i];
    let b = realm.pts[first + i + 1u];
    let d = b.xy - a.xy;
    let l2 = dot(d, d);
    var fr = 0.0;
    if (l2 > 0.0) { fr = clamp(dot(q - a.xy, d) / l2, 0.0, 1.0); }
    let e = q - a.xy - fr * d;
    let d2 = dot(e, e);
    if (d2 < best) {
      best = d2;
      bed = a.z + fr * (b.z - a.z);
      hw = a.w + fr * (b.w - a.w);
      t = (f32(i) + fr) / f32(cnt - 1u);
    }
  }
  return vec4f(sqrt(best), bed, hw, t);
}

// The stream query at q (rlStream) where q lies in the stream's reach, else one that matches nothing
// (so rlTopS, rlWaterS and rlWetS can share a single walk along the stream).
fn rlStreamAt(k: u32, q: vec2f) -> vec4f {
  if (rlInStream(k, q)) { return rlStream(k, q); }
  return vec4f(1e9, 0.0, 1.0, 0.0);
}

const RL_CHANNEL: f32 = ${f(CHANNEL_REACH)};
fn rlChannel(bed: f32, hw: f32, s: f32) -> f32 {
  let w = s / hw;
  if (w < 1.0) { return bed + 0.9 * w * w; }
  let g = max(w - 1.6, 0.0);
  return bed + 0.9 + hw * (0.35 * (w - 1.0) + 1.6 * g * g);
}

// The top at island-local q (meaningful for ρ ≤ 1): relief, lake and stream carved in.
fn rlTop(k: u32, q: vec2f) -> f32 { return rlTopS(k, q, rlStreamAt(k, q)); }
// … with the stream query at q already made (rlStreamAt).
fn rlTopS(k: u32, q: vec2f, n: vec4f) -> f32 {
  let rho = rlRho(k, q);
  var t = rlRelief(k, q, rho);
  let fl = realm.isl[k].m.z;
  if ((fl & RL_LAKE) != 0u) {
    let d = rlLakeDist(k, q);
    if (d < 3.0) {
      t = min(t, rlLakeBed(k, d));
      if (d > 1.0) { t = max(t, realm.isl[k].l0.w + 0.05 * smoothstep(1.0, 1.06, d) - 40.0 * smoothstep(1.15, 1.45, d)); }
    }
  }
  if (n.x < RL_CHANNEL * n.z) { t = min(t, rlChannel(n.y, n.z, n.x)); }
  return t;
}

// The underside at island-local q: cliff wall, inverted mountain, notched where the stream leaves.
fn rlBottom(k: u32, q: vec2f) -> f32 {
  let rho = rlRho(k, q);
  let c = realm.isl[k].c;
  let R = c.z;
  let s = rlSeed(k);
  let u4 = realm.isl[k].u;
  let wall = u4.y * (1.0 - smoothstep(0.86, 1.0, rho));
  let v = max(0.0, 1.0 - rho / 0.86);
  var cone = 0.0;
  if (v > 0.0) { cone = u4.z * pow(v, 0.9) * (0.8 + 0.35 * (0.5 + 0.5 * vnoise(q.x / (0.22 * R), q.y / (0.22 * R), s + 5))); }
  let rock = 0.05 * R * (0.5 + 0.5 * vnoise(q.x / (0.09 * R), q.y / (0.09 * R), s + 6)) * (1.0 - smoothstep(0.8, 1.0, rho));
  // hanging spires of rock, two sizes, out nearly to the rim
  let dn = max(0.0, vnoise(q.x / (0.07 * R), q.y / (0.07 * R), s + 7));
  let dn2 = max(0.0, vnoise(q.x / (0.042 * R), q.y / (0.042 * R), s + 8));
  let drip = 0.17 * R * dn * dn * dn * (1.0 - smoothstep(0.6, 0.95, rho)) + 0.07 * R * dn2 * dn2 * dn2 * (1.0 - smoothstep(0.4, 0.97, rho));
  var b = c.w - RL_RIM - wall - cone - rock - drip;
  // The extra spikes (island.ts): inverted cones off the middle.
  let nl = (realm.isl[k].m.z >> ${RL_LOBES_SHIFT}u) & 7u;
  let lobeRim = 1.0 - smoothstep(0.85, 1.0, rho);
  for (var i = 0u; i < nl; i++) {
    let L = realm.isl[k].lb[i];
    let t = max(0.0, 1.0 - length(q - L.xy) / L.z);
    if (t > 0.0) {
      b = min(b, c.w - RL_RIM - wall - L.w * lobeRim * pow(t, 1.6) * (0.88 + 0.24 * (0.5 + 0.5 * vnoise((q.x - L.x) / (0.3 * L.z), (q.y - L.y) / (0.3 * L.z), s + 9))));
    }
  }
  let sh = realm.isl[k].sh;
  b += sh.z * q.x + sh.w * q.y;
  if (rho > 0.8 && rlInStream(k, q)) {
    let n = rlStream(k, q);
    if (n.x < RL_CHANNEL * n.z) { b = min(b, rlChannel(n.y, n.z, n.x)); }
  }
  return b;
}

// The island whose outline holds chunk-local xz, or −1 (floating rocks too unless skipDebris).
fn rlFind(xz: vec2f, skipDebris: bool) -> i32 {
  for (var c = 0u; c < realm.n.x; c++) {
    let C = realm.cl[c];
    let d = xz - C.c.xy;
    if (dot(d, d) > C.c.z * C.c.z) { continue; }
    for (var k = C.r.x; k < C.r.x + C.r.y; k++) {
      let I = realm.isl[k].c;
      let e = xz - I.xy;
      let rr = I.z * 1.2;
      if (abs(e.x) > rr || abs(e.y) > rr) { continue; }
      if (skipDebris && (realm.isl[k].m.z & RL_DEBRIS) != 0u) { continue; }
      if (rlRho(k, e) <= 1.0) { return i32(k); }
    }
  }
  return -1;
}

// Distances (m) from island-local q to island k's paving and to its footprints (DISTRICT_RANGE where it has no map).
fn rlDistrict(k: u32, q: vec2f) -> vec2f {
  let layer = realm.isl[k].u.w;
  if (layer < 0.0) { return vec2f(${f(DISTRICT_RANGE)}); }
  let v = textureSampleLevel(districtTex, districtSamp, q * realm.isl[k].l2.w + 0.5, i32(layer), 0.0).rg;
  return (v - 0.5) * ${f(2 * DISTRICT_RANGE)};
}

// Is island k's top drawn by the sky rings this frame?
fn rlRinged(k: u32) -> bool {
  if (k >= 128u) { return false; }
  return (realmU.mask[k >> 5u] & (1u << (k & 31u))) != 0u;
}

// For a sky-ring vertex: the ringed island whose outline it lies deepest inside (or nearest outside),
// and how far (m, signed: + inside). (−1, −1e4) when none is within reach.
fn rlNearest(xz: vec2f) -> vec2f {
  var best = -1e4;
  var bk = -1.0;
  for (var c = 0u; c < realm.n.x; c++) {
    let C = realm.cl[c];
    let d = xz - C.c.xy;
    let cr = C.c.z + 80.0;
    if (dot(d, d) > cr * cr) { continue; }
    for (var k = C.r.x; k < C.r.x + C.r.y; k++) {
      if (!rlRinged(k)) { continue; }
      let I = realm.isl[k].c;
      let e = xz - I.xy;
      let rr = I.z * 1.2 + 80.0;
      if (abs(e.x) > rr || abs(e.y) > rr) { continue; }
      let sd = I.z * rlOutline(k, rlAtan2(e.y, e.x)) - length(e);
      if (sd > best) { best = sd; bk = f32(k); }
    }
  }
  return vec2f(bk, best);
}

// The arch deck over chunk-local xz (vec4f: deck height (−1e9: none), arch, t along it, how far across 0..1).
fn rlDeckAt(xz: vec2f) -> vec4f {
  for (var c = 0u; c < realm.n.x; c++) {
    let C = realm.cl[c];
    let d = xz - C.c.xy;
    if (C.r.w == 0u || dot(d, d) > C.c.z * C.c.z) { continue; }
    for (var a = C.r.z; a < C.r.z + C.r.w; a++) {
      let A = realm.arch[a];
      let p = xz - A.a.xy;
      let t = dot(p, A.c.xy) / A.a.w;
      if (t < 0.0 || t > 1.0) { continue; }
      let s = -p.x * A.c.y + p.y * A.c.x - A.b.w * sin(3.14159265 * t);
      let w = A.d.x + (A.c.w - A.d.x) * archSpan(t);
      let across = abs(2.0 * s / w);
      if (across > 0.9) { continue; }
      let y = A.a.z + (A.b.z - A.a.z) * t + A.c.z * sin(3.14159265 * t) + 0.3 * (1.0 - across * across);
      return vec4f(y, f32(a), t, across);
    }
  }
  return vec4f(-1e9, 0.0, 0.0, 0.0);
}

// The water's surface over island-local q (a lake or its stream), else −1e9.
fn rlWater(k: u32, q: vec2f) -> f32 { return rlWaterS(k, q, rlStreamAt(k, q)); }
fn rlWaterS(k: u32, q: vec2f, n: vec4f) -> f32 {
  let fl = realm.isl[k].m.z;
  if ((fl & RL_LAKE) != 0u && (fl & RL_FROZEN) == 0u && rlLakeDist(k, q) < 1.02) { return realm.isl[k].l0.w; }
  if (n.x < 0.75 * n.z) { return n.y + RL_STREAM_DEPTH; }
  return -1e9;
}

// How near the water a point is (1 at its edge, 0 beyond ~2 lake radii / 6 half-widths): damp ground.
fn rlWet(k: u32, q: vec2f) -> f32 { return rlWetS(k, q, rlStreamAt(k, q)); }
fn rlWetS(k: u32, q: vec2f, n: vec4f) -> f32 {
  var w = 0.0;
  if ((realm.isl[k].m.z & RL_LAKE) != 0u) { w = 1.0 - smoothstep(1.0, 2.0, rlLakeDist(k, q)); }
  return max(w, 1.0 - smoothstep(1.0, 6.0, n.x / n.z));
}

// ---- shadows of the realm (its undersides and arches, seen from the sun) ------------------------

fn skyShadowAt(p: vec3f) -> f32 {
  if (realmU.shC.w <= 0.0) { return 1.0; }
  let q = p - u.camPos - realmU.shC.xyz;
  let W = realmU.shC.w;
  let ls = vec3f(dot(q, realmU.sR.xyz), dot(q, realmU.sU.xyz), dot(q, realmU.sF.xyz));
  let uv = vec2f(0.5 + 0.5 * ls.x / W, 0.5 - 0.5 * ls.y / W);
  if (max(abs(uv.x - 0.5), abs(uv.y - 0.5)) >= 0.497) { return 1.0; }
  // Nothing casts a shadow anywhere near: skip the taps.
  let mc = vec2u(clamp(uv * 64.0, vec2f(0.0), vec2f(63.0)));
  let bit = mc.y * 64u + mc.x;
  if ((realmMask[bit >> 7u][(bit >> 5u) & 3u] & (1u << (bit & 31u))) == 0u) { return 1.0; }
  let z = 0.5 + 0.5 * ls.z / realmU.sF.w - 0.0002;
  if (z <= 0.0 || z >= 1.0) { return 1.0; }
  // Four filtered taps a texel apart (a soft edge, as the sun's disc gives up here).
  let o = 0.7 / 2048.0;
  let v = textureSampleCompareLevel(realmShadow, realmShadowSamp, uv + vec2f(o, o * 0.4), z)
        + textureSampleCompareLevel(realmShadow, realmShadowSamp, uv + vec2f(-o * 0.4, o), z)
        + textureSampleCompareLevel(realmShadow, realmShadowSamp, uv + vec2f(-o, -o * 0.4), z)
        + textureSampleCompareLevel(realmShadow, realmShadowSamp, uv + vec2f(o * 0.4, -o), z);
  return mix(0.1, 1.0, 0.25 * v);
}

// ---- shading -------------------------------------------------------------------------------------

// How much snow lies at island-local q (height y, the ground facing up by ny): frost-bound isles, the higher the more.
fn rlSnow(k: u32, q: vec2f, y: f32, ny: f32) -> f32 {
  let b = realm.isl[k].m.y;
  if (BI_SNOW[b] <= 0.0) { return 0.0; }
  let s = rlSeed(k);
  let patchN = 0.5 + 0.5 * vnoise(q.x / 37.0, q.y / 37.0, s + 21);
  let fine = 0.5 + 0.5 * vnoise(q.x / 6.5, q.y / 6.5, s + 22);
  return BI_SNOW[b] * smoothstep(0.55, 0.8, ny) * smoothstep(0.3, 0.6, patchN * 0.6 + 0.4 * fine + 0.2 * smoothstep(2500.0, 2800.0, y));
}

// The grass's colour up here (what its blades average, so a meadow reads as one surface where they thin
// out): the biome's lush and dry greens by patch and damp, a little of the land grass's vivid green.
fn rlGrassTone(k: u32, q: vec2f, wet: f32) -> vec3f {
  let b = realm.isl[k].m.y;
  let patchN = 0.5 + 0.5 * vnoise(q.x / 37.0, q.y / 37.0, rlSeed(k) + 21);
  let own = mix(BI_DRY[b], BI_LUSH[b], clamp(patchN * 1.2 - 0.15 + 0.6 * wet, 0.0, 1.0));
  return mix(own, vec3f(0.1, 0.18, 0.045), 0.3);
}

// Where a crag breaks through the turf (0–1): the ridged knobs' tops.
fn rlCrag(k: u32, q: vec2f, fine: f32) -> f32 {
  let r4 = realm.isl[k].r;
  if (r4.w <= 0.0) { return 0.0; }
  let c = 1.0 - abs(vnoise(q.x / (0.8 * r4.z), q.y / (0.8 * r4.z), rlSeed(k) + 3));
  return smoothstep(0.8, 0.88, c + 0.12 * (fine - 0.5)) * min(r4.w, 1.0);
}

// Far off, an island's trees (beyond the forest's drawing range) are painted onto its top: the canopy
// where its forest grows (flora.ts's stand field and the biome's cover), clumped like crowns.
fn rlCanopy(k: u32, q: vec2f, dist: f32) -> vec4f {
  let far = smoothstep(700.0, 1000.0, dist);
  if (far <= 0.0) { return vec4f(0.0); }
  let b = realm.isl[k].m.y;
  let s = rlSeed(k);
  let field = 0.5 + 0.5 * vnoise(q.x / ${f(SKY_PLACE.field)}, q.y / ${f(SKY_PLACE.field)}, s + 11);
  let cover = BI_CANOPY[b].w * smoothstep(0.45, 0.7, field) + 0.35 * BI_CANOPY[b].w;
  let crowns = smoothstep(0.35, 0.65, 0.5 + 0.5 * vnoise(q.x / 9.0, q.y / 9.0, s + 17));
  let a = clamp(cover * (0.6 + 0.8 * crowns), 0.0, 0.9) * far * (1.0 - smoothstep(0.88, 0.95, rlRho(k, q)));
  return vec4f(BI_CANOPY[b].rgb * (0.7 + 0.5 * crowns), a);
}

// Bench only (pipeline-overridable; the game never sets it): the tops' ground as a constant colour.
override STUB_RLGROUND: bool = false;

// The top's ground at island-local q with normal n: the biome's grass (lush or dry, damp near water),
// soil and rock on the steep and at the rim, a beach by the lake, snow up high.
fn rlGround(k: u32, q: vec2f, n: vec3f, y: f32) -> vec3f {
  if (STUB_RLGROUND) { return vec3f(0.1, 0.16, 0.05); }
  let b = realm.isl[k].m.y;
  let s = rlSeed(k);
  let rho = rlRho(k, q);
  let wet = rlWet(k, q);
  let fine = 0.5 + 0.5 * vnoise(q.x / 6.5, q.y / 6.5, s + 22);
  var col = rlGrassTone(k, q, wet) * 0.72 * (0.9 + 0.2 * fine);
  let steep = 1.0 - n.y;
  col = mix(col, BI_SOIL[b], smoothstep(0.12, 0.3, steep) * 0.8);
  // Crags break through the turf on the knobs' tops.
  col = mix(col, BI_ROCK[b] * (0.85 + 0.3 * fine), max(max(smoothstep(0.3, 0.55, steep), smoothstep(0.955, 0.995, rho)), rlCrag(k, q, fine)));
  // Wet sand and mud where the water laps.
  if ((realm.isl[k].m.z & RL_LAKE) != 0u) {
    let d = rlLakeDist(k, q);
    col = mix(col, mix(BI_SOIL[b], vec3f(0.2, 0.18, 0.13), 0.5), (1.0 - smoothstep(1.0, 1.12, d)) * 0.85);
  }
  // Snow lies on the flatter ground of the frost-bound isles (the higher, the more), rippled by
  // the wind (long low drifts across it) and bluer in their hollows.
  let snow = rlSnow(k, q, y, n.y);
  if (snow > 0.0) {
    let w = vec2f(0.87, 0.49);
    let ripple = 0.5 + 0.5 * sin(dot(q, w) * 0.9 + 3.0 * vnoise(q.x / 9.0, q.y / 9.0, s + 23));
    let drift = 0.5 + 0.5 * vnoise(q.x / 24.0, q.y / 24.0, s + 24);
    let snowCol = mix(vec3f(0.66, 0.73, 0.84), vec3f(0.8, 0.84, 0.9), 0.55 * ripple + 0.45 * drift) * (0.95 + 0.08 * fine);
    col = mix(col, snowCol, snow);
  }
  // An Eyrie's streets and plazas: setts and flagstones, worn, a kerb at their edges; bare earth
  // round its walls.
  let dd = rlDistrict(k, q);
  if (dd.x < 0.6 || dd.y < 0.8) {
    let pal = (realm.isl[k].m.z >> ${RL_PALETTE_SHIFT}u) & 31u;
    let g = q * 1.0;
    let sq = g / vec2f(0.9, 0.6);
    let row = floor(sq.y);
    let sx = sq.x + 0.5 * (row - 2.0 * floor(row * 0.5));
    let cell = vec2f(floor(sx), row);
    let fq = abs(fract(vec2f(sx, sq.y)) - 0.5);
    let joint = 1.0 - smoothstep(0.38, 0.47, max(fq.x, fq.y * 1.2));
    let tint = f32(hash3(i32(cell.x), i32(cell.y), 3201) & 255u) / 255.0;
    var pave = PL_PAVING[pal] * (0.8 + 0.3 * tint) * (0.62 + 0.38 * joint) * (0.9 + 0.2 * fine);
    let kerb = smoothstep(-0.45, -0.2, dd.x) * (1.0 - smoothstep(-0.05, 0.1, dd.x));
    pave = mix(pave, PL_TRIM[pal] * 0.8, kerb);
    col = mix(col, pave, 1.0 - smoothstep(-0.05, 0.15, dd.x));
    // trodden earth round the footprints
    col = mix(col, mix(BI_SOIL[b], PL_PAVING[pal] * 0.6, 0.4), (1.0 - smoothstep(0.0, 0.8, dd.y)) * smoothstep(-0.05, 0.15, dd.x) * 0.85);
  }
  return col;
}

// The underside's rock at world height y, island-local q: strata, darker toward the tip, its rim
// covered (moss, bare stone, ice), a wet streak under the fall's lip.
fn rlRock(k: u32, q: vec2f, y: f32, rho: f32) -> vec3f {
  let b = realm.isl[k].m.y;
  let s = rlSeed(k);
  let warp = 1.6 * vnoise(q.x / 45.0, q.y / 45.0, s + 31);
  let band = y / 11.0 + warp;
  let tone = 0.5 + 0.5 * vnoise(floor(band) * 3.1, 0.5, s + 33);
  let tone2 = 0.5 + 0.5 * vnoise(floor(band) * 3.1 + 7.3, 0.5, s + 33);
  var col = mix(BI_STRATA0[b], BI_STRATA1[b], mix(tone, tone2, smoothstep(0.6, 1.0, fract(band)))) * (1.0 - 0.12 * smoothstep(0.9, 1.0, fract(band)));
  // Weathering runs down the rock: fine vertical flutes and dark streaks below the rim.
  let th = rlAtan2(q.y, q.x) * realm.isl[k].c.z;
  let flute = vnoise(th / 3.5, y / 40.0, s + 35);
  let streak = smoothstep(0.2, 0.75, vnoise(th / 11.0, y / 160.0, s + 37));
  col *= (0.86 + 0.18 * flute) * (1.0 - 0.28 * streak * smoothstep(0.4, 0.95, rho));
  col *= mix(0.62, 1.0, smoothstep(0.0, 0.7, rho));
  let top = realm.isl[k].c.w - RL_RIM;
  col = mix(col, BI_RIM[b], smoothstep(0.9, 0.99, rho) * (1.0 - smoothstep(4.0, 22.0, top - y)) * 0.8);
  let fi = realm.isl[k].l2.z;
  if (fi >= 0.0) {
    let F = realm.fall[u32(fi)];
    let lip = F.a.xz - realm.isl[k].c.xy;
    let side = length(q - lip * (length(q) / max(length(lip), 1.0)));
    let streak = (1.0 - smoothstep(0.6 * F.a.w, 1.6 * F.a.w, side)) * smoothstep(0.75, 0.95, rho);
    col *= 1.0 - 0.45 * streak;
  }
  // What the underside is made of (island.ts UNDERSIDES).
  let under = (realm.isl[k].m.z >> ${RL_UNDER_SHIFT}u) & 7u;
  if (under == ${UNDERSIDES.indexOf('crystal')}u) {
    col = mix(col, vec3f(0.16, 0.3, 0.48), 0.7 * rlVein(k, q, y));
  } else if (under == ${UNDERSIDES.indexOf('roots')}u) {
    // living roots wrapped round the rock, hanging in strands
    let strand = 0.5 + 0.5 * sin(th * 1.9 + 3.0 * vnoise(th / 5.0, y / 28.0, s + 43));
    let rootM = smoothstep(0.15, 0.6, rho) * smoothstep(0.35, 0.65, 0.5 + 0.5 * vnoise(th / 9.0, y / 60.0, s + 45));
    col = mix(col, vec3f(0.085, 0.06, 0.04) * (0.65 + 0.7 * strand), 0.85 * rootM);
  } else if (under == ${UNDERSIDES.indexOf('basalt')}u) {
    // columns of black basalt, their joints darker
    let colJ = 1.0 - smoothstep(0.0, 0.18, abs(fract(th / 2.4) - 0.5) * 2.4 - 0.95);
    col = mix(col, vec3f(0.045, 0.04, 0.04), 0.75) * (1.0 - 0.35 * colJ);
  } else if (under == ${UNDERSIDES.indexOf('ice')}u) {
    let glaze = smoothstep(0.3, 0.8, 0.5 + 0.5 * vnoise(th / 8.0, y / 22.0, s + 47));
    col = mix(col, vec3f(0.42, 0.55, 0.68), 0.35 + 0.35 * glaze);
  }
  return col;
}

// Skystone veins in an underside (crystal), or the cracks lava glows through (basalt): 0 … 1.
fn rlVein(k: u32, q: vec2f, y: f32) -> f32 {
  let th = rlAtan2(q.y, q.x) * realm.isl[k].c.z;
  return smoothstep(0.86, 0.97, 1.0 - abs(vnoise(th / 7.0, y / 11.0, rlSeed(k) + 41)));
}

// Light an underside gives off: skystone veins glowing faintly blue (brighter by night), lava cracks in basalt.
fn rlRockEmit(k: u32, q: vec2f, y: f32) -> vec3f {
  let under = (realm.isl[k].m.z >> ${RL_UNDER_SHIFT}u) & 7u;
  if (under == ${UNDERSIDES.indexOf('crystal')}u) { return vec3f(0.3, 0.6, 1.0) * rlVein(k, q, y) * (0.08 + 0.9 * lampLevel()); }
  if (under == ${UNDERSIDES.indexOf('basalt')}u) {
    let crack = rlVein(k, q, y * 0.7) * smoothstep(0.4, 0.7, 0.5 + 0.5 * vnoise(q.x / 30.0, q.y / 30.0 - y / 40.0, rlSeed(k) + 49));
    return vec3f(1.0, 0.3, 0.05) * crack * 1.6;
  }
  return vec3f(0.0);
}

// Light on a sky surface: the key light (shadowed by the realm, the trees, the hero), the sky's fill,
// the land's bounce from below, and above the cloud deck its brighter bounce.
fn rlLight(albedo: vec3f, p: vec3f, n: vec3f, rel: vec3f, open: f32) -> vec3f {
  var sun = skyShadowAt(p) * treeShadowAt(p);
  if (sun > 0.0) { sun = min(sun, heroShadowAt(rel)); }
  let L = keyDir();
  let ramp = smoothstep(-0.12, 0.45, dot(n, L));
  let down = max(-n.y, 0.0);
  let deck = smoothstep(RL_DECK - 50.0, RL_DECK + 300.0, p.y) * max(L.y, 0.0) * 0.1;
  let amb = skyAmb(n, (0.3 + 0.28 * max(n.y, 0.0)) * open) + (bounceCol(vec3f(0.06, 0.08, 0.04)) + keyCol(RL_DECK) * deck) * down;
  return albedo * (keyCol(p.y) * 0.62 * ramp * sun + amb + lanternLight(p));
}

// The realm is drawn out to REALM_FAR (islands whose nearest edge lies that close): over its last
// stretch they, their falls and their mist sink into the haze (this much more optical depth, or
// this much less of a fall's cover), so none appears out of clear air as the camera comes on.
fn rlReachOd(rel: vec3f) -> f32 { return 12.0 * smoothstep(${f(0.82 * REALM_FAR)}, ${f(REALM_FAR)}, length(rel.xz)); }
fn rlReachFade(rel: vec3f) -> f32 { return 1.0 - smoothstep(${f(0.82 * REALM_FAR)}, ${f(REALM_FAR)}, length(rel.xz)); }

fn rlShade(albedo: vec3f, p: vec3f, n: vec3f, rel: vec3f, open: f32) -> vec4f {
  let dist = length(rel);
  let rd = rel / dist;
  return display(surfaceFogOd(rlLight(albedo, p, n, rel, open), dist, p.y, rd, fogOd(dist, rd) + rlReachOd(rel)));
}

// … with light of its own (glowing veins, lava).
fn rlShadeE(albedo: vec3f, emit: vec3f, p: vec3f, n: vec3f, rel: vec3f, open: f32) -> vec4f {
  let dist = length(rel);
  let rd = rel / dist;
  return display(surfaceFogOd(rlLight(albedo, p, n, rel, open) + glow(emit), dist, p.y, rd, fogOd(dist, rd) + rlReachOd(rel)));
}

// A sky surface at rel behind fog too thick to see through: the fog alone (what rlShade would give,
// without working out the surface), else a colour with alpha −1.
fn rlFogged(rel: vec3f) -> vec4f {
  let dist = length(rel);
  let rd = rel / dist;
  let od = fogOd(dist, rd) + rlReachOd(rel);
  if (od > FOG_OPAQUE) { return display(surfaceFogOd(vec3f(0.0), dist, u.camPos.y + rel.y, rd, od)); }
  return vec4f(0.0, 0.0, 0.0, -1.0);
}

// ---- polar meshes: undersides, and the tops the rings do not draw -------------------------------

struct RlOut {
  @builtin(position) pos: vec4f,
  ${clip ? '@builtin(clip_distances) clip: array<f32, 1>,' : '@location(4) clipD: f32,'}
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) rho: f32,
  @location(3) @interpolate(flat) kb: u32,  // island · 2 + underside
};
struct RlIn {
  @builtin(position) pos: vec4f,
  ${clip ? '' : '@location(4) clipD: f32,'}
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) rho: f32,
  @location(3) @interpolate(flat) kb: u32,
};

// Normalised radius of polar ring i of n (bunched toward the rim, where the cliff is).
fn rlRingRho(i: u32, n: u32) -> f32 {
  let t = 1.0 - f32(i) / f32(n);
  return 1.0 - pow(t, 1.7);
}

fn rlPolarAt(k: u32, bottom: bool, q: vec2f) -> f32 {
  return select(rlTop(k, q), rlBottom(k, q), bottom);
}

// Polar draw: instance (in realmU.inst from firstInstance) → island and surface; vertex → ring i, bearing j.
fn rlPolar(vi: u32, ii: u32, nRings: u32, nSegs: u32) -> RlOut {
  let w = realmU.inst[ii >> 2u][ii & 3u];
  let k = w & 0xffffu;
  let surf = w >> 16u; // 0 top, 1 underside, 2 top clipped out of the sky rings' square
  let bottom = surf == 1u;
  let i = vi / nSegs;
  let j = vi % nSegs;
  let rho = rlRingRho(i, nRings);
  let th = f32(j) * (6.2831853 / f32(nSegs));
  let c = realm.isl[k].c;
  let q = (c.z * rlOutline(k, th) * rho) * vec2f(cos(th), sin(th));
  let y = rlPolarAt(k, bottom, q);
  let e = max(0.5, 0.0015 * c.z);
  let gx = (rlPolarAt(k, bottom, q + vec2f(e, 0.0)) - y) / e;
  let gz = (rlPolarAt(k, bottom, q + vec2f(0.0, e)) - y) / e;
  var n = normalize(vec3f(-gx, 1.0, -gz));
  if (bottom) { n = normalize(vec3f(gx, -1.0, gz)); }
  let xz = c.xy + q;
  let rel = vec3f(xz.x, y, xz.y) - u.camPos;
  var o: RlOut;
  o.pos = project(rel);
  o.rel = rel;
  o.n = n;
  o.rho = rho;
  o.kb = k * 2u + select(0u, 1u, bottom);
  // A top the rings draw near the camera is drawn here only outside their square.
  var cd = 1.0;
  if (surf == 2u) {
    let sq = realmU.sq;
    cd = max(max(sq.x - rel.x, rel.x - sq.z), max(sq.y - rel.z, rel.z - sq.w));
  }
  ${clip ? 'o.clip[0] = cd;' : 'o.clipD = cd;'}
  return o;
}

@vertex fn vsRealmPolar0(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RlOut { return rlPolar(vi, ii, ${POLAR_LODS[0][0]}u, ${POLAR_LODS[0][1]}u); }
@vertex fn vsRealmPolar1(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RlOut { return rlPolar(vi, ii, ${POLAR_LODS[1][0]}u, ${POLAR_LODS[1][1]}u); }
@vertex fn vsRealmPolar2(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RlOut { return rlPolar(vi, ii, ${POLAR_LODS[2][0]}u, ${POLAR_LODS[2][1]}u); }
@vertex fn vsRealmPolarFar(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RlOut { return rlPolar(vi, ii, ${POLAR_FAR[0]}u, ${POLAR_FAR[1]}u); }

@fragment
fn fsRealm(i: RlIn) -> @location(0) vec4f {
  ${clip ? '' : 'if (i.clipD < 0.0) { discard; }'}
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  let k = i.kb / 2u;
  let p = u.camPos + i.rel;
  let q = p.xz - realm.isl[k].c.xy;
  let n = normalize(i.n);
  if ((i.kb & 1u) == 0u) {
    let cn = rlCanopy(k, q, dist);
    return rlShade(mix(rlGround(k, q, n, p.y), cn.rgb, cn.a), p, n, i.rel, 1.0);
  }
  return rlShadeE(rlRock(k, q, p.y, i.rho), rlRockEmit(k, q, p.y), p, n, i.rel, 0.85);
}

// ---- sky rings: the tops near the camera, as fine as the land ------------------------------------

struct RingOut {
  @builtin(position) pos: vec4f,
  ${clip ? '@builtin(clip_distances) clip: array<f32, 1>,' : '@location(3) clipD: f32,'}
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) k: u32,
};
struct RingIn {
  @builtin(position) pos: vec4f,
  ${clip ? '' : '@location(3) clipD: f32,'}
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) k: u32,
};

// The rings' vertices are cached (skyRealm.ts runs realmRingGen for a ring when it moves, or when
// the realm or the islands the rings draw change): per vertex its top's height, its signed distance
// inside the outline (the clip distance), its normal (x, z) and island. The vertex shader only morphs.
struct RealmRingJob {
  corner: vec2f, // the ring's first vertex, chunk-local (whole metres)
  step: f32,     // metres between its vertices
  ring: f32,
};
@group(1) @binding(30) var<uniform> realmRingJob: RealmRingJob;
@group(1) @binding(31) var<storage, read_write> realmRingOut: array<vec4u>;
@group(1) @binding(32) var<storage, read> realmRingCache: array<vec4u>;
const RING_SIDE_S: u32 = ${2 * SKY_RINGS.k + 1}u;
const RING_VERTS: u32 = ${(2 * SKY_RINGS.k + 1) ** 2}u;

@compute @workgroup_size(64)
fn realmRingGen(@builtin(global_invocation_id) id: vec3u) {
  let vi = id.x;
  if (vi >= RING_VERTS) { return; }
  let ci = vec2f(f32(vi % RING_SIDE_S), f32(vi / RING_SIDE_S));
  let xz = realmRingJob.corner + ci * realmRingJob.step;
  let hit = rlNearest(xz);
  var y = 0.0;
  var n = vec3f(0.0, 1.0, 0.0);
  var k = 0u;
  if (hit.x >= 0.0) {
    k = u32(hit.x);
    let q = xz - realm.isl[k].c.xy;
    y = rlTop(k, q);
    let e = 0.35 * realmRingJob.step;
    let gx = (rlTop(k, q + vec2f(e, 0.0)) - y) / e;
    let gz = (rlTop(k, q + vec2f(0.0, e)) - y) / e;
    n = normalize(vec3f(-gx, 1.0, -gz));
  }
  realmRingOut[u32(realmRingJob.ring) * RING_VERTS + vi] = vec4u(bitcast<u32>(y), bitcast<u32>(hit.y), pack2x16snorm(n.xz), k);
}

@vertex
fn vsRealmRing(@builtin(vertex_index) vi: u32, @builtin(instance_index) ring: u32) -> RingOut {
  let R = realmU.rings[ring];
  let step = f32(R.z) * realmU.ring.z;
  let ci = vec2i(i32(vi % RING_SIDE_S), i32(vi / RING_SIDE_S));
  let C = realmU.center[ring];
  // (morph as the land's rings: odd vertices slide onto the coarser ring's lattice near the outer edge)
  let dCells = max(abs(f32(ci.x) - C.x), abs(f32(ci.y) - C.y));
  var mu = clamp((dCells - realmU.morph.x) * realmU.morph.y, 0.0, 1.0);
  if (f32(ring) >= realmU.morph.z) { mu = 0.0; }
  let odd = vec2i(ci.x & 1, ci.y & 1);
  let base = ring * RING_VERTS;
  let a = realmRingCache[base + vi];
  var y = bitcast<f32>(a.x);
  var d = bitcast<f32>(a.y);
  var nxz = unpack2x16snorm(a.z);
  if (mu > 0.0 && (odd.x | odd.y) != 0) {
    let t = realmRingCache[base + u32((ci.y - odd.y) * i32(RING_SIDE_S) + ci.x - odd.x)];
    y = mix(y, bitcast<f32>(t.x), mu);
    d = mix(d, bitcast<f32>(t.y), mu);
    nxz = mix(nxz, unpack2x16snorm(t.z), mu);
  }
  let xz = C.zw + (vec2f(ci) - vec2f(odd) * mu) * step;
  let rel = vec3f(xz.x - u.camPos.x, y - u.camPos.y, xz.y - u.camPos.z);
  var o: RingOut;
  o.pos = project(rel);
  o.rel = rel;
  o.n = vec3f(nxz.x, sqrt(max(1.0 - dot(nxz, nxz), 0.0)), nxz.y);
  o.k = a.w;
  ${clip ? 'o.clip[0] = d;' : 'o.clipD = d;'}
  return o;
}

@fragment
fn fsRealmRing(i: RingIn) -> @location(0) vec4f {
  ${clip ? '' : 'if (i.clipD < 0.0) { discard; }'}
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  let p = u.camPos + i.rel;
  let q = p.xz - realm.isl[i.k].c.xy;
  let n = normalize(i.n);
  let cn = rlCanopy(i.k, q, dist);
  return rlShade(mix(rlGround(i.k, q, n, p.y), cn.rgb, cn.a), p, n, i.rel, 1.0);
}

// ---- arches -----------------------------------------------------------------------------------------

fn archT(A: SkyArch, t: f32) -> vec3f {
  // centre of the deck at t (chunk-local x, deck height, z)
  let bow = A.b.w * sin(3.14159265 * t);
  let xz = A.a.xy + (A.b.xy - A.a.xy) * t + vec2f(-A.c.y, A.c.x) * bow;
  let y = A.a.z + (A.b.z - A.a.z) * t + A.c.z * sin(3.14159265 * t);
  return vec3f(xz.x, y, xz.y);
}
fn archSpan(t: f32) -> f32 { return sqrt(max(0.0, sin(3.14159265 * t))); }

// Point j of the cross-section at t: the crowned deck across its width (j 0..6, left to right),
// then the rock hull from its right edge down under it and up to the left (j 7..15).
fn archPoint(a: u32, t: f32, j: u32) -> vec3f {
  let A = realm.arch[a];
  let w = A.d.x + (A.c.w - A.d.x) * archSpan(t);
  let th = A.d.z + (A.d.y - A.d.z) * archSpan(t);
  let c = archT(A, t);
  let side = vec3f(-A.c.y, 0.0, A.c.x);
  var s: f32;
  var y: f32;
  if (j <= 6u) {
    s = -0.5 * w + w * f32(j) / 6.0;
    let uu = 2.0 * s / w;
    y = c.y + 0.3 * (1.0 - uu * uu);
  } else {
    let ph = 3.14159265 * f32(j - 6u) / 10.0;
    s = 0.5 * w * cos(ph) * (1.0 + 0.22 * sin(ph));
    y = c.y - th * pow(sin(ph), 0.8) - 0.15;
    // Rough rock: bulges along the span and round it, a ragged keel of hanging spurs underneath.
    let g = vec2f(t * A.a.w, f32(j));
    let sd = i32(A.e.x);
    let lump = 0.5 + 0.5 * vnoise(g.x / 11.0, g.y * 0.6, sd);
    let spur = max(0.0, vnoise(g.x / 6.0, 0.37, sd + 2));
    y -= th * (0.22 * lump + 0.9 * spur * spur * spur * smoothstep(0.6, 1.0, sin(ph))) * sin(ph);
    s *= 1.0 + 0.16 * vnoise(g.x / 8.0, g.y * 0.9, sd + 1);
  }
  return vec3f(c.x, y, c.z) + side * s;
}

struct ArchOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) a: u32,
  @location(3) deck: f32,  // 1 on the deck, 0 on the rock
  @location(4) t: f32,
};

@vertex
fn vsArch(@builtin(vertex_index) vi: u32, @builtin(instance_index) a: u32) -> ArchOut {
  let i = vi / ${ARCH_SIDES}u;
  let j = vi % ${ARCH_SIDES}u;
  let t = f32(i) / ${f(ARCH_SEGS)};
  let p = archPoint(a, t, j);
  let dt = 1.0 / ${f(ARCH_SEGS)};
  let pt = archPoint(a, min(t + dt, 1.0), j) - archPoint(a, max(t - dt, 0.0), j);
  let pj = archPoint(a, t, (j + 1u) % ${ARCH_SIDES}u) - archPoint(a, t, (j + ${ARCH_SIDES - 1}u) % ${ARCH_SIDES}u);
  var n = normalize(cross(pj, pt));
  if (j <= 6u && n.y < 0.0) { n = -n; }
  // the hull faces outward from the deck's axis
  let c = archT(realm.arch[a], t);
  if (j > 6u && dot(n, p - vec3f(c.x, p.y + 0.5 * (c.y - p.y), c.z)) < 0.0) { n = -n; }
  let rel = p - u.camPos;
  var o: ArchOut;
  o.pos = project(rel);
  o.rel = rel;
  o.n = n;
  o.a = a;
  o.deck = select(0.0, 1.0, j <= 6u);
  o.t = t;
  return o;
}

@fragment
fn fsArch(i: ArchOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let A = realm.arch[i.a];
  let b = u32(select(A.e.y, A.e.z, i.t > 0.5));
  let p = u.camPos + i.rel;
  let n = normalize(i.n);
  let g = p.xz + u.origin;
  let fine = 0.5 + 0.5 * vnoise(g.x / 5.0, g.y / 5.0, 4111);
  let band = p.y / 5.0 + 1.3 * vnoise(g.x / 30.0, g.y / 30.0, 4113);
  let rock = mix(BI_STRATA0[b], BI_STRATA1[b], 0.5 + 0.5 * vnoise(floor(band) * 2.7, 0.5, 4117)) * (0.85 + 0.3 * fine);
  // Grass on the deck (as the islands' grass reads), worn to rock at its edges; mossy rock below.
  let patchN = 0.5 + 0.5 * vnoise(g.x / 23.0, g.y / 23.0, 4119);
  let grass = mix(mix(BI_DRY[b], BI_LUSH[b], clamp(patchN * 1.2 - 0.1, 0.0, 1.0)), vec3f(0.1, 0.18, 0.045), 0.3) * 0.8 * (0.9 + 0.2 * fine);
  let moss = mix(rock, BI_RIM[b], 0.45 * smoothstep(0.2, 0.6, n.y));
  let albedo = mix(moss, grass, smoothstep(0.6, 0.9, n.y) * i.deck);
  return rlShade(albedo, p, n, i.rel, 0.9);
}

// ---- lakes ------------------------------------------------------------------------------------------

struct LakeOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) @interpolate(flat) k: u32,
  @location(2) d: f32,   // normalised distance from the lake's middle
};

@vertex
fn vsLake(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LakeOut {
  let k = realmU.inst[ii >> 2u][ii & 3u] & 0xffffu;
  let i = vi / ${LAKE_SEGS}u;
  let j = vi % ${LAKE_SEGS}u;
  let d = 1.45 * f32(i) / ${f(LAKE_RINGS)};
  let a = f32(j) * (6.2831853 / ${f(LAKE_SEGS)});
  let L0 = realm.isl[k].l0;
  let L1 = realm.isl[k].l1;
  let L2 = realm.isl[k].l2;
  let w = 1.0 + 0.12 * sin(3.0 * a + L2.x) + 0.07 * sin(5.0 * a + L2.y);
  let uu = cos(a) * L0.z * w * d;
  let vv = sin(a) * L0.z * w * d * L1.x;
  let q = L0.xy + vec2f(uu * L1.y - vv * L1.z, uu * L1.z + vv * L1.y);
  let xz = realm.isl[k].c.xy + q;
  let rel = vec3f(xz.x, L0.w, xz.y) - u.camPos;
  var o: LakeOut;
  o.pos = project(rel);
  o.rel = rel;
  o.k = k;
  o.d = d;
  return o;
}

fn rlWaterNormal(g: vec2f, t: f32, strength: f32) -> vec3f {
  let e = 0.08;
  let h0 = waterHeight(g, t);
  let hx = waterHeight(g + vec2f(e, 0.0), t);
  let hz = waterHeight(g + vec2f(0.0, e), t);
  return normalize(vec3f(-(hx - h0) / e * strength, 1.0, -(hz - h0) / e * strength));
}

// Sky water over a bed depthV m down: the land's lake shading (absorption, ripples, the sky's reflection,
// a glint, foam where it laps), clearer up here and tinted by the biome. Premultiplied.
fn rlWaterColour(p: vec3f, rel: vec3f, depthV: f32, flow: vec2f, foamExtra: f32, k: u32) -> vec4f {
  let b = realm.isl[k].m.y;
  let lagoon = ((realm.isl[k].m.z >> ${RL_TINT_BIT}u) & 1u) == 1u;
  let tS = length(rel);
  let rd = rel / tS;
  let g = p.xz + u.origin - flow * u.time;
  let strength = mix(0.03, 0.12, smoothstep(0.0, 1.2, depthV)) * (1.0 - smoothstep(40.0, 300.0, tS));
  // (past 300 m the water lies flat: its noise is not evaluated)
  var n = vec3f(0.0, 1.0, 0.0);
  if (strength > 0.0) { n = rlWaterNormal(g, u.time, strength); }
  let sh = skyShadowAt(p) * treeShadowAt(p);
  let path = depthV / max(-rd.y, 0.06);
  var absorb = vec3f(0.3, 0.07, 0.045);
  if (b == BI_MIST) { absorb = vec3f(0.45, 0.12, 0.09); }
  // A tended lagoon (the Eyries): clear turquoise over a pale bed.
  if (lagoon) { absorb = vec3f(0.42, 0.06, 0.07); }
  let T = exp(-absorb * path);
  let Tl = dot(T, vec3f(0.2126, 0.7152, 0.0722));
  let kc = keyCol(p.y);
  let up = vec3f(0.0, 1.0, 0.0);
  var bodyTint = vec3f(0.01, 0.075, 0.12);
  if (b == BI_MIST) { bodyTint = vec3f(0.006, 0.05, 0.05); }
  if (lagoon) { bodyTint = vec3f(0.01, 0.15, 0.15); }
  let body = bodyTint * (kc * 0.35 * (0.4 + 0.6 * sh) + skyAmb(up, 0.8));
  let refl = reflect(rd, n);
  let F = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
  let sd = max(dot(refl, keyDir()), 0.0);
  let glint = pow(sd, 900.0) * 14.0 + pow(sd, 120.0) * 0.3;
  let edge = 1.0 - smoothstep(0.0, 0.08, depthV);
  var foam = 0.0;
  // (its noise only where foam shows: at the edge, or where the stream adds it)
  if (max(edge, foamExtra) > 0.0) {
    let fnz = vnoise(g.x / 2.3, g.y / 2.3, 941);
    foam = clamp(max(edge, foamExtra) * (0.6 + 0.4 * fnz), 0.0, 1.0);
  }
  let foamCol = vec3f(0.85, 0.88, 0.9) * (kc * 0.32 * (0.35 + 0.65 * sh) + skyAmb(up, 0.55));
  var col = skyReflect(refl) * F + body * (vec3f(1.0) - T) * (1.0 - F) + kc * glint * sh;
  var alpha = 1.0 - (1.0 - F) * Tl;
  col = mix(col, foamCol, foam);
  alpha = mix(alpha, 1.0, foam);
  let a = max(alpha, 1e-3);
  let shown = display(surfaceFog(col / a, tS, p.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}

@fragment
fn fsLake(i: LakeOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let k = i.k;
  let p = u.camPos + i.rel;
  let b = realm.isl[k].m.y;
  // The bowl's depth under this point (the shore rises out of the water past d = 1).
  let depthV = max(realm.isl[k].l0.w - rlLakeBed(k, i.d), 0.0);
  if ((realm.isl[k].m.z & RL_FROZEN) != 0u) {
    // Ice: pale and cracked, opaque, snow drifted at its edges.
    let g = p.xz + u.origin;
    let crack = smoothstep(0.92, 0.985, 1.0 - abs(vnoise(g.x / 11.0, g.y / 11.0, 4201)));
    let drift = smoothstep(0.55, 1.0, i.d) * (0.5 + 0.5 * vnoise(g.x / 7.0, g.y / 7.0, 4203));
    var ice = mix(vec3f(0.33, 0.45, 0.55), vec3f(0.55, 0.68, 0.78), 0.5 + 0.5 * vnoise(g.x / 23.0, g.y / 23.0, 4205));
    ice = mix(ice * (1.0 - 0.35 * crack), vec3f(0.78, 0.82, 0.88), drift);
    return rlShade(ice, p, vec3f(0.0, 1.0, 0.0), i.rel, 1.0);
  }
  return rlWaterColour(p, i.rel, depthV, vec2f(0.0), 0.0, k);
}

// ---- streams -----------------------------------------------------------------------------------------

struct StreamOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) @interpolate(flat) k: u32,
  @location(2) uv: vec2f,      // across (−1..1), metres along
  @location(3) rapid: f32,     // how steeply the bed falls here (0 calm … 1 white water)
  @location(4) flow: vec2f,    // the water's direction (unit, xz)
};

@vertex
fn vsStream(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> StreamOut {
  let k = realmU.inst[ii >> 2u][ii & 3u] & 0xffffu;
  let m = realm.isl[k].m.w;
  let first = m & 0xffffu;
  let cnt = max(m >> 16u, 2u); // (an island in this list always has its stream; never index past it)
  let sec = vi / 2u;
  let side = f32(vi & 1u) * 2.0 - 1.0;
  let last = (cnt - 1u) * ${STREAM_SUB}u;
  let sc = min(sec, last);
  let g = min(sc / ${STREAM_SUB}u, cnt - 2u);
  let fr = f32(sc - g * ${STREAM_SUB}u) / ${f(STREAM_SUB)};
  let a = realm.pts[first + g];
  let b = realm.pts[first + g + 1u];
  let q = a.xy + (b.xy - a.xy) * fr;
  // Sideways: the stretch's own, eased into the next one's at the joins.
  let d0 = normalize(b.xy - a.xy);
  var d1 = d0;
  if (g + 2u < cnt) { d1 = normalize(realm.pts[first + g + 2u].xy - b.xy); }
  let dir = normalize(mix(d0, d1, smoothstep(0.5, 1.0, fr)) + vec2f(1e-5, 0.0));
  let hw = a.w + (b.w - a.w) * fr;
  let bed = a.z + (b.z - a.z) * fr;
  let sq = q + vec2f(-dir.y, dir.x) * (side * 0.8 * hw);
  let xz = realm.isl[k].c.xy + sq;
  let rel = vec3f(xz.x, bed + RL_STREAM_DEPTH, xz.y) - u.camPos;
  var o: StreamOut;
  o.pos = project(rel);
  o.rel = rel;
  o.k = k;
  var along = fr * length(b.xy - a.xy);
  for (var h = 0u; h < g; h++) { along += length(realm.pts[first + h + 1u].xy - realm.pts[first + h].xy); }
  o.uv = vec2f(side, along);
  let fall = (a.z - b.z) / max(length(b.xy - a.xy), 1.0);
  o.rapid = smoothstep(0.02, 0.15, fall) + smoothstep(0.9, 1.0, f32(sc) / f32(last)) * 0.6;
  o.flow = dir;
  return o;
}

@fragment
fn fsStream(i: StreamOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let p = u.camPos + i.rel;
  let speed = 1.6 + 3.0 * i.rapid;
  // Streaks of foam carried down the stream, more of them in the rapids.
  let along = i.uv.y - u.time * speed;
  let streak = smoothstep(0.55, 0.85, 0.5 + 0.5 * vnoise(i.uv.x * 2.5, along / 3.5, 4301 + i32(i.k)));
  let foam = clamp(i.rapid * (0.35 + 0.65 * streak) + 0.25 * streak * (1.0 - smoothstep(0.5, 1.0, abs(i.uv.x))), 0.0, 1.0);
  let depthV = 0.45 * (1.0 - i.uv.x * i.uv.x);
  let c = rlWaterColour(p, i.rel, depthV, i.flow * speed, foam, i.k);
  // the ribbon's edges tuck under the banks
  let edge = 1.0 - smoothstep(0.85, 1.0, abs(i.uv.x));
  return c * edge;
}

// ---- the realm's shadow map: undersides and arches seen from the sun ---------------------------------

fn rlToShadow(rel: vec3f) -> vec4f {
  let q = rel - realmU.shN.xyz;
  let W = realmU.shC.w;
  let ls = vec3f(dot(q, realmU.nR.xyz), dot(q, realmU.nU.xyz), dot(q, realmU.nF.xyz));
  return vec4f(ls.x / W, ls.y / W, 0.5 + 0.5 * ls.z / realmU.nF.w, 1.0);
}

@vertex
fn vsRealmShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let w = realmU.inst[ii >> 2u][ii & 3u];
  let k = w & 0xffffu;
  let n = ${POLAR_LODS[1][1]}u;
  let i = vi / n;
  let j = vi % n;
  let rho = rlRingRho(i, ${POLAR_LODS[1][0]}u);
  let th = f32(j) * (6.2831853 / f32(n));
  let c = realm.isl[k].c;
  let q = (c.z * rlOutline(k, th) * rho) * vec2f(cos(th), sin(th));
  let xz = c.xy + q;
  return rlToShadow(vec3f(xz.x, rlBottom(k, q), xz.y) - u.camPos);
}

@vertex
fn vsArchShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) a: u32) -> @builtin(position) vec4f {
  let i = vi / ${ARCH_SIDES}u;
  let j = vi % ${ARCH_SIDES}u;
  return rlToShadow(archPoint(a, f32(i) / ${f(ARCH_SEGS)}, j) - u.camPos);
}
`;
}

/** Triangles of a polar grid: rings 0..rings (ring 0: the centre, repeated), bearings wrap. */
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

/** Triangles of an arch: sections 0..ARCH_SEGS × ARCH_SIDES points round each (wrapping). */
export function archIndices(): Uint32Array<ArrayBuffer> {
  const out = new Uint32Array(ARCH_SEGS * ARCH_SIDES * 6);
  let o = 0;
  for (let i = 0; i < ARCH_SEGS; i++) {
    for (let j = 0; j < ARCH_SIDES; j++) {
      const j1 = (j + 1) % ARCH_SIDES;
      const a = i * ARCH_SIDES + j, b = (i + 1) * ARCH_SIDES + j, c = (i + 1) * ARCH_SIDES + j1, d = i * ARCH_SIDES + j1;
      out[o++] = a; out[o++] = b; out[o++] = c;
      out[o++] = a; out[o++] = c; out[o++] = d;
    }
  }
  return out;
}
