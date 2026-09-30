/**
 * Forest on the GPU (WGSL): placement into a cached window of tiles, per-frame
 * culling and level-of-detail selection into instance buckets, the tree
 * shaders, and a sun shadow map of the trees.
 * ---------------------------------------------------------------------------
 * forestGen     one workgroup per tile entering the window (forest/tiles.ts):
 *               each of its 8 × 8 lattice cells decides once whether it holds
 *               a plant, and which: the rules of forest/placement.ts (its
 *               tables compiled in below, its steps mirrored line for line).
 * forestCull    every frame, every slot: distance, a stable thinning far out,
 *               the view frustum, the level of detail (dithered cross-fade over
 *               ±6 % of each switch distance); appends to one bucket per
 *               (species, shape, level) — the instance lists of the indexed
 *               indirect draws — and to the shadow buckets inside the shadow box.
 * forestClamp   caps each bucket's instance count at its capacity.
 * vsTree/fsTree the tree: yaw, scale, wind; stylized leaves and bark, snow.
 * vsTreeShadow  the same transform into the sun's orthographic shadow map.
 * treeShadowAt  sun visibility under the trees (5 filtered comparison taps),
 *               used by every surface, the grass, the hero and the trees.
 *
 * Needs, from the raster shader: u (Frame), WORLD_WGSL (hash3, vnoise,
 * elevation, temperature, moisture, W_LEVEL, SUN, SUN_COL, SKY_TINT),
 * heightAt, sunAt, islandTopAt, islandShadowAt, inRuinSite, project,
 * surfaceFog, display, underwater.
 */

import { PLACE, RULES, ZONES } from '../forest/placement';
import { CELL, SPECIES, SPECIES_INFO, VARIANTS, type Species } from '../forest/species';
import { TILE_CELLS } from '../forest/tiles';
import { wgslF32 as f } from './worldgen.wgsl';

export type ForestSpec = {
  /** Multiplies every plant probability (1 = as designed). */
  density: number;
  /** Tiles per side of the cached window (even); a tile is 8 × 8 cells of 6 m. */
  side: number;
  /** Shadow map texels per side (0: no tree shadows), and its half width (m) around the camera. */
  shadowRes: number;
  shadowRange: number;
  /** Cross-fade levels of detail with a dither (else switch outright). */
  fade?: boolean;
};

export const FOREST_BYTES = 128;
export const SLOT_BYTES = 16;
export const INSTANCE_BYTES = 32;
/** Draw arguments per bucket: indexCount, instanceCount, firstIndex, baseVertex, firstInstance. */
export const ARGS_BYTES = 20;
export const KINDS = SPECIES.length * VARIANTS;
/**
 * Bucket groups, KINDS buckets each: whole trees at levels 0, 1, 2 (drawn
 * without discard, so the GPU's hidden-surface removal works); shadow casters
 * near the camera (level 1 shapes); trees in a level's cross-fade band
 * (dithered); shadow casters farther out (level 2 shapes).
 */
export const GROUPS = 8;
/** Level of detail each group draws. */
export const GROUP_LEVEL = [0, 1, 2, 1, 0, 1, 2, 2] as const;
/** Instances per bucket in each group. */
export const CAP = [256, 1024, 16384, 512, 128, 512, 1024, 2048] as const;
/** Shadow casters within this distance (m) cast their level 1 shape. */
const SHADOW_NEAR = 32;
/** First instance of each group's buckets in the one instance buffer. */
export const REGION = CAP.map((_, g) => CAP.slice(0, g).reduce((p, c) => p + KINDS * c, 0));
export const INSTANCES = REGION[GROUPS - 1] + KINDS * CAP[GROUPS - 1];
export const BUCKETS = GROUPS * KINDS;
/** Beyond this distance (m) a stable share of the plants drops out (the rest grow a little). */
const FAR_THIN = 420;

const arr = (type: string, xs: string[]) => `array<${type}, ${xs.length}>(${xs.join(', ')})`;
const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;

/** forest/placement.ts as WGSL: its tables, then its steps mirrored (keep the two in step). */
function placementWGSL(): string {
  const Z = ZONES.map((z) => RULES[z]);
  const sp = (x: Species) => SPECIES.indexOf(x);
  const pair = (p: readonly [Species, Species, number]) => `vec3f(${sp(p[0])}.0, ${sp(p[1])}.0, ${f(p[2])})`;
  const noise = (wl: number, off: number) => `plNoise(x, z, ${f(wl)}, ${off})`;
  return /* wgsl */ `
var<private> Z_A: array<vec4f, ${Z.length}> = ${arr('vec4f', Z.map((r) => `vec4f(${f(r.forest)}, ${f(r.fill)}, ${f(r.shore)}, ${f(r.shrub)})`))};
var<private> Z_G: array<vec4f, ${Z.length}> = ${arr('vec4f', Z.map((r) => `vec4f(${f(r.groveGrid)}, ${f(r.groveP)}, ${f(r.groveR[0])}, ${f(r.groveR[1])})`))};
var<private> Z_GF: array<f32, ${Z.length}> = ${arr('f32', Z.map((r) => f(r.groveFill)))};
var<private> Z_L: array<vec2f, ${Z.length}> = ${arr('vec2f', Z.map((r) => `vec2f(${f(r.loneGrid)}, ${f(r.loneP)})`))};
var<private> Z_FSP: array<vec3f, ${Z.length}> = ${arr('vec3f', Z.map((r) => pair(r.forestSp)))};
var<private> Z_GSP: array<vec3f, ${Z.length}> = ${arr('vec3f', Z.map((r) => pair(r.groveSp)))};
var<private> Z_LSP: array<vec3f, ${Z.length}> = ${arr('vec3f', Z.map((r) => pair(r.loneSp)))};
var<private> Z_SSP: array<u32, ${Z.length}> = ${arr('u32', Z.map((r) => `${sp(r.shoreSp)}u`))};
const PL_DESERT: i32 = ${ZONES.indexOf('desert')};
const PL_BUSH: u32 = ${sp('bush')}u;

fn plU16(h: u32, shift: u32) -> f32 { return f32((h >> shift) & 0xffffu) / 65535.0; }
fn plNoise(x: f32, z: f32, wl: f32, off: i32) -> f32 { return 0.5 + 0.5 * vnoise(x / wl, z / wl, u.seed + off); }
fn plFdiv(a: i32, b: i32) -> i32 { return i32(floor(f32(a) / f32(b))); }
fn plPair(p: vec3f, v: f32) -> u32 { return select(u32(p.y), u32(p.x), v < 1.0 - p.z); }

fn zoneWeights(T: f32, M: f32, region: f32) -> array<f32, ${Z.length}> {
  let cold = 1.0 - smoothstep(1.5, 5.5, T);
  let alpine = 1.0 - smoothstep(-1.0, 1.0, T);
  let treeline = smoothstep(-3.5, -1.5, T);
  let warm = smoothstep(12.2, 13.2, T);
  let desert = 1.0 - smoothstep(0.14, 0.26, M);
  let dry = 1.0 - smoothstep(0.34, 0.5, M);
  let wet = smoothstep(0.85, 1.05, M);
  let autumn = smoothstep(${f(PLACE.region.lo)}, ${f(PLACE.region.hi)}, region);
  let mild = (1.0 - cold) * (1.0 - warm) * (1.0 - dry);
  return array<f32, ${Z.length}>(
    mild * (1.0 - autumn), mild * autumn, warm * (1.0 - wet) * (1.0 - dry) * (1.0 - cold), warm * wet * (1.0 - dry),
    dry * (1.0 - desert) * (1.0 - cold), desert * (1.0 - cold), cold * (1.0 - alpine), alpine * treeline);
}

fn forestField(x: f32, z: f32) -> f32 {
  let wl = ${f(PLACE.warp.wavelength)};
  let wx = x + ${f(PLACE.warp.amount)} * vnoise(x / wl, z / wl, u.seed + ${PLACE.warp.seed});
  let wz = z + ${f(PLACE.warp.amount)} * vnoise(x / wl + 17.3, z / wl - 9.1, u.seed + ${PLACE.warp.seed + 1});
  return 0.5 + 0.5 * (${PLACE.field.map(([wl, amp, off]) => `${f(amp)} * vnoise(wx / ${f(wl)}, wz / ${f(wl)}, u.seed + ${off})`).join(' + ')});
}

struct PlSite {
  ok: bool, e: f32, slope: f32, hollow: f32, knoll: f32, T: f32,
  zone: i32, dom: i32, level: f32, fill: f32, mask: f32, open: f32,
};

fn plSite(x: f32, z: f32, region: f32, zoneR: f32) -> PlSite {
  var o: PlSite;
  o.ok = false;
  let g = vec2f(x, z);
  o.e = elevation(g, 1.0, u.seed);
  if (o.e < W_LEVEL + 0.4) { return o; }
  o.slope = length(vec2f(elevation(g + vec2f(1.5, 0.0), 1.0, u.seed) - o.e, elevation(g + vec2f(0.0, 1.5), 1.0, u.seed) - o.e)) / 1.5;
  let L = ${f(PLACE.relief.step)};
  let dp = ${f(PLACE.relief.dp)};
  let e0 = elevation(g, dp, u.seed);
  let relief = (elevation(g + vec2f(L, 0.0), dp, u.seed) + elevation(g - vec2f(L, 0.0), dp, u.seed)
    + elevation(g + vec2f(0.0, L), dp, u.seed) + elevation(g - vec2f(0.0, L), dp, u.seed) - 4.0 * e0) / L;
  // Forest bodies follow the hills (84 m), not every hummock.
  let H = ${f(PLACE.hills.step)};
  let hp = ${f(PLACE.hills.dp)};
  let h0 = elevation(g, hp, u.seed);
  let hE = elevation(g + vec2f(H, 0.0), hp, u.seed);
  let hW = elevation(g - vec2f(H, 0.0), hp, u.seed);
  let hN = elevation(g + vec2f(0.0, H), hp, u.seed);
  let hS = elevation(g - vec2f(0.0, H), hp, u.seed);
  let hills = (hE + hW + hN + hS - 4.0 * h0) / H;
  let hillSlope = length(vec2f(hE - hW, hN - hS)) / (2.0 * H);
  o.hollow = smoothstep(0.03, 0.15, relief);
  o.knoll = smoothstep(0.03, 0.15, -relief);
  o.T = temperature(g, o.e);
  let M = moisture(g, u.seed);
  var zw = zoneWeights(o.T, M, region);
  var total = 0.0;
  for (var k = 0; k < ${Z.length}; k++) { total += zw[k]; }
  if (total < 1e-4) { return o; }
  var acc = 0.0;
  o.zone = 0;
  o.dom = 0;
  o.level = 0.0;
  o.fill = 0.0;
  for (var k = 0; k < ${Z.length}; k++) {
    acc += zw[k] / total;
    if (zoneR >= acc) { o.zone = k + 1; }
    if (zw[k] > zw[o.dom]) { o.dom = k; }
    o.level += zw[k] * Z_A[k].x;
    o.fill += zw[k] * Z_A[k].y;
  }
  o.zone = min(o.zone, ${Z.length - 1});
  o.level /= total;
  o.fill /= total;
  let suit = 0.1 * smoothstep(0.02, 0.12, hills) - 0.12 * smoothstep(0.02, 0.12, -hills) + 0.15 * (M - 0.62) - 0.3 * smoothstep(0.55, 0.9, hillSlope);
  o.mask = smoothstep(o.level - ${f(PLACE.edge)}, o.level + ${f(PLACE.edge)}, forestField(x, z) + suit);
  o.open = 1.0 - smoothstep(0.15, 0.4, o.mask);
  o.ok = true;
  return o;
}

// A plant record: offset in the tile (x, z), island height, and packed info (0 = no plant).
fn plRecord(sp: u32, off: vec2f, y: f32, scale: f32, snow: bool, flags: u32, bits: u32) -> vec4u {
  let sb = u32(clamp(round((scale - 0.55) / 1.05 * 255.0), 0.0, 255.0));
  let info = sp | (((bits & 0xffffu) % F_VARIANTS) << 4u) | (((bits >> 16u) & 255u) << 6u) | (sb << 14u) | (((bits >> 8u) & 63u) << 22u)
    | select(0u, F_SNOW, snow) | flags | F_VALID;
  return vec4u(bitcast<u32>(off.x), bitcast<u32>(off.y), bitcast<u32>(y), info);
}

fn plSnow(T: f32, sp: u32) -> bool {
  return T < 0.5 && (sp == SP_PINE || sp == SP_DEAD || sp == SP_BUSH || sp == SP_BIRCH);
}

// Lattice cell (ix, iz), cell lid in its tile, chunk-local xz of its spot, spare random bits.
fn plantAt(ix: i32, iz: i32, lid: vec2f, xz: vec2f, bits: u32) -> vec4u {
  let h0 = hash3(ix, iz, u.seed + 7101);
  let h1 = hash3(ix, iz, u.seed + 7102);
  let h2 = hash3(ix, iz, u.seed + 7103);
  let jit = vec2f(0.08 + 0.84 * plU16(h0, 0u), 0.08 + 0.84 * plU16(h0, 16u));
  let keep = plU16(h1, 0u) / max(forest.camOff.z, 1e-6);
  let pick = plU16(h1, 16u);
  let bush = plU16(h2, 0u) / max(forest.camOff.z, 1e-6);
  let zoneR = plU16(h2, 16u);
  let x = (f32(ix) + jit.x) * F_CELL;
  let z = (f32(iz) + jit.y) * F_CELL;
  var off = (lid + jit) * F_CELL;
  if (inRuinSite(xz)) { return vec4u(0u); }
  let region = ${noise(PLACE.region.wavelength, PLACE.region.seed)};

  // Sky-island tops: lowland broadleaves as lone trees and in groves, a few bushes.
  let top = islandTopAt(xz, 0.82);
  if (top > -1e8) {
    let autumn = region > ${f((PLACE.region.lo + PLACE.region.hi) / 2)};
    let kl = ${36 / CELL};
    let lx = plFdiv(ix, kl);
    let lz = plFdiv(iz, kl);
    let lh = hash3(lx, lz, u.seed + 7231);
    let lp = hash3(lx, lz, u.seed + 7232);
    let fx = (0.15 + 0.7 * plU16(lh, 0u)) * f32(kl);
    let fz = (0.15 + 0.7 * plU16(lh, 16u)) * f32(kl);
    if (ix == lx * kl + i32(floor(fx)) && iz == lz * kl + i32(floor(fz)) && plU16(lp, 0u) < 0.45 * forest.camOff.z) {
      off = (lid + vec2f(fract(fx), fract(fz))) * F_CELL;
      return plRecord(select(SP_OAK, SP_MAPLE, autumn), off, top, 1.1 + 0.2 * pick, false, F_ISLAND, bits);
    }
    if (keep < 0.3 * smoothstep(0.55, 0.75, ${noise(40, 7233)})) {
      return plRecord(select(select(SP_OAK, SP_MAPLE, autumn), SP_BIRCH, pick < 0.3), off, top, 0.9 + 0.2 * pick, false, F_ISLAND, bits);
    }
    if (bush < 0.02) { return plRecord(PL_BUSH, off, top, 0.8 + 0.4 * pick, false, F_ISLAND, bits); }
    return vec4u(0u);
  }

  let st = plSite(x, z, region, zoneR);
  if (!st.ok) { return vec4u(0u); }
  let steep = smoothstep(0.5, 0.9, st.slope);

  // Lone tree: this cell hosts its coarse cell's spot?
  let kl = i32(Z_L[st.dom].x / F_CELL);
  let lx = plFdiv(ix, kl);
  let lz = plFdiv(iz, kl);
  let lh = hash3(lx, lz, u.seed + 7221);
  let lp = hash3(lx, lz, u.seed + 7222);
  let fx = (0.15 + 0.7 * plU16(lh, 0u)) * f32(kl);
  let fz = (0.15 + 0.7 * plU16(lh, 16u)) * f32(kl);
  if (ix == lx * kl + i32(floor(fx)) && iz == lz * kl + i32(floor(fz))) {
    var desert = 0.0;
    if (st.dom == PL_DESERT) { desert = 0.4 * smoothstep(0.55, 0.75, ${noise(PLACE.cactus.wavelength, PLACE.cactus.seed)}); }
    let park = ${f(PLACE.parkland.lo)} + ${f(PLACE.parkland.span)} * smoothstep(0.3, 0.7, ${noise(PLACE.parkland.wavelength, PLACE.parkland.seed)});
    let p = (Z_L[st.dom].y + desert) * park * (${f(PLACE.lone.flat)} + ${f(PLACE.lone.knoll)} * st.knoll) * (1.0 - smoothstep(0.35, 0.6, st.slope)) * st.open;
    if (plU16(lp, 0u) < p * forest.camOff.z) {
      off = (lid + vec2f(fract(fx), fract(fz))) * F_CELL;
      let sp = plPair(Z_LSP[st.dom], plU16(lp, 16u));
      return plRecord(sp, off, 0.0, 1.15 + 0.2 * pick, plSnow(st.T, sp), 0u, bits);
    }
  }

  let glade = smoothstep(${f(PLACE.glade.lo)}, ${f(PLACE.glade.hi)}, ${noise(PLACE.glade.wavelength, PLACE.glade.seed)});
  let fD = st.fill * pow(st.mask, 1.4) * (1.0 - ${f(PLACE.glade.cut)} * glade) * (1.0 - smoothstep(0.8, 1.15, st.slope));

  // Grove: its coarse cell's clump (kept inside the cell).
  let G = Z_G[st.dom];
  let kg = i32(G.x / F_CELL);
  let gx = plFdiv(ix, kg);
  let gz = plFdiv(iz, kg);
  let gh = hash3(gx, gz, u.seed + 7211);
  let gp = hash3(gx, gz, u.seed + 7212);
  let gr = G.z + (G.w - G.z) * plU16(gp, 16u);
  let inset = gr / G.x;
  let gc = (vec2f(f32(gx), f32(gz)) + inset + (1.0 - 2.0 * inset) * vec2f(plU16(gh, 0u), plU16(gh, 16u))) * G.x;
  let gd = length(vec2f(x, z) - gc);
  var gD = 0.0;
  let gpark = ${f(PLACE.parkland.lo)} + ${f(PLACE.parkland.span)} * smoothstep(0.3, 0.7, plNoise(gc.x, gc.y, ${f(PLACE.parkland.wavelength)}, ${PLACE.parkland.seed}));
  if (plU16(gp, 0u) < G.y * gpark * (0.5 + st.hollow) * st.open) { gD = Z_GF[st.dom] * (1.0 - smoothstep(0.45 * gr, gr, gd)) * (1.0 - steep); }

  // Shore: a broken line just above the water.
  let gaps = smoothstep(0.35, 0.55, ${noise(PLACE.gaps.wavelength, PLACE.gaps.seed)});
  let sD = Z_A[st.dom].z * smoothstep(W_LEVEL + 0.4, W_LEVEL + 1.2, st.e) * (1.0 - smoothstep(W_LEVEL + 3.0, W_LEVEL + 6.5, st.e)) * (1.0 - steep) * gaps * st.open;

  let stand = ${noise(PLACE.stand.wavelength, PLACE.stand.seed)};
  if (keep < fD) {
    let giant = select(1.0, 1.3, pick < ${f(PLACE.giant)});
    let sp = plPair(Z_FSP[st.zone], stand);
    return plRecord(sp, off, 0.0, (0.95 + 0.2 * pick) * (0.78 + 0.22 * st.mask) * giant, plSnow(st.T, sp), 0u, bits);
  }
  if (keep < fD + gD) {
    let sp = plPair(Z_GSP[st.dom], plU16(gh, 8u));
    return plRecord(sp, off, 0.0, (0.92 + 0.2 * pick) * (1.08 - 0.16 * gd / gr), plSnow(st.T, sp), 0u, bits);
  }
  if (keep < fD + gD + sD) {
    let sp = Z_SSP[st.dom];
    return plRecord(sp, off, 0.0, 0.85 + 0.2 * pick, plSnow(st.T, sp), 0u, bits);
  }

  // Bushes: sparse in the open, lining forest edges, a few under the trees.
  let edge = 4.0 * st.mask * (1.0 - st.mask);
  let bushP = (Z_A[st.zone].w * st.open + 0.12 * edge * select(0.0, 1.0, st.level < 1.0) + 0.03 * st.mask) * (1.0 - steep);
  if (bush < bushP) { return plRecord(PL_BUSH, off, 0.0, 0.8 + 0.4 * pick, plSnow(st.T, PL_BUSH), 0u, bits); }
  return vec4u(0u);
}
`;
}

export function forestWGSL(spec: ForestSpec, bounds: ReadonlyArray<{ top: number; radius: number }>): string {
  const S = SPECIES.map((s) => SPECIES_INFO[s]);
  const sp = (name: (typeof SPECIES)[number]) => `${SPECIES.indexOf(name)}u`;
  const side = spec.side;
  return /* wgsl */ `
const F_SIDE: u32 = ${side}u;
const F_SLOTS: u32 = ${side * side * TILE_CELLS * TILE_CELLS}u;
const F_TILE_CELLS: i32 = ${TILE_CELLS};
const F_CELL: f32 = ${f(CELL)};
const F_TILE: f32 = ${f(CELL * TILE_CELLS)};
const F_KINDS: u32 = ${KINDS}u;
const F_VARIANTS: u32 = ${VARIANTS}u;
const F_FAR_THIN: f32 = ${f(FAR_THIN)};
const F_SHADOW_RES: f32 = ${f(Math.max(spec.shadowRes, 1))};
const F_VALID: u32 = 0x40000000u;
const F_ISLAND: u32 = 0x20000000u;
const F_SNOW: u32 = 0x10000000u;
const SP_OAK: u32 = ${sp('oak')};
const SP_BIRCH: u32 = ${sp('birch')};
const SP_MAPLE: u32 = ${sp('maple')};
const SP_PINE: u32 = ${sp('pine')};
const SP_PALM: u32 = ${sp('palm')};
const SP_CACTUS: u32 = ${sp('cactus')};
const SP_DEAD: u32 = ${sp('dead')};
const SP_BUSH: u32 = ${sp('bush')};
var<private> F_CAP: array<u32, ${GROUPS}> = ${arr('u32', CAP.map((c) => `${c}u`))};
var<private> F_REGION: array<u32, ${GROUPS}> = ${arr('u32', REGION.map((c) => `${c}u`))};
override TREE_FADE: bool = false;
var<private> SP_TOP: array<f32, ${S.length}> = ${arr('f32', bounds.map((b) => f(b.top)))};
var<private> SP_RAD: array<f32, ${S.length}> = ${arr('f32', bounds.map((b) => f(b.radius)))};
var<private> SP_MAXD: array<f32, ${S.length}> = ${arr('f32', S.map((s) => f(s.maxDist)))};
var<private> SP_LOD: array<vec2f, ${S.length}> = ${arr('vec2f', S.map((s) => `vec2f(${f(s.lod[0])}, ${f(s.lod[1])})`))};
var<private> SP_LEAF0: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.leaf[0])))};
var<private> SP_LEAF1: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.leaf[1])))};
var<private> SP_BARK: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.bark)))};
var<private> SP_ACCENT: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.accent)))};

struct ForestU {
  camTile: vec4i,  // camera's tile x, z (absolute); unused
  camOff: vec4f,   // camera offset in its tile x, z (m); density; dither offset
  shC: vec4f,      // centre of the shadow map sampled now (rendered last frame; camera-relative); half width (m)
  shR: vec4f,      // light right axis; half depth (m)
  shU: vec4f,      // light up axis; texel (m)
  shF: vec4f,      // light forward axis (away from the sun); unused
  wind: vec4f,     // direction x, z; strength (m of sway at the tips); time
  shN: vec4f,      // centre of the shadow map rendered now (camera-relative); unused
};
@group(0) @binding(8) var<uniform> forest: ForestU;
@group(0) @binding(9) var shadowTex: texture_depth_2d;
@group(0) @binding(16) var shadowSamp: sampler_comparison;

// ---- tree shadows --------------------------------------------------------------

fn treeShadowAt(p: vec3f) -> f32 {${spec.shadowRes ? '' : ' return 1.0;'}
  let q = p - u.camPos - forest.shC.xyz;
  let R = forest.shC.w;
  let ls = vec3f(dot(q, forest.shR.xyz), dot(q, forest.shU.xyz), dot(q, forest.shF.xyz));
  let uv = vec2f(0.5 + 0.5 * ls.x / R, 0.5 - 0.5 * ls.y / R);
  let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
  let z = 0.5 + 0.5 * ls.z / forest.shR.w - 0.00008;
  if (edge >= 0.49 || z <= 0.0 || z >= 1.0) { return 1.0; }
  // One filtered comparison (a 2 × 2 bilinear PCF of 9 cm texels; TAA softens it further).
  let v = textureSampleCompareLevel(shadowTex, shadowSamp, uv, z);
  // Leaves let some light through: a full canopy still passes a fifth of the sun.
  return mix(1.0, 0.22 + 0.78 * v, smoothstep(0.49, 0.43, edge));
}

// ---- placement -------------------------------------------------------------------
${placementWGSL()}
@group(1) @binding(23) var<storage, read_write> forestSlots: array<vec4u>;
@group(1) @binding(24) var<storage, read_write> forestTiles: array<vec4i>;
@group(1) @binding(25) var<storage, read> forestJobs: array<vec4i>;

@compute @workgroup_size(8, 8)
fn forestGen(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let job = forestJobs[wg.x];
  let slot = u32(job.z);
  if (lid.x == 0u && lid.y == 0u) { forestTiles[slot] = vec4i(job.x, job.y, 0, 0); }
  let cell = job.xy * F_TILE_CELLS + vec2i(lid.xy);
  let h0 = hash3(cell.x, cell.y, u.seed + 7101);
  let jit = vec2f(0.08 + 0.84 * f32(h0 & 0xffffu) / 65535.0, 0.08 + 0.84 * f32(h0 >> 16u) / 65535.0);
  let xz = u.camPos.xz + vec2f(job.xy - forest.camTile.xy) * F_TILE + (vec2f(lid.xy) + jit) * F_CELL - forest.camOff.xy;
  forestSlots[slot * 64u + lid.y * 8u + lid.x] = plantAt(cell.x, cell.y, vec2f(lid.xy), xz, hash3(cell.x, cell.y, u.seed + 7104));
}

// ---- culling and levels of detail ---------------------------------------------------

struct Inst {
  p: vec4f, // base (camera-relative), scale
  q: vec4u, // info bits; fade (f32 bits: ≥ 1.5 whole, [0, 1] incoming share, < 0 outgoing); sun and wind bend (2 × f16); level
};
@group(1) @binding(21) var<storage, read_write> instOut: array<Inst>;
@group(1) @binding(22) var<storage, read_write> forestArgs: array<atomic<u32>>;

fn forestFrustum(c: vec3f, r: f32) -> bool {
  let v = c * u.basis;
  if (v.z < -r) { return false; }
  let tx = u.tanHalf * u.res.x / u.res.y;
  let ty = u.tanHalf;
  if (abs(v.x) - r * sqrt(1.0 + tx * tx) > v.z * tx) { return false; }
  if (abs(v.y) - r * sqrt(1.0 + ty * ty) > v.z * ty) { return false; }
  return true;
}

// Group: 0–2 whole trees at that level, 3 near shadow casters, 4–6 trees fading at level group − 4, 7 far casters.
fn emitTree(group: u32, level: u32, kind: u32, p: vec4f, info: u32, fade: f32, sunBend: u32) {
  let b = group * F_KINDS + kind;
  let i = atomicAdd(&forestArgs[b * 5u + 1u], 1u);
  if (i >= F_CAP[group]) { return; }
  instOut[F_REGION[group] + kind * F_CAP[group] + i] = Inst(p, vec4u(info, bitcast<u32>(fade), sunBend, level));
}

@compute @workgroup_size(64)
fn forestCull(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= F_SLOTS) { return; }
  let rec = forestSlots[i];
  let info = rec.w;
  if ((info & F_VALID) == 0u) { return; }
  let tile = forestTiles[i / 64u].xy;
  let relXZ = vec2f(tile - forest.camTile.xy) * F_TILE + vec2f(bitcast<f32>(rec.x), bitcast<f32>(rec.y)) - forest.camOff.xy;
  let sp = info & 15u;
  let d = length(relXZ);
  let maxD = SP_MAXD[sp];
  if (d >= maxD) { return; }
  // Far away a stable share drops out and the rest grow a little, keeping forests dense and cheap.
  let far = smoothstep(F_FAR_THIN, maxD, d);
  let cell = tile * F_TILE_CELLS + vec2i(i32(i % 8u), i32((i / 8u) % 8u));
  if (f32(hash3(cell.x, cell.y, 919) & 1023u) / 1023.0 > 1.0 - 0.45 * far) { return; }
  var s = (0.55 + 1.05 * f32((info >> 14u) & 255u) / 255.0) * (1.0 + 0.3 * far);
  s *= smoothstep(maxD, maxD * 0.9, d); // grow in at the edge of the range
  let xz = u.camPos.xz + relXZ;
  var y = bitcast<f32>(rec.z);
  if ((info & F_ISLAND) == 0u) { y = heightAt(xz, vec2f(0.0), d * u.footprint) - 0.08 - 0.0012 * d; }
  let rel = vec3f(relXZ.x, y - u.camPos.y, relXZ.y);
  let kind = sp * F_VARIANTS + ((info >> 4u) & 3u);
  let top = SP_TOP[sp] * s;
  let rad = length(vec2f(0.5 * top, SP_RAD[sp] * s));
  let p = vec4f(rel, s);
  let c = rel + vec3f(0.0, 0.5 * top, 0.0);
  // Wind for the whole tree: gusts travelling over the land, a slow sway on top.
  let g = xz + u.origin;
  let tw = forest.wind.w;
  let wq = g * 0.012 - forest.wind.xy * tw * 0.25;
  let gust = 0.5 + 0.5 * vnoise(wq.x, wq.y, 997);
  let bend = (0.35 + 0.9 * gust * gust) * (1.0 + 0.2 * sin(tw * 1.4 + dot(g, vec2f(0.37, 0.23))));
  // Shadow casters: everything inside the shadow box, seen or not.
  let q = c - forest.shN.xyz;
  if (abs(dot(q, forest.shR.xyz)) < forest.shC.w + rad && abs(dot(q, forest.shU.xyz)) < forest.shC.w + rad) {
    let far = d > ${f(SHADOW_NEAR)};
    emitTree(select(3u, 7u, far), select(1u, 2u, far), kind, p, info, 2.0, pack2x16float(vec2f(1.0, bend)));
  }
  if (!forestFrustum(c, rad)) { return; }
  // Mountain and island shadows, once per tree (at its crown) rather than per pixel.
  let crown = u.camPos + c + vec3f(0.0, 0.1 * top, 0.0);
  let sun = select(sunAt(xz), 1.0, (info & F_ISLAND) != 0u) * islandShadowAt(crown, 0xffffffffu);
  let sb = pack2x16float(vec2f(sun, bend));
  // Level of detail by distance to the crown's middle, bigger trees switching farther.
  let dl = length(c) / sqrt(s);
  let L = SP_LOD[sp];
  let band = ${spec.fade === false ? '0.0' : '0.06'};
  let a0 = L.x * (1.0 - band);
  let a1 = L.x * (1.0 + band);
  let b0 = L.y * (1.0 - band);
  let b1 = L.y * (1.0 + band);
  if (dl < a0) {
    emitTree(0u, 0u, kind, p, info, 2.0, sb);
  } else if (dl < a1) {
    let t = (dl - a0) / (a1 - a0);
    emitTree(4u, 0u, kind, p, info, -1.0 - t, sb);
    emitTree(5u, 1u, kind, p, info, t, sb);
  } else if (dl < b0) {
    emitTree(1u, 1u, kind, p, info, 2.0, sb);
  } else if (dl < b1) {
    let t = (dl - b0) / (b1 - b0);
    emitTree(5u, 1u, kind, p, info, -1.0 - t, sb);
    emitTree(6u, 2u, kind, p, info, t, sb);
  } else {
    emitTree(2u, 2u, kind, p, info, 2.0, sb);
  }
}

@compute @workgroup_size(64)
fn forestClamp(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= ${GROUPS}u * F_KINDS) { return; }
  let n = atomicLoad(&forestArgs[b * 5u + 1u]);
  atomicStore(&forestArgs[b * 5u + 1u], min(n, F_CAP[b / F_KINDS]));
}

// ---- drawing ------------------------------------------------------------------------

@group(1) @binding(20) var<storage, read> instIn: array<Inst>;

struct TreeOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) info: u32,
  @location(3) attr: vec4f,   // part, sway, occlusion, variation
  @location(4) lp: vec3f,     // model-space position (patterns stay on the tree)
  @location(5) @interpolate(flat) fade: f32,
  @location(6) @interpolate(flat) sunLod: vec2f, // mountain × island sun at the tree, level of detail
  @location(7) shade: f32,    // sun under the trees (the shadow map), per vertex
};

// Wind: the tree bends downwind with height (the bend is per tree, from the cull pass); leaves flutter.
fn treePlace(pos: vec3f, attr: vec4f, inst: Inst) -> vec3f {
  let yaw = f32((inst.q.x >> 6u) & 255u) * (6.2831853 / 256.0);
  let cy = cos(yaw);
  let sy = sin(yaw);
  let s = inst.p.w;
  let bend = unpack2x16float(inst.q.z).y;
  var d = vec3f(forest.wind.x, 0.0, forest.wind.y) * (forest.wind.z * attr.y * bend);
  let part = u32(attr.x * 255.0 + 0.5);
  if (part == 1u || part == 2u || part == 4u) {
    let fl = sin(forest.wind.w * 4.7 + yaw * 7.0 + dot(pos, vec3f(1.9, 1.3, 2.3))) * 0.05 * attr.y * (0.3 + 0.6 * bend);
    d += vec3f(fl, 0.5 * fl, -0.7 * fl);
  }
  let rot = vec3f(cy * pos.x + sy * pos.z, pos.y, -sy * pos.x + cy * pos.z);
  return inst.p.xyz + (rot + d) * s;
}

@vertex
fn vsTree(@location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) attr: vec4f, @builtin(instance_index) ii: u32) -> TreeOut {
  let inst = instIn[ii];
  let yaw = f32((inst.q.x >> 6u) & 255u) * (6.2831853 / 256.0);
  let cy = cos(yaw);
  let sy = sin(yaw);
  let rel = treePlace(pos, attr, inst);
  var o: TreeOut;
  o.pos = project(rel);
  o.rel = rel;
  o.n = vec3f(cy * nrm.x + sy * nrm.z, nrm.y, -sy * nrm.x + cy * nrm.z);
  o.info = inst.q.x;
  o.attr = attr;
  o.lp = pos;
  o.fade = bitcast<f32>(inst.q.y);
  o.sunLod = vec2f(unpack2x16float(inst.q.z).x, f32(inst.q.w));
  // Tree shadows per vertex (crowns are finely meshed; far trees lie outside the shadow map).
  o.shade = 1.0;
  if (inst.q.w <= 1u) { o.shade = treeShadowAt(u.camPos + rel + o.n * 0.3); }
  return o;
}

@vertex
fn vsTreeShadow(@location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) attr: vec4f, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let q = treePlace(pos, attr, instIn[ii]) - forest.shN.xyz;
  let R = forest.shC.w;
  return vec4f(dot(q, forest.shR.xyz) / R, dot(q, forest.shU.xyz) / R, 0.5 + 0.5 * dot(q, forest.shF.xyz) / forest.shR.w, 1.0);
}

fn mapleLeaf(tint: f32) -> array<vec3f, 2> {
  // Red, orange or gold trees, the odd one still green.
  if (tint < 0.36) { return array<vec3f, 2>(vec3f(0.28, 0.028, 0.016), vec3f(0.56, 0.1, 0.028)); }
  if (tint < 0.68) { return array<vec3f, 2>(vec3f(0.34, 0.09, 0.014), vec3f(0.64, 0.28, 0.03)); }
  if (tint < 0.92) { return array<vec3f, 2>(vec3f(0.32, 0.2, 0.018), vec3f(0.62, 0.44, 0.05)); }
  return array<vec3f, 2>(SP_LEAF0[SP_OAK], SP_LEAF1[SP_OAK]);
}

@fragment
fn fsTree(i: TreeOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  if (TREE_FADE) {
    // Level-of-detail cross-fade (its own pipeline): complementary screen-space dither, smoothed by TAA.
    let h = fract(52.9829189 * fract(dot(i.pos.xy, vec2f(0.06711056, 0.00583715))) + forest.camOff.w);
    if (i.fade >= 0.0) { if (h >= i.fade) { discard; } } else if (h < -1.0 - i.fade) { discard; }
  }
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < W_LEVEL) { return underwater(rd); }
  let part = u32(i.attr.x * 255.0 + 0.5);
  var n = normalize(i.n);
  if (!front && part == 2u) { n = -n; } // fronds are one-sided sheets seen from both sides
  let p = u.camPos + i.rel;
  let sp = i.info & 15u;
  let tint = f32((i.info >> 22u) & 63u) / 63.0;
  let ao = i.attr.z;
  let vary = i.attr.w;
  let lp = i.lp;
  var albedo: vec3f;
  var leafy = 0.0;
  var pat = 0.5;
  if (part == 1u || part == 2u) {
    var pal = array<vec3f, 2>(SP_LEAF0[sp], SP_LEAF1[sp]);
    if (sp == SP_MAPLE) { pal = mapleLeaf(tint); }
    // Painterly breakup: two octaves of noise fixed to the tree.
    pat = 0.45 * (0.5 + 0.5 * vnoise(lp.x * 2.1 + lp.y * 0.9, lp.z * 2.1 - lp.y * 0.8, 1409))
        + 0.33 * (0.5 + 0.5 * vnoise(lp.x * 5.3 - lp.y * 1.7, lp.z * 5.3 + lp.y * 1.3, 1411));
    // The finest octave only up close.
    pat += 0.22 * select(0.5, 0.5 + 0.5 * vnoise(lp.x * 11.7 + lp.y * 2.9, lp.z * 11.7 - lp.y * 2.3, 1419), i.sunLod.y < 0.5);
    albedo = mix(pal[0], pal[1], clamp(0.1 + 0.45 * vary + 0.35 * ao + 0.25 * (tint - 0.5) + 0.45 * (pat - 0.5), 0.0, 1.0));
    albedo *= 0.8 + 0.4 * pat;
    leafy = 1.0;
  } else if (part == 3u) {
    // Cactus skin: ridges lighter, a faint banding up the column.
    let band = 0.5 + 0.5 * vnoise(atan2(lp.z, lp.x) * 1.3, lp.y * 1.7, 1417);
    albedo = SP_BARK[sp] * (0.8 + 0.35 * band) * (0.9 + 0.2 * vary);
  } else if (part == 4u) {
    albedo = SP_ACCENT[sp] * (0.85 + 0.3 * vary);
  } else {
    let around = atan2(lp.z, lp.x);
    let streak = 0.5 + 0.5 * vnoise(around * 3.0, lp.y * 0.8 + vary * 9.0, 1413);
    albedo = SP_BARK[sp] * (0.72 + 0.5 * streak) * (0.9 + 0.2 * vary);
    if (sp == SP_BIRCH) {
      // Thin dark lenticels running round the trunk, a few darker scars.
      let lent = smoothstep(0.7, 0.8, 0.5 + 0.5 * vnoise(around * 2.6 + vary * 7.0, lp.y * 11.0, 1415));
      let scar = smoothstep(0.78, 0.86, 0.5 + 0.5 * vnoise(around * 1.2, lp.y * 1.6 + vary * 5.0, 1421));
      albedo = mix(albedo, vec3f(0.06, 0.055, 0.05), max(0.8 * lent, 0.6 * scar));
    } else if (sp == SP_PALM) {
      albedo *= 0.72 + 0.28 * smoothstep(0.05, 0.3, abs(fract(lp.y * 2.3) - 0.5));
    }
  }
  if ((i.info & F_SNOW) != 0u) {
    // Snow settles on whatever faces up: tier tops, branch crooks, bush crowns.
    let cover = smoothstep(0.5, 0.82, n.y + 0.55 * (pat - 0.5)) * smoothstep(0.2, 0.6, ao);
    albedo = mix(albedo, vec3f(0.82, 0.86, 0.92), cover);
    leafy *= 1.0 - cover;
  }
  // Light: a soft two-tone ramp, sun through the leaves when backlit, sky and ground fill.
  var sun = i.sunLod.x * i.shade;
  if (sun > 0.0 && i.sunLod.y < 0.5) { sun = min(sun, heroShadowAt(i.rel)); }
  let ndl = dot(n, SUN);
  let ramp = smoothstep(-0.3 * leafy - 0.05, 0.7, ndl);
  let back = pow(clamp(dot(rd, SUN), 0.0, 1.0), 3.0) * leafy * (1.0 - 0.4 * ao);
  let amb = SKY_TINT * (0.3 + 0.24 * n.y) + vec3f(0.035, 0.04, 0.02) * (1.0 - n.y);
  let fill = mix(0.55 + 0.45 * ao, 0.35 + 0.65 * ao, leafy);
  var col = albedo * (SUN_COL * sun * (0.6 * ramp + 0.32 * back) + amb * fill);
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}
`;
}
