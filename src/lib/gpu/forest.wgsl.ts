/**
 * Forest on the GPU (WGSL): placement into a cached window of tiles, per-frame
 * culling and level-of-detail selection into instance buckets, the tree
 * shaders, and a sun shadow map of the trees.
 * ---------------------------------------------------------------------------
 * forestGen     one workgroup per tile entering the window (forest/tiles.ts):
 *               each of its 8 × 8 lattice cells decides once whether it holds
 *               a plant on the land, and which: the rules of forest/placement.ts
 *               (its tables compiled in below, its steps mirrored line for
 *               line); and, in the window's second layer of slots, whether it
 *               holds one up on a sky island (skyworld/flora.ts, skyPlantAt) —
 *               so the land under an island keeps its own trees.
 * forestCull    every frame, every slot: distance, a stable thinning far out,
 *               the view frustum, the level of detail (dithered cross-fade over
 *               ±6 % of each switch distance); appends to one bucket per
 *               (species, shape, level) — the instance lists of the indexed
 *               indirect draws — and to the shadow buckets inside the shadow box.
 * forestClamp   caps each bucket's instance count at its capacity.
 * vsTree/fsTree the tree: yaw, scale, wind; its surfaces (procedural, fixed to
 *               the model): bark by species that bumps the normal, needle tufts
 *               on conifers, leaflets on fronds, cactus ribs and areoles; snow.
 *               Broadleaf crowns are dark inner foliage under their leaf cards.
 * vsCardDepth/fsCardDepth, vsCard/fsCard
 *               leaf cards round near broadleaf crowns, turned to face the camera:
 *               first their depth, a leaf spray (leafAtlas.ts) cut out by a dither
 *               of its coverage; then their colour where that depth is equal, each
 *               leaf lit by the crown's normal bent its own way (shaded once per
 *               pixel, and without discard, so hidden-surface removal works).
 * vsTreeShadow  the same transform into the sun's orthographic shadow map.
 * treeShadowAt  sun visibility under the trees (5 filtered comparison taps),
 *               used by every surface, the grass, the hero and the trees.
 * canopyAt      (canopyWGSL, compiled into the clipmap's material pass) the far
 *               forest: the canopy the trees would make, per lattice point, which
 *               the ground is painted with beyond the trees' drawing range.
 *
 * Needs, from the raster shader: u (Frame), WORLD_WGSL (hash3, vnoise,
 * elevation, temperature, moisture, W_LEVEL), the light API (keyDir, keyCol,
 * skyAmb, bounceCol: sky.wgsl.ts),
 * heightAt, sunAt, inRuinSite, project, skyShadowAt and (with the sky realm)
 * its lookups (rlFind, rlTop, …),
 * surfaceFog, display, underwater.
 */

import { LEAF_SIZE, LEAF_STYLES } from '../forest/leafAtlas';
import { SURFACE_SPAN } from '../forest/models';
import { CANOPY_AVG, PLACE, RULES, ZONES } from '../forest/placement';
import { CELL, SPECIES, SPECIES_INFO, VARIANTS, type Species } from '../forest/species';
import { SKY_BIOMES, SKY_BIOME_INFO, SKY_FLORA, SKY_FLORA_COLOURS } from '../skyworld/biomes';
import { SKY_PLACE } from '../skyworld/flora';
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
  /** Surface detail on the trees: leaf clusters, needles, leaflets, bark (else plain colours; for measuring its cost). */
  detail?: boolean;
};

export const FOREST_BYTES = 176;
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
/** Groups whose trees carry leaf cards: whole and fading trees at levels 0 and 1. */
export const CARD_GROUPS = [0, 1, 4, 5] as const;
/** Beyond this distance (m) a stable share of the plants drops out (the rest grow a little). */
const FAR_THIN = 420;

const arr = (type: string, xs: string[]) => `array<${type}, ${xs.length}>(${xs.join(', ')})`;

/**
 * Painted bark (fsTree's default bark): streaks round a tube (an integer: the
 * patterns close round it), their length (m), how mossy it grows (0–1) and the
 * furrows' depth (m).
 */
const PAINT: Partial<Record<Species, readonly [number, number, number, number]>> = {
  oak: [9, 2.2, 0.8, 0.012],
  maple: [11, 2.6, 0.3, 0.01],
  acacia: [13, 1.6, 0.12, 0.016],
  bush: [6, 0.8, 0.4, 0.004],
};
const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;

const ZONE_RULES = ZONES.map((z) => RULES[z]);
const spIndex = (x: Species) => SPECIES.indexOf(x);
const spPair = (p: readonly [Species, Species, number]) => `vec3f(${spIndex(p[0])}.0, ${spIndex(p[1])}.0, ${f(p[2])})`;

/**
 * forest/placement.ts's tables, its climate zones and its forest-body field (for a given seed), as
 * WGSL: shared by the placement (placementWGSL) and the far forest's canopy (canopyWGSL).
 */
function forestFieldsWGSL(): string {
  const Z = ZONE_RULES;
  const pair = spPair, sp = spIndex;
  return /* wgsl */ `
const Z_A: array<vec4f, ${Z.length}> = ${arr('vec4f', Z.map((r) => `vec4f(${f(r.forest)}, ${f(r.fill)}, ${f(r.shore)}, ${f(r.shrub)})`))};
const Z_G: array<vec4f, ${Z.length}> = ${arr('vec4f', Z.map((r) => `vec4f(${f(r.groveGrid)}, ${f(r.groveP)}, ${f(r.groveR[0])}, ${f(r.groveR[1])})`))};
const Z_GF: array<f32, ${Z.length}> = ${arr('f32', Z.map((r) => f(r.groveFill)))};
const Z_L: array<vec2f, ${Z.length}> = ${arr('vec2f', Z.map((r) => `vec2f(${f(r.loneGrid)}, ${f(r.loneP)})`))};
const Z_FSP: array<vec3f, ${Z.length}> = ${arr('vec3f', Z.map((r) => pair(r.forestSp)))};
const Z_GSP: array<vec3f, ${Z.length}> = ${arr('vec3f', Z.map((r) => pair(r.groveSp)))};
const Z_LSP: array<vec3f, ${Z.length}> = ${arr('vec3f', Z.map((r) => pair(r.loneSp)))};
const Z_SSP: array<u32, ${Z.length}> = ${arr('u32', Z.map((r) => `${sp(r.shoreSp)}u`))};
const PL_DESERT: i32 = ${ZONES.indexOf('desert')};
const PL_BUSH: u32 = ${sp('bush')}u;

fn plU16(h: u32, shift: u32) -> f32 { return f32((h >> shift) & 0xffffu) / 65535.0; }
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

fn forestFieldAt(x: f32, z: f32, seed: i32) -> f32 {
  let wl = ${f(PLACE.warp.wavelength)};
  let wx = x + ${f(PLACE.warp.amount)} * vnoise(x / wl, z / wl, seed + ${PLACE.warp.seed});
  let wz = z + ${f(PLACE.warp.amount)} * vnoise(x / wl + 17.3, z / wl - 9.1, seed + ${PLACE.warp.seed + 1});
  return 0.5 + 0.5 * (${PLACE.field.map(([wl, amp, off]) => `${f(amp)} * vnoise(wx / ${f(wl)}, wz / ${f(wl)}, seed + ${off})`).join(' + ')});
}
`;
}

/** forest/placement.ts as WGSL: its tables, then its steps mirrored (keep the two in step). */
function placementWGSL(islands: boolean): string {
  const Z = ZONE_RULES;
  const noise = (wl: number, off: number) => `plNoise(x, z, ${f(wl)}, ${off})`;
  return /* wgsl */ `${forestFieldsWGSL()}
fn plNoise(x: f32, z: f32, wl: f32, off: i32) -> f32 { return 0.5 + 0.5 * vnoise(x / wl, z / wl, u.seed + off); }
fn forestField(x: f32, z: f32) -> f32 { return forestFieldAt(x, z, u.seed); }

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

${islands ? skyPlacementWGSL() : 'fn skyPlantAt(ix: i32, iz: i32, lid: vec2f, corner: vec2f, bits: u32) -> vec4u { return vec4u(0u); }'}
`;
}

/**
 * The forest's sky layer (skyworld/flora.ts as WGSL; keep the two in step): the plant of
 * lattice cell (ix, iz) up on a sky island, if any — its record as plRecord's, flagged
 * F_ISLAND with the island's height, its sky palette in bits 25–27 and a 3-bit tint in 22–24.
 */
function skyPlacementWGSL(): string {
  const sp = (x: Species) => `${SPECIES.indexOf(x)}u`;
  const T = SKY_BIOMES.map((b) => SKY_BIOME_INFO[b].trees);
  const kinds = T.map((t) => [0, 1, 2].map((n) => t.kinds[n] ?? { ...t.kinds[t.kinds.length - 1], weight: 0 }));
  const pal = (p: (typeof SKY_FLORA)[number]) => `${SKY_FLORA.indexOf(p)}u`;
  return /* wgsl */ `
const SKY_RULES: array<vec4f, ${T.length}> = ${arr('vec4f', T.map((t) => `vec4f(${f(t.forest)}, ${f(t.lone)}, ${f(t.bush)}, 0.0)`))};
const SKY_BUSHPAL: array<u32, ${T.length}> = ${arr('u32', T.map((t) => pal(t.bushPalette)))};
const SKY_KIND_SP: array<vec3u, ${T.length}> = ${arr('vec3u', kinds.map((ks) => `vec3u(${ks.map((x) => sp(x.species)).join(', ')})`))};
const SKY_KIND_PAL: array<vec3u, ${T.length}> = ${arr('vec3u', kinds.map((ks) => `vec3u(${ks.map((x) => pal(x.palette)).join(', ')})`))};
const SKY_KIND_W: array<vec3f, ${T.length}> = ${arr('vec3f', kinds.map((ks) => `vec3f(${ks.map((x) => f(x.weight)).join(', ')})`))};
const SKY_KIND_LO: array<vec3f, ${T.length}> = ${arr('vec3f', kinds.map((ks) => `vec3f(${ks.map((x) => f(x.scale[0])).join(', ')})`))};
const SKY_KIND_HI: array<vec3f, ${T.length}> = ${arr('vec3f', kinds.map((ks) => `vec3f(${ks.map((x) => f(x.scale[1])).join(', ')})`))};
const SKY_KIND_SNOW: array<vec3u, ${T.length}> = ${arr('vec3u', kinds.map((ks) => `vec3u(${ks.map((x) => (x.snow ? '1u' : '0u')).join(', ')})`))};

fn skyRecord(sp: u32, pal: u32, off: vec2f, y: f32, scale: f32, snow: bool, bits: u32) -> vec4u {
  let sb = u32(clamp(round((scale - 0.55) / 1.05 * 255.0), 0.0, 255.0));
  let info = sp | (((bits & 0xffffu) % F_VARIANTS) << 4u) | (((bits >> 16u) & 255u) << 6u) | (sb << 14u)
    | (((bits >> 8u) & 7u) << 22u) | ((pal & 7u) << 25u) | select(0u, F_SNOW, snow) | F_ISLAND | F_VALID;
  return vec4u(bitcast<u32>(off.x), bitcast<u32>(off.y), bitcast<u32>(y), info);
}

fn skyPlantAt(ix: i32, iz: i32, lid: vec2f, corner: vec2f, bits: u32) -> vec4u {
  let h0 = hash3(ix, iz, u.seed + 7301);
  let h1 = hash3(ix, iz, u.seed + 7302);
  let h2 = hash3(ix, iz, u.seed + 7303);
  var jit = vec2f(0.08 + 0.84 * plU16(h0, 0u), 0.08 + 0.84 * plU16(h0, 16u));
  let k = rlFind(corner + jit * F_CELL, true);
  if (k < 0) { return vec4u(0u); }
  let ku = u32(k);
  let b = realm.isl[ku].m.y;
  let R = SKY_RULES[b];
  let keep = plU16(h1, 0u) / max(forest.camOff.z, 1e-6);
  let pick = plU16(h1, 16u);
  let bush = plU16(h2, 0u) / max(forest.camOff.z, 1e-6);
  // Lone tree: this cell hosts its coarse cell's spot?
  let kl = ${SKY_PLACE.lone};
  let lx = plFdiv(ix, kl);
  let lz = plFdiv(iz, kl);
  let lh = hash3(lx, lz, u.seed + 7311);
  let lp = hash3(lx, lz, u.seed + 7312);
  let fx = (0.15 + 0.7 * plU16(lh, 0u)) * f32(kl);
  let fz = (0.15 + 0.7 * plU16(lh, 16u)) * f32(kl);
  let lone = ix == lx * kl + i32(floor(fx)) && iz == lz * kl + i32(floor(fz)) && plU16(lp, 0u) < R.y * forest.camOff.z;
  if (lone) { jit = vec2f(fract(fx), fract(fz)); }
  let xz = corner + jit * F_CELL;
  let off = (lid + jit) * F_CELL;
  // Where nothing grows: near the rim, in or by the water, on the steep.
  let q = xz - realm.isl[ku].c.xy;
  if (rlRho(ku, q) > ${f(SKY_PLACE.rim)}) { return vec4u(0u); }
  if ((realm.isl[ku].m.z & RL_LAKE) != 0u && rlLakeDist(ku, q) < ${f(SKY_PLACE.lake)}) { return vec4u(0u); }
  if (rlInStream(ku, q)) {
    let n = rlStream(ku, q);
    if (n.x < ${f(SKY_PLACE.stream)} * n.z + 1.0) { return vec4u(0u); }
  }
  let y = rlTop(ku, q);
  let slope = length(vec2f(rlTop(ku, q + vec2f(1.5, 0.0)) - y, rlTop(ku, q + vec2f(0.0, 1.5)) - y)) / 1.5;
  if (slope > ${f(SKY_PLACE.slope)}) { return vec4u(0u); }
  let s = rlSeed(ku);
  let field = 0.5 + 0.5 * vnoise(q.x / ${f(SKY_PLACE.field)}, q.y / ${f(SKY_PLACE.field)}, s + 11);
  if (lone || keep < R.x * smoothstep(0.45, 0.7, field)) {
    // Which of the biome's trees: its stand, mixed a little at the edges.
    let stand = 0.5 + 0.5 * vnoise(q.x / ${f(SKY_PLACE.stand)}, q.y / ${f(SKY_PLACE.stand)}, s + 13);
    let v = clamp(stand + 0.3 * (pick - 0.5), 0.0, 0.9999);
    let W = SKY_KIND_W[b];
    var c = 2u;
    if (v < W.x) { c = 0u; } else if (v < W.x + W.y) { c = 1u; }
    let fr = fract(pick * 7.13);
    let scale = (SKY_KIND_LO[b][c] + (SKY_KIND_HI[b][c] - SKY_KIND_LO[b][c]) * fr) * select(1.0, 1.15, lone);
    return skyRecord(SKY_KIND_SP[b][c], SKY_KIND_PAL[b][c], off, y, scale, SKY_KIND_SNOW[b][c] == 1u, bits);
  }
  if (bush < R.z) { return skyRecord(PL_BUSH, SKY_BUSHPAL[b], off, y, 0.8 + 0.4 * pick, BI_SNOW[b] > 0.5, bits); }
  return vec4u(0u);
}
`;
}

/**
 * The far forest's look per species: its canopy's albedo seen from afar (its leaves' two tones,
 * darkened by the shade inside and between the crowns), and the share of the ground one tree's crown
 * covers, in cells (a crown fills about this much of its bounding disc).
 */
const CANOPY_LOOK = { dark: 0.6, shade: 0.8, fill: 0.7, snow: [0.62, 0.66, 0.72], snowCover: 0.45 } as const;

/**
 * The far forest (placement.ts canopyFrom, mirrored): where the renderer no longer draws trees it
 * paints the canopy they would make onto the ground. Run once per clipmap lattice point, after its
 * height (terrain.wgsl.ts horizonShader / sunShader); needs WORLD_WGSL (vnoise, temperature, W_LEVEL).
 */
export function canopyWGSL(): string {
  const S = SPECIES.map((s) => SPECIES_INFO[s]);
  const L = CANOPY_LOOK, A = CANOPY_AVG;
  const noise = (wl: number, off: number) => `(0.5 + 0.5 * vnoise(g.x / ${f(wl)}, g.y / ${f(wl)}, seed + ${off}))`;
  return /* wgsl */ `${forestFieldsWGSL()}
const CN_COL: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3([0, 1, 2].map((c) => L.shade * (L.dark * s.leaf[0][c] + (1 - L.dark) * s.leaf[1][c])))))};
const CN_AREA: array<f32, ${S.length}> = ${arr('f32', S.map((s) => f((L.fill * Math.PI * s.radius * s.radius) / (CELL * CELL))))};

// The far forest at world g, from the ground the clipmap holds there (height e, slope, the relief
// over the hills' step and its slope, moisture M): its canopy's albedo, premultiplied by its cover
// and stored as square roots (8 bits keep the dark greens), and its cover, 1 − e^(−trees per cell ×
// cells under one crown).
fn canopyAt(g: vec2f, e: f32, slope: f32, hills: f32, hillSlope: f32, M: f32, seed: i32) -> vec4f {
  if (e < W_LEVEL + 0.4) { return vec4f(0.0); }
  let T = temperature(g, e);
  var zw = zoneWeights(T, M, ${noise(PLACE.region.wavelength, PLACE.region.seed)});
  var total = 0.0;
  var level = 0.0;
  var fill = 0.0;
  var dom = 0;
  for (var k = 0; k < ${ZONES.length}; k++) {
    total += zw[k];
    if (zw[k] > zw[dom]) { dom = k; }
    level += zw[k] * Z_A[k].x;
    fill += zw[k] * Z_A[k].y;
  }
  if (total < 1e-4) { return vec4f(0.0); }
  level /= total;
  fill /= total;
  let suit = 0.1 * smoothstep(0.02, 0.12, hills) - 0.12 * smoothstep(0.02, 0.12, -hills) + 0.15 * (M - 0.62) - 0.3 * smoothstep(0.55, 0.9, hillSlope);
  let mask = smoothstep(level - ${f(PLACE.edge)}, level + ${f(PLACE.edge)}, forestFieldAt(g.x, g.y, seed) + suit);
  let open = 1.0 - smoothstep(0.15, 0.4, mask);
  let glade = smoothstep(${f(PLACE.glade.lo)}, ${f(PLACE.glade.hi)}, ${noise(PLACE.glade.wavelength, PLACE.glade.seed)});
  let park = ${f(PLACE.parkland.lo)} + ${f(PLACE.parkland.span)} * smoothstep(0.3, 0.7, ${noise(PLACE.parkland.wavelength, PLACE.parkland.seed)});
  let G = Z_G[dom];
  let grove = G.y * park * ${f(A.groveHollow)} * open * Z_GF[dom] * ${f(A.groveShape * Math.PI / 3)} * (G.z * G.z + G.z * G.w + G.w * G.w) / (G.x * G.x)
    * (1.0 - smoothstep(0.5, 0.9, slope));
  let lc = ${f(CELL)} / Z_L[dom].x;
  let lone = Z_L[dom].y * park * ${f(PLACE.lone.flat + A.knolls * PLACE.lone.knoll)} * (1.0 - smoothstep(0.35, 0.6, slope)) * open * lc * lc;
  let forest = fill * pow(mask, 1.4) * (1.0 - ${f(PLACE.glade.cut)} * glade) * (1.0 - smoothstep(0.8, 1.15, slope));
  let sp = plPair(Z_FSP[dom], ${noise(PLACE.stand.wavelength, PLACE.stand.seed)});
  let cover = 1.0 - exp(-(forest + grove + lone) * CN_AREA[sp]);
  var col = CN_COL[sp];
  if (T < 0.5 && (sp == ${spIndex('pine')}u || sp == ${spIndex('dead')}u || sp == ${spIndex('bush')}u || sp == ${spIndex('birch')}u)) {
    col = mix(col, ${v3(L.snow)}, ${f(L.snowCover)});
  }
  return vec4f(sqrt(col) * cover, cover);
}
`;
}

export function forestWGSL(spec: ForestSpec, bounds: ReadonlyArray<{ top: number; radius: number }>, islands = false): string {
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
const SP_JUNGLE: u32 = ${sp('jungle')};
const SP_PALM: u32 = ${sp('palm')};
const SP_CACTUS: u32 = ${sp('cactus')};
const SP_DEAD: u32 = ${sp('dead')};
const SP_BUSH: u32 = ${sp('bush')};
const F_CAP: array<u32, ${GROUPS}> = ${arr('u32', CAP.map((c) => `${c}u`))};
const F_REGION: array<u32, ${GROUPS}> = ${arr('u32', REGION.map((c) => `${c}u`))};
override TREE_FADE: bool = false;
override TREE_DETAIL: bool = true;
const SP_TOP: array<f32, ${S.length}> = ${arr('f32', bounds.map((b) => f(b.top)))};
const SP_RAD: array<f32, ${S.length}> = ${arr('f32', bounds.map((b) => f(b.radius)))};
const SP_MAXD: array<f32, ${S.length}> = ${arr('f32', S.map((s) => f(s.maxDist)))};
const SP_LOD: array<vec2f, ${S.length}> = ${arr('vec2f', S.map((s) => `vec2f(${f(s.lod[0])}, ${f(s.lod[1])})`))};
const SP_LEAF0: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.leaf[0])))};
const SP_LEAF1: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.leaf[1])))};
const SP_BARK: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.bark)))};
const SP_ACCENT: array<vec3f, ${S.length}> = ${arr('vec3f', S.map((s) => v3(s.accent)))};
const SP_CLUSTER: array<f32, ${S.length}> = ${arr('f32', S.map((s) => f(s.cluster)))};
const SP_GLOSS: array<f32, ${S.length}> = ${arr('f32', S.map((s) => f(s.gloss)))};
const SP_PAINT: array<vec4f, ${S.length}> = ${arr('vec4f', SPECIES.map((s) => `vec4f(${(PAINT[s] ?? PAINT.oak!).map(f).join(', ')})`))};
const F_SPAN: f32 = ${f(SURFACE_SPAN)};
const SP_CARD: array<f32, ${S.length}> = ${arr('f32', S.map((s) => f(s.card)))};
const SP_SPRAY: array<i32, ${S.length}> = ${arr('i32', S.map((s) => `${LEAF_STYLES.indexOf(s.spray)}`))};

struct ForestU {
  camTile: vec4i,  // camera's tile x, z (absolute); unused
  camOff: vec4f,   // camera offset in its tile x, z (m); density; dither offset
  shC: vec4f,      // centre of the shadow map sampled now (rendered last frame; camera-relative); half width (m)
  shR: vec4f,      // light right axis; half depth (m)
  shU: vec4f,      // light up axis; texel (m)
  shF: vec4f,      // light forward axis (away from the sun); unused
  wind: vec4f,     // direction x, z; strength (m of sway at the tips); time
  shN: vec4f,      // centre of the shadow map rendered now (camera-relative); unused
  sR: vec4f,       // light axes of the map sampled now (the light may have moved since it was drawn)
  sU: vec4f,
  sF: vec4f,
};
@group(0) @binding(8) var<uniform> forest: ForestU;
@group(0) @binding(9) var shadowTex: texture_depth_2d;
@group(0) @binding(16) var shadowSamp: sampler_comparison;
@group(0) @binding(14) var leafTex: texture_2d_array<f32>;
@group(0) @binding(15) var leafSamp: sampler;

// ---- tree shadows --------------------------------------------------------------

fn treeShadowAt(p: vec3f) -> f32 {${spec.shadowRes ? '' : ' return 1.0;'}
  let q = p - u.camPos - forest.shC.xyz;
  let R = forest.shC.w;
  let ls = vec3f(dot(q, forest.sR.xyz), dot(q, forest.sU.xyz), dot(q, forest.sF.xyz));
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
${placementWGSL(islands)}
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
  // The cell's corner, chunk-local.
  let corner = u.camPos.xz + vec2f(job.xy - forest.camTile.xy) * F_TILE + vec2f(lid.xy) * F_CELL - forest.camOff.xy;
  let i = slot * 64u + lid.y * 8u + lid.x;
  forestSlots[i] = plantAt(cell.x, cell.y, vec2f(lid.xy), corner + jit * F_CELL, hash3(cell.x, cell.y, u.seed + 7104));
  forestSlots[F_SLOTS + i] = skyPlantAt(cell.x, cell.y, vec2f(lid.xy), corner, hash3(cell.x, cell.y, u.seed + 7304));
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
  if (id.x >= 2u * F_SLOTS) { return; }
  let rec = forestSlots[id.x];
  let info = rec.w;
  if ((info & F_VALID) == 0u) { return; }
  let i = id.x % F_SLOTS; // (the second layer: the sky's)
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
  // (leaf cards reach past the model's bounds by about a card)
  let rad = length(vec2f(0.5 * top, SP_RAD[sp] * s)) + 1.1 * SP_CARD[sp] * s;
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
  let sun = select(sunAt(xz), 1.0, (info & F_ISLAND) != 0u) * skyShadowAt(crown);
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
  let n = min(atomicLoad(&forestArgs[b * 5u + 1u]), F_CAP[b / F_KINDS]);
  atomicStore(&forestArgs[b * 5u + 1u], n);
  // The same instances draw the bucket's leaf cards (their arguments follow every bucket's).
  atomicStore(&forestArgs[(${GROUPS}u * F_KINDS + b) * 5u + 1u], n);
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
  @location(6) @interpolate(flat) inst: vec4f, // mountain × island sun at the tree, level of detail, scale, yaw
  @location(7) shade: f32,    // sun under the trees (the shadow map), per vertex
  @location(8) uv: vec2f,     // surface coordinates: round (0 → 1) and metres along a tube; side and metres along a frond
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
fn vsTree(@location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) attr: vec4f, @location(3) uv: vec2f, @builtin(instance_index) ii: u32) -> TreeOut {
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
  o.inst = vec4f(unpack2x16float(inst.q.z).x, f32(inst.q.w), inst.p.w, yaw);
  o.uv = vec2f(uv.x, uv.y * F_SPAN);
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

// Up on a sky island (F_ISLAND) a tree takes its sky palette (bits 25–27) and a coarser tint (bits 22–24).
const SKY_LEAF0: array<vec3f, ${SKY_FLORA.length}> = ${arr('vec3f', SKY_FLORA.map((p) => v3(SKY_FLORA_COLOURS[p].leaf[0])))};
const SKY_LEAF1: array<vec3f, ${SKY_FLORA.length}> = ${arr('vec3f', SKY_FLORA.map((p) => v3(SKY_FLORA_COLOURS[p].leaf[1])))};
const SKY_BARK: array<vec3f, ${SKY_FLORA.length}> = ${arr('vec3f', SKY_FLORA.map((p) => v3(SKY_FLORA_COLOURS[p].bark)))};
fn infoTint(info: u32) -> f32 {
  if ((info & F_ISLAND) != 0u) { return f32((info >> 22u) & 7u) / 7.0; }
  return f32((info >> 22u) & 63u) / 63.0;
}
fn leafPal(info: u32, sp: u32, tint: f32) -> array<vec3f, 2> {
  if ((info & F_ISLAND) != 0u) {
    let k = (info >> 25u) & 7u;
    return array<vec3f, 2>(SKY_LEAF0[k], SKY_LEAF1[k]);
  }
  if (sp == SP_MAPLE) { return mapleLeaf(tint); }
  return array<vec3f, 2>(SP_LEAF0[sp], SP_LEAF1[sp]);
}
fn barkCol(info: u32, sp: u32) -> vec3f {
  if ((info & F_ISLAND) != 0u) { return SKY_BARK[(info >> 25u) & 7u]; }
  return SP_BARK[sp];
}

// ---- surface detail -------------------------------------------------------------------
// All procedural and fixed to the model, so it sways with the tree. Each pattern fades to
// its mean where its cells would shrink below a few pixels (smaller, they would shimmer).
// Leaves bend the normal directly; bark and fronds give their relief's slopes along the
// surface coordinates, which bump the normal smoothly (no screen-space differences of the
// pattern itself, which step in 2 × 2 pixel blocks). Hashes are float-only (integer
// multiplies are slow on the GPU); every lattice here has small coordinates.

// Hashes without sine (D. Hoskins): values in [0, 1) at lattice points.
fn hash12(p: vec2f) -> f32 {
  var q = fract(vec3f(p.xyx) * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
fn hash22(p: vec2f) -> vec2f {
  var q = fract(vec3f(p.xyx) * vec3f(0.1031, 0.103, 0.0973));
  q += dot(q, q.yzx + 33.33);
  return fract((q.xx + q.yz) * q.zy);
}
fn hash33(p: vec3f) -> vec3f {
  var q = fract(p * vec3f(0.1031, 0.103, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.xxy + q.yxx) * q.zyx);
}

// Value noise in [0, 1] and its gradient (.yz), periodic in x over \`per\` (round a tube).
fn pnoiseG(p: vec2f, per: f32) -> vec3f {
  let b = floor(p);
  let f = p - b;
  let w = f * f * (3.0 - 2.0 * f);
  let dw = 6.0 * f * (1.0 - f);
  let x0 = b.x - per * floor(b.x / per);
  let x1 = select(x0 + 1.0, 0.0, x0 + 1.0 >= per);
  let v = vec4f(hash12(vec2f(x0, b.y)), hash12(vec2f(x1, b.y)), hash12(vec2f(x0, b.y + 1.0)), hash12(vec2f(x1, b.y + 1.0)));
  let ab = mix(v.x, v.y, w.x);
  let cd = mix(v.z, v.w, w.x);
  return vec3f(mix(ab, cd, w.y), dw.x * mix(v.y - v.x, v.w - v.z, w.y), dw.y * (cd - ab));
}

fn pnoise(p: vec2f, per: f32) -> f32 { return pnoiseG(p, per).x; }

// smoothstep of a value carrying its gradient (.yz).
fn smoothG(a: f32, b: f32, x: vec3f) -> vec3f {
  let t = clamp((x.x - a) / (b - a), 0.0, 1.0);
  return vec3f(t * t * (3.0 - 2.0 * t), (6.0 * t * (1.0 - t) / (b - a)) * x.yz);
}

struct Cell2 { d1: vec2f, d2: vec2f, cell: vec2f };
// 2D cellular noise, periodic in x over \`per\` cells (round a tube): offsets from the nearest two
// feature points, and the nearest's cell (for its random values).
fn cell2(p: vec2f, per: f32) -> Cell2 {
  let b = floor(p);
  let f = p - b;
  var f1 = 9.0;
  var f2 = 9.0;
  var o = Cell2(vec2f(0.0), vec2f(0.0), vec2f(0.0));
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let cx = b.x + f32(x);
      let cw = vec2f(cx - per * floor(cx / per), b.y + f32(y));
      let d = f - vec2f(f32(x), f32(y)) - hash22(cw);
      let dd = dot(d, d);
      if (dd < f1) { f2 = f1; o.d2 = o.d1; f1 = dd; o.d1 = d; o.cell = cw; } else if (dd < f2) { f2 = dd; o.d2 = d; }
    }
  }
  return o;
}

// F2 − F1 (0 on the border between two cells) and its gradient, in cells.
fn edgeG(c: Cell2) -> vec3f {
  let f1 = max(length(c.d1), 1e-5);
  let f2 = max(length(c.d2), 1e-5);
  return vec3f(f2 - f1, c.d2 / f2 - c.d1 / f1);
}

// Share of a pattern to keep: 1 while its cells (per round, metres long) span over ~5 pixels, 0 below ~2.
fn keep(fw: vec2f, per: f32, len: f32) -> f32 {
  return 1.0 - smoothstep(0.2, 0.45, max(fw.x * per, fw.y / len));
}

// Bump a normal by a relief given as its slopes along the surface coordinates (world metres per
// unit round, per metre along): Mikkelsen's surface gradient (2010), from screen-space derivatives
// of the position and of the coordinates, both smooth over a triangle. Uniform control flow only.
fn bumped(n: vec3f, pos: vec3f, uv: vec2f, g: vec2f) -> vec3f {
  let dpx = dpdx(pos);
  let dpy = dpdy(pos);
  let r1 = cross(dpy, n);
  let r2 = cross(n, dpx);
  let det = dot(dpx, r1);
  let b = abs(det) * n - sign(det) * (dot(g, dpdx(uv)) * r1 + dot(g, dpdy(uv)) * r2);
  return select(n, normalize(b), abs(det) > 1e-12 && dot(b, n) > 0.0);
}

// World axes from model axes (the tree's yaw: rot = (cos, sin)), and back.
fn toWorld(v: vec3f, rot: vec2f) -> vec3f { return vec3f(rot.x * v.x + rot.y * v.z, v.y, -rot.y * v.x + rot.x * v.z); }
fn toModel(v: vec3f, rot: vec2f) -> vec3f { return vec3f(rot.x * v.x - rot.y * v.z, v.y, rot.y * v.x + rot.x * v.z); }

// Squeeze space e times along the unit axis \`ax\` (round domes there are drawn out e times along it
// here), and the normal that goes with it.
fn squeeze(q: vec3f, ax: vec3f, e: f32) -> vec3f { return q - (1.0 - 1.0 / e) * ax * dot(q, ax); }
fn squeezeN(n: vec3f, ax: vec3f, e: f32) -> vec3f { return normalize(n + (e - 1.0) * ax * dot(n, ax)); }

struct Dome { t: vec3f, top: f32, top2: f32, cell: vec3f };
// Leaf clusters: spheres round jittered points of a 3D lattice, seen where they break the crown's
// surface; at each point the one standing highest covers it, so clusters overlap like scales, with
// round edges. Points keep to the middle 40 % of their cells and radii to 0.6–0.8 cells, so the
// 2 × 2 × 2 cells nearest a point hold every sphere that can reach it. \`n\`: the surface normal in
// lattice axes. Returns the covering sphere's offset along the surface (in radii), how high it and
// the next stand over the surface (−9: none), and its cell.
fn domes(q: vec3f, n: vec3f) -> Dome {
  var o = Dome(vec3f(0.0), -9.0, -9.0, vec3f(0.0));
  let b = floor(q - 0.5);
  for (var k = 0u; k < 8u; k++) {
    let c = b + vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32(k >> 2u));
    let hv = hash33(c);
    let d = q - c - 0.3 - 0.4 * hv;
    let R = 0.6 + 0.2 * hv.z;
    let r2 = R * R - dot(d, d);
    // Only a sphere round the point reaches the surface here (most don't: test that first).
    if (r2 > 0.0) {
      let along = dot(d, n);
      let t = d - n * along;
      let top = sqrt(r2 + along * along) - along;
      if (top > o.top) { o.top2 = o.top; o.top = top; o.t = t / R; o.cell = c; } else if (top > o.top2) { o.top2 = top; }
    }
  }
  return o;
}

struct Leaves { n: vec3f, tone: f32, shift: f32, occl: f32, rim: f32, sun: f32 };

// Occlusion of a dome: dark in the crease where it meets the next, a little toward its rim; a gap
// (nothing covers the point) looks deep into the crown.
fn domeOccl(d: Dome, crease: f32) -> f32 {
  if (d.top < -1.0) { return 0.4; }
  return mix(1.0 - crease, 1.0, smoothstep(0.0, 0.25, d.top - d.top2)) * (1.0 - 0.3 * smoothstep(0.7, 1.0, length(d.t)));
}

// A conifer's skirts: needle tufts (domes drawn out down the slope, away from the trunk), lighter
// at their tips; close up, bundles of needles within them and needles streaking along them.
fn needles(lp: vec3f, nw: vec3f, rot: vec2f, sp: u32, px: f32) -> Leaves {
  var o = Leaves(nw, 1.0, 0.0, 1.0, 0.0, 1.0);
  let size = SP_CLUSTER[sp];
  let k = size / max(px, 1e-6); // a tuft's size in pixels
  let detail = smoothstep(2.5, 6.0, k);
  // (the flat, dark underside closing each skirt stays plain)
  if (detail <= 0.0 || nw.y < -0.95) { return o; }
  let n = toModel(nw, rot);
  let out = normalize(vec3f(lp.x, 0.0, lp.z) + vec3f(1e-3, 0.0, 0.0));
  let down = normalize(out - vec3f(0.0, 0.9, 0.0));
  let dm = domes(squeeze(lp / size, down, 2.0), squeezeN(n, down, 2.0));
  let covered = dm.top > -1.0;
  let r = hash33(dm.cell + 19.19);
  var tilt = vec3f(0.0);
  if (covered) { tilt = 1.2 * squeeze(dm.t, down, 2.0) + 0.4 * (r - 0.5); }
  o.n = normalize(nw + detail * toWorld(tilt, rot));
  let oc = domeOccl(dm, 0.5);
  o.occl = mix(1.0, oc, detail);
  o.sun = mix(1.0, select(0.2, 0.35 + 0.65 * oc, covered), detail);
  let tip = smoothstep(0.0, 0.9, dot(dm.t, down)) * select(0.0, 1.0, covered);
  o.tone = 1.0 + detail * (0.3 * r.x - 0.15 + 0.25 * tip);
  o.shift = detail * (0.35 * (r.y - 0.5) + 0.3 * tip);
  o.rim = detail * tip;
  let fine = detail * smoothstep(24.0, 48.0, k) * smoothstep(-0.5, -0.3, nw.y);
  if (fine > 0.0 && covered) {
    let nb = domes(squeeze(lp / size * 2.6 + 3.7, down, 2.6), squeezeN(n, down, 2.6));
    let r2 = hash33(nb.cell + 43.7);
    var t2 = vec3f(0.0);
    if (nb.top > -1.0) { t2 = 0.5 * squeeze(nb.t, down, 2.6) + 0.3 * (r2 - 0.5); }
    o.n = normalize(o.n + fine * toWorld(t2, rot));
    o.occl *= mix(1.0, domeOccl(nb, 0.3), fine);
    o.tone *= 1.0 + fine * (0.2 * r2.x - 0.1 + 0.15 * smoothstep(0.2, 0.9, dot(nb.t, down)));
  }
  // Needles: fine streaks down the tufts, where they span a few pixels (periodic round the tree).
  let rho = length(lp.xz);
  let streak = detail * smoothstep(2.0, 4.0, 6.2831853 * max(rho, 0.3) / 150.0 / max(px, 1e-6));
  if (streak > 0.0) {
    let v = pnoise(vec2f((atan2(lp.z, lp.x) * (1.0 / 6.2831853) + 0.5) * 150.0, (rho - 0.9 * lp.y) * 2.0), 150.0);
    o.tone *= 1.0 + streak * (0.3 * v - 0.15);
  }
  return o;
}

struct Frond { g: vec2f, tone: f32, shift: f32, occl: f32 };
// Palm fronds: leaflets slanting toward the tip, parted by dark slits, a pale midrib; the relief's
// slopes (per unit across, per metre along) fold each leaflet.
fn frond(uv: vec2f, fw: vec2f) -> Frond {
  var o = Frond(vec2f(0.0), 1.0, 0.0, 1.0);
  let side = abs(uv.x - 0.5) * 2.0; // 0 on the midrib, 1 at the edge
  let d = keep(fw, 3.0, 0.14);
  if (d <= 0.0) { return o; }
  let t = uv.y / 0.14 - 2.2 * side;
  let f = fract(t);
  let r = hash12(vec2f(floor(t), 3.1));
  let slit = (1.0 - smoothstep(0.0, 0.08 + 0.1 * side, min(f, 1.0 - f))) * smoothstep(0.06, 0.16, side);
  let rib = 1.0 - smoothstep(0.02, 0.05, side);
  o.occl = 1.0 - 0.7 * slit * d;
  o.tone = 1.0 + d * (0.24 * r - 0.12 + 0.2 * rib);
  o.shift = d * (0.35 * rib - 0.15 * slit);
  // Relief 0.012 · sin(π f) across each leaflet: its slope along t, then along the coordinates.
  let dh = d * 0.012 * 3.14159 * cos(3.14159 * f) * (1.0 - rib);
  o.g = dh * vec2f(-4.4 * sign(uv.x - 0.5), 1.0 / 0.14);
  return o;
}

struct Bark { albedo: vec3f, h: vec3f };
// Bark by species over the tube's surface coordinates (round 0 → 1, metres along): its colour, and
// its relief (m) with the relief's slopes per unit round and per metre along. \`fw\`: how fast those
// coordinates change per pixel; \`up\`: how much the surface faces up (moss grows there).
fn bark(sp: u32, base: vec3f, uv: vec2f, lp: vec3f, fw: vec2f, up: f32) -> Bark {
  var o = Bark(base, vec3f(0.0));
  let s = uv.y;
  switch sp {
    case SP_BIRCH: {
      // Chalk-white bark banded faintly, thin dark lenticels running round it, dark scars, a black cracked foot.
      var col = base * (0.93 + 0.14 * pnoise(vec2f(uv.x * 5.0, s * 1.7), 5.0));
      let dl = keep(fw, 7.0, 1.0 / 16.0);
      if (dl > 0.0) {
        // At most one lenticel per cell of a 7 round × 16 per metre grid, kept inside its cell.
        let g = vec2f(uv.x * 7.0, s * 16.0);
        let gc = floor(g);
        let gw = vec2f(gc.x - 7.0 * floor(gc.x / 7.0), gc.y);
        let r = hash33(vec3f(gw, 1.0));
        let e = length((g - gc - vec2f(0.35 + 0.3 * r.x, 0.35 + 0.3 * r.y)) / vec2f(0.1 + 0.2 * r.z, 0.1));
        let dash = (1.0 - smoothstep(0.7, 1.0, e)) * step(hash12(gw + 17.3), 0.75);
        // (fading out, the dashes leave their mean: a faint grey)
        col = mix(col, vec3f(0.05, 0.045, 0.04), 0.85 * mix(0.05, dash, dl));
      }
      let ds = keep(fw, 3.0, 0.6);
      if (ds > 0.0) {
        // Scars: dark eyes, wide and pointed, on most cells of a 3 round × 1.6 per metre grid (they carry the look far off).
        let g = vec2f(uv.x * 3.0, s * 1.6);
        let gc = floor(g);
        let gw = vec2f(gc.x - 3.0 * floor(gc.x / 3.0), gc.y);
        let r = hash33(vec3f(gw, 2.0));
        let f = (g - gc - vec2f(0.35 + 0.3 * r.x, 0.3 + 0.4 * r.y)) / vec2f(0.2 + 0.12 * r.z, 0.09);
        let e = length(f) + 0.6 * abs(f.x) * abs(f.y);
        let scar = (1.0 - smoothstep(0.8, 1.0, e)) * step(hash12(gw + 5.1), 0.75);
        col = mix(col, vec3f(0.035, 0.03, 0.028), 0.9 * mix(0.1, scar, ds));
      }
      // The foot: dark, rough and fissured, up to a ragged line.
      let foot = 1.0 - smoothstep(0.4, 1.5, lp.y + 0.7 * pnoise(vec2f(uv.x * 6.0, s * 0.8), 6.0));
      if (foot > 0.0) {
        let pl = smoothG(0.02, 0.25, edgeG(cell2(vec2f(uv.x * 9.0, s / 0.26), 9.0)));
        let k = foot * keep(fw, 9.0, 0.26);
        col = mix(col, vec3f(0.075, 0.07, 0.065) * (0.55 + 0.9 * pl.x * k), foot);
        o.h = 0.02 * k * vec3f(pl.x, pl.y * 9.0, pl.z / 0.26);
      }
      o.albedo = col;
    }
    case SP_PINE: {
      // Scaly plates parted by dark fissures, grey-brown below, smoother and orange up the trunk.
      let up = smoothstep(0.3, 0.75, lp.y / SP_TOP[sp]);
      var col = mix(vec3f(0.15, 0.095, 0.065), vec3f(0.42, 0.2, 0.085), up);
      let d = keep(fw, 10.0, 0.45);
      if (d > 0.0) {
        let warp = 0.45 * (pnoise(vec2f(uv.x * 10.0, s / 0.45 * 0.6), 10.0) - 0.5);
        let c = cell2(vec2f(uv.x * 10.0 + warp, s / 0.45), 10.0);
        let r = hash33(vec3f(c.cell, 3.0));
        let e = edgeG(c);
        let pl = smoothG(0.02, 0.1 + 0.2 * r.z, e);
        let k = d * (1.0 - 0.75 * up);
        // Plates flake in layers: a redder, paler rim round each, fine cracks across them.
        let flake = smoothstep(0.4, 0.15, e.x) * pl.x;
        let cracks = smoothstep(0.62, 0.8, pnoise(vec2f(uv.x * 40.0, s * 9.0), 40.0));
        col *= 1.0 + d * (0.4 * r.x - 0.2);
        col = mix(col, vec3f(0.3, 0.14, 0.07) * (0.8 + 0.4 * r.y), 0.55 * flake * k);
        col *= 1.0 - 0.35 * cracks * pl.x * k;
        col = mix(col, vec3f(0.045, 0.035, 0.03), 0.9 * (1.0 - pl.x) * k);
        o.h = 0.035 * k * vec3f(pl.x, pl.y * 10.0, pl.z / 0.45);
      }
      o.albedo = col;
    }
    case SP_PALM: {
      // Ring scars where old fronds fell, the band between each pair swelling; fibres running up.
      let df = keep(fw, 36.0, 0.4);
      let fib = pnoiseG(vec2f(uv.x * 36.0, s * 2.5), 36.0);
      let fv = mix(0.5, fib.x, df);
      var col = base * (0.85 + 0.3 * fv);
      o.h = 0.004 * df * vec3f(fib.x - 0.5, fib.y * 36.0, fib.z * 2.5);
      let dr = 1.0 - smoothstep(0.2, 0.45, fw.y / 0.17);
      if (dr > 0.0) {
        let t = s / 0.17 + 0.35 * pnoise(vec2f(uv.x * 4.0, s * 0.9), 4.0);
        let f = fract(t);
        let m = min(f, 1.0 - f);
        let scar = 1.0 - smoothstep(0.0, 0.12, m);
        let band = hash12(vec2f(floor(t), 9.7));
        col *= 1.0 + dr * (0.3 * band - 0.15);
        col = mix(col, base * 0.35, 0.8 * scar * dr);
        // Each band swells between its scars: 0.025 · smoothstep(0, 0.3, distance to the nearer scar).
        let sw = smoothG(0.0, 0.3, vec3f(m, 0.0, select(-1.0, 1.0, f < 0.5)));
        o.h += 0.025 * dr * vec3f(sw.x, 0.0, sw.z / 0.17);
      }
      o.albedo = col;
    }
    case SP_JUNGLE: {
      // Smooth pale bark blotched with lichen, faint rings.
      var col = base * (0.9 + 0.2 * pnoise(vec2f(uv.x * 3.0, s * 0.4), 3.0));
      let d = keep(fw, 6.0, 0.8);
      if (d > 0.0) {
        let l = 0.65 * pnoise(vec2f(uv.x * 9.0, s * 2.0), 9.0) + 0.35 * pnoise(vec2f(uv.x * 25.0, s * 5.0), 25.0);
        let lichen = smoothstep(0.6, 0.7, l) * d;
        let ring = (1.0 - smoothstep(0.0, 0.05, abs(fract(s * 0.9 + 0.3 * l) - 0.5))) * d;
        col = mix(col, vec3f(0.33, 0.36, 0.28), 0.4 * lichen);
        col *= (1.0 - 0.3 * ring) * (1.0 + 0.12 * d * (pnoise(vec2f(uv.x * 30.0, s * 0.6), 30.0) - 0.5));
      }
      o.albedo = col;
    }
    case SP_DEAD: {
      // Weathered wood: grain twisting up the trunk, long dark cracks.
      let tw = uv.x + s * 0.035;
      let d = keep(fw, 28.0, 0.9);
      var col = base;
      if (d > 0.0) {
        let grain = pnoiseG(vec2f(tw * 28.0, s * 1.1), 28.0);
        let cn = pnoiseG(vec2f(tw * 7.0, s * 0.35), 7.0);
        let ck = smoothG(0.0, 0.035, vec3f(abs(cn.x - 0.5), sign(cn.x - 0.5) * cn.yz));
        let crack = (1.0 - ck.x) * d;
        col = base * (1.0 + d * (0.5 * grain.x - 0.25));
        col = mix(col, vec3f(0.035, 0.03, 0.028), 0.85 * crack);
        // Relief: 0.01 · grain − 0.02 · crack, slopes through tw = round + 0.035 · along.
        let gg = 0.01 * grain.yz * vec2f(28.0, 1.1) + 0.02 * ck.yz * vec2f(7.0, 0.35);
        o.h = d * vec3f(0.01 * grain.x - 0.02 * (1.0 - ck.x), gg.x, 0.035 * gg.x + gg.y);
      }
      o.albedo = col;
    }
    default: {
      // Painted bark (oak, maple, acacia, bush stems): long soft streaks down the trunk, shallow
      // furrows wandering along it, moss on whatever faces up and round the foot.
      let P = SP_PAINT[sp]; // streaks round, their length (m), moss, furrow depth (m)
      let d = keep(fw, 3.0 * P.x, 0.4 * P.y);
      var col = base;
      if (d > 0.0) {
        // Broad bands, streaks and (close up) grain, all running down the trunk.
        let a1 = pnoiseG(vec2f(uv.x * P.x, s / P.y), P.x);
        let a2 = pnoiseG(vec2f(uv.x * 3.0 * P.x, s / P.y * 2.7 + 5.1), 3.0 * P.x);
        let a3 = mix(0.5, pnoise(vec2f(uv.x * 8.0 * P.x, s / P.y * 7.0 + 2.3), 8.0 * P.x), keep(fw, 8.0 * P.x, 0.15 * P.y));
        let streak = 0.45 * a1.x + 0.35 * a2.x + 0.2 * a3;
        // Furrows run where another noise crosses its middle.
        let fz = pnoiseG(vec2f(uv.x * 2.0 * P.x + 0.5, s / P.y * 0.8 + 7.3), 2.0 * P.x);
        let fur = smoothG(0.0, 0.1, vec3f(abs(fz.x - 0.5), sign(fz.x - 0.5) * fz.yz));
        let furrow = 1.0 - fur.x;
        let mottle = pnoise(vec2f(uv.x * 3.0, s * 0.35 + 9.1), 3.0);
        col = mix(base * 0.35, base * 1.75, streak) * (1.0 - 0.6 * furrow) * (0.75 + 0.5 * mottle);
        // Moss in patches: more on whatever faces up and round the foot.
        let mz = pnoise(vec2f(uv.x * 4.0, s * 0.5 + 3.1), 4.0);
        let moss = P.z * smoothstep(0.62, 0.85, mz + 0.3 * up + 0.35 * (1.0 - smoothstep(0.0, 1.6, lp.y)));
        col = mix(col, vec3f(0.12, 0.19, 0.04) * (0.8 + 0.5 * a2.x), moss);
        col = mix(base, col, d);
        // Relief: streaks raised 1 cm, furrows P.w deep; slopes per unit round and per metre along.
        let gs = 0.01 * (0.45 * a1.yz * vec2f(P.x, 1.0 / P.y) + 0.35 * a2.yz * vec2f(3.0 * P.x, 2.7 / P.y));
        o.h = d * vec3f(0.01 * streak - P.w * furrow, gs + P.w * fur.yz * vec2f(2.0 * P.x, 0.8 / P.y));
      }
      o.albedo = col;
    }
  }
  return o;
}

// A cactus's skin: grooves between the ribs darker, crests lighter, felted areoles in rows along each crest.
fn cactusSkin(uv: vec2f, fw: vec2f, base: vec3f) -> vec3f {
  let rib = fract(uv.x * 8.0); // 0 on a crest (models.ts: 8 ribs), ½ in a groove
  let cr = min(rib, 1.0 - rib);
  let d = 1.0 - smoothstep(0.1, 0.3, fw.x * 8.0);
  var col = base * (1.0 + 0.2 * d * (1.0 - 2.0 * smoothstep(0.2, 0.5, cr)));
  col *= 0.92 + 0.16 * pnoise(vec2f(uv.x * 8.0, uv.y * 1.6), 8.0);
  let da = keep(fw, 64.0, 0.09);
  if (da > 0.0) {
    // Areoles 9 cm apart, staggered rib to rib; metres across (a rib is ~0.28 m wide) and along.
    let f = fract(uv.y / 0.09 + 0.5 * floor(uv.x * 8.0 + 0.5)) - 0.5;
    let e = length(vec2f(cr * 0.28, f * 0.09));
    let spot = (1.0 - smoothstep(0.009, 0.013, e)) * da;
    let halo = (1.0 - smoothstep(0.012, 0.03, e)) * da;
    col = mix(col, vec3f(0.55, 0.52, 0.42), 0.8 * spot) + vec3f(0.036, 0.036, 0.03) * halo;
  }
  return col;
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
  let sp = i.info & 15u;
  let tint = infoTint(i.info);
  var ao = i.attr.z;
  let vary = i.attr.w;
  let lp = i.lp;
  let rot = vec2f(cos(i.inst.w), sin(i.inst.w));
  // How fast the surface coordinates change per pixel, and how many model metres a pixel spans.
  let fw = fwidth(i.uv);
  let px = dist * u.footprint / i.inst.z;
  var albedo: vec3f;
  var leafy = 0.0;
  var pat = 0.5;
  var slope = vec2f(0.0); // the relief's slopes along the surface coordinates (model metres)
  var thin = 0.0;
  var shadow = 1.0; // the crown's own shade in the creases between leaf clusters
  var gloss = SP_GLOSS[sp];
  if (part == 1u || part == 2u) {
    let pal = leafPal(i.info, sp, tint);
    // Painterly breakup: two octaves of noise fixed to the tree.
    pat = 0.5 + 0.225 * vnoise(lp.x * 2.1 + lp.y * 0.9, lp.z * 2.1 - lp.y * 0.8, 1409)
        + 0.165 * vnoise(lp.x * 5.3 - lp.y * 1.7, lp.z * 5.3 + lp.y * 1.3, 1411);
    var shift = 0.0;
    var tone = 0.8 + 0.4 * pat;
    if (part == 1u) {
      var lv = Leaves(n, 1.0, 0.0, 1.0, 0.0, 1.0);
      // Seen from inside a crown (the camera in its leaves): the foliage round it is plain and in shade.
      let inside = dot(n, rd) > 0.35;
      if (inside) { lv.occl = 0.55; lv.sun = 0.5; }
      if (TREE_DETAIL && SP_CARD[sp] > 0.0 && i.inst.y < 1.5) {
        // Under its leaf cards a broadleaf crown is the foliage further in: the same sprays, projected
        // three ways onto it, darker and in its own shade, with deep gaps between the leaves.
        let deep = sprayAt(lp, toModel(n, rot), sp, px);
        lv.occl *= 0.55 + 0.25 * deep.x;
        lv.sun *= 0.35 + 0.45 * deep.x;
        lv.tone = 0.55 + 0.55 * deep.y;
      } else if (TREE_DETAIL && !inside && sp == SP_PINE) { lv = needles(lp, n, rot, sp, px); }
      n = lv.n;
      ao *= lv.occl;
      shift = lv.shift;
      tone *= lv.tone;
      thin = lv.rim;
      shadow = lv.sun;
    } else if (TREE_DETAIL) {
      let fr = frond(i.uv, fw);
      slope = fr.g;
      ao *= fr.occl;
      shift = fr.shift;
      tone *= fr.tone;
    }
    albedo = mix(pal[0], pal[1], clamp(0.1 + 0.45 * vary + 0.35 * ao + 0.25 * (tint - 0.5) + 0.45 * (pat - 0.5) + shift, 0.0, 1.0)) * tone;
    leafy = 1.0;
  } else if (part == 3u) {
    albedo = barkCol(i.info, sp) * (0.9 + 0.2 * vary);
    if (TREE_DETAIL) { albedo = cactusSkin(i.uv, fw, albedo); }
  } else if (part == 4u) {
    albedo = SP_ACCENT[sp] * (0.85 + 0.3 * vary);
  } else {
    albedo = barkCol(i.info, sp);
    if (TREE_DETAIL) {
      let b = bark(sp, barkCol(i.info, sp), i.uv, lp, fw, n.y);
      albedo = b.albedo;
      slope = b.h.yz;
    }
    albedo *= 0.9 + 0.2 * vary;
    gloss = 0.0;
  }
  // The relief (bark, leaflets) bends the normal; out here in uniform control flow, for the derivatives.
  if (TREE_DETAIL) { n = bumped(n, i.rel, i.uv, slope * i.inst.z); }
  if ((i.info & F_SNOW) != 0u) {
    // Snow settles on whatever faces up: tier tops, branch crooks, bush crowns.
    let cover = smoothstep(0.5, 0.82, n.y + 0.55 * (pat - 0.5)) * smoothstep(0.2, 0.6, ao);
    albedo = mix(albedo, vec3f(0.82, 0.86, 0.92), cover);
    leafy *= 1.0 - cover;
    gloss *= 1.0 - cover;
  }
  var sun = i.inst.x * i.shade * shadow;
  if (sun > 0.0 && i.inst.y < 0.5) { sun = min(sun, heroShadowAt(i.rel)); }
  return treeLight(albedo, n, i.rel, rd, dist, sun, ao, leafy, thin, gloss);
}

// A crown's leaf spray projected onto it three ways (model axes, one spray tile per 0.8 card, each
// tile flipped its own way): coverage and tone. Sampled at a level from the pixel's footprint
// (\`px\`: model metres a pixel spans), as the caller's control flow is not uniform.
fn sprayAt(lp: vec3f, nm: vec3f, sp: u32, px: f32) -> vec2f {
  let tile = 0.8 * SP_CARD[sp];
  let q = lp / tile;
  var w = nm * nm;
  w = w * w;
  w /= w.x + w.y + w.z;
  let lod = log2(max(px / tile * ${LEAF_SIZE}.0, 1.0));
  let layer = SP_SPRAY[sp];
  var acc = vec4f(0.0);
  for (var k = 0; k < 3; k++) {
    let st = select(select(q.xy, q.xz, k == 1), q.zy, k == 0);
    let flip = hash22(floor(st) + f32(k) * 31.0);
    var uv = fract(st);
    uv = select(uv, 1.0 - uv, flip > vec2f(0.5));
    acc += w[k] * textureSampleLevel(leafTex, leafSamp, uv, layer, lod);
  }
  return vec2f(acc.r, acc.g / max(acc.r, 1e-3));
}

// Light on a tree's surface: a soft two-tone ramp (softer on leaves), sun through leaves when backlit
// (most through thin edges), sky and ground fill, a glint off waxy leaves; then the air between.
fn treeLight(albedo: vec3f, n: vec3f, rel: vec3f, rd: vec3f, dist: f32, sun: f32, ao: f32, leafy: f32, thin: f32, gloss: f32) -> vec4f {
  let p = u.camPos + rel;
  let L = keyDir();
  let kc = keyCol(p.y);
  let ramp = smoothstep(-0.3 * leafy - 0.05, 0.7, dot(n, L));
  let back = pow(clamp(dot(rd, L), 0.0, 1.0), 3.0) * leafy * (1.0 - 0.4 * ao) * (0.75 + 0.5 * thin);
  let amb = skyAmb(n, 0.3 + 0.24 * n.y) + bounceCol(vec3f(0.035, 0.04, 0.02)) * (1.0 - n.y);
  let fill = mix(0.55 + 0.45 * ao, 0.35 + 0.65 * ao, leafy);
  // (no glint with the light at the horizon: glints would line up into a glitter path)
  let glint = pow(max(dot(reflect(rd, n), L), 0.0), 20.0) * gloss * ao * smoothstep(0.0, 0.2, L.y);
  var col = albedo * (kc * sun * (0.6 * ramp + 0.32 * back) + amb * fill) + kc * (sun * 0.12 * glint);
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}

// ---- leaf cards ------------------------------------------------------------------------

// Where a card's corner lands: four vertices sit at the card's centre on the crown (models.ts) and
// spread here across the screen round it, turned and sized per card (a hash of its place on the
// model); the middle level draws fewer, larger cards. Shared by both card passes, whose depths
// must agree to the bit.
struct Corner { rel: vec3f, centre: vec3f, cs: vec2f, uv: vec2f };
fn cardCorner(pos: vec3f, attr: vec4f, uv: vec2f, inst: Inst) -> Corner {
  let c = treePlace(pos, attr, inst);
  let sp = inst.q.x & 15u;
  let h = hash33(floor(pos * 64.0) + 0.5);
  var size = SP_CARD[sp] * (0.8 + 0.4 * h.x) * inst.p.w * mix(1.0, 1.75, min(f32(inst.q.w), 1.0));
  // A camera in among the leaves: the cards round it give way (they would fill the view). Far out
  // in the middle level, where a card spans a few pixels, they shrink away (the crown under them
  // carries its leaves on).
  let reach = SP_LOD[sp].y * sqrt(inst.p.w);
  size *= smoothstep(0.6, 1.6, length(c)) * (1.0 - smoothstep(0.45, 0.6, length(inst.p.xyz) / reach));
  let a = 6.2831853 * h.y;
  let cs = vec2f(cos(a), sin(a));
  let k = vec2f(uv.x - 0.5, 0.5 - uv.y);
  let o = vec2f(cs.x * k.x - cs.y * k.y, cs.y * k.x + cs.x * k.y) * size;
  return Corner(c + u.basis[0] * o.x + u.basis[1] * o.y, c, cs, select(uv, vec2f(1.0 - uv.x, uv.y), h.z > 0.5));
}

struct CardDepthOut {
  @invariant @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) look: vec2f, // level-of-detail fade, spray
};

@vertex
fn vsCardDepth(@location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) attr: vec4f, @location(3) uv: vec2f, @builtin(instance_index) ii: u32) -> CardDepthOut {
  let inst = instIn[ii];
  let k = cardCorner(pos, attr, uv, inst);
  var out: CardDepthOut;
  out.pos = project(k.rel);
  out.uv = k.uv;
  out.look = vec2f(bitcast<f32>(inst.q.y), f32(SP_SPRAY[inst.q.x & 15u]));
  return out;
}

// The cards' depth: two screen-space dithers (TAA blends what is left): the level-of-detail
// cross-fade, then the spray's coverage, which cuts the leaves out and, far off where the mipmaps
// blur it, thins the spray to its mean coverage instead of shrinking it. (No colour is written.)
@fragment
fn fsCardDepth(i: CardDepthOut) -> @location(0) vec4f {
  let tex = textureSample(leafTex, leafSamp, i.uv, i32(i.look.y));
  let h = fract(52.9829189 * fract(dot(i.pos.xy, vec2f(0.06711056, 0.00583715))) + forest.camOff.w);
  let fade = i.look.x;
  if (fade < 1.5) { if (fade >= 0.0) { if (h >= fade) { discard; } } else if (h < -1.0 - fade) { discard; } }
  let h2 = fract(52.9829189 * fract(dot(i.pos.xy, vec2f(0.00583715, 0.06711056))) + 1.6180339 * forest.camOff.w);
  if (tex.r <= max(h2, 0.04)) { discard; }
  return vec4f(0.0);
}

struct CardOut {
  @invariant @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,                        // the crown's normal at the card (world axes)
  @location(2) @interpolate(flat) info: u32,
  @location(3) uv: vec2f,                       // on the spray
  @location(4) @interpolate(flat) look: vec4f,  // sun at the tree, level-of-detail fade, occlusion, variation
  @location(5) shade: f32,                      // sun under the trees (the shadow map)
  @location(6) @interpolate(flat) turn: vec4f,  // the card's x axis on the screen (cos, sin); level of detail; spray
};

@vertex
fn vsCard(@location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) attr: vec4f, @location(3) uv: vec2f, @builtin(instance_index) ii: u32) -> CardOut {
  let inst = instIn[ii];
  let yaw = f32((inst.q.x >> 6u) & 255u) * (6.2831853 / 256.0);
  let cy = cos(yaw);
  let sy = sin(yaw);
  let k = cardCorner(pos, attr, uv, inst);
  var out: CardOut;
  out.pos = project(k.rel);
  out.rel = k.rel;
  out.n = vec3f(cy * nrm.x + sy * nrm.z, nrm.y, -sy * nrm.x + cy * nrm.z);
  out.info = inst.q.x;
  out.uv = k.uv;
  out.look = vec4f(unpack2x16float(inst.q.z).x, bitcast<f32>(inst.q.y), attr.z, attr.w);
  out.shade = 1.0;
  // Shadows of other trees on the near trees' cards (one tap a vertex; the middle level goes without).
  if (inst.q.w == 0u) { out.shade = treeShadowAt(u.camPos + k.centre + out.n * 0.3); }
  out.turn = vec4f(k.cs, f32(inst.q.w), f32(SP_SPRAY[inst.q.x & 15u]));
  return out;
}

// The cards' colour, where the depth pass left each card's own depth (one leaf per pixel, no discard).
@fragment
fn fsCard(i: CardOut) -> @location(0) vec4f {
  let tex = textureSample(leafTex, leafSamp, i.uv, i32(i.turn.w));
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < W_LEVEL) { return underwater(rd); }
  let cov = max(tex.r, 1e-3);
  let tone = tex.g / cov;
  let tilt = tex.ba / cov * 2.0 - 1.0;
  // The leaf's normal: the crown's, bent the way the leaf faces on the card (x right, y down it).
  let ax = u.basis[0] * i.turn.x + u.basis[1] * i.turn.y;
  let ay = u.basis[1] * i.turn.x - u.basis[0] * i.turn.y;
  var n = normalize(normalize(i.n) + 0.9 * (tilt.x * ax - tilt.y * ay) - 0.3 * rd);
  let sp = i.info & 15u;
  let tint = infoTint(i.info);
  let pal = leafPal(i.info, sp, tint);
  let ao = i.look.z;
  let vary = i.look.w;
  var albedo = mix(pal[0], pal[1], clamp(0.05 + 0.4 * vary + 0.35 * ao + 0.25 * (tint - 0.5) + 0.6 * (tone - 0.55), 0.0, 1.0)) * (0.7 + 0.5 * tone);
  var leafy = 1.0;
  if ((i.info & F_SNOW) != 0u) {
    let cover = smoothstep(0.55, 0.85, n.y) * smoothstep(0.3, 0.7, ao);
    albedo = mix(albedo, vec3f(0.82, 0.86, 0.92), cover);
    leafy = 1.0 - cover;
  }
  var sun = i.look.x * i.shade;
  if (sun > 0.0 && i.turn.z < 0.5) { sun = min(sun, heroShadowAt(i.rel)); }
  return treeLight(albedo, n, i.rel, rd, dist, sun, ao, leafy, 1.0, SP_GLOSS[sp] * leafy);
}
`;
}
