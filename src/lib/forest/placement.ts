/**
 * Where trees grow. Rules and numbers shared with the GPU (forest.wgsl.ts
 * compiles the tables below and runs the same steps); this file is also the
 * CPU twin, for tests, surveys and anything that needs a tree's position.
 * ---------------------------------------------------------------------------
 * Open land is the default. Trees come in features, each placed for how it
 * reads in a view:
 *
 * - Forests: bodies of a few hundred metres with lobed edges (a domain-warped
 *   field over 700, 260 and 110 m), so most views hold a wood or two. They grow where the ground suits them: flanks,
 *   hollows, damp ground and the darker woodland soil, never crests or
 *   cliffs. Each has a fringe of smaller trees and bushes, glades inside,
 *   and one species per stand (~200 m patches), not a salt-and-pepper mix.
 * - Groves: clumps of a few dozen metres in open land, most in hollows;
 *   with lone trees they come thicker in parkland than on open plains
 *   (a ~2 km field), so journeys alternate between the two.
 * - Lone trees: at most one per coarse cell, so they never crowd; big and
 *   open-grown, most on gentle rises where they stand against the sky.
 * - Shores: a broken line of trees just above the water — the sea, a lake or a
 *   river at its own level (birches up north, palms in the warm south, oases
 *   in dry land).
 * - Bushes: sparse in the open, thicker along forest edges.
 *
 * How much of each, which species, how many old giants and in which colours
 * comes from the flora zone, blended across zone borders. The zones follow the
 * lands' biome regions (biomes.ts, terrainMath.ts landRegions) — deciduous
 * woods in the temperate forest, giants and palms spread thinly through the
 * rainforest, open meadows with copses of oak on the plains, acacias on the savanna, cactus in
 * the desert, snags and juniper scrub in the badlands, birch and snags in the
 * marsh, pines in the taiga, ancient giants in moss, gold and blossom in the
 * sacred groves — and the air's temperature with height: pines where it is
 * cool (montane), sparse wind-bent pines higher up (alpine), nothing above the
 * tree line. The amounts are TotK's, measured per region from its game data
 * (docs/TOTK_TERRAIN.md): woods are clumps of 30–55 trees/ha, a wooded region
 * carries 5–8 trees/ha with a tree in a third to half of its 50 m blocks, and
 * the jungle spreads ~2.6 trees over two thirds of its blocks.
 */

import { BI, BIOMES, LOOK } from '../biomes';
import { CLIMATE } from '../constants';
import { hash3, landCoarse, lastMoisture, lastRegions, lastSeaTemperature, lastShore, surfaceAt, vnoise } from '../terrainMath';
import { SKY_FLORA } from '../skyworld/biomes';
import { CELL, type Species } from './species';

export const ZONES = [
  'temperate', 'autumn', 'plains', 'savanna', 'rainforest', 'jungle', 'wetland', 'desert', 'badlands',
  'volcanic', 'mountain', 'alpine', 'tundra', 'taiga', 'highland', 'sacred', 'lakes',
] as const;
type Zone = (typeof ZONES)[number];
/** A species, its alternate, and the share of the alternate. */
type Pair = readonly [Species, Species, number];
/** A sky-tree palette (skyworld/biomes.ts SKY_FLORA, one of the first eight), or none: the species' own colours. */
type Pal = (typeof SKY_FLORA)[number] | null;

type Rule = {
  /** Forest field level: the field (~0.5 on average, ±0.2) must exceed it; 9 = no forests. */
  forest: number;
  /** Share of cells holding a tree deep inside a forest, and its species by stand. */
  fill: number;
  forestSp: Pair;
  /** Groves: coarse cell (m, a multiple of CELL), chance per cell, radius range (m), fill at the core. */
  groveGrid: number;
  groveP: number;
  groveR: readonly [number, number];
  groveFill: number;
  groveSp: Pair;
  /** Lone trees: coarse cell (m, a multiple of CELL) and chance per cell. */
  loneGrid: number;
  loneP: number;
  loneSp: Pair;
  /** Shore line: fill just above the water, and its species. */
  shore: number;
  shoreSp: Species;
  /** Bushes: chance per open cell. */
  shrub: number;
  /** Old giants among the forest trees: their share, and their scale (× 0.85 … 1.15 by the pick). */
  giant: readonly [number, number];
  /** The stand's colours, by thirds of the stand field (null: the species' own). */
  pal: readonly [Pal, Pal, Pal];
};

const NONE = [null, null, null] as const;
export const RULES: Record<Zone, Rule> = {
  // dense deciduous woods: oak with birch, glades, streams
  temperate: {
    forest: 0.7, fill: 0.2, forestSp: ['oak', 'birch', 0.3],
    groveGrid: 132, groveP: 0.6, groveR: [16, 36], groveFill: 0.28, groveSp: ['oak', 'birch', 0.35],
    loneGrid: 42, loneP: 0.15, loneSp: ['oak', 'birch', 0.15],
    shore: 0.09, shoreSp: 'birch', shrub: 0.018, giant: [0.05, 1.35], pal: NONE,
  },
  // maple woods in their autumn colours
  autumn: {
    forest: 0.7, fill: 0.2, forestSp: ['maple', 'oak', 0.22],
    groveGrid: 132, groveP: 0.6, groveR: [16, 36], groveFill: 0.28, groveSp: ['maple', 'birch', 0.15],
    loneGrid: 42, loneP: 0.17, loneSp: ['maple', 'oak', 0.25],
    shore: 0.07, shoreSp: 'birch', shrub: 0.0156, giant: [0.04, 1.3], pal: NONE,
  },
  // open grass, rolling hills, scattered trees
  plains: {
    forest: 0.7, fill: 0.16, forestSp: ['oak', 'birch', 0.25],
    groveGrid: 120, groveP: 0.7, groveR: [18, 40], groveFill: 0.33, groveSp: ['oak', 'birch', 0.25],
    loneGrid: 42, loneP: 0.15, loneSp: ['oak', 'oak', 0],
    shore: 0.05, shoreSp: 'birch', shrub: 0.0072, giant: [0.03, 1.3], pal: NONE,
  },
  savanna: {
    forest: 9, fill: 0, forestSp: ['acacia', 'acacia', 0],
    groveGrid: 150, groveP: 0.03, groveR: [10, 18], groveFill: 0.2, groveSp: ['acacia', 'acacia', 0],
    loneGrid: 42, loneP: 0.05, loneSp: ['acacia', 'acacia', 0],
    shore: 0.08, shoreSp: 'palm', shrub: 0.012, giant: [0.02, 1.3], pal: NONE,
  },
  // tall and wet: jungle giants and palms spread thinly over most of the land (TotK's Faron)
  rainforest: {
    forest: 0.24, fill: 0.034, forestSp: ['jungle', 'palm', 0.12],
    groveGrid: 108, groveP: 0.45, groveR: [14, 30], groveFill: 0.14, groveSp: ['jungle', 'palm', 0.3],
    loneGrid: 42, loneP: 0.6, loneSp: ['jungle', 'palm', 0.35],
    shore: 0.25, shoreSp: 'palm', shrub: 0.021, giant: [0.12, 1.6], pal: NONE,
  },
  // the same on the karst: jungle and palms to the tops of the towers
  jungle: {
    forest: 0.22, fill: 0.04, forestSp: ['jungle', 'palm', 0.22],
    groveGrid: 108, groveP: 0.45, groveR: [12, 26], groveFill: 0.14, groveSp: ['jungle', 'palm', 0.35],
    loneGrid: 42, loneP: 0.6, loneSp: ['palm', 'jungle', 0.4],
    shore: 0.25, shoreSp: 'palm', shrub: 0.024, giant: [0.08, 1.45], pal: NONE,
  },
  // marsh: birch carr, drowned snags, thick scrub
  wetland: {
    forest: 0.68, fill: 0.1, forestSp: ['birch', 'dead', 0.3],
    groveGrid: 120, groveP: 0.65, groveR: [10, 22], groveFill: 0.24, groveSp: ['birch', 'dead', 0.25],
    loneGrid: 42, loneP: 0.35, loneSp: ['dead', 'birch', 0.4],
    shore: 0.15, shoreSp: 'birch', shrub: 0.036, giant: [0, 1], pal: NONE,
  },
  desert: {
    forest: 9, fill: 0, forestSp: ['cactus', 'cactus', 0],
    groveGrid: 150, groveP: 0, groveR: [10, 10], groveFill: 0, groveSp: ['cactus', 'cactus', 0],
    loneGrid: 42, loneP: 0.12, loneSp: ['cactus', 'dead', 0.08],
    shore: 0.2, shoreSp: 'palm', shrub: 0.0018, giant: [0, 1], pal: NONE,
  },
  // canyon country: snags, a few acacias, juniper scrub
  badlands: {
    forest: 9, fill: 0, forestSp: ['dead', 'dead', 0],
    groveGrid: 150, groveP: 0.03, groveR: [8, 14], groveFill: 0.2, groveSp: ['acacia', 'dead', 0.3],
    loneGrid: 42, loneP: 0.05, loneSp: ['dead', 'acacia', 0.4],
    shore: 0.12, shoreSp: 'palm', shrub: 0.0084, giant: [0, 1], pal: NONE,
  },
  // scorched (TotK's Eldin): snags in pairs, the odd pine, no snow (its frost)
  volcanic: {
    forest: 9, fill: 0, forestSp: ['dead', 'dead', 0],
    groveGrid: 150, groveP: 0.8, groveR: [8, 16], groveFill: 0.4, groveSp: ['dead', 'pine', 0.2],
    loneGrid: 42, loneP: 0.1, loneSp: ['dead', 'pine', 0.15],
    shore: 0, shoreSp: 'birch', shrub: 0.0036, giant: [0, 1], pal: NONE,
  },
  // montane conifer forest where the air is cool
  mountain: {
    forest: 0.74, fill: 0.15, forestSp: ['pine', 'birch', 0.12],
    groveGrid: 120, groveP: 0.35, groveR: [12, 26], groveFill: 0.22, groveSp: ['pine', 'birch', 0.2],
    loneGrid: 42, loneP: 0.3, loneSp: ['pine', 'dead', 0.06],
    shore: 0.03, shoreSp: 'birch', shrub: 0.0084, giant: [0.03, 1.25], pal: NONE,
  },
  alpine: {
    forest: 0.9, fill: 0.12, forestSp: ['pine', 'pine', 0],
    groveGrid: 150, groveP: 0.2, groveR: [8, 14], groveFill: 0.2, groveSp: ['pine', 'pine', 0],
    loneGrid: 42, loneP: 0.22, loneSp: ['pine', 'dead', 0.45],
    shore: 0, shoreSp: 'birch', shrub: 0.0024, giant: [0, 1], pal: NONE,
  },
  // frozen: snow-laden snags, stunted pines, buried scrub
  tundra: {
    forest: 0.72, fill: 0.14, forestSp: ['pine', 'pine', 0],
    groveGrid: 150, groveP: 0.8, groveR: [10, 18], groveFill: 0.3, groveSp: ['pine', 'dead', 0.3],
    loneGrid: 42, loneP: 0.2, loneSp: ['dead', 'pine', 0.5],
    shore: 0, shoreSp: 'birch', shrub: 0.006, giant: [0, 1], pal: NONE,
  },
  // the boreal forest: pine with birch
  taiga: {
    forest: 0.64, fill: 0.18, forestSp: ['pine', 'birch', 0.15],
    groveGrid: 108, groveP: 0.45, groveR: [12, 26], groveFill: 0.24, groveSp: ['pine', 'birch', 0.25],
    loneGrid: 42, loneP: 0.3, loneSp: ['pine', 'dead', 0.08],
    shore: 0.05, shoreSp: 'birch', shrub: 0.0072, giant: [0.03, 1.25], pal: NONE,
  },
  // alpine plateau: pine and birch groves, heath
  highland: {
    forest: 0.78, fill: 0.14, forestSp: ['pine', 'birch', 0.25],
    groveGrid: 132, groveP: 0.45, groveR: [10, 22], groveFill: 0.24, groveSp: ['pine', 'birch', 0.3],
    loneGrid: 42, loneP: 0.35, loneSp: ['pine', 'birch', 0.3],
    shore: 0.04, shoreSp: 'birch', shrub: 0.0132, giant: [0, 1], pal: NONE,
  },
  // the ancient forest: dense, old giants in moss, gold and blossom
  sacred: {
    forest: 0.68, fill: 0.18, forestSp: ['oak', 'jungle', 0.35],
    groveGrid: 102, groveP: 0.6, groveR: [16, 32], groveFill: 0.24, groveSp: ['oak', 'birch', 0.25],
    loneGrid: 42, loneP: 0.2, loneSp: ['oak', 'jungle', 0.3],
    shore: 0.06, shoreSp: 'birch', shrub: 0.027, giant: [0.28, 2.2], pal: ['mossy', null, 'golden'],
  },
  // the lake district: pine and birch to the shores, oaks in the sheltered bays
  lakes: {
    forest: 0.62, fill: 0.17, forestSp: ['pine', 'birch', 0.4],
    groveGrid: 120, groveP: 0.55, groveR: [12, 26], groveFill: 0.24, groveSp: ['birch', 'oak', 0.35],
    loneGrid: 42, loneP: 0.25, loneSp: ['oak', 'pine', 0.4],
    shore: 0.15, shoreSp: 'birch', shrub: 0.0132, giant: [0.03, 1.3], pal: NONE,
  },
};

/** Shared constants: noise wavelengths (m), seed offsets, thresholds. */
export const PLACE = {
  warp: { wavelength: 450, amount: 150, seed: 7201 },
  field: [[700, 0.62, 7203], [260, 0.28, 7204], [110, 0.1, 7205]] as ReadonlyArray<readonly [number, number, number]>,
  /** Parkland vs open plain (~2 km): scales the chance of groves and lone trees by lo..lo + span. */
  parkland: { wavelength: 2100, seed: 7236, lo: 0.6, span: 0.8 },
  /** Lone trees: chance multiplier on flat ground, plus this much on a knoll. */
  lone: { flat: 0.75, knoll: 0.5 },
  /** Half width of the forest edge band, in field units. */
  edge: 0.045,
  glade: { wavelength: 130, seed: 7206, lo: 0.64, hi: 0.74, cut: 0.9 },
  stand: { wavelength: 210, seed: 7207 },
  region: { wavelength: 4200, seed: 7029, lo: 0.72, hi: 0.8 }, // autumn woods within the temperate forest
  /** Desert plants come in patches (~260 m): the extra chance of a lone cactus there. */
  cactus: { wavelength: 260, seed: 7209, patch: 0.15 },
  gaps: { wavelength: 90, seed: 7208 },
  /** Relief: a Laplacian over this step (m), from heights prefiltered for dp (m): hollows and knolls
   *  for groves and lone trees; the hill-scale one (and its slope) for forest bodies. */
  relief: { step: 36, dp: 12 },
  hills: { step: 84, dp: 24 },
  /**
   * The air with height (°C): pines take over where it is cooler than `montane` (the tropical and
   * dry zones keep their own), sparse alpine pines where it is cooler than `alpine`, nothing above
   * the tree line (`treeline`); each is a smoothstep's range.
   */
  montane: [2.5, 6.0], alpine: [-1.0, 1.5], treeline: [-3.5, -1.5],
  /** The mountain zone gives way to temperate woods where its foot is warm. */
  warmFoot: [22, 26],
} as const;

/** The scale range a plant's 8-bit scale covers (the GPU record's encoding; collision.ts matches it). */
export const SCALE = { lo: 0.55, span: 2.45 } as const;

// ---- helpers (the WGSL mirrors each) ---------------------------------------------

const smooth = (lo: number, hi: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
};
const u16 = (h: number, shift: number) => ((h >>> shift) & 0xffff) / 65535;
/** Floor division for integers (negative too). */
const fdiv = (a: number, b: number) => Math.floor(a / b);

const ZI = Object.fromEntries(ZONES.map((z, i) => [z, i])) as Record<Zone, number>;
const NZ = ZONES.length;
/** Zones that keep their own trees as the air cools (not turned to montane pines). */
export const KEEP_COOL: readonly Zone[] = ['savanna', 'rainforest', 'jungle', 'desert', 'badlands', 'volcanic', 'mountain', 'alpine', 'tundra', 'taiga', 'highland'];

/**
 * Zone weights (summing to ≤ 1; 0 above the tree line) from the biome regions' weights w (biomes.ts
 * order), the air's temperature T there and the autumn field (`region`, 0..1).
 */
function zoneWeights(T: number, w: ArrayLike<number>, region: number, out: Float64Array = new Float64Array(NZ)): Float64Array {
  out.fill(0);
  const autumn = smooth(PLACE.region.lo, PLACE.region.hi, region);
  const warmFoot = smooth(PLACE.warmFoot[0], PLACE.warmFoot[1], T);
  out[ZI.temperate] = w[BI.forest] * (1 - autumn) + w[BI.mountains] * warmFoot;
  out[ZI.autumn] = w[BI.forest] * autumn;
  out[ZI.plains] = w[BI.plains];
  out[ZI.savanna] = w[BI.savanna];
  out[ZI.rainforest] = w[BI.rainforest];
  out[ZI.jungle] = w[BI.jungle];
  out[ZI.wetland] = w[BI.wetland];
  out[ZI.desert] = w[BI.desert];
  out[ZI.badlands] = w[BI.badlands];
  out[ZI.volcanic] = w[BI.volcanic];
  out[ZI.mountain] = w[BI.mountains] * (1 - warmFoot);
  out[ZI.tundra] = w[BI.tundra];
  out[ZI.taiga] = w[BI.taiga];
  out[ZI.highland] = w[BI.highland];
  out[ZI.sacred] = w[BI.sacred];
  out[ZI.lakes] = w[BI.lakes];
  // With height: montane pines where the air is cool, alpine pines higher, nothing above the tree line.
  const mont = 1 - smooth(PLACE.montane[0], PLACE.montane[1], T);
  const alp = 1 - smooth(PLACE.alpine[0], PLACE.alpine[1], T);
  const tree = smooth(PLACE.treeline[0], PLACE.treeline[1], T);
  for (let k = 0; k < NZ; k++) {
    if (k === ZI.alpine || k === ZI.mountain) continue;
    if (KEEP_COOL.includes(ZONES[k])) continue;
    out[ZI.mountain] += out[k] * mont;
    out[k] *= 1 - mont;
  }
  for (let k = 0; k < NZ; k++) {
    if (k === ZI.alpine || k === ZI.tundra || k === ZI.volcanic) continue;
    out[ZI.alpine] += out[k] * alp;
    out[k] *= 1 - alp;
  }
  for (let k = 0; k < NZ; k++) out[k] *= tree;
  return out;
}

/** Forest field at world (x, z): lobed bodies from a domain-warped sum of three octaves (~0.5 ± 0.2). */
function forestField(x: number, z: number, seed: number): number {
  const w = PLACE.warp;
  const wx = x + w.amount * vnoise(x / w.wavelength, z / w.wavelength, seed + w.seed);
  const wz = z + w.amount * vnoise(x / w.wavelength + 17.3, z / w.wavelength - 9.1, seed + w.seed + 1);
  let s = 0;
  for (const [wl, amp, off] of PLACE.field) s += amp * vnoise(wx / wl, wz / wl, seed + off);
  return 0.5 + 0.5 * s;
}

const noise01 = (x: number, z: number, wavelength: number, seed: number) => 0.5 + 0.5 * vnoise(x / wavelength, z / wavelength, seed);

/** The species of a pair for a value in [0, 1). */
const pairPick = (p: Pair, v: number): Species => (v < 1 - p[2] ? p[0] : p[1]);
/** A stand's palette (an index into SKY_FLORA, or −1: the species' own colours). */
const palPick = (r: Rule, stand: number): number => {
  const p = r.pal[stand < 1 / 3 ? 0 : stand < 2 / 3 ? 1 : 2];
  return p ? SKY_FLORA.indexOf(p) : -1;
};

type Feature = 'forest' | 'grove' | 'lone' | 'shore' | 'bush';
export type Plant = {
  species: Species; feature: Feature; x: number; z: number; scale: number; snow: boolean;
  /** A sky palette (SKY_FLORA index) the plant takes instead of its species' colours, or −1. */
  palette: number;
};

/** Optional world context: ruins stay bare. (The sky islands' trees are the forest's other layer: skyworld/flora.ts.) */
export type PlaceWorld = { onRuin?: (x: number, z: number) => boolean };

const DESERT = ZI.desert;
const _zw = new Float64Array(NZ);
const _sf: [number, number] = [0, 0];

type Site = {
  e: number; wt: number; shore: number; slope: number; hollow: number; knoll: number; T: number; M: number;
  /** The temperature snow is judged by: the air's, less the regions' frost. */
  Ts: number;
  /** This spot's zone (picked by weight with the given random, so borders blend) and the dominant one. */
  zone: number; dom: number;
  /** Forest level and core fill blended over the zones; the forest body here (0..1); openness. */
  level: number; fill: number; mask: number; open: number;
};

/** The air's temperature at the last surface evaluated, at height e. */
const airT = (e: number) => lastSeaTemperature() - CLIMATE.lambdaAlt * Math.max(0, e);

/**
 * The regions' frost at the last surface evaluated (biomes.ts LOOK.frost, °C): trees carry snow by the
 * ground's own snow line, so a volcano's ash holds no snowy pines and the tundra's trees are white.
 */
const FROST = BIOMES.map((b) => LOOK[b].frost);
function frostHere(): number {
  const w = lastRegions();
  let f = 0;
  for (let k = 0; k < FROST.length; k++) f += w[k] * FROST[k];
  return f;
}

/** Ground, water, climate, zones and the forest body at world (x, z); null in the water or above the tree line. */
function siteAt(x: number, z: number, seed: number, region: number, zoneR: number): Site | null {
  // Height and the water's level, the regions there, local slope, and relief (+ hollow, − knoll) over 36 m.
  surfaceAt(x, z, seed, _sf, 1);
  const e = _sf[0], wt = _sf[1], shore = lastShore();
  if (e < wt + 0.4) return null;
  const T = airT(e), M = lastMoisture(), Ts = T - frostHere();
  const zw = zoneWeights(T, lastRegions(), region, _zw);
  const slope = Math.hypot(surfaceAt(x + 1.5, z, seed, _sf, 1)[0] - e, surfaceAt(x, z + 1.5, seed, _sf, 1)[0] - e) / 1.5;
  const L = PLACE.relief.step, dp = PLACE.relief.dp;
  const ev = (px: number, pz: number, d: number) => landCoarse(px, pz, seed, d);
  const e0 = ev(x, z, dp);
  const relief = (ev(x + L, z, dp) + ev(x - L, z, dp) + ev(x, z + L, dp) + ev(x, z - L, dp) - 4 * e0) / L;
  // Forest bodies follow the hills (84 m), not every hummock.
  const H = PLACE.hills.step, hp = PLACE.hills.dp;
  const h0 = ev(x, z, hp);
  const hE = ev(x + H, z, hp), hW = ev(x - H, z, hp), hN = ev(x, z + H, hp), hS = ev(x, z - H, hp);
  const hills = (hE + hW + hN + hS - 4 * h0) / H;
  const hillSlope = Math.hypot(hE - hW, hN - hS) / (2 * H);
  const hollow = smooth(0.03, 0.15, relief), knoll = smooth(0.03, 0.15, -relief);
  let total = 0;
  for (let k = 0; k < NZ; k++) total += zw[k];
  if (total < 1e-4) return null;
  let zone = 0, dom = 0, acc = 0, level = 0, fill = 0;
  for (let k = 0; k < NZ; k++) {
    acc += zw[k] / total;
    if (zoneR >= acc) zone = k + 1;
    if (zw[k] > zw[dom]) dom = k;
    level += zw[k] * RULES[ZONES[k]].forest;
    fill += zw[k] * RULES[ZONES[k]].fill;
  }
  level /= total;
  fill /= total;
  // The forest body: the field plus how well the ground suits a forest.
  const suit = 0.1 * smooth(0.02, 0.12, hills) - 0.12 * smooth(0.02, 0.12, -hills) + 0.15 * (M - 0.62) - 0.3 * smooth(0.55, 0.9, hillSlope);
  const mask = smooth(level - PLACE.edge, level + PLACE.edge, forestField(x, z, seed) + suit);
  return { e, wt, shore, slope, hollow, knoll, T, M, Ts, zone: Math.min(zone, NZ - 1), dom, level, fill, mask, open: 1 - smooth(0.15, 0.4, mask) };
}

/** What the far forest needs of the ground: its height, slope, hill-scale relief (84 m) and that relief's slope. */
type CanopyGround = { e: number; slope: number; hills: number; hillSlope: number };
type Canopy = {
  /** Expected share of cells holding a forest tree, and the trees per cell groves and lone trees add on average. */
  forest: number;
  open: number;
  /** The forest's species (the dominant zone's pair, by stand), and whether snow lies on it. */
  species: Species;
  snow: boolean;
};

/**
 * Averages the far forest takes over what it cannot see: a grove's mean fill over its disc as a
 * share of its core fill (its edge falls off from 0.45 r to r); a grove's chance factor (0.5 +
 * hollow) over open land; the share of open land on knolls (lone trees favour them).
 */
export const CANOPY_AVG = { groveShape: 0.54, groveHollow: 0.7, knolls: 0.2 } as const;

/**
 * The forest seen from afar, where the renderer paints it onto the ground instead of drawing its
 * trees (terrain.wgsl.ts shadeGround; WGSL twin forest.wgsl.ts canopyAt, run once per clipmap
 * lattice point): at world (x, z), plantInCell's chances before its draws — a forest tree from the
 * body's fill, its glades and the slope; groves and lone trees on average — and the forest's species.
 * Null in the water or above the tree line.
 */
function canopyFrom(x: number, z: number, seed: number, g: CanopyGround): Canopy | null {
  surfaceAt(x, z, seed, _sf, 1);
  if (g.e < _sf[1] + 0.4) return null;
  const T = airT(g.e), M = lastMoisture(), Ts = T - frostHere();
  const zw = zoneWeights(T, lastRegions(), noise01(x, z, PLACE.region.wavelength, seed + PLACE.region.seed), _zw);
  let total = 0, dom = 0, level = 0, fill = 0;
  for (let k = 0; k < NZ; k++) {
    total += zw[k];
    if (zw[k] > zw[dom]) dom = k;
    level += zw[k] * RULES[ZONES[k]].forest;
    fill += zw[k] * RULES[ZONES[k]].fill;
  }
  if (total < 1e-4) return null;
  level /= total;
  fill /= total;
  const suit = 0.1 * smooth(0.02, 0.12, g.hills) - 0.12 * smooth(0.02, 0.12, -g.hills) + 0.15 * (M - 0.62) - 0.3 * smooth(0.55, 0.9, g.hillSlope);
  const mask = smooth(level - PLACE.edge, level + PLACE.edge, forestField(x, z, seed) + suit);
  const open = 1 - smooth(0.15, 0.4, mask);
  const glade = smooth(PLACE.glade.lo, PLACE.glade.hi, noise01(x, z, PLACE.glade.wavelength, seed + PLACE.glade.seed));
  const D = RULES[ZONES[dom]];
  const park = PLACE.parkland.lo + PLACE.parkland.span * smooth(0.3, 0.7, noise01(x, z, PLACE.parkland.wavelength, seed + PLACE.parkland.seed));
  const [r0, r1] = D.groveR;
  const A = CANOPY_AVG;
  const grove = D.groveP * park * A.groveHollow * open * D.groveFill * A.groveShape * Math.PI * ((r0 * r0 + r0 * r1 + r1 * r1) / 3) / (D.groveGrid * D.groveGrid);
  const lone = D.loneP * park * (PLACE.lone.flat + A.knolls * PLACE.lone.knoll) * (1 - smooth(0.35, 0.6, g.slope)) * open * (CELL / D.loneGrid) ** 2;
  const species = pairPick(D.forestSp, noise01(x, z, PLACE.stand.wavelength, seed + PLACE.stand.seed));
  return {
    forest: fill * Math.pow(mask, 1.4) * (1 - PLACE.glade.cut * glade) * (1 - smooth(0.8, 1.15, g.slope)),
    open: (grove * (1 - smooth(0.5, 0.9, g.slope)) + lone),
    species,
    snow: Ts < 0.5 && (species === 'pine' || species === 'dead' || species === 'bush' || species === 'birch'),
  };
}

/** canopyFrom on the ground at (x, z) as plantInCell sees it (surveys, tests). */
export function forestCanopy(x: number, z: number, seed: number): Canopy | null {
  const ev = (px: number, pz: number, d: number) => surfaceAt(px, pz, seed, _sf, d)[0];
  const e = ev(x, z, 1);
  const slope = Math.hypot(ev(x + 1.5, z, 1) - e, ev(x, z + 1.5, 1) - e) / 1.5;
  const H = PLACE.hills.step, hp = PLACE.hills.dp;
  const ev2 = (px: number, pz: number, d: number) => landCoarse(px, pz, seed, d);
  const h0 = ev2(x, z, hp);
  const hE = ev2(x + H, z, hp), hW = ev2(x - H, z, hp), hN = ev2(x, z + H, hp), hS = ev2(x, z - H, hp);
  return canopyFrom(x, z, seed, { e, slope, hills: (hE + hW + hN + hS - 4 * h0) / H, hillSlope: Math.hypot(hE - hW, hN - hS) / (2 * H) });
}

/** The dominant zone and the forest body at world (x, z) (surveys, tests); null in the water or above the tree line. */
export function forestAt(x: number, z: number, seed: number): { zone: Zone; mask: number } | null {
  const s = siteAt(x, z, seed, noise01(x, z, PLACE.region.wavelength, seed + PLACE.region.seed), 0);
  return s && { zone: ZONES[s.dom], mask: s.mask };
}

/**
 * The plant in lattice cell (ix, iz), or null. Cells are CELL metres; each
 * holds at most one plant, at a hashed spot inside it (a lone tree sits at
 * its coarse cell's own spot, which falls in exactly one fine cell).
 */
export function plantInCell(ix: number, iz: number, seed: number, world: PlaceWorld = {}): Plant | null {
  const h0 = hash3(ix, iz, seed + 7101), h1 = hash3(ix, iz, seed + 7102), h2 = hash3(ix, iz, seed + 7103);
  const jx = 0.08 + 0.84 * u16(h0, 0), jz = 0.08 + 0.84 * u16(h0, 16);
  const keep = u16(h1, 0), pick = u16(h1, 16);
  const bush = u16(h2, 0), zoneR = u16(h2, 16);
  let x = (ix + jx) * CELL, z = (iz + jz) * CELL;
  if (world.onRuin?.(x, z)) return null;

  const region = noise01(x, z, PLACE.region.wavelength, seed + PLACE.region.seed);
  const site = siteAt(x, z, seed, region, zoneR);
  if (!site) return null;
  const { e, shore, slope, hollow, knoll, Ts, dom, mask, open, level, fill } = site;
  const R = RULES[ZONES[site.zone]], D = RULES[ZONES[dom]];

  const steep = smooth(0.5, 0.9, slope);
  const snow = (sp: Species) => Ts < 0.5 && (sp === 'pine' || sp === 'dead' || sp === 'bush' || sp === 'birch');
  const plant = (species: Species, feature: Feature, scale: number, palette = -1): Plant => ({ species, feature, x, z, scale, snow: snow(species), palette });

  // Lone tree: this cell hosts its coarse cell's spot?
  const kl = D.loneGrid / CELL;
  const lx = fdiv(ix, kl), lz = fdiv(iz, kl);
  const lh = hash3(lx, lz, seed + 7221), lp = hash3(lx, lz, seed + 7222);
  const fx = (0.15 + 0.7 * u16(lh, 0)) * kl, fz = (0.15 + 0.7 * u16(lh, 16)) * kl;
  if (ix === lx * kl + Math.floor(fx) && iz === lz * kl + Math.floor(fz)) {
    // Desert plants come in patches; elsewhere lone trees favour gentle rises.
    const desert = dom === DESERT ? PLACE.cactus.patch * smooth(0.55, 0.75, noise01(x, z, PLACE.cactus.wavelength, seed + PLACE.cactus.seed)) : 0;
    const park = PLACE.parkland.lo + PLACE.parkland.span * smooth(0.3, 0.7, noise01(x, z, PLACE.parkland.wavelength, seed + PLACE.parkland.seed));
    const p = (D.loneP + desert) * park * (PLACE.lone.flat + PLACE.lone.knoll * knoll) * (1 - smooth(0.35, 0.6, slope)) * open;
    if (u16(lp, 0) < p) {
      const x2 = (lx * kl + fx) * CELL, z2 = (lz * kl + fz) * CELL;
      // (its own spot, a few metres off: dry and not on a cliff's edge there too)
      surfaceAt(x2, z2, seed, _sf, 1);
      const e2 = _sf[0], w2 = _sf[1];
      const s2 = Math.hypot(surfaceAt(x2 + 1.5, z2, seed, _sf, 1)[0] - e2, surfaceAt(x2, z2 + 1.5, seed, _sf, 1)[0] - e2) / 1.5;
      if (e2 >= w2 + 0.4 && s2 < 0.6) {
        x = x2;
        z = z2;
        return plant(pairPick(D.loneSp, u16(lp, 16)), 'lone', 1.15 + 0.2 * pick);
      }
    }
  }

  const glade = smooth(PLACE.glade.lo, PLACE.glade.hi, noise01(x, z, PLACE.glade.wavelength, seed + PLACE.glade.seed));
  const fD = fill * Math.pow(mask, 1.4) * (1 - PLACE.glade.cut * glade) * (1 - smooth(0.8, 1.15, slope));

  // Grove: its coarse cell's clump (kept inside the cell, so no neighbour can reach here).
  const kg = D.groveGrid / CELL;
  const gx = fdiv(ix, kg), gz = fdiv(iz, kg);
  const gh = hash3(gx, gz, seed + 7211), gp = hash3(gx, gz, seed + 7212);
  const gr = D.groveR[0] + (D.groveR[1] - D.groveR[0]) * u16(gp, 16);
  const inset = gr / D.groveGrid;
  const cx = (gx + inset + (1 - 2 * inset) * u16(gh, 0)) * D.groveGrid, cz = (gz + inset + (1 - 2 * inset) * u16(gh, 16)) * D.groveGrid;
  const gd = Math.hypot(x - cx, z - cz);
  const park = PLACE.parkland.lo + PLACE.parkland.span * smooth(0.3, 0.7, noise01(cx, cz, PLACE.parkland.wavelength, seed + PLACE.parkland.seed));
  const grove = u16(gp, 0) < D.groveP * park * (0.5 + hollow) * open;
  const gD = grove ? D.groveFill * (1 - smooth(0.45 * gr, gr, gd)) * (1 - steep) : 0;

  // Shore: a broken line just above the water beside it (the sea, a lake, a river at its own level).
  const gaps = smooth(0.35, 0.55, noise01(x, z, PLACE.gaps.wavelength, seed + PLACE.gaps.seed));
  const sD = shore > -Infinity ? D.shore * smooth(shore + 0.4, shore + 1.2, e) * (1 - smooth(shore + 3, shore + 6.5, e)) * (1 - steep) * gaps * open : 0;

  const stand = noise01(x, z, PLACE.stand.wavelength, seed + PLACE.stand.seed);
  if (keep < fD) {
    // Old giants: their share of the zone's forest trees, a little larger or smaller by the pick.
    const giant = pick < R.giant[0] ? R.giant[1] * (0.85 + 0.3 * (pick / Math.max(R.giant[0], 1e-6))) : 1;
    return plant(pairPick(R.forestSp, stand), 'forest', (0.95 + 0.2 * pick) * (0.78 + 0.22 * mask) * giant, palPick(R, stand));
  }
  if (keep < fD + gD) return plant(pairPick(D.groveSp, u16(gh, 8)), 'grove', (0.92 + 0.2 * pick) * (1.08 - 0.16 * gd / gr), palPick(D, stand));
  if (keep < fD + gD + sD) return plant(D.shoreSp, 'shore', 0.85 + 0.2 * pick);

  // Bushes: sparse in the open, lining forest edges, a few under the trees.
  const edge = 4 * mask * (1 - mask);
  const bushP = (R.shrub * open + 0.12 * edge * (level < 1 ? 1 : 0) + 0.03 * mask) * (1 - steep);
  if (bush < bushP) return plant('bush', 'bush', 0.8 + 0.4 * pick);
  return null;
}

