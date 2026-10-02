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
 * - Shores: a broken line of trees just above lakes and rivers (birches up
 *   north, palms in the warm south, oases in dry land).
 * - Bushes: sparse in the open, thicker along forest edges.
 *
 * How much of each, and which species, comes from the climate zone, blended
 * across zone borders: temperate, autumn woods, warm lowland, jungle,
 * savanna, desert, mountain, alpine.
 */

import { VALLEY } from '../constants';
import { elevation, hash3, moisture, temperature, vnoise } from '../terrainMath';
import { CELL, type Species } from './species';

export const ZONES = ['temperate', 'autumn', 'warm', 'jungle', 'savanna', 'desert', 'mountain', 'alpine'] as const;
type Zone = (typeof ZONES)[number];
/** A species, its alternate, and the share of the alternate. */
type Pair = readonly [Species, Species, number];

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
};

export const RULES: Record<Zone, Rule> = {
  temperate: {
    forest: 0.63, fill: 0.4, forestSp: ['oak', 'birch', 0.3],
    groveGrid: 132, groveP: 0.7, groveR: [16, 36], groveFill: 0.45, groveSp: ['oak', 'birch', 0.35],
    loneGrid: 48, loneP: 0.7, loneSp: ['oak', 'birch', 0.12],
    shore: 0.07, shoreSp: 'birch', shrub: 0.024,
  },
  autumn: {
    forest: 0.62, fill: 0.42, forestSp: ['maple', 'oak', 0.2],
    groveGrid: 132, groveP: 0.7, groveR: [16, 36], groveFill: 0.47, groveSp: ['maple', 'birch', 0.15],
    loneGrid: 48, loneP: 0.7, loneSp: ['maple', 'oak', 0.25],
    shore: 0.07, shoreSp: 'birch', shrub: 0.024,
  },
  warm: {
    forest: 0.675, fill: 0.38, forestSp: ['oak', 'jungle', 0.35],
    groveGrid: 132, groveP: 0.6, groveR: [16, 32], groveFill: 0.42, groveSp: ['oak', 'palm', 0.2],
    loneGrid: 48, loneP: 0.6, loneSp: ['oak', 'palm', 0.25],
    shore: 0.14, shoreSp: 'palm', shrub: 0.02,
  },
  jungle: {
    forest: 0.525, fill: 0.46, forestSp: ['jungle', 'palm', 0.12],
    groveGrid: 120, groveP: 0.4, groveR: [14, 28], groveFill: 0.45, groveSp: ['jungle', 'palm', 0.25],
    loneGrid: 36, loneP: 0.35, loneSp: ['palm', 'jungle', 0.3],
    shore: 0.2, shoreSp: 'palm', shrub: 0.02,
  },
  savanna: {
    forest: 9, fill: 0, forestSp: ['acacia', 'acacia', 0],
    groveGrid: 150, groveP: 0.25, groveR: [10, 18], groveFill: 0.3, groveSp: ['acacia', 'acacia', 0],
    loneGrid: 42, loneP: 0.42, loneSp: ['acacia', 'acacia', 0],
    shore: 0.16, shoreSp: 'palm', shrub: 0.02,
  },
  desert: {
    forest: 9, fill: 0, forestSp: ['cactus', 'cactus', 0],
    groveGrid: 150, groveP: 0, groveR: [10, 10], groveFill: 0, groveSp: ['cactus', 'cactus', 0],
    loneGrid: 24, loneP: 0.07, loneSp: ['cactus', 'dead', 0.08],
    shore: 0.2, shoreSp: 'palm', shrub: 0.003,
  },
  mountain: {
    forest: 0.535, fill: 0.55, forestSp: ['pine', 'birch', 0.12],
    groveGrid: 120, groveP: 0.35, groveR: [12, 26], groveFill: 0.5, groveSp: ['pine', 'birch', 0.2],
    loneGrid: 36, loneP: 0.35, loneSp: ['pine', 'dead', 0.06],
    shore: 0.03, shoreSp: 'birch', shrub: 0.014,
  },
  alpine: {
    forest: 0.8, fill: 0.3, forestSp: ['pine', 'pine', 0],
    groveGrid: 150, groveP: 0.2, groveR: [8, 14], groveFill: 0.35, groveSp: ['pine', 'pine', 0],
    loneGrid: 48, loneP: 0.25, loneSp: ['pine', 'dead', 0.45],
    shore: 0, shoreSp: 'birch', shrub: 0.004,
  },
};

/** Shared constants: noise wavelengths (m), seed offsets, thresholds. */
export const PLACE = {
  warp: { wavelength: 450, amount: 150, seed: 7201 },
  field: [[700, 0.62, 7203], [260, 0.28, 7204], [110, 0.1, 7205]] as ReadonlyArray<readonly [number, number, number]>,
  /** Parkland vs open plain (~2 km): scales the chance of groves and lone trees by lo..lo + span. */
  parkland: { wavelength: 2100, seed: 7236, lo: 0.6, span: 0.8 }, // offset puts the default spawn in parkland
  /** Lone trees: chance multiplier on flat ground, plus this much on a knoll. */
  lone: { flat: 0.75, knoll: 0.5 },
  /** Half width of the forest edge band, in field units. */
  edge: 0.045,
  glade: { wavelength: 130, seed: 7206, lo: 0.64, hi: 0.74, cut: 0.9 },
  stand: { wavelength: 210, seed: 7207 },
  region: { wavelength: 4200, seed: 7029, lo: 0.72, hi: 0.8 }, // autumn woods; offset keeps the default spawn green
  cactus: { wavelength: 260, seed: 7209 },
  gaps: { wavelength: 90, seed: 7208 },
  /** Relief: a Laplacian over this step (m), from heights prefiltered for dp (m): hollows and knolls
   *  for groves and lone trees; the hill-scale one (and its slope) for forest bodies. */
  relief: { step: 36, dp: 12 },
  hills: { step: 84, dp: 24 },
  /** Chance a forest tree is an old giant (scale × 1.3). */
  giant: 0.04,
} as const;

// ---- helpers (the WGSL mirrors each) ---------------------------------------------

const smooth = (lo: number, hi: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
};
const u16 = (h: number, shift: number) => ((h >>> shift) & 0xffff) / 65535;
/** Floor division for integers (negative too). */
const fdiv = (a: number, b: number) => Math.floor(a / b);

/** Zone weights for a climate (before normalising); 0 everywhere above the tree line. */
function zoneWeights(T: number, M: number, region: number): number[] {
  const cold = 1 - smooth(1.5, 5.5, T);
  const alpine = 1 - smooth(-1.0, 1.0, T);
  const treeline = smooth(-3.5, -1.5, T);
  const warm = smooth(12.2, 13.2, T);
  const desert = 1 - smooth(0.14, 0.26, M);
  const dry = 1 - smooth(0.34, 0.5, M);
  const wet = smooth(0.85, 1.05, M);
  const autumn = smooth(PLACE.region.lo, PLACE.region.hi, region);
  const mild = (1 - cold) * (1 - warm) * (1 - dry);
  return [
    mild * (1 - autumn),                              // temperate
    mild * autumn,                                    // autumn
    warm * (1 - wet) * (1 - dry) * (1 - cold),        // warm
    warm * wet * (1 - dry),                           // jungle
    dry * (1 - desert) * (1 - cold),                  // savanna
    desert * (1 - cold),                              // desert
    cold * (1 - alpine),                              // mountain
    alpine * treeline,                                // alpine
  ];
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

type Feature = 'forest' | 'grove' | 'lone' | 'shore' | 'bush';
export type Plant = { species: Species; feature: Feature; x: number; z: number; scale: number; snow: boolean };

/** Optional world context: ruins stay bare. (The sky islands' trees are the forest's other layer: skyworld/flora.ts.) */
export type PlaceWorld = { onRuin?: (x: number, z: number) => boolean };

const W = VALLEY.wLevel;
const DESERT = ZONES.indexOf('desert');

type Site = {
  e: number; slope: number; hollow: number; knoll: number; T: number; M: number;
  /** This spot's zone (picked by weight with the given random, so borders blend) and the dominant one. */
  zone: number; dom: number;
  /** Forest level and core fill blended over the zones; the forest body here (0..1); openness. */
  level: number; fill: number; mask: number; open: number;
};

/** Ground, climate, zones and the forest body at world (x, z); null on water or above the tree line. */
function siteAt(x: number, z: number, seed: number, region: number, zoneR: number): Site | null {
  // Height, local slope, and relief (+ hollow, − knoll) over 36 m.
  const e = elevation(x, z, seed, 1);
  if (e < W + 0.4) return null;
  const slope = Math.hypot(elevation(x + 1.5, z, seed, 1) - e, elevation(x, z + 1.5, seed, 1) - e) / 1.5;
  const L = PLACE.relief.step, dp = PLACE.relief.dp;
  const e0 = elevation(x, z, seed, dp);
  const relief = (elevation(x + L, z, seed, dp) + elevation(x - L, z, seed, dp) + elevation(x, z + L, seed, dp) + elevation(x, z - L, seed, dp) - 4 * e0) / L;
  // Forest bodies follow the hills (84 m), not every hummock.
  const H = PLACE.hills.step, hp = PLACE.hills.dp;
  const h0 = elevation(x, z, seed, hp);
  const hE = elevation(x + H, z, seed, hp), hW = elevation(x - H, z, seed, hp), hN = elevation(x, z + H, seed, hp), hS = elevation(x, z - H, seed, hp);
  const hills = (hE + hW + hN + hS - 4 * h0) / H;
  const hillSlope = Math.hypot(hE - hW, hN - hS) / (2 * H);
  const hollow = smooth(0.03, 0.15, relief), knoll = smooth(0.03, 0.15, -relief);
  const T = temperature(x, z, e), M = moisture(x, z, e, seed);
  const zw = zoneWeights(T, M, region);
  const total = zw.reduce((p, q) => p + q, 0);
  if (total < 1e-4) return null;
  let zone = 0, dom = 0, acc = 0;
  for (let k = 0; k < zw.length; k++) {
    acc += zw[k] / total;
    if (zoneR >= acc) zone = k + 1;
    if (zw[k] > zw[dom]) dom = k;
  }
  let level = 0, fill = 0;
  ZONES.forEach((name, k) => { level += zw[k] * RULES[name].forest; fill += zw[k] * RULES[name].fill; });
  level /= total;
  fill /= total;
  // The forest body: the field plus how well the ground suits a forest.
  const suit = 0.1 * smooth(0.02, 0.12, hills) - 0.12 * smooth(0.02, 0.12, -hills) + 0.15 * (M - 0.62) - 0.3 * smooth(0.55, 0.9, hillSlope);
  const mask = smooth(level - PLACE.edge, level + PLACE.edge, forestField(x, z, seed) + suit);
  return { e, slope, hollow, knoll, T, M, zone: Math.min(zone, zw.length - 1), dom, level, fill, mask, open: 1 - smooth(0.15, 0.4, mask) };
}

/** What the far forest needs of the ground: its height, slope, hill-scale relief (84 m) and that relief's slope, moisture. */
type CanopyGround = { e: number; slope: number; hills: number; hillSlope: number; M: number };
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
 * Null on water or above the tree line.
 */
function canopyFrom(x: number, z: number, seed: number, g: CanopyGround): Canopy | null {
  if (g.e < W + 0.4) return null;
  const T = temperature(x, z, g.e);
  const zw = zoneWeights(T, g.M, noise01(x, z, PLACE.region.wavelength, seed + PLACE.region.seed));
  let total = 0, dom = 0, level = 0, fill = 0;
  ZONES.forEach((name, k) => {
    total += zw[k];
    if (zw[k] > zw[dom]) dom = k;
    level += zw[k] * RULES[name].forest;
    fill += zw[k] * RULES[name].fill;
  });
  if (total < 1e-4) return null;
  level /= total;
  fill /= total;
  const suit = 0.1 * smooth(0.02, 0.12, g.hills) - 0.12 * smooth(0.02, 0.12, -g.hills) + 0.15 * (g.M - 0.62) - 0.3 * smooth(0.55, 0.9, g.hillSlope);
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
    snow: T < 0.5 && (species === 'pine' || species === 'dead' || species === 'bush' || species === 'birch'),
  };
}

/** canopyFrom on the ground at (x, z) as plantInCell sees it (surveys, tests). */
export function forestCanopy(x: number, z: number, seed: number): Canopy | null {
  const e = elevation(x, z, seed, 1);
  const slope = Math.hypot(elevation(x + 1.5, z, seed, 1) - e, elevation(x, z + 1.5, seed, 1) - e) / 1.5;
  const H = PLACE.hills.step, hp = PLACE.hills.dp;
  const h0 = elevation(x, z, seed, hp);
  const hE = elevation(x + H, z, seed, hp), hW = elevation(x - H, z, seed, hp), hN = elevation(x, z + H, seed, hp), hS = elevation(x, z - H, seed, hp);
  return canopyFrom(x, z, seed, {
    e, slope, hills: (hE + hW + hN + hS - 4 * h0) / H, hillSlope: Math.hypot(hE - hW, hN - hS) / (2 * H), M: moisture(x, z, e, seed),
  });
}

/** The dominant zone and the forest body at world (x, z) (surveys, tests); null on water or above the tree line. */
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
  const { e, slope, hollow, knoll, T, dom, mask, open, level, fill } = site;
  const R = RULES[ZONES[site.zone]], D = RULES[ZONES[dom]];

  const steep = smooth(0.5, 0.9, slope);
  const snow = (sp: Species) => T < 0.5 && (sp === 'pine' || sp === 'dead' || sp === 'bush' || sp === 'birch');
  const plant = (species: Species, feature: Feature, scale: number): Plant => ({ species, feature, x, z, scale, snow: snow(species) });

  // Lone tree: this cell hosts its coarse cell's spot?
  const kl = D.loneGrid / CELL;
  const lx = fdiv(ix, kl), lz = fdiv(iz, kl);
  const lh = hash3(lx, lz, seed + 7221), lp = hash3(lx, lz, seed + 7222);
  const fx = (0.15 + 0.7 * u16(lh, 0)) * kl, fz = (0.15 + 0.7 * u16(lh, 16)) * kl;
  if (ix === lx * kl + Math.floor(fx) && iz === lz * kl + Math.floor(fz)) {
    // Desert plants come in patches; elsewhere lone trees favour gentle rises.
    const desert = dom === DESERT ? 0.4 * smooth(0.55, 0.75, noise01(x, z, PLACE.cactus.wavelength, seed + PLACE.cactus.seed)) : 0;
    const park = PLACE.parkland.lo + PLACE.parkland.span * smooth(0.3, 0.7, noise01(x, z, PLACE.parkland.wavelength, seed + PLACE.parkland.seed));
    const p = (D.loneP + desert) * park * (PLACE.lone.flat + PLACE.lone.knoll * knoll) * (1 - smooth(0.35, 0.6, slope)) * open;
    if (u16(lp, 0) < p) {
      x = (lx * kl + fx) * CELL;
      z = (lz * kl + fz) * CELL;
      return plant(pairPick(D.loneSp, u16(lp, 16)), 'lone', 1.15 + 0.2 * pick);
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

  // Shore: a broken line just above the water.
  const gaps = smooth(0.35, 0.55, noise01(x, z, PLACE.gaps.wavelength, seed + PLACE.gaps.seed));
  const sD = D.shore * smooth(W + 0.4, W + 1.2, e) * (1 - smooth(W + 3, W + 6.5, e)) * (1 - steep) * gaps * open;

  const stand = noise01(x, z, PLACE.stand.wavelength, seed + PLACE.stand.seed);
  if (keep < fD) {
    const giant = pick < PLACE.giant ? 1.3 : 1;
    return plant(pairPick(R.forestSp, stand), 'forest', (0.95 + 0.2 * pick) * (0.78 + 0.22 * mask) * giant);
  }
  if (keep < fD + gD) return plant(pairPick(D.groveSp, u16(gh, 8)), 'grove', (0.92 + 0.2 * pick) * (1.08 - 0.16 * gd / gr));
  if (keep < fD + gD + sD) return plant(D.shoreSp, 'shore', 0.85 + 0.2 * pick);

  // Bushes: sparse in the open, lining forest edges, a few under the trees.
  const edge = 4 * mask * (1 - mask);
  const bushP = (R.shrub * open + 0.12 * edge * (level < 1 ? 1 : 0) + 0.03 * mask) * (1 - steep);
  if (bush < bushP) return plant('bush', 'bush', 0.8 + 0.4 * pick);
  return null;
}
