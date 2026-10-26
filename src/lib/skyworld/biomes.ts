/**
 * Sky biomes: the realm's own climates, unlike anything on the ground.
 * ---------------------------------------------------------------------------
 * The ground's biomes follow temperature (latitude, altitude) and moisture
 * (the sea, rain). Up here other things decide:
 *   - the cloud deck (LIGHT.cloudPlane): islands that float just above it
 *     bathe in its mist every morning, and grow mossy cloud forest;
 *   - altitude: the air thins and chills with height, so the highest islands
 *     are frost-bound and the high ones windswept;
 *   - size: a small isle has no shelter from the wind and holds no water,
 *     a great one keeps soil, a lake and a sheltered heart;
 *   - its archipelago's trait: islands of one archipelago share a lineage
 *     (verdant, blossoming, frost-touched, or wild), so each reads as a place.
 * skyClimate() turns an island into these factors, pickBiome() into a biome.
 *
 * Each biome sets how its islands look and live: the ground's palette, the
 * grass (height, density, wild flowers), the relief (hills, crags, a frozen
 * summit), the rock of the underside, whether it keeps a lake (frozen, up
 * high), and its trees — the ground's tree shapes in sky colours
 * (SKY_FLORA): golden and blossoming crowns, moss-dark cloud-forest giants,
 * tree ferns, wind-stunted and rime-frosted pines. Linear RGB throughout.
 */

import { ISLAND_LOWER } from './heights';
import { LIGHT } from '../constants';
import type { Species } from '../forest/species';

type V3 = readonly [number, number, number];

export const SKY_BIOMES = ['meadow', 'mistwood', 'blossom', 'crags', 'frost', 'garden', 'ember'] as const;
export type SkyBiome = (typeof SKY_BIOMES)[number];

/** Sky tree palettes: leaves (shade, light) and bark, for the ground's tree shapes up here. */
export const SKY_FLORA = ['golden', 'blossom', 'pearl', 'mossy', 'silver', 'fern', 'gale', 'rime', 'wisteria', 'emberred'] as const;
type Palette = (typeof SKY_FLORA)[number];
export const SKY_FLORA_COLOURS: Record<Palette, { leaf: readonly [V3, V3]; bark: V3 }> = {
  golden: { leaf: [[0.3, 0.2, 0.03], [0.62, 0.48, 0.09]], bark: [0.13, 0.11, 0.08] },
  blossom: { leaf: [[0.42, 0.13, 0.2], [0.85, 0.5, 0.62]], bark: [0.09, 0.06, 0.06] },
  pearl: { leaf: [[0.45, 0.4, 0.42], [0.86, 0.82, 0.8]], bark: [0.11, 0.09, 0.08] },
  mossy: { leaf: [[0.02, 0.08, 0.055], [0.07, 0.2, 0.13]], bark: [0.08, 0.1, 0.065] },
  silver: { leaf: [[0.07, 0.13, 0.11], [0.26, 0.36, 0.32]], bark: [0.24, 0.24, 0.22] },
  fern: { leaf: [[0.04, 0.13, 0.03], [0.13, 0.3, 0.06]], bark: [0.07, 0.05, 0.035] },
  gale: { leaf: [[0.025, 0.065, 0.045], [0.07, 0.14, 0.08]], bark: [0.13, 0.1, 0.07] },
  rime: { leaf: [[0.07, 0.12, 0.12], [0.2, 0.28, 0.29]], bark: [0.13, 0.12, 0.12] },
  // wisteria and jacaranda over palace terraces (SKY_CIVILIZATION.md §9.1), ember maples on warm soil
  wisteria: { leaf: [[0.18, 0.08, 0.3], [0.5, 0.33, 0.75]], bark: [0.1, 0.08, 0.07] },
  emberred: { leaf: [[0.32, 0.025, 0.015], [0.72, 0.11, 0.035]], bark: [0.06, 0.04, 0.035] },
};

/** A tree a biome grows: shape, palette, its share of the stands, scale range, and snow on it. */
type SkyTree = { species: Species; palette: Palette; weight: number; scale: readonly [number, number]; snow?: boolean };

type SkyBiomeInfo = {
  name: string;
  /** Ground: grass at its lushest and driest, bare soil, rock; how much of it lies under snow (0–1). */
  lush: V3; dry: V3; soil: V3; rock: V3; snow: number;
  /** Grass: density (0–1), blade height (× the ground's), wild flowers (share of blades) and their colours. */
  grass: { density: number; height: number; flowers: number; flower: readonly [V3, V3, V3] };
  /** Relief: rolling hills, ridged crags and a central summit, each × the island's hill height. */
  relief: { hill: number; crag: number; peak: number };
  /** The underside's rock strata (two tones) and what covers its rim (moss, bare stone, ice). */
  strata: readonly [V3, V3]; rimCover: V3;
  /** Chance of a lake (on an island large enough), and whether it lies frozen (no stream then). */
  lake: number; frozen: boolean;
  /**
   * Trees (the sky layer of the forest, flora.ts): forest cover where the
   * island's stand field is high, lone trees on a 30 m grid, bushes, and
   * the trees picked by weight within a stand.
   */
  trees: { forest: number; lone: number; bush: number; kinds: readonly SkyTree[]; bushPalette: Palette };
};

export const SKY_BIOME_INFO: Record<SkyBiome, SkyBiomeInfo> = {
  // High pasture: golden-green grass full of flowers, lone golden and blossoming trees.
  meadow: {
    name: 'Sky meadow',
    lush: [0.085, 0.13, 0.03], dry: [0.16, 0.145, 0.05], soil: [0.12, 0.09, 0.06], rock: [0.24, 0.21, 0.17], snow: 0,
    grass: { density: 0.95, height: 1.15, flowers: 0.07, flower: [[0.8, 0.8, 0.74], [0.85, 0.62, 0.07], [0.36, 0.17, 0.62]] },
    relief: { hill: 1, crag: 0.1, peak: 0 },
    strata: [[0.22, 0.17, 0.12], [0.33, 0.27, 0.19]], rimCover: [0.07, 0.1, 0.04],
    lake: 0.7, frozen: false,
    trees: {
      forest: 0.12, lone: 0.5, bush: 0.035, bushPalette: 'golden',
      kinds: [
        { species: 'oak', palette: 'golden', weight: 0.55, scale: [1.0, 1.35] },
        { species: 'maple', palette: 'blossom', weight: 0.25, scale: [0.9, 1.2] },
        { species: 'birch', palette: 'silver', weight: 0.2, scale: [0.9, 1.15] },
      ],
    },
  },
  // Cloud forest just above the deck: deep moss, old dark giants, tree ferns, mist.
  mistwood: {
    name: 'Mistwood',
    lush: [0.045, 0.11, 0.06], dry: [0.07, 0.13, 0.07], soil: [0.05, 0.045, 0.035], rock: [0.13, 0.14, 0.13], snow: 0,
    grass: { density: 1, height: 0.6, flowers: 0.015, flower: [[0.4, 0.55, 0.8], [0.7, 0.75, 0.7], [0.6, 0.3, 0.45]] },
    relief: { hill: 0.9, crag: 0.25, peak: 0 },
    strata: [[0.08, 0.08, 0.075], [0.15, 0.15, 0.14]], rimCover: [0.04, 0.09, 0.05],
    lake: 0.95, frozen: false,
    trees: {
      forest: 0.5, lone: 0.2, bush: 0.08, bushPalette: 'fern',
      kinds: [
        { species: 'oak', palette: 'mossy', weight: 0.5, scale: [1.1, 1.55] },
        { species: 'palm', palette: 'fern', weight: 0.3, scale: [0.55, 0.75] },
        { species: 'birch', palette: 'silver', weight: 0.2, scale: [1.0, 1.3] },
      ],
    },
  },
  // Blossom groves: pink and pearl-white crowns over fresh, lilac-flecked grass.
  blossom: {
    name: 'Petal glade',
    lush: [0.09, 0.14, 0.06], dry: [0.14, 0.13, 0.09], soil: [0.13, 0.08, 0.07], rock: [0.26, 0.2, 0.2], snow: 0,
    grass: { density: 0.9, height: 0.85, flowers: 0.1, flower: [[0.85, 0.35, 0.5], [0.85, 0.8, 0.82], [0.55, 0.35, 0.7]] },
    relief: { hill: 0.7, crag: 0, peak: 0 },
    strata: [[0.25, 0.18, 0.17], [0.35, 0.27, 0.25]], rimCover: [0.07, 0.11, 0.05],
    lake: 0.65, frozen: false,
    trees: {
      forest: 0.3, lone: 0.55, bush: 0.05, bushPalette: 'blossom',
      kinds: [
        { species: 'maple', palette: 'blossom', weight: 0.6, scale: [0.95, 1.3] },
        { species: 'oak', palette: 'pearl', weight: 0.4, scale: [0.9, 1.2] },
      ],
    },
  },
  // Windswept heights: short tawny grass, pale rock knobs, wind-stunted pines.
  crags: {
    name: 'Gale crags',
    lush: [0.1, 0.11, 0.04], dry: [0.17, 0.14, 0.07], soil: [0.14, 0.12, 0.09], rock: [0.23, 0.22, 0.2], snow: 0,
    grass: { density: 0.75, height: 0.7, flowers: 0.02, flower: [[0.8, 0.6, 0.05], [0.8, 0.8, 0.75], [0.85, 0.35, 0.06]] },
    relief: { hill: 1.3, crag: 1, peak: 0.3 },
    strata: [[0.28, 0.27, 0.25], [0.4, 0.38, 0.35]], rimCover: [0.2, 0.19, 0.17],
    lake: 0.25, frozen: false,
    trees: {
      forest: 0.08, lone: 0.25, bush: 0.03, bushPalette: 'gale',
      kinds: [
        { species: 'pine', palette: 'gale', weight: 0.75, scale: [0.55, 0.8] },
        { species: 'dead', palette: 'gale', weight: 0.25, scale: [0.6, 0.9] },
      ],
    },
  },
  // The highest isles: snow, blue ice, frozen lakes, rime-frosted pines.
  frost: {
    name: 'Rime isles',
    lush: [0.12, 0.15, 0.13], dry: [0.2, 0.21, 0.22], soil: [0.25, 0.27, 0.3], rock: [0.22, 0.24, 0.28], snow: 0.75,
    grass: { density: 0.35, height: 0.5, flowers: 0, flower: [[0.8, 0.85, 0.9], [0.8, 0.85, 0.9], [0.8, 0.85, 0.9]] },
    relief: { hill: 1.1, crag: 0.5, peak: 1 },
    strata: [[0.2, 0.22, 0.26], [0.3, 0.33, 0.38]], rimCover: [0.55, 0.65, 0.75],
    lake: 0.5, frozen: true,
    trees: {
      forest: 0.12, lone: 0.2, bush: 0.01, bushPalette: 'rime',
      kinds: [
        { species: 'pine', palette: 'rime', weight: 0.8, scale: [0.7, 1.05], snow: true },
        { species: 'dead', palette: 'rime', weight: 0.2, scale: [0.6, 0.9], snow: true },
      ],
    },
  },
  // The two the Eyries add (eyrie/classes.ts): gardens tended round their buildings, and the forge isles' ash.
  // Tended gardens: mown grass full of flowers, orchards of blossoming trees, wisteria; few groves.
  garden: {
    name: 'Eyrie gardens',
    lush: [0.07, 0.135, 0.035], dry: [0.12, 0.14, 0.055], soil: [0.12, 0.09, 0.06], rock: [0.3, 0.27, 0.22], snow: 0,
    grass: { density: 0.85, height: 0.7, flowers: 0.09, flower: [[0.88, 0.42, 0.6], [0.88, 0.86, 0.82], [0.5, 0.36, 0.78]] },
    relief: { hill: 0.45, crag: 0, peak: 0 },
    strata: [[0.3, 0.26, 0.2], [0.42, 0.37, 0.3]], rimCover: [0.07, 0.1, 0.04],
    lake: 0.85, frozen: false,
    trees: {
      forest: 0.06, lone: 0.42, bush: 0.05, bushPalette: 'blossom',
      kinds: [
        { species: 'maple', palette: 'blossom', weight: 0.45, scale: [0.95, 1.3] },
        { species: 'oak', palette: 'pearl', weight: 0.3, scale: [0.9, 1.2] },
        { species: 'maple', palette: 'wisteria', weight: 0.25, scale: [0.9, 1.15] },
      ],
    },
  },
  // Ash and basalt: black soil, sparse tawny grass, ember maples, dead trees.
  ember: {
    name: 'Ember isles',
    lush: [0.06, 0.06, 0.035], dry: [0.11, 0.085, 0.05], soil: [0.045, 0.038, 0.034], rock: [0.07, 0.064, 0.062], snow: 0,
    grass: { density: 0.4, height: 0.6, flowers: 0.02, flower: [[0.85, 0.25, 0.05], [0.9, 0.55, 0.1], [0.6, 0.1, 0.05]] },
    relief: { hill: 1.2, crag: 1, peak: 0.35 },
    strata: [[0.045, 0.04, 0.04], [0.1, 0.09, 0.085]], rimCover: [0.08, 0.06, 0.05],
    lake: 0, frozen: false,
    trees: {
      forest: 0.04, lone: 0.3, bush: 0.02, bushPalette: 'gale',
      kinds: [
        { species: 'maple', palette: 'emberred', weight: 0.7, scale: [0.85, 1.2] },
        { species: 'dead', palette: 'gale', weight: 0.3, scale: [0.6, 0.9] },
      ],
    },
  },
};

/** Each archipelago's lineage, shared by its islands (a lone isle draws its own). */
export const SKY_TRAITS = ['verdant', 'blossom', 'frost', 'wild'] as const;
export type SkyTrait = (typeof SKY_TRAITS)[number];

const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/**
 * What an island's place in the sky makes of its climate (each 0–1), from
 * its top's base height, the lowest its underside reaches (`tip`: an island
 * whose roots hang in the cloud deck drinks its mist) and its radius.
 */
export function skyClimate(top: number, tip: number, radius: number): { cloud: number; cold: number; high: number; exposed: number } {
  // (judged at the height the island had before the heights in use pulled it down, heights.ts: low islands keep
  // the original mix of biomes, the Ice Watch its frost)
  top += ISLAND_LOWER;
  tip += ISLAND_LOWER;
  const deck = LIGHT.cloudPlane;
  return {
    cloud: 1 - smooth(deck + 100, deck + 700, tip),  // its underside steeped in the deck's mist
    cold: smooth(2450, 2800, top),                    // frost-bound
    high: smooth(2150, 2600, top),                    // thin, windy air
    exposed: 1 - smooth(40, 260, radius),             // no shelter from the wind
  };
}

/**
 * The biome of an island (its top's base height, its underside's tip, its
 * outline radius) in an archipelago with `trait`; `jitter` in [0, 1) breaks
 * near-ties the same way every time.
 */
export function pickBiome(top: number, tip: number, radius: number, trait: SkyTrait, jitter: number): SkyBiome {
  const c = skyClimate(top, tip, radius);
  const size = 1 - c.exposed;
  const score: Record<SkyBiome, number> = {
    meadow: 0.55 + 0.1 * size - 0.4 * c.cold - 0.15 * c.cloud + (trait === 'wild' ? 0.1 : 0),
    mistwood: 0.8 * c.cloud + 0.1 * size - 0.6 * c.cold - 0.35 * c.exposed + (trait === 'verdant' ? 0.35 : trait === 'blossom' ? -0.3 : 0),
    blossom: 0.2 - 0.5 * c.cold + (trait === 'blossom' ? 0.85 : 0),
    crags: 0.1 + 0.45 * c.exposed + 0.35 * c.high - 0.25 * c.cloud + (trait === 'wild' ? 0.15 : 0),
    frost: 1.3 * c.cold + (trait === 'frost' ? 0.7 * c.high : 0),
    // (given by an Eyrie's class, never by the climate)
    garden: -Infinity,
    ember: -Infinity,
  };
  let best: SkyBiome = 'meadow';
  let bestScore = -Infinity;
  SKY_BIOMES.forEach((b, k) => {
    // A small hashed nudge per biome, so islands on the border between two biomes split between them.
    const s = score[b] + 0.22 * ((jitter * (k + 1) * 7.31) % 1);
    if (s > bestScore) { bestScore = s; best = b; }
  });
  return best;
}
