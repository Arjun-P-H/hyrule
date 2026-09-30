/**
 * Tree species: their names, model sizes, draw distances, levels of detail
 * and colours. Where each grows is placement.ts; their shapes, models.ts.
 */

export const SPECIES = ['oak', 'birch', 'maple', 'pine', 'jungle', 'palm', 'acacia', 'cactus', 'dead', 'bush'] as const;
export type Species = (typeof SPECIES)[number];
/** Model shapes per species, and levels of detail per shape. */
export const VARIANTS = 3;
export const LODS = 3;

type SpeciesInfo = {
  /** Height of the models at scale 1 (m); they stay below 1.15 × this. */
  height: number;
  /** Crown radius at scale 1 (m): every model fits inside it about its trunk. */
  radius: number;
  /** Not drawn beyond this distance (m). */
  maxDist: number;
  /** Distances (m, at scale 1) where LOD 0 gives way to 1, and 1 to 2. */
  lod: readonly [number, number];
  /** Linear RGB: leaves (shade, light), bark, accent (fruit, flowers). */
  leaf: readonly [readonly number[], readonly number[]];
  bark: readonly number[];
  accent: readonly number[];
};

export const SPECIES_INFO: Record<Species, SpeciesInfo> = {
  oak: {
    height: 11, radius: 6.5, maxDist: 1150, lod: [42, 210],
    leaf: [[0.05, 0.13, 0.028], [0.17, 0.3, 0.055]], bark: [0.085, 0.056, 0.035], accent: [0.2, 0.12, 0.05],
  },
  birch: {
    height: 13, radius: 4.4, maxDist: 1150, lod: [40, 200],
    leaf: [[0.11, 0.19, 0.035], [0.32, 0.4, 0.075]], bark: [0.62, 0.6, 0.56], accent: [0.2, 0.2, 0.05],
  },
  maple: {
    height: 10, radius: 5.8, maxDist: 1150, lod: [42, 210],
    leaf: [[0.3, 0.035, 0.018], [0.52, 0.2, 0.025]], bark: [0.078, 0.052, 0.036], accent: [0.45, 0.3, 0.03],
  },
  pine: {
    height: 17, radius: 4.9, maxDist: 1150, lod: [45, 230],
    leaf: [[0.03, 0.085, 0.055], [0.085, 0.19, 0.11]], bark: [0.2, 0.1, 0.06], accent: [0.25, 0.14, 0.06],
  },
  jungle: {
    height: 21, radius: 10.8, maxDist: 1150, lod: [50, 240],
    leaf: [[0.025, 0.11, 0.028], [0.1, 0.26, 0.05]], bark: [0.24, 0.2, 0.15], accent: [0.5, 0.1, 0.25],
  },
  palm: {
    height: 11, radius: 7.2, maxDist: 1000, lod: [40, 200],
    leaf: [[0.06, 0.15, 0.035], [0.2, 0.33, 0.07]], bark: [0.2, 0.15, 0.1], accent: [0.13, 0.075, 0.03],
  },
  acacia: {
    height: 7.5, radius: 7.4, maxDist: 1000, lod: [40, 200],
    leaf: [[0.075, 0.12, 0.028], [0.22, 0.26, 0.07]], bark: [0.16, 0.11, 0.075], accent: [0.45, 0.4, 0.1],
  },
  cactus: {
    height: 5.5, radius: 1.7, maxDist: 650, lod: [30, 140],
    leaf: [[0.05, 0.12, 0.05], [0.13, 0.24, 0.09]], bark: [0.15, 0.27, 0.1], accent: [0.85, 0.72, 0.62],
  },
  dead: {
    height: 8, radius: 4.2, maxDist: 800, lod: [35, 170],
    leaf: [[0.08, 0.075, 0.07], [0.15, 0.14, 0.12]], bark: [0.2, 0.185, 0.165], accent: [0.1, 0.1, 0.1],
  },
  bush: {
    height: 1.6, radius: 1.6, maxDist: 280, lod: [22, 90],
    leaf: [[0.045, 0.12, 0.028], [0.15, 0.27, 0.05]], bark: [0.12, 0.085, 0.055], accent: [0.5, 0.03, 0.05],
  },
};

/** Placement lattice: one candidate per cell of this size (m). */
export const CELL = 6;
