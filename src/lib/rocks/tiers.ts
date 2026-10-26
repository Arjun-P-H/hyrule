/**
 * The rocks' tiers: the lattices the scatter places them on (gpu/rocks.wgsl.ts rockScatter) and their numbers, which the CPU
 * twin of the big ones (collision.ts) reads as well, so what is drawn is what blocks the hero.
 */

import { ROCK_KINDS, ROCK_LODS } from './models';

export type RockTier = {
  /** Lattice cell (m: a power of two) and how far out the tier's rocks are drawn (m; less than 32 cells). */
  cell: number; radius: number;
  /** Rock size (the unit mesh's scale, m) from … to, skewed toward the small. */
  size: [number, number];
  /** The chance a cell holds a rock where the land is rockiest (before the drifts and slopes). */
  chance: number;
  /** The kinds it picks from (ROCK_KINDS indices; a kind listed twice is twice as likely). */
  kinds: number[];
};

export const ROCK_TIERS: RockTier[] = [
  { cell: 0.5, radius: 15, size: [0.07, 0.24], chance: 0.05, kinds: [7, 7, 6, 4, 7, 6] },
  { cell: 2, radius: 62, size: [0.28, 0.95], chance: 0.018, kinds: [0, 1, 2, 4, 6, 6] },
  { cell: 8, radius: 250, size: [1.0, 3.2], chance: 0.014, kinds: [0, 1, 2, 3, 4, 5] },
  { cell: 32, radius: 900, size: [3.2, 8.5], chance: 0.005, kinds: [0, 1, 3, 5, 0, 1] },
];
/** The tiers from this one up are solid (rocks/field.ts makes them on the CPU too): the stones a hero lifts, the boulders it rolls, the landmarks. */
export const ROCK_SOLID_FROM = 1;
/** The tiers from this one up are big enough to stop an arrow or a foe as a circle (a boulder is over a metre across). */
export const ROCK_BLOCKING_FROM = 2;
/** Cells a side of the box round the hero, per tier, whose rocks the CPU owns and lists for the GPU (half the box; 0: none). */
export const ROCK_ZONE: readonly number[] = [0, 12, 5, 2];
/** Rocks the CPU's list holds (its box's and the free ones). */
export const ROCK_LIST_CAP = 512;
/** Cells kept empty after their rock was taken (the oldest gives way). */
export const ROCK_TAKEN_CAP = 128;
/** Cells a side of every tier's lattice. */
export const ROCK_N = 64;
/** Instances a bucket holds (a rock beyond it is not drawn). */
export const ROCK_CAP = 512;
/** A mesh at a level of detail. */
export const ROCK_MESHES = ROCK_KINDS.length * ROCK_LODS;
/** Buckets: two sets of the meshes, the rocks too small to cast a shadow (a pebble's is under a shadow-map texel) and the rest. */
export const ROCK_BUCKETS = 2 * ROCK_MESHES;
/** A rock this big (the unit mesh's scale, m) is in the second set, and casts a shadow. */
export const ROCK_CAST_SIZE = 0.7;
/** Bytes an instance takes in the buffer. */
export const ROCK_INSTANCE_BYTES = 64;
/**
 * vec4f entries of the parameters' buffer: three a tier (camera, first cell, sizes), one of the frame's (the density, the
 * cells taken, the CPU's rocks listed), the box of cells the CPU owns a tier (vec4i), then the cells taken (vec4i: tier, ix, iz).
 */
export const ROCK_PARAMS_VEC4 = ROCK_TIERS.length * 3 + 1 + ROCK_TIERS.length + ROCK_TAKEN_CAP;

/**
 * Per kind (ROCK_KINDS order): the finest mesh's widest horizontal radius, its top and its flat base's height, at unit scale
 * (rounded: models.ts buildRockModels measures them, and tests/rocks.test.ts pins these to its numbers).
 */
export const ROCK_EXTENT: ReadonlyArray<{ rxz: number; top: number; bottom: number }> = [
  { rxz: 1.054, top: 0.804, bottom: -0.462 },
  { rxz: 1.046, top: 0.867, bottom: -0.506 },
  { rxz: 1.425, top: 0.506, bottom: -0.288 },
  { rxz: 0.762, top: 1.22, bottom: -0.91 },
  { rxz: 0.895, top: 0.891, bottom: -0.516 },
  { rxz: 0.55, top: 1.487, bottom: -1.05 },
  { rxz: 1.132, top: 0.664, bottom: -0.33 },
  { rxz: 0.746, top: 0.385, bottom: -0.125 },
];
