/**
 * Boulders for the arrows and the foes: the big rocks round a point as circles, in the shape of a tree trunk's
 * (forest/collision.ts Trunk), so the arrows meet them the way they meet trunks. The rocks themselves are made in
 * rocks/field.ts, as the GPU scatter places them (gpu/rocks.wgsl.ts rockScatter, whose decisions it repeats in double
 * precision); only the tiers from ROCK_BLOCKING_FROM up have circles (a stone stops no arrow), and a rock taken from its
 * cell has none there. The hero meets the rocks by their own shape (field.ts collideRocks), not by these circles.
 */

import { rockCirclesNear, type RockCircle } from './field';

export type { RockCircle };

/**
 * The boulders that can touch a body at (x, z): those of the 3 × 3 cells round it, in every blocking tier (a rock's circle
 * reaches at most ~13 m past its middle, a cell's rock stands within its middle four fifths: the neighbours are enough),
 * and the big rocks lying where they came to rest. Appends to `out` (reuse it: nothing is allocated once cached) and returns it.
 */
export function rocksNear<T extends RockCircle>(x: number, z: number, seed: number, out: T[]): T[] {
  return rockCirclesNear(x, z, seed, out);
}
