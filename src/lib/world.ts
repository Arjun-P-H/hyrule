/**
 * The two worlds: the land, and the cloud world where the sky islands float
 * (reached through the portal: portal.ts).
 * ---------------------------------------------------------------------------
 * A world is a seed. The cloud world's seed is the user's seed + 2^32: every
 * hash truncates seeds to 32 bits (`seed | 0`) and the GPU gets them as i32,
 * so the islands come out as the user's seed lays them out, while every cache
 * keyed by seed keeps the two worlds apart. Generators ask worldOf(seed).
 */

export type World = 'land' | 'clouds';

/** Added to a user's seed (0 … 2^32 − 1) in the cloud world. */
const CLOUD_TAG = 2 ** 32;

/** The seed the world generators see for the user's `seed` in `world`. */
export function worldSeed(seed: number, world: World): number {
  return world === 'clouds' ? seed + CLOUD_TAG : seed;
}

/** The user's seed behind a generator's seed. */
export function userSeed(seed: number): number {
  return seed >= CLOUD_TAG ? seed - CLOUD_TAG : seed;
}

/** Which world a generator's seed belongs to. */
export function worldOf(seed: number): World {
  return seed >= CLOUD_TAG ? 'clouds' : 'land';
}
