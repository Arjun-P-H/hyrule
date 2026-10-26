/**
 * The cloud world's ground: a sea of cumulus (CLOUDS; the WGSL twin is
 * gpu/worldgen.wgsl.ts cloudSurface). Heaped domes of four sizes on a broad
 * swell, each |noise| (rounded on top, creased between), so seen from above
 * it rolls as cloud tops do; the drawn lobes on its crests and round the
 * islands (puffs.ts billowsNear) round its silhouettes. It is what
 * elevation() returns in the cloud world (terrainMath.ts landSurface): solid,
 * dry, walkable. Pure functions of (x, z, seed); no allocation per point.
 */

import { CLOUDS as V } from '../constants';
import { vnoise } from '../noise';

const BMID = (V.base[0] + V.base[1]) / 2, BHALF = (V.base[1] - V.base[0]) / 2;

/** The cloud sea's height at world (x, z) (m); `dp` is the footprint (the finest heaps fade under it). */
export function cloudSurface(x: number, z: number, seed: number, dp: number): number {
  const fine = 1 - Math.min(1, dp / 35);
  const D = V.domes;
  const e = BMID + BHALF * vnoise(x / 2400, z / 2400, seed + 8101)
    + D[0] * Math.abs(vnoise(x / 1100, z / 1100, seed + 8102))
    + D[1] * Math.abs(vnoise(x / 380, z / 380, seed + 8103))
    + D[2] * fine * Math.abs(vnoise(x / 120, z / 120, seed + 8105))
    + D[3] * fine * Math.abs(vnoise(x / 36, z / 36, seed + 8106));
  return Math.min(e, V.ceiling);
}
