/**
 * The lands' air round the camera (biomes.ts AIR): the haze of the biome
 * regions under it, blended by their weights — humid haze over the rainforest
 * and jungle, fog lying in the marshes and the sacred groves, a warm glare
 * over the desert and the badlands, ash over the volcanoes, crisp air in the
 * mountains and on the tundra — fading back to the day's own air as the camera
 * climbs away from the ground. The day–night sky (gpu/sky.ts) multiplies its
 * haze by `haze`, adds `mist` to the morning's mist and tints the haze.
 */

import { AIR, BIOMES, BIOME_COUNT, BIOME_LABEL } from '../biomes';
import { LANDS } from '../constants';
import { landRegions, lastClimate, lastRegions, surfaceAt } from '../terrainMath';

export type BiomeAir = { haze: number; mist: number; tint: [number, number, number] };

export function createBiomeAir(): BiomeAir {
  return { haze: 1, mist: 0, tint: [1, 1, 1] };
}

/** The air fades out between these heights (m) of the camera over the ground. */
const FADE: readonly [number, number] = [150, 900];

const w = new Float64Array(BIOME_COUNT);
const gw: [number, number] = [0, 0];
const smooth = (lo: number, hi: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
};

/** The air at a camera at world (x, y, z) into `out`. */
export function biomeAir(x: number, y: number, z: number, seed: number, out: BiomeAir): BiomeAir {
  surfaceAt(x, z, seed, gw);
  const k = 1 - smooth(FADE[0], FADE[1], y - Math.max(gw[0], gw[1]));
  out.haze = 1;
  out.mist = 0;
  out.tint[0] = out.tint[1] = out.tint[2] = 1;
  if (k <= 0) return out;
  landRegions(x, z, seed, w);
  let haze = 0, mist = 0, r = 0, g = 0, b = 0;
  for (let i = 0; i < BIOME_COUNT; i++) {
    const a = AIR[BIOMES[i]], wi = w[i];
    haze += wi * a.haze;
    mist += wi * a.mist;
    r += wi * a.tint[0];
    g += wi * a.tint[1];
    b += wi * a.tint[2];
  }
  out.haze = 1 + (haze - 1) * k;
  out.mist = mist * k;
  out.tint[0] = 1 + (r - 1) * k;
  out.tint[1] = 1 + (g - 1) * k;
  out.tint[2] = 1 + (b - 1) * k;
  return out;
}

/**
 * The name of the land under world (x, z) for the HUD: the strongest region there, the open sea
 * offshore, the coast in the band where the land ramps down to the sea.
 */
export function landName(x: number, z: number, seed: number): string {
  surfaceAt(x, z, seed, gw);
  const { C } = lastClimate();
  const K = LANDS.coast;
  if (C < K.level) return 'Open sea';
  const r = lastRegions();
  let best = 0;
  for (let i = 1; i < BIOME_COUNT; i++) if (r[i] > r[best]) best = i;
  const name = BIOME_LABEL[BIOMES[best]];
  return C < K.level + 0.5 * K.inland && gw[0] < LANDS.sea + 40 ? `${name} · coast` : name;
}
