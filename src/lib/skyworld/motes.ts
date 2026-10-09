/**
 * What drifts in the air round the camera up here (SKY_CIVILIZATION.md §9):
 * petals falling in the orchards, snow at Rimewatch and on the frost isles,
 * sparks rising at Cinderforge, leaves falling from the Elder Bough; by day
 * petalwing butterflies over the meadows and gardens, by dusk the
 * lantern-flies rising from water and meadow. A pure function of where the
 * camera is (gpu/motes.ts draws them; motes.wgsl.ts flies them).
 *
 * Only over an island and near its ground: the motes thin out as the camera
 * climbs away from it, and there are none over open sky.
 */

import { SKY_REALM as S } from '../constants';
import { archipelagoInCell, cellBuilt } from './archipelago';
import { islandAt } from './realm';
import { topAt } from './island';

/** Mote kinds (motes.wgsl.ts MOTE_*). */
export const MOTE = { none: 0, petals: 1, snow: 2, embers: 3, leaves: 4, fireflies: 5, butterflies: 6 } as const;

/** The air round a point: what falls or rises (kind, density 0 … 1), what flies by day and by dusk, and the ground's height. */
export type Air = { fall: number; fallK: number; day: number; dayK: number; dusk: number; duskK: number; ground: number };

const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** The motes round world (x, y, z) into `out` (never builds an archipelago: none over one not laid out). */
export function airAt(x: number, y: number, z: number, seed: number, out: Air): Air {
  out.fall = out.day = out.dusk = 0; out.fallK = out.dayK = out.duskK = MOTE.none; out.ground = y;
  const ix = Math.floor(x / S.archCell), iz = Math.floor(z / S.archCell);
  if (!cellBuilt('arch', ix, iz, seed)) return out;
  const isl = islandAt(x, z, seed);
  if (!isl) return out;
  const ground = topAt(isl, x, z);
  // (thinning out as the camera climbs from the ground)
  const near = 1 - smooth(25, 90, y - ground);
  if (near <= 0) return out;
  out.ground = ground;
  const a = archipelagoInCell(ix, iz, seed);
  const cls = a && a.islands.includes(isl) ? a.cls : null;
  const b = isl.biome;
  if (b === 'frost' || cls === 'rime') { out.fallK = MOTE.snow; out.fall = near; }
  else if (cls === 'ember' || b === 'ember') { out.fallK = MOTE.embers; out.fall = 0.6 * near; }
  else if (cls === 'bough') { out.fallK = MOTE.leaves; out.fall = 0.7 * near; }
  else if (b === 'blossom' || cls === 'petal' || cls === 'fallen') { out.fallK = MOTE.petals; out.fall = near; }
  else if (b === 'garden') { out.fallK = MOTE.petals; out.fall = 0.35 * near; }
  const green = b === 'meadow' || b === 'garden' || b === 'blossom' || b === 'mistwood';
  if (green || cls === 'petal' || cls === 'bough' || cls === 'fallen') { out.dayK = MOTE.butterflies; out.day = near; }
  if (green || cls === 'wells' || cls === 'mirror' || cls === 'bough' || isl.lake) { out.duskK = MOTE.fireflies; out.dusk = near; }
  return out;
}
