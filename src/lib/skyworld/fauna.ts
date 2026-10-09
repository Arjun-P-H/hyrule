/**
 * The living sky's birds (SKY_CIVILIZATION.md §9.2): the flocks round the
 * camera, as a pure function of the realm (gpu/fauna.ts draws them, and
 * fauna.wgsl.ts flies each bird).
 * ---------------------------------------------------------------------------
 * Every island of an Eyrie large enough keeps a flock wheeling round it, of
 * the birds of its class: cloud-swifts round the spires, sky-herons low over
 * the lagoons and Mistwells, white birds at Ringhold, gulls at Galeharbour,
 * crows at Cinderforge, snow-hawks at Rimewatch. A pair of gyre-hawks soars
 * in wide circles high over every Eyrie; round each Thread swifts circle by
 * night as by day; wild swifts and crows keep the lone isles.
 *
 * A flock is a circle (its centre, radius, height and pace) that its birds fly
 * round together, each a little off it and wandering; where a bird is at a
 * time is the shader's. Birds roost by night, but for the Thread's swifts.
 */

import { hashU } from '../terrainMath';
import { rng } from './kit/mesh';
import { realmNear, type PendingCell } from './realm';
import { eyrieIfBuilt } from './eyrie/eyrie';
import type { EyrieClass } from './eyrie/classes';
import type { SkyIsland } from './island';

export const SPECIES = ['swift', 'hawk', 'heron', 'crow', 'gull', 'dove', 'snowhawk'] as const;
type Species = (typeof SPECIES)[number];

export type Flock = {
  x: number; y: number; z: number;  // its circle's centre (world m)
  r: number;                         // and radius (m)
  speed: number;                     // m/s along it (its sign the way round)
  count: number;                     // birds (at most FLOCK_BIRDS)
  species: number;                   // SPECIES index
  seed: number;                      // 16 bits
  night: boolean;                    // flies by night too
};

/** Birds a flock may have (fauna.wgsl.ts draws this many instances per flock, hiding the rest). */
export const FLOCK_BIRDS = 24;
/** Flocks listed round the camera at most, and how far (m) from it their circles may lie. */
export const MAX_FLOCKS = 64;
export const FLOCK_RANGE = 2200;

const CLASS_BIRDS: Record<EyrieClass, readonly Species[]> = {
  crown: ['swift', 'dove'], mirror: ['heron', 'swift'], halo: ['dove'], beacon: ['swift'], petal: ['swift', 'dove'],
  bough: ['swift', 'dove', 'crow'], ember: ['crow'], rime: ['snowhawk', 'swift'], fallen: ['swift', 'crow', 'dove'],
  harbour: ['gull'], wells: ['heron', 'swift'],
};
const COUNT: Record<Species, readonly [number, number]> = {
  swift: [14, 24], hawk: [1, 1], heron: [3, 6], crow: [10, 18], gull: [8, 14], dove: [10, 16], snowhawk: [1, 1],
};

const scratch: PendingCell[] = [];

/** How far (m) a flock's circle comes to (x, z). */
const reachTo = (f: Flock, x: number, z: number) => Math.max(0, Math.abs(Math.hypot(f.x - x, f.z - z) - f.r) - 20);

/**
 * The flocks whose circles come within FLOCK_RANGE of (x, z), nearest first, at most MAX_FLOCKS,
 * into `out` (cleared). Only Eyries already laid out are asked (never builds one).
 */
export function flocksNear(x: number, z: number, seed: number, out: Flock[]): Flock[] {
  out.length = 0;
  scratch.length = 0;
  const { archipelagos, isles } = realmNear(x, z, FLOCK_RANGE + 1500, seed, scratch);
  const islandFlock = (isl: SkyIsland, sp: Species, salt: number) => {
    const rnd = rng(hashU(isl.seed ^ salt));
    const low = sp === 'heron';
    const [lo, hi] = COUNT[sp];
    out.push({
      x: isl.cx, y: isl.y + (low ? 6 + 10 * rnd() : 30 + 60 * rnd()), z: isl.cz,
      r: isl.R * (low ? 0.25 + 0.3 * rnd() : 0.4 + 0.45 * rnd()),
      speed: (rnd() < 0.5 ? -1 : 1) * (low ? 7 + 2 * rnd() : sp === 'crow' ? 9 + 3 * rnd() : 11 + 5 * rnd()),
      count: lo + Math.floor(rnd() * (hi - lo + 1)), species: SPECIES.indexOf(sp), seed: Math.floor(rnd() * 65536), night: false,
    });
  };
  for (const a of archipelagos) {
    const birds = CLASS_BIRDS[a.cls];
    a.islands.forEach((isl, i) => {
      if (a.roles[i] !== 'debris' && isl.kind !== 'debris' && isl.R >= 50) islandFlock(isl, birds[i % birds.length], 0xb1d5);
    });
    // gyre-hawks (snow-hawks at Rimewatch) soaring high over it
    const hub = a.islands[0];
    const rnd = rng(hashU(a.ix * 7919 + a.iz * 104729 + seed * 13 + 4401));
    for (let k = 0; k < 2; k++) {
      out.push({
        x: hub.cx + (rnd() - 0.5) * 600, y: hub.y + 170 + 150 * rnd(), z: hub.cz + (rnd() - 0.5) * 600, r: 220 + 320 * rnd(),
        speed: (rnd() < 0.5 ? -1 : 1) * (9 + 4 * rnd()), count: 1, species: SPECIES.indexOf(a.cls === 'rime' ? 'snowhawk' : 'hawk'),
        seed: Math.floor(rnd() * 65536), night: false,
      });
    }
    // swifts round each Thread, by night as by day
    const e = a.cls === 'beacon' ? eyrieIfBuilt(a.ix, a.iz, seed) : null;
    if (e) {
      for (const st of e.structures) for (const f of st.fx ?? []) {
        if (f.k !== 'beam') continue;
        out.push({ x: f.x, y: f.y + 60, z: f.z, r: 30 + 15 * rnd(), speed: 13, count: 20, species: SPECIES.indexOf('swift'), seed: Math.floor(rnd() * 65536), night: true });
      }
    }
  }
  for (const isl of isles) if (isl.R >= 60) islandFlock(isl, (isl.seed & 1) === 0 ? 'swift' : 'crow', 0xc0de);
  // nearest first, those in range
  const keep = out.filter((f) => reachTo(f, x, z) <= FLOCK_RANGE).sort((p, q) => reachTo(p, x, z) - reachTo(q, x, z));
  out.length = 0;
  for (const f of keep.slice(0, MAX_FLOCKS)) out.push(f);
  return out;
}
