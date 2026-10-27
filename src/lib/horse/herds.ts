/**
 * Where the wild horses live: a herd may live in each HORSE.herdCell cell, more surely the more of the cell's middle
 * is grassland (the plains and the savanna; the highland and the lakes' shores a little), never in water or on a
 * steep slope. A pure function of the cell and the seed: the same herd, the same horses (each its own coat and
 * stats) whenever the hero comes back.
 */

import { BI, BIOME_COUNT } from '../biomes';
import { HORSE } from '../constants';
import { hash3, hashU } from '../noise';
import { landRegions, surfaceAt } from '../terrainMath';
import { rng } from '../skyworld/kit/mesh';


export type WildHorse = {
  id: number;          // the same for this horse on every visit
  herd: number;
  x: number; z: number; yaw: number;
  /** The herd's anchor: it wanders round it. */
  ax: number; az: number;
  coat: number;        // the seed of its coat's colouring (horses.wgsl.ts hsCoat): mostly its herd's, its markings its own
  gallop: number;      // × HORSE.gaitSpeed[4]
  spurs: number;       // 2..5
  temper: number;      // 0 (calm) .. 1 (wild): the struggle to tame it lasts longer
};

const W = new Float64Array(BIOME_COUNT);
const S: [number, number] = [0, 0];

/** How much of the land at (x, z) is horse country 0..1. */
export function grassland(x: number, z: number, seed: number): number {
  landRegions(x, z, seed, W);
  return W[BI.plains] + W[BI.savanna] + 0.5 * W[BI.highland] + 0.35 * W[BI.lakes];
}

function dry(x: number, z: number, seed: number): boolean {
  surfaceAt(x, z, seed, S);
  return S[0] > S[1] + 0.3;
}

/** A cell's hash (hash3 alone XORs the coordinates: nearby cells would share it, and their herds with it). */
const cellHash = (cx: number, cz: number, seed: number, salt: number) => hash3(cx, hashU(cz ^ salt), seed);

/** The herd of cell (cx, cz), its horses pushed into out (none: the cell has no herd). */
function herdOf(cx: number, cz: number, seed: number, out: WildHorse[]): void {
  const C = HORSE.herdCell;
  const r = rng(cellHash(cx, cz, seed, 0x40453));
  const ax = (cx + 0.2 + 0.6 * r()) * C, az = (cz + 0.2 + 0.6 * r()) * C;
  if (r() > HORSE.herdChance * Math.min(1, grassland(ax, az, seed) * 1.4)) return;
  if (!dry(ax, az, seed)) return;
  const [lo, hi] = HORSE.herdSize;
  const n = lo + Math.floor(r() * (hi - lo + 1));
  const herd = cellHash(cx, cz, seed, 0x4e7d);

  for (let k = 0; k < n; k++) {
    const a = r() * Math.PI * 2, d = HORSE.herdSpread * Math.sqrt(r());
    const x = ax + Math.cos(a) * d, z = az + Math.sin(a) * d;
    const id = (herd ^ Math.imul(k + 1, 0x9e3779b1)) >>> 0;
    // (a herd's coats run together a little: nearly half its colour, the rest their own)
    const c = r() < 0.45 ? herd : id;
    const gallop = 0.9 + 0.22 * r(), spurs = 2 + Math.floor(r() * 4), temper = r();
    if (!dry(x, z, seed)) continue;
    out.push({ id, herd, x, z, yaw: r() * Math.PI * 2, ax, az, coat: c, gallop, spurs, temper });
  }
}

// each cell's herd once worked out (the land is evaluated a few dozen times for one), for the seed it was for
const cells = new Map<string, WildHorse[]>();
let cellsSeed = NaN;

/** The wild horses of every herd whose cell comes within `range` of (x, z), into out (cleared). */
export function herdsNear(x: number, z: number, seed: number, range: number, out: WildHorse[]): WildHorse[] {
  out.length = 0;
  if (seed !== cellsSeed || cells.size > 4096) { cells.clear(); cellsSeed = seed; }
  const C = HORSE.herdCell;
  const x0 = Math.floor((x - range) / C), x1 = Math.floor((x + range) / C);
  const z0 = Math.floor((z - range) / C), z1 = Math.floor((z + range) / C);
  for (let cz = z0; cz <= z1; cz++) for (let cx = x0; cx <= x1; cx++) {
    const key = `${cx},${cz}`;
    let herd = cells.get(key);
    if (!herd) { herd = []; herdOf(cx, cz, seed, herd); cells.set(key, herd); }
    for (const w of herd) out.push(w);
  }
  return out;
}
