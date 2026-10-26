/**
 * Puffs: small still clouds to stand on in the cloud world, in chains that
 * climb: a ladder rises beside each island but debris, just outside its rim,
 * from the cloud sea under it up to its top, and ends with a level bridge of
 * puffs that leads onto the top. Billows (below) are drawn clouds, not floor.
 * ---------------------------------------------------------------------------
 * Every step can be climbed: a plain puff's top is at most CLOUDS.puffRise
 * below the next's (a jump), a springy one's at most its lift (it throws the
 * hero that high: hero/locomotion.ts land), and the gap between two puffs'
 * edges is at most CLOUDS.puffGap. Ladders are mostly bounces with a rest every
 * fourth step. They are cached per cell, each a pure function of its island and
 * the seed, made only once the island's Eyrie or isle is built (the realm
 * builds them as they come near: realm.ts realmNear).
 */

import { CLOUDS as V, SKY_REALM as S } from '../constants';
import { hash01 } from '../noise';
import { worldOf } from '../world';
import { archipelagoInCell, cellBuilt, isleInCell } from './archipelago';
import { cloudSurface } from './clouds';
import { islandRho, nearWater, outline, topAt, type SkyIsland } from './island';

/** A puff: a squashed sphere, its centre and radii (m), and how high it throws the hero (m; 0: a plain puff). */
export type Puff = { x: number; y: number; z: number; rx: number; ry: number; rz: number; spring: number };

/** A ladder of puffs, in climbing order, the island it climbs, and the top it leads to (the island's at its rim). */
export type PuffChain = {
  puffs: Puff[];
  top: number;
  island: SkyIsland;
  // its puffs' extent (m), for a quick miss
  x0: number; z0: number; x1: number; z1: number;
};

const RY = 0.45;                  // half height: a first puff stands on the cloud a step up
const R_MIN = 1.5, R_MAX = 2.0;   // radii across (m)
const GAP = [0.8, 1.8] as const;  // gap between two puffs' edges (m), within CLOUDS.puffGap
const LADDER_R = 3.2;             // a ladder's radius round its axis (m)
const LADDER_OUT = 6;             // its puffs' clearance from the island's outline (m)
const SLACK = 1.5;                // a spring's next puff this far under its lift: the hero is over it at the top
const DROP = 5;                   // the most a chain's last puff stands over the top it leads onto (m: a hop down, a roll at most)

const reach = (p: Puff) => Math.max(p.rx, p.rz);
const top = (p: Puff) => p.y + p.ry;

/** The puff's top at (x, z), NaN off it. */
export function puffTop(p: Puff, x: number, z: number): number {
  const u = (x - p.x) / p.rx, v = (z - p.z) / p.rz, q = u * u + v * v;
  return q <= 1 ? p.y + p.ry * Math.sqrt(1 - q) : NaN;
}

type Rnd = () => number;
function stream(a: number, b: number, seed: number, salt: number): Rnd {
  let k = 0;
  return () => hash01(a, b, seed + salt + 97 * k++);
}

/**
 * Is (x, z) inside any island? (Laid out now if its cell is not: a ladder is made once, and must not
 * depend on which cells happen to be built.)
 */
function inIsland(x: number, z: number, seed: number): boolean {
  const a = archipelagoInCell(Math.floor(x / S.archCell), Math.floor(z / S.archCell), seed);
  if (a && Math.hypot(x - a.cx, z - a.cz) <= a.reach) for (const isl of a.islands) if (islandRho(isl, x, z) <= 1) return true;
  const i = isleInCell(Math.floor(x / S.isleCell), Math.floor(z / S.isleCell), seed);
  return !!i && islandRho(i, x, z) <= 1;
}

/** Does a puff of radius rad at (x, z) touch any island? (its centre and eight points round its edge) */
function overIsland(x: number, z: number, rad: number, seed: number): boolean {
  if (inIsland(x, z, seed)) return true;
  for (let a = 0; a < 8; a++) if (inIsland(x + rad * Math.cos(a * 0.785), z + rad * Math.sin(a * 0.785), seed)) return true;
  return false;
}

/**
 * Spiral up round (ax, az) at radius rho from angle phi, the first puff a step over the cloud, until the
 * tops reach `end`; a puff is springy where `springy(k)`. Null if a puff would sink into the cloud or touch an island.
 */
function spiral(ax: number, az: number, rho: number, phi: number, end: number, springy: (k: number) => boolean, r: Rnd, seed: number): Puff[] | null {
  const out: Puff[] = [];
  let t = 0;
  for (let k = 0; k < 2000; k++) {
    const rx = R_MIN + (R_MAX - R_MIN) * r(), rz = R_MIN + (R_MAX - R_MIN) * r();
    if (k > 0) {
      const chord = reach(out[k - 1]) + Math.max(rx, rz) + GAP[0] + (GAP[1] - GAP[0]) * r();
      phi += 2 * Math.asin(Math.min(1, chord / (2 * rho)));
    }
    const x = ax + rho * Math.sin(phi), z = az + rho * Math.cos(phi);
    const g = cloudSurface(x, z, seed, 0);
    if (k === 0) t = g + 2 * RY + 0.05;
    else if (g > t - 2 * RY) return null;
    if (overIsland(x, z, Math.max(rx, rz), seed)) return null;
    const last = t >= end - 0.01;
    const spring = !last && springy(k) ? V.springLift[0] + (V.springLift[1] - V.springLift[0]) * r() : 0;
    out.push({ x, y: t - RY, z, rx, ry: RY, rz, spring });
    if (last) return out;
    t = Math.min(t + (spring > 0 ? spring - SLACK : V.puffRise * (0.6 + 0.4 * r())), end);
  }
  return null;
}

/**
 * Level puffs from the chain's last toward (tx, tz) until ground (ground(x, z): its height, NaN off it)
 * is a step up (or a hop down) from the last one's edge, with more ground past it (not a wall). False if none is reached.
 */
function bridge(out: Puff[], tx: number, tz: number, ground: (x: number, z: number) => number, r: Rnd, seed: number): boolean {
  for (let n = 0; n < 60; n++) {
    const p = out[out.length - 1], h = top(p);
    const d = Math.hypot(tx - p.x, tz - p.z), ux = (tx - p.x) / d, uz = (tz - p.z) / d;
    for (let s = reach(p); s <= reach(p) + V.puffGap; s += 0.25) {
      const x = p.x + ux * s, z = p.z + uz * s, g = ground(x, z);
      const g2 = ground(x + 2 * ux, z + 2 * uz); // (and ground on past it, not a wall)
      if (g === g && g - h <= V.puffRise && g - h >= -DROP && g2 === g2 && g2 - g < 2) return true;
    }
    const rx = R_MIN + (R_MAX - R_MIN) * r(), rz = R_MIN + (R_MAX - R_MIN) * r();
    const step = reach(p) + Math.max(rx, rz) + GAP[0] + (GAP[1] - GAP[0]) * r();
    const x = p.x + ux * step, z = p.z + uz * step;
    if (cloudSurface(x, z, seed, 0) > h - 2 * RY || ground(x, z) > h - 2 * RY || overIsland(x, z, Math.max(rx, rz), seed)) return false;
    out.push({ x, y: h - RY, z, rx, ry: RY, rz, spring: 0 });
  }
  return false;
}

function chain(puffs: Puff[], t: number, island: SkyIsland): PuffChain {
  let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
  for (const p of puffs) {
    x0 = Math.min(x0, p.x - reach(p)); x1 = Math.max(x1, p.x + reach(p));
    z0 = Math.min(z0, p.z - reach(p)); z1 = Math.max(z1, p.z + reach(p));
  }
  return { puffs, top: t, island, x0, z0, x1, z1 };
}

/** The ladder up island isl (salted by its place in its cell), or null where none fits. */
function ladder(isl: SkyIsland, salt: number, seed: number): PuffChain | null {
  const r = stream(Math.floor(isl.cx), Math.floor(isl.cz), seed, 8400 + salt);
  const th0 = r() * 2 * Math.PI;
  const ground = (x: number, z: number) => (islandRho(isl, x, z) <= 1 ? topAt(isl, x, z) : NaN);
  for (let k = 0; k < 16; k++) {
    // a bearing off the island's water, the ladder's axis out from its rim there
    const th = th0 + (k * Math.PI) / 8, rim = isl.R * outline(isl, th);
    const ex = isl.cx + 0.9 * rim * Math.cos(th), ez = isl.cz + 0.9 * rim * Math.sin(th);
    if (nearWater(isl, ex, ez, 15)) continue;
    const ad = rim + LADDER_OUT + LADDER_R + R_MAX;
    const ax = isl.cx + ad * Math.cos(th), az = isl.cz + ad * Math.sin(th);
    let clear = true;
    for (let a = 0; a < 12 && clear; a++) {
      const w = (a / 12) * 2 * Math.PI, o = LADDER_R + R_MAX + 1;
      clear = !inIsland(ax + o * Math.sin(w), az + o * Math.cos(w), seed);
    }
    if (!clear) continue;
    // (level with the top's highest near the rim: an Eyrie's raised rim or terrace is stepped down from)
    let end = -Infinity;
    for (let f = 0.95; f <= 1.0; f += 0.01) end = Math.max(end, topAt(isl, isl.cx + f * rim * Math.cos(th), isl.cz + f * rim * Math.sin(th)) - 0.3);
    const puffs = spiral(ax, az, LADDER_R, r() * 2 * Math.PI, end, (j) => j % 4 !== 3, r, seed);
    if (puffs && bridge(puffs, isl.cx, isl.cz, ground, r, seed)) return chain(puffs, end + 0.3, isl);
  }
  return null;
}

// Ladders per cell: Eyries ('a'), lone isles ('i'); one seed's at a time.
const cells = new Map<number, readonly PuffChain[]>();
let cellsSeed = NaN;
function cellChains(kind: 'a' | 'i', ix: number, iz: number, seed: number): readonly PuffChain[] {
  if (seed !== cellsSeed) { cells.clear(); cellsSeed = seed; }
  // (a number, not a string: the hero asks every tick)
  const key = ((ix & 0xffff) * 65536 + (iz & 0xffff)) * 2 + (kind === 'a' ? 0 : 1);
  const hit = cells.get(key);
  if (hit) return hit;
  // (an Eyrie or isle not built yet: no ladders for now, and nothing kept)
  if (!cellBuilt(kind === 'a' ? 'arch' : 'isle', ix, iz, seed)) return [];
  const out: PuffChain[] = [];
  const islands = kind === 'a' ? archipelagoInCell(ix, iz, seed)?.islands ?? [] : [isleInCell(ix, iz, seed)].filter((i) => !!i);
  islands.forEach((isl, k) => {
    if (isl.kind === 'debris') return;
    const c = ladder(isl, k, seed);
    if (c) out.push(c);
  });
  if (cells.size > 4096) cells.clear();
  cells.set(key, out);
  return out;
}

/** Visit the chains whose cells may reach within `range` of (x, z). */
function eachChain(x: number, z: number, range: number, seed: number, fn: (c: PuffChain) => void): void {
  if (worldOf(seed) !== 'clouds') return;
  const visit = (kind: 'a' | 'i', size: number, pad: number) => {
    const c0x = Math.floor((x - range - pad) / size), c1x = Math.floor((x + range + pad) / size);
    const c0z = Math.floor((z - range - pad) / size), c1z = Math.floor((z + range + pad) / size);
    for (let iz = c0z; iz <= c1z; iz++) for (let ix = c0x; ix <= c1x; ix++) for (const c of cellChains(kind, ix, iz, seed)) fn(c);
  };
  visit('a', S.archCell, 60);
  visit('i', S.isleCell, 60);
}

/** The chains with a puff within `range` of (x, z) (the cloud world only), into `out`. */
export function puffChainsNear(x: number, z: number, range: number, seed: number, out: PuffChain[]): PuffChain[] {
  out.length = 0;
  eachChain(x, z, range, seed, (c) => {
    if (x + range >= c.x0 && x - range <= c.x1 && z + range >= c.z0 && z - range <= c.z1) out.push(c);
  });
  return out;
}

/** The highest puff whose top under (x, z) a point at height y is above, or less than `reach` below; null if none. */
export function puffUnder(x: number, y: number, z: number, seed: number, reach = 1.5): Puff | null {
  let best: Puff | null = null, bestY = -Infinity;
  eachChain(x, z, 0, seed, (c) => {
    if (x < c.x0 || x > c.x1 || z < c.z0 || z > c.z1) return;
    for (const p of c.puffs) {
      if (Math.abs(x - p.x) > p.rx || Math.abs(z - p.z) > p.rz) continue;
      const t = puffTop(p, x, z);
      if (t === t && y >= t - reach && t > bestY) { best = p; bestY = t; }
    }
  });
  return best;
}

// ---- billows ------------------------------------------------------------------
//
// Drawn clouds, never floor (gpu/puffs.wgsl.ts draws them with the puffs): round lobes heaped on the cloud
// sea, half sunk in it (cauliflower heads on its rolling tops), and banks of them hanging round each island's
// underside, as in the reference paintings, where the islands' roots trail down into cloud.

const SEA_CELL = 300;                       // m: one lobe per square
const seaCells = new Map<number, Puff>();
let seaSeed = NaN;
const banks = new WeakMap<SkyIsland, readonly Puff[]>();

/** The cloud sea's lobe in square (ix, iz) (SEA_CELL), cached. */
function seaBillow(ix: number, iz: number, seed: number): Puff {
  if (seed !== seaSeed) { seaCells.clear(); seaSeed = seed; }
  const key = (ix & 0xffff) * 65536 + (iz & 0xffff);
  let b = seaCells.get(key);
  if (!b) {
    const r = stream(ix, iz, seed, 8600);
    const x = (ix + 0.1 + 0.8 * r()) * SEA_CELL, z = (iz + 0.1 + 0.8 * r()) * SEA_CELL, R = 40 + 70 * r();
    b = { x, y: cloudSurface(x, z, seed, 0) - 0.45 * R, z, rx: R, ry: 0.8 * R, rz: R, spring: 0 };
    if (seaCells.size > 16384) seaCells.clear();
    seaCells.set(key, b);
  }
  return b;
}

/** The bank of cloud round island isl's underside: lobes under its outline, at the underside's depths, cached. */
export function islandBillows(isl: SkyIsland): readonly Puff[] {
  const hit = banks.get(isl);
  if (hit) return hit;
  const out: Puff[] = [];
  if (isl.kind !== 'debris') {
    let k = 0;
    const r = () => hash01(k++, isl.seed, 8700);
    const n = Math.min(24, Math.max(5, Math.round(isl.R / 14)));
    for (let j = 0; j < n; j++) {
      const a = ((j + 0.6 * r()) / n) * 2 * Math.PI, rho = isl.R * outline(isl, a) * (0.55 + 0.4 * r());
      const R = Math.min(70, Math.max(8, isl.R * (0.14 + 0.12 * r())));
      out.push({ x: isl.cx + Math.cos(a) * rho, y: isl.y - isl.depth * (0.35 + 0.5 * r()), z: isl.cz + Math.sin(a) * rho, rx: R, ry: 0.8 * R, rz: R, spring: 0 });
    }
  }
  banks.set(isl, out);
  return out;
}

/**
 * The billows within `range` of (x, z), the islands' banks within `bankRange` (the cloud world only; banks only
 * round islands already built), into `out`.
 */
export function billowsNear(x: number, z: number, range: number, seed: number, out: Puff[], bankRange = range): Puff[] {
  out.length = 0;
  if (worldOf(seed) !== 'clouds') return out;
  const c0x = Math.floor((x - range) / SEA_CELL), c1x = Math.floor((x + range) / SEA_CELL);
  const c0z = Math.floor((z - range) / SEA_CELL), c1z = Math.floor((z + range) / SEA_CELL);
  for (let iz = c0z; iz <= c1z; iz++) for (let ix = c0x; ix <= c1x; ix++) {
    const b = seaBillow(ix, iz, seed);
    if (Math.hypot(b.x - x, b.z - z) <= range) out.push(b);
  }
  const bank = (isl: SkyIsland | null) => {
    if (isl) for (const b of islandBillows(isl)) if (Math.hypot(b.x - x, b.z - z) <= bankRange) out.push(b);
  };
  const walk = (size: number, pad: number, fn: (ix: number, iz: number) => void) => {
    const r = bankRange + pad;
    for (let iz = Math.floor((z - r) / size); iz <= Math.floor((z + r) / size); iz++)
      for (let ix = Math.floor((x - r) / size); ix <= Math.floor((x + r) / size); ix++) fn(ix, iz);
  };
  walk(S.archCell, S.archReach, (ix, iz) => {
    if (cellBuilt('arch', ix, iz, seed)) for (const isl of archipelagoInCell(ix, iz, seed)?.islands ?? []) bank(isl);
  });
  walk(S.isleCell, 0, (ix, iz) => { if (cellBuilt('isle', ix, iz, seed)) bank(isleInCell(ix, iz, seed)); });
  return out;
}
