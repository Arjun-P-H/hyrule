/**
 * Eyries and lone isles: how the sky realm is laid out.
 * ---------------------------------------------------------------------------
 * One candidate Eyrie per SKY_REALM.archCell square (8,192 m), kept whole
 * inside it (nothing reaches farther than archReach from its centre, and the
 * centre keeps that far from the cell's border), so a point only ever needs
 * its own cell's. Each is built once, from its cell and the seed, and kept in
 * a small cache. An Eyrie (SKY_CIVILIZATION.md §7) is a cluster of islands the
 * Aelari have bound together, of one of eleven classes (eyrie/classes.ts): the
 * class arranges its islands (eyrie/layouts.ts) — their sizes, heights, tops,
 * undersides, biomes, water — and joins them, by built bridges or natural
 * arches; floating rocks hang round them. Then each island's water
 * (hydrology.ts), its bridges' and arches' footings kept dry. The Eyrie's
 * buildings are raised on these islands afterwards (eyrie/eyrie.ts).
 *
 * Between Eyries, one candidate lone isle per SKY_REALM.isleCell square
 * (2,048 m; none inside an Eyrie's reach).
 *
 * Every island is high enough that its underside's tip clears
 * SKY_REALM.floor, above the highest peak, so the realm never touches the
 * land and its falls always have room to turn to mist.
 */

import { SKY_REALM as S } from '../constants';
import { hashU, surfaceAt } from '../terrainMath';

/** Scratch for the land under an island (its ground and its water). */
const gw: [number, number] = [0, 0];
import { type Arch } from './arches';
import { pickBiome, SKY_BIOME_INFO, SKY_TRAITS, type SkyTrait } from './biomes';
import { planWater } from './hydrology';
import { maxDrop, outline, outlineMax, topAt, type IslandKind, type Lobe, type SkyIsland } from './island';
import { CLASS_INFO, eyrieName, pickClass, type EyrieClass } from './eyrie/classes';
import { crowds, LAYOUTS, type IslandRole, type LayoutCtx, type Shape } from './eyrie/layouts';

export type Archipelago = {
  ix: number; iz: number;   // its cell
  cx: number; cz: number;   // centre (world m)
  reach: number;            // everything of it lies within this of the centre (m)
  trait: SkyTrait;
  islands: SkyIsland[];     // the main island first
  arches: Arch[];           // natural arches and built bridges (kind)
  /** The Eyrie: its class, its name, the palette it builds in; what each island is for, and how it was shaped. */
  cls: EyrieClass;
  name: string;
  palette: number;
  roles: IslandRole[];
  shapes: Shape[];
};

const TAU = Math.PI * 2;

/** mulberry32: a small seeded generator of [0, 1) values. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A well-mixed hash of a cell (hash3 alone sends some nearby cells to one value: (0, 1) and (−2, −1)). */
const cellHash = (ix: number, iz: number, salt: number, seed: number) =>
  hashU(hashU(hashU(Math.imul(ix, 0x632be5ab) + salt) ^ Math.imul(iz, 0x85157af5)) ^ Math.imul(seed, 0x9e3779b9));
const cellU = (ix: number, iz: number, salt: number, seed: number) => cellHash(ix, iz, salt, seed) / 4294967296;

const lerp = (r: readonly [number, number], f: number) => r[0] + (r[1] - r[0]) * f;

/**
 * An island of `kind` with outline radius R, its top's base at least `y` (higher if its underside
 * needs it), shaped as its Eyrie wants (`shape`): terraces, a bowl, a tilt, extra hanging spikes
 * (lobes: by default 0–2 on any island bigger than a rock), its underside, biome, relief.
 */
function makeIsland(kind: IslandKind, cx: number, cz: number, R: number, y: number, trait: SkyTrait, rnd: () => number, shape: Shape = {}): SkyIsland {
  const cliff = Math.min(70, Math.max(2, 0.065 * R * (0.8 + 0.4 * rnd())));
  const depth = Math.min(600, Math.max(4, R * (0.5 + 0.35 * rnd()) * (shape.depthK ?? 1)));
  const nl = shape.lobes ?? (kind === 'debris' || R < 40 ? 0 : Math.floor(rnd() * 3));
  const lobes: Lobe[] = [];
  for (let k = 0; k < Math.min(nl, 4); k++) {
    const a = rnd() * TAU, d = (0.14 + 0.42 * rnd()) * R;
    lobes.push({ x: Math.cos(a) * d, z: Math.sin(a) * d, r: (0.16 + 0.22 * rnd()) * R, d: depth * (0.55 + 0.8 * rnd()) });
  }
  const tiltX = shape.tiltX ?? 0, tiltZ = shape.tiltZ ?? 0;
  const drop = maxDrop({ R, cliff, depth, lobes, tiltX, tiltZ });
  const top = Math.min(S.topMax, Math.max(y, S.floor + drop));
  const biome = shape.biome ?? pickBiome(top, top - drop, R, trait, rnd());
  const info = SKY_BIOME_INFO[biome];
  const hill = info.relief.hill * Math.min(42, Math.max(0.6, 0.045 * R)) * (0.75 + 0.5 * rnd()) * (shape.hillK ?? 1);
  return {
    kind, cx, cz, R, y: top,
    amp: lerp(S.outline, rnd()),
    a: [rnd() * TAU, rnd() * TAU, rnd() * TAU, rnd() * TAU],
    hill, wl: Math.min(260, Math.max(12, 0.35 * R)),
    crag: shape.crag ?? info.relief.crag, peak: shape.peak ?? info.relief.peak,
    cliff, depth,
    seed: (Math.floor(rnd() * 0x7fffffff) ^ 0x51ed27) | 0,
    biome,
    terrace: shape.terrace ?? 0, bowl: shape.bowl ?? 0, tiltX, tiltZ,
    lobes, under: shape.under ?? (biome === 'frost' ? 'ice' : 'rock'), tint: shape.tint ?? 0,
    lake: null, stream: null, fall: null,
  };
}

/** A point on an island's outline toward bearing th, pulled `inset` of the way in. */
function rimPoint(isl: SkyIsland, th: number, inset: number): [number, number] {
  const r = isl.R * outline(isl, th) * (1 - inset);
  return [isl.cx + Math.cos(th) * r, isl.cz + Math.sin(th) * r];
}

/** Does the straight span from (ax, az) to (bx, bz) (bowed by `bow`, `w` wide) cross any island but a's and b's? */
function crossesIsland(islands: SkyIsland[], ia: number, ib: number, ax: number, az: number, bx: number, bz: number, bow: number, w: number): boolean {
  const len = Math.hypot(bx - ax, bz - az), dirX = (bx - ax) / len, dirZ = (bz - az) / len;
  for (let k = 0; k < islands.length; k++) {
    if (k === ia || k === ib) continue;
    const o = islands[k];
    for (let j = 1; j < 16; j++) {
      const t = j / 16;
      const x = ax + (bx - ax) * t - dirZ * bow * Math.sin(Math.PI * t), z = az + (bz - az) * t + dirX * bow * Math.sin(Math.PI * t);
      if (Math.hypot(x - o.cx, z - o.cz) < o.R * outlineMax(o) + w) return true;
    }
  }
  return false;
}

/** A natural arch from island a to island b, or null if its deck would be too steep or pass over another island. */
function makeArch(islands: SkyIsland[], ia: number, ib: number, rnd: () => number): Arch | null {
  const A = islands[ia], B = islands[ib];
  const th = Math.atan2(B.cz - A.cz, B.cx - A.cx);
  const [ax, az] = rimPoint(A, th, 0.035 + 25 / A.R);
  const [bx, bz] = rimPoint(B, th + Math.PI, 0.035 + 25 / B.R);
  const len = Math.hypot(bx - ax, bz - az);
  const ay = topAt(A, ax, az) - 0.3, by = topAt(B, bx, bz) - 0.3;
  if (Math.abs(by - ay) > S.linkSlope * len) return null;
  const dirX = (bx - ax) / len, dirZ = (bz - az) / len;
  const wMid = 10 + 8 * rnd();
  // (its rock thickens with its span: a natural arch some 30 times longer than it is thick in the middle)
  const arch: Arch = {
    kind: 'arch', from: ia, to: ib, ax, az, ay, bx, bz, by, dirX, dirZ, len,
    bow: (rnd() - 0.5) * 0.16 * len, rise: (0.015 + 0.02 * rnd()) * len,
    wMid, wEnd: wMid * (2 + 0.6 * rnd()), thMid: (0.028 + 0.017 * rnd()) * len, thEnd: (0.07 + 0.03 * rnd()) * len,
    seed: (Math.floor(rnd() * 0x7fffffff) ^ 0x2c1b3c6d) | 0,
  };
  return crossesIsland(islands, ia, ib, ax, az, bx, bz, arch.bow, arch.wEnd) ? null : arch;
}

/** The steepest a built bridge's deck may climb (a stair-bridge's flights take it). */
export const BRIDGE_SLOPE = 0.38;

/**
 * A vow-bridge (SKY_CIVILIZATION.md §6.7) from island a to island b: straight, level across, its
 * footings a little inside each rim; null if too steep or crossing another island. Its spans,
 * piers and parapets are raised as structures (eyrie/bridges.ts); the hero walks its deck.
 */
function makeBridge(islands: SkyIsland[], ia: number, ib: number, width: number, rnd: () => number): Arch | null {
  const A = islands[ia], B = islands[ib];
  const th = Math.atan2(B.cz - A.cz, B.cx - A.cx);
  const [ax, az] = rimPoint(A, th, Math.min(0.2, 0.02 + 16 / A.R));
  const [bx, bz] = rimPoint(B, th + Math.PI, Math.min(0.2, 0.02 + 16 / B.R));
  const len = Math.hypot(bx - ax, bz - az);
  const ay = topAt(A, ax, az) + 0.15, by = topAt(B, bx, bz) + 0.15;
  if (Math.abs(by - ay) > BRIDGE_SLOPE * len) return null;
  const dirX = (bx - ax) / len, dirZ = (bz - az) / len;
  if (crossesIsland(islands, ia, ib, ax, az, bx, bz, 0, width + 8)) return null;
  return {
    kind: 'bridge', from: ia, to: ib, ax, az, ay, bx, bz, by, dirX, dirZ, len,
    bow: 0, rise: 0, wMid: width, wEnd: width, thMid: 0, thEnd: 0,
    seed: (Math.floor(rnd() * 0x7fffffff) ^ 0x7a1b2c3d) | 0,
  };
}

/** Bridge widths by class (m): the Crown's broad vow-bridges, narrower ones elsewhere. */
const BRIDGE_WIDTH: Record<EyrieClass, readonly [number, number]> = {
  crown: [10, 14], mirror: [8, 11], halo: [5, 7], beacon: [6, 8], petal: [5, 7], bough: [6, 8], ember: [6, 9],
  rime: [7, 10], fallen: [8, 12], harbour: [8, 12], wells: [5, 7],
};

/** (salts chosen so seed 1337's Eyrie 3.9 km ahead of the spawn's view is a Crown) */
const CLASS_SALT = 35275;

/** Cell (ix, iz)'s Eyrie laid out: its islands (eyrie/layouts.ts), then each one's water — yielding between them. */
function* layArchipelago(ix: number, iz: number, seed: number): Generator<void, Archipelago | null, void> {
  if (cellU(ix, iz, 12035, seed) >= S.archProbability) return null;
  const rnd = rng(cellHash(ix, iz, 12037, seed));
  const jit = 0.5 - S.archReach / S.archCell; // the centre keeps archReach from the cell's border
  const cx = (ix + 0.5 + (rnd() - 0.5) * 2 * jit) * S.archCell;
  const cz = (iz + 0.5 + (rnd() - 0.5) * 2 * jit) * S.archCell;
  const cls = pickClass(cellU(ix, iz, CLASS_SALT, seed));
  const trait: SkyTrait = cls === 'rime' ? 'frost' : cls === 'petal' || cls === 'fallen' ? 'blossom' : cls === 'bough' || cls === 'wells' ? 'verdant' : SKY_TRAITS[Math.floor(rnd() * SKY_TRAITS.length)];
  const name = `${eyrieName(rnd)}, ${CLASS_INFO[cls].title}`;
  const islands: SkyIsland[] = [], roles: IslandRole[] = [], shapes: Shape[] = [], arches: Arch[] = [];
  const width = BRIDGE_WIDTH[cls];
  const ctx: LayoutCtx = {
    cls, cx, cz, rnd, islands, roles, shapes, arches,
    island: (kind, x, z, R, y, shape) => makeIsland(kind, x, z, R, y, trait, rnd, shape),
    add(isl, role, shape, gap = 60) {
      if (Math.hypot(isl.cx - cx, isl.cz - cz) + isl.R * outlineMax(isl) > S.archReach) return -1;
      if (islands.some((o) => crowds(isl, o, gap))) return -1;
      islands.push(isl); roles.push(role); shapes.push(shape);
      return islands.length - 1;
    },
    join(a, b, maxLen = 900) {
      const kind = CLASS_INFO[cls].bridges;
      if (kind === 'none' || a < 0 || b < 0 || a === b) return null;
      const A = kind === 'built' ? makeBridge(islands, a, b, lerp(width, rnd()), rnd) : makeArch(islands, a, b, rnd);
      if (!A || A.len > maxLen) return null;
      arches.push(A);
      return A;
    },
    debris(n, r) {
      const clearOfArches = (x: number, z: number, rr: number) => arches.every((A) => {
        const px = x - A.ax, pz = z - A.az;
        const t = (px * A.dirX + pz * A.dirZ) / A.len;
        const s = Math.abs(-px * A.dirZ + pz * A.dirX - A.bow * Math.sin(Math.PI * Math.min(1, Math.max(0, t))));
        return t < -0.05 || t > 1.05 || s > A.wEnd + rr + 30;
      });
      for (let j = 0; j < n && islands.length < 48; j++) {
        const host = islands[Math.floor(rnd() * islands.length)];
        if (host.kind === 'debris') continue;
        const phi = rnd() * TAU;
        const R = lerp(r, rnd());
        const D = host.R * outline(host, phi) + 30 + R + rnd() * 90;
        const rock = makeIsland('debris', host.cx + Math.cos(phi) * D, host.cz + Math.sin(phi) * D, R, host.y - 20 - rnd() * 140 + (cls === 'halo' ? 160 : 0), trait, rnd);
        if (Math.hypot(rock.cx - cx, rock.cz - cz) + R * 1.2 <= S.archReach && islands.every((o) => !crowds(rock, o, 25)) && clearOfArches(rock.cx, rock.cz, R)) {
          islands.push(rock); roles.push('debris'); shapes.push({});
        }
      }
    },
  };
  LAYOUTS[cls](ctx);
  if (!islands.length) return null;

  // Water, the bridges' and arches' footings kept dry (an island's plan is a millisecond or so).
  const ground = (x: number, z: number) => { surfaceAt(x, z, seed, gw); return Math.max(gw[0], gw[1]); };
  for (let i = 0; i < islands.length; i++) {
    yield;
    const isl = islands[i];
    const avoid: number[] = [];
    for (const A of arches) {
      if (A.from === i) avoid.push(Math.atan2(A.az - isl.cz, A.ax - isl.cx));
      if (A.to === i) avoid.push(Math.atan2(A.bz - isl.cz, A.bx - isl.cx));
    }
    planWater(isl, rng(cellHash(i, isl.seed, 12007, seed)), avoid, ground, islands, shapes[i].lake ?? 'auto');
  }
  return { ix, iz, cx, cz, reach: S.archReach, trait, islands, arches, cls, name, palette: CLASS_INFO[cls].palette, roles, shapes };
}

function buildIsle(ix: number, iz: number, seed: number): SkyIsland | null {
  if (cellU(ix, iz, 12011, seed) >= S.isleProbability) return null;
  const rnd = rng(cellHash(ix, iz, 12013, seed));
  const R = lerp(S.loneRadius, rnd());
  const inset = (R * 1.17 + 40) / S.isleCell;
  const cx = (ix + inset + (1 - 2 * inset) * rnd()) * S.isleCell;
  const cz = (iz + inset + (1 - 2 * inset) * rnd()) * S.isleCell;
  // None inside an Eyrie's reach (the Eyrie cell holding this one).
  const arch = archipelagoInCell(Math.floor(cx / S.archCell), Math.floor(cz / S.archCell), seed);
  if (arch && Math.hypot(cx - arch.cx, cz - arch.cz) < arch.reach + R * 1.2 + 300) return null;
  const trait = SKY_TRAITS[Math.floor(rnd() * SKY_TRAITS.length)];
  const isl = makeIsland('lone', cx, cz, R, 1650 + rnd() * 1000, trait, rnd, { lobes: 1 + Math.floor(rnd() * 3) });
  const ground = (x: number, z: number) => { surfaceAt(x, z, seed, gw); return Math.max(gw[0], gw[1]); };
  planWater(isl, rnd, [], ground, [isl]);
  return isl;
}

// Both are pure functions of (cell, seed); once built they are kept, per seed, in maps keyed by
// cell (no allocation on a hit: the hero asks every tick, the realm round the camera every re-pick,
// out to 32 km: ~80 Eyrie and ~1,000 isle cells), the oldest dropped past a bound.
const MAX_ARCHS = 256, MAX_ISLES = 2048;
type CellCaches = {
  arch: Map<number, Archipelago | null>; isle: Map<number, SkyIsland | null>;
  /** Eyries being laid out a slice at a time (archipelagoStep). */
  laying: Map<number, Generator<void, Archipelago | null, void>>;
};
const bySeed = new Map<number, CellCaches>();
const cellKey = (ix: number, iz: number) => (ix + 0x100000) * 0x200000 + (iz + 0x100000);
function cachesFor(seed: number): CellCaches {
  let c = bySeed.get(seed);
  if (!c) {
    if (bySeed.size >= 4) bySeed.clear();
    c = { arch: new Map(), isle: new Map(), laying: new Map() };
    bySeed.set(seed, c);
  }
  return c;
}
function keep<T>(m: Map<number, T>, key: number, v: T, max: number): T {
  if (m.size >= max) m.delete(m.keys().next().value!);
  m.set(key, v);
  return v;
}

/** The Eyrie (archipelago) of cell (ix, iz) (SKY_REALM.archCell squares), or null (laid out now if it is not: a few ms). */
export function archipelagoInCell(ix: number, iz: number, seed: number): Archipelago | null {
  const hit = cachesFor(seed).arch.get(cellKey(ix, iz));
  if (hit !== undefined) return hit;
  archipelagoStep(ix, iz, seed, Infinity);
  return cachesFor(seed).arch.get(cellKey(ix, iz)) ?? null;
}

/**
 * Lay out cell (ix, iz)'s Eyrie for at most about `budgetMs` (at least one slice; it stops early
 * when the next slice would likely overrun): true once it is laid out (archipelagoInCell has it).
 */
export function archipelagoStep(ix: number, iz: number, seed: number, budgetMs: number): boolean {
  const c = cachesFor(seed), key = cellKey(ix, iz);
  if (c.arch.has(key)) return true;
  let it = c.laying.get(key);
  if (!it) {
    // (one the realm stopped asking for is dropped past a bound, to be started again if wanted)
    if (c.laying.size >= 16) c.laying.delete(c.laying.keys().next().value!);
    it = layArchipelago(ix, iz, seed);
    c.laying.set(key, it);
  }
  return runSlices(it, budgetMs, (a) => { c.laying.delete(key); keep(c.arch, key, a, MAX_ARCHS); });
}

/**
 * Run a generator's slices for at most about `budgetMs` (at least one; each next slice is guessed
 * to take as long as the last): true, with `done` called with its result, once it has finished.
 */
export function runSlices<T>(it: Generator<void, T, void>, budgetMs: number, done: (v: T) => void): boolean {
  const t0 = performance.now();
  let t = t0;
  for (;;) {
    const r = it.next();
    if (r.done) { done(r.value); return true; }
    const now = performance.now();
    if (2 * now - t - t0 > budgetMs) return false;
    t = now;
  }
}

/** The lone isle of cell (ix, iz) (SKY_REALM.isleCell squares), or null. */
export function isleInCell(ix: number, iz: number, seed: number): SkyIsland | null {
  const m = cachesFor(seed).isle, key = cellKey(ix, iz);
  const hit = m.get(key);
  return hit !== undefined ? hit : keep(m, key, buildIsle(ix, iz, seed), MAX_ISLES);
}

/** Whether cell (ix, iz)'s Eyrie (kind 'arch') or lone isle ('isle') has been built already. */
export function cellBuilt(kind: 'arch' | 'isle', ix: number, iz: number, seed: number): boolean {
  return cachesFor(seed)[kind].has(cellKey(ix, iz));
}

