/**
 * Archipelagos and lone isles: how the sky realm is laid out.
 * ---------------------------------------------------------------------------
 * One candidate archipelago per SKY_REALM.archCell square (8,192 m), kept
 * whole inside it (nothing reaches farther than archReach from its centre,
 * and the centre keeps that far from the cell's border), so a point only
 * ever needs its own cell's archipelago. Each is built once, from its cell
 * and the seed, and kept in a small cache:
 *   hub         the great island at its heart (0.65–1.25 km);
 *   satellites  3–6 round it across 170–460 m of open sky, most of them
 *               level enough with the hub to be joined by an arch;
 *   isles       small islands farther out, at their own heights (the glider
 *               takes you there);
 *   debris      rocks floating round the islands;
 *   arches      hub–satellite, and between neighbouring satellites when
 *               the span is short and level enough;
 * then each island's water (hydrology.ts), its arches' footings kept dry.
 * The archipelago's trait (biomes.ts) runs through all its islands, whose
 * biomes then follow their own height and size.
 *
 * Between archipelagos, one candidate lone isle per SKY_REALM.isleCell
 * square (2,048 m; none inside an archipelago's reach).
 *
 * Every island is high enough that its underside's tip clears
 * SKY_REALM.floor, above the highest peak, so the realm never touches the
 * land and its falls always have room to turn to mist.
 */

import { SKY_REALM as S, VALLEY } from '../constants';
import { elevation, hashU } from '../terrainMath';
import { type Arch } from './arches';
import { pickBiome, SKY_BIOME_INFO, SKY_TRAITS, type SkyTrait } from './biomes';
import { planWater } from './hydrology';
import { maxDrop, outline, outlineMax, topAt, type IslandKind, type SkyIsland } from './island';

export type Archipelago = {
  ix: number; iz: number;   // its cell
  cx: number; cz: number;   // centre (world m)
  reach: number;            // everything of it lies within this of the centre (m)
  trait: SkyTrait;
  islands: SkyIsland[];     // the hub first
  arches: Arch[];
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
const count = (r: readonly [number, number], f: number) => r[0] + Math.floor(f * (r[1] - r[0] + 1 - 1e-9));

/** An island of `kind` with outline radius R, its top's base at least `y` (higher if its underside needs it). */
function makeIsland(kind: IslandKind, cx: number, cz: number, R: number, y: number, trait: SkyTrait, rnd: () => number): SkyIsland {
  const cliff = Math.min(70, Math.max(2, 0.065 * R * (0.8 + 0.4 * rnd())));
  const depth = Math.min(600, Math.max(4, R * (0.5 + 0.35 * rnd())));
  const drop = maxDrop({ R, cliff, depth });
  const top = Math.min(S.topMax, Math.max(y, S.floor + drop));
  const biome = pickBiome(top, top - drop, R, trait, rnd());
  const info = SKY_BIOME_INFO[biome];
  const hill = info.relief.hill * Math.min(42, Math.max(0.6, 0.045 * R)) * (0.75 + 0.5 * rnd());
  return {
    kind, cx, cz, R, y: top,
    amp: lerp(S.outline, rnd()),
    a: [rnd() * TAU, rnd() * TAU, rnd() * TAU, rnd() * TAU],
    hill, wl: Math.min(260, Math.max(12, 0.35 * R)),
    crag: info.relief.crag, peak: info.relief.peak,
    cliff, depth,
    seed: (Math.floor(rnd() * 0x7fffffff) ^ 0x51ed27) | 0,
    biome,
    lake: null, stream: null, fall: null,
  };
}

/** Do two islands' outlines come within `gap` m of each other (by their largest radii)? */
const crowds = (a: SkyIsland, b: SkyIsland, gap: number) =>
  Math.hypot(a.cx - b.cx, a.cz - b.cz) < a.R * outlineMax(a) + b.R * outlineMax(b) + gap;

/** A point on an island's outline toward bearing th, pulled `inset` of the way in. */
function rimPoint(isl: SkyIsland, th: number, inset: number): [number, number] {
  const r = isl.R * outline(isl, th) * (1 - inset);
  return [isl.cx + Math.cos(th) * r, isl.cz + Math.sin(th) * r];
}

/** An arch from island a to island b, or null if its deck would be too steep or pass over another island. */
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
  // Its span must not cross any other island.
  for (let k = 0; k < islands.length; k++) {
    if (k === ia || k === ib) continue;
    const o = islands[k];
    for (let j = 1; j < 16; j++) {
      const t = j / 16;
      const x = ax + (bx - ax) * t - dirZ * arch.bow * Math.sin(Math.PI * t), z = az + (bz - az) * t + dirX * arch.bow * Math.sin(Math.PI * t);
      if (Math.hypot(x - o.cx, z - o.cz) < o.R * outlineMax(o) + arch.wEnd) return null;
    }
  }
  return arch;
}

function buildArchipelago(ix: number, iz: number, seed: number): Archipelago | null {
  // (salts chosen so seed 1337 has an archipelago 3.9 km ahead of the spawn's view)
  if (cellU(ix, iz, 12035, seed) >= S.archProbability) return null;
  const rnd = rng(cellHash(ix, iz, 12037, seed));
  const jit = 0.5 - S.archReach / S.archCell; // the centre keeps archReach from the cell's border
  const cx = (ix + 0.5 + (rnd() - 0.5) * 2 * jit) * S.archCell;
  const cz = (iz + 0.5 + (rnd() - 0.5) * 2 * jit) * S.archCell;
  const trait = SKY_TRAITS[Math.floor(rnd() * SKY_TRAITS.length)];
  const hubR = lerp(S.hubRadius, rnd());
  // A frost-touched archipelago floats higher, into the cold.
  const hub = makeIsland('hub', cx, cz, hubR, lerp(S.hubTop, rnd()) + (trait === 'frost' ? 350 : 0), trait, rnd);
  const islands: SkyIsland[] = [hub];
  const linked: number[] = [];
  const fits = (isl: SkyIsland) => Math.hypot(isl.cx - cx, isl.cz - cz) + isl.R * outlineMax(isl) <= S.archReach && islands.every((o) => !crowds(isl, o, 120));

  // Satellites round the hub, at roughly even bearings.
  const n = count(S.satellites, rnd());
  const phi0 = rnd() * TAU;
  for (let j = 0; j < n; j++) {
    const phi = phi0 + (j / n) * TAU + (rnd() - 0.5) * (TAU / n) * 0.5;
    const r = lerp(S.satRadius, rnd());
    const gap = lerp(S.satGap, rnd());
    const join = rnd() < 0.8;
    const dy = join ? (rnd() - 0.5) * 2 * 0.22 * gap : (rnd() - 0.5) * 2 * 380;
    const D = hubR * outline(hub, phi) + gap + r * 1.17;
    const sat = makeIsland('satellite', cx + Math.cos(phi) * D, cz + Math.sin(phi) * D, r, hub.y + dy, trait, rnd);
    if (!fits(sat)) continue;
    islands.push(sat);
    if (join) linked.push(islands.length - 1);
  }
  // Small isles farther out, each at its own height.
  const m = count(S.isles, rnd());
  for (let j = 0; j < m; j++) {
    const phi = rnd() * TAU;
    const r = lerp(S.isleRadius, rnd());
    const D = hubR * 1.2 + 350 + rnd() * (S.archReach - hubR * 1.2 - 350 - r * 1.2);
    const isle = makeIsland('isle', cx + Math.cos(phi) * D, cz + Math.sin(phi) * D, r, hub.y + (rnd() - 0.5) * 700, trait, rnd);
    if (fits(isle)) islands.push(isle);
  }

  // Arches: hub to each joined satellite, then between neighbouring satellites.
  const arches: Arch[] = [];
  for (const k of linked) {
    const a = makeArch(islands, 0, k, rnd);
    if (a) arches.push(a);
  }
  // Neighbouring joined satellites round the hub, each pair once.
  const pairs = linked.length > 2 ? linked.map((a, i) => [a, linked[(i + 1) % linked.length]]) : linked.length === 2 ? [linked] : [];
  for (const [a, b] of pairs) {
    if (rnd() > 0.5) continue;
    const arch = makeArch(islands, a, b, rnd);
    if (arch && arch.len < 620) arches.push(arch);
  }

  // Floating rocks round the islands (clear of the arches).
  const k = count(S.debris, rnd());
  const clearOfArches = (x: number, z: number, r: number) => arches.every((A) => {
    const px = x - A.ax, pz = z - A.az;
    const t = (px * A.dirX + pz * A.dirZ) / A.len;
    const s = Math.abs(-px * A.dirZ + pz * A.dirX - A.bow * Math.sin(Math.PI * Math.min(1, Math.max(0, t))));
    return t < -0.05 || t > 1.05 || s > A.wEnd + r + 30;
  });
  for (let j = 0; j < k && islands.length < 31; j++) {
    const host = islands[Math.floor(rnd() * islands.length)];
    if (host.kind === 'debris') continue;
    const phi = rnd() * TAU;
    const r = lerp(S.debrisRadius, rnd());
    const D = host.R * outline(host, phi) + 30 + r + rnd() * 90;
    const rock = makeIsland('debris', host.cx + Math.cos(phi) * D, host.cz + Math.sin(phi) * D, r, host.y - 20 - rnd() * 140, trait, rnd);
    if (Math.hypot(rock.cx - cx, rock.cz - cz) + r * 1.2 <= S.archReach && islands.every((o) => !crowds(rock, o, 25)) && clearOfArches(rock.cx, rock.cz, r)) {
      islands.push(rock);
    }
  }

  // Water, the arches' footings kept dry.
  const ground = (x: number, z: number) => Math.max(elevation(x, z, seed, 0), VALLEY.wLevel);
  islands.forEach((isl, i) => {
    const avoid: number[] = [];
    for (const A of arches) {
      if (A.from === i) avoid.push(Math.atan2(A.az - isl.cz, A.ax - isl.cx));
      if (A.to === i) avoid.push(Math.atan2(A.bz - isl.cz, A.bx - isl.cx));
    }
    planWater(isl, rng(cellHash(i, isl.seed, 12007, seed)), avoid, ground, islands);
  });
  return { ix, iz, cx, cz, reach: S.archReach, trait, islands, arches };
}

function buildIsle(ix: number, iz: number, seed: number): SkyIsland | null {
  if (cellU(ix, iz, 12011, seed) >= S.isleProbability) return null;
  const rnd = rng(cellHash(ix, iz, 12013, seed));
  const R = lerp(S.loneRadius, rnd());
  const inset = (R * 1.17 + 40) / S.isleCell;
  const cx = (ix + inset + (1 - 2 * inset) * rnd()) * S.isleCell;
  const cz = (iz + inset + (1 - 2 * inset) * rnd()) * S.isleCell;
  // None inside an archipelago's reach (the archipelago cell holding this one).
  const arch = archipelagoInCell(Math.floor(cx / S.archCell), Math.floor(cz / S.archCell), seed);
  if (arch && Math.hypot(cx - arch.cx, cz - arch.cz) < arch.reach + R * 1.2 + 300) return null;
  const trait = SKY_TRAITS[Math.floor(rnd() * SKY_TRAITS.length)];
  const isl = makeIsland('lone', cx, cz, R, 1650 + rnd() * 1000, trait, rnd);
  const ground = (x: number, z: number) => Math.max(elevation(x, z, seed, 0), VALLEY.wLevel);
  planWater(isl, rnd, [], ground, [isl]);
  return isl;
}

// Both are pure functions of (cell, seed); recent ones are kept in direct-mapped
// caches (no allocation on a hit: the hero asks every tick).
type ArchSlot = { ix: number; iz: number; seed: number; arch: Archipelago | null };
type IsleSlot = { ix: number; iz: number; seed: number; isle: SkyIsland | null };
const archSlots: Array<ArchSlot | undefined> = new Array(64);
const isleSlots: Array<IsleSlot | undefined> = new Array(256);
const slotOf = (ix: number, iz: number, seed: number, n: number) =>
  (Math.imul(ix, 73856093) ^ Math.imul(iz, 19349663) ^ Math.imul(seed, 83492791)) & (n - 1);

/** The archipelago of cell (ix, iz) (SKY_REALM.archCell squares), or null. */
export function archipelagoInCell(ix: number, iz: number, seed: number): Archipelago | null {
  const k = slotOf(ix, iz, seed, archSlots.length);
  const hit = archSlots[k];
  if (hit && hit.ix === ix && hit.iz === iz && hit.seed === seed) return hit.arch;
  const arch = buildArchipelago(ix, iz, seed);
  archSlots[k] = { ix, iz, seed, arch };
  return arch;
}

/** The lone isle of cell (ix, iz) (SKY_REALM.isleCell squares), or null. */
export function isleInCell(ix: number, iz: number, seed: number): SkyIsland | null {
  const k = slotOf(ix, iz, seed, isleSlots.length);
  const hit = isleSlots[k];
  if (hit && hit.ix === ix && hit.iz === iz && hit.seed === seed) return hit.isle;
  const isle = buildIsle(ix, iz, seed);
  isleSlots[k] = { ix, iz, seed, isle };
  return isle;
}
