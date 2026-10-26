/**
 * The sky realm's public face: what lies at a point, what to draw near the
 * camera, and what the hero stands on or swims in up there. The realm is the
 * cloud world's (world.ts): on the land's seeds every answer is "nothing".
 * ---------------------------------------------------------------------------
 * A point needs at most two lookups: the archipelago of its archCell square
 * and the lone isle of its isleCell square (each never leaves its cell);
 * both are cached, so the hero's questions every tick allocate nothing.
 * The floor under a point is an island's top (a frozen lake's ice counts as
 * floor), an arch's deck or a puff's top (puffs.ts); the water is a lake's or
 * a stream's surface. The land has floors up here too: its model isles' (skymodels.ts).
 */

import { SKY_REALM as S } from '../constants';
import { archGround, type Arch } from './arches';
import { archipelagoInCell, archipelagoStep, cellBuilt, isleInCell, type Archipelago } from './archipelago';
import { islandRho, lakeDist, nearWater, outline, outlineMax, topAt, waterAt, type SkyIsland } from './island';
import { puffTop, puffUnder, type Puff } from './puffs';
import { worldOf } from '../world';
import { modelIsleUnder, modelTop, type ModelIsle } from '../skymodels';

/** Something to stand on up here. */
type SkyFloor = SkyIsland | Arch | Puff | ModelIsle;

/** The island whose outline holds world (x, z), or null. */
export function islandAt(x: number, z: number, seed: number): SkyIsland | null {
  const arch = archipelagoInCell(Math.floor(x / S.archCell), Math.floor(z / S.archCell), seed);
  if (arch && Math.hypot(x - arch.cx, z - arch.cz) <= arch.reach) {
    for (const isl of arch.islands) {
      if (Math.abs(x - isl.cx) > isl.R * 1.2 || Math.abs(z - isl.cz) > isl.R * 1.2) continue;
      if (islandRho(isl, x, z) <= 1) return isl;
    }
  }
  const isle = isleInCell(Math.floor(x / S.isleCell), Math.floor(z / S.isleCell), seed);
  return isle && islandRho(isle, x, z) <= 1 ? isle : null;
}

/** The floor's height at world (x, z): an island's top (a frozen lake as ice), an arch's deck, a puff's top or a model isle's floor; NaN off it. */
export function floorHeight(f: SkyFloor, x: number, z: number): number {
  if ('spring' in f) return puffTop(f, x, z);
  if ('qy' in f) return modelTop(f, x, z, f.qy); // (the layer the query found, or the next one down)
  if ('ax' in f) return archGround(f, x, z); // an arch's or a bridge's deck
  if (islandRho(f, x, z) > 1) return NaN;
  const t = topAt(f, x, z);
  const L = f.lake;
  return L && L.frozen && lakeDist(L, x - f.cx, z - f.cz) < 1 ? Math.max(t, L.level) : t;
}

/**
 * What a point at height y stands over up here: the highest island top,
 * arch deck or puff (puffs.ts) under (x, z) that the point is above, or less than `reach` below
 * (a step onto it holds). Null over open sky.
 */
export function skyFloorUnder(x: number, y: number, z: number, seed: number, reach = 1.5): SkyFloor | null {
  let best: SkyFloor | null = null;
  let bestY = -Infinity;
  const isl = islandAt(x, z, seed);
  if (isl) {
    const t = floorHeight(isl, x, z);
    if (y >= t - reach) { best = isl; bestY = t; }
  }
  const arch = archipelagoInCell(Math.floor(x / S.archCell), Math.floor(z / S.archCell), seed);
  if (arch) {
    for (const A of arch.arches) {
      // (a built bridge is walked on its spans and steps: the Eyrie's colliders, hero/locomotion.ts)
      if (A.kind === 'bridge') continue;
      const t = archGround(A, x, z);
      if (t === t && y >= t - reach && t > bestY) { best = A; bestY = t; }
    }
  }
  // (puffs are the cloud world's: the land's every tick skips them)
  const p = worldOf(seed) === 'clouds' ? puffUnder(x, y, z, seed, reach) : null;
  if (p && puffTop(p, x, z) > bestY) best = p;
  return best ?? modelIsleUnder(x, y, z, seed, reach);
}

/** Height of the floor under (x, y, z) (see skyFloorUnder), else NaN. */
export function skyGround(x: number, y: number, z: number, seed: number, reach = 1.5): number {
  const f = skyFloorUnder(x, y, z, seed, reach);
  return f ? floorHeight(f, x, z) : NaN;
}

/**
 * The surface of the water a point at height y is in or over up here (a lake
 * or a stream; frozen lakes are floor, not water), else NaN. The point must
 * be in the island's layer: no deeper than the bed under it, give or take.
 */
export function skyWater(x: number, y: number, z: number, seed: number): number {
  const isl = islandAt(x, z, seed);
  if (!isl) return NaN;
  const L = isl.lake;
  if (L && L.frozen && lakeDist(L, x - isl.cx, z - isl.cz) < 1.02) return NaN; // ice: floor, not water
  const w = waterAt(isl, x, z);
  return w === w && y >= topAt(isl, x, z) - 2 ? w : NaN;
}

/** A cell of the realm not built yet: an archipelago's or a lone isle's. */
export type PendingCell = { kind: 'arch' | 'isle'; ix: number; iz: number };

/**
 * The realm round (x, z): archipelagos and lone isles whose bounds come within `range` m, nearest
 * first. With `pending`, cells not built yet are left out and listed there instead (to be built
 * a few at a time: buildCell), nearest first.
 */
export function realmNear(x: number, z: number, range: number, seed: number, pending?: PendingCell[]): { archipelagos: Archipelago[]; isles: SkyIsland[] } {
  const archs: Array<{ a: Archipelago; d: number }> = [];
  const later: Array<{ c: PendingCell; d: number }> = [];
  // (an archipelago or isle never leaves its cell: a cell wholly out of range holds nothing in it)
  const out = (ix: number, iz: number, size: number) =>
    Math.hypot(Math.max(ix * size - x, 0, x - (ix + 1) * size), Math.max(iz * size - z, 0, z - (iz + 1) * size)) > range;
  const a0x = Math.floor((x - range - S.archReach) / S.archCell), a1x = Math.floor((x + range + S.archReach) / S.archCell);
  const a0z = Math.floor((z - range - S.archReach) / S.archCell), a1z = Math.floor((z + range + S.archReach) / S.archCell);
  for (let iz = a0z; iz <= a1z; iz++) {
    for (let ix = a0x; ix <= a1x; ix++) {
      if (out(ix, iz, S.archCell)) continue;
      if (pending && !cellBuilt('arch', ix, iz, seed)) {
        later.push({ c: { kind: 'arch', ix, iz }, d: Math.hypot((ix + 0.5) * S.archCell - x, (iz + 0.5) * S.archCell - z) });
        continue;
      }
      const a = archipelagoInCell(ix, iz, seed);
      if (!a) continue;
      const d = Math.max(0, Math.hypot(a.cx - x, a.cz - z) - a.reach);
      if (d <= range) archs.push({ a, d });
    }
  }
  const isles: Array<{ i: SkyIsland; d: number }> = [];
  const i0x = Math.floor((x - range) / S.isleCell), i1x = Math.floor((x + range) / S.isleCell);
  const i0z = Math.floor((z - range) / S.isleCell), i1z = Math.floor((z + range) / S.isleCell);
  for (let iz = i0z; iz <= i1z; iz++) {
    for (let ix = i0x; ix <= i1x; ix++) {
      if (out(ix, iz, S.isleCell)) continue;
      if (pending && !cellBuilt('isle', ix, iz, seed)) {
        later.push({ c: { kind: 'isle', ix, iz }, d: Math.hypot((ix + 0.5) * S.isleCell - x, (iz + 0.5) * S.isleCell - z) });
        continue;
      }
      const i = isleInCell(ix, iz, seed);
      if (!i) continue;
      const d = Math.max(0, Math.hypot(i.cx - x, i.cz - z) - i.R * outlineMax(i));
      if (d <= range) isles.push({ i, d });
    }
  }
  archs.sort((p, q) => p.d - q.d);
  isles.sort((p, q) => p.d - q.d);
  if (pending) {
    later.sort((p, q) => p.d - q.d);
    for (const p of later) pending.push(p.c);
  }
  return { archipelagos: archs.map((p) => p.a), isles: isles.map((p) => p.i) };
}

/**
 * Build a pending cell (realmNear's `pending`) for at most about `budgetMs` (at least a slice):
 * true once it is built. An archipelago is laid out a slice at a time (a few ms in all); a lone
 * isle at once (~1 ms), after the archipelago cell it lies in (it keeps clear of that Eyrie).
 */
export function buildCell(c: PendingCell, seed: number, budgetMs: number): boolean {
  if (c.kind === 'arch') return archipelagoStep(c.ix, c.iz, seed, budgetMs);
  const k = S.archCell / S.isleCell;
  if (!archipelagoStep(Math.floor(c.ix / k), Math.floor(c.iz / k), seed, budgetMs)) return false;
  isleInCell(c.ix, c.iz, seed);
  return true;
}

/**
 * Where to start up in the sky (the game's ?spawn=sky): on the hub of the
 * archipelago nearest (x, z), 40 m back from the footing of one of its
 * arches and facing along it (else half way out, facing the rim), clear of
 * its water. Null when no archipelago lies within 30 km.
 */
export function skySpawn(x: number, z: number, seed: number): { x: number; y: number; z: number; yaw: number } | null {
  const a = realmNear(x, z, 30000, seed).archipelagos[0];
  if (!a) return null;
  const hub = a.islands[0];
  const A = a.arches.find((r) => r.from === 0);
  let dx = 1, dz = 0, sx: number, sz: number;
  if (A) {
    dx = A.dirX; dz = A.dirZ;
    sx = A.ax - dx * 40; sz = A.az - dz * 40;
  } else {
    sx = hub.cx + 0.6 * hub.R * outline(hub, 0); sz = hub.cz;
  }
  // Off the water: step in toward the middle (then round it) until dry.
  for (let k = 0; k < 24 && nearWater(hub, sx, sz, 8); k++) {
    const th = Math.atan2(sz - hub.cz, sx - hub.cx) + 0.3;
    const r = Math.hypot(sx - hub.cx, sz - hub.cz) * 0.97;
    sx = hub.cx + Math.cos(th) * r; sz = hub.cz + Math.sin(th) * r;
  }
  return { x: sx, y: topAt(hub, sx, sz), z: sz, yaw: Math.atan2(dx, dz) };
}
