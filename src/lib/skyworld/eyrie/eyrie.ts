/**
 * Eyries: the archipelagos as the Aelari have settled them (SKY_CIVILIZATION.md
 * §7) — each one's structures (settle.ts), built once per cell and seed and
 * kept, as the archipelagos themselves are.
 */

import { archipelagoInCell, archipelagoStep, runSlices, type Archipelago } from '../archipelago';
import { SKY_REALM as S } from '../../constants';
import type { Structure } from '../structures';
import { settleEyrie } from './settle';
import { buildColliders, type ColliderSet } from '../colliders';

/** An Eyrie: its islands (the archipelago), what is built on them, and what of that the hero bumps into. */
type Eyrie = { arch: Archipelago; structures: Structure[]; colliders: ColliderSet };

// Built once per cell and seed (no allocation on a hit), the oldest dropped past a bound; one being
// built a slice at a time (eyrieStep) is kept with its generator until it finishes.
const MAX_EYRIES = 40;
type Building = { arch: Archipelago; it: Generator<void, Eyrie, void> };

/** An Eyrie built: its structures (settle.ts), then their colliders — yielding as it goes. */
function* build(arch: Archipelago, seed: number): Generator<void, Eyrie, void> {
  const structures = yield* settleEyrie(arch, seed);
  const colliders = yield* buildColliders(structures);
  return { arch, structures, colliders };
}
type Caches = { done: Map<number, Eyrie | null>; building: Map<number, Building> };
const bySeed = new Map<number, Caches>();
const cellKey = (ix: number, iz: number) => (ix + 0x100000) * 0x200000 + (iz + 0x100000);
function cachesFor(seed: number): Caches {
  let m = bySeed.get(seed);
  if (!m) {
    if (bySeed.size >= 4) bySeed.clear();
    m = { done: new Map(), building: new Map() };
    bySeed.set(seed, m);
  }
  return m;
}
function finish(c: Caches, key: number, e: Eyrie | null): Eyrie | null {
  c.building.delete(key);
  if (c.done.size >= MAX_EYRIES) c.done.delete(c.done.keys().next().value!);
  c.done.set(key, e);
  return e;
}

/**
 * Build cell (ix, iz)'s Eyrie (its islands laid out first) for at most about `budgetMs` (at least
 * one slice): true once it is built (eyrieInCell then has it).
 */
export function eyrieStep(ix: number, iz: number, seed: number, budgetMs: number): boolean {
  const c = cachesFor(seed), key = cellKey(ix, iz);
  if (c.done.has(key)) return true;
  let b = c.building.get(key);
  if (!b) {
    const t0 = performance.now();
    if (!archipelagoStep(ix, iz, seed, budgetMs)) return false;
    budgetMs -= performance.now() - t0;
    const arch = archipelagoInCell(ix, iz, seed);
    if (!arch) { finish(c, key, null); return true; }
    b = { arch, it: build(arch, seed) };
    c.building.set(key, b);
  }
  return runSlices(b.it, budgetMs, (e) => finish(c, key, e));
}

/** The Eyrie of archipelago cell (ix, iz), or null (none there): built now if it is not (tens of ms for a Crown). */
export function eyrieInCell(ix: number, iz: number, seed: number): Eyrie | null {
  const c = cachesFor(seed), key = cellKey(ix, iz);
  const hit = c.done.get(key);
  if (hit !== undefined) return hit;
  eyrieStep(ix, iz, seed, Infinity);
  return c.done.get(key) ?? null;
}

/** Cell (ix, iz)'s Eyrie if it has been built, else null — never builds it (the hero asks every tick). */
export function eyrieIfBuilt(ix: number, iz: number, seed: number): Eyrie | null {
  return bySeed.get(seed)?.done.get(cellKey(ix, iz)) ?? null;
}

/** Whether cell (ix, iz)'s Eyrie has been built. */
export function eyrieBuilt(ix: number, iz: number, seed: number): boolean {
  return cachesFor(seed).done.has(cellKey(ix, iz));
}

/** Cells whose Eyrie could reach within `range` of (x, z), nearest first. */
export function eyrieCellsNear(x: number, z: number, range: number): Array<[number, number]> {
  const out: Array<{ c: [number, number]; d: number }> = [];
  const r = range + S.archReach;
  for (let iz = Math.floor((z - r) / S.archCell); iz <= Math.floor((z + r) / S.archCell); iz++) {
    for (let ix = Math.floor((x - r) / S.archCell); ix <= Math.floor((x + r) / S.archCell); ix++) {
      const d = Math.hypot(Math.max(ix * S.archCell - x, 0, x - (ix + 1) * S.archCell), Math.max(iz * S.archCell - z, 0, z - (iz + 1) * S.archCell));
      if (d <= range) out.push({ c: [ix, iz], d });
    }
  }
  out.sort((a, b) => a.d - b.d);
  return out.map((o) => o.c);
}
