/**
 * Tree trunks for collision: the trees of the lattice cells round a point,
 * exactly as the GPU places, shapes and turns them (the placement's CPU twin,
 * and forestGen's shape and yaw from the same cell hash), each a circle round
 * its trunk (models.ts, trunkColliders) standing on the ground or on a sky
 * island's top. Crowns and bushes are left out: the hero passes through them.
 * A cell's tree never changes, so cells are cached (no allocation on a hit).
 */

import { ruinSiteAt } from '../ruins';
import { islandAt, islandRho, islandTop } from '../skyIslands';
import { elevation, hash3 } from '../terrainMath';
import { trunkColliders, type TrunkCollider } from './models';
import { plantInCell, type PlaceWorld } from './placement';
import { CELL, SPECIES, VARIANTS } from './species';

/** A trunk: centre (m) at hip height, the height it stands on, its radius, and how high it reaches above that. */
export type Trunk = { x: number; z: number; y: number; r: number; top: number };

let colliders: Array<TrunkCollider | null> | null = null;

// The world the GPU places trees in: ruins stay bare, island tops are planted.
let worldSeed = NaN;
let world: PlaceWorld = {};
function worldFor(seed: number): PlaceWorld {
  if (seed !== worldSeed) {
    worldSeed = seed;
    world = {
      onRuin: (x, z) => { const r = ruinSiteAt(x, z, seed); return !!r && Math.hypot(x - r.x, z - r.z) < r.radius; },
      islandTop: (x, z) => { const i = islandAt(x, z, seed); return i && islandRho(i, x, z) < 0.82 ? islandTop(i, x, z) : -Infinity; },
    };
  }
  return world;
}

function trunkAt(ix: number, iz: number, seed: number): Trunk | null {
  const p = plantInCell(ix, iz, seed, worldFor(seed));
  if (!p || p.species === 'bush') return null;
  colliders ??= trunkColliders();
  // Shape and facing as forestGen picks them from the cell's spare hash; scale as the GPU stores it (8 bits).
  const bits = hash3(ix, iz, seed + 7104);
  const c = colliders[SPECIES.indexOf(p.species) * VARIANTS + ((bits & 0xffff) % VARIANTS)];
  if (!c) return null;
  const yaw = ((bits >>> 16) & 255) * ((2 * Math.PI) / 256);
  const s = 0.55 + (1.05 * Math.min(255, Math.max(0, Math.round(((p.scale - 0.55) / 1.05) * 255)))) / 255;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const isl = p.feature === 'island' ? islandAt(p.x, p.z, seed) : null;
  return {
    x: p.x + (cy * c.x + sy * c.z) * s,
    z: p.z + (-sy * c.x + cy * c.z) * s,
    y: isl ? islandTop(isl, p.x, p.z) : elevation(p.x, p.z, seed, 0),
    r: c.r * s,
    top: c.top * s,
  };
}

// Direct-mapped cache of cells (a cell's trunk, or none).
type Slot = { ix: number; iz: number; seed: number; trunk: Trunk | null };
const slots: Array<Slot | undefined> = new Array(1024);
function trunkInCell(ix: number, iz: number, seed: number): Trunk | null {
  const k = (Math.imul(ix, 73856093) ^ Math.imul(iz, 19349663) ^ Math.imul(seed, 83492791)) & 1023;
  const hit = slots[k];
  if (hit && hit.ix === ix && hit.iz === iz && hit.seed === seed) return hit.trunk;
  const trunk = trunkAt(ix, iz, seed);
  slots[k] = { ix, iz, seed, trunk };
  return trunk;
}

/**
 * The trunks that can touch a body at (x, z): those of the 3 × 3 cells round
 * it (a trunk's circle reaches at most ~1.3 m past its cell, a body is 0.35 m round).
 * Fills and returns `out` (reuse it: nothing is allocated once cached).
 */
export function trunksNear(x: number, z: number, seed: number, out: Trunk[]): Trunk[] {
  const cx = Math.floor(x / CELL), cz = Math.floor(z / CELL);
  out.length = 0;
  for (let dz = -1; dz <= 1; dz++) {
    for (let dx = -1; dx <= 1; dx++) {
      const t = trunkInCell(cx + dx, cz + dz, seed);
      if (t) out.push(t);
    }
  }
  return out;
}
