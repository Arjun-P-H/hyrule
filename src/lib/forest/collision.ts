/**
 * Tree trunks for collision: the trees of the lattice cells round a point,
 * exactly as the GPU places, shapes and turns them (the placements' CPU
 * twins, and forestGen's shape and yaw from the same cell hash), each a
 * circle round its trunk (models.ts, trunkColliders): the forest's ground
 * layer standing on the land, its sky layer on the islands' tops. Crowns and
 * bushes are left out: the hero passes through them. The boulders (rocks/collision.ts)
 * are circles of the same kind and come with them. A cell's trees never
 * change, so cells are cached (no allocation on a hit).
 */

import { rocksNear } from '../rocks/collision';
import { setTreesNear } from '../rocks/field';
import { ruinSiteAt } from '../ruins';
import { skyPlantInCell } from '../skyworld/flora';
import { elevation, hash3 } from '../terrainMath';
import { trunkColliders, type TrunkCollider } from './models';
import { plantInCell, SCALE, type PlaceWorld } from './placement';
import { CELL, SPECIES, VARIANTS, type Species } from './species';

/**
 * A trunk: centre (m) at hip height, the height it stands on, its radius, and how high it reaches above that;
 * a tree's also its climbing profile (models.ts TrunkCollider.prof, scaled and turned as placed: height over
 * `y`, centre x, z off (x, z), radius). The boulders and the cacti have none (they are not climbed).
 */
export type Trunk = { x: number; z: number; y: number; r: number; top: number; prof?: Float64Array };

/**
 * The trunk's cross-section `h` metres over the ground it stands on, into `out`: its centre x, z and radius,
 * between the profile's knots (below the first, the first's; above the top, the top's); without a profile, its circle.
 */
export function trunkSection(t: Trunk, h: number, out: Float64Array): Float64Array {
  const p = t.prof;
  if (!p) { out[0] = t.x; out[1] = t.z; out[2] = t.r; return out; }
  let i = 4;
  while (i < p.length - 4 && p[i] < h) i += 4;
  const f = Math.min(1, Math.max(0, (h - p[i - 4]) / Math.max(p[i] - p[i - 4], 1e-6)));
  out[0] = t.x + p[i - 3] + (p[i + 1] - p[i - 3]) * f;
  out[1] = t.z + p[i - 2] + (p[i + 2] - p[i - 2]) * f;
  out[2] = p[i - 1] + (p[i + 3] - p[i - 1]) * f;
  return out;
}

let colliders: Array<TrunkCollider | null> | null = null;

// The world the GPU places the ground layer in: ruins stay bare.
let worldSeed = NaN;
let world: PlaceWorld = {};
function worldFor(seed: number): PlaceWorld {
  if (seed !== worldSeed) {
    worldSeed = seed;
    world = { onRuin: (x, z) => { const r = ruinSiteAt(x, z, seed); return !!r && Math.hypot(x - r.x, z - r.z) < r.radius; } };
  }
  return world;
}

/** A plant's trunk, shaped and turned as forestGen does from the cell's spare hash (`salt`), its scale as the GPU stores it (8 bits). */
function trunkOf(species: Species, px: number, pz: number, y: number, scale: number, ix: number, iz: number, salt: number): Trunk | null {
  if (species === 'bush') return null;
  colliders ??= trunkColliders();
  const bits = hash3(ix, iz, salt);
  const c = colliders[SPECIES.indexOf(species) * VARIANTS + ((bits & 0xffff) % VARIANTS)];
  if (!c) return null;
  const yaw = ((bits >>> 16) & 255) * ((2 * Math.PI) / 256);
  const s = SCALE.lo + (SCALE.span * Math.min(255, Math.max(0, Math.round(((scale - SCALE.lo) / SCALE.span) * 255)))) / 255;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  // (a cactus is bumped into, not climbed: no profile)
  if (species === 'cactus') return { x: px + (cy * c.x + sy * c.z) * s, z: pz + (-sy * c.x + cy * c.z) * s, y, r: c.r * s, top: c.top * s };
  const prof = new Float64Array(c.prof.length);
  for (let k = 0; k < prof.length; k += 4) {
    prof[k] = c.prof[k] * s;
    prof[k + 1] = (cy * c.prof[k + 1] + sy * c.prof[k + 2]) * s;
    prof[k + 2] = (-sy * c.prof[k + 1] + cy * c.prof[k + 2]) * s;
    prof[k + 3] = c.prof[k + 3] * s;
  }
  return { x: px + (cy * c.x + sy * c.z) * s, z: pz + (-sy * c.x + cy * c.z) * s, y, r: c.r * s, top: c.top * s, prof };
}

/** The cell's trunks: on the land (the ground layer) and on an island's top (the sky layer). */
function trunksAt(ix: number, iz: number, seed: number): [Trunk | null, Trunk | null] {
  const p = plantInCell(ix, iz, seed, worldFor(seed));
  const q = skyPlantInCell(ix, iz, seed);
  return [
    p && trunkOf(p.species, p.x, p.z, elevation(p.x, p.z, seed, 0), p.scale, ix, iz, seed + 7104),
    q && trunkOf(q.species, q.x, q.z, q.y, q.scale, ix, iz, seed + 7304),
  ];
}

// Direct-mapped cache of cells (a cell's trunks, or none).
type Slot = { ix: number; iz: number; seed: number; trunks: [Trunk | null, Trunk | null] };
const slots: Array<Slot | undefined> = new Array(1024);
function trunksInCell(ix: number, iz: number, seed: number): [Trunk | null, Trunk | null] {
  const k = (Math.imul(ix, 73856093) ^ Math.imul(iz, 19349663) ^ Math.imul(seed, 83492791)) & 1023;
  const hit = slots[k];
  if (hit && hit.ix === ix && hit.iz === iz && hit.seed === seed) return hit.trunks;
  const trunks = trunksAt(ix, iz, seed);
  slots[k] = { ix, iz, seed, trunks };
  return trunks;
}

/**
 * The trunks that can touch a body at (x, z): those of the 3 × 3 cells round
 * it (a trunk's circle reaches at most ~1.3 m past its cell, a body is 0.35 m
 * round), on the land and up on the islands (the caller sorts them by height),
 * and (unless `rocks` is false) the boulders round it (rocks/collision.ts: circles of the same shape).
 * Fills and returns `out` (reuse it: nothing is allocated once cached).
 */
export function trunksNear(x: number, z: number, seed: number, out: Trunk[], rocks = true): Trunk[] {
  const cx = Math.floor(x / CELL), cz = Math.floor(z / CELL);
  out.length = 0;
  for (let dz = -1; dz <= 1; dz++) {
    for (let dx = -1; dx <= 1; dx++) {
      const pair = trunksInCell(cx + dx, cz + dz, seed);
      if (pair[0]) out.push(pair[0]);
      if (pair[1]) out.push(pair[1]);
    }
  }
  return rocks ? rocksNear(x, z, seed, out) : out;
}

// (rocks on the move glance off the trunks: rocks/field.ts asks for them through this)
setTreesNear((x, z, seed, out) => trunksNear(x, z, seed, out, false));
