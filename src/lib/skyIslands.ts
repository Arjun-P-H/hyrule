/**
 * Sky islands — CPU twin (fp64) of the WGSL functions in
 * src/lib/gpu/skyIslands.wgsl.ts: placement, outline, top and underside
 * surfaces, and the walkable top under a point (hero physics).
 *
 * An island lives in one SKY_ISLANDS.cell square (its centre inset from the
 * border by more than its largest radius, so a point only ever needs its own
 * cell's island). ρ is the distance from the centre over the outline radius
 * at that bearing: the top exists for ρ ≤ 1, where it meets the underside.
 */

import { SKY_ISLANDS as S } from './constants';
import { hash01, hash3, vnoise } from './terrainMath';

export type Island = {
  cx: number;
  cz: number;
  radius: number;
  y: number;       // base height of the top surface
  depth: number;   // underside cone depth (m)
  a1: number;      // outline phases
  a2: number;
  a3: number;
  seed: number;    // per-island noise seed (int32)
};

const TAU = Math.PI * 2;

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** The island of cell (ix, iz), or null. */
export function islandInCell(ix: number, iz: number, seed: number): Island | null {
  if (hash01(ix, iz, seed + 12001) >= S.probability) return null;
  const u = (k: number) => hash01(ix, iz, seed + 12001 + k);
  const radius = S.radiusMin + (S.radiusMax - S.radiusMin) * u(4);
  return {
    cx: (ix + S.inset + (1 - 2 * S.inset) * u(2)) * S.cell,
    cz: (iz + S.inset + (1 - 2 * S.inset) * u(3)) * S.cell,
    radius,
    y: S.altitudeMin + (S.altitudeMax - S.altitudeMin) * u(5),
    depth: (S.depthMin + (S.depthMax - S.depthMin) * u(6)) * radius,
    a1: u(7) * TAU,
    a2: u(8) * TAU,
    a3: u(9) * TAU,
    seed: hash3(ix, iz, seed + 12011) | 0,
  };
}

/** Outline radius factor at bearing θ (≈ 1 ± outline). */
export function outline(isl: Island, theta: number): number {
  return 1 + S.outline * (0.6 * Math.sin(3 * theta + isl.a1) + 0.4 * Math.sin(5 * theta + isl.a2) + 0.25 * Math.sin(8 * theta + isl.a3));
}

/** Normalised radius ρ of (x, z) in the island's outline. */
export function islandRho(isl: Island, x: number, z: number): number {
  const dx = x - isl.cx, dz = z - isl.cz;
  return Math.hypot(dx, dz) / (isl.radius * outline(isl, Math.atan2(dz, dx)));
}

/** Height of the top surface at (x, z) (meaningful for ρ ≤ 1). */
export function islandTop(isl: Island, x: number, z: number): number {
  const rho = islandRho(isl, x, z);
  const n = 0.65 * vnoise(x / 90, z / 90, isl.seed) + 0.35 * vnoise(x / 37, z / 37, isl.seed + 1);
  const relief = S.hill * (0.35 * (1 - rho * rho) + 0.65 * (0.5 + 0.5 * n));
  return isl.y + relief * (1 - smoothstep(0.78, 1, rho)) - S.rimDrop * smoothstep(0.85, 1, rho);
}

/** Height of the underside at (x, z) (meaningful for ρ ≤ 1). */
export function islandBottom(isl: Island, x: number, z: number): number {
  const rho = islandRho(isl, x, z);
  const wall = S.cliff * (1 - smoothstep(0.86, 1, rho));
  const cone = isl.depth * Math.pow(Math.max(0, 1 - rho / 0.86), 0.75) * (0.8 + 0.35 * (0.5 + 0.5 * vnoise(x / 60, z / 60, isl.seed + 3)));
  const rock = 10 * (0.5 + 0.5 * vnoise(x / 28, z / 28, isl.seed + 2)) * (1 - smoothstep(0.8, 1, rho));
  const dn = Math.max(0, vnoise(x / 26, z / 26, isl.seed + 4));
  const drip = 38 * dn * dn * dn * (1 - smoothstep(0.55, 0.9, rho));
  return isl.y - S.rimDrop - wall - cone - rock - drip;
}

/** The island whose outline contains (x, z), if any. */
export function islandAt(x: number, z: number, seed: number): Island | null {
  const isl = islandInCell(Math.floor(x / S.cell), Math.floor(z / S.cell), seed);
  return isl && islandRho(isl, x, z) <= 1 ? isl : null;
}

/**
 * The island whose top is the ground under a point at height y: the point is
 * above the top (or less than `reach` below it, so a step onto it holds).
 */
export function islandUnder(x: number, y: number, z: number, seed: number, reach = 1.5): Island | null {
  const isl = islandAt(x, z, seed);
  return isl && y >= islandTop(isl, x, z) - reach ? isl : null;
}

/** Top of the island under (x, y, z) (see islandUnder), else NaN. */
export function islandGround(x: number, y: number, z: number, seed: number, reach = 1.5): number {
  const isl = islandUnder(x, y, z, seed, reach);
  return isl ? islandTop(isl, x, z) : NaN;
}

/** Islands with a centre within `range` metres of (x, z), nearest first, at most `max`. */
export function islandsNear(x: number, z: number, range: number, seed: number, max: number): Island[] {
  const c0x = Math.floor((x - range) / S.cell), c1x = Math.floor((x + range) / S.cell);
  const c0z = Math.floor((z - range) / S.cell), c1z = Math.floor((z + range) / S.cell);
  const out: Array<{ isl: Island; d: number }> = [];
  for (let iz = c0z; iz <= c1z; iz++) {
    for (let ix = c0x; ix <= c1x; ix++) {
      const isl = islandInCell(ix, iz, seed);
      if (!isl) continue;
      const d = Math.hypot(isl.cx - x, isl.cz - z);
      if (d <= range) out.push({ isl, d });
    }
  }
  out.sort((a, b) => a.d - b.d);
  return out.slice(0, max).map((o) => o.isl);
}

export type Waterfall = {
  /** Where the water leaves the rim (world metres) and the bearing it pours toward. */
  x: number;
  y: number;
  z: number;
  dirX: number;
  dirZ: number;
  /** Height it lands at: the terrain below, or the water surface if that is higher. */
  bottom: number;
  width: number;
  seed: number;
};

/**
 * The waterfall pouring off an island (60 % of islands have one), at a hashed
 * bearing on the rim; it lands on the terrain straight below, found with the
 * caller's ground function (terrain E, or the water level where that is higher).
 */
export function islandWaterfall(isl: Island, ground: (x: number, z: number) => number): Waterfall | null {
  const h = hashU32(isl.seed ^ 0x5bd1e995);
  if ((h & 0xffff) / 65536 >= 0.6) return null;
  const theta = ((h >>> 16) / 65536) * TAU;
  const r = isl.radius * outline(isl, theta);
  const dirX = Math.cos(theta), dirZ = Math.sin(theta);
  const x = isl.cx + dirX * r, z = isl.cz + dirZ * r;
  return {
    x, z, dirX, dirZ,
    y: islandTop(isl, x, z),
    bottom: ground(x + dirX * 6, z + dirZ * 6),
    width: 12 + 12 * ((h >>> 8 & 0xff) / 255),
    seed: h | 0,
  };
}

function hashU32(v: number): number {
  let x = v | 0;
  x ^= x >>> 16; x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}
