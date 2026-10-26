/**
 * The rocks as things: what the hero stands on and is stopped by, lifts, throws and rolls.
 * ---------------------------------------------------------------------------
 * A lattice cell's rock (the tiers from ROCK_SOLID_FROM up: the stones a hero can lift, the boulders it can
 * roll, the landmarks' big ones) is made here as the GPU scatter makes it (gpu/rocks.wgsl.ts rockScatter: the
 * cell's hash, the biomes' rockiness, the drifts of stone, the slope, the water and the ruins, repeated in
 * double precision), with the same transform: a RockBody. Its solid is a column height field in world axes
 * (`shape`: the top of its finest mesh, rasterised once when something first comes near it), so the hero is
 * stopped where the rock is drawn, not by a circle, and stands on its top.
 *
 * The field also owns the rocks round the hero (`focusRocks`): in a box of cells round it, per tier, the rocks
 * are the ones made here, sent to the GPU as a list (`packRocks`, rocks.wgsl.ts rockList), and the scatter
 * leaves those cells alone. What is drawn there is exactly what is solid. A rock taken from its cell (lifted,
 * pushed) is `taken`: its cell stays empty, out of the box too (the scatter is told). Taken rocks are free
 * bodies: carried, flying, rolling, or lying where they came to rest (`loose`), until the oldest give way.
 *
 * One field a world (module state, as the trees' and the rocks' caches): it follows the seed it is asked about.
 */

import { BIOMES, GROUND_USE } from '../biomes';
import { CHUNK_SIZE, PLAYER } from '../constants';
import { smoothstep } from '../noise';
import { ruinSiteAt } from '../ruins';
import { elevation, hash3, hashU, landRegions, surfaceAt, vnoise } from '../terrainMath';
import { worldOf } from '../world';
import { ROCK_KINDS, ROCK_LODS, ROCK_VERTEX_BYTES, buildRockModels } from './models';
import { ROCK_EXTENT, ROCK_LIST_CAP, ROCK_SOLID_FROM, ROCK_TAKEN_CAP, ROCK_TIERS, ROCK_ZONE } from './tiers';

type RockState = 'ground' | 'loose' | 'carried' | 'flying' | 'rolling';

/** A rock's solid: its top, a height per node of a square grid in world axes (relative to the rock's origin). */
type Shape = { x0: number; z0: number; cell: number; n: number; m: number; h: Float32Array; top: number; base: number };

export type RockBody = {
  /** The cell's key (tier, ix, iz); a rock that never had a cell has none (NaN). */
  key: number;
  tier: number; ix: number; iz: number;
  kind: number; size: number;
  /** The scatter's random number (its colour's variation). */
  rnd: number;
  /** The mesh's origin (world). */
  x: number; y: number; z: number;
  /** Its turn (a quaternion). */
  qx: number; qy: number; qz: number; qw: number;
  /** The scale of its axes (× size). */
  ax: number; ay: number; az: number;
  /** On its cell: the origin's height over the land there (the GPU sets it on its own land). */
  lift: number;
  state: RockState;
  vx: number; vy: number; vz: number;
  /** Turning (rad/s, world axes). */
  wx: number; wy: number; wz: number;
  /** Its widest horizontal reach from the origin (m), the radius it rolls on, its height. */
  reach: number; roll: number; height: number;
  /** s it has been still (rolling to rest), s since it came free. */
  still: number; age: number;
  /** Foe slots it has struck since it was thrown or pushed (bits). */
  struck: number;
  shape: Shape | null;
  /** Its circle for the arrows and the foes (forest/collision.ts Trunk's shape), for the blocking tiers. */
  circle: RockCircle | null;
};

/** A rock's circle: as forest/collision.ts Trunk (centre, the height it stands on, radius, how high it reaches above that). */
export type RockCircle = { x: number; z: number; y: number; r: number; top: number };

/** How much of a rock's widest radius stops a body in its circle (the arrows', the foes' test). */
const BODY_SHARE = 0.8;
/** A rock lower than this (m) is no circle (it is walked over). */
const WALK_OVER = 0.45;
/** Free rocks kept (lying, rolling, flying, carried): past this the oldest lying farthest gives way. */
const FREE_CAP = 48;

const rocksOf = BIOMES.map((b) => GROUND_USE[b].rocks);
/** The most a cell's chance can be over the tier's own: the rockiest land (1), a full drift (1.7) and the steepest slope the scatter keeps (2.6). */
const MOST = 1 * 1.7 * 2.6;
const W = new Float64Array(BIOMES.length);
const SURF: [number, number] = [0, 0];

/** The scatter's per-cell random numbers (rocks.wgsl.ts rkHash): four values 0 … 1 from a cell and a salt. */
const hs = [0, 0, 0, 0];
function cellHash(ix: number, iz: number, salt: number): void {
  for (let k = 0; k < 4; k++) hs[k] = (hashU(hash3(ix, iz, salt + k)) >>> 8) / 16777216;
}

function newBody(): RockBody {
  return {
    key: NaN, tier: 0, ix: 0, iz: 0, kind: 0, size: 1, rnd: 0, x: 0, y: 0, z: 0, qx: 0, qy: 0, qz: 0, qw: 1, ax: 1, ay: 1, az: 1,
    lift: 0, state: 'ground', vx: 0, vy: 0, vz: 0, wx: 0, wy: 0, wz: 0, reach: 1, roll: 0.5, height: 1, still: 0, age: 0, struck: 0,
    shape: null, circle: null,
  };
}

/** The key of tier t's cell (ix, iz): exact in a double for |ix|, |iz| < 2^20. */
function cellKey(t: number, ix: number, iz: number): number {
  return (t * 2097152 + (ix + 1048576)) * 2097152 + (iz + 1048576);
}

/** Its measures from its kind, size and axes. */
function measure(b: RockBody): void {
  const e = ROCK_EXTENT[b.kind];
  b.reach = e.rxz * b.size * Math.max(b.ax, b.az);
  b.height = (e.top - e.bottom) * b.size * b.ay;
  // (rolling on its mean radius: a rough ball)
  b.roll = 0.5 * (e.rxz * b.size * 0.5 * (b.ax + b.az) + 0.5 * b.height);
}

/** Tier t's cell (ix, iz): its rock as the scatter places it, or none. */
function bodyIn(t: number, ix: number, iz: number, seed: number): RockBody | null {
  if (worldOf(seed) === 'clouds') return null; // (no rocks on the clouds)
  const T = ROCK_TIERS[t];
  cellHash(ix, iz, seed + 9001 + t * 131);
  const hx = hs[0], hy = hs[1], hz = hs[2], hw = hs[3];
  if (hz > T.chance * MOST) return null; // (most cells: no rock, and the chance's own terms are dear)
  const x = (ix + 0.1 + 0.8 * hx) * T.cell, z = (iz + 0.1 + 0.8 * hy) * T.cell;
  const ruin = ruinSiteAt(x, z, seed);
  if (ruin && Math.hypot(x - ruin.x, z - ruin.z) < ruin.radius) return null;
  cellHash(ix, iz, seed + 9501 + t * 131);
  const sx = hs[0], sy = hs[1], sz = hs[2], sw = hs[3];
  const size = T.size[0] * Math.pow(T.size[1] / T.size[0], sx * sx);
  const [land, water] = surfaceAt(x, z, seed, SURF);
  if (land < water + 0.1) return null;
  // where the land is rocky (the biomes', by the regions' weights), thickened by drifts of stone and on slopes
  const w = landRegions(x, z, seed, W);
  let rocky = 0;
  for (let k = 0; k < rocksOf.length; k++) rocky += w[k] * rocksOf[k];
  const fld = 0.5 + 0.5 * vnoise(x / 83, z / 83, seed + 9011);
  const drift = 0.15 + 1.55 * smoothstep(0.4, 0.8, fld);
  const e = Math.max(0.4, 0.4 * size);
  const gx = (elevation(x + e, z, seed, T.cell) - elevation(x - e, z, seed, T.cell)) / (2 * e);
  const gz = (elevation(x, z + e, seed, T.cell) - elevation(x, z - e, seed, T.cell)) / (2 * e);
  const slope = Math.hypot(gx, gz);
  const chance = T.chance * rocky * drift * (1 + 1.6 * smoothstep(0.25, 0.75, slope)) * (1 - smoothstep(1.1, 1.5, slope));
  if (hz > chance) return null;
  const b = newBody();
  b.key = cellKey(t, ix, iz);
  b.tier = t; b.ix = ix; b.iz = iz;
  b.kind = T.kinds[Math.min(Math.floor(sy * 6), 5)];
  b.size = size;
  b.rnd = hash3(ix, iz, seed + 9777) >>> 0;
  b.ax = 0.82 + 0.36 * hw; b.ay = 0.85 + 0.3 * sz; b.az = 0.82 + 0.36 * sy;
  // set a little into the ground
  const sink = size * b.ay * (0.14 + 0.16 * sz);
  b.lift = -sink - ROCK_EXTENT[b.kind].bottom * size * b.ay;
  b.x = x; b.y = land + b.lift; b.z = z;
  // tilted toward the slope (two thirds of the way), then turned about its own axis
  let nx = -gx, ny = 1, nz = -gz;
  let l = Math.hypot(nx, ny, nz);
  nx /= l; ny /= l; nz /= l;
  let tx = 0.66 * nx, ty = 0.34 + 0.66 * ny, tz = 0.66 * nz;
  l = Math.hypot(tx, ty, tz);
  tx /= l; ty /= l; tz /= l;
  // (the turn from up to the tilt: about up × tilt)
  let q1x = 0, q1z = 0, q1w = 1;
  const axl = Math.hypot(tz, tx);
  if (axl > 1e-4) {
    const cs = Math.sqrt(0.5 * (1 + ty)), sn = Math.sqrt(Math.max(0.5 * (1 - ty), 0));
    q1x = (tz / axl) * sn; q1z = (-tx / axl) * sn; q1w = cs;
  }
  const yaw = sw * Math.PI * 2;
  const s2 = Math.sin(0.5 * yaw), c2 = Math.cos(0.5 * yaw);
  // q1 · (0, s2, 0, c2) (rocks.wgsl.ts rkQuatMul)
  b.qx = c2 * q1x - q1z * s2;
  b.qy = q1w * s2;
  b.qz = c2 * q1z + q1x * s2;
  b.qw = q1w * c2;
  measure(b);
  const top = b.height - sink;
  if (t >= 2 && top >= WALK_OVER) {
    const ext = ROCK_EXTENT[b.kind];
    b.circle = { x, z, y: land - sink, r: ext.rxz * size * 0.5 * (b.ax + b.az) * BODY_SHARE, top };
  }
  return b;
}

// ---- the field's state --------------------------------------------------------------------------
let fieldSeed = NaN;
const cells = new Map<number, RockBody | null>();
const takenKeys = new Set<number>();
const taken: number[] = [];     // in the order they were taken (the GPU's list; the oldest gives way)
const free: RockBody[] = [];    // the rocks off their cells
const zone = { cx: NaN, cz: NaN, list: [] as RockBody[], rect: new Int32Array(ROCK_TIERS.length * 4) };

function follow(seed: number): void {
  if (seed === fieldSeed) return;
  fieldSeed = seed;
  cells.clear();
  takenKeys.clear();
  taken.length = 0;
  free.length = 0;
  zone.cx = NaN; zone.list.length = 0;
  zone.rect.fill(0);
  for (let t = 0; t < ROCK_TIERS.length; t++) { zone.rect[t * 4] = 1; zone.rect[t * 4 + 2] = 0; } // (empty: from > to)
}

function bodyInCell(t: number, ix: number, iz: number, seed: number): RockBody | null {
  const k = cellKey(t, ix, iz);
  const hit = cells.get(k);
  if (hit !== undefined) return hit;
  if (cells.size > 40000) cells.clear();
  const b = bodyIn(t, ix, iz, seed);
  cells.set(k, b);
  return b;
}

let rocksOn = true;
/** Whether rocks stand in the world at all (false: a view that draws none, the WebGL2 one: nothing invisible in the way). */
export function setRocksOn(on: boolean): void {
  rocksOn = on;
}

/** Forget every rock moved and every cell made (a new world; tests). */
export function resetRocks(): void {
  fieldSeed = NaN;
  follow(NaN);
}

/**
 * The rocks round (x, z) a body can touch: those of the 3 × 3 cells round it in every solid tier (a rock's reach is
 * within its cell's neighbours), and the free ones within `range`. Taken cells' rocks are gone; a carried one is left out.
 * Fills and returns `out`.
 */
export function rockBodiesNear(x: number, z: number, seed: number, range: number, out: RockBody[]): RockBody[] {
  follow(seed);
  out.length = 0;
  if (!rocksOn) return out;
  for (let t = ROCK_SOLID_FROM; t < ROCK_TIERS.length; t++) {
    const cell = ROCK_TIERS[t].cell;
    const cx = Math.floor(x / cell), cz = Math.floor(z / cell);
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        const b = bodyInCell(t, cx + dx, cz + dz, seed);
        if (b && !takenKeys.has(b.key)) out.push(b);
      }
    }
  }
  for (const b of free) {
    if (b.state === 'carried') continue;
    const dx = b.x - x, dz = b.z - z, r = range + b.reach;
    if (dx * dx + dz * dz < r * r) out.push(b);
  }
  return out;
}

const NEAR_C: RockBody[] = [];
/**
 * The boulders' circles round (x, z) (the tiers from ROCK_BLOCKING_FROM up, and the big free rocks lying still):
 * appends to `out` (reuse it: nothing is allocated once cached) and returns it.
 */
export function rockCirclesNear<T extends RockCircle>(x: number, z: number, seed: number, out: T[]): T[] {
  rockBodiesNear(x, z, seed, 16, NEAR_C);
  for (const b of NEAR_C) {
    if (b.state === 'ground') { if (b.circle) out.push(b.circle as T); continue; }
    if (b.state !== 'loose' || b.height < WALK_OVER || b.tier < 2) continue;
    if (!b.circle) b.circle = { x: b.x, z: b.z, y: b.y - b.roll, r: b.reach * BODY_SHARE, top: b.height };
    out.push(b.circle as T);
  }
  return out;
}

// ---- shapes ---------------------------------------------------------------------------------------
let unit: Float32Array[] | null = null;
/** Kind k's finest mesh, its vertices' positions (unit scale; triangles, three vertices each). */
function unitMesh(k: number): Float32Array {
  if (!unit) {
    const m = buildRockModels();
    const dv = new DataView(m.vertices);
    unit = ROCK_KINDS.map((_, kind) => {
      const { first, count } = m.meshes[kind * ROCK_LODS];
      const p = new Float32Array(count * 3);
      for (let v = 0; v < count; v++) {
        const o = (first + v) * ROCK_VERTEX_BYTES;
        p[v * 3] = dv.getFloat32(o, true); p[v * 3 + 1] = dv.getFloat32(o + 4, true); p[v * 3 + 2] = dv.getFloat32(o + 8, true);
      }
      return p;
    });
  }
  return unit[k];
}

let WV = new Float64Array(0);
/** Rasterise the rock's finest mesh, turned and scaled as drawn, into its column height field. */
function buildShape(b: RockBody): Shape {
  const v = unitMesh(b.kind);
  const n = v.length / 3;
  if (WV.length < v.length) WV = new Float64Array(v.length);
  const sx = b.ax * b.size, sy = b.ay * b.size, sz = b.az * b.size;
  const { qx, qy, qz, qw } = b;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity, base = Infinity, top = -Infinity;
  for (let i = 0; i < n; i++) {
    const px = v[i * 3] * sx, py = v[i * 3 + 1] * sy, pz = v[i * 3 + 2] * sz;
    // rotate by q (v + 2w(q×v) + 2 q×(q×v))
    const tx = 2 * (qy * pz - qz * py), ty = 2 * (qz * px - qx * pz), tz = 2 * (qx * py - qy * px);
    const rx = px + qw * tx + (qy * tz - qz * ty), ry = py + qw * ty + (qz * tx - qx * tz), rz = pz + qw * tz + (qx * ty - qy * tx);
    WV[i * 3] = rx; WV[i * 3 + 1] = ry; WV[i * 3 + 2] = rz;
    if (rx < x0) x0 = rx; if (rx > x1) x1 = rx; if (rz < z0) z0 = rz; if (rz > z1) z1 = rz;
    if (ry < base) base = ry; if (ry > top) top = ry;
  }
  const ext = Math.max(x1 - x0, z1 - z0);
  const cell = Math.min(0.4, Math.max(0.05, ext / 40));
  const gn = Math.ceil((x1 - x0) / cell) + 1, gm = Math.ceil((z1 - z0) / cell) + 1;
  const h = new Float32Array(gn * gm).fill(base);
  const inv = 1 / cell;
  for (let t = 0; t < n; t += 3) {
    const ax = WV[t * 3], ay = WV[t * 3 + 1], az = WV[t * 3 + 2];
    const bx = WV[t * 3 + 3], by = WV[t * 3 + 4], bz = WV[t * 3 + 5];
    const cx = WV[t * 3 + 6], cy = WV[t * 3 + 7], cz = WV[t * 3 + 8];
    // (the corners themselves, at their nearest node: a thin face still marks the field)
    for (const [px, py, pz] of [[ax, ay, az], [bx, by, bz], [cx, cy, cz]] as const) {
      const k = Math.round((px - x0) * inv) + Math.round((pz - z0) * inv) * gn;
      if (py > h[k]) h[k] = py;
    }
    const area = (bx - ax) * (cz - az) - (cx - ax) * (bz - az);
    if (Math.abs(area) < 1e-10) continue;
    const i0 = Math.max(0, Math.ceil((Math.min(ax, bx, cx) - x0) * inv)), i1 = Math.min(gn - 1, Math.floor((Math.max(ax, bx, cx) - x0) * inv));
    const j0 = Math.max(0, Math.ceil((Math.min(az, bz, cz) - z0) * inv)), j1 = Math.min(gm - 1, Math.floor((Math.max(az, bz, cz) - z0) * inv));
    for (let j = j0; j <= j1; j++) {
      const pz = z0 + j * cell;
      for (let i = i0; i <= i1; i++) {
        const px = x0 + i * cell;
        const w0 = ((bx - px) * (cz - pz) - (cx - px) * (bz - pz)) / area;
        const w1 = ((cx - px) * (az - pz) - (ax - px) * (cz - pz)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
        const y = w0 * ay + w1 * by + w2 * cy;
        const k = i + j * gn;
        if (y > h[k]) h[k] = y;
      }
    }
  }
  return { x0, z0, cell, n: gn, m: gm, h, top, base };
}

function shapeOf(b: RockBody): Shape {
  return (b.shape ??= buildShape(b));
}

/** The top of rock b at world (x, z) (bilinear in its field), or −∞ off it. */
export function rockTopAt(b: RockBody, x: number, z: number): number {
  const s = shapeOf(b);
  const u = (x - b.x - s.x0) / s.cell, v = (z - b.z - s.z0) / s.cell;
  if (u < 0 || v < 0 || u > s.n - 1 || v > s.m - 1) return -Infinity;
  const i = Math.min(s.n - 2, Math.floor(u)), j = Math.min(s.m - 2, Math.floor(v));
  const fu = u - i, fv = v - j;
  const k = i + j * s.n;
  const h = s.h;
  const y = (h[k] * (1 - fu) + h[k + 1] * fu) * (1 - fv) + (h[k + s.n] * (1 - fu) + h[k + s.n + 1] * fu) * fv;
  return y <= s.base + 1e-4 ? -Infinity : b.y + y;
}

// ---- bodies meeting rocks -------------------------------------------------------------------------
const NEAR: RockBody[] = [];
/** The slope (∇) of the rock top the last collideRocks stood the body on. */
export const rockFloorSlope = { gx: 0, gz: 0 };

type Body = { x: number; z: number; vx: number; vz: number };

/**
 * A body (radius `radius`, feet at `feetY`, `height` tall) among the rocks: pushed out of any it walks into (and
 * its velocity into the rock dropped, so it slides round it), and the highest rock top under it within a step
 * (`stepUp`), or −∞ (rockFloorSlope: that top's slope). A rock face steeper than `walkSlope` is a wall from 12 cm up,
 * so a body is not stepped up a boulder's side; a lower ledge or a gentle top is walked onto.
 */
export function collideRocks(s: Body, feetY: number, seed: number, radius: number, stepUp: number, height: number, walkSlope: number): number {
  let floor = -Infinity;
  rockFloorSlope.gx = 0; rockFloorSlope.gz = 0;
  rockBodiesNear(s.x, s.z, seed, radius + 0.3, NEAR);
  for (const b of NEAR) {
    let dx = s.x - b.x, dz = s.z - b.z;
    const reach = b.reach + radius + 0.1;
    if (dx * dx + dz * dz > reach * reach) continue;
    if (b.state === 'flying' || b.state === 'rolling') {
      // (a rock on the move: a ball)
      if (feetY > b.y + b.roll || feetY + height < b.y - b.roll) continue;
      const d = Math.hypot(dx, dz), min = radius + b.roll * 0.9;
      if (d >= min) continue;
      const nx = d > 1e-6 ? dx / d : 1, nz = d > 1e-6 ? dz / d : 0;
      s.x += nx * (min - d); s.z += nz * (min - d);
      const vn = s.vx * nx + s.vz * nz;
      if (vn < 0) { s.vx -= vn * nx; s.vz -= vn * nz; }
      continue;
    }
    const sh = shapeOf(b);
    if (b.y + sh.base > feetY + height) continue; // (all of it over the head)
    const thr = feetY + stepUp, low = feetY + 0.12;
    if (b.y + sh.top > low) {
      // the nearest node that is a wall at the feet's height, within the body's radius
      const ox = b.x + sh.x0, oz = b.z + sh.z0, c = sh.cell, h = sh.h, n = sh.n;
      const pad = radius + c;
      const i0 = Math.max(0, Math.floor((s.x - pad - ox) / c)), i1 = Math.min(n - 1, Math.ceil((s.x + pad - ox) / c));
      const j0 = Math.max(0, Math.floor((s.z - pad - oz) / c)), j1 = Math.min(sh.m - 1, Math.ceil((s.z + pad - oz) / c));
      let best = Infinity, bx = 0, bz = 0;
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const k = i + j * n;
          const y = b.y + h[k];
          if (y <= low) continue;
          if (y <= thr) {
            // (between a step and the knees: a wall only where the face is steep)
            const dn = Math.max(
              i > 0 ? h[k] - h[k - 1] : 0, i < n - 1 ? h[k] - h[k + 1] : 0,
              j > 0 ? h[k] - h[k - n] : 0, j < sh.m - 1 ? h[k] - h[k + n] : 0);
            if (dn / c <= walkSlope) continue;
          }
          const px = ox + i * c, pz = oz + j * c;
          const ex = s.x - px, ez = s.z - pz, d2 = ex * ex + ez * ez;
          if (d2 < best) { best = d2; bx = px; bz = pz; }
        }
      }
      const reachW = radius + 0.5 * c;
      if (best < reachW * reachW) {
        let d = Math.sqrt(best), nx: number, nz: number, push: number;
        if (d > 0.25 * c) { nx = (s.x - bx) / d; nz = (s.z - bz) / d; push = reachW - d; }
        else {
          // deep in it: out along the line from its middle, past its last wall node on that line
          const l = Math.hypot(dx, dz);
          nx = l > 1e-6 ? dx / l : 1; nz = l > 1e-6 ? dz / l : 0;
          let r = 0;
          for (let q = 0; q <= 32; q++) {
            const rr = (q / 32) * (b.reach + c);
            if (rockTopAt(b, b.x + nx * rr, b.z + nz * rr) > low) r = rr;
          }
          d = l;
          push = Math.max(0, r + reachW - l);
        }
        s.x += nx * push; s.z += nz * push;
        const vn = s.vx * nx + s.vz * nz;
        if (vn < 0) { s.vx -= vn * nx; s.vz -= vn * nz; }
        dx = s.x - b.x; dz = s.z - b.z;
      }
    }
    // the top under the feet (the middle, or a little round it: a body stands on a rock's edge till its middle is off)
    const fr = radius * 0.6;
    let top = -Infinity;
    for (let q = 0; q < 5; q++) {
      const px = s.x + (q === 1 ? fr : q === 2 ? -fr : 0), pz = s.z + (q === 3 ? fr : q === 4 ? -fr : 0);
      const y = rockTopAt(b, px, pz);
      if (y <= thr && y > top) top = y;
    }
    if (top > floor) {
      floor = top;
      const e = 0.25;
      const lo = top - 0.6;
      const xp = Math.max(lo, rockTopAt(b, s.x + e, s.z)), xm = Math.max(lo, rockTopAt(b, s.x - e, s.z));
      const zp = Math.max(lo, rockTopAt(b, s.x, s.z + e)), zm = Math.max(lo, rockTopAt(b, s.x, s.z - e));
      rockFloorSlope.gx = (xp - xm) / (2 * e);
      rockFloorSlope.gz = (zp - zm) / (2 * e);
    }
  }
  return floor;
}

// ---- the rocks round the hero ---------------------------------------------------------------------
/**
 * The rocks round (x, z) are the field's from now on (the hero's place, or the camera's): the cells of the box round
 * it (ROCK_ZONE cells a side per tier) are listed for the GPU, which leaves them to this list.
 */
export function focusRocks(x: number, z: number, seed: number): void {
  follow(seed);
  const c0 = ROCK_TIERS[ROCK_SOLID_FROM].cell;
  const cx = Math.floor(x / c0), cz = Math.floor(z / c0);
  if (cx === zone.cx && cz === zone.cz) return;
  zone.cx = cx; zone.cz = cz;
  zone.list.length = 0;
  for (let t = 0; t < ROCK_TIERS.length; t++) {
    const half = ROCK_ZONE[t];
    const r = zone.rect;
    if (half <= 0) { r[t * 4] = 1; r[t * 4 + 1] = 1; r[t * 4 + 2] = 0; r[t * 4 + 3] = 0; continue; }
    const cell = ROCK_TIERS[t].cell;
    const tx = Math.floor(x / cell), tz = Math.floor(z / cell);
    r[t * 4] = tx - half; r[t * 4 + 1] = tz - half; r[t * 4 + 2] = tx + half; r[t * 4 + 3] = tz + half;
    for (let iz = tz - half; iz <= tz + half; iz++) {
      for (let ix = tx - half; ix <= tx + half; ix++) {
        const b = bodyInCell(t, ix, iz, seed);
        if (b) zone.list.push(b);
      }
    }
  }
}

/** Rock b leaves its cell (lifted, pushed): its cell stays empty, and it is a free body from now on. */
export function takeRock(b: RockBody): void {
  if (b.state !== 'ground') return;
  if (!takenKeys.has(b.key)) {
    takenKeys.add(b.key);
    taken.push(b.key);
    if (taken.length > ROCK_TAKEN_CAP) {
      // (the oldest taken gives way: its rock is back on its cell, wherever the one taken from it went)
      const k = taken.shift()!;
      takenKeys.delete(k);
      const i = free.findIndex((f) => f.key === k);
      if (i >= 0 && free[i].state === 'loose') free.splice(i, 1);
      else if (i >= 0) free[i].key = NaN;
      cells.delete(k);
    }
  }
  b.state = 'loose';
  b.age = 0;
  b.circle = null;
  free.push(b);
  if (free.length > FREE_CAP) {
    // (the oldest lying still gives way)
    let k = -1;
    for (let i = 0; i < free.length; i++) if (free[i].state === 'loose' && free[i] !== b && (k < 0 || free[i].age > free[k].age)) k = i;
    if (k >= 0) free.splice(k, 1);
  }
}

/** The free rocks (carried, flying, rolling, lying where they came to rest). */
export function freeRocks(): readonly RockBody[] {
  return free;
}

/** A free rock goes (sunk in deep water). */
export function dropRock(b: RockBody): void {
  const i = free.indexOf(b);
  if (i >= 0) free.splice(i, 1);
}

/** The rock moved: its solid is made again when next touched. */
export function movedRock(b: RockBody): void {
  b.shape = null;
  b.circle = null;
}

// ---- rocks on the move ----------------------------------------------------------------------------
const TREES: Array<{ x: number; z: number; y: number; r: number; top: number }> = [];
let treesNear: ((x: number, z: number, seed: number, out: typeof TREES) => typeof TREES) | null = null;
/** The trees' trunks, for rocks on the move (set by forest/collision.ts, which imports this module). */
export function setTreesNear(f: typeof treesNear): void {
  treesNear = f;
}

/** Turn rock b by (wx, wy, wz)·dt (world axes). */
function spin(b: RockBody, dt: number): void {
  const ax = b.wx * dt, ay = b.wy * dt, az = b.wz * dt;
  const a = Math.hypot(ax, ay, az);
  if (a < 1e-7) return;
  const s = Math.sin(0.5 * a) / a, c = Math.cos(0.5 * a);
  const dx = ax * s, dy = ay * s, dz = az * s;
  const { qx, qy, qz, qw } = b;
  b.qx = c * qx + dx * qw + (dy * qz - dz * qy);
  b.qy = c * qy + dy * qw + (dz * qx - dx * qz);
  b.qz = c * qz + dz * qw + (dx * qy - dy * qx);
  b.qw = c * qw - (dx * qx + dy * qy + dz * qz);
  const l = Math.hypot(b.qx, b.qy, b.qz, b.qw);
  b.qx /= l; b.qy /= l; b.qz /= l; b.qw /= l;
}

/** Rolling resistance (× g) of a rock (no ball: its faces catch the ground): a boulder rolls on, a stone stops sooner. */
function rollingDrag(b: RockBody): number {
  return b.tier >= 2 ? 0.12 : 0.22;
}

const OTHERS: RockBody[] = [];

/**
 * One step of a flying or rolling rock: a flying one falls and turns, comes down on the ground (a bounce, then it rolls)
 * or in water (it sinks: 'water'); a rolling one runs down the slopes, turning as it goes, slowed by the ground, off a
 * ledge into the air, round trees and other rocks, and lies still at last ('rest'). 'ground': it came down hard this
 * step (a thud).
 */
export function stepRock(b: RockBody, dt: number, seed: number): 'water' | 'ground' | 'rest' | null {
  b.age += dt;
  const g = PLAYER.gravity;
  let what: 'water' | 'ground' | 'rest' | null = null;
  if (b.state === 'flying') {
    b.vy -= g * dt;
    b.x += b.vx * dt; b.y += b.vy * dt; b.z += b.vz * dt;
    spin(b, dt);
    const [land, water] = surfaceAt(b.x, b.z, seed, SURF);
    if (b.y - b.roll < water && water > land + 0.3) { what = 'water'; return what; }
    if (b.y - b.roll <= land) {
      b.y = land + b.roll;
      const hard = -b.vy;
      if (hard > 3.5) {
        b.vy = hard * 0.3; b.vx *= 0.55; b.vz *= 0.55;
        b.wx *= 0.5; b.wy *= 0.5; b.wz *= 0.5;
        what = 'ground';
      } else {
        b.state = 'rolling';
        b.vy = 0; b.vx *= 0.6; b.vz *= 0.6;
        if (hard > 1.5) what = 'ground';
      }
    }
  } else if (b.state === 'rolling') {
    const e = 0.3;
    const [land, water] = surfaceAt(b.x, b.z, seed, SURF);
    if (water > land + b.roll * 1.2) { what = 'water'; return what; }
    const gx = (elevation(b.x + e, b.z, seed, 0) - elevation(b.x - e, b.z, seed, 0)) / (2 * e);
    const gz = (elevation(b.x, b.z + e, seed, 0) - elevation(b.x, b.z - e, seed, 0)) / (2 * e);
    const m2 = gx * gx + gz * gz;
    // down the slope (a ball rolling: 5/7 of g along it), slowed by the ground
    const k = (5 / 7) * g / (1 + m2);
    b.vx -= gx * k * dt; b.vz -= gz * k * dt;
    let v = Math.hypot(b.vx, b.vz);
    const drag = rollingDrag(b) * g * dt + 0.02 * v * v * dt;
    if (v > 1e-6) { const f = Math.max(0, v - drag) / v; b.vx *= f; b.vz *= f; v *= f; }
    b.x += b.vx * dt; b.z += b.vz * dt;
    const ground = elevation(b.x, b.z, seed, 0);
    if (ground < b.y - b.roll - 0.35 && v > 1) {
      // off a ledge
      b.state = 'flying';
      b.vy = 0;
    } else b.y += (ground + b.roll - b.y) * Math.min(1, 14 * dt);
    // turning as it rolls: about up × v
    if (v > 1e-4) { b.wx = b.vz / b.roll; b.wy = 0; b.wz = -b.vx / b.roll; }
    else { b.wx = 0; b.wy = 0; b.wz = 0; }
    spin(b, dt);
    if (v < 0.15 && Math.sqrt(m2) * (5 / 7) < rollingDrag(b) * 1.5) b.still += dt; else b.still = 0;
    if (b.still > 0.4) {
      b.state = 'loose';
      b.vx = 0; b.vz = 0; b.wx = 0; b.wy = 0; b.wz = 0;
      b.y = ground + b.roll * 0.88; // (settled a little into the ground)
      movedRock(b);
      what = 'rest';
    }
  } else return null;
  // trees and the rocks it meets: it glances off them
  if (treesNear) {
    for (const t of treesNear(b.x, b.z, seed, TREES)) {
      if (b.y - b.roll > t.y + t.top) continue;
      bounceOff(b, t.x, t.z, t.r + b.roll * 0.85);
    }
  }
  rockBodiesNear(b.x, b.z, seed, b.reach + 0.5, OTHERS);
  for (const o of OTHERS) {
    if (o === b || o.state === 'flying' || o.state === 'rolling') continue;
    // (it rolls over a stone much smaller than itself)
    if (b.y - b.roll > o.y + o.height * 0.6 || o.height < b.roll * 0.6) continue;
    bounceOff(b, o.x, o.z, (o.circle ? o.circle.r : o.reach * BODY_SHARE) + b.roll * 0.8);
  }
  movedRock(b);
  return what;
}

function bounceOff(b: RockBody, x: number, z: number, min: number): void {
  const dx = b.x - x, dz = b.z - z, d = Math.hypot(dx, dz);
  if (d >= min) return;
  const nx = d > 1e-6 ? dx / d : 1, nz = d > 1e-6 ? dz / d : 0;
  b.x += nx * (min - d); b.z += nz * (min - d);
  const vn = b.vx * nx + b.vz * nz;
  if (vn < 0) { b.vx -= 1.4 * vn * nx; b.vz -= 1.4 * vn * nz; }
}

// ---- the GPU's list -------------------------------------------------------------------------------
/**
 * What the rocks' GPU side reads each frame (gpu/rocks.ts): per tier the box of cells the field owns (ix0, iz0, ix1,
 * iz1: inclusive; empty when from > to), the cells taken (tier, ix, iz, 0), and the list of rocks to draw
 * (rocks.wgsl.ts RockSrc, 16 words each: chunk-local x, y or the lift over the land, z, size; the turn; the axes'
 * scale, 0; tier, kind, random number, 1: on its cell (the GPU sets it on its land) or 0: where it is).
 */
export type RockFrame = { zone: Int32Array; taken: Int32Array; nTaken: number; list: ArrayBuffer; nList: number };

export function createRockFrame(): RockFrame {
  return { zone: new Int32Array(ROCK_TIERS.length * 4), taken: new Int32Array(ROCK_TAKEN_CAP * 4), nTaken: 0, list: new ArrayBuffer(ROCK_LIST_CAP * 64), nList: 0 };
}

/** The field's rocks for a frame drawn from camera (camX, camZ) (its chunk origin): into `out`. */
export function packRocks(camX: number, camZ: number, out: RockFrame): RockFrame {
  const ox = Math.floor(camX / CHUNK_SIZE) * CHUNK_SIZE, oz = Math.floor(camZ / CHUNK_SIZE) * CHUNK_SIZE;
  out.zone.set(zone.rect);
  let n = 0;
  for (const k of taken) {
    // (back out of the key: tier, ix, iz)
    const iz = (k % 2097152) - 1048576;
    const rest = Math.floor(k / 2097152);
    const ix = (rest % 2097152) - 1048576, t = Math.floor(rest / 2097152);
    out.taken[n * 4] = t; out.taken[n * 4 + 1] = ix; out.taken[n * 4 + 2] = iz; out.taken[n * 4 + 3] = 0;
    n++;
  }
  out.nTaken = n;
  const f = new Float32Array(out.list), u = new Uint32Array(out.list);
  let m = 0;
  const put = (b: RockBody, onCell: boolean) => {
    if (m >= ROCK_LIST_CAP) return;
    const o = m * 16;
    f[o] = b.x - ox; f[o + 1] = onCell ? b.lift : b.y; f[o + 2] = b.z - oz; f[o + 3] = b.size;
    f[o + 4] = b.qx; f[o + 5] = b.qy; f[o + 6] = b.qz; f[o + 7] = b.qw;
    f[o + 8] = b.ax; f[o + 9] = b.ay; f[o + 10] = b.az; f[o + 11] = 0;
    u[o + 12] = b.tier; u[o + 13] = b.kind; u[o + 14] = b.rnd; u[o + 15] = onCell ? 1 : 0;
    m++;
  };
  for (const b of zone.list) if (!takenKeys.has(b.key)) put(b, true);
  for (const b of free) put(b, false);
  out.nList = m;
  return out;
}
