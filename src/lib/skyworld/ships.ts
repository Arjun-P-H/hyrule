/**
 * Skyships under way (SKY_CIVILIZATION.md §7.10, §9): the routes the ships
 * fly round the camera, and where a ship is on its route at a time — pure
 * functions of the realm and the clock (gpu/structures.ts draws them as
 * structures that move).
 * ---------------------------------------------------------------------------
 *   loop   round a circle: inside Galeharbour's ring of islands, ships
 *          circling the anchorage at the islands' height; round every other
 *          Eyrie a ship or two on patrol, just outside its islands.
 *   lane   the trade routes between neighbouring Eyries: an oval, out along
 *          one side of the line between their middles and back along the
 *          other, turning round at each end; high enough to clear every
 *          island and spire under it (and the lone isles between).
 * Ships keep their route's height, rising and falling a little with the wind
 * (the skyship piece's own float); a route's ships are spaced evenly along it.
 */

import { SKY_REALM as S } from '../constants';
import { hashU } from '../terrainMath';
import { archipelagoInCell, cellBuilt, type Archipelago } from './archipelago';
import { rng } from './kit/mesh';
import { outlineMax } from './island';

/** A route: its path (a loop's circle, a lane's two ends and turning radius), height, speed (m/s; a loop's sign the way round), phase (0 … 1) and ships; their palette. */
export type Route = { speed: number; phase: number; ships: number; palette: number } & (
  | { kind: 'loop'; x: number; y: number; z: number; r: number }
  | { kind: 'lane'; ax: number; az: number; bx: number; bz: number; y: number; r: number });

/** How high over the highest island top a lane flies (m): clear of spires, rings and lone isles. */
const LANE_CLEAR = 260;
/** Lone isles reach up to about this (m; their tops). */
const ISLE_TOP = 2700;
const TAU = Math.PI * 2;

const topOf = (a: Archipelago) => Math.max(...a.islands.map((i) => i.y));
const lengthOf = (rt: Route) => rt.kind === 'loop' ? TAU * rt.r : 2 * Math.hypot(rt.bx - rt.ax, rt.bz - rt.az) + TAU * rt.r;

/** An Eyrie's own routes into `out`. */
function eyrieRoutes(a: Archipelago, seed: number, out: Route[]): void {
  const rnd = rng(hashU(a.ix * 7919 + a.iz * 104729 + seed * 17 + 9001));
  const ys = a.islands.filter((i) => i.kind !== 'debris').map((i) => i.y);
  const meanY = ys.reduce((p, q) => p + q, 0) / Math.max(ys.length, 1);
  if (a.cls === 'harbour') {
    // the anchorage: circling inside the ring, clear of its islands
    const inner = Math.min(...a.islands.map((i) => Math.hypot(i.cx - a.cx, i.cz - a.cz) - i.R * outlineMax(i)));
    for (let k = 0; k < 3; k++) {
      const r = Math.max(60, inner * (0.35 + 0.18 * k) - 40);
      out.push({ kind: 'loop', x: a.cx, y: meanY - 15 + 25 * k, z: a.cz, r, speed: (k % 2 ? -1 : 1) * (7 + 3 * rnd()), phase: rnd(), ships: k === 0 ? 1 : 2, palette: a.palette });
    }
    return;
  }
  // patrols just outside the islands
  const outer = Math.max(...a.islands.map((i) => Math.hypot(i.cx - a.cx, i.cz - a.cz) + i.R * outlineMax(i)));
  const n = a.cls === 'crown' ? 2 : a.cls === 'fallen' ? 0 : 1;
  for (let k = 0; k < n; k++) {
    out.push({ kind: 'loop', x: a.cx, y: meanY + 40 + 60 * rnd(), z: a.cz, r: outer + 140 + 120 * rnd(), speed: (rnd() < 0.5 ? -1 : 1) * (9 + 4 * rnd()), phase: rnd(), ships: 1, palette: a.palette });
  }
}

/**
 * The routes of the Eyries laid out within `range` of (x, z) and of the lanes between them and
 * their neighbours, into `out` (cleared). Never lays out an Eyrie.
 */
export function routesNear(x: number, z: number, range: number, seed: number, out: Route[]): Route[] {
  out.length = 0;
  const r = range + S.archReach;
  const i0 = Math.floor((x - r) / S.archCell), i1 = Math.floor((x + r) / S.archCell);
  const j0 = Math.floor((z - r) / S.archCell), j1 = Math.floor((z + r) / S.archCell);
  const at = (ix: number, iz: number) => (cellBuilt('arch', ix, iz, seed) ? archipelagoInCell(ix, iz, seed) : null);
  for (let iz = j0; iz <= j1; iz++) {
    for (let ix = i0; ix <= i1; ix++) {
      const a = at(ix, iz);
      if (!a) continue;
      eyrieRoutes(a, seed, out);
      // a lane to the neighbours east, south-east, south and south-west (each pair once)
      for (const [dx, dz] of [[1, 0], [1, 1], [0, 1], [-1, 1]] as const) {
        const b = at(ix + dx, iz + dz);
        if (!b) continue;
        const rnd = rng(hashU(ix * 92821 + iz * 68917 + (ix + dx) * 31 + (iz + dz) * 7 + seed));
        if (rnd() > 0.55) continue;
        const y = Math.max(topOf(a), topOf(b), ISLE_TOP) + LANE_CLEAR + 60 * rnd();
        out.push({ kind: 'lane', ax: a.cx, az: a.cz, bx: b.cx, bz: b.cz, y, r: 90 + 60 * rnd(), speed: 14 + 6 * rnd(), phase: rnd(), ships: 2, palette: a.palette });
      }
    }
  }
  return out;
}

/**
 * Where ship `k` of a route is at time t (s): x, y, z (world m) and its yaw (local +z, its bow,
 * along its way: toward (sin yaw, cos yaw)), into `o`.
 */
export function shipAt(rt: Route, k: number, t: number, o: Float64Array): Float64Array {
  const len = lengthOf(rt);
  let s = (((rt.phase + k / rt.ships) * len + Math.abs(rt.speed) * t) % len + len) % len;
  if (rt.kind === 'loop') {
    const dir = Math.sign(rt.speed) || 1;
    const th = dir * (s / rt.r);
    o[0] = rt.x + rt.r * Math.cos(th); o[1] = rt.y; o[2] = rt.z + rt.r * Math.sin(th);
    // heading d/dθ (cos, sin) · dir
    o[3] = Math.atan2(-Math.sin(th) * dir, Math.cos(th) * dir);
    return o;
  }
  // the lane: out along one side (offset r to its right), round the far end, back along the other
  const L = Math.hypot(rt.bx - rt.ax, rt.bz - rt.az) || 1;
  const ux = (rt.bx - rt.ax) / L, uz = (rt.bz - rt.az) / L;
  const rx = uz, rz = -ux; // its right (as the structures' yaw: local x)
  o[1] = rt.y;
  if (s < L) {
    o[0] = rt.ax + ux * s + rx * rt.r; o[2] = rt.az + uz * s + rz * rt.r;
    o[3] = Math.atan2(ux, uz);
    return o;
  }
  s -= L;
  const half = Math.PI * rt.r;
  if (s < half) {
    // round the far end: from its right side to its left, about b
    const a = s / rt.r;
    const cx = rx * Math.cos(a) + ux * Math.sin(a), cz = rz * Math.cos(a) + uz * Math.sin(a);
    o[0] = rt.bx + cx * rt.r; o[2] = rt.bz + cz * rt.r;
    o[3] = Math.atan2(-rx * Math.sin(a) + ux * Math.cos(a), -rz * Math.sin(a) + uz * Math.cos(a));
    return o;
  }
  s -= half;
  if (s < L) {
    o[0] = rt.bx - ux * s - rx * rt.r; o[2] = rt.bz - uz * s - rz * rt.r;
    o[3] = Math.atan2(-ux, -uz);
    return o;
  }
  s -= L;
  const a = s / rt.r;
  const cx = -rx * Math.cos(a) - ux * Math.sin(a), cz = -rz * Math.cos(a) - uz * Math.sin(a);
  o[0] = rt.ax + cx * rt.r; o[2] = rt.az + cz * rt.r;
  o[3] = Math.atan2(rx * Math.sin(a) - ux * Math.cos(a), rz * Math.sin(a) - uz * Math.cos(a));
  return o;
}
