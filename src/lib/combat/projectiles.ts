/**
 * What flies: arrows (the hero's, the archers' and other players') and thrown weapons.
 * ---------------------------------------------------------------------------
 * Each tick a projectile moves along its ballistic path; the world
 * (combat/world.ts) tests the segment it swept against the bodies, this
 * module against the world: the ground (the CPU twin's surface, a sky
 * island's top up there; the crossing found by bisection), the trunks of the
 * trees, and the water (an arrow sinks, a weapon goes under with a splash).
 * An arrow that hits the ground or a tree stays stuck in it a while; a thrown
 * weapon tumbles end over end and, where it comes down, is left lying to be
 * picked up again.
 */

import { surfaceAt } from '../terrainMath';
import { skyGround } from '../skyworld/realm';
import { trunksNear, type Trunk } from '../forest/collision';
import { COMBAT } from '../constants';

export type Projectile = {
  live: boolean;
  arrow: boolean;          // an arrow, else a thrown weapon
  friendly: boolean;       // the hero's (strikes foes), else a foe's (strikes the hero)
  x: number; y: number; z: number;
  px: number; py: number; pz: number; // where it was a tick ago (the segment it swept)
  vx: number; vy: number; vz: number;
  gravity: number;
  damage: number;
  weapon: number;          // WEAPONS index (a thrown weapon), and the blows it has left
  wear: number;
  spin: number;            // a thrown weapon's tumble (rad)
  stuck: boolean;          // an arrow stopped in the ground or a tree
  age: number;             // s since it was loosed (or stuck)
  shooter: number;         // the foe slot that loosed it (−1: the hero)
  from: string | null;     // another player's (its id: combat/pvp.ts): it strikes only the hero here
  ref: number;             // the shot's number (the hero's, or the other player's) for what came of it
  dx: number; dy: number; dz: number; // its heading when it stuck
};

export function createProjectile(): Projectile {
  return {
    live: false, arrow: true, friendly: true, x: 0, y: 0, z: 0, px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0,
    gravity: 0, damage: 0, weapon: -1, wear: 0, spin: 0, stuck: false, age: 0, shooter: -1, from: null, ref: 0, dx: 0, dy: 0, dz: 1,
  };
}

/** What a step ran into. */
export type Landed = 'flying' | 'ground' | 'tree' | 'water';

const SURF: [number, number] = [0, 0];
const trunks: Trunk[] = [];

/** The ground and water under (x, y, z): a sky island's top when one is under the point, else the land. */
function groundAt(x: number, y: number, z: number, seed: number): number {
  // (islands hang low over the lowlands: one may be over any point, at any height)
  const t = skyGround(x, y, z, seed, 0.5);
  if (t === t) { SURF[1] = -Infinity; return t; }
  surfaceAt(x, z, seed, SURF);
  return SURF[0];
}

/**
 * Advance projectile p by dt: gravity, the tumble, and the world it runs
 * into. A stuck arrow only ages.
 */
export function stepProjectile(p: Projectile, dt: number, seed: number): Landed {
  p.age += dt;
  p.px = p.x; p.py = p.y; p.pz = p.z;
  if (p.stuck) return 'flying';
  p.vy -= p.gravity * dt;
  p.x += p.vx * dt;
  p.y += p.vy * dt;
  p.z += p.vz * dt;
  if (!p.arrow) p.spin += dt * 17;
  const g = groundAt(p.x, p.y, p.z, seed);
  const water = SURF[1];
  if (p.y <= g) {
    // Bisect the crossing between the last point (above) and this one.
    let a = 0, b = 1;
    for (let k = 0; k < 6; k++) {
      const m = 0.5 * (a + b);
      const x = p.px + (p.x - p.px) * m, y = p.py + (p.y - p.py) * m, z = p.pz + (p.z - p.pz) * m;
      if (y <= groundAt(x, y, z, seed)) b = m; else a = m;
    }
    p.x = p.px + (p.x - p.px) * b; p.y = p.py + (p.y - p.py) * b; p.z = p.pz + (p.z - p.pz) * b;
    if (water > g && p.y < water) return 'water';
    stick(p);
    return 'ground';
  }
  if (water > g && p.y < water) return 'water';
  // tree trunks (a crown lets an arrow through)
  for (const t of trunksNear(p.x, p.z, seed, trunks)) {
    if (p.y < t.y - 0.3 || p.y > t.y + t.top) continue;
    const ex = p.x - t.x, ez = p.z - t.z;
    if (ex * ex + ez * ez < t.r * t.r) {
      // back out to the bark along the way it came
      const l = Math.hypot(ex, ez) || 1;
      p.x = t.x + (ex / l) * t.r; p.z = t.z + (ez / l) * t.r;
      stick(p);
      return 'tree';
    }
  }
  return 'flying';
}

function stick(p: Projectile): void {
  const l = Math.hypot(p.vx, p.vy, p.vz) || 1;
  p.dx = p.vx / l; p.dy = p.vy / l; p.dz = p.vz / l;
  p.vx = 0; p.vy = 0; p.vz = 0;
  p.stuck = p.arrow;
  p.age = 0;
}

/**
 * The launch velocity (into `out`) for a projectile from (x, y, z) at `speed`
 * under `gravity` to reach (tx, ty, tz): the lower of the two ballistic arcs
 * (straight at it when out of range).
 */
export function aimAt(x: number, y: number, z: number, tx: number, ty: number, tz: number, speed: number, gravity: number, out: Float64Array): void {
  const dx = tx - x, dy = ty - y, dz = tz - z;
  const d = Math.hypot(dx, dz);
  const v2 = speed * speed;
  const disc = v2 * v2 - gravity * (gravity * d * d + 2 * dy * v2);
  let pitch: number;
  if (d < 1e-6) pitch = dy >= 0 ? Math.PI / 2 : -Math.PI / 2;
  else if (disc < 0 || gravity <= 0) pitch = Math.atan2(dy, d);
  else pitch = Math.atan2(v2 - Math.sqrt(disc), gravity * d);
  const hx = d > 1e-6 ? dx / d : 0, hz = d > 1e-6 ? dz / d : 0;
  out[0] = hx * Math.cos(pitch) * speed;
  out[1] = Math.sin(pitch) * speed;
  out[2] = hz * Math.cos(pitch) * speed;
}

/** Whether segment a → b passes within r of segment c → d (closest approach), and where along a → b (into hitT). */
export function segmentsMeet(
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number, dx: number, dy: number, dz: number, r: number,
): number {
  // (closest points of two segments, clamped; Ericson, Real-Time Collision Detection 5.1.9)
  const d1x = bx - ax, d1y = by - ay, d1z = bz - az;
  const d2x = dx - cx, d2y = dy - cy, d2z = dz - cz;
  const rx = ax - cx, ry = ay - cy, rz = az - cz;
  const a = d1x * d1x + d1y * d1y + d1z * d1z, e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;
  let s: number, t: number;
  if (a <= 1e-12 && e <= 1e-12) { s = 0; t = 0; }
  else if (a <= 1e-12) { s = 0; t = Math.min(1, Math.max(0, f / e)); }
  else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= 1e-12) { t = 0; s = Math.min(1, Math.max(0, -c / a)); }
    else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const den = a * e - b * b;
      s = den > 1e-12 ? Math.min(1, Math.max(0, (b * f - c * e) / den)) : 0;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = Math.min(1, Math.max(0, -c / a)); }
      else if (t > 1) { t = 1; s = Math.min(1, Math.max(0, (b - c) / a)); }
    }
  }
  const qx = ax + d1x * s - (cx + d2x * t), qy = ay + d1y * s - (cy + d2y * t), qz = az + d1z * s - (cz + d2z * t);
  return qx * qx + qy * qy + qz * qz <= r * r ? s : -1;
}

/** Arrows stay stuck this long. */
export const STUCK_LIFE = COMBAT.arrowLife;
