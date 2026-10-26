/**
 * The portal between the worlds (world.ts): a ring of shrine stone on the
 * land near the spawn, and its twin on a valley floor in the cloud world.
 * Walking through either takes the hero to the other (the trip:
 * engine/engineCore.ts), out in front of the twin, facing away from it.
 * ---------------------------------------------------------------------------
 * A portal is the centre of its ring's base and the way its opening faces
 * (yaw: the direction (sin yaw, cos yaw), as SPAWN.yaw). Pure functions of
 * the user's seed and the world.
 */

import { PORTAL, SPAWN } from './constants';
import { elevation, slope, waterLevel } from './terrainMath';
import { worldSeed, type World } from './world';

type Portal = { x: number; y: number; z: number; yaw: number };

/** How far in front of the ring the hero comes out (m). */
const EXIT = 4;

/** Level, dry ground for the ring round (x, z) on the land. */
function ringGround(x: number, z: number, seed: number): boolean {
  if (elevation(x, z, seed, 0) < waterLevel(x, z, seed) + 1.2) return false; // (the cheap test first: a search may cross a sea)
  if (slope(x, z, seed) > 0.25) return false;
  for (let k = 0; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2, px = x + Math.cos(a) * 5, pz = z + Math.sin(a) * 5;
    if (slope(px, pz, seed) > 0.4 || elevation(px, pz, seed, 0) < waterLevel(px, pz, seed) + 0.5) return false;
  }
  return true;
}

/**
 * The land's ring: the first level, dry spot 18–40 m to the right of the spawn's view, facing the spawn;
 * failing that (a seed whose spawn lies in the sea), the nearest one out to 3 km all round.
 */
function landPortal(seed: number): Portal {
  const at = (a: number, d: number): Portal | null => {
    const x = SPAWN.x + Math.sin(a) * d, z = SPAWN.z + Math.cos(a) * d;
    return ringGround(x, z, seed) ? { x, y: elevation(x, z, seed, 0), z, yaw: Math.atan2(SPAWN.x - x, SPAWN.z - z) } : null;
  };
  for (let d = 18; d <= 40; d += 4) {
    for (const da of [0, 0.3, -0.3, 0.6, -0.6]) {
      const p = at(SPAWN.yaw + Math.PI / 2 + da, d);
      if (p) return p;
    }
  }
  for (let d = 48; d <= 3000; d += 24) {
    for (let k = 0; k < 24; k++) {
      const p = at(SPAWN.yaw + Math.PI / 2 + (k / 24) * Math.PI * 2, d);
      if (p) return p;
    }
  }
  // (no level spot near: beside the spawn all the same)
  const x = SPAWN.x + Math.sin(SPAWN.yaw + Math.PI / 2) * 25, z = SPAWN.z + Math.cos(SPAWN.yaw + Math.PI / 2) * 25;
  return { x, y: elevation(x, z, seed, 0), z, yaw: SPAWN.yaw - Math.PI / 2 };
}

/** The cloud world's ring: where the land's spawn is, on the cloud sea, facing the way the spawn's view does. */
function cloudPortal(seed: number): Portal {
  const s = worldSeed(seed, 'clouds');
  return { x: SPAWN.x, y: elevation(SPAWN.x, SPAWN.z, s, 0), z: SPAWN.z, yaw: SPAWN.yaw };
}

const portals = new Map<string, Portal>();
/** The ring in `world` for the user's `seed` (cached: the hero's every step asks). */
export function portalOf(world: World, seed: number): Portal {
  const key = `${world}${seed}`;
  let p = portals.get(key);
  if (!p) {
    if (portals.size >= 8) portals.clear();
    p = world === 'clouds' ? cloudPortal(seed) : landPortal(seed);
    portals.set(key, p);
  }
  return p;
}

/** Where the hero comes out of `world`'s ring: EXIT m in front of it, facing away from it (on the ground there). */
export function exitOf(world: World, seed: number): Portal {
  const p = portalOf(world, seed);
  const x = p.x + Math.sin(p.yaw) * EXIT, z = p.z + Math.cos(p.yaw) * EXIT;
  return { x, y: elevation(x, z, worldSeed(seed, world), 0), z, yaw: p.yaw };
}

/** Where the hero starts (and respawns) in `world`: the land's spawn, or out of the cloud world's ring. */
export function spawnPoint(world: World, seed: number): Portal {
  return world === 'clouds' ? exitOf('clouds', seed) : { x: SPAWN.x, y: elevation(SPAWN.x, SPAWN.z, seed, 0), z: SPAWN.z, yaw: SPAWN.yaw };
}

/**
 * Does a hero going from a to b (feet, m) pass through the ring's opening? It crosses the ring's
 * plane, its hips (a metre over its feet) inside the circle, half a metre in from the blocks.
 */
export function portalCrossed(p: Portal, ax: number, ay: number, az: number, bx: number, by: number, bz: number): boolean {
  const nx = Math.sin(p.yaw), nz = Math.cos(p.yaw);
  const da = (ax - p.x) * nx + (az - p.z) * nz, db = (bx - p.x) * nx + (bz - p.z) * nz;
  if (da * db > 0 || da === db) return false;
  const t = da / (da - db);
  const x = ax + (bx - ax) * t, y = ay + (by - ay) * t + 1, z = az + (bz - az) * t;
  const across = (x - p.x) * Math.cos(p.yaw) - (z - p.z) * Math.sin(p.yaw), up = y - (p.y + PORTAL.centre);
  return Math.hypot(across, up) < PORTAL.radius - 0.5;
}
