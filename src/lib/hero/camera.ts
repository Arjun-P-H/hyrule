/**
 * Third-person orbit camera for the hero.
 * ---------------------------------------------------------------------------
 * The mouse sets the orbit yaw / pitch; the eye sits `zoom` metres behind a
 * smoothed look-at point above the hero. Terrain between the hero and the eye
 * shortens the boom (snapping in, easing back out); the eye then stays
 * HERO_CAMERA.clearance above the ground and the water plane, and the view
 * direction is re-aimed from the final eye to the target, so the hero stays
 * framed. One elevation() call per settled frame (two while the boom moves).
 */

import { elevation } from '../terrainMath';
import { skyGround, skyWater } from '../skyworld/realm';
import { HERO_CAMERA as C, PLAYER, VALLEY } from '../constants';
import type { HeroState } from './locomotion';

/** Ground under an eye at height y: a sky island's top or an arch's deck when the eye is above one, else the terrain. */
function groundBelow(x: number, y: number, z: number, seed: number): number {
  const top = skyGround(x, y, z, seed, 0.5);
  return top === top ? top : elevation(x, z, seed, 0);
}

/** The water an eye at height y must stay above: a sky lake or stream under it, else the lakes and sea. */
function waterBelow(x: number, y: number, z: number, seed: number): number {
  const w = skyWater(x, y, z, seed);
  return w === w ? w : VALLEY.wLevel;
}

export type HeroCamera = {
  x: number; y: number; z: number;    // eye, global fp64
  tx: number; ty: number; tz: number; // smoothed look-at target
  boom: number;                       // current eye distance (after obstructions)
  zoom: number;                       // requested distance (mouse wheel)
  yaw: number; pitch: number;         // rendered view direction (eye → target)
  snap: boolean;                      // next update jumps straight to the target
};

export function createHeroCamera(): HeroCamera {
  return { x: 0, y: 0, z: 0, tx: 0, ty: 0, tz: 0, boom: C.distance, zoom: C.distance, yaw: 0, pitch: 0, snap: true };
}

function approach(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}

/** Look-at height above the root for the hero's current state. */
function targetHeight(s: HeroState): number {
  return s.move === 'swim' ? 1.35 : s.move === 'glide' ? 1.7 : C.targetH;
}

export function updateHeroCamera(
  c: HeroCamera, s: HeroState, orbitYaw: number, orbitPitch: number, dt: number, seed: number,
): void {
  const gx = s.x, gy = s.y - PLAYER.groundBias + targetHeight(s), gz = s.z;
  if (c.snap) {
    c.tx = gx; c.ty = gy; c.tz = gz;
  } else {
    // Lag behind the hero for smoothness, but never by more than C.maxLag:
    // a sprint or a long fall must not carry the hero out of frame.
    const kxz = approach(C.followXZ, dt), ky = approach(C.followY, dt);
    c.tx = gx + Math.max(-C.maxLag, Math.min(C.maxLag, (c.tx - gx) * (1 - kxz)));
    c.ty = gy + Math.max(-C.maxLag, Math.min(C.maxLag, (c.ty - gy) * (1 - ky)));
    c.tz = gz + Math.max(-C.maxLag, Math.min(C.maxLag, (c.tz - gz) * (1 - kxz)));
  }

  const cp = Math.cos(orbitPitch);
  const fx = Math.sin(orbitYaw) * cp, fy = Math.sin(orbitPitch), fz = Math.cos(orbitYaw) * cp;
  let boom = c.zoom;
  const ex0 = c.tx - fx * boom, ey0 = c.ty - fy * boom, ez0 = c.tz - fz * boom;
  const e0 = s.ground;
  const ec = groundBelow(ex0, ey0, ez0, seed);
  if (ey0 < ec + C.clearance) {
    // The boom dips into the ground: shorten it to where the line meets the
    // ground, linearised between the hero's footing and the eye's.
    const above = c.ty - (e0 + C.clearance);
    const closing = (c.ty - ey0) + (ec - e0);
    const f = above <= 0 ? 0 : closing > 1e-6 ? above / closing : 1;
    boom = Math.max(C.minDistance, Math.min(boom, f * boom));
  }
  c.boom = c.snap || boom < c.boom ? boom : c.boom + (boom - c.boom) * approach(C.boomOut, dt);
  if (Math.abs(boom - c.boom) < 1e-3) c.boom = boom; // settle exactly (lets the sample below be reused)

  const ex = c.tx - fx * c.boom, ez = c.tz - fz * c.boom;
  // Unobstructed and settled (the usual case) the eye is where the ground was
  // just sampled: reuse it (one elevation() per frame instead of two).
  const eyeGround = ex === ex0 && ez === ez0 ? ec : groundBelow(ex, c.ty - fy * c.boom, ez, seed);
  const floor = Math.max(eyeGround, waterBelow(ex, c.ty - fy * c.boom, ez, seed)) + C.clearance;
  const ey = Math.max(c.ty - fy * c.boom, floor);
  c.x = ex; c.y = ey; c.z = ez;

  const vx = c.tx - ex, vy = c.ty - ey, vz = c.tz - ez;
  const l = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (l > 1e-6) {
    c.yaw = Math.atan2(vx, vz);
    c.pitch = Math.asin(Math.max(-1, Math.min(1, vy / l)));
  } else {
    c.yaw = orbitYaw;
    c.pitch = orbitPitch;
  }
  c.snap = false;
}
