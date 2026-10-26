import { CAMERA } from '@/lib/constants';
import { HELD } from './keybinds';

/**
 * Key Look: the look keys held (keybinds.ts: lookUp, lookDown, lookLeft, lookRight) turn the camera as the
 * mouse does, by time instead of by pixels: up / down tilt it, left / right turn it, both together give
 * a diagonal, and two opposite keys cancel. The Mouse Look setting scales the speed, so one setting
 * serves both. A turn starts slowly, for fine aim, and gets to full speed after `ramp` seconds held.
 */
export const KEY_LOOK = {
  rate: 1.9,         // radians a second at full speed with the default Mouse Look (109°/s)
  start: 0.3,        // a turn starts at this share of the rate…
  ramp: 0.4,         // …and reaches all of it after this many seconds held
  pitchLimit: 1.45,  // as far up or down as the mouse can tilt the view
} as const;

/** How long each axis has been held, and which way (so a turn that reverses starts slowly again). */
export function createKeyLook() {
  return { dirX: 0, heldX: 0, dirY: 0, heldY: 0 };
}

const speed = (held: number) => KEY_LOOK.start + (1 - KEY_LOOK.start) * Math.min(1, held / KEY_LOOK.ramp);

/**
 * Turn `view` by the look keys in `mask` (keybinds.ts HELD) over `dt` seconds. `sensitivity` is the Mouse Look
 * setting: its default turns at `rate`, twice the sensitivity at twice that.
 */
export function stepKeyLook(
  look: ReturnType<typeof createKeyLook>,
  mask: number,
  dt: number,
  sensitivity: number,
  view: { yaw: number; pitch: number },
): void {
  const dx = ((mask & HELD.lookRight) ? 1 : 0) - ((mask & HELD.lookLeft) ? 1 : 0);
  const dy = ((mask & HELD.lookUp) ? 1 : 0) - ((mask & HELD.lookDown) ? 1 : 0);
  look.heldX = dx !== 0 && dx === look.dirX ? look.heldX + dt : 0;
  look.heldY = dy !== 0 && dy === look.dirY ? look.heldY + dt : 0;
  look.dirX = dx;
  look.dirY = dy;
  const turn = KEY_LOOK.rate * (sensitivity / CAMERA.sensitivity) * dt;
  if (dx !== 0) view.yaw += dx * turn * speed(look.heldX);
  if (dy !== 0) view.pitch = Math.max(-KEY_LOOK.pitchLimit, Math.min(KEY_LOOK.pitchLimit, view.pitch + dy * turn * speed(look.heldY)));
}
