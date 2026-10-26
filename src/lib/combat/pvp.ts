/**
 * Players against each other (multiplayer): the other players as the combat
 * world (world.ts) sees them, and the messages it trades with theirs through
 * the relay (net/session.ts, server.mjs).
 * ---------------------------------------------------------------------------
 * Each player's own browser settles what strikes its own hero: what it sees
 * is what hits it, and its shield, a Perfect Guard or a dodge answers a blow
 * at the moment the blow lands on its screen. So a player's blows and shots
 * go out stamped with its clock (the clock its snapshots carry,
 * net/remoteHeroes.ts), and the others hold each one until their view of that
 * player has played back to that moment: a blow is then tested over its
 * swing's window, from where they see the attacker, against their own hero; a
 * shot is loosed in their world from where it left the bow and flies there.
 * What came of it goes back to the attacker (a hit, a block, a Perfect Guard
 * that dazes it, a fall), timed the same way, so it shows along with the
 * struck player's flinch. Where the attacker sees the others, its swing and
 * its arrows meet them at once too, for the spark and the hit-stop; they do
 * no harm there.
 *
 *   BLOW   [1, t, swing, damage, knock, reach, arc, ms, rush]   its window opens at t and lasts ms;
 *          rush 1: a Flurry Rush's cut, which no shield or dodge answers (the player is not slowed
 *          as a foe is, so it is the rule instead)
 *   SHOT   [2, t, shot, x, y, z, vx, vy, vz, damage, weapon]   weapon −1: an arrow
 *   RESULT [3, t, kind, ref, outcome, quarters, x, y, z]  what came of the receiver's BLOW / SHOT `ref`
 */

import { COMBAT, PLAYER } from '../constants';
import type { HeroState } from '../hero/locomotion';
import type { CombatPose, HeroRig } from '../hero/rig';

/** Another player as this world sees it: its hero played back (net/remoteHeroes.ts). */
export type Rival = {
  readonly id: string;
  readonly body: HeroState;
  readonly pose: CombatPose;
  readonly rig: HeroRig;
  /** Its hearts left (quarters) and its heart containers, from its snapshots. */
  health: number;
  hearts: number;
  /** Its clock (ms) at the moment this view of it shows: its messages are timed by it. */
  clock: number;
  /** This world's latest swing and shot that met it (one blow a swing; the spark is shown). */
  hitSwing: number;
  hitShot: number;
};

/** Message kinds (each message's first number). */
export const MSG = { blow: 1, shot: 2, result: 3 } as const;

/** What came of a blow or a shot. */
export const OUTCOME = { hit: 0, block: 1, parry: 2, slip: 3, down: 4, crit: 5 } as const;

/** Numbers in the longest message (a SHOT). */
export const MSG_MAX = 11;

/** The hero's head about its joint (standing: its centre above the ground; the skin and hair's radius), for a headshot. */
export const HERO_HEAD = { y: 1.48, r: 0.13 } as const;

/** Quarter hearts a player loses to `damage` (a weapon's damage × its swing's, an arrow's). */
export function pvpQuarters(damage: number): number {
  return Math.max(1, Math.min(COMBAT.pvpMax, Math.round(damage * COMBAT.pvpQuarters)));
}

/** Joint j of rival r in world space (from its rig, about its root on the ground). */
export function rivalJoint(r: Rival, j: number, out: Float64Array, o = 0): void {
  const s = r.body, sy = Math.sin(s.yaw + r.rig.spin), cy = Math.cos(s.yaw + r.rig.spin);
  const x = r.rig.local[3 * j], y = r.rig.local[3 * j + 1], z = r.rig.local[3 * j + 2];
  out[o] = s.x + cy * x + sy * z;
  out[o + 1] = s.y - PLAYER.groundBias + y;
  out[o + 2] = s.z - sy * x + cy * z;
}

/** Whether rival r holds its shield up toward (x, z) (a blow from there is guarded). */
export function rivalGuards(r: Rival, x: number, z: number): boolean {
  if (r.pose.act !== 'guard') return false;
  const b = r.body, dx = x - b.x, dz = z - b.z, d = Math.hypot(dx, dz) || 1;
  return (dx * Math.sin(b.yaw) + dz * Math.cos(b.yaw)) / d > 0.3;
}
