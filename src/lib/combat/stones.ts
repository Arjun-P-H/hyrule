/**
 * Stones in the hero's hands: which rock it can lift or roll, and the numbers of carrying, throwing and pushing.
 * ---------------------------------------------------------------------------
 * By how big it is (its widest reach from its middle, m): a stone (up to a third of a metre) is lifted in one hand,
 * carried at the shoulder and thrown overhand, far and fast; a rock (up to 0.6 m: a metre and a bit across) is heaved
 * up in both hands, carried overhead at a walk and hurled, short and hard, knocking a foe down; a boulder (the third
 * tier's, up to 1.7 m) is pushed with both hands into a roll that runs on down the slopes and through foes. Bigger
 * rocks — a broad slab, the landmarks' — do not move.
 * The rocks themselves (their places, solids, flight and rolling) are rocks/field.ts's; the combat world (world.ts)
 * runs the acts.
 */

import type { HeroState } from '../hero/locomotion';
import { rockBodiesNear, type RockBody } from '../rocks/field';

export const STONE = {
  /** m from the hero to a rock's edge within which it is lifted or pushed. */
  reach: 0.9,
  /** A rock reaching up to this far from its middle (m) is held in one hand; up to `liftMax` in both. */
  handMax: 0.34,
  liftMax: 0.62,
  /** A boulder reaching up to this (m), and up to `rollTall` tall, rolls when pushed (bigger ones do not move). */
  rollMax: 1.7,
  rollTall: 2.6,
  /** s to lift it (one hand, both), to put it down, to throw it (wind-up, release at `release` of it, follow-through). */
  liftTime: [0.5, 0.85],
  downTime: 0.5,
  hurlTime: [0.6, 0.9],
  release: 0.52,
  /** Thrown: speed (m/s), damage, knock-back (m/s) — one hand, both. */
  speed: [17, 10.5],
  /** Aimed at a point nearer than this (m), a throw takes its own arc instead (the crosshair on the ground just ahead). */
  minRange: [14, 7],
  /** A foe within this (m) and ±20° of the view draws a throw to it. */
  aimRange: [26, 12],
  damage: [8, 18],
  knock: [3.5, 7.5],
  /** × the speed on foot while carrying. */
  carrySpeed: [0.85, 0.55],
  /** A push: its length (s), when the hands drive (share of it), the speed it gives a boulder of size 1 (÷ √size). */
  pushTime: 0.8,
  pushAt: 0.4,
  pushSpeed: 5.2,
  /** A rolling boulder's blow: × its speed (m/s) × √size, at most `rollMaxDamage`; no harm under `rollHarm` m/s. */
  rollDamage: 4,
  rollMaxDamage: 48,
  rollHarm: 1.4,
  /** A thrown stone harms from this speed (m/s). */
  flyHarm: 3,
} as const;

/** What can be done with rock b: 0 nothing, 1 lift in one hand, 2 lift in both, 3 push into a roll. */
export function stoneClass(b: RockBody): 0 | 1 | 2 | 3 {
  if (b.state !== 'ground' && b.state !== 'loose') return 0;
  if (b.tier === 1) return b.reach <= STONE.handMax ? 1 : b.reach <= STONE.liftMax ? 2 : 0;
  if (b.tier === 2) return b.reach <= STONE.rollMax && b.height <= STONE.rollTall ? 3 : 0;
  return 0;
}

const NEAR: RockBody[] = [];

/** The rock the hero can lift or push now (the nearest within reach, ahead of it or right at its feet), or null. */
export function stoneNear(s: HeroState, seed: number): RockBody | null {
  const fx = Math.sin(s.yaw), fz = Math.cos(s.yaw);
  let best: RockBody | null = null, bd = Infinity;
  for (const b of rockBodiesNear(s.x, s.z, seed, 3, NEAR)) {
    if (!stoneClass(b)) continue;
    const dx = b.x - s.x, dz = b.z - s.z;
    const d = Math.hypot(dx, dz);
    const edge = d - b.reach * 0.8;
    if (edge > STONE.reach) continue;
    if (Math.abs(b.y - s.y) > 1.4 + b.height) continue;
    // ahead (or right underfoot)
    const ahead = d > 1e-6 ? (dx * fx + dz * fz) / d : 1;
    if (ahead < 0.1 && edge > 0.15) continue;
    const score = edge - 0.4 * ahead;
    if (score < bd) { bd = score; best = b; }
  }
  return best;
}
