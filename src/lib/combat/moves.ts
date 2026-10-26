/**
 * Swings: what the weapon arm does through each attack, shared by the combat
 * simulation (when a swing strikes, how far and how wide, what it does to
 * what it strikes) and the rig (where the hand and the blade are).
 * ---------------------------------------------------------------------------
 * A swing is a few keys over its course u ∈ [0, 1]: the blade's yaw and pitch
 * in the hero's frame (yaw 0 straight ahead, positive to the right; pitch
 * positive up, past 90° back over the shoulder), the hand's distance from the
 * shoulder pivot along the blade, a lift of the hand, and the torso's twist
 * (positive: the chest turned to the right). Between keys a cubic Hermite
 * curve whose tangents are the neighbouring keys' differences, at rest at the
 * first and last key: the arm eases out of the wind-up and whips through the
 * strike without stopping at the keys in between.
 */

/** u, blade yaw (°), blade pitch (°), hand reach (m), hand lift (m), torso twist (°). */
type Key = readonly [number, number, number, number, number, number];

export type Move = {
  readonly name: string;
  readonly dur: number;                    // s (at a weapon pace of 1)
  readonly keys: readonly Key[];
  readonly hit: readonly [number, number]; // the part of u in which it strikes
  readonly chain: number;                  // u from which a press chains the next swing
  readonly lunge: number;                  // m/s forward through `lungeAt`
  readonly lungeAt: readonly [number, number];
  readonly damage: number;                 // × the weapon's damage
  readonly reach: number;                  // m from the swinger's root to the far edge of what it strikes
  readonly arc: number;                    // half-angle (rad) of the sector it sweeps
  readonly knock: number;                  // m/s it drives the struck back
  readonly heavy: boolean;                 // knocks a foe down
};

const D = Math.PI / 180;

function move(m: Omit<Move, 'arc'> & { arc: number }): Move {
  return { ...m, arc: m.arc * D };
}

/** Indices into MOVES. */
export const MOVE = {
  slashR: 0, slashL: 1, slashD: 2, thrust: 3, spin: 4, jump: 5, dash: 6,
  flurryA: 7, flurryB: 8, throw: 9, smash: 10, sweep: 11,
} as const;

/** The sword's four-hit combo, in order (the last a lunging thrust). */
export const COMBO = [MOVE.slashR, MOVE.slashL, MOVE.slashD, MOVE.thrust] as const;

export const MOVES: readonly Move[] = [
  // right to left, flat
  move({ name: 'slash', dur: 0.4, hit: [0.3, 0.64], chain: 0.52, lunge: 2.2, lungeAt: [0.22, 0.6], damage: 1, reach: 2.25, arc: 82, knock: 2.5, heavy: false,
    keys: [[0, 55, -20, 0.32, -0.1, 12], [0.26, 112, 10, 0.38, 0.03, 48], [0.46, 18, 0, 0.53, 0, 4], [0.64, -82, -6, 0.5, -0.02, -40], [1, -70, -22, 0.38, -0.06, -26]] }),
  // back from left to right
  move({ name: 'backhand', dur: 0.4, hit: [0.28, 0.64], chain: 0.52, lunge: 2.2, lungeAt: [0.2, 0.58], damage: 1, reach: 2.25, arc: 82, knock: 2.5, heavy: false,
    keys: [[0, -70, -22, 0.38, -0.06, -26], [0.24, -108, 10, 0.4, 0.05, -46], [0.46, -14, 2, 0.53, 0, 0], [0.64, 86, -4, 0.5, -0.02, 40], [1, 70, -16, 0.38, -0.04, 25]] }),
  // down across, from high on the right
  move({ name: 'downcut', dur: 0.44, hit: [0.34, 0.68], chain: 0.56, lunge: 2.6, lungeAt: [0.26, 0.62], damage: 1.1, reach: 2.25, arc: 62, knock: 3, heavy: false,
    keys: [[0, 70, -16, 0.38, -0.04, 25], [0.3, 52, 74, 0.4, 0.1, 34], [0.5, 6, 14, 0.53, 0.02, 4], [0.68, -44, -42, 0.5, -0.06, -24], [1, -34, -46, 0.4, -0.1, -14]] }),
  // the finisher: pulled back to the hip, then a lunging thrust
  move({ name: 'thrust', dur: 0.56, hit: [0.42, 0.72], chain: 1, lunge: 5.6, lungeAt: [0.38, 0.62], damage: 1.6, reach: 2.6, arc: 34, knock: 7.5, heavy: true,
    keys: [[0, -34, -46, 0.4, -0.1, -14], [0.36, 10, -4, 0.12, -0.05, 36], [0.52, 0, 2, 0.62, 0.02, -10], [0.76, 0, 0, 0.6, 0, -8], [1, 22, -12, 0.4, -0.05, 6]] }),
  // a charged spin: the blade held out to the right while the body turns about once (heroCombat.ts)
  move({ name: 'spin', dur: 0.62, hit: [0.06, 0.86], chain: 1, lunge: 0, lungeAt: [0, 0], damage: 2, reach: 2.7, arc: 180, knock: 6.5, heavy: true,
    keys: [[0, 118, -10, 0.42, -0.04, 38], [0.12, 96, 4, 0.52, 0, 22], [0.86, 96, 4, 0.52, 0, 22], [1, 70, -18, 0.42, -0.04, 18]] }),
  // in the air: up over the head and down through the landing
  move({ name: 'jump', dur: 0.58, hit: [0.4, 0.86], chain: 1, lunge: 0, lungeAt: [0, 0], damage: 2, reach: 2.4, arc: 52, knock: 5, heavy: true,
    keys: [[0, 30, 40, 0.34, 0.06, 6], [0.34, 12, 118, 0.38, 0.14, 4], [0.58, 2, 4, 0.56, 0, 0], [0.76, 0, -56, 0.5, -0.1, 0], [1, 6, -50, 0.42, -0.1, 2]] }),
  // a sprinting slash
  move({ name: 'dash', dur: 0.46, hit: [0.2, 0.62], chain: 1, lunge: 7.5, lungeAt: [0.04, 0.56], damage: 1.3, reach: 2.4, arc: 80, knock: 4.5, heavy: false,
    keys: [[0, 80, 4, 0.34, 0, 30], [0.18, 118, 8, 0.38, 0.03, 46], [0.42, 10, -2, 0.55, 0, 0], [0.6, -90, -8, 0.5, -0.02, -40], [1, -70, -20, 0.4, -0.05, -25]] }),
  // a Flurry Rush: fast slashes, one way then the other
  move({ name: 'flurry', dur: 0.19, hit: [0.25, 0.75], chain: 0, lunge: 0, lungeAt: [0, 0], damage: 1, reach: 3.2, arc: 120, knock: 0.5, heavy: false,
    keys: [[0, 100, 8, 0.4, 0.03, 40], [0.5, 4, 0, 0.53, 0, 0], [1, -90, -6, 0.48, -0.02, -38]] }),
  move({ name: 'flurry', dur: 0.19, hit: [0.25, 0.75], chain: 0, lunge: 0, lungeAt: [0, 0], damage: 1, reach: 3.2, arc: 120, knock: 0.5, heavy: false,
    keys: [[0, -90, -6, 0.48, -0.02, -38], [0.5, -4, 4, 0.53, 0, 0], [1, 96, 10, 0.44, 0.03, 40]] }),
  // a throw: wound up behind the head (held at u = THROW_HOLD while the key is), released over the top
  move({ name: 'throw', dur: 0.62, hit: [1, 1], chain: 1, lunge: 0, lungeAt: [0, 0], damage: 0, reach: 0, arc: 0, knock: 0, heavy: false,
    keys: [[0, 40, 30, 0.3, 0.04, 14], [0.36, 150, 116, 0.32, 0.15, 48], [0.56, 0, 32, 0.52, 0.05, -12], [0.76, -12, -40, 0.5, -0.05, -32], [1, -12, -40, 0.45, -0.05, -20]] }),
  // a Bokoblin's overhead smash: raised high (the tell), held a beat, brought down
  move({ name: 'smash', dur: 1.45, hit: [0.62, 0.76], chain: 1, lunge: 2.4, lungeAt: [0.6, 0.72], damage: 1, reach: 2.35, arc: 48, knock: 6, heavy: false,
    keys: [[0, 34, -40, 0.34, -0.08, 8], [0.46, 26, 124, 0.34, 0.16, 24], [0.58, 22, 120, 0.34, 0.16, 24], [0.68, 2, 4, 0.56, 0, 0], [0.76, 0, -50, 0.52, -0.1, 0], [1, 10, -46, 0.4, -0.1, 6]] }),
  // … and its sideways sweep
  move({ name: 'sweep', dur: 1.3, hit: [0.6, 0.76], chain: 1, lunge: 2, lungeAt: [0.58, 0.72], damage: 1, reach: 2.35, arc: 84, knock: 5, heavy: false,
    keys: [[0, 34, -40, 0.34, -0.08, 8], [0.44, 126, 14, 0.38, 0.05, 50], [0.56, 120, 12, 0.38, 0.05, 50], [0.66, 2, 0, 0.55, 0, 0], [0.76, -88, -8, 0.52, 0, -44], [1, -60, -22, 0.4, -0.05, -24]] }),
];

/** Where a held throw waits for the key's release. */
export const THROW_HOLD = 0.36;
/** Where the throw lets go. */
export const THROW_RELEASE = 0.52;

/** Channels of a swing sample: blade yaw, pitch (rad), reach, lift (m), twist (rad), then yaw' and pitch' (rad per u). */
export const SWING_CHANNELS = 7;

/**
 * Sample move `m` at u (clamped to [0, 1]) into `out` (SWING_CHANNELS
 * numbers): the angles in radians, the last two the blade's angular velocity
 * along u (its edge leads the way it travels).
 */
export function swingAt(m: Move, u: number, out: Float64Array): Float64Array {
  const K = m.keys;
  const n = K.length;
  const t0 = Math.min(1, Math.max(0, u));
  let i = 0;
  while (i < n - 2 && t0 > K[i + 1][0]) i++;
  const a = K[i], b = K[i + 1];
  const h = Math.max(b[0] - a[0], 1e-6);
  const t = Math.min(1, Math.max(0, (t0 - a[0]) / h));
  const t2 = t * t, t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
  const d00 = (6 * t2 - 6 * t) / h, d10 = 3 * t2 - 4 * t + 1, d01 = (6 * t - 6 * t2) / h, d11 = 3 * t2 - 2 * t;
  for (let c = 1; c <= 5; c++) {
    // tangents: the neighbours' difference over their span (none at the ends)
    const ma = i > 0 ? (b[c] - K[i - 1][c]) / (b[0] - K[i - 1][0]) : 0;
    const mb = i + 2 < n ? (K[i + 2][c] - a[c]) / (K[i + 2][0] - a[0]) : 0;
    const v = h00 * a[c] + h10 * h * ma + h01 * b[c] + h11 * h * mb;
    const k = c === 1 || c === 2 || c === 5 ? D : 1;
    out[c - 1] = v * k;
    if (c <= 2) out[4 + c] = (d00 * a[c] + d10 * h * ma + d01 * b[c] + d11 * h * mb) * D;
  }
  return out;
}

/** The blade's direction for yaw and pitch (hero frame: x right, y up, z ahead). */
export function bladeDir(yaw: number, pitch: number, out: Float64Array, o = 0): void {
  const cp = Math.cos(pitch);
  out[o] = Math.sin(yaw) * cp;
  out[o + 1] = Math.sin(pitch);
  out[o + 2] = Math.cos(yaw) * cp;
}
