/**
 * The colossi, after Shadow of the Colossus's Argus and Valus: Herod, a 30 m stone golem (COLOSSUS) roaming the open
 * land, one in many COLOSSUS.cell cells. Its model (public/models/colossus.bin, scripts/colossus/build.py) has a
 * skeleton and no clips: every pose is made here.
 * ---------------------------------------------------------------------------
 * Its mind:
 *   roam / idle      at peace round its spot, walking
 *   hunt             it noticed a hero (closer than COLOSSUS.notice): walking to the nearest
 *   smash            both fists raised high and brought down before it (a shockwave round them)
 *   stomp            a hero at one of its feet: that foot raised over the hero and brought down on it
 *   fall             it leans back, then throws its whole body down forward (Argus's fall)
 *   down / rise      felled by its fall: lying face down, its core glowing, then pushing itself up
 *   recover          a breath between attacks, walking on toward the hero
 *   die / vanish     its health gone: it falls, lies, and goes in a dissolve; slain, it stays gone (by id)
 * With a hero on it, it does not attack: now and then it shakes (`shake`), whipping its body to throw the hero off.
 * Walking is by its feet: each planted where it fell on the ground, the body carried over them, a foot stepping
 * when it falls behind (or turned) and the other is down. The legs reach their feet and the arms their fists by
 * two-bone IK (ik), so the feet never slide and stand on the slope. A step that lands shakes the ground (`quakes`)
 * and crushes the hero under it. Its body follows where the mind wants it on critically damped springs (a rate per
 * phase: slow wind-ups, a fast strike), so it moves with weight and never jerks.
 *
 * Its body is its collision mesh skinned as it is drawn (body.ts, `bodyOf`): what the hero climbs, walks into and
 * strikes (the combat world's beasts ask it: `meets`, `near`), and its eye a sphere just proud of its face; only the
 * eye is hurt (`hurt`). The hero climbs it (`grab`, `climb`): held to a point on a triangle of the mesh, so it rides
 * every move; it crawls over the stone (stepping along it and back onto the nearest surface turned the same way),
 * stands on what is level enough (resting), stabs the eye from close by, and lets go with Jump. Climbing costs
 * stamina, a shake much more; run dry and it is thrown off (`thrown`).
 *
 * Multiplayer: one player (the world's first: `authority`) thinks for every colossus near any player and sends each
 * one's state (`encode`, COLOSSUS_MSG.state) ten times a second; the others run theirs as puppets (`receive`): the
 * same timeline, its choices taken from the state, so each settles the blows on its own hero, and sends what it does
 * to the eye to the thinker (`sent`, COLOSSUS_MSG.hit), whose health is the one that counts.
 */

import { COLOSSUS, HERO, PLAYER } from '../constants';
import { hash3, hashU } from '../noise';
import { surfaceAt } from '../terrainMath';
import { rng } from '../skyworld/kit/mesh';
import { MAT, type HorseModel } from '../horse/model';
import { ColossusBody, createBodyHit } from './body';
import { climbClearance, type HeroState } from '../hero/locomotion';

type Mind = 'roam' | 'idle' | 'hunt' | 'smash' | 'stomp' | 'fall' | 'down' | 'rise' | 'recover' | 'die' | 'vanish';
const MINDS: readonly Mind[] = ['roam', 'idle', 'hunt', 'smash', 'stomp', 'fall', 'down', 'rise', 'recover', 'die', 'vanish'];

/** A foot: where it stands on the ground (world), and its step: from, to, how far along (−1: planted). */
type Foot = { x: number; y: number; z: number; fx: number; fy: number; fz: number; tx: number; ty: number; tz: number; u: number };

// Its stance (where the mind wants the body, and where it is): hips in its frame (m, rad), the fists' targets.
const DROP = 0, SWAY = 1, FWD = 2, ROLL = 3, LEAN = 4, CHEST = 5, HEAD_YAW = 6, HEAD_PITCH = 7, CLAV = 8, FIST = 9;
const STANCE = 15;

export type Colossus = {
  id: number;
  x: number; y: number; z: number; yaw: number;
  hx: number; hz: number;      // its spot: it roams round it
  hp: number;
  hitSwing: number;            // the hero's swing that last struck it (one blow a swing)
  rockHit: object | null;      // the rock that last struck it
  /** Its eye (world), and what a blow last met (EYE_HIT: the eye; 0: stone). */
  eye: Float64Array; hitCap: number;
  /** For the combat world: where a segment first meets it (−1: misses); its surface's distance from a point (into P). */
  meets: (ax: number, ay: number, az: number, bx: number, by: number, bz: number, pad: number) => number;
  near: (px: number, py: number, pz: number, P: Float64Array) => number;
  body: ColossusBody | null; bodyTick: number; // its collision mesh, and the tick it was last skinned
  mind: Mind;
  t: number;                   // s in this mind
  timer: number;               // s left (idle, recover)
  tx: number; tz: number;      // where it walks to (its roam's goal, the hero hunted)
  speed: number;
  turning: number;             // how fast it turns now (rad/s), eased
  feet: [Foot, Foot];
  pose: Float64Array; vel: Float64Array; want: Float64Array; rate: number;
  topple: number;              // rad it has fallen forward about its toes
  side: number;                // the foot that stomps (0 left, 1 right)
  struck: boolean;             // this attack's blow has landed
  aimX: number; aimZ: number;  // where the attack falls (world)
  flash: number;               // a blow's flash
  glow: number;                // its core showing 0..1
  shake: number;               // s left of a shake (a hero on it)
  ridden: number;              // s a hero has been on it…
  shakeAt: number;             // … when the next shake comes
  puppet: boolean;             // another player thinks for it (multiplayer)
  seen: number;                // s since its last state (a puppet)
  rand: () => number;
};

/** One frame of colossi for the renderer (gpu/horses.ts SkinnedGpu): COLOSSUS_STRIDE floats each, camera-relative. */
export type ColossusFrame = { data: Float32Array; u32: Uint32Array; count: number; fading: number };

export const COLOSSUS_URL = '/models/colossus.bin';
export const COLOSSUS_TEXTURE_URL = '/models/colossus.jpg';
export const COLOSSI_DRAWN = 3;
export const COLOSSUS_JOINTS = 17;
/** Floats a colossus: its id, core glow, vanish, flash; then its joints' matrices (camera-relative world, rows). */
export const COLOSSUS_STRIDE = 4 + COLOSSUS_JOINTS * MAT;
/** Its combat messages between players (net/session.ts; combat/pvp.ts has 1–3): its state, a blow on its eye. */
export const COLOSSUS_MSG = { state: 4, hit: 5 } as const;

export function createColossusFrame(): ColossusFrame {
  const data = new Float32Array(COLOSSI_DRAWN * COLOSSUS_STRIDE);
  return { data, u32: new Uint32Array(data.buffer), count: 0, fading: 0 };
}

/** A blow on the hero: quarters, and the fling (m/s). */
type ColossusHit = { q: number; vx: number; vy: number; vz: number };
/** Something heavy meeting the ground: where, and how big (m: the dust's reach). */
type Quake = { x: number; y: number; z: number; size: number };
/** A stab of the hero clinging to it: where, and the damage it took (0: on stone). */
type Stab = { x: number; y: number; z: number; took: number };
/** Where a player is (for whom it hunts, who is on it). */
type Body = { x: number; y: number; z: number };

const FLASH = 0.25;
const W_REST = 2.5;            // its springs' rate at ease (rad/s)…
const W_HEAD = 2.2;            // … the head's

// --- the colossus's spots ----------------------------------------------------------------------------------------

const S2: [number, number] = [0, 0];
const ground = (x: number, z: number, seed: number) => surfaceAt(x, z, seed, S2)[0];

type Spot = { id: number; x: number; z: number };

/** The spot of cell (cx, cz), or null: none roams there (it is wet, or too rough). */
export function spotOf(cx: number, cz: number, seed: number): Spot | null {
  const C = COLOSSUS.cell;
  const id = hash3(cx, hashU(cz ^ 0xc0105), seed) >>> 0;
  const r = rng(id);
  if (r() > COLOSSUS.chance) return null;
  // (a few places in the cell, the first dry and level enough: every biome has its own, its mountains too)
  for (let tries = 0; tries < 4; tries++) {
    const x = (cx + 0.15 + 0.7 * r()) * C, z = (cz + 0.15 + 0.7 * r()) * C;
    surfaceAt(x, z, seed, S2);
    const g = S2[0];
    if (g < S2[1] + 2) continue;
    let lo = g, hi = g;
    for (let k = 0; k < 4; k++) {
      const h = ground(x + 24 * Math.cos(k * 1.571), z + 24 * Math.sin(k * 1.571), seed);
      lo = Math.min(lo, h); hi = Math.max(hi, h);
    }
    if (hi - lo <= COLOSSUS.rough) return { id, x, z };
  }
  return null;
}

// each cell's spot once worked out, for the seed it was for
const spots = new Map<string, Spot | null>();
let spotsSeed = NaN;

function spotAt(cx: number, cz: number, seed: number): Spot | null {
  if (seed !== spotsSeed || spots.size > 4096) { spots.clear(); spotsSeed = seed; }
  const key = `${cx},${cz}`;
  let s = spots.get(key);
  if (s === undefined) spots.set(key, s = spotOf(cx, cz, seed));
  return s;
}

// --- small vector and matrix sums (row-major 3×3) ---------------------------------------------------------------

const ease = (a: number, b: number, rate: number, dt: number) => a + (b - a) * (1 - Math.exp(-rate * dt));
const smooth = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
const wrap = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

/** x pulled toward `to` by a critically damped spring of rate w (its speed in v[i]): no jerk when the goal moves. */
function spring(x: number, to: number, v: Float64Array, i: number, w: number, dt: number): number {
  // (substeps keep a stiff spring stable over a long tick)
  const n = Math.ceil((w * dt) / 0.5);
  const h = dt / n;
  for (let k = 0; k < n; k++) {
    v[i] += (w * w * (to - x) - 2 * w * v[i]) * h;
    x += v[i] * h;
  }
  return x;
}

/** out = A · B (3×3 at offsets). */
function mul(A: Float64Array, a: number, B: Float64Array, b: number, out: Float64Array, o: number): void {
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    out[o + 3 * i + j] = A[a + 3 * i] * B[b + j] + A[a + 3 * i + 1] * B[b + 3 + j] + A[a + 3 * i + 2] * B[b + 6 + j];
  }
}

/** out = rotation by yaw about +Y, then pitch about +X (nose down positive), then roll about +Z. */
function euler(yaw: number, pitch: number, roll: number, out: Float64Array, o: number): void {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
  // Ry · Rx · Rz
  out[o] = cy * cr + sy * sp * sr; out[o + 1] = -cy * sr + sy * sp * cr; out[o + 2] = sy * cp;
  out[o + 3] = cp * sr; out[o + 4] = cp * cr; out[o + 5] = -sp;
  out[o + 6] = -sy * cr + cy * sp * sr; out[o + 7] = sy * sr + cy * sp * cr; out[o + 8] = cy * cp;
}

const F0 = new Float64Array(9), F1 = new Float64Array(9);
/** Orthonormal frame (columns: d, the pole's part across d, their cross) into F. */
function frame(dx: number, dy: number, dz: number, px: number, py: number, pz: number, F: Float64Array): void {
  let l = Math.hypot(dx, dy, dz) || 1;
  dx /= l; dy /= l; dz /= l;
  const k = px * dx + py * dy + pz * dz;
  px -= k * dx; py -= k * dy; pz -= k * dz;
  l = Math.hypot(px, py, pz) || 1;
  px /= l; py /= l; pz /= l;
  F[0] = dx; F[3] = dy; F[6] = dz;
  F[1] = px; F[4] = py; F[7] = pz;
  F[2] = dy * pz - dz * py; F[5] = dz * px - dx * pz; F[8] = dx * py - dy * px;
}

/** The rotation taking direction d0 (its pole p0) onto d1 (its pole p1), into out[o …]. */
function aim(d0: ArrayLike<number>, p: ArrayLike<number>, d1x: number, d1y: number, d1z: number, q: ArrayLike<number>, out: Float64Array, o: number): void {
  frame(d0[0], d0[1], d0[2], p[0], p[1], p[2], F0);
  frame(d1x, d1y, d1z, q[0], q[1], q[2], F1);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    out[o + 3 * i + j] = F1[3 * i] * F0[3 * j] + F1[3 * i + 1] * F0[3 * j + 1] + F1[3 * i + 2] * F0[3 * j + 2];
  }
}

/**
 * Two-bone IK: the middle joint of a limb from `a` (bones la, lb) reaching for `t`, bent toward `pole`, into out.
 * Out of reach, the limb points straight at t.
 */
export function ik(a: ArrayLike<number>, t: ArrayLike<number>, la: number, lb: number, pole: ArrayLike<number>, out: Float64Array): void {
  let ux = t[0] - a[0], uy = t[1] - a[1], uz = t[2] - a[2];
  const d0 = Math.hypot(ux, uy, uz) || 1e-6;
  ux /= d0; uy /= d0; uz /= d0;
  const d = Math.min(Math.max(d0, Math.abs(la - lb) + 1e-3), la + lb - 1e-3);
  const c = Math.min(1, Math.max(-1, (la * la + d * d - lb * lb) / (2 * la * d))), s = Math.sqrt(1 - c * c);
  const k = pole[0] * ux + pole[1] * uy + pole[2] * uz;
  let vx = pole[0] - k * ux, vy = pole[1] - k * uy, vz = pole[2] - k * uz;
  const l = Math.hypot(vx, vy, vz) || 1;
  vx /= l; vy /= l; vz /= l;
  out[0] = a[0] + la * (ux * c + vx * s);
  out[1] = a[1] + la * (uy * c + vy * s);
  out[2] = a[2] + la * (uz * c + vz * s);
}

// --- the rig ------------------------------------------------------------------------------------------------------

/** The joints by name (the model's order), the tips (fists, toes, crown), the bone lengths the IK needs. */
type Rig = {
  rest: Float64Array;          // 3 a joint, its frame
  parent: Int32Array;
  j: Record<string, number>;
  hand: [Float64Array, Float64Array]; toe: [Float64Array, Float64Array]; crown: Float64Array;
  arm: [number, number]; leg: [number, number]; // upper, lower (to the fist's tip; to the ankle)
  ankleH: number;              // the ankle over the ground
};

const NAMES = ['hips', 'chest', 'head', 'clavicle_l', 'shoulder_l', 'elbow_l', 'wrist_l', 'hip_l', 'knee_l', 'ankle_l',
  'clavicle_r', 'shoulder_r', 'elbow_r', 'wrist_r', 'hip_r', 'knee_r', 'ankle_r'];
/** The eye in its face (bind pose), a little proud of it so a shot meets it before the head. */
const EYE: readonly number[] = [0.6, 16.3, 7.0];
const EYE_R = 1.3;
/** A blow on its eye (Colossus.hitCap). */
export const EYE_HIT = 1;

function rigOf(m: HorseModel): Rig {
  if (m.joints !== COLOSSUS_JOINTS || NAMES.some((n, k) => m.names[k] !== n)) throw new Error('colossus.bin: not the expected skeleton');
  const tips = m.tips;
  if (!tips) throw new Error('colossus.bin: no tips');
  const rest = new Float64Array(m.rest.flat());
  const j = Object.fromEntries(NAMES.map((n, k) => [n, k]));
  const v = (k: number) => rest.subarray(3 * k, 3 * k + 3);
  const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const hand: [Float64Array, Float64Array] = [new Float64Array(tips.hand_l), new Float64Array(tips.hand_r)];
  const toe: [Float64Array, Float64Array] = [new Float64Array(tips.toe_l), new Float64Array(tips.toe_r)];
  return {
    rest, parent: new Int32Array(m.parents), j, hand, toe, crown: new Float64Array(tips.crown),
    arm: [dist(v(j.shoulder_l), v(j.elbow_l)), dist(v(j.elbow_l), hand[0])],
    leg: [dist(v(j.hip_l), v(j.knee_l)), dist(v(j.knee_l), v(j.ankle_l))],
    ankleH: rest[3 * j.ankle_l + 1],
  };
}

// --- the colossi ----------------------------------------------------------------------------------------------------

const ROT = new Float64Array(COLOSSUS_JOINTS * 9), POS = new Float64Array(COLOSSUS_JOINTS * 3);
const L9 = new Float64Array(9), T9 = new Float64Array(9);
const V3 = new Float64Array(3), W3 = new Float64Array(3), M3 = new Float64Array(3), A3 = new Float64Array(3);
const D0 = new Float64Array(3), POLE = new Float64Array(3);
const P0 = new Float64Array(3), N0 = new Float64Array(3);
const JM = new Float64Array(COLOSSUS_JOINTS * 12);
const HIT = createBodyHit();

/** A hero holding on: to triangle `tri` of colossus C's body at (u, v) on it; standing on top of it. */
type Cling = { C: Colossus; tri: number; u: number; v: number; top: boolean };

export class Colossi {
  list: Colossus[] = [];
  seed = NaN;
  /** This player thinks for the colossi (alone, or the world's first player); else they are puppets of its states. */
  authority = true;
  /** This tick's blow on the hero (null: none), and what met the ground. */
  hit: ColossusHit | null = null;
  quakes: Quake[] = [];
  /** Slain this tick (for their motes), and slain for good (by id). */
  vanished: Colossus[] = [];
  /** Just noticed a hero this tick. */
  roused: Colossus | null = null;
  /** The hero holding on to a colossus (null: none); this tick its stab, and its throw off (m/s; null: none). */
  cling: Cling | null = null;
  stab: Stab | null = null;
  thrown: [number, number, number] | null = null;
  /** Climbing its walls (off for now: COLOSSUS.climbWalls); without it a hero only lands and stands on top of it. */
  climbWalls: boolean = COLOSSUS.climbWalls;
  /** Blows on the eye a puppet took, for its thinker (multiplayer): COLOSSUS_MSG.hit messages, drained by the engine. */
  sent: number[][] = [];
  private slain = new Set<number>();
  private rig: Rig | null = null;
  private model: HorseModel | null = null;
  private coll: HorseModel['coll'] | null = null;
  private tick = 0;
  private spawnIn = 0;
  private grabIn = 0;
  private feetAt = new Float64Array([0, 0, 0, -9]); // the hero's feet at the last grab check, and that tick
  private stabIn = 0;
  private pose = new Float32Array(COLOSSUS_JOINTS * MAT);
  private hitOut: ColossusHit = { q: 0, vx: 0, vy: 0, vz: 0 };
  private stabOut: Stab = { x: 0, y: 0, z: 0, took: 0 };
  private quakePool: Quake[] = [];
  private others: readonly Body[] = [];

  setModel(m: HorseModel): void {
    this.rig = rigOf(m);
    this.model = m;
    this.coll = m.coll ?? null;
    for (const C of this.list) this.restPose(C);
  }

  reset(seed: number): void {
    this.seed = seed;
    this.list.length = 0;
    this.slain.clear();
    this.spawnIn = 0;
    this.cling = null;
  }

  /** A colossus at (x, z), facing `yaw` (debugging and tests: one need not wait for a spot). */
  spawn(id: number, x: number, z: number, yaw = 0): Colossus {
    const y = ground(x, z, this.seed);
    const foot = (): Foot => ({ x, y, z, fx: x, fy: y, fz: z, tx: x, ty: y, tz: z, u: -1 });
    const C: Colossus = {
      id, x, y, z, yaw, hx: x, hz: z, hp: COLOSSUS.hp, hitSwing: -1, rockHit: null,
      eye: new Float64Array(3), hitCap: -1, body: null, bodyTick: -1,
      meets: (ax, ay, az, bx, by, bz, pad) => this.meets(C, ax, ay, az, bx, by, bz, pad),
      near: (px, py, pz, P) => this.near(C, px, py, pz, P),
      mind: 'idle', t: 0, timer: 2, tx: x, tz: z, speed: 0, turning: 0,
      feet: [foot(), foot()], pose: new Float64Array(STANCE), vel: new Float64Array(STANCE), want: new Float64Array(STANCE), rate: W_REST,
      topple: 0, side: 0, struck: false, aimX: x, aimZ: z, flash: 0, glow: 0,
      shake: 0, ridden: 0, shakeAt: COLOSSUS.shakeAfter, puppet: !this.authority, seen: 0, rand: rng(id ^ 0x5eed),
    };
    C.pose[DROP] = C.want[DROP] = 1.6;
    this.list.push(C);
    this.restPose(C);
    return C;
  }

  /** Its feet where they stand at rest and its fists hanging (once the rig is known). */
  private restPose(C: Colossus): void {
    const R = this.rig;
    if (!R) return;
    for (let s = 0; s < 2; s++) {
      const a = R.rest.subarray(3 * (s ? R.j.ankle_r : R.j.ankle_l));
      this.toWorld(C, a[0], 0, a[2], V3);
      const F = C.feet[s];
      F.x = F.fx = F.tx = V3[0]; F.z = F.fz = F.tz = V3[2];
      F.y = F.fy = F.ty = ground(F.x, F.z, this.seed);
      F.u = -1;
    }
    C.y = 0.5 * (C.feet[0].y + C.feet[1].y);
    this.armsDown(C.pose, 0);
    this.armsDown(C.want, 0);
    this.posed(C);
  }

  private toWorld(C: Colossus, x: number, y: number, z: number, out: Float64Array): void {
    const c = Math.cos(C.yaw), s = Math.sin(C.yaw);
    out[0] = C.x + c * x + s * z; out[1] = C.y + y; out[2] = C.z - s * x + c * z;
  }

  private toLocal(C: Colossus, x: number, y: number, z: number, out: Float64Array): void {
    const c = Math.cos(C.yaw), s = Math.sin(C.yaw), dx = x - C.x, dz = z - C.z;
    out[0] = c * dx - s * dz; out[1] = y - C.y; out[2] = s * dx + c * dz;
  }

  /** The fists hanging by its sides, a little off the ground, swung by `swing` (−1..1: the left forward). */
  private armsDown(S: Float64Array, swing: number): void {
    const R = this.rig!;
    for (let s = 0; s < 2; s++) {
      const h = R.hand[s];
      S[FIST + 3 * s] = h[0] * 0.95; S[FIST + 3 * s + 1] = 1.2; S[FIST + 3 * s + 2] = h[2] + 3.2 * (s ? -swing : swing);
    }
  }

  private fists(S: Float64Array, kx: number, y: number, z: number): void {
    const R = this.rig!;
    for (let s = 0; s < 2; s++) { S[FIST + 3 * s] = R.hand[s][0] * kx; S[FIST + 3 * s + 1] = y; S[FIST + 3 * s + 2] = z; }
  }

  /**
   * The hero's blow of `dmg` on it (`hitCap` says where): its body takes it as it is, its eye more (more again while
   * it lies felled). The damage it took. A puppet's goes to its thinker too.
   */
  hurt(C: Colossus, dmg: number): number {
    if (C.hp <= 0) return 0;
    const took = dmg * (C.hitCap === EYE_HIT ? COLOSSUS.eye * (C.mind === 'down' ? COLOSSUS.downed : 1) : COLOSSUS.body);
    if (C.puppet) this.sent.push([COLOSSUS_MSG.hit, C.id, Math.round(took * 100) / 100]);
    this.damage(C, took);
    return took;
  }

  /** `took` off its health (a blow on its eye; from a puppet, a player's: COLOSSUS_MSG.hit). */
  damage(C: Colossus, took: number): void {
    if (C.hp <= 0) return;
    C.hp = Math.max(0, C.hp - took);
    C.flash = FLASH;
    if (C.puppet) return; // (its thinker says what comes of it)
    if (C.hp <= 0) this.become(C, 'die');
    else if (C.mind === 'roam' || C.mind === 'idle') { this.become(C, 'hunt'); this.roused = C; }
  }

  /** Is it fighting a hero (for its life bar)? */
  engaged(C: Colossus): boolean {
    return C.hp > 0 && (C.hp < COLOSSUS.hp || (C.mind !== 'roam' && C.mind !== 'idle'));
  }

  private become(C: Colossus, m: Mind): void {
    if (m === 'die') for (const F of C.feet) if (F.u >= 0) { F.x = F.tx; F.y = F.ty; F.z = F.tz; F.u = -1; }
    C.mind = m;
    C.t = 0;
    C.struck = false;
  }

  /**
   * A tick: colossi come and go round the players, each one's mind, its steps and pose. (hx, hy, hz): this player's
   * hero's feet; `heroDown`: it is felled; `others`: the other players (multiplayer), whom they hunt too.
   */
  step(dt: number, hx: number, hy: number, hz: number, heroDown: boolean, others: readonly Body[] = []): void {
    this.hit = null;
    this.quakes.length = 0;
    this.vanished.length = 0;
    this.roused = null;
    this.others = others;
    this.tick++;
    if (!this.rig || !(this.seed === this.seed)) return;
    if (this.authority && (this.spawnIn -= dt) <= 0) {
      this.spawnIn = 0.5;
      this.populate(hx, hz);
      for (const o of others) this.populate(o.x, o.z);
    }
    for (const C of this.list) {
      if (C.puppet !== !this.authority) C.puppet = !this.authority;
      this.think(C, dt, hx, hy, hz, heroDown);
    }
    for (let k = this.list.length - 1; k >= 0; k--) {
      const C = this.list[k];
      const lost = C.puppet && (C.seen += dt) > 3 && C.mind !== 'die' && C.mind !== 'vanish';
      if ((C.mind === 'vanish' && C.t >= COLOSSUS.vanishFor) || lost) {
        if (!lost) { this.slain.add(C.id); this.vanished.push(C); }
        if (this.cling?.C === C) this.cling = null;
        this.list.splice(k, 1);
      }
    }
    // (despawned with nobody near)
    if (this.authority) {
      for (let k = this.list.length - 1; k >= 0; k--) {
        const C = this.list[k];
        const far = (x: number, z: number) => Math.hypot(C.x - x, C.z - z) > COLOSSUS.despawnRange;
        if (far(hx, hz) && others.every((o) => far(o.x, o.z)) && C.mind !== 'die' && C.mind !== 'vanish') this.list.splice(k, 1);
      }
    }
  }

  private populate(hx: number, hz: number): void {
    if (this.list.length >= COLOSSUS.maxLive) return;
    const Cl = COLOSSUS.cell, R = COLOSSUS.spawnRange;
    let best: Spot | null = null, bd: number = R;
    for (let cz = Math.floor((hz - R) / Cl); cz <= Math.floor((hz + R) / Cl); cz++) {
      for (let cx = Math.floor((hx - R) / Cl); cx <= Math.floor((hx + R) / Cl); cx++) {
        const s = spotAt(cx, cz, this.seed);
        if (!s || this.slain.has(s.id) || this.list.some((C) => C.id === s.id)) continue;
        const d = Math.hypot(s.x - hx, s.z - hz);
        if (d < bd) { bd = d; best = s; }
      }
    }
    if (best) this.spawn(best.id, best.x, best.z, hashU(best.id) / 4294967296 * Math.PI * 2);
  }

  /** The player it goes for: the nearest of this hero and the others (null: none it would go for). */
  private quarry(C: Colossus, hx: number, hy: number, hz: number, heroDown: boolean): Body | null {
    let best: Body | null = heroDown ? null : { x: hx, y: hy, z: hz };
    let bd = best ? Math.hypot(hx - C.x, hz - C.z) : Infinity;
    for (const o of this.others) {
      const d = Math.hypot(o.x - C.x, o.z - C.z);
      if (d < bd) { bd = d; best = o; }
    }
    return best;
  }

  /** Is a hero on it: this one holding on, or another player up on its body? */
  private carries(C: Colossus): boolean {
    if (this.cling?.C === C) return true;
    for (const o of this.others) {
      if (o.y < ground(o.x, o.z, this.seed) + 2.5 || Math.hypot(o.x - C.x, o.z - C.z) > 40) continue;
      const B = this.bodyOf(C);
      if (B && B.nearest(o.x, o.y + 0.9, o.z, 1.6, HIT)) return true;
    }
    return false;
  }

  private think(C: Colossus, dt: number, hx: number, hy: number, hz: number, heroDown: boolean): void {
    const W = C.want, own = !C.puppet;
    C.t += dt;
    C.flash = Math.max(0, C.flash - dt);
    // whom it goes for: its thinker picks (and says where: tx, tz); a puppet goes where it was told
    const Q = own ? this.quarry(C, hx, hy, hz, heroDown) : null;
    const qx = Q ? Q.x : C.tx, qz = Q ? Q.z : C.tz;
    const d = Math.hypot(qx - C.x, qz - C.z);
    const toQ = Math.atan2(qx - C.x, qz - C.z);
    const off = wrap(toQ - C.yaw);
    const rage = C.hp < COLOSSUS.hp * COLOSSUS.rage ? 0.75 : 1;
    let speed = 0, turnTo = NaN, turnRate = COLOSSUS.turn;
    // (at ease unless an attack says otherwise)
    W[DROP] = 1.6; W[SWAY] = 0; W[FWD] = 0; W[ROLL] = 0; W[LEAN] = 0; W[CHEST] = 0; W[CLAV] = 0; W[HEAD_PITCH] = 0.05; W[HEAD_YAW] = 0;
    let rate = W_REST;
    // a hero on it: no attacks; now and then it shakes (its thinker decides when)
    const carried = own ? this.carries(C) : C.shake > 0;
    if (own) {
      if (carried && C.topple === 0 && (C.mind === 'hunt' || C.mind === 'recover' || C.mind === 'roam' || C.mind === 'idle')) {
        C.ridden += dt;
        if (C.shake <= 0 && C.ridden > C.shakeAt) {
          C.shake = COLOSSUS.shakeFor;
          C.shakeAt = C.ridden + COLOSSUS.shakeFor + COLOSSUS.shakeEvery[0] + (COLOSSUS.shakeEvery[1] - COLOSSUS.shakeEvery[0]) * C.rand() * rage;
        }
      } else if (!carried) { C.ridden = 0; C.shakeAt = COLOSSUS.shakeAfter; }
    }
    C.shake = Math.max(0, C.shake - dt);
    switch (C.mind) {
      case 'idle':
      case 'roam': {
        if (own && Q && d < COLOSSUS.notice) { this.become(C, 'hunt'); this.roused = C; break; }
        if (C.mind === 'idle') {
          if (own && (C.timer -= dt) <= 0) {
            const a = C.rand() * Math.PI * 2, r = COLOSSUS.roam * Math.sqrt(C.rand());
            C.tx = C.hx + Math.cos(a) * r; C.tz = C.hz + Math.sin(a) * r;
            this.become(C, 'roam');
          }
        } else {
          const dd = Math.hypot(C.tx - C.x, C.tz - C.z);
          turnTo = Math.atan2(C.tx - C.x, C.tz - C.z);
          speed = this.gait(wrap(turnTo - C.yaw)) * 0.85;
          if (own && (dd < 8 || C.t > 90)) { C.timer = 1.5 + 2 * C.rand(); this.become(C, 'idle'); }
        }
        break;
      }
      case 'hunt':
      case 'recover': {
        if (own && (!Q || d > COLOSSUS.forget)) { C.timer = 3; this.become(C, 'idle'); break; }
        if (own) { C.tx = qx; C.tz = qz; } // (where a puppet walks)
        turnTo = toQ;
        W[HEAD_YAW] = Math.max(-0.5, Math.min(0.5, off));
        W[HEAD_PITCH] = Math.max(-0.4, Math.min(0.5, Math.atan2(C.y + 18 - (Q ? Q.y : hy), d) * 0.6));
        // (on toward it until it is in reach before it; on through a recovery, slower)
        speed = d > COLOSSUS.reach * 0.55 ? this.gait(off) * (C.mind === 'recover' ? 0.6 : 1) : 0;
        if (!own) break;
        if (C.mind === 'recover') {
          if ((C.timer -= dt) <= 0) this.become(C, 'hunt');
          break;
        }
        // (attacks start with both feet down, never with a hero on it)
        if (!carried && C.feet[0].u < 0 && C.feet[1].u < 0) this.choose(C, Q!, off);
        break;
      }
      case 'smash': {
        const wu = 1.75 * rage, hold = 0.35 * rage, hit = 0.4, back = 1.9 * rage;
        if (C.t < wu + hold) {
          // up: both fists high over its head, leaning back; still turning to the hero while it rises
          if (C.t < wu) { turnTo = Math.atan2(C.aimX - C.x, C.aimZ - C.z); turnRate = COLOSSUS.turn * 1.6; }
          W[DROP] = 0.5; W[LEAN] = -0.08; W[CHEST] = -0.2; W[CLAV] = 0.3; W[HEAD_PITCH] = -0.15;
          this.fists(W, 0.55, 31, 0.5);
          rate = 3.2;
        } else if (C.t < wu + hold + hit + back) {
          // down: the fists to the ground before it, its whole weight behind them
          W[DROP] = 2.8; W[LEAN] = 0.28; W[CHEST] = 0.5; W[CLAV] = -0.05; W[HEAD_PITCH] = 0.3;
          this.fists(W, 0.42, 1.0, 9);
          rate = C.t < wu + hold + hit ? 19 : 6;
          if (!C.struck && C.t > wu + hold + hit * 0.85) {
            C.struck = true;
            for (let s = 0; s < 2; s++) {
              this.toWorld(C, W[FIST + 3 * s], 0, W[FIST + 3 * s + 2], V3);
              V3[1] = ground(V3[0], V3[2], this.seed);
              this.quake(V3[0], V3[1], V3[2], 10);
            }
            this.toWorld(C, 0, 0, 9, V3);
            this.blow(V3[0], V3[2], NaN, NaN, hx, hy, hz, COLOSSUS.smash.inner + 4, COLOSSUS.smash.outer + 4, COLOSSUS.smash.hit, COLOSSUS.smash.ring);
          }
          if (C.t > wu + hold + hit + 0.8) { this.armsDown(W, 0); rate = 2.2; }
        } else if (own) { C.timer = COLOSSUS.cooldown * (rage < 1 ? 0.6 : 1); this.become(C, 'recover'); }
        break;
      }
      case 'stomp': {
        const up = 1.2 * rage, hold = 0.3 * rage, down = 0.25, back = 1.0 * rage;
        const F = C.feet[C.side], sgn = C.side ? 1 : -1;
        W[SWAY] = -sgn * 1.7; W[ROLL] = -sgn * 0.07; W[DROP] = 0.9; W[HEAD_PITCH] = 0.35;
        this.armsDown(W, 0);
        W[FIST + 3 * C.side + 1] = 4; // (the arm on its side out for balance)
        rate = 4;
        if (C.t < up + hold) {
          if (own && C.t < up * 0.7 && Q) { C.aimX = Q.x; C.aimZ = Q.z; }
          const u = smooth(C.t / up);
          this.stompAim(C, V3);
          F.x = F.fx + (V3[0] - F.fx) * u; F.z = F.fz + (V3[2] - F.fz) * u;
          F.y = F.fy + (V3[1] - F.fy) * u + 7.5 * Math.sin(u * Math.PI * 0.5);
        } else if (C.t < up + hold + down) {
          const u = (C.t - up - hold) / down;
          this.stompAim(C, V3);
          F.x = V3[0]; F.z = V3[2];
          F.y = V3[1] + 7.5 * (1 - u * u);
        } else {
          if (!C.struck) {
            C.struck = true;
            this.stompAim(C, V3);
            F.x = F.tx = V3[0]; F.z = F.tz = V3[2]; F.y = F.ty = V3[1];
            F.u = -1;
            this.quake(F.x, F.y, F.z, 9);
            this.blow(F.x, F.z, NaN, NaN, hx, hy, hz, COLOSSUS.stomp.inner, COLOSSUS.stomp.outer, COLOSSUS.stomp.hit, COLOSSUS.stomp.ring);
          }
          if (own && C.t > up + hold + down + back) { C.timer = COLOSSUS.cooldown * 0.6 * (rage < 1 ? 0.6 : 1); this.become(C, 'recover'); }
        }
        break;
      }
      case 'fall':
      case 'die': {
        const lean = C.mind === 'die' ? 0.6 : 1.45 * rage, drop = 1.2;
        if (C.t < lean) {
          // rearing back, arms up and forward, then over it goes
          if (C.mind === 'fall') { turnTo = Math.atan2(C.aimX - C.x, C.aimZ - C.z); turnRate = COLOSSUS.turn * 1.3; }
          W[DROP] = 0.4; W[CHEST] = -0.28; W[LEAN] = -0.1; W[CLAV] = 0.25; W[HEAD_PITCH] = -0.2;
          this.fists(W, 0.9, 25, 9);
          rate = 3;
          if (C.t > lean * 0.5 && ((C.t - dt) * 7 | 0) !== (C.t * 7 | 0)) this.rumble(C, 0.35);
        } else {
          W[DROP] = 0.8; W[CHEST] = 0.1; W[CLAV] = 0.25; W[HEAD_PITCH] = 0.1;
          this.fists(W, 0.9, 25, 9);
          const u = Math.min(1, (C.t - lean) / drop);
          C.topple = TOPPLE * u * u; // (falling faster and faster)
          if (u >= 1 && !C.struck) {
            C.struck = true;
            this.toWorld(C, 0, 0, TOE_Z, V3);
            this.toWorld(C, 0, 0, TOE_Z + 26, W3);
            for (let k = 0; k <= 4; k++) {
              const x = V3[0] + (W3[0] - V3[0]) * k / 4, z = V3[2] + (W3[2] - V3[2]) * k / 4;
              this.quake(x, ground(x, z, this.seed), z, 9);
            }
            // (slain, its fall is harmless: the kill is not punished)
            if (C.mind === 'fall') this.blow(V3[0], V3[2], W3[0], W3[2], hx, hy, hz, COLOSSUS.fall.inner, COLOSSUS.fall.outer, COLOSSUS.fall.hit, COLOSSUS.fall.ring);
          }
          if (u >= 1 && own) {
            if (C.mind === 'die') { if (C.t > lean + drop + COLOSSUS.dieFor) this.become(C, 'vanish'); }
            else this.become(C, 'down');
          }
        }
        break;
      }
      case 'down':
      case 'vanish': {
        W[DROP] = 0.8; W[CHEST] = 0.1; W[CLAV] = 0.25;
        this.fists(W, 0.9, 25, 9);
        C.topple = TOPPLE;
        if (own && C.mind === 'down' && C.t > COLOSSUS.downFor) this.become(C, 'rise');
        break;
      }
      case 'rise': {
        const u = Math.min(1, C.t / 3);
        C.topple = TOPPLE * (1 - smooth(u));
        W[DROP] = 1.6; W[CHEST] = 0.3 * (1 - u); W[CLAV] = 0.2;
        this.armsDown(W, 0);
        rate = 1.8;
        if (u >= 1) { C.topple = 0; if (own) { C.timer = COLOSSUS.cooldown; this.become(C, 'recover'); } }
        break;
      }
    }
    // a shake: its whole body whipping side to side, standing where it is
    if (C.shake > 0) {
      const e = smooth(Math.min(C.shake, COLOSSUS.shakeFor - C.shake) / 0.3); // (in and out over 0.3 s)
      const ph = C.shake * 2 * Math.PI * 2.3;
      W[ROLL] += 0.16 * e * Math.sin(ph); W[SWAY] += 1.6 * e * Math.sin(ph); W[CHEST] += 0.14 * e * Math.sin(ph * 0.8 + 1);
      W[LEAN] += 0.08 * e * Math.cos(ph * 0.5); W[HEAD_YAW] += 0.35 * e * Math.sin(ph * 1.3); W[DROP] += 0.5 * e;
      speed = 0;
      rate = Math.max(rate, 16);
    }
    C.glow = ease(C.glow, C.mind === 'down' ? 1 : C.mind === 'rise' || C.mind === 'die' ? 0.4 : rage < 1 ? 0.25 : 0, 2, dt);
    C.rate = rate;
    // turning and walking (only standing up)
    const upright = C.topple === 0 && C.mind !== 'vanish';
    let turn = 0;
    if (upright && turnTo === turnTo) turn = Math.max(-1, Math.min(1, wrap(turnTo - C.yaw) / 0.3)) * turnRate;
    C.turning = ease(C.turning, turn, 1.5, dt);
    C.speed = ease(C.speed, upright ? speed : 0, 0.8, dt);
    C.yaw = wrap(C.yaw + C.turning * dt);
    C.x += Math.sin(C.yaw) * C.speed * dt;
    C.z += Math.cos(C.yaw) * C.speed * dt;
    if (upright && C.mind !== 'stomp') this.walk(C, dt, hx, hy, hz);
    // the body follows on its springs
    const P = C.pose, V = C.vel;
    for (let i = 0; i < STANCE; i++) P[i] = spring(P[i], W[i], V, i, i === HEAD_YAW || i === HEAD_PITCH ? Math.max(W_HEAD, C.shake > 0 ? 9 : 0) : rate, dt);
    const gy = 0.5 * (C.feet[0].y + C.feet[1].y);
    C.y = ease(C.y, gy, 3, dt);
    this.posed(C);
  }

  /** Its walking speed turning by `off` (rad) toward where it goes: full ahead, slower turning, a crawl turning about. */
  private gait(off: number): number {
    return COLOSSUS.walk * Math.max(0.25, Math.min(1, Math.cos(off) * 1.3 - 0.15));
  }

  /** Where the stomping foot comes down: toward the hero, within the leg's reach of where it stands (world). */
  private stompAim(C: Colossus, out: Float64Array): void {
    const R = this.rig!;
    this.toLocal(C, C.aimX, 0, C.aimZ, V3);
    const nx = R.rest[3 * (C.side ? R.j.ankle_r : R.j.ankle_l)], nz = 1.5;
    let dx = V3[0] - nx, dz = V3[2] - nz;
    const l = Math.hypot(dx, dz), m = 4.5;
    if (l > m) { dx *= m / l; dz *= m / l; }
    this.toWorld(C, nx + dx, 0, nz + dz, out);
    out[1] = ground(out[0], out[2], this.seed);
  }

  /**
   * The attack for the hero `q` (off: its bearing off its heading): a stomp if it stands close by a foot, else a
   * smash or its fall if it is before it, in reach; none otherwise (it walks on).
   */
  private choose(C: Colossus, q: Body, off: number): void {
    const rage = C.hp < COLOSSUS.hp * COLOSSUS.rage;
    let foot = -1, fd: number = COLOSSUS.stompNear;
    for (let s = 0; s < 2; s++) {
      const d = Math.hypot(q.x - C.feet[s].x, q.z - C.feet[s].z);
      if (d < fd) { fd = d; foot = s; }
    }
    this.toLocal(C, q.x, 0, q.z, V3);
    const lx = V3[0], lz = V3[2];
    C.aimX = q.x; C.aimZ = q.z;
    for (const F of C.feet) { F.fx = F.x; F.fy = F.y; F.fz = F.z; }
    if (foot >= 0) {
      C.side = foot;
      this.become(C, 'stomp');
    } else if (lz > 5 && lz < COLOSSUS.reach + 2 && Math.abs(lx) < 12 && Math.abs(off) < 0.45) {
      this.become(C, lz > 9 && C.rand() > (rage ? 0.5 : 0.68) ? 'fall' : 'smash');
    }
  }

  /**
   * Its feet: a planted one that fell behind (or turned) steps once the other is down; a step lifts, carries the
   * foot ahead of where the body will be, and comes down with a quake. The hips sway over the planted foot.
   */
  private walk(C: Colossus, dt: number, hx: number, hy: number, hz: number): void {
    const R = this.rig!, W = C.want;
    let swing = 0;
    for (let s = 0; s < 2; s++) {
      const F = C.feet[s];
      if (F.u < 0) continue;
      F.u = Math.min(1, F.u + dt / COLOSSUS.stepFor);
      const u = smooth(F.u);
      F.x = F.fx + (F.tx - F.fx) * u; F.z = F.fz + (F.tz - F.fz) * u;
      F.y = F.fy + (F.ty - F.fy) * u + COLOSSUS.lift * Math.sin(Math.PI * F.u);
      swing = (s ? 1 : -1) * Math.sin(Math.PI * F.u);
      if (F.u >= 1) {
        F.u = -1;
        F.y = F.ty;
        this.quake(F.x, F.y, F.z, 5);
        // (under its foot)
        if (Math.hypot(hx - F.x, hz - F.z) < 4 && hy - F.y < 3 && this.cling?.C !== C) this.heroHit(10, hx - F.x, hz - F.z, 6, 5);
      }
    }
    const free = C.mind === 'roam' || C.mind === 'hunt' || C.mind === 'idle' || C.mind === 'recover';
    if (!free) return;
    if (swing) {
      // over the planted foot, rising as the other lifts
      W[SWAY] = -swing * 1.1; W[ROLL] = -swing * 0.045; W[DROP] = 1.7 - 0.35 * Math.abs(swing);
      this.armsDown(W, swing * Math.min(1, C.speed / COLOSSUS.walk));
    } else this.armsDown(W, 0);
    if (C.feet[0].u >= 0 || C.feet[1].u >= 0 || C.shake > 0 || C.puppet) return;
    // a step for the foot furthest from where it would stand
    let worst = -1, far = 0;
    for (let s = 0; s < 2; s++) {
      const F = C.feet[s];
      const a = R.rest.subarray(3 * (s ? R.j.ankle_r : R.j.ankle_l));
      this.toLocal(C, F.x, F.y, F.z, V3);
      const e = Math.hypot(V3[0] - a[0], V3[2] - a[2]);
      const need = C.speed > 0.2 ? COLOSSUS.stride * 0.5 : 1.2;
      if (e > need && e > far) { far = e; worst = s; }
    }
    if (worst < 0) return;
    this.stepFoot(C, worst);
  }

  /** Foot s steps: to ahead of where the body will be when it lands. */
  private stepFoot(C: Colossus, s: number): void {
    const R = this.rig!, F = C.feet[s];
    const a = R.rest.subarray(3 * (s ? R.j.ankle_r : R.j.ankle_l));
    const ahead = C.speed * COLOSSUS.stepFor * 1.1, ty = C.yaw + C.turning * COLOSSUS.stepFor;
    const cx = C.x + Math.sin(C.yaw) * C.speed * COLOSSUS.stepFor, cz = C.z + Math.cos(C.yaw) * C.speed * COLOSSUS.stepFor;
    const c = Math.cos(ty), sn = Math.sin(ty);
    F.fx = F.x; F.fy = F.y; F.fz = F.z;
    F.tx = cx + c * a[0] + sn * (a[2] + ahead * 0.5);
    F.tz = cz - sn * a[0] + c * (a[2] + ahead * 0.5);
    F.ty = ground(F.tx, F.tz, this.seed);
    F.u = 0;
  }

  /** The hero struck: q quarters, flung from (dx, dz) away at `out` m/s and `up`. */
  private heroHit(q: number, dx: number, dz: number, out: number, up: number): void {
    if (this.hit && this.hit.q >= q) return;
    const l = Math.hypot(dx, dz) || 1;
    const h = this.hitOut;
    h.q = q; h.vx = (dx / l) * out; h.vy = up; h.vz = (dz / l) * out;
    this.hit = h;
  }

  /**
   * A blow on the ground at (ax, az) (or along the line to (bx, bz)): the hero within `inner` takes `hit` and is
   * flung hard; within `outer`, on the ground, `ring` (a hero in the air rides the shockwave over).
   */
  private blow(ax: number, az: number, bx: number, bz: number, hx: number, hy: number, hz: number, inner: number, outer: number, hit: number, ring: number): void {
    if (this.cling) return; // (holding on to a colossus: off the ground)
    let px = ax, pz = az;
    if (bx === bx) {
      const ux = bx - ax, uz = bz - az, t = Math.max(0, Math.min(1, ((hx - ax) * ux + (hz - az) * uz) / (ux * ux + uz * uz)));
      px = ax + ux * t; pz = az + uz * t;
    }
    const d = Math.hypot(hx - px, hz - pz), air = hy - ground(hx, hz, this.seed);
    if (d < inner && air < 6) this.heroHit(hit, hx - px, hz - pz, 7, 6.5);
    else if (d < outer && air < 1.2) this.heroHit(ring, hx - px, hz - pz, 4.5, 4);
  }

  private quake(x: number, y: number, z: number, size: number): void {
    const q = this.quakePool[this.quakes.length] ??= { x: 0, y: 0, z: 0, size: 0 };
    q.x = x; q.y = y; q.z = z; q.size = size;
    this.quakes.push(q);
  }

  /** A low rumble under it (dust at its feet). */
  private rumble(C: Colossus, size: number): void {
    const F = C.feet[(C.t * 7 | 0) & 1];
    this.quake(F.x, F.y, F.z, size * 6);
  }

  // --- its body ----------------------------------------------------------------------------------------------------

  /** C's collision body skinned as it stands this tick (null: the model has none). */
  private bodyOf(C: Colossus): ColossusBody | null {
    if (!this.coll || !this.rig) return null;
    if (C.body && C.bodyTick === this.tick) return C.body;
    C.body ??= new ColossusBody(this.coll);
    this.posed(C);
    this.worldJoints(C, JM);
    C.body.skin(JM);
    C.bodyTick = this.tick;
    return C.body;
  }

  /** The posed joints (ROT/POS) as world skinning matrices into out (3×4 rows, 12 a joint). */
  private worldJoints(C: Colossus, out: Float64Array): void {
    const rest = this.rig!.rest, c = Math.cos(C.yaw), s = Math.sin(C.yaw);
    for (let k = 0; k < COLOSSUS_JOINTS; k++) {
      const r = 9 * k, o = 12 * k;
      for (let q = 0; q < 4; q++) {
        // (its frame's row i, column q: the rotation, then the translation P − R·rest)
        const at = (i: number) => (q < 3 ? ROT[r + 3 * i + q]
          : POS[3 * k + i] - (ROT[r + 3 * i] * rest[3 * k] + ROT[r + 3 * i + 1] * rest[3 * k + 1] + ROT[r + 3 * i + 2] * rest[3 * k + 2]));
        const a0 = at(0), a1 = at(1), a2 = at(2);
        out[o + q] = c * a0 + s * a2; out[o + 4 + q] = a1; out[o + 8 + q] = -s * a0 + c * a2;
      }
      out[o + 3] += C.x; out[o + 7] += C.y; out[o + 11] += C.z;
    }
  }

  /** Where segment a → b first meets C (its fraction; −1: misses): its eye or its stone (C.hitCap says which). */
  private meets(C: Colossus, ax: number, ay: number, az: number, bx: number, by: number, bz: number, pad: number): number {
    if (Math.min(Math.hypot(ax - C.x, az - C.z), Math.hypot(bx - C.x, bz - C.z)) > 45) return -1;
    const e = C.eye;
    const te = segSphere(ax, ay, az, bx, by, bz, e[0], e[1], e[2], EYE_R + pad);
    const B = this.bodyOf(C);
    let tb = B ? B.ray(ax, ay, az, bx, by, bz, HIT) : -1;
    if (tb < 0 && pad > 0 && B && B.nearest(bx, by, bz, pad, HIT)) tb = 1;
    if (te >= 0 && (tb < 0 || te <= tb)) { C.hitCap = EYE_HIT; return te; }
    if (tb >= 0) { C.hitCap = 0; return tb; }
    return -1;
  }

  /** How far (px, py, pz) is from C's surface (its stone or its eye; C.hitCap says which), the point there into P. */
  private near(C: Colossus, px: number, py: number, pz: number, P: Float64Array): number {
    if (Math.hypot(px - C.x, pz - C.z) > 45) return Infinity;
    let best = Infinity;
    const B = this.bodyOf(C);
    if (B && B.nearest(px, py, pz, 8, HIT)) { best = HIT.d; P[0] = HIT.x; P[1] = HIT.y; P[2] = HIT.z; C.hitCap = 0; }
    const e = C.eye, de = Math.hypot(px - e[0], py - e[1], pz - e[2]);
    if (de - EYE_R < best) {
      best = Math.max(0, de - EYE_R);
      const f = de > 1e-6 ? EYE_R / de : 0;
      P[0] = e[0] + (px - e[0]) * f; P[1] = e[1] + (py - e[1]) * f; P[2] = e[2] + (pz - e[2]) * f;
      C.hitCap = EYE_HIT;
    }
    return best;
  }

  // --- the hero on it -------------------------------------------------------------------------------------------

  /**
   * The hero grabbing hold: its chest touching the stone in the air, or on the ground pushing into it (wish: the
   * way the keys push, world xz, length 0..1). True if it now holds on.
   */
  grab(hero: HeroState, wishX: number, wishZ: number, dt: number): boolean {
    this.grabIn = Math.max(0, this.grabIn - dt);
    const held = !this.cling && this.grabIn <= 0 && !!this.rig && this.grabHold(hero, wishX, wishZ);
    const fa = this.feetAt; // (where its feet were, for next tick's sweep)
    fa[0] = hero.x; fa[1] = hero.y - PLAYER.groundBias; fa[2] = hero.z; fa[3] = this.tick;
    return held;
  }

  private grabHold(hero: HeroState, wishX: number, wishZ: number): boolean {
    const air = hero.move === 'air' || hero.move === 'glide';
    if (!air && hero.move !== 'ground') return false;
    const px = hero.x, py = hero.y - PLAYER.groundBias + 1.1, pz = hero.z;
    for (const C of this.list) {
      if (C.mind === 'vanish' || C.mind === 'die' || Math.hypot(px - C.x, pz - C.z) > 45) continue;
      const B = this.bodyOf(C);
      if (!this.climbWalls) {
        // (no climbing: only landing on top of it, falling, the feet onto stone level enough to stand on — swept from
        // where they were last tick, a fast fall would pass through it)
        if (!air || hero.vy > 0.5 || !B) continue;
        const fy = py - 1.1, was = this.feetAt;
        const swept = was[3] === this.tick - 1 && B.ray(was[0], was[1] + 0.3, was[2], px, fy - 0.1, pz, HIT, true) >= 0;
        if (!swept && !B.nearest(px, fy + 0.2, pz, 0.6, HIT, null, true)) continue;
        if (HIT.ny < 0.72) continue;
        this.cling = { C, tri: HIT.tri, u: HIT.u, v: HIT.v, top: true };
        this.place(hero, 0, 0, 0);
        return true;
      }
      if (!B || !B.nearest(px, py, pz, air ? 1.1 : 0.9, HIT, null, true)) continue;
      if (!air) {
        // (on the ground: only pushing into a side of it)
        const nh = Math.hypot(HIT.nx, HIT.nz);
        if (nh < 0.3 || -(wishX * HIT.nx + wishZ * HIT.nz) / nh < 0.5) continue;
      }
      this.cling = { C, tri: HIT.tri, u: HIT.u, v: HIT.v, top: false };
      this.place(hero, 0, 0, 0);
      return true;
    }
    return false;
  }

  /** Let go (a jump off, struck, the colossus gone): the hero falling from where it was, pushed off at (vx, vy, vz). */
  release(hero: HeroState, vx = 0, vy = 0, vz = 0): void {
    if (!this.cling) return;
    this.cling = null;
    this.grabIn = 1;
    hero.clinging = false;
    hero.move = 'air';
    hero.climbR = 0;
    hero.vx = vx; hero.vy = vy; hero.vz = vz;
  }

  /**
   * The hero holding on, a tick: climbing (forward: up the stone, strafe: across it; standing on top, the keys walk
   * it the camera's way), its stamina, a jump to let go, a stab (`attack`) at the eye with a weapon of `weaponDamage`.
   * Its stamina run dry, it is thrown (`thrown`).
   */
  climb(hero: HeroState, forward: number, strafe: number, camYaw: number, jump: boolean, attack: boolean, weaponDamage: number, dt: number): void {
    this.stab = null;
    this.thrown = null;
    this.stabIn = Math.max(0, this.stabIn - dt);
    const G = this.cling;
    if (!G || !this.rig) return;
    const C = G.C;
    const B = this.list.includes(C) && C.mind !== 'vanish' ? this.bodyOf(C) : null;
    if (!B) { this.release(hero); return; }
    // the stone where it holds, and its way out
    B.at(G.tri, G.u, G.v, P0, N0);
    const nx = N0[0], ny = N0[1], nz = N0[2];
    G.top = G.top ? ny > 0.5 : ny > 0.72;
    if (!G.top && !this.climbWalls) { this.release(hero, hero.vx, 0, hero.vz); return; } // (off its edge: falling)
    // which way the keys push, along the stone
    let dx: number, dy: number, dz: number;
    if (G.top) {
      const sn = Math.sin(camYaw), cs = Math.cos(camYaw);
      dx = sn * forward + cs * strafe; dz = cs * forward - sn * strafe; dy = 0;
    } else {
      // up the stone (the world's up along it), and the hero's right (facing into it)
      let ux = -ny * nx, uy = 1 - ny * ny, uz = -ny * nz;
      const ul = Math.hypot(ux, uy, uz);
      if (ul < 0.35) {
        // (under an overhang: up is out from under it, away from its middle)
        ux = P0[0] - C.x; uy = 0; uz = P0[2] - C.z;
        const k = ux * nx + uz * nz;
        ux -= k * nx; uy -= k * ny; uz -= k * nz;
        const l = Math.hypot(ux, uy, uz) || 1;
        ux /= l; uy /= l; uz /= l;
      } else { ux /= ul; uy /= ul; uz /= ul; }
      const rx = uy * -nz - uz * -ny, ry = uz * -nx - ux * -nz, rz = ux * -ny - uy * -nx; // up × facing
      dx = ux * forward - rx * strafe; dy = uy * forward - ry * strafe; dz = uz * forward - rz * strafe;
    }
    const k = dx * nx + dy * ny + dz * nz;
    dx -= k * nx; dy -= k * ny; dz -= k * nz;
    const dl = Math.hypot(dx, dy, dz);
    let moving = dl > 0.05;
    const shaking = C.shake > 0;
    const speed = COLOSSUS.climb * (shaking ? 0.35 : 1);
    if (moving) {
      // a step along the stone, then back onto it: the nearest surface turned as this is (never through a thin part)
      const st = (speed * dt) / dl;
      const ok = B.nearest(P0[0] + dx * st, P0[1] + dy * st, P0[2] + dz * st, 1.2, HIT, N0, true);
      const gone = ok ? Math.hypot(HIT.x - P0[0], HIT.y - P0[1], HIT.z - P0[2]) : 0;
      if (ok && gone > 0.3 * speed * dt) { G.tri = HIT.tri; G.u = HIT.u; G.v = HIT.v; } else {
        // (held up: stone in the way — a wall rising from a ledge, an overhang over a wall — goes onto it; else stays)
        const r = 0.6 / dl;
        if (B.nearest(P0[0] + dx * r, P0[1] + dy * r, P0[2] + dz * r, 1.0, HIT, null, true) && HIT.tri !== G.tri
          && (HIT.nx * dx + HIT.ny * dy + HIT.nz * dz) / dl < -0.2 && (this.climbWalls || HIT.ny > 0.72)) {
          G.tri = HIT.tri; G.u = HIT.u; G.v = HIT.v;
          G.top = HIT.ny > 0.72;
        } else if (ok) { G.tri = HIT.tri; G.u = HIT.u; G.v = HIT.v; } else moving = false;
      }
    }
    // stamina: holding on costs it, a shake much more, standing on top rests (not through a shake)
    const cost = shaking ? COLOSSUS.shakeCost * (moving ? 1.5 : 1) : G.top ? -HERO.staminaRegen : moving ? COLOSSUS.climbCost : COLOSSUS.hangCost;
    hero.stamina = Math.min(1, hero.stamina - cost * dt);
    if (cost > 0) hero.rest = 0;
    const wx = hero.x, wy = hero.y, wz = hero.z;
    this.place(hero, moving ? (dx / dl) * speed : 0, moving ? (dy / dl) * speed : 0, moving ? (dz / dl) * speed : 0);
    if (hero.stamina <= 0) {
      hero.stamina = 0;
      hero.exhausted = true;
      // thrown off: away from its body, with how it was carried
      const vx = (hero.x - wx) / dt, vy = (hero.y - wy) / dt, vz = (hero.z - wz) / dt;
      this.thrown = [nx * 7 + Math.max(-8, Math.min(8, vx)), 5 + Math.max(-4, Math.min(6, vy)), nz * 7 + Math.max(-8, Math.min(8, vz))];
      this.release(hero, ...this.thrown);
      return;
    }
    if (jump && !hero.jumpHeld) {
      hero.jumpHeld = true;
      this.release(hero, nx * 4, 4.5, nz * 4);
      return;
    }
    hero.jumpHeld = jump;
    // a stab: into the eye if it is within reach of the hands
    if (attack && this.stabIn <= 0 && !G.top) {
      this.stabIn = 0.75;
      const hy = hero.y - PLAYER.groundBias + 1.3, e = C.eye, st = this.stabOut;
      if (Math.hypot(hero.x - e[0], hy - e[1], hero.z - e[2]) < EYE_R + 1.6) {
        C.hitCap = EYE_HIT;
        st.took = this.hurt(C, weaponDamage * COLOSSUS.stab);
        st.x = e[0]; st.y = e[1]; st.z = e[2];
      } else {
        st.took = 0;
        st.x = P0[0]; st.y = P0[1]; st.z = P0[2];
      }
      this.stab = st;
    }
  }

  /** The hero put where it holds on (its root; facing the stone, or standing on it), moving (vx, vy, vz) on it. */
  private place(hero: HeroState, vx: number, vy: number, vz: number): void {
    const G = this.cling!, B = this.bodyOf(G.C)!;
    B.at(G.tri, G.u, G.v, P0, N0);
    hero.vx = vx; hero.vy = vy; hero.vz = vz;
    hero.clinging = true;
    if (G.top) {
      hero.move = 'ground';
      hero.x = P0[0]; hero.y = P0[1] + PLAYER.groundBias; hero.z = P0[2];
      hero.ground = P0[1]; hero.gx = 0; hero.gz = 0;
      if (Math.hypot(vx, vz) > 0.1) hero.yaw = Math.atan2(vx, vz);
      hero.climbR = 0;
      return;
    }
    // hanging: the hips off the stone where it holds by the climb's clearance, level (the rig climbs upright),
    // facing it; the rig's bark is a wide trunk behind the stone
    let hx = N0[0], hz = N0[2];
    const nh = Math.hypot(hx, hz);
    if (nh > 0.15) { hx /= nh; hz /= nh; hero.yaw = Math.atan2(-hx, -hz); } else { hx = -Math.sin(hero.yaw); hz = -Math.cos(hero.yaw); }
    const clear = climbClearance(CLIMB_R);
    hero.move = 'climb';
    hero.x = P0[0] + hx * clear; hero.z = P0[2] + hz * clear;
    hero.y = Math.max(P0[1] - 0.9, ground(hero.x, hero.z, this.seed) + PLAYER.groundBias); // (low down: on its feet, not in the ground)
    hero.climbR = CLIMB_R;
    hero.climbX = P0[0] - hx * CLIMB_R; hero.climbZ = P0[2] - hz * CLIMB_R;
    hero.wallNX = hx; hero.wallNZ = hz;
    hero.climbVA = vy; hero.climbVS = 0;
    hero.climbTaper = 0; hero.climbLeanX = 0; hero.climbLeanZ = 0;
    hero.climbTaperDown = 0; hero.climbLeanDownX = 0; hero.climbLeanDownZ = 0;
    hero.climbTop = P0[1] + 50;
    hero.mantle = 0;
    hero.ground = Math.min(hero.y - 3, ground(hero.x, hero.z, this.seed));
  }

  // --- between players --------------------------------------------------------------------------------------------

  /** Colossus C's state for the other players (its thinker's): COLOSSUS_MSG.state. */
  encode(C: Colossus): number[] {
    const r = (v: number) => Math.round(v * 100) / 100;
    const out = [COLOSSUS_MSG.state, C.id, r(C.x), r(C.z), r(C.yaw), MINDS.indexOf(C.mind), r(C.t), r(C.hp), r(C.speed), r(C.turning),
      r(C.tx), r(C.tz), r(C.aimX), r(C.aimZ), C.side, r(C.shake)];
    for (const F of C.feet) {
      const planted = F.u < 0;
      out.push(planted ? -1 : r(F.u), r(planted ? F.x : F.fx), r(planted ? F.z : F.fz), r(planted ? F.x : F.tx), r(planted ? F.z : F.tz));
    }
    return out;
  }

  /** Another player's state of a colossus (COLOSSUS_MSG.state): this one's puppet follows it (made if new). */
  receive(d: readonly number[]): void {
    if (this.authority || !this.rig || d.length < 26) return;
    const id = d[1] >>> 0, mind = MINDS[d[5]];
    if (!mind || this.slain.has(id)) return;
    let C = this.list.find((c) => c.id === id);
    if (!C) {
      if (mind === 'vanish') return;
      C = this.spawn(id, d[2], d[3], d[4]);
      C.puppet = true;
    }
    C.seen = 0;
    // where it is: eased toward the state (snapped when far off)
    const ex = d[2] - C.x, ez = d[3] - C.z;
    const k = Math.hypot(ex, ez) > 8 ? 1 : 0.5;
    C.x += ex * k; C.z += ez * k;
    C.yaw = wrap(C.yaw + wrap(d[4] - C.yaw) * k);
    if (C.mind !== mind) { C.mind = mind; C.t = d[6]; C.struck = false; } else if (Math.abs(C.t - d[6]) > 0.25) C.t = d[6];
    C.hp = d[7]; C.speed = d[8]; C.turning = d[9]; C.tx = d[10]; C.tz = d[11]; C.aimX = d[12]; C.aimZ = d[13]; C.side = d[14] ? 1 : 0;
    C.shake = d[15];
    for (let s = 0; s < 2; s++) {
      const F = C.feet[s], o = 16 + 5 * s, u = d[o];
      if (C.mind === 'stomp' && s === C.side) continue; // (its own timeline carries the stomping foot)
      if (u < 0) {
        if (F.u >= 0 && F.u > 0.85) continue; // (about to land by itself)
        F.x = F.tx = d[o + 1]; F.z = F.tz = d[o + 2]; F.y = F.ty = ground(F.x, F.z, this.seed); F.u = -1;
        F.fx = F.x; F.fz = F.z; F.fy = F.y;
      } else if (F.u < 0 || Math.abs(F.u - u) > 0.2) {
        F.fx = d[o + 1]; F.fz = d[o + 2]; F.fy = ground(F.fx, F.fz, this.seed);
        F.tx = d[o + 3]; F.tz = d[o + 4]; F.ty = ground(F.tx, F.tz, this.seed);
        F.u = u;
      }
    }
  }

  /** A player's blow on the eye of colossus `id` (COLOSSUS_MSG.hit), for its thinker. */
  struckBy(id: number, took: number): void {
    const C = this.list.find((c) => c.id === id);
    if (C && !C.puppet && took > 0 && took < 1000) this.damage(C, took);
  }

  // --- the pose ---------------------------------------------------------------------------------------------------

  /**
   * Its joints from its stance, its feet and fists (IK) and its fall, into ROT/POS (its frame); its capsules (world).
   */
  private posed(C: Colossus): void {
    const R = this.rig!, P = C.pose, J = R.j, rest = R.rest;
    const set = (k: number, p: number) => {
      // position from the parent's posed frame
      for (let i = 0; i < 3; i++) {
        POS[3 * k + i] = POS[3 * p + i]
          + ROT[9 * p + 3 * i] * (rest[3 * k] - rest[3 * p])
          + ROT[9 * p + 3 * i + 1] * (rest[3 * k + 1] - rest[3 * p + 1])
          + ROT[9 * p + 3 * i + 2] * (rest[3 * k + 2] - rest[3 * p + 2]);
      }
    };
    const local = (k: number, p: number, yaw: number, pitch: number, roll: number) => {
      set(k, p);
      euler(yaw, pitch, roll, L9, 0);
      mul(ROT, 9 * p, L9, 0, ROT, 9 * k);
    };
    // hips, chest, head
    euler(0, P[LEAN], P[ROLL], ROT, 0);
    POS[0] = rest[0] + P[SWAY]; POS[1] = rest[1] - P[DROP]; POS[2] = rest[2] + P[FWD];
    local(J.chest, J.hips, 0, P[CHEST], 0);
    local(J.head, J.chest, P[HEAD_YAW], P[HEAD_PITCH], 0);
    for (let s = 0; s < 2; s++) {
      const sg = s ? 1 : -1;
      const cl = s ? J.clavicle_r : J.clavicle_l, sh = s ? J.shoulder_r : J.shoulder_l, el = s ? J.elbow_r : J.elbow_l, wr = s ? J.wrist_r : J.wrist_l;
      local(cl, J.chest, 0, 0, -sg * P[CLAV]);
      // the arm: the shoulder and the elbow reach the fist's tip
      set(sh, cl);
      const hand = R.hand[s];
      A3[0] = P[FIST + 3 * s]; A3[1] = P[FIST + 3 * s + 1]; A3[2] = P[FIST + 3 * s + 2];
      POLE[0] = sg * 0.8; POLE[1] = -0.1; POLE[2] = -0.6;
      ik(POS.subarray(3 * sh, 3 * sh + 3), A3, R.arm[0], R.arm[1], POLE, M3);
      for (let i = 0; i < 3; i++) D0[i] = rest[3 * el + i] - rest[3 * sh + i];
      aim(D0, POLE, M3[0] - POS[3 * sh], M3[1] - POS[3 * sh + 1], M3[2] - POS[3 * sh + 2], POLE, ROT, 9 * sh);
      set(el, sh);
      for (let i = 0; i < 3; i++) D0[i] = hand[i] - rest[3 * el + i];
      aim(D0, POLE, A3[0] - POS[3 * el], A3[1] - POS[3 * el + 1], A3[2] - POS[3 * el + 2], POLE, ROT, 9 * el);
      set(wr, el);
      ROT.copyWithin(9 * wr, 9 * el, 9 * el + 9);
      // the leg: the hip and the knee reach the ankle over the foot
      const hp = s ? J.hip_r : J.hip_l, kn = s ? J.knee_r : J.knee_l, an = s ? J.ankle_r : J.ankle_l;
      set(hp, J.hips);
      const F = C.feet[s];
      this.toLocal(C, F.x, F.y, F.z, A3);
      A3[1] += R.ankleH;
      // (the knee forward, a little out)
      W3[0] = sg * 0.15; W3[1] = 0; W3[2] = 1;
      ik(POS.subarray(3 * hp, 3 * hp + 3), A3, R.leg[0], R.leg[1], W3, M3);
      for (let i = 0; i < 3; i++) D0[i] = rest[3 * kn + i] - rest[3 * hp + i];
      aim(D0, W3, M3[0] - POS[3 * hp], M3[1] - POS[3 * hp + 1], M3[2] - POS[3 * hp + 2], W3, ROT, 9 * hp);
      set(kn, hp);
      for (let i = 0; i < 3; i++) D0[i] = rest[3 * an + i] - rest[3 * kn + i];
      aim(D0, W3, A3[0] - POS[3 * kn], A3[1] - POS[3 * kn + 1], A3[2] - POS[3 * kn + 2], W3, ROT, 9 * kn);
      set(an, kn);
      // (the foot flat, turned the way the body faces)
      euler(0, 0, 0, ROT, 9 * an);
    }
    // its fall: everything turned forward about its toes
    if (C.topple) {
      euler(0, C.topple, 0, T9, 0);
      for (let k = 0; k < COLOSSUS_JOINTS; k++) {
        mul(T9, 0, ROT, 9 * k, L9, 0);
        ROT.set(L9, 9 * k);
        const y = POS[3 * k + 1], z = POS[3 * k + 2] - TOE_Z;
        POS[3 * k + 1] = T9[4] * y + T9[5] * z;
        POS[3 * k + 2] = TOE_Z + T9[7] * y + T9[8] * z;
      }
    }
    // its eye (world)
    this.jointPoint(C, J.head, EYE[0], EYE[1], EYE[2], C.eye);
  }

  /** Rest-frame point (x, y, z) carried by joint k as posed (ROT/POS), in the world, into out. */
  private jointPoint(C: Colossus, k: number, x: number, y: number, z: number, out: Float64Array): void {
    const R = this.rig!.rest;
    const dx = x - R[3 * k], dy = y - R[3 * k + 1], dz = z - R[3 * k + 2];
    const lx = POS[3 * k] + ROT[9 * k] * dx + ROT[9 * k + 1] * dy + ROT[9 * k + 2] * dz;
    const ly = POS[3 * k + 1] + ROT[9 * k + 3] * dx + ROT[9 * k + 4] * dy + ROT[9 * k + 5] * dz;
    const lz = POS[3 * k + 2] + ROT[9 * k + 6] * dx + ROT[9 * k + 7] * dy + ROT[9 * k + 8] * dz;
    this.toWorld(C, lx, ly, lz, out);
  }

  /**
   * The camera's eye p, looking at (fx, fy, fz) (the hero's head): drawn in in front of any colossus's stone between
   * them, then kept `pad` out of the stone.
   */
  clearView(fx: number, fy: number, fz: number, p: { x: number; y: number; z: number }, pad: number): void {
    for (const C of this.list) {
      if (Math.hypot(p.x - C.x, p.z - C.z) > 45 && Math.hypot(fx - C.x, fz - C.z) > 45) continue;
      const B = this.bodyOf(C);
      if (!B) continue;
      // (from a little off the head: the stone it holds may bulge round it; only stone the sight goes into)
      const L = Math.hypot(p.x - fx, p.y - fy, p.z - fz) || 1, s0 = Math.min(0.5, 0.6 / L);
      const t = B.ray(fx + (p.x - fx) * s0, fy + (p.y - fy) * s0, fz + (p.z - fz) * s0, p.x, p.y, p.z, HIT, true);
      if (t < 0) continue;
      const k = Math.max(0.3, s0 + (1 - s0) * t - pad / L);
      p.x = fx + (p.x - fx) * k; p.y = fy + (p.y - fy) * k; p.z = fz + (p.z - fz) * k;
    }
    this.keepOut(p, pad);
  }

  /** Point p kept `pad` out of every colossus's stone (it would see from inside it). */
  keepOut(p: { x: number; y: number; z: number }, pad: number): void {
    for (const C of this.list) {
      if (Math.hypot(p.x - C.x, p.z - C.z) > 45) continue;
      const B = this.bodyOf(C);
      if (!B || !B.nearest(p.x, p.y, p.z, 3, HIT)) continue;
      // (in front of the nearest face by its normal; behind it: inside)
      const s = (p.x - HIT.x) * HIT.nx + (p.y - HIT.y) * HIT.ny + (p.z - HIT.z) * HIT.nz;
      if (s >= pad) continue;
      p.x = HIT.x + HIT.nx * pad; p.y = HIT.y + HIT.ny * pad; p.z = HIT.z + HIT.nz * pad;
    }
  }

  /** The hero's body (two spheres over its feet at y) pushed out of the stone, level (not holding on). True if moved. */
  collideHero(hero: { x: number; y: number; z: number }): boolean {
    if (this.cling) return false;
    let moved = false;
    for (const C of this.list) {
      if (C.mind === 'vanish' || Math.hypot(hero.x - C.x, hero.z - C.z) > 35) continue;
      const B = this.bodyOf(C);
      if (!B) continue;
      for (const h of [0.5, 1.3]) {
        const cx = hero.x, cy = hero.y + h, cz = hero.z, r = 0.45;
        if (!B.nearest(cx, cy, cz, 2.5, HIT)) continue; // (deep enough to find it again if a limb swept into it)
        const s = (cx - HIT.x) * HIT.nx + (cy - HIT.y) * HIT.ny + (cz - HIT.z) * HIT.nz;
        if (s >= r) continue;
        const l = Math.hypot(HIT.nx, HIT.nz);
        if (l < 0.25) {
          if (s >= 0) continue; // (stone over or under it: its own steps and the ground see to that)
          // inside it, under or over stone: out the shortest way, level
          let best = Infinity, ox = 0, oz = 0;
          for (let a = 0; a < 8; a++) {
            const ux = Math.cos(a * Math.PI / 4), uz = Math.sin(a * Math.PI / 4);
            const t = B.ray(cx, cy, cz, cx + ux * 5, cy, cz + uz * 5, HIT);
            if (t >= 0 && t * 5 < best) { best = t * 5; ox = ux; oz = uz; }
          }
          if (best === Infinity) continue;
          const go = Math.min(1, best + r);
          hero.x += ox * go; hero.z += oz * go;
          moved = true;
          continue;
        }
        const go = Math.min(1, (r - s) / l);
        hero.x += (HIT.nx / l) * go; hero.z += (HIT.nz / l) * go;
        moved = true;
      }
    }
    return moved;
  }

  /** The nearest colossi for the renderer, camera-relative to (cx, cy, cz). */
  pack(cx: number, cy: number, cz: number, out: ColossusFrame): ColossusFrame {
    out.count = 0;
    out.fading = 0;
    if (!this.rig) return out;
    const L = this.list;
    if (L.length > 1) L.sort((a, b) => (a.mind === 'vanish' ? 1 : 0) - (b.mind === 'vanish' ? 1 : 0) || Math.hypot(a.x - cx, a.z - cz) - Math.hypot(b.x - cx, b.z - cz));
    const D = out.data, U = out.u32, S = this.pose;
    for (const C of L) {
      if (out.count >= COLOSSI_DRAWN) break;
      if (Math.hypot(C.x - cx, C.z - cz) > 2500) continue;
      this.posed(C);
      this.skin(S);
      const o = out.count * COLOSSUS_STRIDE;
      U[o] = C.id;
      D[o + 1] = C.glow;
      D[o + 2] = C.mind === 'vanish' ? Math.min(1, C.t / COLOSSUS.vanishFor) : 0;
      D[o + 3] = C.flash / FLASH;
      if (C.mind === 'vanish') out.fading++;
      const c = Math.cos(C.yaw), s = Math.sin(C.yaw);
      const tx = C.x - cx, ty = C.y - cy, tz = C.z - cz;
      for (let j = 0; j < COLOSSUS_JOINTS; j++) {
        const a = j * MAT, d = o + 4 + j * MAT;
        for (let q = 0; q < 4; q++) {
          const a0 = S[a + q], a1 = S[a + 4 + q], a2 = S[a + 8 + q];
          D[d + q] = c * a0 + s * a2; D[d + 4 + q] = a1; D[d + 8 + q] = -s * a0 + c * a2;
        }
        D[d + 3] += tx; D[d + 7] += ty; D[d + 11] += tz;
      }
      out.count++;
    }
    return out;
  }

  /** The posed joints (ROT/POS) as skinning matrices (3×4 rows, its frame): [R | P − R·rest]. */
  private skin(out: Float32Array): void {
    const rest = this.rig!.rest;
    for (let k = 0; k < COLOSSUS_JOINTS; k++) {
      const r = 9 * k, o = k * MAT;
      for (let i = 0; i < 3; i++) {
        out[o + 4 * i] = ROT[r + 3 * i]; out[o + 4 * i + 1] = ROT[r + 3 * i + 1]; out[o + 4 * i + 2] = ROT[r + 3 * i + 2];
        out[o + 4 * i + 3] = POS[3 * k + i] - (ROT[r + 3 * i] * rest[3 * k] + ROT[r + 3 * i + 1] * rest[3 * k + 1] + ROT[r + 3 * i + 2] * rest[3 * k + 2]);
      }
    }
  }
}

/** How far it falls forward (rad), about its toes (TOE_Z ahead of its middle). */
const TOPPLE = 1.3;
const TOE_Z = 2.5;

/** The rig's bark for a hero holding on to a colossus: a wide trunk (m), nearly a wall. */
const CLIMB_R = 3;

/** Where segment a → b first enters the sphere at c of radius r (its fraction; −1: misses; 0: starts inside). */
function segSphere(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number, r: number): number {
  const dx = bx - ax, dy = by - ay, dz = bz - az, mx = ax - cx, my = ay - cy, mz = az - cz;
  const c = mx * mx + my * my + mz * mz - r * r;
  if (c <= 0) return 0;
  const a = dx * dx + dy * dy + dz * dz, b = mx * dx + my * dy + mz * dz;
  if (a < 1e-12 || b > 0) return -1;
  const disc = b * b - a * c;
  if (disc < 0) return -1;
  const t = (-b - Math.sqrt(disc)) / a;
  return t <= 1 ? t : -1;
}
