/**
 * Procedural hero animation and the shader hand-off.
 * ---------------------------------------------------------------------------
 * An 18-joint skeleton is posed every frame from the locomotion state: a gait
 * cycle whose stance feet stay planted on the terrain plane (∇E at the root)
 * with two-bone IK for knees and elbows, plus air, paraglider, climb and swim
 * poses cross-faded by smoothed weights. packHeroUniforms() then writes the
 * joints, the torso / head frames and a bounding sphere camera-relative into
 * one Float32Array for demoShader.ts, which builds the body from capsules —
 * no mesh, no texture. Everything is hero-local (x right, y up, z forward,
 * origin at the root on the ground) until packing. Allocation-free per frame.
 *
 * Over the locomotion poses, the combat layer (poseCombat) takes the arms and
 * the gear as a CombatPose asks (combat/heroCombat.ts for the hero, a foe's AI,
 * a remote player's snapshot): the weapon fetched from the back into the right
 * hand and the shield onto the left arm, each swing's blade path
 * (combat/moves.ts) with the hand riding it and the shoulders twisting with
 * it, the shield raised, the bow drawn with the string hand at the cheek, a
 * throw's wind-up; a flinch or a daze over the upper body; and whole-body
 * turns (a spin attack's yaw, a backflip, a side hop's roll, a knockdown).
 * With a rock (poseRock): stooping to lift it, a stone carried at the right
 * shoulder or a bigger rock overhead in both hands, a throw wound up and
 * flung (overhand, or a two-handed heave over the head), both hands driving
 * into a boulder; the rock's place comes out as HeroRig.rockAt.
 * The gear (sword or club, shield, bow) is packed as frames for the shaders.
 */

import { HERO_SHAPE as P, PLAYER, CLIMB, SKYDIVE } from '../constants';
import { DEFAULT_DRESS } from './characters';
import { climbClearance, snapToClimbed, type HeroState } from './locomotion';
import { MOVE, MOVES, SWING_CHANNELS, bladeDir, swingAt } from '../combat/moves';
import { BOW_MODEL, MODEL, SHIELD_MODEL } from '../combat/weapons';

/** Joint indices, mirrored by the u_hero_j[] reads in demoShader.ts. */
export const J = {
  pelvis: 0, chest: 1, head: 2,
  shoulderL: 3, elbowL: 4, handL: 5,
  shoulderR: 6, elbowR: 7, handR: 8,
  hipL: 9, kneeL: 10, ankleL: 11, toeL: 12,
  hipR: 13, kneeR: 14, ankleR: 15, toeR: 16,
  glider: 17,
} as const;
export const HERO_JOINTS = 18;

/** Float32Array layout read by the engine's uniform uploads (dress: the tunic's rgb, then a pad; gear: GEAR). */
export const HERO_UNIFORM = { joints: 0, torso: 54, head: 63, bound: 72, misc: 76, dress: 80, gear: 84, length: 120 } as const;

/**
 * The gear's frames, vec4 by vec4 (hero-local in HeroRig.gear, camera-relative
 * from HERO_UNIFORM.gear): positions take the root's offset, directions only
 * its turn; each w carries a flag or an amount.
 */
export const GEAR = {
  grip: 0,    // the weapon's grip; w its model (combat/weapons.ts MODEL; 0 none)
  axis: 4,    // grip → tip (unit); w the blade's glow 0..1
  flat: 8,    // the blade's flat (unit, ⟂ axis); w 1: a sword in its sheath (the blade hidden)
  shield: 12, // the shield's centre; w its model (combat/weapons.ts SHIELD_MODEL; 0: not carried)
  shieldX: 16, // its face's right and up (unit); the face looks along right × up
  shieldY: 20,
  bow: 24,    // the bow's grip; w its model (BOW_MODEL; 0: not in hand)
  bowUp: 28,  // along its limbs (unit); w the draw 0..1
  bowFwd: 32, // toward the aim (unit); w 1: an arrow on the string
  length: 36,
} as const;

/** What the arms and the gear do (rig.ts poseCombat). */
export type PoseAct = 'none' | 'swing' | 'charge' | 'guard' | 'aim' | 'throw' | 'hurt' | 'stun' | 'down' | 'dead' | 'lift' | 'carry' | 'hurl' | 'push';

export type CombatPose = {
  weapon: number;   // the weapon's model (combat/weapons.ts MODEL), in hand or on the back; 0 none
  armed: boolean;   // the weapon (and the shield) in hand, else on the back
  shield: boolean;  // carries a shield
  shieldModel: number; // which one (combat/weapons.ts SHIELD_MODEL), while it is carried
  bowModel: number; // the bow drawn (BOW_MODEL); 0: none to draw
  act: PoseAct;
  move: number;     // the swing (combat/moves.ts MOVES) of a 'swing' or 'throw'; −1 none
  u: number;        // how far through the swing (0..1); seconds into any other act
  pitch: number;    // the aim's pitch (rad): the bow, a throw
  draw: number;     // the bow's draw 0..1; −1: no arrow on the string
  glow: number;     // the blade's glow 0..1 (a charged spin)
  spin: number;     // extra body yaw (rad): a spin attack
  flip: number;     // body pitch (rad) about the hips: a backflip (negative: backward)
  roll: number;     // body roll (rad): a side hop
  rock: number;     // the rock in the hands or pushed: its reach (m, its radius); 0 none
  rockHalf: number; // its half height (m)
  rockHand: boolean; // held in one hand (a stone), else in both
  rockFwd: number;  // where it lay when lifted (or will lie, put down), hero-local: ahead (m)…
  rockLow: number;  // … and its middle's height over the feet (m)
  /** The ragdoll's joints (hero/ragdoll.ts, world fp64; null: none) and how far the body is theirs 0..1. */
  rag: Float64Array | null;
  ragW: number;
};

export function createCombatPose(): CombatPose {
  return {
    weapon: MODEL.sword, armed: false, shield: true, shieldModel: SHIELD_MODEL.hylian, bowModel: BOW_MODEL.traveler, act: 'none', move: -1, u: 0,
    pitch: 0, draw: -1, glow: 0, spin: 0, flip: 0, roll: 0, rock: 0, rockHalf: 0, rockHand: false, rockFwd: 0.6, rockLow: 0.3,
    rag: null, ragW: 0,
  };
}

/** The pose of a hero at peace: sword and shield on the back (what callers without a combat layer draw). */
const AT_PEACE = createCombatPose();

const POSES = 8; // ground, air, glide, climb, swim, surf, skydive, ride
const W_GROUND = 0, W_AIR = 1, W_GLIDE = 2, W_CLIMB = 3, W_SWIM = 4, W_SURF = 5, W_SKY = 6, W_RIDE = 7;
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;

// The weapon arm's stances (as a swing's channels: blade yaw, pitch, hand reach, lift, torso twist):
// at the ready the blade low ahead (a club's lower), with the shield up it trails behind.
const READY = [24 * DEG, -38 * DEG, 0.3, -0.13, 8 * DEG];
const READY_CLUB = [30 * DEG, -58 * DEG, 0.28, -0.15, 6 * DEG];
const GUARD_ARM = [138 * DEG, -34 * DEG, 0.32, -0.16, 14 * DEG];
// The shield hand's offset from the chest: at the ready held low at the side, raised in front, out
// for balance through a swing.
const SHIELD_READY = [-0.3, -0.34, 0.15];
const SHIELD_GUARD = [-0.05, -0.06, 0.33];
const SHIELD_SWING = [-0.29, -0.24, 0.21];
// The shield's face (its normal) and up in each, hero-local.
const FACE_READY = [-0.76, 0.06, 0.64], UP_READY = [0, 1, 0.16];
const FACE_GUARD = [0.04, 0.08, 1], UP_GUARD = [0.02, 1, -0.06];
// The archer's stance: the shoulders turned side-on to the aim.
const AIM_TWIST = -42 * DEG;

export type HeroRig = {
  time: number;
  phase: number;       // gait cycle (rad)
  climbPhase: number;
  swimPhase: number;
  speed: number;       // smoothed horizontal speed
  sprint: number;      // smoothed 0..1 blends
  crouch: number;
  slide: number;
  tired: number;
  lean: number;        // forward torso lean (rad)
  bank: number;        // lean into turns (rad)
  land: number;        // landing: the pelvis' dip (m) on an underdamped spring…
  landT: number;       // … s since touchdown…
  landD: number;       // … and its scale (m)
  roll: number;        // s left of a landing roll…
  squat: number;       // … or a heavy landing's squat
  // skydiving (poseSkydive): the posture (0 spread, 1 nosedive), the flare, the steering read off the motion, the bank
  dive: number;
  flare: number;
  steerX: number; steerY: number;
  skyBank: number;
  glide: number;       // paraglider canopy open 0..1
  landings: number;
  lastYaw: number;
  lastSpeed: number;
  snap: boolean;       // next update jumps to the target pose (spawn, view switch)
  // the combat layer (poseCombat): smoothed 0..1 blends
  armed: number;       // weapon and shield fetched into the hands (0.5: the hands at the back)
  guard: number;       // the shield raised
  aim: number;         // the bow out and drawn
  point: number;       // the free arm pointing the way (a throw's wind-up)
  fall: number;        // body pitch (rad) of a knockdown
  spin: number;        // body yaw (rad) added at packing: a spin attack
  rockW: number;       // the hands on a rock (poseRock) 0..1
  /** The rock's middle in the hands, hero-local (x NaN: none held). */
  rockAt: Float64Array;
  // climbing (climbGait, poseClimb)
  treeR: number;       // the climbed surface's radius at the hips (a trunk's; a wall's, as the rig takes it)
  holds: boolean;      // the holds are placed (else: placed at the sockets on the next frame on a trunk)
  /** Per limb (LIMB order) LIMB_STRIDE values: the step's start and the hold (world, fp64), the step's progress (≥ 1: planted on the hold). */
  limbs: Float64Array;
  /** Per limb 4 values: where it is on the bark this frame (angle round the axis off the root's, height over the root, out from the bark), the heel's peel. */
  limbAt: Float64Array;
  shiftX: number;      // the pelvis' weight shift onto the support side (m) and its dip
  shiftY: number;
  lastPair: number;    // the diagonal pair that stepped last
  /** The climbing body's pelvis and chest (hero-local) settling on their spring, their velocities; [12] NaN: not yet; then the reach's
   *  pull on them (reachHolds) on its own spring and its velocities. */
  body: Float64Array;
  arm: Float64Array;   // the weapon arm: blade yaw, pitch, hand reach, lift, torso twist (eased toward the act's)
  shieldArm: Float64Array; // the shield hand's offset from the chest (eased)
  weights: Float64Array;
  local: Float64Array; // blended joints, hero-local
  pose: Float64Array;  // one pose, scratch
  frames: Float64Array; // torso then head frame: columns right, up, forward (hero-local)
  bound: Float64Array;  // hero-local centre xyz, radius
  gear: Float64Array;   // GEAR layout, hero-local
};

export function createHeroRig(): HeroRig {
  return {
    time: 0, phase: 0, climbPhase: 0, swimPhase: 0, speed: 0,
    sprint: 0, crouch: 0, slide: 0, tired: 0, lean: 0, bank: 0, land: 0, landT: Infinity, landD: 0, roll: 0, squat: 0, glide: 0,
    dive: 0, flare: 0, steerX: 0, steerY: 0, skyBank: 0,
    landings: 0, lastYaw: 0, lastSpeed: 0, snap: true,
    armed: 0, guard: 0, aim: 0, point: 0, fall: 0, spin: 0, rockW: 0, rockAt: new Float64Array([NaN, 0, 0]),
    treeR: 0.5, holds: false, limbs: new Float64Array(4 * LIMB_STRIDE), limbAt: new Float64Array(16), shiftX: 0, shiftY: 0, lastPair: 1, body: new Float64Array(25).fill(NaN),
    arm: Float64Array.from(READY),
    shieldArm: Float64Array.from(SHIELD_READY),
    weights: new Float64Array(POSES),
    local: new Float64Array(HERO_JOINTS * 3),
    pose: new Float64Array(HERO_JOINTS * 3),
    frames: new Float64Array(18),
    bound: new Float64Array(4),
    gear: new Float64Array(GEAR.length),
  };
}

/** Smoothing factor, or a jump straight to the target on a snap frame. */
function follow(snap: boolean, rate: number, dt: number): number {
  return snap ? 1 : 1 - Math.exp(-rate * dt);
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function wrapAngle(a: number): number {
  return a - TAU * Math.round(a / TAU);
}

function set(o: Float64Array, j: number, x: number, y: number, z: number): void {
  o[3 * j] = x; o[3 * j + 1] = y; o[3 * j + 2] = z;
}

// Torso frame of the pose being built (right, up, forward), hero-local.
const F = new Float64Array(9);

/**
 * The blended body's torso frame (into F): forward +z, or skydiving — the body tipped past level, head
 * first — its own forward, the shoulder line × the spine (+z's projection off the spine would turn about
 * as the spine passes horizontal, swapping chest and back, and the gear on the back with them).
 */
function bodyFrame(r: HeroRig, o: Float64Array): void {
  if (!(r.weights[W_SKY] > 1e-3)) { torsoFrame(o, 0, 0, 1); return; }
  const rx = o[3 * J.shoulderR] - o[3 * J.shoulderL], ry = o[3 * J.shoulderR + 1] - o[3 * J.shoulderL + 1], rz = o[3 * J.shoulderR + 2] - o[3 * J.shoulderL + 2];
  const sx = o[3 * J.chest] - o[3 * J.pelvis], sy = o[3 * J.chest + 1] - o[3 * J.pelvis + 1], sz = o[3 * J.chest + 2] - o[3 * J.pelvis + 2];
  torsoFrame(o, ry * sz - rz * sy, rz * sx - rx * sz, rx * sy - ry * sx);
}

/** Torso frame from the spine (pelvis → chest) and a forward hint. */
function torsoFrame(o: Float64Array, hx: number, hy: number, hz: number): void {
  let ux = o[3 * J.chest] - o[3 * J.pelvis];
  let uy = o[3 * J.chest + 1] - o[3 * J.pelvis + 1];
  let uz = o[3 * J.chest + 2] - o[3 * J.pelvis + 2];
  let l = Math.sqrt(ux * ux + uy * uy + uz * uz);
  ux /= l; uy /= l; uz /= l;
  const d = hx * ux + hy * uy + hz * uz;
  let fx = hx - ux * d, fy = hy - uy * d, fz = hz - uz * d;
  l = Math.sqrt(fx * fx + fy * fy + fz * fz);
  if (l < 1e-6) { fx = 0; fy = -1; fz = 0; l = 1; }
  fx /= l; fy /= l; fz /= l;
  // right = up × forward (x right, y up, z forward)
  F[0] = uy * fz - uz * fy; F[1] = uz * fx - ux * fz; F[2] = ux * fy - uy * fx;
  F[3] = ux; F[4] = uy; F[5] = uz;
  F[6] = fx; F[7] = fy; F[8] = fz;
}

/** p = base + right·a + up·b + forward·c in the current torso frame. */
function inTorso(o: Float64Array, j: number, base: number, a: number, b: number, c: number): void {
  o[3 * j] = o[3 * base] + F[0] * a + F[3] * b + F[6] * c;
  o[3 * j + 1] = o[3 * base + 1] + F[1] * a + F[4] * b + F[7] * c;
  o[3 * j + 2] = o[3 * base + 2] + F[2] * a + F[5] * b + F[8] * c;
}

/**
 * Two-bone IK: place joint `mid` between `root` and `end` (bone lengths l1,
 * l2), bending toward the pole direction. An unreachable end is pulled in.
 */
function ik(
  o: Float64Array, root: number, mid: number, end: number,
  l1: number, l2: number, px: number, py: number, pz: number,
): void {
  const rx = o[3 * root], ry = o[3 * root + 1], rz = o[3 * root + 2];
  let dx = o[3 * end] - rx, dy = o[3 * end + 1] - ry, dz = o[3 * end + 2] - rz;
  let d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const maxD = (l1 + l2) * 0.9999, minD = Math.abs(l1 - l2) + 1e-4;
  if (d < 1e-9) { dx = 0; dy = -1; dz = 0; d = 1; }
  const dc = Math.min(maxD, Math.max(minD, d));
  dx /= d; dy /= d; dz /= d;
  if (dc !== d) set(o, end, rx + dx * dc, ry + dy * dc, rz + dz * dc);
  const ca = (l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc);
  const sa = Math.sqrt(Math.max(0, 1 - ca * ca));
  const pd = px * dx + py * dy + pz * dz;
  let qx = px - dx * pd, qy = py - dy * pd, qz = pz - dz * pd;
  let ql = Math.sqrt(qx * qx + qy * qy + qz * qz);
  if (ql < 1e-6) { qx = 0; qy = 0; qz = 1; ql = 1; }
  qx /= ql; qy /= ql; qz /= ql;
  set(o, mid, rx + l1 * (ca * dx + sa * qx), ry + l1 * (ca * dy + sa * qy), rz + l1 * (ca * dz + sa * qz));
}

/** Shoulders and head from the torso frame; `nod` tips the head forward. */
function upperBody(o: Float64Array, nod: number): void {
  inTorso(o, J.shoulderL, J.chest, -P.shoulderW, -P.shoulderDrop, 0);
  inTorso(o, J.shoulderR, J.chest, P.shoulderW, -P.shoulderDrop, 0);
  // The neck keeps the head more upright than the spine.
  let hx = F[3] + F[6] * nod, hy = F[4] + 0.6 + F[7] * nod, hz = F[5] + F[8] * nod;
  const l = Math.sqrt(hx * hx + hy * hy + hz * hz);
  hx /= l; hy /= l; hz /= l;
  set(o, J.head,
    o[3 * J.chest] + hx * P.neck + F[6] * 0.025,
    o[3 * J.chest + 1] + hy * P.neck + F[7] * 0.025,
    o[3 * J.chest + 2] + hz * P.neck + F[8] * 0.025);
}

/**
 * Forward-kinematic arm in the torso frame: `swing` forward (rad), `abduct`
 * outward, `bend` at the elbow (forward).
 */
function armFK(o: Float64Array, side: number, swing: number, abduct: number, bend: number): void {
  const sh = side < 0 ? J.shoulderL : J.shoulderR;
  const el = side < 0 ? J.elbowL : J.elbowR;
  const ha = side < 0 ? J.handL : J.handR;
  const sa = Math.sin(abduct), cab = Math.cos(abduct);
  inTorso(o, el, sh, side * sa * P.upperArm, -cab * Math.cos(swing) * P.upperArm, cab * Math.sin(swing) * P.upperArm);
  // The forearm keeps 60 % of the abduction and adds the elbow bend.
  const s2 = swing + bend;
  const fx = side * sa * 0.6, fy = -cab * Math.cos(s2), fz = cab * Math.sin(s2);
  const k = P.forearm / Math.sqrt(fx * fx + fy * fy + fz * fz);
  inTorso(o, ha, el, fx * k, fy * k, fz * k);
}

function toe(o: Float64Array, ankle: number, t: number, dx: number, dy: number, dz: number): void {
  set(o, t, o[3 * ankle] + dx, o[3 * ankle + 1] + dy, o[3 * ankle + 2] + dz);
}

/** Stance feet on the plane y = a·x + b·z (the terrain at the root, hero-local). */
function poseGround(r: HeroRig, o: Float64Array, a: number, b: number): void {
  const v = r.speed;
  const amp = smoothstep(0.15, 1.2, v);
  const run = smoothstep(2.0, 6.5, v);
  const L = Math.min(0.62, 0.22 + 0.055 * v) * amp * (1 - 0.45 * r.crouch);
  const duty = 0.62 - 0.28 * smoothstep(1.5, 6.0, v);
  const lift = amp * (0.07 + 0.15 * run);
  const breathe = (1 - amp) * 0.006 * Math.sin(r.time * 2.3);
  const twist = 0.1 * amp * Math.cos(r.phase);
  let pelvisY = P.pelvisH - 0.3 * r.crouch - r.land - 0.08 * r.slide
    - amp * (0.018 + 0.035 * run) * (0.5 + 0.5 * Math.cos(2 * r.phase));
  const reach = 0.985 * (P.thigh + P.shin);
  let footFwdL = 0, footFwdR = 0, liftL = 0, liftR = 0;
  for (let k = 0; k < 2; k++) {
    const side = k === 0 ? -1 : 1;
    let u = (r.phase + (side > 0 ? Math.PI : 0)) / TAU;
    u -= Math.floor(u);
    let fz: number, fl: number;
    if (u < duty) {
      fz = L * (1 - 2 * u / duty);
      fl = 0;
    } else {
      const t = (u - duty) / (1 - duty);
      fz = -L + 2 * L * t * t * (3 - 2 * t);
      fl = lift * Math.sin(Math.PI * t);
    }
    const fx = side * P.hipW * (0.85 + 0.6 * r.slide);
    const fy = a * fx + b * fz + P.ankleH + fl;
    set(o, side < 0 ? J.ankleL : J.ankleR, fx, fy, fz);
    if (side < 0) { footFwdL = L > 1e-4 ? fz / L : 0; liftL = fl; } else { footFwdR = L > 1e-4 ? fz / L : 0; liftR = fl; }
    // Lower the pelvis until this foot is reachable.
    const hx = fx - side * P.hipW;
    pelvisY = Math.min(pelvisY, fy + Math.sqrt(Math.max(0, reach * reach - hx * hx - fz * fz)) + P.hipDrop);
  }
  set(o, J.pelvis, 0, pelvisY, 0);
  const lean = r.lean + 0.3 * r.crouch + 0.25 * r.tired - 0.15 * r.slide;
  const sb = Math.sin(r.bank), cb = Math.cos(r.bank);
  const spine = P.spine * (1 + breathe);
  set(o, J.chest, sb * spine, pelvisY + cb * Math.cos(lean) * spine, cb * Math.sin(lean) * spine);
  // Hips turn with the stride, the shoulders counter-rotate.
  torsoFrame(o, Math.sin(twist), 0, Math.cos(twist));
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  ik(o, J.hipL, J.kneeL, J.ankleL, P.thigh, P.shin, -0.1, 0, 1);
  ik(o, J.hipR, J.kneeR, J.ankleR, P.thigh, P.shin, 0.1, 0, 1);
  // Toes follow the ground plane in stance and point down in the swing.
  toe(o, J.ankleL, J.toeL, 0, b * P.footLen - 0.035 - Math.min(0.06, liftL * 0.5), P.footLen);
  toe(o, J.ankleR, J.toeR, 0, b * P.footLen - 0.035 - Math.min(0.06, liftR * 0.5), P.footLen);
  torsoFrame(o, Math.sin(-1.4 * twist), 0, Math.cos(-1.4 * twist));
  upperBody(o, 0.05 - 0.25 * r.tired);
  const armAmp = amp * (0.32 + 0.5 * run + 0.25 * r.sprint) * (1 - 0.6 * r.tired);
  const bend = 0.25 + 1.05 * run + 0.25 * r.sprint + 0.5 * r.crouch;
  const abduct = 0.1 + 0.9 * r.slide + 0.15 * r.crouch + 0.02 * Math.sin(r.time * 1.7) * (1 - amp);
  armFK(o, -1, -armAmp * footFwdL + 0.08 * run, abduct, bend);
  armFK(o, 1, -armAmp * footFwdR + 0.08 * run - 0.35 * r.slide, abduct, bend);
}

function poseAir(r: HeroRig, o: Float64Array, vy: number): void {
  const rising = smoothstep(-3.0, 3.0, vy);
  const flail = (1 - rising) * smoothstep(8.0, 16.0, -vy);
  set(o, J.pelvis, 0, P.pelvisH, 0);
  set(o, J.chest, 0, P.pelvisH + Math.cos(0.12) * P.spine, Math.sin(0.12) * P.spine);
  torsoFrame(o, 0, 0, 1);
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  const sc = 0.12 * flail * Math.sin(r.time * 9);
  set(o, J.ankleL, -0.1, 0.42 * rising + 0.1 * (1 - rising) + sc, 0.2 * rising + 0.06 * (1 - rising) + sc);
  set(o, J.ankleR, 0.11, 0.25 * rising + 0.04 * (1 - rising) - sc, -0.18 * rising - 0.1 * (1 - rising) - sc);
  ik(o, J.hipL, J.kneeL, J.ankleL, P.thigh, P.shin, 0, 0, 1);
  ik(o, J.hipR, J.kneeR, J.ankleR, P.thigh, P.shin, 0, 0, 1);
  toe(o, J.ankleL, J.toeL, 0, -0.05, P.footLen * 0.95);
  toe(o, J.ankleR, J.toeR, 0, -0.07, P.footLen * 0.9);
  upperBody(o, 0.1);
  armFK(o, -1, 0.9 * rising + 0.3 * (1 - rising) + 0.3 * flail * Math.sin(r.time * 11), 0.35 * rising + 1.15 * (1 - rising), 0.9 * rising + 0.35 * (1 - rising));
  armFK(o, 1, 0.5 * rising + 0.3 * (1 - rising) - 0.3 * flail * Math.sin(r.time * 11), 0.3 * rising + 1.15 * (1 - rising), 0.8 * rising + 0.35 * (1 - rising));
  // A long fall tips the body forward into a skydiving spread.
  const tilt = 1.1 * flail;
  if (tilt > 1e-3) {
    const ct = Math.cos(tilt), st = Math.sin(tilt), py = P.pelvisH;
    for (let j = 0; j < HERO_JOINTS; j++) {
      const y = o[3 * j + 1] - py, z = o[3 * j + 2];
      o[3 * j + 1] = py + y * ct - z * st;
      o[3 * j + 2] = y * st + z * ct;
    }
  }
}

const SKY_PIVOT = 1.0; // m over the root: the skydiving body turns about here (near its centre of mass)
// The nosedive (degrees): the body tipped head first, the back's arch, the neck's extension and the knees' bend.
// The face looks straight down when TILT + ARCH − NECK = 90 (the chest's axis less the neck's turn off it).
const DIVE_TILT = 160, DIVE_ARCH = -14, DIVE_NECK = 56, DIVE_KNEE = 38;
// The wind's buffeting (§6.2): two harmonics, an amplitude per bone (rad), a phase per limb so no two match.
const FLUTTER_F1 = 14.5, FLUTTER_F2 = 23.0;
const FLUTTER_HAND = 6.5 * DEG, FLUTTER_FOREARM = 3.8 * DEG, FLUTTER_CALF = 4.2 * DEG, FLUTTER_CLAVICLE = 2.2 * DEG, FLUTTER_THIGH = 1.6 * DEG;
// Any skydive buffets at least this much (× the amplitudes): the spread's 16.5 m/s alone would be ~0.15, unseen.
const FLUTTER_FLOOR = 0.5;

function flutter(r: HeroRig, q: number, amp: number, phase: number): number {
  const t = r.time;
  return q * amp * (0.65 * Math.sin(TAU * FLUTTER_F1 * t + phase) + 0.35 * Math.cos(TAU * FLUTTER_F2 * t + 1.7 * phase));
}

/**
 * A limb from joint `root` (torso frame): the upper bone (l1) along d, the lower (l2) bent from it by `bend`
 * toward (tx, ty, tz) — into `mid` and `end`.
 */
function limbFK(
  o: Float64Array, root: number, mid: number, end: number, l1: number, l2: number,
  dx: number, dy: number, dz: number, bend: number, tx: number, ty: number, tz: number,
): void {
  const d = tx * dx + ty * dy + tz * dz;
  let px = tx - dx * d, py = ty - dy * d, pz = tz - dz * d;
  const pl = Math.sqrt(px * px + py * py + pz * pz) || 1;
  px /= pl; py /= pl; pz /= pl;
  const c = Math.cos(bend), sn = Math.sin(bend);
  inTorso(o, mid, root, dx * l1, dy * l1, dz * l1);
  inTorso(o, end, mid, (dx * c + px * sn) * l2, (dy * c + py * sn) * l2, (dz * c + pz * sn) * l2);
}

/**
 * Skydiving, the body's control surfaces (§6.1): built upright, then tipped about SKY_PIVOT — belly to
 * earth spread (dive 0), head first along the fall (dive 1) — and banked into the turn. Turning, the outside
 * arm reaches out and forward into the wind while the inside one tucks (ailerons), the legs scissor (rudders),
 * the spine twists and bends into it; pushing forward flattens the arch and sweeps the arms back. A flare
 * throws everything wide, chest up. Hands, forearms, calves and shoulders flutter with the dynamic pressure.
 */
function poseSkydive(r: HeroRig, o: Float64Array, hv: number, vy: number): void {
  const a = r.dive, f = r.flare, ix = r.steerX, iy = r.steerY;
  const v = Math.sqrt(hv * hv + vy * vy) / SKYDIVE.diveTerminal;
  const q = Math.min(1.25, FLUTTER_FLOOR + (1 - FLUTTER_FLOOR) * v * v);
  const mix = (x: number, flared: number) => x + (flared - x) * f;
  // Spine: arch (negative pitch: the chest back, the belly into the wind), twist and bend into the turn. Diving
  // the back stays gently arched, the body a shallow curve from the head down to the trailing knees.
  const camber = mix((-22 + 14 * iy) * (1 - a) + DIVE_ARCH * a, -32) * DEG;
  const twist = ix * 24 * DEG, bend = ix * 16 * DEG;
  set(o, J.pelvis, 0, P.pelvisH, 0);
  set(o, J.chest, Math.sin(bend) * P.spine, P.pelvisH + Math.cos(camber) * Math.cos(bend) * P.spine, Math.sin(camber) * P.spine);
  torsoFrame(o, Math.sin(twist), 0, Math.cos(twist));
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  upperBody(o, 0.1);
  if (a > 0) {
    // Diving, the neck extends so the face looks down the fall at the ground (head back off the chest's axis).
    const e = DIVE_NECK * DEG, cu = Math.cos(e) * P.neck, cf = -Math.sin(e) * P.neck + 0.025;
    const c3 = 3 * J.chest, h3 = 3 * J.head;
    for (let k = 0; k < 3; k++) o[h3 + k] += (o[c3 + k] + F[3 + k] * cu + F[6 + k] * cf - o[h3 + k]) * a;
  }
  for (let k = 0; k < 2; k++) {
    const sd = k === 0 ? -1 : 1, ph = k === 0 ? 0 : 1.3, pl = k === 0 ? 2.7 : 4.1;
    // Arms: sweep (+ forward) and abduction from hanging; the elbow folds the forearm toward the head.
    // (tucked into the dive the arms lie straight along the sides, parallel to the body: no aileron then)
    const steer = 1 - a;
    const sweep = mix(-6 * a + (-sd * ix * 28 - iy * 18) * steer, 18) * DEG;
    const abduct = mix(80 + (5 - 80) * a - sd * ix * 22 * steer, 105) * DEG + flutter(r, q, FLUTTER_CLAVICLE, ph + 0.6) * steer;
    const elbow = 75 * steer * DEG + (flutter(r, q, FLUTTER_FOREARM, ph + 0.3) + flutter(r, q, FLUTTER_HAND, ph)) * (0.5 + 0.5 * steer);
    const ca = Math.cos(abduct);
    limbFK(o, sd < 0 ? J.shoulderL : J.shoulderR, sd < 0 ? J.elbowL : J.elbowR, sd < 0 ? J.handL : J.handR, P.upperArm, P.forearm,
      sd * Math.sin(abduct), -ca * Math.cos(sweep), ca * Math.sin(sweep), elbow, 0, 1, 0);
    // Legs: frog spread, the rudder's scissor, the knees folding the shins back; toes pointed.
    const spread = mix(34 + (4 - 34) * a, 45) * DEG;
    // (diving: the hips open a little and the knees bend, the shins trailing behind the arch, one more than the other)
    const pitch = (sd * ix * 18 * (1 - 0.5 * a) + 24 * f - 12 * a) * DEG + flutter(r, q, FLUTTER_THIGH, pl + 0.9);
    const knee = (65 + (DIVE_KNEE + sd * 8 - 65) * a) * DEG + flutter(r, q, FLUTTER_CALF, pl);
    const cs = Math.cos(spread);
    const ank = sd < 0 ? J.ankleL : J.ankleR;
    limbFK(o, sd < 0 ? J.hipL : J.hipR, sd < 0 ? J.kneeL : J.kneeR, ank, P.thigh, P.shin,
      sd * Math.sin(spread), -cs * Math.cos(pitch), cs * Math.sin(pitch), knee, 0, 0, -1);
    const kn = 3 * (sd < 0 ? J.kneeL : J.kneeR), an = 3 * ank;
    const sx = o[an] - o[kn], sy = o[an + 1] - o[kn + 1], sz = o[an + 2] - o[kn + 2], sl = Math.sqrt(sx * sx + sy * sy + sz * sz) || 1;
    toe(o, ank, sd < 0 ? J.toeL : J.toeR, sx / sl * P.footLen, sy / sl * P.footLen, sz / sl * P.footLen);
  }
  // Tip over (from vertical: near level spread, tipping down as the stick pushes on; head first diving; the
  // flare sits up) and bank (r.skyBank); a nosedive spins rather than banks.
  const push = Math.min(1, Math.sqrt(ix * ix + iy * iy));
  const tilt = ((82 - 12 * push) * (1 - a) + DIVE_TILT * a - 25 * f) * DEG;
  rotXZ(tilt, r.skyBank * (1 - a));
  for (let j = 0; j < HERO_JOINTS; j++) turnP(o, 3 * j, 0, SKY_PIVOT, 0);
}

function poseGlide(r: HeroRig, o: Float64Array): void {
  const t = r.time;
  set(o, J.pelvis, 0, 0.9, -0.03);
  set(o, J.chest, 0, 0.9 + Math.cos(0.06) * P.spine, -0.03 - Math.sin(0.06) * P.spine);
  torsoFrame(o, 0, 0, 1);
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  upperBody(o, 0.12);
  // Hands up on the handles, arms nearly straight.
  const reach = 0.97 * (P.upperArm + P.forearm);
  for (let k = 0; k < 2; k++) {
    const side = k === 0 ? -1 : 1;
    const sh = side < 0 ? J.shoulderL : J.shoulderR;
    const dx = side * 0.06, dy = 1.0, dz = 0.12;
    const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
    set(o, side < 0 ? J.handL : J.handR,
      o[3 * sh] + dx / l * reach, o[3 * sh + 1] + dy / l * reach, o[3 * sh + 2] + dz / l * reach);
    ik(o, sh, side < 0 ? J.elbowL : J.elbowR, side < 0 ? J.handL : J.handR, P.upperArm, P.forearm, side, 0, -0.4);
    // Legs dangle and sway in the airflow.
    const hip = side < 0 ? J.hipL : J.hipR;
    const ank = side < 0 ? J.ankleL : J.ankleR;
    set(o, ank, side * 0.1, 0.1 + 0.04 * Math.sin(t * 2.6 + side), -0.1 + 0.06 * Math.sin(t * 2.1 + 1.3 * side));
    ik(o, hip, side < 0 ? J.kneeL : J.kneeR, ank, P.thigh, P.shin, 0, 0, 1);
    toe(o, ank, side < 0 ? J.toeL : J.toeR, 0, -0.08, P.footLen * 0.85);
  }
  const hl = 3 * J.handL, hr = 3 * J.handR;
  set(o, J.glider, 0.5 * (o[hl] + o[hr]), 0.5 * (o[hl + 1] + o[hr + 1]) + 0.42, 0.5 * (o[hl + 2] + o[hr + 2]));
}

/**
 * On horseback (horse/horses.ts seats the root so the pelvis is at RIDE_PELVIS over the saddle): sitting astride,
 * the knees out round the barrel and a little forward, the hands low before the withers on the reins, leaning into
 * the pace.
 */
function poseRide(r: HeroRig, o: Float64Array): void {
  const lean = 0.06 + 0.25 * smoothstep(2, 12, r.speed);
  set(o, J.pelvis, 0, 0.9, 0);
  set(o, J.chest, 0, 0.9 + Math.cos(lean) * P.spine, Math.sin(lean) * P.spine);
  torsoFrame(o, 0, 0, 1);
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  upperBody(o, 0.1 - lean * 0.5);
  for (let k = 0; k < 2; k++) {
    const side = k === 0 ? -1 : 1;
    const ank = side < 0 ? J.ankleL : J.ankleR;
    set(o, ank, side * 0.37, 0.3, 0.06);
    ik(o, side < 0 ? J.hipL : J.hipR, side < 0 ? J.kneeL : J.kneeR, ank, P.thigh, P.shin, side * 0.8, 0, 0.6);
    toe(o, ank, side < 0 ? J.toeL : J.toeR, side * 0.02, -0.03, P.footLen * 0.85);
    const sh = side < 0 ? J.shoulderL : J.shoulderR, ha = side < 0 ? J.handL : J.handR;
    set(o, ha, side * 0.11, 1.12 + 0.05 * Math.cos(lean), 0.36 + 0.1 * lean);
    ik(o, sh, side < 0 ? J.elbowL : J.elbowR, ha, P.upperArm, P.forearm, side * 0.6, -0.6, -0.5);
  }
}

// ---- climbing a tree -----------------------------------------------------------
//
// Four holds on the bark, each kept where it was put in the world (an angle round the trunk's axis and a
// height) while the body moves past it. Hands and feet go in diagonal pairs (left hand with right foot,
// right hand with left foot): a pair steps once one of its limbs is stretched too far from its socket
// (where it rests relative to the body) and only while the other pair holds, each step in four phases
// (the pelvis shifting onto the support side to unload it, the heel or wrist peeling off, a swing round
// the bark, contact and a little slip as the weight comes on). The holds are coordinates on the
// cylinder, so a swing follows the bark's curve rather than cutting through it. Fast moves (a climb
// jump, a slide, a grab's slip) carry the limbs with the body instead.

/** Limbs, in HeroRig.limbs order; the diagonal pairs are (handL, footR) and (handR, footL). */
const LIMB = { handL: 0, handR: 1, footL: 2, footR: 3 } as const;
const LIMB_STRIDE = 7;
// HeroRig.limbs per limb: the step's start and its hold (world, fp64), the step's progress (≥ 1: planted on the hold).
const L_START = 0, L_HOLD = 3, L_U = 6;
// HeroRig.limbAt per limb: where it is this frame on the bark (angle round the axis off the root's, height over
// the root, distance out from the bark) and the heel's peel.
const A_ANG = 0, A_H = 1, A_OFF = 2, A_PEEL = 3;
const HAND_H = 1.3, FOOT_H = 0.5;   // the sockets' heights over the root
const HAND_OFF = 0.045, FOOT_OFF = 0.075; // the hand's and the ankle's height off the bark
const WRAP_BEHIND = 0.1;            // m: how far behind a pole's front the hands reach round it
const CONTACT = 0.85;                // a step's contact phase starts here (0..1)…
const HANDOFF = 0.82;                // … and the other pair may start once a step is this far through
// On a wall leaning back (leanUpZ m in per m up) the hands take their holds lower and the chest hugs the wall closer
// — the bark at the hands' height is further in — so the body is not forever stretching to reach them.
const LEAN_LOW = 0.25, LEAN_HUG = 0.12;
const DOWN = 0.25;                   // m the hands' sockets ride lower climbing down (the body drops away from them)
const ALTERNATE = 0.7;               // the pair that did not step last goes at this share of due

function pairOf(i: number): number {
  return i === LIMB.handL || i === LIMB.footR ? 0 : 1;
}

/** How far round a thin pole the limbs wrap: 1 under poleWrap (less a hysteresis), 0 over it. */
function poleWrap(radius: number): number {
  return 1 - smoothstep(CLIMB.poleWrap - 0.04, CLIMB.poleWrap + 0.04, radius);
}

/**
 * Limb i's socket on bark of the given radius — its angle round the axis off the root's, its height (the
 * hands lower by `down` 0..1 climbing down) — and how far round the axis it may hold at most, into SOCK.
 */
const SOCK = new Float64Array(3);
function socket(i: number, radius: number, down: number): void {
  const pole = poleWrap(radius);
  const side = i === LIMB.handL || i === LIMB.footL ? -1 : 1;
  if (i <= LIMB.handR) {
    // (on a pole the hands reach round past its sides — up to ±110° on the thinnest — as far as the arms
    // reach behind it: no more than WRAP_BEHIND behind the plane the chest hugs)
    const most = Math.min(1.92, Math.acos(Math.max(-1, Math.min(1, (radius - WRAP_BEHIND) / (radius + HAND_OFF)))));
    SOCK[0] = side * Math.min(most, Math.max(0.2 / radius, pole * most));
    SOCK[1] = HAND_H - DOWN * down - LEAN_LOW * Math.max(0, leanUpZ);
    SOCK[2] = most;
  } else {
    // (and the feet clamp its sides)
    SOCK[0] = side * Math.min(1.4, Math.max(0.17 / radius, pole * 1.3));
    SOCK[1] = FOOT_H;
    SOCK[2] = 1.4;
  }
}

const FLAT_WALL = 30; // m: a surface's radius when none is known (a flat wall)

/** The limbs at their sockets on bark of radius R, taken as holds (no surface to step on known). */
function holdSockets(r: HeroRig, R: number): void {
  r.treeR = R;
  for (let i = 0; i < 4; i++) {
    socket(i, R, 0);
    const c = 4 * i;
    r.limbAt[c + A_ANG] = SOCK[0]; r.limbAt[c + A_H] = SOCK[1];
    r.limbAt[c + A_OFF] = i <= LIMB.handR ? HAND_OFF : FOOT_OFF; r.limbAt[c + A_PEEL] = 0;
  }
  r.holds = true;
}

// The trunk round the hips, hero-local (barkFrame): how its radius and its centre change going up from the
// hips and going down from them (m per m; its profile bends, so a slope each way).
let taperUp = 0, leanUpX = 0, leanUpZ = 0, taperDown = 0, leanDownX = 0, leanDownZ = 0;
const BARK_H = 0.9; // the cross-section the hero holds is the hips' (locomotion.ts HIPS) over the root

/** The trunk's taper and lean above and below the hips in the hero's axes (right, forward), for the bark's points. */
function barkFrame(s: HeroState): void {
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
  taperUp = s.climbTaper;
  leanUpX = s.climbLeanX * cy - s.climbLeanZ * sy;
  leanUpZ = s.climbLeanX * sy + s.climbLeanZ * cy;
  taperDown = s.climbTaperDown;
  leanDownX = s.climbLeanDownX * cy - s.climbLeanDownZ * sy;
  leanDownZ = s.climbLeanDownX * sy + s.climbLeanDownZ * cy;
}

/** The trunk's axis (hero-local x, z) and radius `dh` metres above the hips (below: negative), into AXIS; R at the hips. */
const AXIS = new Float64Array(3);
function barkAxis(R: number, dh: number): void {
  const up = dh > 0;
  AXIS[0] = (up ? leanUpX : leanDownX) * dh;
  AXIS[1] = R + (up ? leanUpZ : leanDownZ) * dh;
  AXIS[2] = Math.max(0.02, R + (up ? taperUp : taperDown) * dh);
}

/**
 * Point on the bark at angle a off the root's and height h, `off` out from it, hero-local: the trunk of
 * radius R at the hips, its axis at (0, R) there, tapering and leaning (barkFrame) above and below.
 */
function onBark(o: Float64Array, j: number, R: number, a: number, h: number, off: number): void {
  barkAxis(R, h - BARK_H);
  const rr = AXIS[2] + off;
  set(o, j, AXIS[0] + rr * Math.sin(a), h, AXIS[1] - rr * Math.cos(a));
}

/** World point g[at..at+2] on the bark round hero s (radius R at the hips), into BARK: angle, height, distance out from it. */
const BARK = new Float64Array(3);
function toBark(s: HeroState, R: number, g: Float64Array, at: number): void {
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
  const dx = g[at] - s.x, dz = g[at + 2] - s.z;
  const x = cy * dx - sy * dz, z = sy * dx + cy * dz, h = g[at + 1] - (s.y - PLAYER.groundBias);
  barkAxis(R, h - BARK_H);
  const cx = AXIS[0], cz = AXIS[1];
  BARK[0] = Math.atan2(x - cx, cz - z);
  BARK[1] = h;
  BARK[2] = Math.sqrt((x - cx) * (x - cx) + (z - cz) * (z - cz)) - AXIS[2];
}

/** The point at angle a, height h and `off` out from the bark round hero s, into the world at g[at..at+2]. */
const PT = new Float64Array(3);
function fromBark(s: HeroState, R: number, a: number, h: number, off: number, g: Float64Array, at: number): void {
  onBark(PT, 0, R, a, h, off);
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
  g[at] = s.x + cy * PT[0] + sy * PT[2];
  g[at + 1] = s.y - PLAYER.groundBias + PT[1];
  g[at + 2] = s.z - sy * PT[0] + cy * PT[2];
}

/**
 * Advance the holds by dt from the hero on its trunk: step the pairs, run the steps' phases, and put
 * each limb on the bark for the pose (HeroRig.limbAt). A hold is a point in the world, so it stays put
 * however the body moves past it; a swing goes between its ends round the bark, on the trunk's taper
 * and lean at the heights it passes.
 */
function climbGait(r: HeroRig, s: HeroState, dt: number, snap: boolean): void {
  const T = CLIMB;
  barkFrame(s);
  const R = s.climbR, g = r.limbs, at = r.limbAt;
  const reach = R + climbClearance(R);
  // The root's motion on the bark (m/s at the hips): to its right and up.
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
  const vT = ((s.vx * cy - s.vz * sy) * reach) / Math.max(R, 0.05), vB = s.vy;
  const speed = Math.sqrt(vT * vT + vB * vB);
  const down = smoothstep(0.3, 1.2, -vB);
  // (new holds: the first frame on a trunk, a snap, or a body far from its holds — a teleport, another trunk)
  let place = snap || !r.holds;
  for (let i = 0; i < 4 && !place; i++) {
    const b = LIMB_STRIDE * i + L_HOLD;
    // (a teleport, another surface: a hold far out of any reach — a hand on a wall leaning back is metres in)
    const hx = g[b] - s.x, hy = g[b + 1] - s.y, hz = g[b + 2] - s.z;
    if (hx * hx + hy * hy + hz * hz > 16) place = true;
  }
  r.treeR = R;
  r.holds = true;
  if (place || s.grip > 0 || s.sliding || speed > 2.2) {
    // (placed at the sockets; on the move with the body, eased toward them: a leap's hands up, feet down)
    const k = place ? 1 : follow(false, 18, dt);
    const up = smoothstep(1.0, 3.0, vB) - 0.5 * smoothstep(-1.5, -3.5, vB);
    for (let i = 0; i < 4; i++) {
      socket(i, R, 0);
      const b = LIMB_STRIDE * i, c = 4 * i;
      const off = i <= LIMB.handR ? HAND_OFF : FOOT_OFF;
      const lift = i <= LIMB.handR ? 0.28 * up : -0.12 * up;
      if (place) { at[c + A_ANG] = SOCK[0]; at[c + A_H] = SOCK[1] + lift; at[c + A_OFF] = off; at[c + A_PEEL] = 0; }
      at[c + A_ANG] += (SOCK[0] - at[c + A_ANG]) * k;
      at[c + A_H] += (SOCK[1] + lift - at[c + A_H]) * k;
      at[c + A_OFF] += (off - at[c + A_OFF]) * k;
      at[c + A_PEEL] *= 1 - k;
      fromBark(s, R, at[c + A_ANG], at[c + A_H], at[c + A_OFF], g, b + L_HOLD);
      snapToClimbed(s, g, b + L_HOLD, off);
      if (place) { toBark(s, R, g, b + L_HOLD); at[c + A_ANG] = BARK[0]; at[c + A_H] = BARK[1]; at[c + A_OFF] = BARK[2]; }
      g[b + L_U] = 1;
    }
    return;
  }
  // Where each planted limb is on the bark now (its hold, seen from where the body has got to).
  let lifted0 = false, lifted1 = false, moving0 = false, moving1 = false;
  for (let i = 0; i < 4; i++) {
    const b = LIMB_STRIDE * i, c = 4 * i;
    const u = g[b + L_U];
    // (late in its step a limb is coming down onto its hold: the other pair may begin, its own limbs
    // still holding through their pre-load, so one pair is always on the bark)
    if (u < HANDOFF) { if (pairOf(i) === 0) lifted0 = true; else lifted1 = true; }
    if (u < 1) { if (pairOf(i) === 0) moving0 = true; else moving1 = true; continue; }
    toBark(s, R, g, b + L_HOLD);
    at[c + A_ANG] = BARK[0]; at[c + A_H] = BARK[1]; at[c + A_OFF] = BARK[2]; at[c + A_PEEL] = 0;
  }
  // (every step takes the spec's stepTime; a stride runs from the stretch behind to the lead ahead: the body
  // following the holds keeps each in reach, reachHolds)
  const stepTime = T.stepTime;
  // How overdue each pair is: a limb's stretch on the bark from its socket (MaxLimbStretch), or a hold
  // trailing round past the furthest the limb wraps (it is leaving the arm's or leg's reach).
  let worst0 = 0, worst1 = 0;
  for (let i = 0; i < 4; i++) {
    if (g[LIMB_STRIDE * i + L_U] < 1) continue;
    socket(i, R, down);
    const c = 4 * i;
    const ra = R + (i <= LIMB.handR ? HAND_OFF : FOOT_OFF);
    const stretch = i <= LIMB.handR ? T.handStretch : T.footStretch;
    const ang = at[c + A_ANG];
    const px = ra * Math.sin(ang), pz = R - ra * Math.cos(ang);
    const qx = ra * Math.sin(SOCK[0]), qz = R - ra * Math.cos(SOCK[0]);
    const dh = at[c + A_H] - SOCK[1];
    let over = Math.sqrt((px - qx) * (px - qx) + (pz - qz) * (pz - qz) + dh * dh) / stretch;
    const round = ang * SOCK[0] > 0 ? Math.abs(ang) : 0;
    if (round > SOCK[2]) over = Math.max(over, ((round - SOCK[2]) * ra) / (0.15 * stretch));
    if (pairOf(i) === 0) worst0 = Math.max(worst0, over); else worst1 = Math.max(worst1, over);
  }
  // At rest a stretched limb settles back under the body; on the move it waits for its stretch. The pairs
  // take turns: the one that did not step last goes as soon as it is well on its way to due (a pair that
  // kept qualifying first would starve the other until its arm or leg was straight).
  const due = speed < 0.05 ? 0.3 : 1;
  let pick = -1;
  if (!lifted0 && !lifted1) {
    const next = 1 - r.lastPair, wNext = next === 0 ? worst0 : worst1, wLast = next === 0 ? worst1 : worst0;
    const still = (p: number) => !(p === 0 ? moving0 : moving1);
    pick = still(next) && wNext > ALTERNATE * due ? next : still(r.lastPair) && wLast > due ? r.lastPair : -1;
  }
  if (pick >= 0) {
    r.lastPair = pick;
    // (to land ahead of the socket the way the body moves — to its right is a larger angle — never
    // further round than the limb wraps)
    const lt = speed > 0.05 ? (T.lead * vT) / speed : 0, lb = speed > 0.05 ? (T.lead * vB) / speed : 0;
    for (let i = 0; i < 4; i++) {
      if (pairOf(i) !== pick) continue;
      socket(i, R, down);
      const b = LIMB_STRIDE * i;
      const off = i <= LIMB.handR ? HAND_OFF : FOOT_OFF;
      g[b + L_START] = g[b + L_HOLD]; g[b + L_START + 1] = g[b + L_HOLD + 1]; g[b + L_START + 2] = g[b + L_HOLD + 2];
      const a = Math.max(-SOCK[2], Math.min(SOCK[2], SOCK[0] + lt / (R + off)));
      fromBark(s, R, a, SOCK[1] + lb, off, g, b + L_HOLD);
      snapToClimbed(s, g, b + L_HOLD, off); // (onto the surface as it really is there)
      g[b + L_U] = 0;
    }
  }
  // The steps' phases, between the step's ends as they lie on the bark seen from the body now.
  for (let i = 0; i < 4; i++) {
    const b = LIMB_STRIDE * i, c = 4 * i;
    if (g[b + L_U] >= 1) continue;
    const u = Math.min(1, g[b + L_U] + dt / stepTime);
    g[b + L_U] = u;
    toBark(s, R, g, b + L_START);
    const a0 = BARK[0], h0 = BARK[1], o0 = BARK[2];
    toBark(s, R, g, b + L_HOLD);
    const a1 = a0 + wrapAngle(BARK[0] - a0), h1 = BARK[1], o1 = BARK[2];
    let a: number, h: number, off: number, peel: number;
    if (u < 0.18) {
      // pre-load: the hold kept while the pelvis shifts off it
      a = a0; h = h0; off = o0; peel = 0;
    } else if (u < 0.32) {
      // peel-off: the heel (or wrist) lifts, the toes (fingers) still on the bark
      peel = (u - 0.18) / 0.14;
      a = a0; h = h0; off = o0 + 0.03 * peel;
    } else if (u < CONTACT) {
      // swing round the bark, clear of it by a parabola, to just above the new hold
      const w = (u - 0.32) / (CONTACT - 0.32), e = w * w * (3 - 2 * w);
      a = a0 + (a1 - a0) * e;
      h = h0 + (h1 + T.slip - h0) * e;
      off = o0 + (o1 - o0) * e + 0.03 + 4 * T.stepArc * w * (1 - w);
      peel = 1 - e;
    } else {
      // contact: a little slip down the bark as it takes the weight
      const w = (u - CONTACT) / (1 - CONTACT);
      a = a1; h = h1 + T.slip * (1 - w * w * (3 - 2 * w)); off = o1 + 0.03 * (1 - w); peel = 0;
    }
    at[c + A_ANG] = a; at[c + A_H] = h; at[c + A_OFF] = off; at[c + A_PEEL] = peel;
  }
}

const SETTLE = 20; // 1/s: the pelvis and chest follow where the holds put them on a critically damped spring

/**
 * The pelvis and chest of the pose being built (o) settle on a spring toward where the holds put them: a step's
 * weight shift, a hold taken or a stop come through as a smooth sway, not a jolt. (The spine's length kept; the
 * reach is found from where they settle, reachHolds.)
 */
function settleBody(r: HeroRig, o: Float64Array, dt: number, snap: boolean): void {
  const b = r.body;
  if (snap || b[12] !== b[12] || dt <= 0) {
    for (let q = 0; q < 3; q++) { b[q] = o[3 * J.pelvis + q]; b[3 + q] = o[3 * J.chest + q]; }
    b.fill(0, 6, 25);
    b[12] = 1;
    return;
  }
  for (let q = 0; q < 6; q++) spring(b, q, 6 + q, q < 3 ? o[3 * J.pelvis + q] : o[3 * J.chest + q - 3], SETTLE, dt);
  setSpine(o, b[0], b[1], b[2], b[3], b[4], b[5]);
}

/** b[x] (its velocity b[v]) a step dt along a critically damped spring of rate w toward target: x'' = w²(target − x) − 2w x'. */
function spring(b: Float64Array, x: number, v: number, target: number, w: number, dt: number): void {
  const e = Math.exp(-w * dt), d = b[x] - target, c = b[v] + w * d;
  b[x] = target + (d + c * dt) * e;
  b[v] = (b[v] - w * c * dt) * e;
}

/** The pelvis at (px, py, pz) and the chest toward (cx, cy, cz), a spine's length from it. */
function setSpine(o: Float64Array, px: number, py: number, pz: number, cx: number, cy: number, cz: number): void {
  const ex = cx - px, ey = cy - py, ez = cz - pz;
  const el = P.spine / Math.sqrt(ex * ex + ey * ey + ez * ez);
  set(o, J.pelvis, px, py, pz);
  set(o, J.chest, px + ex * el, py + ey * el, pz + ez * el);
}

const PULL = new Float64Array(6); // (scratch: the reach's pull on the pelvis and chest)
const REACH_SETTLE = 28; // 1/s: the body leans to its holds' reach (reachHolds) on a spring of its own, a touch quicker

/**
 * Reach the holds (reachHolds) with the body, but take the pull the reach puts on the pelvis and chest on a spring:
 * a hold taken changes the pull at once, the body leans into it smoothly — and whatever the spring has not yet
 * caught up with, reached at once after all.
 */
function reachSettled(r: HeroRig, o: Float64Array, R: number, roll: number, dt: number): void {
  const b = r.body;
  const p0x = o[3 * J.pelvis], p0y = o[3 * J.pelvis + 1], p0z = o[3 * J.pelvis + 2];
  const c0x = o[3 * J.chest], c0y = o[3 * J.chest + 1], c0z = o[3 * J.chest + 2];
  reachHolds(o, R, roll);
  PULL[0] = o[3 * J.pelvis] - p0x; PULL[1] = o[3 * J.pelvis + 1] - p0y; PULL[2] = o[3 * J.pelvis + 2] - p0z;
  PULL[3] = o[3 * J.chest] - c0x; PULL[4] = o[3 * J.chest + 1] - c0y; PULL[5] = o[3 * J.chest + 2] - c0z;
  for (let q = 0; q < 6; q++) spring(b, 13 + q, 19 + q, PULL[q], REACH_SETTLE, dt);
  setSpine(o, p0x + b[13], p0y + b[14], p0z + b[15], c0x + b[16], c0y + b[17], c0z + b[18]);
  // (and a hold the spring has not caught up with yet is reached all the same: a hand never leaves its hold)
  reachHolds(o, R, roll);
}

const TORSO_OFF = 0.12; // m: the pelvis and the chest stay this far off the bark (the body's half depth)
const REACH_MAX = 0.97; // a limb is reached at this share of its length (the IK's knee or elbow kept bent)
const REACH_IT = 12;
/** Hands and ankles of the pose being built (they are where climbGait put them): set by poseClimb before reachHolds. */
const ENDS = [J.handL, J.handR, J.ankleL, J.ankleR] as const;

/**
 * The body follows its holds. A long stride leaves a hand high above while the other side's foot is
 * still far below (a stride of stretch + lead at the climbing pace): the pelvis and chest placed at the
 * weighted centre would leave that arm or leg short of its hold. As a climber does, the body goes to
 * them: the chest (the spine pivoting on the pelvis) toward a hand out of reach, the pelvis toward a
 * foot out of reach, each pulling the other a little, both hugging in toward the bark — never closer
 * than TORSO_OFF to it. Relaxed a few times; the ends (o's hands and ankles) are the holds.
 */
function reachHolds(o: Float64Array, R: number, roll: number): void {
  const arm = (P.upperArm + P.forearm) * REACH_MAX, leg = (P.thigh + P.shin) * REACH_MAX;
  for (let it = 0; it < REACH_IT; it++) {
    torsoFrame(o, 0, 0, 1);
    inTorso(o, J.shoulderL, J.chest, -P.shoulderW, -P.shoulderDrop, 0);
    inTorso(o, J.shoulderR, J.chest, P.shoulderW, -P.shoulderDrop, 0);
    inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
    inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
    o[3 * J.hipL + 1] += P.hipW * Math.sin(roll);
    o[3 * J.hipR + 1] -= P.hipW * Math.sin(roll);
    let ax = 0, ay = 0, az = 0, lx = 0, ly = 0, lz = 0, na = 0, nl = 0;
    for (let i = 0; i < 4; i++) {
      const e = ENDS[i], hand = i < 2;
      const root = hand ? (i === 0 ? J.shoulderL : J.shoulderR) : (i === 2 ? J.hipL : J.hipR);
      const dx = o[3 * e] - o[3 * root], dy = o[3 * e + 1] - o[3 * root + 1], dz = o[3 * e + 2] - o[3 * root + 2];
      const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const over = dl - (hand ? arm : leg);
      if (over <= 0 || dl < 1e-6) continue;
      const k = over / dl;
      if (hand) { ax += dx * k; ay += dy * k; az += dz * k; na++; } else { lx += dx * k; ly += dy * k; lz += dz * k; nl++; }
    }
    if (na === 0 && nl === 0) return;
    if (na) { ax /= na; ay /= na; az /= na; }
    if (nl) { lx /= nl; ly /= nl; lz /= nl; }
    // (the chest takes the arms' pull and a share of the legs', the pelvis the other way round)
    const c = 3 * J.chest, p = 3 * J.pelvis;
    o[c] += ax + 0.4 * lx; o[c + 1] += ay + 0.4 * ly; o[c + 2] += az + 0.4 * lz;
    o[p] += lx + 0.4 * ax; o[p + 1] += ly + 0.4 * ay; o[p + 2] += lz + 0.4 * az;
    // a spine's length apart again, both off the bark by the body's depth at least
    const ex = o[c] - o[p], ey = o[c + 1] - o[p + 1], ez = o[c + 2] - o[p + 2];
    const el = Math.sqrt(ex * ex + ey * ey + ez * ez);
    const mx = 0.5 * (o[c] + o[p]), my = 0.5 * (o[c + 1] + o[p + 1]), mz = 0.5 * (o[c + 2] + o[p + 2]);
    const hs = (0.5 * P.spine) / el;
    o[c] = mx + ex * hs; o[c + 1] = my + ey * hs; o[c + 2] = mz + ez * hs;
    o[p] = mx - ex * hs; o[p + 1] = my - ey * hs; o[p + 2] = mz - ez * hs;
    for (const j of [J.chest, J.pelvis]) {
      barkAxis(R, o[3 * j + 1] - BARK_H);
      const cx = AXIS[0], cz = AXIS[1], rr = AXIS[2] + TORSO_OFF;
      const qx = o[3 * j] - cx, qz = o[3 * j + 2] - cz, ql = Math.sqrt(qx * qx + qz * qz);
      if (ql < rr && ql > 1e-6) { o[3 * j] = cx + (qx / ql) * rr; o[3 * j + 2] = cz + (qz / ql) * rr; }
    }
  }
}

/**
 * Facing the trunk's axis (+z, at distance R) with the root on its bark: the limbs on their holds
 * (climbGait), the pelvis at the weighted centre of the holds and the root, shifted onto the support
 * side through a step and rolled toward the higher knee, the spine counter-bent to keep the head level,
 * knees and elbows flared out (the frog stance hugging the trunk). Vaulting (HeroState.mantle) the hands
 * go onto the top, the chest over them and the feet up under the body.
 */
function poseClimb(r: HeroRig, s: HeroState, o: Float64Array, dt: number, snap: boolean): void {
  const T = CLIMB;
  if (s.move === 'climb') barkFrame(s);
  const R = r.treeR, g = r.limbs, at = r.limbAt;
  // (the body's clearance from the surface along its normal: on a wall leaning back the bark at the hands' height
  // is further in, and the body hugs it — its clearance measured level would hold it off too far)
  const d = climbClearance(R) / Math.sqrt(1 + leanUpZ * leanUpZ);
  // The weighted centre: 0.35 of the feet's, 0.15 of the hands', 0.5 the root's.
  let fh = 0, hh = 0, fx = 0, hx = 0;
  for (let i = 0; i < 4; i++) {
    const c = 4 * i;
    const x = (R + at[c + A_OFF]) * Math.sin(at[c + A_ANG]);
    if (i <= LIMB.handR) { hh += 0.5 * at[c + A_H]; hx += 0.5 * x; } else { fh += 0.5 * at[c + A_H]; fx += 0.5 * x; }
  }
  // The weight shifts onto the support side as a pair unloads (left hand and right foot stepping: onto
  // the left foot), and dips, from the pre-load to the contact.
  const uA = g[LIMB_STRIDE * LIMB.handL + L_U], uB = g[LIMB_STRIDE * LIMB.handR + L_U];
  const wA = uA < 1 ? smoothstep(0, 0.18, uA) * (1 - smoothstep(0.85, 1, uA)) : 0;
  const wB = uB < 1 ? smoothstep(0, 0.18, uB) * (1 - smoothstep(0.85, 1, uB)) : 0;
  const k = follow(snap, 12, dt);
  r.shiftX += (T.sway * (wB - wA) - r.shiftX) * k;
  r.shiftY += (-T.bob * Math.max(wA, wB) - r.shiftY) * k;
  const px = 0.35 * fx + 0.15 * hx + r.shiftX;
  const py = 0.92 + 0.35 * (fh - FOOT_H) + 0.15 * (hh - HAND_H) + r.shiftY;
  const pa = Math.asin(Math.max(-0.9, Math.min(0.9, px / (R + d))));
  onBark(o, J.pelvis, R, pa, py, d);
  // Roll toward the higher foot (18° per metre, at most 12°); the spine bends back to keep the head level.
  const footL = at[4 * LIMB.footL + A_H], footR = at[4 * LIMB.footR + A_H];
  const roll = Math.max(-12 * DEG, Math.min(12 * DEG, (footL - footR) * 18 * DEG));
  // The chest leans in to reach high holds and out over low ones.
  const lean = Math.max(-1, Math.min(1, (hh - HAND_H) / 0.3));
  const ca = Math.asin(Math.max(-0.9, Math.min(0.9, (px + P.spine * Math.sin(0.35 * roll)) / (R + d))));
  onBark(o, J.chest, R, ca, py + P.spine * 0.985, d + 0.02 - 0.07 * lean - LEAN_HUG * Math.max(0, leanUpZ));
  {
    // (that way from the pelvis, a spine's length)
    const ex = o[3 * J.chest] - o[3 * J.pelvis], ey = o[3 * J.chest + 1] - py, ez = o[3 * J.chest + 2] - o[3 * J.pelvis + 2];
    const el = P.spine / Math.sqrt(ex * ex + ey * ey + ez * ez);
    set(o, J.chest, o[3 * J.pelvis] + ex * el, py + ey * el, o[3 * J.pelvis + 2] + ez * el);
  }
  settleBody(r, o, dt, snap);
  // Vaulting onto the top.
  const m = s.move === 'climb' ? s.mantle : 0;
  if (m === 0) {
    for (let i = 0; i < 4; i++) onBark(o, ENDS[i], R, at[4 * i + A_ANG], at[4 * i + A_H], at[4 * i + A_OFF]);
    reachSettled(r, o, R, roll, dt);
  }
  let up = 0, topY = 0, topZ = 0, topX = 0;
  if (m > 0) {
    const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
    const dx = s.climbX - s.x, dz = s.climbZ - s.z;
    topX = cy * dx - sy * dz; topZ = sy * dx + cy * dz;
    topY = s.climbTop - (s.y - PLAYER.groundBias);
    up = smoothstep(0.2, 1, m);
    // pelvis forward over the top and up to standing; the chest pitched over the hands
    const pz = o[3 * J.pelvis + 2] + (topZ - o[3 * J.pelvis + 2]) * up;
    const ph = o[3 * J.pelvis + 1] + (P.pelvisH - 0.1 - o[3 * J.pelvis + 1]) * up;
    const pitch = 0.9 * Math.sin(Math.PI * Math.min(1, m / 0.85));
    set(o, J.pelvis, o[3 * J.pelvis] * (1 - up), ph, pz);
    set(o, J.chest, o[3 * J.pelvis], ph + P.spine * Math.cos(pitch), pz + P.spine * Math.sin(pitch));
  }
  torsoFrame(o, 0, 0, 1);
  upperBody(o, m > 0 ? 0.3 : -0.25 + 0.4 * smoothstep(-0.3, -1.5, s.vy));
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  o[3 * J.hipL + 1] += P.hipW * Math.sin(roll);
  o[3 * J.hipR + 1] -= P.hipW * Math.sin(roll);
  for (let i = 0; i < 4; i++) {
    const c = 4 * i;
    const hand = i <= LIMB.handR;
    const side = i === LIMB.handL || i === LIMB.footL ? -1 : 1;
    const end = hand ? (side < 0 ? J.handL : J.handR) : (side < 0 ? J.ankleL : J.ankleR);
    onBark(o, end, R, at[c + A_ANG], at[c + A_H], at[c + A_OFF]);
    if (m > 0) {
      if (hand) {
        // both hands flat on the top until the chest is over them, then off it
        const off = smoothstep(0.75, 1, m);
        const hx = topX + side * 0.2, hy = topY + 0.05, hz = topZ - 0.12;
        const ax = o[3 * end] + (hx - o[3 * end]) * smoothstep(0, 0.2, m);
        const ay = o[3 * end + 1] + (hy - o[3 * end + 1]) * smoothstep(0, 0.2, m);
        const az = o[3 * end + 2] + (hz - o[3 * end + 2]) * smoothstep(0, 0.2, m);
        set(o, end, ax + side * 0.15 * off, ay - 0.45 * off, az - 0.1 * off);
      } else {
        // the feet come up under the body (the right first), clear of the edge
        const f = smoothstep(side > 0 ? 0.5 : 0.62, side > 0 ? 0.8 : 0.92, m);
        const ex = o[3 * end] + (side * P.hipW - o[3 * end]) * f;
        const ey = o[3 * end + 1] + (P.ankleH - o[3 * end + 1]) * f + 0.25 * Math.sin(Math.PI * f);
        const ez = o[3 * end + 2] + (topZ * up - o[3 * end + 2]) * f;
        set(o, end, ex, ey, ez);
      }
    }
    if (hand) {
      // elbows out and down, off the bark
      ik(o, side < 0 ? J.shoulderL : J.shoulderR, side < 0 ? J.elbowL : J.elbowR, end, P.upperArm, P.forearm, side * 0.32, -0.3, -0.12);
    } else {
      // knees out to the sides and up, toward the trunk: the frog stance
      ik(o, side < 0 ? J.hipL : J.hipR, side < 0 ? J.kneeL : J.kneeR, end, P.thigh, P.shin, side * 0.38, 0.22, 0.15);
      // toes toward the axis (on a pole: the ankles turned in, soles on its sides), the heel peeling first
      let qx = -o[3 * end], qz = R - o[3 * end + 2];
      const ql = Math.sqrt(qx * qx + qz * qz) || 1;
      qx /= ql; qz /= ql;
      const rise = (0.35 - 0.9 * at[c + A_PEEL]) * (1 - up);
      const n = P.footLen / Math.sqrt(1 + rise * rise);
      toe(o, end, side < 0 ? J.toeL : J.toeR, qx * n * (1 - up), rise * n - 0.03 * up, qz * n * (1 - up) + P.footLen * up);
    }
  }
}

function poseSwim(r: HeroRig, o: Float64Array): void {
  const c = r.swimPhase;
  set(o, J.chest, 0, 1.2, 0.08);
  set(o, J.pelvis, 0, 1.05, -0.26);
  torsoFrame(o, 0, 0, 1);
  upperBody(o, 0.5);
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  // Breaststroke: reach forward, sweep out, tuck.
  const st = 0.5 + 0.5 * Math.sin(c);
  for (let k = 0; k < 2; k++) {
    const side = k === 0 ? -1 : 1;
    const sh = side < 0 ? J.shoulderL : J.shoulderR;
    const hand = side < 0 ? J.handL : J.handR;
    set(o, hand, o[3 * sh] + side * (0.06 + 0.26 * (1 - st)), o[3 * sh + 1] - 0.06, o[3 * sh + 2] + 0.12 + 0.34 * st);
    ik(o, sh, side < 0 ? J.elbowL : J.elbowR, hand, P.upperArm, P.forearm, side, -0.4, -0.5);
    const hip = side < 0 ? J.hipL : J.hipR;
    const ank = side < 0 ? J.ankleL : J.ankleR;
    set(o, ank, o[3 * hip] + side * 0.04, o[3 * hip + 1] - 0.16 + 0.07 * Math.sin(2 * c + side * 1.6), o[3 * hip + 2] - 0.74);
    ik(o, hip, side < 0 ? J.kneeL : J.kneeR, ank, P.thigh, P.shin, 0, -1, 0.2);
    toe(o, ank, side < 0 ? J.toeL : J.toeR, 0, -0.08, -0.12);
  }
}

/** On the shield (lying on the plane y = a·x + b·z): knees bent, feet apart across it, arms out for balance. */
function poseSurf(r: HeroRig, o: Float64Array, a: number, b: number): void {
  const sway = 0.03 * Math.sin(r.time * 3.1);
  for (let k = 0; k < 2; k++) {
    const side = k === 0 ? -1 : 1;
    const fx = side * 0.13, fz = side * 0.035;
    set(o, side < 0 ? J.ankleL : J.ankleR, fx, a * fx + b * fz + P.ankleH + 0.05, fz);
  }
  const pelvisY = 0.76 + sway;
  set(o, J.pelvis, sway, pelvisY, -0.03);
  set(o, J.chest, sway, pelvisY + Math.cos(0.3) * P.spine, -0.03 + Math.sin(0.3) * P.spine);
  torsoFrame(o, 0, 0, 1);
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  ik(o, J.hipL, J.kneeL, J.ankleL, P.thigh, P.shin, -0.45, 0, 1);
  ik(o, J.hipR, J.kneeR, J.ankleR, P.thigh, P.shin, 0.45, 0, 1);
  toe(o, J.ankleL, J.toeL, 0, b * P.footLen - 0.035, P.footLen);
  toe(o, J.ankleR, J.toeR, 0, b * P.footLen - 0.035, P.footLen);
  upperBody(o, -0.05);
  const bal = 0.12 * Math.sin(r.time * 2.3);
  armFK(o, -1, 0.45, 1.05 + bal, 0.55);
  armFK(o, 1, 0.3, 1.0 - bal, 0.6);
}

// ---- the combat layer ---------------------------------------------------------------------------

// Scratch: a swing's sample, the blade's direction and travel, the locomotion's hands and shoulders,
// the torso frame before any twist (the back's: where the sheath and the shield hang).
const SW = new Float64Array(SWING_CHANNELS);
const BD = new Float64Array(3);
const LOCO = new Float64Array(3 * HERO_JOINTS);
const BF = new Float64Array(9);
const TMP = new Float64Array(9);

/** Where torso-frame offset (a, b, c) from the pelvis lies (the back's frame BF), into `g` at `at`. */
function onBack(o: Float64Array, g: Float64Array, at: number, a: number, b: number, c: number): void {
  g[at] = o[3 * J.pelvis] + BF[0] * a + BF[3] * b + BF[6] * c;
  g[at + 1] = o[3 * J.pelvis + 1] + BF[1] * a + BF[4] * b + BF[7] * c;
  g[at + 2] = o[3 * J.pelvis + 2] + BF[2] * a + BF[5] * b + BF[8] * c;
}

/** A direction in the back's frame, into `g` at `at` (normalised). */
function backDir(g: Float64Array, at: number, a: number, b: number, c: number): void {
  const x = BF[0] * a + BF[3] * b + BF[6] * c, y = BF[1] * a + BF[4] * b + BF[7] * c, z = BF[2] * a + BF[5] * b + BF[8] * c;
  const l = Math.sqrt(x * x + y * y + z * z);
  g[at] = x / l; g[at + 1] = y / l; g[at + 2] = z / l;
}

/** Unit `v` (at g[at]) made perpendicular to unit `n` (at g[an]) and normalised; `fx, fy, fz` if they were parallel. */
function orthogonal(g: Float64Array, at: number, an: number, fx: number, fy: number, fz: number): void {
  const d = g[at] * g[an] + g[at + 1] * g[an + 1] + g[at + 2] * g[an + 2];
  let x = g[at] - g[an] * d, y = g[at + 1] - g[an + 1] * d, z = g[at + 2] - g[an + 2] * d;
  let l = Math.sqrt(x * x + y * y + z * z);
  if (l < 1e-5) { x = fx; y = fy; z = fz; l = Math.sqrt(x * x + y * y + z * z); }
  g[at] = x / l; g[at + 1] = y / l; g[at + 2] = z / l;
}

function setv(g: Float64Array, at: number, x: number, y: number, z: number, w: number): void {
  g[at] = x; g[at + 1] = y; g[at + 2] = z; g[at + 3] = w;
}

/** Joint j moved to `t` of the way from (x0, y0, z0) to (x1, y1, z1). */
function setLerp(o: Float64Array, j: number, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, t: number): void {
  set(o, j, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t);
}

/**
 * A hand fetching gear from the back and bringing it to `target`: with the
 * blend a below ½ it reaches from where the locomotion put it to the back
 * (bx, by, bz), above ½ from there to the target. Sets joint `hand`.
 */
function fetch(o: Float64Array, hand: number, a: number, bx: number, by: number, bz: number, tx: number, ty: number, tz: number): void {
  const lx = LOCO[3 * hand], ly = LOCO[3 * hand + 1], lz = LOCO[3 * hand + 2];
  if (a < 0.5) setLerp(o, hand, lx, ly, lz, bx, by, bz, smoothstep(0, 0.5, a));
  else setLerp(o, hand, bx, by, bz, tx, ty, tz, smoothstep(0.5, 1, a));
}

/**
 * The combat layer over the blended locomotion skeleton `o`: arms, a twist of
 * the shoulders, and the gear (r.gear). (ga, gb): the ground's rise per metre
 * along the hero's right and ahead, for the surfing shield.
 */
function poseCombat(r: HeroRig, s: HeroState, c: CombatPose, o: Float64Array, dt: number, snap: boolean, ga: number, gb: number): void {
  const g = r.gear;
  // The poses that need the arms (the glider's handles, a wall, the strokes) keep them.
  const free = 1 - r.weights[W_GLIDE] - r.weights[W_CLIMB] - r.weights[W_SWIM];
  const surf = r.weights[W_SURF];
  const acting = c.act !== 'none' && c.act !== 'hurt';
  r.armed += ((c.armed && free > 0.5 ? 1 : 0) - r.armed) * follow(snap, acting ? 22 : 9, dt);
  if (r.armed < 1e-4) r.armed = 0; else if (r.armed > 1 - 1e-4) r.armed = 1;
  r.aim += ((c.act === 'aim' && free > 0.5 ? 1 : 0) - r.aim) * follow(snap, 14, dt);
  if (r.aim < 1e-4) r.aim = 0;
  r.guard += ((c.act === 'guard' ? 1 : 0) - r.guard) * follow(snap, 18, dt);
  r.point += ((c.act === 'throw' && c.u < 0.6 ? 1 : 0) - r.point) * follow(snap, 12, dt);
  if (r.point < 1e-4) r.point = 0;

  // The weapon arm's channels: a swing's sample, or a stance eased into.
  const model = c.weapon;
  let rate = 14, yawV = 0, pitchV = 0;
  if ((c.act === 'swing' || c.act === 'throw') && c.move >= 0) {
    swingAt(MOVES[c.move], c.u, SW);
    rate = 40; yawV = SW[5]; pitchV = SW[6];
  } else if (c.act === 'charge') {
    swingAt(MOVES[MOVE.spin], 0, SW);
    SW[1] += 0.03 * Math.sin(r.time * 41) * c.glow; // (the blade trembles as the charge builds)
  } else {
    const st = c.act === 'guard' ? GUARD_ARM : model === MODEL.sword || model === MODEL.none ? READY : READY_CLUB;
    for (let k = 0; k < 5; k++) SW[k] = st[k];
  }
  const ka = follow(snap, rate, dt);
  for (let k = 0; k < 5; k++) r.arm[k] += (SW[k] - r.arm[k]) * ka;
  const sh = c.act === 'guard' ? SHIELD_GUARD : c.act === 'swing' || c.act === 'charge' ? SHIELD_SWING : SHIELD_READY;
  const ks = follow(snap, 16, dt);
  for (let k = 0; k < 3; k++) r.shieldArm[k] += (sh[k] - r.shieldArm[k]) * ks;

  // The back's frame (before any twist) and the locomotion's arms, for the blends.
  bodyFrame(r, o);
  for (let q = 0; q < 9; q++) BF[q] = F[q];
  for (let q = 0; q < LOCO.length; q++) LOCO[q] = o[q];

  const a = r.armed;
  const upper = free * Math.max(a, r.aim, r.point);
  if (upper > 1e-4) {
    // Shoulders and head turn with the swing (side-on to the aim while drawing the bow).
    const twist = r.arm[4] * a * (1 - r.aim) + AIM_TWIST * r.aim;
    torsoFrame(o, Math.sin(twist), 0, Math.cos(twist));
    upperBody(o, 0.05 + 0.1 * r.aim);
    for (let k = 0; k < SHOULDERS.length; k++) {
      const j = SHOULDERS[k];
      setLerp(o, j, LOCO[3 * j], LOCO[3 * j + 1], LOCO[3 * j + 2], o[3 * j], o[3 * j + 1], o[3 * j + 2], upper);
    }
    // An arm the layer leaves to the locomotion goes along with its shoulder.
    for (let side = 0; side < 2; side++) {
      const sh = side ? J.shoulderR : J.shoulderL;
      const dx = o[3 * sh] - LOCO[3 * sh], dy = o[3 * sh + 1] - LOCO[3 * sh + 1], dz = o[3 * sh + 2] - LOCO[3 * sh + 2];
      for (let j = sh + 1; j <= sh + 2; j++) { o[3 * j] += dx; o[3 * j + 1] += dy; o[3 * j + 2] += dz; LOCO[3 * j] += dx; LOCO[3 * j + 1] += dy; LOCO[3 * j + 2] += dz; }
    }
  }
  const cx = o[3 * J.chest], cy = o[3 * J.chest + 1], cz = o[3 * J.chest + 2];
  // the aim (hero-local: the hero faces the way it aims)
  const ap = c.pitch, ady = Math.sin(ap), adz = Math.cos(ap);

  // --- the weapon arm ---
  bladeDir(r.arm[0], r.arm[1], BD);
  const reach = r.arm[2];
  const tx = cx + 0.07 + BD[0] * reach, ty = cy - 0.12 + r.arm[3] + BD[1] * reach, tz = cz + 0.03 + BD[2] * reach;
  // the bow's grip and the string hand while drawing
  const bgx = o[3 * J.shoulderL] - 0.02, bgy = o[3 * J.shoulderL + 1] + ady * 0.52, bgz = o[3 * J.shoulderL + 2] + adz * 0.52;
  const draw = Math.max(0, c.draw);
  const pull = 0.1 + 0.5 * draw;
  const nx = bgx - 0.0 + 0.04, ny = bgy - ady * pull + 0.03, nz = bgz - adz * pull;
  const wR = free * Math.max(a, r.aim);
  if (wR > 1e-4) {
    onBack(o, TMP, 0, 0.14, 0.41, -0.2); // the hilt over the right shoulder
    fetch(o, J.handR, a, TMP[0], TMP[1], TMP[2], tx, ty, tz);
    if (r.aim > 0) {
      const k = smoothstep(0, 1, r.aim);
      setLerp(o, J.handR, o[3 * J.handR], o[3 * J.handR + 1], o[3 * J.handR + 2], nx, ny, nz, k);
    }
    const h = 3 * J.handR;
    setLerp(o, J.handR, LOCO[h], LOCO[h + 1], LOCO[h + 2], o[h], o[h + 1], o[h + 2], free);
    ik(o, J.shoulderR, J.elbowR, J.handR, P.upperArm, P.forearm, 0.7, -0.7 + 0.5 * Math.max(0, BD[1]), -0.25);
  }

  // --- the shield arm (or the bow arm, or a throw's pointing arm) ---
  const sa = c.shield ? a : 0;
  const wL = free * Math.max(sa, r.aim, r.point) * (1 - surf);
  if (wL > 1e-4) {
    onBack(o, TMP, 0, -0.12, 0.33, -0.24); // the shield's handle on the back
    const lx = cx + r.shieldArm[0], ly = cy + r.shieldArm[1], lz = cz + r.shieldArm[2];
    fetch(o, J.handL, sa, TMP[0], TMP[1], TMP[2], lx, ly, lz);
    const pt = Math.max(r.aim, r.point);
    if (pt > 0) {
      const k = smoothstep(0, 1, pt);
      setLerp(o, J.handL, o[3 * J.handL], o[3 * J.handL + 1], o[3 * J.handL + 2], bgx, bgy, bgz, k);
    }
    const h = 3 * J.handL;
    setLerp(o, J.handL, LOCO[h], LOCO[h + 1], LOCO[h + 2], o[h], o[h + 1], o[h + 2], free * (1 - surf));
    ik(o, J.shoulderL, J.elbowL, J.handL, P.upperArm, P.forearm, -0.7, -0.75, -0.2);
  }

  // --- the gear ---
  const inHand = a >= 0.5 && r.aim < 0.5 && free > 0.5;
  const hr = 3 * J.handR, hl = 3 * J.handL;
  if (model === MODEL.none) {
    setv(g, GEAR.grip, 0, 0, 0, 0);
    setv(g, GEAR.axis, 0, 1, 0, 0);
    setv(g, GEAR.flat, 0, 0, 1, 0);
  } else if (inHand) {
    setv(g, GEAR.grip, o[hr], o[hr + 1], o[hr + 2], model);
    setv(g, GEAR.axis, BD[0], BD[1], BD[2], c.glow);
    // The flat faces across the way the blade travels (its edge leads); slowing, it turns to face
    // across the way it points (⟂ the blade, level).
    const cpp = Math.cos(r.arm[1]), spp = Math.sin(r.arm[1]), cyy = Math.cos(r.arm[0]), syy = Math.sin(r.arm[0]);
    const vx = cyy * cpp * yawV - syy * spp * pitchV, vy = cpp * pitchV, vz = -syy * cpp * yawV - cyy * spp * pitchV;
    let fx = BD[1] * vz - BD[2] * vy, fy = BD[2] * vx - BD[0] * vz, fz = BD[0] * vy - BD[1] * vx;
    const lvx = -BD[2], lvz = BD[0];
    const lvl = Math.sqrt(lvx * lvx + lvz * lvz) || 1;
    const fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
    if (fx * lvx + fz * lvz < 0) { fx = -fx; fy = -fy; fz = -fz; } // (a flat has no front: keep the one nearer)
    const wv = smoothstep(0.3, 2.5, fl) / Math.max(fl, 1e-6);
    setv(g, GEAR.flat, fx * wv + (lvx / lvl) * (1 - wv * fl), fy * wv, fz * wv + (lvz / lvl) * (1 - wv * fl), 0);
    orthogonal(g, GEAR.flat, GEAR.axis, -BD[2], 0, BD[0]);
  } else if (model === MODEL.sword) {
    // in its sheath: the hilt over the right shoulder, the blade down the back
    onBack(o, g, GEAR.grip, 0.137, 0.411, -0.175);
    g[GEAR.grip + 3] = model;
    backDir(g, GEAR.axis, -0.46, -0.89, 0);
    g[GEAR.axis + 3] = 0;
    backDir(g, GEAR.flat, 0, 0, 1);
    g[GEAR.flat + 3] = 1;
  } else {
    // a club slung across the back, the head by the right hip
    onBack(o, g, GEAR.grip, -0.13, 0.43, -0.21);
    g[GEAR.grip + 3] = model;
    backDir(g, GEAR.axis, 0.4, -0.92, 0);
    g[GEAR.axis + 3] = 0;
    backDir(g, GEAR.flat, 0, 0, 1);
    g[GEAR.flat + 3] = 0;
  }
  // (a shield carried with no model named is the Hylian one)
  const sm = c.shieldModel > 0 ? c.shieldModel : SHIELD_MODEL.hylian;
  if (!c.shield) {
    setv(g, GEAR.shield, 0, 0, 0, 0);
    setv(g, GEAR.shieldX, 1, 0, 0, 0);
    setv(g, GEAR.shieldY, 0, 1, 0, 0);
  } else if (surf > 0.5) {
    // under the feet, on the ground's plane
    const l = Math.sqrt(1 + ga * ga);
    setv(g, GEAR.shield, 0, 0.035, 0, sm);
    setv(g, GEAR.shieldX, 1 / l, ga / l, 0, 0);
    setv(g, GEAR.shieldY, 0, gb, 1, 0);
    orthogonal(g, GEAR.shieldY, GEAR.shieldX, 0, 0, 1);
  } else if (sa >= 0.5 && r.aim < 0.5 && free > 0.5) {
    // on the forearm, its face turned out (forward with the guard up)
    const k = r.guard;
    // (face n and up u, then right = u × n so that right × up = n)
    let nx2 = FACE_READY[0] + (FACE_GUARD[0] - FACE_READY[0]) * k;
    let ny2 = FACE_READY[1] + (FACE_GUARD[1] - FACE_READY[1]) * k;
    let nz2 = FACE_READY[2] + (FACE_GUARD[2] - FACE_READY[2]) * k;
    const nl = Math.sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2);
    nx2 /= nl; ny2 /= nl; nz2 /= nl;
    TMP[0] = nx2; TMP[1] = ny2; TMP[2] = nz2;
    setv(g, GEAR.shieldY, UP_READY[0] + (UP_GUARD[0] - UP_READY[0]) * k, UP_READY[1] + (UP_GUARD[1] - UP_READY[1]) * k, UP_READY[2] + (UP_GUARD[2] - UP_READY[2]) * k, 0);
    TMP[3] = g[GEAR.shieldY]; TMP[4] = g[GEAR.shieldY + 1]; TMP[5] = g[GEAR.shieldY + 2];
    orthogonal(TMP, 3, 0, 0, 1, 0);
    g[GEAR.shieldY] = TMP[3]; g[GEAR.shieldY + 1] = TMP[4]; g[GEAR.shieldY + 2] = TMP[5];
    // right = up × face
    g[GEAR.shieldX] = TMP[4] * nz2 - TMP[5] * ny2;
    g[GEAR.shieldX + 1] = TMP[5] * nx2 - TMP[3] * nz2;
    g[GEAR.shieldX + 2] = TMP[3] * ny2 - TMP[4] * nx2;
    setv(g, GEAR.shield, o[hl] + nx2 * 0.075 + TMP[3] * 0.02, o[hl + 1] + ny2 * 0.075 + TMP[4] * 0.02, o[hl + 2] + nz2 * 0.075 + TMP[5] * 0.02, sm);
  } else {
    // on the back, face out
    onBack(o, g, GEAR.shield, 0, 0.25, -0.222);
    g[GEAR.shield + 3] = sm;
    setv(g, GEAR.shieldX, BF[0], BF[1], BF[2], 0);
    setv(g, GEAR.shieldY, BF[3], BF[4], BF[5], 0);
  }
  if (r.aim >= 0.5 && free > 0.5 && c.bowModel > 0) {
    // the bow upright (canted a little), its string in the right hand
    setv(g, GEAR.bow, o[hl], o[hl + 1], o[hl + 2], c.bowModel);
    setv(g, GEAR.bowFwd, 0, ady, adz, c.draw >= 0 ? 1 : 0);
    setv(g, GEAR.bowUp, 0.2, 1, 0, draw);
    orthogonal(g, GEAR.bowUp, GEAR.bowFwd, 0, 1, 0);
  } else {
    setv(g, GEAR.bow, 0, 0, 0, 0);
    setv(g, GEAR.bowUp, 0, 1, 0, 0);
    setv(g, GEAR.bowFwd, 0, 0, 1, 0);
  }
}

const ROT = new Float64Array(9);

/** ROT = a turn by t about the y axis (positive: the right shoulder back). */
function rotY(t: number): Float64Array {
  const c = Math.cos(t), sn = Math.sin(t);
  ROT[0] = c; ROT[1] = 0; ROT[2] = sn;
  ROT[3] = 0; ROT[4] = 1; ROT[5] = 0;
  ROT[6] = -sn; ROT[7] = 0; ROT[8] = c;
  return ROT;
}

const HL = new Float64Array(3), HR = new Float64Array(3), RC = new Float64Array(3);

/** v = p + (a, b, c) for joint p of `o` (or a point when p < 0: (a, b, c) alone). */
function at(v: Float64Array, o: Float64Array, p: number, a: number, b: number, c: number): void {
  v[0] = (p >= 0 ? o[3 * p] : 0) + a; v[1] = (p >= 0 ? o[3 * p + 1] : 0) + b; v[2] = (p >= 0 ? o[3 * p + 2] : 0) + c;
}
function lerp3(v: Float64Array, x: number, y: number, z: number, t: number): void {
  v[0] += (x - v[0]) * t; v[1] += (y - v[1]) * t; v[2] += (z - v[2]) * t;
}

/**
 * The hands on a rock (CombatPose.rock, its radius): stooping to it and up with it (a lift; run backward, putting it
 * down), carried (a stone on the right palm by the shoulder, a bigger rock overhead between both hands), thrown (a stone
 * overhand with the shoulders turning through, a rock wound back over the head and heaved forward with the whole body),
 * or both hands driving into a boulder, leaning in. The body leans and turns about the pelvis; the rock's middle comes
 * out in r.rockAt (hero-local), for the engine to put the rock there.
 */
function poseRock(r: HeroRig, c: CombatPose, o: Float64Array, dt: number, snap: boolean): void {
  const act = c.act;
  const on = act === 'lift' || act === 'carry' || act === 'hurl' || act === 'push';
  const free = 1 - r.weights[W_GLIDE] - r.weights[W_CLIMB] - r.weights[W_SWIM];
  r.rockW += ((on && free > 0.5 ? 1 : 0) - r.rockW) * follow(snap, 18, dt);
  r.rockAt[0] = NaN;
  if (r.rockW < 1e-3) { r.rockW = 0; return; }
  const w = on ? Math.max(r.rockW, act === 'push' ? 0 : 1) : r.rockW;
  const rho = Math.max(0.12, c.rock), half = Math.max(0.1, c.rockHalf), one = c.rockHand, u = c.u;
  // the body: a stoop to lift, leaning into a push, a heave back over the head and forward, a turn of the shoulders through an overhand throw
  let lean = 0, twist = 0;
  if (act === 'lift') lean = 0.6 * smoothstep(0, 0.32, u) * (1 - smoothstep(0.42, 0.88, u)) + (one ? 0 : 0.06 * (1 - smoothstep(0.8, 1, u)));
  else if (act === 'push') lean = 0.42 + 0.12 * Math.sin(Math.PI * smoothstep(0.2, 0.7, u));
  else if (act === 'hurl') {
    if (one) twist = 0.65 * smoothstep(0, 0.42, u) - 1.1 * smoothstep(0.42, 0.62, u) + 0.45 * smoothstep(0.62, 1, u);
    else lean = -0.32 * smoothstep(0, 0.42, u) + 0.8 * smoothstep(0.42, 0.62, u) - 0.48 * smoothstep(0.66, 1, u);
  }
  lean *= w; twist *= w;
  const px = o[3 * J.pelvis], py = o[3 * J.pelvis + 1], pz = o[3 * J.pelvis + 2];
  if (lean !== 0) {
    rotXZ(lean, 0);
    for (let k = 0; k < UPPER.length; k++) turnP(o, 3 * UPPER[k], px, py, pz);
    turnBody(r, px, py, pz, false, false);
  }
  if (twist !== 0) {
    rotY(twist);
    for (let k = 0; k < UPPER.length; k++) turnP(o, 3 * UPPER[k], px, py, pz);
    turnBody(r, px, py, pz, false, false);
  }
  // where the rock is carried: on the right palm by the shoulder, or overhead
  const sR = 3 * J.shoulderR, hd = 3 * J.head;
  const carryX = one ? o[sR] + 0.05 : 0;
  const carryY = one ? o[sR + 1] + 0.1 + half : o[hd + 1] + 0.1 + half;
  const carryZ = one ? o[sR + 2] + 0.1 : o[hd + 2] + 0.04;
  let hold = true;
  let leftFree = one; // (a stone leaves the left hand to the stride)
  RC[0] = carryX; RC[1] = carryY; RC[2] = carryZ;
  if (act === 'lift') {
    // down to where it lies, then up with it
    const a = smoothstep(0, 0.4, u), b = smoothstep(0.45, 0.9, u);
    const lx = 0, ly = c.rockLow, lz = c.rockFwd;
    RC[0] = lx + (carryX - lx) * b; RC[1] = ly + (carryY - ly) * b; RC[2] = lz + (carryZ - lz) * b;
    // the hands: from the stride to its sides, then with it
    const spread = one ? 0.55 : 0.78;
    const side = rho * (spread + (one ? 0 : (0.62 - spread) * b));
    const drop = rho * (0.1 - (one ? 0.9 : 0.72) * b);
    at(HL, o, J.handL, 0, 0, 0); at(HR, o, J.handR, 0, 0, 0);
    const tL = [RC[0] - side, RC[1] + drop, RC[2] - rho * 0.2 * (1 - b)];
    const tR = one ? [RC[0], RC[1] - half * 0.95 * b + rho * 0.1 * (1 - b), RC[2] - rho * 0.25 * (1 - b)] : [RC[0] + side, RC[1] + drop, RC[2] - rho * 0.2 * (1 - b)];
    lerp3(HL, tL[0], tL[1], tL[2], a * (one ? 1 - b : 1));
    lerp3(HR, tR[0], tR[1], tR[2], a);
    leftFree = false;
    hold = u > 0.36;
  } else if (act === 'carry') {
    if (one) at(HR, o, -1, RC[0], RC[1] - half * 0.95, RC[2] - 0.02);
    else { at(HL, o, -1, RC[0] - rho * 0.7, RC[1] - half * 0.7, RC[2]); at(HR, o, -1, RC[0] + rho * 0.7, RC[1] - half * 0.7, RC[2]); }
  } else if (act === 'hurl') {
    if (one) {
      // back by the ear, flung up and forward, through and down
      const a = smoothstep(0, 0.42, u), b = smoothstep(0.42, 0.58, u), d = smoothstep(0.58, 1, u);
      at(HR, o, -1, carryX, carryY - half * 0.95, carryZ);
      lerp3(HR, o[sR] + 0.1, o[sR + 1] + 0.22, o[sR + 2] - 0.32, a);
      lerp3(HR, o[sR] - 0.05, o[sR + 1] + 0.28, o[sR + 2] + 0.52, b);
      lerp3(HR, o[3 * J.chest] - 0.14, o[3 * J.chest + 1] - 0.22, o[3 * J.chest + 2] + 0.42, d);
      RC[0] = HR[0]; RC[1] = HR[1] + half * 0.95; RC[2] = HR[2] + 0.02;
    } else {
      // wound back over the head, heaved over it and forward, the hands following through low ahead
      const a = smoothstep(0, 0.42, u), b = smoothstep(0.42, 0.6, u), d = smoothstep(0.6, 1, u);
      RC[0] = 0; RC[1] = carryY; RC[2] = carryZ;
      lerp3(RC, 0, o[hd + 1] + 0.05 + rho * 0.75, o[hd + 2] - 0.2 - rho * 0.35, a);
      lerp3(RC, 0, o[hd + 1] + rho * 0.45, o[hd + 2] + 0.5, b);
      at(HL, o, -1, RC[0] - rho * 0.7, RC[1] - half * 0.7, RC[2]); at(HR, o, -1, RC[0] + rho * 0.7, RC[1] - half * 0.7, RC[2]);
      const cy = o[3 * J.chest + 1] - 0.12, cz = o[3 * J.chest + 2] + 0.45;
      lerp3(HL, -0.16, cy, cz, d); lerp3(HR, 0.16, cy, cz, d);
    }
    hold = u < 0.52;
    leftFree = one;
  } else {
    // both hands on the boulder, driving at the push
    const thrust = Math.sin(Math.PI * smoothstep(0.2, 0.7, u));
    const ch = 3 * J.chest;
    at(HL, o, -1, o[ch] - 0.19, o[ch + 1] - 0.04, o[ch + 2] + 0.36 + 0.24 * thrust);
    at(HR, o, -1, o[ch] + 0.19, o[ch + 1] - 0.04, o[ch + 2] + 0.36 + 0.24 * thrust);
    hold = false;
    leftFree = false;
  }
  // the hands there (blended in), the elbows by IK (out to the sides overhead, down and back otherwise)
  const over = act !== 'push' && !one ? 1 : 0;
  if (!leftFree) {
    setLerp(o, J.handL, o[3 * J.handL], o[3 * J.handL + 1], o[3 * J.handL + 2], HL[0], HL[1], HL[2], r.rockW);
    ik(o, J.shoulderL, J.elbowL, J.handL, P.upperArm, P.forearm, -0.7 - 0.3 * over, -0.6 + 0.5 * over, -0.2);
  }
  setLerp(o, J.handR, o[3 * J.handR], o[3 * J.handR + 1], o[3 * J.handR + 2], HR[0], HR[1], HR[2], r.rockW);
  ik(o, J.shoulderR, J.elbowR, J.handR, P.upperArm, P.forearm, 0.7 + 0.3 * over, -0.6 + 0.5 * over, -0.2);
  if (hold && act !== 'push' && r.rockW > 0.5) { r.rockAt[0] = RC[0]; r.rockAt[1] = RC[1]; r.rockAt[2] = RC[2]; }
}

/** Hero-local point p (HeroRig.local's axes) in the world, into `out`, for the hero as packHeroUniforms places it. */
export function heroLocalToWorld(r: HeroRig, s: HeroState, x: number, y: number, z: number, out: Float64Array): void {
  const yaw = s.yaw + r.spin, sy = Math.sin(yaw), cy = Math.cos(yaw);
  out[0] = s.x + cy * x + sy * z;
  out[1] = s.y - PLAYER.groundBias + y;
  out[2] = s.z - sy * x + cy * z;
}

/** Point a[i..i+2] turned by ROT about (px, py, pz). */
function turnP(a: Float64Array, i: number, px: number, py: number, pz: number): void {
  const x = a[i] - px, y = a[i + 1] - py, z = a[i + 2] - pz;
  a[i] = px + ROT[0] * x + ROT[1] * y + ROT[2] * z;
  a[i + 1] = py + ROT[3] * x + ROT[4] * y + ROT[5] * z;
  a[i + 2] = pz + ROT[6] * x + ROT[7] * y + ROT[8] * z;
}

/** Direction a[i..i+2] turned by ROT. */
function turnV(a: Float64Array, i: number): void {
  const x = a[i], y = a[i + 1], z = a[i + 2];
  a[i] = ROT[0] * x + ROT[1] * y + ROT[2] * z;
  a[i + 1] = ROT[3] * x + ROT[4] * y + ROT[5] * z;
  a[i + 2] = ROT[6] * x + ROT[7] * y + ROT[8] * z;
}

/** The upper body's joints (a flinch or a daze turns them about the pelvis). */
const UPPER: readonly number[] = [J.chest, J.head, J.shoulderL, J.elbowL, J.handL, J.shoulderR, J.elbowR, J.handR, J.glider];
/** The joints the combat layer blends back toward the locomotion's. */
const SHOULDERS: readonly number[] = [J.shoulderL, J.shoulderR, J.head];

/** The gear (positions and directions), the joints and the frames turned by ROT about (px, py, pz). */
function turnBody(r: HeroRig, px: number, py: number, pz: number, joints: boolean, frames: boolean): void {
  if (joints) for (let j = 0; j < HERO_JOINTS; j++) turnP(r.local, 3 * j, px, py, pz);
  const g = r.gear;
  turnP(g, GEAR.grip, px, py, pz); turnV(g, GEAR.axis); turnV(g, GEAR.flat);
  turnP(g, GEAR.shield, px, py, pz); turnV(g, GEAR.shieldX); turnV(g, GEAR.shieldY);
  turnP(g, GEAR.bow, px, py, pz); turnV(g, GEAR.bowUp); turnV(g, GEAR.bowFwd);
  if (frames) for (let c = 0; c < 6; c++) turnV(r.frames, 3 * c);
}
/** ROT = a turn by angle t about the x axis (pitch: positive tips forward) then about the z axis (roll). */
function rotXZ(t: number, roll: number): Float64Array {
  const cx = Math.cos(t), sx = Math.sin(t), cz = Math.cos(roll), sz = Math.sin(roll);
  // Rz · Rx, with Rx: y' = y cos − z sin… tipping the head (+y) forward (+z) for t > 0
  ROT[0] = cz; ROT[1] = -sz * cx; ROT[2] = sz * sx;
  ROT[3] = sz; ROT[4] = cz * cx; ROT[5] = -cz * sx;
  ROT[6] = 0; ROT[7] = sx; ROT[8] = cx;
  return ROT;
}

/**
 * Advance the animation by dt from the locomotion state (and a combat pose:
 * at peace with sword and shield on the back when none is given) and rebuild
 * the blended hero-local skeleton, gear, frames and bounding sphere.
 */
export function updateHeroRig(r: HeroRig, s: HeroState, dt: number, pose: CombatPose = AT_PEACE): void {
  const hv = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
  const snap = r.snap;
  r.snap = false;
  r.time += dt;

  r.speed += (hv - r.speed) * follow(snap, 12, dt);
  // (skydiving until the ground is landTTI away: then upright, feet first, to land)
  const tti = (s.y - PLAYER.groundBias - s.ground) / Math.max(1e-3, -s.vy);
  const sky = s.move === 'air' && s.skydive && tti > SKYDIVE.landTTI;
  const target = s.riding ? W_RIDE : s.move === 'surf' || (s.move === 'air' && s.surfing) ? W_SURF
    : sky ? W_SKY : s.move === 'ground' ? W_GROUND : s.move === 'air' ? W_AIR
      : s.move === 'glide' ? W_GLIDE : s.move === 'climb' ? W_CLIMB : W_SWIM;
  let sum = 0;
  const kw = follow(snap, target === W_SKY ? 6 : s.move === 'air' || target === W_GROUND ? 14 : 9, dt);
  for (let i = 0; i < POSES; i++) {
    r.weights[i] += ((i === target ? 1 : 0) - r.weights[i]) * kw;
    sum += r.weights[i];
  }
  for (let i = 0; i < POSES; i++) r.weights[i] /= sum;
  r.glide += ((s.move === 'glide' ? 1 : 0) - r.glide) * follow(snap, s.move === 'glide' ? 5 : 10, dt);
  if (r.glide < 1e-3) r.glide = 0;

  r.sprint += ((s.sprinting ? 1 : 0) - r.sprint) * follow(snap, 6, dt);
  // (the shield up, a charge or the bow drawn: a lower, wider stance)
  const braced = pose.act === 'guard' || pose.act === 'charge' || pose.act === 'aim' ? 0.35
    // (stooping to a rock and up with it, braced against a boulder, a little under a rock overhead)
    : pose.act === 'lift' ? 0.95 * (1 - smoothstep(0.42, 0.9, pose.u)) : pose.act === 'push' ? 0.5
      : (pose.act === 'carry' || pose.act === 'hurl') && !pose.rockHand ? 0.12 : 0;
  // (a heavy landing squats deep; a roll tucks)
  const tuck = Math.max(r.squat > 0 ? 0.85 : 0, r.roll > 0 ? 0.6 : 0);
  r.crouch += ((s.sneaking ? 1 : Math.max(braced, tuck)) - r.crouch) * follow(snap, tuck > 0 ? 14 : 8, dt);
  r.slide += ((s.sliding ? 1 : 0) - r.slide) * follow(snap, 8, dt);
  r.tired += ((s.exhausted && s.move === 'ground' ? 1 : 0) - r.tired) * follow(snap, 3, dt);
  if (s.landings !== r.landings) {
    r.landings = s.landings;
    if (!snap) {
      r.landT = 0;
      r.landD = Math.min(0.48, Math.max(0.08, s.impact * 0.024));
      r.roll = s.rolling ? SKYDIVE.rollTime : 0;
      r.squat = !s.rolling && s.impact >= SKYDIVE.squatMin ? SKYDIVE.squatTime : 0;
    }
  }
  // The landing's dip: an underdamped spring (§6.3: ωn 16 rad/s, ζ 0.62) struck at touchdown.
  r.landT += dt;
  r.land = r.landT < 0.65 ? r.landD * Math.exp(-0.62 * 16 * r.landT) * Math.sin(16 * Math.sqrt(1 - 0.62 * 0.62) * r.landT) : 0;
  r.roll = s.move === 'ground' ? Math.max(0, r.roll - dt) : 0;
  r.squat = s.move === 'ground' ? Math.max(0, r.squat - dt) : 0;
  const accel = snap ? 0 : (hv - r.lastSpeed) / Math.max(dt, 1e-3);
  r.lastSpeed = hv;
  const leanT = 0.05 + 0.03 * r.speed + 0.012 * Math.max(-8, Math.min(8, accel));
  r.lean += (Math.max(-0.1, Math.min(0.42, leanT)) - r.lean) * follow(snap, 6, dt);
  const yawRate = snap ? 0 : wrapAngle(s.yaw - r.lastYaw) / Math.max(dt, 1e-3);
  r.lastYaw = s.yaw;
  r.bank += (Math.max(-0.3, Math.min(0.3, yawRate * r.speed * 0.02)) - r.bank) * follow(snap, 8, dt);
  // Skydiving: the posture, the flare, and the steering read off the motion (so another player's hero, played
  // back, steers the same): the turn's rate and the speed ahead; the bank of a coordinated turn, tan θ = ω v / g
  // (× 1.35, held to 52°).
  r.dive += ((s.skydive ? s.dive : 0) - r.dive) * follow(snap, 8, dt);
  r.flare += ((s.flare > 0 ? 1 : 0) - r.flare) * follow(snap, 10, dt);
  const ahead = s.vx * Math.sin(s.yaw) + s.vz * Math.cos(s.yaw);
  r.steerX += (Math.max(-1, Math.min(1, yawRate * 0.6)) - r.steerX) * follow(snap, 10, dt);
  r.steerY += (Math.max(-1, Math.min(1, ahead / SKYDIVE.glideMaxXZ)) - r.steerY) * follow(snap, 10, dt);
  const bankT = -Math.max(-52 * DEG, Math.min(52 * DEG, yawRate * hv / PLAYER.gravity * 1.35));
  r.skyBank += (bankT - r.skyBank) * follow(snap, 8, dt);

  // Gait: stance feet move backward at the ground speed, so they stay planted.
  const L = Math.min(0.62, 0.22 + 0.055 * r.speed);
  const duty = 0.62 - 0.28 * smoothstep(1.5, 6.0, r.speed);
  r.phase = (r.phase + Math.PI * r.speed * duty / L * dt) % TAU;
  r.climbPhase = (r.climbPhase + (Math.abs(s.vy) + hv) * 3.5 * dt) % TAU;
  r.swimPhase = (r.swimPhase + (1.6 + 0.9 * hv) * dt * (s.sprinting ? 1.6 : 1)) % TAU;

  // Terrain plane at the root in hero-local axes.
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
  const ga = s.gx * cy - s.gz * sy; // rise per metre along local x (right)
  const gb = s.gx * sy + s.gz * cy; // along local z (forward)

  // On a trunk the holds move with the climb (not while vaulting onto the top: they are left as they were).
  if (s.move === 'climb' && s.climbR > 0) {
    if (s.mantle === 0) climbGait(r, s, dt, snap);
    else if (!r.holds) holdSockets(r, s.climbR); // (first seen mid-vault: the limbs at their sockets)
  } else if (s.move === 'climb') {
    // (a snapshot from a build that sends no surface: a flat wall, the limbs at their sockets)
    if (!r.holds) holdSockets(r, FLAT_WALL);
  } else {
    r.holds = false;
    r.body[12] = NaN;
  }

  const out = r.local, o = r.pose;
  const grounded = s.move === 'ground' || s.move === 'surf';
  out.fill(0);
  for (let i = 0; i < POSES; i++) {
    const w = r.weights[i];
    if (w < 1e-3) continue;
    o.fill(0);
    if (i === W_GROUND) poseGround(r, o, grounded ? ga : 0, grounded ? gb : 0);
    else if (i === W_AIR) poseAir(r, o, s.vy);
    else if (i === W_GLIDE) poseGlide(r, o);
    else if (i === W_CLIMB) poseClimb(r, s, o, dt, snap);
    else if (i === W_SWIM) poseSwim(r, o);
    else if (i === W_SKY) poseSkydive(r, o, hv, s.vy);
    else if (i === W_RIDE) poseRide(r, o);
    else poseSurf(r, o, grounded ? ga : 0, grounded ? gb : 0);
    if (i !== W_GLIDE) {
      // Only the glide pose places the canopy; the others park it over the hands.
      set(o, J.glider,
        0.5 * (o[3 * J.handL] + o[3 * J.handR]),
        Math.max(o[3 * J.handL + 1], o[3 * J.handR + 1]) + 0.42,
        0.5 * (o[3 * J.handL + 2] + o[3 * J.handR + 2]));
    }
    for (let q = 0; q < out.length; q++) out[q] += w * o[q];
  }

  // The arms and the gear as the combat pose asks.
  poseCombat(r, s, pose, out, dt, snap, grounded ? ga : 0, grounded ? gb : 0);
  poseRock(r, pose, out, dt, snap);
  // A blow taken snaps the upper body back; a dazed foe sways (both about the pelvis).
  let flinch = 0, sway = 0;
  if (pose.act === 'hurt') flinch = -0.5 * smoothstep(0, 0.06, pose.u) * Math.exp(-6 * Math.max(0, pose.u - 0.06));
  else if (pose.act === 'stun') { flinch = -0.18; sway = 0.22 * Math.sin(pose.u * 4.2); }
  if (flinch !== 0 || sway !== 0) {
    const px = out[3 * J.pelvis], py = out[3 * J.pelvis + 1], pz = out[3 * J.pelvis + 2];
    rotXZ(flinch, sway);
    for (let k = 0; k < UPPER.length; k++) turnP(out, 3 * UPPER[k], px, py, pz);
    turnBody(r, px, py, pz, false, false);
  }

  // Frames of the blended skeleton: torso faces forward (bodyFrame), head leans upright.
  jointFrames(r, out, r.weights[W_SKY]);

  // Whole-body turns: a backflip or a side hop about the hips, a knockdown about the feet (onto
  // the back); a spin attack's turn is added to the facing when packing.
  const down = pose.act === 'down' || pose.act === 'dead';
  r.fall += ((down ? -1.42 : 0) - r.fall) * follow(snap, down ? 9 : 4, dt);
  // (handed back by the ragdoll: lying on the back already)
  if (down && pose.ragW > 0) r.fall = -1.42;
  if (Math.abs(r.fall) < 1e-4) r.fall = 0;
  r.spin = pose.spin;
  if (pose.flip !== 0 || pose.roll !== 0) { rotXZ(pose.flip, pose.roll); turnBody(r, 0, 0.95, 0, true, true); }
  if (r.fall !== 0) { rotXZ(r.fall, 0); turnBody(r, 0, 0.12, 0, true, true); }
  // A landing roll: once head over heels about the tucked body's middle.
  if (r.roll > 0) { rotXZ(TAU * smoothstep(0, 1, 1 - r.roll / SKYDIVE.rollTime), 0); turnBody(r, 0, 0.7, 0, true, true); }
  // Thrown: the body is the ragdoll's (blended in as it takes over and as it hands back).
  if (pose.rag && pose.ragW > 0) ragdollBody(r, s, pose.rag, Math.min(1, pose.ragW), out);

  // Bounding sphere: centre of the joints' and the gear's box, radius to the
  // farthest primitive reach (HERO_SHAPE.*Reach / limbPad / gear, mirrored in the shader).
  heroBounds(out, r.gear, r.glide, r.bound);
}

/** The torso's frame (bodyFrame) and the head's (upright off the neck, facing the torso's way) into r.frames. */
function jointFrames(r: HeroRig, out: Float64Array, skyW: number): void {
  bodyFrame(r, out);
  for (let q = 0; q < 9; q++) r.frames[q] = F[q];
  let ux = out[3 * J.head] - out[3 * J.chest];
  let uy = out[3 * J.head + 1] - out[3 * J.chest + 1] + (skyW > 1e-3 ? 0.12 * (1 - skyW) : 0.12);
  let uz = out[3 * J.head + 2] - out[3 * J.chest + 2];
  let l = Math.sqrt(ux * ux + uy * uy + uz * uz);
  ux /= l; uy /= l; uz /= l;
  // (the face looks the torso's way, off the neck)
  const hx = skyW > 1e-3 ? F[6] : 0, hy = skyW > 1e-3 ? F[7] : 0, hz = skyW > 1e-3 ? F[8] : 1;
  const hd = hx * ux + hy * uy + hz * uz;
  let fx = hx - ux * hd, fy = hy - uy * hd, fz = hz - uz * hd;
  l = Math.sqrt(fx * fx + fy * fy + fz * fz);
  if (l < 1e-6) { fx = r.frames[6]; fy = r.frames[7]; fz = r.frames[8]; l = 1; }
  fx /= l; fy /= l; fz /= l;
  const fr = r.frames;
  fr[9] = uy * fz - uz * fy; fr[10] = uz * fx - ux * fz; fr[11] = ux * fy - uy * fx;
  fr[12] = ux; fr[13] = uy; fr[14] = uz;
  fr[15] = fx; fr[16] = fy; fr[17] = fz;
}

/** Scratch: the torso's frame and the chest before the ragdoll took the body. */
const RAG_F = new Float64Array(12);

/**
 * The joints blended toward the ragdoll's (world `rag`, by `w`), hero-local; the frames made again from them and the
 * gear (on the back, in the hands) carried with the torso, from where it was to where it is now.
 */
function ragdollBody(r: HeroRig, s: HeroState, rag: Float64Array, w: number, out: Float64Array): void {
  for (let q = 0; q < 9; q++) RAG_F[q] = r.frames[q];
  RAG_F[9] = out[3 * J.chest]; RAG_F[10] = out[3 * J.chest + 1]; RAG_F[11] = out[3 * J.chest + 2];
  const yaw = s.yaw + r.spin, sy = Math.sin(yaw), cy = Math.cos(yaw), by = s.y - PLAYER.groundBias;
  for (let j = 0; j < J.glider; j++) {
    const X = rag[3 * j] - s.x, Y = rag[3 * j + 1] - by, Z = rag[3 * j + 2] - s.z;
    out[3 * j] += (cy * X - sy * Z - out[3 * j]) * w;
    out[3 * j + 1] += (Y - out[3 * j + 1]) * w;
    out[3 * j + 2] += (sy * X + cy * Z - out[3 * j + 2]) * w;
  }
  set(out, J.glider, 0.5 * (out[3 * J.handL] + out[3 * J.handR]), Math.max(out[3 * J.handL + 1], out[3 * J.handR + 1]) + 0.42,
    0.5 * (out[3 * J.handL + 2] + out[3 * J.handR + 2]));
  jointFrames(r, out, 0);
  const g = r.gear, fr = r.frames;
  const carry = (i: number, point: boolean) => {
    const x = g[i] - (point ? RAG_F[9] : 0), y = g[i + 1] - (point ? RAG_F[10] : 0), z = g[i + 2] - (point ? RAG_F[11] : 0);
    const a = x * RAG_F[0] + y * RAG_F[1] + z * RAG_F[2], b = x * RAG_F[3] + y * RAG_F[4] + z * RAG_F[5], c = x * RAG_F[6] + y * RAG_F[7] + z * RAG_F[8];
    g[i] = (point ? out[3 * J.chest] : 0) + fr[0] * a + fr[3] * b + fr[6] * c;
    g[i + 1] = (point ? out[3 * J.chest + 1] : 0) + fr[1] * a + fr[4] * b + fr[7] * c;
    g[i + 2] = (point ? out[3 * J.chest + 2] : 0) + fr[2] * a + fr[5] * b + fr[8] * c;
  };
  carry(GEAR.grip, true); carry(GEAR.axis, false); carry(GEAR.flat, false);
  carry(GEAR.shield, true); carry(GEAR.shieldX, false); carry(GEAR.shieldY, false);
  carry(GEAR.bow, true); carry(GEAR.bowUp, false); carry(GEAR.bowFwd, false);
}

/** Up to seven points of the gear a bound must hold, each with its pad: GP (x, y, z, pad). */
const GP = new Float64Array(7 * 4);
let gpN = 0;
function putGP(x: number, y: number, z: number, pad: number): void {
  GP[4 * gpN] = x; GP[4 * gpN + 1] = y; GP[4 * gpN + 2] = z; GP[4 * gpN + 3] = pad; gpN++;
}

/** Fill GP with the gear's points (GEAR layout at `at` of `g`); their count. */
function gearPoints(g: ArrayLike<number>, at: number): number {
  gpN = 0;
  const model = g[at + GEAR.grip + 3];
  if (model > 0) {
    const len = model === MODEL.sword ? P.swordLen : P.clubLen;
    const gx = g[at + GEAR.grip], gy = g[at + GEAR.grip + 1], gz = g[at + GEAR.grip + 2];
    const ax = g[at + GEAR.axis], ay = g[at + GEAR.axis + 1], az = g[at + GEAR.axis + 2];
    putGP(gx - ax * 0.13, gy - ay * 0.13, gz - az * 0.13, P.gearPad);
    putGP(gx + ax * len * 0.5, gy + ay * len * 0.5, gz + az * len * 0.5, P.gearPad);
    putGP(gx + ax * len, gy + ay * len, gz + az * len, P.gearPad);
  }
  if (g[at + GEAR.shield + 3] > 0) putGP(g[at + GEAR.shield], g[at + GEAR.shield + 1], g[at + GEAR.shield + 2], P.shieldReach);
  if (g[at + GEAR.bow + 3] > 0) {
    const bx = g[at + GEAR.bow], by = g[at + GEAR.bow + 1], bz = g[at + GEAR.bow + 2];
    const ux = g[at + GEAR.bowUp], uy = g[at + GEAR.bowUp + 1], uz = g[at + GEAR.bowUp + 2];
    putGP(bx + ux * P.bowHalf, by + uy * P.bowHalf, bz + uz * P.bowHalf, P.gearPad);
    putGP(bx - ux * P.bowHalf, by - uy * P.bowHalf, bz - uz * P.bowHalf, P.gearPad);
    // (an arrow on the string reaches past the grip: the hand holds it back by the draw)
    putGP(bx + g[at + GEAR.bowFwd] * 0.25, by + g[at + GEAR.bowFwd + 1] * 0.25, bz + g[at + GEAR.bowFwd + 2] * 0.25, P.gearPad);
  }
  return gpN;
}

/** The pad a bound gives joint j (HERO_SHAPE reaches, mirrored by the shaders). */
export function jointPad(j: number, glide: number): number {
  return j === J.head ? P.headReach
    : j === J.pelvis || j === J.chest ? P.backReach
      : j === J.glider ? P.gliderReach * glide
        : P.limbPad;
}

/** The bounding sphere (centre xyz, radius) of joints `j` (x, y, z each) and gear `g` (GEAR layout), into `out`. */
function heroBounds(jt: Float64Array, g: Float64Array, glide: number, out: Float64Array): void {
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const nj = glide > 0 ? HERO_JOINTS : HERO_JOINTS - 1;
  const ng = gearPoints(g, 0);
  for (let k = 0; k < nj + ng; k++) {
    const x = k < nj ? jt[3 * k] : GP[4 * (k - nj)];
    const y = k < nj ? jt[3 * k + 1] : GP[4 * (k - nj) + 1];
    const z = k < nj ? jt[3 * k + 2] : GP[4 * (k - nj) + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = 0.5 * (minX + maxX), cyc = 0.5 * (minY + maxY), cz = 0.5 * (minZ + maxZ);
  let rad = 0;
  for (let k = 0; k < nj + ng; k++) {
    const q = k - nj;
    const dx = (k < nj ? jt[3 * k] : GP[4 * q]) - cx;
    const dy = (k < nj ? jt[3 * k + 1] : GP[4 * q + 1]) - cyc;
    const dz = (k < nj ? jt[3 * k + 2] : GP[4 * q + 2]) - cz;
    rad = Math.max(rad, Math.sqrt(dx * dx + dy * dy + dz * dz) + (k < nj ? jointPad(k, glide) : GP[4 * q + 3]));
  }
  out[0] = cx; out[1] = cyc; out[2] = cz; out[3] = rad;
}

/**
 * The gear's points a box round the hero must hold, from its packing
 * (HERO_UNIFORM.gear of `src`, camera-relative): GP holds x, y, z, pad for
 * each of the count returned (shared scratch: read it before the next call).
 */
export function packedGearPoints(src: ArrayLike<number>): { pts: Float64Array; n: number } {
  GPR.n = gearPoints(src, HERO_UNIFORM.gear);
  return GPR;
}
const GPR = { pts: GP, n: 0 };

/**
 * Pack the posed hero camera-relative for the shader (layout HERO_UNIFORM):
 * joints, torso and head frames (column-major mat3), bounding sphere, and
 * misc = (glider open, terrain height under the hero, sin yaw, cos yaw), the
 * dress (characters.ts) and the gear (GEAR). Differences are taken in fp64
 * before the fp32 store, so the body keeps sub-millimetre precision at any
 * distance from the world origin. (A spin attack's turn adds to the yaw.)
 */
export function packHeroUniforms(
  r: HeroRig, s: HeroState, camX: number, camY: number, camZ: number, out: Float32Array,
  dress: ArrayLike<number> = DEFAULT_DRESS,
): void {
  const yaw = s.yaw + r.spin;
  const sy = Math.sin(yaw), cy = Math.cos(yaw);
  const ox = s.x - camX, oy = s.y - PLAYER.groundBias - camY, oz = s.z - camZ;
  const l = r.local;
  for (let j = 0; j < HERO_JOINTS; j++) {
    const x = l[3 * j], y = l[3 * j + 1], z = l[3 * j + 2];
    out[3 * j] = ox + cy * x + sy * z;
    out[3 * j + 1] = oy + y;
    out[3 * j + 2] = oz - sy * x + cy * z;
  }
  const f = r.frames;
  for (let c = 0; c < 6; c++) {
    const x = f[3 * c], y = f[3 * c + 1], z = f[3 * c + 2];
    const o = HERO_UNIFORM.torso + 3 * c;
    out[o] = cy * x + sy * z;
    out[o + 1] = y;
    out[o + 2] = -sy * x + cy * z;
  }
  const b = r.bound, ob = HERO_UNIFORM.bound;
  out[ob] = ox + cy * b[0] + sy * b[2];
  out[ob + 1] = oy + b[1];
  out[ob + 2] = oz - sy * b[0] + cy * b[2];
  out[ob + 3] = b[3];
  const om = HERO_UNIFORM.misc;
  out[om] = r.glide;
  out[om + 1] = s.ground - camY;
  out[om + 2] = sy;
  out[om + 3] = cy;
  const od = HERO_UNIFORM.dress;
  out[od] = dress[0];
  out[od + 1] = dress[1];
  out[od + 2] = dress[2];
  out[od + 3] = 0;
  const g = r.gear;
  for (let v = 0; v < GEAR.length / 4; v++) {
    const i = 4 * v, o = HERO_UNIFORM.gear + i;
    const x = g[i], y = g[i + 1], z = g[i + 2];
    // (vec4 0, 3, 6 are positions: the grip, the shield's centre, the bow's grip)
    const p = v % 3 === 0 ? 1 : 0;
    out[o] = ox * p + cy * x + sy * z;
    out[o + 1] = oy * p + y;
    out[o + 2] = oz * p - sy * x + cy * z;
    out[o + 3] = g[i + 3];
  }
}
