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
 */

import { HERO_SHAPE as P, PLAYER } from '../constants';
import type { HeroState } from './locomotion';

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

/** Float32Array layout read by the engine's uniform uploads. */
export const HERO_UNIFORM = { joints: 0, torso: 54, head: 63, bound: 72, misc: 76, length: 80 } as const;

const POSES = 5; // ground, air, glide, climb, swim
const W_GROUND = 0, W_AIR = 1, W_GLIDE = 2, W_CLIMB = 3, W_SWIM = 4;
const TAU = Math.PI * 2;

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
  land: number;        // landing squash 0..1
  glide: number;       // paraglider canopy open 0..1
  landings: number;
  lastYaw: number;
  lastSpeed: number;
  snap: boolean;       // next update jumps to the target pose (spawn, view switch)
  weights: Float64Array;
  local: Float64Array; // blended joints, hero-local
  pose: Float64Array;  // one pose, scratch
  frames: Float64Array; // torso then head frame: columns right, up, forward (hero-local)
  bound: Float64Array;  // hero-local centre xyz, radius
};

export function createHeroRig(): HeroRig {
  return {
    time: 0, phase: 0, climbPhase: 0, swimPhase: 0, speed: 0,
    sprint: 0, crouch: 0, slide: 0, tired: 0, lean: 0, bank: 0, land: 0, glide: 0,
    landings: 0, lastYaw: 0, lastSpeed: 0, snap: true,
    weights: new Float64Array(POSES),
    local: new Float64Array(HERO_JOINTS * 3),
    pose: new Float64Array(HERO_JOINTS * 3),
    frames: new Float64Array(18),
    bound: new Float64Array(4),
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
  let pelvisY = P.pelvisH - 0.3 * r.crouch - 0.2 * r.land - 0.08 * r.slide
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

// Wall axes of the climb pose being built: up the wall, and away from it.
let tuy = 0, tuz = 1, nwy = 1, nwz = 0;

/** Joint `j` at `x` to the right, `along` up the wall and `off` out from it. */
function atWall(o: Float64Array, j: number, x: number, along: number, off: number): void {
  set(o, j, x, along * tuy + off * nwy, along * tuz + off * nwz);
}

/** Facing uphill (+z) on a wall rising m per metre; the root is on its surface. */
function poseClimb(r: HeroRig, o: Float64Array, m: number): void {
  const n = Math.sqrt(1 + m * m);
  tuy = m / n; tuz = 1 / n;
  nwy = 1 / n; nwz = -m / n;
  const c = r.climbPhase;
  atWall(o, J.pelvis, 0, 0.92, 0.2);
  atWall(o, J.chest, 0, 0.92 + P.spine, 0.22);
  torsoFrame(o, 0, -nwy, -nwz); // chest faces the wall
  upperBody(o, -0.2);
  inTorso(o, J.hipL, J.pelvis, -P.hipW, -P.hipDrop, 0);
  inTorso(o, J.hipR, J.pelvis, P.hipW, -P.hipDrop, 0);
  for (let k = 0; k < 2; k++) {
    const side = k === 0 ? -1 : 1;
    const reachArm = 0.5 + 0.5 * Math.sin(c + (side > 0 ? Math.PI : 0));
    const reachLeg = 0.5 + 0.5 * Math.sin(c + (side > 0 ? 0 : Math.PI));
    atWall(o, side < 0 ? J.handL : J.handR, side * 0.3, 0.92 + P.spine + 0.1 + 0.28 * reachArm, 0.06);
    ik(o, side < 0 ? J.shoulderL : J.shoulderR, side < 0 ? J.elbowL : J.elbowR, side < 0 ? J.handL : J.handR,
      P.upperArm, P.forearm, side * 0.8, nwy, nwz);
    atWall(o, side < 0 ? J.ankleL : J.ankleR, side * 0.17, 0.22 + 0.2 * reachLeg, 0.07);
    ik(o, side < 0 ? J.hipL : J.hipR, side < 0 ? J.kneeL : J.kneeR, side < 0 ? J.ankleL : J.ankleR,
      P.thigh, P.shin, side * 0.7, 0.6 * nwy, 0.6 * nwz);
    toe(o, side < 0 ? J.ankleL : J.ankleR, side < 0 ? J.toeL : J.toeR,
      0, 0.03 * tuy - 0.12 * nwy, 0.03 * tuz - 0.12 * nwz);
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

/**
 * Advance the animation by dt from the locomotion state and rebuild the
 * blended hero-local skeleton, frames and bounding sphere.
 */
export function updateHeroRig(r: HeroRig, s: HeroState, dt: number): void {
  const hv = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
  const snap = r.snap;
  r.snap = false;
  r.time += dt;

  r.speed += (hv - r.speed) * follow(snap, 12, dt);
  const target = s.move === 'ground' ? W_GROUND : s.move === 'air' ? W_AIR
    : s.move === 'glide' ? W_GLIDE : s.move === 'climb' ? W_CLIMB : W_SWIM;
  let sum = 0;
  const kw = follow(snap, s.move === 'air' || target === W_GROUND ? 14 : 9, dt);
  for (let i = 0; i < POSES; i++) {
    r.weights[i] += ((i === target ? 1 : 0) - r.weights[i]) * kw;
    sum += r.weights[i];
  }
  for (let i = 0; i < POSES; i++) r.weights[i] /= sum;
  r.glide += ((s.move === 'glide' ? 1 : 0) - r.glide) * follow(snap, s.move === 'glide' ? 5 : 10, dt);
  if (r.glide < 1e-3) r.glide = 0;

  r.sprint += ((s.sprinting ? 1 : 0) - r.sprint) * follow(snap, 6, dt);
  r.crouch += ((s.sneaking ? 1 : 0) - r.crouch) * follow(snap, 8, dt);
  r.slide += ((s.sliding ? 1 : 0) - r.slide) * follow(snap, 8, dt);
  r.tired += ((s.exhausted && s.move === 'ground' ? 1 : 0) - r.tired) * follow(snap, 3, dt);
  if (s.landings !== r.landings) {
    r.landings = s.landings;
    if (!snap) r.land = Math.max(r.land, Math.min(1, 0.15 + s.impact / 14));
  }
  r.land *= Math.exp(-7 * dt);
  const accel = snap ? 0 : (hv - r.lastSpeed) / Math.max(dt, 1e-3);
  r.lastSpeed = hv;
  const leanT = 0.05 + 0.03 * r.speed + 0.012 * Math.max(-8, Math.min(8, accel));
  r.lean += (Math.max(-0.1, Math.min(0.42, leanT)) - r.lean) * follow(snap, 6, dt);
  const yawRate = snap ? 0 : wrapAngle(s.yaw - r.lastYaw) / Math.max(dt, 1e-3);
  r.lastYaw = s.yaw;
  r.bank += (Math.max(-0.3, Math.min(0.3, yawRate * r.speed * 0.02)) - r.bank) * follow(snap, 8, dt);

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
  const m = Math.sqrt(s.gx * s.gx + s.gz * s.gz);

  const out = r.local, o = r.pose;
  const grounded = s.move === 'ground';
  const climbM = Math.max(m, 1.0);
  out.fill(0);
  for (let i = 0; i < POSES; i++) {
    const w = r.weights[i];
    if (w < 1e-3) continue;
    o.fill(0);
    if (i === W_GROUND) poseGround(r, o, grounded ? ga : 0, grounded ? gb : 0);
    else if (i === W_AIR) poseAir(r, o, s.vy);
    else if (i === W_GLIDE) poseGlide(r, o);
    else if (i === W_CLIMB) poseClimb(r, o, climbM);
    else poseSwim(r, o);
    if (i !== W_GLIDE) {
      // Only the glide pose places the canopy; the others park it over the hands.
      set(o, J.glider,
        0.5 * (o[3 * J.handL] + o[3 * J.handR]),
        Math.max(o[3 * J.handL + 1], o[3 * J.handR + 1]) + 0.42,
        0.5 * (o[3 * J.handL + 2] + o[3 * J.handR + 2]));
    }
    for (let q = 0; q < out.length; q++) out[q] += w * o[q];
  }

  // Frames of the blended skeleton: torso faces forward, head leans upright.
  torsoFrame(out, 0, 0, 1);
  for (let q = 0; q < 9; q++) r.frames[q] = F[q];
  let ux = out[3 * J.head] - out[3 * J.chest];
  let uy = out[3 * J.head + 1] - out[3 * J.chest + 1] + 0.12;
  let uz = out[3 * J.head + 2] - out[3 * J.chest + 2];
  let l = Math.sqrt(ux * ux + uy * uy + uz * uz);
  ux /= l; uy /= l; uz /= l;
  let fx = -ux * uz, fy = -uy * uz, fz = 1 - uz * uz;
  l = Math.sqrt(fx * fx + fy * fy + fz * fz);
  if (l < 1e-6) { fx = r.frames[6]; fy = r.frames[7]; fz = r.frames[8]; l = 1; }
  fx /= l; fy /= l; fz /= l;
  const fr = r.frames;
  fr[9] = uy * fz - uz * fy; fr[10] = uz * fx - ux * fz; fr[11] = ux * fy - uy * fx;
  fr[12] = ux; fr[13] = uy; fr[14] = uz;
  fr[15] = fx; fr[16] = fy; fr[17] = fz;

  // Bounding sphere: centre of the joints' box, radius to the farthest
  // primitive reach (HERO_SHAPE.*Reach / limbPad, mirrored in the shader).
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const nj = r.glide > 0 ? HERO_JOINTS : HERO_JOINTS - 1;
  for (let j = 0; j < nj; j++) {
    const x = out[3 * j], y = out[3 * j + 1], z = out[3 * j + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = 0.5 * (minX + maxX), cyc = 0.5 * (minY + maxY), cz = 0.5 * (minZ + maxZ);
  let rad = 0;
  for (let j = 0; j < nj; j++) {
    const dx = out[3 * j] - cx, dy = out[3 * j + 1] - cyc, dz = out[3 * j + 2] - cz;
    const pad = j === J.head ? P.headReach
      : j === J.pelvis || j === J.chest ? P.backReach
      : j === J.glider ? P.gliderReach * r.glide
      : P.limbPad;
    rad = Math.max(rad, Math.sqrt(dx * dx + dy * dy + dz * dz) + pad);
  }
  r.bound[0] = cx; r.bound[1] = cyc; r.bound[2] = cz; r.bound[3] = rad;
}

/**
 * Pack the posed hero camera-relative for the shader (layout HERO_UNIFORM):
 * joints, torso and head frames (column-major mat3), bounding sphere, and
 * misc = (glider open, terrain height under the hero, sin yaw, cos yaw).
 * Differences are taken in fp64 before the fp32 store, so the body keeps
 * sub-millimetre precision at any distance from the world origin.
 */
export function packHeroUniforms(
  r: HeroRig, s: HeroState, camX: number, camY: number, camZ: number, out: Float32Array,
): void {
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
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
}
