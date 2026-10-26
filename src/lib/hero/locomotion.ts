/**
 * Third-person hero locomotion: run, sprint, sneak, jump, paraglide, climb,
 * swim, shield surf and a stamina wheel, after the open-world Zelda games.
 * ---------------------------------------------------------------------------
 * Same contract as updatePlayer() (terrainMath.ts §31): deterministic, the
 * state is mutated in place (no allocation per tick), terrain queries go
 * through the CPU twin's elevation() on fp64 global coordinates — one ground
 * sample plus two forward differences per tick. The root (x, y, z) sits
 * between the feet; when grounded y = E + PLAYER.groundBias like the first-
 * person eye, so switching views keeps the same collision contract. Up in
 * the sky realm (skyworld/realm.ts) an island's top or an arch's deck is the
 * ground instead, and a sky lake or stream the water; the Eyries' structures
 * (skyworld/colliders.ts) stop the hero at their walls and bear it on their
 * plazas, bridge decks, stairs, ledges and roofs. Rocks (rocks/field.ts) stop
 * it by their own shape and bear it on their tops.
 *
 * The same body carries the Bokoblins (combat/foes.ts), steered by their AI
 * through the optional HeroInput fields (a heading to face, a speed factor,
 * no climbing), and the combat layer (combat/heroCombat.ts) launches it into
 * dodges and knockbacks and puts it on the shield to surf.
 */

import { elevation, lastFlow, lastIce, lastLava, surfaceAt, type PlayerState } from '../terrainMath';
import { floorHeight, skyFloorUnder, skyGround, skyWater } from '../skyworld/realm';
import { collideModelIsle } from '../skymodels';
import { ruinSiteAt, toBoxLocal } from '../ruins';
import { trunkSection, trunksNear, type Trunk } from '../forest/collision';
import { collideRocks, rockBodiesNear, rockFloorSlope, rockTopAt, type RockBody } from '../rocks/field';
import { eyrieIfBuilt } from '../skyworld/eyrie/eyrie';
import { collide } from '../skyworld/colliders';
import { COMBAT, HERO, PLAYER, SKY_REALM, LANDS, CLIMB, SKYDIVE } from '../constants';

/** Scratch for the ground and the water under the hero. */
const SURF: [number, number] = [0, 0];

export type HeroMove = 'ground' | 'air' | 'glide' | 'climb' | 'swim' | 'surf';

export type HeroState = {
  x: number; y: number; z: number;  // root, global fp64
  vx: number; vy: number; vz: number;
  yaw: number;                      // facing: forward = (sin yaw, 0, cos yaw), as the camera
  move: HeroMove;
  stamina: number;                  // 0..1, one wheel
  exhausted: boolean;               // ran dry: no stamina actions until the wheel is full again
  rest: number;                     // seconds since stamina was last spent
  sprinting: boolean;
  sneaking: boolean;
  sliding: boolean;
  surfing: boolean;                 // on the shield: surfing on the ground, or in the air after a surf jump
  jumpHeld: boolean;                // jumps, the glider and climb jumps fire on the press
  surge: number;                    // metres of climb-jump surge left
  ground: number;                   // E under the root at the last sample
  gx: number; gz: number;           // ∇E there (forward differences)
  water: number;                    // the water's surface there: the lakes and sea, a sky lake or stream, or −∞ (dry)
  flowX: number; flowZ: number;     // a river's current there (m/s, world xz); 0 on still water and land
  impact: number;                   // downward speed at the last landing (m/s)
  landings: number;                 // counts landings so the rig reacts to each once
  spring: number;                   // the lift of the springy puff underfoot (m; skyworld/puffs.ts), 0: none
  // Skydiving (a sub-state of 'air'): spread-eagle at dive 0, a head-first nosedive at 1.
  skydive: boolean;
  dive: number;                     // 0..1 posture, eased toward the sprint key
  flare: number;                    // s left of a flare out of a fast dive (0: none)
  recover: number;                  // s left of a hard landing's roll or squat (0: none)…
  rolling: boolean;                 // … rolling on (else squatting)
  // Climbing (climbSurface): the root on the surface where the hips' level meets it, facing it.
  tree: Trunk | null;               // the trunk climbed…
  wall: boolean;                    // … or a wall (the land's, an island's, a rock's)…
  wallNX: number; wallNZ: number;   // … its outward normal at the hold (horizontal, unit)…
  wallInv: number;                  // … and how far in it leans going up (m per m: 0 sheer, 1 at 45°)…
  wallFX: number; wallFY: number; wallFZ: number; // … where it was last framed (frameWall)
  wallTick: number;                 // ticks on the wall (the ground under it is looked at every few)
  climbVA: number; climbVS: number; // the climb's speed up the surface and across it (m/s), eased toward the stick's
  touch: Trunk | null;              // the trunk the body touched at the last ground sample…
  bump: number;                     // … and how fast it ran into it (m/s; the collision took that speed away)
  perch: Trunk | null;              // the tree whose top the hero stands on (perched there: it balances, sneaks off onto the bark)
  // The surface round the hips as the rig takes it (and the snapshot carries it): a cylinder that tapers and leans.
  climbX: number; climbZ: number;   // its axis at the hips (a trunk's centre; vaulting: where the vault ends)…
  climbR: number;                   // … its radius there (0: not climbing; a flat wall's, large)…
  climbTaper: number;               // … how the radius changes going up from there (m per m)…
  climbLeanX: number; climbLeanZ: number; // … how the centre moves going up (m per m)…
  climbTaperDown: number;           // … and the same going down from there
  climbLeanDownX: number; climbLeanDownZ: number;
  climbTop: number;                 // … and the height of its top (a trunk's; a wall's ledge, vaulting onto it)
  grip: number;                     // seconds left of a mid-air grab's slide down the bark…
  slip: number;                     // … and how far it slides (m)
  mantle: number;                   // 0..1 through the vault onto the top (0: none)
  mx: number; my: number; mz: number; // where the vault started
};

export type HeroInput = {
  forward: number; strafe: number;  // −1..1, camera-relative
  sprint: boolean;
  jump: boolean;
  sneak: boolean;
  camYaw: number;
  /** A heading to face whatever the way it moves (the shield up toward a foe, the bow drawn); NaN or absent: the way it moves. */
  face?: number;
  /** × the speed on foot (the shield up, the bow drawn, a swing's lunge); absent: 1. */
  speedScale?: number;
  /** Never grab a wall (foes; the hero mid-swing). */
  noClimb?: boolean;
};

export type HeroAction =
  | 'idle' | 'walk' | 'run' | 'sprint' | 'sneak' | 'tired' | 'slide'
  | 'jump' | 'fall' | 'glide' | 'climb' | 'hang' | 'swim' | 'dash' | 'surf'
  | 'skydive' | 'dive' | 'roll' | 'squat';

const TAU = Math.PI * 2;
const GRAD_H = 0.25; // m — forward-difference step of the ground gradient

function wrapAngle(a: number): number {
  return a - TAU * Math.round(a / TAU);
}

/** Fraction of the remaining gap an exponential approach at `rate` covers in dt. */
function approach(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}

function turnToward(s: HeroState, targetYaw: number, rate: number, dt: number): void {
  s.yaw = wrapAngle(s.yaw + wrapAngle(targetYaw - s.yaw) * approach(rate, dt));
}

const HERO_RADIUS = 0.35; // m — body radius against ruin walls and tree trunks
const STEP_UP = 0.5;      // m — ruin ledges this high are stepped onto
const BODY_HEIGHT = 1.7;

type Body = { x: number; z: number; vx: number; vz: number };

/**
 * Ruins: push a body whose feet are at `feetY` out of any stone box it
 * overlaps (and drop the velocity into that face, so it slides along walls);
 * return the highest box top it stands over within a step's reach, or −∞.
 */
function collideRuins(s: Body, feetY: number, seed: number): number {
  const site = ruinSiteAt(s.x, s.z, seed);
  if (!site) return -Infinity;
  let floor = -Infinity;
  for (const b of site.boxes) {
    const [u, v] = toBoxLocal(b, s.x, s.z);
    const top = b.y + b.hy, bottom = b.y - b.hy;
    if (Math.abs(u) <= b.hx && Math.abs(v) <= b.hz && feetY >= top - STEP_UP) {
      floor = Math.max(floor, top);
      continue;
    }
    if (feetY >= top - STEP_UP || feetY + BODY_HEIGHT <= bottom) continue;
    const du = u - Math.max(-b.hx, Math.min(b.hx, u));
    const dv = v - Math.max(-b.hz, Math.min(b.hz, v));
    const dist = Math.sqrt(du * du + dv * dv);
    if (dist >= HERO_RADIUS) continue;
    let nu: number, nv: number, push: number;
    if (dist > 1e-6) {
      nu = du / dist; nv = dv / dist; push = HERO_RADIUS - dist;
    } else {
      // Centre inside the footprint: leave by the nearest face.
      const pu = b.hx - Math.abs(u), pv = b.hz - Math.abs(v);
      if (pu < pv) { nu = u < 0 ? -1 : 1; nv = 0; push = pu + HERO_RADIUS; } else { nu = 0; nv = v < 0 ? -1 : 1; push = pv + HERO_RADIUS; }
    }
    const cs = Math.cos(b.yaw), sn = Math.sin(b.yaw);
    const wx = cs * nu + sn * nv, wz = -sn * nu + cs * nv;
    s.x += wx * push;
    s.z += wz * push;
    const vn = s.vx * wx + s.vz * wz;
    if (vn < 0) { s.vx -= vn * wx; s.vz -= vn * wz; }
  }
  return floor;
}

/**
 * The Eyries' structures (skyworld/colliders.ts): push a body whose feet are at `feetY` out of their
 * walls; return the highest floor it stands on within a step (a plaza, a deck, a stair, a roof), or −∞.
 * Only an Eyrie already built is asked (the renderer builds the nearest first).
 */
function collideStructures(s: Body, feetY: number, seed: number): number {
  // (the land's model isles: their walls; their floors come by skyFloorUnder)
  collideModelIsle(s, feetY, seed, HERO_RADIUS, STEP_UP);
  const e = eyrieIfBuilt(Math.floor(s.x / SKY_REALM.archCell), Math.floor(s.z / SKY_REALM.archCell), seed);
  return e ? collide(e.colliders, s, feetY, HERO_RADIUS, BODY_HEIGHT) : -Infinity;
}

const nearTrunks: Trunk[] = [];
const SEC = new Float64Array(3); // a trunk's cross-section: centre x, z, radius
const HIPS = 0.9;                // m over the feet: where a body meets a trunk (its cross-section there)
const SHOULDERS = 1.5;           // m over a trunk's foot: below, a body meets its circle (Trunk.r: flare, fins and lean)
/** The trunk collideTrees found the body touching (or null) and the speed it ran into it at: read straight after the call. */
let touched: Trunk | null = null, bumped = 0, perched: Trunk | null = null;

/**
 * Trees: push a body whose feet are at `feetY` out of any trunk it overlaps
 * between feet and head — near the ground its circle (Trunk.r, wide enough for
 * the flare, the fins and the lean from knee to shoulder), higher up its
 * cross-section at the hips (trunkSection: the tapering, leaning trunk) —
 * and drop the velocity into the trunk, so it slides round it. Crowns and
 * bushes do not block. Return the top of a trunk the body stands on (within
 * CLIMB.perch of its axis, after a vault or a landing there), or −∞; the
 * nearest trunk touched is left in `touched`. (Rocks: collideRocks, by their shape.)
 */
function collideTrees(s: Body, feetY: number, seed: number): number {
  let floor = -Infinity, nearest = Infinity;
  touched = null;
  bumped = 0;
  perched = null;
  for (const t of trunksNear(s.x, s.z, seed, nearTrunks, false)) {
    const top = t.y + t.top;
    if (feetY > top - 0.3 && feetY < top + 0.5) {
      trunkSection(t, t.top, SEC);
      const px = s.x - SEC[0], pz = s.z - SEC[1];
      if (px * px + pz * pz < CLIMB.perch * CLIMB.perch) {
        if (top > floor) { floor = top; perched = t; }
        continue;
      }
    }
    if (feetY > top || feetY + BODY_HEIGHT < t.y - 0.5) continue;
    const h = feetY - t.y + HIPS;
    if (h > SHOULDERS) trunkSection(t, h, SEC);
    else { SEC[0] = t.x; SEC[1] = t.z; SEC[2] = t.r; }
    const dx = s.x - SEC[0], dz = s.z - SEC[1];
    const d = Math.sqrt(dx * dx + dz * dz), min = HERO_RADIUS + SEC[2];
    const nx = d > 1e-6 ? dx / d : 1, nz = d > 1e-6 ? dz / d : 0;
    const vn = s.vx * nx + s.vz * nz;
    if (d < min + 0.05 && d - SEC[2] < nearest) { nearest = d - SEC[2]; touched = t; bumped = Math.max(0, -vn); }
    if (d >= min) continue;
    s.x += nx * (min - d);
    s.z += nz * (min - d);
    if (vn < 0) { s.vx -= vn * nx; s.vz -= vn * nz; }
  }
  return floor;
}

/**
 * First person (updatePlayer's extra ground): tree trunks and ruins, with
 * their walls, and the top of a sky island (or a tree) under the feet.
 */
export function firstPersonGround(p: PlayerState, feetY: number, seed: number): number {
  const tree = collideTrees(p, feetY, seed);
  const rock = collideRocks(p, feetY, seed, HERO_RADIUS, STEP_UP, BODY_HEIGHT, HERO.walkSlope);
  const box = Math.max(collideRuins(p, feetY, seed), collideStructures(p, feetY, seed), rock, tree);
  const sky = skyGround(p.x, feetY, p.z, seed);
  return Math.max(box, sky === sky ? sky : -Infinity);
}

function sampleGround(s: HeroState, seed: number): void {
  const tree = collideTrees(s, s.y, seed);
  s.touch = touched;
  const onTop = perched;
  s.bump = bumped;
  const rock = collideRocks(s, s.y, seed, HERO_RADIUS, STEP_UP, BODY_HEIGHT, HERO.walkSlope);
  const rgx = rockFloorSlope.gx, rgz = rockFloorSlope.gz;
  const boxTop = Math.max(collideRuins(s, s.y, seed), collideStructures(s, s.y, seed), tree);
  sampleSurface(s, seed);
  s.perch = null;
  if (boxTop > s.ground && boxTop >= rock) {
    s.ground = boxTop;
    s.gx = 0;
    s.gz = 0;
    if (boxTop === tree) s.perch = onTop;
  } else if (rock > s.ground) {
    // on a rock: its top is the ground, and its slope (a steep one slides the hero off)
    s.ground = rock;
    s.gx = rgx;
    s.gz = rgz;
  }
}

function sampleSurface(s: HeroState, seed: number): void {
  const f = skyFloorUnder(s.x, s.y, s.z, seed);
  s.spring = f && 'spring' in f ? f.spring : 0;
  if (f) {
    const t = floorHeight(f, s.x, s.z);
    // (past an arch's edge or an island's rim the floor ends: no slope across it)
    const tx = floorHeight(f, s.x + GRAD_H, s.z), tz = floorHeight(f, s.x, s.z + GRAD_H);
    s.gx = tx === tx ? (tx - t) / GRAD_H : 0;
    s.gz = tz === tz ? (tz - t) / GRAD_H : 0;
    s.ground = t;
    const w = skyWater(s.x, s.y, s.z, seed);
    s.water = w === w ? w : -Infinity;
    s.flowX = 0;
    s.flowZ = 0;
    return;
  }
  // the ground, and the water there: the sea, a lake or a river's pool at its own level (under the ground where it is dry)
  surfaceAt(s.x, s.z, seed, SURF);
  const e = SURF[0];
  s.water = SURF[1];
  const [fx, fz, fs] = lastFlow();
  s.flowX = fx * fs;
  s.flowZ = fz * fs;
  // Ice (frozen lakes and seas in the cold) and lava (a crater's lake, a volcano's rivers): their surface
  // is ground (no swimming in either).
  if (s.water > e && (lastIce() || lastLava() > 0.5)) {
    s.ground = s.water;
    s.water = -Infinity;
    s.gx = 0;
    s.gz = 0;
    return;
  }
  s.gx = (elevation(s.x + GRAD_H, s.z, seed, 0) - e) / GRAD_H;
  s.gz = (elevation(s.x, s.z + GRAD_H, seed, 0) - e) / GRAD_H;
  s.ground = e;
}

function deepWater(s: HeroState): boolean {
  return s.ground < s.water - HERO.swimDepth;
}

/**
 * The launch speed whose jump tops out h above (gravity g, eased to apexGravity · g while |vy| is
 * under apexBand: the air step's hang at the top).
 */
function springSpeed(h: number): number {
  const g = PLAYER.gravity, b = HERO.apexBand;
  return Math.sqrt(2 * g * (h - (b * b) / (2 * g * HERO.apexGravity)) + b * b);
}

function land(s: HeroState, wx: number, wz: number, mag: number, climbs: boolean, seed: number): void {
  s.impact = Math.max(0, -s.vy);
  s.landings++;
  s.y = s.ground + PLAYER.groundBias;
  s.vy = 0;
  s.move = s.surfing ? 'surf' : 'ground';
  // a springy puff throws the hero straight back up, its lift high
  if (s.spring > 0) {
    s.vy = springSpeed(s.spring);
    s.move = 'air';
    s.rolling = false;
    s.recover = 0;
    return;
  }
  // Hard enough: roll on, the way it was going (moving, or the stick held), or squat under a heavy one.
  const hv = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
  s.rolling = !s.surfing && s.impact >= SKYDIVE.rollMin && s.impact < SKYDIVE.squatMin && (hv > SKYDIVE.rollMove || mag > 0.2);
  s.recover = s.rolling ? SKYDIVE.rollTime : !s.surfing && s.impact >= SKYDIVE.squatMin ? SKYDIVE.squatTime : 0;
  if (s.rolling) s.yaw = hv > SKYDIVE.rollMove ? Math.atan2(s.vx, s.vz) : Math.atan2(wx, wz);
  // (on ground too steep to stand on, pushing into it: a wall to climb)
  if (!s.surfing && Math.sqrt(s.gx * s.gx + s.gz * s.gz) > HERO.walkSlope) grabWall(s, wx, wz, mag, climbs, false, 0, 0, seed);
}

// ---- climbing trees ----------------------------------------------------------
//
// A trunk is a vertical cylinder whose centre and radius follow its height (trunkSection). Climbing, the
// root sits on the bark at an angle round the axis and a height, facing the axis; each tick moves the
// angle (ω = v / (R + clearance), so the hips go round at the wrap speed) and the height, then puts the
// root back on the bark from the cross-section there, so no error builds up however long the hero circles.

/** The pelvis' clearance from bark of radius r (m): hugging a thin pole, held off a wide trunk (rig.ts poseTrunk). */
export function climbClearance(r: number): number {
  const T = CLIMB;
  const t = Math.min(1, Math.max(0, (r - T.poleR) / (T.wideR - T.poleR)));
  return T.clearPole + (T.clearWide - T.clearPole) * t * t * (3 - 2 * t);
}

const TAPER_UP = 0.6, TAPER_DOWN = 0.8; // m above and below the hips the trunk's taper and lean are measured to

/** Put the root on the bark of trunk `t` at angle `ang` round its axis (0: +z), feet at `feet`. */
function holdOnTrunk(s: HeroState, t: Trunk, ang: number, feet: number): void {
  // (the taper and the lean above and below the hips, for the rig to put the hands and feet on the bark at
  // their heights: up to where the hands hold, down to where the feet do)
  const h = feet - t.y + HIPS;
  trunkSection(t, h, SEC);
  const hx = SEC[0], hz = SEC[1], hr = SEC[2];
  trunkSection(t, h + TAPER_UP, SEC);
  s.climbTaper = (SEC[2] - hr) / TAPER_UP; s.climbLeanX = (SEC[0] - hx) / TAPER_UP; s.climbLeanZ = (SEC[1] - hz) / TAPER_UP;
  trunkSection(t, h - TAPER_DOWN, SEC);
  s.climbTaperDown = (hr - SEC[2]) / TAPER_DOWN; s.climbLeanDownX = (hx - SEC[0]) / TAPER_DOWN; s.climbLeanDownZ = (hz - SEC[1]) / TAPER_DOWN;
  trunkSection(t, h, SEC);
  s.climbX = SEC[0]; s.climbZ = SEC[1]; s.climbR = SEC[2];
  s.x = SEC[0] + SEC[2] * Math.sin(ang);
  s.z = SEC[1] + SEC[2] * Math.cos(ang);
  s.y = feet + PLAYER.groundBias;
  s.yaw = wrapAngle(ang + Math.PI);
}

/**
 * Leaving a wall for the air or the water: the root from its hold (where the wall meets the hips' level, in
 * a slope leaning back over the feet's) out to where the feet are, in front of the wall.
 */
function outOfWall(s: HeroState): void {
  if (!s.wall) return;
  const out = HIPS * Math.max(0, -(s.climbLeanDownX * s.wallNX + s.climbLeanDownZ * s.wallNZ)) + HERO_RADIUS + 0.05;
  s.x += s.wallNX * out;
  s.z += s.wallNZ * out;
}

/** Off the surface: the climb state cleared. */
function leaveClimb(s: HeroState): void {
  s.tree = null;
  s.wall = false;
  s.climbVA = 0; s.climbVS = 0;
  s.wallFX = NaN;
  s.climbR = 0;
  s.climbTaper = 0; s.climbLeanX = 0; s.climbLeanZ = 0; s.climbTaperDown = 0; s.climbLeanDownX = 0; s.climbLeanDownZ = 0;
  s.grip = 0;
  s.mantle = 0;
  s.surge = 0;
}

/**
 * The body touching a trunk grabs it: on the ground pushing the stick into it, where it stands (the climb
 * starts from the foot of the tree, however fast the run into it); in the air (or under the glider) also
 * moving into it, no faster than latchSpeed, wherever it meets the bark — a jump at a tree catches it
 * higher up — sliding down the bark a little as the grip takes (v² / 2a). Stumps, saplings and trunks whose
 * top is within reach are not climbed.
 */
function grabTrunk(s: HeroState, wx: number, wz: number, mag: number, ok: boolean, air: boolean, seed: number): boolean {
  const t = s.touch;
  if (!ok || !t || !t.prof || t.top < CLIMB.minTop) return false;
  const feet = s.y - PLAYER.groundBias;
  if (feet + 1.8 > t.y + t.top || feet + HIPS < t.y) return false;
  trunkSection(t, feet - t.y + HIPS, SEC);
  const dx = s.x - SEC[0], dz = s.z - SEC[1];
  const d = Math.sqrt(dx * dx + dz * dz);
  if (d < 1e-6) return false;
  const nx = dx / d, nz = dz / d;
  const push = mag > 0 ? -(wx * nx + wz * nz) : 0;
  if (air) {
    // (s.bump: the speed it met the trunk at, before the collision took it away)
    const hv = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
    const speed = Math.sqrt(hv * hv + s.bump * s.bump + s.vy * s.vy);
    if (!(push > CLIMB.grab || s.bump > 0.5) || speed > CLIMB.latchSpeed) return false;
    const [lo, hi] = CLIMB.latchSlip;
    s.grip = CLIMB.latchTime;
    s.slip = Math.min(hi, Math.max(lo, (s.vy < 0 ? s.vy * s.vy : 0) / (2 * CLIMB.latchDecel)));
    s.surge = 0;
  } else {
    if (!(push > CLIMB.grab)) return false;
    s.grip = 0;
    s.surge = 0;
  }
  s.move = 'climb';
  s.surfing = false;
  s.tree = t;
  s.climbTop = t.y + t.top;
  s.mantle = 0;
  s.vx = 0; s.vy = 0; s.vz = 0;
  holdOnTrunk(s, t, Math.atan2(nx, nz), feet);
  // (on a slope the bark's foot may stand higher than where the feet were)
  sampleSurface(s, seed);
  s.y = Math.max(s.y, s.ground + PLAYER.groundBias);
  return true;
}

/**
 * Standing on a trunk's top (s.perch) and stepping toward its edge: the hero balances there, held within
 * CLIMB.perch of the axis — or, with the sneak key, lowers itself over the edge onto the bark just below
 * the top and hangs there, climbing (true: the move is now the climb's).
 */
function perchEdge(s: HeroState, drop: boolean): boolean {
  const t = s.perch!;
  trunkSection(t, t.top, SEC);
  let dx = s.x - SEC[0], dz = s.z - SEC[1];
  const d = Math.sqrt(dx * dx + dz * dz), edge = CLIMB.perch - 0.05;
  if (d <= edge) return false;
  dx /= d; dz /= d;
  if (drop && t.top >= CLIMB.minTop) {
    s.move = 'climb';
    s.tree = t;
    s.climbTop = t.y + t.top;
    s.perch = null;
    s.mantle = 0; s.grip = 0; s.surge = 0;
    s.vx = 0; s.vy = 0; s.vz = 0;
    holdOnTrunk(s, t, Math.atan2(dx, dz), s.climbTop - 1.6);
    return true;
  }
  s.x = SEC[0] + dx * edge;
  s.z = SEC[1] + dz * edge;
  const vn = s.vx * dx + s.vz * dz;
  if (vn > 0) { s.vx -= vn * dx; s.vz -= vn * dz; }
  return false;
}

function easeOutCubic(x: number): number {
  return 1 - (1 - x) * (1 - x) * (1 - x);
}

// ---- climbing walls ------------------------------------------------------------
//
// The land's ground (or a sky island's top) and the rocks standing on it are one solid under one surface:
// solidTop, their highest top at a point. A wall is where that surface is steeper than walkSlope. The hero
// holds it where the solid comes up to the hips' level, facing it, and moves along that level's contour
// across it and from level to level up and down it, finding the contact afresh every tick. The wall round
// the hold is framed for the rig like a trunk: a cylinder (a convex level's radius, a flat or hollow one's
// FLAT) leaning as the wall does above and below the hips.

const climbRocks: RockBody[] = [];
let rocksX = NaN, rocksZ = NaN, rocksSeed = NaN;

/** Gather the rocks a body at (x, z) can climb on (kept while it stays within a metre of where they were gathered). */
function rocksFor(x: number, z: number, seed: number): void {
  if (seed === rocksSeed && Math.abs(x - rocksX) < 1 && Math.abs(z - rocksZ) < 1) return;
  rocksX = x; rocksZ = z; rocksSeed = seed;
  rockBodiesNear(x, z, seed, 3, climbRocks);
}

/** The top of the solid at (x, z) for a body at height y: the ground (a sky island's top under y), or a rock standing there if higher; −∞ over the void. */
function solidTop(x: number, z: number, y: number, seed: number): number {
  const f = skyFloorUnder(x, y + 2, z, seed);
  let top = f ? floorHeight(f, x, z) : elevation(x, z, seed, 0);
  if (top !== top) top = -Infinity;
  for (const b of climbRocks) {
    if (b.state === 'flying' || b.state === 'rolling') continue;
    const r = rockTopAt(b, x, z);
    if (r > top) top = r;
  }
  return top;
}

const PROBE = 0.12;                      // m between a contact search's steps (then halved to ~2 mm)
const SPOT = new Float64Array(2);
/** The first point from (x, z) along (dx, dz) (unit), within len, where the solid comes up to height y, into SPOT; false: none. */
function touchAt(x: number, z: number, y: number, dx: number, dz: number, len: number, seed: number): boolean {
  if (solidTop(x, z, y, seed) >= y) { SPOT[0] = x; SPOT[1] = z; return true; }
  let a = 0, b = NaN;
  for (let t = PROBE; ; t += PROBE) {
    const u = Math.min(t, len);
    if (solidTop(x + dx * u, z + dz * u, y, seed) >= y) { b = u; break; }
    a = u;
    if (u >= len) return false;
  }
  for (let k = 0; k < 6; k++) {
    const m = 0.5 * (a + b);
    if (solidTop(x + dx * m, z + dz * m, y, seed) >= y) b = m; else a = m;
  }
  SPOT[0] = x + dx * b; SPOT[1] = z + dz * b;
  return true;
}

const FLAT = 30;     // m: the radius a flat or hollow wall is held as
const SIDE = 0.25;   // m either side of the hold its level is followed to frame it…
const RISE = 0.4;    // … and above and below

/**
 * Frame the wall at its contact (px, pz) with the level y, found going in against (nx, nz): its outward normal
 * from the contacts either side, its curvature (how far the hold stands out of their chord), how far it leans
 * in going up and down; into s (wallNX/NZ/Inv; the rig's cylinder climbX/Z/R and its leans).
 */
function frameWall(s: HeroState, px: number, pz: number, y: number, nx: number, nz: number, seed: number): void {
  const rx = -nz, rz = nx; // (the hero's right, facing the wall)
  let ax = px, az = pz, bx = px, bz = pz, sides = 0;
  if (touchAt(px - rx * SIDE + nx * 0.3, pz - rz * SIDE + nz * 0.3, y, -nx, -nz, 0.9, seed)) { ax = SPOT[0]; az = SPOT[1]; sides++; }
  if (touchAt(px + rx * SIDE + nx * 0.3, pz + rz * SIDE + nz * 0.3, y, -nx, -nz, 0.9, seed)) { bx = SPOT[0]; bz = SPOT[1]; sides++; }
  const tx = bx - ax, tz = bz - az, tl = Math.sqrt(tx * tx + tz * tz);
  if (tl > 1e-3) { nx = tz / tl; nz = -tx / tl; }
  if (s.wall && s.wallFX === s.wallFX) {
    // (on the same wall: halfway from the normal it had, so a bend comes round over a few reframings, not at once)
    const mx = 0.5 * (nx + s.wallNX), mz = 0.5 * (nz + s.wallNZ), ml = Math.sqrt(mx * mx + mz * mz);
    if (ml > 1e-3) { nx = mx / ml; nz = mz / ml; }
  }
  let R = FLAT;
  if (sides === 2) {
    const sag = (px - 0.5 * (ax + bx)) * nx + (pz - 0.5 * (az + bz)) * nz, c = 0.5 * tl;
    if (sag > 1e-3) R = Math.max(0.25, Math.min(FLAT, (c * c + sag * sag) / (2 * sag)));
  }
  // (the level above and below: where the wall stands there, seen from out in front)
  let ux = 0, uz = 0, dx = 0, dz = 0;
  if (touchAt(px + nx * 0.4, pz + nz * 0.4, y + RISE, -nx, -nz, 1.6, seed)) { ux = (SPOT[0] - px) / RISE; uz = (SPOT[1] - pz) / RISE; }
  if (touchAt(px + nx * 1.2, pz + nz * 1.2, y - RISE, -nx, -nz, 1.6, seed)) { dx = (px - SPOT[0]) / RISE; dz = (pz - SPOT[1]) / RISE; }
  s.wallNX = nx; s.wallNZ = nz;
  s.wallInv = Math.max(0, -(ux * nx + uz * nz));
  s.climbR = R; s.climbX = px - nx * R; s.climbZ = pz - nz * R;
  s.climbTaper = 0; s.climbLeanX = ux; s.climbLeanZ = uz;
  s.climbTaperDown = 0; s.climbLeanDownX = dx; s.climbLeanDownZ = dz;
}

const REFRAME = 0.15; // m the hold moves before the wall round it is framed again (between, only the contact is found)

/**
 * Put the root on the wall where the level `feet + HIPS` meets it, looking from (x, z) along (dx, dz) within
 * len (framing the wall there afresh when `frame`, or once the hold has moved REFRAME); false: no wall there.
 */
function holdOnWall(s: HeroState, x: number, z: number, dx: number, dz: number, len: number, feet: number, seed: number, frame = true): boolean {
  const y = feet + HIPS;
  if (!touchAt(x, z, y, dx, dz, len, seed)) return false;
  const px = SPOT[0], pz = SPOT[1];
  const mx = px - s.wallFX, my = y - s.wallFY, mz = pz - s.wallFZ;
  if (frame || !(mx * mx + my * my + mz * mz < REFRAME * REFRAME)) {
    frameWall(s, px, pz, y, -dx, -dz, seed);
    s.wallFX = px; s.wallFY = y; s.wallFZ = pz;
  } else {
    // (the same wall: its frame kept, its axis moved with the hold)
    s.climbX = px - s.wallNX * s.climbR; s.climbZ = pz - s.wallNZ * s.climbR;
  }
  s.x = px; s.z = pz;
  s.y = feet + PLAYER.groundBias;
  s.yaw = wrapAngle(Math.atan2(-s.wallNX, -s.wallNZ));
  return true;
}

/**
 * A hand or foot hold the rig put on the surface the hero climbs (world point g[at..at+2]), moved onto the
 * surface as it really is at that height — the rig takes it as a cylinder round the hips, which a rock's
 * knobbly side or a trunk's bend is not, far from them — `off` out from it. (Only on the hero's own climb:
 * another player's, played back, has no surface but the rig's.)
 */
export function snapToClimbed(s: HeroState, g: Float64Array, at: number, off: number): void {
  const t = s.tree;
  if (!t && !s.wall) return;
  const x = g[at], y = g[at + 1], z = g[at + 2];
  if (t) {
    trunkSection(t, y - t.y, SEC);
    const dx = x - SEC[0], dz = z - SEC[1], d = Math.sqrt(dx * dx + dz * dz);
    if (d < 1e-6) return;
    g[at] = SEC[0] + (dx / d) * (SEC[2] + off);
    g[at + 2] = SEC[1] + (dz / d) * (SEC[2] + off);
    return;
  }
  // (out from the rig's axis there: a wall's own normal, near enough)
  let nx = x - s.climbX, nz = z - s.climbZ;
  const nl = Math.sqrt(nx * nx + nz * nz);
  if (nl < 1e-6) return;
  nx /= nl; nz /= nl;
  if (!touchAt(x + nx * 0.45, z + nz * 0.45, y, -nx, -nz, 0.9, rocksSeed)) return;
  g[at] = SPOT[0] + nx * off;
  g[at + 2] = SPOT[1] + nz * off;
}

/** Above the hold, within the hands' reach, is there more wall? (No: the top is in reach.) */
function wallAbove(s: HeroState, feet: number, seed: number): boolean {
  const nx = s.wallNX, nz = s.wallNZ;
  // (one look where the wall leaning as it does would stand at the hands, a little in: solid there, it goes on)
  const d = (REACH_UP - HIPS) * s.wallInv + 0.15;
  if (solidTop(s.x - nx * d, s.z - nz * d, feet + REACH_UP, seed) >= feet + REACH_UP) return true;
  return touchAt(s.x + nx * 0.3, s.z + nz * 0.3, feet + REACH_UP, -nx, -nz, 0.3 + (REACH_UP - HIPS) * s.wallInv + 0.5, seed);
}

/** The ledge over the wall the hero holds: where to stand on its top (into SPOT) and its height (−∞: none to stand on). */
function ledgeOver(s: HeroState, feet: number, seed: number): number {
  const nx = s.wallNX, nz = s.wallNZ, y = feet + REACH_UP;
  let best = -Infinity;
  for (let d = 0.2; d <= 0.81; d += 0.15) {
    const x = s.x - nx * (d + (REACH_UP - HIPS) * s.wallInv), z = s.z - nz * (d + (REACH_UP - HIPS) * s.wallInv);
    const top = solidTop(x, z, y, seed);
    if (top > best && top <= y + 0.05) { best = top; SPOT[0] = x; SPOT[1] = z; if (d >= 0.35) break; }
  }
  return best;
}

/**
 * The body pressing against a wall grabs it: on the ground pushing the stick into it (on ground too steep to
 * stand on, or against a rock's face), in the air (or under the glider) the stick or its motion (vx, vz,
 * before the collision took it) into it, no faster than latchSpeed, sliding down a little as the grip takes.
 * A wall whose top is within the hands' reach is vaulted onto at once (a ledge, a low rock); a step's height
 * is walked up.
 */
function grabWall(s: HeroState, wx: number, wz: number, mag: number, ok: boolean, air: boolean, vx: number, vz: number, seed: number): boolean {
  if (!ok) return false;
  let dx = wx, dz = wz;
  const hv = Math.sqrt(vx * vx + vz * vz);
  if (mag === 0) {
    if (!air || hv < 0.5) return false;
    dx = vx / hv; dz = vz / hv;
  }
  const feet = s.y - PLAYER.groundBias, y = feet + HIPS;
  // (how far ahead the wall can meet the hips: on steep ground, where the slope rises to them)
  const m = Math.sqrt(s.gx * s.gx + s.gz * s.gz);
  const len = HERO_RADIUS + 0.25 + (!air && m > HERO.walkSlope ? HIPS / m : 0);
  rocksFor(s.x, s.z, seed);
  // (one look ahead first: most of the time there is no wall there)
  if (solidTop(s.x + dx * len, s.z + dz * len, y, seed) < y) return !air && mag > 0 && vaultLedge(s, dx, dz, feet, len, seed);
  const x0 = s.x, y0 = s.y, z0 = s.z, yaw0 = s.yaw;
  if (!holdOnWall(s, x0, z0, dx, dz, len, feet, seed) || s.wallInv > 1 / HERO.walkSlope) {
    s.x = x0; s.y = y0; s.z = z0; s.yaw = yaw0;
    return false;
  }
  const nx = s.wallNX, nz = s.wallNZ;
  const push = mag > 0 ? -(wx * nx + wz * nz) : 0, closing = -(vx * nx + vz * nz);
  const into = air ? push > CLIMB.grab || closing > 0.5 : push > CLIMB.grab;
  const speed = Math.sqrt(hv * hv + s.vy * s.vy);
  let low = false;
  if (into && !(air && speed > CLIMB.latchSpeed)) {
    // a ledge in reach: vault onto it (a step's height: just walked up)
    low = !wallAbove(s, feet, seed);
    if (low) {
      const top = ledgeOver(s, feet, seed);
      if (top - feet < 0.15 || air) low = false;
      else {
        startVault(s, SPOT[0], SPOT[1], top);
        return true;
      }
    }
  }
  if (!into || low || (air && speed > CLIMB.latchSpeed)) {
    s.x = x0; s.y = y0; s.z = z0; s.yaw = yaw0;
    return false;
  }
  if (air) {
    const [lo, hi] = CLIMB.latchSlip;
    s.grip = CLIMB.latchTime;
    s.slip = Math.min(hi, Math.max(lo, (s.vy < 0 ? s.vy * s.vy : 0) / (2 * CLIMB.latchDecel)));
  } else {
    s.grip = 0;
  }
  s.surge = 0;
  s.move = 'climb';
  s.surfing = false;
  s.tree = null;
  s.wall = true;
  s.mantle = 0;
  s.vx = 0; s.vy = 0; s.vz = 0;
  return true;
}

const REACH_UP = 1.6; // m over the feet: the hands' reach (over that the top is in reach: vault onto it)
const KNEE = 0.4;     // m over the feet: a ledge lower than the hips is met here…
const VAULT_MIN = 0.2; // … and vaulted onto from this high (lower: a step, walked up)

/**
 * On the ground, pushing the stick into a ledge lower than the hips (a low rock, a bank): no wall at the
 * hips to climb, so vault straight onto its top if it is at least VAULT_MIN high and level enough to stand on.
 */
function vaultLedge(s: HeroState, dx: number, dz: number, feet: number, len: number, seed: number): boolean {
  const y = feet + KNEE;
  if (solidTop(s.x + dx * len, s.z + dz * len, y, seed) < y) return false;
  if (!touchAt(s.x, s.z, y, dx, dz, len, seed)) return false;
  const cx = SPOT[0], cz = SPOT[1];
  // its top: a little way in from its face, and level there
  const ix = cx + dx * 0.35, iz = cz + dz * 0.35;
  const top = solidTop(ix, iz, feet + REACH_UP, seed);
  if (top - feet < VAULT_MIN || top > feet + HIPS) return false;
  const e = 0.2, gx = (solidTop(ix + e, iz, top + 0.5, seed) - solidTop(ix - e, iz, top + 0.5, seed)) / (2 * e);
  const gz = (solidTop(ix, iz + e, top + 0.5, seed) - solidTop(ix, iz - e, top + 0.5, seed)) / (2 * e);
  if (!(Math.sqrt(gx * gx + gz * gz) <= HERO.walkSlope)) return false;
  s.yaw = wrapAngle(Math.atan2(dx, dz));
  s.tree = null; s.wall = false; s.surfing = false;
  s.climbR = FLAT;
  startVault(s, ix, iz, top);
  return true;
}

/** Start the vault onto a top at (x, z), height top, from where the hero holds on. */
function startVault(s: HeroState, x: number, z: number, top: number): void {
  s.move = 'climb';
  s.mx = s.x; s.my = s.y; s.mz = s.z;
  s.climbX = x; s.climbZ = z; s.climbTop = top;
  s.climbTaper = 0; s.climbLeanX = 0; s.climbLeanZ = 0; s.climbTaperDown = 0; s.climbLeanDownX = 0; s.climbLeanDownZ = 0;
  if (!(s.climbR > 0)) s.climbR = FLAT;
  s.mantle = 1e-6;
  s.surge = 0; s.grip = 0;
  s.vx = 0; s.vz = 0; s.vy = 0;
}

/**
 * One tick on a climbable surface (a trunk, a wall); returns the stamina spent. The stick works on the hero,
 * not the camera (whichever way the camera looks as the hero goes round a trunk or along a wall, forward is
 * up): forward and back climb up and down the surface, the sides go across it (round a trunk). A jump surges
 * the way the stick points along the surface (up when it rests), or with the stick pulled well back leaps off
 * — the way the stick points from the camera, if that is away from the surface, else straight out from it;
 * the sneak key or a dry wheel slides down (a jump then pushes off). The top in reach, the hero vaults onto
 * it (a cubic Bézier over mantleTime) and stands there; the ground reached, it steps off; a wall easing into
 * walkable ground, it walks onto it.
 */
function climbSurface(
  s: HeroState, input: HeroInput, wx: number, wz: number, mag: number, jumpPressed: boolean, fresh: boolean,
  dt: number, seed: number,
): number {
  const T = CLIMB;
  const t = s.tree;
  let spend = 0;
  if (s.mantle > 0) {
    // The vault: hands on the top, the chest over them, a foot up, standing.
    s.mantle = Math.min(1, s.mantle + dt / T.mantleTime);
    const u = s.mantle * s.mantle * (3 - 2 * s.mantle), v = 1 - u;
    const ex = s.climbX, ey = s.climbTop + PLAYER.groundBias, ez = s.climbZ;
    let ox = s.mx - ex, oz = s.mz - ez;
    const ol = Math.sqrt(ox * ox + oz * oz);
    if (ol > 1e-6) { ox /= ol; oz /= ol; }
    const b0 = v * v * v, b1 = 3 * v * v * u, b2 = 3 * v * u * u, b3 = u * u * u;
    const x = b0 * s.mx + b1 * s.mx + b2 * (ex + 0.25 * ox) + b3 * ex;
    const y = b0 * s.my + b1 * (s.my + 0.75) + b2 * (ey + 0.55) + b3 * ey;
    const z = b0 * s.mz + b1 * s.mz + b2 * (ez + 0.25 * oz) + b3 * ez;
    s.vx = (x - s.x) / dt; s.vy = (y - s.y) / dt; s.vz = (z - s.z) / dt;
    s.x = x; s.y = y; s.z = z;
    if (s.mantle >= 1) {
      leaveClimb(s);
      s.move = 'ground';
      s.vx = 0; s.vy = 0; s.vz = 0;
      sampleGround(s, seed);
      s.y = Math.max(s.y, s.ground + PLAYER.groundBias);
    }
    return HERO.staminaClimb * dt;
  }
  if (!t) rocksFor(s.x, s.z, seed);

  // Out of the surface, at the hold.
  let nx: number, nz: number;
  if (t) {
    const ang = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
    nx = Math.sin(ang); nz = Math.cos(ang);
  } else {
    nx = s.wallNX; nz = s.wallNZ;
  }
  // The stick (unit length at most): forward is up the surface, right is across it to the hero's right.
  const sl = Math.max(1, Math.sqrt(input.forward * input.forward + input.strafe * input.strafe));
  const toward = input.forward / sl, side = input.strafe / sl;
  let along = 0, around = 0;                     // m/s up the surface, across it to the right
  const gripped = s.grip > 0;
  if (s.grip > 0) {
    // A mid-air grab: slide down the surface, easing out.
    const g0 = 1 - s.grip / T.latchTime;
    s.grip = Math.max(0, s.grip - dt);
    along = -s.slip * (easeOutCubic(1 - s.grip / T.latchTime) - easeOutCubic(g0)) / dt;
    spend += HERO.staminaHang * dt;
  } else if (input.sneak || !fresh) {
    s.sliding = true;
    if (jumpPressed) {
      outOfWall(s);
      leaveClimb(s);
      s.move = 'air';
      s.vx = nx * 4.5; s.vz = nz * 4.5; s.vy = 3.2;
      return 0;
    }
    along = -T.slideSpeed;
  } else {
    if (jumpPressed && s.surge <= 0) {
      spend += HERO.staminaClimbJump;
      if (-toward > T.leapAway) {
        outOfWall(s);
        leaveClimb(s);
        s.move = 'air';
        const away = mag > 0 && wx * nx + wz * nz > T.leapAway;
        s.vx = (away ? wx : nx) * T.leapSpeed; s.vz = (away ? wz : nz) * T.leapSpeed; s.vy = T.leapUp;
        return spend;
      }
      s.surge = HERO.climbJump;
    }
    if (s.surge > 0) {
      // along the surface the way the stick points (up when it rests or pulls away)
      const up = Math.max(0, toward), l = Math.sqrt(up * up + side * side);
      along = HERO.climbJumpSpeed * (l > 0.2 ? up / l : 1);
      around = HERO.climbJumpSpeed * (l > 0.2 ? side / l : 0);
      s.surge -= HERO.climbJumpSpeed * dt;
    } else {
      along = toward * (toward >= 0 ? T.upSpeed : T.downSpeed);
      around = side * T.wrapSpeed;
    }
    spend += (along !== 0 || around !== 0 ? HERO.staminaClimb : HERO.staminaHang) * dt;
  }
  // The body gathers itself into a move and settles out of it: the speed eases toward the stick's (a grab's
  // slip and a climb jump's surge keep their own curves, and the speed carries on from them).
  if (s.grip > 0 || s.surge > 0) {
    s.climbVA = along; s.climbVS = around;
  } else if (gripped) {
    s.climbVA = 0; s.climbVS = 0; // (a grab's slip ends with the grip taken)
  } else {
    const k = approach(T.accel, dt);
    s.climbVA += (along - s.climbVA) * k;
    s.climbVS += (around - s.climbVS) * k;
    if (along === 0 && Math.abs(s.climbVA) < 0.01) s.climbVA = 0;
    if (around === 0 && Math.abs(s.climbVS) < 0.01) s.climbVS = 0;
    along = s.climbVA; around = s.climbVS;
  }

  // Along the surface to height: a metre up a wall leaning in rises 1 / √(1 + lean²).
  const feet0 = s.y - PLAYER.groundBias;
  const feet = feet0 + (along * dt) / (t ? 1 : Math.sqrt(1 + s.wallInv * s.wallInv));
  // The top in reach, climbing up: vault onto it.
  if (along > 0) {
    if (t && feet + REACH_UP >= s.climbTop) {
      const f = Math.min(feet, s.climbTop - REACH_UP);
      holdOnTrunk(s, t, Math.atan2(nx, nz), f);
      trunkSection(t, t.top, SEC);
      startVault(s, SEC[0], SEC[1], s.climbTop);
      s.climbR = SEC[2];
      return spend;
    }
    if (!t && !wallAbove(s, feet0, seed)) {
      const top = ledgeOver(s, feet0, seed);
      if (top > -Infinity) {
        startVault(s, SPOT[0], SPOT[1], top);
        return spend;
      }
    }
  }
  const x0 = s.x, z0 = s.z;
  if (t) {
    const w = Math.max(-T.wrapMax, Math.min(T.wrapMax, around / (s.climbR + climbClearance(s.climbR))));
    holdOnTrunk(s, t, Math.atan2(nx, nz) - w * dt, feet); // (to the right is clockwise from above: the angle falls)
  } else {
    // across the wall along its level, the hips going at the wrap speed (the hold, on a convex wall, less)
    const k = (s.climbR / (s.climbR + climbClearance(s.climbR))) * around * dt;
    const qx = s.x - nz * k + nx * 0.3, qz = s.z + nx * k + nz * 0.3;
    const reach = 0.9 + Math.abs(along * dt) * 2;
    if (!holdOnWall(s, qx, qz, -nx, -nz, reach, feet, seed, false)
      && !holdOnWall(s, x0 + nx * 0.3, z0 + nz * 0.3, -nx, -nz, reach, feet, seed, true)) {
      // (the wall gone from under the hands: let go)
      outOfWall(s);
      leaveClimb(s);
      s.move = 'air';
      s.vx = nx * 1.5; s.vz = nz * 1.5; s.vy = 0;
      return spend;
    }
  }
  s.vx = (s.x - x0) / dt; s.vz = (s.z - z0) / dt; s.vy = along;
  if (!t && s.wallInv > 1 / HERO.climbExitSlope) {
    // The wall eased into walkable ground: step onto it, the feet where they were.
    const wnx = s.wallNX, wnz = s.wallNZ, out = HIPS * s.wallInv + 0.3;
    const on = touchAt(s.x + wnx * out, s.z + wnz * out, feet + 0.05, -wnx, -wnz, out + 0.3, seed);
    const fx = on ? SPOT[0] : s.x, fz = on ? SPOT[1] : s.z;
    leaveClimb(s);
    s.move = 'ground';
    s.x = fx; s.z = fz;
    s.vx = Math.sin(s.yaw) * 1.5; s.vz = Math.cos(s.yaw) * 1.5; s.vy = 0;
    sampleGround(s, seed);
    s.y = s.ground + PLAYER.groundBias;
    return spend;
  }
  // The ground under the hero: under its feet, out in front of the surface (on a wall looked at every few
  // ticks going up: it is only met going down).
  if (!t && along > 0 && (s.wallTick++ % 6) !== 0) return spend;
  const ox = s.x, oz = s.z;
  let gx: number, gz: number;
  if (t) {
    gx = s.x; gz = s.z;
  } else {
    const out = HIPS * Math.max(0, -(s.climbLeanDownX * s.wallNX + s.climbLeanDownZ * s.wallNZ)) + HERO_RADIUS + 0.05;
    gx = s.x + s.wallNX * out; gz = s.z + s.wallNZ * out;
  }
  s.x = gx; s.z = gz;
  sampleSurface(s, seed);
  if (!t) s.ground = Math.max(s.ground, solidTop(gx, gz, feet + 0.5, seed));
  s.x = ox; s.z = oz;
  if (deepWater(s)) {
    leaveClimb(s);
    s.move = 'swim';
    s.x = gx; s.z = gz;
    s.vy = 0;
  } else if (feet <= s.ground && along <= 0) {
    // Down: step off onto the ground.
    leaveClimb(s);
    s.move = 'ground';
    if (t) {
      s.x = SEC[0] + (SEC[2] + HERO_RADIUS + 0.01) * nx;
      s.z = SEC[1] + (SEC[2] + HERO_RADIUS + 0.01) * nz;
    } else {
      s.x = gx; s.z = gz;
    }
    s.vx = 0; s.vy = 0; s.vz = 0;
    sampleGround(s, seed);
    s.y = s.ground + PLAYER.groundBias;
  } else if (t && feet < s.ground) {
    s.y = s.ground + PLAYER.groundBias; // (going round the bark's foot up a slope: the feet stay on the ground)
  }
  return spend;
}

/**
 * Throw the body into the air with a velocity (a side hop, a backflip, a
 * blow's knockback). Climbing or swimming bodies only take the push.
 */
export function launchHero(s: HeroState, vx: number, vy: number, vz: number): void {
  s.vx = vx; s.vz = vz;
  if (s.move === 'climb' || s.move === 'swim') return;
  s.vy = vy;
  s.move = 'air';
}

/** Step onto the shield (on the ground, or in the air: it lands surfing). */
export function startSurf(s: HeroState): void {
  if (s.move !== 'ground' && s.move !== 'air') return;
  s.surfing = true;
  if (s.move === 'ground') {
    s.move = 'surf';
    // (a slow start pushes off the way the hero faces)
    const v = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
    if (v < COMBAT.surfPush) { s.vx = Math.sin(s.yaw) * COMBAT.surfPush; s.vz = Math.cos(s.yaw) * COMBAT.surfPush; }
  }
}

/** Step off the shield. */
export function stopSurf(s: HeroState): void {
  s.surfing = false;
  if (s.move === 'surf') s.move = 'ground';
}

/**
 * Put the hero at a root position with a velocity (spawn, view switch, drop
 * from fly mode) and pick the state the terrain allows there.
 */
export function placeHero(
  s: HeroState, x: number, y: number, z: number,
  vx: number, vy: number, vz: number, yaw: number, seed: number,
): void {
  s.x = x; s.y = y; s.z = z; s.vx = vx; s.vy = vy; s.vz = vz;
  s.yaw = wrapAngle(yaw);
  s.surge = 0;
  s.surfing = false;
  leaveClimb(s);
  s.skydive = false; s.dive = 0; s.flare = 0; s.recover = 0; s.rolling = false;
  s.jumpHeld = true; // a Space still held from the previous mode must not fire
  sampleGround(s, seed); // islands and ruin ledges count from this height
  const floor = s.ground + PLAYER.groundBias;
  if (deepWater(s) && y < s.water) {
    s.move = 'swim';
    s.y = Math.max(y, s.water - HERO.swimFloat - 0.3);
  } else if (y <= floor + 0.05) {
    s.move = 'ground';
    s.y = floor;
    s.vy = 0;
  } else {
    s.move = 'air';
    s.y = y;
  }
}

export function createHeroState(x: number, z: number, yaw: number, seed: number): HeroState {
  const s: HeroState = {
    x, y: 0, z, vx: 0, vy: 0, vz: 0, yaw: 0, move: 'ground',
    stamina: 1, exhausted: false, rest: 0,
    sprinting: false, sneaking: false, sliding: false, surfing: false, jumpHeld: false,
    surge: 0, ground: 0, gx: 0, gz: 0, water: LANDS.sea, flowX: 0, flowZ: 0, impact: 0, landings: 0, spring: 0,
    skydive: false, dive: 0, flare: 0, recover: 0, rolling: false,
    tree: null, wall: false, wallNX: 0, wallNZ: 1, wallInv: 0, wallFX: NaN, wallFY: NaN, wallFZ: NaN, wallTick: 0, climbVA: 0, climbVS: 0, touch: null, bump: 0, perch: null, climbX: 0, climbZ: 0, climbR: 0, climbTaper: 0, climbLeanX: 0, climbLeanZ: 0, climbTaperDown: 0, climbLeanDownX: 0, climbLeanDownZ: 0, climbTop: 0, grip: 0, slip: 0, mantle: 0, mx: 0, my: 0, mz: 0,
  };
  placeHero(s, x, -Infinity, z, 0, 0, 0, yaw, seed);
  s.jumpHeld = false;
  return s;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * One tick of a skydive's velocity. The sprint key tucks into the nosedive (dive → 1); letting it go, or
 * pulling the stick back, from faster than flareFrom flares. Vertically, quadratic drag k = g / Vt² toward
 * the posture's terminal speed (integrated semi-implicitly: steady at Vt exactly, stable at any dt); across
 * the ground the stick's way at up to the posture's speed, plus the spread body's forward glide.
 */
function skydive(s: HeroState, input: HeroInput, wx: number, wz: number, mag: number, dt: number): void {
  const K = SKYDIVE;
  const pull = input.forward < -0.5;
  const diving = input.sprint && !pull;
  if (!diving && s.dive > 0.5 && s.vy < -K.flareFrom && s.flare === 0) s.flare = K.flareTime;
  if (s.flare > 0) {
    s.flare = Math.max(0, s.flare - dt);
    if (s.vy > -K.flareUntil) s.flare = 0;
  }
  s.dive += ((diving ? 1 : 0) - s.dive) * approach(K.diveRate, dt);
  s.sprinting = diving;
  const a = s.dive;
  const vt = s.flare > 0 ? K.flareTerminal : lerp(K.glideTerminal, K.diveTerminal, a * a);
  const vy0 = s.vy, g = PLAYER.gravity;
  s.vy = (vy0 - g * dt) / (1 + (g / (vt * vt)) * Math.abs(vy0) * dt);
  if (s.flare > 0) s.vy = Math.min(s.vy, vy0 + K.flareDecel * dt);

  const vmax = lerp(K.glideMaxXZ, K.diveMaxXZ, a), acc = lerp(K.glideAccelXZ, K.diveAccelXZ, a) * dt;
  // (the torso's pitch from vertical: near level spread, tipping down to glide; along the fall in the dive)
  const pitch = lerp(82 - 12 * mag, 5, a) * Math.PI / 180;
  const lift = s.vy < -K.liftMinFall ? K.liftK * -s.vy * Math.sin(2 * pitch) * (1 - a) : 0;
  const tx = wx * vmax * mag + Math.sin(s.yaw) * lift, tz = wz * vmax * mag + Math.cos(s.yaw) * lift;
  const dx = tx - s.vx, dz = tz - s.vz, d = Math.sqrt(dx * dx + dz * dz);
  if (d <= acc) { s.vx = tx; s.vz = tz; } else { s.vx += dx / d * acc; s.vz += dz / d * acc; }
  if (mag > 0) turnToward(s, Math.atan2(wx, wz), lerp(K.glideTurn, K.diveTurn, a), dt);
}

/** One tick of hero locomotion (any dt; the engine caps it at 50 ms). */
export function updateHero(s: HeroState, input: HeroInput, dt: number, seed: number): void {
  // Camera-relative wish direction, unit length (diagonals are not faster).
  let wx = 0, wz = 0, mag = 0;
  if (input.forward !== 0 || input.strafe !== 0) {
    const sn = Math.sin(input.camYaw), cs = Math.cos(input.camYaw);
    wx = sn * input.forward + cs * input.strafe;
    wz = cs * input.forward - sn * input.strafe;
    const l = Math.sqrt(wx * wx + wz * wz);
    mag = Math.min(1, l);
    wx /= l; wz /= l;
  }
  const jumpPressed = input.jump && !s.jumpHeld;
  s.jumpHeld = input.jump;
  const fresh = !s.exhausted && s.stamina > 0; // stamina actions allowed
  const climbs = fresh && !input.noClimb;
  const face = input.face ?? NaN;
  const scale = input.speedScale ?? 1;
  let spend = 0;
  s.sprinting = false;
  s.sliding = false;
  s.sneaking = false;
  if (s.move === 'glide' || s.move === 'climb' || s.move === 'swim') s.surfing = false;

  switch (s.move) {
    case 'ground': {
      if (grabTrunk(s, wx, wz, mag, climbs, false, seed)) break;
      if (grabWall(s, wx, wz, mag, climbs, false, 0, 0, seed)) break;
      const recovering = s.recover > 0;
      if (recovering) {
        // A hard landing: rolling on at rollSpeed the way it faces, or squatting, braking to a stop.
        s.recover = Math.max(0, s.recover - dt);
        if (s.rolling) {
          s.vx = Math.sin(s.yaw) * SKYDIVE.rollSpeed;
          s.vz = Math.cos(s.yaw) * SKYDIVE.rollSpeed;
        } else {
          const v = Math.sqrt(s.vx * s.vx + s.vz * s.vz), k = v > 1e-6 ? Math.max(0, v - SKYDIVE.squatBrake * dt) / v : 0;
          s.vx *= k; s.vz *= k;
        }
        if (s.recover === 0) s.rolling = false;
      } else {
        s.sneaking = input.sneak;
        let speed: number = s.exhausted ? HERO.exhaustedSpeed : s.sneaking ? HERO.sneakSpeed : HERO.runSpeed;
        if (input.sprint && fresh && !s.sneaking && mag > 0) {
          speed = HERO.sprintSpeed;
          s.sprinting = true;
          spend += HERO.staminaSprint * dt;
        }
        if (s.ground < s.water - 0.3) speed *= HERO.wadeFactor;
        speed *= scale;
        const k = approach(HERO.accel, dt);
        s.vx += (wx * speed * mag - s.vx) * k;
        s.vz += (wz * speed * mag - s.vz) * k;
      }
      const m = Math.sqrt(s.gx * s.gx + s.gz * s.gz);
      if (m > HERO.walkSlope) {
        // Too steep to stand on: slide down the fall line (∇E points uphill).
        s.sliding = true;
        const f = HERO.slideAccel * dt * Math.min(1, (m - HERO.walkSlope) * 2) / m;
        s.vx -= s.gx * f;
        s.vz -= s.gz * f;
      }
      if (recovering) { /* (facing the roll) */ }
      else if (face === face) turnToward(s, face, HERO.turnRate, dt);
      else if (mag > 0) turnToward(s, Math.atan2(wx, wz), HERO.turnRate, dt);
      const prevY = s.y;
      s.x += s.vx * dt;
      s.z += s.vz * dt;
      if (s.perch && !jumpPressed && perchEdge(s, input.sneak && climbs)) break;
      sampleGround(s, seed);
      const floor = s.ground + PLAYER.groundBias;
      if (jumpPressed && !recovering) {
        s.vy = HERO.jumpVelocity;
        s.y = Math.max(prevY, floor);
        s.move = 'air';
      } else if (deepWater(s)) {
        s.move = 'swim';
        s.vy = 0;
      } else if (prevY - floor > 0.05 + Math.sqrt(s.vx * s.vx + s.vz * s.vz) * dt * HERO.fallSlope) {
        s.move = 'air'; // the ground fell away faster than a slope can be followed
        s.vy = 0;
      } else {
        s.y = floor;
        s.vy = 0;
      }
      break;
    }

    case 'air': {
      // High enough and falling fast enough: spread into a skydive (it lasts until the air is left).
      if (!s.skydive && !s.surfing && s.vy < -SKYDIVE.enterSpeed
        && s.y - PLAYER.groundBias - Math.max(s.ground, s.water) >= SKYDIVE.minAltitude) s.skydive = true;
      if (s.skydive) {
        skydive(s, input, wx, wz, mag, dt);
        if (face === face) turnToward(s, face, HERO.turnRate * 0.5, dt);
      } else {
        if (mag > 0 && !s.surfing) {
          const k = approach(HERO.airAccel, dt);
          s.vx += (wx * HERO.runSpeed * scale * mag - s.vx) * k;
          s.vz += (wz * HERO.runSpeed * scale * mag - s.vz) * k;
          if (face !== face) turnToward(s, Math.atan2(wx, wz), HERO.turnRate * 0.5, dt);
        }
        if (face === face) turnToward(s, face, HERO.turnRate * 0.5, dt);
        // (a jump's top hangs a moment: less gravity while barely moving up or down)
        const g = Math.abs(s.vy) <= HERO.apexBand ? PLAYER.gravity * HERO.apexGravity : PLAYER.gravity;
        s.vy = Math.max(s.vy - g * dt, -HERO.terminalVelocity);
      }
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.z += s.vz * dt;
      const avx = s.vx, avz = s.vz; // (before a collision takes the speed into a wall)
      sampleGround(s, seed);
      if (grabTrunk(s, wx, wz, mag, climbs && !s.surfing, true, seed)) break;
      if (grabWall(s, wx, wz, mag, climbs && !s.surfing, true, avx, avz, seed)) break;
      if (jumpPressed && fresh && s.y - s.ground > HERO.glideMinHeight) {
        s.move = 'glide';
        s.surfing = false;
      } else if (deepWater(s) && s.y < s.water - HERO.swimFloat) {
        s.move = 'swim';
        s.surfing = false;
        s.vy *= 0.3;
      } else if (s.y <= s.ground + PLAYER.groundBias) {
        land(s, wx, wz, mag, climbs, seed);
      }
      break;
    }

    case 'surf': {
      // On the shield: gravity along the slope (a = −g ∇E / (1 + |∇E|²) across the ground), a
      // little friction and drag, and the input turning the way it slides.
      const m2 = s.gx * s.gx + s.gz * s.gz;
      const gs = PLAYER.gravity / (1 + m2);
      s.vx -= s.gx * gs * dt;
      s.vz -= s.gz * gs * dt;
      let v = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
      if (v > 1e-6) {
        const slow = (COMBAT.surfFriction * PLAYER.gravity / Math.sqrt(1 + m2) + COMBAT.surfDrag * v * v) * dt;
        const k = Math.max(0, v - slow) / v;
        s.vx *= k; s.vz *= k;
        v *= k;
      }
      if (mag > 0) {
        if (v > 0.8) {
          // steer: the velocity turns toward the wish, at most surfSteer rad/s
          const turn = wrapAngle(Math.atan2(wx, wz) - Math.atan2(s.vx, s.vz));
          const most = COMBAT.surfSteer * mag * dt;
          const a = Math.max(-most, Math.min(most, turn));
          const c = Math.cos(a), sn = Math.sin(a);
          const vx = s.vx * c + s.vz * sn, vz = -s.vx * sn + s.vz * c;
          s.vx = vx; s.vz = vz;
        }
        // (and a gentle push, so a surfer on the flat can still crawl along)
        s.vx += wx * mag * 1.2 * dt;
        s.vz += wz * mag * 1.2 * dt;
      }
      if (v > COMBAT.surfMaxSpeed) { s.vx *= COMBAT.surfMaxSpeed / v; s.vz *= COMBAT.surfMaxSpeed / v; }
      if (v > 0.3) turnToward(s, Math.atan2(s.vx, s.vz), HERO.turnRate, dt);
      const prevY = s.y;
      s.x += s.vx * dt;
      s.z += s.vz * dt;
      sampleGround(s, seed);
      const floor = s.ground + PLAYER.groundBias;
      if (jumpPressed) {
        s.vy = COMBAT.surfJump;
        s.y = Math.max(prevY, floor);
        s.move = 'air';
      } else if (deepWater(s)) {
        s.move = 'swim';
        s.surfing = false;
        s.vy = 0;
      } else if (prevY - floor > 0.05 + v * dt * HERO.fallSlope) {
        s.move = 'air'; // off a ledge: still on the shield
        s.vy = 0;
      } else {
        s.y = floor;
        s.vy = 0;
        if (v < COMBAT.surfMinSpeed && m2 < 0.04) stopSurf(s);
      }
      break;
    }

    case 'glide': {
      spend += HERO.staminaGlide * dt;
      if (jumpPressed || !fresh) s.move = 'air'; // fold the glider (also when the wheel runs dry)
      const tx = mag > 0 ? wx * HERO.glideSpeed * mag : Math.sin(s.yaw) * HERO.glideDrift;
      const tz = mag > 0 ? wz * HERO.glideSpeed * mag : Math.cos(s.yaw) * HERO.glideDrift;
      const k = approach(HERO.glideAccel, dt);
      s.vx += (tx - s.vx) * k;
      s.vz += (tz - s.vz) * k;
      s.vy += (-HERO.glideSink - s.vy) * approach(3.0, dt);
      if (mag > 0) turnToward(s, Math.atan2(wx, wz), HERO.turnRate * 0.35, dt);
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.z += s.vz * dt;
      const gvx = s.vx, gvz = s.vz;
      sampleGround(s, seed);
      // (gliding into a tree or a wall: fold the glider and grab it)
      if (grabTrunk(s, wx, wz, mag, climbs, true, seed) || grabWall(s, wx, wz, mag, climbs, true, gvx, gvz, seed)) break;
      if (deepWater(s) && s.y < s.water - HERO.swimFloat) {
        s.move = 'swim';
        s.vy *= 0.3;
      } else if (s.y <= s.ground + PLAYER.groundBias) {
        land(s, wx, wz, mag, climbs, seed);
      }
      break;
    }

    case 'climb': {
      if (!s.tree && !s.wall && s.mantle === 0) {
        // (put on a surface without a hold — a view switch, a test: the wall it faces, else the ground)
        const feet = s.y - PLAYER.groundBias;
        rocksFor(s.x, s.z, seed);
        if (!holdOnWall(s, s.x, s.z, Math.sin(s.yaw), Math.cos(s.yaw), 1.5, feet, seed) || s.wallInv > 1 / HERO.climbExitSlope) {
          leaveClimb(s);
          s.move = 'ground';
          s.vx = 0; s.vy = 0; s.vz = 0;
          sampleGround(s, seed);
          s.y = s.ground + PLAYER.groundBias;
          break;
        }
        s.wall = true;
      }
      spend += climbSurface(s, input, wx, wz, mag, jumpPressed, fresh, dt, seed);
      break;
    }

    case 'swim': {
      // (swimming into a steep bank or a rock: climb out of the water up it)
      if (grabWall(s, wx, wz, mag, climbs, false, 0, 0, seed)) break;
      const dash = input.sprint && fresh && mag > 0;
      const speed = dash ? HERO.swimDashSpeed : s.exhausted ? HERO.swimSpeed * 0.6 : HERO.swimSpeed;
      if (dash) {
        s.sprinting = true;
        spend += HERO.staminaSwimDash * dt;
      }
      const k = approach(3.0, dt);
      s.vx += (wx * speed * mag - s.vx) * k;
      s.vz += (wz * speed * mag - s.vz) * k;
      if (mag > 0) turnToward(s, Math.atan2(wx, wz), HERO.turnRate * 0.6, dt);
      // Buoyancy: a damped spring settles the root at the float depth.
      s.vy += ((s.water - HERO.swimFloat - s.y) * 12 - s.vy * 5) * dt;
      // (a river's current carries the swimmer downstream)
      s.x += (s.vx + HERO.currentDrift * s.flowX) * dt;
      s.y += s.vy * dt;
      s.z += (s.vz + HERO.currentDrift * s.flowZ) * dt;
      sampleGround(s, seed);
      if (s.ground > s.water - HERO.swimDepth + 0.1) {
        // Shallow enough to stand: wade out.
        s.move = 'ground';
        s.y = Math.max(s.y, s.ground + PLAYER.groundBias);
        s.vy = 0;
      }
      break;
    }
  }

  if (s.move !== 'air') { s.skydive = false; s.dive = 0; s.flare = 0; }
  if (s.move !== 'ground') { s.recover = 0; s.rolling = false; }

  if (spend > 0) {
    s.stamina -= spend;
    s.rest = 0;
    if (s.stamina <= 0) {
      s.stamina = 0;
      s.exhausted = true;
    }
  } else {
    s.rest += dt;
    // No recovery while hanging on a wall or under the glider.
    if (s.move !== 'climb' && s.move !== 'glide' && s.rest > HERO.staminaRegenDelay) {
      const rate = s.exhausted ? HERO.staminaRegenExhausted
        : s.move === 'swim' ? HERO.staminaRegenSwim : HERO.staminaRegen;
      s.stamina = Math.min(1, s.stamina + rate * dt);
      if (s.stamina >= 1) s.exhausted = false;
    }
  }
}

/** What the hero is doing, for the HUD and the animation rig. */
export function heroAction(s: HeroState): HeroAction {
  const hv = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
  switch (s.move) {
    case 'ground':
      if (s.recover > 0) return s.rolling ? 'roll' : 'squat';
      if (s.sliding) return 'slide';
      if (s.sprinting) return 'sprint';
      if (hv < 0.3) return 'idle';
      if (s.sneaking) return 'sneak';
      if (s.exhausted) return 'tired';
      return hv < 3.0 ? 'walk' : 'run';
    case 'air':
      if (s.skydive) return s.dive > 0.5 ? 'dive' : 'skydive';
      return s.surfing ? 'surf' : s.vy > 0 ? 'jump' : 'fall';
    case 'glide':
      return 'glide';
    case 'climb':
      return hv > 0.05 || s.vy !== 0 ? 'climb' : 'hang';
    case 'swim':
      return s.sprinting ? 'dash' : 'swim';
    case 'surf':
      return 'surf';
  }
}
