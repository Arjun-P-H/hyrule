/**
 * Foes: Bokoblin camps placed by hash, and the Bokoblins' minds.
 * ---------------------------------------------------------------------------
 * One candidate camp per FOE.campCell square of the land, at a hashed spot on
 * level, dry ground clear of the ruins; a fire in its middle and two to four
 * Bokoblins round it (red near the spawn, blue and black farther out), the
 * bigger camps with an archer. A camp a few dozen metres up the rise ahead of
 * the spawn is always there, for a first fight.
 *
 * A Bokoblin is a hero body (hero/locomotion.ts) and rig (hero/rig.ts) driven
 * by its AI through HeroInput and a CombatPose: it idles and wanders round
 * its fire; sees the hero in its view cone (not as far when the hero sneaks)
 * or hears it close, and after a moment of noticing is roused (a jump of
 * surprise, its camp-mates roused with it); it chases, waits its turn at a
 * distance while two others swing, then winds up — the tell — and swings
 * (moves.ts smash / sweep); an archer keeps its distance and shoots. Blows
 * make it flinch, heavy ones knock it down, a parry or a headshot dazes it;
 * beaten, it falls and goes up in smoke. Deterministic: its choices come from
 * its own seeded stream.
 */

import { COMBAT, FOE, HERO, PLAYER, SPAWN } from '../constants';
import { elevation, slope, waterLevel } from '../terrainMath';
import { hash01 } from '../noise';
import { ruinSiteAt } from '../ruins';
import { worldOf } from '../world';
import { createHeroState, updateHero, type HeroInput, type HeroState } from '../hero/locomotion';
import { createCombatPose, createHeroRig, updateHeroRig, J, type CombatPose, type HeroRig } from '../hero/rig';
import { MOVE, MOVES } from './moves';
import { BOW, FOOD, SHIELD, WEAPON, WEAPONS } from './weapons';

export type FoeKind = {
  readonly name: string;
  readonly hp: number;
  readonly damage: number;   // quarters of a heart a blow takes
  readonly weapon: number;   // WEAPONS index it carries
  readonly skin: readonly [number, number, number]; // linear albedo (the hero shader's dress slot)
  readonly scale: number;    // body size × the hero's
  readonly pace: number;     // swing speed ×
  readonly speed: number;    // × FOE.chaseSpeed
  readonly food: readonly [number, number]; // FOODS indices it may drop
  readonly shield: number;   // SHIELDS index it may drop
  readonly bow: number;      // BOWS index an archer of the kind may drop
};

export const FOE_KINDS: readonly FoeKind[] = [
  { name: 'Red Bokoblin', hp: 26, damage: 4, weapon: WEAPON.club, skin: [0.33, 0.055, 0.04], scale: 0.92, pace: 1, speed: 1, food: [FOOD.apple, FOOD.shroom], shield: SHIELD.boko, bow: BOW.boko },
  { name: 'Blue Bokoblin', hp: 60, damage: 6, weapon: WEAPON.spiked, skin: [0.035, 0.085, 0.27], scale: 0.97, pace: 1.12, speed: 1.06, food: [FOOD.shroom, FOOD.meat], shield: SHIELD.wooden, bow: BOW.boko },
  { name: 'Black Bokoblin', hp: 120, damage: 8, weapon: WEAPON.bone, skin: [0.032, 0.026, 0.045], scale: 1.04, pace: 1.24, speed: 1.12, food: [FOOD.meat, FOOD.prime], shield: SHIELD.traveler, bow: BOW.soldier },
];

export type Camp = {
  readonly key: number;   // unique per camp (the cell's, or −1 for the spawn's)
  readonly x: number; readonly y: number; readonly z: number; // the fire
  readonly kinds: readonly number[];   // FOE_KINDS index of each foe
  readonly archer: number;             // which of them shoots (−1: none)
};

/** Deterministic stream for one camp. */
function stream(ix: number, iz: number, seed: number): () => number {
  let k = 0;
  return () => hash01(ix, iz, seed + 21001 + 131 * k++);
}

/** Level, dry, clear ground for a camp round (x, z)? */
function campGround(x: number, z: number, seed: number): boolean {
  const e = elevation(x, z, seed, 0);
  if (e < waterLevel(x, z, seed) + 0.8 || slope(x, z, seed) > 0.28) return false;
  for (let k = 0; k < 6; k++) {
    const a = (k / 6) * Math.PI * 2;
    const px = x + Math.cos(a) * FOE.campRadius, pz = z + Math.sin(a) * FOE.campRadius;
    if (slope(px, pz, seed) > 0.5 || elevation(px, pz, seed, 0) < waterLevel(px, pz, seed) + 0.3) return false;
  }
  return !ruinSiteAt(x, z, seed);
}

/** Tiers of a camp at distance d from the spawn: red near it, blue and black farther out. */
function kindAt(d: number, r: number): number {
  const blue = Math.min(0.55, d / 2500), black = Math.max(0, Math.min(0.35, (d - 1200) / 5000));
  return r < black ? 2 : r < black + blue ? 1 : 0;
}

const cellCache = new Map<number, Camp | null>();
const cellKey = (ix: number, iz: number) => ix * 92821 + iz * 7 + 0.5; // (unique for the cells in play)

/** The camp of cell (ix, iz), or null (cached). */
export function campInCell(ix: number, iz: number, seed: number): Camp | null {
  if (worldOf(seed) === 'clouds') return null; // (the cloud world is peaceful)
  const key = cellKey(ix, iz) + seed * 1e-7;
  const hit = cellCache.get(key);
  if (hit !== undefined) return hit;
  let camp: Camp | null = null;
  if (hash01(ix, iz, seed + 21000) < FOE.campProbability) {
    const r = stream(ix, iz, seed);
    const x = (ix + 0.2 + 0.6 * r()) * FOE.campCell, z = (iz + 0.2 + 0.6 * r()) * FOE.campCell;
    const spawn = spawnCamp(seed);
    if (campGround(x, z, seed) && (!spawn || Math.hypot(x - spawn.x, z - spawn.z) > 140)) {
      const d = Math.hypot(x - SPAWN.x, z - SPAWN.z);
      const n = 2 + Math.floor(r() * 3);
      const kinds: number[] = [];
      for (let k = 0; k < n; k++) kinds.push(kindAt(d, r()));
      camp = { key: cellKey(ix, iz), x, y: elevation(x, z, seed, 0), z, kinds, archer: n >= 3 ? n - 1 : -1 };
    }
  }
  if (cellCache.size > 4096) cellCache.clear();
  cellCache.set(key, camp);
  return camp;
}

let spawnCache: { seed: number; camp: Camp | null } | null = null;

/** The camp ahead of the spawn (a first fight): the first level, dry spot 40–75 m up the view. */
export function spawnCamp(seed: number): Camp | null {
  if (worldOf(seed) === 'clouds') return null;
  if (spawnCache && spawnCache.seed === seed) return spawnCache.camp;
  let camp: Camp | null = null;
  search: for (let d = 44; d <= 76; d += 8) {
    for (const da of [0, 0.25, -0.25, 0.5, -0.5, 0.8, -0.8]) {
      const a = SPAWN.yaw + da;
      const x = SPAWN.x + Math.sin(a) * d, z = SPAWN.z + Math.cos(a) * d;
      if (campGround(x, z, seed)) {
        camp = { key: -1, x, y: elevation(x, z, seed, 0), z, kinds: [0, 0], archer: -1 };
        break search;
      }
    }
  }
  spawnCache = { seed, camp };
  return camp;
}

/** The camps whose fire lies within `range` of (x, z), into `out`. */
export function campsNear(x: number, z: number, range: number, seed: number, out: Camp[]): Camp[] {
  out.length = 0;
  const s = spawnCamp(seed);
  if (s && Math.hypot(s.x - x, s.z - z) <= range) out.push(s);
  const c0x = Math.floor((x - range) / FOE.campCell), c1x = Math.floor((x + range) / FOE.campCell);
  const c0z = Math.floor((z - range) / FOE.campCell), c1z = Math.floor((z + range) / FOE.campCell);
  for (let iz = c0z; iz <= c1z; iz++) {
    for (let ix = c0x; ix <= c1x; ix++) {
      const c = campInCell(ix, iz, seed);
      if (c && Math.hypot(c.x - x, c.z - z) <= range) out.push(c);
    }
  }
  return out;
}

export type FoeMind =
  | 'idle'     // round the fire, now and then wandering to another spot
  | 'alert'    // just noticed the hero: a jump of surprise
  | 'chase'    // after the hero
  | 'wait'     // circling at a distance for its turn to swing
  | 'attack'   // a swing: the wind-up (the tell), the blow, the recovery
  | 'shoot'    // an archer drawing and loosing
  | 'flinch'   // struck
  | 'stun'     // parried or shot in the head: dazed
  | 'down'     // knocked flat (getting up at the end)
  | 'dead';    // fallen; gone in smoke after FOE.corpse s

export type Foe = {
  slot: number;
  live: boolean;
  camp: number;            // its camp's key
  kind: number;            // FOE_KINDS index
  archer: boolean;
  body: HeroState;
  rig: HeroRig;
  pose: CombatPose;
  input: HeroInput;
  hp: number;
  mind: FoeMind;
  t: number;               // s in the present state of mind
  homeX: number; homeZ: number;
  goalX: number; goalZ: number; // where it is wandering to
  aware: number;           // 0..1 how near it is to noticing the hero
  unseen: number;          // s since it last sensed the hero
  roused: boolean;         // it knows the hero is there
  move: number;            // its swing (moves.ts) while attacking
  u: number;               // how far through it
  struck: boolean;         // this swing has landed (or missed past its window)
  circle: number;          // ±1 the way it circles while waiting
  weapon: number;          // WEAPONS index in hand (−1 none)
  flash: number;           // s left of the white flash of a blow taken
  draw: number;            // an archer's draw 0..1
  lastSwing: number;       // the hero's swing that last struck it (one blow a swing)
  rest: number;            // s it waits (circling) before it may swing again
  rng: number;             // its random stream's state
};

/** What a foe's mind asks of the world (combat/world.ts). */
export interface FoeWorld {
  readonly seed: number;
  readonly hero: HeroState;
  readonly heroAlive: boolean;
  /** The hero is sneaking / sprinting (harder / easier to notice). */
  readonly heroSneaking: boolean;
  readonly heroSprinting: boolean;
  /** Foes swinging at the hero now (FOE.attackers at once; a foe starting its swing counts itself in). */
  attackers: number;
  /** The foe's blow lands now: the world resolves a guard, a parry, a dodge or a hit. */
  strike(f: Foe): void;
  /** An archer looses an arrow at the hero. */
  shoot(f: Foe): void;
  /** The foe noticed the hero: its camp-mates are roused too. */
  rouse(f: Foe): void;
}

/** A uniform number in [0, 1) from the foe's own stream (mulberry32). */
function rand(f: Foe): number {
  f.rng = (f.rng + 0x6d2b79f5) | 0;
  let t = f.rng;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function createFoe(slot: number): Foe {
  const body = createHeroState(0, 0, 0, 0);
  return {
    slot, live: false, camp: 0, kind: 0, archer: false,
    body, rig: createHeroRig(), pose: createCombatPose(),
    input: { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0, face: NaN, speedScale: 1, noClimb: true },
    hp: 0, mind: 'idle', t: 0, homeX: 0, homeZ: 0, goalX: 0, goalZ: 0,
    aware: 0, unseen: 0, roused: false, move: -1, u: 0, struck: false, circle: 1,
    weapon: -1, flash: 0, draw: 0, lastSwing: -1, rest: 0, rng: 0,
  };
}

/** Wake foe `f` as member `k` of camp `c`, standing round its fire (`exact`: on the fire's spot). */
export function wakeFoe(f: Foe, c: Camp, k: number, seed: number, exact = false): void {
  const kind = FOE_KINDS[c.kinds[k]];
  const a = (k / c.kinds.length) * Math.PI * 2 + hash01(k, c.key | 0, seed + 21900) * 0.8;
  const r = exact ? 0 : FOE.campRadius * (0.55 + 0.3 * hash01(k, c.key | 0, seed + 21901));
  const x = c.x + Math.cos(a) * r, z = c.z + Math.sin(a) * r;
  // (facing the fire)
  const body = createHeroState(x, z, Math.atan2(c.x - x, c.z - z), seed);
  Object.assign(f.body, body);
  f.live = true;
  f.camp = c.key;
  f.kind = c.kinds[k];
  f.archer = k === c.archer;
  f.hp = kind.hp;
  f.mind = 'idle';
  f.t = hash01(k, c.key | 0, seed + 21902) * 4;
  f.homeX = x; f.homeZ = z; f.goalX = x; f.goalZ = z;
  f.aware = 0; f.unseen = 0; f.roused = false;
  f.move = -1; f.u = 0; f.struck = false; f.circle = k % 2 ? 1 : -1;
  f.weapon = kind.weapon;
  f.flash = 0; f.draw = 0; f.lastSwing = -1; f.rest = 0;
  f.rng = (Math.imul(c.key | 0, 2654435761) ^ (k * 40503) ^ seed) | 0;
  f.rig.snap = true;
  const p = f.pose;
  p.weapon = WEAPONS[f.weapon].model; p.armed = true; p.shield = false;
  p.act = 'none'; p.move = -1; p.u = 0; p.pitch = 0; p.draw = -1; p.glow = 0; p.spin = 0; p.flip = 0; p.roll = 0;
}

/** Joint j of foe f in world space (from its rig, scaled about its root on the ground). */
export function foeJoint(f: Foe, j: number, out: Float64Array, o = 0): void {
  const s = f.body, k = FOE_KINDS[f.kind].scale;
  const sy = Math.sin(s.yaw + f.rig.spin), cy = Math.cos(s.yaw + f.rig.spin);
  const x = f.rig.local[3 * j] * k, y = f.rig.local[3 * j + 1] * k, z = f.rig.local[3 * j + 2] * k;
  out[o] = s.x + cy * x + sy * z;
  out[o + 1] = s.y - PLAYER.groundBias + y;
  out[o + 2] = s.z - sy * x + cy * z;
}

const wrap = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

function setMind(f: Foe, m: FoeMind): void {
  f.mind = m;
  f.t = 0;
}

/** Steer the body toward (x, z) at `speed` (m/s, ≤ the run speed unless `sprint`), facing `face` (NaN: the way it goes). */
function steer(f: Foe, x: number, z: number, speed: number, face: number, sprint = false): void {
  const s = f.body, inp = f.input;
  const dx = x - s.x, dz = z - s.z, d = Math.hypot(dx, dz);
  inp.camYaw = Math.atan2(dx, dz);
  inp.forward = d > 0.25 ? 1 : 0;
  inp.strafe = 0;
  inp.sprint = sprint;
  inp.speedScale = speed / HERO.runSpeed;
  inp.face = face;
}

/** Move sideways round (x, z) at distance `keep`, facing it. */
function circleAround(f: Foe, x: number, z: number, keep: number, speed: number): void {
  const s = f.body, inp = f.input;
  const dx = s.x - x, dz = s.z - z, d = Math.hypot(dx, dz) || 1;
  const toward = Math.atan2(-dx, -dz);
  // radial correction toward the ring, tangential drift round it
  const radial = Math.max(-1, Math.min(1, (d - keep) * 0.8));
  inp.camYaw = toward;
  inp.forward = radial;
  inp.strafe = f.circle * 0.7;
  inp.sprint = false;
  inp.speedScale = speed / HERO.runSpeed;
  inp.face = toward;
}

function halt(f: Foe, face: number): void {
  const inp = f.input;
  inp.forward = 0; inp.strafe = 0; inp.sprint = false; inp.face = face; inp.speedScale = 1;
}

/**
 * One tick of foe f's mind and body. dt: the world's (slowed) time step. The
 * world reads f.move / f.u / f.struck to time blows; the AI calls back to land them.
 */
export function stepFoe(f: Foe, w: FoeWorld, dt: number): void {
  const s = f.body, h = w.hero, kind = FOE_KINDS[f.kind];
  f.t += dt;
  f.flash = Math.max(0, f.flash - dt);
  f.rest = Math.max(0, f.rest - dt);
  const dx = h.x - s.x, dz = h.z - s.z;
  const dist = Math.hypot(dx, dz);
  const toHero = Math.atan2(dx, dz);
  const ahead = Math.cos(wrap(toHero - s.yaw));
  const p = f.pose;
  p.act = 'none'; p.move = -1; p.u = 0; p.draw = -1; p.pitch = 0;
  p.armed = f.weapon >= 0;
  p.weapon = f.weapon >= 0 ? WEAPONS[f.weapon].model : 0;
  const run = HERO.runSpeed * FOE.chaseSpeed * kind.speed;

  // Senses: seen in the cone (not as far when the hero sneaks), heard close by (farther if it sprints).
  if (f.mind !== 'dead' && f.mind !== 'down') {
    const sight = w.heroSneaking ? FOE.sightSneak : FOE.sight;
    const hear = w.heroSprinting ? FOE.hearing * 2 : w.heroSneaking ? FOE.hearing * 0.4 : FOE.hearing;
    const dy = Math.abs(h.y - s.y);
    const senses = w.heroAlive && dy < 25 && (dist < hear || (dist < sight && ahead > Math.cos(FOE.cone)));
    if (senses) {
      f.unseen = 0;
      if (!f.roused) {
        f.aware += dt / (FOE.notice * (w.heroSneaking ? 2.5 : 1)) * (dist < hear ? 2 : 1);
        if (f.aware >= 1) {
          f.aware = 1;
          f.roused = true;
          setMind(f, 'alert');
          w.rouse(f);
        }
      }
    } else {
      f.unseen += dt;
      if (!f.roused) f.aware = Math.max(0, f.aware - dt * 0.25);
      if (f.roused && (f.unseen > FOE.forget || !w.heroAlive)) {
        // lost the hero: back to the fire
        f.roused = false;
        f.aware = 0;
        if (f.mind === 'chase' || f.mind === 'wait' || f.mind === 'shoot') setMind(f, 'idle');
      }
    }
  }

  switch (f.mind) {
    case 'idle': {
      // round the fire, now and then strolling to another spot near home
      if (Math.hypot(f.goalX - s.x, f.goalZ - s.z) > 0.6) steer(f, f.goalX, f.goalZ, 1.4, NaN);
      else halt(f, NaN);
      if (f.t > 3 + 5 * rand(f)) {
        f.t = 0;
        const a = rand(f) * Math.PI * 2, r = rand(f) * FOE.campRadius * 0.5;
        f.goalX = f.homeX + Math.cos(a) * r;
        f.goalZ = f.homeZ + Math.sin(a) * r;
      }
      if (f.roused) setMind(f, f.archer ? 'shoot' : 'chase');
      break;
    }
    case 'alert': {
      halt(f, toHero);
      if (f.t < dt * 1.5 && s.move === 'ground') { s.vy = 4.2; s.move = 'air'; } // (a jump of surprise)
      if (f.t > 0.65) setMind(f, f.archer ? 'shoot' : 'chase');
      break;
    }
    case 'chase': {
      if (!f.roused) { setMind(f, 'idle'); break; }
      if (dist < FOE.strikeRange && w.heroAlive) {
        if (w.attackers < FOE.attackers && f.rest <= 0) {
          setMind(f, 'attack');
          w.attackers++;
          f.move = rand(f) < 0.5 ? MOVE.smash : MOVE.sweep;
          f.u = 0;
          f.struck = false;
        } else setMind(f, 'wait');
        halt(f, toHero);
        break;
      }
      if (dist < FOE.keepAway[1] && (w.attackers >= FOE.attackers || f.rest > 0)) { setMind(f, 'wait'); break; }
      steer(f, h.x, h.z, run, NaN, dist > 14);
      break;
    }
    case 'wait': {
      // circling at a distance, now and then changing its way round, until its turn
      if (!f.roused) { setMind(f, 'idle'); break; }
      circleAround(f, h.x, h.z, FOE.keepAway[0] + 0.5 * (FOE.keepAway[1] - FOE.keepAway[0]), 1.6);
      if (f.t > 1.2 && rand(f) < dt * 0.8) f.circle = -f.circle;
      if (w.attackers < FOE.attackers && f.rest <= 0 && f.t > 0.4) setMind(f, 'chase');
      else if (dist > FOE.keepAway[1] + 2.5) setMind(f, 'chase');
      break;
    }
    case 'attack': {
      const m = MOVES[f.move];
      f.u += dt * kind.pace / m.dur;
      p.act = 'swing'; p.move = f.move; p.u = Math.min(1, f.u);
      // turning to track the hero through the wind-up, committed once the blow starts
      const committed = f.u > m.hit[0] - 0.08;
      const inLunge = f.u >= m.lungeAt[0] && f.u <= m.lungeAt[1];
      if (inLunge) steer(f, s.x + Math.sin(s.yaw) * 5, s.z + Math.cos(s.yaw) * 5, m.lunge, s.yaw);
      else halt(f, committed ? s.yaw : toHero);
      if (!f.struck && f.u >= m.hit[0]) {
        // (the world lands it once the blade is through the hero's space, or calls it a miss)
        w.strike(f);
      }
      if (f.u >= 1) {
        f.rest = FOE.rest[0] + (FOE.rest[1] - FOE.rest[0]) * rand(f);
        setMind(f, f.roused ? 'wait' : 'idle');
      }
      break;
    }
    case 'shoot': {
      if (!f.roused) { setMind(f, 'idle'); break; }
      // keep its distance, turn to the hero, draw, loose, and again
      const [near, far] = FOE.archerRange;
      if (dist < near) steer(f, s.x - dx, s.z - dz, run * 0.8, toHero);
      else if (dist > far) steer(f, h.x, h.z, run, NaN, dist > far + 10);
      else halt(f, toHero);
      p.act = 'aim';
      p.pitch = Math.atan2(h.y - s.y, Math.max(dist, 1)) + 0.04 * dist / 10;
      const cycle = 2.4 / kind.pace;
      const ph = f.t % cycle;
      const drawing = ph > 0.5 && ph < 1.7 && dist <= far + 4 && ahead > 0.85;
      f.draw = drawing ? Math.min(1, f.draw + dt / 0.9) : Math.max(0, f.draw - dt * 3);
      p.draw = f.draw > 0 || drawing ? f.draw : 0;
      if (f.draw >= 1 && ph >= 1.7 - dt) {
        w.shoot(f);
        f.draw = 0;
      }
      break;
    }
    case 'flinch': {
      p.act = 'hurt'; p.u = f.t;
      halt(f, NaN);
      if (f.t > 0.42) setMind(f, f.archer ? 'shoot' : 'chase');
      break;
    }
    case 'stun': {
      p.act = 'stun'; p.u = f.t;
      halt(f, NaN);
      if (f.t > FOE.stun) setMind(f, f.archer ? 'shoot' : 'chase');
      break;
    }
    case 'down': {
      p.act = f.t < FOE.knockdown - 0.45 ? 'down' : 'none';
      halt(f, NaN);
      if (f.t > FOE.knockdown) setMind(f, f.archer ? 'shoot' : 'chase');
      break;
    }
    case 'dead': {
      p.act = 'dead'; p.u = f.t;
      p.armed = false;
      halt(f, NaN);
      break;
    }
  }
  updateHero(s, f.input, dt, w.seed);
  updateHeroRig(f.rig, s, dt, p);
}

/** A blow (or an arrow) takes `damage` from foe f: it flinches, is knocked down or dies. True when it dies of it. */
export function harmFoe(f: Foe, damage: number, fromX: number, fromZ: number, knock: number, heavy: boolean): boolean {
  if (f.mind === 'dead') return false;
  f.hp -= damage;
  f.flash = 0.12;
  f.roused = true;
  f.aware = 1;
  f.unseen = 0;
  const s = f.body;
  const dx = s.x - fromX, dz = s.z - fromZ, d = Math.hypot(dx, dz) || 1;
  if (f.hp <= 0) {
    f.hp = 0;
    setMind(f, 'dead');
    s.vx = (dx / d) * Math.max(knock, 3); s.vz = (dz / d) * Math.max(knock, 3);
    if (s.move === 'ground') { s.vy = 3.5; s.move = 'air'; }
    return true;
  }
  s.vx = (dx / d) * knock; s.vz = (dz / d) * knock;
  if (heavy) {
    setMind(f, 'down');
    if (s.move === 'ground') { s.vy = 3.2; s.move = 'air'; }
  } else if (f.mind !== 'down' && f.mind !== 'stun') setMind(f, 'flinch');
  return false;
}

/** Foe f is parried or shot in the head: dazed, its swing broken off. */
export function dazeFoe(f: Foe): void {
  if (f.mind === 'dead' || f.mind === 'down') return;
  f.roused = true;
  setMind(f, 'stun');
  f.move = -1;
  f.draw = 0;
}

/** Whether foe f is unaware of the hero (a sneakstrike lands ×COMBAT.sneakstrike). */
export function unaware(f: Foe): boolean {
  return !f.roused && f.mind === 'idle';
}

/** Where foe f's head is (world), and its radius, for a headshot. */
export function foeHead(f: Foe, out: Float64Array): number {
  foeJoint(f, J.head, out);
  return 0.19 * FOE_KINDS[f.kind].scale;
}

/** A foe's sneakstrike multiplier for a blow now. */
export function sneakMultiplier(f: Foe, sneaking: boolean): number {
  return sneaking && unaware(f) ? COMBAT.sneakstrike : 1;
}
