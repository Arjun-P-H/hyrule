/**
 * The combat world: the hero's fighting, the foes round it, what flies, what
 * lies on the ground, and how blows land — after Tears of the Kingdom.
 * ---------------------------------------------------------------------------
 * The hero (its HeroState from the engine, driven through updateHero here):
 *   - attack (LMB): the sword's four-hit combo (combat/moves.ts COMBO), a
 *     press during a swing chaining the next; held, a spin attack charges and
 *     is let go on release; in the air a jump attack, sprinting a dash attack;
 *     against a foe unaware of it, sneaking, a sneakstrike.
 *   - shield (RMB held): guards blows from ahead (no harm, a push back); one
 *     that lands just after the shield came up is parried — a Perfect Guard:
 *     the foe is dazed, an arrow flies back where it came from. The shield
 *     locks onto the foe nearest the view; locked on, jump with a direction
 *     is a side hop or a backflip, and a blow slipped at its start opens a
 *     Flurry Rush (the world slows; attack to rush in with a string of cuts).
 *     With the shield up, the action key (E) steps onto it to surf.
 *   - bow (Q held): drawn while the key is held, attack looses an arrow along
 *     the view (to where the crosshair meets the ground or a foe, its drop
 *     made up for); in the air the world slows (bullet time) on stamina.
 *   - throw (G held): the weapon in hand wound up, thrown on release; it
 *     tumbles, strikes for twice its damage and lies where it comes down.
 *   - the action key picks up a weapon, shield or bow lying near; swap (X) changes weapons.
 *   - rocks (combat/stones.ts, rocks/field.ts): with nothing to pick up, the action key lifts the rock
 *     ahead — a stone in one hand, a bigger one overhead in both — and attack or throw hurls it along
 *     the view (the action key puts it down); a boulder is pushed instead, and rolls on down the slopes.
 *     A thrown stone or a rolling boulder strikes the foes in its way.
 *   - the pouch (the inventory screen, combat/pouch.ts): shields and bows are carried, worn and
 *     equipped like weapons (with no shield in hand the right button guards nothing, with no bow Q
 *     draws nothing), and food is kept to be eaten there. Pouch calls act at once.
 * Weapons other than the sword wear out and break. Hearts in quarters: a blow
 * staggers the hero and knocks it back; at none it falls, and the engine
 * puts it back at the spawn. Out of a fight hearts come back slowly.
 *
 * Foes (combat/foes.ts) wake with their camps as the hero comes near; their
 * blows and arrows are resolved here against the guard, the parry, a dodge's
 * moment and the hero's body. A blow that lands stops the world a moment
 * (hit-stop). Everything is deterministic for a given input stream, and the
 * world allocates nothing per tick once its pools are made.
 *
 * Other players (multiplayer: `rivals`, combat/pvp.ts) are fought the same
 * way: the hero's blows and shots go out to them in `outbox`, theirs come in
 * through `receive` and are settled here against this hero when the view of
 * their player reaches them (a guard, a Perfect Guard that dazes the
 * attacker, a dodge and its Flurry Rush, a headshot); the shield locks onto
 * them as onto foes. Foes stay each player's own.
 */

import { COMBAT, FOE, HERO, PLAYER } from '../constants';
import { launchHero, startSurf, stopSurf, updateHero, type HeroInput, type HeroState } from '../hero/locomotion';
import { createCombatPose, J, type CombatPose } from '../hero/rig';
import { elevation, surfaceAt } from '../terrainMath';
import { Ragdoll } from '../hero/ragdoll';
import { COMBO, MOVE, MOVES, THROW_HOLD, THROW_RELEASE } from './moves';
import { BOW, BOWS, FOOD, FOODS, ITEM_CODE, MODEL, SHIELD, SHIELDS, WEAPON, WEAPONS, type Carried } from './weapons';
import {
  FOE_KINDS, campsNear, createFoe, dazeFoe, foeHead, foeJoint, harmFoe, sneakMultiplier, stepFoe, wakeFoe,
  type Camp, type Foe, type FoeWorld,
} from './foes';
import { aimAt, createProjectile, segmentsMeet, stepProjectile, STUCK_LIFE, type Projectile } from './projectiles';
import { STONE, stoneClass, stoneNear } from './stones';
import { dropRock, freeRocks, movedRock, stepRock, takeRock, type RockBody } from '../rocks/field';
import { ROCK_EXTENT } from '../rocks/tiers';
import { HERO_HEAD, MSG, MSG_MAX, OUTCOME, pvpQuarters, rivalGuards, rivalJoint, type Rival } from './pvp';

export type CombatInput = {
  attack: boolean;   // held (the mouse's left button)
  guard: boolean;    // the shield (right button)
  aim: boolean;      // the bow
  throw: boolean;
  action: boolean;   // pick up, step onto the shield
  swap: boolean;     // the next weapon
  jump: boolean;
  sprint: boolean;
  sneak: boolean;
  forward: number;   // −1..1, camera-relative
  strafe: number;
  camYaw: number;    // the orbit (the hero's movement is relative to it)
  camPitch: number;
  // the view: the eye (world) and the direction through the screen's centre, for the bow and throws
  eyeX: number; eyeY: number; eyeZ: number;
  dirX: number; dirY: number; dirZ: number;
};

export function createCombatInput(): CombatInput {
  return {
    attack: false, guard: false, aim: false, throw: false, action: false, swap: false, jump: false, sprint: false, sneak: false,
    forward: 0, strafe: 0, camYaw: 0, camPitch: 0, eyeX: 0, eyeY: 0, eyeZ: 0, dirX: 0, dirY: 0, dirZ: 1,
  };
}

export type HeroAct =
  | 'idle' | 'swing' | 'charge' | 'guard' | 'aim' | 'throw' | 'hurt' | 'stun' | 'dodge' | 'flurry' | 'dead'
  | 'lift' | 'carry' | 'hurl' | 'push'
  | 'thrown' | 'rising'; // flung limp (hero/ragdoll.ts: a horse's blow, thrown off a horse), then getting up

/**
 * Most are where they happened, with a number: a swing's move, damage, a foe's kind… `defeat` (this
 * hero felled another player: `a` its index in `rivals`) and `parried` (another player's Perfect
 * Guard turned this hero's blow) come from the other players. `pickup`, `break`, `equip` (−1: put away or
 * dropped) and `eat` name an item by ITEM_CODE; `empty`: 0 no arrows, 1 no shield, 2 no bow, 3 no room.
 */
export type CombatEventType =
  | 'swing' | 'hit' | 'crit' | 'sneak' | 'block' | 'parry' | 'dodge' | 'flurry' | 'fire' | 'thud' | 'splash'
  | 'throw' | 'alert' | 'down' | 'poof' | 'pickup' | 'hurt' | 'die' | 'break' | 'surf' | 'charged' | 'spin'
  | 'draw' | 'swap' | 'return' | 'empty' | 'defeat' | 'parried' | 'equip' | 'eat'
  | 'vanish' // a slain beast gone in a drift of motes (a: its heading)
  | 'quake';  // something colossal meeting the ground (colossus/colossi.ts; a: the dust's reach, m)

/**
 * A big animal the hero's blows, arrows and throws strike (set each tick as `beasts`): a horse (horse/horses.ts) is
 * a body along its heading, BEAST_HALF either way of (x, z), BEAST_R thick, from BEAST_LOW to BEAST_HIGH over y; a
 * colossus (colossus/colossi.ts) answers for itself, from its skinned collision mesh: `meets` (where a segment first
 * meets it, −1: misses) and `near` (its surface's distance from a point, the point into P).
 */
type Beast = {
  x: number; y: number; z: number; yaw: number; hp: number; hitSwing: number; rockHit: object | null;
  meets?: (ax: number, ay: number, az: number, bx: number, by: number, bz: number, pad: number) => number;
  near?: (px: number, py: number, pz: number, P: Float64Array) => number;
};
const BEAST_HALF = 0.85, BEAST_R = 0.5, BEAST_LOW = 0.6, BEAST_HIGH = 1.75;

export type CombatEvent = { type: CombatEventType; x: number; y: number; z: number; a: number };

export type Pickup = {
  live: boolean;
  kind: 'weapon' | 'arrows' | 'heart' | 'shield' | 'bow' | 'food';
  weapon: number;   // the item: a WEAPONS, SHIELDS, BOWS or FOODS index (by kind)
  wear: number;
  count: number;    // arrows, food, or quarters of a heart
  x: number; y: number; z: number;
  vy: number;
  yaw: number;
  age: number;
};

/** What the pouch carries in lists: weapons, shields and bows. */
export type PouchCat = 'weapon' | 'shield' | 'bow';

/** What a cleared camp leaves by its fire (by the camp's hash). */
const SPOILS: ReadonlyArray<readonly [PouchCat, number]> = [
  ['shield', SHIELD.wooden], ['shield', SHIELD.traveler], ['shield', SHIELD.lid], ['shield', SHIELD.boko],
  ['bow', BOW.boko], ['bow', BOW.soldier],
];

type CampLife = { camp: Camp; awake: boolean; cleared: boolean; clearedAt: number; slots: number[] };

/** A message for the other players (pvp.ts): to one of them (its id), or null: to all. */
export type Outgoing = { to: string | null; d: number[] };
/** A rival's message waiting for the view of it to reach its time (a blow's window: `open`). */
type Incoming = { live: boolean; from: Rival | null; open: boolean; d: Float64Array };

const MAX_SHOTS = 48;
const MAX_PICKUPS = 24;
const MAX_EVENTS = 64;
const MAX_OUT = 16;
const MAX_IN = 32;
const NO_RIVALS: readonly Rival[] = [];
const WRAP = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
const round = (v: number, k: number) => Math.round(v * k) / k;
const P3 = new Float64Array(3);
const P3B = new Float64Array(3);
const V3 = new Float64Array(3);

/** Getting up after being thrown: the ragdoll hands the body back over RISE_HANDOVER s (lying on the back), the
 *  knockdown pose rises from there; the hero is its own again after RISE_TIME s. */
const RISE_HANDOVER = 0.4;
const RISE_TIME = 1.35;
/** Lying on the back in the knockdown pose, the pelvis is this far from the feet (the root), head-ward (m). */
const LIE_REACH = 0.8;

export class CombatWorld implements FoeWorld {
  seed: number;
  readonly hero: HeroState;
  /** Foes wake (false: a world without them, e.g. the benchmarks' and the WebGL2 view's). */
  foesOn = true;
  /** The camps wake as the hero comes near (false: only foes placed with spawnFoe, e.g. in tests). */
  campsOn = true;
  /** The foes and what flies stand still (an overlay such as the multiplayer lobby is open); the hero does not. */
  paused = false;

  // ---- the hero ----
  hearts: number = COMBAT.hearts;
  health: number = COMBAT.hearts * 4;   // quarters
  act: HeroAct = 'idle';
  actT = 0;
  move = -1;
  u = 0;
  combo = 0;           // the combo's next step
  sinceSwing = 99;     // s since the last combo swing ended
  queued = false;
  held = 0;            // s the attack has been held
  charge = 0;
  swingId = 0;
  struck = 0;          // foe slots this swing has struck (bits)
  armed = false;
  peace = 99;          // s since the last fight
  readonly weapons: Carried[] = [{ kind: WEAPON.master, wear: Infinity }];
  equipped = 0;
  swordAway = -1;      // the pickup slot the thrown sword lies in (−1: it is carried)
  arrows: number = COMBAT.arrows;
  // the rest of the pouch: shields (the one on the arm: shieldOn, −1 none), bows (bowOn) and food (a count per FOODS entry)
  readonly shields: Carried[] = [{ kind: SHIELD.hylian, wear: Infinity }];
  shieldOn = 0;
  readonly bows: Carried[] = [{ kind: BOW.traveler, wear: Infinity }];
  bowOn = 0;
  readonly food: number[] = FOODS.map((_, i) => (i === FOOD.apple ? COMBAT.startApples : 0));
  private surfT = 0;   // s on the shield since it last wore
  draw = 0;
  nocked = false;
  nock = 0;
  guardT = 99;         // s since the shield went up
  parryLock = 0;
  invulnerable = 0;
  dodgeDir = 0;        // −1 / +1 a side hop, 0 a backflip
  flurryWait = 0;
  flurryFoe = -1;
  flurryN = 0;
  lock = -1;           // the foe slot the shield is locked onto
  regen = 0;
  thrown = false;      // this throw has let go
  /** Seconds the hero has been down (the engine respawns it). */
  downFor = 0;
  /** The body while thrown (act 'thrown'), and the scratch its ground and pelvis are read into. */
  readonly ragdoll = new Ragdoll();
  private ragAt = new Float64Array(3);
  private ragSurf: [number, number] = [0, 0];
  private ragPlane = { x: 0, z: 0, y: 0, gx: 0, gz: 0 };
  // the rocks in its hands (combat/stones.ts): lifted, carried, being thrown or put down; a boulder being pushed
  /** The rock in the hands, or null. */
  stone: RockBody | null = null;
  /** Held in one hand (a stone), else overhead in both. */
  stoneHand = false;
  /** Where the rock lay when lifted, hero-local: ahead (m) and its middle's height over the feet (the rig reaches there). */
  stoneFwd = 0.6;
  stoneLow = 0.3;
  private stoneDown = false;   // the lift run backward: putting it down
  private readonly stoneFrom = new Float64Array(7);  // where it lay and its turn (x, y, z, q)
  private stoneYaw = 0;        // its turn about the hero's axis while carried
  private pushRock: RockBody | null = null;
  private pushed = false;
  readonly pose: CombatPose = createCombatPose();
  /** The input locomotion gets this tick (the combat layer's say over the movement). */
  readonly heroInput: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0, face: NaN, speedScale: 1, noClimb: false };
  private prev = createCombatInput();

  // ---- pace ----
  hitStop = 0;
  /** The world's and the hero's pace this tick (hit-stop, a flurry, bullet time). */
  worldScale = 1;
  heroScale = 1;
  /** The hero's time step this tick (the rig animates by it). */
  heroDt = 0;
  bulletTime = false;
  time = 0;

  // ---- the rest of the world ----
  readonly foes: Foe[] = [];
  readonly shots: Projectile[] = [];
  readonly pickups: Pickup[] = [];
  readonly camps = new Map<number, CampLife>();
  private near: Camp[] = [];
  private campsAt = [NaN, NaN, NaN];
  /** Foes swinging at the hero now. */
  attackers = 0;
  heroAlive = true;
  heroSneaking = false;
  heroSprinting = false;

  // ---- other players (multiplayer, pvp.ts) ----
  /** The other players as this one sees them (set by the engine every frame; none when alone). */
  rivals: readonly Rival[] = NO_RIVALS;
  /** This player's clock (ms, the one its snapshots carry), set by the engine before each step. */
  netTime = 0;
  /** What to send the other players after this tick: the first `outCount` of `outbox`. */
  readonly outbox: Outgoing[] = [];
  outCount = 0;
  /** The rival the shield is locked onto (else `lock`, a foe, or neither). */
  lockRival: Rival | null = null;
  /** The player whose blow or shot felled the hero (its id), until it is back on its feet. */
  downBy: string | null = null;
  private flurryRival: Rival | null = null;
  private blowSent = -1;   // the swing whose blow has gone out
  private shotId = 0;
  private readonly inbox: Incoming[] = [];
  private readonly spare: number[] = [];

  // ---- what happened (drained by the engine each frame: sounds, effects, the HUD) ----
  readonly events: CombatEvent[] = [];
  eventCount = 0;

  constructor(hero: HeroState, seed: number) {
    this.hero = hero;
    this.seed = seed;
    for (let k = 0; k < FOE.max; k++) this.foes.push(createFoe(k));
    for (let k = 0; k < MAX_SHOTS; k++) this.shots.push(createProjectile());
    for (let k = 0; k < MAX_PICKUPS; k++) {
      this.pickups.push({ live: false, kind: 'arrows', weapon: 0, wear: 0, count: 0, x: 0, y: 0, z: 0, vy: 0, yaw: 0, age: 0 });
    }
    for (let k = 0; k < MAX_EVENTS; k++) this.events.push({ type: 'swing', x: 0, y: 0, z: 0, a: 0 });
    for (let k = 0; k < MAX_OUT; k++) this.outbox.push({ to: null, d: [] });
    for (let k = 0; k < MAX_IN; k++) this.inbox.push({ live: false, from: null, open: false, d: new Float64Array(MSG_MAX) });
  }

  /** The beasts the hero can strike this tick, and what a blow on one does (its owner takes the damage; 0: it took none, a clink). */
  beasts: readonly Beast[] = [];
  onBeast: ((b: Beast, damage: number, fromX: number, fromZ: number) => number | void) | null = null;

  /** An effect at (x, y, z) from outside the fight (a beast's vanishing), for the scene and the sounds. */
  effect(type: CombatEventType, x: number, y: number, z: number, a = 0): void {
    this.emit(type, x, y, z, a);
  }

  /**
   * The beast's body point nearest (px, pz) into P (y its middle), its distance across the ground. A colossus:
   * nearest (px, py, pz), its distance from its surface plus BEAST_R (so a swing's reach reads the same).
   */
  private beastNear(b: Beast, px: number, py: number, pz: number, P: Float64Array): number {
    if (b.near) return b.near(px, py, pz, P) + BEAST_R;
    const fx = Math.sin(b.yaw), fz = Math.cos(b.yaw);
    const t = Math.max(-BEAST_HALF, Math.min(BEAST_HALF, (px - b.x) * fx + (pz - b.z) * fz));
    P[0] = b.x + fx * t; P[1] = b.y + 0.5 * (BEAST_LOW + BEAST_HIGH); P[2] = b.z + fz * t;
    return Math.hypot(px - P[0], pz - P[2]);
  }

  /** Where segment (a → b) first meets beast B's body (its fraction along, −1: misses), with `pad` more thickness. */
  private beastMeets(B: Beast, ax: number, ay: number, az: number, bx: number, by: number, bz: number, pad: number): number {
    if (B.meets) return B.meets(ax, ay, az, bx, by, bz, pad);
    const fx = Math.sin(B.yaw) * BEAST_HALF, fz = Math.cos(B.yaw) * BEAST_HALF, y = B.y + 0.5 * (BEAST_LOW + BEAST_HIGH);
    return segmentsMeet(ax, ay, az, bx, by, bz, B.x - fx, y, B.z - fz, B.x + fx, y, B.z + fz, 0.5 * (BEAST_HIGH - BEAST_LOW) + pad);
  }

  private emit(type: CombatEventType, x: number, y: number, z: number, a = 0): void {
    if (this.eventCount >= MAX_EVENTS) return;
    const e = this.events[this.eventCount++];
    e.type = type; e.x = x; e.y = y; e.z = z; e.a = a;
  }

  /** The weapon in hand (null: none). */
  carried(): Carried | null {
    return this.equipped >= 0 && this.equipped < this.weapons.length ? this.weapons[this.equipped] : null;
  }

  // ---------------------------------------------------------------------------------------------
  /** One tick (real dt, capped by the engine). The hero's locomotion runs in here. */
  step(dt: number, inp: CombatInput): void {
    this.eventCount = 0;
    this.outCount = 0;
    this.time += dt;
    const s = this.hero;
    this.heroAlive = this.act !== 'dead';
    this.heroSneaking = s.sneaking;
    this.heroSprinting = s.sprinting;

    // Pace: a hit-stop all but stops everything; a rush slows the world, not the hero; bullet time both.
    // (Another player is never slowed: a rush after slipping its blow waits at full pace.)
    this.bulletTime = this.act === 'aim' && s.move === 'air' && !s.surfing && s.stamina > 0 && !s.exhausted;
    if (this.hitStop > 0) { this.hitStop -= dt; this.worldScale = 0.04; this.heroScale = 0.04; }
    else if (this.act === 'flurry') { this.worldScale = COMBAT.flurrySlow; this.heroScale = 1; }
    else if (this.flurryWait > 0 && !this.flurryRival) { this.worldScale = COMBAT.dodgeSlow; this.heroScale = 0.55; }
    else if (this.bulletTime) { this.worldScale = COMBAT.bulletTime; this.heroScale = COMBAT.bulletTime; }
    else { this.worldScale = 1; this.heroScale = 1; }
    if (this.paused) this.worldScale = 0;
    const dtw = dt * this.worldScale, dth = dt * this.heroScale;
    this.heroDt = dth;

    this.stepHero(dt, dth, inp);
    this.stepRivals();
    this.stepCamps();
    this.attackers = 0;
    for (const f of this.foes) if (f.live && f.mind === 'attack') this.attackers++;
    for (const f of this.foes) {
      if (!f.live || this.paused) continue;
      stepFoe(f, this, dtw);
      if (f.mind === 'dead' && f.t > FOE.corpse) this.vanish(f);
    }
    if (this.act === 'swing' || this.act === 'flurry') this.swingHits();
    if (!this.paused) this.stepShots(dtw, dt);
    if (!this.paused) this.stepStones(dtw);
    this.stepPickups(dtw, inp);

    const p = this.prev;
    p.attack = inp.attack; p.guard = inp.guard; p.aim = inp.aim; p.throw = inp.throw;
    p.action = inp.action; p.swap = inp.swap; p.jump = inp.jump;
  }

  // ---------------------------------------------------------------------------------------------
  private stepHero(dt: number, dth: number, inp: CombatInput): void {
    const s = this.hero, pv = this.prev, hi = this.heroInput;
    const attackP = inp.attack && !pv.attack, attackR = !inp.attack && pv.attack;
    const guardP = inp.guard && !pv.guard;
    const jumpP = inp.jump && !pv.jump;
    const actionP = inp.action && !pv.action;
    const swapP = inp.swap && !pv.swap;
    const aimP = inp.aim && !pv.aim;
    const throwP = inp.throw && !pv.throw;
    this.invulnerable = Math.max(0, this.invulnerable - dt);
    this.parryLock = Math.max(0, this.parryLock - dt);
    this.guardT += dt;
    this.sinceSwing += dth;
    this.peace += dt;
    this.actT += dth;

    // Locomotion's input, as the player gave it; the acts below take their say.
    hi.forward = inp.forward; hi.strafe = inp.strafe; hi.camYaw = inp.camYaw;
    hi.sprint = inp.sprint; hi.jump = inp.jump; hi.sneak = inp.sneak;
    hi.face = NaN; hi.speedScale = 1; hi.noClimb = false;

    const free = s.move === 'ground' || s.move === 'air' || s.move === 'surf';
    const fresh = !s.exhausted && s.stamina > 0;
    const c = this.carried();
    const kind = c ? WEAPONS[c.kind] : null;
    this.checkLock();

    // Climbing, gliding and swimming put the weapons away and end what the arms were doing (a rock in them falls).
    if (!free && this.stone) this.releaseStone(0, 0, 0);
    if (!free && this.act !== 'dead' && this.act !== 'aim') {
      if (this.act !== 'hurt' && this.act !== 'stun' && this.act !== 'idle') this.setAct('idle');
      this.armed = false;
    }
    // A sprint, a surf or a long peace put them away too.
    if (this.act === 'idle' && (s.sprinting || s.surfing || this.peace > COMBAT.sheatheAfter)) this.armed = false;
    // A blow slipped at the last moment: attack (in the air or landed since) rushes in.
    const rush = this.flurryWait > 0 && attackP && this.act !== 'flurry' && this.act !== 'dead' && this.act !== 'hurt' && this.act !== 'stun';
    if (rush) this.startFlurry();

    if (!rush) switch (this.act) {
      case 'dead': {
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
        this.downFor += dt;
        break;
      }
      case 'hurt': {
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
        if (this.actT > COMBAT.hurtTime) this.setAct('idle');
        break;
      }
      case 'thrown': {
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
        this.stepThrown(dt);
        break;
      }
      case 'rising': {
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
        if (this.actT > RISE_TIME) this.setAct('idle');
        break;
      }
      case 'stun': {
        // (another player's Perfect Guard turned the blow: dazed a moment, open to theirs)
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
        if (this.actT > COMBAT.parriedStun) this.setAct('idle');
        break;
      }
      case 'dodge': {
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
        hi.face = this.lockYaw(NaN);
        if (s.move !== 'air' && this.actT > 0.12) {
          this.setAct(inp.guard ? 'guard' : 'idle');
          this.guardT = COMBAT.parryWindow;
        }
        break;
      }
      case 'flurry': {
        this.stepFlurry(dth, hi);
        break;
      }
      case 'swing': {
        const m = MOVES[this.move];
        this.u += dth * (kind ? kind.pace : 1) / m.dur;
        // the lunge: forward along the facing (toward the foe locked onto) through the blow
        hi.forward = this.u >= m.lungeAt[0] && this.u <= m.lungeAt[1] ? m.lunge / HERO.runSpeed : 0;
        hi.strafe = 0;
        hi.camYaw = s.yaw;
        hi.face = this.u < m.hit[0] ? this.lockYaw(s.yaw) : s.yaw;
        hi.sprint = false; hi.jump = false; hi.noClimb = true;
        if (attackP) this.queued = true;
        if (this.move === MOVE.spin) this.pose.spin = -2 * Math.PI * smooth(0.06, 0.8, this.u);
        if (this.queued && this.u >= m.chain && m.chain < 1) {
          this.queued = false;
          this.combo = (this.combo + 1) % COMBO.length;
          this.startSwing(COMBO[this.combo]);
          break;
        }
        if (this.u >= 1) {
          this.sinceSwing = 0;
          this.pose.spin = 0;
          this.setAct(inp.guard ? 'guard' : 'idle');
          if (inp.guard) this.guardT = COMBAT.parryWindow; // (no window from a swing's end)
        }
        break;
      }
      case 'charge': {
        hi.speedScale = 0.35; hi.sprint = false;
        hi.face = this.lockYaw(NaN);
        const was = this.charge;
        this.charge = Math.min(1, this.charge + dth / COMBAT.chargeFull);
        if (was < 1 && this.charge >= 1) this.emit('charged', s.x, s.y + 1.2, s.z);
        s.stamina = Math.max(0, s.stamina - COMBAT.staminaCharge * dt);
        s.rest = 0;
        if (!inp.attack) {
          if (this.charge >= 0.35 && s.stamina > 0) {
            s.stamina = Math.max(0, s.stamina - COMBAT.staminaSpin);
            if (s.stamina <= 0) s.exhausted = true;
            this.startSwing(MOVE.spin);
            this.emit('spin', s.x, s.y + 1, s.z);
          } else this.setAct('idle');
          this.charge = 0;
        } else if (!fresh) {
          this.charge = 0;
          this.setAct('idle');
        }
        break;
      }
      case 'guard': {
        hi.speedScale = COMBAT.guardSpeed; hi.sprint = false; hi.noClimb = true;
        hi.face = this.lockYaw(inp.camYaw);
        if (!inp.guard || this.shieldOn < 0) {
          this.parryLock = this.guardT < COMBAT.parryWindow + 0.05 ? COMBAT.parryLockout : 0;
          this.setAct('idle');
          break;
        }
        if (actionP && (s.move === 'ground' || s.move === 'air')) {
          // step onto the shield
          startSurf(s);
          this.setAct('idle');
          this.emit('surf', s.x, s.y, s.z);
          break;
        }
        if (jumpP && this.lockedBody() && s.move === 'ground') {
          const side = Math.abs(inp.strafe) > 0.3 ? Math.sign(inp.strafe) : 0;
          if (side !== 0 || inp.forward < -0.3) { this.startDodge(side); hi.jump = false; break; }
        }
        if (attackP) { this.attackPress(inp); break; }
        // (a foe locked onto gone: the next one nearest the view, now and then)
        if (!this.lockedBody() && this.actT % 0.4 < dth) this.pickLock(inp);
        break;
      }
      case 'aim': {
        hi.speedScale = COMBAT.aimSpeed; hi.sprint = false; hi.noClimb = true;
        hi.face = inp.camYaw;
        if (s.move === 'glide') s.move = 'air'; // (the glider folded to draw: bullet time)
        if (!inp.aim || this.bowOn < 0 || s.move === 'climb' || s.move === 'swim') {
          this.draw = 0;
          this.setAct('idle');
          break;
        }
        if (this.bulletTime) {
          s.stamina = Math.max(0, s.stamina - COMBAT.staminaBulletTime * dt);
          s.rest = 0;
          if (s.stamina <= 0) s.exhausted = true;
        }
        if (!this.nocked) {
          if (this.arrows > 0) {
            this.nock += dt;
            if (this.nock >= COMBAT.nockTime) { this.nocked = true; this.nock = 0; this.draw = 0; }
          }
        } else {
          const was = this.draw;
          this.draw = Math.min(1, this.draw + dt / COMBAT.drawTime);
          if (was === 0) this.emit('draw', s.x, s.y + 1.4, s.z);
        }
        if (attackP) {
          if (this.nocked && this.draw >= 0.2) this.fireArrow(inp);
          else if (this.arrows <= 0) this.emit('empty', s.x, s.y + 1.4, s.z);
        }
        break;
      }
      case 'throw': {
        hi.speedScale = COMBAT.aimSpeed; hi.sprint = false; hi.noClimb = true;
        hi.face = inp.camYaw;
        const m = MOVES[MOVE.throw];
        const held = inp.throw && !this.thrown;
        this.u = held ? Math.min(THROW_HOLD, this.u + dth / (m.dur * 0.6)) : this.u + dth / m.dur;
        if (!this.thrown && !inp.throw && this.u >= THROW_RELEASE) this.throwWeapon(inp);
        if (this.u >= 1) this.setAct('idle');
        break;
      }
      case 'lift': {
        // stooping to the rock and up with it (run backward: putting it down)
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false; hi.noClimb = true;
        const b = this.stone;
        if (!b) { this.setAct('idle'); break; }
        if (!this.stoneDown) {
          hi.face = Math.atan2(this.stoneFrom[0] - s.x, this.stoneFrom[2] - s.z);
          this.u += dth / STONE.liftTime[this.stoneHand ? 0 : 1];
          if (this.u >= 1) { this.u = 1; this.setAct('carry'); }
        } else {
          this.u -= dth / STONE.downTime;
          if (this.u <= 0.3) { this.releaseStone(Math.sin(s.yaw) * 0.4, 0, Math.cos(s.yaw) * 0.4); this.setAct('idle'); }
        }
        break;
      }
      case 'carry': {
        const k = this.stoneHand ? 0 : 1;
        hi.speedScale = STONE.carrySpeed[k]; hi.noClimb = true;
        hi.sprint = hi.sprint && this.stoneHand;
        if (!this.stoneHand) hi.jump = false;
        if (!this.stone) { this.setAct('idle'); break; }
        if (attackP || throwP) {
          this.setAct('hurl');
          this.u = 0;
          this.thrown = false;
          this.peace = 0;
        } else if (actionP && s.move === 'ground') {
          // put it down ahead: the lift run backward, to where it will lie
          const r = this.stone, f = this.stoneFrom;
          this.setAct('lift');
          this.u = 1;
          this.stoneDown = true;
          this.stoneFwd = 0.4 + r.reach;
          const x = s.x + Math.sin(s.yaw) * this.stoneFwd, z = s.z + Math.cos(s.yaw) * this.stoneFwd;
          const y = elevation(x, z, this.seed, 0) + r.roll * 0.9 - this.stoneMid(r);
          f[0] = x; f[1] = y; f[2] = z; f[3] = r.qx; f[4] = r.qy; f[5] = r.qz; f[6] = r.qw;
          this.stoneLow = y + this.stoneMid(r) - (s.y - PLAYER.groundBias);
        }
        break;
      }
      case 'hurl': {
        hi.speedScale = 0.3; hi.sprint = false; hi.jump = false; hi.noClimb = true;
        hi.face = inp.camYaw;
        this.u += dth / STONE.hurlTime[this.stoneHand ? 0 : 1];
        if (!this.thrown && this.u >= STONE.release) this.throwStone(inp);
        if (this.u >= 1) this.setAct('idle');
        break;
      }
      case 'push': {
        hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false; hi.noClimb = true;
        const b = this.pushRock;
        if (b) hi.face = Math.atan2(b.x - s.x, b.z - s.z);
        this.u += dth / STONE.pushTime;
        if (!this.pushed && this.u >= STONE.pushAt) this.shove();
        if (this.u >= 1) { this.pushRock = null; this.setAct('idle'); }
        break;
      }
      case 'idle': {
        if (actionP && s.move === 'ground' && !s.surfing && this.nearestPickup() < 0) {
          // a rock ahead: lift it, or push a boulder
          const b = stoneNear(s, this.seed);
          const cls = b ? stoneClass(b) : 0;
          if (b && (cls === 1 || cls === 2)) { this.startLift(b, cls === 1); break; }
          if (b && cls === 3) { this.startPush(b); break; }
        }
        if (attackP && free) { this.attackPress(inp); break; }
        if (inp.attack && free && this.armed && c && this.held > COMBAT.chargeAfter && s.move === 'ground' && fresh && !s.surfing) {
          this.setAct('charge');
          this.charge = 0;
          break;
        }
        if (guardP && free && !s.surfing && this.shieldOn < 0) this.emit('empty', s.x, s.y + 1.2, s.z, 1);
        if (inp.guard && free && !s.surfing && this.shieldOn >= 0) {
          this.setAct('guard');
          this.armed = true;
          this.peace = 0;
          if (guardP) {
            this.guardT = this.parryLock > 0 ? COMBAT.parryWindow : 0;
            this.pickLock(inp);
          } else this.guardT = COMBAT.parryWindow;
          break;
        }
        if (aimP && this.bowOn < 0 && (free || s.move === 'glide')) this.emit('empty', s.x, s.y + 1.4, s.z, 2);
        if (inp.aim && this.bowOn >= 0 && (free || s.move === 'glide') && (aimP || this.actT > 0.05)) {
          if (s.surfing) stopSurf(s);
          this.setAct('aim');
          this.draw = 0;
          this.peace = 0;
          break;
        }
        if (throwP && c && free && !s.surfing) {
          this.setAct('throw');
          this.move = MOVE.throw;
          this.u = 0;
          this.thrown = false;
          this.armed = true;
          this.peace = 0;
          break;
        }
        if (actionP && s.surfing) { stopSurf(s); break; }
        if (swapP && this.weapons.length > 1) {
          this.equipped = (this.equipped + 1) % this.weapons.length;
          this.emit('swap', s.x, s.y + 1, s.z, this.weapons[this.equipped].kind);
        }
        break;
      }
    }
    this.held = inp.attack ? this.held + dth : 0;
    if (attackR) this.held = 0;
    // Surfing wears the shield underfoot.
    if (s.surfing && this.shieldOn >= 0) {
      this.surfT += dth;
      if (this.surfT >= COMBAT.surfWear) { this.surfT = 0; this.wearShield(); }
    } else this.surfT = 0;
    if (this.act !== 'flurry') this.flurryWait = Math.max(0, this.flurryWait - dt);

    // Hearts come back out of a fight.
    let fighting = false;
    for (const f of this.foes) if (f.live && f.roused && f.mind !== 'dead') { fighting = true; break; }
    if (fighting) this.peace = Math.min(this.peace, 1);
    this.regen = fighting || this.act === 'dead' ? 0 : this.regen + dt;
    if (this.regen > COMBAT.regenDelay && this.health < this.hearts * 4) {
      this.health = Math.min(this.hearts * 4, this.health + COMBAT.regenRate * dt);
    }

    if (this.act !== 'flurry' && this.act !== 'thrown') updateHero(s, hi, dth, this.seed);
    this.fillPose(inp);
  }

  /** An attack press with nothing under way: a jump attack, a dash attack, or the next of the combo. */
  private attackPress(inp: CombatInput): void {
    const s = this.hero;
    if (!this.carried()) { this.emit('empty', s.x, s.y + 1.2, s.z); return; }
    if (s.surfing) stopSurf(s);
    if (s.move === 'air' && s.vy > -9) { this.startSwing(MOVE.jump); return; }
    if (s.sprinting) { this.startSwing(MOVE.dash); return; }
    this.combo = this.sinceSwing < COMBAT.comboWindow ? (this.combo + 1) % COMBO.length : 0;
    this.startSwing(COMBO[this.combo]);
    void inp;
  }

  private setAct(a: HeroAct): void {
    this.act = a;
    this.actT = 0;
    if (a !== 'swing' && a !== 'throw') this.move = -1;
    this.pose.spin = 0;
  }

  private startSwing(move: number): void {
    const s = this.hero;
    this.setAct('swing');
    this.move = move;
    this.u = 0;
    this.struck = 0;
    this.swingId++;
    this.armed = true;
    this.peace = 0;
    this.queued = false;
    this.emit('swing', s.x, s.y + 1.1, s.z, move);
  }

  private startDodge(side: number): void {
    const s = this.hero;
    this.setAct('dodge');
    this.dodgeDir = side;
    const fx = Math.sin(s.yaw), fz = Math.cos(s.yaw), rx = Math.cos(s.yaw), rz = -Math.sin(s.yaw);
    if (side !== 0) launchHero(s, rx * side * COMBAT.sideHop[0], COMBAT.sideHop[1], rz * side * COMBAT.sideHop[0]);
    else launchHero(s, -fx * COMBAT.backflip[0], COMBAT.backflip[1], -fz * COMBAT.backflip[0]);
    this.emit('dodge', s.x, s.y + 0.5, s.z, side);
  }

  /** What a Flurry Rush is after (the foe's body, or the other player's): null once it is down or gone. */
  private flurryBody(): HeroState | null {
    const r = this.flurryRival;
    if (r) return this.rivals.indexOf(r) >= 0 && r.pose.act !== 'dead' ? r.body : null;
    const f = this.foes[this.flurryFoe];
    return f && f.live && f.mind !== 'dead' ? f.body : null;
  }

  private startFlurry(): void {
    const b = this.flurryBody();
    this.flurryWait = 0;
    if (!b) return;
    this.setAct('flurry');
    this.flurryN = 0;
    this.move = MOVE.flurryA;
    this.u = 0;
    this.struck = 0;
    this.swingId++;
    this.armed = true;
    this.emit('flurry', b.x, b.y + 1, b.z);
  }

  /** A Flurry Rush: the hero rushes in before the foe (or the other player) and cuts, again and again, the world slowed. */
  private stepFlurry(dth: number, hi: HeroInput): void {
    const s = this.hero, b = this.flurryBody();
    const f = this.flurryRival ? null : this.foes[this.flurryFoe];
    hi.forward = 0; hi.strafe = 0; hi.jump = false; hi.sprint = false;
    if (!b) { this.setAct('idle'); return; }
    // in front of it, 1.3 m off, facing it (gliding there fast)
    const dx = s.x - b.x, dz = s.z - b.z, d = Math.hypot(dx, dz) || 1;
    const tx = b.x + (dx / d) * 1.3, tz = b.z + (dz / d) * 1.3;
    const k = 1 - Math.exp(-14 * dth);
    s.x += (tx - s.x) * k; s.z += (tz - s.z) * k;
    s.yaw = Math.atan2(b.x - s.x, b.z - s.z);
    s.vx = 0; s.vz = 0;
    const g = elevation(s.x, s.z, this.seed, 0);
    if (s.move === 'ground' || s.y < g + PLAYER.groundBias) { s.y = g + PLAYER.groundBias; s.vy = 0; s.ground = g; }
    else { s.vy -= PLAYER.gravity * dth; s.y += s.vy * dth; }
    const m = MOVES[this.move];
    this.u += dth / m.dur;
    if (this.u >= 1) {
      this.flurryN++;
      if (this.flurryN >= COMBAT.flurryHits || (f && f.mind === 'dead')) {
        // the last cut drives a foe back
        if (f && f.mind !== 'dead') harmFoe(f, 0, s.x, s.z, 6, false);
        this.setAct('idle');
        return;
      }
      this.move = this.flurryN % 2 ? MOVE.flurryB : MOVE.flurryA;
      this.u = 0;
      this.struck = 0;
      this.swingId++;
      this.emit('swing', s.x, s.y + 1.1, s.z, this.move);
    }
  }

  /** What the rig is to show of the hero this tick. */
  private fillPose(inp: CombatInput): void {
    const p = this.pose, s = this.hero, c = this.carried();
    p.weapon = c ? WEAPONS[c.kind].model : MODEL.none;
    p.armed = this.armed && this.act !== 'aim';
    p.shield = this.shieldOn >= 0;
    p.shieldModel = this.shieldOn >= 0 ? SHIELDS[this.shields[this.shieldOn].kind].model : 0;
    p.bowModel = this.bowOn >= 0 ? BOWS[this.bows[this.bowOn].kind].model : 0;
    p.glow = this.act === 'charge' ? this.charge : this.act === 'swing' && this.move === MOVE.spin ? 1 - this.u : 0;
    p.flip = 0; p.roll = 0;
    p.draw = -1;
    p.rag = this.ragdoll.p;
    p.ragW = this.act === 'thrown' ? 1 : this.act === 'rising' ? 1 - smooth(0, RISE_HANDOVER - 0.05, this.actT) : 0;
    p.pitch = inp.camPitch;
    const st = this.stone ?? this.pushRock;
    p.rock = st ? st.reach : 0;
    p.rockHalf = st ? 0.5 * st.height : 0;
    p.rockHand = this.stoneHand;
    p.rockFwd = this.stoneFwd;
    p.rockLow = this.stoneLow;
    switch (this.act) {
      case 'lift': p.act = 'lift'; p.move = -1; p.u = this.u; p.armed = false; break;
      case 'carry': p.act = 'carry'; p.move = -1; p.u = this.actT; p.armed = false; break;
      case 'hurl': p.act = 'hurl'; p.move = -1; p.u = Math.min(1, this.u); p.armed = false; break;
      case 'push': p.act = 'push'; p.move = -1; p.u = Math.min(1, this.u); p.armed = false; break;
      case 'swing': case 'flurry': p.act = 'swing'; p.move = this.move; p.u = Math.min(1, this.u); break;
      case 'throw': p.act = 'throw'; p.move = MOVE.throw; p.u = Math.min(1, this.u); break;
      case 'charge': p.act = 'charge'; p.move = -1; p.u = this.actT; break;
      case 'guard': p.act = 'guard'; p.move = -1; p.u = this.actT; break;
      case 'aim': p.act = 'aim'; p.move = -1; p.u = this.actT; p.draw = this.nocked ? this.draw : -1; break;
      case 'hurt': p.act = 'hurt'; p.move = -1; p.u = this.actT; break;
      case 'stun': p.act = 'stun'; p.move = -1; p.u = this.actT; break;
      case 'dead': p.act = 'dead'; p.move = -1; p.u = this.actT; break;
      case 'thrown': p.act = 'none'; p.move = -1; p.u = this.actT; p.armed = false; break;
      // (handed back lying on the back, then up from the knockdown pose)
      case 'rising': p.act = this.actT < RISE_HANDOVER ? 'down' : 'none'; p.move = -1; p.u = this.actT; p.armed = false; break;
      case 'dodge': {
        p.act = this.dodgeDir === 0 ? 'none' : 'guard'; p.move = -1; p.u = this.actT;
        const k = smooth(0, 0.5, this.actT);
        if (this.dodgeDir === 0) p.flip = -2 * Math.PI * k;
        else p.roll = -0.35 * this.dodgeDir * Math.sin(Math.PI * k);
        break;
      }
      default: p.act = 'none'; p.move = -1; p.u = this.actT;
    }
    if (p.act !== 'swing') p.spin = 0;
    void s;
  }

  // ---------------------------------------------------------------------------------------------
  /** The yaw from the hero to body b. */
  private yawTo(b: HeroState): number {
    return Math.atan2(b.x - this.hero.x, b.z - this.hero.z);
  }

  /** The body the shield is locked onto (a foe's or another player's), if any. */
  lockedBody(): HeroState | null {
    if (this.lock >= 0) return this.foes[this.lock].body;
    return this.lockRival ? this.lockRival.body : null;
  }

  /** The yaw to what the shield is locked onto, else `otherwise`. */
  private lockYaw(otherwise: number): number {
    const b = this.lockedBody();
    return b ? this.yawTo(b) : otherwise;
  }

  /** Let go of a lock whose foe or player is down, gone or too far. */
  private checkLock(): void {
    if (this.lock >= 0) {
      const f = this.foes[this.lock];
      if (!f.live || f.mind === 'dead' || !this.inLockRange(f.body)) this.lock = -1;
    }
    const r = this.lockRival;
    if (r && (this.rivals.indexOf(r) < 0 || r.pose.act === 'dead' || !this.inLockRange(r.body))) this.lockRival = null;
  }

  private inLockRange(b: HeroState): boolean {
    return Math.hypot(b.x - this.hero.x, b.z - this.hero.z) < COMBAT.lockRange * 1.3;
  }

  /** How well body b (in lock range) suits a lock from this view: higher nearer the view's centre and the hero. */
  private lockScore(b: HeroState, inp: CombatInput): number {
    const s = this.hero;
    const d = Math.hypot(b.x - s.x, b.y - s.y, b.z - s.z);
    if (d > COMBAT.lockRange) return -Infinity;
    const ex = b.x - inp.eyeX, ey = b.y + 1 - inp.eyeY, ez = b.z - inp.eyeZ;
    const el = Math.hypot(ex, ey, ez) || 1;
    const facing = (ex * inp.dirX + ey * inp.dirY + ez * inp.dirZ) / el;
    if (facing < 0.35 && d > 4) return -Infinity;
    return facing * 2 - d / COMBAT.lockRange;
  }

  /** Lock onto the foe or the other player nearest the view's centre within the lock range (or nothing). */
  private pickLock(inp: CombatInput): void {
    let best = -1, bestRival: Rival | null = null, bestScore = -Infinity;
    for (const f of this.foes) {
      if (!f.live || f.mind === 'dead') continue;
      const score = this.lockScore(f.body, inp);
      if (score > bestScore) { bestScore = score; best = f.slot; }
    }
    for (const r of this.rivals) {
      if (r.pose.act === 'dead') continue;
      const score = this.lockScore(r.body, inp);
      if (score > bestScore) { bestScore = score; best = -1; bestRival = r; }
    }
    this.lock = best;
    this.lockRival = bestRival;
  }

  /** The hero's swing against every foe it can reach, once per foe per swing. */
  private swingHits(): void {
    const s = this.hero, m = MOVES[this.move];
    if (this.u < m.hit[0] || this.u > m.hit[1]) return;
    const c = this.carried();
    const base = c ? WEAPONS[c.kind].damage : 0;
    // (a spin sweeps all round; the yaw the blade points now, for the others: the facing)
    const yaw = s.yaw;
    const fx = Math.sin(yaw), fz = Math.cos(yaw);
    const cosArc = Math.cos(m.arc);
    let landed = false, heavy = false;
    for (const f of this.foes) {
      if (!f.live || f.mind === 'dead' || this.struck & (1 << f.slot)) continue;
      if (this.act === 'flurry' && f.slot !== this.flurryFoe) continue;
      const scale = FOE_KINDS[f.kind].scale;
      const dx = f.body.x - s.x, dz = f.body.z - s.z, dy = f.body.y - s.y;
      const d = Math.hypot(dx, dz);
      if (d > m.reach + 0.35 * scale || dy > 1.8 || dy < -1.6) continue;
      if (m.arc < Math.PI && d > 0.5 && (dx * fx + dz * fz) / d < cosArc) continue;
      this.struck |= 1 << f.slot;
      const sneak = sneakMultiplier(f, s.sneaking || this.heroSneaking);
      const dmg = base * m.damage * sneak;
      foeJoint(f, J.chest, P3);
      const died = harmFoe(f, dmg, s.x, s.z, m.knock, m.heavy);
      f.lastSwing = this.swingId;
      this.emit(sneak > 1 ? 'sneak' : 'hit', P3[0], P3[1], P3[2], dmg);
      if (died) this.emit('down', P3[0], P3[1], P3[2], f.kind);
      landed = true;
      heavy ||= m.heavy || died;
      this.peace = 0;
    }
    // Beasts (the horses): their bodies within the swing's reach and arc
    if (this.act !== 'flurry') for (const b of this.beasts) {
      if (b.hp <= 0 || b.hitSwing === this.swingId) continue;
      const d = this.beastNear(b, s.x, s.y - PLAYER.groundBias + 1, s.z, P3);
      const dy = b.near ? 0 : b.y + BEAST_LOW - (s.y - PLAYER.groundBias);
      if (d > m.reach + BEAST_R || dy > 1.6 || dy < -2) continue;
      if (m.arc < Math.PI && d > 0.5 && ((P3[0] - s.x) * fx + (P3[2] - s.z) * fz) / d < cosArc) continue;
      b.hitSwing = this.swingId;
      const dmg = base * m.damage;
      const took = this.onBeast?.(b, dmg, s.x, s.z);
      this.emit(took === 0 ? 'block' : 'hit', P3[0], P3[1], P3[2], dmg);
      landed = true;
      heavy ||= m.heavy;
      this.peace = 0;
    }
    // Other players: the blow goes out to them (each settles it against its own hero, pvp.ts); where
    // this player sees them it lands at once, for the spark, the pause and the wear.
    if (this.rivals.length && (this.act !== 'flurry' || this.flurryRival)) {
      if (this.blowSent !== this.swingId) {
        this.blowSent = this.swingId;
        const ms = ((m.hit[1] - m.hit[0]) * m.dur / (c ? WEAPONS[c.kind].pace : 1)) * 1000 + 60;
        const rush = this.act === 'flurry';
        const d = this.post(rush ? this.flurryRival!.id : null, 9);
        d[0] = MSG.blow; d[1] = Math.round(this.netTime); d[2] = this.swingId; d[3] = round(base * m.damage, 100);
        d[4] = m.knock; d[5] = m.reach; d[6] = round(m.arc, 1000); d[7] = Math.round(ms); d[8] = rush ? 1 : 0;
      }
      for (const r of this.rivals) {
        if (r.hitSwing === this.swingId || r.pose.act === 'dead') continue;
        if (this.act === 'flurry' && r !== this.flurryRival) continue;
        const b = r.body;
        const dx = b.x - s.x, dz = b.z - s.z, dy = b.y - s.y;
        const d = Math.hypot(dx, dz);
        if (d > m.reach + 0.35 || Math.abs(dy) > 1.7) continue;
        if (m.arc < Math.PI && d > 0.5 && (dx * fx + dz * fz) / d < cosArc) continue;
        r.hitSwing = this.swingId;
        rivalJoint(r, J.chest, P3);
        this.emit(this.act !== 'flurry' && rivalGuards(r, s.x, s.z) ? 'block' : 'hit', P3[0], P3[1], P3[2], 0);
        landed = true;
        heavy ||= m.heavy;
        this.engaged();
      }
    }
    if (landed) {
      this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * (heavy ? 2 : 1));
      this.wear();
    }
  }

  /** A blow landed with the weapon in hand: it wears (the sword never does); at none it breaks. */
  private wear(): void {
    const c = this.carried();
    if (!c || c.wear === Infinity) return;
    c.wear--;
    if (c.wear <= 0) {
      const s = this.hero;
      this.emit('break', s.x + Math.sin(s.yaw) * 0.8, s.y + 1.1, s.z + Math.cos(s.yaw) * 0.8, c.kind);
      this.weapons.splice(this.equipped, 1);
      this.equipped = this.weapons.findIndex((w) => w.kind === WEAPON.master);
      if (this.equipped < 0) this.equipped = this.weapons.length ? 0 : -1;
    }
  }

  /** A blow or arrow guarded with the shield wears it (the Hylian Shield never wears); at none it breaks. */
  private wearShield(): void {
    const c = this.shieldOn >= 0 ? this.shields[this.shieldOn] : null;
    if (!c || c.wear === Infinity) return;
    c.wear--;
    if (c.wear <= 0) {
      const s = this.hero;
      this.emit('break', s.x + Math.sin(s.yaw) * 0.45, s.y + 1.15, s.z + Math.cos(s.yaw) * 0.45, ITEM_CODE.shield + c.kind);
      this.shields.splice(this.shieldOn, 1);
      this.shieldOn = this.shields.findIndex((q) => q.kind === SHIELD.hylian);
      if (s.surfing) stopSurf(s);
    }
  }

  /** An arrow loosed wears the bow (the Traveler's Bow never wears); at none it breaks. */
  private wearBow(): void {
    const c = this.bowOn >= 0 ? this.bows[this.bowOn] : null;
    if (!c || c.wear === Infinity) return;
    c.wear--;
    if (c.wear <= 0) {
      const s = this.hero;
      this.emit('break', s.x + Math.sin(s.yaw) * 0.5, s.y + 1.3, s.z + Math.cos(s.yaw) * 0.5, ITEM_CODE.bow + c.kind);
      this.bows.splice(this.bowOn, 1);
      this.bowOn = this.bows.findIndex((q) => q.kind === BOW.traveler);
    }
  }

  /** How hard a guarded blow shoves the hero back, by the shield in hand (× 3.2 m/s). */
  private shieldPush(): number {
    return this.shieldOn >= 0 ? SHIELDS[this.shields[this.shieldOn].kind].push : 1;
  }

  // ---- the foes' side (FoeWorld) ----------------------------------------------------------------
  /** Foe f's blow comes through: guarded, parried, dodged, or it strikes the hero. */
  strike(f: Foe): void {
    const s = this.hero, m = MOVES[f.move];
    if (f.u > m.hit[1]) { f.struck = true; return; }
    if (this.act === 'dead') { f.struck = true; return; }
    const scale = FOE_KINDS[f.kind].scale;
    const dx = s.x - f.body.x, dz = s.z - f.body.z, dy = s.y - f.body.y;
    const d = Math.hypot(dx, dz);
    const reach = m.reach * scale + 0.35;
    // A dodge begun just before the blow slips it perfectly (the hop may already have carried the
    // hero out of reach): a Flurry Rush opens. Later in the hop the blow just misses.
    if (this.act === 'dodge' && d < reach + 1.8 && Math.abs(dy) < 2.5) {
      f.struck = true;
      if (this.actT < COMBAT.dodgeWindow) {
        this.flurryWait = COMBAT.flurryWait;
        this.flurryFoe = f.slot;
        this.flurryRival = null;
        this.emit('dodge', s.x, s.y + 1, s.z, 2);
      }
      return;
    }
    if (d > reach || Math.abs(dy) > 1.7) return;
    const fx = Math.sin(f.body.yaw), fz = Math.cos(f.body.yaw);
    if (d > 0.4 && (dx * fx + dz * fz) / d < Math.cos(m.arc)) return;
    f.struck = true;
    if (this.act === 'flurry' || this.invulnerable > 0) return;
    if (this.guards(f.body.x, f.body.z)) {
      if (this.guardT < COMBAT.parryWindow) {
        dazeFoe(f);
        f.body.vx = -dx / (d || 1) * 4; f.body.vz = -dz / (d || 1) * 4;
        this.guardT = COMBAT.parryWindow; // (one parry a raise)
        this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 2);
        this.emit('parry', s.x + Math.sin(s.yaw) * 0.45, s.y + 1.15, s.z + Math.cos(s.yaw) * 0.45);
      } else {
        const k = 3.2 * this.shieldPush() / (d || 1);
        s.vx = dx * k; s.vz = dz * k;
        this.emit('block', s.x + Math.sin(s.yaw) * 0.45, s.y + 1.15, s.z + Math.cos(s.yaw) * 0.45);
        this.wearShield();
      }
      return;
    }
    this.hurt(FOE_KINDS[f.kind].damage, f.body.x, f.body.z, m.knock);
  }

  /** The shield faces a blow from (x, z): up, and the blow within ±72° of its facing. */
  private guards(x: number, z: number): boolean {
    if (this.act !== 'guard') return false;
    const s = this.hero;
    const dx = x - s.x, dz = z - s.z, d = Math.hypot(dx, dz) || 1;
    return (dx * Math.sin(s.yaw) + dz * Math.cos(s.yaw)) / d > 0.3;
  }

  /**
   * The hero flung limp at (vx, vy, vz) m/s from the pose `local` (the rig's joints, hero-local): a horse's blow,
   * thrown off a horse. It flies and tumbles (hero/ragdoll.ts), lands, comes to rest and gets up.
   */
  throwHero(vx: number, vy: number, vz: number, local: ArrayLike<number>): void {
    const s = this.hero;
    if (this.act === 'dead' || this.act === 'thrown') return;
    if (this.stone) this.releaseStone(0, 0, 0);
    if (s.surfing) stopSurf(s);
    this.charge = 0;
    this.draw = 0;
    this.armed = false;
    this.ragdoll.start(local, s.x, s.y, s.z, s.yaw, vx, vy, vz, 1 / 60);
    this.setAct('thrown');
    this.invulnerable = Math.max(this.invulnerable, COMBAT.invulnerable);
    this.emit('hurt', s.x, s.y + 1.1, s.z, 0);
  }

  /**
   * A colossal blow (colossus/colossi.ts): `q` quarters, past any shield; the hero flung limp at (vx, vy, vz)
   * unless it falls. A dodge (or the moments after a blow) lets it pass.
   */
  smashHero(q: number, vx: number, vy: number, vz: number, local: ArrayLike<number>): void {
    const s = this.hero;
    if (this.invulnerable > 0 || this.act === 'dead' || this.act === 'thrown' || this.act === 'dodge') return;
    this.hurt(q, s.x - vx, s.z - vz, 0);
    if ((this.act as HeroAct) !== 'dead') { this.invulnerable = 0; this.throwHero(vx, vy, vz, local); }
  }

  /** Thrown: the ragdoll a tick on the ground under its pelvis; the hero's root follows it; at rest, handed back. */
  private stepThrown(dt: number): void {
    const s = this.hero, R = this.ragdoll, P = this.ragAt, g = this.ragPlane, sf = this.ragSurf;
    R.at(J.pelvis, P);
    surfaceAt(P[0], P[2], this.seed, sf);
    g.x = P[0]; g.z = P[2]; g.y = sf[0];
    g.gx = (elevation(P[0] + 0.5, P[2], this.seed, 0) - g.y) / 0.5;
    g.gz = (elevation(P[0], P[2] + 0.5, this.seed, 0) - g.y) / 0.5;
    // (into deep water: the body is left to swim)
    if (sf[1] - sf[0] > HERO.swimDepth) { s.move = 'air'; s.vx = s.vz = 0; s.vy = -1; this.setAct('idle'); return; }
    R.step(dt, g);
    R.at(J.pelvis, P);
    s.x = P[0]; s.z = P[2];
    s.y = Math.max(P[1] - 0.9, g.y) + PLAYER.groundBias;
    s.ground = g.y; s.water = sf[1]; s.gx = g.gx; s.gz = g.gz;
    R.velocity(dt, P);
    s.vx = P[0]; s.vy = P[1]; s.vz = P[2];
    s.move = 'ground';
    if (!R.settled) return;
    // At rest: face from the head toward the feet and stand the root where the knockdown pose (lying on the back,
    // rig.ts r.fall) has its feet, so the rig takes the body over where it lies.
    R.at(J.pelvis, P);
    const bx = P[0], bz = P[2];
    R.at(J.head, P);
    const dx = P[0] - bx, dz = P[2] - bz;
    if (Math.hypot(dx, dz) > 0.05) s.yaw = Math.atan2(-dx, -dz);
    s.x = bx + Math.sin(s.yaw) * LIE_REACH; s.z = bz + Math.cos(s.yaw) * LIE_REACH;
    s.y = elevation(s.x, s.z, this.seed, 0) + PLAYER.groundBias;
    s.vx = s.vy = s.vz = 0;
    this.setAct('rising');
  }

  /** The hero takes `q` quarters from a blow from (x, z): it staggers back, or falls. */
  private hurt(q: number, x: number, z: number, knock: number): void {
    const s = this.hero;
    if (this.invulnerable > 0 || this.act === 'dead') return;
    if (this.stone) this.releaseStone(0, 0, 0);
    this.health = Math.max(0, this.health - q);
    this.invulnerable = COMBAT.invulnerable;
    this.regen = 0;
    if (s.surfing) stopSurf(s);
    const dx = s.x - x, dz = s.z - z, d = Math.hypot(dx, dz) || 1;
    launchHero(s, (dx / d) * knock * 0.8, 3.2, (dz / d) * knock * 0.8);
    this.emit('hurt', s.x, s.y + 1.1, s.z, q);
    this.charge = 0;
    this.draw = 0;
    if (this.health <= 0) {
      this.setAct('dead');
      this.downFor = 0;
      this.armed = false;
      this.emit('die', s.x, s.y + 1, s.z);
    } else this.setAct('hurt');
  }

  shoot(f: Foe): void {
    const p = this.freeShot();
    if (!p) return;
    const s = this.hero;
    foeJoint(f, J.handL, P3);
    // at the hero's chest, led a little by its run
    const flight = Math.hypot(s.x - P3[0], s.z - P3[2]) / FOE.arrowSpeed;
    aimAt(P3[0], P3[1], P3[2], s.x + s.vx * flight * 0.6, s.y + 1.1, s.z + s.vz * flight * 0.6, FOE.arrowSpeed, COMBAT.arrowGravity, V3);
    this.launch(p, P3[0], P3[1], P3[2], V3[0], V3[1], V3[2], true, false, FOE.arrowDamage, f.slot);
    this.emit('fire', P3[0], P3[1], P3[2], 1);
  }

  rouse(f: Foe): void {
    foeJoint(f, J.head, P3);
    this.emit('alert', P3[0], P3[1] + 0.45, P3[2], f.slot);
    for (const g of this.foes) {
      if (!g.live || g === f || g.camp !== f.camp || g.roused) continue;
      g.aware = Math.max(g.aware, 0.75);
    }
  }

  /** Foe f goes up in smoke: its weapon, maybe arrows or a heart, are left where it lay. */
  private vanish(f: Foe): void {
    const s = f.body;
    f.live = false;
    this.emit('poof', s.x, s.y + 0.6, s.z, f.kind);
    if (f.weapon >= 0) this.drop('weapon', f.weapon, WEAPONS[f.weapon].durability, 0, s.x, s.y + 0.6, s.z);
    const r = (Math.imul(f.slot + 1, 2654435761) ^ (this.swingId * 7919) ^ Math.floor(this.time * 10)) >>> 0;
    if (f.archer || r % 10 < 4) this.drop('arrows', 0, 0, f.archer ? 5 : 3, s.x + 0.4, s.y + 0.6, s.z);
    if (r % 7 < 2) this.drop('heart', 0, 0, 4, s.x - 0.4, s.y + 0.6, s.z);
    // (the pouch's loot, from the same dice thrown again: food, then a shield — or an archer's bow)
    const kd = FOE_KINDS[f.kind];
    const r2 = Math.imul(r ^ 0x9e3779b9, 2246822519) >>> 0;
    if (r2 % 3 === 0) this.drop('food', kd.food[(r2 >>> 9) & 1], 0, 1, s.x, s.y + 0.6, s.z + 0.45);
    if (f.archer) {
      if (r2 % 2 === 1) this.drop('bow', kd.bow, BOWS[kd.bow].durability, 0, s.x - 0.45, s.y + 0.6, s.z + 0.3);
    } else if ((r2 >>> 4) % 5 === 0) this.drop('shield', kd.shield, SHIELDS[kd.shield].durability, 0, s.x - 0.45, s.y + 0.6, s.z - 0.3);
    const life = this.camps.get(f.camp);
    if (life) {
      const i = life.slots.indexOf(f.slot);
      if (i >= 0) life.slots.splice(i, 1);
      if (!life.slots.length) {
        life.cleared = true; life.clearedAt = this.time; life.awake = false;
        this.spoils(life.camp);
      }
    }
    if (this.lock === f.slot) this.lock = -1;
  }

  // ---- what flies ---------------------------------------------------------------------------------
  private freeShot(): Projectile | null {
    let oldest: Projectile | null = null;
    for (const p of this.shots) {
      if (!p.live) return p;
      if (p.stuck && (!oldest || p.age > oldest.age)) oldest = p;
    }
    return oldest; // (the oldest arrow stuck in the ground gives way)
  }

  private launch(p: Projectile, x: number, y: number, z: number, vx: number, vy: number, vz: number,
    arrow: boolean, friendly: boolean, damage: number, shooter: number): void {
    p.live = true; p.arrow = arrow; p.friendly = friendly;
    p.x = x; p.y = y; p.z = z; p.px = x; p.py = y; p.pz = z;
    p.vx = vx; p.vy = vy; p.vz = vz;
    p.gravity = arrow ? COMBAT.arrowGravity : COMBAT.throwGravity;
    p.damage = damage; p.stuck = false; p.age = 0; p.spin = 0; p.shooter = shooter;
    p.from = null; p.ref = 0;
  }

  /** Where the view's centre meets the ground or a foe within `range` (into P3B), marching the ray. */
  private aimPoint(inp: CombatInput, range: number): void {
    let t = range;
    // the foes first (their bodies as spheres round the chest)
    for (const f of this.foes) {
      if (!f.live || f.mind === 'dead') continue;
      foeJoint(f, J.chest, P3);
      const ox = P3[0] - inp.eyeX, oy = P3[1] - inp.eyeY, oz = P3[2] - inp.eyeZ;
      const b = ox * inp.dirX + oy * inp.dirY + oz * inp.dirZ;
      if (b <= 0 || b > t) continue;
      const d2 = ox * ox + oy * oy + oz * oz - b * b;
      if (d2 < 0.45 * 0.45) t = b;
    }
    // … and the other players
    for (const r of this.rivals) {
      if (r.pose.act === 'dead') continue;
      rivalJoint(r, J.chest, P3);
      const ox = P3[0] - inp.eyeX, oy = P3[1] - inp.eyeY, oz = P3[2] - inp.eyeZ;
      const b = ox * inp.dirX + oy * inp.dirY + oz * inp.dirZ;
      if (b <= 0 || b > t) continue;
      const d2 = ox * ox + oy * oy + oz * oz - b * b;
      if (d2 < 0.45 * 0.45) t = b;
    }
    // the ground: 1.5 m steps, then bisection
    let prev = 0;
    for (let x = 1.5; x < t; x += 1.5) {
      const px = inp.eyeX + inp.dirX * x, py = inp.eyeY + inp.dirY * x, pz = inp.eyeZ + inp.dirZ * x;
      if (py < elevation(px, pz, this.seed, 0)) {
        let a = prev, b = x;
        for (let k = 0; k < 5; k++) {
          const m = 0.5 * (a + b);
          if (inp.eyeY + inp.dirY * m < elevation(inp.eyeX + inp.dirX * m, inp.eyeZ + inp.dirZ * m, this.seed, 0)) b = m; else a = m;
        }
        t = b;
        break;
      }
      prev = x;
    }
    P3B[0] = inp.eyeX + inp.dirX * t; P3B[1] = inp.eyeY + inp.dirY * t; P3B[2] = inp.eyeZ + inp.dirZ * t;
  }

  private fireArrow(inp: CombatInput): void {
    const s = this.hero, p = this.freeShot();
    if (!p) return;
    const speed = COMBAT.arrowSpeed * (0.55 + 0.45 * this.draw);
    // from the bow (ahead of the left shoulder) toward where the crosshair meets something, its drop made up for
    const fx = Math.sin(s.yaw), fz = Math.cos(s.yaw);
    const x = s.x + fx * 0.45 - fz * 0.12, y = s.y - PLAYER.groundBias + 1.42, z = s.z + fz * 0.45 + fx * 0.12;
    this.aimPoint(inp, 150);
    aimAt(x, y, z, P3B[0], P3B[1], P3B[2], speed, COMBAT.arrowGravity, V3);
    const bow = BOWS[this.bowOn >= 0 ? this.bows[this.bowOn].kind : BOW.traveler];
    this.launch(p, x, y, z, V3[0], V3[1], V3[2], true, true, COMBAT.arrowDamage * (0.5 + 0.5 * this.draw) * bow.damage, -1);
    this.sendShot(p);
    this.arrows--;
    this.wearBow();
    this.nocked = false;
    this.nock = 0;
    this.draw = 0;
    this.peace = 0;
    this.emit('fire', x, y, z, 0);
  }

  private throwWeapon(inp: CombatInput): void {
    const s = this.hero, c = this.carried();
    this.thrown = true;
    if (!c) return;
    const p = this.freeShot();
    if (!p) return;
    const fx = Math.sin(s.yaw), fz = Math.cos(s.yaw);
    const x = s.x + fx * 0.3 + fz * 0.2, y = s.y - PLAYER.groundBias + 1.75, z = s.z + fz * 0.3 - fx * 0.2;
    this.aimPoint(inp, 60);
    aimAt(x, y, z, P3B[0], P3B[1], P3B[2], COMBAT.throwSpeed, COMBAT.throwGravity, V3);
    const kind = WEAPONS[c.kind];
    this.launch(p, x, y, z, V3[0], V3[1], V3[2], false, true, kind.damage * COMBAT.throwDamage, -1);
    p.weapon = c.kind;
    p.wear = c.wear;
    this.sendShot(p);
    this.weapons.splice(this.equipped, 1);
    this.equipped = this.weapons.length ? 0 : -1;
    this.peace = 0;
    this.emit('throw', x, y, z, c.kind);
  }

  /** What flies, by dt (the world's pace); another player's by dtReal (that player is never slowed). */
  private stepShots(dt: number, dtReal: number): void {
    const s = this.hero;
    for (const p of this.shots) {
      if (!p.live) continue;
      const pdt = p.from !== null ? dtReal : dt;
      if (p.stuck) {
        p.age += pdt;
        if (p.age > STUCK_LIFE) p.live = false;
        continue;
      }
      const landed = stepProjectile(p, pdt, this.seed);
      // the bodies along the segment it swept
      if (p.from !== null) {
        if (this.act !== 'dead') this.rivalShotHits(p);
      } else if (p.friendly) {
        for (const f of this.foes) {
          if (!f.live || f.mind === 'dead' || !p.live) continue;
          const k = FOE_KINDS[f.kind].scale;
          const hr = foeHead(f, P3);
          const head = segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, P3[0], P3[1], P3[2], P3[0], P3[1], P3[2], hr + (p.arrow ? 0.02 : 0.22));
          foeJoint(f, J.pelvis, P3B);
          foeJoint(f, J.chest, V3);
          const body = head >= 0 ? head : segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, P3B[0], P3B[1] - 0.25 * k, P3B[2], V3[0], V3[1], V3[2], (p.arrow ? 0.3 : 0.45) * k);
          if (body < 0) continue;
          const crit = head >= 0 && p.arrow;
          const dmg = p.damage * (crit ? COMBAT.headshot : 1) * sneakMultiplier(f, this.heroSneaking);
          const died = harmFoe(f, dmg, p.px, p.pz, p.arrow ? 1.2 : 4.5, false);
          if (crit && !died) dazeFoe(f);
          this.emit(crit ? 'crit' : 'hit', p.x, p.y, p.z, dmg);
          if (died) this.emit('down', p.x, p.y, p.z, f.kind);
          this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 0.6);
          if (p.arrow) p.live = false;
          else this.settleWeapon(p, -p.vx * 0.15, -p.vz * 0.15);
        }
        for (const b of this.beasts) {
          if (b.hp <= 0 || !p.live) continue;
          if (this.beastMeets(b, p.px, p.py, p.pz, p.x, p.y, p.z, p.arrow ? 0.02 : 0.2) < 0) continue;
          const took = this.onBeast?.(b, p.damage, p.px, p.pz);
          this.emit(took === 0 ? 'block' : 'hit', p.x, p.y, p.z, p.damage);
          this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 0.6);
          if (p.arrow) p.live = false;
          else this.settleWeapon(p, -p.vx * 0.15, -p.vz * 0.15);
        }
        if (p.live && this.rivals.length) this.shotMeetsRivals(p);
      } else if (this.act !== 'dead' && p.live) {
        const hit = segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, s.x, s.y + 0.35, s.z, s.x, s.y + 1.55, s.z, 0.32);
        if (hit >= 0) {
          const sx = p.shooter >= 0 ? this.foes[p.shooter].body.x : p.px - p.vx;
          const sz = p.shooter >= 0 ? this.foes[p.shooter].body.z : p.pz - p.vz;
          if (this.act === 'dodge' || this.act === 'flurry') { /* slipped */ }
          else if (this.guards(p.px - p.vx, p.pz - p.vz)) {
            if (this.guardT < COMBAT.parryWindow) {
              // a Perfect Guard sends it back where it came from
              const sh = p.shooter >= 0 ? this.foes[p.shooter] : null;
              if (sh && sh.live) {
                foeJoint(sh, J.chest, P3);
                aimAt(p.x, p.y, p.z, P3[0], P3[1], P3[2], FOE.arrowSpeed * 1.3, COMBAT.arrowGravity, V3);
                p.vx = V3[0]; p.vy = V3[1]; p.vz = V3[2];
              } else { p.vx = -p.vx; p.vy = Math.abs(p.vy); p.vz = -p.vz; }
              p.friendly = true;
              p.damage = COMBAT.arrowDamage * 1.5;
              this.guardT = COMBAT.parryWindow;
              this.emit('parry', p.x, p.y, p.z);
            } else {
              p.live = false;
              this.emit('block', p.x, p.y, p.z);
              this.wearShield();
            }
          } else {
            p.live = false;
            this.hurt(p.damage, sx, sz, 1.5);
          }
        }
      }
      if (!p.live) continue;
      // (another player's thrown weapon lies where it came down in that player's world, not here)
      if (landed === 'water') {
        this.emit('splash', p.x, p.y, p.z);
        if (p.arrow || p.from !== null) p.live = false;
        else { this.settleWeapon(p, 0, 0); }
      } else if (landed === 'ground' || landed === 'tree') {
        this.emit('thud', p.x, p.y, p.z, p.arrow ? 0 : 1);
        if (!p.arrow) { if (p.from !== null) p.live = false; else this.settleWeapon(p, 0, 0); }
      }
      if (p.live && !p.stuck && (p.age > 20 || p.y < -500)) p.live = false;
    }
  }

  /** A thrown weapon comes down: it lies there to be picked up (the sword finds its way back in time). */
  private settleWeapon(p: Projectile, vx: number, vz: number): void {
    p.live = false;
    const slot = this.drop('weapon', p.weapon, p.wear, 0, p.x + vx * 0.1, p.y + 0.2, p.z + vz * 0.1);
    if (p.weapon === WEAPON.master) this.swordAway = slot;
  }

  // ---- rocks (combat/stones.ts, rocks/field.ts) -----------------------------------------------------
  /** How far its middle is over its origin (the mesh's: a rock's middle sits a little over it). */
  private stoneMid(b: RockBody): number {
    const e = ROCK_EXTENT[b.kind];
    return 0.5 * (e.top + e.bottom) * b.size * b.ay;
  }

  private startLift(b: RockBody, oneHand: boolean): void {
    const s = this.hero;
    this.setAct('lift');
    this.u = 0;
    this.stoneDown = false;
    this.stoneHand = oneHand;
    this.stone = b;
    this.armed = false;
    const f = this.stoneFrom;
    f[0] = b.x; f[1] = b.y; f[2] = b.z; f[3] = b.qx; f[4] = b.qy; f[5] = b.qz; f[6] = b.qw;
    this.stoneYaw = ((b.rnd & 1023) / 1023) * 2 * Math.PI;
    // where it lies, hero-local (the hero turns to it as it stoops)
    this.stoneFwd = Math.max(0.35, Math.hypot(b.x - s.x, b.z - s.z));
    this.stoneLow = b.y + this.stoneMid(b) - (s.y - PLAYER.groundBias);
    takeRock(b);
    b.state = 'carried';
    this.emit('thud', b.x, b.y, b.z, 0);
  }

  /**
   * The engine's word on where the rig holds the rock now (its middle, world): the rock follows the hands, turned with the
   * hero (lifted: from where it lay, once the hands are on it).
   */
  holdStone(cx: number, cy: number, cz: number): void {
    const b = this.stone;
    if (!b || b.state !== 'carried') return;
    const k = this.act === 'lift' ? smooth(0.36, 0.62, this.u) : 1;
    const f = this.stoneFrom, mid = this.stoneMid(b);
    b.x = f[0] + (cx - f[0]) * k;
    b.y = f[1] + (cy - mid - f[1]) * k;
    b.z = f[2] + (cz - f[2]) * k;
    // (its turn: from the one it lay with to the hero's, the shorter way)
    const yaw = this.hero.yaw + this.stoneYaw;
    let ty = Math.sin(0.5 * yaw), tw = Math.cos(0.5 * yaw);
    if (f[4] * ty + f[6] * tw < 0) { ty = -ty; tw = -tw; }
    const qx = f[3] * (1 - k), qy = f[4] * (1 - k) + ty * k, qz = f[5] * (1 - k), qw = f[6] * (1 - k) + tw * k;
    const l = Math.hypot(qx, qy, qz, qw) || 1;
    b.qx = qx / l; b.qy = qy / l; b.qz = qz / l; b.qw = qw / l;
    movedRock(b);
  }

  /** The rock leaves the hands with a velocity (dropped, put down): it falls where it is. */
  private releaseStone(vx: number, vy: number, vz: number): void {
    const b = this.stone;
    this.stone = null;
    if (!b) return;
    b.state = 'flying';
    b.vx = vx; b.vy = vy; b.vz = vz;
    b.wx = 0; b.wy = 0; b.wz = 0;
    b.struck = 0xffffffff; // (dropped: it harms no one)
    b.still = 0;
  }

  private throwStone(inp: CombatInput): void {
    const b = this.stone, s = this.hero;
    this.thrown = true;
    if (!b) return;
    const k = this.stoneHand ? 0 : 1;
    const mid = this.stoneMid(b);
    this.aimPoint(inp, 60);
    const speed = STONE.speed[k];
    // a foe locked onto, or the one nearest the way it is thrown (within ±20°, in range), draws the throw to its chest
    let target: Foe | null = this.locked();
    if (!target) {
      let best = Math.cos(0.35);
      const hx = Math.sin(inp.camYaw), hz = Math.cos(inp.camYaw);
      for (const f of this.foes) {
        if (!f.live || f.mind === 'dead') continue;
        const dx = f.body.x - b.x, dz = f.body.z - b.z, d = Math.hypot(dx, dz);
        if (d < 1 || d > STONE.aimRange[k]) continue;
        const c = (dx * hx + dz * hz) / d;
        if (c > best) { best = c; target = f; }
      }
    }
    if (target) {
      foeJoint(target, J.chest, P3);
      const t = Math.hypot(P3[0] - b.x, P3[2] - b.z) / speed;
      aimAt(b.x, b.y + mid, b.z, P3[0] + target.body.vx * t, P3[1], P3[2] + target.body.vz * t, speed, PLAYER.gravity, V3);
    } else if (Math.hypot(P3B[0] - b.x, P3B[2] - b.z) > STONE.minRange[k]) aimAt(b.x, b.y + mid, b.z, P3B[0], P3B[1], P3B[2], speed, PLAYER.gravity, V3);
    else {
      // (the crosshair on the ground just ahead: a throw's own arc, up a little from the view)
      const pitch = Math.max(0.12, Math.min(0.9, inp.camPitch + (k ? 0.42 : 0.3)));
      V3[0] = Math.sin(s.yaw) * Math.cos(pitch) * speed; V3[1] = Math.sin(pitch) * speed; V3[2] = Math.cos(s.yaw) * Math.cos(pitch) * speed;
    }
    this.stone = null;
    b.state = 'flying';
    b.vx = V3[0] + s.vx * 0.5; b.vy = V3[1]; b.vz = V3[2] + s.vz * 0.5;
    // (tumbling end over end, about the hero's right)
    const spinRate = (k ? 5 : 9) * (0.8 + 0.4 * ((b.rnd >>> 10) & 255) / 255);
    b.wx = Math.cos(s.yaw) * spinRate; b.wy = 0.6; b.wz = -Math.sin(s.yaw) * spinRate;
    b.struck = 0;
    b.still = 0;
    b.age = 0;
    this.peace = 0;
    this.emit('throw', b.x, b.y, b.z, -1);
  }

  private startPush(b: RockBody): void {
    this.setAct('push');
    this.u = 0;
    this.pushRock = b;
    this.pushed = false;
    this.armed = false;
  }

  /** The hands drive: the boulder rolls off the way the hero faces it. */
  private shove(): void {
    const b = this.pushRock, s = this.hero;
    this.pushed = true;
    if (!b || (b.state !== 'ground' && b.state !== 'loose')) return;
    const dx = b.x - s.x, dz = b.z - s.z, d = Math.hypot(dx, dz) || 1;
    if (d - b.reach > STONE.reach + 0.6) return; // (it was let go of: out of reach)
    takeRock(b);
    const v = STONE.pushSpeed / Math.sqrt(b.size);
    b.state = 'rolling';
    b.vx = (dx / d) * v; b.vz = (dz / d) * v; b.vy = 0;
    b.struck = 0;
    b.still = 0;
    b.age = 0;
    this.peace = 0;
    this.emit('thud', b.x, b.y, b.z, 1);
  }

  /** The rock (or boulder) the action key would take now, for the HUD's prompt: 'Lift', 'Push', or null. */
  stoneHint(): string | null {
    if (this.act === 'carry') return null;
    const s = this.hero;
    if (this.act !== 'idle' || s.move !== 'ground' || s.surfing || this.nearestPickup() >= 0) return null;
    const b = stoneNear(s, this.seed);
    const cls = b ? stoneClass(b) : 0;
    return cls === 1 || cls === 2 ? 'Lift' : cls === 3 ? 'Push' : null;
  }

  /** Rocks thrown and rolling: they fly, roll and come to rest (rocks/field.ts stepRock), and strike the foes in their way. */
  private stepStones(dt: number): void {
    if (dt <= 0) return;
    const rocks = freeRocks();
    for (let i = rocks.length - 1; i >= 0; i--) {
      const b = rocks[i];
      if (b.state !== 'flying' && b.state !== 'rolling') continue;
      const px = b.x, py = b.y, pz = b.z;
      const what = stepRock(b, dt, this.seed);
      if (what === 'water') { this.emit('splash', b.x, b.y, b.z); dropRock(b); continue; }
      if (what === 'ground') this.emit('thud', b.x, b.y, b.z, b.tier >= 2 || !this.isStone(b) ? 1 : 0);
      const v = Math.hypot(b.vx, b.vy, b.vz);
      const harm = b.state === 'flying' ? STONE.flyHarm : STONE.rollHarm;
      if (v < harm || b.struck === 0xffffffff) continue;
      for (const f of this.foes) {
        if (!f.live || f.mind === 'dead' || (b.struck & (1 << f.slot))) continue;
        const k = FOE_KINDS[f.kind].scale;
        foeJoint(f, J.pelvis, P3B);
        foeJoint(f, J.chest, V3);
        const hit = segmentsMeet(px, py + b.roll * 0.3, pz, b.x, b.y + b.roll * 0.3, b.z, P3B[0], P3B[1] - 0.35 * k, P3B[2], V3[0], V3[1] + 0.2 * k, V3[2], 0.42 * k + b.roll * 0.85);
        if (hit < 0) continue;
        b.struck |= 1 << f.slot;
        let dmg: number, knock: number, heavy: boolean;
        if (b.state === 'rolling') {
          dmg = Math.min(STONE.rollMaxDamage, STONE.rollDamage * v * Math.sqrt(b.size));
          knock = 6 + v; heavy = true;
          b.vx *= 0.75; b.vz *= 0.75;
        } else {
          const hand = this.isStone(b) ? 0 : 1;
          dmg = STONE.damage[hand] * Math.min(1.3, v / STONE.speed[hand] + 0.2);
          knock = STONE.knock[hand]; heavy = hand === 1;
          // (it glances off and falls)
          b.vx *= -0.25; b.vz *= -0.25; b.vy = Math.min(b.vy, 1);
        }
        dmg *= sneakMultiplier(f, this.heroSneaking);
        const died = harmFoe(f, dmg, px, pz, knock, heavy);
        this.emit('hit', b.x, b.y + b.roll, b.z, dmg);
        if (died) this.emit('down', b.x, b.y, b.z, f.kind);
        this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 0.6);
        this.peace = 0;
      }
      for (const B of this.beasts) {
        if (B.hp <= 0 || B.rockHit === b) continue;
        if (this.beastMeets(B, px, py + b.roll * 0.3, pz, b.x, b.y + b.roll * 0.3, b.z, 0.3 + b.roll * 0.85) < 0) continue;
        B.rockHit = b;
        const hand = this.isStone(b) ? 0 : 1;
        const dmg = b.state === 'rolling' ? Math.min(STONE.rollMaxDamage, STONE.rollDamage * v * Math.sqrt(b.size))
          : STONE.damage[hand] * Math.min(1.3, v / STONE.speed[hand] + 0.2);
        if (b.state !== 'rolling') { b.vx *= -0.25; b.vz *= -0.25; b.vy = Math.min(b.vy, 1); }
        const took = this.onBeast?.(B, dmg, px, pz);
        this.emit(took === 0 ? 'block' : 'hit', b.x, b.y + b.roll, b.z, dmg);
        this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 0.6);
      }
    }
  }

  /** A stone of one hand (thrown far), else a rock or a boulder. */
  private isStone(b: RockBody): boolean {
    return b.tier === 1 && b.reach <= STONE.handMax;
  }

  // ---- other players (pvp.ts) ---------------------------------------------------------------------
  /**
   * A message from player `id` (the relay's, net/session.ts), held until this world's view of that
   * player has played back to its time. Dropped when malformed, from a player not seen yet, or when
   * nothing is free to hold it.
   */
  receive(id: string, d: readonly number[]): void {
    const k = d[0];
    const n = k === MSG.blow ? 9 : k === MSG.shot ? 11 : k === MSG.result ? 9 : 0;
    if (!n || d.length < n) return;
    const r = this.rivalById(id);
    if (!r) return;
    let m: Incoming | null = null;
    for (const q of this.inbox) if (!q.live) { m = q; break; }
    if (!m) return;
    for (let i = 0; i < n; i++) {
      const v = d[i];
      if (typeof v !== 'number' || !Number.isFinite(v)) return;
      m.d[i] = v;
    }
    m.live = true; m.from = r; m.open = false;
  }

  private rivalById(id: string | null): Rival | null {
    if (id === null) return null;
    for (const r of this.rivals) if (r.id === id) return r;
    return null;
  }

  /** A blow or a shot passed between the hero and another player: a fight (no hearts back, weapons out). */
  private engaged(): void {
    this.regen = 0;
    this.peace = 0;
  }

  /** The rivals' messages whose moment the view of them has reached: blows over their window, shots loosed, outcomes. */
  private stepRivals(): void {
    for (const m of this.inbox) {
      if (!m.live) continue;
      const r = m.from!, d = m.d;
      if (this.rivals.indexOf(r) < 0) { m.live = false; continue; } // (that player is gone)
      if (r.clock < d[1]) continue;
      if (d[0] === MSG.blow) {
        if (!m.open) { m.open = true; this.emit('swing', r.body.x, r.body.y + 1.1, r.body.z, -1); }
        if (this.rivalBlow(r, d) || r.clock > d[1] + Math.min(d[7], 1500)) m.live = false;
      } else {
        if (d[0] === MSG.shot) this.rivalShot(r, d);
        else this.rivalResult(r, d);
        m.live = false;
      }
    }
  }

  /** Rival r's blow (BLOW d) on the hero now: slipped, guarded, parried or struck — false while it has not reached. */
  private rivalBlow(r: Rival, d: Float64Array): boolean {
    const s = this.hero, b = r.body;
    if (this.act === 'dead') return true;
    const dx = s.x - b.x, dz = s.z - b.z, dy = s.y - b.y;
    const dist = Math.hypot(dx, dz);
    const reach = Math.min(d[5], 4) + 0.35;
    // (a Flurry Rush's cut: neither the shield nor a dodge answers it, only being out of reach)
    const rush = d[8] > 0;
    // As a foe's: a dodge begun just before the blow slips it perfectly (a Flurry Rush opens on that
    // player), later in the hop it just misses.
    if (!rush && this.act === 'dodge' && dist < reach + 1.8 && Math.abs(dy) < 2.5) {
      if (this.actT < COMBAT.dodgeWindow) {
        this.flurryWait = COMBAT.flurryWait;
        this.flurryFoe = -1;
        this.flurryRival = r;
        this.emit('dodge', s.x, s.y + 1, s.z, 2);
      }
      this.engaged();
      this.answer(r, MSG.blow, d[2], OUTCOME.slip, 0, s.x, s.y + 1, s.z);
      return true;
    }
    if (dist > reach || Math.abs(dy) > 1.7) return false;
    // (a half-angle of 180° — a spin's — sweeps all round)
    if (d[6] < Math.PI - 0.01 && dist > 0.4 && (dx * Math.sin(b.yaw) + dz * Math.cos(b.yaw)) / dist < Math.cos(d[6])) return false;
    this.engaged();
    if (this.act === 'flurry' || this.invulnerable > 0) {
      this.answer(r, MSG.blow, d[2], OUTCOME.slip, 0, s.x, s.y + 1, s.z);
      return true;
    }
    if (!rush && this.guards(b.x, b.z)) {
      const gx = s.x + Math.sin(s.yaw) * 0.45, gy = s.y + 1.15, gz = s.z + Math.cos(s.yaw) * 0.45;
      if (this.guardT < COMBAT.parryWindow) {
        this.guardT = COMBAT.parryWindow; // (one parry a raise)
        this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 2);
        this.emit('parry', gx, gy, gz);
        this.answer(r, MSG.blow, d[2], OUTCOME.parry, 0, gx, gy, gz);
      } else {
        const k = 3.2 * this.shieldPush() / (dist || 1);
        s.vx = dx * k; s.vz = dz * k;
        this.emit('block', gx, gy, gz);
        this.wearShield();
        this.answer(r, MSG.blow, d[2], OUTCOME.block, 0, gx, gy, gz);
      }
      return true;
    }
    const q = pvpQuarters(d[3]);
    this.hurt(q, b.x, b.z, Math.min(d[4], 10));
    const down = this.health <= 0;
    if (down) this.downBy = r.id;
    this.answer(r, MSG.blow, d[2], down ? OUTCOME.down : OUTCOME.hit, q, s.x, s.y + 1.1, s.z);
    return true;
  }

  /** Rival r's shot (SHOT d), loosed here from where it left that player's bow or hand: it strikes only the hero. */
  private rivalShot(r: Rival, d: Float64Array): void {
    const p = this.freeShot();
    if (!p) return;
    const sp = Math.hypot(d[6], d[7], d[8]), k = sp > 120 ? 120 / sp : 1;
    const arrow = d[10] < 0;
    this.launch(p, d[3], d[4], d[5], d[6] * k, d[7] * k, d[8] * k, arrow, false, Math.min(d[9], 100), -1);
    p.from = r.id;
    p.ref = d[2];
    if (!arrow) { p.weapon = Math.max(0, Math.min(WEAPONS.length - 1, d[10] | 0)); p.wear = 1; }
    this.emit(arrow ? 'fire' : 'throw', d[3], d[4], d[5], arrow ? 0 : p.weapon);
  }

  /** Another player's shot p against the hero (its head, then its body): slipped, guarded, parried, or it strikes. */
  private rivalShotHits(p: Projectile): void {
    const s = this.hero;
    const hy = s.y - PLAYER.groundBias + HERO_HEAD.y;
    const head = segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, s.x, hy, s.z, s.x, hy, s.z, HERO_HEAD.r + (p.arrow ? 0.02 : 0.22));
    const hit = head >= 0 ? head
      : segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, s.x, s.y + 0.35, s.z, s.x, s.y + 1.55, s.z, p.arrow ? 0.32 : 0.45);
    if (hit < 0 || this.act === 'dodge' || this.act === 'flurry') return; // (slipped: it flies on)
    const r = this.rivalById(p.from);
    this.engaged();
    if (this.guards(p.px - p.vx, p.pz - p.vz)) {
      if (this.guardT < COMBAT.parryWindow && p.arrow) {
        // a Perfect Guard sends it back at its archer — out to the others too, to fly in their worlds
        if (r) {
          rivalJoint(r, J.chest, P3);
          aimAt(p.x, p.y, p.z, P3[0], P3[1], P3[2], FOE.arrowSpeed * 1.3, COMBAT.arrowGravity, V3);
          p.vx = V3[0]; p.vy = V3[1]; p.vz = V3[2];
        } else { p.vx = -p.vx; p.vy = Math.abs(p.vy); p.vz = -p.vz; }
        this.answer(r, MSG.shot, p.ref, OUTCOME.parry, 0, p.x, p.y, p.z);
        p.from = null; p.friendly = true; p.shooter = -1;
        p.damage = COMBAT.arrowDamage * 1.5;
        this.guardT = COMBAT.parryWindow;
        this.emit('parry', p.x, p.y, p.z);
        this.sendShot(p);
      } else {
        p.live = false;
        this.emit('block', p.x, p.y, p.z);
        this.wearShield();
        this.answer(r, MSG.shot, p.ref, OUTCOME.block, 0, p.x, p.y, p.z);
      }
      return;
    }
    p.live = false;
    if (this.invulnerable > 0) { this.answer(r, MSG.shot, p.ref, OUTCOME.slip, 0, p.x, p.y, p.z); return; }
    const crit = head >= 0 && p.arrow;
    const q = pvpQuarters(p.damage * (crit ? COMBAT.headshot : 1));
    this.hurt(q, r ? r.body.x : p.px - p.vx, r ? r.body.z : p.pz - p.vz, p.arrow ? 1.5 : 4.5);
    const down = this.health <= 0;
    if (down && r) this.downBy = r.id;
    this.answer(r, MSG.shot, p.ref, down ? OUTCOME.down : crit ? OUTCOME.crit : OUTCOME.hit, q, p.x, p.y, p.z);
  }

  /** The hero's shot p meets another player where this one sees it: the spark (that player's world settles the harm). */
  private shotMeetsRivals(p: Projectile): void {
    for (const r of this.rivals) {
      if (r.pose.act === 'dead') continue;
      rivalJoint(r, J.head, P3);
      const head = segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, P3[0], P3[1], P3[2], P3[0], P3[1], P3[2], HERO_HEAD.r + (p.arrow ? 0.02 : 0.22));
      rivalJoint(r, J.pelvis, P3B);
      rivalJoint(r, J.chest, V3);
      const hit = head >= 0 ? head
        : segmentsMeet(p.px, p.py, p.pz, p.x, p.y, p.z, P3B[0], P3B[1] - 0.25, P3B[2], V3[0], V3[1], V3[2], p.arrow ? 0.3 : 0.45);
      if (hit < 0) continue;
      r.hitShot = p.ref;
      this.emit(rivalGuards(r, p.px - p.vx, p.pz - p.vz) ? 'block' : 'hit', p.x, p.y, p.z, 0);
      this.engaged();
      this.hitStop = Math.max(this.hitStop, COMBAT.hitStop * 0.6);
      if (p.arrow) p.live = false;
      else this.settleWeapon(p, -p.vx * 0.15, -p.vz * 0.15);
      return;
    }
  }

  /** What came of the hero's blow or shot on rival r (RESULT d), shown along with that player's flinch. */
  private rivalResult(r: Rival, d: Float64Array): void {
    const kind = d[2], ref = d[3], outcome = d[4], x = d[6], y = d[7], z = d[8];
    // (where this player saw it meet them, the spark is shown already)
    const seen = kind === MSG.blow ? r.hitSwing === ref : r.hitShot === ref;
    this.engaged();
    switch (outcome) {
      case OUTCOME.hit: case OUTCOME.down: case OUTCOME.crit:
        if (outcome === OUTCOME.crit) this.emit('crit', x, y, z, d[5]);
        else if (!seen) this.emit('hit', x, y, z, d[5]);
        if (outcome === OUTCOME.down) this.emit('defeat', r.body.x, r.body.y + 1, r.body.z, this.rivals.indexOf(r));
        break;
      case OUTCOME.block:
        if (!seen) this.emit('block', x, y, z);
        break;
      case OUTCOME.parry:
        if (kind === MSG.blow) this.stagger(r.body.x, r.body.z, x, y, z);
        break;
    }
  }

  /** Another player's Perfect Guard turned the hero's blow: dazed a moment, knocked back from (fromX, fromZ). */
  private stagger(fromX: number, fromZ: number, x: number, y: number, z: number): void {
    const s = this.hero;
    if (this.act === 'dead') return;
    if (s.surfing) stopSurf(s);
    const dx = s.x - fromX, dz = s.z - fromZ, d = Math.hypot(dx, dz) || 1;
    launchHero(s, (dx / d) * 3.5, 2.6, (dz / d) * 3.5);
    this.setAct('stun');
    this.charge = 0; this.draw = 0; this.queued = false; this.flurryWait = 0;
    this.emit('parried', x, y, z);
  }

  /** What came of rival r's BLOW / SHOT `ref`, back to that player (stamped with this player's clock). */
  private answer(r: Rival | null, kind: number, ref: number, outcome: number, q: number, x: number, y: number, z: number): void {
    if (!r) return;
    const d = this.post(r.id, 9);
    d[0] = MSG.result; d[1] = Math.round(this.netTime); d[2] = kind; d[3] = ref; d[4] = outcome; d[5] = q;
    d[6] = round(x, 1000); d[7] = round(y, 1000); d[8] = round(z, 1000);
  }

  /** Shot p goes out to the other players too: their worlds loose it from here, to strike their heroes. */
  private sendShot(p: Projectile): void {
    p.ref = ++this.shotId;
    if (!this.rivals.length) return;
    const d = this.post(null, 11);
    d[0] = MSG.shot; d[1] = Math.round(this.netTime); d[2] = p.ref;
    d[3] = round(p.x, 1000); d[4] = round(p.y, 1000); d[5] = round(p.z, 1000);
    d[6] = round(p.vx, 1000); d[7] = round(p.vy, 1000); d[8] = round(p.vz, 1000);
    d[9] = round(p.damage, 100); d[10] = p.arrow ? -1 : p.weapon;
  }

  /** A message to send after this tick (to one player, or null: all), its `n` numbers for the caller to fill. */
  private post(to: string | null, n: number): number[] {
    const o = this.outCount < MAX_OUT ? this.outbox[this.outCount++] : null;
    const d = o ? o.d : this.spare; // (none free: written nowhere)
    if (o) o.to = to;
    d.length = n;
    return d;
  }

  // ---- what lies on the ground --------------------------------------------------------------------
  private drop(kind: Pickup['kind'], weapon: number, wear: number, count: number, x: number, y: number, z: number): number {
    let slot = -1, oldest = -1;
    for (let i = 0; i < this.pickups.length && slot < 0; i++) {
      const p = this.pickups[i];
      if (!p.live) slot = i;
      // (else the oldest lying item gives way, never the sword)
      else if (!(p.kind === 'weapon' && p.weapon === WEAPON.master) && (oldest < 0 || p.age > this.pickups[oldest].age)) oldest = i;
    }
    if (slot < 0) slot = oldest;
    if (slot < 0) return -1;
    const p = this.pickups[slot];
    p.live = true; p.kind = kind; p.weapon = weapon; p.wear = wear; p.count = count;
    p.x = x; p.y = y; p.z = z; p.vy = 2.5; p.age = 0; p.yaw = (x * 13.1 + z * 7.7) % (2 * Math.PI);
    return slot;
  }

  /** A cleared camp leaves a shield or a bow by its fire, by the camp's hash. */
  private spoils(c: Camp): void {
    const [cat, kind] = SPOILS[(Math.imul(c.key | 0, 2654435761) >>> 0) % SPOILS.length];
    const wear = (cat === 'shield' ? SHIELDS : BOWS)[kind].durability;
    this.drop(cat, kind, wear, 0, c.x + 0.7, c.y + 0.9, c.z - 0.4);
  }

  /** The gear (a weapon, shield or bow) lying nearest the hero within reach (−1: none), for the action key. */
  nearestPickup(): number {
    const s = this.hero;
    let best = -1, bd = 1.9;
    for (let i = 0; i < this.pickups.length; i++) {
      const p = this.pickups[i];
      if (!p.live || (p.kind !== 'weapon' && p.kind !== 'shield' && p.kind !== 'bow')) continue;
      const d = Math.hypot(p.x - s.x, p.z - s.z);
      if (d < bd && Math.abs(p.y - s.y) < 1.6) { bd = d; best = i; }
    }
    return best;
  }

  /** Whether the hero has room for the gear lying at pickups[k] (a weapon always: the one in hand makes way). */
  canTake(k: number): boolean {
    const p = this.pickups[k];
    if (!p || !p.live) return false;
    return p.kind === 'shield' ? this.shields.length < COMBAT.shieldSlots : p.kind === 'bow' ? this.bows.length < COMBAT.bowSlots : true;
  }

  private stepPickups(dt: number, inp: CombatInput): void {
    const s = this.hero;
    for (let i = 0; i < this.pickups.length; i++) {
      const p = this.pickups[i];
      if (!p.live) continue;
      p.age += dt;
      // fall to the ground (and lie there: no more queries once it has come to rest)
      if (p.vy !== 0) {
        const g = elevation(p.x, p.z, this.seed, 0) + 0.06;
        p.vy -= PLAYER.gravity * dt;
        p.y += p.vy * dt;
        if (p.y <= g) { p.y = g; p.vy = 0; }
      }
      const d = Math.hypot(p.x - s.x, p.z - s.z), dy = Math.abs(p.y - s.y);
      const gear = p.kind === 'weapon' || p.kind === 'shield' || p.kind === 'bow';
      if (!gear && d < 1.3 && dy < 1.6 && this.act !== 'dead') {
        if (p.kind === 'food' && this.food[p.weapon] < COMBAT.maxFood) {
          this.food[p.weapon] = Math.min(COMBAT.maxFood, this.food[p.weapon] + (p.count || 1));
          p.live = false;
          this.emit('pickup', p.x, p.y, p.z, ITEM_CODE.food + p.weapon);
        } else if (p.kind === 'arrows' && this.arrows < COMBAT.maxArrows) {
          this.arrows = Math.min(COMBAT.maxArrows, this.arrows + p.count);
          p.live = false;
          this.emit('pickup', p.x, p.y, p.z, -1);
        } else if (p.kind === 'heart' && this.health < this.hearts * 4) {
          this.health = Math.min(this.hearts * 4, this.health + p.count);
          p.live = false;
          this.emit('pickup', p.x, p.y, p.z, -2);
        }
      }
      if (p.kind === 'weapon' && p.weapon === WEAPON.master && p.age > COMBAT.swordReturn && p.live) {
        // the sword finds its way back to its sheath
        p.live = false;
        this.takeWeapon(WEAPON.master, Infinity);
        this.emit('return', s.x, s.y + 1.3, s.z);
      }
      if (!gear && p.age > 90) p.live = false;
      if (gear && !(p.kind === 'weapon' && p.weapon === WEAPON.master) && p.age > 240) p.live = false;
    }
    const actionP = inp.action && !this.prev.action;
    if (actionP && this.act !== 'guard' && !s.surfing && this.act !== 'dead') {
      const k = this.nearestPickup();
      if (k >= 0) {
        const p = this.pickups[k];
        if (!this.canTake(k)) this.emit('empty', s.x, s.y + 1.2, s.z, 3);
        else {
          p.live = false;
          if (p.kind === 'weapon') this.takeWeapon(p.weapon, p.wear);
          else if (p.kind === 'shield') this.takeGear(this.shields, p.weapon, p.wear, 'shield');
          else this.takeGear(this.bows, p.weapon, p.wear, 'bow');
          this.emit('pickup', p.x, p.y, p.z, ITEM_CODE[p.kind as PouchCat] + p.weapon);
        }
      }
    }
  }

  /** Carry a shield or a bow (taken in hand only if none is: the pouch is where it is changed). */
  private takeGear(list: Carried[], kind: number, wear: number, cat: 'shield' | 'bow'): void {
    list.push({ kind, wear });
    if (cat === 'shield' && this.shieldOn < 0) this.shieldOn = list.length - 1;
    if (cat === 'bow' && this.bowOn < 0) this.bowOn = list.length - 1;
  }

  /** Carry a weapon (and take it in hand); with every slot full the one in hand is dropped for it. */
  private takeWeapon(kind: number, wear: number): void {
    const s = this.hero;
    if (kind === WEAPON.master) this.swordAway = -1;
    if (this.weapons.length >= COMBAT.weaponSlots) {
      const c = this.carried();
      if (c && c.kind !== WEAPON.master) {
        this.drop('weapon', c.kind, c.wear, 0, s.x + Math.sin(s.yaw) * 0.6, s.y + 0.5, s.z + Math.cos(s.yaw) * 0.6);
        this.weapons.splice(this.equipped, 1);
      } else {
        const i = this.weapons.findIndex((w) => w.kind !== WEAPON.master);
        const w = this.weapons[i];
        this.drop('weapon', w.kind, w.wear, 0, s.x, s.y + 0.5, s.z);
        this.weapons.splice(i, 1);
      }
    }
    this.weapons.push({ kind, wear });
    this.equipped = this.weapons.length - 1;
  }

  // ---- camps --------------------------------------------------------------------------------------
  /** Wake the camps the hero has come near, put away those it has left (re-listed every 8 m). */
  private stepCamps(): void {
    const s = this.hero, at = this.campsAt;
    if (!this.foesOn) {
      for (const f of this.foes) f.live = false;
      return;
    }
    if (this.campsOn && !(Math.hypot(s.x - at[0], s.z - at[1]) < 8 && at[2] === this.seed)) {
      at[0] = s.x; at[1] = s.z; at[2] = this.seed;
      campsNear(s.x, s.z, FOE.wakeRange, this.seed, this.near);
      for (const c of this.near) {
        let life = this.camps.get(c.key);
        if (!life) { life = { camp: c, awake: false, cleared: false, clearedAt: 0, slots: [] }; this.camps.set(c.key, life); }
        if (life.cleared && this.time - life.clearedAt > FOE.respawn) life.cleared = false;
        if (life.awake || life.cleared) continue;
        this.wakeCamp(life);
      }
    }
    for (const life of this.camps.values()) {
      if (!life.awake) continue;
      const d = Math.hypot(life.camp.x - s.x, life.camp.z - s.z);
      if (d > FOE.sleepRange) this.sleepCamp(life);
    }
  }

  private wakeCamp(life: CampLife): void {
    const c = life.camp;
    life.slots.length = 0;
    for (let k = 0; k < c.kinds.length; k++) {
      const f = this.foes.find((g) => !g.live);
      if (!f) break;
      wakeFoe(f, c, k, this.seed);
      life.slots.push(f.slot);
    }
    life.awake = life.slots.length > 0;
  }

  private sleepCamp(life: CampLife): void {
    for (const i of life.slots) {
      const f = this.foes[i];
      if (f.live && f.camp === life.camp.key) f.live = false;
      if (this.lock === i) this.lock = -1;
    }
    life.slots.length = 0;
    life.awake = false;
  }

  /** The camps awake, for their fires (the renderer). */
  awakeCamps(out: Camp[]): Camp[] {
    out.length = 0;
    for (const life of this.camps.values()) if (life.awake || (life.cleared && this.time - life.clearedAt < 60)) out.push(life.camp);
    return out;
  }

  // ---- the pouch (the inventory screen) -------------------------------------------------------------
  /** Take item `i` of a category in hand (or on the arm); false when there is no such item. */
  equip(cat: PouchCat, i: number): boolean {
    if (this.act === 'dead') return false;
    const c = this.listOf(cat)[i];
    if (!c) return false;
    if (cat === 'weapon') this.equipped = i; else if (cat === 'shield') this.shieldOn = i; else this.bowOn = i;
    const s = this.hero;
    this.emit('equip', s.x, s.y + 1, s.z, ITEM_CODE[cat] + c.kind);
    return true;
  }

  /** Put away what a category has in hand; false when it has nothing. */
  unequip(cat: PouchCat): boolean {
    if (this.act === 'dead') return false;
    const on = cat === 'weapon' ? this.equipped : cat === 'shield' ? this.shieldOn : this.bowOn;
    if (on < 0) return false;
    if (cat === 'weapon') this.equipped = -1; else if (cat === 'shield') this.shieldOn = -1; else this.bowOn = -1;
    const s = this.hero;
    this.emit('equip', s.x, s.y + 1, s.z, -1);
    return true;
  }

  /** Drop item `i` of a category in front of the hero (the hero's own gear — what never wears out — is kept). */
  discard(cat: PouchCat, i: number): boolean {
    if (this.act === 'dead') return false;
    const list = this.listOf(cat), c = list[i];
    if (!c || c.wear === Infinity) return false;
    const s = this.hero;
    const on = cat === 'weapon' ? this.equipped : cat === 'shield' ? this.shieldOn : this.bowOn;
    list.splice(i, 1);
    const own = cat === 'weapon' ? WEAPON.master : cat === 'shield' ? SHIELD.hylian : BOW.traveler;
    // (the one in hand gone, the hero's own takes its place, else the first)
    let now = on;
    if (on > i) now = on - 1;
    else if (on === i) {
      const k = list.findIndex((q) => q.kind === own);
      now = k >= 0 ? k : list.length ? 0 : -1;
    }
    if (cat === 'weapon') this.equipped = now; else if (cat === 'shield') this.shieldOn = now; else this.bowOn = now;
    this.drop(cat, c.kind, c.wear, 0, s.x + Math.sin(s.yaw) * 0.9, s.y + 0.5, s.z + Math.cos(s.yaw) * 0.9);
    this.emit('equip', s.x, s.y + 1, s.z, -1);
    return true;
  }

  /** Eat one of food `i`: its hearts come back. False when there is none, or the hearts are full. */
  eat(i: number): boolean {
    if (this.act === 'dead' || i < 0 || i >= FOODS.length || this.food[i] <= 0) return false;
    const max = this.hearts * 4;
    if (Math.ceil(this.health) >= max) return false;
    this.food[i]--;
    this.health = Math.min(max, this.health + FOODS[i].heal);
    const s = this.hero;
    this.emit('eat', s.x, s.y + 1.3, s.z, ITEM_CODE.food + i);
    return true;
  }

  /**
   * A full pouch, to try the inventory (?kit=1): one of every weapon, shield and bow, and some of every
   * food — and the hero at three hearts, so there is something to eat for.
   */
  stock(): void {
    for (let k = 0; k < WEAPONS.length; k++) if (!this.weapons.some((c) => c.kind === k)) this.weapons.push({ kind: k, wear: WEAPONS[k].durability });
    for (let k = 0; k < SHIELDS.length; k++) if (!this.shields.some((c) => c.kind === k)) this.shields.push({ kind: k, wear: SHIELDS[k].durability });
    for (let k = 0; k < BOWS.length; k++) if (!this.bows.some((c) => c.kind === k)) this.bows.push({ kind: k, wear: BOWS[k].durability });
    for (let k = 0; k < FOODS.length; k++) this.food[k] = Math.max(this.food[k], 5);
    this.health = Math.min(this.health, 12);
  }

  /** The hands let go of what they were doing as the pouch opens: a raised shield, a drawn bow, a charging spin. */
  cancelActs(): void {
    if (this.act === 'guard' || this.act === 'aim' || this.act === 'charge') this.setAct('idle');
    this.charge = 0;
    this.draw = 0;
  }

  /** The rig's pose for what is carried now, without a step (the world stands still behind the pouch). */
  refreshPose(inp: CombatInput): void {
    this.fillPose(inp);
  }

  private listOf(cat: PouchCat): Carried[] {
    return cat === 'weapon' ? this.weapons : cat === 'shield' ? this.shields : this.bows;
  }

  // ---- the engine's side --------------------------------------------------------------------------
  /** Back on its feet with full hearts (a respawn); foes forget it. */
  revive(): void {
    if (this.stone) this.releaseStone(0, 0, 0);
    this.pushRock = null;
    this.health = this.hearts * 4;
    this.setAct('idle');
    this.invulnerable = 2;
    this.downFor = 0;
    this.armed = false;
    this.lock = -1;
    this.lockRival = null;
    this.flurryWait = 0;
    this.flurryRival = null;
    this.downBy = null;
    this.hitStop = 0;
    // (blows and shots on their way to where the hero was are past; what came of its own still counts)
    for (const m of this.inbox) if (m.d[0] !== MSG.result) m.live = false;
    for (const f of this.foes) { if (f.live) { f.roused = false; f.aware = 0; if (f.mind !== 'dead') f.mind = 'idle'; } }
    if (this.swordAway >= 0 || !this.weapons.some((w) => w.kind === WEAPON.master)) {
      if (this.swordAway >= 0) this.pickups[this.swordAway].live = false;
      this.takeWeapon(WEAPON.master, Infinity);
    }
  }

  /** A new world (seed): every foe, shot and item gone, the camps forgotten. */
  reset(seed: number): void {
    this.seed = seed;
    this.stone = null;
    this.pushRock = null;
    for (const f of this.foes) f.live = false;
    for (const p of this.shots) p.live = false;
    for (const p of this.pickups) p.live = false;
    for (const m of this.inbox) m.live = false;
    this.camps.clear();
    this.campsAt[0] = NaN;
    this.lock = -1;
    this.lockRival = null;
    this.swordAway = -1;
    if (!this.weapons.some((w) => w.kind === WEAPON.master)) this.takeWeapon(WEAPON.master, Infinity);
  }

  /** The foe the shield is locked onto, if any. */
  locked(): Foe | null {
    return this.lock >= 0 ? this.foes[this.lock] : null;
  }

  /** Wake a foe of `kind` (FOE_KINDS) standing at (x, z) facing `yaw`, of no camp (tests, scenes); null when none is free. */
  spawnFoe(kind: number, x: number, z: number, yaw: number, archer = false): Foe | null {
    const f = this.foes.find((g) => !g.live);
    if (!f) return null;
    // (a camp of one, round no fire: the foe stands where it is put)
    wakeFoe(f, { key: -2 - f.slot, x, y: 0, z, kinds: [kind], archer: archer ? 0 : -1 }, 0, this.seed, true);
    f.body.yaw = yaw;
    return f;
  }
}

/** smoothstep */
function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

export { WRAP as wrapAngle };
