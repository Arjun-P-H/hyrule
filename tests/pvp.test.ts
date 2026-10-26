import { describe, it, expect } from 'bun:test';
import { elevation, slope, waterLevel } from '../src/lib/terrainMath';
import { COMBAT, DEFAULT_SEED, PLAYER, SPAWN } from '../src/lib/constants';
import { createHeroState, type HeroState } from '../src/lib/hero/locomotion';
import { J } from '../src/lib/hero/rig';
import { CombatWorld, createCombatInput, type CombatEventType, type CombatInput } from '../src/lib/combat/world';
import { HERO_HEAD, MSG, OUTCOME, pvpQuarters, rivalJoint, type Rival } from '../src/lib/combat/pvp';
import { MOVES, COMBO } from '../src/lib/combat/moves';
import { WEAPON, WEAPONS } from '../src/lib/combat/weapons';
import { RemoteHeroes, SEND_INTERVAL, encodeSnapshot } from '../src/lib/net/remoteHeroes';
import type { NetLink } from '../src/lib/net/link';
import { plantInCell } from '../src/lib/forest/placement';
import { CELL } from '../src/lib/forest/species';

/**
 * Players against each other (src/lib/combat/pvp.ts): two players' combat
 * worlds joined as the game joins them — each one's snapshots played back by
 * the other's RemoteHeroes (120 ms late), the combat messages through a relay
 * with a latency, the two clocks apart — on the real terrain.
 */

const SEED = DEFAULT_SEED;
const DT = 1 / 60;
const FULL = COMBAT.hearts * 4;

function findSpot(pred: (x: number, z: number) => boolean, step = 3, radius = 2500): [number, number] {
  for (let r = 0; r <= radius; r += step) {
    for (let a = 0; a < 64; a++) {
      const x = SPAWN.x + r * Math.cos((a / 64) * Math.PI * 2);
      const z = SPAWN.z + r * Math.sin((a / 64) * Math.PI * 2);
      if (pred(x, z)) return [x, z];
      if (r === 0) break;
    }
  }
  throw new Error('no spot found');
}

function clearOfTrees(x: number, z: number, r: number): boolean {
  for (let iz = Math.floor((z - r) / CELL); iz <= Math.floor((z + r) / CELL); iz++) {
    for (let ix = Math.floor((x - r) / CELL); ix <= Math.floor((x + r) / CELL); ix++) {
      const p = plantInCell(ix, iz, SEED);
      if (p && p.species !== 'bush' && Math.hypot(p.x - x, p.z - z) < r) return false;
    }
  }
  return true;
}

// level, dry, open ground (as tests/combat.test.ts's arena)
const FLAT = findSpot((x, z) => slope(x, z, SEED) < 0.06 && elevation(x, z, SEED, 0) > waterLevel(x, z, SEED) + 2.9
  && [0, 1, 2, 3, 4, 5, 6, 7].every((k) => slope(x + 12 * Math.cos(k * Math.PI / 4), z + 12 * Math.sin(k * Math.PI / 4), SEED) < 0.08)
  && clearOfTrees(x, z, 30), 4);

/** The view from behind hero s, aimed at world point (x, y, z). */
function lookAt(inp: CombatInput, s: HeroState, x: number, y: number, z: number): void {
  const yaw = Math.atan2(x - s.x, z - s.z);
  inp.camYaw = yaw;
  inp.eyeX = s.x - Math.sin(yaw) * 2; inp.eyeY = s.y + 1.7; inp.eyeZ = s.z - Math.cos(yaw) * 2;
  const dx = x - inp.eyeX, dy = y - inp.eyeY, dz = z - inp.eyeZ, l = Math.hypot(dx, dy, dz);
  inp.dirX = dx / l; inp.dirY = dy / l; inp.dirZ = dz / l;
  inp.camPitch = Math.asin(inp.dirY);
}

type Logged = { t: number; type: CombatEventType; a: number };

/** One player: its world and hero, the other as it sees it, and what its session has received. */
type Side = {
  readonly id: string;
  readonly w: CombatWorld;
  readonly hero: HeroState;
  readonly inp: CombatInput;
  readonly remotes: RemoteHeroes;
  readonly snaps: Array<[string, number, number[]]>;
  readonly combat: Array<[string, number, number[]]>;
  readonly link: NetLink;
  /** Its clock minus the shared one (ms). */
  readonly clock: number;
  sentAt: number;
  /** Its world's events, and the messages it sent, with the shared time (ms). */
  readonly log: Logged[];
  readonly sent: Array<{ t: number; d: number[] }>;
};

function side(id: string, other: string, x: number, z: number, yaw: number, clock: number): Side {
  const hero = createHeroState(x, z, yaw, SEED);
  const w = new CombatWorld(hero, SEED);
  w.campsOn = false;
  const inp = createCombatInput();
  lookAt(inp, hero, x + Math.sin(yaw) * 5, hero.y + 1.2, z + Math.cos(yaw) * 5);
  const snaps: Side['snaps'] = [], combat: Side['combat'] = [];
  const peers = new Map([[other, 0]]);
  const link: NetLink = {
    inWorld: () => true,
    peers: () => peers,
    drain: (take) => { for (const [i, at, d] of snaps) take(i, at, d); snaps.length = 0; },
    sendState: () => {},
    drainCombat: (take) => { for (const [i, at, d] of combat) take(i, at, d); combat.length = 0; },
    sendCombat: () => {},
    peerName: (i) => i,
    takeClock: () => null,
    sendClock: () => {},
    keepsClock: () => false,
  };
  return { id, w, hero, inp, remotes: new RemoteHeroes(), snaps, combat, link, clock, sentAt: -Infinity, log: [], sent: [] };
}

/**
 * Two players a few metres apart on the flat, facing each other (a looks along +x), stepped together
 * at 60 Hz as the engine steps each: the other played back, its messages taken, the step, its own
 * messages and (15 times a second) its snapshot out through the relay.
 */
class Duel {
  now = 0;
  readonly a: Side;
  readonly b: Side;
  private wire: Array<{ due: number; to: Side; from: string; kind: 's' | 'c'; d: number[] }> = [];

  constructor(readonly latency = 40, gap = 1.6) {
    this.a = side('a', 'b', FLAT[0], FLAT[1], Math.PI / 2, 0);
    this.b = side('b', 'a', FLAT[0] + gap, FLAT[1], -Math.PI / 2, 7777);
  }

  /** The other player as `s` sees it. */
  rival(s: Side): Rival {
    return s.remotes.list()[0];
  }

  tick(): void {
    this.now += 1000 / 60;
    for (let i = 0; i < this.wire.length; i++) {
      const p = this.wire[i];
      if (p.due > this.now) continue;
      (p.kind === 's' ? p.to.snaps : p.to.combat).push([p.from, this.now + p.to.clock, p.d]);
      this.wire.splice(i--, 1);
    }
    for (const [me, other] of [[this.a, this.b], [this.b, this.a]] as const) {
      const local = this.now + me.clock;
      me.remotes.update(me.link, local, DT);
      me.w.rivals = me.remotes.list();
      me.link.drainCombat((id, _at, d) => me.w.receive(id, d));
      me.w.netTime = local;
      me.w.step(DT, me.inp);
      for (let k = 0; k < me.w.eventCount; k++) {
        const e = me.w.events[k];
        me.log.push({ t: this.now, type: e.type, a: e.a });
      }
      for (let k = 0; k < me.w.outCount; k++) {
        const o = me.w.outbox[k];
        me.sent.push({ t: this.now, d: o.d.slice() });
        if (o.to === null || o.to === other.id) this.wire.push({ due: this.now + this.latency, to: other, from: me.id, kind: 'c', d: o.d.slice() });
      }
      if (local - me.sentAt >= SEND_INTERVAL) {
        me.sentAt = local;
        const d = encodeSnapshot(me.hero, local, [], me.w.pose, me.w.health, me.w.hearts);
        this.wire.push({ due: this.now + this.latency, to: other, from: me.id, kind: 's', d });
      }
    }
  }

  run(seconds: number, each?: () => void): void {
    for (let i = Math.round(seconds / DT); i > 0; i--) {
      each?.();
      this.tick();
    }
  }

  /** Press `key` on side s for one tick. */
  press(s: Side, key: 'attack' | 'guard' | 'jump'): void {
    s.inp[key] = true;
    this.tick();
    s.inp[key] = false;
  }
}

const has = (s: Side, type: CombatEventType) => s.log.some((e) => e.type === type);
const firstAt = (s: Side, type: CombatEventType) => s.log.find((e) => e.type === type)?.t ?? NaN;
const swordBlow = () => pvpQuarters(WEAPONS[WEAPON.master].damage * MOVES[COMBO[0]].damage);

/** A duel warmed up: each player has seen the other a while. */
function duel(latency = 40, gap = 1.6): Duel {
  const d = new Duel(latency, gap);
  d.run(0.6);
  return d;
}

describe('PvP — the sword', () => {
  it('a slash from one player costs the other a heart, settled when its view of the attacker shows the blow', () => {
    const d = duel();
    expect(d.rival(d.a)).toBeDefined();
    expect(d.rival(d.b)).toBeDefined();
    d.press(d.a, 'attack');
    d.run(1.2);
    expect(swordBlow()).toBe(4); // (the Master Sword's slash: one heart)
    expect(d.b.w.health).toBe(FULL - swordBlow());
    expect(has(d.b, 'hurt')).toBe(true);
    // the blow went out once, as the swing reached its window; b took it no sooner than its view of a
    // (the latency, rounded up to a tick, and 120 ms of playback) showed it
    const blows = d.a.sent.filter((m) => m.d[0] === MSG.blow);
    expect(blows.length).toBe(1);
    const delay = firstAt(d.b, 'hurt') - blows[0].t;
    expect(delay).toBeGreaterThanOrEqual(d.latency + 120);
    expect(delay).toBeLessThan(d.latency + 120 + 100);
    // a saw its blade meet b at once (the spark), and heard back what came of it
    expect(firstAt(d.a, 'hit')).toBeLessThanOrEqual(blows[0].t + 1000 / 60);
    const result = d.b.sent.find((m) => m.d[0] === MSG.result)!;
    expect(result.d[4]).toBe(OUTCOME.hit);
    expect(result.d[5]).toBe(swordBlow());
    // … and b's hearts reach a in b's snapshots
    expect(d.rival(d.a).health).toBe(FULL - swordBlow());
    // (a's own hearts are untouched)
    expect(d.a.w.health).toBe(FULL);
  });

  it('a blow out of reach, or behind the swing, does no harm', () => {
    const far = duel(40, 4.5);
    far.press(far.a, 'attack');
    far.run(1.2);
    expect(far.b.w.health).toBe(FULL);
    expect(has(far.a, 'hit')).toBe(false);
    // a turned away from b
    const away = duel();
    away.a.hero.yaw = -Math.PI / 2;
    away.a.inp.camYaw = -Math.PI / 2;
    away.run(0.3);
    away.press(away.a, 'attack');
    away.run(1.2);
    expect(away.b.w.health).toBe(FULL);
  });

  it('a spin attack sweeps a player behind the swinger too', () => {
    const d = duel();
    d.a.hero.yaw = -Math.PI / 2; // (b behind a)
    d.a.inp.camYaw = -Math.PI / 2;
    d.a.inp.attack = true;
    d.run(1.3); // held: the spin charges
    d.a.inp.attack = false;
    d.run(1.4);
    expect(has(d.a, 'spin')).toBe(true);
    expect(d.b.w.health).toBe(FULL - pvpQuarters(WEAPONS[WEAPON.master].damage * MOVES[4].damage));
  });

  it('a fight with another player holds the hearts back (guarded blows too); out of it they return', () => {
    const d = duel();
    d.press(d.a, 'attack');
    d.run(1.2);
    expect(d.b.w.health).toBe(FULL - swordBlow());
    // b guards, a swings at it now and then: still a fight, well past the regen delay
    d.b.inp.guard = true;
    for (let i = 0; i < 4; i++) { d.run(2.6); d.press(d.a, 'attack'); }
    d.run(1.0);
    expect(d.b.log.filter((e) => e.type === 'block').length).toBe(4);
    expect(d.b.w.health).toBe(FULL - swordBlow());
    // left alone, they come back
    d.run(COMBAT.regenDelay + 2);
    expect(d.b.w.health).toBeGreaterThan(FULL - swordBlow());
  });

  it('felled by another player: Game Over names who did it, and the attacker hears of it', () => {
    const d = duel();
    d.b.w.health = 3;
    d.run(0.3); // (b's snapshots carry it)
    d.press(d.a, 'attack');
    d.run(1.5);
    expect(d.b.w.act).toBe('dead');
    expect(d.b.w.downBy).toBe('a');
    const defeat = d.a.log.find((e) => e.type === 'defeat');
    expect(defeat?.a).toBe(0); // (b: a's first rival)
    // a fallen player is no target: not locked onto, no spark
    expect(d.rival(d.a).pose.act).toBe('dead');
    d.a.inp.guard = true;
    d.run(0.2);
    expect(d.a.w.lockRival).toBeNull();
    d.a.inp.guard = false;
    // back on its feet with all its hearts, and no killer
    d.b.w.revive();
    expect(d.b.w.downBy).toBeNull();
    expect(d.b.w.health).toBe(FULL);
  });
});

describe('PvP — the shield and dodges', () => {
  it('the shield up toward the attacker guards its blow', () => {
    const d = duel();
    d.b.inp.guard = true;
    d.run(1.0); // (raised long since: no Perfect Guard)
    d.press(d.a, 'attack');
    d.run(1.2);
    expect(d.b.w.health).toBe(FULL);
    expect(has(d.b, 'block')).toBe(true);
    expect(has(d.b, 'parry')).toBe(false);
    // a saw b's shield up and met it (a clank, not a hit)
    expect(has(d.a, 'block')).toBe(true);
    expect(d.b.sent.find((m) => m.d[0] === MSG.result)?.d[4]).toBe(OUTCOME.block);
  });

  it('raised just as the blow lands: a Perfect Guard, which dazes the attacker', () => {
    const d = duel();
    let raised = false;
    d.press(d.a, 'attack');
    d.run(1.5, () => {
      // b reacts to what it sees: a's swing coming
      if (!raised && d.rival(d.b).pose.act === 'swing') { d.b.inp.guard = true; raised = true; }
    });
    expect(raised).toBe(true);
    expect(has(d.b, 'parry')).toBe(true);
    expect(d.b.w.health).toBe(FULL);
    // a is dazed when its view of b shows the parry (after b's, by the latency and the playback)
    expect(has(d.a, 'parried')).toBe(true);
    expect(firstAt(d.a, 'parried') - firstAt(d.b, 'parry')).toBeGreaterThanOrEqual(d.latency + 120);
    expect(d.a.w.health).toBe(FULL);
  });

  it('a dazed attacker cannot act for a moment', () => {
    const d = duel();
    let raised = false;
    d.press(d.a, 'attack');
    let stunnedFor = 0;
    d.run(2.5, () => {
      if (!raised && d.rival(d.b).pose.act === 'swing') { d.b.inp.guard = true; raised = true; }
      if (d.a.w.act === 'stun') stunnedFor += DT;
    });
    expect(stunnedFor).toBeGreaterThan(COMBAT.parriedStun - 0.05);
    expect(stunnedFor).toBeLessThan(COMBAT.parriedStun + 0.1);
    // dazed, a showed it to b (its pose), and could not swing
    expect(d.a.sent.filter((m) => m.d[0] === MSG.blow).length).toBe(1);
  });

  it('a side hop just as the blow comes slips it, and opens a Flurry Rush on the attacker that its shield cannot stop (the world not slowed)', () => {
    const d = duel();
    // both locked on, shields up: a swings from behind its shield
    d.a.inp.guard = true;
    d.b.inp.guard = true;
    d.run(0.6);
    expect(d.b.w.lockRival).toBe(d.rival(d.b));
    d.press(d.a, 'attack');
    let hopped = false;
    d.run(0.6, () => {
      if (!hopped && d.rival(d.b).pose.act === 'swing') {
        hopped = true;
        d.b.inp.strafe = 1;
        d.b.inp.jump = true;
      } else if (hopped) { d.b.inp.jump = false; }
    });
    d.b.inp.strafe = 0;
    expect(d.b.log.some((e) => e.type === 'dodge' && e.a === 2)).toBe(true);
    expect(d.b.w.health).toBe(FULL);
    expect(d.b.w.flurryWait).toBeGreaterThan(0);
    expect(d.b.w.worldScale).toBe(1); // (another player is never slowed: nor is the world round b)
    // b rushes in: its cuts strike a through its raised shield
    d.b.inp.guard = false;
    d.press(d.b, 'attack');
    expect(d.b.w.act).toBe('flurry');
    d.run(2.0);
    expect(d.a.w.act).toBe('guard');
    expect(d.a.w.health).toBeLessThanOrEqual(FULL - pvpQuarters(WEAPONS[WEAPON.master].damage));
    // (what a answered: hits, and slips while it was still reeling — never a block)
    const answers = d.a.sent.filter((m) => m.d[0] === MSG.result).map((m) => m.d[4]);
    expect(answers).toContain(OUTCOME.hit);
    expect(answers).not.toContain(OUTCOME.block);
    const rush = d.b.sent.filter((m) => m.d[0] === MSG.blow);
    expect(rush.length).toBe(COMBAT.flurryHits);
    expect(rush.every((m) => m.d[8] === 1)).toBe(true);
  });

  it('the shield locks onto another player, and the hero faces it', () => {
    const d = duel();
    d.a.hero.yaw = 0; // (turned aside; b still in view)
    d.a.inp.guard = true;
    d.run(0.8);
    expect(d.a.w.lockRival).toBe(d.rival(d.a));
    expect(Math.abs(d.a.hero.yaw - Math.PI / 2)).toBeLessThan(0.1);
    expect(d.a.w.lockedBody()).toBe(d.rival(d.a).body);
  });
});

describe('PvP — arrows and throws', () => {
  /** a draws the bow at b's chest (as a sees it) and looses at full draw. */
  function loose(d: Duel): void {
    const r = d.rival(d.a);
    const P = new Float64Array(3);
    rivalJoint(r, J.chest, P);
    lookAt(d.a.inp, d.a.hero, P[0], P[1], P[2]);
    d.a.inp.aim = true;
    d.run(0.9);
    rivalJoint(r, J.chest, P);
    lookAt(d.a.inp, d.a.hero, P[0], P[1], P[2]);
    d.press(d.a, 'attack');
  }

  it('an arrow flies in the other player\'s world too, from where it left the bow, and strikes there', () => {
    const d = duel(40, 6);
    loose(d);
    let seen = false;
    d.run(1.0, () => { seen ||= d.b.w.shots.some((p) => p.live && p.from === 'a'); });
    expect(seen).toBe(true);
    expect(d.b.w.health).toBe(FULL - pvpQuarters(COMBAT.arrowDamage));
    const shot = d.a.sent.find((m) => m.d[0] === MSG.shot)!;
    expect(shot.d[10]).toBe(-1);
    expect(d.b.sent.find((m) => m.d[0] === MSG.result)?.d[4]).toBe(OUTCOME.hit);
    // a's own arrow went into b where a saw it (no harm there)
    expect(has(d.a, 'hit')).toBe(true);
  });

  it('a shot through the head counts twice', () => {
    const d = duel(40, 6);
    const s = d.b.hero, r = d.rival(d.b);
    const hy = s.y - PLAYER.groundBias + HERO_HEAD.y;
    // (as a's world sent it: from 5 m off along −x, level with b's head)
    d.b.w.receive('a', [MSG.shot, r.clock - 1, 7, s.x - 5, hy + 0.11, s.z, 55, 0, 0, COMBAT.arrowDamage, -1]);
    d.run(0.4);
    expect(d.b.w.health).toBe(FULL - pvpQuarters(COMBAT.arrowDamage * COMBAT.headshot));
    expect(d.b.sent.find((m) => m.d[0] === MSG.result)?.d[4]).toBe(OUTCOME.crit);
    d.run(0.4);
    expect(has(d.a, 'crit')).toBe(true); // (the Headshot banner at a)
  });

  it('a Perfect Guard sends the arrow back to strike its archer', () => {
    const d = duel(40, 6);
    let raised = false;
    const raise = () => { if (!raised && d.b.w.shots.some((p) => p.live && p.from === 'a')) { d.b.inp.guard = true; raised = true; } };
    const r = d.rival(d.a);
    const P = new Float64Array(3);
    rivalJoint(r, J.chest, P);
    lookAt(d.a.inp, d.a.hero, P[0], P[1], P[2]);
    d.a.inp.aim = true;
    d.run(0.9, raise);
    d.press(d.a, 'attack');
    d.run(1.5, raise);
    expect(raised).toBe(true);
    expect(has(d.b, 'parry')).toBe(true);
    expect(d.b.w.health).toBe(FULL);
    // sent back as b's own shot, it flew in a's world and struck a
    expect(d.b.sent.some((m) => m.d[0] === MSG.shot)).toBe(true);
    expect(d.a.w.health).toBe(FULL - pvpQuarters(COMBAT.arrowDamage * 1.5));
  });

  it('a thrown weapon strikes for twice its damage, and lies only in the thrower\'s world', () => {
    const d = duel(40, 5);
    const r = d.rival(d.a);
    const P = new Float64Array(3);
    rivalJoint(r, J.chest, P);
    lookAt(d.a.inp, d.a.hero, P[0], P[1], P[2]);
    d.a.inp.throw = true;
    d.run(0.5);
    d.a.inp.throw = false;
    d.run(2.0);
    expect(d.a.sent.find((m) => m.d[0] === MSG.shot)?.d[10]).toBe(WEAPON.master);
    expect(d.b.w.health).toBe(FULL - pvpQuarters(WEAPONS[WEAPON.master].damage * COMBAT.throwDamage));
    expect(d.a.w.pickups.some((p) => p.live && p.kind === 'weapon' && p.weapon === WEAPON.master)).toBe(true);
    expect(d.b.w.pickups.some((p) => p.live)).toBe(false);
    expect(d.b.w.weapons.length).toBe(1); // (b never gains a's sword)
  });
});

describe('PvP — the messages', () => {
  it('alone, nothing goes out: a swing, an arrow and a throw send no message', () => {
    const hero = createHeroState(FLAT[0], FLAT[1], Math.PI / 2, SEED);
    const w = new CombatWorld(hero, SEED);
    w.campsOn = false;
    const inp = createCombatInput();
    lookAt(inp, hero, hero.x + 8, hero.y + 1, hero.z);
    let out = 0;
    const step = (n: number) => { for (let i = 0; i < n; i++) { w.step(DT, inp); out += w.outCount; } };
    inp.attack = true; step(1); inp.attack = false; step(60);
    inp.aim = true; step(60); inp.attack = true; step(1); inp.attack = false; step(10); inp.aim = false; step(10);
    inp.throw = true; step(30); inp.throw = false; step(60);
    expect(w.shots.some((p) => p.live)).toBe(true); // (it did loose and throw)
    expect(out).toBe(0);
  });

  it('drops messages from players not seen, malformed ones, and blows on a fallen hero', () => {
    const d = duel();
    const r = d.rival(d.b);
    d.b.w.receive('zz', [MSG.blow, r.clock, 1, 999, 5, 2.25, 1.4, 300, 0]);
    d.b.w.receive('a', [MSG.blow, r.clock, 1, NaN, 5, 2.25, 1.4, 300, 0]);
    d.b.w.receive('a', [MSG.blow, r.clock, 1]);
    d.b.w.receive('a', [9, r.clock, 1, 999, 5, 2.25, 1.4, 300, 0]);
    d.run(0.5);
    expect(d.b.w.health).toBe(FULL);
    // a well-formed one lands (huge damage is capped)
    d.b.w.receive('a', [MSG.blow, d.rival(d.b).clock, 1, 999, 5, 2.25, 1.4, 300, 0]);
    d.run(0.2);
    expect(d.b.w.health).toBe(FULL - COMBAT.pvpMax);
  });

  it('a blow waits for its moment: one stamped later than the view of its player is not settled yet', () => {
    const d = duel();
    const r = d.rival(d.b);
    d.b.w.receive('a', [MSG.blow, r.clock + 500, 1, 12, 2.5, 2.25, 1.4, 300, 0]);
    d.run(0.4);
    expect(d.b.w.health).toBe(FULL);
    d.run(0.3);
    expect(d.b.w.health).toBe(FULL - swordBlow());
  });

  it('a player who leaves takes its blows and the lock with it', () => {
    const d = duel();
    d.a.inp.guard = true;
    d.run(0.4);
    expect(d.a.w.lockRival).not.toBeNull();
    const r = d.rival(d.b);
    d.b.w.receive('a', [MSG.blow, r.clock + 300, 1, 12, 2.5, 2.25, 1.4, 300, 0]);
    // b's session no longer lists a; a's no longer lists b
    (d.b.link.peers() as Map<string, number>).clear();
    (d.a.link.peers() as Map<string, number>).clear();
    d.run(1.0);
    expect(d.b.w.health).toBe(FULL);
    expect(d.a.w.lockRival).toBeNull();
    expect(d.a.w.rivals.length).toBe(0);
  });
});
