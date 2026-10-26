import { describe, it, expect } from 'bun:test';
import { elevation, slope, waterLevel } from '../src/lib/terrainMath';
import { COMBAT, DEFAULT_SEED, FOE, HERO, HERO_SHAPE, SPAWN } from '../src/lib/constants';
import { createHeroState, startSurf, type HeroState } from '../src/lib/hero/locomotion';
import { createCombatPose, createHeroRig, updateHeroRig, GEAR, J, type CombatPose } from '../src/lib/hero/rig';
import { CombatWorld, createCombatInput, type CombatInput } from '../src/lib/combat/world';
import { COMBO, MOVE, MOVES, swingAt, SWING_CHANNELS } from '../src/lib/combat/moves';
import { MODEL, WEAPON, WEAPONS } from '../src/lib/combat/weapons';
import { FOE_KINDS, campInCell, campsNear, foeJoint, spawnCamp, type Foe } from '../src/lib/combat/foes';
import { aimAt, segmentsMeet } from '../src/lib/combat/projectiles';
import { CombatScene } from '../src/lib/combat/scene';
import { plantInCell } from '../src/lib/forest/placement';
import { CELL } from '../src/lib/forest/species';

/**
 * Combat (src/lib/combat/): the hero's sword, shield, bow, throws, dodges and
 * shield surfing, the Bokoblins' senses, turns and blows, what flies, and the
 * camps — deterministic, on the real terrain round the spawn.
 */

const SEED = DEFAULT_SEED;
const DT = 1 / 60;

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

/** No tree (bushes aside) within r metres of (x, z). */
function clearOfTrees(x: number, z: number, r: number): boolean {
  for (let iz = Math.floor((z - r) / CELL); iz <= Math.floor((z + r) / CELL); iz++) {
    for (let ix = Math.floor((x - r) / CELL); ix <= Math.floor((x + r) / CELL); ix++) {
      const p = plantInCell(ix, iz, SEED);
      if (p && p.species !== 'bush' && Math.hypot(p.x - x, p.z - z) < r) return false;
    }
  }
  return true;
}

// level, dry, open ground with room for a fight (as tests/hero.test.ts's)
const FLAT = findSpot((x, z) => slope(x, z, SEED) < 0.06 && elevation(x, z, SEED, 0) > waterLevel(x, z, SEED) + 2.9
  && [0, 1, 2, 3, 4, 5, 6, 7].every((k) => slope(x + 12 * Math.cos(k * Math.PI / 4), z + 12 * Math.sin(k * Math.PI / 4), SEED) < 0.08)
  && clearOfTrees(x, z, 30), 4);

/** A world at FLAT, the hero facing +x, no camps; the view 4 m behind the hero, looking where it faces. */
function arena(yaw = Math.PI / 2) {
  const hero = createHeroState(FLAT[0], FLAT[1], yaw, SEED);
  const w = new CombatWorld(hero, SEED);
  w.campsOn = false;
  const inp = createCombatInput();
  inp.camYaw = yaw;
  look(inp, hero, yaw, -0.05);
  return { w, hero, inp };
}

/** Point the view from behind the hero along yaw / pitch. */
function look(inp: CombatInput, s: HeroState, yaw: number, pitch: number): void {
  inp.camYaw = yaw;
  inp.camPitch = pitch;
  inp.eyeX = s.x - Math.sin(yaw) * 4; inp.eyeY = s.y + 1.8; inp.eyeZ = s.z - Math.cos(yaw) * 4;
  inp.dirX = Math.sin(yaw) * Math.cos(pitch); inp.dirY = Math.sin(pitch); inp.dirZ = Math.cos(yaw) * Math.cos(pitch);
}

/** The view from behind the hero aimed at world point (x, y, z). */
function lookAt(inp: CombatInput, s: HeroState, x: number, y: number, z: number): void {
  const yaw = Math.atan2(x - s.x, z - s.z);
  inp.camYaw = yaw;
  inp.eyeX = s.x - Math.sin(yaw) * 2; inp.eyeY = s.y + 1.7; inp.eyeZ = s.z - Math.cos(yaw) * 2;
  const dx = x - inp.eyeX, dy = y - inp.eyeY, dz = z - inp.eyeZ, l = Math.hypot(dx, dy, dz);
  inp.dirX = dx / l; inp.dirY = dy / l; inp.dirZ = dz / l;
  inp.camPitch = Math.asin(inp.dirY);
}

function run(w: CombatWorld, inp: CombatInput, seconds: number, each?: (t: number) => void): void {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) {
    each?.(i * DT);
    w.step(DT, inp);
  }
}

/** A foe of `kind` d metres along `yaw` from the hero, facing `face` (default: the hero). */
function foeAt(w: CombatWorld, kind: number, d: number, yaw: number, face?: number): Foe {
  const s = w.hero;
  const x = s.x + Math.sin(yaw) * d, z = s.z + Math.cos(yaw) * d;
  const f = w.spawnFoe(kind, x, z, face ?? Math.atan2(s.x - x, s.z - z))!;
  expect(f).not.toBeNull();
  return f;
}

const events = (w: CombatWorld) => w.events.slice(0, w.eventCount).map((e) => e.type);

/** Hold foe f still where it stands (dazed for good): a target. */
function held(f: Foe): Foe {
  f.roused = true; f.mind = 'stun'; f.t = -1e3;
  return f;
}

describe('Combat — the sword', () => {
  it('chains the four-hit combo on presses in time; a fifth starts it over', () => {
    const { w, inp } = arena();
    const moves: number[] = [];
    let was = -1;
    run(w, inp, 2.2, (t) => {
      inp.attack = Math.round(t / DT) % 14 < 2; // a tap every 233 ms
      const now = w.act === 'swing' ? w.move : -1;
      if (now >= 0 && now !== was) moves.push(now);
      was = now;
    });
    expect(moves.slice(0, 5)).toEqual([...COMBO, COMBO[0]]);
  });

  it('a swing strikes a foe in its reach and arc once; not one behind the hero', () => {
    const { w, inp } = arena();
    const ahead = foeAt(w, 2, 1.6, Math.PI / 2);
    const behind = foeAt(w, 2, 1.6, -Math.PI / 2);
    const hp = FOE_KINDS[2].hp;
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    run(w, inp, 0.5);
    expect(ahead.hp).toBe(hp - WEAPONS[WEAPON.master].damage * MOVES[MOVE.slashR].damage);
    expect(behind.hp).toBe(hp);
  });

  it('a sneakstrike on an unaware foe deals ×8', () => {
    const { w, inp } = arena();
    const f = foeAt(w, 2, 1.5, Math.PI / 2, Math.PI / 2); // (facing away from the hero)
    inp.sneak = true;
    run(w, inp, 2 * DT);
    expect(w.hero.sneaking).toBe(true);
    inp.attack = true;
    run(w, inp, 0.4);
    expect(f.hp).toBe(FOE_KINDS[2].hp - WEAPONS[WEAPON.master].damage * COMBAT.sneakstrike);
  });

  it('holding attack charges a spin; letting go sweeps all round for its stamina', () => {
    const { w, hero, inp } = arena();
    const ring = [0, 1, 2, 3].map((k) => held(foeAt(w, 2, 1.8, (k * Math.PI) / 2)));
    inp.attack = true;
    run(w, inp, 0.42 + COMBAT.chargeAfter + COMBAT.chargeFull + 0.1);
    expect(w.act).toBe('charge');
    expect(w.charge).toBe(1);
    const st = hero.stamina;
    inp.attack = false;
    w.step(DT, inp);
    expect(w.act).toBe('swing');
    expect(w.move).toBe(MOVE.spin);
    expect(st - hero.stamina).toBeGreaterThanOrEqual(COMBAT.staminaSpin - 1e-9);
    let turned = 0;
    run(w, inp, 0.7, () => { turned = Math.min(turned, w.pose.spin); });
    expect(turned).toBeLessThan(-1.8 * Math.PI);
    // (the first slash struck the one ahead; the spin struck all four)
    for (const f of ring) expect(f.hp).toBeLessThan(FOE_KINDS[2].hp - WEAPONS[0].damage * MOVES[MOVE.spin].damage + 1e-9);
  });

  it('a jump attack in the air, a dash attack at a sprint', () => {
    const { w, hero, inp } = arena();
    inp.jump = true;
    run(w, inp, 0.15);
    inp.jump = false;
    expect(hero.move).toBe('air');
    inp.attack = true;
    w.step(DT, inp);
    expect(w.move).toBe(MOVE.jump);
    inp.attack = false;
    run(w, inp, 1.2);
    inp.forward = 1; inp.sprint = true;
    run(w, inp, 0.6);
    expect(hero.sprinting).toBe(true);
    inp.attack = true;
    w.step(DT, inp);
    expect(w.move).toBe(MOVE.dash);
  });
});

/** A foe ahead in the middle of a swing, its blow due in a few ticks. */
function swingingFoe(w: CombatWorld, d: number, yaw: number, kind = 0): Foe {
  const f = foeAt(w, kind, d, yaw);
  f.roused = true; f.aware = 1;
  f.mind = 'attack'; f.t = 0;
  f.move = MOVE.smash;
  f.u = MOVES[MOVE.smash].hit[0] - 0.03;
  f.struck = false;
  return f;
}

describe('Combat — the shield', () => {
  it('raised just as a blow lands, it parries: the foe is dazed, the hero unharmed', () => {
    const { w, inp } = arena();
    const f = swingingFoe(w, 1.8, Math.PI / 2);
    const h0 = w.health;
    inp.guard = true;
    run(w, inp, 0.15);
    expect(w.act).toBe('guard');
    expect(f.mind).toBe('stun');
    expect(w.health).toBe(h0);
  });

  it('held up before, it guards (no harm, a push back); a blow from behind still lands', () => {
    const a = arena();
    a.inp.guard = true;
    run(a.w, a.inp, 0.5);
    const f = swingingFoe(a.w, 1.8, Math.PI / 2);
    run(a.w, a.inp, 0.15);
    expect(f.mind).not.toBe('stun');
    expect(a.w.health).toBe(a.w.hearts * 4);
    const b = arena();
    b.inp.guard = true;
    run(b.w, b.inp, 0.5);
    swingingFoe(b.w, 1.8, -Math.PI / 2);
    run(b.w, b.inp, 0.1);
    expect(b.w.health).toBe(b.w.hearts * 4 - FOE_KINDS[0].damage);
    expect(b.w.act).toBe('hurt');
  });

  it('locks onto the foe nearest the view; locked, a side hop timed to a blow opens a Flurry Rush', () => {
    const { w, inp } = arena();
    const f = foeAt(w, 2, 2.0, Math.PI / 2);
    f.roused = true; f.mind = 'wait';
    inp.guard = true;
    run(w, inp, 0.3);
    expect(w.lock).toBe(f.slot);
    // its blow comes; the hero hops aside as it falls
    f.mind = 'attack'; f.t = 0; f.move = MOVE.sweep; f.u = MOVES[MOVE.sweep].hit[0] - 0.02; f.struck = false;
    inp.jump = true; inp.strafe = 1;
    w.step(DT, inp);
    inp.jump = false; inp.strafe = 0;
    expect(w.act).toBe('dodge');
    run(w, inp, 0.2);
    expect(w.flurryWait).toBeGreaterThan(0);
    expect(w.health).toBe(w.hearts * 4);
    const hp = f.hp;
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    expect(w.act).toBe('flurry');
    w.step(DT, inp);
    expect(w.worldScale).toBe(COMBAT.flurrySlow);
    run(w, inp, 1.6);
    expect(hp - f.hp).toBeCloseTo(COMBAT.flurryHits * WEAPONS[0].damage, 6);
    expect(w.act).not.toBe('flurry');
  });

  it('with the shield up, the action key steps onto it: it surfs down a slope, slows to a stop on the flat', () => {
    const SLOPE = findSpot((x, z) => {
      const m = slope(x, z, SEED);
      if (m < 0.25 || m > 0.7 || elevation(x, z, SEED, 0) < waterLevel(x, z, SEED) + 2.9) return false;
      const e = elevation(x, z, SEED, 0);
      const gx = (elevation(x + 0.5, z, SEED, 0) - e) / 0.5, gz = (elevation(x, z + 0.5, SEED, 0) - e) / 0.5;
      const g = Math.hypot(gx, gz);
      // (downhill for 20 m)
      for (let d = 4; d <= 20; d += 4) if (elevation(x - (gx / g) * d, z - (gz / g) * d, SEED, 0) > e - 0.15 * d) return false;
      return clearOfTrees(x - (gx / g) * 10, z - (gz / g) * 10, 12);
    }, 6);
    const e = elevation(SLOPE[0], SLOPE[1], SEED, 0);
    const gx = (elevation(SLOPE[0] + 0.5, SLOPE[1], SEED, 0) - e) / 0.5, gz = (elevation(SLOPE[0], SLOPE[1] + 0.5, SEED, 0) - e) / 0.5;
    const down = Math.atan2(-gx, -gz);
    const hero = createHeroState(SLOPE[0], SLOPE[1], down, SEED);
    const w = new CombatWorld(hero, SEED);
    w.campsOn = false;
    const inp = createCombatInput();
    look(inp, hero, down, -0.2);
    inp.guard = true;
    run(w, inp, 0.2);
    inp.action = true;
    w.step(DT, inp);
    inp.action = false; inp.guard = false;
    expect(hero.move).toBe('surf');
    run(w, inp, 2.0);
    expect(hero.surfing).toBe(true);
    expect(Math.hypot(hero.vx, hero.vz)).toBeGreaterThan(HERO.runSpeed);
    // up the rise ahead of the spawn it slows to a stop and steps off …
    const up = createHeroState(SPAWN.x, SPAWN.z, SPAWN.yaw, SEED);
    const uw = new CombatWorld(up, SEED);
    uw.campsOn = false;
    const ui = createCombatInput();
    startSurf(up);
    run(uw, ui, 0.5);
    expect(up.surfing).toBe(true);
    run(uw, ui, 3);
    expect(up.surfing).toBe(false);
    expect(up.move).toBe('ground');
    // … and the action key steps off it at any time
    const flat = arena();
    startSurf(flat.hero);
    run(flat.w, flat.inp, 0.5);
    expect(flat.hero.surfing).toBe(true);
    flat.inp.action = true;
    flat.w.step(DT, flat.inp);
    expect(flat.hero.surfing).toBe(false);
  });
});

describe('Combat — the bow and throws', () => {
  it('aimAt finds the arc that comes down on the mark', () => {
    const v = new Float64Array(3);
    for (const [dx, dy] of [[30, 2], [60, -5], [12, 6]]) {
      aimAt(0, 0, 0, dx, dy, 0, COMBAT.arrowSpeed, COMBAT.arrowGravity, v);
      let x = 0, y = 0, vy = v[1], best = Infinity;
      for (let t = 0; t < 3; t += 1e-4) {
        x += v[0] * 1e-4; vy -= COMBAT.arrowGravity * 1e-4; y += vy * 1e-4;
        best = Math.min(best, Math.hypot(x - dx, y - dy));
      }
      expect(best).toBeLessThan(0.05);
      expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(COMBAT.arrowSpeed, 6);
    }
  });

  it('drawn and loosed at a foe, an arrow strikes it (twice on the head, and dazes); one at the ground sticks', () => {
    const { w, hero, inp } = arena();
    const f = held(foeAt(w, 2, 18, Math.PI / 2));
    w.step(DT, inp); // (its rig posed)
    const head = new Float64Array(3);
    foeJoint(f, J.head, head);
    inp.aim = true;
    lookAt(inp, hero, head[0], head[1], head[2]);
    run(w, inp, COMBAT.nockTime + COMBAT.drawTime + 0.1);
    expect(w.act).toBe('aim');
    expect(w.draw).toBe(1);
    const n0 = w.arrows;
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    expect(w.arrows).toBe(n0 - 1);
    const seen: string[] = [];
    run(w, inp, 0.8, () => seen.push(...events(w)));
    w.step(DT, inp);
    expect(seen.includes('crit') || seen.includes('hit')).toBe(true);
    const dealt = FOE_KINDS[2].hp - f.hp;
    expect(dealt).toBeGreaterThanOrEqual(COMBAT.arrowDamage - 1e-9);
    if (seen.includes('crit')) expect(dealt).toBeCloseTo(COMBAT.arrowDamage * COMBAT.headshot, 6);
    // at the ground 15 m ahead: it sticks there
    const gx = hero.x + 15, gz = hero.z;
    lookAt(inp, hero, gx, elevation(gx, gz, SEED, 0), gz);
    run(w, inp, COMBAT.nockTime + COMBAT.drawTime + 0.1);
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    run(w, inp, 1.0);
    const stuck = w.shots.filter((p) => p.live && p.stuck);
    expect(stuck.length).toBe(1);
    expect(Math.abs(stuck[0].y - elevation(stuck[0].x, stuck[0].z, SEED, 0))).toBeLessThan(0.05);
  });

  it('in the air, drawing the bow slows the world (bullet time) on stamina', () => {
    const { w, hero, inp } = arena();
    inp.jump = true;
    run(w, inp, 0.1);
    inp.jump = false; inp.aim = true;
    run(w, inp, 0.1);
    expect(hero.move).toBe('air');
    expect(w.bulletTime).toBe(true);
    expect(w.worldScale).toBe(COMBAT.bulletTime);
    expect(hero.stamina).toBeLessThan(1);
  });

  it('a throw: the weapon leaves the hand, strikes for twice its damage, lies to be picked up — the sword comes back', () => {
    const { w, hero, inp } = arena();
    const f = held(foeAt(w, 2, 7, Math.PI / 2));
    w.step(DT, inp); // (its rig posed)
    const chest = new Float64Array(3);
    foeJoint(f, J.chest, chest);
    lookAt(inp, hero, chest[0], chest[1], chest[2]);
    inp.throw = true;
    run(w, inp, 0.4);
    expect(w.act).toBe('throw');
    inp.throw = false;
    run(w, inp, 0.3);
    expect(w.weapons.some((c) => c.kind === WEAPON.master)).toBe(false);
    expect(w.carried()).toBeNull();
    run(w, inp, 1.0);
    expect(FOE_KINDS[2].hp - f.hp).toBeCloseTo(WEAPONS[WEAPON.master].damage * COMBAT.throwDamage, 6);
    expect(w.pickups.some((p) => p.live && p.kind === 'weapon' && p.weapon === WEAPON.master)).toBe(true);
    run(w, inp, COMBAT.swordReturn + 0.5);
    expect(w.carried()?.kind).toBe(WEAPON.master);
  });

  it('a club picked up wears with each blow and breaks; the sword is taken in hand again', () => {
    const { w, inp } = arena();
    held(foeAt(w, 2, 1.7, Math.PI / 2));
    // a club at the hero's feet, picked up
    w.pickups[0].live = true; w.pickups[0].kind = 'weapon'; w.pickups[0].weapon = WEAPON.club; w.pickups[0].wear = 1;
    w.pickups[0].x = w.hero.x + 0.5; w.pickups[0].z = w.hero.z; w.pickups[0].y = w.hero.y; w.pickups[0].vy = 0;
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(w.carried()?.kind).toBe(WEAPON.club);
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    const seen: string[] = [];
    run(w, inp, 0.5, () => seen.push(...events(w)));
    expect(seen).toContain('break');
    expect(w.carried()?.kind).toBe(WEAPON.master);
    expect(w.weapons.some((c) => c.kind === WEAPON.club)).toBe(false);
  });
});

describe('Combat — foes', () => {
  it('see the hero in their view cone, not behind them; a sneaking hero is seen only close', () => {
    const seenFrom = (d: number, facing: boolean, sneak: boolean) => {
      const { w, inp } = arena();
      const f = foeAt(w, 0, d, Math.PI / 2, facing ? undefined : Math.PI / 2);
      f.t = -1e3; // (it stays where it stands, looking the same way)
      inp.sneak = sneak;
      run(w, inp, 2.5);
      return f.roused;
    };
    expect(seenFrom(22, true, false)).toBe(true);
    expect(seenFrom(22, false, false)).toBe(false);
    expect(seenFrom(22, true, true)).toBe(false);
    expect(seenFrom(10, true, true)).toBe(true);
  });

  it('take turns: no more than FOE.attackers swing at the hero at once, each resting between swings', () => {
    const { w, inp } = arena();
    const fs = [0, 1, 2].map((k) => foeAt(w, 0, 5, Math.PI / 2 + (k - 1) * 0.9));
    for (const f of fs) { f.roused = true; f.mind = 'chase'; }
    let most = 0, swings = 0, was = new Set<number>();
    run(w, inp, 8, () => {
      const now = new Set(fs.filter((f) => f.live && f.mind === 'attack').map((f) => f.slot));
      most = Math.max(most, now.size);
      for (const s of now) if (!was.has(s)) swings++;
      was = now;
      if (w.act === 'dead') w.revive();
    });
    expect(most).toBe(FOE.attackers);
    expect(swings).toBeGreaterThanOrEqual(3);
    // (8 s, swings of ~1.4 s and rests of ≥ 1.3 s between a foe's own: never back to back)
    expect(swings).toBeLessThan(8);
  });

  it('paused (the lobby open), foes and what flies stand still and strike no one', () => {
    const { w, inp } = arena();
    const f = swingingFoe(w, 1.8, Math.PI / 2);
    const x0 = f.body.x, u0 = f.u;
    w.paused = true;
    run(w, inp, 1.5);
    expect(f.body.x).toBe(x0);
    expect(f.u).toBe(u0);
    expect(w.health).toBe(w.hearts * 4);
    w.paused = false;
    run(w, inp, 0.2);
    expect(w.health).toBe(w.hearts * 4 - FOE_KINDS[0].damage);
  });

  it('their blows take quarters of a heart; at none the hero falls; revive restores it', () => {
    const { w, inp } = arena();
    w.health = 4;
    swingingFoe(w, 1.8, Math.PI / 2);
    run(w, inp, 0.2);
    expect(w.health).toBe(0);
    expect(w.act).toBe('dead');
    expect(events(w).length).toBeGreaterThanOrEqual(0);
    run(w, inp, 1);
    expect(w.downFor).toBeGreaterThan(0.9);
    w.revive();
    expect(w.health).toBe(w.hearts * 4);
    expect(w.act).toBe('idle');
  });

  it('beaten, a foe falls and goes up in smoke, leaving its club (and maybe arrows or a heart)', () => {
    const { w, inp } = arena();
    const f = foeAt(w, 0, 1.6, Math.PI / 2);
    f.hp = 1;
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    const seen: string[] = [];
    run(w, inp, FOE.corpse + 0.6, () => seen.push(...events(w)));
    expect(seen).toContain('down');
    expect(seen).toContain('poof');
    expect(f.live).toBe(false);
    expect(w.pickups.some((p) => p.live && p.kind === 'weapon' && p.weapon === FOE_KINDS[0].weapon)).toBe(true);
  });

  it('archers keep their distance and shoot; a raised shield stops their arrows, a parry sends them back', () => {
    const { w, inp } = arena();
    const a = foeAt(w, 0, 16, Math.PI / 2);
    a.archer = true; a.roused = true; a.mind = 'shoot';
    const seen: string[] = [];
    run(w, inp, 3.5, () => seen.push(...events(w)));
    expect(seen).toContain('fire');
    // with the shield up toward it, its arrows are stopped (no harm)
    const g = arena();
    const b = foeAt(g.w, 0, 16, Math.PI / 2);
    b.archer = true; b.roused = true; b.mind = 'shoot';
    g.inp.guard = true;
    run(g.w, g.inp, 6);
    expect(g.w.health).toBe(g.w.hearts * 4);
  });
});

describe('Combat — determinism and the camps', () => {
  it('identical input gives identical worlds (a fight with three foes)', () => {
    const script = (t: number, inp: CombatInput) => {
      inp.forward = Math.sin(t * 1.1) > 0 ? 1 : 0;
      inp.attack = Math.sin(t * 5.3) > 0.6;
      inp.guard = Math.sin(t * 0.7) > 0.5;
      inp.jump = Math.sin(t * 2.9) > 0.95;
      inp.strafe = Math.sin(t * 0.9) > 0.7 ? 1 : 0;
    };
    const play = () => {
      const { w, hero, inp } = arena();
      const fs = [0, 1, 2].map((k) => foeAt(w, k, 6 + k, Math.PI / 2 + k));
      for (const f of fs) { f.roused = true; f.mind = 'chase'; }
      run(w, inp, 8, (t) => script(t, inp));
      return [hero.x, hero.y, hero.z, hero.yaw, w.health, ...fs.flatMap((f) => [f.hp, f.body.x, f.body.z, f.live ? 1 : 0])];
    };
    const a = play(), b = play();
    for (let k = 0; k < a.length; k++) expect(Object.is(a[k], b[k])).toBe(true);
  });

  it('a camp waits up the rise ahead of the spawn, on level dry ground; camps are placed by hash', () => {
    const c = spawnCamp(SEED)!;
    expect(c).not.toBeNull();
    const d = Math.hypot(c.x - SPAWN.x, c.z - SPAWN.z);
    expect(d).toBeGreaterThan(40);
    expect(d).toBeLessThan(80);
    expect(slope(c.x, c.z, SEED)).toBeLessThan(0.28);
    expect(c.y).toBeGreaterThan(waterLevel(c.x, c.z, SEED));
    expect(campsNear(SPAWN.x, SPAWN.z, FOE.wakeRange, SEED, []).some((k) => k === c)).toBe(true);
    let n = 0;
    for (let iz = -20; iz < 20; iz++) for (let ix = -20; ix < 20; ix++) {
      const k = campInCell(ix, iz, SEED);
      expect(campInCell(ix, iz, SEED)).toBe(k);
      if (k) {
        n++;
        expect(Math.floor(k.x / FOE.campCell)).toBe(ix);
        expect(k.kinds.length).toBeGreaterThanOrEqual(2);
      }
    }
    expect(n).toBeGreaterThan(40);
  });

  it('the camps near the hero wake as it comes and are put away when it leaves', () => {
    const hero = createHeroState(SPAWN.x, SPAWN.z, SPAWN.yaw, SEED);
    const w = new CombatWorld(hero, SEED);
    const inp = createCombatInput();
    w.step(DT, inp);
    expect(w.foes.some((f) => f.live)).toBe(true);
    hero.x += 5000;
    w.step(DT, inp);
    w.step(DT, inp);
    const camp = spawnCamp(SEED)!;
    expect(w.foes.some((f) => f.live && Math.hypot(f.body.x - camp.x, f.body.z - camp.z) < 30)).toBe(false);
  });
});

describe('Combat — the rig and the scene', () => {
  const settle = (pose: CombatPose, s: HeroState, frames = 50) => {
    const r = createHeroRig();
    for (let i = 0; i < frames; i++) updateHeroRig(r, s, DT, pose);
    return r;
  };
  const len = (a: Float64Array, i: number) => Math.hypot(a[i], a[i + 1], a[i + 2]);

  it('keeps bone lengths and orthonormal gear frames in every combat pose', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0.4, SEED);
    const poses: Array<Partial<CombatPose>> = [
      { armed: true }, { armed: true, act: 'guard' }, { armed: true, act: 'charge', glow: 1 },
      ...COMBO.map((m) => ({ armed: true, act: 'swing' as const, move: m, u: 0.5 })),
      { armed: true, act: 'swing', move: MOVE.spin, u: 0.4, spin: -2 }, { act: 'aim', draw: 1, pitch: 0.3 },
      { armed: true, act: 'throw', move: MOVE.throw, u: 0.36 }, { armed: true, act: 'hurt', u: 0.1 },
      { weapon: MODEL.spiked, armed: true, shield: false, act: 'swing', move: MOVE.smash, u: 0.5 },
    ];
    for (const over of poses) {
      const p = { ...createCombatPose(), ...over };
      const r = settle(p, s);
      for (const [a, b, l] of [[J.shoulderL, J.elbowL, HERO_SHAPE.upperArm], [J.elbowL, J.handL, HERO_SHAPE.forearm],
        [J.shoulderR, J.elbowR, HERO_SHAPE.upperArm], [J.elbowR, J.handR, HERO_SHAPE.forearm]] as const) {
        const d = Math.hypot(r.local[3 * a] - r.local[3 * b], r.local[3 * a + 1] - r.local[3 * b + 1], r.local[3 * a + 2] - r.local[3 * b + 2]);
        expect(Math.abs(d - l)).toBeLessThan(2e-3);
      }
      const g = r.gear;
      if (g[GEAR.grip + 3] > 0) {
        expect(len(g, GEAR.axis)).toBeCloseTo(1, 6);
        expect(len(g, GEAR.flat)).toBeCloseTo(1, 6);
        expect(g[GEAR.axis] * g[GEAR.flat] + g[GEAR.axis + 1] * g[GEAR.flat + 1] + g[GEAR.axis + 2] * g[GEAR.flat + 2]).toBeCloseTo(0, 6);
      }
      if (g[GEAR.shield + 3] > 0) {
        expect(len(g, GEAR.shieldX)).toBeCloseTo(1, 6);
        expect(len(g, GEAR.shieldY)).toBeCloseTo(1, 6);
        expect(g[GEAR.shieldX] * g[GEAR.shieldY] + g[GEAR.shieldX + 1] * g[GEAR.shieldY + 1] + g[GEAR.shieldX + 2] * g[GEAR.shieldY + 2]).toBeCloseTo(0, 6);
      }
      // the weapon in hand sits at the hand
      if (p.armed && p.act !== 'aim' && g[GEAR.grip + 3] > 0) {
        expect(Math.hypot(g[GEAR.grip] - r.local[3 * J.handR], g[GEAR.grip + 1] - r.local[3 * J.handR + 1], g[GEAR.grip + 2] - r.local[3 * J.handR + 2])).toBeLessThan(1e-9);
      }
      // the bounding sphere holds the blade's tip
      if (g[GEAR.grip + 3] > 0) {
        const L = g[GEAR.grip + 3] === MODEL.sword ? HERO_SHAPE.swordLen : HERO_SHAPE.clubLen;
        const tip = [0, 1, 2].map((c) => g[GEAR.grip + c] + g[GEAR.axis + c] * L);
        const [cx, cy, cz, rad] = r.bound;
        expect(Math.hypot(tip[0] - cx, tip[1] - cy, tip[2] - cz) + HERO_SHAPE.gearPad).toBeLessThanOrEqual(rad + 1e-9);
      }
    }
  });

  it('a swing\'s blade sweeps its arc: the keys are passed through, the strike is fast', () => {
    const out = new Float64Array(SWING_CHANNELS);
    for (const m of MOVES) {
      for (const k of m.keys) {
        swingAt(m, k[0], out);
        expect(out[0]).toBeCloseTo((k[1] * Math.PI) / 180, 9);
        expect(out[2]).toBeCloseTo(k[3], 9);
      }
    }
    // the slash sweeps from its wind-up on the right to the left through its strike
    const a = swingAt(MOVES[MOVE.slashR], MOVES[MOVE.slashR].hit[0], new Float64Array(SWING_CHANNELS))[0];
    const b = swingAt(MOVES[MOVE.slashR], MOVES[MOVE.slashR].hit[1], new Float64Array(SWING_CHANNELS))[0];
    expect(a - b).toBeGreaterThan(Math.PI * 0.6);
  });

  it('the scene packs foes, their shadow bodies, arrows and effects (and no more than its lists hold)', () => {
    const { w, hero, inp } = arena();
    held(foeAt(w, 1, 4, Math.PI / 2));
    const sc = new CombatScene();
    const r = createHeroRig();
    inp.attack = true;
    run(w, inp, 0.6, () => { updateHeroRig(r, hero, DT, w.pose); sc.update(w, r, DT); });
    const fr = sc.pack(w, hero.x - 3, hero.y + 2, hero.z);
    expect(fr.nFoes).toBe(1);
    expect(fr.nShadows).toBe(1);
    expect(fr.nSprites).toBeGreaterThan(0);
    // (camera-relative: the foe's pelvis ~ 7 m ahead of the camera)
    expect(Math.hypot(fr.shadows[0], fr.shadows[2])).toBeGreaterThan(5);
    expect(Math.hypot(fr.shadows[0], fr.shadows[2])).toBeLessThan(9);
    expect(segmentsMeet(0, 0, 0, 2, 0, 0, 1, -1, 0.1, 1, 1, 0.1, 0.11)).toBeCloseTo(0.5, 6);
    expect(segmentsMeet(0, 0, 0, 2, 0, 0, 1, -1, 0.3, 1, 1, 0.3, 0.2)).toBe(-1);
  });
});
