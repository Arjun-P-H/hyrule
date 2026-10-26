import { describe, expect, it } from 'bun:test';
import { DEFAULT_SEED, HERO, SPAWN } from '../src/lib/constants';
import { slope } from '../src/lib/terrainMath';
import { createHeroState, updateHero, type HeroState } from '../src/lib/hero/locomotion';
import { createHeroRig, heroLocalToWorld, updateHeroRig } from '../src/lib/hero/rig';
import { CombatWorld, createCombatInput, type CombatInput } from '../src/lib/combat/world';
import { STONE, stoneClass, stoneNear } from '../src/lib/combat/stones';
import {
  createRockFrame, focusRocks, freeRocks, packRocks, resetRocks, rockBodiesNear, rockTopAt, type RockBody,
} from '../src/lib/rocks/field';
import { ROCK_TIERS } from '../src/lib/rocks/tiers';

/**
 * Rocks as things (src/lib/rocks/field.ts, src/lib/combat/stones.ts): the CPU's rocks are solid by their own shape (a
 * body is stopped where the rock is drawn and stands on its top), the box round the hero is listed for the GPU, and the
 * hero lifts a stone, carries it, throws it at a foe and puts it down, and pushes a boulder into a roll — on the real
 * terrain round the spawn.
 */

const SEED = DEFAULT_SEED;
const DT = 1 / 60;

/** Every rock of the solid tiers within r of the spawn whose class is `cls`, nearest first, on gentle open ground. */
function rocksOfClass(cls: number, r = 400): RockBody[] {
  const seen = new Map<number, RockBody>();
  const out: RockBody[] = [];
  for (let dx = -r; dx <= r; dx += 6) for (let dz = -r; dz <= r; dz += 6) for (const b of rockBodiesNear(SPAWN.x + dx, SPAWN.z + dz, SEED, 1, out)) seen.set(b.key, b);
  const all = [...seen.values()];
  return all
    .filter((b) => stoneClass(b) === cls && slope(b.x, b.z, SEED) < 0.25
      // (alone: no other rock within 4 m of it)
      && all.every((o) => o === b || Math.hypot(o.x - b.x, o.z - b.z) > 4 + o.reach))
    .sort((a, b) => Math.hypot(a.x - SPAWN.x, a.z - SPAWN.z) - Math.hypot(b.x - SPAWN.x, b.z - SPAWN.z));
}

/** A world with the hero standing `back` m from rock b, facing it (from the side the ground is flattest). */
function before(b: RockBody, back: number) {
  let best = 0, bm = Infinity;
  for (let k = 0; k < 16; k++) {
    const a = (k / 16) * Math.PI * 2;
    const m = slope(b.x - Math.sin(a) * (back + 3), b.z - Math.cos(a) * (back + 3), SEED) + slope(b.x + Math.sin(a) * 9, b.z + Math.cos(a) * 9, SEED);
    if (m < bm) { bm = m; best = a; }
  }
  const hero = createHeroState(b.x - Math.sin(best) * back, b.z - Math.cos(best) * back, best, SEED);
  hero.yaw = best;
  const w = new CombatWorld(hero, SEED);
  w.campsOn = false;
  const inp = createCombatInput();
  look(inp, hero, best, 0.1);
  return { w, hero, inp, rig: createHeroRig(), yaw: best };
}

function look(inp: CombatInput, s: HeroState, yaw: number, pitch: number): void {
  inp.camYaw = yaw; inp.camPitch = pitch;
  inp.eyeX = s.x - Math.sin(yaw) * 4; inp.eyeY = s.y + 1.8; inp.eyeZ = s.z - Math.cos(yaw) * 4;
  inp.dirX = Math.sin(yaw) * Math.cos(pitch); inp.dirY = Math.sin(pitch); inp.dirZ = Math.cos(yaw) * Math.cos(pitch);
}

const RK = new Float64Array(3);
/** The game's loop for `seconds`: the world's step, the rig, a rock in the hands put where the rig holds it. */
function run(t: ReturnType<typeof before>, seconds: number, each?: () => void): void {
  for (let i = 0; i < Math.round(seconds / DT); i++) {
    each?.();
    t.w.step(DT, t.inp);
    updateHeroRig(t.rig, t.w.hero, t.w.heroDt, t.w.pose);
    if (t.w.stone && t.rig.rockAt[0] === t.rig.rockAt[0]) {
      heroLocalToWorld(t.rig, t.w.hero, t.rig.rockAt[0], t.rig.rockAt[1], t.rig.rockAt[2], RK);
      t.w.holdStone(RK[0], RK[1], RK[2]);
    }
  }
}
function press(t: ReturnType<typeof before>, key: 'action' | 'attack' | 'throw'): void {
  t.inp[key] = true;
  run(t, 2 * DT);
  t.inp[key] = false;
}

describe('rocks are solid by their shape', () => {
  it("a rock's top is the mesh's: within its height, none off its footprint", () => {
    resetRocks();
    for (const b of rocksOfClass(3, 200).slice(0, 4)) {
      let over = 0;
      for (let a = 0; a < 16; a++) {
        const top = rockTopAt(b, b.x + Math.cos(a) * b.reach * 0.3, b.z + Math.sin(a) * b.reach * 0.3);
        if (top > -Infinity) { over++; expect(top).toBeLessThanOrEqual(b.y + b.height * 1.6); }
        expect(rockTopAt(b, b.x + Math.cos(a) * (b.reach + 0.6), b.z + Math.sin(a) * (b.reach + 0.6))).toBe(-Infinity);
      }
      expect(over).toBeGreaterThan(8);
    }
  }, 30_000);

  it('a hero running at a boulder is stopped at its face, never standing in it', () => {
    resetRocks();
    const b = rocksOfClass(3, 300).find((r) => r.height > 1.4)!;
    expect(b).toBeDefined();
    const t = before(b, b.reach + 5);
    let deepest = Infinity;
    for (let i = 0; i < 60 * 4; i++) {
      // (a foe's way: the hero pushing into a boulder climbs it, or vaults onto a low one, hero.test.ts)
      updateHero(t.hero, { forward: 1, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: t.yaw, noClimb: true }, DT, SEED);
      // (where the hero stands, the rock's top is under its knees, or it stands on the rock)
      const top = rockTopAt(b, t.hero.x, t.hero.z);
      if (top > t.hero.y + 0.55) deepest = Math.min(deepest, -1);
      deepest = Math.min(deepest, Math.hypot(t.hero.x - b.x, t.hero.z - b.z));
    }
    expect(deepest).toBeGreaterThan(0);
    // stopped near its face, not 1 m short of it
    expect(deepest).toBeLessThan(b.reach + 0.6);
  });
});

describe('the box round the hero is the CPU\'s', () => {
  it('lists its rocks for the GPU and the scatter leaves its cells, a taken rock gone from both', () => {
    resetRocks();
    const b = rocksOfClass(2, 200)[0];
    focusRocks(b.x, b.z, SEED);
    const f = packRocks(b.x, b.z, createRockFrame());
    expect(f.nList).toBeGreaterThan(0);
    // the box of the stones' tier holds the rock's cell
    const t = 1, c = ROCK_TIERS[t].cell, ix = Math.floor(b.x / c), iz = Math.floor(b.z / c);
    expect(ix >= f.zone[t * 4] && ix <= f.zone[t * 4 + 2] && iz >= f.zone[t * 4 + 1] && iz <= f.zone[t * 4 + 3]).toBe(true);
    const listed = (fr: typeof f) => {
      const u = new Float32Array(fr.list);
      for (let k = 0; k < fr.nList; k++) if (Math.abs(u[k * 16 + 3] - b.size) < 1e-5) return true;
      return false;
    };
    expect(listed(f)).toBe(true);
    expect(f.nTaken).toBe(0);
  });
});

describe('the hero and the stones', () => {
  it('lifts a stone overhead in both hands, carries it at a walk, and puts it down ahead', () => {
    resetRocks();
    const b0 = rocksOfClass(2, 300)[0];
    const t = before(b0, b0.reach + 0.7);
    // (the same rock: its cell's; the scan's objects may have been made again since)
    const b = stoneNear(t.hero, SEED)!;
    expect(b.key).toBe(b0.key);
    expect(t.w.stoneHint()).toBe('Lift');
    press(t, 'action');
    expect(t.w.act).toBe('lift');
    run(t, STONE.liftTime[1] + 0.2);
    expect(t.w.act).toBe('carry');
    expect(t.w.stone).toBe(b);
    // overhead: its bottom over the hero's head
    expect(b.y - b.height * 0.5).toBeGreaterThan(t.hero.y + 1.3);
    // taken: listed as free, its cell empty for the scatter
    focusRocks(t.hero.x, t.hero.z, SEED);
    const f = packRocks(t.hero.x, t.hero.z, createRockFrame());
    expect(f.nTaken).toBe(1);
    expect(f.taken[1] === b.ix && f.taken[2] === b.iz).toBe(true);
    // a walk, slower than a run
    const x0 = t.hero.x, z0 = t.hero.z;
    t.inp.forward = 1;
    run(t, 1);
    t.inp.forward = 0;
    const walked = Math.hypot(t.hero.x - x0, t.hero.z - z0);
    expect(walked).toBeGreaterThan(1);
    expect(walked).toBeLessThan(HERO.runSpeed * 0.7);
    run(t, 0.4);
    press(t, 'action');
    run(t, STONE.downTime + 2.5);
    expect(t.w.stone).toBeNull();
    expect(t.w.act).toBe('idle');
    expect(b.state).toBe('loose');
    // ahead of the hero, on the ground
    const ahead = (b.x - t.hero.x) * Math.sin(t.hero.yaw) + (b.z - t.hero.z) * Math.cos(t.hero.yaw);
    expect(ahead).toBeGreaterThan(0.3);
    expect(ahead).toBeLessThan(2.5);
  });

  it('throws a stone at a foe ahead: it flies, strikes and harms it, and lies where it comes to rest', () => {
    resetRocks();
    const b0 = rocksOfClass(1, 300)[0];
    const t = before(b0, b0.reach + 0.6);
    press(t, 'action');
    run(t, STONE.liftTime[0] + 0.2);
    expect(t.w.act).toBe('carry');
    const b = t.w.stone!;
    expect(b.key).toBe(b0.key);
    const s = t.hero;
    const fx = s.x + Math.sin(t.yaw) * 9, fz = s.z + Math.cos(t.yaw) * 9;
    t.w.foesOn = true;
    const foe = t.w.spawnFoe(0, fx, fz, t.yaw + Math.PI)!;
    expect(foe).not.toBeNull();
    const hp = foe.hp;
    press(t, 'attack');
    expect(t.w.act).toBe('hurl');
    let flew = false;
    run(t, 1.6, () => { if (b.state === 'flying') flew = true; });
    expect(flew).toBe(true);
    expect(foe.hp).toBeLessThan(hp);
    run(t, 6);
    expect(['loose', 'rolling']).toContain(b.state);
    expect(freeRocks()).toContain(b);
  });

  it('pushes a boulder into a roll: it runs off the way it was pushed and comes to rest', () => {
    resetRocks();
    const b0 = rocksOfClass(3, 400)[0];
    const t = before(b0, b0.reach * 0.8 + 0.6);
    expect(t.w.stoneHint()).toBe('Push');
    const b = stoneNear(t.hero, SEED)!;
    expect(b.key).toBe(b0.key);
    const x0 = b.x, z0 = b.z;
    press(t, 'action');
    expect(t.w.act).toBe('push');
    run(t, STONE.pushTime + 0.1);
    expect(b.state === 'rolling' || b.state === 'loose').toBe(true);
    run(t, 20);
    expect(b.state).toBe('loose');
    const d = Math.hypot(b.x - x0, b.z - z0);
    expect(d).toBeGreaterThan(0.8);
    // away from the hero
    expect((b.x - x0) * Math.sin(t.yaw) + (b.z - z0) * Math.cos(t.yaw)).toBeGreaterThan(0);
  });
});
