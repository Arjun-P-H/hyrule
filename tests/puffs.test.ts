import { describe, expect, test } from 'bun:test';
import { CLOUDS, SPAWN } from '../src/lib/constants';
import { createHeroState, placeHero, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { cloudSurface } from '../src/lib/skyworld/clouds';
import { islandRho, topAt } from '../src/lib/skyworld/island';
import { billowsNear, islandBillows, puffChainsNear, puffTop, type Puff } from '../src/lib/skyworld/puffs';
import { islandAt, realmNear, skyGround } from '../src/lib/skyworld/realm';
import { worldSeed } from '../src/lib/world';

const C = worldSeed(1337, 'clouds');
// (4 km, not more: a wider survey leaves enough built behind to slow the timing tests run after this file)
const REALM = realmNear(SPAWN.x, SPAWN.z, 4000, C); // (ladders and banks only by Eyries and isles already built)
const LADDERS = puffChainsNear(SPAWN.x, SPAWN.z, 4000, C, []).slice(0, 30);
const top = (p: Puff) => p.y + p.ry;
const reach = (p: Puff) => Math.max(p.rx, p.rz);

/**
 * Is there ground within a step of the puff's edge, level with its top (ground(x, z): height or NaN)? Looked for
 * toward (tx, tz) (where the chain leads), from the edge out as a chain is made, and all round.
 */
function stepsOnto(p: Puff, ground: (x: number, z: number) => number, tx: number, tz: number): boolean {
  const toward = Math.atan2(tz - p.z, tx - p.x);
  for (let a = -1; a < 64; a++) {
    const th = a < 0 ? toward : (a / 64) * 2 * Math.PI;
    for (let d = a < 0 ? reach(p) : 0; d <= reach(p) + CLOUDS.puffGap; d += 0.25) {
      const g = ground(p.x + Math.cos(th) * d, p.z + Math.sin(th) * d);
      if (g === g && g - top(p) <= CLOUDS.puffRise && g - top(p) >= -5) return true; // (a step up, or a hop down)
    }
  }
  return false;
}

describe('puffs: the ladders up the islands', () => {
  test('there are ladders, only in the cloud world', () => {
    expect(LADDERS.length).toBeGreaterThanOrEqual(10);
    expect(puffChainsNear(SPAWN.x, SPAWN.z, 4000, 1337, []).length).toBe(0);
  });

  test('every ladder is climbable step by step, from the cloud sea up', () => {
    for (const c of LADDERS) {
      const P = c.puffs;
      for (let k = 1; k < P.length; k++) {
        const a = P[k - 1], b = P[k];
        expect(top(b) - top(a)).toBeLessThanOrEqual(a.spring > 0 ? a.spring : CLOUDS.puffRise);
        expect(Math.hypot(b.x - a.x, b.z - a.z) - reach(a) - reach(b)).toBeLessThanOrEqual(CLOUDS.puffGap);
      }
      for (const p of P) {
        if (p.spring > 0) {
          expect(p.spring).toBeGreaterThanOrEqual(CLOUDS.springLift[0]);
          expect(p.spring).toBeLessThanOrEqual(CLOUDS.springLift[1]);
        }
        expect(p.y - p.ry).toBeGreaterThanOrEqual(cloudSurface(p.x, p.z, C, 0) - 0.01); // (never sunk into the cloud)
      }
      expect(top(P[0]) - cloudSurface(P[0].x, P[0].z, C, 0)).toBeLessThanOrEqual(CLOUDS.puffRise);
    }
  });

  test('a ladder ends a step from its island\'s top, and no puff lies inside any island', () => {
    for (const c of LADDERS) {
      const isl = c.island;
      const last = c.puffs[c.puffs.length - 1];
      expect(stepsOnto(last, (x, z) => (islandRho(isl, x, z) <= 1 ? topAt(isl, x, z) : NaN), isl.cx, isl.cz)).toBe(true);
      for (const p of c.puffs) {
        for (let a = 0; a < 8; a++) {
          const x = p.x + Math.cos(a * 0.785) * reach(p), z = p.z + Math.sin(a * 0.785) * reach(p);
          expect(islandAt(x, z, C)).toBeNull();
        }
      }
    }
  });

  test('puffs are floor: the ground over a puff is its top', () => {
    const p = LADDERS[0].puffs[5];
    expect(skyGround(p.x, top(p) + 1, p.z, C)).toBeCloseTo(puffTop(p, p.x, p.z), 6);
    expect(Number.isNaN(puffTop(p, p.x + reach(p) + 0.1, p.z))).toBe(true);
  });

  test('a springy puff throws the hero up by its lift', () => {
    const p = LADDERS[0].puffs.find((q) => q.spring >= 12)!;
    const h = createHeroState(p.x, p.z, 0, C);
    placeHero(h, p.x, top(p) + 2, p.z, 0, 0, 0, 0, C);
    const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };
    let apex = -Infinity, bounced = false;
    for (let i = 0; i < 600; i++) {
      updateHero(h, idle, 1 / 120, C);
      if (h.vy > 0) bounced = true;
      if (bounced) apex = Math.max(apex, h.y);
      if (bounced && h.vy < 0 && h.y < apex - 0.5) break;
    }
    expect(bounced).toBe(true);
    expect(apex - top(p)).toBeGreaterThan(p.spring - 1);
    expect(apex - top(p)).toBeLessThan(p.spring + 1);
  });
});

describe('billows: drawn cloud on the sea and round the islands', () => {
  const near = billowsNear(SPAWN.x, SPAWN.z, 2500, C, []);

  test('the sea is heaped with lobes half sunk in it; the land has none', () => {
    expect(near.length).toBeGreaterThan(200);
    expect(billowsNear(SPAWN.x, SPAWN.z, 2500, 1337, []).length).toBe(0);
  });

  test('every island but debris trails a bank of cloud under its rim, at its underside', () => {
    const islands = [...REALM.archipelagos.flatMap((a) => a.islands), ...REALM.isles].filter((i) => i.kind !== 'debris');
    expect(islands.length).toBeGreaterThan(5);
    for (const isl of islands) {
      const B = islandBillows(isl);
      expect(B.length).toBeGreaterThanOrEqual(5);
      for (const b of B) {
        expect(b.y).toBeLessThan(isl.y);
        expect(b.y).toBeGreaterThan(isl.y - isl.depth);
        expect(islandRho(isl, b.x, b.z)).toBeLessThanOrEqual(1);
      }
    }
  });

  test('billows are not floor', () => {
    const b = near[0];
    const f = skyGround(b.x, b.y + b.ry + 0.5, b.z, C);
    expect(f === f && Math.abs(f - (b.y + b.ry)) < 1e-6).toBe(false);
  });
});

describe('puffs: every player builds the same ladders', () => {
  test('a ladder does not depend on which islands were built first', () => {
    const S = worldSeed(4242, 'clouds');
    const key = (cs: ReturnType<typeof puffChainsNear>) => cs.map((c) => `${c.puffs[0].x.toFixed(3)},${c.puffs.length}`).sort();
    realmNear(SPAWN.x, SPAWN.z, 2000, S);
    const first = key(puffChainsNear(SPAWN.x, SPAWN.z, 2000, S, []));
    realmNear(SPAWN.x, SPAWN.z, 4000, S);
    puffChainsNear(SPAWN.x, SPAWN.z, 1, C, []); // (another seed: the ladders are made again; C, not a new one: the realm keeps four seeds' islands)
    expect(key(puffChainsNear(SPAWN.x, SPAWN.z, 2000, S, []))).toEqual(first);
  });
});
