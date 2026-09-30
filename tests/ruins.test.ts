import { describe, expect, test } from 'bun:test';
import { RUINS, PLAYER } from '../src/lib/constants';
import { ruinSiteAt, ruinSitesNear, toBoxLocal, type RuinSite } from '../src/lib/ruins';
import { createHeroState, firstPersonGround, placeHero, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { elevation, updatePlayer, type InputState, type PlayerState } from '../src/lib/terrainMath';
import { islandTop, islandsNear } from '../src/lib/skyIslands';

const SEED = 1337;

function sites(n: number): RuinSite[] {
  return ruinSitesNear(0, 0, n * RUINS.cell, SEED, 10_000);
}

describe('ruins', () => {
  test('sites are rarer than their cells, stay in the middle of their cell, and are deterministic', () => {
    const all = sites(12);
    const cells = Math.PI * 12 * 12; // cells whose centre can fall within the range, roughly
    expect(all.length).toBeGreaterThan(5);
    expect(all.length / cells).toBeLessThan(RUINS.probability + 0.05);
    for (const s of all) {
      const fx = s.x / RUINS.cell - Math.floor(s.x / RUINS.cell), fz = s.z / RUINS.cell - Math.floor(s.z / RUINS.cell);
      expect(fx).toBeGreaterThanOrEqual(0.25);
      expect(fx).toBeLessThanOrEqual(0.75);
      expect(fz).toBeGreaterThanOrEqual(0.25);
      expect(fz).toBeLessThanOrEqual(0.75);
      expect(ruinSiteAt(s.x, s.z, SEED)).toEqual(s);
    }
    expect(sites(4).map((s) => s.x)).toEqual(sites(4).map((s) => s.x));
  });

  test('every box sits within the site and rests on the platform or below it', () => {
    for (const s of sites(10)) {
      const platform = s.boxes[0];
      expect(platform.y + platform.hy).toBeCloseTo(s.top, 9);
      for (const b of s.boxes) {
        expect(Math.hypot(b.x - s.x, b.z - s.z)).toBeLessThan(s.radius);
        expect(b.y - b.hy).toBeLessThanOrEqual(s.top + 1e-9 + 20);
        expect(b.hx).toBeGreaterThan(0);
        expect(b.hy).toBeGreaterThan(0);
        expect(b.hz).toBeGreaterThan(0);
      }
      // Pillars stand exactly on the platform top.
      const pillars = s.boxes.filter((b) => Math.abs(b.hx - 0.6) < 1e-9 && Math.abs(b.hz - 0.6) < 1e-9);
      expect(pillars.length).toBeGreaterThanOrEqual(4);
      for (const p of pillars) expect(p.y - p.hy).toBeCloseTo(s.top, 9);
    }
  });

  test('the hero stands on a platform and is stopped by a pillar', () => {
    const s = sites(10)[0];
    const hero = createHeroState(s.x, s.z, 0, SEED);
    placeHero(hero, s.x, s.top + 3, s.z, 0, 0, 0, 0, SEED);
    const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };
    for (let i = 0; i < 120; i++) updateHero(hero, idle, 1 / 60, SEED);
    expect(hero.move).toBe('ground');
    expect(Math.abs(hero.y - (s.top + PLAYER.groundBias))).toBeLessThan(0.05);

    // Walk straight at the first pillar from the platform centre; never end up inside it.
    const pillar = s.boxes.find((b) => Math.abs(b.hx - 0.6) < 1e-9 && b.hy > 2)!;
    const aim = Math.atan2(pillar.x - hero.x, pillar.z - hero.z);
    const run: HeroInput = { ...idle, forward: 1, camYaw: aim };
    let closest = Infinity;
    for (let i = 0; i < 60 * 8; i++) {
      updateHero(hero, run, 1 / 60, SEED);
      // While the body overlaps the pillar's height, its centre stays a body radius from the footprint.
      if (hero.y + 1.7 <= pillar.y - pillar.hy) continue;
      const [u, v] = toBoxLocal(pillar, hero.x, hero.z);
      const du = u - Math.max(-pillar.hx, Math.min(pillar.hx, u)), dv = v - Math.max(-pillar.hz, Math.min(pillar.hz, v));
      const dist = Math.hypot(du, dv);
      expect(dist).toBeGreaterThan(0.34);
      closest = Math.min(closest, dist);
    }
    expect(closest).toBeLessThan(0.4); // it did run into the pillar (and slid along it)
  });

  test('first person: stands on a platform, is stopped by a pillar, lands on a sky island', () => {
    const s = sites(10)[0];
    const eye = PLAYER.height + PLAYER.groundBias;
    const p: PlayerState = { x: s.x, y: s.top + 3, z: s.z, vx: 0, vy: 0, vz: 0, grounded: false };
    const idle: InputState = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };
    for (let i = 0; i < 120; i++) updatePlayer(p, idle, 1 / 60, SEED, firstPersonGround);
    expect(p.grounded).toBe(true);
    expect(Math.abs(p.y - (s.top + eye))).toBeLessThan(1e-6);
    // Without the hook the same player sinks to the terrain inside the platform.
    const q: PlayerState = { ...p };
    for (let i = 0; i < 120; i++) updatePlayer(q, idle, 1 / 60, SEED);
    expect(q.y - eye).toBeCloseTo(elevation(q.x, q.z, SEED, 0), 6);

    const pillar = s.boxes.find((b) => Math.abs(b.hx - 0.6) < 1e-9 && b.hy > 2)!;
    const run: InputState = { ...idle, forward: 1, yaw: Math.atan2(pillar.x - p.x, pillar.z - p.z) };
    let closest = Infinity;
    for (let i = 0; i < 60 * 8; i++) {
      updatePlayer(p, run, 1 / 60, SEED, firstPersonGround);
      const [u, v] = toBoxLocal(pillar, p.x, p.z);
      const dist = Math.hypot(u - Math.max(-pillar.hx, Math.min(pillar.hx, u)), v - Math.max(-pillar.hz, Math.min(pillar.hz, v)));
      expect(dist).toBeGreaterThan(0.34);
      closest = Math.min(closest, dist);
    }
    expect(closest).toBeLessThan(0.4);

    const isl = islandsNear(0, 0, 20_000, SEED, 1)[0];
    const top = islandTop(isl, isl.cx, isl.cz);
    const f: PlayerState = { x: isl.cx, y: top + 20, z: isl.cz, vx: 0, vy: 0, vz: 0, grounded: false };
    for (let i = 0; i < 180; i++) updatePlayer(f, idle, 1 / 60, SEED, firstPersonGround);
    expect(f.grounded).toBe(true);
    expect(Math.abs(f.y - (top + eye))).toBeLessThan(1e-6);
  });
});
