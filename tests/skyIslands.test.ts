import { describe, expect, test } from 'bun:test';
import { PLAYER, SKY_ISLANDS as S } from '../src/lib/constants';
import { createHeroState, placeHero, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { islandAt, islandBottom, islandGround, islandInCell, islandRho, islandTop, islandsNear, outline } from '../src/lib/skyIslands';

const SEED = 1337;

function islands(n: number) {
  const out = [];
  for (let iz = -n; iz < n; iz++) for (let ix = -n; ix < n; ix++) {
    const isl = islandInCell(ix, iz, SEED);
    if (isl) out.push({ ix, iz, isl });
  }
  return out;
}

describe('sky islands', () => {
  test('about the configured share of cells hold an island, deterministically', () => {
    const all = islands(20);
    const share = all.length / (40 * 40);
    expect(share).toBeGreaterThan(S.probability - 0.05);
    expect(share).toBeLessThan(S.probability + 0.05);
    expect(islandInCell(3, -7, SEED)).toEqual(islandInCell(3, -7, SEED));
  });

  test('an island never leaves its cell, so a point only needs its own cell', () => {
    const maxOutline = 1 + S.outline * (0.6 + 0.4 + 0.25);
    for (const { ix, iz, isl } of islands(20)) {
      const reach = isl.radius * maxOutline;
      expect(isl.cx - reach).toBeGreaterThan(ix * S.cell);
      expect(isl.cx + reach).toBeLessThan((ix + 1) * S.cell);
      expect(isl.cz - reach).toBeGreaterThan(iz * S.cell);
      expect(isl.cz + reach).toBeLessThan((iz + 1) * S.cell);
    }
  });

  test('top and underside meet at the rim and the top stays above the underside', () => {
    for (const { isl } of islands(8)) {
      for (let k = 0; k < 64; k++) {
        const th = (k / 64) * Math.PI * 2;
        for (const rho of [0, 0.3, 0.6, 0.85, 0.95, 1]) {
          const r = rho * isl.radius * outline(isl, th);
          const x = isl.cx + r * Math.cos(th), z = isl.cz + r * Math.sin(th);
          const top = islandTop(isl, x, z), bot = islandBottom(isl, x, z);
          if (rho === 1) expect(Math.abs(top - bot)).toBeLessThan(1e-9);
          else expect(top).toBeGreaterThan(bot);
        }
      }
    }
  });

  test('islandAt / islandGround find the top from above and ignore points below', () => {
    const { isl } = islands(8)[0];
    const top = islandTop(isl, isl.cx, isl.cz);
    expect(islandAt(isl.cx, isl.cz, SEED)).toEqual(isl);
    expect(islandGround(isl.cx, top + 5, isl.cz, SEED)).toBeCloseTo(top, 9);
    expect(islandGround(isl.cx, top - 0.5, isl.cz, SEED)).toBeCloseTo(top, 9);
    expect(Number.isNaN(islandGround(isl.cx, top - 40, isl.cz, SEED))).toBe(true);
    const out = isl.cx + isl.radius * 2;
    expect(islandRho(isl, out, isl.cz)).toBeGreaterThan(1);
    expect(islandAt(out, isl.cz, SEED)).toBeNull();
  });

  test('islandsNear returns the islands in range, nearest first', () => {
    const near = islandsNear(2016, -2592, 6000, SEED, 32);
    expect(near.length).toBeGreaterThan(0);
    let last = 0;
    for (const isl of near) {
      const d = Math.hypot(isl.cx - 2016, isl.cz + 2592);
      expect(d).toBeLessThanOrEqual(6000);
      expect(d).toBeGreaterThanOrEqual(last);
      last = d;
    }
  });
});


describe('the hero on a sky island', () => {
  test('lands on the top, runs on it, and falls when running off the rim', () => {
    const { isl } = islands(8)[0];
    const s = createHeroState(isl.cx, isl.cz, 0, SEED);
    // Dropped from 20 m above the island's centre.
    placeHero(s, isl.cx, islandTop(isl, isl.cx, isl.cz) + 20, isl.cz, 0, 0, 0, Math.PI / 2, SEED);
    const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: Math.PI / 2 };
    for (let i = 0; i < 180; i++) updateHero(s, idle, 1 / 60, SEED);
    expect(s.move).toBe('ground');
    expect(Math.abs(s.y - (islandTop(isl, s.x, s.z) + PLAYER.groundBias))).toBeLessThan(0.05);

    // Run east (camYaw π/2 → +x) until past the rim; stay on the top while inside it.
    const run: HeroInput = { ...idle, forward: 1 };
    let left = false;
    for (let i = 0; i < 60 * 120 && !left; i++) {
      updateHero(s, run, 1 / 60, SEED);
      const rho = islandRho(isl, s.x, s.z);
      if (rho < 0.95 && s.move === 'ground') {
        expect(Math.abs(s.y - (islandTop(isl, s.x, s.z) + PLAYER.groundBias))).toBeLessThan(0.3);
      }
      if (rho > 1.05) left = true;
    }
    expect(left).toBe(true);
    // Off the edge the ground is the terrain hundreds of metres below.
    for (let i = 0; i < 30; i++) updateHero(s, run, 1 / 60, SEED);
    expect(s.move).toBe('air');
    expect(s.y - s.ground).toBeGreaterThan(100);
  });
});
