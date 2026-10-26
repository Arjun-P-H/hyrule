import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'fs';
import { MAX_WORLD_H, SKY_MODELS as M } from '../src/lib/constants';
import { createHeroState, placeHero, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { modelIsleInCell, modelIslesNear, modelTop, readSkyModel, setSkyModel, type ModelIsle } from '../src/lib/skymodels';
import { skyGround } from '../src/lib/skyworld/realm';
import { worldSeed } from '../src/lib/world';
import { SPECIES } from '../src/lib/forest/species';
import { SKY_FLORA } from '../src/lib/skyworld/biomes';

const SEED = 1337;
/** The castles' scale (scripts/sky-castle/castle.py SCALE): the layouts' coordinates below are before it. */
const K = 2.5;
const models = await Promise.all(M.names.map((n) => {
  const b = readFileSync(`public/models/${n}.bin`);
  return readSkyModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}));

/** The isles of the cells within n of the origin. */
const isles: ModelIsle[] = [];
for (let iz = -6; iz < 6; iz++) for (let ix = -6; ix < 6; ix++) {
  const m = modelIsleInCell(ix, iz, SEED);
  if (m) isles.push(m);
}

describe('model isles', () => {
  test('one every few km, on the land only, inside their cells', () => {
    expect(isles.length).toBeGreaterThan(144 * M.probability * 0.6);
    expect(isles.length).toBeLessThan(144 * M.probability * 1.4);
    for (const m of isles) {
      const ix = Math.floor(m.x / M.cell), iz = Math.floor(m.z / M.cell);
      expect(m.x - ix * M.cell).toBeGreaterThanOrEqual(M.margin);
      expect((ix + 1) * M.cell - m.x).toBeGreaterThanOrEqual(M.margin);
      expect(modelIsleInCell(ix, iz, SEED)).toBe(m);
    }
    for (const m of isles) expect(modelIsleInCell(Math.floor(m.x / M.cell), Math.floor(m.z / M.cell), worldSeed(SEED, 'clouds'))).toBeNull();
    expect(modelIslesNear(0, 0, 10000, SEED, []).every((m) => Math.hypot(m.x, m.z) <= 10000)).toBe(true);
    // every layout appears, in shares near even
    for (let k = 0; k < M.names.length; k++) expect(isles.filter((m) => m.model === k).length).toBeGreaterThan(isles.length / M.names.length / 2);
  });

  test('every model fits its margin and its underside clears the land', () => {
    for (const m of models) {
      const reach = Math.hypot(Math.max(-m.boxMin[0], m.boxMax[0]), Math.max(-m.boxMin[2], m.boxMax[2]));
      expect(reach).toBeLessThanOrEqual(M.margin);
      expect(M.plaza[0] + m.boxMin[1]).toBeGreaterThan(MAX_WORLD_H);
      expect(m.parts[0].lods.length).toBe(3);
      // every prop's chunk holds a known prototype, and the chunks cover the props once each
      expect(m.props.chunks.reduce((n, c) => n + c.count, 0)).toBe(m.props.count);
      expect(m.props.chunks.every((c) => c.proto < m.protos.length)).toBe(true);
      // the land's trees and bushes grow on it, waterfalls pour off it, ivy and water are drawn on it
      expect(m.plants.length).toBeGreaterThan(20);
      expect(m.plants.every((p) => SPECIES.includes(p.species as (typeof SPECIES)[number]) && SKY_FLORA.includes(p.palette as (typeof SKY_FLORA)[number]))).toBe(true);
      expect(m.falls.length).toBeGreaterThan(3);
      expect(m.parts.map((p) => p.kind).sort()).toEqual(['leaf', 'solid', 'solid', 'water']);
    }
  });

  test('the hero lands on the plaza and walks on it', () => {
    models.forEach((m, k) => setSkyModel(k, m));
    const isle = isles.find((m) => modelTop(m, m.x, m.z, Infinity) === modelTop(m, m.x, m.z, Infinity))!;
    // a floor at the plaza's height (the outer plaza: the middle is the castle's terraces and roofs)
    let sx = NaN, sz = NaN;
    for (let r = 0; r < 340 * K && sx !== sx; r += 2) for (let a = 0; a < 6.28 && sx !== sx; a += 0.3) {
      const x = isle.x + r * Math.cos(a), z = isle.z + r * Math.sin(a);
      if (Math.abs(modelTop(isle, x, z, isle.y + 3) - isle.y) < 1) { sx = x; sz = z; }
    }
    expect(sx).not.toBeNaN();
    // from above the floor it holds; from beneath the island it does not
    expect(skyGround(sx, isle.y + 0.5, sz, SEED)).toBeCloseTo(modelTop(isle, sx, sz, isle.y + 2), 3);
    expect(skyGround(sx, isle.y + models[0].boxMin[1] - 10, sz, SEED)).toBeNaN();
    const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0, noClimb: true };
    const s = createHeroState(sx, sz, 0, SEED);
    placeHero(s, sx, isle.y + 8, sz, 0, 0, 0, 0, SEED);
    for (let i = 0; i < 180; i++) updateHero(s, idle, 1 / 60, SEED);
    expect(Math.abs(s.y - isle.y)).toBeLessThan(2);
    setSkyModel(0, undefined);
  });

  test("the castle's walls stop the hero", () => {
    models.forEach((m, k) => setSkyModel(k, m));
    const isle = isles.find((m) => m.model === M.names.indexOf('sky-castle'))!; // (the cathedral: its great hall)
    const c = Math.cos(isle.yaw), s0 = Math.sin(isle.yaw);
    // a point in the castle's own axes (Blender's: x, y, z up; the front -y) in the world
    const W = (bx: number, by: number) => [isle.x + c * bx - s0 * by, isle.z - s0 * bx - c * by];
    // inside the great hall (its floor 22 m up, its side wall 46 m out), running out toward +x
    const [sx, sz] = W(30 * K, 0);
    const [ex, ez] = W(80 * K, 0);
    const s = createHeroState(sx, sz, 0, SEED);
    placeHero(s, sx, isle.y + 24 * K, sz, 0, 0, 0, 0, SEED);
    const run: HeroInput = { forward: 1, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: Math.atan2(ex - sx, ez - sz), noClimb: true };
    for (let i = 0; i < 720; i++) updateHero(s, run, 1 / 60, SEED);
    const out = (s.x - isle.x) * c - (s.z - isle.z) * s0; // (how far along the castle's x it got)
    expect(Math.abs(s.y - (isle.y + 22 * K))).toBeLessThan(1.5);
    expect(out).toBeGreaterThan(38 * K);
    expect(out).toBeLessThan(46 * K);
    setSkyModel(0, undefined);
  });

  test('the hero climbs the grand stairs (a step each, however colossal the castle)', () => {
    models.forEach((m, k) => setSkyModel(k, m));
    const isle = isles.find((m) => m.model === M.names.indexOf('sky-castle'))!;
    const c = Math.cos(isle.yaw), s0 = Math.sin(isle.yaw);
    const W = (bx: number, by: number) => [isle.x + c * bx - s0 * by, isle.z - s0 * bx - c * by];
    // from the outer plaza, past the reflecting pool, up the front stairs to the podium (10 m up, before the scale)
    const [sx, sz] = W(45 * K, -236 * K);
    const [ex, ez] = W(0, -150 * K);
    const s = createHeroState(sx, sz, 0, SEED);
    placeHero(s, sx, isle.y + 2, sz, 0, 0, 0, 0, SEED);
    const run: HeroInput = { forward: 1, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: Math.atan2(ex - sx, ez - sz), noClimb: true };
    for (let i = 0; i < 2400; i++) updateHero(s, run, 1 / 60, SEED);
    expect(s.y).toBeGreaterThan(isle.y + 10 * K - 1);
    for (let k = 0; k < models.length; k++) setSkyModel(k, undefined);
  });

  test("the rotunda's round hall holds the hero in", () => {
    models.forEach((m, k) => setSkyModel(k, m));
    const isle = isles.find((m) => m.model === M.names.indexOf('sky-rotunda'))!;
    const c = Math.cos(isle.yaw), s0 = Math.sin(isle.yaw);
    const W = (bx: number, by: number) => [isle.x + c * bx - s0 * by, isle.z - s0 * bx - c * by];
    // from inside the hall (its floor 22 m up, its wall 62 m out) toward a bay between two columns and two galleries
    const a = (56.25 * Math.PI) / 180;
    const [sx, sz] = W(20 * K * Math.cos(a), 20 * K * Math.sin(a));
    const [ex, ez] = W(120 * K * Math.cos(a), 120 * K * Math.sin(a));
    const s = createHeroState(sx, sz, 0, SEED);
    placeHero(s, sx, isle.y + 24 * K, sz, 0, 0, 0, 0, SEED);
    const run: HeroInput = { forward: 1, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: Math.atan2(ex - sx, ez - sz), noClimb: true };
    for (let i = 0; i < 1200; i++) updateHero(s, run, 1 / 60, SEED);
    const lx = (s.x - isle.x) * c - (s.z - isle.z) * s0, lz = (s.x - isle.x) * s0 + (s.z - isle.z) * c;
    expect(Math.abs(s.y - (isle.y + 22 * K))).toBeLessThan(1.5);
    expect(Math.hypot(lx, lz)).toBeGreaterThan(50 * K);
    expect(Math.hypot(lx, lz)).toBeLessThan(62 * K);
    for (let k = 0; k < models.length; k++) setSkyModel(k, undefined);
  });
});
