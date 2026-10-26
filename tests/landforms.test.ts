/**
 * The landforms' naturalness (terrainMath.ts karstTowers, cragDelta, mesasE, regionsInto; hydro.ts valley walls): karst towers that
 * are neither circles nor cylinders, crags on steep faces only, regions whose edges wander, walls that run in no straight line. What
 * can be pinned without a picture: the constants the code relies on, determinism, bounds, and that nothing steps (a hard step, a cliff
 * of tens of metres in one, was the straightest line of all: a mesa's lowest apron where its region began).
 */

import { describe, expect, it } from 'bun:test';
import { CRAG, DEFAULT_SEED, TOWER } from '../src/lib/constants';
import { LAND_MEANS, elevation, karstTowers, landRegions } from '../src/lib/terrainMath';
import { BIOME_COUNT } from '../src/lib/biomes';

const SEED = DEFAULT_SEED;
const lcg = (seed: number) => {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
};

describe('karst towers', () => {
  it('constants: the reach covers the longest outline, and a tower stays inside the 3 × 3 cells searched', () => {
    // the deepest a flute can pull a tower in is 0.38 of its depth (|n| − 0.38 at |n| = 0); the stretch lengthens one axis by √(1 + stretch)
    const pulled = 0.38 * (TOWER.flute[0] + TOWER.flute[1]);
    const longest = Math.sqrt(1 + TOWER.stretch) / (1 - pulled);
    expect(TOWER.reach).toBeGreaterThanOrEqual(longest * longest);
    // the lattice is 300 m: a tower in a cell two away stands at least 1.2 cells (360 m) from a point in the middle cell
    expect(145 * Math.sqrt(TOWER.reach) + TOWER.warp).toBeLessThan(1.2 * 300);
    // a ledge flattens a face by a share under 1: its slope never turns back (1 − ledge > 0)
    expect(TOWER.ledge).toBeLessThan(1);
    expect(TOWER.absent).toBeGreaterThanOrEqual(0);
    expect(TOWER.absent).toBeLessThan(0.5);
  });

  it('is deterministic, never negative, and under the tallest crown', () => {
    const rnd = lcg(11);
    let tallest = 0;
    for (let i = 0; i < 4000; i++) {
      const x = (rnd() - 0.5) * 40000, z = (rnd() - 0.5) * 40000;
      const a = karstTowers(x, z, SEED, 0);
      expect(karstTowers(x, z, SEED, 0)).toBe(a);
      expect(a).toBeGreaterThanOrEqual(0);
      tallest = Math.max(tallest, a);
    }
    // 220 m tall at most, swelled by the crown (+14 %) and its lumps (7 m)
    expect(tallest).toBeLessThan(220 * (1 + TOWER.crown) + TOWER.bump + 1);
    expect(tallest).toBeGreaterThan(100);
  });

  it('has no hard steps: points a quarter of a metre apart differ by a few metres at most', () => {
    const rnd = lcg(12);
    let worst = 0;
    for (let i = 0; i < 20000; i++) {
      const x = (rnd() - 0.5) * 40000, z = (rnd() - 0.5) * 40000;
      worst = Math.max(worst, Math.abs(karstTowers(x + 0.25, z, SEED, 0) - karstTowers(x, z, SEED, 0)), Math.abs(karstTowers(x, z + 0.25, SEED, 0) - karstTowers(x, z, SEED, 0)));
    }
    expect(worst).toBeLessThan(10); // a slope of 40
  });

  it('its mean is the one the world function fades to (LAND_MEANS.tower)', () => {
    for (const seed of [SEED, 7]) {
      let sum = 0, n = 0;
      for (let ox = 0; ox < 3000; ox += 6) for (let oz = 0; oz < 3000; oz += 6) { sum += karstTowers(20000 + ox, -8000 + oz, seed, 0); n++; }
      expect(Math.abs(sum / n / LAND_MEANS.tower - 1)).toBeLessThan(0.15);
    }
  });

  it('a coarse footprint loses the flutes, ledges and wander but keeps the towers (the same mean, no new ones)', () => {
    const rnd = lcg(13);
    let a = 0, b = 0;
    const N = 6000;
    for (let i = 0; i < N; i++) {
      const x = 20000 + rnd() * 3000, z = -8000 + rnd() * 3000;
      a += karstTowers(x, z, SEED, 0);
      b += karstTowers(x, z, SEED, 60);
      expect(karstTowers(x, z, SEED, 60)).toBeLessThan(270);
    }
    expect(Math.abs(b / a - 1)).toBeLessThan(0.2);
  });

  it('towers are not all alike: the lattice has gaps, and neighbours differ in height and in how steeply they rise', () => {
    // along a long line the tower field has bare stretches (cells with no tower, between them) and crowns of many heights
    let bare = 0, run = 0, longestBare = 0;
    const crowns = new Set<number>();
    let prev = 0, peak = false;
    for (let x = 20000; x < 26000; x += 3) {
      const t = karstTowers(x, -8000, SEED, 0);
      if (t < 0.5) { run += 3; bare += 3; longestBare = Math.max(longestBare, run); } else run = 0;
      if (t > prev && !peak) peak = true;
      if (t < prev && peak) { crowns.add(Math.round(prev / 20)); peak = false; }
      prev = t;
    }
    expect(bare).toBeGreaterThan(500);
    expect(longestBare).toBeGreaterThan(60);
    expect(crowns.size).toBeGreaterThan(3);
  });
});

describe('crags and the wandering edges', () => {
  it('constants: a ledge never turns a face back, the face threshold is a range', () => {
    expect(CRAG.ledge).toBeGreaterThan(0);
    expect(CRAG.ledge).toBeLessThan(1);
    expect(CRAG.face[0]).toBeLessThan(CRAG.face[1]);
    expect(CRAG.period).toBeGreaterThan(0);
    expect(CRAG.rough.length).toBe(3);
    expect(CRAG.mesa.length).toBe(3);
  });

  it('the regions still sum to one, with their edges bent (landRegions)', () => {
    const rnd = lcg(21);
    for (let i = 0; i < 3000; i++) {
      const w = landRegions((rnd() - 0.5) * 120000, (rnd() - 0.5) * 120000, SEED);
      let sum = 0;
      for (let k = 0; k < BIOME_COUNT; k++) {
        expect(w[k]).toBeGreaterThanOrEqual(-1e-9);
        expect(w[k]).toBeLessThanOrEqual(1 + 1e-9);
        sum += w[k];
      }
      expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
    }
  });

  it('the land steps nowhere: along lines across the world no metre drops or rises tens of metres', () => {
    // (at seed 1337's badlands edge the lowest apron of a mesa began 30 m up in one metre, along the whole edge of its region)
    const dx = Math.sin(3.08), dz = Math.cos(3.08);
    let worst = 0;
    for (let d = 400; d <= 600; d += 0.5) {
      const a = elevation(-30034 + dx * d, 8919 + dz * d, SEED, 0), b = elevation(-30034 + dx * (d + 0.5), 8919 + dz * (d + 0.5), SEED, 0);
      worst = Math.max(worst, Math.abs(b - a));
    }
    expect(worst).toBeLessThan(8); // metres per half metre: a slope of 16, a cliff but not a step
    const rnd = lcg(31);
    for (let line = 0; line < 12; line++) {
      const x0 = (rnd() - 0.5) * 80000, z0 = (rnd() - 0.5) * 80000, ang = rnd() * Math.PI * 2;
      let prev = elevation(x0, z0, SEED, 0), w = 0;
      for (let d = 1; d <= 1500; d++) {
        const e = elevation(x0 + Math.cos(ang) * d, z0 + Math.sin(ang) * d, SEED, 0);
        w = Math.max(w, Math.abs(e - prev));
        prev = e;
      }
      expect(w).toBeLessThan(45);
    }
  });
});
