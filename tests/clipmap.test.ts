import { describe, expect, test } from 'bun:test';
import {
  ClipmapScheduler, levelSpacing, rectTexels, usableRadius, windowDelta, windowFor, wrap,
  type ClipmapSpec, type LevelWindow, type UpdateRect,
} from '../src/lib/gpu/clipmap';

const SPEC: ClipmapSpec = { n: 512, levels: 8, spacing0: 0.5, margin: 4 };

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Brute force: every lattice point of `next` is either held by `prev` or in exactly one rect. */
function checkDelta(n: number, prev: LevelWindow | null, next: LevelWindow, rects: UpdateRect[]): void {
  const inWin = (w: LevelWindow, i: number, j: number) => i >= w.i0 && i < w.i0 + n && j >= w.j0 && j < w.j0 + n;
  for (const r of rects) {
    expect(r.w).toBeGreaterThan(0);
    expect(r.h).toBeGreaterThan(0);
  }
  for (let i = next.i0; i < next.i0 + n; i++) {
    for (let j = next.j0; j < next.j0 + n; j++) {
      const hits = rects.filter((r) => i >= r.i0 && i < r.i0 + r.w && j >= r.j0 && j < r.j0 + r.h).length;
      const full = prev === null || Math.abs(next.i0 - prev.i0) >= n || Math.abs(next.j0 - prev.j0) >= n;
      const held = !full && inWin(prev, i, j);
      if (held) expect(hits).toBe(0);
      else expect(hits).toBe(1);
    }
  }
  // Nothing outside the new window is generated.
  for (const r of rects) {
    expect(r.i0).toBeGreaterThanOrEqual(next.i0);
    expect(r.j0).toBeGreaterThanOrEqual(next.j0);
    expect(r.i0 + r.w).toBeLessThanOrEqual(next.i0 + n);
    expect(r.j0 + r.h).toBeLessThanOrEqual(next.j0 + n);
  }
}

describe('clipmap geometry', () => {
  test('spacing doubles per level', () => {
    expect(levelSpacing(SPEC, 0)).toBe(0.5);
    expect(levelSpacing(SPEC, 7)).toBe(64);
  });

  test('toroidal wrap is non-negative', () => {
    expect(wrap(5, 4)).toBe(1);
    expect(wrap(-1, 4)).toBe(3);
    expect(wrap(-4, 4)).toBe(0);
    expect(wrap(-4_000_001, 512)).toBe(((-4_000_001 % 512) + 512) % 512);
  });

  test('the window is centred on the camera and keeps every tap of the usable radius inside', () => {
    const rnd = lcg(7);
    for (let k = 0; k < 2000; k++) {
      const camX = (rnd() - 0.5) * 4e6, camZ = (rnd() - 0.5) * 4e6;
      for (let l = 0; l < SPEC.levels; l++) {
        const s = levelSpacing(SPEC, l);
        const w = windowFor(SPEC, l, camX, camZ);
        const R = usableRadius(SPEC, l);
        // Bilinear footprint of the farthest sample plus a one-texel gradient tap.
        for (const x of [camX - R - s, camX + R + s]) {
          const i = Math.floor(x / s);
          expect(i).toBeGreaterThanOrEqual(w.i0);
          expect(i + 1).toBeLessThan(w.i0 + SPEC.n);
        }
        for (const z of [camZ - R - s, camZ + R + s]) {
          const j = Math.floor(z / s);
          expect(j).toBeGreaterThanOrEqual(w.j0);
          expect(j + 1).toBeLessThan(w.j0 + SPEC.n);
        }
      }
    }
  });

  test('seven levels of 512² cover the 6 km march at 0.5 m', () => {
    expect(usableRadius(SPEC, 6)).toBeGreaterThan(6000);
  });
});

describe('window deltas', () => {
  test('first frame and teleports regenerate the whole level', () => {
    const next = { i0: 10, j0: -20 };
    expect(windowDelta(16, 0, null, next)).toEqual([{ level: 0, i0: 10, j0: -20, w: 16, h: 16 }]);
    expect(windowDelta(16, 0, { i0: 10 - 16, j0: -20 }, next)).toEqual([{ level: 0, i0: 10, j0: -20, w: 16, h: 16 }]);
  });

  test('a still window needs nothing', () => {
    expect(windowDelta(16, 0, { i0: 3, j0: 4 }, { i0: 3, j0: 4 })).toEqual([]);
  });

  test('every move generates exactly the newly exposed lattice points', () => {
    const n = 16;
    const rnd = lcg(42);
    for (let k = 0; k < 400; k++) {
      const prev = { i0: Math.floor((rnd() - 0.5) * 100), j0: Math.floor((rnd() - 0.5) * 100) };
      const big = rnd() < 0.1;
      const d = () => Math.floor((rnd() - 0.5) * (big ? 40 : 10));
      const next = { i0: prev.i0 + d(), j0: prev.j0 + d() };
      const rects = windowDelta(n, 0, prev, next);
      checkDelta(n, prev, next, rects);
      const dx = Math.abs(next.i0 - prev.i0), dz = Math.abs(next.j0 - prev.j0);
      const expected = dx >= n || dz >= n ? n * n : n * n - (n - dx) * (n - dz);
      expect(rectTexels(rects)).toBe(expected);
    }
  });
});

describe('scheduler', () => {
  test('the first update fills every level, later small moves only thin bands', () => {
    const sch = new ClipmapScheduler(SPEC);
    const first = sch.update(2016.3, -2592.7);
    expect(first.length).toBe(SPEC.levels);
    expect(rectTexels(first)).toBe(SPEC.levels * SPEC.n * SPEC.n);
    expect(sch.update(2016.3, -2592.7)).toEqual([]);
    // Sprint at 8.6 m/s for one 60 Hz frame: at most one new column on level 0.
    const r = sch.update(2016.3 + 8.6 / 60, -2592.7);
    expect(rectTexels(r)).toBeLessThanOrEqual(SPEC.n);
  });

  test('flying 200 m/s for one second generates a bounded, exact amount', () => {
    const sch = new ClipmapScheduler(SPEC);
    let x = 1000.25, z = -777.75;
    sch.update(x, z);
    let total = 0;
    for (let f = 0; f < 60; f++) {
      x += 200 / 60;
      z += 50 / 60;
      total += rectTexels(sch.update(x, z));
    }
    // Per level: columns ≈ distance / spacing, times n (plus the row band).
    let bound = 0;
    for (let l = 0; l < SPEC.levels; l++) {
      const s = levelSpacing(SPEC, l);
      bound += (Math.ceil(200 / s) + 1) * SPEC.n + (Math.ceil(50 / s) + 1) * SPEC.n;
    }
    expect(total).toBeLessThanOrEqual(bound);
    expect(total).toBeGreaterThan(0);
  });

  test('invalidate regenerates everything', () => {
    const sch = new ClipmapScheduler(SPEC);
    sch.update(0, 0);
    sch.invalidate();
    expect(rectTexels(sch.update(0, 0))).toBe(SPEC.levels * SPEC.n * SPEC.n);
  });

  test('rejects specs the sampler cannot address', () => {
    expect(() => new ClipmapScheduler({ ...SPEC, n: 500 })).toThrow();
    expect(() => new ClipmapScheduler({ ...SPEC, spacing0: 0.3 })).toThrow();
    expect(() => new ClipmapScheduler({ ...SPEC, margin: 1 })).toThrow();
  });
});
