import { describe, it, expect } from 'bun:test';
import { anomalyE, cellPoint, cellType } from '../src/lib/terrainMath';
import { ANOMALY, DEFAULT_SEED } from '../src/lib/constants';

/**
 * §12 — anomaly neighbour pruning is exact. The reference visits all nine
 * cells with no pruning; the engine prunes neighbours against min(F1², R_out²).
 * Random points almost never land within R_out of an anomaly (P = 0.06 per
 * cell), so the points here are placed around actual anomaly feature points
 * and across the cell borders next to them.
 */
function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function anomalyReference(x: number, y: number, seed: number): number {
  const ax = x / ANOMALY.cell, ay = y / ANOMALY.cell;
  const ipx = Math.floor(ax), ipy = Math.floor(ay);
  const fx = ax - ipx, fy = ay - ipy;
  let f1 = Infinity, bcx = 0, bcy = 0, btype = 0;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const [px, py] = cellPoint(ipx + i, ipy + j, seed + 400);
      const dx = i + px - fx, dy = j + py - fy;
      const d2 = dx * dx + dy * dy;
      if (d2 < f1) { f1 = d2; bcx = ipx + i + px; bcy = ipy + j + py; btype = cellType(ipx + i, ipy + j, seed + 400); }
    }
  }
  if (btype >= ANOMALY.probability) return 0;
  const dx = x - bcx * ANOMALY.cell, dy = y - bcy * ANOMALY.cell;
  const r2 = dx * dx + dy * dy;
  if (r2 >= ANOMALY.rOut * ANOMALY.rOut) return 0;
  const r = Math.sqrt(r2);
  const t = r / ANOMALY.terraceStep;
  const steps = (Math.floor(t) + smooth(0.3, 0.7, t - Math.floor(t))) * ANOMALY.terraceHeight;
  const spire = ANOMALY.spireHeight * Math.exp(-r2 / (ANOMALY.spireRadius * ANOMALY.spireRadius));
  const spiral = ANOMALY.spiralAmp * Math.sin(ANOMALY.spiralTurns * Math.atan2(dy, dx) + ANOMALY.spiralRadial * r);
  return (1 - smooth(ANOMALY.rIn, ANOMALY.rOut, r)) * (steps + spire + spiral);
}

describe('§12 anomaly pruning exactness', () => {
  it('matches the unpruned 3×3 search around real anomaly features and across cell borders', () => {
    // anomaly-type feature points in a 40×40-cell window (~82 km square)
    const features: Array<[number, number]> = [];
    for (let j = -20; j < 20; j++) {
      for (let i = -20; i < 20; i++) {
        if (cellType(i, j, DEFAULT_SEED + 400) < ANOMALY.probability) {
          const [px, py] = cellPoint(i, j, DEFAULT_SEED + 400);
          features.push([(i + px) * ANOMALY.cell, (j + py) * ANOMALY.cell]);
        }
      }
    }
    expect(features.length).toBeGreaterThan(50);

    let s = 0x1234567;
    const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
    let inside = 0, mismatches = 0;
    for (const [cx, cy] of features) {
      for (let k = 0; k < 400; k++) {
        const r = Math.sqrt(rnd()) * 800;               // disk of 800 m > R_out
        const a = rnd() * Math.PI * 2;
        const x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
        const got = anomalyE(x, y, DEFAULT_SEED);
        const want = anomalyReference(x, y, DEFAULT_SEED);
        if (!Object.is(got, want)) mismatches++;
        if (want !== 0) inside++;
      }
    }
    expect(inside).toBeGreaterThan(10_000);            // the test really exercises the anomaly profile
    expect(mismatches).toBe(0);
  });

  it('matches the reference on a dense grid straddling cell borders and corners', () => {
    let mismatches = 0, n = 0;
    const c = ANOMALY.cell;
    for (let j = -3; j <= 3; j++) {
      for (let i = -3; i <= 3; i++) {
        for (let u = -700; u <= 700; u += 23.7) {
          for (const v of [-650, -300, -5, 0, 5, 300, 650]) {
            const x = i * c + u, y = j * c + v;
            if (!Object.is(anomalyE(x, y, DEFAULT_SEED), anomalyReference(x, y, DEFAULT_SEED))) mismatches++;
            if (!Object.is(anomalyE(y, x, DEFAULT_SEED), anomalyReference(y, x, DEFAULT_SEED))) mismatches++;
            n += 2;
          }
        }
      }
    }
    expect(n).toBeGreaterThan(40_000);
    expect(mismatches).toBe(0);
  });
});

