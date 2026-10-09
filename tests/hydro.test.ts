import { describe, expect, test } from 'bun:test';
import {
  HYDRO_STRIDE, OCEAN, PROF_N, SINK, RF,
  accOf, codeOf, downOf, hydroTable, lakeOf, levelOf, meanderAmp, nodeAt, profileOf, riverField, riverLakeLevel, riverLakeOf, upliftAt,
  type HydroNode,
} from '../src/lib/hydro';
import { domainWarp, surfaceAt, lastFlow } from '../src/lib/terrainMath';
import { DEFAULT_SEED, HYDRO, LANDS } from '../src/lib/constants';

/**
 * The geography's rivers (src/lib/hydro.ts): every river runs downhill to the sea or a lake, grows
 * as it goes, never floats over the land, and the lakes lie level inside their basins.
 */
const SEED = DEFAULT_SEED;
const N = 14; // nodes round the origin each way (≈ ±29 km)

/** The world point the warp carries to warped (wx, wz) (fixed-point iteration). */
function worldOf(wx: number, wz: number): [number, number] | null {
  // Newton's method on the warp (its Jacobian by central differences); null where it does not
  // converge (~1 % of nodes: where the warp folds the plane there is no one world point)
  let x = wx, z = wz;
  for (let it = 0; it < 40; it++) {
    const [qx, qz] = domainWarp(x, z, SEED, 0);
    const ex = wx - qx, ez = wz - qz;
    if (Math.hypot(ex, ez) < 1e-4) return [x, z];
    const [ax, az] = domainWarp(x + 1, z, SEED, 0), [bx, bz] = domainWarp(x - 1, z, SEED, 0);
    const [cx, cz] = domainWarp(x, z + 1, SEED, 0), [dx, dz] = domainWarp(x, z - 1, SEED, 0);
    const j11 = (ax - bx) / 2, j21 = (az - bz) / 2, j12 = (cx - dx) / 2, j22 = (cz - dz) / 2;
    const det = j11 * j22 - j12 * j21;
    let sx = (j22 * ex - j12 * ez) / det, sz = (-j21 * ex + j11 * ez) / det;
    if (!(Math.hypot(sx, sz) < 400)) { sx = ex; sz = ez; }
    x += sx; z += sz;
  }
  return null;
}

const nodes: HydroNode[] = [];
for (let j = -N; j <= N; j++) for (let i = -N; i <= N; i++) nodes.push(nodeAt(i, j, SEED));
const land = nodes.filter((n) => codeOf(n, SEED) !== OCEAN);

describe('the drainage graph', () => {
  test('every land node drains, step by step, to the sea or a lake (no cycles)', () => {
    for (const n of land) {
      let m: HydroNode | null = n, steps = 0;
      while (m && steps < 2000) { m = downOf(m, SEED); steps++; }
      expect(steps).toBeLessThan(2000);
    }
  });

  test('a river never climbs: each node lies at or above the one it drains into, and its reach falls all the way', () => {
    for (const n of land) {
      const d = downOf(n, SEED);
      if (!d) continue;
      const Wa = levelOf(n, SEED), Wb = levelOf(d, SEED);
      expect(Wb).toBeLessThanOrEqual(Wa + 1e-9);
      const p = profileOf(n, SEED)!;
      expect(p.length).toBe(PROF_N - 1);
      let prev = Wa;
      for (const v of p) { expect(v).toBeLessThanOrEqual(prev + 1e-9); prev = v; }
      expect(Wb).toBeLessThanOrEqual(prev + 1e-9);
    }
  });

  test('rivers grow downstream: a node drains more land than any node above it', () => {
    for (const n of land) {
      const d = downOf(n, SEED);
      if (d && codeOf(d, SEED) !== OCEAN) expect(accOf(d, SEED)).toBeGreaterThan(accOf(n, SEED));
    }
  });

  test('the land has its rivers, lakes, sea and waterfalls (TotK-scale variety near the origin)', () => {
    const big = land.filter((n) => accOf(n, SEED) >= 8).length;
    // (lakes at sinks, and lakes the rivers widen into)
    const lakes = land.filter((n) => (codeOf(n, SEED) === SINK && lakeOf(n, SEED) > -Infinity) || riverLakeOf(n, SEED) > 0).length;
    const sea = nodes.length - land.length;
    let falls = 0;
    for (const n of land) {
      const d = downOf(n, SEED);
      if (!d) continue;
      const lv = [levelOf(n, SEED), ...profileOf(n, SEED)!, levelOf(d, SEED)];
      for (let k = 0; k + 1 < lv.length; k++) if (lv[k] - lv[k + 1] > HYDRO.fallDrop) { falls++; break; }
    }
    expect(big).toBeGreaterThan(5);
    expect(lakes).toBeGreaterThan(3);
    expect(sea).toBeGreaterThan(0);
    expect(falls).toBeGreaterThan(0);
  });

  test('rivers wind on the flat and run straight down slopes', () => {
    expect(meanderAmp(0)).toBe(HYDRO.meander);
    expect(meanderAmp(0.2)).toBe(HYDRO.meanderMin);
  });
});

describe('the rivers on the land', () => {
  test('water stands in a river\'s channel, flows down it, and never stands above its banks', () => {
    const s: [number, number] = [0, 0];
    let checked = 0;
    for (const n of land) {
      const d = downOf(n, SEED);
      if (!d || accOf(n, SEED) < 3 || codeOf(d, SEED) === OCEAN) continue;
      // the channel's middle (the reach's line in the warped plane; find the world point there)
      const tx = n.x + (d.x - n.x) * 0.25, tz = n.z + (d.z - n.z) * 0.25;
      riverField(tx, tz, SEED, 0.12, true);
      if (RF.L > -Infinity) continue; // (in a lake)
      // search the world near the warped point for a wet cell with flow
      let found = false;
      for (let r = 0; r < 900 && !found; r += 15) {
        for (let a = 0; a < 24 && !found; a++) {
          const x = tx + r * Math.cos(a / 24 * Math.PI * 2), z = tz + r * Math.sin(a / 24 * Math.PI * 2);
          surfaceAt(x, z, SEED, s);
          if (s[1] > s[0] + 0.05 && lastFlow()[2] > 0) {
            found = true;
            const [fx, fz] = lastFlow();
            // walk across the flow to where the water meets the land: the bank there stands at the
            // water's level (water standing over its bank would draw a wall)
            for (const side of [-1, 1]) {
              let Wlast = s[1];
              for (let off = 1; off <= 400; off += 1) {
                const bx = x + side * -fz * off, bz = z + side * fx * off;
                surfaceAt(bx, bz, SEED, s);
                if (s[1] > s[0] + 0.02) { Wlast = s[1]; continue; }
                // (to within 0.6 m: across a meander the walk can meet the channel falling along itself)
                expect(s[0]).toBeGreaterThan(Wlast - 0.6);
                break;
              }
            }
          }
        }
      }
      if (found) checked++;
      if (checked >= 25) break;
    }
    expect(checked).toBeGreaterThan(10);
  });

  test('a lake lies level inside its ring, its shore above it all round', () => {
    const s: [number, number] = [0, 0];
    let lakes = 0;
    for (const n of land) {
      if (codeOf(n, SEED) !== SINK) continue;
      const L = lakeOf(n, SEED);
      if (L === -Infinity) continue;
      const rL = HYDRO.lakeR * HYDRO.node;
      // the ring in the warped plane bounds it: the land there stands above the lake
      for (let k = 0; k < 32; k++) {
        const x = n.x + rL * Math.cos(k / 32 * Math.PI * 2), z = n.z + rL * Math.sin(k / 32 * Math.PI * 2);
        riverField(x, z, SEED, 0.04, false);
        expect(Math.min(upliftAt(x, z, SEED), RF.R)).toBeGreaterThan(L);
      }
      lakes++;
      if (lakes >= 8) break;
    }
    expect(lakes).toBeGreaterThanOrEqual(2);
    // and where the world shows a lake's water, it is that one level
    let wet = 0;
    for (let i = 0; i < 20000 && wet < 50; i++) {
      const x = (i * 9173.7) % 50000 - 25000, z = (i * 31337.3) % 50000 - 25000;
      surfaceAt(x, z, SEED, s);
      if (s[1] > s[0] + 0.5 && RF.L > -Infinity && lastFlow()[2] === 0 && Math.abs(s[1] - RF.L) < 1e-6) wet++;
    }
    expect(wet).toBeGreaterThan(5);
  });

  test('a river widens into a lake at some nodes: level, its shore above it all round', () => {
    const s: [number, number] = [0, 0];
    let lakes = 0;
    for (const n of land) {
      const r = riverLakeOf(n, SEED);
      if (r <= 0) continue;
      const W = riverLakeLevel(n, SEED);
      // at its centre the water stands at the lake's level, over its bowl
      const c = worldOf(n.x, n.z);
      if (!c) continue;
      const [x, z] = c;
      surfaceAt(x, z, SEED, s);
      expect(s[1]).toBeCloseTo(W, 3);
      expect(s[0]).toBeLessThan(W);
      // walking out from the centre along each ray, the first dry ground stands at the water's level
      for (let k = 0; k < 24; k++) {
        const a = k / 24 * Math.PI * 2;
        let Wlast = W;
        for (let d = 2; d <= 2 * r; d += 2) {
          surfaceAt(x + d * Math.cos(a), z + d * Math.sin(a), SEED, s);
          if (s[1] > s[0] + 0.02) { Wlast = s[1]; continue; }
          expect(s[0]).toBeGreaterThan(Wlast - 0.6);
          break;
        }
      }
      lakes++;
      if (lakes >= 12) break;
    }
    expect(lakes).toBeGreaterThan(5);
  });

  test('rivers reach the sea at its level', () => {
    for (const n of land) {
      const d = downOf(n, SEED);
      if (d && codeOf(d, SEED) === OCEAN) expect(levelOf(d, SEED)).toBe(LANDS.sea);
    }
  });
});

describe('the GPU\'s table of nodes', () => {
  test('packs each node as the CPU sees it (positions relative to the window, levels, codes, profiles)', () => {
    const t = hydroTable(0, 0, SEED);
    expect(t.data.length).toBe(4 + t.n * t.n * HYDRO_STRIDE);
    expect([t.data[0], t.data[1], t.data[2], t.data[3]]).toEqual([t.i0, t.j0, t.n, HYDRO.node]);
    for (const [li, lj] of [[0, 0], [7, 3], [t.n - 1, t.n - 1], [t.n >> 1, t.n >> 1]]) {
      const nd = nodeAt(t.i0 + li, t.j0 + lj, SEED);
      const o = 4 + (lj * t.n + li) * HYDRO_STRIDE;
      expect(t.data[o]).toBeCloseTo(nd.x - t.i0 * HYDRO.node, 2);
      expect(t.data[o + 1]).toBeCloseTo(nd.z - t.j0 * HYDRO.node, 2);
      expect(t.data[o + 3]).toBeCloseTo(levelOf(nd, SEED), 3);
      expect(t.data[o + 4]).toBe(codeOf(nd, SEED));
      expect(t.data[o + 5]).toBe(accOf(nd, SEED));
      const p = profileOf(nd, SEED);
      if (p) expect(t.data[o + 8]).toBeCloseTo(p[0], 3);
    }
  });
});
