import { describe, expect, test } from 'bun:test';
import { boxInView, RING_BLOCK, ringBlockOrder, ringIndices, ringLayout } from '../src/lib/gpu/rings';
import { worldWGSL } from '../src/lib/gpu/worldgen.wgsl';
import { gameVariant } from '../src/lib/gpu/engineGpu';
import { rasterShader } from '../src/lib/gpu/terrain.wgsl';

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

describe('ring index buffers', () => {
  test('full and holed grids have the expected triangles, all in range', () => {
    const k = 8, side = 2 * k + 1;
    const full = ringIndices(k, null);
    expect(full.length).toBe(4 * k * k * 6);
    const holed = ringIndices(k, [k / 2 + 1, k / 2]);
    expect(holed.length).toBe(3 * k * k * 6);
    for (const i of full) expect(i).toBeLessThan(side * side);
    // No triangle of the holed grid lies inside the hole.
    for (let t = 0; t < holed.length; t += 3) {
      const cx = (holed[t] % side + holed[t + 1] % side + holed[t + 2] % side) / 3;
      const cz = (Math.floor(holed[t] / side) + Math.floor(holed[t + 1] / side) + Math.floor(holed[t + 2] / side)) / 3;
      const inside = cx > k / 2 + 1 && cx < k / 2 + 1 + k && cz > k / 2 && cz < k / 2 + k;
      expect(inside).toBe(false);
    }
  });

  test('collapsing odd vertices onto their lower even neighbours gives the coarse triangulation', () => {
    // The CDLOD morph at μ = 1: fine ring vertices land on the coarse lattice
    // and the surviving triangles must be exactly the coarse grid's.
    const k = 8, side = 2 * k + 1;
    const fine = ringIndices(k, null);
    const key = (a: number[], b: number[], c: number[]) => [a, b, c].map((v) => v.join(',')).sort().join('|');
    const collapsed = new Set<string>();
    for (let t = 0; t < fine.length; t += 3) {
      const v = [0, 1, 2].map((j) => {
        const i = fine[t + j];
        const x = i % side, z = Math.floor(i / side);
        return [x - (x & 1), z - (z & 1)];
      });
      const area = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]);
      if (area !== 0) collapsed.add(key(v[0], v[1], v[2]));
    }
    // Coarse grid: cells of 2 fine units, same diagonal.
    const coarse = new Set<string>();
    for (let cz = 0; cz < 2 * k; cz += 2) {
      for (let cx = 0; cx < 2 * k; cx += 2) {
        coarse.add(key([cx, cz], [cx + 2, cz + 2], [cx + 2, cz]));
        coarse.add(key([cx, cz], [cx, cz + 2], [cx + 2, cz + 2]));
      }
    }
    expect(collapsed).toEqual(coarse);
  });
});

describe('ring layout', () => {
  test('each ring’s hole is exactly the next finer ring, anywhere in the world', () => {
    const rnd = lcg(3);
    const k = 64, s0 = 0.5;
    for (const stride of [1, 2]) {
      for (let n = 0; n < 500; n++) {
        const x = (rnd() - 0.5) * 4e6, z = (rnd() - 0.5) * 4e6;
        const L = ringLayout(x, z, s0, k, stride, 8);
        for (let r = 1; r < L.length; r++) {
          const fine = L[r - 1], ring = L[r];
          // finer ring extent (level-0 units) vs this ring's hole
          const holeX0 = ring.corner[0] + (k / 2 + ring.hole[0]) * ring.step;
          const holeZ0 = ring.corner[1] + (k / 2 + ring.hole[1]) * ring.step;
          expect(holeX0).toBe(fine.corner[0]);
          expect(holeZ0).toBe(fine.corner[1]);
          expect(holeX0 + k * ring.step).toBe(fine.corner[0] + 2 * k * fine.step);
          expect(holeZ0 + k * ring.step).toBe(fine.corner[1] + 2 * k * fine.step);
        }
        // The camera stays within two cells of every ring's centre.
        for (const ring of L) {
          expect(Math.abs(x / s0 - ring.center[0])).toBeLessThanOrEqual(2 * ring.step);
          expect(Math.abs(z / s0 - ring.center[1])).toBeLessThanOrEqual(2 * ring.step);
        }
      }
    }
  });
});

describe('ring culling blocks', () => {
  test('the block order is a permutation of the blocks, round the ring centre by angle', () => {
    const k = 128, nb = (2 * k) / RING_BLOCK;
    const { order, slot } = ringBlockOrder(k);
    expect(order.length).toBe(nb * nb);
    expect([...order].sort((a, b) => a - b)).toEqual(Array.from({ length: nb * nb }, (_, i) => i));
    for (let s = 0; s < order.length; s++) expect(slot[order[s]]).toBe(s);
    const angle = (b: number) => Math.atan2((Math.floor(b / nb) + 0.5) * RING_BLOCK - k, ((b % nb) + 0.5) * RING_BLOCK - k);
    for (let s = 1; s < order.length; s++) expect(angle(order[s])).toBeGreaterThanOrEqual(angle(order[s - 1]));
  });

  test('boxInView keeps every box a point of the view lies in, and drops boxes behind or beside it', () => {
    const rnd = lcg(7);
    for (let c = 0; c < 200; c++) {
      // A random camera basis (yaw, pitch) as packFrame lays it out.
      const yaw = (rnd() - 0.5) * 6.28, pitch = (rnd() - 0.5) * 1.6;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const fx = sy * cp, fy = sp, fz = cy * cp, rx = cy, rz = -sy;
      const F = [rx, 0, rz, 0, fy * rz, fz * rx - fx * rz, -fy * rx, 0, fx, fy, fz, 0];
      const tx = 1.27, ty = 0.7;
      // A point inside the view (view coordinates within the frustum), and a box round it.
      const vz = 5 + rnd() * 500, vx = (rnd() * 2 - 1) * tx * vz * 0.99, vy = (rnd() * 2 - 1) * ty * vz * 0.99;
      const px = vx * F[0] + vy * F[4] + vz * F[8], py = vx * F[1] + vy * F[5] + vz * F[9], pz = vx * F[2] + vy * F[6] + vz * F[10];
      const r = rnd() * 3;
      expect(boxInView(F, tx, ty, px - r, py - r, pz - r, px + r, py + r, pz + r)).toBe(true);
      // The same point behind the camera, and far beside the view: dropped.
      expect(boxInView(F, tx, ty, -px - 1, -py - 1, -pz - 1, -px + 1, -py + 1, -pz + 1)).toBe(false);
      const bx = 3 * tx * vz * F[0] + vz * F[8], by = 3 * tx * vz * F[1] + vz * F[9], bz = 3 * tx * vz * F[2] + vz * F[10];
      expect(boxInView(F, tx, ty, bx - 1, by - 1, bz - 1, bx + 1, by + 1, bz + 1)).toBe(false);
    }
  });
});

describe('the world is compiled in', () => {
  test('each world\'s shader says which it is; each world\'s renderer draws only its own things', () => {
    expect(worldWGSL('clouds')).toContain('const CLOUD_WORLD: bool = true;');
    expect(worldWGSL('land')).toContain('const CLOUD_WORLD: bool = false;');
    expect(gameVariant('cycle', 'land')).toMatchObject({ islands: true, world: 'land', sky: 'cycle' });
    expect(gameVariant('cycle', 'clouds')).toMatchObject({ islands: true, forest: null, grass: null, rocks: false, world: 'clouds' });
  });

  test('the cloud world draws its puffs; the land none', () => {
    expect(gameVariant('cycle', 'clouds')).toMatchObject({ puffs: true });
    expect((gameVariant('cycle', 'land') as { puffs?: boolean }).puffs).toBeFalsy();
    const opts = { n: 512, levels: 12, blend: 0.25, k: 128, rings: 12, world: 'clouds' as const };
    expect(rasterShader({ ...opts, puffs: true })).toContain('fn vsPuff');
    expect(rasterShader(opts)).not.toContain('fn vsPuff');
  });
});
