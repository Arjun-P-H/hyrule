import { describe, expect, test } from 'bun:test';
import { CLOUDS, SPAWN } from '../src/lib/constants';
import { cloudSurface } from '../src/lib/skyworld/clouds';
import { elevation, surfaceAt, waterLevel } from '../src/lib/terrainMath';
import { worldSeed } from '../src/lib/world';

const C = worldSeed(1337, 'clouds');

describe('the cloud sea', () => {
  test('is finite, inside its band, under the lowest island, and never water (20 000 points over 40 km)', () => {
    let k = 1;
    const rnd = () => ((k = Math.imul(k, 48271) % 2147483647) & 0xffffff) / 0x1000000;
    for (let i = 0; i < 20_000; i++) {
      const x = (rnd() - 0.5) * 40_000, z = (rnd() - 0.5) * 40_000;
      const e = cloudSurface(x, z, C, 0);
      expect(e === e).toBe(true);
      expect(e).toBeGreaterThanOrEqual(CLOUDS.base[0]);
      expect(e).toBeLessThanOrEqual(CLOUDS.ceiling);
      expect(waterLevel(x, z, C)).toBeLessThan(e);
    }
  });

  test('is what elevation and surfaceAt return in the cloud world', () => {
    for (let i = 0; i < 100; i++) {
      const x = SPAWN.x + 137 * i, z = SPAWN.z - 91 * i;
      expect(elevation(x, z, C, 0)).toBe(cloudSurface(x, z, C, 0));
      expect(surfaceAt(x, z, C)[0]).toBe(cloudSurface(x, z, C, 0));
    }
  });

  test('rolls: heaped domes over a few hundred metres, mostly gentle enough to walk', () => {
    // along a 4 km line: the height swings by a good part of its band, and few steps are cliffs
    let lo = Infinity, hi = -Infinity, steep = 0;
    const N = 4000;
    for (let i = 0; i < N; i++) {
      const x = SPAWN.x + i, z = SPAWN.z + 0.37 * i;
      const e = cloudSurface(x, z, C, 0), e1 = cloudSurface(x + 1, z + 0.37, C, 0);
      lo = Math.min(lo, e); hi = Math.max(hi, e);
      if (Math.abs(e1 - e) / Math.hypot(1, 0.37) > 1.2) steep++;
    }
    expect(hi - lo).toBeGreaterThan(100);
    expect(steep / N).toBeLessThan(0.05);
  });

  test('the land is untouched', () => {
    const before = [[2648, -4031, 204.3460992193559], [0, 0, 135.62041136864946], [12345, -6789, 164.5732450385152],
      [-20000, 15000, 61.74206385953626], [40000, 3000, 40.89764194949724]];
    for (const [x, z, e] of before) expect(elevation(x, z, 1337, 0)).toBe(e);
  });
});
