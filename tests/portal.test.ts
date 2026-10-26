import { describe, expect, test } from 'bun:test';
import { CLOUDS, PORTAL } from '../src/lib/constants';
import { exitOf, portalCrossed, portalOf, spawnPoint } from '../src/lib/portal';
import { portalSite, ruinSiteAt, ruinSitesNear } from '../src/lib/ruins';
import { elevation, waterLevel } from '../src/lib/terrainMath';
import { worldSeed } from '../src/lib/world';

describe('the portal: where it stands', () => {
  test("each world's ring stands on its own ground", () => {
    for (const s of [1337, 42, 999_999]) {
      const p = portalOf('land', s);
      expect(Math.abs(p.y - elevation(p.x, p.z, s, 0))).toBeLessThan(0.01);
      expect(elevation(p.x, p.z, s, 0)).toBeGreaterThan(waterLevel(p.x, p.z, s) + 1);
      const q = portalOf('clouds', s);
      expect(Math.abs(q.y - elevation(q.x, q.z, worldSeed(s, 'clouds'), 0))).toBeLessThan(0.01);
      expect(q.y).toBeLessThanOrEqual(CLOUDS.ceiling); // on the cloud sea
    }
  });

  test('you come out 4 m in front of the ring, facing away from it', () => {
    for (const w of ['land', 'clouds'] as const) {
      const p = portalOf(w, 42), e = exitOf(w, 42);
      expect(Math.hypot(e.x - p.x, e.z - p.z)).toBeCloseTo(4, 1);
      expect(e.x - p.x).toBeCloseTo(4 * Math.sin(p.yaw), 3);
      expect(e.z - p.z).toBeCloseTo(4 * Math.cos(p.yaw), 3);
      expect(e.yaw).toBe(p.yaw);
    }
  });

  test('respawning in the cloud world brings you back at its ring, for any seed', () => {
    for (const s of [1337, 42]) {
      const sp = spawnPoint('clouds', s), e = exitOf('clouds', s);
      expect([sp.x, sp.z, sp.yaw]).toEqual([e.x, e.z, e.yaw]);
    }
  });
});

describe('the portal: the ring and passing through it', () => {
  const p = portalOf('land', 1337);
  const nx = Math.sin(p.yaw), nz = Math.cos(p.yaw), ux = Math.cos(p.yaw), uz = -Math.sin(p.yaw);
  const y = p.y + 0.05; // (the hero's feet on the ground there)

  test('only a pass through the opening counts', () => {
    // through the middle, either way
    expect(portalCrossed(p, p.x - nx, y, p.z - nz, p.x + nx, y, p.z + nz)).toBe(true);
    expect(portalCrossed(p, p.x + nx, y, p.z + nz, p.x - nx, y, p.z - nz)).toBe(true);
    // beside it (5 m along the ring's plane), over it, and a step that stops short of it
    expect(portalCrossed(p, p.x + 5 * ux - nx, y, p.z + 5 * uz - nz, p.x + 5 * ux + nx, y, p.z + 5 * uz + nz)).toBe(false);
    expect(portalCrossed(p, p.x - nx, y + 2 * PORTAL.radius + 3, p.z - nz, p.x + nx, y + 2 * PORTAL.radius + 3, p.z + nz)).toBe(false);
    expect(portalCrossed(p, p.x - 2 * nx, y, p.z - 2 * nz, p.x - 0.5 * nx, y, p.z - 0.5 * nz)).toBe(false);
  });

  test('the ring is shrine stone with its opening clear to walk through', () => {
    const s = portalSite('land', 1337);
    expect(s.shrine).toBe(true);
    expect(s.boxes.length).toBe(PORTAL.segments - 3);
    expect(s.boxes.every((b) => b.material === 2)).toBe(true);
    // nothing in the opening from the feet up to the head, along the way through
    for (const b of s.boxes) {
      const bottom = b.y - b.hy, top = b.y + b.hy;
      const nearMiddle = Math.hypot(b.x - p.x, b.z - p.z) < 1.2;
      expect(nearMiddle && bottom < p.y + 2 && top > p.y).toBe(false);
    }
  });

  test('only the inactive cloud world has a ring (the land none: the cloud world is reached by the settings)', () => {
    const q = portalOf('clouds', 1337), seed = worldSeed(1337, 'clouds');
    expect(ruinSitesNear(q.x, q.z, 50, seed, 99).some((s) => s.boxes.length === PORTAL.segments - 3 && Math.hypot(s.x - q.x, s.z - q.z) < 0.01)).toBe(true);
    expect(ruinSiteAt(q.x + 3 * Math.cos(q.yaw), q.z - 3 * Math.sin(q.yaw), seed)?.boxes.length).toBe(PORTAL.segments - 3);
    const l = portalOf('land', 1337);
    expect(ruinSitesNear(l.x, l.z, 50, 1337, 99).some((s) => s.boxes.length === PORTAL.segments - 3)).toBe(false);
  });
});
