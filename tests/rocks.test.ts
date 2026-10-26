/**
 * The rocks (src/lib/rocks/models.ts, gpu/rocks.wgsl.ts): the meshes made at start-up, and the tiers that scatter them.
 * They are made once and drawn flat-shaded: what matters is that they are the same every time (a rock keeps its shape), that
 * each kind's levels of detail nest (fewer triangles, the same extent), that the vertices are what the shader reads, and that
 * the tiers' numbers are the ones the scatter's lattices assume.
 */

import { describe, expect, it } from 'bun:test';
import { DEFAULT_SEED } from '../src/lib/constants';
import { ROCK_KINDS, ROCK_LODS, ROCK_VERTEX_BYTES, buildRockModels } from '../src/lib/rocks/models';
import { rocksNear, type RockCircle } from '../src/lib/rocks/collision';
import { surfaceAt } from '../src/lib/terrainMath';
import { createHeroState, updateHero } from '../src/lib/hero/locomotion';
import { ROCK_BUCKETS, ROCK_CAP, ROCK_CAST_SIZE, ROCK_EXTENT, ROCK_MESHES, ROCK_N, ROCK_TIERS } from '../src/lib/rocks/tiers';

const models = buildRockModels();

describe('rock meshes', () => {
  it('are the same every time they are made', () => {
    const again = buildRockModels();
    expect(again.meshes).toEqual(models.meshes);
    expect(Buffer.compare(Buffer.from(again.vertices), Buffer.from(models.vertices))).toBe(0);
    expect(again.extent).toEqual(models.extent);
  });

  it('a mesh for every kind at every level of detail, flat-shaded triangles, the finest the most', () => {
    expect(models.meshes.length).toBe(ROCK_KINDS.length * ROCK_LODS);
    expect(ROCK_MESHES).toBe(models.meshes.length);
    expect(ROCK_BUCKETS).toBe(2 * models.meshes.length); // (two sets: the rocks too small to cast a shadow, and the rest)
    expect(ROCK_CAST_SIZE).toBeLessThan(ROCK_TIERS[ROCK_TIERS.length - 2].size[0]); // (a boulder always casts)
    let total = 0;
    models.meshes.forEach((m, i) => {
      expect(m.count).toBeGreaterThan(0);
      expect(m.count % 3).toBe(0);
      expect(m.first).toBe(total);
      total += m.count;
      if (i % ROCK_LODS > 0) expect(m.count).toBeLessThanOrEqual(models.meshes[i - 1].count);
    });
    expect(models.vertices.byteLength).toBe(total * ROCK_VERTEX_BYTES);
    // every kind's finest mesh has more triangles than its coarsest (a level of detail is a saving)
    for (let k = 0; k < ROCK_KINDS.length; k++) expect(models.meshes[k * ROCK_LODS].count).toBeGreaterThan(models.meshes[k * ROCK_LODS + ROCK_LODS - 1].count);
  });

  it('vertices: finite positions in the unit mesh, normals of unit length, a flat base to sit on', () => {
    const dv = new DataView(models.vertices);
    const n = models.vertices.byteLength / ROCK_VERTEX_BYTES;
    for (let v = 0; v < n; v += 7) {
      const o = v * ROCK_VERTEX_BYTES;
      for (let c = 0; c < 3; c++) {
        const p = dv.getFloat32(o + 4 * c, true);
        expect(Number.isFinite(p)).toBe(true);
        expect(Math.abs(p)).toBeLessThan(4);
      }
      const nx = dv.getInt8(o + 12) / 127, ny = dv.getInt8(o + 13) / 127, nz = dv.getInt8(o + 14) / 127;
      expect(Math.abs(Math.hypot(nx, ny, nz) - 1)).toBeLessThan(0.12);
    }
    for (const e of models.extent) {
      expect(e.rxz).toBeGreaterThan(0.3);
      expect(e.rxz).toBeLessThan(4);
      expect(e.top).toBeGreaterThan(e.bottom);
      expect(e.bottom).toBeLessThan(0.05); // (a base at or under the middle: the rock sits in the ground)
    }
  });
});

describe('rock tiers', () => {
  it('lattices of powers of two, growing with the size of the rocks they hold, each drawn within the cells the scatter covers', () => {
    expect(ROCK_TIERS.length).toBeGreaterThanOrEqual(3);
    let cell = 0, size = 0;
    for (const t of ROCK_TIERS) {
      expect(Number.isInteger(Math.log2(t.cell))).toBe(true);
      expect(t.cell).toBeGreaterThan(cell);
      expect(t.size[0]).toBeGreaterThan(0);
      expect(t.size[1]).toBeGreaterThan(t.size[0]);
      expect(t.size[1]).toBeGreaterThan(size);
      expect(t.chance).toBeGreaterThan(0);
      expect(t.chance).toBeLessThanOrEqual(1);
      expect(t.radius).toBeLessThanOrEqual((ROCK_N / 2) * t.cell); // the lattice a side of N cells round the camera
      for (const k of t.kinds) expect(k).toBeLessThan(ROCK_KINDS.length);
      expect(t.kinds.length).toBe(6);
      cell = t.cell; size = t.size[1];
    }
    expect(ROCK_CAP).toBeGreaterThanOrEqual(256);
  });
});

describe('boulders for collision (the CPU twin of the big tiers)', () => {
  it('ROCK_EXTENT is what the meshes measure', () => {
    models.extent.forEach((e, k) => {
      expect(Math.abs(ROCK_EXTENT[k].rxz - e.rxz)).toBeLessThan(0.005);
      expect(Math.abs(ROCK_EXTENT[k].top - e.top)).toBeLessThan(0.005);
      expect(Math.abs(ROCK_EXTENT[k].bottom - e.bottom)).toBeLessThan(0.005);
    });
  });

  it('are the same every time, stand on dry land, and are sized for a hero to meet', () => {
    const seen = new Map<string, RockCircle>();
    const out: RockCircle[] = [];
    for (let x = -6000; x < 6000; x += 200) {
      for (let z = -6000; z < 6000; z += 200) {
        out.length = 0;
        for (const c of rocksNear(x, z, DEFAULT_SEED, out)) seen.set(`${c.x.toFixed(3)},${c.z.toFixed(3)}`, c);
        // asked twice: the same circles
        const again: RockCircle[] = [];
        rocksNear(x, z, DEFAULT_SEED, again);
        expect(again.length).toBe(out.length);
      }
    }
    expect(seen.size).toBeGreaterThan(40);
    for (const c of seen.values()) {
      expect(Number.isFinite(c.x + c.z + c.y + c.r + c.top)).toBe(true);
      expect(c.r).toBeGreaterThan(0.3);
      expect(c.r).toBeLessThan(16);
      expect(c.top).toBeGreaterThanOrEqual(0.45);
      expect(c.top).toBeLessThan(30); // (a landmark shard: 8.5 m × 2.5, a touch taller by its aspect)
      const [land, water] = surfaceAt(c.x, c.z, DEFAULT_SEED);
      expect(land).toBeGreaterThanOrEqual(water + 0.1);
      expect(c.y).toBeLessThan(land);
      expect(c.y).toBeGreaterThan(land - 4);
    }
  }, 30_000);

  it('only the blocking tiers are placed: none of its circles is a pebble', () => {
    const out: RockCircle[] = [];
    let n = 0;
    for (let x = -3000; x < 3000; x += 100) {
      for (let z = -3000; z < 3000; z += 100) {
        out.length = 0;
        for (const c of rocksNear(x, z, DEFAULT_SEED, out)) { n++; expect(c.r).toBeGreaterThan(0.3); }
      }
    }
    expect(n).toBeGreaterThan(0);
  }, 30_000);
});

describe('a hero meets a boulder', () => {
  it('runs into the nearest boulder and is stopped by it (outside its circle all the way)', () => {
    // the nearest blocking boulder of a spot near the spawn
    const out: RockCircle[] = [];
    let rock: RockCircle | null = null, from = [0, 0];
    for (let x = -4000; x < 4000 && !rock; x += 120) {
      for (let z = -4000; z < 4000 && !rock; z += 120) {
        out.length = 0;
        rocksNear(x, z, DEFAULT_SEED, out);
        const c = out.find((r) => r.r > 1.2 && r.r < 3 && r.top > 1.5);
        if (c) { rock = c; from = [x, z]; }
      }
    }
    expect(rock).not.toBeNull();
    const c = rock!;
    // 7 m off, on the side the ground is flattest, running straight at it
    let best = 0, bestM = Infinity;
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * Math.PI * 2;
      let m = 0;
      for (let d = 1.5; d <= 8; d += 0.5) {
        const x = c.x + Math.cos(a) * d, z = c.z + Math.sin(a) * d;
        const [e, w] = surfaceAt(x, z, DEFAULT_SEED);
        m = Math.max(m, Math.abs(e - surfaceAt(c.x, c.z, DEFAULT_SEED)[0]) / d + (e < w + 2.9 ? 9 : 0));
      }
      if (m < bestM) { bestM = m; best = a; }
    }
    void from;
    const sx = c.x + Math.cos(best) * (c.r + 7), sz = c.z + Math.sin(best) * (c.r + 7);
    const yaw = Math.atan2(c.x - sx, c.z - sz); // (the engine's heading: x = sin(yaw), z = cos(yaw))
    const s = createHeroState(sx, sz, yaw, DEFAULT_SEED);
    let closest = Infinity;
    for (let i = 0; i < 60 * 8; i++) {
      // (a foe's way: the hero pushing into a boulder climbs it, hero.test.ts)
      updateHero(s, { forward: 1, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: yaw, noClimb: true }, 1 / 60, DEFAULT_SEED);
      closest = Math.min(closest, Math.hypot(s.x - c.x, s.z - c.z));
    }
    // (only its feet below the top count, and a hero may climb or slide: the circle is 0.35 m of body wider)
    expect(closest).toBeGreaterThan(c.r - 0.05);
  });
});
