import { describe, expect, test } from 'bun:test';
import { PLAYER, SPAWN, VALLEY } from '../src/lib/constants';
import { trunksNear } from '../src/lib/forest/collision';
import { buildTreeModels, trunkColliders, VERTEX_BYTES } from '../src/lib/forest/models';
import { createHeroState, firstPersonGround, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { forestAt, plantInCell, RULES, type Plant } from '../src/lib/forest/placement';
import { CELL, LODS, SPECIES, SPECIES_INFO, VARIANTS } from '../src/lib/forest/species';
import { TileWindow } from '../src/lib/forest/tiles';
import { elevation, updatePlayer, type InputState, type PlayerState } from '../src/lib/terrainMath';

const SEED = 1337;
const models = buildTreeModels();
const mesh = (s: number, v: number, l: number) => models.meshes[(s * VARIANTS + v) * LODS + l];

describe('tree models', () => {
  test('every mesh is well formed: indices in range, finite positions, unit normals', () => {
    const f32 = new Float32Array(models.vertices);
    const i8 = new Int8Array(models.vertices);
    expect(models.meshes.length).toBe(SPECIES.length * VARIANTS * LODS);
    for (const m of models.meshes) {
      expect(m.indexCount % 3).toBe(0);
      for (let k = m.firstIndex; k < m.firstIndex + m.indexCount; k++) expect(models.indices[k]).toBeLessThan(m.vertexCount);
      for (let i = 0; i < m.vertexCount; i++) {
        const o = (m.baseVertex + i) * VERTEX_BYTES;
        expect(Number.isFinite(f32[o / 4] + f32[o / 4 + 1] + f32[o / 4 + 2])).toBe(true);
        const n = Math.hypot(i8[o + 12], i8[o + 13], i8[o + 14]) / 127;
        expect(Math.abs(n - 1)).toBeLessThan(0.03);
      }
    }
  });

  test('levels of detail get cheaper and stay within their triangle budgets', () => {
    SPECIES.forEach((_, s) => {
      for (let v = 0; v < VARIANTS; v++) {
        const [a, b, c] = [0, 1, 2].map((l) => mesh(s, v, l).indexCount / 3);
        expect(a).toBeGreaterThan(b);
        expect(b).toBeGreaterThan(c);
        expect(a).toBeLessThanOrEqual(2200);
        expect(b).toBeLessThanOrEqual(520);
        expect(c).toBeLessThanOrEqual(100);
      }
    });
  });

  test('models stand on the ground and fit their species bounds', () => {
    const f32 = new Float32Array(models.vertices);
    SPECIES.forEach((sp, s) => {
      const info = SPECIES_INFO[sp];
      expect(models.bounds[s].radius).toBeLessThanOrEqual(info.radius);
      expect(models.bounds[s].top).toBeLessThanOrEqual(1.2 * info.height);
      expect(models.bounds[s].top).toBeGreaterThan(0.6 * info.height);
      for (let v = 0; v < VARIANTS; v++) {
        const m = mesh(s, v, 0);
        let minY = Infinity;
        for (let i = 0; i < m.vertexCount; i++) minY = Math.min(minY, f32[((m.baseVertex + i) * VERTEX_BYTES) / 4 + 1]);
        expect(minY).toBeGreaterThan(-1);
        expect(minY).toBeLessThan(0);
      }
    });
  });

  test('the same seeds build the same models', () => {
    const again = buildTreeModels();
    expect(Buffer.from(again.vertices).equals(Buffer.from(models.vertices))).toBe(true);
    expect(again.indices).toEqual(models.indices);
  });
});

describe('where trees grow', () => {
  // Sample the land round the spawn: the dominant zone and forest body on a grid.
  const spots: Array<{ x: number; z: number; zone: string; mask: number }> = [];
  for (let z = SPAWN.z - 9000; z <= SPAWN.z + 9000; z += 150) {
    for (let x = SPAWN.x - 9000; x <= SPAWN.x + 9000; x += 150) {
      const f = forestAt(x, z, SEED);
      if (f) spots.push({ x, z, ...f });
    }
  }
  const inZone = (zone: string) => spots.filter((s) => s.zone === zone);
  /** Plants in the 60 m squares round the first few spots of a zone. */
  const plantsIn = (zone: string, n: number) => {
    const out: Plant[] = [];
    let cells = 0;
    for (const s of inZone(zone).filter((_, k) => k % 7 === 0).slice(0, n)) {
      for (let iz = Math.floor((s.z - 30) / CELL); iz < (s.z + 30) / CELL; iz++) {
        for (let ix = Math.floor((s.x - 30) / CELL); ix < (s.x + 30) / CELL; ix++) {
          cells++;
          const p = plantInCell(ix, iz, SEED);
          if (p) out.push(p);
        }
      }
    }
    return { out, ha: (cells * CELL * CELL) / 1e4 };
  };

  test('open land is the rule: forests cover a fifth to a third of mild land, none of savanna or desert', () => {
    const cover = (zone: string) => inZone(zone).filter((s) => s.mask > 0.5).length / Math.max(inZone(zone).length, 1);
    expect(inZone('temperate').length).toBeGreaterThan(1000);
    expect(cover('temperate')).toBeGreaterThan(0.12);
    expect(cover('temperate')).toBeLessThan(0.3);
    expect(cover('autumn')).toBeLessThan(0.38);
    expect(cover('savanna')).toBe(0);
    expect(cover('desert')).toBe(0);
  });

  test('forests are bodies, not speckle: at 50 m a forested spot’s neighbours are mostly forested', () => {
    const step = 50, n = 60;
    const grid: boolean[][] = [];
    for (let j = 0; j < n; j++) {
      grid.push([]);
      for (let i = 0; i < n; i++) grid[j].push((forestAt(SPAWN.x - 3000 + i * step, SPAWN.z + 1500 + j * step, SEED)?.mask ?? 0) > 0.5);
    }
    let forested = 0, edges = 0, same = 0;
    for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
      if (!grid[j][i]) continue;
      forested++;
      for (const t of [grid[j][i + 1], grid[j + 1][i]]) { edges++; if (t) same++; }
    }
    const cover = forested / ((n - 1) * (n - 1));
    expect(cover).toBeGreaterThan(0.05);
    // Compact bodies a few hundred metres across score ~0.7 at this spacing (a disc's rim straddles
    // its edge); a speckle of the same cover would score about the cover itself. Normalised:
    // 0 for speckle, 1 for one solid body.
    expect(same / edges).toBeGreaterThan(0.65);
    expect((same / edges - cover) / (1 - cover)).toBeGreaterThan(0.45);
  });

  test('each climate grows its own trees, in the right amounts', () => {
    const top = (zone: string) => {
      const { out, ha } = plantsIn(zone, 30);
      const trees = out.filter((p) => p.species !== 'bush');
      const count = new Map<string, number>();
      for (const p of trees) count.set(p.species, (count.get(p.species) ?? 0) + 1);
      return { species: [...count.entries()].sort((a, b) => b[1] - a[1])[0]?.[0], perHa: trees.length / ha };
    };
    const temperate = top('temperate'), mountain = top('mountain'), savanna = top('savanna'), desert = top('desert');
    expect(temperate.species).toBe('oak');
    expect(top('autumn').species).toBe('maple');
    expect(mountain.species).toBe('pine');
    expect(savanna.species).toBe('acacia');
    expect(desert.species).toBe('cactus');
    expect(temperate.perHa).toBeGreaterThan(8);
    expect(temperate.perHa).toBeLessThan(45);
    expect(savanna.perHa).toBeLessThan(6);
    expect(desert.perHa).toBeLessThan(5);
  });

  test('lone trees never crowd each other', () => {
    const lone: Plant[] = [];
    for (let iz = -470; iz < -400; iz++) for (let ix = 300; ix < 420; ix++) {
      const p = plantInCell(ix, iz, SEED);
      if (p?.feature === 'lone') lone.push(p);
    }
    expect(lone.length).toBeGreaterThan(5);
    for (const a of lone) for (const b of lone) {
      if (a !== b) expect(Math.hypot(a.x - b.x, a.z - b.z)).toBeGreaterThan(0.29 * RULES.temperate.loneGrid);
    }
  });

  test('never in water, never on cliffs; the same cell always grows the same plant', () => {
    let plants = 0;
    for (let iz = -560; iz < -500; iz++) {
      for (let ix = 280; ix < 400; ix++) {
        const p = plantInCell(ix, iz, SEED);
        expect(plantInCell(ix, iz, SEED)).toEqual(p);
        if (!p) continue;
        plants++;
        const e = elevation(p.x, p.z, SEED, 1);
        expect(e).toBeGreaterThanOrEqual(VALLEY.wLevel + 0.3);
        const slope = Math.hypot(elevation(p.x + 1.5, p.z, SEED, 1) - e, elevation(p.x, p.z + 1.5, SEED, 1) - e) / 1.5;
        expect(slope).toBeLessThan(1.3);
      }
    }
    expect(plants).toBeGreaterThan(50);
  });

  test('ruins stay bare; sky-island tops get their own groves', () => {
    const cells: Array<[number, number]> = [];
    for (let iz = -560; iz < -540; iz++) for (let ix = 300; ix < 340; ix++) cells.push([ix, iz]);
    for (const [ix, iz] of cells) expect(plantInCell(ix, iz, SEED, { onRuin: () => true })).toBeNull();
    const island = cells.map(([ix, iz]) => plantInCell(ix, iz, SEED, { islandTop: () => 600 })).filter((p) => p !== null);
    expect(island.length).toBeGreaterThan(0);
    for (const p of island) expect(p!.feature).toBe('island');
  });
});

describe('tree collision', () => {
  const eye = PLAYER.height + PLAYER.groundBias;
  // A lone oak near the spawn, as the renderer places it (ruins and islands included).
  const lone = (() => {
    const c0 = Math.floor(SPAWN.x / CELL), r0 = Math.floor(SPAWN.z / CELL);
    for (let r = 0; r < 120; r++) for (let dz = -r; dz <= r; dz++) for (let dx = -r; dx <= r; dx++) {
      if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
      const p = plantInCell(c0 + dx, r0 + dz, SEED);
      if (p?.feature === 'lone' && p.species === 'oak') return p;
    }
    throw new Error('no lone oak near the spawn');
  })();
  const trunk = trunksNear(lone.x, lone.z, SEED, []).reduce((a, b) => (Math.hypot(a.x - lone.x, a.z - lone.z) < Math.hypot(b.x - lone.x, b.z - lone.z) ? a : b));

  test('every trunk collider encloses its rendered trunk from knee to shoulder', () => {
    const cs = trunkColliders();
    const f32 = new Float32Array(models.vertices);
    SPECIES.forEach((sp, si) => {
      for (let v = 0; v < VARIANTS; v++) {
        const c = cs[si * VARIANTS + v];
        if (sp === 'bush') { expect(c).toBeNull(); continue; }
        expect(c!.r).toBeGreaterThan(0.15);
        expect(c!.r).toBeLessThan(1.2);
        // The trunk is each mesh's first tube: rings of `sides` vertices from the foot up.
        const m = mesh(si, v, 0);
        const sides = sp === 'cactus' ? 16 : 8;
        const at = (k: number) => { const o = ((m.baseVertex + k) * VERTEX_BYTES) / 4; return [f32[o], f32[o + 1], f32[o + 2]]; };
        const ringY = (i: number) => { let y = 0; for (let k = 0; k < sides; k++) y += at(i * sides + k)[1]; return y / sides; };
        for (const h of [0.3, 0.9, 1.5]) {
          let i = 0;
          while (ringY(i + 1) < h) i++;
          const f = (h - ringY(i)) / (ringY(i + 1) - ringY(i));
          for (let k = 0; k < sides; k++) {
            const a = at(i * sides + k), b = at((i + 1) * sides + k);
            const x = a[0] + (b[0] - a[0]) * f, z = a[2] + (b[2] - a[2]) * f;
            expect(Math.hypot(x - c!.x, z - c!.z)).toBeLessThan(c!.r + 0.05);
          }
        }
      }
    });
  });

  test('trunks stand where the trees are drawn, on the ground under them', () => {
    expect(Math.hypot(trunk.x - lone.x, trunk.z - lone.z)).toBeLessThan(0.6 * lone.scale);
    expect(Math.abs(trunk.y - elevation(lone.x, lone.z, SEED, 0))).toBeLessThan(1e-6);
    expect(trunk.r).toBeGreaterThan(0.4);
  });

  test('the hero runs into a trunk, stops at its bark and slides round it', () => {
    const a = 1.1, from = { x: trunk.x + 9 * Math.sin(a), z: trunk.z + 9 * Math.cos(a) };
    const hero = createHeroState(from.x, from.z, 0, SEED);
    const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };
    for (let i = 0; i < 60; i++) updateHero(hero, idle, 1 / 60, SEED);
    // Aim a little off-centre, so the body slides round the trunk rather than stopping dead.
    const run: HeroInput = { ...idle, forward: 1, camYaw: Math.atan2(trunk.x - hero.x, trunk.z - hero.z) + 0.04 };
    let closest = Infinity, passed = false;
    for (let i = 0; i < 60 * 6; i++) {
      updateHero(hero, run, 1 / 60, SEED);
      const d = Math.hypot(hero.x - trunk.x, hero.z - trunk.z);
      expect(d).toBeGreaterThan(0.35 + trunk.r - 1e-6);
      closest = Math.min(closest, d);
      // Past the trunk: on its far side from where the run began.
      if ((hero.x - trunk.x) * (from.x - trunk.x) + (hero.z - trunk.z) * (from.z - trunk.z) < 0) passed = true;
    }
    expect(closest).toBeLessThan(0.35 + trunk.r + 0.05);
    expect(passed).toBe(true);
  });

  test('first person: the player is stopped by a trunk, and passes over it high above', () => {
    const p: PlayerState = { x: trunk.x - 6, y: elevation(trunk.x - 6, trunk.z, SEED, 0) + eye, z: trunk.z, vx: 0, vy: 0, vz: 0, grounded: false };
    const run: InputState = { forward: 1, strafe: 0, jump: false, sprint: false, yaw: Math.PI / 2 };
    for (let i = 0; i < 60 * 4; i++) {
      updatePlayer(p, run, 1 / 60, SEED, firstPersonGround);
      expect(Math.hypot(p.x - trunk.x, p.z - trunk.z)).toBeGreaterThan(0.35 + trunk.r - 1e-6);
    }
    const high = { x: trunk.x, z: trunk.z, vx: 1, vz: 0, y: 0, vy: 0, grounded: false };
    firstPersonGround(high, trunk.y + trunk.top + 5, SEED);
    expect(high.x).toBe(trunk.x);
    expect(high.z).toBe(trunk.z);
  });

  test('bushes do not block', () => {
    let bushes = 0;
    for (let iz = -470; iz < -420 && bushes < 20; iz++) for (let ix = 330; ix < 400 && bushes < 20; ix++) {
      const p = plantInCell(ix, iz, SEED);
      if (p?.species !== 'bush') continue;
      bushes++;
      for (const t of trunksNear(p.x, p.z, SEED, [])) expect(Math.hypot(t.x - p.x, t.z - p.z)).toBeGreaterThan(0.2);
    }
    expect(bushes).toBeGreaterThan(3);
  });
});

describe('tile window', () => {
  test('the first update fills the window; a step brings in one row; slots never collide', () => {
    const w = new TileWindow(8);
    const all = w.update(10, -3, SEED);
    expect(all.length).toBe(64);
    expect(new Set(all.map((j) => j[2])).size).toBe(64);
    expect(w.update(10, -3, SEED)).toEqual([]);
    const step = w.update(11, -3, SEED);
    expect(step.length).toBe(8);
    for (const [tx, , slot] of step) {
      expect(tx).toBe(11 + 3);
      // The column that left the window (x = 10 − 4) owned these slots.
      expect(all.some(([ox, , os]) => ox === 6 && os === slot)).toBe(true);
    }
    expect(w.update(12, -2, SEED).length).toBe(8 + 8 - 1);
  });

  test('a new seed, a jump or invalidate() regenerate everything', () => {
    const w = new TileWindow(6);
    w.update(0, 0, SEED);
    expect(w.update(0, 0, SEED + 1).length).toBe(36);
    expect(w.update(100, 0, SEED + 1).length).toBe(36);
    w.invalidate();
    expect(w.update(100, 0, SEED + 1).length).toBe(36);
    expect(() => new TileWindow(5)).toThrow();
  });
});
