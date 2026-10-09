import { describe, expect, test } from 'bun:test';
import { FLY_CEILING, HERO, MAX_WORLD_H, PLAYER, SKY_REALM as S, VALLEY } from '../src/lib/constants';
import { createHeroState, placeHero, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { archGround, archY, type Arch } from '../src/lib/skyworld/arches';
import { archipelagoInCell, BRIDGE_SLOPE, isleInCell, type Archipelago } from '../src/lib/skyworld/archipelago';
import { SKY_BIOMES, skyClimate, SKY_BIOME_INFO } from '../src/lib/skyworld/biomes';
import { skyPlantInCell } from '../src/lib/skyworld/flora';
import { bottomAt, CHANNEL_REACH, insideBy, islandRho, lakeDist, maxDrop, outline, outlineMax, streamNear, topAt, type SkyIsland } from '../src/lib/skyworld/island';
import { islandAt, realmNear, skyGround, skyWater } from '../src/lib/skyworld/realm';
import { elevation } from '../src/lib/terrainMath';

const SEED = 1337;

/** Archipelagos of the cells within n of the origin. */
function archipelagos(n: number): Archipelago[] {
  const out: Archipelago[] = [];
  for (let iz = -n; iz < n; iz++) for (let ix = -n; ix < n; ix++) {
    const a = archipelagoInCell(ix, iz, SEED);
    if (a) out.push(a);
  }
  return out;
}
const ARCHS = archipelagos(3);
const ISLANDS = ARCHS.flatMap((a) => a.islands);
const WITH_FALLS = ISLANDS.filter((i) => i.fall);

/** A point at normalised radius rho and bearing th of an island. */
const at = (isl: SkyIsland, rho: number, th: number): [number, number] => {
  const r = rho * isl.R * outline(isl, th);
  return [isl.cx + r * Math.cos(th), isl.cz + r * Math.sin(th)];
};

describe('the sky realm: layout', () => {
  test('archipelagos are deterministic, about the configured share of cells, whole inside their cell', () => {
    expect(ARCHS.length / 36).toBeGreaterThan(S.archProbability - 0.25);
    expect(ARCHS.length / 36).toBeLessThan(S.archProbability + 0.25);
    expect(archipelagoInCell(ARCHS[0].ix, ARCHS[0].iz, SEED)).toBe(ARCHS[0]);
    for (const a of ARCHS) {
      expect(a.islands[0].kind).toBe('hub');
      for (const isl of a.islands) {
        expect(Math.hypot(isl.cx - a.cx, isl.cz - a.cz) + isl.R * outlineMax(isl)).toBeLessThanOrEqual(a.reach + 1e-6);
      }
      expect(a.cx - a.reach).toBeGreaterThanOrEqual(a.ix * S.archCell);
      expect(a.cx + a.reach).toBeLessThanOrEqual((a.ix + 1) * S.archCell);
      expect(a.cz - a.reach).toBeGreaterThanOrEqual(a.iz * S.archCell);
      expect(a.cz + a.reach).toBeLessThanOrEqual((a.iz + 1) * S.archCell);
    }
  });

  test('every Eyrie has its main island first, others round it, and (but the Fallen) bridges or arches between them', () => {
    for (const a of ARCHS) {
      expect(a.islands[0].kind).toBe('hub');
      expect(a.islands.filter((i) => i.kind !== 'debris').length).toBeGreaterThanOrEqual(3);
      expect(a.roles.length).toBe(a.islands.length);
      if (a.cls !== 'fallen') expect(a.arches.length).toBeGreaterThan(0);
      else expect(a.arches.length).toBe(0);
    }
  });

  test('islands never overlap: open sky between every two outlines', () => {
    for (const a of ARCHS) {
      for (let i = 0; i < a.islands.length; i++) for (let j = i + 1; j < a.islands.length; j++) {
        const p = a.islands[i], q = a.islands[j];
        const gap = p.kind === 'debris' || q.kind === 'debris' ? 25 : 30;
        expect(Math.hypot(p.cx - q.cx, p.cz - q.cz)).toBeGreaterThan(p.R * outlineMax(p) + q.R * outlineMax(q) + gap - 1e-6);
      }
    }
  });

  test('lone isles keep out of every archipelago and inside their cell', () => {
    let n = 0;
    for (let iz = -12; iz < 12; iz++) for (let ix = -12; ix < 12; ix++) {
      const isl = isleInCell(ix, iz, SEED);
      if (!isl) continue;
      n++;
      expect(isl.kind).toBe('lone');
      const reach = isl.R * outlineMax(isl);
      expect(isl.cx - reach).toBeGreaterThan(ix * S.isleCell);
      expect(isl.cx + reach).toBeLessThan((ix + 1) * S.isleCell);
      const a = archipelagoInCell(Math.floor(isl.cx / S.archCell), Math.floor(isl.cz / S.archCell), SEED);
      if (a) expect(Math.hypot(isl.cx - a.cx, isl.cz - a.cz)).toBeGreaterThan(a.reach + reach);
    }
    expect(n).toBeGreaterThan(40);
  });

  test('realmNear returns what lies in range, nearest first', () => {
    const { archipelagos: as, isles } = realmNear(2016, -2592, 12000, SEED);
    expect(as.length).toBeGreaterThan(0);
    let last = 0;
    for (const a of as) {
      const d = Math.max(0, Math.hypot(a.cx - 2016, a.cz + 2592) - a.reach);
      expect(d).toBeLessThanOrEqual(12000);
      expect(d).toBeGreaterThanOrEqual(last);
      last = d;
    }
    expect(isles.length).toBeGreaterThan(0);
  });
});

describe('the sky realm: island shapes', () => {
  test('every underside clears the highest peak; the top stays above it; they meet at the rim', () => {
    expect(S.floor).toBeGreaterThan(MAX_WORLD_H);
    for (const isl of ISLANDS) {
      expect(isl.y - maxDrop(isl)).toBeGreaterThanOrEqual(S.floor - 1e-6);
      for (let k = 0; k < 48; k++) {
        const th = (k / 48) * Math.PI * 2;
        for (const rho of [0, 0.25, 0.5, 0.75, 0.9, 0.97, 1]) {
          const [x, z] = at(isl, rho, th);
          const top = topAt(isl, x, z), bot = bottomAt(isl, x, z);
          expect(bot).toBeGreaterThanOrEqual(S.floor);
          if (rho === 1) expect(Math.abs(top - bot)).toBeLessThan(1e-6);
          else expect(top).toBeGreaterThanOrEqual(bot);
        }
      }
    }
  });

  test('the stream notches the rim and the underside follows it down: closed at the lip', () => {
    for (const isl of WITH_FALLS) {
      const lip = isl.stream!.pts[isl.stream!.pts.length - 1];
      const x = isl.cx + lip[0], z = isl.cz + lip[1];
      expect(Math.abs(islandRho(isl, x, z) - 1)).toBeLessThan(1e-6);
      expect(topAt(isl, x, z)).toBeLessThan(isl.y - S.rimDrop - 1);
      expect(Math.abs(topAt(isl, x, z) - bottomAt(isl, x, z))).toBeLessThan(1e-6);
    }
  });

  test('insideBy is the signed distance to the outline along the bearing', () => {
    const isl = ARCHS[0].islands[0];
    const [x, z] = at(isl, 0.9, 1.1);
    expect(insideBy(isl, x, z)).toBeCloseTo(0.1 * isl.R * outline(isl, 1.1), 6);
    const [ox, oz] = at(isl, 1.2, 1.1);
    expect(insideBy(isl, ox, oz)).toBeCloseTo(-0.2 * isl.R * outline(isl, 1.1), 6);
  });
});

describe('the sky realm: water with a source', () => {
  test('many islands keep water; every fall is fed by a stream from a lake', () => {
    expect(WITH_FALLS.length).toBeGreaterThan(ARCHS.length);
    for (const isl of ISLANDS) {
      if (isl.fall) expect(isl.stream && isl.lake && !isl.lake.frozen).toBeTruthy();
      if (isl.stream) expect(isl.lake).toBeTruthy();
    }
  });

  test('a lake never spills: its whole shore stands above its level, its bed below', () => {
    for (const isl of ISLANDS) {
      const L = isl.lake;
      if (!L) continue;
      for (let k = 0; k < 360; k++) {
        const a = (k / 360) * Math.PI * 2;
        for (const d of [1.03, 1.1, 1.2, 1.3]) {
          // a point at normalised distance d along the lake's own direction a
          let lo = 0, hi = 3;
          for (let it = 0; it < 40; it++) {
            const m = (lo + hi) / 2;
            const u = Math.cos(a) * L.r * m, v = Math.sin(a) * L.r * m * L.aspect;
            const qx = L.x + u * L.cos - v * L.sin, qz = L.z + u * L.sin + v * L.cos;
            if (lakeDist(L, qx, qz) < d) lo = m; else hi = m;
          }
          const u = Math.cos(a) * L.r * lo, v = Math.sin(a) * L.r * lo * L.aspect;
          const qx = L.x + u * L.cos - v * L.sin, qz = L.z + u * L.sin + v * L.cos;
          // (the stream's channel leaves through the shore: its outlet is the one way out)
          if (isl.stream && streamNear(isl.stream, qx, qz).s < CHANNEL_REACH * isl.stream.pts[0][3]) continue;
          expect(topAt(isl, isl.cx + qx, isl.cz + qz)).toBeGreaterThan(L.level);
        }
      }
      expect(topAt(isl, isl.cx + L.x, isl.cz + L.z)).toBeLessThan(L.level - 0.5);
    }
  });

  test('a stream runs downhill from under the lake to the lip, in its channel all the way', () => {
    for (const isl of WITH_FALLS) {
      const P = isl.stream!.pts;
      expect(P[0][2]).toBeLessThan(isl.lake!.level);
      for (let i = 1; i < P.length; i++) expect(P[i][2]).toBeLessThan(P[i - 1][2]);
      // Along the axis the top is the bed; the water's surface is below the banks either side.
      for (let i = 0; i + 1 < P.length; i++) {
        for (const f of [0, 0.3, 0.7]) {
          const qx = P[i][0] + (P[i + 1][0] - P[i][0]) * f, qz = P[i][1] + (P[i + 1][1] - P[i][1]) * f;
          const bed = P[i][2] + (P[i + 1][2] - P[i][2]) * f;
          expect(topAt(isl, isl.cx + qx, isl.cz + qz)).toBeLessThanOrEqual(bed + 0.02);
          const nx = -(P[i + 1][1] - P[i][1]), nz = P[i + 1][0] - P[i][0];
          const nl = Math.hypot(nx, nz);
          for (const side of [-1, 1]) {
            const bx = isl.cx + qx + (side * nx / nl) * 1.3 * P[i][3], bz = isl.cz + qz + (side * nz / nl) * 1.3 * P[i][3];
            // (by the outlet the bank may be the lake itself; on a bend, the bank of the stretch it is nearest)
            if (islandRho(isl, bx, bz) > 0.995 || lakeDist(isl.lake!, bx - isl.cx, bz - isl.cz) < 1.35) continue;
            const n = streamNear(isl.stream!, bx - isl.cx, bz - isl.cz);
            expect(topAt(isl, bx, bz)).toBeGreaterThan(n.bed + 0.45);
          }
        }
      }
    }
  });

  test('a fall turns to mist at most half way down to the land below: none reaches a mountain', () => {
    for (const isl of WITH_FALLS) {
      const F = isl.fall!;
      const land = Math.max(elevation(F.x + F.dirX * 25, F.z + F.dirZ * 25, SEED, 0), VALLEY.wLevel);
      expect(F.reach).toBeLessThanOrEqual(0.5 * (F.y - land) + 1e-6);
      expect(F.y - F.reach).toBeGreaterThan(MAX_WORLD_H);
      expect(Math.hypot(F.dirX, F.dirZ)).toBeCloseTo(1, 6);
      // It leaves outward, over the rim.
      expect((F.x - isl.cx) * F.dirX + (F.z - isl.cz) * F.dirZ).toBeGreaterThan(0);
    }
  });

  test('frozen lakes are ice: no stream, no fall', () => {
    for (const isl of ISLANDS) if (isl.lake?.frozen) expect(isl.stream || isl.fall).toBeNull();
  });
});

describe('the sky realm: arches', () => {
  const arches: Array<{ A: Arch; a: Archipelago }> = ARCHS.flatMap((a) => a.arches.map((A) => ({ A, a })));

  test('decks are walkable and root on the islands they join', () => {
    expect(arches.length).toBeGreaterThan(5);
    for (const { A, a } of arches) {
      // (natural arches as before; the Aelari's bridges climb steeper, by their steps)
      expect(Math.abs(A.by - A.ay)).toBeLessThanOrEqual((A.kind === 'arch' ? S.linkSlope : BRIDGE_SLOPE) * A.len + 1e-6);
      for (let k = 0; k < 64; k++) {
        const t0 = k / 64, t1 = (k + 1) / 64;
        expect(Math.abs(archY(A, t1) - archY(A, t0)) / (A.len / 64)).toBeLessThan(0.5);
      }
      const from = a.islands[A.from], to = a.islands[A.to];
      expect(islandRho(from, A.ax, A.az)).toBeLessThan(1);
      expect(islandRho(to, A.bx, A.bz)).toBeLessThan(1);
      // (an arch's deck roots 0.3 m into the ground; a bridge's abutment stands 0.15 m on it)
      const off = A.kind === 'arch' ? 0.3 : -0.15;
      expect(Math.abs(topAt(from, A.ax, A.az) - (A.ay + off))).toBeLessThan(1e-9);
      expect(Math.abs(topAt(to, A.bx, A.bz) - (A.by + off))).toBeLessThan(1e-9);
    }
  });

  test('the deck is found from above and nowhere off it', () => {
    const { A } = arches.find((x) => x.A.kind === 'arch')!;
    const mx = (A.ax + A.bx) / 2 - A.dirZ * A.bow, mz = (A.az + A.bz) / 2 + A.dirX * A.bow;
    const y = archGround(A, mx, mz);
    expect(y).toBeCloseTo(archY(A, 0.5) + 0.3, 6);
    expect(skyGround(mx, y + 3, mz, SEED)).toBeCloseTo(y, 6);
    expect(Number.isNaN(skyGround(mx, y - 30, mz, SEED))).toBe(true);
    expect(Number.isNaN(archGround(A, mx - A.dirZ * A.wMid, mz + A.dirX * A.wMid))).toBe(true);
  });
});

describe('the sky realm: biomes of its own', () => {
  test('every sky biome occurs, and each follows its climate', () => {
    const all = [...ISLANDS, ...Array.from({ length: 400 }, (_, k) => isleInCell(k % 20 - 10, Math.floor(k / 20) - 10, SEED)).filter((i) => i !== null)];
    const seen = new Set(all.map((i) => i.biome));
    for (const b of SKY_BIOMES) expect(seen.has(b)).toBe(true);
    const mean = (b: string, f: (i: SkyIsland) => number) => {
      const xs = all.filter((i) => i.biome === b).map(f);
      return xs.reduce((p, q) => p + q, 0) / xs.length;
    };
    // Frost up high; cloud forest with its roots in the deck's mist; crags on small, exposed isles.
    expect(mean('frost', (i) => i.y)).toBeGreaterThan(mean('meadow', (i) => i.y));
    expect(mean('mistwood', (i) => skyClimate(i.y, i.y - maxDrop(i), i.R).cloud)).toBeGreaterThan(mean('meadow', (i) => skyClimate(i.y, i.y - maxDrop(i), i.R).cloud));
    expect(mean('crags', (i) => i.R)).toBeLessThan(mean('meadow', (i) => i.R));
  });

  test('the sky grows its own flora: sky palettes, nothing in the water or near the rim', () => {
    const isl = ISLANDS.find((i) => i.kind === 'hub' && i.lake)!;
    const plants = [];
    const c0 = Math.floor((isl.cx - isl.R) / 6), c1 = Math.floor((isl.cx + isl.R) / 6);
    const r0 = Math.floor((isl.cz - isl.R) / 6), r1 = Math.floor((isl.cz + isl.R) / 6);
    for (let iz = r0; iz <= r1; iz += 2) for (let ix = c0; ix <= c1; ix += 2) {
      const p = skyPlantInCell(ix, iz, SEED);
      if (!p) continue;
      plants.push(p);
      expect(skyPlantInCell(ix, iz, SEED)).toEqual(p);
      const owner = islandAt(p.x, p.z, SEED)!;
      expect(islandRho(owner, p.x, p.z)).toBeLessThanOrEqual(0.92);
      expect(p.y).toBeCloseTo(topAt(owner, p.x, p.z), 9);
      if (owner.lake) expect(lakeDist(owner.lake, p.x - owner.cx, p.z - owner.cz)).toBeGreaterThanOrEqual(1.15);
      const kinds = SKY_BIOME_INFO[owner.biome].trees.kinds.map((k) => k.species);
      expect([...kinds, 'bush']).toContain(p.species);
    }
    expect(plants.length).toBeGreaterThan(30);
  });
});

describe('the hero in the sky realm', () => {
  test('fly mode climbs past every island: its ceiling clears the highest top', () => {
    const tops = [...ISLANDS, ...Array.from({ length: 400 }, (_, k) => isleInCell(k % 20 - 10, Math.floor(k / 20) - 10, SEED)).filter((i) => i !== null)]
      .map((i) => i.y + 3 * i.hill + 10);
    expect(Math.max(...tops)).toBeLessThan(S.topMax + 200);
    expect(FLY_CEILING.webgpu).toBeGreaterThan(S.topMax + 400);
  });

  const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };

  test('lands on a hub, runs to its lake and swims in it', () => {
    const isl = ISLANDS.find((i) => i.kind === 'hub' && i.lake && !i.lake.frozen)!;
    const L = isl.lake!;
    const lx = isl.cx + L.x, lz = isl.cz + L.z;
    // Start 3 lake radii from the middle, on dry land, facing the water.
    const sx = lx + 3 * L.r * L.cos, sz = lz + 3 * L.r * L.sin;
    const s = createHeroState(sx, sz, 0, SEED);
    placeHero(s, sx, topAt(isl, sx, sz) + 15, sz, 0, 0, 0, 0, SEED);
    for (let i = 0; i < 120; i++) updateHero(s, idle, 1 / 60, SEED);
    expect(s.move).toBe('ground');
    expect(Math.abs(s.y - (skyGround(s.x, s.y, s.z, SEED) + PLAYER.groundBias))).toBeLessThan(0.05);
    const yaw = Math.atan2(lx - s.x, lz - s.z);
    const run: HeroInput = { ...idle, forward: 1, camYaw: yaw };
    let swam = false;
    for (let i = 0; i < 60 * 90 && !swam; i++) {
      updateHero(s, run, 1 / 60, SEED);
      if (s.move === 'swim') swam = true;
    }
    expect(swam).toBe(true);
    expect(s.water).toBeCloseTo(L.level, 6);
    expect(skyWater(s.x, s.y, s.z, SEED)).toBeCloseTo(L.level, 6);
    // Buoyancy settles the hero at the float depth under the lake's surface.
    for (let i = 0; i < 120; i++) updateHero(s, idle, 1 / 60, SEED);
    expect(Math.abs(s.y - (L.level - HERO.swimFloat))).toBeLessThan(0.1);
  });

  test('crosses an arch from the hub to a satellite, on its deck all the way', () => {
    const { A, a } = ARCHS.flatMap((ar) => ar.arches.map((A2) => ({ A: A2, a: ar }))).find(({ A: A2 }) => A2.from === 0)!;
    const s = createHeroState(A.ax, A.az, 0, SEED);
    placeHero(s, A.ax, topAt(a.islands[0], A.ax, A.az) + 2, A.az, 0, 0, 0, 0, SEED);
    for (let i = 0; i < 60; i++) updateHero(s, idle, 1 / 60, SEED);
    const run: HeroInput = { ...idle, forward: 1, camYaw: Math.atan2(A.dirX, A.dirZ) };
    let fell = false;
    for (let i = 0; i < 60 * 240; i++) {
      // Steer back onto the axis (the deck bows sideways a little).
      const px = s.x - A.ax, pz = s.z - A.az;
      const t = (px * A.dirX + pz * A.dirZ) / A.len;
      if (t > 1.02) break;
      const side = -px * A.dirZ + pz * A.dirX - A.bow * Math.sin(Math.PI * Math.min(1, Math.max(0, t)));
      run.camYaw = Math.atan2(A.dirX, A.dirZ) + Math.max(-0.4, Math.min(0.4, side * 0.05));
      updateHero(s, run, 1 / 60, SEED);
      if (s.y < Math.min(A.ay, A.by) - 20) fell = true;
    }
    expect(fell).toBe(false);
    expect(islandAt(s.x, s.z, SEED)).toBe(a.islands[A.to]);
    expect(s.move).toBe('ground');
  });

  test('runs off the rim and falls toward the land far below', () => {
    const isl = ISLANDS.find((i) => i.kind === 'satellite')!;
    const s = createHeroState(isl.cx, isl.cz, 0, SEED);
    placeHero(s, isl.cx, topAt(isl, isl.cx, isl.cz) + 1, isl.cz, 0, 0, 0, 0, SEED);
    for (let i = 0; i < 60; i++) updateHero(s, idle, 1 / 60, SEED);
    const run: HeroInput = { ...idle, forward: 1, sprint: false, camYaw: Math.PI / 2 };
    let off = false;
    for (let i = 0; i < 60 * 200 && !off; i++) {
      updateHero(s, run, 1 / 60, SEED);
      if (islandRho(isl, s.x, s.z) > 1.05 && !islandAt(s.x, s.z, SEED) && Number.isNaN(skyGround(s.x, s.y, s.z, SEED))) off = true;
    }
    expect(off).toBe(true);
    for (let i = 0; i < 30; i++) updateHero(s, run, 1 / 60, SEED);
    expect(s.move).toBe('air');
    expect(s.y - s.ground).toBeGreaterThan(300);
  });
});
