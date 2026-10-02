import { describe, expect, test } from 'bun:test';
import { SKY_REALM, SPAWN } from '../src/lib/constants';
import { SKY_BIOMES } from '../src/lib/skyworld/biomes';
import { insideBy, outline, topAt } from '../src/lib/skyworld/island';
import { packRealm } from '../src/lib/gpu/skyRealm';
import {
  ARCH_SEGS, ARCH_SIDES, archIndices, POLAR_LODS, polarIndices, REALM_LAYOUT, REALM_MAX, RL_FLAGS, skyRealmWGSL,
} from '../src/lib/gpu/skyRealm.wgsl';
import { floatingOrigin } from '../src/lib/gpu/frame';

const SEED = 1337;

describe('the sky realm on the GPU: its buffer', () => {
  test('the packed layout matches the WGSL structs', () => {
    const w = skyRealmWGSL(true);
    // Records: SkyIsl 9 vec4, SkyArch 5, SkyFall 3, SkyCluster 2.
    const fields = (name: string) => {
      const body = w.slice(w.indexOf(`struct ${name} {`), w.indexOf('};', w.indexOf(`struct ${name} {`)));
      return (body.match(/: vec4[fu]/g) ?? []).length;
    };
    expect(fields('SkyIsl') * 16).toBe(REALM_LAYOUT.islStride);
    expect(fields('SkyArch') * 16).toBe(REALM_LAYOUT.archStride);
    expect(fields('SkyFall') * 16).toBe(REALM_LAYOUT.fallStride);
    expect(fields('SkyCluster') * 16).toBe(REALM_LAYOUT.clStride);
    expect(w).toContain(`array<SkyCluster, ${REALM_MAX.clusters}>`);
    expect(w).toContain(`array<SkyIsl, ${REALM_MAX.islands}>`);
    expect(w).toContain(`array<vec4f, ${REALM_MAX.points}>`);
    expect(REALM_LAYOUT.bytes).toBe(16 + REALM_MAX.clusters * 32 + REALM_MAX.islands * 144 + REALM_MAX.points * 16 + REALM_MAX.arches * 80 + REALM_MAX.falls * 48);
  });

  test('clip distances cut the tops when the device has them; discard stands in otherwise', () => {
    expect(skyRealmWGSL(true)).toContain('@builtin(clip_distances)');
    expect(skyRealmWGSL(true)).not.toContain('discard');
    expect(skyRealmWGSL(false)).not.toContain('clip_distances');
    expect(skyRealmWGSL(false)).toContain('if (i.clipD < 0.0) { discard; }');
  });

  test('round the spawn: nearest first, chunk-local, every record where the shader reads it', () => {
    const { ox, oz } = floatingOrigin(SPAWN.x, SPAWN.z);
    const { data, islands, arches, falls } = packRealm(SPAWN.x, SPAWN.z, SEED, ox, oz);
    const f = new Float32Array(data), u = new Uint32Array(data), i32 = new Int32Array(data);
    const L = REALM_LAYOUT;
    const [nc, ni, na, nf] = [u[0], u[1], u[2], u[3]];
    expect(ni).toBe(islands.length);
    expect(na).toBe(arches.length);
    expect(nf).toBe(falls.length);
    expect(nc).toBeGreaterThan(5);
    expect(ni).toBeLessThanOrEqual(REALM_MAX.islands);
    // Clusters: contiguous runs of islands and arches, each island inside its cluster's radius, nearest first.
    let nextIsl = 0, nextArch = 0, lastD = -Infinity;
    for (let c = 0; c < nc; c++) {
      const o = (L.cl + c * L.clStride) / 4;
      const [cx, cz, rad] = [f[o], f[o + 1], f[o + 2]];
      const [fi, n, fa, m] = [u[o + 4], u[o + 5], u[o + 6], u[o + 7]];
      expect(fi).toBe(nextIsl);
      expect(fa).toBe(nextArch);
      nextIsl += n;
      nextArch += m;
      const d = Math.hypot(cx + ox - SPAWN.x, cz + oz - SPAWN.z) - rad;
      expect(d).toBeGreaterThanOrEqual(lastD - 400); // (sorted by distance to their edge, give or take the margin)
      lastD = Math.max(lastD, d);
      for (let k = fi; k < fi + n; k++) {
        const io = (L.isl + k * L.islStride) / 4;
        expect(Math.hypot(f[io] - cx, f[io + 1] - cz) + f[io + 2]).toBeLessThanOrEqual(rad + 1e-3);
      }
    }
    expect(nextIsl).toBe(ni);
    expect(nextArch).toBe(na);
    // Islands: chunk-local centre, shape, seed, biome, flags and the water each flag promises.
    for (let k = 0; k < ni; k++) {
      const isl = islands[k].isl;
      const o = (L.isl + k * L.islStride) / 4;
      expect(f[o]).toBeCloseTo(isl.cx - ox, 2);
      expect(f[o + 1]).toBeCloseTo(isl.cz - oz, 2);
      expect(f[o + 3]).toBeCloseTo(isl.y, 2);
      expect(i32[o + 16]).toBe(isl.seed);
      expect(SKY_BIOMES[u[o + 17]]).toBe(isl.biome);
      const fl = u[o + 18];
      expect(!!(fl & RL_FLAGS.debris)).toBe(isl.kind === 'debris');
      expect(!!(fl & RL_FLAGS.lake)).toBe(!!isl.lake);
      expect(!!(fl & RL_FLAGS.frozen)).toBe(!!isl.lake?.frozen);
      expect(!!(fl & RL_FLAGS.stream)).toBe(islands[k].stream);
      if (islands[k].stream) {
        const first = u[o + 19] & 0xffff, cnt = u[o + 19] >>> 16;
        expect(cnt).toBe(isl.stream!.pts.length);
        for (let p = 0; p < cnt; p++) {
          const po = (L.pts + (first + p) * 16) / 4;
          for (let c = 0; c < 4; c++) expect(f[po + c]).toBeCloseTo(isl.stream!.pts[p][c], 2);
        }
        // Its bounds hold the whole carve.
        const [mx, mz, rad] = [f[o + 32], f[o + 33], f[o + 34]];
        for (const p of isl.stream!.pts) expect(Math.hypot(p[0] - mx, p[1] - mz) + 3 * p[3]).toBeLessThan(rad);
      }
      const fi = f[o + 30];
      if (fi >= 0) {
        const fo = (L.fall + fi * L.fallStride) / 4;
        expect(f[fo + 9]).toBe(k);
        expect(f[fo]).toBeCloseTo(isl.fall!.x - ox, 2);
        expect(f[fo + 7]).toBeCloseTo(isl.fall!.reach, 2);
      } else {
        expect(!!(fl & RL_FLAGS.fall)).toBe(false);
      }
    }
    // Floating rocks only in the archipelagos near the camera (their edge within 6 km).
    for (const e of islands) if (e.isl.kind === 'debris') expect(Math.hypot(e.x - SPAWN.x, e.z - SPAWN.z)).toBeLessThan(6000 + 2 * SKY_REALM.archReach);
    expect(islands.some((e) => e.isl.kind === 'debris')).toBe(true);
  });
});

describe('the sky realm on the GPU: its meshes', () => {
  test('polar meshes: every level closes round the bearings, from the centre to the rim', () => {
    for (const [n, m] of POLAR_LODS) {
      const idx = polarIndices(n, m);
      expect(idx.length).toBe(n * m * 6);
      expect(Math.max(...idx)).toBe((n + 1) * m - 1);
      expect(Math.min(...idx)).toBe(0);
    }
  });

  test('arch meshes: sections along it, closed round each', () => {
    const idx = archIndices();
    expect(idx.length).toBe(ARCH_SEGS * ARCH_SIDES * 6);
    expect(Math.max(...idx)).toBe((ARCH_SEGS + 1) * ARCH_SIDES - 1);
  });

  test("the sky rings' clip distance is the signed distance to the outline", () => {
    const { islands } = packRealm(SPAWN.x, SPAWN.z, SEED, 0, 0);
    const isl = islands.find((e) => e.isl.kind === 'hub')!.isl;
    for (const rho of [0.5, 0.99, 1.01, 1.2]) {
      const th = 2.2;
      const x = isl.cx + Math.cos(th) * rho * isl.R * outline(isl, th), z = isl.cz + Math.sin(th) * rho * isl.R * outline(isl, th);
      expect(Math.sign(insideBy(isl, x, z))).toBe(rho < 1 ? 1 : -1);
      if (rho < 1) expect(topAt(isl, x, z)).toBeGreaterThan(isl.y - 3);
    }
  });
});
