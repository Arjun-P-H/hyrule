import { describe, expect, test } from 'bun:test';
import { elevation } from '../src/lib/terrainMath';
import { ISLAND_LOWER } from '../src/lib/skyworld/heights';
import { SPAWN } from '../src/lib/constants';
import { createHeroState, placeHero, updateHero, type HeroInput } from '../src/lib/hero/locomotion';
import { archipelagoInCell } from '../src/lib/skyworld/archipelago';
import { EYRIE_CLASSES } from '../src/lib/skyworld/eyrie/classes';
import { districtAt, DISTRICT_RANGE } from '../src/lib/skyworld/eyrie/district';
import { eyrieCellsNear, eyrieInCell, eyrieStep } from '../src/lib/skyworld/eyrie/eyrie';
import { skyPlantInCell } from '../src/lib/skyworld/flora';
import { islandRho, lakeDist, outlineMax, topAt } from '../src/lib/skyworld/island';
import { buildKit, KIT_VERTEX_BYTES } from '../src/lib/skyworld/kit/kit';
import { pieceId, PIECES } from '../src/lib/skyworld/kit/pieces';
import { PALETTES } from '../src/lib/skyworld/kit/palettes';
import { islandAt } from '../src/lib/skyworld/realm';
import { INSTANCE_BYTES, pieceBox } from '../src/lib/skyworld/structures';
import { structuresWGSL } from '../src/lib/gpu/structures.wgsl';
import { CELL } from '../src/lib/forest/species';
import { flocksNear, FLOCK_BIRDS, FLOCK_RANGE, MAX_FLOCKS, SPECIES, type Flock } from '../src/lib/skyworld/fauna';
import { routesNear, shipAt, type Route } from '../src/lib/skyworld/ships';
import { RAMP_START, RAMP_SWEEP, STOREY_H } from '../src/lib/skyworld/kit/interiors';
import { collide } from '../src/lib/skyworld/colliders';

const SEED = 1337;
/** The Eyries round the spawn, nearest first. */
const NEAR = eyrieCellsNear(SPAWN.x, SPAWN.z, 26000).map(([ix, iz]) => archipelagoInCell(ix, iz, SEED)).filter((a) => !!a);
const CROWN = archipelagoInCell(0, -1, SEED)!;
const CROWN_E = eyrieInCell(0, -1, SEED)!;

describe('the Eyries: the building kit', () => {
  const kit = buildKit();
  test('every mesh is whole: indices in range, positions finite, normals unit, its levels lighter as they go', () => {
    expect(kit.vertices.byteLength).toBe(kit.vertexCount * KIT_VERTEX_BYTES);
    for (let i = 0; i < kit.indices.length; i++) expect(kit.indices[i]).toBeLessThan(kit.vertexCount);
    const f = new Float32Array(kit.vertices), i8 = new Int8Array(kit.vertices);
    for (let v = 0; v < kit.vertexCount; v++) {
      const o = (v * KIT_VERTEX_BYTES) / 4;
      for (let c = 0; c < 4; c++) expect(Number.isFinite(f[o + c])).toBe(true);
      const n = [0, 1, 2].map((c) => i8[v * KIT_VERTEX_BYTES + 16 + c] / 127);
      expect(Math.abs(Math.hypot(n[0], n[1], n[2]) - 1)).toBeLessThan(0.03);
    }
    PIECES.forEach((p, k) => {
      const t = [0, 1, 2].map((l) => kit.meshes[3 * k + l].indexCount);
      expect(t[0]).toBeGreaterThan(0);
      expect(t[1]).toBeLessThanOrEqual(t[0]);
      expect(t[2]).toBeLessThanOrEqual(t[1]);
    });
  });
  test('the far level of a roost is a few dozen triangles', () => {
    const tris = (name: string) => kit.meshes[3 * pieceId(name) + 2].indexCount / 3;
    expect(tris('body.round') + tris('roof.cone')).toBeLessThan(60);
  });
});

describe('the Eyries: classes and their islands', () => {
  test('every class is found round the spawn, the Crown ahead of it', () => {
    const seen = new Set(NEAR.map((a) => a.cls));
    for (const c of EYRIE_CLASSES) expect(seen.has(c)).toBe(true);
    expect(CROWN.cls).toBe('crown');
    expect(Math.hypot(CROWN.cx - SPAWN.x, CROWN.cz - SPAWN.z)).toBeLessThan(5000);
  });
  test('each has a name and the palette of its class, and its islands keep apart', () => {
    for (const a of NEAR) {
      expect(a.name.length).toBeGreaterThan(8);
      expect(a.palette).toBeGreaterThanOrEqual(0);
      expect(a.palette).toBeLessThan(PALETTES.length);
      for (let i = 0; i < a.islands.length; i++) for (let j = i + 1; j < a.islands.length; j++) {
        const p = a.islands[i], q = a.islands[j];
        expect(Math.hypot(p.cx - q.cx, p.cz - q.cz)).toBeGreaterThan(p.R * outlineMax(p) + q.R * outlineMax(q) + 24);
      }
    }
  });
  test('each class arranges its islands its own way', () => {
    for (const a of NEAR) {
      const main = a.islands[0];
      const others = a.islands.filter((_, i) => a.roles[i] !== 'debris').slice(1);
      if (a.cls === 'crown') {
        expect(a.roles.filter((r) => r === 'ward').length).toBeGreaterThanOrEqual(3);
        expect(a.arches.filter((A) => A.kind === 'bridge' && A.from === 0).length).toBeGreaterThanOrEqual(2);
        expect(main.terrace).toBeGreaterThan(0);
      }
      if (a.cls === 'mirror') {
        // a lagoon filling the bowl, turquoise
        expect(main.lake).not.toBeNull();
        expect(main.lake!.r).toBeGreaterThan(0.3 * main.R);
        expect(main.tint).toBe(1);
        expect(lakeDist(main.lake!, 0, 0)).toBeLessThan(0.2);
      }
      if (a.cls === 'petal') {
        // a chain stepping down
        const chain = a.islands.filter((_, i) => a.roles[i] === 'terrace');
        expect(chain.length).toBeGreaterThanOrEqual(4);
        for (let i = 1; i < chain.length; i++) expect(chain[i].y).toBeLessThan(chain[i - 1].y + 1e-6);
      }
      if (a.cls === 'halo') {
        // the pilgrims' helix rises
        const pil = a.islands.filter((_, i) => a.roles[i] === 'pilgrim');
        expect(pil.length).toBeGreaterThanOrEqual(4);
        expect(pil[pil.length - 1].y).toBeGreaterThan(pil[0].y);
      }
      if (a.cls === 'harbour') {
        // a ring round an open anchorage: nothing in its middle
        for (const isl of a.islands) expect(Math.hypot(isl.cx - a.cx, isl.cz - a.cz)).toBeGreaterThan(200);
      }
      if (a.cls === 'fallen') {
        expect(a.arches.length).toBe(0);
        expect(others.some((isl) => Math.abs(isl.tiltX) + Math.abs(isl.tiltZ) > 0.01)).toBe(true);
      }
      // (the classes' heights, all pulled down by the heights in use: the Ice Watch still the highest, the Wells the lowest)
      if (a.cls === 'rime') expect(main.y).toBeGreaterThan(2550 - ISLAND_LOWER);
      if (a.cls === 'wells') expect(main.y).toBeLessThan(1900);
      if (a.cls === 'ember') expect(main.under).toBe('basalt');
    }
  });
});

describe('the Eyries: what is built', () => {
  test('built once per cell and seed, the same every time; a slice at a time takes a few ms', () => {
    expect(eyrieInCell(0, -1, SEED)).toBe(CROWN_E);
    // another Eyrie, sliced: no slice much over its budget
    const far = eyrieCellsNear(SPAWN.x, SPAWN.z, 26000).find(([ix, iz]) => archipelagoInCell(ix, iz, SEED)?.cls === 'crown' && !(ix === 0 && iz === -1));
    if (far) {
      let worst = 0;
      for (;;) {
        const t = performance.now();
        const done = eyrieStep(far[0], far[1], SEED, 3);
        worst = Math.max(worst, performance.now() - t);
        if (done) break;
      }
      expect(worst).toBeLessThan(25);
    }
  });
  test('the Crown is a city: a cathedral, plazas, hundreds of homes, its bridges raised', () => {
    const has = (name: string) => CROWN_E.structures.some((s) => s.pieces.some((p) => p.piece === pieceId(name)));
    for (const name of ['body.front', 'fountain', 'warden', 'body.round', 'roof.gable', 'bridge.pier', 'lantern.post', 'banner']) expect(has(name)).toBe(true);
    expect(has('bridge.span') || has('bridge.spanPointed')).toBe(true);
    expect(CROWN_E.structures.length).toBeGreaterThan(200);
  });
  test('every class builds its own landmarks, whole and within its reach', () => {
    const SIGNS: Record<string, string[]> = {
      crown: ['body.front', 'fountain', 'plaza.round', 'aqueduct', 'aqueduct.fall', 'spout.fall', 'altar', 'tiers'],
      mirror: ['roof.onion', 'ring.entablature', 'bridge.span', 'aqueduct'],
      halo: ['halo.ring', 'runestone', 'runestone.float'],
      beacon: ['crystal.cluster'],
      petal: ['roof.pagoda', 'gate.torii', 'lantern.paper', 'body.lattice'],
      bough: ['bough.tree', 'bough.roots', 'nest.platform'],
      ember: ['chimney', 'brazier'],
      rime: ['crenel', 'body.wall', 'brazier'],
      fallen: ['ruin.top', 'rubble', 'column.fallen'],
      harbour: ['dock', 'crane', 'skyship.moored', 'stall', 'cargo', 'aqueduct'],
      wells: ['cloud.comb', 'wind.sails'],
    };
    const FX: Record<string, string> = { beacon: 'beam', ember: 'smoke', rime: 'vortex' };
    for (const cls of EYRIE_CLASSES) {
      const a = NEAR.find((x) => x.cls === cls)!;
      const e = eyrieInCell(a.ix, a.iz, SEED)!;
      expect(e.structures.length).toBeGreaterThan(40);
      const used = new Set(e.structures.flatMap((st) => st.pieces.map((p) => PIECES[p.piece].name)));
      for (const name of SIGNS[cls]) expect(`${cls}: ${name} ${used.has(name)}`).toBe(`${cls}: ${name} true`);
      if (FX[cls]) expect(e.structures.some((st) => st.fx?.some((f) => f.k === FX[cls]))).toBe(true);
      for (const st of e.structures) {
        expect(Math.hypot(st.x - a.cx, st.z - a.cz) - st.r).toBeLessThan(a.reach);
        for (const p of st.pieces) {
          for (const v of [p.x, p.y, p.z, p.yaw, p.sx, p.sy, p.sz]) expect(Number.isFinite(v)).toBe(true);
          expect(Math.min(p.sx, p.sy, p.sz)).toBeGreaterThan(0);
          expect(p.palette).toBeLessThan(PALETTES.length);
        }
      }
    }
  });
  test('every structure stands on its Eyrie: its pieces above the undersides, its middle over an island or a bridge', () => {
    const box = new Float64Array(6);
    for (const st of CROWN_E.structures) {
      const onIsland = !!islandAt(st.x, st.z, SEED);
      const onBridge = CROWN.arches.some((A) => {
        const px = st.x - A.ax, pz = st.z - A.az, t = (px * A.dirX + pz * A.dirZ) / A.len;
        return t > -0.1 && t < 1.1 && Math.abs(-px * A.dirZ + pz * A.dirX) < A.wMid;
      });
      expect(onIsland || onBridge).toBe(true);
      for (const p of st.pieces) {
        pieceBox(p, box);
        expect(box[4]).toBeGreaterThan(elevation(0.5 * (box[0] + box[3]), 0.5 * (box[2] + box[5]), SEED, 0));
      }
    }
  });
});

describe('the Eyries: district maps', () => {
  const hub = CROWN.islands[0];
  test('the Crown isle has its map: paved along its avenues, built under its homes', () => {
    expect(hub.district).toBeDefined();
    const dd: [number, number] = [0, 0];
    // along the first bridge's avenue (from its footing inward) it is paved
    const A = CROWN.arches.find((x) => x.kind === 'bridge' && x.from === 0)!;
    let paved = 0;
    for (let k = 1; k <= 10; k++) {
      const x = A.ax - A.dirX * 6 * k, z = A.az - A.dirZ * 6 * k;
      districtAt(hub, x, z, dd);
      if (dd[0] < 0) paved++;
    }
    expect(paved).toBeGreaterThanOrEqual(8);
    // far out of town (the rim) nothing is paved or built
    districtAt(hub, hub.cx + hub.R * 2, hub.cz, dd);
    expect(dd[0]).toBe(DISTRICT_RANGE);
  });
  test('no tree grows in a street or against a wall', () => {
    const dd: [number, number] = [0, 0];
    let checked = 0;
    for (let iz = Math.floor((hub.cz - hub.R) / CELL); iz < (hub.cz + hub.R) / CELL; iz += 2) {
      for (let ix = Math.floor((hub.cx - hub.R) / CELL); ix < (hub.cx + hub.R) / CELL; ix += 2) {
        const p = skyPlantInCell(ix, iz, SEED);
        if (!p) continue;
        districtAt(hub, p.x, p.z, dd);
        if (islandRho(hub, p.x, p.z) < 1) {
          expect(dd[0]).toBeGreaterThanOrEqual(2.4);
          expect(dd[1]).toBeGreaterThanOrEqual(3.4);
          checked++;
        }
      }
    }
    expect(checked).toBeGreaterThan(20);
  });
});

describe('the Eyries: the hero among them', () => {
  const idle: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };
  test('crosses every vow-bridge of the Crown on its deck and lands on the far isle', () => {
    for (const A of CROWN.arches.filter((x) => x.kind === 'bridge')) {
      const sx = A.ax - A.dirX * 3, sz = A.az - A.dirZ * 3;
      const s = createHeroState(sx, sz, 0, SEED);
      placeHero(s, sx, topAt(CROWN.islands[A.from], sx, sz) + 2, sz, 0, 0, 0, 0, SEED);
      for (let i = 0; i < 60; i++) updateHero(s, idle, 1 / 60, SEED);
      const run: HeroInput = { ...idle, forward: 1, camYaw: Math.atan2(A.dirX, A.dirZ) };
      let lowest = Infinity;
      for (let i = 0; i < 60 * 240; i++) {
        const px = s.x - A.ax, pz = s.z - A.az, t = (px * A.dirX + pz * A.dirZ) / A.len;
        if (t > 1.05) break;
        run.camYaw = Math.atan2(A.dirX, A.dirZ) + Math.max(-0.4, Math.min(0.4, (-px * A.dirZ + pz * A.dirX) * 0.05));
        updateHero(s, run, 1 / 60, SEED);
        if (t > 0.02 && t < 0.98) lowest = Math.min(lowest, s.y - (A.ay + (A.by - A.ay) * t));
      }
      expect(lowest).toBeGreaterThan(-3);
      expect(islandAt(s.x, s.z, SEED)).toBe(CROWN.islands[A.to]);
      expect(s.move).toBe('ground');
    }
  });
  test('walks into a roost through its door, onto its floor, and is kept inside by its walls; climbs its ramp', () => {
    const SR = pieceId('storey.round');
    const st = CROWN_E.structures.find((x) => x.pieces.some((p) => p.piece === SR && p.sx >= 5.5))!;
    const sp = st.pieces.find((p) => p.piece === SR)!;
    const fx = Math.sin(sp.yaw), fz = Math.cos(sp.yaw);
    const hub = CROWN.islands[0];
    // from 6 m out in front of the door, walking in
    const s = createHeroState(sp.x + fx * (sp.sx + 6), sp.z + fz * (sp.sx + 6), 0, SEED);
    placeHero(s, s.x, topAt(hub, s.x, s.z) + 1, s.z, 0, 0, 0, 0, SEED);
    for (let i = 0; i < 30; i++) updateHero(s, idle, 1 / 60, SEED);
    const inward: HeroInput = { ...idle, forward: 1, camYaw: Math.atan2(-fx, -fz) };
    for (let i = 0; i < 60 * 3 && Math.hypot(s.x - sp.x, s.z - sp.z) > 1.5; i++) updateHero(s, inward, 1 / 60, SEED);
    expect(Math.hypot(s.x - sp.x, s.z - sp.z)).toBeLessThan(2);
    expect(Math.abs(s.y - sp.y)).toBeLessThan(0.3);
    // on toward the back: its wall keeps the hero in
    for (let i = 0; i < 60 * 6; i++) updateHero(s, inward, 1 / 60, SEED);
    expect(Math.hypot(s.x - sp.x, s.z - sp.z)).toBeLessThan(0.9 * sp.sx);
    expect(Math.abs(s.y - sp.y)).toBeLessThan(2.5);
    // the ramp bears the hero up the wall: its walking surface rises from the floor to under the ceiling
    const set = CROWN_E.colliders;
    const body = { x: 0, z: 0, vx: 0, vz: 0 };
    const tops: number[] = [];
    for (const t of [0.1, 0.5, 0.9]) {
      const a = RAMP_START + t * RAMP_SWEEP, R = 0.9 * sp.sx - 0.5;
      const lx = Math.cos(a) * R, lz = Math.sin(a) * R;
      body.x = sp.x + Math.cos(sp.yaw) * lx + Math.sin(sp.yaw) * lz;
      body.z = sp.z - Math.sin(sp.yaw) * lx + Math.cos(sp.yaw) * lz;
      tops.push(collide(set, body, sp.y + 3.3, 0.3, 1.8) - sp.y);
    }
    expect(tops[0]).toBeGreaterThan(0.2);
    expect(tops[1]).toBeGreaterThan(tops[0] + 1);
    expect(tops[2]).toBeGreaterThan(tops[1] + 1);
    expect(tops[2]).toBeLessThan(STOREY_H);
  });
  test('a roost stops the hero at its wall', () => {
    const body = pieceId('body.round');
    const st = CROWN_E.structures.find((x) => x.pieces.some((p) => p.piece === body && p.sx > 4))!;
    const rb = st.pieces.find((p) => p.piece === body)!;
    const hub = CROWN.islands[0];
    const s = createHeroState(rb.x + 22, rb.z, 0, SEED);
    placeHero(s, rb.x + 22, topAt(hub, rb.x + 22, rb.z) + 1, rb.z, 0, 0, 0, 0, SEED);
    for (let i = 0; i < 30; i++) updateHero(s, idle, 1 / 60, SEED);
    for (let i = 0; i < 60 * 8; i++) updateHero(s, { ...idle, forward: 1, camYaw: -Math.PI / 2 }, 1 / 60, SEED);
    expect(Math.hypot(s.x - rb.x, s.z - rb.z)).toBeGreaterThan(rb.sx + 0.3);
  });
});

describe('the Eyries: life in them', () => {
  test('lakes hold pads and lotus on the water, reeds at the margins, weed on the bed, koi under the surface', () => {
    const a = NEAR.find((x) => x.cls === 'mirror')!;
    const e = eyrieInCell(a.ix, a.iz, SEED)!;
    const lagoon = a.islands[0];
    const L = lagoon.lake!;
    const count = new Map<string, number>();
    for (const st of e.structures) {
      for (const p of st.pieces) {
        const name = PIECES[p.piece].name;
        if (!['lily.pad', 'lotus', 'reeds', 'waterweed', 'koi'].includes(name)) continue;
        count.set(name, (count.get(name) ?? 0) + 1);
        const isl = islandAt(p.x, p.z, SEED);
        expect(isl).not.toBeNull();
        const lake = isl!.lake!;
        const bed = topAt(isl!, p.x, p.z);
        if (name === 'lily.pad' || name === 'lotus') {
          expect(Math.abs(p.y - lake.level)).toBeLessThan(0.05);
          expect(lakeDist(lake, p.x - isl!.cx, p.z - isl!.cz)).toBeLessThan(0.97);
        }
        if (name === 'waterweed') expect(p.y + 2.2 * p.sy).toBeLessThan(lake.level);
        if (name === 'koi') { expect(p.y).toBeLessThan(lake.level - 0.3); expect(p.y).toBeGreaterThan(bed); }
        if (name === 'reeds') expect(Math.abs(p.y - (bed - 0.05))).toBeLessThan(0.2);
      }
    }
    for (const name of ['lily.pad', 'lotus', 'reeds', 'waterweed', 'koi']) expect(count.get(name) ?? 0).toBeGreaterThan(name === 'koi' ? 3 : 20);
    expect(L.frozen).toBe(false);
  });
  test('flocks wheel round the islands near the camera: the birds of each class, hawks above, nearest first', () => {
    const out: Flock[] = [];
    const hub = CROWN.islands[0];
    flocksNear(hub.cx, hub.cz, SEED, out);
    expect(out.length).toBeGreaterThan(4);
    expect(out.length).toBeLessThanOrEqual(MAX_FLOCKS);
    const reach = (f: Flock) => Math.max(0, Math.abs(Math.hypot(f.x - hub.cx, f.z - hub.cz) - f.r) - 20);
    for (let k = 0; k < out.length; k++) {
      const f = out[k];
      expect(f.count).toBeGreaterThan(0);
      expect(f.count).toBeLessThanOrEqual(FLOCK_BIRDS);
      expect(f.r).toBeGreaterThan(5);
      expect(Math.abs(f.speed)).toBeGreaterThan(4);
      expect(reach(f)).toBeLessThanOrEqual(FLOCK_RANGE);
      if (k) expect(reach(f)).toBeGreaterThanOrEqual(reach(out[k - 1]));
    }
    const kinds = new Set(out.map((f) => SPECIES[f.species]));
    expect(kinds.has('swift') || kinds.has('dove')).toBe(true);
    expect(kinds.has('hawk')).toBe(true);
    // the same flocks every time
    const again: Flock[] = [];
    flocksNear(hub.cx, hub.cz, SEED, again);
    expect(again).toEqual(out);
  });
  test('skyships fly their routes smoothly, bow first, the lanes clear of every island under them', () => {
    const routes: Route[] = [];
    const hub = CROWN.islands[0];
    routesNear(hub.cx, hub.cz, 16000, SEED, routes);
    expect(routes.some((r) => r.kind === 'lane')).toBe(true);
    expect(routes.some((r) => r.kind === 'loop')).toBe(true);
    const o = new Float64Array(4), q = new Float64Array(4);
    for (const rt of routes) for (let k = 0; k < rt.ships; k++) {
      for (let t = 0; t < 900; t += 7.3) {
        shipAt(rt, k, t, o);
        shipAt(rt, k, t + 0.05, q);
        const step = Math.hypot(q[0] - o[0], q[2] - o[2]);
        expect(Math.abs(step - Math.abs(rt.speed) * 0.05)).toBeLessThan(0.01);
        expect(((q[0] - o[0]) * Math.sin(o[3]) + (q[2] - o[2]) * Math.cos(o[3])) / step).toBeGreaterThan(0.99);
        if (rt.kind === 'lane') {
          const isl = islandAt(o[0], o[2], SEED);
          if (isl) expect(o[1]).toBeGreaterThan(isl.y + 200);
        }
      }
    }
  });
  test('swifts circle each Thread by night', () => {
    const a = NEAR.find((x) => x.cls === 'beacon')!;
    eyrieInCell(a.ix, a.iz, SEED);
    const out: Flock[] = [];
    flocksNear(a.islands[0].cx, a.islands[0].cz, SEED, out);
    expect(out.some((f) => f.night && SPECIES[f.species] === 'swift')).toBe(true);
  });
});

describe('the Eyries on the GPU', () => {
  test('the instance layout matches its WGSL struct; a palette table per palette', () => {
    const w = structuresWGSL(true);
    const body = w.slice(w.indexOf('struct StInst {'), w.indexOf('};', w.indexOf('struct StInst {')));
    expect((body.match(/: vec4f/g) ?? []).length * 16).toBe(INSTANCE_BYTES);
    expect(w).toContain(`const PL_STONE: array<vec3f, ${PALETTES.length}>`);
    expect(w).toContain('fn vsStructureShadow');
    expect(structuresWGSL(false)).not.toContain('fn vsStructureShadow');
  });
});
