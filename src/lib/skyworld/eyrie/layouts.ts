/**
 * How each Eyrie class arranges its islands (SKY_CIVILIZATION.md §7, §11 rule 2).
 * ---------------------------------------------------------------------------
 * Every class has its own arrangement, so no two classes share a silhouette:
 *   crown    a broad terraced Crown isle, 4–7 ward isles round it a little
 *            lower, each joined to it by a vow-bridge; perch isles beyond;
 *   mirror   a bowl isle holding a lagoon, well isles floating higher round it;
 *   halo     the Halo's isle and a helix of pilgrim isles climbing round it;
 *   beacon   a mountain isle (the Vaelspire's), observatory isles on its
 *            shoulders, small crystal isles floating above;
 *   petal    a chain of terraced isles stepping down a curve (the Stair of Waters);
 *   bough    the Elder Bough's isle and the isles its roots bind (natural arches);
 *   ember    jagged basalt isles packed close;
 *   rime     high ice isles round a citadel;
 *   fallen   a broken Crown: tilted isles drifted apart, debris, no bridges;
 *   harbour  isles in a ring round an open anchorage, bridged round the ring;
 *   wells    low isles hanging in the cloud deck, joined by natural arches.
 * The class also picks each island's top (terraces, a bowl, a summit), its
 * underside (spikes, crystal veins, roots, basalt, ice), its biome and its water.
 */

import { CLASS_INFO, type EyrieClass } from './classes';
import type { SkyBiome } from '../biomes';
import { outlineMax, type IslandKind, type SkyIsland, type Underside } from '../island';
import type { Arch } from '../arches';
import { SKY_REALM as S } from '../../constants';
import { ISLAND_LOWER } from '../heights';

/** What an island is for in its Eyrie's plan (the grammars, eyrie.ts, build by it). */
export type IslandRole =
  | 'crown' | 'ward' | 'perch' | 'bowl' | 'well' | 'halo' | 'pilgrim' | 'peak' | 'observatory' | 'crystal'
  | 'terrace' | 'root' | 'grove' | 'forge' | 'yard' | 'citadel' | 'watch' | 'ruin' | 'quay' | 'comb' | 'debris';

/** An island's shape as its class wants it (archipelago.ts makeIsland). */
export type Shape = Partial<{
  terrace: number; bowl: number; tiltX: number; tiltZ: number; under: Underside; tint: number;
  /** × the island's hill height, crags, summit (else its biome's); × its underside's depth. */
  hillK: number; crag: number; peak: number; depthK: number;
  lobes: number; biome: SkyBiome;
  /** Its water: as its biome would have it, a lagoon filling its middle, a pool it always keeps, none. */
  lake: 'auto' | 'lagoon' | 'pool' | 'none';
}>;

export type LayoutCtx = {
  cls: EyrieClass;
  cx: number; cz: number;
  rnd: () => number;
  islands: SkyIsland[];
  roles: IslandRole[];
  shapes: Shape[];
  arches: Arch[];
  island(kind: IslandKind, x: number, z: number, R: number, y: number, shape?: Shape): SkyIsland;
  /** Add an island if it lies within reach and `gap` m clear of the others; its index, or −1. */
  add(isl: SkyIsland, role: IslandRole, shape: Shape, gap?: number): number;
  /** Join two islands (a built bridge or a natural arch, by the class), unless too steep or crossing another island. */
  join(a: number, b: number, maxLen?: number): Arch | null;
  /** Floating rocks round the islands. */
  debris(n: number, r: readonly [number, number]): void;
};

const TAU = Math.PI * 2;
const lerp = (r: readonly [number, number], f: number) => r[0] + (r[1] - r[0]) * f;
// (pulled down by the heights in use, heights.ts: makeIsland lifts back any underside its floor would not clear)
const classTop = (c: LayoutCtx) => lerp(CLASS_INFO[c.cls].top, c.rnd()) - ISLAND_LOWER;

/**
 * A point `gap` m of open sky beyond island isl toward bearing phi, for an island of radius r (both
 * by their largest radii, as the crowding test measures them: an outline reaches 1.17 radii at most).
 */
function beyond(isl: SkyIsland, phi: number, gap: number, r: number): [number, number] {
  const D = isl.R * outlineMax(isl) + gap + r * 1.18;
  return [isl.cx + Math.cos(phi) * D, isl.cz + Math.sin(phi) * D];
}

/** Ward and satellite isles round island 0 at even bearings: their indices. */
function ring(c: LayoutCtx, n: number, r: readonly [number, number], gap: readonly [number, number], dy: readonly [number, number],
  role: IslandRole, shape: () => Shape, kind: IslandKind = 'satellite'): number[] {
  const hub = c.islands[0];
  const out: number[] = [];
  const phi0 = c.rnd() * TAU;
  for (let j = 0; j < n; j++) {
    const phi = phi0 + (j / n) * TAU + (c.rnd() - 0.5) * (TAU / n) * 0.35;
    const R = lerp(r, c.rnd());
    const [x, z] = beyond(hub, phi, lerp(gap, c.rnd()), R);
    const sh = shape();
    const k = c.add(c.island(kind, x, z, R, hub.y + lerp(dy, c.rnd()), sh), role, sh, 50);
    if (k >= 0) out.push(k);
  }
  return out;
}

/** Small isles scattered between `inner` m from the centre and the Eyrie's reach. */
function scatter(c: LayoutCtx, n: number, r: readonly [number, number], inner: number, dy: readonly [number, number], role: IslandRole, shape: () => Shape): void {
  const hub = c.islands[0];
  for (let j = 0; j < n; j++) {
    const phi = c.rnd() * TAU;
    const R = lerp(r, c.rnd());
    const D = inner + c.rnd() * Math.max(0, S.archReach - inner - R * 1.2);
    const sh = shape();
    c.add(c.island('isle', c.cx + Math.cos(phi) * D, c.cz + Math.sin(phi) * D, R, hub.y + lerp(dy, c.rnd()), sh), role, sh, 35);
  }
}

export const LAYOUTS: Record<EyrieClass, (c: LayoutCtx) => void> = {
  crown(c) {
    const R = lerp([340, 500], c.rnd());
    const sh: Shape = { terrace: lerp([6, 9], c.rnd()), hillK: 0.6, crag: 0, peak: 0.45, biome: 'garden', lobes: 3 + Math.floor(c.rnd() * 2), lake: 'auto', tint: 1 };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'crown', sh, 0);
    const wards = ring(c, 4 + Math.floor(c.rnd() * 4), [110, 200], [60, 130], [-45, -5], 'ward',
      () => ({ terrace: lerp([4, 6], c.rnd()), hillK: 0.5, crag: 0, peak: 0, biome: 'garden', lobes: 2 + Math.floor(c.rnd() * 2), tint: 1 }));
    for (const k of wards) c.join(0, k);
    for (let i = 0; i < wards.length && wards.length > 2; i++) if (c.rnd() < 0.45) c.join(wards[i], wards[(i + 1) % wards.length], 520);
    scatter(c, 3 + Math.floor(c.rnd() * 4), [28, 70], R * 1.3 + 260, [-60, 80], 'perch', () => ({ biome: 'garden', lobes: 1 + Math.floor(c.rnd() * 2), lake: 'none' }));
    c.debris(10 + Math.floor(c.rnd() * 8), [5, 20]);
  },
  mirror(c) {
    const R = lerp([360, 520], c.rnd());
    const sh: Shape = { bowl: 0.9, hillK: 0.55, crag: 0, peak: 0, terrace: 4, biome: 'garden', lobes: 3, lake: 'lagoon', tint: 1 };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'bowl', sh, 0);
    const wells = ring(c, 5 + Math.floor(c.rnd() * 4), [60, 120], [45, 110], [20, 70], 'well',
      () => ({ hillK: 0.8, biome: c.rnd() < 0.5 ? 'garden' : 'blossom', lobes: 2, lake: 'pool', tint: 1 }));
    for (const k of wells) c.join(0, k);
    c.debris(8 + Math.floor(c.rnd() * 6), [4, 16]);
  },
  halo(c) {
    const R = lerp([220, 330], c.rnd());
    const y0 = classTop(c);
    const sh: Shape = { hillK: 0.5, crag: 0.2, peak: 0, under: 'crystal', lobes: 3, lake: 'none' };
    c.add(c.island('hub', c.cx, c.cz, R, y0, sh), 'halo', sh, 0);
    // The pilgrims' helix: each isle a turn further round and higher, joined to the one before.
    const n = 6 + Math.floor(c.rnd() * 4);
    let phi = c.rnd() * TAU, D = R * 1.15 + 150;
    let prev = 0;
    for (let j = 0; j < n; j++) {
      const r = lerp([45, 90], c.rnd());
      const y = y0 - 70 + j * lerp([16, 24], c.rnd());
      const s2: Shape = { hillK: 0.6, under: 'crystal', lobes: 1 + Math.floor(c.rnd() * 2), lake: 'none' };
      const k = c.add(c.island('isle', c.cx + Math.cos(phi) * D, c.cz + Math.sin(phi) * D, r, y, s2), 'pilgrim', s2, 30);
      if (k >= 0) {
        c.join(prev, k, 360);
        prev = k;
      }
      phi += lerp([0.7, 0.95], c.rnd());
      D += lerp([25, 50], c.rnd());
    }
    if (prev !== 0) c.join(prev, 0, 420);
    c.debris(14 + Math.floor(c.rnd() * 8), [3, 9]); // the runestones (and plain rocks)
  },
  beacon(c) {
    const R = lerp([320, 470], c.rnd());
    const sh: Shape = { hillK: 1.25, crag: 0.45, peak: lerp([1.6, 2.2], c.rnd()), under: 'crystal', lobes: 3, lake: 'none' };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'peak', sh, 0);
    const obs = ring(c, 2 + Math.floor(c.rnd() * 2), [90, 150], [80, 160], [-30, 20], 'observatory', () => ({ hillK: 0.5, lobes: 2, under: 'crystal', lake: 'none' }));
    for (const k of obs) c.join(0, k);
    scatter(c, 4 + Math.floor(c.rnd() * 4), [22, 55], R * 1.2 + 120, [40, 220], 'crystal', () => ({ under: 'crystal', lobes: 1, hillK: 1.2, crag: 0.8, lake: 'none' }));
    c.debris(6, [4, 12]);
  },
  petal(c) {
    // The Stair of Waters: a curving chain, each isle lower than the last.
    const n = 5 + Math.floor(c.rnd() * 5);
    let heading = c.rnd() * TAU;
    const turn = (c.rnd() - 0.5) * 0.7;
    let R = lerp([150, 230], c.rnd());
    let x = c.cx - Math.cos(heading) * 0.42 * S.archReach, z = c.cz - Math.sin(heading) * 0.42 * S.archReach;
    let y = classTop(c) + 40;
    let prev = -1;
    for (let j = 0; j < n; j++) {
      const s2: Shape = { terrace: lerp([3, 5], c.rnd()), hillK: 0.7, crag: 0, peak: 0, biome: 'blossom', lobes: 2 + Math.floor(c.rnd() * 2), lake: 'pool', tint: 1 };
      const k = c.add(c.island(j === 0 ? 'hub' : 'satellite', x, z, R, y, s2), 'terrace', s2, 30);
      if (k >= 0) {
        if (prev >= 0) c.join(prev, k, 420);
        prev = k;
      }
      const R2 = lerp([90, 200], c.rnd());
      const gap = lerp([45, 110], c.rnd());
      heading += turn + (c.rnd() - 0.5) * 0.4;
      x += Math.cos(heading) * (R * 1.18 + gap + R2 * 1.18);
      z += Math.sin(heading) * (R * 1.18 + gap + R2 * 1.18);
      y -= lerp([12, 36], c.rnd());
      R = R2;
    }
    c.debris(8, [4, 14]);
  },
  bough(c) {
    const R = lerp([260, 370], c.rnd());
    const sh: Shape = { hillK: 0.7, crag: 0.1, under: 'roots', lobes: 4, biome: 'mistwood', lake: 'auto' };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'root', sh, 0);
    const groves = ring(c, 3 + Math.floor(c.rnd() * 3), [100, 200], [60, 140], [-35, 25], 'grove', () => ({ under: 'roots', lobes: 2, biome: 'mistwood' }));
    for (const k of groves) c.join(0, k);
    c.debris(6, [4, 12]);
  },
  ember(c) {
    const R = lerp([180, 260], c.rnd());
    const sh: Shape = { hillK: 1.2, crag: 1, peak: 0.5, under: 'basalt', lobes: 4, biome: 'ember', lake: 'none', depthK: 1.2 };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'forge', sh, 0);
    const yards = ring(c, 4 + Math.floor(c.rnd() * 4), [80, 200], [35, 90], [-40, 30], 'yard', () => ({ hillK: 1.1, crag: 0.9, under: 'basalt', lobes: 3, biome: 'ember', lake: 'none', depthK: 1.2 }));
    for (const k of yards) c.join(0, k);
    for (let i = 0; i + 1 < yards.length; i++) if (c.rnd() < 0.5) c.join(yards[i], yards[i + 1], 300);
    c.debris(14, [4, 16]);
  },
  rime(c) {
    const R = lerp([260, 400], c.rnd());
    const sh: Shape = { hillK: 0.9, crag: 0.5, peak: 0.6, under: 'ice', lobes: 3, biome: 'frost', lake: 'auto' };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'citadel', sh, 0);
    const watches = ring(c, 2 + Math.floor(c.rnd() * 3), [110, 210], [60, 140], [-30, 40], 'watch', () => ({ under: 'ice', lobes: 2, biome: 'frost' }));
    for (const k of watches) c.join(0, k);
    c.debris(8, [4, 14]);
  },
  fallen(c) {
    // A broken Crown: tilted isles drifted apart, no bridges left, much debris.
    const tilt = () => ({ tiltX: (c.rnd() - 0.5) * 0.16, tiltZ: (c.rnd() - 0.5) * 0.16 });
    const R = lerp([380, 600], c.rnd());
    const sh: Shape = { terrace: lerp([6, 9], c.rnd()), hillK: 0.6, peak: 0.4, biome: 'blossom', lobes: 4, ...tilt() };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'ruin', sh, 0);
    ring(c, 4 + Math.floor(c.rnd() * 4), [110, 220], [140, 320], [-110, 70], 'ruin', () => ({ terrace: 5, hillK: 0.5, biome: 'blossom', lobes: 3, ...tilt() }));
    scatter(c, 4, [25, 60], R * 1.3 + 300, [-120, 100], 'ruin', () => ({ biome: 'blossom', lobes: 1, ...tilt() }));
    c.debris(22 + Math.floor(c.rnd() * 10), [5, 28]);
  },
  harbour(c) {
    // Isles in a ring round an open anchorage, bridged round the ring (one gap for the ships' way in).
    const n = 5 + Math.floor(c.rnd() * 3);
    const A = lerp([300, 420], c.rnd());
    const y0 = classTop(c);
    const phi0 = c.rnd() * TAU;
    const idx: number[] = [];
    for (let j = 0; j < n; j++) {
      const phi = phi0 + (j / n) * TAU + (c.rnd() - 0.5) * 0.25;
      const R = lerp([110, 210], c.rnd());
      const D = A + R * 1.2;
      const s2: Shape = { terrace: 4, hillK: 0.5, crag: 0, biome: 'garden', lobes: 2 + Math.floor(c.rnd() * 2), tint: 1 };
      const k = c.add(c.island(j === 0 ? 'hub' : 'satellite', c.cx + Math.cos(phi) * D, c.cz + Math.sin(phi) * D, R, y0 + (c.rnd() - 0.5) * 50, s2), 'quay', s2, 40);
      if (k >= 0) idx.push(k);
    }
    for (let i = 0; i + 1 < idx.length; i++) c.join(idx[i], idx[i + 1], 520);
    c.debris(8, [4, 14]);
  },
  wells(c) {
    const R = lerp([160, 240], c.rnd());
    const sh: Shape = { hillK: 0.8, biome: 'mistwood', lobes: 3, lake: 'auto' };
    c.add(c.island('hub', c.cx, c.cz, R, classTop(c), sh), 'comb', sh, 0);
    const combs = ring(c, 3 + Math.floor(c.rnd() * 4), [80, 180], [50, 130], [-40, 40], 'comb', () => ({ biome: 'mistwood', lobes: 2 }));
    for (const k of combs) c.join(0, k);
    c.debris(8, [4, 14]);
  },
};

/** Do two islands' outlines come within `gap` m of each other (by their largest radii)? */
export const crowds = (a: SkyIsland, b: SkyIsland, gap: number) =>
  Math.hypot(a.cx - b.cx, a.cz - b.cz) < a.R * outlineMax(a) + b.R * outlineMax(b) + gap;
