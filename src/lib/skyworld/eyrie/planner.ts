/**
 * The planner the Eyries' grammars build with (settle.ts, crown.ts,
 * grammars.ts): per island, what is taken (footprints, the streets), its
 * paving and footprints for the district map, and the helpers every grammar
 * shares — a building fits only well inside the rim, dry, on ground level
 * enough for its plinth, clear of everything placed before.
 */

import type { Archipelago } from '../archipelago';
import { islandRho, nearWater, outline, topAt, type SkyIsland } from '../island';
import type { Structure } from '../structures';
import { groundRange, hallHouse, roost, type RoostOpts, type Site } from './buildings';
import type { DShape } from './district';

/** A disc taken on an island: a footprint, or part of a street (which roosts face). */
type Disc = { x: number; z: number; r: number; street: boolean };

/** Where a bridge lands on an island: its footing, and the bearing out along the bridge. */
export type Footing = { x: number; z: number; out: number; w: number };

/** The wind the Eyries build against (the grass's and banners' wind, x and z). */
export const WIND = Math.atan2(0.49, 0.87);

/** Cells of the planner's grid of taken discs (m). */
const GRID = 16;
const gkey = (i: number, j: number) => (i + 0x8000) * 0x10000 + (j + 0x8000);

export class Planner {
  readonly taken: Disc[] = [];
  /** The taken discs by the 16 m cells they touch (fits() asks only the cells round it). */
  private grid = new Map<number, Disc[]>();
  /** What it paved and built on, for its district map (district.ts). */
  readonly paving: DShape[] = [];
  readonly built: DShape[] = [];
  readonly site: Site;
  constructor(readonly isl: SkyIsland, readonly rnd: () => number, palette: number) {
    this.site = { ground: (x, z) => topAt(isl, x, z), rnd, palette };
  }
  /** Is a footprint of radius r at (x, z) well inside the rim, dry, with at most maxRelief of ground under it, clear of what is taken? */
  fits(x: number, z: number, r: number, maxRelief: number, rim = 0.86): boolean {
    const isl = this.isl;
    if (islandRho(isl, x, z) > rim - r / (isl.R * 1.1)) return false;
    // (the cheap tests first: what is taken round it, then the water, then the ground under it)
    const i0 = Math.floor((x - r) / GRID), i1 = Math.floor((x + r) / GRID), j0 = Math.floor((z - r) / GRID), j1 = Math.floor((z + r) / GRID);
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const l = this.grid.get(gkey(i, j));
        if (l) for (const t of l) if ((t.x - x) ** 2 + (t.z - z) ** 2 < (t.r + r) ** 2) return false;
      }
    }
    if (nearWater(isl, x, z, r + 5)) return false;
    const [lo, hi] = groundRange(this.site, x, z, r);
    return hi - lo <= maxRelief;
  }
  take(x: number, z: number, r: number, street = false): void {
    const d = { x, z, r, street };
    this.taken.push(d);
    for (let j = Math.floor((z - r) / GRID); j <= Math.floor((z + r) / GRID); j++) {
      for (let i = Math.floor((x - r) / GRID); i <= Math.floor((x + r) / GRID); i++) {
        const k = gkey(i, j);
        const l = this.grid.get(k);
        if (l) l.push(d); else this.grid.set(k, [d]);
      }
    }
  }
  /** Take a disc for a building and mark its footprint (a little inside the disc taken). */
  place(x: number, z: number, r: number, foot = r - 2): void { this.take(x, z, r); this.foot(x, z, foot); }
  /** A footprint for the district map (a disc of radius r, or the box hx × hz turned by yaw). */
  foot(x: number, z: number, r: number, box?: { hx: number; hz: number; yaw: number }): void {
    this.built.push(box ? { k: 'box', x, z, ...box } : { k: 'disc', x, z, r });
  }
  /** Paving: a disc (a plaza, a bridgehead). */
  paveDisc(x: number, z: number, r: number): void { this.paving.push({ k: 'disc', x, z, r }); }
  /** A street from a to b (discs every 5 m, and its paving), its half width w. */
  street(ax: number, az: number, bx: number, bz: number, w: number): void {
    const len = Math.hypot(bx - ax, bz - az), n = Math.max(1, Math.ceil(len / 5));
    for (let k = 0; k <= n; k++) this.take(ax + ((bx - ax) * k) / n, az + ((bz - az) * k) / n, w, true);
    this.paving.push({ k: 'seg', ax, az, bx, bz, w: w - 0.5 });
  }
  /** The nearest street point to (x, z) within 160 m, or null (the grid's cells, ring by ring outward). */
  nearestStreet(x: number, z: number): Disc | null {
    let best: Disc | null = null, bd = Infinity;
    const ci = Math.floor(x / GRID), cj = Math.floor(z / GRID);
    for (let ring = 0; ring <= 10; ring++) {
      for (let j = cj - ring; j <= cj + ring; j++) {
        for (let i = ci - ring; i <= ci + ring; i++) {
          if (Math.max(Math.abs(i - ci), Math.abs(j - cj)) !== ring) continue;
          const l = this.grid.get(gkey(i, j));
          if (!l) continue;
          for (const t of l) {
            if (!t.street) continue;
            const d = (t.x - x) ** 2 + (t.z - z) ** 2;
            if (d < bd) { bd = d; best = t; }
          }
        }
      }
      // (anything in a farther ring is at least ring · GRID away)
      if (best && Math.sqrt(bd) < ring * GRID) break;
    }
    return best;
  }
  /** A point on the rim at bearing th, pulled in by `inset` m. */
  rim(th: number, inset: number): [number, number] {
    const r = this.isl.R * outline(this.isl, th) - inset;
    return [this.isl.cx + Math.cos(th) * r, this.isl.cz + Math.sin(th) * r];
  }
}

/** A roost of the class's mix, facing yaw `face`, its size by how near the heart of the isle it stands (0 … 1). */
function roostOpts(rnd: () => number, face: number, heart: number, r: number): RoostOpts {
  const k = rnd();
  return {
    r, face, floors: 4 + Math.floor(rnd() * 3 + heart * 4), square: k < 0.2,
    roof: k < 0.2 ? 'cone' : k < 0.55 ? 'cone' : k < 0.72 ? 'battlement' : k < 0.86 ? 'spire' : 'dome',
    pennant: rnd() < 0.35, turrets: rnd() < 0.55 ? 3 + Math.floor(rnd() * 2) : 0, tier: rnd() < 0.65,
  };
}

/** Roosts along a street from a to b, both sides, every `step` m, set back from its half width w. */
export function* lineStreet(pl: Planner, out: Structure[], ax: number, az: number, bx: number, bz: number, w: number, step: number, heart: (x: number, z: number) => number): Generator<void, void, void> {
  const len = Math.hypot(bx - ax, bz - az);
  const dx = (bx - ax) / len, dz = (bz - az) / len;
  for (let d = step * 0.8; d < len - 8; d += step * (0.85 + 0.3 * pl.rnd())) {
    for (const side of [-1, 1]) {
      if (pl.rnd() < 0.18) continue;
      const face = Math.atan2(dz * side, -dx * side); // toward the street
      if (pl.rnd() < 0.55) {
        // a hall-house, its ridge along the street
        const hl = 12 + 7 * pl.rnd(), hw = 8 + 3 * pl.rnd();
        const off = w + 2 + hw / 2;
        const x = ax + dx * d - dz * side * off, z = az + dz * d + dx * side * off;
        const rr = Math.hypot(hl, hw) / 2;
        if (!pl.fits(x, z, rr + 1, 7)) continue;
        out.push(hallHouse(pl.site, x, z, { face, len: hl, wid: hw, floors: 2 + Math.floor(pl.rnd() * 2 + 1.5 * heart(x, z)), tower: pl.rnd() < 0.7 ? (pl.rnd() < 0.5 ? -1 : 1) : 0, square: pl.rnd() < 0.3 }));
        pl.take(x, z, rr + 1.5);
        pl.foot(x, z, rr, { hx: hl / 2 + 2.5, hz: hw / 2 + 2.5, yaw: face });
      } else {
        const r = 4.5 + 3 * pl.rnd();
        const off = w + 2.2 + r;
        const x = ax + dx * d - dz * side * off, z = az + dz * d + dx * side * off;
        if (!pl.fits(x, z, r + 1.5, 8)) continue;
        out.push(roost(pl.site, x, z, roostOpts(pl.rnd, face, heart(x, z), r)));
        pl.take(x, z, r + 2.5);
        pl.foot(x, z, r + 1.2);
      }
    }
    yield;
  }
}

/** Fill the rest of an isle with roosts (jittered grid, every `step` m), each facing its nearest street. */
export function* fillRoosts(pl: Planner, out: Structure[], step: number, keep: number, heart: (x: number, z: number) => number): Generator<void, void, void> {
  const isl = pl.isl;
  const n = Math.ceil((isl.R * 2) / step);
  let tries = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (++tries % 6 === 0) yield;
      if (pl.rnd() > keep) continue;
      const x = isl.cx - isl.R + (i + 0.2 + 0.6 * pl.rnd()) * step, z = isl.cz - isl.R + (j + 0.2 + 0.6 * pl.rnd()) * step;
      const st = pl.nearestStreet(x, z);
      const face = st ? Math.atan2(st.x - x, st.z - z) : pl.rnd() * 6.28;
      if (pl.rnd() < 0.6) {
        const hl = 11 + 7 * pl.rnd(), hw = 8 + 3 * pl.rnd();
        const rr = Math.hypot(hl, hw) / 2;
        if (!pl.fits(x, z, rr + 2, 6)) continue;
        out.push(hallHouse(pl.site, x, z, { face, len: hl, wid: hw, floors: 2 + Math.floor(pl.rnd() * 1.4 + heart(x, z)), tower: pl.rnd() < 0.55 ? (pl.rnd() < 0.5 ? -1 : 1) : 0, square: pl.rnd() < 0.3 }));
        pl.take(x, z, rr + 3);
        pl.foot(x, z, rr, { hx: hl / 2 + 2.5, hz: hw / 2 + 2.5, yaw: face });
      } else {
        const r = 4.2 + 3 * pl.rnd();
        if (!pl.fits(x, z, r + 2, 7)) continue;
        out.push(roost(pl.site, x, z, roostOpts(pl.rnd, face, heart(x, z), r)));
        pl.take(x, z, r + 3);
        pl.foot(x, z, r + 1.2);
      }
      yield;
    }
  }
}

/** The highest level spot within `within` of the centre for a footprint of radius r (60 tries), or null. */
export function summit(pl: Planner, r: number, within: number, maxRelief: number): { x: number; z: number } | null {
  const isl = pl.isl;
  let best: { x: number; z: number; score: number } | null = null;
  for (let k = 0; k < 60; k++) {
    const a = pl.rnd() * Math.PI * 2, d = Math.sqrt(pl.rnd()) * within;
    const x = isl.cx + Math.cos(a) * d, z = isl.cz + Math.sin(a) * d;
    if (!pl.fits(x, z, r, maxRelief)) continue;
    const [lo, hi] = groundRange(pl.site, x, z, r * 0.8);
    const score = topAt(isl, x, z) - 1.5 * (hi - lo) - 0.004 * d;
    if (!best || score > best.score) best = { x, z, score };
  }
  return best;
}

/** Bridgeheads: a small paved landing with lanterns where each bridge lands (the bridge's ends carry its Wardens). */
export function bridgeheads(pl: Planner, feet: Footing[]): void {
  for (const f of feet) {
    pl.take(f.x, f.z, f.w / 2 + 6, true);
    pl.paveDisc(f.x, f.z, f.w / 2 + 4);
  }
}


/** A grammar builds into `out`, yielding now and then so a frame can stop and resume it (eyrie.ts eyrieStep). */
export type Grammar = (a: Archipelago, i: number, pl: Planner, feet: Footing[], out: Structure[]) => Generator<void, void, void>;

