/**
 * What of the Eyries the hero bumps into and stands on (SKY_CIVILIZATION.md
 * §11 rule 10): each kit piece's colliders, placed with its instances, kept
 * per Eyrie in a grid.
 * ---------------------------------------------------------------------------
 * A collider is a cylinder, a tube (a ring wall: a parapet, a ledge), an
 * oriented box, a ramp (a box whose top rises along it: stairs), a cone, a
 * dome or a gable (roofs), in its piece's unit space (stretchable pieces) or
 * in metres (fixed pieces; radial rings by their offsets beyond the instance's
 * radius). Placed, it becomes a record in a Float64Array; the grid (16 m
 * cells) lists which records touch each cell, so the hero's questions every
 * tick look at a handful and allocate nothing.
 *
 * The hero (hero/locomotion.ts) asks for the highest top under its feet within
 * a step (a floor) and is pushed out of everything it overlaps between its
 * feet and its head — so roosts, walls and statues stop it, and plazas,
 * bridge decks, stairs, ledges and roofs bear it.
 */

import { PIECES } from './kit/pieces';
import { DOOR_HALF, RAMP_START, RAMP_SWEEP, STOREY_H } from './kit/interiors';
import { PFLAG, type PieceInst, type Structure } from './structures';

const K = { cyl: 0, tube: 1, box: 2, ramp: 3, cone: 4, dome: 5, gable: 6, sag: 7, helix: 8, slab: 9 } as const;

/** A collider in its piece's space: centre (x, z) and y range; a radius (and an inner one), or half extents. */
type Def =
  | { k: 'cyl'; x?: number; z?: number; y0: number; y1: number; r: number }
  /** a ring wall; with `gap`, a doorway that many metres either side of its local +z axis */
  | { k: 'tube'; y0: number; y1: number; r0: number; r1: number; gap?: number }
  /** with `gap`, a doorway that many metres either side of its middle (its x 0), through the whole box */
  | { k: 'box'; x?: number; z?: number; y0: number; y1: number; hx: number; hz: number; gap?: number }
  /** a ramp winding round a ring (radial offsets r0 … r1 beyond the instance's radius) from bearing a0 to a1, rising y0 … y1 */
  | { k: 'helix'; y0: number; y1: number; r0: number; r1: number; a0: number; a1: number }
  /** the same as a slab 0.4 m thick, with room under it (stacked in a shaft) */
  | { k: 'slab'; y0: number; y1: number; r0: number; r1: number; a0: number; a1: number }
  | { k: 'ramp'; y0: number; y1: number; hx: number; z0: number; z1: number }
  | { k: 'cone'; y0: number; y1: number; r: number }
  | { k: 'dome'; y0: number; y1: number; r: number }
  | { k: 'gable'; y0: number; y1: number; hx: number; hz: number }
  /** a rope bridge's deck: y1 at its ends (x ±hx), sagging to y0 in the middle */
  | { k: 'sag'; y0: number; y1: number; hx: number; hz: number };

const bodyBox: Def[] = [{ k: 'box', y0: 0, y1: 1, hx: 1, hz: 1 }];
const bodyCyl: Def[] = [{ k: 'cyl', y0: 0, y1: 1, r: 1 }];

/** Colliders by piece name (pieces left out are walked through: finials, pennants, banners' cloth, doors in solid bodies). */
const DEFS: Record<string, Def[]> = {
  'body.round': bodyCyl, 'body.roundHall': bodyCyl, 'body.roundBlind': bodyCyl, 'body.roundBelfry': bodyCyl,
  'body.square': bodyBox, 'body.hall': bodyBox, 'body.front': bodyBox, 'body.wall': bodyBox, 'body.blind': bodyBox,
  'body.belfry': bodyBox, 'body.arcade': bodyBox,
  'body.octagon': [{ k: 'cyl', y0: 0, y1: 1, r: 0.95 }],
  podium: [{ k: 'box', y0: -1, y1: 0, hx: 1, hz: 1 }],
  'plaza.round': [{ k: 'cyl', y0: -1, y1: 0, r: 1 }],
  'roof.cone': [{ k: 'cone', y0: 0, y1: 1, r: 1.1 }],
  'roof.spire': [{ k: 'cone', y0: 0, y1: 1, r: 1.0 }],
  'roof.pyramid': [{ k: 'cone', y0: 0, y1: 1, r: 1.15 }],
  'roof.gable': [{ k: 'gable', y0: 0, y1: 1, hx: 1, hz: 1.08 }],
  'roof.dome': [{ k: 'dome', y0: 0, y1: 1, r: 1.02 }],
  'roof.onion': [{ k: 'dome', y0: 0, y1: 1, r: 1.1 }],
  'roof.glass': [{ k: 'dome', y0: 0, y1: 1, r: 1.0 }],
  // radial rings: offsets beyond the instance's radius (m)
  'ring.ledge': [{ k: 'tube', y0: -0.36, y1: 0.22, r0: -0.3, r1: 0.95 }],
  'ring.parapet12': [{ k: 'tube', y0: -0.9, y1: 2.15, r0: 0.2, r1: 0.62 }, { k: 'tube', y0: -0.9, y1: 0.25, r0: -60, r1: 0.2 }],
  'ring.parapet20': [{ k: 'tube', y0: -0.9, y1: 2.15, r0: 0.2, r1: 0.62 }, { k: 'tube', y0: -0.9, y1: 0.25, r0: -60, r1: 0.2 }],
  'ring.parapet32': [{ k: 'tube', y0: -0.9, y1: 2.15, r0: 0.2, r1: 0.62 }, { k: 'tube', y0: -0.9, y1: 0.25, r0: -60, r1: 0.2 }],
  turret: [{ k: 'cyl', y0: 1.45, y1: 4.9, r: 1.5 }, { k: 'cone', y0: 4.9, y1: 8.2, r: 1.78 }],
  buttress: [{ k: 'box', z: 1.0, y0: 0, y1: 0.95, hx: 0.6, hz: 1.0 }],
  column: [{ k: 'cyl', y0: 0, y1: 1, r: 1.0 }],
  'lantern.post': [{ k: 'cyl', y0: 0, y1: 3.4, r: 0.2 }],
  warden: [{ k: 'box', y0: 0, y1: 1.66, hx: 1.0, hz: 1.0 }],
  fountain: [{ k: 'cyl', y0: -0.2, y1: 0.75, r: 3.7 }],
  'gate.arch': [
    { k: 'box', x: -2.55, y0: 0, y1: 6.8, hx: 0.65, hz: 0.9 }, { k: 'box', x: 2.55, y0: 0, y1: 6.8, hx: 0.65, hz: 0.9 },
    { k: 'box', y0: 5.4, y1: 7.2, hx: 3.45, hz: 1.1 },
  ],
  'bridge.span': [{ k: 'box', y0: -0.16, y1: 0, hx: 1, hz: 1 }],
  'bridge.spanPointed': [{ k: 'box', y0: -0.1, y1: 0, hx: 1, hz: 1 }],
  'bridge.pier': [{ k: 'box', y0: -0.8, y1: 0, hx: 1, hz: 1 }],
  parapet: [{ k: 'box', y0: 0, y1: 1, hx: 1, hz: 1 }],
  balustrade: [{ k: 'box', y0: 0, y1: 1.05, hx: 1, hz: 0.18 }],
  'stair.6': [{ k: 'ramp', y0: 0, y1: 1, hx: 1, z0: 0, z1: 1 }],
  'stair.12': [{ k: 'ramp', y0: 0, y1: 1, hx: 1, z0: 0, z1: 1 }],
  'stair.24': [{ k: 'ramp', y0: 0, y1: 1, hx: 1, z0: 0, z1: 1 }],
  crenel: [{ k: 'box', y0: 0, y1: 1.1, hx: 2, hz: 0.45 }, { k: 'box', x: -1, y0: 1.1, y1: 2.0, hx: 0.55, hz: 0.45 }, { k: 'box', x: 1, y0: 1.1, y1: 2.0, hx: 0.55, hz: 0.45 }],
  'body.lattice': bodyBox,
  'roof.pagoda': [{ k: 'cone', y0: 0, y1: 1, r: 1.3 }],
  'halo.ring': [{ k: 'box', y0: -10, y1: 8, hx: 38, hz: 5.5 }],
  runestone: [{ k: 'cyl', y0: 0, y1: 6, r: 0.9 }],
  'crystal.cluster': [{ k: 'cyl', y0: -0.1, y1: 0.55, r: 0.12 }],
  'gate.torii': [{ k: 'cyl', x: -3, y0: 0, y1: 7, r: 0.45 }, { k: 'cyl', x: 3, y0: 0, y1: 7, r: 0.45 }],
  'bridge.rope': [{ k: 'sag', y0: -1, y1: 0, hx: 1, hz: 1 }],
  dock: [{ k: 'box', x: 0.5, y0: -0.12, y1: 0, hx: 0.5, hz: 1 }],
  crane: [{ k: 'box', y0: 0, y1: 1.6, hx: 1.2, hz: 1.2 }, { k: 'cyl', y0: 1.6, y1: 13.5, r: 0.4 }],
  stall: [{ k: 'box', z: 0.6, y0: 0, y1: 0.95, hx: 1.8, hz: 0.6 }],
  cargo: [{ k: 'box', y0: 0, y1: 1.2, hx: 1.3, hz: 1.3 }],
  skyship: [{ k: 'box', y0: -3.6, y1: 0.07, hx: 3.4, hz: 14 }, { k: 'box', z: -10, y0: 0, y1: 2.9, hx: 2.6, hz: 3 }],
  'cloud.comb': [{ k: 'cyl', x: -10, y0: 0, y1: 16, r: 0.35 }, { k: 'cyl', x: 10, y0: 0, y1: 16, r: 0.35 }],
  chimney: bodyBox,
  brazier: [{ k: 'cyl', y0: 0, y1: 1.95, r: 0.7 }],
  'bough.tree': [{ k: 'cyl', y0: -8, y1: 100, r: 12 }],
  'nest.platform': [{ k: 'cyl', y0: -0.5, y1: 0, r: 4.2 }],
  'ruin.top': [{ k: 'tube', y0: 0, y1: 2.5, r0: -1.2, r1: 0 }],
  'column.fallen': [{ k: 'box', y0: 0, y1: 0.9, hx: 3.6, hz: 0.5 }],
  rubble: [{ k: 'box', y0: 0, y1: 1.1, hx: 2, hz: 2 }],
  aqueduct: [{ k: 'box', z: -0.7, y0: 0, y1: 0.84, hx: 1, hz: 0.1 }, { k: 'box', z: 0.7, y0: 0, y1: 0.84, hx: 1, hz: 0.1 }, { k: 'box', y0: 0, y1: 0.16, hx: 1, hz: 0.6 }],
  // insides (kit/interiors.ts): walls with their doorways, the furnishings, the ramp
  'storey.round': [{ k: 'tube', y0: 0, y1: STOREY_H, r0: 0.9, r1: 1.0, gap: DOOR_HALF }],
  'storey.square': [
    { k: 'box', z: -0.95, y0: 0, y1: STOREY_H, hx: 1, hz: 0.05 }, { k: 'box', x: -0.95, y0: 0, y1: STOREY_H, hx: 0.05, hz: 1 },
    { k: 'box', x: 0.95, y0: 0, y1: STOREY_H, hx: 0.05, hz: 1 }, { k: 'box', z: 0.95, y0: 0, y1: STOREY_H, hx: 1, hz: 0.05, gap: DOOR_HALF },
  ],
  'door.open': [{ k: 'box', x: -1.225, z: 0.15, y0: 0, y1: 3.75, hx: 0.225, hz: 0.16 }, { k: 'box', x: 1.225, z: 0.15, y0: 0, y1: 3.75, hx: 0.225, hz: 0.16 }],
  jars: [{ k: 'cyl', y0: 0, y1: 1.1, r: 0.75 }],
  hearth: [{ k: 'cyl', y0: 0, y1: 0.24, r: 0.8 }],
  loom: [{ k: 'box', y0: 0, y1: 1.8, hx: 0.85, hz: 0.35 }],
  'nest.bowl': [{ k: 'cyl', y0: 0, y1: 0.66, r: 0.8 }],
  table: [{ k: 'cyl', y0: 0, y1: 0.78, r: 0.72 }],
  perch: [{ k: 'box', y0: 0, y1: 1.34, hx: 0.95, hz: 0.12 }],
  shelves: [{ k: 'box', y0: 0, y1: 2.2, hx: 0.9, hz: 0.21 }],
  'ramp.spiral': [{ k: 'helix', y0: 0, y1: STOREY_H - 0.25, r0: -0.97, r1: -0.02, a0: RAMP_START, a1: RAMP_START + RAMP_SWEEP }],
  // a nave's walls: its sides and far end (its front is the west front's)
  'body.hallOpen': [{ k: 'box', x: -0.97, y0: 0, y1: 1, hx: 0.03, hz: 1 }, { k: 'box', x: 0.97, y0: 0, y1: 1, hx: 0.03, hz: 1 }],
  // the apse: a half ring (its whole open half a doorway)
  'body.apse': [{ k: 'tube', y0: 0, y1: 1, r0: 0.96, r1: 1, gap: 1e6 }],
  'roof.apse': [{ k: 'cone', y0: 0, y1: 1, r: 1.04 }],
  'storey.roundOpen': [{ k: 'tube', y0: 0, y1: STOREY_H, r0: 0.9, r1: 1.0, gap: DOOR_HALF }],
  'body.roundOpen': [{ k: 'tube', y0: 0, y1: 1, r0: 0.9, r1: 1 }],
  // (the tiers' steps: ring walls stepped up toward the wall, open at the door)
  tiers: [
    { k: 'tube', y0: 0, y1: 1.35, r0: -1.1, r1: 0, gap: DOOR_HALF + 0.4 }, { k: 'tube', y0: 0, y1: 0.9, r0: -2.2, r1: -1.1, gap: DOOR_HALF + 0.4 },
    { k: 'tube', y0: 0, y1: 0.45, r0: -3.3, r1: -2.2, gap: DOOR_HALF + 0.4 },
  ],
  orrery: [{ k: 'cyl', y0: 0, y1: 2.6, r: 1.2 }],
  'storey.lattice': [
    { k: 'box', z: -0.95, y0: 0, y1: STOREY_H, hx: 1, hz: 0.05 }, { k: 'box', x: -0.95, y0: 0, y1: STOREY_H, hx: 0.05, hz: 1 },
    { k: 'box', x: 0.95, y0: 0, y1: STOREY_H, hx: 0.05, hz: 1 }, { k: 'box', z: 0.95, y0: 0, y1: STOREY_H, hx: 1, hz: 0.05, gap: DOOR_HALF },
  ],
  'storey.octagon': [{ k: 'tube', y0: 0, y1: STOREY_H, r0: 0.83, r1: 0.95, gap: DOOR_HALF }],
  'ramp.slab': [{ k: 'slab', y0: 0, y1: STOREY_H - 0.25, r0: -0.97, r1: -0.02, a0: RAMP_START, a1: RAMP_START + RAMP_SWEEP }],
  'pool.round': [{ k: 'tube', y0: 0, y1: 0.38, r0: 0.98, r1: 1.08 }],
  anvil: [{ k: 'cyl', y0: 0, y1: 0.85, r: 0.45 }],
  furnace: [{ k: 'cyl', y0: 0, y1: 2.5, r: 1.45 }],
  'shrine.small': [{ k: 'box', y0: 0, y1: 2.1, hx: 0.6, hz: 0.45 }],
  telescope: [{ k: 'cyl', y0: 0, y1: 2.6, r: 0.5 }],
  planter: [{ k: 'box', y0: 0, y1: 0.62, hx: 1.2, hz: 0.6 }],
  'table.long': [{ k: 'box', y0: 0, y1: 0.78, hx: 1.8, hz: 1.12 }],
  gallery: [{ k: 'box', z: 0.5, y0: -0.35, y1: 0, hx: 1, hz: 0.5 }, { k: 'box', z: 0.97, y0: 0, y1: 1.07, hx: 1, hz: 0.03 }],
  altar: [{ k: 'cyl', y0: 0, y1: 0.2, r: 2 }, { k: 'cyl', y0: 0.2, y1: 0.4, r: 1.6 }, { k: 'cyl', y0: 0.4, y1: 4.6, r: 1.0 }],
};

/** Per piece index, its colliders (empty: none). */
const BY_PIECE: Def[][] = PIECES.map((p) => DEFS[p.name] ?? []);

/**
 * A body over a storey (PFLAG.upper) is a hollow wall, ring or square: a hero climbing the storey's
 * ramp, its head up in the body's floor, is kept inside rather than thrown out through the wall.
 */
const UPPER: Record<string, Def[]> = {
  'body.round': [{ k: 'tube', y0: 0, y1: 1, r0: 0.9, r1: 1 }],
  'body.octagon': [{ k: 'tube', y0: 0, y1: 1, r0: 0.83, r1: 0.95 }],
  'body.lattice': [
    { k: 'box', z: -0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 }, { k: 'box', z: 0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 },
    { k: 'box', x: -0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 }, { k: 'box', x: 0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 },
  ],
  'body.belfry': [
    { k: 'box', z: -0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 }, { k: 'box', z: 0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 },
    { k: 'box', x: -0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 }, { k: 'box', x: 0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 },
  ],
  'body.arcade': [
    { k: 'box', z: -0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 }, { k: 'box', z: 0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 },
    { k: 'box', x: -0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 }, { k: 'box', x: 0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 },
  ],
  'body.square': [
    { k: 'box', z: -0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 }, { k: 'box', z: 0.95, y0: 0, y1: 1, hx: 1, hz: 0.05 },
    { k: 'box', x: -0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 }, { k: 'box', x: 0.95, y0: 0, y1: 1, hx: 0.05, hz: 1 },
  ],
};
const BY_PIECE_UPPER: Array<Def[] | undefined> = PIECES.map((p) => UPPER[p.name]);

/** Fields of a placed collider record. */
const F = 12;
// 0 kind · 1 x · 2 z · 3 y0 · 4 y1 · 5 cos yaw · 6 sin yaw · 7 a (radius / half x / ramp half x) · 8 b (inner radius / half z) · 9 c (ramp z0) · 10 d (ramp z1) · 11 reach (xz)

/** An Eyrie's colliders and the grid over them. */
export type ColliderSet = {
  data: Float64Array;
  count: number;
  /** cell key → indices of the colliders touching that cell (16 m cells). */
  cells: Map<number, Int32Array>;
};

const CELL = 16;
const cellKey = (i: number, j: number) => (i + 0x100000) * 0x200000 + (j + 0x100000);

/** Place one piece instance's colliders into `out` (records of F numbers). */
function place(p: PieceInst, out: number[]): void {
  const defs = (p.flags & PFLAG.upper && BY_PIECE_UPPER[p.piece]) || BY_PIECE[p.piece];
  if (!defs.length) return;
  const radial = PIECES[p.piece].kind === 'radial';
  const fixed = PIECES[p.piece].kind === 'fixed';
  const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
  for (const d of defs) {
    const sy = fixed ? p.sx : p.sy, sx = p.sx, sz = fixed ? p.sx : p.sz;
    const lx = 'x' in d && d.x !== undefined ? d.x * sx : 0, lz = 'z' in d && d.z !== undefined ? d.z * sz : 0;
    const x = p.x + c * lx + s * lz, z = p.z - s * lx + c * lz;
    const y0 = p.y + d.y0 * sy, y1 = p.y + d.y1 * sy;
    const rec = new Array(F).fill(0);
    rec[0] = K[d.k]; rec[1] = x; rec[2] = z; rec[3] = Math.min(y0, y1); rec[4] = Math.max(y0, y1); rec[5] = c; rec[6] = s;
    if (d.k === 'cyl') { rec[7] = d.r * sx; rec[11] = rec[7]; }
    else if (d.k === 'cone' || d.k === 'dome') { rec[7] = d.r * sx; rec[11] = rec[7]; }
    else if (d.k === 'tube') {
      // radial: offsets beyond the instance's radius
      const r1 = radial ? p.sx + d.r1 : d.r1 * sx, r0 = Math.max(0, radial ? p.sx + d.r0 : d.r0 * sx);
      rec[7] = r1; rec[8] = r0; rec[9] = d.gap ?? 0; rec[11] = r1;
    } else if (d.k === 'helix' || d.k === 'slab') {
      rec[7] = p.sx + d.r1; rec[8] = Math.max(0, p.sx + d.r0); rec[9] = d.a0; rec[10] = d.a1; rec[11] = rec[7];
    } else if (d.k === 'box' && d.gap) {
      // a wall with a doorway through it: its two sides, either side of the doorway
      const hx = d.hx * sx, g = Math.min(d.gap, hx);
      const half = 0.5 * (hx - g);
      for (const side of [-1, 1]) {
        const ox = lx + side * (g + half);
        const r2 = rec.slice();
        r2[1] = p.x + c * ox + s * lz; r2[2] = p.z - s * ox + c * lz;
        r2[7] = half; r2[8] = d.hz * sz; r2[11] = Math.hypot(half, r2[8]);
        out.push(...r2);
      }
      continue;
    } else if (d.k === 'box' || d.k === 'gable' || d.k === 'sag') { rec[7] = d.hx * sx; rec[8] = d.hz * sz; rec[11] = Math.hypot(rec[7], rec[8]); }
    else if (d.k === 'ramp') { rec[7] = d.hx * sx; rec[9] = d.z0 * sz; rec[10] = d.z1 * sz; rec[11] = Math.hypot(rec[7], Math.max(Math.abs(rec[9]), Math.abs(rec[10]))); }
    out.push(...rec);
  }
}

/** The colliders of an Eyrie's structures, and their grid (a generator: yields every few hundred structures). */
export function* buildColliders(structures: readonly Structure[]): Generator<void, ColliderSet, void> {
  const flat: number[] = [];
  let n = 0;
  for (const st of structures) {
    for (const p of st.pieces) place(p, flat);
    if (++n % 250 === 0) yield;
  }
  const data = new Float64Array(flat);
  const count = data.length / F;
  const lists = new Map<number, number[]>();
  for (let k = 0; k < count; k++) {
    if (k % 1500 === 1499) yield;
    const o = k * F, r = data[o + 11] + 1;
    const i0 = Math.floor((data[o + 1] - r) / CELL), i1 = Math.floor((data[o + 1] + r) / CELL);
    const j0 = Math.floor((data[o + 2] - r) / CELL), j1 = Math.floor((data[o + 2] + r) / CELL);
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const key = cellKey(i, j);
        let l = lists.get(key);
        if (!l) { l = []; lists.set(key, l); }
        l.push(k);
      }
    }
  }
  const cells = new Map<number, Int32Array>();
  for (const [key, l] of lists) cells.set(key, Int32Array.from(l));
  return { data, count, cells };
}

/** Whether a body of radius r at (dx, dz) from a ring wall's centre is in its doorway (passing through it). */
function inDoorway(d: Float64Array, o: number, dx: number, dz: number, r: number): boolean {
  const c = d[o + 5], s = d[o + 6];
  const u = dx * c - dz * s, v = dx * s + dz * c;
  return v > 0 && Math.abs(u) < d[o + 9] - 0.5 * r;
}

/** A body the colliders push: its position and velocity (x, z), radius r, from its feet up `height`. */
type CBody = { x: number; z: number; vx: number; vz: number };

/** How high a ledge is stepped onto (m), as the ruins'. */
const STEP = 0.5;

/** How far round a helix (0 … 1, from its foot) the bearing of (dx, dz) from its centre lies, or NaN past its ends. */
function helixT(d: Float64Array, o: number, dx: number, dz: number): number {
  // in the piece's own axes (local x along (cos yaw, −sin yaw), z along (sin yaw, cos yaw)); its bearings as the mesh's (cos a, sin a)
  const c = d[o + 5], s = d[o + 6];
  const u = dx * c - dz * s, v = dx * s + dz * c;
  const a0 = d[o + 9], a1 = d[o + 10];
  let a = Math.atan2(v, u);
  while (a < a0) a += 2 * Math.PI;
  while (a >= a0 + 2 * Math.PI) a -= 2 * Math.PI;
  return a <= a1 ? (a - a0) / (a1 - a0) : NaN;
}

/** The top of collider o at world (x, z) if (x, z) lies on it, else NaN. */
function topAt(d: Float64Array, o: number, x: number, z: number): number {
  const kind = d[o], dx = x - d[o + 1], dz = z - d[o + 2];
  if (kind === K.cyl) return dx * dx + dz * dz <= d[o + 7] * d[o + 7] ? d[o + 4] : NaN;
  if (kind === K.tube) {
    const r2 = dx * dx + dz * dz;
    return r2 <= d[o + 7] * d[o + 7] && r2 >= d[o + 8] * d[o + 8] ? d[o + 4] : NaN;
  }
  if (kind === K.helix || kind === K.slab) {
    const r2 = dx * dx + dz * dz;
    if (r2 > d[o + 7] * d[o + 7] || r2 < d[o + 8] * d[o + 8]) return NaN;
    const t = helixT(d, o, dx, dz);
    return t === t ? d[o + 3] + (d[o + 4] - d[o + 3]) * t : NaN;
  }
  if (kind === K.cone || kind === K.dome) {
    const r = Math.sqrt(dx * dx + dz * dz) / d[o + 7];
    if (r > 1) return NaN;
    const h = d[o + 4] - d[o + 3];
    return d[o + 3] + h * (kind === K.cone ? 1 - r : Math.sqrt(1 - r * r));
  }
  // boxes, ramps and gables: in their own axes
  const c = d[o + 5], s = d[o + 6];
  const u = dx * c - dz * s, v = dx * s + dz * c;
  if (Math.abs(u) > d[o + 7]) return NaN;
  if (kind === K.box) return Math.abs(v) <= d[o + 8] ? d[o + 4] : NaN;
  if (kind === K.gable) return Math.abs(v) <= d[o + 8] ? d[o + 3] + (d[o + 4] - d[o + 3]) * (1 - Math.abs(v) / d[o + 8]) : NaN;
  if (kind === K.sag) {
    if (Math.abs(v) > d[o + 8]) return NaN;
    const q = u / d[o + 7];
    return d[o + 4] - (d[o + 4] - d[o + 3]) * (1 - q * q);
  }
  // ramp: rises from y0 at z0 to y1 at z1
  const z0 = d[o + 9], z1 = d[o + 10];
  if (v < Math.min(z0, z1) || v > Math.max(z0, z1)) return NaN;
  return d[o + 3] + (d[o + 4] - d[o + 3]) * ((v - z0) / (z1 - z0));
}

/**
 * Push body b (feet at feetY, radius r, `height` tall) out of every collider of `set` it overlaps
 * between its feet and its head — sliding along (its velocity into the surface dropped) — and return
 * the highest top under it within a step, or −∞. Allocates nothing.
 */
export function collide(set: ColliderSet, b: CBody, feetY: number, r: number, height: number): number {
  const list = set.cells.get(cellKey(Math.floor(b.x / CELL), Math.floor(b.z / CELL)));
  if (!list) return -Infinity;
  const d = set.data;
  let floor = -Infinity;
  for (let li = 0; li < list.length; li++) {
    const o = list[li] * F;
    const kind = d[o];
    const dx = b.x - d[o + 1], dz = b.z - d[o + 2];
    if (dx * dx + dz * dz > (d[o + 11] + r + 0.5) ** 2) continue;
    const top = topAt(d, o, b.x, b.z);
    if (top === top && feetY >= top - STEP) { if (top > floor) floor = top; continue; }
    // Not standing on it: does it stand in the body's way (between feet and head)? (a rope bridge's deck never does)
    if (kind === K.sag) continue;
    const solidTop = top === top ? top : d[o + 4];
    if (feetY >= solidTop - STEP || feetY + height <= d[o + 3]) continue;
    let nx = 0, nz = 0, push = 0;
    if (kind === K.helix || kind === K.slab) {
      // beside the ramp below its walking surface: its open side stops the body, pushed in toward the
      // middle — a slab only where the body's head comes up into it (under it there is room)
      const dist = Math.sqrt(dx * dx + dz * dz);
      const t = helixT(d, o, dx, dz);
      if (dist <= d[o + 8] - r || t !== t) continue;
      if (kind === K.slab && feetY + height <= d[o + 3] + (d[o + 4] - d[o + 3]) * t - 0.4) continue;
      const ux = dist > 1e-6 ? dx / dist : 1, uz = dist > 1e-6 ? dz / dist : 0;
      nx = -ux; nz = -uz; push = dist - (d[o + 8] - r);
    } else if (kind === K.tube && d[o + 9] > 0 && inDoorway(d, o, dx, dz, r)) {
      continue;
    } else if (kind === K.cyl || kind === K.cone || kind === K.dome || kind === K.tube) {
      const dist = Math.sqrt(dx * dx + dz * dz);
      let R = d[o + 7];
      // (a roof narrows as it rises: at the feet's height)
      if (kind === K.cone) R *= Math.max(0, 1 - (feetY + STEP - d[o + 3]) / (d[o + 4] - d[o + 3]));
      if (kind === K.dome) { const t = Math.min(1, Math.max(0, (feetY + STEP - d[o + 3]) / (d[o + 4] - d[o + 3]))); R *= Math.sqrt(1 - t * t); }
      const ux = dist > 1e-6 ? dx / dist : 1, uz = dist > 1e-6 ? dz / dist : 0;
      if (kind === K.tube && d[o + 8] > 0 && dist < 0.5 * (d[o + 7] + d[o + 8])) {
        // inside a ring wall: kept within it
        const rin = d[o + 8];
        if (dist <= rin - r) continue;
        nx = -ux; nz = -uz; push = dist - (rin - r);
      } else {
        if (dist >= R + r) continue;
        nx = ux; nz = uz; push = R + r - dist;
      }
    } else {
      // a box (or a ramp or gable's sides) in its own axes
      const c = d[o + 5], s = d[o + 6];
      const u = dx * c - dz * s, v = dx * s + dz * c;
      let hz: number, vc: number;
      if (kind === K.ramp) { const z0 = Math.min(d[o + 9], d[o + 10]), z1 = Math.max(d[o + 9], d[o + 10]); hz = 0.5 * (z1 - z0); vc = v - 0.5 * (z0 + z1); }
      else { hz = d[o + 8]; vc = v; }
      const hx = d[o + 7];
      const cu = Math.max(-hx, Math.min(hx, u)), cv = Math.max(-hz, Math.min(hz, vc));
      const du = u - cu, dv = vc - cv;
      const dist = Math.sqrt(du * du + dv * dv);
      if (dist >= r) continue;
      let lu: number, lv: number;
      if (dist > 1e-6) { lu = du / dist; lv = dv / dist; push = r - dist; }
      else {
        const pu = hx - Math.abs(u), pv = hz - Math.abs(vc);
        if (pu < pv) { lu = u < 0 ? -1 : 1; lv = 0; push = pu + r; } else { lu = 0; lv = vc < 0 ? -1 : 1; push = pv + r; }
      }
      // back to world: local u along (c, −s), v along (s, c)
      nx = c * lu + s * lv; nz = -s * lu + c * lv;
    }
    b.x += nx * push;
    b.z += nz * push;
    const vn = b.vx * nx + b.vz * nz;
    if (vn < 0) { b.vx -= vn * nx; b.vz -= vn * nz; }
  }
  return floor;
}
