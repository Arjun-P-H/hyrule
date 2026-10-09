/**
 * Structures: what the Eyries' grammars build (skyworld/eyrie/), as the
 * renderer draws it (gpu/structures.ts) and the hero walks it.
 * ---------------------------------------------------------------------------
 * A structure is one culling group — a roost, a hall, a bridge span, a plaza
 * — with a bounding sphere and its piece instances: a kit piece (kit/pieces.ts)
 * scaled per axis, turned about y by its yaw, placed in world metres, in a
 * palette (kit/palettes.ts). The renderer picks one level of detail for all of
 * a structure's pieces, so its parts never disagree; a piece is drawn at its
 * structure's level when that level is at most its `detail` (props near only).
 *
 * Yaw: local (x, z) turns to world (x cos + z sin, −x sin + z cos), as the ruins'
 * boxes, so local +z faces (sin yaw, cos yaw): the camera's forward at that yaw.
 */

import { buildKit } from './kit/kit';
import { PIECES, type Lod } from './kit/pieces';

/** Per-instance flags (the shader's): bits 22 … 31 of the packed word. */
export const PFLAG = {
  /** The body does not start at the ground: its first floor has windows too. */
  upper: 1,
  /** Broken and weathered (Hushfall; the Long Gale's marks). */
  ruined: 2,
  /** Windows and lanterns stay dark (abandoned). */
  dark: 4,
  /** Material variants: marble and dressed stone in vermilion lacquer; stone in timber; stone in white plaster. */
  lacquer: 8,
  timber: 16,
  plaster: 32,
  /** Inside a building (furnishings): lit as an inside is, by the room's share of the sky and its lamps, never the sun. */
  indoor: 64,
} as const;

export type PieceInst = {
  piece: number;
  x: number; y: number; z: number;  // world m
  yaw: number;
  sx: number; sy: number; sz: number;
  palette: number;
  seed: number;                      // 16 bits of variety (window lights, tints)
  flags: number;                     // PFLAG
  /** Coarsest structure level it is drawn at (default: its piece's). */
  detail?: Lod;
};

/**
 * Light and air a structure gives off (gpu/effects.ts draws them): the Vaelspire's Thread of light,
 * smoke from a forge's chimney, the Gyre's Eye turning over Rimewatch.
 */
export type Effect =
  | { k: 'beam'; x: number; y: number; z: number; r: number }
  | { k: 'smoke'; x: number; y: number; z: number; r: number }
  | { k: 'vortex'; x: number; y: number; z: number; r: number };

export type Structure = {
  x: number; y: number; z: number;   // bounding sphere's centre (world m)
  r: number;                         // and radius
  pieces: PieceInst[];
  fx?: Effect[];
};

/** Bytes per instance in the GPU's buffer (structures.wgsl.ts StInst). */
export const INSTANCE_BYTES = 32;

/** World box of a piece instance (x0, y0, z0, x1, y1, z1) into `out`. */
export function pieceBox(p: PieceInst, out: Float64Array): Float64Array {
  const b = buildKit().bounds[p.piece];
  const def = PIECES[p.piece];
  if (def.kind === 'radial') {
    const R = p.sx + b.radial;
    out[0] = p.x - R; out[2] = p.z - R; out[3] = p.x + R; out[5] = p.z + R;
    out[1] = p.y + b.min[1] * p.sy; out[4] = p.y + b.max[1] * p.sy;
    return out;
  }
  // The scaled box's corners turned by the yaw: its extent along world x and z.
  const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
  const x0 = b.min[0] * p.sx, x1 = b.max[0] * p.sx, z0 = b.min[2] * p.sz, z1 = b.max[2] * p.sz;
  const mx = 0.5 * (x0 + x1), mz = 0.5 * (z0 + z1), hx = 0.5 * (x1 - x0), hz = 0.5 * (z1 - z0);
  const ex = Math.abs(c) * hx + Math.abs(s) * hz, ez = Math.abs(s) * hx + Math.abs(c) * hz;
  const cx = p.x + c * mx + s * mz, cz = p.z - s * mx + c * mz;
  out[0] = cx - ex; out[3] = cx + ex; out[2] = cz - ez; out[5] = cz + ez;
  out[1] = p.y + b.min[1] * p.sy; out[4] = p.y + b.max[1] * p.sy;
  return out;
}

const box = new Float64Array(6);

/** A structure from its pieces: its bounding sphere encloses every piece's box. */
export function makeStructure(pieces: PieceInst[]): Structure {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (const p of pieces) {
    pieceBox(p, box);
    x0 = Math.min(x0, box[0]); y0 = Math.min(y0, box[1]); z0 = Math.min(z0, box[2]);
    x1 = Math.max(x1, box[3]); y1 = Math.max(y1, box[4]); z1 = Math.max(z1, box[5]);
  }
  const x = 0.5 * (x0 + x1), y = 0.5 * (y0 + y1), z = 0.5 * (z0 + z1);
  return { x, y, z, r: 0.5 * Math.hypot(x1 - x0, y1 - y0, z1 - z0), pieces };
}

/** Pack a piece instance into the GPU's layout at word `o` of f / u, chunk-local to (ox, oz). */
export function packInstance(p: PieceInst, f: Float32Array, u: Uint32Array, o: number, ox: number, oz: number): void {
  f[o] = p.x - ox; f[o + 1] = p.y; f[o + 2] = p.z - oz; f[o + 3] = p.yaw;
  f[o + 4] = p.sx; f[o + 5] = p.sy; f[o + 6] = p.sz;
  u[o + 7] = (p.palette & 63) | ((p.seed & 0xffff) << 6) | ((p.flags & 1023) << 22);
}
