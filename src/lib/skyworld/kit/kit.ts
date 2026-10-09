/**
 * The kit built: every piece (pieces.ts) at its three levels of detail packed
 * into one vertex buffer and one index buffer, with each mesh's index range
 * and each piece's unit-space bounds (for culling its instances).
 */

import { KIT_VERTEX_BYTES, KitBuilder, type V3 } from './mesh';
import { PIECES, type Lod } from './pieces';

type KitMesh = { firstIndex: number; indexCount: number };
type PieceBounds = {
  /** Unit-space box of the piece's level 0 (radial pieces: x and z hold the offsets beyond the radius, ± the largest). */
  min: V3; max: V3;
  /** Radial pieces: the largest offset beyond the instance's radius (m); else 0. */
  radial: number;
};
export type Kit = {
  vertices: ArrayBuffer;
  indices: Uint32Array<ArrayBuffer>;
  /** Mesh of piece p at level l: meshes[3 p + l]. */
  meshes: KitMesh[];
  bounds: PieceBounds[];
  /** Vertices in all (for tests and stats). */
  vertexCount: number;
};

const KIT_LODS = 3;

let cached: Kit | null = null;

/** The kit, built once (~50 ms). */
export function buildKit(): Kit {
  if (cached) return cached;
  const b = new KitBuilder();
  const meshes: KitMesh[] = [];
  const bounds: PieceBounds[] = [];
  for (const piece of PIECES) {
    for (let lod = 0 as Lod; lod < KIT_LODS; lod = (lod + 1) as Lod) {
      const v0 = b.count, i0 = b.idx.length;
      b.mat = 0; b.facade = 0; b.anim = 0; b.uAxis = 0; b.vAxis = 1;
      piece.build(b, lod);
      meshes.push({ firstIndex: i0, indexCount: b.idx.length - i0 });
      if (lod === 0) {
        const min: V3 = [Infinity, Infinity, Infinity], max: V3 = [-Infinity, -Infinity, -Infinity];
        let radial = 0;
        for (let v = v0; v < b.count; v++) {
          const w = b.pos[4 * v + 3];
          const isRadial = piece.kind === 'radial';
          if (isRadial) radial = Math.max(radial, w);
          for (let c = 0; c < 3; c++) {
            const x = isRadial && c !== 1 ? b.pos[4 * v + c] * (1 + w) : b.pos[4 * v + c];
            min[c] = Math.min(min[c], x);
            max[c] = Math.max(max[c], x);
          }
        }
        if (!(min[0] <= max[0])) { min.fill(0); max.fill(0); }
        bounds.push({ min, max, radial: piece.kind === 'radial' ? radial : 0 });
      }
    }
  }
  const { vertices, indices, count } = b.finish();
  cached = { vertices, indices, meshes, bounds, vertexCount: count };
  return cached;
}

export { KIT_VERTEX_BYTES };
