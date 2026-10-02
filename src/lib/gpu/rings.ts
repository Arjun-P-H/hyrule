/**
 * Geometry-clipmap ring layout and index buffers (pure; see rasterRenderer.ts).
 */

/** Triangle indices of a (2k)² cell grid, skipping the k² cells from `hole` (cell coords). */
export function ringIndices(k: number, hole: [number, number] | null): Uint32Array<ArrayBuffer> {
  const side = 2 * k + 1;
  const cells = hole ? 4 * k * k - k * k : 4 * k * k;
  const out = new Uint32Array(cells * 6);
  let o = 0;
  for (let cj = 0; cj < 2 * k; cj++) {
    for (let ci = 0; ci < 2 * k; ci++) {
      if (hole && ci >= hole[0] && ci < hole[0] + k && cj >= hole[1] && cj < hole[1] + k) continue;
      const v00 = cj * side + ci, v10 = v00 + 1, v01 = v00 + side, v11 = v01 + 1;
      // Diagonal v00–v11 in every cell: collapsing odd vertices onto their
      // lower even neighbours then yields exactly the coarser ring's cells.
      out[o++] = v00; out[o++] = v11; out[o++] = v10;
      out[o++] = v00; out[o++] = v01; out[o++] = v11;
    }
  }
  return out;
}

type RingPlacement = {
  /** Level-0 lattice units per cell. */
  step: number;
  /** First vertex and centre, in world level-0 lattice units (x, z). */
  corner: [number, number];
  center: [number, number];
  /** Where the next finer ring's square starts, minus k/2 (0 or 1 per axis); unused for ring 0. */
  hole: [number, number];
};

/**
 * Ring r: centre = the camera snapped to an even cell of spacing
 * spacing0 · 2^r · stride, extent ± k cells. The finer ring r − 1 then covers
 * cells [k/2 + hole, 3k/2 + hole) of ring r exactly.
 */
export function ringLayout(x: number, z: number, spacing0: number, k: number, stride: number, rings: number): RingPlacement[] {
  const out: RingPlacement[] = [];
  for (let r = 0; r < rings; r++) {
    const step = 2 ** r * stride;
    const S = spacing0 * step;
    const cx = x / S, cz = z / S;
    const ax = 2 * Math.floor(cx / 2), az = 2 * Math.floor(cz / 2);
    out.push({
      step,
      corner: [(ax - k) * step, (az - k) * step],
      center: [ax * step, az * step],
      hole: [Math.floor(cx) - ax, Math.floor(cz) - az],
    });
  }
  return out;
}

/** Cells per side of a culling block of a ring (rasterRenderer.ts draws a ring's blocks in view). */
export const RING_BLOCK = 16;

/**
 * A ring's blocks in the order its index buffer lays them out: round its
 * centre by angle, so the blocks a view sees (a wedge from the camera, which
 * sits by the centre) lie in one or two runs. `order[s]` is the block (row
 * major, (2k / RING_BLOCK)² of them) in slot s; `slot[b]` the inverse.
 */
export function ringBlockOrder(k: number): { order: Uint16Array; slot: Uint16Array } {
  const nb = (2 * k) / RING_BLOCK;
  const angle = (b: number) => Math.atan2(((Math.floor(b / nb) + 0.5) * RING_BLOCK) - k, ((b % nb) + 0.5) * RING_BLOCK - k);
  const order = Uint16Array.from({ length: nb * nb }, (_, b) => b).sort((a, b) => angle(a) - angle(b));
  const slot = new Uint16Array(nb * nb);
  order.forEach((b, s) => { slot[b] = s; });
  return { order, slot };
}

/**
 * Whether a camera-relative box [lo, hi] can show in the view: outside none
 * of the four side planes of the frustum (basis F as packFrame lays it out:
 * right F[0..2], up F[4..6], forward F[8..10]; tx, ty the tangents of the
 * half angles) nor wholly behind the camera.
 */
export function boxInView(F: ArrayLike<number>, tx: number, ty: number,
  lx: number, ly: number, lz: number, hx: number, hy: number, hz: number): boolean {
  // Each plane's inward normal n: the box is outside when its corner farthest along n is behind it.
  const out = (nx: number, ny: number, nz: number) =>
    (nx >= 0 ? hx : lx) * nx + (ny >= 0 ? hy : ly) * ny + (nz >= 0 ? hz : lz) * nz < 0;
  const fx = F[8], fy = F[9], fz = F[10];
  if (out(fx, fy, fz)) return false;
  if (out(tx * fx - F[0], tx * fy - F[1], tx * fz - F[2])) return false;
  if (out(tx * fx + F[0], tx * fy + F[1], tx * fz + F[2])) return false;
  if (out(ty * fx - F[4], ty * fy - F[5], ty * fz - F[6])) return false;
  if (out(ty * fx + F[4], ty * fy + F[5], ty * fz + F[6])) return false;
  return true;
}
