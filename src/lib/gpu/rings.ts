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
