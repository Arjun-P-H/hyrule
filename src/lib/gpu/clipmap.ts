/**
 * Clipmap scheduling — which texels of the terrain cache to (re)generate.
 * ---------------------------------------------------------------------------
 * The world stays the pure function E(x, z, seed); the clipmap caches it on
 * nested square grids around the camera. Level l has spacing
 * s_l = spacing0 · 2^l and holds the n × n lattice points (i·s_l, j·s_l) for
 * i ∈ [i0, i0 + n), j ∈ [j0, j0 + n), a window re-centred on the camera
 * every frame. Storage is toroidal: lattice point (i, j) lives at texel
 * (i mod n, j mod n), so a window that slides by k texels only generates the
 * k newly exposed rows or columns and never moves the rest.
 *
 * Everything here is integer bookkeeping on the CPU; the GPU generator and
 * the renderer read the results (see terrainRenderer.ts).
 */

export type ClipmapSpec = {
  /** Texels per side of every level (power of two). */
  n: number;
  levels: number;
  /** Metres between lattice points at level 0 (power of two). */
  spacing0: number;
  /** Texels kept clear of the window edge: bilinear taps, gradient taps, level blend. */
  margin: number;
};

/** World lattice index of a level window's first texel along x and z. */
export type LevelWindow = { i0: number; j0: number };

/** A rectangle of lattice points to generate, in world lattice coordinates of `level`. */
export type UpdateRect = { level: number; i0: number; j0: number; w: number; h: number };

export function levelSpacing(spec: ClipmapSpec, level: number): number {
  return spec.spacing0 * 2 ** level;
}

/**
 * Chebyshev radius around the camera inside which `level` can be sampled with
 * all its taps inside the window.
 */
export function usableRadius(spec: ClipmapSpec, level: number): number {
  return (spec.n / 2 - spec.margin) * levelSpacing(spec, level);
}

/** Window of `level` centred on the camera (world metres, x and z). */
export function windowFor(spec: ClipmapSpec, level: number, camX: number, camZ: number): LevelWindow {
  const s = levelSpacing(spec, level);
  return { i0: Math.floor(camX / s) - spec.n / 2, j0: Math.floor(camZ / s) - spec.n / 2 };
}

/** Non-negative remainder: the toroidal texel index of lattice index `i`. */
export function wrap(i: number, n: number): number {
  return ((i % n) + n) % n;
}

/**
 * Lattice points in `next` that `prev` does not hold. At most two rectangles:
 * the newly exposed columns over the full new height, and the newly exposed
 * rows over the columns both windows share. A jump of a full window or more
 * (teleport, respawn, first frame) regenerates the whole level.
 */
export function windowDelta(n: number, level: number, prev: LevelWindow | null, next: LevelWindow): UpdateRect[] {
  if (!prev || Math.abs(next.i0 - prev.i0) >= n || Math.abs(next.j0 - prev.j0) >= n) {
    return [{ level, i0: next.i0, j0: next.j0, w: n, h: n }];
  }
  const rects: UpdateRect[] = [];
  const keepI0 = Math.max(prev.i0, next.i0);
  const keepI1 = Math.min(prev.i0, next.i0) + n;
  if (next.i0 < prev.i0) rects.push({ level, i0: next.i0, j0: next.j0, w: prev.i0 - next.i0, h: n });
  if (next.i0 > prev.i0) rects.push({ level, i0: prev.i0 + n, j0: next.j0, w: next.i0 - prev.i0, h: n });
  if (next.j0 < prev.j0) rects.push({ level, i0: keepI0, j0: next.j0, w: keepI1 - keepI0, h: prev.j0 - next.j0 });
  if (next.j0 > prev.j0) rects.push({ level, i0: keepI0, j0: prev.j0 + n, w: keepI1 - keepI0, h: next.j0 - prev.j0 });
  return rects;
}

/**
 * The clipmap's CPU-side state: the window each level currently holds.
 * `update` moves every window to the camera and returns what to generate.
 */
export class ClipmapScheduler {
  readonly spec: ClipmapSpec;
  private windows: Array<LevelWindow | null>;

  constructor(spec: ClipmapSpec) {
    if (!Number.isInteger(Math.log2(spec.n))) throw new Error(`clipmap n must be a power of two: ${spec.n}`);
    if (!Number.isInteger(Math.log2(spec.spacing0))) throw new Error(`clipmap spacing0 must be a power of two: ${spec.spacing0}`);
    if (spec.margin < 3 || spec.margin >= spec.n / 4) throw new Error(`clipmap margin out of range: ${spec.margin}`);
    this.spec = spec;
    this.windows = new Array(spec.levels).fill(null);
  }

  /** Forget every level (next update regenerates everything). */
  invalidate(): void {
    this.windows.fill(null);
  }

  window(level: number): LevelWindow | null {
    return this.windows[level];
  }

  update(camX: number, camZ: number): UpdateRect[] {
    const rects: UpdateRect[] = [];
    for (let l = 0; l < this.spec.levels; l++) {
      const next = windowFor(this.spec, l, camX, camZ);
      const prev = this.windows[l];
      if (prev && prev.i0 === next.i0 && prev.j0 === next.j0) continue;
      rects.push(...windowDelta(this.spec.n, l, prev, next));
      this.windows[l] = next;
    }
    return rects;
  }
}

/** Texels a set of rectangles covers (the generation work of one update). */
export function rectTexels(rects: readonly UpdateRect[]): number {
  let s = 0;
  for (const r of rects) s += r.w * r.h;
  return s;
}
