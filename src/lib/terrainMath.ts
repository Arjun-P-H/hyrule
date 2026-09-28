/**
 * W(x,y) — CPU physics twin, TypeScript reference implementation (double precision).
 * ---------------------------------------------------------------------------
 * This is the exact math of the GPU raymarcher (demoShader.ts /
 * terrain_raymarch.gdshader), evaluated on the CPU:
 *   - walk-mode collision in the WebGL demo (spec §31 movement loop),
 *   - the hillshade relief map on the hero panel,
 *   - the live worked numerical example (spec §30).
 *
 * PARITY NOTES
 *  - Hashing is strict uint32 integer math (spec §27): Math.imul + >>> 0.
 *  - The GPU evaluates the same functions in fp32 on the fp32 global
 *    coordinate `chunk_local + u_chunk_origin`, so its precision follows the
 *    distance from the world origin. Measured by GPU readback (`bun run bench
 *    --suites=parity`): max |E_gpu − E_cpu| 0.6 mm at 3 km, 3 mm at 12 km,
 *    15 mm at 108 km — absorbed by PLAYER.groundBias (see constants.ts).
 *  - Field lookup convention: y-up world, terrain field is E(p.x, p.z);
 *    the spec's math z (elevation) maps to engine y.
 *  - Allocation: JSC and SpiderMonkey never allocate here. V8 boxes a double
 *    (16-byte HeapNumber) at every non-inlined call boundary, ~1.2 KB per
 *    elevation() call (bench/alloc.ts); restructuring for V8 inlining was
 *    measured slower in both engines and was not adopted.
 */

import {
  F, MACRO, MOUNTAIN, WARP, VALLEY, CONTINENT, REGION, ANOMALY,
  CLIMATE, MARCH, SHADOW, MAX_WORLD_H, MAX_DIST, PLAYER,
} from "./constants";

/* ------------------------------------------------------------------ */
/* §4 — Seed & deterministic hashing (uint32, Murmur3 finalizer)       */
/* ------------------------------------------------------------------ */

/** §4 — MurmurHash3 finalizer — avalanche. Input/output are uint32 bit patterns. */
export function hashU(x: number): number {
  let h = x | 0;
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * §4 — H(x_i, y_i, s) = finalize(x_i ⊕ (y_i · 0x27D4EB2F) ⊕ (s · 0x9E3779B9))
 * xi/yi are integer lattice coords (may be negative — two's-complement wrap).
 */
export function hash3(xi: number, yi: number, seed: number): number {
  const h =
    (xi | 0) ^
    Math.imul(yi | 0, 0x27d4eb2f) ^
    Math.imul(seed | 0, 0x9e3779b9);
  return hashU(h);
}

const INV_U32 = 1.0 / 4294967295.0;
const INV_U32_X2 = 2.0 / 4294967295.0;

/** §4 — Uniform in [0,1] from an integer lattice cell. */
export function hash01(xi: number, yi: number, seed: number): number {
  return hash3(xi, yi, seed) * INV_U32;
}

/** §4 — Feature-point coords in [0,1]² packed from the two 12-bit halves of one hash. */
export function cellPoint(ix: number, iy: number, seed: number): [number, number] {
  const h = hash3(ix, iy, seed);
  return [(h & 0xfff) / 4095.0, ((h >>> 12) & 0xfff) / 4095.0];
}

/** §4 — Cell type byte packed from the top 8 bits of the same hash (0..1). */
export function cellType(ix: number, iy: number, seed: number): number {
  return (hash3(ix, iy, seed) >>> 24) / 255.0;
}

/* ------------------------------------------------------------------ */
/* §5 — Quintic value noise (C² continuous) & Nyquist weights          */
/* ------------------------------------------------------------------ */

/** §5 — Quintic C² polynomial S(t) = 6t⁵ − 15t⁴ + 10t³ evaluated via Horner's rule. */
export function quintic(t: number): number {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

/** §5 — Quintic value noise with bilinear interpolation and Dyadic Murmur3 lattice. */
export function vnoise(x: number, y: number, seed: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
  const uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

  const ux0 = ix | 0;
  const ux1 = (ux0 + 1) | 0;
  const iy0 = iy | 0;
  const uy0 = Math.imul(iy0, 0x27d4eb2f);
  const uy1 = (uy0 + 0x27d4eb2f) | 0;
  const sTerm = Math.imul(seed | 0, 0x9e3779b9);

  const h0 = (uy0 ^ sTerm) | 0;
  const h1 = (uy1 ^ sTerm) | 0;

  let ha = (ux0 ^ h0) | 0;
  ha ^= ha >>> 16; ha = Math.imul(ha, 0x85ebca6b); ha ^= ha >>> 13; ha = Math.imul(ha, 0xc2b2ae35); ha ^= ha >>> 16;
  const a = (ha >>> 0) * INV_U32_X2;

  let hb = (ux1 ^ h0) | 0;
  hb ^= hb >>> 16; hb = Math.imul(hb, 0x85ebca6b); hb ^= hb >>> 13; hb = Math.imul(hb, 0xc2b2ae35); hb ^= hb >>> 16;
  const b = (hb >>> 0) * INV_U32_X2;

  let hc = (ux0 ^ h1) | 0;
  hc ^= hc >>> 16; hc = Math.imul(hc, 0x85ebca6b); hc ^= hc >>> 13; hc = Math.imul(hc, 0xc2b2ae35); hc ^= hc >>> 16;
  const c = (hc >>> 0) * INV_U32_X2;

  let hd = (ux1 ^ h1) | 0;
  hd ^= hd >>> 16; hd = Math.imul(hd, 0x85ebca6b); hd ^= hd >>> 13; hd = Math.imul(hd, 0xc2b2ae35); hd ^= hd >>> 16;
  const d = (hd >>> 0) * INV_U32_X2;

  return (a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy) - 1.0;
}

/**
 * Nyquist octave weight (spec §5): fade out between 0.5 and 1 cycle/pixel.
 * Δp(t) = t · 2tan(FOV/2) / Res_y is the world-space pixel footprint.
 */
function nyqW(freq: number, dp: number): number {
  return Math.min(1, Math.max(0, 2.0 - (freq * dp) * 2.0));
}

// Precomputed totalAsum constants: sum_{i=0}^{octaves-1} (initial_amp * gain^i)
const ASUM_MACRO = 1.99609375; // 9 octaves, gain 0.5, amp 1.0
const ASUM_RIDGE = 0.5 * (1.0 - Math.pow(0.48, 5)) / (1.0 - 0.48); // 5 octaves, gain 0.48, amp 0.5
const ASUM_WARP = 1.5; // 2 octaves, gain 0.5, amp 1.0

/**
 * §5/§6/§14 — octave tables (frequency multipliers and amplitudes) for the
 * macro fBm, ridged multifractal and domain warp; mirrored verbatim by the
 * shader's MACRO_/RIDGE_/WARP_ const arrays (tests/floatingOrigin.test.ts).
 */
export const MACRO_FREQ = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0] as const;
export const MACRO_AMP  = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625] as const;
export const RIDGE_FREQ = [1.0, 2.17, 4.7089, 10.218313, 22.17373921] as const;
export const RIDGE_AMP  = [0.5, 0.24, 0.1152, 0.055296, 0.02654208] as const;
export const WARP_FREQ  = [1.0, 2.0] as const;
export const WARP_AMP   = [1.0, 0.5] as const;

/**
 * fBm with Nyquist attenuation + amplitude renormalization (spec §5).
 * `px, py` are pre-scaled coordinates (p · f0); `f0` is the base world
 * frequency so the Nyquist weight uses the ABSOLUTE octave frequency
 * f_i = f0 · lac^i against the metre-space footprint Δp.
 */
function fbm(
  x: number, y: number, seed: number,
  octaves: number, lacunarity: number, gain: number, dp: number,
  f0 = 1.0,
): number {
  // Fast path for near-field / physics / hillshade (dp <= 0):
  // All Nyquist weights are identically 1.0, wsum === asum, zero division!
  if (dp <= 0.0) {
    if (octaves === 9 && lacunarity === 2.0 && gain === 0.5) {
      return vnoise(x, y, seed)
        + 0.5 * vnoise(x * 2.0, y * 2.0, seed + 40503)
        + 0.25 * vnoise(x * 4.0, y * 4.0, seed + 81006)
        + 0.125 * vnoise(x * 8.0, y * 8.0, seed + 121509)
        + 0.0625 * vnoise(x * 16.0, y * 16.0, seed + 162012)
        + 0.03125 * vnoise(x * 32.0, y * 32.0, seed + 202515)
        + 0.015625 * vnoise(x * 64.0, y * 64.0, seed + 243018)
        + 0.0078125 * vnoise(x * 128.0, y * 128.0, seed + 283521)
        + 0.00390625 * vnoise(x * 256.0, y * 256.0, seed + 324024);
    }
    if (octaves === 2 && lacunarity === 2.0 && gain === 0.5) {
      return vnoise(x, y, seed) + 0.5 * vnoise(x * 2.0, y * 2.0, seed + 40503);
    }
    let freq = 1.0, amp = 1.0, sum = 0.0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * vnoise(x * freq, y * freq, seed + i * 0x9e37);
      freq *= lacunarity;
      amp *= gain;
    }
    return sum;
  }

  // Precomputed or evaluated totalAsum
  const totalAsum = (gain === 0.5 && octaves === 9) ? ASUM_MACRO
    : (gain === 0.5 && octaves === 2) ? ASUM_WARP
    : (1.0 - Math.pow(gain, octaves)) / (1.0 - gain);

  const f0_dp_2 = f0 * dp * 2.0;

  if (octaves === 9 && lacunarity === 2.0 && gain === 0.5) {
    let sum = 0.0, wsum = 0.0;
    for (let i = 0; i < 9; i++) {
      const freq = MACRO_FREQ[i];
      const w = Math.min(1, Math.max(0, 2.0 - freq * f0_dp_2));
      if (w <= 0.0) break;
      const amp = MACRO_AMP[i];
      sum += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
      wsum += amp * w;
    }
    return sum * (totalAsum / Math.max(wsum, 1e-6));
  }

  if (octaves === 2 && lacunarity === 2.0 && gain === 0.5) {
    let sum = 0.0, wsum = 0.0;
    for (let i = 0; i < 2; i++) {
      const freq = WARP_FREQ[i];
      const w = Math.min(1, Math.max(0, 2.0 - freq * f0_dp_2));
      if (w <= 0.0) break;
      const amp = WARP_AMP[i];
      sum += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
      wsum += amp * w;
    }
    return sum * (totalAsum / Math.max(wsum, 1e-6));
  }

  let freq = 1.0, amp = 1.0, sum = 0.0, wsum = 0.0;
  for (let i = 0; i < octaves; i++) {
    const w = Math.min(1, Math.max(0, 2.0 - freq * f0_dp_2));
    if (w <= 0.0) break;
    sum += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
    wsum += amp * w;
    freq *= lacunarity;
    amp *= gain;
  }
  return sum * (totalAsum / Math.max(wsum, 1e-6));
}

/**
 * Ridged multifractal (spec §6): (1 − |N|)² octaves; ridged features double
 * the effective bandwidth, so the Nyquist weight uses 2f.
 */
function ridged(
  x: number, y: number, seed: number,
  octaves: number, lacunarity: number, gain: number, dp: number,
  f0 = 1.0,
): number {
  // Fast path for near-field / physics (dp <= 0)
  if (dp <= 0.0) {
    if (octaves === 5 && lacunarity === 2.17 && gain === 0.48) {
      const n0 = 1.0 - Math.abs(vnoise(x, y, seed));
      const n1 = 1.0 - Math.abs(vnoise(x * 2.17, y * 2.17, seed + 34283));
      const n2 = 1.0 - Math.abs(vnoise(x * 4.7089, y * 4.7089, seed + 68566));
      const n3 = 1.0 - Math.abs(vnoise(x * 10.218313, y * 10.218313, seed + 102849));
      const n4 = 1.0 - Math.abs(vnoise(x * 22.17373921, y * 22.17373921, seed + 137132));
      return 0.5 * (n0 * n0) + 0.24 * (n1 * n1) + 0.1152 * (n2 * n2) + 0.055296 * (n3 * n3) + 0.02654208 * (n4 * n4);
    }
    let freq = 1.0, amp = 0.5, sum = 0.0;
    for (let i = 0; i < octaves; i++) {
      const n = 1.0 - Math.abs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
      sum += amp * (n * n);
      freq *= lacunarity;
      amp *= gain;
    }
    return sum;
  }

  const totalAsum = (gain === 0.48 && octaves === 5) ? ASUM_RIDGE
    : 0.5 * (1.0 - Math.pow(gain, octaves)) / (1.0 - gain);

  const f0_dp_4 = 4.0 * f0 * dp;

  if (octaves === 5 && lacunarity === 2.17 && gain === 0.48) {
    let sum = 0.0, wsum = 0.0;
    for (let i = 0; i < 5; i++) {
      const freq = RIDGE_FREQ[i];
      const w = Math.min(1, Math.max(0, 2.0 - freq * f0_dp_4));
      if (w <= 0.0) break;
      const n = 1.0 - Math.abs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
      const amp = RIDGE_AMP[i];
      sum += amp * w * (n * n);
      wsum += amp * w;
    }
    return sum * (totalAsum / Math.max(wsum, 1e-6));
  }

  let freq = 1.0, amp = 0.5, sum = 0.0, wsum = 0.0;
  for (let i = 0; i < octaves; i++) {
    const w = Math.min(1, Math.max(0, 2.0 - freq * f0_dp_4));
    if (w <= 0.0) break;
    const n = 1.0 - Math.abs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
    sum += amp * w * (n * n);
    wsum += amp * w;
    freq *= lacunarity;
    amp *= gain;
  }
  return sum * (totalAsum / Math.max(wsum, 1e-6));
}

/* ------------------------------------------------------------------ */
/* §7 / §12 / §13 — Cellular (Worley) fields                           */
/* ------------------------------------------------------------------ */

type VoronoiResult = {
  f1: number;          // nearest feature distance (cell units)
  f2: number;          // second nearest
  edge: number;        // F2 − F1 (spec §7 edge distance)
  centerX: number;     // nearest feature point, input-space coords
  centerY: number;
  cellHash: number;    // type byte (top 8 bits) of the owning cell
};

// 3x3 Worley cell visitation order: center cell (0, 0) first for tight initial bounds,
// followed by 4 orthogonal neighbors, then 4 diagonal corners with distance pruning.
const CELL_I = [0,  1,  0, -1,  0,  1, -1,  1, -1];
const CELL_J = [0,  0,  1,  0, -1,  1,  1, -1, -1];

/**
 * §7 — 3×3 Worley scan. Compares squared distances to avoid 7 Math.sqrt calls,
 * computes hash3 once per cell (extracting point and type together), and
 * avoids heap allocations.
 */
export function voronoi(x: number, y: number, seed: number): VoronoiResult {
  const ipx = Math.floor(x), ipy = Math.floor(y);
  const fx = x - ipx, fy = y - ipy;
  let f1_sq = 64.0, f2_sq = 64.0;
  let bcx = 0, bcy = 0, btype = 0;
  for (let k = 0; k < 9; k++) {
    const i = CELL_I[k], j = CELL_J[k];
    const minDx = i === 1 ? (1.0 - fx) : (i === -1 ? fx : 0.0);
    const minDy = j === 1 ? (1.0 - fy) : (j === -1 ? fy : 0.0);
    if (minDx * minDx + minDy * minDy >= f2_sq) continue;

    const cx = ipx + i, cy = ipy + j;
    const h = hash3(cx, cy, seed);
    const px = (h & 0xfff) / 4095.0;
    const py = ((h >>> 12) & 0xfff) / 4095.0;
    const dx = i + px - fx;
    const dy = j + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) {
      f2_sq = f1_sq; f1_sq = d2;
      bcx = cx + px; bcy = cy + py;
      btype = (h >>> 24) / 255.0;
    } else if (d2 < f2_sq) {
      f2_sq = d2;
    }
  }
  const f1 = Math.sqrt(f1_sq);
  const f2 = Math.sqrt(f2_sq);
  return { f1, f2, edge: f2 - f1, centerX: bcx, centerY: bcy, cellHash: btype };
}

const INV_4095 = 1.0 / 4095.0;
const INV_255 = 1.0 / 255.0;
const R_OUT_CELL_SQ = (ANOMALY.rOut / ANOMALY.cell) * (ANOMALY.rOut / ANOMALY.cell);

/**
 * §7 — Worley edge distance D_edge = F2 − F1 (river network). Tracks only the
 * two nearest squared distances (no centre or type byte) and skips neighbour
 * cells whose border is already farther than F2.
 */
export function worleyEdge(x: number, y: number, seed: number): number {
  const ipx = Math.floor(x) | 0, ipy = Math.floor(y) | 0;
  const fx = x - ipx, fy = y - ipy;
  const sTerm = Math.imul(seed | 0, 0x9e3779b9);

  // Cell (0, 0)
  const cy0Term = Math.imul(ipy, 0x27d4eb2f);
  let h0 = (ipx ^ cy0Term ^ sTerm) | 0;
  h0 ^= h0 >>> 16; h0 = Math.imul(h0, 0x85ebca6b); h0 ^= h0 >>> 13; h0 = Math.imul(h0, 0xc2b2ae35); h0 ^= h0 >>> 16;
  const dx0 = (h0 & 0xfff) * INV_4095 - fx;
  const dy0 = ((h0 >>> 12) & 0xfff) * INV_4095 - fy;
  let f1_sq = dx0 * dx0 + dy0 * dy0;
  let f2_sq = 64.0;

  const fx0_2 = fx * fx;
  const fx1 = 1.0 - fx;
  const fx1_2 = fx1 * fx1;
  const fy0_2 = fy * fy;
  const fy1 = 1.0 - fy;
  const fy1_2 = fy1 * fy1;

  const ipx_p1 = (ipx + 1) | 0;
  const ipx_m1 = (ipx - 1) | 0;
  const ipy_p1 = (ipy + 1) | 0;
  const ipy_m1 = (ipy - 1) | 0;
  const cyp1Term = Math.imul(ipy_p1, 0x27d4eb2f);
  const cym1Term = Math.imul(ipy_m1, 0x27d4eb2f);

  // 4 Orthogonal neighbors
  if (fx1_2 < f2_sq) {
    let hn = (ipx_p1 ^ cy0Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = 1.0 + (hn & 0xfff) * INV_4095 - fx;
    const dy = ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  if (fy1_2 < f2_sq) {
    let hn = (ipx ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = (hn & 0xfff) * INV_4095 - fx;
    const dy = 1.0 + ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  if (fx0_2 < f2_sq) {
    let hn = (ipx_m1 ^ cy0Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = -1.0 + (hn & 0xfff) * INV_4095 - fx;
    const dy = ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  if (fy0_2 < f2_sq) {
    let hn = (ipx ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = (hn & 0xfff) * INV_4095 - fx;
    const dy = -1.0 + ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }

  // 4 Diagonal corners
  if (fx1_2 + fy1_2 < f2_sq) {
    let hn = (ipx_p1 ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = 1.0 + (hn & 0xfff) * INV_4095 - fx;
    const dy = 1.0 + ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  if (fx0_2 + fy1_2 < f2_sq) {
    let hn = (ipx_m1 ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = -1.0 + (hn & 0xfff) * INV_4095 - fx;
    const dy = 1.0 + ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  if (fx1_2 + fy0_2 < f2_sq) {
    let hn = (ipx_p1 ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = 1.0 + (hn & 0xfff) * INV_4095 - fx;
    const dy = -1.0 + ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  if (fx0_2 + fy0_2 < f2_sq) {
    let hn = (ipx_m1 ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const dx = -1.0 + (hn & 0xfff) * INV_4095 - fx;
    const dy = -1.0 + ((hn >>> 12) & 0xfff) * INV_4095 - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; } else if (d2 < f2_sq) { f2_sq = d2; }
  }

  return Math.sqrt(f2_sq) - Math.sqrt(f1_sq);
}

/* ------------------------------------------------------------------ */
/* §14 — Domain warping                                                */
/* ------------------------------------------------------------------ */

/** §14 — Domain warping for organic geological folding and continuous coordinate displacement. */
export function domainWarp(
  x: number, y: number, seed: number, dp: number,
): [number, number] {
  const wx = x * F.warp, wy = y * F.warp;
  const qx = fbm(wx, wy, seed + 11, WARP.octaves, 2.0, 0.5, dp, F.warp);
  const qy = fbm(wx + 5.2, wy + 1.3, seed + 37, WARP.octaves, 2.0, 0.5, dp, F.warp);
  return [x + WARP.strength * qx, y + WARP.strength * qy];
}

/* ------------------------------------------------------------------ */
/* §12 — Geological anomalies (Zonai-style terraces, spire, spiral)    */
/* ------------------------------------------------------------------ */

/** §12 — Geological anomalies (Zonai-style terraces, spire, spiral); neighbour cells pruned to R_out. */
export function anomalyE(x: number, y: number, seed: number): number {
  const ax = x * (1.0 / ANOMALY.cell), ay = y * (1.0 / ANOMALY.cell);
  const ipx = Math.floor(ax) | 0, ipy = Math.floor(ay) | 0;
  const fx = ax - ipx, fy = ay - ipy;
  const sTerm = Math.imul((seed + 400) | 0, 0x9e3779b9);

  // Cell (0, 0)
  const cy0Term = Math.imul(ipy, 0x27d4eb2f);
  let h0 = (ipx ^ cy0Term ^ sTerm) | 0;
  h0 ^= h0 >>> 16; h0 = Math.imul(h0, 0x85ebca6b); h0 ^= h0 >>> 13; h0 = Math.imul(h0, 0xc2b2ae35); h0 ^= h0 >>> 16;
  const px0 = (h0 & 0xfff) * INV_4095;
  const py0 = ((h0 >>> 12) & 0xfff) * INV_4095;
  const dx0 = px0 - fx, dy0 = py0 - fy;
  let f1_sq = dx0 * dx0 + dy0 * dy0;
  let bcx = ipx + px0, bcy = ipy + py0;
  let btype = (h0 >>> 24) * INV_255;

  // A neighbour cell can only change the result if its border lies within
  // ANOMALY.rOut (beyond it the anomaly is 0 whichever feature owns the point)
  // and it could hold a nearer feature than the best so far. Pruning against
  // min(F1², R_out²) visits ~1.5 of 8 neighbours on average and subsumes the
  // former cell-interior fast path. Mirrored in demoShader anomalyE.
  const fx0_2 = fx * fx;
  const fx1 = 1.0 - fx;
  const fx1_2 = fx1 * fx1;
  const fy0_2 = fy * fy;
  const fy1 = 1.0 - fy;
  const fy1_2 = fy1 * fy1;

  const ipx_p1 = (ipx + 1) | 0;
  const ipx_m1 = (ipx - 1) | 0;
  const ipy_p1 = (ipy + 1) | 0;
  const ipy_m1 = (ipy - 1) | 0;
  const cyp1Term = Math.imul(ipy_p1, 0x27d4eb2f);
  const cym1Term = Math.imul(ipy_m1, 0x27d4eb2f);

  if (fx1_2 < f1_sq && fx1_2 < R_OUT_CELL_SQ) {
    let hn = (ipx_p1 ^ cy0Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = 1.0 + px - fx, dy = py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx_p1 + px; bcy = ipy + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fy1_2 < f1_sq && fy1_2 < R_OUT_CELL_SQ) {
    let hn = (ipx ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = px - fx, dy = 1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx + px; bcy = ipy_p1 + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fx0_2 < f1_sq && fx0_2 < R_OUT_CELL_SQ) {
    let hn = (ipx_m1 ^ cy0Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = -1.0 + px - fx, dy = py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx_m1 + px; bcy = ipy + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fy0_2 < f1_sq && fy0_2 < R_OUT_CELL_SQ) {
    let hn = (ipx ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = px - fx, dy = -1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx + px; bcy = ipy_m1 + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fx1_2 + fy1_2 < f1_sq && fx1_2 + fy1_2 < R_OUT_CELL_SQ) {
    let hn = (ipx_p1 ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = 1.0 + px - fx, dy = 1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx_p1 + px; bcy = ipy_p1 + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fx0_2 + fy1_2 < f1_sq && fx0_2 + fy1_2 < R_OUT_CELL_SQ) {
    let hn = (ipx_m1 ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = -1.0 + px - fx, dy = 1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx_m1 + px; bcy = ipy_p1 + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fx1_2 + fy0_2 < f1_sq && fx1_2 + fy0_2 < R_OUT_CELL_SQ) {
    let hn = (ipx_p1 ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = 1.0 + px - fx, dy = -1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx_p1 + px; bcy = ipy_m1 + py; btype = (hn >>> 24) * INV_255; }
  }
  if (fx0_2 + fy0_2 < f1_sq && fx0_2 + fy0_2 < R_OUT_CELL_SQ) {
    let hn = (ipx_m1 ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = -1.0 + px - fx, dy = -1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f1_sq = d2; bcx = ipx_m1 + px; bcy = ipy_m1 + py; btype = (hn >>> 24) * INV_255; }
  }

  if (btype >= ANOMALY.probability) return 0.0;
  const cx = bcx * ANOMALY.cell;
  const cy = bcy * ANOMALY.cell;
  const dx = x - cx, dy = y - cy;
  const r2 = dx * dx + dy * dy;
  if (r2 >= ANOMALY.rOut * ANOMALY.rOut) return 0.0;
  const r = Math.sqrt(r2);
  const theta = Math.atan2(dy, dx);
  const fall = 1.0 - smoothstep(ANOMALY.rIn, ANOMALY.rOut, r);
  const t = r / ANOMALY.terraceStep;
  const ft = t - Math.floor(t);
  const steps = (Math.floor(t) + smoothstep(0.3, 0.7, ft)) * ANOMALY.terraceHeight;
  const spire = ANOMALY.spireHeight * Math.exp(-r2 / (ANOMALY.spireRadius * ANOMALY.spireRadius));
  const spiral = ANOMALY.spiralAmp * Math.sin(ANOMALY.spiralTurns * theta + ANOMALY.spiralRadial * r);
  return fall * (steps + spire + spiral);
}

/* ------------------------------------------------------------------ */
/* §13 — Regional mathematical rules                                   */
/* ------------------------------------------------------------------ */

export type RegionInfo = {
  wRegion: number;      // 0 at cell borders → 1 in cell interior
  mountFactor: number;  // H_MOUNT multiplier for this region
  duneAmount: number;   // 0..1 crescent-dune blend
};

// Module-level scratch state for zero-allocation region field inside elevation
let _wRegion = 0, _mountFactor = 0, _duneAmount = 0;
function evalRegionInternal(x: number, y: number, seed: number): void {
  const rx = x * (1.0 / REGION.cell), ry = y * (1.0 / REGION.cell);
  const ipx = Math.floor(rx), ipy = Math.floor(ry);
  const fx = rx - ipx, fy = ry - ipy;
  let f1_sq = 64.0, f2_sq = 64.0;
  let btype = 0;
  const sTerm = Math.imul((seed + 900) | 0, 0x9e3779b9);

  // Center (0, 0)
  const cyTerm = Math.imul(ipy | 0, 0x27d4eb2f);
  let h0 = (ipx ^ cyTerm ^ sTerm) | 0;
  h0 ^= h0 >>> 16; h0 = Math.imul(h0, 0x85ebca6b); h0 ^= h0 >>> 13; h0 = Math.imul(h0, 0xc2b2ae35); h0 ^= h0 >>> 16;
  const px0 = (h0 & 0xfff) * INV_4095;
  const py0 = ((h0 >>> 12) & 0xfff) * INV_4095;
  const dx0 = px0 - fx, dy0 = py0 - fy;
  f1_sq = dx0 * dx0 + dy0 * dy0;
  btype = (h0 >>> 24) * INV_255;

  const fx0_2 = fx * fx;
  const fx1_2 = (1.0 - fx) * (1.0 - fx);
  const fy0_2 = fy * fy;
  const fy1_2 = (1.0 - fy) * (1.0 - fy);

  const ipx_p1 = (ipx + 1) | 0;
  const ipx_m1 = (ipx - 1) | 0;
  const ipy_p1 = (ipy + 1) | 0;
  const ipy_m1 = (ipy - 1) | 0;
  const cyp1Term = Math.imul(ipy_p1, 0x27d4eb2f);
  const cym1Term = Math.imul(ipy_m1, 0x27d4eb2f);

  // (+1, 0)
  if (fx1_2 < f2_sq) {
    let hn = (ipx_p1 ^ cyTerm ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = 1.0 + px - fx, dy = py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (0, +1)
  if (fy1_2 < f2_sq) {
    let hn = (ipx ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = px - fx, dy = 1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (-1, 0)
  if (fx0_2 < f2_sq) {
    let hn = (ipx_m1 ^ cyTerm ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = -1.0 + px - fx, dy = py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (0, -1)
  if (fy0_2 < f2_sq) {
    let hn = (ipx ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = px - fx, dy = -1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (+1, +1)
  if (fx1_2 + fy1_2 < f2_sq) {
    let hn = (ipx_p1 ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = 1.0 + px - fx, dy = 1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (-1, +1)
  if (fx0_2 + fy1_2 < f2_sq) {
    let hn = (ipx_m1 ^ cyp1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = -1.0 + px - fx, dy = 1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (+1, -1)
  if (fx1_2 + fy0_2 < f2_sq) {
    let hn = (ipx_p1 ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = 1.0 + px - fx, dy = -1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }
  // (-1, -1)
  if (fx0_2 + fy0_2 < f2_sq) {
    let hn = (ipx_m1 ^ cym1Term ^ sTerm) | 0;
    hn ^= hn >>> 16; hn = Math.imul(hn, 0x85ebca6b); hn ^= hn >>> 13; hn = Math.imul(hn, 0xc2b2ae35); hn ^= hn >>> 16;
    const px = (hn & 0xfff) * INV_4095;
    const py = ((hn >>> 12) & 0xfff) * INV_4095;
    const dx = -1.0 + px - fx, dy = -1.0 + py - fy;
    const d2 = dx * dx + dy * dy;
    if (d2 < f1_sq) { f2_sq = f1_sq; f1_sq = d2; btype = (hn >>> 24) * INV_255; } else if (d2 < f2_sq) { f2_sq = d2; }
  }

  const edge = Math.sqrt(f2_sq) - Math.sqrt(f1_sq);
  _wRegion = smoothstep(REGION.blendLo, REGION.blendHi, edge);
  _mountFactor = btype < 0.33 ? REGION.hillsFactor
    : btype < 0.66 ? REGION.mountainFactor
    : REGION.duneFactor;
  _duneAmount = smoothstep(0.66, 0.8, btype) * _wRegion;
}

/** §13 — Regional biome field determining mountain/hills/dune factors and blending. */
export function regionField(x: number, y: number, seed: number): RegionInfo {
  evalRegionInternal(x, y, seed);
  return { wRegion: _wRegion, mountFactor: _mountFactor, duneAmount: _duneAmount };
}

/** §11/§13 — Crescent (barchan) dunes — trigonometric region rule f₁ (spec §13). */
export function dunesE(x: number, y: number, seed: number, dp: number, amount: number): number {
  if (amount <= 0.0) return 0.0;
  const phase = x * F.dune * (2.0 * Math.PI) +
    3.0 * fbm(x * F.warp, y * F.warp, seed + 301, 2, 2.0, 0.5, dp, F.warp);
  const c = 1.0 - Math.abs(Math.sin(phase));
  const shaped = c * c * c;
  return REGION.duneHeight * amount * shaped;
}

/* ------------------------------------------------------------------ */
/* Master elevation field — spec §2 environment composition            */
/* ------------------------------------------------------------------ */

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** §4/§31 — Scalar clamp helper bounded between lo and hi. */
export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/**
 * §2/§6/§14 — E(x, z) — terrain elevation in metres. `dp` is the Nyquist pixel footprint
 * (spec §5): pass ~0 for full-detail CPU evaluation, or the screen footprint
 * to mirror the GPU's filtered result. All lookups are chunk-local (spec §25).
 * No object or array allocation (see PARITY NOTES for V8 number boxing).
 */
export function elevation(x: number, z: number, seed: number, dp: number): number {
  // §14 — warp the domain (fluid geological folding, zero-allocation inline)
  let wqx = x, wqy = z;
  if (dp <= 0.0) {
    const wx = x * F.warp, wy = z * F.warp;
    const qx = vnoise(wx, wy, seed + 11) + 0.5 * vnoise(wx * 2.0, wy * 2.0, seed + 40514);
    const qy = vnoise(wx + 5.2, wy + 1.3, seed + 37) + 0.5 * vnoise((wx + 5.2) * 2.0, (wy + 1.3) * 2.0, seed + 40540);
    wqx += WARP.strength * qx;
    wqy += WARP.strength * qy;
  } else {
    const wx = x * F.warp, wy = z * F.warp;
    const qx = fbm(wx, wy, seed + 11, WARP.octaves, 2.0, 0.5, dp, F.warp);
    const qy = fbm(wx + 5.2, wy + 1.3, seed + 37, WARP.octaves, 2.0, 0.5, dp, F.warp);
    wqx += WARP.strength * qx;
    wqy += WARP.strength * qy;
  }

  // §13 — regional rules on the raw (un-warped) plane: zero-allocation scratch
  evalRegionInternal(x, z, seed);
  const wRegion = _wRegion, mountFactor = _mountFactor, duneAmount = _duneAmount;

  // §9 support — continent field (base amplitude + ocean-moisture proxy)
  const cont = smoothstep(
    CONTINENT.lo, CONTINENT.hi,
    0.5 + 0.5 * vnoise(x * F.cont, z * F.cont, seed + 801),
  );

  // §5 — macro fBm hills
  const base = MACRO.hBase * (0.5 + 0.5 * fbm(wqx * F.macro, wqy * F.macro, seed + 1,
    MACRO.octaves, MACRO.lacunarity, MACRO.gain, dp, F.macro));

  // §6 — ridged mountains gated by a macro mask
  const maskVal = 0.5 + 0.5 * vnoise(wqx * F.mask, wqy * F.mask, seed + 77);
  const mask = smoothstep(MOUNTAIN.maskLo, MOUNTAIN.maskHi, maskVal);
  const hMount = MOUNTAIN.hMount * (1.0 + (mountFactor - 1.0) * wRegion);
  
  let ridge = 0.0;
  if (mask > 0.0) {
    ridge = ridged(wqx * F.ridge, wqy * F.ridge, seed + 5,
      MOUNTAIN.octaves, MOUNTAIN.lacunarity, MOUNTAIN.gain, dp, F.ridge);
  }

  // §7 — cellular valleys carve the network of rivers (fast worleyEdge)
  const de = worleyEdge(wqx * F.valley, wqy * F.valley, seed + 21);
  const sv = 1.0 - smoothstep(0, VALLEY.wValley, de);
  const valley = VALLEY.cDepth * (sv * sv);

  // §13 — dune rule + §12 — anomalies (raw plane, rigid Zonai geometry)
  const dune = duneAmount > 0.0 ? dunesE(x, z, seed, dp, duneAmount) : 0.0;
  const anom = anomalyE(x, z, seed);

  return base * (CONTINENT.minFactor + (1.0 - CONTINENT.minFactor) * cont)
    + hMount * mask * ridge
    + dune
    - valley
    + anom;
}

/** §5/§17 — Central-difference gradient at the pixel-footprint scale (spec §17). */
export function gradient(
  x: number, z: number, seed: number, eps: number, out?: [number, number],
): [number, number] {
  const e = Math.max(eps, 0.05);
  const inv2e = 0.5 / e;
  const gx = (elevation(x + e, z, seed, 0) - elevation(x - e, z, seed, 0)) * inv2e;
  const gz = (elevation(x, z + e, seed, 0) - elevation(x, z - e, seed, 0)) * inv2e;
  if (out) {
    out[0] = gx;
    out[1] = gz;
    return out;
  }
  return [gx, gz];
}

/** §5/§17/§31 — Slope magnitude ‖∇E‖ — the Lipschitz factor + slide gate input (zero-allocation). */
export function slope(x: number, z: number, seed: number): number {
  const gx = (elevation(x + 0.25, z, seed, 0) - elevation(x - 0.25, z, seed, 0)) * 2.0;
  const gz = (elevation(x, z + 0.25, seed, 0) - elevation(x, z - 0.25, seed, 0)) * 2.0;
  return Math.sqrt(gx * gx + gz * gz);
}

/* ------------------------------------------------------------------ */
/* §9 — Climate fields                                                 */
/* ------------------------------------------------------------------ */

/** §9/§13 — Temperature equation based on latitude lapse and elevation lapse. */
export function temperature(
  x: number, z: number, e: number, _seed = 0, globalY?: number,
): number {
  const lat = globalY !== undefined ? globalY : z;
  return CLIMATE.tBase
    - CLIMATE.lambdaLat * Math.abs(lat)
    - CLIMATE.lambdaAlt * Math.max(0, e);
}

/** §9/§13 — Moisture equation driven by moisture noise fBm and continent distance proxy. */
export function moisture(x: number, z: number, e: number, seed: number): number {
  const n = fbm(x * F.moist, z * F.moist, seed + 61, 2, 2.0, 0.5, 0, F.moist);
  const cont = smoothstep(CONTINENT.lo, CONTINENT.hi,
    0.5 + 0.5 * vnoise(x * F.cont, z * F.cont, seed + 801));
  // λ_ocean·exp(−D_ocean/σ) approximated by the continent field itself
  // (smooth, deterministic, identical CPU/GPU — see worklog).
  return CLIMATE.mBase + CLIMATE.mNoiseAmp * n + CLIMATE.mOceanLambda * (1 - cont);
}

/* ------------------------------------------------------------------ */
/* §16 — Lipschitz-corrected raymarch + bisection refinement           */
/* ------------------------------------------------------------------ */

type MarchRow = {
  i: number;
  t: number;
  px: number; py: number; pz: number;
  e: number;
  gap: number;        // P.y − E
  dp: number;         // pixel footprint Δp(t)
  grad: number;       // ‖∇E‖
  step: number;       // Δt applied
  wMacro: number[];   // Nyquist weights of macro octaves at this step
  wRidge: number[];   // Nyquist weights (2f) of ridged octaves
};

export type MarchResult = {
  hit: boolean;
  t: number;
  rows: MarchRow[];
  bisectRows: { i: number; t: number; gap: number }[];
};

function octaveWeights(
  f0: number, octaves: number, lac: number, dp: number, doubled: boolean,
): number[] {
  const out: number[] = [];
  let f = f0;
  for (let i = 0; i < octaves; i++) {
    out.push(nyqW(doubled ? 2 * f : f, dp));
    f *= lac;
  }
  return out;
}

/**
 * §8/§16/§30 — March one ray against E. Records every step so the worked example (§30)
 * can show the Nyquist weights collapsing and the Lipschitz denominator
 * shrinking the steps near cliffs.
 */
export function marchRay(
  ro: [number, number, number],
  rd: [number, number, number],
  seed: number,
  opts: {
    steps: number;
    footprint: number;     // 2·tan(FOV/2)/Res_y
    maxDist?: number;
    record?: boolean;
  },
): MarchResult {
  const maxDist = opts.maxDist ?? MAX_DIST;
  const rows: MarchRow[] = [];
  let t: number = MARCH.startT;
  let tPrev: number = t;
  let grad = 0;
  let gradSq = 0;

  for (let i = 0; i < opts.steps; i++) {
    const px = ro[0] + rd[0] * t;
    const py = ro[1] + rd[1] * t;
    const pz = ro[2] + rd[2] * t;

    if (py > MAX_WORLD_H && rd[1] > 0) return { hit: false, t, rows, bisectRows: [] }; // §26 early exit

    const dp = Math.max(t * opts.footprint, 1e-4);
    const e = elevation(px, pz, seed, dp);

    if (opts.record && rows.length < 40) {
      rows.push({
        i, t, px, py, pz, e, gap: py - e, dp, grad,
        step: t - tPrev,
        wMacro: octaveWeights(F.macro, MACRO.octaves, MACRO.lacunarity, dp, false),
        wRidge: octaveWeights(F.ridge, MOUNTAIN.octaves, MOUNTAIN.lacunarity, dp, true),
      });
    }

    if (py < e) {
      // crossed the surface — bisect over [tPrev, t] (spec §16)
      const bisectRows: { i: number; t: number; gap: number }[] = [];
      let lo: number = tPrev, hi: number = t;
      for (let b = 0; b < MARCH.bisectSteps; b++) {
        const tm = 0.5 * (lo + hi);
        const mx = ro[0] + rd[0] * tm;
        const my = ro[1] + rd[1] * tm;
        const mz = ro[2] + rd[2] * tm;
        const gap = my - elevation(mx, mz, seed, tm * opts.footprint);
        bisectRows.push({ i: b, t: tm, gap });
        if (gap < 0) hi = tm; else lo = tm;
        if (!opts.record && (hi - lo) < MARCH.bisectTol) break;
      }
      return { hit: true, t: 0.5 * (lo + hi), rows, bisectRows };
    }

    // Lipschitz factor ‖∇E‖ via forward differences at the pixel-footprint
    // scale, refreshed every MARCH.gradCadence-th sample and only within
    // MARCH.gradGate clamp-lengths of the surface (bisection catches overshoot;
    // the horizon clamp bounds the stale window — §16/§34.1). Mirrors the GPU.
    const clampT = t * MARCH.clampSlope + MARCH.clampBase; // horizon clamp (anti-tunnel)
    const farAbove = (py - e) > clampT * MARCH.gradGate;
    if ((i % MARCH.gradCadence) === 0) {
      if (!farAbove) {
        const eps = Math.max(dp, MARCH.gradEpsMin);
        const ex = elevation(px + eps, pz, seed, dp);
        const ez = elevation(px, pz + eps, seed, dp);
        const invEps = 1.0 / eps;
        const dx = (ex - e) * invEps;
        const dz = (ez - e) * invEps;
        gradSq = dx * dx + dz * dz;
        if (opts.record) grad = Math.sqrt(gradSq);
      } else {
        gradSq = 0.0;
        grad = 0.0;
      }
    }

    tPrev = t;
    let step = Math.max(
      MARCH.deltaMin,
      (py - e) / Math.sqrt(gradSq + 1.0) * MARCH.lambdaStep,
    );
    step *= 1.0 + t * MARCH.stepGrowth; // §26 adaptive growth
    // Far above the surface the gradient is not refreshed and the clamp alone
    // paces the march, so it may widen there without loosening the near cap.
    step = Math.min(step, farAbove ? t * MARCH.clampSlopeFar + MARCH.clampBase : clampT);
    t += step;

    if (t > maxDist) return { hit: false, t, rows, bisectRows: [] };
  }
  return { hit: false, t, rows, bisectRows: [] };
}

/** §18 — soft shadow: march toward the sun with a 4× inflated footprint. */
export function softShadow(
  p: [number, number, number],
  sunDir: [number, number, number],
  seed: number,
  steps: number,
  footprint: number,
): number {
  let res = 1.0;
  let t = SHADOW.startT;
  for (let i = 0; i < steps; i++) {
    const qx = p[0] + sunDir[0] * t;
    const qy = p[1] + sunDir[1] * t;
    const qz = p[2] + sunDir[2] * t;
    if (qy > MAX_WORLD_H) break;
    const h = qy - elevation(qx, qz, seed, SHADOW.footprintScale * t * footprint); // §18: 4×Δp
    if (h <= 0.0) return 0.0;
    res = Math.min(res, SHADOW.penumbraK * h / t); // penumbra kernel
    if (res < SHADOW.cutoff) return 0.0;
    t += clamp(h * SHADOW.stepScale, SHADOW.stepMin, SHADOW.stepMax) + t * SHADOW.stepGrowth;
  }
  return clamp(res, 0, 1);
}

/* ------------------------------------------------------------------ */
/* §31 — CPU physics twin movement loop (jump-preserving)              */
/* ------------------------------------------------------------------ */

export type PlayerState = {
  x: number; y: number; z: number;   // global, double precision
  vx: number; vy: number; vz: number;
  grounded: boolean;
};

export type InputState = {
  forward: number; strafe: number;   // −1..1
  jump: boolean;
  sprint: boolean;
  yaw: number;                        // camera yaw for wish direction
};

/**
 * §31 loop verbatim: gravity always integrates; the ground clamp only ever
 * zeroes NEGATIVE vertical velocity so jumps are never killed; the slope
 * gate slides the player off steep warped ridges. Evaluated on fp64 global
 * coordinates; PLAYER.groundBias absorbs the GPU's fp32 divergence.
 */
export function updatePlayer(
  s: PlayerState, input: InputState, dt: number, seed: number,
): void {
  const speed = PLAYER.walkSpeed * (input.sprint ? PLAYER.sprintMul : 1.0);
  let wishX = 0, wishZ = 0;
  if (input.forward !== 0 || input.strafe !== 0) {
    const sin = Math.sin(input.yaw), cos = Math.cos(input.yaw);
    // forward = (sin yaw, 0, cos yaw) — matches demo camera basis
    wishX = (sin * input.forward + cos * input.strafe) * speed;
    wishZ = (cos * input.forward - sin * input.strafe) * speed;
  }

  // horizontal velocity: exponential approach (feels like critically damped)
  const k = 1 - Math.exp(-12 * dt);
  s.vx += (wishX - s.vx) * k;
  s.vz += (wishZ - s.vz) * k;

  if (input.jump && s.grounded) s.vy = PLAYER.jumpVelocity;

  s.vy -= PLAYER.gravity * dt;              // GRAVITY
  s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt;

  // Global-frame evaluation (spec §25/§27): the CPU twin computes in double
  // precision on GLOBAL coordinates; the GPU evaluates fp32(local + origin).
  // The measured difference (constants.ts, PLAYER.groundBias) is absorbed by
  // GROUND_BIAS below.
  const groundZ = elevation(s.x, s.z, seed, 0) + PLAYER.groundBias + PLAYER.height;

  s.grounded = false;
  if (s.y < groundZ) {
    s.y = groundZ;
    if (s.vy < 0.0) s.vy = 0.0;            // stop FALLING only — preserve jumps
    s.grounded = true;
  }

  // Slope gate: slide off steep warped ridges (spec §31 insert, zero-allocation)
  if (s.grounded) {
    const gx = (elevation(s.x + 0.25, s.z, seed, 0) - elevation(s.x - 0.25, s.z, seed, 0)) * 2.0;
    const gz = (elevation(s.x, s.z + 0.25, seed, 0) - elevation(s.x, s.z - 0.25, seed, 0)) * 2.0;
    const mSq = gx * gx + gz * gz;
    const slopeMaxSq = PLAYER.slopeMax * PLAYER.slopeMax;
    if (mSq > slopeMaxSq) {
      const m = Math.sqrt(mSq);
      const inv = 1 / m;
      const factor = inv * (m - PLAYER.slopeMax) * PLAYER.slideAccel * dt;
      s.vx -= gx * factor;
      s.vz -= gz * factor;
    }
  }
}
