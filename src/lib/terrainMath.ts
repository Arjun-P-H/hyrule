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
  F, MACRO, MOUNTAIN, WARP, LANDS,
  CLIMATE, MARCH, SHADOW, MAX_WORLD_H, MAX_DIST, PLAYER,
} from "./constants";
import { BI, BIOMES, BIOME_COUNT, LANDFORM, LANDFORM_KEYS, LOOK } from "./biomes";
import { CRATER_RIM, RF, UP, VOL, continentAt, riverField, upliftAt } from "./hydro";
import { HYDRO } from "./constants";
import { erosionDelta } from "./erosion";
import { n01, smoothstep } from "./noise";

export { hashU, hash3, hash01, cellPoint, cellType, quintic, vnoise } from './noise';
import { hash3, vnoise } from './noise';

/**
 * Nyquist octave weight (spec §5): fade out between 0.5 and 1 cycle/pixel.
 * Δp(t) = t · 2tan(FOV/2) / Res_y is the world-space pixel footprint.
 */
function nyqW(freq: number, dp: number): number {
  return Math.min(1, Math.max(0, 2.0 - (freq * dp) * 2.0));
}

// Precomputed totalAsum constants: sum_{i=0}^{octaves-1} (initial_amp * gain^i)
const ASUM_MACRO = 1.99609375; // 9 octaves, gain 0.5, amp 1.0
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
/* §13 — The lands: biome regions (biomes.ts)                          */
/* ------------------------------------------------------------------ */


/** §4/§31 — Scalar clamp helper bounded between lo and hi. */
export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

const NB = BIOME_COUNT;
const NK = LANDFORM_KEYS.length;
/** biomes.ts LANDFORM flattened: LF_TABLE[biome · NK + key] (keys in LANDFORM_KEYS order). */
const LF_TABLE = (() => {
  const t = new Float64Array(NB * NK);
  BIOMES.forEach((b, i) => LANDFORM_KEYS.forEach((k, j) => { t[i * NK + j] = LANDFORM[b][k]; }));
  return t;
})();
const K_HILLS = LANDFORM_KEYS.indexOf('hills'), K_RIDGE = LANDFORM_KEYS.indexOf('ridge');
const K_KARST = LANDFORM_KEYS.indexOf('karst'), K_MESA = LANDFORM_KEYS.indexOf('mesa'), K_DUNE = LANDFORM_KEYS.indexOf('dune');
const K_FLAT = LANDFORM_KEYS.indexOf('flat'), K_K = LANDFORM_KEYS.indexOf('k');
const K_ERO = LANDFORM_KEYS.indexOf('erosion'), K_POND = LANDFORM_KEYS.indexOf('pond');

/**
 * What filtered-out landforms fade to (the WGSL twin reads these): E[(1 − |n|)²] for the value noise
 * (a ridged octave), the karst towers' and the dunes' mean heights (m, per unit of their weight).
 */
export const LAND_MEANS = { ridge: 0.4466, tower: 21.0, dune: 9.0 } as const;
const RIDGE_MEAN = LAND_MEANS.ridge;
const TOWER_MEAN = LAND_MEANS.tower;
const DUNE_MEAN = LAND_MEANS.dune;
/** Where the land is dry its water table runs this far (m) under the ground (clearly dry to every test). */
export const DRY_TABLE = 3.0;
/** Soft ceiling: the land eases toward MAX_WORLD_H − 10 m above this. */
export const CEIL_START = 1150.0;
export const CEIL_SPAN = MAX_WORLD_H - 10.0 - CEIL_START;

// Module scratch (zero allocation): the region weights, the blended landform, the climate and the
// rare features at the last point landRegions / landSurface evaluated.
const _w = new Float64Array(NB);
const _lf = new Float64Array(NK);
let _T0 = 0, _M = 0, _C = 0, _U = 0;
let _vW = 0, _sW = 0;
let _wx = 0, _wz = 0;
let _e = 0, _wt = 0, _land = 1, _riverK = 0, _lakeK = 0, _fx = 0, _fz = 0, _speed = 0, _lava = 0, _shore = -Infinity;

/** §14 — the warped plane at (x, z) into _wx, _wz (the fBm warp, dp-free). */
function warpInto(x: number, z: number, seed: number): void {
  const wx = x * F.warp, wy = z * F.warp;
  const qx = vnoise(wx, wy, seed + 11) + 0.5 * vnoise(wx * 2.0, wy * 2.0, seed + 40514);
  const qy = vnoise(wx + 5.2, wy + 1.3, seed + 37) + 0.5 * vnoise((wx + 5.2) * 2.0, (wy + 1.3) * 2.0, seed + 40540);
  _wx = x + WARP.strength * qx;
  _wz = z + WARP.strength * qy;
}

/**
 * §13 — The biome regions at raw (x, z) / warped (wx, wz), after upliftAt there (UP, VOL): weights
 * into w (biomes.ts order, summing to 1), the climate into _T0 (°C at sea level), _M, _C (the
 * continent). The geography decides the high country (the ranges and massifs → mountains, the
 * tablelands → highland); the lowland follows the climate (cold → tundra, cool → taiga or lakes,
 * mild → plains, badlands, forest, wetland or lakes, hot → desert, badlands, savanna, rainforest,
 * jungle or wetland); volcanoes and sacred groves are laid over whatever was there.
 */
function regionsInto(x: number, z: number, wx: number, wz: number, seed: number, w: Float64Array): void {
  const T0 = CLIMATE.tBase + CLIMATE.tAmp * (vnoise(wx * F.temp, wz * F.temp, seed + 1201) + CLIMATE.tFine * vnoise(wx * F.tempfine, wz * F.tempfine, seed + 1202))
    - CLIMATE.lambdaLat * Math.abs(z);
  const C = UP.C;
  const S = LANDS.coast.level;
  const M = CLIMATE.mBase + CLIMATE.mNoiseAmp * (vnoise(wx * F.moistr, wz * F.moistr, seed + 1301) + CLIMATE.mFine * vnoise(wx * F.moistfine, wz * F.moistfine, seed + 1302))
    + CLIMATE.mOceanLambda * (1 - smoothstep(S, S + 0.12, C));
  _T0 = T0; _M = M; _C = C; _U = UP.range;

  const mtn = smoothstep(0.26, 0.5, Math.max(UP.range, 0.9 * UP.massif));
  const plat = smoothstep(0.35, 0.75, UP.plat) * (1 - mtn);
  const low = 1 - mtn - plat;
  const a = smoothstep(LANDS.cold[0], LANDS.cold[1], T0), b = smoothstep(LANDS.cool[0], LANDS.cool[1], T0), c = smoothstep(LANDS.hot[0], LANDS.hot[1], T0);
  const cold = 1 - a, cool = a * (1 - b), mild = a * b * (1 - c), hot = a * b * c;
  const dry = 1 - smoothstep(LANDS.dry[0], LANDS.dry[1], M);
  const hdry = 1 - smoothstep(LANDS.hotDry[0], LANDS.hotDry[1], M);
  const arid = 1 - smoothstep(LANDS.arid[0], LANDS.arid[1], M);
  const wet = smoothstep(LANDS.wet[0], LANDS.wet[1], M);
  const bad = smoothstep(LANDS.rugged[0], LANDS.rugged[1], n01(wx, wz, F.rugged, seed + 1501));
  const lake = smoothstep(LANDS.lakedist[0], LANDS.lakedist[1], n01(wx, wz, F.lakedist, seed + 1601));
  const karst = smoothstep(LANDS.karst[0], LANDS.karst[1], n01(wx, wz, F.karst, seed + 1701));
  const marsh = wet * smoothstep(LANDS.marsh[0], LANDS.marsh[1], n01(wx, wz, F.marsh, seed + 1801));
  w.fill(0);
  // the lowland
  w[BI.tundra] += low * cold;
  w[BI.taiga] += low * cool * (1 - lake);
  w[BI.lakes] += low * (cool + mild * (1 - dry)) * lake;
  const mildL = low * mild * (1 - lake * (1 - dry));
  const badWet = smoothstep(LANDS.badlandsWet[0], LANDS.badlandsWet[1], M);
  w[BI.plains] += mildL * dry * (1 - bad * (1 - badWet));
  w[BI.badlands] += mildL * dry * bad * (1 - badWet);
  w[BI.forest] += mildL * (1 - dry) * (1 - marsh);
  w[BI.wetland] += mildL * (1 - dry) * marsh;
  const hotL = low * hot;
  w[BI.desert] += hotL * arid * (1 - bad);
  w[BI.badlands] += hotL * (hdry * bad + arid * bad * (1 - hdry));
  w[BI.savanna] += hotL * hdry * (1 - arid) * (1 - bad);
  const hotM = hotL * (1 - hdry);
  w[BI.rainforest] += hotM * (1 - karst) * (1 - marsh);
  w[BI.jungle] += hotM * karst * (1 - marsh);
  w[BI.wetland] += hotM * marsh;
  // the tablelands: alpine highland, mesa country in the hot dry, karst highland in the hot wet
  w[BI.highland] += plat * (cold + cool + mild);
  w[BI.badlands] += plat * hot * hdry;
  w[BI.jungle] += plat * hot * (1 - hdry);
  // the ranges and massifs
  w[BI.mountains] += mtn;

  // A volcano (upliftAt found it): its region over whatever was there.
  const v = VOL.w;
  // A sacred grove: one candidate per cell of a lattice offset by half a cell, in mild moist land. The
  // lattice lies in the warped plane, like the volcanoes': a grove's whole disc, and so its fade, lies
  // in its own cell (a cut in the world plane would end a grove in a wall where the warp pulls it in).
  const G = LANDS.sacred;
  const sx = Math.floor(wx / G.cell + 0.5), sz = Math.floor(wz / G.cell + 0.5);
  const sh = hash3(sx, sz, seed + 2201);
  let sc = 0;
  if ((sh >>> 24) * INV_255 < G.p) {
    const cx = (sx - 0.5 + 0.3 + 0.4 * (sh & 0xfff) * INV_4095) * G.cell, cz = (sz - 0.5 + 0.3 + 0.4 * ((sh >>> 12) & 0xfff) * INV_4095) * G.cell;
    const R = G.radius[0] + (G.radius[1] - G.radius[0]) * (hash3(sx, sz, seed + 2202) & 0xffff) / 65535;
    const dx = wx - cx, dz = wz - cz, r2 = dx * dx + dz * dz;
    if (r2 < R * R) {
      sc = (1 - smoothstep(0.6 * R, R, Math.sqrt(r2))) * smoothstep(2, 5, T0) * smoothstep(0.3, 0.4, M) * (1 - mtn);
    }
  }
  if (v > 0) {
    for (let k = 0; k < NB; k++) w[k] *= 1 - v;
    w[BI.volcanic] += v;
  }
  if (sc > 0) {
    const s = sc * (1 - v);
    for (let k = 0; k < NB; k++) w[k] *= 1 - s;
    w[BI.sacred] += s;
  }
  _vW = v; _sW = sc;
}

/** The blend of the regions' landforms into _lf, from the weights in _w. */
function blendLandform(): void {
  _lf.fill(0);
  for (let i = 0; i < NB; i++) {
    const wi = _w[i];
    if (wi <= 0) continue;
    const o = i * NK;
    for (let k = 0; k < NK; k++) _lf[k] += wi * LF_TABLE[o + k];
  }
}

/**
 * §13 — the biome region weights at world (x, z) (biomes.ts order, summing to 1) into `out`, and
 * the climate there (sea-level temperature, moisture). For placement, the HUD, tests.
 */
export function landRegions(x: number, z: number, seed: number, out: Float64Array = new Float64Array(NB)): Float64Array {
  warpInto(x, z, seed);
  upliftAt(_wx, _wz, seed);
  regionsInto(x, z, _wx, _wz, seed, out);
  return out;
}

/** The climate at the last landRegions / elevation call: sea-level temperature (°C), moisture, continent, range (0–1). */
export function lastClimate(): { T0: number; M: number; C: number; U: number } {
  return { T0: _T0, M: _M, C: _C, U: _U };
}

/**
 * The region weights at the last elevation / surfaceAt / waterLevel call (biomes.ts order; the
 * module's own scratch array, overwritten by the next call: copy it to keep it).
 */
export function lastRegions(): Float64Array {
  return _w;
}
/** biomes.ts LOOK.frost per biome (°C the snow line is lowered by). */
const FROST = BIOMES.map((b) => LOOK[b].frost);
/**
 * The snow line (m) at the last elevation / surfaceAt call: where the air is CLIMATE.snowT, lowered by
 * the regions' frost (the WGSL twin bakes the same into the ground's mat layer).
 */
function lastSnowLine(): number {
  let frost = 0;
  for (let k = 0; k < NB; k++) frost += _w[k] * FROST[k];
  return (_T0 - CLIMATE.snowT - frost) / CLIMATE.lambdaAlt;
}
/** Is the water at the last elevation / surfaceAt call frozen (ice: walkable, drawn as ice)? */
export function lastIce(): boolean {
  return _lava <= 0 && _wt > lastSnowLine() + CLIMATE.iceAbove;
}
/** Is the liquid at the last surfaceAt call lava (a volcano's crater lake or its lava rivers; 0–1)? */
export function lastLava(): number {
  return _lava;
}
/** The level of the water beside the last surfaceAt point (a river's bank, a lake, a pond, the coast, a marsh; −Infinity: none near). */
export function lastShore(): number {
  return _shore;
}
/** The flow at the last surfaceAt call: direction (unit, xz) and speed (m/s); zero on still water and land. */
export function lastFlow(): [number, number, number] {
  return [_fx, _fz, _speed];
}
/** The sea-level temperature (°C) and moisture at the last elevation / surfaceAt / landRegions call. */
export function lastSeaTemperature(): number { return _T0; }
export function lastMoisture(): number { return _M; }

/* ---- §6/§11 landforms ------------------------------------------------- */
/** Karst towers: the highest of the steep-sided cones of the 3 × 3 cells of a 300 m lattice round the point (m). */
function karstTowers(x: number, z: number, seed: number): number {
  const f = F.tower;
  const px = x * f, pz = z * f;
  const ipx = Math.floor(px), ipz = Math.floor(pz);
  let best = 0;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const h = hash3(ipx + i, ipz + j, seed + 2301);
      const cx = ipx + i + 0.2 + 0.6 * (h & 0xfff) * INV_4095, cz = ipz + j + 0.2 + 0.6 * ((h >>> 12) & 0xfff) * INV_4095;
      const dx = px - cx, dz = pz - cz;
      const r = (75 + 70 * ((h >>> 24) & 15) / 15) * f; // in cells
      const d2 = dx * dx + dz * dz;
      if (d2 >= r * r) continue;
      const q = 1 - Math.sqrt(d2) / r;
      const H = 50 + 170 * ((h >>> 28) & 15) / 15;
      const q1 = 1 - q, q2 = q1 * q1;
      const t = H * (1 - q2 * q1 * Math.sqrt(q1)); // 1 − (1 − q)^3.5: sheer sides, a rounded top
      if (t > best) best = t;
    }
  }
  return best;
}

/**
 * §11 — Dunes: transverse ridges across a wind that turns over the field, their crests bent and broken
 * into crescents by the height field, each a long windward rise to a sharp brink and a steep slip face;
 * cross ripples over them (m, before the region's weight).
 */
export function dunesE(x: number, z: number, seed: number, dp: number): number {
  const wMain = clamp(2 - 2 * dp * (2 * F.dune), 0, 1);
  const wCross = clamp(2 - 2 * dp * (2 * F.dunecross), 0, 1);
  const amp = 0.3 + 0.7 * n01(x, z, F.duneamp, seed + 2403);
  if (wMain <= 0) return DUNE_MEAN * amp;
  const ang = 0.6 + 0.9 * vnoise(x * F.dunefield, z * F.dunefield, seed + 2401);
  const ca = Math.cos(ang), sa = Math.sin(ang);
  const u = (x * ca + z * sa) * F.dune + 0.9 * vnoise(x * F.warp, z * F.warp, seed + 2402) + 0.35 * vnoise(x * F.dunebend, z * F.dunebend, seed + 2405);
  const s = u - Math.floor(u);
  const prof = s < 0.86 ? Math.pow(smoothstep(0, 0.86, s), 1.3) : 1 - smoothstep(0.86, 1.0, s);
  let d = 30 * prof;
  if (wCross > 0) {
    const u2 = (z * ca - x * sa) * F.dunecross + 0.6 * vnoise(x * F.dunebend, z * F.dunebend, seed + 2404);
    const s2 = u2 - Math.floor(u2);
    d += 5 * wCross * (s2 < 0.75 ? smoothstep(0, 0.75, s2) : 1 - smoothstep(0.75, 1.0, s2));
  }
  return amp * (wMain * d + (1 - wMain) * DUNE_MEAN);
}

/** Mesas: three tiers of flat-topped buttes, each a cliff over a talus skirt (m). Cliffs soften as dp grows. */
function mesasE(wx: number, wz: number, seed: number, dp: number): number {
  const n = 0.7 * n01(wx, wz, F.mesa, seed + 2501) + 0.3 * n01(wx, wz, F.mesafine, seed + 2502);
  const cw = Math.max(0.011, 0.0012 * dp);
  return 80 * (0.82 * smoothstep(0, cw, n - 0.42) + 0.18 * smoothstep(-0.07, 0, n - 0.42))
    + 70 * (0.82 * smoothstep(0, cw, n - 0.52) + 0.18 * smoothstep(-0.07, 0, n - 0.52))
    + 60 * (0.82 * smoothstep(0, cw, n - 0.61) + 0.18 * smoothstep(-0.07, 0, n - 0.61));
}


/** The rolling hills' fBm at warped macro coords (mx, mz), centred on zero (× the region's LANDFORM.hills). */
function hillsAt(mx: number, mz: number, seed: number, dp: number): number {
  const hLow = vnoise(mx, mz, seed + 1) + 0.5 * vnoise(mx * 2.0, mz * 2.0, seed + 1 + 40503);
  let hHigh = 0;
  if (dp <= 0) {
    hHigh = 0.25 * vnoise(mx * 4.0, mz * 4.0, seed + 1 + 81006)
      + 0.125 * vnoise(mx * 8.0, mz * 8.0, seed + 1 + 121509)
      + 0.0625 * vnoise(mx * 16.0, mz * 16.0, seed + 1 + 162012)
      + 0.03125 * vnoise(mx * 32.0, mz * 32.0, seed + 1 + 202515)
      + 0.015625 * vnoise(mx * 64.0, mz * 64.0, seed + 1 + 243018)
      + 0.0078125 * vnoise(mx * 128.0, mz * 128.0, seed + 1 + 283521)
      + 0.00390625 * vnoise(mx * 256.0, mz * 256.0, seed + 1 + 324024);
  } else {
    const k = 2.0 * F.macro * dp;
    for (let i = 2; i < 9; i++) {
      const fr = MACRO_FREQ[i];
      const wgt = Math.min(1, Math.max(0, 2.0 - fr * k));
      if (wgt <= 0.0) break;
      hHigh += MACRO_AMP[i] * wgt * vnoise(mx * fr, mz * fr, seed + 1 + i * 0x9e37);
    }
  }
  return 0.2833333 * hLow + 0.5 * hHigh;
}

/** The mountains' ridged octaves at warped (wx, wz), centred on their mean (× LANDFORM.ridge × the range there). */
function ridgesAt(wx: number, wz: number, seed: number, dp: number): number {
  const rx = wx * F.ridge, rz = wz * F.ridge;
  const k = 4.0 * F.ridge * dp;
  let rh = 0;
  for (let i = 0; i < 5; i++) {
    const fr = RIDGE_FREQ[i];
    const wgt = dp <= 0 ? 1 : Math.min(1, Math.max(0, 2.0 - fr * k));
    if (wgt > 0) {
      const n = 1.0 - Math.abs(vnoise(rx * fr, rz * fr, seed + 5 + i * 0x85eb));
      rh += RIDGE_AMP[i] * wgt * (n * n - RIDGE_MEAN);
    }
  }
  return rh;
}

/** The step (m) of the gullies' gradient, and the footprint their land is prefiltered at. */
export const ERO_STEP = 128;

/** The land the gullies follow at warped (wx, wz): the uplift U there and the hills and ridges over it, prefiltered at ERO_STEP (m). */
function erosionLand(wx: number, wz: number, U: number, seed: number, hills: number, ridge: number): number {
  const b = U + hills * hillsAt(wx * F.macro, wz * F.macro, seed, ERO_STEP);
  return ridge > 0 ? b + ridge * ridgesAt(wx, wz, seed, ERO_STEP) : b;
}

/** erosionLand at world (x, z). */
function erosionBase(x: number, z: number, seed: number, hills: number, ridge: number): number {
  warpInto(x, z, seed);
  const wx = _wx, wz = _wz;
  return erosionLand(wx, wz, upliftAt(wx, wz, seed), seed, hills, ridge);
}

/* ---- the land's surface and its water --------------------------------- */

/** A marsh's pools lie where its weight exceeds this; its rim (no pool, the land over the pools' level) fades in by MARSH_RIM. */
const MARSH_POOLS = 0.8, MARSH_RIM = 0.05;

/** The soft ceiling: heights ease toward MAX_WORLD_H − 10 m above CEIL_START. */
const ceilH = (h: number) => (h > CEIL_START ? CEIL_START + CEIL_SPAN * Math.tanh((h - CEIL_START) / CEIL_SPAN) : h);

/**
 * The land's surface at world (x, z): its height into _e (prefiltered for the footprint dp, spec §5)
 * and the water's into _wt (the sea, a lake, a river; dp-free), the flow into _fx, _fz, _speed.
 *  1. the geography: the uplift surface (hydro.ts upliftAt) and the regions over it;
 *  2. the regions' detail (hills, mountain ridges, karst, dunes, mesas) and the gully
 *     erosion over the hills and mountains, all fading out on the valley floors and in the lakes;
 *  3. the rivers' valleys cut into it (hydro.ts riverField: min of the land and every valley);
 *     banks held up along a channel where the land dips below its level; marshes laid flat on the
 *     nearest river's level;
 *  4. the coast and the sea floor, the soft ceiling;
 *  5. the water: a river's channel, a lake, a marsh's pools, the sea; elsewhere the table runs
 *     DRY_TABLE under the ground. A volcano's crater and its rivers carry lava.
 */
/**
 * The regions' detail over the uplift U at world (x, z), warped (wx, wz), before any mask (m): rolling
 * hills, mountain ridges, karst towers, dunes and mesas where their regions rule, and the gullies
 * over them all. `lf` is the blended landform there.
 */
function landDetail(x: number, z: number, wx: number, wz: number, seed: number, dp: number, U: number, range: number, massif: number, lf: Float64Array): number {
  let detail = lf[K_HILLS] * hillsAt(wx * F.macro, wz * F.macro, seed, dp);
  const rs = Math.min(1, 2 * range + 1.6 * massif);
  if (lf[K_RIDGE] > 0) detail += lf[K_RIDGE] * (0.3 + 0.7 * rs) * ridgesAt(wx, wz, seed, dp);
  // (karst, dunes and mesas only where their region rules: a few tiers in a meadow would be cliffs)
  const kw = smoothstep(0.35, 0.75, lf[K_KARST]), dw = smoothstep(0.35, 0.75, lf[K_DUNE]), mw = smoothstep(0.35, 0.75, lf[K_MESA]);
  if (kw > 0.001) {
    const wK = clamp(2 - 2 * dp / 150, 0, 1);
    detail += kw * (wK > 0 ? wK * karstTowers(wx, wz, seed) + (1 - wK) * TOWER_MEAN : TOWER_MEAN);
  }
  if (dw > 0.001) detail += dw * dunesE(x, z, seed, dp);
  if (mw > 0.001) detail += mw * mesasE(wx, wz, seed, dp);
  if (lf[K_ERO] > 0.01) {
    // (the gullies run down the land itself, its hills and ridges, and are laid out in the world: laid
    // out in the warped plane and down the uplift alone, they ran side by side for kilometres, a comb
    // of even crests. The step and the prefilter keep the twins' millimetre differences from turning them)
    const e = ERO_STEP, rk = lf[K_RIDGE] > 0 ? lf[K_RIDGE] * (0.3 + 0.7 * rs) : 0;
    const b0 = erosionLand(wx, wz, U, seed, lf[K_HILLS], rk);
    const gx = (erosionBase(x + e, z, seed, lf[K_HILLS], rk) - b0) / e;
    const gz = (erosionBase(x, z + e, seed, lf[K_HILLS], rk) - b0) / e;
    _wx = wx; _wz = wz;
    // (fade target 0: the filter carves gullies only. A target set by height, +1 on peaks and −1 in
    // valleys, lifted every gentle crest by up to ~150 m but not its steep flanks: walls along the
    // ridges, their tops cut into teeth by the gullies)
    detail += lf[K_ERO] * erosionDelta(x, z, U + detail, gx, gz, 0, dp);
  }
  return detail;
}

/** The land at world (x, z) before its rivers, lakes, marshes, ponds and the sea: the uplift and the regions' detail (m). */
function landBare(x: number, z: number, seed: number, dp: number): number {
  warpInto(x, z, seed);
  const wx = _wx, wz = _wz;
  const U = upliftAt(wx, wz, seed);
  const range = UP.range, massif = UP.massif, taper = UP.taper;
  regionsInto(x, z, wx, wz, seed, _w);
  blendLandform();
  return U + landDetail(x, z, wx, wz, seed, dp, U, range, massif, _lf) * taper;
}

/** The land a pond is judged on is prefiltered at this footprint (m), the same at every level of detail. */
const POND_DP = 4;
/** …at its centre and at three points round its rim (unit directions, the same in every twin). */
const POND_RING = [[1, 0], [-0.5, 0.8660254037844386], [-0.5, -0.8660254037844386]] as const;

function landSurface(x: number, z: number, seed: number, dp: number): void {
  // a pond first (pondAt evaluates the land at the pond's centre, overwriting the scratch)
  pondAt(x, z, seed);
  landCore(x, z, seed, dp, POND.L, POND.d, POND.D);
}

/** landSurface with the pond (level pL, or −Infinity: none; distance pd in radii, depth pD) given. */
function landCore(x: number, z: number, seed: number, dp: number, pL: number, pd: number, pD: number): void {
  warpInto(x, z, seed);
  const wx = _wx, wz = _wz;
  // 1. the geography and the regions
  const U = upliftAt(wx, wz, seed);
  const range = UP.range, massif = UP.massif, C = UP.C, cliff = UP.cliff, taper = UP.taper;
  const vx = VOL.x, vz = VOL.z, vR = VOL.R, lavaLvl = UP.lava, craterRR = UP.craterRR;
  regionsInto(x, z, wx, wz, seed, _w);
  blendLandform();
  const lf = _lf;
  riverField(wx, wz, seed, lf[K_K], true);
  // (no detail in a lake's ring, on a valley floor, nor in a crater: their floors are their own)
  const m = smoothstep(0, 50, RF.xFloor) * (1 - RF.lakeMask) * (lavaLvl > -Infinity ? smoothstep(CRATER_RIM * 0.9, CRATER_RIM * 1.3, craterRR) : 1);

  // where marshes lie (not in a lake's ring, nor in a river's channel; only on flat land: marsh pools
  // stepping up a hillside would stand at many levels)
  let flat = lf[K_FLAT] * (1 - RF.lakeMask) * (RF.vis > 0 ? smoothstep(RF.w + 2, RF.w + 8, RF.ch) : 1);
  if (flat > 0) {
    const gx = (upliftAt(wx + 8, wz, seed) - U) / 8, gz = (upliftAt(wx, wz + 8, seed) - U) / 8;
    flat *= 1 - smoothstep(0.012, 0.035, Math.hypot(gx, gz));
  }

  // 2. the regions' detail, and the gullies
  let detail = 0;
  if (m > 0) {
    detail = landDetail(x, z, wx, wz, seed, dp, U, range, massif, lf);
    detail *= m * taper; // (none toward a beach: the coast ramps down smoothly; cliffs keep theirs)
    // (round a river the detail only rises: a hollow beside it would leave the river standing over it)
    const near = 1 - smoothstep(50, 450, RF.xFloor);
    if (near > 0) detail += (0.5 * (detail + Math.sqrt(detail * detail + 36)) - 3 - detail) * near;
    // (and round a marsh: its pools lie just under the smooth land, and a hollow at its edge would spill them)
    const nearM = smoothstep(0, 0.3, flat);
    if (nearM > 0) detail += (0.5 * (detail + Math.sqrt(detail * detail + 0.36)) - 0.3 - detail) * nearM;
  }
  let e0 = U + detail;

  // 3. the valleys cut in, the terraces built up where the land falls away beside a river (never
  // above any valley: no terrace buries another river's channel), the marshes
  e0 = Math.min(Math.max(e0, RF.F), RF.R);
  const vis = RF.vis;
  // the river's water: its level, but never over a bank another valley has cut lower
  const riverW = Math.min(RF.W, RF.cap);
  // a crater's rim held above its lava lake (valleys and gullies would breach it)
  if (lavaLvl > -Infinity) {
    const band = smoothstep(CRATER_RIM * 0.82, CRATER_RIM, craterRR) * (1 - smoothstep(CRATER_RIM * 1.3, CRATER_RIM * 1.6, craterRR));
    if (lavaLvl + 4 > e0) e0 += (lavaLvl + 4 - e0) * band;
  }
  // the marshes
  let marshW = -Infinity;
  if (flat > 0) {
    // a marsh lies flat on the smooth land (the uplift, a little under it): hummocks, and pools between
    const wH = clamp(2 - 2 * dp * F.hummock, 0, 1), wH2 = clamp(2 - 2 * dp * F.hummockfine, 0, 1);
    const hum = 1.4 * (wH * vnoise(x * F.hummock, z * F.hummock, seed + 2801) + 0.5 * wH2 * vnoise(x * F.hummockfine, z * F.hummockfine, seed + 2802)) - 0.2;
    const lvl = Math.max(U - 1.2, LANDS.sea);
    e0 += (lvl + hum - e0) * flat;
    marshW = lvl + 0.25;
    // pools lie only in the marsh's heart; round it the hummocks stand at least a little over them (a
    // rim faded in from the marsh's very edge), so no pool spills where the land beyond lies low
    // (raised toward the rim in proportion: a switch of tens of metres would turn the twins' rounding
    // in the marsh's weight into steps of the land)
    e0 += Math.max(0, marshW + 0.05 - e0) * smoothstep(0, MARSH_RIM, flat) * (1 - smoothstep(MARSH_POOLS, MARSH_POOLS + 0.02, flat));
  }

  // a pond in a meadow hollow: its bowl, and round it the land eased to its shore's level (a hollow
  // where the land stands higher, a low rise where it falls away)
  let pondL = -Infinity;
  if (pL > -Infinity) {
    if (pd < 1) {
      e0 = pL - pD * (1 - smoothstep(0.45, 0.92, pd)) + HYDRO.pondRim * smoothstep(0.8, 1, pd);
      pondL = pL;
    } else {
      e0 += (pL + HYDRO.pondRim - e0) * (1 - smoothstep(1, HYDRO.pondReach, pd));
    }
  }

  // 4. the coast: the land meets the sea (U already ramps down to beaches, or stands as cliffs)
  const K = LANDS.coast;
  _land = smoothstep(K.level - 0.004 - 0.03 * (1 - cliff), K.level + 0.002, C);
  const seaFloor = LANDS.sea - 6 - K.shelf * (1 - smoothstep(K.level - 0.12, K.level, C));
  const e = seaFloor + (e0 - seaFloor) * _land;

  // 5. the water
  _fx = 0; _fz = 0; _speed = 0; _riverK = 0;
  let w = e - DRY_TABLE;
  // (lava stands only on a volcano's gentler reaches: down its steep flanks a channel is cooled rock)
  const hot = vR > 0 && Math.hypot(wx - vx, wz - vz) < 0.6 * LANDS.volcano.cone * vR;
  // (toward the sea the river's water sinks with its banks: they ease down to the sea floor together)
  const riverWc = riverW > -Infinity ? seaFloor + (riverW - seaFloor) * _land : -Infinity;
  if (vis > 0 && RF.ch < RF.w + 0.3 && riverWc > e && !(hot && RF.fall > 0.3)) {
    w = Math.max(w, riverWc);
    _riverK = vis;
    // the flow: down the reach, faster where it falls (m/s)
    const sp = clamp(0.35 + 26 * RF.fall, 0.3, 4.5);
    // the reach's direction is in the warped plane: back to the world's through the warp's Jacobian
    const dfx = RF.fx, dfz = RF.fz;
    warpInto(x + 1, z, seed); const exx = _wx - wx, exz = _wz - wz;
    warpInto(x, z + 1, seed); const ezx = _wx - wx, ezz = _wz - wz;
    _wx = wx; _wz = wz;
    const det = exx * ezz - ezx * exz;
    let fx = (ezz * dfx - ezx * dfz) / det, fz = (-exz * dfx + exx * dfz) / det;
    const fl = Math.max(Math.hypot(fx, fz), 1e-6);
    fx /= fl; fz /= fl;
    _fx = fx; _fz = fz; _speed = sp;
  }
  _lakeK = 0;
  const lakeL = RF.L > -Infinity ? seaFloor + (RF.L - seaFloor) * _land : -Infinity;
  if (lakeL > e) { w = Math.max(w, lakeL); _lakeK = RF.lakeK; _fx = 0; _fz = 0; _speed = 0; }
  if (pondL > e) { w = Math.max(w, pondL); _fx = 0; _fz = 0; _speed = 0; } // a pond
  if (e < LANDS.sea && C < LANDS.coast.level + LANDS.coast.inland) w = Math.max(w, LANDS.sea); // a lagoon
  if (lavaLvl > e && craterRR < CRATER_RIM) { w = Math.max(w, lavaLvl); _fx = 0; _fz = 0; _speed = 0; } // a crater's lava lake
  const marshWc = marshW > -Infinity ? seaFloor + (marshW - seaFloor) * _land : -Infinity;
  if (flat > MARSH_POOLS && marshWc > e) w = Math.max(w, marshWc);
  // (the sea fills what lies below its level by the coast: offshore, the beaches' foot, a cliff's —
  // never more: water eased toward the sea's level would climb the shore)
  // the soft ceiling, on the ground and on every water's level alike (monotone: what lay under
  // the water still does)
  // the level of the water beside this point, if any (a river's bank, a lake's ring, a pond's
  // hollow, the coast, a marsh): what shores are judged from (dry land's table is no water)
  let shore = -Infinity;
  if (vis > 0 && RF.ch < RF.w + 14 && !(hot && RF.fall > 0.3)) shore = riverWc;
  if (RF.Lnear > -Infinity) shore = Math.max(shore, seaFloor + (RF.Lnear - seaFloor) * _land);
  if (pL > -Infinity && pd < HYDRO.pondReach) shore = Math.max(shore, pL);
  if (C < LANDS.coast.level + LANDS.coast.inland) shore = Math.max(shore, LANDS.sea);
  if (flat > MARSH_POOLS) shore = Math.max(shore, marshWc);
  _shore = shore > -Infinity ? ceilH(shore) : -Infinity;
  _e = ceilH(e);
  _wt = ceilH(w);
  if (_land < 1) { _fx *= _land; _fz *= _land; _speed *= _land; }
  // lava: the liquid of a volcano's crater and of the rivers that rise high on its cone
  _lava = 0;
  if (vR > 0 && w > e - 0.01) {
    const r = Math.hypot(wx - vx, wz - vz) / (LANDS.volcano.cone * vR);
    _lava = 1 - smoothstep(0.45, 0.6, r);
  }
}

/**
 * The land's broad shape at world (x, z) (m): the uplift surface and the regions' rolling hills,
 * prefiltered for dp; no valleys, landforms or gullies. Cheap: what the forest's placement reads
 * its hollows, knolls and hill-scale relief from (WGSL twin: worldgen.wgsl.ts landCoarse).
 */
export function landCoarse(x: number, z: number, seed: number, dp: number): number {
  warpInto(x, z, seed);
  const wx = _wx, wz = _wz;
  const U = upliftAt(wx, wz, seed);
  regionsInto(x, z, wx, wz, seed, _w);
  let hills = 0;
  for (let i = 0; i < NB; i++) if (_w[i] > 0) hills += _w[i] * LF_TABLE[i * NK + K_HILLS];
  return U + hills * hillsAt(wx * F.macro, wz * F.macro, seed, dp);
}

/** The pond pondAt found (module scratch): its level (m; −Infinity: none), distance from its centre over its radius, its depth. */
const POND = { L: -Infinity, d: 1, D: 0, R: 0 };
/** The last pond cell judged and its verdict (level, or −Infinity: none; depth): the points round a pond share it. */
const PCELL = { ix: NaN, iz: NaN, seed: NaN, L: -Infinity, D: 0 };

/**
 * A pond whose reach (HYDRO.pondReach radii) holds world (x, z): one candidate per F.pond cell, kept inside its
 * cell, in a region whose LANDFORM.pond allows it, on flat land away from rivers and lakes; its level
 * just under the land itself at its centre (landBare, all its detail). Fills POND.
 */
function pondAt(x: number, z: number, seed: number): void {
  POND.L = -Infinity;
  const cell = 1 / F.pond;
  const ix = Math.floor(x * F.pond), iz = Math.floor(z * F.pond);
  const h = hash3(ix, iz, seed + 3401);
  const u = (h >>> 24) / 255;
  const R = HYDRO.pondR[0] + (HYDRO.pondR[1] - HYDRO.pondR[0]) * u * u * u;
  const inset = HYDRO.pondReach * 1.3 * R / cell; // (the lobed reach: pondReach radii of up to 1.3 R)
  const cx = (ix + inset + (1 - 2 * inset) * (h & 0xfff) / 4095) * cell, cz = (iz + inset + (1 - 2 * inset) * ((h >>> 12) & 0xfff) / 4095) * cell;
  // (a lobed shore: the radius breathes with a noise over the pond's own scale)
  const d = Math.hypot(x - cx, z - cz) / (R * (1 + 0.3 * vnoise((x - cx) / R * 1.2, (z - cz) / R * 1.2, seed + 3403)));
  if (d >= HYDRO.pondReach) return;
  if (ix === PCELL.ix && iz === PCELL.iz && seed === PCELL.seed) {
    if (PCELL.L > -Infinity) { POND.L = PCELL.L; POND.d = d; POND.R = R; POND.D = PCELL.D; }
    return;
  }
  PCELL.ix = ix; PCELL.iz = iz; PCELL.seed = seed; PCELL.L = -Infinity;
  const h2 = hash3(ix, iz, seed + 3402);
  // the centre's region (its chance of a pond)
  warpInto(cx, cz, seed);
  const wx = _wx, wz = _wz;
  const U = upliftAt(wx, wz, seed);
  if (UP.C < LANDS.coast.level + 0.02) return;
  const range = UP.range, massif = UP.massif, taper = UP.taper;
  regionsInto(cx, cz, wx, wz, seed, _w);
  let chance = 0;
  for (let i = 0; i < NB; i++) if (_w[i] > 0) chance += _w[i] * LF_TABLE[i * NK + K_POND];
  if ((h2 & 0xffff) / 65535 >= chance) return;
  // the land itself at its centre, with all its detail (the pond lies just under it), and round its
  // rim: a pond needs flat land (judged on the broad shape alone, a pond in mesa or hill country
  // would sink a shaft to it)
  blendLandform();
  const ec = U + landDetail(cx, cz, wx, wz, seed, POND_DP, U, range, massif, _lf) * taper;
  for (const [ux, uz] of POND_RING) {
    if (Math.abs(landBare(cx + 1.25 * R * ux, cz + 1.25 * R * uz, seed, POND_DP) - ec) > 0.06 * R) return;
  }
  // away from rivers and lakes
  riverField(wx, wz, seed, 0.12, true);
  if (RF.xFloor < 2 * R + 40 || RF.lakeMask > 0) return;
  POND.L = PCELL.L = ec - HYDRO.pondSink;
  POND.d = d;
  POND.R = R;
  POND.D = PCELL.D = HYDRO.pondD[0] + (HYDRO.pondD[1] - HYDRO.pondD[0]) * (h2 >>> 16) / 65535;
}

/**
 * §2/§6/§14 — E(x, z) — terrain elevation in metres. `dp` is the Nyquist pixel footprint
 * (spec §5): pass ~0 for full-detail CPU evaluation, or the screen footprint
 * to mirror the GPU's filtered result. All lookups are chunk-local (spec §25).
 * No object or array allocation (see PARITY NOTES for V8 number boxing).
 */
export function elevation(x: number, z: number, seed: number, dp: number): number {
  landSurface(x, z, seed, dp);
  return _e;
}

/**
 * The water's surface at world (x, z): the sea, a lake's level or a river's pool (m). Where it lies
 * above elevation() there is water (its depth the difference); elsewhere it runs under the ground.
 */
export function waterLevel(x: number, z: number, seed: number): number {
  landSurface(x, z, seed, 0);
  return _wt;
}

/** The ground and the water at world (x, z) in one evaluation: [elevation (prefiltered for dp), water level]. */
export function surfaceAt(x: number, z: number, seed: number, out: [number, number] = [0, 0], dp = 0): [number, number] {
  landSurface(x, z, seed, dp);
  out[0] = _e;
  out[1] = _wt;
  return out;
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

/** §9 — sea-level temperature (°C) at world (x, z): the 5.8 km climate field less the latitude term. */
export function seaTemperature(x: number, z: number, seed: number): number {
  warpInto(x, z, seed);
  return CLIMATE.tBase + CLIMATE.tAmp * (vnoise(_wx * F.temp, _wz * F.temp, seed + 1201) + CLIMATE.tFine * vnoise(_wx * F.tempfine, _wz * F.tempfine, seed + 1202))
    - CLIMATE.lambdaLat * Math.abs(z);
}

/** §9/§13 — Temperature (°C) at world (x, z) and elevation e: the sea-level field less the lapse with height. */
export function temperature(x: number, z: number, e: number, seed: number): number {
  return seaTemperature(x, z, seed) - CLIMATE.lambdaAlt * Math.max(0, e);
}

/** §9/§13 — Moisture (≈ 0–1.1) at world (x, z): the 3.2 km field, wetter toward the sea. */
export function moisture(x: number, z: number, _e: number, seed: number): number {
  warpInto(x, z, seed);
  const C = continentAt(_wx, _wz, seed);
  const S = LANDS.coast.level;
  return CLIMATE.mBase + CLIMATE.mNoiseAmp * (vnoise(_wx * F.moistr, _wz * F.moistr, seed + 1301) + CLIMATE.mFine * vnoise(_wx * F.moistfine, _wz * F.moistfine, seed + 1302))
    + CLIMATE.mOceanLambda * (1 - smoothstep(S, S + 0.12, C));
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
/**
 * Walkable surfaces over the terrain for updatePlayer (sky islands, ruins):
 * given the player and the height of its feet, the ground there, or −∞. It
 * may push the player out of walls (x, z, and the velocity into them).
 */
type ExtraGround = (s: PlayerState, feetY: number, seed: number) => number;

export function updatePlayer(
  s: PlayerState, input: InputState, dt: number, seed: number, extraGround?: ExtraGround,
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
  const extra = extraGround ? extraGround(s, s.y - PLAYER.height - PLAYER.groundBias, seed) : -Infinity;
  const terrain = elevation(s.x, s.z, seed, 0);
  const onExtra = extra > terrain;
  const groundZ = (onExtra ? extra : terrain) + PLAYER.groundBias + PLAYER.height;

  s.grounded = false;
  if (s.y < groundZ) {
    s.y = groundZ;
    if (s.vy < 0.0) s.vy = 0.0;            // stop FALLING only — preserve jumps
    s.grounded = true;
  }

  // Slope gate: slide off steep warped ridges (spec §31 insert, zero-allocation)
  if (s.grounded && !onExtra) {
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
