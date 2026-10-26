/**
 * The map's ground, drawn from the world's own maths (terrainMath.ts) into square tiles.
 * ---------------------------------------------------------------------------
 * A tile is TILE_PX × TILE_PX pixels at a zoom level: level L has 2·2^L metres to the pixel, and tile
 * (tx, tz) covers x ∈ [tx, tx+1)·span and z ∈ [tz, tz+1)·span, span = TILE_PX · mpp. The map is north-up
 * with north = +z (facing +z the camera's right is +x, so east lies to the right): image row 0 is the
 * tile's northern edge.
 *
 * Each pixel takes the colour of the lands there (the biome regions' weights over a palette), lightens
 * with height to bare rock and snow where the air is below freezing, is shaded by the slope under a light
 * from the north-west, and drawn over with contour lines. Water is blue, deeper darker; ice and lava
 * have their own colours. Pure maths, no DOM: tilesWorker.ts runs it off the main thread.
 */

import { BIOMES } from '../biomes';
import { CLIMATE, LANDS } from '../constants';
import { lastClimate, lastIce, lastLava, lastRegions, surfaceAt } from '../terrainMath';

export const TILE_PX = 128;
/** Metres to the pixel at level 0; each level doubles it. */
const BASE_MPP = 2;
export const MAX_LEVEL = 7;

export const levelMpp = (level: number): number => BASE_MPP * 2 ** level;
export const levelSpan = (level: number): number => TILE_PX * levelMpp(level);

type RGB = readonly [number, number, number];

const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];

/** The ground's colour per biome (BIOMES order), muted and earthy so the markers read on any of them. */
const PALETTE: Record<(typeof BIOMES)[number], RGB> = {
  plains: hex('#9cb765'), forest: hex('#5b8a4e'), rainforest: hex('#3d7a4d'), jungle: hex('#4b8a5f'),
  wetland: hex('#6f9877'), desert: hex('#dcbd7e'), badlands: hex('#c07a47'), savanna: hex('#bcae5d'),
  volcanic: hex('#675450'), mountains: hex('#9a9487'), highland: hex('#9db172'), tundra: hex('#cdd7da'),
  taiga: hex('#587a5f'), sacred: hex('#67a965'), lakes: hex('#84ae79'),
};
const BIOME_COLOUR: readonly RGB[] = BIOMES.map((b) => PALETTE[b]);

const ROCK: RGB = [0.62, 0.6, 0.56];
const SNOW: RGB = [0.94, 0.96, 0.98];
const SHALLOW: RGB = [0.42, 0.7, 0.8];
const DEEP: RGB = [0.07, 0.22, 0.42];
const ICE: RGB = [0.83, 0.91, 0.96];
const LAVA: RGB = [0.96, 0.4, 0.1];

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

/** Metres between contour lines at a zoom level (and every fifth one is an index line). */
export function contourInterval(level: number): number {
  return level <= 1 ? 20 : level <= 3 ? 50 : level <= 5 ? 100 : 250;
}

const LIGHT = (() => {
  const l: [number, number, number] = [-0.55, 0.55, 0.62];
  const n = Math.hypot(l[0], l[1], l[2]);
  return [l[0] / n, l[1] / n, l[2] / n] as const;
})();
/** How steep the ground reads in the shading (the heights are exaggerated). */
const EXAGGERATE = 2.4;

/**
 * Draw tile (level, tx, tz) of the world `seed` into `out` (TILE_PX² RGBA, opaque). The tile's heights
 * are sampled with a pixel's border all round, so the shading and the contours meet across tiles.
 */
export function renderTile(seed: number, level: number, tx: number, tz: number, out: Uint8ClampedArray | Uint8Array): void {
  const mpp = levelMpp(level);
  const span = TILE_PX * mpp;
  const x0 = tx * span;
  const zTop = (tz + 1) * span;
  const N = TILE_PX + 2;
  const E = new Float32Array(N * N);
  const colour = new Float32Array(TILE_PX * TILE_PX * 3);
  // 0 land, 1 water, 2 ice, 3 lava (per pixel); the water's depth (m) for the water
  const kind = new Uint8Array(TILE_PX * TILE_PX);
  const depth = new Float32Array(TILE_PX * TILE_PX);
  const snow = new Float32Array(TILE_PX * TILE_PX);
  const at: [number, number] = [0, 0];
  const sea = LANDS.sea;

  for (let b = 0; b < N; b++) {
    const z = zTop - (b - 0.5) * mpp;
    for (let a = 0; a < N; a++) {
      const x = x0 + (a - 0.5) * mpp;
      surfaceAt(x, z, seed, at, mpp);
      const e = at[0];
      E[b * N + a] = e;
      if (a === 0 || b === 0 || a === N - 1 || b === N - 1) continue;
      const p = (b - 1) * TILE_PX + (a - 1);
      const water = at[1] - e;
      if (water > 0.05) {
        const lava = lastLava() > 0.5;
        kind[p] = lava ? 3 : lastIce() ? 2 : 1;
        depth[p] = water;
        continue;
      }
      // the lands: the regions' weights over the palette
      const w = lastRegions();
      let r = 0, g = 0, bl = 0;
      for (let k = 0; k < BIOME_COLOUR.length; k++) {
        const c = BIOME_COLOUR[k], wk = w[k];
        r += wk * c[0]; g += wk * c[1]; bl += wk * c[2];
      }
      // bare rock with height, snow where the air is below freezing
      const rock = clamp01((e - 350) / 900) * 0.7;
      r = mix(r, ROCK[0], rock); g = mix(g, ROCK[1], rock); bl = mix(bl, ROCK[2], rock);
      const air = lastClimate().T0 - CLIMATE.lambdaAlt * Math.max(0, e);
      snow[p] = clamp01((CLIMATE.snowT + 2 - air) / 3);
      colour[p * 3] = r; colour[p * 3 + 1] = g; colour[p * 3 + 2] = bl;
    }
  }

  const ci = contourInterval(level);
  const inv = 1 / (2 * mpp);
  for (let j = 0; j < TILE_PX; j++) {
    for (let i = 0; i < TILE_PX; i++) {
      const p = j * TILE_PX + i;
      const g = (j + 1) * N + (i + 1);
      const e = E[g];
      let r: number, gr: number, bl: number;
      // the slope under the light from the north-west
      const gx = (E[g + 1] - E[g - 1]) * inv * EXAGGERATE;
      const gy = (E[g - N] - E[g + N]) * inv * EXAGGERATE;
      const nl = Math.hypot(gx, gy, 1);
      const lit = (-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) / nl / LIGHT[2];
      const shade = Math.min(1.3, Math.max(0.55, 1 + 0.85 * (lit - 1)));
      const k = kind[p];
      if (k === 0) {
        r = colour[p * 3]; gr = colour[p * 3 + 1]; bl = colour[p * 3 + 2];
        const s = snow[p];
        if (s > 0) { r = mix(r, SNOW[0], s); gr = mix(gr, SNOW[1], s); bl = mix(bl, SNOW[2], s); }
        r *= shade; gr *= shade; bl *= shade;
        // contour lines: where the next pixel east or south is over a multiple of the interval
        const h0 = Math.floor((e - sea) / ci);
        const hE = Math.floor((E[g + 1] - sea) / ci);
        const hS = Math.floor((E[g + N] - sea) / ci);
        if (h0 !== hE || h0 !== hS) {
          // (every fifth line, counted from the sea, is an index line)
          const line = h0 !== hE ? Math.max(h0, hE) : Math.max(h0, hS);
          const dark = line % 5 === 0 ? 0.62 : 0.8;
          r *= dark; gr *= dark; bl *= dark;
        }
      } else if (k === 1) {
        const t = 1 - Math.exp(-depth[p] / 45);
        r = mix(SHALLOW[0], DEEP[0], t); gr = mix(SHALLOW[1], DEEP[1], t); bl = mix(SHALLOW[2], DEEP[2], t);
        const wave = 0.96 + 0.04 * shade; // (the water takes a hint of the relief under it)
        r *= wave; gr *= wave; bl *= wave;
      } else if (k === 2) {
        r = ICE[0] * (0.9 + 0.1 * shade); gr = ICE[1] * (0.9 + 0.1 * shade); bl = ICE[2] * (0.9 + 0.1 * shade);
      } else {
        r = LAVA[0]; gr = LAVA[1]; bl = LAVA[2];
      }
      const o = p * 4;
      out[o] = Math.round(clamp01(r) * 255);
      out[o + 1] = Math.round(clamp01(gr) * 255);
      out[o + 2] = Math.round(clamp01(bl) * 255);
      out[o + 3] = 255;
    }
  }
}

/** Tile indices of the tiles at `level` that a square of half-width `r` round (x, z) touches. */
export function tilesAround(level: number, x: number, z: number, r: number): { tx0: number; tx1: number; tz0: number; tz1: number } {
  const span = levelSpan(level);
  return {
    tx0: Math.floor((x - r) / span), tx1: Math.floor((x + r) / span),
    tz0: Math.floor((z - r) / span), tz1: Math.floor((z + r) / span),
  };
}

/** The zoom level whose tiles suit a view at `viewMpp` metres to the pixel (0…MAX_LEVEL). */
export function levelFor(viewMpp: number): number {
  return Math.min(MAX_LEVEL, Math.max(0, Math.round(Math.log2(viewMpp / BASE_MPP))));
}
