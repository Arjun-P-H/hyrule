/**
 * District maps: where an Eyrie's island is paved and where it is built on,
 * as two distance fields over its square (SKY_CIVILIZATION.md §11 rule 9).
 * ---------------------------------------------------------------------------
 * Per settled island, DISTRICT_N² texels over the square of side 2·h round its
 * centre (h its largest radius): channel 0 the signed distance (m) to its
 * paving — streets, plazas, bridgeheads — and channel 1 to its footprints,
 * each stored as 0.5 + d / (2 · DISTRICT_RANGE) in a byte (±DISTRICT_RANGE m,
 * 1/8 m steps), negative inside. Bilinear filtering of a distance field keeps a
 * straight edge exact between texels, so even the Crown's 4–5 m texels draw a
 * street's edges sharply.
 *
 * The ground's shader paints cobbles and flagstones where it is paved; the
 * grass keeps off paving and footprints, the trees farther off (gpu: realm
 * district textures; CPU: flora.ts through districtAt), so nothing grows in a
 * street or through a wall.
 */

import type { DistrictMap, SkyIsland } from '../island';

export const DISTRICT_N = 320;
/** Bumped whenever an island gets its map (the realm uploads maps again when it changes). */
export let districtVersion = 0;
export function districtAdded(): void { districtVersion++; }
/** Distances are kept to ±this (m). */
export const DISTRICT_RANGE = 16;

/** A paved or built shape, island-local: a disc, a segment with a half width, or an oriented box (half extents along its yaw's axes). */
export type DShape =
  | { k: 'disc'; x: number; z: number; r: number }
  | { k: 'seg'; ax: number; az: number; bx: number; bz: number; w: number }
  | { k: 'box'; x: number; z: number; hx: number; hz: number; yaw: number };


function bounds(s: DShape): [number, number, number, number] {
  if (s.k === 'disc') return [s.x - s.r, s.z - s.r, s.x + s.r, s.z + s.r];
  if (s.k === 'seg') return [Math.min(s.ax, s.bx) - s.w, Math.min(s.az, s.bz) - s.w, Math.max(s.ax, s.bx) + s.w, Math.max(s.az, s.bz) + s.w];
  const e = Math.hypot(s.hx, s.hz);
  return [s.x - e, s.z - e, s.x + e, s.z + e];
}

/** Texels stamped between yields (~1 ms). */
const STEP_TEXELS = 40000;

/** Row j's texels i0 … i1 of field F (row centre z, first texel centre x) take the smaller of theirs and the distance to s. */
function stampRow(F: Float32Array, s: DShape, k0: number, i0: number, i1: number, x: number, cell: number, z: number): void {
  if (s.k === 'disc') {
    const dz2 = (z - s.z) ** 2;
    for (let i = i0; i <= i1; i++, x += cell) {
      const d = Math.sqrt((x - s.x) ** 2 + dz2) - s.r;
      if (d < F[k0 + i]) F[k0 + i] = d;
    }
  } else if (s.k === 'seg') {
    const dx = s.bx - s.ax, dz = s.bz - s.az;
    const l2 = dx * dx + dz * dz, inv = l2 > 0 ? 1 / l2 : 0;
    const pz = z - s.az;
    for (let i = i0; i <= i1; i++, x += cell) {
      const px = x - s.ax;
      const t = Math.min(1, Math.max(0, (px * dx + pz * dz) * inv));
      const d = Math.sqrt((px - t * dx) ** 2 + (pz - t * dz) ** 2) - s.w;
      if (d < F[k0 + i]) F[k0 + i] = d;
    }
  } else {
    // in the box's own axes (local x along (cos yaw, −sin yaw), z along (sin yaw, cos yaw), as the structures')
    const c = Math.cos(s.yaw), sn = Math.sin(s.yaw);
    const pz = z - s.z;
    for (let i = i0; i <= i1; i++, x += cell) {
      const px = x - s.x;
      const qx = Math.abs(px * c - pz * sn) - s.hx, qz = Math.abs(px * sn + pz * c) - s.hz;
      const ox = qx > 0 ? qx : 0, oz = qz > 0 ? qz : 0;
      const d = Math.sqrt(ox * ox + oz * oz) + Math.min(qx > qz ? qx : qz, 0);
      if (d < F[k0 + i]) F[k0 + i] = d;
    }
  }
}

/**
 * Build an island's district map from its paving and footprints (world coordinates), a few rows
 * of texels per step (a generator: yields every STEP_TEXELS texels, about a millisecond).
 */
export function* buildDistrict(isl: SkyIsland, paving: readonly DShape[], built: readonly DShape[]): Generator<void, DistrictMap, void> {
  const N = DISTRICT_N, R = DISTRICT_RANGE;
  const half = isl.R * (1 + 1.12 * isl.amp) + 4;
  const cell = (2 * half) / N;
  const field = [new Float32Array(N * N).fill(R), new Float32Array(N * N).fill(R)];
  const x0 = isl.cx - half, z0 = isl.cz - half;
  let work = 0;
  for (const [ch, list] of [[0, paving], [1, built]] as const) {
    const F = field[ch];
    for (const s of list) {
      const [bx0, bz0, bx1, bz1] = bounds(s);
      const i0 = Math.max(0, Math.floor((bx0 - R - x0) / cell)), i1 = Math.min(N - 1, Math.ceil((bx1 + R - x0) / cell));
      const j0 = Math.max(0, Math.floor((bz0 - R - z0) / cell)), j1 = Math.min(N - 1, Math.ceil((bz1 + R - z0) / cell));
      if (i1 < i0) continue;
      for (let j = j0; j <= j1; j++) {
        stampRow(F, s, j * N, i0, i1, x0 + (i0 + 0.5) * cell, cell, z0 + (j + 0.5) * cell);
        work += i1 - i0 + 1;
        if (work >= STEP_TEXELS) { work = 0; yield; }
      }
    }
  }
  // (the encoding, a quarter of the rows per step; a clamped array's stores clamp and round)
  const data = new Uint8Array(N * N * 2);
  const out = new Uint8ClampedArray(data.buffer);
  const P = field[0], B = field[1], k255 = 255 / (2 * R);
  for (let q = 0; q < 4; q++) {
    yield;
    for (let k = (q * N * N) / 4, k1 = ((q + 1) * N * N) / 4; k < k1; k++) {
      out[2 * k] = 127.5 + P[k] * k255;
      out[2 * k + 1] = 127.5 + B[k] * k255;
    }
  }
  return { half, cell, data };
}

/**
 * The distances (m) from world (x, z) to an island's paving and to its footprints (bilinear, as the
 * GPU samples them), into `out`; DISTRICT_RANGE each where it has no map. Allocates nothing.
 */
export function districtAt(isl: SkyIsland, x: number, z: number, out: [number, number]): [number, number] {
  const m = isl.district;
  out[0] = DISTRICT_RANGE; out[1] = DISTRICT_RANGE;
  if (!m) return out;
  const N = DISTRICT_N;
  const fx = (x - isl.cx + m.half) / m.cell - 0.5, fz = (z - isl.cz + m.half) / m.cell - 0.5;
  if (fx < 0 || fz < 0 || fx > N - 1 || fz > N - 1) return out;
  const i = Math.floor(fx), j = Math.floor(fz), tx = fx - i, tz = fz - j;
  const i1 = Math.min(i + 1, N - 1), j1 = Math.min(j + 1, N - 1);
  for (let c = 0; c < 2; c++) {
    const a = m.data[2 * (j * N + i) + c], b = m.data[2 * (j * N + i1) + c], d = m.data[2 * (j1 * N + i) + c], e = m.data[2 * (j1 * N + i1) + c];
    const v = ((a * (1 - tx) + b * tx) * (1 - tz) + (d * (1 - tx) + e * tx) * tz) / 255;
    out[c] = (v - 0.5) * 2 * DISTRICT_RANGE;
  }
  return out;
}
