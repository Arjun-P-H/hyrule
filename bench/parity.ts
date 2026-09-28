/**
 * GPU fp32 vs CPU fp64 parity, measured on the real GPU.
 * ---------------------------------------------------------------------------
 * The probe shader evaluates the production elevation() at points expressed
 * exactly as the renderer does — an fp32 chunk-local offset added to
 * u_chunk_origin (`p.xz + u_chunk_origin`) — and reads E back through an
 * RGBA32F target together with the fp32 global coordinate the GPU actually
 * used. The CPU twin then evaluates the intended fp64 point.
 *
 * Divergence is split into:
 *   total      |E_gpu − E_cpu(x_true)|         what the player feels (vs GROUND_BIAS)
 *   arithmetic |E_gpu − E_cpu(x_gpu_fp32)|     fp32 math inside elevation()
 *   coordinate |E_cpu(x_gpu_fp32) − E_cpu(x_true)|  rounding of local + origin
 *   rebase     |E_gpu(from chunk k−1) − E_gpu(from chunk k)|  same world point,
 *              adjacent floating origins (the chunk-crossing "seam")
 *
 * Note: the live shader re-globalizes before evaluating noise, so its
 * precision is governed by |x_global|, not by the chunk size; the distance
 * sweep below is what bounds GROUND_BIAS.
 */

import type { ProbeSpec } from './gpu/runner';
import type { BlobRecord } from './gpu/runner';
import type { TerrainMathModule } from './lib/revision';
import { CHUNK_SIZE, DEFAULT_SEED } from '../src/lib/constants';

const N = 64;
// Non-dyadic spacings and offsets, like real camera positions: dyadic values
// make local + origin exact in fp32 and would hide coordinate rounding.
const FINE = 0.2371;  // m — player-scale sampling
const COARSE = 31.37; // m — 2 km patch through mixed terrain

function at(id: string, worldX: number, worldZ: number, step: number, w = N, h = N): ProbeSpec {
  const ox = Math.floor(worldX / CHUNK_SIZE) * CHUNK_SIZE;
  const oz = Math.floor(worldZ / CHUNK_SIZE) * CHUNK_SIZE;
  return { id, origin: [ox, oz], local0: [worldX - ox, worldZ - oz], step, w, h, dp: 1e-4 };
}

export const DISTANCES = [0, 10_000, 100_000, 250_000, 500_000, 1_000_000];
export const SEAMS = [0, 1, 12, 122];

/** Probe grids: spawn, distance sweep (fine + coarse), and chunk-boundary straddles. */
export function probeSpecs(): ProbeSpec[] {
  const specs: ProbeSpec[] = [];
  for (const d of DISTANCES) {
    specs.push(at(`dist${d}-fine`, 2016.137 + d, -2592.419 + d * 0.37, FINE));
    specs.push(at(`dist${d}-coarse`, 2016.137 + d, -2592.419 + d * 0.37, COARSE));
  }
  for (const k of SEAMS) {
    const bx = k * CHUNK_SIZE;
    const z0 = -2592.419 + k * 101;
    const oz = Math.floor(z0 / CHUNK_SIZE) * CHUNK_SIZE;
    // same world points [bx − 8, bx + 8) seen from both sides of the boundary
    specs.push({ id: `seam${k}-left`, origin: [bx - CHUNK_SIZE, oz], local0: [CHUNK_SIZE - 7.913, z0 - oz], step: FINE, w: N, h: N, dp: 1e-4 });
    specs.push({ id: `seam${k}-right`, origin: [bx, oz], local0: [-7.913, z0 - oz], step: FINE, w: N, h: N, dp: 1e-4 });
  }
  return specs;
}

export type ParityStats = {
  id: string;
  distance: number;
  points: number;
  totalMax: number;
  totalP99: number;
  totalMean: number;
  arithmeticMax: number;
  coordinateMax: number;
  rebaseMax?: number;
};

function quant(xs: Float64Array, q: number): number {
  const s = Float64Array.from(xs).sort();
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
}

export function analyzeParity(
  specs: ProbeSpec[],
  blobs: BlobRecord[],
  variant: string,
  mod: TerrainMathModule,
): ParityStats[] {
  const byId = new Map<string, Float32Array>();
  for (const b of blobs) {
    if (b.query.kind === 'probe' && b.query.variant === variant) {
      byId.set(b.query.id, new Float32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4));
    }
  }
  const out: ParityStats[] = [];
  for (const s of specs) {
    const buf = byId.get(s.id);
    if (!buf) continue;
    const n = s.w * s.h;
    const total = new Float64Array(n);
    let arith = 0, coord = 0, sum = 0;
    for (let j = 0; j < s.h; j++) {
      for (let i = 0; i < s.w; i++) {
        const k = j * s.w + i;
        const xTrue = s.origin[0] + s.local0[0] + i * s.step;
        const zTrue = s.origin[1] + s.local0[1] + j * s.step;
        const eGpu = buf[k * 4];
        const xg = buf[k * 4 + 1], zg = buf[k * 4 + 2];
        const eTrue = mod.elevation(xTrue, zTrue, DEFAULT_SEED, 0);
        const eAtG = mod.elevation(xg, zg, DEFAULT_SEED, 0);
        total[k] = Math.abs(eGpu - eTrue);
        sum += total[k];
        arith = Math.max(arith, Math.abs(eGpu - eAtG));
        coord = Math.max(coord, Math.abs(eAtG - eTrue));
      }
    }
    const dist = Math.hypot(s.origin[0] + s.local0[0], s.origin[1] + s.local0[1]);
    out.push({
      id: s.id,
      distance: dist,
      points: n,
      totalMax: quant(total, 1),
      totalP99: quant(total, 0.99),
      totalMean: sum / n,
      arithmeticMax: arith,
      coordinateMax: coord,
    });
  }
  // rebase seams: same world points from adjacent origins
  for (const k of SEAMS) {
    const L = byId.get(`seam${k}-left`), R = byId.get(`seam${k}-right`);
    const st = out.find((o) => o.id === `seam${k}-left`);
    if (!L || !R || !st) continue;
    let m = 0;
    for (let i = 0; i < L.length; i += 4) m = Math.max(m, Math.abs(L[i] - R[i]));
    st.rebaseMax = m;
  }
  return out;
}
