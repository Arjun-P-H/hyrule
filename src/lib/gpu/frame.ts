/**
 * Types and uniform packing shared by the WebGPU terrain renderers.
 */

import { CHUNK_SIZE } from '../constants';
import type { ClipmapSpec } from './clipmap';
import type { PassTimestamps } from './clipmapCache';
import type { GrassSpec } from './grass.wgsl';
import type { ForestSpec } from './forest.wgsl';

export type FrameParams = {
  /** Camera, world metres (fp64). */
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  width: number;
  height: number;
  tanHalf: number;
  seed: number;
  marchSteps: number;
  shadowSteps: number;
  time: number;
  /** Third-person hero, rig.ts HERO_UNIFORM layout relative to this camera; null or absent: no hero. */
  hero?: Float32Array | null;
  /**
   * With TAA, the scene renders at width × renderScale (the output keeps
   * width × height and is upscaled into). Ignored without TAA. Default 1.
   */
  renderScale?: number;
};

type CacheOptions = {
  clip: ClipmapSpec;
  /** E is prefiltered for a footprint of genDpScale × lattice spacing. */
  genDpScale: number;
  /** log2-distance width of the level blend band in heightAt(). */
  blend: number;
};

export type TerrainVariant =
  | { height: 'direct' }
  | ({ height: 'cached' } & CacheOptions)
  | ({
      height: 'raster'; k: number; stride: number; morphStart: number;
      grass?: GrassSpec | null; forest?: ForestSpec | null; islands?: boolean; ruins?: boolean;
      /**
       * Per-cell diagonals from the heights (keeps diagonal ridge crests whole), else one fixed
       * diagonal; 'always' re-triangulates every ring every frame (to measure that cost).
       */
      diag?: boolean | 'always';
      /** Anti-aliasing: 4× MSAA, or temporal (TAA, with upscaling by renderScale). */
      msaa?: 1 | 4; taa?: boolean; renderScale?: number;
      /** TAA tuning: presented sharpening (0..1, default 0.5), history blend weight of a new frame (default 0.1). */
      taaSharpen?: number; taaAlpha?: number;
    } & CacheOptions);

export interface TerrainRendererLike {
  readonly variant: TerrainVariant;
  compileMs: number;
  init(): Promise<void>;
  setFrame(fp: FrameParams): void;
  pendingTexels(): number;
  invalidate(): void;
  encodeGenerate(enc: GPUCommandEncoder, ts?: PassTimestamps): number;
  encodeRender(enc: GPUCommandEncoder, target: GPUTextureView, ts?: PassTimestamps): void;
  destroy(): void;
}

/** Chunk origin and chunk-local camera for a world position (the engine's floating origin). */
export function floatingOrigin(x: number, z: number): { ox: number; oz: number; lx: number; lz: number } {
  const ox = Math.floor(x / CHUNK_SIZE) * CHUNK_SIZE;
  const oz = Math.floor(z / CHUNK_SIZE) * CHUNK_SIZE;
  return { ox, oz, lx: x - ox, lz: z - oz };
}

/**
 * Pack the Frame uniform block (terrain.wgsl.ts FRAME_WGSL) through two views
 * of one buffer, for a render target of w × h (default fp.width × fp.height)
 * and a projection jitter in NDC (default none).
 */
export function packFrame(fp: FrameParams, f: Float32Array, i: Int32Array, w = fp.width, h = fp.height, jx = 0, jy = 0): void {
  const ox = Math.floor(fp.x / CHUNK_SIZE) * CHUNK_SIZE;
  const oz = Math.floor(fp.z / CHUNK_SIZE) * CHUNK_SIZE;
  const lx = fp.x - ox, lz = fp.z - oz;
  // Basis as in useTerrainEngine / page.js: columns right, up, forward.
  const cy = Math.cos(fp.yaw), sy = Math.sin(fp.yaw);
  const cp = Math.cos(fp.pitch), sp = Math.sin(fp.pitch);
  const fx = sy * cp, fy = sp, fz = cy * cp, rx = cy, rz = -sy;
  f[0] = rx; f[1] = 0; f[2] = rz; f[3] = 0;
  f[4] = fy * rz; f[5] = fz * rx - fx * rz; f[6] = -fy * rx; f[7] = 0;
  f[8] = fx; f[9] = fy; f[10] = fz; f[11] = 0;
  f[12] = lx; f[13] = fp.y; f[14] = lz; f[15] = fp.time;
  f[16] = ox; f[17] = oz; f[18] = w; f[19] = h;
  f[20] = fp.tanHalf; f[21] = (2 * fp.tanHalf) / Math.max(h, 2);
  i[22] = fp.seed; i[23] = fp.marchSteps; i[24] = fp.shadowSteps;
  f[25] = 0; f[26] = jx; f[27] = jy;
}

