/**
 * Types and uniform packing shared by the WebGPU terrain renderers.
 */

import { CHUNK_SIZE } from '../constants';
import { waterLevel } from '../terrainMath';
import type { ClipmapSpec } from './clipmap';
import type { PassTimestamps } from './clipmapCache';
import type { GrassSpec } from './grass.wgsl';
import type { ForestSpec } from './forest.wgsl';

/** Parts of the sky realm (skyRealm.ts) a bench may leave out (all drawn by default) to measure what each costs. */
export type RealmSpec = {
  shadows?: boolean; water?: boolean; falls?: boolean; rings?: boolean; arches?: boolean;
  /** Bench only: the tops' ground a constant colour (skyRealm.wgsl.ts STUB_RLGROUND), to measure its shading. */
  stubGround?: boolean;
  /** Bench only: false leaves out the islands beyond REALM_NEAR (skyRealm.ts packRealm), to measure them. */
  far?: boolean;
  /** False leaves out the Eyries' structures (structures.ts), to measure them. */
  structures?: boolean;
  /** Bench only: build every Eyrie in range at once (not a slice a frame), so a still shows them whole. */
  syncBuild?: boolean;
};

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
  /** Other players' heroes (multiplayer), the same layout; at most MAX_REMOTE_HEROES are drawn. */
  remotes?: readonly Float32Array[] | null;
  /**
   * With TAA, the scene renders at width × renderScale (the output keeps
   * width × height and is upscaled into). Ignored without TAA. Default 1.
   */
  renderScale?: number;
  /** The day–night sky's clock: game days since day 0, 00:00 (the hour is its fraction). Ignored by the classic sky. */
  days?: number;
};

/** Game time a frame without a clock is drawn at: day 0, 15:30. */
export const DEFAULT_DAYS = 15.5 / 24;

/** Other players' heroes drawn per frame (a world holds five players: src/lib/hero/characters.ts). */
export const MAX_REMOTE_HEROES = 4;

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
      grass?: GrassSpec | null; forest?: ForestSpec | null; ruins?: boolean;
      /** The sky realm (skyRealm.ts): true, or with parts left out to measure their cost. */
      islands?: boolean | RealmSpec;
      /**
       * Per-cell diagonals from the heights (keeps diagonal ridge crests whole), else one fixed
       * diagonal; 'always' re-triangulates every ring every frame (to measure that cost).
       */
      diag?: boolean | 'always';
      /** Anti-aliasing: 4× MSAA, or temporal (TAA, with upscaling by renderScale). */
      msaa?: 1 | 4; taa?: boolean; renderScale?: number;
      /** TAA tuning: presented sharpening (0..1, default 0.5), history blend weight of a new frame (default 0.1). */
      taaSharpen?: number; taaAlpha?: number;
      /** The classic fixed sky (default), or the day–night cycle (sky.wgsl.ts; FrameParams.days sets the hour). */
      sky?: 'classic' | 'cycle';
      /** Bench only: the shaders' lookup tables as `var<private>` (as before), not `const`, to compare. */
      privateTables?: boolean;
      /** Bench only: the TAA's colour arithmetic in f32 (as before), not f16, to compare. */
      taaF32?: boolean;
      /** Bench only: the surfaces' colour arithmetic in f32 (as before), not f16, to compare. */
      shadeF32?: boolean;
      /** Bench only: the hero drawn after the trees and grass (as before), not first, to compare. */
      heroLate?: boolean;
      /** Bench only: STUB_* constants switched on for every pipeline (their defaults rewritten), to measure them. */
      stubAll?: Array<'STUB_DISPLAY' | 'STUB_FOGALL' | 'NO_BLADE_CACHE'>;
      /** Bench only: draws left out of the main pass, to measure them. */
      skip?: Array<'terrain' | 'water' | 'sky' | 'hero' | 'forest' | 'grass' | 'realm' | 'ruins' | 'structures' | 'fauna'>;
      /** Bench only: parts of the terrain's shading left out (terrain.wgsl.ts STUB_* constants), to measure them. */
      stub?: Partial<Record<'STUB_BIOME' | 'STUB_GRAD' | 'STUB_SHADOW' | 'STUB_FOG' | 'STUB_SHADE' | 'PER_PIXEL_GROUND' | 'CACHED_GRAD' | 'CANOPY_MODE', number>>;
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
  // The water's surface under the camera (a lake, a river's pool or the sea): below it the view is underwater.
  f[25] = waterLevel(fp.x, fp.z, fp.seed); f[26] = jx; f[27] = jy;
}

