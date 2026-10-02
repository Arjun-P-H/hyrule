/**
 * WebGPU renderer experiments (headless Chrome: Dawn/Metal next to ANGLE/Metal).
 * ---------------------------------------------------------------------------
 *   bun bench/webgpu.ts port       WGSL port vs the WebGL2 renderer: frames diffed
 *                                  pixel by pixel, elevation probes vs the GLSL
 *                                  probe and the fp64 CPU twin
 *   bun bench/webgpu.ts cost       GPU ms per frame along the benchmark camera path:
 *                                  WGSL direct, WGSL clipmap-cached variants, and
 *                                  the WebGL2 renderer at the same resolution
 *   bun bench/webgpu.ts gen        clipmap generation: full rebuilds and
 *                                  incremental updates at walk … fast-fly speeds
 *   bun bench/webgpu.ts fidelity   every variant against the converged WebGL2
 *                                  ground truth (transformShader 'truth')
 *   bun bench/webgpu.ts horizons   the day–night sky's terrain horizons (the
 *                                  clipmap's sun layer read back) against a
 *                                  brute-force march of the CPU twin
 *   bun bench/webgpu.ts shimmer    temporal stability of chosen views: a slow
 *                                  pan, flicker per variant (say, the trees'
 *                                  surface detail against plain trees)
 *
 * Options: --res=1280x800 --reps=2 --frames=all|0,6,12 --variants=a,b --warmup=1 (a copy of the
 *          first variant runs first and is left out of the ratios: see pickVariants)
 *          --hours=6,12,18.2 (stills of day–night variants at those hours; --day=n the game day)
 *          --hour=18.2 (timing of day–night variants at that hour)
 * Results: bench/out/webgpu-<mode>.json, frames in bench/out/webgpu/.
 */

import path from 'node:path';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { runSession, transformShader, type SessionConfig, type SessionResult, type Variant } from './gpu/runner';
import { collectShots, SHOT_FRAMES } from './shots';
import { encodePNG, perceptualDiff, type RGBAImage } from './lib/image';
import { loadShader, WORKTREE, loadTerrainMath } from './lib/revision';
import { fmt, pairedRatio, summarize } from './lib/stats';
import { analyzeParity, probeSpecs } from './parity';
import type { CameraFrame } from './spline';
import { CAMERA, CHUNK_SIZE, DEFAULT_SEED, type QualityPreset } from '../src/lib/constants';
import type { TerrainVariant } from '../src/lib/gpu/terrainRenderer';
import { GAME_VARIANT } from '../src/lib/gpu/engineGpu';

const GAME = GAME_VARIANT as Extract<TerrainVariant, { height: 'raster' }>;
import { plantInCell, type PlaceWorld } from '../src/lib/forest/placement';
import { SPECIES } from '../src/lib/forest/species';
import { ruinSiteAt } from '../src/lib/ruins';
import { skyPlantInCell } from '../src/lib/skyworld/flora';
import { elevation } from '../src/lib/terrainMath';
import { PATH_N, SKY } from '../src/lib/sky/celestial';

const ROOT = path.resolve(import.meta.dir, '..');
const OUT = path.join(ROOT, 'bench/out');
const args = process.argv.slice(2);
const mode = args[0] ?? 'port';
const opt = (k: string, d: string) => args.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;

const FRAMES: CameraFrame[] = JSON.parse(readFileSync(path.join(ROOT, 'bench/baseline/camera_path.json'), 'utf8'));
const [W, H] = opt('res', '1280x800').split('x').map(Number);
const MARCH_STEPS = Number(opt('march', '128'));
const SHADOW_STEPS = Number(opt('shadow', '28'));
const REPS = Number(opt('reps', '2'));

const VARIANTS: Record<string, TerrainVariant> = {
  // The game's renderer as it ships (engineGpu.ts), at the Medium and High presets' internal scales
  // (QUALITY_PRESETS resScale; the output stays at --res, as the canvas does).
  'game-med': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5 } as TerrainVariant,
  'game-high': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.75 } as TerrainVariant,
  // The far forest against the trees it stands for (terrain.wgsl.ts CANOPY_MODE): the trees alone, no
  // canopy painted; the canopy painted at every distance and no trees; the canopy's cover and the
  // distance as raw colours (which pixels to compare).
  'canopy-trees': { ...GAME, sky: 'cycle', renderScale: 0.75, stub: { CANOPY_MODE: 1 } } as TerrainVariant,
  'canopy-paint': { ...GAME, sky: 'cycle', renderScale: 0.75, stub: { CANOPY_MODE: 2 }, skip: ['forest'] } as TerrainVariant,
  'canopy-mask': { ...GAME, sky: 'cycle', renderScale: 0.75, stub: { CANOPY_MODE: 3 }, stubAll: ['STUB_DISPLAY'], skip: ['forest', 'grass', 'ruins'] } as TerrainVariant,
  // The land's reach as before (8 levels: 8 km) against the game's (12: 131 km); without the far
  // forest painted on; without the islands beyond REALM_NEAR.
  'game-med-l8': { ...GAME, sky: 'cycle', renderScale: 0.5, clip: { ...GAME.clip, levels: 8 } } as TerrainVariant,
  'game-high-l8': { ...GAME, sky: 'cycle', renderScale: 0.75, clip: { ...GAME.clip, levels: 8 } } as TerrainVariant,
  'game-med-nocanopy': { ...GAME, sky: 'cycle', renderScale: 0.5, stub: { CANOPY_MODE: 1 } } as TerrainVariant,
  'game-med-nofar': { ...GAME, sky: 'cycle', renderScale: 0.5, islands: { far: false } } as TerrainVariant,
  // Ablations of game-med: what each part of the frame costs is the difference.
  'game-med-nograss': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, grass: null } as TerrainVariant,
  'game-med-noforest': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, forest: null } as TerrainVariant,
  'game-med-noislands': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, islands: false } as TerrainVariant,
  'game-med-noruins': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, ruins: false } as TerrainVariant,
  'game-med-classic': { ...GAME_VARIANT, sky: 'classic', renderScale: 0.5 } as TerrainVariant,
  'game-med-bare': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, grass: null, forest: null, islands: false, ruins: false } as TerrainVariant,
  // Draws of the main pass left out, one at a time (and the terrain's geometry at a quarter of the triangles).
  ...Object.fromEntries((['terrain', 'water', 'sky', 'hero'] as const).map((k) => [`game-med-skip${k}`,
    { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, skip: [k] } as TerrainVariant])),
  'game-med-k64': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, k: 64, stride: 2 } as TerrainVariant,
  // The fixed (resolution-independent) part of the frame: the game at a quarter of the canvas, and its parts left out.
  'game-q': { ...GAME, sky: 'cycle', renderScale: 0.25 } as TerrainVariant,
  'game-q-nograss': { ...GAME, sky: 'cycle', renderScale: 0.25, grass: null } as TerrainVariant,
  'game-q-noforest': { ...GAME, sky: 'cycle', renderScale: 0.25, forest: null } as TerrainVariant,
  'game-q-noislands': { ...GAME, sky: 'cycle', renderScale: 0.25, islands: false } as TerrainVariant,
  'game-q-k64': { ...GAME, sky: 'cycle', renderScale: 0.25, k: 64, stride: 2 } as TerrainVariant,
  'game-q-bare': { ...GAME, sky: 'cycle', renderScale: 0.25, grass: null, forest: null, islands: false, ruins: false } as TerrainVariant,
  ...Object.fromEntries((['grass', 'forest', 'hero', 'sky', 'water', 'realm'] as const).map((k) => [`game-q-skip${k}`,
    { ...GAME, sky: 'cycle', renderScale: 0.25, skip: [k] } as TerrainVariant])),
  // The lookup tables as var<private> (before), against const.
  'game-med-private': { ...GAME, sky: 'cycle', renderScale: 0.5, privateTables: true } as TerrainVariant,
  'game-q-private': { ...GAME, sky: 'cycle', renderScale: 0.25, privateTables: true } as TerrainVariant,
  'game-high-private': { ...GAME, sky: 'cycle', renderScale: 0.75, privateTables: true } as TerrainVariant,
  // Every surface without tone mapping and gamma, without fog.
  'game-high-nodisplay': { ...GAME, sky: 'cycle', renderScale: 0.75, stubAll: ['STUB_DISPLAY'] } as TerrainVariant,
  'game-high-nofogall': { ...GAME, sky: 'cycle', renderScale: 0.75, stubAll: ['STUB_FOGALL'] } as TerrainVariant,
  // The sky's blades made afresh every frame (as before the blade cache).
  'game-high-nocache': { ...GAME, sky: 'cycle', renderScale: 0.75, stubAll: ['NO_BLADE_CACHE'] } as TerrainVariant,
  'game-high-notreeshadow': { ...GAME, sky: 'cycle', renderScale: 0.75, forest: { ...GAME.forest!, shadowRes: 0 } } as TerrainVariant,
  // The surfaces' colour arithmetic in f32 (TAA in f16 as the game), against f16.
  'game-high-shade32': { ...GAME, sky: 'cycle', renderScale: 0.75, shadeF32: true } as TerrainVariant,
  // The TAA's colour arithmetic in f32 (as before) instead of f16.
  'game-med-taa32': { ...GAME, sky: 'cycle', renderScale: 0.5, taaF32: true } as TerrainVariant,
  'game-high-taa32': { ...GAME, sky: 'cycle', renderScale: 0.75, taaF32: true } as TerrainVariant,
  // The sky realm at High: without its draws, with its tops' ground a constant colour.
  'game-high-skiprealm': { ...GAME, sky: 'cycle', renderScale: 0.75, skip: ['realm'] } as TerrainVariant,
  'game-high-rlstub': { ...GAME, sky: 'cycle', renderScale: 0.75, islands: { stubGround: true } } as TerrainVariant,
  // The hero drawn after the trees and grass (as before) instead of first.
  'game-high-herolate': { ...GAME, sky: 'cycle', renderScale: 0.75, heroLate: true } as TerrainVariant,
  'game-med-herolate': { ...GAME, sky: 'cycle', renderScale: 0.5, heroLate: true } as TerrainVariant,
  // High: the terrain's shading part by part (STUB_*), and the draws left out.
  ...Object.fromEntries((['BIOME', 'GRAD', 'SHADOW', 'FOG', 'SHADE'] as const).map((k) => [`game-high-no${k.toLowerCase()}`,
    { ...GAME, sky: 'cycle', renderScale: 0.75, stub: { [`STUB_${k}`]: 1 } } as TerrainVariant])),
  ...Object.fromEntries((['grass', 'forest', 'hero', 'water', 'realm', 'sky'] as const).map((k) => [`game-high-skip${k}`,
    { ...GAME, sky: 'cycle', renderScale: 0.75, skip: [k] } as TerrainVariant])),
  // The sky realm's parts at the Medium preset: without its lakes and streams, falls and mist, rings, arches.
  ...Object.fromEntries((['water', 'falls', 'rings', 'arches'] as const).map((part) => [`game-med-sky-no${part}`,
    { ...GAME, sky: 'cycle', renderScale: 0.5, islands: { [part]: false } } as TerrainVariant])),
  ...Object.fromEntries((['realm', 'grass', 'forest', 'hero', 'terrain'] as const).map((k) => [`game-med-skip${k}`,
    { ...GAME, sky: 'cycle', renderScale: 0.5, skip: [k] } as TerrainVariant])),
  // Without the sky realm's shadow map (its pass and every surface's taps), without the trees' shadow map.
  'game-med-noskyshadow': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, islands: { shadows: false } } as TerrainVariant,
  'game-med-notreeshadow': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, forest: { ...GAME.forest!, shadowRes: 0 } } as TerrainVariant,
  // The ground shaded per pixel as before the clipmap's material layer (terrain.wgsl.ts PER_PIXEL_GROUND).
  'game-med-pixel': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, stub: { PER_PIXEL_GROUND: 1 } } as TerrainVariant,
  // Its normal from the sun layer's cached (smooth) gradient instead of gradientAt (terrain.wgsl.ts CACHED_GRAD).
  'game-med-cgrad': { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, stub: { CACHED_GRAD: 1 } } as TerrainVariant,
  // The terrain's shading, part by part (STUB_* in terrain.wgsl.ts): its biome colours, normals, shadows, fog; none of it.
  ...Object.fromEntries((['BIOME', 'GRAD', 'SHADOW', 'FOG', 'SHADE'] as const).map((k) => [`game-med-no${k.toLowerCase()}`,
    { ...GAME_VARIANT, sky: 'cycle', renderScale: 0.5, stub: { [`STUB_${k}`]: 1 } } as TerrainVariant])),
  direct: { height: 'direct' },
  'clip512': { height: 'cached', clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  'clip512-dp2': { height: 'cached', clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 2, blend: 0.25 },
  'clip1024': { height: 'cached', clip: { n: 1024, levels: 7, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  'clip256': { height: 'cached', clip: { n: 256, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  // Raster geometry clipmaps over the clip512 cache: k cells per half ring, one vertex every `stride` texels.
  'raster128': { height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  'raster64x2': { height: 'raster', k: 64, stride: 2, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  // E5: raster128 plus scattered grass blades (rings × n² candidates, 0.125 m cells in ring 0).
  'grass': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
  },
  'grass-msaa': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 4,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
  },
  'world': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 4,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 },
  },
  'world-sky': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 4,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  // E6: anti-aliasing of the complete world — none, temporal (TAA), temporal with 0.67× upscaling.
  'world-noaa': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 1,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'world-taa': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true, ruins: true, diag: true,
  },
  // The game's world under the day–night sky (the hour from --hours / --hour).
  'world-cycle': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true, ruins: true, diag: true, sky: 'cycle',
  },
  // The day–night world with the trees' surfaces plain (no leaf clusters, needles, leaflets or bark): their cost.
  'world-cycle-plain': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90, detail: false }, islands: true, ruins: true, diag: true, sky: 'cycle',
  },
  // The day–night world without the sky realm (its cost is the difference).
  'world-cycle-nosky': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: false, ruins: true, diag: true, sky: 'cycle',
  },
  // The sky realm's parts (its cost, piece by piece): without its shadow map, its lakes and streams, its falls, the sky rings, the arches.
  ...Object.fromEntries((['shadows', 'water', 'falls', 'rings', 'arches'] as const).map((part) => [`world-cycle-sky-no${part}`, {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: { [part]: false }, ruins: true, diag: true, sky: 'cycle',
  } as TerrainVariant])),
  // The forest's parts: without shadows, without the level-of-detail dither.
  'world-taa-noshadow': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 0, shadowRange: 90 }, islands: true, ruins: true, diag: true,
  },
  'world-taa-nofade': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90, fade: false }, islands: true, ruins: true, diag: true,
  },
  // The game's world without its trees (the forest's cost is the difference).
  'world-taa-bare': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    islands: true, ruins: true, diag: true,
  },
  // Far field: the same world on a 1024² cache (each level reaches twice as far, so a point is shaded one level finer).
  'world-n1024': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 1024, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true, ruins: true, diag: true,
  },
  'raster128-diag': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, diag: true,
  },
  'raster128-diag-always': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, diag: 'always',
  },
  'world-taa-soft': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true, taaSharpen: 0,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'world-taa-a1': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true, taaSharpen: 0, taaAlpha: 1,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'world-taau': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true, renderScale: 0.67,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'grass-dense': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25,
    grass: { rings: 4, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1000 },
  },
};

/**
 * The variants named by --variants. With --warmup=1 a copy of the first one, named 'warm-up',
 * runs ahead of them: on some views whichever variant renders a view first pays for arriving
 * there (measured: ×1.01 in one order, ×1.14 in the other), and the copy absorbs that.
 */
function pickVariants(def: string): Array<{ name: string; variant: TerrainVariant }> {
  const picked = opt('variants', def).split(',').map((name) => {
    const variant = VARIANTS[name];
    if (!variant) throw new Error(`unknown variant ${name}; known: ${Object.keys(VARIANTS).join(', ')}`);
    return { name, variant };
  });
  return opt('warmup', '') ? [{ name: 'warm-up', variant: picked[0].variant }, ...picked] : picked;
}

function frameList(def: number[]): number[] {
  const f = opt('frames', '');
  if (f === 'all') return FRAMES.map((_, i) => i);
  if (f) return f.split(',').map(Number);
  return def;
}

let pageJs: string | null = null;
async function webgpuPage(): Promise<string> {
  if (pageJs) return pageJs;
  const build = await Bun.build({ entrypoints: [path.join(import.meta.dir, 'gpu/webgpu-page.ts')], target: 'browser' });
  if (!build.success) throw new Error(build.logs.map(String).join('\n'));
  pageJs = await build.outputs[0].text();
  return pageJs;
}

const CHROME_WEBGPU = ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'];

type GpuPreset = { id: string; w: number; h: number; marchSteps: number; shadowSteps: number };
const PRESET: GpuPreset = { id: `${W}x${H}`, w: W, h: H, marchSteps: MARCH_STEPS, shadowSteps: SHADOW_STEPS };

async function runWebGPU(cfg: Record<string, unknown>): Promise<SessionResult> {
  const all: Record<string, unknown> = {
    variants: [], presets: [PRESET], frames: FRAMES, seed: DEFAULT_SEED, fov: CAMERA.fov, reps: REPS,
    warmupMs: 2000, sentinelEvery: 10, batch: 8, repeat: Number(opt('repeat', '3')), shotFrames: [], probes: [],
    gen: { starts: [0, 12, 24, 35, 47], rebuilds: 3, cases: [] },
    stallMs: 180_000, timeoutMs: 40 * 60_000,
    ...cfg,
  };
  return runSession(all, { pageJs: await webgpuPage(), chromeArgs: CHROME_WEBGPU });
}

function webglBase(modes: SessionConfig['modes'], variants: Variant[], extra: Partial<SessionConfig> = {}): SessionConfig {
  const preset: QualityPreset = { id: 'high', label: PRESET.id, resScale: 1, marchSteps: MARCH_STEPS, shadowSteps: SHADOW_STEPS, note: '' };
  return {
    modes, variants, presets: [preset], frames: FRAMES, seed: DEFAULT_SEED, fov: CAMERA.fov, chunkSize: CHUNK_SIZE,
    baseW: W, baseH: H, reps: REPS, warmupMs: 2000, depth: 1, calibrateDraws: 8, calibrateFrames: [0],
    sentinelEvery: 10, shotFrames: [], instrumentFrames: [0], probes: [],
    evalRes: 256, evalK: 16, evalDps: [0.01], evalBases: [[0, 0]], evalReps: 1,
    stallMs: 180_000, timeoutMs: 40 * 60_000,
    ...extra,
  };
}

/** WebGL readPixels rows run bottom-up; WebGPU copies run top-down. */
function flipRows(img: RGBAImage): RGBAImage {
  const { width: w, height: h, data } = img;
  const out = new Uint8Array(data.length);
  for (let y = 0; y < h; y++) out.set(data.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
  return { width: w, height: h, data: out };
}

function saveShots(res: SessionResult, dir: string, flip: boolean): Map<string, RGBAImage> {
  mkdirSync(dir, { recursive: true });
  const { shots } = collectShots(res.blobs, null);
  const out = new Map<string, RGBAImage>();
  for (const [key, img0] of shots) {
    const img = flip ? flipRows(img0) : img0;
    const [variant, , frame] = key.split('|');
    out.set(`${variant}|${frame}`, img);
    Bun.write(path.join(dir, `${variant.replace(/[^\w.-]/g, '_')}_f${frame.padStart(2, '0')}.png`), encodePNG(img));
  }
  return out;
}

function diffTable(a: Map<string, RGBAImage>, aName: string, b: Map<string, RGBAImage>, bName: string, frames: number[]) {
  const rows: Array<{ frame: number } & ReturnType<typeof perceptualDiff>> = [];
  for (const f of frames) {
    const A = a.get(`${aName}|${f}`), B = b.get(`${bName}|${f}`);
    if (!A || !B) continue;
    rows.push({ frame: f, ...perceptualDiff(A, B) });
  }
  const mean = (k: 'identical' | 'meanDE' | 'fracJND' | 'fracVisible') => rows.reduce((s, r) => s + r[k], 0) / Math.max(rows.length, 1);
  return {
    a: aName, b: bName, frames: rows.length,
    identical: mean('identical'), meanDE: mean('meanDE'), fracJND: mean('fracJND'), fracVisible: mean('fracVisible'),
    worstFrameJND: Math.max(0, ...rows.map((r) => r.fracJND)), p99DE: Math.max(0, ...rows.map((r) => r.p99DE)),
    perFrame: rows.map((r) => ({ frame: r.frame, identical: r.identical, meanDE: r.meanDE, fracJND: r.fracJND, p99DE: r.p99DE })),
  };
}

function printDiff(d: ReturnType<typeof diffTable>): void {
  console.log(`  ${d.b.padEnd(14)} vs ${d.a.padEnd(14)} frames=${d.frames} identical=${(d.identical * 100).toFixed(2)}% ` +
    `meanΔE=${d.meanDE.toFixed(4)} >JND=${(d.fracJND * 100).toFixed(2)}% (worst frame ${(d.worstFrameJND * 100).toFixed(2)}%) ` +
    `>0.05=${(d.fracVisible * 100).toFixed(2)}% p99ΔE=${d.p99DE.toFixed(4)}`);
}

// ---------------------------------------------------------------------------
async function port(): Promise<void> {
  const frames = frameList(SHOT_FRAMES);
  const probes = probeSpecs();
  console.log(`WGSL port check at ${W}×${H}, frames ${frames.join(',')}`);
  const gpu = await runWebGPU({ modes: ['shots', 'probe'], variants: pickVariants('direct'), shotFrames: frames, probes });
  console.log('  webgpu env:', JSON.stringify(gpu.env), 'compile:', JSON.stringify(gpu.compile));
  const src = await loadShader(WORKTREE);
  const gl = await runSession(webglBase(['shots'], [{ name: 'webgl', vs: src.vs, fs: src.fs }], { shotFrames: frames }));
  const glProbe = await runSession(webglBase(['probe'], [{ name: 'glsl', vs: src.vs, fs: transformShader(src.fs, 'probe') }], { probes }));
  const dir = path.join(OUT, 'webgpu', 'port');
  const a = saveShots(gl, dir, true);
  const b = saveShots(gpu, dir, false);
  const d = diffTable(a, 'webgl', b, 'direct', frames);
  printDiff(d);

  const { mod } = await loadTerrainMath(WORKTREE);
  const wgsl = analyzeParity(probes, gpu.blobs, 'wgsl', mod);
  const glsl = analyzeParity(probes, glProbe.blobs, 'glsl', mod);
  // WGSL vs GLSL directly
  const byId = (res: SessionResult, v: string) => new Map(res.blobs.filter((x) => x.query.kind === 'probe' && x.query.variant === v)
    .map((x) => [x.query.id, new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4)] as const));
  const W1 = byId(gpu, 'wgsl'), G1 = byId(glProbe, 'glsl');
  console.log('  probe                    |E_wgsl−E_cpu| max   |E_glsl−E_cpu| max   |E_wgsl−E_glsl| max   identical');
  const probeRows = [];
  for (const s of probes) {
    const a1 = W1.get(s.id), b1 = G1.get(s.id);
    if (!a1 || !b1) continue;
    let m = 0, same = 0;
    for (let i = 0; i < a1.length; i += 4) {
      m = Math.max(m, Math.abs(a1[i] - b1[i]));
      if (a1[i] === b1[i]) same++;
    }
    const w = wgsl.find((x) => x.id === s.id)!, g = glsl.find((x) => x.id === s.id)!;
    probeRows.push({ id: s.id, wgslCpuMax: w.totalMax, glslCpuMax: g.totalMax, wgslGlslMax: m, identical: same / (a1.length / 4) });
    console.log(`  ${s.id.padEnd(24)} ${(w.totalMax * 1000).toFixed(3).padStart(10)} mm ${(g.totalMax * 1000).toFixed(3).padStart(14)} mm ` +
      `${(m * 1000).toFixed(3).padStart(15)} mm ${(same / (a1.length / 4) * 100).toFixed(1).padStart(10)}%`);
  }
  writeFileSync(path.join(OUT, 'webgpu-port.json'), JSON.stringify({ res: [W, H], env: gpu.env, compile: gpu.compile, image: d, probes: probeRows }, null, 1));
}

type TSample = { v: number; f: number; r: number; s: number; ms: number; first: number; rawRender: number; gen: number | null; texels: number; passes?: Record<string, number>; raw?: Record<string, number> };

/** Per variant, each pass's median cost over the samples (ms), in the order the passes run. */
function passTable(t: TSample[], variants: Array<{ name: string }>): void {
  const med = (xs: number[]) => { const q = [...xs].sort((a, b) => a - b); return q[Math.floor(q.length / 2)]; };
  for (const [vi, v] of variants.entries()) {
    const own = t.filter((x) => x.v === vi && x.passes);
    const labels: string[] = [];
    for (const x of own) for (const k of Object.keys(x.passes!)) if (!labels.includes(k)) labels.push(k);
    const cells = labels.map((k) => `${k} ${fmt(med(own.map((x) => x.passes![k] ?? 0)), 2)}`);
    console.log(`    ${v.name} passes (ms, median gap from the pass before): ${cells.join(' · ')}`);
    const raw = labels.map((k) => `${k} ${fmt(med(own.map((x) => x.raw?.[k] ?? 0)), 2)}`);
    console.log(`    ${v.name} passes (ms, median begin → end): ${raw.join(' · ')}`);
  }
}

async function cost(): Promise<void> {
  const variants = pickVariants('direct,clip512,clip512-dp2');
  const frames = frameList(FRAMES.map((_, i) => i));
  const sel = frames.map((i) => FRAMES[i]);
  console.log(`GPU cost at ${W}×${H} (march ${MARCH_STEPS}, shadow ${SHADOW_STEPS}), ${sel.length} frames × ${REPS} reps: ${variants.map((v) => v.name).join(', ')}`);
  const gpu = await runWebGPU({ modes: ['timing'], variants, frames: sel });
  console.log('  compile:', gpu.compile.map((c) => `${c.name} ${c.compileMs.toFixed(0)} ms`).join(', '));
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((s) => !s.s);
  const perV = variants.map((_, vi) => {
    const byF = new Map<number, number[]>();
    for (const s of t) if (s.v === vi) byF.set(s.f, [...(byF.get(s.f) ?? []), s.ms]);
    return [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  });
  const gens = variants.map((_, vi) => t.filter((s) => s.v === vi && s.gen !== null).map((s) => s.gen as number));
  let webgl: number[] | null = null;
  if (opt('webgl', '1') === '1') {
    const src = await loadShader(WORKTREE);
    const gl = await runSession(webglBase(['timing'], [{ name: 'webgl', vs: src.vs, fs: src.fs }], { frames: sel }));
    const gs = (gl.results.timing as Array<{ samples: Array<{ v: number; f: number; s: number; ms: number }> }>)[0].samples.filter((s) => !s.s);
    const byF = new Map<number, number[]>();
    for (const s of gs) byF.set(s.f, [...(byF.get(s.f) ?? []), s.ms]);
    webgl = [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  }
  const rows: Record<string, unknown>[] = [];
  const line = (name: string, xs: number[], ref: number[] | null) => {
    const s = summarize(xs);
    const r = ref ? pairedRatio(ref, xs) : null;
    console.log(`  ${name.padEnd(14)} p50 ${fmt(s.p50, 2).padStart(8)} ms  p95 ${fmt(s.p95, 2).padStart(8)} ms  max ${fmt(s.max, 2).padStart(8)} ms` +
      (r ? `   × ${fmt(r.ratio, 4)} [${fmt(r.lo, 4)}, ${fmt(r.hi, 4)}] vs direct` : ''));
    rows.push({ name, ...s, ratioVsDirect: r });
  };
  if (webgl) line('webgl', webgl, perV[0]);
  variants.forEach((v, vi) => line(v.name, perV[vi], vi === 0 ? null : perV[0]));
  writeFileSync(path.join(OUT, 'webgpu-cost.json'), JSON.stringify({
    res: [W, H], march: MARCH_STEPS, shadow: SHADOW_STEPS, frames, env: gpu.env, compile: gpu.compile,
    variants: variants.map((v) => v.name), perFrameMin: perV, webgl, rows, gen: gens,
  }, null, 1));
}

async function gen(): Promise<void> {
  const variants = pickVariants('clip512,clip1024');
  const cases = [
    { name: 'walk', speed: 5.4, frames: 60, heading: 0.7 },
    { name: 'sprint', speed: 8.6, frames: 60, heading: 0.7 },
    { name: 'fly', speed: 40, frames: 60, heading: 0.7 },
    { name: 'fast-fly', speed: 200, frames: 60, heading: 0.7 },
  ];
  console.log(`clipmap generation (${variants.map((v) => v.name).join(', ')}), frames rendered at ${W}×${H}`);
  const res = await runWebGPU({ modes: ['gen'], variants, gen: { starts: [0, 12, 24, 35, 47], rebuilds: 3, cases } });
  type G = {
    variant: string;
    rebuild: Array<{ frameMs: number; steadyMs: number; rawGenMs: number; texels: number }>;
    moves: Array<{ case: string; speed: number; still: number[]; moving: number[]; rawGen: number[]; texels: number[] }>;
  };
  const out = res.results.gen as G[];
  for (const g of out) {
    const added = summarize(g.rebuild.map((r) => r.frameMs - r.steadyMs));
    const raw = summarize(g.rebuild.map((r) => r.rawGenMs));
    const tex = g.rebuild[0]?.texels ?? 0;
    console.log(`  ${g.variant}: full rebuild of ${(tex / 1e6).toFixed(2)} M texels adds p50 ${fmt(added.p50, 2)} ms (max ${fmt(added.max, 2)}) to its frame; ` +
      `generator pass p50 ${fmt(raw.p50, 2)} ms → ${fmt(tex / raw.p50 / 1e3, 0)} M texels/s`);
    for (const c of cases) {
      const ms = g.moves.filter((m) => m.case === c.name);
      const moving = ms.flatMap((m) => m.moving), still = ms.flatMap((m) => m.still);
      const raw2 = summarize(ms.flatMap((m) => m.rawGen.filter((x) => x > 0)));
      const sm = summarize(moving), ss = summarize(still), st = summarize(ms.flatMap((m) => m.texels));
      console.log(`    ${c.name.padEnd(9)} ${String(c.speed).padStart(4)} m/s  frame p50 ${fmt(sm.p50, 2)} ms (still ${fmt(ss.p50, 2)}) p95 ${fmt(sm.p95, 2)} (still ${fmt(ss.p95, 2)})` +
        `  generator pass p50 ${fmt(raw2.p50, 3)} ms max ${fmt(raw2.max, 3)}  texels/frame p50 ${fmt(st.p50, 0)} max ${fmt(st.max, 0)}`);
    }
  }
  writeFileSync(path.join(OUT, 'webgpu-gen.json'), JSON.stringify({ res: [W, H], env: res.env, results: out }, null, 1));
}

/** Still shots of gallery frames for the listed variants (TAA converged over --converge frames). */
async function stills(): Promise<void> {
  const config = opt('config', 'gallery');
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const step = Number(opt('every', '1'));
  const shotFrames = opt('frames', '1,3') === 'all' ? framed.map((_, i) => i).filter((i) => i % step === 0) : opt('frames', '1,3').split(',').map(Number);
  const variants = pickVariants('world-noaa,world-taa-a1,world-taa-soft');
  const timed = !!opt('timing', '');
  // --noshots=1: timing only (over the frames picked), no images
  const noShots = !!opt('noshots', '');
  const hours = opt('hours', '').split(',').filter(Boolean).map(Number);
  const gpu = await runWebGPU({
    modes: noShots ? ['timing'] : timed ? ['shots', 'timing'] : ['shots'], variants,
    frames: noShots ? shotFrames.map((i) => framed[i]) : framed, shotFrames: noShots ? [] : shotFrames, converge: Number(opt('converge', '32')),
    hours, day: Number(opt('day', '0')), hour: Number(opt('hour', '15.5')),
    ...(opt('time', '') ? { time: Number(opt('time', '0')) } : {}),
  });
  if (opt('env', '')) console.log('  env:', JSON.stringify(gpu.env));
  if (!noShots) {
    saveShots(gpu, path.join(OUT, 'webgpu', config === 'gallery' ? 'fidelity' : `stills-${config}`), false);
    console.log('stills saved', variants.map((v) => v.name).join(', '));
  }
  const shotsInfo = (gpu.results.shots ?? []) as Array<{ variant: string; frame: number; trees?: [number, number]; sky?: { light: number[]; exposure: number; hour: number } | null }>;
  const skies = shotsInfo.filter((x) => x.sky);
  if (skies.length) {
    writeFileSync(path.join(OUT, `webgpu-sky-${config}.json`), JSON.stringify(skies.map((x) => ({ variant: x.variant, frame: x.frame, ...x.sky })), null, 1));
    const v = (l: number[], i: number) => `(${[0, 1, 2].map((c) => l[4 * i + c].toFixed(3)).join(',')})`;
    for (const x of skies) {
      const l = x.sky!.light;
      console.log(`  sky ${x.variant} f${x.frame}: exposure ${x.sky!.exposure.toFixed(2)} ambUp ${v(l, 0)} sun ${v(l, 1)} anti ${v(l, 2)} down ${v(l, 3)} bounce ${v(l, 4)} hor ${v(l, 5)} hor·cos ${v(l, 6)} hor·cos2 ${v(l, 7)} zenith ${v(l, 8)} grade ${v(l, 9)}`);
    }
  }
  const bl = (shotsInfo as Array<{ variant: string; frame: number; blades?: number }>).filter((x) => x.blades);
  if (bl.length) console.log('  grass blades per frame:', bl.map((x) => `${x.variant} f${x.frame}: ${x.blades}`).join(', '));
  const tr = shotsInfo.filter((x) => x.trees && (x.trees[0] || x.trees[1]));
  if (tr.length) console.log('  trees (near/far) per frame:', tr.map((x) => `${x.variant} f${x.frame}: ${x.trees![0]}/${x.trees![1]}`).join(', '));
  if (!timed) return;
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((x) => !x.s);
  // Per frame, the fastest rep of each variant; ratios are paired by frame against the first variant.
  const perFrame = variants.map((_, vi) => {
    const byF = new Map<number, number[]>();
    for (const x of t) if (x.v === vi) byF.set(x.f, [...(byF.get(x.f) ?? []), x.ms]);
    return [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  });
  // Ratios against the first variant measured (not the warm-up).
  const b0 = variants[0].name === 'warm-up' ? 1 : 0;
  for (const [vi, v] of variants.entries()) {
    const s = summarize(t.filter((x) => x.v === vi).map((x) => x.ms));
    const r = vi > b0 ? pairedRatio(perFrame[b0], perFrame[vi]) : null;
    console.log(`  ${v.name}: p50 ${fmt(s.p50, 2)} ms  p95 ${fmt(s.p95, 2)} ms` + (r ? `   × ${fmt(r.ratio, 3)} [${fmt(r.lo, 3)}, ${fmt(r.hi, 3)}] vs ${variants[b0].name}` : ''));
  }
  if (opt('perframe', '')) {
    console.log('  per frame (fastest rep, ms): ' + perFrame[0].map((_, f) => `f${f}: ${perFrame.map((pv) => fmt(pv[f], 2)).join(' / ')}`).join('  '));
  }
  passTable(t, variants);
}

/** The TAA passes alone (webgpu-page.ts taa mode): resolve and present per variant, over the gallery's views. */
async function taa(): Promise<void> {
  const config = opt('config', 'gallery');
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const variants = pickVariants('game-med');
  const gpu = await runWebGPU({ modes: ['taa'], variants, frames: framed, repeat: Number(opt('repeat', '8')) });
  const t = (gpu.results.taa as Array<{ samples: Array<{ v: number; f: number; ms: number; resolve: number; present: number }> }>)[0].samples;
  const b0 = variants[0].name === 'warm-up' ? 1 : 0;
  const per = (vi: number, k: 'ms' | 'resolve' | 'present') => {
    const byF = new Map<number, number[]>();
    for (const x of t) if (x.v === vi) byF.set(x.f, [...(byF.get(x.f) ?? []), x[k]]);
    return [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  };
  for (const [vi, v] of variants.entries()) {
    const s = (k: 'ms' | 'resolve' | 'present') => summarize(per(vi, k)).p50;
    const r = vi > b0 ? pairedRatio(per(b0, 'ms'), per(vi, 'ms')) : null;
    console.log(`  ${v.name}: TAA ${fmt(s('ms'), 3)} ms (resolve ${fmt(s('resolve'), 3)}, present ${fmt(s('present'), 3)})` +
      (r ? `   × ${fmt(r.ratio, 3)} [${fmt(r.lo, 3)}, ${fmt(r.hi, 3)}] vs ${variants[b0].name}` : ''));
  }
}

async function fidelity(): Promise<void> {
  const frames = frameList(SHOT_FRAMES);
  const variants = pickVariants('direct,clip512,clip512-dp2');
  console.log(`fidelity at ${W}×${H} vs the converged WebGL2 truth, frames ${frames.join(',')}`);
  const src = await loadShader(WORKTREE);
  const gl = await runSession(webglBase(['shots'], [
    { name: 'truth', vs: src.vs, fs: transformShader(src.fs, 'truth') },
    { name: 'webgl', vs: src.vs, fs: src.fs },
  ], { shotFrames: frames }));
  const gpu = await runWebGPU({ modes: ['shots'], variants, shotFrames: frames });
  const dir = path.join(OUT, 'webgpu', 'fidelity');
  const a = saveShots(gl, dir, true);
  const b = saveShots(gpu, dir, false);
  const all = new Map([...a, ...b]);
  const diffs = [diffTable(all, 'truth', all, 'webgl', frames), ...variants.map((v) => diffTable(all, 'truth', all, v.name, frames))];
  for (const d of diffs) printDiff(d);
  const vsDirect = variants.slice(1).map((v) => diffTable(all, 'direct', all, v.name, frames));
  for (const d of vsDirect) printDiff(d);
  writeFileSync(path.join(OUT, 'webgpu-fidelity.json'), JSON.stringify({ res: [W, H], frames, vsTruth: diffs, vsDirect }, null, 1));
}

/**
 * Third person: the hero framed on the camera path (bench/hero.ts `frames`
 * output) drawn by the raster renderer next to the WebGL2 hero program;
 * frames diffed, and GPU cost with and without the hero.
 */
async function hero(): Promise<void> {
  const config = opt('config', 'run');
  const file = path.join(OUT, '.cache', `framed-${config}.json`);
  const framed: CameraFrame[] = JSON.parse(readFileSync(file, 'utf8'));
  const frames = opt('frames', '') === 'all' || config === 'gallery' && !opt('frames', '')
    ? framed.map((_, i) => i)
    : frameList(SHOT_FRAMES).filter((i) => i < framed.length);
  const variants = pickVariants('raster128');
  console.log(`hero (${config}) at ${W}×${H}, frames ${frames.join(',')}: ${variants.map((v) => v.name).join(', ')} vs WebGL2`);
  const src = await loadShader(WORKTREE);
  const { withHero } = await import('../src/components/terrain/demoShader');
  const gl = await runSession(webglBase(['shots'], [{ name: 'webgl-hero', vs: src.vs, fs: withHero(src.fs), hero: true }], { frames: framed, shotFrames: frames }));
  const gpu = await runWebGPU({ modes: ['shots', 'timing'], variants, frames: framed, shotFrames: frames });
  const dir = path.join(OUT, 'webgpu', `hero-${config}`);
  if (config === 'gallery') console.log('  scenes:', frames.map((i) => `${i}:${framed[i].scene}`).join(' '));
  const a = saveShots(gl, dir, true);
  const b = saveShots(gpu, dir, false);
  for (const v of variants) printDiff(diffTable(a, 'webgl-hero', b, v.name, frames));
  const shotsInfo = gpu.results.shots as Array<{ variant: string; frame: number; blades?: number; trees?: [number, number] }>;
  const bl = shotsInfo.filter((x) => x.blades);
  if (bl.length) console.log('  blades per frame:', bl.map((x) => `${x.variant} f${x.frame}: ${x.blades}`).join(', '));
  const tr = shotsInfo.filter((x) => x.trees && (x.trees[0] || x.trees[1]));
  if (tr.length) console.log('  trees (near/far) per frame:', tr.map((x) => `${x.variant} f${x.frame}: ${x.trees![0]}/${x.trees![1]}`).join(', '));
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((x) => !x.s);
  for (const [vi, v] of variants.entries()) {
    const s = summarize(t.filter((x) => x.v === vi).map((x) => x.ms));
    console.log(`  ${v.name} with the hero: p50 ${fmt(s.p50, 2)} ms  p95 ${fmt(s.p95, 2)} ms`);
  }
}

/**
 * Anti-aliasing (E6): still shots of ground-level views (TAA given 32 frames of
 * history) scored against a 2 × 2 supersampled render of the same variant
 * family (4× MSAA at twice the resolution, box-filtered down); a slow pan for
 * temporal stability; GPU cost of each.
 */
async function aa(): Promise<void> {
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', 'framed-gallery.json'), 'utf8'));
  const shotFrames = opt('frames', '1,3,5,7,13').split(',').map(Number);
  const variants = pickVariants('world-noaa,world-sky,world-taa,world-taa-soft,world-taau');
  console.log(`anti-aliasing at ${W}×${H}: ${variants.map((v) => v.name).join(', ')}; reference: world-sky at ${2 * W}×${2 * H}, box 2×2`);
  const ref = await runSession({
    variants: pickVariantsFrom(['world-sky']), presets: [{ id: 'ref', w: 2 * W, h: 2 * H, marchSteps: MARCH_STEPS, shadowSteps: SHADOW_STEPS }],
    frames: framed, seed: DEFAULT_SEED, fov: CAMERA.fov, reps: 1, warmupMs: 500, sentinelEvery: 10, batch: 8, repeat: 3,
    shotFrames, probes: [], gen: { starts: [], rebuilds: 0, cases: [] }, modes: ['shots'], stallMs: 180_000, timeoutMs: 40 * 60_000,
  }, { pageJs: await webgpuPage(), chromeArgs: CHROME_WEBGPU });
  // A slow pure yaw (≈ 0.3 px per frame at 1280 px), time held still: what changes between
  // frames beyond the sub-pixel drift is aliasing shimmer.
  const pan: CameraFrame[] = [];
  const f0 = framed[3];
  for (let k = 0; k < 32; k++) pan.push({ ...f0, frame: k, yaw: f0.yaw + 0.0004 * k });
  const gpu = await runWebGPU({ modes: ['shots', 'seq', 'timing'], variants, frames: framed, shotFrames, converge: 32, seq: pan, seqFreeze: true });
  const box = (img: RGBAImage): RGBAImage => {
    const w = img.width / 2, h = img.height / 2, out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) {
      const at = (xx: number, yy: number) => img.data[(yy * img.width + xx) * 4 + c];
      out[(y * w + x) * 4 + c] = Math.round((at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1)) / 4);
    }
    return { width: w, height: h, data: out };
  };
  const refShots = saveShots(ref, path.join(OUT, 'webgpu', 'aa-ref'), false);
  const shots = saveShots(gpu, path.join(OUT, 'webgpu', 'aa'), false);
  const refs = new Map([...refShots].map(([k, img]) => [k.replace('world-sky|', 'ref|'), box(img)]));
  const seqRes = gpu.results.seq as Array<{ variant: string; meanDelta: number[]; flicker: number[] }>;
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((x) => !x.s);
  console.log('  variant        vs 2×2 SSAA: meanΔE  >JND    | pan: mean |ΔY|  flicker px | GPU p50   p95');
  variants.forEach((v, vi) => {
    const d = diffTable(refs, 'ref', shots, v.name, shotFrames);
    const sq = seqRes.find((x) => x.variant === v.name)!;
    const md = sq.meanDelta.reduce((a, b) => a + b, 0) / sq.meanDelta.length;
    const fl = sq.flicker.reduce((a, b) => a + b, 0) / sq.flicker.length;
    const st = summarize(t.filter((x) => x.v === vi).map((x) => x.ms));
    console.log(`  ${v.name.padEnd(14)} ${fmt(d.meanDE, 4).padStart(13)} ${(d.fracJND * 100).toFixed(2).padStart(6)}% | ${fmt(md, 3).padStart(14)} ${(fl * 100).toFixed(3).padStart(9)}% | ${fmt(st.p50, 2).padStart(7)} ${fmt(st.p95, 2).padStart(6)}`);
  });
  writeFileSync(path.join(OUT, 'webgpu-aa.json'), JSON.stringify({ res: [W, H], shotFrames, seq: seqRes, timing: t }, null, 1));
}

function pickVariantsFrom(names: string[]): Array<{ name: string; variant: TerrainVariant }> {
  return names.map((name) => ({ name, variant: VARIANTS[name] }));
}

/**
 * Forest placement: the GPU's window of plants (forestGen, read back after a
 * shot) against the CPU twins, cell by cell: the land's layer (forest/placement.ts,
 * plantInCell) and the sky islands' (skyworld/flora.ts, skyPlantInCell).
 */
async function parity(): Promise<void> {
  const config = opt('config', 'biomes');
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const frames = opt('frames', '0,6,10,14,16').split(',').map(Number);
  const gpu = await runWebGPU({ modes: ['shots'], variants: pickVariants('world-taa'), frames: framed, shotFrames: frames, forestDump: true });
  const seed = DEFAULT_SEED;
  const world: PlaceWorld = {
    onRuin: (x, z) => { const r = ruinSiteAt(x, z, seed); return !!r && Math.hypot(x - r.x, z - r.z) < r.radius; },
  };
  console.log('forest placement, GPU vs CPU twin (land, then sky): cells, plants (GPU / CPU), agree, only GPU, only CPU, other species, moved (> 5 cm)');
  for (const b of gpu.blobs.filter((x) => x.query.kind === 'forest')) {
    const side = Number(b.query.side);
    const buf = b.data.buffer.slice(b.data.byteOffset, b.data.byteOffset + b.data.byteLength);
    const tiles = new Int32Array(buf, 0, side * side * 4);
    const slots = new Uint32Array(buf, side * side * 16);
    const slotF = new Float32Array(buf, side * side * 16);
    for (const layer of [0, 1]) {
      let cells = 0, gpuN = 0, cpuN = 0, agree = 0, onlyG = 0, onlyC = 0, other = 0, moved = 0;
      for (let t = 0; t < side * side; t++) {
        const tx = tiles[4 * t], tz = tiles[4 * t + 1];
        for (let c = 0; c < 64; c++) {
          const i = (layer * side * side * 64 + t * 64 + c) * 4;
          const valid = (slots[i + 3] & 0x40000000) !== 0;
          const ix = tx * 8 + (c % 8), iz = tz * 8 + Math.floor(c / 8);
          const p = layer === 0 ? plantInCell(ix, iz, seed, world) : skyPlantInCell(ix, iz, seed);
          cells++;
          if (valid) gpuN++;
          if (p) cpuN++;
          if (!valid && !p) { agree++; continue; }
          if (valid && !p) { onlyG++; continue; }
          if (!valid && p) { onlyC++; continue; }
          if (SPECIES[slots[i + 3] & 15] !== p!.species) { other++; continue; }
          agree++;
          if (Math.hypot(tx * 48 + slotF[i] - p!.x, tz * 48 + slotF[i + 1] - p!.z) > 0.05) moved++;
        }
      }
      console.log(`  f${b.query.frame} ${layer ? 'sky ' : 'land'}: ${cells} cells, ${gpuN} / ${cpuN} plants, ${(100 * agree / cells).toFixed(3)} % agree, ${onlyG} only GPU, ${onlyC} only CPU, ${other} other species, ${moved} moved`);
    }
  }
}

/**
 * The horizons the GPU baked along the sun's path (horizonShader) against a
 * brute-force march of the CPU twin in the same tilted plane: every 1 % of
 * the distance (the GPU steps ~9 %), the land's top found by a damped fixed
 * point, the same curvature, sea-level dip and reach. Heights are the CPU
 * twin's at the footprint of the clipmap level round each sample.
 */
async function horizons(): Promise<void> {
  const variants = pickVariants('world-cycle');
  const v = variants[0].variant;
  if (v.height !== 'raster') throw new Error('horizons: needs a raster variant');
  const config = opt('config', 'sky');
  const all: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const picks = opt('frames', '0,3,5').split(',').map(Number);
  const cams = picks.map((i) => all[i]);
  const levels = opt('levels', '0,3,6').split(',').map(Number);
  const res = await runWebGPU({ modes: ['horizon'], variants, frames: cams, horizonLevels: levels });
  const { n, spacing0, margin, levels: L } = v.clip;
  const cosLat = Math.cos(SKY.latitude), sinLat = Math.sin(SKY.latitude);
  const curv = 1 / (2 * SKY.planetRadius);
  const DEG = 180 / Math.PI;
  let seed0 = 20260930;
  const rnd = () => { seed0 = (seed0 * 1664525 + 1013904223) >>> 0; return seed0 / 4294967296; };
  type Row = { frame: number; level: number; samples: number; medianDeg: number; p95Deg: number; maxDeg: number; within1Deg: number };
  const rows: Row[] = [];
  for (const b of res.blobs.filter((x) => x.query.kind === 'horizon')) {
    const fi = Number(b.query.frame), level = Number(b.query.level), s = Number(b.query.spacing);
    const cam = cams[fi];
    const data = new Float32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4);
    // Height at the footprint of the finest level whose window holds the point, as the clipmap stores it.
    const H = (x: number, z: number) => {
      const d = Math.max(Math.abs(x - cam.x), Math.abs(z - cam.z));
      let l = 0;
      while (l < L - 1 && d > (n / 2 - margin) * spacing0 * 2 ** l) l++;
      return elevation(x, z, DEFAULT_SEED, spacing0 * 2 ** l * v.genDpScale);
    };
    const edge = (n / 2 - margin) * spacing0 * 2 ** (L - 1);
    const horizon = (px: number, py: number, pz: number, sgn: number) => {
      let th = -Math.asin(Math.min(Math.sin(Math.sqrt(4 * Math.max(py, 0) * curv)) / cosLat, 1));
      for (let a = Math.max(0.6, 1.5 * s); a < 8000; a += Math.max(0.25, a * 0.01)) {
        let bb = a * Math.tan(th);
        const qx = px + sgn * a;
        if (Math.max(Math.abs(qx - cam.x), Math.abs(pz - cam.z)) > edge) break;
        // the land's top in the plane at this distance: damped fixed point from the current horizon
        let top = -Infinity;
        for (let k = 0; k < 14; k++) {
          const qz = pz + PATH_N[2] * bb;
          const drop = (a * a + bb * bb * sinLat * sinLat) * curv;
          const want = (H(qx, qz) - drop - py) / cosLat;
          if (k === 0 && want <= bb) break; // below the current horizon here
          top = want;
          bb = bb + 0.6 * (want - bb);
        }
        if (top > -Infinity) th = Math.max(th, Math.atan2(top, a));
        if (py + a * Math.tan(th) * cosLat + a * a * curv > 1300) break;
      }
      return th;
    };
    const errs: number[] = [];
    const signed: number[] = [];
    const lim = (n / 2 - margin - 2) * s;
    for (let k = 0; k < Number(opt('samples', '160')); k++) {
      const x = cam.x + (rnd() * 2 - 1) * lim, z = cam.z + (rnd() * 2 - 1) * lim;
      const I = Math.round(x / s), J = Math.round(z / s);
      const tx = ((I % n) + n) % n, tz = ((J % n) + n) % n;
      const o = (tz * n + tx) * 4;
      const gRise = data[o], gSet = data[o + 1];
      const wx = I * s, wz = J * s;
      const e = H(wx, wz);
      const gx = (H(wx + s, wz) - H(wx - s, wz)) / (2 * s), gz = (H(wx, wz + s) - H(wx, wz - s)) / (2 * s);
      const inv = 1 / Math.hypot(gx, 1, gz), off = 0.15 + 0.5 * s;
      const px = wx - gx * inv * off, py = e + inv * off, pz = wz - gz * inv * off;
      const cRise = horizon(px, py, pz, 1), cSet = Math.PI - horizon(px, py, pz, -1);
      errs.push(Math.abs(gRise - cRise) * DEG, Math.abs(gSet - cSet) * DEG);
      // signed: the GPU's horizon lower (shadows too short) or higher than the brute force's
      signed.push((gRise - cRise) * DEG, (cSet - gSet) * DEG);
    }
    const low = signed.filter((x) => x < -1).length / signed.length, high = signed.filter((x) => x > 1).length / signed.length;
    console.log(`    signed: GPU lower by > 1° ${(low * 100).toFixed(1)} %, higher by > 1° ${(high * 100).toFixed(1)} %`);
    errs.sort((a, b) => a - b);
    const q = (p: number) => errs[Math.min(errs.length - 1, Math.floor(p * errs.length))];
    const row = { frame: picks[fi], level, samples: errs.length, medianDeg: q(0.5), p95Deg: q(0.95), maxDeg: errs[errs.length - 1], within1Deg: errs.filter((x) => x < 1).length / errs.length };
    rows.push(row);
    console.log(`  camera ${row.frame} level ${level} (${s} m): |Δθ| median ${row.medianDeg.toFixed(3)}°  p95 ${row.p95Deg.toFixed(2)}°  max ${row.maxDeg.toFixed(2)}°  within 1° ${(row.within1Deg * 100).toFixed(1)} %  (1° = 4 game minutes of shadow timing)`);
  }
  writeFileSync(path.join(OUT, 'webgpu-horizons.json'), JSON.stringify({ config, cameras: picks, levels, rows }, null, 1));
}

/**
 * Temporal stability of chosen views (say, the trees' surface detail against plain trees): at each
 * view 24 frames held still (TAA settles), then a slow pure yaw of 32 frames (≈ 0.3 px per frame at
 * 1280 px), time frozen. Per view and variant: the pan's mean |ΔY| between consecutive frames and the
 * share of pixels changing by more than 24 (flicker); what the variants add beyond the drift is shimmer.
 */
async function shimmer(): Promise<void> {
  const config = opt('config', 'biomes');
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const views = opt('frames', '0').split(',').map(Number);
  const variants = pickVariants('world-cycle-plain,world-cycle');
  const HOLD = 24, PAN = 32;
  const seq: CameraFrame[] = [];
  for (const v of views) for (let k = 0; k < HOLD + PAN; k++) seq.push({ ...framed[v], frame: seq.length, yaw: framed[v].yaw + 0.0004 * Math.max(0, k - HOLD) });
  const gpu = await runWebGPU({ modes: ['seq'], variants, frames: framed, seq, seqFreeze: true, hour: Number(opt('hour', '15.5')) });
  const res = gpu.results.seq as Array<{ variant: string; meanDelta: number[]; flicker: number[] }>;
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
  console.log(`shimmer at ${W}×${H}, ${config} views ${views.join(',')}: pan mean |ΔY| (0–255) / flicker (% of pixels changing > 24)`);
  views.forEach((v, vi) => {
    // Step i compares frame i + 1 with frame i: the pan's steps within this view's block.
    const from = vi * (HOLD + PAN) + HOLD, to = from + PAN - 1;
    console.log(`  view ${String(v).padStart(2)} ${String(framed[v].scene ?? '').padEnd(22)} ` + res.map((r) =>
      `${r.variant}: ${fmt(mean(r.meanDelta.slice(from, to)), 3)} / ${(mean(r.flicker.slice(from, to)) * 100).toFixed(3)} %`).join('   '));
  });
}

const modes: Record<string, () => Promise<void>> = { port, cost, gen, fidelity, hero, aa, stills, parity, horizons, shimmer, taa };
if (!modes[mode]) throw new Error(`unknown mode ${mode}; one of ${Object.keys(modes).join(', ')}`);
await modes[mode]();
