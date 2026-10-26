/**
 * WebGPU backend for the game engine (engineCore.ts): the raster clipmap
 * renderer on the page's canvas. Loaded on demand (dynamic import) only when
 * the page asks for WebGPU (`?renderer=webgpu`), so the WebGL2 path ships
 * without it.
 */

import { createTerrainRenderer, type FrameParams, type TerrainRendererLike, type TerrainVariant } from './terrainRenderer';
import type { World } from '../world';

/** The sky the game draws: the day–night cycle, or the classic fixed sky. */
export type SkyMode = 'cycle' | 'classic';

/**
 * The game's renderer (docs/EXPERIMENTS.md): E3's raster clipmap (12 rings of
 * 256² cells over 12 levels of 512², 0.5 m at level 0, each cell split along
 * the diagonal the heights favour; the land reaches 131 km, where the haze
 * has long hidden it) with E5's grass (5 rings of 256²
 * candidates, 0.125 m cells near the camera), the forest (ten species; forest
 * bodies, groves and lone trees placed by climate zone and terrain on a 6 m
 * lattice cached out to ~1.1 km; a sun shadow map), E7's sky islands,
 * waterfalls, ruins and shrines, the land's model isles (skymodels.ts), and E6's
 * temporal anti-aliasing, upscaling the internal resolution to the canvas,
 * and combat (the foes, arrows, sparks: combat.wgsl.ts).
 * The sky (the day–night cycle or the classic sky) is chosen per backend.
 */
export const GAME_VARIANT: TerrainVariant = {
  height: 'raster', k: 128, stride: 1, morphStart: 0.75, taa: true,
  clip: { n: 512, levels: 12, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25,
  grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
  forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 },
  islands: true,
  ruins: true,
  diag: true,
  combat: true,
  rocks: true,
  modelIsles: true,
};

export type GpuBackend = {
  /** Adapter description, for the HUD and logs. */
  readonly label: string;
  /** The canvas shows the display resolution; the scene renders at FrameParams.renderScale of it. */
  readonly upscales: boolean;
  /** Frames submitted and not yet finished on the GPU. */
  inFlight(): number;
  /** Generate what the camera exposed and draw one frame into the canvas. */
  draw(fp: FrameParams): void;
  /** The sky being drawn. */
  sky(): SkyMode;
  /**
   * Switch skies: the renderer for the new sky compiles in the background
   * while the current one keeps drawing, then takes over (a request made
   * meanwhile is served after it).
   */
  setSky(mode: SkyMode): void;
  /** The world being drawn. */
  world(): World;
  /** Switch worlds (the portal: portal.ts), the same way as setSky. */
  setWorld(world: World): void;
  dispose(): void;
};

/**
 * The game's renderer for a sky and a world: the land as ever (its sky realm overhead); the cloud world
 * (inactive: reached only by the settings' switch, for inspection) the realm over its cloud ground, with
 * none of the land's grass, trees and rocks.
 */
export function gameVariant(sky: SkyMode, world: World): TerrainVariant {
  return (world === 'clouds'
    ? { ...GAME_VARIANT, sky, world, grass: null, forest: null, rocks: false, puffs: true, modelIsles: false }
    : { ...GAME_VARIANT, sky, world }) as TerrainVariant;
}

export async function createGpuBackend(canvas: HTMLCanvasElement, initialSky: SkyMode = 'cycle', initialWorld: World = 'land'): Promise<GpuBackend> {
  if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  // The clipmap is sampled with hardware bilinear filtering of rg32float (heights, water).
  if (!adapter.features.has('float32-filterable')) throw new Error('WebGPU adapter lacks float32-filterable');
  // Clip distances cut the sky islands' tops along their outlines (else the renderer discards there);
  // half-precision shader arithmetic runs the TAA's colour maths at twice the rate (else in f32).
  const optional: GPUFeatureName[] = ['clip-distances', 'shader-f16'];
  const device = await adapter.requestDevice({
    requiredFeatures: ['float32-filterable', ...optional.filter((f) => adapter.features.has(f))],
  });
  const ctx = canvas.getContext('webgpu');
  if (!ctx) {
    device.destroy();
    throw new Error('WebGPU canvas context unavailable');
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  let renderer: TerrainRendererLike = createTerrainRenderer(device, format, gameVariant(initialSky, initialWorld));
  let current = initialSky, currentWorld = initialWorld;
  try {
    await renderer.init();
  } catch (e) {
    ctx.unconfigure();
    device.destroy();
    throw e;
  }

  let inflight = 0;
  let lost = false;
  let disposed = false;
  let wanted = initialSky, wantedWorld = initialWorld;
  let switching: Promise<void> | null = null;
  // Build renderers until the one drawing is the one wanted (sky and world); the old one is
  // destroyed once the frames it drew have left the GPU.
  const switchSky = async () => {
    while ((wanted !== current || wantedWorld !== currentWorld) && !lost && !disposed) {
      const target = wanted, targetWorld = wantedWorld;
      const next = createTerrainRenderer(device, format, gameVariant(target, targetWorld));
      try {
        await next.init();
      } catch (e) {
        console.error(`WebGPU: the ${target} sky over the ${targetWorld} failed to build`, e);
        next.destroy();
        wanted = current;
        wantedWorld = currentWorld;
        break;
      }
      if (lost || disposed) { next.destroy(); break; }
      const old = renderer;
      renderer = next;
      current = target;
      currentWorld = targetWorld;
      void device.queue.onSubmittedWorkDone().then(() => old.destroy());
    }
    switching = null;
  };
  void device.lost.then((info) => {
    lost = true;
    if (info.reason !== 'destroyed') console.error(`WebGPU device lost: ${info.message}`);
  });
  const finished = () => { inflight--; };
  const info = adapter.info;

  return {
    label: info.description || info.vendor || 'WebGPU',
    upscales: GAME_VARIANT.height === 'raster' && !!GAME_VARIANT.taa,
    inFlight: () => inflight,
    sky: () => current,
    setSky(mode) {
      wanted = mode;
      if (!switching && wanted !== current) switching = switchSky();
    },
    world: () => currentWorld,
    setWorld(w) {
      wantedWorld = w;
      if (!switching && wantedWorld !== currentWorld) switching = switchSky();
    },
    draw(fp) {
      if (lost) return;
      renderer.setFrame(fp);
      const enc = device.createCommandEncoder();
      renderer.encodeGenerate(enc);
      renderer.encodeRender(enc, ctx.getCurrentTexture().createView());
      device.queue.submit([enc.finish()]);
      inflight++;
      void device.queue.onSubmittedWorkDone().then(finished);
    },
    dispose() {
      // No ctx.unconfigure(): a backend created after this one (StrictMode
      // remount) may already have configured the same canvas context.
      disposed = true;
      renderer.destroy();
      device.destroy();
    },
  };
}
