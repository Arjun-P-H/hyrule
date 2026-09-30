/**
 * WebGPU backend for the game engine (engineCore.ts): the raster clipmap
 * renderer on the page's canvas. Loaded on demand (dynamic import) only when
 * the page asks for WebGPU (`?renderer=webgpu`), so the WebGL2 path ships
 * without it.
 */

import { createTerrainRenderer, type FrameParams, type TerrainVariant } from './terrainRenderer';

/**
 * The game's renderer (docs/EXPERIMENTS.md): E3's raster clipmap (8 rings of
 * 256² cells over 8 levels of 512², 0.5 m at level 0, each cell split along
 * the diagonal the heights favour) with E5's grass (5 rings of 256²
 * candidates, 0.125 m cells near the camera), the forest (ten species; forest
 * bodies, groves and lone trees placed by climate zone and terrain on a 6 m
 * lattice cached out to ~1.1 km; a sun shadow map), E7's sky islands,
 * waterfalls, ruins and shrines, and E6's
 * temporal anti-aliasing, upscaling the internal resolution to the canvas.
 */
export const GAME_VARIANT: TerrainVariant = {
  height: 'raster', k: 128, stride: 1, morphStart: 0.75, taa: true,
  clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25,
  grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
  forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 },
  islands: true,
  ruins: true,
  diag: true,
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
  dispose(): void;
};

export async function createGpuBackend(canvas: HTMLCanvasElement): Promise<GpuBackend> {
  if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  // The clipmap is sampled with hardware bilinear filtering of r32float.
  if (!adapter.features.has('float32-filterable')) throw new Error('WebGPU adapter lacks float32-filterable');
  const device = await adapter.requestDevice({ requiredFeatures: ['float32-filterable'] });
  const ctx = canvas.getContext('webgpu');
  if (!ctx) {
    device.destroy();
    throw new Error('WebGPU canvas context unavailable');
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const renderer = createTerrainRenderer(device, format, GAME_VARIANT);
  try {
    await renderer.init();
  } catch (e) {
    ctx.unconfigure();
    device.destroy();
    throw e;
  }

  let inflight = 0;
  let lost = false;
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
      renderer.destroy();
      device.destroy();
    },
  };
}
