/**
 * WebGPU terrain renderers.
 * ---------------------------------------------------------------------------
 *   const r = createTerrainRenderer(device, 'rgba8unorm', variant);
 *   await r.init();
 *   r.setFrame(frame);                 // uniforms + clipmap windows for this camera
 *   r.encodeGenerate(encoder, ts);     // generate newly exposed lattice points
 *   r.encodeRender(encoder, view, ts); // draw the frame
 *
 * Variants: `direct` ray-marches E itself (the WebGL2 renderer, ported);
 * `cached` ray-marches the clipmap; `raster` rasterizes the clipmap as a
 * geometry clipmap (rasterRenderer.ts). The camera is given in fp64 world
 * metres; the renderers apply the WebGL2 engine's floating origin (chunk =
 * CHUNK_SIZE) so every value the GPU sees is chunk-local fp32.
 */

import { hydroBufferFor } from './hydroBuffer';
import { ClipmapCache, type PassTimestamps } from './clipmapCache';
import { floatingOrigin, packFrame, type FrameParams, type TerrainRendererLike, type TerrainVariant } from './frame';
import { FRAME_BYTES, terrainShader } from './terrain.wgsl';
import { RasterTerrainRenderer } from './rasterRenderer';

export type { PassTimestamps } from './clipmapCache';
export type { FrameParams, TerrainRendererLike, TerrainVariant } from './frame';

export function createTerrainRenderer(device: GPUDevice, format: GPUTextureFormat, variant: TerrainVariant): TerrainRendererLike {
  return variant.height === 'raster'
    ? new RasterTerrainRenderer(device, format, variant)
    : new TerrainRenderer(device, format, variant);
}

/** Ray-march renderer (direct or clipmap-cached heights). */
class TerrainRenderer implements TerrainRendererLike {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  readonly variant: TerrainVariant;
  compileMs = 0;

  private frameBuf: GPUBuffer;
  private frameData = new ArrayBuffer(FRAME_BYTES);
  private frameF32 = new Float32Array(this.frameData);
  private frameI32 = new Int32Array(this.frameData);
  private renderPipe!: GPURenderPipeline;
  private renderBind!: GPUBindGroup;
  private cache: ClipmapCache | null = null;

  constructor(device: GPUDevice, format: GPUTextureFormat, variant: TerrainVariant) {
    if (variant.height === 'raster') throw new Error('use RasterTerrainRenderer');
    this.device = device;
    this.format = format;
    this.variant = variant;
    this.frameBuf = device.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const d = this.device;
    const t0 = performance.now();
    const v = this.variant;
    const module = d.createShaderModule({
      code: terrainShader(v.height === 'cached' ? { height: 'cached', n: v.clip.n, levels: v.clip.levels, blend: v.blend } : { height: 'direct' }),
    });
    const renderP = d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    if (v.height === 'cached') {
      this.cache = new ClipmapCache(d, v.clip, v.genDpScale);
      await this.cache.init();
    }
    this.renderPipe = await renderP;
    this.compileMs = performance.now() - t0;
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: this.frameBuf } }];
    if (this.cache) {
      entries.push(
        { binding: 1, resource: { buffer: this.cache.clipBuf } },
        { binding: 2, resource: this.cache.view },
        { binding: 3, resource: this.cache.sampler },
      );
    }
    // the direct height source evaluates the land per pixel: it reads the rivers' node table
    if (v.height === 'direct') entries.push(hydroBufferFor(d).entry());
    this.renderBind = d.createBindGroup({ layout: this.renderPipe.getBindGroupLayout(0), entries });
  }

  pendingTexels(): number {
    return this.cache ? this.cache.pendingTexels() : 0;
  }

  invalidate(): void {
    this.cache?.invalidate();
  }

  setFrame(fp: FrameParams): void {
    packFrame(fp, this.frameF32, this.frameI32);
    this.device.queue.writeBuffer(this.frameBuf, 0, this.frameData);
    if (this.cache) {
      const { ox, oz, lx, lz } = floatingOrigin(fp.x, fp.z);
      this.cache.setCamera(fp.x, fp.z, lx, lz, ox, oz, fp.seed);
    } else {
      hydroBufferFor(this.device).update(fp.x, fp.z, fp.seed);
    }
  }

  encodeGenerate(enc: GPUCommandEncoder, ts?: PassTimestamps): number {
    return this.cache ? this.cache.encodeGenerate(enc, ts) : 0;
  }

  encodeRender(enc: GPUCommandEncoder, target: GPUTextureView, ts?: PassTimestamps): void {
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      ...(ts ? { timestampWrites: { querySet: ts.querySet, beginningOfPassWriteIndex: ts.begin, endOfPassWriteIndex: ts.end } } : {}),
    });
    pass.setPipeline(this.renderPipe);
    pass.setBindGroup(0, this.renderBind);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this.frameBuf.destroy();
    this.cache?.destroy();
  }
}
