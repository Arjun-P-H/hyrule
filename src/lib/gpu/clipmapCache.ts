/**
 * The GPU side of the clipmap: the height texture array, the generator
 * pipeline and the sampling uniforms (see clipmap.ts for the scheduling).
 * ---------------------------------------------------------------------------
 * One r32float layer per level; lattice point (i, j) of level l is stored at
 * texel (i mod n, j mod n) of layer l. The renderers bind `view`, `sampler`
 * and `clipBuf` (heightAt() in terrain.wgsl.ts) and call setCamera() /
 * encodeGenerate() once per frame. With `sun`, a second layer array holds the
 * ground's sun visibility, marched once per lattice point right after its
 * height is generated (sunShader), bound by the renderers as `sunView`.
 */

import { ClipmapScheduler, levelSpacing, rectTexels, usableRadius, wrap, type ClipmapSpec, type UpdateRect } from './clipmap';
import { GEN_BYTES, clipBytes, genShader, sunShader } from './terrain.wgsl';

export type PassTimestamps = { querySet: GPUQuerySet; begin: number; end: number };

const GEN_SLOT = 256;
const GEN_SLOTS = 64;

export class ClipmapCache {
  readonly spec: ClipmapSpec;
  readonly genDpScale: number;
  readonly scheduler: ClipmapScheduler;
  compileMs = 0;
  texture!: GPUTexture;
  view!: GPUTextureView;
  sampler!: GPUSampler;
  clipBuf!: GPUBuffer;

  private device: GPUDevice;
  private clipData: Float32Array<ArrayBuffer>;
  private genPipe!: GPUComputePipeline;
  private genBind!: GPUBindGroup;
  private genBuf!: GPUBuffer;
  private genData = new ArrayBuffer(GEN_BYTES);
  private pending: UpdateRect[] = [];
  private origin: [number, number] = [0, 0];
  private seed: number | null = null;
  private sunOpts: { blend: number } | null;
  sunTexture: GPUTexture | null = null;
  sunView: GPUTextureView | null = null;
  private sunPipe: GPUComputePipeline | null = null;
  private sunBind: GPUBindGroup | null = null;

  constructor(device: GPUDevice, spec: ClipmapSpec, genDpScale: number, sun: { blend: number } | null = null) {
    this.device = device;
    this.spec = spec;
    this.genDpScale = genDpScale;
    this.sunOpts = sun;
    this.scheduler = new ClipmapScheduler(spec);
    this.clipData = new Float32Array(clipBytes(spec.levels) / 4);
  }

  async init(): Promise<void> {
    const d = this.device;
    const spec = this.spec;
    const t0 = performance.now();
    const layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: GEN_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d-array' } },
      ],
    });
    const sunLayout = this.sunOpts ? d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: GEN_BYTES } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d-array' } },
      ],
    }) : null;
    const [genPipe, sunPipe] = await Promise.all([
      d.createComputePipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: d.createShaderModule({ code: genShader(spec.n) }), entryPoint: 'main' },
      }),
      sunLayout && this.sunOpts ? d.createComputePipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [sunLayout] }),
        compute: { module: d.createShaderModule({ code: sunShader({ n: spec.n, levels: spec.levels, blend: this.sunOpts.blend }) }), entryPoint: 'main' },
      }) : Promise.resolve(null),
    ]);
    this.genPipe = genPipe;
    this.sunPipe = sunPipe;
    this.compileMs = performance.now() - t0;
    this.texture = d.createTexture({
      size: [spec.n, spec.n, spec.levels],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView({ dimension: '2d-array' });
    this.sampler = d.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear' });
    this.clipBuf = d.createBuffer({ size: clipBytes(spec.levels), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.genBuf = d.createBuffer({ size: GEN_SLOT * GEN_SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.genBind = d.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.genBuf, size: GEN_BYTES } },
        { binding: 1, resource: this.view },
      ],
    });
    if (sunLayout) {
      this.sunTexture = d.createTexture({
        size: [spec.n, spec.n, spec.levels],
        format: 'r32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.sunView = this.sunTexture.createView({ dimension: '2d-array' });
      this.sunBind = d.createBindGroup({
        layout: sunLayout,
        entries: [
          { binding: 0, resource: { buffer: this.genBuf, size: GEN_BYTES } },
          { binding: 1, resource: { buffer: this.clipBuf } },
          { binding: 2, resource: this.view },
          { binding: 3, resource: this.sampler },
          { binding: 4, resource: this.sunView },
        ],
      });
    }
  }

  pendingTexels(): number {
    return rectTexels(this.pending);
  }

  invalidate(): void {
    this.scheduler.invalidate();
  }

  /**
   * Move the level windows to the camera (world metres x, z; chunk-local
   * lx, lz; chunk origin ox, oz) and upload the sampling uniforms.
   */
  setCamera(x: number, z: number, lx: number, lz: number, ox: number, oz: number, seed: number): void {
    // Rebasing is seamless to 0.14 mm (demoShader.ts header), so the cache
    // survives a new chunk origin; a new seed is a new world.
    if (seed !== this.seed) {
      this.scheduler.invalidate();
      this.pending.length = 0;
      this.seed = seed;
    }
    this.origin = [ox, oz];
    this.pending.push(...this.scheduler.update(x, z));
    const spec = this.spec;
    const cd = this.clipData;
    for (let l = 0; l < spec.levels; l++) {
      const s = levelSpacing(spec, l);
      cd[l * 4 + 0] = wrapF(x / s + 0.5, spec.n);
      cd[l * 4 + 1] = wrapF(z / s + 0.5, spec.n);
      cd[l * 4 + 2] = 1 / s;
      cd[l * 4 + 3] = s;
    }
    const c = spec.levels * 4;
    cd[c + 0] = lx;
    cd[c + 1] = lz;
    cd[c + 2] = 1 / usableRadius(spec, 0);
    cd[c + 3] = 0;
    this.device.queue.writeBuffer(this.clipBuf, 0, cd);
  }

  /** Generate every scheduled lattice point; returns the texel count. */
  encodeGenerate(enc: GPUCommandEncoder, ts?: PassTimestamps): number {
    const rects = this.pending.splice(0);
    if (rects.length === 0) return 0;
    if (rects.length > GEN_SLOTS) throw new Error(`clipmap: ${rects.length} rectangles exceed ${GEN_SLOTS} slots`);
    const f = new Float32Array(this.genData);
    const i = new Int32Array(this.genData);
    const u = new Uint32Array(this.genData);
    const [ox, oz] = this.origin;
    const n = this.spec.n;
    rects.forEach((r, k) => {
      const s = levelSpacing(this.spec, r.level);
      f[0] = ox; f[1] = oz;
      i[2] = r.i0 - ox / s; i[3] = r.j0 - oz / s;
      u[4] = wrap(r.i0, n); u[5] = wrap(r.j0, n);
      u[6] = r.w; u[7] = r.h;
      f[8] = s; f[9] = s * this.genDpScale;
      u[10] = r.level; i[11] = this.seed ?? 0;
      this.device.queue.writeBuffer(this.genBuf, k * GEN_SLOT, this.genData);
    });
    const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: ts.querySet, beginningOfPassWriteIndex: ts.begin, endOfPassWriteIndex: ts.end } } : {});
    pass.setPipeline(this.genPipe);
    rects.forEach((r, k) => {
      pass.setBindGroup(0, this.genBind, [k * GEN_SLOT]);
      pass.dispatchWorkgroups(Math.ceil(r.w / 8), Math.ceil(r.h / 8));
    });
    pass.end();
    // Sun visibility of the same lattice points, once every level's heights are in.
    if (this.sunPipe && this.sunBind) {
      const sp = enc.beginComputePass();
      sp.setPipeline(this.sunPipe);
      rects.forEach((r, k) => {
        sp.setBindGroup(0, this.sunBind!, [k * GEN_SLOT]);
        sp.dispatchWorkgroups(Math.ceil(r.w / 8), Math.ceil(r.h / 8));
      });
      sp.end();
    }
    return rectTexels(rects);
  }

  destroy(): void {
    this.texture?.destroy();
    this.sunTexture?.destroy();
    this.clipBuf?.destroy();
    this.genBuf?.destroy();
  }
}

function wrapF(x: number, n: number): number {
  return ((x % n) + n) % n;
}
