/**
 * The ground's detail textures (host side of groundTex.wgsl.ts).
 * ---------------------------------------------------------------------------
 * One texture array, a layer per material (soil, turf, rock …), painted on the GPU the first time a
 * device asks for it and shared by every renderer on that device (the game builds a second renderer
 * when the sky changes: the textures are not painted again). Binding 41 is the array, 42 its sampler.
 */

import { GROUND_LAYERS, GROUND_MIPS, GROUND_MIP_WGSL, GROUND_SIZE, groundPaintWGSL, groundShadeWGSL } from './groundTex.wgsl';

const GROUND_TEX_BINDING = 41;
const GROUND_SAMPLER_BINDING = 42;

export class GroundTextures {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly sampler: GPUSampler;
  private ready: Promise<void> | null = null;
  /** How long the first painting took (ms): building the pipelines, then the GPU's work (diagnostics). */
  paintMs = { compile: 0, run: 0 };
  /** When each layer's painter was ready, from the start of the pipelines' building (ms; diagnostics). */
  layerMs: number[] = [];

  constructor(private readonly device: GPUDevice) {
    this.texture = device.createTexture({
      size: [GROUND_SIZE, GROUND_SIZE, GROUND_LAYERS.length], mipLevelCount: GROUND_MIPS, format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.view = this.texture.createView({ dimension: '2d-array' });
    this.sampler = device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 2,
    });
  }

  /** Entries the ground textures add to group 0 (read by the fragment shaders only). */
  static layoutEntries(): GPUBindGroupLayoutEntry[] {
    return [
      { binding: GROUND_TEX_BINDING, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: GROUND_SAMPLER_BINDING, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ];
  }

  bindEntries(): GPUBindGroupEntry[] {
    return [
      { binding: GROUND_TEX_BINDING, resource: this.view },
      { binding: GROUND_SAMPLER_BINDING, resource: this.sampler },
    ];
  }

  /** Paint the layers and their mips (once; later calls return the same promise). */
  init(): Promise<void> {
    return (this.ready ??= this.paint());
  }

  private async paint(): Promise<void> {
    const d = this.device;
    const C = GPUShaderStage.COMPUTE;
    const paintModule = d.createShaderModule({ code: groundPaintWGSL() });
    const shadeModule = d.createShaderModule({ code: groundShadeWGSL() });
    const mipModule = d.createShaderModule({ code: GROUND_MIP_WGSL });
    const paintLayout = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' } }],
    });
    const readWriteLayout = (format: GPUTextureFormat) => d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 1, visibility: C, storageTexture: { access: 'write-only', format, viewDimension: '2d-array' } },
      ],
    });
    const shadeLayout = readWriteLayout('rgba8unorm');
    const mipLayout = readWriteLayout('rgba8unorm');
    const t0 = performance.now();
    const [paintPipes, shadePipe, mipPipe] = await Promise.all([
      Promise.all(GROUND_LAYERS.map((_, k) => d.createComputePipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [paintLayout] }),
        compute: { module: paintModule, entryPoint: 'paint', constants: { LAYER: k } },
      }).then((p) => { this.layerMs[k] = performance.now() - t0; return p; }))),
      d.createComputePipelineAsync({ layout: d.createPipelineLayout({ bindGroupLayouts: [shadeLayout] }), compute: { module: shadeModule, entryPoint: 'shade' } }),
      d.createComputePipelineAsync({ layout: d.createPipelineLayout({ bindGroupLayouts: [mipLayout] }), compute: { module: mipModule, entryPoint: 'mip' } }),
    ]);
    const t1 = performance.now();
    // Heights, tones and cavities of every layer, as floats (the slopes are taken from these).
    const scratch = d.createTexture({
      size: [GROUND_SIZE, GROUND_SIZE, GROUND_LAYERS.length], format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    const scratchView = scratch.createView({ dimension: '2d-array' });
    const level = (mip: number) => this.texture.createView({ dimension: '2d-array', baseMipLevel: mip, mipLevelCount: 1 });
    const enc = d.createCommandEncoder();
    const paintBind = d.createBindGroup({ layout: paintLayout, entries: [{ binding: 0, resource: scratchView }] });
    for (const pipe of paintPipes) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, paintBind);
      pass.dispatchWorkgroups(GROUND_SIZE / 8, GROUND_SIZE / 8, 1);
      pass.end();
    }
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(shadePipe);
      pass.setBindGroup(0, d.createBindGroup({ layout: shadeLayout, entries: [{ binding: 0, resource: scratchView }, { binding: 1, resource: level(0) }] }));
      pass.dispatchWorkgroups(GROUND_SIZE / 8, GROUND_SIZE / 8, GROUND_LAYERS.length);
      pass.end();
    }
    for (let m = 1; m < GROUND_MIPS; m++) {
      const n = Math.max(GROUND_SIZE >> m, 1);
      const pass = enc.beginComputePass();
      pass.setPipeline(mipPipe);
      pass.setBindGroup(0, d.createBindGroup({ layout: mipLayout, entries: [{ binding: 0, resource: level(m - 1) }, { binding: 1, resource: level(m) }] }));
      pass.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(n / 8), GROUND_LAYERS.length);
      pass.end();
    }
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    scratch.destroy();
    this.paintMs = { compile: t1 - t0, run: performance.now() - t1 };
  }

  /** Mip `mip` of every layer, layer after layer, RGBA8 rows top-down (debug readback; waits for the GPU). */
  async read(mip = 0): Promise<Uint8Array> {
    await this.init();
    const d = this.device;
    const n = Math.max(GROUND_SIZE >> mip, 1);
    const bpr = Math.max(4 * n, 256);
    const rb = d.createBuffer({ size: bpr * n * GROUND_LAYERS.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    for (let k = 0; k < GROUND_LAYERS.length; k++) {
      enc.copyTextureToBuffer({ texture: this.texture, mipLevel: mip, origin: [0, 0, k] }, { buffer: rb, offset: k * bpr * n, bytesPerRow: bpr, rowsPerImage: n }, [n, n, 1]);
    }
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(rb.getMappedRange());
    const out = new Uint8Array(4 * n * n * GROUND_LAYERS.length);
    for (let k = 0; k < GROUND_LAYERS.length; k++) {
      for (let y = 0; y < n; y++) out.set(raw.subarray(k * bpr * n + y * bpr, k * bpr * n + y * bpr + 4 * n), (k * n + y) * 4 * n);
    }
    rb.unmap();
    rb.destroy();
    return out;
  }

  destroy(): void {
    this.texture.destroy();
  }
}

const perDevice = new WeakMap<GPUDevice, GroundTextures>();

/** The device's ground textures (made on first use; init() paints them). */
export function groundTexturesFor(device: GPUDevice): GroundTextures {
  let g = perDevice.get(device);
  if (!g) { g = new GroundTextures(device); perDevice.set(device, g); }
  return g;
}
