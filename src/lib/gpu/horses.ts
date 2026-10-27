/**
 * Skinned models' GPU resources and passes: the horses (host side of horses.wgsl.ts) and the colossi (colossus.wgsl.ts).
 * ---------------------------------------------------------------------------
 * The model (horse/model.ts loadModel) and its texture load in the background; until they are in, nothing is
 * drawn. Per frame (setFrame) the instances the CPU posed (FrameParams.horses: horse/horses.ts pack;
 * FrameParams.colossi: colossus/colossi.ts pack) are uploaded; they are instances of one indexed draw in the main
 * pass (draw: the vanishing ones, last in the list, with the dissolving fragment shader), and of another into the
 * trees' shadow map (drawShadow).
 */

import { COLOSSI_DRAWN, COLOSSUS_JOINTS, COLOSSUS_STRIDE, COLOSSUS_TEXTURE_URL, COLOSSUS_URL } from '../colossus/colossi';
import { HORSE_JOINTS, HORSE_STRIDE, MAX_DRAWN } from '../horse/horses';
import { HORSE_TEXTURE_URL, HORSE_VERTEX_BYTES, loadHorse, loadModel, type HorseModel } from '../horse/model';

/** What a skinned model is drawn from: its model, texture, sizes, the shader's entry points (vs<entry>, fs<entry>…) and first binding. */
type SkinnedSpec = {
  name: string; load: () => Promise<HorseModel>; texture: string;
  joints: number; stride: number; maxDrawn: number; entry: string; binding: number;
};
export const HORSES_SPEC: SkinnedSpec = {
  name: 'horses', load: loadHorse, texture: HORSE_TEXTURE_URL, joints: HORSE_JOINTS, stride: HORSE_STRIDE, maxDrawn: MAX_DRAWN, entry: 'Horse', binding: 70,
};
export const COLOSSI_SPEC: SkinnedSpec = {
  name: 'colossi', load: () => loadModel(COLOSSUS_URL), texture: COLOSSUS_TEXTURE_URL, joints: COLOSSUS_JOINTS, stride: COLOSSUS_STRIDE,
  maxDrawn: COLOSSI_DRAWN, entry: 'Colossus', binding: 73,
};
type SkinnedFrame = { data: Float32Array; count: number; fading: number };

const VERTEX: GPUVertexBufferLayout = {
  arrayStride: HORSE_VERTEX_BYTES,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x3' },
    { shaderLocation: 2, offset: 24, format: 'float32x2' },
    { shaderLocation: 3, offset: 32, format: 'uint8x4' },
    { shaderLocation: 4, offset: 36, format: 'unorm8x4' },
    { shaderLocation: 5, offset: 40, format: 'unorm8x4' },
  ],
};

export class SkinnedGpu {
  private device: GPUDevice;
  private spec: SkinnedSpec;
  private data: GPUBuffer;
  private vbuf: GPUBuffer | null = null;
  private ibuf: GPUBuffer | null = null;
  private indexFormat: GPUIndexFormat = 'uint16';
  private indexCount = 0;
  private bind: GPUBindGroup | null = null;
  private layout!: GPUBindGroupLayout;
  private pipe: GPURenderPipeline | null = null;
  private fadePipe: GPURenderPipeline | null = null;
  private fading = 0;
  private shadowPipe: GPURenderPipeline | null = null;
  private count = 0;
  private destroyed = false;

  constructor(device: GPUDevice, spec: SkinnedSpec) {
    this.device = device;
    this.spec = spec;
    this.data = device.createBuffer({ size: spec.maxDrawn * spec.stride * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  async init(
    module: GPUShaderModule, renderLayout: GPUBindGroupLayout, target: GPUColorTargetState[], depth: GPUDepthStencilState,
    multisample: GPUMultisampleState, shadowLayout: GPUBindGroupLayout | null,
  ): Promise<void> {
    const d = this.device;
    const V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT, b = this.spec.binding, e = this.spec.entry;
    this.layout = d.createBindGroupLayout({
      entries: [
        { binding: b, visibility: V, buffer: { type: 'read-only-storage' } },
        { binding: b + 1, visibility: F, texture: { sampleType: 'float' } },
        { binding: b + 2, visibility: F, sampler: { type: 'filtering' } },
      ],
    });
    const main = (fs: string) => d.createRenderPipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [renderLayout, this.layout] }),
      vertex: { module, entryPoint: `vs${e}`, buffers: [VERTEX] },
      fragment: { module, entryPoint: fs, targets: target },
      // (the model's frame is mirrored into the game's, the winding with it: both faces drawn)
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
    });
    const [pipe, fadePipe, shadowPipe] = await Promise.all([
      main(`fs${e}`), main(`fs${e}Fade`),
      shadowLayout ? d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [shadowLayout, this.layout] }),
        vertex: { module, entryPoint: `vs${e}Shadow`, buffers: [VERTEX] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBiasSlopeScale: 2 },
      }) : Promise.resolve(null),
    ]);
    this.pipe = pipe;
    this.fadePipe = fadePipe;
    this.shadowPipe = shadowPipe;
    void this.load();
  }

  /** The mesh and the texture, in the background (a failure leaves the model undrawn). */
  private async load(): Promise<void> {
    const d = this.device, url = this.spec.texture;
    try {
      const [m, img] = await Promise.all([
        this.spec.load(),
        fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.blob(); }).then((b) => createImageBitmap(b)),
      ]);
      if (this.destroyed) { img.close(); return; }
      if (m.joints !== this.spec.joints) throw new Error(`${this.spec.name}: ${m.joints} joints`);
      const levels = Math.floor(Math.log2(Math.max(img.width, img.height))) + 1;
      const tex = d.createTexture({
        size: [img.width, img.height], format: 'rgba8unorm-srgb', mipLevelCount: levels,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      for (let l = 0; l < levels; l++) {
        const w = Math.max(1, img.width >> l), h = Math.max(1, img.height >> l);
        const src = l ? await createImageBitmap(img, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' }) : img;
        d.queue.copyExternalImageToTexture({ source: src }, { texture: tex, mipLevel: l }, [w, h]);
        if (l) src.close();
      }
      img.close();
      this.vbuf = d.createBuffer({ size: m.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(this.vbuf, 0, m.vertices.buffer as ArrayBuffer, m.vertices.byteOffset, m.vertices.byteLength);
      const ib = m.indices;
      this.ibuf = d.createBuffer({ size: (ib.byteLength + 3) & ~3, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(this.ibuf, 0, ib.buffer, ib.byteOffset, (ib.byteLength + 3) & ~3);
      this.indexFormat = ib instanceof Uint32Array ? 'uint32' : 'uint16';
      this.indexCount = ib.length;
      const sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8 });
      this.bind = d.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: this.spec.binding, resource: { buffer: this.data } },
          { binding: this.spec.binding + 1, resource: tex.createView() },
          { binding: this.spec.binding + 2, resource: sampler },
        ],
      });
    } catch (e) {
      console.warn(`${this.spec.name}: not drawn —`, e);
    }
  }

  setFrame(f: SkinnedFrame | null | undefined): void {
    this.count = f ? Math.min(f.count, this.spec.maxDrawn) : 0;
    this.fading = f ? Math.min(f.fading, this.count) : 0;
    if (this.count && this.bind) this.device.queue.writeBuffer(this.data, 0, f!.data.buffer, f!.data.byteOffset, this.count * this.spec.stride * 4);
  }

  /** Instances [first, first + n) with `pipe`. */
  private drawWith(pass: GPURenderPassEncoder, pipe: GPURenderPipeline | null, first: number, n: number): void {
    if (!pipe || !this.bind || n <= 0) return;
    pass.setPipeline(pipe);
    pass.setBindGroup(1, this.bind);
    pass.setVertexBuffer(0, this.vbuf!);
    pass.setIndexBuffer(this.ibuf!, this.indexFormat);
    pass.drawIndexed(this.indexCount, n, 0, 0, first);
  }

  /** Every instance in the main pass (group 0 already set): the whole ones, then the vanishing. */
  draw(pass: GPURenderPassEncoder): void {
    const whole = this.count - this.fading;
    this.drawWith(pass, this.pipe, 0, whole);
    this.drawWith(pass, this.fadePipe, whole, this.fading);
  }

  /** The whole ones into the trees' shadow map (the shadow pass's group 0 already set). */
  drawShadow(pass: GPURenderPassEncoder): void { this.drawWith(pass, this.shadowPipe, 0, this.count - this.fading); }

  destroy(): void {
    this.destroyed = true;
    this.data.destroy();
    this.vbuf?.destroy();
    this.ibuf?.destroy();
  }
}
