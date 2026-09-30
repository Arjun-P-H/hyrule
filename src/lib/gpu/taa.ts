/**
 * Temporal anti-aliasing and upscaling (host side of taa.wgsl.ts).
 * ---------------------------------------------------------------------------
 *   taa.resize(internalW, internalH, outW, outH)   // targets for this size
 *   const j = taa.jitter()                          // NDC offset for this frame's projection
 *   …render the scene into taa.sceneView / taa.depthView with that jitter…
 *   taa.setCamera(yaw, pitch, x, y, z, tanHalf)     // world camera (fp64)
 *   taa.encode(encoder, outputView)                 // resolve, then present
 *
 * History resets on a resize, a new seed, or a camera jump of more than
 * 40 m between frames (teleport, respawn).
 */

import { CAMERA } from '../constants';
import { TAA_BYTES, TAA_WGSL } from './taa.wgsl';

export const SCENE_FORMAT: GPUTextureFormat = 'rgba16float';

/** Halton low-discrepancy value of `i` in base `b`, in [0, 1). */
function halton(i: number, b: number): number {
  let f = 1, r = 0;
  for (let k = i; k > 0; k = Math.floor(k / b)) {
    f /= b;
    r += f * (k % b);
  }
  return r;
}

export class TemporalResolve {
  private device: GPUDevice;
  private outFormat: GPUTextureFormat;
  private layout!: GPUBindGroupLayout;
  private resolvePipe!: GPURenderPipeline;
  private presentPipe!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private ubuf!: GPUBuffer;
  private data = new ArrayBuffer(TAA_BYTES);
  private f = new Float32Array(this.data);
  private scene: GPUTexture | null = null;
  private depth: GPUTexture | null = null;
  private hist: GPUTexture[] = [];
  private histViews: GPUTextureView[] = [];
  private resolveBind: GPUBindGroup[] = [];
  private presentBind: GPUBindGroup[] = [];
  private frame = 0;
  private cur = 0;
  private valid = false;
  private size = [0, 0, 0, 0];
  private prev: { x: number; y: number; z: number; basis: Float32Array } | null = null;
  private jit: [number, number] = [0, 0];
  sceneView: GPUTextureView | null = null;
  depthView: GPUTextureView | null = null;
  /** Weight of the new sample in the history blend. */
  alpha = 0.1;
  /** 0..1, contrast-adaptive sharpening of the presented image. */
  sharpen = 0.5;

  constructor(device: GPUDevice, outFormat: GPUTextureFormat) {
    this.device = device;
    this.outFormat = outFormat;
  }

  async init(): Promise<void> {
    const d = this.device;
    const F = GPUShaderStage.FRAGMENT;
    this.layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 4, visibility: F, sampler: { type: 'filtering' } },
      ],
    });
    const module = d.createShaderModule({ code: TAA_WGSL });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    [this.resolvePipe, this.presentPipe] = await Promise.all([
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module, entryPoint: 'vsFull' },
        fragment: { module, entryPoint: 'fsResolve', targets: [{ format: SCENE_FORMAT }] },
        primitive: { topology: 'triangle-list' },
      }),
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module, entryPoint: 'vsFull' },
        fragment: { module, entryPoint: 'fsPresent', targets: [{ format: this.outFormat }] },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.ubuf = d.createBuffer({ size: TAA_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Allocate the targets for this internal and output size (no-op when unchanged). */
  resize(w: number, h: number, ow: number, oh: number): void {
    if (this.size[0] === w && this.size[1] === h && this.size[2] === ow && this.size[3] === oh) return;
    this.size = [w, h, ow, oh];
    const d = this.device;
    for (const t of [this.scene, this.depth, ...this.hist]) t?.destroy();
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.scene = d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT });
    this.depth = d.createTexture({ size: [w, h], format: 'depth32float', usage: RT });
    this.hist = [0, 1].map(() => d.createTexture({ size: [w, h], format: SCENE_FORMAT, usage: RT }));
    this.sceneView = this.scene.createView();
    this.depthView = this.depth.createView();
    this.histViews = this.hist.map((t) => t.createView());
    // resolveBind[i] reads history i and is drawn into history 1 − i; presentBind[i] shows history i.
    const bind = (hist: GPUTextureView) => d.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: this.sceneView! },
        { binding: 2, resource: this.depthView! },
        { binding: 3, resource: hist },
        { binding: 4, resource: this.sampler },
      ],
    });
    this.resolveBind = this.histViews.map((v) => bind(v));
    this.presentBind = this.histViews.map((v) => bind(v));
    this.valid = false;
  }

  /** This frame's projection jitter in NDC (Halton 2, 3 over 16 frames). */
  jitter(): [number, number] {
    const i = (this.frame % 16) + 1;
    const [w, h] = this.size;
    this.jit = [((halton(i, 2) - 0.5) * 2) / Math.max(w, 1), ((halton(i, 3) - 0.5) * 2) / Math.max(h, 1)];
    return this.jit;
  }

  /** Forget the history (the next frame shows only its own sample). */
  reset(): void {
    this.valid = false;
  }

  /** The camera of the frame just rendered (world metres, fp64). */
  setCamera(yaw: number, pitch: number, x: number, y: number, z: number, tanHalf: number): void {
    const f = this.f;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const fx = sy * cp, fy = sp, fz = cy * cp, rx = cy, rz = -sy;
    const basis = new Float32Array([rx, 0, rz, 0, fy * rz, fz * rx - fx * rz, -fy * rx, 0, fx, fy, fz, 0]);
    f.set(basis, 0);
    const p = this.prev;
    const jump = p ? Math.hypot(x - p.x, y - p.y, z - p.z) > 40 : true;
    f.set(p && !jump ? p.basis : basis, 12);
    f[24] = p && !jump ? x - p.x : 0;
    f[25] = p && !jump ? y - p.y : 0;
    f[26] = p && !jump ? z - p.z : 0;
    f[27] = this.alpha;
    const [w, h, ow, oh] = this.size;
    f[28] = tanHalf; f[29] = w / h; f[30] = CAMERA.near; f[31] = this.valid && !jump ? 1 : 0;
    f[32] = this.jit[0]; f[33] = this.jit[1]; f[34] = this.sharpen; f[35] = 0;
    f[36] = w; f[37] = h; f[38] = 1 / w; f[39] = 1 / h;
    f[40] = ow; f[41] = oh; f[42] = 1 / ow; f[43] = 1 / oh;
    this.device.queue.writeBuffer(this.ubuf, 0, this.data);
    this.prev = { x, y, z, basis };
  }

  /** Resolve the scene into the history, then present it (scaled, sharpened) into `out`. */
  encode(enc: GPUCommandEncoder, out: GPUTextureView): void {
    const src = this.cur, dst = 1 - this.cur;
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.histViews[dst], loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    rp.setPipeline(this.resolvePipe);
    rp.setBindGroup(0, this.resolveBind[src]);
    rp.draw(3);
    rp.end();
    const pp = enc.beginRenderPass({ colorAttachments: [{ view: out, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pp.setPipeline(this.presentPipe);
    pp.setBindGroup(0, this.presentBind[dst]);
    pp.draw(3);
    pp.end();
    this.cur = dst;
    this.valid = true;
    this.frame++;
  }

  destroy(): void {
    for (const t of [this.scene, this.depth, ...this.hist]) t?.destroy();
    this.ubuf?.destroy();
  }
}
