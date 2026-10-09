/**
 * The living sky's birds on the GPU (host side of fauna.wgsl.ts).
 * ---------------------------------------------------------------------------
 * The flocks round the camera (skyworld/fauna.ts) are listed again when the
 * camera has moved REPICK metres, the floating origin or the seed changed,
 * or every RELIST_MS (so the flocks of Eyries laid out meanwhile join in),
 * and uploaded chunk-local; every bird of them is then one instance of a
 * single draw (FLOCK_BIRDS a flock), flown and shaded by the shader. Birds are
 * a few triangles each, so they are not culled on the CPU.
 */

import { flocksNear, FLOCK_BIRDS, MAX_FLOCKS, type Flock } from '../skyworld/fauna';
import type { FrameParams } from './frame';
import { BIRD_VERTS, FLOCK_BYTES } from './fauna.wgsl';

const REPICK = 150;
const RELIST_MS = 2500;

export class Fauna {
  private device: GPUDevice;
  private buf: GPUBuffer;
  private data = new ArrayBuffer(MAX_FLOCKS * FLOCK_BYTES);
  private f32 = new Float32Array(this.data);
  private u32 = new Uint32Array(this.data);
  private list: Flock[] = [];
  private n = 0;
  private at = new Float64Array(5).fill(NaN);
  private listed = -Infinity;
  private layout1: GPUBindGroupLayout;
  private bind: GPUBindGroup;
  private pipe: GPURenderPipeline | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.buf = device.createBuffer({ size: MAX_FLOCKS * FLOCK_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.layout1 = device.createBindGroupLayout({ entries: [{ binding: 43, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] });
    this.bind = device.createBindGroup({ layout: this.layout1, entries: [{ binding: 43, resource: { buffer: this.buf } }] });
  }

  /** The pipeline (group 0 the renderer's `layout`): opaque, as the structures. */
  async init(module: GPUShaderModule, layout: GPUBindGroupLayout, target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState): Promise<void> {
    this.pipe = await this.device.createRenderPipelineAsync({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, this.layout1] }),
      vertex: { module, entryPoint: 'vsBird' }, fragment: { module, entryPoint: 'fsBird', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
    });
  }

  /** List the flocks round the camera again when it has moved (or now and then), chunk-local to (ox, oz). */
  setFrame(fp: FrameParams, ox: number, oz: number): void {
    const at = this.at, now = performance.now();
    if (Math.hypot(fp.x - at[0], fp.z - at[1]) < REPICK && at[2] === fp.seed && at[3] === ox && at[4] === oz && now - this.listed < RELIST_MS) return;
    at[0] = fp.x; at[1] = fp.z; at[2] = fp.seed; at[3] = ox; at[4] = oz;
    this.listed = now;
    const L = flocksNear(fp.x, fp.z, fp.seed, this.list);
    const f = this.f32, u = this.u32, w = FLOCK_BYTES / 4;
    for (let k = 0; k < L.length; k++) {
      const F = L[k], o = k * w;
      f[o] = F.x - ox; f[o + 1] = F.y; f[o + 2] = F.z - oz; f[o + 3] = F.r;
      f[o + 4] = F.speed;
      u[o + 5] = Math.min(F.count, FLOCK_BIRDS) | (F.species << 8) | (F.night ? 1 << 15 : 0);
      u[o + 6] = F.seed & 0xffff; u[o + 7] = 0;
    }
    this.n = L.length;
    if (this.n) this.device.queue.writeBuffer(this.buf, 0, this.data, 0, this.n * FLOCK_BYTES);
  }

  /** Every bird of the flocks listed (group 0 already set): one instanced draw. */
  draw(pass: GPURenderPassEncoder): void {
    if (!this.pipe || !this.n) return;
    pass.setPipeline(this.pipe);
    pass.setBindGroup(1, this.bind);
    pass.draw(BIRD_VERTS, this.n * FLOCK_BIRDS);
  }

  /** Flocks and birds listed (debug, benches). */
  stats(): { flocks: number; birds: number } {
    let birds = 0;
    for (const F of this.list) birds += Math.min(F.count, FLOCK_BIRDS);
    return { flocks: this.n, birds };
  }

  destroy(): void { this.buf.destroy(); }
}
