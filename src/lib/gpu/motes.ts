/**
 * The motes round the camera on the GPU (host side of motes.wgsl.ts): what
 * drifts in the air here (skyworld/motes.ts) is looked up a few times a
 * second into a small uniform; every mote of the three kinds (what falls or
 * rises, what flies by day, what by dusk) is one instance of a single
 * blended draw, after the effects.
 */

import { airAt, type Air } from '../skyworld/motes';
import type { FrameParams } from './frame';
import { MOTES_BYTES, MOTES_N } from './motes.wgsl';

/** How often the air is looked up again (ms), and the wind the motes drift on (x, z; m/s). */
const LOOK_MS = 250;
const WIND: readonly [number, number] = [0.87 * 1.4, 0.49 * 1.4];

export class Motes {
  private device: GPUDevice;
  private buf: GPUBuffer;
  private data = new Float32Array(MOTES_BYTES / 4);
  private air: Air = { fall: 0, fallK: 0, day: 0, dayK: 0, dusk: 0, duskK: 0, ground: 0 };
  private looked = -Infinity;
  private any = false;
  private layout1: GPUBindGroupLayout;
  private bind: GPUBindGroup;
  private pipe: GPURenderPipeline | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.buf = device.createBuffer({ size: MOTES_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.layout1 = device.createBindGroupLayout({ entries: [{ binding: 44, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }] });
    this.bind = device.createBindGroup({ layout: this.layout1, entries: [{ binding: 44, resource: { buffer: this.buf } }] });
  }

  /** The pipeline (group 0 the renderer's `layout`): blended premultiplied, tested against depth, not writing it. */
  async init(module: GPUShaderModule, layout: GPUBindGroupLayout, target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState): Promise<void> {
    const blend: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' };
    this.pipe = await this.device.createRenderPipelineAsync({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout, this.layout1] }),
      vertex: { module, entryPoint: 'vsMote' }, fragment: { module, entryPoint: 'fsMote', targets: [{ format: target[0].format, blend: { color: blend, alpha: blend } }] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: false }, multisample,
    });
  }

  /** Look up the air round the camera again (a few times a second). */
  setFrame(fp: FrameParams): void {
    const now = performance.now();
    if (now - this.looked < LOOK_MS) return;
    this.looked = now;
    const a = airAt(fp.x, fp.y, fp.z, fp.seed, this.air);
    const d = this.data;
    d[0] = a.fallK; d[1] = a.dayK; d[2] = a.duskK; d[3] = a.ground;
    d[4] = a.fall; d[5] = a.day; d[6] = a.dusk; d[7] = 0;
    d[8] = WIND[0]; d[9] = WIND[1]; d[10] = 0; d[11] = 0;
    this.any = a.fall + a.day + a.dusk > 0;
    this.device.queue.writeBuffer(this.buf, 0, d);
  }

  /** Every mote of the three kinds (group 0 already set): one instanced draw. */
  draw(pass: GPURenderPassEncoder): void {
    if (!this.pipe || !this.any) return;
    pass.setPipeline(this.pipe);
    pass.setBindGroup(1, this.bind);
    pass.draw(4, 3 * MOTES_N);
  }

  destroy(): void { this.buf.destroy(); }
}
