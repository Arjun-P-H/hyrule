/**
 * Combat on the GPU (host side of combat.wgsl.ts): the foes, each traced in
 * the scissor rectangle its box covers with its block at a dynamic offset of
 * one uniform buffer; the parts (lit boxes) and the sprites (blended quads),
 * each one instanced draw over a storage buffer refilled every frame.
 */

import { FOE } from '../constants';
import { FOE_BYTES, FOE_STRIDE, PART_BYTES, SPRITE_BYTES, foeWorldBox, foeWorldSphere } from './combat.wgsl';
import type { CombatFrame } from './frame';
import { cornersScissor, sphereScissor } from './scissor';

export const MAX_PARTS = 512;
export const MAX_SPRITES = 768;

export class CombatGpu {
  private device: GPUDevice;
  private foeBuf: GPUBuffer;
  private partBuf: GPUBuffer;
  private spriteBuf: GPUBuffer;
  private foeLayout: GPUBindGroupLayout;
  private partLayout: GPUBindGroupLayout;
  private spriteLayout: GPUBindGroupLayout;
  private foeBind: GPUBindGroup;
  private partBind: GPUBindGroup;
  private spriteBind: GPUBindGroup;
  private foePipe: GPURenderPipeline | null = null;
  private partPipe: GPURenderPipeline | null = null;
  private spritePipe: GPURenderPipeline | null = null;
  private scissors: Array<[number, number, number, number] | null> = new Array(FOE.max).fill(null);
  private nParts = 0;
  private nSprites = 0;
  private size: [number, number] = [2, 2];
  private box = new Float64Array(6);
  private sphere = new Float64Array(4);

  constructor(device: GPUDevice) {
    this.device = device;
    this.foeBuf = device.createBuffer({ size: FOE_STRIDE * FOE.max, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.partBuf = device.createBuffer({ size: PART_BYTES * MAX_PARTS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.spriteBuf = device.createBuffer({ size: SPRITE_BYTES * MAX_SPRITES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.foeLayout = device.createBindGroupLayout({
      entries: [{ binding: 50, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: FOE_BYTES } }],
    });
    this.partLayout = device.createBindGroupLayout({ entries: [{ binding: 51, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] });
    this.spriteLayout = device.createBindGroupLayout({
      entries: [{ binding: 52, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }],
    });
    this.foeBind = device.createBindGroup({ layout: this.foeLayout, entries: [{ binding: 50, resource: { buffer: this.foeBuf, size: FOE_BYTES } }] });
    this.partBind = device.createBindGroup({ layout: this.partLayout, entries: [{ binding: 51, resource: { buffer: this.partBuf } }] });
    this.spriteBind = device.createBindGroup({ layout: this.spriteLayout, entries: [{ binding: 52, resource: { buffer: this.spriteBuf } }] });
  }

  /** The pipelines (group 0 the renderer's `layout`). */
  async init(module: GPUShaderModule, layout: GPUBindGroupLayout, target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState): Promise<void> {
    const d = this.device;
    const blended: GPUColorTargetState[] = [{
      format: target[0].format,
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    }];
    [this.foePipe, this.partPipe, this.spritePipe] = await Promise.all([
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout, this.foeLayout] }),
        vertex: { module, entryPoint: 'vsSky' }, fragment: { module, entryPoint: 'fsFoe', targets: target },
        primitive: { topology: 'triangle-list' }, depthStencil: depth, multisample,
      }),
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout, this.partLayout] }),
        vertex: { module, entryPoint: 'vsPart' }, fragment: { module, entryPoint: 'fsPart', targets: target },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
      }),
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout, this.spriteLayout] }),
        vertex: { module, entryPoint: 'vsSprite' }, fragment: { module, entryPoint: 'fsSprite', targets: blended },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: depth.format, depthWriteEnabled: false, depthCompare: 'greater' }, multisample,
      }),
    ]);
  }

  /**
   * This frame's foes, parts and sprites; F: the frame block's basis (as the
   * renderer packs it), sx, sy the projection's scales, w × h the target.
   */
  setFrame(c: CombatFrame | null | undefined, F: Float32Array, sx: number, sy: number, w: number, h: number): void {
    this.size[0] = w; this.size[1] = h;
    const q = this.device.queue;
    const nf = c ? Math.min(c.nFoes, FOE.max) : 0;
    for (let k = 0; k < FOE.max; k++) {
      this.scissors[k] = null;
      if (k >= nf || !c) continue;
      const o = (k * FOE_STRIDE) / 4;
      foeWorldSphere(c.foes, o, this.sphere);
      const s = sphereScissor(F, this.sphere[0], this.sphere[1], this.sphere[2], this.sphere[3], sx, sy, w, h);
      if (!s) continue;
      foeWorldBox(c.foes, o, this.box);
      const b = this.box;
      this.scissors[k] = cornersScissor(F, b[0], b[1], b[2], b[3], b[4], b[5], sx, sy, w, h, s);
    }
    if (c && nf) q.writeBuffer(this.foeBuf, 0, c.foes.buffer, c.foes.byteOffset, nf * FOE_STRIDE);
    this.nParts = c ? Math.min(c.nParts, MAX_PARTS) : 0;
    if (c && this.nParts) q.writeBuffer(this.partBuf, 0, c.parts.buffer, c.parts.byteOffset, this.nParts * PART_BYTES);
    this.nSprites = c ? Math.min(c.nSprites, MAX_SPRITES) : 0;
    if (c && this.nSprites) q.writeBuffer(this.spriteBuf, 0, c.sprites.buffer, c.sprites.byteOffset, this.nSprites * SPRITE_BYTES);
  }

  /** The foes, each in its rectangle (group 0 already set; the scissor is left whole). */
  drawFoes(pass: GPURenderPassEncoder): void {
    if (!this.foePipe) return;
    let set = false;
    for (let k = 0; k < FOE.max; k++) {
      const r = this.scissors[k];
      if (!r) continue;
      if (!set) { pass.setPipeline(this.foePipe); set = true; }
      pass.setBindGroup(1, this.foeBind, [k * FOE_STRIDE]);
      pass.setScissorRect(r[0], r[1], r[2], r[3]);
      pass.draw(3);
    }
    if (set) pass.setScissorRect(0, 0, this.size[0], this.size[1]);
  }

  /** The lit boxes: arrows, weapons, a camp's logs. */
  drawParts(pass: GPURenderPassEncoder): void {
    if (!this.partPipe || !this.nParts) return;
    pass.setPipeline(this.partPipe);
    pass.setBindGroup(1, this.partBind);
    pass.draw(36, this.nParts);
  }

  /** The blended quads: sparks, smoke, trails, flames (after the opaque draws). */
  drawSprites(pass: GPURenderPassEncoder): void {
    if (!this.spritePipe || !this.nSprites) return;
    pass.setPipeline(this.spritePipe);
    pass.setBindGroup(1, this.spriteBind);
    pass.draw(6, this.nSprites);
  }

  destroy(): void {
    this.foeBuf.destroy();
    this.partBuf.destroy();
    this.spriteBuf.destroy();
  }
}
