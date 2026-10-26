/**
 * The rocks' GPU resources and passes (host side of rocks.wgsl.ts).
 * ---------------------------------------------------------------------------
 * Owns the rock meshes (one vertex buffer), the instance buffer (a bucket for each kind and level of detail,
 * ROCK_CAP instances each), the buckets' indirect draw arguments and the tiers' parameters. Per frame:
 *   setFrame(fp)   the tiers' lattices round the camera (each cell's absolute index, the camera's share of
 *                  one), the draw arguments afresh (no instances), and the CPU's rocks (fp.rocks, rocks/field.ts
 *                  packRocks): the box of cells it owns round the hero, the cells taken, its list
 *   encode(enc)    the scatter, a thread a cell of every tier, and the CPU's list, a thread a rock
 *   draw(pass)     one indirect draw a bucket, in the main pass
 */

import { ROCK_KINDS, ROCK_LODS, ROCK_VERTEX_BYTES } from '../rocks/models';
import type { FrameParams } from './frame';
import { stamped } from './profile';
import { ROCK_BUCKETS, ROCK_CAP, ROCK_INSTANCE_BYTES, ROCK_LIST_CAP, ROCK_MESHES, ROCK_N, ROCK_PARAMS_VEC4, ROCK_TAKEN_CAP, ROCK_TIERS } from '../rocks/tiers';
import { rockModels } from './rocks.wgsl';

export class Rocks {
  private device: GPUDevice;
  private vbuf: GPUBuffer;
  private inst: GPUBuffer;
  private args: GPUBuffer;
  private params: GPUBuffer;
  private list: GPUBuffer;
  private nList = 0;
  private argsTemplate = new Uint32Array(ROCK_BUCKETS * 4);
  private pdata = new ArrayBuffer(ROCK_PARAMS_VEC4 * 16);
  private pf32 = new Float32Array(this.pdata);
  private pi32 = new Int32Array(this.pdata);
  private offsets = new Uint32Array(ROCK_BUCKETS);
  private scatterPipe!: GPUComputePipeline;
  private listPipe!: GPUComputePipeline;
  private drawPipe!: GPURenderPipeline;
  private shadowPipe: GPURenderPipeline | null = null;
  private scatterBind!: GPUBindGroup;
  private drawBind!: GPUBindGroup;
  /** Density multiplier of every tier (1: as the biomes say). */
  density = 1;

  constructor(device: GPUDevice) {
    this.device = device;
    const B = GPUBufferUsage;
    const m = rockModels();
    this.vbuf = device.createBuffer({ size: m.vertices.byteLength, usage: B.VERTEX | B.COPY_DST });
    device.queue.writeBuffer(this.vbuf, 0, m.vertices);
    this.inst = device.createBuffer({ size: ROCK_BUCKETS * ROCK_CAP * ROCK_INSTANCE_BYTES, usage: B.STORAGE | B.COPY_SRC });
    this.args = device.createBuffer({ size: ROCK_BUCKETS * 16, usage: B.STORAGE | B.INDIRECT | B.COPY_DST | B.COPY_SRC });
    this.params = device.createBuffer({ size: this.pdata.byteLength, usage: B.STORAGE | B.COPY_DST });
    this.list = device.createBuffer({ size: ROCK_LIST_CAP * 64, usage: B.STORAGE | B.COPY_DST });
    // Each bucket draws its kind's mesh at its level: vertex count, no instances yet, first vertex, first instance.
    m.meshes.forEach((mesh, k) => {
      for (let set = 0; set < 2; set++) {
        const b = set * ROCK_MESHES + k;
        this.argsTemplate.set([mesh.count, 0, mesh.first, 0], 4 * b);
        this.offsets[b] = b * ROCK_CAP * ROCK_INSTANCE_BYTES;
      }
    });
  }

  async init(
    module: GPUShaderModule, computeLayout: GPUBindGroupLayout, renderLayout: GPUBindGroupLayout,
    target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState,
    shadowLayout: GPUBindGroupLayout | null = null,
  ): Promise<void> {
    const d = this.device;
    const C = GPUShaderStage.COMPUTE;
    const scatterLayout = d.createBindGroupLayout({
      entries: [
        { binding: 31, visibility: C, buffer: { type: 'storage' } },
        { binding: 32, visibility: C, buffer: { type: 'storage' } },
        { binding: 33, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 34, visibility: C, buffer: { type: 'read-only-storage' } },
      ],
    });
    const drawLayout = d.createBindGroupLayout({
      entries: [{ binding: 30, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage', hasDynamicOffset: true } }],
    });
    const vertex: GPUVertexBufferLayout = {
      arrayStride: ROCK_VERTEX_BYTES,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
        { shaderLocation: 2, offset: 16, format: 'unorm8x4' },
      ],
    };
    // (the shadow pass reads the position only)
    const shadowP = shadowLayout ? d.createRenderPipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [shadowLayout, drawLayout] }),
      vertex: { module, entryPoint: 'vsRockShadow', buffers: [vertex] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBiasSlopeScale: 2 },
    }) : Promise.resolve(null);
    [this.scatterPipe, this.listPipe, this.drawPipe] = await Promise.all([
      d.createComputePipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [computeLayout, scatterLayout] }),
        compute: { module, entryPoint: 'rockScatter' },
      }),
      d.createComputePipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [computeLayout, scatterLayout] }),
        compute: { module, entryPoint: 'rockList' },
      }),
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [renderLayout, drawLayout] }),
        vertex: { module, entryPoint: 'vsRock', buffers: [vertex] },
        fragment: { module, entryPoint: 'fsRock', targets: target },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
      }),
    ]);
    this.shadowPipe = await shadowP;
    this.scatterBind = d.createBindGroup({
      layout: scatterLayout,
      entries: [
        { binding: 31, resource: { buffer: this.inst } }, { binding: 32, resource: { buffer: this.args } },
        { binding: 33, resource: { buffer: this.params } }, { binding: 34, resource: { buffer: this.list } },
      ],
    });
    this.drawBind = d.createBindGroup({
      layout: drawLayout,
      entries: [{ binding: 30, resource: { buffer: this.inst, offset: 0, size: ROCK_CAP * ROCK_INSTANCE_BYTES } }],
    });
  }

  setFrame(fp: FrameParams): void {
    const f = this.pf32, i = this.pi32;
    ROCK_TIERS.forEach((tier, t) => {
      const cx = fp.x / tier.cell, cz = fp.z / tier.cell;
      const fx = Math.floor(cx), fz = Math.floor(cz);
      f[t * 12] = cx - fx; f[t * 12 + 1] = cz - fz; f[t * 12 + 2] = tier.cell; f[t * 12 + 3] = tier.radius;
      i[t * 12 + 4] = fx - ROCK_N / 2; i[t * 12 + 5] = fz - ROCK_N / 2; i[t * 12 + 6] = 0; i[t * 12 + 7] = 0;
      f[t * 12 + 8] = tier.size[0]; f[t * 12 + 9] = tier.size[1]; f[t * 12 + 10] = tier.chance; f[t * 12 + 11] = tier.kinds.length;
    });
    const o = ROCK_TIERS.length * 12;
    const rf = fp.rocks;
    this.nList = rf ? Math.min(rf.nList, ROCK_LIST_CAP) : 0;
    const nTaken = rf ? Math.min(rf.nTaken, ROCK_TAKEN_CAP) : 0;
    f[o] = this.density; f[o + 1] = nTaken; f[o + 2] = this.nList; f[o + 3] = 0;
    // the box of cells the CPU owns a tier (none without its frame: from > to), the cells taken
    const z = o + 4;
    for (let t = 0; t < ROCK_TIERS.length; t++) {
      if (rf) for (let c = 0; c < 4; c++) i[z + t * 4 + c] = rf.zone[t * 4 + c];
      else { i[z + t * 4] = 1; i[z + t * 4 + 1] = 1; i[z + t * 4 + 2] = 0; i[z + t * 4 + 3] = 0; }
    }
    if (rf && nTaken) i.set(rf.taken.subarray(0, nTaken * 4), z + ROCK_TIERS.length * 4);
    this.device.queue.writeBuffer(this.params, 0, this.pdata, 0, (z + ROCK_TIERS.length * 4 + nTaken * 4) * 4);
    if (rf && this.nList) this.device.queue.writeBuffer(this.list, 0, rf.list, 0, this.nList * 64);
    this.device.queue.writeBuffer(this.args, 0, this.argsTemplate);
  }

  encode(enc: GPUCommandEncoder, group0: GPUBindGroup): void {
    const cp = enc.beginComputePass(stamped('rocks.scatter'));
    cp.setPipeline(this.scatterPipe);
    cp.setBindGroup(0, group0);
    cp.setBindGroup(1, this.scatterBind);
    cp.dispatchWorkgroups(ROCK_N / 8, ROCK_N / 8, ROCK_TIERS.length);
    if (this.nList) {
      cp.setPipeline(this.listPipe);
      cp.dispatchWorkgroups(Math.ceil(this.nList / 64));
    }
    cp.end();
  }

  /**
   * The rocks into the trees' shadow map (the shadow pass's group 0 already set): the two finest levels of every kind,
   * their positions only. The finest level alone would make the pebbles' and the stones' shadows cost as much as the boulders'.
   */
  drawShadow(pass: GPURenderPassEncoder): void {
    if (!this.shadowPipe) return;
    pass.setPipeline(this.shadowPipe);
    pass.setVertexBuffer(0, this.vbuf);
    for (let b = ROCK_MESHES; b < ROCK_BUCKETS; b++) { // (the second set: the first is rocks too small to cast)
      if (b % ROCK_LODS === ROCK_LODS - 1) continue;
      pass.setBindGroup(1, this.drawBind, this.offsets, b, 1);
      pass.drawIndirect(this.args, b * 16);
    }
  }

  /** The rocks in the main pass (group 0 already set). */
  draw(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.drawPipe);
    pass.setVertexBuffer(0, this.vbuf);
    for (let b = 0; b < ROCK_BUCKETS; b++) {
      pass.setBindGroup(1, this.drawBind, this.offsets, b, 1);
      pass.drawIndirect(this.args, b * 16);
    }
  }

  /** Instances drawn last frame per kind (debug readback; waits for the GPU). */
  async counts(): Promise<Record<string, number>> {
    const rb = this.device.createBuffer({ size: ROCK_BUCKETS * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.args, 0, rb, 0, ROCK_BUCKETS * 16);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Uint32Array(rb.getMappedRange());
    const out: Record<string, number> = {};
    ROCK_KINDS.forEach((k, ki) => { for (let l = 0; l < ROCK_LODS; l++) out[`${k}${l}`] = [0, ROCK_MESHES].reduce((n, o) => n + Math.min(a[(o + ki * ROCK_LODS + l) * 4 + 1], ROCK_CAP), 0); });
    rb.unmap();
    rb.destroy();
    return out;
  }

  /**
   * The buckets' draw arguments (vertex count, instances, first vertex, first instance a bucket) and instances as of the last
   * frame (debug readback; waits for the GPU): the check of the CPU twin of the big rocks (rocks/collision.ts) reads them.
   */
  async dump(): Promise<{ args: Uint32Array; inst: Float32Array }> {
    const B = GPUBufferUsage;
    const ab = ROCK_BUCKETS * 16, ib = ROCK_BUCKETS * ROCK_CAP * ROCK_INSTANCE_BYTES;
    const rb = this.device.createBuffer({ size: ab + ib, usage: B.COPY_DST | B.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.args, 0, rb, 0, ab);
    enc.copyBufferToBuffer(this.inst, 0, rb, ab, ib);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const all = rb.getMappedRange().slice(0);
    rb.unmap();
    rb.destroy();
    return { args: new Uint32Array(all, 0, ROCK_BUCKETS * 4), inst: new Float32Array(all, ab) };
  }

  destroy(): void {
    for (const b of [this.vbuf, this.inst, this.args, this.params, this.list]) b.destroy();
  }
}
