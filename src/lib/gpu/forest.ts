/**
 * The forest's GPU resources and passes (host side of forest.wgsl.ts).
 * ---------------------------------------------------------------------------
 * Owns the tree models (one vertex and one index buffer), the cached window
 * of placement slots, the instance buckets with their indirect draw
 * arguments, and two shadow maps: each frame renders one while every surface
 * samples the other, finished the frame before (reading a map in the frame
 * that draws it would stall the GPU until the shadow pass ends). Per frame:
 *   setFrame(fp)          move the tile window (queue the tiles that entered),
 *                         pack the uniform (camera tile, shadow box snapped to
 *                         shadow texels, wind), reset the bucket counts
 *   encode(enc, group0)   placement for new tiles, culling, the bucket clamp,
 *                         then the shadow pass
 *   draw(pass)            one indexed indirect draw per bucket, in the main pass
 */

import { LIGHT, VALLEY } from '../constants';
import { buildTreeModels, VERTEX_BYTES, type TreeModels } from '../forest/models';
import { CELL, LODS } from '../forest/species';
import { TILE_CELLS, TileWindow } from '../forest/tiles';
import { islandGround } from '../skyIslands';
import { elevation } from '../terrainMath';
import type { FrameParams } from './frame';
import { ARGS_BYTES, BUCKETS, CAP, FOREST_BYTES, GROUPS, GROUP_LEVEL, INSTANCES, INSTANCE_BYTES, KINDS, REGION, SLOT_BYTES, type ForestSpec } from './forest.wgsl';

const TILE = CELL * TILE_CELLS;
/** Half depth (m) of the shadow box along the sun's direction. */
const SHADOW_DEPTH = 400;

type V3 = [number, number, number];
const norm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// Light axes: forward away from the sun, right and up across it.
const LF = norm([-LIGHT.sunDir[0], -LIGHT.sunDir[1], -LIGHT.sunDir[2]]);
const LR = norm(cross(LF, [0, 1, 0]));
const LU = cross(LR, LF);

export class Forest {
  readonly spec: ForestSpec;
  readonly models: TreeModels;
  private device: GPUDevice;
  private window: TileWindow;
  private ubuf: GPUBuffer;
  private data = new ArrayBuffer(FOREST_BYTES);
  private f32 = new Float32Array(this.data);
  private i32 = new Int32Array(this.data);
  private vbuf: GPUBuffer;
  private ibuf: GPUBuffer;
  private slots: GPUBuffer;
  private tiles: GPUBuffer;
  private jobs: GPUBuffer;
  private inst: GPUBuffer;
  private args: GPUBuffer;
  private argsTemplate: Uint32Array<ArrayBuffer>;
  private shadows: GPUTexture[];
  private shadowViews: GPUTextureView[];
  /** The map rendered this frame; the other one is sampled (round the centre it was drawn at, lastCentre). */
  private renderIdx = 0;
  private shadowSampler: GPUSampler;
  private genPipe!: GPUComputePipeline;
  private cullPipe!: GPUComputePipeline;
  private clampPipe!: GPUComputePipeline;
  private treePipe!: GPURenderPipeline;
  private fadePipe!: GPURenderPipeline;
  private shadowPipe!: GPURenderPipeline;
  private genBind!: GPUBindGroup;
  private cullBind!: GPUBindGroup;
  private drawBinds: GPUBindGroup[] = [];
  private shadowBind!: GPUBindGroup;
  private jobCount = 0;
  private frame = 0;
  /** Each bucket's dynamic offset into the instance buffer (index group · KINDS + kind): no allocation per draw. */
  private offsets = new Uint32Array(BUCKETS);
  private lastCentre = new Float64Array(3);
  private hasLast = false;

  constructor(device: GPUDevice, spec: ForestSpec) {
    this.device = device;
    this.spec = spec;
    this.models = buildTreeModels();
    this.window = new TileWindow(spec.side);
    const d = device;
    const B = GPUBufferUsage;
    this.ubuf = d.createBuffer({ size: FOREST_BYTES, usage: B.UNIFORM | B.COPY_DST });
    this.vbuf = d.createBuffer({ size: this.models.vertices.byteLength, usage: B.VERTEX | B.COPY_DST });
    d.queue.writeBuffer(this.vbuf, 0, this.models.vertices);
    this.ibuf = d.createBuffer({ size: this.models.indices.byteLength, usage: B.INDEX | B.COPY_DST });
    d.queue.writeBuffer(this.ibuf, 0, this.models.indices);
    const tiles = spec.side * spec.side;
    this.slots = d.createBuffer({ size: tiles * TILE_CELLS * TILE_CELLS * SLOT_BYTES, usage: B.STORAGE | B.COPY_SRC });
    this.tiles = d.createBuffer({ size: tiles * 16, usage: B.STORAGE | B.COPY_SRC });
    this.jobs = d.createBuffer({ size: tiles * 16, usage: B.STORAGE | B.COPY_DST });
    this.inst = d.createBuffer({ size: INSTANCES * INSTANCE_BYTES, usage: B.STORAGE });
    this.args = d.createBuffer({ size: BUCKETS * ARGS_BYTES, usage: B.STORAGE | B.INDIRECT | B.COPY_DST | B.COPY_SRC });
    // Each bucket draws its (species, shape)'s mesh at its group's level.
    this.argsTemplate = new Uint32Array(BUCKETS * 5);
    for (let g = 0; g < GROUPS; g++) {
      for (let k = 0; k < KINDS; k++) {
        const m = this.models.meshes[k * LODS + GROUP_LEVEL[g]];
        this.argsTemplate.set([m.indexCount, 0, m.firstIndex, m.baseVertex, 0], (g * KINDS + k) * 5);
        this.offsets[g * KINDS + k] = (REGION[g] + k * CAP[g]) * INSTANCE_BYTES;
      }
    }
    const res = Math.max(spec.shadowRes, 1);
    this.shadows = [0, 1].map(() => d.createTexture({ size: [res, res], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
    this.shadowViews = this.shadows.map((t) => t.createView());
    // Both maps start empty (nothing casts a shadow until the first one is drawn).
    const enc = d.createCommandEncoder();
    for (const view of this.shadowViews) {
      enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } }).end();
    }
    d.queue.submit([enc.finish()]);
    this.shadowSampler = d.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  }

  /** Entries the forest adds to group 0: its uniform, the shadow map and its comparison sampler. */
  static layoutEntries(): GPUBindGroupLayoutEntry[] {
    const VFC = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    return [
      { binding: 8, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: 9, visibility: VFC, texture: { sampleType: 'depth' } },
      { binding: 16, visibility: VFC, sampler: { type: 'comparison' } },
    ];
  }

  /** Group-0 entries for frames that sample shadow map `which` (see sampled()). */
  bindEntries(which: number): GPUBindGroupEntry[] {
    return [
      { binding: 8, resource: { buffer: this.ubuf } },
      { binding: 9, resource: this.shadowViews[which] },
      { binding: 16, resource: this.shadowSampler },
    ];
  }

  /** The shadow map this frame samples (pick the group-0 bind group made with it). */
  sampled(): number {
    return 1 - this.renderIdx;
  }

  async init(
    module: GPUShaderModule, computeLayout: GPUBindGroupLayout, renderLayout: GPUBindGroupLayout, frameBuf: GPUBuffer,
    target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState,
  ): Promise<void> {
    const d = this.device;
    const C = GPUShaderStage.COMPUTE, V = GPUShaderStage.VERTEX;
    const storage = (binding: number, type: GPUBufferBindingType = 'storage') => ({ binding, visibility: C, buffer: { type } });
    const genLayout = d.createBindGroupLayout({ entries: [storage(23), storage(24), storage(25, 'read-only-storage')] });
    const cullLayout = d.createBindGroupLayout({ entries: [storage(21), storage(22), storage(23), storage(24)] });
    const drawLayout = d.createBindGroupLayout({ entries: [{ binding: 20, visibility: V, buffer: { type: 'read-only-storage', hasDynamicOffset: true } }] });
    const shadowLayout = d.createBindGroupLayout({ entries: [0, 8].map((binding) => ({ binding, visibility: V, buffer: { type: 'uniform' as const } })) });
    const vertex: GPUVertexBufferLayout = {
      arrayStride: VERTEX_BYTES,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
        { shaderLocation: 2, offset: 16, format: 'unorm8x4' },
      ],
    };
    const cl = (layout: GPUBindGroupLayout) => d.createPipelineLayout({ bindGroupLayouts: [computeLayout, layout] });
    const treeDesc = (fade: boolean): GPURenderPipelineDescriptor => ({
      layout: d.createPipelineLayout({ bindGroupLayouts: [renderLayout, drawLayout] }),
      vertex: { module, entryPoint: 'vsTree', buffers: [vertex] },
      fragment: { module, entryPoint: 'fsTree', targets: target, constants: { TREE_FADE: fade ? 1 : 0 } },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
    });
    [this.genPipe, this.cullPipe, this.clampPipe, this.treePipe, this.fadePipe, this.shadowPipe] = await Promise.all([
      d.createComputePipelineAsync({ layout: cl(genLayout), compute: { module, entryPoint: 'forestGen' } }),
      d.createComputePipelineAsync({ layout: cl(cullLayout), compute: { module, entryPoint: 'forestCull' } }),
      d.createComputePipelineAsync({ layout: cl(cullLayout), compute: { module, entryPoint: 'forestClamp' } }),
      d.createRenderPipelineAsync(treeDesc(false)),
      d.createRenderPipelineAsync(treeDesc(true)),
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [shadowLayout, drawLayout] }),
        vertex: { module, entryPoint: 'vsTreeShadow', buffers: [vertex] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBiasSlopeScale: 2 },
      }),
    ]);
    this.genBind = d.createBindGroup({ layout: genLayout, entries: [
      { binding: 23, resource: { buffer: this.slots } }, { binding: 24, resource: { buffer: this.tiles } }, { binding: 25, resource: { buffer: this.jobs } },
    ] });
    this.cullBind = d.createBindGroup({ layout: cullLayout, entries: [
      { binding: 21, resource: { buffer: this.inst } }, { binding: 22, resource: { buffer: this.args } },
      { binding: 23, resource: { buffer: this.slots } }, { binding: 24, resource: { buffer: this.tiles } },
    ] });
    this.drawBinds = CAP.map((cap) => d.createBindGroup({ layout: drawLayout, entries: [{ binding: 20, resource: { buffer: this.inst, offset: 0, size: cap * INSTANCE_BYTES } }] }));
    this.shadowBind = d.createBindGroup({ layout: shadowLayout, entries: [
      { binding: 0, resource: { buffer: frameBuf } }, { binding: 8, resource: { buffer: this.ubuf } },
    ] });
  }

  /** Regenerate every tile next frame. */
  invalidate(): void {
    this.window.invalidate();
    this.hasLast = false;
  }

  setFrame(fp: FrameParams): void {
    const tx = Math.floor(fp.x / TILE), tz = Math.floor(fp.z / TILE);
    const jobs = this.window.update(tx, tz, fp.seed);
    this.jobCount = jobs.length;
    if (jobs.length) {
      const j = new Int32Array(jobs.length * 4);
      jobs.forEach(([x, z, slot], k) => j.set([x, z, slot, 0], 4 * k));
      this.device.queue.writeBuffer(this.jobs, 0, j);
    }
    const f = this.f32, i = this.i32;
    i[0] = tx; i[1] = tz; i[2] = 0; i[3] = 0;
    f[4] = fp.x - tx * TILE; f[5] = fp.z - tz * TILE; f[6] = this.spec.density; f[7] = (this.frame * 0.618034) % 1;
    // Shadow box: ahead of the camera, on the surface under it, snapped to whole texels in light space.
    // This frame draws the map into one texture; surfaces sample the other, drawn last frame round lastCentre.
    this.renderIdx = 1 - this.renderIdx;
    const R = this.spec.shadowRange;
    const cx = fp.x + Math.sin(fp.yaw) * 0.35 * R, cz = fp.z + Math.cos(fp.yaw) * 0.35 * R;
    const isl = islandGround(fp.x, fp.y, fp.z, fp.seed);
    const cy = Number.isFinite(isl) ? isl : Math.max(elevation(cx, cz, fp.seed, 8), VALLEY.wLevel);
    const texel = (2 * R) / Math.max(this.spec.shadowRes, 1);
    const lx = cx * LR[0] + cy * LR[1] + cz * LR[2], ly = cx * LU[0] + cy * LU[1] + cz * LU[2];
    const sx = Math.round(lx / texel) * texel - lx, sy = Math.round(ly / texel) * texel - ly;
    const nx = cx + LR[0] * sx + LU[0] * sy, ny = cy + LR[1] * sx + LU[1] * sy, nz = cz + LR[2] * sx + LU[2] * sy;
    const L = this.lastCentre;
    if (!this.hasLast) { L[0] = fp.x + 1e7; L[1] = fp.y; L[2] = fp.z; } // no map yet: sample nothing
    f[8] = L[0] - fp.x; f[9] = L[1] - fp.y; f[10] = L[2] - fp.z;
    f[11] = R;
    f[28] = nx - fp.x; f[29] = ny - fp.y; f[30] = nz - fp.z; f[31] = 0;
    L[0] = nx; L[1] = ny; L[2] = nz;
    this.hasLast = true;
    f.set(LR, 12); f[15] = SHADOW_DEPTH;
    f.set(LU, 16); f[19] = texel;
    f.set(LF, 20); f[23] = 0;
    f[24] = 0.87; f[25] = 0.49; f[26] = 0.4; f[27] = fp.time;
    this.device.queue.writeBuffer(this.ubuf, 0, this.data);
    this.device.queue.writeBuffer(this.args, 0, this.argsTemplate);
    this.frame++;
  }

  /** Placement for the tiles that entered the window, culling into the buckets, then the shadow map. */
  encode(enc: GPUCommandEncoder, group0: GPUBindGroup): void {
    const cp = enc.beginComputePass();
    cp.setBindGroup(0, group0);
    if (this.jobCount) {
      cp.setPipeline(this.genPipe);
      cp.setBindGroup(1, this.genBind);
      cp.dispatchWorkgroups(this.jobCount);
      this.jobCount = 0;
    }
    cp.setPipeline(this.cullPipe);
    cp.setBindGroup(1, this.cullBind);
    cp.dispatchWorkgroups(Math.ceil((this.spec.side * this.spec.side * TILE_CELLS * TILE_CELLS) / 64));
    cp.setPipeline(this.clampPipe);
    cp.dispatchWorkgroups(Math.ceil(BUCKETS / 64));
    cp.end();
    if (!this.spec.shadowRes) return;
    const sp = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowViews[this.renderIdx], depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    sp.setPipeline(this.shadowPipe);
    sp.setBindGroup(0, this.shadowBind);
    sp.setVertexBuffer(0, this.vbuf);
    sp.setIndexBuffer(this.ibuf, 'uint32');
    this.drawGroup(sp, 3);
    this.drawGroup(sp, 7);
    sp.end();
  }

  /** One indexed indirect draw per bucket of group g (pipeline, group 0 and buffers already set). */
  private drawGroup(pass: GPURenderPassEncoder, g: number): void {
    for (let k = 0; k < KINDS; k++) {
      pass.setBindGroup(1, this.drawBinds[g], this.offsets, g * KINDS + k, 1);
      pass.drawIndexedIndirect(this.args, (g * KINDS + k) * ARGS_BYTES);
    }
  }

  /** The trees in the main pass (group 0 already set): whole trees, then those cross-fading. */
  draw(pass: GPURenderPassEncoder): void {
    pass.setVertexBuffer(0, this.vbuf);
    pass.setIndexBuffer(this.ibuf, 'uint32');
    pass.setPipeline(this.treePipe);
    for (let g = 0; g < 3; g++) this.drawGroup(pass, g);
    pass.setPipeline(this.fadePipe);
    for (let g = 4; g < 7; g++) this.drawGroup(pass, g);
  }

  /** Instances drawn last frame: levels 0 + 1, level 2, shadow casters (fading trees counted at both levels; debug readback). */
  async counts(): Promise<[number, number, number]> {
    const rb = this.device.createBuffer({ size: BUCKETS * ARGS_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.args, 0, rb, 0, BUCKETS * ARGS_BYTES);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Uint32Array(rb.getMappedRange());
    const group = (g: number) => { let s = 0; for (let k = 0; k < KINDS; k++) s += a[(g * KINDS + k) * 5 + 1]; return s; };
    const out: [number, number, number] = [group(0) + group(1) + group(4) + group(5), group(2) + group(6), group(3) + group(7)];
    rb.unmap();
    rb.destroy();
    return out;
  }

  /** The placement window as generated: tiles (side² × vec4i), then slots (side² × 64 × vec4u) (debug readback). */
  async readSlots(): Promise<ArrayBuffer> {
    const tb = this.spec.side * this.spec.side * 16, sb = tb * TILE_CELLS * TILE_CELLS;
    const rb = this.device.createBuffer({ size: tb + sb, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.tiles, 0, rb, 0, tb);
    enc.copyBufferToBuffer(this.slots, 0, rb, tb, sb);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0);
    rb.unmap();
    rb.destroy();
    return out;
  }

  destroy(): void {
    for (const b of [this.ubuf, this.vbuf, this.ibuf, this.slots, this.tiles, this.jobs, this.inst, this.args]) b.destroy();
    for (const t of this.shadows) t.destroy();
  }
}
