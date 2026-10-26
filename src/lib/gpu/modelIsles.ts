/**
 * The land's model isles' GPU resources and passes (host side of modelIsles.wgsl.ts).
 * ---------------------------------------------------------------------------
 * The models (skymodels.ts loadSkyModels) and their base colours load in the background; until they are in,
 * nothing is drawn. Per frame (setFrame) the isles within SKY_MODELS.range (re-picked every REPICK m) and in the
 * view each draw every part (a part with a `near` only within it) at a level of detail by distance: packed into one
 * uniform, grouped by model, part and level, one instanced draw a group — the solids and the ivy (cut out) in the
 * main pass (draw), the water after the land's (drawWater). Within PROP_RANGE an isle's props are drawn too, chunk
 * by chunk (each chunk in range of its prototype and in the view), a prototype's level by the chunk's distance; each
 * (isle, prototype, level) takes a slot (the isle's place and the prototype's box) by a dynamic offset.
 * The isles' waterfalls (within FALL_RANGE) are the sky realm's falls and mist, drawn last (drawFalls). The plants
 * of the isles near enough for the forest's window are written for the forest (plantsEntry): it grows them as its
 * sky layer, and is told to regrow (plantsVersion) when that set of isles changes.
 */

import { SKY_MODELS } from '../constants';
import { SPECIES } from '../forest/species';
import { SKY_FLORA } from '../skyworld/biomes';
import { loadSkyModels, modelIslesNear, type ModelIsle, type SkyMesh, type SkyModel } from '../skymodels';
import type { FrameParams } from './frame';
import {
  MODEL_FALLS_MAX, MODEL_ISLES_MAX, MODEL_PLANT_ISLES, MODEL_PLANTS_BYTES, MODEL_PLANTS_MAX, MODEL_PROP_SLOT_BYTES, MODEL_PROP_SLOTS,
} from './modelIsles.wgsl';
import { FALL_SEGMENTS, MIST_PER_FALL } from './waterfalls.wgsl';

/** The isles are re-picked after the camera moves this far (m). */
const REPICK = 250;
/** A part's levels of detail: the finest within LOD_NEAR[0] m, the next within LOD_NEAR[1], the coarsest beyond. */
const LOD_NEAR = [5000, 20000];
const INST_BYTES = 48;
/** Props are drawn for isles within this distance (m). */
const PROP_RANGE = 20000;
/** Per prototype (by name): its chunks are drawn within the first distance (m), its levels change at the others. */
const PROP_LODS: Record<string, number[]> = { pinnacle: [PROP_RANGE, 1700, 6000], baluster: [450] };
/** Waterfalls are drawn for isles within this distance (m), the nearest first (beyond, a fall is a thread). */
const FALL_RANGE = 12000;
/** A fall's speed over the lip (m/s). */
const FALL_SPEED = 6;
/** The forest grows the plants of the isles within this distance (m): its window's reach and an isle's. */
const PLANT_RANGE = 3600;
const FALL_BYTES = 48;

type Mesh = { vbuf: GPUBuffer; ibuf: GPUBuffer; count: number; lo: number[]; ext: number[] };
type Loaded = {
  model: SkyModel; radius: number; tex: GPUTexture;
  parts: Mesh[][]; protos: Mesh[][]; props: GPUBuffer | null;
  bind: GPUBindGroup; propBind: GPUBindGroup | null;
};
type Pipes = { solid: GPURenderPipeline; prop: GPURenderPipeline; leaf: GPURenderPipeline | null; water: GPURenderPipeline; fall: GPURenderPipeline; mist: GPURenderPipeline };

/** A mip chain of an image, each level made by the browser's resampler from the full image. */
async function mipTexture(device: GPUDevice, url: string): Promise<GPUTexture> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`sky model texture ${url}: ${r.status}`);
  const img = await createImageBitmap(await r.blob());
  const levels = Math.floor(Math.log2(Math.max(img.width, img.height))) + 1;
  const tex = device.createTexture({
    size: [img.width, img.height], format: 'rgba8unorm-srgb', mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  for (let l = 0; l < levels; l++) {
    const w = Math.max(1, img.width >> l), h = Math.max(1, img.height >> l);
    const src = l ? await createImageBitmap(img, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' }) : img;
    device.queue.copyExternalImageToTexture({ source: src }, { texture: tex, mipLevel: l }, [w, h]);
    if (l) src.close();
  }
  img.close();
  return tex;
}

/** A white texel (a model coloured by its vertices alone). */
function whiteTexture(device: GPUDevice): GPUTexture {
  const tex = device.createTexture({ size: [1, 1], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1]);
  return tex;
}

export class ModelIsles {
  private device: GPUDevice;
  private inst: GPUBuffer;
  private data = new Float32Array(MODEL_ISLES_MAX * INST_BYTES / 4);
  private slots: GPUBuffer;
  private slotData = new Float32Array(MODEL_PROP_SLOTS * MODEL_PROP_SLOT_BYTES / 4);
  private falls: GPUBuffer;
  private fallData = new Float32Array(MODEL_FALLS_MAX * FALL_BYTES / 4);
  private fallCount = 0;
  private fallBind!: GPUBindGroup;
  private plants: GPUBuffer;
  private plantData = new Float32Array(MODEL_PLANTS_BYTES / 4);
  /** The isles whose plants were last written, and the chunk they were written for. */
  private plantKey = '';
  /** Bumped when the isles whose plants the forest grows change: the forest regrows its window. */
  plantsVersion = 0;
  private sampler: GPUSampler;
  private pipes!: Pipes;
  private layout!: GPUBindGroupLayout;
  private propLayout!: GPUBindGroupLayout;
  private loaded: Loaded[] | null = null;
  private destroyed = false;
  private near: ModelIsle[] = [];
  private nearAt: [number, number, number] = [NaN, NaN, NaN];
  /** This frame's part draws: model, part, level, instance count, first instance (solids and leaves; water). */
  private draws: number[] = [];
  private waterDraws: number[] = [];
  /** This frame's prop draws: model, prototype, level, slot, first prop, count. */
  private propDraws: number[] = [];
  private offs = new Uint32Array(1);

  constructor(device: GPUDevice) {
    this.device = device;
    const B = GPUBufferUsage;
    this.inst = device.createBuffer({ size: this.data.byteLength, usage: B.UNIFORM | B.COPY_DST });
    this.slots = device.createBuffer({ size: this.slotData.byteLength, usage: B.UNIFORM | B.COPY_DST });
    this.falls = device.createBuffer({ size: this.fallData.byteLength, usage: B.STORAGE | B.COPY_DST });
    // (written empty: the forest reads it from its first frame)
    this.plants = device.createBuffer({ size: MODEL_PLANTS_BYTES, usage: B.STORAGE | B.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8 });
  }

  /** Group 0's entry for the forest: the plants of the isles near the camera (modelIsles.wgsl.ts miPlants). */
  static plantsLayoutEntry(): GPUBindGroupLayoutEntry {
    return { binding: 66, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } };
  }

  plantsEntry(): GPUBindGroupEntry {
    return { binding: 66, resource: { buffer: this.plants } };
  }

  /** `forest`: the raster shader has the forest (its leaf sprays light the ivy; without it the ivy is not drawn). */
  async init(module: GPUShaderModule, renderLayout: GPUBindGroupLayout, target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState, forest: boolean): Promise<void> {
    const d = this.device;
    const V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
    const texEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 61, visibility: F, texture: { sampleType: 'float' } },
      { binding: 62, visibility: F, sampler: { type: 'filtering' } },
    ];
    this.layout = d.createBindGroupLayout({ entries: [{ binding: 60, visibility: V, buffer: { type: 'uniform' } }, ...texEntries] });
    this.propLayout = d.createBindGroupLayout({
      entries: [
        { binding: 63, visibility: V, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INST_BYTES } },
        { binding: 64, visibility: V, buffer: { type: 'read-only-storage' } },
        ...texEntries,
      ],
    });
    const fallLayout = d.createBindGroupLayout({ entries: [{ binding: 65, visibility: V | F, buffer: { type: 'read-only-storage' } }] });
    this.fallBind = d.createBindGroup({ layout: fallLayout, entries: [{ binding: 65, resource: { buffer: this.falls } }] });
    const vertex = (entryPoint: string): GPUVertexState => ({
      module, entryPoint,
      buffers: [{
        arrayStride: 20,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'unorm16x4' },
          { shaderLocation: 1, offset: 8, format: 'snorm8x4' },
          { shaderLocation: 2, offset: 12, format: 'unorm16x2' },
          { shaderLocation: 3, offset: 16, format: 'unorm8x4' },
        ],
      }],
    });
    // (premultiplied over what is drawn, as the land's water and the realm's falls)
    const blended: GPUColorTargetState = {
      format: target[0].format,
      blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } },
    };
    const desc = (layout: GPUBindGroupLayout, vs: string, fs: string, targets = target, depthStencil = depth): GPURenderPipelineDescriptor => ({
      layout: d.createPipelineLayout({ bindGroupLayouts: [renderLayout, layout] }),
      vertex: vertex(vs),
      fragment: { module, entryPoint: fs, targets },
      // (the models' faces are not all wound outward: drawn both sides)
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil, multisample,
    });
    const strip = (vs: string, fs: string): GPURenderPipelineDescriptor => ({
      layout: d.createPipelineLayout({ bindGroupLayouts: [renderLayout, fallLayout] }),
      vertex: { module, entryPoint: vs }, fragment: { module, entryPoint: fs, targets: [blended] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: false }, multisample,
    });
    const [solid, prop, leaf, water, fall, mist] = await Promise.all([
      d.createRenderPipelineAsync(desc(this.layout, 'vsModelIsle', 'fsModelIsle')),
      d.createRenderPipelineAsync(desc(this.propLayout, 'vsModelProp', 'fsModelIsle')),
      forest ? d.createRenderPipelineAsync(desc(this.layout, 'vsModelIsle', 'fsModelLeaf')) : Promise.resolve(null),
      d.createRenderPipelineAsync(desc(this.layout, 'vsModelIsle', 'fsModelWater', [blended])),
      d.createRenderPipelineAsync(strip('vsFall', 'fsFall')),
      d.createRenderPipelineAsync(strip('vsMist', 'fsMist')),
    ]);
    this.pipes = { solid, prop, leaf, water, fall, mist };
    // (in the background: the land draws meanwhile)
    void this.load().catch((e) => console.error('model isles failed to load', e));
  }

  private async load(): Promise<void> {
    const d = this.device;
    const models = await loadSkyModels();
    const buf = (data: ArrayBufferView, usage: number) => {
      const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
      if (data.byteLength) d.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
      return b;
    };
    const mesh = (m: SkyMesh): Mesh => ({
      vbuf: buf(m.verts, GPUBufferUsage.VERTEX), ibuf: buf(m.index, GPUBufferUsage.INDEX), count: m.index.length,
      lo: m.lo, ext: m.hi.map((h, k) => h - m.lo[k]),
    });
    const loaded = await Promise.all(models.map(async (model, k): Promise<Loaded> => {
      const tex = model.texture ? await mipTexture(d, `/models/${SKY_MODELS.names[k]}.webp`) : whiteTexture(d);
      const texRes: GPUBindGroupEntry[] = [{ binding: 61, resource: tex.createView() }, { binding: 62, resource: this.sampler }];
      const props = model.props.count ? buf(model.props.data, GPUBufferUsage.STORAGE) : null;
      const { boxMin: lo, boxMax: hi } = model;
      return {
        model, tex, props,
        radius: 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]),
        parts: model.parts.map((p) => p.lods.map(mesh)),
        protos: model.protos.map((p) => p.lods.map(mesh)),
        bind: d.createBindGroup({ layout: this.layout, entries: [{ binding: 60, resource: { buffer: this.inst } }, ...texRes] }),
        propBind: props && d.createBindGroup({
          layout: this.propLayout,
          entries: [{ binding: 63, resource: { buffer: this.slots, size: INST_BYTES } }, { binding: 64, resource: { buffer: props } }, ...texRes],
        }),
      };
    }));
    if (this.destroyed) { for (const L of loaded) ModelIsles.free(L); return; }
    this.loaded = loaded;
  }

  setFrame(fp: FrameParams, ox: number, oz: number): void {
    this.draws.length = 0;
    this.waterDraws.length = 0;
    this.propDraws.length = 0;
    this.fallCount = 0;
    const L = this.loaded;
    if (!L) return;
    const [nx, nz, ns] = this.nearAt;
    if (ns !== fp.seed || Math.hypot(fp.x - nx, fp.z - nz) > REPICK) {
      modelIslesNear(fp.x, fp.z, SKY_MODELS.range + REPICK, fp.seed, this.near);
      this.nearAt = [fp.x, fp.z, fp.seed];
    }
    // the view's cone (its diagonal), its forward as packFrame's
    const cp = Math.cos(fp.pitch);
    const fx = Math.sin(fp.yaw) * cp, fy = Math.sin(fp.pitch), fz = Math.cos(fp.yaw) * cp;
    const cone = Math.atan(fp.tanHalf * Math.hypot(1, fp.width / Math.max(fp.height, 1)));
    const seen = (dx: number, dy: number, dz: number, r: number) => {
      const dist = Math.hypot(dx, dy, dz);
      return dist <= r || Math.acos(Math.min(1, (dx * fx + dy * fy + dz * fz) / dist)) <= cone + Math.asin(r / dist);
    };
    // the isles in range and in view, with their distances
    const vis: Array<{ m: ModelIsle; dist: number }> = [];
    for (const m of this.near) {
      const R = L[m.model].radius;
      const dx = m.x - fp.x, dy = m.y - fp.y, dz = m.z - fp.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist <= SKY_MODELS.range && seen(dx, dy, dz, R)) vis.push({ m, dist });
    }
    // parts: grouped by model, part and level, in the uniform in that order
    const f = this.data;
    let n = 0;
    L.forEach(({ model, parts }, k) => {
      parts.forEach((lods, part) => {
        const { near, kind } = model.parts[part];
        if (kind === 'leaf' && !this.pipes.leaf) return;
        for (let lod = 0; lod < lods.length; lod++) {
          const first = n;
          for (const { m, dist } of vis) {
            if (m.model !== k || n >= MODEL_ISLES_MAX || (near !== null && dist > near)) continue;
            const want = dist < LOD_NEAR[0] ? 0 : dist < LOD_NEAR[1] ? 1 : 2;
            if (Math.min(want, lods.length - 1) !== lod) continue;
            this.packInst(f, n * (INST_BYTES / 4), m, ox, oz, lods[lod]);
            n++;
          }
          if (n > first) (kind === 'water' ? this.waterDraws : this.draws).push(k, part, lod, n - first, first);
        }
      });
    });
    if (n) this.device.queue.writeBuffer(this.inst, 0, f, 0, n * (INST_BYTES / 4));
    // props: per near isle, per chunk in range and in view
    const sf = this.slotData, stride = MODEL_PROP_SLOT_BYTES / 4;
    let slots = 0;
    for (const { m, dist } of vis) {
      const Lm = L[m.model];
      if (dist > PROP_RANGE || !Lm.props) continue;
      const slotOf = new Map<number, number>();
      const c = Math.cos(m.yaw), s = Math.sin(m.yaw);
      for (const ch of Lm.model.props.chunks) {
        const proto = Lm.model.protos[ch.proto];
        const lods = PROP_LODS[proto.name] ?? [PROP_RANGE];
        // the chunk's middle in the world (the model's turn: modelIsles.wgsl.ts miTurn)
        const wx = m.x + c * ch.c[0] + s * ch.c[2], wy = m.y + ch.c[1], wz = m.z - s * ch.c[0] + c * ch.c[2];
        const dx = wx - fp.x, dy = wy - fp.y, dz = wz - fp.z;
        const cd = Math.max(0, Math.hypot(dx, dy, dz) - ch.r);
        if (cd > lods[0] || !seen(dx, dy, dz, ch.r)) continue;
        let lod = 0;
        while (lod + 1 < lods.length && cd > lods[lod + 1]) lod++;
        lod = Math.min(lod, Lm.protos[ch.proto].length - 1);
        const key = ch.proto * 8 + lod;
        let slot = slotOf.get(key);
        if (slot === undefined) {
          if (slots >= MODEL_PROP_SLOTS) continue;
          slot = slots++;
          slotOf.set(key, slot);
          this.packInst(sf, slot * stride, m, ox, oz, Lm.protos[ch.proto][lod]);
        }
        this.propDraws.push(m.model, ch.proto, lod, slot, ch.first, ch.count);
      }
    }
    if (slots) this.device.queue.writeBuffer(this.slots, 0, sf, 0, slots * stride);
    this.writeFalls(fp, ox, oz);
    this.writePlants(fp, ox, oz);
  }

  /** The falls of the isles within FALL_RANGE, the nearest first (seen or not: a fall hangs far below its isle), chunk-local. */
  private writeFalls(fp: FrameParams, ox: number, oz: number): void {
    const L = this.loaded!, F = this.fallData, W = FALL_BYTES / 4;
    let n = 0;
    const isles = this.near.filter((m) => Math.hypot(m.x - fp.x, m.z - fp.z) <= FALL_RANGE)
      .sort((a, b) => Math.hypot(a.x - fp.x, a.z - fp.z) - Math.hypot(b.x - fp.x, b.z - fp.z));
    for (const m of isles) {
      const c = Math.cos(m.yaw), s = Math.sin(m.yaw);
      for (const fl of L[m.model].model.falls) {
        if (n >= MODEL_FALLS_MAX) break;
        const [lx, ly, lz] = fl.lip;
        const o = n * W;
        F[o] = m.x + c * lx + s * lz - ox; F[o + 1] = m.y + ly; F[o + 2] = m.z - s * lx + c * lz - oz; F[o + 3] = fl.width;
        F[o + 4] = c * fl.dir[0] + s * fl.dir[1]; F[o + 5] = -s * fl.dir[0] + c * fl.dir[1]; F[o + 6] = FALL_SPEED; F[o + 7] = fl.reach;
        F[o + 8] = 11 + 137 * n; F[o + 9] = 0; F[o + 10] = 0; F[o + 11] = 0;
        n++;
      }
    }
    this.fallCount = n;
    if (n) this.device.queue.writeBuffer(this.falls, 0, F, 0, n * W);
  }

  /**
   * The plants of the isles within PLANT_RANGE, chunk-local, for the forest's sky layer (modelIsles.wgsl.ts miPlants);
   * written again when that set of isles or the chunk changes, and the forest regrown when the set does.
   */
  private writePlants(fp: FrameParams, ox: number, oz: number): void {
    const L = this.loaded!;
    const isles = this.near.filter((m) => L[m.model].model.plants.length && Math.hypot(m.x - fp.x, m.z - fp.z) < PLANT_RANGE + L[m.model].radius)
      .sort((a, b) => Math.hypot(a.x - fp.x, a.z - fp.z) - Math.hypot(b.x - fp.x, b.z - fp.z)).slice(0, MODEL_PLANT_ISLES);
    const set = isles.map((m) => `${m.x},${m.z}`).join(';');
    const key = `${set}|${ox},${oz}`;
    if (key === this.plantKey) return;
    if (set !== this.plantKey.split('|')[0]) this.plantsVersion++;
    this.plantKey = key;
    const P = this.plantData;
    P.fill(0);
    const base = 4 * (1 + 2 * MODEL_PLANT_ISLES);
    let j = 0;
    isles.forEach((m, k) => {
      const plants = L[m.model].model.plants;
      const c = Math.cos(m.yaw), s = Math.sin(m.yaw);
      const first = j;
      for (const pl of plants) {
        if (j >= MODEL_PLANTS_MAX) break;
        const [lx, ly, lz] = pl.p;
        const o = base + 8 * j;
        P[o] = m.x + c * lx + s * lz - ox; P[o + 1] = m.z - s * lx + c * lz - oz; P[o + 2] = m.y + ly; P[o + 3] = pl.scale;
        P[o + 4] = Math.max(0, SPECIES.indexOf(pl.species as (typeof SPECIES)[number]));
        P[o + 5] = Math.max(0, SKY_FLORA.indexOf(pl.palette as (typeof SKY_FLORA)[number]));
        j++;
      }
      const h = 4 * (1 + 2 * k);
      P[h] = m.x - ox; P[h + 1] = m.z - oz; P[h + 2] = L[m.model].radius; P[h + 3] = first;
      P[h + 4] = j - first;
    });
    P[0] = isles.length;
    this.device.queue.writeBuffer(this.plants, 0, P, 0, base + 8 * j);
  }

  /** An instance's block (MiInst): the isle's place and turn, the mesh's box. */
  private packInst(f: Float32Array, o: number, m: ModelIsle, ox: number, oz: number, M: Mesh): void {
    f[o] = m.x - ox; f[o + 1] = m.y; f[o + 2] = m.z - oz; f[o + 3] = m.yaw;
    f[o + 4] = M.lo[0]; f[o + 5] = M.lo[1]; f[o + 6] = M.lo[2];
    f[o + 8] = M.ext[0]; f[o + 9] = M.ext[1]; f[o + 10] = M.ext[2];
  }

  /** One instanced draw a group of this frame's part draws `list` (group 0 set; `leaf`: the cut-out ivy's pipeline for leaf parts). */
  private drawParts(pass: GPURenderPassEncoder, list: number[], pipe: GPURenderPipeline): void {
    const L = this.loaded!;
    let current: GPURenderPipeline | null = null;
    for (let i = 0; i < list.length; i += 5) {
      const [k, part, lod, count, first] = list.slice(i, i + 5);
      const p = L[k].model.parts[part].kind === 'leaf' ? this.pipes.leaf! : pipe;
      if (p !== current) { pass.setPipeline(p); current = p; }
      const M = L[k].parts[part][lod];
      pass.setBindGroup(1, L[k].bind);
      pass.setVertexBuffer(0, M.vbuf);
      pass.setIndexBuffer(M.ibuf, 'uint32');
      pass.drawIndexed(M.count, count, 0, 0, first);
    }
  }

  /** The isles' solids, ivy and props in the main pass (group 0 already set). */
  draw(pass: GPURenderPassEncoder): void {
    const L = this.loaded;
    if (!L) return;
    this.drawParts(pass, this.draws, this.pipes.solid);
    if (this.propDraws.length) pass.setPipeline(this.pipes.prop);
    for (let i = 0; i < this.propDraws.length; i += 6) {
      const [k, proto, lod, slot, first, count] = this.propDraws.slice(i, i + 6);
      const M = L[k].protos[proto][lod];
      this.offs[0] = slot * MODEL_PROP_SLOT_BYTES;
      pass.setBindGroup(1, L[k].propBind!, this.offs);
      pass.setVertexBuffer(0, M.vbuf);
      pass.setIndexBuffer(M.ibuf, 'uint32');
      pass.drawIndexed(M.count, count, 0, 0, first);
    }
  }

  /** The isles' pools, after the land's water (group 0 already set). */
  drawWater(pass: GPURenderPassEncoder): void {
    if (this.loaded && this.waterDraws.length) this.drawParts(pass, this.waterDraws, this.pipes.water);
  }

  /** The isles' waterfalls and their mist, last (group 0 already set). */
  drawFalls(pass: GPURenderPassEncoder): void {
    if (!this.fallCount) return;
    pass.setBindGroup(1, this.fallBind);
    pass.setPipeline(this.pipes.fall);
    pass.draw(2 * (FALL_SEGMENTS + 1), this.fallCount);
    pass.setPipeline(this.pipes.mist);
    pass.draw(4, this.fallCount * MIST_PER_FALL);
  }

  private static free(L: Loaded): void {
    L.tex.destroy();
    for (const M of [...L.parts.flat(), ...L.protos.flat()]) { M.vbuf.destroy(); M.ibuf.destroy(); }
    L.props?.destroy();
  }

  destroy(): void {
    this.destroyed = true;
    if (this.loaded) for (const L of this.loaded) ModelIsles.free(L);
    this.loaded = null;
    this.inst.destroy();
    this.slots.destroy();
    this.falls.destroy();
    this.plants.destroy();
  }
}
