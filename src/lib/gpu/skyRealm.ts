/**
 * The sky realm's GPU resources and passes (host side of skyRealm.wgsl.ts
 * and waterfalls.wgsl.ts).
 * ---------------------------------------------------------------------------
 * Every few hundred metres the camera moves, the realm within REALM_NEAR
 * (skyworld/realm.ts) is packed, chunk-local, into one storage buffer: the
 * nearest archipelagos and isles first, floating rocks left out far away; and
 * beyond it, out to REALM_FAR, the islands alone (built a few a frame as they
 * come into range, and packed once all are).
 * Per frame (setFrame):
 *   - the sky rings: when the camera is up among the islands, nested grid
 *     rings round it draw the tops of the islands that reach into their
 *     square (polar tops are clipped out of it);
 *   - culling and levels of detail: each island's polar underside (and top,
 *     unless the rings draw all of it) at one of three densities by how big
 *     it shows; the lakes and streams near enough to see; the falls and their
 *     mist; the arches;
 *   - the shadow map's box: the undersides and arches seen from the sun,
 *     drawn into one of two maps while every surface samples the other
 *     (drawn before). The islands never move, so a map is drawn only when it
 *     would come out different: the sun has turned past SHADOW_TURN since the
 *     one sampled, the camera has gone SHADOW_MOVE from its middle, or the
 *     casters round the camera changed;
 * all of it in one uniform (realmU). Draws: encodeShadow before the main
 * pass; draw (opaque) after the terrain; drawWater after the land's water;
 * drawFalls last.
 */

import { LIGHT } from '../constants';
import type { Archipelago } from '../skyworld/archipelago';
import { SKY_BIOMES } from '../skyworld/biomes';
import { CHANNEL_REACH, maxDrop, outlineMax, type SkyIsland } from '../skyworld/island';
import { buildCell, realmNear, type PendingCell } from '../skyworld/realm';
import type { FrameParams, RealmSpec } from './frame';
import { ringIndices } from './rings';
import {
  archIndices, LAKE_RINGS, LAKE_SEGS, POLAR_FAR, POLAR_LODS, polarIndices, REALM_FAR, REALM_FRAME_BYTES, REALM_LAYOUT,
  REALM_MAX, REALM_NEAR, RL_FLAGS, SKY_RINGS, STREAM_SUB,
} from './skyRealm.wgsl';
import { FALL_SEGMENTS, MIST_PER_FALL } from './waterfalls.wgsl';
import { stamped } from './profile';

/** The realm is re-picked after the camera moves this far (m); within REALM_NEAR all of it, its islands out to REALM_FAR. */
const REPICK = 250;
/**
 * Far cells not built yet are built a few each frame, nearest first, for at most this long (ms; at
 * least one: an archipelago takes ~5 ms), and the realm packed again once all are.
 */
const BUILD_BUDGET_MS = 3;
/** Floating rocks are left out of archipelagos farther than this (m). */
const DEBRIS_RANGE = 6000;
/** Shadow map: texels per side, half width and half depth of its box (m). */
const SHADOW_RES = 2048;
const SHADOW_RES_WGSL = () => `${SHADOW_RES}`;
const SHADOW_HALF = 9000;
const SHADOW_DEPTH = 24000;
/**
 * The map is drawn again once the sun has turned this far (rad) since the one sampled — a shadow
 * cast from 2.5 km up then moves 2 m, a quarter of a texel (a day of 12 real minutes turns the sun
 * 0.05° in 0.1 s) — or once the camera is this far (m) from its middle.
 */
const SHADOW_TURN = 0.05 * (Math.PI / 180);
const SHADOW_MOVE = 500;
/**
 * The shadow map's caster mask: MASK_CELLS² cells of the map, a bit each for whether any caster lies
 * in it or within the lookups' reach of it (skyShadowAt then skips its taps where none does).
 */
const MASK_CELLS = 64;
const MASK_WGSL = /* wgsl */ `
@group(0) @binding(0) var shadowMap: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> mask: array<atomic<u32>, ${(MASK_CELLS * MASK_CELLS) / 32}>;
var<workgroup> found: atomic<u32>;
// One workgroup a cell: its ${SHADOW_RES_WGSL()} / ${MASK_CELLS} texels a side and two beyond each edge (the taps' reach).
@compute @workgroup_size(8, 8)
fn shadowMask(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&found, 0u); }
  workgroupBarrier();
  let side = ${SHADOW_RES_WGSL()} / ${MASK_CELLS};
  let base = vec2i(wg.xy) * side - 2;
  var any = false;
  for (var y = i32(lid.y); y < side + 4; y += 8) {
    for (var x = i32(lid.x); x < side + 4; x += 8) {
      let t = clamp(base + vec2i(x, y), vec2i(0), vec2i(${SHADOW_RES_WGSL()} - 1));
      if (textureLoad(shadowMap, t, 0) < 1.0) { any = true; }
    }
  }
  if (any) { atomicOr(&found, 1u); }
  workgroupBarrier();
  if (li == 0u && atomicLoad(&found) != 0u) {
    let bit = wg.y * ${MASK_CELLS}u + wg.x;
    atomicOr(&mask[bit >> 5u], 1u << (bit & 31u));
  }
}
`;

/** Lakes and streams are drawn within this distance (m). */
const WATER_RANGE = 6000;
/** An island smaller than this on screen (output pixels) takes the coarsest polar mesh (POLAR_FAR). */
const FAR_PX = 24;
/** The instance lists: polar levels 0–2, lakes, streams, shadow casters, polar POLAR_FAR. */
const LISTS = 7, FAR_LIST = 6;
/** realmU: where the ring mask starts (u32 index), and the instance list after it. */
const MASK = 4 * (10 + 2 * SKY_RINGS.rings + 1);
const INST = MASK + 4;

type V3 = [number, number, number];

/** Light axes for a light toward `to`: forward away from it (f), right (r) and up (u) across it. */
function lightAxes(to: ArrayLike<number>, f: V3, r: V3, u: V3): void {
  const l = Math.hypot(to[0], to[1], to[2]);
  f[0] = -to[0] / l; f[1] = -to[1] / l; f[2] = -to[2] / l;
  const rl = Math.hypot(f[2], f[0]) || 1;
  r[0] = -f[2] / rl; r[1] = 0; r[2] = f[0] / rl;
  u[0] = r[1] * f[2] - r[2] * f[1]; u[1] = r[2] * f[0] - r[0] * f[2]; u[2] = r[0] * f[1] - r[1] * f[0];
}

/** What the CPU keeps of each uploaded island for culling (and whether its stream made it into the buffer). */
type Uploaded = { isl: SkyIsland; x: number; z: number; reach: number; top: number; tip: number; stream: boolean };

export class SkyRealm {
  private device: GPUDevice;
  private readonly spec: RealmSpec;
  private buf: GPUBuffer;
  private data = new ArrayBuffer(REALM_LAYOUT.bytes);
  private ubuf: GPUBuffer;
  private u = new ArrayBuffer(REALM_FRAME_BYTES);
  private uf = new Float32Array(this.u);
  private ui = new Int32Array(this.u);
  private uu = new Uint32Array(this.u);
  private polarIdx: GPUBuffer[] = [];
  private polarCount: number[] = [];
  private farIdx: GPUBuffer;
  private farCount: number;
  private ringFull!: GPUBuffer;
  private ringHoles: GPUBuffer[] = [];
  private ringCount = [0, 0];
  // the rings' vertex cache, and per ring its job (where it lies) and what it was last made for
  private ringCache: GPUBuffer;
  private ringJobs: GPUBuffer[] = [];
  private ringGenBinds: GPUBindGroup[] = [];
  private ringDrawBind!: GPUBindGroup;
  private ringGenPipe!: GPUComputePipeline;
  private ringDirty: boolean[] = [];
  private version = 0;
  private archIdx: GPUBuffer;
  private archCount: number;
  private lakeIdx: GPUBuffer;
  private lakeCount: number;
  private shadows: GPUTexture[];
  private shadowViews: GPUTextureView[];
  private shadowSampler: GPUSampler;
  // per map, its caster mask (written by shadowMask after the map is drawn, read as a uniform)
  private masks: GPUBuffer[];
  private maskPipe!: GPUComputePipeline;
  private maskBinds: GPUBindGroup[] = [];
  /** The map drawn this frame (−1: none), and the one sampled. */
  private drawIdx = -1;
  private sampleIdx = 1;
  private pipes!: {
    polar: GPURenderPipeline[]; polarFar: GPURenderPipeline; ring: GPURenderPipeline; arch: GPURenderPipeline; lake: GPURenderPipeline; stream: GPURenderPipeline;
    fall: GPURenderPipeline; mist: GPURenderPipeline; shadow: GPURenderPipeline; archShadow: GPURenderPipeline;
  };
  private shadowLayout!: GPUBindGroupLayout;
  private shadowBind: GPUBindGroup | null = null;
  // the uploaded realm, and the far cells still to build (nearest first)
  private pending: PendingCell[] = [];
  private islands: Uploaded[] = [];
  private arches: UploadedArch[] = [];
  private falls: UploadedFall[] = [];
  private at = new Float64Array(5).fill(NaN);
  // this frame's draws: instance lists (LISTS) and their ranges in realmU.inst
  private lists = Array.from({ length: LISTS }, () => new Uint32Array(REALM_MAX.instances));
  private listN = new Int32Array(LISTS);
  private ranges = new Int32Array(2 * LISTS);
  private ringsOn = 0;
  // per frame, preallocated: the view, the rings (first vertex x, z, step, hole x, z per ring), their square, ring jobs
  private view = new Float64Array(10);
  private rl = new Float64Array(6 * SKY_RINGS.rings);
  private sq = new Float64Array(4);
  private ringed = new Uint8Array(REALM_MAX.islands);
  private ringKey = new Float64Array(8 * SKY_RINGS.rings).fill(NaN);
  private job = new Float32Array(4);
  private fallCount = 0;
  private archDraw = 0;
  // light axes and middle of the map last drawn, and of the one sampled; the light now; what the
  // last map was drawn for (its casters: the realm's version, the arches, the islands near enough)
  private lf: V3 = [0, -1, 0]; private lr: V3 = [1, 0, 0]; private lu: V3 = [0, 0, 1];
  private sf: V3 = [0, -1, 0]; private sr: V3 = [1, 0, 0]; private su: V3 = [0, 0, 1];
  private tf: V3 = [0, -1, 0]; private tr: V3 = [1, 0, 0]; private tu: V3 = [0, 0, 1];
  private drawnCentre = new Float64Array(3);
  private lastCentre = new Float64Array(3);
  private hasLast = false;
  private hasDrawn = false;
  private drawnKey = 0;

  constructor(device: GPUDevice, spec: RealmSpec = {}) {
    this.device = device;
    this.spec = spec;
    const d = device;
    this.buf = d.createBuffer({ size: REALM_LAYOUT.bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.ubuf = d.createBuffer({ size: REALM_FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const upload = (data: Uint32Array<ArrayBuffer>) => {
      const b = d.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    for (const [n, m] of POLAR_LODS) {
      const idx = polarIndices(n, m);
      this.polarIdx.push(upload(idx));
      this.polarCount.push(idx.length);
    }
    const far = polarIndices(POLAR_FAR[0], POLAR_FAR[1]);
    this.farIdx = upload(far);
    this.farCount = far.length;
    const k = SKY_RINGS.k;
    const full = ringIndices(k, null);
    this.ringFull = upload(full);
    for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) this.ringHoles.push(upload(ringIndices(k, [k / 2 + dx, k / 2 + dz])));
    this.ringCount = [full.length, 3 * k * k * 6];
    const verts = (2 * k + 1) ** 2;
    this.ringCache = d.createBuffer({ size: SKY_RINGS.rings * verts * 16, usage: GPUBufferUsage.STORAGE });
    for (let r = 0; r < SKY_RINGS.rings; r++) {
      this.ringJobs.push(d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
      this.ringDirty.push(false);
    }
    const arch = archIndices();
    this.archIdx = upload(arch);
    this.archCount = arch.length;
    const lake = polarIndices(LAKE_RINGS, LAKE_SEGS);
    this.lakeIdx = upload(lake);
    this.lakeCount = lake.length;
    this.shadows = [0, 1].map(() => d.createTexture({ size: [SHADOW_RES, SHADOW_RES], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
    this.shadowViews = this.shadows.map((t) => t.createView());
    const enc = d.createCommandEncoder();
    for (const view of this.shadowViews) {
      enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } }).end();
    }
    d.queue.submit([enc.finish()]);
    this.shadowSampler = d.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.masks = [0, 1].map(() => d.createBuffer({ size: (MASK_CELLS * MASK_CELLS) / 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }

  /** Entries the realm adds to group 0: its buffer, its frame uniform, the shadow map and its comparison sampler. */
  static layoutEntries(): GPUBindGroupLayoutEntry[] {
    const VFC = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    return [
      { binding: 10, visibility: VFC, buffer: { type: 'read-only-storage' } },
      { binding: 11, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: 26, visibility: VFC, texture: { sampleType: 'depth' } },
      { binding: 27, visibility: VFC, sampler: { type: 'comparison' } },
      { binding: 29, visibility: VFC, buffer: { type: 'uniform' } },
    ];
  }

  /** Group-0 entries for frames that sample shadow map `which` (see sampled()). */
  bindEntries(which: number): GPUBindGroupEntry[] {
    return [
      { binding: 10, resource: { buffer: this.buf } },
      { binding: 11, resource: { buffer: this.ubuf } },
      { binding: 26, resource: this.shadowViews[which] },
      { binding: 27, resource: this.shadowSampler },
      { binding: 29, resource: { buffer: this.masks[which] } },
    ];
  }

  /** The shadow map this frame samples (pick the group-0 bind group made with it). */
  sampled(): number {
    return this.sampleIdx;
  }

  async init(
    module: GPUShaderModule, computeLayout: GPUBindGroupLayout, layout: GPUBindGroupLayout,
    target: GPUColorTargetState[], depth: GPUDepthStencilState, multisample: GPUMultisampleState,
  ): Promise<void> {
    const d = this.device;
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    // The sky rings' vertex cache: written by realmRingGen (group 1: its job, the cache), read by vsRealmRing.
    const genLayout = d.createBindGroupLayout({ entries: [
      { binding: 30, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 31, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });
    const drawLayout = d.createBindGroupLayout({ entries: [{ binding: 32, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }] });
    this.ringGenBinds = this.ringJobs.map((job) => d.createBindGroup({
      layout: genLayout, entries: [{ binding: 30, resource: { buffer: job } }, { binding: 31, resource: { buffer: this.ringCache } }],
    }));
    this.ringDrawBind = d.createBindGroup({ layout: drawLayout, entries: [{ binding: 32, resource: { buffer: this.ringCache } }] });
    const stub: Record<string, number> = this.spec.stubGround ? { STUB_RLGROUND: 1 } : {};
    const ringDesc: GPURenderPipelineDescriptor = {
      layout: d.createPipelineLayout({ bindGroupLayouts: [layout, drawLayout] }),
      vertex: { module, entryPoint: 'vsRealmRing' }, fragment: { module, entryPoint: 'fsRealmRing', targets: target, constants: stub },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
    };
    const genP = d.createComputePipelineAsync({ layout: d.createPipelineLayout({ bindGroupLayouts: [computeLayout, genLayout] }), compute: { module, entryPoint: 'realmRingGen' } });
    const blended: GPUColorTargetState = {
      format: target[0].format,
      blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } },
    };
    const noWrite: GPUDepthStencilState = { ...depth, depthWriteEnabled: false };
    const opaque = (vs: string, fs: string): GPURenderPipelineDescriptor => ({
      layout: pl, vertex: { module, entryPoint: vs }, fragment: { module, entryPoint: fs, targets: target, constants: stub },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
    });
    const blend = (vs: string, fs: string, topology: GPUPrimitiveTopology, write: boolean): GPURenderPipelineDescriptor => ({
      layout: pl, vertex: { module, entryPoint: vs }, fragment: { module, entryPoint: fs, targets: [blended] },
      primitive: { topology, cullMode: 'none' }, depthStencil: write ? depth : noWrite, multisample,
    });
    // The shadow pass sees group 0's frame and realm only (its own map is not bound while drawn).
    this.shadowLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 10, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 11, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });
    const shadowDesc = (vs: string): GPURenderPipelineDescriptor => ({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.shadowLayout] }),
      vertex: { module, entryPoint: vs },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBiasSlopeScale: 1.5 },
    });
    const [p0, p1, p2, pFar, ring, arch, lake, stream, fall, mist, shadow, archShadow] = await Promise.all([
      d.createRenderPipelineAsync(opaque('vsRealmPolar0', 'fsRealm')),
      d.createRenderPipelineAsync(opaque('vsRealmPolar1', 'fsRealm')),
      d.createRenderPipelineAsync(opaque('vsRealmPolar2', 'fsRealm')),
      d.createRenderPipelineAsync(opaque('vsRealmPolarFar', 'fsRealm')),
      d.createRenderPipelineAsync(ringDesc),
      d.createRenderPipelineAsync(opaque('vsArch', 'fsArch')),
      // Lakes write depth (ice is walked on, water hides what lies under it from the falls' mist).
      d.createRenderPipelineAsync(blend('vsLake', 'fsLake', 'triangle-list', true)),
      d.createRenderPipelineAsync(blend('vsStream', 'fsStream', 'triangle-strip', false)),
      d.createRenderPipelineAsync(blend('vsFall', 'fsFall', 'triangle-strip', false)),
      d.createRenderPipelineAsync(blend('vsMist', 'fsMist', 'triangle-strip', false)),
      d.createRenderPipelineAsync(shadowDesc('vsRealmShadow')),
      d.createRenderPipelineAsync(shadowDesc('vsArchShadow')),
    ]);
    this.pipes = { polar: [p0, p1, p2], polarFar: pFar, ring, arch, lake, stream, fall, mist, shadow, archShadow };
    this.ringGenPipe = await genP;
    this.maskPipe = await d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code: MASK_WGSL }), entryPoint: 'shadowMask' } });
    this.maskBinds = [0, 1].map((k) => d.createBindGroup({ layout: this.maskPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.shadowViews[k] }, { binding: 1, resource: { buffer: this.masks[k] } },
    ] }));
  }

  /** The frame uniform buffer (binding 0) the shadow pass needs. */
  setFrameBuffer(frame: GPUBuffer): void {
    this.shadowBind = this.device.createBindGroup({
      layout: this.shadowLayout,
      entries: [{ binding: 0, resource: { buffer: frame } }, { binding: 10, resource: { buffer: this.buf } }, { binding: 11, resource: { buffer: this.ubuf } }],
    });
  }

  /** Re-pick and pack the realm round (x, z) into the storage buffer, chunk-local to (ox, oz). */
  private upload(x: number, z: number, seed: number, ox: number, oz: number): void {
    this.pending.length = 0;
    const packed = packRealm(x, z, seed, ox, oz, this.data, this.pending, this.spec.far !== false);
    this.islands = packed.islands;
    this.arches = packed.arches;
    this.falls = packed.falls;
    this.device.queue.writeBuffer(this.buf, 0, this.data);
    this.version++; // (every ring's cache was made for the old realm)
  }

  /**
   * Re-pick the realm when the camera has moved far enough (or the seed or chunk changed), then lay
   * out this frame's draws. Allocates nothing (everything per frame lives in preallocated arrays).
   */
  setFrame(fp: FrameParams, ox: number, oz: number, toLight: ArrayLike<number> = LIGHT.sunDir): void {
    const at = this.at;
    const mx = fp.x - at[0], mz = fp.z - at[1];
    if (!(Math.sqrt(mx * mx + mz * mz) < REPICK && at[2] === fp.seed && at[3] === ox && at[4] === oz)) {
      at[0] = fp.x; at[1] = fp.z; at[2] = fp.seed; at[3] = ox; at[4] = oz;
      this.upload(fp.x, fp.z, fp.seed, ox, oz);
    } else if (this.pending.length) {
      // Far cells that came into range: built a few a frame, nearest first; packed once all are.
      const t0 = performance.now();
      let built = 0;
      while (built < this.pending.length && (built === 0 || performance.now() - t0 < BUILD_BUDGET_MS)) buildCell(this.pending[built++], fp.seed);
      this.pending.splice(0, built);
      if (!this.pending.length) this.upload(at[0], at[1], fp.seed, ox, oz);
    }
    const F = this.uf, I = this.ui, U = this.uu;
    // ---- the view (as the camera sees it), for culling ----
    const cy = Math.cos(fp.yaw), sy = Math.sin(fp.yaw), cp = Math.cos(fp.pitch), sp = Math.sin(fp.pitch);
    const v = this.view;
    v[0] = sy * cp; v[1] = sp; v[2] = cy * cp;      // forward
    v[3] = cy; v[4] = -sy;                          // right (x, z)
    v[5] = -sp * sy; v[6] = cp; v[7] = -sp * cy;    // up
    v[8] = fp.tanHalf * (fp.width / Math.max(fp.height, 1)); v[9] = fp.tanHalf;
    const pxPerRad = fp.height / (2 * fp.tanHalf);

    // ---- sky rings: where each lies (the land's ring layout), and the square they cover ----
    const s0 = SKY_RINGS.spacing0, k = SKY_RINGS.k, nr = SKY_RINGS.rings;
    const R = this.rl;
    for (let r = 0; r < nr; r++) {
      const step = 2 ** r, S = s0 * step;
      const cxr = fp.x / S, czr = fp.z / S;
      const ax = 2 * Math.floor(cxr / 2), az = 2 * Math.floor(czr / 2);
      R[6 * r] = (ax - k) * step; R[6 * r + 1] = (az - k) * step;                    // first vertex (ring-0 cells)
      R[6 * r + 2] = step;
      R[6 * r + 3] = Math.floor(cxr) - ax; R[6 * r + 4] = Math.floor(czr) - az;        // where the finer ring's square starts
    }
    const lr = 6 * (nr - 1);
    const sq = this.sq;
    sq[0] = R[lr] * s0 - fp.x; sq[1] = R[lr + 1] * s0 - fp.z;
    sq[2] = (R[lr] + 2 * k * R[lr + 2]) * s0 - fp.x; sq[3] = (R[lr + 1] + 2 * k * R[lr + 2]) * s0 - fp.z;
    // The islands they draw: those the camera is up with, reaching into their square.
    U.fill(0, MASK, MASK + 4);
    const ringed = this.ringed;
    ringed.fill(0);
    let anyRinged = false;
    const isl = this.islands;
    for (let idx = 0; idx < isl.length && idx < 128; idx++) {
      const e = isl[idx];
      if (e.isl.kind === 'debris' || e.isl.R < 60 || fp.y < e.isl.y - 60) continue;
      const rx = e.x - fp.x, rz = e.z - fp.z;
      if (rx + e.reach < sq[0] || rx - e.reach > sq[2] || rz + e.reach < sq[1] || rz - e.reach > sq[3]) continue;
      U[MASK + (idx >> 5)] |= 1 << (idx & 31);
      ringed[idx] = 1;
      anyRinged = true;
    }
    this.ringsOn = anyRinged && this.spec.rings !== false ? nr : 0;
    F[32] = 0; F[33] = 0; F[34] = s0; F[35] = this.ringsOn;
    F[36] = sq[0]; F[37] = sq[1]; F[38] = sq[2]; F[39] = sq[3];
    for (let r = 0; r < nr; r++) {
      const step = R[6 * r + 2];
      // Chunk-local first vertex (whole metres: exact in f32) and the centre in the ring's own cells (k, k).
      const cx = R[6 * r] * s0 - ox, cz = R[6 * r + 1] * s0 - oz;
      I[40 + 4 * r] = 0; I[41 + 4 * r] = 0; I[42 + 4 * r] = step; I[43 + 4 * r] = r;
      const c = 40 + 4 * nr + 4 * r;
      F[c] = k; F[c + 1] = k; F[c + 2] = cx; F[c + 3] = cz;
      // Its cache is remade when it moves, or the realm or the islands the rings draw change.
      const key = this.ringKey, o = 8 * r;
      if (this.ringsOn && (key[o] !== cx || key[o + 1] !== cz || key[o + 2] !== step || key[o + 3] !== this.version
        || key[o + 4] !== U[MASK] || key[o + 5] !== U[MASK + 1] || key[o + 6] !== U[MASK + 2] || key[o + 7] !== U[MASK + 3])) {
        key[o] = cx; key[o + 1] = cz; key[o + 2] = step; key[o + 3] = this.version;
        key[o + 4] = U[MASK]; key[o + 5] = U[MASK + 1]; key[o + 6] = U[MASK + 2]; key[o + 7] = U[MASK + 3];
        this.ringDirty[r] = true;
        const job = this.job;
        job[0] = cx; job[1] = cz; job[2] = step * s0; job[3] = r;
        this.device.queue.writeBuffer(this.ringJobs[r], 0, job);
      }
    }
    const mo = 40 + 8 * nr;
    const start = 0.75 * k;
    // Up among the islands: a top or an arch's deck within reach of the grass round the camera.
    const reachH = 600, reachV = 450;
    let aloft = false;
    for (let idx = 0; idx < isl.length && !aloft; idx++) {
      const e = isl[idx];
      const dx = e.x - fp.x, dz = e.z - fp.z;
      aloft = Math.abs(fp.y - e.isl.y) < reachV && Math.sqrt(dx * dx + dz * dz) < e.reach + reachH;
    }
    for (let a = 0; a < this.arches.length && !aloft; a++) {
      const A = this.arches[a];
      const d0 = Math.sqrt((A.x0 - fp.x) ** 2 + (A.z0 - fp.z) ** 2), d1 = Math.sqrt((A.x1 - fp.x) ** 2 + (A.z1 - fp.z) ** 2);
      aloft = Math.abs(fp.y - A.y) < reachV && Math.min(d0, d1) < Math.sqrt((A.x1 - A.x0) ** 2 + (A.z1 - A.z0) ** 2) + reachH;
    }
    F[mo] = start; F[mo + 1] = 1 / (k - 1 - start); F[mo + 2] = nr - 1; F[mo + 3] = aloft ? 1 : 0;

    // ---- instance lists: polar surfaces by level of detail, lakes, streams, shadow casters ----
    const L = this.lists, n = this.listN;
    n.fill(0); // (filled by put)
    for (let idx = 0; idx < isl.length; idx++) {
      const e = isl[idx];
      const mid = 0.5 * (e.top + e.tip), half = 0.5 * (e.top - e.tip);
      const r = Math.sqrt(e.reach * e.reach + half * half);
      const dx = e.x - fp.x, dy = mid - fp.y, dz = e.z - fp.z;
      const dxz = Math.sqrt(dx * dx + dz * dz), d = Math.sqrt(dxz * dxz + dy * dy);
      if (dxz < SHADOW_HALF + r && this.spec.shadows !== false) this.put(5, idx);
      // In view? (a sphere against the frustum)
      const vz = dx * v[0] + dy * v[1] + dz * v[2];
      if (vz < -r) continue;
      const vx = dx * v[3] + dz * v[4], vy = dx * v[5] + dy * v[6] + dz * v[7];
      if (Math.abs(vx) - r * Math.sqrt(1 + v[8] * v[8]) > vz * v[8] || Math.abs(vy) - r * Math.sqrt(1 + v[9] * v[9]) > vz * v[9]) continue;
      const size = (e.reach / Math.max(d - r, 1)) * pxPerRad;
      const lod = size < FAR_PX ? FAR_LIST : size < 60 ? 0 : size < 320 ? 1 : 2;
      this.put(lod, idx | (1 << 16));
      // The top: by the rings when they hold all of it, else here (clipped out of their square).
      const inside = dx - e.reach > sq[0] && dx + e.reach < sq[2] && dz - e.reach > sq[1] && dz + e.reach < sq[3];
      if (!(ringed[idx] && inside)) this.put(lod, idx | (ringed[idx] ? 2 << 16 : 0));
      if (d < WATER_RANGE + e.reach && this.spec.water !== false) {
        if (e.isl.lake) this.put(3, idx);
        if (e.stream) this.put(4, idx);
      }
    }
    // Into realmU.inst, list after list; ranges[2l], ranges[2l + 1]: first instance and count of list l.
    let m = 0;
    const ranges = this.ranges;
    for (let l = 0; l < LISTS; l++) {
      ranges[2 * l] = m;
      const c = Math.min(n[l], REALM_MAX.instances - m);
      for (let q = 0; q < c; q++) U[INST + m + q] = L[l][q];
      m += c;
      ranges[2 * l + 1] = c;
    }
    this.archDraw = this.spec.arches === false ? 0 : this.arches.length;
    this.fallCount = this.spec.falls === false ? 0 : this.falls.length;

    // ---- the shadow map: the one drawn last frame is sampled from now on; draw another only if it would differ ----
    const Lc = this.lastCentre, D = this.drawnCentre;
    if (this.drawIdx >= 0) {
      this.sampleIdx = this.drawIdx;
      for (let c = 0; c < 3; c++) { this.sf[c] = this.lf[c]; this.sr[c] = this.lr[c]; this.su[c] = this.lu[c]; Lc[c] = D[c]; }
      this.hasLast = true;
    }
    this.drawIdx = -1;
    lightAxes(toLight, this.tf, this.tr, this.tu);
    let key = (this.version * 0x9e3779b1 + this.archDraw) >>> 0;
    for (let q = 0; q < n[5]; q++) key = Math.imul(key ^ L[5][q], 0x85ebca6b) >>> 0;
    key = (key ^ n[5]) >>> 0;
    const cyy = Math.min(fp.y, 1500);
    const lf = this.lf, tf = this.tf;
    const turned = lf[0] * tf[0] + lf[1] * tf[1] + lf[2] * tf[2] < Math.cos(SHADOW_TURN);
    const moved = Math.hypot(fp.x - D[0], cyy - D[1], fp.z - D[2]) > SHADOW_MOVE;
    if (this.spec.shadows !== false && (!this.hasDrawn || turned || moved || key !== this.drawnKey)) {
      // Round the camera, snapped to whole texels in light space.
      this.drawIdx = 1 - this.sampleIdx;
      for (let c = 0; c < 3; c++) { this.lf[c] = tf[c]; this.lr[c] = this.tr[c]; this.lu[c] = this.tu[c]; }
      const LR = this.lr, LU = this.lu;
      const texel = (2 * SHADOW_HALF) / SHADOW_RES;
      const lxp = fp.x * LR[0] + cyy * LR[1] + fp.z * LR[2], lyp = fp.x * LU[0] + cyy * LU[1] + fp.z * LU[2];
      const sx = Math.round(lxp / texel) * texel - lxp, syy = Math.round(lyp / texel) * texel - lyp;
      D[0] = fp.x + LR[0] * sx + LU[0] * syy; D[1] = cyy + LR[1] * sx + LU[1] * syy; D[2] = fp.z + LR[2] * sx + LU[2] * syy;
      this.drawnKey = key;
      this.hasDrawn = true;
    }
    // Sampling: the map sampled, round its middle (none before the first is drawn).
    F[0] = Lc[0] - fp.x; F[1] = Lc[1] - fp.y; F[2] = Lc[2] - fp.z; F[3] = this.hasLast && this.spec.shadows !== false ? SHADOW_HALF : 0;
    F.set(this.sr, 4); F.set(this.su, 8); F.set(this.sf, 12); F[15] = SHADOW_DEPTH;
    // Drawing (when a map is drawn this frame): its middle and axes.
    F[16] = D[0] - fp.x; F[17] = D[1] - fp.y; F[18] = D[2] - fp.z; F[19] = (2 * SHADOW_HALF) / SHADOW_RES;
    F.set(this.lr, 20); F.set(this.lu, 24); F.set(this.lf, 28); F[31] = SHADOW_DEPTH;
    this.device.queue.writeBuffer(this.ubuf, 0, this.u);
  }

  /** Append w to instance list `list` (while it has room). */
  private put(list: number, w: number): void {
    const n = this.listN, L = this.lists[list];
    if (n[list] < L.length) L[n[list]++] = w;
  }

  /** Remake the caches of the sky rings that moved (before the main pass; `group0` the compute bind group). */
  encodeRings(enc: GPUCommandEncoder, group0: GPUBindGroup): void {
    if (!this.ringsOn || !this.ringDirty.some(Boolean)) return;
    const cp = enc.beginComputePass(stamped('realm.rings'));
    cp.setPipeline(this.ringGenPipe);
    cp.setBindGroup(0, group0);
    const verts = (2 * SKY_RINGS.k + 1) ** 2;
    for (let r = 0; r < SKY_RINGS.rings; r++) {
      if (!this.ringDirty[r]) continue;
      cp.setBindGroup(1, this.ringGenBinds[r]);
      cp.dispatchWorkgroups(Math.ceil(verts / 64));
      this.ringDirty[r] = false;
    }
    cp.end();
  }

  /** The undersides and arches into a new shadow map, when this frame draws one (before the main pass). */
  encodeShadow(enc: GPUCommandEncoder): void {
    if (!this.shadowBind || this.drawIdx < 0) return;
    const sp = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowViews[this.drawIdx], depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      ...stamped('realm.shadow'),
    });
    sp.setBindGroup(0, this.shadowBind);
    const first = this.ranges[10], count = this.ranges[11];
    if (count) {
      sp.setPipeline(this.pipes.shadow);
      sp.setIndexBuffer(this.polarIdx[1], 'uint32');
      sp.drawIndexed(this.polarCount[1], count, 0, 0, first);
    }
    if (this.archDraw) {
      sp.setPipeline(this.pipes.archShadow);
      sp.setIndexBuffer(this.archIdx, 'uint32');
      sp.drawIndexed(this.archCount, this.archDraw);
    }
    sp.end();
    // Its caster mask (read from the frame it is first sampled).
    enc.clearBuffer(this.masks[this.drawIdx]);
    const cp = enc.beginComputePass(stamped('realm.mask'));
    cp.setPipeline(this.maskPipe);
    cp.setBindGroup(0, this.maskBinds[this.drawIdx]);
    cp.dispatchWorkgroups(MASK_CELLS, MASK_CELLS);
    cp.end();
  }

  /** The islands, the sky rings and the arches (opaque; group 0 already set). */
  draw(pass: GPURenderPassEncoder): void {
    for (let l = 0; l < POLAR_LODS.length; l++) {
      const first = this.ranges[2 * l], count = this.ranges[2 * l + 1];
      if (!count) continue;
      pass.setPipeline(this.pipes.polar[l]);
      pass.setIndexBuffer(this.polarIdx[l], 'uint32');
      pass.drawIndexed(this.polarCount[l], count, 0, 0, first);
    }
    const farFirst = this.ranges[2 * FAR_LIST], farN = this.ranges[2 * FAR_LIST + 1];
    if (farN) {
      pass.setPipeline(this.pipes.polarFar);
      pass.setIndexBuffer(this.farIdx, 'uint32');
      pass.drawIndexed(this.farCount, farN, 0, 0, farFirst);
    }
    if (this.ringsOn) {
      pass.setPipeline(this.pipes.ring);
      pass.setBindGroup(1, this.ringDrawBind);
      pass.setIndexBuffer(this.ringFull, 'uint32');
      pass.drawIndexed(this.ringCount[0], 1, 0, 0, 0);
      for (let r = 1; r < this.ringsOn; r++) {
        const dx = this.rl[6 * r + 3], dz = this.rl[6 * r + 4];
        pass.setIndexBuffer(this.ringHoles[dx + 2 * dz], 'uint32');
        pass.drawIndexed(this.ringCount[1], 1, 0, 0, r);
      }
    }
    if (this.archDraw) {
      pass.setPipeline(this.pipes.arch);
      pass.setIndexBuffer(this.archIdx, 'uint32');
      pass.drawIndexed(this.archCount, this.archDraw);
    }
  }

  /** Lakes and streams (blended, after the land's water). */
  drawWater(pass: GPURenderPassEncoder): void {
    const lf = this.ranges[6], ln = this.ranges[7];
    if (ln) {
      pass.setPipeline(this.pipes.lake);
      pass.setIndexBuffer(this.lakeIdx, 'uint32');
      pass.drawIndexed(this.lakeCount, ln, 0, 0, lf);
    }
    const sf = this.ranges[8], sn = this.ranges[9];
    if (sn) {
      pass.setPipeline(this.pipes.stream);
      pass.draw(2 * (15 * STREAM_SUB + 1), sn, 0, sf);
    }
  }

  /** The falls and their mist (blended, last). */
  drawFalls(pass: GPURenderPassEncoder): void {
    if (!this.fallCount) return;
    pass.setPipeline(this.pipes.fall);
    pass.draw(2 * (FALL_SEGMENTS + 1), this.fallCount);
    pass.setPipeline(this.pipes.mist);
    pass.draw(4, this.fallCount * MIST_PER_FALL);
  }

  /** What was uploaded and drawn last frame (debug, benches). */
  stats(): { islands: number; arches: number; falls: number; polar: number[]; rings: number; lakes: number; streams: number; casters: number } {
    return {
      islands: this.islands.length, arches: this.arches.length, falls: this.falls.length,
      polar: [this.ranges[2 * FAR_LIST + 1], this.ranges[1], this.ranges[3], this.ranges[5]], rings: this.ringsOn, lakes: this.ranges[7], streams: this.ranges[9], casters: this.ranges[11],
    };
  }

  destroy(): void {
    for (const b of [this.buf, this.ubuf, ...this.polarIdx, this.farIdx, this.ringFull, ...this.ringHoles, this.archIdx, this.lakeIdx, this.ringCache, ...this.ringJobs]) b.destroy();
    for (const t of this.shadows) t.destroy();
    for (const b of this.masks) b.destroy();
  }
}

/** An uploaded arch, for culling. */
type UploadedArch = { x0: number; z0: number; x1: number; z1: number; y: number };
/** An uploaded fall, for culling. */
type UploadedFall = { x: number; y: number; z: number; reach: number; w: number };

/**
 * The realm round (x, z) packed, chunk-local to (ox, oz), into `data`
 * (REALM_LAYOUT; the realm buffer skyRealm.wgsl.ts reads). Within REALM_NEAR
 * the nearest archipelagos and isles first while they fit, each a cluster of
 * its islands and arches; floating rocks only within DEBRIS_RANGE; streams
 * while their points fit (an island whose stream does not is drawn dry).
 * Beyond it, out to REALM_FAR, the islands alone (no floating rocks, arches or
 * streams; their falls while they fit), in no cluster: nothing near the camera
 * looks them up. With `pending`, far cells not built yet are left out and
 * listed there (SkyRealm builds them a few a frame); without, all are built.
 * Returns what the CPU keeps of it for culling.
 */
export function packRealm(
  x: number, z: number, seed: number, ox: number, oz: number, data = new ArrayBuffer(REALM_LAYOUT.bytes), pending?: PendingCell[], withFar = true,
): { data: ArrayBuffer; islands: Uploaded[]; arches: UploadedArch[]; falls: UploadedFall[] } {
  const { archipelagos, isles } = realmNear(x, z, REALM_NEAR, seed);
  type Cluster = { islands: SkyIsland[]; arch: Archipelago | null; cx: number; cz: number; reach: number; d: number };
  const cluster = (a: Archipelago): Cluster => ({ islands: a.islands, arch: a, cx: a.cx, cz: a.cz, reach: a.reach, d: Math.hypot(a.cx - x, a.cz - z) - a.reach });
  const lone = (i: SkyIsland): Cluster => ({ islands: [i], arch: null, cx: i.cx, cz: i.cz, reach: i.R * outlineMax(i), d: Math.hypot(i.cx - x, i.cz - z) - i.R });
  const clusters: Cluster[] = [...archipelagos.map(cluster), ...isles.map(lone)].sort((a, b) => a.d - b.d);
  const near = new Set<Archipelago | SkyIsland>([...archipelagos, ...isles]);
  const far = withFar ? realmNear(x, z, REALM_FAR, seed, pending) : { archipelagos: [], isles: [] };
  const farClusters: Cluster[] = [
    ...far.archipelagos.filter((a) => !near.has(a)).map(cluster),
    ...far.isles.filter((i) => !near.has(i)).map(lone),
  ].sort((a, b) => a.d - b.d);
  const L = REALM_LAYOUT;
  const f = new Float32Array(data), u = new Uint32Array(data), i32 = new Int32Array(data);
  u.fill(0);
  let nc = 0, ni = 0, np = 0, na = 0, nf = 0;
  const islands: Uploaded[] = [], arches: UploadedArch[] = [], falls: UploadedFall[] = [];
  const put = (isl: SkyIsland, streams: boolean) => {
    const k = ni++;
    const o = (L.isl + k * L.islStride) / 4;
    f[o] = isl.cx - ox; f[o + 1] = isl.cz - oz; f[o + 2] = isl.R; f[o + 3] = isl.y;
    f.set(isl.a, o + 4);
    f[o + 8] = isl.amp; f[o + 9] = isl.hill; f[o + 10] = isl.wl; f[o + 11] = isl.crag;
    f[o + 12] = isl.peak; f[o + 13] = isl.cliff; f[o + 14] = isl.depth; f[o + 15] = 0;
    i32[o + 16] = isl.seed;
    u[o + 17] = SKY_BIOMES.indexOf(isl.biome);
    let flags = isl.kind === 'debris' ? RL_FLAGS.debris : 0;
    const Lk = isl.lake;
    if (Lk) {
      flags |= RL_FLAGS.lake | (Lk.frozen ? RL_FLAGS.frozen : 0);
      f[o + 20] = Lk.x; f[o + 21] = Lk.z; f[o + 22] = Lk.r; f[o + 23] = Lk.level;
      f[o + 24] = Lk.aspect; f[o + 25] = Lk.cos; f[o + 26] = Lk.sin; f[o + 27] = Lk.depth;
      f[o + 28] = Lk.b0; f[o + 29] = Lk.b1;
    }
    f[o + 30] = -1;
    const withStream = streams && !!isl.stream && np + isl.stream.pts.length <= REALM_MAX.points;
    if (withStream) {
      flags |= RL_FLAGS.stream;
      const P = isl.stream!.pts;
      u[o + 19] = np | (P.length << 16);
      let mx = 0, mz = 0;
      for (const p of P) { mx += p[0] / P.length; mz += p[1] / P.length; }
      let rad = 0;
      for (const p of P) {
        f.set(p, (L.pts + np * 16) / 4);
        np++;
        rad = Math.max(rad, Math.hypot(p[0] - mx, p[1] - mz) + CHANNEL_REACH * p[3] + 1);
      }
      f[o + 32] = mx; f[o + 33] = mz; f[o + 34] = rad;
    }
    if (isl.fall && nf < REALM_MAX.falls) {
      flags |= RL_FLAGS.fall;
      const F = isl.fall;
      const fo = (L.fall + nf * L.fallStride) / 4;
      f[fo] = F.x - ox; f[fo + 1] = F.y; f[fo + 2] = F.z - oz; f[fo + 3] = F.width;
      f[fo + 4] = F.dirX; f[fo + 5] = F.dirZ; f[fo + 6] = F.speed; f[fo + 7] = F.reach;
      f[fo + 8] = F.seed & 0xffff; f[fo + 9] = k; f[fo + 10] = u[o + 17];
      f[o + 30] = nf;
      falls.push({ x: F.x, y: F.y, z: F.z, reach: F.reach, w: F.width });
      nf++;
    }
    u[o + 18] = flags;
    islands.push({ isl, x: isl.cx, z: isl.cz, reach: isl.R * outlineMax(isl), top: isl.y + 3 * isl.hill + 5, tip: isl.y - maxDrop(isl), stream: withStream });
  };
  for (const c of clusters) {
    if (nc >= REALM_MAX.clusters) break;
    const list = c.islands.filter((isl) => isl.kind !== 'debris' || c.d < DEBRIS_RANGE);
    const archs = c.arch ? c.arch.arches.length : 0;
    if (ni + list.length > REALM_MAX.islands || na + archs > REALM_MAX.arches) continue;
    const firstIsl = ni, firstArch = na;
    for (const isl of list) put(isl, true);
    for (const A of c.arch ? c.arch.arches : []) {
      const k = na++;
      const o = (L.arch + k * L.archStride) / 4;
      f[o] = A.ax - ox; f[o + 1] = A.az - oz; f[o + 2] = A.ay; f[o + 3] = A.len;
      f[o + 4] = A.bx - ox; f[o + 5] = A.bz - oz; f[o + 6] = A.by; f[o + 7] = A.bow;
      f[o + 8] = A.dirX; f[o + 9] = A.dirZ; f[o + 10] = A.rise; f[o + 11] = A.wMid;
      f[o + 12] = A.wEnd; f[o + 13] = A.thMid; f[o + 14] = A.thEnd;
      const from = c.arch!.islands[A.from], to = c.arch!.islands[A.to];
      f[o + 16] = A.seed & 0xffff; f[o + 17] = SKY_BIOMES.indexOf(from.biome); f[o + 18] = SKY_BIOMES.indexOf(to.biome);
      arches.push({ x0: A.ax, z0: A.az, x1: A.bx, z1: A.bz, y: Math.max(A.ay, A.by) });
    }
    const co = (L.cl + nc * L.clStride) / 4;
    f[co] = c.cx - ox; f[co + 1] = c.cz - oz; f[co + 2] = c.reach + 10;
    u[co + 4] = firstIsl; u[co + 5] = ni - firstIsl; u[co + 6] = firstArch; u[co + 7] = na - firstArch;
    nc++;
  }
  for (const c of farClusters) {
    const list = c.islands.filter((isl) => isl.kind !== 'debris');
    if (ni + list.length > REALM_MAX.islands) break;
    for (const isl of list) put(isl, false);
  }
  u[0] = nc; u[1] = ni; u[2] = na; u[3] = nf;
  return { data, islands, arches, falls };
}
