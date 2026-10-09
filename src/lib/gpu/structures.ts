/**
 * The Eyries' structures on the GPU (host side of structures.wgsl.ts).
 * ---------------------------------------------------------------------------
 * The kit (skyworld/kit/) in one vertex and one index buffer; every piece
 * instance of the Eyries round the camera in one storage buffer, re-packed
 * (chunk-local) when the camera has moved REPICK metres or Eyries have been
 * built; Eyries not built yet are built a few milliseconds a frame, nearest
 * first, as the realm's far cells are.
 *
 * Per frame, on the CPU (a few thousand spheres): each structure is culled
 * against the view and given one level of detail by how large it shows, so
 * all its pieces agree; a piece is drawn at its structure's level when that
 * is at most its `detail` (props near only). A counting sort lays the visible
 * instances out by (piece, level) — the kit's meshes in that order — and each
 * bucket in use is one instanced indexed draw. Structures near the trees'
 * shadow box are listed again for the trees' shadow map (drawn into it by
 * Forest.encode), so buildings shade the ground, the trees and each other.
 * Allocates nothing per frame.
 */

import { buildKit, KIT_VERTEX_BYTES, type Kit } from '../skyworld/kit/kit';
import { pieceId, PIECES } from '../skyworld/kit/pieces';
import { routesNear, shipAt, type Route } from '../skyworld/ships';
import { eyrieBuilt, eyrieCellsNear, eyrieInCell, eyrieStep } from '../skyworld/eyrie/eyrie';
import { INSTANCE_BYTES, packInstance, PFLAG } from '../skyworld/structures';
import type { FrameParams } from './frame';
import { REALM_NEAR } from './skyRealm.wgsl';
import { FX_BEAM_SEG, FX_BYTES, FX_EYEWALL_SEG, FX_SMOKE_PUFFS, FX_VORTEX_LAYERS } from './effects.wgsl';

/** Piece instances and structures the buffers hold. */
const MAX_INSTANCES = 1 << 17;
const MAX_STRUCTURES = 16384;
/** Structures are drawn within this distance (m); re-picked after the camera moves REPICK (m). */
const RANGE = REALM_NEAR;
const REPICK = 250;
/** Eyries not yet built are built for at most this long a frame (ms; at least one). */
const BUILD_BUDGET_MS = 3;
/** Levels of detail by a structure's on-screen radius (output pixels): level 0 above the first, 1 above the second, 2 above the third, else culled. */
const LOD_PX = [90, 18, 1.2] as const;
/** Structures whose spheres come this close (m) to the camera cast into the trees' shadow map. */
const SHADOW_RANGE = 170;
/** Skyships under way at most (each a structure of one piece, moved every frame), and the radius round one (m). */
const MAX_SHIPS = 64;
const SHIP_R = 27;
/** Effects the list holds; smoke is drawn within SMOKE_RANGE (m), the Thread and the Eye as far as the structures. */
const MAX_FX = 512;
const SMOKE_RANGE = 3500;
/** Effect kinds in the order they are listed and drawn (effects.wgsl.ts). */
const FX_KINDS = ['vortex', 'smoke', 'beam'] as const;
/** A beam's height (m; effects.wgsl.ts FX_BEAM_H) and a smoke column's reach, for culling. */
const BEAM_H = 3200, SMOKE_REACH = 60;

export class Structures {
  readonly kit: Kit;
  private device: GPUDevice;
  private vbuf: GPUBuffer;
  private ibuf: GPUBuffer;
  private instBuf: GPUBuffer;
  private drawBuf: GPUBuffer;
  private instData = new ArrayBuffer(MAX_INSTANCES * INSTANCE_BYTES);
  private instF = new Float32Array(this.instData);
  private instU = new Uint32Array(this.instData);
  private drawData = new Uint32Array(2 * MAX_INSTANCES);
  // the structures uploaded, and their pieces' kinds
  private sX = new Float64Array(MAX_STRUCTURES);
  private sY = new Float64Array(MAX_STRUCTURES);
  private sZ = new Float64Array(MAX_STRUCTURES);
  private sR = new Float32Array(MAX_STRUCTURES);
  private sFirst = new Uint32Array(MAX_STRUCTURES);
  private sCount = new Uint32Array(MAX_STRUCTURES);
  private sLod = new Int8Array(MAX_STRUCTURES);
  private sShadow = new Int8Array(MAX_STRUCTURES);
  private nS = 0;
  private nI = 0;
  // skyships under way: their routes (re-listed with the structures), placed after the static ones each frame
  private routes: Route[] = [];
  private nShips = 0;
  private shipPos = new Float64Array(4);
  private readonly shipPiece = pieceId('skyship');
  // the lanterns of the Eyries round here (lanterns.ts makes their light): each light's x, y, z, strength
  private lights = new Float64Array(4 * 4096);
  private nLights = 0;
  private lightVersion = 0;
  private lightOut = { pts: this.lights, n: 0, version: 0 };
  private readonly lightOf: Map<number, readonly [number, number]> = new Map([
    // piece → (the light's height over the piece's place × its scale, its strength)
    [pieceId('lantern.post'), [3.0, 1.0]], [pieceId('lantern.hang'), [-1.25, 0.8]], [pieceId('lantern.paper'), [-0.8, 0.6]], [pieceId('brazier'), [1.9, 1.3]],
  ]);
  private pPiece = new Uint16Array(MAX_INSTANCES);
  private pDetail = new Uint8Array(MAX_INSTANCES);
  // per (piece, level): main pass counts and first instances, then the shadow pass's
  private readonly B = PIECES.length * 3;
  private counts = new Int32Array(2 * PIECES.length * 3);
  private starts = new Int32Array(2 * PIECES.length * 3);
  private fill = new Int32Array(2 * PIECES.length * 3);
  private mainTotal = 0;
  private shadowTotal = 0;
  private at = new Float64Array(5).fill(NaN);
  private pending: Array<[number, number]> = [];
  private sinceRepack = 0;
  private bind!: GPUBindGroup;
  private layout1: GPUBindGroupLayout;
  private pipe!: GPURenderPipeline;
  private shadowPipe: GPURenderPipeline | null = null;
  // the effects round the camera (as uploaded: chunk-local; and in world m, for culling), then those in view by kind
  private fxBuf: GPUBuffer;
  private fxAll = new Float32Array((MAX_FX * FX_BYTES) / 4);
  private fxWorld = new Float64Array(3 * MAX_FX);
  private fxKind = new Uint8Array(MAX_FX);
  private nFx = 0;
  private fxData = new Float32Array((MAX_FX * FX_BYTES) / 4);
  private fxCount = new Int32Array(FX_KINDS.length);
  private fxPipes: GPURenderPipeline[] | null = null;

  /** `sync` (benches): build every Eyrie in range when re-picking, not a slice a frame. */
  constructor(device: GPUDevice, private readonly sync = false) {
    this.device = device;
    this.kit = buildKit();
    const d = device;
    this.vbuf = d.createBuffer({ size: this.kit.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.vbuf, 0, this.kit.vertices);
    this.ibuf = d.createBuffer({ size: this.kit.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.ibuf, 0, this.kit.indices);
    this.instBuf = d.createBuffer({ size: MAX_INSTANCES * INSTANCE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.drawBuf = d.createBuffer({ size: 2 * MAX_INSTANCES * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fxBuf = d.createBuffer({ size: MAX_FX * FX_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.layout1 = d.createBindGroupLayout({ entries: [
      { binding: 40, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 41, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 42, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ] });
    this.bind = d.createBindGroup({ layout: this.layout1, entries: [
      { binding: 40, resource: { buffer: this.instBuf } }, { binding: 41, resource: { buffer: this.drawBuf } },
      { binding: 42, resource: { buffer: this.fxBuf } },
    ] });
  }

  /**
   * Pipelines: the main pass (group 0 the renderer's `layout`) and, given the trees' shadow layout,
   * the shadow pass into their map.
   */
  async init(module: GPUShaderModule, layout: GPUBindGroupLayout, target: GPUColorTargetState[], depth: GPUDepthStencilState,
    multisample: GPUMultisampleState, shadowLayout?: GPUBindGroupLayout | null): Promise<void> {
    const d = this.device;
    const vertex: GPUVertexBufferLayout = {
      arrayStride: KIT_VERTEX_BYTES,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'snorm8x4' },
        { shaderLocation: 2, offset: 20, format: 'uint8x4' },
        { shaderLocation: 3, offset: 24, format: 'float32x2' },
      ],
    };
    // the effects: blended premultiplied (the beam adds itself, its alpha 0), tested against depth, not writing it
    const blended: GPUColorTargetState = {
      format: target[0].format,
      blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } },
    };
    const fxPipe = (vs: string, fs: string) => d.createRenderPipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [layout, this.layout1] }),
      vertex: { module, entryPoint: vs }, fragment: { module, entryPoint: fs, targets: [blended] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: { ...depth, depthWriteEnabled: false }, multisample,
    });
    const fxPipes = Promise.all([fxPipe('vsVortex', 'fsVortex'), fxPipe('vsSmoke', 'fsSmoke'), fxPipe('vsBeam', 'fsBeam'), fxPipe('vsEyewall', 'fsEyewall')]);
    const [pipe, shadow] = await Promise.all([
      d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout, this.layout1] }),
        vertex: { module, entryPoint: 'vsStructure', buffers: [vertex] },
        fragment: { module, entryPoint: 'fsStructure', targets: target },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth, multisample,
      }),
      shadowLayout ? d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [shadowLayout, this.layout1] }),
        vertex: { module, entryPoint: 'vsStructureShadow', buffers: [vertex] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBiasSlopeScale: 2 },
      }) : Promise.resolve(null),
    ]);
    this.pipe = pipe;
    this.shadowPipe = shadow;
    this.fxPipes = await fxPipes;
  }

  /** Re-pack the structures of the Eyries built round (x, z), chunk-local to (ox, oz); list the cells still to build. */
  private repack(x: number, z: number, seed: number, ox: number, oz: number): void {
    this.pending.length = 0;
    let nS = 0, nI = 0, nFx = 0;
    const f = this.instF, u = this.instU, A = this.fxAll, W = this.fxWorld;
    for (const [ix, iz] of eyrieCellsNear(x, z, RANGE)) {
      if (!eyrieBuilt(ix, iz, seed) && !this.sync) { this.pending.push([ix, iz]); continue; }
      const e = eyrieInCell(ix, iz, seed);
      if (!e) continue;
      for (const st of e.structures) {
        if (nS >= MAX_STRUCTURES || nI + st.pieces.length > MAX_INSTANCES) break;
        this.sX[nS] = st.x; this.sY[nS] = st.y; this.sZ[nS] = st.z; this.sR[nS] = st.r;
        this.sFirst[nS] = nI; this.sCount[nS] = st.pieces.length;
        for (const p of st.pieces) {
          packInstance(p, f, u, nI * 8, ox, oz);
          this.pPiece[nI] = p.piece;
          this.pDetail[nI] = p.detail ?? PIECES[p.piece].detail;
          nI++;
        }
        nS++;
        if (st.fx) {
          for (const e of st.fx) {
            if (nFx >= MAX_FX) break;
            const o = nFx * (FX_BYTES / 4);
            A[o] = e.x - ox; A[o + 1] = e.y; A[o + 2] = e.z - oz; A[o + 3] = e.r;
            // (a seed from where it stands: each column and Eye out of step with the others)
            A[o + 4] = (((e.x * 0.0137 + e.z * 0.0291) % 1) + 1) % 1; A[o + 5] = e.k === 'smoke' ? 3.2 : 0; A[o + 6] = 1; A[o + 7] = 0;
            W[3 * nFx] = e.x; W[3 * nFx + 1] = e.y; W[3 * nFx + 2] = e.z;
            this.fxKind[nFx] = FX_KINDS.indexOf(e.k);
            nFx++;
          }
        }
      }
    }
    this.nS = nS;
    this.nI = nI;
    this.nFx = nFx;
    // the lanterns lit outdoors (lanterns.ts): their light's place and strength
    let nl = 0;
    const Lt = this.lights;
    for (const [ix, iz] of eyrieCellsNear(x, z, RANGE)) {
      const e = eyrieBuilt(ix, iz, seed) || this.sync ? eyrieInCell(ix, iz, seed) : null;
      if (!e) continue;
      for (const st of e.structures) {
        if (Math.hypot(st.x - x, st.z - z) - st.r > 600) continue;
        for (const p of st.pieces) {
          const l = this.lightOf.get(p.piece);
          if (!l || p.flags & PFLAG.indoor || nl >= Lt.length / 4) continue;
          Lt[4 * nl] = p.x; Lt[4 * nl + 1] = p.y + l[0] * p.sy; Lt[4 * nl + 2] = p.z; Lt[4 * nl + 3] = l[1];
          nl++;
        }
      }
    }
    this.nLights = nl;
    this.lightVersion++;
    // the ships' routes round here (they are placed every frame: moveShips)
    routesNear(x, z, RANGE, seed, this.routes);
    let ships = 0;
    for (const rt of this.routes) ships += rt.ships;
    this.nShips = Math.min(ships, MAX_SHIPS, MAX_STRUCTURES - nS, MAX_INSTANCES - nI);
    if (nI) this.device.queue.writeBuffer(this.instBuf, 0, this.instData, 0, nI * INSTANCE_BYTES);
  }

  /** Build Eyries that came into range (a few a frame), re-pack when needed, then cull and lay out this frame's draws. */
  setFrame(fp: FrameParams, ox: number, oz: number, shadowCentre?: ArrayLike<number> | null): void {
    const at = this.at;
    if (!(Math.hypot(fp.x - at[0], fp.z - at[1]) < REPICK && at[2] === fp.seed && at[3] === ox && at[4] === oz)) {
      at[0] = fp.x; at[1] = fp.z; at[2] = fp.seed; at[3] = ox; at[4] = oz;
      this.repack(fp.x, fp.z, fp.seed, ox, oz);
    } else if (this.pending.length) {
      // The nearest first, a slice at a time, until this frame's budget is spent.
      const t0 = performance.now();
      let built = 0;
      while (built < this.pending.length) {
        const left = BUILD_BUDGET_MS - (performance.now() - t0);
        if (left <= 0 && built > 0) break;
        const [ix, iz] = this.pending[built];
        if (!eyrieStep(ix, iz, fp.seed, Math.max(left, 0.5))) break;
        built++;
      }
      this.pending.splice(0, built);
      // (packing every instance costs a few ms: done when all are built, and now and then meanwhile)
      if ((!this.pending.length && built) || (built && ++this.sinceRepack >= 12)) {
        this.sinceRepack = 0;
        const rest = this.pending.slice();
        this.repack(at[0], at[1], fp.seed, ox, oz);
        this.pending = rest;
      }
    }
    this.moveShips(fp.time, ox, oz);
    this.cull(fp, shadowCentre);
  }

  /** The lanterns lit outdoors round the last re-pick (world x, y, z, strength each; `version` changes with them). */
  lanternPoints(): { pts: Float64Array; n: number; version: number } {
    const o = this.lightOut;
    o.n = this.nLights; o.version = this.lightVersion;
    return o;
  }

  /** The skyships where they are now (t: the clock, s), as structures after the static ones; their instances uploaded. */
  private moveShips(t: number, ox: number, oz: number): void {
    if (!this.nShips) return;
    const o = this.shipPos, f = this.instF, u = this.instU;
    let m = 0;
    for (const rt of this.routes) {
      for (let k = 0; k < rt.ships && m < this.nShips; k++, m++) {
        shipAt(rt, k, t, o);
        const s = this.nS + m, i = this.nI + m;
        this.sX[s] = o[0]; this.sY[s] = o[1] + 5; this.sZ[s] = o[2]; this.sR[s] = SHIP_R;
        this.sFirst[s] = i; this.sCount[s] = 1;
        this.pPiece[i] = this.shipPiece; this.pDetail[i] = 2;
        packInstance({ piece: this.shipPiece, x: o[0], y: o[1], z: o[2], yaw: o[3], sx: 1, sy: 1, sz: 1, palette: rt.palette, seed: (m * 7919) & 0xffff, flags: 0 }, f, u, i * 8, ox, oz);
      }
    }
    this.device.queue.writeBuffer(this.instBuf, this.nI * INSTANCE_BYTES, this.instData, this.nI * INSTANCE_BYTES, this.nShips * INSTANCE_BYTES);
  }

  private cull(fp: FrameParams, shadowCentre?: ArrayLike<number> | null): void {
    const cy = Math.cos(fp.yaw), sy = Math.sin(fp.yaw), cp = Math.cos(fp.pitch), sp = Math.sin(fp.pitch);
    const fx = sy * cp, fyv = sp, fz = cy * cp;      // forward
    const rx = cy, rz = -sy;                         // right
    const ux = -sp * sy, uy = cp, uz = -sp * cy;     // up
    const ty = fp.tanHalf * 1.02, tx = fp.tanHalf * (fp.width / Math.max(fp.height, 1)) * 1.02;
    const kx = Math.sqrt(1 + tx * tx), ky = Math.sqrt(1 + ty * ty);
    const pxPerRad = fp.height / (2 * fp.tanHalf);
    const scx = shadowCentre ? shadowCentre[0] : fp.x, scy = shadowCentre ? shadowCentre[1] : fp.y, scz = shadowCentre ? shadowCentre[2] : fp.z;
    const B = this.B, C = this.counts;
    C.fill(0);
    const casts = !!this.shadowPipe;
    const nAll = this.nS + this.nShips;
    for (let s = 0; s < nAll; s++) {
      const dx = this.sX[s] - fp.x, dy = this.sY[s] - fp.y, dz = this.sZ[s] - fp.z, r = this.sR[s];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      let lod = -1;
      const vz = dx * fx + dy * fyv + dz * fz;
      if (vz > -r) {
        const vx = dx * rx + dz * rz, vy = dx * ux + dy * uy + dz * uz;
        if (!(Math.abs(vx) - r * kx > vz * tx || Math.abs(vy) - r * ky > vz * ty)) {
          const px = d <= r ? Infinity : (r / d) * pxPerRad;
          lod = px > LOD_PX[0] ? 0 : px > LOD_PX[1] ? 1 : px > LOD_PX[2] ? 2 : -1;
        }
      }
      this.sLod[s] = lod;
      let sh = -1;
      if (casts) {
        const ds = Math.hypot(this.sX[s] - scx, this.sY[s] - scy, this.sZ[s] - scz) - r;
        if (ds < SHADOW_RANGE) sh = lod >= 0 ? Math.min(lod, 1) : 1;
      }
      this.sShadow[s] = sh;
      if (lod < 0 && sh < 0) continue;
      const first = this.sFirst[s], end = first + this.sCount[s];
      for (let k = first; k < end; k++) {
        const det = this.pDetail[k], b = this.pPiece[k] * 3;
        if (lod >= 0 && det >= lod) C[b + lod]++;
        if (sh >= 0 && det >= sh) C[B + b + sh]++;
      }
    }
    // First instances: the main pass's buckets, then the shadow pass's, one list.
    let m = 0;
    for (let b = 0; b < 2 * B; b++) {
      this.starts[b] = m;
      this.fill[b] = m;
      m += C[b];
      if (b === B - 1) this.mainTotal = m;
    }
    this.shadowTotal = m - this.mainTotal;
    const L = this.drawData, F = this.fill;
    for (let s = 0; s < nAll; s++) {
      const lod = this.sLod[s], sh = this.sShadow[s];
      if (lod < 0 && sh < 0) continue;
      const first = this.sFirst[s], end = first + this.sCount[s];
      for (let k = first; k < end; k++) {
        const det = this.pDetail[k], b = this.pPiece[k] * 3;
        if (lod >= 0 && det >= lod) L[F[b + lod]++] = k;
        if (sh >= 0 && det >= sh) L[F[B + b + sh]++] = k;
      }
    }
    if (m) this.device.queue.writeBuffer(this.drawBuf, 0, L, 0, m);
    // The effects in view, listed by kind (the Eye, smoke, beams): each as a sphere round what it fills.
    const FW = this.fxWorld, FA = this.fxAll, FD = this.fxData, FC = this.fxCount, words = FX_BYTES / 4;
    FC.fill(0);
    let w = 0;
    for (let kind = 0; kind < FX_KINDS.length; kind++) {
      for (let e = 0; e < this.nFx; e++) {
        if (this.fxKind[e] !== kind) continue;
        const r0 = FA[e * words + 3];
        const lift = kind === 2 ? 0.5 * BEAM_H : kind === 1 ? 0.5 * SMOKE_REACH : 0;
        const r = kind === 2 ? 0.5 * BEAM_H + r0 : kind === 1 ? SMOKE_REACH : r0 * 1.05;
        const dx = FW[3 * e] - fp.x, dy = FW[3 * e + 1] + lift - fp.y, dz = FW[3 * e + 2] - fp.z;
        if (kind === 1 && dx * dx + dy * dy + dz * dz > SMOKE_RANGE * SMOKE_RANGE) continue;
        const vz = dx * fx + dy * fyv + dz * fz;
        if (vz <= -r) continue;
        const vx = dx * rx + dz * rz, vy = dx * ux + dy * uy + dz * uz;
        if (Math.abs(vx) - r * kx > vz * tx || Math.abs(vy) - r * ky > vz * ty) continue;
        FD.set(FA.subarray(e * words, (e + 1) * words), w * words);
        w++;
        FC[kind]++;
      }
    }
    if (w) this.device.queue.writeBuffer(this.fxBuf, 0, FD, 0, w * words);
  }

  /** The structures in view (group 0 already set): one instanced draw per (piece, level) in use. */
  draw(pass: GPURenderPassEncoder): void {
    if (!this.mainTotal) return;
    this.drawBuckets(pass, this.pipe, 0);
  }

  /** The effects in view (blended, after the falls; group 0 already set): the Gyre's Eye, smoke, then the Thread. */
  drawEffects(pass: GPURenderPassEncoder): void {
    const p = this.fxPipes, c = this.fxCount;
    if (!p || !(c[0] + c[1] + c[2])) return;
    pass.setBindGroup(1, this.bind);
    // (the first instance carries where each kind's effects start in the list)
    if (c[0]) {
      pass.setPipeline(p[3]);
      pass.draw(2 * (FX_EYEWALL_SEG + 1), c[0], 0, 0);
      pass.setPipeline(p[0]);
      pass.draw(4, c[0] * FX_VORTEX_LAYERS, 0, 0);
    }
    if (c[1]) { pass.setPipeline(p[1]); pass.draw(4, c[1] * FX_SMOKE_PUFFS, 0, c[0] * FX_SMOKE_PUFFS); }
    if (c[2]) { pass.setPipeline(p[2]); pass.draw(2 * (FX_BEAM_SEG + 1), c[2], 0, c[0] + c[1]); }
  }

  /** The casters near the camera into the trees' shadow map (Forest.encode's pass, its group 0 set). */
  drawShadow(pass: GPURenderPassEncoder): void {
    if (!this.shadowTotal || !this.shadowPipe) return;
    this.drawBuckets(pass, this.shadowPipe, this.B);
  }

  private drawBuckets(pass: GPURenderPassEncoder, pipe: GPURenderPipeline, off: number): void {
    pass.setPipeline(pipe);
    pass.setBindGroup(1, this.bind);
    pass.setVertexBuffer(0, this.vbuf);
    pass.setIndexBuffer(this.ibuf, 'uint32');
    const meshes = this.kit.meshes;
    for (let b = 0; b < this.B; b++) {
      const c = this.counts[off + b];
      if (!c) continue;
      const mesh = meshes[b];
      pass.drawIndexed(mesh.indexCount, c, mesh.firstIndex, 0, this.starts[off + b]);
    }
  }

  /** What was laid out last frame (debug, benches). */
  stats(): { structures: number; instances: number; drawn: number; shadow: number; draws: number } {
    let draws = 0;
    for (let b = 0; b < this.B; b++) if (this.counts[b]) draws++;
    return { structures: this.nS, instances: this.nI, drawn: this.mainTotal, shadow: this.shadowTotal, draws };
  }

  destroy(): void {
    for (const b of [this.vbuf, this.ibuf, this.instBuf, this.drawBuf, this.fxBuf]) b.destroy();
  }
}
