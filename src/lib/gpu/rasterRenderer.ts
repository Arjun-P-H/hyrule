/**
 * Raster terrain: the clipmap drawn as a geometry clipmap.
 * ---------------------------------------------------------------------------
 * Ring r (one per clipmap level) is a (2k)² grid of cells, spacing
 * spacing0 · 2^r · stride, centred on the camera snapped to twice its cell
 * size; rings r ≥ 1 leave out the square the finer ring covers (one of four
 * precomputed index buffers, by where that square falls). Vertices read the
 * cached heights at lattice points and morph onto the next ring near their
 * outer edge (terrain.wgsl.ts rasterShader). Shading is the ray-march
 * renderer's shadeSurface(); the hero is sphere-traced inside the scissor
 * rectangle of its bounding sphere and writes its own depth; sky and clouds
 * fill what is left on the far plane.
 */

import { CAMERA, RUINS, VALLEY } from '../constants';
import { HERO_UNIFORM } from '../hero/rig';
import { ClipmapCache, type PassTimestamps } from './clipmapCache';
import { FRAME_BYTES, meshBytes, rasterShader } from './terrain.wgsl';
import { HERO_BYTES, packHeroWgsl } from './hero.wgsl';
import { BLADE_BYTES, BLADE_VERTS, grassBytes } from './grass.wgsl';
import { ISLANDS_BYTES, ISLAND_RINGS, ISLAND_SEGMENTS, MAX_ISLANDS, polarIndices } from './skyIslands.wgsl';
import { islandWaterfall, islandsNear, type Island, type Waterfall } from '../skyIslands';
import { FALLS_BYTES, FALL_SEGMENTS, MAX_FALLS } from './waterfalls.wgsl';
import { MAX_RUIN_BOXES, MAX_RUIN_SITES, RUIN_BOX_BYTES, RUIN_SITES_BYTES } from './ruins.wgsl';
import { ruinSitesNear } from '../ruins';
import { elevation, hash01 } from '../terrainMath';
import { DEFAULT_DAYS, floatingOrigin, packFrame, type FrameParams, type TerrainRendererLike, type TerrainVariant } from './frame';
import { ringIndices, ringLayout } from './rings';
import { SCENE_FORMAT, TemporalResolve } from './taa';
import { Forest } from './forest';
import { SkyCycle, type SkyBindings } from './sky';

type RasterVariant = Extract<TerrainVariant, { height: 'raster' }>;

/** Where the day–night sky's resources sit in group 0 (sky.wgsl.ts cycleSkyWGSL). */
const SKY_BINDINGS: SkyBindings = { sky: 17, light: 18, view: 19 };

export class RasterTerrainRenderer implements TerrainRendererLike {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  readonly variant: RasterVariant;
  compileMs = 0;

  private cache: ClipmapCache;
  private frameBuf: GPUBuffer;
  private frameData = new ArrayBuffer(FRAME_BYTES);
  private frameF32 = new Float32Array(this.frameData);
  private frameI32 = new Int32Array(this.frameData);
  private meshBuf!: GPUBuffer;
  private meshData: ArrayBuffer;
  private meshF32: Float32Array<ArrayBuffer>;
  private meshI32: Int32Array<ArrayBuffer>;
  private heroBuf: GPUBuffer;
  private heroData = new Float32Array(HERO_BYTES / 4);
  private heroScissor: [number, number, number, number] | null = null;
  private bind!: GPUBindGroup;
  private computeBind!: GPUBindGroup;
  // With the forest or the day–night sky, group 0 comes in two versions, one per shadow map and sky set
  // sampled (Forest.sampled(), SkyCycle.sampled(): both flip every frame, in step).
  private binds: GPUBindGroup[] = [];
  private computeBinds: GPUBindGroup[] = [];
  private terrainPipe!: GPURenderPipeline;
  private waterPipe!: GPURenderPipeline;
  private heroPipe!: GPURenderPipeline;
  private skyPipe!: GPURenderPipeline;
  private fullIdx: GPUBuffer | null = null;
  private holeIdx: GPUBuffer[] = [];
  private fullCount = 0;
  private holeCount = 0;
  private holes: Array<[number, number]> = [];
  // variant.diag: one index buffer per ring, rebuilt by ringTris when the ring moves
  private diagPipe: GPUComputePipeline | null = null;
  private ringIdx: GPUBuffer[] = [];
  private ringJobBuf: GPUBuffer[] = [];
  private ringJobBind: GPUBindGroup[] = [];
  private ringKeys: number[][] = [];
  private ringDirty: boolean[] = [];
  private depth: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  // MSAA (variant.msaa = 4): multisampled colour resolved into the target
  private msaaColor: GPUTexture | null = null;
  private msaaView: GPUTextureView | null = null;
  // TAA (variant.taa): the scene renders into its targets at the internal size, then resolve + present
  private taa: TemporalResolve | null = null;
  private cam = { yaw: 0, pitch: 0, x: 0, y: 0, z: 0, tanHalf: 1 };
  private size: [number, number] = [0, 0];
  private rings: number;
  // grass (variant.grass): uniform, blade buffer, indirect draw arguments
  private grassBuf: GPUBuffer | null = null;
  private grassF32: Float32Array<ArrayBuffer> | null = null;
  private grassI32: Int32Array<ArrayBuffer> | null = null;
  private bladeBuf: GPUBuffer | null = null;
  private argsBuf: GPUBuffer | null = null;
  private scatterPipe: GPUComputePipeline | null = null;
  private grassPipe: GPURenderPipeline | null = null;
  private scatterBind: GPUBindGroup | null = null;
  private drawBind: GPUBindGroup | null = null;
  // sky islands (variant.islands): the nearest MAX_ISLANDS, chunk-local, and the shared polar mesh
  private islandBuf: GPUBuffer | null = null;
  private islandData = new ArrayBuffer(ISLANDS_BYTES);
  private islandIdxBuf: GPUBuffer | null = null;
  private islandIdxCount = 0;
  private islandPipe: GPURenderPipeline | null = null;
  private islands: Island[] = [];
  private islandsAt: [number, number, number] = [NaN, NaN, NaN];
  private islandOrigin: [number, number] = [NaN, NaN];
  // waterfalls off those islands
  private fallsBuf: GPUBuffer | null = null;
  private fallsData = new ArrayBuffer(FALLS_BYTES);
  private falls: Waterfall[] = [];
  private fallPipe: GPURenderPipeline | null = null;
  private mistPipe: GPURenderPipeline | null = null;
  // ruins (variant.ruins): boxes of the sites near the camera, and the site list for the scatters
  private ruinBoxBuf: GPUBuffer | null = null;
  private ruinSitesBuf: GPUBuffer | null = null;
  private ruinBoxData = new ArrayBuffer(MAX_RUIN_BOXES * RUIN_BOX_BYTES);
  private ruinSitesData = new ArrayBuffer(RUIN_SITES_BYTES);
  private ruinBoxCount = 0;
  private ruinPipe: GPURenderPipeline | null = null;
  private ruinsAt: [number, number, number, number, number] = [NaN, NaN, NaN, NaN, NaN];
  // trees (variant.forest): placement cache, culling, draws, shadow map
  private forest: Forest | null = null;
  // the day–night sky (variant.sky = 'cycle'): its LUTs, the frame's light rig and the probe of the sky
  private sky: SkyCycle | null = null;

  constructor(device: GPUDevice, format: GPUTextureFormat, variant: RasterVariant) {
    const { clip, k, stride } = variant;
    if (k % 2 !== 0) throw new Error(`raster: k must be even: ${k}`);
    if (k * stride > clip.n / 2 - 4) throw new Error(`raster: k·stride = ${k * stride} exceeds the clipmap window (n/2 − 4 = ${clip.n / 2 - 4})`);
    this.device = device;
    this.format = format;
    this.variant = variant;
    this.rings = clip.levels;
    this.cache = new ClipmapCache(device, clip, variant.genDpScale, { blend: variant.blend, mode: variant.sky === 'cycle' ? 'path' : 'classic' });
    this.frameBuf = device.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.heroBuf = device.createBuffer({ size: HERO_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.meshData = new ArrayBuffer(meshBytes(this.rings));
    this.meshF32 = new Float32Array(this.meshData);
    this.meshI32 = new Int32Array(this.meshData);
  }

  async init(): Promise<void> {
    const d = this.device;
    const v = this.variant;
    const grass = v.grass ?? null;
    if (v.forest) this.forest = new Forest(d, v.forest);
    const cycle = v.sky === 'cycle';
    if (cycle) this.sky = new SkyCycle(d);
    const skyReady = this.sky?.init();
    const islands = !!v.islands;
    const ruins = !!v.ruins;
    const t0 = performance.now();
    const VFC = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    const shared: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: 2, visibility: VFC, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: VFC, sampler: { type: 'filtering' } },
      { binding: 4, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: 5, visibility: VFC, buffer: { type: 'uniform' } },
      ...(grass ? [{ binding: 6, visibility: VFC, buffer: { type: 'uniform' as const } }] : []),
      { binding: 7, visibility: VFC, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      ...(this.forest ? Forest.layoutEntries() : []),
      ...(islands ? [{ binding: 10, visibility: VFC, buffer: { type: 'uniform' as const } }] : []),
      ...(islands ? [{ binding: 11, visibility: VFC, buffer: { type: 'uniform' as const } }] : []),
      ...(ruins ? [{ binding: 13, visibility: VFC, buffer: { type: 'uniform' as const } }] : []),
      ...(cycle ? SkyCycle.layoutEntries(SKY_BINDINGS) : []),
    ];
    // Group 0 for the compute passes, and for drawing: the latter adds the ruin boxes.
    const computeLayout = d.createBindGroupLayout({ entries: shared });
    const renderOnly: GPUBindGroupLayoutEntry[] = [
      ...(ruins ? [{ binding: 12, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' as const } }] : []),
    ];
    const layout = renderOnly.length ? d.createBindGroupLayout({ entries: [...shared, ...renderOnly] }) : computeLayout;
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const shaders = d.createShaderModule({
      code: rasterShader({ n: v.clip.n, levels: v.clip.levels, blend: v.blend, k: v.k, rings: this.rings, grass, islands, ruins, diag: !!v.diag,
        forest: this.forest ? { spec: this.forest.spec, bounds: this.forest.models.bounds } : null, sky: cycle ? 'cycle' : 'classic' }),
    });
    const target = [{ format: v.taa ? SCENE_FORMAT : this.format }];
    const depthState = (write: boolean, compare: GPUCompareFunction): GPUDepthStencilState => ({ format: 'depth32float', depthWriteEnabled: write, depthCompare: compare });
    const multisample: GPUMultisampleState = { count: v.taa ? 1 : v.msaa ?? 1 };
    if (v.taa) {
      this.taa = new TemporalResolve(d, this.format);
      // Sharpen what is upscaled; at 1:1 the resolved image is left soft (closest to supersampling).
      this.taa.sharpen = v.taaSharpen ?? ((v.renderScale ?? 1) < 0.99 ? 0.5 : 0.15);
      if (v.taaAlpha !== undefined) this.taa.alpha = v.taaAlpha;
      await this.taa.init();
    }
    const scatterLayout = grass ? d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    }) : null;
    const drawLayout = grass ? d.createBindGroupLayout({
      entries: [{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
    }) : null;
    const blended: GPUColorTargetState = { format: target[0].format, blend: {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    } };
    const fallPipesP = islands ? Promise.all([
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsFall' }, fragment: { module: shaders, entryPoint: 'fsFall', targets: [blended] },
        primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: depthState(false, 'greater'), multisample,
      }),
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsMist' }, fragment: { module: shaders, entryPoint: 'fsMist', targets: [blended] },
        primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: depthState(false, 'greater'), multisample,
      }),
    ]) : Promise.resolve([null, null] as const);
    const ruinPipeP = ruins ? d.createRenderPipelineAsync({
      layout: pl, vertex: { module: shaders, entryPoint: 'vsRuin' }, fragment: { module: shaders, entryPoint: 'fsRuin', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
    }) : Promise.resolve(null);
    const islandPipeP = islands ? d.createRenderPipelineAsync({
      layout: pl, vertex: { module: shaders, entryPoint: 'vsIsland' }, fragment: { module: shaders, entryPoint: 'fsIsland', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
    }) : Promise.resolve(null);
    const ringJobLayout = v.diag ? d.createBindGroupLayout({
      entries: [
        { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    }) : null;
    const diagPipeP = ringJobLayout ? d.createComputePipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [computeLayout, ringJobLayout] }),
      compute: { module: shaders, entryPoint: 'ringTris' },
    }) : Promise.resolve(null);
    const [terrain, water, heroP, sky, , scatterP, grassP] = await Promise.all([
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsTerrain' }, fragment: { module: shaders, entryPoint: 'fsTerrain', targets: target },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
      }),
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsWater' },
        // Premultiplied alpha over the lake bed already drawn.
        fragment: { module: shaders, entryPoint: 'fsWater', targets: [{ format: target[0].format, blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        } }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
      }),
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsSky' }, fragment: { module: shaders, entryPoint: 'fsHero', targets: target },
        primitive: { topology: 'triangle-list' }, depthStencil: depthState(true, 'greater'), multisample,
      }),
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsSky' }, fragment: { module: shaders, entryPoint: 'fsSky', targets: target },
        primitive: { topology: 'triangle-list' }, depthStencil: depthState(false, 'greater-equal'), multisample,
      }),
      this.cache.init(),
      scatterLayout ? d.createComputePipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [computeLayout, scatterLayout] }),
        compute: { module: shaders, entryPoint: 'grassScatter' },
      }) : Promise.resolve(null),
      drawLayout ? d.createRenderPipelineAsync({
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout, drawLayout] }),
        vertex: { module: shaders, entryPoint: 'vsGrass' }, fragment: { module: shaders, entryPoint: 'fsGrass', targets: target },
        primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
      }) : Promise.resolve(null),
    ]);
    this.terrainPipe = terrain;
    this.waterPipe = water;
    this.heroPipe = heroP;
    this.skyPipe = sky;
    this.scatterPipe = scatterP;
    this.grassPipe = grassP;
    this.islandPipe = await islandPipeP;
    [this.fallPipe, this.mistPipe] = await fallPipesP;
    this.ruinPipe = await ruinPipeP;
    this.diagPipe = await diagPipeP;
    await skyReady;
    if (this.forest) await this.forest.init(shaders, computeLayout, layout, this.frameBuf, target, depthState(true, 'greater'), multisample);
    if (ringJobLayout) {
      const k = v.k;
      for (let r = 0; r < this.rings; r++) {
        const cells = r === 0 ? 4 * k * k : 3 * k * k;
        const idx = d.createBuffer({ size: cells * 6 * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE });
        const job = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.ringIdx.push(idx);
        this.ringJobBuf.push(job);
        this.ringJobBind.push(d.createBindGroup({
          layout: ringJobLayout,
          entries: [{ binding: 14, resource: { buffer: job } }, { binding: 15, resource: { buffer: idx } }],
        }));
        this.ringKeys.push([NaN, NaN, NaN, NaN, NaN]);
        this.ringDirty.push(true);
      }
    }
    if (ruins) {
      this.ruinBoxBuf = d.createBuffer({ size: MAX_RUIN_BOXES * RUIN_BOX_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.ruinSitesBuf = d.createBuffer({ size: RUIN_SITES_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (islands) {
      this.fallsBuf = d.createBuffer({ size: FALLS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.islandBuf = d.createBuffer({ size: ISLANDS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const idx = polarIndices(ISLAND_RINGS, ISLAND_SEGMENTS);
      this.islandIdxBuf = d.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(this.islandIdxBuf, 0, idx);
      this.islandIdxCount = idx.length;
    }
    this.compileMs = performance.now() - t0;

    if (grass && scatterLayout && drawLayout) {
      const blades = grass.rings * grass.n * grass.n;
      this.grassBuf = d.createBuffer({ size: grassBytes(grass.rings), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const gd = new ArrayBuffer(grassBytes(grass.rings));
      this.grassF32 = new Float32Array(gd);
      this.grassI32 = new Int32Array(gd);
      this.bladeBuf = d.createBuffer({ size: blades * BLADE_BYTES, usage: GPUBufferUsage.STORAGE });
      this.argsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      d.queue.writeBuffer(this.argsBuf, 0, new Uint32Array([BLADE_VERTS, 0, 0, 0]));
      this.scatterBind = d.createBindGroup({
        layout: scatterLayout,
        entries: [
          { binding: 0, resource: { buffer: this.bladeBuf } },
          { binding: 1, resource: { buffer: this.argsBuf } },
        ],
      });
      this.drawBind = d.createBindGroup({ layout: drawLayout, entries: [{ binding: 2, resource: { buffer: this.bladeBuf } }] });
    }

    const k = v.k;
    const upload = (data: Uint32Array<ArrayBuffer>) => {
      const b = d.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.fullCount = 4 * k * k * 6;
    this.holeCount = 3 * k * k * 6;
    if (!v.diag) {
      this.fullIdx = upload(ringIndices(k, null));
      for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) this.holeIdx.push(upload(ringIndices(k, [k / 2 + dx, k / 2 + dz])));
    }
    this.meshBuf = d.createBuffer({ size: meshBytes(this.rings), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sharedEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.frameBuf } },
      { binding: 1, resource: { buffer: this.cache.clipBuf } },
      { binding: 2, resource: this.cache.view },
      { binding: 3, resource: this.cache.sampler },
      { binding: 4, resource: { buffer: this.meshBuf } },
      { binding: 5, resource: { buffer: this.heroBuf } },
      ...(this.grassBuf ? [{ binding: 6, resource: { buffer: this.grassBuf } }] : []),
      { binding: 7, resource: this.cache.sunView! },
      ...(this.islandBuf ? [{ binding: 10, resource: { buffer: this.islandBuf } }] : []),
      ...(this.fallsBuf ? [{ binding: 11, resource: { buffer: this.fallsBuf } }] : []),
      ...(this.ruinSitesBuf ? [{ binding: 13, resource: { buffer: this.ruinSitesBuf } }] : []),
    ];
    const renderEntries: GPUBindGroupEntry[] = [
      ...(this.ruinBoxBuf ? [{ binding: 12, resource: { buffer: this.ruinBoxBuf } }] : []),
    ];
    for (const which of this.forest || this.sky ? [0, 1] : [0]) {
      const shared = [
        ...sharedEntries,
        ...(this.forest ? this.forest.bindEntries(which) : []),
        ...(this.sky ? this.sky.bindEntries(SKY_BINDINGS, which) : []),
      ];
      const cb = d.createBindGroup({ layout: computeLayout, entries: shared });
      this.computeBinds.push(cb);
      this.binds.push(renderEntries.length ? d.createBindGroup({ layout, entries: [...shared, ...renderEntries] }) : cb);
    }
    [this.computeBind, this.bind] = [this.computeBinds[0], this.binds[0]];
  }

  pendingTexels(): number {
    return this.cache.pendingTexels();
  }

  invalidate(): void {
    this.cache.invalidate();
    this.forest?.invalidate();
  }

  setFrame(fp: FrameParams): void {
    const F = this.frameF32;
    // With TAA the scene renders at the internal size (renderScale) and is upscaled into the output.
    const scale = this.taa ? fp.renderScale ?? this.variant.renderScale ?? 1 : 1;
    const W = Math.max(2, Math.round(fp.width * scale)), Hh = Math.max(2, Math.round(fp.height * scale));
    let jx = 0, jy = 0;
    if (this.taa) {
      if (this.variant.taaSharpen === undefined) this.taa.sharpen = scale < 0.99 ? 0.5 : 0.15;
      this.taa.resize(W, Hh, fp.width, fp.height);
      [jx, jy] = this.taa.jitter();
      const c = this.cam;
      c.yaw = fp.yaw; c.pitch = fp.pitch; c.x = fp.x; c.y = fp.y; c.z = fp.z; c.tanHalf = fp.tanHalf;
    }
    packFrame(fp, F, this.frameI32, W, Hh, jx, jy);
    this.device.queue.writeBuffer(this.frameBuf, 0, this.frameData);
    const { ox, oz, lx, lz } = floatingOrigin(fp.x, fp.z);
    this.cache.setCamera(fp.x, fp.z, lx, lz, ox, oz, fp.seed);

    const v = this.variant;
    const s0 = v.clip.spacing0, n = v.clip.n, k = v.k;
    const refX = Math.floor(fp.x / s0), refZ = Math.floor(fp.z / s0);
    const f = this.meshF32;
    const i = this.meshI32;
    f[0] = fp.x - refX * s0; f[1] = fp.y; f[2] = fp.z - refZ * s0; f[3] = s0;
    const f0 = 1 / fp.tanHalf;
    const aspect = W / Hh;
    f[4] = f0 / aspect; f[5] = f0; f[6] = CAMERA.near; f[7] = 0;
    const start = v.morphStart * k;
    f[8] = start; f[9] = 1 / (k - 1 - start); f[10] = this.rings - 1; f[11] = 0;
    const ringsAt = 12, centerAt = ringsAt + 4 * this.rings, texAt = centerAt + 4 * this.rings;
    const layout = ringLayout(fp.x, fp.z, s0, k, v.stride, this.rings);
    this.holes.length = 0;
    for (let r = 0; r < layout.length; r++) {
      const ring = layout[r];
      i[ringsAt + 4 * r + 0] = ring.corner[0] - refX;
      i[ringsAt + 4 * r + 1] = ring.corner[1] - refZ;
      i[ringsAt + 4 * r + 2] = ring.step;
      i[ringsAt + 4 * r + 3] = r;
      f[centerAt + 4 * r + 0] = ring.center[0] - refX;
      f[centerAt + 4 * r + 1] = ring.center[1] - refZ;
      this.holes.push(ring.hole);
      if (this.diagPipe) this.markRing(r, ring.corner, ring.hole, fp.seed);
    }
    for (let l = 0; l <= this.rings; l++) {
      const sc = 2 ** -l;
      f[texAt + 4 * l + 0] = ((refX * sc) % n + n) % n;
      f[texAt + 4 * l + 1] = ((refZ * sc) % n + n) % n;
      f[texAt + 4 * l + 2] = sc;
      f[texAt + 4 * l + 3] = 0;
    }
    this.device.queue.writeBuffer(this.meshBuf, 0, this.meshData);
    if (this.taa) {
      this.depthView = this.taa.depthView;
      this.size = [W, Hh];
    } else if (this.size[0] !== W || this.size[1] !== Hh) {
      this.depth?.destroy();
      const sampleCount = v.msaa ?? 1;
      this.depth = this.device.createTexture({ size: [W, Hh], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount });
      this.depthView = this.depth.createView();
      this.msaaColor?.destroy();
      this.msaaColor = sampleCount > 1
        ? this.device.createTexture({ size: [W, Hh], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount })
        : null;
      this.msaaView = this.msaaColor ? this.msaaColor.createView() : null;
      this.size = [W, Hh];
    }

    // Hero: WGSL block, and the pixels its bounding sphere can cover.
    const H = this.heroData;
    this.heroScissor = null;
    if (fp.hero && fp.hero[HERO_UNIFORM.bound + 3] > 0) {
      packHeroWgsl(fp.hero, H);
      this.heroScissor = sphereScissor(F, H[96], H[97], H[98], H[99], f0 / aspect, f0, W, Hh);
    } else {
      H[99] = 0;
    }
    this.device.queue.writeBuffer(this.heroBuf, 0, H);

    const grass = v.grass;
    if (grass && this.grassBuf && this.grassF32 && this.grassI32) {
      const gf = this.grassF32, gi = this.grassI32;
      const R = grass.rings;
      for (let r = 0; r < R; r++) {
        const g = grass.spacing0 * 2 ** r;
        const cx = fp.x / g, cz = fp.z / g;
        const fx = Math.floor(cx), fz = Math.floor(cz);
        gf[4 * r + 0] = cx - fx;
        gf[4 * r + 1] = cz - fz;
        gf[4 * r + 2] = g;
        gf[4 * r + 3] = (grass.n / 2 - 2) * g;
        gi[4 * (R + r) + 0] = fx - grass.n / 2;
        gi[4 * (R + r) + 1] = fz - grass.n / 2;
      }
      const o = 8 * R;
      gf[o] = grass.blend; gf[o + 1] = grass.density; gf[o + 2] = fp.time; gf[o + 3] = 0;
      gf[o + 4] = 0.87; gf[o + 5] = 0.49; gf[o + 6] = 0.32; gf[o + 7] = 0.055; // wind: direction, strength, gust scale (1/m)
      this.device.queue.writeBuffer(this.grassBuf, 0, gf);
    }

    if (this.islandBuf) this.uploadIslands(fp, ox, oz);
    if (this.ruinBoxBuf) this.uploadRuins(fp, ox, oz);
    if (this.sky) this.sky.setFrame(fp.days ?? DEFAULT_DAYS, fp.y, fp.time);
    if (this.forest) this.forest.setFrame(fp, this.sky?.rig.key);
    if (this.forest || this.sky) {
      const which = this.forest ? this.forest.sampled() : this.sky!.sampled();
      [this.computeBind, this.bind] = [this.computeBinds[which], this.binds[which]];
    }
  }

  /**
   * The islands near the camera, chunk-local. The list is re-picked when the
   * camera has moved 200 m (or the seed or chunk changed); the uniform is
   * rewritten only then.
   */
  private uploadIslands(fp: FrameParams, ox: number, oz: number): void {
    const [lx, lz, ls] = this.islandsAt;
    if (Math.hypot(fp.x - lx, fp.z - lz) < 200 && ls === fp.seed && this.islandOrigin[0] === ox && this.islandOrigin[1] === oz) return;
    this.islandsAt = [fp.x, fp.z, fp.seed];
    this.islandOrigin = [ox, oz];
    this.islands = islandsNear(fp.x, fp.z, 7000, fp.seed, MAX_ISLANDS);
    const u32 = new Uint32Array(this.islandData);
    const f = new Float32Array(this.islandData);
    const i32 = new Int32Array(this.islandData);
    u32.fill(0);
    u32[0] = this.islands.length;
    this.islands.forEach((isl, k) => {
      const a = 4 + 4 * k, b = 4 + 4 * MAX_ISLANDS + 4 * k, s = 4 + 8 * MAX_ISLANDS + 4 * k;
      f[a] = isl.cx - ox; f[a + 1] = isl.cz - oz; f[a + 2] = isl.radius; f[a + 3] = isl.y;
      f[b] = isl.depth; f[b + 1] = isl.a1; f[b + 2] = isl.a2; f[b + 3] = isl.a3;
      i32[s] = isl.seed;
    });
    this.device.queue.writeBuffer(this.islandBuf!, 0, this.islandData);

    // Their waterfalls, landing on the terrain (or the water) straight below.
    const ground = (x: number, z: number) => Math.max(elevation(x, z, fp.seed, 0), VALLEY.wLevel);
    this.falls = [];
    for (const isl of this.islands) {
      const w = islandWaterfall(isl, ground);
      if (w && this.falls.length < MAX_FALLS) this.falls.push(w);
    }
    const fu = new Uint32Array(this.fallsData);
    const ff = new Float32Array(this.fallsData);
    fu.fill(0);
    fu[0] = this.falls.length;
    this.falls.forEach((w, k) => {
      const a = 4 + 4 * k, b = 4 + 4 * MAX_FALLS + 4 * k;
      ff[a] = w.x - ox; ff[a + 1] = w.y; ff[a + 2] = w.z - oz; ff[a + 3] = w.width;
      ff[b] = w.dirX; ff[b + 1] = w.dirZ; ff[b + 2] = w.bottom; ff[b + 3] = (w.seed & 0xffff) * 0.37;
    });
    this.device.queue.writeBuffer(this.fallsBuf!, 0, this.fallsData);
  }

  /** Queue ring r's triangulation when its cells (or the world under them) changed. */
  private markRing(r: number, corner: [number, number], hole: [number, number], seed: number): void {
    const key = this.ringKeys[r];
    const hx = r === 0 ? 0 : hole[0], hz = r === 0 ? 0 : hole[1];
    if (key[0] === corner[0] && key[1] === corner[1] && key[2] === hx && key[3] === hz && key[4] === seed) return;
    key[0] = corner[0]; key[1] = corner[1]; key[2] = hx; key[3] = hz; key[4] = seed;
    this.device.queue.writeBuffer(this.ringJobBuf[r], 0, new Int32Array([r, r === 0 ? 0 : 1, hx, hz]));
    this.ringDirty[r] = true;
  }

  /** The ruin sites near the camera (re-picked every 100 m), their boxes chunk-local. */
  private uploadRuins(fp: FrameParams, ox: number, oz: number): void {
    const [lx, lz, ls, lox, loz] = this.ruinsAt;
    if (Math.hypot(fp.x - lx, fp.z - lz) < 100 && ls === fp.seed && lox === ox && loz === oz) return;
    this.ruinsAt = [fp.x, fp.z, fp.seed, ox, oz];
    const sites = ruinSitesNear(fp.x, fp.z, RUINS.drawRange, fp.seed, MAX_RUIN_SITES);
    const bf = new Float32Array(this.ruinBoxData);
    let n = 0;
    for (const site of sites) {
      site.boxes.forEach((b, j) => {
        if (n >= MAX_RUIN_BOXES) return;
        const o = n * 8;
        bf[o] = b.x - ox; bf[o + 1] = b.y; bf[o + 2] = b.z - oz; bf[o + 3] = b.yaw;
        // The fraction carries a per-box hash (block tints, shrine patterns).
        bf[o + 4] = b.hx; bf[o + 5] = b.hy; bf[o + 6] = b.hz;
        bf[o + 7] = b.material + 0.99 * hash01(j, Math.floor(site.x), fp.seed + Math.floor(site.z));
        n++;
      });
    }
    this.ruinBoxCount = n;
    if (n) this.device.queue.writeBuffer(this.ruinBoxBuf!, 0, this.ruinBoxData, 0, n * RUIN_BOX_BYTES);
    const su = new Uint32Array(this.ruinSitesData);
    const sf = new Float32Array(this.ruinSitesData);
    su.fill(0);
    su[0] = sites.length;
    sites.forEach((site, k) => {
      sf[4 + 4 * k] = site.x - ox; sf[5 + 4 * k] = site.z - oz; sf[6 + 4 * k] = site.radius; sf[7 + 4 * k] = site.top;
    });
    this.device.queue.writeBuffer(this.ruinSitesBuf!, 0, this.ruinSitesData);
  }

  encodeGenerate(enc: GPUCommandEncoder, ts?: PassTimestamps): number {
    return this.cache.encodeGenerate(enc, ts);
  }

  encodeRender(enc: GPUCommandEncoder, target: GPUTextureView, ts?: PassTimestamps): void {
    if (!this.depthView) throw new Error('raster: setFrame before encodeRender');
    // The sky's LUT and light first: every surface and the fog read them.
    this.sky?.encode(enc);
    // Trees next: their shadow map is read by the grass scatter and every surface.
    this.forest?.encode(enc, this.computeBind);
    const grass = this.variant.grass;
    if (grass && this.scatterPipe && this.argsBuf && this.scatterBind) {
      enc.clearBuffer(this.argsBuf, 4, 4); // instance count
      const cp = enc.beginComputePass();
      cp.setPipeline(this.scatterPipe);
      cp.setBindGroup(0, this.computeBind);
      cp.setBindGroup(1, this.scatterBind);
      cp.dispatchWorkgroups(grass.n / 8, grass.n / 8, grass.rings);
      cp.end();
    }
    if (this.variant.diag === 'always') this.ringDirty.fill(true);
    if (this.diagPipe && this.ringDirty.some(Boolean)) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.diagPipe);
      cp.setBindGroup(0, this.computeBind);
      const k = this.variant.k;
      for (let r = 0; r < this.rings; r++) {
        if (!this.ringDirty[r]) continue;
        cp.setBindGroup(1, this.ringJobBind[r]);
        cp.dispatchWorkgroups(((r === 0 ? 4 : 3) * k * k) / 64);
        this.ringDirty[r] = false;
      }
      cp.end();
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [this.taa
        ? { view: this.taa.sceneView!, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }
        : this.msaaView
          ? { view: this.msaaView, resolveTarget: target, loadOp: 'clear', storeOp: 'discard', clearValue: { r: 0, g: 0, b: 0, a: 1 } }
          : { view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: this.depthView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: this.taa ? 'store' : 'discard' },
      ...(ts ? { timestampWrites: { querySet: ts.querySet, beginningOfPassWriteIndex: ts.begin, endOfPassWriteIndex: ts.end } } : {}),
    });
    pass.setBindGroup(0, this.bind);
    pass.setPipeline(this.terrainPipe);
    if (this.diagPipe) {
      for (let r = 0; r < this.rings; r++) {
        pass.setIndexBuffer(this.ringIdx[r], 'uint32');
        pass.drawIndexed(r === 0 ? this.fullCount : this.holeCount, 1, 0, 0, r);
      }
    } else {
      pass.setIndexBuffer(this.fullIdx!, 'uint32');
      pass.drawIndexed(this.fullCount, 1, 0, 0, 0);
      for (let r = 1; r < this.rings; r++) {
        const [dx, dz] = this.holes[r];
        pass.setIndexBuffer(this.holeIdx[dx + 2 * dz], 'uint32');
        pass.drawIndexed(this.holeCount, 1, 0, 0, r);
      }
    }
    if (this.ruinPipe && this.ruinBoxCount) {
      pass.setPipeline(this.ruinPipe);
      pass.draw(36, this.ruinBoxCount);
    }
    if (this.islandPipe && this.islandIdxBuf && this.islands.length) {
      pass.setPipeline(this.islandPipe);
      pass.setIndexBuffer(this.islandIdxBuf, 'uint32');
      pass.drawIndexed(this.islandIdxCount, 2 * this.islands.length);
    }
    this.forest?.draw(pass);
    if (this.grassPipe && this.drawBind && this.argsBuf) {
      pass.setPipeline(this.grassPipe);
      pass.setBindGroup(1, this.drawBind);
      pass.drawIndirect(this.argsBuf, 0);
    }
    // The hero before the water, so the water's blend tints whatever of it is submerged.
    if (this.heroScissor) {
      const [x, y, w, h] = this.heroScissor;
      pass.setScissorRect(x, y, w, h);
      pass.setPipeline(this.heroPipe);
      pass.draw(3);
      pass.setScissorRect(0, 0, this.size[0], this.size[1]);
    }
    pass.setPipeline(this.skyPipe);
    pass.draw(3);
    // Blended passes last: water over the lake bed, then waterfalls and their mist.
    pass.setPipeline(this.waterPipe);
    pass.draw(6);
    if (this.fallPipe && this.mistPipe && this.falls.length) {
      pass.setPipeline(this.fallPipe);
      pass.draw(2 * (FALL_SEGMENTS + 1), this.falls.length);
      pass.setPipeline(this.mistPipe);
      pass.draw(4, this.falls.length);
    }
    pass.end();
    if (this.taa) {
      const c = this.cam;
      this.taa.setCamera(c.yaw, c.pitch, c.x, c.y, c.z, c.tanHalf);
      this.taa.encode(enc, target);
    }
  }

  /** Trees drawn last frame: near (levels 0 and 1) and far (debug readback; waits for the GPU). */
  async treeCount(): Promise<[number, number]> {
    if (!this.forest) return [0, 0];
    const [near, far] = await this.forest.counts();
    return [near, far];
  }

  /** The forest's placement window: side, then its tiles and slots (debug readback; waits for the GPU). */
  async forestSlots(): Promise<{ side: number; data: ArrayBuffer } | null> {
    return this.forest ? { side: this.forest.spec.side, data: await this.forest.readSlots() } : null;
  }

  /** One level of the clipmap's sun layer and its spacing (debug readback; waits for the GPU). */
  async sunLayer(level: number): Promise<{ n: number; spacing: number; channels: number; data: Float32Array }> {
    const { channels, data } = await this.cache.readSunLayer(level);
    return { n: this.variant.clip.n, spacing: this.variant.clip.spacing0 * 2 ** level, channels, data };
  }

  /** The day–night sky's probe of the last frame and its light rig (debug readback; waits for the GPU). */
  async skyProbe(): Promise<{ light: number[]; exposure: number; hour: number; key: number[] } | null> {
    if (!this.sky) return null;
    const light = Array.from(await this.sky.readLight());
    const r = this.sky.rig;
    return { light, exposure: light[36], hour: r.hour, key: Array.from(r.keyCols) };
  }

  /** Blades the last scatter kept (debug readback; waits for the GPU). */
  async grassCount(): Promise<number> {
    if (!this.argsBuf) return 0;
    const rb = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.argsBuf, 0, rb, 0, 16);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const n = new Uint32Array(rb.getMappedRange())[1];
    rb.unmap();
    rb.destroy();
    return n;
  }

  destroy(): void {
    this.frameBuf.destroy();
    this.heroBuf.destroy();
    this.grassBuf?.destroy();
    this.bladeBuf?.destroy();
    this.argsBuf?.destroy();
    for (const b of [this.islandBuf, this.islandIdxBuf, this.fallsBuf, this.ruinBoxBuf, this.ruinSitesBuf]) b?.destroy();
    this.meshBuf?.destroy();
    this.fullIdx?.destroy();
    for (const b of this.holeIdx) b.destroy();
    for (const b of [...this.ringIdx, ...this.ringJobBuf]) b.destroy();
    this.depth?.destroy();
    this.msaaColor?.destroy();
    this.taa?.destroy();
    this.forest?.destroy();
    this.sky?.destroy();
    this.cache.destroy();
  }
}

/**
 * Framebuffer rectangle [x, y, w, h] covering a camera-relative sphere, from
 * the frame's basis (F[0..2] right, F[4..6] up, F[8..10] forward) and the
 * projection scales sx = f/aspect, sy = f; null when it is off screen, the
 * whole framebuffer when the sphere reaches the near plane.
 */
function sphereScissor(
  F: Float32Array, cx: number, cy: number, cz: number, r: number, sx: number, sy: number, w: number, h: number,
): [number, number, number, number] | null {
  const vx = cx * F[0] + cy * F[1] + cz * F[2];
  const vy = cx * F[4] + cy * F[5] + cz * F[6];
  const vz = cx * F[8] + cy * F[9] + cz * F[10];
  if (vz + r <= CAMERA.near) return null;
  if (vz - r <= CAMERA.near) return [0, 0, w, h];
  // Extremes of X/Z over the view-space box around the sphere (conservative).
  const x0 = Math.min((vx - r) / (vz - r), (vx - r) / (vz + r)) * sx;
  const x1 = Math.max((vx + r) / (vz - r), (vx + r) / (vz + r)) * sx;
  const y0 = Math.min((vy - r) / (vz - r), (vy - r) / (vz + r)) * sy;
  const y1 = Math.max((vy + r) / (vz - r), (vy + r) / (vz + r)) * sy;
  const px0 = Math.max(0, Math.floor((x0 * 0.5 + 0.5) * w));
  const px1 = Math.min(w, Math.ceil((x1 * 0.5 + 0.5) * w));
  const py0 = Math.max(0, Math.floor((0.5 - y1 * 0.5) * h));
  const py1 = Math.min(h, Math.ceil((0.5 - y0 * 0.5) * h));
  if (px1 <= px0 || py1 <= py0) return null;
  return [px0, py0, px1 - px0, py1 - py0];
}
