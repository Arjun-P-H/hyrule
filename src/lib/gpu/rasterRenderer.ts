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

import { HydroBuffer, hydroBufferFor } from './hydroBuffer';
import { GroundTextures, groundTexturesFor } from './groundTex';
import { CAMERA, LIGHT, MAX_WORLD_H, RUINS } from '../constants';
import { CombatGpu } from './combatGpu';
import { cornersScissor, sphereScissor } from './scissor';
import { HERO_UNIFORM } from '../hero/rig';
import { ClipmapCache, type PassTimestamps } from './clipmapCache';
import { FRAME_BYTES, meshBytes, rasterShader } from './terrain.wgsl';
import { HERO_BOX, HERO_BYTES, packHeroOthers, packHeroWgsl } from './hero.wgsl';
import { BLADE_BYTES, BLADE_SEGS, bladeVerts, GRASS_FAR_RING, grassBytes } from './grass.wgsl';
import { SkyRealm } from './skyRealm';
import { Structures } from './structures';
import { Fauna } from './fauna';
import { Motes } from './motes';
import { LanternMap } from './lanterns';
import { MAX_RUIN_BOXES, MAX_RUIN_SITES, RUIN_BOX_BYTES, RUIN_SITES_BYTES } from './ruins.wgsl';
import { ruinSitesNear } from '../ruins';
import { MAX_PUFFS, PUFF_BYTES, PUFF_VERTS } from './puffs.wgsl';
import { billowsNear, puffChainsNear, type Puff, type PuffChain } from '../skyworld/puffs';
import { hash01 } from '../terrainMath';
import { DEFAULT_DAYS, MAX_REMOTE_HEROES, floatingOrigin, packFrame, type FrameParams, type TerrainRendererLike, type TerrainVariant } from './frame';
import { boxInView, RING_BLOCK, ringBlockOrder, ringIndices, ringLayout } from './rings';
import { SCENE_FORMAT, TemporalResolve } from './taa';
import { Forest } from './forest';
import { Rocks } from './rocks';
import { ModelIsles } from './modelIsles';
import { SkyCycle, type SkyBindings } from './sky';
import { biomeAir, createBiomeAir } from '../sky/air';
import { stamped } from './profile';

type RasterVariant = Extract<TerrainVariant, { height: 'raster' }>;

/** Where the day–night sky's resources sit in group 0 (sky.wgsl.ts cycleSkyWGSL). */
const SKY_BINDINGS: SkyBindings = { sky: 17, light: 18, view: 19 };

/** The classic sky's fixed sun (unit). */
const CLASSIC_SUN = (() => {
  const [x, y, z] = LIGHT.sunDir;
  const l = Math.hypot(x, y, z);
  return [x / l, y / l, z / l];
})();


/** How far from the camera puffs are drawn (m): past it they are under a pixel or two. */
const PUFF_RANGE = 1500;
/** How far the cloud sea's lobes are drawn (m): past it the sea's own domes stand alone. */
const BILLOW_RANGE = 2000;
/** How far the islands' banks of cloud are drawn (m): a few lobes each, so far out. */
const BANK_RANGE = 6000;
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
  // other players' heroes (multiplayer): a block each, drawn by the hero pass with group 0 pointing at it
  private remoteBufs: GPUBuffer[] = [];
  private remoteData = Array.from({ length: MAX_REMOTE_HEROES }, () => new Float32Array(HERO_BYTES / 4));
  private remoteScissor: Array<[number, number, number, number] | null> = new Array(MAX_REMOTE_HEROES).fill(null);
  private remoteBinds: GPUBindGroup[][] = []; // [slot][version of group 0]
  private bindVersion = 0;
  private bind!: GPUBindGroup;
  private computeBind!: GPUBindGroup;
  // Group 0 comes in a version for each combination of the double-buffered resources sampled: the
  // trees' shadow map (Forest.sampled(): flips every frame), the sky's LUT set (SkyCycle.sampled())
  // and the realm's shadow map (SkyRealm.sampled()), each of the last two flipping when one is made.
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
  // variant.diag: one index buffer per ring, rebuilt by ringTris when the ring moves, its cells block by
  // block (rings.ts ringBlockOrder); each frame only the blocks in view are drawn (ringInView).
  private diagPipe: GPUComputePipeline | null = null;
  private blockOrder: Uint16Array = new Uint16Array(0);
  /** Per ring this frame: first vertex (level-0 lattice units, world) x, z; level-0 units per cell; hole x, z. */
  private ringAt = new Float64Array(0);
  private basis = new Float32Array(12);
  private view = { tx: 1, ty: 1, x: 0, y: 0, z: 0 };
  private ringIdx: GPUBuffer[] = [];
  private ringJobBuf: GPUBuffer[] = [];
  private ringJobBind: GPUBindGroup[] = [];
  // the rings' cells that hold water (ringTris appends them) and per ring its indirect draw (vsWaterCell)
  private waterCells: GPUBuffer | null = null;
  private waterArgs: GPUBuffer | null = null;
  private waterBind: GPUBindGroup | null = null;
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
  // (two blade sets, near and far: grass.wgsl.ts GRASS_FAR_RING; their draw arguments 16 bytes apart)
  private bladeBuf: GPUBuffer | null = null;
  private bladeFarBuf: GPUBuffer | null = null;
  /** The sky's blades per cell of every ring (grass.wgsl.ts skyBlade), made once a cell enters a ring. */
  private bladeCacheBuf: GPUBuffer | null = null;
  private argsBuf: GPUBuffer | null = null;
  private scatterPipe: GPUComputePipeline | null = null;
  private grassPipe: GPURenderPipeline | null = null;
  private grassFarPipe: GPURenderPipeline | null = null;
  private scatterBind: GPUBindGroup | null = null;
  private drawBind: GPUBindGroup | null = null;
  private drawFarBind: GPUBindGroup | null = null;
  // the sky realm (variant.islands): archipelagos and isles round the camera, their water, arches, falls, shadows
  private realm: SkyRealm | null = null;
  // the Eyries' structures on the realm's islands (with the realm), and the district maps last seen
  private structures: Structures | null = null;
  // the birds round the camera (with the realm)
  private fauna: Fauna | null = null;
  // what drifts in the air round the camera up there (with the realm)
  private motes: Motes | null = null;
  // the lanterns' light round the camera (with the realm)
  private lanterns: LanternMap | null = null;
  // combat (variant.combat): the foes, what flies and flares
  private combat: CombatGpu | null = null;
  private seenDistricts = 0;
  private seenPlants = 0;
  // ruins (variant.ruins): boxes of the sites near the camera, and the site list for the scatters
  private ruinBoxBuf: GPUBuffer | null = null;
  private ruinSitesBuf: GPUBuffer | null = null;
  private ruinBoxData = new ArrayBuffer(MAX_RUIN_BOXES * RUIN_BOX_BYTES);
  private ruinSitesData = new ArrayBuffer(RUIN_SITES_BYTES);
  private ruinBoxCount = 0;
  private ruinPipe: GPURenderPipeline | null = null;
  private ruinsAt: [number, number, number, number, number] = [NaN, NaN, NaN, NaN, NaN];
  // puffs (variant.puffs): the cloud world's stepping clouds near the camera
  private puffBuf: GPUBuffer | null = null;
  private puffData = new ArrayBuffer(MAX_PUFFS * PUFF_BYTES);
  private puffCount = 0;
  private puffPipe: GPURenderPipeline | null = null;
  private puffsAt: [number, number, number, number, number] = [NaN, NaN, NaN, NaN, NaN];
  private puffChains: PuffChain[] = [];
  private billows: Puff[] = [];
  // trees (variant.forest): placement cache, culling, draws, shadow map
  private forest: Forest | null = null;
  // rocks (variant.rocks): the scatter and the draws
  private rocks: Rocks | null = null;
  // the land's model isles (variant.modelIsles)
  private modelIsles: ModelIsles | null = null;
  // the day–night sky (variant.sky = 'cycle'): its LUTs, the frame's light rig and the probe of the sky
  private sky: SkyCycle | null = null;
  // the lands' air round the camera (sky/air.ts), for the day–night sky's haze
  private air = createBiomeAir();

  constructor(device: GPUDevice, format: GPUTextureFormat, variant: RasterVariant) {
    const { clip, k, stride } = variant;
    if (k % 2 !== 0) throw new Error(`raster: k must be even: ${k}`);
    if (k * stride > clip.n / 2 - 4) throw new Error(`raster: k·stride = ${k * stride} exceeds the clipmap window (n/2 − 4 = ${clip.n / 2 - 4})`);
    this.device = device;
    this.format = format;
    this.variant = variant;
    this.rings = clip.levels;
    this.cache = new ClipmapCache(device, clip, variant.genDpScale, { blend: variant.blend, mode: variant.sky === 'cycle' ? 'path' : 'classic' }, variant.world);
    this.frameBuf = device.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.heroBuf = device.createBuffer({ size: HERO_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let k = 0; k < MAX_REMOTE_HEROES; k++) {
      this.remoteBufs.push(device.createBuffer({ size: HERO_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
    this.meshData = new ArrayBuffer(meshBytes(this.rings));
    this.meshF32 = new Float32Array(this.meshData);
    this.meshI32 = new Int32Array(this.meshData);
  }

  async init(): Promise<void> {
    const d = this.device;
    const v = this.variant;
    const grass = v.grass ?? null;
    if (v.forest) this.forest = new Forest(d, v.forest);
    if (v.rocks) this.rocks = new Rocks(d);
    if (v.modelIsles) this.modelIsles = new ModelIsles(d);
    const cycle = v.sky === 'cycle';
    if (cycle) this.sky = new SkyCycle(d);
    const skyReady = this.sky?.init();
    // The ground's textures: painted once per device (the other pipelines are built while it is).
    const ground = groundTexturesFor(d);
    const groundReady = ground.init();
    const islands = !!v.islands;
    // Clip distances cut the sky islands' tops along their outlines (without them: discard).
    const clip = d.features.has('clip-distances');
    if (islands) this.realm = new SkyRealm(d, typeof v.islands === 'object' ? v.islands : {});
    if (islands && !(typeof v.islands === 'object' && v.islands.structures === false)) this.structures = new Structures(d, typeof v.islands === 'object' && !!v.islands.syncBuild);
    if (islands) this.fauna = new Fauna(d);
    if (islands) this.motes = new Motes(d);
    if (islands) this.lanterns = new LanternMap(d);
    if (v.combat) this.combat = new CombatGpu(d);
    const ruins = !!v.ruins;
    const puffs = !!v.puffs;
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
      { binding: 28, visibility: VFC, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 25, visibility: VFC, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      // the ground's biome layers (clipmapCache.ts bioViews): soil + cover, grass + blade, rock + strata
      ...[38, 39, 40].map((binding) => ({ binding, visibility: VFC, texture: { sampleType: 'float' as const, viewDimension: '2d-array' as const } })),
      // the ground's detail textures (groundTex.ts), read by the surfaces' fragment shaders
      ...GroundTextures.layoutEntries(),
      ...(this.forest ? Forest.layoutEntries() : []),
      ...(islands ? SkyRealm.layoutEntries() : []),
      ...(islands ? LanternMap.layoutEntries() : []),
      // the plants of the land's model isles, for the forest's sky layer
      ...(this.modelIsles && this.forest ? [ModelIsles.plantsLayoutEntry()] : []),
      ...(ruins ? [{ binding: 13, visibility: VFC, buffer: { type: 'uniform' as const } }] : []),
      ...(cycle ? SkyCycle.layoutEntries(SKY_BINDINGS) : []),
      // the rivers' node table (the forest's placement evaluates the land)
      HydroBuffer.layoutEntry(GPUShaderStage.COMPUTE),
    ];
    // Group 0 for the compute passes, and for drawing: the latter adds the ruin boxes.
    const computeLayout = d.createBindGroupLayout({ entries: shared });
    const renderOnly: GPUBindGroupLayoutEntry[] = [
      ...(ruins ? [{ binding: 12, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' as const } }] : []),
      ...(puffs ? [{ binding: 30, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' as const } }] : []),
    ];
    const layout = renderOnly.length ? d.createBindGroupLayout({ entries: [...shared, ...renderOnly] }) : computeLayout;
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
    const f16 = d.features.has('shader-f16') && !v.shadeF32;
    let code = rasterShader({ n: v.clip.n, levels: v.clip.levels, blend: v.blend, world: v.world, k: v.k, rings: this.rings, grass, islands, clip, ruins, puffs, diag: !!v.diag, f16,
      forest: this.forest ? { spec: this.forest.spec, bounds: this.forest.models.bounds } : null, sky: cycle ? 'cycle' : 'classic', combat: !!v.combat, rocks: !!v.rocks, modelIsles: !!v.modelIsles });
    if (v.privateTables) code = code.replace(/^const ([A-Za-z_0-9]+): array</gm, 'var<private> $1: array<');
    for (const k of v.stubAll ?? []) code = code.replace(`override ${k}: bool = false;`, `override ${k}: bool = true;`);
    const shaders = d.createShaderModule({ code });
    const target = [{ format: v.taa ? SCENE_FORMAT : this.format }];
    const depthState = (write: boolean, compare: GPUCompareFunction): GPUDepthStencilState => ({ format: 'depth32float', depthWriteEnabled: write, depthCompare: compare });
    const multisample: GPUMultisampleState = { count: v.taa ? 1 : v.msaa ?? 1 };
    if (v.taa) {
      this.taa = new TemporalResolve(d, this.format);
      this.taa.halfPrecision = !v.taaF32;
      // Sharpen what is upscaled; at 1:1 the resolved image is left soft (closest to supersampling).
      this.taa.sharpen = v.taaSharpen ?? ((v.renderScale ?? 1) < 0.99 ? 0.5 : 0.15);
      if (v.taaAlpha !== undefined) this.taa.alpha = v.taaAlpha;
      await this.taa.init();
    }
    const scatterLayout = grass ? d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    }) : null;
    const drawLayout = grass ? d.createBindGroupLayout({
      entries: [{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
    }) : null;
    const ruinPipeP = ruins ? d.createRenderPipelineAsync({
      layout: pl, vertex: { module: shaders, entryPoint: 'vsRuin' }, fragment: { module: shaders, entryPoint: 'fsRuin', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
    }) : Promise.resolve(null);
    const puffPipeP = puffs ? d.createRenderPipelineAsync({
      layout: pl, vertex: { module: shaders, entryPoint: 'vsPuff' }, fragment: { module: shaders, entryPoint: 'fsPuff', targets: target },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
    }) : Promise.resolve(null);
    const realmP = this.realm?.init(shaders, computeLayout, layout, target, depthState(true, 'greater'), multisample);
    const ringJobLayout = v.diag ? d.createBindGroupLayout({
      entries: [
        { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    }) : null;
    // The water's draw over the rings' water lists (with diag): its cells, read by vsWaterCell.
    const waterLayout = v.diag ? d.createBindGroupLayout({
      entries: [{ binding: 18, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
    }) : null;
    const diagPipeP = ringJobLayout ? d.createComputePipelineAsync({
      layout: d.createPipelineLayout({ bindGroupLayouts: [computeLayout, ringJobLayout] }),
      compute: { module: shaders, entryPoint: 'ringTris' },
    }) : Promise.resolve(null);
    const grassDesc = (segs: number): GPURenderPipelineDescriptor => ({
      layout: d.createPipelineLayout({ bindGroupLayouts: [layout, drawLayout!] }),
      vertex: { module: shaders, entryPoint: 'vsGrass', constants: { BLADE_SEGS: segs } },
      fragment: { module: shaders, entryPoint: 'fsGrass', targets: target },
      primitive: { topology: 'triangle-strip', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
    });
    const [terrain, water, heroP, sky, , scatterP, grassP, grassFarP] = await Promise.all([
      d.createRenderPipelineAsync({
        layout: pl, vertex: { module: shaders, entryPoint: 'vsTerrain' }, fragment: { module: shaders, entryPoint: 'fsTerrain', targets: target, ...(v.stub ? { constants: v.stub } : {}) },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depthState(true, 'greater'), multisample,
      }),
      d.createRenderPipelineAsync({
        layout: waterLayout ? d.createPipelineLayout({ bindGroupLayouts: [layout, waterLayout] }) : pl,
        vertex: { module: shaders, entryPoint: waterLayout ? 'vsWaterCell' : 'vsWater' },
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
      drawLayout ? d.createRenderPipelineAsync(grassDesc(BLADE_SEGS[0])) : Promise.resolve(null),
      drawLayout ? d.createRenderPipelineAsync(grassDesc(BLADE_SEGS[1])) : Promise.resolve(null),
    ]);
    this.terrainPipe = terrain;
    this.waterPipe = water;
    this.heroPipe = heroP;
    this.skyPipe = sky;
    this.scatterPipe = scatterP;
    this.grassPipe = grassP;
    this.grassFarPipe = grassFarP;
    await realmP;
    this.ruinPipe = await ruinPipeP;
    this.puffPipe = await puffPipeP;
    this.diagPipe = await diagPipeP;
    await skyReady;
    await groundReady;
    if (this.forest) await this.forest.init(shaders, computeLayout, layout, this.frameBuf, target, depthState(true, 'greater'), multisample);
    await this.structures?.init(shaders, layout, target, depthState(true, 'greater'), multisample, this.forest?.shadowLayout);
    await this.fauna?.init(shaders, layout, target, depthState(true, 'greater'), multisample);
    await this.motes?.init(shaders, layout, target, depthState(true, 'greater'), multisample);
    await this.combat?.init(shaders, layout, target, depthState(true, 'greater'), multisample);
    await this.rocks?.init(shaders, computeLayout, layout, target, depthState(true, 'greater'), multisample, this.forest?.shadowLayout ?? null);
    await this.modelIsles?.init(shaders, layout, target, depthState(true, 'greater'), multisample, !!this.forest);
    if (ringJobLayout && waterLayout) {
      const k = v.k;
      this.blockOrder = ringBlockOrder(k).order;
      this.ringAt = new Float64Array(5 * this.rings);
      const ringCells = 4 * k * k;
      this.waterCells = d.createBuffer({ size: this.rings * ringCells * 4, usage: GPUBufferUsage.STORAGE });
      this.waterArgs = d.createBuffer({ size: this.rings * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
      // (each ring's draw: no vertices until its list is made; one instance; its list's first vertex)
      d.queue.writeBuffer(this.waterArgs, 0, new Uint32Array(Array.from({ length: this.rings }, (_, r) => [0, 1, 6 * r * ringCells, 0]).flat()));
      this.waterBind = d.createBindGroup({ layout: waterLayout, entries: [{ binding: 18, resource: { buffer: this.waterCells } }] });
      for (let r = 0; r < this.rings; r++) {
        // (every cell has its place, those under the finer ring degenerate)
        const cells = 4 * k * k;
        const idx = d.createBuffer({ size: cells * 6 * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE });
        const job = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.ringIdx.push(idx);
        this.ringJobBuf.push(job);
        this.ringJobBind.push(d.createBindGroup({
          layout: ringJobLayout,
          entries: [
            { binding: 14, resource: { buffer: job } }, { binding: 15, resource: { buffer: idx } },
            { binding: 16, resource: { buffer: this.waterCells } }, { binding: 17, resource: { buffer: this.waterArgs } },
          ],
        }));
        this.ringKeys.push([NaN, NaN, NaN, NaN, NaN]);
        this.ringDirty.push(true);
      }
    }
    if (ruins) {
      this.ruinBoxBuf = d.createBuffer({ size: MAX_RUIN_BOXES * RUIN_BOX_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.ruinSitesBuf = d.createBuffer({ size: RUIN_SITES_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (puffs) this.puffBuf = d.createBuffer({ size: MAX_PUFFS * PUFF_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.realm?.setFrameBuffer(this.frameBuf);
    this.compileMs = performance.now() - t0;

    if (grass && scatterLayout && drawLayout) {
      const nearRings = Math.min(grass.rings, GRASS_FAR_RING);
      const ringBlades = grass.n * grass.n;
      this.grassBuf = d.createBuffer({ size: grassBytes(grass.rings), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const gd = new ArrayBuffer(grassBytes(grass.rings));
      this.grassF32 = new Float32Array(gd);
      this.grassI32 = new Int32Array(gd);
      this.bladeBuf = d.createBuffer({ size: nearRings * ringBlades * BLADE_BYTES, usage: GPUBufferUsage.STORAGE });
      this.bladeFarBuf = d.createBuffer({ size: Math.max(grass.rings - nearRings, 1) * ringBlades * BLADE_BYTES, usage: GPUBufferUsage.STORAGE });
      this.bladeCacheBuf = d.createBuffer({ size: grass.rings * ringBlades * 16, usage: GPUBufferUsage.STORAGE });
      this.argsBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      d.queue.writeBuffer(this.argsBuf, 0, new Uint32Array([bladeVerts(BLADE_SEGS[0]), 0, 0, 0, bladeVerts(BLADE_SEGS[1]), 0, 0, 0]));
      this.scatterBind = d.createBindGroup({
        layout: scatterLayout,
        entries: [
          { binding: 0, resource: { buffer: this.bladeBuf } },
          { binding: 1, resource: { buffer: this.argsBuf } },
          { binding: 3, resource: { buffer: this.bladeFarBuf } },
          { binding: 4, resource: { buffer: this.bladeCacheBuf } },
        ],
      });
      this.drawBind = d.createBindGroup({ layout: drawLayout, entries: [{ binding: 2, resource: { buffer: this.bladeBuf } }] });
      this.drawFarBind = d.createBindGroup({ layout: drawLayout, entries: [{ binding: 2, resource: { buffer: this.bladeFarBuf } }] });
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
      { binding: 28, resource: this.cache.matView! },
      { binding: 25, resource: this.cache.canopyView! },
      ...this.cache.bioViews.map((v, i) => ({ binding: 38 + i, resource: v })),
      ...ground.bindEntries(),
      ...(this.ruinSitesBuf ? [{ binding: 13, resource: { buffer: this.ruinSitesBuf } }] : []),
      hydroBufferFor(d).entry(),
      ...(this.modelIsles && this.forest ? [this.modelIsles.plantsEntry()] : []),
    ];
    const renderEntries: GPUBindGroupEntry[] = [
      ...(this.ruinBoxBuf ? [{ binding: 12, resource: { buffer: this.ruinBoxBuf } }] : []),
      ...(this.puffBuf ? [{ binding: 30, resource: { buffer: this.puffBuf } }] : []),
    ];
    // Version which = (forest map sampled) + 2 × (sky set sampled) + 4 × (realm map sampled).
    for (let which = 0; which < 8; which++) {
      if ((which & 1 && !this.forest) || (which & 2 && !this.sky) || (which & 4 && !this.realm)) continue;
      const shared = [
        ...sharedEntries,
        ...(this.forest ? this.forest.bindEntries(which & 1) : []),
        ...(this.sky ? this.sky.bindEntries(SKY_BINDINGS, (which >> 1) & 1) : []),
        ...(this.realm ? this.realm.bindEntries(which >> 2) : []),
        ...(this.lanterns ? this.lanterns.bindEntries() : []),
      ];
      const cb = d.createBindGroup({ layout: computeLayout, entries: shared });
      this.computeBinds[which] = cb;
      this.binds[which] = renderEntries.length ? d.createBindGroup({ layout, entries: [...shared, ...renderEntries] }) : cb;
      // The same group with another player's hero in binding 5 (only the hero pass draws with it).
      for (let k = 0; k < MAX_REMOTE_HEROES; k++) {
        (this.remoteBinds[k] ??= [])[which] = d.createBindGroup({
          layout,
          entries: [...shared, ...renderEntries].map((e) => (e.binding === 5 ? { binding: 5, resource: { buffer: this.remoteBufs[k] } } : e)),
        });
      }
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
    // (and 1 / the outermost ring's reach: where the land, and the water with it, end)
    f[8] = start; f[9] = 1 / (k - 1 - start); f[10] = this.rings - 1; f[11] = 1 / (k * s0 * v.stride * 2 ** (this.rings - 1));
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
      if (this.diagPipe) {
        this.markRing(r, ring.corner, ring.hole, fp.seed);
        const a = this.ringAt;
        a[5 * r] = ring.corner[0]; a[5 * r + 1] = ring.corner[1]; a[5 * r + 2] = ring.step;
        a[5 * r + 3] = ring.hole[0]; a[5 * r + 4] = ring.hole[1];
      }
    }
    // The view, for culling the rings' blocks.
    for (let c = 0; c < 12; c++) this.basis[c] = F[c];
    const vw = this.view;
    // (2 % wider: the TAA's jitter shifts the image by up to a pixel)
    vw.ty = fp.tanHalf * 1.02; vw.tx = fp.tanHalf * aspect * 1.02; vw.x = fp.x; vw.y = fp.y; vw.z = fp.z;
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

    // Hero: WGSL block, and the pixels it can cover (its bounding sphere's, cut down to its box's).
    const H = this.heroData;
    this.heroScissor = null;
    if (fp.hero && fp.hero[HERO_UNIFORM.bound + 3] > 0) {
      packHeroWgsl(fp.hero, H);
      const s = sphereScissor(F, H[96], H[97], H[98], H[99], f0 / aspect, f0, W, Hh);
      this.heroScissor = s && boxScissor(F, H, f0 / aspect, f0, W, Hh, s);
    } else {
      H[99] = 0;
    }
    // (their shadows fall in view even when they don't; the foes' after the other players')
    packHeroOthers(fp.remotes, H, fp.combat?.shadows, fp.combat?.nShadows ?? 0);
    this.device.queue.writeBuffer(this.heroBuf, 0, H);
    this.combat?.setFrame(fp.combat, F, f0 / aspect, f0, W, Hh);
    // Other players: the same, a block each.
    for (let k = 0; k < MAX_REMOTE_HEROES; k++) {
      const src = fp.remotes?.[k];
      this.remoteScissor[k] = null;
      if (!src || !(src[HERO_UNIFORM.bound + 3] > 0)) continue;
      const R = this.remoteData[k];
      packHeroWgsl(src, R);
      const s = sphereScissor(F, R[96], R[97], R[98], R[99], f0 / aspect, f0, W, Hh);
      this.remoteScissor[k] = s && boxScissor(F, R, f0 / aspect, f0, W, Hh, s);
      if (this.remoteScissor[k]) this.device.queue.writeBuffer(this.remoteBufs[k], 0, R);
    }

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
      gf[o] = grass.blend; gf[o + 1] = grass.density; gf[o + 2] = fp.time; gf[o + 3] = Math.max(0, this.realm?.districts ?? 0) % 4096;
      gf[o + 4] = 0.87; gf[o + 5] = 0.49; gf[o + 6] = 0.32; gf[o + 7] = 0.055; // wind: direction, strength, gust scale (1/m)
      this.device.queue.writeBuffer(this.grassBuf, 0, gf);
    }

    if (this.ruinBoxBuf) this.uploadRuins(fp, ox, oz);
    if (this.puffBuf) this.uploadPuffs(fp, ox, oz);
    if (this.sky) this.sky.setFrame(fp.days ?? DEFAULT_DAYS, fp.y, fp.time, biomeAir(fp.x, fp.y, fp.z, fp.seed, this.air));
    // bloom by the hour: lamps, windows and the Thread glow most by night
    if (this.taa) this.taa.bloomStrength = this.sky ? 0.06 + 0.5 * this.sky.rig.night + 0.15 * this.sky.rig.twilight : 0.06;
    // An Eyrie near the camera finished building: its trees keep off its streets and walls from now on.
    // (keyed on the maps the realm has put on the GPU, so the trees are made again once they are there)
    const districts = this.realm?.districts ?? 0;
    if (this.forest && districts !== this.seenDistricts) { this.seenDistricts = districts; this.forest.invalidate(); }
    // (likewise when the model isles whose plants the forest grows change)
    const plants = this.modelIsles?.plantsVersion ?? 0;
    if (this.forest && plants !== this.seenPlants) { this.seenPlants = plants; this.forest.invalidate(); }
    if (this.forest) this.forest.setFrame(fp, this.sky?.rig.key);
    this.rocks?.setFrame(fp);
    this.modelIsles?.setFrame(fp, ox, oz);
    if (this.realm) this.realm.setFrame(fp, ox, oz, this.sky?.rig.key ?? CLASSIC_SUN);
    this.structures?.setFrame(fp, ox, oz);
    if (this.structures && this.lanterns) {
      const L = this.structures.lanternPoints();
      this.lanterns.update(L.pts, L.n, L.version, fp.x, fp.y, fp.z, ox, oz);
    }
    this.fauna?.setFrame(fp, ox, oz);
    this.motes?.setFrame(fp);
    if (this.forest || this.sky || this.realm) {
      const which = (this.forest ? this.forest.sampled() : 0) + 2 * (this.sky ? this.sky.sampled() : 0) + 4 * (this.realm ? this.realm.sampled() : 0);
      [this.computeBind, this.bind] = [this.computeBinds[which], this.binds[which]];
      this.bindVersion = which;
    }
  }

  /** The hero, inside the scissor rectangle it can cover (before the water: its blend tints what of the hero is submerged). */
  private drawHero(pass: GPURenderPassEncoder, on: boolean): void {
    if (!on) return;
    pass.setPipeline(this.heroPipe);
    if (this.heroScissor) {
      const [x, y, w, h] = this.heroScissor;
      pass.setScissorRect(x, y, w, h);
      pass.draw(3);
    }
    // Other players' heroes: each traced in its own rectangle, with its block in group 0. (They
    // cast no shadow on the world: the world's shading reads the player's own hero only.)
    let rebound = false;
    for (let k = 0; k < MAX_REMOTE_HEROES; k++) {
      const r = this.remoteScissor[k];
      if (!r) continue;
      pass.setBindGroup(0, this.remoteBinds[k][this.bindVersion]);
      pass.setScissorRect(r[0], r[1], r[2], r[3]);
      pass.draw(3);
      rebound = true;
    }
    if (rebound) pass.setBindGroup(0, this.bind);
    pass.setScissorRect(0, 0, this.size[0], this.size[1]);
  }

  /**
   * Ring r's blocks that can show in this frame's view, as runs of its index buffer (the blocks lie
   * round the ring by angle, so a view's are one or two runs, plus a few round the camera). A block's
   * box: its cells, a cell wider each way (the morph moves vertices up to a cell), every height the
   * land can take (−300 m … MAX_WORLD_H + 200 m).
   */
  private drawRingInView(pass: GPURenderPassEncoder, r: number): void {
    const k = this.variant.k, nb = (2 * k) / RING_BLOCK, s0 = this.variant.clip.spacing0;
    const a = this.ringAt, vw = this.view, F = this.basis, order = this.blockOrder;
    const cx = a[5 * r], cz = a[5 * r + 1], step = a[5 * r + 2];
    const hx = k / 2 + a[5 * r + 3], hz = k / 2 + a[5 * r + 4];
    const ly = -300 - vw.y, hy = MAX_WORLD_H + 200 - vw.y;
    const per = RING_BLOCK * RING_BLOCK * 6;
    let run = -1;
    for (let sl = 0; sl <= order.length; sl++) {
      let vis = false;
      if (sl < order.length) {
        const b = order[sl], bx = (b % nb) * RING_BLOCK, bz = Math.floor(b / nb) * RING_BLOCK;
        // (wholly under the finer ring: never drawn)
        const hole = r > 0 && bx >= hx && bx + RING_BLOCK <= hx + k && bz >= hz && bz + RING_BLOCK <= hz + k;
        if (!hole) {
          const x0 = (cx + (bx - 1) * step) * s0 - vw.x, x1 = (cx + (bx + RING_BLOCK + 1) * step) * s0 - vw.x;
          const z0 = (cz + (bz - 1) * step) * s0 - vw.z, z1 = (cz + (bz + RING_BLOCK + 1) * step) * s0 - vw.z;
          vis = boxInView(F, vw.tx, vw.ty, x0, ly, z0, x1, hy, z1);
        }
      }
      if (vis && run < 0) run = sl;
      if (!vis && run >= 0) {
        pass.drawIndexed((sl - run) * per, 1, run * per, 0, r);
        run = -1;
      }
    }
  }

  /** Every ring's cells in view (the pipeline set says what they carry: the ground, or the water's surface). */
  private drawRings(pass: GPURenderPassEncoder): void {
    if (this.diagPipe) {
      for (let r = 0; r < this.rings; r++) {
        pass.setIndexBuffer(this.ringIdx[r], 'uint32');
        this.drawRingInView(pass, r);
      }
      return;
    }
    pass.setIndexBuffer(this.fullIdx!, 'uint32');
    pass.drawIndexed(this.fullCount, 1, 0, 0, 0);
    for (let r = 1; r < this.rings; r++) {
      const [dx, dz] = this.holes[r];
      pass.setIndexBuffer(this.holeIdx[dx + 2 * dz], 'uint32');
      pass.drawIndexed(this.holeCount, 1, 0, 0, r);
    }
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

  /** The puffs (within PUFF_RANGE) and the stacks' billows (within BILLOW_RANGE) near the camera (re-picked every 50 m), nearest first, chunk-local. */
  private uploadPuffs(fp: FrameParams, ox: number, oz: number): void {
    const [lx, lz, ls, lox, loz] = this.puffsAt;
    if (Math.hypot(fp.x - lx, fp.z - lz) < 50 && ls === fp.seed && lox === ox && loz === oz) return;
    this.puffsAt = [fp.x, fp.z, fp.seed, ox, oz];
    const near: Array<{ p: Puff; d: number; billow: boolean }> = [];
    for (const c of puffChainsNear(fp.x, fp.z, PUFF_RANGE, fp.seed, this.puffChains)) {
      for (const p of c.puffs) {
        const d = Math.hypot(p.x - fp.x, p.z - fp.z);
        if (d < PUFF_RANGE) near.push({ p, d, billow: false });
      }
    }
    for (const p of billowsNear(fp.x, fp.z, BILLOW_RANGE, fp.seed, this.billows, BANK_RANGE)) near.push({ p, d: Math.hypot(p.x - fp.x, p.z - fp.z), billow: true });
    if (near.length > MAX_PUFFS) near.sort((a, b) => a.d - b.d);
    const bf = new Float32Array(this.puffData);
    const n = Math.min(near.length, MAX_PUFFS);
    for (let k = 0; k < n; k++) {
      const p = near[k].p, o = k * 8;
      bf[o] = p.x - ox; bf[o + 1] = p.y; bf[o + 2] = p.z - oz; bf[o + 3] = near[k].billow ? -1 : p.spring;
      bf[o + 4] = p.rx; bf[o + 5] = p.ry; bf[o + 6] = p.rz; bf[o + 7] = hash01(k, Math.floor(p.x), Math.floor(p.z));
    }
    this.puffCount = n;
    if (n) this.device.queue.writeBuffer(this.puffBuf!, 0, this.puffData, 0, n * PUFF_BYTES);
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
    // The sky realm's shadow map (its undersides and arches seen from the sun), and its rings' caches.
    this.realm?.encodeShadow(enc);
    this.realm?.encodeRings(enc, this.computeBind);
    // The rocks' scatter first (they cast into the trees' shadow map), then the trees: their shadow map is read by the
    // grass scatter and every surface.
    this.rocks?.encode(enc, this.computeBind);
    const casters = this.structures && (!this.variant.skip || !this.variant.skip.includes('structures'));
    const rockCasts = this.rocks && (!this.variant.skip || !this.variant.skip.includes('rocks'));
    this.forest?.encode(enc, this.computeBind, casters || rockCasts ? (sp) => {
      if (rockCasts) this.rocks!.drawShadow(sp);
      if (casters) this.structures!.drawShadow(sp);
    } : undefined);
    const grass = this.variant.grass;
    if (grass && this.scatterPipe && this.argsBuf && this.scatterBind) {
      enc.clearBuffer(this.argsBuf, 4, 4); // instance counts, near and far
      enc.clearBuffer(this.argsBuf, 20, 4);
      const cp = enc.beginComputePass(stamped('grass.scatter'));
      cp.setPipeline(this.scatterPipe);
      cp.setBindGroup(0, this.computeBind);
      cp.setBindGroup(1, this.scatterBind);
      cp.dispatchWorkgroups(grass.n / 8, grass.n / 8, grass.rings);
      cp.end();
    }
    if (this.variant.diag === 'always') this.ringDirty.fill(true);
    if (this.diagPipe && this.ringDirty.some(Boolean)) {
      for (let r = 0; r < this.rings; r++) if (this.ringDirty[r] && this.waterArgs) enc.clearBuffer(this.waterArgs, 16 * r, 4);
      const cp = enc.beginComputePass(stamped('terrain.tris'));
      cp.setPipeline(this.diagPipe);
      cp.setBindGroup(0, this.computeBind);
      const k = this.variant.k;
      for (let r = 0; r < this.rings; r++) {
        if (!this.ringDirty[r]) continue;
        cp.setBindGroup(1, this.ringJobBind[r]);
        // (its water list starts afresh: the vertex count was cleared before the pass)
        cp.dispatchWorkgroups((4 * k * k) / 64);
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
      ...(ts ? { timestampWrites: { querySet: ts.querySet, beginningOfPassWriteIndex: ts.begin, endOfPassWriteIndex: ts.end } } : stamped('render')),
    });
    pass.setBindGroup(0, this.bind);
    const skip = this.variant.skip;
    const drawn = (what: NonNullable<RasterVariant['skip']>[number]) => !skip || !skip.includes(what);
    // Leaf cards' depth first: every opaque draw then skips what lies behind leaves.
    if (drawn('forest')) this.forest?.drawCardDepth(pass);
    // The hero next: it writes its own depth from its shader, so whatever lies behind it, drawn after,
    // is hidden before it is shaded (drawn later it would have every tile under it shaded first).
    if (!this.variant.heroLate) this.drawHero(pass, drawn('hero'));
    // (the foes likewise: traced bodies that write their own depth)
    if (drawn('combat')) this.combat?.drawFoes(pass);
    pass.setPipeline(this.terrainPipe);
    if (drawn('terrain')) this.drawRings(pass);
    if (this.ruinPipe && this.ruinBoxCount && drawn('ruins')) {
      pass.setPipeline(this.ruinPipe);
      pass.draw(36, this.ruinBoxCount);
    }
    if (this.puffPipe && this.puffCount && drawn('puffs')) {
      pass.setPipeline(this.puffPipe);
      pass.draw(PUFF_VERTS, this.puffCount);
    }
    if (drawn('realm')) this.realm?.draw(pass);
    if (drawn('structures')) this.structures?.draw(pass);
    if (drawn('modelIsles')) this.modelIsles?.draw(pass);
    if (drawn('fauna')) this.fauna?.draw(pass);
    if (drawn('combat')) this.combat?.drawParts(pass);
    if (drawn('forest')) this.forest?.draw(pass);
    if (drawn('rocks')) this.rocks?.draw(pass);
    if (this.grassPipe && this.grassFarPipe && this.drawBind && this.drawFarBind && this.argsBuf && drawn('grass')) {
      pass.setPipeline(this.grassPipe);
      pass.setBindGroup(1, this.drawBind);
      pass.drawIndirect(this.argsBuf, 0);
      pass.setPipeline(this.grassFarPipe);
      pass.setBindGroup(1, this.drawFarBind);
      pass.drawIndirect(this.argsBuf, 16);
    }
    if (this.variant.heroLate) this.drawHero(pass, drawn('hero'));
    // The leaf cards' colour, where their depth survived (no discard: one leaf a pixel).
    if (drawn('forest')) this.forest?.drawCardColour(pass);
    if (drawn('sky')) {
      pass.setPipeline(this.skyPipe);
      pass.draw(3);
    }
    // Blended passes last: water over the lake bed, the sky's lakes and streams, its falls and their mist, then the effects.
    if (drawn('water')) {
      // The water's surface: the rings' cells that hold water (their lists), or the rings whole.
      pass.setPipeline(this.waterPipe);
      if (this.waterArgs && this.waterBind) {
        pass.setBindGroup(1, this.waterBind);
        for (let r = 0; r < this.rings; r++) pass.drawIndirect(this.waterArgs, 16 * r);
      } else {
        this.drawRings(pass);
      }
    }
    if (drawn('realm')) {
      this.realm?.drawWater(pass);
      this.realm?.drawFalls(pass);
    }
    if (drawn('modelIsles')) {
      this.modelIsles?.drawWater(pass);
      this.modelIsles?.drawFalls(pass);
    }
    // the Eyries' light and air: the Gyre's Eye, forge smoke, the Thread
    if (drawn('structures')) this.structures?.drawEffects(pass);
    if (drawn('fauna')) this.motes?.draw(pass);
    // combat's sparks, smoke, trails and flames
    if (drawn('combat')) this.combat?.drawSprites(pass);
    pass.end();
    if (this.taa) {
      const c = this.cam;
      this.taa.setCamera(c.yaw, c.pitch, c.x, c.y, c.z, c.tanHalf);
      this.taa.encode(enc, target);
    }
  }

  /** The TAA passes alone, over the scene last drawn (bench: what they cost). */
  encodeTaa(enc: GPUCommandEncoder, target: GPUTextureView): void {
    this.taa?.encode(enc, target);
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
  /** The rocks' buckets as of the last frame (debug readback; null without rocks). */
  async rockDump(): Promise<{ args: Uint32Array; inst: Float32Array } | null> {
    return this.rocks ? this.rocks.dump() : null;
  }

  async grassCount(): Promise<number> {
    if (!this.argsBuf) return 0;
    const rb = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.argsBuf, 0, rb, 0, 32);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Uint32Array(rb.getMappedRange());
    const n = a[1] + a[5];
    rb.unmap();
    rb.destroy();
    return n;
  }

  destroy(): void {
    this.frameBuf.destroy();
    this.heroBuf.destroy();
    for (const b of this.remoteBufs) b.destroy();
    this.grassBuf?.destroy();
    this.bladeBuf?.destroy();
    this.bladeFarBuf?.destroy();
    this.bladeCacheBuf?.destroy();
    this.argsBuf?.destroy();
    for (const b of [this.ruinBoxBuf, this.ruinSitesBuf, this.puffBuf]) b?.destroy();
    this.realm?.destroy();
    this.structures?.destroy();
    this.fauna?.destroy();
    this.motes?.destroy();
    this.lanterns?.destroy();
    this.combat?.destroy();
    this.meshBuf?.destroy();
    this.fullIdx?.destroy();
    for (const b of this.holeIdx) b.destroy();
    for (const b of [...this.ringIdx, ...this.ringJobBuf]) b.destroy();
    this.waterCells?.destroy();
    this.waterArgs?.destroy();
    this.depth?.destroy();
    this.msaaColor?.destroy();
    this.taa?.destroy();
    this.forest?.destroy();
    this.rocks?.destroy();
    this.modelIsles?.destroy();
    this.sky?.destroy();
    this.cache.destroy();
  }
}

/**
 * The part of framebuffer rectangle `r` ([x, y, w, h], updated in place) the
 * hero's box (packHeroWgsl: min and max corners at HERO_BOX) can cover
 * (scissor.ts cornersScissor).
 */
function boxScissor(
  F: Float32Array, H: Float32Array, sx: number, sy: number, w: number, h: number, r: [number, number, number, number],
): [number, number, number, number] | null {
  return cornersScissor(F, H[HERO_BOX], H[HERO_BOX + 1], H[HERO_BOX + 2], H[HERO_BOX + 4], H[HERO_BOX + 5], H[HERO_BOX + 6], sx, sy, w, h, r);
}
