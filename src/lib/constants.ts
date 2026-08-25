/**
 * W(x,y) Continuous World Engine — Shared Parameter Contract
 * ------------------------------------------------------------
 * Single source of truth for every implementation of the terrain math:
 *   - src/lib/terrainMath.ts            (TypeScript CPU twin, double precision)
 *   - src/components/terrain/demoShader.ts  (WebGL2 / GLSL ES 3.00)
 *   - godot-project/shaders/terrain_raymarch.gdshader (Godot 4 / Metal)
 *   - godot-project/gdextension/src/terrain_cpu.h     (C++ GDExtension twin)
 *   - godot-project/scripts/terrain_math_fallback.gd (pure GDScript twin)
 *
 * SPEC PARITY RULE (§27, §31):
 *   Global state = double, local (chunk-relative) state = float32,
 *   hashing = strict uint32 integer math. GROUND_BIAS = 0.08 m absorbs the
 *   residual fp divergence between the CPU twin and the GPU shader.
 *
 * DYADIC FREQUENCY DISCIPLINE (§25, §27):
 *   Every spatial frequency is the rational n / 2^20 so that
 *   chunk_origin * f (chunk = 8192 = 2^13) is exactly representable in fp32.
 *   Noise lattices therefore align perfectly across chunk borders —
 *   no seams, no jitter at floating-origin rebasing.
 */

export const CHUNK_SIZE = 8192.0; // 2^13 (spec §25: 2^12..2^13)
export const MAX_WORLD_H = 1300.0; // early-exit ceiling (spec §26)
export const MAX_DIST = 6000.0; // raymarch cutoff, hidden by Beer-Lambert fog
export const DEFAULT_SEED = 1337;

/** Spatial frequencies — all dyadic rationals n / 2^20 (see file header). */
export const F = {
  macro: 512 / 1048576, // 1/2048      rolling-hills fBm
  ridge: 341 / 1048576, // ~1/3074.6   ridged multifractal mountains
  mask: 128 / 1048576, // 1/8192     mountain macro mask
  warp: 655 / 1048576, // ~1/1600.9   domain warp
  valley: 1942 / 1048576, // ~1/539.9    cellular valley network
  cont: 64 / 1048576, // 1/16384    continent field
  moist: 512 / 1048576, // 1/2048     moisture noise
  dune: 23 / 1048576, // ~1/45.6 m   dune wavelength phase
  cloud: 403 / 1048576, // ~1/2601.9   cloud coverage
  veg: 14.0, // 1/14 m      vegetation clumps (integer → lattice exact)
} as const;

/** Macro terrain (spec §5). 9 octaves: 2048 m continents down to 8 m gravel
 *  (the top octaves exist to demonstrate Nyquist attenuation at vista range). */
export const MACRO = {
  octaves: 9,
  lacunarity: 2.0,
  gain: 0.5,
  hBase: 90.0,
} as const;

/** Mountains (spec §6 — ridged multifractal, Nyquist weight uses 2f). */
export const MOUNTAIN = {
  octaves: 5,
  lacunarity: 2.17,
  gain: 0.48,
  hMount: 880.0,
  maskLo: 0.28,
  maskHi: 0.60,
} as const;

/** Domain warp (spec §14). */
export const WARP = {
  octaves: 2,
  strength: 620.0,
} as const;

/** Valleys / rivers (spec §7, §8). */
export const VALLEY = {
  cDepth: 48.0,
  wValley: 0.42, // edge-space width of the carve
  wLevel: 5.0, // absolute water plane height (y-up)
  wWidth: 0.18, // river shore softness (edge-space)
} as const;

/** Continent field (feeds climate ocean-moisture term, spec §9). */
export const CONTINENT = {
  lo: 0.22,
  hi: 0.44,
  minFactor: 0.2, // base amplitude kept under open water
} as const;

/** Regional mathematical rules (spec §13 — macro Voronoi, cell 8192). */
export const REGION = {
  cell: 8192.0,
  blendLo: 0.04, // F2−F1 (cell units) where region params fade to global
  blendHi: 0.18,
  hillsFactor: 0.45,
  mountainFactor: 1.3,
  duneFactor: 0.85,
  duneHeight: 42.0,
} as const;

/** Geological anomalies — TotK shrines / Zonai-style ruins (spec §12). */
export const ANOMALY = {
  cell: 2048.0,
  probability: 0.06,
  rIn: 380.0,
  rOut: 620.0,
  terraceStep: 70.0, // metres per concentric terrace ring
  terraceHeight: 15.0,
  spireHeight: 250.0,
  spireRadius: 150.0,
  spiralAmp: 16.0,
  spiralTurns: 6.0,
  spiralRadial: 0.05,
} as const;

/** Climate (spec §9). Game-tuned lapse rate so ~650 m peaks reach the
 *  snow line (TotK-style snowy summits); latitude term spans the far field. */
export const CLIMATE = {
  tBase: 14.0,
  lambdaLat: 1.2e-5, // °C per metre of |Y_global|
  lambdaAlt: 0.02, // °C per metre of elevation (game-tuned lapse rate)
  mBase: 0.5,
  mNoiseAmp: 0.4,
  mOceanLambda: 0.35, // moisture boost scale for (1 − continent)
} as const;

/** §10/§11/§21 — biome RBF anchors in OKLab + (T, M, E, slope) space.
 *  Weight w_k = exp(−Σ ((x_j − c_j,k)/σ_j,k)²); colors blend in OKLab. */
export type BiomeAnchor = {
  name: string;
  oklab: [number, number, number]; // L, a, b
  t: number; m: number; e: number; slope: number; // climate centre
  sT: number; sM: number; sE: number; sS: number; // RBF widths
  veg: number; // §11 vegetation base density
};

export const BIOME_ANCHORS: BiomeAnchor[] = [
  { name: "Alpine Snow",  oklab: [0.93, 0.0, -0.015], t: -4, m: 0.55, e: 800, slope: 1.0, sT: 4.5, sM: 0.45, sE: 600, sS: 1.1, veg: 0.0 },
  { name: "Bare Rock",    oklab: [0.44, 0.004, 0.008], t: 9,  m: 0.45, e: 550, slope: 1.7, sT: 22,  sM: 0.55, sE: 600, sS: 0.8, veg: 0.06 },
  { name: "Grassland",   oklab: [0.52, -0.058, 0.048], t: 13, m: 0.72, e: 90,  slope: 0.32, sT: 6.5, sM: 0.22, sE: 260, sS: 0.5, veg: 0.92 },
  { name: "Forest",      oklab: [0.32, -0.072, 0.032], t: 11, m: 0.9,  e: 230, slope: 0.45, sT: 7,   sM: 0.15, sE: 380, sS: 0.55, veg: 1.0 },
  { name: "Arid Sand",   oklab: [0.79, 0.012, 0.072],  t: 16, m: 0.13, e: 55,  slope: 0.35, sT: 8,   sM: 0.17, sE: 320, sS: 0.5, veg: 0.05 },
];

/** Raymarching (spec §16, §26). */
export const MARCH = {
  lambdaStep: 0.8,
  deltaMin: 0.05,
  bisectSteps: 6,
  stepGrowth: 0.0015, // adaptive: dt *= 1 + t·growth (spec §26)
  startT: 0.4,
} as const;

/** Lighting / sky / atmosphere (spec §18–§22). */
export const LIGHT = {
  sunDir: [0.55, 0.42, 0.72] as const, // normalize() at use site; y-up
  sunColor: [1.0, 0.955, 0.885] as const,
  sunEnergy: 3.4,
  skyZenith: [0.2, 0.36, 0.52] as const,
  skyHorizon: [0.78, 0.8, 0.82] as const,
  cloudPlane: 1500.0,
  cloudOctaves: 4,
  sigmaFog: 9.0e-4, // 1/m extinction at sea level
  fogScaleHeight: 900.0,
} as const;

/** Player physics twin (spec §31). */
export const PLAYER = {
  height: 1.7,
  groundBias: 0.08, // absorbs CPU/GPU float divergence (spec §31)
  gravity: 22.0,
  walkSpeed: 6.0,
  sprintMul: 1.8,
  jumpVelocity: 8.5,
  flySpeed: 40.0,
  slopeMax: 1.5, // gradient-magnitude gate for the slide-off gate
  slideAccel: 14.0,
  // Camera smoothing (spec §24)
  rotK: 14.0, // exponential smoothing rate for yaw/pitch
  springK: 60.0, // critically damped position spring (kD = 2√kS)
} as const;

export type QualityPreset = {
  id: "low" | "med" | "high";
  label: string;
  resScale: number;
  marchSteps: number;
  shadowSteps: number;
  note: string;
};

export const QUALITY_PRESETS: QualityPreset[] = [
  {
    id: "low",
    label: "Low",
    resScale: 0.35,
    marchSteps: 64,
    shadowSteps: 12,
    note: "Integrated GPUs / software rasterizers",
  },
  {
    id: "med",
    label: "Medium",
    resScale: 0.5,
    marchSteps: 96,
    shadowSteps: 20,
    note: "Apple M-series default (TBDR budget)",
  },
  {
    id: "high",
    label: "High",
    resScale: 0.75,
    marchSteps: 128,
    shadowSteps: 28,
    note: "Discrete / M-Pro class",
  },
];

/** Camera (spec §23, §24). */
export const CAMERA = {
  fov: 70.0,
  near: 0.1,
  far: 16000.0,
  sensitivity: 0.0022,
};

/** Demo spawn (tuned against seed 1337 — see worked example, spec §30). */
export const SPAWN = {
  x: 2016.0,
  z: -2592.0,
  yaw: 2.056, // radians — facing the tuned 530 m ridge at ~2.15 km
  pitch: 0.1,
};

/** Worked-example ray (spec §30) — tuned against seed 1337. The camera
 *  stands 30 m above the spawn, aimed 25 m below the summit of a 530 m
 *  warped ridge at ~2.15 km; the march brackets it and bisects to t*. */
export const WORKED_EXAMPLE = {
  camHeight: 30.0,
  aimBelowSummit: 25.0,
  expectedT: 2147.4,
  resY: 540, // half-res render target — the real pixel footprint
};

/**
 * §29 parameter table (rendered on the page). `cpu` / `gpu` name the exact
 * constant in the CPU twin and the shader so parity is auditable.
 */
export type ParamRow = {
  group: string;
  symbol: string;
  name: string;
  value: string;
  cpu: string;
  gpu: string;
  spec: string;
};

export const PARAM_TABLE: ParamRow[] = [
  { group: "World / Precision", symbol: "ChunkSize", name: "Floating-origin chunk size (2^13)", value: "8192", cpu: "CHUNK_SIZE", gpu: "u_chunk / const", spec: "§25" },
  { group: "World / Precision", symbol: "MAX_WORLD_H", name: "Early-exit ceiling height", value: "1300 m", cpu: "MAX_WORLD_H", gpu: "MAX_WORLD_H", spec: "§26" },
  { group: "World / Precision", symbol: "MAX_DIST", name: "Raymarch cutoff distance", value: "6000 m", cpu: "MAX_DIST", gpu: "u_max_dist", spec: "§16" },
  { group: "Macro Terrain", symbol: "S_macro", name: "Macro fBm frequency (dyadic 512/2^20)", value: "1/2048", cpu: "F.macro", gpu: "F_MACRO", spec: "§5" },
  { group: "Macro Terrain", symbol: "Octaves", name: "fBm octave count", value: "5", cpu: "MACRO.octaves", gpu: "OCT_MACRO", spec: "§5" },
  { group: "Macro Terrain", symbol: "Roughness", name: "fBm persistence (gain)", value: "0.50", cpu: "MACRO.gain", gpu: "GAIN_MACRO", spec: "§5" },
  { group: "Macro Terrain", symbol: "H_BASE", name: "Rolling-hills amplitude", value: "90 m", cpu: "MACRO.hBase", gpu: "H_BASE", spec: "§5" },
  { group: "Mountains", symbol: "S_mountain", name: "Ridged frequency (341/2^20)", value: "≈1/3075", cpu: "F.ridge", gpu: "F_RIDGE", spec: "§6" },
  { group: "Mountains", symbol: "Lac / Gain", name: "Ridged lacunarity / gain", value: "2.17 / 0.48", cpu: "MOUNTAIN.*", gpu: "LAC/GAIN_RIDGE", spec: "§6" },
  { group: "Mountains", symbol: "H_MOUNT", name: "Peak amplitude", value: "620 m", cpu: "MOUNTAIN.hMount", gpu: "H_MOUNT", spec: "§6" },
  { group: "Mountains", symbol: "M_mask", name: "Macro mask smoothstep", value: "0.35 → 0.75", cpu: "MOUNTAIN.mask*", gpu: "MASK_LO/HI", spec: "§6" },
  { group: "Nyquist", symbol: "Δp(t)", name: "Pixel footprint", value: "t·2tan(FOV/2)/Res_y", cpu: "footprint()", gpu: "u_footprint·t", spec: "§5" },
  { group: "Nyquist", symbol: "w_i", name: "Per-octave weight", value: "clamp(2 − f·Δp/0.5, 0, 1)", cpu: "nyqW()", gpu: "nyq_w()", spec: "§5" },
  { group: "Nyquist", symbol: "w′", name: "Ridged octave weight (2f)", value: "clamp(2 − 2f·Δp/0.5)", cpu: "nyqW(2f)", gpu: "nyq_w(2f)", spec: "§5n" },
  { group: "Warp", symbol: "S_warp", name: "Domain-warp strength", value: "620 m", cpu: "WARP.strength", gpu: "S_WARP", spec: "§14" },
  { group: "Warp", symbol: "F_warp", name: "Warp fBm frequency", value: "≈1/1601", cpu: "F.warp", gpu: "F_WARP", spec: "§14" },
  { group: "Valleys", symbol: "C_depth", name: "Cellular carve depth", value: "48 m", cpu: "VALLEY.cDepth", gpu: "C_DEPTH", spec: "§7" },
  { group: "Valleys", symbol: "W_valley", name: "Edge-distance falloff width", value: "0.42", cpu: "VALLEY.wValley", gpu: "W_VALLEY", spec: "§7" },
  { group: "Water", symbol: "W_level", name: "Water plane height", value: "5 m", cpu: "VALLEY.wLevel", gpu: "W_LEVEL", spec: "§8" },
  { group: "Water", symbol: "W_width", name: "River shore softness", value: "0.18", cpu: "VALLEY.wWidth", gpu: "W_WIDTH", spec: "§8" },
  { group: "Climate", symbol: "λ_lat", name: "Latitude temperature drop", value: "1.2e-5 /m", cpu: "CLIMATE.lambdaLat", gpu: "L_LAT", spec: "§9" },
  { group: "Climate", symbol: "λ_alt", name: "Altitude lapse rate", value: "0.0065 /m", cpu: "CLIMATE.lambdaAlt", gpu: "L_ALT", spec: "§9" },
  { group: "Climate", symbol: "M_base", name: "Base moisture", value: "0.5", cpu: "CLIMATE.mBase", gpu: "M_BASE", spec: "§9" },
  { group: "Anomaly", symbol: "P_anom", name: "Anomaly cell probability", value: "0.06", cpu: "ANOMALY.probability", gpu: "P_ANOM", spec: "§12" },
  { group: "Anomaly", symbol: "R_anom", name: "Falloff radius (in → out)", value: "380 → 620 m", cpu: "ANOMALY.r*", gpu: "ANOM_R_*", spec: "§12" },
  { group: "Regions", symbol: "S_region", name: "Macro Voronoi cell", value: "8192 m", cpu: "REGION.cell", gpu: "REGION_CELL", spec: "§13" },
  { group: "Marching", symbol: "λ_step", name: "Lipschitz step relax factor", value: "0.8", cpu: "MARCH.lambdaStep", gpu: "LAMBDA_STEP", spec: "§16" },
  { group: "Marching", symbol: "Δ_min", name: "Minimum step", value: "0.05 m", cpu: "MARCH.deltaMin", gpu: "DELTA_MIN", spec: "§16" },
  { group: "Marching", symbol: "N_bisect", name: "Bisection refinements", value: "6", cpu: "MARCH.bisectSteps", gpu: "BISECT_STEPS", spec: "§16" },
  { group: "Shadows", symbol: "Shadow_res", name: "Shadow ray steps (Med preset)", value: "20", cpu: "QUALITY_PRESETS", gpu: "u_shadow_steps", spec: "§18" },
  { group: "Shadows", symbol: "4×Δp", name: "Shadow footprint inflation (penumbra)", value: "4", cpu: "softShadow()", gpu: "soft_shadow()", spec: "§18" },
  { group: "Atmosphere", symbol: "σ_fog", name: "Extinction coefficient", value: "9e-4 /m", cpu: "LIGHT.sigmaFog", gpu: "SIGMA_FOG", spec: "§19" },
  { group: "Atmosphere", symbol: "H_fog", name: "Fog scale height", value: "900 m", cpu: "LIGHT.fogScaleHeight", gpu: "FOG_H", spec: "§19" },
  { group: "Player", symbol: "H_p", name: "Player height", value: "1.7 m", cpu: "PLAYER.height", gpu: "—", spec: "§31" },
  { group: "Player", symbol: "b_skin", name: "GROUND_BIAS (parity absorber)", value: "0.08 m", cpu: "PLAYER.groundBias", gpu: "—", spec: "§31" },
  { group: "Player", symbol: "g", name: "Gravity", value: "22 m/s²", cpu: "PLAYER.gravity", gpu: "—", spec: "§31" },
  { group: "Player", symbol: "Slope gate", name: "Slide-off gradient threshold", value: "1.5", cpu: "PLAYER.slopeMax", gpu: "—", spec: "§31" },
  { group: "Camera", symbol: "FOV", name: "Field of view", value: "70°", cpu: "CAMERA.fov", gpu: "u_tan_half_fov", spec: "§23" },
  { group: "Camera", symbol: "k_rot", name: "Rotation exponential smoothing", value: "14", cpu: "PLAYER.rotK", gpu: "—", spec: "§24" },
  { group: "Camera", symbol: "k_s / k_d", name: "Position spring (critical damping)", value: "60 / 2√60", cpu: "PLAYER.springK", gpu: "—", spec: "§24" },
];
