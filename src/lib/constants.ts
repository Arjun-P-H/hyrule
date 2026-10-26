/**
 * W(x,y) Continuous World Engine — Shared Parameter Contract
 * ------------------------------------------------------------
 * Single source of truth for every implementation of the terrain math:
 *   - src/lib/terrainMath.ts            (TypeScript CPU twin, double precision)
 *   - src/components/terrain/demoShader.ts  (WebGL2 / GLSL ES 3.00)
 *   - src/app/game/page.tsx             (Fullscreen browser game controller)
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

/** 
 * Spatial frequencies — all exact dyadic rationals f = n / 2^20 (spec §25, §27).
 * Because CHUNK_SIZE = 8192 = 2^13, the product CHUNK_SIZE * f = (2^13 * n) / 2^20 = n / 2^7 = n / 128.
 * In IEEE 754 float32 / float64, division of an integer by 128 (2^7) is bit-exact with zero mantissa
 * roundoff error (only changes the binary exponent). This guarantees zero lattice drift or seam artifacts
 * across chunk boundaries.
 */
export const F = {
  macro: 512 / 1048576,   // n = 512 = 2^9   (1/2048 m)    rolling-hills fBm base frequency
  ridge: 341 / 1048576,   // n = 341         (~1/3074.6 m) ridged multifractal mountain base frequency
  mask: 128 / 1048576,    // n = 128 = 2^7   (1/8192 m)    mountain macro mask
  warp: 655 / 1048576,    // n = 655         (~1/1600.9 m) domain warp displacement frequency
  valley: 1942 / 1048576, // n = 1942        (~1/539.9 m)  cellular valley river network
  cont: 64 / 1048576,     // n = 64 = 2^6    (1/16384 m)   continental shelf field
  moist: 512 / 1048576,   // n = 512 = 2^9   (1/2048 m)    moisture climate field
  cloud: 403 / 1048576,   // n = 403         (~1/2601.9 m) cloud coverage layer
  veg: 14.0,              // wavelength = 14 m             vegetation clump wavelength (spec §11)
  // ---- the lands' biome regions (biomes.ts, terrainMath.ts landRegions) ----
  temp: 180 / 1048576,      // n = 180   (5825 m)   sea-level temperature
  tempfine: 376 / 1048576,  // n = 376   (2789 m)   …and its finer octave
  moistr: 330 / 1048576,    // n = 330   (3178 m)   regional moisture
  moistfine: 640 / 1048576, // n = 640   (1638 m)   …and its finer octave
  plateau: 96 / 1048576,    // n = 96    (10923 m)  highland plateaus
  rugged: 240 / 1048576,    // n = 240   (4369 m)   rugged dry country (badlands)
  lakedist: 262 / 1048576,  // n = 262   (4002 m)   lake districts
  karst: 240 / 1048576,     // n = 240   (4369 m)   karst country (jungle)
  marsh: 320 / 1048576,     // n = 320   (3277 m)   marshes in wet lowland
  coast: 256 / 1048576,     // n = 256   (4096 m)   the coastline's finer bays and capes
  cliffs: 420 / 1048576,    // n = 420   (2497 m)   cliffed vs beached coast
  // ---- landforms ----
  tower: 3495 / 1048576,    // n = 3495  (300.0 m)  karst tower lattice
  towerwarp: 12288 / 1048576, // n = 12288 (85.3 m)  …the wander of their outlines
  towercrown: 6144 / 1048576, // n = 6144  (170.7 m) …the swell of their crowns
  towerbump: 40960 / 1048576, // n = 40960 (25.6 m)  …and the lumps on them
  mesa: 874 / 1048576,      // n = 874   (1199.8 m) mesa tiers
  mesafine: 2185 / 1048576, // n = 2185  (479.9 m)  …their finer outline
  dune: 4369 / 1048576,     // n = 4369  (240.0 m)  crescent dune spacing (phase, see dunesE)
  dunecross: 6900 / 1048576,// n = 6900  (152.0 m)  cross ripples (phase)
  dunefield: 48 / 1048576,  // n = 48    (21845 m)  the dune field's wind direction
  duneamp: 2097 / 1048576,  // n = 2097  (500.0 m)  dune heights along the crests (breaks them into crescents)
  dunebend: 2200 / 1048576, // n = 2200  (476.6 m)  the ripples' bends
  hummock: 28339 / 1048576, // n = 28339 (37.0 m)   marsh hummocks and pools
  hummockfine: 80660 / 1048576, // n = 80660 (13.0 m)
  // ---- the geography (hydro.ts): the uplift surface ----
  upl: 85 / 1048576,        // n = 85    (12336 m)  lowland undulation
  upl2: 210 / 1048576,      // n = 210   (4993 m)   …and its finer swell
  rangeu: 40 / 1048576,     // n = 40    (26214 m)  mountain-range crests: along the chain…
  rangev: 112 / 1048576,    // n = 112   (9362 m)   …and across it (zero lines of an elongated noise)
  rangeblend: 20 / 1048576, // n = 20    (52429 m)  which of two chain directions rules
  rangewarp: 120 / 1048576, // n = 120   (8738 m)   the chains' bends
  rangemask: 84 / 1048576,  // n = 84    (12483 m)  where ranges rise at all (mountain countries a few km across)
  peaks: 400 / 1048576,     // n = 400   (2621 m)   peaks and passes along a crest
  massif: 150 / 1048576,    // n = 150   (6991 m)   lone massifs and landmark peaks
  platfine: 420 / 1048576,  // n = 420   (2497 m)   tablelands' ragged outlines
  platcoarse: 80 / 1048576, // n = 80    (13107 m)  where tablelands stand
  platbay: 1748 / 1048576,  // n = 1748  (600 m)    tablelands' (and mesas') bays and spurs
  platnotch: 6990 / 1048576,// n = 6990  (150 m)    …their notches
  platwidth: 583 / 1048576, // n = 583   (1799 m)   where a wall is a sheer cliff and where a ramp
  cragbig: 19661 / 1048576, // n = 19661 (53.3 m)   rock faces' buttresses and gullies
  crag: 40960 / 1048576,    // n = 40960 (25.6 m)   …their knobs and ribs
  cragfine: 98304 / 1048576,// n = 98304 (10.7 m)   …their finer octave
  cragstrata: 17476 / 1048576, // n = 17476 (60.0 m) how the strata tilt and break
  // ---- water ----
  meander: 2497 / 1048576,  // n = 2497  (419.9 m)  river meanders (along a reach)
  floorwob: 1498 / 1048576, // n = 1498  (700.0 m)  how a valley floor's width wanders (along a reach)
  pond: 4096 / 1048576,     // n = 4096  (256.0 m)  pond lattice
} as const;

/**
 * The geography (hydro.ts), TotK-scaled: fields of rolling lowland (median ~50 m over the sea),
 * tablelands with cliff walls ~190 m higher, mountain ranges whose crests run for kilometres with peaks and
 * passes along them up to ~0.9 km, lake basins; then the rivers that drain it all, on a 2,048 m
 * lattice of nodes (each drains to its steepest-descent neighbour, the sea, or a lake).
 */
export const GEO = {
  /** Lowland: base, undulation (F.upl), swell (F.upl2) (m). */
  low: [6, 64, 20],
  /** Ranges: the mask's band (F.rangemask), the crest's sharpness, its height (m), the share kept at passes. */
  rangeMask: [0.52, 0.68],
  rangePow: 2.4,
  /** The two chain directions (rad) and the bend of the chains (m). */
  rangeDir: [0.45, 1.75],
  rangeWarp: 3200,
  /** …and bent again, at three finer scales (F.peaks, F.floorwob, F.tower) by up to these (m): a crest between the 9 km bends ran straight for kilometres. */
  rangeBend: [380, 130, 45],
  rangeH: 520,
  pass: 0.42,
  /** Massifs: lone mountains and landmark peaks off the ranges (F.massif): band, height (m). */
  massif: [0.8, 0.95],
  massifH: 340,
  /**
   * Tablelands (hydro.ts plateauAt): edge (noise level); the wall's narrowest band (noise units) and the width it
   * may add where it is a ramp; the warp that bends the outline (m: bays and spurs, notches: F.platbay, platnotch;
   * the outline of a smooth field ran straight for kilometres) and the most it can move the field (noise units: the
   * warp's reach × the field's steepest slope, so far from the edge the plateau is wholly there or not); height (m).
   */
  plat: 0.72,
  platEdge: 0.018,
  platRamp: 0.12,
  platWarp: [210, 60],
  platBand: 0.07,
  platH: 190,
  /** Mesas (terrainMath.ts mesasE): the same warp of their outline (m), and how much wider than a sheer face a wall may be (× its band). */
  mesaWarp: [200, 55],
  mesaRamp: 3,
  /** …and how far the mesa region's weight (1 in its heart) lowers the field the tiers stand on where it is less (a smooth region edge was a straight wall of the mesas' full height). */
  mesaFade: 0.4,
  /** Sea cliffs: their height range where a cliffed coast meets the sea (m). */
  cliffH: [45, 95],
  /** Lake basins: cell (m), chance, radius (m), depth (m). */
  basinCell: 4096,
  basinP: 0.55,
  basinR: [450, 900],
  basinD: [14, 34],
} as const;

/**
 * Karst towers (terrainMath.ts karstTowers). Every tower's outline wanders (`warp`, m) and is an ellipse that may be
 * longer than wide (`stretch`: the share its long axis' square adds); the flutes down its walls are two octaves of
 * ridged noise round it (`flute`: how deep, shares of its radius; `rho`: how many round it); its wall rises over `wall`
 * of its radius (the rest is the crown), with ledges (`ledge`) and a dome (`dome`: the share of its height the rim is under
 * the summit); the crowns swell by `crown` (share of the height) and are lumpy by `bump` (m); `absent` of the lattice's
 * cells have none; `reach` the longest reach of an outline (× the radius, squared).
 */
export const TOWER = {
  reach: 2.31,
  warp: 26,
  stretch: 0.9,
  flute: [0.16, 0.07],
  rho: [2.3, 5.2],
  wall: 0.7,
  ledge: 0.5,
  dome: 0.4,
  crown: 0.14,
  bump: 7,
  absent: 2 / 16,
} as const;

/**
 * Crags (terrainMath.ts cragDelta, mesasE): rock faces break into buttresses, knobs, ribs and ledges. `rough`: the three octaves'
 * depth (m per unit of slope on a face; field units on a mesa's wall), `period`: the strata's thickness (m), `ledge`: how far a ledge
 * flattens the face (share), `face`: the slope (rise per run) from which ground counts as a rock face, fully so by the second.
 */
export const CRAG = {
  rough: [6.5, 4.0, 1.6],
  mesa: [0.012, 0.007, 0.003],
  period: 24,
  ledge: 0.55,
  face: [0.7, 1.6],
} as const;

/** The rivers (hydro.ts): the drainage lattice, incision, widths, meanders, lakes, falls. */
export const HYDRO = {
  /** Node lattice spacing (m) and the jitter of a node inside its cell (share of the cell). */
  node: 2048,
  jitter: 0.7,
  /** River level under the land at a node: base + per doubling of the drainage + share of the height above the sea (m); none at a source. */
  incise: [2.0, 2.4, 0.045],
  /**
   * Channel full width (m): w0 + w1 · (acc − 1)^pow, at most `max`, acc the drainage at that point
   * of the reach (its head's plus the share of the reach run, so a river widens as it goes), ± `vary`;
   * the channel's depth below the water (m): d0 + d1 · √acc. Every river flows from a lake, so none
   * is a mere stream: 14 m at a reservoir's outlet, ~34 m a node down, to 180 m (TotK's, measured off
   * its map: 4–10 m streams, a median 21 m along all channels, 40–90 m for the named rivers, ~140 m
   * for the Hylia River at its widest).
   */
  width: [14, 20, 0.9],
  widthMax: 180,
  widthVary: 0.3,
  depth: [3.0, 1.0],
  /** Meanders inside the valley floor: amplitude (m) by the floor's flatness, along-reach frequency F.meander. */
  /** Banks of unequal slope: the one bank's reach beyond the channel's half width, the other's short of it, by up to this share. */
  bankSkew: 0.25,
  /** A valley floor's width wanders, widening it by up to this (m) along the reach (nothing at the nodes): walls that run straight for kilometres are not valleys. */
  floorWobble: 70,
  /** A wall's run beyond its floor's rim is stretched (by up to these shares either way: three scales, F.floorwob, F.tower and F.platnotch), so the far walls of a valley, whose contours are as smooth as the channel's meanders are not, bend and bay. */
  wallVary: [0.22, 0.16, 0.10],
  /** …and shifted sideways, past 40 m from the rim, by noise (F.tower, F.platnotch) of up to this (m): bays and spurs along the walls of a canyon whose floor edges ran in parallel lines. */
  wallLat: 110,
  meander: 120,
  meanderMin: 55,
  /** The valley floor beyond the meander belt (m, + per √acc), the floor's rise to its edge (m). */
  floor: [10, 5],
  floorRise: 1.4,
  /** Valley walls steepen with distance from the floor (per m²): a far river's valley never truncates a mountain. */
  wallCurve: 3e-4,
  /** …and rise from the floor's rim at the floor's own slope, steepening to theirs over about this (m): a
   *  shallow valley is a grassy swale, not a trench with a rock riser on each side. */
  wallRound: 12,
  /** Where the land falls away beside a reach, its floor stands as a terrace with a riser of this slope down to the land. */
  riser: 0.45,
  /** A drop of more than this (m) within one sixteenth of a reach falls at once (a waterfall). */
  fallDrop: 12,
  /** …but falls at most this high (m); the rest of such a drop runs as cascades. */
  maxFall: 45,
  /** Round a node a reach's level stays within this of the node's (m): no fall in its first or last sixteenth. */
  nodePool: 1.0,
  /** Routing: a reach may cross land rising this much over its head (m); else the lowest crossing under `gorge` (m) cuts a gorge; else a sink. */
  barrier: 12,
  gorge: 90,
  /** Lakes at sinks: radius (share of a node cell), most depth (m), the ring's samples. */
  lakeR: 0.22,
  lakeMaxDepth: 25,
  lakeRing: 64,
  /**
   * Lakes on rivers (the river widens into a lake round a node on flat land): the chance at such a
   * node, its radius range (m; skewed small), and its depth (m) as d0 + d1 · radius. TotK, measured
   * off its map: lakes 100–600 m across (median ~220 m; Lake Hylia, the largest, ~0.3 km²).
   */
  riverLakeP: 0.5,
  riverLakeR: [50, 300],
  riverLakeD: [6, 0.07],
  /** …only where the river leaving it falls at most this much (m) before it is past the lake's far shore (so a river
   *  flowing in meets the lake's level within as much). */
  riverLakeDrop: 0.5,
  /**
   * Reservoirs: every river that shows flows from one, a lake where no river flowing in shows: the
   * chance at a node where a river may start (its region lets one show from its drainage), the steepest
   * the reach leaving it may fall, the radius tried first (m; uniform, then shrinking by a quarter until
   * the lake fits), and how far (m) its level may lie under the node's.
   */
  reservoirP: 1.0,
  reservoirSlope: 0.06,
  reservoirR: [160, 420],
  reservoirDrop: 4,
  /**
   * Ponds: radius (m; skewed small, R0 + (R1 − R0) · u³ for a uniform u), depth (m), how far under
   * the land at their centre they lie (m), the rim's height over the water (m), and how far (in
   * radii) the land round a pond is eased to its shore's level. TotK, measured off its map: ~2.5
   * ponds per km² of land, median ~31 m across, 90 % under ~76 m.
   */
  pondR: [14, 64],
  pondD: [4.0, 8.0],
  pondSink: 0.9,
  pondRim: 0.35,
  pondReach: 1.5,
  /** The GPU's window of nodes round the camera (nodes a side). */
  window: 64,
} as const;

/** Macro terrain (spec §5). 9 octaves: 2048 m continents down to 8 m gravel
 *  (the top octaves exist to demonstrate Nyquist attenuation at vista range).
 *  Their amplitude is each biome region's own (biomes.ts LANDFORM.hills). */
export const MACRO = {
  octaves: 9,
  lacunarity: 2.0,
  gain: 0.5,
} as const;

/** Mountains (spec §6 — ridged multifractal, Nyquist weight uses 2f). Their
 *  amplitude is each region's (LANDFORM.ridge: 860 m in the ranges, which lie
 *  scaled by the range there: hydro.ts rangeAt). */
export const MOUNTAIN = {
  octaves: 5,
  lacunarity: 2.17,
  gain: 0.48,
} as const;

/** Domain warp (spec §14). */
export const WARP = {
  octaves: 2,
  strength: 620.0,
} as const;

/** Valleys (spec §7): the dales' network (their depth is each region's, LANDFORM.valley), and the sea's level. */
export const VALLEY = {
  wValley: 0.42, // edge-space width of the carve
  wLevel: 5.0, // the sea's water plane (y-up); lakes and rivers lie at their own level (terrainMath.waterLevel)
} as const;

/**
 * Climate (spec §9, redesigned with the lands' biomes). Sea-level temperature
 * T0 is a 5.8 km field (F.temp, plus a finer one) less a latitude term; the air
 * cools with height at a game-tuned lapse rate, so the high peaks of mild
 * country carry snow and a highland plateau an alpine meadow. Moisture is a
 * 3.2 km field (F.moistr), wetter by the sea. Snow lies where T < snowT
 * (biomes.ts LOOK.frost moves that line).
 */
export const CLIMATE = {
  tBase: 14.0,
  tAmp: 19.0,       // °C per unit of the temperature noise (n + 0.35·n_fine)
  tFine: 0.35,
  lambdaLat: 1.2e-5, // °C per metre of |Y_global|
  lambdaAlt: 0.0105, // °C per metre of elevation (game-tuned: snow on the high peaks of mild country, all over the cold)
  snowT: -1.0,      // °C: snow lies below this
  /** Water freezes this far (m) above the snow line (tundra lakes, the sea in the far cold, high tarns). */
  iceAbove: 80,
  mBase: 0.42,
  mNoiseAmp: 0.5,
  mFine: 0.35,
  mOceanLambda: 0.3, // moisture boost toward the sea, (1 − inland)
} as const;

/**
 * The lands (biomes.ts, terrainMath.ts landRegions / landSurface): where each
 * biome region lies (smooth splits of the climate and geology fields), the
 * coast, and the rare features placed by hash (volcanoes, sacred groves). The
 * water (rivers, lakes, ponds) is hydro.ts, tuned by HYDRO. Ranges are
 * [lo, hi] of a smoothstep unless noted.
 */
export const LANDS = {
  /** Sea level (m): the water plane of the oceans (= VALLEY.wLevel). */
  sea: 5.0,
  /** The continent field (F.cont, F.coast): sea below `level`; the land ramps down over `inland`
   *  (beaches), except along cliffed coasts (F.cliffs above `cliffs`); the shelf deepens to `shelf` m. */
  coast: { level: 0.27, inland: 0.07, cliffs: [0.55, 0.7], shelf: 120, beach: 2.5 },
  /** Where the regions' fields are read, the plane is shifted by two scales of noise (F.tower, F.platnotch) by up to these (m): a region's edge was a contour of a field kilometres across, as smooth as it. */
  edge: [170, 56],
  /** Geology (the ranges, tablelands and basins are hydro.ts GEO): rugged country, lake districts, karst, marsh. */
  rugged: [0.7, 0.78],
  lakedist: [0.74, 0.8],
  karst: [0.5, 0.58],
  marsh: [0.42, 0.52],
  /** Climate splits (°C at sea level; moisture). */
  cold: [-2, 2], cool: [3.5, 6], hot: [18, 22],
  dry: [0.36, 0.46], hotDry: [0.47, 0.57], arid: [0.36, 0.46], wet: [0.72, 0.82], badlandsWet: [0.3, 0.4],
  /** Volcanoes: one candidate per cell (kept to the middle of it), its chance, its region's radius and its cone's
   *  height (m), and the least sea-level temperature (°C) at its centre. */
  volcano: { cell: 8192, p: 1.0, radius: [1900, 2290], height: [440, 600], cone: 0.78, crater: 0.86, minT: 4 },
  /** Sacred groves: one candidate per cell (a lattice offset by half a cell), in mild moist land. */
  sacred: { cell: 8192, p: 1.0, radius: [1100, 1500] },
} as const;

/** Raymarching (spec §16, §26). Mirrored in demoShader.ts (LAMBDA … CLAMP_*). */
export const MARCH = {
  lambdaStep: 0.8,
  deltaMin: 0.05,
  bisectSteps: 6,
  bisectTol: 0.001, // m — bisection stops once the bracket is sub-millimetre
  stepGrowth: 0.0015, // adaptive: dt *= 1 + t·growth (spec §26)
  startT: 0.4,
  /**
   * Horizon clamp (anti-tunnel, §26): within gradGate clamp-lengths of the
   * surface no step may exceed clampSlope·t + clampBase. Samples comfortably
   * above it (gap > gradGate·clamp — where ‖∇E‖ is not even refreshed and the
   * clamp alone paces the march) may step up to clampSlopeFar·t + clampBase.
   * 0.12 was chosen against a converged ground-truth render: fewer pixels
   * off-truth than 0.06 on every preset (Low 20.4 → 12.8 %, Medium 5.5 → 5.2 %,
   * High 1.60 → 1.56 % above JND) at 14–16 % less GPU time on Medium/High.
   */
  clampSlope: 0.06,
  clampSlopeFar: 0.12,
  clampBase: 2.0,
  gradGate: 2.5, // refresh ‖∇E‖ only within gradGate·clamp of the surface…
  gradCadence: 3, // …and only on every Nth sample (§16/§34.1)
  gradEpsMin: 0.15, // m — forward-difference step floor
} as const;

/** Soft shadows (spec §18). Mirrored in demoShader.ts (SH_*). */
export const SHADOW = {
  startT: 0.6,
  footprintScale: 4.0, // 4×Δp inflated footprint → free penumbra
  penumbraK: 10.0, // res = min(res, k·h/t)
  cutoff: 0.005, // fully shadowed below this
  stepScale: 0.8,
  stepMin: 0.5,
  stepMax: 60.0,
  stepGrowth: 0.04,
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
  sigmaFog: 2.8e-4, // 1/m extinction at sea level (TotK-clear air: the land ~4 km off still reads, ~12 km off is blue)
  fogScaleHeight: 1100.0,
} as const;

/** Player physics twin (spec §31). */
export const PLAYER = {
  height: 1.7,
  /**
   * GROUND_BIAS = 0.08 m absorbs the GPU (fp32) vs CPU (fp64) divergence of E
   * (spec §27, §31). Measured, not assumed: bench/baseline/parity.json holds
   * GPU readbacks of the production elevation() against this twin
   * (`bun run bench --suites=parity`, Apple M1 / ANGLE-Metal):
   *
   *     |x| from world origin   3 km   12 km   108 km   268 km   534 km   1,067 km
   *     max |E_gpu − E_cpu|    0.6 mm  3.2 mm  15.0 mm  28.9 mm  99.4 mm  113 mm
   *
   * The shader evaluates noise on the fp32 *global* coordinate, so the error
   * grows with distance from the origin (rebasing itself is seamless: ≤ 0.14 mm).
   * 0.08 m therefore holds with ≥ 2.8× margin within ±300 km (5.3× within
   * ±108 km) and is exceeded beyond ~500 km. tests/floatingOrigin.test.ts
   * enforces the margin against the recorded measurement.
   */
  groundBias: 0.08,
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

/**
 * Third-person hero locomotion (src/lib/hero/locomotion.ts), tuned after the
 * open-world Zelda games: run by default, sprint / climb / glide / swim-dash
 * spend one stamina wheel (1.0), running dry leaves the hero exhausted until
 * the wheel refills. Speeds in m/s, rates per second, slopes as ‖∇E‖.
 */
export const HERO = {
  runSpeed: 5.4,
  sprintSpeed: 8.6,
  sneakSpeed: 1.7,
  exhaustedSpeed: 2.4,
  wadeFactor: 0.7,        // speed factor with the feet in shallow water
  accel: 10.0,            // ground velocity approach rate
  airAccel: 2.5,          // steering rate in the air
  turnRate: 12.0,         // facing approach rate
  jumpVelocity: 7.2,      // ≈ 1.2 m apex at PLAYER.gravity
  terminalVelocity: 50.0, // skydiving fall speed (drops from fly mode)
  walkSlope: 1.0,         // steeper ground (> 45°) cannot be run up: climb or slide
  climbExitSlope: 0.85,   // hysteresis: a climb ends once the wall eases below this
  fallSlope: 2.2,         // ground falling away faster than this per metre → airborne
  slideAccel: 16.0,
  climbJump: 1.4,         // metres surged uphill per climb jump
  climbJumpSpeed: 5.0,
  glideMinHeight: 2.5,    // clearance needed to open the paraglider
  glideSpeed: 7.0,        // with steering input
  glideDrift: 3.0,        // hands off
  glideSink: 2.2,
  glideAccel: 2.0,
  swimDepth: 1.3,         // water deeper than this swims; shallower wades
  currentDrift: 0.8,      // a swimmer drifts with this share of a river's current
  swimFloat: 1.12,        // root depth below the surface while swimming (head above water)
  swimSpeed: 2.2,
  swimDashSpeed: 4.2,
  staminaSprint: 0.2,
  staminaClimb: 0.1,
  staminaHang: 0.035,
  staminaClimbJump: 0.18,
  staminaGlide: 0.06,
  staminaSwimDash: 0.2,
  staminaRegen: 0.34,
  staminaRegenExhausted: 0.25,
  staminaRegenSwim: 0.1,
  staminaRegenDelay: 0.6,
  apexBand: 1.8,          // |vy| under this at a jump's top…
  apexGravity: 0.55,      // … falls at this share of gravity (hang time)
} as const;

/**
 * Skydiving (locomotion.ts updateHero 'air', rig.ts poseSkydive, camera.ts): a fall from high enough
 * spreads into a belly-to-earth glide; the sprint key tucks into a head-first nosedive, letting go
 * (or pulling back) at speed flares. Quadratic drag k = g / Vt² holds each posture's terminal speed;
 * landing hard rolls on, landing harder squats.
 */
export const SKYDIVE = {
  minAltitude: 8.0,       // m of air under the feet…
  enterSpeed: 8.5,        // … and falling faster than this: spread into a skydive
  glideTerminal: 16.5,    // m/s spread-eagle
  diveTerminal: 42.0,     // m/s nosedive
  flareTerminal: 8.5,     // m/s while flaring…
  flareDecel: 52.0,       // … braking no harder than this (m/s², ≈ 2.4 g of the land's 22)
  flareFrom: 20.0,        // m/s: letting the dive go faster than this flares…
  flareUntil: 18.0,       // … until slower than this…
  flareTime: 0.38,        // … or this long (s)
  diveRate: 8.0,          // the posture eases toward the key at this rate (1/s)
  glideMaxXZ: 14.5, diveMaxXZ: 2.5,   // m/s across the ground, spread and tucked
  glideAccelXZ: 22.0, diveAccelXZ: 5.0, // m/s² steering toward that
  glideTurn: 4.2, diveTurn: 1.5,      // the facing's approach rate to the stick's way, spread and tucked (1/s)
  liftK: 0.28,            // forward glide from the spread body's pitch: liftK · |vy| · sin 2θ
  liftMinFall: 6.0,       // m/s of fall before the spread body glides forward
  landTTI: 0.42,          // s to impact: the body comes upright to land
  // (the spec's 8 and 22 m/s at 9.81 m/s², as the same drop heights — 3.3 m, 25 m — under the land's 22)
  rollMin: 12.0,          // m/s impact or more, moving: roll on…
  squatMin: 33.0,         // … this or more: a heavy squat
  rollMove: 2.5,          // m/s across the ground (or the stick held) to roll
  rollTime: 0.55, rollSpeed: 6.8,
  squatTime: 0.68, squatBrake: 28.0,
} as const;

/**
 * Climbing (src/lib/hero/locomotion.ts climbSurface, rig.ts climbGait / poseClimb), one way on
 * everything climbable: tree trunks (round their profiled cylinder, forest/collision.ts trunkSection),
 * and the walls of the land, the sky islands' hills and the rocks (round the contour of their solid at
 * the hips' height), after Tears of the Kingdom and Assassin's Creed III. Speeds in m/s (up and down
 * along the surface); lengths in m; the gait's step in seconds.
 */
export const CLIMB = {
  upSpeed: 0.95,          // up the surface
  downSpeed: 1.1,         // down it
  wrapSpeed: 0.8,         // across it (round a trunk), at the hips (ω = v / (R + clearance))
  accel: 6.0,             // the climb's speed eases toward the stick's at this rate (≈ 0.4 s to start or stop)
  wrapMax: 2.44,          // rad/s round a thin pole (140°/s)
  grab: 0.5,              // pushing within 60° of the trunk's centre grabs it
  minTop: 2.0,            // shorter trunks (stumps, saplings) are stepped round, not climbed
  latchSpeed: 14.0,       // faster than this a body glances off a trunk instead of grabbing it
  latchTime: 0.18,        // a mid-air grab slides down the bark for this long…
  latchDecel: 18.0,       // … braking at this rate: v² / 2a, held to latchSlip
  latchSlip: [0.05, 0.5],
  slideSpeed: 3.8,        // the sneak key (or a dry wheel) slides down the bark
  leapAway: 0.42,         // a jump with the stick > 65° off the trunk leaps away from it…
  leapSpeed: 5.5,         // … this fast, the way the stick points…
  leapUp: 2.4,            // … and up
  mantleTime: 0.65,       // the vault onto a trunk's top
  perch: 0.45,            // a trunk's top bears a body this close to its axis
  // the pelvis' clearance from the bark: hugging a thin pole, held off a wide trunk
  clearPole: 0.22, clearWide: 0.38, poleR: 0.15, wideR: 0.8,
  poleWrap: 0.35,         // thinner trunks are hugged: hands round past the sides, feet clamping
  // the gait (rig.ts): diagonal pairs, a limb steps once this far from its socket (MaxLimbStretch) to `lead`
  // past it, over stepTime; the swing's arc off the bark and the slip down it at contact
  handStretch: 0.52, footStretch: 0.56, lead: 0.26, stepTime: 0.28, stepArc: 0.14, slip: 0.018,
  sway: 0.075, bob: 0.035,
} as const;

/**
 * Hero body (src/lib/hero/rig.ts) and the extents the renderer needs. The
 * *Reach / limbPad values bound every primitive of heroMap() in
 * demoShader.ts around its anchor joint (mirrored there as HERO_*; the
 * bounding sphere the CPU uploads is built from them, so they must cover the
 * shader's geometry — tests/hero.test.ts pins the mirror).
 */
export const HERO_SHAPE = {
  pelvisH: 0.93,
  hipW: 0.095,
  hipDrop: 0.03,
  thigh: 0.42,
  shin: 0.41,
  ankleH: 0.075,
  footLen: 0.15,
  spine: 0.37,
  neck: 0.2,
  shoulderW: 0.172,
  shoulderDrop: 0.035,
  upperArm: 0.27,
  forearm: 0.25,
  limbPad: 0.09,
  headReach: 0.46,
  backReach: 0.52,
  gliderReach: 1.3,
  // the gear (rig.ts heroGear): the sword from its grip to the tip, a club's head, the shield
  // round its centre, the bow's limbs and the arrow on its string
  swordLen: 0.95,
  clubLen: 0.74,
  gearPad: 0.09,
  shieldReach: 0.37,
  bowHalf: 0.6,
  arrowLen: 0.8,
} as const;

/** Third-person camera (src/lib/hero/camera.ts). */
export const HERO_CAMERA = {
  distance: 4.6,
  minDistance: 1.6,       // obstructions never pull the eye closer than this
  maxDistance: 11.0,
  targetH: 1.45,          // look-at height above the root
  clearance: 0.45,        // camera stays this far above terrain and water
  pitchMin: -1.2,
  pitchMax: 0.85,
  pitch: -0.24,           // default orbit pitch (looking slightly down)
  followXZ: 16.0,
  followY: 7.0,
  maxLag: 0.6,            // metres the look-at point may trail the hero
  boomOut: 3.0,           // boom regrowth rate after an obstruction clears
  aimDistance: 2.5,       // drawing the bow or winding up a throw: over the right shoulder
  aimShoulder: 0.62,      //   (m to the right of the hero)
  aimTargetH: 1.68,
  aimRate: 10.0,          // blend rate into and out of the over-the-shoulder view
  lockRate: 5.0,          // the orbit turns toward a locked-on foe at this rate
} as const;

/**
 * Combat (src/lib/combat/), after Tears of the Kingdom: hearts in quarters; a four-hit sword
 * combo, a charged spin, jump and dash attacks, sneakstrikes; the shield guards and, raised
 * just as a blow lands, parries (a Perfect Guard); side hops and backflips that slip a blow
 * at the last moment open a Flurry Rush; the bow, with bullet time in the air; throwing the
 * weapon in hand; shield surfing. Times in s, speeds in m/s, damage in hit points.
 */
export const COMBAT = {
  hearts: 5,              // heart containers at the start (4 quarters each)
  hurtTime: 0.42,         // stagger after taking a blow
  invulnerable: 0.85,     // no further damage this long after one
  regenDelay: 9,          // s out of a fight before hearts come back
  regenRate: 0.6,         // quarters per second then
  comboWindow: 0.3,       // s after a swing ends in which the next press still chains
  chargeAfter: 0.3,       // s of attack held (nothing chained) before a spin starts to charge
  chargeFull: 0.85,       // s to a full charge
  staminaSpin: 0.22,      // a spin's cost
  staminaCharge: 0.1,     // per second of charging
  parryWindow: 0.24,      // s after raising the shield in which a blow is parried
  parryLockout: 0.3,      // s after a window closes unused before the next can open
  guardSpeed: 0.5,        // speed factor with the shield up
  aimSpeed: 0.42,         // speed factor while drawing the bow or winding up a throw
  drawTime: 0.38,         // s to a full draw
  nockTime: 0.28,         // s to nock the next arrow
  arrowSpeed: 55,
  arrowGravity: 9.0,
  arrowDamage: 15,
  arrowLife: 12,          // s an arrow stays stuck where it hit
  headshot: 2,            // × an arrow's damage on the head (and a stun)
  bulletTime: 0.25,       // the world's pace while drawing the bow in the air
  staminaBulletTime: 0.18,
  throwSpeed: 21,
  throwGravity: 13,
  throwDamage: 2,         // × the weapon's damage
  dodgeWindow: 0.26,      // s into a dodge in which a blow slipped is perfect
  flurryWait: 1.2,        // s a perfect dodge leaves to start a Flurry Rush (the world slowed)
  flurrySlow: 0.12,       // the world's pace through the rush
  flurryHits: 6,
  dodgeSlow: 0.3,         // the world's pace while a rush can start
  sideHop: [6.6, 4.4],    // m/s sideways, up
  backflip: [6.2, 6.0],   // m/s back, up
  sneakstrike: 8,         // × damage on a foe unaware of the hero, struck while sneaking
  hitStop: 0.05,          // s the world all but stops on a hit (twice that on a heavy one)
  surfFriction: 0.03,     // shield surfing: kinetic friction coefficient (it glides a while on the flat)
  surfDrag: 0.0035,       // quadratic drag (1/m)
  surfSteer: 2.4,         // rad/s turn at full input
  surfPush: 4.0,          // m/s: stepping onto the shield from a stand pushes off at this
  surfMinSpeed: 0.6,      // slower than this on gentle ground: step off the shield
  surfMaxSpeed: 26,
  surfJump: 5.8,
  arrows: 30,             // in the quiver at the start
  maxArrows: 99,
  lockRange: 28,          // m: the shield locks onto the foe nearest the view within this
  sheatheAfter: 7,        // s of peace before the weapons go back onto the back
  weaponSlots: 8,         // melee weapons carried (the sword among them)
  shieldSlots: 6,         // shields carried (the Hylian Shield among them)
  bowSlots: 5,            // bows carried (the Traveler's Bow among them)
  maxFood: 999,           // of one kind in the pouch
  startApples: 3,         // in the pouch at the start
  surfWear: 4,            // s of shield surfing that cost the shield one blow's worth of wear
  swordReturn: 9,         // s a thrown sword lies before it finds its way back to its sheath
  // other players (multiplayer, combat/pvp.ts): their blows and shots cost quarter hearts by damage
  pvpQuarters: 1 / 3,     // quarters per point of damage (the Master Sword's slash: a heart)
  pvpMax: 12,             // quarters at most from one blow or shot (three hearts)
  parriedStun: 0.9,       // s dazed when another player's Perfect Guard turns the hero's blow
} as const;

/**
 * Foes (src/lib/combat/foes.ts): Bokoblin camps placed by hash, a few to a camp, waking when the
 * hero comes near. Senses, pacing and the camps' life cycle; each kind's own numbers sit with it.
 */
export const FOE = {
  campCell: 224,          // m: one candidate camp per cell
  campProbability: 0.32,
  campRadius: 6,          // m: where its foes stand round the fire
  wakeRange: 120,         // m: a camp's foes come to life within this of the hero
  sleepRange: 175,        // … and are put away beyond this
  respawn: 300,           // s before a cleared camp is manned again
  max: 12,                // foes awake at once
  sight: 32,              // m they see the hero in their view cone
  sightSneak: 13,         // … when it sneaks
  hearing: 6,             // m they notice it at any angle (twice that while it sprints)
  cone: 1.2,              // half-angle of the view cone (rad)
  notice: 0.75,           // s in sight before a foe is roused (2.5 × as long when the hero sneaks)
  forget: 8,              // s without the hero before a roused foe calms down
  attackers: 1,           // foes swinging at the hero at once (the others circle, waiting their turn)
  rest: [1.3, 2.8],       // s a foe waits after its swing before it may swing again
  keepAway: [2.3, 3.6],   // m a foe waiting its turn keeps round the hero
  strikeRange: 2.1,       // m at which a foe starts its swing
  chaseSpeed: 0.82,       // × HERO.runSpeed
  archerRange: [9, 26],   // m an archer keeps from its target
  arrowSpeed: 30,
  arrowDamage: 2,         // quarters of a heart
  stun: 1.1,              // s a parried or headshot foe is dazed
  knockdown: 1.5,         // s a foe lies after a heavy blow
  corpse: 0.9,            // s before a beaten foe vanishes in smoke
} as const;

export type QualityPreset = {
  id: "low" | "med" | "high";
  label: string;
  resScale: number;
  marchSteps: number;
  shadowSteps: number;
  note: string;
};

export type PresetId = QualityPreset['id'];

/**
 * Step budgets re-profiled after the Phase 1–2 march changes (bench, Apple M1,
 * distance to a converged ground-truth render, GPU time relative to the
 * previous budget): fewer march steps drop every preset below its Phase 0
 * image quality, while shadow steps turned out to be the dominant quality
 * lever — Low 12→20 shadow steps: 12.8 → 8.0 % of pixels above JND for +5.7 %
 * time; Medium 20→28: 5.2 → 2.0 % for +3.3 %. High (128/28) is unchanged.
 */
export const QUALITY_PRESETS: QualityPreset[] = [
  {
    id: "low",
    label: "Low",
    resScale: 0.35,
    marchSteps: 64,
    shadowSteps: 20,
    note: "Integrated GPUs / software rasterizers",
  },
  {
    id: "med",
    label: "Medium",
    resScale: 0.5,
    marchSteps: 96,
    shadowSteps: 28,
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

/**
 * The sky realm (src/lib/skyworld/): archipelagos — a great hub island,
 * satellites round it joined by natural rock arches, small isles and
 * floating rocks — and lone isles between them, all above the highest peak
 * and the cloud deck. Each island has a sky biome of its own and, where it
 * is wet enough, a lake whose stream pours off the rim as a waterfall that
 * turns to mist on the way down. Placed by hash (pure functions of cell and
 * seed); ranges are [min, max]. The WGSL twin (gpu/skyIslands.wgsl.ts)
 * evaluates the same surfaces from what the CPU uploads.
 */
export const SKY_REALM = {
  archCell: 8192,           // m — one candidate archipelago per cell (= CHUNK_SIZE)
  archProbability: 0.45,    // Eyries are rare: each a destination (SKY_CIVILIZATION.md §7)
  archReach: 3000,          // m — nothing of an archipelago lies farther from its centre
  hubRadius: [650, 1250],   // m — outline radius before the wobble
  hubTop: [2050, 2500],     // m — base height of the hub's top
  satellites: [3, 6],
  satRadius: [150, 420],
  satGap: [170, 460],       // m — open sky between a satellite's outline and the hub's
  isles: [2, 6],            // small isles farther out
  isleRadius: [45, 120],
  debris: [5, 12],          // floating rocks round the islands
  debrisRadius: [5, 18],
  isleCell: 2048,           // m — lone isles between archipelagos, one candidate per cell
  isleProbability: 0.12,
  loneRadius: [80, 280],
  clearance: 150,           // m — every underside's tip at least this far over the highest ground under its island
  floor: 1400,              // m — the original heights (skyworld/heights.ts): no underside lower (the terrain tops out at MAX_WORLD_H)
  topMax: 2900,             // m
  lower: 1150,              // m — the low heights (skyworld/heights.ts): every island pulled this far down from its class's (and lone isles') height, as low as the ground under it lets it
  rimDrop: 2.5,             // m — the top rounds down this far at the rim
  outline: [0.07, 0.15],    // radial wobble of an outline
  linkSlope: 0.3,           // steepest arch deck (rise per metre)
} as const;

/**
 * The land's model isles (src/lib/skymodels.ts): sky islands baked from models
 * (scripts/bake-sky-model.py, public/models/), floating over the land in the
 * WebGPU renderer, one candidate per cell. Ranges are [min, max].
 */
export const SKY_MODELS = {
  names: ['sky-castle', 'sky-rotunda'], // public/models/<name>.bin (and .webp if textured); a cell picks one by hash
  cell: 6144,               // m — one candidate model isle per cell
  probability: 0.5,         // so one every ~9 km
  margin: 1150,             // m — an isle's middle keeps this far inside its cell (the 1.6 km castles at any turn)
  plaza: [2600, 3000],      // m — its plaza's height (its hanging rock, ~1.25 km with its pendant, clears MAX_WORLD_H)
  range: 40000,             // m — drawn out to here (the haze has hidden it by then)
} as const;

/**
 * The cloud world's ground (src/lib/skyworld/clouds.ts): a sea of cumulus,
 * heaped domes on a broad swell, all under MAX_WORLD_H (so under every
 * island's underside, SKY_REALM.floor). Puffs climb the islands and drawn lobes
 * heap its crests and wreathe the islands (skyworld/puffs.ts). Ranges are [min, max].
 */
export const CLOUDS = {
  base: [880, 980],         // m — the broad swell under the domes
  domes: [260, 90, 30, 8],  // m — the domes' heights at 1100, 380, 120 and 36 m across
  ceiling: 1280,            // m — no cloud higher (under MAX_WORLD_H)
  puffRise: 1.0,            // m — the most a plain puff's top rises over the last
  puffGap: 2.5,             // m — the widest gap between two puffs' edges
  springLift: [10, 20],     // m — how high a springy puff throws the hero
  albedo: [0.46, 0.48, 0.52],      // a little over snow's (biomes.ts SNOW_ALBEDO): any brighter clips to flat white
} as const;

/**
 * The portal between the worlds (src/lib/portal.ts): a ring of shrine-stone
 * blocks (the bottom three left out: the way through), the fade either side
 * of a trip, and how long after one before the ring takes the hero again.
 */
export const PORTAL = {
  radius: 3.5,              // m — the blocks' circle
  segments: 12,             // blocks round the full circle
  centre: 2.3,              // m — the circle's centre over the ground
  fade: 0.45,               // s — out, and again in
  cooldown: 2.0,            // s
} as const;

/**
 * Fly mode's ceiling (m): the WebGL2 ray-march keeps its camera under its
 * early-exit height (MAX_WORLD_H, and draws no sky realm); WebGPU flies up
 * past the sky realm's highest tops and the land's sky castles' spires (SKY_MODELS:
 * a plaza up to 3 km, a spire 1.15 km over it), so every island can be reached.
 */
export const FLY_CEILING = { webgl2: 1150, webgpu: 4600 } as const;

/**
 * Ruins and shrines (Zonai-style), placed by hash: src/lib/ruins.ts builds
 * each site's stone boxes, shared by the renderer and the hero's collision.
 */
export const RUINS = {
  cell: 1024,         // m — one candidate site per cell
  probability: 0.3,
  maxRelief: 4,       // m — ground this uneven under the platform: no site
  shrineShare: 0.3,   // share of sites with a glowing shrine monolith
  drawRange: 1500,    // m — sites drawn around the camera
} as const;

/** Camera (spec §23, §24). */
export const CAMERA = {
  fov: 70.0,
  near: 0.1,
  far: 16000.0,
  sensitivity: 0.0022,
};

/** Demo spawn (tuned against seed 1337 — see worked example, spec §30). */
export const SPAWN = {
  x: 2648.0,
  z: -4031.0,
  // in the open fields (~214 m), copses of oak round it, facing east-southeast up a gentle rise toward
  // the Crown Eyrie floating 2.6 km ahead
  yaw: 1.88,
  pitch: 0.06,
};

/** Worked-example ray (spec §30) — tuned against seed 1337. The camera
 *  stands 30 m above the spawn, aimed (yaw, pitch) at a hillside ~2.0 km off;
 *  the march brackets it and bisects to t*. */
export const WORKED_EXAMPLE = {
  camHeight: 30.0,
  aimBelowSummit: 25.0,
  yaw: 3.32,
  pitch: 0.01,
  expectedT: 1912.9,
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
  { group: "Macro Terrain", symbol: "Octaves", name: "fBm octave count", value: "9", cpu: "MACRO.octaves", gpu: "OCT_MACRO", spec: "§5" },
  { group: "Macro Terrain", symbol: "Roughness", name: "fBm persistence (gain)", value: "0.50", cpu: "MACRO.gain", gpu: "GAIN_MACRO", spec: "§5" },
  { group: "Macro Terrain", symbol: "H_hills", name: "Rolling-hills amplitude (per region, ± the uplift)", value: "8–78 m", cpu: "LANDFORM.hills", gpu: "LF_TABLE", spec: "§5" },
  { group: "Geography", symbol: "U_low", name: "Lowland uplift (base + undulation + swell)", value: "6 + 64 + 20 m", cpu: "GEO.low", gpu: "upliftAt()", spec: "§13" },
  { group: "Geography", symbol: "H_range", name: "Range crests (two chain directions, peaks and passes)", value: "520 m", cpu: "GEO.rangeH", gpu: "rangeAt()", spec: "§6" },
  { group: "Geography", symbol: "H_plat", name: "Tableland cliffs over talus", value: "190 m", cpu: "GEO.platH", gpu: "upliftAt()", spec: "§13" },
  { group: "Geography", symbol: "S_node", name: "Drainage lattice (one node per cell)", value: "2048 m", cpu: "HYDRO.node", gpu: "hydroBuf", spec: "§8" },
  { group: "Geography", symbol: "W_river", name: "Channel width, w0 + w1·(acc − 1)^p, capped", value: "14 + 20·(acc − 1)^0.9 ≤ 180 m", cpu: "HYDRO.width/widthMax", gpu: "riverHalfW()", spec: "§8" },
  { group: "Geography", symbol: "D_fall", name: "Waterfall: a drop per sixteenth over / at most", value: "12 / 45 m", cpu: "HYDRO.fallDrop/maxFall", gpu: "reachLF()", spec: "§8" },
  { group: "Geography", symbol: "E_gully", name: "Gully erosion (Johansen filter, MPL-2.0), down the land's slope", value: "4 octaves, 500 → 63 m", cpu: "erosion.ts, ERO_STEP", gpu: "erosionDelta()", spec: "§6" },
  { group: "Mountains", symbol: "S_mountain", name: "Ridged frequency (341/2^20)", value: "≈1/3075", cpu: "F.ridge", gpu: "F_RIDGE", spec: "§6" },
  { group: "Mountains", symbol: "Lac / Gain", name: "Ridged lacunarity / gain", value: "2.17 / 0.48", cpu: "MOUNTAIN.*", gpu: "LAC/GAIN_RIDGE", spec: "§6" },
  { group: "Mountains", symbol: "H_ridge", name: "Ridge detail (per region, × the range there)", value: "0–200 m", cpu: "LANDFORM.ridge", gpu: "LF_TABLE", spec: "§6" },
  { group: "Nyquist", symbol: "Δp(t)", name: "Pixel footprint", value: "t·2tan(FOV/2)/Res_y", cpu: "footprint()", gpu: "u_footprint·t", spec: "§5" },
  { group: "Nyquist", symbol: "w_i", name: "Per-octave weight", value: "clamp(2 − f·Δp/0.5, 0, 1)", cpu: "nyqW()", gpu: "nyq_w()", spec: "§5" },
  { group: "Nyquist", symbol: "w′", name: "Ridged octave weight (2f)", value: "clamp(2 − 2f·Δp/0.5)", cpu: "nyqW(2f)", gpu: "nyq_w(2f)", spec: "§5n" },
  { group: "Warp", symbol: "S_warp", name: "Domain-warp strength", value: "620 m", cpu: "WARP.strength", gpu: "S_WARP", spec: "§14" },
  { group: "Warp", symbol: "F_warp", name: "Warp fBm frequency", value: "≈1/1601", cpu: "F.warp", gpu: "F_WARP", spec: "§14" },
  { group: "Valleys", symbol: "k_wall", name: "Valley walls' slope (per region), reached over a 12 m rounding, + curve", value: "0.04–0.9 + 3e-4·x²", cpu: "LANDFORM.k, HYDRO.wallRound, HYDRO.wallCurve", gpu: "wallRise()", spec: "§7" },
  { group: "Water", symbol: "W_level", name: "Sea level (rivers, lakes, ponds: their own levels)", value: "5 m", cpu: "VALLEY.wLevel", gpu: "W_LEVEL", spec: "§8" },
  { group: "Climate", symbol: "T_0", name: "Sea-level temperature field (5.8 km)", value: "14 ± 19 °C", cpu: "CLIMATE.tBase/tAmp", gpu: "T_BASE/T_AMP", spec: "§9" },
  { group: "Climate", symbol: "λ_lat", name: "Latitude temperature drop", value: "1.2e-5 /m", cpu: "CLIMATE.lambdaLat", gpu: "L_LAT", spec: "§9" },
  { group: "Climate", symbol: "λ_alt", name: "Altitude lapse rate (game-tuned)", value: "0.0105 /m", cpu: "CLIMATE.lambdaAlt", gpu: "L_ALT", spec: "§9" },
  { group: "Climate", symbol: "M", name: "Moisture field (3.2 km), wetter by the sea", value: "0.42 ± 0.5", cpu: "CLIMATE.mBase/mNoiseAmp", gpu: "M_BASE/M_AMP", spec: "§9" },
  { group: "Climate", symbol: "T_snow", name: "Snow lies below (°C)", value: "−1 °C", cpu: "CLIMATE.snowT", gpu: "SNOW_T", spec: "§9" },
  { group: "Lands", symbol: "N_biome", name: "Biome regions (biomes.ts)", value: "15", cpu: "BIOMES", gpu: "landRegions()", spec: "§13" },
  { group: "Lands", symbol: "C_sea", name: "Continent field below which the sea lies", value: "0.27", cpu: "LANDS.coast.level", gpu: "COAST_LEVEL", spec: "§13" },
  { group: "Lands", symbol: "R_volcano", name: "Volcano cell, chance, region radius", value: "8192 m · 1.0 · 1.9–2.29 km", cpu: "LANDS.volcano", gpu: "landRegions()", spec: "§13" },
  { group: "Lands", symbol: "R_sacred", name: "Sacred grove cell, chance, radius", value: "8192 m · 1.0 · 1.1–1.5 km", cpu: "LANDS.sacred", gpu: "landRegions()", spec: "§13" },
  { group: "Lands", symbol: "S_basin", name: "Lake basin cells (uplift bowls; lakes at drainage sinks)", value: "4096 m", cpu: "GEO.basinCell", gpu: "basinAt()", spec: "§8" },
  { group: "Lands", symbol: "S_pond", name: "Pond cells (flat meadows, away from rivers)", value: "256 m", cpu: "F.pond", gpu: "pondAt()", spec: "§8" },
  { group: "Marching", symbol: "λ_step", name: "Lipschitz step relax factor", value: "0.8", cpu: "MARCH.lambdaStep", gpu: "LAMBDA_STEP", spec: "§16" },
  { group: "Marching", symbol: "Δ_min", name: "Minimum step", value: "0.05 m", cpu: "MARCH.deltaMin", gpu: "DELTA_MIN", spec: "§16" },
  { group: "Marching", symbol: "N_bisect", name: "Bisection refinements", value: "6", cpu: "MARCH.bisectSteps", gpu: "BISECT_STEPS", spec: "§16" },
  { group: "Shadows", symbol: "Shadow_res", name: "Shadow ray steps (Med preset)", value: "28", cpu: "QUALITY_PRESETS", gpu: "u_shadow_steps", spec: "§18" },
  { group: "Shadows", symbol: "4×Δp", name: "Shadow footprint inflation (penumbra)", value: "4", cpu: "softShadow()", gpu: "soft_shadow()", spec: "§18" },
  { group: "Atmosphere", symbol: "σ_fog", name: "Extinction coefficient", value: "2.8e-4 /m", cpu: "LIGHT.sigmaFog", gpu: "SIGMA_FOG", spec: "§19" },
  { group: "Atmosphere", symbol: "H_fog", name: "Fog scale height", value: "1100 m", cpu: "LIGHT.fogScaleHeight", gpu: "FOG_H", spec: "§19" },
  { group: "Player", symbol: "H_p", name: "Player height", value: "1.7 m", cpu: "PLAYER.height", gpu: "—", spec: "§31" },
  { group: "Player", symbol: "b_skin", name: "GROUND_BIAS (parity absorber)", value: "0.08 m", cpu: "PLAYER.groundBias", gpu: "—", spec: "§31" },
  { group: "Player", symbol: "g", name: "Gravity", value: "22 m/s²", cpu: "PLAYER.gravity", gpu: "—", spec: "§31" },
  { group: "Player", symbol: "Slope gate", name: "Slide-off gradient threshold", value: "1.5", cpu: "PLAYER.slopeMax", gpu: "—", spec: "§31" },
  { group: "Camera", symbol: "FOV", name: "Field of view", value: "70°", cpu: "CAMERA.fov", gpu: "u_tan_half_fov", spec: "§23" },
  { group: "Camera", symbol: "k_rot", name: "Rotation exponential smoothing", value: "14", cpu: "PLAYER.rotK", gpu: "—", spec: "§24" },
  { group: "Camera", symbol: "k_s / k_d", name: "Position spring (critical damping)", value: "60 / 2√60", cpu: "PLAYER.springK", gpu: "—", spec: "§24" },
];
