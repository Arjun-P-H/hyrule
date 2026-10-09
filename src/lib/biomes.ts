/**
 * The lands' biomes: one table that every part of the world reads.
 * ---------------------------------------------------------------------------
 * The ground is divided into fifteen biome regions, a few kilometres across,
 * blended over a few hundred metres at their borders. Which region lies where
 * follows the climate at sea level (a temperature and a moisture field, ~16 km
 * and ~8 km), the geology (mountain ranges, highland plateaus, rugged dry
 * country, karst, lake districts, marshes) and two rare features placed by
 * hash: volcanoes and sacred groves (terrainMath.ts landRegions). Each region
 * shapes the land its own way (LANDFORM: plains roll, mountains soar, mesas
 * stand in tiers, karst rises in towers, dunes march, marshes lie flat at the
 * water), colours its ground (LOOK: soil, grass, rock and their accents),
 * grows its own plants (forest/placement.ts zones) and has its own air (AIR:
 * humid haze, marsh fog, desert glare, volcanic ash). The coast, rivers, lakes
 * and waterfalls run through all of them (terrainMath.ts landSurface).
 *
 * Tables here are plain numbers: the CPU twin (terrainMath.ts) reads them
 * directly and the WGSL twin (gpu/worldgen.wgsl.ts) compiles them into const
 * arrays, so the two cannot drift. Colours are linear RGB.
 */

export const BIOMES = [
  'plains', 'forest', 'rainforest', 'jungle', 'wetland', 'desert', 'badlands', 'savanna',
  'volcanic', 'mountains', 'highland', 'tundra', 'taiga', 'sacred', 'lakes',
] as const;
type Biome = (typeof BIOMES)[number];
export const BIOME_COUNT = BIOMES.length;
/** Index of each biome in BIOMES (and in every per-biome table and weight array). */
export const BI = Object.fromEntries(BIOMES.map((b, i) => [b, i])) as Record<Biome, number>;

/**
 * How a region shapes the land (metres unless noted), on top of the geography (hydro.ts: the
 * uplift surface and the rivers' valleys). The parameters blend linearly across region borders,
 * so landforms merge instead of meeting at seams.
 */
type Landform = {
  /** Rolling hills: amplitude of the hills' fBm (2 km down to 8 m), ± about the uplift surface. */
  hills: number;
  /** Mountain detail: amplitude of the ridged octaves (1.4 km spurs down), scaled by the range or massif there. */
  ridge: number;
  /** Karst towers (0–1): steep-sided limestone towers on a 300 m lattice. */
  karst: number;
  /** Mesas (0–1): tiers of flat-topped buttes with cliff walls and talus skirts. */
  mesa: number;
  /** Dunes (0–1): crescentic sand ridges with steep slip faces, and ripples. */
  dune: number;
  /** Marsh (0–1): the land flattened onto the nearest river's level, hummocks and pools. */
  flat: number;
  /** Valley walls' slope beyond the floor (0.06 meadow valleys … 1.2 canyon walls). */
  k: number;
  /** Rivers show water once this many nodes drain through them (dry country only its larger rivers). */
  riverMin: number;
  /** Gully erosion's strength (0 none … 1.4 deeply carved mountains). */
  erosion: number;
  /** Ponds: the chance a 384 m cell holds one. */
  pond: number;
};

const LF = (hills: number, ridge: number, karst: number, mesa: number, dune: number, flat: number,
  k: number, riverMin: number, erosion: number, pond: number): Landform =>
  ({ hills, ridge, karst, mesa, dune, flat, k, riverMin, erosion, pond });

export const LANDFORM: Record<Biome, Landform> = {
  //               hills ridge karst mesa dune flat  k     riverMin erosion pond
  plains:     LF(42, 0, 0, 0, 0, 0, 0.07, 1, 0.5, 0.675),
  forest:     LF(62, 30, 0, 0, 0, 0, 0.12, 1, 0.75, 0.45),
  rainforest: LF(78, 60, 0, 0, 0, 0, 0.2, 1, 1.0, 0.3375),
  jungle:     LF(48, 40, 1, 0, 0, 0, 0.25, 1, 0.8, 0.225),
  wetland:    LF(8, 0, 0, 0, 0, 0.92, 0.04, 1, 0, 0.95),
  desert:     LF(22, 0, 0, 0, 1, 0, 0.1, 6, 0.25, 0.045),
  badlands:   LF(30, 20, 0, 1, 0, 0, 0.9, 3, 1.1, 0),
  savanna:    LF(30, 0, 0, 0, 0, 0, 0.08, 3, 0.4, 0.225),
  volcanic:   LF(36, 40, 0, 0, 0, 0, 0.3, 4, 0.9, 0),
  mountains:  LF(55, 200, 0, 0, 0, 0, 0.6, 1, 1.3, 0.09),
  highland:   LF(50, 40, 0, 0, 0, 0, 0.3, 1, 0.8, 0.675),
  tundra:     LF(32, 20, 0, 0, 0, 0, 0.1, 1, 0.4, 0.7875),
  taiga:      LF(58, 40, 0, 0, 0, 0, 0.15, 1, 0.65, 0.5625),
  sacred:     LF(48, 20, 0, 0, 0, 0, 0.12, 1, 0.6, 0.45),
  lakes:      LF(40, 20, 0, 0, 0, 0, 0.12, 1, 0.5, 0.95),
};
/** The landform parameters in the order the WGSL tables and the blend loop use. */
export const LANDFORM_KEYS = ['hills', 'ridge', 'karst', 'mesa', 'dune', 'flat', 'k', 'riverMin', 'erosion', 'pond'] as const;

/**
 * How a region's ground looks: blended per clipmap lattice point (the ground's
 * baked biome layers) and finished per pixel by slope, altitude, water and the
 * patch fields. Linear RGB.
 */
type Look = {
  /** Bare ground: soil, sand, ash, mud. */
  soil: readonly [number, number, number];
  /** Grass and low plants at their lushest, and where they dry out (patches, dry slopes). */
  grass: readonly [number, number, number];
  dry: readonly [number, number, number];
  /** Rock: cliffs, outcrops and scree. */
  rock: readonly [number, number, number];
  /** How much of the ground grass covers (0–1), and the blades' height (× the grass spec). */
  cover: number;
  blade: number;
  /** Rock banded in strata (0–1): red and cream layers on cliffs. */
  strata: number;
  /** Ground wet and dark (0–1): mud, peat, damp leaf litter. */
  wet: number;
  /** Snow lies lower than the air's temperature alone says (°C added to the snow line's). */
  frost: number;
  /** Flowers among the grass (share of blades). */
  bloom: number;
};

type V3 = readonly [number, number, number];
const LK = (soil: V3, grass: V3, dry: V3, rock: V3, cover: number, blade: number, strata: number, wet: number, frost: number, bloom: number): Look =>
  ({ soil, grass, dry, rock, cover, blade, strata, wet, frost, bloom });

export const LOOK: Record<Biome, Look> = {
  plains:     LK([0.2, 0.15, 0.09], [0.13, 0.215, 0.045], [0.3, 0.27, 0.1], [0.17, 0.155, 0.13], 0.95, 1.25, 0, 0, 0, 0.06),
  forest:     LK([0.095, 0.07, 0.042], [0.055, 0.14, 0.035], [0.16, 0.15, 0.05], [0.15, 0.145, 0.12], 0.82, 0.85, 0, 0.35, 0, 0.015),
  rainforest: LK([0.085, 0.055, 0.032], [0.03, 0.125, 0.032], [0.08, 0.14, 0.04], [0.12, 0.13, 0.1], 1, 1.15, 0, 0.6, 0, 0.02),
  jungle:     LK([0.15, 0.08, 0.042], [0.05, 0.16, 0.03], [0.12, 0.16, 0.04], [0.16, 0.16, 0.13], 0.95, 1.05, 0, 0.4, 0, 0.02),
  wetland:    LK([0.075, 0.065, 0.045], [0.1, 0.145, 0.05], [0.2, 0.18, 0.08], [0.13, 0.13, 0.115], 0.9, 2.3, 0, 1, 0, 0.01),
  desert:     LK([0.44, 0.3, 0.155], [0.3, 0.26, 0.12], [0.42, 0.34, 0.16], [0.46, 0.3, 0.17], 0.015, 0.5, 0.25, 0, 0, 0),
  badlands:   LK([0.42, 0.2, 0.09], [0.24, 0.2, 0.08], [0.36, 0.27, 0.11], [0.52, 0.23, 0.1], 0.06, 0.55, 1, 0, 0, 0),
  savanna:    LK([0.4, 0.27, 0.14], [0.33, 0.27, 0.08], [0.45, 0.36, 0.14], [0.36, 0.3, 0.23], 0.92, 1.7, 0, 0, 0, 0.02),
  volcanic:   LK([0.052, 0.045, 0.042], [0.07, 0.08, 0.035], [0.085, 0.07, 0.045], [0.065, 0.055, 0.05], 0.1, 0.5, 0, 0, -40, 0),
  mountains:  LK([0.18, 0.16, 0.13], [0.08, 0.13, 0.05], [0.2, 0.18, 0.1], [0.2, 0.195, 0.19], 0.5, 0.6, 0, 0, 0, 0.02),
  highland:   LK([0.16, 0.12, 0.08], [0.12, 0.15, 0.06], [0.24, 0.19, 0.1], [0.18, 0.17, 0.15], 0.8, 0.7, 0, 0.2, 0, 0.05),
  tundra:     LK([0.15, 0.14, 0.12], [0.16, 0.15, 0.09], [0.22, 0.2, 0.14], [0.19, 0.19, 0.2], 0.4, 0.4, 0, 0, 6, 0),
  taiga:      LK([0.1, 0.08, 0.06], [0.06, 0.12, 0.05], [0.14, 0.13, 0.07], [0.16, 0.16, 0.155], 0.6, 0.6, 0, 0.2, 1.5, 0.01),
  sacred:     LK([0.07, 0.08, 0.04], [0.05, 0.2, 0.09], [0.1, 0.2, 0.08], [0.13, 0.15, 0.12], 1, 0.5, 0, 0.3, 0, 0.08),
  lakes:      LK([0.12, 0.1, 0.07], [0.08, 0.16, 0.045], [0.18, 0.17, 0.07], [0.16, 0.155, 0.14], 0.9, 0.9, 0, 0.2, 0, 0.04),
};

/**
 * Snow's albedo (linear), on every renderer. Lower than fresh snow's (~0.8): the display is exposed
 * for the green lands, and snow much brighter than this sits in the tone curve's shoulder, flat
 * white, its slopes and drifts lost.
 */
export const SNOW_ALBEDO: V3 = [0.4, 0.44, 0.5];

/**
 * A region's air, round the camera: haze (× the day's), mist lying in the low
 * ground (added to the morning's), and the haze's tint.
 */
type Air = { haze: number; mist: number; tint: readonly [number, number, number] };
export const AIR: Record<Biome, Air> = {
  plains:     { haze: 1.0, mist: 0, tint: [1, 1, 1] },
  forest:     { haze: 1.1, mist: 0.15, tint: [0.98, 1.0, 0.97] },
  rainforest: { haze: 1.7, mist: 0.55, tint: [0.97, 1.02, 0.96] },
  jungle:     { haze: 1.5, mist: 0.45, tint: [0.97, 1.02, 0.97] },
  wetland:    { haze: 1.35, mist: 1.0, tint: [0.97, 1.0, 0.98] },
  desert:     { haze: 1.25, mist: 0, tint: [1.06, 0.99, 0.88] },
  badlands:   { haze: 1.1, mist: 0, tint: [1.05, 0.98, 0.9] },
  savanna:    { haze: 1.2, mist: 0, tint: [1.04, 1.0, 0.92] },
  volcanic:   { haze: 1.25, mist: 0.1, tint: [0.93, 0.89, 0.86] },
  mountains:  { haze: 0.8, mist: 0.1, tint: [0.98, 1.0, 1.03] },
  highland:   { haze: 0.9, mist: 0.3, tint: [0.98, 1.0, 1.02] },
  tundra:     { haze: 0.75, mist: 0.15, tint: [0.96, 0.99, 1.05] },
  taiga:      { haze: 1.0, mist: 0.3, tint: [0.98, 1.0, 1.01] },
  sacred:     { haze: 1.3, mist: 0.7, tint: [0.98, 1.03, 0.97] },
  lakes:      { haze: 1.0, mist: 0.4, tint: [0.98, 1.0, 1.02] },
};

/** A readable name for each biome (HUD, docs). */
export const BIOME_LABEL: Record<Biome, string> = {
  plains: 'Grassland plains', forest: 'Temperate forest', rainforest: 'Tropical rainforest', jungle: 'Jungle karst',
  wetland: 'Wetland marsh', desert: 'Desert dunes', badlands: 'Canyon badlands', savanna: 'Savanna',
  volcanic: 'Volcanic fields', mountains: 'Rocky mountains', highland: 'Alpine highland', tundra: 'Snowy tundra',
  taiga: 'Boreal taiga', sacred: 'Ancient forest', lakes: 'Lake district',
};
