/**
 * Trees on the sky islands: the forest's sky layer — the CPU twin of the
 * rules forest.wgsl.ts runs on the GPU (skyPlantAt; keep the two in step).
 * ---------------------------------------------------------------------------
 * Every 6 m lattice cell of the forest (placement.ts) may hold a plant up
 * here as well as one on the land below, so an island no longer leaves a
 * bare patch on the ground under it. A cell's spot is hashed (its own
 * hashes, not the ground layer's); on an island, its biome (biomes.ts)
 * decides what grows:
 *   forest   where the island's own stand field (70 m) runs high, cover of
 *            the biome's density (dense cloud forest, sparse crags);
 *   lone     one spot per 30 m cell, kept with the biome's chance;
 *   bushes   the biome's share of the rest, in its bush palette;
 * which tree: the biome's kinds by weight, picked by a 45 m stand field
 * (stands of one kind, mixed at their edges). Nothing grows near the rim,
 * in or beside the water, on steep ground, on the floating rocks, or in an
 * Eyrie's streets and buildings (its district map).
 * Trees are the ground's shapes (species.ts) in the sky's colours
 * (SKY_FLORA): the palette rides along in the plant record.
 */

import { CELL, type Species } from '../forest/species';
import { hash3, vnoise } from '../terrainMath';
import { SKY_BIOME_INFO, SKY_FLORA } from './biomes';
import { islandRho, lakeDist, streamNear, topAt } from './island';
import { islandAt } from './realm';
import { districtAt } from './eyrie/district';

const dd: [number, number] = [0, 0];

type SkyPlant = {
  species: Species;
  palette: number;          // index into SKY_FLORA
  x: number; z: number;     // world m
  y: number;                // the island's top there
  scale: number;
  snow: boolean;
};

/** Rules shared with the GPU. */
export const SKY_PLACE = {
  field: 70,      // m — wavelength of an island's stand field (where its forest is)
  stand: 45,      // m — wavelength of the field that picks the kind of tree
  lone: 5,        // cells per side of a lone tree's coarse cell (30 m)
  rim: 0.92,      // nothing grows past this share of the radius
  slope: 0.6,     // nor on ground steeper than this
  lake: 1.15,     // nor within this many lake radii of its middle
  stream: 1.6,    // nor within this many half-widths (+ 1 m) of a stream
  paved: 2.5,     // nor within this many metres of an Eyrie's paving (eyrie/district.ts)
  built: 3.5,     // nor of its footprints
} as const;

const u16 = (h: number, shift: number) => ((h >>> shift) & 0xffff) / 65535;
const smooth = (lo: number, hi: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
};

/** The sky plant in lattice cell (ix, iz), or null. */
export function skyPlantInCell(ix: number, iz: number, seed: number): SkyPlant | null {
  const h0 = hash3(ix, iz, seed + 7301), h1 = hash3(ix, iz, seed + 7302), h2 = hash3(ix, iz, seed + 7303);
  let x = (ix + 0.08 + 0.84 * u16(h0, 0)) * CELL, z = (iz + 0.08 + 0.84 * u16(h0, 16)) * CELL;
  const isl = islandAt(x, z, seed);
  if (!isl || isl.kind === 'debris') return null;
  const rules = SKY_BIOME_INFO[isl.biome].trees;
  const keep = u16(h1, 0), pick = u16(h1, 16), bush = u16(h2, 0);

  // Lone tree: this cell hosts its coarse cell's spot?
  const k = SKY_PLACE.lone;
  const lx = Math.floor(ix / k), lz = Math.floor(iz / k);
  const lh = hash3(lx, lz, seed + 7311), lp = hash3(lx, lz, seed + 7312);
  const fx = (0.15 + 0.7 * u16(lh, 0)) * k, fz = (0.15 + 0.7 * u16(lh, 16)) * k;
  const lone = ix === lx * k + Math.floor(fx) && iz === lz * k + Math.floor(fz) && u16(lp, 0) < rules.lone;
  if (lone) { x = (lx * k + fx) * CELL; z = (lz * k + fz) * CELL; }

  // Where nothing grows.
  if (islandRho(isl, x, z) > SKY_PLACE.rim) return null;
  const qx = x - isl.cx, qz = z - isl.cz;
  if (isl.district) {
    districtAt(isl, x, z, dd);
    if (dd[0] < SKY_PLACE.paved || dd[1] < SKY_PLACE.built) return null;
  }
  if (isl.lake && lakeDist(isl.lake, qx, qz) < SKY_PLACE.lake) return null;
  if (isl.stream) {
    const n = streamNear(isl.stream, qx, qz);
    if (n.s < SKY_PLACE.stream * n.hw + 1) return null;
  }
  const y = topAt(isl, x, z);
  const slope = Math.hypot(topAt(isl, x + 1.5, z) - y, topAt(isl, x, z + 1.5) - y) / 1.5;
  if (slope > SKY_PLACE.slope) return null;

  const field = 0.5 + 0.5 * vnoise(qx / SKY_PLACE.field, qz / SKY_PLACE.field, isl.seed + 11);
  const forest = rules.forest * smooth(0.45, 0.7, field);
  if (lone || keep < forest) {
    // Which of the biome's trees: its stand, mixed a little at the edges.
    const stand = 0.5 + 0.5 * vnoise(qx / SKY_PLACE.stand, qz / SKY_PLACE.stand, isl.seed + 13);
    const v = Math.min(0.9999, Math.max(0, stand + 0.3 * (pick - 0.5)));
    let acc = 0;
    let kind = rules.kinds[rules.kinds.length - 1];
    for (const kd of rules.kinds) {
      acc += kd.weight;
      if (v < acc) { kind = kd; break; }
    }
    const f = (pick * 7.13) % 1;
    return {
      species: kind.species, palette: SKY_FLORA.indexOf(kind.palette), x, z, y,
      scale: (kind.scale[0] + (kind.scale[1] - kind.scale[0]) * f) * (lone ? 1.15 : 1), snow: !!kind.snow,
    };
  }
  if (bush < rules.bush) {
    return { species: 'bush', palette: SKY_FLORA.indexOf(rules.bushPalette), x, z, y, scale: 0.8 + 0.4 * pick, snow: SKY_BIOME_INFO[isl.biome].snow > 0.5 };
  }
  return null;
}
