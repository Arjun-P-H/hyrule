/**
 * The five places the terrain's artifacts were reported at (seed DEFAULT_SEED, 12:00), and the
 * grid the forms tooling samples round each: 2 × 2 km at 0.5 m, cell centres, x across a row.
 */
export type Place = { id: string; name: string; x: number; z: number; camY?: number; landform: string };

export const PLACES: Place[] = [
  { id: 'p1', name: 'jungle karst: parallel walls cut into a comb of teeth', x: 5767, z: -25258, landform: 'karst' },
  { id: 'p2', name: 'mountain flank with regular dark corrugations, rippled hill', x: 5571.8, z: -25027.4, camY: 122, landform: 'ridge' },
  { id: 'p3', name: 'stair-stepped river banks and pond edges', x: 1849, z: -2753, landform: 'water' },
  { id: 'p4', name: 'sea cliff: toothed rim, melted face', x: 1205, z: -2644, landform: 'cliff' },
  { id: 'p5', name: 'karst towers as smooth domes with stretched rock', x: -2585.4, z: -3520.1, camY: 233, landform: 'karst' },
];

/** Samples per side, spacing (m) and the prefilter footprint they are taken at (ring 0's). */
export const GRID = { n: 4000, step: 0.5, dp: 0.5 };

/** The terms forms/dump.ts writes, one float32 grid each (terrainMath.ts TERMS). */
export const TERM_KEYS = ['E', 'W', 'U', 'hills', 'ridges', 'karst', 'dunes', 'mesas', 'gullies', 'scale', 'lift', 'terraces', 'valleys', 'rest'] as const;

/** World x, z of sample (i, j) of a place's grid. */
export function gridXZ(p: Place, i: number, j: number, n = GRID.n, step = GRID.step): [number, number] {
  return [p.x - (n / 2) * step + (i + 0.5) * step, p.z - (n / 2) * step + (j + 0.5) * step];
}
