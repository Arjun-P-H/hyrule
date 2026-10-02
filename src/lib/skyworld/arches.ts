/**
 * Arches: the natural rock bridges that join an archipelago's islands.
 * ---------------------------------------------------------------------------
 * An arch springs from inside one island's rim to inside another's: a deck
 * of grass over rock, wide where it roots into the islands and slender in
 * the middle, bowed a little sideways and humped a little up, with a rock
 * body under it that thins toward the middle (the arch's span). Its deck
 * never climbs steeper than SKY_REALM.linkSlope, so the hero can walk from
 * island to island. Deck coordinates: t along the axis A → B (0..1), s
 * across it, measured along the axis's fixed sideways direction (so a point
 * maps back to (t, s) exactly, bow included). The renderer draws the same
 * surfaces (gpu/skyIslands.wgsl.ts); the hero stands on archGround.
 *
 * `kind` leaves room for the sky civilization to come: its built bridges
 * (stone, timber, rope) will be arches of another kind on the same axis.
 */

export type Arch = {
  kind: 'arch';
  from: number; to: number;            // the islands it joins (indices in the archipelago)
  ax: number; az: number; ay: number;  // deck axis start (world m), inside the first island's rim
  bx: number; bz: number; by: number;  // end, inside the second's
  dirX: number; dirZ: number;          // along the axis (unit); sideways is (−dirZ, dirX)
  len: number;                         // m
  bow: number;                         // sideways bow at the middle (m)
  rise: number;                        // hump at the middle (m)
  wMid: number; wEnd: number;          // deck width, middle and ends (m)
  thMid: number; thEnd: number;        // rock under the deck, middle and ends (m)
  seed: number;
};

/** How far toward full width the deck has grown at t (0 at the ends … 1 in the middle). */
const span = (t: number) => Math.sqrt(Math.max(0, Math.sin(Math.PI * t)));

/** Deck width at t (the rock under it, thMid … thEnd, the renderer shapes itself). */
const archWidth = (A: Arch, t: number) => A.wEnd + (A.wMid - A.wEnd) * span(t);

/** Deck axis height at t. */
export const archY = (A: Arch, t: number) => A.ay + (A.by - A.ay) * t + A.rise * Math.sin(Math.PI * t);

/** Sideways offset of the axis at t (the bow). */
const archBow = (A: Arch, t: number) => A.bow * Math.sin(Math.PI * t);

/** The deck's surface across it (crowned: highest on the axis), at t and s, for |s| ≤ half its width. */
function archDeckY(A: Arch, t: number, s: number): number {
  const w = archWidth(A, t);
  const u = (2 * s) / w;
  return archY(A, t) + 0.3 * (1 - u * u);
}

/**
 * The deck under world (x, z) where it can be stood on (the outer tenth of
 * each side rounds off), else NaN. Deck coordinates: t along the axis, s
 * across from it (bow included). Allocates nothing (the hero asks every tick).
 */
export function archGround(A: Arch, x: number, z: number): number {
  const px = x - A.ax, pz = z - A.az;
  const t = (px * A.dirX + pz * A.dirZ) / A.len;
  if (t < 0 || t > 1) return NaN;
  const s = -px * A.dirZ + pz * A.dirX - archBow(A, t);
  if (Math.abs(s) > 0.45 * archWidth(A, t)) return NaN;
  return archDeckY(A, t, s);
}
