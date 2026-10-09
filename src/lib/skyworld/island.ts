/**
 * One sky island's shape — the CPU twin (fp64) of the WGSL in
 * gpu/skyIslands.wgsl.ts, which evaluates the same expressions in fp32 on
 * what the CPU uploads (keep the two in step).
 * ---------------------------------------------------------------------------
 * The noise runs on island-local coordinates (x − cx, z − cz), so it is as
 * precise a thousand kilometres out as at the origin. ρ is the distance from
 * the centre over the outline's radius at that bearing: the top exists for
 * ρ ≤ 1, where it meets the underside.
 *
 * Top: the biome's relief (a low dome, rolling hills, ridged crags, a summit)
 * fading out toward the rim, which rounds down by rimDrop; then the water
 * carved into it — the lake's bowl, its shore rising gently from the water
 * line, a low levee where the land would let the water spill, and the
 * stream's channel from the lake's outlet to a notch in the rim, its bed
 * never rising downstream (hydrology.ts chose it so).
 * Underside: a cliff wall under the rim, then an inverted mountain of
 * layered rock (cone, boulders, drip-stone spikes), notched where the stream
 * leaves so top and underside still close at the lip.
 */

import { SKY_REALM as S } from '../constants';
import { vnoise } from '../terrainMath';
import type { SkyBiome } from './biomes';

export type IslandKind = 'hub' | 'satellite' | 'isle' | 'debris' | 'lone';

/** What hangs under an island (SKY_CIVILIZATION.md §2, §7): bare rock, rock veined with glowing skystone, roots, basalt, ice. */
export const UNDERSIDES = ['rock', 'crystal', 'roots', 'basalt', 'ice'] as const;
export type Underside = (typeof UNDERSIDES)[number];

/** Where an Eyrie has paved and built on an island (eyrie/district.ts builds and reads it). */
export type DistrictMap = {
  /** Half side of its square (m) and texel size (m); the square is centred on the island. */
  half: number;
  cell: number;
  /** DISTRICT_N² × 2 bytes: paving, footprints (rg8unorm). */
  data: Uint8Array<ArrayBuffer>;
};

/** An extra hanging spike under an island: island-local centre, radius, how far it hangs below the underside's base. */
export type Lobe = { x: number; z: number; r: number; d: number };

/** A lake: an elliptical bowl with a wobbly shore, island-local. */
export type Lake = {
  x: number; z: number;     // centre (island-local m)
  r: number;                // radius along its major axis (m), before the shore's wobble
  aspect: number;           // minor / major axis
  cos: number; sin: number; // direction of the major axis
  b0: number; b1: number;   // shore wobble phases
  level: number;            // water surface (m)
  depth: number;            // at the middle (m)
  frozen: boolean;
};

/** A stream from the lake's outlet to the lip on the rim: island-local points, bed heights (never rising), half-widths. */
type Stream = { pts: Array<readonly [number, number, number, number]> };

/** Water depth over a stream's bed (m). */
export const STREAM_DEPTH = 0.45;

/** Where a stream pours off the rim, and how far it falls before it has all turned to mist. */
type Fall = {
  x: number; y: number; z: number;  // the lip (world m): the water's surface at the rim
  dirX: number; dirZ: number;       // horizontal direction it leaves in (unit)
  width: number;                    // m, at the lip
  speed: number;                    // m/s over the lip
  reach: number;                    // m fallen when the last of it has become mist
  seed: number;
};

export type SkyIsland = {
  kind: IslandKind;
  cx: number; cz: number;           // centre (world m)
  R: number;                        // outline radius before the wobble (m)
  y: number;                        // base height of the top (m); the rim lies rimDrop below
  amp: number;                      // outline wobble
  a: readonly [number, number, number, number]; // outline phases
  hill: number; wl: number;         // relief height (m) and wavelength (m)
  crag: number; peak: number;       // ridged knobs, central summit (× hill)
  cliff: number; depth: number;     // underside: rim wall, inverted mountain (m)
  seed: number;                     // noise seed (int32)
  biome: SkyBiome;
  /** Shape the Eyries give their islands (0: none): terraces this many metres high, a rim raised round a bowl (× hill), a tilt (rise per metre along x, z). */
  terrace: number; bowl: number; tiltX: number; tiltZ: number;
  /** Extra hanging spikes under it (≤ 4), and what its underside is made of. */
  lobes: Lobe[];
  under: Underside;
  /** Its water's tint: 0 the biome's, 1 the turquoise of a tended lagoon. */
  tint: number;
  /** Where its Eyrie has paved and built on it (eyrie/district.ts), once its buildings are raised. */
  district?: DistrictMap;
  lake: Lake | null;
  stream: Stream | null;
  fall: Fall | null;
};

const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** Outline radius factor at bearing θ (1 ± ~1.12 amp). */
export function outline(isl: SkyIsland, th: number): number {
  const a = isl.a;
  return 1 + isl.amp * (0.5 * Math.sin(3 * th + a[0]) + 0.3 * Math.sin(5 * th + a[1]) + 0.2 * Math.sin(8 * th + a[2]) + 0.12 * Math.sin(13 * th + a[3]));
}

/** The outline's largest radius factor, for bounds. */
export const outlineMax = (isl: SkyIsland) => 1 + 1.12 * isl.amp;

/** Normalised radius ρ of world (x, z). */
export function islandRho(isl: SkyIsland, x: number, z: number): number {
  const qx = x - isl.cx, qz = z - isl.cz;
  return Math.hypot(qx, qz) / (isl.R * outline(isl, Math.atan2(qz, qx)));
}

/** Signed distance (m) inside the outline along the bearing: positive inside, negative outside. */
export function insideBy(isl: SkyIsland, x: number, z: number): number {
  const qx = x - isl.cx, qz = z - isl.cz;
  return isl.R * outline(isl, Math.atan2(qz, qx)) - Math.hypot(qx, qz);
}

/**
 * Terraces: a height h folded into level steps T high, each rising to the next over the last
 * quarter of its run (a steep bank the Eyries face with walls and stairs). Shared with the WGSL.
 */
function terraced(h: number, T: number): number {
  if (T <= 0) return h;
  const k = Math.floor(h / T), f = h / T - k;
  return T * (k + smooth(0.75, 1, f));
}

/** The top before the water is carved into it, at island-local (qx, qz) with normalised radius rho. */
function relief(isl: SkyIsland, qx: number, qz: number, rho: number): number {
  const wl = isl.wl, s = isl.seed;
  const n = 0.5 + 0.5 * (0.6 * vnoise(qx / wl, qz / wl, s) + 0.28 * vnoise(qx / (0.45 * wl), qz / (0.45 * wl), s + 1)
    + 0.12 * vnoise(qx / (0.2 * wl), qz / (0.2 * wl), s + 2));
  let h = isl.hill * (0.35 * (1 - rho * rho) + 0.65 * n);
  if (isl.crag > 0) {
    const k = 1 - Math.abs(vnoise(qx / (0.8 * wl), qz / (0.8 * wl), s + 3));
    h += isl.hill * isl.crag * 1.1 * k * k;
  }
  if (isl.peak > 0) h += isl.hill * isl.peak * 2.2 * Math.exp(-(rho / 0.32) * (rho / 0.32));
  if (isl.bowl > 0) h += isl.hill * isl.bowl * smooth(0.25, 0.85, rho);
  h = terraced(h, isl.terrace);
  return isl.y + h * (1 - smooth(0.72, 1, rho)) - S.rimDrop * smooth(0.84, 1, rho) + isl.tiltX * qx + isl.tiltZ * qz;
}

/** The top before the water is carved into it (hydrology.ts plans the water on it). */
export function baseTop(isl: SkyIsland, x: number, z: number): number {
  const qx = x - isl.cx, qz = z - isl.cz;
  const rho = Math.hypot(qx, qz) / (isl.R * outline(isl, Math.atan2(qz, qx)));
  return relief(isl, qx, qz, rho);
}

/** Normalised distance from a lake's middle (1 on its shore), island-local. */
export function lakeDist(L: Lake, qx: number, qz: number): number {
  const px = qx - L.x, pz = qz - L.z;
  const u = px * L.cos + pz * L.sin;
  const v = (-px * L.sin + pz * L.cos) / L.aspect;
  const th = Math.atan2(v, u);
  return Math.hypot(u, v) / (L.r * (1 + 0.12 * Math.sin(3 * th + L.b0) + 0.07 * Math.sin(5 * th + L.b1)));
}

/**
 * The lake's bowl and shore at normalised distance d (the top takes the lower
 * of this and the land): a gentle beach out to 1.5 radii, then rising steeply,
 * so by 3 radii it stands above any land and the carve ends without a seam.
 */
function lakeBed(L: Lake, d: number): number {
  const c = Math.min(d, 1);
  const out = Math.max(d - 1.5, 0);
  return L.level - L.depth * Math.pow(1 - c * c, 0.7) + L.r * (0.12 * Math.max(d - 1, 0) + 6 * out * out);
}

type StreamPoint = { s: number; bed: number; hw: number; t: number };

/** Nearest point of a stream into `out`: distance (m) from its axis, the bed there, its half-width, and how far along it (0 outlet … 1 lip). */
function nearestOnStream(st: Stream, qx: number, qz: number, out: StreamPoint): StreamPoint {
  const P = st.pts;
  let best = Infinity, bed = 0, hw = 1, t = 0;
  for (let i = 0; i + 1 < P.length; i++) {
    const a = P[i], b = P[i + 1];
    const dx = b[0] - a[0], dz = b[1] - a[1];
    const l2 = dx * dx + dz * dz;
    const f = l2 > 0 ? Math.min(1, Math.max(0, ((qx - a[0]) * dx + (qz - a[1]) * dz) / l2)) : 0;
    const ex = qx - a[0] - f * dx, ez = qz - a[1] - f * dz;
    const d2 = ex * ex + ez * ez;
    if (d2 < best) {
      best = d2;
      bed = a[2] + f * (b[2] - a[2]);
      hw = a[3] + f * (b[3] - a[3]);
      t = (i + f) / (P.length - 1);
    }
  }
  out.s = Math.sqrt(best); out.bed = bed; out.hw = hw; out.t = t;
  return out;
}

// The surfaces' own scratch (the hero asks every tick: nothing is allocated).
const near: StreamPoint = { s: 0, bed: 0, hw: 1, t: 0 };

/** Nearest point of a stream: distance (m) from its axis, the bed there, its half-width, and how far along it (0 outlet … 1 lip). */
export function streamNear(st: Stream, qx: number, qz: number): StreamPoint {
  return nearestOnStream(st, qx, qz, { s: 0, bed: 0, hw: 1, t: 0 });
}

/** How far from its axis (× half-width) a stream's channel is carved: past it the banks stand above any land. */
export const CHANNEL_REACH = 5;

/**
 * The stream's channel at distance s from its axis: the bed, rounding up to
 * its banks, a gentle slope, then steepening (a ravine where it cuts through
 * high ground; hydrology.ts keeps those shallow), so by CHANNEL_REACH
 * half-widths it stands above any land.
 */
function channel(bed: number, hw: number, s: number): number {
  const u = s / hw;
  if (u < 1) return bed + 0.9 * u * u;
  const g = Math.max(u - 1.6, 0);
  return bed + 0.9 + hw * (0.35 * (u - 1) + 1.6 * g * g);
}

/** The top at world (x, z): the relief with the lake and the stream carved in (meaningful for ρ ≤ 1). */
export function topAt(isl: SkyIsland, x: number, z: number): number {
  const qx = x - isl.cx, qz = z - isl.cz;
  const rho = Math.hypot(qx, qz) / (isl.R * outline(isl, Math.atan2(qz, qx)));
  let t = relief(isl, qx, qz, rho);
  const L = isl.lake;
  if (L) {
    const d = lakeDist(L, qx, qz);
    if (d < 3) {
      t = Math.min(t, lakeBed(L, d));
      // A low levee where the land would let the water spill (never inside the lake).
      if (d > 1) t = Math.max(t, L.level + 0.05 * smooth(1, 1.06, d) - 40 * smooth(1.15, 1.45, d));
    }
  }
  if (isl.stream) {
    const n = nearestOnStream(isl.stream, qx, qz, near);
    if (n.s < CHANNEL_REACH * n.hw) t = Math.min(t, channel(n.bed, n.hw, n.s));
  }
  return t;
}

/** The underside at world (x, z) (meaningful for ρ ≤ 1); it meets the top at the rim. */
export function bottomAt(isl: SkyIsland, x: number, z: number): number {
  const qx = x - isl.cx, qz = z - isl.cz;
  const rho = Math.hypot(qx, qz) / (isl.R * outline(isl, Math.atan2(qz, qx)));
  const R = isl.R, s = isl.seed;
  const wall = isl.cliff * (1 - smooth(0.86, 1, rho));
  const v = Math.max(0, 1 - rho / 0.86);
  const cone = isl.depth * Math.pow(v, 0.9) * (0.8 + 0.35 * (0.5 + 0.5 * vnoise(qx / (0.22 * R), qz / (0.22 * R), s + 5)));
  const rock = 0.05 * R * (0.5 + 0.5 * vnoise(qx / (0.09 * R), qz / (0.09 * R), s + 6)) * (1 - smooth(0.8, 1, rho));
  // Hanging spires of rock, two sizes, out nearly to the rim.
  const dn = Math.max(0, vnoise(qx / (0.07 * R), qz / (0.07 * R), s + 7));
  const dn2 = Math.max(0, vnoise(qx / (0.042 * R), qz / (0.042 * R), s + 8));
  const drip = 0.17 * R * dn * dn * dn * (1 - smooth(0.6, 0.95, rho)) + 0.07 * R * dn2 * dn2 * dn2 * (1 - smooth(0.4, 0.97, rho));
  let b = isl.y - S.rimDrop - wall - cone - rock - drip;
  // The extra spikes: inverted cones off the middle, so an underside hangs as a cluster of spires.
  // (they die away toward the rim, where top and underside meet)
  const lobeRim = 1 - smooth(0.85, 1, rho);
  for (const L of isl.lobes) {
    const t = Math.max(0, 1 - Math.hypot(qx - L.x, qz - L.z) / L.r);
    if (t > 0) b = Math.min(b, isl.y - S.rimDrop - wall - L.d * lobeRim * Math.pow(t, 1.6) * (0.88 + 0.24 * (0.5 + 0.5 * vnoise((qx - L.x) / (0.3 * L.r), (qz - L.z) / (0.3 * L.r), s + 9))));
  }
  b += isl.tiltX * qx + isl.tiltZ * qz;
  // Where the stream notches the rim, the underside follows the notch down.
  if (isl.stream && rho > 0.8) {
    const n = nearestOnStream(isl.stream, qx, qz, near);
    if (n.s < CHANNEL_REACH * n.hw) b = Math.min(b, channel(n.bed, n.hw, n.s));
  }
  return b;
}

/** The deepest the underside can reach below the top's base height (m): its tip clears SKY_REALM.floor. */
export function maxDrop(isl: Pick<SkyIsland, 'R' | 'cliff' | 'depth'> & { lobes?: readonly Lobe[]; tiltX?: number; tiltZ?: number }): number {
  const lobes = isl.lobes ? Math.max(0, ...isl.lobes.map((L) => L.d)) : 0;
  const tilt = (Math.abs(isl.tiltX ?? 0) + Math.abs(isl.tiltZ ?? 0)) * isl.R * 1.2;
  return S.rimDrop + isl.cliff + Math.max(1.15 * isl.depth + 0.05 * isl.R + 0.24 * isl.R, 1.12 * lobes) + tilt;
}

/** The water's surface at world (x, z) on this island — its lake or its stream — else NaN. */
export function waterAt(isl: SkyIsland, x: number, z: number): number {
  const qx = x - isl.cx, qz = z - isl.cz;
  const L = isl.lake;
  if (L && lakeDist(L, qx, qz) < 1.02) return L.level;
  if (isl.stream) {
    const n = nearestOnStream(isl.stream, qx, qz, near);
    if (n.s < 0.75 * n.hw) return n.bed + STREAM_DEPTH;
  }
  return NaN;
}

/** Is world (x, z) within `margin` m of this island's water (the lake's shore, the stream's banks)? */
export function nearWater(isl: SkyIsland, x: number, z: number, margin: number): boolean {
  const qx = x - isl.cx, qz = z - isl.cz;
  const L = isl.lake;
  if (L && lakeDist(L, qx, qz) < 1 + margin / L.r) return true;
  if (isl.stream) {
    const n = nearestOnStream(isl.stream, qx, qz, near);
    if (n.s < n.hw + margin) return true;
  }
  return false;
}
