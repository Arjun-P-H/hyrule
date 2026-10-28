/**
 * The world's geography and its rivers (TotK-style, infinite, deterministic).
 * ---------------------------------------------------------------------------
 * 1. The uplift surface U(w) at warped coordinates: rolling lowland, tablelands with cliff walls,
 *    mountain ranges whose crests run for tens of kilometres with peaks and passes along them, lone
 *    massifs, lake basins, all tapered down to the coast (cliffed coasts keep their height).
 * 2. The drainage graph: one node per 2,048 m cell (jittered). Each land node drains to its
 *    steepest-descent neighbour (or the sea); a node with no lower neighbour is a sink, and holds a
 *    lake. Its drainage `acc` is the exact count of nodes upstream of it (memoised; every basin ends
 *    at the sea or a lake, so it is finite and local in practice). The river's level at a node is
 *    U less an incision that grows downstream, so levels fall monotonically to the mouth.
 * 3. Each reach (node → downstream node) carries a profile: its level at eighths of its length,
 *    following U down (a running minimum, held at or above the lower end): pools where the land
 *    is flat, rapids where it falls, a waterfall where it drops more than HYDRO.fallDrop at once.
 * 4. riverField(w): the valleys every reach cuts (a channel meandering inside a valley floor, the
 *    floor, walls up to the land), the nearest channel (level, width, flow) and the lakes.
 *
 * The CPU computes nodes in fp64 and caches them; the GPU reads the same nodes from a table
 * (hydroTable) round the camera, so the river network has one topology on every side.
 */

import { CLIMATE, F, GEO, HYDRO, LANDS } from './constants';
import { hash3, n01, smoothstep, vnoise } from './noise';
import { LANDFORM } from './biomes';

const INV_4095 = 1.0 / 4095.0;
const INV_65535 = 1.0 / 65535.0;
/** A volcano's crater rim as a share of its cone's radius (where the cone reaches LANDS.volcano.crater of its height). */
export const CRATER_RIM = 1 - Math.pow(LANDS.volcano.crater, 1 / 1.9);

/* ------------------------------------------------------------------ */
/* 1. The uplift surface                                               */
/* ------------------------------------------------------------------ */

/** The continent field at warped (wx, wz): the sea lies below LANDS.coast.level. */
export function continentAt(wx: number, wz: number, seed: number): number {
  return 0.5 + 0.5 * (0.78 * vnoise(wx * F.cont, wz * F.cont, seed + 801) + 0.22 * vnoise(wx * F.coast, wz * F.coast, seed + 802));
}

/** What upliftAt found at the last point: continent, range (0–1), tableland (0–1), massif (0–1), coast taper, cliffed coast. */
export const UP = { C: 0, range: 0, plat: 0, massif: 0, taper: 1, cliff: 0, lava: -Infinity, craterRR: 1 };

/** A lake basin's depth at (wx, wz) (m): one candidate bowl per GEO.basinCell, kept inside its cell. */
function basinAt(wx: number, wz: number, seed: number): number {
  const cell = GEO.basinCell;
  const ix = Math.floor(wx / cell), iz = Math.floor(wz / cell);
  const h = hash3(ix, iz, seed + 3151);
  if ((h >>> 24) / 255 >= GEO.basinP) return 0;
  const cx = (ix + 0.3 + 0.4 * (h & 0xfff) * INV_4095) * cell, cz = (iz + 0.3 + 0.4 * ((h >>> 12) & 0xfff) * INV_4095) * cell;
  const h2 = hash3(ix, iz, seed + 3152);
  const R = GEO.basinR[0] + (GEO.basinR[1] - GEO.basinR[0]) * (h2 & 0xffff) * INV_65535;
  const d = Math.hypot(wx - cx, wz - cz) / R;
  if (d >= 1) return 0;
  const D = GEO.basinD[0] + (GEO.basinD[1] - GEO.basinD[0]) * (h2 >>> 16) * INV_65535;
  const q = 1 - smoothstep(0, 1, d);
  return D * q * Math.sqrt(q);
}

/**
 * Mountain ranges at warped (wx, wz) (0–1): the zero lines of an elongated noise in two chain
 * directions (blended over ~50 km, bent by a 9 km warp), raised where the 37 km mask allows, with
 * peaks and passes along the crest.
 */
function rangeAt(wx: number, wz: number, seed: number): number {
  const rm = smoothstep(GEO.rangeMask[0], GEO.rangeMask[1], n01(wx, wz, F.rangemask, seed + 3121));
  if (rm <= 0) return 0;
  // two chain directions, blended over ~50 km; the chains bent by a 9 km warp
  const bx = wx + GEO.rangeWarp * vnoise(wx * F.rangewarp, wz * F.rangewarp, seed + 3113);
  const bz = wz + GEO.rangeWarp * vnoise(wx * F.rangewarp + 3.7, wz * F.rangewarp - 1.9, seed + 3114);
  const ca = Math.cos(GEO.rangeDir[0]), sa = Math.sin(GEO.rangeDir[0]);
  const cb = Math.cos(GEO.rangeDir[1]), sb = Math.sin(GEO.rangeDir[1]);
  const c1 = 1 - Math.abs(vnoise((bx * ca + bz * sa) * F.rangeu, (bz * ca - bx * sa) * F.rangev, seed + 3111));
  const c2 = 1 - Math.abs(vnoise((bx * cb + bz * sb) * F.rangeu, (bz * cb - bx * sb) * F.rangev, seed + 3112));
  const c = c1 + (c2 - c1) * smoothstep(0.38, 0.62, n01(wx, wz, F.rangeblend, seed + 3115));
  const pk = GEO.pass + (1 - GEO.pass) * n01(wx, wz, F.peaks, seed + 3131);
  return rm * Math.pow(c, GEO.rangePow) * pk;
}

/** The volcano upliftAt found at the last point: centre, radius of its region, cone height, its region's weight (0: none). */
export const VOL = { x: 0, z: 0, R: 0, H: 0, w: 0, crater: -Infinity, rr: 1 };

/**
 * A volcano's cone at warped (wx, wz) (m): one candidate per LANDS.volcano.cell, kept in the
 * middle of its cell (so only its own cell can reach the point), on land; a stratovolcano's
 * concave flanks and its crater. Fills VOL.
 */
function volcanoAt(wx: number, wz: number, seed: number): number {
  const V = LANDS.volcano;
  VOL.w = 0; VOL.R = 0; VOL.crater = -Infinity; VOL.rr = 1;
  const ix = Math.floor(wx / V.cell), iz = Math.floor(wz / V.cell);
  const h = hash3(ix, iz, seed + 2101);
  if ((h >>> 24) / 255 >= V.p) return 0;
  const cx = (ix + 0.28 + 0.44 * (h & 0xfff) * INV_4095) * V.cell, cz = (iz + 0.28 + 0.44 * ((h >>> 12) & 0xfff) * INV_4095) * V.cell;
  const h2 = hash3(ix, iz, seed + 2102);
  const R = V.radius[0] + (V.radius[1] - V.radius[0]) * (h2 & 0xffff) * INV_65535;
  const r = Math.hypot(wx - cx, wz - cz);
  // (on land, not on a mountain range, and in land no colder than V.minT at sea level: a volcano stands
  // on its own, in warm country as TotK's does, never out of a snowfield)
  if (r >= R || continentAt(cx, cz, seed) <= LANDS.coast.level + 0.08 || rangeAt(cx, cz, seed) > 0.08) return 0;
  const T0 = CLIMATE.tBase + CLIMATE.tAmp * (vnoise(cx * F.temp, cz * F.temp, seed + 1201) + CLIMATE.tFine * vnoise(cx * F.tempfine, cz * F.tempfine, seed + 1202)) - CLIMATE.lambdaLat * Math.abs(cz);
  if (T0 < V.minT) return 0;
  VOL.x = cx; VOL.z = cz; VOL.R = R;
  VOL.H = V.height[0] + (V.height[1] - V.height[0]) * (h2 >>> 16) * INV_65535;
  VOL.w = 1 - smoothstep(0.55 * R, R, r); // (to nothing at R, where the region ends)
  const Rc = V.cone * R;
  if (r >= Rc) return 0;
  const q = 1 - r / Rc;
  let cone = VOL.H * q * Math.pow(q, 0.9);
  const top = VOL.H * V.crater;
  if (cone > top) cone = top - (cone - top) * 0.6;
  VOL.rr = r / Rc;
  // the crater's lava lake (on the cone's own scale), known out past its rim so the rim can be held up
  if (VOL.rr < CRATER_RIM * 1.6) VOL.crater = top - 0.05 * VOL.H;
  return cone;
}

/**
 * The land's height over the shore (m) for `raw` inland: the full height `inland` of the coastal band
 * in, toward the sea a beach (0) or, on a cliffed coast, a sea cliff of at most GEO.cliffH.
 */
function coastHeight(raw: number, inland: number, cliff: number, wx: number, wz: number, seed: number): number {
  if (inland >= 1) return raw;
  const cap = GEO.cliffH[0] + (GEO.cliffH[1] - GEO.cliffH[0]) * n01(wx, wz, F.cliffs, seed + 2702);
  return raw * inland + (1 - inland) * cliff * Math.min(raw, cap);
}

/** The uplift's raw height over the shore at warped (wx, wz) without a volcano (m): lowland, ranges, massifs, tablelands, basins. */
function baseAt(wx: number, wz: number, seed: number): number {
  const low = GEO.low[0] + GEO.low[1] * n01(wx, wz, F.upl, seed + 3101) + GEO.low[2] * n01(wx, wz, F.upl2, seed + 3102);
  const rng = rangeAt(wx, wz, seed);
  const ms = smoothstep(GEO.massif[0], GEO.massif[1], n01(wx, wz, F.massif, seed + 3141));
  const pn = 0.72 * n01(wx, wz, F.platcoarse, seed + 1401) + 0.28 * n01(wx, wz, F.platfine, seed + 1402);
  const pl = 0.82 * smoothstep(GEO.plat, GEO.plat + GEO.platEdge, pn) + 0.18 * smoothstep(GEO.plat - 0.05, GEO.plat, pn);
  const notRange = 1 - smoothstep(0.15, 0.45, rng);
  return low + Math.max(GEO.rangeH * rng, GEO.massifH * ms * ms) + GEO.platH * pl * notRange - basinAt(wx, wz, seed) * notRange;
}

/** The continent field's span over which the land rises from the sea floor at the coastline (~10 m). */
export const COAST_RAMP = 0.0006;

/** The uplift surface at warped (wx, wz) (m, absolute; the sea floor offshore). Fills UP and VOL. */
export function upliftAt(wx: number, wz: number, seed: number): number {
  const K = LANDS.coast;
  const C = continentAt(wx, wz, seed);
  UP.C = C;
  const cliff = smoothstep(K.cliffs[0], K.cliffs[1], n01(wx, wz, F.cliffs, seed + 2701));
  const inland = smoothstep(K.level, K.level + K.inland, C);
  const taper = cliff + (1 - cliff) * inland;
  UP.cliff = cliff;
  UP.taper = taper;
  // lowland: broad undulation and a finer swell
  const low = GEO.low[0] + GEO.low[1] * n01(wx, wz, F.upl, seed + 3101) + GEO.low[2] * n01(wx, wz, F.upl2, seed + 3102);
  const rng = rangeAt(wx, wz, seed);
  // lone massifs (landmark mountains off the ranges)
  const ms = smoothstep(GEO.massif[0], GEO.massif[1], n01(wx, wz, F.massif, seed + 3141));
  const massif = ms * ms;
  // tablelands: a cliff wall over a talus skirt
  const pn = 0.72 * n01(wx, wz, F.platcoarse, seed + 1401) + 0.28 * n01(wx, wz, F.platfine, seed + 1402);
  const pl = 0.82 * smoothstep(GEO.plat, GEO.plat + GEO.platEdge, pn) + 0.18 * smoothstep(GEO.plat - 0.05, GEO.plat, pn);
  UP.range = rng;
  UP.massif = massif;
  UP.plat = pl;
  const hi = Math.max(GEO.rangeH * rng, GEO.massifH * massif);
  const notRange = 1 - smoothstep(0.15, 0.45, rng);
  const base = low + hi + GEO.platH * pl * notRange - basinAt(wx, wz, seed) * notRange;
  const cone = volcanoAt(wx, wz, seed);
  // a cone stands on the land under its summit (blended out toward its foot), so its crater is level
  let raw = base + cone;
  UP.lava = -Infinity;
  if (VOL.rr < 1) {
    const bc = baseAt(VOL.x, VOL.z, seed);
    raw += (bc - base) * (1 - smoothstep(0.35, 1.0, VOL.rr));
    // its lava lake lies level, at the land under the summit plus the lake's height on the cone
    // (a volcano stands well inland: no coast taper there)
    if (VOL.crater > -Infinity) UP.lava = LANDS.sea + K.beach + Math.max(bc + VOL.crater, 1);
  }
  UP.craterRR = VOL.rr;
  const sea = LANDS.sea - 4 - 110 * smoothstep(K.level, K.level - 0.1, C);
  if (C < K.level - COAST_RAMP) return sea;
  // (the land rises from the sea floor over a steep but continuous face, ~10 m: a sea cliff, a beach's foot)
  const land = LANDS.sea + K.beach + coastHeight(Math.max(raw, 1), inland, cliff, wx, wz, seed);
  return sea + (land - sea) * smoothstep(K.level - COAST_RAMP, K.level, C);
}

/* ------------------------------------------------------------------ */
/* 2. The drainage graph                                               */
/* ------------------------------------------------------------------ */

const NODE = HYDRO.node;
const JIT = HYDRO.jitter;
const DI = [-1, 0, 1, -1, 1, -1, 0, 1];
const DJ = [-1, -1, -1, 0, 0, 1, 1, 1];
/** Node codes: 0–7 drains to that neighbour; SINK holds a lake; OCEAN is the sea. */
export const SINK = 8, OCEAN = 9;

export type HydroNode = {
  i: number; j: number;
  x: number; z: number;
  U: number;
  code: number;     // -1: not yet known
  acc: number;      // 0: not yet known
  W: number;        // NaN: not yet known
  I: number;        // incision (m)
  prof: Float64Array | null; // the reach's levels at t = 1/16 … 15/16 (along its meandering channel)
  prof0: Float64Array | null; // …following the land alone
  prof1: Float64Array | null; // …then below the valley it flows into
  L: number;        // a sink's lake level (NaN: not yet known; -Infinity: none)
  lr: number;       // the radius of the lake the river widens into here (m; 0: none; NaN: not yet known)
  lw: number;       // …its level (m): the river's where it leaves the lake (NaN: not yet known)
  fedIn: number;    // a river flowing in shows (1) or none does (0); -1: not yet known
  fed: number;      // this node's river shows: it leaves a lake here or upstream (1) or not (0); -1: not yet known
};

let cacheSeed = NaN;
const cache = new Map<number, HydroNode>();
const key = (i: number, j: number) => (i + 0x100000) * 0x200000 + (j + 0x100000);

/** A node's position (m, warped plane): its cell jittered by a hash. */
function nodeX(i: number, j: number, seed: number): number {
  return (i + 0.5 * (1 - JIT) + JIT * (hash3(i, j, seed + 3201) & 0xfff) * INV_4095) * NODE;
}
function nodeZ(i: number, j: number, seed: number): number {
  return (j + 0.5 * (1 - JIT) + JIT * ((hash3(i, j, seed + 3201) >>> 12) & 0xfff) * INV_4095) * NODE;
}

/** The node of cell (i, j): its position and uplift (the rest is filled in on demand). */
export function nodeAt(i: number, j: number, seed: number): HydroNode {
  if (seed !== cacheSeed || cache.size > 400000) { cache.clear(); cacheSeed = seed; }
  const k = key(i, j);
  let n = cache.get(k);
  if (n) return n;
  const x = nodeX(i, j, seed), z = nodeZ(i, j, seed);
  const U = upliftAt(x, z, seed);
  n = { i, j, x, z, U, code: UP.C < LANDS.coast.level ? OCEAN : -1, acc: 0, W: NaN, I: NaN, prof: null, prof0: null, prof1: null, L: NaN, lr: NaN, lw: NaN, fedIn: -1, fed: -1 };
  cache.set(k, n);
  return n;
}

/**
 * May a reach run from node (i, j) along the diagonal (di, dj)? Each 2 × 2 block of nodes allows one
 * of its diagonals, the shorter, so no two reaches ever cross.
 */
function diagonalOpen(i: number, j: number, di: number, dj: number, seed: number): boolean {
  const bi = Math.min(i, i + di), bj = Math.min(j, j + dj);
  const main = Math.hypot(nodeX(bi + 1, bj + 1, seed) - nodeX(bi, bj, seed), nodeZ(bi + 1, bj + 1, seed) - nodeZ(bi, bj, seed));
  const anti = Math.hypot(nodeX(bi, bj + 1, seed) - nodeX(bi + 1, bj, seed), nodeZ(bi, bj + 1, seed) - nodeZ(bi + 1, bj, seed));
  return (di === dj) === (main <= anti);
}

/** Where node n drains: a neighbour 0–7, SINK or OCEAN. */
export function codeOf(n: HydroNode, seed: number): number {
  if (n.code >= 0) return n.code;
  // The steepest descent to a lower neighbour whose way there crosses no ridge (the land along the
  // line never rises more than HYDRO.barrier over this node); failing that, the lowest crossing
  // (a gorge) if it is under HYDRO.gorge; failing that, a sink. (Diagonals only where their block
  // allows them: reaches never cross.)
  let best = 0, code = SINK, gorge = Infinity, gcode = SINK;
  for (let k = 0; k < 8; k++) {
    if (DI[k] !== 0 && DJ[k] !== 0 && !diagonalOpen(n.i, n.j, DI[k], DJ[k], seed)) continue;
    const m = nodeAt(n.i + DI[k], n.j + DJ[k], seed);
    if (m.U >= n.U) continue;
    const s = (m.U - n.U) / Math.hypot(m.x - n.x, m.z - n.z);
    let bar = -Infinity;
    if (m.code !== OCEAN) {
      for (let q = 1; q < 8; q++) {
        const t = q / 8;
        const u = upliftAt(n.x + (m.x - n.x) * t, n.z + (m.z - n.z) * t, seed) - n.U;
        if (u > bar) bar = u;
      }
    }
    if (bar <= HYDRO.barrier) { if (s < best) { best = s; code = k; } }
    else if (bar < gorge) { gorge = bar; gcode = k; }
  }
  if (code === SINK && gorge < HYDRO.gorge) code = gcode;
  n.code = code;
  return code;
}

/** The node n drains into (null: a sink or the sea). */
export function downOf(n: HydroNode, seed: number): HydroNode | null {
  const c = codeOf(n, seed);
  return c < 8 ? nodeAt(n.i + DI[c], n.j + DJ[c], seed) : null;
}

/** Drainage: the number of nodes whose water passes through n (itself included). */
export function accOf(n: HydroNode, seed: number): number {
  if (n.acc > 0) return n.acc;
  // iterative post-order over the upstream tree
  const stack: HydroNode[] = [n];
  while (stack.length) {
    const top = stack[stack.length - 1];
    if (top.acc > 0) { stack.pop(); continue; }
    let pending = false, sum = 1;
    if (codeOf(top, seed) !== OCEAN) {
      for (let k = 0; k < 8; k++) {
        const m = nodeAt(top.i + DI[k], top.j + DJ[k], seed);
        if (codeOf(m, seed) !== 7 - k) continue; // m drains into top (the opposite neighbour)
        if (m.acc > 0) sum += m.acc;
        else { stack.push(m); pending = true; }
      }
    }
    if (!pending) { top.acc = sum; stack.pop(); }
  }
  return n.acc;
}

/** The river's level at node n (m). */
function incisionOf(n: HydroNode, seed: number): number {
  if (!Number.isNaN(n.I)) return n.I;
  if (codeOf(n, seed) === OCEAN) { n.I = 0; return 0; }
  // (none at a source: its valley starts at the land's own level, or every source would sink a pit)
  const acc = accOf(n, seed);
  n.I = acc <= 1 ? 0 : HYDRO.incise[0] + HYDRO.incise[1] * Math.log2(acc) + HYDRO.incise[2] * Math.max(0, n.U - LANDS.sea);
  return n.I;
}

/** A channel's half width where the drainage is acc (m; acc ≥ 1, may be fractional along a reach). */
function riverHalfW(acc: number): number {
  return 0.5 * Math.min(HYDRO.widthMax, HYDRO.width[0] + HYDRO.width[1] * Math.pow(Math.max(acc - 1, 0), HYDRO.width[2]));
}
/** The widest half width of the reach from a node of drainage acc (m: at its end, the width noise at its top). */
function channelW(acc: number): number {
  return (1 + HYDRO.widthVary) * riverHalfW(acc + 1);
}
/** Where m's meandering channel to n runs at t (warped plane). */
function channelX(m: HydroNode, n: HydroNode, t: number, seed: number): number {
  const dx = n.x - m.x, dz = n.z - m.z, len = Math.hypot(dx, dz);
  return m.x + dx * t - dz / len * meanderAt(t * len, t, meanderAmp((m.U - n.U) / len), hash3(m.i, m.j, seed + 3301) & 0xffff);
}
function channelZ(m: HydroNode, n: HydroNode, t: number, seed: number): number {
  const dx = n.x - m.x, dz = n.z - m.z, len = Math.hypot(dx, dz);
  return m.z + dz * t + dx / len * meanderAt(t * len, t, meanderAmp((m.U - n.U) / len), hash3(m.i, m.j, seed + 3301) & 0xffff);
}
/** The valley floor's half width of m's reach to n, at its narrowest (m). */
function reachFloor(m: HydroNode, n: HydroNode, seed: number): number {
  const acc = accOf(m, seed);
  return meanderAmp((m.U - n.U) / Math.hypot(n.x - m.x, n.z - m.z)) + (1 - HYDRO.widthVary) * riverHalfW(acc) + HYDRO.floor[0] + HYDRO.floor[1] * Math.sqrt(acc);
}

/** A valley floor's half width round a node of drainage acc, where its meanders die out (m). */
function nodeFloor(acc: number): number {
  return channelW(acc) + HYDRO.floor[0] + HYDRO.floor[1] * Math.sqrt(acc);
}

/**
 * The lowest land (the uplift) across the floor of m's reach to n at t: under its meandering channel
 * and at the floor's two edges. A river runs at the level of the lowest land of its floor, or where
 * the land falls away to one side it would stand on an embankment.
 */
function corridorU(m: HydroNode, n: HydroNode, t: number, seed: number): number {
  const dx = n.x - m.x, dz = n.z - m.z, len = Math.hypot(dx, dz);
  const ux = dx / len, uz = dz / len;
  const amp = meanderAmp((m.U - n.U) / len);
  const o = meanderAt(t * len, t, amp, hash3(m.i, m.j, seed + 3301) & 0xffff);
  const half = amp + nodeFloor(accOf(m, seed));
  const cx = m.x + dx * t, cz = m.z + dz * t;
  return Math.min(
    upliftAt(cx - uz * o, cz + ux * o, seed),
    upliftAt(cx - uz * half, cz + ux * half, seed),
    upliftAt(cx + uz * half, cz - ux * half, seed));
}

/** The lowest land round node n (the uplift at it and out across its floor). */
function nodeLowU(n: HydroNode, seed: number): number {
  const r = nodeFloor(accOf(n, seed));
  return Math.min(n.U, upliftAt(n.x + r, n.z, seed), upliftAt(n.x - r, n.z, seed), upliftAt(n.x, n.z + r, seed), upliftAt(n.x, n.z - r, seed));
}

/** The lowest the land across m's reach to n falls (less the incision there), between them. */
function pathLow(m: HydroNode, n: HydroNode, seed: number): number {
  const Im = incisionOf(m, seed), In = incisionOf(n, seed);
  let lo = Infinity;
  for (let k = 1; k < PROF_N; k++) {
    const t = k / PROF_N;
    const v = corridorU(m, n, t, seed) - (Im + (In - Im) * t);
    if (v < lo) lo = v;
  }
  return lo;
}

/**
 * The river's level at node n (m): the uplift less the incision, but never above the lowest point of
 * any reach flowing in (or the river would float over the dip) nor above the nodes upstream, nor below the sea.
 */
export function levelOf(n: HydroNode, seed: number): number {
  if (!Number.isNaN(n.W)) return n.W;
  if (codeOf(n, seed) === OCEAN) { n.I = 0; n.W = LANDS.sea; return n.W; }
  // iterative post-order over the upstream tree (like accOf)
  const stack: HydroNode[] = [n];
  while (stack.length) {
    const top = stack[stack.length - 1];
    if (!Number.isNaN(top.W)) { stack.pop(); continue; }
    let pending = false;
    let W = nodeLowU(top, seed) - incisionOf(top, seed);
    for (let k = 0; k < 8; k++) {
      const m = nodeAt(top.i + DI[k], top.j + DJ[k], seed);
      if (codeOf(m, seed) !== 7 - k) continue; // m drains into top
      if (Number.isNaN(m.W)) { stack.push(m); pending = true; continue; }
      W = Math.min(W, pathLow(m, top, seed), m.W - 0.05);
    }
    // (never below the sea: a river near the coast runs at its level to the mouth)
    if (!pending) { top.W = Math.max(W, LANDS.sea); stack.pop(); }
  }
  return n.W;
}

/** Samples of a reach's profile: its levels at t = k / PROF_N (k = 1 … PROF_N − 1). */
export const PROF_N = 16;

/** The meander amplitude of a reach whose land falls `slope` per metre (m): rivers wind on the flat, run straight down slopes. */
export function meanderAmp(slope: number): number {
  return HYDRO.meanderMin + (HYDRO.meander - HYDRO.meanderMin) * (1 - smoothstep(0.012, 0.05, slope));
}

/**
 * The levels of n's reach at t = 1/16 … 15/16 (a sink or the sea has none): the lowest uplift across its
 * meandering channel less the incision, a running minimum held at or above the lower end.
 */
export function profileOf(n: HydroNode, seed: number): Float64Array | null {
  if (n.prof) return n.prof;
  const p1 = profile1(n, seed);
  if (!p1) return null;
  // Then below the valleys of the other reaches flowing into the same node (at their own levels before
  // this rule: whichever of two runs higher comes down to the other's valley).
  const d = downOf(n, seed)!;
  const p = Float64Array.from(p1);
  const Wb = levelOf(d, seed);
  const D = 2 * channelW(accOf(n, seed)) + 1;
  for (let q = 0; q < 8; q++) {
    const sb = nodeAt(d.i + DI[q], d.j + DJ[q], seed);
    if (sb === n || codeOf(sb, seed) !== 7 - q) continue;
    belowValley(n, d, p, sb, d, profile1(sb, seed)!, levelOf(sb, seed), Wb, reachFloor(sb, d, seed), D, seed);
  }
  n.prof = p;
  return p;
}

/**
 * Hold n's reach to d (profile p, in place) below the valley of the reach from o to e (its profile po,
 * levels Wo … We, floor fl) out to this channel's far bank D, walls of the gentlest slope; never under
 * d's level, and falling all the way still.
 */
function belowValley(n: HydroNode, d: HydroNode, p: Float64Array, o: HydroNode, e: HydroNode, po: Float64Array, Wo: number, We: number,
  fl: number, D: number, seed: number): void {
  const Wb = levelOf(d, seed);
  const ex = e.x - o.x, ez = e.z - o.z, L2 = ex * ex + ez * ez, Lo = Math.sqrt(L2);
  for (let k = 1; k < PROF_N; k++) {
    const t = k / PROF_N;
    const cx = channelX(n, d, t, seed), cz = channelZ(n, d, t, seed);
    const tl = Math.min(Math.max(((cx - o.x) * ex + (cz - o.z) * ez) / L2, 0), 1);
    const x = Math.hypot(cx - o.x - ex * tl, cz - o.z - ez * tl) - fl;
    const lv = reachLevel(Wo, po, We, tl, false);
    const fall = reachFall(Wo, po, We, tl, false) / Lo;
    const h = x > 0 ? wallRise(K_MIN, x, fl) : 0;
    const sl = (x > 0 ? wallSlope(K_MIN, x, fl) : 0.03) + fall + HYDRO.wallCurve * D;
    p[k - 1] = Math.max(Wb, Math.min(p[k - 1], lv + h - sl * D - 0.2));
  }
  for (let k = 1; k < PROF_N - 1; k++) p[k] = Math.min(p[k], p[k - 1]);
}

/**
 * n's profile below the valley it flows into: the reach leaving its downstream node, at that reach's
 * levels as the land alone sets them (profileOf's first stage).
 */
function profile1(n: HydroNode, seed: number): Float64Array | null {
  if (n.prof1) return n.prof1;
  const p0 = profile0(n, seed);
  if (!p0) return null;
  const d = downOf(n, seed)!;
  const p = Float64Array.from(p0);
  // Toward the node it joins, a reach runs below the valley of the reach leaving that node: a tributary
  // never hangs over the valley it flows into, it comes down into it.
  const out = downOf(d, seed);
  if (out) belowValley(n, d, p, d, out, profile0(d, seed)!, levelOf(d, seed), levelOf(out, seed), reachFloor(d, out, seed), 2 * channelW(accOf(n, seed)) + 1, seed);
  n.prof1 = p;
  return p;
}

/** n's profile as the land alone sets it: the lowest land across its floor, falling all the way. */
function profile0(n: HydroNode, seed: number): Float64Array | null {
  if (n.prof0) return n.prof0;
  const d = downOf(n, seed);
  if (!d) return null;
  const Wa = levelOf(n, seed), Wb = levelOf(d, seed);
  const Ia = incisionOf(n, seed), Ib = incisionOf(d, seed);
  const p = new Float64Array(PROF_N - 1);
  let run = Wa;
  for (let k = 1; k < PROF_N; k++) {
    const t = k / PROF_N;
    const s = corridorU(n, d, t, seed) - (Ia + (Ib - Ia) * t);
    if (s < run) run = s;
    p[k - 1] = Math.max(run, Wb);
  }
  // (no fall in the first or last sixteenth: round a node every reach meeting there runs at its level,
  // or one's pool would stand over another's floor)
  p[0] = Math.max(p[0], Wa - HYDRO.nodePool);
  p[PROF_N - 2] = Math.min(p[PROF_N - 2], Wb + HYDRO.nodePool);
  n.prof0 = p;
  return p;
}

/** The level of n's reach at t (0 at n, 1 at its downstream node): linear in each sixteenth, or a fall. */
function reachLevel(Wa: number, p: Float64Array | ArrayLike<number>, Wb: number, t: number, falls = true): number {
  const s = Math.min(Math.max(t, 0), 1) * PROF_N;
  const k = Math.min(Math.floor(s), PROF_N - 1);
  const u = s - k;
  const a = k === 0 ? Wa : p[k - 1];
  const b = k === PROF_N - 1 ? Wb : p[k];
  const D = a - b;
  // (a fall marks a concentrated drop: a reach steep all along runs as a chute of rapids instead)
  if (!falls || D <= Math.max(HYDRO.fallDrop, 3 * (Wa - Wb) / PROF_N)) return a + (b - a) * u;
  // a waterfall of at most HYDRO.maxFall, the rest of the drop as cascades along the eighth
  const fallH = Math.min(D, HYDRO.maxFall);
  return a - fallH * smoothstep(0.46, 0.54, u) - (D - fallH) * u;
}

/** How fast the level of n's reach falls at t (m per unit t, ≥ 0): reachLevel's derivative. */
function reachFall(Wa: number, p: Float64Array | ArrayLike<number>, Wb: number, t: number, falls: boolean): number {
  const s = Math.min(Math.max(t, 0), 1) * PROF_N;
  const k = Math.min(Math.floor(s), PROF_N - 1);
  const u = s - k;
  const a = k === 0 ? Wa : p[k - 1];
  const b = k === PROF_N - 1 ? Wb : p[k];
  const D = a - b;
  if (!falls || D <= Math.max(HYDRO.fallDrop, 3 * (Wa - Wb) / PROF_N)) return D * PROF_N;
  const fallH = Math.min(D, HYDRO.maxFall);
  const q = Math.min(Math.max((u - 0.46) / 0.08, 0), 1);
  return (fallH * 6 * q * (1 - q) / 0.08 + D - fallH) * PROF_N;
}

/* ------------------------------------------------------------------ */
/* 3. The field: valleys, the nearest channel, lakes                   */
/* ------------------------------------------------------------------ */

/** What riverField found (module scratch, zero allocation). */
export const RF = {
  /** The lowest valley surface over all reaches (m; +Infinity: none). */
  R: Infinity,
  /** The highest river terrace (m; −Infinity: none): a reach's floor at its level, then a riser down to the land. */
  F: -Infinity,
  /** Distance (m) past the nearest valley floor's edge (negative inside a floor). */
  xFloor: Infinity,
  /** The nearest channel: its level, distance from its centre line, half width, depth, drainage, the reach's U at its head. */
  W: -Infinity, ch: Infinity, w: 0, depth: 0, acc: 0, Ua: 0,
  /** …its flow: direction (unit, warped plane) and the water surface's fall per metre. */
  fx: 0, fz: 0, fall: 0,
  /** Lakes: the level of the lake whose ring the point is in (-Infinity: none), how far inside (1 deep in, 0 at the ring), a mask for detail. */
  L: -Infinity, lakeK: 0, lakeMask: 0,
  /** …the level of a lake whose shore the point is on or in (out a third past its edge; -Infinity: none). */
  Lnear: -Infinity,
  /** The nearest channel's water shows (1: its river flows from a lake; 0: a dry valley). */
  vis: 0,
  /**
   * The highest the nearest channel's water may stand (m): under every other reach's surface (its
   * ground, or its water in its channel) out to this channel's far bank, so the water never stands
   * over a bank another valley has cut lower.
   */
  cap: Infinity,
  /**
   * The valley walls' slope at the point (warped plane): the tensor g gᵀ of each reach's gradient
   * across its valley, averaged with weights falling off by e per HYDRO.wallSoft its surface stands
   * over the lowest, so it changes smoothly where two valleys' walls meet. A wall's slope seen through a
   * map J of the plane is √(gᵀ J Jᵀ g).
   */
  gxx: 0, gxz: 0, gzz: 0,
};

/** The gentlest valley walls of any region: a lake's ring is judged on them, so no region's valleys can breach it. */
const K_MIN = Math.min(...Object.values(LANDFORM).map((l) => l.k));

/** Each reach's surface at the point (ground, or water in its visible channel) and its slope (scratch). */
const S_AT = new Float64Array(34), S_LAT = new Float64Array(34), S_ALG = new Float64Array(34), S_CURV = new Float64Array(34);
/** …the direction it rises across (unit), and how loosely that holds (meanders bend a floor's slopes). */
const S_NX = new Float64Array(34), S_NZ = new Float64Array(34), S_LOOSE = new Float64Array(34);
/** …and the lowest it can fall to: its own river's level there (no valley sinks below its river). */
const S_WLO = new Float64Array(34);
/** …and its reach's two nodes (cell keys; lakes: none), so reaches meeting at a node know each other. */
const S_NA = new Float64Array(34), S_NB = new Float64Array(34);
/** The river lakes round the point (scratch): distance, lobed radius, level, depth, the way out from the centre. */
const LK_X = new Float64Array(9), LK_R = new Float64Array(9), LK_W = new Float64Array(9), LK_D = new Float64Array(9), LK_NX = new Float64Array(9), LK_NZ = new Float64Array(9);

/** The meander's offset (m) from a reach's line at `along` metres from its head, t its share of the reach. */
function meanderAt(along: number, t: number, amp: number, segSeed: number): number {
  if (amp <= 0) return 0;
  const taper = smoothstep(0, 0.12, t) * smoothstep(1, 0.88, t);
  return amp * taper * vnoise(along * F.meander, 0.5, segSeed);
}

/** Valley shape above the reach's level at distance dd from its line and ch from its channel (m). */
/**
 * A valley wall's rise x metres out from its floor's rim (floor half width fl), walls of slope k: it leaves
 * the rim at the floor's own slope and steepens to k over about HYDRO.wallRound, then by HYDRO.wallCurve.
 */
function wallRise(k: number, x: number, fl: number): number {
  const r = HYDRO.wallRound, s0 = 2 * (HYDRO.floorRise - 0.3) / fl;
  return HYDRO.floorRise + s0 * x + (k - s0) * (x - r * (1 - Math.exp(-x / r))) + HYDRO.wallCurve * x * x;
}
/** …and its slope there (m per m). */
function wallSlope(k: number, x: number, fl: number): number {
  const s0 = 2 * (HYDRO.floorRise - 0.3) / fl;
  return s0 + (k - s0) * (1 - Math.exp(-x / HYDRO.wallRound)) + 2 * HYDRO.wallCurve * x;
}

function valleyShape(dd: number, ch: number, w: number, depth: number, floor: number, k: number): number {
  if (ch < w) {
    const q = ch / w;
    return -depth * (1 - q * q);
  }
  const fr = HYDRO.floorRise;
  const base = dd < floor ? 0.3 + (fr - 0.3) * (dd / floor) * (dd / floor) : wallRise(k, dd - floor, floor);
  return base * smoothstep(w, w + 5, ch);
}

/** How steeply valleyShape rises across the valley there (m per m, a bound). */
function valleySlope(dd: number, ch: number, w: number, depth: number, floor: number, k: number): number {
  if (ch < w) return 2 * depth * ch / (w * w);
  const fr = HYDRO.floorRise;
  const x = dd - floor;
  const base = dd < floor ? 0.3 + (fr - 0.3) * (dd / floor) * (dd / floor) : wallRise(k, x, floor);
  const db = dd < floor ? 2 * (fr - 0.3) * dd / (floor * floor) : wallSlope(k, x, floor);
  const q = Math.min(Math.max((ch - w) / 5, 0), 1);
  return db * q * q * (3 - 2 * q) + base * 6 * q * (1 - q) / 5;
}

/**
 * The rivers and lakes at warped (wx, wz): the valleys of every reach whose head lies in the 5 × 5
 * node cells round the point (into RF), with valley walls of slope k; a river shows only where it
 * flows from a lake (fedOf), and one that does not cuts no channel. `lakes`: also the lakes.
 */
export function riverField(wx: number, wz: number, seed: number, k: number, lakes: boolean): void {
  const ci = Math.floor(wx / NODE), cj = Math.floor(wz / NODE);
  // the lakes first: a lake's level is found with riverField itself (which overwrites RF)
  let L = -Infinity, lakeK = 0, lakeMask = 0, Lnear = -Infinity;
  if (lakes) {
    const rL = HYDRO.lakeR * NODE;
    for (let j = cj - 1; j <= cj + 1; j++) {
      for (let i = ci - 1; i <= ci + 1; i++) {
        const s = nodeAt(i, j, seed);
        if (codeOf(s, seed) !== SINK) continue;
        const lv = lakeOf(s, seed);
        if (lv === -Infinity) continue;
        const d = Math.hypot(wx - s.x, wz - s.z) / rL;
        const m = 1 - smoothstep(1.0, 1.2, d); // no detail inside the ring: its ground is the ring's own
        if (m > lakeMask) lakeMask = m;
        if (d < 1 && lv > L) { L = lv; lakeK = 1 - smoothstep(0.85, 1, d); }
        if (d < 1.35 && lv > Lnear) Lnear = lv;
      }
    }
  }
  // lakes on rivers: the river widens into a lake round some nodes (their bowls are cut below)
  let nLk = 0;
  for (let j = cj - 1; j <= cj + 1; j++) {
    for (let i = ci - 1; i <= ci + 1; i++) {
      const s = nodeAt(i, j, seed);
      const r = riverLakeOf(s, seed);
      if (r <= 0) continue;
      const dx = wx - s.x, dz = wz - s.z, dist = Math.hypot(dx, dz);
      if (dist > 2.5 * LAKE_RMAX) continue;
      const rl = lakeLobe(r, dx, dz, seed), lv = riverLakeLevel(s, seed);
      LK_X[nLk] = dist; LK_R[nLk] = rl; LK_W[nLk] = lv; LK_D[nLk] = HYDRO.riverLakeD[0] + HYDRO.riverLakeD[1] * r;
      LK_NX[nLk] = dx / Math.max(dist, 1e-6); LK_NZ[nLk++] = dz / Math.max(dist, 1e-6);
      if (!lakes) continue;
      const d = dist / rl;
      const m = 1 - smoothstep(1.0, 1.35, d);
      if (m > lakeMask) lakeMask = m;
      if (d < 1 && lv > L) { L = lv; lakeK = 1 - smoothstep(0.85, 1, d); }
      if (d < 1.35 && lv > Lnear) Lnear = lv;
    }
  }
  RF.R = Infinity; RF.F = -Infinity; RF.xFloor = Infinity; RF.W = -Infinity; RF.ch = Infinity; RF.w = 0; RF.depth = 0; RF.acc = 0; RF.Ua = 0;
  RF.fx = 0; RF.fz = 0; RF.fall = 0; RF.L = L; RF.Lnear = Lnear; RF.lakeK = lakeK; RF.lakeMask = lakeMask; RF.vis = 0; RF.cap = Infinity;
  let bestCh = Infinity, best = -1, nR = 0;
  for (let j = cj - 2; j <= cj + 2; j++) {
    for (let i = ci - 2; i <= ci + 2; i++) {
      const a = nodeAt(i, j, seed);
      const b = downOf(a, seed);
      if (!b) continue;
      const Wa = levelOf(a, seed), Wb = levelOf(b, seed);
      const p = profileOf(a, seed)!;
      const acc = accOf(a, seed);
      const dx = b.x - a.x, dz = b.z - a.z;
      const len = Math.hypot(dx, dz), ux = dx / len, uz = dz / len;
      const rx = wx - a.x, rz = wz - a.z;
      const along = rx * ux + rz * uz;
      const t = along / len;
      // (the river widens as it goes: its head's drainage plus the share of the reach run)
      const w0 = riverHalfW(acc + Math.min(Math.max(t, 0), 1));
      const w = w0 * (1 + HYDRO.widthVary * vnoise(along * F.meander * 2.3, 7.5, hash3(a.i, a.j, seed + 3302) & 0xffff));
      const vis = fedOf(a, seed) ? 1 : 0;
      // (a river whose water does not show cuts no channel: a dry wash, not a ditch)
      const depth = (HYDRO.depth[0] + HYDRO.depth[1] * Math.sqrt(acc)) * vis;
      const slope = (Wa - Wb) / len;
      const amp = meanderAmp((a.U - b.U) / len);
      // (a dry valley has no floodplain for a river to wander: a narrow floor, its walls a V)
      const floor = (vis > 0 ? amp + w : 0) + HYDRO.floor[0] + HYDRO.floor[1] * Math.sqrt(acc);
      let dd: number, ch: number;
      if (t <= 0) { dd = Math.hypot(rx, rz); ch = dd; }
      else if (t >= 1) { dd = Math.hypot(wx - b.x, wz - b.z); ch = dd; }
      else {
        const ds = -rx * uz + rz * ux;
        dd = Math.abs(ds);
        if (dd - floor > 400) { // far outside this valley's floor: only its wall can matter
          const x = dd - floor;
          const lv = reachLevel(Wa, p, Wb, t, false);
          const cone = lv + wallRise(k, x, floor);
          if (cone < RF.R) RF.R = cone;
          const fill = lv + HYDRO.floorRise - HYDRO.riser * x;
          if (fill > RF.F) RF.F = fill;
          if (dd - floor < RF.xFloor) RF.xFloor = dd - floor;
          S_AT[nR] = cone; S_LAT[nR] = wallSlope(k, x, floor); S_ALG[nR] = reachFall(Wa, p, Wb, t, false) / len;
          S_CURV[nR] = HYDRO.wallCurve; S_NX[nR] = -uz; S_NZ[nR] = ux; S_LOOSE[nR] = 0.1; S_WLO[nR] = lv;
          S_NA[nR] = key(a.i, a.j); S_NB[nR++] = key(b.i, b.j);
          continue;
        }
        ch = Math.abs(ds - meanderAt(along, t, amp, (hash3(a.i, a.j, seed + 3301) & 0xffff)));
      }
      // (a waterfall steps the channel and its floor; the walls follow the reach's smooth fall, or
      // every fall would draw a scarp across the land)
      const W = reachLevel(Wa, p, Wb, t);
      // (and a dry valley no falls: with no water to fall, a step in its floor would be a scarp)
      const Wv = vis > 0 ? W + (reachLevel(Wa, p, Wb, t, false) - W) * smoothstep(floor, floor + 60, dd) : reachLevel(Wa, p, Wb, t, false);
      // (a dry valley has no channel, only its floor's bowl and its walls: a channel's width with no depth
      // would stand a riser at its edge wherever the channel reaches past the narrow floor)
      const chV = vis > 0 ? ch : Infinity, wV = vis > 0 ? w : 0;
      const cone = Wv + valleyShape(dd, chV, wV, depth, floor, k);
      if (cone < RF.R) RF.R = cone;
      // the terrace: the floor at the river's level (where the land falls away beside a reach), then a riser
      const fill = Wv + (dd < floor ? 0.3 + (HYDRO.floorRise - 0.3) * (dd / floor) * (dd / floor) : HYDRO.floorRise - HYDRO.riser * (dd - floor));
      if (fill > RF.F) RF.F = fill;
      if (dd - floor < RF.xFloor) RF.xFloor = dd - floor;
      // this reach's surface (its water in its channel, else its ground), how fast it changes across its
      // valley and along it, and which way across is (round a node: outward from it)
      const end = t <= 0 || t >= 1;
      const fallT = end ? 0 : Math.max(reachFall(Wa, p, Wb, t, true), reachFall(Wa, p, Wb, t, false)) / len;
      let nx = -uz, nz = ux;
      if (end) { const ox = t <= 0 ? rx : wx - b.x, oz = t <= 0 ? rz : wz - b.z; nx = ox / Math.max(dd, 1e-6); nz = oz / Math.max(dd, 1e-6); }
      S_NX[nR] = nx; S_NZ[nR] = nz; S_WLO[nR] = Math.min(W, Wv); S_NA[nR] = key(a.i, a.j); S_NB[nR] = key(b.i, b.j);
      if (vis > 0 && ch < w + 0.3) { S_AT[nR] = Math.max(cone, W); S_LAT[nR] = 0; S_ALG[nR] = fallT; S_CURV[nR] = 0; S_LOOSE[nR] = 0.1; }
      else {
        S_AT[nR] = cone; S_LAT[nR] = valleySlope(dd, chV, wV, depth, floor, k); S_ALG[nR] = fallT; S_CURV[nR] = HYDRO.wallCurve;
        S_LOOSE[nR] = dd < floor + 60 ? 0.35 : 0.1;
      }
      // (the nearest channel that shows: a dry valley beside a river never takes its water away)
      if (vis > 0 && ch - w < bestCh) {
        bestCh = ch - w; best = nR;
        RF.W = W; RF.ch = ch; RF.w = w; RF.depth = depth; RF.acc = acc; RF.Ua = a.U; RF.vis = vis;
        RF.fx = ux; RF.fz = uz; RF.fall = slope;
        // across this channel: across its meander's own heading (round a node: outward)
        if (end) { BN[0] = nx; BN[1] = nz; }
        else {
          const sg = hash3(a.i, a.j, seed + 3301) & 0xffff;
          const dm = 0.5 * (meanderAt(along + 1, t + 1 / len, amp, sg) - meanderAt(along - 1, t - 1 / len, amp, sg));
          const hx = ux - uz * dm, hz = uz + ux * dm, hl = Math.hypot(hx, hz);
          BN[0] = -hz / hl; BN[1] = hx / hl;
        }
      }
      nR++;
    }
  }
  // the river lakes' bowls: below the water, a shore rising gently, then walls (they are valleys too)
  for (let q = 0; q < nLk; q++) {
    const dist = LK_X[q], rl = LK_R[q], fl = rl + HYDRO.floor[0] + 20;
    const cone = LK_W[q] + valleyShape(dist, dist, rl, LK_D[q], fl, k);
    if (cone < RF.R) RF.R = cone;
    if (dist - fl < RF.xFloor) RF.xFloor = dist - fl;
    S_AT[nR] = Math.max(cone, dist < rl ? LK_W[q] : -Infinity);
    S_LAT[nR] = dist < rl ? 0 : valleySlope(dist, dist, rl, LK_D[q], fl, k); S_ALG[nR] = 0; S_CURV[nR] = HYDRO.wallCurve;
    S_NX[nR] = LK_NX[q]; S_NZ[nR] = LK_NZ[q]; S_LOOSE[nR] = 0.3; S_WLO[nR] = LK_W[q]; S_NA[nR] = -1; S_NB[nR++] = -2;
  }
  // the walls' slope: each reach's gradient across, weighted by a soft minimum over the surfaces
  let gw = 0, gxx = 0, gxz = 0, gzz = 0;
  for (let r = 0; r < nR; r++) {
    const d = S_AT[r] - RF.R;
    if (d > 10 * HYDRO.wallSoft) continue;
    const wt = Math.exp(-d / HYDRO.wallSoft), g2 = wt * S_LAT[r] * S_LAT[r];
    gw += wt; gxx += g2 * S_NX[r] * S_NX[r]; gxz += g2 * S_NX[r] * S_NZ[r]; gzz += g2 * S_NZ[r] * S_NZ[r];
  }
  if (gw > 0) { RF.gxx = gxx / gw; RF.gxz = gxz / gw; RF.gzz = gzz / gw; } else { RF.gxx = RF.gxz = RF.gzz = 0; }
  // the nearest channel's water stays under every other reach's surface out to its far bank (that
  // surface falls toward the bank by its slope along the line across the channel)
  if (best >= 0) {
    const D = RF.w + RF.ch + 1;
    let cap = Infinity;
    for (let r = 0; r < nR; r++) {
      if (r === best) continue;
      const cn = Math.abs(BN[0] * S_NX[r] + BN[1] * S_NZ[r]), cu = Math.abs(BN[0] * S_NZ[r] - BN[1] * S_NX[r]);
      const fall = (cn * S_LAT[r] + cu * S_ALG[r] + S_LOOSE[r] * (S_LAT[r] + S_ALG[r]) + S_CURV[r] * D) * D;
      // (falling by its slope toward the far bank, but never below its own river less that river's fall
      // there; reaches meeting this one at a node run on into it, level with it where they meet)
      const kin = S_NA[r] === S_NA[best] || S_NA[r] === S_NB[best] || S_NB[r] === S_NA[best] || S_NB[r] === S_NB[best];
      const c = Math.max(S_AT[r] - fall, S_WLO[r] - ((kin ? 0 : cu) + 0.05) * S_ALG[r] * D);
      if (c < cap) cap = c;
    }
    RF.cap = cap - 0.1;
  }
}
const BN = [0, 0];

/** The largest lake on a river (m): the most a lake's shore can reach from its node. */
const LAKE_RMAX = Math.max(HYDRO.riverLakeR[1], HYDRO.reservoirR[1]);

/**
 * The least drainage (nodes) a river needs to show near warped (wx, wz): the regions' LANDFORM.riverMin
 * as their climate decides them (hot and arid: the desert's; hot and half-dry: the savanna's; else 1).
 * Only a lake's place is judged by it (on the CPU, once per node), so the climate fields suffice.
 */
function riverMinNear(wx: number, wz: number, seed: number): number {
  const T0 = CLIMATE.tBase + CLIMATE.tAmp * (vnoise(wx * F.temp, wz * F.temp, seed + 1201) + CLIMATE.tFine * vnoise(wx * F.tempfine, wz * F.tempfine, seed + 1202))
    - CLIMATE.lambdaLat * Math.abs(wz);
  const S = LANDS.coast.level;
  const M = CLIMATE.mBase + CLIMATE.mNoiseAmp * (vnoise(wx * F.moistr, wz * F.moistr, seed + 1301) + CLIMATE.mFine * vnoise(wx * F.moistfine, wz * F.moistfine, seed + 1302))
    + CLIMATE.mOceanLambda * (1 - smoothstep(S, S + 0.12, continentAt(wx, wz, seed)));
  const hot = smoothstep(LANDS.hot[0], LANDS.hot[1], T0);
  const arid = 1 - smoothstep(LANDS.arid[0], LANDS.arid[1], M);
  const half = (1 - arid) * (1 - smoothstep(LANDS.hotDry[0], LANDS.hotDry[1], M));
  return 1 + hot * (arid * (LANDFORM.desert.riverMin - 1) + half * (LANDFORM.savanna.riverMin - 1));
}

/**
 * Where the lake round node n holds its level (m), if one of radius r fits there: the river's level
 * where it leaves the lake (past its farthest shore), within `drop` of the node's own (or the lake would
 * lie far under the land it is cut into); NaN if not.
 */
function lakeFits(n: HydroNode, d: HydroNode, r: number, drop: number, seed: number): number {
  const len = Math.hypot(d.x - n.x, d.z - n.z), t = Math.min(1, (1.3 * r + 25) / len);
  const p = profileOf(n, seed)!, Wa = levelOf(n, seed), Wb = levelOf(d, seed);
  const lw = Math.min(reachLevel(Wa, p, Wb, t), reachLevel(Wa, p, Wb, t, false));
  return Wa - lw <= drop ? lw : NaN;
}

/**
 * Does a lake of radius r round node n, at level lw, keep clear of every valley floor lower than it?
 * (Its own reach runs at or above its level as far as its far shore, and the reaches flowing into it
 * above it; any other floor lower than the lake within its lobed shore would leave its water standing
 * over that valley's ground.)
 */
function lakeClear(n: HydroNode, r: number, lw: number, seed: number): boolean {
  const reach = 1.35 * 1.28 * r + 30;
  for (let j = n.j - 2; j <= n.j + 2; j++) {
    for (let i = n.i - 2; i <= n.i + 2; i++) {
      const a = nodeAt(i, j, seed);
      const b = downOf(a, seed);
      if (!b || a === n || b === n) continue;
      const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz);
      const t = Math.min(Math.max(((n.x - a.x) * dx + (n.z - a.z) * dz) / (len * len), 0), 1);
      const acc = accOf(a, seed);
      const floor = meanderAmp((a.U - b.U) / len) + channelW(acc + t) + HYDRO.floor[0] + HYDRO.floor[1] * Math.sqrt(acc);
      if (Math.hypot(n.x - a.x - dx * t, n.z - a.z - dz * t) - floor > reach) continue;
      if (reachLevel(levelOf(a, seed), profileOf(a, seed)!, levelOf(b, seed), t, false) < lw + 0.3) return false;
    }
  }
  return true;
}

/**
 * Does the land just past the lobed shore of a lake of radius r round node n stand over its level lw
 * all round? (Its detail is masked there, so the uplift is its ground.) A lake held only by its outlet
 * would stand over any side where the land falls away.
 */
function lakeHeld(n: HydroNode, r: number, lw: number, seed: number): boolean {
  for (let k = 0; k < 24; k++) {
    const ux = Math.cos(k * Math.PI / 12), uz = Math.sin(k * Math.PI / 12);
    for (const f of [1, 1.28]) {
      const rl = Math.max(lakeLobe(r, ux * r * f, uz * r * f, seed), r * f) * 1.06 + 4;
      if (upliftAt(n.x + ux * rl, n.z + uz * rl, seed) < lw + 0.3) return false;
    }
  }
  return true;
}

/**
 * The radius (m) of the lake at node n (0: none), on flat land (the reach leaving n falls gently),
 * away from the coast:
 * - where no river flows in that shows (fedIn), a reservoir: at HYDRO.reservoirP of the nodes whose
 *   drainage lets a river show there, the largest lake that fits (HYDRO.reservoirR, shrinking until it
 *   does). A river shows only below one: none is ever seen to start from nothing.
 * - along a river that shows, a lake it widens into: at HYDRO.riverLakeP of the nodes, sized
 *   HYDRO.riverLakeR skewed small (TotK's lakes: most 100–300 m across, a few to ~600 m).
 */
export function riverLakeOf(n: HydroNode, seed: number): number {
  if (!Number.isNaN(n.lr)) return n.lr;
  // (whether a river flowing in shows: fedOf settles that, and this node's lake with it)
  if (n.fedIn < 0) { fedOf(n, seed); if (!Number.isNaN(n.lr)) return n.lr; }
  n.lr = 0;
  const d = downOf(n, seed);
  if (!d || !(n.U > LANDS.sea + 6)) return 0;
  // (a reservoir may sit on a gentler slope than a lake along a river: it must be held all round anyway)
  const slope = (n.U - d.U) / Math.hypot(d.x - n.x, d.z - n.z);
  if (slope >= (n.fedIn === 1 ? 0.035 : HYDRO.reservoirSlope)) return 0;
  const h = hash3(n.i, n.j, seed + 3501);
  const q = (h & 0xffff) * INV_65535;
  if (n.fedIn === 1) {
    if ((h >>> 16) * INV_65535 >= HYDRO.riverLakeP) return 0;
    const r = HYDRO.riverLakeR[0] + (HYDRO.riverLakeR[1] - HYDRO.riverLakeR[0]) * q * q;
    const lw = lakeFits(n, d, r, HYDRO.riverLakeDrop, seed);
    if (!Number.isNaN(lw) && lakeHeld(n, r, lw, seed) && lakeClear(n, r, lw, seed)) { n.lr = r; n.lw = lw; }
    return n.lr;
  }
  if ((h >>> 16) * INV_65535 >= HYDRO.reservoirP) return 0;
  // (the region must let a river show from here: deserts only from big drainage; high snowfields from less)
  if (accOf(n, seed) + Math.max(0, (n.U - 420) / 120) < riverMinNear(n.x, n.z, seed)) return 0;
  upliftAt(n.x, n.z, seed);
  if (VOL.w > 0.2) return 0; // (no reservoir on a volcano's slopes)
  for (let r = HYDRO.reservoirR[0] + (HYDRO.reservoirR[1] - HYDRO.reservoirR[0]) * q; r >= HYDRO.riverLakeR[0] * 1.8; r *= 0.75) {
    const lw = lakeFits(n, d, r, HYDRO.reservoirDrop, seed);
    if (!Number.isNaN(lw) && lakeHeld(n, r, lw, seed) && lakeClear(n, r, lw, seed)) { n.lr = r; n.lw = lw; break; }
  }
  return n.lr;
}

/**
 * Does node n's river show: does it leave a lake at n or upstream (a reservoir, or a lake on a river
 * that shows)? Settles n.fedIn, and the lake at every node upstream on the way.
 */
function fedOf(n: HydroNode, seed: number): boolean {
  if (n.fed >= 0) return n.fed === 1;
  // iterative post-order over the upstream tree (like accOf)
  const stack: HydroNode[] = [n];
  while (stack.length) {
    const top = stack[stack.length - 1];
    if (top.fed >= 0) { stack.pop(); continue; }
    let pending = false, fedIn = false;
    if (codeOf(top, seed) !== OCEAN) {
      for (let k = 0; k < 8; k++) {
        const m = nodeAt(top.i + DI[k], top.j + DJ[k], seed);
        if (codeOf(m, seed) !== 7 - k) continue; // m drains into top
        if (m.fed < 0) { stack.push(m); pending = true; } else if (m.fed === 1) fedIn = true;
      }
    }
    if (pending) continue;
    top.fedIn = fedIn ? 1 : 0;
    top.fed = fedIn || riverLakeOf(top, seed) > 0 ? 1 : 0;
    stack.pop();
  }
  return n.fed === 1;
}

/**
 * The level of the lake a river widens into round node n (m): the river's own where it leaves the lake
 * (past its farthest shore), so the lake is held by its outlet and never spills down it.
 */
export function riverLakeLevel(n: HydroNode, seed: number): number {
  return riverLakeOf(n, seed) > 0 ? n.lw : -Infinity;
}

/** A river lake's shore radius toward (dx, dz) from its node (m): lobed, so it is no circle. */
function lakeLobe(r: number, dx: number, dz: number, seed: number): number {
  return r * (1 + 0.28 * vnoise(dx / r * 1.3, dz / r * 1.3, seed + 3503));
}

/**
 * A sink's lake level (m; -Infinity: no lake): just under the lowest ground on a ring of radius
 * HYDRO.lakeR cells round it (the valleys' floors where reaches cross the ring included), so the
 * water never reaches the ring, at most HYDRO.lakeMaxDepth over the sink's river level.
 */
export function lakeOf(s: HydroNode, seed: number): number {
  if (!Number.isNaN(s.L)) return s.L;
  const rL = HYDRO.lakeR * NODE;
  const W0 = levelOf(s, seed);
  let lo = Infinity;
  const N = HYDRO.lakeRing;
  for (let q = 0; q < N; q++) {
    const a = (q / N) * Math.PI * 2;
    const x = s.x + rL * Math.cos(a), z = s.z + rL * Math.sin(a);
    riverField(x, z, seed, K_MIN, false);
    const g = Math.min(upliftAt(x, z, seed), RF.R);
    if (g < lo) lo = g;
  }
  // reaches crossing the ring: their levels where they cross it
  for (let j = s.j - 2; j <= s.j + 2; j++) {
    for (let i = s.i - 2; i <= s.i + 2; i++) {
      const a = nodeAt(i, j, seed);
      const b = downOf(a, seed);
      if (!b) continue;
      const da = Math.hypot(a.x - s.x, a.z - s.z), db = Math.hypot(b.x - s.x, b.z - s.z);
      if ((da - rL) * (db - rL) > 0) continue;
      // the crossing's share of the reach (by bisection on the distance to the sink)
      let t0 = 0, t1 = 1;
      for (let it = 0; it < 24; it++) {
        const tm = 0.5 * (t0 + t1);
        const dm = Math.hypot(a.x + (b.x - a.x) * tm - s.x, a.z + (b.z - a.z) * tm - s.z);
        if ((dm - rL) * (da - rL) > 0) t0 = tm; else t1 = tm;
      }
      const lv = reachLevel(levelOf(a, seed), profileOf(a, seed)!, levelOf(b, seed), 0.5 * (t0 + t1));
      if (lv < lo) lo = lv;
    }
  }
  const L = Math.min(lo - 0.6, W0 + HYDRO.lakeMaxDepth);
  s.L = L > W0 + 1.5 ? L : -Infinity;
  return s.L;
}

/* ------------------------------------------------------------------ */
/* 4. The GPU's table of nodes round the camera                        */
/* ------------------------------------------------------------------ */

/** Floats per node in the table (six vec4). */
export const HYDRO_STRIDE = 24;

export type HydroTable = { i0: number; j0: number; n: number; data: Float32Array<ArrayBuffer> };

/**
 * The nodes of the window HYDRO.window² cells centred on world-warped (cx, cz), packed for the GPU:
 * a header vec4 (i0, j0, n, node size), then per node (row-major from (i0, j0)):
 *   (x − i0·NODE, z − j0·NODE, U, W), (code, acc, L or -1e9, fed), levels at 1/16 … 15/16, and the radius
 *   of the lake at that node (0: none); L is a sink's lake level or that lake's; fed: the river shows (1)
 *   or runs dry (0).
 */
export function hydroTable(cx: number, cz: number, seed: number, out?: HydroTable): HydroTable {
  const n = HYDRO.window;
  const i0 = Math.floor(cx / NODE) - (n >> 1), j0 = Math.floor(cz / NODE) - (n >> 1);
  const data = out && out.data.length === 4 + n * n * HYDRO_STRIDE ? out.data : new Float32Array(4 + n * n * HYDRO_STRIDE);
  data[0] = i0; data[1] = j0; data[2] = n; data[3] = NODE;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const nd = nodeAt(i0 + i, j0 + j, seed);
      const o = 4 + (j * n + i) * HYDRO_STRIDE;
      const code = codeOf(nd, seed);
      data[o] = nd.x - i0 * NODE; data[o + 1] = nd.z - j0 * NODE; data[o + 2] = nd.U; data[o + 3] = levelOf(nd, seed);
      data[o + 4] = code; data[o + 5] = accOf(nd, seed);
      // (a sink's lake level, or the level of the lake the river widens into there)
      data[o + 6] = code === SINK ? Math.max(lakeOf(nd, seed), -1e9) : riverLakeOf(nd, seed) > 0 ? riverLakeLevel(nd, seed) : -1e9;
      data[o + 7] = fedOf(nd, seed) ? 1 : 0;
      const p = profileOf(nd, seed);
      for (let k = 0; k < PROF_N - 1; k++) data[o + 8 + k] = p ? p[k] : 0;
      data[o + 23] = riverLakeOf(nd, seed);
    }
  }
  return { i0, j0, n, data };
}
