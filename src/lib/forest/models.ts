/**
 * Procedural tree models, built once at start-up from seeded random numbers:
 * per species VARIANTS shapes at LODS levels of detail, packed into one
 * vertex buffer and one index buffer for instanced indirect draws.
 * ---------------------------------------------------------------------------
 * Each shape starts as a skeleton (trunk and branch paths, leaf clumps,
 * conifer boughs, palm fronds, cactus arms); every level of detail is emitted
 * from the same skeleton, so switching levels never changes the silhouette.
 *
 * - Trunks and branches: tapered tubes along bent paths (parallel-transport
 *   frames), normals from the surface itself (so flares, buttresses and
 *   cactus ribs shade right).
 * - Canopies: clusters of lumpy leaf clumps (displaced icospheres) whose
 *   normals lean outward from the crown's centre, so a crown shades as one
 *   soft volume while its outline stays lumpy.
 * - Conifers: a tall bare trunk under whorls of boughs, each a feather-edged
 *   sheet folded along its midrib, that leaves the trunk angled down and droops
 *   toward its tip (far away, a skirt round each whorl). Palms: curved ringed trunks under
 *   V-folded, serrated, drooping fronds. Cacti: ribbed columns with bent arms.
 *
 * Vertex (VERTEX_BYTES): position (3 × f32, metres, trunk foot at the origin,
 * y up), normal (snorm8 × 4), attributes (unorm8 × 4): part, wind sway (0 at
 * the foot, 1 at the tips), baked occlusion (dark inside the crown and at
 * the foot), variation (per clump or limb, for colour); surface coordinates
 * (unorm16 × 2) for the shader's patterns: on trunks, limbs and cacti the
 * turn round the tube (0 → 1, a seam vertex closes each ring) and the metres
 * along it (/ SURFACE_SPAN); on fronds and boughs the side (0 one edge, ½ the
 * midrib, 1 the other) and the metres from the stalk; on leaf cards the corner (0 or
 * 1 each way). Clumps use their position.
 *
 * Leaf cards: near levels of broadleaf crowns also carry cards spread over
 * each clump (four vertices at the card's centre, one per corner; the shader
 * turns them to face the camera and paints a leaf spray on them, leafAtlas.ts).
 * They come after each mesh's other triangles, as their own index range, so
 * they can be drawn by a pipeline that cuts out the leaves' shapes.
 */

import { BOUGH_ASPECT } from './leafAtlas';
import { LODS, SPECIES, SPECIES_INFO, VARIANTS, type Species } from './species';

export const VERTEX_BYTES = 24;
/** Metres along a tube or frond that the second surface coordinate spans (0 → 1). */
export const SURFACE_SPAN = 32;
const PART = { bark: 0, leaf: 1, frond: 2, cactus: 3, accent: 4, card: 5 } as const;
/** Leaf cards per clump: this many times its area over a card's (before those buried in other clumps are dropped). */
const CARD_DENSITY = 2.6;
/** Share of the cards the middle level keeps (the shader draws them larger there). */
const CARDS_FAR = 0.28;

type V3 = [number, number, number];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: V3) => Math.sqrt(dot(a, a));
const norm = (a: V3): V3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const smooth = (lo: number, hi: number, x: number) => { const t = clamp01((x - lo) / (hi - lo)); return t * t * (3 - 2 * t); };
/** Unit vector at azimuth a (radians, from +x toward +z) and elevation e above the horizon. */
const dirAE = (a: number, e: number): V3 => [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];

/** mulberry32: a small seeded generator of [0, 1) values. */
function rng(seed: number): (lo?: number, hi?: number) => number {
  let s = seed >>> 0;
  return (lo = 0, hi = 1) => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return lo + (hi - lo) * (((t ^ (t >>> 14)) >>> 0) / 4294967296);
  };
}
type Rng = ReturnType<typeof rng>;

class Builder {
  readonly pos: number[] = [];
  readonly nrm: number[] = [];
  readonly att: number[] = [];
  readonly uv: number[] = [];
  readonly idx: number[] = [];
  /** Leaf cards' triangles (their own range after idx). */
  readonly cardIdx: number[] = [];
  get count(): number { return this.pos.length / 3; }
  /** `u`, `v`: the surface coordinates (round and metres along a tube, side and metres along a frond). */
  vertex(p: V3, n: V3, part: number, sway: number, ao: number, vary: number, u = 0, v = 0): number {
    this.pos.push(p[0], p[1], p[2]);
    this.nrm.push(n[0], n[1], n[2]);
    this.att.push(part, clamp01(sway), clamp01(ao), clamp01(vary));
    this.uv.push(clamp01(u), clamp01(v / SURFACE_SPAN));
    return this.count - 1;
  }
  tri(a: number, b: number, c: number): void { this.idx.push(a, b, c); }
  triCard(a: number, b: number, c: number): void { this.cardIdx.push(a, b, c); }
}

type Crown = { c: V3; r: number };
/** Occlusion of a leaf point: dark in the crown's core and underside, light on its outer top. */
function leafAO(p: V3, crown: Crown): number {
  const d = sub(p, crown.c);
  const out = smooth(0.25, 1.0, len(d) / crown.r);
  const up = Math.max(-1, Math.min(1, d[1] / crown.r));
  return clamp01((0.3 + 0.7 * out) * (0.8 + 0.2 * up));
}

// ---- tubes -------------------------------------------------------------------

type TubeOpts = {
  part: number;
  sides: number;
  height: number;              // tree height, for sway
  crown?: Crown;               // bark inside it is shaded darker
  crownShade?: number;         // by how much there (0.25 unless given)
  vary: number;
  tip?: boolean;               // close the far end with a point
  /** Radius multiplier around the tube (θ) and along it (point index): ribs, buttresses, flares. */
  shape?: (theta: number, i: number, p: V3) => number;
  /** Metres along the tube where its surface coordinate starts (limbs start apart, so their bark differs). */
  along0?: number;
};

/** A tapered tube through `path` with one radius per point; returns nothing, appends to b. */
function tube(b: Builder, path: V3[], radii: number[], o: TubeOpts): void {
  const m = path.length;
  const T: V3[] = path.map((_, i) => norm(sub(path[Math.min(i + 1, m - 1)], path[Math.max(i - 1, 0)])));
  // Parallel-transport frames (projection form: fine for gently bent paths).
  let N: V3 = Math.abs(T[0][1]) < 0.9 ? norm(cross(T[0], [0, 1, 0])) : norm(cross(T[0], [1, 0, 0]));
  const rings: V3[][] = [];
  for (let i = 0; i < m; i++) {
    N = norm(sub(N, mul(T[i], dot(N, T[i]))));
    const B = cross(T[i], N);
    const ring: V3[] = [];
    for (let s = 0; s < o.sides; s++) {
      const th = (2 * Math.PI * s) / o.sides;
      const dir = add(mul(N, Math.cos(th)), mul(B, Math.sin(th)));
      const r = radii[i] * (o.shape ? o.shape(th, i, path[i]) : 1);
      ring.push(add(path[i], mul(dir, r)));
    }
    rings.push(ring);
  }
  // Metres along the path at each ring (the bark's patterns run along it).
  const along = [o.along0 ?? 0];
  for (let i = 1; i < m; i++) along.push(along[i - 1] + len(sub(path[i], path[i - 1])));
  const base = b.count;
  // Each ring repeats its first vertex at the end (round 1 instead of 0), so the patterns close without a seam.
  const ring = o.sides + 1;
  for (let i = 0; i < m; i++) {
    for (let k = 0; k <= o.sides; k++) {
      const s = k % o.sides;
      const p = rings[i][s];
      const tangent = sub(rings[i][(s + 1) % o.sides], rings[i][(s + o.sides - 1) % o.sides]);
      const up = sub(rings[Math.min(i + 1, m - 1)][s], rings[Math.max(i - 1, 0)][s]);
      let n = norm(cross(tangent, up));
      if (len(n) < 0.5 || !Number.isFinite(n[0])) n = norm(sub(p, path[i]));
      const inCrown = o.crown ? 1 - smooth(0.55, 1.05, len(sub(p, o.crown.c)) / o.crown.r) : 0;
      const ao = (0.62 + 0.38 * smooth(0.0, 1.6, p[1])) * (1 - (o.crownShade ?? 0.25) * inCrown);
      b.vertex(p, n, o.part, Math.pow(clamp01(p[1] / o.height), 1.5), ao, o.vary, k / o.sides, along[i]);
    }
  }
  for (let i = 0; i < m - 1; i++) {
    for (let s = 0; s < o.sides; s++) {
      const a = base + i * ring + s, c = a + ring;
      b.tri(a, a + 1, c);
      b.tri(a + 1, c + 1, c);
    }
  }
  if (o.tip) {
    const last = path[m - 1];
    const tipLen = Math.max(radii[m - 1] * 1.5, 0.02);
    const tip = add(last, mul(T[m - 1], tipLen));
    const t = b.vertex(tip, T[m - 1], o.part, Math.pow(clamp01(tip[1] / o.height), 1.5), 0.85, o.vary, 0.5, along[m - 1] + tipLen);
    const r0 = base + (m - 1) * ring;
    for (let s = 0; s < o.sides; s++) b.tri(r0 + s, r0 + s + 1, t);
  }
}

/** Resample a path to n points (linear in arc length); radii follow. */
function resample(path: V3[], radii: number[], n: number): [V3[], number[]] {
  if (n >= path.length) return [path, radii];
  const acc = [0];
  for (let i = 1; i < path.length; i++) acc.push(acc[i - 1] + len(sub(path[i], path[i - 1])));
  const total = acc[acc.length - 1];
  const P: V3[] = [], R: number[] = [];
  for (let k = 0; k < n; k++) {
    const s = (total * k) / (n - 1);
    let i = 1;
    while (i < acc.length - 1 && acc[i] < s) i++;
    const t = (s - acc[i - 1]) / Math.max(acc[i] - acc[i - 1], 1e-9);
    P.push(add(path[i - 1], mul(sub(path[i], path[i - 1]), t)));
    R.push(radii[i - 1] + (radii[i] - radii[i - 1]) * t);
  }
  return [P, R];
}

/** A limb growing from `start` along `dir`, bending toward `bendTo` a little each segment. */
function grow(r: Rng, start: V3, dir: V3, length: number, segs: number, bendTo: V3, bend: number, wobble: number): V3[] {
  const pts: V3[] = [start];
  let d = norm(dir);
  let p = start;
  for (let i = 0; i < segs; i++) {
    const jitter: V3 = [r(-1, 1) * wobble, r(-1, 1) * wobble * 0.5, r(-1, 1) * wobble];
    d = norm(add(add(d, mul(bendTo, bend)), jitter));
    p = add(p, mul(d, length / segs));
    pts.push(p);
  }
  return pts;
}

// ---- leaf clumps ---------------------------------------------------------------

const icoCache = new Map<number, { v: V3[]; f: Array<[number, number, number]> }>();
function icosphere(subdiv: number): { v: V3[]; f: Array<[number, number, number]> } {
  const hit = icoCache.get(subdiv);
  if (hit) return hit;
  const t = (1 + Math.sqrt(5)) / 2;
  let v: V3[] = ([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]] as V3[]).map(norm);
  let f: Array<[number, number, number]> = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  for (let k = 0; k < subdiv; k++) {
    const mid = new Map<string, number>();
    const m = (a: number, b: number) => {
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      let i = mid.get(key);
      if (i === undefined) { i = v.length; v = [...v, norm(mul(add(v[a], v[b]), 0.5))]; mid.set(key, i); }
      return i;
    };
    const nf: Array<[number, number, number]> = [];
    for (const [a, b, c] of f) {
      const ab = m(a, b), bc = m(b, c), ca = m(c, a);
      nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    f = nf;
  }
  const out = { v, f };
  icoCache.set(subdiv, out);
  return out;
}

type Clump = { c: V3; r: V3; vary: number; seed: number };

/** A lumpy clump: an icosphere scaled to radii r and displaced by three random waves. */
function clump(b: Builder, k: Clump, subdiv: number, amp: number, crown: Crown, height: number, part: number = PART.leaf): void {
  const ico = icosphere(subdiv);
  const r = rng(k.seed);
  const waves = [0, 1, 2].map(() => ({ d: norm([r(-1, 1), r(-1, 1), r(-1, 1)] as V3), f: r(2.2, 3.6), ph: r(0, 6.283) }));
  const P = ico.v.map((u) => {
    const bump = 1 + amp * waves.reduce((s, w) => s + Math.sin(dot(u, w.d) * w.f + w.ph), 0) / 3;
    return add(k.c, [u[0] * k.r[0] * bump, u[1] * k.r[1] * bump, u[2] * k.r[2] * bump] as V3);
  });
  // Surface normals (area-weighted face average), leaned outward from the crown.
  const acc: V3[] = P.map(() => [0, 0, 0]);
  for (const [a, bb, c] of ico.f) {
    const fn = cross(sub(P[bb], P[a]), sub(P[c], P[a]));
    for (const i of [a, bb, c]) acc[i] = add(acc[i], fn);
  }
  const base = b.count;
  P.forEach((p, i) => {
    const out = norm(sub(p, crown.c));
    const n = norm(add(mul(norm(acc[i]), 0.3), mul(out, 0.7)));
    b.vertex(p, n, part, 0.15 + Math.pow(clamp01(p[1] / height), 1.2), leafAO(p, crown), k.vary);
  });
  for (const [a, bb, c] of ico.f) b.tri(base + a, base + bb, base + c);
}

/** The crown round a set of clumps: their weighted centre and the radius reaching every clump's edge. */
function crownOf(clumps: Clump[]): Crown {
  let w = 0;
  let c: V3 = [0, 0, 0];
  for (const k of clumps) { const v = k.r[0] * k.r[1] * k.r[2]; c = add(c, mul(k.c, v)); w += v; }
  c = mul(c, 1 / Math.max(w, 1e-9));
  const r = Math.max(...clumps.map((k) => len(sub(k.c, c)) + Math.max(...k.r)), 0.5);
  return { c, r };
}

/** Two big blobs standing in for a crown far away: the clumps split across their widest horizontal axis. */
function mergeClumps(clumps: Clump[], shrink: number): Clump[] {
  const cr = crownOf(clumps);
  let far = clumps[0];
  for (const k of clumps) if (Math.hypot(k.c[0] - cr.c[0], k.c[2] - cr.c[2]) > Math.hypot(far.c[0] - cr.c[0], far.c[2] - cr.c[2])) far = k;
  const axis = norm([far.c[0] - cr.c[0], 0, far.c[2] - cr.c[2]] as V3);
  const groups = [clumps.filter((k) => dot(sub(k.c, cr.c), axis) >= 0), clumps.filter((k) => dot(sub(k.c, cr.c), axis) < 0)].filter((g) => g.length);
  return groups.map((g, gi) => {
    const c = crownOf(g).c;
    let rx = 0, ry = 0;
    for (const k of g) {
      rx = Math.max(rx, Math.hypot(k.c[0] - c[0], k.c[2] - c[2]) + Math.max(k.r[0], k.r[2]));
      ry = Math.max(ry, Math.abs(k.c[1] - c[1]) + k.r[1]);
    }
    return { c, r: [rx * shrink, ry * shrink, rx * shrink] as V3, vary: g[0].vary, seed: g[0].seed + 101 * gi };
  });
}

// ---- skeletons -----------------------------------------------------------------

type Limb = {
  path: V3[];
  r0: number; r1: number;   // radius at the start and end (linear), unless `radii` gives one per point
  radii?: number[];
  level: 0 | 1 | 2;         // trunk, limb, twig
  vary: number;
  tip?: boolean;            // close the far end with a point
  shaped?: boolean;         // apply the skeleton's shape function (buttresses, flare, ribs)
};
/** A conifer's bough: its midrib from the trunk to the tip (equal arc lengths) and how wide the sheet round it grows. */
type Bough = { a: number; path: V3[]; len: number; width: number; vary: number; seed: number };
type Frond = { a: number; e: number; len: number; width: number; droop: number; seed: number };
type Skeleton = {
  height: number;
  limbs: Limb[];
  clumps: Clump[];
  /** Conifer boughs in runs up their spiral (bottom first), each run in order of azimuth; then the spire. */
  whorls?: Bough[][];
  fronds?: { base: V3; list: Frond[] };
  accents?: Clump[];
  shape?: TubeOpts['shape'];
  part?: number;            // limb material (cactus skin for cacti)
  lod2Blobs?: 1 | 2;        // merged blobs standing in for the crown far away
  lod2Levels?: 0 | 1;       // limb levels still drawn far away (bare and flat-topped trees need theirs)
  keepPoints?: boolean;     // use every path point at LOD 0 (shaped profiles)
  shade?: Crown;            // bark inside it is shaded darker (a crown of boughs, which are not clumps)
  shadeDepth?: number;      // by how much (TubeOpts crownShade)
};

/** Oak and maple: a stout trunk forking into a few rising limbs, a broad lumpy crown. */
function broadleaf(r: Rng, species: 'oak' | 'maple'): Skeleton {
  const oak = species === 'oak';
  const H = SPECIES_INFO[species].height * r(0.9, 1.08);
  const k = H / 11;
  const fork = H * r(oak ? 0.34 : 0.38, oak ? 0.44 : 0.46);
  const leanA = r(0, 6.283), lean = r(0.15, 0.7);
  const trunkPath: V3[] = [];
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    const off = lean * Math.pow(t, 1.5);
    trunkPath.push([Math.cos(leanA) * off + r(-0.05, 0.05), -0.6 + (fork + 0.6) * t, Math.sin(leanA) * off + r(-0.05, 0.05)]);
  }
  const top = trunkPath[5];
  // Rings close together at the foot, where the roots spread (the same path, more points on it).
  const lerp3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const trunk = [trunkPath[0], lerp3(trunkPath[0], trunkPath[1], 1 / 3), lerp3(trunkPath[0], trunkPath[1], 2 / 3), trunkPath[1], lerp3(trunkPath[1], trunkPath[2], 0.5), ...trunkPath.slice(2)];
  const limbs: Limb[] = [{ path: trunk, r0: 0.44 * k, r1: 0.3 * k, level: 0, vary: r(), shaped: true }];
  const clumps: Clump[] = [];
  const n = r() < 0.5 ? 3 : 4;
  const a0 = r(0, 6.283);
  for (let i = 0; i < n; i++) {
    const a = a0 + (i * 2 * Math.PI) / n + r(-0.35, 0.35);
    const e = (oak ? r(32, 52) : r(42, 62)) * Math.PI / 180;
    const L = H * r(0.32, 0.42);
    const path = grow(r, add(top, [0, -0.2, 0]), dirAE(a, e), L, 4, [0, 1, 0], r(0.12, 0.22), 0.08);
    limbs.push({ path, r0: 0.23 * k, r1: 0.07 * k, level: 1, vary: r() });
    const tip = path[path.length - 1];
    clumps.push({ c: add(tip, [0, 0.5 * k, 0]), r: [r(1.9, 2.5) * k, r(1.5, 1.9) * k, r(1.9, 2.5) * k], vary: r(), seed: (r() * 1e9) | 0 });
    // Two side shoots per limb, each carrying its own clump.
    for (const at of [2, 3]) {
      const sa = a + (at === 2 ? 1 : -1) * r(0.45, 0.8);
      const sp = grow(r, path[at], dirAE(sa, e + r(0.1, 0.35)), L * r(0.36, 0.48), 3, [0, 1, 0], 0.15, 0.06);
      limbs.push({ path: sp, r0: 0.09 * k, r1: 0.03 * k, level: 2, vary: r(), tip: true });
      clumps.push({ c: add(sp[sp.length - 1], [0, 0.3 * k, 0]), r: [r(1.4, 1.9) * k, r(1.1, 1.5) * k, r(1.4, 1.9) * k], vary: r(), seed: (r() * 1e9) | 0 });
    }
  }
  // Leader up the middle, and the crown's top clump.
  const leader = grow(r, top, dirAE(r(0, 6.283), (80 - r(0, 12)) * Math.PI / 180), H - fork - 2.2 * k, 3, [0, 1, 0], 0.1, 0.05);
  limbs.push({ path: leader, r0: 0.2 * k, r1: 0.06 * k, level: 1, vary: r() });
  const lt = leader[leader.length - 1];
  clumps.push({ c: add(lt, [0, 0.4 * k, 0]), r: [r(2.3, 2.8) * k, r(1.8, 2.2) * k, r(2.3, 2.8) * k], vary: r(), seed: (r() * 1e9) | 0 });
  // Fill between the limbs so the crown has no holes.
  const mid = crownOf(clumps).c;
  for (let i = 0; i < 3; i++) {
    const a = r(0, 6.283);
    clumps.push({ c: add(mid, [Math.cos(a) * 1.4 * k, r(-0.8, 0.4) * k, Math.sin(a) * 1.4 * k]), r: [2.2 * k, 1.7 * k, 2.2 * k], vary: r(), seed: (r() * 1e9) | 0 });
  }
  // Root flare: the foot swells and splits into roots spreading over the ground.
  const rootTurn = r(0, 6.283), roots = r() < 0.5 ? 4 : 5;
  const shape = (th: number, _i: number, p: V3) => {
    const y = p[1] + 0.6; // above the trunk's foot (below the ground)
    const foot = clamp01(1 - y / (2.2 * k));
    return 1 + 0.5 * Math.exp(-y / 0.7) + 2.0 * foot * foot * Math.pow(Math.max(0, Math.cos(roots * th + rootTurn)), 1.5);
  };
  return { height: H, limbs, clumps, shape, lod2Blobs: 2, keepPoints: true };
}

/** Birch: a slender pale trunk, short upswept branches, a narrow oval crown of small clumps. */
function birch(r: Rng): Skeleton {
  const H = SPECIES_INFO.birch.height * r(0.9, 1.08);
  const leanA = r(0, 6.283), lean = r(0.3, 0.9);
  const trunk: V3[] = [];
  for (let i = 0; i <= 7; i++) {
    const t = i / 7;
    const off = lean * t * t + 0.18 * Math.sin(t * 5 + leanA);
    trunk.push([Math.cos(leanA) * off, -0.5 + (H * 0.93 + 0.5) * t, Math.sin(leanA) * off]);
  }
  const at = (y: number): V3 => {
    const t = (y + 0.5) / (H * 0.93 + 0.5), i = Math.min(6, Math.floor(t * 7)), f = t * 7 - i;
    return add(trunk[i], mul(sub(trunk[i + 1], trunk[i]), f));
  };
  const limbs: Limb[] = [{ path: trunk, r0: 0.25, r1: 0.05, level: 0, vary: r(), tip: true }];
  const clumps: Clump[] = [];
  const n = 7 + Math.floor(r(0, 3));
  for (let i = 0; i < n; i++) {
    const y = H * (0.42 + (0.44 * i) / n) + r(-0.3, 0.3);
    const a = i * 2.39996 + r(-0.3, 0.3);
    const L = H * (0.27 - 0.1 * (i / n)) * r(0.85, 1.15);
    const path = grow(r, at(y), dirAE(a, r(30, 52) * Math.PI / 180), L, 3, [0, 1, 0], 0.18, 0.08);
    limbs.push({ path, r0: 0.065, r1: 0.02, level: 1, vary: r(), tip: true });
    const s = r(1.05, 1.45);
    clumps.push({ c: add(path[path.length - 1], [0, 0.2, 0]), r: [s, s * 1.15, s], vary: r(), seed: (r() * 1e9) | 0 });
  }
  for (const f of [0.58, 0.72, 0.84, 0.93]) {
    const s = r(1.25, 1.6) * (f > 0.9 ? 0.8 : 1);
    clumps.push({ c: add(at(H * f), [r(-0.4, 0.4), 0, r(-0.4, 0.4)]), r: [s, s * 1.2, s], vary: r(), seed: (r() * 1e9) | 0 });
  }
  return { height: H, limbs, clumps, lod2Blobs: 2 };
}

const TAU = 2 * Math.PI;
/** Steps in a bough's midrib. */
const BOUGH_STEPS = 16;

/**
 * A bough's midrib, trunk to tip, in `BOUGH_STEPS` equal steps: it leaves the trunk at elevation `e0`,
 * bends down by `bend` (radians) along its length, more toward the tip, and curls back up by `curl`
 * over its outer part; scaled so that it reaches `reach` metres out from the trunk's axis.
 */
function boughPath(root: V3, a: number, e0: number, bend: number, reach: number, curl = 0): { path: V3[]; len: number } {
  const unit: V3[] = [[0, 0, 0]];
  let x = 0, y = 0;
  for (let i = 1; i <= BOUGH_STEPS; i++) {
    const f = (i - 0.5) / BOUGH_STEPS;
    const e = e0 - bend * Math.pow(f, 1.4) + curl * Math.pow(f, 2.5);
    x += Math.cos(e) / BOUGH_STEPS;
    y += Math.sin(e) / BOUGH_STEPS;
    unit.push([x, y, 0]);
  }
  const len = reach / Math.max(x, 0.2);
  const ca = Math.cos(a), sa = Math.sin(a);
  return { len, path: unit.map((p) => add(root, [p[0] * ca * len, p[1] * len, p[0] * sa * len])) };
}

/**
 * Spruce / fir: a tall, straight, bare trunk and, from nearly two-fifths of the way up, whorls of boughs that
 * leave it angled a little down and droop toward their tips, the lowest the longest and the most bowed, shorter
 * and more upright toward a thin spire. The boughs are staggered turn by turn, so the layers show
 * through each other.
 */
function pine(r: Rng): Skeleton {
  const H = SPECIES_INFO.pine.height * r(0.88, 1.1);
  const k = H / 17;
  const lean: V3 = [r(-0.3, 0.3), 0, r(-0.3, 0.3)];
  const bow: V3 = [r(-0.14, 0.14), 0, r(-0.14, 0.14)];
  const trunkAt = (y: number): V3 => {
    const t = clamp01((y + 0.6) / (H + 0.6));
    const w = Math.sin(Math.PI * t);
    return [lean[0] * t * t + bow[0] * w, y, lean[2] * t * t + bow[2] * w];
  };
  // A column, hardly thinner at the top than at the foot, that comes to a sharp point in its last
  // few metres (more rings up there, for the point).
  const trunk: V3[] = [];
  const radii: number[] = [];
  for (const t of [0, 0.12, 0.25, 0.4, 0.55, 0.7, 0.82, 0.89, 0.94, 0.975, 1]) {
    trunk.push(trunkAt(-0.6 + (0.86 * H + 0.6) * t)); // (its tip hidden in the leader's spike)
    const point = t <= 0.82 ? 1 : Math.pow(clamp01(1 - (t - 0.82) / 0.18), 0.75);
    radii.push(Math.max(0.035, 0.44 * k * (1 - 0.22 * t) * point));
  }
  // A point on the trunk's surface at height y, toward azimuth a (a little inside it, so what grows from it is joined on).
  const onBark = (y: number, a: number): V3 => {
    const t = clamp01((y + 0.6) / (0.86 * H + 0.6));
    const rad = 0.9 * 0.44 * k * (1 - 0.22 * t) * (t <= 0.82 ? 1 : Math.pow(clamp01(1 - (t - 0.82) / 0.18), 0.75));
    return add(trunkAt(y), [rad * Math.cos(a), 0, rad * Math.sin(a)]);
  };
  // The foot swells and spreads a little into roots.
  const rootTurn = r(0, 6.283);
  const shape = (th: number, _i: number, p: V3) => {
    const y = p[1] + 0.6;
    return 1 + 0.7 * Math.exp(-y / 0.55) + 0.25 * Math.exp(-y / 0.9) * Math.pow(Math.max(0, Math.cos(4 * th + rootTurn)), 2);
  };

  const y0 = H * r(0.37, 0.41);
  const yTop = H * 0.9;
  const W = 8 + Math.floor(r(0, 2));
  const spin = r(0, 6.283);
  // As many boughs as whorls of three or four would hold, but each on its own: they spiral up the trunk, each
  // a little higher than the last and turned the golden angle from it, so at every height some bough
  // crosses in front of the trunk whichever way it is seen from (closer together toward the top, where the
  // boughs are short, so they still close over it there).
  let N = 0;
  for (let j = 0; j < W; j++) N += Math.max(3, Math.round(4.4 - 1.2 * Math.pow(j / (W - 1), 1.2) + r(-0.5, 0.5)));
  const spiral: Bough[] = [];
  for (let n = 0; n < N; n++) {
    const u = n / (N - 1);
    const y = y0 + (yTop - y0) * Math.pow(u, 0.62) + r(-0.04, 0.04) * k;
    const a = (((spin + n * 2.39996 + r(-0.15, 0.15)) % TAU) + TAU) % TAU;
    // Reach: graded from the bottom up, as the Hyrule spruces are: long boughs below, shorter through the
    // middle, short ones under the spire; each a little longer or shorter.
    const reach = Math.max(1.2 * k, 4.4 * k * (1 - 0.66 * Math.pow(u, 1.1)) * r(0.78, 1.12));
    // (they leave the trunk sloping down, run on down most of their length and curl up a little at the tip,
    // which still points a little down)
    const e0 = (-27 + 8 * Math.pow(u, 1.2) + r(-4, 4)) * Math.PI / 180;
    const curl = (23 - 6 * u + r(-5, 5)) * Math.PI / 180;
    // (each grows from the trunk's surface, so where it starts shows)
    const { path, len } = boughPath(onBark(y, a), a, e0, 0, reach, curl);
    // (the short ones up top relatively broader, so they still span the trunk)
    spiral.push({ a, path, len, width: 0.5 * BOUGH_ASPECT * Math.max(len, 0.55 * len + 1.2 * k) * r(0.95, 1.12), vary: r(), seed: (r() * 1e9) | 0 });
  }
  // (in runs of four up the spiral, each run a "whorl" for the skirts that stand in for the boughs far off)
  const whorls: Bough[][] = [];
  for (let n = 0; n < N; n += 4) whorls.push(spiral.slice(n, n + 4).sort((p, q) => p.a - q.a));
  // The spire: a few sprigs round the leader's foot, swept up along it, and the leader itself, a tall narrow
  // spike of two crossed sheets standing straight up from below the trunk's tip, so the top of the trunk is inside it.
  const spire: Bough[] = [];
  for (let i = 0; i < 5; i++) {
    const a = spin + (i * TAU) / 5 + r(-0.3, 0.3);
    const y = H * r(0.8, 0.88);
    // (swept up along the spike, so the head comes to a point)
    const { path, len } = boughPath(trunkAt(y), a, (52 + r(-6, 6)) * Math.PI / 180, 0.35, 1.0 * k * r(0.85, 1.15));
    spire.push({ a, path, len, width: 0.32 * BOUGH_ASPECT * len, vary: r(), seed: (r() * 1e9) | 0 });
  }
  const tipLen = 4.2 * k * r(0.92, 1.05);
  for (const a of [spin, spin + Math.PI / 2]) {
    const root = trunkAt(H * 0.74);
    const path = Array.from({ length: BOUGH_STEPS + 1 }, (_, i) => add(root, [0, (tipLen * i) / BOUGH_STEPS, 0]));
    spire.push({ a, path, len: tipLen, width: 0.3 * BOUGH_ASPECT * tipLen, vary: r(), seed: (r() * 1e9) | 0 });
  }
  // (bottom whorl first: drawn in that order, the boughs above fail the depth test before their cut-out is shaded)
  whorls.push(spire);
  const limbs: Limb[] = [{ path: trunk, r0: 0, r1: 0, radii, level: 0, vary: r(), tip: true, shaped: true }];
  // (the trunk darkens where the crown's boughs shade it: deep in their shade, as the spruces of the Hyrule games are)
  return { height: H, limbs, clumps: [], whorls, shape, keepPoints: true, shade: { c: [0, 0.66 * H, 0], r: 0.4 * H }, shadeDepth: 0.72 };
}

/** Rainforest giant: a buttressed pillar, spreading limbs, a wide flat layered canopy. */
function jungle(r: Rng): Skeleton {
  const H = SPECIES_INFO.jungle.height * r(0.9, 1.06);
  const k = H / 21;
  const crownY = H * r(0.66, 0.72);
  const la = r(0, 6.283), lean = r(0.3, 1.0);
  const trunk: V3[] = [];
  for (let i = 0; i <= 6; i++) {
    const t = i / 6;
    trunk.push([Math.cos(la) * lean * t * t, -0.6 + (crownY + 0.6) * t, Math.sin(la) * lean * t * t]);
  }
  const top = trunk[6];
  const fins = r(0, 6.283);
  const limbs: Limb[] = [{ path: trunk, r0: 0.6 * k, r1: 0.3 * k, level: 0, vary: r(), shaped: true }];
  const clumps: Clump[] = [];
  const n = 4 + Math.floor(r(0, 3));
  const a0 = r(0, 6.283);
  for (let i = 0; i < n; i++) {
    const a = a0 + (i * 2 * Math.PI) / n + r(-0.3, 0.3);
    const L = H * r(0.28, 0.36);
    const path = grow(r, add(top, [0, -0.3, 0]), dirAE(a, r(12, 28) * Math.PI / 180), L, 4, [0, 1, 0], r(0.15, 0.26), 0.07);
    limbs.push({ path, r0: 0.26 * k, r1: 0.08 * k, level: 1, vary: r() });
    const tip = path[path.length - 1];
    clumps.push({ c: add(tip, [0, 0.4 * k, 0]), r: [r(2.5, 3.2) * k, r(1.1, 1.4) * k, r(2.5, 3.2) * k], vary: r(), seed: (r() * 1e9) | 0 });
    const mid = path[2];
    clumps.push({ c: add(mid, [0, 1.0 * k, 0]), r: [r(2.0, 2.6) * k, r(0.9, 1.2) * k, r(2.0, 2.6) * k], vary: r(), seed: (r() * 1e9) | 0 });
  }
  clumps.push({ c: add(top, [0, 2.2 * k, 0]), r: [3.0 * k, 1.4 * k, 3.0 * k], vary: r(), seed: (r() * 1e9) | 0 });
  // A lower, smaller layer on two short limbs.
  for (let i = 0; i < 2; i++) {
    const y = H * r(0.5, 0.58);
    const a = r(0, 6.283);
    const s = add(trunk[Math.round((y + 0.6) / (crownY + 0.6) * 6)], [0, 0, 0]);
    const path = grow(r, s, dirAE(a, r(15, 30) * Math.PI / 180), H * 0.16, 2, [0, 1, 0], 0.2, 0.05);
    limbs.push({ path, r0: 0.12 * k, r1: 0.05 * k, level: 2, vary: r(), tip: true });
    clumps.push({ c: add(path[path.length - 1], [0, 0.3 * k, 0]), r: [1.8 * k, 0.85 * k, 1.8 * k], vary: r(), seed: (r() * 1e9) | 0 });
  }
  // Buttress fins flare the foot of the trunk.
  const shape = (th: number, _i: number, p: V3) => {
    const foot = clamp01(1 - (p[1] + 0.6) / (3.2 * k));
    return 1 + foot * foot * (0.25 + 1.1 * Math.pow(Math.max(0, Math.cos(4 * th + fins)), 3));
  };
  return { height: H, limbs, clumps, shape, lod2Blobs: 2 };
}

/** Palm: a curved ringed trunk leaning then rising, a rosette of drooping fronds. */
function palm(r: Rng): Skeleton {
  const H = SPECIES_INFO.palm.height * r(0.85, 1.05);
  const trunkH = H * 0.86;
  const a = r(0, 6.283), lean = r(1.2, 3.2);
  const trunk: V3[] = [];
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    const off = lean * (1 - (1 - t) * (1 - t));
    trunk.push([Math.cos(a) * off, -0.5 + (trunkH + 0.5) * t, Math.sin(a) * off]);
  }
  const top = trunk[8];
  const list: Frond[] = [];
  const n = 11;
  const a0 = r(0, 6.283);
  for (let i = 0; i < n; i++) {
    list.push({
      a: a0 + (i * 2 * Math.PI) / n + r(-0.2, 0.2),
      e: (i % 2 === 0 ? r(18, 38) : r(-12, 8)) * Math.PI / 180,
      len: r(3.8, 5.0),
      width: r(0.7, 0.9),
      droop: r(0.28, 0.45),
      seed: (r() * 1e9) | 0,
    });
  }
  const accents: Clump[] = [];
  for (let i = 0; i < 4; i++) {
    const ca = r(0, 6.283);
    accents.push({ c: add(top, [Math.cos(ca) * 0.3, -0.35 + r(-0.1, 0.1), Math.sin(ca) * 0.3]), r: [0.14, 0.16, 0.14], vary: r(), seed: (r() * 1e9) | 0 });
  }
  const shape = (_th: number, _i: number, p: V3) => 1 + 0.35 * clamp01(1 - (p[1] + 0.5) / 1.2);
  return { height: H, limbs: [{ path: trunk, r0: 0.25, r1: 0.17, level: 0, vary: r(), shaped: true }], clumps: [], fronds: { base: top, list }, accents, shape };
}

/** Acacia: a short leaning trunk splitting into wide limbs under a flat umbrella of foliage. */
function acacia(r: Rng): Skeleton {
  const H = SPECIES_INFO.acacia.height * r(0.88, 1.08);
  const k = H / 7.5;
  const fork = H * r(0.28, 0.38);
  const la = r(0, 6.283);
  const trunk: V3[] = [[0, -0.5, 0], [Math.cos(la) * 0.2, fork * 0.5, Math.sin(la) * 0.2], [Math.cos(la) * 0.45, fork, Math.sin(la) * 0.45]];
  const top = trunk[2];
  const limbs: Limb[] = [{ path: trunk, r0: 0.3 * k, r1: 0.2 * k, level: 0, vary: r() }];
  const clumps: Clump[] = [];
  const n = r() < 0.45 ? 2 : 3;
  const a0 = r(0, 6.283);
  const crownY = H * r(0.78, 0.86);
  for (let i = 0; i < n; i++) {
    const a = a0 + (i * 2 * Math.PI) / n + r(-0.4, 0.4);
    const reach = r(2.6, 4.0) * k;
    const e = Math.atan2(crownY - fork, reach);
    const path = grow(r, top, dirAE(a, e), Math.hypot(reach, crownY - fork), 4, [0, -0.1, 0], 0.05, 0.1);
    limbs.push({ path, r0: 0.17 * k, r1: 0.06 * k, level: 1, vary: r() });
    const tip = path[path.length - 1];
    clumps.push({ c: add(tip, [0, 0.35 * k, 0]), r: [r(2.2, 2.9) * k, r(0.5, 0.68) * k, r(2.2, 2.9) * k], vary: r(), seed: (r() * 1e9) | 0 });
    const side = grow(r, path[2], dirAE(a + r(0.6, 1.0), e * 0.6), reach * 0.55, 2, [0, 1, 0], 0.2, 0.08);
    limbs.push({ path: side, r0: 0.07 * k, r1: 0.03 * k, level: 2, vary: r(), tip: true });
    clumps.push({ c: add(side[side.length - 1], [0, 0.3 * k, 0]), r: [r(1.6, 2.1) * k, r(0.45, 0.6) * k, r(1.6, 2.1) * k], vary: r(), seed: (r() * 1e9) | 0 });
  }
  return { height: H, limbs, clumps, lod2Blobs: 2, lod2Levels: 1 };
}

/** Saguaro: a ribbed column with a rounded top and up to three bent arms. */
function cactus(r: Rng): Skeleton {
  const H = SPECIES_INFO.cactus.height * r(0.8, 1.12);
  const R = r(0.32, 0.4);
  const col: V3[] = [];
  const colR: number[] = [];
  const capFrom = 1 - (R * 1.3) / H;
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const tt = capFrom + (1 - capFrom) * Math.max(0, (t - 0.6) / 0.4);
    const y = t < 0.6 ? -0.3 + (H * capFrom + 0.3) * (t / 0.6) : H * tt;
    col.push([0.03 * Math.sin(t * 4), y, 0.03 * Math.cos(t * 3)]);
    const c = t < 0.6 ? 1 : Math.sqrt(Math.max(0, 1 - ((tt - capFrom) / (1 - capFrom)) ** 2));
    colR.push(R * Math.max(c, 0.05));
  }
  const limbs: Limb[] = [{ path: col, r0: 0, r1: 0, radii: colR, level: 0, vary: r(), shaped: true }];
  const roll = r();
  const arms = roll < 0.15 ? 0 : roll < 0.5 ? 1 : roll < 0.85 ? 2 : 3;
  const a0 = r(0, 6.283);
  const accents: Clump[] = [];
  for (let i = 0; i < arms; i++) {
    const a = a0 + (i * 2 * Math.PI) / Math.max(arms, 1) + r(-0.5, 0.5);
    const y = H * r(0.35, 0.58);
    const out = r(0.45, 0.7), rise = r(0.9, 1.9);
    const d: V3 = [Math.cos(a), 0, Math.sin(a)];
    const rr = r(0.21, 0.27);
    const path: V3[] = [];
    const ar: number[] = [];
    // Out, a quarter bend, then up; the last two points round the top.
    const bendR = 0.35;
    const pts: Array<[number, number]> = [[0, 0], [out * 0.5, 0.02], [out, 0.08], [out + bendR * 0.7, bendR * 0.3], [out + bendR, bendR], [out + bendR, bendR + rise * 0.5], [out + bendR, bendR + rise], [out + bendR, bendR + rise + rr * 0.7]];
    pts.forEach(([h, v], j) => {
      path.push(add([0, y, 0], add(mul(d, h), [0, v, 0])));
      ar.push(j === pts.length - 1 ? rr * 0.35 : rr);
    });
    limbs.push({ path, r0: 0, r1: 0, radii: ar, level: 1, vary: r(), shaped: true });
    accents.push({ c: path[path.length - 1], r: [0.09, 0.07, 0.09], vary: r(), seed: (r() * 1e9) | 0 });
  }
  accents.push({ c: col[col.length - 1], r: [0.1, 0.08, 0.1], vary: r(), seed: (r() * 1e9) | 0 });
  const ribs = (th: number) => 1 + 0.09 * Math.cos(8 * th);
  return { height: H, limbs, clumps: [], accents, shape: ribs, part: PART.cactus, lod2Levels: 1, keepPoints: true };
}

/** Dead tree: a crooked grey trunk, often snapped, and bare twisting limbs. */
function dead(r: Rng): Skeleton {
  const H = SPECIES_INFO.dead.height * r(0.8, 1.05);
  const broken = r() < 0.4;
  const tH = broken ? H * r(0.55, 0.75) : H * 0.92;
  const trunk: V3[] = [];
  let p: V3 = [0, -0.5, 0];
  for (let i = 0; i <= 6; i++) {
    trunk.push(p);
    p = add(p, [r(-0.3, 0.3), (tH + 0.5) / 6, r(-0.3, 0.3)]);
  }
  const limbs: Limb[] = [{ path: trunk, r0: 0.3, r1: broken ? 0.16 : 0.05, level: 0, vary: r(), tip: !broken }];
  const n = 4 + Math.floor(r(0, 3));
  for (let i = 0; i < n; i++) {
    const at = 2 + Math.floor(r(0, 4));
    const a = r(0, 6.283);
    const L = H * r(0.25, 0.45);
    const path = grow(r, trunk[Math.min(at, 6)], dirAE(a, r(18, 58) * Math.PI / 180), L, 4, [0, 0.4, 0], 0.08, 0.35);
    limbs.push({ path, r0: 0.1, r1: 0.02, level: 1, vary: r(), tip: true });
    const tw = grow(r, path[2], dirAE(a + r(-1, 1), r(20, 60) * Math.PI / 180), L * 0.4, 2, [0, 1, 0], 0.1, 0.3);
    limbs.push({ path: tw, r0: 0.04, r1: 0.012, level: 2, vary: r(), tip: true });
  }
  return { height: H, limbs, clumps: [], lod2Levels: 1 };
}

/** Bush: a mound of small clumps on a few hidden stems, berries on some. */
function bush(r: Rng, variant: number): Skeleton {
  const H = SPECIES_INFO.bush.height * r(0.8, 1.05);
  const k = H / 1.6;
  const clumps: Clump[] = [];
  const limbs: Limb[] = [];
  const n = 4 + Math.floor(r(0, 4));
  for (let i = 0; i < n; i++) {
    const a = r(0, 6.283), d = r(0.15, 0.6) * k;
    const s = r(0.45, 0.72) * k;
    const c: V3 = [Math.cos(a) * d, r(0.45, 0.95) * k, Math.sin(a) * d];
    clumps.push({ c, r: [s, s * 0.82, s], vary: r(), seed: (r() * 1e9) | 0 });
    if (i < 3) limbs.push({ path: [[0, -0.2, 0], mul(c, 0.5), c], r0: 0.05, r1: 0.025, level: 2, vary: r() });
  }
  const accents: Clump[] = [];
  if (variant === 0) {
    const cr = crownOf(clumps);
    for (let i = 0; i < 12; i++) {
      const u = norm([r(-1, 1), r(-0.2, 1), r(-1, 1)] as V3);
      accents.push({ c: add(cr.c, mul(u, cr.r * 0.78)), r: [0.06, 0.06, 0.06], vary: r(), seed: (r() * 1e9) | 0 });
    }
  }
  return { height: H, limbs, clumps, accents, lod2Blobs: 1 };
}

function skeleton(species: Species, variant: number): Skeleton {
  const r = rng(0x5eed + 7919 * SPECIES.indexOf(species) + 104729 * variant);
  switch (species) {
    case 'oak': return broadleaf(r, 'oak');
    case 'maple': return broadleaf(r, 'maple');
    case 'birch': return birch(r);
    case 'pine': return pine(r);
    case 'jungle': return jungle(r);
    case 'palm': return palm(r);
    case 'acacia': return acacia(r);
    case 'cactus': return cactus(r);
    case 'dead': return dead(r);
    case 'bush': return bush(r, variant);
  }
}

/** Heights (m, model space) a trunk collider covers: knees, hips (its centre), shoulders. */
const BODY_SPAN = [0.3, 0.9, 1.5];

/**
 * A trunk as the hero bumps into it: a circle round the trunk's centre at hip
 * height wide enough for it from knee to shoulder (flare, buttress fins and
 * lean included; model space, scale 1), and the height its trunk reaches
 * (above that only crowns, which the hero passes through). `prof` follows the
 * trunk up for climbing: PROFILE knots of (height, centre x, z off the circle's,
 * radius), its rings as drawn (at their widest) from knee height to the top.
 */
export type TrunkCollider = { x: number; z: number; r: number; top: number; prof: number[] };

/** Knots of a trunk's climbing profile (TrunkCollider.prof). */
const PROFILE = 8;

/** One collider per (species, shape), index species · VARIANTS + variant; null for bushes. */
export function trunkColliders(): Array<TrunkCollider | null> {
  const out: Array<TrunkCollider | null> = [];
  for (const sp of SPECIES) {
    for (let v = 0; v < VARIANTS; v++) {
      if (sp === 'bush') { out.push(null); continue; }
      const s = skeleton(sp, v);
      const t = s.limbs[0];
      const n = t.path.length;
      // The rings of the detailed mesh (emit, level 0): the same resampled path, each ring at its widest.
      const cactus = s.part === PART.cactus;
      const [path, radii] = resample(t.path, t.radii ?? t.path.map((_, i) => t.r0 + (t.r1 - t.r0) * (i / (n - 1))), s.keepPoints || cactus ? n : DETAIL[0].points[0]);
      const ring = path.map((p, i) => {
        let widest = 1;
        if (t.shaped && s.shape) for (let k = 0; k < 64; k++) widest = Math.max(widest, s.shape((k / 64) * 2 * Math.PI, i, p));
        return radii[i] * widest;
      });
      // Centre and radius at height h, between the rings round it (as the mesh interpolates).
      const at = (h: number) => {
        let i = 1;
        while (i < path.length - 1 && path[i][1] < h) i++;
        const a = path[i - 1], b = path[i];
        const f = clamp01((h - a[1]) / Math.max(b[1] - a[1], 1e-6));
        return { p: add(a, mul(sub(b, a), f)), r: ring[i - 1] + (ring[i] - ring[i - 1]) * f };
      };
      const c = at(BODY_SPAN[1]).p;
      const r = Math.max(...BODY_SPAN.map((h) => { const q = at(h); return Math.hypot(q.p[0] - c[0], q.p[2] - c[2]) + q.r; }));
      const top = t.path[n - 1][1];
      // (the rings themselves from the knees to the top: the trunk tapering and leaning as it is drawn)
      const prof: number[] = [];
      const lo = Math.min(BODY_SPAN[0], top);
      for (let k = 0; k < PROFILE; k++) {
        const h = lo + ((top - lo) * k) / (PROFILE - 1);
        const q = at(h);
        prof.push(h, q.p[0] - c[0], q.p[2] - c[2], q.r);
      }
      out.push({ x: c[0], z: c[2], r, top, prof });
    }
  }
  return out;
}

// ---- emission per level of detail ----------------------------------------------

/** Sides, path points and clump subdivision per LOD for trunk (0), limbs (1) and twigs (2). */
const DETAIL = [
  { sides: [12, 6, 5], points: [8, 5, 4], subdiv: 1, levels: 2, amp: 0.13 },
  { sides: [5, 4, 3], points: [4, 3, 2], subdiv: 0, levels: 1, amp: 0.1 },
  { sides: [4, 3, 3], points: [2, 2, 2], subdiv: 0, levels: 0, amp: 0.06 },
] as const;

/** The point a fraction t of the way along a path of equal steps. */
function along(path: V3[], t: number): V3 {
  const f = clamp01(t) * (path.length - 1);
  const i = Math.min(path.length - 2, Math.floor(f));
  return add(path[i], mul(sub(path[i + 1], path[i]), f - i));
}

/** The share of its width a bough's sheet has a fraction t of the way along: narrower toward both ends, and already broad where it leaves the trunk (the texture's own outline does the rest). */
/** How far a bough's edges hang below its midrib, as a share of its half-width. */
const FOLD = 1;

const sheetWidth = (t: number) => Math.min(1, 0.9 + t, 0.35 + 3.25 * (1 - t));

/**
 * Boughs: each a sheet along its midrib, folded (both halves sag from the midrib), that the shader cuts a
 * feathered bough out of (the 'fir' spray, leafAtlas.ts): the sheet's triangles are the mesh's cards
 * (the second index range, drawn cut out in two steps, forest.ts). Their texture coordinates run
 * across the sheet and along it, 0 to 1. (Far away, a skirt per whorl stands in for them: emitSkirts.)
 */
function emitBoughs(b: Builder, s: Skeleton, lod: number): void {
  const H = s.height;
  if (lod === 2) { emitSkirts(b, s); return; }
  s.whorls!.forEach((whorl) => whorl.forEach((g) => {
    const r = rng(g.seed);
    const rows = lod === 0 ? (g.len > 2.4 ? 6 : g.len > 1.6 ? 4 : 3) : g.len > 3.6 ? 3 : 2;
    const flip = r() < 0.5 ? 1 : 0; // (which side of the texture is which: the boughs are not all alike)
    const side: V3 = [-Math.sin(g.a), 0, Math.cos(g.a)];
    const out: V3 = norm([Math.cos(g.a) * 0.78, 0.62, Math.sin(g.a) * 0.78]); // the crown's own lean: outward and up
    const rootY = g.path[0][1];
    const high = clamp01(rootY / H);
    const ids: number[][] = [];
    for (let i = 0; i <= rows; i++) {
      const t = i / rows;
      const m = along(g.path, t);
      const T = norm(sub(along(g.path, Math.min(1, t + 0.04)), along(g.path, Math.max(0, t - 0.04))));
      const up = norm(cross(side, T)); // the sheet's upper face
      const w = g.width * sheetWidth(t);
      const sway = clamp01(Math.pow(high, 1.5) + 0.1 * t);
      const ao = clamp01((0.42 + 0.58 * smooth(0, 0.8, t)) * (0.6 + 0.4 * Math.pow(high, 0.8)));
      const vary = clamp01(g.vary + r(-0.1, 0.1));
      const v = t * SURFACE_SPAN; // (stored as the fraction of the way along)
      // Folded along the midrib: a ridge there, both halves sagging down from it, more toward the edge.
      const across = (sgn: 1 | -1, f: number): number => {
        const p = add(m, add(mul(side, sgn * f * w), mul(up, -FOLD * Math.pow(f, 1.3) * w)));
        const n = norm(add(mul(norm(add(up, mul(side, sgn * 1.3 * FOLD * Math.pow(f, 0.3)))), 0.7), mul(out, 0.3)));
        return b.vertex(p, n, PART.frond, sway, ao, vary, 0.5 + 0.5 * f * sgn * (2 * flip - 1), v);
      };
      const mid = b.vertex(m, norm(add(mul(up, 0.62), mul(out, 0.38))), PART.frond, sway, ao, vary, 0.5, v);
      ids.push(lod === 0 && g.len > 2.2 ? [across(1, 1), across(1, 0.5), mid, across(-1, 0.5), across(-1, 1)] : [across(1, 1), mid, across(-1, 1)]);
    }
    for (let i = 0; i < rows; i++) {
      for (let c = 0; c + 1 < ids[i].length; c++) {
        const a0 = ids[i][c], a1 = ids[i][c + 1], b0 = ids[i + 1][c], b1 = ids[i + 1][c + 1];
        b.triCard(a0, a1, b0); b.triCard(a1, b1, b0);
      }
    }
  }));
}

/**
 * Far away, a skirt round each whorl stands in for its boughs: a fan from the trunk to a rim through
 * the boughs' tips, opaque, its colour the mean of the feathered boughs it replaces.
 */
function emitSkirts(b: Builder, s: Skeleton): void {
  const H = s.height;
  for (const whorl of s.whorls!) {
    if (whorl.length < 3) continue;
    const root = whorl[0].path[0];
    const reach = whorl.reduce((p, g) => p + Math.hypot(g.path[BOUGH_STEPS][0] - root[0], g.path[BOUGH_STEPS][2] - root[2]), 0) / whorl.length;
    const attach = Math.pow(clamp01(root[1] / H), 1.5);
    const apex = b.vertex(add(root, [0, 0.14 * reach + 0.2, 0]), [0, 1, 0], PART.frond, attach, 0.65, whorl[0].vary, 0.5, 0);
    // (six tips at most: at this range a seventh would not be seen)
    const rim = Array.from({ length: Math.min(whorl.length, 6) }, (_, i) => whorl[Math.floor((i * whorl.length) / Math.min(whorl.length, 6))]).map((g) => {
      const tip = g.path[BOUGH_STEPS];
      const e = norm(sub(tip, root));
      return b.vertex(tip, norm(add(mul([e[0], 0, e[2]], 0.78), [0, 0.62, 0])), PART.frond, clamp01(attach + 0.1), 0.9, g.vary, 0, g.len);
    });
    for (let k = 0; k < rim.length; k++) b.tri(apex, rim[k], rim[(k + 1) % rim.length]);
  }
}

function emitFronds(b: Builder, s: Skeleton, lod: number): void {
  const { base, list } = s.fronds!;
  const keep = [list.length, 7, 5][lod];
  const segs = [7, 4, 2][lod];
  const step = list.length / keep;
  for (let q = 0; q < keep; q++) {
    const f = list[Math.floor(q * step)];
    const r = rng(f.seed);
    const d: V3 = [Math.cos(f.a), 0, Math.sin(f.a)];
    const side: V3 = [-d[2], 0, d[0]];
    const rows: number[][] = [];
    for (let i = 0; i <= segs; i++) {
      const t = i / segs;
      const along = f.len * t;
      const c = add(base, add(mul(d, along * Math.cos(f.e)), [0, along * Math.sin(f.e) - f.droop * along * along / f.len * 1.6, 0]));
      const serr = lod === 0 ? (i % 2 === 0 ? 1.12 : 0.82) : 1;
      const w = f.width * Math.pow(Math.sin(Math.PI * Math.min(t * 1.08 + 0.04, 1)), 0.6) * (1 - 0.3 * t) * serr;
      const fold = 0.35 * w;
      const tan = norm(add(mul(d, Math.cos(f.e)), [0, Math.sin(f.e) - 2 * f.droop * 1.6 * t, 0]));
      const up = norm(cross(side, tan));
      const upN = up[1] < 0 ? mul(up, -1) : up;
      const sway = 0.35 + 0.65 * t;
      const ao = 0.55 + 0.45 * t;
      // Surface coordinates: the side across the frond (½ on the midrib), metres along it.
      const L = b.vertex(add(c, add(mul(side, w), [0, -fold, 0])), norm(add(upN, mul(side, 0.35))), PART.frond, sway, ao, r(), 0, along);
      const M = b.vertex(c, upN, PART.frond, sway, ao, r(), 0.5, along);
      const R = b.vertex(add(c, add(mul(side, -w), [0, -fold, 0])), norm(add(upN, mul(side, -0.35))), PART.frond, sway, ao, r(), 1, along);
      rows.push([L, M, R]);
    }
    for (let i = 0; i < segs; i++) {
      const [l0, m0, r0] = rows[i], [l1, m1, r1] = rows[i + 1];
      b.tri(l0, m0, l1); b.tri(m0, m1, l1);
      b.tri(m0, r0, m1); b.tri(r0, r1, m1);
    }
  }
}

/** One blob round every clump (a bush far away). */
function singleBlob(clumps: Clump[], shrink: number): Clump {
  const c = crownOf(clumps).c;
  let rx = 0, ry = 0;
  for (const k of clumps) {
    rx = Math.max(rx, Math.hypot(k.c[0] - c[0], k.c[2] - c[2]) + Math.max(k.r[0], k.r[2]));
    ry = Math.max(ry, Math.abs(k.c[1] - c[1]) + k.r[1]);
  }
  return { c, r: [rx * shrink, ry * shrink, rx * shrink], vary: clumps[0].vary, seed: clumps[0].seed };
}

/** Ellipsoid surface area (Knud Thomsen's approximation). */
function ellipsoidArea(r: V3): number {
  const p = 1.6075;
  return 4 * Math.PI * Math.pow((Math.pow(r[0] * r[1], p) + Math.pow(r[0] * r[2], p) + Math.pow(r[1] * r[2], p)) / 3, 1 / p);
}

/** Whether point p lies inside clump k shrunk by f. */
const inClump = (k: Clump, p: V3, f: number) => ((p[0] - k.c[0]) / k.r[0]) ** 2 + ((p[1] - k.c[1]) / k.r[1]) ** 2 + ((p[2] - k.c[2]) / k.r[2]) ** 2 < f * f;

/**
 * Leaf cards over a crown: points spread evenly over each clump's surface
 * (a turned Fibonacci sphere), none buried deep inside another clump (those
 * near the seams between clumps stay: they cover the seams). Each card is
 * four vertices at its centre, corners in the surface coordinates; its normal
 * leans out from the crown's middle, so a crown of cards shades as one soft
 * volume.
 */
function emitCards(b: Builder, s: Skeleton, crown: Crown, size: number, lod: number): void {
  const H = s.height;
  for (const k of s.clumps) {
    const r = rng(k.seed ^ 0x5ca1ab1e);
    const n = Math.max(4, Math.round(CARD_DENSITY * (lod === 0 ? 1 : CARDS_FAR) * ellipsoidArea(k.r) / (size * size)));
    const turn = r(0, 2 * Math.PI);
    for (let i = 0; i < n; i++) {
      const z = 1 - (2 * i + 1) / n;
      const ph = turn + i * 2.39996323 + r(-0.25, 0.25);
      const rad = Math.sqrt(1 - z * z);
      const u: V3 = [rad * Math.cos(ph), z, rad * Math.sin(ph)];
      const p = add(k.c, [u[0] * k.r[0], u[1] * k.r[1], u[2] * k.r[2]]);
      if (s.clumps.some((o) => o !== k && inClump(o, p, 0.72))) continue;
      const surface = norm([u[0] / k.r[0], u[1] / k.r[1], u[2] / k.r[2]]);
      const nn = norm(add(mul(norm(sub(p, crown.c)), 0.65), mul(surface, 0.35)));
      const sway = 0.15 + Math.pow(clamp01(p[1] / H), 1.2);
      const ao = leafAO(p, crown), vary = r();
      const v0 = b.count;
      for (const [cx, cy] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) b.vertex(p, nn, PART.card, sway, ao, vary, cx, cy * SURFACE_SPAN);
      b.cardIdx.push(v0, v0 + 1, v0 + 2, v0 + 1, v0 + 3, v0 + 2);
    }
  }
}

/** `card`: leaf-card size (m) for broadleaf crowns, 0 for none. */
function emit(s: Skeleton, lod: number, card = 0): Builder {
  const b = new Builder();
  const D = DETAIL[lod];
  const H = s.height;
  const crown = s.clumps.length ? crownOf(s.clumps) : undefined;
  const cactus = s.part === PART.cactus;
  const levels = lod === 2 ? s.lod2Levels ?? 0 : D.levels;
  for (const l of s.limbs) {
    if (l.level > levels) continue;
    const n = l.path.length;
    const radii = l.radii ?? l.path.map((_, i) => l.r0 + (l.r1 - l.r0) * (i / (n - 1)));
    const points = lod === 0 && s.keepPoints ? n : cactus ? [n, 6, 4][lod] : D.points[l.level];
    const [path, rr] = resample(l.path, radii, points);
    tube(b, path, rr, {
      part: s.part ?? PART.bark, sides: cactus ? [16, 8, 5][lod] : D.sides[l.level], height: H, crown: crown ?? s.shade, crownShade: s.shadeDepth, vary: l.vary,
      tip: l.tip && lod < 2,
      shape: l.shaped && (lod < 2 || cactus) ? s.shape : undefined,
      along0: l.level === 0 ? 0 : 8 * l.vary,
    });
  }
  if (s.whorls) emitBoughs(b, s, lod);
  if (s.fronds) emitFronds(b, s, lod);
  if (crown) {
    const list = lod < 2 ? s.clumps : s.lod2Blobs === 1 ? [singleBlob(s.clumps, 0.92)] : mergeClumps(s.clumps, 0.9);
    for (const k of list) clump(b, lod === 1 ? { ...k, r: mul(k.r, 1.06) } : k, D.subdiv, D.amp, crown, H);
    if (card > 0 && lod < 2) emitCards(b, s, crown, card, lod);
  }
  if (s.accents && lod === 0) for (const a of s.accents) clump(b, a, 0, 0.05, crown ?? { c: a.c, r: 1 }, H, PART.accent);
  return b;
}

// ---- packing ---------------------------------------------------------------------

/** A mesh's triangles, then its leaf cards' (cardFirst = firstIndex + indexCount; cardCount 0 without cards). */
type MeshRange = { firstIndex: number; indexCount: number; cardFirst: number; cardCount: number; baseVertex: number; vertexCount: number };
export type TreeModels = {
  /** VERTEX_BYTES per vertex. */
  vertices: ArrayBuffer;
  indices: Uint32Array<ArrayBuffer>;
  /** Index (species · VARIANTS + variant) · LODS + lod. */
  meshes: MeshRange[];
  /** Per species, over every shape and level: highest point and reach from the trunk axis (m, scale 1). */
  bounds: Array<{ top: number; radius: number }>;
};

/** Every species' shapes at every level of detail, in one vertex and one index buffer. */
export function buildTreeModels(): TreeModels {
  const built: Builder[] = [];
  for (const sp of SPECIES) {
    for (let v = 0; v < VARIANTS; v++) {
      const s = skeleton(sp, v);
      for (let l = 0; l < LODS; l++) built.push(emit(s, l, SPECIES_INFO[sp].card));
    }
  }
  const nv = built.reduce((p, b) => p + b.count, 0);
  const ni = built.reduce((p, b) => p + b.idx.length + b.cardIdx.length, 0);
  const vertices = new ArrayBuffer(nv * VERTEX_BYTES);
  const f32 = new Float32Array(vertices);
  const i8 = new Int8Array(vertices);
  const u8 = new Uint8Array(vertices);
  const u16 = new Uint16Array(vertices);
  const indices = new Uint32Array(ni);
  const meshes: MeshRange[] = [];
  const bounds = SPECIES.map(() => ({ top: 0, radius: 0 }));
  let vo = 0, io = 0;
  built.forEach((b, mi) => {
    const bd = bounds[Math.floor(mi / (VARIANTS * LODS))];
    for (let i = 0; i < b.count; i++) {
      bd.top = Math.max(bd.top, b.pos[3 * i + 1]);
      bd.radius = Math.max(bd.radius, Math.hypot(b.pos[3 * i], b.pos[3 * i + 2]));
    }
    for (let i = 0; i < b.count; i++) {
      const o = (vo + i) * VERTEX_BYTES;
      f32[o / 4] = b.pos[3 * i]; f32[o / 4 + 1] = b.pos[3 * i + 1]; f32[o / 4 + 2] = b.pos[3 * i + 2];
      for (let c = 0; c < 3; c++) i8[o + 12 + c] = Math.max(-127, Math.min(127, Math.round(b.nrm[3 * i + c] * 127)));
      i8[o + 15] = 0;
      u8[o + 16] = b.att[4 * i];
      for (let c = 1; c < 4; c++) u8[o + 16 + c] = Math.round(b.att[4 * i + c] * 255);
      u16[o / 2 + 10] = Math.round(b.uv[2 * i] * 65535);
      u16[o / 2 + 11] = Math.round(b.uv[2 * i + 1] * 65535);
    }
    indices.set(b.idx, io);
    indices.set(b.cardIdx, io + b.idx.length);
    meshes.push({ firstIndex: io, indexCount: b.idx.length, cardFirst: io + b.idx.length, cardCount: b.cardIdx.length, baseVertex: vo, vertexCount: b.count });
    vo += b.count;
    io += b.idx.length + b.cardIdx.length;
  });
  return { vertices, indices, meshes, bounds };
}
