/**
 * The rocks' meshes, made once at start-up (there are no model files, as the trees': forest/models.ts).
 * ---------------------------------------------------------------------------
 * Each kind is an icosphere pulled into an ellipsoid, cut by planes (which leave flat faces and sharp
 * edges, the way stone breaks), roughened by two octaves of noise and sat on a flat base. Three levels of
 * detail share a kind's planes, so a rock keeps its shape as it recedes. The mesh is drawn flat-shaded
 * (every face its own three vertices), each vertex's normal its face's softened toward the smooth normal
 * a third of the way, so the facets read without the sparkle of hard edges. A vertex also carries its
 * height over the base (the dirt the rock sits in, the moss on its top) and an occlusion from the noise's
 * hollows. The unit mesh has a radius of about 1: an instance scales it to its size.
 *
 *   boulder  round, three soft cuts          crag    angular, seven hard ones
 *   slab     flat and broad                  tall    a standing stone
 *   block    cut square                      shard   a tapering spike
 *   stone    a smooth river stone            cluster a handful of pebbles in one mesh
 */

export const ROCK_KINDS = ['boulder', 'crag', 'slab', 'tall', 'block', 'shard', 'stone', 'cluster'] as const;
export type RockKind = (typeof ROCK_KINDS)[number];
export const ROCK_LODS = 3;
/** Bytes a vertex takes: position (3 × f32), normal (4 × snorm8), attributes (4 × unorm8: occlusion, height, hollow, unused). */
export const ROCK_VERTEX_BYTES = 20;

type V3 = [number, number, number];

/** mulberry32: a small seeded generator of [0, 1) values. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const hash3i = (x: number, y: number, z: number, s: number): number => {
  let h = Math.imul(x, 0x27d4eb2f) ^ Math.imul(y, 0x165667b1) ^ Math.imul(z, 0x9e3779b9) ^ Math.imul(s, 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
};

/** Value noise on a 3-D lattice, −1 … 1 (quintic). */
function noise3(x: number, y: number, z: number, seed: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const u = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  const ux = u(fx), uy = u(fy), uz = u(fz);
  const c = (a: number, b: number, d: number) => hash3i(ix + a, iy + b, iz + d, seed);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const x00 = lerp(c(0, 0, 0), c(1, 0, 0), ux), x10 = lerp(c(0, 1, 0), c(1, 1, 0), ux);
  const x01 = lerp(c(0, 0, 1), c(1, 0, 1), ux), x11 = lerp(c(0, 1, 1), c(1, 1, 1), ux);
  return 2 * lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz) - 1;
}

/** An icosphere's unit vertices and faces after `sub` subdivisions. */
function icosphere(sub: number): { v: V3[]; f: Array<[number, number, number]> } {
  const t = (1 + Math.sqrt(5)) / 2;
  const v: V3[] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  for (const p of v) { const l = Math.hypot(...p); p[0] /= l; p[1] /= l; p[2] /= l; }
  let f: Array<[number, number, number]> = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  for (let s = 0; s < sub; s++) {
    const cache = new Map<number, number>();
    const mid = (a: number, b: number): number => {
      const k = a < b ? a * 65536 + b : b * 65536 + a;
      let m = cache.get(k);
      if (m === undefined) {
        const p: V3 = [(v[a][0] + v[b][0]) / 2, (v[a][1] + v[b][1]) / 2, (v[a][2] + v[b][2]) / 2];
        const l = Math.hypot(...p);
        v.push([p[0] / l, p[1] / l, p[2] / l]);
        m = v.length - 1;
        cache.set(k, m);
      }
      return m;
    };
    const next: Array<[number, number, number]> = [];
    for (const [a, b, c] of f) {
      const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    f = next;
  }
  return { v, f };
}

type Shape = {
  /** Ellipsoid radii. */
  r: V3;
  /** Planar cuts: how many, where along each (a share of the ellipsoid's reach: lo … hi) and how softly (m of rounding). */
  cuts: number; lo: number; hi: number; soft: number;
  /** Noise octaves: [frequency, amplitude]. */
  noise: Array<[number, number]>;
  /** The base's height (a share of the −y radius kept) and a taper of the top (0 none … 1 to a point). */
  base: number; taper: number;
  /** Bias of the cuts' normals toward the horizontal (1: all round the sides, as a standing stone's). */
  side: number;
};

const SHAPES: Record<Exclude<RockKind, 'cluster'>, Shape> = {
  boulder: { r: [1.0, 0.84, 0.95], cuts: 4, lo: 0.62, hi: 0.8, soft: 0.2, noise: [[0.9, 0.15], [2.3, 0.07]], base: 0.55, taper: 0, side: 0.2 },
  crag: { r: [1.0, 0.92, 0.86], cuts: 10, lo: 0.5, hi: 0.7, soft: 0.0, noise: [[1.2, 0.1], [3.1, 0.05]], base: 0.55, taper: 0, side: 0.3 },
  slab: { r: [1.4, 0.48, 1.08], cuts: 5, lo: 0.62, hi: 0.82, soft: 0.05, noise: [[1.1, 0.06], [2.8, 0.03]], base: 0.6, taper: 0, side: 0.7 },
  tall: { r: [0.66, 1.3, 0.78], cuts: 5, lo: 0.6, hi: 0.8, soft: 0.05, noise: [[1.0, 0.07], [2.6, 0.035]], base: 0.7, taper: 0.18, side: 0.8 },
  block: { r: [0.94, 0.86, 0.9], cuts: 8, lo: 0.42, hi: 0.54, soft: 0.0, noise: [[1.4, 0.06], [3.4, 0.03]], base: 0.6, taper: 0, side: 0.5 },
  shard: { r: [0.52, 1.5, 0.46], cuts: 4, lo: 0.55, hi: 0.75, soft: 0.0, noise: [[1.3, 0.05], [3.0, 0.025]], base: 0.7, taper: 0.62, side: 0.9 },
  stone: { r: [1.1, 0.66, 0.86], cuts: 1, lo: 0.8, hi: 0.9, soft: 0.5, noise: [[1.0, 0.05], [2.4, 0.02]], base: 0.5, taper: 0, side: 0.0 },
};

type Mesh = { pos: Float32Array; nrm: Int8Array; attr: Uint8Array; count: number };

/** A rock's planes (unit normals and offsets), the same at every level of detail. */
function planesOf(shape: Shape, rand: () => number): Array<{ n: V3; c: number }> {
  const out: Array<{ n: V3; c: number }> = [];
  for (let k = 0; k < shape.cuts; k++) {
    let y = (rand() * 2 - 1) * (1 - shape.side * 0.85);
    // (not a cut straight from below: the base is its own)
    y = Math.max(y, -0.3);
    const a = rand() * Math.PI * 2, s = Math.sqrt(Math.max(1 - y * y, 0));
    const n: V3 = [Math.cos(a) * s, y, Math.sin(a) * s];
    out.push({ n, c: shape.lo + (shape.hi - shape.lo) * rand() });
  }
  return out;
}

function shapeVertex(dir: V3, shape: Shape, planes: Array<{ n: V3; c: number }>, seed: number): { p: V3; hollow: number } {
  let x = dir[0] * shape.r[0], y = dir[1] * shape.r[1], z = dir[2] * shape.r[2];
  // taper toward a point at the top
  if (shape.taper > 0) {
    const t = Math.max(0, (y / shape.r[1] + 0.2) / 1.2);
    const k = 1 - shape.taper * t * t;
    x *= k; z *= k;
  }
  // planar cuts: what lies past a plane is pressed back onto it (softly, if the shape is round)
  for (const { n, c } of planes) {
    const reach = Math.abs(n[0] * shape.r[0]) + Math.abs(n[1] * shape.r[1]) + Math.abs(n[2] * shape.r[2]);
    const d = x * n[0] + y * n[1] + z * n[2] - c * reach;
    if (d > 0) {
      // (a soft cut presses back less near its edge, so the edge is rounded over about `soft`)
      const eased = shape.soft > 0 ? d * (1 - 0.5 * Math.exp(-d / shape.soft)) : d;
      x -= n[0] * eased; y -= n[1] * eased; z -= n[2] * eased;
    }
  }
  // roughness along the direction from the middle
  let nz = 0;
  for (let o = 0; o < shape.noise.length; o++) {
    const [f, a] = shape.noise[o];
    nz += a * noise3(x * f + 11.3, y * f - 4.7, z * f + 2.1, seed + o * 977);
  }
  const l = Math.hypot(x, y, z) || 1;
  x += (x / l) * nz; y += (y / l) * nz; z += (z / l) * nz;
  // the flat base
  const floor = -shape.base * shape.r[1];
  if (y < floor) y = floor;
  return { p: [x, y, z], hollow: nz };
}

/** One level of a kind (sub = icosphere subdivisions), flat-shaded: three vertices a face. */
function buildMesh(kind: Exclude<RockKind, 'cluster'>, sub: number, seed: number): Mesh {
  const shape = SHAPES[kind];
  const rand = rng(seed * 7919 + 13);
  const planes = planesOf(shape, rand);
  const ico = icosphere(sub);
  // a little jitter of the vertices (in the sphere's own directions) so the tessellation is not a regular lattice
  const jr = rng(seed * 104729 + 7);
  const dirs = ico.v.map((d): V3 => {
    const j = 0.05 / (1 << sub);
    const p: V3 = [d[0] + (jr() - 0.5) * j * 4, d[1] + (jr() - 0.5) * j * 4, d[2] + (jr() - 0.5) * j * 4];
    const l = Math.hypot(...p);
    return [p[0] / l, p[1] / l, p[2] / l];
  });
  const shaped = dirs.map((d) => shapeVertex(d, shape, planes, seed));
  const P = shaped.map((s) => s.p);
  // smooth normals (area-weighted face normals), and the faces' own
  const sm: V3[] = P.map(() => [0, 0, 0]);
  const fn: V3[] = [];
  for (const [a, b, c] of ico.f) {
    const ux = P[b][0] - P[a][0], uy = P[b][1] - P[a][1], uz = P[b][2] - P[a][2];
    const vx = P[c][0] - P[a][0], vy = P[c][1] - P[a][1], vz = P[c][2] - P[a][2];
    const n: V3 = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
    for (const i of [a, b, c]) { sm[i][0] += n[0]; sm[i][1] += n[1]; sm[i][2] += n[2]; }
    const l = Math.hypot(...n) || 1;
    fn.push([n[0] / l, n[1] / l, n[2] / l]);
  }
  for (const n of sm) { const l = Math.hypot(...n) || 1; n[0] /= l; n[1] /= l; n[2] /= l; }
  // extent for the height share
  let lo = Infinity, hi = -Infinity;
  for (const p of P) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
  const count = ico.f.length * 3;
  const pos = new Float32Array(count * 3), nrm = new Int8Array(count * 4), attr = new Uint8Array(count * 4);
  let o = 0;
  ico.f.forEach(([a, b, c], fi) => {
    for (const i of [a, b, c]) {
      pos[o * 3] = P[i][0]; pos[o * 3 + 1] = P[i][1]; pos[o * 3 + 2] = P[i][2];
      // the face's normal, softened toward the smooth one a third of the way
      const n: V3 = [fn[fi][0] * 0.68 + sm[i][0] * 0.32, fn[fi][1] * 0.68 + sm[i][1] * 0.32, fn[fi][2] * 0.68 + sm[i][2] * 0.32];
      const l = Math.hypot(...n) || 1;
      nrm[o * 4] = Math.round((n[0] / l) * 127); nrm[o * 4 + 1] = Math.round((n[1] / l) * 127); nrm[o * 4 + 2] = Math.round((n[2] / l) * 127);
      const h = (P[i][1] - lo) / Math.max(hi - lo, 1e-6);
      const hollow = shaped[i].hollow;
      // occlusion: the base is shaded by the ground, hollows by their walls
      const ao = Math.min(1, Math.max(0, 0.55 + 0.45 * Math.min(1, h * 2.2) + 0.9 * hollow));
      attr[o * 4] = Math.round(ao * 255);
      attr[o * 4 + 1] = Math.round(Math.min(1, Math.max(0, h)) * 255);
      attr[o * 4 + 2] = Math.round(Math.min(1, Math.max(0, 0.5 + 2.5 * hollow)) * 255);
      attr[o * 4 + 3] = 255;
      o++;
    }
  });
  return { pos, nrm, attr, count };
}

/** A handful of pebbles in one mesh: small stones on a disc, the biggest first. */
function buildCluster(sub: number, seed: number): Mesh {
  const rand = rng(seed * 31337 + 5);
  const n = sub >= 1 ? 6 : 4;
  const parts: Mesh[] = [];
  for (let i = 0; i < n; i++) {
    const m = buildMesh(rand() < 0.5 ? 'stone' : 'block', Math.max(sub - 1, 0), seed * 101 + i);
    const a = rand() * Math.PI * 2, r = 0.2 + 0.65 * Math.sqrt(rand());
    const s = (i === 0 ? 0.42 : 0.2 + 0.2 * rand()) * (1 - 0.35 * (i / n));
    const ox = Math.cos(a) * r * (i === 0 ? 0.3 : 1), oz = Math.sin(a) * r * (i === 0 ? 0.3 : 1);
    const ya = rand() * Math.PI * 2, cy = Math.cos(ya), sy = Math.sin(ya);
    for (let v = 0; v < m.count; v++) {
      const x = m.pos[v * 3], y = m.pos[v * 3 + 1], z = m.pos[v * 3 + 2];
      m.pos[v * 3] = (cy * x + sy * z) * s + ox;
      m.pos[v * 3 + 1] = y * s + s * 0.5 - 0.12;
      m.pos[v * 3 + 2] = (-sy * x + cy * z) * s + oz;
      const nx = m.nrm[v * 4], nz = m.nrm[v * 4 + 2];
      m.nrm[v * 4] = Math.round(cy * nx + sy * nz); m.nrm[v * 4 + 2] = Math.round(-sy * nx + cy * nz);
    }
    parts.push(m);
  }
  const count = parts.reduce((s, m) => s + m.count, 0);
  const pos = new Float32Array(count * 3), nrm = new Int8Array(count * 4), attr = new Uint8Array(count * 4);
  let o = 0;
  for (const m of parts) { pos.set(m.pos, o * 3); nrm.set(m.nrm, o * 4); attr.set(m.attr, o * 4); o += m.count; }
  return { pos, nrm, attr, count };
}

export type RockModels = {
  /** All the meshes' vertices, ROCK_VERTEX_BYTES each: kind-major, then level of detail (0 the finest). */
  vertices: ArrayBuffer;
  /** Per mesh (kind × ROCK_LODS + lod): its first vertex and vertex count. */
  meshes: Array<{ first: number; count: number }>;
  /** Per kind: the largest horizontal and vertical radius of its finest mesh, its top and its flat base's height (unit scale), for colliders, culling and setting it on the ground. */
  extent: Array<{ rxz: number; ry: number; top: number; bottom: number }>;
};

const SUBS: readonly number[] = [2, 1, 0];

export function buildRockModels(): RockModels {
  const meshes: Mesh[] = [];
  const extent: RockModels['extent'] = [];
  ROCK_KINDS.forEach((kind, ki) => {
    SUBS.forEach((sub, lod) => {
      meshes.push(kind === 'cluster' ? buildCluster(sub + (lod === 2 ? 0 : 1), ki + 1) : buildMesh(kind, sub, ki + 1));
      if (lod === 0) {
        const m = meshes[meshes.length - 1];
        let rxz = 0, ry = 0, top = -Infinity, bottom = Infinity;
        for (let v = 0; v < m.count; v++) {
          rxz = Math.max(rxz, Math.hypot(m.pos[v * 3], m.pos[v * 3 + 2]));
          ry = Math.max(ry, Math.abs(m.pos[v * 3 + 1]));
          top = Math.max(top, m.pos[v * 3 + 1]);
          bottom = Math.min(bottom, m.pos[v * 3 + 1]);
        }
        extent.push({ rxz, ry, top, bottom });
      }
    });
  });
  const total = meshes.reduce((s, m) => s + m.count, 0);
  const buf = new ArrayBuffer(total * ROCK_VERTEX_BYTES);
  const dv = new DataView(buf);
  const info: RockModels['meshes'] = [];
  let first = 0;
  for (const m of meshes) {
    for (let v = 0; v < m.count; v++) {
      const o = (first + v) * ROCK_VERTEX_BYTES;
      dv.setFloat32(o, m.pos[v * 3], true); dv.setFloat32(o + 4, m.pos[v * 3 + 1], true); dv.setFloat32(o + 8, m.pos[v * 3 + 2], true);
      for (let c = 0; c < 4; c++) dv.setInt8(o + 12 + c, m.nrm[v * 4 + c]);
      for (let c = 0; c < 4; c++) dv.setUint8(o + 16 + c, m.attr[v * 4 + c]);
    }
    info.push({ first, count: m.count });
    first += m.count;
  }
  return { vertices: buf, meshes: info, extent };
}
