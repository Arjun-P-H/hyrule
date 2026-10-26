/**
 * Model isles: sky islands baked from models (scripts/sky-castle/build.py, scripts/bake-sky-model.py), floating over the land.
 * ---------------------------------------------------------------------------
 * One candidate per SKY_MODELS.cell square, on the land's seeds only, placed by hash: which model,
 * where in its cell, its plaza's height, its turn about the vertical. The renderer (gpu/modelIsles.ts)
 * loads the models (loadSkyModels) and draws them; once a model is loaded its floors hold the hero
 * (skyworld/realm.ts skyFloorUnder): its baked floor layers, the highest under a point that the point
 * is above (or within reach of); and its walls stop him (collideModelIsle): a floor is baked blocked where
 * something stands within a body above it.
 */

import { SKY_MODELS as M } from './constants';
import { hash3, hashU } from './noise';
import { worldOf } from './world';

type V3 = [number, number, number];
/** A mesh: 20-byte vertices over its own box (scripts/skymodel_pack.py) and u32 indices. */
export type SkyMesh = { verts: Uint8Array; index: Uint32Array; lo: V3; hi: V3 };
/** A baked model (the .bin's header and sections: scripts/skymodel_pack.py). */
export type SkyModel = {
  boxMin: V3;
  boxMax: V3;
  /** A base colour texture (<name>.webp) on its uvs, else its vertex colours alone. */
  texture: boolean;
  /**
   * Its parts, each drawn within `near` m (null: always), levels of detail finest first: solids (the exterior, an
   * interior), leaf cards (ivy: the land's leaf sprays) and water (over pools' beds: uv.x its depth / 16 m).
   */
  parts: Array<{ kind: 'solid' | 'leaf' | 'water'; near: number | null; lods: SkyMesh[] }>;
  /** Plants of the land's forest standing on it (forest/species.ts species, skyworld/biomes.ts SKY_FLORA palette). */
  plants: Array<{ species: string; p: V3; scale: number; palette: string }>;
  /** Waterfalls off it: lip (the water's surface), direction (x, z), width at the lip, metres fallen when all mist. */
  falls: Array<{ lip: V3; dir: [number, number]; width: number; reach: number }>;
  /** Instanced pieces (pinnacles, balusters), levels of detail finest first. */
  protos: Array<{ name: string; lods: SkyMesh[] }>;
  /** Their instances, 32 bytes each, sorted into chunks (a prototype, a range, a bounding sphere in the model's axes). */
  props: { data: Uint8Array; count: number; chunks: Array<{ proto: number; first: number; count: number; c: V3; r: number }> };
  /** Every level surface under each cell of a grid, highest first: cell c's run from off[c] to off[c + 1] in h. */
  floor: { nx: number; nz: number; cell: number; x0: number; z0: number; min: number; max: number; off: Uint32Array; h: Uint16Array };
};

/** A model isle: its model (SKY_MODELS.names), its middle and plaza (world m), its turn (rad). */
export type ModelIsle = {
  model: number;
  x: number; y: number; z: number;
  yaw: number;
  /** The floor a query last asked for lies at or under this height (skyFloorUnder sets it, floorHeight reads it). */
  qy: number;
};

type MeshInfo = { verts: number; tris: number; lo: V3; hi: V3 };

/** Parse a baked model (.bin, unzipped: readSkyModel). */
function parseSkyModel(buf: ArrayBuffer): SkyModel {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x4d594b53) throw new Error('not a sky model (SKYM)');
  const hl = dv.getUint32(4, true);
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  if (h.version !== 2) throw new Error(`sky model version ${h.version}: bake it again`);
  let o = 8 + hl;
  const take = (bytes: number) => { const at = o; o += Math.ceil(bytes / 4) * 4; return at; };
  const mesh = (l: MeshInfo): SkyMesh => ({
    verts: new Uint8Array(buf, take(l.verts * 20), l.verts * 20),
    index: new Uint32Array(buf, take(l.tris * 12), l.tris * 3),
    lo: l.lo, hi: l.hi,
  });
  const parts = h.parts.map((p: { kind?: string; near: number | null; lods: MeshInfo[] }) => ({ kind: p.kind ?? 'solid', near: p.near, lods: p.lods.map(mesh) }));
  const protos = h.protos.map((p: { name: string; lods: MeshInfo[] }) => ({ name: p.name, lods: p.lods.map(mesh) }));
  const props = { ...h.props, data: new Uint8Array(buf, take(h.props.count * 32), h.props.count * 32) };
  const f = h.floor;
  const cells = f.nx * f.nz + 1;
  const at = take(cells * 4 + f.count * 2);
  const floor = { ...f, off: new Uint32Array(buf, at, cells), h: new Uint16Array(buf, at + cells * 4, f.count) };
  return { boxMin: h.boxMin, boxMax: h.boxMax, texture: h.texture, parts, protos, props, floor, plants: h.plants ?? [], falls: h.falls ?? [] };
}

/** A baked model from its file's bytes (gzipped, as the bake writes it). */
export async function readSkyModel(buf: ArrayBuffer): Promise<SkyModel> {
  const raw = new Uint8Array(buf);
  const plain = raw[0] === 0x1f && raw[1] === 0x8b
    ? await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
    : buf;
  return parseSkyModel(plain);
}

const models: Array<SkyModel | undefined> = [];
let loading: Promise<SkyModel[]> | null = null;

/** Fetch every model (once), and let their floors hold the hero. */
export function loadSkyModels(): Promise<SkyModel[]> {
  return (loading ??= Promise.all(M.names.map(async (name, k) => {
    const r = await fetch(`/models/${name}.bin`);
    if (!r.ok) throw new Error(`sky model ${name}: ${r.status}`);
    return (models[k] = await readSkyModel(await r.arrayBuffer()));
  })));
}

/** Give model k its floors without fetching (tests, which read the .bin themselves). */
export function setSkyModel(k: number, m: SkyModel | undefined): void {
  models[k] = m;
}

const cache = new Map<number, ModelIsle | null>();
let cacheSeed = NaN;

/** The model isle of cell (ix, iz), or null (none there, or not the land). */
export function modelIsleInCell(ix: number, iz: number, seed: number): ModelIsle | null {
  if (seed !== cacheSeed) { cache.clear(); cacheSeed = seed; }
  const key = (ix & 0xffff) * 65536 + (iz & 0xffff);
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  const h = hash3(ix, iz, seed ^ 0x5ca1ab1e);
  const u = (k: number) => hashU(h + Math.imul(k, 0x9e3779b9)) / 4294967296;
  let isle: ModelIsle | null = null;
  if (worldOf(seed) === 'land' && u(0) < M.probability) {
    const span = M.cell - 2 * M.margin;
    isle = {
      model: Math.floor(u(1) * M.names.length),
      x: ix * M.cell + M.margin + u(2) * span,
      y: M.plaza[0] + u(3) * (M.plaza[1] - M.plaza[0]),
      z: iz * M.cell + M.margin + u(4) * span,
      yaw: u(5) * Math.PI * 2,
      qy: Infinity,
    };
  }
  if (cache.size >= 1024) cache.delete(cache.keys().next().value!);
  cache.set(key, isle);
  return isle;
}

/** The model isles whose middles lie within `range` of (x, z), into `out` (emptied first). */
export function modelIslesNear(x: number, z: number, range: number, seed: number, out: ModelIsle[]): ModelIsle[] {
  out.length = 0;
  const c = M.cell;
  for (let iz = Math.floor((z - range) / c); iz <= Math.floor((z + range) / c); iz++) {
    for (let ix = Math.floor((x - range) / c); ix <= Math.floor((x + range) / c); ix++) {
      const m = modelIsleInCell(ix, iz, seed);
      if (m && Math.hypot(m.x - x, m.z - z) <= range) out.push(m);
    }
  }
  return out;
}

/** The isle's highest floor at world (x, z) at or under height `below` (world m), else NaN (off it, or its model not loaded). */
export function modelTop(isle: ModelIsle, x: number, z: number, below: number): number {
  const m = models[isle.model];
  if (!m) return NaN;
  const F = m.floor;
  const dx = x - isle.x, dz = z - isle.z;
  const c = Math.cos(isle.yaw), s = Math.sin(isle.yaw);
  // (the model's axes: the inverse of the renderer's turn, modelIsles.wgsl.ts miTurn)
  const ci = Math.floor((c * dx - s * dz - F.x0) / F.cell);
  const cj = Math.floor((s * dx + c * dz - F.z0) / F.cell);
  if (ci < 0 || cj < 0 || ci >= F.nx || cj >= F.nz) return NaN;
  const at = cj * F.nx + ci;
  for (let k = F.off[at]; k < F.off[at + 1]; k++) {
    const y = layerY(isle, F, F.h[k]);
    if (y <= below) return y;
  }
  return NaN;
}

/** A floor layer's height (world m): its upper 15 bits over the floors' range (its low bit: blocked). */
const layerY = (isle: ModelIsle, F: SkyModel['floor'], q: number) => isle.y + F.min + (((q >> 1) - 1) / 32766) * (F.max - F.min);

/** Whether a body with its feet at `feetY` (world m) is blocked in floor cell (ci, cj): the floor it would stand on there, within a step, is baked blocked. */
function cellBlocked(isle: ModelIsle, F: SkyModel['floor'], ci: number, cj: number, feetY: number, step: number): boolean {
  if (ci < 0 || cj < 0 || ci >= F.nx || cj >= F.nz) return false;
  const c = cj * F.nx + ci;
  for (let k = F.off[c]; k < F.off[c + 1]; k++) {
    if (layerY(isle, F, F.h[k]) <= feetY + step) return (F.h[k] & 1) === 1;
  }
  return false;
}

/**
 * A model isle's walls: push a body (world x, z and velocity) whose feet are at `feetY` out of the floor cells
 * blocked for it within `radius`, each a square of the model's grid (and drop the velocity into the face it meets,
 * so it slides along walls). Nothing happens off an isle or before its model is loaded.
 */
export function collideModelIsle(s: { x: number; z: number; vx: number; vz: number }, feetY: number, seed: number, radius: number, step: number): void {
  const isle = modelIsleInCell(Math.floor(s.x / M.cell), Math.floor(s.z / M.cell), seed);
  const m = isle && models[isle.model];
  if (!isle || !m) return;
  const F = m.floor;
  const c = Math.cos(isle.yaw), sn = Math.sin(isle.yaw);
  // the body in the grid's units (cells), in the model's axes
  const dx = s.x - isle.x, dz = s.z - isle.z;
  let u = (c * dx - sn * dz - F.x0) / F.cell, v = (sn * dx + c * dz - F.z0) / F.cell;
  if (u < -1 || v < -1 || u > F.nx + 1 || v > F.nz + 1) return;
  const r = radius / F.cell;
  let pu = 0, pv = 0;
  for (let cj = Math.floor(v - r); cj <= Math.floor(v + r); cj++) {
    for (let ci = Math.floor(u - r); ci <= Math.floor(u + r); ci++) {
      if (!cellBlocked(isle, F, ci, cj, feetY, step)) continue;
      const du = u - Math.max(ci, Math.min(ci + 1, u)), dv = v - Math.max(cj, Math.min(cj + 1, v));
      const d = Math.hypot(du, dv);
      if (d >= r) continue;
      if (d > 1e-6) {
        pu += (du / d) * (r - d); pv += (dv / d) * (r - d);
        u += (du / d) * (r - d); v += (dv / d) * (r - d);
      } else {
        // (the middle inside the cell: out by its nearest face, to a radius beyond it)
        const faces = [u - ci, ci + 1 - u, v - cj, cj + 1 - v];
        const k = faces.indexOf(Math.min(...faces));
        const d2 = (faces[k] + r) * (k % 2 ? 1 : -1);
        if (k < 2) { pu += d2; u += d2; } else { pv += d2; v += d2; }
      }
    }
  }
  if (pu === 0 && pv === 0) return;
  // back to the world's axes (the turn's inverse: the renderer's miTurn)
  const wx = (c * pu + sn * pv) * F.cell, wz = (-sn * pu + c * pv) * F.cell;
  s.x += wx;
  s.z += wz;
  const L = Math.hypot(wx, wz), nx = wx / L, nz = wz / L;
  const vn = s.vx * nx + s.vz * nz;
  if (vn < 0) { s.vx -= vn * nx; s.vz -= vn * nz; }
}

/** The model isle with a floor under (x, z) that a point at height y is above, or less than `reach` below; else null. */
export function modelIsleUnder(x: number, y: number, z: number, seed: number, reach: number): ModelIsle | null {
  const isle = modelIsleInCell(Math.floor(x / M.cell), Math.floor(z / M.cell), seed);
  if (!isle || !models[isle.model]) return null;
  const t = modelTop(isle, x, z, y + reach);
  if (t !== t) return null;
  isle.qy = y + reach;
  return isle;
}
