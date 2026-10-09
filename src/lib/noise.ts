/**
 * The noise primitives every twin mirrors: the uint32 Murmur3 lattice hash (spec §4) and the C²
 * quintic value noise (spec §5). Their own module so the world's geography (hydro.ts) and the
 * world function (terrainMath.ts, which re-exports them) share them without an import cycle.
 */

/* ------------------------------------------------------------------ */
/* §4 — Seed & deterministic hashing (uint32, Murmur3 finalizer)       */
/* ------------------------------------------------------------------ */

/** §4 — MurmurHash3 finalizer — avalanche. Input/output are uint32 bit patterns. */
export function hashU(x: number): number {
  let h = x | 0;
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * §4 — H(x_i, y_i, s) = finalize(x_i ⊕ (y_i · 0x27D4EB2F) ⊕ (s · 0x9E3779B9))
 * xi/yi are integer lattice coords (may be negative — two's-complement wrap).
 */
export function hash3(xi: number, yi: number, seed: number): number {
  const h =
    (xi | 0) ^
    Math.imul(yi | 0, 0x27d4eb2f) ^
    Math.imul(seed | 0, 0x9e3779b9);
  return hashU(h);
}

const INV_U32 = 1.0 / 4294967295.0;
const INV_U32_X2 = 2.0 / 4294967295.0;

/** §4 — Uniform in [0,1] from an integer lattice cell. */
export function hash01(xi: number, yi: number, seed: number): number {
  return hash3(xi, yi, seed) * INV_U32;
}

/** §4 — Feature-point coords in [0,1]² packed from the two 12-bit halves of one hash. */
export function cellPoint(ix: number, iy: number, seed: number): [number, number] {
  const h = hash3(ix, iy, seed);
  return [(h & 0xfff) / 4095.0, ((h >>> 12) & 0xfff) / 4095.0];
}

/** §4 — Cell type byte packed from the top 8 bits of the same hash (0..1). */
export function cellType(ix: number, iy: number, seed: number): number {
  return (hash3(ix, iy, seed) >>> 24) / 255.0;
}

/* ------------------------------------------------------------------ */
/* §5 — Quintic value noise (C² continuous) & Nyquist weights          */
/* ------------------------------------------------------------------ */

/** §5 — Quintic C² polynomial S(t) = 6t⁵ − 15t⁴ + 10t³ evaluated via Horner's rule. */
export function quintic(t: number): number {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

/** §5 — Quintic value noise with bilinear interpolation and Dyadic Murmur3 lattice. */
export function vnoise(x: number, y: number, seed: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
  const uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

  const ux0 = ix | 0;
  const ux1 = (ux0 + 1) | 0;
  const iy0 = iy | 0;
  const uy0 = Math.imul(iy0, 0x27d4eb2f);
  const uy1 = (uy0 + 0x27d4eb2f) | 0;
  const sTerm = Math.imul(seed | 0, 0x9e3779b9);

  const h0 = (uy0 ^ sTerm) | 0;
  const h1 = (uy1 ^ sTerm) | 0;

  let ha = (ux0 ^ h0) | 0;
  ha ^= ha >>> 16; ha = Math.imul(ha, 0x85ebca6b); ha ^= ha >>> 13; ha = Math.imul(ha, 0xc2b2ae35); ha ^= ha >>> 16;
  const a = (ha >>> 0) * INV_U32_X2;

  let hb = (ux1 ^ h0) | 0;
  hb ^= hb >>> 16; hb = Math.imul(hb, 0x85ebca6b); hb ^= hb >>> 13; hb = Math.imul(hb, 0xc2b2ae35); hb ^= hb >>> 16;
  const b = (hb >>> 0) * INV_U32_X2;

  let hc = (ux0 ^ h1) | 0;
  hc ^= hc >>> 16; hc = Math.imul(hc, 0x85ebca6b); hc ^= hc >>> 13; hc = Math.imul(hc, 0xc2b2ae35); hc ^= hc >>> 16;
  const c = (hc >>> 0) * INV_U32_X2;

  let hd = (ux1 ^ h1) | 0;
  hd ^= hd >>> 16; hd = Math.imul(hd, 0x85ebca6b); hd ^= hd >>> 13; hd = Math.imul(hd, 0xc2b2ae35); hd ^= hd >>> 16;
  const d = (hd >>> 0) * INV_U32_X2;

  return (a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy) - 1.0;
}

/** Value noise remapped to [0, 1] at frequency f. */
export const n01 = (x: number, z: number, f: number, seed: number): number => 0.5 + 0.5 * vnoise(x * f, z * f, seed);

/** Hermite smoothstep (GLSL semantics, edges may be reversed). */
export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
