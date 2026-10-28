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

/** vnoiseG's gradient (∂/∂x, ∂/∂y) at the point it was last called at. */
export const VN = { dx: 0, dy: 0 };
/** vnoise with its gradient (into VN): the same value, bit for bit. */
export function vnoiseG(x: number, y: number, seed: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
  const uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
  const dux = 30 * fx * fx * (fx * (fx - 2) + 1), duy = 30 * fy * fy * (fy * (fy - 2) + 1);
  const ux0 = ix | 0, ux1 = (ux0 + 1) | 0;
  const uy0 = Math.imul(iy | 0, 0x27d4eb2f), uy1 = (uy0 + 0x27d4eb2f) | 0;
  const sTerm = Math.imul(seed | 0, 0x9e3779b9);
  const h0 = (uy0 ^ sTerm) | 0, h1 = (uy1 ^ sTerm) | 0;
  const a = hashU(ux0 ^ h0) * INV_U32_X2, b = hashU(ux1 ^ h0) * INV_U32_X2;
  const c = hashU(ux0 ^ h1) * INV_U32_X2, d = hashU(ux1 ^ h1) * INV_U32_X2;
  const k = a - b - c + d;
  VN.dx = dux * ((b - a) + k * uy);
  VN.dy = duy * ((c - a) + k * ux);
  return (a + (b - a) * ux + (c - a) * uy + k * ux * uy) - 1.0;
}

/** Value noise remapped to [0, 1] at frequency f. */
export const n01 = (x: number, z: number, f: number, seed: number): number => 0.5 + 0.5 * vnoise(x * f, z * f, seed);

/** Hermite smoothstep (GLSL semantics, edges may be reversed). */
export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/* ------------------------------------------------------------------ */
/* 2D simplex gradient noise with its gradient                         */
/* ------------------------------------------------------------------ */

/** The skew to the simplex lattice and back: (√3 − 1)/2 and (3 − √3)/6. */
export const SIMPLEX_F2 = 0.36602540378443865;
export const SIMPLEX_G2 = 0.21132486540518713;
/** Its 16 unit gradients, at (k + ½)·22.5°: a corner's is SIMPLEX_GRAD[hash3(corner) >>> 28]. */
export const SIMPLEX_GRAD: readonly (readonly [number, number])[] = Array.from({ length: 16 }, (_, k) => {
  const a = ((k + 0.5) * Math.PI) / 8;
  return [Math.cos(a), Math.sin(a)] as const;
});
/** snoise's gradient (∂/∂x, ∂/∂y) at the point it was last called at. */
export const SN = { dx: 0, dy: 0 };

/**
 * 2D simplex noise: three corners, each a unit gradient faded by (½ − r²)⁴, which reaches 0 on the
 * triangle's far edges, so the noise is C³ and a point the twins put in different triangles (fp32 at
 * an edge) sees the same value. Unlike vnoise it has no flat spot at its lattice points and no square
 * contours. Unscaled (|v| ≲ 0.02); its gradient lands in SN.
 */
export function snoise(x: number, y: number, seed: number): number {
  const s = (x + y) * SIMPLEX_F2;
  const i = Math.floor(x + s), j = Math.floor(y + s);
  const t = (i + j) * SIMPLEX_G2;
  const x0 = x - i + t, y0 = y - j + t;
  const i1 = x0 > y0 ? 1 : 0, j1 = 1 - i1;
  const x1 = x0 - i1 + SIMPLEX_G2, y1 = y0 - j1 + SIMPLEX_G2;
  const x2 = x0 - 1 + 2 * SIMPLEX_G2, y2 = y0 - 1 + 2 * SIMPLEX_G2;
  let v = 0, dx = 0, dy = 0;
  const corner = (cx: number, cy: number, h: number) => {
    const r = 0.5 - cx * cx - cy * cy;
    if (r <= 0) return;
    const g = SIMPLEX_GRAD[h >>> 28];
    const gd = g[0] * cx + g[1] * cy, r2 = r * r, r3 = r2 * r, r4 = r2 * r2;
    v += r4 * gd;
    dx += r4 * g[0] - 8 * r3 * gd * cx;
    dy += r4 * g[1] - 8 * r3 * gd * cy;
  };
  corner(x0, y0, hash3(i, j, seed));
  corner(x1, y1, hash3(i + i1, j + j1, seed));
  corner(x2, y2, hash3(i + 1, j + 1, seed));
  SN.dx = dx; SN.dy = dy;
  return v;
}
