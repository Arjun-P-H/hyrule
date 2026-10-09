import { describe, it, expect } from 'bun:test';
import { elevation, MACRO_FREQ, RIDGE_FREQ, WARP_FREQ } from '../src/lib/terrainMath';
import { CHUNK_SIZE, DEFAULT_SEED, F, GEO, HYDRO, LANDS, PLAYER } from '../src/lib/constants';
import { readFileSync } from 'node:fs';
import { TERRAIN_FS } from '../src/components/terrain/demoShader';

/**
 * §25/§27 — floating-origin and dyadic-frequency hardening.
 *
 * Architecture note (measured, see BASELINE.md §4): the live shader evaluates
 * noise on fp32 *global* coordinates, `elevation(p.xz + u_chunk_origin)`; it
 * never multiplies a frequency by the chunk origin on its own. Rebase seams are
 * therefore governed by whether both floating origins round a world point to
 * the same fp32 value (GPU readback: 0.000 mm at 8.6 / 98 / 1000 km), and the
 * dyadic rule below is a guard for any future origin-split evaluation.
 */

const f32 = Math.fround;
const ORIGINS = Array.from({ length: 257 }, (_, i) => (i - 128) * CHUNK_SIZE); // ±1,048,576 m

/** exact in fp32 and on the 1/128 grid: chunkOrigin · f = n / 128 */
function isDyadicAtOrigins(freq: number): boolean {
  return ORIGINS.every((o) => {
    const prod = o * freq;
    return Number.isInteger(prod * 128) && f32(prod) === prod;
  });
}

type FreqEntry = { name: string; value: number; dyadic: boolean; why?: string };

/**
 * Every spatial frequency the GPU evaluates. Non-dyadic entries are pinned
 * with the reason they cannot seam; adding an unregistered F_* constant to
 * the shader fails the test below.
 */
const REGISTRY: FreqEntry[] = [
  ...Object.entries(F)
    .filter(([k]) => k !== 'veg')
    .map(([k, v]) => ({ name: `F.${k}`, value: v, dyadic: true })),
  ...MACRO_FREQ.map((m, i) => ({ name: `macro octave ${i}`, value: F.macro * m, dyadic: true })),
  ...WARP_FREQ.map((m, i) => ({ name: `warp octave ${i}`, value: F.warp * m, dyadic: true })),
  ...RIDGE_FREQ.map((m, i) => ({
    name: `ridge octave ${i}`,
    value: F.ridge * m,
    dyadic: i === 0,
    why: i === 0 ? undefined : 'lacunarity 2.17 (spec §6) predates the dyadic rule; changing it would change the world',
  })),
  { name: 'volcano cell', value: 1 / LANDS.volcano.cell, dyadic: true },
  { name: 'sacred grove cell', value: 1 / LANDS.sacred.cell, dyadic: true },
  { name: 'lake basin cell', value: 1 / GEO.basinCell, dyadic: true },
  { name: 'drainage node lattice', value: 1 / HYDRO.node, dyadic: true },
  { name: 'range crests along / across a chain', value: F.rangeu * Math.cos(GEO.rangeDir[0]), dyadic: false, why: 'an elongated noise in a rotated frame: a shape, its lattice not aligned to the chunks (the heights it feeds are smooth)' },
  { name: 'gully erosion cells (1 / (0.42·0.7·1200 m), ×2 per octave)', value: 1 / (0.42 * 0.7 * 1200), dyadic: false, why: 'erosion.ts units; cells hashed on integer lattice coords, evaluated in fp32 globals like the rest' },
  { name: 'vegetation 1/14 m (shading only)', value: 1 / F.veg, dyadic: false, why: 'albedo modulation, not geometry' },
  { name: 'water ripple 1/5.5 m', value: 1 / 5.5, dyadic: false, why: 'animated shading normal, not geometry' },
  { name: 'water ripple 1/9.7 m', value: 1 / 9.7, dyadic: false, why: 'animated shading normal, not geometry' },
  { name: 'shore foam 1/3.1 m', value: 1 / 3.1, dyadic: false, why: 'shading only' },
  { name: 'dune phase F.dune along a turning wind', value: F.dune * Math.cos(0.6), dyadic: false, why: 'fract() phase of the rotated global coordinate; periodic, a shape, not a lattice' },
  { name: 'volcano gullies 7 per radian', value: 7 / 1000, dyadic: false, why: 'noise round the cone on the unit circle of directions, local to its centre' },
  { name: 'snow drifts 1/23 m, 1/7 m (shading only)', value: 1 / 23, dyadic: false, why: 'albedo, not geometry' },
  { name: 'strata warp 1/90 m (shading only)', value: 1 / 90, dyadic: false, why: 'albedo, not geometry' },
  { name: 'ice cracks 1/9 m (shading only)', value: 1 / 9, dyadic: false, why: 'albedo, not geometry' },
];

describe('§25 dyadic frequency registry', () => {
  it('every registered dyadic frequency gives chunkOrigin·f = n/128, exact in fp32, for origins up to ±1,048,576 m', () => {
    for (const e of REGISTRY.filter((x) => x.dyadic)) {
      expect({ name: e.name, dyadic: isDyadicAtOrigins(e.value) }).toEqual({ name: e.name, dyadic: true });
    }
  });

  it('every non-dyadic frequency is registered as such with a reason (and is in fact non-dyadic)', () => {
    for (const e of REGISTRY.filter((x) => !x.dyadic)) {
      expect(typeof e.why).toBe('string');
      expect({ name: e.name, dyadic: isDyadicAtOrigins(e.value) }).toEqual({ name: e.name, dyadic: false });
    }
  });

  it('every F_* frequency constant declared in the shader is registered and matches constants.ts', () => {
    const decl = [...TERRAIN_FS.matchAll(/const float (F_[A-Z]+)\s*=\s*([\d.]+)\s*\/\s*([\d.]+);/g)];
    expect(decl.length).toBeGreaterThanOrEqual(9);
    for (const [, name, num, den] of decl) {
      const key = name.slice(2).toLowerCase() as keyof typeof F;
      expect(Object.prototype.hasOwnProperty.call(F, key)).toBe(true);
      expect(Number(num) / Number(den)).toBe(F[key]);
      expect(REGISTRY.some((e) => e.name === `F.${key}` && e.dyadic)).toBe(true);
    }
  });

  it('shader octave tables match the CPU twin tables exactly', () => {
    const table = (name: string) => {
      const m = TERRAIN_FS.match(new RegExp(`const float ${name}\\[\\d+\\]\\s*=\\s*float\\[\\d+\\]\\(([^)]*)\\)`));
      return m ? m[1].split(',').map((x) => Number(x.trim())) : null;
    };
    expect(table('MACRO_FREQ')).toEqual([...MACRO_FREQ]);
    expect(table('RIDGE_FREQ')).toEqual([...RIDGE_FREQ]);
    expect(table('WARP_FREQ')).toEqual([...WARP_FREQ]);
  });
});

describe('§25 floating-origin rebase fuzz (GPU coordinate path emulated in fp32)', () => {
  // The renderer forms g = fp32(fp32(local) + fp32(origin)); emulate that exactly.
  const gpuGlobal = (x: number, origin: number) => f32(f32(x - origin) + f32(origin));

  let s = 0x2545f491;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };

  // fp32 spacing at magnitude m (one ulp)
  const ulp32 = (m: number) => 2 ** (Math.floor(Math.log2(Math.max(m, 2 ** -126))) - 23);

  it('the same world point seen from either side of a chunk boundary lands within one fp32 ulp (±2,000 km)', () => {
    let worst = 0;
    for (let n = 0; n < 4000; n++) {
      const k = Math.round((rnd() * 2 - 1) * 244); // boundaries out to ±2,000 km
      const B = k * CHUNK_SIZE;
      const x = B + (rnd() * 2 - 1) * 8.0;         // within 8 m of the boundary
      const gLeft = gpuGlobal(x, B - CHUNK_SIZE);
      const gRight = gpuGlobal(x, B);
      // the rounding chain touches |local| (up to 8192 m) and |global|: the
      // coarser of the two spacings bounds the rebase shift
      const bound = ulp32(Math.max(Math.abs(x), Math.abs(x - B + CHUNK_SIZE), Math.abs(x - B)));
      expect(Math.abs(gLeft - gRight)).toBeLessThanOrEqual(bound);
      worst = Math.max(worst, Math.abs(gLeft - gRight));
    }
    // sub-millimetre near the origin (ulp(8192) ≈ 0.98 mm), one global ulp far out
    expect(worst).toBeLessThanOrEqual(ulp32(2e6));
  });

  it('elevation is C0-continuous across chunk boundaries far from the origin', () => {
    for (let n = 0; n < 600; n++) {
      const k = Math.round((rnd() * 2 - 1) * 120);   // ±1,000 km
      const B = k * CHUNK_SIZE;
      const z = (rnd() * 2 - 1) * 1e6;
      const d = 1e-6;
      const eL = elevation(B - d, z, DEFAULT_SEED, 0);
      const eR = elevation(B + d, z, DEFAULT_SEED, 0);
      expect(Number.isFinite(eL) && Number.isFinite(eR)).toBe(true);
      // Lipschitz bound of E is < 50 m/m; a seam would show as a jump
      expect(Math.abs(eL - eR)).toBeLessThan(50 * 2 * d + 1e-6);
    }
  });

  it('evaluating at the rebased fp32 coordinate moves E by no more than the coordinate rounding allows', () => {
    for (let n = 0; n < 600; n++) {
      const x = (rnd() * 2 - 1) * 2e6;
      const z = (rnd() * 2 - 1) * 2e6;
      const ox = Math.floor(x / CHUNK_SIZE) * CHUNK_SIZE;
      const oz = Math.floor(z / CHUNK_SIZE) * CHUNK_SIZE;
      const gx = gpuGlobal(x, ox), gz = gpuGlobal(z, oz);
      const dist = Math.hypot(gx - x, gz - z);
      const dE = Math.abs(elevation(gx, gz, DEFAULT_SEED, 0) - elevation(x, z, DEFAULT_SEED, 0));
      expect(dE).toBeLessThanOrEqual(50 * dist + 1e-9);
    }
  });
});

describe('§27/§31 GROUND_BIAS is justified by the recorded GPU parity measurement', () => {
  type Probe = { id: string; distanceKm: number; maxM: number; rebaseSeamM?: number };
  const parity = JSON.parse(readFileSync(new URL('../bench/baseline/parity.json', import.meta.url), 'utf8')) as { probes: Probe[] };

  it('covers the measured divergence with ≥ 2.5× margin within the documented ±300 km radius', () => {
    const inside = parity.probes.filter((p) => p.distanceKm <= 300);
    expect(inside.length).toBeGreaterThanOrEqual(8);
    for (const p of inside) expect({ id: p.id, ok: PLAYER.groundBias >= 2.5 * p.maxM }).toEqual({ id: p.id, ok: true });
  });

  it('documents where it stops holding (so the radius in constants.ts stays honest)', () => {
    const beyond = parity.probes.filter((p) => p.distanceKm > 500);
    expect(beyond.some((p) => p.maxM > PLAYER.groundBias)).toBe(true);
  });

  it('rebasing across a chunk boundary moves the surface by under a millimetre on the GPU', () => {
    const seams = parity.probes.filter((p) => p.rebaseSeamM !== undefined);
    expect(seams.length).toBeGreaterThanOrEqual(3);
    for (const p of seams) expect(p.rebaseSeamM!).toBeLessThan(0.001);
  });
});
