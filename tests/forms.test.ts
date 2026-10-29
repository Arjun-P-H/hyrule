import { describe, expect, test } from 'bun:test';
import { rasterShader } from '../src/lib/gpu/terrain.wgsl';
import { WATER_COARSE, WATER_WET, waterCellDrawn, waterCornerH } from '../src/lib/gpu/rings';
import { ROCK_AXES, ROCK_WGSL, rockStretch } from '../src/lib/gpu/rockLook';
import { elevation, surfaceAt, TERMS } from '../src/lib/terrainMath';
import { RF } from '../src/lib/hydro';
import { PLACES } from '../bench/forms/places';

// The land's forms, judged by measures (docs/TOTK_TERRAIN.md, the fourth pass): each of these failed
// on the build before it and passes now. Seed 1337, the places of bench/forms/places.ts.
const SEED = 1337;

type CellRule = (g: ArrayLike<number>, w: ArrayLike<number>, cellM: number) => boolean;

/** The rings' old rule: on the coarse rings a cell held water only with two corners wet. */
const twoWetCorners: CellRule = (g, w, cellM) => {
  let nWet = 0, wMax = -Infinity, gMin = Infinity;
  for (let q = 0; q < 4; q++) {
    if (w[q] > g[q] + WATER_WET) nWet++;
    wMax = Math.max(wMax, w[q]); gMin = Math.min(gMin, g[q]);
  }
  return nWet >= (cellM > WATER_COARSE ? 2 : 1) && wMax > gMin - 0.6;
};

/**
 * The water's shore as a ring of S m cells draws it round (cx, cz), `half` m each way: of the places
 * where water meets land in the picture (sampled 8 × 8 a cell), the share that lies on an edge of a cell
 * left out of the water's draw while its neighbour's water still stands at that edge. Such a shore runs
 * along the cell lattice: a stair of S m steps.
 */
function shoreOnCellEdges(cx: number, cz: number, S: number, half: number, rule: CellRule): { shore: number; share: number } {
  const n = Math.round((2 * half) / S), n1 = n + 1;
  const x0 = Math.floor((cx - half) / S) * S, z0 = Math.floor((cz - half) / S) * S;
  const G = new Float64Array(n1 * n1), W = new Float64Array(n1 * n1);
  const o: [number, number] = [0, 0];
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) {
    surfaceAt(x0 + i * S, z0 + j * S, SEED, o, S);
    G[j * n1 + i] = o[0]; W[j * n1 + i] = o[1];
  }
  // Each drawn cell's corners (00, 10, 01, 11): the water sheet's heights, split 00–11, and the ground's,
  // split along the diagonal with the lower midpoint (the rings' choice without a finer level).
  const drawn = new Uint8Array(n * n), hw = new Float64Array(4 * n * n), hg = new Float64Array(4 * n * n);
  const g = [0, 0, 0, 0], w = [0, 0, 0, 0];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const a = j * n1 + i, k = [a, a + 1, a + n1, a + n1 + 1], c = j * n + i;
    let wetMin = Infinity;
    for (let q = 0; q < 4; q++) {
      g[q] = G[k[q]]; w[q] = W[k[q]];
      if (w[q] > g[q] + WATER_WET) wetMin = Math.min(wetMin, w[q]);
    }
    if (!rule(g, w, S)) continue;
    drawn[c] = 1;
    for (let q = 0; q < 4; q++) { hw[4 * c + q] = waterCornerH(g[q], w[q], wetMin); hg[4 * c + q] = g[q]; }
  }
  const visible = (i: number, j: number, u: number, v: number): boolean => {
    if (i < 0 || j < 0 || i >= n || j >= n || !drawn[j * n + i]) return false;
    const c = 4 * (j * n + i), h = hw, e = hg;
    const wv = u >= v ? h[c] + (h[c + 1] - h[c]) * u + (h[c + 3] - h[c + 1]) * v : h[c] + (h[c + 2] - h[c]) * v + (h[c + 3] - h[c + 2]) * u;
    let gv: number;
    if (0.5 * (e[c + 1] + e[c + 2]) > 0.5 * (e[c] + e[c + 3])) {
      gv = u + v <= 1 ? e[c] + (e[c + 1] - e[c]) * u + (e[c + 2] - e[c]) * v : e[c + 3] + (e[c + 2] - e[c + 3]) * (1 - u) + (e[c + 1] - e[c + 3]) * (1 - v);
    } else {
      gv = u >= v ? e[c] + (e[c + 1] - e[c]) * u + (e[c + 3] - e[c + 1]) * v : e[c] + (e[c + 2] - e[c]) * v + (e[c + 3] - e[c + 2]) * u;
    }
    return wv > gv;
  };
  const m = 8, N = n * m;
  const pic = new Uint8Array(N * N);
  for (let J = 0; J < N; J++) for (let I = 0; I < N; I++) {
    pic[J * N + I] = visible(Math.floor(I / m), Math.floor(J / m), ((I % m) + 0.5) / m, ((J % m) + 0.5) / m) ? 1 : 0;
  }
  let shore = 0, onEdge = 0;
  for (let J = 0; J < N; J++) for (let I = 0; I < N; I++) {
    for (const [dI, dJ] of [[1, 0], [0, 1]]) {
      const I2 = I + dI, J2 = J + dJ;
      if (I2 >= N || J2 >= N) continue;
      const a = pic[J * N + I], b = pic[J2 * N + I2];
      if (a === b) continue;
      shore++;
      const ci = Math.floor(I / m), cj = Math.floor(J / m), ci2 = Math.floor(I2 / m), cj2 = Math.floor(J2 / m);
      if (ci === ci2 && cj === cj2) continue;
      // (across a cell's edge into one left out: is the wet side's water still showing at the edge itself?)
      const [wi, wj, di, dj] = a ? [ci, cj, ci2, cj2] : [ci2, cj2, ci, cj];
      if (drawn[dj * n + di]) continue;
      const e = 1e-6;
      let u = ((((a ? I : I2) % m) + 0.5) / m), v = ((((a ? J : J2) % m) + 0.5) / m);
      if (dI) u = a ? 1 - e : e; else v = a ? 1 - e : e;
      if (visible(wi, wj, u, v)) onEdge++;
    }
  }
  return { shore, share: onEdge / Math.max(1, shore) };
}

describe('water edges', () => {
  // Place 3: river banks and pond edges (1849, −2753), on the coarse rings' 8, 16 and 32 m cells.
  const windows: [number, number][] = [[8, 400], [16, 800], [32, 1600]];
  for (const [S, half] of windows) {
    test(`the shore at place 3 follows the ground, not the ${S} m cell lattice`, () => {
      const now = shoreOnCellEdges(1849, -2753, S, half, waterCellDrawn);
      const old = shoreOnCellEdges(1849, -2753, S, half, twoWetCorners);
      expect(now.shore).toBeGreaterThan(200);
      // (the detector finds the stair the old rule drew)
      expect(old.share).toBeGreaterThan(0.15);
      expect(now.share).toBeLessThan(0.02);
    }, 60_000);
  }

  test('the rings list water cells by the same rule', () => {
    const w = rasterShader({ n: 512, levels: 12, blend: 0.25, k: 128, rings: 12, diag: true });
    expect(w).toContain('let spike = nWet == 1 && cellM > ');
    expect(w).not.toContain('select(1, 2, cellM > 4.0)');
  });
});

// The comb: long walls cut into a row of teeth (place 1, the jungle karst). Along a contour on a steep
// cell the slope's aspect should turn with the form, not swing back and forth every few tens of metres.
const TRACE = 128, STEP = 4;

/** The aspect (rad) at TRACE points STEP m apart along the contour through (x0, z0), or null where it is flat or turns back. */
function contourAspect(x0: number, z0: number): Float64Array | null {
  const E = (x: number, z: number) => elevation(x, z, SEED, 0);
  const h0 = E(x0, z0), th = new Float64Array(TRACE), half = TRACE / 2;
  for (const sign of [1, -1]) {
    let x = x0, z = z0;
    for (let k = sign > 0 ? 0 : 1; k <= half; k++) {
      const gx = (E(x + 1, z) - E(x - 1, z)) / 2, gz = (E(x, z + 1) - E(x, z - 1)) / 2, g2 = gx * gx + gz * gz;
      if (g2 < 1e-6) return null;
      // (back onto the contour by a Newton step, then on along it)
      const e = E(x, z);
      x -= ((e - h0) * gx) / g2; z -= ((e - h0) * gz) / g2;
      const idx = half + sign * k;
      if (idx < TRACE) th[idx] = Math.atan2(gz, gx);
      const gl = Math.sqrt(g2);
      x += (sign * -gz / gl) * STEP; z += (sign * gx / gl) * STEP;
    }
  }
  for (let i = 1; i < TRACE; i++) {
    while (th[i] - th[i - 1] > Math.PI) th[i] -= 2 * Math.PI;
    while (th[i] - th[i - 1] < -Math.PI) th[i] += 2 * Math.PI;
  }
  return Math.abs(th[TRACE - 1] - th[0]) > Math.PI / 2 ? null : th;
}

/** The amplitude (degrees) of the strongest swing of a contour's aspect with a wavelength of 32–160 m (detrended, Hann-windowed). */
function combAmplitude(th: Float64Array): number {
  const N = TRACE;
  let su = 0, sv = 0, suu = 0, suv = 0;
  for (let i = 0; i < N; i++) { const u = i - N / 2; su += u; sv += th[i]; suu += u * u; suv += u * th[i]; }
  const b = (N * suv - su * sv) / (N * suu - su * su), a = (sv - b * su) / N;
  const r = new Float64Array(N);
  for (let i = 0; i < N; i++) r[i] = (th[i] - a - b * (i - N / 2)) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
  let best = 0;
  for (let k = 1; k <= N / 2; k++) {
    const lam = (N * STEP) / k;
    if (lam < 32 || lam > 160) continue;
    let re = 0, im = 0;
    for (let i = 0; i < N; i++) { const t = (-2 * Math.PI * k * i) / N; re += r[i] * Math.cos(t); im += r[i] * Math.sin(t); }
    best = Math.max(best, re * re + im * im);
  }
  return ((4 * Math.sqrt(best)) / N) * (180 / Math.PI);
}

describe('the comb', () => {
  test('contours on the steep cells at place 1 carry no strong 32–160 m swing', () => {
    let s = 12345;
    const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) | 0) >>> 0) / 4294967296;
    const E = (x: number, z: number) => elevation(x, z, SEED, 0);
    const amps: number[] = [];
    for (let tries = 0; tries < 2500 && amps.length < 40; tries++) {
      const x = 5767 + (rnd() - 0.5) * 1200, z = -25258 + (rnd() - 0.5) * 1200;
      // (the slope over a 16 m baseline: 20–60°, flanks and steep ground below the walls)
      const H = 16, sl = Math.atan(Math.hypot(E(x + H, z) - E(x - H, z), E(x, z + H) - E(x, z - H)) / (2 * H)) * (180 / Math.PI);
      if (sl < 20 || sl >= 60) continue;
      const th = contourAspect(x, z);
      if (th) amps.push(combAmplitude(th));
    }
    amps.sort((a, b) => a - b);
    expect(amps.length).toBeGreaterThanOrEqual(20);
    // (most contours carry no swing at all, so the teeth show in the tail: on the build before, 30–40 % of
    // these traces swung 20° or more and the upper quartile 30–35°; now 0–9 % and 10–12°, over three samples)
    expect(amps.filter((a) => a >= 20).length / amps.length).toBeLessThan(0.15);
    expect(amps[Math.floor(amps.length * 0.75)]).toBeLessThan(18);
  }, 120_000);
});

// A tower's foot by a river: the ceiling a river's floor puts over the towers once rose at 72° from 4 m
// back, so it cut the apron off in a sheer step a few metres tall. Now it lays the apron back at the
// talus's angle and keeps its steep part for the walls, in the world's metres.
describe('the towers by a river', () => {
  test('their aprons meet the floor at the talus, not in a step', () => {
    let s = 777;
    const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) | 0) >>> 0) / 4294967296;
    const E = (x: number, z: number) => elevation(x, z, SEED, 0);
    const at = [PLACES[0], PLACES[4]], o: [number, number] = [0, 0];
    const slopes: number[] = [];
    for (let tries = 0; tries < 200_000 && slopes.length < 400; tries++) {
      const p = at[tries % 2], x = p.x + (rnd() - 0.5) * 1600, z = p.z + (rnd() - 0.5) * 1600;
      TERMS.on = true;
      surfaceAt(x, z, SEED, o, 0);
      const k = TERMS.karst * TERMS.scale, xf = RF.xFloor;
      TERMS.on = false;
      // (the toe: the towers' first 8 m, within 30 m of a valley floor's edge)
      if (!(xf > 0 && xf < 30 && k > 0.25 && k < 8)) continue;
      const H = 0.5;
      slopes.push(Math.atan(Math.hypot(E(x + H, z) - E(x - H, z), E(x, z + H) - E(x, z - H)) / (2 * H)) * (180 / Math.PI));
    }
    expect(slopes.length).toBeGreaterThanOrEqual(200);
    // (on this pass's first build, 31 % of these toes stood over 60° and 28 % over 70°; now 0.1 % and
    // none. The round domes before it had gentle feet.)
    expect(slopes.filter((a) => a > 60).length / slopes.length).toBeLessThan(0.02);
    expect(slopes.filter((a) => a >= 25 && a < 50).length / slopes.length).toBeGreaterThan(0.6);
  }, 60_000);
});

// The rock's detail on a face: drawn on the ground plane alone (as before) it smeared down a wall
// 1/cos(slope) times; drawn on the two of five planes that face it most (rockLook.ts), at most 1.47×.
describe('rock stretch', () => {
  test('no face that looks up stretches the rock past 1.5×', () => {
    let worst = 0;
    for (let el = 0; el <= 90; el += 0.25) {
      for (let az = 0; az < 360; az += 0.5) {
        const e = (el * Math.PI) / 180, a = (az * Math.PI) / 180;
        worst = Math.max(worst, rockStretch([Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)]));
      }
    }
    expect(worst).toBeLessThan(1.5);
    expect(worst).toBeGreaterThan(1.4);
  });

  test('nor any face at the five places, where the ground plane alone stretched it far more', () => {
    const E = (x: number, z: number) => elevation(x, z, SEED, 0);
    let worst = 0, before = 0;
    for (const pl of PLACES) {
      for (let j = 0; j < 60; j++) for (let i = 0; i < 60; i++) {
        const x = pl.x - 177 + i * 6, z = pl.z - 177 + j * 6;
        const gx = (E(x + 1, z) - E(x - 1, z)) / 2, gz = (E(x, z + 1) - E(x, z - 1)) / 2, l = Math.hypot(gx, gz, 1);
        const n: [number, number, number] = [-gx / l, 1 / l, -gz / l];
        worst = Math.max(worst, rockStretch(n));
        before = Math.max(before, 1 / n[1]);
      }
    }
    expect(worst).toBeLessThan(1.5);
    expect(before).toBeGreaterThan(3);
  }, 60_000);

  test('the shader draws on the same planes', () => {
    const w = rasterShader({ n: 512, levels: 12, blend: 0.25, k: 128, rings: 12 });
    expect(w).toContain(ROCK_WGSL);
    expect(ROCK_AXES.length).toBe(5);
  });
});
