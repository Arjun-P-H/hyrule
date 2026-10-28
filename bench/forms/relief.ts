/**
 * The relief numbers docs/TOTK_TERRAIN.md holds the land to: heights above the sea (VALLEY.wLevel)
 * of the dry land each region dominates, over a square round the origin (fp64 CPU twin).
 *
 *   bun bench/forms/relief.ts [--seed=1337] [--size=120000] [--step=300] [--workers=4]
 *
 * Prints, per group (fields: plains, wetland, lakes; plateaus: highland; mountains), the median, the
 * 99th percentile and the highest sample, and the land's share of each; then the summit (climbed from
 * the group's 40 highest samples to its top, to a metre) as a summit falls between 300 m samples.
 */
import { cpus } from 'node:os';
import { DEFAULT_SEED, VALLEY } from '../../src/lib/constants';
import { elevation } from '../../src/lib/terrainMath';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d;
const seed = Number(arg('seed', String(DEFAULT_SEED)));
const size = Number(arg('size', '120000'));
const step = Number(arg('step', '300'));
const workers = Number(arg('workers', String(Math.min(4, cpus().length))));
const n = Math.round(size / step);

const GROUPS: Record<string, string[]> = {
  fields: ['plains', 'wetland', 'lakes'],
  forest: ['forest', 'taiga'],
  plateaus: ['highland'],
  mountains: ['mountains'],
  karst: ['jungle', 'rainforest'],
  volcanic: ['volcanic'],
};

const rows = Math.ceil(n / workers);
const parts = await Promise.all(Array.from({ length: workers }, (_, w) => new Promise<{ e: Float32Array; b: Uint8Array }>((resolve, reject) => {
  const wk = new Worker(new URL('./reliefWorker.ts', import.meta.url).href);
  wk.onmessage = (ev) => { wk.terminate(); resolve(ev.data); };
  wk.onerror = (e) => reject(e);
  wk.postMessage({ seed, n, step, size, j0: w * rows, j1: Math.min(n, (w + 1) * rows) });
})));
const { BIOMES } = await import('../../src/lib/biomes');
const by = new Map<string, number[]>();
const tops = new Map<string, [number, number][]>(); // per group (height, sample index)
let land = 0;
for (let w = 0; w < parts.length; w++) {
  const { e, b } = parts[w];
  for (let i = 0; i < e.length; i++) {
    const h = e[i] - VALLEY.wLevel;
    if (h <= 0) continue;
    land++;
    const name = BIOMES[b[i]];
    for (const [g, list] of Object.entries(GROUPS)) if (list.includes(name)) {
      (by.get(g) ?? by.set(g, []).get(g)!).push(h);
      (tops.get(g) ?? tops.set(g, []).get(g)!).push([h, w * rows * n + i]);
    }
  }
}
/** The top of the hill under sample k: steepest ascent on the CPU twin, the step halving from 150 m to 1 m. */
function summit(k: number): number {
  let x = -size / 2 + ((k % n) + 0.5) * step, z = -size / 2 + (Math.floor(k / n) + 0.5) * step;
  let h = elevation(x, z, seed, 0);
  for (let d = step / 2; d >= 1; d /= 2) {
    for (let moved = true; moved;) {
      moved = false;
      for (const [dx, dz] of [[d, 0], [-d, 0], [0, d], [0, -d], [d, d], [-d, d], [d, -d], [-d, -d]]) {
        const e = elevation(x + dx, z + dz, seed, 0);
        if (e > h) { h = e; x += dx; z += dz; moved = true; }
      }
    }
  }
  return h - VALLEY.wLevel;
}
const q = (a: number[], p: number) => a[Math.min(a.length - 1, Math.floor(p * a.length))];
console.log(`seed ${seed}, ${size / 1000} km square at ${step} m: ${land} land samples`);
for (const g of Object.keys(GROUPS)) {
  const a = (by.get(g) ?? []).sort((x, y) => x - y);
  if (!a.length) { console.log(`${g.padEnd(10)} none`); continue; }
  const top = (tops.get(g) ?? []).sort((u, v) => v[0] - u[0]).slice(0, 40).reduce((m, [, k]) => Math.max(m, summit(k)), 0);
  console.log(`${g.padEnd(10)} share ${((100 * a.length) / land).toFixed(1).padStart(5)} %  median ${q(a, 0.5).toFixed(0).padStart(4)} m  p99 ${q(a, 0.99).toFixed(0).padStart(4)} m  highest ${a[a.length - 1].toFixed(0).padStart(4)} m  summit ${top.toFixed(0).padStart(4)} m`);
}
