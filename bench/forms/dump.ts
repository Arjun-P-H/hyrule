/**
 * Dump the land's height and each of its terms (terrainMath.ts TERMS, the fp64 CPU twin) over a
 * place's 2 × 2 km grid at 0.5 m, as raw little-endian float32 (n × n, rows along x, j along z):
 *   out/forms/<seed>/<place>/<term>.f32 and meta.json.
 *
 *   bun bench/forms/dump.ts [--seed=1337] [--places=p1,p3] [--workers=4] [--n=4000] [--step=0.5]
 *   --at=r2:-21731.2:7452.1[,id:x:z…] dumps ad-hoc places instead of the five.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { cpus } from 'node:os';
import { DEFAULT_SEED } from '../../src/lib/constants';
import { GRID, PLACES, TERM_KEYS, type Place } from './places';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d;
const seed = Number(arg('seed', String(DEFAULT_SEED)));
const want = arg('places', PLACES.map((p) => p.id).join(',')).split(',');
const workers = Number(arg('workers', String(Math.min(4, cpus().length))));
const n = Number(arg('n', String(GRID.n)));
const step = Number(arg('step', String(GRID.step)));
const dp = Number(arg('dp', String(step)));
const root = arg('out', path.join(process.cwd(), 'out/forms', String(seed)));
const at: Place[] = arg('at', '').split(',').filter(Boolean).map((s) => {
  const [id, x, z] = s.split(':');
  return { id, name: 'probe', x: Number(x), z: Number(z), landform: 'probe' };
});

async function dumpPlace(place: Place): Promise<void> {
  const t0 = performance.now();
  const buffers = TERM_KEYS.map(() => new SharedArrayBuffer(n * n * 4));
  const rows = Math.ceil(n / workers);
  await Promise.all(Array.from({ length: workers }, (_, w) => new Promise<void>((resolve, reject) => {
    const wk = new Worker(new URL('./dumpWorker.ts', import.meta.url).href);
    wk.onmessage = () => { wk.terminate(); resolve(); };
    wk.onerror = (e) => reject(e);
    wk.postMessage({ place, seed, n, step, dp, j0: w * rows, j1: Math.min(n, (w + 1) * rows), buffers });
  })));
  const dir = path.join(root, place.id);
  mkdirSync(dir, { recursive: true });
  TERM_KEYS.forEach((k, i) => writeFileSync(path.join(dir, `${k}.f32`), new Uint8Array(buffers[i])));
  writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ ...place, seed, n, step, dp, terms: TERM_KEYS }, null, 2));
  console.log(`${place.id} ${((performance.now() - t0) / 1000).toFixed(1)} s → ${dir}`);
}

for (const p of at.length ? at : PLACES.filter((p) => want.includes(p.id))) await dumpPlace(p);
