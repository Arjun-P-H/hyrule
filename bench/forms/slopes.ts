/**
 * Slope histograms of tower, mesa, canyon and mountain land (the fp64 CPU twin): where a region rules,
 * the slope (°) at random points, from central differences over 0.5 m at a 0.5 m footprint.
 *
 *   bun bench/forms/slopes.ts [--seed=1337] [--size=160000] [--n=12000] [--only=<id,…>] [--root=<tree>] [--json=<file>] [--from=<file>]
 *
 * --root measures another checkout's terrainMath (a worktree of an older commit, say); --from reprints
 * a --json file. Prints each class's histogram in 5° bins, its shares in the bench (< 15°), talus
 * (28–40°), the 45–60° gap and cliff (> 60°) bands, the modes of its 1° histogram smoothed over 5°, and
 * the form brief's test (docs/FORMS.md): separate talus and cliff modes, not one smear over 45–70°.
 * The test is for tower, mesa and canyon land; mountain land (ridge) is measured for its page alone.
 */
import path from 'node:path';
import { readFileSync, writeFileSync } from 'node:fs';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d;
const root = path.resolve(arg('root', path.join(import.meta.dir, '../..')));
const T = await import(path.join(root, 'src/lib/terrainMath.ts'));
// (the points are chosen by this tree, whose regions and rivers any tree measured shares)
const C = await import('../../src/lib/terrainMath');
const { BIOMES } = await import('../../src/lib/biomes');
const seed = Number(arg('seed', '1337'));
const size = Number(arg('size', '160000'));
const quota = Number(arg('n', '12000'));
const coarse = 400;

/**
 * The classes: the regions that make them and the range of their weight, for canyons a valley cut at
 * least 10 m deep, and whether the brief's test applies. `edge` is tower land's edge, where the
 * region's weight is still rising.
 */
const CLASSES = [
  { id: 'tower', biomes: ['jungle'], w: [0.6, 2], valley: false, test: true },
  { id: 'edge', biomes: ['jungle'], w: [0.35, 0.75], valley: false, test: true },
  { id: 'mesa', biomes: ['badlands'], w: [0.6, 2], valley: false, test: true },
  { id: 'canyon', biomes: ['badlands', 'desert', 'savanna'], w: [0.6, 2], valley: true, test: true },
  { id: 'ridge', biomes: ['mountains'], w: [0.6, 2], valley: false, test: false },
] as const;

const w = new Float64Array(BIOMES.length);
const s: [number, number] = [0, 0];
const E = (x: number, z: number) => T.elevation(x, z, seed, 0.5);
/** Dry land at (x, z) in the tree measured. */
const dry = (x: number, z: number) => { const [g, wt] = T.surfaceAt(x, z, seed, s, 0.5); return g > wt + 0.01; };
function rules(x: number, z: number, biomes: readonly string[], [lo, hi]: readonly number[]): boolean {
  C.landRegions(x, z, seed, w);
  let sum = 0;
  for (const b of biomes) sum += w[(BIOMES as readonly string[]).indexOf(b)];
  return sum >= lo && sum < hi;
}
/** How deep the rivers' valleys cut at (x, z) (m, ≥ 0), from the CPU twin's term record. */
function valleyCut(x: number, z: number): number {
  C.TERMS.on = true;
  C.surfaceAt(x, z, seed, s, 0.5);
  C.TERMS.on = false;
  return -C.TERMS.valleys;
}

let st = 24601;
const rnd = () => ((st = (Math.imul(st, 1664525) + 1013904223) | 0) >>> 0) / 4294967296;
const n = Math.round(size / coarse);
const from = arg('from', '');
const only = arg('only', '').split(',').filter(Boolean);
const out: Record<string, { hist: number[]; n: number; cells: number }> = from ? JSON.parse(readFileSync(from, 'utf8')) : {};
for (const c of from ? [] : CLASSES.filter((k) => !only.length || only.includes(k.id))) {
  // the coarse cells the class's regions rule, in a fixed order; then points at random within them
  const cells: [number, number][] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = -size / 2 + (i + 0.5) * coarse, z = -size / 2 + (j + 0.5) * coarse;
    if (rules(x, z, c.biomes, c.w) && dry(x, z)) cells.push([x, z]);
  }
  const hist = new Array(91).fill(0);
  let got = 0;
  for (let tries = 0; got < quota && cells.length && tries < quota * 40; tries++) {
    const [cx, cz] = cells[Math.floor(rnd() * cells.length)];
    const x = cx + (rnd() - 0.5) * coarse, z = cz + (rnd() - 0.5) * coarse;
    if (c.valley && valleyCut(x, z) < 10) continue;
    if (!dry(x, z)) continue;
    const gx = (E(x + 0.5, z) - E(x - 0.5, z)) / 1, gz = (E(x, z + 0.5) - E(x, z - 0.5)) / 1;
    hist[Math.min(90, Math.floor((Math.atan(Math.hypot(gx, gz)) * 180) / Math.PI))]++;
    got++;
  }
  out[c.id] = { hist, n: got, cells: cells.length };
}

const share = (h: number[], lo: number, hi: number, N: number) => (100 * h.slice(lo, hi).reduce((a, b) => a + b, 0)) / Math.max(1, N);
function modes(h: number[]): string {
  const sm = h.map((_, i) => { let a = 0, k = 0; for (let d = -2; d <= 2; d++) if (h[i + d] !== undefined) { a += h[i + d]; k++; } return a / k; });
  const peak = Math.max(...sm);
  const m: string[] = [];
  for (let i = 0; i <= 90; i++) {
    const l = sm[i - 1] ?? -1, r = sm[i + 1] ?? -1;
    if (sm[i] > l && sm[i] >= r && sm[i] > 0.04 * peak) m.push(`${i}°`);
  }
  return m.join(' ');
}
/**
 * The form brief's test (docs/FORMS.md): in 5° bins, the largest over 30–40° (talus) and the largest
 * over 65–90° (cliff) each stand at least as high as both their neighbours, and the lowest bin between
 * them is at most half the lower of the two.
 */
function separate(bins: number[]): { talus: number; cliff: number; trough: number; pass: boolean } {
  const peak = (lo: number, hi: number) => { let k = lo; for (let b = lo; b < hi; b++) if (bins[b] > bins[k]) k = b; return k; };
  const t = peak(6, 8), c = peak(13, 18);
  const local = (k: number) => bins[k] >= (bins[k - 1] ?? 0) && bins[k] >= (bins[k + 1] ?? 0);
  const lo = Math.min(bins[t], bins[c]);
  const trough = lo > 0 ? Math.min(...bins.slice(t + 1, c)) / lo : Infinity;
  return { talus: 5 * t, cliff: 5 * c, trough, pass: local(t) && local(c) && trough <= 0.5 };
}
console.log(`seed ${seed}, ${from ? from : root === path.resolve(import.meta.dir, '../..') ? 'this tree' : root}`);
for (const [id, { hist, n: N, cells }] of Object.entries(out)) {
  const bins = Array.from({ length: 18 }, (_, b) => share(hist, 5 * b, 5 * b + 5, N));
  console.log(`${id.padEnd(7)} n ${N} (${cells} cells) bench ${share(hist, 0, 15, N).toFixed(1)} % talus ${share(hist, 28, 41, N).toFixed(1)} % gap45–60 ${share(hist, 45, 60, N).toFixed(1)} % cliff ${share(hist, 60, 91, N).toFixed(1)} % | modes ${modes(hist)}`);
  console.log(`        5° bins ${bins.map((v) => v.toFixed(1)).join(' ')}`);
  const v = separate(bins);
  const tested = CLASSES.find((c) => c.id === id)?.test ?? true;
  console.log(`        talus peak ${v.talus}–${v.talus + 5}°, cliff peak ${v.cliff}–${v.cliff + 5}°, trough ${v.trough.toFixed(2)} → ${v.pass ? 'separate' : 'smear'}${tested ? '' : ' (not tested)'}`);
}
const json = arg('json', '');
if (json) writeFileSync(json, JSON.stringify(out));
