/**
 * The comb across the world (the fp64 CPU twin), the measure tests/forms.test.ts applies at place 1:
 * from random points near each place, the contour is traced 512 m (128 steps of 4 m, a Newton step back
 * onto it at each), and the swing of the slope's aspect along it is taken apart by a Hann-windowed DFT.
 * A trace is a comb when its strongest swing between 32 and 160 m is at least 8° and holds at least 30 %
 * of the swing's power: teeth, not the form turning. Points are sorted by the slope over a 16 m baseline
 * into gentle (< 12°), flank (20–40°), steep (40–60°) and wall (> 60°) land, up to 40 traces each.
 *
 *   bun bench/forms/comb.ts [--seed=1337] [--root=<tree>]
 *
 * Prints, per place and over all, each band's traces, the median and 90th percentile swing (°) and the
 * share that is a comb. Four of the five places (place 3 is a river) and twelve drawn over 80 km.
 */
import path from 'node:path';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d;
const root = path.resolve(arg('root', path.join(import.meta.dir, '../..')));
const T = await import(path.join(root, 'src/lib/terrainMath.ts'));
const seed = Number(arg('seed', '1337'));
const E = (x: number, z: number): number => T.elevation(x, z, seed, 0);

const places: [string, number, number][] = [
  ['p1', 5767, -25258], ['p2', 5571.8, -25027.4], ['p4', 1205, -2644], ['p5', -2585.4, -3520.1],
];
let st0 = 777;
const r0 = () => ((st0 = (Math.imul(st0, 1664525) + 1013904223) | 0) >>> 0) / 4294967296;
for (let i = 0; i < 12; i++) places.push([`r${i}`, (r0() - 0.5) * 80000, (r0() - 0.5) * 80000]);
let st = 12345;
const rnd = () => ((st = (Math.imul(st, 1664525) + 1013904223) | 0) >>> 0) / 4294967296;

const N = 128, DS = 4;
const hann = Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
const lam = Array.from({ length: N / 2 + 1 }, (_, k) => (k ? (N * DS) / k : Infinity));

/** The aspect (rad, unwrapped) at N points DS m apart along the contour through (x0, z0), or null where it is flat. */
function trace(x0: number, z0: number): Float64Array | null {
  const h0 = E(x0, z0), th = new Float64Array(N), half = N / 2;
  for (const sign of [1, -1]) {
    let x = x0, z = z0;
    for (let k = sign > 0 ? 0 : 1; k <= half; k++) {
      const gx = (E(x + 1, z) - E(x - 1, z)) / 2, gz = (E(x, z + 1) - E(x, z - 1)) / 2, g2 = gx * gx + gz * gz;
      if (g2 < 1e-6) return null;
      const e = E(x, z);
      x -= ((e - h0) * gx) / g2; z -= ((e - h0) * gz) / g2;
      const idx = half + sign * k;
      if (idx < N) th[idx] = Math.atan2(gz, gx);
      const g = Math.sqrt(g2);
      x += sign * (-gz / g) * DS; z += sign * (gx / g) * DS;
    }
  }
  for (let i = 1; i < N; i++) {
    while (th[i] - th[i - 1] > Math.PI) th[i] -= 2 * Math.PI;
    while (th[i] - th[i - 1] < -Math.PI) th[i] += 2 * Math.PI;
  }
  return th;
}

/** The strongest 32–160 m swing: its amplitude (°), wavelength (m) and share of the detrended power. */
function peak(v: Float64Array): { A: number; l: number; frac: number } {
  let su = 0, sv = 0, suu = 0, suv = 0;
  for (let i = 0; i < N; i++) { const u = i - N / 2; su += u; sv += v[i]; suu += u * u; suv += u * v[i]; }
  const b = (N * suv - su * sv) / (N * suu - su * su), a = (sv - b * su) / N;
  const r = new Float64Array(N);
  for (let i = 0; i < N; i++) r[i] = (v[i] - a - b * (i - N / 2)) * hann[i];
  const P = new Float64Array(N / 2 + 1);
  for (let k = 1; k <= N / 2; k++) {
    let re = 0, im = 0;
    for (let i = 0; i < N; i++) { const t = (-2 * Math.PI * k * i) / N; re += r[i] * Math.cos(t); im += r[i] * Math.sin(t); }
    P[k] = re * re + im * im;
  }
  let tot = 0, best = 0, bk = 0;
  for (let k = 2; k <= N / 2; k++) { tot += P[k]; if (lam[k] >= 32 && lam[k] <= 160 && P[k] > best) { best = P[k]; bk = k; } }
  return { A: ((4 * Math.sqrt(P[bk])) / N) * (180 / Math.PI), l: lam[bk], frac: (P[bk - 1] + P[bk] + P[bk + 1]) / tot };
}

const BANDS = [['gentle<12', 0, 12], ['flank20-40', 20, 40], ['steep40-60', 40, 60], ['wall>60', 60, 90]] as const;
const QUOTA = 40;
const isComb = (q: { A: number; frac: number }) => q.A >= 8 && q.frac >= 0.3;
function line(rs: { A: number; frac: number }[][]): string {
  return rs.map((r, ci) => {
    if (!r.length) return `${BANDS[ci][0]} -`;
    const As = r.map((q) => q.A).sort((a, b) => a - b);
    return `${BANDS[ci][0]} n${r.length} A50 ${As[r.length >> 1].toFixed(1)} A90 ${As[Math.floor(r.length * 0.9)].toFixed(1)} comb ${((100 * r.filter(isComb).length) / r.length).toFixed(0)}%`;
  }).join(' | ');
}

const all = BANDS.map(() => [] as { A: number; frac: number }[]);
for (const [id, px, pz] of places) {
  const res = BANDS.map(() => [] as { A: number; l: number; frac: number }[]);
  for (let tries = 0; tries < 2500 && res.some((a) => a.length < QUOTA); tries++) {
    const x = px + (rnd() - 0.5) * 1200, z = pz + (rnd() - 0.5) * 1200;
    const H = 16, gx = (E(x + H, z) - E(x - H, z)) / (2 * H), gz = (E(x, z + H) - E(x, z - H)) / (2 * H);
    const sl = (Math.atan(Math.hypot(gx, gz)) * 180) / Math.PI;
    const ci = BANDS.findIndex(([, a, b]) => sl >= a && sl < b);
    if (ci < 0 || res[ci].length >= QUOTA) continue;
    const th = trace(x, z);
    // (a contour that turns back on itself over the trace is a summit or a hollow, not a wall)
    if (!th || Math.abs(th[N - 1] - th[0]) > Math.PI / 2) continue;
    res[ci].push(peak(th));
  }
  res.forEach((r, ci) => all[ci].push(...r));
  console.log(id.padEnd(4), line(res));
}
console.log('ALL ', line(all));
