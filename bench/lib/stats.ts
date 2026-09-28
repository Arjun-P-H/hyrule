/**
 * Small, dependency-free statistics helpers for the benchmark harness.
 * Percentiles use linear interpolation between closest ranks (Hyndman–Fan
 * type 7, the NumPy/Excel default) so p95/p99 of small samples are not
 * silently the maximum.
 */

export type Summary = {
  n: number;
  min: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
  mean: number;
  sd: number;
};

export function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const h = (sorted.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo);
}

export function summarize(xs: readonly number[]): Summary {
  const s = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return { n: 0, min: NaN, p50: NaN, p95: NaN, p99: NaN, max: NaN, mean: NaN, sd: NaN };
  let sum = 0;
  for (const x of s) sum += x;
  const mean = sum / n;
  let ss = 0;
  for (const x of s) ss += (x - mean) * (x - mean);
  return {
    n,
    min: s[0],
    p50: quantileSorted(s, 0.5),
    p95: quantileSorted(s, 0.95),
    p99: quantileSorted(s, 0.99),
    max: s[n - 1],
    mean,
    sd: n > 1 ? Math.sqrt(ss / (n - 1)) : 0,
  };
}

export function median(xs: readonly number[]): number {
  return quantileSorted([...xs].sort((a, b) => a - b), 0.5);
}

/** Deterministic 32-bit LCG → [0, 1). Used for bootstrap resampling and point sets. */
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Paired speed ratio candidate/baseline with a bootstrap 95% CI.
 * `a[i]` and `b[i]` must be measurements of the same workload item (same
 * frame, same preset) so per-item scene cost cancels. Returns the geometric
 * mean of b[i]/a[i]: < 1 means the candidate is faster.
 */
export function pairedRatio(a: readonly number[], b: readonly number[], iters = 4000): {
  ratio: number;
  lo: number;
  hi: number;
  n: number;
} {
  const logs: number[] = [];
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] > 0 && b[i] > 0 && Number.isFinite(a[i]) && Number.isFinite(b[i])) {
      logs.push(Math.log(b[i] / a[i]));
    }
  }
  const n = logs.length;
  if (n === 0) return { ratio: NaN, lo: NaN, hi: NaN, n: 0 };
  const meanLog = logs.reduce((x, y) => x + y, 0) / n;
  const rnd = lcg(0xc0ffee);
  const boots: number[] = [];
  for (let k = 0; k < iters; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += logs[Math.floor(rnd() * n)];
    boots.push(s / n);
  }
  boots.sort((x, y) => x - y);
  return {
    ratio: Math.exp(meanLog),
    lo: Math.exp(quantileSorted(boots, 0.025)),
    hi: Math.exp(quantileSorted(boots, 0.975)),
    n,
  };
}

export function fmt(x: number, digits = 3): string {
  return Number.isFinite(x) ? x.toFixed(digits) : 'n/a';
}
