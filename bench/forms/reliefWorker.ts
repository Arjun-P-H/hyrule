/** forms/relief.ts's worker: the height and the dominant region at rows [j0, j1) of the square. */
import { elevation, landRegions } from '../../src/lib/terrainMath';

declare const self: Worker;

self.onmessage = (ev: MessageEvent) => {
  const { seed, n, step, size, j0, j1 } = ev.data as { seed: number; n: number; step: number; size: number; j0: number; j1: number };
  const e = new Float32Array((j1 - j0) * n), b = new Uint8Array((j1 - j0) * n);
  const w = new Float64Array(32);
  for (let j = j0; j < j1; j++) {
    for (let i = 0; i < n; i++) {
      const x = -size / 2 + (i + 0.5) * step, z = -size / 2 + (j + 0.5) * step;
      const o = (j - j0) * n + i;
      e[o] = elevation(x, z, seed, 0);
      landRegions(x, z, seed, w);
      let best = 0;
      for (let k = 1; k < w.length; k++) if (w[k] > w[best]) best = k;
      b[o] = best;
    }
  }
  self.postMessage({ e, b });
};
