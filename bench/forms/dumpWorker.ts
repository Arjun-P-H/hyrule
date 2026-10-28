/** forms/dump.ts's worker: fills rows [j0, j1) of every term's grid (shared buffers) at one place. */
import { surfaceAt, TERMS } from '../../src/lib/terrainMath';
import { TERM_KEYS, gridXZ, type Place } from './places';

declare const self: Worker;

self.onmessage = (ev: MessageEvent) => {
  const { place, seed, n, step, dp, j0, j1, buffers } = ev.data as {
    place: Place; seed: number; n: number; step: number; dp: number; j0: number; j1: number; buffers: SharedArrayBuffer[];
  };
  const out = buffers.map((b) => new Float32Array(b));
  const rec = TERMS as unknown as Record<string, number>;
  const s: [number, number] = [0, 0];
  TERMS.on = true;
  for (let j = j0; j < j1; j++) {
    for (let i = 0; i < n; i++) {
      const [x, z] = gridXZ(place, i, j, n, step);
      surfaceAt(x, z, seed, s, dp);
      const o = j * n + i;
      for (let k = 0; k < TERM_KEYS.length; k++) out[k][o] = rec[TERM_KEYS[k]];
    }
  }
  TERMS.on = false;
  self.postMessage({ done: true });
};
