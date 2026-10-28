/**
 * Golden-frame cameras at the five places: near, mid and far views of each place's landform, written
 * as a framed config for the stills runner.
 *
 *   bun bench/forms/frames.ts
 *   CHROME_PATH=… bun bench/webgpu.ts stills --config=forms --frames=all --variants=game-med --hours=12
 *
 * Each place's subject is found from the CPU twin, not picked by eye: for a landform the cell within
 * 400 m with the most relief over 128 m (a tower, a wall, a cliff: not a knoll), looked at uphill (the
 * camera downhill of the face, the face square to the view); for water the bank nearest the place,
 * looked down on. The three cameras stand on one bearing at 120, 450 and 1500 m and aim at the
 * subject, each high enough to clear the trees round it.
 */

import path from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';
import { elevation, surfaceAt, TERMS } from '../../src/lib/terrainMath';
import { DEFAULT_SEED } from '../../src/lib/constants';
import type { CameraFrame } from '../spline';
import { PLACES, type Place } from './places';

const SEED = DEFAULT_SEED;
const DIST = [['near', 120], ['mid', 450], ['far', 1500]] as const;
const E = (x: number, z: number) => elevation(x, z, SEED, 0);

/** The subject of a place and the unit direction the cameras look along (x, z). */
export function subject(p: Place): { x: number; z: number; y: number; dir: [number, number] } {
  const R = p.landform === 'ridge' ? 160 : 400, st = 8, n = Math.round((2 * R) / st) + 1;
  const g = new Float64Array(n * n), wet = new Uint8Array(n * n), tower = new Uint8Array(n * n);
  TERMS.on = p.landform === 'karst';
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = p.x - R + i * st, z = p.z - R + j * st;
    const [e, wt] = surfaceAt(x, z, SEED);
    g[j * n + i] = e;
    wet[j * n + i] = wt > e + 0.02 ? 1 : 0;
    tower[j * n + i] = TERMS.on && TERMS.karst * TERMS.scale > 20 ? 1 : 0;
  }
  TERMS.on = false;
  let best = -1, bi = n >> 1, bj = n >> 1;
  const w = p.landform === 'water' ? 4 : 8; // 64 m window round a bank, 128 m round a face
  for (let j = w; j < n - w; j++) for (let i = w; i < n - w; i++) {
    let score: number;
    if (p.landform === 'water') {
      // a dry cell next to a wet one, nearest the place
      if (wet[j * n + i]) continue;
      if (!(wet[j * n + i + 1] | wet[j * n + i - 1] | wet[(j + 1) * n + i] | wet[(j - 1) * n + i])) continue;
      score = -Math.hypot(i - (n >> 1), j - (n >> 1));
    } else {
      if (p.landform === 'karst' && !tower[j * n + i]) continue; // a tower's flank, not a hill beside one
      let lo = Infinity, hi = -Infinity;
      for (let b = -w; b <= w; b++) for (let a = -w; a <= w; a++) {
        const v = g[(j + b) * n + i + a]; lo = Math.min(lo, v); hi = Math.max(hi, v);
      }
      score = hi - lo;
    }
    if (score > best || best === -1) { best = score; bi = i; bj = j; }
  }
  const x = p.x - R + bi * st, z = p.z - R + bj * st;
  if (p.landform === 'water') {
    // look from the land side across the bank, toward the water
    const gx = E(x + 16, z) - E(x - 16, z), gz = E(x, z + 16) - E(x, z - 16);
    const l = Math.hypot(gx, gz) || 1;
    return { x, z, y: E(x, z), dir: [-gx / l, -gz / l] };
  }
  const h = 30;
  const gx = E(x + h, z) - E(x - h, z), gz = E(x, z + h) - E(x, z - h);
  const l = Math.hypot(gx, gz) || 1;
  // the face's mid height: half way between the subject and the ground 60 m downhill of it
  const y = 0.5 * (E(x, z) + E(x - (60 * gx) / l, z - (60 * gz) / l));
  return { x, z, y, dir: [gx / l, gz / l] };
}

/**
 * A camera d from the subject: of 32 bearings, the one nearest square to the face (dir) whose line of
 * sight is clear, standing over the canopy (a tree stands up to ~30 m, and one beside a camera at head
 * height fills the frame) and looking down on water, across at walls.
 */
function camera(s: ReturnType<typeof subject>, d: number, water: boolean): { x: number; y: number; z: number; yaw: number; pitch: number } {
  const down = water ? 0.45 : 0.18;
  let best: { x: number; y: number; z: number; yaw: number; pitch: number } | null = null, bestScore = -Infinity;
  for (let k = 0; k < 32; k++) {
    const a = Math.atan2(s.dir[1], s.dir[0]) + (k * Math.PI) / 16;
    const ux = Math.cos(a), uz = Math.sin(a);
    const cx = s.x - ux * d, cz = s.z - uz * d;
    let ground = -Infinity;
    for (let i = -24; i <= 24; i += 12) for (let j = -24; j <= 24; j += 12) ground = Math.max(ground, E(cx + i, cz + j));
    const ideal = s.y + d * down, y = Math.max(ground + 35, ideal);
    // the share of the sight line's last 95 % that the land rises over (the subject hidden behind a hill)
    let blocked = 0;
    for (let t = 1; t <= 48; t++) {
      const f = (t / 48) * 0.95;
      if (E(cx + ux * d * f, cz + uz * d * f) > y + (s.y - y) * f) blocked++;
    }
    const square = ux * s.dir[0] + uz * s.dir[1]; // 1: the camera downhill of the face, looking at it
    const score = square - 4 * (blocked / 48) - 0.01 * (y - ideal); // (100 m over the ideal height costs as much as a side view)
    if (score > bestScore) {
      bestScore = score;
      best = { x: cx, y, z: cz, yaw: Math.atan2(ux, uz), pitch: Math.atan2(s.y - y, d) };
    }
  }
  return best!;
}

export function placeFrames(): CameraFrame[] {
  const frames: CameraFrame[] = [];
  for (const p of PLACES) {
    const s = subject(p);
    for (const [label, d] of DIST) {
      const c = camera(s, d, p.landform === 'water');
      frames.push({ frame: frames.length, ...c, scene: `${p.id} ${label}` });
    }
  }
  return frames;
}

if (import.meta.main) {
  const frames = placeFrames();
  const out = path.resolve(import.meta.dir, '../out/.cache');
  mkdirSync(out, { recursive: true });
  writeFileSync(path.join(out, 'framed-forms.json'), JSON.stringify(frames));
  for (const f of frames) console.log(`${String(f.frame).padStart(2)} ${f.scene.padEnd(8)} (${f.x.toFixed(1)}, ${f.y.toFixed(1)}, ${f.z.toFixed(1)}) yaw ${f.yaw.toFixed(3)} pitch ${f.pitch.toFixed(3)}`);
}
