/**
 * CPU-twin microbenchmarks (terrainMath.ts), isolated from rendering.
 * ---------------------------------------------------------------------------
 * - Point set: 1,024 jittered points around each of the five spline scenes
 *   (plains, alpine, river valley, Zonai anomaly, dunes) — not a 256 m square
 *   of spawn plains.
 * - Each "run" evaluates the full workload once; runs alternate between the
 *   revisions under test (A/B/A/B…) so JIT tier-up, frequency scaling and
 *   core migration hit every revision equally. Reported: median ns/call over
 *   runs, with the interquartile range.
 * - The harness itself does not allocate inside timed loops (inputs are
 *   mutated in place), so any allocation seen by bench/alloc.ts belongs to
 *   the code under test.
 */

import { CAMERA_SPLINE_KEYFRAMES } from './spline';
import { lcg, quantileSorted } from './lib/stats';
import type { TerrainMathModule } from './lib/revision';
import type { PlayerState, InputState } from '../src/lib/terrainMath';
import { DEFAULT_SEED, SPAWN, CAMERA, WORKED_EXAMPLE } from '../src/lib/constants';

export type CpuMetric = {
  name: string;
  unit: 'ns/call';
  perRev: Record<string, { median: number; q1: number; q3: number; runs: number }>;
  checksum: Record<string, number>;
};

type Workload = {
  name: string;
  callsPerRun: number;
  run: (m: TerrainMathModule) => number; // returns a checksum
};

const SCENES = CAMERA_SPLINE_KEYFRAMES.slice(0, 5);
const PTS_PER_SCENE = 1024;

function scenePoints(): Float64Array {
  const rnd = lcg(0x5eed);
  const out = new Float64Array(SCENES.length * PTS_PER_SCENE * 2);
  let o = 0;
  for (const k of SCENES) {
    for (let i = 0; i < PTS_PER_SCENE; i++) {
      out[o++] = k.x + (rnd() * 2 - 1) * 1500;
      out[o++] = k.z + (rnd() * 2 - 1) * 1500;
    }
  }
  return out;
}

const PTS = scenePoints();
const NPTS = PTS.length / 2;
// Raymarch-style footprints: t·(2·tan(FOV/2)/Res_y) for t in 1..3000 m at the
// Medium preset (Res_y = 360).
const FOOT = (2 * Math.tan((CAMERA.fov * Math.PI) / 180 / 2)) / 360;
const DPS = new Float64Array(NPTS);
{
  const rnd = lcg(0xd9);
  for (let i = 0; i < NPTS; i++) DPS[i] = Math.max(1e-4, Math.exp(Math.log(1) + rnd() * Math.log(3000)) * FOOT);
}

/** 60 s of scripted 60 Hz walk-mode input from spawn: forward with strafes, sprint bursts, jumps. */
const TICKS = 3600;
function scriptInput(i: number, inp: InputState): void {
  inp.forward = (i % 900) < 780 ? 1 : 0;
  inp.strafe = (i % 240) < 60 ? 1 : (i % 240) < 120 ? -1 : 0;
  inp.sprint = (i % 600) < 300;
  inp.jump = (i % 90) === 0;
  inp.yaw = SPAWN.yaw + Math.sin(i * 0.004) * 0.9;
}

const WORKLOADS: Workload[] = [
  {
    name: 'elevation(dp=0)  [physics]',
    callsPerRun: NPTS,
    run: (m) => {
      let s = 0;
      for (let i = 0; i < NPTS; i++) s += m.elevation(PTS[2 * i], PTS[2 * i + 1], DEFAULT_SEED, 0);
      return s;
    },
  },
  {
    name: 'elevation(dp>0)  [raymarch LOD]',
    callsPerRun: NPTS,
    run: (m) => {
      let s = 0;
      for (let i = 0; i < NPTS; i++) s += m.elevation(PTS[2 * i], PTS[2 * i + 1], DEFAULT_SEED, DPS[i]);
      return s;
    },
  },
  {
    name: 'gradient(eps=0.25)',
    callsPerRun: NPTS,
    run: (m) => {
      const out: [number, number] = [0, 0];
      let s = 0;
      for (let i = 0; i < NPTS; i++) {
        m.gradient(PTS[2 * i], PTS[2 * i + 1], DEFAULT_SEED, 0.25, out);
        s += out[0] + out[1];
      }
      return s;
    },
  },
  {
    name: 'updatePlayer (60 Hz walk script)',
    callsPerRun: TICKS,
    run: (m) => {
      const ground = m.elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
      const p: PlayerState = { x: SPAWN.x, y: ground + 1.78, z: SPAWN.z, vx: 0, vy: 0, vz: 0, grounded: true };
      const inp: InputState = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };
      for (let i = 0; i < TICKS; i++) {
        scriptInput(i, inp);
        m.updatePlayer(p, inp, 1 / 60, DEFAULT_SEED);
      }
      return p.x + p.y + p.z;
    },
  },
  {
    name: 'marchRay (§30 worked-example ray)',
    callsPerRun: 50,
    run: (m) => {
      const foot = (2 * Math.tan((CAMERA.fov * Math.PI) / 360)) / WORKED_EXAMPLE.resY;
      const ro: [number, number, number] = [SPAWN.x, m.elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0) + WORKED_EXAMPLE.camHeight, SPAWN.z];
      const cp = Math.cos(WORKED_EXAMPLE.pitch), sp = Math.sin(WORKED_EXAMPLE.pitch);
      const rd: [number, number, number] = [Math.sin(WORKED_EXAMPLE.yaw) * cp, sp, Math.cos(WORKED_EXAMPLE.yaw) * cp];
      let s = 0;
      for (let i = 0; i < 50; i++) s += m.marchRay(ro, rd, DEFAULT_SEED, { steps: 96, footprint: foot }).t;
      return s;
    },
  },
];

/** Count grounded vs airborne ticks in the walk script (cost differs 5×). */
export function walkScriptProfile(m: TerrainMathModule): { grounded: number; airborne: number } {
  const ground = m.elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
  const p: PlayerState = { x: SPAWN.x, y: ground + 1.78, z: SPAWN.z, vx: 0, vy: 0, vz: 0, grounded: true };
  const inp: InputState = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };
  let g = 0;
  for (let i = 0; i < TICKS; i++) {
    scriptInput(i, inp);
    m.updatePlayer(p, inp, 1 / 60, DEFAULT_SEED);
    if (p.grounded) g++;
  }
  return { grounded: g, airborne: TICKS - g };
}

export function runCpuBench(
  mods: Array<{ rev: string; mod: TerrainMathModule }>,
  runs = 25,
  warmupRuns = 4,
): CpuMetric[] {
  const metrics: CpuMetric[] = [];
  for (const w of WORKLOADS) {
    const samples: Record<string, number[]> = {};
    const checksum: Record<string, number> = {};
    for (const { rev } of mods) samples[rev] = [];
    for (let r = 0; r < warmupRuns + runs; r++) {
      // rotate order each run so no revision always goes first
      for (let j = 0; j < mods.length; j++) {
        const { rev, mod } = mods[(j + r) % mods.length];
        const t0 = performance.now();
        const cs = w.run(mod);
        const dt = performance.now() - t0;
        checksum[rev] = cs;
        if (r >= warmupRuns) samples[rev].push((dt * 1e6) / w.callsPerRun);
      }
    }
    const perRev: CpuMetric['perRev'] = {};
    for (const [rev, xs] of Object.entries(samples)) {
      const s = [...xs].sort((a, b) => a - b);
      perRev[rev] = { median: quantileSorted(s, 0.5), q1: quantileSorted(s, 0.25), q3: quantileSorted(s, 0.75), runs: s.length };
    }
    metrics.push({ name: w.name, unit: 'ns/call', perRev, checksum });
  }
  return metrics;
}
