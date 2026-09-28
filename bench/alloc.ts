/**
 * Allocation trace for the CPU twin's 60 Hz path in V8 (Chrome's engine).
 * ---------------------------------------------------------------------------
 * Method: terrainMath.ts is bundled to plain ESM and exercised in a Node child
 * process with a fixed 128 MB semispace (min = max), so no scavenge runs
 * inside a measured window and bytes allocated = heapUsed delta. Each window
 * is bracketed by v8.GCProfiler, which reports collections synchronously; a
 * window containing any GC (which would free the garbage and hide it) is
 * dropped. A positive control (one escaping object per tick) must register or
 * the run fails. V8's sampling heap profiler supplies per-frame attribution.
 *
 * Why V8 and not Bun: JSC NaN-boxes doubles, so float-heavy code never
 * allocates there; V8 boxes a double (16-byte HeapNumber) whenever it crosses
 * a non-inlined call boundary or lands in a context slot. The game runs in
 * Chrome, so V8 is the engine that matters.
 */

import { $ } from 'bun';
import path from 'node:path';
import { mkdirSync } from 'node:fs';
import { resolveRev, WORKTREE } from './lib/revision';

const ROOT = path.resolve(import.meta.dir, '..');
const CACHE = path.join(ROOT, 'bench/out/.cache');

export type AllocResult = {
  rev: string;
  ticks: number;
  controlBytesPerTick: number;   // positive control: one {a,b} per tick on top of the walk
  bytesPerTick: number;          // updatePlayer walk, bytes allocated per tick
  bytesPerSecondAt60Hz: number;
  elevationBytesPerCall: number;
  validWindows: number;          // GC-free measurement windows (of 5)
  topFrames: Array<{ fn: string; share: number }>;
};

async function bundle(rev: string): Promise<string> {
  mkdirSync(CACHE, { recursive: true });
  const short = rev === WORKTREE ? 'worktree' : await resolveRev(rev);
  let entry = path.join(ROOT, 'src/lib/terrainMath.ts');
  if (rev !== WORKTREE) {
    const dir = path.join(CACHE, `rev-${short}`);
    for (const f of ['src/lib/terrainMath.ts', 'src/lib/constants.ts']) {
      await Bun.write(path.join(dir, f), await $`git -C ${ROOT} show ${short}:${f}`.text());
    }
    entry = path.join(dir, 'src/lib/terrainMath.ts');
  }
  const out = path.join(CACHE, `terrainMath-${short}.mjs`);
  const res = await Bun.build({ entrypoints: [entry], target: 'node', format: 'esm', minify: false });
  if (!res.success) throw new Error(res.logs.map(String).join('\n'));
  await Bun.write(out, await res.outputs[0].text());
  return out;
}

const DRIVER = String.raw`
import { Session } from 'node:inspector/promises';
import v8 from 'node:v8';
const [,, modPath, ticksArg] = process.argv;
const m = await import(modPath);
const TICKS = Number(ticksArg);
const SEED = 1337, SX = 2016, SZ = -2592, YAW = 2.056;
const inp = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };
function script(i) {
  inp.forward = (i % 900) < 780 ? 1 : 0;
  inp.strafe = (i % 240) < 60 ? 1 : (i % 240) < 120 ? -1 : 0;
  inp.sprint = (i % 600) < 300;
  inp.jump = (i % 90) === 0;
  inp.yaw = YAW + Math.sin(i * 0.004) * 0.9;
}
let sink = 0;
const p = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, grounded: true };
const ring = new Array(1024).fill(null);
function walk(n, control) {
  p.x = SX; p.y = m.elevation(SX, SZ, SEED, 0) + 1.78; p.z = SZ; p.vx = p.vy = p.vz = 0; p.grounded = true;
  for (let i = 0; i < n; i++) {
    script(i);
    m.updatePlayer(p, inp, 1 / 60, SEED);
    if (control) ring[i & 1023] = { a: p.x, b: i };  // positive control: escapes, so it must allocate
  }
  sink += p.x;
}
function elev(n) { for (let i = 0; i < n; i++) sink += m.elevation(SX + (i & 1023), SZ + (i >> 10), SEED, 0); }

// warm up to optimized tiers first; allocation profile of cold code is noise
for (let k = 0; k < 3; k++) { walk(20000, false); walk(20000, true); elev(200000); }
await new Promise((r) => setTimeout(r, 50));

async function bytesPerOp(fn, ops) {
  const vals = [];
  for (let k = 0; k < 7; k++) {
    globalThis.gc();
    await new Promise((r) => setTimeout(r, 5));
    // v8.GCProfiler reports GC events synchronously, so a window with any
    // collection (which would hide allocated bytes) is detected and dropped.
    const prof = new v8.GCProfiler();
    prof.start();
    const h0 = process.memoryUsage().heapUsed;
    fn();
    const h1 = process.memoryUsage().heapUsed;
    const events = prof.stop().statistics.length;
    if (events !== 0) continue;
    vals.push((h1 - h0) / ops);
  }
  vals.sort((a, b) => a - b);
  return { median: vals[Math.floor(vals.length / 2)], valid: vals.length };
}

const ctl = await bytesPerOp(() => walk(TICKS, true), TICKS);
const tick = await bytesPerOp(() => walk(TICKS, false), TICKS);
const el = await bytesPerOp(() => elev(TICKS), TICKS);

// attribution only: relative shares among terrainMath frames (sampled sizes
// are rate-scaled estimates, so absolute totals are not used)
const s = new Session();
s.connect();
await s.post('HeapProfiler.enable');
await s.post('HeapProfiler.startSampling', { samplingInterval: 32, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
walk(TICKS, false);
const { profile } = await s.post('HeapProfiler.stopSampling');
const byFrame = new Map();
let total = 0;
const visit = (node) => {
  const cf = node.callFrame;
  if (node.selfSize > 0 && cf.url.includes('terrainMath')) {
    total += node.selfSize;
    const key = (cf.functionName || '(anonymous)') + ':' + (cf.lineNumber + 1);
    byFrame.set(key, (byFrame.get(key) || 0) + node.selfSize);
  }
  for (const c of node.children) visit(c);
};
visit(profile.head);
const shares = [...byFrame.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([fn, b]) => ({ fn, share: total ? b / total : 0 }));
console.log(JSON.stringify({ ticks: TICKS, ctl, tick, el, shares, sink: Number.isFinite(sink) }));
`;

export async function runAllocTrace(rev = WORKTREE, ticks = 20_000): Promise<AllocResult> {
  const modPath = await bundle(rev);
  const driver = path.join(CACHE, 'alloc-driver.mjs');
  await Bun.write(driver, DRIVER);
  const out = await $`node --expose-gc --min-semi-space-size=128 --max-semi-space-size=128 ${driver} ${modPath} ${ticks}`.text();
  const r = JSON.parse(out.trim().split('\n').pop()!);
  if (!(r.ctl.median > r.tick.median + 16)) throw new Error('alloc positive control not detected — method invalid');
  return {
    rev: rev === WORKTREE ? 'WORKTREE' : await resolveRev(rev),
    ticks,
    controlBytesPerTick: r.ctl.median,
    bytesPerTick: r.tick.median,
    bytesPerSecondAt60Hz: r.tick.median * 60,
    elevationBytesPerCall: r.el.median,
    validWindows: Math.min(r.ctl.valid, r.tick.valid, r.el.valid),
    topFrames: r.shares,
  };
}
