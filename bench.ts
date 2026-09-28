/**
 * W(x,y) Hyrule Continuous World Engine — benchmark harness (`bun run bench`).
 * ---------------------------------------------------------------------------
 *   bun run bench                         cpu, alloc, gpu, evalcost, parity, shots
 *   bun run bench --cpu-only              cpu + alloc (CI; no GPU needed)
 *   bun run bench --suites=gpu,shots      pick suites (also: bundle, quality, all)
 *   bun run bench --against=<rev>         A/B vs a pinned revision: GPU variants
 *                                         interleaved in one browser session, CPU
 *                                         runs alternated in one process
 *   bun run bench --freeze                write bench/baseline/phase0.json (explicit
 *                                         only; a normal run never overwrites it)
 *   bun run bench --software              GPU suites on SwiftShader (no-GPU floor)
 *
 * Every run writes bench/out/latest.json and bench/out/report-<rev>-<time>.json.
 * With no --against, runs compare against the revision pinned in
 * bench/baseline/phase0.json when it exists.
 */

import { $ } from 'bun';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { runCpuBench, walkScriptProfile, type CpuMetric } from './bench/cpu';
import { runAllocTrace, type AllocResult } from './bench/alloc';
import { runSession, runSessionPerPreset, transformShader, type SessionConfig, type Variant } from './bench/gpu/runner';
import { probeSpecs, analyzeParity, type ParityStats } from './bench/parity';
import { SHOT_FRAMES, collectShots, diffSets, type DiffSummary } from './bench/shots';
import { runQuality, type QualityReport } from './bench/quality';
import { measureRoutes, type RouteBundle } from './bench/bundle';
import { runPlaySession, type PlayResult } from './bench/play';
import { buildProduction, withProductionServer } from './bench/lib/server';
import { loadShader, loadTerrainMath, loadConstants, describeWorktree, resolveRev, WORKTREE } from './bench/lib/revision';
import { generateCameraPath, type CameraFrame } from './bench/spline';
import { summarize, pairedRatio, median, fmt, type Summary } from './bench/lib/stats';
import { QUALITY_PRESETS, CHUNK_SIZE, DEFAULT_SEED, CAMERA } from './src/lib/constants';

const ROOT = import.meta.dir;
const OUT = path.join(ROOT, 'bench/out');
const BASELINE_DIR = path.join(ROOT, 'bench/baseline');
const PHASE0 = path.join(BASELINE_DIR, 'phase0.json');
const CAMERA_PATH = path.join(BASELINE_DIR, 'camera_path.json');

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const flag = (name: string) => argv.includes(`--${name}`);
const opt = (name: string, dflt?: string) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};

const ALL = ['cpu', 'alloc', 'gpu', 'evalcost', 'parity', 'shots', 'bundle', 'play', 'quality'] as const;
type Suite = (typeof ALL)[number];
const DEFAULT_SUITES: Suite[] = ['cpu', 'alloc', 'gpu', 'evalcost', 'parity', 'shots'];
let suites: Suite[] = flag('cpu-only') ? ['cpu', 'alloc'] : DEFAULT_SUITES;
const suitesOpt = opt('suites');
if (suitesOpt) suites = suitesOpt === 'all' ? [...ALL] : (suitesOpt.split(',') as Suite[]);
for (const s of suites) if (!ALL.includes(s)) throw new Error(`unknown suite ${s}`);

const software = flag('software');
const nFrames = Number(opt('frames', '60'));
const reps = Number(opt('reps', '3'));

// ---------------------------------------------------------------------------
// Baseline pinning
// ---------------------------------------------------------------------------
type Phase0 = { baselineRev: string; report: Report };
const phase0: Phase0 | null = existsSync(PHASE0) ? JSON.parse(readFileSync(PHASE0, 'utf8')) : null;
const againstOpt = opt('against');
const against = againstOpt === 'none' ? null : againstOpt ?? (flag('freeze') ? null : phase0?.baselineRev ?? null);

function loadCameraPath(): CameraFrame[] {
  if (existsSync(CAMERA_PATH)) return JSON.parse(readFileSync(CAMERA_PATH, 'utf8'));
  const frames = generateCameraPath(60);
  mkdirSync(BASELINE_DIR, { recursive: true });
  writeFileSync(CAMERA_PATH, JSON.stringify(frames, null, 1));
  return frames;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
type GpuPresetStats = Summary & { width: number; height: number; marchSteps: number; shadowSteps: number };
type Report = {
  meta: Record<string, unknown>;
  cpu?: { metrics: CpuMetric[]; walk: { grounded: number; airborne: number } };
  alloc?: AllocResult[];
  gpu?: {
    env: Record<string, unknown>;
    compileMs: Record<string, number>;
    calibrate: unknown;
    timing: Record<string, Record<string, GpuPresetStats>>;
    ratio?: Record<string, { ratio: number; lo: number; hi: number; n: number }>;
    perScene: Record<string, Record<string, Record<string, number>>>;
    drift: Record<string, { first: number; last: number; min: number; max: number }>;
    instrument: unknown;
  };
  evalcost?: Record<string, Record<string, { nsPerEval: number; samples: number }>>;
  parity?: Record<string, ParityStats[]>;
  shots?: { diffs: DiffSummary[]; hashes: Record<string, string> };
  bundle?: RouteBundle[];
  play?: PlayResult[];
  quality?: QualityReport;
};

async function machine(): Promise<Record<string, string>> {
  const q = async (cmd: string[]) => (await $`${cmd}`.nothrow().quiet().text()).trim();
  return {
    model: await q(['sysctl', '-n', 'hw.model']),
    cpu: await q(['sysctl', '-n', 'machdep.cpu.brand_string']),
    os: await q(['sw_vers', '-productVersion']),
    bun: Bun.version,
    power: (await q(['pmset', '-g', 'batt'])).split('\n')[0] ?? '',
  };
}

async function thermal(): Promise<string> {
  return (await $`pmset -g therm`.nothrow().quiet().text()).trim().replace(/\n+/g, ' | ');
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------
async function variantsFor(kind: 'render' | 'instrument' | 'probe' | 'evalcost'): Promise<Variant[]> {
  const list: Variant[] = [];
  if (against) {
    const b = await loadShader(against);
    // the baseline renders with the preset budgets it shipped with
    const budgets = Object.fromEntries((await loadConstants(against)).QUALITY_PRESETS.map((p) => [p.id, { marchSteps: p.marchSteps, shadowSteps: p.shadowSteps }]));
    list.push({ name: `base@${b.rev}`, vs: b.vs, fs: transformShader(b.fs, kind), budgets });
  }
  const w = await loadShader(WORKTREE);
  list.push({ name: `wt@${w.rev}`, vs: w.vs, fs: transformShader(w.fs, kind) });
  return list;
}

function sessionBase(frames: CameraFrame[]): Omit<SessionConfig, 'modes' | 'variants'> {
  const within = (xs: number[]) => {
    const ok = xs.filter((f) => f < frames.length);
    return ok.length ? ok : [0];
  };
  return {
    presets: QUALITY_PRESETS,
    frames,
    seed: DEFAULT_SEED,
    fov: CAMERA.fov,
    chunkSize: CHUNK_SIZE,
    baseW: 1280,
    baseH: 720,
    reps,
    warmupMs: 2000,
    depth: 1, // depth 2 doubles ANGLE/Metal TIME_ELAPSED (queue wait) — see calibrate
    calibrateDraws: 8,
    calibrateFrames: within([0, 30]),
    sentinelEvery: 10,
    shotFrames: within(SHOT_FRAMES),
    instrumentFrames: within([0, 12, 24, 35, 47]),
    probes: [],
    evalRes: 256,
    evalK: 16,
    evalDps: [0.01, 40.0], // near: all octaves; far (~7 km at Low): macro octaves 6-8 culled
    evalBases: [[1000, -3500], [3000, -5000], [-3000, -2500], [-4300, 3000], [5500, 5000]],
    evalReps: 12,
    software,
  };
}

type TimingSample = { v: number; f: number; r: number; s: number; ms: number };

async function suiteGpu(report: Report, frames: CameraFrame[]): Promise<void> {
  console.log('  [gpu] calibrate + timing (timer queries, depth 1, variants interleaved)');
  const variants = await variantsFor('render');
  const t0 = await thermal();
  const res = await runSessionPerPreset({ ...sessionBase(frames), modes: ['calibrate', 'timing'], variants });
  const t1 = await thermal();
  const timing: Record<string, Record<string, GpuPresetStats>> = {};
  const ratio: NonNullable<Report['gpu']>['ratio'] = {};
  const perScene: Record<string, Record<string, Record<string, number>>> = {};
  const drift: Record<string, { first: number; last: number; min: number; max: number }> = {};
  for (const block of res.results.timing as Array<{ preset: string; width: number; height: number; samples: TimingSample[] }>) {
    const preset = QUALITY_PRESETS.find((p) => p.id === block.preset)!;
    const perFrame: number[][] = variants.map(() => new Array(frames.length).fill(NaN));
    variants.forEach((v, vi) => {
      const xs = block.samples.filter((s) => s.v === vi && !s.s).map((s) => s.ms);
      (timing[v.name] ??= {})[block.preset] = { ...summarize(xs), width: block.width, height: block.height, marchSteps: preset.marchSteps, shadowSteps: preset.shadowSteps };
      for (let f = 0; f < frames.length; f++) {
        perFrame[vi][f] = median(block.samples.filter((s) => s.v === vi && s.f === f && !s.s).map((s) => s.ms));
      }
      const scenes: Record<string, number> = {};
      for (const scene of new Set(frames.map((f) => f.scene))) {
        scenes[scene] = median(frames.filter((f) => f.scene === scene).map((f) => perFrame[vi][f.frame]));
      }
      (perScene[v.name] ??= {})[block.preset] = scenes;
    });
    if (variants.length === 2) ratio[block.preset] = pairedRatio(perFrame[0], perFrame[1]);
    const sent = block.samples.filter((s) => s.s).map((s) => s.ms);
    drift[block.preset] = { first: sent[0], last: sent[sent.length - 1], min: Math.min(...sent), max: Math.max(...sent) };
  }

  console.log('  [gpu] instrumentation counters');
  const inst = await runSessionPerPreset({ ...sessionBase(frames), modes: ['instrument'], variants: await variantsFor('instrument') });

  report.gpu = {
    env: { ...res.env, thermalBefore: t0, thermalAfter: t1 },
    compileMs: Object.fromEntries(res.compile.map((c) => [c.name, c.compileMs])),
    calibrate: res.results.calibrate,
    timing,
    ratio: variants.length === 2 ? ratio : undefined,
    perScene,
    drift,
    instrument: inst.results.instrument,
  };
}

async function suiteEvalcost(report: Report, frames: CameraFrame[]): Promise<void> {
  console.log('  [evalcost] isolated elevation() throughput (K=16 per pixel, 256², 5 regions)');
  const variants = await variantsFor('evalcost');
  const res = await runSession({ ...sessionBase(frames), modes: ['evalcost'], variants });
  const out: NonNullable<Report['evalcost']> = {};
  for (const block of res.results.evalcost as Array<{ dp: number; evalsPerSample: number; samples: Array<{ v: number; b: number; ms: number }> }>) {
    variants.forEach((v, vi) => {
      // median per region, then mean over regions: every region weighs equally
      const regions = [...new Set(block.samples.map((s) => s.b))];
      const perRegion = regions.map((b) => median(block.samples.filter((s) => s.v === vi && s.b === b).map((s) => s.ms)));
      const ms = perRegion.reduce((a, x) => a + x, 0) / perRegion.length;
      (out[v.name] ??= {})[`dp=${block.dp}`] = { nsPerEval: (ms * 1e6) / block.evalsPerSample, samples: block.samples.filter((s) => s.v === vi).length };
    });
  }
  report.evalcost = out;
}

async function suiteParity(report: Report, frames: CameraFrame[]): Promise<void> {
  console.log('  [parity] GPU elevation readback vs CPU fp64 twin');
  const specs = probeSpecs();
  const variants = await variantsFor('probe');
  const res = await runSession({ ...sessionBase(frames), modes: ['probe'], variants, probes: specs });
  const { mod } = await loadTerrainMath(WORKTREE);
  const out: Record<string, ParityStats[]> = {};
  for (const v of variants) out[v.name] = analyzeParity(specs, res.blobs, v.name, mod);
  report.parity = out;
}

async function suiteShots(report: Report, frames: CameraFrame[]): Promise<void> {
  console.log('  [shots] frozen frames + converged ground truth, perceptual ΔE_ok');
  const variants = await variantsFor('render');
  // Ground truth: the pinned revision's shader (or the worktree's) with a tiny
  // horizon clamp, no growth and 3,000-step loops — independent of the march
  // parameters being compared (see transformShader 'truth').
  const src = await loadShader(against ?? WORKTREE);
  const truth: Variant = { name: 'truth', vs: src.vs, fs: transformShader(src.fs, 'truth') };
  const res = await runSessionPerPreset({ ...sessionBase(frames), modes: ['shots'], variants: [...variants, truth] });
  const stamp = variants.map((v) => v.name).join('_vs_');
  const { shots, hashes } = collectShots(res.blobs, path.join(OUT, 'frames', stamp));
  const diffs: DiffSummary[] = [];
  const base = variants[0].name, cand = variants[variants.length - 1].name;
  const fs = sessionBase(frames).shotFrames;
  for (const p of QUALITY_PRESETS) {
    if (variants.length === 2) diffs.push(diffSets(shots, base, p.id, cand, p.id, fs));
    for (const v of variants) diffs.push(diffSets(shots, 'truth', p.id, v.name, p.id, fs));
  }
  report.shots = { diffs, hashes };
}

let built = false;
async function ensureBuild(report: Report): Promise<void> {
  if (built) return;
  const b = await buildProduction();
  report.meta.buildMs = b.ms;
  built = true;
}

async function suiteBundle(report: Report): Promise<void> {
  console.log('  [bundle] production build + first-load JS per route (headless Chrome)');
  await ensureBuild(report);
  report.bundle = await withProductionServer((base) => measureRoutes(base, ['/', '/game']));
}

async function suitePlay(report: Report): Promise<void> {
  console.log('  [play] scripted 30 s walk sessions on /game (fence on / off) + in-game GPU time + SpectorJS');
  await ensureBuild(report);
  report.play = await withProductionServer(async (base) => [
    await runPlaySession(base, { seconds: 30 }),
    await runPlaySession(base, { seconds: 30, fenceOff: true }),
    await runPlaySession(base, { seconds: 12, gpuTiming: true, spector: true }),
  ]);
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------
function printReport(r: Report): void {
  const line = '─'.repeat(96);
  console.log(`\n${line}\n W(x,y) benchmark — worktree ${r.meta.worktree}${r.meta.against ? `  vs  ${r.meta.against}` : ''}\n${line}`);
  if (r.gpu) {
    console.log(` GPU frame time (ms, timer query) — ${r.gpu.env.renderer}`);
    for (const [v, presets] of Object.entries(r.gpu.timing)) {
      for (const [p, s] of Object.entries(presets)) {
        console.log(`   ${v.padEnd(22)} ${p.padEnd(5)} ${String(s.width)}×${s.height}  p50 ${fmt(s.p50, 2).padStart(7)}  p95 ${fmt(s.p95, 2).padStart(7)}  p99 ${fmt(s.p99, 2).padStart(7)}  mean ${fmt(s.mean, 2).padStart(7)}  n=${s.n}`);
      }
    }
    if (r.gpu.ratio) {
      for (const [p, q] of Object.entries(r.gpu.ratio)) {
        console.log(`   paired per-frame ratio ${p.padEnd(5)} ${fmt(q.ratio)}  (95% CI ${fmt(q.lo)}–${fmt(q.hi)}, ${q.n} frames)  <1 = candidate faster`);
      }
    }
    for (const [p, d] of Object.entries(r.gpu.drift)) console.log(`   drift sentinel ${p.padEnd(5)} first ${fmt(d.first, 1)}  last ${fmt(d.last, 1)}  range ${fmt(d.min, 1)}–${fmt(d.max, 1)} ms`);
    for (const x of r.gpu.instrument as Array<Record<string, number | string | boolean>>) {
      if (x.shadowOff) continue;
      const off = (r.gpu.instrument as Array<Record<string, number | string | boolean>>).find((y) => y.shadowOff && y.variant === x.variant && y.preset === x.preset);
      console.log(`   evals/pixel ${String(x.variant).padEnd(22)} ${String(x.preset).padEnd(5)} elevation ${fmt(x.meanElev as number, 2)}  (shadow ${fmt((x.meanElev as number) - (off?.meanElev as number), 2)})  bound ${fmt(x.meanBound as number, 2)}  march iters ${fmt(x.meanSteps as number, 2)}`);
    }
  }
  if (r.evalcost) {
    for (const [v, m] of Object.entries(r.evalcost)) {
      console.log(` elevation() isolated cost ${v.padEnd(22)} ${Object.entries(m).map(([k, x]) => `${k}: ${fmt(x.nsPerEval, 2)} ns`).join('   ')}`);
    }
  }
  if (r.cpu) {
    console.log(` CPU twin (median ns/call over alternated runs; walk script ${r.cpu.walk.grounded} grounded / ${r.cpu.walk.airborne} airborne ticks)`);
    for (const m of r.cpu.metrics) {
      console.log(`   ${m.name.padEnd(36)} ${Object.entries(m.perRev).map(([rev, x]) => `${rev}: ${fmt(x.median, 1)} [${fmt(x.q1, 1)}–${fmt(x.q3, 1)}]`).join('   ')}`);
    }
  }
  if (r.alloc) {
    for (const a of r.alloc) {
      console.log(` V8 allocations ${String(a.rev).padEnd(12)} updatePlayer ${fmt(a.bytesPerTick, 1)} B/tick (${fmt(a.bytesPerSecondAt60Hz / 1024, 1)} KiB/s @60 Hz)  elevation ${fmt(a.elevationBytesPerCall, 1)} B/call  [control ${fmt(a.controlBytesPerTick, 1)} B/tick, ${a.validWindows}/7 GC-free windows]`);
    }
  }
  if (r.parity) {
    for (const [v, stats] of Object.entries(r.parity)) {
      console.log(` fp32/fp64 parity ${v}`);
      for (const s of stats) {
        console.log(`   ${s.id.padEnd(20)} |x|≈${fmt(s.distance / 1000, 1).padStart(7)} km  total max ${fmt(s.totalMax, 5)} p99 ${fmt(s.totalP99, 5)} m  (arith ${fmt(s.arithmeticMax, 5)}, coord ${fmt(s.coordinateMax, 5)})${s.rebaseMax !== undefined ? `  rebase seam ${fmt(s.rebaseMax, 5)} m` : ''}`);
      }
    }
  }
  if (r.shots) {
    console.log(' Perceptual diff (ΔE_ok; JND ≈ 0.02)');
    for (const d of r.shots.diffs) {
      console.log(`   ${d.preset.padEnd(14)} ${d.a} → ${d.b}: mean ${fmt(d.meanDE, 5)}  worst-frame ${fmt(d.worstFrameMeanDE, 5)}  >JND ${fmt(d.fracJND * 100, 3)}%  >0.05 ${fmt(d.fracVisible * 100, 3)}%  identical ${fmt(d.identical * 100, 2)}%`);
    }
  }
  if (r.bundle) {
    for (const b of r.bundle) {
      console.log(` bundle ${b.route.padEnd(6)} first-load ${b.firstLoad.files} files ${fmt(b.firstLoad.rawBytes / 1024, 1)} KiB raw / ${fmt(b.firstLoad.gzipBytes / 1024, 1)} KiB gz   lazy ${b.lazy.files} files ${fmt(b.lazy.rawBytes / 1024, 1)} KiB raw / ${fmt(b.lazy.gzipBytes / 1024, 1)} KiB gz`);
    }
  }
  if (r.play) {
    for (const p of r.play) {
      const tag = p.options.fenceOff ? 'fence off' : p.options.gpuTiming ? 'gpu timing' : 'fence on';
      console.log(` play /game ${tag.padEnd(10)} rAF ${fmt(p.rafPerSec, 1)}/s  renders ${fmt(p.drawsPerSec, 1)}/s  alloc ${fmt(p.sampledBytesPerSec / 1024, 1)} KiB/s (engine chunk ${fmt((p.allocByScript[0]?.bytesPerSec ?? 0) / 1024, 1)})  DRS ${p.canvas.first.join('×')}→${p.canvas.last.join('×')} (${p.canvas.changes} changes, ${p.canvas.reversals} reversals)${p.gpuMs ? `  GPU p50 ${fmt(p.gpuMs.p50, 1)} p95 ${fmt(p.gpuMs.p95, 1)} ms` : ''}${p.spector ? `  spector: ${String(p.spector.commandCount)} cmds, ${String(p.spector.drawCalls)} draw, ${String(p.spector.texturesCreated)} textures` : ''}`);
    }
  }
  if (r.quality) {
    const q = r.quality;
    console.log(` quality: tsc errors ${q.tscErrors} · madge cycles ${q.madgeCircular} · dead exports (src) ${q.deadExports.unused} unused (${q.deadExports.unusedJustified} framework entry points) + ${q.deadExports.internalOnly} export-only-used-internally · tests ${q.tests.pass} pass ${q.tests.fail} fail ${q.tests.expects} expects · coverage funcs ${q.coverage.funcs}% lines ${q.coverage.lines}%`);
  }
  console.log(line);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
const report: Report = {
  meta: {
    timestamp: new Date().toISOString(),
    worktree: await describeWorktree(),
    head: (await $`git rev-parse HEAD`.text()).trim(),
    against: against ? await resolveRev(against) : null,
    suites,
    frames: nFrames,
    reps,
    software,
    machine: await machine(),
  },
};

const frames = loadCameraPath().slice(0, nFrames);
console.log(`W(x,y) bench — suites: ${suites.join(', ')}${against ? ` — A/B vs ${report.meta.against}` : ''}`);

if (suites.includes('cpu')) {
  console.log('  [cpu] CPU twin microbenchmarks');
  const mods = [await loadTerrainMath(WORKTREE)];
  if (against) mods.unshift(await loadTerrainMath(against));
  report.cpu = { metrics: runCpuBench(mods), walk: walkScriptProfile(mods[mods.length - 1].mod) };
}
if (suites.includes('alloc')) {
  console.log('  [alloc] V8 allocation trace (Node, GC-free windows)');
  report.alloc = [await runAllocTrace(WORKTREE)];
  if (against) report.alloc.unshift(await runAllocTrace(against));
}
if (suites.includes('gpu')) await suiteGpu(report, frames);
if (suites.includes('evalcost')) await suiteEvalcost(report, frames);
if (suites.includes('parity')) await suiteParity(report, frames);
if (suites.includes('shots')) await suiteShots(report, frames);
if (suites.includes('bundle')) await suiteBundle(report);
if (suites.includes('play')) await suitePlay(report);
if (suites.includes('quality')) {
  console.log('  [quality] tsc / madge / ts-prune / tests / coverage');
  report.quality = await runQuality();
}

printReport(report);
mkdirSync(OUT, { recursive: true });
const stamp = String(report.meta.timestamp).replace(/[:.]/g, '-');
writeFileSync(path.join(OUT, `report-${report.meta.worktree}-${stamp}.json`), JSON.stringify(report, null, 1));
writeFileSync(path.join(OUT, 'latest.json'), JSON.stringify(report, null, 1));
if (flag('freeze')) {
  mkdirSync(BASELINE_DIR, { recursive: true });
  const rev = String(report.meta.worktree).replace(/-dirty$/, '');
  if (String(report.meta.worktree).endsWith('-dirty')) throw new Error('refusing to freeze a baseline from a dirty src/ tree');
  writeFileSync(PHASE0, JSON.stringify({ baselineRev: rev, report }, null, 1));
  console.log(`froze baseline → ${path.relative(ROOT, PHASE0)} (rev ${rev})`);
}
process.exit(0);
