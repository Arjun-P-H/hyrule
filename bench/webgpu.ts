/**
 * WebGPU renderer experiments (headless Chrome: Dawn/Metal next to ANGLE/Metal).
 * ---------------------------------------------------------------------------
 *   bun bench/webgpu.ts port       WGSL port vs the WebGL2 renderer: frames diffed
 *                                  pixel by pixel, elevation probes vs the GLSL
 *                                  probe and the fp64 CPU twin
 *   bun bench/webgpu.ts cost       GPU ms per frame along the benchmark camera path:
 *                                  WGSL direct, WGSL clipmap-cached variants, and
 *                                  the WebGL2 renderer at the same resolution
 *   bun bench/webgpu.ts gen        clipmap generation: full rebuilds and
 *                                  incremental updates at walk … fast-fly speeds
 *   bun bench/webgpu.ts fidelity   every variant against the converged WebGL2
 *                                  ground truth (transformShader 'truth')
 *
 * Options: --res=1280x800 --reps=2 --frames=all|0,6,12 --variants=a,b
 * Results: bench/out/webgpu-<mode>.json, frames in bench/out/webgpu/.
 */

import path from 'node:path';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { runSession, transformShader, type SessionConfig, type SessionResult, type Variant } from './gpu/runner';
import { collectShots, SHOT_FRAMES } from './shots';
import { encodePNG, perceptualDiff, type RGBAImage } from './lib/image';
import { loadShader, WORKTREE, loadTerrainMath } from './lib/revision';
import { fmt, pairedRatio, summarize } from './lib/stats';
import { analyzeParity, probeSpecs } from './parity';
import type { CameraFrame } from './spline';
import { CAMERA, CHUNK_SIZE, DEFAULT_SEED, type QualityPreset } from '../src/lib/constants';
import type { TerrainVariant } from '../src/lib/gpu/terrainRenderer';
import { plantInCell, type PlaceWorld } from '../src/lib/forest/placement';
import { SPECIES } from '../src/lib/forest/species';
import { ruinSiteAt } from '../src/lib/ruins';
import { islandAt, islandRho, islandTop } from '../src/lib/skyIslands';

const ROOT = path.resolve(import.meta.dir, '..');
const OUT = path.join(ROOT, 'bench/out');
const args = process.argv.slice(2);
const mode = args[0] ?? 'port';
const opt = (k: string, d: string) => args.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;

const FRAMES: CameraFrame[] = JSON.parse(readFileSync(path.join(ROOT, 'bench/baseline/camera_path.json'), 'utf8'));
const [W, H] = opt('res', '1280x800').split('x').map(Number);
const MARCH_STEPS = Number(opt('march', '128'));
const SHADOW_STEPS = Number(opt('shadow', '28'));
const REPS = Number(opt('reps', '2'));

const VARIANTS: Record<string, TerrainVariant> = {
  direct: { height: 'direct' },
  'clip512': { height: 'cached', clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  'clip512-dp2': { height: 'cached', clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 2, blend: 0.25 },
  'clip1024': { height: 'cached', clip: { n: 1024, levels: 7, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  'clip256': { height: 'cached', clip: { n: 256, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  // Raster geometry clipmaps over the clip512 cache: k cells per half ring, one vertex every `stride` texels.
  'raster128': { height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  'raster64x2': { height: 'raster', k: 64, stride: 2, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25 },
  // E5: raster128 plus scattered grass blades (rings × n² candidates, 0.125 m cells in ring 0).
  'grass': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
  },
  'grass-msaa': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 4,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
  },
  'world': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 4,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 },
  },
  'world-sky': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 4,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  // E6: anti-aliasing of the complete world — none, temporal (TAA), temporal with 0.67× upscaling.
  'world-noaa': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, msaa: 1,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'world-taa': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true, ruins: true, diag: true,
  },
  // The forest's parts: without shadows, without the level-of-detail dither.
  'world-taa-noshadow': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 0, shadowRange: 90 }, islands: true, ruins: true, diag: true,
  },
  'world-taa-nofade': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90, fade: false }, islands: true, ruins: true, diag: true,
  },
  // The game's world without its trees (the forest's cost is the difference).
  'world-taa-bare': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    islands: true, ruins: true, diag: true,
  },
  // Far field: the same world on a 1024² cache (each level reaches twice as far, so a point is shaded one level finer).
  'world-n1024': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 1024, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true, ruins: true, diag: true,
  },
  'raster128-diag': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, diag: true,
  },
  'raster128-diag-always': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, diag: 'always',
  },
  'world-taa-soft': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true, taaSharpen: 0,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'world-taa-a1': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true, taaSharpen: 0, taaAlpha: 1,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'world-taau': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25, taa: true, renderScale: 0.67,
    grass: { rings: 5, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1 },
    forest: { density: 1, side: 48, shadowRes: 2048, shadowRange: 90 }, islands: true,
  },
  'grass-dense': {
    height: 'raster', k: 128, stride: 1, morphStart: 0.75, clip: { n: 512, levels: 8, spacing0: 0.5, margin: 4 }, genDpScale: 1, blend: 0.25,
    grass: { rings: 4, n: 256, spacing0: 0.125, blend: 0.3, height: [0.22, 0.62], width: 0.045, density: 1000 },
  },
};

function pickVariants(def: string): Array<{ name: string; variant: TerrainVariant }> {
  return opt('variants', def).split(',').map((name) => {
    const variant = VARIANTS[name];
    if (!variant) throw new Error(`unknown variant ${name}; known: ${Object.keys(VARIANTS).join(', ')}`);
    return { name, variant };
  });
}

function frameList(def: number[]): number[] {
  const f = opt('frames', '');
  if (f === 'all') return FRAMES.map((_, i) => i);
  if (f) return f.split(',').map(Number);
  return def;
}

let pageJs: string | null = null;
async function webgpuPage(): Promise<string> {
  if (pageJs) return pageJs;
  const build = await Bun.build({ entrypoints: [path.join(import.meta.dir, 'gpu/webgpu-page.ts')], target: 'browser' });
  if (!build.success) throw new Error(build.logs.map(String).join('\n'));
  pageJs = await build.outputs[0].text();
  return pageJs;
}

const CHROME_WEBGPU = ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'];

type GpuPreset = { id: string; w: number; h: number; marchSteps: number; shadowSteps: number };
const PRESET: GpuPreset = { id: `${W}x${H}`, w: W, h: H, marchSteps: MARCH_STEPS, shadowSteps: SHADOW_STEPS };

async function runWebGPU(cfg: Record<string, unknown>): Promise<SessionResult> {
  const all: Record<string, unknown> = {
    variants: [], presets: [PRESET], frames: FRAMES, seed: DEFAULT_SEED, fov: CAMERA.fov, reps: REPS,
    warmupMs: 2000, sentinelEvery: 10, batch: 8, repeat: Number(opt('repeat', '3')), shotFrames: [], probes: [],
    gen: { starts: [0, 12, 24, 35, 47], rebuilds: 3, cases: [] },
    stallMs: 180_000, timeoutMs: 40 * 60_000,
    ...cfg,
  };
  return runSession(all, { pageJs: await webgpuPage(), chromeArgs: CHROME_WEBGPU });
}

function webglBase(modes: SessionConfig['modes'], variants: Variant[], extra: Partial<SessionConfig> = {}): SessionConfig {
  const preset: QualityPreset = { id: 'high', label: PRESET.id, resScale: 1, marchSteps: MARCH_STEPS, shadowSteps: SHADOW_STEPS, note: '' };
  return {
    modes, variants, presets: [preset], frames: FRAMES, seed: DEFAULT_SEED, fov: CAMERA.fov, chunkSize: CHUNK_SIZE,
    baseW: W, baseH: H, reps: REPS, warmupMs: 2000, depth: 1, calibrateDraws: 8, calibrateFrames: [0],
    sentinelEvery: 10, shotFrames: [], instrumentFrames: [0], probes: [],
    evalRes: 256, evalK: 16, evalDps: [0.01], evalBases: [[0, 0]], evalReps: 1,
    stallMs: 180_000, timeoutMs: 40 * 60_000,
    ...extra,
  };
}

/** WebGL readPixels rows run bottom-up; WebGPU copies run top-down. */
function flipRows(img: RGBAImage): RGBAImage {
  const { width: w, height: h, data } = img;
  const out = new Uint8Array(data.length);
  for (let y = 0; y < h; y++) out.set(data.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
  return { width: w, height: h, data: out };
}

function saveShots(res: SessionResult, dir: string, flip: boolean): Map<string, RGBAImage> {
  mkdirSync(dir, { recursive: true });
  const { shots } = collectShots(res.blobs, null);
  const out = new Map<string, RGBAImage>();
  for (const [key, img0] of shots) {
    const img = flip ? flipRows(img0) : img0;
    const [variant, , frame] = key.split('|');
    out.set(`${variant}|${frame}`, img);
    Bun.write(path.join(dir, `${variant.replace(/[^\w.-]/g, '_')}_f${frame.padStart(2, '0')}.png`), encodePNG(img));
  }
  return out;
}

function diffTable(a: Map<string, RGBAImage>, aName: string, b: Map<string, RGBAImage>, bName: string, frames: number[]) {
  const rows: Array<{ frame: number } & ReturnType<typeof perceptualDiff>> = [];
  for (const f of frames) {
    const A = a.get(`${aName}|${f}`), B = b.get(`${bName}|${f}`);
    if (!A || !B) continue;
    rows.push({ frame: f, ...perceptualDiff(A, B) });
  }
  const mean = (k: 'identical' | 'meanDE' | 'fracJND' | 'fracVisible') => rows.reduce((s, r) => s + r[k], 0) / Math.max(rows.length, 1);
  return {
    a: aName, b: bName, frames: rows.length,
    identical: mean('identical'), meanDE: mean('meanDE'), fracJND: mean('fracJND'), fracVisible: mean('fracVisible'),
    worstFrameJND: Math.max(0, ...rows.map((r) => r.fracJND)), p99DE: Math.max(0, ...rows.map((r) => r.p99DE)),
    perFrame: rows.map((r) => ({ frame: r.frame, identical: r.identical, meanDE: r.meanDE, fracJND: r.fracJND, p99DE: r.p99DE })),
  };
}

function printDiff(d: ReturnType<typeof diffTable>): void {
  console.log(`  ${d.b.padEnd(14)} vs ${d.a.padEnd(14)} frames=${d.frames} identical=${(d.identical * 100).toFixed(2)}% ` +
    `meanΔE=${d.meanDE.toFixed(4)} >JND=${(d.fracJND * 100).toFixed(2)}% (worst frame ${(d.worstFrameJND * 100).toFixed(2)}%) ` +
    `>0.05=${(d.fracVisible * 100).toFixed(2)}% p99ΔE=${d.p99DE.toFixed(4)}`);
}

// ---------------------------------------------------------------------------
async function port(): Promise<void> {
  const frames = frameList(SHOT_FRAMES);
  const probes = probeSpecs();
  console.log(`WGSL port check at ${W}×${H}, frames ${frames.join(',')}`);
  const gpu = await runWebGPU({ modes: ['shots', 'probe'], variants: pickVariants('direct'), shotFrames: frames, probes });
  console.log('  webgpu env:', JSON.stringify(gpu.env), 'compile:', JSON.stringify(gpu.compile));
  const src = await loadShader(WORKTREE);
  const gl = await runSession(webglBase(['shots'], [{ name: 'webgl', vs: src.vs, fs: src.fs }], { shotFrames: frames }));
  const glProbe = await runSession(webglBase(['probe'], [{ name: 'glsl', vs: src.vs, fs: transformShader(src.fs, 'probe') }], { probes }));
  const dir = path.join(OUT, 'webgpu', 'port');
  const a = saveShots(gl, dir, true);
  const b = saveShots(gpu, dir, false);
  const d = diffTable(a, 'webgl', b, 'direct', frames);
  printDiff(d);

  const { mod } = await loadTerrainMath(WORKTREE);
  const wgsl = analyzeParity(probes, gpu.blobs, 'wgsl', mod);
  const glsl = analyzeParity(probes, glProbe.blobs, 'glsl', mod);
  // WGSL vs GLSL directly
  const byId = (res: SessionResult, v: string) => new Map(res.blobs.filter((x) => x.query.kind === 'probe' && x.query.variant === v)
    .map((x) => [x.query.id, new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4)] as const));
  const W1 = byId(gpu, 'wgsl'), G1 = byId(glProbe, 'glsl');
  console.log('  probe                    |E_wgsl−E_cpu| max   |E_glsl−E_cpu| max   |E_wgsl−E_glsl| max   identical');
  const probeRows = [];
  for (const s of probes) {
    const a1 = W1.get(s.id), b1 = G1.get(s.id);
    if (!a1 || !b1) continue;
    let m = 0, same = 0;
    for (let i = 0; i < a1.length; i += 4) {
      m = Math.max(m, Math.abs(a1[i] - b1[i]));
      if (a1[i] === b1[i]) same++;
    }
    const w = wgsl.find((x) => x.id === s.id)!, g = glsl.find((x) => x.id === s.id)!;
    probeRows.push({ id: s.id, wgslCpuMax: w.totalMax, glslCpuMax: g.totalMax, wgslGlslMax: m, identical: same / (a1.length / 4) });
    console.log(`  ${s.id.padEnd(24)} ${(w.totalMax * 1000).toFixed(3).padStart(10)} mm ${(g.totalMax * 1000).toFixed(3).padStart(14)} mm ` +
      `${(m * 1000).toFixed(3).padStart(15)} mm ${(same / (a1.length / 4) * 100).toFixed(1).padStart(10)}%`);
  }
  writeFileSync(path.join(OUT, 'webgpu-port.json'), JSON.stringify({ res: [W, H], env: gpu.env, compile: gpu.compile, image: d, probes: probeRows }, null, 1));
}

type TSample = { v: number; f: number; r: number; s: number; ms: number; first: number; rawRender: number; gen: number | null; texels: number };

async function cost(): Promise<void> {
  const variants = pickVariants('direct,clip512,clip512-dp2');
  const frames = frameList(FRAMES.map((_, i) => i));
  const sel = frames.map((i) => FRAMES[i]);
  console.log(`GPU cost at ${W}×${H} (march ${MARCH_STEPS}, shadow ${SHADOW_STEPS}), ${sel.length} frames × ${REPS} reps: ${variants.map((v) => v.name).join(', ')}`);
  const gpu = await runWebGPU({ modes: ['timing'], variants, frames: sel });
  console.log('  compile:', gpu.compile.map((c) => `${c.name} ${c.compileMs.toFixed(0)} ms`).join(', '));
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((s) => !s.s);
  const perV = variants.map((_, vi) => {
    const byF = new Map<number, number[]>();
    for (const s of t) if (s.v === vi) byF.set(s.f, [...(byF.get(s.f) ?? []), s.ms]);
    return [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  });
  const gens = variants.map((_, vi) => t.filter((s) => s.v === vi && s.gen !== null).map((s) => s.gen as number));
  let webgl: number[] | null = null;
  if (opt('webgl', '1') === '1') {
    const src = await loadShader(WORKTREE);
    const gl = await runSession(webglBase(['timing'], [{ name: 'webgl', vs: src.vs, fs: src.fs }], { frames: sel }));
    const gs = (gl.results.timing as Array<{ samples: Array<{ v: number; f: number; s: number; ms: number }> }>)[0].samples.filter((s) => !s.s);
    const byF = new Map<number, number[]>();
    for (const s of gs) byF.set(s.f, [...(byF.get(s.f) ?? []), s.ms]);
    webgl = [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  }
  const rows: Record<string, unknown>[] = [];
  const line = (name: string, xs: number[], ref: number[] | null) => {
    const s = summarize(xs);
    const r = ref ? pairedRatio(ref, xs) : null;
    console.log(`  ${name.padEnd(14)} p50 ${fmt(s.p50, 2).padStart(8)} ms  p95 ${fmt(s.p95, 2).padStart(8)} ms  max ${fmt(s.max, 2).padStart(8)} ms` +
      (r ? `   × ${fmt(r.ratio, 4)} [${fmt(r.lo, 4)}, ${fmt(r.hi, 4)}] vs direct` : ''));
    rows.push({ name, ...s, ratioVsDirect: r });
  };
  if (webgl) line('webgl', webgl, perV[0]);
  variants.forEach((v, vi) => line(v.name, perV[vi], vi === 0 ? null : perV[0]));
  writeFileSync(path.join(OUT, 'webgpu-cost.json'), JSON.stringify({
    res: [W, H], march: MARCH_STEPS, shadow: SHADOW_STEPS, frames, env: gpu.env, compile: gpu.compile,
    variants: variants.map((v) => v.name), perFrameMin: perV, webgl, rows, gen: gens,
  }, null, 1));
}

async function gen(): Promise<void> {
  const variants = pickVariants('clip512,clip1024');
  const cases = [
    { name: 'walk', speed: 5.4, frames: 60, heading: 0.7 },
    { name: 'sprint', speed: 8.6, frames: 60, heading: 0.7 },
    { name: 'fly', speed: 40, frames: 60, heading: 0.7 },
    { name: 'fast-fly', speed: 200, frames: 60, heading: 0.7 },
  ];
  console.log(`clipmap generation (${variants.map((v) => v.name).join(', ')}), frames rendered at ${W}×${H}`);
  const res = await runWebGPU({ modes: ['gen'], variants, gen: { starts: [0, 12, 24, 35, 47], rebuilds: 3, cases } });
  type G = {
    variant: string;
    rebuild: Array<{ frameMs: number; steadyMs: number; rawGenMs: number; texels: number }>;
    moves: Array<{ case: string; speed: number; still: number[]; moving: number[]; rawGen: number[]; texels: number[] }>;
  };
  const out = res.results.gen as G[];
  for (const g of out) {
    const added = summarize(g.rebuild.map((r) => r.frameMs - r.steadyMs));
    const raw = summarize(g.rebuild.map((r) => r.rawGenMs));
    const tex = g.rebuild[0]?.texels ?? 0;
    console.log(`  ${g.variant}: full rebuild of ${(tex / 1e6).toFixed(2)} M texels adds p50 ${fmt(added.p50, 2)} ms (max ${fmt(added.max, 2)}) to its frame; ` +
      `generator pass p50 ${fmt(raw.p50, 2)} ms → ${fmt(tex / raw.p50 / 1e3, 0)} M texels/s`);
    for (const c of cases) {
      const ms = g.moves.filter((m) => m.case === c.name);
      const moving = ms.flatMap((m) => m.moving), still = ms.flatMap((m) => m.still);
      const raw2 = summarize(ms.flatMap((m) => m.rawGen.filter((x) => x > 0)));
      const sm = summarize(moving), ss = summarize(still), st = summarize(ms.flatMap((m) => m.texels));
      console.log(`    ${c.name.padEnd(9)} ${String(c.speed).padStart(4)} m/s  frame p50 ${fmt(sm.p50, 2)} ms (still ${fmt(ss.p50, 2)}) p95 ${fmt(sm.p95, 2)} (still ${fmt(ss.p95, 2)})` +
        `  generator pass p50 ${fmt(raw2.p50, 3)} ms max ${fmt(raw2.max, 3)}  texels/frame p50 ${fmt(st.p50, 0)} max ${fmt(st.max, 0)}`);
    }
  }
  writeFileSync(path.join(OUT, 'webgpu-gen.json'), JSON.stringify({ res: [W, H], env: res.env, results: out }, null, 1));
}

/** Still shots of gallery frames for the listed variants (TAA converged over --converge frames). */
async function stills(): Promise<void> {
  const config = opt('config', 'gallery');
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const shotFrames = opt('frames', '1,3') === 'all' ? framed.map((_, i) => i) : opt('frames', '1,3').split(',').map(Number);
  const variants = pickVariants('world-noaa,world-taa-a1,world-taa-soft');
  const timed = !!opt('timing', '');
  const gpu = await runWebGPU({ modes: timed ? ['shots', 'timing'] : ['shots'], variants, frames: framed, shotFrames, converge: Number(opt('converge', '32')) });
  saveShots(gpu, path.join(OUT, 'webgpu', config === 'gallery' ? 'fidelity' : `stills-${config}`), false);
  console.log('stills saved', variants.map((v) => v.name).join(', '));
  const shotsInfo = gpu.results.shots as Array<{ variant: string; frame: number; trees?: [number, number] }>;
  const tr = shotsInfo.filter((x) => x.trees && (x.trees[0] || x.trees[1]));
  if (tr.length) console.log('  trees (near/far) per frame:', tr.map((x) => `${x.variant} f${x.frame}: ${x.trees![0]}/${x.trees![1]}`).join(', '));
  if (!timed) return;
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((x) => !x.s);
  // Per frame, the fastest rep of each variant; ratios are paired by frame against the first variant.
  const perFrame = variants.map((_, vi) => {
    const byF = new Map<number, number[]>();
    for (const x of t) if (x.v === vi) byF.set(x.f, [...(byF.get(x.f) ?? []), x.ms]);
    return [...byF.entries()].sort((a, b) => a[0] - b[0]).map(([, xs]) => Math.min(...xs));
  });
  for (const [vi, v] of variants.entries()) {
    const s = summarize(t.filter((x) => x.v === vi).map((x) => x.ms));
    const r = vi ? pairedRatio(perFrame[0], perFrame[vi]) : null;
    console.log(`  ${v.name}: p50 ${fmt(s.p50, 2)} ms  p95 ${fmt(s.p95, 2)} ms` + (r ? `   × ${fmt(r.ratio, 3)} [${fmt(r.lo, 3)}, ${fmt(r.hi, 3)}] vs ${variants[0].name}` : ''));
  }
}

async function fidelity(): Promise<void> {
  const frames = frameList(SHOT_FRAMES);
  const variants = pickVariants('direct,clip512,clip512-dp2');
  console.log(`fidelity at ${W}×${H} vs the converged WebGL2 truth, frames ${frames.join(',')}`);
  const src = await loadShader(WORKTREE);
  const gl = await runSession(webglBase(['shots'], [
    { name: 'truth', vs: src.vs, fs: transformShader(src.fs, 'truth') },
    { name: 'webgl', vs: src.vs, fs: src.fs },
  ], { shotFrames: frames }));
  const gpu = await runWebGPU({ modes: ['shots'], variants, shotFrames: frames });
  const dir = path.join(OUT, 'webgpu', 'fidelity');
  const a = saveShots(gl, dir, true);
  const b = saveShots(gpu, dir, false);
  const all = new Map([...a, ...b]);
  const diffs = [diffTable(all, 'truth', all, 'webgl', frames), ...variants.map((v) => diffTable(all, 'truth', all, v.name, frames))];
  for (const d of diffs) printDiff(d);
  const vsDirect = variants.slice(1).map((v) => diffTable(all, 'direct', all, v.name, frames));
  for (const d of vsDirect) printDiff(d);
  writeFileSync(path.join(OUT, 'webgpu-fidelity.json'), JSON.stringify({ res: [W, H], frames, vsTruth: diffs, vsDirect }, null, 1));
}

/**
 * Third person: the hero framed on the camera path (bench/hero.ts `frames`
 * output) drawn by the raster renderer next to the WebGL2 hero program;
 * frames diffed, and GPU cost with and without the hero.
 */
async function hero(): Promise<void> {
  const config = opt('config', 'run');
  const file = path.join(OUT, '.cache', `framed-${config}.json`);
  const framed: CameraFrame[] = JSON.parse(readFileSync(file, 'utf8'));
  const frames = opt('frames', '') === 'all' || config === 'gallery' && !opt('frames', '')
    ? framed.map((_, i) => i)
    : frameList(SHOT_FRAMES).filter((i) => i < framed.length);
  const variants = pickVariants('raster128');
  console.log(`hero (${config}) at ${W}×${H}, frames ${frames.join(',')}: ${variants.map((v) => v.name).join(', ')} vs WebGL2`);
  const src = await loadShader(WORKTREE);
  const { withHero } = await import('../src/components/terrain/demoShader');
  const gl = await runSession(webglBase(['shots'], [{ name: 'webgl-hero', vs: src.vs, fs: withHero(src.fs), hero: true }], { frames: framed, shotFrames: frames }));
  const gpu = await runWebGPU({ modes: ['shots', 'timing'], variants, frames: framed, shotFrames: frames });
  const dir = path.join(OUT, 'webgpu', `hero-${config}`);
  if (config === 'gallery') console.log('  scenes:', frames.map((i) => `${i}:${framed[i].scene}`).join(' '));
  const a = saveShots(gl, dir, true);
  const b = saveShots(gpu, dir, false);
  for (const v of variants) printDiff(diffTable(a, 'webgl-hero', b, v.name, frames));
  const shotsInfo = gpu.results.shots as Array<{ variant: string; frame: number; blades?: number; trees?: [number, number] }>;
  const bl = shotsInfo.filter((x) => x.blades);
  if (bl.length) console.log('  blades per frame:', bl.map((x) => `${x.variant} f${x.frame}: ${x.blades}`).join(', '));
  const tr = shotsInfo.filter((x) => x.trees && (x.trees[0] || x.trees[1]));
  if (tr.length) console.log('  trees (near/far) per frame:', tr.map((x) => `${x.variant} f${x.frame}: ${x.trees![0]}/${x.trees![1]}`).join(', '));
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((x) => !x.s);
  for (const [vi, v] of variants.entries()) {
    const s = summarize(t.filter((x) => x.v === vi).map((x) => x.ms));
    console.log(`  ${v.name} with the hero: p50 ${fmt(s.p50, 2)} ms  p95 ${fmt(s.p95, 2)} ms`);
  }
}

/**
 * Anti-aliasing (E6): still shots of ground-level views (TAA given 32 frames of
 * history) scored against a 2 × 2 supersampled render of the same variant
 * family (4× MSAA at twice the resolution, box-filtered down); a slow pan for
 * temporal stability; GPU cost of each.
 */
async function aa(): Promise<void> {
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', 'framed-gallery.json'), 'utf8'));
  const shotFrames = opt('frames', '1,3,5,7,13').split(',').map(Number);
  const variants = pickVariants('world-noaa,world-sky,world-taa,world-taa-soft,world-taau');
  console.log(`anti-aliasing at ${W}×${H}: ${variants.map((v) => v.name).join(', ')}; reference: world-sky at ${2 * W}×${2 * H}, box 2×2`);
  const ref = await runSession({
    variants: pickVariantsFrom(['world-sky']), presets: [{ id: 'ref', w: 2 * W, h: 2 * H, marchSteps: MARCH_STEPS, shadowSteps: SHADOW_STEPS }],
    frames: framed, seed: DEFAULT_SEED, fov: CAMERA.fov, reps: 1, warmupMs: 500, sentinelEvery: 10, batch: 8, repeat: 3,
    shotFrames, probes: [], gen: { starts: [], rebuilds: 0, cases: [] }, modes: ['shots'], stallMs: 180_000, timeoutMs: 40 * 60_000,
  }, { pageJs: await webgpuPage(), chromeArgs: CHROME_WEBGPU });
  // A slow pure yaw (≈ 0.3 px per frame at 1280 px), time held still: what changes between
  // frames beyond the sub-pixel drift is aliasing shimmer.
  const pan: CameraFrame[] = [];
  const f0 = framed[3];
  for (let k = 0; k < 32; k++) pan.push({ ...f0, frame: k, yaw: f0.yaw + 0.0004 * k });
  const gpu = await runWebGPU({ modes: ['shots', 'seq', 'timing'], variants, frames: framed, shotFrames, converge: 32, seq: pan, seqFreeze: true });
  const box = (img: RGBAImage): RGBAImage => {
    const w = img.width / 2, h = img.height / 2, out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) {
      const at = (xx: number, yy: number) => img.data[(yy * img.width + xx) * 4 + c];
      out[(y * w + x) * 4 + c] = Math.round((at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1)) / 4);
    }
    return { width: w, height: h, data: out };
  };
  const refShots = saveShots(ref, path.join(OUT, 'webgpu', 'aa-ref'), false);
  const shots = saveShots(gpu, path.join(OUT, 'webgpu', 'aa'), false);
  const refs = new Map([...refShots].map(([k, img]) => [k.replace('world-sky|', 'ref|'), box(img)]));
  const seqRes = gpu.results.seq as Array<{ variant: string; meanDelta: number[]; flicker: number[] }>;
  const t = (gpu.results.timing as Array<{ samples: TSample[] }>)[0].samples.filter((x) => !x.s);
  console.log('  variant        vs 2×2 SSAA: meanΔE  >JND    | pan: mean |ΔY|  flicker px | GPU p50   p95');
  variants.forEach((v, vi) => {
    const d = diffTable(refs, 'ref', shots, v.name, shotFrames);
    const sq = seqRes.find((x) => x.variant === v.name)!;
    const md = sq.meanDelta.reduce((a, b) => a + b, 0) / sq.meanDelta.length;
    const fl = sq.flicker.reduce((a, b) => a + b, 0) / sq.flicker.length;
    const st = summarize(t.filter((x) => x.v === vi).map((x) => x.ms));
    console.log(`  ${v.name.padEnd(14)} ${fmt(d.meanDE, 4).padStart(13)} ${(d.fracJND * 100).toFixed(2).padStart(6)}% | ${fmt(md, 3).padStart(14)} ${(fl * 100).toFixed(3).padStart(9)}% | ${fmt(st.p50, 2).padStart(7)} ${fmt(st.p95, 2).padStart(6)}`);
  });
  writeFileSync(path.join(OUT, 'webgpu-aa.json'), JSON.stringify({ res: [W, H], shotFrames, seq: seqRes, timing: t }, null, 1));
}

function pickVariantsFrom(names: string[]): Array<{ name: string; variant: TerrainVariant }> {
  return names.map((name) => ({ name, variant: VARIANTS[name] }));
}

/**
 * Forest placement: the GPU's window of plants (forestGen, read back after a
 * shot) against the CPU twin (forest/placement.ts, plantInCell), cell by cell.
 */
async function parity(): Promise<void> {
  const config = opt('config', 'biomes');
  const framed: CameraFrame[] = JSON.parse(readFileSync(path.join(OUT, '.cache', `framed-${config}.json`), 'utf8'));
  const frames = opt('frames', '0,6,10,14,16').split(',').map(Number);
  const gpu = await runWebGPU({ modes: ['shots'], variants: pickVariants('world-taa'), frames: framed, shotFrames: frames, forestDump: true });
  const seed = DEFAULT_SEED;
  const world: PlaceWorld = {
    onRuin: (x, z) => { const r = ruinSiteAt(x, z, seed); return !!r && Math.hypot(x - r.x, z - r.z) < r.radius; },
    islandTop: (x, z) => { const i = islandAt(x, z, seed); return i && islandRho(i, x, z) < 0.82 ? islandTop(i, x, z) : -Infinity; },
  };
  console.log('forest placement, GPU vs CPU twin: cells, plants (GPU / CPU), agree, only GPU, only CPU, other species, moved (> 5 cm)');
  for (const b of gpu.blobs.filter((x) => x.query.kind === 'forest')) {
    const side = Number(b.query.side);
    const buf = b.data.buffer.slice(b.data.byteOffset, b.data.byteOffset + b.data.byteLength);
    const tiles = new Int32Array(buf, 0, side * side * 4);
    const slots = new Uint32Array(buf, side * side * 16);
    const slotF = new Float32Array(buf, side * side * 16);
    let cells = 0, gpuN = 0, cpuN = 0, agree = 0, onlyG = 0, onlyC = 0, other = 0, moved = 0;
    for (let t = 0; t < side * side; t++) {
      const tx = tiles[4 * t], tz = tiles[4 * t + 1];
      for (let c = 0; c < 64; c++) {
        const i = (t * 64 + c) * 4;
        const valid = (slots[i + 3] & 0x40000000) !== 0;
        const p = plantInCell(tx * 8 + (c % 8), tz * 8 + Math.floor(c / 8), seed, world);
        cells++;
        if (valid) gpuN++;
        if (p) cpuN++;
        if (!valid && !p) { agree++; continue; }
        if (valid && !p) { onlyG++; continue; }
        if (!valid && p) { onlyC++; continue; }
        if (SPECIES[slots[i + 3] & 15] !== p!.species) { other++; continue; }
        agree++;
        if (Math.hypot(tx * 48 + slotF[i] - p!.x, tz * 48 + slotF[i + 1] - p!.z) > 0.05) moved++;
      }
    }
    console.log(`  f${b.query.frame}: ${cells} cells, ${gpuN} / ${cpuN} plants, ${(100 * agree / cells).toFixed(3)} % agree, ${onlyG} only GPU, ${onlyC} only CPU, ${other} other species, ${moved} moved`);
  }
}

const modes: Record<string, () => Promise<void>> = { port, cost, gen, fidelity, hero, aa, stills, parity };
if (!modes[mode]) throw new Error(`unknown mode ${mode}; one of ${Object.keys(modes).join(', ')}`);
await modes[mode]();
