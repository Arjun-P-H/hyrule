/**
 * Third-person hero harness (headless Chrome, ANGLE/Metal, the production shader).
 * ---------------------------------------------------------------------------
 *   bun bench/hero.ts gallery    stills of every locomotion state, simulated with
 *                                the real locomotion / rig / camera modules
 *                                → bench/out/hero/*.png
 *   bun bench/hero.ts cost       GPU cost, interleaved per frame along the
 *                                benchmark camera path with the hero framed
 *                                where the third-person camera puts it (4.6 m
 *                                ahead on the view axis), running and
 *                                paragliding. Variants: `terrain` (the shader
 *                                first person and fly compile: no hero code),
 *                                `hero-off` (withHero() program, hero disabled),
 *                                `hero-on`, and the pinned revision `prev@…`
 *                                (first person must be unchanged: timing and
 *                                bit-identical frames) → bench/out/hero-cost.json
 *   bun bench/hero.ts cpu        per-frame CPU cost of the third-person path
 *                                (updateHero + rig + camera + packing) next to
 *                                the first-person updatePlayer tick, on one
 *                                scripted 60 s walk: ns/tick in JSC (Bun) and
 *                                V8 (Node), and V8 bytes allocated per tick in
 *                                GC-free windows with a positive control
 *                                (method of bench/alloc.ts)
 */

import path from 'node:path';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { runSession, runSessionPerPreset, type Variant, type SessionConfig } from './gpu/runner';
import { collectShots, diffSets } from './shots';
import { loadShader, WORKTREE } from './lib/revision';
import { median, pairedRatio, summarize } from './lib/stats';
import type { CameraFrame } from './spline';
import { elevation, slope, softShadow } from '../src/lib/terrainMath';
import { withHero } from '../src/components/terrain/demoShader';
import { CAMERA, CHUNK_SIZE, DEFAULT_SEED, HERO_CAMERA, LIGHT, QUALITY_PRESETS, SPAWN, VALLEY, type QualityPreset } from '../src/lib/constants';
import { createHeroState, placeHero, updateHero, type HeroInput, type HeroState } from '../src/lib/hero/locomotion';
import { createHeroRig, updateHeroRig, packHeroUniforms, HERO_UNIFORM, type HeroRig } from '../src/lib/hero/rig';
import { createHeroCamera, updateHeroCamera } from '../src/lib/hero/camera';

const ROOT = path.resolve(import.meta.dir, '..');
const OUT = path.join(ROOT, 'bench/out');
const SEED = DEFAULT_SEED;
const DT = 1 / 60;
const args = process.argv.slice(2);
const mode = args[0] ?? 'gallery';
const opt = (k: string, d: string) => args.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;

// ---------------------------------------------------------------------------
// Scripted scenarios on the real terrain around the spawn
// ---------------------------------------------------------------------------
function findSpot(pred: (x: number, z: number) => boolean, step = 3, radius = 600): [number, number] {
  for (let r = 0; r <= radius; r += step) {
    for (let a = 0; a < 64; a++) {
      const x = SPAWN.x + r * Math.cos((a / 64) * Math.PI * 2);
      const z = SPAWN.z + r * Math.sin((a / 64) * Math.PI * 2);
      if (pred(x, z)) return [x, z];
      if (r === 0) break;
    }
  }
  throw new Error('no spot found');
}

const sl = Math.hypot(...LIGHT.sunDir);
const SUN: [number, number, number] = [LIGHT.sunDir[0] / sl, LIGHT.sunDir[1] / sl, LIGHT.sunDir[2] / sl];
const sunlit = (x: number, z: number) => softShadow([x, elevation(x, z, SEED, 0) + 0.3, z], SUN, SEED, 40, 0.002) > 0.95;
/** Heading whose next `len` metres stay flattest (so a run stays a run). */
function flattestYaw(x: number, z: number, len = 16): number {
  let best = 0, bestM = Infinity;
  for (let k = 0; k < 24; k++) {
    const yaw = (k / 24) * Math.PI * 2;
    let m = 0;
    for (let d = 1; d <= len; d += 1) m = Math.max(m, slope(x + Math.sin(yaw) * d, z + Math.cos(yaw) * d, SEED));
    if (m < bestM) { bestM = m; best = yaw; }
  }
  return best;
}
const flat = findSpot((x, z) => slope(x, z, SEED) < 0.08 && elevation(x, z, SEED, 0) > VALLEY.wLevel + 3 && sunlit(x, z), 4, 1500);
const flatYaw = flattestYaw(flat[0], flat[1]);
const deep = findSpot((x, z) => elevation(x, z, SEED, 0) < VALLEY.wLevel - 3, 2);
// A wall: steep for several metres uphill, so a climb has room.
const wall = findSpot((x, z) => {
  const g = 0.25;
  const e = elevation(x, z, SEED, 0);
  const gx = (elevation(x + g, z, SEED, 0) - e) / g, gz = (elevation(x, z + g, SEED, 0) - e) / g;
  const m = Math.hypot(gx, gz);
  if (m < 1.35) return false;
  for (let k = 1; k <= 4; k++) if (slope(x + (gx / m) * k, z + (gz / m) * k, SEED) < 1.2) return false;
  return true;
}, 2);

type Scenario = {
  name: string;
  start: [number, number];
  yaw?: number;
  setup?: (s: HeroState) => void;
  input: (t: number, s: HeroState) => Partial<HeroInput>;
  seconds: number;
  view: { dYaw: number; pitch: number; zoom: number };
};

const uphill = (x: number, z: number) => {
  const e = elevation(x, z, SEED, 0);
  return Math.atan2(elevation(x + 0.25, z, SEED, 0) - e, elevation(x, z + 0.25, SEED, 0) - e);
};
const wallYaw = uphill(wall[0], wall[1]);
const behind = (pitch = -0.24, zoom: number = HERO_CAMERA.distance) => ({ dYaw: 0, pitch, zoom });
const side = (pitch = -0.08, zoom = 3.2) => ({ dYaw: Math.PI / 2, pitch, zoom });
const front = (pitch = -0.06, zoom = 2.4) => ({ dYaw: Math.PI, pitch, zoom });
const run = { forward: 1 };

const SCENARIOS: Scenario[] = [
  { name: 'portrait', start: flat, input: () => ({}), seconds: 1.5, view: { dYaw: Math.PI - 0.35, pitch: 0.02, zoom: 1.7 } },
  { name: 'idle-behind', start: flat, input: () => ({}), seconds: 1.5, view: behind() },
  { name: 'idle-front', start: flat, input: () => ({}), seconds: 1.5, view: front() },
  { name: 'idle-side', start: flat, input: () => ({}), seconds: 1.5, view: side() },
  { name: 'idle-three-quarter', start: flat, input: () => ({}), seconds: 1.5, view: { dYaw: 2.4, pitch: -0.12, zoom: 2.8 } },
  { name: 'run-side-a', start: flat, input: () => run, seconds: 1.3, view: side() },
  { name: 'run-side-b', start: flat, input: () => run, seconds: 1.42, view: side() },
  { name: 'run-behind', start: flat, input: () => run, seconds: 1.3, view: behind() },
  { name: 'sprint-side', start: flat, input: () => ({ ...run, sprint: true }), seconds: 1.4, view: side() },
  { name: 'sneak-side', start: flat, input: () => ({ ...run, sneak: true }), seconds: 1.4, view: side() },
  { name: 'tired-side', start: flat, setup: (s) => { s.stamina = 0; s.exhausted = true; }, input: () => run, seconds: 1.4, view: side() },
  { name: 'jump-rise', start: flat, input: (t) => ({ ...run, jump: t > 0.6 && t < 0.7 }), seconds: 0.78, view: side() },
  { name: 'jump-fall', start: flat, input: (t) => ({ ...run, jump: t > 0.6 && t < 0.7 }), seconds: 1.05, view: side() },
  {
    name: 'glide-behind', start: flat, setup: (s) => placeHero(s, s.x, s.ground + 30, s.z, 0, 0, 0, s.yaw, SEED),
    input: (t) => ({ forward: 1, jump: t > 0.3 && t < 0.4 }), seconds: 2.2, view: behind(-0.2, 5.5),
  },
  {
    name: 'glide-side', start: flat, setup: (s) => placeHero(s, s.x, s.ground + 30, s.z, 0, 0, 0, s.yaw, SEED),
    input: (t) => ({ forward: 1, jump: t > 0.3 && t < 0.4 }), seconds: 2.2, view: side(-0.05, 4.5),
  },
  { name: 'fall-flail', start: flat, setup: (s) => placeHero(s, s.x, s.ground + 40, s.z, 0, 0, 0, s.yaw, SEED), input: () => ({}), seconds: 1.2, view: side(-0.05, 4) },
  {
    name: 'climb-side', start: [wall[0] - Math.sin(wallYaw) * 0.6, wall[1] - Math.cos(wallYaw) * 0.6], yaw: wallYaw,
    input: () => run, seconds: 1.6, view: { dYaw: Math.PI / 2, pitch: -0.1, zoom: 3.4 },
  },
  {
    name: 'climb-behind', start: [wall[0] - Math.sin(wallYaw) * 0.6, wall[1] - Math.cos(wallYaw) * 0.6], yaw: wallYaw,
    input: () => run, seconds: 1.6, view: behind(-0.3, 4.2),
  },
  { name: 'swim-behind', start: deep, input: () => run, seconds: 1.5, view: behind(-0.35, 4.2) },
  { name: 'swim-side', start: deep, input: () => run, seconds: 1.5, view: side(-0.2, 3.4) },
];

function simulate(sc: Scenario): { s: HeroState; rig: HeroRig; frame: CameraFrame } {
  const yaw = sc.yaw ?? flatYaw;
  const s = createHeroState(sc.start[0], sc.start[1], yaw, SEED);
  sc.setup?.(s);
  const rig = createHeroRig();
  const cam = createHeroCamera();
  cam.zoom = sc.view.zoom;
  const input: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: yaw };
  const steps = Math.round(sc.seconds / DT);
  for (let i = 0; i < steps; i++) {
    const t = i * DT;
    const want = sc.input(t, s);
    input.forward = want.forward ?? 0;
    input.strafe = want.strafe ?? 0;
    input.sprint = want.sprint ?? false;
    input.jump = want.jump ?? false;
    input.sneak = want.sneak ?? false;
    updateHero(s, input, DT, SEED);
    updateHeroRig(rig, s, DT);
    updateHeroCamera(cam, s, s.yaw + sc.view.dYaw, sc.view.pitch, DT, SEED);
  }
  const packed = new Float32Array(HERO_UNIFORM.length);
  packHeroUniforms(rig, s, cam.x, cam.y, cam.z, packed);
  return {
    s, rig,
    frame: { frame: 0, x: cam.x, y: cam.y, z: cam.z, yaw: cam.yaw, pitch: cam.pitch, scene: sc.name, hero: Array.from(packed) },
  };
}

function baseConfig(frames: CameraFrame[], presets: QualityPreset[], w: number, h: number): Omit<SessionConfig, 'modes' | 'variants'> {
  return {
    presets, frames, seed: SEED, fov: CAMERA.fov, chunkSize: CHUNK_SIZE, baseW: w, baseH: h,
    reps: Number(opt('reps', '3')), warmupMs: 2000, depth: 1, calibrateDraws: 8, calibrateFrames: [0],
    sentinelEvery: 10, shotFrames: frames.map((_, i) => i), instrumentFrames: [0], probes: [],
    evalRes: 256, evalK: 16, evalDps: [0.01], evalBases: [[0, 0]], evalReps: 1,
  };
}

async function gallery(): Promise<void> {
  const frames: CameraFrame[] = [];
  const only = opt('only', '');
  const list = SCENARIOS.filter((s) => !only || only.split(',').some((o) => s.name.includes(o)));
  for (const sc of list) {
    const { s, frame } = simulate(sc);
    frames.push({ ...frame, frame: frames.length });
    console.log(`  ${sc.name.padEnd(20)} move=${s.move.padEnd(6)} stamina=${s.stamina.toFixed(2)} ` +
      `pos=(${s.x.toFixed(1)}, ${s.y.toFixed(2)}, ${s.z.toFixed(1)}) ground=${s.ground.toFixed(2)}`);
  }
  // Ground-level third-person views for other renderers (bench/webgpu.ts hero --config=gallery).
  mkdirSync(path.join(OUT, '.cache'), { recursive: true });
  writeFileSync(path.join(OUT, '.cache', 'framed-gallery.json'), JSON.stringify(frames));
  const w = Number(opt('w', '960')), h = Number(opt('h', '540'));
  const preset: QualityPreset = { ...QUALITY_PRESETS[2], resScale: 1 };
  const src = await loadShader(WORKTREE);
  const variants: Variant[] = [{ name: 'hero', vs: src.vs, fs: withHero(src.fs), hero: true }];
  const res = await runSession({ ...baseConfig(frames, [preset], w, h), modes: ['shots'], variants });
  const dir = path.join(OUT, 'hero');
  mkdirSync(dir, { recursive: true });
  const { shots } = collectShots(res.blobs, null);
  const { encodePNG } = await import('./lib/image');
  frames.forEach((f, i) => {
    const img = shots.get(`hero|${preset.id}|${i}`);
    if (img) writeFileSync(path.join(dir, `${f.scene}.png`), encodePNG(img, true));
  });
  console.log(`  compile ${res.compile.map((c) => `${c.name} ${c.compileMs.toFixed(0)} ms`).join(', ')} → ${dir}`);
}

/** Hero posed so the third-person look-at point sits 4.6 m down the view axis. */
function framedHero(f: CameraFrame, gliding: boolean, t: number): number[] {
  const cp = Math.cos(f.pitch);
  const fx = Math.sin(f.yaw) * cp, fy = Math.sin(f.pitch), fz = Math.cos(f.yaw) * cp;
  const d = HERO_CAMERA.distance;
  const x = f.x + fx * d, z = f.z + fz * d;
  const y = f.y + fy * d - (gliding ? 1.7 : HERO_CAMERA.targetH);
  const s = createHeroState(x, z, f.yaw, SEED);
  placeHero(s, x, y, z, Math.sin(f.yaw) * 5.4, 0, Math.cos(f.yaw) * 5.4, f.yaw, SEED);
  s.move = gliding ? 'glide' : 'ground';
  s.gx = 0; s.gz = 0;
  const rig = createHeroRig();
  for (let i = 0; i < 40 + (t % 17); i++) updateHeroRig(rig, s, DT);
  const packed = new Float32Array(HERO_UNIFORM.length);
  packHeroUniforms(rig, s, f.x, f.y, f.z, packed);
  return Array.from(packed);
}

async function cost(): Promise<void> {
  const path0 = JSON.parse(readFileSync(path.join(ROOT, 'bench/baseline/camera_path.json'), 'utf8')) as CameraFrame[];
  const src = await loadShader(WORKTREE);
  const base = await loadShader(opt('against', 'HEAD'));
  // Long High-preset sessions can wedge the ANGLE/Metal GPU process, so every
  // (configuration, preset) pair runs in its own browser session and is saved
  // as soon as it finishes; `--frames=N` subsamples the path for shorter ones.
  const presets = QUALITY_PRESETS.filter((p) => opt('presets', 'low,med,high').split(',').includes(p.id));
  const nFrames = Number(opt('frames', String(path0.length)));
  const pathN = Array.from({ length: nFrames }, (_, i) => path0[Math.round((i * (path0.length - 1)) / Math.max(1, nFrames - 1))]);
  const file = path.join(OUT, 'hero-cost.json');
  mkdirSync(OUT, { recursive: true });
  const out: Record<string, Record<string, unknown>> = existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : {};
  for (const config of opt('configs', 'run,glide').split(',')) {
    const gliding = config === 'glide';
    const frames = pathN.map((f, i) => ({ ...f, frame: i, hero: framedHero(f, gliding, i) }));
    const variants: Variant[] = [
      { name: 'terrain', vs: src.vs, fs: src.fs },
      { name: 'hero-off', vs: src.vs, fs: withHero(src.fs) },
      { name: 'hero-on', vs: src.vs, fs: withHero(src.fs), hero: true },
    ];
    if (!gliding) variants.push({ name: `prev@${base.rev}`, vs: base.vs, fs: base.fs });
    const vi = (name: string) => variants.findIndex((v) => v.name === name);
    for (const preset of presets) {
      const res = await runSessionPerPreset({ ...baseConfig(frames, [preset], 1280, 720), modes: ['timing', 'shots'], variants });
      const t = (res.results.timing as Array<{ preset: string; samples: Array<{ v: number; f: number; s: number; ms: number }> }>)[0];
      const per = variants.map((_, v) => frames.map((_, f) => median(t.samples.filter((x) => x.v === v && x.f === f && !x.s).map((x) => x.ms))));
      const ratio = (a: string, b: string) => pairedRatio(per[vi(b)], per[vi(a)]); // a over b (pairedRatio(x, y) = y / x)
      const { shots, hashes } = collectShots(res.blobs, path.join(OUT, 'hero-cost', `${config}-${preset.id}`));
      const frameIdx = frames.map((_, i) => i);
      const identical = (a: string, b: string) => {
        let same = 0;
        for (const f of frameIdx) if (hashes[`${a}|${preset.id}|${f}`] === hashes[`${b}|${preset.id}|${f}`]) same++;
        return `${same}/${frameIdx.length}`;
      };
      const block = {
        frames: frames.length,
        reps: Number(opt('reps', '3')),
        summary: Object.fromEntries(variants.map((v, k) => [v.name, summarize(t.samples.filter((x) => x.v === k && !x.s).map((x) => x.ms))])),
        heroOnOverTerrain: ratio('hero-on', 'terrain'),    // third person vs first person, same view
        heroOnOverOff: ratio('hero-on', 'hero-off'),       // drawing the hero
        heroCodeOverTerrain: ratio('hero-off', 'terrain'), // hero program, hero disabled
        ...(gliding ? {} : { terrainOverPrev: ratio('terrain', `prev@${base.rev}`) }),
        bitIdentical: {
          heroOffVsTerrain: identical('hero-off', 'terrain'),
          ...(gliding ? {} : { terrainVsPrev: identical('terrain', `prev@${base.rev}`) }),
        },
        onVsOff: diffSets(shots, 'hero-off', preset.id, 'hero-on', preset.id, frameIdx),
        compile: res.compile,
        meta: { worktree: src.rev, against: base.rev, date: new Date().toISOString() },
      };
      (out[config] ??= {})[preset.id] = block;
      writeFileSync(file, JSON.stringify(out, null, 1));
      const f = (r: { ratio: number; lo: number; hi: number }) => `${r.ratio.toFixed(3)} (${r.lo.toFixed(3)}–${r.hi.toFixed(3)})`;
      console.log(`  ${config} ${preset.id}: hero-on/terrain ${f(block.heroOnOverTerrain)} · hero-on/off ${f(block.heroOnOverOff)} · ` +
        `hero-code/terrain ${f(block.heroCodeOverTerrain)}${'terrainOverPrev' in block ? ` · terrain/prev ${f(block.terrainOverPrev as never)}` : ''} · ` +
        `bit-identical ${JSON.stringify(block.bitIdentical)}`);
    }
  }
}

// ---------------------------------------------------------------------------
// CPU: one tick of each path on the same scripted walk
// ---------------------------------------------------------------------------
const CPU_ENTRY = `
export { elevation, updatePlayer } from '${ROOT}/src/lib/terrainMath.ts';
export { createHeroState, placeHero, updateHero } from '${ROOT}/src/lib/hero/locomotion.ts';
export { createHeroRig, updateHeroRig, packHeroUniforms, HERO_UNIFORM } from '${ROOT}/src/lib/hero/rig.ts';
export { createHeroCamera, updateHeroCamera } from '${ROOT}/src/lib/hero/camera.ts';
`;

// Plain JS so the identical driver runs in Bun (JSC) and Node (V8).
const CPU_DRIVER = String.raw`
const [modPath, ticksArg, engine] = typeof Bun !== 'undefined' ? Bun.argv.slice(2) : process.argv.slice(2);
const m = await import(modPath);
const TICKS = Number(ticksArg);
const SEED = 1337, SX = 2016, SZ = -2592, YAW = 2.056, DT = 1 / 60;
const pin = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };
const hin = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };
function script(i) {
  const f = (i % 900) < 780 ? 1 : 0, st = (i % 240) < 60 ? 1 : (i % 240) < 120 ? -1 : 0;
  const sp = (i % 600) < 300, j = (i % 90) < 6, yaw = YAW + Math.sin(i * 0.004) * 0.9;
  pin.forward = f; pin.strafe = st; pin.sprint = sp; pin.jump = (i % 90) === 0; pin.yaw = yaw;
  hin.forward = f; hin.strafe = st; hin.sprint = sp; hin.jump = j; hin.camYaw = yaw;
}
let sink = 0;
const p = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, grounded: true };
const hero = m.createHeroState(SX, SZ, YAW, SEED);
let rig = m.createHeroRig(), cam = m.createHeroCamera();
// Every measured walk starts from the same state, so windows compare equal work.
function resetHero() {
  m.placeHero(hero, SX, m.elevation(SX, SZ, SEED, 0) + 0.08, SZ, 0, 0, 0, YAW, SEED);
  hero.stamina = 1; hero.exhausted = false; hero.jumpHeld = false;
  rig = m.createHeroRig(); cam = m.createHeroCamera();
}
const buf = new Float32Array(m.HERO_UNIFORM.length);
const ring = new Array(1024).fill(null);
function playerWalk(n) {
  p.x = SX; p.y = m.elevation(SX, SZ, SEED, 0) + 1.78; p.z = SZ; p.vx = p.vy = p.vz = 0; p.grounded = true;
  for (let i = 0; i < n; i++) { script(i); m.updatePlayer(p, pin, DT, SEED); }
  sink += p.x;
}
function heroWalk(n, control) {
  for (let i = 0; i < n; i++) {
    script(i);
    m.updateHero(hero, hin, DT, SEED);
    m.updateHeroRig(rig, hero, DT);
    m.updateHeroCamera(cam, hero, hin.camYaw, -0.24, DT, SEED);
    m.packHeroUniforms(rig, hero, cam.x, cam.y, cam.z, buf);
    if (control) ring[i & 1023] = { a: hero.x, b: i }; // positive control: escapes, must allocate
  }
  sink += hero.x + buf[0];
}
for (let k = 0; k < 3; k++) { playerWalk(20000); heroWalk(20000, false); heroWalk(5000, true); }

// Runs alternate player / hero so load and clock changes hit both equally.
const runsP = [], runsH = [];
for (let k = 0; k < 11; k++) {
  let t0 = performance.now(); playerWalk(TICKS); runsP.push((performance.now() - t0) * 1e6 / TICKS);
  resetHero();
  t0 = performance.now(); heroWalk(TICKS, false); runsH.push((performance.now() - t0) * 1e6 / TICKS);
}
const stats = (runs) => { const r = [...runs].sort((a, b) => a - b); return { median: r[5], q1: r[3], q3: r[7] }; };
const ratios = runsH.map((h, k) => h / runsP[k]).sort((a, b) => a - b);
const out = { engine, ticks: TICKS, playerNs: stats(runsP), heroNs: stats(runsH), heroOverPlayer: { median: ratios[5], q1: ratios[3], q3: ratios[7] } };
if (engine === 'v8') {
  const v8 = await import('node:v8');
  async function bytesPerTick(fn) {
    const vals = [];
    for (let k = 0; k < 7; k++) {
      resetHero();
      globalThis.gc();
      await new Promise((r) => setTimeout(r, 5));
      const prof = new v8.GCProfiler();
      prof.start();
      const h0 = process.memoryUsage().heapUsed;
      fn();
      const h1 = process.memoryUsage().heapUsed;
      if (prof.stop().statistics.length !== 0) continue;
      vals.push((h1 - h0) / TICKS);
    }
    vals.sort((a, b) => a - b);
    return { median: vals[Math.floor(vals.length / 2)], valid: vals.length };
  }
  out.controlBytes = await bytesPerTick(() => heroWalk(TICKS, true));
  out.heroBytes = await bytesPerTick(() => heroWalk(TICKS, false));
  out.playerBytes = await bytesPerTick(() => playerWalk(TICKS));
}
out.finite = Number.isFinite(sink);
console.log(JSON.stringify(out));
`;

async function cpu(): Promise<void> {
  const cache = path.join(OUT, '.cache');
  mkdirSync(cache, { recursive: true });
  const entry = path.join(cache, 'hero-cpu-entry.ts');
  writeFileSync(entry, CPU_ENTRY);
  const built = await Bun.build({ entrypoints: [entry], target: 'node', format: 'esm', minify: false });
  if (!built.success) throw new Error(built.logs.map(String).join('\n'));
  const mod = path.join(cache, 'hero-cpu.mjs');
  writeFileSync(mod, await built.outputs[0].text());
  const driver = path.join(cache, 'hero-cpu-driver.mjs');
  writeFileSync(driver, CPU_DRIVER);
  const ticks = opt('ticks', '3600');
  const { $ } = await import('bun');
  const jsc = JSON.parse((await $`bun ${driver} ${mod} ${ticks} jsc`.text()).trim().split('\n').pop()!);
  const v8 = JSON.parse((await $`node --expose-gc --min-semi-space-size=128 --max-semi-space-size=128 ${driver} ${mod} ${ticks} v8`.text()).trim().split('\n').pop()!);
  if (!(v8.controlBytes.median > v8.heroBytes.median + 16)) throw new Error('positive control not detected — allocation method invalid');
  const result = { jsc, v8 };
  writeFileSync(path.join(OUT, 'hero-cpu.json'), JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result, null, 1));
}

if (mode === 'gallery') await gallery();
else if (mode === 'cost') await cost();
else if (mode === 'frames') {
  // the cost workload as JSON, for ad-hoc shader experiments with the runner
  const path0 = JSON.parse(readFileSync(path.join(ROOT, 'bench/baseline/camera_path.json'), 'utf8')) as CameraFrame[];
  mkdirSync(path.join(OUT, '.cache'), { recursive: true });
  for (const config of ['run', 'glide']) {
    const frames = path0.map((f, i) => ({ ...f, hero: framedHero(f, config === 'glide', i) }));
    writeFileSync(path.join(OUT, '.cache', `framed-${config}.json`), JSON.stringify(frames));
  }
}
else if (mode === 'cpu') await cpu();
else throw new Error(`unknown mode ${mode}`);
