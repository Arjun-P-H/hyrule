/**
 * GPU benchmark session runner.
 * ---------------------------------------------------------------------------
 * Serves bench/gpu/page.js to headless Chrome, hands it a workload over
 * /config, and collects results (JSON + binary readbacks). Shader variants are
 * derived from the production TERRAIN_FS of any git revision with small,
 * anchored text transforms, so a pinned baseline and the working tree can be
 * measured side by side in one browser session:
 *
 *   render     — the production shader, unmodified
 *   instrument — counters injected into elevation() and the march loop
 *   probe      — production main renamed; main() writes elevation(g) + fp32 g
 *   evalcost   — production main renamed; main() runs K elevation() calls/px
 */

import path from 'node:path';
import { launchChrome } from '../lib/chrome';
import type { CameraFrame } from '../spline';
import type { QualityPreset } from '../../src/lib/constants';

const PAGE_JS = path.join(import.meta.dir, 'page.js');

export type VariantKind = 'render' | 'instrument' | 'probe' | 'evalcost' | 'truth';
/**
 * `budgets` lets a revision render with its own preset step budgets (a pinned
 * baseline keeps the march/shadow steps it shipped with) instead of the
 * working tree's QUALITY_PRESETS. `hero` uploads each frame's packed hero
 * uniforms (CameraFrame.hero); without it the hero is disabled.
 */
export type Variant = {
  name: string;
  vs: string;
  fs: string;
  budgets?: Record<string, { marchSteps: number; shadowSteps: number }>;
  hero?: boolean;
};

function mustFind(src: string, anchor: string): number {
  const i = src.indexOf(anchor);
  if (i < 0) throw new Error(`shader transform: anchor not found: "${anchor}"`);
  if (src.indexOf(anchor, i + anchor.length) >= 0) throw new Error(`shader transform: anchor not unique: "${anchor}"`);
  return i;
}

function insertAfter(src: string, anchor: string, text: string): string {
  const i = mustFind(src, anchor) + anchor.length;
  return src.slice(0, i) + text + src.slice(i);
}

function renameMain(src: string): string {
  const i = mustFind(src, 'void main() {');
  return src.slice(0, i) + 'void w_prod_main() {' + src.slice(i + 'void main() {'.length);
}

// Optional anchors: functions that only exist in some revisions. Each entry is
// [anchor, counter]; missing anchors are skipped, duplicate ones are errors.
const OPTIONAL_COUNTERS: Array<[string, string]> = [
  ['float elevationBound(vec2 g, float dp, int seed) {', 'W_IB'],
];

/**
 * Converged ground truth for perceptual scoring: tiny horizon clamp, no step
 * growth, gradient refreshed every 2nd sample without gating, 3,000-step and
 * 600-shadow-step loops, 14 bisection steps. Independent of the march
 * parameters under test, unlike rendering a revision at its own loop caps.
 */
function truthShader(fs: string): string {
  const rep = (s: string, a: RegExp | string, b: string) => {
    const out = s.replace(a, b);
    if (out === s) throw new Error(`truth transform: anchor not found: ${String(a)}`);
    return out;
  };
  let s = rep(fs, /for \(int i = 0; i < 160; i\+\+\) \{\n\s*if \(i >= u_march_steps\) break;/, 'for (int i = 0; i < 3000; i++) {');
  s = rep(s, /if \(\(i % (?:3|GRAD_CADENCE)\) == 0\) \{/, 'if ((i % 2) == 0) {');
  s = rep(s, /if \((?:\(p\.y - e\) <= \(t \* 0\.06 \+ 2\.0\) \* 2\.5|!farAbove)\) \{/, 'if (true) {');
  s = rep(s, /step \*= 1\.0 \+ t \* GROWTH;[^\n]*\n/, '\n');
  s = rep(s, /step = min\(step, [^;]+\);/, 'step = min(step, t * 0.01 + 0.25);');
  s = rep(s, 'const int   BISECT  = 6;', 'const int   BISECT  = 14;');
  s = rep(s, /for \(int i = 0; i < 40; i\+\+\) \{\n\s*if \(i >= u_shadow_steps\) break;/, 'for (int i = 0; i < 600; i++) {');
  s = rep(s, /t \+= clamp\(h \* (?:0\.8|SH_STEP_SCALE), (?:0\.5|SH_STEP_MIN), (?:60\.0|SH_STEP_MAX)\) \+ t \* (?:0\.04|SH_GROWTH);/, 't += clamp(h * 0.5, 0.2, 15.0) + t * 0.01;');
  return s;
}

export function transformShader(fs: string, kind: VariantKind, evalK = 16): string {
  if (kind === 'render') return fs;
  if (kind === 'truth') return truthShader(fs);
  const globals =
    kind === 'instrument'
      ? '\nint W_IE = 0; int W_IS = 0; int W_IB = 0; int W_IX = 0;\n'
      : '\nuniform vec2 u_probe_local;\nuniform vec2 u_probe_step;\nuniform float u_probe_dp;\nuniform float u_bench_dp;\n';
  let out = insertAfter(fs, 'out vec4 fragColor;', globals);
  out = renameMain(out);
  if (kind === 'instrument') {
    out = insertAfter(out, 'float elevation(vec2 g, float dp, int seed) {', ' W_IE++;');
    out = insertAfter(out, 'for (int i = 0; i < 160; i++) {', ' W_IS++;');
    for (const [anchor, counter] of OPTIONAL_COUNTERS) {
      if (out.includes(anchor)) out = insertAfter(out, anchor, ` ${counter}++;`);
    }
    return `${out}
void main() {
  w_prod_main();
  fragColor = vec4(float(W_IE), float(W_IS), float(W_IB), float(W_IX));
}
`;
  }
  if (kind === 'probe') {
    return `${out}
void main() {
  vec2 local = u_probe_local + (gl_FragCoord.xy - 0.5) * u_probe_step;
  vec2 g = local + u_chunk_origin;
  fragColor = vec4(elevation(g, u_probe_dp, u_seed), g.x, g.y, 0.0);
}
`;
  }
  return `${out}
void main() {
  vec2 base = gl_FragCoord.xy * 3.0;
  float acc = 0.0;
  for (int k = 0; k < ${evalK}; k++) {
    vec2 g = base + vec2(float(k) * 97.0, float(k) * 53.0) + u_chunk_origin;
    acc += elevation(g, u_bench_dp, u_seed);
  }
  fragColor = vec4(fract(acc * 1e-3), 0.0, 0.0, 1.0);
}
`;
}

export type ProbeSpec = {
  id: string;
  origin: [number, number];
  local0: [number, number];
  step: number;
  w: number;
  h: number;
  dp: number;
};

export type SessionConfig = {
  modes: Array<'timing' | 'calibrate' | 'shots' | 'instrument' | 'probe' | 'evalcost'>;
  variants: Variant[];
  presets: QualityPreset[];
  frames: CameraFrame[];
  seed: number;
  fov: number;
  chunkSize: number;
  baseW: number;
  baseH: number;
  reps: number;
  warmupMs: number;
  depth: number;
  calibrateDraws: number;
  calibrateFrames: number[];
  sentinelEvery: number;
  shotFrames: number[];
  instrumentFrames: number[];
  probes: ProbeSpec[];
  evalRes: number;
  evalK: number;
  evalDps: number[];
  evalBases: Array<[number, number]>;
  evalReps: number;
  software?: boolean;
  timeoutMs?: number;
  stallMs?: number;
};

export type BlobRecord = { query: Record<string, string>; data: Uint8Array };

export type SessionResult = {
  env: Record<string, unknown>;
  compile: Array<{ name: string; compileMs: number }>;
  results: Record<string, unknown>;
  blobs: BlobRecord[];
};

/**
 * Another page than page.js: the JavaScript to serve as /page.js (e.g. the
 * bundled WebGPU page) and extra Chrome flags. The session protocol — /config,
 * /env, /compile, /progress, /blob, /heartbeat, /done, /error — is the same.
 */
export type PageOptions = { pageJs?: string; chromeArgs?: string[] };

export type SessionLimits = { software?: boolean; timeoutMs?: number; stallMs?: number };

export async function runSession(cfg: SessionConfig | (SessionLimits & Record<string, unknown>), page: PageOptions = {}): Promise<SessionResult> {
  const limits = cfg as SessionLimits;
  let env: Record<string, unknown> = {};
  let compile: SessionResult['compile'] = [];
  const blobs: BlobRecord[] = [];
  let resolveDone!: (r: Record<string, unknown>) => void;
  let rejectDone!: (e: Error) => void;
  const done = new Promise<Record<string, unknown>>((res, rej) => { resolveDone = res; rejectDone = rej; });

  const html = `<!doctype html><html><head><meta charset="utf-8"><title>W(x,y) GPU bench</title></head>
<body style="margin:0;background:#000"><canvas id="c"></canvas><script src="/page.js"></script></body></html>`;

  // Stall watchdog: every page message refreshes it; silence for stallMs aborts
  // the session with the last thing the page reported.
  let lastMessage = 'session start';
  let lastAt = Date.now();
  const touch = (what: string) => { lastMessage = what; lastAt = Date.now(); };
  const stallMs = limits.stallMs ?? 120_000;

  const server = Bun.serve({
    port: 0,
    maxRequestBodySize: 1 << 30,
    async fetch(req) {
      const url = new URL(req.url);
      touch(url.pathname);
      switch (url.pathname) {
        case '/heartbeat': {
          const h = await req.json();
          touch(`heartbeat ${h.done}/${h.total}`);
          return new Response('ok');
        }
        case '/': return new Response(html, { headers: { 'content-type': 'text/html' } });
        case '/page.js': return new Response(page.pageJs ?? Bun.file(PAGE_JS), { headers: { 'content-type': 'text/javascript' } });
        case '/config': return Response.json(cfg);
        case '/env': env = await req.json(); return new Response('ok');
        case '/compile': compile = await req.json(); return new Response('ok');
        case '/progress': {
          const p = await req.json();
          process.stdout.write(`      · ${p.mode} ${p.preset} done\n`);
          return new Response('ok');
        }
        case '/blob': {
          blobs.push({ query: Object.fromEntries(url.searchParams), data: new Uint8Array(await req.arrayBuffer()) });
          return new Response('ok');
        }
        case '/done': resolveDone(await req.json()); return new Response('ok');
        case '/error': {
          const e = await req.json();
          rejectDone(new Error(e.message));
          return new Response('ok');
        }
        default: return new Response('not found', { status: 404 });
      }
    },
  });

  const chrome = await launchChrome({ url: `http://127.0.0.1:${server.port}/`, software: limits.software, extraArgs: page.chromeArgs });
  const timeout = limits.timeoutMs ?? 20 * 60_000;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    let watchdog: ReturnType<typeof setInterval> | undefined;
    const stalled = new Promise<never>((_, rej) => {
      watchdog = setInterval(() => {
        if (Date.now() - lastAt > stallMs) rej(new Error(`GPU session stalled: no message for ${stallMs} ms (last: ${lastMessage})`));
      }, 5000);
    });
    const results = await Promise.race([
      done,
      stalled,
      new Promise<never>((_, rej) => { timer = setTimeout(() => rej(new Error(`GPU session timed out after ${timeout} ms`)), timeout); }),
    ]).finally(() => clearInterval(watchdog));
    return { env, compile, results, blobs };
  } finally {
    clearTimeout(timer);
    await chrome.close();
    server.stop(true);
  }
}

/**
 * Long sessions on ANGLE/Metal occasionally wedge the GPU process: a
 * synchronous WebGL query blocks the renderer indefinitely (0 % CPU, no
 * error). Bounding each browser session to one preset and retrying a stalled
 * preset once keeps a single wedge from losing a whole run.
 */
export async function runSessionPerPreset(cfg: SessionConfig, retries = 1): Promise<SessionResult> {
  const merged: SessionResult = { env: {}, compile: [], results: {}, blobs: [] };
  for (const preset of cfg.presets) {
    let attempt = 0;
    for (;;) {
      try {
        const r = await runSession({ ...cfg, presets: [preset] });
        merged.env = r.env;
        merged.compile = r.compile;
        for (const [mode, value] of Object.entries(r.results)) {
          const prev = merged.results[mode];
          merged.results[mode] = Array.isArray(prev) && Array.isArray(value) ? [...prev, ...value] : value;
        }
        merged.blobs.push(...r.blobs);
        break;
      } catch (e) {
        if (attempt++ >= retries || !/stalled|timed out/.test(String(e))) throw e;
        process.stdout.write(`      ! ${preset.id}: ${(e as Error).message} — retrying\n`);
      }
    }
  }
  return merged;
}
