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

export type VariantKind = 'render' | 'instrument' | 'probe' | 'evalcost';
export type Variant = { name: string; vs: string; fs: string };

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

export function transformShader(fs: string, kind: VariantKind, evalK = 16): string {
  if (kind === 'render') return fs;
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
};

export type BlobRecord = { query: Record<string, string>; data: Uint8Array };

export type SessionResult = {
  env: Record<string, unknown>;
  compile: Array<{ name: string; compileMs: number }>;
  results: Record<string, unknown>;
  blobs: BlobRecord[];
};

export async function runSession(cfg: SessionConfig): Promise<SessionResult> {
  let env: Record<string, unknown> = {};
  let compile: SessionResult['compile'] = [];
  const blobs: BlobRecord[] = [];
  let resolveDone!: (r: Record<string, unknown>) => void;
  let rejectDone!: (e: Error) => void;
  const done = new Promise<Record<string, unknown>>((res, rej) => { resolveDone = res; rejectDone = rej; });

  const html = `<!doctype html><html><head><meta charset="utf-8"><title>W(x,y) GPU bench</title></head>
<body style="margin:0;background:#000"><canvas id="c"></canvas><script src="/page.js"></script></body></html>`;

  const server = Bun.serve({
    port: 0,
    maxRequestBodySize: 1 << 30,
    async fetch(req) {
      const url = new URL(req.url);
      switch (url.pathname) {
        case '/': return new Response(html, { headers: { 'content-type': 'text/html' } });
        case '/page.js': return new Response(Bun.file(PAGE_JS), { headers: { 'content-type': 'text/javascript' } });
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

  const chrome = await launchChrome({ url: `http://127.0.0.1:${server.port}/`, software: cfg.software });
  const timeout = cfg.timeoutMs ?? 20 * 60_000;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const results = await Promise.race([
      done,
      new Promise<never>((_, rej) => { timer = setTimeout(() => rej(new Error(`GPU session timed out after ${timeout} ms`)), timeout); }),
    ]);
    return { env, compile, results, blobs };
  } finally {
    clearTimeout(timer);
    await chrome.close();
    server.stop(true);
  }
}
