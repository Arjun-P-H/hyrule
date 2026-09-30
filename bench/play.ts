/**
 * Scripted steady-state play session against the production build.
 * ---------------------------------------------------------------------------
 * Drives /game in headless Chrome (ANGLE/Metal, 1280×720, DPR 1) through the
 * DevTools protocol: first-person walk (or the third-person hero with
 * `mode: 'hero'`), W held, periodic jumps and strafes — the real 60 Hz path
 * (useTerrainEngine frame loop → updatePlayer / updateHero → uniforms → draw).
 *
 * Measures, without touching production code:
 *   - JS heap allocation rate: V8 young generation pinned at 64 MB so no
 *     scavenge runs in the window (verified by a monotonic usedSize trace),
 *     allocation = usedSize delta / seconds. Sampling heap profiler supplies
 *     per-script attribution.
 *   - rAF rate (= physics tick rate) via one pre-bound rAF counter, render
 *     rate via a drawArrays counter, DRS via canvas-size samples, in-game GPU
 *     time via timer queries wrapped around the engine's draw (optional).
 *   - fence A/B: `fenceOff` stubs gl.fenceSync → null, which makes the
 *     engine's `if (fence && …)` pacing check a no-op.
 *   - a real SpectorJS capture of one frame (commands, draw calls, programs,
 *     textures) — the previous "spector snapshot" was a hard-coded literal.
 */

import path from 'node:path';
import { readFileSync } from 'node:fs';
import { launchChrome, connectToPage, type CDPSession } from './lib/chrome';

const ROOT = path.resolve(import.meta.dir, '..');

export type PlayOptions = {
  seconds: number;
  preset?: 'low' | 'med' | 'high';
  fenceOff?: boolean;
  gpuTiming?: boolean;
  spector?: boolean;
  software?: boolean;
  dpr?: number;
  /** 'walk' (first person, the default) or 'hero' (third person). */
  mode?: 'walk' | 'hero';
  /** Seconds of play before the measured window (default 3: JIT tiers, DRS warm-up). */
  settleSeconds?: number;
  /**
   * Graphics API, as the `renderer` URL parameter (default 'webgl2', which keeps
   * sessions comparable with revisions that predate WebGPU); 'auto' opens /game
   * bare, as a player would. WebGPU frames are counted per GPUQueue.submit.
   */
  renderer?: 'webgl2' | 'webgpu' | 'auto';
  /** Play in a visible Chrome window (real display, vsync) instead of headless. */
  headed?: boolean;
  /** More URL parameters for /game, e.g. 'time=21' or 'sky=classic'. */
  query?: string;
};

export type PlayResult = {
  options: PlayOptions;
  seconds: number;
  rafPerSec: number;
  drawsPerSec: number;
  /** V8 sampling-profiler estimate incl. collected objects (validated within ~3% of GC-free ground truth in Node). */
  sampledBytesPerSec: number;
  /** usedSize delta / s; only meaningful when no GC ran (Chrome also collects on idle), else null. */
  heapBytesPerSec: number | null;
  allocByScript: Array<{ script: string; bytesPerSec: number; share: number }>;
  /** Top allocating functions (sampled), `script:function:line`. */
  allocByFunction: Array<{ fn: string; bytesPerSec: number; share: number }>;
  canvas: { first: [number, number]; last: [number, number]; minW: number; maxW: number; changes: number; reversals: number; timeline: Array<[number, number]> };
  gpuMs?: { n: number; p50: number; p95: number; mean: number };
  hudFps?: string;
  /** ms from navigation start: first contentful paint, DOM interactive, first engine draw */
  load?: { fcp: number; firstDraw: number; domInteractive: number };
  spector?: Record<string, unknown>;
};

// Injected before any page script. No allocation per frame: pre-bound
// functions, fixed-size typed arrays for samples.
const INSTRUMENT = (fenceOff: boolean, gpuTiming: boolean) => `(() => {
  const W = window.__wb = { raf: 0, draws: 0, firstDrawAt: 0, sizes: new Float64Array(3 * 4096), nSizes: 0, gpu: new Float64Array(8192), nGpu: 0, pending: [], ext: null };
  const raf = window.requestAnimationFrame.bind(window);
  function tick() { W.raf++; raf(tick); }
  raf(tick);
  const proto = WebGL2RenderingContext.prototype;
  const draw = proto.drawArrays;
  const timing = ${gpuTiming};
  proto.drawArrays = function (m, f, c) {
    if (W.draws === 0) W.firstDrawAt = performance.now();
    W.draws++;
    if (!timing) return draw.call(this, m, f, c);
    if (!W.ext) W.ext = this.getExtension('EXT_disjoint_timer_query_webgl2');
    for (let i = 0; i < W.pending.length; i++) {
      const q = W.pending[i];
      if (this.getQueryParameter(q, this.QUERY_RESULT_AVAILABLE)) {
        if (W.nGpu < W.gpu.length) W.gpu[W.nGpu++] = this.getQueryParameter(q, this.QUERY_RESULT) / 1e6;
        this.deleteQuery(q); W.pending.splice(i, 1); i--;
      }
    }
    const q = this.createQuery();
    this.beginQuery(W.ext.TIME_ELAPSED_EXT, q);
    draw.call(this, m, f, c);
    this.endQuery(W.ext.TIME_ELAPSED_EXT);
    W.pending.push(q);
  };
  ${fenceOff ? 'proto.fenceSync = function () { return null; };' : ''}
  // WebGPU: the engine submits one command buffer per rendered frame.
  if (window.GPUQueue) {
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function (bufs) {
      if (W.draws === 0) W.firstDrawAt = performance.now();
      W.draws++;
      return submit.call(this, bufs);
    };
  }
  setInterval(() => {
    const c = document.querySelector('canvas');
    if (c && W.nSizes < 4096) { const o = 3 * W.nSizes++; W.sizes[o] = performance.now(); W.sizes[o + 1] = c.width; W.sizes[o + 2] = c.height; }
  }, 100);
})();`;

// Only `code`/`key`/windowsVirtualKeyCode: on macOS `nativeVirtualKeyCode`
// is a kVK_* code, and passing Windows VK values there freezes the page.
async function key(cdp: CDPSession, code: string, keyName: string, vk: number, type: 'keyDown' | 'keyUp') {
  await cdp.send('Input.dispatchKeyEvent', { type, code, key: keyName, windowsVirtualKeyCode: vk });
}

async function evalJson<T>(cdp: CDPSession, expression: string): Promise<T> {
  const r = await cdp.send<{ result: { value: T } }>('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  return r.result.value;
}

export async function runPlaySession(baseUrl: string, opts: PlayOptions): Promise<PlayResult> {
  const chrome = await launchChrome({
    url: 'about:blank',
    remoteDebugging: true,
    software: opts.software,
    headed: opts.headed,
    extraArgs: ['--js-flags=--min-semi-space-size=64 --max-semi-space-size=64'],
  });
  try {
    const cdp = await connectToPage(chrome.wsEndpoint!);
    await cdp.send('Page.enable');
    await cdp.send('Runtime.enable');
    await cdp.send('HeapProfiler.enable');
    if (opts.dpr && opts.dpr !== 1) {
      await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: opts.dpr, mobile: false });
    }
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: INSTRUMENT(!!opts.fenceOff, !!opts.gpuTiming) });
    const loaded = new Promise<void>((r) => cdp.on('Page.loadEventFired', () => r()));
    const renderer = opts.renderer ?? 'webgl2';
    const params = [renderer === 'auto' ? '' : `renderer=${renderer}`, opts.query ?? ''].filter(Boolean).join('&');
    await cdp.send('Page.navigate', { url: `${baseUrl}/game${params ? `?${params}` : ''}` });
    await loaded;
    await Bun.sleep(1500);
    const load = await evalJson<{ fcp: number; firstDraw: number; domInteractive: number }>(cdp,
      `({ fcp: (performance.getEntriesByName('first-contentful-paint')[0] || {}).startTime || 0,
          firstDraw: __wb.firstDrawAt,
          domInteractive: performance.getEntriesByType('navigation')[0].domInteractive })`);

    // quality preset buttons are the first text buttons in the top-right bar
    if (opts.preset && opts.preset !== 'med') {
      const label = opts.preset === 'low' ? 'LOW' : 'HIGH';
      await evalJson(cdp, `[...document.querySelectorAll('button')].find(b => b.textContent.trim() === '${label}')?.click(), true`);
    }
    // focus page, enter the requested mode, hold W. Revisions without the
    // data-engine-mode marker start in fly mode (F → walk, V → hero); newer
    // ones start as the hero (V → walk).
    await cdp.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 640, y: 360 });
    const want = opts.mode ?? 'walk';
    let mode = '';
    for (let i = 0; i < 50 && !mode; i++) {
      mode = await evalJson<string>(cdp, `(() => { const m = document.querySelector('[data-engine-mode]'); return m ? m.getAttribute('data-engine-mode') : 'fly'; })()`);
      if (!mode) await Bun.sleep(200);
    }
    if (mode !== want) {
      const [code, name, vk] = want === 'walk' && mode === 'fly' ? ['KeyF', 'f', 70] as const : ['KeyV', 'v', 86] as const;
      await key(cdp, code, name, vk, 'keyDown');
      await key(cdp, code, name, vk, 'keyUp');
    }
    await key(cdp, 'KeyW', 'w', 87, 'keyDown');
    await Bun.sleep((opts.settleSeconds ?? 3) * 1000); // settle: JIT tiers, DRS warm-up window (2.5 s)

    await cdp.send('HeapProfiler.collectGarbage');
    await cdp.send('HeapProfiler.startSampling', { samplingInterval: 128, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
    const start = await evalJson<{ raf: number; draws: number; t: number; nSizes: number; nGpu: number }>(cdp,
      `({ raf: __wb.raf, draws: __wb.draws, t: performance.now(), nSizes: __wb.nSizes, nGpu: __wb.nGpu })`);
    const heap: number[] = [];
    const t0 = Date.now();
    let jumpAt = 0, strafe = 0;
    while (Date.now() - t0 < opts.seconds * 1000) {
      const { usedSize } = await cdp.send<{ usedSize: number }>('Runtime.getHeapUsage');
      heap.push(usedSize);
      const el = Date.now() - t0;
      if (el - jumpAt > 1500) { jumpAt = el; await key(cdp, 'Space', ' ', 32, 'keyDown'); await Bun.sleep(80); await key(cdp, 'Space', ' ', 32, 'keyUp'); }
      if (Math.floor(el / 4000) !== strafe) {
        strafe = Math.floor(el / 4000);
        const [c, k, vk] = strafe % 2 ? ['KeyD', 'd', 68] as const : ['KeyA', 'a', 65] as const;
        await key(cdp, c, k, vk, 'keyDown'); await Bun.sleep(400); await key(cdp, c, k, vk, 'keyUp');
      }
      await Bun.sleep(200);
    }
    const end = await evalJson<{ raf: number; draws: number; t: number; nSizes: number; nGpu: number; sizes: number[]; gpu: number[]; hud: string }>(cdp,
      `({ raf: __wb.raf, draws: __wb.draws, t: performance.now(), nSizes: __wb.nSizes, nGpu: __wb.nGpu,
          sizes: Array.from(__wb.sizes.subarray(0, 3 * __wb.nSizes)), gpu: Array.from(__wb.gpu.subarray(0, __wb.nGpu)),
          hud: (document.querySelector('.text-emerald-400.font-bold')?.textContent || '') })`);
    const { profile } = await cdp.send<{ profile: { head: SampleNode } }>('HeapProfiler.stopSampling');
    await key(cdp, 'KeyW', 'w', 87, 'keyUp');

    const secs = (end.t - start.t) / 1000;
    let monotonic = true;
    for (let i = 1; i < heap.length; i++) if (heap[i] < heap[i - 1]) monotonic = false;
    const byScript = new Map<string, number>();
    const byFn = new Map<string, number>();
    let total = 0;
    const visit = (n: SampleNode) => {
      if (n.selfSize > 0) {
        const u = n.callFrame.url ? n.callFrame.url.split('/').pop()! : '(native/harness)';
        byScript.set(u, (byScript.get(u) ?? 0) + n.selfSize);
        const f = `${u}:${n.callFrame.functionName || '(anonymous)'}:${n.callFrame.lineNumber + 1}:${n.callFrame.columnNumber + 1}`;
        byFn.set(f, (byFn.get(f) ?? 0) + n.selfSize);
        total += n.selfSize;
      }
      for (const c of n.children) visit(c);
    };
    visit(profile.head);

    const timeline: Array<[number, number]> = [];
    for (let i = 0; i < end.sizes.length; i += 3) if (end.sizes[i] >= start.t) timeline.push([Math.round(end.sizes[i] - start.t), end.sizes[i + 1]]);
    const sizes: Array<[number, number, number]> = [];
    for (let i = 0; i < end.sizes.length; i += 3) if (end.sizes[i] >= start.t) sizes.push([end.sizes[i], end.sizes[i + 1], end.sizes[i + 2]]);
    let changes = 0, reversals = 0, lastDir = 0;
    for (let i = 1; i < sizes.length; i++) {
      const d = Math.sign(sizes[i][1] - sizes[i - 1][1]);
      if (d !== 0) { changes++; if (lastDir !== 0 && d !== lastDir) reversals++; lastDir = d; }
    }
    const gpu = end.gpu.slice(start.nGpu).filter((x) => x > 0).sort((a, b) => a - b);

    const result: PlayResult = {
      options: opts,
      seconds: secs,
      rafPerSec: (end.raf - start.raf) / secs,
      drawsPerSec: (end.draws - start.draws) / secs,
      sampledBytesPerSec: total / secs,
      heapBytesPerSec: monotonic ? (heap[heap.length - 1] - heap[0]) / secs : null,
      allocByScript: [...byScript.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([script, b]) => ({ script, bytesPerSec: b / secs, share: b / total })),
      allocByFunction: [...byFn.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([fn, b]) => ({ fn, bytesPerSec: b / secs, share: b / total })),
      canvas: {
        first: sizes.length ? [sizes[0][1], sizes[0][2]] : [0, 0],
        last: sizes.length ? [sizes[sizes.length - 1][1], sizes[sizes.length - 1][2]] : [0, 0],
        minW: Math.min(...sizes.map((s) => s[1])),
        maxW: Math.max(...sizes.map((s) => s[1])),
        changes,
        reversals,
        // [ms since window start, canvas width] at each width change
        timeline: timeline.filter((x, i) => i === 0 || x[1] !== timeline[i - 1][1]),
      },
      hudFps: end.hud,
      load,
    };
    if (gpu.length) {
      result.gpuMs = { n: gpu.length, p50: gpu[Math.floor(gpu.length / 2)], p95: gpu[Math.floor(gpu.length * 0.95)], mean: gpu.reduce((a, b) => a + b, 0) / gpu.length };
    }
    if (opts.spector) result.spector = await spectorCapture(cdp);
    cdp.close();
    return result;
  } finally {
    await chrome.close();
  }
}

type SampleNode = { selfSize: number; callFrame: { url: string; functionName: string; lineNumber: number; columnNumber: number }; children: SampleNode[] };

async function spectorCapture(cdp: CDPSession): Promise<Record<string, unknown>> {
  const bundle = readFileSync(path.join(ROOT, 'node_modules/spectorjs/dist/spector.bundle.js'), 'utf8');
  await cdp.send('Runtime.evaluate', { expression: bundle });
  return evalJson<Record<string, unknown>>(cdp, `new Promise((resolve) => {
    const spector = new SPECTOR.Spector();
    const canvas = document.querySelector('canvas');
    spector.onCapture.add((cap) => {
      const cmds = cap.commands || [];
      const names = {};
      for (const c of cmds) names[c.name] = (names[c.name] || 0) + 1;
      const draw = cmds.filter((c) => /^draw/.test(c.name));
      const ctx = cap.context || {};
      resolve({
        spectorVersion: SPECTOR.Spector.version || 'n/a',
        commandCount: cmds.length,
        drawCalls: draw.length,
        drawCommands: draw.map((c) => c.name + '(' + (c.commandArguments ? Object.values(c.commandArguments).join(',') : '') + ')'),
        commandHistogram: names,
        texturesCreated: names.createTexture || 0,
        buffersCreated: names.createBuffer || 0,
        uniformCalls: Object.entries(names).filter(([k]) => k.startsWith('uniform')).reduce((a, [, v]) => a + v, 0),
        contextVersion: ctx.version,
        extensionsEnabled: ctx.extensions ? Object.keys(ctx.extensions).length : undefined,
        frameDurationMs: cap.endTime - cap.startTime,
      });
    });
    spector.captureCanvas(canvas);
    setTimeout(() => resolve({ error: 'spector capture timed out' }), 15000);
  })`);
}
