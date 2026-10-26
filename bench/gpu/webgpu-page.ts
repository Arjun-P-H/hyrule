/**
 * WebGPU benchmark page — bundled by bench/webgpu.ts and served by
 * bench/gpu/runner.ts to headless Chrome (Dawn on Metal). Receives its
 * workload from /config and reports back over fetch(), like page.js does for
 * WebGL2.
 *
 * Modes
 *   timing  per-pass GPU time (timestamp queries): clipmap generation and the
 *           ray-march render, variants interleaved per frame and rep
 *   shots   RGBA8 frames (rows top-down) per variant and frame
 *   probe   elevation readback on a grid (parity with the CPU twin / GLSL)
 *   gen     clipmap generation cost: full rebuilds, and incremental updates
 *           while the camera moves at a set speed
 */

import { hydroBufferFor } from '../../src/lib/gpu/hydroBuffer';
import { createTerrainRenderer, type FrameParams, type PassTimestamps, type TerrainRendererLike, type TerrainVariant } from '../../src/lib/gpu/terrainRenderer';
import { PROBE_BYTES, probeWGSL } from '../../src/lib/gpu/terrain.wgsl';
import { setPassStamp } from '../../src/lib/gpu/profile';

type Frame = { x: number; y: number; z: number; yaw: number; pitch: number; hero?: number[]; heroF32?: Float32Array };
type Preset = { id: string; w: number; h: number; marchSteps: number; shadowSteps: number };
type VariantCfg = { name: string; variant: TerrainVariant };
type Probe = { id: string; origin: [number, number]; local0: [number, number]; step: number; w: number; h: number; dp: number };
type GenCase = { name: string; speed: number; frames: number; heading: number };

type Config = {
  modes: Array<'timing' | 'shots' | 'probe' | 'gen' | 'seq' | 'horizon' | 'taa'>;
  variants: VariantCfg[];
  presets: Preset[];
  frames: Frame[];
  seed: number;
  fov: number;
  reps: number;
  warmupMs: number;
  sentinelEvery: number;
  batch: number;
  /** Render passes per timing task (steady state = gaps between their ends). */
  repeat: number;
  shotFrames: number[];
  /** Frames rendered at the same camera and time before each shot (TAA history converges). */
  converge?: number;
  probes: Probe[];
  /** The world the probes read (default: the land). */
  probeWorld?: 'land' | 'clouds';
  gen: { starts: number[]; rebuilds: number; cases: GenCase[] };
  /** Camera sequence rendered back to back at 60 Hz time steps (temporal stability). */
  seq?: Frame[];
  /** Hold time (wind, water, clouds) still during the sequence: only the camera moves. */
  seqFreeze?: boolean;
  /** With each shot, post the forest's placement window (kind=forest blobs). */
  forestDump?: boolean;
  rockDump?: boolean;
  /** Day–night variants: shots at each of these hours (named variant@hour), timing at `hour`, on game day `day`. */
  hours?: number[];
  hour?: number;
  day?: number;
  /** horizon mode: the clipmap levels whose sun layer is read back at each camera. */
  horizonLevels?: number[];
  /** shots: the clock (s: wind, water, clouds, twinkle) for every shot, instead of frame / 60. */
  time?: number;
};

const post = (path: string, body: BodyInit, headers: Record<string, string> = {}) => fetch(path, { method: 'POST', body, headers });
const postJSON = (path: string, obj: unknown) => post(path, JSON.stringify(obj), { 'content-type': 'application/json' });
const fail = (msg: unknown) => postJSON('/error', { message: String(msg instanceof Error ? msg.stack : msg) });
window.addEventListener('error', (e) => { void fail(e.error || e.message); });
window.addEventListener('unhandledrejection', (e) => { void fail(e.reason); });

const FORMAT: GPUTextureFormat = 'rgba8unorm';

async function main(): Promise<void> {
  const cfg = (await (await fetch('/config')).json()) as Config;
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const wanted: GPUFeatureName[] = ['timestamp-query', 'float32-filterable', 'clip-distances', 'shader-f16'];
  const features = wanted.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures: features });
  device.addEventListener('uncapturederror', (e) => { void fail(`WebGPU error: ${(e as GPUUncapturedErrorEvent).error.message}`); });
  void device.lost.then((info) => fail(`device lost: ${info.reason} ${info.message}`));
  const info = adapter.info;
  await postJSON('/env', {
    api: 'webgpu',
    vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
    features, adapterFeatures: [...adapter.features].sort(), userAgent: navigator.userAgent,
    limits: { storageTextures: adapter.limits.maxStorageTexturesPerShaderStage, sampledTextures: adapter.limits.maxSampledTexturesPerShaderStage },
  });

  const renderers = await Promise.all(cfg.variants.map(async (v) => {
    const r = createTerrainRenderer(device, FORMAT, v.variant);
    await r.init();
    return { name: v.name, r };
  }));
  await postJSON('/compile', renderers.map((x) => ({ name: x.name, compileMs: x.r.compileMs })));

  const tanHalf = Math.tan((cfg.fov * Math.PI / 180) * 0.5);
  const dayOf = (hour: number) => (cfg.day ?? 0) + hour / 24;
  const frameParams = (fr: Frame, p: Preset, time: number, days = dayOf(cfg.hour ?? 15.5)): FrameParams => ({
    x: fr.x, y: fr.y, z: fr.z, yaw: fr.yaw, pitch: fr.pitch, width: p.w, height: p.h, tanHalf,
    seed: cfg.seed, marchSteps: p.marchSteps, shadowSteps: p.shadowSteps, time,
    hero: fr.hero ? (fr.heroF32 ??= new Float32Array(fr.hero)) : null,
    days,
  });
  const shotHours = (r: TerrainRendererLike): Array<number | null> =>
    r.variant.height === 'raster' && r.variant.sky === 'cycle' && cfg.hours?.length ? cfg.hours : [null];

  function makeTarget(w: number, h: number): GPUTexture {
    return device.createTexture({ size: [w, h], format: FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  }

  // ---- timestamp batches -------------------------------------------------------
  // Dawn/Metal samples a render pass's begin timestamp at the start of its
  // vertex stage, which on a tile-based GPU overlaps the previous pass's
  // fragment work: begin→end over-reports whenever passes are queued back to
  // back. Consecutive *end* timestamps of passes that write the same target
  // are serialized, so costs are taken as end-to-end differences inside a
  // batch that keeps the GPU busy (Apple GPUs drop clocks within milliseconds
  // of idling). Raw begin→end durations are kept for reference.
  const MAX_TS = 4096;
  const querySet = device.createQuerySet({ type: 'timestamp', count: MAX_TS });
  const resolveBuf = device.createBuffer({ size: MAX_TS * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: MAX_TS * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  type Alloc = (label: string) => PassTimestamps;
  type Task = { setup: () => void; encode: (enc: GPUCommandEncoder, ts: Alloc) => number };
  type Pass = { label: string; begin: number; end: number };
  type TaskResult = { passes: Pass[]; texels: number };

  async function runBatch(tasks: Task[]): Promise<TaskResult[]> {
    let next = 0;
    const slots: Array<Array<{ label: string; i: number }>> = [];
    const texels: number[] = [];
    for (const t of tasks) {
      t.setup();
      const enc = device.createCommandEncoder();
      const mine: Array<{ label: string; i: number }> = [];
      const alloc: Alloc = (label) => {
        if (next + 2 > MAX_TS) throw new Error('timestamp slots exhausted');
        const i = next;
        next += 2;
        mine.push({ label, i });
        return { querySet, begin: i, end: i + 1 };
      };
      // Every pass the renderer begins (not only those handed a slot) is stamped under its label.
      setPassStamp((label) => {
        const a = alloc(label);
        return { querySet: a.querySet, beginningOfPassWriteIndex: a.begin, endOfPassWriteIndex: a.end };
      });
      try {
        texels.push(t.encode(enc, alloc));
      } finally {
        setPassStamp(null);
      }
      slots.push(mine);
      device.queue.submit([enc.finish()]);
    }
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(querySet, 0, next, resolveBuf, 0);
    enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, next * 8);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ, 0, next * 8);
    const ts = new BigUint64Array(readBuf.getMappedRange(0, next * 8).slice(0));
    readBuf.unmap();
    const t0 = ts[0];
    const ms = (i: number) => Number(ts[i] - t0) / 1e6;
    return slots.map((mine, k) => ({ passes: mine.map(({ label, i }) => ({ label, begin: ms(i), end: ms(i + 1) })), texels: texels[k] }));
  }

  async function runAll(tasks: Task[]): Promise<TaskResult[]> {
    const out: TaskResult[] = [];
    for (let i = 0; i < tasks.length; i += cfg.batch) {
      out.push(...(await runBatch(tasks.slice(i, i + cfg.batch))));
      await postJSON('/heartbeat', { done: Math.min(i + cfg.batch, tasks.length), total: tasks.length });
    }
    return out;
  }

  /** One frame: generate what the camera exposed, then `renders` identical render passes. */
  const frameTask = (R: TerrainRendererLike, view: GPUTextureView, fp: FrameParams, renders = 1, before?: () => void): Task => ({
    setup: () => { before?.(); R.setFrame(fp); },
    encode: (enc, ts) => {
      const texels = R.pendingTexels() > 0 ? R.encodeGenerate(enc, ts('gen')) : 0;
      // A raster renderer stamps its own passes (the main one as 'render') in the order it begins them.
      const own = R.variant.height === 'raster';
      for (let j = 0; j < renders; j++) R.encodeRender(enc, view, own ? undefined : ts('render'));
      return texels;
    },
  });

  const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

  const modes = {
    // Steady-state render cost: per task, `repeat` identical render passes of
    // one variant and frame after that frame's clipmap generation; the cost is
    // the median gap between consecutive render ends.
    async timing() {
      const results = [];
      const K = cfg.repeat;
      for (const preset of cfg.presets) {
        const target = makeTarget(preset.w, preset.h);
        const view = target.createView();
        const t0 = performance.now();
        let k = 0;
        while (performance.now() - t0 < cfg.warmupMs) {
          await runBatch(renderers.map((x, j) => frameTask(x.r, view, frameParams(cfg.frames[(k + j) % cfg.frames.length], preset, 0), K)));
          k++;
        }
        type Meta = { v: number; f: number; r: number; s: number };
        const meta: Meta[] = [];
        const tasks: Task[] = [];
        const sentinel = () => {
          meta.push({ v: 0, f: 0, r: -1, s: 1 });
          tasks.push(frameTask(renderers[0].r, view, frameParams(cfg.frames[0], preset, 0), K));
        };
        for (let f = 0; f < cfg.frames.length; f++) {
          if (f % cfg.sentinelEvery === 0) sentinel();
          for (let r = 0; r < cfg.reps; r++) {
            for (let j = 0; j < renderers.length; j++) {
              const v = (j + r) % renderers.length;
              meta.push({ v, f, r, s: 0 });
              tasks.push(frameTask(renderers[v].r, view, frameParams(cfg.frames[f], preset, f / 60), K));
            }
          }
        }
        sentinel();
        const res = await runAll(tasks);
        results.push({
          preset: preset.id, width: preset.w, height: preset.h,
          samples: meta.map((m, i) => {
            const P = res[i].passes;
            const renders = P.filter((p) => p.label === 'render');
            const gaps = renders.slice(1).map((p, j) => p.end - renders[j].end);
            const g = P.find((p) => p.label === 'gen');
            // Per pass: the gap from the previous pass's end, over each steady-state frame
            // (from one main pass's end to the next: last frame's TAA, this frame's passes).
            const ends = P.map((p) => p.end);
            const at = P.map((p, i) => (p.label === 'render' ? i : -1)).filter((i) => i >= 0);
            const parts: Record<string, number[]> = {};
            for (let j = 1; j < at.length; j++) {
              const sum: Record<string, number> = {};
              for (let i = at[j - 1] + 1; i <= at[j]; i++) sum[P[i].label] = (sum[P[i].label] ?? 0) + ends[i] - ends[i - 1];
              for (const [k, v] of Object.entries(sum)) (parts[k] ??= []).push(v);
            }
            const passes = Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, median(v)]));
            // and begin → end of each (compute passes overlap render passes on this GPU, so gaps can go negative)
            const raws: Record<string, number[]> = {};
            for (let j = 1; j < at.length; j++) {
              for (let i = at[j - 1] + 1; i <= at[j]; i++) (raws[P[i].label] ??= []).push(P[i].end - P[i].begin);
            }
            const raw = Object.fromEntries(Object.entries(raws).map(([k, v]) => [k, median(v)]));
            return {
              ...m,
              passes,
              raw,
              ms: median(gaps),
              first: renders[0].end - (g ? g.end : renders[0].begin),
              gen: g ? g.end - g.begin : null,
              rawRender: median(renders.map((p) => p.end - p.begin)),
              texels: res[i].texels,
            };
          }),
        });
        target.destroy();
        await postJSON('/progress', { mode: 'timing', preset: preset.id });
      }
      return results;
    },

    async shots() {
      const out = [];
      for (const preset of cfg.presets) {
        const target = makeTarget(preset.w, preset.h);
        const view = target.createView();
        const bpr = Math.ceil((preset.w * 4) / 256) * 256;
        const buf = device.createBuffer({ size: bpr * preset.h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const packed = new Uint8Array(preset.w * preset.h * 4);
        for (const f of cfg.shotFrames) {
          for (const x of renderers) for (const hour of shotHours(x.r)) {
            const days = hour === null ? undefined : dayOf(hour);
            const name = hour === null ? x.name : `${x.name}@${hour}`;
            const t = cfg.time ?? f / 60;
            for (let c = 1; c < (cfg.converge ?? 1); c++) {
              x.r.setFrame(frameParams(cfg.frames[f], preset, t, days));
              const warm = device.createCommandEncoder();
              x.r.encodeGenerate(warm);
              x.r.encodeRender(warm, view);
              device.queue.submit([warm.finish()]);
            }
            x.r.setFrame(frameParams(cfg.frames[f], preset, t, days));
            const enc = device.createCommandEncoder();
            x.r.encodeGenerate(enc);
            x.r.encodeRender(enc, view);
            enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr }, [preset.w, preset.h]);
            device.queue.submit([enc.finish()]);
            await buf.mapAsync(GPUMapMode.READ);
            const src = new Uint8Array(buf.getMappedRange());
            for (let y = 0; y < preset.h; y++) packed.set(src.subarray(y * bpr, y * bpr + preset.w * 4), y * preset.w * 4);
            buf.unmap();
            await post(`/blob?kind=shot&variant=${encodeURIComponent(name)}&preset=${preset.id}&frame=${f}&w=${preset.w}&h=${preset.h}`, packed);
            const gc = (x.r as { grassCount?: () => Promise<number> }).grassCount;
            const blades = gc ? await gc.call(x.r) : undefined;
            const tc = (x.r as { treeCount?: () => Promise<[number, number]> }).treeCount;
            const trees = tc ? await tc.call(x.r) : undefined;
            const sp = (x.r as { skyProbe?: () => Promise<unknown> }).skyProbe;
            const sky = sp ? await sp.call(x.r) : undefined;
            out.push({ variant: name, preset: preset.id, frame: f, w: preset.w, h: preset.h, blades, trees, sky });
            const fs = (x.r as { forestSlots?: () => Promise<{ side: number; data: ArrayBuffer } | null> }).forestSlots;
            const dump = cfg.forestDump && fs ? await fs.call(x.r) : null;
            if (dump) await post(`/blob?kind=forest&variant=${encodeURIComponent(x.name)}&frame=${f}&side=${dump.side}`, new Uint8Array(dump.data));
            const rk = (x.r as { rockDump?: () => Promise<{ args: Uint32Array; inst: Float32Array } | null> }).rockDump;
            const rdump = cfg.rockDump && rk ? await rk.call(x.r) : null;
            if (rdump) {
              const bytes = new Uint8Array(rdump.args.byteLength + rdump.inst.byteLength);
              bytes.set(new Uint8Array(rdump.args.buffer, rdump.args.byteOffset, rdump.args.byteLength), 0);
              bytes.set(new Uint8Array(rdump.inst.buffer, rdump.inst.byteOffset, rdump.inst.byteLength), rdump.args.byteLength);
              await post(`/blob?kind=rocks&variant=${encodeURIComponent(x.name)}&frame=${f}&argBytes=${rdump.args.byteLength}`, bytes);
            }
          }
        }
        buf.destroy();
        target.destroy();
      }
      return out;
    },

    async probe() {
      const probeModule = device.createShaderModule({ code: probeWGSL(cfg.probeWorld ?? 'land') });
      const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: probeModule, entryPoint: 'main' } });
      const ubuf = device.createBuffer({ size: PROBE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const out = [];
      for (const pr of cfg.probes) {
        const bytes = pr.w * pr.h * 16;
        const sbuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const rbuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const data = new ArrayBuffer(PROBE_BYTES);
        const f = new Float32Array(data), u = new Uint32Array(data), i = new Int32Array(data);
        f[0] = pr.origin[0]; f[1] = pr.origin[1]; f[2] = pr.local0[0]; f[3] = pr.local0[1];
        f[4] = pr.step; f[5] = pr.dp; u[6] = pr.w; i[7] = cfg.seed;
        device.queue.writeBuffer(ubuf, 0, data);
        const hydro = hydroBufferFor(device);
        hydro.update(pr.origin[0] + pr.local0[0], pr.origin[1] + pr.local0[1], cfg.seed);
        const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: sbuf } }, hydro.entry()] });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(pr.w / 8), Math.ceil(pr.h / 8));
        pass.end();
        enc.copyBufferToBuffer(sbuf, 0, rbuf, 0, bytes);
        device.queue.submit([enc.finish()]);
        await rbuf.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(rbuf.getMappedRange().slice(0));
        rbuf.unmap();
        await post(`/blob?kind=probe&variant=wgsl&id=${encodeURIComponent(pr.id)}`, copy);
        out.push({ variant: 'wgsl', id: pr.id });
        sbuf.destroy();
        rbuf.destroy();
      }
      return out;
    },

    // Temporal stability: the sequence rendered back to back; per consecutive
    // pair, the mean absolute change of luma (0..255) over all pixels, and the
    // share of pixels that change by more than 24 (flicker). The camera moves
    // identically for every variant, so differences between variants are the
    // anti-aliasing's shimmer (or smear).
    async seq() {
      const out = [];
      const frames = cfg.seq ?? [];
      for (const preset of cfg.presets) {
        const target = makeTarget(preset.w, preset.h);
        const view = target.createView();
        const bpr = Math.ceil((preset.w * 4) / 256) * 256;
        const buf = device.createBuffer({ size: bpr * preset.h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const n = preset.w * preset.h;
        for (const x of renderers) {
          x.r.invalidate();
          let prev: Float32Array | null = null;
          const meanDelta: number[] = [];
          const flicker: number[] = [];
          for (let k = 0; k < frames.length; k++) {
            x.r.setFrame(frameParams(frames[k], preset, cfg.seqFreeze ? 0 : k / 60));
            const enc = device.createCommandEncoder();
            x.r.encodeGenerate(enc);
            x.r.encodeRender(enc, view);
            enc.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: bpr }, [preset.w, preset.h]);
            device.queue.submit([enc.finish()]);
            await buf.mapAsync(GPUMapMode.READ);
            const src = new Uint8Array(buf.getMappedRange());
            const lum = new Float32Array(n);
            for (let yy = 0; yy < preset.h; yy++) {
              for (let xx = 0; xx < preset.w; xx++) {
                const o = yy * bpr + xx * 4;
                lum[yy * preset.w + xx] = 0.2126 * src[o] + 0.7152 * src[o + 1] + 0.0722 * src[o + 2];
              }
            }
            buf.unmap();
            if (prev) {
              let sum = 0, big = 0;
              for (let i = 0; i < n; i++) {
                const dl = Math.abs(lum[i] - prev[i]);
                sum += dl;
                if (dl > 24) big++;
              }
              meanDelta.push(sum / n);
              flicker.push(big / n);
            }
            prev = lum;
            await postJSON('/heartbeat', { done: k + 1, total: frames.length });
          }
          out.push({ variant: x.name, preset: preset.id, meanDelta, flicker });
        }
        buf.destroy();
        target.destroy();
      }
      return out;
    },

    // The clipmap's sun layer at each camera, after a whole rebuild there
    // (kind=horizon blobs: the layer's texels, channels per texel).
    async horizon() {
      const preset = cfg.presets[0];
      const target = makeTarget(preset.w, preset.h);
      const view = target.createView();
      const out = [];
      for (const x of renderers) {
        const layer = (x.r as { sunLayer?: (l: number) => Promise<{ n: number; spacing: number; channels: number; data: Float32Array }> }).sunLayer;
        if (!layer) continue;
        for (let fi = 0; fi < cfg.frames.length; fi++) {
          x.r.invalidate();
          for (let k = 0; k < 2; k++) {
            x.r.setFrame(frameParams(cfg.frames[fi], preset, 0));
            const enc = device.createCommandEncoder();
            x.r.encodeGenerate(enc);
            x.r.encodeRender(enc, view);
            device.queue.submit([enc.finish()]);
          }
          await device.queue.onSubmittedWorkDone();
          for (const level of cfg.horizonLevels ?? [0]) {
            const r = await layer.call(x.r, level);
            await post(`/blob?kind=horizon&variant=${encodeURIComponent(x.name)}&frame=${fi}&level=${level}&n=${r.n}&spacing=${r.spacing}&channels=${r.channels}`, new Uint8Array(r.data.buffer as ArrayBuffer));
            out.push({ variant: x.name, frame: fi, level });
          }
        }
      }
      target.destroy();
      return out;
    },

    // Clipmap generation cost as the GPU frame time it adds. Every task is a
    // whole frame (generate + render) and a batch runs frames back to back, so
    // frame k costs end(k) − end(k−1). Rebuild: a settled frame, then the same
    // frame after invalidate() (all levels regenerated). Moves: the camera
    // travels at the case's speed for case.frames 60 Hz frames, against the
    // same number of frames with the camera held still.
    // The TAA passes alone over each frame's scene: per task one frame, then `repeat` resolve +
    // present pairs back to back (each resolve overwrites the history the present before it read,
    // so they run in order); costs are gaps between consecutive pass ends.
    async taa() {
      const out = [];
      type Taa = TerrainRendererLike & { encodeTaa?: (enc: GPUCommandEncoder, view: GPUTextureView) => void };
      for (const preset of cfg.presets) {
        const target = makeTarget(preset.w, preset.h);
        const view = target.createView();
        const K = cfg.repeat;
        const task = (R: Taa, f: number): Task => ({
          setup: () => R.setFrame(frameParams(cfg.frames[f], preset, f / 60)),
          encode: (enc) => {
            if (R.pendingTexels() > 0) R.encodeGenerate(enc);
            R.encodeRender(enc, view);
            for (let k = 0; k < K; k++) R.encodeTaa?.(enc, view);
            return 0;
          },
        });
        const t0 = performance.now();
        for (let k = 0; performance.now() - t0 < cfg.warmupMs; k++) {
          await runBatch(renderers.map((x, j) => task(x.r, (k + j) % cfg.frames.length)));
        }
        const meta: Array<{ v: number; f: number; r: number }> = [];
        const tasks: Task[] = [];
        for (let f = 0; f < cfg.frames.length; f++) {
          for (let r = 0; r < cfg.reps; r++) {
            for (let j = 0; j < renderers.length; j++) {
              const v = (j + r) % renderers.length;
              meta.push({ v, f, r });
              tasks.push(task(renderers[v].r, f));
            }
          }
        }
        const res = await runAll(tasks);
        out.push({
          preset: preset.id,
          samples: meta.map((m, i) => {
            const P = res[i].passes;
            const pres = P.map((p, k) => (p.label === 'taa.present' ? k : -1)).filter((k) => k >= 0);
            const both: number[] = [], resolve: number[] = [];
            for (let j = 1; j < pres.length; j++) {
              both.push(P[pres[j]].end - P[pres[j - 1]].end);
              resolve.push(P[pres[j] - 1].end - P[pres[j - 1]].end);
            }
            return { ...m, ms: median(both), resolve: median(resolve), present: median(both) - median(resolve) };
          }),
        });
        target.destroy();
      }
      return out;
    },

    async gen() {
      const preset = cfg.presets[0];
      const target = makeTarget(preset.w, preset.h);
      const view = target.createView();
      const out = [];
      const ends = (res: TaskResult[]) => res.map((t) => t.passes[t.passes.length - 1].end);
      const gaps = (res: TaskResult[]) => { const e = ends(res); return e.slice(1).map((x, j) => x - e[j]); };
      for (const x of renderers) {
        if (x.r.variant.height === 'direct') continue;
        const rebuild: Array<{ start: number; frameMs: number; steadyMs: number; rawGenMs: number; texels: number }> = [];
        for (let rep = 0; rep < cfg.gen.rebuilds; rep++) {
          for (const fi of cfg.gen.starts) {
            const fp = frameParams(cfg.frames[fi], preset, 0);
            const res = await runBatch([
              frameTask(x.r, view, fp), frameTask(x.r, view, fp), frameTask(x.r, view, fp),
              frameTask(x.r, view, fp, 1, () => x.r.invalidate()),
            ]);
            const g = gaps(res);
            const genPass = res[3].passes.find((p) => p.label === 'gen');
            rebuild.push({ start: fi, frameMs: g[2], steadyMs: g[1], rawGenMs: genPass ? genPass.end - genPass.begin : 0, texels: res[3].texels });
          }
        }
        const moves = [];
        for (const c of cfg.gen.cases) {
          for (const fi of cfg.gen.starts) {
            const f0 = cfg.frames[fi];
            const dx = Math.sin(c.heading) * c.speed / 60, dz = Math.cos(c.heading) * c.speed / 60;
            const still: Task[] = [], moving: Task[] = [];
            for (let k = 0; k <= c.frames; k++) {
              still.push(frameTask(x.r, view, frameParams(f0, preset, k / 60)));
              moving.push(frameTask(x.r, view, frameParams({ ...f0, x: f0.x + dx * k, z: f0.z + dz * k }, preset, k / 60)));
            }
            await runBatch([frameTask(x.r, view, frameParams(f0, preset, 0))]); // settle the windows
            const rs = await runBatch(still);
            const rm = await runBatch(moving);
            moves.push({
              case: c.name, speed: c.speed, start: fi,
              still: gaps(rs), moving: gaps(rm),
              rawGen: rm.map((t) => { const g = t.passes.find((p) => p.label === 'gen'); return g ? g.end - g.begin : 0; }),
              texels: rm.map((t) => t.texels),
            });
          }
        }
        out.push({ variant: x.name, rebuild, moves });
      }
      target.destroy();
      return out;
    },
  };

  const results: Record<string, unknown> = {};
  for (const m of cfg.modes) results[m] = await modes[m]();
  await postJSON('/done', results);
}

main().catch(fail);
