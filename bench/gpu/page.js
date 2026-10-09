// W(x,y) GPU benchmark page — runs inside headless Chrome (ANGLE/Metal).
// Served by bench/gpu/runner.ts; receives its workload from /config and
// reports back over fetch(). Plain JS on purpose: it is loaded verbatim.

const post = (path, body, headers = {}) => fetch(path, { method: 'POST', body, headers });
const postJSON = (path, obj) => post(path, JSON.stringify(obj), { 'content-type': 'application/json' });
const fail = (msg) => postJSON('/error', { message: String(msg && msg.stack ? msg.stack : msg) });
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));

// A macrotask yield without setTimeout's 4 ms clamp. WebGL only publishes
// query results after control returns to the event loop.
const yieldTask = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });

async function main() {
  const cfg = await (await fetch('/config')).json();
  const canvas = document.getElementById('c');
  const gl = canvas.getContext('webgl2', {
    powerPreference: 'high-performance', antialias: false, depth: false, stencil: false,
    alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: false, desynchronized: true,
  });
  if (!gl) return fail('WebGL2 unavailable');
  const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const cbf = gl.getExtension('EXT_color_buffer_float');
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const env = {
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
    userAgent: navigator.userAgent,
    timerQuery: !!timer,
    colorBufferFloat: !!cbf,
  };
  await postJSON('/env', env);

  // ---- program cache ---------------------------------------------------------
  function compile(type, src, label) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`${label}: ${gl.getShaderInfoLog(s)}`);
    return s;
  }
  const UNIFORMS = ['u_res', 'u_time', 'u_cam_pos', 'u_chunk_origin', 'u_basis', 'u_tan_half', 'u_seed',
    'u_march_steps', 'u_shadow_steps', 'u_footprint', 'u_probe_local', 'u_probe_step', 'u_probe_dp',
    'u_bench_dp', 'u_bench_base', 'u_hero_bound', 'u_hero_j', 'u_hero_torso', 'u_hero_head', 'u_hero_misc', 'u_hero_dress',
    'u_hydro', 'u_hydro_head'];
  // The rivers' node table (src/lib/hydro.ts hydroTable; runner.ts attaches it): six RGBA32F texels a node.
  let hydroHead = [0, 0, 0, 2048];
  if (cfg.hydro) {
    const bytes = Uint8Array.from(atob(cfg.hydro.data), (ch) => ch.charCodeAt(0));
    const n = cfg.hydro.head[2];
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, n * 6, n);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n * 6, n, gl.RGBA, gl.FLOAT, new Float32Array(bytes.buffer));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.activeTexture(gl.TEXTURE0);
    hydroHead = cfg.hydro.head;
  }
  const bindHydro = (L) => {
    if (L.u_hydro) gl.uniform1i(L.u_hydro, 1);
    if (L.u_hydro_head) gl.uniform4fv(L.u_hydro_head, hydroHead);
  };
  const programs = cfg.variants.map((v) => {
    const t0 = performance.now();
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, v.vs, `${v.name} VS`));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, v.fs, `${v.name} FS`));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`${v.name} link: ${gl.getProgramInfoLog(p)}`);
    const loc = {};
    for (const u of UNIFORMS) loc[u] = gl.getUniformLocation(p, u);
    return { name: v.name, prog: p, loc, budgets: v.budgets || null, hero: !!v.hero, compileMs: performance.now() - t0 };
  });
  await postJSON('/compile', programs.map((p) => ({ name: p.name, compileMs: p.compileMs })));

  const tanHalf = Math.tan((cfg.fov * Math.PI / 180) * 0.5);
  const basis = new Float32Array(9);

  // Mirrors useTerrainEngine exactly: floating origin + camera basis.
  function setCamera(P, frame, w, h, preset, time) {
    const L = P.loc;
    gl.useProgram(P.prog);
    const ox = Math.floor(frame.x / cfg.chunkSize) * cfg.chunkSize;
    const oz = Math.floor(frame.z / cfg.chunkSize) * cfg.chunkSize;
    gl.uniform2f(L.u_res, w, h);
    gl.uniform1f(L.u_time, time);
    gl.uniform3f(L.u_cam_pos, frame.x - ox, frame.y, frame.z - oz);
    gl.uniform2f(L.u_chunk_origin, ox, oz);
    const cy = Math.cos(frame.yaw), sy = Math.sin(frame.yaw);
    const cp = Math.cos(frame.pitch), sp = Math.sin(frame.pitch);
    const fx = sy * cp, fy = sp, fz = cy * cp, rx = cy, rz = -sy;
    basis[0] = rx; basis[1] = 0; basis[2] = rz;
    basis[3] = fy * rz; basis[4] = fz * rx - fx * rz; basis[5] = -fy * rx;
    basis[6] = fx; basis[7] = fy; basis[8] = fz;
    gl.uniformMatrix3fv(L.u_basis, false, basis);
    gl.uniform1f(L.u_tan_half, tanHalf);
    gl.uniform1i(L.u_seed, cfg.seed);
    bindHydro(L);
    const b = (P.budgets && P.budgets[preset.id]) || preset;
    gl.uniform1i(L.u_march_steps, b.marchSteps);
    gl.uniform1i(L.u_shadow_steps, preset.shadowOff ? 0 : b.shadowSteps);
    gl.uniform1f(L.u_footprint, (2 * tanHalf) / Math.max(h, 2));
    // Third-person hero: the frame carries the packed uniforms (rig.ts
    // HERO_UNIFORM layout, camera-relative to this frame's camera); variants
    // without `hero` — and revisions without the uniforms — draw no hero.
    if (P.hero && frame.hero) {
      const hb = frame.heroF32 || (frame.heroF32 = new Float32Array(frame.hero));
      gl.uniform3fv(L.u_hero_j, hb.subarray(0, 54));
      gl.uniformMatrix3fv(L.u_hero_torso, false, hb.subarray(54, 63));
      gl.uniformMatrix3fv(L.u_hero_head, false, hb.subarray(63, 72));
      gl.uniform4fv(L.u_hero_bound, hb.subarray(72, 76));
      gl.uniform4fv(L.u_hero_misc, hb.subarray(76, 80));
      // the tunic (src/lib/hero/characters.ts); frames packed before it existed wear the classic green
      if (L.u_hero_dress) gl.uniform3fv(L.u_hero_dress, hb.length >= 83 ? hb.subarray(80, 83) : [0.028, 0.115, 0.022]);
    } else if (L.u_hero_bound) {
      gl.uniform4f(L.u_hero_bound, 0, 0, 0, 0);
    }
  }

  function resize(w, h) {
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
  }

  // ---- pipelined timer-query sampler ----------------------------------------
  // Keeps up to `depth` measured draws in flight so the GPU never idles between
  // samples (idle gaps let Apple GPUs drop clocks and inflate the next sample).
  async function timeAll(tasks, depth = 2) {
    if (!timer) throw new Error('EXT_disjoint_timer_query_webgl2 unavailable');
    const out = new Float64Array(tasks.length).fill(NaN);
    const inflight = [];
    let next = 0, disjoint = 0;
    while (next < tasks.length || inflight.length) {
      while (next < tasks.length && inflight.length < depth) {
        tasks[next].setup();
        const q = gl.createQuery();
        gl.beginQuery(timer.TIME_ELAPSED_EXT, q);
        tasks[next].draw();
        gl.endQuery(timer.TIME_ELAPSED_EXT);
        gl.flush();
        inflight.push({ q, i: next++ });
      }
      await yieldTask();
      while (inflight.length && gl.getQueryParameter(inflight[0].q, gl.QUERY_RESULT_AVAILABLE)) {
        const { q, i } = inflight.shift();
        if (gl.getParameter(timer.GPU_DISJOINT_EXT)) disjoint++;
        else out[i] = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
        gl.deleteQuery(q);
        if ((i + 1) % 50 === 0) postJSON('/heartbeat', { done: i + 1, total: tasks.length });
      }
      if (gl.isContextLost()) throw new Error('WebGL context lost during timing');
    }
    return { ms: Array.from(out), disjoint };
  }

  const drawTri = () => gl.drawArrays(gl.TRIANGLES, 0, 3);

  // Apple GPUs drop clocks within milliseconds of idling; the first samples
  // after a resize or a pause read up to 2x slow. Warm up with the exact
  // measurement loop (depth 1) until cfg.warmupMs of GPU time has elapsed.
  async function warmUp(w, h, preset) {
    const t0 = performance.now();
    let k = 0;
    while (performance.now() - t0 < cfg.warmupMs) {
      const batch = programs.map((P, j) => ({ setup: () => setCamera(P, cfg.frames[(k + j) % cfg.frames.length], w, h, preset, 0), draw: drawTri }));
      await timeAll(batch, 1);
      k++;
    }
  }

  // ---- modes ------------------------------------------------------------------
  const modes = {
    // GPU frame time along the camera spline; variants interleaved per frame/rep.
    async timing() {
      const results = [];
      for (const preset of cfg.presets) {
        const w = Math.round(cfg.baseW * preset.resScale), h = Math.round(cfg.baseH * preset.resScale);
        resize(w, h);
        await warmUp(w, h, preset);
        const tasks = [];
        const sentinel = () => {
          // Fixed re-measurement of frame 0 / variant 0: exposes thermal drift
          // over the session independently of scene cost.
          const P = programs[0];
          tasks.push({ v: 0, f: 0, r: -1, sentinel: true, setup: () => setCamera(P, cfg.frames[0], w, h, preset, 0), draw: drawTri });
        };
        for (let f = 0; f < cfg.frames.length; f++) {
          if (f % cfg.sentinelEvery === 0) sentinel();
          for (let r = 0; r < cfg.reps; r++) {
            // rotate variant order per rep so neither always runs first
            for (let j = 0; j < programs.length; j++) {
              const v = (j + r) % programs.length;
              const P = programs[v];
              tasks.push({ v, f, r, setup: () => setCamera(P, cfg.frames[f], w, h, preset, f / 60), draw: drawTri });
            }
          }
        }
        sentinel();
        const { ms, disjoint } = await timeAll(tasks, cfg.depth);
        results.push({
          preset: preset.id, width: w, height: h, disjoint,
          samples: tasks.map((t, i) => ({ v: t.v, f: t.f, r: t.r, s: t.sentinel ? 1 : 0, ms: ms[i] })),
        });
        await postJSON('/progress', { mode: 'timing', preset: preset.id });
      }
      return results;
    },

    // Timer query vs wall clock on identical work. Each wall-clock draw is
    // followed by a 1-px readPixels so it completes on its own: back-to-back
    // fullscreen draws in one render pass are merged by tile-based hidden
    // surface removal and would under-report. Also reports whether in-flight
    // depth 2 inflates the query (it includes queue wait on ANGLE/Metal).
    async calibrate() {
      const out = [];
      const P = programs[0];
      for (const preset of cfg.presets) {
        const w = Math.round(cfg.baseW * preset.resScale), h = Math.round(cfg.baseH * preset.resScale);
        resize(w, h);
        const px = new Uint8Array(4);
        for (const fi of cfg.calibrateFrames) {
          setCamera(P, cfg.frames[fi], w, h, preset, 0);
          for (let i = 0; i < 3; i++) { drawTri(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); }
          const N = cfg.calibrateDraws;
          const mk = () => Array.from({ length: N }, () => ({ setup: () => {}, draw: drawTri }));
          const d1 = await timeAll(mk(), 1);
          const d2 = await timeAll(mk(), 2);
          const wall = [];
          for (let i = 0; i < N; i++) {
            const t0 = performance.now();
            drawTri();
            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
            wall.push(performance.now() - t0);
            await yieldTask();
          }
          const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
          out.push({ preset: preset.id, frame: fi, timerDepth1: med(d1.ms), timerDepth2: med(d2.ms), wall: med(wall),
            depth1: d1.ms, wallAll: wall });
        }
      }
      return out;
    },

    // Deterministic frames for perceptual diffs: RGBA8 readback per variant.
    async shots() {
      const out = [];
      for (const preset of cfg.presets) {
        const w = Math.round(cfg.baseW * preset.resScale), h = Math.round(cfg.baseH * preset.resScale);
        resize(w, h);
        const buf = new Uint8Array(w * h * 4);
        for (const f of cfg.shotFrames) {
          for (const P of programs) {
            setCamera(P, cfg.frames[f], w, h, preset, f / 60);
            drawTri();
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            await post(`/blob?kind=shot&variant=${encodeURIComponent(P.name)}&preset=${preset.id}&frame=${f}&w=${w}&h=${h}`, buf);
            out.push({ variant: P.name, preset: preset.id, frame: f, w, h });
          }
        }
      }
      return out;
    },

    // Per-pixel instrumentation counters (RGBA32F): elevation calls, march
    // iterations, bound calls — summarised in-page to keep transfers small.
    async instrument() {
      if (!cbf) throw new Error('EXT_color_buffer_float unavailable');
      const out = [];
      for (const preset of cfg.presets) {
        const w = Math.round(cfg.baseW * preset.resScale), h = Math.round(cfg.baseH * preset.resScale);
        const fb = gl.createFramebuffer();
        const rb = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA32F, w, h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
        gl.viewport(0, 0, w, h);
        const buf = new Float32Array(w * h * 4);
        for (const P of programs) {
          for (const shadowOff of [false, true]) {
            const sums = [0, 0, 0, 0];
            const hist = new Map();
            let n = 0;
            for (const f of cfg.instrumentFrames) {
              const pr = shadowOff ? { ...preset, shadowSteps: 0, shadowOff: true } : preset;
              setCamera(P, cfg.frames[f], w, h, pr, f / 60);
              drawTri();
              gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
              for (let i = 0; i < w * h; i++) {
                for (let c = 0; c < 4; c++) sums[c] += buf[i * 4 + c];
                const k = buf[i * 4 + 1] | 0; // march iterations histogram
                hist.set(k, (hist.get(k) || 0) + 1);
              }
              n += w * h;
            }
            out.push({
              variant: P.name, preset: preset.id, shadowOff, pixels: n,
              meanElev: sums[0] / n, meanSteps: sums[1] / n, meanBound: sums[2] / n, meanAux: sums[3] / n,
              stepsHist: [...hist.entries()].sort((a, b) => a[0] - b[0]),
            });
          }
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fb);
        gl.deleteRenderbuffer(rb);
      }
      return out;
    },

    // GPU elevation readback (RGBA32F): E, fp32 global x, fp32 global z.
    async probe() {
      if (!cbf) throw new Error('EXT_color_buffer_float unavailable');
      const out = [];
      for (const pr of cfg.probes) {
        const fb = gl.createFramebuffer();
        const rb = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA32F, pr.w, pr.h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
        gl.viewport(0, 0, pr.w, pr.h);
        const buf = new Float32Array(pr.w * pr.h * 4);
        for (const P of programs) {
          const L = P.loc;
          gl.useProgram(P.prog);
          gl.uniform2f(L.u_chunk_origin, pr.origin[0], pr.origin[1]);
          gl.uniform2f(L.u_probe_local, pr.local0[0], pr.local0[1]);
          gl.uniform2f(L.u_probe_step, pr.step, pr.step);
          gl.uniform1f(L.u_probe_dp, pr.dp);
          gl.uniform1i(L.u_seed, cfg.seed);
          bindHydro(L);
          drawTri();
          gl.readPixels(0, 0, pr.w, pr.h, gl.RGBA, gl.FLOAT, buf);
          await post(`/blob?kind=probe&variant=${encodeURIComponent(P.name)}&id=${encodeURIComponent(pr.id)}`, buf);
          out.push({ variant: P.name, id: pr.id });
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fb);
        gl.deleteRenderbuffer(rb);
      }
      return out;
    },

    // Isolated math-core throughput: K elevation() calls per pixel at fixed dp.
    // One draw per timed sample: several opaque fullscreen draws inside one
    // query are merged by tile-based hidden-surface removal (only the last is
    // shaded), which under-reports cost by the draw count.
    async evalcost() {
      const out = [];
      const w = cfg.evalRes, h = cfg.evalRes;
      resize(w, h);
      for (const dp of cfg.evalDps) {
        const t0 = performance.now();
        let k = 0;
        while (performance.now() - t0 < cfg.warmupMs) {
          const P = programs[k % programs.length];
          await timeAll([{ setup: () => { gl.useProgram(P.prog); gl.uniform1i(P.loc.u_seed, cfg.seed); gl.uniform1f(P.loc.u_bench_dp, dp); gl.uniform2f(P.loc.u_chunk_origin, cfg.evalBases[k % cfg.evalBases.length][0], cfg.evalBases[k % cfg.evalBases.length][1]); }, draw: drawTri }], 1);
          k++;
        }
        const tasks = [];
        for (let r = 0; r < cfg.evalReps; r++) {
          for (let bi = 0; bi < cfg.evalBases.length; bi++) {
            for (let j = 0; j < programs.length; j++) {
              const v = (j + r + bi) % programs.length;
              const P = programs[v];
              const b = cfg.evalBases[bi];
              tasks.push({
                v, b: bi,
                setup: () => { gl.useProgram(P.prog); gl.uniform1i(P.loc.u_seed, cfg.seed); gl.uniform1f(P.loc.u_bench_dp, dp); gl.uniform2f(P.loc.u_chunk_origin, b[0], b[1]); },
                draw: drawTri,
              });
            }
          }
        }
        const { ms } = await timeAll(tasks, 1);
        out.push({ dp, evalsPerSample: w * h * cfg.evalK, samples: tasks.map((t, i) => ({ v: t.v, b: t.b, ms: ms[i] })) });
      }
      return out;
    },
  };

  const results = {};
  for (const m of cfg.modes) results[m] = await modes[m]();
  await postJSON('/done', results);
}

main().catch(fail);
