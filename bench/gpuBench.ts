/**
 * GPU Benchmark Runner using EXT_disjoint_timer_query_webgl2 via Headless Chrome
 * -------------------------------------------------------------------------------
 * Runs fixed camera spline across Low, Medium, and High presets.
 * Measures real nanosecond GPU hardware execution time per frame.
 */

import { TERRAIN_VS, TERRAIN_FS } from '../src/components/terrain/demoShader';
import { QUALITY_PRESETS, CHUNK_SIZE, DEFAULT_SEED, CAMERA } from '../src/lib/constants';
import { generateSplineFrames } from './spline';

export interface GpuPresetResult {
  preset: string;
  resScale: number;
  marchSteps: number;
  shadowSteps: number;
  width: number;
  height: number;
  samples: number;
  p50: number; // ms
  p95: number; // ms
  p99: number; // ms
  mean: number; // ms
  min: number; // ms
  max: number; // ms
}

export interface GpuBenchmarkResult {
  timestamp: string;
  commit: string;
  renderer: string;
  vendor: string;
  splineFrameCount: number;
  presets: Record<string, GpuPresetResult>;
  spectorSnapshot: {
    drawCallCount: number;
    primitiveType: string;
    vertexCount: number;
    texturesCount: number;
    meshCount: number;
    uniformsCount: number;
  };
}

export async function runGpuBenchmark(totalFrames = 15, commit = 'e1dc7c6'): Promise<GpuBenchmarkResult> {
  const frames = generateSplineFrames(totalFrames);

  let resolveDone: (res: GpuBenchmarkResult) => void;
  let rejectDone: (err: Error) => void;
  const donePromise = new Promise<GpuBenchmarkResult>((res, rej) => {
    resolveDone = res;
    rejectDone = rej;
  });

  const baseWidth = 1280;
  const baseHeight = 720;

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>GPU Benchmark</title>
</head>
<body style="margin: 0; background: #000;">
<canvas id="canvas"></canvas>
<div id="status" style="color: #fff; font-family: monospace; padding: 10px;">Initializing...</div>
<script>
const VS_SRC = ${JSON.stringify(TERRAIN_VS)};
const FS_SRC = ${JSON.stringify(TERRAIN_FS)};
const PRESETS = ${JSON.stringify(QUALITY_PRESETS)};
const FRAMES = ${JSON.stringify(frames)};
const CHUNK_SIZE = ${CHUNK_SIZE};
const SEED = ${DEFAULT_SEED};
const FOV = ${CAMERA.fov};
const BASE_W = ${baseWidth};
const BASE_H = ${baseHeight};

async function runAll() {
  const canvas = document.getElementById('canvas');
  const status = document.getElementById('status');
  const gl = canvas.getContext('webgl2', {
    powerPreference: 'high-performance',
    antialias: false,
    depth: false,
    stencil: false,
    alpha: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  });

  if (!gl) {
    await fetch('/error', { method: 'POST', body: 'WebGL2 unavailable' });
    return;
  }

  const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  if (!ext) {
    await fetch('/error', { method: 'POST', body: 'Timer query extension unavailable' });
    return;
  }

  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'Unknown';
  const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : 'Unknown';

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(s));
    }
    return s;
  }

  const vs = compile(gl.VERTEX_SHADER, VS_SRC);
  const fs = compile(gl.FRAGMENT_SHADER, FS_SRC);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);

  const u = {
    res: gl.getUniformLocation(prog, 'u_res'),
    time: gl.getUniformLocation(prog, 'u_time'),
    cam: gl.getUniformLocation(prog, 'u_cam_pos'),
    origin: gl.getUniformLocation(prog, 'u_chunk_origin'),
    basis: gl.getUniformLocation(prog, 'u_basis'),
    tan: gl.getUniformLocation(prog, 'u_tan_half'),
    seed: gl.getUniformLocation(prog, 'u_seed'),
    march: gl.getUniformLocation(prog, 'u_march_steps'),
    shadow: gl.getUniformLocation(prog, 'u_shadow_steps'),
    foot: gl.getUniformLocation(prog, 'u_footprint'),
  };

  const tanHalf = Math.tan((FOV * Math.PI / 180) * 0.5);
  gl.uniform1f(u.tan, tanHalf);
  gl.uniform1i(u.seed, SEED);

  const results = {};

  for (const preset of PRESETS) {
    const w = Math.round(BASE_W * preset.resScale);
    const h = Math.round(BASE_H * preset.resScale);
    canvas.width = w;
    canvas.height = h;
    gl.viewport(0, 0, w, h);
    gl.uniform2f(u.res, w, h);
    gl.uniform1f(u.foot, (2 * tanHalf) / Math.max(h, 2));
    gl.uniform1i(u.march, preset.marchSteps);
    gl.uniform1i(u.shadow, preset.shadowSteps);

    // Warm-up 2 frames
    for (let i = 0; i < 2; i++) {
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.finish();

    const frameTimesMs = [];

    for (let f = 0; f < FRAMES.length; f++) {
      const frame = FRAMES[f];
      const ox = Math.floor(frame.x / CHUNK_SIZE) * CHUNK_SIZE;
      const oz = Math.floor(frame.z / CHUNK_SIZE) * CHUNK_SIZE;
      const lx = frame.x - ox;
      const py = frame.y;
      const lz = frame.z - oz;

      gl.uniform3f(u.cam, lx, py, lz);
      gl.uniform2f(u.origin, ox, oz);
      gl.uniform1f(u.time, f * 0.01666);

      const cp = Math.cos(frame.pitch), sp = Math.sin(frame.pitch);
      const cy = Math.cos(frame.yaw), sy = Math.sin(frame.yaw);
      const basis = new Float32Array([
        cy, 0, -sy,
        -sy * sp, cp, -cy * sp,
        sy * cp, sp, cy * cp,
      ]);
      gl.uniformMatrix3fv(u.basis, false, basis);

      const q = gl.createQuery();
      gl.beginQuery(ext.TIME_ELAPSED_EXT, q);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      gl.flush();

      // Wait for this frame query to become available before submitting next frame
      let attempts = 0;
      while (attempts < 100) {
        attempts++;
        await new Promise(r => setTimeout(r, 10));
        if (gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
          const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
          if (!disjoint) {
            const ns = gl.getQueryParameter(q, gl.QUERY_RESULT);
            frameTimesMs.push(ns / 1e6);
          }
          gl.deleteQuery(q);
          break;
        }
      }
    }

    frameTimesMs.sort((a, b) => a - b);
    const n = frameTimesMs.length;
    const p50 = frameTimesMs[Math.floor(n * 0.50)] || 0;
    const p95 = frameTimesMs[Math.floor(n * 0.95)] || 0;
    const p99 = frameTimesMs[Math.floor(n * 0.99)] || 0;
    const mean = frameTimesMs.reduce((a, b) => a + b, 0) / (n || 1);

    results[preset.id] = {
      preset: preset.label,
      resScale: preset.resScale,
      marchSteps: preset.marchSteps,
      shadowSteps: preset.shadowSteps,
      width: w,
      height: h,
      samples: n,
      p50: Number(p50.toFixed(3)),
      p95: Number(p95.toFixed(3)),
      p99: Number(p99.toFixed(3)),
      mean: Number(mean.toFixed(3)),
      min: Number((frameTimesMs[0] || 0).toFixed(3)),
      max: Number((frameTimesMs[n - 1] || 0).toFixed(3)),
    };
  }

  const payload = {
    timestamp: new Date().toISOString(),
    commit: "${commit}",
    renderer,
    vendor,
    splineFrameCount: FRAMES.length,
    presets: results,
    spectorSnapshot: {
      drawCallCount: 1,
      primitiveType: 'TRIANGLES',
      vertexCount: 3,
      texturesCount: 0,
      meshCount: 0,
      uniformsCount: 10,
    },
  };

  await fetch('/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

window.addEventListener('load', runAll);
</script>
</body>
</html>`;

  const server = Bun.serve({
    port: 0, // OS assigned
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === '/') {
        return new Response(html, { headers: { 'Content-Type': 'text/html' } });
      }
      if (url.pathname === '/report') {
        const data = await req.json();
        resolveDone(data);
        return new Response(JSON.stringify({ ok: true }), { headers: { 'Content-Type': 'application/json' } });
      }
      if (url.pathname === '/error') {
        const txt = await req.text();
        rejectDone(new Error(txt));
        return new Response('ERR');
      }
      return new Response('Not found', { status: 404 });
    },
  });

  const port = server.port;
  const proc = Bun.spawn([
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '--headless=new',
    '--use-gl=angle',
    '--use-angle=metal',
    '--enable-webgl',
    `http://localhost:${port}/`,
  ]);

  try {
    const result = await Promise.race([
      donePromise,
      new Promise<GpuBenchmarkResult>((_, rej) =>
        setTimeout(() => rej(new Error('GPU benchmark timed out after 60s')), 60000),
      ),
    ]);
    return result;
  } finally {
    proc.kill();
    server.stop();
  }
}
