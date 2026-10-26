/**
 * The ground textures' preview page — bundled by bench/ground.ts, served by bench/gpu/runner.ts to
 * headless Chrome. Paints the ground atlas (src/lib/gpu/groundTex.ts) and posts back the mips asked for,
 * every layer, as blobs (kind=ground, mip, layer count).
 */

import { groundTexturesFor } from '../../src/lib/gpu/groundTex';
import { GROUND_LAYERS } from '../../src/lib/gpu/groundTex.wgsl';

const post = (path: string, body: BodyInit, headers: Record<string, string> = {}) => fetch(path, { method: 'POST', body, headers });
const postJSON = (path: string, obj: unknown) => post(path, JSON.stringify(obj), { 'content-type': 'application/json' });
const fail = (msg: unknown) => postJSON('/error', { message: String(msg instanceof Error ? msg.stack : msg) });
window.addEventListener('error', (e) => { void fail(e.error || e.message); });
window.addEventListener('unhandledrejection', (e) => { void fail(e.reason); });

async function main(): Promise<void> {
  const cfg = (await (await fetch('/config')).json()) as { mips: number[] };
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice({ requiredFeatures: ['float32-filterable'].filter((f) => adapter.features.has(f as GPUFeatureName)) as GPUFeatureName[] });
  device.addEventListener('uncapturederror', (e) => { void fail(`WebGPU error: ${(e as GPUUncapturedErrorEvent).error.message}`); });
  await postJSON('/env', { limits: Object.fromEntries(['maxStorageTexturesPerShaderStage', 'maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage', 'maxTextureArrayLayers', 'maxBindGroups', 'maxBindingsPerBindGroup', 'maxColorAttachmentBytesPerSample', 'maxComputeWorkgroupStorageSize', 'maxUniformBufferBindingSize', 'maxStorageBufferBindingSize', 'maxBufferSize'].map((k) => [k, (adapter.limits as unknown as Record<string, number>)[k]])) });
  const g = groundTexturesFor(device);
  const t0 = performance.now();
  await g.init();
  const paintMs = performance.now() - t0;
  for (const mip of cfg.mips) {
    const data = await g.read(mip);
    await post(`/blob?kind=ground&mip=${mip}&layers=${GROUND_LAYERS.length}`, new Uint8Array(data));
  }
  await postJSON('/done', { paintMs, compileMs: g.paintMs.compile, runMs: g.paintMs.run, layerMs: g.layerMs, layers: GROUND_LAYERS });
}

main().catch(fail);
