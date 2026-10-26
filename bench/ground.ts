/**
 * The ground textures, looked at (headless Chrome, WebGPU).
 * ---------------------------------------------------------------------------
 *   bun bench/ground.ts [--mips=0] [--out=bench/out/ground]
 *
 * Paints the atlas (src/lib/gpu/groundTex.ts) and writes, per layer, a PNG of three panels: its tone,
 * its normal map, and the surface lit from the upper left, tiled 2 × 2 (so a seam or an obvious repeat
 * shows). Prints each layer's mean tone and slope (the tone should average 1: the biome's colour
 * stays the albedo) and how long the painting took.
 */

import path from 'node:path';
import { mkdirSync } from 'node:fs';
import { runSession } from './gpu/runner';
import { encodePNG, type RGBAImage } from './lib/image';
import { GROUND_LAYERS, GROUND_SIZE } from '../src/lib/gpu/groundTex.wgsl';

const ROOT = path.resolve(import.meta.dir, '..');
const args = process.argv.slice(2);
const opt = (k: string, d: string) => args.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const mips = opt('mips', '0').split(',').map(Number);
const out = path.resolve(ROOT, opt('out', 'bench/out/ground'));
mkdirSync(out, { recursive: true });

const build = await Bun.build({ entrypoints: [path.join(import.meta.dir, 'gpu/ground-page.ts')], target: 'browser' });
if (!build.success) throw new Error(build.logs.map(String).join('\n'));
const pageJs = await build.outputs[0].text();

const res = await runSession({ mips, stallMs: 120_000, timeoutMs: 5 * 60_000 }, { pageJs, chromeArgs: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'] });
console.log('env', JSON.stringify(res.env));
console.log('painters ready at (ms):', (res.results.layerMs as number[]).map((x, i) => `${GROUND_LAYERS[i]} ${x.toFixed(0)}`).join(', '));
console.log('painted in', (res.results.paintMs as number).toFixed(1), 'ms (pipelines', (res.results.compileMs as number).toFixed(1), 'ms, GPU', (res.results.runMs as number).toFixed(1), 'ms)');

for (const b of res.blobs) {
  if (b.query.kind !== 'ground') continue;
  const mip = Number(b.query.mip);
  const n = Math.max(GROUND_SIZE >> mip, 1);
  const layers = Number(b.query.layers);
  for (let k = 0; k < layers; k++) {
    const px = (x: number, y: number, c: number) => b.data[((k * n + ((y + n) % n)) * n + ((x + n) % n)) * 4 + c];
    // panels: tone, normal map, lit and tiled 2 × 2 (each n × n, downsampled to ≤ 384 by sampling)
    const P = Math.min(n, 384);
    const W = 3 * P + 8, H = P;
    const img: RGBAImage = { width: W, height: H, data: new Uint8Array(W * H * 4) };
    let sumTone = 0, sumSlope = 0, sumCav = 0;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      sumTone += px(x, y, 2) / 255 * 2; sumCav += px(x, y, 3) / 255;
      sumSlope += Math.hypot(px(x, y, 0) / 255 * 2 - 1, px(x, y, 1) / 255 * 2 - 1);
    }
    const put = (ox: number, x: number, y: number, r: number, g: number, bl: number) => {
      const o = (y * W + ox + x) * 4;
      img.data[o] = Math.max(0, Math.min(255, r)); img.data[o + 1] = Math.max(0, Math.min(255, g)); img.data[o + 2] = Math.max(0, Math.min(255, bl)); img.data[o + 3] = 255;
    };
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
      const sx = Math.floor((x * n) / P), sy = Math.floor((y * n) / P);
      const tone = px(sx, sy, 2);
      put(0, x, y, tone, tone, tone);
      put(P + 4, x, y, px(sx, sy, 0), px(sx, sy, 1), 255);
      // 2 × 2 tiling of the lit surface, each tile P / 2 wide
      const tx = (x * 2 * n) / P, ty = (y * 2 * n) / P;
      const ix = Math.floor(tx) % n, iy = Math.floor(ty) % n;
      const nx = px(ix, iy, 0) / 255 * 2 - 1, ny = px(ix, iy, 1) / 255 * 2 - 1;
      const lit = Math.max(0, 0.62 + 1.4 * (nx * -0.6 + ny * -0.6)) * (px(ix, iy, 2) / 255 * 2) * (0.55 + 0.45 * px(ix, iy, 3) / 255);
      put(2 * P + 8, x, y, lit * 150, lit * 150, lit * 150);
    }
    const name = `${GROUND_LAYERS[k]}${mip ? `-mip${mip}` : ''}.png`;
    await Bun.write(path.join(out, name), encodePNG(img));
    console.log(`${GROUND_LAYERS[k].padEnd(7)} mip ${mip}: mean tone ${(sumTone / (n * n)).toFixed(3)}  mean slope ${(sumSlope / (n * n)).toFixed(3)}  mean cavity ${(sumCav / (n * n)).toFixed(3)}  → ${name}`);
  }
}
console.log('written to', out);
