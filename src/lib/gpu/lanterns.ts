/**
 * Lantern light (docs/EYRIES.md §3.6): the warm pools the Eyries' lanterns throw on the paving and
 * the walls round them by night, as one texture fetch a pixel however many lanterns there are.
 * ---------------------------------------------------------------------------
 * A LANTERN_N² map, a metre a texel, anchored to the world and wrapping round (texel = world metre
 * mod LANTERN_N; the chunk origin is a multiple of it, so shaders index it by chunk-local metres):
 * made again round the camera when it has moved 32 m or the lanterns round it changed. Every lantern
 * near enough adds its light, falling off to nothing at its reach (red), and the height its light
 * comes from, weighted (green, half float). Shaders light a surface from the map where it lies
 * within LANTERN_SPAN of the camera and a few metres of that height (lanternLight, by night only).
 * No uniform: the stages are at their limit of uniform buffers.
 */

import { CHUNK_SIZE } from '../constants';
import { wgslF32 as f } from './worldgen.wgsl';

const LANTERN_N = 256;
/** How far round the camera the map is good (m): its half size less the 32 m it may lag. */
const LANTERN_SPAN = LANTERN_N / 2 - 32;
/** How far a lantern's light reaches (m), by its strength. */
const REACH = 7;

if (CHUNK_SIZE % LANTERN_N !== 0) throw new Error('lanterns: the chunk size must be a multiple of the map');

const h32 = new Float32Array(1), h32u = new Uint32Array(h32.buffer);
/** A half float's bits (rounding toward zero; tiny values flushed). */
function half(v: number): number {
  h32[0] = v;
  const x = h32u[0];
  const sign = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
  if (e <= 0) return sign;
  if (e >= 31) return sign | 0x7c00;
  return sign | (e << 10) | ((x & 0x7fffff) >>> 13);
}

export class LanternMap {
  private device: GPUDevice;
  private tex: GPUTexture;
  private view: GPUTextureView;
  private sampler: GPUSampler;
  private acc = new Float32Array(LANTERN_N * LANTERN_N);
  private accH = new Float32Array(LANTERN_N * LANTERN_N);
  private texels = new Uint16Array(LANTERN_N * LANTERN_N * 2);
  private at = new Float64Array(4).fill(NaN);
  private version = -1;

  constructor(device: GPUDevice) {
    this.device = device;
    this.tex = device.createTexture({ size: [LANTERN_N, LANTERN_N], format: 'rg16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.view = this.tex.createView();
    // (repeating: the map wraps round, so its edges filter into each other)
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
  }

  /** (fragment shaders only read it) */
  static layoutEntries(): GPUBindGroupLayoutEntry[] {
    return [
      { binding: 35, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 37, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ];
  }

  bindEntries(): GPUBindGroupEntry[] {
    return [{ binding: 35, resource: this.view }, { binding: 37, resource: this.sampler }];
  }

  /**
   * The lanterns round the camera (world x, y, z of each light and its strength, `n` of them;
   * `version` changes when they do) into the map, if it has moved 32 m or they changed. Chunk-local
   * to (ox, oz).
   */
  update(pts: Float64Array, n: number, version: number, cx: number, cy: number, cz: number, ox: number, oz: number): void {
    const at = this.at;
    if (Math.hypot(cx - at[0], cz - at[1]) < 32 && version === this.version && at[2] === ox && at[3] === oz) return;
    at[0] = cx; at[1] = cz; at[2] = ox; at[3] = oz;
    this.version = version;
    const N = LANTERN_N, acc = this.acc, accH = this.accH;
    acc.fill(0);
    accH.fill(0);
    // (the map's cells: chunk-local metres round the camera, each at its own place mod N)
    const lx = Math.floor(cx - ox) - N / 2, lz = Math.floor(cz - oz) - N / 2;
    for (let k = 0; k < n; k++) {
      const px = pts[4 * k] - ox, py = pts[4 * k + 1], pz = pts[4 * k + 2] - oz, s = pts[4 * k + 3];
      const R = REACH * Math.sqrt(s);
      const i0 = Math.max(lx, Math.floor(px - R)), i1 = Math.min(lx + N - 1, Math.ceil(px + R));
      const j0 = Math.max(lz, Math.floor(pz - R)), j1 = Math.min(lz + N - 1, Math.ceil(pz + R));
      for (let j = j0; j <= j1; j++) {
        const dz = j + 0.5 - pz, row = (((j % N) + N) % N) * N;
        for (let i = i0; i <= i1; i++) {
          const dx = i + 0.5 - px;
          const d = Math.sqrt(dx * dx + dz * dz) / R;
          if (d >= 1) continue;
          const w = s * (1 - d) * (1 - d), t = row + (((i % N) + N) % N);
          acc[t] += w;
          accH[t] += w * py;
        }
      }
    }
    const T = this.texels;
    for (let t = 0; t < N * N; t++) {
      const w = acc[t];
      T[2 * t] = half(Math.min(1, w));
      T[2 * t + 1] = half(w > 0 ? accH[t] / w : 0);
    }
    this.device.queue.writeTexture({ texture: this.tex }, T, { bytesPerRow: N * 4, rowsPerImage: N }, [N, N, 1]);
  }

  destroy(): void {
    this.tex.destroy();
  }
}

/** The map in WGSL: lanternLight(p), the warm light falling on a surface at p by night. */
export function lanternsWGSL(): string {
  return /* wgsl */ `
// ---- lantern light (lanterns.ts) ---------------------------------------------------------------------
@group(0) @binding(35) var lanternTex: texture_2d<f32>;
@group(0) @binding(37) var lanternSamp: sampler;
const LANTERN_WARM = vec3f(1.0, 0.58, 0.26);

// The lanterns' light on a surface at p (to multiply by its albedo): near the camera, within a few
// metres of the height it comes from, by night.
fn lanternLight(p: vec3f) -> vec3f {
  let off = abs(p.xz - u.camPos.xz);
  if (max(off.x, off.y) > ${f(LANTERN_SPAN)}) { return vec3f(0.0); }
  let t = textureSampleLevel(lanternTex, lanternSamp, p.xz / ${f(LANTERN_N)}, 0.0);
  let near = 1.0 - smoothstep(${f(2.5)}, ${f(7.5)}, abs(p.y - t.g));
  let fade = 1.0 - smoothstep(${f(LANTERN_SPAN - 16)}, ${f(LANTERN_SPAN)}, max(off.x, off.y));
  return LANTERN_WARM * (t.r * near * fade * lampLevel() * ${f(0.014)});
}
`;
}
