/**
 * The day–night sky on the GPU (host side of sky.wgsl.ts's compute passes).
 * ---------------------------------------------------------------------------
 *   const sky = new SkyCycle(device); await sky.init();  // transmittance and multiple-scattering LUTs, once
 *   sky.setFrame(days, camY, time)   // this frame's light rig (lightRig.ts), packed into the sky uniform
 *   sky.encode(enc)                  // the sky-view LUT, then the probe of it
 * The sky-view LUT and the probe's results are double-buffered: each frame
 * computes one set while the frame's passes read the other, finished the
 * frame before (reading a texture in the frame that writes it stalls the M1;
 * a sky a frame old is invisible). The renderer binds the uniform and set
 * sampled() in its group 0 (layoutEntries / bindEntries). The rig stays
 * readable (sky.rig: the key light for the tree shadow map, the hour for the HUD).
 */

import { computeLightRig, createLightRig, KEY_ALTS, KEY_TOP, MOON_E, SUN_E, moonLightShare, type LightRig } from '../sky/lightRig';
import { MS_LUT, SKY_BYTES, SKY_LIGHT_BYTES, SKY_LUT_WGSL, TRANS_LUT, VIEW_LUT } from './sky.wgsl';

const DEG = Math.PI / 180;
/**
 * Brightness of the sky's scattered light over the sunlight on the land: the
 * horizon at the classic sun's elevation as bright as the classic horizon.
 */
const SKY_GAIN = 2.7;
/**
 * The moonlit sky, relative to how brightly the moon lights the land: much
 * darker than physics would have it at a night exposure, so the night sky
 * stays a deep blue for the stars while the land stays readable.
 */
const MOON_SKY = 0.4;
/** Angular radii of the discs as drawn: a little larger than life, the moon much larger (as a storybook moon). */
const SUN_RADIUS = 0.6 * DEG;
const MOON_RADIUS = 2.0 * DEG;
/** Exposure over what the probe's adaptation picks (1: as adapted). */
const EXPOSURE_BIAS = 1;
/** Morning mist: extinction (1/m) at its densest, and its scale height (m) over the water level. */
const MIST_DENSITY = 2.4e-3;
const MIST_HEIGHT = 38;

export type SkyBindings = { sky: number; light: number; view: number };

/** Write a vec4 (xyz = a × k, w) at float offset o. */
function put(D: Float32Array, o: number, a: ArrayLike<number>, w: number, k = 1): void {
  D[o] = a[0] * k; D[o + 1] = a[1] * k; D[o + 2] = a[2] * k; D[o + 3] = w;
}

export class SkyCycle {
  readonly rig: LightRig = createLightRig();
  compileMs = 0;
  uniform!: GPUBuffer;
  private lights: GPUBuffer[] = [];
  private device: GPUDevice;
  private data = new Float32Array(SKY_BYTES / 4);
  private textures: GPUTexture[] = [];
  private views: GPUTextureView[] = [];
  private viewPipe!: GPUComputePipeline;
  private probePipe!: GPUComputePipeline;
  private viewBinds: GPUBindGroup[] = [];
  private probeBinds: GPUBindGroup[] = [];
  /** The set computed this frame (the other is sampled); both are computed on the first frame. */
  private cur = 0;
  private fresh = true;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(): Promise<void> {
    const d = this.device;
    const t0 = performance.now();
    const shaders = d.createShaderModule({ code: SKY_LUT_WGSL });
    const [transPipe, msPipe, viewPipe, probePipe] = await Promise.all(
      ['transmittanceLut', 'multiScatterLut', 'skyViewLut', 'skyProbe'].map((entryPoint) =>
        d.createComputePipelineAsync({ layout: 'auto', compute: { module: shaders, entryPoint } })),
    );
    this.compileMs = performance.now() - t0;
    this.viewPipe = viewPipe;
    this.probePipe = probePipe;
    const tex = (w: number, h: number) => {
      const t = d.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
      this.textures.push(t);
      return t;
    };
    const trans = tex(TRANS_LUT[0], TRANS_LUT[1]).createView();
    const ms = tex(MS_LUT, MS_LUT).createView();
    const samp = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.uniform = d.createBuffer({ size: SKY_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const transBind = d.createBindGroup({ layout: transPipe.getBindGroupLayout(0), entries: [{ binding: 4, resource: trans }] });
    const msBind = d.createBindGroup({ layout: msPipe.getBindGroupLayout(0), entries: [
      { binding: 1, resource: trans }, { binding: 3, resource: samp }, { binding: 4, resource: ms },
    ] });
    for (let k = 0; k < 2; k++) {
      const view = tex(VIEW_LUT[0], VIEW_LUT[1]).createView();
      // written by the probe as storage, read by the next frame's passes as a uniform (cached far better)
      const light = d.createBuffer({ size: SKY_LIGHT_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC });
      this.views.push(view);
      this.lights.push(light);
      this.viewBinds.push(d.createBindGroup({ layout: viewPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: trans }, { binding: 2, resource: ms },
        { binding: 3, resource: samp }, { binding: 4, resource: view },
      ] }));
      this.probeBinds.push(d.createBindGroup({ layout: probePipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uniform } }, { binding: 3, resource: samp }, { binding: 5, resource: view },
        { binding: 6, resource: { buffer: light } },
      ] }));
    }
    // The atmosphere's fixed LUTs, once.
    const enc = d.createCommandEncoder();
    const tp = enc.beginComputePass();
    tp.setPipeline(transPipe);
    tp.setBindGroup(0, transBind);
    tp.dispatchWorkgroups(Math.ceil(TRANS_LUT[0] / 8), Math.ceil(TRANS_LUT[1] / 8));
    tp.end();
    const mp = enc.beginComputePass();
    mp.setPipeline(msPipe);
    mp.setBindGroup(0, msBind);
    mp.dispatchWorkgroups(MS_LUT, MS_LUT);
    mp.end();
    d.queue.submit([enc.finish()]);
  }

  /** The light at `days` (game days since day 0, 00:00) for a camera at altitude camY (m); `time` (s) moves the stars' twinkle. */
  setFrame(days: number, camY: number, time: number): void {
    this.cur = 1 - this.cur;
    const r = computeLightRig(days, camY, this.rig);
    const D = this.data;
    put(D, 0, r.sun, r.sunTheta);
    put(D, 4, r.moon, r.moonTheta);
    put(D, 8, r.key, r.keyTheta);
    put(D, 12, SUN_E, Math.max(camY, 0) * 0.001 + 0.002, SKY_GAIN);
    put(D, 16, MOON_E, r.moonIllum, SKY_GAIN * MOON_SKY * moonLightShare(r.moonIllum));
    put(D, 20, r.sunDisc, SUN_RADIUS);
    put(D, 24, r.moonDisc, MOON_RADIUS);
    put(D, 28, r.starRows[0], r.stars);
    put(D, 32, r.starRows[1], time);
    put(D, 36, r.starRows[2], r.night);
    D[40] = EXPOSURE_BIAS; D[41] = r.saturation; D[42] = r.night; D[43] = 0;
    D[44] = r.haze; D[45] = r.mist * MIST_DENSITY; D[46] = MIST_HEIGHT; D[47] = camY;
    D[48] = r.cloudShift; D[49] = r.cloudLight; D[50] = r.milkyWay; D[51] = r.twilight;
    const kl = Math.hypot(r.key[0], r.key[2]) || 1;
    D[52] = r.key[0] / kl; D[53] = r.key[2] / kl; D[54] = 0; D[55] = 0;
    // the key light at the camera's altitude, as keyCol() interpolates it
    const kf = Math.sqrt(Math.min(Math.max(camY / KEY_TOP, 0), 1)) * (KEY_ALTS - 1);
    const ki = Math.min(Math.floor(kf), KEY_ALTS - 2), kt = kf - ki;
    for (let c = 0; c < 3; c++) D[56 + c] = r.keyCols[ki * 4 + c] * (1 - kt) + r.keyCols[(ki + 1) * 4 + c] * kt;
    D[59] = 0;
    D.set(r.keyCols, 60);
    this.device.queue.writeBuffer(this.uniform, 0, D);
  }

  /** The set this frame's passes sample (pick the group-0 bind group made with it). */
  sampled(): number {
    return 1 - this.cur;
  }

  /** This frame's sky-view LUT and the probe of it, into the set not sampled (the first frame fills both). */
  encode(enc: GPUCommandEncoder): void {
    const p = enc.beginComputePass();
    for (const k of this.fresh ? [0, 1] : [this.cur]) {
      p.setPipeline(this.viewPipe);
      p.setBindGroup(0, this.viewBinds[k]);
      p.dispatchWorkgroups(Math.ceil(VIEW_LUT[0] / 8), Math.ceil(VIEW_LUT[1] / 8));
      p.setPipeline(this.probePipe);
      p.setBindGroup(0, this.probeBinds[k]);
      p.dispatchWorkgroups(1);
    }
    this.fresh = false;
    p.end();
  }

  /** What the probe measured last (SkyLight, sky.wgsl.ts; debug readback, waits for the GPU). */
  async readLight(): Promise<Float32Array> {
    const d = this.device;
    const rb = d.createBuffer({ size: SKY_LIGHT_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.lights[this.cur], 0, rb, 0, SKY_LIGHT_BYTES);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return out;
  }

  /** Entries the sky adds to a renderer's group 0. */
  static layoutEntries(b: SkyBindings): GPUBindGroupLayoutEntry[] {
    const VFC = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    return [
      { binding: b.sky, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: b.light, visibility: VFC, buffer: { type: 'uniform' } },
      { binding: b.view, visibility: VFC, texture: { sampleType: 'float', viewDimension: '2d' } },
    ];
  }

  /** Group-0 entries for frames that sample set `which` (see sampled()). */
  bindEntries(b: SkyBindings, which: number): GPUBindGroupEntry[] {
    return [
      { binding: b.sky, resource: { buffer: this.uniform } },
      { binding: b.light, resource: { buffer: this.lights[which] } },
      { binding: b.view, resource: this.views[which] },
    ];
  }

  destroy(): void {
    this.uniform?.destroy();
    for (const b of this.lights) b.destroy();
    for (const t of this.textures) t.destroy();
  }
}

