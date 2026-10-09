/**
 * The rivers' node table on the GPU (hydro.ts hydroTable): a window of HYDRO.window² drainage
 * nodes round the camera, computed on the CPU in fp64 and uploaded, so every GPU pass and the CPU
 * twin share one river topology. Bound at binding 63 of group 0 (worldgen.wgsl.ts hydroBuf) by
 * every pipeline that evaluates the land's surface. One per device; re-centred when the camera
 * has travelled a quarter of the window from its centre.
 */

import { HYDRO } from '../constants';
import { HYDRO_STRIDE, hydroTable, type HydroTable } from '../hydro';

const HYDRO_BINDING = 63;

export class HydroBuffer {
  readonly buffer: GPUBuffer;
  private table: HydroTable | null = null;
  private seed = NaN;

  constructor(private readonly device: GPUDevice) {
    this.buffer = device.createBuffer({
      size: (4 + HYDRO.window * HYDRO.window * HYDRO_STRIDE) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  /** Keep the window round world (x, z) for `seed`; true when the table was rebuilt. */
  update(x: number, z: number, seed: number): boolean {
    const t = this.table;
    if (t && seed === this.seed) {
      const half = 0.5 * HYDRO.window * HYDRO.node;
      const cx = t.i0 * HYDRO.node + half, cz = t.j0 * HYDRO.node + half;
      if (Math.abs(x - cx) < 0.5 * half && Math.abs(z - cz) < 0.5 * half) return false;
    }
    this.table = hydroTable(x, z, seed, t ?? undefined);
    this.seed = seed;
    this.device.queue.writeBuffer(this.buffer, 0, this.table.data);
    return true;
  }

  /** Its bind group layout entry. */
  static layoutEntry(visibility: number): GPUBindGroupLayoutEntry {
    return { binding: HYDRO_BINDING, visibility, buffer: { type: 'read-only-storage' } };
  }

  /** Its bind group entry. */
  entry(): GPUBindGroupEntry {
    return { binding: HYDRO_BINDING, resource: { buffer: this.buffer } };
  }
}

const perDevice = new WeakMap<GPUDevice, HydroBuffer>();

/** The device's node table (made on first use). */
export function hydroBufferFor(device: GPUDevice): HydroBuffer {
  let h = perDevice.get(device);
  if (!h) { h = new HydroBuffer(device); perDevice.set(device, h); }
  return h;
}
