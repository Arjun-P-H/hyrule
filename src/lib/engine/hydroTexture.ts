/**
 * The rivers' node table for the WebGL2 renderer (hydro.ts hydroTable; the WebGPU twin is
 * gpu/hydroBuffer.ts): an RGBA32F texture of six texels a node, HYDRO.window nodes a side, read
 * with texelFetch as u_hydro (header u_hydro_head) by landsGlsl.ts. Re-centred when the camera has
 * travelled a quarter of the window from its centre.
 */

import { HYDRO } from '../constants';
import { hydroTable, type HydroTable } from '../hydro';

export class HydroTexture {
  readonly tex: WebGLTexture;
  /** u_hydro_head: (i0, j0, n, node size). */
  readonly head = new Float32Array(4);
  private table: HydroTable | null = null;
  private seed = NaN;

  constructor(private readonly gl: WebGL2RenderingContext) {
    const tex = gl.createTexture();
    if (!tex) throw new Error('Failed to create the rivers\' node texture');
    this.tex = tex;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, HYDRO.window * 6, HYDRO.window);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Keep the window round world (x, z) for `seed`; true when the table was rebuilt (re-upload u_hydro_head). */
  update(x: number, z: number, seed: number): boolean {
    const t = this.table;
    if (t && seed === this.seed) {
      const half = 0.5 * HYDRO.window * HYDRO.node;
      const cx = t.i0 * HYDRO.node + half, cz = t.j0 * HYDRO.node + half;
      if (Math.abs(x - cx) < 0.5 * half && Math.abs(z - cz) < 0.5 * half) return false;
    }
    const nt = hydroTable(x, z, seed, t ?? undefined);
    this.table = nt;
    this.seed = seed;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, HYDRO.window * 6, HYDRO.window, gl.RGBA, gl.FLOAT, nt.data.subarray(4));
    this.head.set(nt.data.subarray(0, 4));
    return true;
  }
}
