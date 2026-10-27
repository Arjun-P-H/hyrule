/**
 * A colossus's body for collision: its collision mesh (colossus.bin `coll`: the drawn mesh decimated to ~10 k
 * triangles, weighted as the drawn one) skinned on the CPU with the very joints it is drawn with, so what the hero
 * climbs, walks into and strikes is the stone it sees. With each skin a hash grid over the triangles is rebuilt
 * (cells of CELL m); the queries walk only the cells they reach.
 * ---------------------------------------------------------------------------
 * A point on it is a triangle and its barycentric (u, v) (the weights of its second and third corners): it stays on
 * the same stone as the body moves. Vertex normals (the faces' summed) give a smooth way out of the surface.
 */

const CELL = 1.5;
const BUCKETS = 8192; // (a power of two: the grid's hash table; cells sharing a bucket only add candidates)

/** Where a query met the body: the triangle and (u, v) on it, the point, its distance, the face's normal. */
type BodyHit = { tri: number; u: number; v: number; x: number; y: number; z: number; d: number; nx: number; ny: number; nz: number };

export function createBodyHit(): BodyHit {
  return { tri: -1, u: 0, v: 0, x: 0, y: 0, z: 0, d: 0, nx: 0, ny: 1, nz: 0 };
}

type CollMesh = { pos: Float32Array; joints: Uint8Array; weights: Uint8Array; tris: Uint16Array };

const bucket = (ix: number, iy: number, iz: number) => (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663) ^ Math.imul(iz, 83492791)) & (BUCKETS - 1);

export class ColossusBody {
  private m: CollMesh;
  readonly triCount: number;
  /** Skinned positions and vertex normals (world), face normals. */
  readonly pos: Float64Array;
  readonly nrm: Float64Array;
  private fn: Float64Array;
  // the grid: each bucket's first entry, each entry's next and its triangle
  private head = new Int32Array(BUCKETS);
  private next = new Int32Array(1 << 16);
  private triOf = new Int32Array(1 << 16);
  private seen: Uint32Array;
  private mark = 0;

  constructor(m: CollMesh) {
    this.m = m;
    this.triCount = m.tris.length / 3;
    this.pos = new Float64Array(m.pos.length);
    this.nrm = new Float64Array(m.pos.length);
    this.fn = new Float64Array(this.triCount * 3);
    this.seen = new Uint32Array(this.triCount);
  }

  /** Skinned by joint matrices M (12 a joint: 3×4 rows, world), the normals and the grid with it. */
  skin(M: Float64Array): void {
    const { pos: P0, joints: J, weights: Wt, tris: T } = this.m, P = this.pos, N = this.nrm, F = this.fn;
    const n = P0.length / 3;
    for (let v = 0; v < n; v++) {
      const x = P0[3 * v], y = P0[3 * v + 1], z = P0[3 * v + 2];
      let wx = 0, wy = 0, wz = 0;
      for (let k = 0; k < 4; k++) {
        const w = Wt[4 * v + k];
        if (!w) continue;
        const o = 12 * J[4 * v + k], s = w / 255;
        wx += s * (M[o] * x + M[o + 1] * y + M[o + 2] * z + M[o + 3]);
        wy += s * (M[o + 4] * x + M[o + 5] * y + M[o + 6] * z + M[o + 7]);
        wz += s * (M[o + 8] * x + M[o + 9] * y + M[o + 10] * z + M[o + 11]);
      }
      P[3 * v] = wx; P[3 * v + 1] = wy; P[3 * v + 2] = wz;
    }
    N.fill(0);
    this.head.fill(-1);
    let e = 0;
    for (let t = 0; t < this.triCount; t++) {
      const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
      const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      N[a] += nx; N[a + 1] += ny; N[a + 2] += nz; N[b] += nx; N[b + 1] += ny; N[b + 2] += nz; N[c] += nx; N[c + 1] += ny; N[c + 2] += nz;
      const l = Math.hypot(nx, ny, nz) || 1;
      F[3 * t] = nx / l; F[3 * t + 1] = ny / l; F[3 * t + 2] = nz / l;
      const x0 = Math.floor(Math.min(P[a], P[b], P[c]) / CELL), x1 = Math.floor(Math.max(P[a], P[b], P[c]) / CELL);
      const y0 = Math.floor(Math.min(P[a + 1], P[b + 1], P[c + 1]) / CELL), y1 = Math.floor(Math.max(P[a + 1], P[b + 1], P[c + 1]) / CELL);
      const z0 = Math.floor(Math.min(P[a + 2], P[b + 2], P[c + 2]) / CELL), z1 = Math.floor(Math.max(P[a + 2], P[b + 2], P[c + 2]) / CELL);
      for (let ix = x0; ix <= x1; ix++) for (let iy = y0; iy <= y1; iy++) for (let iz = z0; iz <= z1; iz++) {
        if (e === this.next.length) this.grow();
        const k = bucket(ix, iy, iz);
        this.triOf[e] = t; this.next[e] = this.head[k]; this.head[k] = e++;
      }
    }
    for (let v = 0; v < n; v++) {
      const l = Math.hypot(N[3 * v], N[3 * v + 1], N[3 * v + 2]) || 1;
      N[3 * v] /= l; N[3 * v + 1] /= l; N[3 * v + 2] /= l;
    }
  }

  private grow(): void {
    const n = new Int32Array(this.next.length * 2), t = new Int32Array(this.next.length * 2);
    n.set(this.next); t.set(this.triOf);
    this.next = n; this.triOf = t;
  }

  /** Each triangle in the cells of the box (once a query). */
  private each(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, fn: (t: number) => void): void {
    if (++this.mark === 0xffffffff) { this.seen.fill(0); this.mark = 1; }
    const ix0 = Math.floor(x0 / CELL), ix1 = Math.floor(x1 / CELL), iy0 = Math.floor(y0 / CELL), iy1 = Math.floor(y1 / CELL);
    const iz0 = Math.floor(z0 / CELL), iz1 = Math.floor(z1 / CELL);
    for (let ix = ix0; ix <= ix1; ix++) for (let iy = iy0; iy <= iy1; iy++) for (let iz = iz0; iz <= iz1; iz++) {
      for (let e = this.head[bucket(ix, iy, iz)]; e >= 0; e = this.next[e]) {
        const t = this.triOf[e];
        if (this.seen[t] === this.mark) continue;
        this.seen[t] = this.mark;
        fn(t);
      }
    }
  }

  /**
   * The nearest point of the surface to (px, py, pz) within `reach`, into out (false: none). With `facing`, only
   * faces turned within about 80° of it count (a climb stays on the stone it is on, never through to the far side
   * of a thin part); with `front`, only faces the point is in front of (or just behind).
   */
  nearest(px: number, py: number, pz: number, reach: number, out: BodyHit, facing: ArrayLike<number> | null = null, front = false): boolean {
    const P = this.pos, T = this.m.tris, F = this.fn;
    let best = reach;
    out.tri = -1;
    this.each(px - reach, py - reach, pz - reach, px + reach, py + reach, pz + reach, (t) => {
      const fx = F[3 * t], fy = F[3 * t + 1], fz = F[3 * t + 2];
      if (facing && fx * facing[0] + fy * facing[1] + fz * facing[2] < 0.17) return;
      const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
      closestOnTri(px, py, pz, P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2], P[c], P[c + 1], P[c + 2]);
      const dx = px - Q[0], dy = py - Q[1], dz = pz - Q[2], d = Math.hypot(dx, dy, dz);
      if (d >= best) return;
      if (front && dx * fx + dy * fy + dz * fz < -0.15) return;
      best = d;
      out.tri = t; out.u = Q[3]; out.v = Q[4]; out.x = Q[0]; out.y = Q[1]; out.z = Q[2]; out.d = d;
      out.nx = fx; out.ny = fy; out.nz = fz;
    });
    return out.tri >= 0;
  }

  /** Where segment a → b first meets the surface (its fraction along; −1: misses), into out; `entering`: only going in. */
  ray(ax: number, ay: number, az: number, bx: number, by: number, bz: number, out: BodyHit, entering = false): number {
    const P = this.pos, T = this.m.tris, F = this.fn;
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    let best = 2;
    this.each(Math.min(ax, bx), Math.min(ay, by), Math.min(az, bz), Math.max(ax, bx), Math.max(ay, by), Math.max(az, bz), (t) => {
      if (entering && F[3 * t] * dx + F[3 * t + 1] * dy + F[3 * t + 2] * dz > 0) return;
      const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
      // (Möller–Trumbore, both faces)
      const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
      const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
      const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) return;
      const inv = 1 / det, sx = ax - P[a], sy = ay - P[a + 1], sz = az - P[a + 2];
      const u = (sx * px + sy * py + sz * pz) * inv;
      if (u < 0 || u > 1) return;
      const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * inv;
      if (v < 0 || u + v > 1) return;
      const f = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (f < 0 || f > 1 || f >= best) return;
      best = f;
      out.tri = t; out.u = u; out.v = v; out.x = ax + dx * f; out.y = ay + dy * f; out.z = az + dz * f; out.d = 0;
      out.nx = F[3 * t]; out.ny = F[3 * t + 1]; out.nz = F[3 * t + 2];
    });
    return best <= 1 ? best : -1;
  }

  /** The point (u, v) on triangle t (world), and the smooth way out there (its vertex normals), into p and n. */
  at(t: number, u: number, v: number, p: Float64Array, n: Float64Array): void {
    const T = this.m.tris, P = this.pos, N = this.nrm, w = 1 - u - v;
    const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
    for (let i = 0; i < 3; i++) {
      p[i] = w * P[a + i] + u * P[b + i] + v * P[c + i];
      n[i] = w * N[a + i] + u * N[b + i] + v * N[c + i];
    }
    const l = Math.hypot(n[0], n[1], n[2]) || 1;
    n[0] /= l; n[1] /= l; n[2] /= l;
  }
}

/** The closest point of triangle abc to p into Q (x, y, z, and the weights of b and c); Ericson, RTCD 5.1.5. */
const Q = new Float64Array(5);
function closestOnTri(px: number, py: number, pz: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number): void {
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let u = 0, v = 0;
  if (d1 <= 0 && d2 <= 0) { u = 0; v = 0; } else {
    const bpx = px - bx, bpy = py - by, bpz = pz - bz;
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) { u = 1; v = 0; } else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) { u = d1 / (d1 - d3); v = 0; } else {
        const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
        const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) { u = 0; v = 1; } else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) { u = 0; v = d2 / (d2 - d6); } else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { v = (d4 - d3) / ((d4 - d3) + (d5 - d6)); u = 1 - v; } else {
              const den = 1 / (va + vb + vc);
              u = vb * den; v = vc * den;
            }
          }
        }
      }
    }
  }
  Q[0] = ax + abx * u + acx * v; Q[1] = ay + aby * u + acy * v; Q[2] = az + abz * u + acz * v; Q[3] = u; Q[4] = v;
}
