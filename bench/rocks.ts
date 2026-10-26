/**
 * The rock meshes, looked at (a small software rasteriser: no browser needed).
 * ---------------------------------------------------------------------------
 *   bun bench/rocks.ts [--out=bench/out/rocks]
 *
 * Writes sheet.png: a row a kind, a column a level of detail (finest first), each lit from the upper left
 * with its occlusion, seen from the front and a little above; and prints the vertex counts and extents.
 */

import path from 'node:path';
import { mkdirSync } from 'node:fs';
import { encodePNG, type RGBAImage } from './lib/image';
import { buildRockModels, ROCK_KINDS, ROCK_LODS, ROCK_VERTEX_BYTES } from '../src/lib/rocks/models';

const args = process.argv.slice(2);
const out = path.resolve(import.meta.dir, '..', args.find((a) => a.startsWith('--out='))?.split('=')[1] ?? 'bench/out/rocks');
mkdirSync(out, { recursive: true });

const models = buildRockModels();
const dv = new DataView(models.vertices);
const S = 200;
const W = S * ROCK_LODS, H = S * ROCK_KINDS.length;
const img: RGBAImage = { width: W, height: H, data: new Uint8Array(W * H * 4) };
for (let i = 0; i < W * H; i++) { img.data[i * 4] = 214; img.data[i * 4 + 1] = 226; img.data[i * 4 + 2] = 236; img.data[i * 4 + 3] = 255; }

const L = (() => { const v = [-0.5, 0.7, 0.55]; const l = Math.hypot(...v); return v.map((x) => x / l); })();
ROCK_KINDS.forEach((kind, ki) => {
  for (let lod = 0; lod < ROCK_LODS; lod++) {
    const m = models.meshes[ki * ROCK_LODS + lod];
    const zbuf = new Float32Array(S * S).fill(-1e9);
    const cx = lod * S + S / 2, cy = ki * S + S * 0.62;
    const scale = S * 0.34;
    const pitch = 0.32, cp = Math.cos(pitch), sp = Math.sin(pitch), yaw = 0.5, cyw = Math.cos(yaw), syw = Math.sin(yaw);
    const vert = (i: number) => {
      const o = (m.first + i) * ROCK_VERTEX_BYTES;
      const x = dv.getFloat32(o, true), y = dv.getFloat32(o + 4, true), z = dv.getFloat32(o + 8, true);
      const n = [dv.getInt8(o + 12) / 127, dv.getInt8(o + 13) / 127, dv.getInt8(o + 14) / 127];
      const ao = dv.getUint8(o + 16) / 255, h = dv.getUint8(o + 17) / 255;
      // view: yaw about y, then tilt down by pitch
      const rx = cyw * x + syw * z, rz = -syw * x + cyw * z;
      const ry = cp * y - sp * rz, rz2 = sp * y + cp * rz;
      const nx = cyw * n[0] + syw * n[2], nz = -syw * n[0] + cyw * n[2];
      return { sx: cx + rx * scale, sy: cy - ry * scale, z: rz2, n: [nx, cp * n[1] - sp * nz, sp * n[1] + cp * nz], ao, h };
    };
    for (let t = 0; t < m.count; t += 3) {
      const a = vert(t), b = vert(t + 1), c = vert(t + 2);
      const minX = Math.max(Math.floor(Math.min(a.sx, b.sx, c.sx)), lod * S), maxX = Math.min(Math.ceil(Math.max(a.sx, b.sx, c.sx)), lod * S + S - 1);
      const minY = Math.max(Math.floor(Math.min(a.sy, b.sy, c.sy)), ki * S), maxY = Math.min(Math.ceil(Math.max(a.sy, b.sy, c.sy)), ki * S + S - 1);
      const d = (b.sx - a.sx) * (c.sy - a.sy) - (c.sx - a.sx) * (b.sy - a.sy);
      if (Math.abs(d) < 1e-6) continue;
      for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) {
        const w1 = ((x - a.sx) * (c.sy - a.sy) - (c.sx - a.sx) * (y - a.sy)) / d;
        const w2 = ((b.sx - a.sx) * (y - a.sy) - (x - a.sx) * (b.sy - a.sy)) / d;
        const w0 = 1 - w1 - w2;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
        const z = w0 * a.z + w1 * b.z + w2 * c.z;
        const zi = (y - ki * S) * S + (x - lod * S);
        if (z <= zbuf[zi]) continue;
        zbuf[zi] = z;
        const n = [0, 1, 2].map((k) => w0 * a.n[k] + w1 * b.n[k] + w2 * c.n[k]);
        const nl = Math.hypot(n[0], n[1], n[2]) || 1;
        const ndl = Math.max(0, (n[0] * L[0] + n[1] * L[1] + n[2] * L[2]) / nl);
        const ao = w0 * a.ao + w1 * b.ao + w2 * c.ao, h = w0 * a.h + w1 * b.h + w2 * c.h;
        const shade = (0.28 + 0.8 * ndl) * (0.55 + 0.45 * ao);
        // grey stone, a little moss on the upper faces, dirt at the foot
        const up = Math.max(0, n[1] / nl);
        const moss = Math.max(0, up - 0.55) * 1.4;
        const dirt = Math.max(0, 0.22 - h) * 2.5;
        const col = [150, 144, 136].map((v, k) => v * (1 - moss) + [70, 110, 50][k] * moss).map((v, k) => v * (1 - dirt) + [100, 78, 52][k] * dirt);
        const o = (y * W + x) * 4;
        for (let k = 0; k < 3; k++) img.data[o + k] = Math.min(255, col[k] * shade);
      }
    }
  }
  const counts = Array.from({ length: ROCK_LODS }, (_, lod) => models.meshes[ki * ROCK_LODS + lod].count / 3);
  const e = models.extent[ki];
  console.log(`${kind.padEnd(8)} faces ${counts.join(' / ').padEnd(14)} radius xz ${e.rxz.toFixed(2)} y ${e.ry.toFixed(2)} top ${e.top.toFixed(2)}`);
});
console.log('vertices', models.meshes.reduce((s, m) => s + m.count, 0), '→', (models.vertices.byteLength / 1024).toFixed(0), 'KB');
await Bun.write(path.join(out, 'sheet.png'), encodePNG(img));
console.log('written', path.join(out, 'sheet.png'));
