/**
 * Minimal RGBA8 PNG codec + OKLab perceptual difference for frozen-frame
 * comparisons. No dependencies beyond node:zlib (bundled with Bun).
 */

import { deflateSync, inflateSync } from 'node:zlib';

export type RGBAImage = { width: number; height: number; data: Uint8Array };

const PNG_SIG = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

function crc32(buf: Uint8Array): number {
  return Bun.hash.crc32(buf) >>> 0;
}

function chunk(type: string, payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + payload.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, payload.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(payload, 8);
  dv.setUint32(8 + payload.length, crc32(out.subarray(4, 8 + payload.length)));
  return out;
}

/** WebGL readPixels rows are bottom-up; pass flipY to store top-down. */
export function encodePNG(img: RGBAImage, flipY = false): Uint8Array {
  const { width: w, height: h, data } = img;
  const stride = w * 4;
  const raw = new Uint8Array((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const srcRow = flipY ? h - 1 - y : y;
    raw[y * (stride + 1)] = 0; // filter: none
    raw.set(data.subarray(srcRow * stride, srcRow * stride + stride), y * (stride + 1) + 1);
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w);
  dv.setUint32(4, h);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  const parts = [PNG_SIG, chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 6 })), chunk('IEND', new Uint8Array(0))];
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

export function decodePNG(buf: Uint8Array): RGBAImage {
  for (let i = 0; i < 8; i++) if (buf[i] !== PNG_SIG[i]) throw new Error('not a PNG');
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 8;
  let w = 0, h = 0;
  const idat: Uint8Array[] = [];
  while (o < buf.length) {
    const len = dv.getUint32(o);
    const type = String.fromCharCode(buf[o + 4], buf[o + 5], buf[o + 6], buf[o + 7]);
    const payload = buf.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') {
      const pdv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
      w = pdv.getUint32(0);
      h = pdv.getUint32(4);
      if (payload[8] !== 8 || payload[9] !== 6) throw new Error('only RGBA8 PNGs are supported');
    } else if (type === 'IDAT') {
      idat.push(payload);
    }
    o += 12 + len;
  }
  const joined = new Uint8Array(idat.reduce((s, p) => s + p.length, 0));
  let jo = 0;
  for (const p of idat) { joined.set(p, jo); jo += p.length; }
  const raw = inflateSync(joined);
  const stride = w * 4;
  const out = new Uint8Array(stride * h);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= 4 ? out[y * stride + x - 4] : 0;
      const b = y > 0 ? out[(y - 1) * stride + x] : 0;
      const c = x >= 4 && y > 0 ? out[(y - 1) * stride + x - 4] : 0;
      let v = row[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[y * stride + x] = v & 255;
    }
  }
  return { width: w, height: h, data: out };
}

// ---------------------------------------------------------------------------
// OKLab perceptual difference (Björn Ottosson, 2020). ΔE_ok ≈ 0.02 is roughly
// one just-noticeable difference for flat patches; we report the fraction of
// pixels above 0.02 (JND) and 0.05 (clearly visible).
// ---------------------------------------------------------------------------

const SRGB_TO_LINEAR = new Float64Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function toOklab(r8: number, g8: number, b8: number, out: Float64Array): void {
  const r = SRGB_TO_LINEAR[r8], g = SRGB_TO_LINEAR[g8], b = SRGB_TO_LINEAR[b8];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  out[0] = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  out[1] = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  out[2] = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
}

export type PerceptualDiff = {
  pixels: number;
  identical: number;      // fraction of bit-identical pixels
  meanDE: number;         // mean ΔE_ok
  p99DE: number;
  maxDE: number;
  fracJND: number;        // fraction of pixels with ΔE_ok > 0.02
  fracVisible: number;    // fraction of pixels with ΔE_ok > 0.05
  psnr: number;           // dB over RGB, Infinity when identical
};

export function perceptualDiff(a: RGBAImage, b: RGBAImage): PerceptualDiff {
  if (a.width !== b.width || a.height !== b.height) throw new Error('image size mismatch');
  const n = a.width * a.height;
  const la = new Float64Array(3), lb = new Float64Array(3);
  const des = new Float64Array(n);
  let same = 0, sum = 0, max = 0, jnd = 0, vis = 0, se = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const ar = a.data[o], ag = a.data[o + 1], ab = a.data[o + 2];
    const br = b.data[o], bg = b.data[o + 1], bb = b.data[o + 2];
    if (ar === br && ag === bg && ab === bb) { same++; des[i] = 0; continue; }
    se += (ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2;
    toOklab(ar, ag, ab, la);
    toOklab(br, bg, bb, lb);
    const de = Math.hypot(la[0] - lb[0], la[1] - lb[1], la[2] - lb[2]);
    des[i] = de;
    sum += de;
    if (de > max) max = de;
    if (de > 0.02) jnd++;
    if (de > 0.05) vis++;
  }
  des.sort();
  const mse = se / (n * 3);
  return {
    pixels: n,
    identical: same / n,
    meanDE: sum / n,
    p99DE: des[Math.min(n - 1, Math.floor(0.99 * (n - 1)))],
    maxDE: max,
    fracJND: jnd / n,
    fracVisible: vis / n,
    psnr: mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse),
  };
}

export function sha256(data: Uint8Array): string {
  return new Bun.CryptoHasher('sha256').update(data).digest('hex');
}
