/**
 * Frozen-frame perceptual comparison.
 * ---------------------------------------------------------------------------
 * The baseline is a pinned git revision, rendered in the same browser session
 * as the candidate (same driver, same GPU), so the diff isolates the change.
 * Both are also scored against a converged ground-truth render (tiny horizon
 * clamp, no step growth, 3,000-step loops; transformShader 'truth') so a march
 * approximation is judged by whether it moves the image toward the truth.
 * Frame SHA-256s are recorded so a re-render of the baseline can be verified
 * as bit-identical across sessions.
 */

import path from 'node:path';
import { mkdirSync } from 'node:fs';
import type { BlobRecord } from './gpu/runner';
import { encodePNG, perceptualDiff, sha256, type PerceptualDiff, type RGBAImage } from './lib/image';

export const SHOT_FRAMES = [0, 6, 12, 18, 24, 30, 35, 41, 47, 53];

export type ShotSet = Map<string, RGBAImage>; // key: `${variant}|${preset}|${frame}`

export function collectShots(blobs: BlobRecord[], outDir: string | null): { shots: ShotSet; hashes: Record<string, string> } {
  const shots: ShotSet = new Map();
  const hashes: Record<string, string> = {};
  if (outDir) mkdirSync(outDir, { recursive: true });
  for (const b of blobs) {
    if (b.query.kind !== 'shot') continue;
    const img: RGBAImage = { width: Number(b.query.w), height: Number(b.query.h), data: b.data };
    const key = `${b.query.variant}|${b.query.preset}|${b.query.frame}`;
    shots.set(key, img);
    hashes[key] = sha256(b.data);
    if (outDir) {
      const name = `${b.query.variant.replace(/[^\w.-]/g, '_')}_${b.query.preset}_f${String(b.query.frame).padStart(2, '0')}.png`;
      Bun.write(path.join(outDir, name), encodePNG(img, true));
    }
  }
  return { shots, hashes };
}

export type DiffSummary = {
  preset: string;
  a: string;
  b: string;
  frames: number;
  meanDE: number;        // mean over frames of per-frame mean ΔE_ok
  worstFrameMeanDE: number;
  p99DE: number;         // max over frames of per-frame p99
  fracJND: number;       // mean over frames
  fracVisible: number;
  identical: number;     // mean fraction of bit-identical pixels
  minPsnr: number;
};

export function diffSets(shots: ShotSet, a: string, aPreset: string, b: string, bPreset: string, frames: number[]): DiffSummary {
  const per: PerceptualDiff[] = [];
  for (const f of frames) {
    const A = shots.get(`${a}|${aPreset}|${f}`), B = shots.get(`${b}|${bPreset}|${f}`);
    if (A && B) per.push(perceptualDiff(A, B));
  }
  const mean = (k: keyof PerceptualDiff) => per.reduce((s, d) => s + (d[k] as number), 0) / Math.max(per.length, 1);
  return {
    preset: aPreset === bPreset ? aPreset : `${aPreset}→${bPreset}`,
    a,
    b,
    frames: per.length,
    meanDE: mean('meanDE'),
    worstFrameMeanDE: Math.max(0, ...per.map((d) => d.meanDE)),
    p99DE: Math.max(0, ...per.map((d) => d.p99DE)),
    fracJND: mean('fracJND'),
    fracVisible: mean('fracVisible'),
    identical: mean('identical'),
    minPsnr: Math.min(Infinity, ...per.map((d) => d.psnr)),
  };
}
