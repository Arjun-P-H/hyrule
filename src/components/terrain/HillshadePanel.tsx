'use client';

/**
 * Hillshade relief map — the CPU twin rendered as a 2D vista (spec §30 panel).
 * Samples E(x, z) on a grid around the tuned spawn, shades with the §17
 * analytic normal against the contract sun, and paints the water plane at
 * W_LEVEL. Computed row-by-row across animation frames so the UI never janks.
 */

import { useEffect, useRef } from 'react';
import { clamp, elevation } from '@/lib/terrainMath';
import { DEFAULT_SEED, SPAWN, LIGHT, VALLEY } from '@/lib/constants';

const SIZE = 200; // grid cells per side
const EXTENT = 3600.0; // metres across

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
const clamp01 = (x: number) => clamp(x, 0, 1);

export default function HillshadePanel({
  className = '',
  seed = DEFAULT_SEED,
}: {
  className?: string;
  seed?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = SIZE;
    canvas.height = SIZE;

    const sun = LIGHT.sunDir;
    const sl = Math.hypot(sun[0], sun[1], sun[2]);
    const sx = sun[0] / sl, sy = sun[1] / sl, sz = sun[2] / sl;

    const step = EXTENT / (SIZE - 1);
    const data = new Float32Array(SIZE * SIZE);
    let row = 0;
    let cancelled = false;
    let rafId = 0;

    const sample = () => {
      if (cancelled) return;
      if (row >= SIZE) { paint(); return; }
      // sample 8 rows per frame (~2.5 ms budget for silky smooth, non-blocking display)
      for (let r = 0; r < 8 && row < SIZE; r++, row++) {
        const z = SPAWN.z - EXTENT / 2 + row * step;
        for (let j = 0; j < SIZE; j++) {
          const x = SPAWN.x - EXTENT / 2 + j * step;
          data[row * SIZE + j] = elevation(x, z, seed, 0);
        }
      }
      rafId = requestAnimationFrame(sample);
    };

    const paint = () => {
      const img = ctx.createImageData(SIZE, SIZE);
      for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
          const e = data[i * SIZE + j];
          // §17 gradient from the sampled grid (cell step = 18.09 m)
          const eL = data[i * SIZE + Math.max(j - 1, 0)];
          const eR = data[i * SIZE + Math.min(j + 1, SIZE - 1)];
          const eD = data[Math.max(i - 1, 0) * SIZE + j];
          const eU = data[Math.min(i + 1, SIZE - 1) * SIZE + j];
          const gx = (eR - eL) / (2 * step);
          const gz = (eU - eD) / (2 * step);
          const gl = Math.hypot(gx, 1, gz);
          const ndl = Math.max((-gx / gl) * sx + (1 / gl) * sy + (-gz / gl) * sz, 0);

          let r: number, g: number, b: number;
          if (e < VALLEY.wLevel) {
            const d = clamp01((VALLEY.wLevel - e) / 22);
            r = lerp(0.10, 0.03, d); g = lerp(0.30, 0.10, d); b = lerp(0.29, 0.13, d);
          } else {
            // elevation tint: grass → rock → snow
            let tr: number, tg: number, tb: number;
            const he = clamp01((e - 40) / 700);
            if (he < 0.5) {
              const t2 = he * 2;
              tr = lerp(0.31, 0.43, t2); tg = lerp(0.43, 0.39, t2); tb = lerp(0.19, 0.34, t2);
            } else {
              const t2 = (he - 0.5) * 2;
              tr = lerp(0.43, 0.92, t2); tg = lerp(0.39, 0.93, t2); tb = lerp(0.34, 0.95, t2);
            }
            const shade = 0.30 + 0.70 * ndl;
            r = tr * shade; g = tg * shade; b = tb * shade;
          }
          const o = (i * SIZE + j) * 4;
          img.data[o] = Math.round(Math.min(255, r * 255));
          img.data[o + 1] = Math.round(Math.min(255, g * 255));
          img.data[o + 2] = Math.round(Math.min(255, b * 255));
          img.data[o + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      // spawn crosshair
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(SIZE / 2 - 7, SIZE / 2); ctx.lineTo(SIZE / 2 + 7, SIZE / 2);
      ctx.moveTo(SIZE / 2, SIZE / 2 - 7); ctx.lineTo(SIZE / 2, SIZE / 2 + 7);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = '9px monospace';
      ctx.fillText('spawn', SIZE / 2 + 10, SIZE / 2 + 3);
      ctx.fillText('N', 5, 12);
      ctx.beginPath(); ctx.moveTo(6, 16); ctx.lineTo(6, 26); ctx.stroke();
    };

    let idleId: number | null = null;
    let timerId: ReturnType<typeof setTimeout> | null = null;

    const startSampling = () => {
      if (!cancelled) {
        rafId = requestAnimationFrame(sample);
      }
    };

    if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
      idleId = (window as unknown as { requestIdleCallback: (cb: () => void, opts?: { timeout: number }) => number }).requestIdleCallback(
        startSampling,
        { timeout: 200 }
      );
    } else {
      timerId = setTimeout(startSampling, 50);
    }

    return () => {
      cancelled = true;
      if (idleId !== null && typeof window !== 'undefined' && 'cancelIdleCallback' in window) {
        (window as unknown as { cancelIdleCallback: (id: number) => void }).cancelIdleCallback(idleId);
      }
      if (timerId !== null) clearTimeout(timerId);
      cancelAnimationFrame(rafId);
    };
  }, [seed]);

  return (
    <canvas
      ref={ref}
      width={SIZE}
      height={SIZE}
      className={`block aspect-square h-auto w-full rounded-lg border border-stone-800 bg-stone-900/40 image-rendering-pixelated ${className}`}
      style={{ imageRendering: 'pixelated', aspectRatio: '1 / 1' }}
      role="img"
      aria-label="Hillshade relief map of the terrain field around the spawn point, computed live by the CPU twin"
    />
  );
}
