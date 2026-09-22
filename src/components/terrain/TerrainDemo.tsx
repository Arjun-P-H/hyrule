'use client';

/**
 * W(x,y) — live WebGL2 raymarch harness (spec §30 Stage-0 GPU parity demo).
 * ---------------------------------------------------------------------------
 * - Driven by unified useTerrainEngine hook.
 * - Walk mode: jump-preserving ground clamp, GROUND_BIAS parity absorber, slope slide gate.
 * - Fly mode: elevation clamped against the CPU twin.
 * - Toroidal floating origin: 8192-meter chunk bounds preventing fp32 jitter.
 */

import { useRef, useState } from 'react';
import { useTerrainEngine, type PresetId } from '@/lib/engine/useTerrainEngine';
import { DEFAULT_SEED } from '@/lib/constants';

export default function TerrainDemo({
  seed = DEFAULT_SEED,
  preset = 'med' as PresetId,
}: {
  seed?: number;
  preset?: PresetId;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [running, setRunning] = useState(true);

  const { stats, error, respawn } = useTerrainEngine({
    canvasRef,
    seed,
    preset,
    running,
    enablePointerLock: false,
  });

  return (
    <div className="relative w-full overflow-hidden rounded-xl border bg-stone-950">
      <canvas
        ref={canvasRef}
        className="block h-[46vh] min-h-[320px] w-full cursor-grab touch-none active:cursor-grabbing sm:h-[56vh]"
        aria-label="Live procedural terrain raymarcher — drag to look, WASD to move, F to toggle walk mode"
        role="img"
      />

      {/* HUD — top left */}
      {stats && !error && (
        <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1 font-mono text-[10px] leading-tight text-stone-200 sm:text-[11px]">
          <span className="rounded bg-stone-950/70 px-2 py-1 backdrop-blur-sm">
            {stats.fps} fps · {preset.toUpperCase()} · {stats.mode.toUpperCase()}
            {stats.mode === 'walk' ? (stats.grounded ? ' · grounded' : ' · airborne') : ''}
            {' '}· {stats.resPct}% res
          </span>
          <span className="rounded bg-stone-950/70 px-2 py-1 backdrop-blur-sm">
            pos [{stats.x.toFixed(1)}, {stats.y.toFixed(1)}, {stats.z.toFixed(1)}]
          </span>
          <span className="rounded bg-stone-950/70 px-2 py-1 backdrop-blur-sm">
            chunk-origin [{stats.originX}, {stats.originZ}] (fp32-exact)
          </span>
        </div>
      )}

      {/* pause / respawn / error controls — top right */}
      <div className="absolute right-3 top-3 flex gap-2">
        {!error && (
          <>
            <button
              type="button"
              onClick={respawn}
              title="Respawn at initial position (R)"
              className="rounded bg-stone-950/70 px-2 py-1 font-mono text-[10px] text-stone-200 backdrop-blur-sm transition hover:bg-stone-900 hover:text-emerald-400 sm:text-[11px]"
            >
              ↺ respawn
            </button>
            <button
              type="button"
              onClick={() => setRunning((v) => !v)}
              className="rounded bg-stone-950/70 px-2 py-1 font-mono text-[10px] text-stone-200 backdrop-blur-sm transition hover:bg-stone-900 sm:text-[11px]"
            >
              {running ? '❚❚ pause' : '▶ resume'}
            </button>
          </>
        )}
      </div>

      {/* controls hint — bottom left */}
      {!error && (
        <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-stone-950/70 px-2 py-1 font-mono text-[10px] text-stone-300 backdrop-blur-sm sm:text-[11px]">
          drag&nbsp;=look · WASD/arrows&nbsp;move · shift&nbsp;sprint · F&nbsp;walk/fly ·
          space&nbsp;jump/up · C&nbsp;down · R&nbsp;respawn
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <p className="max-w-md text-center font-mono text-sm text-red-300">
            WebGL2 failed: {error}
          </p>
        </div>
      )}
    </div>
  );
}
