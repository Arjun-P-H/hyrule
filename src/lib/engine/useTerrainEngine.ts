'use client';

import { useEffect, useRef, useState, type RefObject } from 'react';
import { createStatsStore } from './statsStore';
import type { EngineHandle } from './engineCore';
import { DEFAULT_SEED, CAMERA, type PresetId } from '@/lib/constants';

export type { PresetId };

// Fetch the engine chunk as soon as this module evaluates in the browser, in
// parallel with hydration, rather than after the first commit; the effect
// awaits the same promise. (The no-op catch only silences an early rejection;
// the effect reports it.)
const engineCore = typeof window === 'undefined' ? null : import('./engineCore');
engineCore?.catch(() => {});

export type TerrainEngineOptions = {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  seed?: number;
  preset?: PresetId;
  fov?: number;
  sensitivity?: number;
  speedMultiplier?: number;
  enablePointerLock?: boolean;
  running?: boolean;
  onPointerLockChange?: (locked: boolean) => void;
};

/**
 * React binding for the terrain engine. The engine itself (engineCore.ts: the
 * math twin, the shader source and the frame loop) is a dynamic import(), so
 * it ships as its own chunk: the page hydrates without waiting for it while
 * the chunk downloads in parallel.
 */
export function useTerrainEngine({
  canvasRef,
  seed = DEFAULT_SEED,
  preset = 'med',
  fov = CAMERA.fov,
  sensitivity = CAMERA.sensitivity,
  speedMultiplier = 1.0,
  enablePointerLock = false,
  running = true,
  onPointerLockChange,
  onToggleHud,
}: TerrainEngineOptions & { onToggleHud?: () => void }) {
  const [statsStore] = useState(createStatsStore);
  const [error, setError] = useState<string | null>(null);

  const seedRef = useRef(seed);
  const presetRef = useRef(preset);
  const runRef = useRef(running);
  const fovRef = useRef(fov);
  const sensitivityRef = useRef(sensitivity);
  const speedMultiplierRef = useRef(speedMultiplier);
  const hudToggleRef = useRef(onToggleHud);
  const onPointerLockChangeRef = useRef(onPointerLockChange);
  const handleRef = useRef<EngineHandle | null>(null);

  seedRef.current = seed;
  presetRef.current = preset;
  runRef.current = running;
  fovRef.current = fov;
  sensitivityRef.current = sensitivity;
  speedMultiplierRef.current = speedMultiplier;
  hudToggleRef.current = onToggleHud;
  onPointerLockChangeRef.current = onPointerLockChange;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    (engineCore ?? import('./engineCore'))
      .then(({ startEngine }) => {
        if (cancelled) return;
        handleRef.current = startEngine(canvas, statsStore, {
          seed: seedRef,
          preset: presetRef,
          running: runRef,
          fov: fovRef,
          sensitivity: sensitivityRef,
          speedMultiplier: speedMultiplierRef,
          onToggleHud: hudToggleRef,
          onPointerLockChange: onPointerLockChangeRef,
        }, enablePointerLock);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      handleRef.current?.dispose();
      handleRef.current = null;
    };
  }, [canvasRef, enablePointerLock, statsStore]);

  return {
    statsStore,
    error,
    respawn: () => handleRef.current?.respawn(),
    toggleMode: () => handleRef.current?.toggleMode(),
  };
}
