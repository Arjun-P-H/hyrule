'use client';

import { useEffect, useRef, useState, type RefObject } from 'react';
import { createStatsStore, type CamMode } from './statsStore';
import type { EngineHandle } from './engineCore';
import { DEFAULT_SEED, CAMERA, type PresetId } from '@/lib/constants';

export type { PresetId };

// Fetch the engine chunk as soon as this module evaluates in the browser, in
// parallel with hydration, rather than after the first commit; the effect
// awaits the same promise. (The no-op catch only silences an early rejection;
// the effect reports it.)
const engineCore = typeof window === 'undefined' ? null : import('./engineCore');
engineCore?.catch(() => {});

/**
 * Which graphics API to draw with. 'auto' (the game) takes the WebGPU raster
 * clipmap renderer (src/lib/gpu/engineGpu.ts, its own chunk) where the
 * browser supports it and WebGL2 otherwise; `?renderer=webgl2` or
 * `?renderer=webgpu` in the URL overrides. Backend creation is serialized:
 * under React StrictMode the effect mounts twice on one canvas, and the
 * canvas holds a single WebGPU configuration.
 */
export type RendererChoice = 'auto' | 'webgl2' | 'webgpu';

function wantsWebGpu(choice: RendererChoice): boolean {
  if (typeof window === 'undefined') return false;
  const param = new URLSearchParams(window.location.search).get('renderer');
  const want = param === 'webgl2' || param === 'webgpu' ? param : choice;
  return want === 'webgpu' || (want === 'auto' && 'gpu' in navigator);
}
let gpuChain: Promise<unknown> = Promise.resolve();
function createGpuBackendSerialized(canvas: HTMLCanvasElement) {
  const next = gpuChain.then(async () => {
    const { createGpuBackend } = await import('@/lib/gpu/engineGpu');
    return createGpuBackend(canvas);
  });
  gpuChain = next.catch(() => {});
  return next;
}

export type TerrainEngineOptions = {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  seed?: number;
  preset?: PresetId;
  fov?: number;
  sensitivity?: number;
  speedMultiplier?: number;
  enablePointerLock?: boolean;
  running?: boolean;
  /** Mode on start: 'hero' (third person), 'walk' (first person) or 'fly'. */
  initialMode?: CamMode;
  /** Graphics API (default 'webgl2'); see RendererChoice. */
  renderer?: RendererChoice;
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
  initialMode = 'fly',
  renderer = 'webgl2',
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
    const gpu = wantsWebGpu(renderer)
      ? createGpuBackendSerialized(canvas).catch((e: unknown) => {
          console.warn('WebGPU unavailable, drawing with WebGL2:', e);
          return null;
        })
      : Promise.resolve(null);
    Promise.all([engineCore ?? import('./engineCore'), gpu])
      .then(([{ startEngine }, backend]) => {
        if (cancelled) {
          backend?.dispose();
          return;
        }
        handleRef.current = startEngine(canvas, statsStore, {
          seed: seedRef,
          preset: presetRef,
          running: runRef,
          fov: fovRef,
          sensitivity: sensitivityRef,
          speedMultiplier: speedMultiplierRef,
          onToggleHud: hudToggleRef,
          onPointerLockChange: onPointerLockChangeRef,
        }, enablePointerLock, initialMode, backend);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      handleRef.current?.dispose();
      handleRef.current = null;
    };
  }, [canvasRef, enablePointerLock, initialMode, renderer, statsStore]);

  return {
    statsStore,
    error,
    respawn: () => handleRef.current?.respawn(),
    toggleMode: () => handleRef.current?.toggleMode(),
    toggleView: () => handleRef.current?.toggleView(),
  };
}
