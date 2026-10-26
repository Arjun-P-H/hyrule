'use client';

import { useEffect, useRef, useState, type RefObject } from 'react';
import { createStatsStore, type CamMode } from './statsStore';
import { createCombatHudStore } from '@/lib/combat/hud';
import { createPouchStore, type PouchCommand } from '@/lib/combat/pouch';
import { createMapHud } from '@/lib/map/mapHud';
import type { ClockStart, EngineHandle, SpawnAt } from './engineCore';
import type { SkyMode } from '@/lib/gpu/engineGpu';
import type { NetLink } from '@/lib/net/link';
import { DEFAULT_KEYMAP, type Keymap } from './keybinds';
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
function createGpuBackendSerialized(canvas: HTMLCanvasElement, sky: SkyMode) {
  const next = gpuChain.then(async () => {
    const { createGpuBackend } = await import('@/lib/gpu/engineGpu');
    return createGpuBackend(canvas, sky);
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
  /** The sky: the day–night cycle (default) or the classic fixed sky (WebGPU; WebGL2 draws the classic sky). */
  sky?: SkyMode;
  /** Where the day starts (read at start). */
  clockStart?: ClockStart;
  /** Where the hero starts: on the land (default) or up on the nearest sky archipelago (read at start). */
  spawnAt?: SpawnAt;
  /** The keys do nothing while true (an overlay such as the multiplayer lobby is open). */
  inputBlocked?: boolean;
  /** The hero's character (src/lib/hero/characters.ts): its dress. */
  character?: number;
  /** Multiplayer session (src/lib/net/session.ts), when there is one. */
  net?: NetLink | null;
  /** Foes in the world (default true; WebGPU draws them). */
  foes?: boolean;
  /** Start with a full pouch (read at start; ?kit=1): one of every weapon, shield and bow, and some of every food. */
  kit?: boolean;
  /** Which action each key presses (src/lib/engine/keybinds.ts); the default keys when omitted. */
  keymap?: Keymap;
  /** The inventory screen is open: the hero is left alone (and alone in the world, the world stands still). */
  menuOpen?: boolean;
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
  sky = 'cycle',
  clockStart,
  spawnAt,
  inputBlocked = false,
  character = 0,
  net = null,
  foes = true,
  kit = false,
  keymap = DEFAULT_KEYMAP,
  menuOpen = false,
  onPointerLockChange,
  onToggleHud,
  onHudStep,
  onToggleInventory,
  onToggleMap,
}: TerrainEngineOptions & { onToggleHud?: () => void; onHudStep?: (delta: 1 | -1) => void; onToggleInventory?: () => void; onToggleMap?: () => void }) {
  const [statsStore] = useState(createStatsStore);
  const [combatHud] = useState(createCombatHudStore);
  const [pouch] = useState(createPouchStore);
  const [mapHud] = useState(createMapHud);
  const [error, setError] = useState<string | null>(null);

  const seedRef = useRef(seed);
  const presetRef = useRef(preset);
  const runRef = useRef(running);
  const fovRef = useRef(fov);
  const sensitivityRef = useRef(sensitivity);
  const speedMultiplierRef = useRef(speedMultiplier);
  const hudToggleRef = useRef(onToggleHud);
  const hudStepRef = useRef(onHudStep);
  const onPointerLockChangeRef = useRef(onPointerLockChange);
  const skyRef = useRef(sky);
  const clockStartRef = useRef(clockStart);
  const spawnAtRef = useRef(spawnAt);
  const inputBlockedRef = useRef(inputBlocked);
  const characterRef = useRef(character);
  const netRef = useRef(net);
  const foesRef = useRef(foes);
  const kitRef = useRef(kit);
  const keymapRef = useRef(keymap);
  const menuOpenRef = useRef(menuOpen);
  const inventoryToggleRef = useRef(onToggleInventory);
  const mapToggleRef = useRef(onToggleMap);
  const handleRef = useRef<EngineHandle | null>(null);

  seedRef.current = seed;
  presetRef.current = preset;
  runRef.current = running;
  fovRef.current = fov;
  sensitivityRef.current = sensitivity;
  speedMultiplierRef.current = speedMultiplier;
  hudToggleRef.current = onToggleHud;
  hudStepRef.current = onHudStep;
  onPointerLockChangeRef.current = onPointerLockChange;
  skyRef.current = sky;
  inputBlockedRef.current = inputBlocked;
  characterRef.current = character;
  netRef.current = net;
  foesRef.current = foes;
  keymapRef.current = keymap;
  menuOpenRef.current = menuOpen;
  inventoryToggleRef.current = onToggleInventory;
  mapToggleRef.current = onToggleMap;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    const gpu = wantsWebGpu(renderer)
      ? createGpuBackendSerialized(canvas, skyRef.current).catch((e: unknown) => {
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
          onHudStep: hudStepRef,
          onPointerLockChange: onPointerLockChangeRef,
          sky: skyRef,
          inputBlocked: inputBlockedRef,
          character: characterRef,
          net: netRef,
          foes: foesRef,
          kit: kitRef,
          keymap: keymapRef,
          menuOpen: menuOpenRef,
          onToggleInventory: inventoryToggleRef,
          onToggleMap: mapToggleRef,
          combatHud,
          pouch,
          mapHud,
        }, enablePointerLock, initialMode, backend, clockStartRef.current, spawnAtRef.current);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      handleRef.current?.dispose();
      handleRef.current = null;
    };
  }, [canvasRef, enablePointerLock, initialMode, renderer, statsStore, combatHud, pouch, mapHud]);

  return {
    statsStore,
    combatHud,
    pouch,
    mapHud,
    error,
    respawn: () => handleRef.current?.respawn(),
    travel: () => handleRef.current?.travel(),
    toggleMode: () => handleRef.current?.toggleMode(),
    toggleView: () => handleRef.current?.toggleView(),
    setHour: (hour: number) => handleRef.current?.setHour(hour),
    setTimePaused: (paused: boolean) => handleRef.current?.setTimePaused(paused),
    setDayMinutes: (minutes: number) => handleRef.current?.setDayMinutes(minutes),
    clock: () => handleRef.current?.clock() ?? null,
    pouchCommand: (cmd: PouchCommand) => handleRef.current?.pouch(cmd) ?? false,
  };
}
