import { useSyncExternalStore } from 'react';
import type { HeroAction } from '@/lib/hero/locomotion';
import type { SkyMode } from '@/lib/gpu/engineGpu';

/**
 * §30 HUD telemetry channel between the engine's rAF loop and React.
 * ---------------------------------------------------------------------------
 * The engine publishes a snapshot at most every 200 ms. Components subscribe
 * to exactly the slice they render, so the 5 Hz telemetry re-renders the HUD
 * only — not the page that owns the engine (buttons, tooltips, dialogs).
 */

/** 'hero': third-person character · 'walk': first person · 'fly': free camera. */
export type CamMode = 'hero' | 'walk' | 'fly';

/** The graphics API the engine draws with. */
export type RenderApi = 'webgl2' | 'webgpu';

export type EngineStats = {
  fps: number;
  x: number;
  y: number;
  z: number;
  mode: CamMode;
  grounded: boolean;
  originX: number;
  originZ: number;
  resPct: number;
  action: HeroAction | null; // what the hero is doing (third person only)
  stamina: number;           // hero stamina wheel, 0..1
  exhausted: boolean;
  api: RenderApi;
  /** The sky drawn ('cycle' | 'classic'; null: this renderer has only the classic sky), and the one asked for. */
  sky: SkyMode | null;
  skyWanted: SkyMode;
  /** The sky's clock: hour of the day (0–24), stopped, real minutes per game day. */
  hour: number;
  paused: boolean;
  dayMinutes: number;
  /** The land under the camera (sky/air.ts landName). */
  land: string;
};

export type StatsStore = {
  get: () => EngineStats | null;
  set: (s: EngineStats) => void;
  subscribe: (onChange: () => void) => () => void;
};

export function createStatsStore(): StatsStore {
  let current: EngineStats | null = null;
  const listeners = new Set<() => void>();
  return {
    get: () => current,
    set: (s) => {
      current = s;
      for (const l of listeners) l();
    },
    subscribe: (onChange) => {
      listeners.add(onChange);
      return () => listeners.delete(onChange);
    },
  };
}

const serverSnapshot = () => null;

/** Full snapshot: re-renders the caller at the telemetry rate (≤ 5 Hz). */
export function useEngineStats(store: StatsStore): EngineStats | null {
  return useSyncExternalStore(store.subscribe, store.get, serverSnapshot);
}

/** Mode slice snapshot: a primitive, so React bails out unless it changes. */
export function selectMode(store: StatsStore): CamMode | null {
  return store.get()?.mode ?? null;
}

/** Camera mode only: re-renders the caller when the mode actually changes. */
export function useEngineMode(store: StatsStore): CamMode | null {
  return useSyncExternalStore(store.subscribe, () => selectMode(store), serverSnapshot);
}

/** The graphics API slice: re-renders the caller only when it changes (once, at start). */
export function useEngineApi(store: StatsStore): RenderApi | null {
  return useSyncExternalStore(store.subscribe, () => store.get()?.api ?? null, serverSnapshot);
}

/** The sky and its clock, to the minute (a string slice: React re-renders only when it changes). */
function selectSky(store: StatsStore): string | null {
  const s = store.get();
  if (!s) return null;
  return `${s.sky ?? 'none'}|${s.skyWanted}|${Math.floor(s.hour * 60)}|${s.paused ? 1 : 0}|${s.dayMinutes}`;
}

type SkyState = { sky: SkyMode | null; wanted: SkyMode; minutes: number; paused: boolean; dayMinutes: number };

/** The sky and its clock: re-renders the caller once a game minute at most. */
export function useSkyState(store: StatsStore): SkyState | null {
  const key = useSyncExternalStore(store.subscribe, () => selectSky(store), serverSnapshot);
  if (!key) return null;
  const [sky, wanted, minutes, paused, dayMinutes] = key.split('|');
  return { sky: sky === 'none' ? null : (sky as SkyMode), wanted: wanted as SkyMode, minutes: Number(minutes), paused: paused === '1', dayMinutes: Number(dayMinutes) };
}

/** Stamina in whole percent while in third person, else null (a primitive slice). */
export function selectStamina(store: StatsStore): number | null {
  const s = store.get();
  return s && s.mode === 'hero' ? Math.round(s.stamina * 100) : null;
}

/** Stamina wheel: re-renders the caller only when the shown value changes. */
export function useHeroStamina(store: StatsStore): { percent: number | null; exhausted: boolean } {
  const percent = useSyncExternalStore(store.subscribe, () => selectStamina(store), serverSnapshot);
  const exhausted = useSyncExternalStore(store.subscribe, () => store.get()?.exhausted ?? false, () => false);
  return { percent, exhausted };
}
