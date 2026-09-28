import { useSyncExternalStore } from 'react';

/**
 * §30 HUD telemetry channel between the engine's rAF loop and React.
 * ---------------------------------------------------------------------------
 * The engine publishes a snapshot at most every 200 ms. Components subscribe
 * to exactly the slice they render, so the 5 Hz telemetry re-renders the HUD
 * only — not the page that owns the engine (buttons, tooltips, dialogs).
 */

export type CamMode = 'walk' | 'fly';

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

/** Camera mode only: re-renders the caller when walk/fly actually changes. */
export function useEngineMode(store: StatsStore): CamMode | null {
  return useSyncExternalStore(store.subscribe, () => selectMode(store), serverSnapshot);
}
