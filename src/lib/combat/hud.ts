/**
 * What the combat HUD shows (src/app/game/CombatHud.tsx), out of the engine's
 * loop: the hearts, the weapon, shield and bow in hand and what each has left, the food, the arrows, the
 * crosshair while drawing, a prompt, a banner for a Perfect Guard or a Flurry
 * Rush, the fall. Published only when it changes (a string key), so React
 * re-renders on events, not every frame.
 *
 * Markers over the world (the lock-on reticle, the foes' hearts, their "!" and
 * "?") move every frame: the engine writes their screen positions into one
 * mutable list the HUD's own animation frame reads (no React re-render).
 */

import { useSyncExternalStore } from 'react';

export type CombatHudState = {
  hearts: number;          // heart containers
  health: number;          // quarters
  weapon: string | null;   // the weapon in hand
  wear: number;            // what it has left 0..1 (1: never wears)
  weapons: number;         // carried
  weaponKind: number;      // its WEAPONS index (−1: none)
  shield: string | null;   // the shield on the arm
  shieldKind: number;      // its SHIELDS index (−1: none)
  shieldWear: number;      // what it has left 0..1 (1: never wears)
  bow: string | null;      // the bow in hand to draw
  bowKind: number;         // its BOWS index (−1: none)
  bowWear: number;         // what it has left 0..1 (1: never wears)
  food: number;            // meals kept in the pouch
  arrows: number;
  aiming: boolean;         // the bow drawn: the crosshair
  draw: number;            // 0..1 (tenths)
  charge: number;          // a spin's charge 0..1 (tenths)
  prompt: string | null;   // e.g. "E  Pick up Boko Club"
  banner: string | null;   // e.g. "Perfect Guard"
  bannerId: number;        // changes with each banner (to replay its animation)
  dead: boolean;
  downBy: string | null;   // the player who felled the hero (multiplayer), while it is down
  slowed: boolean;         // bullet time or a Flurry Rush's slowed world
  rush: boolean;           // a Flurry Rush can start: attack now
};

export type HudMarker = {
  kind: 'lock' | 'bar' | 'alert' | 'notice';
  x: number; y: number;    // CSS px from the canvas's top left
  value: number;           // a bar's fill 0..1, a notice's 0..1
  near: number;            // 1 close … 0 far (its size)
};

export type CombatHudStore = {
  get: () => CombatHudState | null;
  set: (s: CombatHudState) => void;
  subscribe: (fn: () => void) => () => void;
  /** Markers this frame: the first `count` of `list`. */
  markers: { list: HudMarker[]; count: number; version: number };
};

export function createCombatHudStore(): CombatHudStore {
  let current: CombatHudState | null = null;
  let key = '';
  const listeners = new Set<() => void>();
  const list: HudMarker[] = [];
  for (let k = 0; k < 24; k++) list.push({ kind: 'bar', x: 0, y: 0, value: 0, near: 0 });
  return {
    get: () => current,
    set: (s) => {
      const k = JSON.stringify(s);
      if (k === key) return;
      key = k;
      current = s;
      for (const l of listeners) l();
    },
    subscribe: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    markers: { list, count: 0, version: 0 },
  };
}

const none = () => null;

export function useCombatHud(store: CombatHudStore): CombatHudState | null {
  return useSyncExternalStore(store.subscribe, store.get, none);
}
