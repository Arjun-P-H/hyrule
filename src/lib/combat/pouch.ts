/**
 * The pouch as the inventory screen (src/app/game/InventoryScreen.tsx) lists it: the weapons, bows
 * and shields carried with what each has left and which is in hand, the food kept, the quiver and the
 * hearts. The engine reads it out of the combat world while the screen is open and publishes it on
 * change (a string key, as the combat HUD does); the screen answers with commands, which the engine
 * hands the world (equip, unequip, drop, eat).
 */

import { useSyncExternalStore } from 'react';
import { COMBAT } from '../constants';
import type { CombatWorld, PouchCat } from './world';
import { BOWS, FOODS, SHIELDS, WEAPONS } from './weapons';

/** One weapon, bow or shield carried. */
type PouchItem = {
  index: number;      // its place in the list (what the commands take)
  kind: number;       // its entry in WEAPONS, BOWS or SHIELDS
  name: string;
  info: string;
  stat: string;       // e.g. "Attack 12 · Speed ×1"
  wear: number;       // blows (or arrows) it has left; −1: it never wears out
  max: number;        // … and what it had new; −1: never wears out
  equipped: boolean;
  keeps: boolean;     // the hero's own: it can be put away but not dropped
};

/** One kind of food kept. */
type PouchFood = { kind: number; name: string; info: string; count: number; heal: number };

export type PouchState = {
  weapons: PouchItem[];
  bows: PouchItem[];
  shields: PouchItem[];
  food: PouchFood[];
  slots: { weapon: number; bow: number; shield: number };
  arrows: number;
  hearts: number;     // heart containers
  health: number;     // quarters, rounded up as the hearts show
};

export type PouchCommand =
  | { op: 'equip' | 'drop'; cat: PouchCat; index: number }
  | { op: 'unequip'; cat: PouchCat }
  | { op: 'eat'; index: number };

const hold = (push: number) => (push <= 0.8 ? 'Firm' : push <= 1.1 ? 'Steady' : 'Shaky');
const num = (v: number) => String(Math.round(v * 100) / 100);

/** The pouch now. */
export function readPouch(w: CombatWorld): PouchState {
  const left = (c: { wear: number }, max: number) => (c.wear === Infinity ? -1 : Math.min(c.wear, max));
  return {
    weapons: w.weapons.map((c, i) => {
      const k = WEAPONS[c.kind];
      return {
        index: i, kind: c.kind, name: k.name, info: k.info, stat: `Attack ${k.damage} · Speed ×${num(k.pace)}`,
        wear: left(c, k.durability), max: k.durability === Infinity ? -1 : k.durability, equipped: i === w.equipped, keeps: c.wear === Infinity,
      };
    }),
    bows: w.bows.map((c, i) => {
      const k = BOWS[c.kind];
      return {
        index: i, kind: c.kind, name: k.name, info: k.info, stat: `Damage ×${num(k.damage)}`,
        wear: left(c, k.durability), max: k.durability === Infinity ? -1 : k.durability, equipped: i === w.bowOn, keeps: c.wear === Infinity,
      };
    }),
    shields: w.shields.map((c, i) => {
      const k = SHIELDS[c.kind];
      return {
        index: i, kind: c.kind, name: k.name, info: k.info, stat: `Hold: ${hold(k.push)}`,
        wear: left(c, k.durability), max: k.durability === Infinity ? -1 : k.durability, equipped: i === w.shieldOn, keeps: c.wear === Infinity,
      };
    }),
    food: FOODS.map((k, i) => ({ kind: i, name: k.name, info: k.info, count: w.food[i], heal: k.heal })).filter((f) => f.count > 0),
    slots: { weapon: COMBAT.weaponSlots, bow: COMBAT.bowSlots, shield: COMBAT.shieldSlots },
    arrows: w.arrows,
    hearts: w.hearts,
    health: Math.ceil(w.health),
  };
}

export type PouchStore = {
  get: () => PouchState | null;
  set: (s: PouchState) => void;
  subscribe: (fn: () => void) => () => void;
};

export function createPouchStore(): PouchStore {
  let current: PouchState | null = null;
  let key = '';
  const listeners = new Set<() => void>();
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
  };
}

const none = () => null;

export function usePouch(store: PouchStore): PouchState | null {
  return useSyncExternalStore(store.subscribe, store.get, none);
}
