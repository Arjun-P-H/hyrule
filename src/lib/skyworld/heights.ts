/**
 * Which heights the sky islands float at (Engine Settings → Sky islands): `low`, each one pulled down
 * SKY_REALM.lower as far as the ground under it lets it (archipelago.ts floorUnder), or `original`, the first
 * heights, every underside over SKY_REALM.floor. Read once when the game starts (islands are laid out once
 * and kept); switching saves the choice and reloads the page.
 */

import { SKY_REALM as S } from '../constants';

export type IslandHeights = 'low' | 'original';

const KEY = 'hyrule.islandHeights';

function read(): IslandHeights {
  try {
    return typeof window !== 'undefined' && window.localStorage.getItem(KEY) === 'original' ? 'original' : 'low';
  } catch {
    return 'low'; // (storage blocked: the default)
  }
}

/** The heights in use this run. */
export const ISLAND_HEIGHTS: IslandHeights = read();

/** How far every island is pulled down from its class's (and lone isles') height (m). */
export const ISLAND_LOWER = ISLAND_HEIGHTS === 'low' ? S.lower : 0;

/** Use `h` from now on: saved, then the page reloads to lay the islands out again. */
export function setIslandHeights(h: IslandHeights): void {
  try { window.localStorage.setItem(KEY, h); } catch { /* (storage blocked: nothing to keep) */ }
  window.location.reload();
}
