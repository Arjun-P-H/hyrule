import { useSyncExternalStore } from 'react';
import { DEFAULT_BINDINGS, DEFAULT_KEYMAP, compileKeymap, isDefault, parseBindings, type Bindings, type Keymap } from './keybinds';

/**
 * The key bindings in use, saved in the browser so they outlast the page. Like the telemetry store, a plain
 * external store the React side subscribes to; the saved keys are read on first use, in the browser only
 * (the server renders, and a page hydrates, with the default keys, then swaps in the saved ones).
 */

const STORAGE_KEY = 'hyrule.keybinds';

type KeybindState = { bindings: Bindings; keymap: Keymap };
type KeyStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

const DEFAULT_STATE: KeybindState = { bindings: DEFAULT_BINDINGS, keymap: DEFAULT_KEYMAP };

export function createKeybindStore(storage: () => KeyStorage | null) {
  let state: KeybindState | null = null;
  const listeners = new Set<() => void>();
  return {
    get: (): KeybindState => {
      if (!state) {
        state = DEFAULT_STATE;
        try {
          const raw = storage()?.getItem(STORAGE_KEY);
          if (raw) {
            const bindings = parseBindings(JSON.parse(raw));
            state = { bindings, keymap: compileKeymap(bindings) };
          }
        } catch { /* storage blocked, or not JSON: the default keys */ }
      }
      return state;
    },
    set: (bindings: Bindings) => {
      state = { bindings, keymap: compileKeymap(bindings) };
      try {
        // (the default keys are saved as nothing, so a later change of the defaults reaches those who never moved from them)
        if (isDefault(bindings)) storage()?.removeItem(STORAGE_KEY);
        else storage()?.setItem(STORAGE_KEY, JSON.stringify(bindings));
      } catch { /* storage blocked: the keys last until the page is closed */ }
      for (const l of listeners) l();
    },
    subscribe: (onChange: () => void) => {
      listeners.add(onChange);
      return () => { listeners.delete(onChange); };
    },
  };
}

const store = createKeybindStore(() => {
  try {
    return typeof window === 'undefined' ? null : window.localStorage;
  } catch {
    return null;
  }
});

/** The key bindings and the keymap compiled from them, and the setter that changes and saves them. */
export function useKeybinds() {
  const state = useSyncExternalStore(store.subscribe, store.get, () => DEFAULT_STATE);
  return [state, store.set] as const;
}
