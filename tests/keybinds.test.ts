import { describe, it, expect } from 'bun:test';
import {
  ACTIONS,
  DEFAULT_BINDINGS,
  DEFAULT_KEYMAP,
  HELD,
  bindKey,
  clearSlot,
  compileKeymap,
  heldMask,
  isBindable,
  isDefault,
  keyHint,
  keyLabel,
  keyNote,
  lookKeys,
  looksWithKeys,
  lookWithArrows,
  lookWithoutKeys,
  moveKeys,
  normalizeCode,
  parseBindings,
  type ActionId,
  type Bindings,
} from '../src/lib/engine/keybinds';
import { createKeybindStore } from '../src/lib/engine/keybindStore';

/** The keys the game had before they could be rebound (engineCore.ts's old key table), by what they do. */
const ORIGINAL: Record<ActionId, readonly string[]> = {
  forward: ['KeyW', 'ArrowUp'],
  back: ['KeyS', 'ArrowDown'],
  left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'],
  sprint: ['Shift'],
  jump: ['Space'],
  crouch: ['KeyC', 'Control'],
  whistle: ['KeyZ'], // (added with the horses)
  lookUp: [], // (the camera turned with the mouse alone)
  lookDown: [],
  lookLeft: [],
  lookRight: [],
  attack: ['KeyJ'],
  guard: ['KeyK'],
  aim: ['KeyQ'],
  throw: ['KeyG'],
  interact: ['KeyE'],
  swap: ['KeyX'],
  inventory: ['KeyI'], // (new with the inventory screen)
  toggleMode: ['KeyF'],
  toggleView: ['KeyV'],
  respawn: ['KeyR'],
  fastTime: ['KeyT'],
  toggleHud: ['KeyH'],
  hudMore: ['Equal', 'NumpadAdd'], // (new: the HUD in three steps)
  hudLess: ['Minus', 'NumpadSubtract'],
  map: ['KeyM'], // (new with the map: the sound moved over to N)
  mute: ['KeyN'],
};

const ids = ACTIONS.map((a) => a.id);
const codesOf = (b: Bindings) => ids.flatMap((id) => b[id].filter((c): c is string => c !== null));

describe('rebindable controls: the defaults', () => {
  it('are the keys the game had before they could be rebound', () => {
    expect(Object.keys(ORIGINAL).sort()).toEqual([...ids].sort());
    for (const id of ids) {
      expect(DEFAULT_BINDINGS[id].filter((c) => c !== null)).toEqual([...ORIGINAL[id]]);
    }
    expect(isDefault(DEFAULT_BINDINGS)).toBe(true);
  });

  it('give every key to one action, and every key can be bound', () => {
    const codes = codesOf(DEFAULT_BINDINGS);
    expect(new Set(codes).size).toBe(codes.length);
    for (const c of codes) expect(isBindable(c)).toBe(true);
    expect(DEFAULT_KEYMAP.size).toBe(codes.length);
  });

  it('make the physical keys of the old table press the same actions', () => {
    // KeyboardEvent.code → action, as engineCore's keyCodeToBit and its one-shot keys had it
    const old: Record<string, ActionId> = {
      KeyW: 'forward', ArrowUp: 'forward', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left',
      KeyD: 'right', ArrowRight: 'right', ShiftLeft: 'sprint', ShiftRight: 'sprint', Space: 'jump', KeyC: 'crouch',
      ControlLeft: 'crouch', ControlRight: 'crouch', KeyQ: 'aim', KeyE: 'interact', KeyG: 'throw', KeyX: 'swap',
      KeyJ: 'attack', KeyK: 'guard', KeyF: 'toggleMode', KeyV: 'toggleView', KeyR: 'respawn', KeyT: 'fastTime',
      KeyH: 'toggleHud', KeyM: 'map', KeyN: 'mute', Equal: 'hudMore', NumpadAdd: 'hudMore', Minus: 'hudLess', NumpadSubtract: 'hudLess',
    };
    for (const [code, id] of Object.entries(old)) expect(DEFAULT_KEYMAP.get(normalizeCode(code))).toBe(id);
  });
});

describe('rebindable controls: keys', () => {
  it('count the left and right Shift and Ctrl as one key', () => {
    expect(normalizeCode('ShiftLeft')).toBe('Shift');
    expect(normalizeCode('ShiftRight')).toBe('Shift');
    expect(normalizeCode('ControlLeft')).toBe('Control');
    expect(normalizeCode('ControlRight')).toBe('Control');
    expect(normalizeCode('KeyW')).toBe('KeyW');
    expect(normalizeCode('AltLeft')).toBe('AltLeft');
  });

  it('bind letters, digits, the numpad, arrows, Space, Shift, Ctrl and punctuation', () => {
    for (const c of ['KeyA', 'KeyZ', 'Digit0', 'Digit9', 'Numpad5', 'NumpadAdd', 'NumpadEnter', 'ArrowLeft', 'Space', 'Shift', 'Control',
      'Backquote', 'Minus', 'Equal', 'BracketLeft', 'Backslash', 'Semicolon', 'Quote', 'Comma', 'Period', 'Slash', 'PageUp', 'Home']) {
      expect(isBindable(c)).toBe(true);
    }
  });

  it('leave out the keys that cancel, navigate, belong to the browser or swallow other keys', () => {
    for (const c of ['Escape', 'Tab', 'Enter', 'Backspace', 'Delete', 'F1', 'F5', 'F11', 'F12', 'AltLeft', 'AltRight', 'MetaLeft', 'MetaRight',
      'CapsLock', 'ContextMenu', 'NumLock', 'PrintScreen', 'ShiftLeft', 'ControlRight', 'KeyWW', 'key', '', 'Lang1']) {
      expect(isBindable(c)).toBe(false);
    }
  });

  it('are labelled the way a player reads them', () => {
    expect(keyLabel('KeyW')).toBe('W');
    expect(keyLabel('Digit7')).toBe('7');
    expect(keyLabel('ArrowUp')).toBe('↑');
    expect(keyLabel('ArrowRight')).toBe('→');
    expect(keyLabel('Space')).toBe('Space');
    expect(keyLabel('Shift')).toBe('Shift');
    expect(keyLabel('Control')).toBe('Ctrl');
    expect(keyLabel('Backquote')).toBe('`');
    expect(keyLabel('Numpad5')).toBe('Num 5');
    expect(keyLabel('NumpadAdd')).toBe('Num +');
    expect(keyLabel('PageDown')).toBe('PgDn');
    expect(keyLabel('SomethingNew')).toBe('SomethingNew');
  });
});

describe('rebindable controls: changing a key', () => {
  it('puts the key in the slot and leaves the bindings it came from alone', () => {
    const { bindings, from } = bindKey(DEFAULT_BINDINGS, 'attack', 1, 'KeyU');
    expect(bindings.attack).toEqual(['KeyJ', 'KeyU']);
    expect(from).toBeNull();
    expect(DEFAULT_BINDINGS.attack).toEqual(['KeyJ', null]);
    expect(isDefault(bindings)).toBe(false);
  });

  it('replaces the key a slot had', () => {
    const { bindings } = bindKey(DEFAULT_BINDINGS, 'forward', 0, 'KeyI');
    expect(bindings.forward).toEqual(['KeyI', 'ArrowUp']);
    expect(compileKeymap(bindings).get('KeyW')).toBeUndefined();
    expect(compileKeymap(bindings).get('KeyI')).toBe('forward');
  });

  it('takes a key from the action that had it, and names that action', () => {
    const { bindings, from } = bindKey(DEFAULT_BINDINGS, 'attack', 0, 'KeyR');
    expect(from).toBe('respawn');
    expect(bindings.attack).toEqual(['KeyR', null]);
    expect(bindings.respawn).toEqual([null, null]);
    expect(compileKeymap(bindings).get('KeyR')).toBe('attack');
    expect(compileKeymap(bindings).get('KeyJ')).toBeUndefined();
  });

  it("moves a key between an action's own slots without naming anyone", () => {
    const { bindings, from } = bindKey(DEFAULT_BINDINGS, 'forward', 0, 'ArrowUp');
    expect(from).toBeNull();
    expect(bindings.forward).toEqual(['ArrowUp', null]);
  });

  it('is the same again when the slot already holds the key', () => {
    const { bindings, from } = bindKey(DEFAULT_BINDINGS, 'forward', 0, 'KeyW');
    expect(from).toBeNull();
    expect(isDefault(bindings)).toBe(true);
  });

  it('can leave a slot empty, and a control with no keys at all', () => {
    const one = clearSlot(DEFAULT_BINDINGS, 'forward', 1);
    expect(one.forward).toEqual(['KeyW', null]);
    const none = clearSlot(clearSlot(DEFAULT_BINDINGS, 'forward', 0), 'forward', 1);
    expect(none.forward).toEqual([null, null]);
    expect(compileKeymap(none).get('KeyW')).toBeUndefined();
    expect(DEFAULT_BINDINGS.forward).toEqual(['KeyW', 'ArrowUp']);
  });

  it('keeps every key in one place whatever is bound and cleared', () => {
    // a seeded walk through 3000 changes
    let seed = 12345;
    const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const pool = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyJ', 'KeyR', 'Space', 'Shift', 'Control', 'ArrowUp', 'Digit1', 'Numpad4', 'Slash', 'KeyZ'];
    let b: Bindings = DEFAULT_BINDINGS;
    for (let i = 0; i < 3000; i++) {
      const id = ids[Math.floor(rnd() * ids.length)];
      const slot = rnd() < 0.5 ? 0 : 1;
      b = rnd() < 0.15 ? clearSlot(b, id, slot) : bindKey(b, id, slot, pool[Math.floor(rnd() * pool.length)]).bindings;
      const codes = codesOf(b);
      expect(new Set(codes).size).toBe(codes.length);
      expect(compileKeymap(b).size).toBe(codes.length);
    }
  });
});

describe('rebindable controls: the saved keys', () => {
  it('come back as they were saved', () => {
    let b = bindKey(DEFAULT_BINDINGS, 'attack', 1, 'KeyU').bindings;
    b = bindKey(b, 'jump', 0, 'KeyZ').bindings;
    b = clearSlot(b, 'respawn', 0);
    expect(parseBindings(JSON.parse(JSON.stringify(b)))).toEqual(b);
  });

  it('are the defaults when nothing usable was saved', () => {
    for (const junk of [undefined, null, 7, 'keys', true, [], [1, 2], {}, { forward: 'KeyW' }, { forward: 3 }]) {
      expect(parseBindings(junk)).toEqual(DEFAULT_BINDINGS);
    }
  });

  it('drop a key that cannot be bound and keep the rest', () => {
    const b = parseBindings({ forward: ['Escape', 'KeyI'], back: ['F5', 42] });
    expect(b.forward).toEqual([null, 'KeyI']);
    expect(b.back).toEqual([null, null]);
    expect(b.left).toEqual(DEFAULT_BINDINGS.left); // (never saved: its defaults)
  });

  it('keep a key with the first action that holds it, and a control that was left without keys', () => {
    const b = parseBindings({ forward: ['KeyX', null], back: ['KeyX', 'KeyX'], swap: [null, null] });
    expect(b.forward).toEqual(['KeyX', null]);
    expect(b.back).toEqual([null, null]);
    expect(b.swap).toEqual([null, null]); // (X was taken, and its own slots were saved empty)
    const codes = codesOf(b);
    expect(new Set(codes).size).toBe(codes.length);
  });

  it('give a control added since its default keys only where they are still free', () => {
    // saved before "map" existed, with M given to another control
    const saved = Object.fromEntries(ids.filter((id) => id !== 'map' && id !== 'mute').map((id) => [id, DEFAULT_BINDINGS[id]]));
    const b = parseBindings({ ...saved, toggleHud: ['KeyM', null] });
    expect(b.map).toEqual([null, null]);
    expect(b.toggleHud).toEqual(['KeyM', null]);
    expect(parseBindings(saved).map).toEqual(['KeyM', null]);
  });

  it('move a sound key still on M (the old default) to N, for the map to have M', () => {
    const saved = Object.fromEntries(ids.filter((id) => id !== 'map' && id !== 'mute').map((id) => [id, DEFAULT_BINDINGS[id]]));
    const b = parseBindings({ ...saved, mute: ['KeyM', null] });
    expect(b.map).toEqual(['KeyM', null]);
    expect(b.mute).toEqual(['KeyN', null]);
    // one the player moved themselves stays where they put it
    expect(parseBindings({ ...saved, mute: ['KeyU', null] }).mute).toEqual(['KeyU', null]);
    expect(parseBindings({ ...saved, mute: ['KeyU', null] }).map).toEqual(['KeyM', null]);
  });

  it('count the left and right Shift as the one key', () => {
    expect(parseBindings({ sprint: ['ShiftRight', null] }).sprint).toEqual(['Shift', null]);
  });
});

describe('rebindable controls: what the engine reads', () => {
  it('turns the keys held into the actions held', () => {
    const w = heldMask(DEFAULT_KEYMAP, new Set(['KeyW']));
    expect(w).toBe(HELD.forward);
    expect(heldMask(DEFAULT_KEYMAP, new Set(['KeyW', 'Shift', 'Space']))).toBe(HELD.forward | HELD.sprint | HELD.jump);
    expect(heldMask(DEFAULT_KEYMAP, new Set())).toBe(0);
  });

  it('keeps an action held while either of its keys is', () => {
    expect(heldMask(DEFAULT_KEYMAP, new Set(['KeyW', 'ArrowUp']))).toBe(HELD.forward);
    expect(heldMask(DEFAULT_KEYMAP, new Set(['ArrowUp']))).toBe(HELD.forward); // (W let go)
    expect(heldMask(DEFAULT_KEYMAP, new Set(['KeyC']))).toBe(HELD.crouch);
    expect(heldMask(DEFAULT_KEYMAP, new Set(['Control']))).toBe(HELD.crouch);
  });

  it('ignores keys that press nothing and the one-shot keys', () => {
    expect(heldMask(DEFAULT_KEYMAP, new Set(['KeyL', 'Digit5', 'Escape']))).toBe(0);
    expect(heldMask(DEFAULT_KEYMAP, new Set(['KeyF', 'KeyV', 'KeyR', 'KeyT', 'KeyH', 'KeyM']))).toBe(0);
  });

  it('gives each polled action a bit of its own', () => {
    const bits = Object.values(HELD);
    for (const bit of bits) expect(bit & (bit - 1)).toBe(0); // a power of two
    expect(new Set(bits).size).toBe(bits.length);
    expect(Math.max(...bits)).toBeLessThan(2 ** 31);
  });

  it('follows a rebinding at once, for the keys already held', () => {
    const held = new Set(['KeyI']);
    expect(heldMask(DEFAULT_KEYMAP, held)).toBe(0);
    const km = compileKeymap(bindKey(DEFAULT_BINDINGS, 'forward', 0, 'KeyI').bindings);
    expect(heldMask(km, held)).toBe(HELD.forward);
    // W no longer walks, the arrow still does
    expect(heldMask(km, new Set(['KeyW']))).toBe(0);
    expect(heldMask(km, new Set(['ArrowUp']))).toBe(HELD.forward);
  });

  it('lets a polled action be moved onto a one-shot key and back', () => {
    // F (fly / land) given to the attack: it is held, no longer a toggle
    const { bindings } = bindKey(DEFAULT_BINDINGS, 'attack', 0, 'KeyF');
    const km = compileKeymap(bindings);
    expect(km.get('KeyF')).toBe('attack');
    expect(heldMask(km, new Set(['KeyF']))).toBe(HELD.attack);
    expect([...km.values()]).not.toContain('toggleMode');
  });
});

describe('rebindable controls: the hints', () => {
  it('name the key of a control, with a note for after its name', () => {
    expect(keyHint(DEFAULT_BINDINGS, 'toggleMode')).toBe('F');
    expect(keyNote(DEFAULT_BINDINGS, 'toggleMode')).toBe(' (F)');
    expect(keyHint(DEFAULT_BINDINGS, 'jump')).toBe('Space');
    const moved = bindKey(DEFAULT_BINDINGS, 'toggleMode', 0, 'KeyP').bindings;
    expect(keyNote(moved, 'toggleMode')).toBe(' (P)');
  });

  it('fall back on the alternate key, and say nothing for a control with none', () => {
    const b = clearSlot(DEFAULT_BINDINGS, 'forward', 0);
    expect(keyHint(b, 'forward')).toBe('↑');
    const none = clearSlot(DEFAULT_BINDINGS, 'respawn', 0);
    expect(keyHint(none, 'respawn')).toBe('');
    expect(keyNote(none, 'respawn')).toBe('');
  });

  it('show the movement keys as WASD, or as whatever they were set to', () => {
    expect(moveKeys(DEFAULT_BINDINGS)).toBe('WASD');
    let b = DEFAULT_BINDINGS;
    for (const [id, code] of [['forward', 'KeyI'], ['left', 'KeyJ'], ['back', 'KeyK'], ['right', 'KeyL']] as const) b = bindKey(b, id, 0, code).bindings;
    expect(moveKeys(b)).toBe('IJKL');
    let arrows = DEFAULT_BINDINGS;
    for (const id of ['forward', 'left', 'back', 'right'] as const) arrows = clearSlot(arrows, id, 0);
    expect(moveKeys(arrows)).toBe('↑←↓→');
    expect(moveKeys(bindKey(arrows, 'forward', 0, 'Numpad8').bindings)).toBe('Num 8/←/↓/→');
    for (const id of ['forward', 'left', 'back', 'right'] as const) arrows = clearSlot(arrows, id, 1);
    expect(moveKeys(arrows)).toBe('');
  });
});

describe('rebindable controls: looking with keys', () => {
  const LOOKS = ['lookUp', 'lookDown', 'lookLeft', 'lookRight'] as const;

  it('is off by default: no key turns the camera, and the arrows move', () => {
    expect(looksWithKeys(DEFAULT_BINDINGS)).toBe(false);
    for (const id of LOOKS) expect(DEFAULT_BINDINGS[id]).toEqual([null, null]);
    expect(DEFAULT_KEYMAP.get('ArrowUp')).toBe('forward');
    expect(lookKeys(DEFAULT_BINDINGS)).toBe('');
  });

  it('turns on with the arrow keys, which leave the moves they were the alternates of', () => {
    const on = lookWithArrows(DEFAULT_BINDINGS);
    expect(looksWithKeys(on)).toBe(true);
    expect(on.lookUp).toEqual(['ArrowUp', null]);
    expect(on.lookDown).toEqual(['ArrowDown', null]);
    expect(on.lookLeft).toEqual(['ArrowLeft', null]);
    expect(on.lookRight).toEqual(['ArrowRight', null]);
    expect(on.forward).toEqual(['KeyW', null]);
    expect(on.right).toEqual(['KeyD', null]);
    const km = compileKeymap(on);
    expect(km.get('ArrowLeft')).toBe('lookLeft');
    expect(km.get('KeyA')).toBe('left'); // (WASD still moves)
    const codes = codesOf(on);
    expect(new Set(codes).size).toBe(codes.length);
    expect(lookKeys(on)).toBe('↑←↓→');
    expect(lookWithArrows(on)).toEqual(on);
    expect(DEFAULT_BINDINGS.forward).toEqual(['KeyW', 'ArrowUp']); // (the input is left as it was)
  });

  it('turns off again to just what it was', () => {
    const off = lookWithoutKeys(lookWithArrows(DEFAULT_BINDINGS));
    expect(off).toEqual(DEFAULT_BINDINGS);
    expect(isDefault(off)).toBe(true);
    expect(lookWithoutKeys(DEFAULT_BINDINGS)).toEqual(DEFAULT_BINDINGS);
  });

  it('turns off whatever keys were chosen, and gives only the arrows back to the moves', () => {
    let b = lookWithArrows(DEFAULT_BINDINGS);
    for (const [id, code] of [['lookUp', 'KeyI'], ['lookLeft', 'KeyJ'], ['lookDown', 'KeyK'], ['lookRight', 'KeyL']] as const) b = bindKey(b, id, 0, code).bindings;
    expect(lookKeys(b)).toBe('IJKL');
    expect(b.attack).toEqual([null, null]); // (J and K were taken from attack and shield)
    b = clearSlot(b, 'crouch', 1); // Ctrl given up on purpose: not Key Look's to bring back
    const off = lookWithoutKeys(b);
    expect(looksWithKeys(off)).toBe(false);
    expect(off.forward).toEqual(['KeyW', 'ArrowUp']);
    expect(off.back).toEqual(['KeyS', 'ArrowDown']);
    expect(off.left).toEqual(['KeyA', 'ArrowLeft']);
    expect(off.right).toEqual(['KeyD', 'ArrowRight']);
    expect(off.crouch).toEqual(['KeyC', null]);
    expect(off.attack).toEqual([null, null]);
    const codes = codesOf(off);
    expect(new Set(codes).size).toBe(codes.length);
  });

  it('does not take an arrow back from the control it was given to', () => {
    let b = lookWithArrows(DEFAULT_BINDINGS);
    b = bindKey(b, 'respawn', 1, 'ArrowUp').bindings; // the arrow moved on to respawn (Look up lost it)
    expect(b.lookUp).toEqual([null, null]);
    const off = lookWithoutKeys(b);
    expect(off.respawn).toEqual(['KeyR', 'ArrowUp']);
    expect(off.forward).toEqual(['KeyW', null]);
    expect(off.back).toEqual(['KeyS', 'ArrowDown']);
  });

  it('does not put an arrow on a move whose alternate was set to another key', () => {
    let b = lookWithArrows(DEFAULT_BINDINGS);
    b = bindKey(b, 'forward', 1, 'KeyI').bindings;
    expect(lookWithoutKeys(b).forward).toEqual(['KeyW', 'KeyI']);
  });

  it('shows the look keys in the order up, left, down, right', () => {
    let b = DEFAULT_BINDINGS;
    for (const [id, code] of [['lookUp', 'KeyI'], ['lookLeft', 'KeyJ'], ['lookDown', 'KeyK'], ['lookRight', 'KeyL']] as const) b = bindKey(b, id, 0, code).bindings;
    expect(lookKeys(b)).toBe('IJKL');
    expect(looksWithKeys(clearSlot(clearSlot(clearSlot(clearSlot(lookWithArrows(DEFAULT_BINDINGS), 'lookUp', 0), 'lookDown', 0), 'lookLeft', 0), 'lookRight', 0))).toBe(false);
  });

  it('is on as soon as a single key turns the camera, and is kept with the saved keys', () => {
    const one = bindKey(DEFAULT_BINDINGS, 'lookLeft', 1, 'Comma').bindings;
    expect(looksWithKeys(one)).toBe(true);
    expect(lookKeys(one)).toBe('–,––'); // (up, left, down, right: only left has a key)
    expect(parseBindings(JSON.parse(JSON.stringify(lookWithArrows(DEFAULT_BINDINGS))))).toEqual(lookWithArrows(DEFAULT_BINDINGS));
  });

  it('is read from a save made before the look keys existed as off', () => {
    const before = Object.fromEntries(ids.filter((id) => !id.startsWith('look')).map((id) => [id, DEFAULT_BINDINGS[id]]));
    const b = parseBindings(before);
    expect(looksWithKeys(b)).toBe(false);
    expect(isDefault(b)).toBe(true);
  });

  it('makes the keys held into look bits for the engine', () => {
    const on = compileKeymap(lookWithArrows(DEFAULT_BINDINGS));
    expect(heldMask(on, new Set(['ArrowLeft']))).toBe(HELD.lookLeft);
    expect(heldMask(on, new Set(['ArrowUp', 'ArrowRight', 'KeyW']))).toBe(HELD.lookUp | HELD.lookRight | HELD.forward);
    expect(heldMask(DEFAULT_KEYMAP, new Set(['ArrowLeft']))).toBe(HELD.left); // (off: it walks)
  });
});

describe('rebindable controls: the saved bindings store', () => {
  const memory = () => {
    const data = new Map<string, string>();
    return {
      data,
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
      removeItem: (k: string) => void data.delete(k),
    };
  };

  it('starts with the default keys, and hands out the same state until it changes', () => {
    const store = createKeybindStore(() => memory());
    const first = store.get();
    expect(first.bindings).toEqual(DEFAULT_BINDINGS);
    expect(first.keymap).toBe(DEFAULT_KEYMAP);
    expect(store.get()).toBe(first); // (useSyncExternalStore needs a stable snapshot)
  });

  it('saves a change, tells its subscribers, and takes the save back up', () => {
    const storage = memory();
    const store = createKeybindStore(() => storage);
    let calls = 0;
    const off = store.subscribe(() => calls++);
    const next = bindKey(DEFAULT_BINDINGS, 'attack', 1, 'KeyU').bindings;
    store.set(next);
    expect(calls).toBe(1);
    expect(store.get().bindings).toEqual(next);
    expect(store.get().keymap.get('KeyU')).toBe('attack');
    expect(JSON.parse(storage.data.get('hyrule.keybinds') ?? 'null')).toEqual(next);
    off();
    store.set(DEFAULT_BINDINGS);
    expect(calls).toBe(1);

    store.set(next);
    const reopened = createKeybindStore(() => storage); // another page load
    expect(reopened.get().bindings).toEqual(next);
  });

  it('saves the default keys as nothing', () => {
    const storage = memory();
    const store = createKeybindStore(() => storage);
    store.set(bindKey(DEFAULT_BINDINGS, 'attack', 1, 'KeyU').bindings);
    expect(storage.data.has('hyrule.keybinds')).toBe(true);
    store.set(DEFAULT_BINDINGS);
    expect(storage.data.has('hyrule.keybinds')).toBe(false);
    expect(store.get().bindings).toEqual(DEFAULT_BINDINGS);
  });

  it('survives a save that is not JSON, or two controls saved with one key', () => {
    const storage = memory();
    storage.data.set('hyrule.keybinds', '{not json');
    expect(createKeybindStore(() => storage).get().bindings).toEqual(DEFAULT_BINDINGS);
    storage.data.set('hyrule.keybinds', JSON.stringify({ attack: ['KeyR', null], respawn: ['KeyR', null] }));
    const b = createKeybindStore(() => storage).get().bindings;
    expect(codesOf(b).filter((c) => c === 'KeyR')).toHaveLength(1);
    expect(b.attack).toEqual(['KeyR', null]);
  });

  it('works, for the page that is open, where the browser gives it no storage', () => {
    const noStorage = createKeybindStore(() => null);
    const next = bindKey(DEFAULT_BINDINGS, 'attack', 1, 'KeyU').bindings;
    noStorage.set(next);
    expect(noStorage.get().bindings).toEqual(next);

    const throwing = createKeybindStore(() => ({
      getItem: () => { throw new Error('blocked'); },
      setItem: () => { throw new Error('blocked'); },
      removeItem: () => { throw new Error('blocked'); },
    }));
    expect(throwing.get().bindings).toEqual(DEFAULT_BINDINGS);
    throwing.set(next);
    expect(throwing.get().bindings).toEqual(next);
  });
});
