/**
 * Rebindable controls.
 * ---------------------------------------------------------------------------
 * Each control is an action with up to two keys. A key is a KeyboardEvent.code
 * (its place on the keyboard, not the letter printed on it, so W A S D stay a
 * cluster on any layout) except that the two Shift keys and the two Ctrl keys
 * are one key each. A key belongs to at most one action.
 *
 * The settings panel edits the Bindings (keybindStore.ts saves them); the
 * engine reads the Keymap compiled from them: which action a key presses, and,
 * for the actions it polls every frame (HELD), a bit mask of those the keys
 * held make up.
 */

type Group = 'Movement' | 'Look' | 'Combat (third person)' | 'Camera & world';
type RawDef = { id: string; group: Group; label: string; keys: readonly string[]; also?: string };

const DEFS = [
  { id: 'forward', group: 'Movement', label: 'Move forward', keys: ['KeyW', 'ArrowUp'] },
  { id: 'back', group: 'Movement', label: 'Move back', keys: ['KeyS', 'ArrowDown'] },
  { id: 'left', group: 'Movement', label: 'Move left', keys: ['KeyA', 'ArrowLeft'] },
  { id: 'right', group: 'Movement', label: 'Move right', keys: ['KeyD', 'ArrowRight'] },
  { id: 'sprint', group: 'Movement', label: 'Sprint / boost', keys: ['Shift'] },
  { id: 'jump', group: 'Movement', label: 'Jump / glide / rise', keys: ['Space'] },
  { id: 'crouch', group: 'Movement', label: 'Sneak / descend', keys: ['KeyC', 'Control'] },
  { id: 'whistle', group: 'Movement', label: 'Whistle (call your horse)', keys: ['KeyZ'] },
  // turning the camera with keys: no key by default (the mouse looks); the settings' Key Look gives it the arrows
  { id: 'lookUp', group: 'Look', label: 'Look up', keys: [] },
  { id: 'lookDown', group: 'Look', label: 'Look down', keys: [] },
  { id: 'lookLeft', group: 'Look', label: 'Look left', keys: [] },
  { id: 'lookRight', group: 'Look', label: 'Look right', keys: [] },
  { id: 'attack', group: 'Combat (third person)', label: 'Attack', keys: ['KeyJ'], also: 'LMB' },
  { id: 'guard', group: 'Combat (third person)', label: 'Shield / lock on', keys: ['KeyK'], also: 'RMB' },
  { id: 'aim', group: 'Combat (third person)', label: 'Bow (hold)', keys: ['KeyQ'] },
  { id: 'throw', group: 'Combat (third person)', label: 'Throw (hold)', keys: ['KeyG'] },
  { id: 'interact', group: 'Combat (third person)', label: 'Action / surf', keys: ['KeyE'] },
  { id: 'swap', group: 'Combat (third person)', label: 'Next weapon', keys: ['KeyX'] },
  { id: 'inventory', group: 'Combat (third person)', label: 'Inventory', keys: ['KeyI'] },
  { id: 'toggleMode', group: 'Camera & world', label: 'Fly / land', keys: ['KeyF'] },
  { id: 'toggleView', group: 'Camera & world', label: 'First / third person', keys: ['KeyV'] },
  { id: 'respawn', group: 'Camera & world', label: 'Respawn', keys: ['KeyR'] },
  { id: 'map', group: 'Camera & world', label: 'Open / close the map', keys: ['KeyM'] },
  { id: 'fastTime', group: 'Camera & world', label: 'Fast time (hold)', keys: ['KeyT'] },
  { id: 'toggleHud', group: 'Camera & world', label: 'Show / hide HUD', keys: ['KeyH'] },
  // the HUD in three steps: everything, the essentials (the start), nothing (the plus and minus keys, the numpad's too)
  { id: 'hudMore', group: 'Camera & world', label: 'HUD: show more (+)', keys: ['Equal', 'NumpadAdd'] },
  { id: 'hudLess', group: 'Camera & world', label: 'HUD: show less (−)', keys: ['Minus', 'NumpadSubtract'] },
  { id: 'mute', group: 'Camera & world', label: 'Sound on / off', keys: ['KeyN'] },
] as const satisfies readonly RawDef[];

export type ActionId = (typeof DEFS)[number]['id'];

/** Every action in the order the settings list them: its label, its group, its default keys, and a mouse button that does it too. */
export const ACTIONS: readonly (RawDef & { id: ActionId })[] = DEFS;

type Slots = readonly [string | null, string | null];

/** The keys of every action: a primary and an alternate, either of them empty. */
export type Bindings = Readonly<Record<ActionId, Slots>>;

export const DEFAULT_BINDINGS: Bindings = Object.fromEntries(
  ACTIONS.map((a) => [a.id, [a.keys[0] ?? null, a.keys[1] ?? null] as Slots]),
) as Bindings;

/** Whether `b` is the default keys throughout. */
export function isDefault(b: Bindings): boolean {
  return ACTIONS.every((a) => b[a.id][0] === DEFAULT_BINDINGS[a.id][0] && b[a.id][1] === DEFAULT_BINDINGS[a.id][1]);
}

/** A KeyboardEvent.code as the bindings hold it: the left and right Shift (and Ctrl) keys are one. */
export function normalizeCode(code: string): string {
  if (code === 'ShiftLeft' || code === 'ShiftRight') return 'Shift';
  if (code === 'ControlLeft' || code === 'ControlRight') return 'Control';
  return code;
}

// Letters, digits, the numpad, arrows, Space, Shift and Ctrl, punctuation and the navigation block. Left
// out: Esc (cancels the panel), Tab, Enter and Backspace (the page's), F-keys (the browser's), Alt, the
// Windows / Command key and Caps Lock (they swallow other keys' events), and keys that do not hold.
const BINDABLE =
  /^(Key[A-Z]|Digit[0-9]|Numpad[A-Za-z0-9]+|Arrow(Up|Down|Left|Right)|Space|Shift|Control|Backquote|Minus|Equal|Bracket(Left|Right)|Backslash|IntlBackslash|Semicolon|Quote|Comma|Period|Slash|Insert|Home|End|Page(Up|Down))$/;

/** Whether a normalized code may be bound. */
export function isBindable(code: string): boolean {
  return BINDABLE.test(code);
}

const LABELS: Readonly<Record<string, string>> = {
  ArrowUp: '↑', ArrowDown: '↓', ArrowLeft: '←', ArrowRight: '→', Control: 'Ctrl',
  Backquote: '`', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Backslash: '\\', IntlBackslash: '\\',
  Semicolon: ';', Quote: "'", Comma: ',', Period: '.', Slash: '/', PageUp: 'PgUp', PageDown: 'PgDn', Insert: 'Ins',
};
const NUMPAD: Readonly<Record<string, string>> = { Add: '+', Subtract: '-', Multiply: '*', Divide: '/', Decimal: '.', Equal: '=' };

/** A key as the panel and the hints show it: W, ↑, Space, Ctrl, Num 5. */
export function keyLabel(code: string): string {
  const named = LABELS[code];
  if (named) return named;
  if (/^Key[A-Z]$/.test(code)) return code.slice(3);
  if (/^Digit[0-9]$/.test(code)) return code.slice(5);
  if (code.startsWith('Numpad')) return `Num ${NUMPAD[code.slice(6)] ?? code.slice(6)}`;
  return code;
}

/** The label of the first key bound to `id`; '' when it has none. */
export function keyHint(b: Bindings, id: ActionId): string {
  const code = b[id][0] ?? b[id][1];
  return code ? keyLabel(code) : '';
}

/** The key as a note to put after a control's name: " (F)"; nothing when it has none. */
export function keyNote(b: Bindings, id: ActionId): string {
  const key = keyHint(b, id);
  return key ? ` (${key})` : '';
}

/** Four keys of a direction cluster as the banner shows them (up, left, down, right): "WASD"; '' when none has a key. */
function cluster(b: Bindings, ids: readonly ActionId[]): string {
  const hints = ids.map((id) => keyHint(b, id));
  if (hints.every((h) => !h)) return '';
  const labels = hints.map((h) => h || '–');
  return labels.every((l) => l.length === 1) ? labels.join('') : labels.join('/');
}

/** The movement keys, forward, left, back, right: "WASD" by default. */
export const moveKeys = (b: Bindings) => cluster(b, ['forward', 'left', 'back', 'right']);

/** The keys that turn the camera, up, left, down, right: "↑←↓→" with Key Look on its arrows; '' with none. */
export const lookKeys = (b: Bindings) => cluster(b, ['lookUp', 'lookLeft', 'lookDown', 'lookRight']);

const withSlot = (s: Slots, slot: number, code: string | null): Slots => (slot === 0 ? [code, s[1]] : [s[0], code]);

/**
 * Bind a (normalized, bindable) key to a slot of an action. A key lives in one place: whichever action
 * held it loses it (`from` names it, unless it was the same action's other slot).
 */
export function bindKey(b: Bindings, id: ActionId, slot: number, code: string): { bindings: Bindings; from: ActionId | null } {
  const next: Record<string, Slots> = { ...b };
  let from: ActionId | null = null;
  for (const a of ACTIONS) {
    const s = next[a.id];
    if (s[0] !== code && s[1] !== code) continue;
    if (a.id !== id) from = a.id;
    next[a.id] = [s[0] === code ? null : s[0], s[1] === code ? null : s[1]];
  }
  next[id] = withSlot(next[id], slot, code);
  return { bindings: next as Bindings, from };
}

/** Leave a slot of an action empty. */
export function clearSlot(b: Bindings, id: ActionId, slot: number): Bindings {
  return { ...b, [id]: withSlot(b[id], slot, null) };
}

const LOOKS = ['lookUp', 'lookDown', 'lookLeft', 'lookRight'] as const satisfies readonly ActionId[];
const LOOK_ARROWS = { lookUp: 'ArrowUp', lookDown: 'ArrowDown', lookLeft: 'ArrowLeft', lookRight: 'ArrowRight' } as const;

/** Whether any key turns the camera (Key Look is on). */
export function looksWithKeys(b: Bindings): boolean {
  return LOOKS.some((id) => b[id][0] !== null || b[id][1] !== null);
}

/** Key Look on: the four arrow keys turn the camera. They are taken from whatever held them: by default the moves' alternates, so WASD still moves. */
export function lookWithArrows(b: Bindings): Bindings {
  let next = clearLooks(b);
  for (const id of LOOKS) next = bindKey(next, id, 0, LOOK_ARROWS[id]).bindings;
  return next;
}

/** Key Look off: no key turns the camera, and the arrow keys go back to the moves they are the default alternates of, where free. */
export function lookWithoutKeys(b: Bindings): Bindings {
  let next = clearLooks(b);
  for (const id of ['forward', 'back', 'left', 'right'] as const) {
    const arrow = DEFAULT_BINDINGS[id][1];
    if (arrow && next[id][1] === null && !compileKeymap(next).has(arrow)) next = bindKey(next, id, 1, arrow).bindings;
  }
  return next;
}

function clearLooks(b: Bindings): Bindings {
  let next = b;
  for (const id of LOOKS) next = clearSlot(clearSlot(next, id, 0), id, 1);
  return next;
}

/**
 * Bindings out of whatever was saved: anything unusable is dropped, a key kept by two actions stays with the
 * first, an action never saved (added since) takes its default keys if they are still free.
 */
export function parseBindings(raw: unknown): Bindings {
  const src = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
  const used = new Set<string>();
  const take = (v: unknown): string | null => {
    if (typeof v !== 'string') return null;
    const code = normalizeCode(v);
    if (!isBindable(code) || used.has(code)) return null;
    used.add(code);
    return code;
  };
  // The sound key moved from M to N when M became the map's: bindings saved before that, with the sound
  // still on M (never moved), follow the new default.
  const mute = src.mute;
  const muteOnOldKey = src.map === undefined && Array.isArray(mute) && mute[0] === 'KeyM' && (mute[1] ?? null) === null;
  const out: Record<string, Slots> = {};
  for (const a of ACTIONS) {
    const v = src[a.id];
    if (Array.isArray(v) && !(a.id === 'mute' && muteOnOldKey)) out[a.id] = [take(v[0]), take(v[1])];
  }
  for (const a of ACTIONS) out[a.id] ??= [take(a.keys[0]), take(a.keys[1])];
  return out as Bindings;
}

/** Which action each key presses. */
export type Keymap = ReadonlyMap<string, ActionId>;

export function compileKeymap(b: Bindings): Keymap {
  const map = new Map<string, ActionId>();
  for (const a of ACTIONS) for (const code of b[a.id]) if (code) map.set(code, a.id);
  return map;
}

export const DEFAULT_KEYMAP: Keymap = compileKeymap(DEFAULT_BINDINGS);

const HELD_ACTIONS = [
  'forward', 'back', 'left', 'right', 'sprint', 'jump', 'crouch', 'lookUp', 'lookDown', 'lookLeft', 'lookRight',
  'attack', 'guard', 'aim', 'throw', 'interact', 'swap', 'whistle',
] as const satisfies readonly ActionId[];

/** The actions the engine polls every frame, one bit each. */
export const HELD = Object.fromEntries(HELD_ACTIONS.map((id, i) => [id, 1 << i])) as Record<(typeof HELD_ACTIONS)[number], number>;

/** The HELD actions the keys held (normalized codes) make up. */
export function heldMask(keymap: Keymap, held: ReadonlySet<string>): number {
  let mask = 0;
  for (const code of held) {
    const id = keymap.get(code);
    if (id !== undefined) mask |= (HELD as Partial<Record<ActionId, number>>)[id] ?? 0;
  }
  return mask;
}
