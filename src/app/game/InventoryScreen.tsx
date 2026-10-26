'use client';

import React, { useEffect, useEffectEvent, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { usePouch, type PouchCommand, type PouchState, type PouchStore } from '@/lib/combat/pouch';
import { keyHint, normalizeCode, type Bindings } from '@/lib/engine/keybinds';
import { HeartRow } from './CombatHud';
import { ItemIcon, type IconCat } from './ItemIcon';

/**
 * The inventory, after Breath of the Wild's pause menu: tabs for the weapons, bows, shields and food
 * carried, a grid of what is in each, and the selected item's details with what can be done with it
 * (equip, put away, drop, eat). Opened with the inventory key; behind it the world stands still
 * (alone in the world) and the hero shows what it wears. Mouse or keys: arrows / WASD move, Q / E (or
 * Tab) change tabs, Enter equips or eats, X drops, I or Esc closes. The pouch comes from the engine
 * (combat/pouch.ts) and every action goes back to it as a command.
 */

type Tab = 'weapon' | 'bow' | 'shield' | 'food';

const TABS: readonly { id: Tab; label: string }[] = [
  { id: 'weapon', label: 'Weapons' },
  { id: 'bow', label: 'Bows' },
  { id: 'shield', label: 'Shields' },
  { id: 'food', label: 'Food' },
];
const COLS = 5;
const CELLS = COLS * 4;

/** What a cell and the details panel show of one thing. */
type Entry = {
  cat: IconCat;
  kind: number;
  index: number;      // its place in the pouch's list (food: its FOODS index)
  name: string;
  info: string;
  stat: string;
  wear: number;       // blows or arrows left; −1: never wears out
  max: number;
  equipped: boolean;
  keeps: boolean;     // cannot be dropped
  count: number;
};

/** Quarters of a heart in words: 2 → "½ heart", 8 → "2 hearts". */
function healText(q: number): string {
  const whole = Math.floor(q / 4), part = ['', '¼', '½', '¾'][q % 4];
  return whole === 0 ? `${part} heart` : `${whole}${part} heart${whole > 1 || part ? 's' : ''}`;
}

function entriesOf(s: PouchState, tab: Tab): Entry[] {
  if (tab === 'food') {
    return s.food.map((f) => ({
      cat: 'food', kind: f.kind, index: f.kind, name: f.name, info: f.info, stat: `Restores ${healText(f.heal)}`,
      wear: -1, max: -1, equipped: false, keeps: true, count: f.count,
    }));
  }
  const list = tab === 'weapon' ? s.weapons : tab === 'bow' ? s.bows : s.shields;
  return list.map((it) => ({
    cat: tab, kind: it.kind, index: it.index, name: it.name, info: it.info, stat: it.stat,
    wear: it.wear, max: it.max, equipped: it.equipped, keeps: it.keeps, count: 1,
  }));
}

const used = (s: PouchState, tab: Tab): [number, number] =>
  tab === 'food' ? [s.food.length, 0] : tab === 'weapon' ? [s.weapons.length, s.slots.weapon] : tab === 'bow' ? [s.bows.length, s.slots.bow] : [s.shields.length, s.slots.shield];

const PANEL = 'rounded-2xl border border-stone-700/80 bg-stone-950/85 p-4 shadow-2xl backdrop-blur-md';
const keep = (e: React.MouseEvent) => e.preventDefault(); // (a click does not take the keyboard from the screen)

function Slot({ entry, selected, onSelect, onUse, slot }: { entry: Entry | null; selected: boolean; onSelect: () => void; onUse: () => void; slot: number }) {
  if (!entry) return <div aria-hidden className="aspect-square rounded-lg border border-stone-800/70 bg-stone-900/40" />;
  const frac = entry.wear < 0 ? 1 : entry.wear / Math.max(entry.max, 1);
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      aria-label={entry.name}
      data-slot={slot}
      data-equipped={entry.equipped || undefined}
      onMouseDown={keep}
      onMouseMove={onSelect}
      onClick={onSelect}
      onDoubleClick={onUse}
      className={`relative aspect-square rounded-lg border transition-colors ${selected ? 'border-amber-300 bg-stone-800 ring-2 ring-amber-300/60' : 'border-stone-700/80 bg-stone-900/70 hover:border-stone-500'} ${frac < 0.2 ? 'animate-pulse' : ''}`}
    >
      <span className="flex h-full w-full items-center justify-center"><ItemIcon cat={entry.cat} kind={entry.kind} size={40} /></span>
      {entry.equipped && <span aria-hidden className="absolute -right-1 -top-1 flex size-4 items-center justify-center rounded-full bg-emerald-400 text-[10px] font-bold text-stone-950">✓</span>}
      {entry.cat === 'food' && <span className="absolute bottom-0.5 right-1.5 font-mono text-[11px] font-bold text-stone-100 drop-shadow">×{entry.count}</span>}
      {entry.wear >= 0 && (
        <span aria-hidden className="absolute inset-x-1.5 bottom-1 h-[3px] rounded bg-stone-700">
          <span className={`block h-[3px] rounded ${frac < 0.25 ? 'bg-red-500' : 'bg-emerald-400'}`} style={{ width: `${frac * 100}%` }} />
        </span>
      )}
    </button>
  );
}

export default function InventoryScreen({ store, command, onClose, bindings }: {
  store: PouchStore;
  command: (cmd: PouchCommand) => boolean;
  onClose: () => void;
  bindings: Bindings;
}) {
  const s = usePouch(store);
  const [tab, setTab] = useState<Tab>('weapon');
  // the selected place in each tab (−1: not chosen yet: the one in hand)
  const [sel, setSel] = useState<Record<Tab, number>>({ weapon: -1, bow: -1, shield: -1, food: 0 });
  const [note, setNote] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);

  const entries = s ? entriesOf(s, tab) : [];
  const chosen = sel[tab] >= 0 ? sel[tab] : Math.max(0, entries.findIndex((e) => e.equipped));
  const at = Math.min(chosen, entries.length - 1);
  const entry: Entry | null = at >= 0 ? entries[at] : null;
  const full = !!s && s.health >= s.hearts * 4;

  const select = (i: number) => {
    setSel((v) => (v[tab] === i ? v : { ...v, [tab]: i }));
    setNote(null);
  };
  const move = (d: number) => {
    if (!entries.length) return;
    select(Math.min(entries.length - 1, Math.max(0, at + d)));
  };
  const changeTab = (d: number) => {
    setTab(TABS[(TABS.findIndex((t) => t.id === tab) + d + TABS.length) % TABS.length].id);
    setNote(null);
  };
  const run = (cmd: PouchCommand, refused: string) => {
    if (!command(cmd)) setNote(refused);
    else setNote(null);
  };
  // the main action: equip or put away, eat
  const use = () => {
    if (!entry || !s) return;
    if (tab === 'food') run({ op: 'eat', index: entry.index }, 'Your hearts are full.');
    else if (entry.equipped) run({ op: 'unequip', cat: tab }, 'Nothing to put away.');
    else run({ op: 'equip', cat: tab, index: entry.index }, 'It cannot be equipped.');
  };
  const drop = () => {
    if (!entry || tab === 'food') return;
    if (entry.keeps) setNote('The hero\'s own gear is kept.');
    else run({ op: 'drop', cat: tab, index: entry.index }, 'It cannot be dropped.');
  };

  const onKey = useEffectEvent((e: KeyboardEvent) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
    const code = normalizeCode(e.code);
    // (the inventory key itself is the engine's: it closes the screen)
    if (bindings.inventory.includes(code)) return;
    let handled = true;
    switch (code) {
      case 'Escape': onClose(); break;
      case 'ArrowLeft': case 'KeyA': move(-1); break;
      case 'ArrowRight': case 'KeyD': move(1); break;
      case 'ArrowUp': case 'KeyW': move(-COLS); break;
      case 'ArrowDown': case 'KeyS': move(COLS); break;
      case 'KeyQ': changeTab(-1); break;
      case 'KeyE': changeTab(1); break;
      case 'Tab': changeTab(e.shiftKey ? -1 : 1); break;
      case 'Enter': case 'Space': case 'KeyF': use(); break;
      case 'KeyX': case 'Delete': drop(); break;
      default: handled = false;
    }
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  });
  useEffect(() => {
    const h = (e: KeyboardEvent) => onKey(e);
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);
  useEffect(() => { root.current?.focus(); }, []);
  // a note fades after a moment
  useEffect(() => {
    if (!note) return;
    const t = window.setTimeout(() => setNote(null), 2600);
    return () => window.clearTimeout(t);
  }, [note]);

  const [count, slots] = s ? used(s, tab) : [0, 0];
  const durability = entry && entry.wear >= 0
    ? `${entry.wear} / ${entry.max} ${tab === 'bow' ? 'arrows' : 'blows'}`
    : 'Never wears out';
  const closeKey = keyHint(bindings, 'inventory');

  return (
    <div
      ref={root}
      tabIndex={-1}
      role="dialog"
      aria-modal="true"
      aria-label="Inventory"
      data-inventory
      className="absolute inset-0 z-40 flex flex-col bg-gradient-to-b from-stone-950/85 via-stone-950/30 to-stone-950/90 text-stone-100 outline-none animate-in fade-in duration-150 pointer-events-auto"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 px-4 pt-4 sm:px-8 sm:pt-6">
        <div className="flex items-center gap-3">
          <span className="font-serif text-lg tracking-wide text-amber-50">Inventory</span>
          {s && <HeartRow health={s.health} hearts={s.hearts} />}
        </div>
        <div role="tablist" aria-label="Inventory tabs" className="flex gap-1 rounded-full border border-stone-700/80 bg-stone-950/80 p-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              data-tab={t.id}
              onMouseDown={keep}
              onClick={() => { setTab(t.id); setNote(null); }}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 font-mono text-xs transition-colors ${tab === t.id ? 'bg-amber-300 font-bold text-stone-900' : 'text-stone-300 hover:bg-stone-800'}`}
            >
              <ItemIcon cat={t.id} kind={0} size={16} />
              <span>{t.label}</span>
            </button>
          ))}
        </div>
        <button
          type="button"
          aria-label="Close Inventory"
          data-action="close"
          onMouseDown={keep}
          onClick={onClose}
          className="flex h-8 items-center gap-1.5 rounded-md border border-stone-700 bg-stone-950/80 px-2.5 font-mono text-xs text-stone-300 hover:bg-stone-900 hover:text-white"
        >
          <X className="size-3.5" /> Close{closeKey && <kbd className="rounded bg-stone-100 px-1 text-[10px] font-bold text-stone-900">{closeKey}</kbd>}
        </button>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-auto px-4 py-4 sm:px-8 md:grid-cols-[minmax(18rem,24rem)_1fr_minmax(16rem,22rem)]">
        <section className={`${PANEL} self-start`} aria-label={TABS.find((t) => t.id === tab)?.label}>
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="font-serif text-xl text-amber-50">{TABS.find((t) => t.id === tab)?.label}</h2>
            <span className="font-mono text-xs text-stone-400">{slots ? `${count} / ${slots}` : count}</span>
          </div>
          <div role="listbox" aria-label={TABS.find((t) => t.id === tab)?.label} className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))` }}>
            {Array.from({ length: CELLS }, (_, i) => (
              <Slot key={i} slot={i} entry={entries[i] ?? null} selected={i === at} onSelect={() => select(i)} onUse={use} />
            ))}
          </div>
          {tab === 'bow' && s && (
            <div className="mt-3 flex items-center gap-2 font-mono text-xs text-stone-300">
              <span className="text-lg leading-none">➶</span> Arrows <strong className={s.arrows === 0 ? 'text-red-400' : 'text-stone-100'}>× {s.arrows}</strong>
            </div>
          )}
          {!entries.length && <p className="mt-3 font-mono text-xs text-stone-400">{tab === 'food' ? 'No food. Foes drop some, and so do camps.' : 'Nothing carried.'}</p>}
        </section>

        <div aria-hidden className="hidden md:block" />

        <aside className={`${PANEL} self-start`} aria-live="polite" data-details>
          {entry ? (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <ItemIcon cat={entry.cat} kind={entry.kind} size={56} />
                <div className="min-w-0">
                  <h2 className="font-serif text-xl leading-tight text-amber-50" data-name>{entry.name}</h2>
                  {entry.equipped && <span className="font-mono text-[11px] text-emerald-300">Equipped</span>}
                </div>
              </div>
              <p className="font-mono text-xs leading-relaxed text-stone-300">{entry.info}</p>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
                <dt className="text-stone-400">{tab === 'food' ? 'Effect' : 'Stats'}</dt>
                <dd className="text-right text-stone-100">{entry.stat}</dd>
                {tab !== 'food' && <><dt className="text-stone-400">Durability</dt><dd className="text-right text-stone-100">{durability}</dd></>}
                {tab === 'food' && <><dt className="text-stone-400">Kept</dt><dd className="text-right text-stone-100">× {entry.count}</dd></>}
              </dl>
              {entry.wear >= 0 && (
                <span aria-hidden className="h-1.5 w-full rounded bg-stone-700">
                  <span className={`block h-1.5 rounded ${entry.wear / Math.max(entry.max, 1) < 0.25 ? 'bg-red-500' : 'bg-emerald-400'}`} style={{ width: `${(entry.wear / Math.max(entry.max, 1)) * 100}%` }} />
                </span>
              )}
              <div className="flex gap-2">
                <button
                  type="button"
                  data-action={tab === 'food' ? 'eat' : entry.equipped ? 'unequip' : 'equip'}
                  disabled={tab === 'food' && full}
                  onMouseDown={keep}
                  onClick={use}
                  className="flex-1 rounded-md bg-amber-300 px-3 py-2 font-mono text-xs font-bold text-stone-900 transition-colors hover:bg-amber-200 disabled:cursor-not-allowed disabled:bg-stone-700 disabled:text-stone-400"
                >
                  {tab === 'food' ? 'Eat' : entry.equipped ? 'Put away' : 'Equip'}
                </button>
                {tab !== 'food' && (
                  <button
                    type="button"
                    data-action="drop"
                    disabled={entry.keeps}
                    onMouseDown={keep}
                    onClick={drop}
                    className="rounded-md border border-stone-600 px-3 py-2 font-mono text-xs text-stone-200 transition-colors hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Drop
                  </button>
                )}
              </div>
              {tab === 'food' && full && !note && <p className="font-mono text-[11px] text-stone-400">Your hearts are full.</p>}
              {note && <p role="status" className="font-mono text-[11px] text-amber-300" data-note>{note}</p>}
            </div>
          ) : (
            <p className="font-mono text-xs text-stone-400">Nothing selected.</p>
          )}
        </aside>
      </div>

      <footer className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 px-4 pb-4 font-mono text-[11px] text-stone-400 sm:px-8">
        <span><strong className="text-stone-200">←↑↓→</strong> Select</span>
        <span><strong className="text-stone-200">Q / E</strong> Tabs</span>
        <span><strong className="text-stone-200">Enter</strong> {tab === 'food' ? 'Eat' : 'Equip'}</span>
        {tab !== 'food' && <span><strong className="text-stone-200">X</strong> Drop</span>}
        <span><strong className="text-stone-200">{closeKey || 'Esc'}</strong> / <strong className="text-stone-200">Esc</strong> Close</span>
      </footer>
    </div>
  );
}
