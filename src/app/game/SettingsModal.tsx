'use client';

import React, { useEffect, useEffectEvent, useState, useSyncExternalStore } from 'react';
import { ChevronLeft, Dices, Keyboard, Pause, Play, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSkyState, type RenderApi, type StatsStore } from '@/lib/engine/statsStore';
import {
  ACTIONS,
  DEFAULT_BINDINGS,
  bindKey,
  clearSlot,
  isBindable,
  isDefault,
  keyHint,
  keyLabel,
  keyNote,
  lookKeys,
  looksWithKeys,
  lookWithArrows,
  lookWithoutKeys,
  normalizeCode,
  type ActionId,
  type Bindings,
} from '@/lib/engine/keybinds';
import type { SkyMode } from '@/lib/gpu/engineGpu';
import { ISLAND_HEIGHTS, setIslandHeights } from '@/lib/skyworld/heights';
import { DAY_MINUTES, clockText, dayPhase } from '@/lib/sky/clock';

export type SkyControls = {
  statsStore: StatsStore;
  api: RenderApi | null;
  sky: SkyMode;
  setSky: (mode: SkyMode) => void;
  setHour: (hour: number) => void;
  setTimePaused: (paused: boolean) => void;
  setDayMinutes: (minutes: number) => void;
  /** Go to the other world, as through the portal (for inspection). */
  travel: () => void;
};

export interface SettingsModalProps {
  showSettings: boolean;
  onClose: () => void;
  seedText: string;
  onSeedChange: (v: string) => void;
  randomizeSeed: () => void;
  speedMultiplier: number;
  setSpeedMultiplier: (v: number) => void;
  fov: number;
  setFov: (v: number) => void;
  sensitivity: number;
  setSensitivity: (v: number) => void;
  mode?: string;
  toggleMode: () => void;
  toggleView: () => void;
  showHud: boolean;
  setShowHud: React.Dispatch<React.SetStateAction<boolean>>;
  skyControls: SkyControls;
  /** The keys of every control, and the setter that changes (and saves) them. */
  bindings: Bindings;
  setBindings: (b: Bindings) => void;
}

const choice = (active: boolean) =>
  `px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
    active
      ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 font-bold'
      : 'border-stone-800 bg-stone-900 text-stone-400 hover:text-stone-200'
  }`;

/** The sky: day–night or classic, and the clock (the hour, stopped or running, the length of a day). */
function SkySection({ statsStore, api, sky, setSky, setHour, setTimePaused, setDayMinutes, travel, fastTimeKey }: SkyControls & { fastTimeKey: string }) {
  const state = useSkyState(statsStore);
  const world = useSyncExternalStore(statsStore.subscribe, () => statsStore.get()?.world ?? null, () => null);
  // While the slider is dragged it shows the drag, not the clock (which reports five times a second).
  const [drag, setDrag] = useState<number | null>(null);
  const cycleAvailable = api !== 'webgl2';
  const minutes = drag ?? state?.minutes ?? 0;
  const building = !!state && state.sky !== null && state.sky !== state.wanted;
  return (
    <div className="flex flex-col gap-2 pt-1 border-t border-stone-800/80">
      <div className="flex items-center justify-between">
        <span className="text-stone-400 text-[11px]">Sky</span>
        <div className="flex gap-1">
          <button onClick={() => setSky('cycle')} disabled={!cycleAvailable} className={`${choice(sky === 'cycle' && cycleAvailable)} disabled:opacity-40`}>
            Day &amp; night
          </button>
          <button onClick={() => setSky('classic')} className={choice(sky === 'classic' || !cycleAvailable)}>
            Classic
          </button>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-stone-400 text-[11px]">World</span>
        <div className="flex gap-1">
          <button onClick={() => world === 'clouds' && travel()} disabled={!world} className={choice(world === 'land')}>
            Land
          </button>
          <button onClick={() => world === 'land' && travel()} disabled={!world || !cycleAvailable} className={`${choice(world === 'clouds')} disabled:opacity-40`}>
            Cloud world
          </button>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-stone-400 text-[11px]">Sky islands</span>
        <div className="flex gap-1">
          <button onClick={() => ISLAND_HEIGHTS !== 'low' && setIslandHeights('low')} className={choice(ISLAND_HEIGHTS === 'low')}>
            Low
          </button>
          <button onClick={() => ISLAND_HEIGHTS !== 'original' && setIslandHeights('original')} className={choice(ISLAND_HEIGHTS === 'original')}>
            Original
          </button>
        </div>
      </div>
      <span className="text-stone-500 text-[10px]">Switching the islands&apos; height reloads the game.</span>
      {!cycleAvailable && <span className="text-stone-500 text-[10px]">The day–night sky needs WebGPU; WebGL2 draws the classic sky.</span>}
      {building && <span className="text-amber-300/80 text-[10px]">Building the {sky === 'cycle' ? 'day–night' : 'classic'} sky…</span>}
      {cycleAvailable && sky === 'cycle' && state && (
        <>
          <div className="flex items-center justify-between">
            <span className="text-stone-400 text-[11px]">Time of day</span>
            <div className="flex items-center gap-2">
              <span className="text-stone-200 tabular-nums">{clockText(minutes / 60)}</span>
              <span className="text-stone-500 text-[10px] w-[74px] text-right">{dayPhase(minutes / 60)}</span>
              <button
                onClick={() => setTimePaused(!state.paused)}
                aria-label={state.paused ? 'Run the clock' : 'Stop the clock'}
                className="h-6 w-6 flex items-center justify-center rounded border border-stone-800 bg-stone-900 text-stone-300 hover:text-emerald-300"
              >
                {state.paused ? <Play className="size-3" /> : <Pause className="size-3" />}
              </button>
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={1439}
            step={1}
            value={minutes}
            aria-label="Time of day"
            onChange={(e) => { const v = Number(e.target.value); setDrag(v); setHour(v / 60); }}
            onPointerUp={() => setDrag(null)}
            onBlur={() => setDrag(null)}
            className="w-full accent-emerald-500"
          />
          <div className="flex items-center justify-between">
            <span className="text-stone-400 text-[11px]">Day length</span>
            <div className="flex gap-1">
              {DAY_MINUTES.map((m) => (
                <button key={m} onClick={() => setDayMinutes(m)} className={choice(state.dayMinutes === m)}>
                  {m} min
                </button>
              ))}
            </div>
          </div>
          {fastTimeKey && <span className="text-stone-500 text-[10px]">Hold {fastTimeKey} to run time fast (an hour a second).</span>}
        </>
      )}
    </div>
  );
}

/** The popover's frame: a header (a way back, when this is not the first page, and one to close) over a body that scrolls on a short window. */
function Panel({ title, onBack, onClose, children }: { title: string; onBack?: () => void; onClose: () => void; children: React.ReactNode }) {
  return (
    <div className="absolute top-16 right-4 z-30 w-80 max-h-[calc(100dvh-5rem)] rounded-lg border border-stone-800 bg-stone-950/95 backdrop-blur-md text-stone-200 shadow-2xl font-mono text-xs flex flex-col pointer-events-auto">
      <div className="flex items-center justify-between border-b border-stone-800 px-4 pt-4 pb-2 shrink-0">
        {onBack ? (
          <button
            onClick={onBack}
            className="flex items-center gap-1 font-semibold text-stone-100 uppercase tracking-wider hover:text-emerald-300"
            aria-label="Back to Engine Settings"
          >
            <ChevronLeft className="size-3.5" /> {title}
          </button>
        ) : (
          <span className="font-semibold text-stone-100 uppercase tracking-wider">{title}</span>
        )}
        <button
          onClick={onClose}
          className="text-stone-500 hover:text-stone-200 text-sm"
          aria-label="Close Settings"
        >
          ✕
        </button>
      </div>
      <div className="flex flex-col gap-3 overflow-y-auto px-4 pt-3 pb-4">{children}</div>
    </div>
  );
}

const GROUPS = [...new Set(ACTIONS.map((a) => a.group))];

/** One key of a control: click it, then press the key wanted. */
function KeySlot({ name, label, listening, changed, onClick }: {
  name: string;
  label: string | null;
  listening: boolean;
  changed: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      data-key-slot
      onClick={onClick}
      title={label ?? undefined}
      aria-pressed={listening}
      aria-label={`${name}: ${listening ? 'press a key' : label ?? 'no key'}. Click to change`}
      className={`h-6 w-[54px] truncate rounded border px-1 text-center text-[10px] font-mono transition-colors ${
        listening
          ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 animate-pulse'
          : label === null
            ? 'border-stone-800 bg-stone-900/50 text-stone-600 hover:text-stone-300'
            : `border-stone-800 bg-stone-900 hover:border-stone-600 ${changed ? 'text-emerald-300' : 'text-stone-200'}`
      }`}
    >
      {listening ? 'press…' : label ?? '—'}
    </button>
  );
}

/** Every control with its two keys. Keys that differ from the defaults are green; a control with no key is amber. */
function ControlsPanel({ bindings, setBindings }: { bindings: Bindings; setBindings: (b: Bindings) => void }) {
  const [listening, setListening] = useState<{ id: ActionId; slot: number } | null>(null);
  // (the key is taken: the slot stops listening when the key is let go)
  const [taken, setTaken] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  // A key pressed while a slot waits for one: Esc leaves it, Backspace empties it, any other key is bound to it.
  const press = useEffectEvent((code: string): 'wait' | 'taken' | 'done' => {
    if (!listening) return 'done';
    if (code === 'Escape') {
      setNotice(null);
      setListening(null);
      return 'done';
    }
    if (code === 'Backspace' || code === 'Delete') {
      setBindings(clearSlot(bindings, listening.id, listening.slot));
      setNotice(null);
      setListening(null);
      return 'done';
    }
    const key = normalizeCode(code);
    if (!isBindable(key)) {
      setNotice(`${keyLabel(key) || 'That key'} can't be a control key.`);
      return 'wait';
    }
    const { bindings: next, from } = bindKey(bindings, listening.id, listening.slot, key);
    setBindings(next);
    // (a control that lost its key says so: it may be left with none)
    setNotice(from ? `${ACTIONS.find((a) => a.id === from)?.label} lost ${keyLabel(key)}${next[from][0] === null && next[from][1] === null ? ' and has no key now' : ''}.` : null);
    return 'taken';
  });

  // While a slot listens the keys are its own: the game's key handler (a window listener) never sees them.
  useEffect(() => {
    if (!listening) return;
    let releasing: string | null = null; // the key just taken, until it is let go (its release must not click the slot again)
    const down = (e: KeyboardEvent) => {
      if (e.metaKey || e.altKey) return; // the browser's and the system's shortcuts stay theirs
      e.preventDefault();
      e.stopPropagation();
      if (e.repeat || releasing) return;
      if (press(e.code) === 'taken') {
        releasing = e.code;
        setTaken(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code !== releasing) return;
      e.preventDefault();
      e.stopPropagation();
      setListening(null);
    };
    const away = (e: Event) => {
      if (e.target instanceof Element && e.target.closest('[data-key-slot]')) return;
      setNotice(null);
      setListening(null);
    };
    const stop = () => setListening(null);
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    window.addEventListener('pointerdown', away, true);
    window.addEventListener('blur', stop);
    return () => {
      window.removeEventListener('keydown', down, true);
      window.removeEventListener('keyup', up, true);
      window.removeEventListener('pointerdown', away, true);
      window.removeEventListener('blur', stop);
    };
  }, [listening]);

  const listen = (id: ActionId, slot: number) => {
    setNotice(null);
    setTaken(false);
    setListening((cur) => (cur && cur.id === id && cur.slot === slot ? null : { id, slot }));
  };

  return (
    <>
      <div className="flex items-start justify-between gap-2">
        <span className="text-stone-500 text-[10px] leading-snug">
          Click a key, then press the one you want (Esc cancels, Backspace clears). Each key does one thing: giving it here takes it from its old control. Saved in this browser.
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={isDefault(bindings)}
          onClick={() => {
            setBindings(DEFAULT_BINDINGS);
            setListening(null);
            setNotice(null);
          }}
          title="Every control back to its default keys"
          className="h-6 px-2 text-[10px] shrink-0 border-stone-800 bg-stone-900 font-mono text-stone-200 disabled:opacity-40"
        >
          <RotateCcw className="size-3" /> Reset
        </Button>
      </div>
      {notice && <span role="status" className="text-amber-300/90 text-[10px] leading-snug">{notice}</span>}
      {GROUPS.map((group) => (
        <div key={group} className="flex flex-col gap-1.5 pt-1 border-t border-stone-800/80">
          <span className="text-stone-500 text-[10px] uppercase tracking-wider">{group}</span>
          {ACTIONS.filter((a) => a.group === group).map((a) => {
            const slots = bindings[a.id];
            // (a control that has lost the keys it came with; the look keys come with none)
            const unbound = slots[0] === null && slots[1] === null && a.keys.length > 0;
            return (
              <div key={a.id} className="flex items-center justify-between gap-2">
                <span className={`text-[11px] ${unbound ? 'text-amber-300' : 'text-stone-400'}`}>
                  {a.label}
                  {a.also && <span className="text-stone-500 text-[10px] whitespace-nowrap"> · {a.also}</span>}
                </span>
                <div className="flex gap-1 shrink-0">
                  {[0, 1].map((slot) => {
                    const code = slots[slot];
                    return (
                      <KeySlot
                        key={slot}
                        name={a.label}
                        label={code === null ? null : keyLabel(code)}
                        listening={!taken && listening?.id === a.id && listening.slot === slot}
                        changed={code !== DEFAULT_BINDINGS[a.id][slot]}
                        onClick={() => listen(a.id, slot)}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </>
  );
}

export default function SettingsModal({
  showSettings,
  onClose,
  seedText,
  onSeedChange,
  randomizeSeed,
  speedMultiplier,
  setSpeedMultiplier,
  fov,
  setFov,
  sensitivity,
  setSensitivity,
  mode = 'fly',
  toggleMode,
  toggleView,
  showHud,
  setShowHud,
  skyControls,
  bindings,
  setBindings,
}: SettingsModalProps) {
  // 'controls': the keys of every control, in place of the settings
  const [view, setView] = useState<'settings' | 'controls'>('settings');
  if (!showSettings) return null;

  if (view === 'controls') {
    return (
      <Panel title="Controls" onBack={() => setView('settings')} onClose={onClose}>
        <ControlsPanel bindings={bindings} setBindings={setBindings} />
      </Panel>
    );
  }

  // Key Look is on while any key turns the camera (the arrows, or whatever Controls gave it)
  const keyLook = looksWithKeys(bindings);

  return (
    <Panel title="Engine Settings" onClose={onClose}>
      <div className="flex flex-col gap-1.5">
        <span className="text-stone-400 text-[11px]">Terrain Seed (uint32)</span>
        <div className="flex gap-2">
          <Input
            type="number"
            value={seedText}
            onChange={(e) => onSeedChange(e.target.value)}
            className="h-8 border-stone-800 bg-stone-900 font-mono text-xs text-stone-200"
          />
          <Button
            variant="outline"
            size="icon"
            onClick={randomizeSeed}
            className="h-8 w-8 border-stone-800 bg-stone-900 hover:bg-stone-800 text-stone-300 shrink-0"
          >
            <Dices className="size-3.5" />
          </Button>
        </div>
      </div>

      <SkySection {...skyControls} fastTimeKey={keyHint(bindings, 'fastTime')} />

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Flight Speed</span>
        <div className="flex gap-1">
          {[1, 2, 3].map((mult) => (
            <button
              key={mult}
              onClick={() => setSpeedMultiplier(mult)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
                speedMultiplier === mult
                  ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 font-bold'
                  : 'border-stone-800 bg-stone-900 text-stone-400 hover:text-stone-200'
              }`}
            >
              {mult}x
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Field of View</span>
        <div className="flex gap-1">
          {[
            { deg: 60, label: '60°' },
            { deg: 75, label: '75°' },
            { deg: 90, label: '90°' },
          ].map((opt) => (
            <button
              key={opt.deg}
              onClick={() => setFov(opt.deg)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
                fov === opt.deg
                  ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 font-bold'
                  : 'border-stone-800 bg-stone-900 text-stone-400 hover:text-stone-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Mouse Look</span>
        <div className="flex gap-1">
          {[
            { val: 0.0012, label: 'Low' },
            { val: 0.0022, label: 'Med' },
            { val: 0.0038, label: 'High' },
          ].map((opt) => (
            <button
              key={opt.label}
              onClick={() => setSensitivity(opt.val)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
                sensitivity === opt.val
                  ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 font-bold'
                  : 'border-stone-800 bg-stone-900 text-stone-400 hover:text-stone-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-1.5 pt-1 border-t border-stone-800/80">
        <div className="flex items-center justify-between">
          <span className="text-stone-400 text-[11px]">Key Look</span>
          <div className="flex gap-1">
            <button onClick={() => keyLook && setBindings(lookWithoutKeys(bindings))} aria-pressed={!keyLook} className={choice(!keyLook)}>
              Off
            </button>
            <button onClick={() => !keyLook && setBindings(lookWithArrows(bindings))} aria-pressed={keyLook} className={choice(keyLook)}>
              On
            </button>
          </div>
        </div>
        <span className="text-stone-500 text-[10px]">
          {keyLook
            ? `${lookKeys(bindings)} turn the camera, at the Mouse Look speed. Pick other keys under Controls.`
            : 'Turn the camera with the arrow keys instead of the mouse; WASD still moves.'}
        </span>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Control Mode</span>
        <Button
          variant="outline"
          size="sm"
          onClick={toggleMode}
          className="h-6 px-2 text-[10px] border-stone-800 bg-stone-900 font-mono text-stone-200 uppercase"
        >
          {mode}{keyNote(bindings, 'toggleMode')}
        </Button>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Camera</span>
        <Button
          variant="outline"
          size="sm"
          onClick={toggleView}
          className="h-6 px-2 text-[10px] border-stone-800 bg-stone-900 font-mono text-stone-200 uppercase"
        >
          {mode === 'hero' ? 'Third person' : mode === 'walk' ? 'First person' : 'Free'}{keyNote(bindings, 'toggleView')}
        </Button>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">HUD Overlay</span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowHud((v) => !v)}
          className="h-6 px-2 text-[10px] border-stone-800 bg-stone-900"
        >
          {showHud ? 'Visible' : 'Hidden'}{keyNote(bindings, 'toggleHud')}
        </Button>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Controls</span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setView('controls')}
          className="h-6 px-2 text-[10px] border-stone-800 bg-stone-900 font-mono text-stone-200"
        >
          <Keyboard className="size-3" /> Configure keys
        </Button>
      </div>
    </Panel>
  );
}
