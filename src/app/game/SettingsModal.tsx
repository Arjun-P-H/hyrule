'use client';

import React, { useState } from 'react';
import { Dices, Pause, Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useSkyState, type RenderApi, type StatsStore } from '@/lib/engine/statsStore';
import type { SkyMode } from '@/lib/gpu/engineGpu';
import { DAY_MINUTES, clockText, dayPhase } from '@/lib/sky/clock';

export type SkyControls = {
  statsStore: StatsStore;
  api: RenderApi | null;
  sky: SkyMode;
  setSky: (mode: SkyMode) => void;
  setHour: (hour: number) => void;
  setTimePaused: (paused: boolean) => void;
  setDayMinutes: (minutes: number) => void;
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
}

const choice = (active: boolean) =>
  `px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
    active
      ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 font-bold'
      : 'border-stone-800 bg-stone-900 text-stone-400 hover:text-stone-200'
  }`;

/** The sky: day–night or classic, and the clock (the hour, stopped or running, the length of a day). */
function SkySection({ statsStore, api, sky, setSky, setHour, setTimePaused, setDayMinutes }: SkyControls) {
  const state = useSkyState(statsStore);
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
          <span className="text-stone-500 text-[10px]">Hold T to run time fast (an hour a second).</span>
        </>
      )}
    </div>
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
}: SettingsModalProps) {
  if (!showSettings) return null;

  return (
    <div className="absolute top-16 right-4 z-30 w-80 rounded-lg border border-stone-800 bg-stone-950/95 backdrop-blur-md p-4 text-stone-200 shadow-2xl font-mono text-xs flex flex-col gap-3 pointer-events-auto">
      <div className="flex items-center justify-between border-b border-stone-800 pb-2">
        <span className="font-semibold text-stone-100 uppercase tracking-wider">Engine Settings</span>
        <button
          onClick={onClose}
          className="text-stone-500 hover:text-stone-200 text-sm"
          aria-label="Close Settings"
        >
          ✕
        </button>
      </div>

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

      <SkySection {...skyControls} />

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

      <div className="flex items-center justify-between pt-1 border-t border-stone-800/80">
        <span className="text-stone-400 text-[11px]">Control Mode</span>
        <Button
          variant="outline"
          size="sm"
          onClick={toggleMode}
          className="h-6 px-2 text-[10px] border-stone-800 bg-stone-900 font-mono text-stone-200 uppercase"
        >
          {mode} (F)
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
          {mode === 'hero' ? 'Third person' : mode === 'walk' ? 'First person' : 'Free'} (V)
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
          {showHud ? 'Visible (H)' : 'Hidden (H)'}
        </Button>
      </div>
    </div>
  );
}
