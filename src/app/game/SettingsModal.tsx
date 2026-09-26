'use client';

import React from 'react';
import { Dices } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

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
  showHud: boolean;
  setShowHud: React.Dispatch<React.SetStateAction<boolean>>;
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
  showHud,
  setShowHud,
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
