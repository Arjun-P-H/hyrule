'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import {
  Compass,
  Eye,
  Maximize2,
  Minimize2,
  Sliders,
  Sparkles,
  ArrowLeft,
  Move,
  PersonStanding,
  RotateCcw,
  SwitchCamera,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useTerrainEngine, type PresetId } from '@/lib/engine/useTerrainEngine';
import { useEngineApi, useEngineMode, useEngineStats, useHeroStamina, type CamMode, type StatsStore } from '@/lib/engine/statsStore';
import { useSeedState } from '@/lib/engine/useSeedState';
import {
  DEFAULT_SEED,
  CAMERA,
  QUALITY_PRESETS,
} from '@/lib/constants';

const SettingsModal = dynamic(() => import('./SettingsModal'), {
  ssr: false,
});

// Subscribes to engine telemetry itself, so its 5 Hz updates re-render only
// this panel and not the page (tooltips, buttons, dialogs) around it.
const HudOverlay = React.memo(function HudOverlay({
  statsStore,
  preset,
  showHud,
}: {
  statsStore: StatsStore;
  preset: PresetId;
  showHud: boolean;
}) {
  const stats = useEngineStats(statsStore);
  if (!showHud || !stats) return null;
  return (
    <div className="flex flex-col gap-1.5 p-3 rounded-lg border border-stone-800/80 bg-stone-950/85 backdrop-blur-md font-mono text-xs text-stone-200 shadow-2xl w-[280px] max-w-[calc(100vw-32px)]">
      <div className="flex items-center justify-between border-b border-stone-800 pb-1.5">
        <span className="text-emerald-400 font-bold text-sm">
          {stats.fps} <span className="text-[10px] text-stone-400 font-normal">FPS</span>
        </span>
        <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-stone-800 text-stone-300">
          {preset} quality
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] pt-0.5">
        <span className="text-stone-400">Mode:</span>
        <span className="text-amber-300 font-semibold text-right uppercase">
          {stats.mode} {stats.mode === 'walk' && (stats.grounded ? '(ground)' : '(air)')}
          {stats.mode === 'hero' && stats.action && `· ${stats.action}`}
        </span>
        <span className="text-stone-400">Pos (X,Y,Z):</span>
        <span className="text-stone-200 text-right">
          {stats.x.toFixed(1)}, {stats.y.toFixed(1)}, {stats.z.toFixed(1)}
        </span>
        <span className="text-stone-400">Chunk Origin:</span>
        <span className="text-stone-400 text-right">
          [{stats.originX}, {stats.originZ}]
        </span>
        <span className="text-stone-400">Resolution:</span>
        <span className="text-stone-300 text-right">{stats.resPct}% native</span>
        <span className="text-stone-400">Renderer:</span>
        <span className="text-stone-300 text-right">{stats.api === 'webgpu' ? 'WebGPU · raster clipmap' : 'WebGL2 · ray-march'}</span>
      </div>
    </div>
  );
});

// The stamina wheel beside the hero (third person only): hidden while full,
// red while exhausted. Updated at the 5 Hz telemetry rate; the CSS transition
// fills the gaps between samples.
const STAMINA_R = 15;
const STAMINA_C = 2 * Math.PI * STAMINA_R;
const StaminaWheel = React.memo(function StaminaWheel({ statsStore }: { statsStore: StatsStore }) {
  const { percent, exhausted } = useHeroStamina(statsStore);
  if (percent === null) return null;
  const hidden = percent >= 100 && !exhausted;
  // Beside the hero's head: the orbit camera frames the chest at the screen
  // centre with ~6.4 m of height in view, so the offsets scale with the viewport.
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute left-[calc(50%+5vh)] top-[calc(50%-12vh)] z-10 transition-opacity duration-500 ${hidden ? 'opacity-0' : 'opacity-100'}`}
    >
      <svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r={STAMINA_R} fill="rgba(12,10,9,0.45)" stroke="rgba(255,255,255,0.14)" strokeWidth="5" />
        <circle
          cx="20"
          cy="20"
          r={STAMINA_R}
          fill="none"
          stroke={exhausted ? '#ef4444' : '#4ade80'}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={STAMINA_C}
          strokeDashoffset={STAMINA_C * (1 - percent / 100)}
          transform="rotate(-90 20 20)"
          style={{ transition: 'stroke-dashoffset 200ms linear, stroke 200ms' }}
        />
      </svg>
    </div>
  );
});

const MODE_LABEL: Record<CamMode, string> = { hero: 'HERO', walk: 'WALK', fly: 'FLY' };

function ControlsBanner({ mode, isPointerLocked }: { mode: CamMode | null; isPointerLocked: boolean }) {
  const keys: Array<[string, string]> =
    mode === 'fly'
      ? [['WASD', 'Move'], ['Shift', 'Boost'], ['Space', 'Up'], ['C', 'Down'], ['F', 'Land'], ['V', 'Hero'], ['R', 'Respawn']]
      : mode === 'walk'
        ? [['WASD', 'Move'], ['Shift', 'Sprint'], ['Space', 'Jump'], ['V', 'Third person'], ['F', 'Fly'], ['R', 'Respawn']]
        : [['WASD', 'Move'], ['Shift', 'Sprint'], ['Space', 'Jump · Glide'], ['C', 'Sneak'], ['Wheel', 'Zoom'], ['V', 'First person'], ['F', 'Fly'], ['R', 'Respawn']];
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 px-4 py-2 rounded-full border border-stone-800/80 bg-stone-950/80 backdrop-blur-md font-mono text-[11px] text-stone-300 shadow-xl">
      {keys.map(([k, what]) => (
        <React.Fragment key={k}>
          <span><strong className="text-emerald-400">{k}</strong> {what}</span>
          <span className="text-stone-700">•</span>
        </React.Fragment>
      ))}
      <span className="text-amber-300">
        {isPointerLocked ? 'Mouse Locked (Esc to free)' : 'Click to Lock Mouse'}
      </span>
    </div>
  );
}

export default function GamePage() {
  const { seed, seedText, onSeedChange, randomizeSeed } = useSeedState(DEFAULT_SEED);
  const [preset, setPreset] = useState<PresetId>('med');
  const [showSettings, setShowSettings] = useState(false);
  const [showHud, setShowHud] = useState(true);
  const [isPointerLocked, setIsPointerLocked] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [fov, setFov] = useState<number>(CAMERA.fov);
  const [sensitivity, setSensitivity] = useState<number>(CAMERA.sensitivity);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { statsStore, error, respawn, toggleMode, toggleView } = useTerrainEngine({
    canvasRef,
    seed,
    preset,
    fov,
    sensitivity,
    speedMultiplier,
    enablePointerLock: true,
    running: true,
    initialMode: 'hero',
    renderer: 'auto',
    onPointerLockChange: setIsPointerLocked,
    onToggleHud: () => setShowHud((v) => !v),
  });
  const mode = useEngineMode(statsStore);
  const api = useEngineApi(statsStore);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  return (
    <TooltipProvider>
      <main className="relative h-screen w-screen overflow-hidden bg-black select-none font-sans">
        {/* Fullscreen Canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full block cursor-crosshair touch-none"
          tabIndex={0}
        />

        {/* Top Left Navigation & HUD */}
        <div className="absolute top-4 left-4 z-20 flex flex-col gap-2 pointer-events-auto max-w-[calc(100vw-32px)]">
          <div className="flex items-center gap-2">
            <Link href="/" passHref>
              <Button
                variant="outline"
                size="sm"
                className="h-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:text-white hover:bg-stone-900 font-mono text-xs"
              >
                <ArrowLeft className="mr-1.5 size-3.5" /> Spec Console
              </Button>
            </Link>
            <span className="hidden sm:flex h-8 items-center px-2.5 rounded-md bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-mono text-xs font-semibold">
              <Sparkles className="mr-1.5 size-3.5" /> {api === 'webgpu' ? 'WebGPU Engine' : 'WebGL2 Engine'}
            </span>
          </div>

          <HudOverlay statsStore={statsStore} preset={preset} showHud={showHud} />
        </div>

        <StaminaWheel statsStore={statsStore} />

        {/* Top Right Quick Controls */}
        <div className="absolute top-4 right-4 z-20 flex items-center gap-1.5 sm:gap-2 pointer-events-auto">
          {/* Preset Buttons */}
          <div className="flex bg-stone-950/80 backdrop-blur-md border border-stone-800 rounded-md p-0.5">
            {QUALITY_PRESETS.map((p) => {
              const active = preset === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => setPreset(p.id)}
                  className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                    active
                      ? 'bg-emerald-500 text-black font-bold'
                      : 'text-stone-400 hover:text-stone-200'
                  }`}
                >
                  {p.label.toUpperCase()}
                </button>
              );
            })}
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                onClick={toggleMode}
                data-engine-mode={mode ?? ''}
                className="h-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900 hover:text-emerald-400 font-mono text-xs px-2.5 flex items-center gap-1.5"
              >
                {mode === 'fly' ? (
                  <Compass className="size-3.5 text-emerald-400" />
                ) : mode === 'walk' ? (
                  <Eye className="size-3.5 text-amber-400" />
                ) : (
                  <PersonStanding className="size-3.5 text-amber-400" />
                )}
                <span>{mode ? MODE_LABEL[mode] : '—'}</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              {mode === 'fly' ? 'Land (F)' : 'Take off: Flight Mode (F)'}
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={toggleView}
                aria-label="Switch third / first person"
                className="h-8 w-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900 hover:text-emerald-400"
              >
                <SwitchCamera className="size-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              {mode === 'hero' ? 'First Person (V)' : 'Third Person Hero (V)'}
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={respawn}
                className="h-8 w-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900 hover:text-emerald-400"
              >
                <RotateCcw className="size-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              Respawn at Spawn Point (R)
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowSettings((v) => !v)}
                className="h-8 w-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900"
              >
                <Sliders className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              World Settings & Seed
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                onClick={toggleFullscreen}
                className="h-8 w-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900"
              >
                {isFullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              Toggle Fullscreen (F11)
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Settings Modal Popover */}
        {showSettings && (
          <SettingsModal
            showSettings={showSettings}
            onClose={() => setShowSettings(false)}
            seedText={seedText}
            onSeedChange={onSeedChange}
            randomizeSeed={randomizeSeed}
            speedMultiplier={speedMultiplier}
            setSpeedMultiplier={setSpeedMultiplier}
            fov={fov}
            setFov={setFov}
            sensitivity={sensitivity}
            setSensitivity={setSensitivity}
            mode={mode ?? undefined}
            toggleMode={toggleMode}
            toggleView={toggleView}
            showHud={showHud}
            setShowHud={setShowHud}
          />
        )}

        {/* Bottom Center Controls Banner */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none w-max max-w-[calc(100vw-32px)]">
          <ControlsBanner mode={mode} isPointerLocked={isPointerLocked} />
        </div>

        {/* Pointer lock hint on canvas initial click */}
        {!isPointerLocked && (
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10 pointer-events-none opacity-80 transition-opacity">
            <div className="px-5 py-2.5 rounded-xl border border-stone-700 bg-stone-950/90 backdrop-blur-md text-stone-300 font-mono text-xs flex items-center gap-2.5 shadow-2xl animate-pulse">
              <Move className="size-4 text-emerald-400" />
              <span>Click anywhere to lock mouse and look around</span>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-stone-950/90 p-6">
            <div className="p-6 rounded-xl border border-red-900 bg-red-950/40 max-w-md text-center">
              <h3 className="text-red-400 font-bold mb-2">WebGL Engine Error</h3>
              <p className="text-xs font-mono text-stone-300">{error}</p>
            </div>
          </div>
        )}
      </main>
    </TooltipProvider>
  );
}
