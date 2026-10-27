'use client';

import React, { useState, useEffect, useEffectEvent, useRef, useSyncExternalStore } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import {
  Compass,
  Eye,
  Map as MapIcon,
  Maximize2,
  Minimize2,
  Moon,
  Pause,
  Sliders,
  Sparkles,
  Sun,
  Sunrise,
  Sunset,
  ArrowLeft,
  Backpack,
  Move,
  PersonStanding,
  RotateCcw,
  SwitchCamera,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useTerrainEngine, type PresetId } from '@/lib/engine/useTerrainEngine';
import { useEngineApi, useEngineMode, useEngineStats, useHeroStamina, useSkyState, type CamMode, type StatsStore } from '@/lib/engine/statsStore';
import { useSeedState } from '@/lib/engine/useSeedState';
import { useKeybinds } from '@/lib/engine/keybindStore';
import { keyHint, keyNote, lookKeys, moveKeys, type ActionId, type Bindings } from '@/lib/engine/keybinds';
import type { SkyMode } from '@/lib/gpu/engineGpu';
import type { SpawnAt } from '@/lib/engine/engineCore';
import { clockText5, dayPhase } from '@/lib/sky/clock';
import { NetSession, useNetView } from '@/lib/net/session';
import { CHARACTERS } from '@/lib/hero/characters';
import { Lobby, PlayersPanel } from './Multiplayer';
import { CombatHud, Hearts } from './CombatHud';
import { MapHud } from './MapHud';
import {
  DEFAULT_SEED,
  CAMERA,
  QUALITY_PRESETS,
} from '@/lib/constants';

const SettingsModal = dynamic(() => import('./SettingsModal'), {
  ssr: false,
});

// The inventory screen is fetched the first time it is opened.
const InventoryScreen = dynamic(() => import('./InventoryScreen'), {
  ssr: false,
});

// So is the full map.
const MapScreen = dynamic(() => import('./MapScreen'), {
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
        <span className="text-stone-400">Land:</span>
        <span className="text-emerald-300 text-right">{stats.land}</span>
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

// The stamina wheel beside the hero (third person only): hidden while full, a bright green ring that
// turns amber as it runs low and red while exhausted. Updated at the 5 Hz telemetry rate; the CSS
// transition fills the gaps between samples.
const STAMINA_R = 14;
const STAMINA_C = 2 * Math.PI * STAMINA_R;
const StaminaWheel = React.memo(function StaminaWheel({ statsStore }: { statsStore: StatsStore }) {
  const { percent, exhausted } = useHeroStamina(statsStore);
  if (percent === null) return null;
  const hidden = percent >= 100 && !exhausted;
  const colour = exhausted ? '#ff4a3d' : percent < 30 ? '#ffc53d' : '#2ee66b';
  // Beside the hero's head: the orbit camera frames the chest at the screen
  // centre with ~6.4 m of height in view, so the offsets scale with the viewport.
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute left-[calc(50%+5vh)] top-[calc(50%-12vh)] z-10 transition-opacity duration-500 ${hidden ? 'opacity-0' : 'opacity-100'}`}
    >
      <svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="17" fill="rgba(8,14,12,0.42)" />
        <circle cx="20" cy="20" r={STAMINA_R} fill="none" stroke="rgba(255,255,255,0.14)" strokeWidth="4.5" />
        <circle
          cx="20"
          cy="20"
          r={STAMINA_R}
          fill="none"
          stroke={colour}
          strokeWidth="4.5"
          strokeLinecap="round"
          strokeDasharray={STAMINA_C}
          strokeDashoffset={STAMINA_C * (1 - percent / 100)}
          transform="rotate(-90 20 20)"
          style={{ transition: 'stroke-dashoffset 200ms linear, stroke 200ms' }}
        />
        <circle cx="20" cy="20" r="3.2" fill="rgba(255,255,255,0.55)" />
      </svg>
    </div>
  );
});

// The time of day beside the engine badge, while the day–night sky is drawn.
const SkyClock = React.memo(function SkyClock({ statsStore }: { statsStore: StatsStore }) {
  const s = useSkyState(statsStore);
  if (!s || s.sky !== 'cycle') return null;
  const hour = s.minutes / 60;
  const phase = dayPhase(hour);
  const Icon = phase === 'Night' || phase === 'Dusk' ? Moon
    : phase === 'Dawn' || phase === 'Sunrise' || phase === 'Morning glow' ? Sunrise
      : phase === 'Golden hour' || phase === 'Sunset' ? Sunset : Sun;
  return (
    <span className="hidden sm:flex h-8 items-center gap-1.5 px-2.5 rounded-md bg-stone-950/80 backdrop-blur border border-stone-800 font-mono text-xs text-stone-200">
      <Icon className="size-3.5 text-amber-300" />
      <span className="tabular-nums">{clockText5(hour)}</span>
      <span className="text-stone-500">{phase}</span>
      {s.paused && <Pause className="size-3 text-stone-500" aria-label="Time stopped" />}
    </span>
  );
});

// ?sky=classic|cycle, ?time=18.5 or ?time=18:30 and ?spawn=sky start the page there (read in the browser only).
function urlParam(name: string): string | null {
  return typeof window === 'undefined' ? null : new URLSearchParams(window.location.search).get(name);
}
function urlSky(): SkyMode {
  return urlParam('sky') === 'classic' ? 'classic' : 'cycle';
}
// ?spawn=sky starts the hero up on the nearest sky archipelago's hub.
function urlSpawn(): SpawnAt {
  return urlParam('spawn') === 'sky' ? 'sky' : 'land';
}
// ?foes=0: a peaceful world (no Bokoblin camps).
function urlFoes(): boolean {
  return urlParam('foes') !== '0';
}
// ?kit=1: start with a full pouch (one of every weapon, shield and bow, and some of every food).
function urlKit(): boolean {
  return urlParam('kit') === '1';
}
function urlHour(): number {
  const v = urlParam('time');
  if (!v) return 15.5;
  const [h, m] = v.split(':').map(Number);
  const hour = h + (m || 0) / 60;
  return Number.isFinite(hour) ? ((hour % 24) + 24) % 24 : 15.5;
}

// The name and hero last played here.
function savedName(): string {
  try {
    return typeof window === 'undefined' ? '' : localStorage.getItem('hyrule.name') ?? '';
  } catch {
    return '';
  }
}
function savedCharacter(): number {
  try {
    const c = typeof window === 'undefined' ? 0 : Number(localStorage.getItem('hyrule.character') ?? 0);
    return Number.isInteger(c) && c >= 0 && c < CHARACTERS.length ? c : 0;
  } catch {
    return 0;
  }
}
const noSubscribe = () => () => {};

const MODE_LABEL: Record<CamMode, string> = { hero: 'HERO', walk: 'WALK', fly: 'FLY' };

function ControlsBanner({ mode, isPointerLocked, timeKey, bindings }: { mode: CamMode | null; isPointerLocked: boolean; timeKey: boolean; bindings: Bindings }) {
  const k = (id: ActionId) => keyHint(bindings, id);
  const keys: Array<[string, string]> =
    mode === 'fly'
      ? [[moveKeys(bindings), 'Move'], [lookKeys(bindings), 'Look'], [k('sprint'), 'Boost'], [k('jump'), 'Up'], [k('crouch'), 'Down'], [k('toggleMode'), 'Land'], [k('toggleView'), 'Hero'], [k('map'), 'Map'], [k('respawn'), 'Respawn']]
      : mode === 'walk'
        ? [[moveKeys(bindings), 'Move'], [lookKeys(bindings), 'Look'], [k('sprint'), 'Sprint'], [k('jump'), 'Jump'], [k('toggleView'), 'Third person'], [k('toggleMode'), 'Fly'], [k('map'), 'Map'], [k('respawn'), 'Respawn']]
        : [[moveKeys(bindings), 'Move'], [lookKeys(bindings), 'Look'], [k('sprint'), 'Sprint'], [k('jump'), 'Jump · Glide'], [k('crouch'), 'Sneak'], ['LMB', 'Attack (hold: spin)'], ['RMB', 'Shield · Lock on'],
          [k('aim'), 'Bow'], [k('throw'), 'Throw'], [k('interact'), 'Action · Surf · Mount'], [k('whistle'), 'Whistle'], [k('swap'), 'Swap'], [k('inventory'), 'Inventory'], [k('map'), 'Map'], [k('toggleView'), 'First person'], [k('toggleMode'), 'Fly'], [k('respawn'), 'Respawn']];
  if (timeKey) keys.push([k('fastTime'), 'Time ≫']);
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 px-4 py-2 rounded-full border border-stone-800/80 bg-stone-950/80 backdrop-blur-md font-mono text-[11px] text-stone-300 shadow-xl">
      {/* (a control with no key has nothing to press) */}
      {keys.filter(([key]) => key).map(([key, what]) => (
        <React.Fragment key={what}>
          <span><strong className="text-emerald-400">{key}</strong> {what}</span>
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
  const { seed, seedText, onSeedChange, randomizeSeed, setSeed, setSeedText } = useSeedState(DEFAULT_SEED);
  const [preset, setPreset] = useState<PresetId>('med');
  const [showSettings, setShowSettings] = useState(false);
  const [showInventory, setShowInventory] = useState(false);
  const [showMap, setShowMap] = useState(false);
  const [showHud, setShowHud] = useState(true);
  // The HUD in three steps (the + and − keys): 2 everything, 1 the essentials (the hearts, the minimap with its
  // temperature and sound gauges, the clock), which is where it starts, 0 nothing.
  const [hudLevel, setHudLevel] = useState<0 | 1 | 2>(1);
  const hudFull = hudLevel === 2;
  const hudOn = hudLevel >= 1;
  const [isPointerLocked, setIsPointerLocked] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [fov, setFov] = useState<number>(CAMERA.fov);
  const [sensitivity, setSensitivity] = useState<number>(CAMERA.sensitivity);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);
  const [sky, setSky] = useState<SkyMode>(urlSky);
  const [clockStart] = useState(() => ({ hour: urlHour(), dayMinutes: 24 }));
  const [spawnAt] = useState(urlSpawn);
  const [foes] = useState(urlFoes);
  const [kit] = useState(urlKit);
  // Which key does what (changed in Engine Settings → Controls, saved in the browser).
  const [{ bindings, keymap }, setBindings] = useKeybinds();

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Multiplayer: the lobby first (a name and one of the five heroes), then the world, shared with
  // whoever else is on this server (server.mjs). ?solo=1 goes straight in, alone (benchmarks).
  const [net] = useState(() => new NetSession());
  const netView = useNetView(net);
  const mounted = useSyncExternalStore(noSubscribe, () => true, () => false);
  const [solo] = useState(() => urlParam('solo') === '1');
  const [entered, setEntered] = useState(false);
  // (refused on entering, or on entering again after a lost connection: back to the lobby)
  const phase = !mounted ? 'init' : solo || (entered && !netView.denied) ? 'play' : 'lobby';
  const [picked, setPicked] = useState(savedCharacter);
  const [playerName, setPlayerName] = useState(savedName);
  // The hero picked, unless someone else in the world plays it by now: then the first one free.
  const isFree = (c: number) => !netView.players.some((p) => p.char === c && p.id !== netView.selfId);
  const character = isFree(picked) ? picked : (CHARACTERS.find((c) => isFree(c.id))?.id ?? picked);
  const netSeed = useRef<number | null>(null); // the last seed the world set (not to be sent back)

  // The inventory screen (the inventory key, or its button) and the map (the map key, or the minimap): behind the lobby there are none,
  // and one is open at a time.
  const inventoryOpen = showInventory && phase === 'play' && !showMap;
  const mapOpen = showMap && phase === 'play' && !showInventory;

  const { statsStore, combatHud, pouch, pouchCommand, mapHud, error, respawn, travel, toggleMode, toggleView, setHour, setTimePaused, setDayMinutes, clock } = useTerrainEngine({
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
    sky,
    clockStart,
    spawnAt,
    inputBlocked: phase !== 'play',
    character,
    net,
    foes,
    kit,
    keymap,
    menuOpen: inventoryOpen || mapOpen,
    onPointerLockChange: setIsPointerLocked,
    onToggleHud: () => setShowHud((v) => !v),
    onHudStep: (d) => setHudLevel((l) => Math.max(0, Math.min(2, l + d)) as 0 | 1 | 2),
    onToggleInventory: () => { if (!showMap) setShowInventory((v) => !v); },
    onToggleMap: () => { if (!showInventory) setShowMap((v) => !v); },
  });
  const closeInventory = () => {
    setShowInventory(false);
    canvasRef.current?.focus();
  };
  const closeMap = () => {
    setShowMap(false);
    canvasRef.current?.focus();
  };
  const mode = useEngineMode(statsStore);
  const api = useEngineApi(statsStore);

  // The session (none with ?solo=1): who is in the world, and the world's seed.
  useEffect(() => {
    if (solo) return;
    const unlisten = net.listenSeed((s) => {
      netSeed.current = s;
      setSeed(s);
      setSeedText(String(s));
    });
    net.start();
    return () => {
      unlisten();
      net.stop();
    };
  }, [net, solo, setSeed, setSeedText]);

  const enterWorld = () => {
    const name = playerName.trim() || `${CHARACTERS[character].name} hero`;
    try {
      localStorage.setItem('hyrule.name', playerName.trim());
      localStorage.setItem('hyrule.character', String(character));
    } catch { /* storage blocked */ }
    setPicked(character);
    net.join(name, character, seed, clock());
    setEntered(true);
    canvasRef.current?.focus();
  };

  // In the world: everyone starts at the spawn, side by side in character order — on entering, and
  // whenever the world's seed changes (whoever changed it: everyone moves to the new world together).
  const onWorldChange = useEffectEvent(() => respawn());
  useEffect(() => {
    if (netView.inWorld) onWorldChange();
  }, [netView.inWorld, seed]);
  // A seed changed here goes to everyone.
  useEffect(() => {
    if (netView.inWorld && seed !== netSeed.current) {
      netSeed.current = seed;
      net.sendSeed(seed);
    }
  }, [net, netView.inWorld, seed]);

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
        <div className={`absolute top-4 left-4 z-20 flex flex-col gap-2 pointer-events-auto max-w-[calc(100vw-32px)] ${inventoryOpen ? 'invisible' : ''}`}>
          {mode === 'hero' && phase === 'play' && hudOn && <Hearts store={combatHud} />}
          <div className={`flex items-center gap-2 ${hudFull ? '' : 'hidden'}`}>
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
            <SkyClock statsStore={statsStore} />
          </div>

          <HudOverlay statsStore={statsStore} preset={preset} showHud={showHud && hudFull} />
          {phase === 'play' && hudFull && <PlayersPanel view={netView} api={api} />}
        </div>

        {hudOn && <StaminaWheel statsStore={statsStore} />}
        {phase === 'play' && hudOn && !inventoryOpen && !mapOpen && <MapHud statsStore={statsStore} mapHud={mapHud} onOpen={() => setShowMap(true)} />}
        <CombatHud store={combatHud} show={mode === 'hero' && phase === 'play' && !inventoryOpen && hudOn} full={hudFull} />

        {/* Top Right Quick Controls */}
        <div className={`absolute top-4 right-4 z-20 flex items-center gap-1.5 sm:gap-2 pointer-events-auto ${inventoryOpen ? 'invisible' : ''} ${hudFull ? '' : 'hidden'}`}>
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
              {mode === 'fly' ? 'Land' : 'Take off: Flight Mode'}{keyNote(bindings, 'toggleMode')}
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
              {mode === 'hero' ? 'First Person' : 'Third Person Hero'}{keyNote(bindings, 'toggleView')}
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
              Respawn at Spawn Point{keyNote(bindings, 'respawn')}
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                aria-label="Open Map"
                data-action="map"
                disabled={phase !== 'play'}
                onClick={() => setShowMap(true)}
                className="h-8 w-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900 hover:text-emerald-400"
              >
                <MapIcon className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              Map{keyNote(bindings, 'map')}
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                aria-label="Open Inventory"
                data-action="inventory"
                disabled={mode !== 'hero' || phase !== 'play'}
                onClick={() => setShowInventory(true)}
                className="h-8 w-8 border-stone-800 bg-stone-950/80 backdrop-blur text-stone-300 hover:bg-stone-900 hover:text-amber-300"
              >
                <Backpack className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent className="bg-stone-900 border-stone-700 font-mono text-xs">
              Inventory{keyNote(bindings, 'inventory')}
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
        {showSettings && hudFull && (
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
            skyControls={{ statsStore, api, sky, setSky, setHour, setTimePaused, setDayMinutes, travel }}
            bindings={bindings}
            setBindings={setBindings}
          />
        )}

        {/* Bottom Center Controls Banner */}
        <div className={`absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none w-max max-w-[calc(100vw-32px)] md:max-w-[calc(100vw-600px)] ${inventoryOpen ? 'invisible' : ''} ${hudFull ? '' : 'hidden'}`}>
          <ControlsBanner mode={mode} isPointerLocked={isPointerLocked} timeKey={api === 'webgpu' && sky === 'cycle'} bindings={bindings} />
        </div>

        {inventoryOpen && <InventoryScreen store={pouch} command={pouchCommand} onClose={closeInventory} bindings={bindings} />}
        {mapOpen && <MapScreen mapHud={mapHud} selfName={playerName.trim() || `${CHARACTERS[character].name} hero`} selfChar={character} onClose={closeMap} bindings={bindings} />}

        {phase === 'lobby' && (
          <Lobby
            view={netView}
            name={playerName}
            setName={setPlayerName}
            character={character}
            setCharacter={setPicked}
            onEnter={enterWorld}
            api={api}
          />
        )}

        {/* Pointer lock hint on canvas initial click */}
        {!isPointerLocked && phase === 'play' && !inventoryOpen && hudFull && (
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
