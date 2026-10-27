/**
 * §23–§31 — engine core: the 60 Hz loop (input, physics twin, floating origin,
 * DRS, telemetry) and all listeners, drawing through a View: WebGL2 (context,
 * shader programs, uniforms, frame fence) or, when the page supplies one, the
 * WebGPU backend (src/lib/gpu/engineGpu.ts). Loaded on demand by
 * useTerrainEngine via dynamic import(), so the math twin and the 36 KB shader
 * source are a separate chunk that never delays hydration of the page around
 * the canvas (Phase 5).
 *
 * Camera modes: 'hero' (third-person character, src/lib/hero/), 'walk' (first
 * person, updatePlayer) and 'fly'. V switches between the two views on foot,
 * F takes off / lands in the view last used on foot. On WebGL2, first person
 * and fly draw with the terrain-only program; third person with
 * withHero(TERRAIN_FS), compiled in the background
 * (KHR_parallel_shader_compile) so neither pays for the other.
 *
 * Time of day: the loop runs the sky's clock (src/lib/sky/clock.ts) and hands
 * the WebGPU view the game time each frame; holding T runs it a game hour a
 * second. The page picks the sky (the day–night cycle or the classic sky) and
 * sets the hour, pause and day length through the handle.
 *
 * Combat (src/lib/combat/, third person): the hero's locomotion runs inside
 * the combat world, which takes the mouse's buttons (attack, shield), Q (bow),
 * G (throw), E (action) and X (swap) besides the movement keys; the camera
 * turns toward a foe the shield is locked onto and eases over the shoulder to
 * aim; the world's events play sounds, shake the view and feed the combat HUD
 * (combat/hud.ts) with its markers. Foes are drawn by the WebGPU view only, so
 * the WebGL2 view's world has none. In multiplayer the other players' heroes,
 * played back, are the combat world's rivals (combat/pvp.ts): before each step
 * it is handed them and their blows and shots, after it its own go out.
 *
 * The inventory (combat/pouch.ts, the page's InventoryScreen): its key asks the page to open the
 * screen, and while it is open the keys and the mouse's buttons leave the hero alone, the hands let
 * go of what they held and the pointer is freed. Alone in the world, everything stands still behind
 * it (the clock, the foes, what flies, the hero) but the pose, which follows what is carried; in a
 * shared world it goes on, as behind the lobby. The screen's commands (equip, unequip, drop, eat)
 * come through the handle.
 *
 * Keys: the keys named in this file are the defaults. Every control is an
 * action with up to two keys the player may rebind (keybinds.ts); the page
 * hands the engine the compiled keymap, and a change takes hold at once. The
 * look keys (none until Key Look is turned on) turn the camera as the mouse
 * does, at a speed that follows the Mouse Look setting (keyLook.ts).
 */

import { HydroTexture } from './hydroTexture';
import { assertNever } from '@/lib/utils';
import type { CamMode, RenderApi, StatsStore } from './statsStore';
import { TERRAIN_FS, TERRAIN_VS, minifyGlsl, withHero } from '@/components/terrain/demoShader';
import { elevation, updatePlayer, type PlayerState, type InputState } from '@/lib/terrainMath';
import { DRS_HARDWARE, DRS_SOFTWARE, createDrsState, drsStep, updateFpsEma } from './drs';
import { createHeroState, firstPersonGround, heroAction, placeHero, type HeroState } from '@/lib/hero/locomotion';
import { createHeroRig, heroLocalToWorld, packHeroUniforms, updateHeroRig, HERO_UNIFORM } from '@/lib/hero/rig';
import { createRockFrame, focusRocks, freeRocks, packRocks, setRocksOn, type RockFrame } from '@/lib/rocks/field';
import { Horses, createHorseFrame, createHorseInput, type Horse, type HorseFrame } from '@/lib/horse/horses';
import { loadHorse, loadModel } from '@/lib/horse/model';
import { Colossi, COLOSSUS_MSG, COLOSSUS_URL, createColossusFrame, type Colossus, type ColossusFrame } from '@/lib/colossus/colossi';
import { createHeroCamera, updateHeroCamera } from '@/lib/hero/camera';
import { exitOf, portalCrossed, portalOf, spawnPoint } from '@/lib/portal';
import { skySpawn } from '@/lib/skyworld/realm';
import { PortalTrip } from '@/lib/portalTrip';
import { worldSeed, type World } from '@/lib/world';
import type { GpuBackend, SkyMode } from '@/lib/gpu/engineGpu';
import { advanceClock, createSkyClock, hourOf, setHour } from '@/lib/sky/clock';
import { landName } from '@/lib/sky/air';
import { MAX_REMOTE_HEROES, type FrameParams } from '@/lib/gpu/frame';
import { characterOf } from '@/lib/hero/characters';
import type { ClockState, NetLink } from '@/lib/net/link';
import { RemoteHeroes, SEND_INTERVAL, encodeSnapshot, type RemoteHero } from '@/lib/net/remoteHeroes';
import { ambientTemp, createNoiseMeter, heroNoise, landingLoudness, noisePulse, stepNoise, walkNoise } from '@/lib/map/gauges';
import { MAP_PEERS, type MapHudState } from '@/lib/map/mapHud';
import { CombatWorld, createCombatInput, type CombatEvent, type Pickup } from '@/lib/combat/world';
import { rivalJoint } from '@/lib/combat/pvp';
import { CombatScene } from '@/lib/combat/scene';
import { CombatSfx } from '@/lib/combat/sfx';
import { FOE_KINDS, foeJoint } from '@/lib/combat/foes';
import { BOWS, SHIELDS, WEAPONS, itemName } from '@/lib/combat/weapons';
import type { CombatHudState, CombatHudStore } from '@/lib/combat/hud';
import { readPouch, type PouchCommand, type PouchStore } from '@/lib/combat/pouch';
import type { CombatFrame } from '@/lib/gpu/frame';
import { J } from '@/lib/hero/rig';
import { DEFAULT_KEYMAP, HELD, heldMask, normalizeCode, type Keymap } from './keybinds';
import { createKeyLook, stepKeyLook } from './keyLook';
import {
  CHUNK_SIZE,
  FLY_CEILING,
  SPAWN,
  PLAYER,
  HERO_CAMERA,
  HORSE,
  COLOSSUS,
  QUALITY_PRESETS,
  type PresetId,
} from '@/lib/constants';

type CamState = {
  yaw: number;   // orbit angles in third person, view angles otherwise
  pitch: number;
  mode: CamMode;
  onFoot: 'hero' | 'walk'; // the view F lands in
  fly: { x: number; y: number; z: number };
  flyVel: { x: number; y: number; z: number };
  player: PlayerState;
  hero: HeroState;
};

type Ref<T> = { current: T };

/** Live values the React side updates between frames (read every frame). */
export type EngineInputs = {
  seed: Ref<number>;
  preset: Ref<PresetId>;
  running: Ref<boolean>;
  fov: Ref<number>;
  sensitivity: Ref<number>;
  speedMultiplier: Ref<number>;
  onToggleHud: Ref<(() => void) | undefined>;
  /** The HUD one step more (+1: the essentials, then everything) or less (−1: the essentials, then nothing). */
  onHudStep: Ref<((delta: 1 | -1) => void) | undefined>;
  onPointerLockChange: Ref<((locked: boolean) => void) | undefined>;
  /** The sky wanted (drawn by the WebGPU view; the WebGL2 view has only the classic sky). */
  sky: Ref<SkyMode>;
  /** The keys do nothing while this is true (the multiplayer lobby is open). */
  inputBlocked?: Ref<boolean>;
  /** The hero's character (src/lib/hero/characters.ts): its dress, and its place in the spawn line-up. */
  character?: Ref<number>;
  /** Multiplayer: the session the hero is sent through and other players' heroes come from. */
  net?: Ref<NetLink | null>;
  /** Foes wake round the hero (false: a peaceful world, e.g. for benchmarks). */
  foes?: Ref<boolean>;
  /** The combat HUD's channel (hearts, weapon, arrows, markers). */
  combatHud?: CombatHudStore | null;
  /** Which action each key presses (keybinds.ts); the default keys when absent. */
  keymap?: Ref<Keymap>;
  /** The inventory screen is open: the hero is left alone (and alone in the world, the world stands still). */
  menuOpen?: Ref<boolean>;
  /** The inventory key was pressed: the page opens or closes the screen. */
  onToggleInventory?: Ref<(() => void) | undefined>;
  /** The inventory's channel (what the screen lists while it is open). */
  pouch?: PouchStore | null;
  /** Start with a full pouch: one of every weapon, shield and bow, and some of every food (?kit=1). */
  kit?: Ref<boolean>;
  /** The map's channel (where the hero and the other players are, the temperature, the noise: map/mapHud.ts). */
  mapHud?: MapHudState | null;
  /** The map key was pressed: the page opens or closes the map. */
  onToggleMap?: Ref<(() => void) | undefined>;
};

/** Where the day starts: the hour, and real minutes per game day. */
export type ClockStart = { hour: number; dayMinutes: number };

/** Where the game starts: the spawn on the land, or up on the nearest sky archipelago's hub. */
export type SpawnAt = 'land' | 'sky';

export type EngineHandle = {
  dispose: () => void;
  respawn: () => void;
  toggleMode: () => void;
  toggleView: () => void;
  /** Time of day: jump to an hour (0–24), stop or run the clock, set real minutes per game day. */
  setHour: (hour: number) => void;
  setTimePaused: (paused: boolean) => void;
  setDayMinutes: (minutes: number) => void;
  /** The sky clock now (multiplayer: an empty world takes the first player's). */
  clock: () => ClockState;
  /** An inventory command (equip, unequip, drop, eat); whether it did anything. */
  pouch: (cmd: PouchCommand) => boolean;
  /** Go to the other world as through the portal (the settings' world switch, for inspection). */
  travel: () => void;
};

/** Game hours per real second while T is held. */
const FAST_TIME = 1;

/** Metres between characters side by side at the spawn (multiplayer). */
const SPAWN_GAP = 1.6;
const NO_BEASTS: readonly Horse[] = [];
const NO_REMOTES: readonly Float32Array[] = [];
const NO_RIVALS: readonly RemoteHero[] = [];

/** What the loop hands the view each frame: one object, refilled. */
type DrawState = {
  time: number;          // seconds since start
  fov: number;           // degrees
  seed: number;
  marchSteps: number;
  shadowSteps: number;
  yaw: number;
  pitch: number;
  basis: Float32Array;   // view → world, columns right / up / forward
  basisDirty: boolean;   // set when yaw / pitch change, cleared once uploaded
  x: number;             // camera, world metres
  y: number;
  z: number;
  lx: number;            // chunk-local camera and chunk origin
  lz: number;
  ox: number;
  oz: number;
  thirdPerson: boolean;
  /** Internal resolution over canvas resolution (views that upscale; 1 otherwise). */
  renderScale: number;
  /** Game days since day 0, 00:00 (the sky's clock). */
  days: number;
  /** Pack the hero for this camera (rig.ts HERO_UNIFORM) and return the buffer. */
  packHero: () => Float32Array;
  /** Pack the other players' heroes for this camera (multiplayer; none without). */
  packRemotes: () => readonly Float32Array[];
  /** Pack combat (the foes, what flies and flares) for this camera; null when there is none to draw. */
  packCombat: () => CombatFrame | null;
  /** The rocks the CPU owns round the hero (rocks/field.ts), for the frame drawn from ds's camera. */
  packRocks: () => RockFrame | null;
  /** The horses round the hero (horse/horses.ts), for the frame drawn from ds's camera. */
  packHorses: () => HorseFrame | null;
  /** The colossi round the hero (colossus/colossi.ts), for the frame drawn from ds's camera. */
  packColossi: () => ColossusFrame | null;
};

type View = {
  api: RenderApi;
  software: boolean;
  /** The canvas keeps the display resolution and the view scales the scene into it. */
  upscales: boolean;
  /** The previous frame is still on the GPU: skip this one (input latency stays one frame). */
  busy(): boolean;
  draw(ds: DrawState): void;
  /** The sky drawn, and a request for another (views without a day–night sky: null, no-op). */
  sky(): SkyMode | null;
  setSky(mode: SkyMode): void;
  /** The world drawn, and a request for the other (views that draw only the land: null, no-op). */
  world(): World | null;
  setWorld(world: World): void;
  dispose(): void;
};

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error('Failed to create WebGL shader handle');
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(info ?? 'Shader compile failed');
  }
  return sh;
}

function linkProgram(gl: WebGL2RenderingContext, vs: WebGLShader, fs: WebGLShader): WebGLProgram {
  const prog = gl.createProgram();
  if (!prog) throw new Error('Failed to create WebGL program');
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  return prog;
}

function uniformLocations(gl: WebGL2RenderingContext, prog: WebGLProgram) {
  const U = (n: string) => gl.getUniformLocation(prog, n);
  return {
    res: U('u_res'),
    time: U('u_time'),
    cam: U('u_cam_pos'),
    origin: U('u_chunk_origin'),
    basis: U('u_basis'),
    tan: U('u_tan_half'),
    seed: U('u_seed'),
    march: U('u_march_steps'),
    shadow: U('u_shadow_steps'),
    foot: U('u_footprint'),
    heroJ: U('u_hero_j'),
    heroTorso: U('u_hero_torso'),
    heroHead: U('u_hero_head'),
    heroBound: U('u_hero_bound'),
    heroMisc: U('u_hero_misc'),
    heroDress: U('u_hero_dress'),
    heroGear: U('u_hero_gear'),
    hydro: U('u_hydro'),
    hydroHead: U('u_hydro_head'),
  };
}

/** The texture unit the rivers' node table is bound to. */
const HYDRO_UNIT = 1;

/** The WebGL2 view: terrain program, background-compiled hero program, cached uniforms, frame fence. */
function createGlView(canvas: HTMLCanvasElement, initialMode: CamMode): View {
  const gl = canvas.getContext('webgl2', {
    powerPreference: 'high-performance',
    antialias: false,
    depth: false,
    stencil: false,
    alpha: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    desynchronized: true,
  });
  if (!gl) throw new Error('WebGL2 is unavailable in this environment');

  let software = false;
  try {
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : '';
    software = /swiftshader|software|llvmpipe|basic/i.test(renderer);
  } catch {
    /* renderer info unavailable — assume hardware */
  }

  const vs = compileShader(gl, gl.VERTEX_SHADER, TERRAIN_VS);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, minifyGlsl(TERRAIN_FS));
  const prog = linkProgram(gl, vs, fs); // terrain only: first person, fly
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(prog) ?? 'Program link failed');
  }
  gl.useProgram(prog);
  const terrainU = uniformLocations(gl, prog);
  // the rivers' node table round the camera (landsGlsl.ts u_hydro)
  const hydro = new HydroTexture(gl);
  gl.activeTexture(gl.TEXTURE0 + HYDRO_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, hydro.tex);
  gl.activeTexture(gl.TEXTURE0);
  let hydroDirty = true;
  let u = terrainU;
  let activeProg: WebGLProgram = prog;
  let fence: WebGLSync | null = null;

  // The hero program compiles and links without a status query, so drivers
  // with KHR_parallel_shader_compile build it off the main thread while the
  // terrain program draws; it is checked (once) when third person needs it.
  // Started up front when the page opens in third person, else on first use.
  const parallel = gl.getExtension('KHR_parallel_shader_compile');
  let heroFs: WebGLShader | null = null;
  let heroProg: WebGLProgram | null = null; // terrain + hero: third person
  let heroU: ReturnType<typeof uniformLocations> | null = null;
  let heroStatus: 'none' | 'pending' | 'ready' | 'failed' = 'none';
  const compileHero = () => {
    heroStatus = 'pending';
    heroFs = gl.createShader(gl.FRAGMENT_SHADER);
    if (!heroFs) { heroStatus = 'failed'; return; }
    gl.shaderSource(heroFs, minifyGlsl(withHero(TERRAIN_FS)));
    gl.compileShader(heroFs);
    heroProg = linkProgram(gl, vs, heroFs);
  };
  if (initialMode === 'hero') compileHero();

  // Views of the packed hero buffer, made on first use (the buffer is reused).
  let heroViews: { j: Float32Array; torso: Float32Array; head: Float32Array; bound: Float32Array; misc: Float32Array; dress: Float32Array; gear: Float32Array } | null = null;

  let lastW = -1, lastH = -1;
  let lastFov = -1;
  let lastTanHalf = 0;
  let lastSeed = -999999;
  let lastMarch = -1;
  let lastShadow = -1;
  let lastCamX = NaN, lastCamY = NaN, lastCamZ = NaN;
  let lastOx = NaN, lastOz = NaN;

  return {
    api: 'webgl2',
    software,
    upscales: false,
    busy: () => fence !== null && gl.clientWaitSync(fence, 0, 0) === gl.TIMEOUT_EXPIRED,
    sky: () => null,
    setSky: () => {},
    world: () => null,
    setWorld: () => {},
    draw(ds) {
      // Third person draws with the hero program once it has compiled; until
      // then (a moment after start) the terrain program draws without the hero.
      let want: WebGLProgram = prog;
      if (ds.thirdPerson) {
        if (heroStatus === 'none') compileHero();
        if (heroStatus === 'pending' && heroProg && (!parallel || gl.getProgramParameter(heroProg, parallel.COMPLETION_STATUS_KHR))) {
          if (gl.getProgramParameter(heroProg, gl.LINK_STATUS)) {
            heroU = uniformLocations(gl, heroProg);
            heroStatus = 'ready';
          } else {
            heroStatus = 'failed';
            console.error('Hero shader failed to link:', gl.getProgramInfoLog(heroProg));
          }
        }
        if (heroStatus === 'ready' && heroProg) want = heroProg;
      }
      if (want !== activeProg) {
        // Uniforms are per program: switch, then re-upload everything once.
        activeProg = want;
        u = want === prog ? terrainU : heroU!;
        gl.useProgram(want);
        lastW = -1; lastH = -1; lastFov = -1; lastSeed = -999999; lastMarch = -1; lastShadow = -1;
        lastCamX = NaN; lastCamY = NaN; lastCamZ = NaN; lastOx = NaN; lastOz = NaN;
        ds.basisDirty = true;
        hydroDirty = true;
      }

      if (hydro.update(ds.x, ds.z, ds.seed)) hydroDirty = true;
      if (hydroDirty) {
        hydroDirty = false;
        gl.uniform1i(u.hydro, HYDRO_UNIT);
        gl.uniform4fv(u.hydroHead, hydro.head);
      }

      let tanHalf = lastTanHalf;
      let fovChanged = false;
      if (ds.fov !== lastFov) {
        lastFov = ds.fov;
        tanHalf = Math.tan((ds.fov * Math.PI / 180) * 0.5);
        lastTanHalf = tanHalf;
        gl.uniform1f(u.tan, tanHalf);
        fovChanged = true;
      }

      const sizeChanged = canvas.width !== lastW || canvas.height !== lastH;
      if (sizeChanged) {
        lastW = canvas.width;
        lastH = canvas.height;
        gl.viewport(0, 0, lastW, lastH);
        gl.uniform2f(u.res, lastW, lastH);
      }

      if (sizeChanged || fovChanged) {
        gl.uniform1f(u.foot, (2 * tanHalf) / Math.max(lastH, 2));
      }

      if (ds.seed !== lastSeed) {
        lastSeed = ds.seed;
        gl.uniform1i(u.seed, ds.seed | 0);
      }

      if (ds.marchSteps !== lastMarch) {
        lastMarch = ds.marchSteps;
        gl.uniform1i(u.march, ds.marchSteps);
      }

      if (ds.shadowSteps !== lastShadow) {
        lastShadow = ds.shadowSteps;
        gl.uniform1i(u.shadow, ds.shadowSteps);
      }

      gl.uniform1f(u.time, ds.time);

      if (ds.lx !== lastCamX || ds.y !== lastCamY || ds.lz !== lastCamZ) {
        lastCamX = ds.lx;
        lastCamY = ds.y;
        lastCamZ = ds.lz;
        gl.uniform3f(u.cam, ds.lx, ds.y, ds.lz);
      }

      if (ds.ox !== lastOx || ds.oz !== lastOz) {
        lastOx = ds.ox;
        lastOz = ds.oz;
        gl.uniform2f(u.origin, ds.ox, ds.oz);
      }

      if (ds.basisDirty) {
        ds.basisDirty = false;
        gl.uniformMatrix3fv(u.basis, false, ds.basis);
      }

      if (activeProg === heroProg) {
        const hb = ds.packHero();
        heroViews ??= {
          j: hb.subarray(HERO_UNIFORM.joints, HERO_UNIFORM.torso),
          torso: hb.subarray(HERO_UNIFORM.torso, HERO_UNIFORM.head),
          head: hb.subarray(HERO_UNIFORM.head, HERO_UNIFORM.bound),
          bound: hb.subarray(HERO_UNIFORM.bound, HERO_UNIFORM.misc),
          misc: hb.subarray(HERO_UNIFORM.misc, HERO_UNIFORM.dress),
          dress: hb.subarray(HERO_UNIFORM.dress, HERO_UNIFORM.dress + 3),
          gear: hb.subarray(HERO_UNIFORM.gear, HERO_UNIFORM.length),
        };
        gl.uniform3fv(u.heroJ, heroViews.j);
        gl.uniformMatrix3fv(u.heroTorso, false, heroViews.torso);
        gl.uniformMatrix3fv(u.heroHead, false, heroViews.head);
        gl.uniform4fv(u.heroBound, heroViews.bound);
        gl.uniform4fv(u.heroMisc, heroViews.misc);
        gl.uniform3fv(u.heroDress, heroViews.dress);
        gl.uniform4fv(u.heroGear, heroViews.gear);
      }

      gl.drawArrays(gl.TRIANGLES, 0, 3);

      if (fence) gl.deleteSync(fence);
      fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    },
    dispose() {
      if (fence) gl.deleteSync(fence);
      if (heroProg) {
        gl.detachShader(heroProg, vs);
        if (heroFs) {
          gl.detachShader(heroProg, heroFs);
          gl.deleteShader(heroFs);
        }
        gl.deleteProgram(heroProg);
      }
      gl.detachShader(prog, vs);
      gl.deleteShader(vs);
      gl.detachShader(prog, fs);
      gl.deleteShader(fs);
      gl.deleteProgram(prog);
    },
  };
}

/** The WebGPU view: the raster clipmap renderer (engineGpu.ts); it frames its own camera from yaw / pitch. */
function createGpuView(canvas: HTMLCanvasElement, gpu: GpuBackend): View {
  const fp: FrameParams = {
    x: 0, y: 0, z: 0, yaw: 0, pitch: 0, width: 2, height: 2, tanHalf: 1,
    seed: 0, marchSteps: 0, shadowSteps: 0, time: 0, hero: null, renderScale: 1, days: 0, combat: null,
  };
  let lastFov = -1;
  return {
    api: 'webgpu',
    software: false,
    upscales: gpu.upscales,
    busy: () => gpu.inFlight() >= 2,
    sky: () => gpu.sky(),
    setSky: (mode) => gpu.setSky(mode),
    world: () => gpu.world(),
    setWorld: (w) => gpu.setWorld(w),
    draw(ds) {
      if (ds.fov !== lastFov) {
        lastFov = ds.fov;
        fp.tanHalf = Math.tan((ds.fov * Math.PI / 180) * 0.5);
      }
      fp.x = ds.x; fp.y = ds.y; fp.z = ds.z;
      fp.yaw = ds.yaw; fp.pitch = ds.pitch;
      fp.width = canvas.width; fp.height = canvas.height;
      fp.seed = ds.seed; fp.marchSteps = ds.marchSteps; fp.shadowSteps = ds.shadowSteps;
      fp.time = ds.time;
      fp.days = ds.days;
      fp.renderScale = ds.renderScale;
      fp.hero = ds.thirdPerson ? ds.packHero() : null;
      fp.remotes = ds.packRemotes();
      fp.combat = ds.thirdPerson ? ds.packCombat() : null;
      fp.rocks = ds.packRocks();
      fp.horses = ds.packHorses();
      fp.colossi = ds.packColossi();
      ds.basisDirty = false;
      gpu.draw(fp);
    },
    dispose: () => gpu.dispose(),
  };
}

/**
 * Start the engine on `canvas`: WebGL2, or the given WebGPU backend (the
 * canvas then already carries its WebGPU context). Throws if WebGL2 or the
 * shader is unavailable.
 */
export function startEngine(
  canvas: HTMLCanvasElement,
  statsStore: StatsStore,
  inputs: EngineInputs,
  enablePointerLock: boolean,
  initialMode: CamMode = 'fly',
  gpu: GpuBackend | null = null,
  clockStart: ClockStart = { hour: 15.5, dayMinutes: 24 },
  spawnAt: SpawnAt = 'land',
): EngineHandle {
  const seedRef = inputs.seed;
  const presetRef = inputs.preset;
  const runRef = inputs.running;
  const fovRef = inputs.fov;
  const sensitivityRef = inputs.sensitivity;
  const speedMultiplierRef = inputs.speedMultiplier;
  const hudToggleRef = inputs.onToggleHud;
  const hudStepRef = inputs.onHudStep;
  const onPointerLockChangeRef = inputs.onPointerLockChange;
  const skyRef = inputs.sky;
  const charRef = inputs.character;
  const netRef = inputs.net;
  const menuRef = inputs.menuOpen;
  const inventoryToggleRef = inputs.onToggleInventory;
  const mapToggleRef = inputs.onToggleMap;
  const mapHud = inputs.mapHud ?? null;
  const pouchStore = inputs.pouch ?? null;
  const clock = createSkyClock(clockStart.hour, clockStart.dayMinutes);
  let fastTime = false;
  // Multiplayer: the clock changed here (sent at the next frame), when the last snapshot / clock went.
  let clockDirty = false;
  let sentStateAt = -Infinity, sentClockAt = -Infinity;

  let raf = 0;
  let ro: ResizeObserver | null = null;
  // The keys: which action each presses (the settings rebind them: keybinds.ts), the keys held now, and the
  // actions polled every frame (HELD) that those make up, as bits.
  let keymap: Keymap = inputs.keymap?.current ?? DEFAULT_KEYMAP;
  const held = new Set<string>();
  let keyMask = 0;
  const keyLook = createKeyLook(); // (the look keys turn the camera: keyLook.ts)
  // Rebound since the last look: the keys held are read again, by the new bindings.
  const syncKeymap = () => {
    const km = inputs.keymap?.current ?? DEFAULT_KEYMAP;
    if (km === keymap) return;
    keymap = km;
    keyMask = heldMask(keymap, held);
  };
  // (a page-scrolling key never scrolls the page while the game has the keyboard)
  const SCROLL_KEYS = new Set(['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End']);

  try {
    const view = gpu ? createGpuView(canvas, gpu) : createGlView(canvas, initialMode);
    const software = view.software;
    const drsCfg = software ? DRS_SOFTWARE : DRS_HARDWARE;
    const auto = createDrsState(drsCfg);

    // Third-person hero: locomotion state, animation rig, orbit camera and the
    // packed uniforms (buffer created once; the loop only refills it).
    const rig = createHeroRig();
    const tp = createHeroCamera();
    const heroBuf = new Float32Array(HERO_UNIFORM.length);

    // The world the hero is in (the land; the inactive cloud world only through the settings' switch), and the
    // seed its generators see (world.ts): everything below asks seedNow(), never the user's seed itself.
    let world: World = 'land';
    const seedNow = () => worldSeed(seedRef.current, world);
    // The start: the world's spawn (portal.ts spawnPoint). In a multiplayer world the five characters
    // start side by side (SPAWN_GAP apart across the view).
    const spawnFor = () => {
      const sky = spawnAt === 'sky' && world === 'land' ? skySpawn(SPAWN.x, SPAWN.z, seedNow()) : null;
      if (sky) return sky;
      const p = spawnPoint(world, seedRef.current);
      const slot = netRef?.current?.inWorld() ? (charRef?.current ?? 0) - 2 : 0;
      const x = p.x + Math.cos(p.yaw) * slot * SPAWN_GAP, z = p.z - Math.sin(p.yaw) * slot * SPAWN_GAP;
      return { x, y: elevation(x, z, seedNow(), 0), z, yaw: p.yaw };
    };
    let spawn = spawnFor();

    const cam: CamState = {
      yaw: spawn.yaw,
      pitch: initialMode === 'hero' ? HERO_CAMERA.pitch : SPAWN.pitch,
      mode: initialMode,
      onFoot: initialMode === 'hero' ? 'hero' : 'walk',
      fly: { x: spawn.x, y: 0, z: spawn.z },
      flyVel: { x: 0, y: 0, z: 0 },
      player: {
        x: spawn.x,
        y: 0,
        z: spawn.z,
        vx: 0,
        vy: 0,
        vz: 0,
        grounded: false,
      },
      hero: createHeroState(spawn.x, spawn.z, spawn.yaw, seedNow()),
    };
    // (up in the sky the hero starts on the island's top, not on the land below)
    if (spawnAt === 'sky') placeHero(cam.hero, spawn.x, spawn.y + PLAYER.groundBias, spawn.z, 0, 0, 0, spawn.yaw, seedNow());

    cam.fly.y = spawn.y + 30;
    cam.player.y = cam.fly.y;
    if (initialMode === 'walk') {
      cam.player.y = cam.hero.y + PLAYER.height;
      cam.player.grounded = true;
    }

    const playerInput: InputState = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };

    // Combat: the world round the hero (its locomotion runs in there in third person), its view for
    // the renderer, its sounds, and what the HUD shows. Foes only where they can be drawn (WebGPU).
    const combat = new CombatWorld(cam.hero, seedNow());
    // (development builds: the hero, the camera and the fight on window.__hyrule, for scripted checks in a browser)
    if (process.env.NODE_ENV === 'development') (window as unknown as { __hyrule?: object }).__hyrule = { cam, combat, rig, freeRocks, get horses() { return horses; }, get colossi() { return colossi; } };
    if (inputs.kit?.current) combat.stock();
    const combatScene = new CombatScene();
    const sfx = new CombatSfx();
    const combatIn = createCombatInput();
    // Horses: the herds round the hero and the hero's own, ridden from the combat world's hero (WebGPU draws them).
    const horses = new Horses();
    const horseIn = createHorseInput();
    const horseFrame = createHorseFrame();
    if (gpu) loadHorse().then((m) => horses.setModel(m), (e) => console.warn('horses: not loaded —', e));
    // Colossi: Herod roaming the land (colossus/colossi.ts): its blows on the hero, the shaking ground.
    const colossi = new Colossi();
    const colossusFrame = createColossusFrame();
    if (gpu) loadModel(COLOSSUS_URL).then((m) => colossi.setModel(m), (e) => console.warn('colossi: not loaded —', e));
    // (the hero's blows, arrows and throws strike them all: combat.beasts, set each tick)
    const beasts: Array<Horse | Colossus> = [];
    let colossiSentAt = 0;
    combat.onBeast = (b, dmg, x, z) => (b.meets ? colossi.hurt(b as Colossus, dmg) : horses.hurt(b as Horse, dmg, x, z));
    const hud = inputs.combatHud ?? null;
    const noise = createNoiseMeter(); // (the sound meter: map/gauges.ts)
    let mouseAttack = false, mouseGuard = false;
    let shake = 0;                                      // the view's shake (m), dying away
    const alertUntil = new Float64Array(16);            // per foe slot: until when its "!" shows
    let banner: string | null = null, bannerId = 0, bannerUntil = 0;
    let hudAt = 0;
    let combatTime = 0;
    const MK = new Float64Array(3);
    // rocks: the CPU owns those round the hero (rocks/field.ts), a rock in the hands follows the rig
    const RK3 = new Float64Array(3);
    const rockFrame = createRockFrame();
    // (only the WebGPU view draws rocks: without it none stand in the way)
    setRocksOn(!!gpu);
    let stoneHint: string | null = null;

    const say = (text: string) => { banner = text; bannerId++; bannerUntil = combatTime + 1.6; };

    // The portal (portal.ts, portalTrip.ts): the hero's crossing of the ring (or standing by it) is found in
    // its step below; the trip itself steps every frame, whatever the view, and on arrival puts the hero (and
    // the other views' cameras) out of the other world's ring. Health, the pouch, the clock and the link go on
    // as they were (the fight follows the seed's change).
    const portalTrip = new PortalTrip();
    let crossed = false, lastFade = 0;
    let travelAsked = false; // (the settings' world switch: a trip without the ring)
    let seedSeen = seedRef.current;
    const arrive = (to: World) => {
      world = to;
      const e = exitOf(world, seedRef.current);
      placeHero(cam.hero, e.x, e.y + PLAYER.groundBias, e.z, 0, 0, 0, e.yaw, seedNow());
      cam.yaw = e.yaw;
      cam.fly = { x: e.x, y: e.y + 3, z: e.z };
      cam.flyVel = { x: 0, y: 0, z: 0 };
      cam.player = { x: e.x, y: e.y + PLAYER.height, z: e.z, vx: 0, vy: 0, vz: 0, grounded: false };
      rig.snap = true;
      tp.snap = true;
    };

    /** What happened in the fight this tick: sounds, the view's shake, banners, the foes' "!". */
    const combatEvents = () => {
      for (let k = 0; k < combat.eventCount; k++) {
        const e: CombatEvent = combat.events[k];
        sfx.play(e, tp.x, tp.y, tp.z, tp.yaw);
        switch (e.type) {
          case 'hit': shake = Math.max(shake, 0.05); noisePulse(noise, 0.7); break;
          case 'crit': shake = Math.max(shake, 0.07); noisePulse(noise, 0.8); say('Headshot'); break;
          case 'sneak': shake = Math.max(shake, 0.08); say('Sneakstrike'); break;
          case 'hurt': shake = Math.max(shake, 0.14); noisePulse(noise, 0.75); break;
          case 'block': shake = Math.max(shake, 0.06); noisePulse(noise, 0.8); break;
          case 'parry': shake = Math.max(shake, 0.08); noisePulse(noise, 0.9); say('Perfect Guard'); break;
          case 'dodge': if (e.a === 2) say('Perfect Dodge'); break;
          case 'flurry': say('Flurry Rush'); break;
          case 'break': noisePulse(noise, 0.9); say(`${itemName(e.a)} broke!`); break;
          case 'return': say('The Master Sword returned'); break;
          case 'pickup': if (e.a >= 0) say(`Got ${itemName(e.a)}`); break;
          case 'swap': say(WEAPONS[e.a].name); break;
          case 'empty':
            if (e.a === 1) say('No shield');
            else if (e.a === 2) say('No bow');
            else if (e.a === 3) say('No room for that');
            else if (combat.act === 'aim') say('No arrows');
            break;
          case 'alert': if (e.a >= 0 && e.a < alertUntil.length) alertUntil[e.a] = combatTime + 1.1; break;
          case 'parried': shake = Math.max(shake, 0.08); break;
          case 'defeat': {
            // (another player felled: `a` its place among the rivals)
            const r = combat.rivals[e.a];
            say(r && netRef?.current ? `Defeated ${netRef.current.peerName(r.id)}` : 'Defeated');
            break;
          }
        }
      }
      if (banner && combatTime > bannerUntil) banner = null;
    };

    /** What a lying weapon, shield or bow is called. */
    const pickName = (p: Pickup) => (p.kind === 'shield' ? SHIELDS : p.kind === 'bow' ? BOWS : WEAPONS)[p.weapon].name;

    /** The combat HUD's state (published when it changes, at most every 50 ms). */
    const publishCombatHud = (now: number) => {
      if (!hud) return;
      hudAt = now;
      const w = combat, c = w.carried();
      const kind = c ? WEAPONS[c.kind] : null;
      const sh = w.shieldOn >= 0 ? w.shields[w.shieldOn] : null, bw = w.bowOn >= 0 ? w.bows[w.bowOn] : null;
      const shk = sh ? SHIELDS[sh.kind] : null, bwk = bw ? BOWS[bw.kind] : null;
      const left = (g: { wear: number } | null, max: number) => (!g || g.wear === Infinity ? 1 : Math.round((g.wear / max) * 20) / 20);
      const pick = w.nearestPickup();
      const horsePrompt = world === 'land' ? horses.prompt(cam.hero) : null;
      // (the colossus fighting the hero, nearest first)
      let boss: Colossus | null = null;
      if (world === 'land') for (const C of colossi.list) {
        if (colossi.engaged(C) && Math.hypot(C.x - cam.hero.x, C.z - cam.hero.z) < COLOSSUS.forget
          && (!boss || Math.hypot(C.x - cam.hero.x, C.z - cam.hero.z) < Math.hypot(boss.x - cam.hero.x, boss.z - cam.hero.z))) boss = C;
      }
      const s: CombatHudState = {
        hearts: w.hearts,
        health: Math.ceil(w.health),
        weapon: kind ? kind.name : null,
        wear: !c || !kind || c.wear === Infinity ? 1 : Math.round((c.wear / kind.durability) * 20) / 20,
        weapons: w.weapons.length,
        weaponKind: c ? c.kind : -1,
        shield: shk ? shk.name : null,
        shieldKind: sh ? sh.kind : -1,
        shieldWear: left(sh, shk ? shk.durability : 1),
        bow: bwk ? bwk.name : null,
        bowKind: bw ? bw.kind : -1,
        bowWear: left(bw, bwk ? bwk.durability : 1),
        food: w.food.reduce((a, b) => a + b, 0),
        arrows: w.arrows,
        aiming: w.act === 'aim',
        draw: Math.round(w.draw * 10) / 10,
        charge: Math.round(w.charge * 10) / 10,
        prompt: horsePrompt ? horsePrompt : colossi.cling ? (colossi.cling.top ? 'Space|Let go' : 'Click|Stab  ·  Space  Let go') : w.flurryWait > 0 ? 'Click|Flurry Rush'
          : w.act === 'carry' ? 'Click|Throw  ·  E  Put down'
          : pick >= 0 ? `E|${w.canTake(pick) ? 'Pick up' : 'No room for'} ${pickName(w.pickups[pick])}`
            : (stoneHint = w.stoneHint()) ? `E|${stoneHint}`
            : cam.hero.surfing ? 'E|Step off the shield'
              : w.act === 'guard' && cam.hero.move === 'ground' ? (w.lockedBody() ? 'Space + A/D/S|Dodge' : 'E|Shield surf') : null,
        banner,
        bannerId,
        dead: w.act === 'dead',
        downBy: w.act === 'dead' && w.downBy !== null && netRef?.current ? netRef.current.peerName(w.downBy) : null,
        slowed: w.worldScale < 0.5 && w.hitStop <= 0,
        rush: w.flurryWait > 0,
        boss: boss ? COLOSSUS.name : null,
        bossHp: boss ? Math.ceil((boss.hp / COLOSSUS.hp) * 100) / 100 : 0,
        bossDown: boss?.mind === 'down',
      };
      hud.set(s);
    };

    /** A marker over the world at (x, y, z), if in view; false off screen. */
    const putMarker = (kind: 'lock' | 'bar' | 'alert' | 'notice', x: number, y: number, z: number, value: number): void => {
      const mk = hud!.markers;
      if (mk.count >= mk.list.length) return;
      const b = ds.basis;
      const rx0 = x - ds.x, ry0 = y - ds.y, rz0 = z - ds.z;
      const vz = rx0 * b[6] + ry0 * b[7] + rz0 * b[8];
      if (vz < 0.3) return;
      const W = canvas.clientWidth, H = canvas.clientHeight;
      const th = Math.tan((ds.fov * Math.PI) / 360);
      const sx = (rx0 * b[0] + ry0 * b[1] + rz0 * b[2]) / (vz * th * (W / Math.max(H, 1)));
      const sy = (rx0 * b[3] + ry0 * b[4] + rz0 * b[5]) / (vz * th);
      if (Math.abs(sx) > 1.1 || Math.abs(sy) > 1.1) return;
      const m = mk.list[mk.count++];
      m.kind = kind; m.x = (sx * 0.5 + 0.5) * W; m.y = (0.5 - sy * 0.5) * H; m.value = value;
      m.near = Math.max(0, Math.min(1, 1 - (vz - 3) / 35));
    };

    /** Where the foes' markers are this frame (lock-on, their hearts, "!" and "?"). */
    const placeMarkers = () => {
      if (!hud) return;
      const mk = hud.markers;
      mk.count = 0;
      mk.version++;
      if (cam.mode !== 'hero') return;
      for (const f of combat.foes) {
        if (!f.live || f.mind === 'dead') continue;
        const kind = FOE_KINDS[f.kind];
        foeJoint(f, J.head, MK);
        const d = Math.hypot(MK[0] - ds.x, MK[1] - ds.y, MK[2] - ds.z);
        if (d > 48) continue;
        const locked = combat.lock === f.slot;
        const top = MK[1] + 0.5 * kind.scale;
        if (alertUntil[f.slot] > combatTime) putMarker('alert', MK[0], top + 0.1, MK[2], 1);
        else if (!f.roused && f.aware > 0.1) putMarker('notice', MK[0], top + 0.1, MK[2], f.aware);
        if ((locked || f.hp < kind.hp) && d < 36) putMarker('bar', MK[0], top - 0.08, MK[2], f.hp / kind.hp);
        if (locked) {
          foeJoint(f, J.chest, MK);
          putMarker('lock', MK[0], MK[1], MK[2], 1);
        }
      }
      // horses: their health once hurt
      if (world === 'land') for (const H of horses.list) {
        if (H.hp <= 0 || H.hp >= HORSE.hp) continue;
        if (Math.hypot(H.x - ds.x, H.y + 2 - ds.y, H.z - ds.z) < 36) putMarker('bar', H.x, H.y + 2.25, H.z, H.hp / HORSE.hp);
      }
      // other players: their hearts once hurt (or locked onto), the lock-on reticle
      for (const r of combat.rivals) {
        if (r.pose.act === 'dead') continue;
        rivalJoint(r, J.head, MK);
        const d = Math.hypot(MK[0] - ds.x, MK[1] - ds.y, MK[2] - ds.z);
        if (d > 48) continue;
        const locked = combat.lockRival === r;
        const full = Math.max(1, r.hearts * 4);
        if ((locked || r.health < full) && d < 36) putMarker('bar', MK[0], MK[1] + 0.42, MK[2], r.health / full);
        if (locked) {
          rivalJoint(r, J.chest, MK);
          putMarker('lock', MK[0], MK[1], MK[2], 1);
        }
      }
    };

    /** Drop into third person with the hero's root (feet) at x, y, z. */
    const enterHero = (x: number, y: number, z: number, vx: number, vy: number, vz: number) => {
      placeHero(cam.hero, x, y, z, vx, vy, vz, cam.yaw, seedNow());
      cam.mode = 'hero';
      cam.onFoot = 'hero';
      cam.pitch = HERO_CAMERA.pitch;
      rig.snap = true;
      tp.snap = true;
    };

    const doRespawn = () => {
      const s = seedNow();
      spawn = spawnFor();
      const ground = spawn.y;
      const spawnY = ground + 30;
      cam.yaw = spawn.yaw;
      cam.pitch = SPAWN.pitch;
      cam.fly = { x: spawn.x, y: spawnY, z: spawn.z };
      cam.flyVel = { x: 0, y: 0, z: 0 };
      cam.player = {
        x: spawn.x,
        y: ground + 1.8,
        z: spawn.z,
        vx: 0,
        vy: 0,
        vz: 0,
        grounded: false,
      };
      placeHero(cam.hero, spawn.x, ground + PLAYER.groundBias, spawn.z, 0, 0, 0, spawn.yaw, s);
      cam.hero.stamina = 1;
      cam.hero.exhausted = false;
      if (combat.seed !== s) combat.reset(s);
      combat.revive();
      rig.snap = true;
      tp.snap = true;
      if (cam.mode === 'hero') cam.pitch = HERO_CAMERA.pitch;
    };

    // F: take off, or land in the view last used on foot.
    const doToggleMode = () => {
      switch (cam.mode) {
        case 'fly':
          if (cam.onFoot === 'hero') {
            enterHero(cam.fly.x, cam.fly.y - PLAYER.height, cam.fly.z, 0, 0, 0);
            break;
          }
          cam.mode = 'walk';
          cam.player = {
            x: cam.fly.x,
            y: cam.fly.y,
            z: cam.fly.z,
            vx: 0,
            vy: 0,
            vz: 0,
            grounded: false,
          };
          break;
        case 'walk':
          cam.mode = 'fly';
          cam.onFoot = 'walk';
          cam.fly = { x: cam.player.x, y: cam.player.y, z: cam.player.z };
          cam.flyVel = { x: 0, y: 0, z: 0 };
          break;
        case 'hero':
          // take off from where the third-person camera is, looking the same way
          cam.mode = 'fly';
          cam.onFoot = 'hero';
          cam.fly = { x: tp.x, y: tp.y, z: tp.z };
          cam.flyVel = { x: 0, y: 0, z: 0 };
          cam.yaw = tp.yaw;
          cam.pitch = tp.pitch;
          break;
        default:
          assertNever(cam.mode);
      }
    };

    // V: third person ⇄ first person (from fly: drop the hero here).
    const doToggleView = () => {
      switch (cam.mode) {
        case 'hero': {
          const h = cam.hero;
          const grounded = h.move === 'ground';
          cam.mode = 'walk';
          cam.onFoot = 'walk';
          cam.player = {
            x: h.x,
            y: h.y + PLAYER.height,
            z: h.z,
            vx: h.vx,
            vy: grounded ? 0 : h.vy,
            vz: h.vz,
            grounded,
          };
          cam.yaw = tp.yaw;
          cam.pitch = Math.max(-0.35, Math.min(0.35, tp.pitch));
          break;
        }
        case 'walk': {
          const pl = cam.player;
          enterHero(pl.x, pl.y - PLAYER.height, pl.z, pl.vx, pl.vy, pl.vz);
          break;
        }
        case 'fly':
          enterHero(cam.fly.x, cam.fly.y - PLAYER.height, cam.fly.z, cam.flyVel.x, cam.flyVel.y, cam.flyVel.z);
          break;
        default:
          assertNever(cam.mode);
      }
    };

    const onKey = (e: KeyboardEvent, down: boolean) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') {
        return;
      }
      // The inventory key opens and closes the screen (it still works while the screen holds the keys;
      // not behind the lobby, nor to open it when the hero is down).
      if (down && !e.repeat && !inputs.inputBlocked?.current && cam.mode === 'hero' && !e.metaKey && !e.ctrlKey) {
        syncKeymap();
        if (keymap.get(normalizeCode(e.code)) === 'inventory' && (menuRef?.current || combat.act !== 'dead')) {
          e.preventDefault();
          inventoryToggleRef?.current?.();
          return;
        }
      }
      // The map key opens and closes the map in every view (it still works while the map holds the keys).
      if (down && !e.repeat && !inputs.inputBlocked?.current && !e.metaKey && !e.ctrlKey) {
        syncKeymap();
        if (keymap.get(normalizeCode(e.code)) === 'map') {
          e.preventDefault();
          mapToggleRef?.current?.();
          return;
        }
      }
      if (inputs.inputBlocked?.current || menuRef?.current) {
        held.clear();
        keyMask = 0;
        fastTime = false;
        return;
      }

      syncKeymap();
      const code = normalizeCode(e.code);
      // (every key is tracked, so a release is never missed; those that press nothing held make no bit)
      if (down) held.add(code);
      else held.delete(code);
      keyMask = heldMask(keymap, held);

      const isInteracting =
        enablePointerLock ||
        document.pointerLockElement === canvas ||
        document.activeElement === canvas ||
        dragging;

      if (isInteracting && SCROLL_KEYS.has(code)) {
        e.preventDefault();
      }

      // The one-shot keys leave the browser's own shortcuts (Ctrl+R, Cmd+H…) alone.
      const plain = (!e.metaKey && !e.ctrlKey) || code === 'Control';
      switch (keymap.get(code)) {
        case 'toggleMode':
          if (down && !e.repeat) doToggleMode();
          return;
        case 'toggleView':
          if (down && !e.repeat && plain) doToggleView();
          return;
        case 'respawn':
          if (down && plain) doRespawn();
          return;
        case 'fastTime':
          if (plain) fastTime = down;
          return;
        case 'toggleHud':
          if (down && plain) hudToggleRef.current?.();
          return;
        case 'hudMore':
          if (down && !e.repeat && plain) hudStepRef.current?.(1);
          return;
        case 'hudLess':
          if (down && !e.repeat && plain) hudStepRef.current?.(-1);
          return;
        case 'mute':
          if (down && !e.repeat && plain) sfx.muted = !sfx.muted;
          return;
      }
    };

    const kd = (e: KeyboardEvent) => onKey(e, true);
    const ku = (e: KeyboardEvent) => onKey(e, false);

    let dragging = false;
    let lookAt = -Infinity;                    // when the view was last turned up or down by hand (ms)
    let pitchFloor: number = HERO_CAMERA.pitchMin; // the orbit's lowest pitch (lower while skydiving)
    let lastX = 0, lastY = 0;

    const pd = (e: PointerEvent) => {
      sfx.wake(); // (a gesture: the browser lets audio start now)
      if (enablePointerLock && document.pointerLockElement !== canvas) {
        // Refused in embedded frames and without a user gesture (the promise
        // rejects); dragging still looks around, so the refusal is ignored.
        const lock = canvas.requestPointerLock?.() as Promise<void> | undefined;
        lock?.catch?.(() => {});
      }
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      if (!enablePointerLock) {
        canvas.setPointerCapture(e.pointerId);
      }
    };

    const pu = (e?: PointerEvent) => {
      dragging = false;
      if (e && !enablePointerLock && canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
      }
    };

    const clearKeys = () => { held.clear(); keyMask = 0; fastTime = false; mouseAttack = false; mouseGuard = false; };

    // The mouse's buttons fight (once the pointer is locked: the click that locks it does nothing
    // else). mousedown, not pointerdown: a second button pressed while one is held is only a
    // pointermove to pointer events.
    const md = (e: MouseEvent) => {
      if (inputs.inputBlocked?.current || menuRef?.current || document.pointerLockElement !== canvas) return;
      if (e.button === 0) mouseAttack = true;
      else if (e.button === 2) mouseGuard = true;
    };
    const mu = (e: MouseEvent) => {
      if (e.button === 0) mouseAttack = false;
      else if (e.button === 2) mouseGuard = false;
    };
    const noMenu = (e: Event) => e.preventDefault();

    // Mouse wheel: third-person camera distance.
    const onWheel = (e: WheelEvent) => {
      if (cam.mode !== 'hero') return;
      tp.zoom = Math.max(HERO_CAMERA.minDistance, Math.min(HERO_CAMERA.maxDistance, tp.zoom * Math.exp(e.deltaY * 0.0015)));
    };

    const pm = (e: PointerEvent | MouseEvent) => {
      const sens = sensitivityRef.current;
      if (enablePointerLock && document.pointerLockElement === canvas) {
        const mx = (e as MouseEvent).movementX || 0;
        const my = (e as MouseEvent).movementY || 0;
        if (Math.abs(mx) > 250 || Math.abs(my) > 250) return;
        cam.yaw += mx * sens * 0.8;
        cam.pitch -= my * sens * 0.8;
        if (my !== 0) lookAt = performance.now();
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
      } else if (dragging) {
        cam.yaw += (e.clientX - lastX) * sens;
        cam.pitch -= (e.clientY - lastY) * sens;
        if (e.clientY !== lastY) lookAt = performance.now();
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
        lastX = e.clientX;
        lastY = e.clientY;
      }
    };

    const handlePointerLockChange = () => {
      const locked = document.pointerLockElement === canvas;
      if (!locked) dragging = false;
      onPointerLockChangeRef.current?.(locked);
    };

    if (enablePointerLock) {
      document.addEventListener('pointerlockchange', handlePointerLockChange);
    }

    window.addEventListener('keydown', kd);
    window.addEventListener('keyup', ku);
    window.addEventListener('blur', clearKeys);
    canvas.addEventListener('pointerdown', pd);
    window.addEventListener('pointerup', pu);
    window.addEventListener('pointermove', pm);
    canvas.addEventListener('pointercancel', pu);
    canvas.addEventListener('wheel', onWheel, { passive: true });
    canvas.addEventListener('mousedown', md);
    window.addEventListener('mouseup', mu);
    canvas.addEventListener('contextmenu', noMenu);

    const PRESET_MAP: Record<PresetId, (typeof QUALITY_PRESETS)[number]> = {
      low: QUALITY_PRESETS[0],
      med: QUALITY_PRESETS[1],
      high: QUALITY_PRESETS[2],
    };

    const applySize = () => {
      const p = PRESET_MAP[software ? 'low' : (presetRef.current ?? 'med')] ?? PRESET_MAP.med;
      const dpr = software ? 1 : Math.min(window.devicePixelRatio || 1, 2);
      // An upscaling view keeps the canvas at the display resolution; the preset
      // and DRS scale then only the internal resolution (ds.renderScale).
      const s = view.upscales ? 1 : p.resScale * auto.scale;
      const w = Math.max(2, Math.round(canvas.clientWidth * dpr * s));
      const h = Math.max(2, Math.round(canvas.clientHeight * dpr * s));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };

    ro = new ResizeObserver(applySize);
    ro.observe(canvas);
    applySize();

    let isIntersecting = true;
    let isTabVisible = typeof document !== 'undefined' ? !document.hidden : true;
    let io: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== 'undefined') {
      io = new IntersectionObserver(
        ([entry]) => {
          isIntersecting = entry.isIntersecting;
        },
        { threshold: 0.01 }
      );
      io.observe(canvas);
    }

    const handleVisibilityChange = () => {
      isTabVisible = typeof document !== 'undefined' ? !document.hidden : true;
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    const t0 = performance.now();
    let last = t0;
    let lastRender = t0;
    let fpsEma = 60;
    let statsAt = 0;
    let lastYaw = -9999, lastPitch = -9999;
    let lastPreset = presetRef.current;
    let fx = 0, fy = 0, fz = 1, rx = 1, rz = 0;

    const ds: DrawState = {
      time: 0, fov: 0, seed: 0, marchSteps: 0, shadowSteps: 0,
      yaw: 0, pitch: 0, basis: new Float32Array(9), basisDirty: true,
      x: 0, y: 0, z: 0, lx: 0, lz: 0, ox: 0, oz: 0,
      thirdPerson: false, renderScale: 1, days: clock.days,
      packHero: () => {
        packHeroUniforms(rig, cam.hero, ds.x, ds.y, ds.z, heroBuf, characterOf(charRef?.current ?? 0).dress);
        return heroBuf;
      },
      packRemotes: () => (netRef?.current ? remotes.pack(ds.x, ds.y, ds.z, MAX_REMOTE_HEROES) : NO_REMOTES),
      packCombat: () => (gpu ? combatScene.pack(combat, ds.x, ds.y, ds.z) : null),
      packRocks: () => packRocks(ds.x, ds.z, rockFrame),
      packHorses: () => (gpu && world === 'land' ? horses.pack(ds.x, ds.y, ds.z, horseFrame) : null),
      packColossi: () => (gpu && world === 'land' ? colossi.pack(ds.x, ds.y, ds.z, colossusFrame) : null),
    };

    // Multiplayer: other players' heroes, this player's snapshot (the hero, or the first-person /
    // flying camera as a hero standing or gliding there), and the shared clock.
    const remotes = new RemoteHeroes();
    const snap: number[] = [];
    const proxy = createHeroState(spawn.x, spawn.z, spawn.yaw, seedNow());
    // Before the step: the others played back for now (the combat world's rivals) and their blows and
    // shots (fought in third person only: the first-person and flying views take none).
    const takeCombat = (id: string, _at: number, d: readonly number[]) => {
      // (a colossus's state from its thinker, a player's blow on its eye: for every view)
      if (d[0] === COLOSSUS_MSG.state) colossi.receive(d);
      else if (d[0] === COLOSSUS_MSG.hit) colossi.struckBy(d[1], d[2]);
      else if (cam.mode === 'hero') combat.receive(id, d);
    };
    const netIn = (net: NetLink, now: number, dt: number) => {
      remotes.update(net, now, dt, world);
      combat.rivals = net.inWorld() ? remotes.list() : NO_RIVALS;
      net.drainCombat(takeCombat);
    };
    // After it: this player's snapshot and the clock.
    const netOut = (net: NetLink, now: number) => {
      if (!net.inWorld()) return;
      const c = net.takeClock();
      if (c) {
        clock.days = c.days;
        clock.paused = c.paused;
        clock.dayMinutes = c.dayMinutes;
        clockDirty = false;
      }
      if (now - sentStateAt >= SEND_INTERVAL) {
        sentStateAt = now;
        let h = cam.hero;
        if (cam.mode !== 'hero') {
          h = proxy;
          const fly = cam.mode === 'fly';
          const p = fly ? cam.fly : cam.player;
          h.x = p.x; h.y = p.y - PLAYER.height; h.z = p.z;
          if (fly) { h.vx = cam.flyVel.x; h.vy = cam.flyVel.y; h.vz = cam.flyVel.z; }
          else { h.vx = cam.player.vx; h.vy = cam.player.vy; h.vz = cam.player.vz; }
          h.yaw = cam.yaw;
          h.move = fly ? 'glide' : cam.player.grounded ? 'ground' : 'air';
          h.ground = fly || !cam.player.grounded ? h.y - 50 : h.y - PLAYER.groundBias;
          h.gx = 0; h.gz = 0;
        }
        net.sendState(cam.mode === 'hero' ? encodeSnapshot(h, now, snap, combat.pose, combat.health, combat.hearts, world) : encodeSnapshot(h, now, snap, undefined, undefined, undefined, world));
      }
      // The world's clock: when changed here (T held: a few times a second), and now and then from its keeper.
      if ((clockDirty && now - sentClockAt > 250) || (net.keepsClock() && now - sentClockAt > 2000)) {
        sentClockAt = now;
        clockDirty = false;
        net.sendClock({ days: clock.days, paused: clock.paused, dayMinutes: clock.dayMinutes });
      }
    };

    // Telemetry for the HUD at most every 200 ms; created once so the 60 Hz
    // loop allocates no closure per frame.
    const publishStats = (
      now: number, x: number, y: number, z: number, grounded: boolean,
      originX: number, originZ: number, resScale: number,
    ) => {
      statsAt = now;
      const land = landName(x, z, seedNow());
      if (mapHud) {
        const t = ambientTemp(x, y, z, seedNow(), hourOf(clock.days));
        // (eased, so crossing from one land to the next is not a jump)
        mapHud.temp = Math.abs(t - mapHud.temp) > 12 || !mapHud.live ? t : mapHud.temp + (t - mapHud.temp) * 0.4;
        mapHud.land = land;
      }
      statsStore.set({
        fps: Math.round(fpsEma),
        x, y, z,
        mode: cam.mode,
        grounded,
        originX,
        originZ,
        resPct: Math.round(resScale * auto.scale * 100),
        action: cam.mode === 'hero' ? heroAction(cam.hero) : null,
        stamina: cam.hero.stamina,
        exhausted: cam.hero.exhausted,
        api: view.api,
        sky: view.sky(),
        skyWanted: skyRef.current,
        hour: hourOf(clock.days),
        paused: clock.paused,
        dayMinutes: clock.dayMinutes,
        land,
        world,
      });
    };

    // The map's channel, every frame: where the hero is and which way the view faces, the other players
    // (as played back), and how loud the hero is.
    const feedMap = (m: MapHudState, dt: number, px: number, py: number, pz: number, silent: boolean) => {
      const hero = cam.mode === 'hero';
      m.live = true;
      m.seed = seedRef.current;
      m.world = world;
      m.mode = cam.mode;
      m.x = hero ? cam.hero.x : px;
      m.y = hero ? cam.hero.y : py;
      m.z = hero ? cam.hero.z : pz;
      m.heading = lastYaw;
      m.facing = hero ? cam.hero.yaw : lastYaw;
      let loud = 0;
      if (hero) {
        const h = cam.hero;
        if (h.landings !== noise.landings) {
          if (noise.landings >= 0) noisePulse(noise, landingLoudness(h.impact));
          noise.landings = h.landings;
        }
        loud = heroNoise(h);
      } else if (cam.mode === 'walk') {
        loud = walkNoise(Math.hypot(cam.player.vx, cam.player.vz), (keyMask & HELD.sprint) !== 0);
      }
      m.noise = stepNoise(noise, dt, silent ? 0 : loud);
      const net = netRef?.current;
      let n = 0;
      if (net && net.inWorld()) {
        const list = remotes.list();
        for (let k = 0; k < list.length && n < MAP_PEERS; k++) {
          const r = list[k], p = m.peers[n++];
          if (p.id !== r.id || p.name === 'Someone') { p.id = r.id; p.name = net.peerName(r.id); }
          p.char = r.char;
          p.x = r.body.x; p.y = r.body.y; p.z = r.body.z;
          p.yaw = r.body.yaw;
        }
      }
      m.peerCount = n;
    };

    const setBasis = (yaw: number, pitch: number) => {
      if (yaw === lastYaw && pitch === lastPitch) return;
      lastYaw = yaw;
      lastPitch = pitch;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      fx = sy * cp; fy = sp; fz = cy * cp;
      rx = cy; rz = -sy;
      const upx = fy * rz;
      const upy = fz * rx - fx * rz;
      const upz = -fy * rx;

      const basis = ds.basis;
      basis[0] = rx;   basis[1] = 0;   basis[2] = rz;
      basis[3] = upx;  basis[4] = upy; basis[5] = upz;
      basis[6] = fx;   basis[7] = fy;  basis[8] = fz;
      ds.yaw = yaw;
      ds.pitch = pitch;
      ds.basisDirty = true;
    };

    // The inventory screen opens: the keys and buttons are let go, so are the hands' acts, and the pointer is freed.
    let menuWas = false;
    let pouchAt = -Infinity;
    // (the shader's clock — the wind in the grass, the water — stands still with the world: the time it has stood)
    let frozenTotal = 0, frozenSince = -1;
    const onMenu = (open: boolean) => {
      if (open) {
        clearKeys();
        combat.cancelActs();
        if (document.pointerLockElement === canvas) document.exitPointerLock();
        pouchAt = -Infinity;
      } else if (enablePointerLock && document.pointerLockElement !== canvas) {
        (canvas.requestPointerLock?.() as Promise<void> | undefined)?.catch?.(() => {});
      }
    };

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const active = isIntersecting && isTabVisible && runRef.current;
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (!active) {
        lastRender = now;
        return;
      }

      // A new seed while in the cloud world: out of its ring (alone on the land the hero stays where it is,
      // as it always has; in a shared world the page respawns everyone)
      if (seedRef.current !== seedSeen) {
        seedSeen = seedRef.current;
        if (world === 'clouds') doRespawn();
      }
      const s = seedNow();
      const p = PRESET_MAP[software ? 'low' : (presetRef.current ?? 'med')] ?? PRESET_MAP.med;
      // The inventory screen: the hero is left alone; alone in the world, everything stands still behind it.
      const menu = !!menuRef?.current;
      if (menu !== menuWas) { menuWas = menu; onMenu(menu); }
      const blocked = menu || !!inputs.inputBlocked?.current;
      const frozen = menu && !netRef?.current?.inWorld();
      if (frozen) { if (frozenSince < 0) frozenSince = now; }
      else if (frozenSince >= 0) { frozenTotal += now - frozenSince; frozenSince = -1; }
      // The sky: the one wanted (the view builds it in the background), and the clock.
      if (view.sky() !== null && skyRef.current !== view.sky()) view.setSky(skyRef.current);
      if (frozen) { /* (the clock stands still) */ }
      else if (fastTime) {
        clock.days += (dt * FAST_TIME) / 24;
        clockDirty = true;
      } else advanceClock(clock, dt);
      // A new preset changes the render resolution now, not at the next DRS step.
      if (presetRef.current !== lastPreset) {
        lastPreset = presetRef.current;
        applySize();
      }

      // Locomotion update via zero-allocation bitmask
      syncKeymap();
      const fwdKey = ((keyMask & HELD.forward) ? 1 : 0) - ((keyMask & HELD.back) ? 1 : 0);
      const strKey = ((keyMask & HELD.right) ? 1 : 0) - ((keyMask & HELD.left) ? 1 : 0);
      const sprint = (keyMask & HELD.sprint) !== 0;
      const jump = (keyMask & HELD.jump) !== 0;
      const descend = (keyMask & HELD.crouch) !== 0;
      // The look keys turn the view, as the mouse does (behind the lobby they wait with the rest).
      if (!blocked) stepKeyLook(keyLook, keyMask, dt, sensitivityRef.current, cam);
      if (keyLook.dirY !== 0) lookAt = now;

      // Third person aims from its own eye after the hero moves (below).
      if (cam.mode !== 'hero') setBasis(cam.yaw, cam.pitch);

      const net = netRef?.current;
      if (net) netIn(net, now, dt);

      let px: number, py: number, pz: number, grounded = false;

      switch (cam.mode) {
      case 'hero': {
        // Skydiving, the view tips to look down the fall (−68°, −82° in a nosedive) unless the mouse or the
        // look keys moved it lately; the floor comes back up gently after.
        const sky = cam.hero.move === 'air' && cam.hero.skydive;
        if (sky && now - lookAt > 700) {
          cam.pitch += (-(68 + 14 * cam.hero.dive) * Math.PI / 180 - cam.pitch) * (1 - Math.exp(-1.6 * dt));
        }
        pitchFloor = sky ? -1.45 : Math.min(HERO_CAMERA.pitchMin, pitchFloor + 0.8 * dt);
        if (cam.pitch < pitchFloor) cam.pitch = pitchFloor;
        else if (cam.pitch > HERO_CAMERA.pitchMax) cam.pitch = HERO_CAMERA.pitchMax;
        if (combat.seed !== s) combat.reset(s);
        combat.foesOn = !!gpu && (inputs.foes?.current ?? true);
        // (behind the lobby the foes wait)
        combat.paused = blocked;
        // The hero moves and fights in the combat world (its locomotion runs in there).
        const ci = combatIn;
        ci.forward = fwdKey;
        ci.strafe = strKey;
        ci.sprint = sprint;
        ci.jump = jump;
        ci.sneak = (keyMask & HELD.crouch) !== 0;
        ci.attack = mouseAttack || (keyMask & HELD.attack) !== 0;
        ci.guard = mouseGuard || (keyMask & HELD.guard) !== 0;
        ci.aim = (keyMask & HELD.aim) !== 0;
        ci.throw = (keyMask & HELD.throw) !== 0;
        ci.action = (keyMask & HELD.interact) !== 0;
        ci.swap = (keyMask & HELD.swap) !== 0;
        ci.camYaw = cam.yaw;
        ci.camPitch = tp.pitch;
        // The horses first: mounting, the ridden horse carrying the hero (its own steps then stand aside), the herds.
        if (!frozen && world === 'land') {
          if (horses.seed !== s) horses.reset(s);
          const hi = horseIn;
          hi.forward = fwdKey; hi.strafe = strKey; hi.camYaw = cam.yaw;
          hi.sprint = sprint; hi.jump = jump; hi.sneak = ci.sneak;
          const down = combat.act === 'dead' || combat.act === 'thrown' || combat.act === 'rising';
          hi.action = !blocked && ci.action && !down; hi.whistle = !blocked && (keyMask & HELD.whistle) !== 0;
          horses.step(blocked ? 0 : dt, cam.hero, hi, hourOf(clock.days));
          if (horses.tookAction) ci.action = false;
          if (horses.event === 'tamed') say('Tamed!');
          else if (horses.event === 'thrown') {
            say('Thrown off');
            combat.throwHero(cam.hero.vx, cam.hero.vy, cam.hero.vz, rig.local);
          }
          if (horses.riding) {
            ci.forward = ci.strafe = 0;
            ci.sprint = ci.jump = ci.sneak = ci.attack = ci.guard = ci.aim = ci.throw = ci.swap = false;
          }
        }
        // (the view: last frame's eye and the way through the screen's centre, for the bow and throws)
        ci.eyeX = tp.x; ci.eyeY = tp.y; ci.eyeZ = tp.z;
        const cpv = Math.cos(tp.pitch);
        ci.dirX = Math.sin(tp.yaw) * cpv; ci.dirY = Math.sin(tp.pitch); ci.dirZ = Math.cos(tp.yaw) * cpv;
        combat.netTime = now;
        if (frozen) {
          // (standing still behind the inventory: only the pose follows what is carried)
          combat.refreshPose(ci);
          updateHeroRig(rig, cam.hero, 0, combat.pose);
        } else {
          const hx0 = cam.hero.x, hy0 = cam.hero.y, hz0 = cam.hero.z;
          if (world === 'land') {
            if (colossi.seed !== s) colossi.reset(s);
            // (one player thinks for them all in multiplayer: the world's first; the others are hunted too)
            colossi.authority = !net || !net.inWorld() || net.keepsClock();
            const others = combat.rivals.filter((r) => r.pose.act !== 'dead').map((r) => ({ x: r.body.x, y: r.body.y - PLAYER.groundBias, z: r.body.z }));
            colossi.step(dt, cam.hero.x, cam.hero.y - PLAYER.groundBias, cam.hero.z, combat.act === 'dead', others);
            // climbing one: grabbing hold (in the air touching it, or pushing into it), climbing, stabbing, letting go
            const down = combat.act === 'dead' || combat.act === 'thrown' || combat.act === 'rising' || combat.act === 'hurt';
            if (colossi.cling && down) colossi.release(cam.hero);
            if (!colossi.cling && !down && !horses.riding) {
              const sn = Math.sin(cam.yaw), cs = Math.cos(cam.yaw);
              colossi.grab(cam.hero, sn * fwdKey + cs * strKey, cs * fwdKey - sn * strKey, dt);
            }
            if (colossi.cling) {
              const c = combat.carried();
              colossi.climb(cam.hero, fwdKey, strKey, cam.yaw, jump, ci.attack && !colossi.cling.top, c ? WEAPONS[c.kind].damage : 4, dt);
              if (colossi.cling && !colossi.cling.top) ci.attack = false; // (a stab, not a swing)
              ci.jump = false;
              if (colossi.thrown) { say('Thrown off'); combat.throwHero(colossi.thrown[0], colossi.thrown[1], colossi.thrown[2], rig.local); }
              const st = colossi.stab;
              if (st) combat.effect(st.took > 0 ? 'crit' : 'block', st.x, st.y, st.z);
            }
            beasts.length = 0;
            for (const H of horses.list) beasts.push(H);
            for (const C of colossi.list) beasts.push(C);
          }
          combat.beasts = world === 'land' ? beasts : NO_BEASTS;
          combat.step(dt, ci);
          // (slain horses going, in a drift of motes)
          for (const H of horses.vanished) combat.effect('vanish', H.x, H.y + 1.1, H.z, H.yaw);
          if (horses.riding) {
            // (struck on horseback: thrown off it)
            if (combat.act === 'hurt' || combat.act === 'dead') {
              horses.dismount(cam.hero, true);
              combat.throwHero(cam.hero.vx, cam.hero.vy, cam.hero.vz, rig.local);
            } else horses.seat(cam.hero);
          } else if (world === 'land' && combat.act !== 'thrown' && combat.act !== 'rising' && combat.act !== 'dead'
            && horses.collideHero(cam.hero, RK3)) {
            // (run down by a horse: flung)
            combat.throwHero(RK3[0], RK3[1], RK3[2], rig.local);
          }
          if (world === 'land') {
            // the colossi: the ground shaking under them, their blows (past any shield), their legs in the way
            for (const q of colossi.quakes) {
              combat.effect('quake', q.x, q.y, q.z, q.size);
              shake = Math.max(shake, (q.size * 0.04) / (1 + Math.hypot(q.x - cam.hero.x, q.z - cam.hero.z) / 25));
            }
            for (const C of colossi.vanished) for (let k = -1; k <= 1; k++) combat.effect('vanish', C.x + Math.sin(C.yaw) * (8 + 8 * k), C.y + 4, C.z + Math.cos(C.yaw) * (8 + 8 * k), C.yaw);
            if (colossi.roused) say(COLOSSUS.name);
            const ch = colossi.hit;
            if (ch && combat.act !== 'dead') {
              if (horses.riding) horses.dismount(cam.hero, true);
              combat.smashHero(ch.q, ch.vx, ch.vy, ch.vz, rig.local);
            }
            if (!horses.riding && combat.act !== 'thrown' && combat.act !== 'dead') colossi.collideHero(cam.hero);
          }
          // (only the cloud world has a ring: the land's is gone while that world is inactive)
          if (world === 'clouds') {
            const ring = portalOf(world, seedRef.current);
            crossed = portalCrossed(ring, hx0, hy0, hz0, cam.hero.x, cam.hero.y, cam.hero.z);
          }
          // (what the fight sends the other players: its blows and shots, what came of theirs; the colossi's states, blows on their eyes)
          if (net) for (let k = 0; k < combat.outCount; k++) net.sendCombat(combat.outbox[k].d, combat.outbox[k].to);
          if (net && net.inWorld() && world === 'land') {
            for (const m of colossi.sent) net.sendCombat(m, null);
            if (colossi.authority && combat.rivals.length && now - colossiSentAt >= 100) {
              colossiSentAt = now;
              for (const C of colossi.list) net.sendCombat(colossi.encode(C), null);
            }
          }
          colossi.sent.length = 0;
          combat.outCount = 0;
          combatTime += dt;
          updateHeroRig(rig, cam.hero, combat.heroDt, combat.pose);
        }
        // a rock in the hands goes where the rig holds it
        if (combat.stone && rig.rockAt[0] === rig.rockAt[0]) {
          heroLocalToWorld(rig, cam.hero, rig.rockAt[0], rig.rockAt[1], rig.rockAt[2], RK3);
          combat.holdStone(RK3[0], RK3[1], RK3[2]);
        }
        // Locked on: the orbit turns to keep the foe (or the other player) ahead.
        const lb = frozen ? null : combat.lockedBody();
        if (lb) {
          const want = Math.atan2(lb.x - cam.hero.x, lb.z - cam.hero.z);
          const d = want - cam.yaw;
          cam.yaw += (d - 2 * Math.PI * Math.round(d / (2 * Math.PI))) * (1 - Math.exp(-HERO_CAMERA.lockRate * dt));
        }
        updateHeroCamera(tp, cam.hero, cam.yaw, cam.pitch, dt, s, combat.act === 'aim' || combat.act === 'throw');
        // (never seeing from inside a colossus, nor its stone between the eye and the hero)
        if (world === 'land' && colossi.list.length && cam.mode === 'hero') colossi.clearView(cam.hero.x, cam.hero.y + 1.5, cam.hero.z, tp, 0.5);
        if (!frozen) {
          combatScene.update(combat, rig, dt);
          combatEvents();
        }
        // fallen: back at the spawn after a moment
        if (combat.act === 'dead' && combat.downFor > 3.2) doRespawn();
        // a blow shakes the view a moment
        if (shake > 1e-3) {
          tp.x += (Math.random() - 0.5) * shake; tp.y += (Math.random() - 0.5) * shake; tp.z += (Math.random() - 0.5) * shake;
          shake *= Math.exp(-16 * dt);
        } else shake = 0;
        setBasis(tp.yaw, tp.pitch);
        px = tp.x;
        py = tp.y;
        pz = tp.z;
        grounded = cam.hero.move === 'ground';
        if (now - hudAt > 50) publishCombatHud(now);
        if (menu && pouchStore && now - pouchAt > 100) { pouchAt = now; pouchStore.set(readPouch(combat)); }
        break;
      }
      case 'walk': {
        playerInput.forward = fwdKey;
        playerInput.strafe = strKey;
        playerInput.jump = jump;
        playerInput.sprint = sprint;
        playerInput.yaw = cam.yaw;
        updatePlayer(
          cam.player,
          playerInput,
          dt,
          s,
          firstPersonGround,
        );
        px = cam.player.x;
        py = cam.player.y;
        pz = cam.player.z;
        grounded = cam.player.grounded;
        break;
      }
      case 'fly': {
        const boost = sprint ? 2.5 : 1;
        const speed = PLAYER.flySpeed * boost * speedMultiplierRef.current;
        const wx = (fx * fwdKey + rx * strKey) * speed;
        const wy = fy * fwdKey * speed + (jump ? speed : 0) - (descend ? speed : 0);
        const wz = (fz * fwdKey + rz * strKey) * speed;

        const k = 1 - Math.exp(-8 * dt);
        cam.flyVel.x += (wx - cam.flyVel.x) * k;
        cam.flyVel.y += (wy - cam.flyVel.y) * k;
        cam.flyVel.z += (wz - cam.flyVel.z) * k;
        cam.fly.x += cam.flyVel.x * dt;
        cam.fly.y += cam.flyVel.y * dt;
        cam.fly.z += cam.flyVel.z * dt;

        // Free-fly mode (§IV.K.1): disable physics calculations entirely
        // (no gravity, no ground clamp, no slope sliding). Zero CPU elevation overhead.
        // The ceiling clears the sky realm on WebGPU (the WebGL2 march stays under its own).
        cam.fly.y = Math.min(cam.fly.y, gpu ? FLY_CEILING.webgpu : FLY_CEILING.webgl2);
        px = cam.fly.x;
        py = cam.fly.y;
        pz = cam.fly.z;
        break;
      }
      default:
        assertNever(cam.mode);
      }
      const tr = portalTrip.step(dt, crossed || travelAsked, false, world, view);
      crossed = false;
      travelAsked = false;
      if (tr.arrive) arrive(tr.arrive);
      if (tr.say) say(tr.say);
      const fade = portalTrip.fade();
      if (fade !== lastFade) {
        lastFade = fade;
        canvas.style.filter = fade > 0 ? `brightness(${(1 - fade).toFixed(3)})` : '';
      }
      // (the rocks round the hero, or round the camera out of the hero's view, are the CPU's: solid and drawn alike)
      if (cam.mode === 'hero') focusRocks(cam.hero.x, cam.hero.z, s); else focusRocks(px, pz, s);

      const ox = Math.floor(px / CHUNK_SIZE) * CHUNK_SIZE;
      const oz = Math.floor(pz / CHUNK_SIZE) * CHUNK_SIZE;

      if (net) netOut(net, now);
      if (mapHud) feedMap(mapHud, dt, px, py, pz, blocked);

      if (view.busy()) {
        if (now - statsAt > 200) publishStats(now, px, py, pz, grounded, ox, oz, p.resScale);
        return;
      }

      ds.time = (now - t0 - frozenTotal - (frozenSince >= 0 ? now - frozenSince : 0)) * 0.001;
      // (drawing the bow, the view narrows a little)
      ds.fov = fovRef.current * (cam.mode === 'hero' ? (1 - 0.14 * tp.aim) * tp.fov : 1);
      ds.seed = s;
      ds.marchSteps = software ? Math.min(p.marchSteps, 48) : p.marchSteps;
      ds.shadowSteps = software ? Math.min(p.shadowSteps, 10) : p.shadowSteps;
      ds.x = px;
      ds.y = py;
      ds.z = pz;
      ds.lx = px - ox;
      ds.lz = pz - oz;
      ds.ox = ox;
      ds.oz = oz;
      ds.thirdPerson = cam.mode === 'hero';
      ds.renderScale = p.resScale * auto.scale;
      ds.days = clock.days;
      placeMarkers();
      view.draw(ds);

      const rdt = Math.min((now - lastRender) / 1000, 1.0);
      lastRender = now;
      // The first frame's rAF timestamp can precede t0; a non-positive
      // interval would read as 10,000 FPS and skew the average for seconds.
      if (rdt > 0) fpsEma = updateFpsEma(fpsEma, rdt);

      // Dynamic Resolution Scaling with deadband and hysteresis (§28, drs.ts)
      if (drsStep(auto, drsCfg, fpsEma, now, t0) && !view.upscales) applySize();

      if (now - statsAt > 200) publishStats(now, px, py, pz, grounded, ox, oz, p.resScale);
    };

    raf = requestAnimationFrame(frame);

    const dispose = () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
      io?.disconnect();
      document.removeEventListener('visibilitychange', handleVisibilityChange);

      if (enablePointerLock) {
        document.removeEventListener('pointerlockchange', handlePointerLockChange);
      }
      window.removeEventListener('keydown', kd);
      window.removeEventListener('keyup', ku);
      window.removeEventListener('blur', clearKeys);
      canvas.removeEventListener('pointerdown', pd);
      window.removeEventListener('pointerup', pu);
      window.removeEventListener('pointermove', pm);
      canvas.removeEventListener('pointercancel', pu);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('mousedown', md);
      window.removeEventListener('mouseup', mu);
      canvas.removeEventListener('contextmenu', noMenu);
      sfx.dispose();

      view.dispose();
    };
    return {
      dispose, respawn: doRespawn, toggleMode: doToggleMode, toggleView: doToggleView, travel: () => { travelAsked = true; },
      setHour: (hour) => { setHour(clock, hour); clockDirty = true; },
      setTimePaused: (paused) => { clock.paused = paused; clockDirty = true; },
      setDayMinutes: (minutes) => { clock.dayMinutes = minutes; clockDirty = true; },
      clock: () => ({ days: clock.days, paused: clock.paused, dayMinutes: clock.dayMinutes }),
      pouch: (cmd) => {
        // (the world may be standing still, so what the command did is heard and shown here and now)
        combat.eventCount = 0;
        let ok = false;
        switch (cmd.op) {
          case 'equip': ok = combat.equip(cmd.cat, cmd.index); break;
          case 'unequip': ok = combat.unequip(cmd.cat); break;
          case 'drop': ok = combat.discard(cmd.cat, cmd.index); break;
          case 'eat': ok = combat.eat(cmd.index); break;
          default: assertNever(cmd);
        }
        combatEvents();
        combat.eventCount = 0;
        if (ok) {
          pouchStore?.set(readPouch(combat));
          publishCombatHud(performance.now());
        }
        return ok;
      },
    };
  } catch (e) {
    ro?.disconnect();
    throw e;
  }
}
