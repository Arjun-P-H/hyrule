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
 */

import { assertNever } from '@/lib/utils';
import type { CamMode, RenderApi, StatsStore } from './statsStore';
import { TERRAIN_FS, TERRAIN_VS, minifyGlsl, withHero } from '@/components/terrain/demoShader';
import { elevation, updatePlayer, type PlayerState, type InputState } from '@/lib/terrainMath';
import { DRS_HARDWARE, DRS_SOFTWARE, createDrsState, drsStep, updateFpsEma } from './drs';
import { createHeroState, firstPersonGround, heroAction, placeHero, updateHero, type HeroInput, type HeroState } from '@/lib/hero/locomotion';
import { createHeroRig, packHeroUniforms, updateHeroRig, HERO_UNIFORM } from '@/lib/hero/rig';
import { createHeroCamera, updateHeroCamera } from '@/lib/hero/camera';
import { skySpawn } from '@/lib/skyworld/realm';
import type { GpuBackend, SkyMode } from '@/lib/gpu/engineGpu';
import { advanceClock, createSkyClock, hourOf, setHour } from '@/lib/sky/clock';
import type { FrameParams } from '@/lib/gpu/frame';
import {
  CHUNK_SIZE,
  FLY_CEILING,
  SPAWN,
  PLAYER,
  HERO_CAMERA,
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
  onPointerLockChange: Ref<((locked: boolean) => void) | undefined>;
  /** The sky wanted (drawn by the WebGPU view; the WebGL2 view has only the classic sky). */
  sky: Ref<SkyMode>;
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
};

/** Game hours per real second while T is held. */
const FAST_TIME = 1;

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
  };
}

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
  let heroViews: { j: Float32Array; torso: Float32Array; head: Float32Array; bound: Float32Array; misc: Float32Array } | null = null;

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
          misc: hb.subarray(HERO_UNIFORM.misc, HERO_UNIFORM.length),
        };
        gl.uniform3fv(u.heroJ, heroViews.j);
        gl.uniformMatrix3fv(u.heroTorso, false, heroViews.torso);
        gl.uniformMatrix3fv(u.heroHead, false, heroViews.head);
        gl.uniform4fv(u.heroBound, heroViews.bound);
        gl.uniform4fv(u.heroMisc, heroViews.misc);
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
    seed: 0, marchSteps: 0, shadowSteps: 0, time: 0, hero: null, renderScale: 1, days: 0,
  };
  let lastFov = -1;
  return {
    api: 'webgpu',
    software: false,
    upscales: gpu.upscales,
    busy: () => gpu.inFlight() >= 2,
    sky: () => gpu.sky(),
    setSky: (mode) => gpu.setSky(mode),
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
  const onPointerLockChangeRef = inputs.onPointerLockChange;
  const skyRef = inputs.sky;
  const clock = createSkyClock(clockStart.hour, clockStart.dayMinutes);
  let fastTime = false;

  let raf = 0;
  let ro: ResizeObserver | null = null;
  let keyMask = 0;
  const KEY_W = 1 << 0;
  const KEY_S = 1 << 1;
  const KEY_A = 1 << 2;
  const KEY_D = 1 << 3;
  const KEY_UP = 1 << 4;
  const KEY_DOWN = 1 << 5;
  const KEY_LEFT = 1 << 6;
  const KEY_RIGHT = 1 << 7;
  const KEY_SHIFT = 1 << 8;
  const KEY_SPACE = 1 << 9;
  const KEY_C = 1 << 10;
  const KEY_CTRL = 1 << 11;

  const keyCodeToBit = (code: string): number => {
    switch (code) {
      case 'KeyW': return KEY_W;
      case 'KeyS': return KEY_S;
      case 'KeyA': return KEY_A;
      case 'KeyD': return KEY_D;
      case 'ArrowUp': return KEY_UP;
      case 'ArrowDown': return KEY_DOWN;
      case 'ArrowLeft': return KEY_LEFT;
      case 'ArrowRight': return KEY_RIGHT;
      case 'ShiftLeft':
      case 'ShiftRight': return KEY_SHIFT;
      case 'Space': return KEY_SPACE;
      case 'KeyC': return KEY_C;
      case 'ControlLeft':
      case 'ControlRight': return KEY_CTRL;
      default: return 0;
    }
  };

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

    // The start: the land's spawn, or (?spawn=sky) a spot up on the nearest archipelago's hub.
    const spawnFor = (seed: number) => {
      const sky = spawnAt === 'sky' ? skySpawn(SPAWN.x, SPAWN.z, seed) : null;
      return sky ?? { x: SPAWN.x, y: elevation(SPAWN.x, SPAWN.z, seed, 0), z: SPAWN.z, yaw: SPAWN.yaw };
    };
    let spawn = spawnFor(seedRef.current);

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
      hero: createHeroState(spawn.x, spawn.z, spawn.yaw, seedRef.current),
    };
    // (up in the sky the hero starts on the island's top, not on the land below)
    if (spawnAt === 'sky') placeHero(cam.hero, spawn.x, spawn.y + PLAYER.groundBias, spawn.z, 0, 0, 0, spawn.yaw, seedRef.current);

    cam.fly.y = spawn.y + 30;
    cam.player.y = cam.fly.y;
    if (initialMode === 'walk') {
      cam.player.y = cam.hero.y + PLAYER.height;
      cam.player.grounded = true;
    }

    const playerInput: InputState = { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 };
    const heroInput: HeroInput = { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0 };

    /** Drop into third person with the hero's root (feet) at x, y, z. */
    const enterHero = (x: number, y: number, z: number, vx: number, vy: number, vz: number) => {
      placeHero(cam.hero, x, y, z, vx, vy, vz, cam.yaw, seedRef.current);
      cam.mode = 'hero';
      cam.onFoot = 'hero';
      cam.pitch = HERO_CAMERA.pitch;
      rig.snap = true;
      tp.snap = true;
    };

    const doRespawn = () => {
      const s = seedRef.current;
      spawn = spawnFor(s);
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

      if (e.code === 'KeyF' && down) {
        if (!e.repeat) doToggleMode();
        return;
      }

      if (e.code === 'KeyV' && down && !e.metaKey && !e.ctrlKey) {
        if (!e.repeat) doToggleView();
        return;
      }

      if (e.code === 'KeyR' && down && !e.metaKey && !e.ctrlKey) {
        doRespawn();
        return;
      }

      if (e.code === 'KeyT' && !e.metaKey && !e.ctrlKey) {
        fastTime = down;
        return;
      }

      if (e.code === 'KeyH' && down && !e.metaKey && !e.ctrlKey) {
        hudToggleRef.current?.();
        return;
      }

      const bit = keyCodeToBit(e.code);
      if (down) keyMask |= bit;
      else keyMask &= ~bit;

      const isInteracting =
        enablePointerLock ||
        document.pointerLockElement === canvas ||
        document.activeElement === canvas ||
        dragging;

      if (isInteracting && ['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
        e.preventDefault();
      }
    };

    const kd = (e: KeyboardEvent) => onKey(e, true);
    const ku = (e: KeyboardEvent) => onKey(e, false);

    let dragging = false;
    let lastX = 0, lastY = 0;

    const pd = (e: PointerEvent) => {
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

    const clearKeys = () => { keyMask = 0; fastTime = false; };

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
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
      } else if (dragging) {
        cam.yaw += (e.clientX - lastX) * sens;
        cam.pitch -= (e.clientY - lastY) * sens;
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
        packHeroUniforms(rig, cam.hero, ds.x, ds.y, ds.z, heroBuf);
        return heroBuf;
      },
    };

    // Telemetry for the HUD at most every 200 ms; created once so the 60 Hz
    // loop allocates no closure per frame.
    const publishStats = (
      now: number, x: number, y: number, z: number, grounded: boolean,
      originX: number, originZ: number, resScale: number,
    ) => {
      statsAt = now;
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
      });
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

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const active = isIntersecting && isTabVisible && runRef.current;
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (!active) {
        lastRender = now;
        return;
      }

      const s = seedRef.current;
      const p = PRESET_MAP[software ? 'low' : (presetRef.current ?? 'med')] ?? PRESET_MAP.med;
      // The sky: the one wanted (the view builds it in the background), and the clock.
      if (view.sky() !== null && skyRef.current !== view.sky()) view.setSky(skyRef.current);
      if (fastTime) clock.days += (dt * FAST_TIME) / 24;
      else advanceClock(clock, dt);
      // A new preset changes the render resolution now, not at the next DRS step.
      if (presetRef.current !== lastPreset) {
        lastPreset = presetRef.current;
        applySize();
      }

      // Locomotion update via zero-allocation bitmask
      const fwdKey = ((keyMask & (KEY_W | KEY_UP)) ? 1 : 0) - ((keyMask & (KEY_S | KEY_DOWN)) ? 1 : 0);
      const strKey = ((keyMask & (KEY_D | KEY_RIGHT)) ? 1 : 0) - ((keyMask & (KEY_A | KEY_LEFT)) ? 1 : 0);
      const sprint = (keyMask & KEY_SHIFT) !== 0;
      const jump = (keyMask & KEY_SPACE) !== 0;
      const descend = (keyMask & (KEY_C | KEY_CTRL)) !== 0;

      // Third person aims from its own eye after the hero moves (below).
      if (cam.mode !== 'hero') setBasis(cam.yaw, cam.pitch);

      let px: number, py: number, pz: number, grounded = false;

      switch (cam.mode) {
      case 'hero': {
        if (cam.pitch < HERO_CAMERA.pitchMin) cam.pitch = HERO_CAMERA.pitchMin;
        else if (cam.pitch > HERO_CAMERA.pitchMax) cam.pitch = HERO_CAMERA.pitchMax;
        heroInput.forward = fwdKey;
        heroInput.strafe = strKey;
        heroInput.sprint = sprint;
        heroInput.jump = jump;
        heroInput.sneak = (keyMask & KEY_C) !== 0;
        heroInput.camYaw = cam.yaw;
        updateHero(cam.hero, heroInput, dt, s);
        updateHeroRig(rig, cam.hero, dt);
        updateHeroCamera(tp, cam.hero, cam.yaw, cam.pitch, dt, s);
        setBasis(tp.yaw, tp.pitch);
        px = tp.x;
        py = tp.y;
        pz = tp.z;
        grounded = cam.hero.move === 'ground';
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

      const ox = Math.floor(px / CHUNK_SIZE) * CHUNK_SIZE;
      const oz = Math.floor(pz / CHUNK_SIZE) * CHUNK_SIZE;

      if (view.busy()) {
        if (now - statsAt > 200) publishStats(now, px, py, pz, grounded, ox, oz, p.resScale);
        return;
      }

      ds.time = (now - t0) * 0.001;
      ds.fov = fovRef.current;
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

      view.dispose();
    };
    return {
      dispose, respawn: doRespawn, toggleMode: doToggleMode, toggleView: doToggleView,
      setHour: (hour) => setHour(clock, hour),
      setTimePaused: (paused) => { clock.paused = paused; },
      setDayMinutes: (minutes) => { clock.dayMinutes = minutes; },
    };
  } catch (e) {
    ro?.disconnect();
    throw e;
  }
}
