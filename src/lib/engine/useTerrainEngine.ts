'use client';

import { useEffect, useRef, useState, type RefObject } from 'react';
import { TERRAIN_FS, TERRAIN_VS } from '@/components/terrain/demoShader';
import { elevation, updatePlayer, type PlayerState } from '@/lib/terrainMath';
import {
  CHUNK_SIZE,
  DEFAULT_SEED,
  SPAWN,
  CAMERA,
  PLAYER,
  QUALITY_PRESETS,
  type PresetId,
} from '@/lib/constants';

export type { PresetId };
type CamMode = 'walk' | 'fly';

export type EngineStats = {
  fps: number;
  x: number;
  y: number;
  z: number;
  mode: CamMode;
  grounded: boolean;
  originX: number;
  originZ: number;
  resPct: number;
};

type CamState = {
  yaw: number;
  pitch: number;
  mode: CamMode;
  fly: { x: number; y: number; z: number };
  flyVel: { x: number; y: number; z: number };
  player: PlayerState;
};

export type TerrainEngineOptions = {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  seed?: number;
  preset?: PresetId;
  fov?: number;
  sensitivity?: number;
  speedMultiplier?: number;
  enablePointerLock?: boolean;
  running?: boolean;
  onPointerLockChange?: (locked: boolean) => void;
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

export function useTerrainEngine({
  canvasRef,
  seed = DEFAULT_SEED,
  preset = 'med',
  fov = CAMERA.fov,
  sensitivity = CAMERA.sensitivity,
  speedMultiplier = 1.0,
  enablePointerLock = false,
  running = true,
  onPointerLockChange,
  onToggleHud,
}: TerrainEngineOptions & { onToggleHud?: () => void }) {
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  const seedRef = useRef(seed);
  const presetRef = useRef(preset);
  const runRef = useRef(running);
  const fovRef = useRef(fov);
  const sensitivityRef = useRef(sensitivity);
  const speedMultiplierRef = useRef(speedMultiplier);
  const hudToggleRef = useRef(onToggleHud);
  const onPointerLockChangeRef = useRef(onPointerLockChange);

  const respawnTriggerRef = useRef<(() => void) | null>(null);
  const modeToggleRef = useRef<(() => void) | null>(null);

  seedRef.current = seed;
  presetRef.current = preset;
  runRef.current = running;
  fovRef.current = fov;
  sensitivityRef.current = sensitivity;
  speedMultiplierRef.current = speedMultiplier;
  hudToggleRef.current = onToggleHud;
  onPointerLockChangeRef.current = onPointerLockChange;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
    let ro: ResizeObserver | null = null;
    const keys = new Set<string>();

    let gl: WebGL2RenderingContext | null = null;
    let vs: WebGLShader | null = null;
    let fs: WebGLShader | null = null;
    let prog: WebGLProgram | null = null;
    let fence: WebGLSync | null = null;

    try {
      gl = canvas.getContext('webgl2', {
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

      const auto = { scale: software ? 0.25 : 1.0, at: 0 };

      vs = compileShader(gl, gl.VERTEX_SHADER, TERRAIN_VS);
      fs = compileShader(gl, gl.FRAGMENT_SHADER, TERRAIN_FS);
      prog = gl.createProgram();
      if (!prog) throw new Error('Failed to create WebGL program');

      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(prog) ?? 'Program link failed');
      }
      gl.useProgram(prog);

      const U = (n: string) => gl!.getUniformLocation(prog!, n);
      const u = {
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
      };

      const cam: CamState = {
        yaw: SPAWN.yaw,
        pitch: SPAWN.pitch,
        mode: 'fly',
        fly: { x: SPAWN.x, y: 0, z: SPAWN.z },
        flyVel: { x: 0, y: 0, z: 0 },
        player: {
          x: SPAWN.x,
          y: 0,
          z: SPAWN.z,
          vx: 0,
          vy: 0,
          vz: 0,
          grounded: false,
        },
      };

      cam.fly.y = elevation(SPAWN.x, SPAWN.z, seedRef.current, 0) + 30;
      cam.player.y = cam.fly.y;

      const doRespawn = () => {
        const s = seedRef.current;
        const ground = elevation(SPAWN.x, SPAWN.z, s, 0);
        const spawnY = ground + 30;
        cam.yaw = SPAWN.yaw;
        cam.pitch = SPAWN.pitch;
        cam.fly = { x: SPAWN.x, y: spawnY, z: SPAWN.z };
        cam.flyVel = { x: 0, y: 0, z: 0 };
        cam.player = {
          x: SPAWN.x,
          y: ground + 1.8,
          z: SPAWN.z,
          vx: 0,
          vy: 0,
          vz: 0,
          grounded: false,
        };
      };
      respawnTriggerRef.current = doRespawn;

      const doToggleMode = () => {
        if (cam.mode === 'fly') {
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
        } else {
          cam.mode = 'fly';
          cam.fly = { x: cam.player.x, y: cam.player.y, z: cam.player.z };
          cam.flyVel = { x: 0, y: 0, z: 0 };
        }
      };
      modeToggleRef.current = doToggleMode;

      const onKey = (e: KeyboardEvent, down: boolean) => {
        if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') {
          return;
        }

        if (e.code === 'KeyF' && down) {
          doToggleMode();
          return;
        }

        if (e.code === 'KeyR' && down && !e.metaKey && !e.ctrlKey) {
          doRespawn();
          return;
        }

        if (e.code === 'KeyH' && down && !e.metaKey && !e.ctrlKey) {
          hudToggleRef.current?.();
          return;
        }

        if (down) keys.add(e.code);
        else keys.delete(e.code);

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
          canvas.requestPointerLock?.();
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

      const clearKeys = () => keys.clear();

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

      const PRESET_MAP: Record<PresetId, (typeof QUALITY_PRESETS)[number]> = {
        low: QUALITY_PRESETS[0],
        med: QUALITY_PRESETS[1],
        high: QUALITY_PRESETS[2],
      };

      const applySize = () => {
        if (!gl || !canvas) return;
        const p = PRESET_MAP[presetRef.current] ?? PRESET_MAP.med;
        const dpr = software ? 1 : Math.min(window.devicePixelRatio || 1, 2);
        const s = p.resScale * auto.scale;
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
      const basis = new Float32Array(9);

      const frame = (now: number) => {
        raf = requestAnimationFrame(frame);
        const active = isIntersecting && isTabVisible && runRef.current;
        const dt = Math.min((now - last) / 1000, 0.05);
        last = now;
        if (!active || !gl || !prog) {
          lastRender = now;
          return;
        }

        const s = seedRef.current;
        const p = PRESET_MAP[presetRef.current] ?? PRESET_MAP.med;

        // Locomotion update
        const fwdKey = (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0);
        const strKey = (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0);
        const sprint = keys.has('ShiftLeft') || keys.has('ShiftRight');
        const jump = keys.has('Space');

        const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
        const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
        const fx = sy * cp, fy = sp, fz = cy * cp;
        const rx = cy, rz = -sy;
        const upx = fy * rz;
        const upy = fz * rx - fx * rz;
        const upz = -fy * rx;

        basis[0] = rx;   basis[1] = 0;   basis[2] = rz;
        basis[3] = upx;  basis[4] = upy; basis[5] = upz;
        basis[6] = fx;   basis[7] = fy;  basis[8] = fz;

        let px: number, py: number, pz: number, grounded = false;

        if (cam.mode === 'walk') {
          updatePlayer(
            cam.player,
            { forward: fwdKey, strafe: strKey, jump, sprint, yaw: cam.yaw },
            dt,
            s,
          );
          px = cam.player.x;
          py = cam.player.y;
          pz = cam.player.z;
          grounded = cam.player.grounded;
        } else {
          const boost = sprint ? 2.5 : 1;
          const speed = PLAYER.flySpeed * boost * speedMultiplierRef.current;
          const wx = (fx * fwdKey + rx * strKey) * speed;
          const wy = fy * fwdKey * speed + (keys.has('Space') ? speed : 0) - (keys.has('KeyC') || keys.has('ControlLeft') ? speed : 0);
          const wz = (fz * fwdKey + rz * strKey) * speed;

          const k = 1 - Math.exp(-8 * dt);
          cam.flyVel.x += (wx - cam.flyVel.x) * k;
          cam.flyVel.y += (wy - cam.flyVel.y) * k;
          cam.flyVel.z += (wz - cam.flyVel.z) * k;
          cam.fly.x += cam.flyVel.x * dt;
          cam.fly.y += cam.flyVel.y * dt;
          cam.fly.z += cam.flyVel.z * dt;

          if (cam.fly.y < 950) {
            const minY = elevation(cam.fly.x, cam.fly.z, s, 0) + 1.9;
            if (cam.fly.y < minY) {
              cam.fly.y = minY;
              cam.flyVel.y = Math.max(cam.flyVel.y, 0);
            }
          }
          cam.fly.y = Math.min(cam.fly.y, 1150);
          px = cam.fly.x;
          py = cam.fly.y;
          pz = cam.fly.z;
        }

        const ox = Math.floor(px / CHUNK_SIZE) * CHUNK_SIZE;
        const oz = Math.floor(pz / CHUNK_SIZE) * CHUNK_SIZE;
        const lx = px - ox;
        const lz = pz - oz;

        const emitStats = () => {
          if (now - statsAt > 200) {
            statsAt = now;
            setStats({
              fps: Math.round(fpsEma),
              x: px,
              y: py,
              z: pz,
              mode: cam.mode,
              grounded,
              originX: ox,
              originZ: oz,
              resPct: Math.round(p.resScale * auto.scale * 100),
            });
          }
        };

        if (fence && gl.clientWaitSync(fence, 0, 0) === gl.TIMEOUT_EXPIRED) {
          emitStats();
          return;
        }

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform2f(u.res, canvas.width, canvas.height);
        gl.uniform1f(u.time, (now - t0) / 1000);
        gl.uniform3f(u.cam, lx, py, lz);
        gl.uniform2f(u.origin, ox, oz);
        gl.uniformMatrix3fv(u.basis, false, basis);
        const tanHalf = Math.tan((fovRef.current * Math.PI / 180) * 0.5);
        gl.uniform1f(u.tan, tanHalf);
        gl.uniform1i(u.seed, s | 0);
        gl.uniform1i(u.march, software ? Math.min(p.marchSteps, 48) : p.marchSteps);
        gl.uniform1i(u.shadow, software ? Math.min(p.shadowSteps, 10) : p.shadowSteps);
        gl.uniform1f(u.foot, (2 * tanHalf) / Math.max(canvas.height, 2));
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        if (fence) gl.deleteSync(fence);
        fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);

        const rdt = Math.min((now - lastRender) / 1000, 1.0);
        lastRender = now;
        fpsEma += (1 / Math.max(rdt, 1e-4) - fpsEma) * 0.1;

        if (now - t0 > 2000 && now - auto.at > 1500 && fpsEma < 14 && auto.scale > 0.07) {
          auto.scale *= 0.5;
          auto.at = now;
          fpsEma = 30;
          applySize();
        }

        emitStats();
      };

      raf = requestAnimationFrame(frame);

      return () => {
        cancelAnimationFrame(raf);
        ro?.disconnect();
        io?.disconnect();
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        if (fence && gl) gl.deleteSync(fence);

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

        if (gl) {
          if (prog) {
            if (vs) {
              gl.detachShader(prog, vs);
              gl.deleteShader(vs);
            }
            if (fs) {
              gl.detachShader(prog, fs);
              gl.deleteShader(fs);
            }
            gl.deleteProgram(prog);
          }
        }

        respawnTriggerRef.current = null;
        modeToggleRef.current = null;
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return () => {
        ro?.disconnect();
      };
    }
  }, [canvasRef, enablePointerLock]);

  return {
    stats,
    error,
    respawn: () => respawnTriggerRef.current?.(),
    toggleMode: () => modeToggleRef.current?.(),
  };
}
