'use client';

/**
 * W(x,y) — live WebGL2 raymarch harness (spec §30 Stage-0 GPU parity demo).
 * ---------------------------------------------------------------------------
 * - The GPU draws the world with demoShader.ts (fp32, chunk-local).
 * - Walk mode is driven by src/lib/terrainMath.ts updatePlayer() — the SAME
 *   §31 movement loop as the Godot GDExtension C++ twin: jump-preserving
 *   ground clamp, GROUND_BIAS parity absorber, slope slide-off gate.
 * - Fly mode clamps its altitude against the CPU twin's elevation() — the
 *   twin and the shader agree to < 1 mm, so the camera never clips the field.
 * - Floating origin (§25): chunk origin = 8192·floor(pos/8192) recomputed
 *   every frame from double-precision positions; the shader receives
 *   (local fp32, origin) — no fract() precision jitter, ever.
 */

import { useEffect, useRef, useState } from 'react';
import { TERRAIN_FS, TERRAIN_VS } from './demoShader';
import {
  elevation, updatePlayer, type PlayerState,
} from '@/lib/terrainMath';
import {
  CHUNK_SIZE, DEFAULT_SEED, SPAWN, CAMERA, QUALITY_PRESETS,
} from '@/lib/constants';

type PresetId = 'low' | 'med' | 'high';

type CamState = {
  yaw: number;
  pitch: number;
  mode: 'walk' | 'fly';
  fly: { x: number; y: number; z: number };
  flyVel: { x: number; y: number; z: number };
  player: PlayerState;
};

type Stats = {
  fps: number;
  x: number; y: number; z: number;
  mode: 'walk' | 'fly';
  grounded: boolean;
  originX: number; originZ: number;
  resPct: number; // effective resolution scale (adaptive quality, §26)
};

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) ?? 'shader compile failed');
  }
  return sh;
}

export default function TerrainDemo({
  seed = DEFAULT_SEED,
  preset = 'med' as PresetId,
}: {
  seed?: number;
  preset?: PresetId;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(true);

  // Refs so the render loop never re-binds on React state changes.
  const seedRef = useRef(seed);
  const presetRef = useRef(preset);
  const runRef = useRef(true);
  useEffect(() => { seedRef.current = seed; }, [seed]);
  useEffect(() => { presetRef.current = preset; }, [preset]);
  useEffect(() => { runRef.current = running; }, [running]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
    let ro: ResizeObserver | null = null;
    const keys = new Set<string>();

    try {
      const gl = canvas.getContext('webgl2', {
        antialias: false,
        alpha: false,
        depth: false,
        stencil: false,
        // Keep the last presented frame readable — screenshot/toDataURL
        // support for QA + share buttons. Negligible cost for one fullscreen
        // triangle; the fence-sync pacing above already idles the GPU.
        preserveDrawingBuffer: true,
        powerPreference: 'high-performance',
      });
      if (!gl) throw new Error('WebGL2 unavailable in this browser');

      // Software rasterizer (SwiftShader/llvmpipe in headless or VMs):
      // start tiny and let the adaptive scaler climb if real HW shows up.
      let software = false;
      try {
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        const renderer = dbg
          ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL))
          : '';
        software = /swiftshader|software|llvmpipe|basic/i.test(renderer);
      } catch { /* renderer info unavailable — assume hardware */ }
      const auto = { scale: software ? 0.12 : 1.0, at: 0 };

      const prog = gl.createProgram()!;
      gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, TERRAIN_VS));
      gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, TERRAIN_FS));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(prog) ?? 'link failed');
      }
      gl.useProgram(prog);

      const U = (n: string) => gl.getUniformLocation(prog, n);
      const u = {
        res: U('u_res'), time: U('u_time'), cam: U('u_cam_pos'),
        origin: U('u_chunk_origin'), basis: U('u_basis'), tan: U('u_tan_half'),
        seed: U('u_seed'), march: U('u_march_steps'), shadow: U('u_shadow_steps'),
        foot: U('u_footprint'),
      };

      const cam: CamState = {
        yaw: SPAWN.yaw,
        pitch: SPAWN.pitch,
        mode: 'fly',
        fly: { x: SPAWN.x, y: 0, z: SPAWN.z },
        flyVel: { x: 0, y: 0, z: 0 },
        player: {
          x: SPAWN.x, y: 0, z: SPAWN.z,
          vx: 0, vy: 0, vz: 0, grounded: false,
        },
      };
      // Spawn 30 m above the field (worked example, §30).
      cam.fly.y = elevation(SPAWN.x, SPAWN.z, seed, 0) + 30;
      cam.player.y = cam.fly.y;

      // ---- input -----------------------------------------------------------
      const onKey = (e: KeyboardEvent, down: boolean) => {
        if (e.code === 'KeyF' && down) {
          if (cam.mode === 'fly') {
            cam.mode = 'walk';
            cam.player = {
              x: cam.fly.x, y: cam.fly.y, z: cam.fly.z,
              vx: 0, vy: 0, vz: 0, grounded: false,
            };
          } else {
            cam.mode = 'fly';
            cam.fly = { x: cam.player.x, y: cam.player.y, z: cam.player.z };
            cam.flyVel = { x: 0, y: 0, z: 0 };
          }
          return;
        }
        if (down) keys.add(e.code); else keys.delete(e.code);
        if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
          e.preventDefault();
        }
      };
      const kd = (e: KeyboardEvent) => onKey(e, true);
      const ku = (e: KeyboardEvent) => onKey(e, false);

      let dragging = false;
      let lastX = 0, lastY = 0;
      const pd = (e: PointerEvent) => {
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
      };
      const pu = (e: PointerEvent) => {
        dragging = false;
        if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      };
      const clearKeys = () => keys.clear();
      const pm = (e: PointerEvent) => {
        if (!dragging) return;
        cam.yaw += (e.clientX - lastX) * CAMERA.sensitivity;
        cam.pitch -= (e.clientY - lastY) * CAMERA.sensitivity;
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
        lastX = e.clientX; lastY = e.clientY;
      };

      window.addEventListener('keydown', kd);
      window.addEventListener('keyup', ku);
      window.addEventListener('blur', clearKeys);
      canvas.addEventListener('pointerdown', pd);
      canvas.addEventListener('pointerup', pu);
      canvas.addEventListener('pointermove', pm);
      canvas.addEventListener('pointercancel', pu);

      // ---- resize (resolution scale from quality preset, §29) --------------
      const applySize = () => {
        const p = QUALITY_PRESETS.find((q) => q.id === presetRef.current)
          ?? QUALITY_PRESETS[1];
        // Adaptive resolution (§26 budget): effective scale = preset × auto.
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

      // ---- main loop --------------------------------------------------------
      const tanHalf = Math.tan((CAMERA.fov * Math.PI / 180) * 0.5);
      const t0 = performance.now();
      let last = t0;
      let lastRender = t0;
      let fpsEma = 60;
      let statsAt = 0;
      // Frame pacing via fence sync (§26 budget discipline): never queue a
      // new march while the GPU is still rasterising the previous one. On
      // real hardware the fence is signaled within a frame (no-op); on
      // software rasterizers this keeps the command buffer from backlogging
      // and freezing page interactions.
      let fence: WebGLSync | null = null;

      const frame = (now: number) => {
        raf = requestAnimationFrame(frame);
        const dt = Math.min((now - last) / 1000, 0.05);
        last = now;
        if (!runRef.current) return;

        const s = seedRef.current;
        const p = QUALITY_PRESETS.find((q) => q.id === presetRef.current)
          ?? QUALITY_PRESETS[1];

        // -- movement (CPU twin drives walk mode — §31 loop verbatim) -------
        const fwdKey = (keys.has('KeyW') ? 1 : 0) - (keys.has('KeyS') ? 1 : 0);
        const strKey = (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0);
        const sprint = keys.has('ShiftLeft') || keys.has('ShiftRight');
        const jump = keys.has('Space');

        let px: number, py: number, pz: number, grounded = false;

        if (cam.mode === 'walk') {
          updatePlayer(
            cam.player,
            { forward: fwdKey, strafe: strKey, jump, sprint, yaw: cam.yaw },
            dt,
            s,
          );
          px = cam.player.x; py = cam.player.y; pz = cam.player.z;
          grounded = cam.player.grounded;
        } else {
          const boost = sprint ? 2.5 : 1;
          const speed = 40 * boost; // PLAYER.flySpeed
          const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
          const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
          // forward includes pitch; strafe is horizontal (LH basis — see header)
          const fx = sy * cp, fy = sp, fz = cy * cp;
          const rx = cy, rz = -sy;
          const wx = (fx * fwdKey + rx * strKey) * speed;
          const wy = fy * fwdKey * speed + (keys.has('Space') ? speed : 0)
            - (keys.has('KeyC') || keys.has('ControlLeft') ? speed : 0);
          const wz = (fz * fwdKey + rz * strKey) * speed;
          const k = 1 - Math.exp(-8 * dt);
          cam.flyVel.x += (wx - cam.flyVel.x) * k;
          cam.flyVel.y += (wy - cam.flyVel.y) * k;
          cam.flyVel.z += (wz - cam.flyVel.z) * k;
          cam.fly.x += cam.flyVel.x * dt;
          cam.fly.y += cam.flyVel.y * dt;
          cam.fly.z += cam.flyVel.z * dt;
          // clamp against the CPU twin — parity keeps this seamless
          const minY = elevation(cam.fly.x, cam.fly.z, s, 0) + 1.9;
          if (cam.fly.y < minY) { cam.fly.y = minY; cam.flyVel.y = Math.max(cam.flyVel.y, 0); }
          cam.fly.y = Math.min(cam.fly.y, 1150); // stay under the cloud deck
          px = cam.fly.x; py = cam.fly.y; pz = cam.fly.z;
        }

        // -- floating origin (§25): origin is a multiple of 8192 → fp32-exact
        const ox = Math.floor(px / CHUNK_SIZE) * CHUNK_SIZE;
        const oz = Math.floor(pz / CHUNK_SIZE) * CHUNK_SIZE;
        const lx = px - ox, lz = pz - oz;

        // -- camera basis (view→world, columns right/up/forward) -------------
        const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
        const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
        const f = [sy * cp, sp, cy * cp];
        const r = [cy, 0, -sy];
        const up = [
          f[1] * r[2] - f[2] * r[1],
          f[2] * r[0] - f[0] * r[2],
          f[0] * r[1] - f[1] * r[0],
        ];
        const basis = new Float32Array([
          r[0], r[1], r[2],
          up[0], up[1], up[2],
          f[0], f[1], f[2],
        ]);

        // -- GPU-paced submission: skip the march while the previous frame is
        // still rasterising (the simulation above keeps running at wall
        // clock, so movement never slows down under GPU load).
        if (fence && gl.clientWaitSync(fence, 0, 0) === gl.TIMEOUT_EXPIRED) {
          if (now - statsAt > 400) {
            statsAt = now;
            setStats({
              fps: Math.round(fpsEma), x: px, y: py, z: pz,
              mode: cam.mode, grounded, originX: ox, originZ: oz,
              resPct: Math.round(p.resScale * auto.scale * 100),
            });
          }
          return;
        }

        applySize();
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(prog);
        gl.uniform2f(u.res, canvas.width, canvas.height);
        gl.uniform1f(u.time, (now - t0) / 1000);
        gl.uniform3f(u.cam, lx, py, lz);
        gl.uniform2f(u.origin, ox, oz);
        gl.uniformMatrix3fv(u.basis, false, basis);
        gl.uniform1f(u.tan, tanHalf);
        gl.uniform1i(u.seed, s | 0);
        // Software rasterizers also get a step budget cut — the march is the
        // dominant cost, and distant steps are Nyquist-attenuated anyway.
        gl.uniform1i(u.march, software ? Math.min(p.marchSteps, 48) : p.marchSteps);
        gl.uniform1i(u.shadow, software ? Math.min(p.shadowSteps, 10) : p.shadowSteps);
        gl.uniform1f(u.foot, (2 * tanHalf) / Math.max(canvas.height, 2));
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        // Pace the next submission on this frame's completion.
        if (fence) gl.deleteSync(fence);
        fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);

        // -- throttled HUD stats + adaptive quality (§26 budget) ---------
        // fps is measured on COMPLETED renders — honest under GPU load.
        const rdt = Math.min((now - lastRender) / 1000, 1.0);
        lastRender = now;
        fpsEma += (1 / Math.max(rdt, 1e-4) - fpsEma) * 0.08;
        // If the sustained frame rate cannot hold ~14 fps, halve the
        // internal resolution (floor 7%) — SwiftShader/llvmpipe fallback.
        if (now - t0 > 2000 && now - auto.at > 1500 && fpsEma < 14 && auto.scale > 0.07) {
          auto.scale *= 0.5;
          auto.at = now;
          fpsEma = 30; // reset so we measure the new scale before degrading again
          applySize();
        }
        if (now - statsAt > 400) {
          statsAt = now;
          setStats({
            fps: Math.round(fpsEma), x: px, y: py, z: pz,
            mode: cam.mode, grounded, originX: ox, originZ: oz,
            resPct: Math.round(p.resScale * auto.scale * 100),
          });
        }
      };
      raf = requestAnimationFrame(frame);

      return () => {
        cancelAnimationFrame(raf);
        ro?.disconnect();
        if (fence) gl.deleteSync(fence);
        window.removeEventListener('keydown', kd);
        window.removeEventListener('keyup', ku);
        window.removeEventListener('blur', clearKeys);
        canvas.removeEventListener('pointerdown', pd);
        canvas.removeEventListener('pointerup', pu);
        canvas.removeEventListener('pointermove', pm);
        canvas.removeEventListener('pointercancel', pu);
        gl.deleteProgram(prog);
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return () => { ro?.disconnect(); };
    }
  }, []);

  return (
    <div className="relative w-full overflow-hidden rounded-xl border bg-stone-950">
      <canvas
        ref={canvasRef}
        className="block h-[46vh] min-h-[320px] w-full cursor-grab touch-none active:cursor-grabbing sm:h-[56vh]"
        aria-label="Live procedural terrain raymarcher — drag to look, WASD to move, F to toggle walk mode"
        role="img"
      />

      {/* HUD — top left */}
      {stats && !error && (
        <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1 font-mono text-[10px] leading-tight text-stone-200 sm:text-[11px]">
          <span className="rounded bg-stone-950/70 px-2 py-1 backdrop-blur-sm">
            {stats.fps} fps · {preset.toUpperCase()} · {stats.mode.toUpperCase()}
            {stats.mode === 'walk' ? (stats.grounded ? ' · grounded' : ' · airborne') : ''}
            {' '}· {stats.resPct}% res
          </span>
          <span className="rounded bg-stone-950/70 px-2 py-1 backdrop-blur-sm">
            pos [{stats.x.toFixed(1)}, {stats.y.toFixed(1)}, {stats.z.toFixed(1)}]
          </span>
          <span className="rounded bg-stone-950/70 px-2 py-1 backdrop-blur-sm">
            chunk-origin [{stats.originX}, {stats.originZ}] (fp32-exact)
          </span>
        </div>
      )}

      {/* pause / error controls — top right */}
      <div className="absolute right-3 top-3 flex gap-2">
        {!error && (
          <button
            type="button"
            onClick={() => setRunning((v) => !v)}
            className="rounded bg-stone-950/70 px-2 py-1 font-mono text-[10px] text-stone-200 backdrop-blur-sm transition hover:bg-stone-900 sm:text-[11px]"
          >
            {running ? '❚❚ pause' : '▶ resume'}
          </button>
        )}
      </div>

      {/* controls hint — bottom left */}
      {!error && (
        <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-stone-950/70 px-2 py-1 font-mono text-[10px] text-stone-300 backdrop-blur-sm sm:text-[11px]">
          drag&nbsp;=look · WASD&nbsp;move · shift&nbsp;sprint · F&nbsp;walk/fly ·
          space&nbsp;jump/up · C&nbsp;down
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <p className="max-w-md text-center font-mono text-sm text-red-300">
            WebGL2 failed: {error}
          </p>
        </div>
      )}
    </div>
  );
}
