import { describe, it, expect } from 'bun:test';
import {
  DRS_HARDWARE,
  DRS_SOFTWARE,
  createDrsState,
  drsStep,
  updateFpsEma,
  type DrsConfig,
} from '../src/lib/engine/drs';

/**
 * §28 — closed-loop DRS harness. Simulates what the engine's frame loop does
 * with the real controller: 60 Hz vsync, a fence that only lets a new frame
 * start once the GPU finished the previous one, the per-frame FPS EMA, and a
 * GPU cost model cost(scale) = c0 + c1·scale² ms (pixel-proportional) with
 * deterministic ±jitter. Returns the scale trajectory.
 */
function simulate(opts: {
  cfg: DrsConfig;
  c0: number;
  c1: number;
  seconds: number;
  jitter?: number;
  load?: (tMs: number) => number; // optional time-varying multiplier on c1
}) {
  const V = 1000 / 60;
  const state = createDrsState(opts.cfg);
  let rnd = 0x9e3779b9;
  const noise = () => {
    rnd = (Math.imul(rnd, 1664525) + 1013904223) >>> 0;
    return rnd / 4294967296 - 0.5;
  };
  let fpsEma = 60;
  let gpuBusyUntil = 0;
  let lastRender = 0;
  const changes: Array<{ t: number; scale: number; dir: number }> = [];
  const fpsTrace: number[] = [];
  const t0 = 0;
  for (let k = 1; k * V <= opts.seconds * 1000; k++) {
    const now = k * V;
    if (gpuBusyUntil > now) continue; // fence not signalled: skip this rAF
    const load = opts.load ? opts.load(now) : 1;
    const cost = (opts.c0 + opts.c1 * load * state.scale * state.scale) * (1 + (opts.jitter ?? 0.1) * noise());
    gpuBusyUntil = now + cost;
    fpsEma = updateFpsEma(fpsEma, Math.min((now - lastRender) / 1000, 1.0));
    lastRender = now;
    const before = state.scale;
    if (drsStep(state, opts.cfg, fpsEma, now, t0)) {
      changes.push({ t: now, scale: state.scale, dir: Math.sign(state.scale - before) });
    }
    fpsTrace.push(fpsEma);
  }
  let reversals = 0;
  const reversalTimes: number[] = [];
  for (let i = 1; i < changes.length; i++) {
    if (changes[i].dir !== changes[i - 1].dir) { reversals++; reversalTimes.push(changes[i].t); }
  }
  return { state, changes, reversals, reversalTimes, fpsEma, fpsTrace };
}

describe('§28 DRS controller — closed loop on the engine’s real code', () => {
  it('sustained 14 FPS load: converges by downscaling only, then holds (no thrash)', () => {
    // 71.4 ms at full scale ≈ 14 FPS
    const r = simulate({ cfg: DRS_HARDWARE, c0: 5, c1: 66.4, seconds: 120 });
    expect(r.changes.length).toBeGreaterThan(0);
    expect(r.changes.every((c) => c.dir < 0)).toBe(true);
    expect(r.reversals).toBe(0);
    // settled: no change in the last 60 s, EMA inside the deadband
    expect(r.changes.filter((c) => c.t > 60_000).length).toBe(0);
    expect(r.fpsEma).toBeGreaterThanOrEqual(DRS_HARDWARE.downFps);
    expect(r.state.scale).toBeGreaterThan(DRS_HARDWARE.minScale);
  });

  it('load parked on the downscale threshold (20–30 FPS vsync steps) does not oscillate', () => {
    for (const c1 of [30, 33, 36, 40, 45]) {
      const r = simulate({ cfg: DRS_HARDWARE, c0: 2, c1, seconds: 180, jitter: 0.25 });
      expect(r.reversals).toBeLessThanOrEqual(1);
      for (let i = 1; i < r.reversalTimes.length; i++) {
        expect(r.reversalTimes[i] - r.reversalTimes[i - 1]).toBeGreaterThan(10_000);
      }
    }
  });

  it('load parked on the upscale threshold (60↔30 FPS boundary) does not oscillate', () => {
    for (const c1 of [14, 15.5, 16.5, 17.5, 19]) {
      const r = simulate({ cfg: DRS_HARDWARE, c0: 1, c1, seconds: 180, jitter: 0.25 });
      expect(r.reversals).toBeLessThanOrEqual(1);
    }
  });

  it('load step (sky → terrain → sky) adapts in both directions with bounded changes', () => {
    const r = simulate({
      cfg: DRS_HARDWARE,
      c0: 2,
      c1: 70,
      seconds: 240,
      load: (t) => (t < 60_000 ? 0.15 : t < 150_000 ? 1.0 : 0.15),
    });
    expect(r.changes.some((c) => c.dir < 0)).toBe(true);
    expect(r.changes.some((c) => c.dir > 0)).toBe(true);
    // each regime change produces one monotone run of steps, not a flip-flop
    expect(r.reversals).toBeLessThanOrEqual(3);
    expect(r.changes.length).toBeLessThan(30);
  });

  it('never leaves [minScale, maxScale] for hardware and software configs', () => {
    for (const cfg of [DRS_HARDWARE, DRS_SOFTWARE]) {
      const heavy = simulate({ cfg, c0: 50, c1: 400, seconds: 60 });
      const light = simulate({ cfg, c0: 0.5, c1: 1, seconds: 60 });
      for (const r of [heavy, light]) {
        for (const c of r.changes) {
          expect(c.scale).toBeGreaterThanOrEqual(cfg.minScale - 1e-12);
          expect(c.scale).toBeLessThanOrEqual(cfg.maxScale + 1e-12);
        }
      }
      expect(heavy.state.scale).toBeCloseTo(cfg.minScale, 12);
      expect(light.state.scale).toBeCloseTo(cfg.maxScale, 12);
    }
  });

  it('one step cannot cross the deadband: step cost ratios < upFps/downFps', () => {
    const cfg = DRS_HARDWARE;
    expect(cfg.upFps / cfg.downFps).toBeGreaterThan(1 / (cfg.downFactor * cfg.downFactor));
    expect(cfg.upFps / cfg.downFps).toBeGreaterThan(cfg.upFactor * cfg.upFactor);
    // the 60 → 30 FPS vsync step (2×) also fits inside the deadband ratio
    expect(cfg.upFps / cfg.downFps).toBeGreaterThan(2);
  });

  it('warm-up window ignores low frame rates', () => {
    const state = createDrsState(DRS_HARDWARE);
    for (let t = 0; t <= DRS_HARDWARE.warmupMs; t += 16) expect(drsStep(state, DRS_HARDWARE, 5, t, 0)).toBe(false);
    expect(state.scale).toBe(DRS_HARDWARE.maxScale);
  });
});
