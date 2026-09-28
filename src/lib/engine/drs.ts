/**
 * §28 — Dynamic Resolution Scaling controller (pure, framework-free).
 * ---------------------------------------------------------------------------
 * Extracted from useTerrainEngine so the exact code the engine runs can be
 * exercised by tests in a closed loop (tests/drs.test.ts) instead of a copy.
 *
 * Hysteresis: a [downFps, upFps] deadband with asymmetric dwell times. One
 * downscale step multiplies pixel count by downFactor² (0.7225) and one
 * upscale step by upFactor² (1.3225); since upFps / downFps (2.05) exceeds both
 * the per-step cost ratio and the 60→30 FPS vsync quantization step, a single
 * step can never carry the frame rate from one threshold across the other.
 */

export type DrsConfig = {
  warmupMs: number;       // ignore everything before this (shader compile, first frames)
  downFps: number;        // downscale when the frame-rate EMA falls below this
  upFps: number;          // upscale when it rises above this
  downDwellMs: number;    // min time between consecutive downscales
  upDwellMs: number;      // min time between consecutive upscales
  upCooldownMs: number;   // min time after a downscale before any upscale
  downFactor: number;     // linear scale multiplier per downscale
  upFactor: number;       // linear scale multiplier per upscale
  minScale: number;
  maxScale: number;
};

export type DrsState = {
  scale: number;
  lastDownscaleAt: number;
  lastUpscaleAt: number;
};

export const DRS_HARDWARE: DrsConfig = {
  warmupMs: 2500,
  downFps: 22,
  upFps: 45,
  downDwellMs: 1500,
  upDwellMs: 4000,
  upCooldownMs: 5000,
  downFactor: 0.85,
  upFactor: 1.15,
  minScale: 0.35,
  maxScale: 1.0,
};

/** Software rasterizers (SwiftShader, llvmpipe) start and stay small. */
export const DRS_SOFTWARE: DrsConfig = { ...DRS_HARDWARE, minScale: 0.15, maxScale: 0.25 };

export function createDrsState(cfg: DrsConfig): DrsState {
  return { scale: cfg.maxScale, lastDownscaleAt: 0, lastUpscaleAt: 0 };
}

/** Per-rendered-frame EMA of instantaneous FPS (α = 0.1), as the HUD shows it. */
export function updateFpsEma(fpsEma: number, renderDtSeconds: number): number {
  return fpsEma + (1 / Math.max(renderDtSeconds, 1e-4) - fpsEma) * 0.1;
}

/**
 * Advance the controller once per rendered frame. Mutates `state`; returns
 * true when the resolution scale changed (caller resizes the canvas).
 */
export function drsStep(state: DrsState, cfg: DrsConfig, fpsEma: number, now: number, t0: number): boolean {
  if (now - t0 <= cfg.warmupMs) return false;
  if (fpsEma < cfg.downFps && now - state.lastDownscaleAt > cfg.downDwellMs && state.scale > cfg.minScale) {
    state.scale = Math.max(cfg.minScale, state.scale * cfg.downFactor);
    state.lastDownscaleAt = now;
    state.lastUpscaleAt = now; // restart the upscale dwell: asymmetric recovery
    return true;
  }
  if (
    fpsEma > cfg.upFps &&
    now - state.lastUpscaleAt > cfg.upDwellMs &&
    now - state.lastDownscaleAt > cfg.upCooldownMs &&
    state.scale < cfg.maxScale
  ) {
    state.scale = Math.min(cfg.maxScale, state.scale * cfg.upFactor);
    state.lastUpscaleAt = now;
    return true;
  }
  return false;
}
