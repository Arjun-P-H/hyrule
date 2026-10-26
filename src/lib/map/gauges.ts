/**
 * The two gauges beside the minimap: the air's temperature round the hero, and how loud the hero is.
 * ---------------------------------------------------------------------------
 * Temperature: the world's climate (terrainMath.ts temperature: the sea-level field less the lapse with
 * height, so a climb to the peaks or the sky islands is a climb into the cold) with the day's swing on
 * top — warmest at 15:00, coldest at 03:00, wider where the air is dry.
 *
 * Noise: what a listener would hear (the sound meter): nothing standing still, a murmur sneaking, a
 * steady beat running, a roar sprinting, a spike on a hard landing or a blow. Quick to rise, slow to fall.
 */

import { moisture, temperature } from '../terrainMath';
import { HERO } from '../constants';
import type { HeroState } from '../hero/locomotion';

/** The temperature the gauge's arc starts and ends at (°C). */
export const TEMP_MIN = -15;
export const TEMP_MAX = 45;

/** The air temperature (°C) at world (x, z), height y, hour of the day (0–24). */
export function ambientTemp(x: number, y: number, z: number, seed: number, hour: number): number {
  const dry = Math.min(1, Math.max(0, 1 - moisture(x, z, 0, seed)));
  const swing = 2 + 4 * dry;
  return temperature(x, z, y, seed) + swing * Math.cos(((hour - 15) / 24) * 2 * Math.PI);
}

/** Where a temperature falls on the gauge's arc, 0 (coldest) … 1 (hottest). */
export function tempFraction(t: number): number {
  return Math.min(1, Math.max(0, (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)));
}

/** A word for the temperature, for the gauge's tooltip. */
export function tempWord(t: number): string {
  if (t < -5) return 'Freezing';
  if (t < 5) return 'Cold';
  if (t < 14) return 'Cool';
  if (t < 26) return 'Mild';
  if (t < 34) return 'Warm';
  return 'Scorching';
}

/** How loud the hero is on its feet, 0–1: what the sound meter settles to. */
export function heroNoise(h: Readonly<HeroState>): number {
  const speed = Math.hypot(h.vx, h.vz);
  switch (h.move) {
    case 'ground':
      if (speed < 0.4) return 0;
      if (h.surfing) return 0.62;
      if (h.sliding) return 0.55;
      if (h.sneaking) return 0.1;
      if (h.sprinting) return 0.9;
      return speed > HERO.runSpeed * 0.6 ? 0.55 : 0.3;
    case 'swim': return speed < 0.4 ? 0.05 : speed > HERO.swimSpeed * 1.3 ? 0.6 : 0.32;
    case 'climb': return speed < 0.2 ? 0 : 0.28;
    case 'surf': return 0.62;
    case 'glide':
    case 'air':
      return 0.05;
  }
}

/** How loud a first-person walk is (the view on foot: the speed against the walking pace). */
export function walkNoise(speed: number, sprinting: boolean): number {
  if (speed < 0.4) return 0;
  return sprinting ? 0.9 : 0.35;
}

type NoiseMeter = {
  /** The level now (0–1). */
  level: number;
  /** A spike (a landing, a blow) dying away. */
  spike: number;
  /** The landings counted so far (locomotion.ts HeroState.landings): a new one is a spike. */
  landings: number;
};

export function createNoiseMeter(): NoiseMeter {
  return { level: 0, spike: 0, landings: -1 };
}

/** A sudden sound of this loudness (0–1). */
export function noisePulse(m: NoiseMeter, loudness: number): void {
  if (loudness > m.spike) m.spike = Math.min(1, loudness);
}

/** A hard landing is a spike: its loudness for the downward speed it came down at (m/s). */
export function landingLoudness(impact: number): number {
  return Math.min(1, Math.max(0, (impact - 4) / 14));
}

/** Move the meter on by dt seconds towards the `target` loudness; returns the level shown. */
export function stepNoise(m: NoiseMeter, dt: number, target: number): number {
  const rate = target > m.level ? 16 : 4;
  m.level += (target - m.level) * (1 - Math.exp(-rate * dt));
  if (m.level < 1e-3 && target === 0) m.level = 0;
  m.spike = Math.max(0, m.spike - dt * 2.2);
  return Math.max(m.level, m.spike);
}
