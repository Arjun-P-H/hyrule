/**
 * The time of day (pure). Game time is `days` since day 0, 00:00; the hour is
 * its fraction × 24. A day lasts `dayMinutes` of real time on average, but
 * not evenly: time lingers where the light is most beautiful (golden hour,
 * sunset, blue hour and their mirror at dawn, ×0.55 of the mean pace) and
 * hurries through the deep night (×1.7) — about 7 of a 24-minute day's
 * minutes are twilight, 6 night and 11 day. The pace depends only on the
 * sun's elevation and is normalised so a whole day still takes dayMinutes.
 */

import { pathElevation, sunTheta } from './celestial';

type SkyClock = {
  /** Game days since day 0, 00:00. */
  days: number;
  /** Time stands still. */
  paused: boolean;
  /** Real minutes per game day. */
  dayMinutes: number;
};

/** Day lengths the settings offer (real minutes per game day). */
export const DAY_MINUTES = [12, 24, 48] as const;

export function createSkyClock(hour = 15.5, dayMinutes = 24): SkyClock {
  return { days: hour / 24, paused: false, dayMinutes };
}

const ss = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Unnormalised pace at a sun elevation (degrees). */
function rawPace(elevDeg: number): number {
  const night = 1 - ss(-24, -13, elevDeg);
  const day = ss(18, 28, elevDeg);
  const twilight = Math.max(0, 1 - night - day);
  return night * 1.7 + twilight * 0.55 + day * 1.05;
}

const hourElevDeg = (hour: number) => (pathElevation(sunTheta(hour)) * 180) / Math.PI;

// Normalisation: ∫ dh / pace over the day must be 24 h, so a day takes dayMinutes.
const PACE_SCALE = (() => {
  const n = 2880;
  let s = 0;
  for (let i = 0; i < n; i++) s += 1 / rawPace(hourElevDeg(((i + 0.5) / n) * 24));
  return s / n;
})();

/** Game hours per real second at this hour, relative to the day's mean. */
export function pace(hour: number): number {
  return rawPace(hourElevDeg(hour)) * PACE_SCALE;
}

export function hourOf(days: number): number {
  return (days - Math.floor(days)) * 24;
}

/** Move the clock on by dt real seconds (nothing while paused). */
export function advanceClock(c: SkyClock, dt: number): void {
  if (c.paused || !(dt > 0)) return;
  const perSecond = 1 / (c.dayMinutes * 60); // days per real second at the mean pace
  c.days += dt * perSecond * pace(hourOf(c.days));
}

/** Jump to an hour of the current day. */
export function setHour(c: SkyClock, hour: number): void {
  c.days = Math.floor(c.days) + (((hour % 24) + 24) % 24) / 24;
}

type DayPhase = 'Night' | 'Dawn' | 'Sunrise' | 'Morning glow' | 'Morning' | 'Midday' | 'Afternoon' | 'Golden hour' | 'Sunset' | 'Dusk';

/** A name for the light at this hour. */
export function dayPhase(hour: number): DayPhase {
  const e = hourElevDeg(hour);
  const am = hour < 12;
  if (e < -12) return 'Night';
  if (e < -4) return am ? 'Dawn' : 'Dusk';
  if (e < 2) return am ? 'Sunrise' : 'Sunset';
  if (e < 12) return am ? 'Morning glow' : 'Golden hour';
  if (e < 40) return am ? 'Morning' : 'Afternoon';
  return 'Midday';
}

/** "HH:MM" for an hour of the day. */
export function clockText(hour: number): string {
  const m = Math.floor(hour * 60) % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}
