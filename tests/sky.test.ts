import { describe, expect, test } from 'bun:test';
import {
  horizonDip, moonElongation, moonIllumination, moonTheta, pathDir, pathElevation, POLE, SKY, starBasis, sunTheta, wrapTheta,
} from '../src/lib/sky/celestial';
import { advanceClock, clockText, createSkyClock, dayPhase, hourOf, pace, setHour } from '../src/lib/sky/clock';
import { transmittance } from '../src/lib/sky/atmosphere';
import { computeLightRig, createLightRig, keyAltitude, KEY_ALTS } from '../src/lib/sky/lightRig';

const DEG = Math.PI / 180;
const lum = (c: ArrayLike<number>, o = 0) => 0.2126 * c[o] + 0.7152 * c[o + 1] + 0.0722 * c[o + 2];

describe('celestial', () => {
  test('the sun rises due east at 06:00, culminates 60° high toward −z at noon and sets due west at 18:00', () => {
    const rise = pathDir(sunTheta(6));
    expect(rise[0]).toBeCloseTo(1, 9);
    expect(rise[1]).toBeCloseTo(0, 9);
    const noon = pathDir(sunTheta(12));
    expect(Math.asin(noon[1]) / DEG).toBeCloseTo(90 - SKY.latitude / DEG, 6);
    expect(noon[2]).toBeLessThan(0);
    expect(noon[0]).toBeCloseTo(0, 9);
    const set = pathDir(sunTheta(18));
    expect(set[0]).toBeCloseTo(-1, 9);
    expect(pathDir(sunTheta(0))[1]).toBeLessThan(-0.8);
  });

  test('path directions are unit length, their elevation matches, and the pole is square to the path', () => {
    for (let h = 0; h < 24; h += 0.37) {
      const d = pathDir(sunTheta(h));
      expect(Math.hypot(...d)).toBeCloseTo(1, 12);
      expect(pathElevation(sunTheta(h))).toBeCloseTo(Math.asin(d[1]), 12);
      expect(d[0] * POLE[0] + d[1] * POLE[1] + d[2] * POLE[2]).toBeCloseTo(0, 12);
    }
  });

  test('path angles wrap into (−π/2, 3π/2], so a body just below the eastern horizon is a small negative angle', () => {
    expect(sunTheta(5.9)).toBeLessThan(0);
    expect(sunTheta(5.9)).toBeGreaterThan(-0.05);
    for (let h = 0; h < 24; h += 0.25) {
      const t = sunTheta(h);
      expect(t).toBeGreaterThan(-Math.PI / 2);
      expect(t).toBeLessThanOrEqual(1.5 * Math.PI + 1e-12);
    }
    expect(wrapTheta(-0.1)).toBeCloseTo(-0.1, 12);
    expect(wrapTheta(2 * Math.PI + 0.3)).toBeCloseTo(0.3, 12);
  });

  test('the moon goes round in SKY.moonPeriod days: new with the sun, full opposite it, rising later every day', () => {
    const perDay = moonElongation(1) - moonElongation(0);
    expect(perDay).toBeCloseTo((2 * Math.PI) / SKY.moonPeriod, 9);
    // the first evening: a waxing gibbous well up in the east
    const days = 18 / 24;
    expect(moonIllumination(moonElongation(days))).toBeGreaterThan(0.8);
    expect(moonIllumination(moonElongation(days))).toBeLessThan(0.95);
    const m = pathDir(moonTheta(18, days));
    expect(m[0]).toBeGreaterThan(0.5);
    expect(Math.asin(m[1]) / DEG).toBeGreaterThan(25);
    // full moon: opposite the sun
    for (let d = 0; d < SKY.moonPeriod; d += 0.01) {
      const e = moonElongation(d);
      if (Math.abs(e - Math.PI) < 0.01) {
        const s = pathDir(sunTheta(21)), mo = pathDir(moonTheta(21, d));
        expect(s[0] * mo[0] + s[1] * mo[1] + s[2] * mo[2]).toBeLessThan(-0.99);
      }
    }
    expect(moonIllumination(0)).toBeCloseTo(0, 12);
    expect(moonIllumination(Math.PI)).toBeCloseTo(1, 12);
  });

  test('the land\'s horizon dips with altitude: ~4.7° at 1 km', () => {
    expect(horizonDip(0)).toBe(0);
    expect(horizonDip(-5)).toBe(0);
    expect(horizonDip(1000) / DEG).toBeGreaterThan(4.5);
    expect(horizonDip(1000) / DEG).toBeLessThan(4.9);
  });

  test('the star frame is a rotation that turns with the sun about the pole', () => {
    const [a, b, c] = starBasis(0.7, 0.3);
    const dot = (p: number[], q: number[]) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    expect(dot(a, a)).toBeCloseTo(1, 12);
    expect(dot(b, b)).toBeCloseTo(1, 12);
    expect(dot(a, b)).toBeCloseTo(0, 12);
    expect(dot(a, c)).toBeCloseTo(0, 12);
    expect(c).toEqual([...POLE]);
    // a star at the celestial origin sits on the path, a fixed angle from the sun
    const s = pathDir(0.7), s2 = pathDir(0.7 + 0.3);
    expect(dot(a, s2)).toBeCloseTo(1, 12);
    expect(dot(a, s)).toBeCloseTo(Math.cos(0.3), 12);
  });
});

describe('clock', () => {
  test('a whole day takes dayMinutes of real time, however unevenly', () => {
    const c = createSkyClock(0, 24);
    let t = 0;
    const dt = 1 / 60;
    while (c.days < 1) { advanceClock(c, dt); t += dt; }
    expect(Math.abs(t / 60 - 24)).toBeLessThan(0.1);
  });

  test('time lingers through golden hour and dusk and hurries through the night', () => {
    expect(pace(17.6)).toBeLessThan(0.65);
    expect(pace(18.4)).toBeLessThan(0.65);
    expect(pace(6.2)).toBeLessThan(0.65);
    expect(pace(0)).toBeGreaterThan(1.5);
    expect(pace(12)).toBeGreaterThan(0.95);
    expect(pace(12)).toBeLessThan(1.2);
    // it changes smoothly
    for (let h = 0; h < 24; h += 0.05) expect(Math.abs(pace(h + 0.05) - pace(h))).toBeLessThan(0.12);
  });

  test('pausing stops time; setHour keeps the day; the phase names follow the sun', () => {
    const c = createSkyClock(10);
    c.paused = true;
    advanceClock(c, 100);
    expect(hourOf(c.days)).toBeCloseTo(10, 9);
    c.paused = false;
    c.days = 3.5;
    setHour(c, 18.25);
    expect(Math.floor(c.days)).toBe(3);
    expect(hourOf(c.days)).toBeCloseTo(18.25, 9);
    setHour(c, -1);
    expect(hourOf(c.days)).toBeCloseTo(23, 9);
    expect(dayPhase(0)).toBe('Night');
    expect(dayPhase(5.5)).toBe('Dawn');
    expect(dayPhase(6.05)).toBe('Sunrise');
    expect(dayPhase(12)).toBe('Midday');
    expect(dayPhase(17.5)).toBe('Golden hour');
    expect(dayPhase(18.05)).toBe('Sunset');
    expect(dayPhase(18.5)).toBe('Dusk');
    expect(clockText(18.25)).toBe('18:15');
    expect(clockText(0)).toBe('00:00');
    expect(clockText(23.999)).toBe('23:59');
  });
});

describe('atmosphere', () => {
  test('sunlight through the air: reddens and dims toward the horizon, none below the ground', () => {
    const zen = transmittance(0, 1);
    for (const c of zen) expect(c).toBeGreaterThan(0.7);
    expect(zen[0]).toBeGreaterThan(zen[1]);
    expect(zen[1]).toBeGreaterThan(zen[2]);
    let prev = zen;
    for (const e of [60, 30, 15, 8, 4, 2, 1]) {
      const t = transmittance(0, Math.sin(e * DEG));
      for (let c = 0; c < 3; c++) expect(t[c]).toBeLessThan(prev[c] + 1e-12);
      expect(t[2] / t[0]).toBeLessThan(prev[2] / prev[0] + 1e-12);
      prev = t;
    }
    expect(transmittance(0.5, -0.3)).toEqual([0, 0, 0]);
    // higher up, less air: the same low sun is brighter on a peak
    expect(transmittance(1.5, Math.sin(2 * DEG))[1]).toBeGreaterThan(transmittance(0, Math.sin(2 * DEG))[1] * 1.3);
  });
});

describe('light rig', () => {
  const rig = createLightRig();
  const at = (hour: number, days = 0, camY = 50) => computeLightRig(days + hour / 24, camY, rig);

  test('midday light is a near-neutral white; the low sun turns it gold, then orange, then red', () => {
    at(12);
    const k = rig.keyCols;
    expect(rig.keyIsSun).toBe(true);
    expect(k[2] / k[0]).toBeGreaterThan(0.93);
    const warmth: number[] = [];
    for (const h of [12, 16, 17, 17.5, 17.85]) { at(h); warmth.push(rig.keyCols[2] / rig.keyCols[0]); }
    for (let i = 1; i < warmth.length; i++) expect(warmth[i]).toBeLessThan(warmth[i - 1]);
    expect(warmth[warmth.length - 1]).toBeLessThan(0.1);
  });

  test('alpenglow: after the valley loses the sun, the peaks keep red light for a while', () => {
    at(18.2);
    expect(lum(rig.keyCols, 0)).toBe(0);
    const top = (KEY_ALTS - 1) * 4;
    expect(lum(rig.keyCols, top)).toBeGreaterThan(0.1);
    expect(rig.keyCols[top]).toBeGreaterThan(rig.keyCols[top + 2] * 4);
    // lit from the top down: each key at least as bright as the one below it
    for (let k = 1; k < KEY_ALTS; k++) expect(lum(rig.keyCols, 4 * k)).toBeGreaterThanOrEqual(lum(rig.keyCols, 4 * (k - 1)));
    expect(keyAltitude(KEY_ALTS - 1)).toBe(1600);
  });

  test('the key light changes smoothly through the day, and passes from sun to moon in the dark', () => {
    let prev: Float32Array | null = null;
    let switches = 0, prevSun = true;
    for (let m = 0; m < 1440; m++) {
      at(m / 60, 0.25);
      // No jumps: under a tenth of the midday light per game minute (the low sun reddens fast, but smoothly).
      if (prev) {
        for (let i = 0; i < rig.keyCols.length; i++) expect(Math.abs(rig.keyCols[i] - prev[i])).toBeLessThan(0.34);
      }
      if (rig.keyIsSun !== prevSun) {
        switches++;
        expect(lum(rig.keyCols, 0)).toBeLessThan(0.01);
        expect(lum(rig.keyCols, (KEY_ALTS - 1) * 4)).toBeLessThan(0.01);
      }
      prevSun = rig.keyIsSun;
      prev = rig.keyCols.slice();
    }
    expect(switches).toBe(2);
  });

  test('night is moonlit or starlit: dim and cool, never brighter than a tenth of day', () => {
    at(12);
    const day = lum(rig.keyCols, 0);
    for (const d of [0, 1, 2, 3, 4, 5, 6, 7]) {
      at(23, d);
      expect(rig.keyIsSun).toBe(false);
      expect(lum(rig.keyCols, 0)).toBeLessThan(day / 10);
      // silver-blue once the moon is well up (low down, the air warms it like the sun)
      if (rig.moonElev > 25 * DEG && rig.moonIllum > 0.3) expect(rig.keyCols[2]).toBeGreaterThan(rig.keyCols[0]);
      expect(rig.stars).toBe(1);
      expect(rig.night).toBe(1);
    }
  });

  test('the moon lights the land a thousand times less than the sun, and its disc fades with it', () => {
    at(12);
    const day = lum(rig.keyCols, 0);
    at(22, 1); // full moon, high
    expect(rig.moonIllum).toBeGreaterThan(0.95);
    expect(lum(rig.keyCols, 0)).toBeGreaterThan(day / 1500);
    expect(lum(rig.keyCols, 0)).toBeLessThan(day / 500);
    expect(lum(rig.moonDisc)).toBeGreaterThan(1);
    at(12, 1);
    expect(lum(rig.moonDisc)).toBeLessThan(0.5);
  });

  test('morning mist forms before dawn and is gone by ten', () => {
    at(6);
    expect(rig.mist).toBeGreaterThan(0.9);
    at(10);
    expect(rig.mist).toBe(0);
    at(15);
    expect(rig.mist).toBe(0);
  });
});
