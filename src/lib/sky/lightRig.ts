/**
 * The light of the hour (pure; allocation-free per frame once warm): where
 * the sun and moon stand, which of them lights the land (the key light), the
 * colour its light arrives with at each altitude, how the sky shows them, and
 * how the camera exposes and grades the frame.
 * ---------------------------------------------------------------------------
 * Colour from the air, not from a palette: the key light's colour at a point
 * is the sunlight (or moonlight) the atmosphere lets through toward that
 * point's own horizon (atmosphere.ts). Low sun → long path → golden, then
 * orange, then red light; and because the land's horizon dips with altitude
 * (celestial.ts, horizonDip), peaks and sky islands still hold red light
 * after the valleys have lost the sun (alpenglow). The colour is tabulated at
 * KEY_ALTS altitudes packed toward the ground, where it changes fastest; the
 * shaders interpolate between them.
 *
 * The sun keys the land until it is 7.5° down (nothing on the land sees it
 * then), the moon from 8° down, fading in over 3°: the switch happens in the
 * dark, so no shadow jumps. The moon lights the land at ~1/900 of the sun,
 * darker than the blue hour's sky, as it is: the camera's exposure adapts to
 * the light on the land (sky.wgsl.ts, the probe), only partly, so dusk reads
 * darker than day and night darker than dusk, but always readable.
 */

import { transmittance } from './atmosphere';
import { horizonDip, moonElongation, moonIllumination, moonTheta, pathDir, pathElevation, starBasis, sunTheta, type V3 } from './celestial';

const DEG = Math.PI / 180;

export const KEY_ALTS = 8;
/** Altitude (m) of the highest key; above it the key light is the top key's. */
export const KEY_TOP = 1600;
/** Altitude (m) of key k: dense low down, where the land's horizon dips fastest (dip ∝ √y). */
export const keyAltitude = (k: number) => KEY_TOP * (k / (KEY_ALTS - 1)) ** 2;

// Transmittance from each key altitude, by elevation above that altitude's own horizon (0°–90°, 0.25° steps).
const T_STEP = 0.25 * DEG;
const T_N = 361;
let tables: Float32Array | null = null;
function tabulated(): Float32Array {
  if (tables) return tables;
  const t = new Float32Array(KEY_ALTS * T_N * 3);
  const o: V3 = [0, 0, 0];
  for (let k = 0; k < KEY_ALTS; k++) {
    for (let i = 0; i < T_N; i++) {
      transmittance(keyAltitude(k) / 1000, Math.sin(Math.max(i * T_STEP, 1e-4)), o);
      t.set(o, (k * T_N + i) * 3);
    }
  }
  return (tables = t);
}

/** Transmittance at key altitude k toward elevation e (rad) above its horizon; below the horizon it fades over the disc's width. */
function keyTrans(k: number, e: number, out: V3): V3 {
  const t = tabulated();
  const x = Math.min(Math.max(e, 0), 90 * DEG) / T_STEP;
  const i = Math.min(Math.floor(x), T_N - 2), f = x - i;
  const fade = smooth(-0.5 * DEG, 0.5 * DEG, e);
  const a = (k * T_N + i) * 3;
  for (let c = 0; c < 3; c++) out[c] = (t[a + c] * (1 - f) + t[a + 3 + c] * f) * fade;
  return out;
}

function smooth(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Piecewise curve through [x, y] points (x increasing), smoothstepped between them. */
function curve(pts: ReadonlyArray<readonly [number, number]>, x: number): number {
  if (x <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    if (x <= x1) {
      const [x0, y0] = pts[i - 1];
      return y0 + (y1 - y0) * smooth(x0, x1, x);
    }
  }
  return pts[pts.length - 1][1];
}

// Sunlight above the air: the sun at 45° over sea level shines 3.4 × a slightly warm white (the classic sky's sun).
const T45 = transmittance(0, Math.sin(45 * DEG));
export const SUN_E: V3 = [3.4 * 1.0 / T45[0], 3.4 * 0.97 / T45[1], 3.4 * 0.93 / T45[2]];
/** Full-moon light above the air: 1/900 of the sun's, silver-blue (the eye's night vision, not the moon's slightly red sunlight). */
export const MOON_E: V3 = [(3.4 / 900) * 0.74 / T45[0], (3.4 / 900) * 0.86 / T45[1], (3.4 / 900) * 1.0 / T45[2]];
/** Share of the full moon's light at a lit share of its disc (a half moon gives ~a quarter). */
export const moonLightShare = (illum: number) => illum ** 1.8;
/** The sun's disc as seen (radiance over the key light: blinding by day, a red coal at sunset). */
const SUN_DISC_GAIN = 14;
/** The moon's disc on the display (after exposure): pale by day, bright at night. */
const MOON_DISC_DAY = 0.4;
const MOON_DISC_NIGHT = 1.3;

export type LightRig = {
  hour: number;
  sun: V3;
  moon: V3;
  sunTheta: number;
  moonTheta: number;
  /** Elevations (rad) above the horizontal. */
  sunElev: number;
  moonElev: number;
  /** The moon's angle from the sun (0 new, π full) and the lit share of its disc. */
  elongation: number;
  moonIllum: number;
  /** Direction toward the key light and its path angle (the terrain horizons compare against it). */
  key: V3;
  keyTheta: number;
  /** The key light is the sun (else the moon). */
  keyIsSun: boolean;
  /** Key light colour at the KEY_ALTS altitudes, rgb + 0. */
  keyCols: Float32Array;
  /** The sun's disc seen from the camera (radiance), and the moon's (display brightness, before its phase). */
  sunDisc: V3;
  moonDisc: V3;
  /** Saturation, night (0 day → 1 night: bluer, less saturated), star visibility. */
  saturation: number;
  night: number;
  stars: number;
  /** Haze density (× the classic fog), and morning mist over the low land (0–1). */
  haze: number;
  mist: number;
  /**
   * Clouds: a shift of their coverage threshold (fair-weather clouds build
   * through the afternoon to catch the sunset, then dissolve after dusk, so
   * nights are clear for the stars and mornings fresh), and how brightly they
   * are lit (dim silver under the moon).
   */
  cloudShift: number;
  cloudLight: number;
  /** How much of the Milky Way shows (washed out by a bright moon). */
  milkyWay: number;
  /** Twilight (0–1): while it lasts, a sky that shows dim on the display is lifted toward a luminous pastel (sky.wgsl.ts). */
  twilight: number;
  /** World → celestial rows for the stars. */
  starRows: [V3, V3, V3];
};

export function createLightRig(): LightRig {
  return {
    hour: 0, sun: [0, 1, 0], moon: [0, 1, 0], sunTheta: 0, moonTheta: 0, sunElev: 0, moonElev: 0,
    elongation: 0, moonIllum: 0, key: [0, 1, 0], keyTheta: 0, keyIsSun: true,
    keyCols: new Float32Array(KEY_ALTS * 4), sunDisc: [0, 0, 0], moonDisc: [0, 0, 0],
    saturation: 1, night: 0, stars: 0, haze: 1, mist: 0, cloudShift: 0, cloudLight: 1, milkyWay: 0, twilight: 0,
    starRows: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
  };
}

/** Turns the star field so the Milky Way arches across the evening sky. */
const STAR_OFFSET = 2.1;

// Saturation by sun elevation: richer in the golden hour, eased at sunset (the light is coloured enough), grey-blue at night.
const SATURATION: ReadonlyArray<readonly [number, number]> = [[-16, 0.84], [-8, 0.96], [-3, 1.05], [2, 1.02], [8, 1.1], [22, 1.07]];
// Haze by sun elevation: golden in the low sun, thinner at dusk (the land must not drown in the twilight air), crisp by day.
const HAZE: ReadonlyArray<readonly [number, number]> = [[-14, 0.8], [-4, 0.95], [3, 1.12], [14, 1.05], [35, 0.9]];

const tmp: V3 = [0, 0, 0];

/**
 * The rig at `days` since day 0, 00:00, for a camera at altitude camY (m).
 * Writes into and returns `out`.
 */
export function computeLightRig(days: number, camY: number, out: LightRig): LightRig {
  const hour = (days - Math.floor(days)) * 24;
  out.hour = hour;
  const ts = sunTheta(hour), tm = moonTheta(hour, days);
  out.sunTheta = ts;
  out.moonTheta = tm;
  pathDir(ts, out.sun);
  pathDir(tm, out.moon);
  const es = pathElevation(ts), em = pathElevation(tm);
  out.sunElev = es;
  out.moonElev = em;
  out.elongation = moonElongation(days);
  out.moonIllum = moonIllumination(out.elongation);
  const esDeg = es / DEG;

  // Key light: the sun until it has left every altitude, then the moon, fading in while the land is dark.
  const keyIsSun = esDeg > -7.5;
  out.keyIsSun = keyIsSun;
  const src = keyIsSun ? out.sun : out.moon;
  out.key[0] = src[0]; out.key[1] = src[1]; out.key[2] = src[2];
  out.keyTheta = keyIsSun ? ts : tm;
  // The moon's light grows fast with phase (a half moon gives ~a quarter of the full moon's light).
  const moonW = keyIsSun ? 0 : smooth(-8, -11, esDeg) * moonLightShare(out.moonIllum);
  const E = keyIsSun ? SUN_E : MOON_E;
  const w = keyIsSun ? 1 : moonW;
  const elev = keyIsSun ? es : em;
  for (let k = 0; k < KEY_ALTS; k++) {
    keyTrans(k, elev + horizonDip(keyAltitude(k)), tmp);
    for (let c = 0; c < 3; c++) out.keyCols[k * 4 + c] = E[c] * tmp[c] * w;
    out.keyCols[k * 4 + 3] = 0;
  }

  // The discs, through the air between the camera and them.
  const kc = camKey(camY);
  keyTransAt(kc, es + horizonDip(camY), tmp);
  for (let c = 0; c < 3; c++) out.sunDisc[c] = SUN_E[c] * tmp[c] * SUN_DISC_GAIN;
  out.night = 1 - smooth(-11, -3, esDeg);
  keyTransAt(kc, em + horizonDip(camY), tmp);
  const moonShow = MOON_DISC_DAY + (MOON_DISC_NIGHT - MOON_DISC_DAY) * out.night;
  for (let c = 0; c < 3; c++) out.moonDisc[c] = (tmp[c] / T45[c]) * moonShow;

  out.saturation = curve(SATURATION, esDeg);
  out.stars = 1 - smooth(-11.5, -4.5, esDeg);
  out.haze = curve(HAZE, esDeg);
  // Morning mist: settles in the low land before dawn, burns off by mid-morning.
  out.mist = smooth(2.5, 5.2, hour) * (1 - smooth(7.2, 9.6, hour));
  // Clouds: build 12:00–17:00, dissolve 19:00–21:00, clear through the night and morning.
  const build = smooth(11.5, 16.5, hour) * (1 - smooth(18.8, 21, hour));
  out.cloudShift = 0.1 * out.night * (1 - build) + 0.045 * (1 - out.night) * (1 - build) - 0.03 * build;
  out.cloudLight = 1 - 0.55 * out.night;
  const moonGlare = out.moonIllum * smooth(-0.05, 0.3, Math.sin(em));
  out.milkyWay = out.stars * (1 - 0.8 * moonGlare);
  out.twilight = smooth(-13, -9, esDeg) * (1 - smooth(2, 6, esDeg));
  starBasis(ts, STAR_OFFSET, out.starRows);
  return out;
}

/** Fractional key index of an altitude (the inverse of keyAltitude). */
function camKey(y: number): number {
  return Math.min(Math.sqrt(Math.max(y, 0) / KEY_TOP), 1) * (KEY_ALTS - 1);
}

const ta: V3 = [0, 0, 0];
const tb: V3 = [0, 0, 0];
function keyTransAt(kf: number, e: number, out: V3): V3 {
  const k = Math.min(Math.floor(kf), KEY_ALTS - 2), f = kf - k;
  keyTrans(k, e, ta);
  keyTrans(k + 1, e, tb);
  for (let c = 0; c < 3; c++) out[c] = ta[c] * (1 - f) + tb[c] * f;
  return out;
}
