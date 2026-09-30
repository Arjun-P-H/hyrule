/**
 * Sun, moon and stars over the world (pure; lightRig.ts turns them into the
 * renderer's light, sky.wgsl.ts draws them).
 * ---------------------------------------------------------------------------
 * The world stands at 30° latitude on an equinox that never ends: the sun
 * rises due east at 06:00, culminates 60° high toward −z at noon and sets due
 * west at 18:00, every day. Its daily path is a great circle, so a direction
 * on it is one angle, θ: 0 rising, π/2 at the top, π setting, below the
 * horizon from π to 2π. The moon runs on the same circle, a phase angle
 * behind the sun, so one horizon per terrain texel (two angles along this
 * circle, where light rises over the land and where it sets) shades the land
 * for either of them at any hour. The stars turn with the sun about the
 * celestial pole, 30° above the northern horizon.
 *
 * Axes: +x east, +y up, +z north (the camera at yaw 0 looks north).
 */

export type V3 = [number, number, number];

const DEG = Math.PI / 180;
const TAU = 2 * Math.PI;

export const SKY = {
  /** Tilt of the sun's path from the zenith (the latitude); noon elevation 90° − latitude. */
  latitude: 30 * DEG,
  /** Game days from new moon to new moon. */
  moonPeriod: 8,
  /** Age of the moon (days since new) at day 0, 00:00: a waxing gibbous stands in the east at the first sunset. */
  moonAge0: 2.36,
  /**
   * Radius (m) of the planet the land's horizon curves over. Far smaller than
   * the Earth's, so a peak or an island 1 km up sees the sun ~4.7° below the
   * valley's horizon and keeps its light ~20 game minutes after the valley loses it.
   */
  planetRadius: 300_000,
} as const;

const SIN_LAT = Math.sin(SKY.latitude);
const COS_LAT = Math.cos(SKY.latitude);

/** The path's axis 90° from east (+x) along it, up and leaning toward −z: dir(θ) = cos θ·(1, 0, 0) + sin θ·N. */
export const PATH_N: V3 = [0, COS_LAT, -SIN_LAT];
/** The celestial pole the sky turns about (normal of the path's plane). */
export const POLE: V3 = [0, SIN_LAT, COS_LAT];

/** Direction toward a body at path angle θ. */
export function pathDir(theta: number, out: V3 = [0, 0, 0]): V3 {
  const c = Math.cos(theta), s = Math.sin(theta);
  out[0] = c;
  out[1] = s * COS_LAT;
  out[2] = -s * SIN_LAT;
  return out;
}

/** Elevation (rad) above the horizontal of the direction at path angle θ. */
export function pathElevation(theta: number): number {
  return Math.asin(Math.sin(theta) * COS_LAT);
}

/**
 * θ in (−π/2, 3π/2]: the range the terrain horizons use (rising side
 * (−π/2, π/2), setting side (π/2, 3π/2)), so a body just below the eastern
 * horizon is a small negative angle.
 */
export function wrapTheta(theta: number): number {
  let t = theta % TAU;
  if (t < 0) t += TAU;
  return t > 1.5 * Math.PI ? t - TAU : t;
}

/** The sun's path angle at an hour of the day (0–24). */
export function sunTheta(hour: number): number {
  return wrapTheta(((hour - 6) / 24) * TAU);
}

/** The moon's angle from the sun along the path (0 new, π full), `days` after day 0, 00:00. */
export function moonElongation(days: number): number {
  const age = (days + SKY.moonAge0) / SKY.moonPeriod;
  return (age - Math.floor(age)) * TAU;
}

/** Share of the moon's disc that is lit, from its elongation. */
export function moonIllumination(elongation: number): number {
  return 0.5 * (1 - Math.cos(elongation));
}

/** The moon's path angle: its elongation behind the sun. */
export function moonTheta(hour: number, days: number): number {
  return wrapTheta(sunTheta(hour) - moonElongation(days));
}

/** How far (rad) the land's horizon dips below the horizontal at altitude y (m). */
export function horizonDip(y: number): number {
  return Math.sqrt((2 * Math.max(y, 0)) / SKY.planetRadius);
}

/**
 * World → celestial rotation at sun angle θ_s (rows: the sky's x, y, z axes
 * in world space). Stars are fixed in the celestial frame; `offset` turns the
 * star field so the Milky Way arches overhead in the evening.
 */
export function starBasis(thetaSun: number, offset: number, out: [V3, V3, V3] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]): [V3, V3, V3] {
  pathDir(thetaSun + offset, out[0]);
  pathDir(thetaSun + offset + Math.PI / 2, out[1]);
  out[2][0] = POLE[0]; out[2][1] = POLE[1]; out[2][2] = POLE[2];
  return out;
}
