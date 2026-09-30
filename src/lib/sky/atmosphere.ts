/**
 * The atmosphere the sky is drawn from (sky.wgsl.ts computes its LUTs on the
 * GPU; this is the CPU twin of the part the light rig needs: how much of the
 * sun's or moon's light gets through the air to a point). An Earth-like
 * atmosphere after Hillaire (2020, "A Scalable and Production Ready Sky and
 * Atmosphere Rendering Technique"): Rayleigh scattering (the blue sky and the
 * reddened low sun), Mie scattering by haze (the glow round the sun), and
 * ozone, which absorbs orange light and keeps the twilight sky blue. Lengths
 * in km; extinction in 1/km.
 */

import type { V3 } from './celestial';

export const ATMOSPHERE = {
  bottom: 6360,
  top: 6460,
  rayleigh: [5.802e-3, 13.558e-3, 33.1e-3] as V3,
  rayleighHeight: 8,
  mieScatter: 3.996e-3,
  mieExtinct: 4.4e-3,
  mieHeight: 1.2,
  mieG: 0.8,
  // Ozone's Chappuis band, with red absorbed at ~0.64 of green as measured (Hillaire's 0.35 turns the blue hour violet).
  ozone: [1.2e-3, 1.881e-3, 0.085e-3] as V3,
  ozoneCentre: 25,
  ozoneHalfWidth: 15,
  groundAlbedo: 0.3,
} as const;

const A = ATMOSPHERE;

/** Extinction (1/km) of the air at altitude h (km). */
function extinctionAt(h: number, out: V3 = [0, 0, 0]): V3 {
  const r = Math.exp(-h / A.rayleighHeight);
  const m = Math.exp(-h / A.mieHeight) * A.mieExtinct;
  const o = Math.max(0, 1 - Math.abs(h - A.ozoneCentre) / A.ozoneHalfWidth);
  for (let c = 0; c < 3; c++) out[c] = A.rayleigh[c] * r + m + A.ozone[c] * o;
  return out;
}

/** Distance from radius r along direction cosine mu to the sphere of radius R (the far root; −1 if none). */
function sphereExit(r: number, mu: number, R: number): number {
  const d = r * r * (mu * mu - 1) + R * R;
  return d < 0 ? -1 : -r * mu + Math.sqrt(d);
}

/** Whether the ray from radius r along mu hits the ground. */
function hitsGround(r: number, mu: number): boolean {
  return mu < 0 && r * r * (mu * mu - 1) + A.bottom * A.bottom >= 0;
}

const ext: V3 = [0, 0, 0];

/**
 * Transmittance from altitude h (km) toward direction cosine mu (from the
 * zenith) out of the atmosphere; zero if the ray meets the ground. Samples
 * crowd near the start, where the air is densest.
 */
export function transmittance(h: number, mu: number, out: V3 = [0, 0, 0], steps = 64): V3 {
  const r = A.bottom + Math.max(h, 0);
  if (hitsGround(r, mu)) { out[0] = out[1] = out[2] = 0; return out; }
  const L = sphereExit(r, mu, A.top);
  let d0 = 0, d1 = 0, d2 = 0, prev = 0;
  for (let i = 1; i <= steps; i++) {
    const u = i / steps;
    const s = L * u * u;
    const sm = 0.5 * (s + prev);
    const hh = Math.sqrt(r * r + sm * sm + 2 * r * mu * sm) - A.bottom;
    extinctionAt(hh, ext);
    const ds = s - prev;
    d0 += ext[0] * ds; d1 += ext[1] * ds; d2 += ext[2] * ds;
    prev = s;
  }
  out[0] = Math.exp(-d0); out[1] = Math.exp(-d1); out[2] = Math.exp(-d2);
  return out;
}
