/**
 * Water on a sky island: where it gathers, where it runs, where it leaves.
 * ---------------------------------------------------------------------------
 * The realm's water cycle: the cloud deck below wets the islands (most of all
 * the low ones that sit in its morning mist), so a wet enough island keeps a
 * lake in its lowest hollow, fed by that dew and by springs. The lake drains
 * by one stream that runs only downhill to a notch in the rim and pours off
 * as a waterfall. A fall hundreds of metres high does not reach the ground
 * as a column: it frays into spray and then mist on the way down (as the
 * world's tallest falls do), and the mist drifts back into the clouds it
 * came from. So every fall has a source, and none lands on a dry mountain.
 * Frozen lakes (the frost biome) hold their water as ice: no stream.
 *
 * planWater(isl, …) fills in isl.lake, isl.stream and isl.fall:
 *   lake    the lowest of a dozen candidate hollows well inside the rim; its
 *           level a little below the lowest point of its shore, so it never
 *           spills (topAt adds a low levee for what lies between the samples);
 *   lip     a bearing on the rim scored by how little high ground the stream
 *           must cut through, how far the land lies below for the fall, and
 *           kept clear of the arches' footings and of any island below;
 *   stream  16 points from the lake's outlet to the lip, meandering; its bed
 *           starts under the lake's level, never rises, and lies at least
 *           0.8 m under the land along every segment, so the water always
 *           runs in its channel;
 *   fall    leaves the lip over the rim; it has all turned to mist by
 *           `reach` metres down, at most half way to the land below.
 */

import { SKY_REALM, VALLEY } from '../constants';
import { SKY_BIOME_INFO } from './biomes';
import { baseTop, lakeDist, outline, outlineMax, STREAM_DEPTH, type Lake, type SkyIsland } from './island';

const TAU = Math.PI * 2;
const STREAM_POINTS = 16;
/** Where across a stream (× its half-width) its bed is kept under the land: its axis (to compare courses), its whole corridor. */
const AXIS = [0] as const;
const CORRIDOR = [-1.6, -0.8, 0, 0.8, 1.6] as const;

/** Angle between two bearings (rad, 0..π). */
const angleBetween = (a: number, b: number) => Math.abs(((a - b + 3 * Math.PI) % TAU) - Math.PI);

/**
 * Plan the island's lake, stream and fall. `rnd` draws its random numbers
 * (deterministic per island); `avoid` are bearings (from the island's centre)
 * where arches stand; `ground` is the land under a point (the terrain, or
 * the water where that is higher); `others` the islands round it.
 */
export function planWater(
  isl: SkyIsland, rnd: () => number, avoid: readonly number[], ground: (x: number, z: number) => number, others: readonly SkyIsland[],
): void {
  const info = SKY_BIOME_INFO[isl.biome];
  if (isl.kind === 'debris' || isl.R < 60) return;
  if (rnd() >= info.lake * (isl.R < 130 ? 0.45 : 1)) return;
  const lake = placeLake(isl, rnd);
  if (!lake) return;
  isl.lake = lake;
  if (lake.frozen) return;

  // The lip: the best of 36 bearings round the rim.
  type Plan = { score: number; pts: Array<[number, number, number, number]>; th: number };
  let best: Plan | null = null;
  const phase = rnd() * TAU;
  const hw0 = Math.min(6, Math.max(1.6, 0.012 * isl.R));
  const meanderK = rnd() < 0.5 ? 1 : 2, meanderPh = rnd() * TAU, meanderA = 0.04 + 0.05 * rnd();
  for (let k = 0; k < 36; k++) {
    const th = phase + (k / 36) * TAU;
    if (avoid.some((b) => angleBetween(b, th) < 0.45)) continue;
    const pr = isl.R * outline(isl, th);
    const px = Math.cos(th) * pr, pz = Math.sin(th) * pr;
    // The fall's column must not land on an island below.
    const lipX = isl.cx + px, lipZ = isl.cz + pz;
    if (others.some((o) => o !== isl && Math.hypot(lipX - o.cx, lipZ - o.cz) < o.R * outlineMax(o) + 120)) continue;
    const pts = streamPath(isl, lake, px, pz, hw0, meanderK, meanderPh, meanderA, AXIS);
    if (!pts) continue;
    let gorge = 0, len = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      len += Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
      gorge = Math.max(gorge, baseTop(isl, isl.cx + pts[i][0], isl.cz + pts[i][1]) - pts[i][2]);
    }
    // No course through high ground: the channel would be a ravine (and its carve would run out).
    if (gorge > 3 * hw0 + 4) continue;
    const dx = Math.cos(th), dz = Math.sin(th);
    const below = isl.y - ground(lipX + dx * 25, lipZ + dz * 25);
    const score = -2 * gorge - 0.01 * len + 0.002 * below + 0.5 * rnd();
    if (!best || score > best.score) best = { score, pts, th };
  }
  if (!best) return;
  // The chosen course again, its bed now kept under the land across the whole channel, not just its axis.
  const P0 = best.pts[STREAM_POINTS - 1];
  const pts = streamPath(isl, lake, P0[0], P0[1], hw0, meanderK, meanderPh, meanderA, CORRIDOR)!;
  best.pts = pts;
  isl.stream = { pts };

  // The fall: over the lip, leaving along the stream and away from the island.
  const P = best.pts[STREAM_POINTS - 1], Q = best.pts[STREAM_POINTS - 2];
  let fx = P[0] - Q[0], fz = P[1] - Q[1];
  const fl = Math.hypot(fx, fz) || 1;
  fx = fx / fl + Math.cos(best.th);
  fz = fz / fl + Math.sin(best.th);
  const dl = Math.hypot(fx, fz) || 1;
  fx /= dl; fz /= dl;
  const x = isl.cx + P[0], z = isl.cz + P[1], y = P[2] + STREAM_DEPTH;
  const land = Math.max(ground(x + fx * 25, z + fz * 25), VALLEY.wLevel);
  isl.fall = {
    x, y, z, dirX: fx, dirZ: fz,
    width: 1.5 * P[3], speed: 2 + 0.25 * P[3],
    reach: Math.max(90, Math.min(0.5 * (y - land), 220 + 40 * 1.5 * P[3])),
    seed: (isl.seed * 31 + 7) | 0,
  };
}

/** The lowest of a dozen candidate hollows well inside the rim, its level just under its shore. */
function placeLake(isl: SkyIsland, rnd: () => number): Lake | null {
  const info = SKY_BIOME_INFO[isl.biome];
  const inner = 0.7 * isl.R * (1 - 1.12 * isl.amp); // the rim never comes closer than this
  let best: { score: number; lake: Lake } | null = null;
  for (let k = 0; k < 12; k++) {
    const rc = (isl.R < 300 ? 0.05 + 0.25 * rnd() : 0.08 + 0.38 * rnd());
    const th = rnd() * TAU;
    const r = Math.min(230, Math.max(14, isl.R * (0.1 + 0.1 * rnd())));
    const aspect = 0.65 + 0.35 * rnd();
    const ang = rnd() * Math.PI;
    const cr = rc * isl.R * outline(isl, th);
    const x = Math.cos(th) * cr, z = Math.sin(th) * cr;
    if (cr + 1.25 * 1.2 * r > inner) continue;
    // Prefer hollows: the middle low against a ring round it.
    let ring = 0;
    for (let j = 0; j < 8; j++) {
      const a = (j / 8) * TAU;
      ring += baseTop(isl, isl.cx + x + Math.cos(a) * 1.8 * r, isl.cz + z + Math.sin(a) * 1.8 * r);
    }
    const score = ring / 8 - baseTop(isl, isl.cx + x, isl.cz + z) + 0.02 * isl.hill * rnd();
    if (!best || score > best.score) {
      best = {
        score,
        lake: {
          x, z, r, aspect, cos: Math.cos(ang), sin: Math.sin(ang), b0: rnd() * TAU, b1: rnd() * TAU,
          level: 0, depth: Math.min(9, Math.max(1.2, 0.07 * r)), frozen: info.frozen,
        },
      };
    }
  }
  if (!best) return null;
  const L = best.lake;
  // The level: just under the lowest point of the shore (two rings of 48), so it never spills.
  let low = Infinity;
  for (const d of [1.0, 1.15, 1.3]) {
    for (let j = 0; j < 48; j++) {
      const p = shorePoint(L, (j / 48) * TAU, d);
      low = Math.min(low, baseTop(isl, isl.cx + p[0], isl.cz + p[1]));
    }
  }
  L.level = low - 0.35;
  return L;
}

/** Island-local point at normalised distance d from a lake's middle, at angle a in its own (stretched) frame. */
function shorePoint(L: Lake, a: number, d: number): [number, number] {
  const w = 1 + 0.12 * Math.sin(3 * a + L.b0) + 0.07 * Math.sin(5 * a + L.b1);
  const u = Math.cos(a) * L.r * w * d, v = Math.sin(a) * L.r * w * d * L.aspect;
  return [L.x + u * L.cos - v * L.sin, L.z + u * L.sin + v * L.cos];
}

/**
 * The stream from the lake's outlet (on its shore, toward the lip) to the lip
 * (px, pz) on the rim: 16 meandering points with their bed heights and
 * half-widths. The bed starts 0.6 m under the lake's level and never rises;
 * over every segment it stays 0.8 m under the land (sampled 8 times along it,
 * at each of `across` × the half-width to the side), and it falls at least
 * 0.4 % so the water flows; at the lip it notches 1.2 m into the rim. Null
 * when the outlet would not lie on this side of the lake.
 */
function streamPath(
  isl: SkyIsland, L: Lake, px: number, pz: number, hw0: number, k: number, ph: number, amp: number, across: readonly number[],
): Array<[number, number, number, number]> | null {
  // The outlet: where the line from the lake's middle to the lip crosses its shore.
  let ox = L.x, oz = L.z;
  const dx = px - L.x, dz = pz - L.z;
  const dlen = Math.hypot(dx, dz);
  for (let f = 0; f <= 1; f += 1 / 256) {
    if (lakeDist(L, L.x + dx * f, L.z + dz * f) >= 1.0) { ox = L.x + dx * f; oz = L.z + dz * f; break; }
  }
  const len = Math.hypot(px - ox, pz - oz);
  if (len < 0.15 * dlen || len < 20) return null;
  const nx = -(pz - oz) / len, nz = (px - ox) / len;
  const xy: Array<[number, number]> = [];
  for (let i = 0; i < STREAM_POINTS; i++) {
    const t = i / (STREAM_POINTS - 1);
    const off = amp * len * Math.sin(Math.PI * t) * Math.sin(TAU * k * t + ph);
    xy.push([ox + (px - ox) * t + nx * off, oz + (pz - oz) * t + nz * off]);
  }
  const land = (x: number, z: number) => baseTop(isl, isl.cx + x, isl.cz + z);
  const hwAt = (i: number) => hw0 * (1 + 0.3 * (i / (STREAM_POINTS - 1)));
  const segMin = (i: number) => {
    const ax = xy[i][0], az = xy[i][1], dx = xy[i + 1][0] - ax, dz = xy[i + 1][1] - az;
    const l = Math.hypot(dx, dz) || 1;
    const sx = -dz / l, sz = dx / l;
    let m = Infinity;
    for (let j = 0; j <= 8; j++) {
      const f = j / 8;
      const hw = hwAt(i) + (hwAt(i + 1) - hwAt(i)) * f;
      for (const c of across) {
        // (the corridor stays on the island: past the rim there is no land to hold the water)
        const x = ax + dx * f + sx * c * hw, z = az + dz * f + sz * c * hw;
        if (c !== 0 && Math.hypot(x, z) > isl.R * outline(isl, Math.atan2(z, x))) continue;
        m = Math.min(m, land(x, z));
      }
    }
    return m - 0.8;
  };
  const pts: Array<[number, number, number, number]> = [];
  let bed = L.level - 0.6;
  const rim = isl.y - SKY_REALM.rimDrop - 1.2; // the rim's height less the notch
  for (let i = 0; i < STREAM_POINTS; i++) {
    if (i > 0) bed -= 0.004 * Math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]);
    if (i < STREAM_POINTS - 1) bed = Math.min(bed, segMin(i));
    else bed = Math.min(bed, rim);
    pts.push([xy[i][0], xy[i][1], bed, hwAt(i)]);
  }
  return pts;
}
