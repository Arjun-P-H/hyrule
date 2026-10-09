/**
 * The Crown's grammars (SKY_CIVILIZATION.md §7.1): on its main isle the Gyre
 * Cathedral on the summit, its plaza and the Moot's rotunda beside it, palace
 * towers round it; avenues from the plaza to every bridgehead, roosts and
 * hall-houses lining them and filling the blocks between, doors to the
 * street; a wind wall with turrets along the windward rim; a Silent Spire on
 * a rim point. Its ward isles each have their task — market, archive, garden,
 * roosts, cisterns — and its perch isles a watchtower each.
 */

import { nearWater, outline } from '../island';
import {
  archiveSpire, cathedral, greenhouse, marketHall, plaza, roost, rotunda, silentSpire, watchtower, windWall,
} from './buildings';
import { bridgeheads, fillRoosts, lineStreet, summit, WIND, type Grammar } from './planner';

/** The Crown's main isle (§7.1). */
export const crownIsle: Grammar = function* (a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const heart = (x: number, z: number) => Math.max(0, 1 - Math.hypot(x - isl.cx, z - isl.cz) / isl.R);
  const top = summit(pl, 62, isl.R * 0.3, 30);
  if (!top) return;
  // The cathedral faces the bridgehead its avenue runs to (the nearest), its plaza before it.
  const main = feet.length ? feet.reduce((p, q) => (Math.hypot(p.x - top.x, p.z - top.z) < Math.hypot(q.x - top.x, q.z - top.z) ? p : q)) : null;
  const face = main ? Math.atan2(main.x - top.x, main.z - top.z) : pl.rnd() * Math.PI * 2;
  const L = 60 + 20 * pl.rnd(), W = 20 + 6 * pl.rnd(), H = 26 + 8 * pl.rnd();
  out.push(...cathedral(pl.site, top.x, top.z, { face, length: L, width: W, height: H, spires: ([1, 2, 3, 5] as const)[Math.floor(pl.rnd() * 4)] }));
  yield;
  const cr = Math.max(L, W) * 0.62 + 6;
  pl.place(top.x, top.z, cr, cr - 4);
  const pR = 17 + 4 * pl.rnd();
  const pd = cr + pR + 3;
  const px = top.x + Math.sin(face) * pd, pz = top.z + Math.cos(face) * pd;
  let hubX = top.x, hubZ = top.z;
  if (pl.fits(px, pz, pR, 16)) {
    out.push(plaza(pl.site, px, pz, pR, 6));
    pl.take(px, pz, pR + 1, true);
    pl.paveDisc(px, pz, pR + 0.5);
    hubX = px; hubZ = pz;
    // the Moot's rotunda beside the plaza
    const ma = face + (pl.rnd() < 0.5 ? 1 : -1) * Math.PI / 2;
    const mx = px + Math.sin(ma) * (pR + 16), mz = pz + Math.cos(ma) * (pR + 16);
    if (pl.fits(mx, mz, 13, 10)) {
      out.push(rotunda(pl.site, mx, mz, 10, 9, { columns: 16, hall: true }));
      pl.place(mx, mz, 14, 13.5);
    }
  }
  // Palace towers round the cathedral.
  for (let k = 0; k < 6; k++) {
    const ang = face + Math.PI / 4 + (k / 6) * Math.PI * 2;
    const d = cr + 12 + 10 * pl.rnd();
    const x = top.x + Math.sin(ang) * d, z = top.z + Math.cos(ang) * d;
    const r = 6 + 2.5 * pl.rnd();
    if (!pl.fits(x, z, r + 2, 10)) continue;
    out.push(roost(pl.site, x, z, { r, floors: 7 + Math.floor(pl.rnd() * 4), face: Math.atan2(top.x - x, top.z - z), roof: pl.rnd() < 0.5 ? 'spire' : 'dome', pennant: true, turrets: 4, tier: true }));
    pl.place(x, z, r + 4, r + 1.5);
  }
  yield;
  // Avenues to the bridgeheads, roosts along them; a ring street; the rest filled.
  for (const f of feet) pl.street(hubX, hubZ, f.x, f.z, 5);
  for (const f of feet) yield* lineStreet(pl, out, hubX, hubZ, f.x, f.z, 5, 21, heart);
  if (isl.R > 300) {
    const rr = isl.R * 0.55, n = 40;
    let prev: [number, number] | null = null;
    for (let k = 0; k <= n; k++) {
      const th = (k / n) * Math.PI * 2;
      const p: [number, number] = [isl.cx + Math.cos(th) * rr * outline(isl, th), isl.cz + Math.sin(th) * rr * outline(isl, th)];
      if (prev) pl.street(prev[0], prev[1], p[0], p[1], 3.5);
      prev = p;
    }
    prev = null;
    for (let k = 0; k <= n; k++) {
      const th = (k / n) * Math.PI * 2;
      const p: [number, number] = [isl.cx + Math.cos(th) * rr * outline(isl, th), isl.cz + Math.sin(th) * rr * outline(isl, th)];
      if (prev) yield* lineStreet(pl, out, prev[0], prev[1], p[0], p[1], 3.5, 20, heart);
      prev = p;
    }
  }
  yield;
  // The wind wall along the windward rim (gaps where the avenues leave), a turret every run.
  const wallPts: Array<[number, number]> = [];
  const span = 0.9 + 0.4 * pl.rnd();
  for (let k = 0; k <= 14; k++) {
    const th = WIND - span + (2 * span * k) / 14;
    const p = pl.rim(th, 12);
    if (feet.some((f) => Math.hypot(f.x - p[0], f.z - p[1]) < f.w + 30) || nearWater(isl, p[0], p[1], 6)) {
      if (wallPts.length > 1) out.push(...windWall(pl.site, wallPts, 7 + 3 * pl.rnd()));
      wallPts.length = 0;
      continue;
    }
    wallPts.push(p);
    pl.place(p[0], p[1], 6, 5);
  }
  if (wallPts.length > 1) out.push(...windWall(pl.site, wallPts, 7 + 3 * pl.rnd()));
  // The Silent Spire, on the lee rim.
  for (let k = 0; k < 8; k++) {
    const [x, z] = pl.rim(WIND + Math.PI + (pl.rnd() - 0.5) * 1.2, 16);
    if (pl.fits(x, z, 5, 6, 1)) { out.push(silentSpire(pl.site, x, z)); pl.place(x, z, 6, 4); break; }
  }
  yield* fillRoosts(pl, out, 34, 0.6, heart);
};

/** A ward isle of the Crown (§7.1): its task by index round the Crown. */
export const WARD_TASKS = ['market', 'archive', 'garden', 'roosts', 'cistern', 'market', 'garden'] as const;
export const wardIsle = (task: (typeof WARD_TASKS)[number]): Grammar => function* (a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const heart = (x: number, z: number) => 0.5 * Math.max(0, 1 - Math.hypot(x - isl.cx, z - isl.cz) / isl.R);
  const mid = summit(pl, 22, isl.R * 0.3, 12) ?? { x: isl.cx, z: isl.cz };
  const face = feet.length ? Math.atan2(feet[0].x - mid.x, feet[0].z - mid.z) : 0;
  if (task === 'market') {
    out.push(plaza(pl.site, mid.x, mid.z, 15, 6));
    pl.take(mid.x, mid.z, 16, true);
    pl.paveDisc(mid.x, mid.z, 15.5);
    for (let k = 0; k < 5; k++) {
      const ang = face + (k / 5) * Math.PI * 2 + 0.3;
      const x = mid.x + Math.sin(ang) * 32, z = mid.z + Math.cos(ang) * 32;
      if (!pl.fits(x, z, 14, 8)) continue;
      out.push(marketHall(pl.site, x, z, ang, 22 + 6 * pl.rnd(), 10 + 2 * pl.rnd()));
      pl.place(x, z, 15, 13);
    }
  } else if (task === 'archive') {
    out.push(...archiveSpire(pl.site, mid.x, mid.z, face));
    pl.place(mid.x, mid.z, 30, 12);
  } else if (task === 'garden') {
    out.push(rotunda(pl.site, mid.x, mid.z, 5, 5, { columns: 8, onion: pl.rnd() < 0.4 }));
    pl.place(mid.x, mid.z, 9, 8);
    const gx = mid.x + Math.sin(face + 2) * 40, gz = mid.z + Math.cos(face + 2) * 40;
    if (pl.fits(gx, gz, 11, 6)) { out.push(greenhouse(pl.site, gx, gz, 8 + 3 * pl.rnd())); pl.place(gx, gz, 12, 11); }
  } else if (task === 'cistern') {
    out.push(plaza(pl.site, mid.x, mid.z, 12, 4));
    pl.take(mid.x, mid.z, 13, true);
    pl.paveDisc(mid.x, mid.z, 12.5);
    out.push(marketHall(pl.site, mid.x + Math.sin(face) * 28, mid.z + Math.cos(face) * 28, face, 18, 12));
    pl.place(mid.x + Math.sin(face) * 28, mid.z + Math.cos(face) * 28, 13, 12);
  } else {
    out.push(plaza(pl.site, mid.x, mid.z, 12, 4));
    pl.take(mid.x, mid.z, 13, true);
    pl.paveDisc(mid.x, mid.z, 12.5);
  }
  for (const f of feet) pl.street(mid.x, mid.z, f.x, f.z, 3.5);
  for (const f of feet) yield* lineStreet(pl, out, mid.x, mid.z, f.x, f.z, 3.5, 20, heart);
  yield* fillRoosts(pl, out, task === 'garden' ? 44 : 28, task === 'roosts' ? 0.95 : 0.7, heart);
};

/** A perch isle: one watchtower. */
export const perchIsle: Grammar = function* (_a, _i, pl, _feet, out) {
  const isl = pl.isl;
  for (let k = 0; k < 12; k++) {
    const x = isl.cx + (pl.rnd() - 0.5) * isl.R * 0.5, z = isl.cz + (pl.rnd() - 0.5) * isl.R * 0.5;
    if (!pl.fits(x, z, 4.5, 6, 0.95)) continue;
    out.push(watchtower(pl.site, x, z, 3 + pl.rnd(), 18 + 10 * pl.rnd(), pl.rnd() * 6.28));
    pl.foot(x, z, 5);
    return;
  }
};

