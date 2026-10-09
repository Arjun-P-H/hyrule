/**
 * The grammars of every Eyrie class but the Crown (crown.ts), island role by
 * island role (SKY_CIVILIZATION.md §7.2–7.11):
 *   Mirrorwell   the bowl's lagoon with the Rotunda of the Accord on its islet,
 *                a causeway of arches out to it, a lantern-lit promenade round
 *                the shore with its pavilions, homes up the bowl's terraces;
 *                its well isles small temples and cisterns;
 *   Ringhold     the Great Halo standing on its isle, buttressed, runestones
 *                floating round it, the Stonesingers' cloister and bell tower;
 *                a wayshrine on every pilgrim isle;
 *   Lanternpeak  the Vaelspire on the summit sending up its Thread, the
 *                lamp-wardens' tower; observatories on their isles; crystals
 *                growing on the small isles floating above;
 *   Petalfall    wind-houses (tiered pagodas) and tea pavilions by the pools,
 *                gate-arches at the bridgeheads, the orchards round them;
 *   Galeharbour  docks out over the anchorage with cranes, cargo and moored
 *                skyships; market stalls round a plaza; warehouses; a
 *                lighthouse where the ships come in;
 *   Mistwells    cloud-combs along the rims, wind-wheel towers, cisterns;
 *   Cinderforge  forges with smoking chimneys, the old forge-lords' citadel,
 *                braziers; skyship yards on the leeward isles;
 *   Rimewatch    the keep, wind walls round the whole rim, battlemented
 *                towers, braziers burning on the watch isles; the Gyre's Eye
 *                turning overhead;
 *   Elder Bough  the colossal tree, its roots hanging under the isle, nest
 *                platforms in its limbs, lanterns; nests in the groves;
 *   Hushfall     everything the Long Gale broke: an unroofed cathedral,
 *                roosts cut short, fallen columns, bridges ending in the air —
 *                dark, overgrown, still guarded by a few Wardens.
 */

import { outline } from '../island';
import { makeStructure, PFLAG, type PieceInst, type Structure } from '../structures';
import { pieceId } from '../kit/pieces';
import { bridgeStructures } from './bridges';
import {
  brokenBridge, cathedral, cistern, comb, elderBough, forge, haloMonument, hallHouse, harbourDock, keep, marketHall,
  observatory, pagoda, plaza, roost, rotunda, ruinedHall, ruinedRoost, shrine, stalls, teaPavilion, torii, vaelspire, watchtower,
  windwheel, windWall,
} from './buildings';
import { bridgeheads, fillRoosts, lineStreet, summit, WIND, type Footing, type Grammar, type Planner } from './planner';
import type { Arch } from '../arches';

const TAU = Math.PI * 2;

/** The bearing (yaw-style, (sin, cos)) from a to b. */
const toward = (ax: number, az: number, bx: number, bz: number) => Math.atan2(bx - ax, bz - az);

/** A few homes of the class round the heart of a small isle. */
function* homes(pl: Planner, out: Structure[], step: number, keep: number): Generator<void, void, void> {
  const isl = pl.isl;
  yield* fillRoosts(pl, out, step, keep, (x, z) => 0.4 * Math.max(0, 1 - Math.hypot(x - isl.cx, z - isl.cz) / isl.R));
}

/** Streets from (x, z) to each bridgehead (paved, homes along them). */
function* streets(pl: Planner, out: Structure[], x: number, z: number, feet: Footing[], w: number): Generator<void, void, void> {
  const isl = pl.isl;
  for (const f of feet) pl.street(x, z, f.x, f.z, w);
  for (const f of feet) yield* lineStreet(pl, out, x, z, f.x, f.z, w, 20, (px, pz) => 0.5 * Math.max(0, 1 - Math.hypot(px - isl.cx, pz - isl.cz) / isl.R));
}

// ---- Mirrorwell (§7.2) ---------------------------------------------------------------------------------

const bowlIsle: Grammar = function* (a, _i, pl, feet, out) {
  const isl = pl.isl, L = isl.lake;
  bridgeheads(pl, feet);
  if (!L) { yield* homes(pl, out, 30, 0.5); return; }
  const cx = isl.cx + L.x, cz = isl.cz + L.z, level = L.level;
  // The islet: a round plaza standing in the lagoon, the Rotunda of the Accord on it.
  const iR = Math.min(16, L.r * 0.16);
  const top = level + 1.6;
  const s = pl.site;
  out.push(islet(pl, cx, cz, iR, top, level - L.depth - 1.5));
  // The causeway: arches from the shore toward the nearest bridgehead out to the islet.
  const f0 = feet[0];
  const bearing = f0 ? Math.atan2(f0.z - cz, f0.x - cx) : pl.rnd() * TAU;
  const ux = Math.cos(bearing), uz = Math.sin(bearing);
  const shoreD = L.r * 1.12, sx = cx + ux * shoreD, sz = cz + uz * shoreD;
  const shoreY = s.ground(sx, sz);
  const A: Arch = {
    kind: 'bridge', from: 0, to: 0, ax: sx, az: sz, ay: Math.max(shoreY, top) + 0.1, bx: cx + ux * (iR - 0.5), bz: cz + uz * (iR - 0.5), by: top,
    dirX: -ux, dirZ: -uz, len: shoreD - iR + 0.5, bow: 0, rise: 0, wMid: 7, wEnd: 7, thMid: 0, thEnd: 0, seed: 1,
  };
  out.push(...bridgeStructures(A, { palette: a.palette, pointed: false, span: [12, 16], gate: false, hang: [0.4, 0.6] }, pl.rnd));
  pl.take(sx, sz, 6, true); pl.paveDisc(sx, sz, 5);
  yield;
  // The promenade round the shore: paved, lanterns along it, pavilions at intervals.
  const pr = L.r * 1.3, n = 36;
  const ring: Array<[number, number]> = [];
  for (let k = 0; k <= n; k++) {
    const th = (k / n) * TAU;
    const w = 1 + 0.12 * Math.sin(3 * th + L.b0) + 0.07 * Math.sin(5 * th + L.b1);
    ring.push([cx + Math.cos(th) * pr * w, cz + Math.sin(th) * pr * w * (0.5 + 0.5 * L.aspect)]);
  }
  for (let k = 0; k < n; k++) pl.street(ring[k][0], ring[k][1], ring[k + 1][0], ring[k + 1][1], 3);
  for (let k = 0; k < n; k += 3) {
    const [px, pz] = ring[k];
    out.push(structOf(pl, 'lantern.post', px, pz, toward(px, pz, cx, cz)));
  }
  for (let k = 0; k < 5; k++) {
    const th = (k / 5) * TAU + 0.4;
    const gx = cx + Math.cos(th) * (pr + 12), gz = cz + Math.sin(th) * (pr + 12);
    if (pl.fits(gx, gz, 7, 6)) { out.push(rotunda(s, gx, gz, 4.5, 4.6, { columns: 8, onion: k % 2 === 0 })); pl.place(gx, gz, 7.5, 6.5); }
  }
  yield;
  // Homes up the bowl's terraces, facing the lagoon.
  for (let k = 0; k < n; k += 2) yield* lineStreet(pl, out, ring[k][0], ring[k][1], ring[k + 2 > n ? n : k + 2][0], ring[k + 2 > n ? n : k + 2][1], 3, 19, () => 0.4);
  yield* streets(pl, out, cx + ux * pr, cz + uz * pr, feet, 3.5);
  yield* homes(pl, out, 32, 0.55);
};

/** The Rotunda of the Accord on its islet: a round plaza rising from the lagoon's bed, a colonnade, a turquoise dome. */
function islet(pl: Planner, x: number, z: number, R: number, top: number, bed: number): Structure {
  const s = pl.site;
  const parts = [
    rotunda({ ...s, ground: () => top - 0.45 }, x, z, R - 4, 7.5, { columns: 12, hall: true, measure: true }),
  ];
  // its foot down into the water, below the rotunda's own podium
  const foot = structOf(pl, 'plaza.round', x, z, 0, R, top - bed, R);
  foot.pieces[0].y = top - 0.4;
  return makeStructure([...foot.pieces, ...parts[0].pieces]);
}

/** One piece as a structure (a lantern, a statue): at (x, z) on the ground, yaw, uniform scale or (sx, sy, sz). */
function structOf(pl: Planner, name: string, x: number, z: number, yaw: number, sx = 1, sy = sx, sz = sx): Structure {
  const piece = pieceId(name);
  const y = pl.site.ground(x, z);
  return { x, y: y + 2, z, r: Math.max(4, sx * 1.5), pieces: [{ piece, x, y, z, yaw, sx, sy, sz, palette: pl.site.palette, seed: Math.floor(pl.rnd() * 65536), flags: 0 }] };
}

const wellIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const mid = summit(pl, 10, isl.R * 0.3, 8);
  if (mid) {
    out.push(rotunda(pl.site, mid.x, mid.z, 5, 5.5, { columns: 8, onion: pl.rnd() < 0.5 }));
    pl.place(mid.x, mid.z, 9, 8);
    const th = pl.rnd() * TAU;
    const cxw = mid.x + Math.cos(th) * 22, czw = mid.z + Math.sin(th) * 22;
    if (pl.fits(cxw, czw, 9, 6)) { out.push(cistern(pl.site, cxw, czw, 7)); pl.place(cxw, czw, 9, 8); }
    yield* streets(pl, out, mid.x, mid.z, feet, 3);
  }
  yield* homes(pl, out, 30, 0.45);
};

// ---- Ringhold (§7.3) -------------------------------------------------------------------------------------

const haloIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const radius = Math.min(150, Math.max(70, isl.R * 0.42));
  const yaw = WIND + Math.PI / 2 + (pl.rnd() - 0.5) * 0.6;
  out.push(...haloMonument(pl.site, isl.cx, isl.cz, yaw, radius));
  // its foot, the cloister and the bell tower keep clear of it
  const ux = Math.cos(yaw), uz = -Math.sin(yaw);
  for (let k = -3; k <= 3; k++) pl.place(isl.cx + ux * k * radius * 0.12, isl.cz + uz * k * radius * 0.12, 16, 10);
  yield;
  // The Stonesingers' cloister: four arcaded wings round a court, off to one side of the ring.
  const nx = Math.sin(yaw), nz = Math.cos(yaw);
  const side = pl.rnd() < 0.5 ? -1 : 1;
  const qx = isl.cx + nx * side * (radius * 0.55 + 30), qz = isl.cz + nz * side * (radius * 0.55 + 30);
  if (pl.fits(qx, qz, 30, 12)) {
    for (let w = 0; w < 4; w++) {
      const a = yaw + (w / 4) * TAU;
      const [wx, wz] = [qx + Math.sin(a) * 15, qz + Math.cos(a) * 15];
      out.push(marketHall(pl.site, wx, wz, a + Math.PI / 2, 26, 7));
    }
    out.push(plaza(pl.site, qx, qz, 9, 0));
    pl.place(qx, qz, 30, 24);
    pl.paveDisc(qx, qz, 9);
    const bx = qx + Math.sin(yaw) * 26, bz = qz + Math.cos(yaw) * 26;
    if (pl.fits(bx, bz, 7, 10)) { out.push(roost(pl.site, bx, bz, { r: 5, floors: 7, face: toward(bx, bz, qx, qz), roof: 'spire', pennant: true })); pl.place(bx, bz, 7, 6); }
  }
  yield* streets(pl, out, isl.cx + nx * radius * 0.4, isl.cz + nz * radius * 0.4, feet, 3);
  yield* homes(pl, out, 36, 0.3);
};

const pilgrimIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const mid = summit(pl, 8, isl.R * 0.25, 6) ?? { x: isl.cx, z: isl.cz };
  const face = feet.length ? toward(mid.x, mid.z, feet[0].x, feet[0].z) : pl.rnd() * TAU;
  out.push(shrine(pl.site, mid.x, mid.z, face));
  pl.place(mid.x, mid.z, 8, 6);
  pl.paveDisc(mid.x, mid.z, 6);
  yield* streets(pl, out, mid.x, mid.z, feet, 2.5);
};

// ---- Lanternpeak (§7.4) ------------------------------------------------------------------------------------

const peakIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  // the summit: the highest ground near the middle
  let best = { x: isl.cx, z: isl.cz, y: -Infinity };
  for (let k = 0; k < 80; k++) {
    const a = pl.rnd() * TAU, d = Math.sqrt(pl.rnd()) * isl.R * 0.4;
    const x = isl.cx + Math.cos(a) * d, z = isl.cz + Math.sin(a) * d, y = pl.site.ground(x, z);
    if (y > best.y) best = { x, z, y };
  }
  const h = 70 + 50 * pl.rnd();
  out.push(vaelspire(pl.site, best.x, best.z, h));
  pl.place(best.x, best.z, h * 0.3, h * 0.2);
  yield;
  // the lamp-wardens' tower a little below the summit, an observatory on a shoulder
  for (const [d, kind] of [[0.28, 'tower'], [0.42, 'obs'], [0.5, 'obs']] as const) {
    const th = pl.rnd() * TAU;
    const x = best.x + Math.cos(th) * isl.R * d, z = best.z + Math.sin(th) * isl.R * d;
    if (!pl.fits(x, z, 10, 12)) continue;
    out.push(kind === 'tower' ? watchtower(pl.site, x, z, 4.5, 28, toward(x, z, best.x, best.z)) : observatory(pl.site, x, z, toward(x, z, best.x, best.z)));
    pl.place(x, z, 11, 8);
  }
  yield* streets(pl, out, best.x + (isl.cx - best.x) * 0.3, best.z + (isl.cz - best.z) * 0.3, feet, 2.5);
  yield* homes(pl, out, 40, 0.2);
};

const observatoryIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const mid = summit(pl, 12, isl.R * 0.3, 8) ?? { x: isl.cx, z: isl.cz };
  out.push(observatory(pl.site, mid.x, mid.z, feet.length ? toward(mid.x, mid.z, feet[0].x, feet[0].z) : 0));
  pl.place(mid.x, mid.z, 12, 9);
  const g = summit(pl, 7, isl.R * 0.5, 6);
  if (g) { out.push(rotunda(pl.site, g.x, g.z, 4, 4.5, { columns: 8 })); pl.place(g.x, g.z, 7, 6); }
  yield* streets(pl, out, mid.x, mid.z, feet, 2.5);
  yield* homes(pl, out, 34, 0.35);
};

const crystalIsle: Grammar = function* (_a, _i, pl, _feet, out) {
  const isl = pl.isl;
  out.push(vaelspireSmall(pl, isl.cx, isl.cz, Math.min(36, 12 + isl.R * 0.4)));
};
function vaelspireSmall(pl: Planner, x: number, z: number, h: number): Structure {
  const st = vaelspire(pl.site, x, z, h);
  st.fx = [];
  return st;
}

// ---- Petalfall (§7.5) ----------------------------------------------------------------------------------------

const terraceIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  // gate-arches at the bridgeheads, facing along the bridge
  for (const f of feet) {
    const gx = f.x - Math.sin(f.out) * 9, gz = f.z - Math.cos(f.out) * 9;
    out.push(torii(pl.site, gx, gz, f.out, 0.9 + f.w / 18));
  }
  const top = summit(pl, 16, isl.R * 0.35, 9);
  if (top) {
    const face = feet.length ? toward(top.x, top.z, feet[0].x, feet[0].z) : pl.rnd() * TAU;
    out.push(pagoda(pl.site, top.x, top.z, face, 3 + Math.floor(pl.rnd() * 3), 6 + 2 * pl.rnd()));
    pl.place(top.x, top.z, 16, 11);
    yield* streets(pl, out, top.x, top.z, feet, 2.5);
  }
  // tea pavilions by the pool
  const L = isl.lake;
  if (L) {
    for (let k = 0; k < 2; k++) {
      const th = pl.rnd() * TAU;
      const px = isl.cx + L.x + Math.cos(th) * L.r * 1.5, pz = isl.cz + L.z + Math.sin(th) * L.r * 1.5;
      if (pl.fits(px, pz, 6, 4)) { out.push(teaPavilion(pl.site, px, pz, toward(px, pz, isl.cx + L.x, isl.cz + L.z), 3.2)); pl.place(px, pz, 6, 5); }
    }
  }
  yield;
  // wind-houses of the orchard keepers among the trees (plastered hall-houses), fewer than a city's
  for (let k = 0; k < 6; k++) {
    const a = pl.rnd() * TAU, d = isl.R * (0.3 + 0.45 * pl.rnd());
    const x = isl.cx + Math.cos(a) * d, z = isl.cz + Math.sin(a) * d;
    if (!pl.fits(x, z, 11, 6)) continue;
    if (pl.rnd() < 0.4) out.push(pagoda(pl.site, x, z, pl.rnd() * TAU, 2, 4.5 + pl.rnd()));
    else {
      const st = hallHouse(pl.site, x, z, { face: pl.rnd() * TAU, len: 12, wid: 8, floors: 2, tower: 0 });
      for (const p of st.pieces) p.flags |= PFLAG.plaster;
      out.push(st);
    }
    pl.place(x, z, 11, 9);
    yield;
  }
};

// ---- Galeharbour (§7.10) -------------------------------------------------------------------------------------

const quayIsle: Grammar = function* (a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  // docks on the side facing the anchorage (the Eyrie's middle)
  const inward = Math.atan2(a.cz - isl.cz, a.cx - isl.cx);
  for (let k = -1; k <= 1; k++) {
    const th = inward + k * 0.42;
    const [x, z] = pl.rim(th, 6);
    const out0 = Math.atan2(Math.cos(th), Math.sin(th)); // yaw toward the bearing th
    if (feet.some((f) => Math.hypot(f.x - x, f.z - z) < 30)) continue;
    out.push(...harbourDock(pl.site, x, z, out0, 30 + 12 * pl.rnd(), pl.rnd() < 0.75));
    pl.place(x, z, 14, 9);
    yield;
  }
  // the market: stalls round a plaza, warehouses about it
  const mid = summit(pl, 18, isl.R * 0.35, 10);
  if (mid) {
    out.push(plaza(pl.site, mid.x, mid.z, 16, 6));
    out.push(stalls(pl.site, mid.x, mid.z, 16, 8));
    pl.take(mid.x, mid.z, 17, true);
    pl.paveDisc(mid.x, mid.z, 16);
    yield* streets(pl, out, mid.x, mid.z, feet, 3.5);
  }
  // a lighthouse on the outer rim
  const [lx, lz] = pl.rim(inward + Math.PI + (pl.rnd() - 0.5), 14);
  if (pl.fits(lx, lz, 6, 8, 1)) { out.push(watchtower(pl.site, lx, lz, 4, 30, inward)); pl.place(lx, lz, 6, 5); }
  // warehouses (timbered hall-houses) and homes
  for (let k = 0; k < 8; k++) {
    const a2 = pl.rnd() * TAU, d = isl.R * (0.25 + 0.5 * pl.rnd());
    const x = isl.cx + Math.cos(a2) * d, z = isl.cz + Math.sin(a2) * d;
    if (!pl.fits(x, z, 12, 6)) continue;
    const st = hallHouse(pl.site, x, z, { face: pl.rnd() * TAU, len: 16, wid: 10, floors: 2, tower: 0 });
    for (const p of st.pieces) p.flags |= PFLAG.timber;
    out.push(st);
    pl.place(x, z, 12, 10);
  }
  yield* homes(pl, out, 30, 0.5);
};

// ---- Mistwells (§7.11) ------------------------------------------------------------------------------------------

const combIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const n = 3 + Math.floor(pl.rnd() * 4);
  for (let k = 0; k < n; k++) {
    const th = pl.rnd() * TAU;
    const [x, z] = pl.rim(th, 10);
    if (!pl.fits(x, z, 11, 8, 1)) continue;
    out.push(comb(pl.site, x, z, Math.atan2(Math.cos(th), Math.sin(th))));
    pl.place(x, z, 11, 3);
  }
  const mid = summit(pl, 8, isl.R * 0.35, 8);
  if (mid) {
    out.push(windwheel(pl.site, mid.x, mid.z, WIND + Math.PI / 2));
    pl.place(mid.x, mid.z, 8, 4.5);
    const cx = mid.x + 22, cz = mid.z + 6;
    if (pl.fits(cx, cz, 9, 6)) { out.push(cistern(pl.site, cx, cz, 7)); pl.place(cx, cz, 9, 8); }
    yield* streets(pl, out, mid.x, mid.z, feet, 2.5);
  }
  yield* homes(pl, out, 40, 0.25);
};

// ---- Cinderforge (§7.7) -----------------------------------------------------------------------------------------

const forgeIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const top = summit(pl, 50, isl.R * 0.25, 30);
  if (top) {
    const face = feet.length ? toward(top.x, top.z, feet[0].x, feet[0].z) : 0;
    // the old forge-lords' citadel, now the Hall of Penance
    out.push(...cathedral(pl.site, top.x, top.z, { face, length: 48, width: 18, height: 24, spires: 5 }));
    pl.place(top.x, top.z, 40, 34);
  }
  yield;
  for (let k = 0; k < 4; k++) {
    const a2 = pl.rnd() * TAU, d = isl.R * (0.35 + 0.35 * pl.rnd());
    const x = isl.cx + Math.cos(a2) * d, z = isl.cz + Math.sin(a2) * d;
    if (!pl.fits(x, z, 18, 10)) continue;
    out.push(forge(pl.site, x, z, pl.rnd() * TAU));
    pl.place(x, z, 18, 15);
    yield;
  }
  yield* streets(pl, out, top ? top.x : isl.cx, top ? top.z : isl.cz, feet, 3);
  yield* homes(pl, out, 34, 0.35);
};

const yardIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  for (let k = 0; k < 2; k++) {
    const th = pl.rnd() * TAU;
    const [x, z] = pl.rim(th, 6);
    if (!pl.fits(x, z, 14, 10, 1)) continue;
    out.push(...harbourDock(pl.site, x, z, Math.atan2(Math.cos(th), Math.sin(th)), 30, true));
    pl.place(x, z, 14, 9);
  }
  const mid = summit(pl, 18, isl.R * 0.3, 10);
  if (mid) { out.push(forge(pl.site, mid.x, mid.z, pl.rnd() * TAU)); pl.place(mid.x, mid.z, 18, 15); }
  yield* homes(pl, out, 36, 0.3);
};

// ---- Rimewatch (§7.8) ---------------------------------------------------------------------------------------------

const citadelIsle: Grammar = function* (a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const top = summit(pl, 16, isl.R * 0.3, 12) ?? { x: isl.cx, z: isl.cz };
  const k0 = keep(pl.site, top.x, top.z, feet.length ? toward(top.x, top.z, feet[0].x, feet[0].z) : 0, 10 + 3 * pl.rnd(), 30 + 10 * pl.rnd());
  // the Gyre's Eye turns over the citadel
  k0.fx = [{ k: 'vortex', x: a.cx, y: isl.y + 1100, z: a.cz, r: 2400 }];
  out.push(k0);
  pl.place(top.x, top.z, 18, 15);
  yield;
  // wind walls round the whole rim (open where the bridges land)
  const pts: Array<[number, number]> = [];
  const n = 40;
  for (let k = 0; k <= n; k++) {
    const th = (k / n) * TAU;
    const p = pl.rim(th, 10);
    if (feet.some((f) => Math.hypot(f.x - p[0], f.z - p[1]) < f.w + 26)) {
      if (pts.length > 1) out.push(...windWall(pl.site, pts, 9));
      pts.length = 0;
      continue;
    }
    pts.push(p);
    pl.place(p[0], p[1], 6, 5);
  }
  if (pts.length > 1) out.push(...windWall(pl.site, pts, 9));
  yield;
  yield* streets(pl, out, top.x, top.z, feet, 3.5);
  yield* homes(pl, out, 30, 0.55);
};

const watchIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  const mid = summit(pl, 7, isl.R * 0.3, 8) ?? { x: isl.cx, z: isl.cz };
  out.push(watchtower(pl.site, mid.x, mid.z, 4.5, 24 + 8 * pl.rnd(), pl.rnd() * TAU));
  pl.place(mid.x, mid.z, 7, 5.5);
  yield* streets(pl, out, mid.x, mid.z, feet, 2.5);
  yield* homes(pl, out, 34, 0.35);
};

// ---- the Elder Bough (§7.6) -------------------------------------------------------------------------------------

const rootIsle: Grammar = function* (_a, _i, pl, feet, out) {
  const isl = pl.isl;
  bridgeheads(pl, feet);
  out.push(...elderBough(pl.site, isl, Math.min(1.25, Math.max(0.8, isl.R / 300))));
  pl.place(isl.cx, isl.cz, 50, 30);
  yield;
  yield* streets(pl, out, isl.cx + 40, isl.cz, feet, 2.5);
  // its people's homes are in its limbs: only a few roosts below, timbered
  const before = out.length;
  yield* homes(pl, out, 44, 0.25);
  for (let k = before; k < out.length; k++) for (const p of out[k].pieces) p.flags |= PFLAG.timber;
};

const groveIsle: Grammar = function* (_a, _i, pl, feet, out) {
  bridgeheads(pl, feet);
  const before = out.length;
  yield* homes(pl, out, 40, 0.3);
  for (let k = before; k < out.length; k++) for (const p of out[k].pieces) p.flags |= PFLAG.timber;
};

// ---- Hushfall (§7.9) ----------------------------------------------------------------------------------------------

const ruinIsle: Grammar = function* (a, i, pl, _feet, out) {
  const isl = pl.isl;
  const main = i === 0;
  if (main) {
    const top = summit(pl, 50, isl.R * 0.3, 30);
    if (top) {
      const face = pl.rnd() * TAU;
      // the cathedral as the Long Gale left it: its west front and towers standing, its nave unroofed
      const parts = cathedral(pl.site, top.x, top.z, { face, length: 58, width: 20, height: 26, spires: 2 });
      for (const st of parts) {
        const kept: PieceInst[] = [];
        for (const p of st.pieces) {
          if (ROOFS.has(p.piece)) continue;
          p.flags |= PFLAG.ruined | PFLAG.dark;
          kept.push(p);
          // (the crossing tower's walls end broken where its dome fell in)
          if (p.piece === OCTAGON) kept.push({ ...p, piece: RUIN_TOP, y: p.y + p.sy - 0.05, sy: 1, seed: p.seed ^ 0x5a5a });
        }
        st.pieces = kept;
      }
      out.push(...parts);
      out.push(ruinedHall(pl.site, top.x + Math.sin(face + 1.6) * 60, top.z + Math.cos(face + 1.6) * 60, face, 30, 14));
      pl.place(top.x, top.z, 48, 40);
    }
  }
  yield;
  // roosts cut short, scattered as a city was; rubble; fallen columns; a few Wardens still standing
  const n = Math.round((isl.R * isl.R) / 2600);
  for (let k = 0; k < n; k++) {
    const a2 = pl.rnd() * TAU, d = Math.sqrt(pl.rnd()) * isl.R * 0.8;
    const x = isl.cx + Math.cos(a2) * d, z = isl.cz + Math.sin(a2) * d;
    const r = 4 + 3 * pl.rnd();
    if (!pl.fits(x, z, r + 2, 9)) continue;
    out.push(ruinedRoost(pl.site, x, z, r, pl.rnd() * TAU));
    pl.place(x, z, r + 3, r + 1);
    if (k % 6 === 0) yield;
  }
  for (let k = 0; k < 4; k++) {
    const a2 = pl.rnd() * TAU, d = isl.R * (0.2 + 0.5 * pl.rnd());
    const x = isl.cx + Math.cos(a2) * d, z = isl.cz + Math.sin(a2) * d;
    if (pl.fits(x, z, 3, 4)) { out.push(structOf(pl, 'warden', x, z, pl.rnd() * TAU)); pl.place(x, z, 3, 2); }
  }
  // bridges that end in the air, out from the rim toward the islands they once reached
  for (let k = 0; k < (main ? 3 : 1); k++) {
    const other = a.islands[(i + 1 + Math.floor(pl.rnd() * (a.islands.length - 1))) % a.islands.length];
    const th = Math.atan2(other.cz - isl.cz, other.cx - isl.cx);
    const [x, z] = pl.rim(th, 12);
    if (!pl.fits(x, z, 6, 6, 1)) continue;
    out.push(...brokenBridge(pl.site, x, z, Math.atan2(Math.cos(th), Math.sin(th)), 1 + Math.floor(pl.rnd() * 3), 10));
  }
  yield;
};

/** What the Long Gale tore off first: roofs, spires, finials, pinnacles. */
const ROOFS = new Set(['roof.gable', 'roof.spire', 'roof.cone', 'roof.pyramid', 'roof.dome', 'roof.apse', 'finial', 'pinnacle'].map(pieceId));
const OCTAGON = pieceId('body.octagon'), RUIN_TOP = pieceId('ruin.top');

/** The roles of every class but the Crown, and their grammars. */
export const CLASS_GRAMMARS: Record<string, Grammar> = {
  bowl: bowlIsle, well: wellIsle, halo: haloIsle, pilgrim: pilgrimIsle, peak: peakIsle, observatory: observatoryIsle, crystal: crystalIsle,
  terrace: terraceIsle, quay: quayIsle, comb: combIsle, forge: forgeIsle, yard: yardIsle, citadel: citadelIsle, watch: watchIsle,
  root: rootIsle, grove: groveIsle, ruin: ruinIsle,
};

