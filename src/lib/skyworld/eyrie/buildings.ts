/**
 * The Eyries' buildings: each turns a few choices (where, how big, which way it
 * faces, its palette) into kit pieces bedded into the ground, as one or more
 * structures (structures.ts). The grammars (classes.ts) decide what stands
 * where; these decide how it is put together (SKY_CIVILIZATION.md §6, §8).
 */

import { pieceId } from '../kit/pieces';
import { RAMP_START, RAMP_SWEEP, STOREY_H } from '../kit/interiors';
import { makeStructure, PFLAG, type Effect, type PieceInst, type Structure } from '../structures';

/** The ground under world (x, z) (an island's top), and a stream of [0, 1) numbers. */
export type Site = { ground: (x: number, z: number) => number; rnd: () => number; palette: number };

const P = {
  bodyRound: pieceId('body.round'), bodyRoundHall: pieceId('body.roundHall'), bodyRoundBlind: pieceId('body.roundBlind'),
  bodySquare: pieceId('body.square'), bodyHall: pieceId('body.hall'), bodyFront: pieceId('body.front'), bodyBelfry: pieceId('body.belfry'),
  bodyLattice: pieceId('body.lattice'),
  bodyRoundBelfry: pieceId('body.roundBelfry'),
  bodyBlind: pieceId('body.blind'), bodyOctagon: pieceId('body.octagon'), podium: pieceId('podium'), plaza: pieceId('plaza.round'),
  roofCone: pieceId('roof.cone'), roofSpire: pieceId('roof.spire'), roofPyramid: pieceId('roof.pyramid'), roofGable: pieceId('roof.gable'),
  roofDome: pieceId('roof.dome'), roofOnion: pieceId('roof.onion'), roofGlass: pieceId('roof.glass'), bodyArcade: pieceId('body.arcade'),
  crenel: pieceId('crenel'), bodyWall: pieceId('body.wall'),
  haloRing: pieceId('halo.ring'), runestone: pieceId('runestone'), runestoneFloat: pieceId('runestone.float'), crystal: pieceId('crystal.cluster'),
  roofPagoda: pieceId('roof.pagoda'), torii: pieceId('gate.torii'), paperLantern: pieceId('lantern.paper'), ropeBridge: pieceId('bridge.rope'),
  dock: pieceId('dock'), crane: pieceId('crane'), awning: pieceId('awning'), stall: pieceId('stall'), cargo: pieceId('cargo'), skyship: pieceId('skyship.moored'),
  comb: pieceId('cloud.comb'), windSails: pieceId('wind.sails'), chimney: pieceId('chimney'), brazier: pieceId('brazier'),
  boughTree: pieceId('bough.tree'), boughRoots: pieceId('bough.roots'), nest: pieceId('nest.platform'), ruinTop: pieceId('ruin.top'),
  columnFallen: pieceId('column.fallen'), rubble: pieceId('rubble'),
  cornice: pieceId('ring.cornice'), ledge: pieceId('ring.ledge'), parapet12: pieceId('ring.parapet12'), parapet20: pieceId('ring.parapet20'),
  parapet32: pieceId('ring.parapet32'), curb: pieceId('ring.curb'), entablature: pieceId('ring.entablature'),
  finial: pieceId('finial'), pinnacle: pieceId('pinnacle'), turret: pieceId('turret'), buttress: pieceId('buttress'), column: pieceId('column'),
  door: pieceId('door'), lanternPost: pieceId('lantern.post'), lanternHang: pieceId('lantern.hang'), banner: pieceId('banner'), pennant: pieceId('pennant'),
  warden: pieceId('warden'), fountain: pieceId('fountain'), gate: pieceId('gate.arch'),
  storeyRound: pieceId('storey.round'), storeySquare: pieceId('storey.square'), doorOpen: pieceId('door.open'),
  jars: pieceId('jars'), hearth: pieceId('hearth'), loom: pieceId('loom'), nestBowl: pieceId('nest.bowl'), rug: pieceId('rug'),
  table: pieceId('table'), perch: pieceId('perch'), shelves: pieceId('shelves'), ramp: pieceId('ramp.spiral'),
  bodyHallOpen: pieceId('body.hallOpen'), tableLong: pieceId('table.long'), gallery: pieceId('gallery'), altar: pieceId('altar'),
  bodyApse: pieceId('body.apse'), roofApse: pieceId('roof.apse'),
  storeyRoundOpen: pieceId('storey.roundOpen'), bodyRoundOpen: pieceId('body.roundOpen'), tiers: pieceId('tiers'), orrery: pieceId('orrery'),
  telescope: pieceId('telescope'), planter: pieceId('planter'),
  storeyLattice: pieceId('storey.lattice'), storeyOctagon: pieceId('storey.octagon'), rampSlab: pieceId('ramp.slab'), pool: pieceId('pool.round'),
  rod: pieceId('rod.measure'), anvil: pieceId('anvil'), furnace: pieceId('furnace'), shrineSmall: pieceId('shrine.small'),
  span: pieceId('bridge.span'), spanPointed: pieceId('bridge.spanPointed'), pier: pieceId('bridge.pier'), parapet: pieceId('parapet'),
  balustrade: pieceId('balustrade'), stair6: pieceId('stair.6'), stair12: pieceId('stair.12'), stair24: pieceId('stair.24'),
};

/** Piece-instance helper: world position, yaw, scale, the site's palette, a seed from the site's stream. */
function inst(s: Site, piece: number, x: number, y: number, z: number, yaw: number, sx: number, sy: number, sz: number, flags = 0): PieceInst {
  return { piece, x, y, z, yaw, sx, sy, sz, palette: s.palette, seed: Math.floor(s.rnd() * 65536), flags };
}

/** The lowest and highest ground in a disc of radius r round (x, z) (centre and 12 round it). */
export function groundRange(s: Site, x: number, z: number, r: number): [number, number] {
  let lo = s.ground(x, z), hi = lo;
  for (let k = 0; k < 12; k++) {
    const a = (k / 12) * Math.PI * 2;
    for (const f of [0.55, 1]) {
      const g = s.ground(x + Math.cos(a) * r * f, z + Math.sin(a) * r * f);
      lo = Math.min(lo, g); hi = Math.max(hi, g);
    }
  }
  return [lo, hi];
}

/** Local (u along the yaw's right, v along its forward) → world, at (x, z) with yaw. */
const at = (x: number, z: number, yaw: number, u: number, v: number): [number, number] =>
  [x + Math.cos(yaw) * u + Math.sin(yaw) * v, z - Math.sin(yaw) * u + Math.cos(yaw) * v];

/**
 * Steps up to a doorway at (x, z) facing `face` whose floor is fl, from the ground in front of it
 * (none where that is within a step of the floor): a flight `half` metres either side of its axis.
 */
function doorSteps(s: Site, pieces: PieceInst[], x: number, z: number, face: number, fl: number, half: number): void {
  const fx = Math.sin(face), fz = Math.cos(face);
  const rise = fl - s.ground(x + fx * 1.5, z + fz * 1.5);
  if (rise < 0.25) return;
  const run = Math.max(0.8, rise * 1.6);
  // (its foot out in front, climbing toward the door: the flight's local +z points up it)
  pieces.push(inst(s, rise > 1.2 ? P.stair12 : P.stair6, x + fx * run, fl - rise - 0.05, z + fz * run, face + Math.PI, half, rise + 0.05, run));
}

/**
 * The ground floor's furnishings (SKY_CIVILIZATION.md §8: a roost's store jars, the flightless's ramp,
 * a hearth, a loom, a nest, perch bars): in a room of inner radius (or half width) R at floor fl, its
 * door at the local +z. In a round room the ramp winds up the wall from beside the door to a hatch, and
 * the rest keep inside it; a square room has shelves on its walls instead. As many as the room holds.
 */
function furnish(s: Site, pieces: PieceInst[], x: number, z: number, face: number, R: number, fl: number, square: boolean): void {
  // local polar (a: the mesh's bearing, the door at π/2; d: from the middle) to world, facing the middle
  const put = (piece: number, a: number, d: number, k = 1, y = fl, turn = 0) => {
    const [px, pz] = at(x, z, face, Math.cos(a) * d, Math.sin(a) * d);
    pieces.push({ ...inst(s, piece, px, y, pz, face + Math.atan2(-Math.cos(a), -Math.sin(a)) + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  const free = square ? R - 0.6 : R - 1.05; // (clear of the ramp, or the shelves)
  if (!square) pieces.push({ ...inst(s, P.ramp, x, fl, z, face, R, 1, R, PFLAG.indoor), detail: 0 });
  put(P.rug, 0, 0, Math.min(1, free / 1.5));
  put(P.lanternHang, 0, 0, 1, fl + STOREY_H - 0.06);
  put(P.hearth, 1.5 * Math.PI, Math.max(0, free - 0.85), Math.min(1, free / 1.4));
  // against the wall clear of the ramp (round), or of the door (square): jars, a loom
  put(P.jars, -0.55, square ? R - 0.75 : R - 0.55);
  if (free >= 1.6) put(P.loom, 0.4, square ? R - 0.45 : R - 0.5);
  if (free >= 2.0) {
    put(P.table, Math.PI + 0.55, 0.55 * free);
    put(P.nestBowl, 2 * Math.PI - 0.9, 0.5 * free);
  }
  if (free >= 2.6) put(P.perch, Math.PI + 1.4, 0.75 * free, 1, fl, Math.PI / 2);
  if (square) {
    put(P.shelves, Math.PI, R - 0.25);
    if (free >= 2.4) put(P.shelves, 1.5 * Math.PI + 0.6, R - 0.25);
  }
}

/**
 * A hall's furnishings (its room ±hx along its length, ±hz across, floor fl, its door in the middle of
 * its +z wall): the long table and its benches on a rug, a hearth at one end, shelves along the back
 * wall, store jars in a corner, a loom by the far end, two lamps hung over the table.
 */
function furnishHall(s: Site, pieces: PieceInst[], x: number, z: number, yaw: number, hx: number, hz: number, fl: number): void {
  const put = (piece: number, u: number, v: number, turn: number, k = 1, y = fl) => {
    const [px, pz] = at(x, z, yaw, u, v);
    pieces.push({ ...inst(s, piece, px, y, pz, yaw + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  put(P.rug, 0, -0.1 * hz, 0, Math.min(2, Math.min(hx, hz) / 1.6));
  if (hx >= 3.2) put(P.tableLong, 0, -0.1 * hz, 0);
  put(P.hearth, -hx + 1.0, 0, Math.PI / 2, 0.9);
  for (const side of [-1, 1]) if (hx >= 2.6) put(P.shelves, side * hx * 0.5, -hz + 0.25, 0);
  put(P.jars, hx - 0.8, -hz + 0.85, -Math.PI / 4);
  if (hx >= 4) put(P.loom, hx - 0.45, 0.25 * hz, -Math.PI / 2);
  for (const side of [-1, 1]) put(P.lanternHang, side * hx * 0.35, -0.1 * hz, 0, 1, fl + STOREY_H - 0.06);
}

/**
 * A rotunda hall's inside (its inner radius R, floor fl, `tall` to its open top under the dome, its
 * door at the local +z): tiered perch-rings round the wall; inside them the Moot's speaking floor
 * with the speaker's perch, or the Accord's Measure (its basin, the measuring steps the tiers); a
 * chandelier hung from the dome.
 */
function furnishRotunda(s: Site, pieces: PieceInst[], x: number, z: number, face: number, R: number, fl: number, tall: number, measure: boolean): void {
  const add = (piece: number, px: number, py: number, pz: number, yaw: number, a: number, b: number, c: number) =>
    pieces.push({ ...inst(s, piece, px, py, pz, yaw, a, b, c, PFLAG.indoor), detail: 0 });
  add(P.tiers, x, fl, z, face, R, 1, R);
  const floorR = Math.max(1.5, R - 3.6);
  add(P.plaza, x, fl + 0.04, z, 0, floorR, 0.05, floorR);
  if (measure) add(P.fountain, x, fl + 0.04, z, 0, Math.min(0.8, floorR / 4), Math.min(0.8, floorR / 4), Math.min(0.8, floorR / 4));
  else add(P.perch, x, fl + 0.04, z, face, 1, 1, 1);
  add(P.lanternHang, x, fl + tall - 1.2, z, 0, 2.2, 2.2, 2.2);
}

export type RoostOpts = {
  r: number;          // body radius (m)
  floors: number;     // 3.6 m each
  square?: boolean;
  roof?: 'cone' | 'battlement' | 'spire' | 'dome';
  face: number;       // bearing its door faces (yaw)
  pennant?: boolean;
  turrets?: number;   // bartizans round its top (battlement roofs)
  tier?: boolean;     // a narrower upper tower on top
};

/**
 * A roost (§8): a brood's tower-house. A buried plinth, the body with a perch-ledge at every
 * second floor, a cornice, then a roof — a bell-cast cone, battlements (with turrets), a needle
 * spire or a dome — a finial, perhaps a pennant; its door at the foot facing `face`, a lantern by it.
 */
export function roost(s: Site, x: number, z: number, o: RoostOpts): Structure {
  const pieces: PieceInst[] = [];
  const [lo, hi] = groundRange(s, x, z, o.r);
  const base = lo + 0.05;
  // its ground storey's floor over the highest ground under it (no ground shows inside), the storey
  // walked into through its door, the body above it
  const fl = hi + 0.12;
  const storeyTop = fl + STOREY_H;
  const top = Math.max(base + o.floors * 3.6 + 0.5 + (hi - lo), storeyTop + 3.6);
  const body = o.square ? P.bodySquare : P.bodyRound;
  const hw = o.square ? o.r * 0.88 : o.r;
  // plinth into the ground, up to the floor
  pieces.push({ ...inst(s, o.square ? P.bodyBlind : P.bodyRoundBlind, x, base - 6, z, o.face, hw * 1.06, fl - base + 6, hw * 1.06), detail: 1 });
  pieces.push(inst(s, o.square ? P.storeySquare : P.storeyRound, x, fl, z, o.face, hw, 1, hw));
  pieces.push(inst(s, body, x, storeyTop, z, o.face, hw, top - storeyTop, hw, PFLAG.upper));
  if (!o.square) {
    for (let f = 2 + Math.floor(s.rnd() * 2); f < o.floors; f += 3) pieces.push(inst(s, P.ledge, x, storeyTop + (f - 1) * 3.6, z, s.rnd() * 6.28, o.r, 1, o.r));
    pieces.push(inst(s, P.cornice, x, top - 0.05, z, 0, o.r, 1, o.r));
  }
  let crown = top;
  const roof = o.roof ?? 'cone';
  if (roof === 'battlement' && !o.square) {
    const pp = o.r < 4.5 ? P.parapet12 : o.r < 7.5 ? P.parapet20 : P.parapet32;
    pieces.push(inst(s, pp, x, top + 0.6, z, s.rnd() * 6.28, o.r, 1, o.r));
    crown = top + 2.8;
    const nt = o.turrets ?? 0;
    for (let k = 0; k < nt; k++) {
      const a = o.face + Math.PI / 4 + (k / nt) * Math.PI * 2;
      const tr = o.r + 0.6;
      pieces.push(inst(s, P.turret, x + Math.sin(a) * tr, top - 1.6, z + Math.cos(a) * tr, a, 1, 1, 1));
    }
    if (o.tier) {
      // a narrower upper tower rising from the walk, with its own cone
      const r2 = o.r * 0.55, h2 = 3.6 * 2 + 0.5;
      pieces.push(inst(s, P.bodyRound, x, top + 0.2, z, o.face, r2, h2, r2, PFLAG.upper));
      pieces.push(inst(s, P.cornice, x, top + 0.2 + h2 - 0.05, z, 0, r2, 1, r2));
      const rh = r2 * (2.4 + 0.8 * s.rnd());
      pieces.push(inst(s, P.roofCone, x, top + 0.2 + h2 + 0.6, z, s.rnd() * 6.28, r2 * 1.05, rh, r2 * 1.05));
      pieces.push(inst(s, P.finial, x, top + 0.2 + h2 + 0.6 + rh - 0.15, z, 0, 1, 1, 1));
      crown = top + 0.2 + h2 + 0.6 + rh + 2.2;
    }
  } else if (roof === 'dome') {
    const dh = o.r * 0.95;
    pieces.push(inst(s, P.roofDome, x, top + 0.55, z, 0, o.r * 0.98, dh, o.r * 0.98));
    pieces.push(inst(s, P.finial, x, top + 0.55 + dh - 0.1, z, 0, 1.2, 1.2, 1.2));
    crown = top + dh + 3;
  } else {
    const spire = roof === 'spire';
    const rh = o.r * (spire ? 4.2 + 1.5 * s.rnd() : 2.1 + 1.0 * s.rnd());
    const rp = o.square ? P.roofPyramid : spire ? P.roofSpire : P.roofCone;
    const rr = o.square ? hw * 1.0 : o.r * (spire ? 0.95 : 1.0);
    pieces.push(inst(s, rp, x, top + (o.square ? 0 : 0.6), z, o.square ? o.face : s.rnd() * 6.28, rr, rh, rr));
    const ft = top + (o.square ? 0 : 0.6) + rh - (spire ? 0.3 : 0.15);
    pieces.push(inst(s, P.finial, x, ft, z, 0, 1, 1, 1));
    if (o.pennant) pieces.push(inst(s, P.pennant, x, ft + 1.6, z, s.rnd() * 6.28, 1, 1, 1));
    crown = ft + 2.4;
  }
  // its doorway's portal, steps up to it where the ground falls away, a lantern by it
  const dr = hw + 0.02;
  const [dx, dz] = [x + Math.sin(o.face) * dr, z + Math.cos(o.face) * dr];
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, o.face, 1, 1, 1));
  doorSteps(s, pieces, dx, dz, o.face, fl, 1.15);
  const [lx, lz] = at(dx, dz, o.face, 2.4, 1.2);
  pieces.push(inst(s, P.lanternPost, lx, s.ground(lx, lz) - 0.05, lz, o.face, 1, 1, 1));
  // and inside, what a brood keeps on its ground floor
  furnish(s, pieces, x, z, o.face, 0.9 * hw, fl, !!o.square);
  const st = makeStructure(pieces);
  st.y = Math.max(st.y, (base + crown) / 2);
  return st;
}

type CathedralOpts = {
  face: number;       // yaw its west front faces (out to its plaza)
  length: number;     // nave (m)
  width: number;      // nave (m)
  height: number;     // nave walls (m)
  spires: 1 | 2 | 3 | 5;
};

/**
 * The Gyre Cathedral (§7.1, §8): on a podium, a nave with lancet windows and buttresses crowned by
 * pinnacles under a steep gabled roof; a west front with a rose window between two square towers
 * with needle spires; a crossing tower with a spire (or a dome); a round apse; Wardens at the door.
 * Several structures (the culling groups: front, nave, crossing, apse).
 */
export function cathedral(s: Site, x: number, z: number, o: CathedralOpts): Structure[] {
  const out: Structure[] = [];
  const yaw = o.face;
  const L = o.length, W = o.width, H = o.height;
  // podium under it all
  const [lo, hi] = groundRange(s, x, z, Math.max(L, W) * 0.62);
  const floor = hi + 0.6;
  // (long enough behind for the apse)
  const podL = Math.max(L / 2 + 8, L / 2 + W / 2 + 3), podW = W / 2 + 7;
  const pod = [inst(s, P.podium, x, floor, z, yaw, podW, floor - lo + 4, podL)];
  // a grand stair down the front
  const [sx, sz] = at(x, z, yaw, 0, podL + 0.0);
  const fall = floor - s.ground(sx, sz);
  if (fall > 0.6) {
    const run = Math.max(3, fall * 1.7);
    const [qx, qz] = at(x, z, yaw, 0, podL + run);
    pod.push(inst(s, fall > 3 ? P.stair24 : P.stair12, qx, floor - fall - 0.05, qz, yaw + Math.PI, 7, fall + 0.05, run));
  }
  out.push(makeStructure(pod));
  // nave: its long axis along the yaw's forward
  const nave: PieceInst[] = [];
  // (open at its west end, which the front closes, and at its top: inside, its roof is its ceiling)
  nave.push(inst(s, P.bodyHallOpen, x, floor, z, yaw, W / 2, H, L / 2));
  const roofH = W * 0.62;
  nave.push(inst(s, P.roofGable, x, floor + H + 0.2, z, yaw + Math.PI / 2, L / 2, roofH, W / 2 + 0.2));
  const bays = Math.max(3, Math.round(L / 4.8));
  for (let k = 1; k < bays; k++) {
    const v = -L / 2 + (k / bays) * L;
    for (const side of [-1, 1]) {
      const [bx, bz] = at(x, z, yaw, side * (W / 2), v);
      const by = yaw + side * Math.PI / 2;
      nave.push({ ...inst(s, P.buttress, bx, floor, bz, by, 1, H * 0.86, 1), detail: 1 });
      const [px, pz] = at(x, z, yaw, side * (W / 2 + 1.1), v);
      nave.push(inst(s, P.pinnacle, px, floor + H * 0.86, pz, by, 0.9, 0.9, 0.9));
    }
  }
  out.push(makeStructure(nave));
  // west front: a rose window between two towers
  const front: PieceInst[] = [];
  const fz = L / 2 + 1.5;
  // the great door (±DH, DT high): plain masonry either side of it up to its head, the front with its
  // rose window over it (its tympanum filling the portal's arch)
  const DH = 2.2, DT = 6.0;
  for (const side of [-1, 1]) {
    const [px, pz] = at(x, z, yaw, side * (DH + (W / 2 - DH) / 2), fz);
    front.push(inst(s, P.bodyBlind, px, floor, pz, yaw, (W / 2 - DH) / 2, DT, 2.2));
  }
  const [fx0, fz0] = at(x, z, yaw, 0, fz);
  front.push(inst(s, P.bodyFront, fx0, floor + DT, fz0, yaw, W / 2, H + roofH * 0.7 - DT, 2.2));
  const twH = H + roofH + 6;
  for (const side of [-1, 1]) {
    const [tx, tz] = at(x, z, yaw, side * (W / 2 + 3.6), fz - 1.0);
    front.push(inst(s, P.bodyBelfry, tx, floor, tz, yaw, 4.2, twH, 4.2));
    const sh = 30 + 12 * s.rnd();
    if (o.spires >= 2) {
      front.push(inst(s, P.roofSpire, tx, floor + twH, tz, yaw, 4.6, sh, 4.6));
      front.push(inst(s, P.finial, tx, floor + twH + sh - 0.5, tz, 0, 1.3, 1.3, 1.3));
    } else {
      front.push(inst(s, P.roofPyramid, tx, floor + twH, tz, yaw, 4.4, 6, 4.4));
    }
    for (const cu of [-1, 1]) {
      const [px, pz] = at(tx, tz, yaw, cu * 3.8, 3.8);
      front.push(inst(s, P.pinnacle, px, floor + twH, pz, yaw, 1, 1, 1));
    }
  }
  const [dx, dz] = at(x, z, yaw, 0, fz + 2.22);
  front.push(inst(s, P.doorOpen, dx, floor, dz, yaw, DH, DH, DH));
  for (const side of [-1, 1]) {
    const [wx, wz] = at(x, z, yaw, side * 5.5, fz + 6);
    front.push(inst(s, P.warden, wx, floor, wz, yaw, 1, 1, 1));
    const [lx, lz] = at(x, z, yaw, side * 3.4, fz + 4);
    front.push(inst(s, P.lanternPost, lx, floor, lz, yaw, 1, 1, 1));
  }
  out.push(makeStructure(front));
  // crossing tower at the east third, and the apse
  const east: PieceInst[] = [];
  const [cx, cz] = at(x, z, yaw, 0, -L * 0.2);
  const ch = roofH + 9;
  east.push(inst(s, P.bodyOctagon, cx, floor + H, cz, yaw, W / 2 + 0.8, ch, W / 2 + 0.8));
  if (o.spires === 3 || o.spires === 5 || o.spires === 1) {
    const sh = 44 + 18 * s.rnd();
    east.push(inst(s, P.roofSpire, cx, floor + H + ch, cz, yaw, W / 2 + 0.9, sh, W / 2 + 0.9));
    east.push(inst(s, P.finial, cx, floor + H + ch + sh - 0.6, cz, 0, 1.6, 1.6, 1.6));
  } else {
    const dh = W / 2 + 1;
    east.push(inst(s, P.roofDome, cx, floor + H + ch, cz, 0, W / 2 + 0.8, dh, W / 2 + 0.8));
    east.push(inst(s, P.finial, cx, floor + H + ch + dh - 0.2, cz, 0, 1.6, 1.6, 1.6));
  }
  // the apse closing the nave's far end: its half round open to the nave, as tall, under a half cone
  const [ax, az] = at(x, z, yaw, 0, -L / 2);
  east.push(inst(s, P.bodyApse, ax, floor, az, yaw, W / 2, H, W / 2));
  east.push(inst(s, P.roofApse, ax, floor + H, az, yaw, W / 2 + 0.2, W * 0.5, W / 2 + 0.2));
  if (o.spires === 5) {
    for (const side of [-1, 1]) {
      const [tx, tz] = at(x, z, yaw, side * (W / 2 + 2.5), -L * 0.2);
      east.push(inst(s, P.bodyRound, tx, floor, tz, yaw, 2.6, H + 4, 2.6));
      east.push(inst(s, P.roofSpire, tx, floor + H + 4, tz, 0, 2.7, 18, 2.7));
      east.push(inst(s, P.finial, tx, floor + H + 21.6, tz, 0, 1, 1, 1));
    }
  }
  out.push(makeStructure(east));
  // Inside the nave (§8): galleries of perches at three heights down both walls, columns bearing the
  // lowest, the feather-mosaic floor of the crossing with the wind altar on it and Wardens by it,
  // rows of perches either side of the aisle facing it, chandeliers down the middle.
  const inside: PieceInst[] = [];
  const pin = (piece: number, u: number, v: number, y: number, turn: number, sx = 1, sy = 1, sz = 1) => {
    const [px, pz] = at(x, z, yaw, u, v);
    inside.push({ ...inst(s, piece, px, y, pz, yaw + turn, sx, sy, sz, PFLAG.indoor), detail: 0 });
  };
  // (the open hall's walls are 3 % of its half width thick)
  const inner = (W / 2) * 0.94;
  const v0 = -L / 2 + 3, v1 = L / 2 - 1;
  const g0 = 0.27 * H;
  for (const side of [-1, 1]) {
    for (const h of [0.27, 0.49, 0.71]) pin(P.gallery, side * inner, (v0 + v1) / 2, floor + h * H, -side * Math.PI / 2, (v1 - v0) / 2, 1, 2.2);
    for (let v = v0 + 3; v < v1 - 2; v += 6) pin(P.column, side * (inner - 1.9), v, floor, 0, 0.42, g0 - 0.35, 0.42);
  }
  const vc = -L * 0.2;
  pin(P.plaza, 0, vc, floor + 0.04, 0, W / 2 - 3.5, 0.05, W / 2 - 3.5);
  pin(P.altar, 0, vc, floor + 0.04, 0);
  for (const side of [-1, 1]) pin(P.warden, side * 4.2, vc + 2.6, floor, 0);
  for (let v = vc + 6.5; v < L / 2 - 5; v += 2.8) {
    for (const side of [-1, 1]) for (let u = 2.1; u + 1.0 < inner - 3.4; u += 2.15) pin(P.perch, side * u, v, floor, 0);
  }
  for (const v of [vc + 9, 0.5 * (vc + L / 2) + 5, L / 2 - 7]) pin(P.lanternHang, 0, v, floor + H - 2, 0, 2.5, 2.5, 2.5);
  out.push(makeStructure(inside));
  return out;
}

/**
 * A fountain plaza (§8): a round mosaic floor with a curb, a tiered fountain, lanterns round it,
 * banners at its rim. Its top lies on the highest ground under it (+0.25 m), its skirt down past the lowest.
 */
export function plaza(s: Site, x: number, z: number, R: number, banners = 4): Structure {
  const [lo, hi] = groundRange(s, x, z, R);
  const top = hi + 0.25;
  const pieces: PieceInst[] = [
    inst(s, P.plaza, x, top, z, 0, R, top - lo + 2.5, R),
    inst(s, P.curb, x, top, z, 0, R, 1, R),
    inst(s, P.fountain, x, top, z, s.rnd() * 6.28, 1, 1, 1),
  ];
  const nl = Math.max(4, Math.round(R / 3));
  for (let k = 0; k < nl; k++) {
    const a = (k / nl) * Math.PI * 2;
    pieces.push(inst(s, P.lanternPost, x + Math.cos(a) * (R - 1.4), top, z + Math.sin(a) * (R - 1.4), a, 1, 1, 1));
  }
  for (let k = 0; k < banners; k++) {
    const a = ((k + 0.5) / banners) * Math.PI * 2;
    pieces.push(inst(s, P.banner, x + Math.cos(a) * (R - 0.6), top, z + Math.sin(a) * (R - 0.6), Math.atan2(Math.cos(a), Math.sin(a)), 1, 1, 1));
  }
  return makeStructure(pieces);
}


/**
 * A rotunda (§8): columns in a ring under an entablature and a dome, on a round podium — the Moot's
 * hall (with a drum inside the colonnade: `hall`) or an open gazebo with a shrine-stone.
 */
export function rotunda(s: Site, x: number, z: number, r: number, h: number, o: { columns: number; hall?: boolean; onion?: boolean; measure?: boolean }): Structure {
  const [lo, hi] = groundRange(s, x, z, r + 3);
  const top = hi + 0.45;
  const pieces: PieceInst[] = [
    inst(s, P.plaza, x, top, z, 0, r + 2.6, top - lo + 2, r + 2.6),
    inst(s, P.curb, x, top, z, 0, r + 2.6, 1, r + 2.6),
  ];
  for (let k = 0; k < o.columns; k++) {
    const a = ((k + 0.5) / o.columns) * Math.PI * 2;
    pieces.push(inst(s, P.column, x + Math.cos(a) * r, top, z + Math.sin(a) * r, 0, 0.42 * Math.min(1.6, h / 6), h, 0.42 * Math.min(1.6, h / 6)));
  }
  pieces.push(inst(s, P.entablature, x, top + h, z, 0, r, 1, r));
  if (o.hall) {
    // its hall inside the colonnade, walked into, its walls rising open to the dome
    const rh = r - 2.4, face = s.rnd() * Math.PI * 2;
    pieces.push(inst(s, P.storeyRoundOpen, x, top, z, face, rh, 1, rh));
    pieces.push(inst(s, P.bodyRoundOpen, x, top + STOREY_H, z, face, rh, h + 1.6 - STOREY_H, rh, PFLAG.upper));
    pieces.push(inst(s, P.doorOpen, x + Math.sin(face) * (rh + 0.02), top, z + Math.cos(face) * (rh + 0.02), face, 1, 1, 1));
    furnishRotunda(s, pieces, x, z, face, 0.9 * rh, top, h + 1.6, !!o.measure);
  }
  const dr = r + 0.4, dh = o.onion ? r * 1.4 : r * 0.85;
  pieces.push(inst(s, o.onion ? P.roofOnion : P.roofDome, x, top + h + 1.55, z, 0, dr, dh, dr));
  pieces.push(inst(s, P.finial, x, top + h + 1.55 + dh - 0.15, z, 0, 1 + r / 12, 1 + r / 12, 1 + r / 12));
  if (!o.hall) pieces.push(inst(s, P.fountain, x, top, z, 0, Math.min(0.5, r / 8), Math.min(0.5, r / 8), Math.min(0.5, r / 8)));
  return makeStructure(pieces);
}

/** A watchtower (§8): slender, slit windows, a belfry under battlements, a beacon lantern on top. */
export function watchtower(s: Site, x: number, z: number, r: number, h: number, face: number): Structure {
  const [lo] = groundRange(s, x, z, r);
  const pieces: PieceInst[] = [
    inst(s, P.bodyRoundBlind, x, lo - 5, z, face, r * 1.08, 5.2, r * 1.08),
    inst(s, P.bodyRoundBelfry, x, lo, z, face, r, h, r),
    inst(s, P.parapet12, x, lo + h + 0.6, z, s.rnd() * 6.28, r, 1, r),
    inst(s, P.lanternPost, x, lo + h + 0.25, z, face, 1.6, 1.6, 1.6),
    inst(s, P.door, x + Math.sin(face) * r, lo, z + Math.cos(face) * r, face, 0.8, 0.8, 0.8),
  ];
  return makeStructure(pieces);
}

/** A Silent Spire (§4.4): the sky-burial spire at a rim point — tall, bare, a needle on top. */
export function silentSpire(s: Site, x: number, z: number): Structure {
  const [lo] = groundRange(s, x, z, 3);
  const h = 30 + 10 * s.rnd();
  const pieces: PieceInst[] = [
    inst(s, P.bodyRoundBlind, x, lo - 4, z, 0, 2.9, h + 4, 2.9),
    inst(s, P.cornice, x, lo + h - 0.05, z, 0, 2.9, 1, 2.9),
    inst(s, P.roofSpire, x, lo + h + 0.6, z, 0, 2.9, 20 + 6 * s.rnd(), 2.9),
  ];
  return makeStructure(pieces);
}

/** An archive spire (§8): a tall octagon round its open shaft, a needle spire, a cloister of columns at its foot. */
export function archiveSpire(s: Site, x: number, z: number, face: number): Structure[] {
  const r = 7 + 2 * s.rnd(), h = 34 + 10 * s.rnd();
  const [lo, hi] = groundRange(s, x, z, r + 6);
  const fl = hi + 0.12, top = Math.max(lo + h, fl + 24);
  // its octagon walked into, the shaft rising open to its roof
  const tower: PieceInst[] = [
    { ...inst(s, P.bodyRoundBlind, x, lo - 1, z, face, r, fl - lo + 1, r), detail: 1 },
    inst(s, P.storeyOctagon, x, fl, z, face, r, 1, r),
    inst(s, P.bodyOctagon, x, fl + STOREY_H, z, face, r, top - fl - STOREY_H, r, PFLAG.upper),
    inst(s, P.cornice, x, top - 0.05, z, 0, r * 0.94, 1, r * 0.94),
    inst(s, P.roofSpire, x, top + 0.6, z, face, r * 0.98, h * 0.9, r * 0.98),
    inst(s, P.finial, x, top + 0.6 + h * 0.9 - 0.4, z, 0, 1.4, 1.4, 1.4),
  ];
  const [dx, dz] = at(x, z, face, 0, r * Math.cos(Math.PI / 8) + 0.02);
  tower.push(inst(s, P.doorOpen, dx, fl, dz, face, 1, 1, 1));
  doorSteps(s, tower, dx, dz, face, fl, 1.15);
  for (let k = 0; k < 8; k++) {
    const a = face + (k / 8) * Math.PI * 2 + Math.PI / 8;
    const px = x + Math.sin(a) * (r * 0.93 + 0.8), pz = z + Math.cos(a) * (r * 0.93 + 0.8);
    tower.push(inst(s, P.pinnacle, px, top, pz, a, 1, 1, 1));
  }
  // inside (§8): the flightless's ramp winding up the shaft, turn on turn; the cord racks on its walls
  // between the turns (reached on the wing); reading perches and a table on the floor
  const R = 0.9 * r * Math.cos(Math.PI / 8);
  const inside: PieceInst[] = [];
  const add = (piece: number, px: number, py: number, pz: number, yaw: number, a: number, b: number, c: number) =>
    inside.push({ ...inst(s, piece, px, py, pz, yaw, a, b, c, PFLAG.indoor), detail: 0 });
  const turns = Math.floor((top - fl - 3) / (STOREY_H - 0.25));
  for (let k = 0; k < turns; k++) add(P.rampSlab, x, fl + k * (STOREY_H - 0.25), z, face - k * RAMP_SWEEP, R, 1, R);
  // (the ramp passes a bearing ψ every 2π / RAMP_SWEEP turns' height: racks fit between the passes)
  const pitch = ((STOREY_H - 0.25) * 2 * Math.PI) / RAMP_SWEEP;
  for (let f = 0; f < 8; f++) {
    if (f === 1) continue; // (the door's face)
    const psi = Math.PI / 8 + (f + 0.5) * (Math.PI / 4);
    let ya = psi;
    while (ya < RAMP_START) ya += 2 * Math.PI;
    for (let pass = fl + ((ya - RAMP_START) / RAMP_SWEEP) * (STOREY_H - 0.25); pass + 2.4 + 2.3 < top - 1; pass += pitch) {
      const [px, pz] = at(x, z, face, Math.cos(psi) * (R - 0.25), Math.sin(psi) * (R - 0.25));
      add(P.shelves, px, pass + 2.4, pz, face + Math.atan2(-Math.cos(psi), -Math.sin(psi)), 1, 1, 1);
    }
  }
  const free = R - 1.1;
  for (const [u, v, turn] of [[0, -0.35 * free, 0], [-0.45 * free, 0.2 * free, Math.PI / 3], [0.45 * free, 0.2 * free, -Math.PI / 3]] as const) {
    const [px, pz] = at(x, z, face, u, v);
    add(P.perch, px, fl, pz, face + turn, 1, 1, 1);
  }
  { const [px, pz] = at(x, z, face, 0, 0.15 * free); add(P.table, px, fl, pz, face, 0.85, 0.85, 0.85); }
  { const [px, pz] = at(x, z, face, 0, 0); add(P.lanternHang, px, fl + 7, pz, 0, 2, 2, 2); }
  return [makeStructure(tower), makeStructure(inside), rotunda(s, x + Math.sin(face) * (r + 16), z + Math.cos(face) * (r + 16), 7, 5, { columns: 10 })];
}

/** A market hall (§8): an arcade open at the ground, a gabled roof, banners at its ends. */
export function marketHall(s: Site, x: number, z: number, yaw: number, len: number, w: number): Structure {
  const [lo, hi] = groundRange(s, x, z, Math.max(len, w) * 0.6);
  const fl = hi + 0.1;
  // open: its gable on a colonnade round a paved floor, walked into from any side
  const pieces: PieceInst[] = [
    inst(s, P.bodyBlind, x, lo - 4, z, yaw, w / 2 + 0.3, fl - lo + 4, len / 2 + 0.3),
    inst(s, P.roofGable, x, fl + 6.6, z, yaw + Math.PI / 2, len / 2, w * 0.5, w / 2 + 0.2),
  ];
  const nc = Math.max(2, Math.round(len / 4.2));
  for (const side of [-1, 1]) {
    for (let k = 0; k <= nc; k++) {
      const [cx, cz] = at(x, z, yaw, side * (w / 2 - 0.4), -len / 2 + 0.4 + (k / nc) * (len - 0.8));
      pieces.push(inst(s, P.column, cx, fl, cz, 0, 0.38, 6.6, 0.38));
    }
  }
  // two rows of stalls down its length facing the middle, goods in crates behind them, lamps hung
  const put = (piece: number, u: number, v: number, turn: number, k = 1, y = fl) => {
    const [px, pz] = at(x, z, yaw, u, v);
    pieces.push({ ...inst(s, piece, px, y, pz, yaw + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  const ns = Math.max(1, Math.floor((len - 4) / 4.5));
  for (const side of [-1, 1]) {
    for (let k = 0; k < ns; k++) {
      const v = -len / 2 + 2.5 + ((k + 0.5) / ns) * (len - 5);
      put(P.stall, side * (w / 2 - 2.6), v, side > 0 ? -Math.PI / 2 : Math.PI / 2);
      if (k % 2 === 0) put(P.cargo, side * (w / 2 - 1.1), v + 1.6, 0.4 * side);
    }
  }
  for (const v of [-len / 4, len / 4]) put(P.lanternHang, 0, v, 0, 1, fl + 6.2);
  for (const e of [-1, 1]) {
    const [bx, bz] = at(x, z, yaw, w / 2 + 1.2, e * (len / 2 - 2));
    pieces.push(inst(s, P.banner, bx, fl, bz, yaw + Math.PI / 2, 1, 1, 1));
  }
  return makeStructure(pieces);
}

/** A greenhouse (§8): a low drum under a glass dome, warm crops for cold isles. */
export function greenhouse(s: Site, x: number, z: number, r: number): Structure {
  const [lo, hi] = groundRange(s, x, z, r);
  const fl = hi + 0.12, face = Math.PI / 2;
  const pieces: PieceInst[] = [
    { ...inst(s, P.bodyRoundBlind, x, lo - 1, z, face, r * 1.03, fl - lo + 1, r * 1.03), detail: 1 },
    inst(s, P.storeyRoundOpen, x, fl, z, face, r, 1, r),
    inst(s, P.cornice, x, fl + STOREY_H - 0.3, z, 0, r, 0.6, r),
    inst(s, P.roofGlass, x, fl + STOREY_H, z, 0, r * 1.02, r * 0.9, r * 1.02),
  ];
  const [dx, dz] = [x + Math.sin(face) * (r + 0.02), z + Math.cos(face) * (r + 0.02)];
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, face, 1, 1, 1));
  doorSteps(s, pieces, dx, dz, face, fl, 1.15);
  // inside (§8): beds round a little pool, a potting table by the door
  const R = 0.9 * r;
  const nb = Math.max(3, Math.min(8, Math.floor((2 * Math.PI * 0.6 * R) / 3.2)));
  for (let k = 0; k < nb; k++) {
    const a = Math.PI / 2 + ((k + 0.5) / nb) * 2 * Math.PI;
    if (Math.abs(((a - Math.PI / 2 + Math.PI) % (2 * Math.PI)) - Math.PI) < 0.5) continue; // (clear of the door)
    const [px, pz] = at(x, z, face, Math.cos(a) * 0.62 * R, Math.sin(a) * 0.62 * R);
    pieces.push({ ...inst(s, P.planter, px, fl, pz, face + Math.atan2(-Math.cos(a), -Math.sin(a)), 1, 1, 1, PFLAG.indoor), detail: 0 });
  }
  pieces.push({ ...inst(s, P.fountain, x, fl + 0.03, z, 0, Math.min(0.45, R / 9), Math.min(0.45, R / 9), Math.min(0.45, R / 9), PFLAG.indoor), detail: 0 });
  return makeStructure(pieces);
}

/**
 * A wind wall (§8) along points round a rim: crenellated wall runs between round turrets, as
 * structures a run or two each. `pts` world xz along the wall, in order.
 */
export function windWall(s: Site, pts: ReadonlyArray<readonly [number, number]>, h: number): Structure[] {
  const out: Structure[] = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const [x0, z0] = pts[i], [x1, z1] = pts[i + 1];
    const len = Math.hypot(x1 - x0, z1 - z0);
    if (len < 4) continue;
    const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
    const yaw = Math.atan2(-(z1 - z0), x1 - x0);
    const [lo, hi] = groundRange(s, mx, mz, len / 2);
    const pieces: PieceInst[] = [inst(s, P.bodyWall, mx, lo - 3, mz, yaw, len / 2, hi - lo + h + 3, 1.4)];
    const nc = Math.max(1, Math.floor(len / 4));
    for (let k = 0; k < nc; k++) {
      const f = (k + 0.5) / nc;
      pieces.push({ ...inst(s, P.crenel, x0 + (x1 - x0) * f, hi + h, z0 + (z1 - z0) * f, yaw, len / nc / 4, 1, 1.4 / 0.45 * 0.5), detail: 1 });
    }
    out.push(makeStructure(pieces));
    // a turret at the run's start
    const [tl] = groundRange(s, x0, z0, 4);
    out.push(makeStructure([
      inst(s, P.bodyRound, x0, tl - 3, z0, yaw, 3.6, hi - tl + h + 7, 3.6, PFLAG.upper),
      inst(s, P.parapet12, x0, hi + h + 4.6, z0, s.rnd() * 6.28, 3.6, 1, 3.6),
      inst(s, P.roofCone, x0, hi + h + 5.4, z0, s.rnd() * 6.28, 3.9, 8 + 3 * s.rnd(), 3.9),
    ]));
  }
  return out;
}

type HouseOpts = {
  face: number;      // yaw its front (and door) faces
  len: number;       // along its ridge (m)
  wid: number;       // across, front to back (m)
  floors: number;    // 3.6 m each
  tower: -1 | 0 | 1; // a round stair-tower at its left or right front corner, or none
  square?: boolean;  // a square stair-tower (else round)
};

/**
 * A hall-house (§8): a brood's long house — gabled, steep-roofed, two or three floors over a plinth
 * levelled into the slope — with a stair-tower at a front corner rising above its ridge (the
 * flightless climb it; the brood launch from its loft). Its front faces the street.
 */
export function hallHouse(s: Site, x: number, z: number, o: HouseOpts): Structure {
  const yaw = o.face;
  const [lo, hi] = groundRange(s, x, z, Math.max(o.len, o.wid) * 0.55);
  const floor = hi + 0.15;
  const H = Math.max(o.floors * 3.6 + 0.4, STOREY_H);
  // its hall on the ground floor (walked into by the door in the middle of its front), the floors over it
  const pieces: PieceInst[] = [
    { ...inst(s, P.bodyBlind, x, lo - 3, z, yaw, o.len / 2 + 0.25, floor - lo + 3, o.wid / 2 + 0.25), detail: 1 },
    inst(s, P.storeySquare, x, floor, z, yaw, o.len / 2, 1, o.wid / 2),
  ];
  if (H - STOREY_H > 0.5) pieces.push(inst(s, P.bodySquare, x, floor + STOREY_H, z, yaw, o.len / 2, H - STOREY_H, o.wid / 2, PFLAG.upper));
  const roofH = o.wid * (0.55 + 0.25 * s.rnd());
  pieces.push(inst(s, P.roofGable, x, floor + H, z, yaw, o.len / 2, roofH, o.wid / 2 + 0.25));
  const [dx, dz] = at(x, z, yaw, 0, o.wid / 2 + 0.02);
  pieces.push(inst(s, P.doorOpen, dx, floor, dz, yaw, 1, 1, 1));
  doorSteps(s, pieces, dx, dz, yaw, floor, 1.15);
  furnishHall(s, pieces, x, z, yaw, 0.9 * (o.len / 2), 0.9 * (o.wid / 2), floor);
  if (o.tower !== 0) {
    const tr = o.square ? 2.6 : 2.8 + 0.6 * s.rnd();
    const [tx, tz] = at(x, z, yaw, o.tower * (o.len / 2 + tr * 0.35), o.wid / 2 - tr * 0.2);
    const th = H + roofH + 3.6 + 3.6 * Math.floor(s.rnd() * 2);
    if (o.square) {
      pieces.push(inst(s, P.bodySquare, tx, lo - 2, tz, yaw, tr, floor - lo + 2 + th, tr, PFLAG.upper));
      const rh = tr * (2.4 + s.rnd());
      pieces.push(inst(s, P.roofPyramid, tx, floor + th, tz, yaw, tr, rh, tr));
      pieces.push(inst(s, P.finial, tx, floor + th + rh - 0.1, tz, 0, 0.8, 0.8, 0.8));
    } else {
      pieces.push(inst(s, P.bodyRound, tx, lo - 2, tz, yaw, tr, floor - lo + 2 + th, tr, PFLAG.upper));
      pieces.push(inst(s, P.cornice, tx, floor + th - 0.05, tz, 0, tr, 1, tr));
      const rh = tr * (2.6 + 1.2 * s.rnd());
      pieces.push(inst(s, P.roofCone, tx, floor + th + 0.6, tz, s.rnd() * 6.28, tr, rh, tr));
      pieces.push(inst(s, P.finial, tx, floor + th + 0.6 + rh - 0.1, tz, 0, 0.8, 0.8, 0.8));
      if (s.rnd() < 0.3) pieces.push(inst(s, P.pennant, tx, floor + th + 0.6 + rh + 1.4, tz, s.rnd() * 6.28, 0.8, 0.8, 0.8));
    }
  }
  const [lx, lz] = at(dx, dz, yaw, 1.6, 0.9);
  pieces.push(inst(s, P.lanternHang, lx, floor + 3.4, lz, yaw, 1, 1, 1));
  return makeStructure(pieces);
}

// ---- the classes' buildings (SKY_CIVILIZATION.md §7) ------------------------------------------------

/**
 * A wind-house (§7.5): the orchard keepers' tiered tower — a plastered and timbered storey per tier,
 * each under its own upswept roof, smaller as they rise; paper lanterns at the eaves' corners; a
 * gilded spire on top. Its seed jars and drying racks are inside.
 */
export function pagoda(s: Site, x: number, z: number, face: number, tiers: number, base: number): Structure {
  const [lo, hi] = groundRange(s, x, z, base * 1.4);
  const pieces: PieceInst[] = [inst(s, P.podium, x, hi + 0.6, z, face, base * 1.25, hi - lo + 2.6, base * 1.25, PFLAG.timber)];
  let y = hi + 0.6;
  const fl = y;
  for (let k = 0; k < tiers; k++) {
    const hw = base * (1 - 0.15 * k), bh = k === 0 ? 5.2 : 3.6, rh = hw * 0.62;
    if (k === 0) {
      // its ground tier walked into: the storey, its lattice carried on up to the first roof
      pieces.push(inst(s, P.storeyLattice, x, y, z, face, hw, 1, hw, PFLAG.plaster));
      pieces.push(inst(s, P.bodyLattice, x, y + STOREY_H, z, face, hw, bh - STOREY_H, hw, PFLAG.plaster | PFLAG.upper));
    } else {
      pieces.push(inst(s, P.bodyLattice, x, y, z, face, hw, bh, hw, PFLAG.plaster | PFLAG.upper));
    }
    pieces.push(inst(s, P.roofPagoda, x, y + bh, z, face, hw, rh, hw));
    for (let c = 0; c < 4; c++) {
      const a = face + Math.PI / 4 + (c / 4) * Math.PI * 2;
      const d = hw * 1.3 * Math.SQRT2 * 0.95;
      pieces.push(inst(s, P.paperLantern, x + Math.sin(a) * d, y + bh + rh * 0.12, z + Math.cos(a) * d, a, 1, 1, 1));
    }
    y += bh + rh * 0.42;
  }
  pieces.push(inst(s, P.finial, x, y + base * 0.2, z, 0, 1.6, 1.6, 1.6));
  const [dx, dz] = at(x, z, face, 0, base + 0.02);
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, face, 1, 1, 1, PFLAG.timber | PFLAG.lacquer));
  // inside (§8): the orchard's seed jars, its drying racks, the keepers' shrine at the back
  const R = 0.9 * base;
  const put = (piece: number, u: number, v: number, turn: number, k = 1, yy = fl) => {
    const [px, pz] = at(x, z, face, u, v);
    pieces.push({ ...inst(s, piece, px, yy, pz, face + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  put(P.rug, 0, 0, 0, Math.min(1.2, R / 2));
  put(P.shrineSmall, 0, -(R - 0.6), 0);
  for (const side of [-1, 1]) {
    put(P.shelves, side * (R - 0.25), -0.15 * R, -side * Math.PI / 2);
    put(P.jars, side * (R - 0.8), R - 1.1, side * 0.6);
  }
  put(P.lanternHang, 0, 0, 0, 1, fl + STOREY_H - 0.06);
  return makeStructure(pieces);
}

/** A tea pavilion (§7.5): four vermilion pillars under an upswept roof on a timber deck, lanterns at its corners. */
export function teaPavilion(s: Site, x: number, z: number, face: number, r: number): Structure {
  const [lo, hi] = groundRange(s, x, z, r + 1);
  const top = hi + 0.5;
  const pieces: PieceInst[] = [inst(s, P.podium, x, top, z, face, r + 0.8, top - lo + 1.5, r + 0.8, PFLAG.timber)];
  for (const [u, v] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    const [cx, cz] = at(x, z, face, u * r, v * r);
    pieces.push(inst(s, P.column, cx, top, cz, face, 0.26, 3.4, 0.26, PFLAG.lacquer));
    const [lx, lz] = at(x, z, face, u * r * 1.3, v * r * 1.3);
    pieces.push(inst(s, P.paperLantern, lx, top + 3.55, lz, face, 1, 1, 1));
  }
  pieces.push(inst(s, P.roofPagoda, x, top + 3.4, z, face, r * 1.05, r * 0.75, r * 1.05));
  return makeStructure(pieces);
}

/** A gate-arch over a path (§7.5), facing along it; `scale` 1 spans 8 m. */
export function torii(s: Site, x: number, z: number, face: number, scale = 1): Structure {
  // (its pillars stand either side of the path, across it: the path runs along the piece's z)
  return makeStructure([inst(s, P.torii, x, s.ground(x, z) - 0.1, z, face, scale, scale, scale)]);
}

/**
 * A dock (§7.10) from a rim point out over the air along bearing `out` (yaw-style: (sin, cos)):
 * a timber pier `len` m long on struts, mooring posts, a crane at its root, cargo, and a skyship
 * moored alongside.
 */
export function harbourDock(s: Site, x: number, z: number, out: number, len: number, ship: boolean): Structure[] {
  const ox = Math.sin(out), oz = Math.cos(out);
  const y = s.ground(x, z) + 0.2;
  const yaw = Math.atan2(-oz, ox); // the dock's local x outward
  const pieces: PieceInst[] = [inst(s, P.dock, x, y, z, yaw, len, Math.max(4, len * 0.12), 4.5)];
  const [cx, cz] = at(x, z, out, 5.5, 3);
  pieces.push(inst(s, P.crane, cx, s.ground(cx, cz), cz, out + Math.PI / 2 + Math.PI, 1, 1, 1));
  for (const f of [0.35, 0.7]) pieces.push(inst(s, P.cargo, x + ox * len * f + Math.cos(out) * 2, y, z + oz * len * f - Math.sin(out) * 2, s.rnd() * 6.28, 1, 1, 1));
  pieces.push(inst(s, P.lanternPost, x + ox * len - Math.cos(out) * 3.6, y, z + oz * len + Math.sin(out) * 3.6, out, 1, 1, 1));
  const out2: Structure[] = [makeStructure(pieces)];
  if (ship) {
    // moored alongside the pier's outer half, its bow outward, its deck at the pier's height
    const side = s.rnd() < 0.5 ? -1 : 1;
    const sx = x + ox * len * 0.62 + Math.cos(out) * side * 9.5, sz = z + oz * len * 0.62 - Math.sin(out) * side * 9.5;
    out2.push(makeStructure([inst(s, P.skyship, sx, y - 0.5, sz, out, 1, 1, 1)]));
  }
  return out2;
}

/** A market stall row (§8): stalls round a plaza's rim facing in, with cargo behind. */
export function stalls(s: Site, x: number, z: number, R: number, n: number): Structure {
  const pieces: PieceInst[] = [];
  for (let k = 0; k < n; k++) {
    const a = (k / n) * Math.PI * 2 + s.rnd() * 0.2;
    const px = x + Math.cos(a) * (R - 2.6), pz = z + Math.sin(a) * (R - 2.6);
    const face = Math.atan2(x - px, z - pz);
    pieces.push(inst(s, P.stall, px, s.ground(px, pz) + 0.25, pz, face, 1, 1, 1));
  }
  return makeStructure(pieces);
}

/** A wind-wheel tower (§7.11): a round tower whose sails, turned to the wind, pump the cisterns. */
export function windwheel(s: Site, x: number, z: number, face: number): Structure {
  const [lo] = groundRange(s, x, z, 4);
  const h = 15 + 4 * s.rnd();
  const pieces: PieceInst[] = [
    inst(s, P.bodyRound, x, lo - 2, z, face, 3.4, h + 2, 3.4, PFLAG.timber),
    inst(s, P.roofCone, x, lo + h + 0.5, z, face, 3.7, 6, 3.7),
    inst(s, P.windSails, x + Math.sin(face) * 4.4, lo + h - 2.5, z + Math.cos(face) * 4.4, face, 1, 1, 1),
    inst(s, P.door, x - Math.sin(face) * 3.4, lo, z - Math.cos(face) * 3.4, face + Math.PI, 0.8, 0.8, 0.8, PFLAG.timber),
  ];
  return makeStructure(pieces);
}

/** A cloud-comb (§7.11) at a rim point, facing out to the cloud sea. */
export function comb(s: Site, x: number, z: number, out: number): Structure {
  return makeStructure([inst(s, P.comb, x, s.ground(x, z) - 0.3, z, out + Math.PI / 2, 1, 1, 1)]);
}

/** A cistern house (§8): a low round tank-house under a dome, by its pool. */
export function cistern(s: Site, x: number, z: number, r: number): Structure {
  const [lo, hi] = groundRange(s, x, z, r);
  const fl = hi + 0.12, face = Math.PI / 2, top = fl + STOREY_H + 0.6;
  const pieces: PieceInst[] = [
    { ...inst(s, P.bodyRoundBlind, x, lo - 1, z, face, r * 1.03, fl - lo + 1, r * 1.03), detail: 1 },
    inst(s, P.storeyRoundOpen, x, fl, z, face, r, 1, r),
    inst(s, P.bodyRoundOpen, x, fl + STOREY_H, z, face, r, top - fl - STOREY_H, r, PFLAG.upper),
    inst(s, P.cornice, x, top, z, 0, r, 1, r),
    inst(s, P.roofDome, x, top + 0.6, z, 0, r * 1.02, r * 0.6, r * 1.02),
  ];
  const [dx, dz] = [x + Math.sin(face) * (r + 0.02), z + Math.cos(face) * (r + 0.02)];
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, face, 1, 1, 1));
  doorSteps(s, pieces, dx, dz, face, fl, 1.15);
  // inside (§8): the store sunk in the floor, its measuring rods standing in it, the keeper's shelves
  const R = 0.9 * r, pr = Math.max(1.2, 0.55 * R);
  const put = (piece: number, u: number, v: number, turn: number, k = 1, y = fl) => {
    const [px, pz] = at(x, z, face, u, v);
    pieces.push({ ...inst(s, piece, px, y, pz, face + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  pieces.push({ ...inst(s, P.pool, x, fl, z, face, pr, 1, pr, PFLAG.indoor), detail: 0 });
  put(P.rod, 0.45 * pr, 0.2 * pr, 0);
  put(P.rod, -0.5 * pr, -0.3 * pr, 0);
  put(P.shelves, 0, -(R - 0.25), 0);
  put(P.lanternHang, 0, 0.6 * R, 0, 1, fl + STOREY_H - 0.06);
  return makeStructure(pieces);
}

/** A forge (§7.7): a long hall of basalt, its arcade glowing with the furnaces, two or three chimneys smoking. */
export function forge(s: Site, x: number, z: number, face: number): Structure {
  const len = 22 + 8 * s.rnd(), wid = 13 + 3 * s.rnd();
  const [lo, hi] = groundRange(s, x, z, len * 0.55);
  const fl = hi + 0.12;
  // the hall walked into by its door, its arcade of furnace-light above, the gable over all
  const pieces: PieceInst[] = [
    inst(s, P.bodyBlind, x, lo - 3, z, face, len / 2 + 0.3, fl - lo + 3, wid / 2 + 0.3),
    inst(s, P.storeySquare, x, fl, z, face, len / 2, 1, wid / 2),
    inst(s, P.bodyArcade, x, fl + STOREY_H, z, face, len / 2, 9 - STOREY_H, wid / 2, PFLAG.upper),
    inst(s, P.roofGable, x, fl + 9, z, face, len / 2, wid * 0.45, wid / 2 + 0.3),
  ];
  const fx: Effect[] = [];
  const put = (piece: number, u: number, v: number, turn: number, k = 1, y = fl) => {
    const [px, pz] = at(x, z, face, u, v);
    pieces.push({ ...inst(s, piece, px, y, pz, face + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  const nc = 2 + Math.floor(s.rnd() * 2);
  for (let k = 0; k < nc; k++) {
    const u = -len / 2 + ((k + 0.5) / nc) * len, v = -wid * 0.2;
    const [cx, cz] = at(x, z, face, u, v);
    const ch = 22 + 10 * s.rnd();
    // each chimney rising from its furnace's crown; an anvil before the furnace's mouth
    pieces.push(inst(s, P.chimney, cx, fl + 3.2, cz, face, 1.0, ch - 3.2, 1.0));
    put(P.furnace, u, v, 0);
    put(P.anvil, u + 0.4, v + 2.6, Math.PI / 2);
    fx.push({ k: 'smoke', x: cx, y: fl + ch + 0.5, z: cz, r: 3 + s.rnd() * 2 });
  }
  // the workbench along the front, crates of keel-iron, tools racked on the end walls, a lamp
  const hx = 0.9 * (len / 2), hz = 0.9 * (wid / 2);
  put(P.tableLong, -hx * 0.35, hz - 1.4, 0);
  put(P.cargo, hx - 1.6, hz - 1.6, 0.3);
  put(P.cargo, hx - 1.4, hz - 4.0, -0.2);
  for (const side of [-1, 1]) put(P.shelves, side * (hx - 0.25), 0, -side * Math.PI / 2);
  put(P.lanternHang, 0, hz * 0.4, 0, 1, fl + STOREY_H - 0.06);
  const [dx, dz] = at(x, z, face, 0, wid / 2 + 0.02);
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, face, 1, 1, 1));
  doorSteps(s, pieces, dx, dz, face, fl, 1.15);
  for (const u of [-1, 1]) {
    const [bx, bz] = at(x, z, face, u * 3.5, wid / 2 + 2.5);
    pieces.push(inst(s, P.brazier, bx, s.ground(bx, bz), bz, 0, 1, 1, 1));
  }
  const st = makeStructure(pieces);
  st.fx = fx;
  return st;
}

/** A keep (§7.8): a great square tower with a belfry, battlements on corbels, a corner turret at each angle, a pyramid roof. */
export function keep(s: Site, x: number, z: number, face: number, hw: number, h: number): Structure {
  const [lo, hi] = groundRange(s, x, z, hw * 1.2);
  const fl = hi + 0.12, top = Math.max(lo + h, fl + STOREY_H + 6);
  // its garrison hall on the ground floor, walked into; the tower over it
  const pieces: PieceInst[] = [
    { ...inst(s, P.bodyBlind, x, lo - 3, z, face, hw * 1.03, fl - lo + 3, hw * 1.03), detail: 1 },
    inst(s, P.storeySquare, x, fl, z, face, hw, 1, hw),
    inst(s, P.bodyBelfry, x, fl + STOREY_H, z, face, hw, top - fl - STOREY_H, hw, PFLAG.upper),
    inst(s, P.roofPyramid, x, top, z, face, hw * 1.02, hw * 1.6, hw * 1.02),
  ];
  for (let c = 0; c < 4; c++) {
    const a = face + Math.PI / 4 + (c / 4) * Math.PI * 2;
    const d = hw * Math.SQRT2;
    pieces.push(inst(s, P.turret, x + Math.sin(a) * d, top - 2.5, z + Math.cos(a) * d, a, 1.2, 1.2, 1.2));
    const [bx, bz] = [x + Math.sin(a) * d * 0.7, z + Math.cos(a) * d * 0.7];
    pieces.push(inst(s, P.brazier, bx, top + 0.1, bz, 0, 1, 1, 1));
  }
  const [dx, dz] = at(x, z, face, 0, hw + 0.02);
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, face, 1.3, 1.3, 1.3));
  doorSteps(s, pieces, dx, dz, face, fl, 1.3);
  furnishHall(s, pieces, x, z, face, 0.9 * hw, 0.9 * hw, fl);
  return makeStructure(pieces);
}

/**
 * The Great Halo (§7.3): the ring of stone, `radius` m, standing upright on its isle (its foot a
 * little buried), buttressed at its feet; runestones floating round it.
 */
export function haloMonument(s: Site, x: number, z: number, yaw: number, radius: number): Structure[] {
  const k = radius / 100;
  const [lo] = groundRange(s, x, z, radius * 0.4);
  const base = lo - 0.12 * radius;
  const ring = makeStructure([inst(s, P.haloRing, x, base, z, yaw, k, k, k)]);
  const feet: PieceInst[] = [];
  const ux = Math.cos(yaw), uz = -Math.sin(yaw); // the ring's plane, along it
  const nx = Math.sin(yaw), nz = Math.cos(yaw);  // across it
  for (const along of [-1, 1]) for (const side of [-1, 1]) {
    const px = x + ux * along * radius * 0.3 + nx * side * 5.5 * k, pz = z + uz * along * radius * 0.3 + nz * side * 5.5 * k;
    feet.push(inst(s, P.buttress, px, s.ground(px, pz) - 1, pz, side > 0 ? yaw : yaw + Math.PI, 2.2 * k, 0.32 * radius, 2.2 * k));
  }
  const stones: PieceInst[] = [];
  for (let j = 0; j < 10; j++) {
    const a = (j / 10) * Math.PI * 2 + s.rnd() * 0.3;
    const d = radius * (0.25 + 0.5 * s.rnd());
    const r = radius * (1.15 + 0.2 * s.rnd());
    // round the ring in its own plane and a little out of it
    const px = x + ux * Math.cos(a) * r + nx * (s.rnd() - 0.5) * 40, pz = z + uz * Math.cos(a) * r + nz * (s.rnd() - 0.5) * 40;
    stones.push(inst(s, P.runestoneFloat, px, base + radius + Math.sin(a) * r * 0.9 + d * 0.1, pz, s.rnd() * 6.28, 1.6 + s.rnd() * 1.4, 1.6, 1.6));
  }
  return [ring, makeStructure(feet), makeStructure(stones)];
}

/** The Vaelspire (§7.4): the beacon's crystal cluster on a summit, and the Thread of light it sends up. */
export function vaelspire(s: Site, x: number, z: number, height: number): Structure {
  const [lo] = groundRange(s, x, z, height * 0.25);
  const st = makeStructure([inst(s, P.crystal, x, lo - 0.06 * height, z, s.rnd() * 6.28, height, height, height)]);
  st.fx = [{ k: 'beam', x, y: lo + 0.86 * height, z, r: 0.9 + 0.012 * height }];
  return st;
}

/** An observatory (§8): a round tower under a bronze dome, its star-court beside it. */
export function observatory(s: Site, x: number, z: number, face: number): Structure {
  const r = 6 + 2 * s.rnd();
  const [lo, hi] = groundRange(s, x, z, r);
  const fl = hi + 0.12, h = Math.max(11 + 5 * s.rnd(), fl - lo + STOREY_H + 4);
  const pieces: PieceInst[] = [
    { ...inst(s, P.bodyRoundBlind, x, lo - 2, z, face, r * 1.04, fl - lo + 2, r * 1.04), detail: 1 },
    inst(s, P.storeyRoundOpen, x, fl, z, face, r, 1, r),
    inst(s, P.bodyRoundOpen, x, fl + STOREY_H, z, face, r, lo + h - fl - STOREY_H, r, PFLAG.upper),
    inst(s, P.cornice, x, lo + h - 0.05, z, 0, r, 1, r),
    inst(s, P.roofDome, x, lo + h + 0.6, z, face, r * 1.02, r * 0.85, r * 1.02),
    inst(s, P.finial, x, lo + h + 0.6 + r * 0.85 - 0.1, z, 0, 1, 1, 1),
  ];
  const [dx, dz] = [x + Math.sin(face) * (r + 0.02), z + Math.cos(face) * (r + 0.02)];
  pieces.push(inst(s, P.doorOpen, dx, fl, dz, face, 1, 1, 1));
  doorSteps(s, pieces, dx, dz, face, fl, 1.15);
  // inside (§8): the orrery in the middle, the telescope raised to the dome, a chart table, the charts' shelves
  const R = 0.9 * r;
  const put = (piece: number, u: number, v: number, turn: number, k = 1, y = fl) => {
    const [px, pz] = at(x, z, face, u, v);
    pieces.push({ ...inst(s, piece, px, y, pz, face + turn, k, k, k, PFLAG.indoor), detail: 0 });
  };
  put(P.rug, 0, 0, 0, Math.min(1.4, R / 2.4));
  put(P.orrery, 0, 0, 0);
  put(P.telescope, 0, -0.55 * R, Math.PI);
  put(P.table, 0.55 * R, 0.1 * R, 0, 0.9);
  for (const side of [-1, 1]) put(P.shelves, side * (R - 0.3), -0.25 * R, -side * Math.PI / 2);
  put(P.lanternHang, -0.4 * R, 0.3 * R, 0, 1, fl + STOREY_H + 1.5);
  return makeStructure(pieces);
}

/** A wayshrine (§7.12): a small paved round, a gate-arch, a Warden, a runestone, a lantern. */
export function shrine(s: Site, x: number, z: number, face: number): Structure {
  const [lo, hi] = groundRange(s, x, z, 6);
  const top = hi + 0.2;
  const [gx, gz] = at(x, z, face, 0, 5.5), [wx, wz] = at(x, z, face, -2.2, -2), [rx, rz] = at(x, z, face, 2.2, -2.2);
  return makeStructure([
    inst(s, P.plaza, x, top, z, 0, 6, top - lo + 1.5, 6),
    inst(s, P.gate, gx, top, gz, face, 0.62, 0.62, 0.62),
    inst(s, P.warden, wx, top, wz, face, 0.75, 0.75, 0.75),
    inst(s, P.runestone, rx, top, rz, s.rnd() * 6.28, 0.6, 0.6, 0.6),
    inst(s, P.lanternPost, x, top, z, face, 1, 1, 1),
  ]);
}

/**
 * The Elder Bough (§7.6): the colossal tree on its isle, its roots hanging under the isle, nest
 * platforms in its limbs, paper lanterns hung through it. `scale` 1 is 200 m.
 */
export function elderBough(s: Site, isl: { cx: number; cz: number; R: number; y: number; cliff: number; depth: number }, scale: number): Structure[] {
  const x = isl.cx, z = isl.cz;
  const [lo] = groundRange(s, x, z, 18 * scale);
  const tree = makeStructure([inst(s, P.boughTree, x, lo, z, s.rnd() * 6.28, scale, scale, scale)]);
  const roots = makeStructure([inst(s, P.boughRoots, x, isl.y - isl.cliff - 0.3 * isl.depth, z, s.rnd() * 6.28, isl.R * 0.8, isl.depth * 1.1 + 60, isl.R * 0.8)]);
  const nests: PieceInst[] = [];
  for (let k = 0; k < 9; k++) {
    const a = (k / 9) * Math.PI * 2 + s.rnd() * 0.4;
    const d = (18 + 26 * s.rnd()) * scale, y = lo + (45 + 60 * s.rnd()) * scale;
    nests.push(inst(s, P.nest, x + Math.cos(a) * d, y, z + Math.sin(a) * d, a, 1, 1, 1));
    nests.push(inst(s, P.paperLantern, x + Math.cos(a + 0.2) * d * 1.2, y + 6, z + Math.sin(a + 0.2) * d * 1.2, a, 1.4, 1.4, 1.4));
  }
  return [tree, roots, makeStructure(nests)];
}

/** A roost the Long Gale broke (§3.5): its body cut short and jagged, no roof, rubble at its foot; dark. */
export function ruinedRoost(s: Site, x: number, z: number, r: number, face: number): Structure {
  const [lo, hi] = groundRange(s, x, z, r);
  const h = (2 + Math.floor(s.rnd() * 4)) * 3.6 + (hi - lo);
  const f = PFLAG.ruined | PFLAG.dark;
  const pieces: PieceInst[] = [
    { ...inst(s, P.bodyRoundBlind, x, lo - 4, z, face, r * 1.06, 4.1, r * 1.06, f), detail: 1 },
    inst(s, P.bodyRound, x, lo, z, face, r, h, r, f),
    inst(s, P.ruinTop, x, lo + h - 0.05, z, s.rnd() * 6.28, r, 1, r, f),
  ];
  const [rx, rz] = at(x, z, face + 1, r + 2.5, 0);
  pieces.push(inst(s, P.rubble, rx, s.ground(rx, rz) - 0.3, rz, s.rnd() * 6.28, 0.8 + 0.5 * s.rnd(), 0.8, 0.8));
  return makeStructure(pieces);
}

/** A hall the Long Gale unroofed: its walls standing to their windows, gable-less, fallen columns and rubble about it. */
export function ruinedHall(s: Site, x: number, z: number, face: number, len: number, wid: number): Structure {
  const [lo, hi] = groundRange(s, x, z, len * 0.55);
  const f = PFLAG.ruined | PFLAG.dark;
  const pieces: PieceInst[] = [
    inst(s, P.bodyHall, x, lo - 1, z, face, wid / 2, hi - lo + 9 + 6 * s.rnd(), len / 2, f),
  ];
  for (let k = 0; k < 3; k++) {
    const [cx, cz] = at(x, z, face, (s.rnd() - 0.5) * (wid + 10), (s.rnd() - 0.5) * (len + 10));
    pieces.push(inst(s, s.rnd() < 0.5 ? P.columnFallen : P.rubble, cx, s.ground(cx, cz), cz, s.rnd() * 6.28, 1, 1, 1, f));
  }
  return makeStructure(pieces);
}

/**
 * A bridge the Long Gale broke (§7.9): its first `spans` spans standing out from a rim along bearing
 * `out`, its piers still hanging, the last span ending in the air.
 */
export function brokenBridge(s: Site, x: number, z: number, out: number, spans: number, width: number): Structure[] {
  const ox = Math.sin(out), oz = Math.cos(out);
  const yaw = Math.atan2(-oz, ox);
  const y = s.ground(x, z) + 0.15;
  const S = 22, sy = 0.75 * S;
  const f = PFLAG.ruined | PFLAG.dark;
  const outS: Structure[] = [];
  for (let k = 0; k < spans; k++) {
    const t = (k + 0.5) * S;
    const sx = x + ox * t, sz = z + oz * t;
    const pieces: PieceInst[] = [inst(s, P.spanPointed, sx, y, sz, yaw, S / 2, sy, width / 2, f)];
    if (k > 0) pieces.push(inst(s, P.pier, x + ox * k * S, y - 0.62 * sy, z + oz * k * S, yaw, 0.11 * S, S * (0.6 + 0.6 * s.rnd()), width / 2, f));
    for (const side of [-1, 1]) if (s.rnd() < 0.7) pieces.push(inst(s, P.parapet, sx - oz * side * (width / 2 - 0.25), y, sz + ox * side * (width / 2 - 0.25), yaw, S / 2 * (0.4 + 0.6 * s.rnd()), 1.1, 0.24, f));
    outS.push(makeStructure(pieces));
  }
  return outS;
}
