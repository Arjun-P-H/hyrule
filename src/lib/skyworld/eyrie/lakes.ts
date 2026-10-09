/**
 * Life in an Eyrie's lakes and lagoons (SKY_CIVILIZATION.md §9): drifts of
 * lily pads out from the shore with the lotus flowering among them, reeds
 * along the shallow margins, waterweed on the bed wherever the water is deep
 * enough to hold it, and in the larger lakes a few schools of the koi the
 * Aelari keep. The kit's water pieces (kit/water.ts), placed as structures
 * (a drift, a stretch of reeds, a bed of weed, a school apiece).
 */

import { lakeDist, topAt, type Lake, type SkyIsland } from '../island';
import { pieceId } from '../kit/pieces';
import { makeStructure, type PieceInst, type Structure } from '../structures';

const P = { pad: pieceId('lily.pad'), lotus: pieceId('lotus'), reeds: pieceId('reeds'), weed: pieceId('waterweed'), koi: pieceId('koi') };
const TAU = Math.PI * 2;
/** How far a koi swims from its place (m; structures.wgsl.ts ANIM.swim: a circle up to 6 m in radius). */
const KOI_REACH = 6.5;

/** Island-local point at normalised distance d from a lake's middle, at angle a in its own (stretched) frame. */
function lakePoint(L: Lake, a: number, d: number): [number, number] {
  const w = 1 + 0.12 * Math.sin(3 * a + L.b0) + 0.07 * Math.sin(5 * a + L.b1);
  const u = Math.cos(a) * L.r * w * d, v = Math.sin(a) * L.r * w * d * L.aspect;
  return [L.x + u * L.cos - v * L.sin, L.z + u * L.sin + v * L.cos];
}

/** An island's lake life into `out` (a generator: a slice per few structures). Frozen lakes have none. */
export function* lakeLife(isl: SkyIsland, rnd: () => number, palette: number, out: Structure[]): Generator<void, void, void> {
  const L = isl.lake;
  if (!L || L.frozen) return;
  const inst = (piece: number, x: number, y: number, z: number, k: number): PieceInst =>
    ({ piece, x, y, z, yaw: rnd() * TAU, sx: k, sy: k, sz: k, palette, seed: Math.floor(rnd() * 65536), flags: 0 });
  /** The water's depth at island-local (lx, lz) (negative on dry land). */
  const depth = (lx: number, lz: number) => L.level - topAt(isl, isl.cx + lx, isl.cz + lz);
  const perim = Math.PI * L.r * (1 + L.aspect);

  // Drifts of lily pads, a lotus here and there among them.
  const drifts = Math.min(60, Math.round(perim / 16));
  for (let k = 0; k < drifts; k++) {
    if (rnd() > 0.65) continue;
    const [cx, cz] = lakePoint(L, rnd() * TAU, 0.6 + 0.32 * rnd());
    const pieces: PieceInst[] = [];
    const n = 5 + Math.floor(rnd() * 10), spread = 2 + 3 * rnd();
    for (let j = 0; j < n; j++) {
      const a = rnd() * TAU, d = Math.sqrt(rnd()) * spread;
      const lx = cx + Math.cos(a) * d, lz = cz + Math.sin(a) * d;
      if (lakeDist(L, lx, lz) > 0.96 || depth(lx, lz) < 0.25) continue;
      const k2 = 0.6 + 0.8 * rnd();
      pieces.push(inst(P.pad, isl.cx + lx, L.level + 0.004, isl.cz + lz, k2));
      // (the lotus a little larger than life, as the Aelari breed it)
      if (rnd() < 0.22) pieces.push(inst(P.lotus, isl.cx + lx, L.level + 0.02, isl.cz + lz, 1.6 + 0.6 * rnd()));
    }
    if (pieces.length) out.push(makeStructure(pieces));
    if (k % 8 === 7) yield;
  }

  // Reeds along the shallow margins (not up the bank, not out in deep water), six clumps a stretch.
  const clumps = Math.min(220, Math.round(perim / 3.5));
  let run: PieceInst[] = [];
  for (let k = 0; k < clumps; k++) {
    if (rnd() < 0.35) continue;
    const [lx, lz] = lakePoint(L, ((k + rnd()) / clumps) * TAU, 0.96 + 0.09 * rnd());
    const dep = depth(lx, lz);
    if (dep < -0.5 || dep > 0.9) continue;
    run.push(inst(P.reeds, isl.cx + lx, L.level - dep - 0.05, isl.cz + lz, 0.8 + 0.5 * rnd()));
    if (run.length === 6) { out.push(makeStructure(run)); run = []; }
    if (k % 30 === 29) yield;
  }
  if (run.length) out.push(makeStructure(run));
  yield;

  // Waterweed on the bed: beds of three or four clumps where it is deep enough, never breaking the surface.
  const beds = Math.min(40, Math.round((Math.PI * L.r * L.r * L.aspect) / 600));
  for (let k = 0; k < beds; k++) {
    const [lx, lz] = lakePoint(L, rnd() * TAU, 0.25 + 0.65 * Math.sqrt(rnd()));
    const pieces: PieceInst[] = [];
    for (let j = 0; j < 4; j++) {
      const qx = lx + (rnd() - 0.5) * 3, qz = lz + (rnd() - 0.5) * 3;
      const dep = depth(qx, qz);
      if (dep < 0.9 || dep > 7) continue;
      pieces.push(inst(P.weed, isl.cx + qx, L.level - dep - 0.05, isl.cz + qz, Math.min(1.3, (dep - 0.3) / 2.2)));
    }
    if (pieces.length) out.push(makeStructure(pieces));
    if (k % 10 === 9) yield;
  }

  // Koi, a few schools in a lake large enough for their circles, each a little under the surface.
  if (L.r >= 25) {
    const schools = Math.min(4, 1 + Math.floor(L.r / 45));
    for (let k = 0; k < schools; k++) {
      const [lx, lz] = lakePoint(L, rnd() * TAU, 0.2 + 0.35 * rnd());
      if (depth(lx, lz) < 1.4) continue;
      const pieces: PieceInst[] = [];
      const n = 2 + Math.floor(rnd() * 4);
      for (let j = 0; j < n; j++) {
        pieces.push(inst(P.koi, isl.cx + lx + (rnd() - 0.5) * 3, L.level - 0.4 - 0.3 * rnd(), isl.cz + lz + (rnd() - 0.5) * 3, 1.1 + 0.6 * rnd()));
      }
      const st = makeStructure(pieces);
      st.r += KOI_REACH;
      out.push(st);
    }
  }
}
