/**
 * Vow-bridges (SKY_CIVILIZATION.md §6.7): the built bridges that bind an
 * Eyrie's islands, raised along their axis (archipelago.ts makeBridge) as
 * structures — one per span, so each is culled and detailed on its own.
 * ---------------------------------------------------------------------------
 * A bridge is a row of equal arched spans (bridge.span: round arches, pointed
 * in the gothic Eyries) whose piers cannot stand on ground: each hangs below
 * the springing and tapers into a plumb-stone of skystone, glowing faintly at
 * night. Each span's deck is level at its own height; where two spans' decks
 * differ, a short flight of steps joins them (a stair-bridge climbs that way).
 * Parapets run along both edges, lanterns stand on every second pier, and at
 * each end a Warden on either side and lanterns — on the Crown's bridges a
 * gate-arch over the deck too. In the water-rich Eyries a level bridge often
 * carries an aqueduct inside one parapet, from a cistern head at its higher
 * end to a spout pouring into a basin at its lower.
 */

import { pieceId } from '../kit/pieces';
import { makeStructure, PFLAG, type PieceInst, type Structure } from '../structures';
import type { Arch } from '../arches';

const P = {
  span: pieceId('bridge.span'), spanPointed: pieceId('bridge.spanPointed'), pier: pieceId('bridge.pier'), parapet: pieceId('parapet'),
  lantern: pieceId('lantern.post'), warden: pieceId('warden'), gate: pieceId('gate.arch'), stair6: pieceId('stair.6'), stair12: pieceId('stair.12'),
  rope: pieceId('bridge.rope'), torii: pieceId('gate.torii'), paper: pieceId('lantern.paper'),
  aqueduct: pieceId('aqueduct'), aqFall: pieceId('aqueduct.fall'), spout: pieceId('spout.fall'), fountain: pieceId('fountain'),
  tank: pieceId('body.roundBlind'), dome: pieceId('roof.dome'),
};

export type BridgeStyle = {
  palette: number; pointed: boolean; span: readonly [number, number]; gate: boolean; hang: readonly [number, number];
  /** Petalfall's: timber spans with vermilion rails, and a rope bridge where the span is long. */
  timber?: boolean;
  /**
   * Water-rich Eyries': an aqueduct channel along one side of a bridge level enough to carry it,
   * from a cistern head at its higher end to a spout pouring into a basin at its lower.
   */
  aqueduct?: boolean;
};

/** A bridge's structures: its spans (each with its parapets, the pier before it, its lanterns), and its two ends. */
export function bridgeStructures(A: Arch, style: BridgeStyle, rnd: () => number): Structure[] {
  const out: Structure[] = [];
  const W = A.wMid;
  const flags = style.timber ? PFLAG.timber : 0, railFlags = style.timber ? PFLAG.lacquer | PFLAG.timber : 0;
  if (style.timber && A.len > 110) return ropeBridge(A, style, rnd);
  const target = style.span[0] + (style.span[1] - style.span[0]) * rnd();
  const n = Math.max(1, Math.round(A.len / target));
  const S = A.len / n;
  const dx = A.dirX, dz = A.dirZ, sx = -dz, sz = dx;          // along, and across (to the left)
  const yaw = Math.atan2(-dz, dx);                              // a span's local x along the bridge
  const yawUp = Math.atan2(dx, dz);                             // a stair's local z along the bridge
  const sy = 0.75 * S;                                          // so the arch rises ~0.35 of its span
  const deck = (t: number) => A.ay + (A.by - A.ay) * t;
  const spanY = (i: number) => deck((i + 0.5) / n);
  const seed = () => Math.floor(rnd() * 65536);
  const inst = (piece: number, x: number, y: number, z: number, yw: number, a: number, b: number, c: number, f = flags): PieceInst =>
    ({ piece, x, y, z, yaw: yw, sx: a, sy: b, sz: c, palette: style.palette, seed: seed(), flags: f });
  const hang = style.hang[0] + (style.hang[1] - style.hang[0]) * rnd();
  // an aqueduct along one side (inside its parapet), flowing from the higher end to the lower: level
  // along each span, falling into the next span's channel where the decks step down
  const aq = !!style.aqueduct && W >= 7 && rnd() < 0.6;
  const aqSide = rnd() < 0.5 ? -1 : 1, aqOff = W / 2 - 1.15, toB = A.ay >= A.by;
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n;
    const x = A.ax + dx * A.len * t, z = A.az + dz * A.len * t, y = spanY(i);
    const pieces: PieceInst[] = [inst(style.pointed ? P.spanPointed : P.span, x, y, z, yaw, S / 2, sy, W / 2)];
    if (aq) pieces.push(inst(P.aqueduct, x + sx * aqSide * aqOff, y, z + sz * aqSide * aqOff, toB ? yaw : yaw + Math.PI, S / 2 + 0.02, 1, 1));
    for (const side of [-1, 1]) {
      pieces.push(inst(P.parapet, x + sx * side * (W / 2 - 0.25), y, z + sz * side * (W / 2 - 0.25), yaw, S / 2, 1.1, 0.24, railFlags));
    }
    // the pier at this span's start (between it and the span before), hanging from the lower springing
    if (i > 0) {
      const tj = i / n;
      const jx = A.ax + dx * A.len * tj, jz = A.az + dz * A.len * tj;
      const spring = Math.min(spanY(i - 1), y) - 0.62 * sy;
      pieces.push(inst(P.pier, jx, spring + 0.05, jz, yaw, 0.11 * S, hang * S, W / 2 * 1.02));
      // steps where the two decks differ (clear of the aqueduct's side, if it has one), and the
      // channel's water falling from the higher span's into the lower's
      const d = y - spanY(i - 1);
      if (Math.abs(d) > 0.22) {
        const run = Math.max(1.2, Math.abs(d) * 1.8);
        const up = d > 0 ? 1 : -1;
        // the flight rises toward the higher deck, its top at the joint
        const shift = aq ? -aqSide * 0.9 : 0;
        const fx = jx - dx * up * run + sx * shift, fz = jz - dz * up * run + sz * shift;
        pieces.push(inst(Math.abs(d) > 1.2 ? P.stair12 : P.stair6, fx, Math.min(y, spanY(i - 1)), fz, up > 0 ? yawUp : yawUp + Math.PI, W / 2 - 0.5 - (aq ? 0.9 : 0), Math.abs(d), run));
        if (aq) {
          const ax2 = jx + sx * aqSide * aqOff, az2 = jz + sz * aqSide * aqOff;
          pieces.push(inst(P.aqFall, ax2, Math.max(y, spanY(i - 1)) + 0.62, az2, d < 0 ? yaw : yaw + Math.PI, 0.7, Math.abs(d), 1));
        }
      }
      if (i % 2 === 0) {
        for (const side of [-1, 1]) pieces.push({ ...inst(P.lantern, jx + sx * side * (W / 2 - 0.7), Math.max(y, spanY(i - 1)), jz + sz * side * (W / 2 - 0.7), yaw, 1, 1, 1), detail: 1 });
      }
    }
    out.push(makeStructure(pieces));
  }
  // The ends: Wardens either side, lanterns, perhaps a gate over the deck.
  for (const [t, outward] of [[0, -1], [1, 1]] as const) {
    const x = A.ax + dx * A.len * t, z = A.az + dz * A.len * t, y = deck(t);
    // (just off the bridge, on the island)
    const ox = x - dx * outward * 3, oz = z - dz * outward * 3;
    const face = Math.atan2(dx * outward, dz * outward) + Math.PI; // facing back along the bridge's approach
    const pieces: PieceInst[] = [];
    for (const side of [-1, 1]) {
      pieces.push(inst(P.warden, ox + sx * side * (W / 2 + 2.2), y - 0.2, oz + sz * side * (W / 2 + 2.2), face, 1, 1, 1));
      pieces.push(inst(P.lantern, ox + sx * side * (W / 2 + 0.6), y - 0.1, oz + sz * side * (W / 2 + 0.6), face, 1, 1, 1));
    }
    if (style.gate && W <= 12.5) pieces.push(inst(P.gate, x - dx * outward * 1.5, y - 0.3, z - dz * outward * 1.5, face, W / 6.4 + 0.15, W / 6.4 + 0.15, 1));
    if (aq) {
      // the channel's ends, on the island beyond (within the bridgehead it keeps clear): at its source
      // a cistern head under a dome; at its mouth a spout pouring into a basin
      const cx = x + sx * aqSide * aqOff, cz = z + sz * aqSide * aqOff;
      const source = toB ? t === 0 : t === 1;
      if (source) {
        const hx = cx + dx * outward * 4.5, hz = cz + dz * outward * 4.5;
        pieces.push(inst(P.tank, hx, y - 1.5, hz, yaw, 1.7, 4.1, 1.7), inst(P.dome, hx, y + 2.6, hz, 0, 1.75, 1.2, 1.75));
      } else {
        // (the spout's local +x out along the bridge, away from it)
        pieces.push(inst(P.spout, cx, y, cz, outward > 0 ? yaw : yaw + Math.PI, 1, 1, 1));
        pieces.push(inst(P.fountain, cx + dx * outward * 1.5, y - 0.45, cz + dz * outward * 1.5, 0, 0.45, 0.45, 0.45));
      }
    }
    out.push(makeStructure(pieces));
  }
  return out;
}

/**
 * A rope bridge (Petalfall's long crossings): one sagging deck of planks between two gate-arches,
 * rope rails, paper lanterns on the posts. Its deck sags a fortieth of its length.
 */
function ropeBridge(A: Arch, style: BridgeStyle, rnd: () => number): Structure[] {
  const W = Math.min(A.wMid, 5);
  const mx = (A.ax + A.bx) / 2, mz = (A.az + A.bz) / 2, my = Math.max(A.ay, A.by);
  const yaw = Math.atan2(-A.dirZ, A.dirX);
  const seed = () => Math.floor(rnd() * 65536);
  const piece = (p: number, x: number, y: number, z: number, yw: number, a: number, b: number, c: number): PieceInst =>
    ({ piece: p, x, y, z, yaw: yw, sx: a, sy: b, sz: c, palette: style.palette, seed: seed(), flags: 0 });
  const out: Structure[] = [makeStructure([piece(P.rope, mx, my, mz, yaw, A.len / 2, A.len / 40 + Math.abs(A.by - A.ay) * 0.5, W / 2)])];
  for (const [t, y] of [[0, A.ay], [1, A.by]] as const) {
    const x = A.ax + A.dirX * A.len * t, z = A.az + A.dirZ * A.len * t;
    out.push(makeStructure([
      piece(P.torii, x, y - 0.15, z, Math.atan2(A.dirX, A.dirZ), W / 8 + 0.2, W / 8 + 0.2, W / 8 + 0.2),
      piece(P.paper, x - A.dirZ * (W / 2 + 0.6), y + 3.6, z + A.dirX * (W / 2 + 0.6), 0, 1, 1, 1),
      piece(P.paper, x + A.dirZ * (W / 2 + 0.6), y + 3.6, z - A.dirX * (W / 2 + 0.6), 0, 1, 1, 1),
    ]));
  }
  return out;
}
