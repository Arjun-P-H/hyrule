/**
 * Insides (SKY_CIVILIZATION.md §8, §11 rule 10): what lets a building be
 * entered, and what is in it by its purpose. Catalogued with the rest in
 * pieces.ts.
 * ---------------------------------------------------------------------------
 * A storey is a ground floor that can be walked into: its wall a ring (or a
 * square) STOREY_H metres high whose doorway is DOOR_HALF metres either side
 * of the door's axis (local +z) whatever the building's size — the doorway's
 * edge vertices carry ANIM.door, and the vertex shader puts them there — its
 * jambs and soffit, its threshold, a floor and a ceiling. Its inner faces are
 * built facing out, as the outer: seen from inside they are back faces, which
 * the shader finishes as an inside (limewash, a dado, the day through the
 * windows; beams; a painted dome's stars). Unit radius (or half width), y in
 * metres: an instance's y scale is 1.
 *
 * The furnishings are what the catalogue puts on a roost's ground floor and in
 * a hall: store jars, a hearth with its pot, a loom, a nest bowl, a rug, a
 * table and stools, perch bars, shelves; and the flightless's ramp, spiralling
 * up the wall to a hatch (a radial piece: fitted to the room's inner radius).
 */

import { ANIM, AXIS, FACADE, KitBuilder, MAT, rng, type Style, type V3 } from './mesh';

type Lod = 0 | 1 | 2;
const n3 = (lod: Lod, a: number, b: number, c: number) => (lod === 0 ? a : lod === 1 ? b : c);
const TAU = Math.PI * 2;

/** A storey's height (m), its doorway's half width and the height of its head (m), its wall's thickness (unit). */
export const STOREY_H = 4.0;
export const DOOR_HALF = 1.05;
const DOOR_TOP = 3.3;
const WALL = 0.1;
/** The arc either side of the door the ring's own vertices keep clear of (rad): the doorway at its narrowest tower (r 3). */
const GAP = 0.42;

/** The ground storey of a round tower: radius 1 (unit), y 0 … STOREY_H (m); its doorway at local +z. */
export const storeyRound = (b: KitBuilder, lod: Lod) => storeyRing(b, lod, true);
/** The same, open above (no ceiling): the ground storey of a hall whose walls rise on to a dome. */
export const storeyRoundOpen = (b: KitBuilder, lod: Lod) => storeyRing(b, lod, false);

function storeyRing(b: KitBuilder, lod: Lod, ceiling: boolean): void {
  const n = n3(lod, 24, 14, 7);
  const out: Style = { mat: MAT.stone, facade: FACADE.roost | FACADE.ROUND, uAxis: AXIS.x, vAxis: AXIS.y };
  if (lod === 2) { b.frustum(0, 0, 0, STOREY_H, 1.02, 1, n, out); return; }
  // the ring's columns: the doorway's edge (ANIM.door: moved to the jamb), the ring round, the other edge
  const a0 = Math.PI / 2 + GAP, a1 = Math.PI / 2 + TAU - GAP;
  const angles = Array.from({ length: n + 1 }, (_, k) => a0 + ((a1 - a0) * k) / n);
  const door: Style = { anim: ANIM.door };
  const wall = (rad: number, st: Style) => {
    const col = (a: number, y: number, edge: boolean) =>
      b.vertex([Math.cos(a) * rad, y, Math.sin(a) * rad], [Math.cos(a), 0, Math.sin(a)], a, y, edge ? { ...st, ...door } : st);
    const lowCols = [[col(a0, 0, true), col(a0, DOOR_TOP, true)], ...angles.map((a) => [col(a, 0, false), col(a, DOOR_TOP, false)]), [col(a1, 0, true), col(a1, DOOR_TOP, true)]];
    for (let k = 0; k + 1 < lowCols.length; k++) b.quad(lowCols[k][0], lowCols[k + 1][0], lowCols[k + 1][1], lowCols[k][1]);
    // above the doorway's head, all the way round (the lintel over it)
    const upCols = [[col(a0, DOOR_TOP, true), col(a0, STOREY_H, true)], ...angles.map((a) => [col(a, DOOR_TOP, false), col(a, STOREY_H, false)]), [col(a1, DOOR_TOP, true), col(a1, STOREY_H, true)]];
    for (let k = 0; k + 1 < upCols.length; k++) b.quad(upCols[k][0], upCols[k + 1][0], upCols[k + 1][1], upCols[k][1]);
    const last = upCols.length - 1;
    b.quad(upCols[last][0], upCols[0][0], upCols[0][1], upCols[last][1]);
  };
  wall(1, out);
  // the inner face: limewashed (and seen from inside, as the outer's back)
  wall(1 - WALL, { mat: MAT.plaster, facade: FACADE.none | FACADE.ROUND, uAxis: AXIS.x, vAxis: AXIS.y });
  // jambs, soffit and threshold, through the wall's thickness
  const st: Style = { mat: MAT.trim, uAxis: AXIS.z, vAxis: AXIS.y, ...door };
  for (const [a, nx] of [[a0, 1], [a1, -1]] as const) {
    const p = (rad: number, y: number): V3 => [Math.cos(a) * rad, y, Math.sin(a) * rad];
    const ids = [p(1, 0), p(1 - WALL, 0), p(1 - WALL, DOOR_TOP), p(1, DOOR_TOP)].map((q) => b.vertex(q, [nx, 0, 0], q[2], q[1], st));
    b.quad(ids[0], ids[1], ids[2], ids[3]);
  }
  const corner = (a: number, rad: number, y: number) => b.vertex([Math.cos(a) * rad, y, Math.sin(a) * rad], [0, -1, 0], Math.cos(a) * rad, Math.sin(a) * rad, st);
  b.quad(corner(a0, 1, DOOR_TOP), corner(a1, 1, DOOR_TOP), corner(a1, 1 - WALL, DOOR_TOP), corner(a0, 1 - WALL, DOOR_TOP));
  const sill = (a: number, rad: number) => b.vertex([Math.cos(a) * rad, 0.03, Math.sin(a) * rad], [0, 1, 0], Math.cos(a) * rad, Math.sin(a) * rad, { mat: MAT.paving, ...door });
  b.quad(sill(a0, 1), sill(a0, 1 - WALL), sill(a1, 1 - WALL), sill(a1, 1));
  // the floor and the ceiling (inner faces: facing down, and up)
  b.disc(0, 0, 0.03, 1 - WALL, n, false, { mat: MAT.timber });
  if (ceiling) b.disc(0, 0, STOREY_H - 0.04, 1 - WALL, n, true, { mat: MAT.timber });
}

/** A round hall's wall rising over its storey, open at both ends (radius 1, height 1, round-headed windows): under a dome, its inside. */
export function bodyRoundOpen(b: KitBuilder, lod: Lod): void {
  b.frustum(0, 0, 0, 1, 1, 1, n3(lod, 24, 14, 7), { mat: MAT.stone, facade: FACADE.hall | FACADE.ROUND });
}

/**
 * Tiered perch-rings (radial: fitted to the hall's inner radius): three steps round the wall, 1.1 m
 * deep and 0.45 m high each (the highest against the wall), a perch bar along each step's edge; open
 * ±0.42 rad round the door (local +z).
 */
export function tiers(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 32, 16, 8);
  const a0 = Math.PI / 2 + 0.42, a1 = Math.PI / 2 + TAU - 0.42;
  const top: Style = { mat: MAT.paving }, riser: Style = { mat: MAT.trim };
  for (let k = 0; k < 3; k++) {
    const out = -1.1 * k, inn = -1.1 * (k + 1), h = 0.45 * (3 - k);
    for (let i = 0; i < n; i++) {
      const p = a0 + ((a1 - a0) * i) / n, q = a0 + ((a1 - a0) * (i + 1)) / n;
      const t = [b.radial(p, h, out, [0, 1, 0], p, 0, top), b.radial(p, h, inn, [0, 1, 0], p, 1, top), b.radial(q, h, inn, [0, 1, 0], q, 1, top), b.radial(q, h, out, [0, 1, 0], q, 0, top)];
      b.quad(t[0], t[1], t[2], t[3]);
      const r = [b.radial(p, 0, inn, [-1, 0, 0], p, 0, riser), b.radial(q, 0, inn, [-1, 0, 0], q, 0, riser), b.radial(q, h, inn, [-1, 0, 0], q, h, riser), b.radial(p, h, inn, [-1, 0, 0], p, h, riser)];
      b.quad(r[0], r[1], r[2], r[3]);
      if (lod === 0) {
        // the perch bar along its edge, on a post every other segment
        const bar = [b.radial(p, h + 0.62, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber }), b.radial(q, h + 0.62, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber }),
          b.radial(q, h + 0.7, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber }), b.radial(p, h + 0.7, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber })];
        b.quad(bar[0], bar[1], bar[2], bar[3]);
        if (i % 2 === 0) {
          const post = [b.radial(p, h, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber }), b.radial(p + 0.02, h, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber }),
            b.radial(p + 0.02, h + 0.66, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber }), b.radial(p, h + 0.66, inn + 0.15, [-1, 0, 0], 0, 0, { mat: MAT.timber })];
          b.quad(post[0], post[1], post[2], post[3]);
        }
      }
    }
  }
}

/** An orrery (2.6 m tall): brass rings of the sky's paths on a stand, the spheres riding them, a sunstone at their heart. */
export function orrery(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 28, 16, 8);
  b.frustum(0, 0, 0, 0.9, 0.35, 0.12, 8, { mat: MAT.bronze, metric: true });
  b.lathe([[0.001, 1.25], [0.22, 1.32], [0.24, 1.45], [0.001, 1.58]], 8, { mat: MAT.lantern, metric: true });
  if (lod === 2) return;
  const ring = (r: number, tilt: number, phase: number) => {
    b.push().translate(0, 1.42, 0).rotateZ(tilt).rotateY(phase);
    for (let k = 0; k < n; k++) {
      const a = (k / n) * TAU, c = ((k + 1) / n) * TAU;
      const v = [b.vertex([Math.cos(a) * r, -0.015, Math.sin(a) * r], [0, 1, 0], 0, 0, { mat: MAT.bronze }), b.vertex([Math.cos(c) * r, -0.015, Math.sin(c) * r], [0, 1, 0], 0, 0, { mat: MAT.bronze }),
        b.vertex([Math.cos(c) * r, 0.015, Math.sin(c) * r], [0, 1, 0], 0, 0, { mat: MAT.bronze }), b.vertex([Math.cos(a) * r, 0.015, Math.sin(a) * r], [0, 1, 0], 0, 0, { mat: MAT.bronze })];
      b.quad(v[0], v[1], v[2], v[3]);
    }
    b.push().translate(r, 0, 0);
    b.lathe([[0.001, -0.08], [0.08, 0], [0.001, 0.08]], 6, { mat: phase > 1 ? MAT.crystal : MAT.bronze, metric: true });
    b.pop();
    b.pop();
  };
  ring(0.55, 0.15, 0.4);
  ring(0.85, -0.25, 2.1);
  ring(1.15, 0.35, 4.0);
}

/** A telescope (3.2 m long) on its mount, raised to the sky through the dome's slit (local +z). */
export function telescope(b: KitBuilder, lod: Lod): void {
  b.frustum(0, 0, 0, 1.3, 0.3, 0.18, 8, { mat: MAT.timber, metric: true });
  b.push().translate(0, 1.4, 0).rotateX(-0.75);
  b.push().rotateX(Math.PI / 2);
  b.frustum(0, 0, -1.2, 2.0, 0.16, 0.2, n3(lod, 12, 8, 5), { mat: MAT.bronze, metric: true, capTop: true, capBottom: true });
  b.pop();
  b.pop();
}

/** A planting bed (2.4 × 1.2 m): a stone kerb, its soil, its plants in rows. */
export function planter(b: KitBuilder, lod: Lod): void {
  b.box([0, 0.3, 0], [1.2, 0.3, 0.6], { mat: MAT.stone });
  b.box([0, 0.605, 0], [1.12, 0.005, 0.52], { mat: MAT.clay });
  if (lod === 2) return;
  const r = rng(909);
  for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) {
    const h = 0.25 + 0.25 * r();
    b.push().translate(-0.95 + i * 0.38, 0, -0.22 + j * 0.44);
    b.lathe([[0.001, 0.6], [0.16, 0.6 + h * 0.4], [0.12, 0.6 + h * 0.8], [0.001, 0.6 + h]], 5, { mat: MAT.leaf, facade: 1, metric: true, phase: r() });
    b.pop();
  }
}


/** The ground storey of a square building: ±1 (unit), y 0 … STOREY_H (m); its doorway in the +z face. */
export function storeySquare(b: KitBuilder, lod: Lod): void {
  storeySquareOf(b, lod, FACADE.roost);
}

function storeySquareOf(b: KitBuilder, lod: Lod, facade: number): void {
  const out: Style = { mat: MAT.stone, facade: facade | FACADE.FLAT };
  if (lod === 2) { b.box([0, STOREY_H / 2, 0], [1, STOREY_H / 2, 1], { ...out, faces: 0b110011, sideFacade: facade | FACADE.FLAT }); return; }
  const inner: Style = { mat: MAT.plaster, facade: FACADE.FLAT };
  for (const [h, st] of [[1, out], [1 - WALL, inner]] as const) {
    // back, left and right faces whole; the front in three: either side of the doorway, and over it
    b.box([0, STOREY_H / 2, 0], [h, STOREY_H / 2, h], { ...st, faces: 0b100011, sideFacade: st.facade });
    const v = (x: number, y: number, edge: boolean) => b.vertex([x, y, h], [0, 0, 1], x, y, edge ? { ...st, uAxis: AXIS.x, vAxis: AXIS.y, anim: ANIM.door } : { ...st, uAxis: AXIS.x, vAxis: AXIS.y });
    for (const s of [-1, 1]) {
      // (the edge at x = ±ε, moved out to ±DOOR_HALF metres by the shader)
      const e = s * 0.001;
      b.quad(v(e, 0, true), v(s * h, 0, false), v(s * h, DOOR_TOP, false), v(e, DOOR_TOP, true));
    }
    b.quad(v(-h, DOOR_TOP, false), v(h, DOOR_TOP, false), v(h, STOREY_H, false), v(-h, STOREY_H, false));
  }
  const st: Style = { mat: MAT.trim, facade: FACADE.FLAT, uAxis: AXIS.z, vAxis: AXIS.y, anim: ANIM.door };
  for (const s of [-1, 1]) {
    const x = s * 0.001;
    const ids = ([[1, 0], [1 - WALL, 0], [1 - WALL, DOOR_TOP], [1, DOOR_TOP]] as const).map(([z, y]) => b.vertex([x, y, z], [-s, 0, 0], z, y, st));
    b.quad(ids[0], ids[1], ids[2], ids[3]);
  }
  const q = (x: number, z: number, y: number, ny: number, mat: number) => b.vertex([x, y, z], [0, ny, 0], x, z, { ...st, mat });
  b.quad(q(-0.001, 1, DOOR_TOP, -1, MAT.trim), q(0.001, 1, DOOR_TOP, -1, MAT.trim), q(0.001, 1 - WALL, DOOR_TOP, -1, MAT.trim), q(-0.001, 1 - WALL, DOOR_TOP, -1, MAT.trim));
  b.quad(q(-0.001, 1, 0.03, 1, MAT.paving), q(-0.001, 1 - WALL, 0.03, 1, MAT.paving), q(0.001, 1 - WALL, 0.03, 1, MAT.paving), q(0.001, 1, 0.03, 1, MAT.paving));
  // floor and ceiling (inner faces)
  b.box([0, 0.03, 0], [1 - WALL, 0.001, 1 - WALL], { mat: MAT.timber, faces: 0b001000 });
  b.box([0, STOREY_H - 0.04, 0], [1 - WALL, 0.001, 1 - WALL], { mat: MAT.timber, faces: 0b000100 });
}

/** An open doorway's portal (2.9 m wide, 3.7 m tall): a pointed stone frame 0.3 m proud of the wall, its opening ±1 m to 3.2 m. */
export function doorOpen(b: KitBuilder, lod: Lod): void {
  if (lod === 2) return;
  b.push().translate(0, 0, 0.15);
  b.archWall({ w: 1.45, y0: 0, y1: 3.75, t: 0.3, open: 1.0, spring: 2.3, crown: 3.2, pointed: true, segs: lod === 0 ? 10 : 6, keystone: lod === 0, style: { mat: MAT.trim } });
  b.pop();
}

// ---- furnishings ---------------------------------------------------------------------------------------

const clay: Style = { mat: MAT.clay };

/** Store jars: three amphorae of different sizes and a crate, in a corner (1.6 m across). */
export function jars(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 10, 7, 5);
  const amph = (x: number, z: number, h: number) => {
    b.push().translate(x, 0, z);
    b.lathe([[0.001, 0], [0.12, 0.02], [0.28, 0.25 * h], [0.32, 0.55 * h], [0.24, 0.85 * h], [0.11, 0.95 * h], [0.14, h]].map(([r, y]) => [r * (0.7 + 0.3 * h), y] as [number, number]),
      n, { ...clay, metric: true });
    b.pop();
  };
  amph(-0.35, 0.1, 1.0);
  amph(0.35, -0.15, 0.8);
  amph(0.05, -0.55, 1.15);
  if (lod !== 2) b.box([0.45, 0.25, 0.45], [0.3, 0.25, 0.3], { mat: MAT.timber });
}

/** A hearth-stone: a ring of stones round a fire, an iron tripod and its pot over it (1.5 m across). */
export function hearth(b: KitBuilder, lod: Lod): void {
  const r = rng(77077);
  const k = n3(lod, 9, 6, 4);
  for (let i = 0; i < k; i++) {
    const a = (i / k) * TAU;
    b.push().translate(Math.cos(a) * 0.62, 0.1, Math.sin(a) * 0.62).rotateY(a + r());
    b.box([0, 0, 0], [0.18, 0.11 + 0.03 * r(), 0.14], { mat: MAT.stone });
    b.pop();
  }
  b.disc(0, 0, 0.06, 0.5, n3(lod, 10, 6, 4), true, { mat: MAT.lava });
  if (lod === 2) return;
  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * TAU;
    b.push().translate(Math.cos(a) * 0.45, 0.6, Math.sin(a) * 0.45).rotateY(-a).rotateZ(0.22);
    b.box([0, 0, 0], [0.025, 0.62, 0.025], { mat: MAT.iron });
    b.pop();
  }
  b.lathe([[0.001, 0.62], [0.2, 0.66], [0.25, 0.8], [0.22, 0.95], [0.2, 0.96]], n3(lod, 9, 6, 4), { mat: MAT.iron, metric: true });
}

/** A loom: a timber frame 1.6 m wide, 1.8 m tall, cloth half woven on it in the House's colours. */
export function loom(b: KitBuilder, lod: Lod): void {
  const t: Style = { mat: MAT.timber };
  for (const s of [-1, 1]) b.box([s * 0.8, 0.9, 0], [0.05, 0.9, 0.12], t);
  b.box([0, 1.75, 0], [0.85, 0.05, 0.06], t);
  b.box([0, 0.35, 0], [0.85, 0.05, 0.06], t);
  if (lod === 2) return;
  b.box([0, 1.15, 0.0], [0.72, 0.55, 0.005], { mat: MAT.cloth });
  b.box([0, 0.55, 0.25], [0.8, 0.04, 0.2], t); // the bench
}

/** A nest bowl: a woven bowl 1.6 m across, cushioned in cloth, on a low timber stand. */
export function nestBowl(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 14, 9, 6);
  b.frustum(0, 0, 0, 0.25, 0.35, 0.4, n, { mat: MAT.timber, metric: true });
  b.lathe([[0.3, 0.25], [0.65, 0.35], [0.8, 0.6], [0.78, 0.66], [0.6, 0.45], [0.001, 0.4]], n, { mat: MAT.woven, metric: true, crease: 1.0 });
  if (lod !== 2) b.lathe([[0.001, 0.46], [0.45, 0.48], [0.58, 0.55], [0.001, 0.58]], n, { mat: MAT.cloth2, metric: true });
}

/** A round rug, 2.8 m across, in the House's colours with its emblem (the cloth material's border and sign). */
export function rug(b: KitBuilder, lod: Lod): void {
  b.disc(0, 0, 0.012, 1.4, n3(lod, 20, 12, 6), true, { mat: MAT.cloth });
}

/** A round table (1.4 m across) and three stools. */
export function table(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 14, 8, 5);
  const t: Style = { mat: MAT.timber };
  b.frustum(0, 0, 0.72, 0.78, 0.7, 0.7, n, { ...t, metric: true, capTop: true, capBottom: true });
  b.frustum(0, 0, 0, 0.72, 0.12, 0.08, 6, { ...t, metric: true });
  if (lod === 2) return;
  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * TAU + 0.5;
    b.frustum(Math.cos(a) * 1.0, Math.sin(a) * 1.0, 0, 0.45, 0.2, 0.2, 7, { ...t, metric: true, capTop: true });
  }
}

/** Perch bars: a timber bar 1.8 m long on two posts 1.3 m tall, where the Aelari rest. */
export function perch(b: KitBuilder, lod: Lod): void {
  const t: Style = { mat: MAT.timber };
  for (const s of [-1, 1]) b.box([s * 0.85, 0.65, 0], [0.05, 0.65, 0.05], t);
  b.box([0, 1.3, 0], [0.95, 0.04, 0.04], t);
  if (lod !== 2) b.box([0, 0.03, 0], [0.95, 0.03, 0.18], t);
}

/** Shelves: a timber case 1.8 m wide, 2.2 m tall, 0.4 m deep, with jars and bound cords on it (back at z −0.2). */
export function shelves(b: KitBuilder, lod: Lod): void {
  const t: Style = { mat: MAT.timber };
  for (const s of [-1, 1]) b.box([s * 0.88, 1.1, 0], [0.03, 1.1, 0.2], t);
  b.box([0, 1.1, -0.19], [0.88, 1.1, 0.01], t);
  const r = rng(321);
  for (let k = 0; k < 4; k++) {
    const y = 0.1 + k * 0.55;
    b.box([0, y, 0], [0.86, 0.02, 0.19], t);
    if (lod === 2) continue;
    for (let j = 0; j < 4; j++) {
      const x = -0.65 + j * 0.43 + 0.1 * r();
      if (r() < 0.5) b.frustum(x, -0.02, y + 0.02, y + 0.3, 0.09, 0.07, 6, { ...clay, metric: true, capTop: true });
      else b.box([x, y + 0.14, -0.02], [0.12, 0.12, 0.14], { mat: MAT.woven });
    }
  }
}

/**
 * The flightless's ramp (radial: fitted to the room's inner radius, its outer edge against the wall):
 * 0.95 m wide, rising from the floor beside the door, round the wall by RAMP_SWEEP rad to STOREY_H
 * − 0.2 m, ending under a hatch in the ceiling; solid beneath, a rail along its open side.
 */
export const RAMP_SWEEP = 4.2;
export const RAMP_START = Math.PI / 2 + 0.55;
const RAMP_W = 0.95;
export function rampSpiral(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 28, 14, 7);
  const top: Style = { mat: MAT.paving, uAxis: AXIS.x, vAxis: AXIS.z };
  const side: Style = { mat: MAT.stone, uAxis: AXIS.x, vAxis: AXIS.y };
  const H = STOREY_H - 0.25;
  for (let k = 0; k < n; k++) {
    const t0 = k / n, t1 = (k + 1) / n;
    const a0 = RAMP_START + t0 * RAMP_SWEEP, a1 = RAMP_START + t1 * RAMP_SWEEP;
    const y0 = t0 * H, y1 = t1 * H;
    // its walking surface
    const s = [b.radial(a0, y0, -0.02, [0, 1, 0], a0, 0, top), b.radial(a0, y0, -0.02 - RAMP_W, [0, 1, 0], a0, RAMP_W, top),
      b.radial(a1, y1, -0.02 - RAMP_W, [0, 1, 0], a1, RAMP_W, top), b.radial(a1, y1, -0.02, [0, 1, 0], a1, 0, top)];
    b.quad(s[0], s[3], s[2], s[1]);
    // its open side, down to the floor
    const w = [b.radial(a0, 0, -0.02 - RAMP_W, [-1, 0, 0], a0, 0, side), b.radial(a1, 0, -0.02 - RAMP_W, [-1, 0, 0], a1, 0, side),
      b.radial(a1, y1, -0.02 - RAMP_W, [-1, 0, 0], a1, y1, side), b.radial(a0, y0, -0.02 - RAMP_W, [-1, 0, 0], a0, y0, side)];
    b.quad(w[0], w[3], w[2], w[1]);
    if (lod === 0 && k % 2 === 0) {
      // a rail post
      const p = [b.radial(a0, y0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0, { mat: MAT.timber }), b.radial(a0, y0 + 1.0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0 + 1, { mat: MAT.timber }),
        b.radial(a0 + 0.03, y0 + 1.0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0 + 1, { mat: MAT.timber }), b.radial(a0 + 0.03, y0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0, { mat: MAT.timber })];
      b.quad(p[0], p[1], p[2], p[3]);
    }
  }
  // the hatch over its top: a dark square in the ceiling
  const aE = RAMP_START + RAMP_SWEEP;
  const h = [b.radial(aE - 0.25, STOREY_H - 0.05, -0.1, [0, -1, 0], 0, 0, { mat: MAT.dark }), b.radial(aE - 0.25, STOREY_H - 0.05, -0.1 - RAMP_W, [0, -1, 0], 0, 0, { mat: MAT.dark }),
    b.radial(aE, STOREY_H - 0.05, -0.1 - RAMP_W, [0, -1, 0], 0, 0, { mat: MAT.dark }), b.radial(aE, STOREY_H - 0.05, -0.1, [0, -1, 0], 0, 0, { mat: MAT.dark })];
  b.quad(h[0], h[1], h[2], h[3]);
}

// ---- halls ---------------------------------------------------------------------------------------------

/**
 * A long hall's side walls, open at both ends and its top: ±1, height 1 (unit, as body.hall),
 * lancet windows down them — the Gyre Cathedral's nave, whose west front closes one end and its
 * apse the other, and whose roof, seen from inside, is its rafters.
 */
export function bodyHallOpen(b: KitBuilder): void {
  b.mat = MAT.stone;
  // (its sides only: the west front closes its front, the apse its far end)
  b.box([0, 0.5, 0], [1, 0.5, 1], { faces: 0b000011, sideFacade: FACADE.lancet | FACADE.FLAT });
}

/** A long table (3.6 × 1 m) and a bench along each side. */
export function tableLong(b: KitBuilder, lod: Lod): void {
  const t: Style = { mat: MAT.timber };
  b.box([0, 0.74, 0], [1.8, 0.04, 0.5], t);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) b.box([sx * 1.6, 0.36, sz * 0.38], [0.05, 0.36, 0.05], t);
  if (lod === 2) return;
  for (const sz of [-1, 1]) {
    b.box([0, 0.44, sz * 0.95], [1.7, 0.03, 0.17], t);
    for (const sx of [-1, 1]) b.box([sx * 1.5, 0.21, sz * 0.95], [0.04, 0.21, 0.13], t);
  }
}

/**
 * A gallery of perches along a wall (stretch: x ±1 its length, z 0 → 1 out from the wall, y in
 * metres): a stone floor 0.35 m thick on corbels, a parapet 1 m high at its edge with its rail, and
 * a perch bar along it.
 */
export function gallery(b: KitBuilder, lod: Lod): void {
  b.box([0, -0.175, 0.5], [1, 0.175, 0.5], { mat: MAT.stone, faces: 0b111111 });
  b.box([0, 0.5, 0.97], [1, 0.5, 0.03], { mat: MAT.trim, faces: 0b111111 });
  if (lod === 2) return;
  b.box([0, 1.03, 0.97], [1, 0.04, 0.06], { mat: MAT.trim });
  b.box([0, 1.25, 0.6], [1, 0.035, 0.035], { mat: MAT.timber });
  // (corbels: a row of brackets under it)
  for (let k = -4; k <= 4; k++) b.box([k * 0.24, -0.6, 0.25], [0.02, 0.25, 0.25], { mat: MAT.stone });
}

/**
 * The wind altar (2 m across): a marble drum on two steps, and over it the Gyre in bronze — a
 * ribbon of feathers spiralling up to a skystone held at its crown, glowing.
 */
export function altar(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 24, 14, 8);
  b.frustum(0, 0, 0, 0.2, 2.0, 2.0, n, { mat: MAT.trim, metric: true, capTop: true });
  b.frustum(0, 0, 0.2, 0.4, 1.6, 1.6, n, { mat: MAT.trim, metric: true, capTop: true });
  b.frustum(0, 0, 0.4, 1.4, 1.0, 1.0, n, { mat: MAT.marble, metric: true, capTop: true });
  if (lod === 2) return;
  // the ribbon: a helix of feathers rising 2.6 m, narrowing
  const turns = 1.6, segs = n3(lod, 40, 20, 8);
  const rows: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const a = t * turns * TAU, r = 0.75 * (1 - 0.7 * t), y = 1.45 + 2.6 * t, w = 0.22 * (1 - 0.5 * t);
    const nrm: V3 = [Math.cos(a), 0.3, Math.sin(a)];
    rows.push([b.vertex([Math.cos(a) * r, y - w, Math.sin(a) * r], nrm, t * 6, 0, { mat: MAT.bronze }), b.vertex([Math.cos(a) * r, y + w, Math.sin(a) * r], nrm, t * 6, 1, { mat: MAT.bronze })]);
  }
  for (let i = 0; i < segs; i++) b.quad(rows[i][0], rows[i + 1][0], rows[i + 1][1], rows[i][1]);
  b.push().translate(0, 4.25, 0);
  b.lathe([[0.001, -0.35], [0.2, -0.1], [0.24, 0.1], [0.001, 0.4]], 6, { mat: MAT.crystal, metric: true, flatSides: true });
  b.pop();
}

/**
 * An apse: the half of a round hall behind (−z) its diameter, open toward +z where the nave meets it
 * — radius 1, height 1 (unit), lancet windows round it; seen from the nave, its inside.
 */
export function bodyApse(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 16, 10, 5);
  const st: Style = { mat: MAT.stone, facade: FACADE.lancet | FACADE.ROUND, uAxis: AXIS.x, vAxis: AXIS.y };
  const cols: number[][] = [];
  for (let k = 0; k <= n; k++) {
    const a = Math.PI + (k / n) * Math.PI;
    cols.push([b.vertex([Math.cos(a), 0, Math.sin(a)], [Math.cos(a), 0, Math.sin(a)], a, 0, st), b.vertex([Math.cos(a), 1, Math.sin(a)], [Math.cos(a), 0, Math.sin(a)], a, 1, st)]);
  }
  for (let k = 0; k < n; k++) b.quad(cols[k][0], cols[k + 1][0], cols[k + 1][1], cols[k][1]);
}

/** The apse's roof: a half cone over the −z half (radius 1 at y 0 to its apex at y 1). */
export function roofApse(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 16, 10, 5);
  const st: Style = { mat: MAT.roof, uAxis: 3, vAxis: 3 };
  const apex = b.vertex([0, 1, 0], [0, 1, 0], 0, 0, st);
  const rim: number[] = [];
  for (let k = 0; k <= n; k++) {
    const a = Math.PI + (k / n) * Math.PI;
    rim.push(b.vertex([Math.cos(a) * 1.04, -0.03, Math.sin(a) * 1.04], [Math.cos(a) * 0.7, 0.7, Math.sin(a) * 0.7], a, 1.04, st));
  }
  for (let k = 0; k < n; k++) b.tri(apex, rim[k], rim[k + 1]);
}

// ---- the rest of the catalogue's insides --------------------------------------------------------------

/** A wind-house's ground storey (as storeySquare, its outside a timber lattice). */
export function storeyLattice(b: KitBuilder, lod: Lod): void {
  storeySquareOf(b, lod, FACADE.lattice);
}

/**
 * An octagonal storey (the archive spire's): radius 1 to its corners (unit), y 0 … STOREY_H (m), open
 * above (the shaft rises on through the body over it); its doorway in the face toward +z.
 */
export function storeyOctagon(b: KitBuilder, lod: Lod): void {
  const out: Style = { mat: MAT.stone, facade: FACADE.hall | FACADE.ROUND, uAxis: AXIS.x, vAxis: AXIS.y };
  if (lod === 2) { b.frustum(0, 0, 0, STOREY_H, 1, 1, 8, { ...out, flat: true, phase: Math.PI / 8 }); return; }
  const corner = (k: number) => Math.PI / 8 + (k * Math.PI) / 4; // corners at π/8 + kπ/4: the face round π/2 is the door's
  for (const [rad, mat] of [[1, MAT.stone], [1 - WALL, MAT.plaster]] as const) {
    const st: Style = { ...out, mat, facade: mat === MAT.stone ? out.facade : FACADE.ROUND };
    for (let k = 0; k < 8; k++) {
      const a0 = corner(k), a1 = corner(k + 1), am = 0.5 * (a0 + a1);
      const nrm: V3 = [Math.cos(am), 0, Math.sin(am)];
      const p = (a: number, y: number): V3 => [Math.cos(a) * rad, y, Math.sin(a) * rad];
      const v = (q: V3, a: number, door = false) => b.vertex(q, nrm, a * rad, q[1], door ? { ...st, facade: FACADE.FLAT, anim: ANIM.door } : st);
      if (k !== 1) {
        const ids = [v(p(a0, 0), a0), v(p(a1, 0), a1), v(p(a1, STOREY_H), a1), v(p(a0, STOREY_H), a0)];
        b.quad(ids[0], ids[1], ids[2], ids[3]);
        continue;
      }
      // the door's face (corners 3π/8 … 5π/8, flat at z = cos(π/8) · rad): either side of the doorway, then over it
      const z = Math.cos(Math.PI / 8) * rad, x0 = Math.cos(a0) * rad, x1 = Math.cos(a1) * rad;
      const q = (x: number, y: number, door: boolean) => v([x, y, z], Math.atan2(z, x), door);
      b.quad(q(x0, 0, false), q(0.001, 0, true), q(0.001, DOOR_TOP, true), q(x0, DOOR_TOP, false));
      b.quad(q(-0.001, 0, true), q(x1, 0, false), q(x1, DOOR_TOP, false), q(-0.001, DOOR_TOP, true));
      b.quad(q(x0, DOOR_TOP, false), q(x1, DOOR_TOP, false), q(x1, STOREY_H, false), q(x0, STOREY_H, false));
    }
  }
  // jambs, soffit and threshold through the face
  const zo = Math.cos(Math.PI / 8), zi = zo * (1 - WALL);
  const st: Style = { mat: MAT.trim, facade: FACADE.FLAT, uAxis: AXIS.z, vAxis: AXIS.y, anim: ANIM.door };
  for (const s of [-1, 1]) {
    const x = s * 0.001;
    const ids = ([[zo, 0], [zi, 0], [zi, DOOR_TOP], [zo, DOOR_TOP]] as const).map(([z, y]) => b.vertex([x, y, z], [-s, 0, 0], z, y, st));
    b.quad(ids[0], ids[1], ids[2], ids[3]);
  }
  const f = (x: number, z: number, y: number, ny: number, mat: number) => b.vertex([x, y, z], [0, ny, 0], x, z, { ...st, mat });
  b.quad(f(-0.001, zo, DOOR_TOP, -1, MAT.trim), f(0.001, zo, DOOR_TOP, -1, MAT.trim), f(0.001, zi, DOOR_TOP, -1, MAT.trim), f(-0.001, zi, DOOR_TOP, -1, MAT.trim));
  b.quad(f(-0.001, zo, 0.03, 1, MAT.paving), f(-0.001, zi, 0.03, 1, MAT.paving), f(0.001, zi, 0.03, 1, MAT.paving), f(0.001, zo, 0.03, 1, MAT.paving));
  // its floor (facing down: an inside)
  b.disc(0, 0, 0.03, (1 - WALL) * 0.98, 8, false, { mat: MAT.paving }, Math.PI / 8);
}

/**
 * A ramp slab winding up a shaft (radial: fitted to the shaft's inner radius), as rampSpiral but a
 * slab 0.4 m thick with nothing under it and no hatch: stacked, turn on turn (each the next's start).
 */
export function rampSlab(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 28, 14, 7);
  const top: Style = { mat: MAT.paving, uAxis: AXIS.x, vAxis: AXIS.z };
  const side: Style = { mat: MAT.stone, uAxis: AXIS.x, vAxis: AXIS.y };
  const H = STOREY_H - 0.25;
  for (let k = 0; k < n; k++) {
    const t0 = k / n, t1 = (k + 1) / n;
    const a0 = RAMP_START + t0 * RAMP_SWEEP, a1 = RAMP_START + t1 * RAMP_SWEEP;
    const y0 = t0 * H, y1 = t1 * H;
    const s = [b.radial(a0, y0, -0.02, [0, 1, 0], a0, 0, top), b.radial(a0, y0, -0.02 - RAMP_W, [0, 1, 0], a0, RAMP_W, top),
      b.radial(a1, y1, -0.02 - RAMP_W, [0, 1, 0], a1, RAMP_W, top), b.radial(a1, y1, -0.02, [0, 1, 0], a1, 0, top)];
    b.quad(s[0], s[3], s[2], s[1]);
    const u = [b.radial(a0, y0 - 0.4, -0.02, [0, -1, 0], a0, 0, side), b.radial(a0, y0 - 0.4, -0.02 - RAMP_W, [0, -1, 0], a0, RAMP_W, side),
      b.radial(a1, y1 - 0.4, -0.02 - RAMP_W, [0, -1, 0], a1, RAMP_W, side), b.radial(a1, y1 - 0.4, -0.02, [0, -1, 0], a1, 0, side)];
    b.quad(u[0], u[1], u[2], u[3]);
    const w = [b.radial(a0, y0 - 0.4, -0.02 - RAMP_W, [-1, 0, 0], a0, 0, side), b.radial(a1, y1 - 0.4, -0.02 - RAMP_W, [-1, 0, 0], a1, 0, side),
      b.radial(a1, y1, -0.02 - RAMP_W, [-1, 0, 0], a1, 0.4, side), b.radial(a0, y0, -0.02 - RAMP_W, [-1, 0, 0], a0, 0.4, side)];
    b.quad(w[0], w[3], w[2], w[1]);
    if (lod === 0 && k % 2 === 0) {
      const p = [b.radial(a0, y0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0, { mat: MAT.timber }), b.radial(a0, y0 + 1.0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0 + 1, { mat: MAT.timber }),
        b.radial(a0 + 0.03, y0 + 1.0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0 + 1, { mat: MAT.timber }), b.radial(a0 + 0.03, y0, -0.06 - RAMP_W, [-1, 0, 0], a0, y0, { mat: MAT.timber })];
      b.quad(p[0], p[1], p[2], p[3]);
    }
  }
}

/**
 * A sunken pool (stretch: radius 1 unit, y in metres): a kerb 0.35 m high round it, its water 0.2 m
 * under the floor, its sides going down 1.2 m to a paved bed — a cistern's store.
 */
export function poolRound(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 28, 16, 8);
  b.lathe([[1.08, 0], [1.08, 0.35], [1.0, 0.38], [0.98, 0.35], [0.98, -1.2], [0.001, -1.2]], n, { mat: MAT.trim, crease: 0.6 });
  b.disc(0, 0, -0.2, 0.98, n, true, { mat: MAT.water });
}

/** A measuring rod (fixed, 2.6 m): a marked stone post, a bronze band every half metre, standing in a pool. */
export function measuringRod(b: KitBuilder, lod: Lod): void {
  b.frustum(0, 0, -1.2, 1.4, 0.09, 0.07, 6, { mat: MAT.marble, metric: true, capTop: true });
  if (lod !== 2) for (let k = 0; k < 5; k++) b.frustum(0, 0, -0.9 + k * 0.5, -0.86 + k * 0.5, 0.1, 0.1, 6, { mat: MAT.bronze, metric: true });
}

/** An anvil on its stump (fixed, 1 m). */
export function anvil(b: KitBuilder, lod: Lod): void {
  b.frustum(0, 0, 0, 0.55, 0.32, 0.3, n3(lod, 9, 6, 4), { mat: MAT.timber, metric: true, capTop: true });
  b.box([0, 0.65, 0], [0.42, 0.1, 0.16], { mat: MAT.iron });
  b.box([0, 0.78, 0], [0.32, 0.04, 0.13], { mat: MAT.iron });
  if (lod !== 2) {
    b.push().translate(0.5, 0.74, 0).rotateZ(-0.6);
    b.box([0, 0, 0], [0.16, 0.05, 0.09], { mat: MAT.iron });
    b.pop();
  }
}

/** A furnace (fixed, 3.2 m): a brick dome on a plinth, its mouth glowing, a flue rising from its crown. */
export function furnace(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 14, 9, 6);
  b.frustum(0, 0, 0, 0.6, 1.45, 1.4, n, { mat: MAT.stone, metric: true });
  b.lathe([[1.35, 0.6], [1.3, 1.3], [1.0, 2.0], [0.55, 2.4], [0.35, 2.5]], n, { mat: MAT.clay, metric: true });
  b.frustum(0, 0, 2.45, 3.3, 0.32, 0.28, 6, { mat: MAT.stone, metric: true });
  // the mouth at +z, glowing
  b.box([0, 1.05, 1.3], [0.45, 0.35, 0.06], { mat: MAT.lava });
}

/** A small shrine (fixed, 2 m): a stone plinth, a runestone on it, a paper lantern hung beside. */
export function shrineSmall(b: KitBuilder, lod: Lod): void {
  b.box([0, 0.4, 0], [0.6, 0.4, 0.45], { mat: MAT.trim });
  b.push().translate(0, 0.8, 0).scale(0.22);
  runestoneLite(b, lod);
  b.pop();
  if (lod !== 2) b.lathe([[0.001, 1.95], [0.14, 1.8], [0.17, 1.65], [0.12, 1.5], [0.001, 1.48]], 6, { mat: MAT.lantern, metric: true });
}
function runestoneLite(b: KitBuilder, lod: Lod): void {
  b.lathe([[0.9, 0], [1.0, 0.6], [0.85, 3.2], [0.7, 5.0], [0.3, 5.9], [0.001, 6.1]], lod === 0 ? 6 : 4, { mat: MAT.rune, flatSides: true, crease: 0.5 });
}
