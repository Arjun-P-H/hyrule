/**
 * The Eyries' building kit: every piece, its shape at three levels of detail.
 * ---------------------------------------------------------------------------
 * Units (mesh.ts): a stretchable piece is modelled in unit space and an
 * instance scales it per axis (a round body: radius 1, height 1, scaled to
 * the tower's radius and height); a fixed piece is modelled in metres and
 * scaled uniformly; a radial ring is fitted to the instance's radius (its x
 * scale) with its profile in metres. Level 0 is seen near, 1 at middle
 * distance, 2 far (a few dozen triangles).
 *
 * Why each piece looks as it does is SKY_CIVILIZATION.md §6: round towers
 * that shed the wind, a ledge at every floor, steep scale-tiled roofs,
 * needle spires with finials, pointed arches with skystone keystones, bridges
 * on hanging piers that end in glowing plumb-stones, lanterns everywhere.
 */

import { ANIM, AXIS, FACADE, KitBuilder, MAT, smooth, type Style, type V3 } from './mesh';
import {
  awning, boughRoots, boughTree, brazier, bridgeRope, cargo, chimney, cloudComb, columnFallen, crane, crystalCluster, dock, gateTorii, haloRing,
  lanternPaper, nestPlatform, roofPagoda, rubble, ruinTop, runestone, skyship, stall, windSails,
} from './artefacts';
import { aqueduct, aqueductFall, koi, lilyPad, lotus, reeds, spoutFall, waterweed } from './water';
import {
  altar, bodyApse, bodyHallOpen, bodyRoundOpen, doorOpen, gallery, hearth, jars, loom, nestBowl, orrery, perch, planter, rampSpiral, roofApse, rug, shelves,
  storeyRound, storeyRoundOpen, storeySquare, table, tableLong, telescope, tiers, storeyLattice, storeyOctagon, rampSlab, poolRound,
  measuringRod, anvil, furnace, shrineSmall,
} from './interiors';

export type Lod = 0 | 1 | 2;
type PieceKind = 'stretch' | 'fixed' | 'radial';

type PieceDef = {
  name: string;
  kind: PieceKind;
  /**
   * The coarsest level of its structure it is still drawn at: 0 near only (props, small details),
   * 1 near and middle, 2 always (massing).
   */
  detail: Lod;
  build: (b: KitBuilder, lod: Lod) => void;
};

const n3 = (lod: Lod, a: number, b: number, c: number) => (lod === 0 ? a : lod === 1 ? b : c);
const TAU = Math.PI * 2;

/** Occlusion that darkens toward a piece's foot (unit height 1). */
const footAO = (h: number) => (p: V3) => 0.7 + 0.3 * smooth(0, h, p[1]);

// ---- bodies (stretchable) -------------------------------------------------------------------

/** A round tower's body: radius 1, height 1, a slight flare at its foot, a cap. Roost windows. */
function bodyRound(facade: number) {
  return (b: KitBuilder, lod: Lod) => {
    const n = n3(lod, 22, 12, 6);
    b.mat = MAT.stone; b.facade = facade | FACADE.ROUND;
    if (lod === 2) { b.frustum(0, 0, 0, 1, 1.02, 1, n, { capTop: true }); return; }
    b.frustum(0, 0, 0, 0.035, 1.05, 1, n, { ao: footAO(0.05) });
    b.frustum(0, 0, 0.035, 1, 1, 1, n, { capTop: true, ao: footAO(0.05), v0: 0.035 });
  };
}

/** A square body ±1, height 1, a cap. */
function bodySquare(facade: number) {
  return (b: KitBuilder) => {
    b.mat = MAT.stone;
    b.box([0, 0.5, 0], [1, 0.5, 1], { faces: 0b110111, sideFacade: facade | FACADE.FLAT, ao: footAO(0.05) });
  };
}

/** An octagonal body (flat faces), radius 1 to its corners, height 1. */
function bodyOctagon(facade: number) {
  return (b: KitBuilder) => {
    b.mat = MAT.stone; b.facade = facade | FACADE.ROUND;
    b.frustum(0, 0, 0, 1, 1, 1, 8, { flat: true, capTop: true, phase: Math.PI / 8, ao: footAO(0.05) });
  };
}

/** A podium or terrace: battered sides from its top (y 0, paved) down to y −1, ±1 at the top. */
function podium(b: KitBuilder, lod: Lod) {
  const r = Math.SQRT2;
  b.mat = MAT.stone;
  b.lathe([[r * 1.06, -1], [r * 1.0, -0.04], [r * 1.0, 0]], 4, { flatSides: true, phase: Math.PI / 4, ao: (p) => 0.8 + 0.2 * smooth(-1, -0.2, p[1]) });
  b.disc(0, 0, 0, r, 4, true, { mat: MAT.paving }, Math.PI / 4);
  if (lod === 2) return;
}

/** A round plaza: a mosaic disc of radius 1 at y 0, its skirt down to y −1. */
function plazaRound(b: KitBuilder, lod: Lod) {
  const n = n3(lod, 48, 24, 10);
  b.mat = MAT.stone;
  b.frustum(0, 0, -1, 0, 1.03, 1, n, { ao: (p) => 0.8 + 0.2 * smooth(-1, -0.1, p[1]) });
  b.disc(0, 0, 0, 1, n, true, { mat: MAT.mosaic });
}

// ---- roofs (stretchable) ---------------------------------------------------------------------

/** Bell-cast cone: eave overhang and lip, then a concave sweep to the apex (radius 1 at y 0, apex y 1). */
const CONE: ReadonlyArray<readonly [number, number]> = [
  [0.97, -0.045], [1.15, -0.045], [1.15, -0.012], [1.02, 0.03], [0.8, 0.2], [0.55, 0.45], [0.3, 0.72], [0.12, 0.9], [0, 1],
];
function roofCone(b: KitBuilder, lod: Lod) {
  b.mat = MAT.roof;
  if (lod === 2) { b.lathe([[1.1, -0.02], [0.6, 0.38], [0, 1]], 6, { uAxis: 3, vAxis: 3 }); return; }
  if (lod === 1) { b.lathe([[1.15, -0.045], [1.15, -0.012], [0.8, 0.2], [0.3, 0.72], [0, 1]], 12, { uAxis: 3, vAxis: 3, crease: 0.6 }); return; }
  b.lathe(CONE, n3(lod, 26, 12, 6), { uAxis: 3, vAxis: 3, crease: 0.6, ao: (p) => (p[1] < 0 ? 0.6 : 1) });
}

/** Needle spire: octagonal, a short eave, lucarnes (dormers) near its foot, a gold band near its tip. */
function roofSpire(b: KitBuilder, lod: Lod) {
  b.mat = MAT.roof;
  if (lod === 2) { b.lathe([[1.05, 0], [0.5, 0.5], [0, 1]], 4, { uAxis: 3, vAxis: 3, flatSides: true, phase: Math.PI / 4 }); return; }
  const prof: ReadonlyArray<readonly [number, number]> = [[0.92, -0.01], [1.1, -0.01], [1.1, 0.008], [0.92, 0.05], [0.5, 0.5], [0.06, 0.96], [0.03, 1]];
  b.lathe(prof, 8, { uAxis: 3, vAxis: 3, flatSides: true, phase: Math.PI / 8, crease: 0.4 });
  // a gold band
  b.lathe([[0.075, 0.93], [0.08, 0.935], [0.068, 0.955]], 8, { mat: MAT.gold, flatSides: true, phase: Math.PI / 8 });
  if (lod === 1) return;
  // lucarnes: small gabled dormers on four faces
  for (let k = 0; k < 4; k++) {
    b.push().rotateY((k / 4) * TAU + Math.PI / 4).translate(0, 0.2, 0.74);
    b.box([0, 0.03, 0], [0.11, 0.03, 0.08], { mat: MAT.stone, faces: 0b010111 });
    // its little gable roof
    gableShape(b, 0.13, 0.05, 0.09, 0.06, MAT.roof, MAT.stone);
    b.pop();
  }
}

/** Four-sided bell-cast pyramid (half width 1 to its eaves' middles). */
function roofPyramid(b: KitBuilder, lod: Lod) {
  b.mat = MAT.roof;
  const r = Math.SQRT2;
  const prof: ReadonlyArray<readonly [number, number]> = lod === 2 ? [[1.1 * r, -0.02], [0, 1]]
    : [[0.97 * r, -0.04], [1.13 * r, -0.04], [1.13 * r, -0.01], [1.0 * r, 0.04], [0.55 * r, 0.42], [0.2 * r, 0.82], [0, 1]];
  b.lathe(prof, 4, { uAxis: 3, vAxis: 3, flatSides: true, phase: Math.PI / 4, crease: 0.5 });
  if (lod === 0) b.lathe([[0.05, 0.98], [0.05, 1.03], [0, 1.04]], 4, { mat: MAT.trim, flatSides: true, phase: Math.PI / 4 });
}

/**
 * A gabled roof shape in the current frame: ridge along x (±hx), eaves at z ±hz (with overhang),
 * ridge height h above y 0; gable ends of `wall` material.
 */
function gableShape(b: KitBuilder, hx: number, h: number, hz: number, over: number, roofMat: number, wall: number): void {
  const ex = hx + over * 0.6, ez = hz + over;
  const slopeN = (s: number): V3 => {
    const l = Math.hypot(h, ez);
    return [0, ez / l, (s * h) / l];
  };
  for (const s of [1, -1]) {
    const n = slopeN(s);
    const st: Style = { mat: roofMat, uAxis: AXIS.x, vAxis: 3 };
    const q = [
      b.vertex([-ex, 0 - over * 0.25, s * ez], n, -ex, s * ez, st), b.vertex([ex, 0 - over * 0.25, s * ez], n, ex, s * ez, st),
      b.vertex([ex, h, 0], n, ex, 0, st), b.vertex([-ex, h, 0], n, -ex, 0, st),
    ];
    if (s > 0) b.quad(q[0], q[1], q[2], q[3]); else b.quad(q[0], q[3], q[2], q[1]);
    // soffit under the overhang
    const sf = [b.vertex([-ex, -over * 0.25, s * ez], [0, -1, 0], -ex, ez, { mat: roofMat }, 0.6), b.vertex([ex, -over * 0.25, s * ez], [0, -1, 0], ex, ez, { mat: roofMat }, 0.6),
      b.vertex([ex, 0, s * hz], [0, -1, 0], ex, hz, { mat: roofMat }, 0.6), b.vertex([-ex, 0, s * hz], [0, -1, 0], -ex, hz, { mat: roofMat }, 0.6)];
    if (s > 0) b.quad(sf[0], sf[3], sf[2], sf[1]); else b.quad(sf[0], sf[1], sf[2], sf[3]);
  }
  // gable ends
  for (const s of [1, -1]) {
    const x = s * hx;
    const n: V3 = [s, 0, 0];
    const st: Style = { mat: wall, uAxis: AXIS.z, vAxis: AXIS.y };
    const t = [b.vertex([x, 0, -hz], n, -hz, 0, st), b.vertex([x, 0, hz], n, hz, 0, st), b.vertex([x, h, 0], n, 0, h, st)];
    if (s > 0) b.tri(t[0], t[2], t[1]); else b.tri(t[0], t[1], t[2]);
  }
}

/** A gabled roof: ridge along x over ±1, eaves at z ±1 (+ overhang), ridge at y 1. */
function roofGable(b: KitBuilder, lod: Lod) {
  gableShape(b, 1, 1, 1, 0.1, MAT.roof, MAT.stone);
  if (lod === 0) b.box([0, 1.005, 0], [1.03, 0.02, 0.025], { mat: MAT.trim, faces: 0b110111 });
}

/** A dome: radius 1 at y 0 rising to y 1, slightly pointed; ribbed sheathing. */
function roofDome(b: KitBuilder, lod: Lod) {
  const prof: ReadonlyArray<readonly [number, number]> = lod === 2 ? [[1.04, 0], [0.92, 0.42], [0.55, 0.82], [0, 1]]
    : [[1.06, -0.03], [1.06, 0.02], [1.0, 0.05], [0.98, 0.2], [0.9, 0.42], [0.74, 0.64], [0.52, 0.82], [0.26, 0.95], [0.05, 1]];
  b.lathe(prof, n3(lod, 32, 16, 7), { mat: MAT.dome, uAxis: 3, vAxis: 3, crease: 0.9 });
}

/** A greenhouse's dome: leaded glass on iron ribs (radius 1 at y 0, crown at y 1). */
function roofGlass(b: KitBuilder, lod: Lod) {
  const prof: ReadonlyArray<readonly [number, number]> = [[1.02, 0], [1.0, 0.04], [0.97, 0.3], [0.86, 0.56], [0.66, 0.78], [0.38, 0.94], [0.08, 1]];
  b.lathe(prof, n3(lod, 24, 14, 8), { mat: MAT.glass, uAxis: 3, vAxis: 3 });
  if (lod !== 2) b.lathe([[0.1, 0.99], [0.12, 1.03], [0.05, 1.08], [0, 1.12]], 6, { mat: MAT.gold });
}

/** An onion dome over a narrow neck. */
function roofOnion(b: KitBuilder, lod: Lod) {
  const prof: ReadonlyArray<readonly [number, number]> = lod === 2 ? [[0.9, 0], [1.2, 0.3], [0.82, 0.64], [0, 1]]
    : [[0.9, 0], [0.92, 0.04], [1.12, 0.16], [1.2, 0.3], [1.1, 0.46], [0.82, 0.64], [0.45, 0.8], [0.18, 0.91], [0.05, 1]];
  b.lathe(prof, n3(lod, 28, 14, 6), { mat: MAT.dome, uAxis: 3, vAxis: 3 });
}

// ---- rings (radial: fitted to the instance's radius, profiles in metres) ----------------------

/** A moulded cornice band, 0.7 m tall, projecting 0.38 m, its top running in under a roof. */
function ringCornice(b: KitBuilder, lod: Lod) {
  b.mat = MAT.trim;
  if (lod !== 0) { b.radialRing([[-0.02, 0], [0.36, 0.4], [0.38, 0.68], [-0.8, 0.68]], lod === 1 ? 16 : 8, { crease: 0.5 }); return; }
  b.radialRing([[-0.02, 0], [0.1, 0.06], [0.1, 0.22], [0.3, 0.4], [0.38, 0.58], [0.38, 0.68], [-0.8, 0.68]], 40, { crease: 0.5 });
}

/** A perch-ledge: a slab projecting 0.95 m at floor level on corbels, a low curb at its rim. */
function ringLedge(b: KitBuilder, lod: Lod) {
  b.mat = MAT.trim;
  if (lod !== 0) {
    b.radialRing([[-0.02, -0.36], [0.95, -0.12], [0.95, 0.22], [-0.02, 0.22]], lod === 1 ? 14 : 8, { crease: 0.4 });
    return;
  }
  const n = 40;
  b.radialRing([[-0.02, -0.36], [0.2, -0.32], [0.95, -0.14], [0.95, 0.06]], n, { crease: 0.4 });
  b.radialRing([[0.95, 0.06], [0.95, 0.24], [0.8, 0.24], [0.8, 0.1], [-0.02, 0.1]], n, { mat: MAT.paving, crease: 0.4 });
  // corbels under it
  for (let k = 0; k < 18; k++) {
    const a = (k / 18) * TAU, da = 0.045;
    corbel(b, a, da, -0.85, -0.32, 0.0, 0.62);
  }
}

/** A radial corbel at bearing a (±da rad), from y0 to y1, standing out from o0 to o1 m (wedge, its face sloping). */
function corbel(b: KitBuilder, a: number, da: number, y0: number, y1: number, o0: number, o1: number): void {
  const st: Style = { mat: MAT.trim };
  const out: V3 = [1, 0.35, 0];
  for (const s of [-1, 1]) {
    // side faces
    const ids = [b.radial(a + s * da, y1, o0, [0, 0, s], 0, y1, st), b.radial(a + s * da, y1, o1, [0, 0, s], 0.8, y1, st), b.radial(a + s * da, y0, o0, [0, 0, s], 0, y0, st)];
    if (s > 0) b.tri(ids[0], ids[1], ids[2]); else b.tri(ids[0], ids[2], ids[1]);
  }
  const f = [b.radial(a - da, y0, o0, out, 0, y0, st), b.radial(a + da, y0, o0, out, 0.1, y0, st), b.radial(a + da, y1, o1, out, 0.1, y1, st), b.radial(a - da, y1, o1, out, 0, y1, st)];
  b.quad(f[0], f[1], f[2], f[3]);
}

/**
 * Battlements on machicolations: a corbel course projecting 0.6 m, a parapet 1.2 m tall, then
 * `merlons` merlons 0.95 m tall with crenels between; walkway inside.
 */
function ringParapet(merlons: number) {
  return (b: KitBuilder, lod: Lod) => {
    b.mat = MAT.stone;
    const n = n3(lod, Math.max(48, merlons * 2), Math.max(16, merlons), 10);
    // corbel course and parapet: outer face, top of the parapet runs inward to the walk
    if (lod === 0) b.radialRing([[-0.02, -0.9], [0.18, -0.6], [0.6, -0.08], [0.62, 0], [0.62, 1.2]], n, { mat: MAT.trim, crease: 0.4 });
    else b.radialRing([[-0.02, -0.9], [0.62, 0], [0.62, 1.2]], n, { mat: MAT.trim, crease: 0.4 });
    b.radialRing(lod === 0 ? [[0.62, 1.2], [0.2, 1.2], [0.2, 0.25], [-0.6, 0.25]] : [[0.62, 1.2], [-0.6, 1.2]], n, { crease: 0.4 });
    if (lod === 2) return;
    if (lod === 0) for (let k = 0; k < merlons; k++) corbel(b, ((k + 0.5) / merlons) * TAU, Math.min(0.06, 1.2 / merlons), -0.9, -0.1, -0.02, 0.6);
    // merlons (only the faces that show from outside and above, far)
    const w = (0.55 * TAU) / merlons / 2;
    for (let k = 0; k < merlons; k++) {
      const a = (k / merlons) * TAU;
      const y0 = 1.2, y1 = 2.15, o0 = 0.2, o1 = 0.62;
      const st: Style = { mat: MAT.stone, uAxis: AXIS.x, vAxis: AXIS.y };
      const P = (da: number, y: number, o: number, nrm: V3, uu: number) => b.radial(a + da, y, o, nrm, uu, y, st);
      // outer, inner, top, two sides
      const q1 = [P(-w, y0, o1, [1, 0, 0], a - w), P(w, y0, o1, [1, 0, 0], a + w), P(w, y1, o1, [1, 0, 0], a + w), P(-w, y1, o1, [1, 0, 0], a - w)];
      b.quad(q1[0], q1[1], q1[2], q1[3]);
      if (lod === 1) {
        const q3 = [P(-w, y1, o0, [0, 1, 0], a - w), P(w, y1, o0, [0, 1, 0], a + w), P(w, y1, o1, [0, 1, 0], a + w), P(-w, y1, o1, [0, 1, 0], a - w)];
        b.quad(q3[0], q3[3], q3[2], q3[1]);
        continue;
      }
      const q2 = [P(-w, y0, o0, [-1, 0, 0], a - w), P(w, y0, o0, [-1, 0, 0], a + w), P(w, y1, o0, [-1, 0, 0], a + w), P(-w, y1, o0, [-1, 0, 0], a - w)];
      b.quad(q2[0], q2[3], q2[2], q2[1]);
      const q3 = [P(-w, y1, o0, [0, 1, 0], a - w), P(w, y1, o0, [0, 1, 0], a + w), P(w, y1, o1, [0, 1, 0], a + w), P(-w, y1, o1, [0, 1, 0], a - w)];
      b.quad(q3[0], q3[3], q3[2], q3[1]);
      for (const s of [-1, 1]) {
        const q = [P(s * w, y0, o0, [0, 0, s], o0), P(s * w, y0, o1, [0, 0, s], o1), P(s * w, y1, o1, [0, 0, s], o1), P(s * w, y1, o0, [0, 0, s], o0)];
        if (s > 0) b.quad(q[0], q[1], q[2], q[3]); else b.quad(q[0], q[3], q[2], q[1]);
      }
    }
  };
}

/** Battlements along a straight wall top: 4 m (x ±2) of parapet with two merlons, 0.9 m thick (z ±0.45). Fixed. */
function crenel(b: KitBuilder, lod: Lod) {
  b.mat = MAT.stone;
  b.box([0, 0.55, 0], [2, 0.55, 0.45], { faces: 0b110111 });
  if (lod === 2) return;
  for (const x of [-1, 1]) b.box([x, 1.55, 0], [0.55, 0.45, 0.45], { faces: 0b110111 });
}

/** A curb round a plaza's rim, 0.3 m tall and 0.45 m wide (inside the radius). */
function ringCurb(b: KitBuilder, lod: Lod) {
  b.radialRing([[0.01, -0.05], [0.01, 0.3], [-0.45, 0.3], [-0.45, 0]], n3(lod, 64, 32, 12), { mat: MAT.trim, crease: 0.4 });
}

/** A rotunda's entablature: architrave, frieze and cornice, 1.6 m, round the column tops (offset 0 at their axis). */
function ringEntablature(b: KitBuilder, lod: Lod) {
  b.mat = MAT.marble;
  const n = n3(lod, 48, 24, 10);
  b.radialRing([[0.55, 0], [0.55, 0.55], [0.6, 0.6], [0.6, 1.1], [0.9, 1.35], [0.9, 1.6], [-0.9, 1.6], [-0.9, 1.35], [-0.6, 1.1], [-0.6, 0.6], [-0.55, 0.55], [-0.55, 0], [0.55, 0]], n, { crease: 0.5 });
}

// ---- details (fixed, metres) ------------------------------------------------------------------

/** A gilded finial, 2.4 m: a knop, a spike, an orb. */
function finial(b: KitBuilder, lod: Lod) {
  const prof: ReadonlyArray<readonly [number, number]> = lod === 0
    ? [[0.3, 0], [0.3, 0.1], [0.18, 0.2], [0.26, 0.38], [0.1, 0.56], [0.14, 0.66], [0.04, 0.8], [0.03, 1.6], [0.1, 1.7], [0.1, 1.8], [0.03, 1.9], [0.012, 2.4]]
    : [[0.3, 0], [0.24, 0.4], [0.05, 0.8], [0.1, 1.75], [0, 2.4]];
  b.lathe(prof, n3(lod, 10, 5, 4), { mat: MAT.gold, metric: true });
}

/** A gothic pinnacle, 4.6 m: a shaft with gablets, an octagonal spirelet, a gilded knob. */
function pinnacle(b: KitBuilder, lod: Lod) {
  b.mat = MAT.trim;
  b.box([0, 1.1, 0], [0.32, 1.1, 0.32], { faces: 0b110111, ao: footAO(0.4) });
  if (lod === 0) {
    for (let k = 0; k < 4; k++) {
      b.push().rotateY((k / 4) * TAU).translate(0, 2.0, 0.24);
      gableShape(b, 0.3, 0.42, 0.12, 0.02, MAT.trim, MAT.trim);
      b.pop();
    }
  }
  b.lathe([[0.42, 2.2], [0.38, 2.32], [0.02, 4.35]], lod === 2 ? 4 : 8, { flatSides: true, phase: Math.PI / 8, mat: MAT.trim });
  if (lod !== 2) b.lathe([[0, 4.3], [0.09, 4.38], [0.09, 4.48], [0, 4.6]], 6, { mat: MAT.gold });
}

/** A bartizan: a small round turret corbelled out at a tower's top, 1.5 m radius, ~8.5 m tall. */
function turret(b: KitBuilder, lod: Lod) {
  const n = n3(lod, 14, 8, 5);
  b.lathe(lod === 0 ? [[0.25, 0], [0.7, 0.5], [1.15, 1.0], [1.5, 1.45]] : [[0.25, 0], [1.5, 1.45]], n, { mat: MAT.trim, metric: true, crease: 0.5 });
  b.frustum(0, 0, 1.45, 4.9, 1.5, 1.5, n, { mat: MAT.stone, facade: FACADE.slit | FACADE.ROUND, metric: true, v0: 1.45 });
  b.lathe(lod === 0 ? [[1.42, 4.85], [1.78, 4.85], [1.78, 4.95], [1.6, 5.1], [1.0, 6.0], [0.4, 7.4], [0, 8.2]] : [[1.78, 4.88], [0.9, 6.2], [0, 8.2]], n, { mat: MAT.roof, uAxis: 3, vAxis: 3, metric: true, crease: 0.5 });
  if (lod !== 2) b.lathe([[0.08, 8.1], [0.08, 8.4], [0.15, 8.55], [0, 8.8]], 6, { mat: MAT.gold });
}

/** A buttress against a wall at z 0 (the wall behind, −z): 1.2 m wide, 2 m deep at its foot, stepping back twice up to y 1 (unit: stretch y). */
function buttress(b: KitBuilder, lod: Lod) {
  b.mat = MAT.stone;
  const steps: Array<[number, number, number]> = [[0, 0.45, 2.0], [0.45, 0.78, 1.45], [0.78, 1.0, 0.9]];
  for (const [y0, y1, d] of steps) {
    b.box([0, (y0 + y1) / 2, d / 2], [0.6, (y1 - y0) / 2, d / 2], { faces: 0b010011, ao: footAO(0.1) });
    // weathering slope on top of each step (a wedge)
    const top = y1, dz = d;
    const nx: V3 = [0, 0.6, 0.8];
    const q = [b.vertex([-0.6, top, dz], nx, -0.6, dz), b.vertex([0.6, top, dz], nx, 0.6, dz), b.vertex([0.6, top + 0.02, 0], nx, 0.6, 0), b.vertex([-0.6, top + 0.02, 0], nx, -0.6, 0)];
    b.quad(q[0], q[1], q[2], q[3]);
  }
  if (lod === 2) return;
}

/** A column (unit: radius 1 at the shaft's foot, height 1): base, a shaft with entasis, a capital. */
function column(b: KitBuilder, lod: Lod) {
  const prof: ReadonlyArray<readonly [number, number]> = [
    [1.5, 0], [1.5, 0.03], [1.32, 0.045], [1.22, 0.06], [1.0, 0.075], [0.97, 0.5], [0.86, 0.9], [0.92, 0.915], [1.2, 0.95], [1.42, 0.965], [1.42, 1.0], [0, 1.0],
  ];
  b.lathe(prof, n3(lod, 14, 8, 5), { mat: MAT.marble, crease: 0.6 });
}

/** A door: an arched stone portal (2.6 × 4.4 m) standing 0.3 m proud of the wall at z 0, timber leaves set back in it. */
function door(b: KitBuilder, lod: Lod) {
  if (lod === 2) { b.box([0, 2.0, 0.1], [1.2, 2.0, 0.12], { mat: MAT.dark, faces: 0b010111 }); return; }
  b.push().translate(0, 0, 0.15);
  b.archWall({ w: 1.75, y0: 0, y1: 4.6, t: 0.32, open: 1.2, spring: 2.6, crown: 3.9, pointed: true, segs: lod === 0 ? 10 : 6, keystone: lod === 0, style: { mat: MAT.trim } });
  b.pop();
  // the leaves, set back
  b.push().translate(0, 0, -0.05);
  const st: Style = { mat: MAT.timber, uAxis: AXIS.x, vAxis: AXIS.y };
  const pts: Array<[number, number]> = [[-1.2, 0], [1.2, 0]];
  const segs = lod === 0 ? 10 : 6;
  for (let i = segs; i >= 0; i--) {
    // (the arch's outline, right to left, closes the leaves' top)
    const t = i / segs;
    pts.push(archPt(t, 1.2, 2.6, 1.3));
  }
  const c = b.vertex([0, 1.6, 0], [0, 0, 1], 0, 1.6, st);
  const ids = pts.map(([x, y]) => b.vertex([x, y, 0], [0, 0, 1], x, y, st));
  for (let i = 0; i + 1 < ids.length; i++) b.tri(c, ids[i], ids[i + 1]);
  b.tri(c, ids[ids.length - 1], ids[0]);
  b.pop();
}
function archPt(t: number, w: number, spring: number, h: number): [number, number] {
  // pointed: two arcs (mesh.ts archCurve, inlined for the door's leaves)
  if (h <= w) { const a = Math.PI * (1 - t); return [Math.cos(a) * w, spring + Math.sin(a) * h]; }
  const c = (h * h - w * w) / (2 * w), R = c + w;
  if (t < 0.5) { const a1 = Math.atan2(h, -c); const a = Math.PI + (a1 - Math.PI) * (t / 0.5); return [c + R * Math.cos(a), spring + R * Math.sin(a)]; }
  const a0 = Math.atan2(h, c); const a = a0 * (1 - (t - 0.5) / 0.5); return [-c + R * Math.cos(a), spring + R * Math.sin(a)];
}

/** A lantern post, 3.4 m: a plinth, an octagonal iron post, a sunstone lantern in a glass cage. */
function lanternPost(b: KitBuilder, lod: Lod) {
  if (lod === 2) { b.box([0, 3.0, 0], [0.2, 0.26, 0.2], { mat: MAT.lantern, faces: 0b111111 }); return; }
  b.box([0, 0.15, 0], [0.28, 0.15, 0.28], { mat: MAT.trim, faces: 0b110111 });
  b.lathe(lod === 0 ? [[0.11, 0.3], [0.08, 0.5], [0.07, 2.55], [0.12, 2.65], [0.16, 2.7]] : [[0.1, 0.3], [0.07, 2.6]], lod === 0 ? 8 : 4, { mat: MAT.iron, flatSides: true });
  b.box([0, 3.0, 0], [0.2, 0.28, 0.2], { mat: MAT.lantern, faces: 0b111111 });
  b.lathe([[0.28, 3.28], [0.28, 3.31], [0, 3.55]], 4, { mat: MAT.iron, flatSides: true, phase: Math.PI / 4 });
  if (lod === 0) {
    for (let k = 0; k < 4; k++) {
      const a = (k / 4) * TAU + Math.PI / 4;
      b.box([Math.cos(a) * 0.27, 3.0, Math.sin(a) * 0.27], [0.025, 0.3, 0.025], { mat: MAT.iron, faces: 0b110011 });
    }
    b.lathe([[0.04, 3.55], [0.07, 3.62], [0, 3.72]], 5, { mat: MAT.gold });
  }
}

/** A hanging lantern, 1.6 m from its hook (y 0) down: a chain, a cage, a sunstone. */
function lanternHang(b: KitBuilder, lod: Lod) {
  if (lod === 2) { b.box([0, -1.25, 0], [0.18, 0.24, 0.18], { mat: MAT.lantern }); return; }
  b.box([0, -0.45, 0], [0.025, 0.45, 0.025], { mat: MAT.iron, faces: 0b110011 });
  b.lathe([[0, -0.9], [0.22, -0.98], [0.25, -1.02]], 6, { mat: MAT.iron });
  b.lathe([[0.2, -1.02], [0.2, -1.48]], 6, { mat: MAT.lantern, flatSides: true });
  b.lathe([[0.24, -1.5], [0.12, -1.6], [0, -1.62]], 6, { mat: MAT.iron });
}

/** A banner on its pole: an 8 m mast, a crossbar, a swallow-tailed cloth 1.6 × 4.6 m that sways in the wind. */
function banner(b: KitBuilder, lod: Lod) {
  b.lathe([[0.09, 0], [0.07, 8.0], [0.12, 8.1], [0.05, 8.35]], lod === 2 ? 4 : 6, { mat: MAT.iron });
  if (lod !== 2) b.box([0, 7.7, 0.0], [0.95, 0.05, 0.05], { mat: MAT.iron, faces: 0b110011 });
  const rows = n3(lod, 10, 5, 2), cols = lod === 2 ? 1 : 2;
  const top = 7.65, hgt = 4.6, w = 0.8;
  const st: Style = { mat: MAT.cloth, anim: ANIM.cloth, uAxis: AXIS.x, vAxis: AXIS.y };
  const ids: number[][] = [];
  for (let i = 0; i <= rows; i++) {
    const row: number[] = [];
    const t = i / rows;
    for (let j = 0; j <= cols; j++) {
      const s = j / cols;
      // swallowtail: the middle of the hem rises toward the end
      const notch = t === 1 ? 0.9 * (1 - Math.abs(2 * s - 1)) : 0;
      const y = top - hgt * t + notch;
      row.push(b.vertex([-w + 2 * w * s, y, 0.06], [0, 0, 1], -w + 2 * w * s, top - y, st));
    }
    ids.push(row);
  }
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) b.quad(ids[i][j], ids[i + 1][j], ids[i + 1][j + 1], ids[i][j + 1]);
}

/** A pennant streaming from a 3 m mast's top: a long tapering tail 5.5 m, waving. */
function pennant(b: KitBuilder, lod: Lod) {
  b.lathe([[0.06, 0], [0.045, 3.0], [0.09, 3.06], [0, 3.2]], 5, { mat: MAT.gold });
  const segs = n3(lod, 12, 6, 2);
  const st: Style = { mat: MAT.cloth, anim: ANIM.cloth, uAxis: AXIS.x, vAxis: AXIS.y };
  const ids: Array<[number, number]> = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const h = 0.55 * (1 - t) + 0.02;
    // (its surface coordinates: across it, then the metres from the mast it streams: how far it sways)
    ids.push([b.vertex([5.5 * t, 2.95, 0], [0, 0, 1], 0, 5.5 * t, st), b.vertex([5.5 * t, 2.95 - h, 0], [0, 0, 1], h, 5.5 * t, st)]);
  }
  for (let i = 0; i < segs; i++) b.quad(ids[i][0], ids[i + 1][0], ids[i + 1][1], ids[i][1]);
}

/** A Warden: an ancestor in stone, robed, wings raised and spread, on a plinth — 6.6 m in all. */
function warden(b: KitBuilder, lod: Lod) {
  b.mat = MAT.marble;
  b.box([0, 0.75, 0], [0.95, 0.75, 0.95], { faces: 0b110111, mat: MAT.trim, ao: footAO(0.5) });
  b.box([0, 1.58, 0], [1.08, 0.08, 1.08], { faces: 0b111111, mat: MAT.trim });
  const n = n3(lod, 12, 7, 5);
  // robe and body
  b.lathe([[0.75, 1.66], [0.62, 2.4], [0.42, 3.6], [0.48, 4.0], [0.5, 4.2], [0.34, 4.45], [0.16, 4.55]], n, { crease: 0.7 });
  // head, beak
  b.lathe([[0, 4.5], [0.2, 4.58], [0.3, 4.78], [0.28, 4.98], [0.16, 5.12], [0, 5.16]], n, {});
  if (lod !== 2) {
    b.push().translate(0, 4.8, 0.26).rotateX(Math.PI / 2 - 0.25);
    b.lathe([[0.1, 0], [0.04, 0.32], [0, 0.4]], 5, {});
    b.pop();
  }
  // wings: raised and spread, a feathered trailing edge
  const fe = n3(lod, 7, 4, 2);
  for (const s of [1, -1]) {
    const lead: V3[] = [], trail: V3[] = [];
    for (let i = 0; i <= fe; i++) {
      const t = i / fe;
      const x = s * (0.35 + 2.3 * t);
      const yl = 4.2 + 1.5 * Math.sin(t * Math.PI * 0.6) + 0.4 * t;
      lead.push([x, yl, -0.25 - 0.2 * t]);
      const feather = i % 2 === 0 ? 0 : 0.35;
      trail.push([x * (0.98 - 0.08 * t), 3.0 + 1.1 * t + feather - (t > 0.8 ? 0.4 * (t - 0.8) * 5 : 0), -0.45 - 0.15 * t]);
    }
    const nrm: V3 = [0, 0, -1];
    for (let i = 0; i < fe; i++) {
      const q = [b.vertex(lead[i], nrm, lead[i][0], lead[i][1]), b.vertex(lead[i + 1], nrm, lead[i + 1][0], lead[i + 1][1]),
        b.vertex(trail[i + 1], nrm, trail[i + 1][0], trail[i + 1][1]), b.vertex(trail[i], nrm, trail[i][0], trail[i][1])];
      b.quad(q[0], q[1], q[2], q[3]);
      // the front face (thin wing: both sides)
      const f = [b.vertex(lead[i], [0, 0, 1], lead[i][0], lead[i][1]), b.vertex(lead[i + 1], [0, 0, 1], lead[i + 1][0], lead[i + 1][1]),
        b.vertex(trail[i + 1], [0, 0, 1], trail[i + 1][0], trail[i + 1][1]), b.vertex(trail[i], [0, 0, 1], trail[i][0], trail[i][1])];
      b.quad(f[0], f[3], f[2], f[1]);
    }
  }
}

/** A tiered fountain, 7 m across: an outer basin of still water, a pedestal, an upper bowl, a spout. */
function fountain(b: KitBuilder, lod: Lod) {
  const n = n3(lod, 32, 16, 8);
  b.lathe([[3.55, -0.2], [3.55, 0.6], [3.7, 0.65], [3.7, 0.75], [3.3, 0.75], [3.3, 0.35]], n, { mat: MAT.trim, metric: true, crease: 0.5 });
  b.disc(0, 0, 0.5, 3.3, n, true, { mat: MAT.water });
  b.lathe([[0.7, 0.4], [0.55, 0.8], [0.4, 1.9], [0.6, 2.05], [1.3, 2.25], [1.4, 2.5], [1.2, 2.5]], n, { mat: MAT.marble, metric: true, crease: 0.6 });
  b.disc(0, 0, 2.42, 1.2, n, true, { mat: MAT.water });
  if (lod !== 2) b.lathe([[0.25, 2.4], [0.18, 2.9], [0.32, 3.2], [0.1, 3.5], [0, 3.6]], 8, { mat: MAT.bronze });
}

/** A freestanding gate-arch: two piers and a pointed arch, 6.4 × 6.8 m, 1.8 m deep, a cornice, pinnacles. */
function gateArch(b: KitBuilder, lod: Lod) {
  b.archWall({ w: 3.2, y0: 0, y1: 6.8, t: 0.9, open: 1.9, spring: 3.4, crown: 5.4, pointed: true, segs: n3(lod, 14, 8, 4), keystone: lod !== 2, feet: false, style: { mat: MAT.stone } });
  b.box([0, 7.0, 0], [3.45, 0.2, 1.1], { mat: MAT.trim, faces: 0b111111 });
  if (lod === 0) {
    for (const s of [-1, 1]) {
      b.push().translate(s * 2.75, 7.2, 0).scale(0.8);
      pinnacle(b, 1);
      b.pop();
    }
  }
}

// ---- bridges -----------------------------------------------------------------------------------

/**
 * A bridge span between two pier centres (x ±1): the deck's top at y 0, its spandrels and a round
 * (or pointed) arch springing at y −0.62 from the half-piers at its ends, a keystone at the crown.
 * Unit: stretch x by half the span, y by the depth to the springing / 0.62, z by half the width.
 */
function bridgeSpan(pointed: boolean) {
  return (b: KitBuilder, lod: Lod) => {
    b.archWall({
      w: 1, y0: -0.62, y1: 0, t: 1, open: 0.8, spring: -0.62, crown: pointed ? -0.1 : -0.16, pointed,
      segs: n3(lod, 16, 8, 4), keystone: lod !== 2, top: false, feet: false, style: { mat: MAT.stone },
    });
    b.box([0, 0, 0], [1, 0, 1], { faces: 4, mat: MAT.paving });
  };
}

/**
 * A hanging pier under a span's springing (y 0) down to a point at y −1: an octagonal shaft with
 * cutwater edges, a moulded band, then the plumb-stone — skystone that glows faintly at night.
 */
function bridgePier(b: KitBuilder, lod: Lod) {
  const sides = lod === 2 ? 4 : 8;
  const ph = Math.PI / 8;
  b.lathe([[1.0, 0.0], [1.04, -0.52], [1.12, -0.55], [1.12, -0.6], [0.95, -0.64], [0.7, -0.78]], sides, { mat: MAT.stone, flatSides: true, phase: ph, crease: 0.4 });
  b.lathe([[0.7, -0.78], [0.4, -0.9], [0.14, -0.97], [0, -1]], sides, { mat: MAT.keystone, flatSides: true, phase: ph, crease: 0.4 });
}

/** A parapet: a low wall (x ±1 along it, height y 0 … 1, z ±1 across) with a coping. Stretch. */
function parapet(b: KitBuilder, lod: Lod) {
  b.box([0, 0.44, 0], [1, 0.44, 1], { mat: MAT.stone, faces: 0b111111 & ~8, ao: footAO(0.2) });
  b.box([0, 0.94, 0], [1, 0.06, 1.25], { mat: MAT.trim, faces: lod === 2 ? 0b110100 : 0b111111 });
}

/** A balustrade 2 m long (x ±1), 1.05 m tall: rails and five vase balusters. */
function balustrade(b: KitBuilder, lod: Lod) {
  b.box([0, 0.08, 0], [1, 0.08, 0.17], { mat: MAT.trim, faces: 0b110111 });
  b.box([0, 0.98, 0], [1, 0.07, 0.19], { mat: MAT.trim, faces: 0b111111 });
  if (lod === 2) { b.box([0, 0.55, 0], [1, 0.4, 0.06], { mat: MAT.trim, faces: 0b110011 }); return; }
  for (let k = 0; k < 5; k++) {
    b.push().translate(-0.8 + 0.4 * k, 0.16, 0);
    b.lathe([[0.07, 0], [0.11, 0.12], [0.06, 0.3], [0.11, 0.55], [0.06, 0.7], [0.08, 0.75]], lod === 0 ? 8 : 4, { mat: MAT.trim });
    b.pop();
  }
}

/** A straight flight of n steps (x ±1 across, rising y 0 → 1 over z 0 → 1). */
function stair(n: number) {
  return (b: KitBuilder) => b.stairs(n, -1, 1, 0, 1, 0, 1, { mat: MAT.stone, tread: MAT.paving });
}

// ---- the catalogue ---------------------------------------------------------------------------------

export const PIECES: readonly PieceDef[] = [
  { name: 'body.round', kind: 'stretch', detail: 2, build: bodyRound(FACADE.roost) },
  { name: 'body.roundHall', kind: 'stretch', detail: 2, build: bodyRound(FACADE.hall) },
  { name: 'body.roundBlind', kind: 'stretch', detail: 2, build: bodyRound(FACADE.none) },
  { name: 'body.square', kind: 'stretch', detail: 2, build: bodySquare(FACADE.roost) },
  { name: 'body.hall', kind: 'stretch', detail: 2, build: bodySquare(FACADE.lancet) },
  { name: 'body.front', kind: 'stretch', detail: 2, build: bodySquare(FACADE.rose) },
  { name: 'body.wall', kind: 'stretch', detail: 2, build: bodySquare(FACADE.slit) },
  { name: 'body.arcade', kind: 'stretch', detail: 2, build: bodySquare(FACADE.arcade) },
  { name: 'body.belfry', kind: 'stretch', detail: 2, build: bodySquare(FACADE.belfry) },
  { name: 'body.lattice', kind: 'stretch', detail: 2, build: bodySquare(FACADE.lattice) },
  { name: 'body.roundBelfry', kind: 'stretch', detail: 2, build: bodyRound(FACADE.belfry) },
  { name: 'body.blind', kind: 'stretch', detail: 2, build: bodySquare(FACADE.none) },
  { name: 'body.octagon', kind: 'stretch', detail: 2, build: bodyOctagon(FACADE.hall) },
  { name: 'podium', kind: 'stretch', detail: 2, build: podium },
  { name: 'plaza.round', kind: 'stretch', detail: 2, build: plazaRound },
  { name: 'roof.cone', kind: 'stretch', detail: 2, build: roofCone },
  { name: 'roof.spire', kind: 'stretch', detail: 2, build: roofSpire },
  { name: 'roof.pyramid', kind: 'stretch', detail: 2, build: roofPyramid },
  { name: 'roof.gable', kind: 'stretch', detail: 2, build: roofGable },
  { name: 'roof.dome', kind: 'stretch', detail: 2, build: roofDome },
  { name: 'roof.onion', kind: 'stretch', detail: 2, build: roofOnion },
  { name: 'roof.glass', kind: 'stretch', detail: 2, build: roofGlass },
  { name: 'ring.cornice', kind: 'radial', detail: 1, build: ringCornice },
  { name: 'ring.ledge', kind: 'radial', detail: 1, build: ringLedge },
  { name: 'ring.parapet12', kind: 'radial', detail: 1, build: ringParapet(12) },
  { name: 'ring.parapet20', kind: 'radial', detail: 1, build: ringParapet(20) },
  { name: 'ring.parapet32', kind: 'radial', detail: 1, build: ringParapet(32) },
  { name: 'ring.curb', kind: 'radial', detail: 1, build: ringCurb },
  { name: 'crenel', kind: 'fixed', detail: 1, build: crenel },
  { name: 'ring.entablature', kind: 'radial', detail: 2, build: ringEntablature },
  { name: 'finial', kind: 'fixed', detail: 1, build: finial },
  { name: 'pinnacle', kind: 'fixed', detail: 1, build: pinnacle },
  { name: 'turret', kind: 'fixed', detail: 1, build: turret },
  { name: 'buttress', kind: 'stretch', detail: 1, build: buttress },
  { name: 'column', kind: 'stretch', detail: 1, build: column },
  { name: 'door', kind: 'fixed', detail: 0, build: door },
  { name: 'lantern.post', kind: 'fixed', detail: 1, build: lanternPost },
  { name: 'lantern.hang', kind: 'fixed', detail: 1, build: lanternHang },
  { name: 'banner', kind: 'fixed', detail: 1, build: banner },
  { name: 'pennant', kind: 'fixed', detail: 1, build: pennant },
  { name: 'warden', kind: 'fixed', detail: 1, build: warden },
  { name: 'fountain', kind: 'fixed', detail: 1, build: fountain },
  { name: 'gate.arch', kind: 'fixed', detail: 1, build: gateArch },
  { name: 'bridge.span', kind: 'stretch', detail: 2, build: bridgeSpan(false) },
  { name: 'bridge.spanPointed', kind: 'stretch', detail: 2, build: bridgeSpan(true) },
  { name: 'bridge.pier', kind: 'stretch', detail: 2, build: bridgePier },
  { name: 'parapet', kind: 'stretch', detail: 1, build: parapet },
  { name: 'balustrade', kind: 'fixed', detail: 0, build: balustrade },
  { name: 'stair.6', kind: 'stretch', detail: 1, build: stair(6) },
  { name: 'stair.12', kind: 'stretch', detail: 1, build: stair(12) },
  { name: 'stair.24', kind: 'stretch', detail: 1, build: stair(24) },
  // the classes' own (artefacts.ts)
  { name: 'halo.ring', kind: 'fixed', detail: 2, build: haloRing },
  { name: 'runestone', kind: 'fixed', detail: 1, build: runestone },
  { name: 'runestone.float', kind: 'fixed', detail: 2, build: (b, lod) => { b.anim = ANIM.bob; runestone(b, lod); } },
  { name: 'crystal.cluster', kind: 'fixed', detail: 2, build: crystalCluster },
  { name: 'roof.pagoda', kind: 'stretch', detail: 2, build: roofPagoda },
  { name: 'gate.torii', kind: 'fixed', detail: 1, build: gateTorii },
  { name: 'lantern.paper', kind: 'fixed', detail: 0, build: lanternPaper },
  { name: 'bridge.rope', kind: 'stretch', detail: 1, build: bridgeRope },
  { name: 'dock', kind: 'stretch', detail: 1, build: dock },
  { name: 'crane', kind: 'fixed', detail: 1, build: crane },
  { name: 'awning', kind: 'fixed', detail: 0, build: awning },
  { name: 'stall', kind: 'fixed', detail: 0, build: stall },
  { name: 'cargo', kind: 'fixed', detail: 0, build: cargo },
  { name: 'skyship', kind: 'fixed', detail: 2, build: skyship },
  { name: 'skyship.moored', kind: 'fixed', detail: 2, build: (b, lod) => skyship(b, lod, false) },
  { name: 'cloud.comb', kind: 'fixed', detail: 1, build: cloudComb },
  { name: 'wind.sails', kind: 'fixed', detail: 1, build: windSails },
  { name: 'chimney', kind: 'stretch', detail: 2, build: chimney },
  { name: 'brazier', kind: 'fixed', detail: 0, build: brazier },
  { name: 'bough.tree', kind: 'fixed', detail: 2, build: boughTree },
  { name: 'bough.roots', kind: 'stretch', detail: 2, build: boughRoots },
  { name: 'nest.platform', kind: 'fixed', detail: 1, build: nestPlatform },
  { name: 'ruin.top', kind: 'radial', detail: 1, build: ruinTop },
  { name: 'column.fallen', kind: 'fixed', detail: 1, build: columnFallen },
  { name: 'rubble', kind: 'fixed', detail: 1, build: rubble },
  // life in the water (water.ts)
  { name: 'lily.pad', kind: 'fixed', detail: 2, build: lilyPad },
  { name: 'lotus', kind: 'fixed', detail: 1, build: lotus },
  { name: 'reeds', kind: 'fixed', detail: 1, build: reeds },
  { name: 'waterweed', kind: 'fixed', detail: 0, build: waterweed },
  { name: 'koi', kind: 'fixed', detail: 0, build: koi },
  { name: 'aqueduct', kind: 'stretch', detail: 1, build: aqueduct },
  { name: 'spout.fall', kind: 'fixed', detail: 0, build: spoutFall },
  { name: 'aqueduct.fall', kind: 'stretch', detail: 1, build: aqueductFall },
  // insides (interiors.ts)
  { name: 'storey.round', kind: 'stretch', detail: 2, build: storeyRound },
  { name: 'storey.square', kind: 'stretch', detail: 2, build: storeySquare },
  { name: 'door.open', kind: 'fixed', detail: 1, build: doorOpen },
  { name: 'jars', kind: 'fixed', detail: 0, build: jars },
  { name: 'hearth', kind: 'fixed', detail: 0, build: hearth },
  { name: 'loom', kind: 'fixed', detail: 0, build: loom },
  { name: 'nest.bowl', kind: 'fixed', detail: 0, build: nestBowl },
  { name: 'rug', kind: 'fixed', detail: 0, build: rug },
  { name: 'table', kind: 'fixed', detail: 0, build: table },
  { name: 'perch', kind: 'fixed', detail: 0, build: perch },
  { name: 'shelves', kind: 'fixed', detail: 0, build: shelves },
  { name: 'ramp.spiral', kind: 'radial', detail: 0, build: rampSpiral },
  { name: 'body.hallOpen', kind: 'stretch', detail: 2, build: bodyHallOpen },
  { name: 'table.long', kind: 'fixed', detail: 0, build: tableLong },
  { name: 'gallery', kind: 'stretch', detail: 1, build: gallery },
  { name: 'altar', kind: 'fixed', detail: 1, build: altar },
  { name: 'body.apse', kind: 'stretch', detail: 2, build: bodyApse },
  { name: 'roof.apse', kind: 'stretch', detail: 2, build: roofApse },
  { name: 'storey.roundOpen', kind: 'stretch', detail: 2, build: storeyRoundOpen },
  { name: 'body.roundOpen', kind: 'stretch', detail: 2, build: bodyRoundOpen },
  { name: 'tiers', kind: 'radial', detail: 0, build: tiers },
  { name: 'orrery', kind: 'fixed', detail: 0, build: orrery },
  { name: 'telescope', kind: 'fixed', detail: 0, build: telescope },
  { name: 'planter', kind: 'fixed', detail: 0, build: planter },
  { name: 'storey.lattice', kind: 'stretch', detail: 2, build: storeyLattice },
  { name: 'storey.octagon', kind: 'stretch', detail: 2, build: storeyOctagon },
  { name: 'ramp.slab', kind: 'radial', detail: 0, build: rampSlab },
  { name: 'pool.round', kind: 'stretch', detail: 1, build: poolRound },
  { name: 'rod.measure', kind: 'fixed', detail: 0, build: measuringRod },
  { name: 'anvil', kind: 'fixed', detail: 0, build: anvil },
  { name: 'furnace', kind: 'fixed', detail: 0, build: furnace },
  { name: 'shrine.small', kind: 'fixed', detail: 0, build: shrineSmall },
];

/** A piece's index by name (throws on a typo, at start-up). */
const PIECE: Record<string, number> = Object.fromEntries(PIECES.map((p, i) => [p.name, i]));
export function pieceId(name: string): number {
  const k = PIECE[name];
  if (k === undefined) throw new Error(`kit: no piece named ${name}`);
  return k;
}

