/**
 * Life in the Eyries' still water (SKY_CIVILIZATION.md §9, the lagoons and
 * their gardens): lily pads and the lotus flowering on them, reeds and their
 * cattails along the shore, waterweed swaying on the bed, and the koi the
 * Aelari keep, swimming their slow circles under the surface (ANIM.swim). All
 * fixed pieces, in metres; catalogued with the rest in pieces.ts.
 *
 * The leaf material's facade byte says which green: 0 a pad's (waxy, its veins
 * radiating), 1 a reed's (paling up the blade), 2 waterweed's (dark, its tips
 * lighter), 3 a cattail's brown head. The petal material's: 0 a petal (white,
 * flushing pink to the tip, v 0 … 1 along it), 1 the seed pod's gold.
 */

import { ANIM, AXIS, KitBuilder, MAT, rng, type Style, type V3 } from './mesh';

type Lod = 0 | 1 | 2;
const n3 = (lod: Lod, a: number, b: number, c: number) => (lod === 0 ? a : lod === 1 ? b : c);
const TAU = Math.PI * 2;

/** A lily pad: a disc 0.5 m in radius with its notch, lying on the water (its top at y 0.01), its rim a little raised. */
export function lilyPad(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 18, 10, 6);
  const notch = 0.32; // rad
  const st: Style = { mat: MAT.leaf, facade: 0, uAxis: AXIS.x, vAxis: AXIS.z };
  const c = b.vertex([0, 0.01, 0], [0, 1, 0], 0, 0, st);
  const rim: number[] = [];
  for (let k = 0; k <= n; k++) {
    const a = notch / 2 + (k / n) * (TAU - notch);
    const r = 0.5 * (1 + 0.03 * Math.sin(5 * a));
    rim.push(b.vertex([Math.cos(a) * r, 0.025, Math.sin(a) * r], [0, 1, 0], Math.cos(a) * r, Math.sin(a) * r, st));
  }
  // (the notch's two edges meet the middle)
  for (let k = 0; k < n; k++) b.tri(c, rim[k + 1], rim[k]);
}

/**
 * A lotus in flower, standing on its pad (0.3 m across, 0.22 m tall): three rings of cupped petals
 * opening wider ring by ring round a golden seed pod; far off, a closed bud.
 */
export function lotus(b: KitBuilder, lod: Lod): void {
  const petal: Style = { mat: MAT.petal, facade: 0, uAxis: AXIS.x, vAxis: AXIS.y };
  if (lod === 2) {
    b.lathe([[0.02, 0], [0.09, 0.06], [0.07, 0.15], [0.0, 0.2]], 5, { ...petal, metric: true });
    return;
  }
  const r = rng(4242);
  const rings: Array<[number, number, number, number]> = lod === 0
    ? [[6, 0.35, 0.12, 0.035], [8, 0.75, 0.15, 0.045], [8, 1.15, 0.16, 0.05]]  // petals, tilt from upright, length, half width
    : [[6, 0.55, 0.13, 0.04], [7, 1.05, 0.16, 0.05]];
  rings.forEach(([count, tilt, L, w], ring) => {
    for (let k = 0; k < count; k++) {
      const phi = ((k + 0.5 * ring) / count) * TAU + 0.15 * r();
      const t = tilt + 0.1 * (r() - 0.5);
      const out: V3 = [Math.cos(phi), 0, Math.sin(phi)], across: V3 = [-Math.sin(phi), 0, Math.cos(phi)];
      // along the petal: base, middle, tip; its edges cupped up toward the middle line
      const pts = [0, 0.5, 1].map((f) => {
        const bend = t * (0.7 + 0.6 * f); // curling outward toward its tip
        const rr = 0.03 + L * f * Math.sin(bend), y = 0.02 + L * f * Math.cos(bend);
        return { c: [out[0] * rr, y, out[2] * rr] as V3, hw: w * (f === 0 ? 0.45 : f === 0.5 ? 1 : 0.25), f };
      });
      const nrm: V3 = [out[0] * Math.cos(t), -Math.sin(t) * 0.3 + 0.7, out[2] * Math.cos(t)];
      const row = pts.map(({ c, hw, f }) => [-1, 0, 1].map((s) => {
        const cup = 0.35 * hw * s * s; // edges raised toward the axis
        const p: V3 = [c[0] + across[0] * s * hw - out[0] * cup, c[1] + cup * 0.6, c[2] + across[2] * s * hw - out[2] * cup];
        return b.vertex(p, nrm, s * hw, f, petal);
      }));
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) b.quad(row[i][j], row[i][j + 1], row[i + 1][j + 1], row[i + 1][j]);
    }
  });
  b.lathe([[0.0, 0.07], [0.035, 0.075], [0.04, 0.1], [0.0, 0.105]], n3(lod, 8, 6, 4), { mat: MAT.petal, facade: 1, metric: true });
}

/** A blade bent along its height: a tapering strip of `segs` segments, v the height (m) for the sway. */
function blade(b: KitBuilder, x: number, z: number, h: number, w: number, lean: number, dir: number, segs: number, st: Style, wave = 0): void {
  const across: V3 = [-Math.sin(dir), 0, Math.cos(dir)];
  const rows: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const f = i / segs;
    const y = h * f;
    const off = lean * f * f * h + wave * Math.sin(f * 7.0 + dir) * 0.06;
    const cx = x + Math.cos(dir) * off, cz = z + Math.sin(dir) * off;
    const hw = w * (1 - 0.85 * f);
    const nrm: V3 = [Math.cos(dir), 0.25, Math.sin(dir)];
    rows.push([-1, 1].map((s) => b.vertex([cx + across[0] * s * hw, y, cz + across[2] * s * hw], nrm, s * hw, y, st)));
  }
  for (let i = 0; i < segs; i++) b.quad(rows[i][0], rows[i][1], rows[i + 1][1], rows[i + 1][0]);
}

/** Reeds in a clump by the shore (1.2–2 m), a few with a cattail's brown head; they sway (v the height). */
export function reeds(b: KitBuilder, lod: Lod): void {
  const r = rng(515);
  const st: Style = { mat: MAT.leaf, facade: 1, uAxis: AXIS.x, vAxis: AXIS.y, anim: ANIM.sway };
  const count = n3(lod, 13, 7, 4), segs = n3(lod, 3, 2, 1);
  for (let k = 0; k < count; k++) {
    const a = r() * TAU, d = Math.sqrt(r()) * 0.45;
    const h = 1.2 + 0.8 * r();
    const x = Math.cos(a) * d, z = Math.sin(a) * d;
    blade(b, x, z, h, 0.022 + 0.012 * r(), 0.08 + 0.12 * r(), a, segs, st);
    if (lod === 0 && k % 4 === 0) {
      // a cattail on its own stem, its head 0.2 m of brown
      const top = h * 0.95;
      b.frustum(x, z, top - 0.24, top, 0.018, 0.016, 5, { mat: MAT.leaf, facade: 3, anim: ANIM.sway, metric: true });
    }
  }
}

/** Waterweed on a lake's bed: ribbons 0.8–2.2 m long, waving as they rise, swaying in the water (v the height). */
export function waterweed(b: KitBuilder, lod: Lod): void {
  const r = rng(616);
  const st: Style = { mat: MAT.leaf, facade: 2, uAxis: AXIS.x, vAxis: AXIS.y, anim: ANIM.cloth };
  const count = n3(lod, 8, 5, 3), segs = n3(lod, 5, 3, 2);
  for (let k = 0; k < count; k++) {
    const a = r() * TAU, d = Math.sqrt(r()) * 0.5;
    blade(b, Math.cos(a) * d, Math.sin(a) * d, 0.8 + 1.4 * r(), 0.05 + 0.04 * r(), 0.15 * r(), a, segs, st, 1);
  }
}

/**
 * A koi, 0.62 m long, facing +z: a spindle of a body (its section a little narrower than tall), a
 * forked tail fin, a dorsal fin and two pectoral fins. It swims (ANIM.swim): round a slow circle
 * a few metres across, its tail beating; its colours are its own (the koi material, by its seed).
 */
export function koi(b: KitBuilder, lod: Lod): void {
  const st: Style = { mat: MAT.koi, uAxis: AXIS.x, vAxis: AXIS.z, anim: ANIM.swim };
  const n = n3(lod, 10, 7, 4);
  // the body: a lathe round y, turned to lie along z and squeezed side to side
  b.push().rotateX(Math.PI / 2).scale(0.8, 1, 1);
  // (the lathe's +y turns to +z: the head)
  b.lathe([[0.0, -0.31], [0.03, -0.27], [0.05, -0.17], [0.072, -0.02], [0.075, 0.08], [0.06, 0.2], [0.035, 0.27], [0.0, 0.31]], n, { ...st, metric: true });
  b.pop();
  if (lod === 2) return;
  const fin = (pts: V3[], nrm: V3) => {
    const v = pts.map((p) => b.vertex(p, nrm, p[0], p[2], st));
    for (let k = 1; k + 1 < v.length; k++) b.tri(v[0], v[k], v[k + 1]);
  };
  // the tail: a fork in the body's vertical plane
  fin([[0, 0, -0.27], [0, 0.11, -0.43], [0, 0.02, -0.37], [0, -0.02, -0.37], [0, -0.11, -0.43]], [1, 0, 0]);
  // dorsal
  fin([[0, 0.06, 0.08], [0, 0.11, -0.02], [0, 0.065, -0.12]], [1, 0, 0]);
  // pectorals, out and back
  for (const s of [-1, 1]) fin([[s * 0.05, -0.03, 0.12], [s * 0.13, -0.05, 0.04], [s * 0.05, -0.04, 0.04]], [0, 1, 0]);
}

// ---- aqueducts (SKY_CIVILIZATION.md §8: on the vow-bridges, water from the cisterns) -------------------

/**
 * An aqueduct channel along a bridge span (stretch: x ±1 along it; y and z in metres, its instance's
 * y and z scale 1): a stone trough 1.6 m wide, its walls 0.84 m high, its water 0.62 m up flowing
 * along +x (the water material's facade 1: ripples carried downstream).
 */
export function aqueduct(b: KitBuilder, lod: Lod): void {
  b.box([0, 0.08, 0], [1, 0.08, 0.8], { mat: MAT.stone });
  for (const s of [-1, 1]) b.box([0, 0.42, s * 0.7], [1, 0.42, 0.1], { mat: MAT.trim, faces: lod === 2 ? 0b110100 : 0b111111 });
  const w: Style = { mat: MAT.water, facade: 1, uAxis: AXIS.x, vAxis: AXIS.z };
  const ids = ([[-1, -0.6], [1, -0.6], [1, 0.6], [-1, 0.6]] as const).map(([x, z]) => b.vertex([x, 0.62, z], [0, 1, 0], x, z, w));
  b.quad(ids[0], ids[3], ids[2], ids[1]);
}

/**
 * A channel's spout and the water pouring from it (fixed, metres): a stone lip 0.6 m out over the
 * drop at +x, and the sheet of water arcing out 0.9 m and down 1.1 m (facade 2: falling, white where
 * it breaks).
 */
export function spoutFall(b: KitBuilder, lod: Lod): void {
  b.box([0.3, 0.6, 0], [0.32, 0.08, 0.36], { mat: MAT.trim });
  const n = n3(lod, 8, 5, 3);
  const w: Style = { mat: MAT.water, facade: 2, uAxis: AXIS.z, vAxis: AXIS.y };
  const rows: number[][] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const x = 0.6 + 0.9 * t, y = 0.66 - 1.1 * t * t, half = 0.3 - 0.06 * t;
    const nrm: V3 = [1, 0.4 + t, 0];
    rows.push([b.vertex([x, y, -half], nrm, -half, t * 1.6, w), b.vertex([x, y, half], nrm, half, t * 1.6, w)]);
  }
  for (let i = 0; i < n; i++) b.quad(rows[i][0], rows[i + 1][0], rows[i + 1][1], rows[i][1]);
}

/**
 * The water falling from one span's channel into the next, lower one (stretch: x 0 → 1 its reach,
 * y 0 → −1 its drop; z in metres, ±0.55): a sheet leaving level and curving down (facade 2).
 */
export function aqueductFall(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 8, 5, 3);
  const w: Style = { mat: MAT.water, facade: 2, uAxis: AXIS.z, vAxis: AXIS.y };
  const rows: number[][] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    rows.push([b.vertex([t, -t * t, -0.55], [1, 0.6, 0], -0.55, t, w), b.vertex([t, -t * t, 0.55], [1, 0.6, 0], 0.55, t, w)]);
  }
  for (let i = 0; i < n; i++) b.quad(rows[i][0], rows[i + 1][0], rows[i + 1][1], rows[i][1]);
}
