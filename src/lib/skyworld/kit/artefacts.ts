/**
 * The Eyries' signature artefacts (SKY_CIVILIZATION.md §7, §8): the pieces
 * that make each class itself — the Great Halo, runestones, the Vaelspire's
 * crystals, wind-house roofs and gate-arches, paper lanterns, rope bridges,
 * docks, cranes, awnings and stalls, skyships, cloud-combs and wind-wheels,
 * forge chimneys and braziers, the Elder Bough, nest platforms, and the Long
 * Gale's ruins. Built with the kit's primitives (mesh.ts); catalogued with
 * the rest in pieces.ts.
 */

import { ANIM, AXIS, FACADE, KitBuilder, MAT, rng, type Style, type V3 } from './mesh';

type Lod = 0 | 1 | 2;
const n3 = (lod: Lod, a: number, b: number, c: number) => (lod === 0 ? a : lod === 1 ? b : c);
const TAU = Math.PI * 2;

// ---- the Halo (§7.3) ----------------------------------------------------------------------------

/**
 * The Great Halo: a ring of stone 100 m in radius (scaled to the Eyrie's), standing upright in the
 * x–y plane with its foot at y 0; a squared section 14 m deep and 10 m thick, chamfered, its inner
 * face carved with runes that glow; raised bands where its stones were joined.
 */
export function haloRing(b: KitBuilder, lod: Lod): void {
  const R = 100, segs = n3(lod, 160, 72, 28);
  const path = (r: number) => Array.from({ length: segs + 1 }, (_, i) => {
    const t = (i / segs) * TAU;
    const out: V3 = [Math.cos(t), Math.sin(t), 0];
    return { p: [Math.cos(t) * r, R + Math.sin(t) * r, 0] as V3, side: out, up: [0, 0, 1] as V3 };
  });
  const sec: Array<[number, number]> = lod === 2
    ? [[-7, -5], [7, -5], [7, 5], [-7, 5]]
    : [[-6.2, -5], [6.2, -5], [7, -4.2], [7, 4.2], [6.2, 5], [-6.2, 5], [-7, 4.2], [-7, -4.2]];
  b.sweep(path(R), sec, { mat: MAT.stone, uAxis: AXIS.x, vAxis: AXIS.z });
  if (lod === 2) return;
  // the rune band on the inner face, a little proud of it
  b.sweep(path(R - 7.05), [[-0.06, -2.4], [0.06, -2.4], [0.06, 2.4], [-0.06, 2.4]], { mat: MAT.rune, uAxis: AXIS.x, vAxis: AXIS.z });
  // raised bands round the ring every 15°, where its blocks were joined
  if (lod === 0) {
    for (let k = 0; k < 24; k++) {
      const t = (k / 24) * TAU;
      const pathB = [-0.7, 0.7].map((d) => {
        const tt = t + d / R;
        return { p: [Math.cos(tt) * R, R + Math.sin(tt) * R, 0] as V3, side: [Math.cos(tt), Math.sin(tt), 0] as V3, up: [0, 0, 1] as V3 };
      });
      b.sweep(pathB, [[-7.5, -5.5], [7.5, -5.5], [7.5, 5.5], [-7.5, 5.5]], { mat: MAT.trim, uAxis: AXIS.x, vAxis: AXIS.z, closedPath: false });
    }
  }
}

/** A runestone: a rough four-sided standing stone, 6 m, its faces cut with glowing runes, a capstone. Bobs when it floats. */
export function runestone(b: KitBuilder, lod: Lod): void {
  const r = rng(77);
  const prof: Array<[number, number]> = [[0.9, 0], [1.0, 0.6], [0.85, 3.2], [0.7, 5.0], [0.3, 5.9], [0, 6.1]];
  b.lathe(prof.map(([x, y]) => [x * (0.9 + 0.2 * r()), y] as [number, number]), lod === 2 ? 4 : 5, { mat: MAT.rune, flatSides: true, phase: 0.3, metric: true, crease: 0.3 });
}

// ---- the Vaelspire (§7.4) --------------------------------------------------------------------------

/**
 * The Vaelspire: a cluster of skystone crystals (unit height: the tallest reaches y 1, scaled to
 * 60–120 m), hexagonal prisms with pointed tips leaning out from a common root.
 */
export function crystalCluster(b: KitBuilder, lod: Lod): void {
  const r = rng(4242);
  const n = n3(lod, 9, 7, 4);
  for (let k = 0; k < n; k++) {
    const main = k === 0;
    const h = main ? 0.82 : 0.25 + 0.4 * r();
    const w = main ? 0.075 : 0.03 + 0.035 * r();
    const lean = main ? 0 : 0.25 + 0.35 * r();
    const az = (k / n) * TAU + r() * 0.5;
    const off = main ? 0 : 0.04 + 0.05 * r();
    b.push().translate(Math.cos(az) * off, 0, Math.sin(az) * off).rotateY(-az).rotateZ(-lean);
    b.lathe([[w, -0.05], [w, h], [0, h + w * 2.2]], 6, { mat: MAT.crystal, flatSides: true, phase: r() * 1.0, crease: 0.3 });
    b.pop();
  }
}

// ---- Petalfall (§7.5) --------------------------------------------------------------------------------

/**
 * A wind-house roof: four concave slopes rising to a point, their eaves overhanging and swept up at
 * the corners so the wind slips over (unit: half width 1 at the wall line, apex at y 1).
 */
export function roofPagoda(b: KitBuilder, lod: Lod): void {
  const nu = n3(lod, 10, 6, 2), ns = n3(lod, 7, 4, 2);
  const over = 1.38;
  for (let f = 0; f < 4; f++) {
    const a = (f / 4) * TAU;
    const out: V3 = [Math.cos(a), 0, Math.sin(a)], tan: V3 = [-Math.sin(a), 0, Math.cos(a)];
    const grid: number[][] = [];
    for (let j = 0; j <= ns; j++) {
      const s = j / ns; // 0 at the eave, 1 at the apex
      const w = over * (1 - s);
      const y = Math.pow(s, 1.7) * 1.0 - 0.05;
      const row: number[] = [];
      for (let i = 0; i <= nu; i++) {
        const u = -1 + (2 * i) / nu;
        const lift = 0.22 * Math.pow(Math.abs(u), 5) * Math.pow(1 - s, 2);
        const p: V3 = [out[0] * w + tan[0] * u * w, y + lift, out[2] * w + tan[2] * u * w];
        // the slope's normal: outward and up by how steep it is here
        const dy = 1.7 * Math.pow(Math.max(s, 0.02), 0.7);
        const n: V3 = [out[0], over / Math.max(dy, 0.3), out[2]];
        row.push(b.vertex(p, n, u * w * 2, (1 - s) * 1.6, { mat: MAT.roof, uAxis: AXIS.x, vAxis: AXIS.y }));
      }
      grid.push(row);
    }
    for (let j = 0; j < ns; j++) for (let i = 0; i < nu; i++) b.quad(grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]);
  }
  // ridge cap and finial post
  if (lod !== 2) b.lathe([[0.06, 0.9], [0.05, 1.25], [0.1, 1.3], [0, 1.42]], 6, { mat: MAT.gold });
}

/** A gate-arch of the orchards: two vermilion pillars, a straight tie beam, an upswept lintel capped in black. 8 × 7.5 m. */
export function gateTorii(b: KitBuilder, lod: Lod): void {
  const sides = n3(lod, 12, 8, 5);
  for (const s of [-1, 1]) {
    b.push().translate(s * 3.0, 0, 0).rotateZ(s * 0.03);
    b.frustum(0, 0, 0, 7.0, 0.42, 0.36, sides, { mat: MAT.lacquer, metric: true });
    b.frustum(0, 0, -0.1, 0.5, 0.55, 0.5, sides, { mat: MAT.iron, metric: true, capTop: true });
    b.pop();
  }
  // tie beam
  b.box([0, 5.6, 0], [3.7, 0.25, 0.22], { mat: MAT.lacquer });
  // lintel: a sweep that rises toward its ends
  const segs = n3(lod, 16, 8, 3);
  const path = Array.from({ length: segs + 1 }, (_, i) => {
    const t = -1 + (2 * i) / segs;
    return { p: [t * 4.8, 6.75 + 0.45 * t * t * t * t, 0] as V3, side: [0, 0, 1] as V3, up: [0, 1, 0] as V3 };
  });
  b.sweep(path, [[-0.35, -0.3], [0.35, -0.3], [0.35, 0.2], [-0.35, 0.2]], { mat: MAT.lacquer, uAxis: AXIS.x, vAxis: AXIS.y });
  if (lod !== 2) b.sweep(path.map((q) => ({ ...q, p: [q.p[0] * 1.03, q.p[1] + 0.35, 0] as V3 })), [[-0.42, -0.08], [0.42, -0.08], [0.42, 0.12], [-0.42, 0.12]], { mat: MAT.iron, uAxis: AXIS.x, vAxis: AXIS.y });
  if (lod === 0) b.box([0, 6.2, 0], [0.18, 0.4, 0.18], { mat: MAT.lacquer });
}

/** A paper lantern: a glowing paper globe, dark cap and foot, on its cord from y 0 down. */
export function lanternPaper(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 10, 6, 4);
  if (lod !== 2) b.box([0, -0.25, 0], [0.012, 0.25, 0.012], { mat: MAT.woven, faces: 0b110011 });
  b.lathe([[0.12, -0.5], [0.3, -0.62], [0.36, -0.8], [0.3, -0.98], [0.12, -1.1]], n, { mat: MAT.lantern, crease: 1.2 });
  if (lod !== 2) {
    b.disc(0, 0, -0.5, 0.14, n, true, { mat: MAT.iron });
    b.disc(0, 0, -1.1, 0.14, n, false, { mat: MAT.iron });
  }
}

/**
 * A rope bridge (unit: x ±1 along, z ±1 across, its deck at y 0 at the ends sagging to −1 in the
 * middle — stretch y by the sag): a plank deck, two rope hand-rails 1.1 m up, hangers between.
 */
export function bridgeRope(b: KitBuilder, lod: Lod): void {
  const segs = n3(lod, 24, 12, 6);
  const deck = Array.from({ length: segs + 1 }, (_, i) => {
    const x = -1 + (2 * i) / segs;
    return { p: [x, -(1 - x * x), 0] as V3, side: [0, 0, 1] as V3, up: [0, 1, 0] as V3 };
  });
  b.sweep(deck, [[-1, -0.04], [1, -0.04], [1, 0], [-1, 0]], { mat: MAT.timber, uAxis: AXIS.x, vAxis: AXIS.z });
  if (lod === 2) return;
  for (const s of [-1, 1]) {
    // (the rails sag less than the deck, hanging from the posts at each end)
    const rail = deck.map((q) => ({ ...q, p: [q.p[0], q.p[1] * 0.75 + 0.05, s * 0.98] as V3 }));
    b.sweep(rail, [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]], { mat: MAT.woven });
  }
}

// ---- Galeharbour (§7.10) ------------------------------------------------------------------------------

/** A dock (unit: from the rim at x 0 out to x 1, z ±1 across; its deck at y 0): planks on beams, struts below, mooring posts. */
export function dock(b: KitBuilder, lod: Lod): void {
  b.box([0.5, -0.06, 0], [0.5, 0.06, 1], { mat: MAT.timber });
  if (lod === 2) return;
  for (const s of [-1, 1]) {
    b.box([0.5, -0.25, s * 0.85], [0.5, 0.13, 0.08], { mat: MAT.timber });
    for (let k = 0; k <= 4; k++) b.box([0.06 + 0.22 * k, 0.35, s * 0.92], [0.02, 0.35, 0.05], { mat: MAT.timber });
  }
  if (lod === 0) {
    // struts down to the rock under the rim
    for (const s of [-0.6, 0.6]) {
      b.push().translate(0.25, -0.6, s).rotateZ(0.9);
      b.box([0, 0, 0], [0.5, 0.05, 0.05], { mat: MAT.timber });
      b.pop();
    }
  }
}

/** A crane (14 m): a timber mast on a stone foot, a jib reaching out over the anchorage, a rope and a crate. */
export function crane(b: KitBuilder, lod: Lod): void {
  b.box([0, 0.8, 0], [1.2, 0.8, 1.2], { mat: MAT.stone, faces: 0b110111 });
  b.box([0, 7.5, 0], [0.35, 6, 0.35], { mat: MAT.timber, faces: 0b110111 });
  b.push().translate(4.5, 12.6, 0).rotateZ(0.18);
  b.box([0, 0, 0], [5.5, 0.22, 0.22], { mat: MAT.timber });
  b.pop();
  if (lod === 2) return;
  b.box([9.3, 9.0, 0], [0.025, 4.2, 0.025], { mat: MAT.woven });
  b.box([9.3, 4.4, 0], [0.7, 0.55, 0.7], { mat: MAT.timber });
  if (lod === 0) {
    b.push().translate(1.4, 10.5, 0).rotateZ(-0.75);
    b.box([0, 0, 0], [2.2, 0.12, 0.12], { mat: MAT.timber });
    b.pop();
  }
}

/** An awning: striped cloth sloping from 3.2 m at the back to 2.4 m at the front on two poles, 4 m wide, its front edge fluttering. */
export function awning(b: KitBuilder, lod: Lod): void {
  for (const s of [-1, 1]) b.box([s * 1.95, 1.2, 1.4], [0.05, 1.2, 0.05], { mat: MAT.timber, faces: 0b110011 });
  const nx = n3(lod, 6, 3, 1), nz = n3(lod, 4, 2, 1);
  const ids: number[][] = [];
  for (let j = 0; j <= nz; j++) {
    const t = j / nz;
    const row: number[] = [];
    for (let i = 0; i <= nx; i++) {
      const x = -2 + (4 * i) / nx;
      row.push(b.vertex([x, 3.2 - 0.8 * t, -0.2 + 1.8 * t], [0, 0.9, 0.43], x + 2, 1.8 * t * t * 2.6, { mat: MAT.cloth, facade: FACADE.stripes, anim: ANIM.cloth, uAxis: AXIS.x, vAxis: AXIS.z }));
    }
    ids.push(row);
  }
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) b.quad(ids[j][i], ids[j][i + 1], ids[j + 1][i + 1], ids[j + 1][i]);
}

/** A market stall: a trestle table, jars and bundles and crates on and under it, an awning over. */
export function stall(b: KitBuilder, lod: Lod): void {
  awning(b, lod);
  b.box([0, 0.9, 0.6], [1.8, 0.05, 0.6], { mat: MAT.timber });
  if (lod === 2) return;
  for (const x of [-1.6, 1.6]) for (const z of [0.15, 1.05]) b.box([x, 0.45, z], [0.05, 0.45, 0.05], { mat: MAT.timber, faces: 0b110011 });
  const r = rng(91);
  for (let k = 0; k < (lod === 0 ? 7 : 3); k++) {
    const x = -1.5 + 3 * r(), z = 0.3 + 0.6 * r();
    if (r() < 0.5) {
      b.push().translate(x, 0, z);
      b.lathe([[0.12, 0.95], [0.18, 1.1], [0.14, 1.3], [0.06, 1.38], [0.07, 1.45]], 6, { mat: MAT.bronze });
      b.pop();
    } else b.box([x, 1.05, z], [0.15 + 0.1 * r(), 0.1, 0.12], { mat: r() < 0.5 ? MAT.cloth2 : MAT.timber });
  }
  b.box([1.2, 0.3, -0.3], [0.35, 0.3, 0.3], { mat: MAT.timber });
}

/** A stack of cargo: crates and barrels (props on docks and quays). */
export function cargo(b: KitBuilder, lod: Lod): void {
  const r = rng(303);
  for (let k = 0; k < (lod === 2 ? 2 : 5); k++) {
    const x = r() * 2.4 - 1.2, z = r() * 2.4 - 1.2, h = 0.45 + 0.2 * r();
    if (k % 2) b.frustum(x, z, 0, h * 2, 0.42, 0.42, lod === 0 ? 10 : 6, { mat: MAT.timber, capTop: true, metric: true });
    else b.box([x, h, z], [0.5, h, 0.5], { mat: MAT.timber });
  }
}

/**
 * A skyship (32 m from stem to stern along +z): a timber hull on a keel of glowing skystone, a deck,
 * a cabin aft, two masts with square sails bellied by the wind, a bowsprit.
 */
export function skyship(b: KitBuilder, lod: Lod, wings = true): void {
  const segs = n3(lod, 14, 8, 4), ring = n3(lod, 10, 6, 4);
  // the hull: sections along z, each a U narrowing toward the ends
  const L = 32;
  const sec = (z: number) => {
    const t = z / L, w = 3.6 * Math.sqrt(Math.max(0, 1 - Math.pow(2 * t - 1, 2))) + 0.3, d = 3.2 * (0.6 + 0.4 * Math.sqrt(Math.max(0, 1 - Math.pow(2 * t - 1, 2))));
    return { w, d };
  };
  const rows: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const z = (i / segs) * L - L / 2 + 0.01;
    const { w, d } = sec(z + L / 2);
    const row: number[] = [];
    for (let j = 0; j <= ring; j++) {
      const a = Math.PI * (j / ring); // 0 port deck edge … π starboard deck edge, through the keel
      const x = -Math.cos(a) * w, y = -Math.sin(a) * d;
      row.push(b.vertex([x, y, z], [-Math.cos(a), -Math.sin(a) * 0.8, 0], j / ring * 8, z, { mat: MAT.timber, uAxis: AXIS.x, vAxis: AXIS.z }));
    }
    rows.push(row);
  }
  for (let i = 0; i < segs; i++) for (let j = 0; j < ring; j++) b.quad(rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], rows[i][j + 1]);
  // deck
  b.box([0, 0.02, 0], [3.4, 0.05, L / 2 - 2], { mat: MAT.timber, faces: 4 });
  // the skystone keel
  b.box([0, -3.3, 0], [0.45, 0.35, L / 2 - 4], { mat: MAT.keystone });
  if (lod === 2) { b.box([0, 9, 2], [0.1, 9, 0.1], { mat: MAT.timber }); b.box([0, 10, 2], [5, 4, 0.06], { mat: MAT.cloth2 }); return; }
  // cabin aft
  b.box([0, 1.4, -L / 2 + 6], [2.6, 1.4, 3], { mat: MAT.timber });
  b.push().translate(0, 2.8, -L / 2 + 6);
  b.box([0, 0.15, 0], [2.9, 0.15, 3.3], { mat: MAT.roof2 });
  b.pop();
  // masts and sails
  for (const [z, h] of [[4, 18], [-4, 15]] as const) {
    b.frustum(0, z, 0, h, 0.28, 0.18, 6, { mat: MAT.timber, metric: true });
    b.box([0, h - 3, z], [5, 0.12, 0.12], { mat: MAT.timber });
    // the sail: bellied forward (+z) in its middle
    const nu = n3(lod, 6, 3, 1), nv = n3(lod, 6, 3, 1);
    const ids: number[][] = [];
    for (let j = 0; j <= nv; j++) {
      const v = j / nv;
      const row: number[] = [];
      for (let i = 0; i <= nu; i++) {
        const u = -1 + (2 * i) / nu;
        const belly = 1.4 * (1 - u * u) * Math.sin(Math.PI * Math.min(1, v * 1.1));
        row.push(b.vertex([u * 4.6, h - 3.2 - v * (h - 6), z + belly], [0, 0, 1], u * 4.6 + 5, v * 4.6, { mat: MAT.cloth2, anim: ANIM.sway, uAxis: AXIS.x, vAxis: AXIS.y }));
      }
      ids.push(row);
    }
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) b.quad(ids[j][i], ids[j][i + 1], ids[j + 1][i + 1], ids[j + 1][i]);
  }
  // bowsprit
  b.push().translate(0, 1.6, L / 2 + 2).rotateX(-0.35);
  b.box([0, 0, 0], [0.12, 0.12, 4], { mat: MAT.timber });
  b.pop();
  // lanterns at the bow and either side of the stern (lit from dusk: ships under way by night)
  b.box([0, 2.2, L / 2 - 1.2], [0.25, 0.35, 0.25], { mat: MAT.lantern });
  for (const s of [-1, 1]) b.box([s * 2.4, 3.5, -L / 2 + 3.2], [0.22, 0.32, 0.22], { mat: MAT.lantern });
  // (moored, its wings are furled)
  if (wings) shipWings(b, lod);
}

/** A rod between two points (a four-sided prism of half width r). */
function rod(b: KitBuilder, a: V3, c: V3, r: number, st: Style): void {
  const d: V3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  const t: V3 = [d[0] / l, d[1] / l, d[2] / l];
  const ref: V3 = Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u0: V3 = [t[1] * ref[2] - t[2] * ref[1], t[2] * ref[0] - t[0] * ref[2], t[0] * ref[1] - t[1] * ref[0]];
  const ul = Math.hypot(u0[0], u0[1], u0[2]);
  const u: V3 = [u0[0] / ul, u0[1] / ul, u0[2] / ul];
  const v: V3 = [t[1] * u[2] - t[2] * u[1], t[2] * u[0] - t[0] * u[2], t[0] * u[1] - t[1] * u[0]];
  const ring = (p: V3, w: number) => [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([i, j]) => {
    const n: V3 = [u[0] * i + v[0] * j, u[1] * i + v[1] * j, u[2] * i + v[2] * j];
    return b.vertex([p[0] + n[0] * w, p[1] + n[1] * w, p[2] + n[2] * w], n, 0, 0, st);
  });
  const A = ring(a, r), C = ring(c, r * 0.6);
  for (let k = 0; k < 4; k++) b.quad(A[k], A[(k + 1) % 4], C[(k + 1) % 4], C[k]);
}

/**
 * A skyship's wing-sails: on either side at the deck, a fan of four ribs swept back and raised,
 * cloth stretched between them in scallops (fluttering a little: v the distance from the root).
 * They spill and catch the wind as a gull's wings do; far off, one triangle a side.
 */
function shipWings(b: KitBuilder, lod: Lod): void {
  const cloth: Style = { mat: MAT.cloth2, anim: ANIM.sway, uAxis: AXIS.x, vAxis: AXIS.z };
  const ribs: Array<[number, number, number]> = [[0.12, 8.5, 2.6], [0.45, 10, 2.9], [0.8, 9.2, 2.5], [1.15, 7, 1.8]]; // sweep (rad), length, rise
  for (const s of [-1, 1]) {
    const R: V3 = [s * 3.3, 0.7, 3.0];
    const tips = ribs.map(([sw, len, rise]): V3 => [R[0] + s * Math.cos(sw) * len, R[1] + rise, R[2] - Math.sin(sw) * len]);
    const vert = (p: V3) => b.vertex(p, [0, 1, 0], p[0], Math.hypot(p[0] - R[0], p[2] - R[2]), cloth);
    if (lod === 2) { b.tri(vert(R), vert(tips[0]), vert(tips[3])); continue; }
    const r0 = vert(R);
    for (let k = 0; k + 1 < tips.length; k++) {
      const a = tips[k], c = tips[k + 1];
      // the scallop: the cloth's edge pulled in between the ribs
      const m: V3 = [R[0] + ((a[0] + c[0]) / 2 - R[0]) * 0.82, (a[1] + c[1]) / 2 - 0.2, R[2] + ((a[2] + c[2]) / 2 - R[2]) * 0.82];
      const va = vert(a), vm = vert(m), vc = vert(c);
      b.tri(r0, va, vm);
      b.tri(r0, vm, vc);
    }
    for (const tip of tips) rod(b, R, tip, 0.13, { mat: MAT.timber });
  }
}

// ---- Mistwells (§7.11) --------------------------------------------------------------------------------

/** A cloud-comb (20 m wide, 16 m tall): two masts, a lattice net of cords strung between them, a gutter trough below. */
export function cloudComb(b: KitBuilder, lod: Lod): void {
  for (const s of [-1, 1]) b.frustum(s * 10, 0, 0, 16, 0.3, 0.2, n3(lod, 8, 5, 4), { mat: MAT.timber, metric: true });
  b.box([0, 1.6, 0], [10, 0.3, 0.45], { mat: MAT.timber });
  const nx = n3(lod, 16, 8, 3), ny = n3(lod, 10, 5, 2);
  for (let i = 0; i <= nx; i++) b.box([-10 + (20 * i) / nx, 9, 0], [0.04, 7, 0.04], { mat: MAT.woven, faces: 0b110011 });
  for (let j = 0; j <= ny; j++) b.box([0, 2 + (14 * j) / ny, 0], [10, 0.04, 0.04], { mat: MAT.woven, faces: 0b111100 });
}

/** A wind-wheel's sails: six sails on spars round a hub, turning about z (the shader turns them). 9 m radius. */
export function windSails(b: KitBuilder, lod: Lod): void {
  const n = 6;
  b.lathe([[0.6, -0.4], [0.6, 0.4], [0, 0.6]], 8, { mat: MAT.iron, anim: ANIM.rotor });
  for (let k = 0; k < n; k++) {
    const a = (k / n) * TAU;
    b.push().rotateZ(a);
    b.box([4.5, 0, 0], [4.5, 0.1, 0.1], { mat: MAT.timber, anim: ANIM.rotor });
    if (lod !== 2) b.box([5.2, 0.9, 0.05], [3.6, 0.8, 0.02], { mat: MAT.cloth2, anim: ANIM.rotor });
    b.pop();
  }
}

// ---- Cinderforge (§7.7) ------------------------------------------------------------------------------

/** A forge chimney (unit: ±1 square, y 0 … 1): a tapering stack with a heavy cap. Stretch. */
export function chimney(b: KitBuilder, lod: Lod): void {
  const r = Math.SQRT2;
  b.lathe([[r * 1.0, 0], [r * 0.82, 0.94], [r * 1.05, 0.95], [r * 1.05, 1.0], [r * 0.7, 1.0]], 4, { mat: MAT.stone, flatSides: true, phase: Math.PI / 4, crease: 0.4 });
  if (lod !== 2) b.disc(0, 0, 0.99, r * 0.7, 4, true, { mat: MAT.lava }, Math.PI / 4);
}

/** A brazier: a bronze bowl of fire on a tripod, 2 m. */
export function brazier(b: KitBuilder, lod: Lod): void {
  b.lathe([[0.08, 0], [0.06, 1.4], [0.5, 1.5], [0.7, 1.9], [0.62, 1.95]], n3(lod, 10, 6, 4), { mat: MAT.bronze, metric: true });
  b.disc(0, 0, 1.85, 0.6, n3(lod, 10, 6, 4), true, { mat: MAT.lava });
}

// ---- the Elder Bough (§7.6) ---------------------------------------------------------------------------

/** A sweep of a tapering tube along a path (bark), for the Bough's trunk, limbs and roots. */
function limb(b: KitBuilder, pts: V3[], r0: number, r1: number, sides: number, style: Style): void {
  const path = pts.map((p, i) => {
    const a = pts[Math.max(i - 1, 0)], c = pts[Math.min(i + 1, pts.length - 1)];
    const t: V3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const tl = Math.hypot(t[0], t[1], t[2]) || 1;
    const T: V3 = [t[0] / tl, t[1] / tl, t[2] / tl];
    const ref: V3 = Math.abs(T[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    let s: V3 = [T[1] * ref[2] - T[2] * ref[1], T[2] * ref[0] - T[0] * ref[2], T[0] * ref[1] - T[1] * ref[0]];
    const sl = Math.hypot(s[0], s[1], s[2]) || 1;
    s = [s[0] / sl, s[1] / sl, s[2] / sl];
    const u: V3 = [T[1] * s[2] - T[2] * s[1], T[2] * s[0] - T[0] * s[2], T[0] * s[1] - T[1] * s[0]];
    const r = r0 + (r1 - r0) * (i / (pts.length - 1));
    return { p, side: [s[0] * r, s[1] * r, s[2] * r] as V3, up: [u[0] * r, u[1] * r, u[2] * r] as V3 };
  });
  const shape: Array<[number, number]> = Array.from({ length: sides }, (_, k) => [Math.cos((k / sides) * TAU), Math.sin((k / sides) * TAU)]);
  b.sweep(path, shape, style);
}

/**
 * The Elder Bough (§7.6): a colossal tree, 200 m in all (scaled to the Eyrie's): buttressed roots
 * spreading over the ground, a trunk 22 m thick, limbs reaching out and up, a crown of great clumps of
 * leaves. Its hanging roots under the isle are bough.roots.
 */
export function boughTree(b: KitBuilder, lod: Lod): void {
  const r = rng(1234);
  const sides = n3(lod, 14, 9, 6);
  const bark: Style = { mat: MAT.timber, uAxis: AXIS.x, vAxis: AXIS.z };
  // trunk
  const trunk: V3[] = [[0, -8, 0], [1, 20, 2], [-2, 50, 1], [0, 80, -2], [2, 100, 0]];
  limb(b, trunk, 13, 7, sides, bark);
  // buttress roots over the ground
  for (let k = 0; k < (lod === 2 ? 4 : 7); k++) {
    const a = (k / 7) * TAU + r() * 0.4;
    const L = 26 + 14 * r();
    limb(b, [[0, 10, 0], [Math.cos(a) * L * 0.4, 3, Math.sin(a) * L * 0.4], [Math.cos(a) * L, -2, Math.sin(a) * L]], 7, 1.5, n3(lod, 8, 6, 4), bark);
  }
  // limbs and the crown's clumps
  const clumps: Array<[V3, number]> = [];
  for (let k = 0; k < (lod === 2 ? 4 : 8); k++) {
    const a = (k / 8) * TAU + r() * 0.5;
    const L = 50 + 30 * r(), up = 25 + 30 * r();
    const base: V3 = [0, 70 + 25 * r(), 0];
    const mid: V3 = [Math.cos(a) * L * 0.5, base[1] + up * 0.7, Math.sin(a) * L * 0.5];
    const tip: V3 = [Math.cos(a) * L, base[1] + up, Math.sin(a) * L];
    limb(b, [base, mid, tip], 5.5, 1.6, n3(lod, 8, 6, 4), bark);
    clumps.push([tip, 26 + 12 * r()], [mid, 18 + 8 * r()]);
  }
  clumps.push([[0, 135, 0], 40]);
  for (const [c, rad] of clumps) {
    const n = n3(lod, 12, 8, 6);
    const prof: Array<[number, number]> = [];
    for (let i = 0; i <= n3(lod, 8, 5, 4); i++) {
      const t = -Math.PI / 2 + (Math.PI * i) / n3(lod, 8, 5, 4);
      prof.push([Math.max(0.001, Math.cos(t) * rad * (0.9 + 0.2 * r())), Math.sin(t) * rad * 0.75]);
    }
    b.push().translate(c[0], c[1], c[2]);
    b.lathe(prof, n, { mat: MAT.moss, crease: 1.4, ao: (p) => 0.55 + 0.45 * Math.max(0, Math.min(1, (p[1] - c[1] + rad * 0.4) / rad)) });
    b.pop();
  }
}

/** The Bough's hanging roots: tapering strands hanging from y 0 down to −1 (unit: stretch to the isle's depth). */
export function boughRoots(b: KitBuilder, lod: Lod): void {
  const r = rng(5678);
  for (let k = 0; k < n3(lod, 14, 8, 4); k++) {
    const a = r() * TAU, d = 0.1 + 0.8 * r(), len = 0.4 + 0.6 * r();
    const x = Math.cos(a) * d, z = Math.sin(a) * d;
    const pts: V3[] = [[x, 0.05, z], [x * 1.05 + (r() - 0.5) * 0.08, -len * 0.5, z * 1.05], [x * 1.1, -len, z * 1.1]];
    limb(b, pts, 0.035 + 0.02 * r(), 0.004, n3(lod, 6, 5, 3), { mat: MAT.timber, uAxis: AXIS.x, vAxis: AXIS.z });
  }
}

/** A nest platform in the Bough (8 m across): a timber deck on its ring beam, a rail of posts, a great woven nest, a lantern. */
export function nestPlatform(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 16, 10, 6);
  b.frustum(0, 0, -0.5, 0, 4.2, 4.0, n, { mat: MAT.timber, capTop: true, capBottom: true, metric: true });
  b.lathe([[1.6, 0], [2.3, 0.5], [2.4, 1.0], [2.0, 1.05], [1.6, 0.6], [0.001, 0.5]], n, { mat: MAT.woven, metric: true, crease: 1.2 });
  if (lod === 2) return;
  for (let k = 0; k < 10; k++) {
    const a = (k / 10) * TAU;
    b.box([Math.cos(a) * 3.85, 0.55, Math.sin(a) * 3.85], [0.06, 0.55, 0.06], { mat: MAT.timber, faces: 0b110011 });
  }
}

// ---- the Long Gale's ruins (§3.5, §7.9) -------------------------------------------------------------------

/** A broken top for a round tower (radial): a ring wall 1.2 m thick whose top edge is broken jagged, 0 … 5 m. */
export function ruinTop(b: KitBuilder, lod: Lod): void {
  const r = rng(999);
  const n = n3(lod, 32, 16, 8);
  const hs = Array.from({ length: n + 1 }, (_, i) => (i === n ? 0 : 0.3 + 4.5 * Math.pow(r(), 1.6)));
  hs[n] = hs[0];
  const st: Style = { mat: MAT.stone, uAxis: AXIS.x, vAxis: AXIS.y };
  for (const [off, out] of [[0, 1], [-1.2, -1]] as const) {
    const base = b.count;
    for (let i = 0; i <= n; i++) {
      const a = (i / n) * TAU;
      b.radial(a, 0, off, [out, 0, 0], a, 0, st);
      b.radial(a, hs[i], off, [out, 0, 0], a, hs[i], st);
    }
    for (let i = 0; i < n; i++) {
      const k = base + 2 * i;
      if (out > 0) b.quad(k, k + 1, k + 3, k + 2); else b.quad(k, k + 2, k + 3, k + 1);
    }
  }
  // the broken top surface
  const base = b.count;
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * TAU;
    b.radial(a, hs[i], 0, [0, 1, 0], a, 0, st);
    b.radial(a, hs[i], -1.2, [0, 1, 0], a, 1.2, st);
  }
  for (let i = 0; i < n; i++) { const k = base + 2 * i; b.quad(k, k + 2, k + 3, k + 1); }
}

/** A fallen column, lying in three drums along x, 7 m. */
export function columnFallen(b: KitBuilder, lod: Lod): void {
  const n = n3(lod, 12, 8, 5);
  for (let k = 0; k < 3; k++) {
    b.push().translate(-3 + k * 2.4 + (k === 2 ? 0.4 : 0), 0.45, (k - 1) * 0.15).rotateZ(Math.PI / 2).rotateX(k * 0.2);
    b.frustum(0, 0, -1.05, 1.05, 0.45, 0.43, n, { mat: MAT.marble, capTop: true, capBottom: true, metric: true });
    b.pop();
  }
}

/** A pile of rubble: tumbled blocks, 5 m across. */
export function rubble(b: KitBuilder, lod: Lod): void {
  const r = rng(4321);
  for (let k = 0; k < n3(lod, 9, 5, 3); k++) {
    const x = (r() - 0.5) * 4, z = (r() - 0.5) * 4, s = 0.4 + 0.6 * r();
    b.push().translate(x, s * 0.6, z).rotateY(r() * TAU).rotateX((r() - 0.5) * 0.6).rotateZ((r() - 0.5) * 0.6);
    b.box([0, 0, 0], [s * 1.2, s * 0.6, s * 0.8], { mat: MAT.stone });
    b.pop();
  }
}

