/**
 * The Eyries' building kit: a mesh builder and its primitives.
 * ---------------------------------------------------------------------------
 * Every piece (pieces.ts) is built once at start-up from these primitives,
 * in its own unit space, at three levels of detail, and drawn instanced
 * (gpu/structures.ts). An instance scales a piece per axis, turns it about y
 * and places it. Pieces come in two kinds:
 *   stretchable  bodies, walls, roofs, decks: simple shapes whose detail the
 *                shader paints in metric surface coordinates (uv × the
 *                instance's scale along the axis each coordinate follows), so
 *                stone courses and windows keep their size however stretched;
 *   fixed        cornices, pinnacles, lanterns, statues: modelled in metres
 *                and scaled uniformly.
 * Radial vertices (rings round a tower: cornices, ledges, parapets) hold a
 * unit direction in xz and a metric offset in w: the vertex shader puts them
 * at the instance's radius plus that offset, so a ring fits any tower with its
 * profile in true metres.
 *
 * Vertex (KIT_VERTEX_BYTES = 32): position f32 × 4 (x, y, z, radial offset),
 * normal snorm8 × 4 (w unused), attributes uint8 × 4 (material, occlusion
 * 0–255, facade, bits: u axis | v axis << 2 | animation << 4 | radial << 7),
 * surface coordinates f32 × 2 (unit space; metric once scaled).
 */

export const KIT_VERTEX_BYTES = 32;

/** Materials (the shader's palette lookups, structures.wgsl.ts). */
export const MAT = {
  stone: 0,     // ashlar: coursed blocks, mortar, weathering, moss
  trim: 1,      // dressed stone: cornices, quoins, mouldings (lighter, smooth)
  marble: 2,    // smooth pale stone with veins: columns, statues, domes' drums
  roof: 3,      // scale tiles laid like feathers
  timber: 4,    // planks
  gold: 5,      // gilded metal: finials, crowns of domes
  iron: 6,      // dark metal: rails, ribs, hinges
  glass: 7,     // leaded glass in iron ribs (greenhouses)
  crystal: 8,   // skystone crystal (Vaelspire, runestones)
  cloth: 9,     // banners, awnings: the palette's colour with its emblem
  cloth2: 10,   // the second colour (stripes, linings)
  mosaic: 11,   // radial mosaic floors (plazas, rotundas)
  plaster: 12,  // white plaster (Petalfall)
  woven: 13,    // nests, baskets, rope
  lacquer: 14,  // vermilion lacquer (gate-arches, pillars)
  lantern: 15,  // sunstone in glass: glows from dusk to dawn
  keystone: 16, // skystone set in stone: faint blue glow at night
  lava: 17,     // molten rock (Cinderforge)
  moss: 18,     // living green (planters, hanging moss)
  paving: 19,   // flagstones (decks, stairs, plazas)
  dark: 20,     // an opening's depth (doorways, arcades seen into)
  rune: 21,     // carved stone with glowing runes
  water: 22,    // still water in a basin (opaque)
  bronze: 23,   // weathered bronze: bells, instruments, dome ribs
  roof2: 24,    // the palette's second roof colour (domes, accents)
  dome: 25,     // ribbed dome sheathing (copper, verdigris, turquoise)
  leaf: 26,     // living leaves (lily pads, reeds, waterweed: the facade byte picks which, water.ts)
  petal: 27,    // a lotus's petals (white flushing pink) and its golden seed pod
  koi: 28,      // a koi's scales: white or orange, patched red and black (by its seed)
  clay: 29,     // fired clay: jars, pots, tiles of an inside
} as const;

/** Window patterns painted on walls (low nibble) and the surface they lie on (high bit). */
export const FACADE = {
  none: 0,
  roost: 1,     // small pointed windows, a row per 3.6 m floor, every ~3 m
  lancet: 2,    // tall pointed windows with tracery (cathedrals, halls)
  hall: 3,      // round-headed windows, one tall row
  slit: 4,      // arrow slits (walls, watchtowers)
  rose: 5,      // a rose window high in the face's middle (west fronts)
  lattice: 6,   // timber lattice screens (Petalfall)
  arcade: 7,    // dark arched openings at ground level (markets, cloisters)
  belfry: 8,    // slits below, a band of tall louvred openings under the top (bell towers)
  stripes: 9,   // (cloth) an awning's stripes, in its two colours
  /** u runs round the piece (0 … 2π at unit radius); else across a flat face (−1 … 1). */
  ROUND: 0x00,
  FLAT: 0x80,
} as const;

/**
 * Vertex animation (vertex shader): cloth in the wind (by its distance v from where it hangs), sails
 * swaying gently, a rotor turning about the piece's z axis, floating things bobbing and turning,
 * moored ships rising and falling a little, fish swimming round a slow circle; and not an
 * animation: a doorway's edge, put DOOR_HALF metres from the door's axis whatever the body's size
 * (interiors.ts).
 */
export const ANIM = { none: 0, cloth: 1, rotor: 2, sway: 3, bob: 4, float: 5, swim: 6, door: 7 } as const;

/** Which instance scale a surface coordinate follows. */
export const AXIS = { x: 0, y: 1, z: 2 } as const;

export type V3 = [number, number, number];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: V3) => Math.sqrt(dot(a, a));
const norm = (a: V3): V3 => { const l = len(a); return l > 1e-12 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 1, 0]; };
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
export const smooth = (lo: number, hi: number, x: number) => { const t = clamp01((x - lo) / (hi - lo)); return t * t * (3 - 2 * t); };
const TAU = Math.PI * 2;

/** mulberry32: a small seeded generator of [lo, hi) values. */
export function rng(seed: number): (lo?: number, hi?: number) => number {
  let s = seed >>> 0;
  return (lo = 0, hi = 1) => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return lo + (hi - lo) * (((t ^ (t >>> 14)) >>> 0) / 4294967296);
  };
}

/** A 3 × 4 affine matrix, rows. */
type M34 = [number, number, number, number, number, number, number, number, number, number, number, number];
const IDENT: M34 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
function mm(a: M34, b: M34): M34 {
  const o = new Array(12) as M34;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + (c === 3 ? a[r * 4 + 3] : 0);
    }
  }
  return o;
}

/** Per-vertex shading inputs a primitive may set (else the builder's current state). */
export type Style = {
  mat?: number;
  facade?: number;
  anim?: number;
  /** Axes the surface coordinates follow (stretchable pieces). */
  uAxis?: number;
  vAxis?: number;
  /** Occlusion at a point (1 open … 0 enclosed); default 1. */
  ao?: (p: V3, n: V3) => number;
};

export class KitBuilder {
  readonly pos: number[] = [];
  readonly nrm: number[] = [];
  readonly att: number[] = [];
  readonly uv: number[] = [];
  readonly idx: number[] = [];
  mat: number = MAT.stone;
  facade: number = FACADE.none;
  anim: number = ANIM.none;
  uAxis: number = AXIS.x;
  vAxis: number = AXIS.y;
  private m: M34 = [...IDENT];
  private stack: Array<{ m: M34; mat: number; facade: number; anim: number; uAxis: number; vAxis: number }> = [];
  /** Normal matrix (inverse transpose of m's 3 × 3) and the uniform scale surface coordinates take. */
  private nm: number[] = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  private uvs = 1;

  get count(): number { return this.pos.length / 4; }

  push(): this {
    this.stack.push({ m: [...this.m], mat: this.mat, facade: this.facade, anim: this.anim, uAxis: this.uAxis, vAxis: this.vAxis });
    return this;
  }
  pop(): this {
    const s = this.stack.pop()!;
    this.m = s.m; this.mat = s.mat; this.facade = s.facade; this.anim = s.anim; this.uAxis = s.uAxis; this.vAxis = s.vAxis;
    this.refresh();
    return this;
  }
  private apply(t: M34): this { this.m = mm(this.m, t); this.refresh(); return this; }
  translate(x: number, y: number, z: number): this { return this.apply([1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z]); }
  rotateY(a: number): this { const c = Math.cos(a), s = Math.sin(a); return this.apply([c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0]); }
  rotateX(a: number): this { const c = Math.cos(a), s = Math.sin(a); return this.apply([1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0]); }
  rotateZ(a: number): this { const c = Math.cos(a), s = Math.sin(a); return this.apply([c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0]); }
  scale(x: number, y = x, z = x): this { return this.apply([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0]); }
  private refresh(): void {
    const m = this.m;
    const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    // inverse transpose = cofactor matrix / det
    this.nm = [A, B, C, -(b * i - c * h), a * i - c * g, -(a * h - b * g), b * f - c * e, -(a * f - c * d), a * e - b * d].map((v) => v / (det || 1));
    this.uvs = Math.cbrt(Math.abs(det)) || 1;
  }

  /** A vertex at piece-local p (transformed), normal n, surface coordinates (u, v) in metres before the transform's scale. */
  vertex(p: V3, n: V3, u: number, v: number, style?: Style, ao = 1): number {
    const m = this.m, q = this.nm;
    const x = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3];
    const y = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7];
    const z = m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11];
    const nn = norm([q[0] * n[0] + q[1] * n[1] + q[2] * n[2], q[3] * n[0] + q[4] * n[1] + q[5] * n[2], q[6] * n[0] + q[7] * n[1] + q[8] * n[2]]);
    const occ = clamp01(ao * (style?.ao ? style.ao([x, y, z], nn) : 1));
    this.pos.push(x, y, z, 0);
    this.nrm.push(nn[0], nn[1], nn[2]);
    this.att.push(style?.mat ?? this.mat, Math.round(occ * 255), style?.facade ?? this.facade,
      (style?.uAxis ?? this.uAxis) | ((style?.vAxis ?? this.vAxis) << 2) | ((style?.anim ?? this.anim) << 4));
    this.uv.push(u * this.uvs, v * this.uvs);
    return this.count - 1;
  }

  /**
   * A radial vertex (no transform): at bearing th, height y, `off` metres beyond the instance's
   * radius (the shader adds it to the instance's x scale); normal n in the ring's own frame.
   */
  radial(th: number, y: number, off: number, n: V3, u: number, v: number, style?: Style, ao = 1): number {
    const c = Math.cos(th), s = Math.sin(th);
    // n given as (out, up, along the ring) at bearing th
    const nw = norm([n[0] * c - n[2] * s, n[1], n[0] * s + n[2] * c]);
    this.pos.push(c, y, s, off);
    this.nrm.push(nw[0], nw[1], nw[2]);
    const occ = clamp01(ao * (style?.ao ? style.ao([c, y, s], nw) : 1));
    this.att.push(style?.mat ?? this.mat, Math.round(occ * 255), style?.facade ?? this.facade,
      (style?.uAxis ?? this.uAxis) | ((style?.vAxis ?? this.vAxis) << 2) | ((style?.anim ?? this.anim) << 4) | 0x80);
    this.uv.push(u, v);
    return this.count - 1;
  }

  tri(a: number, b: number, c: number): void { this.idx.push(a, b, c); }
  /** Quad a b c d (in order round it). */
  quad(a: number, b: number, c: number, d: number): void { this.idx.push(a, b, c, a, c, d); }

  // ---- primitives -------------------------------------------------------------------------

  /**
   * An axis-aligned box: centre c, half extents h. Faces: bit 0 +x, 1 −x, 2 +y, 3 −y, 4 +z, 5 −z.
   * Surface coordinates: on the sides along the face (x or z) and up (y); on top and bottom x and z.
   */
  box(c: V3, h: V3, style?: Style & { faces?: number; sideFacade?: number }): void {
    const faces = style?.faces ?? 63;
    const F: Array<{ n: V3; u: V3; v: V3; ua: number; va: number; side: boolean }> = [
      { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0], ua: AXIS.z, va: AXIS.y, side: true },
      { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0], ua: AXIS.z, va: AXIS.y, side: true },
      { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1], ua: AXIS.x, va: AXIS.z, side: false },
      { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1], ua: AXIS.x, va: AXIS.z, side: false },
      { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0], ua: AXIS.x, va: AXIS.y, side: true },
      { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0], ua: AXIS.x, va: AXIS.y, side: true },
    ];
    F.forEach((f, k) => {
      if (!(faces & (1 << k))) return;
      const st: Style = { ...style, uAxis: f.ua, vAxis: f.va, facade: f.side ? (style?.sideFacade ?? style?.facade ?? this.facade) : FACADE.none };
      const corner = (su: number, sv: number): V3 => [
        c[0] + h[0] * (f.n[0] + f.u[0] * su + f.v[0] * sv), c[1] + h[1] * (f.n[1] + f.u[1] * su + f.v[1] * sv), c[2] + h[2] * (f.n[2] + f.u[2] * su + f.v[2] * sv),
      ];
      const uv = (p: V3): [number, number] => {
        // signed coordinates along the face: the facade's flat surfaces expect u in −1 … 1 at unit half width
        const uu = f.ua === AXIS.x ? p[0] * (f.u[0] || 1) : p[2] * (f.u[2] || 1);
        const vv = f.va === AXIS.y ? p[1] : p[2] * (f.v[2] || 1);
        return [uu, vv];
      };
      const ids = ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as const).map(([su, sv]) => {
        const p = corner(su, sv);
        const [uu, vv] = uv(p);
        return this.vertex(p, f.n, uu, vv, st);
      });
      this.quad(ids[0], ids[1], ids[2], ids[3]);
    });
  }

  /**
   * A frustum round the y axis at (cx, cz): radius r0 at y0 to r1 at y1, `sides` faces. Smooth
   * normals unless `flat`. Surface coordinates: u round it (θ × radius when `metric`, else θ: it
   * then follows the instance's x scale), v up it.
   */
  frustum(cx: number, cz: number, y0: number, y1: number, r0: number, r1: number, sides: number,
    style?: Style & { flat?: boolean; capTop?: boolean; capBottom?: boolean; phase?: number; metric?: boolean; v0?: number }): void {
    const ph = style?.phase ?? 0;
    const slope = (r0 - r1) / Math.max(y1 - y0, 1e-6);
    const um = (r: number) => (style?.metric ? r : 1);
    const v0 = style?.v0 ?? y0;
    const st: Style = { uAxis: AXIS.x, vAxis: AXIS.y, ...style };
    if (style?.flat) {
      for (let s = 0; s < sides; s++) {
        const a0 = ph + (s / sides) * TAU, a1 = ph + ((s + 1) / sides) * TAU, am = 0.5 * (a0 + a1);
        const n = norm([Math.cos(am), slope * Math.cos(Math.PI / sides), Math.sin(am)]);
        const p = (a: number, r: number, y: number): V3 => [cx + Math.cos(a) * r, y, cz + Math.sin(a) * r];
        const ids = [
          this.vertex(p(a0, r0, y0), n, a0 * um(r0), v0, st),
          this.vertex(p(a1, r0, y0), n, a1 * um(r0), v0, st),
          this.vertex(p(a1, r1, y1), n, a1 * um(r1), v0 + y1 - y0, st),
          this.vertex(p(a0, r1, y1), n, a0 * um(r1), v0 + y1 - y0, st),
        ];
        this.quad(ids[0], ids[3], ids[2], ids[1]);
      }
    } else {
      const base = this.count;
      for (let s = 0; s <= sides; s++) {
        const a = ph + (s / sides) * TAU;
        const n = norm([Math.cos(a), slope, Math.sin(a)]);
        this.vertex([cx + Math.cos(a) * r0, y0, cz + Math.sin(a) * r0], n, (a - ph) * um(r0), v0, st);
        this.vertex([cx + Math.cos(a) * r1, y1, cz + Math.sin(a) * r1], n, (a - ph) * um(r1), v0 + y1 - y0, st);
      }
      for (let s = 0; s < sides; s++) {
        const a = base + 2 * s;
        this.quad(a, a + 1, a + 3, a + 2);
      }
    }
    if (style?.capTop && r1 > 0) this.disc(cx, cz, y1, r1, sides, true, { ...style, facade: FACADE.none, ao: undefined }, ph);
    if (style?.capBottom && r0 > 0) this.disc(cx, cz, y0, r0, sides, false, { ...style, facade: FACADE.none, ao: undefined }, ph);
  }

  /** A flat disc (facing up or down), surface coordinates x and z. */
  disc(cx: number, cz: number, y: number, r: number, sides: number, up: boolean, style?: Style, phase = 0): void {
    const n: V3 = [0, up ? 1 : -1, 0];
    const st: Style = { ...style, uAxis: AXIS.x, vAxis: AXIS.z };
    const c = this.vertex([cx, y, cz], n, cx, cz, st);
    const ring: number[] = [];
    for (let s = 0; s < sides; s++) {
      const a = phase + (s / sides) * TAU;
      const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
      ring.push(this.vertex([x, y, z], n, x, z, st));
    }
    for (let s = 0; s < sides; s++) {
      if (up) this.tri(c, ring[(s + 1) % sides], ring[s]);
      else this.tri(c, ring[s], ring[(s + 1) % sides]);
    }
  }

  /** A flat ring (annulus) at y, r0 … r1, facing up or down. */
  annulus(y: number, r0: number, r1: number, sides: number, up: boolean, style?: Style, phase = 0): void {
    const n: V3 = [0, up ? 1 : -1, 0];
    const st: Style = { ...style, uAxis: AXIS.x, vAxis: AXIS.z };
    const base = this.count;
    for (let s = 0; s <= sides; s++) {
      const a = phase + (s / sides) * TAU;
      this.vertex([Math.cos(a) * r0, y, Math.sin(a) * r0], n, Math.cos(a) * r0, Math.sin(a) * r0, st);
      this.vertex([Math.cos(a) * r1, y, Math.sin(a) * r1], n, Math.cos(a) * r1, Math.sin(a) * r1, st);
    }
    for (let s = 0; s < sides; s++) {
      const a = base + 2 * s;
      if (up) this.quad(a, a + 2, a + 3, a + 1);
      else this.quad(a, a + 1, a + 3, a + 2);
    }
  }

  /**
   * A surface of revolution round the y axis through `profile` ([radius, y] from bottom to top).
   * Normals follow the profile, split where it turns by more than `crease` (rad). Surface
   * coordinates: u round it (θ × radius when `metric`, else θ), v along the profile (metres).
   */
  lathe(profile: ReadonlyArray<readonly [number, number]>, sides: number,
    style?: Style & { metric?: boolean; phase?: number; crease?: number; flatSides?: boolean }): void {
    const ph = style?.phase ?? 0;
    const crease = style?.crease ?? 0.75;
    const st: Style = { uAxis: AXIS.x, vAxis: AXIS.y, ...style };
    const P = profile;
    // Segment normals (outward, in the (r, y) plane), then per-point normals split at creases.
    const segN: Array<[number, number]> = [];
    for (let i = 0; i + 1 < P.length; i++) {
      const dr = P[i + 1][0] - P[i][0], dy = P[i + 1][1] - P[i][1];
      const l = Math.hypot(dr, dy) || 1;
      segN.push([dy / l, -dr / l]);
    }
    type Ring = { r: number; y: number; n: [number, number]; v: number };
    const rings: Array<[Ring, Ring]> = []; // per profile point: (end of the segment before, start of the one after)
    let v = 0;
    for (let i = 0; i < P.length; i++) {
      if (i > 0) v += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
      const a = i > 0 ? segN[i - 1] : segN[0];
      const b = i < segN.length ? segN[i] : segN[segN.length - 1];
      const turn = Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1])));
      if (turn > crease) {
        rings.push([{ r: P[i][0], y: P[i][1], n: a, v }, { r: P[i][0], y: P[i][1], n: b, v }]);
      } else {
        const m = [a[0] + b[0], a[1] + b[1]];
        const l = Math.hypot(m[0], m[1]) || 1;
        const n: [number, number] = [m[0] / l, m[1] / l];
        rings.push([{ r: P[i][0], y: P[i][1], n, v }, { r: P[i][0], y: P[i][1], n, v }]);
      }
    }
    const um = (r: number) => (style?.metric ? Math.max(r, 1e-3) : 1);
    for (let i = 0; i + 1 < rings.length; i++) {
      const A = rings[i][1], B = rings[i + 1][0];
      if (A.r <= 1e-6 && B.r <= 1e-6) continue;
      const base = this.count;
      if (style?.flatSides) {
        for (let s = 0; s < sides; s++) {
          const a0 = ph + (s / sides) * TAU, a1 = ph + ((s + 1) / sides) * TAU, am = 0.5 * (a0 + a1);
          const n0 = norm([Math.cos(am) * A.n[0], A.n[1], Math.sin(am) * A.n[0]]);
          const p = (a: number, R: Ring): V3 => [Math.cos(a) * R.r, R.y, Math.sin(a) * R.r];
          const q = [
            this.vertex(p(a0, A), n0, (a0 - ph) * um(A.r), A.v, st), this.vertex(p(a1, A), n0, (a1 - ph) * um(A.r), A.v, st),
            this.vertex(p(a1, B), n0, (a1 - ph) * um(B.r), B.v, st), this.vertex(p(a0, B), n0, (a0 - ph) * um(B.r), B.v, st),
          ];
          this.quad(q[0], q[3], q[2], q[1]);
        }
        continue;
      }
      for (let s = 0; s <= sides; s++) {
        const a = ph + (s / sides) * TAU;
        const c = Math.cos(a), sn = Math.sin(a);
        this.vertex([c * A.r, A.y, sn * A.r], norm([c * A.n[0], A.n[1], sn * A.n[0]]), (a - ph) * um(A.r), A.v, st);
        this.vertex([c * B.r, B.y, sn * B.r], norm([c * B.n[0], B.n[1], sn * B.n[0]]), (a - ph) * um(B.r), B.v, st);
      }
      for (let s = 0; s < sides; s++) {
        const k = base + 2 * s;
        this.quad(k, k + 1, k + 3, k + 2);
      }
    }
  }

  /**
   * A ring of radial vertices (no transform: centred on the piece's axis) through `profile`
   * ([metres beyond the instance's radius, y] from bottom to top): cornices, ledges, parapets that
   * fit any tower. Surface coordinates: u round it (θ: follows the x scale), v along the profile.
   */
  radialRing(profile: ReadonlyArray<readonly [number, number]>, sides: number, style?: Style & { phase?: number; crease?: number }): void {
    const ph = style?.phase ?? 0;
    const crease = style?.crease ?? 0.75;
    const st: Style = { uAxis: AXIS.x, vAxis: AXIS.y, ...style };
    const P = profile;
    const segN: Array<[number, number]> = [];
    for (let i = 0; i + 1 < P.length; i++) {
      const dr = P[i + 1][0] - P[i][0], dy = P[i + 1][1] - P[i][1];
      const l = Math.hypot(dr, dy) || 1;
      segN.push([dy / l, -dr / l]);
    }
    let v = 0;
    const pts: Array<[{ o: number; y: number; n: [number, number]; v: number }, { o: number; y: number; n: [number, number]; v: number }]> = [];
    for (let i = 0; i < P.length; i++) {
      if (i > 0) v += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
      const a = i > 0 ? segN[i - 1] : segN[0];
      const b = i < segN.length ? segN[i] : segN[segN.length - 1];
      const turn = Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1])));
      if (turn > crease) pts.push([{ o: P[i][0], y: P[i][1], n: a, v }, { o: P[i][0], y: P[i][1], n: b, v }]);
      else {
        const m = [a[0] + b[0], a[1] + b[1]];
        const l = Math.hypot(m[0], m[1]) || 1;
        const n: [number, number] = [m[0] / l, m[1] / l];
        pts.push([{ o: P[i][0], y: P[i][1], n, v }, { o: P[i][0], y: P[i][1], n, v }]);
      }
    }
    for (let i = 0; i + 1 < pts.length; i++) {
      const A = pts[i][1], B = pts[i + 1][0];
      const base = this.count;
      for (let s = 0; s <= sides; s++) {
        const a = ph + (s / sides) * TAU;
        this.radial(a, A.y, A.o, [A.n[0], A.n[1], 0], a - ph, A.v, st);
        this.radial(a, B.y, B.o, [B.n[0], B.n[1], 0], a - ph, B.v, st);
      }
      for (let s = 0; s < sides; s++) {
        const k = base + 2 * s;
        this.quad(k, k + 1, k + 3, k + 2);
      }
    }
  }

  /**
   * A wall slab with an arched opening: x −w … w, y y0 … y1, z −t … t; the opening ±open wide up
   * to the springing `spring`, then a round or pointed arch to `crown`. Builds both faces (the
   * spandrels), the arch's soffit and jambs, the outer ends, the top and the piers' feet.
   * `keystone`: a block of MAT.keystone at the crown on both faces.
   */
  archWall(o: {
    w: number; y0: number; y1: number; t: number; open: number; spring: number; crown: number;
    pointed?: boolean; segs?: number; keystone?: boolean; ends?: boolean; top?: boolean; feet?: boolean; style?: Style;
  }): void {
    const segs = o.segs ?? 12;
    const st = o.style ?? {};
    // The opening's outline, left jamb foot → arch → right jamb foot.
    const inner: Array<[number, number]> = [[-o.open, o.y0], [-o.open, o.spring]];
    for (let i = 1; i < segs; i++) inner.push(archCurve(i / segs, o.open, o.spring, o.crown - o.spring, !!o.pointed));
    inner.push([o.open, o.spring], [o.open, o.y0]);
    // The outer outline (left side up, the top, right side down), point for point by arc length, its
    // two top corners kept exact (their own points, paired with the opening's outline there).
    const H = o.y1 - o.y0, outerLen = 2 * H + 2 * o.w;
    const outerAt = (s: number): [number, number] => {
      const d = s * outerLen;
      if (d <= H) return [-o.w, o.y0 + d];
      if (d <= H + 2 * o.w) return [-o.w + (d - H), o.y1];
      return [o.w, o.y1 - (d - H - 2 * o.w)];
    };
    const cum = [0];
    for (let i = 1; i < inner.length; i++) cum.push(cum[i - 1] + Math.hypot(inner[i][0] - inner[i - 1][0], inner[i][1] - inner[i - 1][1]));
    const total = cum[cum.length - 1];
    const innerAt = (s: number): [number, number] => {
      const d = s * total;
      let i = 1;
      while (i < cum.length - 1 && cum[i] < d) i++;
      const f = (d - cum[i - 1]) / Math.max(cum[i] - cum[i - 1], 1e-9);
      return [inner[i - 1][0] + (inner[i][0] - inner[i - 1][0]) * f, inner[i - 1][1] + (inner[i][1] - inner[i - 1][1]) * f];
    };
    const fr = new Set<number>(cum.map((c) => c / total));
    fr.add(H / outerLen); fr.add((H + 2 * o.w) / outerLen);
    const params = [...fr].sort((a, b) => a - b);
    const A = params.map(innerAt), Bo = params.map(outerAt);
    const n = params.length;
    for (const side of [1, -1]) {
      const z = side * o.t;
      const nrm: V3 = [0, 0, side];
      const st2: Style = { ...st, uAxis: AXIS.x, vAxis: AXIS.y };
      const a = A.map(([x, y]) => this.vertex([x, y, z], nrm, side * x, y, st2, archAO(x, y, o)));
      const b = Bo.map(([x, y]) => this.vertex([x, y, z], nrm, side * x, y, st2));
      for (let i = 0; i + 1 < n; i++) {
        if (side > 0) this.quad(a[i], b[i], b[i + 1], a[i + 1]);
        else this.quad(a[i], a[i + 1], b[i + 1], b[i]);
      }
    }
    // Soffit and jambs: the opening's outline swept through the wall.
    let along = 0;
    for (let i = 0; i + 1 < inner.length; i++) {
      const [x0, y0] = inner[i], [x1, y1] = inner[i + 1];
      const dx = x1 - x0, dy = y1 - y0;
      const l = Math.hypot(dx, dy);
      if (l < 1e-6) continue; // (jambs of no height: an arch springing at the wall's foot)
      const nrm: V3 = [dy / l, -dx / l, 0]; // into the opening (the jambs face each other, the soffit down)
      const sty: Style = { ...st, uAxis: AXIS.z, vAxis: AXIS.y };
      const q = [
        this.vertex([x0, y0, o.t], nrm, o.t, along, sty, 0.55),
        this.vertex([x1, y1, o.t], nrm, o.t, along + l, sty, 0.55),
        this.vertex([x1, y1, -o.t], nrm, -o.t, along + l, sty, 0.55),
        this.vertex([x0, y0, -o.t], nrm, -o.t, along, sty, 0.55),
      ];
      this.quad(q[0], q[1], q[2], q[3]);
      along += l;
    }
    if (o.ends !== false) {
      this.box([-o.w, (o.y0 + o.y1) / 2, 0], [0, H / 2, o.t], { ...st, faces: 2 });
      this.box([o.w, (o.y0 + o.y1) / 2, 0], [0, H / 2, o.t], { ...st, faces: 1 });
    }
    if (o.top !== false) this.box([0, o.y1, 0], [o.w, 0, o.t], { ...st, faces: 4 });
    if (o.feet) {
      const fw = (o.w - o.open) / 2;
      this.box([-o.w + fw, o.y0, 0], [fw, 0, o.t], { ...st, faces: 8 });
      this.box([o.w - fw, o.y0, 0], [fw, 0, o.t], { ...st, faces: 8 });
    }
    if (o.keystone) {
      // (from the crown up, never through the top: 60 % of what lies between, at most 0.9 m)
      const kh = Math.min(0.6 * (o.y1 - o.crown), 0.9), kw = Math.max(0.14 * o.open, Math.min(0.25, 0.3 * o.open));
      this.box([0, o.crown + kh * 0.5 - 0.02 * kh, 0], [kw, kh * 0.5, o.t + 0.04 * Math.min(1, o.t)], { mat: MAT.keystone });
    }
  }

  /** A straight flight of `n` steps across x0 … x1, rising y0 → y1 over z0 → z1 (treads, risers, stringer sides, soffit). */
  stairs(n: number, x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, style?: Style & { tread?: number }): void {
    const rise = (y1 - y0) / n, run = (z1 - z0) / n;
    const st: Style = { ...style };
    const tread: Style = { ...style, mat: style?.tread ?? MAT.paving };
    for (let i = 0; i < n; i++) {
      const yb = y0 + i * rise, yt = yb + rise, za = z0 + i * run, zb = za + run;
      // riser (facing −z) and tread (up)
      const r = [
        this.vertex([x0, yb, za], [0, 0, -1], x0, yb, { ...st, uAxis: AXIS.x, vAxis: AXIS.y }),
        this.vertex([x1, yb, za], [0, 0, -1], x1, yb, { ...st, uAxis: AXIS.x, vAxis: AXIS.y }),
        this.vertex([x1, yt, za], [0, 0, -1], x1, yt, { ...st, uAxis: AXIS.x, vAxis: AXIS.y }),
        this.vertex([x0, yt, za], [0, 0, -1], x0, yt, { ...st, uAxis: AXIS.x, vAxis: AXIS.y }),
      ];
      this.quad(r[0], r[1], r[2], r[3]);
      const t = [
        this.vertex([x0, yt, za], [0, 1, 0], x0, za, { ...tread, uAxis: AXIS.x, vAxis: AXIS.z }),
        this.vertex([x1, yt, za], [0, 1, 0], x1, za, { ...tread, uAxis: AXIS.x, vAxis: AXIS.z }),
        this.vertex([x1, yt, zb], [0, 1, 0], x1, zb, { ...tread, uAxis: AXIS.x, vAxis: AXIS.z }),
        this.vertex([x0, yt, zb], [0, 1, 0], x0, zb, { ...tread, uAxis: AXIS.x, vAxis: AXIS.z }),
      ];
      this.quad(t[0], t[3], t[2], t[1]);
    }
    // Stringers: each side a sawtooth down to a line from the foot to under the top step's back.
    for (const [x, sgn] of [[x0, -1], [x1, 1]] as const) {
      const nrm: V3 = [sgn, 0, 0];
      const sts: Style = { ...st, uAxis: AXIS.z, vAxis: AXIS.y };
      for (let i = 0; i < n; i++) {
        const yb = y0 + i * rise, yt = yb + rise, za = z0 + i * run, zb = za + run;
        const bot0 = y0 + (za - z0) / (z1 - z0) * (y1 - y0) - 2 * rise, bot1 = y0 + (zb - z0) / (z1 - z0) * (y1 - y0) - 2 * rise;
        const q = [this.vertex([x, Math.min(bot0, yb), za], nrm, za, bot0, sts), this.vertex([x, Math.min(bot1, yb), zb], nrm, zb, bot1, sts),
          this.vertex([x, yt, zb], nrm, zb, yt, sts), this.vertex([x, yt, za], nrm, za, yt, sts)];
        if (sgn > 0) this.quad(q[0], q[1], q[2], q[3]);
        else this.quad(q[0], q[3], q[2], q[1]);
      }
    }
    // Soffit: under the flight.
    const dz = z1 - z0, dy = y1 - y0;
    const l = Math.hypot(dz, dy) || 1;
    const nrm: V3 = [0, -dz / l, dy / l];
    const s = [
      this.vertex([x0, y0 - 2 * rise, z0], nrm, x0, 0, { ...st, uAxis: AXIS.x, vAxis: AXIS.z }),
      this.vertex([x1, y0 - 2 * rise, z0], nrm, x1, 0, { ...st, uAxis: AXIS.x, vAxis: AXIS.z }),
      this.vertex([x1, y1 - 2 * rise, z1], nrm, x1, l, { ...st, uAxis: AXIS.x, vAxis: AXIS.z }),
      this.vertex([x0, y1 - 2 * rise, z1], nrm, x0, l, { ...st, uAxis: AXIS.x, vAxis: AXIS.z }),
    ];
    this.quad(s[0], s[1], s[2], s[3]);
  }

  /**
   * A tube along a path of frames: at each point p the cross-section `shape` ([across, up] in the
   * frame's (side, up) axes) is placed; closes the ring when `closed`. Surface coordinates: u round
   * the section (metres), v along the path (metres).
   */
  sweep(path: ReadonlyArray<{ p: V3; side: V3; up: V3 }>, shape: ReadonlyArray<readonly [number, number]>, style?: Style & { closedPath?: boolean; smooth?: boolean }): void {
    const m = shape.length;
    const st: Style = { uAxis: AXIS.x, vAxis: AXIS.z, ...style };
    const along = [0];
    for (let i = 1; i < path.length; i++) along.push(along[i - 1] + len(sub(path[i].p, path[i - 1].p)));
    const per: number[] = [0];
    for (let k = 1; k <= m; k++) per.push(per[k - 1] + Math.hypot(shape[k % m][0] - shape[k - 1][0], shape[k % m][1] - shape[k - 1][1]));
    // flat-shaded faces round the section (crisp mouldings), smooth along the path
    for (let k = 0; k < m; k++) {
      const a = shape[k], b = shape[(k + 1) % m];
      const base = this.count;
      for (let i = 0; i < path.length; i++) {
        const F = path[i];
        const tan = norm(sub(path[Math.min(i + 1, path.length - 1)].p, path[Math.max(i - 1, 0)].p));
        const pa = add(F.p, add(mul(F.side, a[0]), mul(F.up, a[1])));
        const pb = add(F.p, add(mul(F.side, b[0]), mul(F.up, b[1])));
        const e = sub(pb, pa);
        let nrm = norm(cross(e, tan));
        // outward from the section's middle
        const mid = add(F.p, add(mul(F.side, 0.5 * (a[0] + b[0])), mul(F.up, 0.5 * (a[1] + b[1]))));
        if (dot(nrm, sub(mid, F.p)) < 0) nrm = mul(nrm, -1);
        this.vertex(pa, nrm, per[k], along[i], st);
        this.vertex(pb, nrm, per[k + 1], along[i], st);
      }
      for (let i = 0; i + 1 < path.length; i++) {
        const q = base + 2 * i;
        this.quad(q, q + 2, q + 3, q + 1);
      }
      if (style?.closedPath) {
        const q = base + 2 * (path.length - 1);
        this.quad(q, base, base + 1, q + 1);
      }
    }
  }

  /** Packed vertex and index data. */
  finish(): { vertices: ArrayBuffer; indices: Uint32Array<ArrayBuffer>; count: number } {
    const n = this.count;
    const buf = new ArrayBuffer(n * KIT_VERTEX_BYTES);
    const f = new Float32Array(buf), i8 = new Int8Array(buf), u8 = new Uint8Array(buf);
    for (let k = 0; k < n; k++) {
      const o = k * KIT_VERTEX_BYTES;
      f.set(this.pos.slice(4 * k, 4 * k + 4), o / 4);
      for (let c = 0; c < 3; c++) i8[o + 16 + c] = Math.round(Math.max(-1, Math.min(1, this.nrm[3 * k + c])) * 127);
      i8[o + 19] = 0;
      for (let c = 0; c < 4; c++) u8[o + 20 + c] = this.att[4 * k + c];
      f[o / 4 + 6] = this.uv[2 * k];
      f[o / 4 + 7] = this.uv[2 * k + 1];
    }
    return { vertices: buf, indices: new Uint32Array(this.idx), count: n };
  }
}

/**
 * A point on an arch over an opening ±w wide springing at `spring` and rising `h` to its crown,
 * at t (0 the left springing … 1 the right): a half circle (h = w gives a true one), or two arcs
 * meeting at a point, each centred on the springing line toward the far side.
 */
function archCurve(t: number, w: number, spring: number, h: number, pointed: boolean): [number, number] {
  if (!pointed || h <= w) {
    const a = Math.PI * (1 - t);
    return [Math.cos(a) * w, spring + Math.sin(a) * h];
  }
  const c = (h * h - w * w) / (2 * w), R = c + w;
  if (t < 0.5) {
    const a1 = Math.atan2(h, -c);
    const a = Math.PI + (a1 - Math.PI) * (t / 0.5);
    return [c + R * Math.cos(a), spring + R * Math.sin(a)];
  }
  const a0 = Math.atan2(h, c);
  const a = a0 * (1 - (t - 0.5) / 0.5);
  return [-c + R * Math.cos(a), spring + R * Math.sin(a)];
}

/** Occlusion inside an arch: darker toward its feet and deep under it. */
function archAO(x: number, y: number, o: { open: number; y0: number; spring: number }): number {
  const inJamb = 1 - smooth(0, 0.6, o.open - Math.abs(x));
  return 0.7 + 0.3 * smooth(o.y0, o.spring, y) * (1 - 0.2 * inJamb);
}
