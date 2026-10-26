/**
 * Leaf sprays, painted once at start-up (there are no image files): one layer
 * per leaf style, each a spray of overlapping leaves round its middle, with its
 * mipmaps. The leaf cards round near crowns draw them (forest.wgsl.ts). One
 * style is not a spray: a conifer's bough, a feathered sheet (its sprigs, their
 * needles) that the boughs of the pines are cut out of.
 * ---------------------------------------------------------------------------
 * Texel (RGBA8, every channel premultiplied by coverage, so mipmaps average
 * right): coverage; tone (the leaf's own shade, darker toward its base and
 * along its midrib, deeper leaves darker); and the leaf's tilt in the card's
 * plane (x, y; ½ = facing straight out of the card), which the shader turns
 * into the leaf's normal. Texture coordinates: x right, y down the card.
 */

export const LEAF_SIZE = 256;
export const LEAF_STYLES = ['lance', 'palmate', 'ovate', 'broad', 'pinnate', 'round', 'fir'] as const;
export type LeafStyle = (typeof LEAF_STYLES)[number];
/** Mip levels per layer: 256² down to 1². */
export const LEAF_MIPS = Math.log2(LEAF_SIZE) + 1;

/** Per layer, its mip levels (level 0 first), each LEAF_SIZE >> level squared, RGBA8. */
type LeafAtlas = Uint8Array<ArrayBuffer>[][];

/** mulberry32: a small seeded generator of [0, 1) values. */
function rng(seed: number): (lo?: number, hi?: number) => number {
  let s = seed >>> 0;
  return (lo = 0, hi = 1) => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return lo + (hi - lo) * (((t ^ (t >>> 14)) >>> 0) / 4294967296);
  };
}

type Style = {
  /** Leaves in a spray, their length (in tiles) and greatest half-width (in lengths). */
  count: number;
  length: [number, number];
  width: number;
  /** Half-width along the leaf (t: 0 at the stalk, 1 at the tip), as a share of the greatest. */
  profile: (t: number) => number;
  /** How far from the middle the leaves' midpoints lie (in tiles). */
  spread: number;
};

const sinPow = (t: number, p: number) => Math.pow(Math.max(0, Math.sin(Math.PI * t)), p);

type SprayStyle = Exclude<LeafStyle, 'fir'>;

const STYLES: Record<SprayStyle, Style> = {
  // Long pointed leaves, like the lowland trees' (oak).
  lance: { count: 28, length: [0.3, 0.42], width: 0.27, profile: (t) => sinPow(Math.pow(t, 0.8), 0.75), spread: 0.31 },
  // Maple: lobed leaves (each leaf five lobes from one stalk, drawn as five leaves).
  palmate: { count: 34, length: [0.25, 0.34], width: 0.25, profile: (t) => sinPow(Math.pow(t, 0.7), 0.7), spread: 0.31 },
  // Birch: small toothed ovals.
  ovate: { count: 42, length: [0.18, 0.25], width: 0.36, profile: (t) => sinPow(Math.pow(t, 0.75), 0.6) * (1 + 0.07 * Math.sin(t * 44)), spread: 0.34 },
  // Jungle: few broad leaves with heavy midribs.
  broad: { count: 13, length: [0.4, 0.5], width: 0.33, profile: (t) => sinPow(Math.pow(t, 0.85), 0.55), spread: 0.26 },
  // Acacia: compound leaves, rows of tiny leaflets along a stalk.
  pinnate: { count: 30, length: [0.28, 0.38], width: 0.07, profile: (t) => sinPow(t, 0.5), spread: 0.31 },
  // Bushes: small round leaves, densely.
  round: { count: 52, length: [0.13, 0.19], width: 0.45, profile: (t) => sinPow(t, 0.45), spread: 0.36 },
};

type Leaf = { x: number; y: number; dx: number; dy: number; len: number; w: number; tone: number; tilt: [number, number]; profile: (t: number) => number; rib: number };

/** One leaf painted over the layer (premultiplied "over"), clipped to its bounding box. */
function paint(px: Float32Array, n: number, leaf: Leaf): void {
  const { x, y, dx, dy, len, w } = leaf;
  const ex = x + dx * len, ey = y + dy * len;
  const r = w * len + 2 / n;
  const x0 = Math.max(0, Math.floor((Math.min(x, ex) - r) * n)), x1 = Math.min(n - 1, Math.ceil((Math.max(x, ex) + r) * n));
  const y0 = Math.max(0, Math.floor((Math.min(y, ey) - r) * n)), y1 = Math.min(n - 1, Math.ceil((Math.max(y, ey) + r) * n));
  for (let j = y0; j <= y1; j++) {
    for (let i = x0; i <= x1; i++) {
      const qx = (i + 0.5) / n - x, qy = (j + 0.5) / n - y;
      const a = qx * dx + qy * dy;              // along the leaf (tiles)
      const b = -qx * dy + qy * dx;             // across it
      const t = a / len;
      if (t < 0 || t > 1) continue;
      const hw = leaf.profile(t) * w * len;
      // Distance inside the edge, in texels: one texel of antialiasing.
      const d = Math.min(hw - Math.abs(b), a, len - a) * n;
      const cov = Math.min(1, Math.max(0, d + 0.5));
      if (cov <= 0) continue;
      const across = hw > 1e-6 ? b / hw : 0;
      const rib = 1 - leaf.rib * (1 - Math.min(1, Math.abs(across) / 0.09));
      const tone = leaf.tone * (0.62 + 0.38 * t) * rib;
      // Tilt: the leaf's own facing, rounded across it and curling toward the tip.
      const tx = leaf.tilt[0] - 0.4 * across * dy + 0.25 * (t - 0.4) * dx;
      const ty = leaf.tilt[1] + 0.4 * across * dx + 0.25 * (t - 0.4) * dy;
      const o = 4 * (j * n + i);
      const k = 1 - cov;
      px[o] = cov + k * px[o];
      px[o + 1] = cov * tone + k * px[o + 1];
      px[o + 2] = cov * (0.5 + 0.5 * Math.max(-1, Math.min(1, tx))) + k * px[o + 2];
      px[o + 3] = cov * (0.5 + 0.5 * Math.max(-1, Math.min(1, ty))) + k * px[o + 3];
    }
  }
}

/** One spray: leaves over a disc, pointing outward from its middle, deeper ones (painted first) darker. */
function spray(style: SprayStyle, seed: number): Float32Array {
  const n = LEAF_SIZE;
  const st = STYLES[style];
  const r = rng(seed);
  const px = new Float32Array(n * n * 4);
  const leaves: Leaf[] = [];
  for (let k = 0; k < st.count; k++) {
    // A midpoint over the disc; the leaf points outward from the middle, give or take.
    const ang = r(0, 2 * Math.PI);
    const rad = st.spread * Math.sqrt(r());
    const dir = (rad > 0.04 ? ang : r(0, 2 * Math.PI)) + r(-0.8, 0.8);
    const dx = Math.cos(dir), dy = Math.sin(dir);
    let len = r(st.length[0], st.length[1]);
    let x = 0.5 + Math.cos(ang) * rad - 0.5 * dx * len, y = 0.5 + Math.sin(ang) * rad - 0.5 * dy * len;
    // Keep the leaf (its stalk, its tip and its width) inside the tile, with a margin.
    const inside = (v: number) => v > 0.04 && v < 0.96;
    for (let tries = 0; tries < 6 && !(inside(x) && inside(y) && inside(x + dx * len) && inside(y + dy * len)); tries++) {
      x += 0.1 * dx * len; y += 0.1 * dy * len; len *= 0.82;
    }
    if (!(inside(x) && inside(y) && inside(x + dx * len) && inside(y + dy * len)) || len < 0.45 * st.length[0]) continue;
    const depth = k / st.count; // 0: painted first, deepest
    const ta = r(0, 2 * Math.PI), tm = r(0.15, 0.6);
    const base: Leaf = {
      x, y, dx, dy, len, w: st.width, tone: r(0.45, 1) * (0.62 + 0.38 * depth),
      tilt: [Math.cos(ta) * tm, Math.sin(ta) * tm], profile: st.profile, rib: style === 'broad' ? 0.35 : style === 'pinnate' ? 0 : 0.18,
    };
    if (style === 'palmate') {
      // Five lobes from the stalk, the middle one longest.
      for (const [da, f] of [[-1.15, 0.55], [-0.58, 0.85], [0, 1], [0.58, 0.85], [1.15, 0.55]] as const) {
        const a2 = dir + da;
        leaves.push({ ...base, dx: Math.cos(a2), dy: Math.sin(a2), len: len * f });
      }
    } else if (style === 'pinnate') {
      // The stalk, then pairs of tiny leaflets along it.
      leaves.push({ ...base, w: 0.012, profile: () => 1, tone: base.tone * 0.7 });
      const pairs = 9;
      for (let p = 1; p <= pairs; p++) {
        const s = (p / (pairs + 1)) * len;
        for (const side of [-1, 1]) {
          const a2 = dir + side * 1.05;
          leaves.push({ ...base, x: x + dx * s, y: y + dy * s, dx: Math.cos(a2), dy: Math.sin(a2), len: len * 0.16 * (1 - 0.35 * (p / pairs)), w: 0.32, profile: (t) => sinPow(t, 0.55), tone: base.tone * r(0.85, 1.1) });
        }
      }
    } else {
      leaves.push(base);
    }
  }
  for (const leaf of leaves) paint(px, n, leaf);
  return px;
}

/** A conifer bough's sheet is this wide (both sides) as a share of its length: models.ts sizes boughs by it, and the strokes of fir() are laid out for it. */
export const BOUGH_ASPECT = 0.52;

/**
 * A conifer's bough, trunk to tip down the tile (x across it, the midrib at ½), painted the way the
 * spruces of the Hyrule games are: a broad spade of a branch, narrow at the trunk, widest past the
 * middle, rounded to a point at the tip; dense, a dark underlayer of needles under brushy tufts
 * that all sweep toward the tip and out to the sides, so its rim is a fringe of needle points
 * (the tufts out there are longer and some are missing). Dark toward the trunk and the midrib, pale
 * at the rim and the tip, where the light catches the new growth.
 * The sheet it is drawn on is BOUGH_ASPECT as wide as it is long, so the strokes are laid out in that
 * space and squeezed into the tile.
 */
function fir(seed: number): Float32Array {
  const n = LEAF_SIZE;
  const px = new Float32Array(n * n * 4);
  const A = BOUGH_ASPECT;
  /** A stroke from tile point (x, y) at `angle` off the bough's axis (+: toward x > ½), `length` in bough lengths. */
  const stroke = (x: number, y: number, angle: number, length: number, w: number, tone: number, tilt: [number, number], profile: (t: number) => number): Leaf => {
    const vx = Math.sin(angle) / A, vy = Math.cos(angle);
    const m = Math.hypot(vx, vy);
    return { x, y, dx: vx / m, dy: vy / m, len: length * m, w, tone, tilt, profile, rib: 0 };
  };
  /** The bough's half-width (bough lengths) a fraction y of the way to its tip: the spade. */
  // (broad from where it leaves the trunk, so the boughs round a trunk hide it; rounded to a point at the tip)
  const half = (y: number) => (A / 2) * 0.8 * Math.max(
    Math.pow(Math.max(0, Math.sin(Math.PI * Math.pow(Math.min(1, Math.max(0, y / 0.93)), 1.2))), 0.7) * (0.35 + 0.65 * Math.min(1, y / 0.3)),
    0.8 * Math.min(1, 0.85 + 3 * y) * (1 - Math.min(1, Math.max(0, (y - 0.5) / 0.35))));
  const tuft = (t: number) => Math.pow(1 - t, 0.6) * Math.min(1, 0.35 + 3 * t);
  const sprigs = 17;
  // The midrib, under everything.
  paint(px, n, stroke(0.5, 0.015, 0, 0.8, 0.012, 0.3, [0, 0.2], (t) => 1 - 0.6 * t));
  // Three passes: a dark underlayer (the depth between the tufts), the sprigs and their tufts, then a
  // last scatter of pale tufts over the outer half (the lit tops of the needles).
  for (const pass of [0, 1, 2]) {
    const r = rng(seed + 977 * pass);
    for (let k = 0; k < sprigs * 2; k++) {
      const side = k % 2 ? -1 : 1;
      const y = 0.03 + 0.8 * ((k >> 1) + (side > 0 ? 0.25 : 0.75)) / sprigs + r(-0.01, 0.01);
      // Sprigs leave the midrib steeply near the trunk and sweep forward toward the tip.
      const angle = side * (1.15 - 0.45 * y + r(-0.08, 0.08));
      // Long enough to reach the spade's edge where the sprig ends (found by halving).
      let lo = 0, hi = 1.5;
      for (let it = 0; it < 24; it++) {
        const mid = 0.5 * (lo + hi);
        if (mid * Math.sin(Math.abs(angle)) < half(y + mid * Math.cos(angle))) lo = mid; else hi = mid;
      }
      const len = lo * (pass === 0 ? 0.82 : 0.9) * r(0.92, 1.04);
      if (len < 0.02) continue;
      const tone = pass === 0 ? r(0.18, 0.3) : Math.min(1, r(0.5, 0.75) * (0.75 + 0.4 * y));
      const tilt: [number, number] = [side * r(0.15, 0.4), r(0.05, 0.3)];
      const s = stroke(0.5, y, angle, len, 0.02, tone * 0.8, tilt, (t) => 1 - 0.5 * t);
      if (pass < 2) paint(px, n, s);
      // Tufts along the sprig, both sides of it, all sweeping forward; out at the rim they are
      // longer, uneven and now and then missing: that is the fringe.
      const count = [18, 40, 22][pass];
      for (let m = 0; m < count; m++) {
        const f = pass === 2 ? r(0.35, 1) : (m + r(0.1, 0.9)) / count;
        const bx = 0.5 + s.dx * s.len * f, by = y + s.dy * s.len * f;
        for (const sgn of [1, -1]) {
          if (r() < (pass === 2 ? 0.45 : 0.06 + 0.3 * Math.max(0, f - 0.7) / 0.3)) continue;
          // (off the sprig by ~30°, kept pointing forward: within 85° of the bough's axis)
          const a2 = side * Math.min(1.48, Math.max(0.04, Math.abs(angle) + sgn * (0.5 + r(-0.15, 0.15)) - 0.12));
          const rim = Math.max(0, f - 0.6) / 0.4;
          const l2 = (0.038 + 0.045 * rim * r(0.3, 1.6)) * r(0.8, 1.2) * (1 - 0.3 * y) * (pass === 0 ? 1.3 : 1);
          const t2 = pass === 0 ? tone * r(0.8, 1.2) : Math.min(1, tone * r(0.75, 1.1) * (0.8 + 0.45 * f) + (pass === 2 ? 0.12 : 0));
          paint(px, n, stroke(bx, by, a2, l2, pass === 0 ? 0.2 : 0.12, t2, [tilt[0] + sgn * r(0.05, 0.3), tilt[1] + r(-0.1, 0.25)], tuft));
        }
      }
    }
  }
  // The leader: a tuft straight down the axis at the tip.
  const r = rng(seed + 31);
  for (let m = 0; m < 9; m++) {
    const y = 0.76 + 0.1 * m / 9;
    for (const sgn of [1, -1]) paint(px, n, stroke(0.5, y, sgn * r(0.25, 0.55), r(0.04, 0.065), 0.15, r(0.7, 1), [sgn * 0.2, 0.2], tuft));
  }
  // (the odd fringe tuft reaches the tile's edge: clear a border, or the sheet would show a cut)
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) if (i < 2 || j < 2 || i >= n - 2 || j >= n - 2) px.fill(0, 4 * (j * n + i), 4 * (j * n + i) + 4);
  return px;
}

/** Premultiplied float texels to RGBA8, and their mip chain (box-filtered, which premultiplication makes exact). */
function mips(px: Float32Array): Uint8Array<ArrayBuffer>[] {
  const out: Uint8Array<ArrayBuffer>[] = [];
  let level = px;
  let n = LEAF_SIZE;
  for (;;) {
    const bytes = new Uint8Array(level.length);
    for (let k = 0; k < level.length; k++) bytes[k] = Math.round(Math.min(1, Math.max(0, level[k])) * 255);
    out.push(bytes);
    if (n === 1) break;
    const h = n >> 1;
    const next = new Float32Array(h * h * 4);
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < h; i++) {
        for (let c = 0; c < 4; c++) {
          const at = (x: number, y: number) => level[4 * (y * n + x) + c];
          next[4 * (j * h + i) + c] = 0.25 * (at(2 * i, 2 * j) + at(2 * i + 1, 2 * j) + at(2 * i, 2 * j + 1) + at(2 * i + 1, 2 * j + 1));
        }
      }
    }
    level = next;
    n = h;
  }
  return out;
}

/** Every leaf style's spray, in LEAF_STYLES order, with mipmaps. */
export function buildLeafAtlas(): LeafAtlas {
  return LEAF_STYLES.map((style, k) => mips(style === 'fir' ? fir(0x1eaf + 7919 * k) : spray(style, 0x1eaf + 7919 * k)));
}
