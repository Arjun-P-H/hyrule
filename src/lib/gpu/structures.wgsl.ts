/**
 * The Eyries' structures in WGSL: the kit's pieces drawn instanced
 * (gpu/structures.ts culls and buckets them), shaded by material and palette.
 * ---------------------------------------------------------------------------
 * Vertex: a piece's unit-space vertex (skyworld/kit/mesh.ts) scaled by its
 * instance per axis (radial vertices put at the instance's radius plus their
 * metric offset), animated (cloth in the wind), turned by the yaw and placed.
 * Its surface coordinates leave metric: × the scale of the axis each follows,
 * round a cone or dome θ × its radius there and down it the distance below the
 * apex, so painted detail keeps its size on any stretch.
 *
 * Fragment: one shader for every material — ashlar (courses, joints, tints,
 * weathering, moss), scale tiles, ribbed domes, marble, timber, gilding, iron,
 * bronze, glass, crystal, cloth with the House emblem, mosaics, paving, plaster,
 * lacquer, sunstone lanterns, skystone keystones, water — and the facades'
 * windows painted per floor, which light up from dusk to dawn. Lit by the
 * shared light API (key light with the realm's, the trees' (and the buildings'
 * own, drawn into that map) and the hero's shadows; sky fill; bounce), fogged
 * as the realm (its reach included), shown through display().
 *
 * Needs, from the raster shader: u (Frame), vnoise, hash3, the light API with
 * lampLevel(), skyReflect, surfaceFogOd, fogOd, display, underwater, W_LEVEL,
 * skyShadowAt, treeShadowAt, heroShadowAt, rlReachOd, rlFogged, project().
 */

import { ANIM, FACADE, MAT } from '../skyworld/kit/mesh';
import { DOOR_HALF } from '../skyworld/kit/interiors';
import { PALETTES } from '../skyworld/kit/palettes';
import { wgslF32 as f } from './worldgen.wgsl';

const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;
const arr = (type: string, xs: string[]) => `array<${type}, ${xs.length}>(${xs.join(', ')})`;
const pal = (name: string, pick: (p: (typeof PALETTES)[number]) => readonly number[]) =>
  `const ${name}: array<vec3f, ${PALETTES.length}> = ${arr('vec3f', PALETTES.map((p) => v3(pick(p))))};`;

/** `shadow`: with the trees' shadow map (forest.wgsl.ts), the pass drawing structures into it. */
export function structuresWGSL(shadow: boolean): string {
  return /* wgsl */ `
// ---- instances (structures.ts packs them) -------------------------------------------------------
struct StInst {
  p: vec4f, // position (chunk-local m), yaw
  s: vec4f, // scale x, y, z; w: bits — palette (6) | seed (16) << 6 | flags (10) << 22
};
@group(1) @binding(40) var<storage, read> stInst: array<StInst>;
@group(1) @binding(41) var<storage, read> stDraw: array<u32>;

const ST_UPPER: u32 = 1u;
const ST_RUINED: u32 = 2u;
const ST_DARK: u32 = 4u;
const ST_LACQUER: u32 = 8u;
const ST_TIMBER: u32 = 16u;
const ST_PLASTER: u32 = 32u;
const ST_INDOOR: u32 = 64u;

// ---- palettes (kit/palettes.ts) --------------------------------------------------------------------
${pal('PL_STONE', (p) => p.stone)}
${pal('PL_TRIM', (p) => p.trim)}
${pal('PL_MARBLE', (p) => p.marble)}
${pal('PL_ROOF', (p) => p.roof)}
${pal('PL_ROOF2', (p) => p.roof2)}
${pal('PL_DOME', (p) => p.dome)}
${pal('PL_TIMBER', (p) => p.timber)}
${pal('PL_METAL', (p) => p.metal)}
${pal('PL_IRON', (p) => p.iron)}
${pal('PL_BRONZE', (p) => p.bronze)}
${pal('PL_CLOTH', (p) => p.cloth)}
${pal('PL_CLOTH2', (p) => p.cloth2)}
${pal('PL_EMBLEM', (p) => p.emblem)}
${pal('PL_PLASTER', (p) => p.plaster)}
${pal('PL_LACQUER', (p) => p.lacquer)}
${pal('PL_GLASS', (p) => p.glass)}
${pal('PL_CRYSTAL', (p) => p.crystal)}
${pal('PL_LAMP', (p) => p.lamp)}
${pal('PL_WINDOW', (p) => p.window)}
${pal('PL_MOSAIC0', (p) => p.mosaic[0])}
${pal('PL_MOSAIC1', (p) => p.mosaic[1])}
${pal('PL_MOSAIC2', (p) => p.mosaic[2])}
${pal('PL_PAVING', (p) => p.paving)}
const PL_STONEK: array<vec4f, ${PALETTES.length}> = ${arr('vec4f', PALETTES.map((p) => `vec4f(${f(p.moss)}, ${f(p.wear)}, ${f(p.course)}, ${f(p.block)})`))};

// ---- vertex ------------------------------------------------------------------------------------------
struct StOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,                     // metric surface coordinates
  @location(3) lp: vec3f,                     // piece-local position, scaled (m)
  @location(4) @interpolate(flat) m: vec4u,   // material, facade, palette, seed | flags << 16
  @location(5) ao: f32,
  @location(6) @interpolate(flat) fw: vec2f,  // facade: the face's width round or across it (m), the piece's height (m)
};

fn stAxis(s: vec3f, a: u32) -> f32 { return select(select(s.x, s.y, a == 1u), s.z, a == 2u); }

struct StPlaced { world: vec3f, n: vec3f, lp: vec3f, uv: vec2f, fw: vec2f };

// A kit vertex placed by its instance (shared by the main pass and the shadow pass).
fn stPlace(pos: vec4f, nrm: vec3f, attr: vec4u, uv: vec2f, I: StInst) -> StPlaced {
  let s = I.s.xyz;
  let bits = attr.w;
  let ua = bits & 3u;
  let va = (bits >> 2u) & 3u;
  let anim = (bits >> 4u) & 7u;
  let radial = (bits >> 7u) & 1u;
  let word = bitcast<u32>(I.s.w);
  let seed = f32((word >> 6u) & 0xffffu);
  var lp: vec3f;
  var n = nrm;
  var mu: f32;
  var mv: f32;
  if (radial == 1u) {
    let r = s.x + pos.w;
    lp = vec3f(pos.x * r, pos.y * s.y, pos.z * r);
    mu = uv.x * r;
    mv = uv.y;
  } else {
    lp = pos.xyz * s;
    n = nrm / s;
    if (ua == 3u) { mu = uv.x * length(pos.xz) * s.x; } else { mu = uv.x * stAxis(s, ua); }
    if (va == 3u) {
      let horiz = select(abs(pos.z) * s.z, length(pos.xz) * s.x, ua == 3u);
      mv = length(vec2f(horiz, (1.0 - pos.y) * s.y));
    } else { mv = uv.y * stAxis(s, va); }
  }
  let t = u.time;
  let ph = seed * 0.0123;
  if (anim == ${ANIM.cloth}u || anim == ${ANIM.sway}u) {
    // Cloth in the wind: the farther from where it hangs (its v holds that distance), the more it sways;
    // sails only belly and ease.
    let d = uv.y;
    let w = pow(clamp(d / 4.5, 0.0, 1.6), 1.5) * select(1.0, 0.12, anim == ${ANIM.sway}u);
    lp.z += w * (0.55 + 0.45 * sin(t * 1.7 + ph + d * 0.9)) * 0.85;
    lp.x += w * sin(t * 2.3 + d * 1.3 + ph) * 0.22;
    lp.y += w * sin(t * 2.9 - d * 1.7 + ph) * 0.12;
  } else if (anim == ${ANIM.rotor}u) {
    // a wind-wheel's sails turning about the piece's z axis
    let a = t * (0.35 + 0.25 * fract(ph * 7.0)) + ph;
    let ca = cos(a);
    let sa = sin(a);
    lp = vec3f(ca * lp.x - sa * lp.y, sa * lp.x + ca * lp.y, lp.z);
    n = vec3f(ca * n.x - sa * n.y, sa * n.x + ca * n.y, n.z);
  } else if (anim == ${ANIM.float}u) {
    // moored: rising and falling a little on the wind
    lp.y += 0.45 * sin(t * 0.5 + ph * 3.0);
  } else if (anim == ${ANIM.door}u) {
    // a doorway's edge (interiors.ts): DOOR_HALF metres from the door's axis (local +z), on the
    // round wall's circle there, or in the square wall's face
    let side = select(-1.0, 1.0, pos.x > 0.0);
    lp.x = side * ${f(DOOR_HALF)};
    if ((attr.z & ${FACADE.FLAT}u) == 0u) {
      let rr = length(pos.xz) * s.x;
      lp.z = sqrt(max(rr * rr - ${f(DOOR_HALF * DOOR_HALF)}, 0.0));
      let half = asin(min(${f(DOOR_HALF)} / max(rr, 1e-3), 1.0));
      mu = select(1.5707963 + half, 7.8539816 - half, side > 0.0) * s.x;
      if (abs(nrm.z) > 0.5) { n = vec3f(lp.x, 0.0, lp.z) / max(rr, 1e-3); }
    } else {
      mu = lp.x;
    }
  } else if (anim == ${ANIM.swim}u) {
    // swimming round a slow circle (its radius and pace from the seed), facing its way, the tail beating
    let R = 2.0 + 4.0 * fract(ph * 13.1);
    let a = t * (0.3 + 0.25 * fract(ph * 5.7)) / R + ph * 6.2831853;
    lp.x += sin(t * 7.0 + lp.z * 9.0 + ph * 3.0) * 0.12 * max(0.1 - lp.z, 0.0);
    let ca = cos(a);
    let sa = sin(a);
    lp = vec3f(ca * lp.x - sa * lp.z + R * ca, lp.y + 0.08 * sin(t * 0.7 + ph), sa * lp.x + ca * lp.z + R * sa);
    n = vec3f(ca * n.x - sa * n.z, n.y, sa * n.x + ca * n.z);
  } else if (anim == ${ANIM.bob}u) {
    // floating: a slow rise and fall and a slow turn
    let a = t * 0.12 + ph;
    let ca = cos(a);
    let sa = sin(a);
    lp = vec3f(ca * lp.x + sa * lp.z, lp.y + 1.6 * sin(t * 0.37 + ph * 3.0), -sa * lp.x + ca * lp.z);
    n = vec3f(ca * n.x + sa * n.z, n.y, -sa * n.x + ca * n.z);
  }
  let c = cos(I.p.w);
  let sn = sin(I.p.w);
  var o: StPlaced;
  o.world = vec3f(I.p.x + c * lp.x + sn * lp.z, I.p.y + lp.y, I.p.z - sn * lp.x + c * lp.z);
  o.n = normalize(vec3f(c * n.x + sn * n.z, n.y, -sn * n.x + c * n.z));
  o.lp = lp;
  o.uv = vec2f(mu, mv);
  // The face's width (round: 2π × the radius; flat: 2 × the half width along u) and the piece's height.
  let isRound = (attr.z & ${FACADE.FLAT}u) == 0u;
  o.fw = vec2f(select(2.0 * stAxis(s, ua), 6.2831853 * s.x, isRound), s.y);
  return o;
}

@vertex
fn vsStructure(@location(0) pos: vec4f, @location(1) nrm: vec4f, @location(2) attr: vec4u, @location(3) uv: vec2f,
  @builtin(instance_index) ii: u32) -> StOut {
  let I = stInst[stDraw[ii]];
  let P = stPlace(pos, nrm.xyz, attr, uv, I);
  var o: StOut;
  o.rel = P.world - u.camPos;
  o.pos = project(o.rel);
  o.n = P.n;
  o.uv = P.uv;
  o.lp = P.lp;
  let word = bitcast<u32>(I.s.w);
  o.m = vec4u(attr.x, attr.z, word & 63u, ((word >> 6u) & 0xffffu) | ((word >> 22u) << 16u));
  o.ao = f32(attr.y) / 255.0;
  o.fw = P.fw;
  return o;
}

// ---- patterns -------------------------------------------------------------------------------------------

fn stH(a: f32, b: f32, salt: u32) -> f32 {
  return f32(hash3(i32(floor(a)), i32(floor(b)), i32(salt)) & 0xffffu) / 65535.0;
}

// Ashlar: courses of blocks with mortar joints, a tint per block, weathering streaks, moss where water lies.
fn stAshlar(uv: vec2f, base: vec3f, pal: u32, seed: u32, n: vec3f, aa: f32) -> vec3f {
  let K = PL_STONEK[pal];
  let H = K.z;
  let L = K.w;
  let row = floor(uv.y / H);
  let x = uv.x / L + 0.5 * (row - 2.0 * floor(row * 0.5)) + 0.23 * stH(row, f32(seed), 3101u);
  let cell = vec2f(floor(x), row);
  let fx = fract(x);
  let fy = fract(uv.y / H);
  let dj = min(min(fx, 1.0 - fx) * L, min(fy, 1.0 - fy) * H);
  let joint = (1.0 - smoothstep(0.012, 0.03 + aa, dj)) * (1.0 - smoothstep(0.15, 0.45, aa));
  let tint = stH(cell.x + f32(seed) * 0.37, cell.y, 3103u);
  var c = base * (0.86 + 0.26 * tint);
  // a block's face slightly darker toward its edges (dressed, worn round)
  c *= 1.0 - 0.06 * (1.0 - smoothstep(0.0, 0.12, dj)) * (1.0 - smoothstep(0.15, 0.45, aa));
  c = mix(c, base * 0.45, joint);
  // weathering: broad stains and streaks running down
  let wear = K.y;
  let stain = 0.5 + 0.5 * vnoise(uv.x * 0.21, uv.y * 0.09, 3107 + i32(seed & 255u));
  let streak = smoothstep(0.55, 0.9, 0.5 + 0.5 * vnoise(uv.x * 0.9, uv.y * 0.06, 3109));
  c *= 1.0 - wear * (0.22 * stain + 0.18 * streak);
  return c;
}

// Scale tiles laid like feathers: rows down the slope, each tile's rounded lip over the next.
fn stTiles(uv: vec2f, base: vec3f, seed: u32, aa: f32) -> vec3f {
  let W = 0.44;
  let Hr = 0.3;
  let row = floor(uv.y / Hr);
  let x = uv.x / W + 0.5 * (row - 2.0 * floor(row * 0.5));
  let fx = fract(x) - 0.5;
  let fy = fract(uv.y / Hr);
  let edge = 0.5 + 0.5 * sqrt(max(1.0 - 4.0 * fx * fx, 0.0));
  let far = smoothstep(0.08, 0.25, aa);
  let tint = stH(floor(x) + f32(seed) * 0.13, row, 3121u);
  var shade = mix(0.78, 1.1, smoothstep(0.0, edge, fy));
  // the shadow under the lip of the tile above, and the gap where the tiles meet
  shade *= 1.0 - 0.45 * (1.0 - smoothstep(0.0, 0.1 + aa * 2.0, abs(fy - edge))) * step(edge, fy + 0.08);
  shade *= 1.0 - 0.3 * (1.0 - smoothstep(0.0, 0.06 + aa, 0.5 - abs(fx))) * step(fy, edge);
  shade = mix(shade, 0.95, far);
  return base * shade * (0.88 + 0.24 * tint);
}

// Ribbed sheathing (domes): ribs converging to the crown, seams across, patina streaks.
fn stDome(uv: vec2f, base: vec3f, rib: vec3f, aa: f32) -> vec3f {
  let ribW = 2.6;
  let rx = abs(fract(uv.x / ribW) - 0.5) * ribW;
  let ribM = (1.0 - smoothstep(0.08, 0.16 + aa, rx)) * (1.0 - smoothstep(0.4, 1.0, aa));
  let seam = (1.0 - smoothstep(0.015, 0.04 + aa, abs(fract(uv.y / 0.9) - 0.5) * 0.9 - 0.41)) * (1.0 - smoothstep(0.1, 0.3, aa));
  let pat = 0.5 + 0.5 * vnoise(uv.x * 0.7, uv.y * 0.18, 3131);
  var c = base * (0.82 + 0.3 * pat) * (1.0 - 0.15 * seam);
  return mix(c, rib, ribM);
}

// Flagstones: irregular slabs, worn joints.
fn stPaving(uv: vec2f, base: vec3f, aa: f32) -> vec3f {
  let q = uv / vec2f(1.1, 0.75);
  let row = floor(q.y);
  let x = q.x + 0.37 * row + 0.3 * stH(row, 7.0, 3141u);
  let fx = fract(x);
  let fy = fract(q.y);
  let dj = min(min(fx, 1.0 - fx) * 1.1, min(fy, 1.0 - fy) * 0.75);
  let joint = (1.0 - smoothstep(0.01, 0.03 + aa, dj)) * (1.0 - smoothstep(0.2, 0.5, aa));
  let tint = stH(floor(x), row, 3143u);
  return mix(base * (0.85 + 0.3 * tint), base * 0.5, joint);
}

// A radial mosaic (plazas, rotunda floors) round the piece's centre: rings and rays, a star at the middle.
fn stMosaic(lp: vec2f, pal: u32, aa: f32) -> vec3f {
  let r = length(lp);
  let th = atan2(lp.y, lp.x + 1e-6);
  let ring = floor(r / 1.6);
  let rays = 16.0 + 8.0 * floor(ring / 3.0);
  let sector = floor((th / 6.2831853 + 0.5) * rays);
  let band = fract(r / 1.6);
  var c = PL_MOSAIC0[pal];
  if (ring - 2.0 * floor(ring * 0.5) > 0.5) { c = mix(c, PL_MOSAIC1[pal], 0.85); }
  if (band < 0.12) { c = PL_MOSAIC2[pal]; }
  if (sector - 2.0 * floor(sector * 0.5) > 0.5 && ring > 0.5) { c *= 0.9; }
  // the star at the middle
  let star = abs(cos(th * 4.0)) * 2.2 + 0.8;
  if (r < star) { c = mix(PL_MOSAIC2[pal], PL_MOSAIC1[pal], smoothstep(star - 0.3, star, r)); }
  // tesserae
  let tq = abs(fract(lp / 0.14) - 0.5);
  let tess = (1.0 - smoothstep(0.38, 0.5, max(tq.x, tq.y)));
  let jit = stH(floor(lp.x / 0.14), floor(lp.y / 0.14), 3147u);
  let wear = 0.5 + 0.5 * vnoise(lp.x * 0.35, lp.y * 0.35, 3149);
  return c * (0.82 + 0.16 * jit) * (0.85 + 0.2 * wear) * (1.0 - 0.25 * (1.0 - tess) * (1.0 - smoothstep(0.02, 0.06, aa)));
}

// The House emblem on cloth (uv: across −0.8 … 0.8, metres down from where it hangs): a winged ring.
fn stEmblem(uv: vec2f) -> f32 {
  let p = vec2f(uv.x, uv.y - 1.75);
  let ring = abs(length(p) - 0.32) - 0.06;
  // wings: two arcs sweeping up and out
  let q = vec2f(abs(p.x), p.y);
  let wing = length(q - vec2f(0.32, -0.62)) - 0.62;
  let wingCut = length(q - vec2f(0.38, -0.85)) - 0.66;
  let w = max(abs(wing) - 0.07, -wingCut);
  let body = max(abs(p.x) - 0.05, abs(p.y + 0.15) - 0.42);
  return 1.0 - smoothstep(-0.01, 0.02, min(min(ring, w), body));
}

// ---- facades: windows painted per floor -------------------------------------------------------------------

struct StWin { glass: f32, frame: f32, lit: f32, tint: vec3f, band: f32, dark: f32 };

// Signed distance to a window: a rectangle of half width hw from y0 up to ys, then an arch to its
// crown (pointed: two arcs of radius 2 hw; round: a half circle).
fn stWinSd(p: vec2f, hw: f32, y0: f32, ys: f32, pointed: bool) -> f32 {
  if (p.y < ys) { return max(abs(p.x) - hw, y0 - p.y); }
  if (pointed) {
    let R = 2.0 * hw;
    let c = R - hw;
    return max(length(p - vec2f(-c, ys)) - R, length(p - vec2f(c, ys)) - R);
  }
  return length(p - vec2f(0.0, ys)) - hw;
}

fn stFacade(style: u32, uv: vec2f, fw: vec2f, isRound: bool, flags: u32, seed: u32, aa: f32) -> StWin {
  var o: StWin;
  o.glass = 0.0; o.frame = 0.0; o.lit = 0.0; o.tint = vec3f(1.0); o.band = 0.0; o.dark = 0.0;
  if (style == 0u) { return o; }
  let uu = select(uv.x + 0.5 * fw.x, uv.x, isRound);
  let near = 1.0 - smoothstep(0.2, 0.5, aa);
  // Corner quoins on flat faces: long and short dressed blocks up each edge.
  if (!isRound) {
    let edge = min(uu, fw.x - uu);
    let course = floor(uv.y / 0.55);
    let qw = select(0.45, 0.8, course - 2.0 * floor(course * 0.5) > 0.5);
    o.band = max(o.band, (1.0 - smoothstep(qw - 0.02, qw + 0.02 + aa, edge)) * near * 0.85);
  }
  var pitch: f32;
  var hw: f32;
  var y0: f32;
  var ys: f32;
  var floorH = 1e6;
  var pointed = true;
  var row = 0.0;
  var lv = uv.y;
  if (style == ${FACADE.roost}u) {
    pitch = 3.2; hw = 0.42; y0 = 1.05; ys = 2.2; floorH = 3.6;
    row = floor(uv.y / floorH);
    lv = uv.y - row * floorH;
    // a string course at every floor (and its shadow beneath)
    if (row > 0.5 || (flags & ST_UPPER) != 0u) {
      o.band = max(o.band, (1.0 - smoothstep(0.1, 0.13 + aa, abs(lv - 0.05))) * near);
      o.dark = (1.0 - smoothstep(0.14, 0.3 + aa, lv)) * step(0.14, lv) * near * 0.35;
    }
    if ((row < 0.5 && (flags & ST_UPPER) == 0u) || uv.y > fw.y - 1.2) { return o; }
  } else if (style == ${FACADE.lancet}u) {
    pitch = 4.8; hw = 0.72; y0 = 0.2 * fw.y; ys = 0.62 * fw.y;
  } else if (style == ${FACADE.hall}u) {
    pitch = 3.6; hw = 0.62; y0 = 0.35 * fw.y; ys = 0.35 * fw.y + 2.0; pointed = false;
    if (fw.y < 4.5) { y0 = 0.3 * fw.y; ys = 0.62 * fw.y; }
  } else if (style == ${FACADE.slit}u) {
    pitch = 4.4; hw = 0.1; floorH = 6.0;
    row = floor(uv.y / floorH);
    lv = uv.y - row * floorH;
    y0 = 2.4; ys = 3.6;
    if (row < 0.5 || uv.y > fw.y - 1.5) { return o; }
  } else if (style == ${FACADE.rose}u) {
    // one rose window high in the middle of the face, with its tracery
    let c = vec2f(0.5 * fw.x, 0.66 * fw.y);
    let R = min(0.3 * fw.x, 0.2 * fw.y);
    let q = vec2f(uu, uv.y) - c;
    let r = length(q);
    let d = r - R;
    o.frame = (1.0 - smoothstep(0.0, 0.2 + aa, abs(d + 0.1))) * step(d, 0.25);
    if (d < -0.12) {
      let th = atan2(q.y, q.x);
      let spoke = abs(fract(th / 6.2831853 * 12.0) - 0.5) * 2.0 * 3.14159 / 12.0 * r;
      let ringT = abs(r - 0.55 * R);
      let tr = 1.0 - smoothstep(0.05, 0.1 + aa, min(spoke, ringT));
      o.glass = 1.0 - tr;
      o.frame = max(o.frame, tr);
      let k = floor(th / 6.2831853 * 12.0) + 12.0 * step(0.55 * R, r);
      o.tint = mix(vec3f(0.25, 0.4, 1.0), vec3f(1.0, 0.35, 0.25), stH(k, 1.0, 3153u));
      o.lit = 1.0;
    }
    return o;
  } else if (style == ${FACADE.arcade}u) {
    pitch = 3.4; hw = 1.15; y0 = -1.0; ys = 2.6; pointed = false;
  } else if (style == ${FACADE.lattice}u) {
    // timber lattice screens, a row per storey, framed in dark timber
    pitch = 2.6; hw = 0.85; floorH = 3.6; pointed = false;
    row = floor(uv.y / floorH);
    lv = uv.y - row * floorH;
    y0 = 0.9; ys = 2.9;
    if (uv.y > fw.y - 0.8) { return o; }
  } else if (style == ${FACADE.belfry}u) {
    if (uv.y > fw.y - 7.5) {
      // the belfry: tall louvred openings under the top
      pitch = 3.0; hw = 0.62; y0 = fw.y - 7.0; ys = fw.y - 3.2;
      row = 9.0;
    } else {
      pitch = 4.4; hw = 0.1; floorH = 6.0;
      row = floor(uv.y / floorH);
      lv = uv.y - row * floorH;
      y0 = 2.4; ys = 3.6;
      if (row < 0.5) { return o; }
    }
  } else {
    return o;
  }
  let count = max(1.0, round(fw.x / pitch));
  let pp = fw.x / count;
  let k = floor(uu / pp);
  let x = (uu - k * pp) - 0.5 * pp;
  var d = stWinSd(vec2f(x, lv), hw, y0, ys, pointed);
  if (style == ${FACADE.lattice}u) {
    d = max(abs(x) - hw, abs(lv - 0.5 * (y0 + ys)) - 0.5 * (ys - y0));
  }
  // Near, the window itself; far (its pixels larger than its frame), what it averages to over the wall:
  // its share of the wall's area, lit as often as the windows are.
  let fade = 1.0 - smoothstep(0.25, 0.6, aa);
  let span = min(floorH, fw.y);
  let cover = clamp((2.0 * hw * max(ys - y0, 0.0) + 1.5 * hw * hw) / (pp * span), 0.0, 0.5) * select(1.0, 0.0, lv < y0 - 1.0 && floorH < 1e5);
  o.frame = (1.0 - smoothstep(0.1, 0.16 + aa, abs(d - 0.06))) * fade;
  o.glass = mix(cover, 1.0 - smoothstep(-0.02, aa, d), fade);
  if (style == ${FACADE.lattice}u && o.glass > 0.0) {
    // the lattice: dark timber bars across the paper screen
    let gq = abs(fract(vec2f(x, lv) / 0.42) - 0.5);
    let bar = 1.0 - smoothstep(0.38, 0.45, max(gq.x, gq.y));
    o.dark = max(o.dark, (1.0 - bar) * o.glass * 0.7);
    o.tint = vec3f(1.0, 0.92, 0.75);
  }
  if (style == ${FACADE.lancet}u && o.glass > 0.0) {
    // tracery: a mullion and a small round light in the arch
    let mull = 1.0 - smoothstep(0.04, 0.07 + aa, abs(x));
    let oc = 1.0 - smoothstep(0.04, 0.07 + aa, abs(length(vec2f(x, lv - ys - 0.55 * hw)) - 0.3 * hw));
    let tr = max(mull * step(lv, ys + 0.3 * hw), oc);
    o.frame = max(o.frame, tr * o.glass);
    o.glass *= 1.0 - tr;
    o.tint = mix(vec3f(0.3, 0.45, 1.0), vec3f(1.0, 0.45, 0.3), stH(k, floor(lv / 1.3), 3155u + seed));
  }
  let h = stH(k + f32(seed) * 0.71, row, 3157u);
  o.lit = mix(0.72, select(0.0, 1.0, h < 0.72), fade);
  if (style == ${FACADE.arcade}u) { o.lit = 0.35; }
  if (style == ${FACADE.belfry}u && row > 8.5) { o.lit = 0.0; o.dark = o.glass; }
  return o;
}

// ---- shading ----------------------------------------------------------------------------------------------

struct StSurf { albedo: vec3f, emit: vec3f, gloss: f32, metal: f32 };

// A surface seen from inside its building (from behind), as the Aelari finish an inside: walls
// limewashed over a timber dado a metre high (lp.y is metres above the piece's own floor), the
// windows bright with the day outside (dark by night, when the room's own lamps light it);
// ceilings boarded between beams; under a roof its rafters; inside a dome the night sky, gold
// stars on deep blue.
fn stInterior(mat: u32, facade: u32, uv: vec2f, lp: vec3f, n: vec3f, fw: vec2f, pal: u32, seed: u32, flags: u32, lamps: f32, aa: f32, rd: vec3f) -> StSurf {
  var s: StSurf;
  s.emit = vec3f(0.0); s.gloss = 0.0; s.metal = 0.0;
  let grain = 0.92 + 0.12 * vnoise(uv.x * 1.3, uv.y * 0.25, 3401);
  if (mat == ${MAT.glass}u) {
    // a greenhouse's glass from inside: the sky through its panes, between the iron ribs
    let g = abs(fract(uv / 0.8) - 0.5);
    let rib = 1.0 - smoothstep(0.42, 0.47, max(g.x, g.y));
    s.albedo = mix(PL_IRON[pal], vec3f(0.02), rib);
    s.emit = skyReflect(rd) * 0.85 * rib;
    return s;
  }
  if (mat == ${MAT.dome}u) {
    // stars in cells of the sphere of directions from the dome's middle (round wherever they lie)
    let d = normalize(lp + vec3f(0.0, 0.01, 0.0));
    let c = floor(d * 22.0);
    let st = stH(c.x + 37.0 * c.z, c.y, 3403u);
    let centre = normalize((c + vec3f(0.25 + 0.5 * stH(c.x, c.z, 3413u), 0.5, 0.25 + 0.5 * stH(c.z, c.y, 3415u))) / 22.0);
    let star = step(0.8, st) * (1.0 - smoothstep(0.004, 0.008 + 0.4 * aa / max(length(lp), 1.0), acos(clamp(dot(d, centre), -1.0, 1.0))));
    s.albedo = mix(vec3f(0.04, 0.07, 0.2), PL_METAL[pal], star);
    s.emit = PL_METAL[pal] * star * 0.25;
    return s;
  }
  if (mat == ${MAT.roof}u || mat == ${MAT.roof2}u) {
    // rafters every 0.6 m over boarding
    let raft = 1.0 - smoothstep(0.08, 0.12 + aa, abs(fract(uv.x / 0.6) - 0.5));
    s.albedo = mix(PL_TIMBER[pal] * 0.8, PL_TIMBER[pal] * 0.45, raft) * grain;
    return s;
  }
  if (n.y > 0.7) {
    // a floor: boards 0.22 m wide, their ends staggered
    let row = floor(uv.y / 0.22);
    let joint = 1.0 - smoothstep(0.02, 0.04 + aa, abs(fract(uv.y / 0.22) - 0.5) * 0.22 - 0.09);
    let end = 1.0 - smoothstep(0.01, 0.03 + aa, abs(fract(uv.x / 2.4 + stH(row, 0.0, 3409u)) - 0.5) * 2.4 - 1.17);
    let board = 0.85 + 0.25 * stH(row, floor(uv.x / 2.4 + stH(row, 0.0, 3409u)), 3411u);
    s.albedo = (PL_TIMBER[pal] + vec3f(0.16, 0.1, 0.05)) * 1.3 * board * grain * (1.0 - 0.45 * max(joint, end));
    return s;
  }
  if (n.y < -0.7) {
    // a ceiling (a cap seen from under it): boards, a beam every 1.8 m
    let beam = 1.0 - smoothstep(0.1, 0.14 + aa, abs(fract(uv.x / 1.8) - 0.5));
    s.albedo = mix(PL_TIMBER[pal] * 1.1, PL_TIMBER[pal] * 0.55, beam) * grain;
    return s;
  }
  // a wall: limewash over a dado, a dark skirting
  let wash = vec3f(0.68, 0.63, 0.55) * (0.95 + 0.06 * vnoise(uv.x * 0.7, uv.y * 0.7, 3405));
  let dado = 1.0 - smoothstep(0.98, 1.02 + aa, lp.y);
  s.albedo = mix(wash, PL_TIMBER[pal] * grain, dado);
  s.albedo *= 1.0 - 0.5 * (1.0 - smoothstep(0.08, 0.12 + aa, lp.y));
  // its windows, from inside: the day through the glass
  if (facade != 0u) {
    let w = stFacade(facade & 0x7fu, uv, fw, (facade & ${FACADE.FLAT}u) == 0u, flags, seed, aa);
    s.albedo = mix(s.albedo, PL_TIMBER[pal] * 0.6, w.frame);
    s.albedo = mix(s.albedo, vec3f(0.02, 0.025, 0.035), w.glass);
    s.emit = (skyAmb(vec3f(0.0, 1.0, 0.0), 1.0) * 1.6 * (1.0 - lamps) + vec3f(0.01, 0.015, 0.035)) * w.glass;
  }
  return s;
}

@fragment
fn fsStructure(i: StOut) -> @location(0) vec4f {
  // (derivatives first, in uniform control flow)
  let aa = length(fwidth(i.uv));
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  let p = u.camPos + i.rel;
  var n = normalize(i.n);
  let pal = i.m.z;
  let seed = i.m.w & 0xffffu;
  let flags = i.m.w >> 16u;
  // an instance's material variant (orchard lacquer, timber, plaster)
  var mat = i.m.x;
  if ((flags & ST_LACQUER) != 0u && (mat == ${MAT.marble}u || mat == ${MAT.trim}u)) { mat = ${MAT.lacquer}u; }
  if ((flags & ST_TIMBER) != 0u && (mat == ${MAT.stone}u || mat == ${MAT.trim}u || mat == ${MAT.paving}u)) { mat = ${MAT.timber}u; }
  if ((flags & ST_PLASTER) != 0u && mat == ${MAT.stone}u) { mat = ${MAT.plaster}u; }
  let uv = i.uv;
  let lamps = select(lampLevel(), 0.0, (flags & ST_DARK) != 0u);
  let night = lamps;
  // seen from behind: from inside the building (what is double-sided is its own)
  let inside = dot(n, rd) > 0.0 && mat != ${MAT.cloth}u && mat != ${MAT.cloth2}u && mat != ${MAT.leaf}u && mat != ${MAT.petal}u
    && mat != ${MAT.koi}u && mat != ${MAT.water}u && mat != ${MAT.woven}u;
  var s: StSurf;
  s.albedo = PL_STONE[pal]; s.emit = vec3f(0.0); s.gloss = 0.0; s.metal = 0.0;
  let K = PL_STONEK[pal];
  var mossy = K.x;
  switch (mat) {
    case ${MAT.stone}u: { s.albedo = stAshlar(uv, PL_STONE[pal], pal, seed, n, aa); }
    case ${MAT.trim}u: {
      let j = 1.0 - smoothstep(0.01, 0.03 + aa, abs(fract(uv.x / 1.7) - 0.5) * 1.7 - 0.83);
      s.albedo = PL_TRIM[pal] * (0.92 + 0.12 * vnoise(uv.x * 0.8, uv.y * 0.8, 3161)) * (1.0 - 0.25 * j * (1.0 - smoothstep(0.1, 0.3, aa)));
      mossy *= 0.6;
    }
    case ${MAT.marble}u: {
      let vein = smoothstep(0.82, 0.97, 1.0 - abs(vnoise(uv.x * 0.6 + i.lp.y * 0.3, uv.y * 0.6, 3163)));
      s.albedo = PL_MARBLE[pal] * (0.95 + 0.08 * vnoise(uv.x * 2.0, uv.y * 2.0, 3165)) * (1.0 - 0.18 * vein);
      s.gloss = 0.12; mossy *= 0.3;
    }
    case ${MAT.roof}u: { s.albedo = stTiles(uv, PL_ROOF[pal], seed, aa); s.gloss = 0.1; mossy *= 0.35; }
    case ${MAT.roof2}u: { s.albedo = stTiles(uv, PL_ROOF2[pal], seed, aa); s.gloss = 0.1; mossy *= 0.35; }
    case ${MAT.dome}u: { s.albedo = stDome(uv, PL_DOME[pal], PL_METAL[pal] * 0.8, aa); s.gloss = 0.25; mossy = 0.0; }
    case ${MAT.timber}u: {
      let plank = abs(fract(uv.x / 0.24) - 0.5);
      let grain = 0.5 + 0.5 * vnoise(uv.x * 9.0, uv.y * 0.6, 3171);
      s.albedo = PL_TIMBER[pal] * (0.8 + 0.35 * grain) * (1.0 - 0.45 * (1.0 - smoothstep(0.42, 0.48, 0.5 - plank + 0.42)) * (1.0 - smoothstep(0.1, 0.3, aa)));
      mossy *= 0.5;
    }
    case ${MAT.gold}u: { s.albedo = PL_METAL[pal]; s.gloss = 0.9; s.metal = 1.0; mossy = 0.0; }
    case ${MAT.iron}u: { s.albedo = PL_IRON[pal]; s.gloss = 0.35; s.metal = 0.6; mossy = 0.0; }
    case ${MAT.bronze}u: {
      let pat = smoothstep(0.4, 0.75, 0.5 + 0.5 * vnoise(uv.x * 1.3, uv.y * 1.3, 3173));
      s.albedo = mix(PL_BRONZE[pal], vec3f(0.12, 0.32, 0.26), 0.6 * pat); s.gloss = 0.5; s.metal = 0.7 * (1.0 - pat); mossy = 0.0;
    }
    case ${MAT.glass}u: {
      let g = abs(fract(uv / 0.8) - 0.5);
      let rib = 1.0 - smoothstep(0.42, 0.47, max(g.x, g.y));
      s.albedo = mix(PL_GLASS[pal] * 0.25, PL_IRON[pal], 1.0 - rib); s.gloss = mix(0.95, 0.3, 1.0 - rib); mossy = 0.0;
      s.emit = PL_WINDOW[pal] * 0.6 * night * rib;
    }
    case ${MAT.crystal}u: {
      let fr = pow(1.0 - abs(dot(n, -rd)), 2.0);
      s.albedo = PL_CRYSTAL[pal] * (0.25 + 0.5 * fr); s.gloss = 0.7; mossy = 0.0;
      s.emit = PL_CRYSTAL[pal] * (0.2 + 0.3 * night) * (0.5 + 0.5 * fr) * (0.85 + 0.15 * sin(u.time * 1.3 + uv.y * 0.2));
    }
    case ${MAT.cloth}u, ${MAT.cloth2}u: {
      if (dot(n, rd) > 0.0) { n = -n; }
      var c = select(PL_CLOTH2[pal], PL_CLOTH[pal], mat == ${MAT.cloth}u);
      if (i.m.y == ${FACADE.stripes}u) {
        // an awning's stripes
        c = select(PL_CLOTH[pal], PL_CLOTH2[pal], fract(uv.x / 0.9) > 0.5);
      } else if (mat == ${MAT.cloth}u) {
        let border = 1.0 - smoothstep(0.06, 0.09, 0.8 - abs(uv.x));
        c = mix(c, PL_CLOTH2[pal], border);
        c = mix(c, PL_EMBLEM[pal], stEmblem(uv));
      }
      let fold = 0.85 + 0.15 * sin(uv.x * 9.0 + uv.y * 0.6);
      s.albedo = c * fold; mossy = 0.0;
    }
    case ${MAT.mosaic}u: { s.albedo = stMosaic(i.lp.xz, pal, aa); s.gloss = 0.15; mossy *= 0.25; }
    case ${MAT.plaster}u: { s.albedo = PL_PLASTER[pal] * (0.94 + 0.08 * vnoise(uv.x * 1.5, uv.y * 1.5, 3181)); mossy *= 0.3; }
    case ${MAT.woven}u: {
      let w = sin(uv.x * 30.0) * sin(uv.y * 30.0);
      s.albedo = mix(vec3f(0.22, 0.15, 0.07), vec3f(0.36, 0.27, 0.13), 0.5 + 0.5 * w); mossy = 0.0;
    }
    case ${MAT.lacquer}u: { s.albedo = PL_LACQUER[pal] * (0.92 + 0.1 * vnoise(uv.x, uv.y, 3183)); s.gloss = 0.55; mossy = 0.0; }
    case ${MAT.lantern}u: {
      s.albedo = vec3f(0.55, 0.4, 0.2); s.gloss = 0.6; mossy = 0.0;
      let flick = 0.9 + 0.1 * sin(u.time * (5.0 + f32(seed & 7u)) + f32(seed));
      s.emit = PL_LAMP[pal] * (0.05 + 1.5 * lamps) * flick;
    }
    case ${MAT.keystone}u: {
      let vein = smoothstep(0.7, 0.95, 1.0 - abs(vnoise(uv.x * 1.6, uv.y * 1.6, 3187)));
      s.albedo = mix(vec3f(0.18, 0.3, 0.42), vec3f(0.45, 0.65, 0.85), vein); s.gloss = 0.4; mossy = 0.0;
      s.emit = PL_CRYSTAL[pal] * (0.02 + 0.5 * night) * (0.4 + 0.6 * vein);
    }
    case ${MAT.lava}u: {
      let c = 0.5 + 0.5 * vnoise(uv.x * 0.5 - u.time * 0.05, uv.y * 0.5, 3189);
      s.albedo = vec3f(0.05, 0.02, 0.01); mossy = 0.0;
      s.emit = mix(vec3f(1.0, 0.25, 0.03), vec3f(1.0, 0.7, 0.2), c) * 2.5;
    }
    case ${MAT.moss}u: { s.albedo = vec3f(0.05, 0.1, 0.03) * (0.8 + 0.4 * vnoise(uv.x * 2.0, uv.y * 2.0, 3191)); mossy = 0.0; }
    case ${MAT.paving}u: { s.albedo = stPaving(uv, PL_PAVING[pal], aa); mossy *= 0.5; }
    case ${MAT.rune}u: {
      // carved stone with runes cut in it that glow skystone-blue, brighter by night, pulsing as the wind sings
      let g = uv / vec2f(1.1, 1.6);
      let cell = floor(g);
      let f = fract(g) - 0.5;
      let h = stH(cell.x, cell.y, 3205u);
      let stroke = select(abs(f.x + (h - 0.5) * 0.3), abs(f.y + (h - 0.5) * 0.3), h > 0.5);
      let ring = abs(length(f) - 0.28);
      let glyph = (1.0 - smoothstep(0.035, 0.06 + aa, min(stroke, ring + 0.02 * h))) * step(abs(f.x), 0.4) * step(abs(f.y), 0.4) * step(0.25, h);
      s.albedo = stAshlar(uv, PL_STONE[pal], pal, seed, n, aa) * (1.0 - 0.6 * glyph);
      let pulse = 0.75 + 0.25 * sin(u.time * 0.9 + cell.x * 0.7 + cell.y * 0.3);
      s.emit = PL_CRYSTAL[pal] * glyph * (0.12 + 0.9 * night) * pulse;
    }
    case ${MAT.dark}u: { s.albedo = vec3f(0.012, 0.011, 0.01); mossy = 0.0; s.emit = PL_WINDOW[pal] * 0.25 * lamps; }
    case ${MAT.leaf}u: {
      // living green (water.ts): a pad's waxy disc with its veins, a reed paling up its blade,
      // waterweed dark with lighter tips, a cattail's brown head
      if (dot(n, rd) > 0.0) { n = -n; }
      let tint = 0.88 + 0.24 * fract(f32(seed) * 0.618);
      let kind = i.m.y;
      if (kind == 0u) {
        let r = length(i.lp.xz);
        let a = atan2(i.lp.z, i.lp.x + 1e-5);
        let vein = (1.0 - smoothstep(0.0, 0.12 + aa * 4.0, abs(fract(a * 1.4324) - 0.5) * 2.0)) * smoothstep(0.05, 0.15, r);
        var c = mix(vec3f(0.1, 0.27, 0.06), vec3f(0.2, 0.4, 0.1), 0.5 + 0.5 * vnoise(i.lp.x * 9.0, i.lp.z * 9.0, 3311));
        c = mix(c, vec3f(0.3, 0.16, 0.08), smoothstep(0.44, 0.5, r) * 0.6);
        s.albedo = c * tint * (1.0 - 0.18 * vein); s.gloss = 0.45;
      } else if (kind == 1u) {
        s.albedo = mix(vec3f(0.16, 0.24, 0.07), vec3f(0.5, 0.55, 0.22), smoothstep(0.2, 1.9, i.lp.y)) * tint;
      } else if (kind == 2u) {
        s.albedo = mix(vec3f(0.04, 0.1, 0.03), vec3f(0.12, 0.24, 0.06), smoothstep(0.3, 2.0, i.lp.y)) * tint; s.gloss = 0.2;
      } else {
        s.albedo = vec3f(0.24, 0.13, 0.05) * (0.85 + 0.3 * vnoise(i.lp.y * 40.0, i.lp.x * 40.0, 3313));
      }
      mossy = 0.0;
    }
    case ${MAT.petal}u: {
      if (dot(n, rd) > 0.0) { n = -n; }
      if (i.m.y == 1u) { s.albedo = vec3f(0.8, 0.6, 0.12); } else {
        s.albedo = mix(vec3f(0.93, 0.9, 0.88), vec3f(0.92, 0.38, 0.6), smoothstep(0.3, 1.0, uv.y));
      }
      s.gloss = 0.2; mossy = 0.0;
    }
    case ${MAT.clay}u: {
      s.albedo = vec3f(0.52, 0.25, 0.13) * (0.85 + 0.25 * vnoise(uv.x * 3.0, uv.y * 3.0, 3407)); s.gloss = 0.12; mossy = 0.0;
    }
    case ${MAT.koi}u: {
      // its own colours from its seed: white or orange, patched red, some with black
      let q = i.lp;
      let sd = f32(seed & 255u);
      var c = select(vec3f(0.92, 0.9, 0.86), vec3f(0.97, 0.46, 0.08), (seed & 1u) == 1u);
      let p1 = smoothstep(0.05, 0.25, vnoise(q.z * 11.0 + sd, q.x * 9.0 + q.y * 11.0, 3317));
      let p2 = smoothstep(0.3, 0.5, vnoise(q.z * 8.0 - sd, q.y * 12.0 + q.x * 6.0, 3319));
      c = mix(c, vec3f(0.85, 0.13, 0.04), p1 * f32((seed & 6u) != 0u));
      c = mix(c, vec3f(0.03, 0.03, 0.035), p2 * f32((seed & 8u) != 0u));
      s.albedo = c; s.gloss = 0.55; mossy = 0.0;
    }
    case ${MAT.water}u: {
      if (i.m.y == 0u) {
        // still (basins, fountains): ripples in place
        let g = p.xz + u.origin;
        let ripple = vnoise(g.x * 1.3 + u.time * 0.4, g.y * 1.3, 3193) + vnoise(g.x * 2.1, g.y * 2.1 - u.time * 0.5, 3195);
        n = normalize(vec3f(0.05 * ripple, 1.0, 0.05 * ripple));
        s.albedo = vec3f(0.01, 0.07, 0.08); s.gloss = 1.0;
      } else {
        // flowing (an aqueduct along u; a spout's fall along v): ripples carried downstream, white
        // where a fall breaks
        let fall = i.m.y == 2u;
        let f = select(uv, uv.yx, fall);
        let speed = select(1.6, 4.0, fall);
        let r1 = vnoise(f.x * 1.2 - u.time * speed, f.y * 3.0, 3197);
        let r2 = vnoise(f.x * 2.6 - u.time * speed * 1.3, f.y * 6.0, 3199);
        n = normalize(n + vec3f(0.12 * r1, 0.0, 0.12 * r2));
        s.albedo = vec3f(0.02, 0.09, 0.1); s.gloss = 0.9;
        if (fall) { let wh = 0.45 + 0.35 * r1 + 0.3 * smoothstep(0.6, 1.6, uv.y); s.albedo = mix(s.albedo, vec3f(0.72, 0.78, 0.8), clamp(wh, 0.0, 1.0)); s.gloss = 0.35; }
      }
      mossy = 0.0;
    }
    default: {}
  }
  // Windows, painted on the walls that have them.
  // (the materials that use the facade byte for their own variants have none)
  if (i.m.y != 0u && mat != ${MAT.cloth}u && mat != ${MAT.cloth2}u && mat != ${MAT.water}u && mat != ${MAT.leaf}u && mat != ${MAT.petal}u) {
    let w = stFacade(i.m.y & 0x7fu, uv, i.fw, (i.m.y & ${FACADE.FLAT}u) == 0u, flags, seed, aa);
    s.albedo = mix(s.albedo, PL_TRIM[pal] * 0.98, w.band) * (1.0 - w.dark * 0.5);
    if (w.frame > 0.0 || w.glass > 0.0) {
      s.albedo = mix(s.albedo, PL_TRIM[pal] * 1.05, w.frame);
      let glass = vec3f(0.018, 0.022, 0.03) * mix(vec3f(1.0), w.tint, 0.6);
      s.albedo = mix(s.albedo, glass, w.glass);
      s.gloss = mix(s.gloss, 0.55, w.glass);
      mossy *= 1.0 - w.glass;
      let flick = 0.88 + 0.12 * sin(u.time * 2.1 + f32(seed) + uv.x);
      s.emit += PL_WINDOW[pal] * mix(vec3f(1.0), w.tint * 1.4, 0.5) * w.glass * w.lit * lamps * 0.6 * flick;
    }
  }
  // Moss on what faces up, and creeping up from the foot.
  if (mossy > 0.0) {
    let mn = 0.5 + 0.5 * vnoise(uv.x * 0.45 + i.lp.x * 0.1, uv.y * 0.45 + i.lp.z * 0.1, 3197);
    let up = smoothstep(0.35, 0.85, n.y);
    let foot = 1.0 - smoothstep(0.0, 2.5, i.lp.y);
    let m = clamp(mossy * (up * 0.9 + foot * 0.7) * smoothstep(0.35, 0.75, mn), 0.0, 0.9);
    s.albedo = mix(s.albedo, vec3f(0.06, 0.1, 0.035) * (0.75 + 0.5 * mn), m);
    s.gloss *= 1.0 - m;
  }
  if (inside) {
    n = -n;
    s = stInterior(mat, i.m.y, uv, i.lp, n, i.fw, pal, seed, flags, lamps, aa, rd);
  }
  // Light: the key light (shadowed by the realm, the trees and buildings, the hero), sky fill, bounce;
  // inside, a third of the sky's light and the warmth of the room's lamps and hearth.
  let L = keyDir();
  var sun = skyShadowAt(p) * treeShadowAt(p);
  if (sun > 0.0) { sun = min(sun, heroShadowAt(i.rel)); }
  let ndl = dot(n, L);
  let ramp = smoothstep(-0.1, 0.5, ndl);
  let ao = i.ao;
  var amb = skyAmb(n, (0.3 + 0.28 * max(n.y, 0.0)) * ao) + bounceCol(vec3f(0.07, 0.075, 0.05)) * (0.35 + 0.65 * max(-n.y, 0.0)) * ao;
  // (no sun indoors: what comes in by the door and windows is the room's share of the sky)
  // (by night the eye has opened to the dark: the lamps' warmth is a little, as the windows' light)
  if (inside || (flags & ST_INDOOR) != 0u) {
    amb = amb * (0.62 - 0.2 * lamps) + vec3f(0.9, 0.55, 0.25) * (0.09 - 0.078 * lamps) * ao;
    sun = 0.0; s.gloss = 0.0; s.metal = 0.0;
  }
  var col = s.albedo * (keyCol(p.y) * 0.62 * ramp * sun + amb);
  // the lanterns' warm pools by night (outdoors: an inside has its own lamps)
  if (!inside && (flags & ST_INDOOR) == 0u) { col += s.albedo * lanternLight(p); }
  if (s.gloss > 0.0) {
    let refl = reflect(rd, n);
    let F = mix(0.04, 1.0, pow(1.0 - max(dot(-rd, n), 0.0), 5.0));
    let env = skyReflect(refl) * mix(vec3f(1.0), s.albedo * 2.2, s.metal) * ao;
    col = mix(col, env, clamp(F * s.gloss + 0.55 * s.metal * s.gloss, 0.0, 0.9));
    let sd = max(dot(refl, L), 0.0);
    col += keyCol(p.y) * (pow(sd, 120.0) * 2.0 + pow(sd, 18.0) * 0.15) * sun * s.gloss * mix(vec3f(1.0), s.albedo * 2.0, s.metal);
  }
  col += glow(s.emit);
  return display(surfaceFogOd(col, dist, p.y, rd, fogOd(dist, rd) + rlReachOd(i.rel)));
}

// ---- shadow (into the trees' near map) ---------------------------------------------------------------------
${shadow ? `
@vertex
fn vsStructureShadow(@location(0) pos: vec4f, @location(1) nrm: vec4f, @location(2) attr: vec4u, @location(3) uv: vec2f,
  @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let P = stPlace(pos, nrm.xyz, attr, uv, stInst[stDraw[ii]]);
  let q = P.world - u.camPos - forest.shN.xyz;
  let R = forest.shC.w;
  return vec4f(dot(q, forest.shR.xyz) / R, dot(q, forest.shU.xyz) / R, 0.5 + 0.5 * dot(q, forest.shF.xyz) / forest.shR.w, 1.0);
}` : ''}
`;
}
