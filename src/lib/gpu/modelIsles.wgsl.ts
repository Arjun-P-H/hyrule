/**
 * The land's model isles (gpu/modelIsles.ts; skymodels.ts): a baked model's parts, its instanced props, its ivy, its
 * pools' water, its waterfalls and its plants (grown by the land's forest).
 * ---------------------------------------------------------------------------
 * A vertex (scripts/skymodel_pack.py) is its position over its mesh's box with a u16 holding its material and the sky
 * it sees, its normal and its ambient occlusion, its uv, its colour (sRGB) and its glow. `vsModelIsle` draws a part's
 * level of detail for an instance (turned about the vertical, set at its place); `vsModelProp` a prototype (a
 * pinnacle, a baluster) for a range of props, each scaled, turned and set in the model's axes, the model's instance
 * given by the draw's slot.
 *
 * `fsModelIsle` paints each material procedurally in the model's own axes (so the patterns stay on the stones as
 * the isle turns), on the plane its face looks along, its fine lines filtered by their screen size and the detail
 * faded out with distance: veined marble laid in great blocks, ashlar stone, hammered gold, the domes' lapis
 * scales, banner cloth in folds, leaded stained glass, polished floors, the pools' lapis mosaic; and the rock hanging
 * under the islands as the sky realm paints its own islands' (skyRealm.wgsl.ts rlRock). Then it lights them: the sun by the sky the surface sees
 * (none indoors), the sky's ambient by its occlusion, indoors a warm candlelight (a glow: steady by night); gold
 * reflects the sky and glints; glass, candles and the halos glow.
 *
 * `fsModelLeaf` draws the ivy's cards with the land's leaf sprays (forest.wgsl.ts leafTex; cut out) lit as the
 * land's leaves (treeLight); `fsModelWater` the pools as the land's water (fsWater: its ripples, the sky in it, the
 * bed through it). The falls are the sky realm's (waterfalls.wgsl.ts, through fallAt); the plants are the forest's
 * sky layer (forest.wgsl.ts skyPlantAt, through modelPlantAt), from the isles near the camera (miPlants).
 *
 * Needs, from the raster shader: u (Frame), project(), keyDir, keyCol, skyAmb, bounceCol, skyReflect, glow, lampLevel,
 * heroShadowAt, surfaceFog, display, underwater, vnoise, waterNormal; with the forest:
 * leafTex, leafSamp, treeLight, F_* and SCALE_*.
 */

import { LEAF_STYLES } from '../forest/leafAtlas';

/** Instances of parts drawn at most per frame; slots (a model instance and a prototype's box) for the props' draws. */
export const MODEL_ISLES_MAX = 128;
export const MODEL_PROP_SLOTS = 32;
/** A prop slot's stride in its uniform buffer (the dynamic offset's alignment). */
export const MODEL_PROP_SLOT_BYTES = 256;
/** Waterfalls drawn at most; isles whose plants the forest grows at most, and their plants. */
export const MODEL_FALLS_MAX = 64;
export const MODEL_PLANT_ISLES = 4;
export const MODEL_PLANTS_MAX = 2048;
/** The plants' buffer: a header, two vec4 an isle, then two vec4 a plant. */
export const MODEL_PLANTS_BYTES = 16 * (1 + 2 * MODEL_PLANT_ISLES + 2 * MODEL_PLANTS_MAX);

export function modelIslesWGSL(opts: { forest: boolean; falls: boolean }): string {
  return /* wgsl */ `
// materials (scripts/sky-castle/kit.py M_*: keep in step)
const MI_MARBLE: u32 = 1u;
const MI_STONE: u32 = 2u;
const MI_GOLD: u32 = 3u;
const MI_ROOF: u32 = 4u;
const MI_CLOTH: u32 = 5u;
const MI_GLASS: u32 = 6u;
const MI_ROCK: u32 = 7u;
const MI_FLOOR: u32 = 8u;
const MI_POOL: u32 = 9u;
// the detail (patterns, joints, grain) fades out between these distances (m)
const MI_DETAIL: vec2f = vec2f(250.0, 1400.0);
// the hanging rock's strata, as the sky realm's 'garden' islands' (skyworld/biomes.ts strata, rimCover: linear)
const MI_STRATA0: vec3f = vec3f(0.3, 0.26, 0.2);
const MI_STRATA1: vec3f = vec3f(0.42, 0.37, 0.3);
const MI_RIM_COVER: vec3f = vec3f(0.07, 0.1, 0.04);
// (the realm's undersides' sizes, made the castles': scripts/sky-castle/castle.py SCALE)
const MI_ROCK_SCALE: f32 = 2.5;

struct MiInst {
  p: vec4f,     // the model's origin (chunk-local), its turn (rad)
  lo: vec4f,    // the mesh's box: its low corner (m, the model's axes)
  ext: vec4f,   // and its size
};
@group(1) @binding(60) var<uniform> miInst: array<MiInst, ${MODEL_ISLES_MAX}>;
@group(1) @binding(61) var miTex: texture_2d<f32>;
@group(1) @binding(62) var miSamp: sampler;
// a props draw: the model's instance and the prototype's box (MiInst's layout), and the props (two vec4 each:
// x, y, z, turn; sx, sy, sz, sky seen | occlusion << 8)
@group(1) @binding(63) var<uniform> miSlot: MiInst;
@group(1) @binding(64) var<storage, read> miProps: array<vec4f>;

/** The model's axes turned by yaw about the vertical (skymodels.ts modelTop turns back). */
fn miTurn(v: vec3f, yaw: f32) -> vec3f {
  let c = cos(yaw);
  let s = sin(yaw);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

struct MiOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(3) col: vec4f,
  @location(4) shade: vec2f,   // the sky seen, the occlusion
  @location(5) loc: vec3f,     // where, in the model's axes (m)
  @location(6) nl: vec3f,      // the normal there, in the model's axes
  @location(7) @interpolate(flat) mat: u32,
  @location(8) @interpolate(flat) yaw: f32,   // the model's turn
};

/** A vertex's packed u16 (position's w): its material (high byte) and the sky it sees (low). */
fn miUnpack(w: f32) -> vec2u {
  let k = u32(round(w * 65535.0));
  return vec2u(k >> 8u, k & 255u);
}

@vertex
fn vsModelIsle(@builtin(instance_index) ii: u32, @location(0) q: vec4f, @location(1) nrm: vec4f, @location(2) uv: vec2f, @location(3) col: vec4f) -> MiOut {
  let I = miInst[ii];
  var o: MiOut;
  o.loc = I.lo.xyz + q.xyz * I.ext.xyz;
  o.rel = miTurn(o.loc, I.p.w) + (I.p.xyz - u.camPos);
  o.pos = project(o.rel);
  o.nl = nrm.xyz;
  o.n = miTurn(nrm.xyz, I.p.w);
  o.uv = uv;
  o.col = col;
  let k = miUnpack(q.w);
  o.mat = k.x;
  o.shade = vec2f(f32(k.y) / 255.0, nrm.w);
  o.yaw = I.p.w;
  return o;
}

@vertex
fn vsModelProp(@builtin(instance_index) ii: u32, @location(0) q: vec4f, @location(1) nrm: vec4f, @location(2) uv: vec2f, @location(3) col: vec4f) -> MiOut {
  let a = miProps[2u * ii];
  let b = miProps[2u * ii + 1u];
  let S = miSlot;
  var o: MiOut;
  o.loc = miTurn((S.lo.xyz + q.xyz * S.ext.xyz) * b.xyz, a.w) + a.xyz;
  o.rel = miTurn(o.loc, S.p.w) + (S.p.xyz - u.camPos);
  o.pos = project(o.rel);
  o.nl = miTurn(nrm.xyz / b.xyz, a.w);
  o.n = miTurn(o.nl, S.p.w);
  o.uv = uv;
  o.col = col;
  let k = miUnpack(q.w);
  o.mat = k.x;
  let sh = bitcast<u32>(b.w);
  o.shade = vec2f(f32(k.y) / 255.0 * f32(sh & 255u) / 255.0, nrm.w * f32((sh >> 8u) & 255u) / 255.0);
  o.yaw = S.p.w;
  return o;
}

// ---- the materials' patterns ------------------------------------------------------------------

/** A thin line (half width w) at the integer crossings of t, filtered by its screen footprint fw: 1 on the line. */
fn miLine(t: f32, w: f32, fw: f32) -> f32 {
  let d = abs(fract(t + 0.5) - 0.5);
  return (1.0 - smoothstep(w - fw, w + fw, d)) * (1.0 - smoothstep(0.15, 0.4, fw));
}

fn miHash(c: vec2f) -> f32 {
  return fract(sin(dot(c, vec2f(127.1, 311.7))) * 43758.5453);
}

/** Two octaves of value noise. */
fn miFbm(p: vec2f, seed: i32) -> f32 {
  return 0.66 * vnoise(p.x, p.y, seed) + 0.34 * vnoise(p.x * 2.13 + 7.1, p.y * 2.13 - 3.4, seed + 1);
}

/**
 * The hanging rock, as the sky realm paints its undersides (skyRealm.wgsl.ts rlRock): strata, darker toward the
 * tip, fine flutes and dark streaks running down below the rim, the rim's cover; rho: how far out (0 tip, 1 rim).
 */
fn miRock(loc: vec3f, rho: f32) -> vec3f {
  let q = loc.xz / MI_ROCK_SCALE;
  let y = loc.y / MI_ROCK_SCALE;
  let warp = 1.6 * vnoise(q.x / 45.0, q.y / 45.0, 31);
  let band = y / 11.0 + warp;
  let tone = 0.5 + 0.5 * vnoise(floor(band) * 3.1, 0.5, 33);
  let tone2 = 0.5 + 0.5 * vnoise(floor(band) * 3.1 + 7.3, 0.5, 33);
  var col = mix(MI_STRATA0, MI_STRATA1, mix(tone, tone2, smoothstep(0.6, 1.0, fract(band)))) * (1.0 - 0.12 * smoothstep(0.9, 1.0, fract(band)));
  let th = atan2(q.y, q.x) * 300.0;
  let flute = vnoise(th / 3.5, y / 40.0, 35);
  let streak = smoothstep(0.2, 0.75, vnoise(th / 11.0, y / 160.0, 37));
  col *= (0.86 + 0.18 * flute) * (1.0 - 0.28 * streak * smoothstep(0.4, 0.95, rho));
  col *= mix(0.62, 1.0, smoothstep(0.0, 0.7, rho));
  // (the rim lies 22 m under the plaza, before the scale)
  return mix(col, MI_RIM_COVER, smoothstep(0.9, 0.99, rho) * (1.0 - smoothstep(4.0, 22.0, -22.0 - y)) * 0.8);
}

/** What a material paints: albedo multiplied, normal tilt (model axes), gloss (0..1), glow multiplied. */
struct MiSurf { tone: vec3f, tilt: vec3f, gloss: f32, glow: f32 };

/**
 * A material's pattern at loc (model axes) on a face lying along uvp (its plane's coordinates: a wall's run and
 * height, a floor's x and z), fwp their screen footprint; dk: the detail left.
 */
fn miSurface(mat: u32, loc: vec3f, uvp: vec2f, fwp: vec2f, wall: bool, dk: f32) -> MiSurf {
  var s = MiSurf(vec3f(1.0), vec3f(0.0), 0.0, 1.0);
  let fw = max(fwp.x, fwp.y);
  if (mat == MI_MARBLE || mat == MI_FLOOR) {
    // veins: a sine through turbulence, thin grey threads; a slow warm/cool drift across the stone
    let turb = miFbm(uvp * 0.06, 41) * 5.0 + miFbm(uvp * 0.21, 43) * 1.2;
    let v = abs(sin(dot(uvp, vec2f(0.045, 0.021)) * 6.2831 + turb));
    let vein = (1.0 - smoothstep(0.0, 0.07 + fw * 0.2, v)) * dk;
    let drift = miFbm(uvp * 0.012, 47);
    s.tone = mix(vec3f(1.0 + 0.04 * drift, 1.0, 1.0 - 0.03 * drift), vec3f(0.74, 0.72, 0.7), vein * 0.55);
    if (mat == MI_FLOOR) { s.gloss = 0.35; s.tone *= 1.0 - 0.06 * vnoise(uvp.x * 0.7, uvp.y * 0.7, 49) * dk; }
    // a marble wall laid in great blocks: their joints
    if (wall && mat == MI_MARBLE) {
      let row = floor(uvp.y / 4.2);
      let j = max(miLine(uvp.y / 4.2, 0.012, fwp.y / 4.2), miLine(uvp.x / 9.0 + 0.5 * row, 0.006, fwp.x / 9.0));
      s.tone *= 1.0 - 0.28 * j * dk;
    }
  } else if (mat == MI_STONE) {
    // ashlar: blocks in courses, each its own shade, fine grain, the joints shadowed
    let row = floor(uvp.y / 2.6);
    let col = floor(uvp.x / 5.2 + 0.5 * row);
    var j = 0.0;
    if (wall) { j = max(miLine(uvp.x / 5.2 + 0.5 * row, 0.012, fwp.x / 5.2), miLine(uvp.y / 2.6, 0.025, fwp.y / 2.6)); }
    let block = miHash(vec2f(row, col));
    let grain = vnoise(uvp.x * 1.3, uvp.y * 1.3, 51);
    s.tone = vec3f(1.0 + dk * (0.07 * (block - 0.5) + 0.05 * grain)) * (1.0 - 0.32 * j * dk);
  } else if (mat == MI_GOLD) {
    // hammered: small dents (a tilt and a shade), a brushed grain running down
    let h = vec2f(vnoise(uvp.x * 1.7, uvp.y * 1.7, 61), vnoise(uvp.x * 1.7 + 9.0, uvp.y * 1.7 - 4.0, 62));
    let brush = vnoise(uvp.x * 9.0, uvp.y * 0.6, 63);
    s.tone = vec3f(0.92 + 0.12 * h.x * dk + 0.06 * brush * dk);
    s.tilt = vec3f(h.x, 0.0, h.y) * 0.18 * dk;
    s.gloss = 1.0;
  } else if (mat == MI_ROOF) {
    // the domes' lapis scales: rows down the dome, each row offset, every scale's lower edge shadowed, gold flecks
    let r = loc.y / 2.0;
    let a = (loc.x + loc.z) / 2.4 + 0.5 * floor(r);
    let c = vec2f(fract(a) - 0.5, fract(r));
    let edge = smoothstep(0.32, 0.5, length(c * vec2f(1.0, 0.7)));
    let id = miHash(vec2f(floor(a), floor(r)));
    s.tone = vec3f(1.0 - 0.35 * edge * dk + 0.12 * (id - 0.5) * dk);
    s.tone = mix(s.tone, vec3f(4.0, 3.0, 1.2), smoothstep(0.985, 1.0, id) * dk * 0.5);
    s.gloss = 0.45;
  } else if (mat == MI_CLOTH) {
    // cloth: soft folds hanging down it, a fine weave
    let fold = sin(uvp.x * 1.4 + 1.3 * vnoise(uvp.x * 0.3, uvp.y * 0.05, 71));
    let weave = sin(uvp.x * 60.0) * sin(uvp.y * 60.0);
    s.tone = vec3f(1.0 + 0.16 * fold * dk + 0.03 * weave * dk * (1.0 - smoothstep(0.0, 0.08, fw)));
    s.tilt = vec3f(cos(uvp.x * 1.4) * 0.25 * dk, 0.0, 0.0);
  } else if (mat == MI_GLASS) {
    // leaded quarries: diamond panes, each its own shade (now and then gold or ruby among the blue), the lead dark
    let d = vec2f(uvp.x + uvp.y, uvp.x - uvp.y) / 3.2;
    let lead = max(miLine(d.x, 0.04, fw / 3.2), miLine(d.y, 0.04, fw / 3.2));
    let id = miHash(floor(d));
    var pane = vec3f(0.8 + 0.4 * id);
    if (id > 0.93) { pane = vec3f(1.6, 1.1, 0.3); } else if (id > 0.88) { pane = vec3f(1.56, 0.3, 0.36); }
    s.tone = mix(pane, vec3f(0.12), lead * dk);
    s.glow = mix(1.0 + 0.4 * (id - 0.5), 0.05, lead * dk);
    s.gloss = 0.5;
  } else if (mat == MI_POOL) {
    // a mosaic: small square tesserae, each its own lapis, the grout pale; now and then a gold one
    let t = uvp / 0.75;
    let grout = max(miLine(t.x, 0.06, fwp.x / 0.75), miLine(t.y, 0.06, fwp.y / 0.75));
    let id = miHash(floor(t));
    s.tone = mix(vec3f(0.75 + 0.5 * id, 0.85 + 0.3 * id, 1.0), vec3f(3.2, 3.0, 2.6), grout * dk * 0.6);
    if (id > 0.95) { s.tone = vec3f(4.0, 3.0, 1.0); }
  }
  return s;
}

@fragment
fn fsModelIsle(i: MiOut) -> @location(0) vec4f {
  let tex = textureSample(miTex, miSamp, i.uv).rgb;
  // (derivatives where the control flow is uniform: the patterns' footprints)
  let fl = fwidth(i.loc);
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let p = u.camPos + i.rel;
  var n = normalize(i.n);
  var nl = normalize(i.nl);
  if (dot(n, rd) > 0.0) { n = -n; nl = -nl; }
  // the plane the face lies along: a wall's run and height, a floor's x and z
  let an = abs(nl);
  let wall = an.y < 0.6;
  var uvp = select(vec2f(i.loc.x, i.loc.y), vec2f(i.loc.z, i.loc.y), an.x > an.z);
  var fwp = select(vec2f(fl.x, fl.y), vec2f(fl.z, fl.y), an.x > an.z);
  if (!wall) { uvp = i.loc.xz; fwp = fl.xz; }
  let dk = 1.0 - smoothstep(MI_DETAIL.x, MI_DETAIL.y, dist);
  var surf = MiSurf(vec3f(1.0), vec3f(0.0), 0.0, 1.0);
  if (i.mat != MI_ROCK && (dk > 0.0 || i.mat == MI_GLASS || i.mat == MI_GOLD)) {
    surf = miSurface(i.mat, i.loc, uvp, fwp, wall, dk);
  }
  // (the tilt is in the model's axes: turned with it)
  n = normalize(n + miTurn(surf.tilt, i.yaw));
  // (whites are painted near 1: toned toward the land's brightest, snow's ~0.5, else they clip to flat white)
  var albedo = i.col.rgb * i.col.rgb * tex * surf.tone * 0.7;
  // the hanging rock: the realm's undersides' (its colour's red channel: how far out the face lies)
  if (i.mat == MI_ROCK) { albedo = miRock(i.loc, i.col.r); }
  let vis = i.shade.x;
  // (the occlusion is baked over 30 m: in a crowded castle that is most of everything, so it darkens by half at most)
  let ao = mix(0.45, 1.0, i.shade.y);
  let L = keyDir();
  let ndl = dot(n, L);
  // (no sun where no sky is seen: indoors)
  let sun = smoothstep(0.02, 0.2, vis) * select(1.0, heroShadowAt(i.rel), ndl > 0.0);
  var amb = skyAmb(n, (0.3 + 0.28 * max(n.y, 0.0)) * ao) + bounceCol(vec3f(0.07, 0.075, 0.05)) * (0.35 + 0.65 * max(-n.y, 0.0)) * ao;
  // indoors the sky is unseen: what light there is comes by the windows, from the glass and the candles
  let lamps = lampLevel();
  let indoor = 1.0 - smoothstep(0.05, 0.4, vis);
  amb = amb * mix(1.0, 0.62 - 0.2 * lamps, indoor);
  var col = albedo * (keyCol(p.y) * 0.62 * smoothstep(-0.1, 0.5, ndl) * sun + amb);
  // gold: the sky in it, the sun's glint; glossy stone, glass and lapis: some of the sky at a grazing look
  let refl = reflect(rd, n);
  let F = mix(0.04, 1.0, pow(1.0 - max(dot(-rd, n), 0.0), 5.0));
  let sd = max(dot(refl, L), 0.0);
  if (i.mat == MI_GOLD) {
    let env = skyReflect(refl) * albedo * 2.2 * mix(0.25, 1.0, vis) * ao;
    col = mix(col, env, 0.45);
    col += keyCol(p.y) * (pow(sd, 60.0) * 1.5 + pow(sd, 10.0) * 0.12) * sun * albedo * 2.0;
  } else {
    col = mix(col, skyReflect(refl) * ao, F * (0.18 + 0.4 * surf.gloss) * vis);
    col += keyCol(p.y) * pow(sd, 90.0) * surf.gloss * sun * 0.6;
  }
  // the halls' candlelight: a glow (steady on screen whatever the eye has adapted to — by night it would flood white
  // as a light), dimmer and warmer by night
  col += glow(albedo * vec3f(0.9, 0.62, 0.34) * (0.16 - 0.08 * lamps) * ao * indoor);
  col += glow(i.col.rgb * i.col.rgb * surf.tone * i.col.a * surf.glow * 1.6);
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}

// ---- the pools' water: the land's ------------------------------------------------------------------

@fragment
fn fsModelWater(i: MiOut) -> @location(0) vec4f {
  let tS = length(i.rel);
  let rd = i.rel / tS;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let p = u.camPos + i.rel;
  let g = p.xz + u.origin;
  let depthV = i.uv.x * 16.0;
  // ripples (calm: a basin), faded with distance as the land's
  let strength = 0.07 * (1.0 - smoothstep(40.0, 300.0, tS));
  var n = vec3f(0.0, 1.0, 0.0);
  if (strength > 0.0) { n = waterNormal(g, u.time, strength); }
  let sh = smoothstep(0.02, 0.2, i.shade.x) * heroShadowAt(i.rel);
  let path = depthV / max(-rd.y, 0.06);
  let T = exp(-vec3f(0.5, 0.14, 0.09) * path);
  let Tl = dot(T, vec3f(0.2126, 0.7152, 0.0722));
  let kc = keyCol(p.y);
  let up = vec3f(0.0, 1.0, 0.0);
  let body = vec3f(0.006, 0.065, 0.12) * (kc * 0.35 * (0.4 + 0.6 * sh) + skyAmb(up, 0.8));
  let refl = reflect(rd, n);
  let F = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
  let sd = max(dot(refl, keyDir()), 0.0);
  let glint = pow(sd, 900.0) * 14.0 + pow(sd, 120.0) * 0.3;
  // premultiplied: the sky in it and the light it scatters, over the bed seen through it
  let col = skyReflect(refl) * F + body * (vec3f(1.0) - T) * (1.0 - F) + kc * glint * sh;
  let alpha = 1.0 - (1.0 - F) * Tl;
  let a = max(alpha, 1e-3);
  let shown = display(surfaceFog(col / a, tS, p.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}
${opts.forest ? leafWGSL() : ''}
${opts.falls ? fallsWGSL() : ''}
`;
}

/** The ivy's cards (the land's leaf sprays, cut out, lit as the land's leaves) and the forest's plants on the isles. */
function leafWGSL(): string {
  return /* wgsl */ `
const MI_LEAF_LAYER: i32 = ${LEAF_STYLES.indexOf('palmate')};

@fragment
fn fsModelLeaf(i: MiOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  // (the spray: coverage, tone, the leaf's tilt across the card; premultiplied by coverage)
  let tex = textureSample(leafTex, leafSamp, i.uv, MI_LEAF_LAYER);
  if (tex.r < 0.45) { discard; }
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let tone = tex.g / tex.r;
  let tilt = tex.ba / tex.r * 2.0 - 1.0;
  var n = normalize(i.n) * select(-1.0, 1.0, front);
  // the leaf's own tilt, across the card (its side and down it), and a little toward the sky
  let side = normalize(cross(n, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-4, 0.0, 0.0));
  n = normalize(n + side * tilt.x * 0.8 + vec3f(0.0, 1.0, 0.0) * (tilt.y * 0.8 + 0.35));
  let albedo = i.col.rgb * i.col.rgb * (0.45 + 0.75 * tone);
  let sun = smoothstep(0.02, 0.2, i.shade.x) * heroShadowAt(i.rel);
  return treeLight(albedo, n, i.rel, rd, dist, sun, i.shade.y, 1.0, 0.7, 0.2);
}

// ---- the forest's sky layer: the isles' plants -----------------------------------------------------

// The isles near the camera and their plants (gpu/modelIsles.ts): a header (how many isles), per isle its middle
// (chunk-local x, z), its reach and its first plant, then its count; per plant x, z (chunk-local), y, scale; its
// species and palette (forest/species.ts SPECIES, skyworld/biomes.ts SKY_FLORA).
@group(0) @binding(66) var<storage, read> miPlants: array<vec4f>;

fn miPlantRecord(sp: u32, pal: u32, off: vec2f, y: f32, scale: f32, bits: u32) -> vec4u {
  let sb = u32(clamp(round((scale - SCALE_LO) / SCALE_SPAN * 255.0), 0.0, 255.0));
  let info = sp | (((bits & 0xffffu) % F_VARIANTS) << 4u) | (((bits >> 16u) & 255u) << 6u) | (sb << 14u)
    | (((bits >> 8u) & 7u) << 22u) | ((pal & 7u) << 25u) | ((pal & 8u) << 28u) | F_ISLAND | F_VALID;
  return vec4u(bitcast<u32>(off.x), bitcast<u32>(off.y), bitcast<u32>(y), info);
}

// The plant (if any) of lattice cell (ix, iz) up on a model isle: the isle's plant standing in the cell.
fn modelPlantAt(ix: i32, iz: i32, lid: vec2f, corner: vec2f, bits: u32) -> vec4u {
  let n = u32(miPlants[0].x);
  for (var k = 0u; k < n; k++) {
    let h = miPlants[1u + 2u * k];
    if (length(corner - h.xy) > h.z + 8.0) { continue; }
    let first = u32(h.w);
    let count = u32(miPlants[2u + 2u * k].x);
    for (var j = first; j < first + count; j++) {
      let a = miPlants[${1 + 2 * MODEL_PLANT_ISLES}u + 2u * j];
      let d = a.xy - corner;
      if (d.x < 0.0 || d.y < 0.0 || d.x >= F_CELL || d.y >= F_CELL) { continue; }
      let b = miPlants[${2 + 2 * MODEL_PLANT_ISLES}u + 2u * j];
      return miPlantRecord(u32(b.x), u32(b.y), lid * F_CELL + d, a.z, a.w, bits);
    }
  }
  return vec4u(0u);
}
`;
}

/** The isles' waterfalls, for the sky realm's waterfalls.wgsl.ts (its SkyFall, through fallAt). */
function fallsWGSL(): string {
  return /* wgsl */ `
struct SkyFall {
  a: vec4f,   // lip x, y, z (chunk-local; y: the water's surface), width at the lip
  b: vec4f,   // direction x, z (unit), speed over the lip, reach (m fallen when it has all become mist)
  c: vec4f,   // seed, unused
};
@group(1) @binding(65) var<storage, read> miFalls: array<SkyFall>;
fn fallAt(k: u32) -> SkyFall { return miFalls[k]; }
// (the realm fades its falls out at its reach; the isles' are drawn within FALL_RANGE: gpu/modelIsles.ts)
fn rlReachFade(rel: vec3f) -> f32 { return 1.0 - smoothstep(${0.82 * 25000}.0, 25000.0, length(rel.xz)); }
`;
}
