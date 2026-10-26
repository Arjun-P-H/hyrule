/**
 * The day–night sky (WGSL): the atmosphere's lookup tables, the sky dome
 * (sun, moon, stars, Milky Way, clouds) and the light every surface is lit
 * with at the hour (the "light API" the surface shaders call).
 * ---------------------------------------------------------------------------
 * Light API — both skies implement it, so the surfaces (terrain, grass,
 * trees, ruins, islands, waterfalls, water, hero) are shaded by one code path:
 *   keyDir()          toward the key light (the sun, or at night the moon)
 *   keyCol(y)         its colour × intensity arriving at altitude y (m)
 *   skyAmb(n, w)      sky fill on a surface facing n; w is the shader's own
 *                     weight (its tuned k0 + k1·n.y), so the classic balance
 *                     of sun and fill carries over
 *   bounceCol(c)      light off the ground (c: the classic constant)
 *   skyHorCol()       the horizon's colour (rim light)
 *   underwaterCol()   the colour under the water surface
 *   glow(c)           emissive light c (bright by day, not blinding at night)
 *   CURVATURE         1/(2R): the land falls away with distance (horizons dip)
 * CLASSIC_LIGHT_WGSL returns the classic constants (the classic sky is
 * unchanged); cycleSkyWGSL reads the frame's light rig (lightRig.ts) and what
 * the probe pass measured of the sky.
 *
 * The atmosphere (after Hillaire 2020): a transmittance LUT and a
 * multiple-scattering LUT, both computed once; a sky-view LUT (every
 * azimuth × elevation, dense near the horizon) integrated each frame for the
 * sun and the moon together from the camera's altitude; then one small probe
 * pass averages that sky into the fill light, the horizon's colour round the
 * compass (a cosine series about the key light, the haze's colour) and the
 * camera's exposure. The LUT and the probe are double-buffered (sky.ts). Night
 * adds what the air does not scatter: stars in a turning celestial frame, the
 * Milky Way, the moon's disc with its phase, shooting stars and airglow.
 */

import { ATMOSPHERE } from '../sky/atmosphere';
import { KEY_ALTS, KEY_TOP } from '../sky/lightRig';
import { POLE, SKY } from '../sky/celestial';
import { wgslF32 as f } from './worldgen.wgsl';

const v3 = (a: readonly number[]) => `vec3f(${a.map(f).join(', ')})`;

/** Sky uniform: 15 vec4, then the key light colour at KEY_ALTS altitudes. */
export const SKY_BYTES = 16 * (15 + KEY_ALTS);
/** What the probe pass writes (and the render passes read as a uniform): 10 vec4. */
export const SKY_LIGHT_BYTES = 16 * 10;
export const TRANS_LUT: [number, number] = [256, 64];
export const MS_LUT = 32;
export const VIEW_LUT: [number, number] = [192, 108];

/**
 * Calibration against the classic sky, measured from the probe with the sun
 * at the classic sun's elevation (25°), so the day reads like the classic
 * world: the fill (sky light → the classic SKY_TINT's luminance, 0.57) keeps
 * 45 % of the sky's blue; the bounce (the classic constants were tuned under
 * ground light of luminance ~1.75).
 */
const AMB_GAIN = 1.0;
const AMB_SAT = 0.45;
const FILL_REF = 0.57;
const BOUNCE_REF = 1 / 1.75;

const A = ATMOSPHERE;
const T_H = Math.sqrt(A.top * A.top - A.bottom * A.bottom);

/** The sky uniform (sky.ts packs it from the light rig). */
const SKY_STRUCT_WGSL = /* wgsl */ `
struct SkyU {
  sun: vec4f,       // toward the sun; w: its path angle
  moon: vec4f,      // toward the moon; w: its path angle
  key: vec4f,       // toward the key light (sun or moon); w: its path angle
  sunE: vec4f,      // sunlight above the air (rgb, scene units); w: camera altitude (km)
  moonE: vec4f,     // moonlight above the air (rgb); w: lit share of the moon's disc
  sunDisc: vec4f,   // the sun's disc as seen (rgb); w: angular radius (rad)
  moonDisc: vec4f,  // the moon's disc as seen (rgb); w: angular radius (rad)
  star0: vec4f,     // world → celestial rows; star0.w: star visibility
  star1: vec4f,     //   star1.w: time (s), for twinkling
  star2: vec4f,     //   star2.w: night (0 day … 1 night)
  grade: vec4f,     // exposure bias, saturation, night vision (blue shift), the haze's tint (blue)
  fog: vec4f,       // haze (× the classic density), mist density (1/m), mist scale height (m), camera altitude (m)
  clouds: vec4f,    // coverage threshold shift, light (dim at night), Milky Way visibility, twilight (0–1)
  keyAz: vec4f,     // the key light's azimuth: its horizontal direction x, z (unit); the haze's tint (red, green)
  camKey: vec4f,    // the key light's colour at the camera's altitude (the haze's forward glow)
  keyCols: array<vec4f, ${KEY_ALTS}>, // key light colour at altitudes (k / ${KEY_ALTS - 1})² · ${KEY_TOP} m
};
struct SkyLight {
  ambUp: vec4f,     // sky fill on up-facing ground (rgb, shading units); w: its luminance
  ambSun: vec4f,    // hue of the fill on ground facing the key light's azimuth (at ambUp's luminance)
  ambAnti: vec4f,   // … facing away from it
  ambDown: vec4f,   // … on down-facing surfaces (the lit ground below)
  bounce: vec4f,    // ground bounce: multiplier of the classic bounce constants (rgb)
  hor: vec4f,       // the horizon's colour round the compass (sky units): mean,
  hor1: vec4f,      //   and its cos φ and cos 2φ terms, φ the azimuth from the key light's
  hor2: vec4f,
  zenith: vec4f,    // the sky overhead (sky units)
  grade: vec4f,     // exposure, 1 / exposure, the light on the land it adapted to (luminance), the sky's twilight lift
};
`;

// ---- the atmosphere (km) -------------------------------------------------------------
const ATMOS_WGSL = /* wgsl */ `
const PI_A: f32 = 3.14159265;
const A_BOTTOM: f32 = ${f(A.bottom)};
const A_TOP: f32 = ${f(A.top)};
const A_RAY: vec3f = ${v3(A.rayleigh)};
const A_RAY_H: f32 = ${f(A.rayleighHeight)};
const A_MIE_S: f32 = ${f(A.mieScatter)};
const A_MIE_E: f32 = ${f(A.mieExtinct)};
const A_MIE_H: f32 = ${f(A.mieHeight)};
const A_MIE_G: f32 = ${f(A.mieG)};
const A_OZONE: vec3f = ${v3(A.ozone)};
const A_OZONE_C: f32 = ${f(A.ozoneCentre)};
const A_OZONE_W: f32 = ${f(A.ozoneHalfWidth)};
const A_ALBEDO: f32 = ${f(A.groundAlbedo)};
const T_HOR: f32 = ${f(T_H)};
const TRANS_W: f32 = ${f(TRANS_LUT[0])};
const TRANS_H: f32 = ${f(TRANS_LUT[1])};
const MS_N: f32 = ${f(MS_LUT)};
const VIEW_W: f32 = ${f(VIEW_LUT[0])};
const VIEW_H: f32 = ${f(VIEW_LUT[1])};

struct Medium { rayS: vec3f, mieS: f32, ext: vec3f };
fn medium(h: f32) -> Medium {
  let r = exp(-h / A_RAY_H);
  let m = exp(-h / A_MIE_H);
  let o = max(0.0, 1.0 - abs(h - A_OZONE_C) / A_OZONE_W);
  var md: Medium;
  md.rayS = A_RAY * r;
  md.mieS = A_MIE_S * m;
  md.ext = A_RAY * r + vec3f(A_MIE_E * m) + A_OZONE * o;
  return md;
}

// Distances from radius r along direction cosine mu to the sphere of radius R (−1: missed).
fn sphereFar(r: f32, mu: f32, R: f32) -> f32 {
  let d = r * r * (mu * mu - 1.0) + R * R;
  if (d < 0.0) { return -1.0; }
  return -r * mu + sqrt(d);
}
fn sphereNear(r: f32, mu: f32, R: f32) -> f32 {
  let d = r * r * (mu * mu - 1.0) + R * R;
  if (d < 0.0) { return -1.0; }
  return -r * mu - sqrt(d);
}
fn hitsGround(r: f32, mu: f32) -> bool {
  return mu < 0.0 && r * r * (mu * mu - 1.0) + A_BOTTOM * A_BOTTOM >= 0.0;
}

fn rayleighPhase(nu: f32) -> f32 { return 3.0 / (16.0 * PI_A) * (1.0 + nu * nu); }
fn miePhase(nu: f32) -> f32 {
  // Cornette-Shanks
  let g = A_MIE_G;
  let k = 3.0 / (8.0 * PI_A) * (1.0 - g * g) / (2.0 + g * g);
  return k * (1.0 + nu * nu) / pow(max(1.0 + g * g - 2.0 * g * nu, 1e-4), 1.5);
}

// Transmittance LUT coordinates (Bruneton 2017): x from the distance to the top, y from the altitude.
fn transUv(r: f32, mu: f32) -> vec2f {
  let rho = sqrt(max(r * r - A_BOTTOM * A_BOTTOM, 0.0));
  let d = max(sphereFar(r, mu, A_TOP), 0.0);
  let dMin = A_TOP - r;
  let dMax = rho + T_HOR;
  let x = vec2f((d - dMin) / max(dMax - dMin, 1e-6), rho / T_HOR);
  let n = vec2f(TRANS_W, TRANS_H);
  return (x * (n - 1.0) + 0.5) / n;
}
`;

/** The atmosphere's compute passes (sky.ts runs them): LUTs and the probe. */
export const SKY_LUT_WGSL = /* wgsl */ `
${SKY_STRUCT_WGSL}
${ATMOS_WGSL}
@group(0) @binding(0) var<uniform> sky: SkyU;
@group(0) @binding(1) var transLut: texture_2d<f32>;
@group(0) @binding(2) var msLut: texture_2d<f32>;
@group(0) @binding(3) var lutSamp: sampler;
@group(0) @binding(4) var lutOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var viewLut: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> skyLightOut: SkyLight;

fn transmittanceAt(r: f32, mu: f32) -> vec3f {
  if (hitsGround(r, mu)) { return vec3f(0.0); }
  return textureSampleLevel(transLut, lutSamp, transUv(r, mu), 0.0).rgb;
}
fn msAt(r: f32, mu: f32) -> vec3f {
  let uv = vec2f(clamp(0.5 + 0.5 * mu, 0.0, 1.0), clamp((r - A_BOTTOM) / (A_TOP - A_BOTTOM), 0.0, 1.0));
  return textureSampleLevel(msLut, lutSamp, (uv * (MS_N - 1.0) + 0.5) / MS_N, 0.0).rgb;
}

@compute @workgroup_size(8, 8)
fn transmittanceLut(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= TRANS_W || f32(id.y) >= TRANS_H) { return; }
  let x = vec2f(id.xy) / (vec2f(TRANS_W, TRANS_H) - 1.0);
  let rho = T_HOR * x.y;
  let r = sqrt(rho * rho + A_BOTTOM * A_BOTTOM);
  let dMin = A_TOP - r;
  let dMax = rho + T_HOR;
  let d = dMin + x.x * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = clamp((T_HOR * T_HOR - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0); }
  let L = max(sphereFar(r, mu, A_TOP), 0.0);
  var od = vec3f(0.0);
  let n = 48;
  for (var i = 0; i < n; i++) {
    let t = (f32(i) + 0.5) / f32(n) * L;
    let h = sqrt(r * r + t * t + 2.0 * r * mu * t) - A_BOTTOM;
    od += medium(h).ext;
  }
  textureStore(lutOut, id.xy, vec4f(exp(-od * (L / f32(n))), 1.0));
}

// Multiple scattering (Hillaire 2020 §5.5): per texel (sun cosine × altitude),
// 64 directions in a workgroup, second-order light with an isotropic phase,
// summed to all orders as L2 / (1 − f_ms).
var<workgroup> msSumL: array<vec3f, 64>;
var<workgroup> msSumF: array<vec3f, 64>;
@compute @workgroup_size(64)
fn multiScatterLut(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let uv = (vec2f(wg.xy) + 0.5) / MS_N;
  let muS = uv.x * 2.0 - 1.0;
  let r = A_BOTTOM + max(uv.y * (A_TOP - A_BOTTOM), 0.005);
  let sunDir = vec3f(sqrt(max(1.0 - muS * muS, 0.0)), muS, 0.0);
  let fi = f32(li);
  let z = 1.0 - (fi + 0.5) / 32.0;
  let ph = fi * 2.39996323;
  let sr = sqrt(max(1.0 - z * z, 0.0));
  let dir = vec3f(sr * cos(ph), z, sr * sin(ph));
  let ground = hitsGround(r, dir.y);
  var tMax = sphereFar(r, dir.y, A_TOP);
  if (ground) { tMax = sphereNear(r, dir.y, A_BOTTOM); }
  let n = 20;
  let dt = tMax / f32(n);
  var L = vec3f(0.0);
  var F = vec3f(0.0);
  var T = vec3f(1.0);
  let iso = 1.0 / (4.0 * PI_A);
  for (var k = 0; k < n; k++) {
    let t = (f32(k) + 0.3) * dt;
    let p = vec3f(0.0, r, 0.0) + dir * t;
    let pr = length(p);
    let md = medium(pr - A_BOTTOM);
    let scat = md.rayS + vec3f(md.mieS);
    let ts = transmittanceAt(pr, dot(p / pr, sunDir));
    let segT = exp(-md.ext * dt);
    let ext = max(md.ext, vec3f(1e-7));
    L += T * (scat * iso * ts - scat * iso * ts * segT) / ext;
    F += T * (scat - scat * segT) / ext;
    T *= segT;
  }
  if (ground) {
    let pg = vec3f(0.0, r, 0.0) + dir * tMax;
    let up = normalize(pg);
    let mu = dot(up, sunDir);
    L += T * transmittanceAt(A_BOTTOM + 1e-3, mu) * max(mu, 0.0) * A_ALBEDO / PI_A;
  }
  msSumL[li] = L;
  msSumF[li] = F;
  workgroupBarrier();
  if (li == 0u) {
    var sL = vec3f(0.0);
    var sF = vec3f(0.0);
    for (var k = 0; k < 64; k++) { sL += msSumL[k]; sF += msSumF[k]; }
    let psi = (sL / 64.0) / max(vec3f(1.0) - sF / 64.0, vec3f(1e-3));
    textureStore(lutOut, wg.xy, vec4f(psi, 1.0));
  }
}

// View directions of the sky-view LUT: azimuth round u, elevation along v, dense at the horizon.
fn viewLutDir(uv: vec2f) -> vec3f {
  let az = uv.x * 2.0 * PI_A;
  let s = uv.y - 0.5;
  let el = sign(s) * (2.0 * s) * (2.0 * s) * 0.5 * PI_A;
  return vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}

// Light scattered toward the camera along a view ray, by the sun and the moon.
@compute @workgroup_size(8, 8)
fn skyViewLut(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= VIEW_W || f32(id.y) >= VIEW_H) { return; }
  let dir = viewLutDir((vec2f(id.xy) + 0.5) / vec2f(VIEW_W, VIEW_H));
  let r0 = A_BOTTOM + sky.sunE.w;
  let ground = hitsGround(r0, dir.y);
  var tMax = sphereFar(r0, dir.y, A_TOP);
  if (ground) { tMax = sphereNear(r0, dir.y, A_BOTTOM); }
  tMax = min(tMax, 900.0);
  let n = 32;
  var L = vec3f(0.0);
  var T = vec3f(1.0);
  let nuS = dot(dir, sky.sun.xyz);
  let nuM = dot(dir, sky.moon.xyz);
  let phRS = rayleighPhase(nuS);
  let phMS = miePhase(nuS);
  let phRM = rayleighPhase(nuM);
  let phMM = miePhase(nuM);
  var tPrev = 0.0;
  for (var k = 0; k < n; k++) {
    let a = (f32(k) + 1.0) / f32(n);
    let t1 = tMax * a * a;
    let dt = t1 - tPrev;
    let t = tPrev + 0.5 * dt;
    tPrev = t1;
    let p = vec3f(0.0, r0, 0.0) + dir * t;
    let pr = length(p);
    let up = p / pr;
    let md = medium(pr - A_BOTTOM);
    let scat = md.rayS + vec3f(md.mieS);
    let muS = dot(up, sky.sun.xyz);
    let muM = dot(up, sky.moon.xyz);
    let S = sky.sunE.rgb * ((md.rayS * phRS + md.mieS * phMS) * transmittanceAt(pr, muS) + scat * msAt(pr, muS))
          + sky.moonE.rgb * ((md.rayS * phRM + md.mieS * phMM) * transmittanceAt(pr, muM) + scat * msAt(pr, muM));
    let segT = exp(-md.ext * dt);
    L += T * (S - S * segT) / max(md.ext, vec3f(1e-7));
    T *= segT;
  }
  textureStore(lutOut, id.xy, vec4f(L, 1.0));
}

fn viewLutAt(d: vec3f) -> vec3f {
  let el = asin(clamp(d.y, -1.0, 1.0));
  let v = 0.5 + 0.5 * sign(el) * sqrt(abs(el) / (0.5 * PI_A));
  let u = fract(atan2(d.z, d.x) / (2.0 * PI_A) + 1.0);
  return textureSampleLevel(viewLut, lutSamp, vec2f(u, clamp(v, 0.5 / VIEW_H, 1.0 - 0.5 / VIEW_H)), 0.0).rgb;
}

// The probe: the sky's light on up-facing and on vertical surfaces toward and
// away from the key light (equal-area samples of the upper hemisphere; the
// lower one is the ground, lit by the key light and the sky), and the colour
// of the horizon's haze in 16 directions.
var<workgroup> pUp: array<vec3f, 64>;
var<workgroup> pSun: array<vec3f, 64>;
var<workgroup> pAnti: array<vec3f, 64>;
var<workgroup> pRing: array<vec3f, 16>;
const LUMA_A: vec3f = vec3f(0.2126, 0.7152, 0.0722);
// Starlight and airglow on the land (shading units): a moonless night, ~1/6 of a full moon's light.
const NIGHT_FILL: vec3f = vec3f(0.00042, 0.00062, 0.0012);
// Exposure: adapted to the light on the land round the camera, partly (displayed
// brightness ∝ light^(1 − ADAPT)), never below the day's.
const E_REF: f32 = 1.94;   // the land's light under the classic sun (25°)
const ADAPT: f32 = 0.78;
const EXPOSURE_DAY: f32 = 1.05;
// Twilight: the horizon shown at least this bright (display luminance), lifted by up to ×2.6.
const TWILIGHT_SKY: f32 = 0.42;
const AMB_GAIN: f32 = ${f(AMB_GAIN)};
const AMB_SAT: f32 = ${f(AMB_SAT)};
const BOUNCE_REF: f32 = ${f(BOUNCE_REF)};
const GROUND_ALBEDO: vec3f = vec3f(0.19, 0.24, 0.15);
fn desat(c: vec3f, k: f32) -> vec3f { return mix(vec3f(dot(c, LUMA_A)), c, k); }
fn atLuma(c: vec3f, l: f32) -> vec3f { return c * (l / max(dot(c, LUMA_A), 1e-7)); }
@compute @workgroup_size(64)
fn skyProbe(@builtin(local_invocation_index) li: u32) {
  let kd = normalize(vec3f(sky.key.x, 0.0, sky.key.z) + vec3f(1e-5, 0.0, 0.0));
  var up = vec3f(0.0);
  var sn = vec3f(0.0);
  var an = vec3f(0.0);
  for (var j = 0u; j < 8u; j++) {
    let idx = li * 8u + j;
    let se = (f32(idx / 32u) + 0.5) / 16.0;
    let ce = sqrt(1.0 - se * se);
    let az = (f32(idx % 32u) + 0.5) / 32.0 * 2.0 * PI_A;
    let d = vec3f(ce * cos(az), se, ce * sin(az));
    let L = viewLutAt(d);
    up += L * se;
    sn += L * max(dot(d, kd), 0.0);
    an += L * max(-dot(d, kd), 0.0);
  }
  pUp[li] = up;
  pSun[li] = sn;
  pAnti[li] = an;
  if (li < 16u) {
    let az = (f32(li) + 0.5) / 16.0 * 2.0 * PI_A;
    var c = vec3f(0.0);
    for (var e = 0; e < 4; e++) {
      let el = (0.5 + 2.0 * f32(e)) * PI_A / 180.0;
      c += viewLutAt(vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az)));
    }
    pRing[li] = c * 0.25;
  }
  workgroupBarrier();
  if (li == 0u) {
    var U = vec3f(0.0);
    var S = vec3f(0.0);
    var N = vec3f(0.0);
    for (var k = 0; k < 64; k++) { U += pUp[k]; S += pSun[k]; N += pAnti[k]; }
    // Sky light (the LUT's units) → the fill the shaders add, a little less blue than the sky itself.
    let w = 2.0 * PI_A / 512.0 * AMB_GAIN;
    let night = sky.star2.w;
    U = desat(U * w, AMB_SAT) + NIGHT_FILL * night;
    S = desat(S * w, AMB_SAT) + NIGHT_FILL * 0.5 * night;
    N = desat(N * w, AMB_SAT) + NIGHT_FILL * 0.5 * night;
    // The ground below, lit by the key light and the fill: half the view of a vertical surface.
    let groundE = sky.keyCols[0].rgb * max(sky.key.y, 0.0) + U;
    S += groundE * GROUND_ALBEDO * 0.5;
    N += groundE * GROUND_ALBEDO * 0.5;
    let D = groundE * GROUND_ALBEDO;
    let lu = max(dot(U, LUMA_A), 1e-7);
    skyLightOut.ambUp = vec4f(U, lu);
    skyLightOut.ambSun = vec4f(atLuma(S, lu), dot(S, LUMA_A));
    skyLightOut.ambAnti = vec4f(atLuma(N, lu), dot(N, LUMA_A));
    skyLightOut.ambDown = vec4f(atLuma(D, lu), dot(D, LUMA_A));
    skyLightOut.bounce = vec4f(desat(groundE, 0.5) * BOUNCE_REF, 0.0);
    skyLightOut.zenith = vec4f(viewLutAt(vec3f(0.0, 1.0, 0.0)), 0.0);
    // The horizon round the compass as a cosine series about the key light's azimuth
    // (the sky is near symmetric about the light that makes it): the fog needs no lookup.
    var h = vec3f(0.0);
    var h1 = vec3f(0.0);
    var h2 = vec3f(0.0);
    let kaz = atan2(kd.z, kd.x);
    for (var k = 0; k < 16; k++) {
      let phi = (f32(k) + 0.5) / 16.0 * 2.0 * PI_A - kaz;
      h += pRing[k];
      h1 += pRing[k] * cos(phi);
      h2 += pRing[k] * cos(2.0 * phi);
    }
    skyLightOut.hor = vec4f(h / 16.0, 0.0);
    skyLightOut.hor1 = vec4f(h1 / 8.0, 0.0);
    skyLightOut.hor2 = vec4f(h2 / 8.0, 0.0);
    // The light on the land: the key light at the camera's altitude (on ground facing it at ≥ 37°:
    // slopes, trunks and grass blades face a low light) and the fill.
    let E = dot(sky.camKey.rgb, LUMA_A) * max(sky.key.y, 0.6) + lu;
    let ex = clamp(EXPOSURE_DAY * max(1.0, pow(E_REF / max(E, 1e-6), ADAPT)), EXPOSURE_DAY, 1500.0) * sky.grade.x;
    // At twilight a camera exposed for lit peaks shows the sky dim: lift the dome and the haze (not the light on the land).
    let horShown = dot(h, LUMA_A) / 16.0 * ex;
    let lift = 1.0 + (clamp(sqrt(TWILIGHT_SKY / max(horShown, 1e-4)), 1.0, 2.6) - 1.0) * sky.clouds.w;
    skyLightOut.grade = vec4f(ex, 1.0 / ex, E, lift);
  }
}
`;

/** The classic sky's light: the fixed sun and hazy sky every surface was first tuned under. */
export const CLASSIC_LIGHT_WGSL = /* wgsl */ `
fn keyDir() -> vec3f { return SUN; }
fn keyCol(y: f32) -> vec3f { return SUN_COL; }
fn skyAmb(n: vec3f, w: f32) -> vec3f { return SKY_TINT * w; }
fn bounceCol(c: vec3f) -> vec3f { return c; }
fn skyHorCol() -> vec3f { return SKY_HOR; }
fn glow(c: vec3f) -> vec3f { return c; }
// Sunstone lanterns and lit windows (the Eyries): dark under the classic sky's fixed sun.
fn lampLevel() -> f32 { return 0.0; }
const UNDERWATER: vec3f = vec3f(0.012, 0.055, 0.075);
fn underwaterCol() -> vec3f { return UNDERWATER; }
const CURVATURE: f32 = 0.0;
`;


/**
 * The day–night sky and light for the raster renderer. Bindings in group 0:
 * the sky uniform, the probe's results and the sky-view LUT; the LUT is read
 * with the clipmap's sampler (hSamp: repeats round the azimuth; the elevation
 * is clamped by hand).
 */
export function cycleSkyWGSL(b: { sky: number; light: number; view: number }): string {
  return /* wgsl */ `
${SKY_STRUCT_WGSL}
@group(0) @binding(${b.sky}) var<uniform> sky: SkyU;
@group(0) @binding(${b.light}) var<uniform> skyLight: SkyLight;
@group(0) @binding(${b.view}) var skyView: texture_2d<f32>;
const PI_S: f32 = 3.14159265;
const LUMA: vec3f = vec3f(0.2126, 0.7152, 0.0722);
const VIEW_H_S: f32 = ${f(VIEW_LUT[1])};
const KEY_TOP_S: f32 = ${f(KEY_TOP)};
const CURVATURE: f32 = ${f(1 / (2 * SKY.planetRadius))};
const CEL_POLE: vec3f = ${v3(POLE)};

// ---- the light API --------------------------------------------------------------------
fn keyDir() -> vec3f { return sky.key.xyz; }
fn keyCol(y: f32) -> vec3f {
  let kf = sqrt(clamp(y / KEY_TOP_S, 0.0, 1.0)) * ${f(KEY_ALTS - 1)};
  let k = min(u32(kf), ${KEY_ALTS - 2}u);
  return mix(sky.keyCols[k].rgb, sky.keyCols[k + 1u].rgb, kf - f32(k));
}
// Sky fill: the hue of the part of the sky a surface faces (toward the key
// light's azimuth or away, the zenith, the ground below), at the level of the
// sky's light on open ground; the shader's own weight w shapes it by normal.
fn skyAmb(n: vec3f, w: f32) -> vec3f {
  // (n.xz unnormalised: a surface facing up sees both sides of the sky alike)
  var c = mix(skyLight.ambAnti.rgb, skyLight.ambSun.rgb, 0.5 + 0.5 * dot(n.xz, sky.keyAz.xy));
  c = mix(c, skyLight.ambUp.rgb, max(n.y, 0.0));
  c = mix(c, skyLight.ambDown.rgb, max(-n.y, 0.0));
  return c * w;
}
fn bounceCol(c: vec3f) -> vec3f { return c * skyLight.bounce.rgb; }
fn skyHorCol() -> vec3f { return skyLight.hor.rgb; }
// Emissive light (the shrines' glyphs): its day brightness, up to three times that on the display at night.
fn glow(c: vec3f) -> vec3f { return c * min(1.0, 3.15 * skyLight.grade.y); }
// Sunstone lanterns and lit windows (the Eyries): they wake as the sun sinks toward the horizon
// and sleep once it has risen a few degrees (0 by day … 1 by night).
fn lampLevel() -> f32 { return 1.0 - smoothstep(-0.035, 0.09, sky.sun.y); }
const UNDERWATER: vec3f = vec3f(0.012, 0.055, 0.075);
fn underwaterCol() -> vec3f { return UNDERWATER * min(skyLight.ambUp.w / ${f(FILL_REF)}, 1.0); }

// ---- the sky -------------------------------------------------------------------------
fn skyLut(d: vec3f) -> vec3f {
  let el = asin(clamp(d.y, -1.0, 1.0));
  let v = 0.5 + 0.5 * sign(el) * sqrt(abs(el) / (0.5 * PI_S));
  let u = atan2(d.z, d.x) / (2.0 * PI_S);
  return textureSampleLevel(skyView, hSamp, vec2f(u, clamp(v, 0.5 / VIEW_H_S, 1.0 - 0.5 / VIEW_H_S)), 0.0).rgb * skyLight.grade.w;
}

// Haze colour toward rd: the horizon's sky round that azimuth (its glow
// toward the sun belongs to the horizon: looking down, the haze takes the
// horizon's mean colour), and light scattered forward from the key light (a
// round glow of haze toward a low sun). Lifted at twilight less than the dome.
const FOG_MIE: f32 = 0.55;
fn fogInscatter(rd: vec3f) -> vec3f {
  // (colours in half precision: hf, hv3)
  let c1 = hf(dot(rd.xz, sky.keyAz.xy) * inverseSqrt(max(dot(rd.xz, rd.xz), 1e-8)));
  let ring = max(hv3(skyLight.hor.rgb) + hv3(skyLight.hor1.rgb) * c1 + hv3(skyLight.hor2.rgb) * (2.0 * c1 * c1 - 1.0), hv3(0.0));
  let air = mix(hv3(skyLight.hor.rgb), ring, exp(hf(-abs(rd.y) * 7.0)));
  let mu = hf(dot(rd, keyDir()));
  let g = hf(0.72);
  let x = max(1.0 + g * g - 2.0 * g * mu, hf(1e-3));
  let hg = (1.0 - g * g) / (hf(4.0 * PI_S) * x * sqrt(x));
  // Looking up (at a sky island) the haze takes the colour of the sky behind it, not the horizon's
  // (the dome looked up only where it shows: not along the ground, where nearly every pixel looks).
  let up = smoothstep(0.06, 0.45, rd.y);
  // (tinted by the lands' air round the camera: sky/air.ts)
  var haze = air * hf(sqrt(skyLight.grade.w)) * hv3(vec3f(sky.keyAz.z, sky.keyAz.w, sky.grade.w));
  if (up > 0.0) { haze = mix(haze, hv3(skyLut(vec3f(rd.x, max(rd.y, 0.0), rd.z))), hf(up)); }
  return softClip(vec3f(haze + hv3(sky.camKey.rgb) * (hg * hf(FOG_MIE))), skyLight.grade.x);
}

// The cloud deck (CLOUD_PLANE) at chunk-local xz, seen from tc metres: its coverage and thickness.
fn cloudDeck(xz: vec2f, tc: f32) -> vec2f {
  let cp = xz + u.origin + vec2f(u.time * 6.0, u.time * 2.0);
  let dpc = max(tc * u.footprint, 0.02);
  let c = fbmN(cp.x * F_CLOUD, cp.y * F_CLOUD, u.seed + 501, CLOUD_OCT, 2.0, 0.5, dpc, F_CLOUD, ASUM_CLOUD);
  return vec2f(smoothstep(0.42 + sky.clouds.x, 0.78 + sky.clouds.x, 0.5 + 0.5 * c), smoothstep(0.2, 0.9, c));
}

// Its colour seen along rd, lit by the key light at its height (so it keeps the sunset's glow after
// the land loses it): from below silver toward the light and lit from beneath when it is low; from
// above (a sky island's view), its sunlit tops.
fn cloudColour(rd: vec3f, thick: f32, above: bool) -> vec3f {
  let kd = keyDir();
  let lc = keyCol(CLOUD_PLANE) * sky.clouds.y;
  let mu = max(dot(rd, kd), 0.0);
  let amb = (skyLight.ambUp.rgb * (0.62 + 0.32 * (1.0 - thick)) + skyLight.hor.rgb * 0.25) * sky.clouds.y;
  if (above) {
    let tops = lc * (0.12 + 0.55 * max(kd.y, 0.0) + 0.25 * pow(mu, 4.0)) * (0.8 + 0.3 * thick);
    return softClip(amb * 1.1 + tops, skyLight.grade.x);
  }
  let silver = pow(mu, 6.0) * (1.4 - thick) * 0.9;
  let under = smoothstep(0.1, -0.04, kd.y);
  let lit = lc * (0.2 + 0.2 * max(kd.y, 0.0) + silver + under * (0.6 - 0.25 * thick));
  return softClip(amb + lit * (1.0 - 0.45 * thick), skyLight.grade.x);
}

// The cloud deck between the camera and a surface at height h, t along rd: up through it at a sky
// island, or down through it on the land from one (its coverage there over the surface).
fn cloudVeil(col: vec3f, t: f32, h: f32, rd: vec3f) -> vec3f {
  let y0 = u.camPos.y;
  // (no cover at all within 0.02 of level: the fade below is 0 there, so the deck's noise is skipped)
  if ((y0 - CLOUD_PLANE) * (h - CLOUD_PLANE) >= 0.0 || abs(rd.y) <= 0.02) { return col; }
  let tc = (CLOUD_PLANE - y0) / rd.y;
  if (tc <= 0.0 || tc >= t) { return col; }
  let cl = cloudDeck(u.camPos.xz + rd.xz * tc, tc);
  let fade = exp(-tc * 6.0e-5) * smoothstep(0.02, 0.12, abs(rd.y));
  return mix(col, cloudColour(rd, cl.y, y0 > CLOUD_PLANE), cl.x * fade);
}

// Height fog along the view ray (as the classic raster sky), with its density
// set by the hour, a low mist layer in the morning, and the air's own colour.
fn fogDepth(y0: f32, t: f32, rdy: f32, H: f32) -> f32 {
  let k = rdy * t / H;
  if (abs(k) < 1e-3) { return t * exp(-y0 / H); }
  return t * (exp(-y0 / H) - exp(-(y0 / H + k))) / k;
}
// The fog's optical depth over t metres along rd (surfaceFogOd); past FOG_OPAQUE the surface's own
// colour no longer shows, so a surface there need not be shaded at all.
const FOG_OPAQUE: f32 = 5.82;
fn fogOd(t: f32, rd: vec3f) -> f32 {
  let y0 = max(u.camPos.y, 0.0);
  var od = SIGMA_FOG * sky.fog.x * fogDepth(y0, t, rd.y, FOG_H);
  if (sky.fog.y > 0.0) { od += sky.fog.y * fogDepth(max(y0 - u.camWater, 0.0), t, rd.y, sky.fog.z); }
  return od;
}
fn surfaceFog(col: vec3f, t: f32, h: f32, rd: vec3f) -> vec3f { return surfaceFogOd(col, t, h, rd, fogOd(t, rd)); }
fn surfaceFogOd(col: vec3f, t: f32, h: f32, rd: vec3f, od: f32) -> vec3f {
  if (STUB_FOGALL) { return col; }
  // (and over it all the cloud world's warm haze: terrain.wgsl.ts worldHaze)
  if (od < 0.0005) { return worldHaze(cloudVeil(col, t, h, rd), t, rd); }
  let ins = fogInscatter(rd);
  if (od > FOG_OPAQUE) { return worldHaze(cloudVeil(ins, t, h, rd), t, rd); }
  let T = exp(hv3(-0.58, -0.72, -0.95) * hf(od));
  // (the deck veils the fogged surface as it veils the sky round it)
  return worldHaze(cloudVeil(vec3f(hv3(col) * T + hv3(ins) * (1.0 - T)), t, h, rd), t, rd);
}

// The water's sky: the dome without its discs (the key light's glint stands in).
fn skyReflect(r: vec3f) -> vec3f {
  return skyLut(vec3f(r.x, max(r.y, 0.0), r.z));
}

// ---- night: stars, the Milky Way, airglow ------------------------------------------------
const STAR_CELLS: f32 = 220.0;
// A star is at least a pixel wide: with larger pixels (a low internal
// resolution) its light spreads over more sky, so its peak dims by the
// pixels' area over that of a 1280 × 800 view (it keeps its brightness).
const STAR_FOOTPRINT: f32 = 0.00175;
// Four random numbers per star cell (the cell's coordinates hashed together first:
// hashU4 mixes each lane on its own).
fn starRand(face: u32, c: vec2i) -> vec4f {
  let h0 = hash3(c.x, c.y, i32(face) * 7919 + 3301);
  let h = hashU4(vec4u(h0, h0 ^ 0x68bc21ebu, h0 + 0x02e5be93u, h0 * 0x967a889bu + 1u));
  return vec4f(h) * (1.0 / 4294967295.0);
}
fn starTint(t: f32) -> vec3f {
  if (t < 0.18) { return vec3f(0.72, 0.82, 1.0); }
  if (t < 0.62) { return vec3f(0.97, 0.97, 1.0); }
  if (t < 0.88) { return vec3f(1.0, 0.9, 0.74); }
  return vec3f(1.0, 0.74, 0.52);
}
// Galactic frame (celestial coordinates): the band's pole and the direction of its bright core.
const MW_POLE: vec3f = vec3f(0.46, -0.52, 0.72);
const MW_CORE: vec3f = vec3f(0.86, 0.47, -0.2);
fn milkyWay(d: vec3f) -> vec3f {
  let b = dot(d, MW_POLE);
  let band = exp(-b * b * 22.0);
  if (band < 0.02) { return vec3f(0.0); }
  let e1 = normalize(MW_CORE - MW_POLE * dot(MW_CORE, MW_POLE));
  let e2 = cross(MW_POLE, e1);
  let x = dot(d, e1);
  let y = dot(d, e2);
  let n1 = 0.5 + 0.5 * vnoise(x * 5.0 + b * 7.0, y * 5.0 - b * 9.0, 3301);
  let n2 = 0.5 + 0.5 * vnoise(x * 13.0 - b * 21.0, y * 13.0 + b * 17.0, 3303);
  let clouds = n1 * 0.65 + n2 * 0.35;
  // dark dust lanes along the middle, broken up
  let lq = (b + 0.05 * (n2 - 0.5)) * 14.0;
  let lane = exp(-lq * lq) * smoothstep(0.35, 0.75, 0.5 + 0.5 * vnoise(x * 7.0, y * 7.0, 3307));
  let core = max(x, 0.0);
  let c = mix(vec3f(0.55, 0.64, 0.86), vec3f(0.98, 0.84, 0.66), core * core);
  return c * band * (0.25 + 0.95 * clouds * clouds) * (1.0 - 0.75 * lane) * (0.55 + 0.9 * core * core);
}
fn starField(rd: vec3f, fade: f32) -> vec3f {
  let d = vec3f(dot(sky.star0.xyz, rd), dot(sky.star1.xyz, rd), dot(sky.star2.xyz, rd));
  let a = abs(d);
  var face: u32;
  var uv: vec2f;
  if (a.x >= a.y && a.x >= a.z) { face = select(1u, 0u, d.x > 0.0); uv = d.yz / a.x; }
  else if (a.y >= a.z) { face = select(3u, 2u, d.y > 0.0); uv = d.xz / a.y; }
  else { face = select(5u, 4u, d.z > 0.0); uv = d.xy / a.z; }
  let g = (uv * 0.5 + 0.5) * STAR_CELLS;
  // pixels per cell: a cell spans ~2/STAR_CELLS rad where the face is flattest
  let px = max(u.footprint * STAR_CELLS * 0.5 * (1.0 + dot(uv, uv)), 1e-4);
  let mw = milkyWay(d);
  // a star in ~7 % of cells, twice as many along the Milky Way; most faint, a few bright
  let p = 0.07 * (1.0 + 1.2 * min(dot(mw, vec3f(2.5)), 1.0));
  let base = floor(g - 0.5);
  var col = vec3f(0.0);
  for (var j = 0; j < 4; j++) {
    let c = base + vec2f(f32(j & 1), f32(j >> 1));
    let h = starRand(face, vec2i(c));
    let e = (h.x - (1.0 - p)) / p;
    if (e <= 0.0) { continue; }
    // most faint, some clear, ~1 % brilliant (with a small glow)
    let e4 = e * e * e * e;
    let bright = pow(e, 80.0);
    let sp = c + 0.08 + 0.84 * h.yz;
    let dpx = length(g - sp) / px;
    let core = exp(-dpx * dpx * 1.7) + 0.25 * bright * exp(-dpx * dpx * 0.2);
    let tw = 1.0 + 0.45 * (1.0 - smoothstep(0.05, 0.5, rd.y)) * sin(sky.star1.w * (2.1 + 5.0 * h.y) + h.z * 60.0);
    col += starTint(h.w) * (0.035 + 0.32 * e4 + 2.2 * bright) * core * tw;
  }
  let spread = STAR_FOOTPRINT / u.footprint;
  return (col * clamp(spread * spread, 0.06, 1.6) + mw * (0.16 * sky.clouds.z)) * fade;
}

// A shooting star now and then: every 23 s a 25 % chance of a streak across
// the upper sky, a second long, brightest at its head (display terms).
fn meteor(rd: vec3f, time: f32) -> f32 {
  let k = floor(time / 23.0);
  let h0 = hashU(u32(k) ^ 0x6d2b79f5u);
  let h = vec4f(hashU4(vec4u(h0, h0 ^ 0x1b873593u, h0 + 0x2545f491u, h0 * 0x9e3779b9u + 1u))) * (1.0 / 4294967295.0);
  if (h.w > 0.25) { return 0.0; }
  let age = (time - k * 23.0 - 3.0 * h.x) / 1.1;
  if (age < 0.0 || age > 1.0) { return 0.0; }
  let az = h.x * 6.2831853;
  let el = mix(0.5, 1.15, h.y);
  let p0 = vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
  // travels mostly downward, a little sideways
  let side = normalize(cross(p0, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-4, 0.0, 0.0));
  let down = normalize(cross(side, p0));
  let dir = normalize(-down + side * (h.z * 2.0 - 1.0) * 0.8);
  let head = normalize(p0 + dir * (0.3 * age));
  let tail = normalize(p0 + dir * max(0.3 * age - 0.09, 0.0));
  let seg = head - tail;
  let f = clamp(dot(rd - tail, seg) / max(dot(seg, seg), 1e-8), 0.0, 1.0);
  let d = length(rd - (tail + seg * f)) / max(u.footprint * 1.3, 2e-4);
  return exp(-d * d) * f * f * sin(3.14159265 * age) * 1.6;
}

// The moon: a sphere lit by the sun (its phase), grey maria, a faint earthshine on the dark side.
fn moonColor(rd: vec3f) -> vec4f {
  let M = sky.moon.xyz;
  let r = sky.moonDisc.w;
  let c = dot(rd, M);
  let sr = sin(r);
  if (c < cos(r * 1.05)) { return vec4f(0.0); }
  let t2 = normalize(CEL_POLE - M * dot(CEL_POLE, M));
  let t1 = cross(t2, M);
  let q = vec2f(dot(rd, t1), dot(rd, t2)) / sr;
  let rr = dot(q, q);
  let edge = 1.0 - smoothstep(1.0 - 2.5 * u.footprint / sr, 1.0, sqrt(rr));
  if (edge <= 0.0) { return vec4f(0.0); }
  let n = q.x * t1 + q.y * t2 - sqrt(max(1.0 - rr, 0.0)) * M;
  let lit = smoothstep(-0.04, 0.1, dot(n, sky.sun.xyz));
  let mare = smoothstep(0.42, 0.66, 0.5 + 0.5 * vnoise(q.x * 2.3 + 11.0, q.y * 2.3 + 4.0, 3401) + 0.18 * vnoise(q.x * 6.0, q.y * 6.0, 3403));
  let crater = smoothstep(0.55, 0.9, 0.5 + 0.5 * vnoise(q.x * 17.0, q.y * 17.0, 3407));
  let alb = mix(1.0, 0.58, mare) * (0.9 + 0.16 * crater);
  let col = sky.moonDisc.rgb * alb * (lit + 0.015) * (1.0 - 0.12 * rr);
  return vec4f(col, edge);
}

// ---- the dome -------------------------------------------------------------------------
// The brightest light in view keeps its colour: above 0.6 on the display its
// strongest channel is eased toward 1.6 (the sunset's pink on the clouds
// would otherwise burn to white at a dusk exposure).
fn softClip(c: vec3f, ex: f32) -> vec3f {
  let m = max(max(c.r, c.g), c.b) * ex;
  if (m <= 0.6) { return c; }
  let fm = 0.6 + (m - 0.6) / (1.0 + (m - 0.6));
  return c * (fm / m);
}
fn shadeSky(ro: vec3f, rd: vec3f) -> vec3f {
  // Below the horizon lies land beyond the rings' reach, as far off as land can be: only its haze
  // shows, the colour the land and its water fade into before the rings end (terrain.wgsl.ts reachOd).
  if (rd.y < 0.0) { return cloudVeil(fogInscatter(rd), 1e30, 0.0, rd); }
  let d = vec3f(rd.x, max(rd.y, 0.0), rd.z);
  var col = softClip(skyLut(d), skyLight.grade.x);
  let night = sky.star2.w;
  // What is drawn in display terms (airglow, stars, the moon) is scaled by 1 / exposure.
  let inv = skyLight.grade.y;
  // airglow and starlight: the night sky is never black
  col += (vec3f(0.012, 0.018, 0.04) + vec3f(0.01, 0.024, 0.014) * exp(-max(rd.y, 0.0) * 9.0)) * (night * inv);
  let m = moonColor(rd);
  if (sky.star0.w > 0.001 && rd.y > -0.01) {
    // stars dim in a bright sky (round the moon) and in the thick air low down, and hide behind the moon
    let vis = sky.star0.w * exp(-dot(col * skyLight.grade.x, LUMA) * 7.0) * smoothstep(-0.01, 0.2, rd.y) * (1.0 - m.a);
    if (vis > 0.002) { col += (starField(rd, vis) + vec3f(0.85, 0.9, 1.0) * meteor(rd, sky.star1.w) * vis) * inv; }
  }
  // the sun: a limb-darkened disc and a tight corona (the air's glow round it is in the LUT)
  let cs = dot(rd, sky.sun.xyz);
  if (cs > 0.99) {
    let ang = acos(min(cs, 1.0));
    let rs = sky.sunDisc.w;
    let x = min(ang / rs, 1.0);
    let disc = 1.0 - smoothstep(1.0 - 1.5 * u.footprint / rs, 1.0, ang / rs);
    col += sky.sunDisc.rgb * (disc * (1.0 - 0.45 * (1.0 - sqrt(1.0 - x * x))) + 0.02 * exp(-max(ang - rs, 0.0) / 0.01));
  }
  // the moon over the sky behind it, and a soft corona over the air's own scattering of its light
  col += (m.rgb * m.a + sky.moonDisc.rgb * 0.05 * exp(-acos(clamp(dot(rd, sky.moon.xyz), -1.0, 1.0)) / 0.05) * sky.moonE.w * night) * inv;
  // clouds (cloudDeck, cloudColour): seen from below, against the sky
  if (rd.y > 0.02 && ro.y < CLOUD_PLANE) {
    let tc = (CLOUD_PLANE - ro.y) / rd.y;
    let cl = cloudDeck(ro.xz + rd.xz * tc, tc);
    let fade = exp(-tc * 6.0e-5) * smoothstep(0.02, 0.12, rd.y);
    col = mix(col, cloudColour(rd, cl.y, false), cl.x * fade);
  }
  return col;
}

// ---- grade ------------------------------------------------------------------------------
// As the classic raster grade (cool lifted shadows, warm light, a filmic
// shoulder), with the exposure the probe adapted, the hour's saturation, and
// a shift toward blue for the eye's night vision.
fn outputTransform(col: vec3f) -> vec3f {
  // (in half precision: hf, hv3)
  let LH = hv3(LUMA);
  var c = hv3(col * skyLight.grade.x);
  let l = dot(c, LH);
  c *= mix(hv3(0.9, 0.97, 1.14), hv3(1.04, 1.0, 0.94), smoothstep(hf(0.015), hf(0.35), l));
  c += hv3(0.004, 0.006, 0.011) * (1.0 - smoothstep(hf(0.0), hf(0.08), l));
  let ln = dot(c, LH);
  c = mix(c, ln * hv3(0.62, 0.82, 1.18), hf(sky.grade.z * 0.4));
  c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), hv3(0.0), hv3(1.0));
  let l2 = dot(c, LH);
  return vec3f(mix(hv3(l2), c, hf(sky.grade.y)));
}
`;
}
