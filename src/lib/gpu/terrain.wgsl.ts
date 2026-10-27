/**
 * WebGPU terrain shaders: the ray-march renderer, the raster clipmap
 * renderer, the clipmap generator and the elevation probe.
 * ---------------------------------------------------------------------------
 * `terrainShader({ height: 'direct' })` is a port of demoShader.ts main()
 * (no hero yet): it evaluates E(x, z) for every march sample exactly like the
 * WebGL2 renderer, so the two can be diffed frame by frame.
 * `terrainShader({ height: 'cached', … })` runs the same march and shading
 * but reads heights from the clipmap (see clipmap.ts) that `genShader`
 * fills: one evaluation of E per lattice point, instead of ~90 per pixel per
 * frame. Every height query goes through heightAt(local, offset, dp) with the
 * GLSL's own operand order, so the direct variant rounds like the original.
 * `rasterShader(…)` draws the same cached surface as nested morphing grid
 * rings (a geometry clipmap) and shades it with the same shadeSurface().
 */

import { worldWGSL, wgslF32 as f } from './worldgen.wgsl';
import type { World } from '../world';
import { HERO_WGSL, NO_HERO_WGSL } from './hero.wgsl';
import { combatWGSL } from './combat.wgsl';
import { rocksWGSL } from './rocks.wgsl';
import { horsesWGSL } from './horses.wgsl';
import { colossusWGSL } from './colossus.wgsl';
import { modelIslesWGSL } from './modelIsles.wgsl';
import { GROUND_SAMPLE_WGSL } from './groundTex.wgsl';
import { GROUND_DETAIL_WGSL } from './groundDetail.wgsl';
import { GROUND_COVER_WGSL } from './groundCover.wgsl';
import { grassWGSL, type GrassSpec } from './grass.wgsl';
import { skyRealmWGSL } from './skyRealm.wgsl';
import { WATERFALLS_WGSL } from './waterfalls.wgsl';
import { structuresWGSL } from './structures.wgsl';
import { effectsWGSL } from './effects.wgsl';
import { faunaWGSL } from './fauna.wgsl';
import { motesWGSL } from './motes.wgsl';
import { lanternsWGSL } from './lanterns';
import { RUINS_WGSL, RUIN_SITES_WGSL } from './ruins.wgsl';
import { PUFFS_WGSL } from './puffs.wgsl';
import { canopyWGSL, forestWGSL, type ForestSpec } from './forest.wgsl';
import { PLACE } from '../forest/placement';
import { SPECIES, SPECIES_INFO } from '../forest/species';
import { CLASSIC_LIGHT_WGSL, cycleSkyWGSL } from './sky.wgsl';
import { CAMERA, CLIMATE, LANDS, MARCH, SHADOW } from '../constants';
import { BIOME_COUNT, SNOW_ALBEDO } from '../biomes';
import { RING_BLOCK, ringBlockOrder } from './rings';
import { PATH_N, SKY } from '../sky/celestial';

/** The farthest any tree is drawn (m): beyond it the forest is painted on the ground. */
const TREE_REACH = Math.max(...SPECIES.map((s) => SPECIES_INFO[s].maxDist));

type CachedOptions = {
  /** Clipmap texels per side. */
  n: number;
  levels: number;
  /** Width, in log2 distance, of the band where a level fades into the next. */
  blend: number;
  /**
   * Bind the cached sun layer as sunTex / sunAt() (binding 7): 'classic', the
   * ground's visibility of the fixed sun (sunShader); 'path', the ground's
   * horizons along the sun's path (horizonShader), for a sun that moves.
   */
  sun?: 'classic' | 'path';
  /** The world whose surface the shader evaluates (default: the land). */
  world?: World;
};

type TerrainShaderOptions = { height: 'direct'; world?: World } | ({ height: 'cached' } & CachedOptions);

/** Frame uniform block: 112 bytes (see setFrame in terrainRenderer.ts). */
export const FRAME_BYTES = 112;
/** Clipmap uniform block: one vec4 per level plus the camera vec4. */
export const clipBytes = (levels: number) => 16 * (levels + 1);
/** Generator uniform block (one per dispatched rectangle, 256-byte slots). */
export const GEN_BYTES = 48;
export const PROBE_BYTES = 32;
/** Raster mesh uniform block: 3 vec4, then per ring a corner and a centre vec4, then rings + 1 texture offsets. */
export const meshBytes = (rings: number) => 16 * (3 + 2 * rings + rings + 1);

/**
 * The colour arithmetic's types: hf / hv3 are f16 / vec3<f16> where the
 * device has shader-f16 (half the registers, twice the rate on this GPU:
 * the fog's colours, the tone map and the gamma, after the lighting), else
 * f32 / vec3f. Positions, distances and depths stay f32 everywhere.
 */
const HALF_WGSL = (f16: boolean) => f16
  ? 'enable f16;\nalias hf = f16;\nalias hv3 = vec3<f16>;'
  : 'alias hf = f32;\nalias hv3 = vec3f;';

const NO_TINT_WGSL = /* wgsl */ `
fn grassDensityFrom(veg: f32, slope: f32, patchN: f32) -> f32 {
  return veg * smoothstep(0.15, 0.55, patchN + 0.25 * veg) * (1.0 - smoothstep(0.7, 1.1, slope));
}
`;

/**
 * The ground's biome look (its tables: worldgen.wgsl.ts LK_*), and
 * its encoding in the clipmap's biome layers (biomeShader writes them per lattice point, the raster
 * ground and grass read them): mat (rgba16f) — the snow line (m), wetness (≥ 0) or lava (< 0), the
 * meadow patches, flowers; bioA — soil (√rgb), grass cover; bioB — grass (√rgb, its hue patches baked
 * in), blade height / BLADE_SCALE; bioC — rock (√rgb), strata.
 */
const BLADE_SCALE = 6;
const lookWGSL = () => /* wgsl */ `
const BLADE_SCALE: f32 = ${f(BLADE_SCALE)};
`;

/**
 * The far forest's canopy at a lattice point (forest.wgsl.ts canopyAt), into the clipmap's canopy
 * layer (binding 6), from the ground there: its height e, its water, gradient, and its relief over
 * the forest bodies' hill step (placement.ts PLACE.hills; no finer than the lattice).
 */
const CANOPY_LAYER_WGSL = /* wgsl */ `
@group(0) @binding(6) var canopyOut: texture_storage_2d_array<rgba8unorm, write>;
${canopyWGSL()}
fn writeCanopy(st: vec2u, xz: vec2f, e: f32, grad: vec2f, s: f32) {
  let H = max(${f(PLACE.hills.step)}, s);
  let hE = heightAt(xz, vec2f(H, 0.0), s);
  let hW = heightAt(xz, vec2f(-H, 0.0), s);
  let hN = heightAt(xz, vec2f(0.0, H), s);
  let hS = heightAt(xz, vec2f(0.0, -H), s);
  let hills = (hE + hW + hN + hS - 4.0 * e) / H;
  let hillSlope = length(vec2f(hE - hW, hN - hS)) / (2.0 * H);
  textureStore(canopyOut, st, gen.level, canopyAt(xz + gen.origin, e, waterAt(xz), length(grad), hills, hillSlope, gen.seed));
}
`;

const FRAME_WGSL = /* wgsl */ `
struct Frame {
  basis: mat3x3f,      // view → world, columns right / up / forward
  camPos: vec3f,       // chunk-local
  time: f32,
  origin: vec2f,       // chunk origin, multiples of CHUNK_SIZE (fp32-exact)
  res: vec2f,
  tanHalf: f32,
  footprint: f32,      // 2·tan(FOV/2)/res.y
  seed: i32,
  marchSteps: i32,
  shadowSteps: i32,
  camWater: f32,       // the water's surface under the camera (lakes, rivers, the sea): below it, the view is underwater
  jitter: vec2f,       // TAA projection jitter (NDC); 0 without TAA
};
@group(0) @binding(0) var<uniform> u: Frame;
`;

function heightSourceWGSL(opts: TerrainShaderOptions): string {
  if (opts.height === 'direct') {
    return /* wgsl */ `
fn heightAt(xz: vec2f, off: vec2f, dp: f32) -> f32 {
  return elevation(xz + u.origin + off, dp, u.seed);
}
fn waterAt(xz: vec2f) -> f32 {
  return landSurface(xz + u.origin, 0.0, u.seed).w;
}
fn flowAt(xz: vec2f) -> vec2f {
  return landSurface(xz + u.origin, 0.0, u.seed).flow;
}
`;
  }
  return /* wgsl */ `
struct Clip {
  lv: array<vec4f, ${opts.levels}>, // per level: lattice coordinate of the camera mod n (+0.5), 1/spacing, spacing
  cam: vec4f,                       // camera x, z (chunk-local), 1/usableRadius(level 0), unused
};
@group(0) @binding(1) var<uniform> clip: Clip;
@group(0) @binding(2) var hTex: texture_2d_array<f32>;
@group(0) @binding(3) var hSamp: sampler;
const CLIP_N: f32 = ${f(opts.n)};
const CLIP_INV_N: f32 = ${f(1 / opts.n)};
const CLIP_LMAX: f32 = ${f(opts.levels - 1)};
const CLIP_BLEND: f32 = ${f(opts.blend)};

fn clipSample(t: texture_2d_array<f32>, rel: vec2f, l: i32) -> f32 {
  let L = clip.lv[l];
  return textureSampleLevel(t, hSamp, (rel * L.z + L.xy) * CLIP_INV_N, l, 0.0).r;
}

// The finest level whose window holds the point, faded into the next level
// over the outer CLIP_BLEND (log2 distance) of its range so the surface stays
// continuous across level borders.
fn clipLookup(t: texture_2d_array<f32>, xz: vec2f, off: vec2f) -> f32 {
  let rel = (xz - clip.cam.xy) + off;
  let d = max(abs(rel.x), abs(rel.y));
  let lf = log2(max(d * clip.cam.z, 1e-6));
  let lc = clamp(ceil(lf), 0.0, CLIP_LMAX);
  let l = i32(lc);
  let h = clipSample(t, rel, l);
  let b = clamp((lf - lc) / CLIP_BLEND + 1.0, 0.0, 1.0);
  if (b > 0.0 && lc < CLIP_LMAX) {
    return mix(h, clipSample(t, rel, l + 1), b);
  }
  return h;
}

fn heightAt(xz: vec2f, off: vec2f, dp: f32) -> f32 {
  return clipLookup(hTex, xz, off);
}
// The clipmap's two channels at chunk-local xz: the ground's height and the water's surface (the
// sea, a lake, a river's pool), from the level (or two, in a blend band) that holds it.
fn clipLookupRG(t: texture_2d_array<f32>, xz: vec2f) -> vec2f {
  let rel = xz - clip.cam.xy;
  let lf = log2(max(max(abs(rel.x), abs(rel.y)) * clip.cam.z, 1e-6));
  let lc = clamp(ceil(lf), 0.0, CLIP_LMAX);
  let l = i32(lc);
  let L = clip.lv[l];
  var v = textureSampleLevel(t, hSamp, (rel * L.z + L.xy) * CLIP_INV_N, l, 0.0).rg;
  let b = clamp((lf - lc) / CLIP_BLEND + 1.0, 0.0, 1.0);
  if (b > 0.0 && lc < CLIP_LMAX) {
    let L1 = clip.lv[l + 1];
    v = mix(v, textureSampleLevel(t, hSamp, (rel * L1.z + L1.xy) * CLIP_INV_N, l + 1, 0.0).rg, b);
  }
  return v;
}
fn waterAt(xz: vec2f) -> f32 { return clipLookupRG(hTex, xz).y; }
// The rivers' flow at chunk-local xz (m/s, world xz): the clipmap's third and fourth channels.
fn flowAt(xz: vec2f) -> vec2f {
  let rel = xz - clip.cam.xy;
  let lf = log2(max(max(abs(rel.x), abs(rel.y)) * clip.cam.z, 1e-6));
  let l = i32(clamp(ceil(lf), 0.0, CLIP_LMAX));
  let L = clip.lv[l];
  return textureSampleLevel(hTex, hSamp, (rel * L.z + L.xy) * CLIP_INV_N, l, 0.0).ba;
}
${opts.sun === 'classic' ? `
// Cached sun visibility of the ground (clipmapCache.ts, sunShader below).
@group(0) @binding(7) var sunTex: texture_2d_array<f32>;
fn sunAt(xz: vec2f) -> f32 {
  return clipLookup(sunTex, xz, vec2f(0.0));
}
// The visibility in a texel of the sun layer already fetched (groundCacheAt).
fn sunFrom(s: vec4f) -> f32 { return s.x; }
` : opts.sun === 'path' ? `
// Cached horizons of the ground along the sun's path (clipmapCache.ts,
// horizonShader below): the key light (sky.key, on the path at angle w) is
// seen above the angle it clears the land at when rising (r) and below the
// one it sinks behind when setting (g), with a soft edge (as softShadow's).
@group(0) @binding(7) var sunTex: texture_2d_array<f32>;
fn clipSample2(t: texture_2d_array<f32>, rel: vec2f, l: i32) -> vec2f {
  let L = clip.lv[l];
  return textureSampleLevel(t, hSamp, (rel * L.z + L.xy) * CLIP_INV_N, l, 0.0).rg;
}
fn clipLookup2(t: texture_2d_array<f32>, xz: vec2f) -> vec2f {
  let rel = xz - clip.cam.xy;
  let d = max(abs(rel.x), abs(rel.y));
  let lf = log2(max(d * clip.cam.z, 1e-6));
  let lc = clamp(ceil(lf), 0.0, CLIP_LMAX);
  let l = i32(lc);
  let h = clipSample2(t, rel, l);
  let b = clamp((lf - lc) / CLIP_BLEND + 1.0, 0.0, 1.0);
  if (b > 0.0 && lc < CLIP_LMAX) {
    return mix(h, clipSample2(t, rel, l + 1), b);
  }
  return h;
}
const PATH_SOFT: f32 = ${f(SHADOW.penumbraK * 1.2)};
fn sunAt(xz: vec2f) -> f32 {
  let h = clipLookup2(sunTex, xz);
  let th = sky.key.w;
  return clamp(PATH_SOFT * (th - h.x), 0.0, 1.0) * clamp(PATH_SOFT * (h.y - th), 0.0, 1.0);
}
// The same from a texel of the sun layer already fetched (groundCacheAt).
fn sunFrom(s: vec4f) -> f32 {
  let th = sky.key.w;
  return clamp(PATH_SOFT * (th - s.x), 0.0, 1.0) * clamp(PATH_SOFT * (s.y - th), 0.0, 1.0);
}
` : ''}`;
}

const SHADE_WGSL = /* wgsl */ `
// Bench-only stubs (pipeline-overridable constants; the game never sets them): leave out one part
// of the terrain's shading to measure what it costs.
override STUB_BIOME: bool = false;
override STUB_GRAD: bool = false;
override STUB_SHADOW: bool = false;
override STUB_FOG: bool = false;
override STUB_SHADE: bool = false;
// Bench only: shade the raster ground per pixel as before its clipmap material (shadeSurface), to compare.
override PER_PIXEL_GROUND: bool = false;
// Bench only, for every surface: no tone mapping or gamma (display), no fog (surfaceFog).
override STUB_DISPLAY: bool = false;
override STUB_FOGALL: bool = false;
// Bench only: the raster ground's normal from the sun layer's cached gradient (smooth) instead of the
// cached heights' own slopes at the pixel's footprint (gradientAt, as shadeSurface).
override CACHED_GRAD: bool = false;
// Bench only, to match the far forest to the trees it stands for: 1 no canopy painted (the trees
// alone), 2 the canopy painted at every distance in full, 3 its cover and the distance as colours.
override CANOPY_MODE: u32 = 0u;
const SH_START: f32 = ${f(SHADOW.startT)};
const SH_FOOT: f32 = ${f(SHADOW.footprintScale)};
const SH_K: f32 = ${f(SHADOW.penumbraK)};
const SH_CUTOFF: f32 = ${f(SHADOW.cutoff)};
const SH_STEP_SCALE: f32 = ${f(SHADOW.stepScale)};
const SH_STEP_MIN: f32 = ${f(SHADOW.stepMin)};
const SH_STEP_MAX: f32 = ${f(SHADOW.stepMax)};
const SH_GROWTH: f32 = ${f(SHADOW.stepGrowth)};

// §17 central differences at the pixel footprint.
fn gradientAt(xz: vec2f, eps: f32, dp: f32) -> vec2f {
  let e = max(eps, 0.05);
  let inv2e = 0.5 / e;
  let gx = (heightAt(xz, vec2f(e, 0.0), dp) - heightAt(xz, vec2f(-e, 0.0), dp)) * inv2e;
  let gz = (heightAt(xz, vec2f(0.0, e), dp) - heightAt(xz, vec2f(0.0, -e), dp)) * inv2e;
  return vec2f(gx, gz);
}

// §18 soft shadow ray (4× footprint → free penumbra).
fn softShadow(p: vec3f, footprint: f32) -> f32 {
  var res = 1.0;
  var t = SH_START;
  let fp4 = SH_FOOT * footprint;
  for (var i = 0; i < 40; i++) {
    if (i >= u.shadowSteps) { break; }
    let q = p + keyDir() * t;
    if (q.y > MAXH) { break; }
    let h = q.y - heightAt(q.xz, vec2f(0.0), t * fp4) + t * t * CURVATURE;
    if (h <= 0.0) { return 0.0; }
    res = min(res, SH_K * h / t);
    if (res < SH_CUTOFF) { return 0.0; }
    t += clamp(h * SH_STEP_SCALE, SH_STEP_MIN, SH_STEP_MAX) + t * SH_GROWTH;
  }
  return clamp(res, 0.0, 1.0);
}

// Terrain (terrain = true) or water surface point p (chunk-local) seen at
// distance tS along rd. One soft-shadow call site for both.
fn shadeSurface(terrain: bool, p: vec3f, rd: vec3f, tS: f32, eWater: f32) -> vec3f {
  let g = p.xz + u.origin;
  let dp = max(tS * u.footprint, 1e-4);

  var n: vec3f;
  var gradLenSq = 0.0;
  if (terrain && STUB_GRAD) {
    n = vec3f(0.0, 1.0, 0.0);
  } else if (terrain) {
    let grad2 = gradientAt(p.xz, dp, dp);
    gradLenSq = dot(grad2, grad2);
    let invLen = inverseSqrt(gradLenSq + 1.0);
    n = vec3f(-grad2.x, 1.0, -grad2.y) * invLen;
  } else {
    let r1 = vnoise(g.x / 5.5 + u.time * 0.35, g.y / 5.5 - u.time * 0.22, u.seed + 911);
    let r2 = vnoise(g.x / 9.7 - u.time * 0.20, g.y / 9.7 + u.time * 0.31, u.seed + 913);
    let wr = vec2f(-0.22 * r1, -0.22 * r2);
    let invWLen = inverseSqrt(dot(wr, wr) + 1.0);
    n = vec3f(wr.x, 1.0, wr.y) * invWLen;
  }

  let ndl = max(dot(n, keyDir()), 0.0);
  var sh = 0.0;
  if (terrain && STUB_SHADOW) {
    sh = 1.0;
  } else if (!terrain || ndl > 0.0) {
    var so = vec3f(0.0, 0.15, 0.0);
    if (terrain) { so = n * 0.15; }
    sh = sunVisibility(p, so);
    if (sh > 0.0) { sh = min(sh, heroShadowAt(p - u.camPos)); }
  }

  var col: vec3f;
  if (terrain) {
    let e = p.y;
    var albedo = vec3f(0.1, 0.16, 0.05);
    if (!STUB_BIOME) {
      let wl = waterAt(p.xz);
      albedo = bedTint(groundAt(g, e, sqrt(gradLenSq), wl, u.seed).col, p, wl);
    }
    let sunTerm = albedo * keyCol(p.y) * ndl * sh * causticsAt(p, W_LEVEL);
    let ambTerm = albedo * skyAmb(n, 0.22 + 0.28 * max(n.y, 0.0)) * heroAOAt(p - u.camPos);
    col = sunTerm + ambTerm;
    if (!STUB_FOG) { col = surfaceFog(col, tS, p.y, rd); }
  } else {
    let depth = max(W_LEVEL - eWater, 0.0);
    let shallow = vec3f(0.110, 0.320, 0.300);
    let deep = vec3f(0.016, 0.075, 0.100);
    var body = mix(shallow, deep, smoothstep(1.5, 14.0, depth))
      * (0.35 + 0.65 * sh) * (0.6 + 0.4 * max(dot(n, keyDir()), 0.0));
    let foamN = vnoise(g.x / 3.1, g.y / 3.1, u.seed + 917);
    body += vec3f(0.9) * (1.0 - smoothstep(0.05, 0.6, depth)) * (0.25 + 0.35 * foamN);

    let refld = reflect(rd, n);
    let refl = skyColor(refld);
    let f1 = 1.0 - max(dot(-rd, n), 0.0);
    let f2 = f1 * f1;
    let f4 = f2 * f2;
    let fres = 0.02 + 0.98 * (f4 * f1);

    let sdot = max(dot(refld, keyDir()), 0.0);
    var spec = 0.0;
    if (sdot > 0.8) {
      let sp2 = sdot * sdot;
      let sp4 = sp2 * sp2;
      let sp8 = sp4 * sp4;
      let sp16 = sp8 * sp8;
      let sp32 = sp16 * sp16;
      let sp64 = sp32 * sp32;
      let sp128 = sp64 * sp64;
      spec = (sp128 * sp64) * (sp32 * sp16) * 3.0;
    }
    col = mix(body, refl, fres) + keyCol(p.y) * spec / 3.4;
    col = surfaceFog(col, tS, W_LEVEL, rd);
  }
  return col;
}

fn display(col: vec3f) -> vec4f {
  if (STUB_DISPLAY) { return vec4f(col, 1.0); }
  return vec4f(vec3f(pow(max(hv3(outputTransform(col)), hv3(0.0)), hv3(1.0 / 2.2))), 1.0);
}

// Camera ray through a framebuffer position (WebGPU y runs down, the GLSL's
// gl_FragCoord.y runs up).
fn cameraRay(fc: vec2f) -> vec3f {
  let ndc = vec2f(fc.x, u.res.y - fc.y) / u.res * 2.0 - 1.0 - u.jitter;
  let aspect = u.res.x / u.res.y;
  let dirView = vec3f(ndc.x * aspect * u.tanHalf, ndc.y * u.tanHalf, 1.0);
  return normalize(u.basis * dirView);
}

fn underwater(rd: vec3f) -> vec4f {
  return vec4f(pow(underwaterCol() * (0.6 + 0.4 * max(rd.y, 0.0)), vec3f(1.0 / 2.2)), 1.0);
}
`;

/** The classic sky dome (§20/§22): the fixed gradient and halo, and the cloud layer (as the function `name`). */
const classicDomeWGSL = (name: string) => /* wgsl */ `
fn ${name}(ro: vec3f, rd: vec3f) -> vec3f {
  var col = skyGrade(skyColor(rd), rd);
  if (rd.y > 0.02 && ro.y < CLOUD_PLANE) {
    let tc = (CLOUD_PLANE - ro.y) / rd.y;
    let cp = ro.xz + rd.xz * tc + u.origin + vec2f(u.time * 6.0, u.time * 2.0);
    let dpc = max(tc * u.footprint, 0.02);
    let c = fbmN(cp.x * F_CLOUD, cp.y * F_CLOUD, u.seed + 501, CLOUD_OCT, 2.0, 0.5, dpc, F_CLOUD, ASUM_CLOUD);
    let cov = smoothstep(0.42, 0.78, 0.5 + 0.5 * c);
    let shade = 0.85 + 0.30 * (1.0 - smoothstep(0.2, 0.9, c));
    let cloudCol = vec3f(shade) * (0.62 + 0.38 * max(SUN.y, 0.0));
    let fade = exp(-tc * 6.0e-5) * smoothstep(0.02, 0.12, rd.y);
    col = mix(col, cloudCol, cov * fade);
  }
  return col;
}

`;

/** The ray-march renderer: fullscreen triangle, one fragment per pixel. */
export function terrainShader(opts: TerrainShaderOptions): string {
  return /* wgsl */ `${HALF_WGSL(false)}
${worldWGSL(opts.world ?? 'land')}
${FRAME_WGSL}
${heightSourceWGSL(opts)}
${CLASSIC_LIGHT_WGSL}
${SHADE_WGSL}
${classicDomeWGSL('shadeSky')}
${NO_HERO_WGSL}
fn sunVisibility(p: vec3f, so: vec3f) -> f32 { return softShadow(p + so, u.footprint); }
fn surfaceFog(col: vec3f, t: f32, h: f32, rd: vec3f) -> vec3f { return applyFog(col, t, h, rd); }
fn outputTransform(col: vec3f) -> vec3f { return col; }
fn skyGrade(col: vec3f, rd: vec3f) -> vec3f { return col; }
fn bedTint(albedo: vec3f, p: vec3f, wl: f32) -> vec3f { return albedo; }
fn causticsAt(p: vec3f, wl: f32) -> f32 { return 1.0; }

const LAMBDA: f32 = ${f(MARCH.lambdaStep)};
const DMIN: f32 = ${f(MARCH.deltaMin)};
const GROWTH: f32 = ${f(MARCH.stepGrowth)};
const START_T: f32 = ${f(MARCH.startT)};
const BISECT: i32 = ${MARCH.bisectSteps};
const BISECT_TOL: f32 = ${f(MARCH.bisectTol)};
const CLAMP_SLOPE: f32 = ${f(MARCH.clampSlope)};
const CLAMP_SLOPE_FAR: f32 = ${f(MARCH.clampSlopeFar)};
const CLAMP_BASE: f32 = ${f(MARCH.clampBase)};
const GRAD_GATE: f32 = ${f(MARCH.gradGate)};
const GRAD_EPS_MIN: f32 = ${f(MARCH.gradEpsMin)};
const GRAD_CADENCE: i32 = ${MARCH.gradCadence};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let rd = cameraRay(fc.xy);
  let ro = u.camPos;

  if (ro.y < u.camWater) { return underwater(rd); }
  if (ro.y > MAXH && rd.y >= 0.0) { return display(skyColor(rd)); }

  let tMax = MAXDIST;
  var tWaterPlane = -1.0;
  if (rd.y < -1e-5 && ro.y > W_LEVEL) { tWaterPlane = (W_LEVEL - ro.y) / rd.y; }
  var waterChecked = false;

  var t = START_T;
  var tPrev = t;
  var gradSq = 0.0;
  var tWater = -1.0;
  var eWater = 0.0;
  var hit = false;
  var tHit = 0.0;

  for (var i = 0; i < 160; i++) {
    if (i >= u.marchSteps) { break; }
    let p = ro + rd * t;
    if (p.y > MAXH && rd.y > 0.0) { break; }

    let dp = max(t * u.footprint, 1e-4);
    let e = heightAt(p.xz, vec2f(0.0), dp);

    if (!waterChecked && tWaterPlane > 0.0 && t >= tWaterPlane) {
      waterChecked = true;
      let cw = ro.xz + rd.xz * tWaterPlane;
      let dpw = max(tWaterPlane * u.footprint, 1e-4);
      let ew = heightAt(cw, vec2f(0.0), dpw);
      if (ew < W_LEVEL - 0.25) {
        tWater = tWaterPlane;
        eWater = ew;
        break;
      }
    }

    if (p.y < e) { hit = true; break; }

    let clampT = t * CLAMP_SLOPE + CLAMP_BASE;
    let farAbove = (p.y - e) > clampT * GRAD_GATE;
    if ((i % GRAD_CADENCE) == 0) {
      if (!farAbove) {
        let eps = max(dp, GRAD_EPS_MIN);
        let ex = heightAt(p.xz, vec2f(eps, 0.0), dp);
        let ez = heightAt(p.xz, vec2f(0.0, eps), dp);
        let invEps = 1.0 / eps;
        let gx = (ex - e) * invEps;
        let gz = (ez - e) * invEps;
        gradSq = gx * gx + gz * gz;
      } else {
        gradSq = 0.0;
      }
    }

    tPrev = t;
    var stepLen = max(DMIN, (p.y - e) * inverseSqrt(gradSq + 1.0) * LAMBDA);
    stepLen *= 1.0 + t * GROWTH;
    var cap = clampT;
    if (farAbove) { cap = t * CLAMP_SLOPE_FAR + CLAMP_BASE; }
    stepLen = min(stepLen, cap);
    t += stepLen;

    if (t > tMax) {
      if (tMax >= MAXDIST || tPrev >= tMax) { break; }
      t = tMax;
    }
  }

  if (hit) {
    var lo = tPrev;
    var hi = t;
    for (var b = 0; b < BISECT; b++) {
      let tm = 0.5 * (lo + hi);
      let m = ro + rd * tm;
      let gap = m.y - heightAt(m.xz, vec2f(0.0), tm * u.footprint);
      if (gap < 0.0) { hi = tm; } else { lo = tm; }
      if ((hi - lo) < BISECT_TOL) { break; }
    }
    tHit = 0.5 * (lo + hi);
  }

  let terrainFront = hit && (tWater < 0.0 || tHit < tWater);
  let waterFront = !terrainFront && tWater > 0.0;
  if (terrainFront || waterFront) {
    var tS = tWater;
    if (terrainFront) { tS = tHit; }
    return display(shadeSurface(terrainFront, ro + rd * tS, rd, tS, eWater));
  }
  return display(shadeSky(ro, rd));
}
`;
}


/**
 * The classic sky in the raster renderer: its grade, a deeper blue overhead
 * than the march renderer's, height fog integrated from the camera, the dome
 * and the water's reflection of it.
 */
const CLASSIC_RASTER_SKY_WGSL = /* wgsl */ `
// Stylized grade (linear in, display-referred linear out): warm light and
// cooler, lifted shadows, a filmic shoulder (ACES fit, Narkowicz 2015) and a
// little more saturation.
const GRADE_EXPOSURE: f32 = 1.05;
fn outputTransform(col: vec3f) -> vec3f {
  var c = col * GRADE_EXPOSURE;
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c *= mix(vec3f(0.9, 0.97, 1.14), vec3f(1.04, 1.0, 0.94), smoothstep(0.015, 0.35, l));
  c += vec3f(0.004, 0.006, 0.011) * (1.0 - smoothstep(0.0, 0.08, l));
  c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  let l2 = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  return mix(vec3f(l2), c, 1.07);
}

// A clearer, deeper blue overhead than the march renderer's hazy sky.
fn skyGrade(col: vec3f, rd: vec3f) -> vec3f {
  return mix(col, col * vec3f(0.74, 0.9, 1.2), smoothstep(0.03, 0.55, rd.y));
}

// Height fog integrated along the view ray from the camera (the march
// renderer's applyFog takes the density at the surface point for the whole
// ray, which fogs a view from a sky island as if it were at sea level).
// The fog's optical depth over t metres along rd (surfaceFogOd); past FOG_OPAQUE the surface's own
// colour no longer shows, so a surface there need not be shaded at all.
const FOG_OPAQUE: f32 = 5.82;
fn fogOd(t: f32, rd: vec3f) -> f32 {
  let y0 = max(u.camPos.y, 0.0);
  let k = rd.y * t / FOG_H;
  var along = 1.0;
  if (abs(k) > 1e-3) { along = (1.0 - exp(-k)) / k; }
  return SIGMA_FOG * t * exp(-y0 / FOG_H) * along;
}
// In-scatter: the sky's colour toward rd without the sun's disc (the halo stays); looking up
// (at a sky island) the colour of the sky behind it rather than the horizon's haze.
fn fogInscatter(rd: vec3f) -> vec3f {
  let disc = SUN_COL * 6.0 * smoothstep(0.99955, 0.99985, max(dot(rd, SUN), 0.0));
  return mix((skyColor(rd) - disc + SKY_HOR) * 0.5, skyGrade(skyColor(rd) - disc, rd), smoothstep(0.06, 0.45, rd.y));
}
fn surfaceFog(col: vec3f, t: f32, h: f32, rd: vec3f) -> vec3f { return surfaceFogOd(col, t, h, rd, fogOd(t, rd)); }
fn surfaceFogOd(col: vec3f, t: f32, h: f32, rd: vec3f, od: f32) -> vec3f {
  if (od < 0.0005) { return worldHaze(cloudVeil(col, t, h, rd), t, rd); }
  let inscatter = fogInscatter(rd);
  if (od > FOG_OPAQUE) { return worldHaze(cloudVeil(inscatter, t, h, rd), t, rd); }
  let T = exp(-vec3f(0.58, 0.72, 0.95) * od);
  return worldHaze(cloudVeil(col * T + inscatter * (1.0 - T), t, h, rd), t, rd);
}

// The cloud deck (CLOUD_PLANE) at chunk-local xz, seen from tc metres: its coverage and thickness.
fn cloudDeck(xz: vec2f, tc: f32) -> vec2f {
  let cp = xz + u.origin + vec2f(u.time * 6.0, u.time * 2.0);
  let dpc = max(tc * u.footprint, 0.02);
  let c = fbmN(cp.x * F_CLOUD, cp.y * F_CLOUD, u.seed + 501, CLOUD_OCT, 2.0, 0.5, dpc, F_CLOUD, ASUM_CLOUD);
  return vec2f(smoothstep(0.42, 0.78, 0.5 + 0.5 * c), c);
}
// Its colour: as the classic dome draws it from below; its sunlit tops from above.
fn cloudColour(rd: vec3f, c: f32, above: bool) -> vec3f {
  if (above) { return vec3f(0.95) * (0.7 + 0.35 * max(SUN.y, 0.0)); }
  let shade = 0.85 + 0.30 * (1.0 - smoothstep(0.2, 0.9, c));
  return vec3f(shade) * (0.62 + 0.38 * max(SUN.y, 0.0));
}
// The deck between the camera and a surface at height h, t along rd (see the day–night sky's).
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

// The sky's gradient only: the sun's disc and halo come back as the glint.
fn skyReflect(r: vec3f) -> vec3f {
  return skyGrade(mix(SKY_HOR, SKY_ZEN, pow(max(r.y, 0.0), 0.55)), r);
}
${classicDomeWGSL('shadeDome')}
// Below the horizon lies land beyond the rings' reach: only its haze shows (see the day–night sky's).
fn shadeSky(ro: vec3f, rd: vec3f) -> vec3f {
  if (rd.y < 0.0) { return cloudVeil(fogInscatter(rd), 1e30, 0.0, rd); }
  return shadeDome(ro, rd);
}
`;

type RasterOptions = CachedOptions & {
  /** Half-precision colour arithmetic (the device has shader-f16): HALF_WGSL. */
  f16?: boolean;
  /** Mesh cells per half ring (even). */
  k: number;
  /** Rings (one per clipmap level, starting at level 0). */
  rings: number;
  grass?: GrassSpec | null;
  /** The sky realm (skyRealm.wgsl.ts), its tops cut along their outlines by clip distances (`clip`) or by discard. */
  islands?: boolean;
  clip?: boolean;
  ruins?: boolean;
  /** The cloud world's puffs (puffs.wgsl.ts). */
  puffs?: boolean;
  /** Trees (forest.wgsl.ts), with the measured bounds of their models. */
  forest?: { spec: ForestSpec; bounds: ReadonlyArray<{ top: number; radius: number }> } | null;
  /** Per-cell diagonals chosen from the heights (ringTris), else one fixed diagonal. */
  diag?: boolean;
  /** The classic fixed sky, or the day–night cycle (sky.wgsl.ts; bindings 17–19). */
  sky?: 'classic' | 'cycle';
  /** Combat (combat.wgsl.ts): the foes, the parts and the sprites. */
  combat?: boolean;
  /** Rocks (rocks.wgsl.ts): the scatter and the draws. */
  rocks?: boolean;
  /** Horses (horses.wgsl.ts) and colossi (colossus.wgsl.ts): skinned, textured, casting into the trees' shadow map. */
  horses?: boolean;
  /** The land's model isles (modelIsles.wgsl.ts). */
  modelIsles?: boolean;
};

/**
 * Adaptive ring triangulation (compute, one dispatch per ring whose cells
 * changed): each cell is split along the diagonal whose midpoint lies nearer
 * the true height at the cell's centre, read from the next finer clipmap
 * level (without one, ring 0 and the few outermost cells keep the higher
 * midpoint, which keeps ridge crests whole). Crests and valley floors that
 * run diagonally to the grid then follow triangle edges instead of zig-zagging.
 * A cell whose lower corner is odd on both axes becomes, once its ring has
 * fully morphed, the coarser ring's cell round it; it takes that cell's choice
 * (corners on the coarser level, centre = its own lower corner) so the morph
 * ends exactly on the coarser ring's triangles.
 */
/**
 * ringTris: a ring's index buffer, its cells block by block (RING_BLOCK² cells
 * a block, the blocks round the ring's centre by angle: rings.ts
 * ringBlockOrder) so the renderer can draw only the blocks in view; the
 * cells under the next finer ring left degenerate.
 */
function ringTrisWGSL(k: number): string {
  const { slot } = ringBlockOrder(k);
  return /* wgsl */ `
struct RingJob {
  ring: u32,
  hole: u32,     // 1: leave out the k² cells under the next finer ring
  holeOff: vec2i, // where they start, minus k/2
};
@group(1) @binding(14) var<uniform> ringJob: RingJob;
@group(1) @binding(15) var<storage, read_write> ringTrisOut: array<u32>;
// The ring's cells that hold water (the water's draw: vsWaterCell), and per ring its indirect draw
// arguments (vertex count, instances, first vertex, first instance; the count grows by 6 a cell).
@group(1) @binding(16) var<storage, read_write> waterCellsOut: array<u32>;
@group(1) @binding(17) var<storage, read_write> waterArgs: array<atomic<u32>>;
const RING_B: u32 = ${RING_BLOCK}u;
// Each block's slot in the buffer (row-major blocks).
const RING_SLOT = array<u32, ${slot.length}>(${Array.from(slot, (x) => `${x}u`).join(', ')});

fn antiDiagonal(h00: f32, h10: f32, h01: f32, h11: f32, hc: f32, haveCentre: bool) -> bool {
  let ma = 0.5 * (h00 + h11);
  let mb = 0.5 * (h10 + h01);
  if (haveCentre) { return abs(mb - hc) < abs(ma - hc); }
  return mb > ma;
}

@compute @workgroup_size(64)
fn ringTris(@builtin(global_invocation_id) id: vec3u) {
  let k = RING_SIDE / 2u;
  let side2 = 2u * k;
  let i = id.x;
  if (i >= side2 * side2) { return; }
  let ci = i % side2;
  let cj = i / side2;
  let nb = side2 / RING_B;
  let o = (RING_SLOT[(cj / RING_B) * nb + ci / RING_B] * RING_B * RING_B + (cj % RING_B) * RING_B + ci % RING_B) * 6u;
  let v00 = cj * RING_SIDE + ci;
  if (ringJob.hole != 0u) {
    let hx = k / 2u + u32(ringJob.holeOff.x);
    let hz = k / 2u + u32(ringJob.holeOff.y);
    if (ci >= hx && ci < hx + k && cj >= hz && cj < hz + k) {
      for (var q = 0u; q < 6u; q++) { ringTrisOut[o + q] = v00; }
      return;
    }
  }
  let r = ringJob.ring;
  let R = mesh.rings[r];
  let l = R.w;
  let st = R.z;
  let P = R.xy + vec2i(i32(ci), i32(cj)) * st;
  var anti: bool;
  if ((ci & cj & 1u) == 1u && f32(r) < mesh.morph.z) {
    let Q = P - vec2i(st);
    let s2 = 2 * st;
    anti = antiDiagonal(
      meshHeight(vec2f(Q), l + 1), meshHeight(vec2f(Q + vec2i(s2, 0)), l + 1),
      meshHeight(vec2f(Q + vec2i(0, s2)), l + 1), meshHeight(vec2f(Q + vec2i(s2, s2)), l + 1),
      meshHeight(vec2f(P), l), true);
  } else {
    var hc = 0.0;
    var haveCentre = false;
    if (l > 0) {
      let Pc = vec2f(P) + 0.5 * f32(st);
      let rel = Pc * mesh.cam.w - mesh.cam.xz;
      if (max(abs(rel.x), abs(rel.y)) * clip.cam.z <= exp2(f32(l - 1)) * 0.98) {
        hc = meshHeight(Pc, l - 1);
        haveCentre = true;
      }
    }
    anti = antiDiagonal(
      meshHeight(vec2f(P), l), meshHeight(vec2f(P + vec2i(st, 0)), l),
      meshHeight(vec2f(P + vec2i(0, st)), l), meshHeight(vec2f(P + vec2i(st, st)), l),
      hc, haveCentre);
  }
  let v10 = v00 + 1u;
  let v01 = v00 + RING_SIDE;
  let v11 = v01 + 1u;
  // Water here (some corner truly wet, and its water above another's ground, with a margin): onto the
  // ring's water list. (Dry corners' tables lie under their ground: a cliff with none wet holds none.)
  let w00 = meshGW(vec2f(P), l);
  let w10 = meshGW(vec2f(P + vec2i(st, 0)), l);
  let w01 = meshGW(vec2f(P + vec2i(0, st)), l);
  let w11 = meshGW(vec2f(P + vec2i(st, st)), l);
  // (on the coarse rings a cell needs two wet corners: a channel narrower than a cell would only clip
  // one, and stand as a sliver of steep water — a spike — on a slope)
  let nWet = select(0, 1, w00.y > w00.x + 0.02) + select(0, 1, w10.y > w10.x + 0.02) + select(0, 1, w01.y > w01.x + 0.02) + select(0, 1, w11.y > w11.x + 0.02);
  let cellM = f32(st) * mesh.cam.w; // (P and st are in level-0 lattice units)
  let anyWet = nWet >= select(1, 2, cellM > 4.0);
  if (anyWet && max(max(w00.y, w10.y), max(w01.y, w11.y)) > min(min(w00.x, w10.x), min(w01.x, w11.x)) - 0.6) {
    let n = atomicAdd(&waterArgs[ringJob.ring * 4u], 6u) / 6u;
    waterCellsOut[ringJob.ring * side2 * side2 + n] = (ringJob.ring << 20u) | i;
  }
  if (anti) {
    ringTrisOut[o] = v00; ringTrisOut[o + 1u] = v01; ringTrisOut[o + 2u] = v10;
    ringTrisOut[o + 3u] = v10; ringTrisOut[o + 4u] = v01; ringTrisOut[o + 5u] = v11;
  } else {
    ringTrisOut[o] = v00; ringTrisOut[o + 1u] = v11; ringTrisOut[o + 2u] = v10;
    ringTrisOut[o + 3u] = v00; ringTrisOut[o + 4u] = v01; ringTrisOut[o + 5u] = v11;
  }
}
`;
}

/**
 * Geometry-clipmap renderer: ring r is a (2k)² grid of cells at spacing
 * s_r · stride around the camera (ring 0 full, the others with the next finer
 * ring's square cut out by the index buffer). Near its outer edge a ring's odd
 * vertices slide onto the next ring's lattice and take its heights (CDLOD
 * morph), so neighbouring rings meet vertex for vertex. Vertex positions are
 * integers in level-0 lattice units relative to a per-frame reference point,
 * so shared vertices round identically in every ring (watertight).
 */
/**
 * The cloud world's warm haze (the reference paintings): a golden layer of air over the cloud sea, densest
 * low down (HAZE_H scale height over HAZE_BASE), glowing toward the sun, over every surface's fog and over
 * the sky alike; on the land (CLOUD_WORLD false) nothing.
 */
const WORLD_HAZE_WGSL = /* wgsl */ `
const HAZE_BASE: f32 = 1000.0;
const HAZE_H: f32 = 800.0;
const HAZE_SIGMA: f32 = ${f(1 / 4500)};
fn worldHaze(col: vec3f, t: f32, rd: vec3f) -> vec3f {
  if (!CLOUD_WORLD) { return col; }
  let a = (u.camPos.y - HAZE_BASE) / HAZE_H;
  let k = rd.y * t / HAZE_H;
  var depth = t * exp(-a);
  if (abs(k) > 1e-3) { depth = t * (exp(-a) - exp(-(a + k))) / k; }
  let T = exp(-HAZE_SIGMA * depth);
  let sun = pow(max(dot(rd, keyDir()), 0.0), 6.0);
  // (gold, as bright as the horizon's air: the hour's light, the paintings' hue)
  let haze = dot(skyHorCol(), vec3f(0.2126, 0.7152, 0.0722)) * vec3f(1.18, 0.95, 0.62) + keyCol(u.camPos.y) * vec3f(1.0, 0.8, 0.5) * (0.45 * sun);
  return mix(haze, col, T);
}
`;

export function rasterShader(opts: RasterOptions): string {
  const cycle = opts.sky === 'cycle';
  const islands = !!opts.islands;
  return /* wgsl */ `${islands && opts.clip ? 'enable clip_distances;' : ''}
${HALF_WGSL(!!opts.f16)}
${worldWGSL(opts.world ?? 'land')}
${FRAME_WGSL}
${heightSourceWGSL({ height: 'cached', ...opts, sun: cycle ? 'path' : 'classic' })}
${SHADE_WGSL}
${HERO_WGSL}
fn sunVisibility(p: vec3f, so: vec3f) -> f32 { return sunAt(p.xz) * treeShadowAt(p) * skyShadowAt(p); }

${cycle ? cycleSkyWGSL({ sky: 17, light: 18, view: 19 }) : CLASSIC_LIGHT_WGSL + CLASSIC_RASTER_SKY_WGSL}
${WORLD_HAZE_WGSL}
${opts.grass ? grassWGSL(opts.grass, islands) : NO_TINT_WGSL}
${opts.forest ? forestWGSL(opts.forest.spec, opts.forest.bounds, islands, !!opts.modelIsles && !islands) : 'fn treeShadowAt(p: vec3f) -> f32 { return 1.0; }'}
${opts.ruins ? RUIN_SITES_WGSL + RUINS_WGSL : 'fn inRuinSite(xz: vec2f) -> bool { return false; }'}
${opts.puffs ? PUFFS_WGSL : ''}
${islands ? skyRealmWGSL(!!opts.clip) + WATERFALLS_WGSL + structuresWGSL(!!opts.forest) + effectsWGSL() + faunaWGSL() + motesWGSL() + lanternsWGSL() : 'fn skyShadowAt(p: vec3f) -> f32 { return 1.0; }'}
${opts.combat ? combatWGSL() : ''}
${opts.rocks ? rocksWGSL(!!opts.forest && opts.forest.spec.shadowRes > 0) : ''}
${opts.horses ? horsesWGSL(!!opts.forest && opts.forest.spec.shadowRes > 0) + colossusWGSL(!!opts.forest && opts.forest.spec.shadowRes > 0) : ''}
${opts.modelIsles ? modelIslesWGSL({ forest: !!opts.forest, falls: !islands }) + (islands ? '' : WATERFALLS_WGSL) : ''}

struct Mesh {
  cam: vec4f,                        // camera rel. the reference lattice point: x, y (height), z; spacing0
  proj: vec4f,                       // f/aspect, f, near, unused
  morph: vec4f,                      // morph start (cells from the ring centre), 1/morph width, last ring, 1/reach (m)
  rings: array<vec4i, ${opts.rings}>,    // corner (level-0 units rel. reference) x, z; level-0 units per cell; clipmap level
  center: array<vec4f, ${opts.rings}>,   // ring centre (level-0 units rel. reference) x, z
  texOff: array<vec4f, ${opts.rings + 1}>, // per clipmap level: reference / 2^l mod n (x, z), 2^-l, unused
};
@group(0) @binding(4) var<uniform> mesh: Mesh;
const RING_SIDE: u32 = ${2 * opts.k + 1}u;
const CAMERA_NEAR: f32 = ${f(CAMERA.near)};

// The ground's biome layers (clipmapCache.ts, biomeShader): mat — the snow line (m), wetness (≥ 0) or
// lava (< 0), the meadow patches, flowers; bioA — soil (√rgb), grass cover; bioB — grass (√rgb), blade
// height / BLADE_SCALE; bioC — rock (√rgb), strata.
@group(0) @binding(28) var matTex: texture_2d_array<f32>;
@group(0) @binding(38) var bioATex: texture_2d_array<f32>;
@group(0) @binding(39) var bioBTex: texture_2d_array<f32>;
@group(0) @binding(40) var bioCTex: texture_2d_array<f32>;
${lookWGSL()}
${GROUND_SAMPLE_WGSL}
${GROUND_DETAIL_WGSL}
// Bench only: the ground's detail left out (the textures are not read: terrain.wgsl.ts shadeGround).
override STUB_DETAIL: bool = false;
// (diagnostic: the detail's code in the shader, its lookups never made: what having it costs, apart from running it)
override DETAIL_DORMANT: bool = false;
// The far forest's canopy layer (forest.wgsl.ts canopyAt): its albedo (√, premultiplied by its cover), its cover.
@group(0) @binding(25) var canopyTex: texture_2d_array<f32>;

// The trees are drawn out to TREE_REACH, the farthest shrinking away over its last tenth; the forest
// painted on the ground (shadeGround) comes in from CANOPY_NEAR, all of it by TREE_REACH, so the
// gaps the shrinking trees leave fill with canopy and none is left bare.
const TREE_REACH: f32 = ${f(TREE_REACH)};
const CANOPY_NEAR: f32 = ${f(0.75 * TREE_REACH)};
const CANOPY_ON: bool = ${opts.forest ? 'true' : 'false'};
// The canopy's light, as treeLight's for a leafy crown: the ramp's start (wrapped round the crown),
// light through the leaves against the sun, the share of the sky's light its depths get.
const CANOPY_WRAP: f32 = -0.35;
const CANOPY_BACK: f32 = 0.8;
const CANOPY_FILL: f32 = 0.7;
const CANOPY_FILL_SIDE: f32 = 0.5;
const CANOPY_SIDE: f32 = 0.5;

// The canopy layer at xz, from the level (or two, in a blend band) that holds it, as groundCacheAt.
fn canopyFetch(xz: vec2f) -> vec4f {
  let rel = xz - clip.cam.xy;
  let lf = log2(max(max(abs(rel.x), abs(rel.y)) * clip.cam.z, 1e-6));
  let lc = clamp(ceil(lf), 0.0, CLIP_LMAX);
  let l = i32(lc);
  let L = clip.lv[l];
  var c = textureSampleLevel(canopyTex, hSamp, (rel * L.z + L.xy) * CLIP_INV_N, l, 0.0);
  let b = clamp((lf - lc) / CLIP_BLEND + 1.0, 0.0, 1.0);
  if (b > 0.0 && lc < CLIP_LMAX) {
    let L1 = clip.lv[l + 1];
    c = mix(c, textureSampleLevel(canopyTex, hSamp, (rel * L1.z + L1.xy) * CLIP_INV_N, l + 1, 0.0), b);
  }
  return c;
}

const GRAD_NEAR: f32 = 60.0;
const GRAD_FAR: f32 = 120.0;

// The land ends where the outermost ring does, a square ±reach round the camera (131 km). Over the
// outer quarter of that the land and its water sink into the haze (this much more optical depth),
// so they are wholly haze, the colour the sky shows below the horizon past them, before they end.
// The fog alone has all but hidden them there: from the fly ceiling it leaves ~2 % of a peak.
fn reachOd(rel: vec2f) -> f32 {
  return 12.0 * smoothstep(0.75, 0.97, max(abs(rel.x), abs(rel.y)) * mesh.morph.w);
}

// Everything the ground's shading reads from the clipmap at xz, from the one level (or two, in a
// blend band) that holds it: height, the sun layer (light, gradient) and the material.
struct GroundCache { h: f32, wt: f32, sun: vec4f, mat: vec4f, bioA: vec4f, bioB: vec4f, bioC: vec4f };
fn groundCacheAt(xz: vec2f) -> GroundCache {
  let rel = xz - clip.cam.xy;
  let d = max(abs(rel.x), abs(rel.y));
  let lf = log2(max(d * clip.cam.z, 1e-6));
  let lc = clamp(ceil(lf), 0.0, CLIP_LMAX);
  let l = i32(lc);
  let L = clip.lv[l];
  let uv = (rel * L.z + L.xy) * CLIP_INV_N;
  let hw = textureSampleLevel(hTex, hSamp, uv, l, 0.0).rg;
  var g = GroundCache(hw.x, hw.y, textureSampleLevel(sunTex, hSamp, uv, l, 0.0), textureSampleLevel(matTex, hSamp, uv, l, 0.0),
    textureSampleLevel(bioATex, hSamp, uv, l, 0.0), textureSampleLevel(bioBTex, hSamp, uv, l, 0.0), textureSampleLevel(bioCTex, hSamp, uv, l, 0.0));
  let b = clamp((lf - lc) / CLIP_BLEND + 1.0, 0.0, 1.0);
  if (b > 0.0 && lc < CLIP_LMAX) {
    let L1 = clip.lv[l + 1];
    let uv1 = (rel * L1.z + L1.xy) * CLIP_INV_N;
    let hw1 = textureSampleLevel(hTex, hSamp, uv1, l + 1, 0.0).rg;
    g.h = mix(g.h, hw1.x, b);
    g.wt = mix(g.wt, hw1.y, b);
    g.sun = mix(g.sun, textureSampleLevel(sunTex, hSamp, uv1, l + 1, 0.0), b);
    g.mat = mix(g.mat, textureSampleLevel(matTex, hSamp, uv1, l + 1, 0.0), b);
    g.bioA = mix(g.bioA, textureSampleLevel(bioATex, hSamp, uv1, l + 1, 0.0), b);
    g.bioB = mix(g.bioB, textureSampleLevel(bioBTex, hSamp, uv1, l + 1, 0.0), b);
    g.bioC = mix(g.bioC, textureSampleLevel(bioCTex, hSamp, uv1, l + 1, 0.0), b);
  }
  return g;
}

// The ground's mat layer alone at xz (the snow line, wetness or lava, the patches, flowers).
fn matAt(xz: vec2f) -> vec4f {
  let rel = xz - clip.cam.xy;
  let lf = log2(max(max(abs(rel.x), abs(rel.y)) * clip.cam.z, 1e-6));
  let lc = clamp(ceil(lf), 0.0, CLIP_LMAX);
  let l = i32(lc);
  let L = clip.lv[l];
  var m = textureSampleLevel(matTex, hSamp, (rel * L.z + L.xy) * CLIP_INV_N, l, 0.0);
  let b = clamp((lf - lc) / CLIP_BLEND + 1.0, 0.0, 1.0);
  if (b > 0.0 && lc < CLIP_LMAX) {
    let L1 = clip.lv[l + 1];
    m = mix(m, textureSampleLevel(matTex, hSamp, (rel * L1.z + L1.xy) * CLIP_INV_N, l + 1, 0.0), b);
  }
  return m;
}
// Water freezes this far above the snow line (terrainMath.ts lastIce).
const ICE_ABOVE: f32 = ${f(CLIMATE.iceAbove)};

// ---- the ground's biome look, finished per pixel ------------------------------------------------
const SAND: vec3f = vec3f(0.6, 0.5, 0.34);
const SNOW: vec3f = vec3f(${SNOW_ALBEDO.map(f).join(', ')});
const LAVA_GLOW: vec3f = vec3f(5.0, 1.15, 0.14);

// Snow cover at p: above the biome's snow line (mat.x), broken into drifts, none on cliffs.
fn snowAt(gc: GroundCache, p: vec3f, slope: f32) -> f32 {
  let snowAlt = gc.mat.x;
  if (p.y < snowAlt - 40.0) { return 0.0; }
  let g = p.xz + u.origin;
  let drift = vnoise(g.x / 23.0, g.y / 23.0, 4451) * 18.0 + vnoise(g.x / 7.0, g.y / 7.0, 4452) * 5.0;
  return smoothstep(snowAlt - 15.0, snowAlt + 15.0, p.y + drift) * (1.0 - smoothstep(0.85, 1.35, slope));
}

// Sand on the sea's low, gentle shores, and by inland water where little grows (an oasis, a tundra
// lake's gravel); grass runs down to the rivers and lakes elsewhere.
fn beachAt(gc: GroundCache, p: vec3f, slope: f32) -> f32 {
  let above = p.y - gc.wt;
  if (above > 1.8) { return 0.0; }
  let seaShore = 1.0 - smoothstep(SEA + 0.5, SEA + 1.5, gc.wt);
  return (1.0 - smoothstep(0.3, 1.8, above)) * (1.0 - smoothstep(0.35, 0.7, slope)) * (1.0 - 0.75 * max(gc.mat.y, 0.0))
    * max(seaShore, 0.8 * (1.0 - gc.bioA.a));
}

struct GroundLook { albedo: vec3f, glow: vec3f, snow: f32, w: vec4f };
// The ground's colour at p from its biome layers (gc), its slope and its height above the water:
// grass over soil, rock on the steep (banded in strata in canyon country), sand on low gentle shores,
// mud by the water in wet lands, snow above the snow line, lava's glow on a volcano. wg: the grass's share
// of the ground that is not rock (grassDensityFrom, ragged by the turf's tufts where it thins); the
// shares of grass, soil, rock and sand in the colour come back as look.w (the snow's is the rest).
fn groundLook(gc: GroundCache, p: vec3f, slope: f32, wg: f32) -> GroundLook {
  let soil = gc.bioA.rgb * gc.bioA.rgb;
  let cover = gc.bioA.a;
  let grassCol = gc.bioB.rgb * gc.bioB.rgb;
  let rock = gc.bioC.rgb * gc.bioC.rgb;
  let strata = gc.bioC.a;
  let wetLava = gc.mat.y;
  let g = p.xz + u.origin;
  var col = mix(soil, grassCol * 0.8, wg);
  // rock: the steep, and outcrops where little grows
  let rockAmt = smoothstep(0.72, 1.15, slope + (1.0 - cover) * 0.22);
  if (rockAmt > 0.0) {
    var rk = rock;
    if (strata > 0.01) {
      // Strata: layers by height, warped a little, each its own tint (cream, rust, purple-grey).
      let hb = p.y / 7.5 + 0.6 * vnoise(g.x / 90.0, g.y / 90.0, 4441);
      let layer = floor(hb);
      let pick = fract(sin(layer * 12.9898) * 43758.547);
      var tint = vec3f(1.0, 0.86, 0.72);
      if (pick < 0.3) { tint = vec3f(1.3, 1.18, 1.0); } else if (pick > 0.75) { tint = vec3f(0.78, 0.66, 0.66); }
      let edge = smoothstep(0.42, 0.5, abs(fract(hb) - 0.5));
      rk = mix(rock, rock * tint * (1.0 - 0.25 * edge), strata);
    }
    col = mix(col, rk, rockAmt);
  }
  // the shore: sand (beachAt), damp and darker by the water, more so in wet lands
  let above = p.y - gc.wt;
  let wet = max(wetLava, 0.0);
  let beach = beachAt(gc, p, slope);
  col = mix(col, mix(SAND, soil, 0.3), beach * 0.85);
  col *= 1.0 - 0.4 * ((1.0 - smoothstep(0.0, 1.5, above)) * 0.55 + wet * 0.35);
  let snow = snowAt(gc, p, slope);
  col = mix(col, SNOW, snow);
  let sandW = beach * 0.85;
  let ground = (1.0 - snow) * (1.0 - sandW);
  let shares = vec4f(wg * (1.0 - rockAmt) * ground, (1.0 - wg) * (1.0 - rockAmt) * ground, rockAmt * ground, sandW * (1.0 - snow));
  var glow = vec3f(0.0);
  if (wetLava < -0.01) {
    // Lava: a dark crust broken over the glow, slowly drifting and pulsing.
    let lv = -wetLava;
    let crust = smoothstep(0.3, 0.75, 0.5 + 0.5 * vnoise(g.x / 3.5 + u.time * 0.05, g.y / 3.5, 4461));
    col = mix(col, vec3f(0.02, 0.016, 0.014), lv);
    glow = LAVA_GLOW * lv * (0.3 + 0.7 * (1.0 - crust)) * (0.85 + 0.15 * sin(u.time * 1.3 + g.x * 0.05));
  }
  return GroundLook(col, glow, snow, shares);
}

// Wind-packed snow near the camera: drifts (~12 m across) and sastrugi ridges (~2.5 m) lying across
// the wind, as a tilt of the normal n (the cover, amt), each faded out as it shrinks under the
// pixel's footprint dp. The snow is bright, so its relief is what shows its form.
fn snowRelief(n: vec3f, p: vec3f, dp: f32, amt: f32) -> vec3f {
  let kD = amt * (1.0 - smoothstep(1.5, 4.0, dp));
  if (kD <= 0.0) { return n; }
  let kS = amt * (1.0 - smoothstep(0.3, 0.8, dp));
  let g = p.xz + u.origin;
  let wd = vec2f(0.8, 0.6);
  let a = dot(g, wd);
  let b = dot(g, vec2f(-wd.y, wd.x));
  // h(a, b) = 1.1·N(a/12, b/26) + 0.16·N(a/2.5, b/7): its slopes along and across the wind
  let e = 0.25;
  let d0 = vnoise(a / 12.0, b / 26.0, 4471);
  var ha = 1.1 * kD * (vnoise((a + e) / 12.0, b / 26.0, 4471) - d0) / e;
  var hb = 1.1 * kD * (vnoise(a / 12.0, (b + e) / 26.0, 4471) - d0) / e;
  if (kS > 0.0) {
    let s0 = vnoise(a / 2.5, b / 7.0, 4472);
    ha += 0.16 * kS * (vnoise((a + e) / 2.5, b / 7.0, 4472) - s0) / e;
    hb += 0.16 * kS * (vnoise(a / 2.5, (b + e) / 7.0, 4472) - s0) / e;
  }
  let hx = ha * wd.x - hb * wd.y;
  let hz = ha * wd.y + hb * wd.x;
  return normalize(n - vec3f(hx, 0.0, hz) * n.y);
}

// The ground at p (its height from gc), seen at distance tS along rd: shadeSurface's terrain
// shading with its normal, sun and noise fields read from the clipmap instead of evaluated per
// pixel; the biome and the grass's tint, which turn sharply with height and slope, per pixel.
fn shadeGround(p: vec3f, rd: vec3f, tS: f32, gc: GroundCache, ddx: vec3f, ddy: vec3f) -> vec3f {
  // Behind fog too thick to see through, only the fog shows.
  let od = fogOd(tS, rd) + reachOd(p.xz - u.camPos.xz);
  if (od > FOG_OPAQUE && !STUB_FOG) { return surfaceFogOd(vec3f(0.0), tS, p.y, rd, od); }
  var n = vec3f(0.0, 1.0, 0.0);
  var grad = vec2f(0.0);
  if (!STUB_GRAD) {
    // Near the camera the cached surface's own slopes at the pixel's footprint (gradientAt, as
    // shadeSurface); out beyond GRAD_NEAR … GRAD_FAR the sun layer's gradient (central differences
    // over the lattice), which follows the land's smooth contours where the surface's facets would
    // draw its slopes, and the colours they choose, in texel steps.
    grad = gc.sun.zw;
    let far = select(smoothstep(GRAD_NEAR, GRAD_FAR, tS), 1.0, CACHED_GRAD);
    if (far < 1.0) {
      let dp = max(tS * u.footprint, 1e-4);
      grad = mix(gradientAt(p.xz, dp, dp), grad, far);
    }
    n = vec3f(-grad.x, 1.0, -grad.y) * inverseSqrt(dot(grad, grad) + 1.0);
  }
  var albedo = vec3f(0.1, 0.16, 0.05);
  var glow = vec3f(0.0);
  var cav = 1.0;
  if (CLOUD_WORLD) {
    // cloud: white, a faint mottle; deep in the creases and down the steep flanks grey-blue, the domes'
    // tops bright (no ground texture)
    let deep = (1.0 - smoothstep(900.0, 1250.0, p.y)) * 0.6 + (1.0 - smoothstep(0.25, 0.9, n.y)) * 0.4;
    albedo = CLOUD_ALBEDO * (0.95 + 0.05 * vnoise(p.x * 0.05, p.z * 0.05, 8105)) * mix(vec3f(1.04), vec3f(0.5, 0.55, 0.66), deep);
  } else if (!STUB_BIOME) {
    let slope = length(grad);
    // The detail (groundDetail), near the camera: the textures' mips have averaged it away past DETAIL_NEAR.
    var wg = clamp(grassDensityFrom(gc.bioA.a, slope, gc.mat.z) * 1.2, 0.0, 0.9);
    // (a lawn needs none past a few tens of metres: there the blades hide the ground, and the far lawn is the colour alone;
    // near, the ground shows between the tufts)
    let lawn = smoothstep(0.55, 0.85, wg) * (1.0 - smoothstep(0.35, 0.6, slope));
    let bare = 1.0 - lawn * smoothstep(4.0, 11.0, tS);
    let dk = select(0.0, (1.0 - smoothstep(DETAIL_NEAR, DETAIL_FAR, tS)) * bare, !STUB_DETAIL && !STUB_GRAD && !(DETAIL_DORMANT && u.footprint > 0.0));
    var turf = vec4f(0.5, 0.5, 0.5, 1.0);
    if (dk > 0.0 && wg > 0.01) {
      // Where the grass thins, the turf's tufts hold on and bare earth shows between them: a ragged edge.
      turf = gTop(GL_TURF, p.xz, ddx.xz, ddy.xz);
      wg = mix(wg, 0.9 * smoothstep(0.32, 0.68, wg / 0.9 + (turf.a - 0.62) * 0.85), dk);
    }
    let look = groundLook(gc, p, slope, wg);
    albedo = look.albedo;
    if (dk > 0.0) {
      let d = groundDetail(gc, p, n, slope, look.w, turf, ddx, ddy, dk, tS);
      let mossCol = gc.bioB.rgb * gc.bioB.rgb * 0.75;
      albedo = mix(albedo, mossCol, d.moss) * d.tone;
      albedo = mix(albedo, vec3f(dot(albedo, vec3f(0.3, 0.59, 0.11))) * 0.95, 0.6 * d.grey);
      n = d.n;
      cav = d.cav;
    }
    // Under a lawn the blades shade the ground (they are drawn out to a couple of hundred metres, thinning with
    // the ring): the ground between the tufts is not the blades' bright green.
    let shade = (1.0 - smoothstep(25.0, 110.0, tS)) * lawn * select(1.0, 0.0, STUB_DETAIL);
    albedo *= 1.0 - 0.3 * shade;
    albedo = bedTint(albedo, p, gc.wt);
    glow = look.glow;
    if (look.snow > 0.01 && !STUB_GRAD) { n = snowRelief(n, p, tS * u.footprint, look.snow); }
  }
  let L = keyDir();
  // (cloud wraps the light round its billows)
  let ndl = select(max(dot(n, L), 0.0), clamp((dot(n, L) + 0.05) / 1.05, 0.0, 1.0), CLOUD_WORLD);
  // Past the trees' reach the forest is painted on (canopyAt), lit as its trees are: it needs the
  // sun even where the ground faces away from it, its crowns' light wrapping round them.
  let canopyFar = CANOPY_ON && (tS > CANOPY_NEAR || CANOPY_MODE >= 2u) && CANOPY_MODE != 1u && !STUB_BIOME;
  var sh = 0.0;
  if (STUB_SHADOW) {
    sh = 1.0;
  } else if (ndl > 0.0 || (canopyFar && L.y > CANOPY_WRAP)) {
    sh = sunFrom(gc.sun) * treeShadowAt(p) * skyShadowAt(p);
    if (sh > 0.0) { sh = min(sh, heroShadowAt(p - u.camPos)); }
  }
  let kc = keyCol(p.y);
  let sunTerm = albedo * kc * ndl * sh * causticsAt(p, gc.wt) * mix(1.0, cav, 0.4);
  // (cloud: less fill, so its shaded flanks stay shaded under the exposure)
  let ambTerm = albedo * skyAmb(n, 0.22 + 0.28 * max(n.y, 0.0)) * heroAOAt(p - u.camPos) * cav * select(1.0, 0.55, CLOUD_WORLD);
  var col = sunTerm + ambTerm + glow;
  // The far forest over the ground (once the ground is lit, so little else is held meanwhile): its
  // canopy by its cover, broken into crowns while they are bigger than a pixel.
  if (canopyFar) {
    let cn = canopyFetch(p.xz);
    if (CANOPY_MODE == 3u) { return vec3f(cn.a, tS / 2000.0, 0.0); }
    var a = cn.a * select(smoothstep(CANOPY_NEAR, TREE_REACH, tS), 1.0, CANOPY_MODE == 2u);
    if (a > 0.004) {
      let k = 1.0 - smoothstep(2.5, 8.0, tS * u.footprint);
      var crowns = 0.5;
      if (k > 0.0) {
        let g = p.xz + u.origin;
        crowns = mix(0.5, 0.5 + 0.5 * vnoise(g.x / 8.0, g.y / 8.0, u.seed + 7311), k);
      }
      a = min(a * (0.55 + 0.9 * crowns), 1.0);
      // Seen aslant, the crowns hide the ground between them: the gaps left open from above, to the
      // power of 1 / the sine of the view's descent (a canopy's gap fraction).
      a = 1.0 - pow(1.0 - a, 1.0 / max(-rd.y, 0.2));
      // Lit as the trees are (forest.wgsl.ts treeLight): the light wrapped round a crown, through
      // its leaves against the sun, and the sky's into its depth. From above the crowns show their
      // tops; aslant more and more their flanks toward the camera, deeper in their own shade.
      let c = cn.rgb / cn.a;
      let up = vec3f(0.0, 1.0, 0.0);
      let vh = normalize(vec3f(-rd.x, 0.0, -rd.z) + vec3f(1e-5, 0.0, 0.0));
      let side = CANOPY_SIDE * (1.0 - smoothstep(0.1, 0.8, -rd.y));
      let ramp = mix(smoothstep(CANOPY_WRAP, 0.7, L.y), smoothstep(CANOPY_WRAP, 0.7, dot(vh, L)), side);
      let back = pow(clamp(dot(rd, L), 0.0, 1.0), 3.0) * CANOPY_BACK;
      let sky = mix(skyAmb(up, 0.54), skyAmb(vh, 0.3), side) * mix(CANOPY_FILL, CANOPY_FILL_SIDE, side);
      col = mix(col, c * c * (0.75 + 0.5 * crowns) * (kc * sh * (0.6 * ramp + 0.32 * back) + sky), a);
    }
  }
  if (!STUB_FOG) { col = surfaceFogOd(col, tS, p.y, rd, od); }
  return col;
}

fn meshHeight(P: vec2f, l: i32) -> f32 {
  let o = mesh.texOff[l];
  return textureSampleLevel(hTex, hSamp, (P * o.z + o.xy + 0.5) * CLIP_INV_N, l, 0.0).r;
}
// The ground and the water at a lattice point (the clipmap's two channels).
fn meshGW(P: vec2f, l: i32) -> vec2f {
  let o = mesh.texOff[l];
  return textureSampleLevel(hTex, hSamp, (P * o.z + o.xy + 0.5) * CLIP_INV_N, l, 0.0).rg;
}
// The water's surface at a lattice point (the clipmap's second channel).
fn meshWater(P: vec2f, l: i32) -> f32 {
  let o = mesh.texOff[l];
  return textureSampleLevel(hTex, hSamp, (P * o.z + o.xy + 0.5) * CLIP_INV_N, l, 0.0).g;
}

struct TOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,   // camera-relative world position
};

fn project(rel: vec3f) -> vec4f {
  let v = rel * u.basis; // (right, up, forward) components
  // reversed-Z, infinite far; the TAA jitter shifts the image by u.jitter (NDC)
  return vec4f(v.x * mesh.proj.x + u.jitter.x * v.z, v.y * mesh.proj.y + u.jitter.y * v.z, mesh.proj.z, v.z);
}

// The water mesh's height at a lattice point (ground, water): the water where it stands; where it is dry,
// flatTo (a water cell's lowest wet corner) held just under the ground, so the surface runs on level
// under a bank and the ground hides it there — never up the bank, a sloped sheet over a shore or a cliff.
// (flatTo ≥ 1e8: no cell known; the table there kept just under the ground.)
fn waterVertexH(gw: vec2f, flatTo: f32) -> f32 {
  if (gw.y > gw.x + 0.02) { return gw.y; }
  if (flatTo < 1e8) { return min(flatTo, gw.x - 0.05); }
  return max(gw.y, gw.x - 0.5);
}

// A ring's vertex: on the ground, or (water) on the water's surface there.
fn ringVertex(vi: u32, ring: u32, water: bool, flatTo: f32) -> TOut {
  let R = mesh.rings[ring];
  let ci = vec2i(i32(vi % RING_SIDE), i32(vi / RING_SIDE));
  let P = R.xy + ci * R.z;
  let C = mesh.center[ring].xy;
  let dCells = max(abs(f32(P.x) - C.x), abs(f32(P.y) - C.y)) / f32(R.z);
  var mu = clamp((dCells - mesh.morph.x) * mesh.morph.y, 0.0, 1.0);
  if (f32(ring) >= mesh.morph.z) { mu = 0.0; }
  let odd = vec2f(f32(ci.x & 1), f32(ci.y & 1));
  let Pm = vec2f(P) - odd * (f32(R.z) * mu);
  var h: f32;
  if (water) {
    // (where it is dry the table lies far under the ground; the mesh keeps it just under, so a shore's
    // triangles meet the land gently instead of plunging under the bank)
    h = waterVertexH(meshGW(Pm, R.w), flatTo);
    if (mu > 0.0) { h = mix(h, waterVertexH(meshGW(Pm, R.w + 1), flatTo), mu); }
  } else {
    h = meshHeight(Pm, R.w);
    if (mu > 0.0) { h = mix(h, meshHeight(Pm, R.w + 1), mu); }
  }
  let rel = vec3f(Pm.x * mesh.cam.w - mesh.cam.x, h - mesh.cam.y, Pm.y * mesh.cam.w - mesh.cam.z);
  var o: TOut;
  o.pos = project(rel);
  o.rel = rel;
  return o;
}

@vertex
fn vsTerrain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ring: u32) -> TOut {
  return ringVertex(vi, ring, false, 1e9);
}

${opts.diag ? ringTrisWGSL(opts.k) : ''}
@fragment
fn fsTerrain(i: TOut) -> @location(0) vec4f {
  // (the position's derivatives, for the ground textures' filtering: taken where the control flow is uniform)
  let ddx = dpdx(i.rel);
  let ddy = dpdy(i.rel);
  let ro = u.camPos;
  let tS = length(i.rel);
  let rd = i.rel / tS;
  if (ro.y < u.camWater) { return underwater(rd); }
  if (STUB_SHADE) { return vec4f(0.3, 0.45, 0.2, 1.0); }
  // Shade on the cached surface under the mesh point: the triangles cut
  // below it in concave spots, and a shadow ray starting there self-shadows.
  var p = ro + i.rel;
  if (PER_PIXEL_GROUND) {
    p.y = heightAt(p.xz, vec2f(0.0), tS * u.footprint);
    return display(shadeSurface(true, p, rd, tS, 0.0));
  }
  let gc = groundCacheAt(p.xz);
  p.y = gc.h;
  return display(shadeGround(p, rd, tS, gc, ddx, ddy));
}

// Water: the rings again, on the water's surface (the sea, lakes, rivers' pools; where the surface
// runs under the ground the ground hides it). Where the table steps the surface stands steep: a fall.
@vertex
fn vsWater(@builtin(vertex_index) vi: u32, @builtin(instance_index) ring: u32) -> TOut {
  return ringVertex(vi, ring, true, 1e9);
}
${opts.diag ? `
// With the rings' water lists (ringTris): only the cells that hold water, six vertices a cell, each
// entry its ring (high bits) and its cell.
@group(1) @binding(18) var<storage, read> waterCellsIn: array<u32>;
const WC_DX = array<u32, 6>(0u, 1u, 1u, 0u, 0u, 1u);
const WC_DZ = array<u32, 6>(0u, 1u, 0u, 0u, 1u, 1u);
@vertex
fn vsWaterCell(@builtin(vertex_index) vi: u32) -> TOut {
  let e = waterCellsIn[vi / 6u];
  let ring = e >> 20u;
  let c = e & 0xfffffu;
  let side2 = RING_SIDE - 1u;
  let k = vi % 6u;
  // the cell's lowest wet corner: its dry corners are laid level with it
  let R = mesh.rings[ring];
  let P = R.xy + vec2i(i32(c % side2), i32(c / side2)) * R.z;
  var wetMin = 1e9;
  for (var q = 0u; q < 4u; q++) {
    let gw = meshGW(vec2f(P + vec2i(i32(q & 1u), i32(q >> 1u)) * R.z), R.w);
    if (gw.y > gw.x + 0.02) { wetMin = min(wetMin, gw.y); }
  }
  return ringVertex((c / side2 + WC_DZ[k]) * RING_SIDE + c % side2 + WC_DX[k], ring, true, wetMin);
}
` : ''}
// ---- stylized water (drawn blended over the lake bed, premultiplied alpha) ----
// Lake bed under the surface: sand toward the water, caustics in the shallows.
fn bedTint(albedo: vec3f, p: vec3f, wl: f32) -> vec3f {
  let d = wl - p.y;
  if (d <= 0.0) { return albedo; }
  return mix(albedo, vec3f(0.2, 0.18, 0.12), smoothstep(0.0, 0.6, d) * 0.7);
}
fn causticsAt(p: vec3f, wl: f32) -> f32 {
  let d = wl - p.y;
  if (d <= 0.0) { return 1.0; }
  let g = p.xz + u.origin;
  let t = u.time;
  let a = vnoise(g.x * 0.55 + t * 0.4, g.y * 0.55 - t * 0.3, 931);
  let b = vnoise(g.x * 0.55 - t * 0.35 + 3.1, g.y * 0.55 + t * 0.25, 937);
  let ridge = 1.0 - abs(a - b); // bright where the two drifting fields cross
  let r2 = ridge * ridge;
  let caus = r2 * r2 * r2 * 2.2;
  return mix(1.0, 0.72 + caus, exp(-d * 0.35) * smoothstep(0.0, 0.3, d));
}

fn waterHeight(g: vec2f, t: f32) -> f32 {
  return 0.5 * vnoise(g.x / 2.4 + t * 0.45, g.y / 2.4 - t * 0.3, 911)
       + 0.3 * vnoise(g.x / 1.05 - t * 0.7, g.y / 1.05 + t * 0.55, 913)
       + 0.15 * vnoise(g.x / 0.45 + t * 1.2, g.y / 0.45 - t * 0.9, 917);
}
fn waterNormal(g: vec2f, t: f32, strength: f32) -> vec3f {
  let e = 0.08;
  let h0 = waterHeight(g, t);
  let hx = waterHeight(g + vec2f(e, 0.0), t);
  let hz = waterHeight(g + vec2f(0.0, e), t);
  return normalize(vec3f(-(hx - h0) / e * strength, 1.0, -(hz - h0) / e * strength));
}
// The ripples' slope on flowing water (a river's flow, m/s): the ripple field carried downstream
// in two phases half a cycle apart, each faded out as it restarts (flow maps, Vlachos 2010), and
// stretched along the current. Still water keeps its own ripples.
fn flowRippleSlope(g: vec2f, flow: vec2f, t: f32) -> vec2f {
  let sp = length(flow);
  let dir = flow / max(sp, 1e-4);
  let side = vec2f(-dir.y, dir.x);
  // ripples in a frame stretched along the current (streaks), more so as it runs faster
  let stretch = 1.0 + 1.6 * smoothstep(0.3, 2.5, sp);
  let PERIOD = 1.6;
  let ph0 = fract(t / PERIOD);
  let ph1 = fract(t / PERIOD + 0.5);
  let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
  var grad = vec2f(0.0);
  for (var k = 0; k < 2; k++) {
    let ph = select(ph1, ph0, k == 0);
    let wk = select(1.0 - w0, w0, k == 0);
    let q0 = g - flow * (ph * PERIOD) + select(vec2f(13.7, -5.3), vec2f(0.0), k == 0);
    let q = vec2f(dot(q0, dir) / stretch, dot(q0, side));
    let e = 0.08;
    let h0 = waterHeight(q, t * 0.35);
    let hu = waterHeight(q + vec2f(e, 0.0), t * 0.35);
    let hv = waterHeight(q + vec2f(0.0, e), t * 0.35);
    grad += wk * ((hu - h0) / (e * stretch) * dir + (hv - h0) / e * side);
  }
  return grad;
}
// Molten lava's surface at world g (flow: its creep, m/s): rafts of dark crust broken into plates that
// drift with the flow and slowly turn over, molten rock glowing through the cracks and in the open
// channels between the rafts. Returns (crust shade, crack glow, crust cover).
fn lavaSurface(g: vec2f, flow: vec2f, t: f32) -> vec3f {
  let PERIOD = 9.0;
  let ph0 = fract(t / PERIOD);
  let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
  let creep = flow * 0.35 + vec2f(0.05, 0.03);
  var crack = 0.0;
  var swirl = 0.0;
  for (var k = 0; k < 2; k++) {
    let ph = select(fract(t / PERIOD + 0.5), ph0, k == 0);
    let wk = select(1.0 - w0, w0, k == 0);
    let q = g - creep * (ph * PERIOD) + select(vec2f(7.1, 3.9), vec2f(0.0), k == 0);
    // plates ~4 m, cracks between them; finer cracks inside the plates
    let warp = vec2f(vnoise(q.x / 6.0, q.y / 6.0, 971), vnoise(q.x / 6.0 + 4.1, q.y / 6.0 - 2.7, 972));
    let e1 = voronoiEdge((q + 1.6 * warp) / 4.2, 973);
    let e2 = voronoiEdge((q + 0.8 * warp) / 1.3, 974);
    crack += wk * max(1.0 - smoothstep(0.02, 0.16, e1), 0.55 * (1.0 - smoothstep(0.01, 0.08, e2)));
    swirl += wk * (0.5 + 0.5 * vnoise(q.x / 3.0 + t * 0.07, q.y / 3.0, 975));
  }
  // the rafts: a slow ~25 m field of crust, open molten channels between
  let raft = vnoise(g.x / 26.0 - creep.x * t * 0.3, g.y / 26.0 - creep.y * t * 0.3, 976) + 0.35 * vnoise(g.x / 9.0, g.y / 9.0 + t * 0.02, 977);
  return vec3f(0.55 + 0.45 * swirl, crack, smoothstep(-0.25, 0.2, raft));
}

@fragment
fn fsWater(i: TOut) -> @location(0) vec4f {
  // The surface's own slope, from the mesh (before any early return: derivatives need the whole
  // quad): level on a pool, a lake, the sea; steep where the water table steps — a fall.
  let dX = dpdx(i.rel);
  let dY = dpdy(i.rel);
  let ro = u.camPos;
  let tS = length(i.rel);
  let rd = i.rel / tS;
  if (ro.y < u.camWater) { return underwater(rd); }
  var nGeo = normalize(cross(dX, dY));
  if (nGeo.y < 0.0) { nGeo = -nGeo; }
  let fall = 1.0 - smoothstep(0.55, 0.92, nGeo.y);
  let p = ro + i.rel;
  let wy = p.y;
  // Behind fog too thick to see through: the fog alone, laid on opaquely (as the full path would end).
  let od = fogOd(tS, rd) + reachOd(i.rel.xz);
  if (od > FOG_OPAQUE) { return display(surfaceFogOd(vec3f(0.0), tS, wy, rd, od)); }
  let g = p.xz + u.origin;
  let depthV = max(wy - heightAt(p.xz, vec2f(0.0), 0.0), 0.0);
  // Ripples: calmer at the shore, faded out with distance (no sub-pixel noise; none past 300 m,
  // where the water's normal is its plane's and its noise is not evaluated).
  let strength = mix(0.03, 0.14, smoothstep(0.0, 1.5, depthV)) * (1.0 - smoothstep(40.0, 300.0, tS)) * (1.0 - fall);
  let flow = flowAt(p.xz);
  let speed = length(flow);
  let lava = clamp(-matAt(p.xz).y, 0.0, 1.0);
  var n = vec3f(0.0, 1.0, 0.0);
  var rapids = 0.0;
  if (strength > 0.0 && lava < 0.5) {
    if (speed > 0.08) {
      // flowing: the ripples ride the current, rougher and foaming where it runs fast
      let sl = flowRippleSlope(g, flow, u.time) * strength * (1.0 + 0.8 * smoothstep(0.8, 3.0, speed));
      n = normalize(vec3f(-sl.x, 1.0, -sl.y));
      let fz = 0.5 + 0.5 * vnoise(dot(g - flow * u.time, normalize(flow)) / 1.6, dot(g, vec2f(-flow.y, flow.x) / speed) / 0.7, 981);
      rapids = smoothstep(1.4, 3.2, speed) * smoothstep(0.55, 0.8, fz);
    } else {
      n = waterNormal(g, u.time, strength);
    }
  }
  let sh = sunVisibility(p, vec3f(0.0, 0.15, 0.0));
  // Light through the water column under this point, per channel (red dies first): clear over a
  // shallow bed, blue-green over a deep one (a third of the light reaches a bed 8 m down).
  let path = depthV / max(-rd.y, 0.06);
  let T = exp(-vec3f(0.5, 0.14, 0.09) * path);
  let Tl = dot(T, vec3f(0.2126, 0.7152, 0.0722));
  let kc = keyCol(wy);
  let up = vec3f(0.0, 1.0, 0.0);
  let body = vec3f(0.006, 0.065, 0.12) * (kc * 0.35 * (0.4 + 0.6 * sh) + skyAmb(up, 0.8));
  let refl = reflect(rd, n);
  let F = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
  let sd = max(dot(refl, keyDir()), 0.0);
  let glint = pow(sd, 900.0) * 14.0 + pow(sd, 120.0) * 0.3;
  // Shoreline foam: bands running in, and a line where water meets land (none past 0.55 m deep).
  var foam = 0.0;
  if (depthV < 0.55) {
    let fnz = vnoise(g.x / 2.3 + u.time * 0.15, g.y / 2.3, 941);
    let bands = smoothstep(0.55, 0.85, sin(depthV * 14.0 - u.time * 2.4 + fnz * 3.0) * 0.5 + 0.5) * (1.0 - smoothstep(0.05, 0.55, depthV));
    let edge = 1.0 - smoothstep(0.0, 0.07, depthV);
    foam = clamp(max(bands * 0.85, edge) * (0.6 + 0.4 * fnz), 0.0, 1.0);
  }
  let foamCol = vec3f(0.85, 0.88, 0.9) * (kc * 0.32 * (0.35 + 0.65 * sh) + skyAmb(up, 0.55));
  // Premultiplied: sky reflection and in-scatter over the transmitted bed.
  var col = skyReflect(refl) * F + body * (vec3f(1.0) - T) * (1.0 - F) + kc * glint * sh;
  var alpha = 1.0 - (1.0 - F) * Tl;
  foam = max(foam, rapids * (1.0 - smoothstep(60.0, 220.0, tS)));
  col = mix(col, foamCol, foam);
  alpha = mix(alpha, 1.0, foam);
  // Molten lava: opaque, its crust lit as ground, its cracks glowing (and the glow lights the crust).
  if (lava > 0.01) {
    let ls = lavaSurface(g, flow, u.time);
    let pulse = 0.85 + 0.15 * sin(u.time * 1.1 + g.x * 0.07 + g.y * 0.05);
    let crustAlb = vec3f(0.03, 0.02, 0.017) * ls.x;
    let heat = ls.y * pulse;
    // the crusted rafts: dark rock lit as ground, faint warmth, the cracks glowing
    let crusted = crustAlb * (kc * max(keyDir().y, 0.0) * sh + skyAmb(up, 0.5)) + crustAlb * LAVA_GLOW * 0.25
      + LAVA_GLOW * (0.02 + 1.1 * heat) * mix(vec3f(1.0), vec3f(1.0, 1.25, 1.9), heat * heat);
    // the open melt: bright, hotter (yellower) where it churns
    let molten = LAVA_GLOW * (0.3 + 0.28 * ls.x) * mix(vec3f(1.0), vec3f(1.0, 1.2, 1.8), 0.3 * ls.x);
    let lavaCol = mix(molten, crusted, ls.z);
    col = mix(col, lavaCol, lava);
    alpha = mix(alpha, 1.0, lava);
  }
  // Ice where the water is frozen (high above the snow line: tundra lakes, the cold sea, high tarns):
  // pale, cracked, dusted with snow, opaque, lit as ground.
  let ice = smoothstep(ICE_ABOVE - 10.0, ICE_ABOVE + 10.0, wy - matAt(p.xz).x) * (1.0 - fall) * (1.0 - lava);
  if (ice > 0.01) {
    let cr = smoothstep(0.93, 0.985, 1.0 - abs(vnoise(g.x / 9.0, g.y / 9.0, 961))) + 0.6 * smoothstep(0.95, 0.99, 1.0 - abs(vnoise(g.x / 3.1, g.y / 3.1, 962)));
    let dust = smoothstep(0.45, 0.85, 0.5 + 0.5 * vnoise(g.x / 23.0, g.y / 23.0, 963));
    let iceAlb = mix(vec3f(0.16, 0.27, 0.34), SNOW, dust) * (1.0 - 0.35 * min(cr, 1.0) * (1.0 - dust));
    // clear ice is glossy: it keeps some of the water's reflection of the sky
    let iceCol = iceAlb * (kc * max(keyDir().y, 0.0) * sh + skyAmb(up, 0.55)) + skyReflect(refl) * F * 0.6 * (1.0 - dust)
      + kc * glint * sh * 0.25;
    col = mix(col, iceCol, ice);
    alpha = mix(alpha, 1.0, ice);
  }
  // A fall: white water pouring down the step, streaked, all but opaque (lava keeps its own look).
  if (fall > 0.01 && lava < 0.5) {
    let down = normalize(nGeo.xz + vec2f(1e-5, 0.0));
    let a = dot(g, vec2f(-down.y, down.x));
    let v = wy + u.time * 7.0;
    let streak = 0.6 * vnoise(a * 0.9, v * 0.12, 951) + 0.4 * vnoise(a * 2.3, v * 0.35, 953);
    let white = smoothstep(-0.3, 0.5, streak);
    let fallCol = mix(vec3f(0.42, 0.56, 0.6), vec3f(0.93, 0.96, 0.98), white) * (kc * 0.45 * (0.4 + 0.6 * sh) + skyAmb(nGeo, 0.7));
    col = mix(col, fallCol, fall);
    alpha = mix(alpha, 0.86 + 0.12 * white, fall);
  }
  let a = max(alpha, 1e-3);
  let shown = display(surfaceFogOd(col / a, tS, wy, rd, od));
  return vec4f(shown.rgb * alpha, alpha);
}

// Hero: sphere-traced per pixel inside the scissor rectangle of its bounding
// sphere; writes its own depth so terrain in front hides it.
struct HeroOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fsHero(@builtin(position) fc: vec4f) -> HeroOut {
  if (hero.bound.w <= 0.0 || u.camPos.y < u.camWater) { discard; }
  let rd = cameraRay(fc.xy);
  let hit = heroTrace(rd);
  let tH = hit.x;
  if (tH < 0.0) { discard; }
  let pr = rd * tH;
  let n = heroNormal(pr, tH);
  var sh = 0.0;
  if (dot(n, keyDir()) > 0.0) {
    sh = softShadow(u.camPos + pr + n * 0.05, u.footprint) * treeShadowAt(u.camPos + pr) * skyShadowAt(u.camPos + pr);
    if (sh > 0.0) { sh = min(sh, heroShadow(pr + n * 0.03)); }
  }
  var o: HeroOut;
  o.color = display(heroShade(rd, tH, hit.y, n, sh));
  o.depth = CAMERA_NEAR / dot(pr, u.basis[2]);
  return o;
}

// Sky: a fullscreen triangle on the far plane (depth 0 under reversed Z).
@vertex
fn vsSky(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fsSky(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let rd = cameraRay(fc.xy);
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  return display(worldHaze(shadeSky(u.camPos, rd), 1e6, rd));
}
`;
}

/**
 * Clipmap generator: one invocation per lattice point of an update rectangle.
 * The point's chunk-local position is exact (integer × power-of-two spacing)
 * and E is evaluated on the same fp32 global the renderer would form.
 */
export function genShader(n: number, world: World = 'land'): string {
  return /* wgsl */ `
${worldWGSL(world)}
struct Gen {
  origin: vec2f,   // chunk origin
  rel0: vec2i,     // first lattice point of the rectangle, relative to the origin
  store0: vec2u,   // its toroidal texel
  size: vec2u,     // rectangle size
  spacing: f32,
  dp: f32,         // footprint E is prefiltered for
  level: u32,
  seed: i32,
};
@group(0) @binding(0) var<uniform> gen: Gen;
@group(0) @binding(1) var hOut: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= gen.size)) { return; }
  let local = vec2f(gen.rel0 + vec2i(id.xy)) * gen.spacing;
  // The ground's height and the water's surface (the sea, a lake, a river's pool) at the lattice point.
  let s = landSurface(local + gen.origin, gen.dp, gen.seed);
  let st = (gen.store0 + id.xy) % vec2u(${n}u);
  textureStore(hOut, st, gen.level, vec4f(s.e, s.w, s.flow));
}
`;
}

/**
 * Cached sun visibility: for each lattice point of an update rectangle (after
 * its heights were generated), a soft-shadow march toward the sun over the
 * clipmap, written to the sun layer. Shading then samples it (sunAt) instead
 * of marching per pixel. Occlusion only: the surface's own N·L is applied by
 * the shading.
 */
export function sunShader(opts: CachedOptions): string {
  return /* wgsl */ `
${worldWGSL(opts.world ?? 'land')}
struct Gen {
  origin: vec2f,
  rel0: vec2i,
  store0: vec2u,
  size: vec2u,
  spacing: f32,
  dp: f32,
  level: u32,
  seed: i32,
};
@group(0) @binding(0) var<uniform> gen: Gen;
${heightSourceWGSL({ height: 'cached', ...opts, sun: undefined })}
@group(0) @binding(4) var sunOut: texture_storage_2d_array<rgba16float, write>;
${CANOPY_LAYER_WGSL}
const SH_K: f32 = ${f(SHADOW.penumbraK)};
const SH_CUTOFF: f32 = ${f(SHADOW.cutoff)};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= gen.size)) { return; }
  let s = gen.spacing;
  let xz = vec2f(gen.rel0 + vec2i(id.xy)) * s;
  let e = heightAt(xz, vec2f(0.0), s);
  let gx = (heightAt(xz, vec2f(s, 0.0), s) - heightAt(xz, vec2f(-s, 0.0), s)) / (2.0 * s);
  let gz = (heightAt(xz, vec2f(0.0, s), s) - heightAt(xz, vec2f(0.0, -s), s)) / (2.0 * s);
  let n = normalize(vec3f(-gx, 1.0, -gz));
  let st = (gen.store0 + id.xy) % vec2u(${opts.n}u);
  writeCanopy(st, xz, e, vec2f(gx, gz), s);
  let p = vec3f(xz.x, e, xz.y) + n * (0.15 + 0.5 * s);
  var res = 1.0;
  var t = max(0.6, 1.5 * s);
  for (var i = 0; i < 48; i++) {
    let q = p + SUN * t;
    if (q.y > MAXH) { break; }
    let h = q.y - heightAt(q.xz, vec2f(0.0), 0.0);
    if (h <= 0.0) { res = 0.0; break; }
    res = min(res, SH_K * h / t);
    if (res < SH_CUTOFF) { res = 0.0; break; }
    t += clamp(h * 0.8, 0.5, 80.0) + t * 0.05;
  }
  // The visibility, and the ground's gradient (the shading's normal) beside it.
  textureStore(sunOut, st, gen.level, vec4f(clamp(res, 0.0, 1.0), 0.0, gx, gz));
}
`;
}

/**
 * The ground's biome layers (clipmapCache.ts): for each lattice point of an update rectangle, the
 * biome regions there (worldgen.wgsl.ts landRegions) and their looks blended (lookWGSL), the snow
 * line from the sea-level temperature, the volcano's lava (its crater and the flows down its gullies),
 * and the meadow patches; written to the mat and bioA/B/C layers (see lookWGSL for the encoding).
 */
export function biomeShader(opts: CachedOptions): string {
  return /* wgsl */ `
${worldWGSL(opts.world ?? 'land')}
struct Gen {
  origin: vec2f,
  rel0: vec2i,
  store0: vec2u,
  size: vec2u,
  spacing: f32,
  dp: f32,
  level: u32,
  seed: i32,
};
@group(0) @binding(0) var<uniform> gen: Gen;
${heightSourceWGSL({ height: 'cached', ...opts, sun: undefined })}
@group(0) @binding(4) var matOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var bioAOut: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(6) var bioBOut: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(7) var bioCOut: texture_storage_2d_array<rgba8unorm, write>;
${lookWGSL()}
${GROUND_COVER_WGSL}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= gen.size)) { return; }
  let s = gen.spacing;
  let xz = vec2f(gen.rel0 + vec2i(id.xy)) * s;
  let g = xz + gen.origin;
  let st = (gen.store0 + id.xy) % vec2u(${opts.n}u);
  let r = landRegions(g, domainWarp(g.x, g.y, gen.seed, 0.0), gen.seed);
  var soil = vec3f(0.0);
  var grass = vec3f(0.0);
  var dry = vec3f(0.0);
  var rock = vec3f(0.0);
  var a = vec4f(0.0);
  var b = vec2f(0.0);
  var gUse = vec4f(0.0);
  for (var k = 0; k < ${BIOME_COUNT}; k++) {
    let w = r.w[k];
    if (w <= 0.0) { continue; }
    soil += w * LK_SOIL[k];
    grass += w * LK_GRASS[k];
    dry += w * LK_DRY[k];
    rock += w * LK_ROCK[k];
    a += w * LK_A[k];
    b += w * LK_B[k];
    gUse += w * LK_USE[k];
  }
  // The rock's colour in patches of tens of metres, warmer and cooler, lighter and darker: a cliff of one flat tone is a painted wall.
  let rockV = vnoise(g.x / 41.0, g.y / 41.0, gen.seed + 7811);
  let rockW = vnoise(g.x / 130.0 + 3.1, g.y / 130.0 - 1.7, gen.seed + 7812);
  rock = rock * (1.0 + 0.16 * rockV) * vec3f(1.0 + 0.07 * rockW, 1.0, 1.0 - 0.09 * rockW);
  // Roads, bare patches, long and short grass (groundCover.wgsl.ts).
  let gu = groundUse(g, s, gen.seed, gUse);
  // Meadow patches (density, 22 m) and the grass's hue patches (30 m): lush to dry, drier in dry air.
  let patchN = 0.5 + 0.5 * vnoise(g.x * 0.045, g.y * 0.045, gen.seed + 7777);
  let patchHue = 0.5 + 0.5 * vnoise(g.x * 0.033, g.y * 0.033, gen.seed + 7707);
  let dryness = clamp(patchHue * 1.3 - 0.45 + (0.5 - r.M) * 0.8 + gu.z, 0.0, 1.0);
  let grassCol = mix(grass, dry, dryness * 0.75);
  // The snow line (m): where the air is SNOW_T, lowered by the biome's frost.
  let snowAlt = (r.T0 - SNOW_T - b.x) / L_ALT;
  // Lava: the liquid of a volcano's crater lake and of the rivers high on its cone (terrainMath.ts lastLava).
  var lava = 0.0;
  if (r.vol.z > 0.0 && waterAt(xz) > heightAt(xz, vec2f(0.0), 0.0) - 0.01) {
    let w = domainWarp(g.x, g.y, gen.seed, 0.0);
    lava = 1.0 - smoothstep(0.45, 0.6, length(w - r.vol.xy) / (${f(LANDS.volcano.cone)} * r.vol.z));
  }
  let wl = select(a.w, -lava, lava > 0.01);
  textureStore(matOut, st, gen.level, vec4f(snowAlt, wl, patchN, b.y));
  // (trodden soil is paler and drier; a road or a bare patch is cover 0, the grass round it thin and short)
  textureStore(bioAOut, st, gen.level, vec4f(sqrt(soil * mix(vec3f(1.0), vec3f(1.12, 1.07, 0.96), gu.w)), a.x * gu.x));
  textureStore(bioBOut, st, gen.level, vec4f(sqrt(grassCol), clamp(a.y * gu.y / BLADE_SCALE, 0.0, 1.0)));
  textureStore(bioCOut, st, gen.level, vec4f(sqrt(rock), a.z));
}
`;
}

/**
 * Cached horizons along the sun's path (the day–night sky). For each lattice
 * point of an update rectangle, the highest angle on the sun's path toward its
 * rising (r) and toward its setting (g) at which the land still stands in
 * front of the sky; a light on the path at angle θ (the sun, or the moon on
 * the same circle) reaches the point when r < θ < g (sunAt, softened). The
 * march runs in the path's own plane through the point, tilted like the path:
 * at distances growing ~9 % a step out to 8 km (64 lattice spacings on the
 * coarse levels that shade the distant land: 65 km on the coarsest, for the
 * long shadows of evening) or the edge of the coarsest level, where the land
 * rises above the current horizon it lifts the
 * horizon to the land's top there. The land falls away with distance over the
 * small planet of celestial.ts, so from high ground the horizon lies below
 * the horizontal and peaks keep the sun after the valleys lose it; beyond the
 * march the land is taken to lie at sea level.
 */
export function horizonShader(opts: CachedOptions): string {
  const cosLat = Math.cos(SKY.latitude), sinLat = Math.sin(SKY.latitude);
  return /* wgsl */ `
${worldWGSL(opts.world ?? 'land')}
struct Gen {
  origin: vec2f,
  rel0: vec2i,
  store0: vec2u,
  size: vec2u,
  spacing: f32,
  dp: f32,
  level: u32,
  seed: i32,
};
@group(0) @binding(0) var<uniform> gen: Gen;
${heightSourceWGSL({ height: 'cached', ...opts, sun: undefined })}
@group(0) @binding(4) var horizonOut: texture_storage_2d_array<rgba16float, write>;
${CANOPY_LAYER_WGSL}
const PATH_UP: vec3f = vec3f(${f(PATH_N[0])}, ${f(PATH_N[1])}, ${f(PATH_N[2])});
const COS_LAT: f32 = ${f(cosLat)};
const SIN_LAT: f32 = ${f(sinLat)};
const CURV: f32 = ${f(1 / (2 * SKY.planetRadius))};
const REACH: f32 = 8000.0;

// The land's height for the march: the finest level holding the point, without
// heightAt's blend into the next level near its edge (half the fetches; the
// horizon only needs the land's top, not a seamless surface).
fn marchHeight(xz: vec2f) -> f32 {
  let rel = xz - clip.cam.xy;
  let lf = log2(max(max(abs(rel.x), abs(rel.y)) * clip.cam.z, 1e-6));
  return clipSample(hTex, rel, i32(clamp(ceil(lf), 0.0, CLIP_LMAX)));
}

// The horizon's angle on the path from p toward the rising (sgn = 1, +x) or setting (−1) side.
fn pathHorizon(p: vec3f, sgn: f32, s: f32) -> f32 {
  let dip = sqrt(4.0 * max(p.y, 0.0) * CURV);
  var th = -asin(min(sin(dip) / COS_LAT, 1.0));
  var a = max(0.6, 1.5 * s);
  let edge = exp2(CLIP_LMAX) / clip.cam.z;
  let reach = max(REACH, 64.0 * s);
  let minStep = max(0.5 * s, 0.5);
  for (var i = 0; i < 200; i++) {
    let b = a * tan(th);
    let q = p + vec3f(sgn * a, 0.0, 0.0) + PATH_UP * b;
    let rel = q.xz - clip.cam.xy;
    if (max(abs(rel.x), abs(rel.y)) > edge) { break; }
    let drop = (a * a + b * b * SIN_LAT * SIN_LAT) * CURV;
    let clear = q.y - (marchHeight(q.xz) - drop);
    if (clear < 0.0) {
      // Lift the horizon to the land's top at this distance (its crossing of
      // the plane, two fixed-point steps up the plane's slope).
      var bb = b;
      for (var k = 0; k < 4; k++) {
        let qq = p + vec3f(sgn * a, 0.0, 0.0) + PATH_UP * bb;
        let dr = (a * a + bb * bb * SIN_LAT * SIN_LAT) * CURV;
        bb = max(bb, (marchHeight(qq.xz) - dr - p.y) / COS_LAT);
      }
      th = atan2(bb, a);
    }
    // Past this the horizon's ray runs above anything the land can reach.
    if (p.y + a * tan(th) * COS_LAT + a * a * CURV > MAXH) { break; }
    a += max(minStep, max(a * 0.045, clear * 0.35));
    if (a > reach) { break; }
  }
  return th;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= gen.size)) { return; }
  let s = gen.spacing;
  let xz = vec2f(gen.rel0 + vec2i(id.xy)) * s;
  let e = heightAt(xz, vec2f(0.0), s);
  let gx = (heightAt(xz, vec2f(s, 0.0), s) - heightAt(xz, vec2f(-s, 0.0), s)) / (2.0 * s);
  let gz = (heightAt(xz, vec2f(0.0, s), s) - heightAt(xz, vec2f(0.0, -s), s)) / (2.0 * s);
  let n = normalize(vec3f(-gx, 1.0, -gz));
  let st = (gen.store0 + id.xy) % vec2u(${opts.n}u);
  writeCanopy(st, xz, e, vec2f(gx, gz), s);
  let p = vec3f(xz.x, e, xz.y) + n * (0.15 + 0.5 * s);
  let rise = pathHorizon(p, 1.0, s);
  let sinks = 3.14159265 - pathHorizon(p, -1.0, s);
  // The horizons, and the ground's gradient (the shading's normal) beside them.
  textureStore(horizonOut, st, gen.level, vec4f(rise, sinks, gx, gz));
}
`;
}

/** Elevation readback on a grid (parity with the CPU twin and the GLSL probe). */
export function probeWGSL(world: World): string {
  return /* wgsl */ `
${worldWGSL(world)}
struct Probe {
  origin: vec2f,
  local0: vec2f,
  step: f32,
  dp: f32,
  w: u32,
  seed: i32,
};
@group(0) @binding(0) var<uniform> pr: Probe;
@group(0) @binding(1) var<storage, read_write> outBuf: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let h = arrayLength(&outBuf) / pr.w;
  if (id.x >= pr.w || id.y >= h) { return; }
  let local = pr.local0 + vec2f(id.xy) * pr.step;
  let g = local + pr.origin;
  let sf = landSurface(g, pr.dp, pr.seed);
  outBuf[id.y * pr.w + id.x] = vec4f(sf.e, g.x, g.y, sf.w);
}
`;
}
