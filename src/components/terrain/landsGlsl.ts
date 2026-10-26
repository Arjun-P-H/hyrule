/**
 * The lands' world function in GLSL ES 3.00 — the WebGL2 renderer's twin of terrainMath.ts landSurface
 * / landRegions, hydro.ts (the geography, the rivers read from the node table) and erosion.ts, and of
 * the WebGPU ground look (gpu/worldgen.wgsl.ts, gpu/terrain.wgsl.ts biomeShader + groundLook).
 * ---------------------------------------------------------------------------
 * Generated from the same tables (constants.ts F, GEO, HYDRO, LANDS, CLIMATE; biomes.ts LANDFORM,
 * LOOK), so the three cannot drift. demoShader.ts interpolates LANDS_CONST_GLSL into its constants and
 * landsGlsl() after its noise helpers (vnoise, hash3, voronoiEdge, domainWarp); the rivers' node table
 * is the texture u_hydro (hydro.ts hydroTable, six texels a node) with its header u_hydro_head.
 */

import { CLIMATE, CRAG, F, GEO, HYDRO, LANDS, TOWER } from '../../lib/constants';
import { BI, BIOMES, BIOME_COUNT, LANDFORM, LANDFORM_KEYS, LOOK, SNOW_ALBEDO } from '../../lib/biomes';
import { CEIL_SPAN, CEIL_START, DRY_TABLE, ERO_STEP, LAND_MEANS, RIDGE_AMP, RIDGE_FREQ } from '../../lib/terrainMath';
import { COAST_RAMP, CRATER_RIM, SINK } from '../../lib/hydro';
import { EROSION_GLSL } from '../../lib/erosion';

/** A GLSL float literal. */
const f = (x: number): string => {
  const s = String(x);
  return /[.eE]/.test(s) ? s : `${s}.0`;
};
const v3 = (a: readonly number[]) => `vec3(${a.map(f).join(', ')})`;
const NB = BIOME_COUNT, NK = LANDFORM_KEYS.length;
const KI = (k: (typeof LANDFORM_KEYS)[number]) => LANDFORM_KEYS.indexOf(k);

/** Every spatial frequency as `n / 1048576.0` (the dyadic registry's form; tests/floatingOrigin.test.ts). */
export const LANDS_CONST_GLSL = Object.entries(F)
  .filter(([k]) => k !== 'veg')
  .map(([k, v]) => `const float F_${k.toUpperCase()} = ${f(Math.round(v * 1048576))} / 1048576.0;`)
  .join('\n');

/** The world function (the geography, regions, landforms, rivers, lakes, ponds, the water) and the ground's look. */
export function landsGlsl(): string {
  const LF = BIOMES.flatMap((b) => LANDFORM_KEYS.map((k) => f(LANDFORM[b][k])));
  const L = BIOMES.map((b) => LOOK[b]);
  const V = LANDS.volcano, G = LANDS.sacred, K = LANDS.coast, H = HYDRO;
  return /* glsl */ `
// ---- the lands' constants ----------------------------------------------------
const float SEA = ${f(LANDS.sea)};
const float COAST_LEVEL = ${f(K.level)};
const float T_AMP = ${f(CLIMATE.tAmp)}, T_FINE = ${f(CLIMATE.tFine)}, M_FINE = ${f(CLIMATE.mFine)};
const float SNOW_T = ${f(CLIMATE.snowT)};
const float RIDGE_MEAN = ${f(LAND_MEANS.ridge)}, TOWER_MEAN = ${f(LAND_MEANS.tower)}, DUNE_MEAN = ${f(LAND_MEANS.dune)};
const float CEIL_START = ${f(CEIL_START)}, CEIL_SPAN = ${f(CEIL_SPAN)};
const float RIDGE_FREQS[5] = float[5](${RIDGE_FREQ.map(f).join(', ')});
const float RIDGE_AMPS[5] = float[5](${RIDGE_AMP.map(f).join(', ')});
// per biome its landform (LANDFORM_KEYS order): LF_TABLE[b * ${NK} + k]
const float LF_TABLE[${LF.length}] = float[${LF.length}](${LF.join(', ')});
const vec3 LK_SOIL[${NB}] = vec3[${NB}](${L.map((l) => v3(l.soil)).join(', ')});
const vec3 LK_GRASS[${NB}] = vec3[${NB}](${L.map((l) => v3(l.grass)).join(', ')});
const vec3 LK_DRY[${NB}] = vec3[${NB}](${L.map((l) => v3(l.dry)).join(', ')});
const vec3 LK_ROCK[${NB}] = vec3[${NB}](${L.map((l) => v3(l.rock)).join(', ')});
const vec4 LK_A[${NB}] = vec4[${NB}](${L.map((l) => `vec4(${f(l.cover)}, ${f(l.blade)}, ${f(l.strata)}, ${f(l.wet)})`).join(', ')});
const float LK_FROST[${NB}] = float[${NB}](${L.map((l) => f(l.frost)).join(', ')});

// The rivers' node table round the camera (hydro.ts hydroTable): six texels a node, row j, texel i·6 + k;
// header (i0, j0, n, node size).
uniform highp sampler2D u_hydro;
uniform vec4 u_hydro_head;

float n01(vec2 g, float fr, int seed) { return 0.5 + 0.5 * vnoise(g.x * fr, g.y * fr, seed); }


// ---- the geography (mirror hydro.ts): the uplift surface ---------------------------
float continentAt(vec2 w, int seed) {
  return 0.5 + 0.5 * (0.78 * vnoise(w.x * F_CONT, w.y * F_CONT, seed + 801) + 0.22 * vnoise(w.x * F_COAST, w.y * F_COAST, seed + 802));
}
float basinAt(vec2 w, int seed) {
  float cell = ${f(GEO.basinCell)};
  int ix = int(floor(w.x / cell)), iz = int(floor(w.y / cell));
  uint h = hash3(ix, iz, seed + 3151);
  if (float(h >> 24u) / 255.0 >= ${f(GEO.basinP)}) return 0.0;
  vec2 c = (vec2(float(ix), float(iz)) + 0.3 + 0.4 * vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) / 4095.0) * cell;
  uint h2 = hash3(ix, iz, seed + 3152);
  float R = ${f(GEO.basinR[0])} + ${f(GEO.basinR[1] - GEO.basinR[0])} * float(h2 & 0xffffu) / 65535.0;
  float d = length(w - c) / R;
  if (d >= 1.0) return 0.0;
  float D = ${f(GEO.basinD[0])} + ${f(GEO.basinD[1] - GEO.basinD[0])} * float(h2 >> 16u) / 65535.0;
  float q = 1.0 - smoothstep(0.0, 1.0, d);
  return D * q * sqrt(q);
}
float rangeAt(vec2 w, int seed) {
  float rm = smoothstep(${f(GEO.rangeMask[0])}, ${f(GEO.rangeMask[1])}, n01(w, F_RANGEMASK, seed + 3121));
  if (rm <= 0.0) return 0.0;
  // (bent again at three finer scales: hydro.ts rangeAt)
  vec2 b = w + ${f(GEO.rangeWarp)} * vec2(vnoise(w.x * F_RANGEWARP, w.y * F_RANGEWARP, seed + 3113), vnoise(w.x * F_RANGEWARP + 3.7, w.y * F_RANGEWARP - 1.9, seed + 3114))
    + vec2(
      ${f(GEO.rangeBend[0])} * vnoise(w.x * F_PEAKS, w.y * F_PEAKS, seed + 3116) + ${f(GEO.rangeBend[1])} * vnoise(w.x * F_FLOORWOB, w.y * F_FLOORWOB, seed + 3117) + ${f(GEO.rangeBend[2])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3118),
      ${f(GEO.rangeBend[0])} * vnoise(w.x * F_PEAKS, w.y * F_PEAKS, seed + 3119) + ${f(GEO.rangeBend[1])} * vnoise(w.x * F_FLOORWOB, w.y * F_FLOORWOB, seed + 3120) + ${f(GEO.rangeBend[2])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3122));
  float ca = ${f(Math.cos(GEO.rangeDir[0]))}, sa = ${f(Math.sin(GEO.rangeDir[0]))};
  float cb = ${f(Math.cos(GEO.rangeDir[1]))}, sb = ${f(Math.sin(GEO.rangeDir[1]))};
  float c1 = 1.0 - abs(vnoise((b.x * ca + b.y * sa) * F_RANGEU, (b.y * ca - b.x * sa) * F_RANGEV, seed + 3111));
  float c2 = 1.0 - abs(vnoise((b.x * cb + b.y * sb) * F_RANGEU, (b.y * cb - b.x * sb) * F_RANGEV, seed + 3112));
  float c = mix(c1, c2, smoothstep(0.38, 0.62, n01(w, F_RANGEBLEND, seed + 3115)));
  float pk = ${f(GEO.pass)} + ${f(1 - GEO.pass)} * n01(w, F_PEAKS, seed + 3131);
  return rm * pow(c, ${f(GEO.rangePow)}) * pk;
}
// A tableland's weight at warped w, 0–1 (mirror hydro.ts plateauAt): its outline bent, its wall's width varied.
float plateauAt(vec2 w, int seed) {
  float pn0 = 0.72 * n01(w, F_PLATCOARSE, seed + 1401) + 0.28 * n01(w, F_PLATFINE, seed + 1402);
  if (pn0 < ${f(GEO.plat - 0.05 - GEO.platBand)}) return 0.0;
  if (pn0 > ${f(GEO.plat + GEO.platEdge + GEO.platRamp + GEO.platBand)}) return 1.0;
  vec2 p = w + vec2(
    ${f(GEO.platWarp[0])} * vnoise(w.x * F_PLATBAY, w.y * F_PLATBAY, seed + 1411) + ${f(GEO.platWarp[1])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 1412),
    ${f(GEO.platWarp[0])} * vnoise(w.x * F_PLATBAY + 5.3, w.y * F_PLATBAY - 2.1, seed + 1413) + ${f(GEO.platWarp[1])} * vnoise(w.x * F_PLATNOTCH + 3.7, w.y * F_PLATNOTCH - 9.1, seed + 1414));
  float pn = 0.72 * n01(p, F_PLATCOARSE, seed + 1401) + 0.28 * n01(p, F_PLATFINE, seed + 1402);
  float ew = ${f(GEO.platEdge)} + ${f(GEO.platRamp)} * smoothstep(0.35, 0.7, n01(w, F_PLATWIDTH, seed + 1415));
  return 0.82 * smoothstep(${f(GEO.plat)}, ${f(GEO.plat)} + ew, pn) + 0.18 * smoothstep(${f(GEO.plat - 0.05)}, ${f(GEO.plat)}, pn);
}
float baseAt(vec2 w, int seed) {
  float low = ${f(GEO.low[0])} + ${f(GEO.low[1])} * n01(w, F_UPL, seed + 3101) + ${f(GEO.low[2])} * n01(w, F_UPL2, seed + 3102);
  float rng = rangeAt(w, seed);
  float ms = smoothstep(${f(GEO.massif[0])}, ${f(GEO.massif[1])}, n01(w, F_MASSIF, seed + 3141));
  float pl = plateauAt(w, seed);
  float notRange = 1.0 - smoothstep(0.15, 0.45, rng);
  return low + max(${f(GEO.rangeH)} * rng, ${f(GEO.massifH)} * ms * ms) + ${f(GEO.platH)} * pl * notRange - basinAt(w, seed) * notRange;
}
float coastHeight(float raw, float inland, float cliff, vec2 w, int seed) {
  if (inland >= 1.0) return raw;
  float cap = ${f(GEO.cliffH[0])} + ${f(GEO.cliffH[1] - GEO.cliffH[0])} * n01(w, F_CLIFFS, seed + 2702);
  return raw * inland + (1.0 - inland) * cliff * min(raw, cap);
}
struct Uplift {
  float U, C, range, massif, plat, cliff, taper;
  vec4 vol;     // the volcano: centre (warped), radius of its region (0: none), cone height
  float vW, cone, crater, lava, rr;
};
void volcanoAt(vec2 w, int seed, inout Uplift o) {
  float VC = ${f(V.cell)};
  int ix = int(floor(w.x / VC)), iz = int(floor(w.y / VC));
  uint h = hash3(ix, iz, seed + 2101);
  if (float(h >> 24u) / 255.0 >= ${f(V.p)}) return;
  vec2 c = vec2((float(ix) + 0.28 + 0.44 * float(h & 0xfffu) / 4095.0) * VC, (float(iz) + 0.28 + 0.44 * float((h >> 12u) & 0xfffu) / 4095.0) * VC);
  uint h2 = hash3(ix, iz, seed + 2102);
  float R = ${f(V.radius[0])} + ${f(V.radius[1] - V.radius[0])} * float(h2 & 0xffffu) / 65535.0;
  float r = length(w - c);
  if (r >= R || continentAt(c, seed) <= COAST_LEVEL + 0.08 || rangeAt(c, seed) > 0.08) return;
  // (warm land only, as hydro.ts volcanoAt)
  float T0 = T_BASE + T_AMP * (vnoise(c.x * F_TEMP, c.y * F_TEMP, seed + 1201) + T_FINE * vnoise(c.x * F_TEMPFINE, c.y * F_TEMPFINE, seed + 1202)) - L_LAT * abs(c.y);
  if (T0 < ${f(V.minT)}) return;
  float Hc = ${f(V.height[0])} + ${f(V.height[1] - V.height[0])} * float(h2 >> 16u) / 65535.0;
  o.vol = vec4(c, R, Hc);
  o.vW = 1.0 - smoothstep(0.55 * R, R, r);
  float Rc = ${f(V.cone)} * R;
  if (r >= Rc) return;
  float q = 1.0 - r / Rc;
  float cone = Hc * q * pow(q, 0.9);
  float top = Hc * ${f(V.crater)};
  if (cone > top) cone = top - (cone - top) * 0.6;
  o.rr = r / Rc;
  if (o.rr < ${f(CRATER_RIM * 1.6)}) o.crater = top - 0.05 * Hc;
  o.cone = cone;
}
Uplift upliftAt(vec2 w, int seed) {
  Uplift o;
  o.vol = vec4(0.0); o.vW = 0.0; o.cone = 0.0; o.crater = -1e9; o.lava = -1e9; o.rr = 1.0;
  float C = continentAt(w, seed);
  o.C = C;
  float cliff = smoothstep(${f(K.cliffs[0])}, ${f(K.cliffs[1])}, n01(w, F_CLIFFS, seed + 2701));
  float inland = smoothstep(COAST_LEVEL, COAST_LEVEL + ${f(K.inland)}, C);
  o.cliff = cliff;
  o.taper = cliff + (1.0 - cliff) * inland;
  float low = ${f(GEO.low[0])} + ${f(GEO.low[1])} * n01(w, F_UPL, seed + 3101) + ${f(GEO.low[2])} * n01(w, F_UPL2, seed + 3102);
  float rng = rangeAt(w, seed);
  float ms = smoothstep(${f(GEO.massif[0])}, ${f(GEO.massif[1])}, n01(w, F_MASSIF, seed + 3141));
  float massif = ms * ms;
  float pl = plateauAt(w, seed);
  o.range = rng; o.massif = massif; o.plat = pl;
  float hi = max(${f(GEO.rangeH)} * rng, ${f(GEO.massifH)} * massif);
  float notRange = 1.0 - smoothstep(0.15, 0.45, rng);
  volcanoAt(w, seed, o);
  float base = low + hi + ${f(GEO.platH)} * pl * notRange - basinAt(w, seed) * notRange;
  float raw = base + o.cone;
  if (o.rr < 1.0) {
    float bc = baseAt(o.vol.xy, seed);
    raw += (bc - base) * (1.0 - smoothstep(0.35, 1.0, o.rr));
    if (o.crater > -1e8) o.lava = SEA + ${f(K.beach)} + max(bc + o.crater, 1.0);
  }
  o.U = SEA - 4.0 - 110.0 * smoothstep(COAST_LEVEL, COAST_LEVEL - 0.1, C);
  if (C >= COAST_LEVEL - ${f(COAST_RAMP)}) {
    // (the land rises from the sea floor over a steep but continuous face: a sea cliff, a beach's foot)
    float land = SEA + ${f(K.beach)} + coastHeight(max(raw, 1.0), inland, cliff, w, seed);
    o.U += (land - o.U) * smoothstep(COAST_LEVEL - ${f(COAST_RAMP)}, COAST_LEVEL, C);
  }
  return o;
}

// ---- §13 the biome regions (mirror terrainMath.ts regionsInto) ----------------
// The regions' weights into wt; the climate into clim (T0, M, C, range); the volcano's and the sacred
// grove's weights into sp (x, y).
void regionsFrom(vec2 g, vec2 w, Uplift up, int seed, out float wt[${NB}], out vec4 clim, out vec2 sp) {
  // (the regions' edges wander: terrainMath.ts regionsInto)
  vec2 r = w + vec2(
    ${f(LANDS.edge[0])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 1211) + ${f(LANDS.edge[1])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 1212),
    ${f(LANDS.edge[0])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 1213) + ${f(LANDS.edge[1])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 1214));
  float T0 = T_BASE + T_AMP * (vnoise(r.x * F_TEMP, r.y * F_TEMP, seed + 1201) + T_FINE * vnoise(r.x * F_TEMPFINE, r.y * F_TEMPFINE, seed + 1202))
    - L_LAT * abs(g.y);
  float C = up.C;
  float M = M_BASE + M_AMP * (vnoise(r.x * F_MOISTR, r.y * F_MOISTR, seed + 1301) + M_FINE * vnoise(r.x * F_MOISTFINE, r.y * F_MOISTFINE, seed + 1302))
    + M_OCEAN * (1.0 - smoothstep(COAST_LEVEL, COAST_LEVEL + 0.12, C));
  clim = vec4(T0, M, C, up.range);
  float mtn = smoothstep(0.26, 0.5, max(up.range, 0.9 * up.massif));
  float plat = smoothstep(0.35, 0.75, up.plat) * (1.0 - mtn);
  float low = 1.0 - mtn - plat;
  float a = smoothstep(${f(LANDS.cold[0])}, ${f(LANDS.cold[1])}, T0);
  float b = smoothstep(${f(LANDS.cool[0])}, ${f(LANDS.cool[1])}, T0);
  float c = smoothstep(${f(LANDS.hot[0])}, ${f(LANDS.hot[1])}, T0);
  float cold = 1.0 - a, cool = a * (1.0 - b), mild = a * b * (1.0 - c), hot = a * b * c;
  float dry = 1.0 - smoothstep(${f(LANDS.dry[0])}, ${f(LANDS.dry[1])}, M);
  float hdry = 1.0 - smoothstep(${f(LANDS.hotDry[0])}, ${f(LANDS.hotDry[1])}, M);
  float arid = 1.0 - smoothstep(${f(LANDS.arid[0])}, ${f(LANDS.arid[1])}, M);
  float wet = smoothstep(${f(LANDS.wet[0])}, ${f(LANDS.wet[1])}, M);
  float bad = smoothstep(${f(LANDS.rugged[0])}, ${f(LANDS.rugged[1])}, n01(r, F_RUGGED, seed + 1501));
  float lake = smoothstep(${f(LANDS.lakedist[0])}, ${f(LANDS.lakedist[1])}, n01(r, F_LAKEDIST, seed + 1601));
  float karst = smoothstep(${f(LANDS.karst[0])}, ${f(LANDS.karst[1])}, n01(r, F_KARST, seed + 1701));
  float marsh = wet * smoothstep(${f(LANDS.marsh[0])}, ${f(LANDS.marsh[1])}, n01(r, F_MARSH, seed + 1801));
  for (int k = 0; k < ${NB}; k++) wt[k] = 0.0;
  wt[${BI.tundra}] += low * cold;
  wt[${BI.taiga}] += low * cool * (1.0 - lake);
  wt[${BI.lakes}] += low * (cool + mild * (1.0 - dry)) * lake;
  float mildL = low * mild * (1.0 - lake * (1.0 - dry));
  float badWet = smoothstep(${f(LANDS.badlandsWet[0])}, ${f(LANDS.badlandsWet[1])}, M);
  wt[${BI.plains}] += mildL * dry * (1.0 - bad * (1.0 - badWet));
  wt[${BI.badlands}] += mildL * dry * bad * (1.0 - badWet);
  wt[${BI.forest}] += mildL * (1.0 - dry) * (1.0 - marsh);
  wt[${BI.wetland}] += mildL * (1.0 - dry) * marsh;
  float hotL = low * hot;
  wt[${BI.desert}] += hotL * arid * (1.0 - bad);
  wt[${BI.badlands}] += hotL * (hdry * bad + arid * bad * (1.0 - hdry));
  wt[${BI.savanna}] += hotL * hdry * (1.0 - arid) * (1.0 - bad);
  float hotM = hotL * (1.0 - hdry);
  wt[${BI.rainforest}] += hotM * (1.0 - karst) * (1.0 - marsh);
  wt[${BI.jungle}] += hotM * karst * (1.0 - marsh);
  wt[${BI.wetland}] += hotM * marsh;
  wt[${BI.highland}] += plat * (cold + cool + mild);
  wt[${BI.badlands}] += plat * hot * hdry;
  wt[${BI.jungle}] += plat * hot * (1.0 - hdry);
  wt[${BI.mountains}] += mtn;
  float v = up.vW;
  float GC = ${f(G.cell)};
  int sx = int(floor(w.x / GC + 0.5)), sz = int(floor(w.y / GC + 0.5)); // (the warped plane, as regionsInto)
  uint sh = hash3(sx, sz, seed + 2201);
  float sc = 0.0;
  if (float(sh >> 24u) * (1.0 / 255.0) < ${f(G.p)}) {
    float cx = (float(sx) - 0.5 + 0.3 + 0.4 * float(sh & 0xfffu) * (1.0 / 4095.0)) * GC;
    float cz = (float(sz) - 0.5 + 0.3 + 0.4 * float((sh >> 12u) & 0xfffu) * (1.0 / 4095.0)) * GC;
    float R = ${f(G.radius[0])} + ${f(G.radius[1] - G.radius[0])} * float(hash3(sx, sz, seed + 2202) & 0xffffu) / 65535.0;
    vec2 d = w - vec2(cx, cz);
    if (dot(d, d) < R * R) {
      sc = (1.0 - smoothstep(0.6 * R, R, length(d))) * smoothstep(2.0, 5.0, T0) * smoothstep(0.3, 0.4, M) * (1.0 - mtn);
    }
  }
  if (v > 0.0) {
    for (int k = 0; k < ${NB}; k++) wt[k] *= 1.0 - v;
    wt[${BI.volcanic}] += v;
  }
  if (sc > 0.0) {
    float s = sc * (1.0 - v);
    for (int k = 0; k < ${NB}; k++) wt[k] *= 1.0 - s;
    wt[${BI.sacred}] += s;
  }
  sp = vec2(v, sc);
}

// The blended landform parameter k (LANDFORM_KEYS order).
float lfParam(float wt[${NB}], int k) {
  float p = 0.0;
  for (int i = 0; i < ${NB}; i++) p += wt[i] * LF_TABLE[i * ${NK} + k];
  return p;
}

// ---- §6/§11 landforms ---------------------------------------------------------
float karstTowers(vec2 w, int seed, float dp) {
  float wW = clamp(2.0 - 2.0 * dp * F_TOWERWARP, 0.0, 1.0);
  float wB = clamp(2.0 - 2.0 * dp * F_TOWERBUMP, 0.0, 1.0);
  float wC = clamp(2.0 - 2.0 * dp * F_TOWERCROWN, 0.0, 1.0);
  float w1 = clamp(2.0 - dp / 15.0, 0.0, 1.0);
  float w2 = clamp(2.0 - dp / 7.0, 0.0, 1.0);
  float aw = ${f(TOWER.warp)} * wW;
  vec2 wp = w + aw * vec2(vnoise(w.x * F_TOWERWARP, w.y * F_TOWERWARP, seed + 2311), vnoise(w.x * F_TOWERWARP, w.y * F_TOWERWARP, seed + 2312));
  float crown = 1.0 + ${f(TOWER.crown)} * wC * vnoise(w.x * F_TOWERCROWN, w.y * F_TOWERCROWN, seed + 2313);
  float bump = 0.0;
  if (wB > 0.0) bump = ${f(TOWER.bump)} * wB * vnoise(w.x * F_TOWERBUMP, w.y * F_TOWERBUMP, seed + 2314);
  vec2 p = wp * F_TOWER;
  vec2 ip = floor(p);
  float best = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      int ci = int(ip.x) + i;
      int cj = int(ip.y) + j;
      uint h = hash3(ci, cj, seed + 2301);
      vec2 d = p - (vec2(float(ci), float(cj)) + 0.2 + 0.6 * vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0));
      float r = (75.0 + 70.0 * float((h >> 24u) & 15u) / 15.0) * F_TOWER;
      if (dot(d, d) >= ${f(TOWER.reach)} * r * r) continue;
      uint g = hash3(ci, cj, seed + 2302);
      if ((g >> 28u) < ${Math.round(TOWER.absent * 16)}u) continue;
      float al = 0.5 + 1.6 * (float(g & 63u) / 63.0 - 0.5);
      float ca = cos(al), sa = sin(al);
      float sq = sqrt(1.0 + ${f(TOWER.stretch)} * float((g >> 6u) & 15u) / 15.0);
      vec2 uv = vec2(dot(d, vec2(ca, sa)) / (r * sq), (d.y * ca - d.x * sa) * sq / r);
      float q0 = length(uv);
      float q = q0, n1 = 0.0;
      if (w1 > 0.0 && q0 > 0.3) {
        vec2 cs = uv / q0;
        vec2 o = vec2(float((g >> 10u) & 63u), float((g >> 16u) & 63u));
        n1 = vnoise(${f(TOWER.rho[0])} * cs.x + o.x + 0.9 * q0, ${f(TOWER.rho[0])} * cs.y + o.y - 0.6 * q0, seed + 2321);
        float D = ${f(TOWER.flute[0])} * w1 * (abs(n1) - 0.38);
        if (w2 > 0.0) D += ${f(TOWER.flute[1])} * w2 * (abs(vnoise(${f(TOWER.rho[1])} * cs.x + o.x, ${f(TOWER.rho[1])} * cs.y + o.y, seed + 2322)) - 0.38);
        q = q0 * (1.0 + D * smoothstep(0.3, 0.65, q0));
      }
      if (q >= 1.0) continue;
      float sx = min((1.0 - q) / ${f(TOWER.wall)}, 1.0);
      float y = sx * sx * sx * (sx * (sx * 6.0 - 15.0) + 10.0);
      float ns = 5.0 + 1.5 * float((g >> 22u) & 3u);
      float ph = y * ns + float((g >> 24u) & 15u) / 15.0 + 1.2 * n1;
      float fl = floor(ph);
      y += ${f(TOWER.ledge)} * (fl + smoothstep(0.3, 0.7, ph - fl) - ph) / ns * smoothstep(0.0, 0.12, y);
      float H = (50.0 + 170.0 * float((h >> 28u) & 15u) / 15.0) * crown;
      best = max(best, (H * (1.0 - ${f(TOWER.dome)} * q * q) + bump) * y);
    }
  }
  return best;
}

float dunesE(vec2 g, int seed, float dp) {
  float wMain = clamp(2.0 - 2.0 * dp * (2.0 * F_DUNE), 0.0, 1.0);
  float wCross = clamp(2.0 - 2.0 * dp * (2.0 * F_DUNECROSS), 0.0, 1.0);
  float amp = 0.3 + 0.7 * n01(g, F_DUNEAMP, seed + 2403);
  if (wMain <= 0.0) return DUNE_MEAN * amp;
  float ang = 0.6 + 0.9 * vnoise(g.x * F_DUNEFIELD, g.y * F_DUNEFIELD, seed + 2401);
  float ca = cos(ang), sa = sin(ang);
  // (the crests' bends at three scales: terrainMath.ts dunesE)
  float u = (g.x * ca + g.y * sa) * F_DUNE + 0.9 * vnoise(g.x * F_WARP, g.y * F_WARP, seed + 2402) + 0.55 * vnoise(g.x * F_DUNEBEND, g.y * F_DUNEBEND, seed + 2405)
    + 0.16 * clamp(2.0 - 2.0 * dp * F_PLATNOTCH, 0.0, 1.0) * vnoise(g.x * F_PLATNOTCH, g.y * F_PLATNOTCH, seed + 2406);
  float s = fract(u);
  float prof = s < 0.86 ? pow(smoothstep(0.0, 0.86, s), 1.3) : 1.0 - smoothstep(0.86, 1.0, s);
  float d = 30.0 * prof;
  if (wCross > 0.0) {
    float u2 = (g.y * ca - g.x * sa) * F_DUNECROSS + 0.6 * vnoise(g.x * F_DUNEBEND, g.y * F_DUNEBEND, seed + 2404);
    float s2 = fract(u2);
    d += 5.0 * wCross * (s2 < 0.75 ? smoothstep(0.0, 0.75, s2) : 1.0 - smoothstep(0.75, 1.0, s2));
  }
  return amp * (wMain * d + (1.0 - wMain) * DUNE_MEAN);
}

float mesasE(vec2 w, int seed, float dp, float region0) {
  // the region's edge wanders (terrainMath.ts mesasE)
  float region = clamp(region0 + 4.0 * region0 * (1.0 - region0) * (0.45 * clamp(2.0 - 2.0 * dp * F_TOWER, 0.0, 1.0) * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 2541)
    + 0.25 * clamp(2.0 - 2.0 * dp * F_PLATNOTCH, 0.0, 1.0) * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 2542)), 0.0, 1.0);
  // the outline bent (as the tablelands': bays and spurs, notches), the finer scale fading out where the footprint would alias it
  float kb = clamp(2.0 - 4.0 * dp * F_PLATBAY, 0.0, 1.0);
  float kn = clamp(2.0 - 4.0 * dp * F_PLATNOTCH, 0.0, 1.0);
  float A = ${f(GEO.mesaWarp[0])} * kb;
  float B = ${f(GEO.mesaWarp[1])} * kn;
  vec2 m = w + vec2(
    A * vnoise(w.x * F_PLATBAY, w.y * F_PLATBAY, seed + 2511) + B * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 2512),
    A * vnoise(w.x * F_PLATBAY + 5.3, w.y * F_PLATBAY - 2.1, seed + 2513) + B * vnoise(w.x * F_PLATNOTCH + 3.7, w.y * F_PLATNOTCH - 9.1, seed + 2514));
  // (toward the region's edge the field is lowered: buttes on its fringe, one plateau in its heart, no wall along its edge)
  float n = 0.7 * n01(m, F_MESA, seed + 2501) + 0.3 * n01(m, F_MESAFINE, seed + 2502) - (1.0 - region) * ${f(GEO.mesaFade)} * (0.6 + 0.8 * n01(w, F_PLATBAY, seed + 2531));
  // (the crags: ridged knobs in the field, which only the walls, where the field is steep in height, show)
  float c0 = clamp(2.0 - 2.0 * dp * F_CRAGBIG, 0.0, 1.0);
  float c1 = clamp(2.0 - 2.0 * dp * F_CRAG, 0.0, 1.0);
  float c2 = clamp(2.0 - 2.0 * dp * F_CRAGFINE, 0.0, 1.0);
  if (c0 > 0.0) { float a = 1.0 - abs(vnoise(w.x * F_CRAGBIG, w.y * F_CRAGBIG, seed + 2523)); n += ${f(CRAG.mesa[0])} * c0 * (a * a - RIDGE_MEAN); }
  if (c1 > 0.0) { float a = 1.0 - abs(vnoise(w.x * F_CRAG, w.y * F_CRAG, seed + 2521)); n += ${f(CRAG.mesa[1])} * c1 * (a * a - RIDGE_MEAN); }
  if (c2 > 0.0) { float a = 1.0 - abs(vnoise(w.x * F_CRAGFINE, w.y * F_CRAGFINE, seed + 2522)); n += ${f(CRAG.mesa[2])} * c2 * (a * a - RIDGE_MEAN); }
  // the walls: sheer, or a slope where the field says so
  float ramp = smoothstep(0.45, 0.78, n01(w, F_PLATWIDTH, seed + 2515));
  float cw = max(0.025, 0.02 * dp) * (1.0 + ${f(GEO.mesaRamp)} * ramp);
  // (scaled in over the region's first quarter: landDetail calls this only where the region's weight is above 0)
  return smoothstep(0.0, 0.25, region) * (80.0 * (0.62 * smoothstep(0.0, cw, n - 0.42) + 0.38 * smoothstep(-0.2, 0.0, n - 0.42))
    + 70.0 * (0.62 * smoothstep(0.0, cw, n - 0.52) + 0.38 * smoothstep(-0.2, 0.0, n - 0.52))
    + 60.0 * (0.62 * smoothstep(0.0, cw, n - 0.61) + 0.38 * smoothstep(-0.2, 0.0, n - 0.61)));
}

float hillsAt(vec2 m, int seed, float dp) {
  float hLow = vnoise(m.x, m.y, seed + 1) + 0.5 * vnoise(m.x * 2.0, m.y * 2.0, seed + 1 + 40503);
  float hHigh = 0.0;
  float k = 2.0 * F_MACRO * dp;
  for (int i = 2; i < 9; i++) {
    float fr = exp2(float(i));
    float wgt = dp <= 0.0 ? 1.0 : clamp(2.0 - fr * k, 0.0, 1.0);
    if (wgt <= 0.0) break;
    hHigh += exp2(-float(i)) * wgt * vnoise(m.x * fr, m.y * fr, seed + 1 + i * 40503);
  }
  return 0.2833333 * hLow + 0.5 * hHigh;
}
float ridgesAt(vec2 w, int seed, float dp) {
  vec2 rw = w * F_RIDGE;
  float kr = 4.0 * F_RIDGE * dp;
  float rh = 0.0;
  for (int i = 0; i < 5; i++) {
    float fr = RIDGE_FREQS[i];
    float wgt = dp <= 0.0 ? 1.0 : clamp(2.0 - fr * kr, 0.0, 1.0);
    if (wgt > 0.0) {
      float n = 1.0 - abs(vnoise(rw.x * fr, rw.y * fr, seed + 5 + i * 34283));
      rh += RIDGE_AMPS[i] * wgt * (n * n - RIDGE_MEAN);
    }
  }
  return rh;
}
const float ERO_STEP = ${f(ERO_STEP)};
float erosionLand(vec2 w, float U, int seed, float hills, float ridge) {
  float b = U + hills * hillsAt(w * F_MACRO, seed, ERO_STEP);
  if (ridge > 0.0) b += ridge * ridgesAt(w, seed, ERO_STEP);
  return b;
}
${EROSION_GLSL}

// ---- the rivers (mirror hydro.ts riverField), read from the node table ----------------------
const int HY_DI[8] = int[8](-1, 0, 1, -1, 1, -1, 0, 1);
const int HY_DJ[8] = int[8](-1, -1, -1, 0, 0, 1, 1, 1);
// The node of cell (i, j)'s texel k (k = 0 … 5), and whether it is in the window.
bool hyIn(int i, int j) {
  int li = i - int(u_hydro_head.x), lj = j - int(u_hydro_head.y), n = int(u_hydro_head.z);
  return li >= 0 && lj >= 0 && li < n && lj < n;
}
vec4 hyAt(int i, int j, int k) {
  return texelFetch(u_hydro, ivec2((i - int(u_hydro_head.x)) * 6 + k, j - int(u_hydro_head.y)), 0);
}
// A valley wall's rise x out from its floor's rim, walls of slope k, and its slope (mirror hydro.ts wallRise, wallSlope).
float wallRise(float k, float x, float fl) {
  float s0 = ${f(2 * (H.floorRise - 0.3))} / fl;
  return ${f(H.floorRise)} + s0 * x + (k - s0) * (x - ${f(H.wallRound)} * (1.0 - exp(-x / ${f(H.wallRound)}))) + ${f(H.wallCurve)} * x * x;
}
float wallSlope(float k, float x, float fl) {
  float s0 = ${f(2 * (H.floorRise - 0.3))} / fl;
  return s0 + (k - s0) * (1.0 - exp(-x / ${f(H.wallRound)})) + ${f(2 * H.wallCurve)} * x;
}
float valleyShape(float dd, float ch, float w, float depth, float floorW, float k) {
  if (ch < w) { float q = ch / w; return -depth * (1.0 - q * q); }
  float fr = ${f(H.floorRise)};
  float base = dd < floorW ? 0.3 + (fr - 0.3) * (dd / floorW) * (dd / floorW) : wallRise(k, dd - floorW, floorW);
  return base * smoothstep(w, w + 5.0, ch);
}
// How steeply valleyShape rises across the valley there (m per m, a bound).
float valleySlope(float dd, float ch, float w, float depth, float floorW, float k) {
  if (ch < w) return 2.0 * depth * ch / (w * w);
  float fr = ${f(H.floorRise)};
  float x = dd - floorW;
  float base = dd < floorW ? 0.3 + (fr - 0.3) * (dd / floorW) * (dd / floorW) : wallRise(k, x, floorW);
  float db = dd < floorW ? 2.0 * (fr - 0.3) * dd / (floorW * floorW) : wallSlope(k, x, floorW);
  float q = clamp((ch - w) / 5.0, 0.0, 1.0);
  return db * q * q * (3.0 - 2.0 * q) + base * 1.2 * q * (1.0 - q);
}
// A reach's level at t with its falls and without (smooth), and how fast each falls (m per unit t), in
// one pass over its profile (reachLevel, reachFall).
vec4 reachLF(float Wa, vec4 p0, vec4 p1, vec4 p2, vec4 p3, float Wb, float t) {
  float s = clamp(t, 0.0, 1.0) * 16.0;
  int k = min(int(floor(s)), 15);
  float u = s - float(k);
  float lv[17] = float[17](Wa, p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, Wb);
  float a = lv[k], b = lv[k + 1];
  float D = a - b;
  float smooth_ = a + (b - a) * u;
  if (D <= max(${f(H.fallDrop)}, 3.0 * (Wa - Wb) / 16.0)) return vec4(smooth_, smooth_, D * 16.0, D * 16.0);
  float fallH = min(D, ${f(H.maxFall)});
  float qf = clamp((u - 0.46) / 0.08, 0.0, 1.0);
  return vec4(a - fallH * smoothstep(0.46, 0.54, u) - (D - fallH) * u, smooth_, (fallH * 75.0 * qf * (1.0 - qf) + D - fallH) * 16.0, D * 16.0);
}
struct River { float R, F, xFloor, W, ch, w, depth, acc, Ua, fall, L, lakeK, lakeMask, vis, cap, Lnear; vec2 dir; };
// A channel's half width where the drainage is acc (m; mirror hydro.ts riverHalfW).
float riverHalfW(float acc) {
  return 0.5 * min(${f(H.widthMax)}, ${f(H.width[0])} + ${f(H.width[1])} * pow(max(acc - 1.0, 1e-6), ${f(H.width[2])}));
}
// The meander's offset (m) from a reach's line at along metres from its head, t its share of the reach.
float meanderOff(float along, float t, float amp, int sg) {
  if (amp <= 0.0) return 0.0;
  return amp * smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t) * (0.74 * vnoise(along * F_MEANDER, 0.5, sg) + 0.26 * vnoise(along * F_MEANDER * 3.1, 4.5, sg));
}
// How much wider than its base a reach's valley floor is at along (m): a smooth noise, nothing at the nodes.
float floorWobbleAt(float along, float t, int sg) {
  return ${f(H.floorWobble)} * smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t) * (0.5 + 0.5 * (0.6 * vnoise(along * F_FLOORWOB, 11.5, sg) + 0.4 * vnoise(along * F_FLOORWOB * 2.7, 5.5, sg)));
}
// How much of its width one side of a reach's floor keeps, along the reach (mirror hydro.ts floorSideAt).
float floorSideAt(float along, float t, int sg, float side) {
  float taper = smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t);
  float o = side * 40.7;
  float s = 0.5 + 0.5 * (0.62 * vnoise(along * F_FLOORWOB * 1.9 + o, 21.5, sg) + 0.38 * vnoise(along * F_FLOORWOB * 5.3 + o, 9.5, sg));
  return 1.0 - taper * 0.75 * (1.0 - s);
}
River riverField(vec2 w, int seed, float k) {
  River o;
  o.R = 1e9; o.F = -1e9; o.xFloor = 1e9; o.W = -1e9; o.ch = 1e9; o.w = 0.0; o.depth = 0.0; o.acc = 0.0; o.Ua = 0.0;
  o.dir = vec2(0.0); o.fall = 0.0; o.L = -1e9; o.lakeK = 0.0; o.lakeMask = 0.0; o.vis = 0.0; o.cap = 1e9; o.Lnear = -1e9;
  // each reach's (and river lake's) surface here: (surface, slope across, slope along, curvature) and
  // (the way across x, z; how loosely that holds; the lowest it can fall to: its own river's level)
  vec4 sA[34], sB[34];
  ivec2 sN[34];   // the reach's two nodes (cells; lakes: none)
  int nR = 0, best = -1;
  vec2 BN = vec2(0.0);
  float NODE = u_hydro_head.w;
  vec2 q = w - u_hydro_head.xy * NODE;
  int ci = int(floor(w.x / NODE)), cj = int(floor(w.y / NODE));
  float rL = ${f(H.lakeR)} * NODE;
  // (the walls' run wanders: hydro.ts riverField)
  float wk = 1.0 + ${f(H.wallVary[0])} * vnoise(w.x * F_FLOORWOB, w.y * F_FLOORWOB, seed + 3306) + ${f(H.wallVary[1])} * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3307)
    + ${f(H.wallVary[2])} * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 3308);
  float lat = ${f(H.wallLat)} * (0.6 * vnoise(w.x * F_TOWER, w.y * F_TOWER, seed + 3310) + 0.4 * vnoise(w.x * F_PLATNOTCH, w.y * F_PLATNOTCH, seed + 3311));
  // the lakes at sinks, and the lakes the rivers widen into (their bowls are cut below)
  vec4 lk[9];
  vec2 lkN[9];
  int nLk = 0;
  for (int j = cj - 1; j <= cj + 1; j++) {
    for (int i = ci - 1; i <= ci + 1; i++) {
      if (!hyIn(i, j)) continue;
      vec4 n0 = hyAt(i, j, 0), n1 = hyAt(i, j, 1);
      if (n1.x == ${SINK}.0 && n1.z > -1e8) {
        float d = length(q - n0.xy) / rL;
        o.lakeMask = max(o.lakeMask, 1.0 - smoothstep(1.0, 1.2, d));
        if (d < 1.0 && n1.z > o.L) { o.L = n1.z; o.lakeK = 1.0 - smoothstep(0.85, 1.0, d); }
        if (d < 1.35) o.Lnear = max(o.Lnear, n1.z);
      }
      float rr = hyAt(i, j, 5).w;
      if (rr > 0.0) {
        vec2 dv = q - n0.xy;
        float dist = length(dv);
        if (dist > ${f(2.5 * Math.max(H.riverLakeR[1], H.reservoirR[1]))}) continue;
        // (its shore: the noise's own place and a stretch along one axis, from the node's hash: hydro.ts lakeLobe)
        vec2 lq = dv / rr;
        uint lid = hash3(i, j, seed + 3510);
        int lns = int(lid & 0xffffu);
        float la2 = 6.283185307179586 * float(lid >> 24u) / 255.0;
        float le = 0.06 + 0.16 * float((lid >> 16u) & 255u) / 255.0;
        float ld2 = dot(dv, dv);
        float ell = 1.0 + (ld2 > 1e-9 ? le * ((dv.x * dv.x - dv.y * dv.y) * cos(la2) + 2.0 * dv.x * dv.y * sin(la2)) / ld2 : 0.0);
        float rl = rr * ell * (1.0 + 0.28 * (0.5 * vnoise(lq.x * 1.3, lq.y * 1.3, seed + 3503 + lns) + 0.3 * vnoise(lq.x * 3.1 + 9.2, lq.y * 3.1 - 4.4, seed + 3504 + lns) + 0.2 * vnoise(lq.x * 7.3 - 5.1, lq.y * 7.3 + 2.7, seed + 3505 + lns)));
        lk[nLk] = vec4(dist, rl, n1.z, ${f(H.riverLakeD[0])} + ${f(H.riverLakeD[1])} * rr);
        lkN[nLk] = dv / max(dist, 1e-6);
        nLk++;
        float d = dist / rl;
        o.lakeMask = max(o.lakeMask, 1.0 - smoothstep(1.0, 1.35, d));
        if (d < 1.0 && n1.z > o.L) { o.L = n1.z; o.lakeK = 1.0 - smoothstep(0.85, 1.0, d); }
        if (d < 1.35) o.Lnear = max(o.Lnear, n1.z);
      }
    }
  }
  float bestCh = 1e9;
  for (int j = cj - 2; j <= cj + 2; j++) {
    for (int i = ci - 2; i <= ci + 2; i++) {
      if (!hyIn(i, j)) continue;
      vec4 a0 = hyAt(i, j, 0), a1 = hyAt(i, j, 1);
      int code = int(a1.x);
      if (code > 7) continue;
      int bi = i + HY_DI[code], bj = j + HY_DJ[code];
      if (!hyIn(bi, bj)) continue;
      vec4 b0 = hyAt(bi, bj, 0);
      float Wa = a0.w, Wb = b0.w, acc = a1.y;
      vec2 dv = b0.xy - a0.xy;
      float len = length(dv);
      vec2 u = dv / len;
      vec2 r = q - a0.xy;
      float along = dot(r, u);
      float t = along / len;
      int sg = int(hash3(i, j, seed + 3301) & 0xffffu);
      // (the river widens as it goes: its head's drainage plus the share of the reach run)
      int sw = int(hash3(i, j, seed + 3302) & 0xffffu);
      float wd = riverHalfW(acc + clamp(t, 0.0, 1.0))
        * (1.0 + ${f(H.widthVary)} * (0.62 * vnoise(along * F_MEANDER * 2.3, 7.5, sw) + 0.38 * vnoise(along * F_MEANDER * 6.1, 3.5, sw)));
      float vis = a1.w; // (the river shows only below a lake: hydro.ts fedOf)
      // (a river whose water does not show cuts no channel: a dry wash, not a ditch)
      float depth = (${f(H.depth[0])} + ${f(H.depth[1])} * sqrt(acc)) * vis;
      float slope = (Wa - Wb) / len;
      float amp = ${f(H.meanderMin)} + ${f(H.meander - H.meanderMin)} * (1.0 - smoothstep(0.012, 0.05, (a0.z - b0.z) / len));
      // (the floor: hydro.ts riverField: its base width, of which each side keeps a share but not less than the channel and a margin)
      float baseW = (vis > 0.0 ? amp + wd : 0.0) + ${f(H.floor[0])} + ${f(H.floor[1])} * sqrt(acc) + floorWobbleAt(along, t, int(hash3(i, j, seed + 3305) & 0xffffu));
      float least = vis > 0.0 ? wd + 8.0 : ${f(H.floor[0])};
      float floorW = baseW;
      vec4 p0 = hyAt(i, j, 2), p1 = hyAt(i, j, 3), p2 = hyAt(i, j, 4), p3 = hyAt(i, j, 5);
      float dd, ch;
      if (t <= 0.0) { dd = length(r); ch = dd; }
      else if (t >= 1.0) { dd = length(q - b0.xy); ch = dd; }
      else {
        float ds = -r.x * u.y + r.y * u.x;
        dd = abs(ds);
        floorW = max(baseW * floorSideAt(along, t, int(hash3(i, j, seed + 3309) & 0xffffu), ds >= 0.0 ? 0.0 : 1.0), least);
        if (dd - floorW > 400.0) {
          float x = dd - floorW;
          vec4 lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
          float lvl = lf.y;
          float cone = lvl + wallRise(k, x * wk + lat, floorW);
          o.R = min(o.R, cone);
          o.F = max(o.F, lvl + ${f(H.floorRise)} - ${f(H.riser)} * x);
          o.xFloor = min(o.xFloor, x);
          if (nR < 34) {
            sA[nR] = vec4(cone, wallSlope(k, x * wk + lat, floorW) * wk, lf.w / len, ${f(H.wallCurve)});
            sB[nR] = vec4(-u.y, u.x, 0.1, lvl);
            sN[nR] = ivec2(j * 4096 + i, bj * 4096 + bi);
            nR++;
          }
          continue;
        }
        // (the banks of unequal reach: the channel's distance scaled so each bank stands where its own half width puts it)
        float sd = ds - meanderOff(along, t, amp, sg);
        float skew = ${f(H.bankSkew)} * vnoise(along * F_MEANDER * 1.7, 13.1, int(hash3(i, j, seed + 3304) & 0xffffu));
        ch = abs(sd) / (1.0 + (sd >= 0.0 ? skew : -skew));
      }
      vec4 lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
      float Wt = lf.x;
      float Wv = vis > 0.0 ? mix(Wt, lf.y, smoothstep(floorW, floorW + 60.0, dd)) : lf.y;
      // (a dry valley has no channel: hydro.ts riverField)
      float chV = vis > 0.0 ? ch : 1e9;
      float wV = vis > 0.0 ? wd : 0.0;
      float ddS = dd > floorW ? floorW + (dd - floorW) * wk + lat * smoothstep(0.0, 40.0, dd - floorW) : dd;
      float wkS = dd > floorW ? wk : 1.0;
      float cone = Wv + valleyShape(ddS, chV, wV, depth, floorW, k);
      o.R = min(o.R, cone);
      float fillH = dd < floorW ? 0.3 + ${f(H.floorRise - 0.3)} * (dd / floorW) * (dd / floorW) : ${f(H.floorRise)} - ${f(H.riser)} * (dd - floorW);
      o.F = max(o.F, Wv + fillH);
      o.xFloor = min(o.xFloor, dd - floorW);
      // this reach's surface (its water in its channel, else its ground), how fast it changes across its
      // valley and along it, and which way across is (round a node: outward from it)
      bool isEnd = t <= 0.0 || t >= 1.0;
      float fallT = isEnd ? 0.0 : max(lf.z, lf.w) / len;
      vec2 nrm = isEnd ? (t <= 0.0 ? r : q - b0.xy) / max(dd, 1e-6) : vec2(-u.y, u.x);
      if (nR < 34) {
        sN[nR] = ivec2(j * 4096 + i, bj * 4096 + bi);
        if (vis > 0.0 && ch < wd + 0.3) { sA[nR] = vec4(max(cone, Wt), 0.0, fallT, 0.0); sB[nR] = vec4(nrm, 0.1, min(Wt, Wv)); }
        else { sA[nR] = vec4(cone, valleySlope(ddS, chV, wV, depth, floorW, k) * wkS, fallT, ${f(H.wallCurve)}); sB[nR] = vec4(nrm, dd < floorW + 60.0 ? 0.35 : 0.1, min(Wt, Wv)); }
      }
      if (vis > 0.0 && ch - wd < bestCh) {
        bestCh = ch - wd; best = nR;
        o.W = Wt; o.ch = ch; o.w = wd; o.depth = depth; o.acc = acc; o.Ua = a0.z; o.vis = vis;
        o.dir = u; o.fall = slope;
        // across this channel: across its meander's own heading (round a node: outward)
        if (isEnd) BN = nrm;
        else {
          float dm = 0.5 * (meanderOff(along + 1.0, t + 1.0 / len, amp, sg) - meanderOff(along - 1.0, t - 1.0 / len, amp, sg));
          vec2 hdg = normalize(vec2(u.x - u.y * dm, u.y + u.x * dm));
          BN = vec2(-hdg.y, hdg.x);
        }
      }
      if (nR < 34) nR++;
    }
  }
  // the river lakes' bowls: below the water, a shore rising gently, then walls (they are valleys too)
  for (int m = 0; m < 9; m++) {
    if (m >= nLk) break;
    vec4 L4 = lk[m];
    float fl = L4.y + ${f(H.floor[0] + 20)};
    float dS = L4.x > fl ? fl + (L4.x - fl) * wk + lat * smoothstep(0.0, 40.0, L4.x - fl) : L4.x;
    float cone = L4.z + valleyShape(dS, L4.x, L4.y, L4.w, fl, k);
    o.R = min(o.R, cone);
    o.xFloor = min(o.xFloor, L4.x - fl);
    if (nR < 34) {
      sA[nR] = L4.x < L4.y ? vec4(max(cone, L4.z), 0.0, 0.0, ${f(H.wallCurve)}) : vec4(cone, valleySlope(dS, L4.x, L4.y, L4.w, fl, k) * (L4.x > fl ? wk : 1.0), 0.0, ${f(H.wallCurve)});
      sB[nR] = vec4(lkN[m], 0.3, L4.z);
      sN[nR] = ivec2(-1, -2);
      nR++;
    }
  }
  // the nearest channel's water stays under every other reach's surface out to its far bank (that surface
  // falls toward the bank by its slope along the line across the channel, but never below its own river)
  if (best >= 0) {
    float D = (o.w + o.ch) * ${f(1 + H.bankSkew)} + 1.0;
    float cap = 1e9;
    for (int r = 0; r < 34; r++) {
      if (r >= nR) break;
      if (r == best) continue;
      vec4 A = sA[r], Bv = sB[r];
      float cn = abs(dot(BN, Bv.xy));
      float cu = abs(BN.x * Bv.y - BN.y * Bv.x);
      float fall = (cn * A.y + cu * A.z + Bv.z * (A.y + A.z) + A.w * D) * D;
      // (reaches meeting this one at a node run on into it, level with it where they meet)
      ivec2 Nr = sN[r], Nb = sN[best];
      bool kin = Nr.x == Nb.x || Nr.x == Nb.y || Nr.y == Nb.x || Nr.y == Nb.y;
      cap = min(cap, max(A.x - fall, Bv.w - ((kin ? 0.0 : cu) + 0.05) * A.z * D));
    }
    o.cap = cap - 0.1;
  }
  return o;
}

float ceilH(float h) { return h > CEIL_START ? CEIL_START + CEIL_SPAN * tanh((h - CEIL_START) / CEIL_SPAN) : h; }

// The crags (m): rock faces break into knobs, ribs and ledges (mirror terrainMath.ts cragDelta).
float cragDelta(vec2 g, float e, float slope, int seed, float dp) {
  float s = smoothstep(${f(CRAG.face[0])}, ${f(CRAG.face[1])}, slope);
  if (s <= 0.0) return 0.0;
  float sl = min(slope, 3.2);
  float w0 = clamp(2.0 - 2.0 * dp * F_CRAGBIG, 0.0, 1.0);
  float w1 = clamp(2.0 - 2.0 * dp * F_CRAG, 0.0, 1.0);
  float w2 = clamp(2.0 - 2.0 * dp * F_CRAGFINE, 0.0, 1.0);
  float d = 0.0;
  if (w0 > 0.0) { float a = 1.0 - abs(vnoise(g.x * F_CRAGBIG, g.y * F_CRAGBIG, seed + 5124)); d += ${f(CRAG.rough[0])} * w0 * (a * a - RIDGE_MEAN); }
  if (w1 > 0.0) { float a = 1.0 - abs(vnoise(g.x * F_CRAG, g.y * F_CRAG, seed + 5121)); d += ${f(CRAG.rough[1])} * w1 * (a * a - RIDGE_MEAN); }
  if (w2 > 0.0) { float a = 1.0 - abs(vnoise(g.x * F_CRAGFINE, g.y * F_CRAGFINE, seed + 5122)); d += ${f(CRAG.rough[2])} * w2 * (a * a - RIDGE_MEAN); }
  float wl = clamp(2.0 - 2.0 * dp * sl / ${f(CRAG.period)}, 0.0, 1.0);
  float led = 0.0;
  if (wl > 0.0) {
    float ph = e / ${f(CRAG.period)} + 1.6 * vnoise(g.x * F_CRAGSTRATA, g.y * F_CRAGSTRATA, seed + 5123);
    float fl = floor(ph);
    led = ${f(CRAG.ledge)} * (fl + smoothstep(0.3, 0.7, ph - fl) - ph) * ${f(CRAG.period)} * wl;
  }
  return s * (sl * d + led);
}

// The regions' detail over the uplift at world g, warped w, before any mask (mirror terrainMath.ts landDetail).
float landDetail(vec2 g, vec2 w, Uplift up, float hills, float ridge, float karst, float dune, float mesa, float eros, int seed, float dp) {
  float detail = hills * hillsAt(w * F_MACRO, seed, dp);
  float rs = min(1.0, 2.0 * up.range + 1.6 * up.massif);
  if (ridge > 0.0) detail += ridge * (0.3 + 0.7 * rs) * ridgesAt(w, seed, dp);
  float kw = smoothstep(0.1, 0.9, karst);
  if (kw > 0.001) {
    float wK = clamp(2.0 - 2.0 * dp / 150.0, 0.0, 1.0);
    detail += kw * (wK > 0.0 ? wK * karstTowers(w, seed, dp) + (1.0 - wK) * TOWER_MEAN : TOWER_MEAN);
  }
  float dw = smoothstep(0.35, 0.75, dune);
  if (dw > 0.001) detail += dw * dunesE(g, seed, dp);
  float mw = smoothstep(0.05, 0.95, mesa);
  if (mw > 0.001) detail += mesasE(w, seed, dp, mw);
  if (eros > 0.01) {
    // (down the land itself, laid out in the world: terrainMath.ts landDetail)
    float rk = ridge > 0.0 ? ridge * (0.3 + 0.7 * rs) : 0.0;
    // (the three samples in one loop: with them inlined side by side, WebGL2 on an Apple GPU put the
    // gradient metres off although each sample alone was exact)
    vec3 eb = vec3(0.0);
    for (int k = 0; k < 3; k++) {
      vec2 ew = w;
      float eU = up.U;
      if (k > 0) {
        domainWarp(g.x + (k == 1 ? ERO_STEP : 0.0), g.y + (k == 2 ? ERO_STEP : 0.0), seed, 0.0, ew);
        eU = upliftAt(ew, seed).U;
      }
      eb[k] = erosionLand(ew, eU, seed, hills, rk);
    }
    vec2 gr = (eb.yz - eb.x) / ERO_STEP;
    // (the gullies' strength wanders, so their flutes are not evenly spaced for kilometres; the crags break every steep face)
    detail += eros * (0.55 + 0.9 * n01(g, F_PLATNOTCH, seed + 5101)) * erosionDelta(g, up.U + detail, gr, 0.0, dp); // (no fade target: terrainMath.ts landDetail)
    detail += smoothstep(0.05, 0.4, eros) * cragDelta(g, up.U + detail, length(gr), seed, dp);
  }
  return detail;
}

// The land's surface at world g with the pond (level or -1e9, distance in radii, depth, radius) given:
// x its height (prefiltered for dp), y the water's (dp-free), z a river channel's share, w a lake's.
vec4 landCore(vec2 g, float dp, int seed, vec4 pond) {
  vec2 w;
  domainWarp(g.x, g.y, seed, 0.0, w);
  Uplift up = upliftAt(w, seed);
  float wt[${NB}];
  vec4 clim;
  vec2 sp;
  regionsFrom(g, w, up, seed, wt, clim, sp);
  float hills = lfParam(wt, ${KI('hills')}), ridge = lfParam(wt, ${KI('ridge')}), karst = lfParam(wt, ${KI('karst')});
  float mesa = lfParam(wt, ${KI('mesa')}), dune = lfParam(wt, ${KI('dune')}), flat_ = lfParam(wt, ${KI('flat')});
  float kWall = lfParam(wt, ${KI('k')}), eros = lfParam(wt, ${KI('erosion')});
  River rv = riverField(w, seed, kWall);
  float m = smoothstep(0.0, 50.0, rv.xFloor) * (1.0 - rv.lakeMask);
  if (up.lava > -1e8) m *= smoothstep(${f(CRATER_RIM * 0.9)}, ${f(CRATER_RIM * 1.3)}, up.rr);
  // where marshes lie (not in a lake's ring, nor in a river's channel; only on flat land)
  float flat2 = flat_ * (1.0 - rv.lakeMask);
  if (rv.vis > 0.0) flat2 *= smoothstep(rv.w + 2.0, rv.w + 8.0, rv.ch);
  if (flat2 > 0.0) {
    vec2 gU = vec2(upliftAt(w + vec2(8.0, 0.0), seed).U - up.U, upliftAt(w + vec2(0.0, 8.0), seed).U - up.U) / 8.0;
    flat2 *= 1.0 - smoothstep(0.012, 0.035, length(gU));
  }
  float detail = 0.0;
  if (m > 0.0) {
    detail = landDetail(g, w, up, hills, ridge, karst, dune, mesa, eros, seed, dp);
    detail *= m * up.taper;
    // (round a river the detail only rises: a hollow beside it would leave the river standing over it)
    float near = 1.0 - smoothstep(50.0, 450.0, rv.xFloor);
    if (near > 0.0) detail += (0.5 * (detail + sqrt(detail * detail + 36.0)) - 3.0 - detail) * near;
    // (and round a marsh: its pools lie just under the smooth land, and a hollow at its edge would spill them)
    float nearM = smoothstep(0.0, 0.3, flat2);
    if (nearM > 0.0) detail += (0.5 * (detail + sqrt(detail * detail + 0.36)) - 0.3 - detail) * nearM;
  }
  float e0 = min(max(up.U + detail, rv.F), rv.R);
  float vis = rv.vis;
  if (up.lava > -1e8) {
    float band = smoothstep(${f(CRATER_RIM * 0.82)}, ${f(CRATER_RIM)}, up.rr) * (1.0 - smoothstep(${f(CRATER_RIM * 1.3)}, ${f(CRATER_RIM * 1.6)}, up.rr));
    if (up.lava + 4.0 > e0) e0 += (up.lava + 4.0 - e0) * band;
  }
  float marshW = -1e9;
  if (flat2 > 0.0) {
    float wH = clamp(2.0 - 2.0 * dp * F_HUMMOCK, 0.0, 1.0);
    float wH2 = clamp(2.0 - 2.0 * dp * F_HUMMOCKFINE, 0.0, 1.0);
    float hum = 1.4 * (wH * vnoise(g.x * F_HUMMOCK, g.y * F_HUMMOCK, seed + 2801) + 0.5 * wH2 * vnoise(g.x * F_HUMMOCKFINE, g.y * F_HUMMOCKFINE, seed + 2802)) - 0.2;
    float lvl = max(up.U - 1.2, SEA);
    e0 += (lvl + hum - e0) * flat2;
    marshW = lvl + 0.25;
    // pools only in the marsh's heart; round it a rim a little over them, faded in from the marsh's edge
    e0 += max(0.0, marshW + 0.05 - e0) * smoothstep(0.0, 0.05, flat2) * (1.0 - smoothstep(0.8, 0.82, flat2));
  }
  float pondL = -1e9;
  if (pond.x > -1e8) {
    if (pond.y < 1.0) {
      e0 = pond.x - pond.z * (1.0 - smoothstep(0.45, 0.92, pond.y)) + ${f(H.pondRim)} * smoothstep(0.8, 1.0, pond.y);
      pondL = pond.x;
    } else {
      e0 += (pond.x + ${f(H.pondRim)} - e0) * (1.0 - smoothstep(1.0, ${f(H.pondReach)}, pond.y));
    }
  }
  float land = smoothstep(COAST_LEVEL - 0.004 - 0.03 * (1.0 - up.cliff), COAST_LEVEL + 0.002, up.C);
  float seaFloor = SEA - 6.0 - ${f(K.shelf)} * (1.0 - smoothstep(COAST_LEVEL - 0.12, COAST_LEVEL, up.C));
  float e = seaFloor + (e0 - seaFloor) * land;
  float wl = e - ${f(DRY_TABLE)};
  float riverK = 0.0, lakeK = 0.0;
  bool hot = up.vol.z > 0.0 && length(w - up.vol.xy) < ${f(0.6 * V.cone)} * up.vol.z;
  // the river's water: its level, never over a bank another valley has cut lower; toward the sea it
  // sinks with its banks (they ease down to the sea floor together)
  float riverW = seaFloor + (min(rv.W, rv.cap) - seaFloor) * land;
  if (vis > 0.0 && rv.ch < rv.w + 0.3 && riverW > e && !(hot && rv.fall > 0.3)) { wl = max(wl, riverW); riverK = vis; }
  float lakeL = rv.L > -1e8 ? seaFloor + (rv.L - seaFloor) * land : -1e9;
  if (lakeL > e) { wl = max(wl, lakeL); lakeK = rv.lakeK; }
  float marshWc = marshW > -1e8 ? seaFloor + (marshW - seaFloor) * land : -1e9;
  if (flat2 > 0.8 && marshWc > e) wl = max(wl, marshWc);
  if (pondL > e) wl = max(wl, pondL);
  if (e < SEA && up.C < COAST_LEVEL + ${f(K.inland)}) wl = max(wl, SEA);
  if (up.lava > e && up.rr < ${f(CRATER_RIM)}) wl = max(wl, up.lava);
  // (the sea fills only what lies below its level by the coast: water eased toward it would climb the shore)
  return vec4(ceilH(e), ceilH(wl), riverK, lakeK);
}

// The land's broad shape at world g (m): uplift and rolling hills, prefiltered for dp (mirror terrainMath.ts landCoarse).
float landCoarse(vec2 g, float dp, int seed) {
  vec2 w;
  domainWarp(g.x, g.y, seed, 0.0, w);
  Uplift up = upliftAt(w, seed);
  float wt[${NB}];
  vec4 clim;
  vec2 sp;
  regionsFrom(g, w, up, seed, wt, clim, sp);
  return up.U + lfParam(wt, ${KI('hills')}) * hillsAt(w * F_MACRO, seed, dp);
}

// The land at world g before its rivers, lakes, marshes, ponds and the sea (mirror terrainMath.ts landBare).
float landBare(vec2 g, float dp, int seed) {
  vec2 w;
  domainWarp(g.x, g.y, seed, 0.0, w);
  Uplift up = upliftAt(w, seed);
  float wt[${NB}];
  vec4 clim;
  vec2 sp;
  regionsFrom(g, w, up, seed, wt, clim, sp);
  return up.U + landDetail(g, w, up, lfParam(wt, ${KI('hills')}), lfParam(wt, ${KI('ridge')}), lfParam(wt, ${KI('karst')}),
    lfParam(wt, ${KI('dune')}), lfParam(wt, ${KI('mesa')}), lfParam(wt, ${KI('erosion')}), seed, dp) * up.taper;
}

// A pond whose reach (pondReach radii) holds world g (mirror terrainMath.ts pondAt): (level or -1e9, distance
// from its centre over its radius, depth, radius).
vec4 pondAt(vec2 g, int seed) {
  vec4 none = vec4(-1e9, 1.0, 0.0, 0.0);
  float cell = 1.0 / F_POND;
  int ix = int(floor(g.x * F_POND)), iz = int(floor(g.y * F_POND));
  uint h = hash3(ix, iz, seed + 3401);
  float u = float(h >> 24u) / 255.0;
  float R = ${f(H.pondR[0])} + ${f(H.pondR[1] - H.pondR[0])} * u * u * u;
  float inset = ${f(H.pondReach * 1.6)} * R / cell;
  vec2 c = (vec2(float(ix), float(iz)) + inset + (1.0 - 2.0 * inset) * vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) / 4095.0) * cell;
  // (a place in the noise of its own and a stretch along one axis: terrainMath.ts pondAt)
  vec2 qq = (g - c) / R;
  uint g3 = hash3(ix, iz, seed + 3406);
  int pns = int(g3 & 0xffffu);
  vec2 pdv = g - c;
  float pd2 = dot(pdv, pdv);
  float pa2 = 6.283185307179586 * float(g3 >> 24u) / 255.0;
  float pe = 0.06 + 0.16 * float((g3 >> 16u) & 255u) / 255.0;
  float pell = 1.0 + (pd2 > 1e-9 ? pe * ((pdv.x * pdv.x - pdv.y * pdv.y) * cos(pa2) + 2.0 * pdv.x * pdv.y * sin(pa2)) / pd2 : 0.0);
  float d = sqrt(pd2) / (R * pell * (1.0 + 0.3 * (0.6 * vnoise(qq.x * 1.2, qq.y * 1.2, seed + 3403 + pns) + 0.4 * vnoise(qq.x * 3.3 + 7.7, qq.y * 3.3 - 2.9, seed + 3404 + pns))));
  if (d >= ${f(H.pondReach)}) return none;
  uint h2 = hash3(ix, iz, seed + 3402);
  vec2 w;
  domainWarp(c.x, c.y, seed, 0.0, w);
  Uplift up = upliftAt(w, seed);
  if (up.C < COAST_LEVEL + 0.02) return none;
  float wt[${NB}];
  vec4 clim;
  vec2 sp;
  regionsFrom(c, w, up, seed, wt, clim, sp);
  if (float(h2 & 0xffffu) / 65535.0 >= lfParam(wt, ${KI('pond')})) return none;
  // the land itself at its centre (the pond lies just under it) and round its rim: a pond needs flat land
  float ec = up.U + landDetail(c, w, up, lfParam(wt, ${KI('hills')}), lfParam(wt, ${KI('ridge')}), lfParam(wt, ${KI('karst')}),
    lfParam(wt, ${KI('dune')}), lfParam(wt, ${KI('mesa')}), lfParam(wt, ${KI('erosion')}), seed, 4.0) * up.taper;
  if (abs(landBare(c + 1.25 * R * vec2(1.0, 0.0), 4.0, seed) - ec) > 0.06 * R) return none;
  if (abs(landBare(c + 1.25 * R * vec2(-0.5, 0.8660254037844386), 4.0, seed) - ec) > 0.06 * R) return none;
  if (abs(landBare(c + 1.25 * R * vec2(-0.5, -0.8660254037844386), 4.0, seed) - ec) > 0.06 * R) return none;
  // away from rivers and lakes
  River rv = riverField(w, seed, 0.12);
  if (rv.xFloor < 2.0 * R + 40.0 || rv.lakeMask > 0.0) return none;
  return vec4(ec - ${f(H.pondSink)}, d, ${f(H.pondD[0])} + ${f(H.pondD[1] - H.pondD[0])} * float(h2 >> 16u) / 65535.0, R);
}

vec4 landSurface(vec2 g, float dp, int seed) { return landCore(g, dp, seed, pondAt(g, seed)); }

// ---- master elevation E(x, z) — spec §2 (mirror terrainMath.elevation) ---------
float elevation(vec2 g, float dp, int seed) {
  return landSurface(g, dp, seed).x;
}

// §17 — central differences at the pixel footprint (4 taps; the CPU twin queries ε = 0.25).
vec2 gradient(vec2 g, float eps, float dp, int seed) {
  float e = max(eps, 0.05);
  float inv2e = 0.5 / e;
  float gx = (elevation(g + vec2(e, 0.0), dp, seed) - elevation(g - vec2(e, 0.0), dp, seed)) * inv2e;
  float gz = (elevation(g + vec2(0.0, e), dp, seed) - elevation(g - vec2(0.0, e), dp, seed)) * inv2e;
  return vec2(gx, gz);
}

// ---- §9 climate ----------------------------------------------------------------
float temperature(vec2 g, float e, int seed) {
  vec2 w;
  domainWarp(g.x, g.y, seed, 0.0, w);
  return T_BASE + T_AMP * (vnoise(w.x * F_TEMP, w.y * F_TEMP, seed + 1201) + T_FINE * vnoise(w.x * F_TEMPFINE, w.y * F_TEMPFINE, seed + 1202))
    - L_LAT * abs(g.y) - L_ALT * max(e, 0.0);
}

// ---- the ground's biome look, per pixel (as the WebGPU bake + groundLook) --------------------------
// Albedo at world g (height e, slope, the water's level wl), and lava's glow into glow.
vec3 groundAlbedo(vec2 g, float e, float slope, float wl, int seed, out vec3 glow) {
  vec2 w;
  domainWarp(g.x, g.y, seed, 0.0, w);
  Uplift up = upliftAt(w, seed);
  float wt[${NB}];
  vec4 clim;
  vec2 sp;
  regionsFrom(g, w, up, seed, wt, clim, sp);
  vec3 soil = vec3(0.0), grass = vec3(0.0), dry = vec3(0.0), rock = vec3(0.0);
  vec4 a = vec4(0.0);
  float frost = 0.0;
  for (int k = 0; k < ${NB}; k++) {
    float wk = wt[k];
    soil += wk * LK_SOIL[k]; grass += wk * LK_GRASS[k]; dry += wk * LK_DRY[k]; rock += wk * LK_ROCK[k];
    a += wk * LK_A[k]; frost += wk * LK_FROST[k];
  }
  float patchN = 0.5 + 0.5 * vnoise(g.x * 0.045, g.y * 0.045, seed + 7777);
  float patchHue = 0.5 + 0.5 * vnoise(g.x * 0.033, g.y * 0.033, seed + 7707);
  vec3 grassCol = mix(grass, dry, clamp(patchHue * 1.3 - 0.45 + (0.5 - clim.y) * 0.8, 0.0, 1.0) * 0.75);
  float dens = a.x * smoothstep(0.15, 0.55, patchN + 0.25 * a.x) * (1.0 - smoothstep(0.7, 1.1, slope));
  vec3 col = mix(soil, grassCol * 0.8, clamp(dens * 1.2, 0.0, 0.9));
  float rockAmt = smoothstep(0.72, 1.15, slope + (1.0 - a.x) * 0.22);
  vec3 rk = rock;
  if (a.z > 0.01) {
    float hb = e / 7.5 + 0.6 * vnoise(g.x / 90.0, g.y / 90.0, 4441);
    float pick = fract(sin(floor(hb) * 12.9898) * 43758.547);
    vec3 tint = pick < 0.3 ? vec3(1.3, 1.18, 1.0) : (pick > 0.75 ? vec3(0.78, 0.66, 0.66) : vec3(1.0, 0.86, 0.72));
    rk = mix(rock, rock * tint, a.z);
  }
  col = mix(col, rk, rockAmt);
  float above = e - wl;
  float seaShore = 1.0 - smoothstep(SEA + 0.5, SEA + 1.5, wl);
  float beach = (1.0 - smoothstep(0.3, 1.8, above)) * (1.0 - smoothstep(0.35, 0.7, slope)) * (1.0 - 0.75 * a.w) * max(seaShore, 0.8 * (1.0 - a.x));
  col = mix(col, mix(vec3(0.6, 0.5, 0.34), soil, 0.3), beach * 0.85);
  col *= 1.0 - 0.4 * ((1.0 - smoothstep(0.0, 1.5, above)) * 0.55 + a.w * 0.35);
  float snowAlt = (clim.x - SNOW_T - frost) / L_ALT;
  float drift = vnoise(g.x / 23.0, g.y / 23.0, 4451) * 18.0 + vnoise(g.x / 7.0, g.y / 7.0, 4452) * 5.0;
  float snow = smoothstep(snowAlt - 15.0, snowAlt + 15.0, e + drift) * (1.0 - smoothstep(0.85, 1.35, slope));
  col = mix(col, vec3(${SNOW_ALBEDO.map(f).join(', ')}), snow);
  glow = vec3(0.0);
  // lava: the liquid of a volcano's crater lake and of its rivers (terrainMath.ts lastLava)
  if (up.vol.z > 0.0 && wl > e - 0.01) {
    float lava = 1.0 - smoothstep(0.45, 0.6, length(w - up.vol.xy) / (${f(V.cone)} * up.vol.z));
    if (lava > 0.01) {
      float crack = 1.0 - smoothstep(0.02, 0.16, voronoiEdge(w / 4.2, 973));
      col = mix(col, vec3(0.03, 0.02, 0.016), lava);
      glow = vec3(5.0, 1.15, 0.14) * lava * (0.12 + 1.1 * crack);
    }
  }
  return col;
}
`;
}
