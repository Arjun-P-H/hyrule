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

import { CLIMATE, F, GEO, HYDRO, LANDS } from '../../lib/constants';
import { BI, BIOMES, BIOME_COUNT, LANDFORM, LANDFORM_KEYS, LOOK, SNOW_ALBEDO } from '../../lib/biomes';
import { CANYON, CEIL_SPAN, CEIL_START, DRY_TABLE, ERO_STEP, GULLY_SLOPE, KARST_BANK, KARST_CEIL, KARST_CEIL_OFF, KARST_EDGE, KARST_EDGE_FADE, KARST_EDGE_LIFT, KARST_EDGE_STEP, KARST_NORMALS, KARST_TALUS, KARST_TURNS, LAND_MEANS, MESA, RIDGE_AMP, RIDGE_DAMP, RIDGE_FREQ, RIDGE_SCALE, RIDGE_SPAN, SEA_CLIFF } from '../../lib/terrainMath';
import { SIMPLEX_F2, SIMPLEX_G2, SIMPLEX_GRAD } from '../../lib/noise';
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
  const V = LANDS.volcano, G = LANDS.sacred, K = LANDS.coast, H = HYDRO, SC = SEA_CLIFF;
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
const float RIDGE_SCALE = ${f(RIDGE_SCALE)}, RIDGE_SPAN = ${f(RIDGE_SPAN)}, RIDGE_DAMP = ${f(RIDGE_DAMP)};
const float SIMPLEX_F2 = ${f(SIMPLEX_F2)}, SIMPLEX_G2 = ${f(SIMPLEX_G2)};
const vec2 SIMPLEX_GRAD[16] = vec2[16](${SIMPLEX_GRAD.map((g) => `vec2(${f(g[0])}, ${f(g[1])})`).join(', ')});
const vec2 KARST_NORMALS[${KARST_NORMALS.length}] = vec2[${KARST_NORMALS.length}](${KARST_NORMALS.map((v) => `vec2(${f(v[0])}, ${f(v[1])})`).join(', ')});
const vec2 KARST_TURNS[${KARST_TURNS.length}] = vec2[${KARST_TURNS.length}](${KARST_TURNS.map((v) => `vec2(${f(v[0])}, ${f(v[1])})`).join(', ')});
const float KARST_CEIL = ${f(KARST_CEIL)}, KARST_BANK = ${f(KARST_BANK)}, KARST_TALUS = ${f(KARST_TALUS)}, KARST_CEIL_OFF = ${f(KARST_CEIL_OFF)};
const float KARST_EDGE = ${f(KARST_EDGE)}, KARST_EDGE_LIFT = ${f(KARST_EDGE_LIFT)};
const vec2 KARST_EDGE_FADE = vec2(${KARST_EDGE_FADE.map(f).join(', ')});
const vec4 GULLY_SLOPE = vec4(${GULLY_SLOPE.map(f).join(', ')});
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
  vec2 b = w + ${f(GEO.rangeWarp)} * vec2(vnoise(w.x * F_RANGEWARP, w.y * F_RANGEWARP, seed + 3113), vnoise(w.x * F_RANGEWARP + 3.7, w.y * F_RANGEWARP - 1.9, seed + 3114));
  float ca = ${f(Math.cos(GEO.rangeDir[0]))}, sa = ${f(Math.sin(GEO.rangeDir[0]))};
  float cb = ${f(Math.cos(GEO.rangeDir[1]))}, sb = ${f(Math.sin(GEO.rangeDir[1]))};
  float c1 = 1.0 - abs(vnoise((b.x * ca + b.y * sa) * F_RANGEU, (b.y * ca - b.x * sa) * F_RANGEV, seed + 3111));
  float c2 = 1.0 - abs(vnoise((b.x * cb + b.y * sb) * F_RANGEU, (b.y * cb - b.x * sb) * F_RANGEV, seed + 3112));
  float c = mix(c1, c2, smoothstep(0.38, 0.62, n01(w, F_RANGEBLEND, seed + 3115)));
  float pk = ${f(GEO.pass)} + ${f(1 - GEO.pass)} * n01(w, F_PEAKS, seed + 3131);
  return rm * pow(c, ${f(GEO.rangePow)}) * pk;
}
float baseAt(vec2 w, int seed) {
  float low = ${f(GEO.low[0])} + ${f(GEO.low[1])} * n01(w, F_UPL, seed + 3101) + ${f(GEO.low[2])} * n01(w, F_UPL2, seed + 3102);
  float rng = rangeAt(w, seed);
  float ms = smoothstep(${f(GEO.massif[0])}, ${f(GEO.massif[1])}, n01(w, F_MASSIF, seed + 3141));
  float pn = 0.72 * n01(w, F_PLATCOARSE, seed + 1401) + 0.28 * n01(w, F_PLATFINE, seed + 1402);
  float pl = 0.82 * smoothstep(${f(GEO.plat)}, ${f(GEO.plat + GEO.platEdge)}, pn) + 0.18 * smoothstep(${f(GEO.plat - 0.05)}, ${f(GEO.plat)}, pn);
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
  float pn = 0.72 * n01(w, F_PLATCOARSE, seed + 1401) + 0.28 * n01(w, F_PLATFINE, seed + 1402);
  float pl = 0.82 * smoothstep(${f(GEO.plat)}, ${f(GEO.plat + GEO.platEdge)}, pn) + 0.18 * smoothstep(${f(GEO.plat - 0.05)}, ${f(GEO.plat)}, pn);
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
  float T0 = T_BASE + T_AMP * (vnoise(w.x * F_TEMP, w.y * F_TEMP, seed + 1201) + T_FINE * vnoise(w.x * F_TEMPFINE, w.y * F_TEMPFINE, seed + 1202))
    - L_LAT * abs(g.y);
  float C = up.C;
  float M = M_BASE + M_AMP * (vnoise(w.x * F_MOISTR, w.y * F_MOISTR, seed + 1301) + M_FINE * vnoise(w.x * F_MOISTFINE, w.y * F_MOISTFINE, seed + 1302))
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
  float bad = smoothstep(${f(LANDS.rugged[0])}, ${f(LANDS.rugged[1])}, n01(w, F_RUGGED, seed + 1501));
  float lake = smoothstep(${f(LANDS.lakedist[0])}, ${f(LANDS.lakedist[1])}, n01(w, F_LAKEDIST, seed + 1601));
  float karst = smoothstep(${f(LANDS.karst[0])}, ${f(LANDS.karst[1])}, n01(w, F_KARST, seed + 1701));
  float marsh = wet * smoothstep(${f(LANDS.marsh[0])}, ${f(LANDS.marsh[1])}, n01(w, F_MARSH, seed + 1801));
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
// Polynomial smooth max / min of a and b, the crease between them rounded over k (> 0).
float ksmax(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / k;
  return max(a, b) + 0.25 * h * h * k;
}
float ksmin(float a, float b, float k) {
  float h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - 0.25 * h * h * k;
}
// Karst towers at world g (mirror terrainMath.ts karstTowers): the highest of the seven-sided prisms
// (cap, wall, talus apron) of the 3 × 3 cells round g (m), and how much of g is tower.
vec2 karstTowers(vec2 g, int seed, float dp) {
  vec2 p = g * F_TOWER;
  vec2 ip = floor(p);
  float ws = clamp(dp, 1.0, 24.0), k2 = 4.0 * ws, k3 = 0.7 * ws + 1.5;
  float best = 0.0, cov = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      uint h2 = hash3(int(ip.x) + i, int(ip.y) + j, seed + 2302);
      if ((h2 >> 27u) < 4u) continue;
      uint h = hash3(int(ip.x) + i, int(ip.y) + j, seed + 2301);
      vec2 c = vec2(float(i), float(j)) + 0.2 + 0.6 * vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) * (1.0 / 4095.0);
      float H = 50.0 + 170.0 * float((h >> 28u) & 15u) / 15.0;
      float Rw = H / (2.0 * (0.7 + 0.8 * float((h >> 24u) & 15u) / 15.0));
      float Ht = (0.12 + 0.08 * float((h2 >> 18u) & 7u) / 7.0) * H;
      float Sw = 3.5 + 2.8 * float((h2 >> 21u) & 7u) / 7.0, St = 0.62 + 0.13 * float((h2 >> 24u) & 7u) / 7.0;
      float Rz = Rw + (Ht + k3) / St;
      vec2 u = (p - ip - c) / F_TOWER;
      if (dot(u, u) * 0.5184 >= Rz * Rz) continue;
      vec2 tr = KARST_TURNS[int(h2 & 15u)];
      vec2 v = vec2(tr.x * u.x + tr.y * u.y, tr.x * u.y - tr.y * u.x);
      float kr = 0.12 * Rw;
      float s = 0.0;
      for (int k = 0; k < ${KARST_NORMALS.length}; k++) {
        float d = dot(v, KARST_NORMALS[k]) / (0.8 + 0.4 * float((h2 >> uint(4 + 2 * k)) & 3u) / 3.0);
        s = k == 0 ? d : ksmax(s, d, kr);
      }
      float wall = Ht + Sw * (Rw - s), talus = Ht - St * (s - Rw);
      float Rtop = Rw - (H - Ht) / Sw;
      float cap = H - (0.6 * Rtop * s * s) / (4.0 * Rtop * Rtop + s * s);
      float pr = ksmin(cap, ksmax(wall, talus, k2), 5.0 * ws + 0.03 * H);
      best = max(best, ksmax(pr, 0.0, k3));
      cov = max(cov, smoothstep(0.5 * Ht, Ht + 0.1 * H, pr));
    }
  }
  return vec2(best, cov);
}

float dunesE(vec2 g, int seed, float dp) {
  float wMain = clamp(2.0 - 2.0 * dp * (2.0 * F_DUNE), 0.0, 1.0);
  float wCross = clamp(2.0 - 2.0 * dp * (2.0 * F_DUNECROSS), 0.0, 1.0);
  float amp = 0.3 + 0.7 * n01(g, F_DUNEAMP, seed + 2403);
  if (wMain <= 0.0) return DUNE_MEAN * amp;
  float ang = 0.6 + 0.9 * vnoise(g.x * F_DUNEFIELD, g.y * F_DUNEFIELD, seed + 2401);
  float ca = cos(ang), sa = sin(ang);
  float u = (g.x * ca + g.y * sa) * F_DUNE + 0.9 * vnoise(g.x * F_WARP, g.y * F_WARP, seed + 2402) + 0.35 * vnoise(g.x * F_DUNEBEND, g.y * F_DUNEBEND, seed + 2405);
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

// vnoise with its gradient (noise.ts vnoiseG): (value, d/dx, d/dy), the value vnoise's bit for bit.
vec3 vnoiseG(float x, float y, int seed) {
  vec2 p = vec2(x, y);
  vec2 ip = floor(p);
  vec2 fr = p - ip;
  vec2 u = fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * fr * fr * (fr * (fr - 2.0) + 1.0);
  uint ux0 = uint(int(ip.x));
  uint ux1 = ux0 + 1u;
  uint uy0 = uint(int(ip.y)) * 0x27d4eb2fu;
  uint uy1 = uy0 + 0x27d4eb2fu;
  uint s_term = uint(seed) * 0x9e3779b9u;
  uint h0 = uy0 ^ s_term;
  uint h1 = uy1 ^ s_term;
  uvec4 v = uvec4(ux0 ^ h0, ux1 ^ h0, ux0 ^ h1, ux1 ^ h1);
  v ^= v >> 16u; v *= 0x85ebca6bu;
  v ^= v >> 13u; v *= 0xc2b2ae35u;
  v ^= v >> 16u;
  vec4 abcd = vec4(v) * INV_U32_X2;
  vec2 ab_cd = mix(abcd.xz, abcd.yw, u.x);
  float k = abcd.x - abcd.y - abcd.z + abcd.w;
  return vec3(mix(ab_cd.x, ab_cd.y, u.y) - 1.0, du.x * ((abcd.y - abcd.x) + k * u.y), du.y * ((abcd.z - abcd.x) + k * u.x));
}
// domainWarp's Jacobian at world g (terrainMath.ts warpJ): (∂wx/∂x, ∂wx/∂z, ∂wz/∂x, ∂wz/∂z).
vec4 warpJ(vec2 g, int seed) {
  float wx = g.x * F_WARP, wy = g.y * F_WARP, k = S_WARP * F_WARP;
  vec2 a = vnoiseG(wx, wy, seed + 11).yz + vnoiseG(wx * 2.0, wy * 2.0, seed + 40514).yz;
  vec2 b = vnoiseG(wx + 5.2, wy + 1.3, seed + 37).yz + vnoiseG((wx + 5.2) * 2.0, (wy + 1.3) * 2.0, seed + 40540).yz;
  return vec4(1.0 + k * a.x, k * a.y, k * b.x, 1.0 + k * b.y);
}

// Mesas and buttes (terrainMath.ts MESA): the tiers (n's threshold, height m).
const vec2 MESA_TIERS[3] = vec2[3](${MESA.tiers.map((t) => `vec2(${f(t[0])}, ${f(t[1])})`).join(', ')});
// The mesas' plan noise n at world g.
float mesaPlan(vec2 g, int seed) {
  return 0.5 + 0.35 * vnoise(g.x * F_MESA, g.y * F_MESA, seed + 2501) + 0.15 * vnoise(g.x * F_MESAFINE, g.y * F_MESAFINE, seed + 2502);
}
// The mesas' and the karst's shares of the landform at world g (their own warp, uplift and regions).
vec2 sharesAt(vec2 g, int seed) {
  vec2 w;
  domainWarp(g.x, g.y, seed, 0.0, w);
  float wt[${NB}];
  vec4 clim;
  vec2 sp;
  regionsFrom(g, w, upliftAt(w, seed), seed, wt, clim, sp);
  return vec2(lfParam(wt, ${KI('mesa')}), lfParam(wt, ${KI('karst')}));
}
// The towers' ceiling u m in from where it starts, over a point that is tower by c (terrainMath.ts karstCeiling).
float karstCeiling(float u, float c) {
  float ct = KARST_TALUS * u;
  return ct + max(0.0, KARST_CEIL * u - KARST_CEIL_OFF - ct) * c;
}
// The mesas at world g where their region weighs mw, its gradient mg, the weight applied, and into cov
// how much of g is mesa (mirror terrainMath.ts mesasE: each tier a cap, a wall and a talus apron laid
// out by the distance in from its contour, refined by a secant step).
float mesasE(vec2 g, int seed, float dp, float mw, vec2 mg, out float cov) {
  cov = 0.0;
  float hm = smoothstep(0.001, ${f(MESA.rise)}, mw);
  if (hm <= 0.0) return 0.0;
  vec3 a = vnoiseG(g.x * F_MESA, g.y * F_MESA, seed + 2501);
  vec3 b = vnoiseG(g.x * F_MESAFINE, g.y * F_MESAFINE, seed + 2502);
  float n = 0.5 + 0.35 * a.x + 0.15 * b.x;
  vec2 gr = 0.35 * F_MESA * a.yz + 0.15 * F_MESAFINE * b.yz;
  // (the tiers retreat with the region's weight, along its gradient: landDetail's)
  gr += ${f(MESA.retreat)} * mg;
  float gl = max(length(gr), 1e-7);
  vec2 u = gr / gl;
  float mu = ${f(MESA.retreat)} * dot(mg, u);
  float hd0 = n01(g, F_MESA, seed + 2503);
  float lift = ${f(MESA.retreat)} * (1.0 - mw);
  float ws = max(dp, 1.0);
  float k3 = (0.7 * ws + 1.5) * hm + 1e-3;
  float e = 0.0, top = 0.0;
  for (int i = 0; i < 3; i++) {
    float H = MESA_TIERS[i].y * hm;
    float Cp = top, C = top + H;
    top = C;
    float ei = n - MESA_TIERS[i].x - lift;
    float d = ei / gl;
    if (d > -${f(MESA.reach[1])} && d < ${f(MESA.reach[0])}) {
      float nq = mesaPlan(g - d * u, seed);
      float ds = d / clamp((n - nq + d * mu) / (abs(ei) > 1e-12 ? ei : 1e-12), 0.5, 2.0);
      d = ds + (d - ds) * smoothstep(0.75, 1.0, d > 0.0 ? d / ${f(MESA.reach[0])} : -d / ${f(MESA.reach[1])});
    }
    float hd = clamp(hd0 + ${f(MESA.harder)} * float(i - 1), 0.0, 1.0);
    float Sw = ${f(MESA.cliff[0])} + ${f(MESA.cliff[1] - MESA.cliff[0])} * hd;
    float hw = (${f(MESA.wall[0])} + ${f(MESA.wall[1] - MESA.wall[0])} * hd) * H;
    float St = ${f(MESA.talus[0])} + ${f(MESA.talus[1] - MESA.talus[0])} * hd;
    float cr = ${f(MESA.caprock[1])} * hw;
    // (the wall no steeper than its footprint can show)
    float m = min(1.0, (cr / ${f(MESA.caprock[0])} + (hw - cr) / Sw) / (1.5 * ws));
    float sc = ${f(MESA.caprock[0])} * m, sw = Sw * m;
    float foot = -(cr / sc + (hw - cr) / sw);
    float wall = ksmax(C + sc * d, C - cr + sw * (d + cr / sc), ws);
    float talus = C - hw + St * (d - foot);
    float T = ksmin(C, ksmax(wall, talus, 3.0 * ws), 3.0 * ws + 0.02 * H);
    e += ksmax(T - Cp, 0.0, k3);
    cov = max(cov, smoothstep(foot - 0.5 * (H - hw) / St, foot, d));
  }
  cov *= hm;
  return e;
}

// Canyon walls (terrainMath.ts CANYON): a layer's profile, bottom, thickness, and its three parts' runs
// (shares) and D at their ends.
struct CLayer { float b, H, ac, ab, at, d1, d2; };
// Layer i's boundary (m).
float canyonB(int i, int seed) {
  return ${f(CANYON.layer)} * (float(i) + ${f(CANYON.jitter)} * (float(hash3(i, 0, seed + 2611) & 0xfffu) / 4095.0 - 0.5));
}
// A layer from b, H thick, of hardness hd, on a wall of slope s.
CLayer canyonProfile(float b, float H, float hd, float s) {
  float sc = ${f(CANYON.cliff[0])} + ${f(CANYON.cliff[1] - CANYON.cliff[0])} * hd;
  float st = min(${f(CANYON.talus[0])} + ${f(CANYON.talus[1] - CANYON.talus[0])} * hd, s);
  float rc = ${f(CANYON.share[0])} + ${f(CANYON.share[1] - CANYON.share[0])} * hd;
  float sb = ${f(CANYON.bench)};
  float A = rc / sc + (1.0 - rc) / st;
  float pb = clamp((1.0 / s - A) / (1.0 / sb - A), 0.0, 0.9);
  float pc = rc * (1.0 - pb), pt = (1.0 - rc) * (1.0 - pb);
  vec3 r = vec3(pc * s / sc, pb * s / sb, pt * s / st);
  r /= r.x + r.y + r.z;
  return CLayer(b, H, r.x, r.y, r.z, pc - r.x, pc + pb - r.x - r.y);
}
// Layer i's profile on a wall of slope s.
CLayer canyonLayer(int i, float s, int seed) {
  uint h = hash3(i, 0, seed + 2611);
  float b = ${f(CANYON.layer)} * (float(i) + ${f(CANYON.jitter)} * (float(h & 0xfffu) / 4095.0 - 0.5));
  return canyonProfile(b, canyonB(i + 1, seed) - b, float((h >> 12u) & 0xfffu) / 4095.0, s);
}
// ∫₀ᵗ D over layer c (t its share of the layer's run).
float canyonJ(CLayer c, float t) {
  if (t <= c.ac) return c.d1 * t * t / (2.0 * max(c.ac, 1e-6));
  float j1 = 0.5 * c.d1 * c.ac;
  if (t <= c.ac + c.ab) {
    float u = t - c.ac;
    return j1 + c.d1 * u + (c.d2 - c.d1) * u * u / (2.0 * max(c.ab, 1e-6));
  }
  float u = min(t, 1.0) - c.ac - c.ab;
  return j1 + 0.5 * (c.d1 + c.d2) * c.ab + c.d2 * u - c.d2 * u * u / (2.0 * max(c.at, 1e-6));
}
// The layer holding height h (m).
int canyonAt(float h, int seed) {
  int i = int(floor(h / ${f(CANYON.layer)}));
  if (h < canyonB(i, seed)) return i - 1;
  return h >= canyonB(i + 1, seed) ? i + 1 : i;
}
// The canyon's raise (m, ≥ 0) at height h on a wall of slope s, box-filtered over heights h ± w/2; over
// more than a layer it fades to the strata's mean (mirror terrainMath.ts canyonRaise).
float canyonRaise(float h, float s, float w, int seed) {
  float L = ${f(CANYON.layer)};
  float fl = smoothstep(0.72 * L, 1.2 * L, w);
  float r = 0.0;
  if (fl < 1.0) {
    if (w < 1e-3) {
      CLayer c = canyonLayer(canyonAt(h, seed), s, seed);
      float t = (h - c.b) / c.H;
      float D = c.d2 * (1.0 - t) / max(c.at, 1e-6);
      if (t <= c.ac) D = c.d1 * t / max(c.ac, 1e-6);
      else if (t <= c.ac + c.ab) D = c.d1 + (c.d2 - c.d1) * (t - c.ac) / max(c.ab, 1e-6);
      r = c.H * D;
    } else {
      float h1 = h - 0.5 * w, h2 = h + 0.5 * w;
      int i1 = canyonAt(h1, seed), i2 = canyonAt(h2, seed);
      CLayer c = canyonLayer(i1, s, seed);
      float I = -c.H * c.H * canyonJ(c, (h1 - c.b) / c.H);
      for (int j = 0; j < 5; j++) {
        if (i1 + j >= i2) break;
        I += c.H * c.H * canyonJ(c, 1.0);
        c = canyonLayer(i1 + j + 1, s, seed);
      }
      I += c.H * c.H * canyonJ(c, (h2 - c.b) / c.H);
      r = I / w;
    }
  }
  if (fl > 0.0) r += (L * canyonJ(canyonProfile(0.0, L, 0.5, s), 1.0) - r) * fl;
  return r;
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
// 2-D simplex noise (noise.ts snoise): (value, d/dx, d/dy).
vec3 snoiseCorner(vec2 d, uint h) {
  float r = 0.5 - dot(d, d);
  if (r <= 0.0) return vec3(0.0);
  vec2 g = SIMPLEX_GRAD[h >> 28];
  float gd = dot(g, d), r2 = r * r, r4 = r2 * r2;
  return vec3(r4 * gd, r4 * g - 8.0 * r2 * r * gd * d);
}
vec3 snoise(float x, float y, int seed) {
  vec2 p = vec2(x, y);
  vec2 ip = floor(p + (x + y) * SIMPLEX_F2);
  vec2 d0 = p - ip + (ip.x + ip.y) * SIMPLEX_G2;
  ivec2 o = d0.x > d0.y ? ivec2(1, 0) : ivec2(0, 1);
  vec2 d1 = d0 - vec2(o) + SIMPLEX_G2;
  vec2 d2 = d0 - 1.0 + 2.0 * SIMPLEX_G2;
  int ix = int(ip.x), iy = int(ip.y);
  return snoiseCorner(d0, hash3(ix, iy, seed)) + snoiseCorner(d1, hash3(ix + o.x, iy + o.y, seed))
    + snoiseCorner(d2, hash3(ix + 1, iy + 1, seed));
}
// The mountains' ridged octaves: simplex, each octave damped by the slope of those under it (terrainMath.ts ridgesAt).
float ridgesAt(vec2 w, int seed, float dp) {
  vec2 rw = w * F_RIDGE * RIDGE_SPAN;
  float kr = 4.0 * F_RIDGE * dp;
  float rh = 0.0, slope2 = 0.0;
  for (int i = 0; i < 5; i++) {
    float fr = RIDGE_FREQS[i];
    float wgt = dp <= 0.0 ? 1.0 : clamp(2.0 - fr * kr, 0.0, 1.0);
    if (wgt > 0.0) {
      vec3 sn = snoise(rw.x * fr, rw.y * fr, seed + 5 + i * 34283);
      float n = 1.0 - abs(RIDGE_SCALE * sn.x);
      float a = RIDGE_AMPS[i] * wgt / (1.0 + RIDGE_DAMP * slope2);
      rh += a * (n * n - RIDGE_MEAN);
      float g = 2.0 * n * a * fr * RIDGE_SPAN * RIDGE_SCALE * length(sn.yz);
      slope2 += g * g;
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
struct River { float R, F, xFloor, W, ch, w, depth, acc, Ua, fall, L, lakeK, lakeMask, vis, cap, Lnear; vec2 dir; vec3 gt; };
// A channel's half width where the drainage is acc (m; mirror hydro.ts riverHalfW).
float riverHalfW(float acc) {
  return 0.5 * min(${f(H.widthMax)}, ${f(H.width[0])} + ${f(H.width[1])} * pow(max(acc - 1.0, 1e-6), ${f(H.width[2])}));
}
// The meander's offset (m) from a reach's line at along metres from its head, t its share of the reach.
float meanderOff(float along, float t, float amp, int sg) {
  if (amp <= 0.0) return 0.0;
  return amp * smoothstep(0.0, 0.12, t) * smoothstep(1.0, 0.88, t) * vnoise(along * F_MEANDER, 0.5, sg);
}
River riverField(vec2 w, int seed, float k) {
  River o;
  o.R = 1e9; o.F = -1e9; o.xFloor = 1e9; o.W = -1e9; o.ch = 1e9; o.w = 0.0; o.depth = 0.0; o.acc = 0.0; o.Ua = 0.0;
  o.dir = vec2(0.0); o.fall = 0.0; o.L = -1e9; o.lakeK = 0.0; o.lakeMask = 0.0; o.vis = 0.0; o.cap = 1e9; o.Lnear = -1e9;
  o.gt = vec3(0.0);
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
        float rl = rr * (1.0 + 0.28 * vnoise(dv.x / rr * 1.3, dv.y / rr * 1.3, seed + 3503));
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
      float wd = riverHalfW(acc + clamp(t, 0.0, 1.0))
        * (1.0 + ${f(H.widthVary)} * vnoise(along * F_MEANDER * 2.3, 7.5, int(hash3(i, j, seed + 3302) & 0xffffu)));
      float vis = a1.w; // (the river shows only below a lake: hydro.ts fedOf)
      // (a river whose water does not show cuts no channel: a dry wash, not a ditch)
      float depth = (${f(H.depth[0])} + ${f(H.depth[1])} * sqrt(acc)) * vis;
      float slope = (Wa - Wb) / len;
      float amp = ${f(H.meanderMin)} + ${f(H.meander - H.meanderMin)} * (1.0 - smoothstep(0.012, 0.05, (a0.z - b0.z) / len));
      float floorW = (vis > 0.0 ? amp + wd : 0.0) + ${f(H.floor[0])} + ${f(H.floor[1])} * sqrt(acc);
      vec4 p0 = hyAt(i, j, 2), p1 = hyAt(i, j, 3), p2 = hyAt(i, j, 4), p3 = hyAt(i, j, 5);
      float dd, ch;
      if (t <= 0.0) { dd = length(r); ch = dd; }
      else if (t >= 1.0) { dd = length(q - b0.xy); ch = dd; }
      else {
        float ds = -r.x * u.y + r.y * u.x;
        dd = abs(ds);
        if (dd - floorW > 400.0) {
          float x = dd - floorW;
          vec4 lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
          float lvl = lf.y;
          float cone = lvl + wallRise(k, x, floorW);
          o.R = min(o.R, cone);
          o.F = max(o.F, lvl + ${f(H.floorRise)} - ${f(H.riser)} * x);
          o.xFloor = min(o.xFloor, x);
          if (nR < 34) {
            sA[nR] = vec4(cone, wallSlope(k, x, floorW), lf.w / len, ${f(H.wallCurve)});
            sB[nR] = vec4(-u.y, u.x, 0.1, lvl);
            sN[nR] = ivec2(j * 4096 + i, bj * 4096 + bi);
            nR++;
          }
          continue;
        }
        ch = abs(ds - meanderOff(along, t, amp, sg));
      }
      vec4 lf = reachLF(Wa, p0, p1, p2, p3, Wb, t);
      float Wt = lf.x;
      float Wv = vis > 0.0 ? mix(Wt, lf.y, smoothstep(floorW, floorW + 60.0, dd)) : lf.y;
      // (a dry valley has no channel: hydro.ts riverField)
      float chV = vis > 0.0 ? ch : 1e9;
      float wV = vis > 0.0 ? wd : 0.0;
      float cone = Wv + valleyShape(dd, chV, wV, depth, floorW, k);
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
        else { sA[nR] = vec4(cone, valleySlope(dd, chV, wV, depth, floorW, k), fallT, ${f(H.wallCurve)}); sB[nR] = vec4(nrm, dd < floorW + 60.0 ? 0.35 : 0.1, min(Wt, Wv)); }
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
    float cone = L4.z + valleyShape(L4.x, L4.x, L4.y, L4.w, fl, k);
    o.R = min(o.R, cone);
    o.xFloor = min(o.xFloor, L4.x - fl);
    if (nR < 34) {
      sA[nR] = L4.x < L4.y ? vec4(max(cone, L4.z), 0.0, 0.0, ${f(H.wallCurve)}) : vec4(cone, valleySlope(L4.x, L4.x, L4.y, L4.w, fl, k), 0.0, ${f(H.wallCurve)});
      sB[nR] = vec4(lkN[m], 0.3, L4.z);
      sN[nR] = ivec2(-1, -2);
      nR++;
    }
  }
  // the walls' slope: each reach's slope across, weighted by a soft minimum over the surfaces (hydro.ts riverField)
  float gw = 0.0;
  vec3 gt = vec3(0.0);
  for (int r = 0; r < 34; r++) {
    if (r >= nR) break;
    float d = sA[r].x - o.R;
    if (d > ${f(10 * H.wallSoft)}) continue;
    float wt = exp(-d / ${f(H.wallSoft)});
    vec2 nn = sB[r].xy;
    gw += wt;
    gt += wt * sA[r].y * sA[r].y * vec3(nn.x * nn.x, nn.x * nn.y, nn.y * nn.y);
  }
  if (gw > 0.0) o.gt = gt / gw;
  // the nearest channel's water stays under every other reach's surface out to its far bank (that surface
  // falls toward the bank by its slope along the line across the channel, but never below its own river)
  if (best >= 0) {
    float D = o.w + o.ch + 1.0;
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

// The regions' detail over the uplift at world g, warped w, before any mask, and into kt the towers' height
// (mirror terrainMath.ts landDetail: with river the towers are left out of the detail).
float landDetail(vec2 g, vec2 w, Uplift up, float hills, float ridge, float karst, float dune, float mesa, float eros, int seed, float dp, bool river, out vec2 kt) {
  float detail = hills * hillsAt(w * F_MACRO, seed, dp);
  float rs = min(1.0, 2.0 * up.range + 1.6 * up.massif);
  if (ridge > 0.0) detail += ridge * (0.3 + 0.7 * rs) * ridgesAt(w, seed, dp);
  float kw = smoothstep(0.35, 0.75, karst);
  float mw = smoothstep(0.35, 0.75, mesa);
  float wK = clamp(2.0 - 2.0 * dp / 150.0, 0.0, 1.0);
  // (the karst's share's and the mesas' weight's gradients where not yet 1: terrainMath.ts landDetail)
  bool fdK = kw > 0.001 && wK > 0.0 && karst < KARST_EDGE_FADE.y, fdM = mw > 0.001 && mesa < 0.85;
  vec2 kg = vec2(0.0), mg = vec2(0.0);
  if (fdK || fdM) {
    float far = length(g) / ${f(MESA.fdSpan)};
    // (the mesas' two steps, then the karst's, in one loop)
    for (int k = 0; k < 4; k++) {
      bool isK = k >= 2;
      if (isK ? !fdK : !fdM) continue;
      float h = isK ? max(${f(KARST_EDGE_STEP)}, far) : max(1.0, far);
      bool zs = (k & 1) == 1;
      vec2 sh = sharesAt(g + (zs ? vec2(0.0, h) : vec2(h, 0.0)), seed);
      float d = isK ? (sh.y - karst) / h : (smoothstep(0.35, 0.75, sh.x) - mw) / h;
      if (isK) { if (zs) kg.y = d; else kg.x = d; } else { if (zs) mg.y = d; else mg.x = d; }
    }
  }
  float kc = 0.0;
  kt = vec2(0.0); // (the towers' height, and how much of it is tower)
  if (kw > 0.001) {
    float t = kw * TOWER_MEAN;
    if (wK > 0.0) {
      vec2 kk = karstTowers(g, seed, dp);
      kt.y = wK * kk.y;
      // (whole within their region, cut at its edge under a ceiling: terrainMath.ts KARST_EDGE)
      float k = kk.x, cut = 1.0;
      if (fdK) {
        float u = (karst - KARST_EDGE) / max(length(kg), 1e-9) + KARST_EDGE_LIFT * smoothstep(KARST_EDGE_FADE.x, KARST_EDGE_FADE.y, karst);
        k = max(0.0, min(k, karstCeiling(u, kt.y)));
        cut = clamp(0.1 * u, 0.0, 1.0);
      }
      t = wK * k + (1.0 - wK) * t;
      kc = cut * kt.y;
    }
    kt.x = t;
    if (!river) detail += kt.x;
  }
  float dw = smoothstep(0.35, 0.75, dune);
  if (dw > 0.001) detail += dw * dunesE(g, seed, dp);
  if (mw > 0.001) { float mc; detail += mesasE(g, seed, dp, mw, mg, mc); kc = max(kc, mc); }
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
    // (only on flanks and aprons: terrainMath.ts landDetail)
    float S = length(gr);
    float keep = smoothstep(GULLY_SLOPE.x, GULLY_SLOPE.y, S) * (1.0 - smoothstep(GULLY_SLOPE.z, GULLY_SLOPE.w, S)) * (1.0 - kc);
    if (keep > 0.0) detail += eros * keep * erosionDelta(g, up.U + detail, gr, 0.0, dp); // (no fade target: terrainMath.ts landDetail)
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
  float crater = up.lava > -1e8 ? smoothstep(${f(CRATER_RIM * 0.9)}, ${f(CRATER_RIM * 1.3)}, up.rr) : 1.0;
  float m = smoothstep(0.0, 50.0, rv.xFloor) * (1.0 - rv.lakeMask) * crater;
  // where marshes lie (not in a lake's ring, nor in a river's channel; only on flat land)
  float flat2 = flat_ * (1.0 - rv.lakeMask);
  if (rv.vis > 0.0) flat2 *= smoothstep(rv.w + 2.0, rv.w + 8.0, rv.ch);
  if (flat2 > 0.0) {
    vec2 gU = vec2(upliftAt(w + vec2(8.0, 0.0), seed).U - up.U, upliftAt(w + vec2(0.0, 8.0), seed).U - up.U) / 8.0;
    flat2 *= 1.0 - smoothstep(0.012, 0.035, length(gU));
  }
  float detail = 0.0;
  vec2 kt = vec2(0.0);
  if (m > 0.0) {
    detail = landDetail(g, w, up, hills, ridge, karst, dune, mesa, eros, seed, dp, true, kt);
    detail *= m * up.taper;
    // (round a river the detail only rises: a hollow beside it would leave the river standing over it)
    float near = 1.0 - smoothstep(50.0, 450.0, rv.xFloor);
    if (near > 0.0) detail += (0.5 * (detail + sqrt(detail * detail + 36.0)) - 3.0 - detail) * near;
    // (and round a marsh: its pools lie just under the smooth land, and a hollow at its edge would spill them)
    float nearM = smoothstep(0.0, 0.3, flat2);
    if (nearM > 0.0) detail += (0.5 * (detail + sqrt(detail * detail + 0.36)) - 0.3 - detail) * nearM;
  }
  float e0 = up.U + detail;
  // the canyons: where mesas rule, a valley's steep walls cut through the strata (terrainMath.ts landCore)
  float Rs = rv.R;
  float mwC = smoothstep(0.35, 0.75, mesa);
  float cwC = mwC > 0.0 && rv.R < max(e0, rv.F) ? mwC * smoothstep(${f(CANYON.rim[0])}, ${f(CANYON.rim[1])}, rv.xFloor) * (1.0 - rv.lakeMask) : 0.0;
  // (the towers' ceiling and the canyons' walls measured in the world, the warp's metric J Jᵀ: terrainMath.ts landCore)
  vec3 M = vec3(1.0, 0.0, 1.0);
  if (kt.x > 0.0 || cwC > 0.0) {
    vec4 J = warpJ(g, seed);
    M = vec3(J.x * J.x + J.y * J.y, J.x * J.z + J.y * J.w, J.z * J.z + J.w * J.w);
  }
  // (the towers under the ceiling a river's floor puts over them: terrainMath.ts landCore)
  float towers = 0.0;
  if (kt.x > 0.0) {
    float h = 0.5 * (M.x - M.z);
    float sig = sqrt(0.5 * (M.x + M.z) + sqrt(h * h + M.y * M.y));
    float u = (min(rv.xFloor, 1000.0) - KARST_BANK) / sig;
    towers = max(0.0, min(kt.x, karstCeiling(u, kt.y) * (1.0 - rv.lakeMask))) * up.taper * crater;
  }
  if (cwC > 0.0) {
    float s = sqrt(max(0.0, M.x * rv.gt.x + 2.0 * M.y * rv.gt.y + M.z * rv.gt.z));
    cwC *= smoothstep(${f(CANYON.steep[0])}, ${f(CANYON.steep[1])}, s);
    if (cwC > 0.0) {
      float dip = ${f(CANYON.dip)} * vnoise(g.x * F_MESA, g.y * F_MESA, seed + 2613);
      Rs = rv.R + cwC * canyonRaise(rv.R + dip, s, s * max(dp, 0.5), seed);
    }
  }
  e0 = min(max(e0, rv.F), Rs) + towers;
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
  // the coast, the sea floor; a cliffed coast cut in one profile (terrainMath.ts SEA_CLIFF)
  float land = smoothstep(COAST_LEVEL - 0.004 - 0.03 * (1.0 - up.cliff), COAST_LEVEL + 0.002, up.C);
  float seaFloor = SEA - 6.0 - ${f(K.shelf)} * (1.0 - smoothstep(COAST_LEVEL - 0.12, COAST_LEVEL, up.C));
  float e = seaFloor + (e0 - seaFloor) * land;
  float cw = smoothstep(${f(SC.weight[0])}, ${f(SC.weight[1])}, up.cliff) * (1.0 - smoothstep(${f(0.6 * SC.band)}, ${f(SC.band)}, abs(up.C - COAST_LEVEL)));
  if (cw > 0.0) {
    // the distance in from the coastline: C's rise over its world gradient
    vec2 wX, wZ;
    domainWarp(g.x + ${f(SC.h)}, g.y, seed, 0.0, wX);
    domainWarp(g.x, g.y + ${f(SC.h)}, seed, 0.0, wZ);
    float cX = continentAt(wX, seed), cZ = continentAt(wZ, seed);
    float d = (up.C - COAST_LEVEL) * ${f(SC.h)} / max(length(vec2(cX - up.C, cZ - up.C)), 1e-9);
    float apron = ${f(SC.apron)} * smoothstep(0.45, 0.85, n01(w, F_TOWER, seed + 2703));
    float d0 = d - apron - ${f(SC.toe)};
    float base = SEA - ${f(SC.platform)};
    float wall = base + ${f(SC.wall)} * d0, talus = base + ${f(SC.talus)} * (d0 + apron);
    float flr = max(seaFloor, base + ${f(SC.shelf)} * min(d, 0.0)) - ${f(SC.wall)} * max(d0, 0.0);
    float kr = ${f(SC.rim)} + ${f(SC.wall)} * dp, kt = 1.0 + ${f(SC.wall)} * dp, kf = 0.3 + ${f(SC.talus)} * dp;
    float eC = ksmax(flr, ksmin(e0, ksmax(talus, wall, kt), kr), kf);
    e += (eC - e) * cw;
    land += (smoothstep(0.0, apron + ${f(SC.toe)}, d) - land) * cw;
  }
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
  vec2 kt;
  return up.U + landDetail(g, w, up, lfParam(wt, ${KI('hills')}), lfParam(wt, ${KI('ridge')}), lfParam(wt, ${KI('karst')}),
    lfParam(wt, ${KI('dune')}), lfParam(wt, ${KI('mesa')}), lfParam(wt, ${KI('erosion')}), seed, dp, false, kt) * up.taper;
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
  float inset = ${f(H.pondReach * 1.3)} * R / cell;
  vec2 c = (vec2(float(ix), float(iz)) + inset + (1.0 - 2.0 * inset) * vec2(float(h & 0xfffu), float((h >> 12u) & 0xfffu)) / 4095.0) * cell;
  vec2 qq = (g - c) / R;
  float d = length(g - c) / (R * (1.0 + 0.3 * vnoise(qq.x * 1.2, qq.y * 1.2, seed + 3403)));
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
  vec2 kt;
  float ec = up.U + landDetail(c, w, up, lfParam(wt, ${KI('hills')}), lfParam(wt, ${KI('ridge')}), lfParam(wt, ${KI('karst')}),
    lfParam(wt, ${KI('dune')}), lfParam(wt, ${KI('mesa')}), lfParam(wt, ${KI('erosion')}), seed, 4.0, false, kt) * up.taper;
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
