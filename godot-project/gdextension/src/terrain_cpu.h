/**
 * ============================================================================
 *  W(x,y) — Continuous World Engine · CPU physics twin
 *  Godot 4 GDExtension · Apple Silicon (arm64) · Metal (TBDR) · Forward+ SDFGI
 * ============================================================================
 *
 *  THE PARITY CONTRACT (spec §27, §31)
 *  -----------------------------------
 *  This header is the double-precision reference implementation of the SAME
 *  field the GPU raymarcher draws (shaders/terrain_raymarch.gdshader):
 *
 *    - Hashing is strict uint32 integer math. Identical bit-for-bit on CPU
 *      (uint32_t), GPU (GLSL uint), GDScript fallback (int64 masked to 32
 *      bits) and the WebGL2 harness (Math.imul). The MurmurHash3 finalizer
 *      avalanche makes the lattice deterministic for a given seed anywhere
 *      in the infinite plane.
 *
 *    - The CPU evaluates GLOBAL coordinates in double precision. The GPU
 *      evaluates (chunk_local_fp32 + chunk_origin_fp32). The floating-origin
 *      chunk size is 8192 = 2^13 and every spatial frequency in the contract
 *      is a dyadic rational n/2^20, so `origin * f` is exactly representable
 *      in fp32 and noise lattices align across chunk borders. Residual
 *      divergence between the two evaluation paths is < 1 mm on a 700 m peak
 *      and is absorbed by GROUND_BIAS = 0.08 m (§31).
 *
 *    - JUMP-PRESERVING GROUND CLAMP (§31): gravity always integrates; the
 *      clamp only ever zeroes NEGATIVE vertical velocity, so an upward jump
 *      is never killed by the contact test. Slope gate slides the player off
 *      gradients steeper than SLOPE_MAX = 1.5.
 *
 *  WHY A CPU TWIN AT ALL (spec §31)
 *  --------------------------------
 *  The GPU surface exists only as per-pixel math inside the fragment shader;
 *  it is NOT a mesh, so the engine physics servers cannot see it. Gameplay
 *  (walking, jumping, arrows, spawn queries, shrine detection) therefore
 *  queries this twin. The GPU writes DEPTH/NORMAL per pixel so that ordinary
 *  meshes depth-test and SDFGI-voxelize against the terrain seamlessly.
 *
 *  APPLE-SILICON NOTES (Metal, TBDR)
 *  ---------------------------------
 *  This file is pure CPU math — but the reasons it looks the way it does are
 *  GPU-driven: the per-octave Nyquist weights mirror the bandwidth the tile
 *  renderer actually resolves per pixel, and the analytic slope budget used
 *  by the shader's Lipschitz stepper is derived from these amplitudes. On
 *  TBDR hardware every fragment is paid for per-tile, so the twin's cheap
 *  gradient (2 extra field taps) is what keeps the shader's step sizes safe
 *  without a full central difference at every march sample.
 *
 *  Build with `-ffp-contract=off` (see SConstruct): FMA contraction differs
 *  between the CPU compiler and the Metal toolchain and would break the
 *  ≤ 1 mm parity budget for no performance gain in this hot path.
 * ============================================================================
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace terrain {

/* ========================================================================== */
/*  Contract constants — MIRROR EXACTLY:                                      */
/*    src/lib/constants.ts (web harness) · demoShader.ts (WebGL2)             */
/*    shaders/terrain_raymarch.gdshader (Godot) · scripts/terrain_math_fallback.gd */
/* ========================================================================== */

// ---- world / precision (§16, §25, §26) ------------------------------------
constexpr double CHUNK_SIZE   = 8192.0;  // 2^13 floating-origin block
constexpr double MAX_WORLD_H  = 1300.0;  // early-exit ceiling
constexpr double MAX_DIST     = 6000.0;  // march cutoff (hidden by fog)

// ---- spatial frequencies — dyadic rationals n/2^20 (§25/§27) --------------
constexpr double F_MACRO   = 512.0  / 1048576.0; // 1/2048     rolling hills
constexpr double F_RIDGE   = 341.0  / 1048576.0; // ~1/3074.6  ridged mountains
constexpr double F_MASK    = 128.0  / 1048576.0; // 1/8192     mountain mask
constexpr double F_WARP    = 655.0  / 1048576.0; // ~1/1600.9  domain warp
constexpr double F_VALLEY  = 1942.0 / 1048576.0; // ~1/539.9   cellular valleys
constexpr double F_CONT    = 64.0   / 1048576.0; // 1/16384    continent field
constexpr double F_MOIST   = 512.0  / 1048576.0; // 1/2048     moisture noise
constexpr double F_DUNE    = 23.0   / 1048576.0; // ~45.6 m    dune wavelength
constexpr double F_CLOUD   = 403.0  / 1048576.0; // ~1/2601.9  cloud coverage
constexpr double VEG_WL    = 14.0;               // 14 m       veg clumps

// ---- macro terrain (§5) ----------------------------------------------------
constexpr int    OCT_MACRO = 9;
constexpr double LAC_MACRO = 2.0,  GAIN_MACRO = 0.5,  H_BASE = 90.0;

// ---- mountains (§6 — ridged multifractal, Nyquist weight uses 2f) ----------
constexpr int    OCT_RIDGE = 5;
constexpr double LAC_RIDGE = 2.17, GAIN_RIDGE = 0.48, H_MOUNT = 880.0;
constexpr double MASK_LO = 0.28,  MASK_HI = 0.60;

// ---- domain warp (§14) -----------------------------------------------------
constexpr int    OCT_WARP  = 2;
constexpr double S_WARP    = 620.0;

// ---- valleys / rivers (§7, §8) ---------------------------------------------
constexpr double C_DEPTH  = 48.0,  W_VALLEY = 0.42;
constexpr double W_LEVEL  = 5.0,   W_WIDTH  = 0.18;

// ---- continent (§9) --------------------------------------------------------
constexpr double CONT_LO = 0.22, CONT_HI = 0.44, CONT_MIN = 0.2;

// ---- regions (§13 — macro Voronoi, cell 8192) ------------------------------
constexpr double REGION_CELL  = 8192.0;
constexpr double REG_BLEND_LO = 0.04, REG_BLEND_HI = 0.18;
constexpr double REG_HILLS = 0.45, REG_MTN = 1.3, REG_DUNE = 0.85, DUNE_H = 42.0;

// ---- anomalies (§12 — Zonai terraces / spire / spiral, cell 2048) ----------
constexpr double ANOM_CELL = 2048.0, P_ANOM = 0.06;
constexpr double ANOM_R_IN = 380.0,  ANOM_R_OUT = 620.0;
constexpr double TERRACE_STEP = 70.0, TERRACE_H = 15.0;
constexpr double SPIRE_H = 250.0, SPIRE_R = 150.0;
constexpr double SPIRAL_AMP = 16.0, SPIRAL_TURNS = 6.0, SPIRAL_RAD = 0.05;

// ---- climate (§9) ----------------------------------------------------------
constexpr double T_BASE = 14.0, L_LAT = 1.2e-5, L_ALT = 0.02;
constexpr double M_BASE = 0.5,  M_AMP = 0.40,  M_OCEAN = 0.35;

// ---- raymarch (§16, §26) ---------------------------------------------------
constexpr double LAMBDA_STEP = 0.8;
constexpr double DELTA_MIN   = 0.05;
constexpr int    BISECT_STEPS = 6;
constexpr double STEP_GROWTH = 0.0015;
constexpr double START_T     = 0.4;

// ---- player physics twin (§31) ---------------------------------------------
constexpr double PLAYER_HEIGHT = 1.7;
constexpr double GROUND_BIAS   = 0.08; // absorbs CPU/GPU fp divergence
constexpr double GRAVITY       = 22.0;
constexpr double WALK_SPEED    = 6.0;
constexpr double SPRINT_MUL    = 1.8;
constexpr double JUMP_V        = 8.5;
constexpr double FLY_SPEED     = 40.0;
constexpr double SLOPE_MAX     = 1.5;
constexpr double SLIDE_ACCEL   = 14.0;

/* ========================================================================== */
/*  §4 — deterministic hashing (uint32, Murmur3 finalizer)                    */
/* ========================================================================== */

/** MurmurHash3 finalizer — avalanche. All math mod 2^32. */
inline uint32_t hash_u32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

/** H(x_i, y_i, s) = fmix(x_i ⊕ (y_i·0x27D4EB2F) ⊕ (s·0x9E3779B9)). */
inline uint32_t hash3(int32_t xi, int32_t yi, int32_t seed) {
    const uint32_t h =
        static_cast<uint32_t>(xi)
        ^ (static_cast<uint32_t>(yi) * 0x27d4eb2fu)
        ^ (static_cast<uint32_t>(seed) * 0x9e3779b9u);
    return hash_u32(h);
}

/** Uniform in [0, 1] from an integer lattice cell. */
inline double hash01(int32_t xi, int32_t yi, int32_t seed) {
    return static_cast<double>(hash3(xi, yi, seed)) / 4294967295.0;
}

/** Feature point in [0,1]² from the two 12-bit halves of ONE hash (GPU budget). */
inline void cell_point(int32_t ix, int32_t iy, int32_t seed, double &px, double &py) {
    const uint32_t h = hash3(ix, iy, seed);
    px = static_cast<double>(h & 0xfffu) / 4095.0;
    py = static_cast<double>((h >> 12) & 0xfffu) / 4095.0;
}

/** Cell type byte (top 8 bits of the same hash) in [0,1]. */
inline double cell_type(int32_t ix, int32_t iy, int32_t seed) {
    return static_cast<double>(hash3(ix, iy, seed) >> 24) / 255.0;
}

/* ========================================================================== */
/*  §5 — quintic value noise (C²) + Nyquist weights                           */
/* ========================================================================== */

/** S(t) = 6t⁵ − 15t⁴ + 10t³ — C² continuous, zero 1st/2nd derivatives at 0,1. */
inline double quintic(double t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

/** Value noise on the integer lattice, mapped to [−1, 1]. */
inline double vnoise(double x, double y, int32_t seed) {
    const double ix = std::floor(x), iy = std::floor(y);
    const double fx = x - ix,        fy = y - iy;
    const double ux = quintic(fx),   uy = quintic(fy);
    const auto XI = static_cast<int32_t>(ix);
    const auto YI = static_cast<int32_t>(iy);
    const double a = hash01(XI,     YI,     seed);
    const double b = hash01(XI + 1, YI,     seed);
    const double c = hash01(XI,     YI + 1, seed);
    const double d = hash01(XI + 1, YI + 1, seed);
    // identical bilinear expansion as the GPU (NOT lerp(lerp)) — bit parity
    const double v = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
    return v * 2.0 - 1.0;
}

/**
 * Nyquist octave weight (§5): fade an octave out between 0.5 and 1 cycle per
 * pixel footprint Δp(t) = t·2tan(FOV/2)/Res_y. CPU callers pass Δp = 0
 * (full detail) or the screen footprint to mirror the GPU's filtered field.
 */
inline double nyq_w(double f, double dp) {
    const double w = 2.0 - (f * dp) / 0.5;
    return w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
}

/** fBm with Nyquist attenuation + amplitude renormalisation (§5). */
inline double fbm(double x, double y, int32_t seed,
                  int octaves, double lacunarity, double gain, double dp,
                  double f0) {
    double freq = 1.0, amp = 1.0, sum = 0.0, wsum = 0.0, asum = 0.0;
    for (int i = 0; i < octaves; ++i) {
        const double w = nyq_w(f0 * freq, dp);
        sum  += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
        wsum += amp * w;
        asum += amp;
        freq *= lacunarity;
        amp  *= gain;
    }
    return sum * (asum / (wsum > 1e-6 ? wsum : 1e-6));
}

/** Ridged multifractal (§6): (1 − |N|)²; Nyquist weight uses 2f. */
inline double ridged(double x, double y, int32_t seed,
                     int octaves, double lacunarity, double gain, double dp,
                     double f0) {
    double freq = 1.0, amp = 0.5, sum = 0.0, wsum = 0.0, asum = 0.0;
    for (int i = 0; i < octaves; ++i) {
        const double w = nyq_w(2.0 * f0 * freq, dp);
        const double n = 1.0 - std::fabs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
        const double r = n * n;
        sum  += amp * w * r;
        wsum += amp * w;
        asum += amp;
        freq *= lacunarity;
        amp  *= gain;
    }
    return sum * (asum / (wsum > 1e-6 ? wsum : 1e-6));
}

/* ========================================================================== */
/*  §7 / §12 / §13 — cellular (Worley) fields                                 */
/* ========================================================================== */

struct VoronoiResult {
    double f1;        // nearest feature distance (cell units)
    double f2;        // second nearest
    double edge;      // F2 − F1 (§7 edge distance)
    double center_x;  // nearest feature point, input-space coords
    double center_y;
    double cell_hash; // type byte (top 8 bits) of the owning cell
};

/** 3×3 Worley scan — ONE hash per cell packs point + type (GPU budget). */
inline VoronoiResult voronoi(double x, double y, int32_t seed) {
    const double ipx = std::floor(x), ipy = std::floor(y);
    const double fx = x - ipx,        fy = y - ipy;
    VoronoiResult v{8.0, 8.0, 0.0, 0.0, 0.0, 0.0};
    const auto ipxi = static_cast<int32_t>(ipx);
    const auto ipyi = static_cast<int32_t>(ipy);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            const int32_t cx = ipxi + i, cy = ipyi + j;
            double px, py;
            cell_point(cx, cy, seed, px, py);
            const double dx = i + px - fx;
            const double dy = j + py - fy;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < v.f1) {
                v.f2 = v.f1; v.f1 = d;
                v.center_x = cx + px; v.center_y = cy + py;
                v.cell_hash = cell_type(cx, cy, seed);
            } else if (d < v.f2) {
                v.f2 = d;
            }
        }
    }
    v.edge = v.f2 - v.f1;
    return v;
}

/* ========================================================================== */
/*  §14 — domain warp                                                         */
/* ========================================================================== */

inline void domain_warp(double x, double y, int32_t seed, double dp,
                        double &qx, double &qy) {
    const double wx = x * F_WARP, wy = y * F_WARP;
    const double nx = fbm(wx, wy,             seed + 11, OCT_WARP, 2.0, 0.5, dp, F_WARP);
    const double ny = fbm(wx + 5.2, wy + 1.3, seed + 37, OCT_WARP, 2.0, 0.5, dp, F_WARP);
    qx = x + S_WARP * nx;
    qy = y + S_WARP * ny;
}

/* ========================================================================== */
/*  §12 — geological anomalies (Zonai terraces + spire + spiral)              */
/* ========================================================================== */

inline double sstep(double e0, double e1, double x) {
    const double t = (x - e0) / (e1 - e0);
    const double c = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return c * c * (3.0 - 2.0 * c);
}

inline double anomaly_e(double x, double y, int32_t seed) {
    const VoronoiResult v = voronoi(x / ANOM_CELL, y / ANOM_CELL, seed + 400);
    if (v.cell_hash >= P_ANOM) return 0.0;
    const double cx = v.center_x * ANOM_CELL;
    const double cy = v.center_y * ANOM_CELL;
    const double dx = x - cx, dy = y - cy;
    const double r = std::sqrt(dx * dx + dy * dy);
    if (r >= ANOM_R_OUT) return 0.0;
    const double theta = std::atan2(dy, dx);
    const double fall = 1.0 - sstep(ANOM_R_IN, ANOM_R_OUT, r);
    const double t = r / TERRACE_STEP;
    const double steps = (std::floor(t) + sstep(0.3, 0.7, t - std::floor(t))) * TERRACE_H;
    const double spire = SPIRE_H * std::exp(-(r / SPIRE_R) * (r / SPIRE_R));
    const double spiral = SPIRAL_AMP * std::sin(SPIRAL_TURNS * theta + SPIRAL_RAD * r);
    return fall * (steps + spire + spiral);
}

/* ========================================================================== */
/*  §13 — regional mathematical rules                                         */
/* ========================================================================== */

struct RegionInfo {
    double w_region;    // 0 at cell borders → 1 in cell interior
    double mount_factor;// H_MOUNT multiplier for this region
    double dune_amount; // 0..1 crescent-dune blend
};

inline RegionInfo region_field(double x, double y, int32_t seed) {
    const VoronoiResult v = voronoi(x / REGION_CELL, y / REGION_CELL, seed + 900);
    RegionInfo r;
    r.w_region = sstep(REG_BLEND_LO, REG_BLEND_HI, v.edge);
    const double h = v.cell_hash;
    r.mount_factor = h < 0.33 ? REG_HILLS : (h < 0.66 ? REG_MTN : REG_DUNE);
    r.dune_amount = sstep(0.66, 0.8, h) * r.w_region;
    return r;
}

/** Crescent (barchan) dunes — trigonometric region rule (§13). */
inline double dunes_e(double x, double y, int32_t seed, double dp, double amount) {
    if (amount <= 0.0) return 0.0;
    const double phase = x * F_DUNE * 6.283185307179586
        + 3.0 * fbm(x * F_WARP, y * F_WARP, seed + 301, 2, 2.0, 0.5, dp, F_WARP);
    const double c = 1.0 - std::fabs(std::sin(phase));
    return DUNE_H * amount * c * c * c;
}

/* ========================================================================== */
/*  Master elevation field — spec §2 composition                              */
/*  E(x, z): y-up world, field lookup is E(p.x, p.z). Spec z ↔ engine y.      */
/* ========================================================================== */

/**
 * E(x, z) in metres. `dp` = Nyquist pixel footprint (§5): pass 0 for the
 * full-detail CPU field (physics / gameplay), or the screen footprint to
 * reproduce exactly what the GPU renders at that distance.
 */
inline double elevation(double x, double z, int32_t seed, double dp) {
    // §14 — warp the domain (fluid geological folding)
    double wx, wy;
    domain_warp(x, z, seed, dp, wx, wy);

    // §13 — regional rules on the RAW (un-warped) plane
    const RegionInfo reg = region_field(x, z, seed);

    // §9 — continent field (base amplitude + ocean-moisture proxy)
    const double cont = sstep(CONT_LO, CONT_HI,
        0.5 + 0.5 * vnoise(x * F_CONT, z * F_CONT, seed + 801));

    // §5 — macro fBm hills (warped domain)
    const double base = H_BASE * (0.5 + 0.5 * fbm(
        wx * F_MACRO, wy * F_MACRO, seed + 1,
        OCT_MACRO, LAC_MACRO, GAIN_MACRO, dp, F_MACRO));

    // §6 — ridged mountains gated by a macro mask (warped domain)
    const double mask = sstep(MASK_LO, MASK_HI,
        0.5 + 0.5 * vnoise(wx * F_MASK, wy * F_MASK, seed + 77));
    const double ridge = ridged(wx * F_RIDGE, wy * F_RIDGE, seed + 5,
        OCT_RIDGE, LAC_RIDGE, GAIN_RIDGE, dp, F_RIDGE);
    const double h_mount = H_MOUNT * (1.0 + (reg.mount_factor - 1.0) * reg.w_region);

    // §7 — cellular valley network carves rivers (warped domain)
    const VoronoiResult vv = voronoi(wx * F_VALLEY, wy * F_VALLEY, seed + 21);
    const double sv = 1.0 - sstep(0.0, W_VALLEY, vv.edge);
    const double valley = C_DEPTH * sv * sv;

    // §13 dunes + §12 anomalies (raw plane — rigid Zonai geometry)
    const double dune = dunes_e(x, z, seed, dp, reg.dune_amount);
    const double anom = anomaly_e(x, z, seed);

    return base * (CONT_MIN + (1.0 - CONT_MIN) * cont)
         + h_mount * mask * ridge
         + dune
         - valley
         + anom;
}

/** Central-difference gradient (§17). ε ≥ 0.05 floor for lattice stability. */
inline void gradient(double x, double z, int32_t seed, double eps,
                     double &gx, double &gz) {
    const double e = eps > 0.05 ? eps : 0.05;
    gx = (elevation(x + e, z, seed, 0) - elevation(x - e, z, seed, 0)) / (2.0 * e);
    gz = (elevation(x, z + e, seed, 0) - elevation(x, z - e, seed, 0)) / (2.0 * e);
}

/** Slope magnitude ‖∇E‖ — the Lipschitz factor + slide gate input. */
inline double slope(double x, double z, int32_t seed) {
    double gx, gz;
    gradient(x, z, seed, 0.25, gx, gz);
    return std::sqrt(gx * gx + gz * gz);
}

/* ========================================================================== */
/*  §9 — climate fields                                                       */
/* ========================================================================== */

inline double temperature(double x, double z, double global_y, double e, int32_t seed) {
    (void)x; (void)z; (void)seed;
    return T_BASE - L_LAT * std::fabs(global_y) - L_ALT * (e > 0 ? e : 0.0);
}

inline double moisture(double x, double z, int32_t seed) {
    const double n = fbm(x * F_MOIST, z * F_MOIST, seed + 61, 2, 2.0, 0.5, 0.0, F_MOIST);
    const double cont = sstep(CONT_LO, CONT_HI,
        0.5 + 0.5 * vnoise(x * F_CONT, z * F_CONT, seed + 801));
    // λ_ocean·exp(−D_ocean/σ) approximated by the continent field itself
    // (smooth, deterministic, identical CPU/GPU — see worklog).
    return M_BASE + M_AMP * n + M_OCEAN * (1.0 - cont);
}

/* ========================================================================== */
/*  §16 / §26 — Lipschitz-corrected raymarch + bisection (CPU gameplay ray)   */
/* ========================================================================== */

struct MarchResult {
    bool   hit;
    double t;
};

/** CPU raycast against E — mirrors the shader march step-for-step. */
inline MarchResult march_ray(double ox, double oy, double oz,
                             double dx, double dy, double dz,
                             int32_t seed, int steps, double footprint,
                             double max_dist = MAX_DIST) {
    double t = START_T, t_prev = t, grad = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double px = ox + dx * t, py = oy + dy * t, pz = oz + dz * t;
        if (py > MAX_WORLD_H && dy > 0.0) return {false, t};   // §26 early exit

        const double dp = t * footprint > 1e-4 ? t * footprint : 1e-4;
        const double e = elevation(px, pz, seed, dp);

        if (py < e) {
            // crossed the surface — bisect over [t_prev, t] (§16)
            double lo = t_prev, hi = t;
            for (int b = 0; b < BISECT_STEPS; ++b) {
                const double tm = 0.5 * (lo + hi);
                const double gap = (oy + dy * tm) - elevation(ox + dx * tm, oz + dz * tm, seed, tm * footprint);
                if (gap < 0.0) hi = tm; else lo = tm;
            }
            return {true, 0.5 * (lo + hi)};
        }

        if ((i & 1) == 0) { // ‖∇E‖ refreshed every 2nd sample (§16/§34.1)
            const double eps = dp > 0.15 ? dp : 0.15;
            const double ex = elevation(px + eps, pz, seed, dp);
            const double ez = elevation(px, pz + eps, seed, dp);
            grad = std::sqrt(((ex - e) / eps) * ((ex - e) / eps)
                           + ((ez - e) / eps) * ((ez - e) / eps));
        }

        t_prev = t;
        double step = (py - e) / std::sqrt(grad * grad + 1.0) * LAMBDA_STEP;
        if (step < DELTA_MIN) step = DELTA_MIN;
        step *= 1.0 + t * STEP_GROWTH;              // §26 adaptive growth
        if (step > t * 0.06 + 2.0) step = t * 0.06 + 2.0; // horizon clamp
        t += step;
        if (t > max_dist) return {false, t};
    }
    return {false, t};
}

/** §18 — soft shadow toward the sun with a 4× inflated footprint. */
inline double soft_shadow(double px, double py, double pz,
                          double sx, double sy, double sz,
                          int32_t seed, int steps, double footprint) {
    double res = 1.0, t = 0.6;
    for (int i = 0; i < steps; ++i) {
        const double qx = px + sx * t, qy = py + sy * t, qz = pz + sz * t;
        if (qy > MAX_WORLD_H) break;
        const double h = qy - elevation(qx, qz, seed, 4.0 * t * footprint); // §18: 4×Δp
        const double r = 10.0 * h / t;
        if (r < res) res = r;
        if (res < 0.005) break;
        double st = h * 0.8;
        if (st < 0.5) st = 0.5; else if (st > 60.0) st = 60.0;
        t += st + t * 0.04;
    }
    return res < 0.0 ? 0.0 : (res > 1.0 ? 1.0 : res);
}

/* ========================================================================== */
/*  §31 — CPU physics twin movement loop (jump-preserving)                    */
/* ========================================================================== */

struct PlayerState {
    double x, y, z;      // global, double precision
    double vx, vy, vz;
    bool   grounded;
};

struct InputState {
    double forward;      // −1..1
    double strafe;       // −1..1
    bool   jump;
    bool   sprint;
    double yaw;          // camera yaw for the wish direction
};

/**
 * §31 loop verbatim: gravity ALWAYS integrates; the ground clamp only ever
 * zeroes NEGATIVE vertical velocity so jumps are never killed; the slope
 * gate slides the player off steep warped ridges.
 */
inline void update_player(PlayerState &s, const InputState &in, double dt,
                          int32_t seed) {
    const double speed = WALK_SPEED * (in.sprint ? SPRINT_MUL : 1.0);
    const double sin_y = std::sin(in.yaw), cos_y = std::cos(in.yaw);
    // forward = (sin yaw, 0, cos yaw) — matches the camera basis convention
    const double wish_x = (sin_y * in.forward + cos_y * in.strafe) * speed;
    const double wish_z = (cos_y * in.forward - sin_y * in.strafe) * speed;

    // horizontal velocity: exponential approach (critically damped feel)
    const double k = 1.0 - std::exp(-12.0 * dt);
    s.vx += (wish_x - s.vx) * k;
    s.vz += (wish_z - s.vz) * k;

    if (in.jump && s.grounded) s.vy = JUMP_V;

    s.vy -= GRAVITY * dt;                       // GRAVITY always integrates
    s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt;

    // CPU twin: global frame, double precision (the GPU evaluates the same
    // field on chunk-local fp32 — divergence absorbed by GROUND_BIAS below).
    const double ground_z = elevation(s.x, s.z, seed, 0.0)
        + GROUND_BIAS + PLAYER_HEIGHT;

    s.grounded = false;
    if (s.y < ground_z) {
        s.y = ground_z;
        if (s.vy < 0.0) s.vy = 0.0;             // stop FALLING only — preserve jumps
        s.grounded = true;
    }

    // Slope gate: slide off steep warped ridges (§31 insert)
    if (s.grounded) {
        double gx, gz;
        gradient(s.x, s.z, seed, 0.25, gx, gz);
        const double m = std::sqrt(gx * gx + gz * gz);
        if (m > SLOPE_MAX) {
            const double inv = 1.0 / m;
            s.vx += -gx * inv * (m - SLOPE_MAX) * SLIDE_ACCEL * dt;
            s.vz += -gz * inv * (m - SLOPE_MAX) * SLIDE_ACCEL * dt;
        }
    }
}

} // namespace terrain
