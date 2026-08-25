// ============================================================================
//  W(x,y) — Continuous World Engine · GPU surface
//  Godot 4 Spatial Shader · Forward+ · Metal (Apple Silicon TBDR) · SDFGI
// ============================================================================
//
//  THE SURFACE IS NOT A MESH (spec §1, §15).
//  Every pixel is a raymarch against the continuous 2D field E(x, z) —
//  the SAME field the CPU twin evaluates (gdextension/src/terrain_cpu.h,
//  scripts/terrain_math_fallback.gd, and the WebGL2 parity harness).
//
//  ── APPLE SILICON / TBDR CONSTRAINTS (Metal) ─────────────────────────────
//  * render_mode cull_disabled — the fullscreen quad must survive any camera
//    orientation; a giant custom_aabb (set by scripts/terrain_renderer.gd)
//    defeats frustum culling.
//  * render_priority = -100 (set on the MeshInstance3D) — on a TBDR GPU the
//    tile binning happens before fragment execution, so drawing the terrain
//    FIRST in the opaque queue lets subsequent per-tile fragments depth-test
//    against our analytically written DEPTH for free (no overdraw blending
//    cost, no late-Z surprises in the tile transient memory).
//  * depth_draw_always — we write DEPTH for every fragment (hit AND miss);
//    reversed-Z far (0.0) for sky so real geometry still sorts correctly.
//  * MSAA is OFF project-wide (see project.godot): analytic coverage from the
//    Nyquist-weighted field is the anti-aliaser; FXAA is enabled instead.
//    MSAA would multiply per-tile bandwidth for zero benefit on a surface
//    whose geometry exists only as depth writes.
//  * The march is bounded (u_march_steps) and the per-step cost falls with
//    distance because the Nyquist weights switch octaves off — TBDR cost is
//    per-tile, and distant tiles evaluate a cheaper field.
//
//  ── SDFGI CONSTRAINTS (Forward+ voxel GI) ────────────────────────────────
//  * We write ALBEDO + NORMAL (view space) + DEPTH, which is exactly what
//    SDFGI voxelization consumes — the mathematical terrain therefore
//    participates in global illumination like any mesh.
//  * Direct sun lighting is computed HERE (EMISSION) and the DirectionalLight
//    uses a light_cull_mask that excludes the terrain's layer (set in
//    scripts/main.gd) — otherwise the engine would light the quad a second
//    time. SDFGI INDIRECT light still reaches the terrain through ALBEDO.
//  * Sky fragments write DEPTH = 0.0 (reversed-Z far) → outside every SDFGI
//    cascade → the sky is never voxelized as glowing geometry.
//  * VALIDATION (spec §31): drop a red DebugBox at a cascade boundary and
//    confirm no light leaking where the cascade changes; the analytic DEPTH
//    is continuous, so seams can only come from cascade resolution.
//
//  ── REVERSED-Z DEPTH WRITE (Metal NDC) ───────────────────────────────────
//  Godot 4's projection is reversed-Z ([1 = near, 0 = far]) on Metal. For a
//  hit at world point p we reconstruct the exact clip position the engine
//  would produce for a mesh vertex at p:
//        p_view  = transpose(view→world) · (p − cam_pos)
//        clip    = PROJECTION_MATRIX · p_view
//        DEPTH   = clip.z / clip.w
//  This is bit-compatible with rasterized geometry — meshes, particles and
//  the player capsule inter-occlude with the terrain correctly (§16).
//
//  ── FLOATING ORIGIN (spec §25) ────────────────────────────────────────────
//  The shader receives (chunk-local fp32 position, u_chunk_origin). Every
//  frequency in the contract is a dyadic rational n/2^20 and the chunk size
//  is 2^13, so origin·f is EXACT in fp32 — noise lattices align across
//  rebases with zero jitter. Global coordinates never leave fp32 territory.
// ============================================================================

shader_type spatial;
render_mode cull_disabled, fog_disabled, depth_draw_always;

// ---- per-frame uniforms (scripts/terrain_renderer.gd) ----------------------
uniform vec2  u_chunk_origin = vec2(0.0); // multiples of 8192 — fp32-exact
uniform int   u_seed = 1337;
uniform float u_time = 0.0;               // water ripple / cloud drift
uniform int   u_march_steps = 96;         // quality preset (§26)
uniform int   u_shadow_steps = 20;
uniform float u_tan_half = 0.7002;        // tan(70°/2)
uniform float u_aspect = 1.7778;
uniform float u_footprint = 0.0026;       // 2·tan(FOV/2)/Res_y (Nyquist Δp/t)

// ---- varyings: camera plumbing from vertex to fragment ---------------------
// Godot does not expose VIEW/PROJECTION matrices in fragment(), so the vertex
// stage forwards them as varyings (constant across the quad → exact).
varying vec2 v_ndc;                        // quad coords, [-1, 1]
varying vec3 v_cam;                        // camera position (node = world)
varying vec3 v_r;                          // view→world basis: right
varying vec3 v_u;                          //                   up
varying vec3 v_b;                          //                   back
varying vec4 v_pc0;                        // PROJECTION_MATRIX column 0
varying vec4 v_pc1;                        //                      column 1
varying vec4 v_pc2;                        //                      column 2
varying vec4 v_pc3;                        //                      column 3

// ---- contract constants (MIRROR constants.ts / terrain_cpu.h / demoShader) -
const float MAXH     = 1300.0;
const float MAXDIST  = 6000.0;
const float F_MACRO  = 512.0  / 1048576.0;
const float F_RIDGE  = 341.0  / 1048576.0;
const float F_MASK   = 128.0  / 1048576.0;
const float F_WARP   = 655.0  / 1048576.0;
const float F_VALLEY = 1942.0 / 1048576.0;
const float F_CONT   = 64.0   / 1048576.0;
const float F_MOIST  = 512.0  / 1048576.0;
const float F_DUNE   = 23.0   / 1048576.0;
const float F_CLOUD  = 403.0  / 1048576.0;
const float VEG_WL   = 14.0;

const int   OCT_MACRO = 9;
const float LAC_MACRO = 2.0,  GAIN_MACRO = 0.5,  H_BASE  = 90.0;
const int   OCT_RIDGE = 5;
const float LAC_RIDGE = 2.17, GAIN_RIDGE = 0.48, H_MOUNT = 880.0;
const float MASK_LO = 0.28,  MASK_HI = 0.60;
const int   OCT_WARP = 2;
const float S_WARP   = 620.0;
const float C_DEPTH  = 48.0,  W_VALLEY = 0.42, W_LEVEL = 5.0, W_WIDTH = 0.18;
const float CONT_LO  = 0.22,  CONT_HI  = 0.44, CONT_MIN = 0.2;
const float REGION_CELL = 8192.0;
const float REG_BLEND_LO = 0.04, REG_BLEND_HI = 0.18;
const float REG_HILLS = 0.45, REG_MTN = 1.3, REG_DUNE = 0.85, DUNE_H = 42.0;
const float ANOM_CELL = 2048.0, P_ANOM = 0.06;
const float ANOM_R_IN = 380.0,  ANOM_R_OUT = 620.0;
const float TERRACE_STEP = 70.0, TERRACE_H = 15.0;
const float SPIRE_H = 250.0, SPIRE_R = 150.0;
const float SPIRAL_AMP = 16.0, SPIRAL_TURNS = 6.0, SPIRAL_RAD = 0.05;
const float T_BASE = 14.0, L_LAT = 1.2e-5, L_ALT = 0.02;
const float M_BASE = 0.5,  M_AMP = 0.40, M_OCEAN = 0.35;

const float LAMBDA  = 0.8;
const float DMIN    = 0.05;
const float GROWTH  = 0.0015;
const float START_T = 0.4;
const int   BISECT  = 6;

const vec3  SUN     = vec3(0.550744, 0.420568, 0.720974); // normalize(0.55,0.42,0.72)
const vec3  SUN_COL = vec3(1.0, 0.955, 0.885) * 3.4;
const vec3  SKY_ZEN = vec3(0.20, 0.36, 0.52);
const vec3  SKY_HOR = vec3(0.78, 0.80, 0.82);
const float CLOUD_PLANE = 1500.0;
const int   CLOUD_OCT = 4;
const float SIGMA_FOG = 9.0e-4;
const float FOG_H     = 900.0;

// ---------------------------------------------------------------------------
// §4 — deterministic hashing (uint32, Murmur3 finalizer)
// ---------------------------------------------------------------------------
uint hashU(uint x) {
        x ^= x >> 16; x *= 0x85ebca6bu;
        x ^= x >> 13; x *= 0xc2b2ae35u;
        x ^= x >> 16;
        return x;
}
uint hash3(int xi, int yi, int seed) {
        uint h = uint(xi) ^ (uint(yi) * 0x27d4eb2fu) ^ (uint(seed) * 0x9e3779b9u);
        return hashU(h);
}
float hash01(int xi, int yi, int seed) {
        return float(hash3(xi, yi, seed)) / 4294967295.0;
}
void cellPoint(int ix, int iy, int seed, out vec2 p) {
        uint h = hash3(ix, iy, seed);
        p = vec2(float(h & 0xfffu), float((h >> 12) & 0xfffu)) / 4095.0;
}
float cellType(int ix, int iy, int seed) {
        return float(hash3(ix, iy, seed) >> 24) / 255.0;
}

// ---------------------------------------------------------------------------
// §5 — quintic value noise (C²) + Nyquist weights
// ---------------------------------------------------------------------------
float quintic(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float vnoise(float x, float y, int seed) {
        float ix = floor(x), iy = floor(y);
        float fx = x - ix,   fy = y - iy;
        float ux = quintic(fx), uy = quintic(fy);
        int XI = int(ix), YI = int(iy);
        float a = hash01(XI,     YI,     seed);
        float b = hash01(XI + 1, YI,     seed);
        float c = hash01(XI,     YI + 1, seed);
        float d = hash01(XI + 1, YI + 1, seed);
        float v = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
        return v * 2.0 - 1.0;
}

float nyqW(float f, float dp) {
        return clamp(2.0 - (f * dp) / 0.5, 0.0, 1.0);
}

float fbm(float x, float y, int seed, int oct, float lac, float gain, float dp, float f0) {
        float freq = 1.0, amp = 1.0, sum = 0.0, wsum = 0.0, asum = 0.0;
        for (int i = 0; i < oct; i++) {
                float w = nyqW(f0 * freq, dp);
                sum  += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
                wsum += amp * w;
                asum += amp;
                freq *= lac;
                amp  *= gain;
        }
        return sum * (asum / max(wsum, 1e-6));
}

float ridged(float x, float y, int seed, int oct, float lac, float gain, float dp, float f0) {
        float freq = 1.0, amp = 0.5, sum = 0.0, wsum = 0.0, asum = 0.0;
        for (int i = 0; i < oct; i++) {
                float w = nyqW(2.0 * f0 * freq, dp);
                float n = 1.0 - abs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
                float r = n * n;
                sum  += amp * w * r;
                wsum += amp * w;
                asum += amp;
                freq *= lac;
                amp  *= gain;
        }
        return sum * (asum / max(wsum, 1e-6));
}

// ---------------------------------------------------------------------------
// §7/§12/§13 — cellular (Worley) fields
// ---------------------------------------------------------------------------
struct Vor {
        float f1;    float f2;    float edge;
        float cx;    float cy;    float type;
};

Vor voronoi(float x, float y, int seed) {
        float ipx = floor(x), ipy = floor(y);
        float fx = x - ipx,   fy = y - ipy;
        Vor v;
        v.f1 = 8.0; v.f2 = 8.0; v.edge = 0.0;
        v.cx = 0.0;  v.cy = 0.0;  v.type = 0.0;
        for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                        int cxi = int(ipx) + i, cyi = int(ipy) + j;
                        vec2 p; cellPoint(cxi, cyi, seed, p);
                        float dx = float(i) + p.x - fx;
                        float dy = float(j) + p.y - fy;
                        float d = sqrt(dx * dx + dy * dy);
                        if (d < v.f1) {
                                v.f2 = v.f1; v.f1 = d;
                                v.cx = float(cxi) + p.x; v.cy = float(cyi) + p.y;
                                v.type = cellType(cxi, cyi, seed);
                        } else if (d < v.f2) {
                                v.f2 = d;
                        }
                }
        }
        v.edge = v.f2 - v.f1;
        return v;
}

// ---------------------------------------------------------------------------
// §14 — domain warp
// ---------------------------------------------------------------------------
void domainWarp(float x, float y, int seed, float dp, out vec2 q) {
        float wx = x * F_WARP, wy = y * F_WARP;
        float qx = fbm(wx, wy,             seed + 11, OCT_WARP, 2.0, 0.5, dp, F_WARP);
        float qy = fbm(wx + 5.2, wy + 1.3, seed + 37, OCT_WARP, 2.0, 0.5, dp, F_WARP);
        q = vec2(x + S_WARP * qx, y + S_WARP * qy);
}

// ---------------------------------------------------------------------------
// §12 — geological anomalies (Zonai terraces + spire + spiral)
// ---------------------------------------------------------------------------
float anomalyE(float x, float y, int seed) {
        Vor v = voronoi(x / ANOM_CELL, y / ANOM_CELL, seed + 400);
        if (v.type >= P_ANOM) return 0.0;
        float cx = v.cx * ANOM_CELL;
        float cy = v.cy * ANOM_CELL;
        float dx = x - cx, dy = y - cy;
        float r = sqrt(dx * dx + dy * dy);
        if (r >= ANOM_R_OUT) return 0.0;
        float theta = atan2(dy, dx);
        float fall = 1.0 - smoothstep(ANOM_R_IN, ANOM_R_OUT, r);
        float t = r / TERRACE_STEP;
        float ft = t - floor(t);
        float steps = (floor(t) + smoothstep(0.3, 0.7, ft)) * TERRACE_H;
        float spire = SPIRE_H * exp(-(r / SPIRE_R) * (r / SPIRE_R));
        float spiral = SPIRAL_AMP * sin(SPIRAL_TURNS * theta + SPIRAL_RAD * r);
        return fall * (steps + spire + spiral);
}

// ---------------------------------------------------------------------------
// §13 — regional rules
// ---------------------------------------------------------------------------
struct Reg { float wRegion; float mountFactor; float duneAmount; };

Reg regionField(float x, float y, int seed) {
        Vor v = voronoi(x / REGION_CELL, y / REGION_CELL, seed + 900);
        Reg r;
        r.wRegion = smoothstep(REG_BLEND_LO, REG_BLEND_HI, v.edge);
        float h = v.type;
        r.mountFactor = h < 0.33 ? REG_HILLS : (h < 0.66 ? REG_MTN : REG_DUNE);
        r.duneAmount = smoothstep(0.66, 0.8, h) * r.wRegion;
        return r;
}

float dunesE(float x, float y, int seed, float dp, float amount) {
        if (amount <= 0.0) return 0.0;
        float phase = x * F_DUNE * 6.28318530718
                + 3.0 * fbm(x * F_WARP, y * F_WARP, seed + 301, 2, 2.0, 0.5, dp, F_WARP);
        float c = 1.0 - abs(sin(phase));
        return DUNE_H * amount * c * c * c;
}

// ---------------------------------------------------------------------------
// Master elevation field — spec §2 composition (mirror terrain_cpu.h)
// ---------------------------------------------------------------------------
float elevation(vec2 g, float dp, int seed) {
        vec2 wq;
        domainWarp(g.x, g.y, seed, dp, wq);

        Reg reg = regionField(g.x, g.y, seed);

        float cont = smoothstep(CONT_LO, CONT_HI,
                0.5 + 0.5 * vnoise(g.x * F_CONT, g.y * F_CONT, seed + 801));

        float base = H_BASE * (0.5 + 0.5 * fbm(
                wq.x * F_MACRO, wq.y * F_MACRO, seed + 1,
                OCT_MACRO, LAC_MACRO, GAIN_MACRO, dp, F_MACRO));

        float mask = smoothstep(MASK_LO, MASK_HI,
                0.5 + 0.5 * vnoise(wq.x * F_MASK, wq.y * F_MASK, seed + 77));
        float ridge = ridged(wq.x * F_RIDGE, wq.y * F_RIDGE, seed + 5,
                OCT_RIDGE, LAC_RIDGE, GAIN_RIDGE, dp, F_RIDGE);
        float hMount = H_MOUNT * (1.0 + (reg.mountFactor - 1.0) * reg.wRegion);

        Vor vv = voronoi(wq.x * F_VALLEY, wq.y * F_VALLEY, seed + 21);
        float sv = 1.0 - smoothstep(0.0, W_VALLEY, vv.edge);
        float valley = C_DEPTH * sv * sv;

        float dune = dunesE(g.x, g.y, seed, dp, reg.duneAmount);
        float anom = anomalyE(g.x, g.y, seed);

        return base * (CONT_MIN + (1.0 - CONT_MIN) * cont)
             + hMount * mask * ridge
             + dune
             - valley
             + anom;
}

// §17 — analytic normal via central differences at the pixel footprint.
vec2 gradient(vec2 g, float eps, float dp, int seed) {
        float e = max(eps, 0.05);
        float gx = (elevation(g + vec2(e, 0.0), dp, seed) - elevation(g - vec2(e, 0.0), dp, seed)) / (2.0 * e);
        float gz = (elevation(g + vec2(0.0, e), dp, seed) - elevation(g - vec2(0.0, e), dp, seed)) / (2.0 * e);
        return vec2(gx, gz);
}

// ---------------------------------------------------------------------------
// §9 — climate fields
// ---------------------------------------------------------------------------
float temperature(vec2 g, float e, int seed) {
        return T_BASE - L_LAT * abs(g.y) - L_ALT * max(e, 0.0);
}
float moisture(vec2 g, int seed) {
        float n = fbm(g.x * F_MOIST, g.y * F_MOIST, seed + 61, 2, 2.0, 0.5, 0.0, F_MOIST);
        float cont = smoothstep(CONT_LO, CONT_HI,
                0.5 + 0.5 * vnoise(g.x * F_CONT, g.y * F_CONT, seed + 801));
        return M_BASE + M_AMP * n + M_OCEAN * (1.0 - cont);
}

// ---------------------------------------------------------------------------
// §10/§11/§21 — biome RBF anchors in OKLab
// ---------------------------------------------------------------------------
vec3 oklabToLinear(vec3 lab) {
        float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
        float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
        float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
        float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
        return vec3(
                +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

float rbf(float x, float c, float s) {
        float d = (x - c) / s;
        return exp(-d * d);
}

vec3 biomeColor(float T, float M, float e, float slopeM, float vegNoise, out float vegOut) {
        float w0 = rbf(T, -4.0, 4.5)  * rbf(M, 0.55, 0.45) * rbf(e, 800.0, 600.0) * rbf(slopeM, 1.0, 1.1);
        float w1 = rbf(T, 9.0, 22.0)  * rbf(M, 0.45, 0.55) * rbf(e, 550.0, 600.0) * rbf(slopeM, 1.7, 0.8);
        float w2 = rbf(T, 13.0, 6.5)  * rbf(M, 0.72, 0.22) * rbf(e, 90.0, 260.0)  * rbf(slopeM, 0.32, 0.5);
        float w3 = rbf(T, 11.0, 7.0)  * rbf(M, 0.9, 0.15)  * rbf(e, 230.0, 380.0) * rbf(slopeM, 0.45, 0.55);
        float w4 = rbf(T, 16.0, 8.0)  * rbf(M, 0.13, 0.17) * rbf(e, 55.0, 320.0)  * rbf(slopeM, 0.35, 0.5);

        float wsum = max(w0 + w1 + w2 + w3 + w4, 1e-6);
        vec3 lab = (
                w0 * vec3(0.93,  0.0,    -0.015) +
                w1 * vec3(0.44,  0.004,   0.008) +
                w2 * vec3(0.52, -0.058,   0.048) +
                w3 * vec3(0.32, -0.072,   0.032) +
                w4 * vec3(0.79,  0.012,   0.072)) / wsum;

        vegOut = (w2 * 0.92 + w3 * 1.0 + w1 * 0.06 + w4 * 0.05) / wsum;

        float clump = smoothstep(0.2, 0.8, 0.5 + 0.5 * vegNoise);
        vec3 col = oklabToLinear(lab);
        return mix(col, col * 0.52 + vec3(0.002, 0.020, 0.004), vegOut * clump * 0.85);
}

// ---------------------------------------------------------------------------
// §18 — soft shadow ray (4× footprint → free penumbra)
// ---------------------------------------------------------------------------
float softShadow(vec3 p, int seed, float footprint) {
        float res = 1.0;
        float t = 0.6;
        for (int i = 0; i < 40; i++) {
                if (i >= u_shadow_steps) break;
                vec3 q = p + SUN * t;
                if (q.y > MAXH) break;
                float h = q.y - elevation(q.xz + u_chunk_origin, 4.0 * t * footprint, seed);
                res = min(res, 10.0 * h / t);
                if (res < 0.005) break;
                t += clamp(h * 0.8, 0.5, 60.0) + t * 0.04;
        }
        return clamp(res, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// §20–§22 — sky, clouds, Beer-Lambert aerial perspective
// ---------------------------------------------------------------------------
vec3 skyColor(vec3 rd) {
        float sd = max(dot(rd, SUN), 0.0);
        float k = pow(max(rd.y, 0.0), 0.55);
        vec3 sky = mix(SKY_HOR, SKY_ZEN, k);
        sky = mix(SKY_HOR * 1.04, sky, smoothstep(-0.08, 0.06, rd.y));
        sky += SUN_COL * (pow(sd, 8.0) * 0.10 + pow(sd, 90.0) * 0.30);
        sky += SUN_COL * smoothstep(0.99955, 0.99985, sd) * 6.0;
        return sky;
}

vec3 applyFog(vec3 col, float t, float h, vec3 rd) {
        float od = SIGMA_FOG * t * exp(-max(h, 0.0) / FOG_H);
        vec3 T = exp(-vec3(0.58, 0.72, 0.95) * od);
        vec3 inscatter = mix(skyColor(rd), SKY_HOR, 0.5);
        return col * T + inscatter * (1.0 - T);
}

// ---------------------------------------------------------------------------
// Vertex — fullscreen clip-space quad + camera plumbing
// ---------------------------------------------------------------------------
void vertex() {
        v_ndc = VERTEX.xy; // QuadMesh(size 2) spans [-1, 1]²
        v_cam = INV_VIEW_MATRIX[3].xyz;
        v_r = INV_VIEW_MATRIX[0].xyz;
        v_u = INV_VIEW_MATRIX[1].xyz;
        v_b = INV_VIEW_MATRIX[2].xyz;
        v_pc0 = PROJECTION_MATRIX[0];
        v_pc1 = PROJECTION_MATRIX[1];
        v_pc2 = PROJECTION_MATRIX[2];
        v_pc3 = PROJECTION_MATRIX[3];
        // Overriding POSITION emits the quad directly in clip space at the far
        // plane (z = 0 = reversed-Z far). Depth comes from the raymarch below.
        POSITION = vec4(VERTEX.xy, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Fragment — Lipschitz march, bisection, water plane, shading, DEPTH write
// ---------------------------------------------------------------------------
void fragment() {
        mat3 v2w = mat3(v_r, v_u, v_b);          // view → world rotation
        mat3 w2v = transpose(v2w);               // world → view rotation
        mat4 proj = mat4(v_pc0, v_pc1, v_pc2, v_pc3);

        vec3 dirView = vec3(v_ndc.x * u_aspect * u_tan_half, v_ndc.y * u_tan_half, -1.0);
        vec3 rd = normalize(v2w * dirView);
        vec3 ro = v_cam;

        // Underwater camera: cheap depth fog (rare gameplay state).
        if (ro.y < W_LEVEL) {
                EMISSION = vec3(0.012, 0.055, 0.075) * (0.6 + 0.4 * max(rd.y, 0.0));
                ALBEDO = vec3(0.0);
                ROUGHNESS = 1.0;
                SPECULAR = 0.0;
                DEPTH = 0.0;
                return;
        }

        // §26 early exit: above the ceiling and climbing → pure sky
        if (ro.y > MAXH && rd.y >= 0.0) {
                EMISSION = skyColor(rd);
                ALBEDO = vec3(0.0);
                ROUGHNESS = 1.0;
                SPECULAR = 0.0;
                DEPTH = 0.0; // reversed-Z far — SDFGI ignores it (outside cascades)
                return;
        }

        float t = START_T;
        float tPrev = t;
        float grad = 0.0;
        float prevY = ro.y;
        float tWater = -1.0;
        float eWater = 0.0;
        bool hit = false;
        float tHit = 0.0;

        for (int i = 0; i < 160; i++) {
                if (i >= u_march_steps) break;
                vec3 p = ro + rd * t;

                if (p.y > MAXH && rd.y > 0.0) break;

                float dp = max(t * u_footprint, 1e-4);
                float e = elevation(p.xz + u_chunk_origin, dp, u_seed);

                // water plane crossing (§8): accept only over a submerged column
                if (tWater < 0.0 && prevY > W_LEVEL && p.y <= W_LEVEL) {
                        float f = (W_LEVEL - prevY) / (p.y - prevY);
                        float tw = tPrev + (t - tPrev) * f;
                        vec2 cw = ro.xz + rd.xz * tw + u_chunk_origin;
                        float ew = elevation(cw, dp, u_seed);
                        if (ew < W_LEVEL - 0.25) {
                                tWater = tw;
                                eWater = ew;
                                break; // water is above any later terrain hit
                        }
                }
                prevY = p.y;

                if (p.y < e) { hit = true; break; }

                // Lipschitz ‖∇E‖ via forward differences, refreshed every 2nd sample
                if (i - (i / 2) * 2 == 0) {
                        float eps = max(dp, 0.15);
                        vec2 g = p.xz + u_chunk_origin;
                        float ex = elevation(g + vec2(eps, 0.0), dp, u_seed);
                        float ez = elevation(g + vec2(0.0, eps), dp, u_seed);
                        grad = length(vec2((ex - e) / eps, (ez - e) / eps));
                }

                tPrev = t;
                float step = max(DMIN, (p.y - e) / sqrt(grad * grad + 1.0) * LAMBDA);
                step *= 1.0 + t * GROWTH;         // §26 adaptive growth
                step = min(step, t * 0.06 + 2.0); // horizon clamp (anti-tunnel)
                t += step;

                if (t > MAXDIST) break;
        }

        if (hit) {
                // bisection refinement over [tPrev, t] (§16)
                float lo = tPrev, hi = t;
                for (int b = 0; b < BISECT; b++) {
                        float tm = 0.5 * (lo + hi);
                        vec3 m = ro + rd * tm;
                        float gap = m.y - elevation(m.xz + u_chunk_origin, tm * u_footprint, u_seed);
                        if (gap < 0.0) hi = tm; else lo = tm;
                }
                tHit = 0.5 * (lo + hi);
        }

        if (hit && (tWater < 0.0 || tHit < tWater)) {
                // ---------------- terrain shading ----------------
                vec3 p = ro + rd * tHit;
                vec2 g = p.xz + u_chunk_origin;
                float dp = max(tHit * u_footprint, 1e-4);

                vec2 grad2 = gradient(g, dp, dp, u_seed);   // §17: ε = pixel footprint
                vec3 n = normalize(vec3(-grad2.x, 1.0, -grad2.y));

                float e = elevation(g, dp, u_seed);
                float T = temperature(g, e, u_seed);
                float M = moisture(g, u_seed);

                float vegNoise = vnoise(g.x / VEG_WL, g.y / VEG_WL, u_seed + 171);
                float veg;
                vec3 albedo = biomeColor(T, M, e, length(grad2), vegNoise, veg);

                float ndl = max(dot(n, SUN), 0.0);
                float sh = softShadow(p + n * 0.15, u_seed, u_footprint);

                vec3 sunTerm = albedo * SUN_COL * ndl * sh;
                vec3 skyTint = mix(SKY_HOR, SKY_ZEN, 0.5);
                vec3 ambTerm = albedo * skyTint * (0.22 + 0.28 * max(n.y, 0.0));

                // Beer-Lambert inscatter only (transmittance applies to lit terms):
                vec3 fogInscatter = applyFog(vec3(0.0), tHit, p.y, rd);
                vec3 sunLit = sunTerm * exp(-vec3(0.58, 0.72, 0.95)
                        * (SIGMA_FOG * tHit * exp(-max(p.y, 0.0) / FOG_H)));

                // SDFGI: ALBEDO + NORMAL feed voxelization → indirect light reaches
                // the terrain; the sun's DIRECT term lives in EMISSION (the engine
                // light is culled from this layer to avoid double lighting). A
                // half-strength analytic sky-ambient floor keeps the surface sane
                // when voxel GI is off; SDFGI bounce adds the rest on top.
                ALBEDO = albedo;
                NORMAL = normalize(w2v * n); // view space, as the engine expects
                EMISSION = sunLit + 0.5 * ambTerm + fogInscatter;
                ROUGHNESS = 1.0;
                SPECULAR = 0.0;
                METALLIC = 0.0;

                // REVERSED-Z DEPTH (Metal NDC): exact clip depth for a mesh vertex
                // at p — bit-compatible with rasterized geometry (§16).
                vec4 pView = vec4(w2v * (p - v_cam), 1.0);
                vec4 clip = proj * pView;
                DEPTH = clip.z / clip.w;
        } else if (tWater > 0.0) {
                // ---------------- water plane (§8) ----------------
                vec3 p = ro + rd * tWater;
                vec2 g = p.xz + u_chunk_origin;
                float depth = max(W_LEVEL - eWater, 0.0);

                float r1 = vnoise(g.x / 5.5 + u_time * 0.35, g.y / 5.5 - u_time * 0.22, u_seed + 911);
                float r2 = vnoise(g.x / 9.7 - u_time * 0.20, g.y / 9.7 + u_time * 0.31, u_seed + 913);
                vec3 n = normalize(vec3(-0.22 * r1, 1.0, -0.22 * r2));

                float sh = softShadow(p + vec3(0.0, 0.15, 0.0), u_seed, u_footprint);
                vec3 shallow = vec3(0.110, 0.320, 0.300);
                vec3 deep    = vec3(0.016, 0.075, 0.100);
                vec3 body = mix(shallow, deep, smoothstep(1.5, 14.0, depth))
                        * (0.35 + 0.65 * sh) * (0.6 + 0.4 * max(dot(n, SUN), 0.0));

                float foamN = vnoise(g.x / 3.1, g.y / 3.1, u_seed + 917);
                body += vec3(0.9) * (1.0 - smoothstep(0.05, 0.6, depth)) * (0.25 + 0.35 * foamN);

                vec3 refl = skyColor(reflect(rd, n));
                float fres = 0.02 + 0.98 * pow(1.0 - max(dot(-rd, n), 0.0), 5.0);
                float spec = pow(max(dot(reflect(rd, n), SUN), 0.0), 240.0) * 3.0;

                vec3 col = applyFog(mix(body, refl, fres) + SUN_COL * spec / 3.4, tWater, W_LEVEL, rd);

                ALBEDO = vec3(0.0); // water is fully custom-lit
                NORMAL = normalize(w2v * n);
                EMISSION = col;
                ROUGHNESS = 0.1;
                SPECULAR = 0.0;
                METALLIC = 0.0;

                vec4 pView = vec4(w2v * (p - v_cam), 1.0);
                vec4 clip = proj * pView;
                DEPTH = clip.z / clip.w;
        } else {
                // ---------------- sky + clouds (§20/§22) ----------------
                vec3 col = skyColor(rd);
                if (rd.y > 0.02 && ro.y < CLOUD_PLANE) {
                        float tc = (CLOUD_PLANE - ro.y) / rd.y;
                        vec2 cp = ro.xz + rd.xz * tc + u_chunk_origin + vec2(u_time * 6.0, u_time * 2.0);
                        float dpc = max(tc * u_footprint, 0.02);
                        float c = fbm(cp.x * F_CLOUD, cp.y * F_CLOUD, u_seed + 501,
                                CLOUD_OCT, 2.0, 0.5, dpc, F_CLOUD);
                        float cov = smoothstep(0.42, 0.78, 0.5 + 0.5 * c);
                        float shade = 0.85 + 0.30 * (1.0 - smoothstep(0.2, 0.9, c));
                        vec3 cloudCol = vec3(shade) * (0.62 + 0.38 * max(SUN.y, 0.0));
                        float fade = exp(-tc * 6.0e-5) * smoothstep(0.02, 0.12, rd.y);
                        col = mix(col, cloudCol, cov * fade);
                }
                EMISSION = col;
                ALBEDO = vec3(0.0);
                ROUGHNESS = 1.0;
                SPECULAR = 0.0;
                DEPTH = 0.0; // reversed-Z far
        }
}
