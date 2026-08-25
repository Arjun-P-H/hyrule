# Worklog — W(x,y) Continuous World Engine

Project: TotK-style procedural exploration terrain, implemented per mathematical spec
(user message + upload/Pasted Content_1787651579100.txt). Target: Godot 4 / Metal /
Apple Silicon (TBDR, SDFGI, reversed-Z). This sandbox is a Next.js 16 web environment,
so the deliverable is twofold:

1. `godot-project/` — the real, complete Godot 4 project (shader, GDScripts,
   GDExtension C++ CPU twin, scenes, README).
2. `/` (src/app/page.tsx) — an engineering "spec console" web page featuring a
   live WebGL2 raymarcher that runs the *exact same math* (Stage-0 GPU parity
   harness), a live-computed worked numerical example (§30), the parameter
   table (§29), and a code browser that serves the real Godot files.

## Coordinate convention (ALL agents must follow)

- Spec math is written with field coords (x,y) and elevation z.
- Engine/demo convention is **y-up** (matches Godot): field lookup is
  `E(p.x, p.z)`, elevation along +y. Spec z ↔ engine y.
- Global positions: double (JS/GDScript are double-native; C++ uses double).
- Floating origin: CHUNK_SIZE = 8192.0 (2^13). All hash lattices receive
  chunk-local float32 coords. Frequencies are dyadic rationals n/2^20 so that
  `origin * f` stays exactly representable in fp32 and noise cells align
  across chunk borders (error analysis documented on the web page).

## Shared constants (single source of truth — MUST match in ALL of:
src/lib/constants.ts, demoShader.ts, terrain_raymarch.gdshader,
terrain_cpu.h, terrain_math_fallback.gd)

```
WORLD / CHUNK
  CHUNK_SIZE      = 8192.0        (2^13)
  MAX_WORLD_H     = 1300.0        (early-exit ceiling, §26)
  MAX_DIST        = 6000.0        (march cutoff, hidden by fog)

MACRO TERRAIN (§5)
  F_MACRO  = 512/1048576   = 1/2048
  LAC_MACRO = 2.0   GAIN_MACRO = 0.5   OCT_MACRO = 5
  H_BASE   = 90.0          (max rolling-hills amplitude)

MOUNTAINS (§6)
  F_RIDGE  = 341/1048576  (~1/3074.6)
  LAC_RIDGE = 2.17  GAIN_RIDGE = 0.48  OCT_RIDGE = 5
  H_MOUNT  = 620.0
  F_MASK   = 128/1048576  = 1/8192    (mountain macro mask freq)
  MASK_LO  = 0.35   MASK_HI = 0.75    (smoothstep range on 0.5+0.5*N)

DOMAIN WARP (§14)
  F_WARP   = 655/1048576  (~1/1600.9)
  OCT_WARP = 2     S_WARP = 620.0     (world units)

VALLEYS / RIVERS (§7, §8)
  F_VALLEY = 1942/1048576 (~1/539.9)
  C_DEPTH  = 48.0   W_VALLEY = 0.42   (edge-space)
  W_LEVEL  = 5.0    (water plane height, y-up)
  W_WIDTH  = 0.18   (river shore softness, edge-space)

CONTINENT (§9 support)
  F_CONT = 64/1048576 = 1/16384
  CONT_LO = 0.30  CONT_HI = 0.55  (smoothstep → base × [0.12, 1])

REGIONS (§13) — macro Voronoi, cell 8192
  REGION_CELL = 8192.0
  region hash < 0.33 → hills (H_MOUNT×0.45)
              < 0.66 → mountains (H_MOUNT×1.30)
              else   → dune-land (H_MOUNT×0.85 + dune term)
  REGION_BLEND: smoothstep(0.04, 0.18, F2−F1)  (0 at borders, 1 interior)
  DUNES: F_DUNE = 23/1048576 phase on x, 42.0 m amplitude crescents

ANOMALIES (§12) — Voronoi cell 2048, P_ANOM = 0.06
  R_IN = 380  R_OUT = 620 (falloff smoothstep)
  terraces: (floor(r/70) + smoothstep(0.30,0.70,fract(r/70))) × 15 m
  spire:    250·exp(−(r/150)²)
  spiral:   16·sin(6θ + 0.05r)

CLIMATE (§9)
  T_BASE = 26.0 °C   L_LAT = 1.2e-5 /m   L_ALT = 0.0065 /m (lapse)
  M_BASE = 0.5       F_MOIST = 512/2^20 (1/2048)  M_AMP = 0.40
  Ocean-proximity moisture approximated by (1 − continent) term
  (documented substitution; identical CPU/GPU).

RAYMARCH (§16, §26)
  LAMBDA_STEP = 0.8   DELTA_MIN = 0.05   BISECT = 6
  STEP_GROWTH = 1 + t·0.0015   (adaptive, §26)
  MARCH_STEPS: Low 64 / Med 96 / High 128
  SHADOW_STEPS: Low 12 / Med 20 / High 28, footprint 4×Δp (§18)

LIGHT / SKY / ATMOS (§18–§22)
  SUN_DIR = normalize(0.55, 0.42, 0.72)   (y-up)
  SUN_COL = (1.00, 0.955, 0.885) × 3.4
  SKY_ZENITH (0.20, 0.36, 0.52)  SKY_HORIZON (0.78, 0.80, 0.82)
  CLOUD_PLANE = 1500 m, coverage fbm 4 oct @ 1/2600
  SIGMA_FOG = 9.0e-4 /m, scale height 900 m
  GAMMA 1/2.2 (demo only — Godot tonemapper owns this, §32)

PLAYER (§31)
  PLAYER_HEIGHT = 1.7   GROUND_BIAS = 0.08   GRAVITY = 22.0
  WALK_SPEED = 6.0  SPRINT = 1.8×  JUMP_V = 8.5  FLY_SPEED = 40
  SLOPE_MAX = 1.5 (gradient mag)  slide accel = (slope−SLOPE_MAX)·14
  Cam smoothing (§24): k_rot = 14 exp; spring k_s = 60, k_d = 2√k_s

QUALITY PRESETS
  Low:  res scale 0.35, 64 steps, 12 shadow
  Med:  res scale 0.50, 96 steps, 20 shadow
  High: res scale 0.75, 128 steps, 28 shadow

DEFAULT SEED = 1337

Hash (§4): murmur3 finalizer on
  h = xi ^ (yi·0x27D4EB2F) ^ (seed·0x9E3779B9)   [uint32 wrap]
  finalize: h^=h>>16; h*=0x85EBCA6B; h^=h>>13; h*=0xC2B2AE35; h^=h>>16
Value noise: quintic S(t)=6t⁵−15t⁴+10t³, corner values hash01 ∈ [0,1] → [−1,1]
Nyquist (§5): w(f,Δp) = clamp(2 − f·Δp/0.5, 0, 1); ridged uses 2f;
fBm renormalized × (Σa / Σa·w)
```

---

Task ID: 1
Agent: main (Z.ai Code)
Task: Environment inspection, plan, worklog + constants lock

Work Log:
- Read uploaded spec (34 KB) and user message spec (Godot-specific §31–34).
- Inspected project: Next.js 16.1.3, React 19, Tailwind 4, full shadcn/ui set,
  framer-motion, react-syntax-highlighter, recharts available. Dev server live
  on :3000.
- Defined coordinate convention, floating-origin strategy, dyadic frequency
  discipline, and the shared constant table above (parity contract).

Stage Summary:
- Constants locked. All implementations (TS port, WebGL2 shader, Godot shader,
  C++ CPU twin, GDScript fallback) MUST implement exactly these values.
- Next: TS math port + vista tuning, then WebGL2 shader, then launch subagents.
