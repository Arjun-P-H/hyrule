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

(NOTE: constants.ts is the AUTHORITATIVE source — the values below were
refreshed after the vista-tuning pass to match it exactly; Task 7 audit.)

```
WORLD / CHUNK
  CHUNK_SIZE      = 8192.0        (2^13)
  MAX_WORLD_H     = 1300.0        (early-exit ceiling, §26)
  MAX_DIST        = 6000.0        (march cutoff, hidden by fog)

MACRO TERRAIN (§5)
  F_MACRO  = 512/1048576   = 1/2048
  LAC_MACRO = 2.0   GAIN_MACRO = 0.5   OCT_MACRO = 9
  H_BASE   = 90.0          (max rolling-hills amplitude)

MOUNTAINS (§6)
  F_RIDGE  = 341/1048576  (~1/3074.6)
  LAC_RIDGE = 2.17  GAIN_RIDGE = 0.48  OCT_RIDGE = 5
  H_MOUNT  = 880.0
  F_MASK   = 128/1048576  = 1/8192    (mountain macro mask freq)
  MASK_LO  = 0.28   MASK_HI = 0.60    (smoothstep range on 0.5+0.5*N)

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
  CONT_LO = 0.22  CONT_HI = 0.44  (smoothstep → base × [0.2, 1])

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
  T_BASE = 14.0 °C   L_LAT = 1.2e-5 /m   L_ALT = 0.02 /m (game-tuned lapse)
  M_BASE = 0.5       F_MOIST = 512/2^20 (1/2048)  M_AMP = 0.40
  M_OCEAN = 0.35     (moisture boost scale for (1 − continent))
  Ocean-proximity moisture approximated by (1 − continent) term
  (documented substitution; identical CPU/GPU).

RAYMARCH (§16, §26)
  LAMBDA_STEP = 0.8   DELTA_MIN = 0.05   BISECT = 6   START_T = 0.4
  STEP_GROWTH = 1 + t·0.0015   (adaptive, §26)
  Horizon clamp: step ≤ t·0.06 + 2.0 (anti-tunnel)
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

---
Task ID: 7
Agent: general-purpose (parity reviewer)
Task: Cross-implementation parity audit

Work Log:
- Read worklog.md (constants lock + parity contract; noted its constants block is
  stale vs authoritative src/lib/constants.ts) and constants.ts in full.
- Read all five terrain implementations fully and line-by-line diffed:
  terrainMath.ts (reference), demoShader.ts, terrain_cpu.h,
  terrain_raymarch.gdshader, terrain_math_fallback.gd.
- Category A (constants): every numeric constant verified across all 5 files —
  F table, octaves/lac/gain, H_BASE/H_MOUNT 880, MASK 0.28/0.60, WARP 620,
  VALLEY 48/0.42/5.0/0.18, CONTINENT 0.22/0.44/0.2, REGION 8192/0.04/0.18/
  0.45/1.3/0.85/42, ANOMALY 2048/0.06/380/620/70/15/250/150/16/6/0.05,
  CLIMATE 14/1.2e-5/0.02/0.5/0.4/0.35, MARCH 0.8/0.05/6/0.0015/0.4,
  MAX_WORLD_H 1300, MAX_DIST 6000, PLAYER (1.7/0.08/22/6/1.8/8.5/40/1.5/14),
  SUN/SKY/CLOUD/FOG, presets 64/12, 96/20, 128/28 — all match constants.ts.
- Category B (hash chain): hashU order, hash3 mixing constants, cellPoint 12-bit
  packing /4095, cellType >>24 /255, hash01 /4294967295, seed offsets
  i*0x9e37 (fbm) vs i*0x85eb (ridged) — identical in all 5. Brute-force test
  (bun script): emulated GDScript _mul32/hash_u32 vs Math.imul — 0 mismatches
  over 200k random uint32; hash3 with negative lattice coords — 0 mismatches
  over 100k. Verified GLSL `i - (i/2)*2 == i%2` for i>=0 (analytic + 1000 cases).
- Category C (noise/fbm): quintic, bilinear expansion + *2-1, nyqW, fbm
  renormalisation asum/max(wsum,1e-6), ridged (amp 0.5, 2f, n^2) — identical.
- Category D (composition): order and raw-vs-warped inputs identical
  (region/cont/dunes/anomaly raw; macro/mask/ridge/valley warped); valley =
  C_DEPTH*sv^2; hMount region blend; continent multiplier — identical.
- Category E (march): START_T, early exits, dp clamp 1e-4, every-2nd-step
  gradient refresh, eps=max(dp,0.15), step/growth/horizon clamp t*0.06+2.0,
  bisection dp=tm*footprint — identical in all 5.
- Category F (shadow): t=0.6, 4x footprint, min(res,10h/t), 0.005 break,
  clamp(h*0.8,0.5,60)+t*0.04, GPU 40-cap loop — identical; noted C++
  soft_shadow is not ClassDB-bound (no unbounded caller possible).
- Category G (physics): k=1-exp(-12dt), jump 8.5, gravity 22, ground =
  elevation+0.08+1.7, vy<0-only zeroing, slope gate (m-1.5)*14 — identical in
  TS/C++/GDScript.
- Category H (API): get_elevation(x,z,seed), update_player(state,input,dt,seed)
  dict keys forward/strafe/jump/sprint/yaw, march_ray arg order — consistent
  between terrain_cpu.cpp, fallback, and player_controller call sites.
- Category I (WebGL harness): all 10 uniform names match; basis columns
  (right/up/forward, orthonormal — verified up=f x r); floating origin
  8192*floor; footprint 2*tanHalf/canvas.height; fly clamp via CPU twin.
- Category J (Godot shader): varyings, POSITION override, mat3(v_r,v_u,v_b)
  reconstruction, transpose for world->view, dirView z=-1, DEPTH=clip.z/clip.w,
  miss DEPTH=0, uint ops, atan2, dynamic breaks, struct returns — all correct
  Godot-4 shader language.
- Category K/L (GDScript + TS/React sanity): group names ("floating_origin",
  "rebasable"), PRESETS keys vs quality enum, hooks deps/cleanup/null checks.
- Ran `bun run lint` — 1 error (react-hooks/set-state-in-effect in
  WorkedExample.tsx:52), exit code 1. Reported verbatim, not fixed (review-only).

Stage Summary:
- Verdict: NOT parity clean — 15 findings (1 BLOCKER, 3 MAJOR, 11 MINOR).
- BLOCKER: terrain_cpu.h:498 references undefined `MAXH` (header defines
  MAX_WORLD_H) → GDExtension cannot compile. Fix: use MAX_WORLD_H.
- MAJOR: player_controller.gd fly mode never syncs state["x"/"y"/"z"] →
  teleport when toggling fly->walk. MAJOR: camera rig (top-level) is not in
  the "rebasable" group → 8 km camera swoop on every chunk rebase.
  MAJOR: WorkedExample.tsx:52 lint error fails `bun run lint`.
- MINOR: worklog constants block stale (OCT_MACRO, H_MOUNT, MASK, CONT, T_BASE,
  L_ALT, M_OCEAN, START_T); SUN pre-normalised digits off by ~2e-6; fallback
  missing W_WIDTH const and get_constants "fly_speed" key; reversed-edge
  smoothstep calls (GLSL-spec UB, works in practice) in both GPU shaders;
  main.gd uses fo.origin.y for the z axis; pitch-clamp mismatch between
  harnesses (±1.45 vs [-1.2,1.35]); TerrainDemo keys not cleared on window
  blur; terrainMath hard-codes 1300/6000 literals; fly-mode tuning constants
  (x2.5, k=8, +1.9, 1150) not in the constants lock (mutually consistent).
- Math parity across the five field implementations: CLEAN (bit-identical hash
  chain, identical constants/formulas/order). Issues found are in the C++
  build, Godot player integration, and web lint — not in the field math.

---
Task ID: 6
Agent: frontend-styling-expert
Task: Build / route spec console page + FileBrowser

Work Log:
- Read worklog (constants lock + parity contract), constants.ts, terrainMath.ts
  header, and the prop surfaces of TerrainDemo / HillshadePanel / WorkedExample
  plus /api/files (tree + single-file GET, root-locked to godot-project/).
- Created src/components/FileBrowser.tsx ('use client'): fetches /api/files on
  mount, defaults to README.md, per-path content cache (Map + stale-response
  token guard), lg:grid-cols-[280px_1fr] Card layout. Left pane: max-h-96
  file list with FileCode2 icons colour-coded by extension (.h/.cpp amber,
  .gdshader/.ts emerald, .gd stone-400, .md stone-300), selected row
  bg-emerald-950/40 + border-l-2 border-emerald-500, min-h-9 touch targets.
  Right pane: mono path + lines/size badges + Copy button (copied state,
  Check icon), line-numbered <pre> (sticky-left stone-600 gutter,
  max-h-[560px] overflow-auto, text-[11px] whitespace-pre), escape-first
  comment tokenizer (// for C-family/GLSL, # for .gd, -- for lua/sql) via
  dangerouslySetInnerHTML — no new deps. Thin scrollbars via Tailwind
  arbitrary values only ([scrollbar-width:thin], webkit variants); globals.css
  untouched. Loading skeletons (animate-pulse) + error states.
- Created src/app/page.tsx ('use client', holds seed/preset state only —
  TerrainDemo owns all interaction): sticky blurred header (emerald W(x,y)
  mark, anchor nav, Godot 4 · Metal · SDFGI badge); hero (2-col lg) with
  4 mono stat cards + HillshadePanel card; live demo section with seed Input
  + Dices random button, LOW/MED/HIGH preset buttons from QUALITY_PRESETS
  with resScale/steps tooltips, engine-constant chips (CHUNK_SIZE/PLAYER from
  constants), <TerrainDemo seed preset /> and 3 hint cards; worked example
  section; math section with 10 formula cards (§4–§33) + BIOME_ANCHORS table
  with live OKLab→sRGB swatches; §29 PARAM_TABLE grouped, max-h-[480px]
  scroll container with sticky thead (symbol emerald / value amber / spec
  Badge outline); parity section (5-row implementation table, amber
  divergence-budget note card, 6 TBDR/SDFGI constraint cards); files section
  with FileBrowser; mt-auto footer. Stone-950/emerald/amber only — zero
  blue/indigo UI chrome. Subtle framer-motion whileInView fades on sections.
- Responsive fixes found by headless-browser testing at 375px: wrapped
  <WorkedExample /> in overflow-x-auto (its own table min-content exceeds
  the viewport; that file is off-limits) and added min-w-0 to flex/grid
  items hosting min-w-[560px] tables (biome + parity cards) — mobile
  horizontal overflow eliminated (scrollWidth 375 == clientWidth 375).
- Verified in headless Chromium: FileBrowser loads tree (14 files), default
  selects README.md (140 lines, 6.5 kB), clicking terrain_raymarch.gdshader
  renders 651 lines with 148 comment-token spans, preset buttons toggle
  aria-pressed/emerald state, random-seed button updates the input
  (1337 → 284692), no page errors, no WebGL failure overlay.

Stage Summary:
- Files created: src/app/page.tsx (871 lines), src/components/FileBrowser.tsx
  (349 lines). No other file touched (worklog append excepted).
- Lint: `bunx eslint src/app/page.tsx src/components/FileBrowser.tsx` →
  0 errors, 0 warnings. Project-wide `bun run lint` still fails on ONE
  pre-existing error in src/components/terrain/WorkedExample.tsx:52
  (react-hooks/set-state-in-effect) — file owned by a previous agent, not
  modified per contract.
- dev.log: page compiles clean (✓ Compiled, GET / 200, GET /api/files 200,
  GET /api/files?path=... 200); no compile errors for either new file.
- Emitted CSS verified to contain the arbitrary utilities
  (scrollbar-width:thin, scrollbar-color, grid-template-columns:280px 1fr,
  max-height:480px/560px); footer pinned via min-h-screen flex-col + mt-auto.

---
Task ID: 2, 3
Agent: main (Z.ai Code)
Task: WebGL2 demo shader + interactive harness (GPU parity proof)

Work Log:
- src/components/terrain/demoShader.ts — full GLSL ES 3.00 port of
  terrainMath.ts: uint32 hash chain, quintic vnoise (identical bilinear
  expansion), Nyquist-weighted fbm/ridged, Worley F2−F1, domain warp,
  region/dune/anomaly fields, master elevation, Lipschitz march +
  6-step bisection + water-plane tracking, 4×Δp soft shadows, OKLab
  biome RBF, Beer-Lambert fog, sky + cloud deck, gamma 1/2.2 (demo only).
- src/components/terrain/TerrainDemo.tsx — walk/fly harness. Walk mode runs
  updatePlayer() (the §31 twin) verbatim; fly mode clamps altitude against
  the twin. Floating origin recomputed per frame (8192·floor), LH camera
  basis matching the twin's yaw convention. Fence-sync frame pacing +
  adaptive resolution (SwiftShader/llvmpipe auto-detect) added during
  browser verification to keep software-GL from backlogging the compositor.

Stage Summary:
- Same-field parity harness live at / ; verified end-to-end in headless
  Chromium (SwiftShader): pixel readback shows sky→terrain gradient,
  spawn pos [2016, 76.8, −2592], chunk-origin [0, −8192].

---
Task ID: 4-a, 4-b, 4-c, 4-d
Agent: main (Z.ai Code)
Task: Complete Godot 4 project (the primary deliverable)

Work Log:
- gdextension/src/terrain_cpu.h — double-precision reference twin; full
  contract constants, hash/vnoise/fbm/ridged/voronoi/warp/regions/dunes/
  anomalies/elevation/gradient/climate/march_ray/soft_shadow/update_player.
  Documented TBDR/SDFGI rationale + -ffp-contract=off parity flag.
- gdextension/src/terrain_cpu.cpp — TerrainCPU GDExtension class
  (get_elevation/gradient/normal/climate, query_anomaly, march_ray,
  update_player Dictionary API, get_constants).
- gdextension/src/register_types.{h,cpp}, SConstruct (arm64, ffp-contract
  off), terrain_cpu.gdextension manifest.
- shaders/terrain_raymarch.gdshader — spatial shader, cull_disabled +
  fog_disabled + depth_draw_always; fullscreen quad via POSITION override;
  camera/proj plumbing as varyings; DEPTH = clip.z/clip.w (reversed-Z
  Metal NDC) for hits, 0.0 for sky; NORMAL view-space; ALBEDO for SDFGI
  voxelization; EMISSION carries direct sun + fog inscatter; light-cull-
  mask note; TBDR + SDFGI constraint block in the header.
- scripts/floating_origin.gd (§25 rebase, group "rebasable"),
  terrain_renderer.gd (render_priority −100, layers 2, giant AABB,
  per-frame uniforms), player_controller.gd (§24 spring camera, §31
  movement, mouse capture), terrain_math_fallback.gd (pure-GDScript twin,
  identical Dictionary API, int64-masked uint32 via 16-bit split mul),
  main.gd (SDFGI env, sun with cull mask 4294967293, proof props, HUD).
- project.godot (forward_plus, MSAA 0 + FXAA 1), scenes/main.tscn, README
  (build steps, parity table, spec→file cross-reference, validation).

Stage Summary:
- godot-project/ complete; runs without the extension via the fallback
  twin; C++ path is the shipping configuration.

---
Task ID: 5, 8
Agent: main (Z.ai Code)
Task: Files API + parity-review fixes + browser verification

Work Log:
- src/app/api/files/route.ts — traversal-safe code browser backend for
  godot-project/ (tree + content).
- Applied Task 7 findings: terrain_cpu.h MAXH→MAX_WORLD_H (blocker);
  player_controller.gd fly→walk state sync + rig added to "rebasable";
  WorkedExample restructured to lazy useState (lint gate);
  reversed-smoothstep fixes + SUN digits (0.550744/0.420568/0.720974) in
  both shaders + main.gd; fallback W_WIDTH/FLY_SPEED constants;
  TerrainDemo blur→keys.clear(); terrainMath MAX_WORLD_H/MAX_DIST imports;
  main.gd origin.z fix; PARAM_TABLE + worklog constants block refreshed.
- Agent-Browser verification: desktop + 375px mobile (no overflow), hero
  hillshade renders, raymarch canvas pixel-verified (sky [238,247,254] →
  terrain [95,136,109]), HUD correct, HIGH preset + random seed + file
  browser interactions work, §31 twin verified live (F→WALK→airborne→
  grounded, Space jump, held-key re-jump), sticky footer bottom===vh.
  bun run lint clean; dev.log clean.

Stage Summary:
- All Task 7 blockers/majors fixed; site browser-verified interactive.
  Open item for the user: compile the GDExtension on a Mac (scons
  platform=macos arch=arm64) — everything else runs as-is.
