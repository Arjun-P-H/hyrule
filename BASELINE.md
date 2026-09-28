# Hyrule Engine — Phase 0 Baseline

- **Commit:** `b91e7c6f3680b1418295268ca7249bf2e9445b51` (engine sources `src/` identical to HEAD at session start)
- **Measured:** 2026-09-28 (bench run `2026-09-28T15:20:36Z`, play sessions merged at `15:5x`)
- **Machine:** MacBook Air (MacBookAir10,1) — Apple M1, **7-core GPU, fanless**, 8 GB, macOS 26.6.2, AC power
- **Browser / runtimes:** headless Chrome 154 (ANGLE → Metal), Bun 1.3.14, Node 26.1.0 (V8, allocation traces)
- **Machine-readable baseline:** [`bench/baseline/phase0.json`](bench/baseline/phase0.json) (every number below), frozen camera path [`bench/baseline/camera_path.json`](bench/baseline/camera_path.json), real SpectorJS capture [`bench/baseline/spector_capture.json`](bench/baseline/spector_capture.json)

Reproduce: `bun run bench --freeze --suites=all` (writes `phase0.json`; refuses a dirty `src/`).
Compare later work: `bun run bench` — A/B against the pinned revision is automatic.

## Why this replaces the previous BASELINE.md

The earlier baseline (and the Phase 1–8 claims built on it) came from a harness that could not
measure what it reported. Each defect below was reproduced before being fixed:

| Previous harness | What was actually happening | Now |
| :--- | :--- | :--- |
| 15 GPU samples; p95 = p99 = max | `floor(0.95·15)` indexes the last sample | 60 frames × 3 reps per preset, interpolated percentiles |
| Camera spline through 5 scenes | **28 of 60 samples were underground** (up to 404 m below terrain): cheap "inside the rock" frames | Path lifted to ≥ 20 m clearance, frozen to JSON |
| One timer query per 10 ms `setTimeout` | GPU idles between samples; first sample after idle reads ~2× slow | Time-based warm-up, back-to-back depth-1 queries; drift sentinel re-measures frame 0 every 10 frames |
| — | Two draws in flight **double** ANGLE/Metal `TIME_ELAPSED` (queue wait is counted) | Depth 1; calibrated: query vs per-draw wall clock agree within 2–5 % |
| Wall-clock cross-checks with stacked draws | Tile-based hidden-surface removal shades only the last of N opaque fullscreen draws | One draw per timed sample |
| `gpu_baseline.json` rewritten on every run, stamped `commit: "e1dc7c6"` | Baseline silently overwritten; label wrong | `--freeze` only; commit taken from git; A/B vs pinned rev |
| "SpectorJS snapshot" | A hard-coded object literal; `spector_capture.json` was `null` | Real capture of the production `/game` frame |
| "0 bytes allocated (Zero Allocation Verified)" | Clamped `heapUsed` deltas across GCs, measured in JSC (NaN-boxing) | V8 with GC-free windows (fixed 128 MB semispace, `v8.GCProfiler`), positive control |
| "Chunk boundary seam error 2.7e-7 m" | Compared `elevation((k−1)·8192+8192)` with `elevation(k·8192)` — the same fp64 number | GPU readback of the same world points from both floating origins |
| "fp32 emulation" parity test | Rounded the *inputs* to fp32, then ran the fp64 function | Real GPU `elevation()` read back through an RGBA32F target |

The A/A test (baseline shader against itself, interleaved) gives paired ratios 1.011 / 1.013 / 1.003
with 95 % CIs containing 1.0 and 100 % bit-identical frames: the harness resolves ~±2–3 % and the
perceptual-diff noise floor is exactly zero.

## 1. GPU frame time (real GPU time, `EXT_disjoint_timer_query_webgl2`)

60-frame fly-through (plains → alpine → river valley → Zonai anomaly → dunes → plains), 3 reps each,
1280×720 base resolution × preset scale, seed 1337.

| Preset | Resolution | March / shadow steps | p50 | p95 | p99 | mean |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Low | 448×252 | 64 / 12 | **34.19 ms** | 39.70 ms | 42.18 ms | 33.83 ms |
| Medium | 640×360 | 96 / 20 | **89.55 ms** | 103.19 ms | 108.57 ms | 88.25 ms |
| High | 960×540 | 128 / 28 | **217.63 ms** | 273.86 ms | 300.65 ms | 219.31 ms |

- Thermal: drift sentinel stable at Low (34.2 → 34.5 ms) and Medium (94 → 86 ms); **High throttles**
  (200 → 253 ms over the run), which inflates High's p95/p99. Interleaved A/B ratios are immune to this.
- Cost structure (per-pixel counters injected into `elevation()` and the march loop):

| Preset | `elevation()` calls / pixel | of which shadow | march iterations / pixel (budget) |
| :--- | ---: | ---: | ---: |
| Low | 65.3 | 3.9 | 49.3 (64) |
| Medium | 79.6 | 7.9 | 56.8 (96) |
| High | 83.2 | 11.0 | 57.1 (128) |

- Isolated math-core cost (16 `elevation()` calls/pixel, 256², 5 regions): **3.19 ns/eval** near
  (all octaves), **2.89 ns/eval** far (dp = 40).
- Pipeline (SpectorJS, production `/game`): **12 GL commands / frame, 1 draw call** (`drawArrays(TRIANGLES, 0, 3)`),
  0 textures, 0 buffers, 2 uniform uploads (dirty-checked). Zero-mesh / zero-texture invariant holds.

## 2. CPU twin (Bun/JSC, median over 25 alternated runs; IQR in brackets)

| Call | Median |
| :--- | ---: |
| `elevation(dp = 0)` (physics path) | **404.5 ns** [386.7–436.9] |
| `elevation(dp > 0)` (raymarch LOD) | 442.2 ns [418.1–448.7] |
| `gradient()` (4 taps) | 1,441.6 ns |
| `updatePlayer()` (60 Hz walk script, 55 % of ticks grounded) | **950.2 ns / tick** [946.0–962.2] |
| `marchRay()` (§30 worked-example ray) | 45.9 µs |

## 3. Bundle & memory

- First-load JS (production standalone build, gzip, measured from what Chrome downloads):
  `/` **559 KiB raw / 170 KiB gz** (10 files); `/game` **592 KiB raw / 180 KiB gz** (10 files).
  Largest `/game`-only chunk (engine + math core + shader): 78 KiB raw / 23 KiB gz.
  `@next/bundle-analyzer` in `next.config.ts` never runs: it only supports webpack, and Next 16 builds with Turbopack.
- V8 allocations of the CPU twin (GC-free windows, positive control registered):
  **3,589 B per `updatePlayer` tick = 210 KiB/s at 60 Hz**, **1,178 B per `elevation()` call** —
  boxed doubles (16-byte HeapNumbers) at every non-inlined call boundary (`elevation` 32 %, `vnoise` 27 %,
  `fbm` 23 %, `ridged` 12 %). JSC and SpiderMonkey NaN-box doubles and would show ~0; Chrome does not.
- In-game, 30 s scripted walk on `/game` (Medium, 1280×634 viewport, DPR 1): **917 KiB/s** allocated
  (V8 sampling profiler incl. collected objects), of which the engine chunk is 326 KiB/s and React/Next
  frameworks ~590 KiB/s (the 5 Hz `setStats` re-renders the whole `GamePage` tree).

## 4. Parity / correctness (real GPU readback vs CPU fp64)

The live shader re-globalizes before evaluating noise (`elevation(p.xz + u_chunk_origin)`), so fp32
precision is governed by distance from the world origin, not by chunk size.

| World distance | max \|E_gpu − E_cpu\| (p99) | of which fp32 arithmetic / coordinate rounding |
| :--- | ---: | :--- |
| 3 km | 0.64 mm (0.31 mm) | 0.64 mm / 0.32 mm |
| 12 km | 3.2 mm (2.1 mm) | 3.0 mm / 1.4 mm |
| 108 km | 15.0 mm (8.9 mm) | 13.0 mm / 11.7 mm |
| 1,067 km | **113.3 mm** (48.5 mm) | 62.2 mm / 83.7 mm |

- Rebase seam (the same world points evaluated from adjacent floating origins at 8.6, 98 and 1,000 km):
  **0.000 mm** — both sides round to the same fp32 global grid.
- `GROUND_BIAS = 0.08 m` covers the measured divergence out to ≥ 108 km with 5× margin; it is **exceeded at ~1,000 km**.
- Perceptual distance to the converged image (baseline shader at 160 march / 40 shadow steps), ΔE_ok:

| Preset | mean ΔE | pixels > JND (0.02) | pixels > 0.05 |
| :--- | ---: | ---: | ---: |
| Low | 0.0311 | **19.8 %** | 16.6 % |
| Medium | 0.0046 | 4.7 % | 2.6 % |
| High | 0.0007 | 0.9 % | 0.3 % |

  Low's 64-step budget ends the view at ~1.4 km (horizon clamp `0.06t + 2`), so distant terrain renders as sky.

## 5. Code quality

- `tsc --noEmit` (strict): **0 errors** · `madge --circular src`: **0 cycles**
- `ts-prune` (src/): 5 unused exports — all Next.js entry points (`default`, `metadata`, `GET`) — plus
  **13 exports used only inside their own module**
- Tests: **72 pass / 0 fail, 37,853 `expect()` calls, 1 file** · Coverage (files loaded by tests): 66.67 % funcs, 68.06 % lines
  (`terrainMath.ts` 100 % / 94.2 %; `demoShader.ts` 0 % funcs — `minifyGlsl` untested)

## 6. Hardware floor (30 s play sessions, production build)

| Device / setting | Renders/s | Physics ticks/s | DRS in 30 s | Final render size |
| :--- | ---: | ---: | :--- | :--- |
| M1, Medium, DPR 1 | 25.2 | 59.2 | 4 downscales, 0 reversals | 284×140 |
| M1, **Low**, DPR 1 | 26.0 | 59.9 | 2 downscales, 0 reversals (GPU p50 23.0 ms, p95 28.6 ms) | 275×136 |
| M1, Low, **DPR 2** (the Air's real display) | 11.7 | 34.8 | 7 downscales → pinned at DRS minimum scale | 314×176 |
| SwiftShader (no GPU) | 6.3 | 19.1 | 4 downscales | 67×33 |

No measured tier holds the Low preset without DRS firing. On the M1 at DPR 1, a 23 ms GPU frame
lands on 2–3 vsync intervals; the 20 FPS dips cross the 22 FPS downscale threshold.

Frame fence A/B (Medium, DPR 1): fence on → rAF/physics **59.2 Hz**, 25.2 renders/s; fence off → rAF
collapses to the GPU rate (**24.0 Hz**) at the same 24.0 renders/s. The fence costs no throughput here
and keeps input sampling and physics at 60 Hz.

## Gate

- [x] `bun run bench` exists, is reproducible (A/A: ratios 1.00 ± 0.03, frames bit-identical)
- [x] Real GPU timer-query numbers per preset, calibrated against wall clock
- [x] Checked-in camera path through plains, mountains, river valley, Zonai anomaly, dunes
- [x] Real SpectorJS capture checked in
- [x] Bundle sizes per route and a 30 s steady-state play-session allocation profile
- [x] Numbers written here with timestamp and exact commit
