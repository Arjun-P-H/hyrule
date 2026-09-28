# Hyrule Engine — Optimization & Hardening Report

- **Range:** Phase 0 = `b91e7c6` (frozen in [`bench/baseline/phase0.json`](../bench/baseline/phase0.json), see [`BASELINE.md`](../BASELINE.md)) → final = `78f6eee` and later.
- **Machine:** MacBook Air M1 (7-core GPU, fanless, 8 GB), macOS 26.6.2, headless Chrome 154 (ANGLE → Metal), Bun 1.3.14, Node 26.1.
- **Reproduce:** `bun run bench --suites=all` (A/B against the pinned Phase 0 revision is automatic), `bun run quality`.

**How to read the GPU numbers.** The M1 Air throttles within minutes, and during the final runs the
machine was also carrying other applications' CPU and GPU load. Every comparison below is therefore
made *inside one browser session* with both versions interleaved frame by frame, each at the preset
budgets it shipped with, and reported as a paired per-frame ratio with a 95 % bootstrap CI. The
harness's A/A test (Phase 0 against itself) gives ratios of 1.00 ± 0.03 and bit-identical frames.
Cool-start absolute numbers for Phase 0 are in `BASELINE.md` (Low 34.2 / Medium 89.6 / High 217.6 ms p50).

## Phase 0 vs final

| Axis | Phase 0 | Final |
| :--- | :--- | :--- |
| **GPU frame time** (60-frame fly-through, same session; p50 / p95 / p99 ms) | Low 36.5 / 54.9 / 60.0 · Medium 153.7 / 222.0 / 282.2 · High 393.0 / 534.5 / 628.4 | Low **30.2** / 43.8 / 49.2 · Medium **106.3** / 164.6 / 200.2 · High **249.7** / 364.9 / 445.1 |
| GPU time ratio, final ÷ Phase 0 (95 % CI) | — | Low **0.803** (0.784–0.822) · Medium **0.675** (0.661–0.690) · High **0.642** (0.625–0.659) |
| Image quality: pixels visibly off a converged ground-truth render (ΔE_ok > 0.02) | Low 20.4 % · Medium 5.55 % · High 1.60 % | Low **7.97 %** · Medium **1.98 %** · High **1.56 %** |
| **CPU tick** (median, idle machine, alternated runs) | `updatePlayer` 971–977 ns · `elevation` 424–428 ns | `updatePlayer` 949–965 ns (−1.2 / −2.3 %, within noise) · `elevation` 410–411 ns (−3 / −4 %) |
| **Bundle** first-load JS, `/game` (raw / gzip) | 591.7 KiB / 179.8 KiB | **535.5 KiB / 163.6 KiB** (−9 % gzip) |
| Bundle first-load JS, `/` | 558.9 / 169.7 KiB | 559.2 / 169.6 KiB |
| **Heap allocation**, 30 s scripted walk on `/game` | 917 – 919 KiB/s | **372 – 382 KiB/s** (−58 %) |
| V8 allocation in the physics tick | 3,589 B/tick (210 KiB/s at 60 Hz) | 3,589 B/tick (unchanged; boxed doubles, see WASM ADR) |
| **fp32/fp64 divergence**, GPU readback (E(x,z) unchanged) | not measured (the previous "2.7 × 10⁻⁷ m" compared identical fp64 values) | 0.64 mm at 3 km · 15 mm at 108 km · 29 mm at 268 km · 99 mm at 534 km; rebase seam ≤ 0.14 mm. `GROUND_BIAS` 0.08 m holds within ±300 km. |
| **TS strict errors** | 0 | 0 |
| **Circular deps** (madge, `src/`) | 0 (claimed, not in CI) | 0, enforced in CI |
| **Dead exports** (ts-prune, `src/`) | 5 framework entry points + 13 export-only-used-internally, unjustified | 5 + 10, every one justified in `bench/quality-gate.ts`, enforced in CI |
| **Tests** | 72 / 72 pass, 37,853 assertions, 1 file | **116 / 116** pass, **43,997** assertions, 5 files |
| **Coverage** (files loaded by tests, funcs / lines) | 66.7 % / 68.1 % | 73.3 % / 78.7 % |
| **Hardware floor** (Low preset, 30–45 s walks) | M1 DPR 1: 25.5 renders/s, GPU p50 22.6 ms, DRS fires 3×. M1 DPR 2: 7.8 renders/s, pinned at the DRS minimum. SwiftShader: 5.7 renders/s. | M1 DPR 1: 28.6 renders/s, GPU p50 19.5 ms, DRS fires 2× then holds. M1 DPR 2: **12.4** renders/s (physics 23 → 35 Hz), still at the DRS minimum. SwiftShader: 5.8 renders/s. |

No measured tier holds the Low preset at full resolution without DRS firing, before or after; the
M1 at DPR 1 now settles after two downscale steps instead of three. At DPR 1 physics runs at 58–60 Hz
thanks to the frame fence (24 Hz without it); at DPR 2 the compositor holds rAF to 35 Hz (23 Hz at Phase 0).

## Gates

| Phase | Gate | Status |
| :--- | :--- | :--- |
| 0 | Reproducible `bun run bench`, real numbers in `BASELINE.md` | Met — harness rebuilt; A/A 1.00 ± 0.03 |
| 1 | Per-eval `elevation()` cost lower; parity tests green | Met via the brief's second method: at a fixed step sequence (identical frames and eval counts) frame time fell 19–21 %. The isolated microbenchmark shows no significant change (0.998 / 0.992): the gain is in how the unrolled octaves compile inside the march loop. |
| 2 | Faster fly-through at equal-or-better perceptual score, all presets | Met against the converged ground truth; against Phase 0's own frames the image changes by design (Low renders terrain the old step budget cut off) |
| 3 | New tests; `GROUND_BIAS` justified in `constants.ts` by measured divergence | Met — dyadic registry, rebase fuzz to ±2,000 km, bias tied to a recorded GPU readback |
| 4 | Median `updatePlayer()` cost drops; zero net new allocations in the 60 Hz path | Allocation met (−58 % in-game, nothing new). Tick cost −1.2 / −2.3 % in two runs, inside run-to-run noise — not claimed. |
| 5 | `/game` first-load ≤ Phase 0; madge / ts-prune at zero; WASM recommendation with numbers | Met — −16.2 KiB gzip; zero unjustified; ADR rewritten from measurements (do not port) |
| 6 | `tsc` clean under strict; zero unjustified dead exports; § doc on every exported math function | Met |
| 7 | DRS does not thrash at the threshold; presets re-justified | Met — closed-loop tests on the engine's real controller; budgets re-profiled (march unchanged, shadow ↑ on Low/Medium) |
| 8 | Written audit per bullet | Met — `docs/AUDIT_PHASE8.md` |

## Changelog (measured deltas)

| Commit | Change | Measured delta |
| :--- | :--- | :--- |
| `2a5c72b` `784f509` `60142ca` | Measurement harness: calibrated GPU timer queries, interleaved A/B, drift sentinel, GPU parity readback, V8 allocation trace, play sessions, bundle-from-browser, ground-truth scoring | Replaced numbers that could not be measured by the old harness (underground frames, 15 samples, fabricated Spector snapshot, GC-masked "0 bytes") |
| `994db48` | Phase 1: unrolled fBm/ridged octaves; vectorized Worley (valley + region); anomaly neighbours pruned to R_out (both twins) | GPU 0.812 / 0.802 / 0.792, frames identical; CPU `elevation` −3–4 % |
| `4e52f9c` | Phase 2: dual horizon clamp (0.06t+2 near the surface, 0.12t+2 comfortably above it), both twins; march/shadow constants named and mirror-tested | 0.915 / 0.807 / 0.814 on top of Phase 1; off-truth Low 20.4 → 12.8 %, Medium 5.55 → 5.19 %, High 1.60 → 1.56 %; `marchRay` −8 % |
| `fff778f` | Phase 7: shadow steps Low 12 → 20, Medium 20 → 28 | Low +5.7 % time for 12.8 → 8.0 % off-truth; Medium +3.3 % for 5.2 → 2.0 % |
| `1231c33` | Phase 7: DRS controller extracted and tested in a closed loop | Behaviour-neutral; 0 reversals in all synthetic threshold loads |
| `9c75a30` | Phases 4/8: HUD telemetry via an external store; no per-frame closure; exhaustive mode switches | In-game allocation 907–911 → 374–382 KiB/s (−58 %); React DOM chunk −82 %; rAF 56 → 58.6 Hz |
| `61ce366` | Phase 5: engine core code-split from the `/game` shell, prefetched at module load | `/game` first-load −16.2 KiB gzip; first contentful paint ~−20 ms; first engine frame ~+80 ms on cold loads (n = 12, spread ±150 ms) |
| `2a26374` | Phase 3: dyadic registry, rebase fuzz, `GROUND_BIAS` tied to measured parity | +3 test suites; validity radius ±300 km documented |
| `0b01d46` | Wall-clock unit tests warm up and take the best of five | Flake rate 1 in 6 → 0 in 5 |
| `1afca45` | CI: quality gate (madge, ts-prune, tests, coverage), Bun pinned | Previously claimed checks now actually run |
| `1510ef0` `78f6eee` | Docs corrected where measurements contradicted them | — |

## Measured and rejected

| Idea | Result |
| :--- | :--- |
| Rigorous upper bound of E to prove clamp-limited steps (Phase 2 "coarse-to-fine bound") | Exact, but a bound cheap enough to help has ≥ 42 m slack; only 8 % of attempts succeed (83 % of steps are clamp-limited, but 57 % need ≤ 25 m slack). Net loss. |
| Caching lattice-corner hashes across march steps | Exact, but 28–70 % slower: register pressure and per-octave branches cost more than the hashes on this GPU |
| Restructuring the CPU twin for V8 inlining (to stop boxing) | Bit-identical, but slower in both V8 (551 → 687 ns) and JSC (386 → 441 ns) |
| Uniform 0.12t horizon clamp | Faster, but High fell to 1.83 % off-truth |
| Gradient refresh every 2nd step | +7.6 % time, no quality gain — the existing 3rd-step cadence is now justified by measurement |
| Fewer march steps on any preset | Drops each preset below its Phase 0 image quality |
| Prior-session WIP pieces: anomaly row hoisting, `exp`→`exp2`, ceiling clip | No measurable effect (1.00); the other two WIP changes were adopted |
| Build-time GLSL comment stripping (`.glsl` extraction) | Would save 2.9 KiB gzip; superseded by the engine split, not worth a custom Turbopack loader |

## Stretch items (not started — need sign-off)

- **Temporal reprojection / checkerboard.** The frame is GPU-bound at ~41 march iterations and 61–67
  `elevation()` calls per pixel; shading half the pixels per frame would roughly halve cost, i.e. about
  2× effective resolution at today's frame time. It changes how a frame is assembled and competes with
  DRS for the same budget, so it needs a design first.
- **CPU-precomputed lattice data in a uniform buffer.** Region/anomaly/continent/mask hashes around the
  camera computed once per frame on the CPU would be bit-exact and save ~15 % of hashes per
  evaluation, but it is a per-frame CPU→GPU dependency and a lookup table standing in for part of the
  math core.
- **True chunk-local GPU evaluation.** The live shader re-globalizes before evaluating noise, so fp32
  error grows with distance (99 mm at 534 km, beyond `GROUND_BIAS`). Evaluating on chunk-local
  coordinates with dyadic origin offsets would fix far-field precision; it changes how the GPU twin
  computes E and needs its own parity work.
- **Full WASM port** — not recommended; see `docs/WASM_EVALUATION.md`.
- **Mesh occlusion for anomaly clusters** — no evidence of need: after pruning, anomalies cost ~2.5
  hashes of ~96 per evaluation.

## Caveats

- One device. Tiers below the M1 were not measured; SwiftShader is the only other floor.
- Absolute GPU numbers in the final sessions include other applications' load; the interleaved ratios
  do not depend on it.
- In one of three final Low/DPR 1 sessions DRS reversed direction twice while another application was
  using the GPU; two dedicated reruns showed 0 reversals. The controller checks a single EMA sample
  after each dwell; requiring the condition to hold for the whole dwell would harden it further.
- The game page's "120-240 FPS WebGL2 Engine" badge is contradicted by every measurement here and was
  left unchanged.
