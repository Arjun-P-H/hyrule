# Hyrule Engine — Phase 0 Performance & Quality Baseline

- **Timestamp:** 2026-09-27T06:50:00Z
- **Baseline Git Commit:** `e1dc7c674e2d3126f5d81e370cf9006509f6e5e8`
- **Environment:**
  - OS: macOS (Darwin 25.1.0, arm64, Apple M1)
  - Runtime: Bun v1.3.14 / Node v26.1.0
  - WebGL2 Driver: ANGLE (Apple, ANGLE Metal Renderer: Apple M1, Unspecified Version)
  - Hardware GPU Timer: `EXT_disjoint_timer_query_webgl2` active

---

## 1. Real Hardware GPU Frame Times

Measured using `EXT_disjoint_timer_query_webgl2` executing the full closed-form procedural raymarcher (`demoShader.ts`) along the checked-in camera spline (`bench/spline.ts`). The spline navigates across:
1. Plains Vista (Spawn, x=2016, z=-2592, e=45.8m)
2. Alpine Mountain Summit (x=4000, z=-4000, e=527.9m)
3. River Valley & Water Shoreline (x=-2000, z=-1700, e=15.8m, water=5.0m)
4. Zonai Anomaly Ruins & Terraced Spire (x=-3300, z=3900, e=534.7m, anom=11.95m)
5. Arid Desert Dunes (x=6500, z=6000, e=398.3m, duneFactor=0.85)

| Quality Preset | Viewport Resolution | March Steps | Shadow Steps | GPU Frame Time p50 | GPU Frame Time p95 | GPU Frame Time p99 | Mean Frame Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** | 448 × 252 (0.35×) | 64 | 12 | **28.05 ms** | 45.55 ms | 45.55 ms | 23.62 ms |
| **Medium** | 640 × 360 (0.50×) | 96 | 20 | **66.36 ms** | 204.82 ms | 204.82 ms | 63.05 ms |
| **High** | 960 × 540 (0.75×) | 128 | 28 | **140.82 ms** | 484.36 ms | 484.36 ms | 146.76 ms |

---

## 2. CPU Math & Physics Throughput

Isolated CPU benchmarks evaluated via `bun run bench`:

| Metric | Measured Value | Section VII Target | Status |
| :--- | :--- | :--- | :--- |
| **Elevation `E(x,z)` (dp=0, physics)** p50 | **0.334 µs** | < 1.000 µs | PASS |
| **Elevation `E(x,z)` (dp=0, physics)** mean | **0.668 µs** | < 2.000 µs | PASS |
| **Elevation Throughput (dp=0)** | **1,497.0k evals/sec** | > 500k/s | PASS |
| **Elevation `E(x,z)` (dp>0, raymarch filtered)** p50 | **0.420 µs** | < 1.000 µs | PASS |
| **Elevation `E(x,z)` (dp>0, raymarch filtered)** mean | **0.460 µs** | < 1.000 µs | PASS |
| **Slope `‖∇E‖` (4-tap finite difference)** mean | **1.427 µs** | < 10.00 µs | PASS |
| **Physics Locomotion Tick (`updatePlayer`)** mean | **0.980 µs** | < 500.0 µs | PASS |
| **CPU Raymarcher (`marchRay`, 96 steps)** | **43.61 µs** (0.044 ms) | < 200.0 µs | PASS |
| **Hillshade Radar Full Generation (200×200 = 40k pts)** | **16.46 ms** | < 50.00 ms | PASS |

---

## 3. Bundle Size & Steady-State Allocations

- **First-load JS bundle for `/game`:**
  - Game client component chunk (`app/game/page`): **10.0 KB** raw (**3.5 KB** gzipped)
  - Common framework bundle: **185 KB** raw (**58 KB** gzipped)
  - App shell runtime: **513 B**
  - Total first-load JS: **~238 KB** raw (**~78 KB** gzipped)
- **Production build time:** **3.2 s** (Next.js Turbopack) / **1.78 s** (standalone)
- **Steady-State Allocations:**
  - 100,000 elevation evaluations: **0 bytes** allocated
  - 100,000 physics ticks: **0 bytes** allocated
  - 30-second continuous simulation (1,800 ticks at 60 Hz): **0 bytes** allocated (**0.0 B/s** GC pressure)

---

## 4. Numerical Determinism & Floating-Origin Parity

- **10,000-point pseudo-random determinism delta:** **0** (identical bit-level reproduction)
- **Chunk boundary seam error across 8,192 m grid:** **2.7274 × 10⁻⁷ m** (< 1e-6 m boundary continuity)
- **CPU/GPU Ground Bias:** **0.08 m** (absorbs fp32 chunk-local / fp64 global difference)
- **Worked-example vista ridge raymarch convergence:**
  - Expected distance: 2,147.4 m
  - Bisected distance: **2,147.36 m** (< 0.05 m tolerance)

---

## 5. WebGL2 Pipeline & SpectorJS Snapshot

Checked into `bench/baseline/spector_snapshot.json`:
- **Draw Call Count per frame:** Exactly **1** (`gl.drawArrays(gl.TRIANGLES, 0, 3)`)
- **Meshes bound:** **0** (Zero-mesh invariant)
- **Textures bound:** **0** (Zero-texture invariant)
- **Framebuffers:** **1** (Default backbuffer, no intermediate blits)
- **Active uniforms updated per frame:** **10** (`u_cam_pos`, `u_chunk_origin`, `u_basis`, `u_time`, `u_res`, `u_tan_half`, `u_seed`, `u_march_steps`, `u_shadow_steps`, `u_footprint`)

---

## 6. Code Quality Metrics

- **TypeScript strict-mode errors (`tsc --noEmit`):** **0**
- **Circular dependencies (`madge --circular src/`):** **0**
- **Dead exports (`ts-prune`):** **0** unjustified (only Next.js framework entry points and contractual exports)
- **Unit test coverage:** **29 suites / 60 tests / 4,674 assertions — 100% PASS (0 failures)**

---

## 7. Baseline Gate Check

- [x] Scripted benchmark harness `bun run bench` exists and is reproducible.
- [x] Hardware GPU timer query measurement via `EXT_disjoint_timer_query_webgl2` implemented.
- [x] Checked-in camera spline through plains, mountains, river valley, Zonai anomaly, and dunes exists.
- [x] SpectorJS pipeline snapshot captured and verified.
- [x] Bundle analyzer and steady-state heap allocation profiles recorded.
- [x] Real baseline numbers recorded in `BASELINE.md` for diffing against subsequent phases.
