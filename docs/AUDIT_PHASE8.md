# Phase 8 — UI, controls, and test/file structure audit

Audit of the live repository (2026-09-28), each finding backed by a measurement or a pointer to code.
Numbers come from `bun run bench` suites and `bench/play.ts` (30 s scripted walks on the production
build, headless Chrome on an Apple M1).

## 1. Render-loop isolation — issue found, fixed

**Requirement:** the WebGL loop runs on its own `requestAnimationFrame`; 60 Hz data that only affects
the canvas must not flow through React state.

**Finding.** The loop itself was isolated: `useTerrainEngine` self-schedules `requestAnimationFrame`,
physics and uniforms mutate plain records and a `Float32Array`, and uniform uploads are dirty-checked
(SpectorJS: 12 GL commands and 1 draw call per frame, 2 uniform uploads). But HUD telemetry was
published with `setState` from inside the page component every 200 ms, so each publish re-rendered the
entire `/game` tree — Radix tooltips, icon buttons, the settings dialog — and the frame loop allocated a
fresh `emitStats` closure on every rAF.

**Fix (9c75a30).** The engine writes snapshots into a small external store
(`src/lib/engine/statsStore.ts`); `HudOverlay` subscribes with `useSyncExternalStore` and the page
subscribes only to the camera mode, which changes on user action. `publishStats` is created once.
Alternated A/B against the Phase 0 build: total JS allocation **907–911 → 374–382 KiB/s (−58 %)**,
React DOM chunk **222 → 41 KiB/s (−82 %)**, rAF 56 → 58.6 Hz.

## 2. Input latency — fine, with one measured confirmation

Keyboard state is a bitmask updated by window listeners and decoded at the top of each frame, before
physics; pointer-lock mouse deltas mutate `cam.yaw/pitch` directly in the event handler. Nothing is
routed through React scheduling.

The frame fence matters here: with `fenceSync` stubbed out, Chrome throttles rAF to the GPU's render
rate, so input sampling and physics drop to **24 Hz**; with the fence the engine skips submitting a new
frame while the GPU is busy and rAF — and the physics tick — stay at **59 Hz** at the same 24–26
renders/s. On this integrated GPU the fence costs no throughput (Phase 7 A/B), so it stays.

## 3. Test organization — reorganized forward, monolith kept

`tests/terrainMath.test.ts` (1.4 k lines, 31 `describe` blocks) is kept: splitting it by § would churn
blame for no behavioural gain, and `bun test` already runs every file as one command. New tests were
written as topic files instead:

| File | Covers |
| :--- | :--- |
| `tests/mathCore.test.ts` | §12 anomaly pruning exactness (Object.is vs an unpruned search), march/shadow constants mirrored GLSL↔TS, preset budgets |
| `tests/floatingOrigin.test.ts` | §25 dyadic-frequency registry, rebase fuzz to ±2,000 km, GROUND_BIAS vs the recorded GPU parity measurement |
| `tests/drs.test.ts` | §28 DRS closed-loop thrash tests on the engine's real controller |
| `tests/statsStore.test.ts` | §30 telemetry store re-render semantics |

Two hygiene fixes along the way: suite 30 tested a copy of the DRS logic pasted into the test file
(now calls the engine's module), and the suite-17 wall-clock thresholds flaked under load (now warm up
and take the best of five). 116 tests, 43,997 assertions (Phase 0: 72 / 37,853).

## 4. CI — runs on every push and PR; gaps closed

`.github/workflows/ci.yml` already ran typecheck, tests, the CPU bench and the production build. Found
and fixed:

- `bun-version: latest` → pinned to 1.3.14 (bench and test behaviour are version-sensitive).
- `madge` and `ts-prune` were claimed as CI gates but never ran; `bun run quality` now fails on any
  circular dependency, strict-mode error, failing test or unjustified dead export.
- The bench step would have failed on a shallow checkout once the harness began A/B-ing against the
  pinned Phase 0 revision; it now runs as a smoke test (`--against=none`).
- GPU suites (timing, parity, frames, play sessions) need a real GPU and run locally with
  `bun run bench`; they are not in CI.
