# Phase 8 — UI, Controls & Test/File Structure Audit

This document records the architectural findings for Phase 8 of the Hyrule Continuous World Engine optimization gauntlet.

---

### 1. Render-Loop Isolation
- **Requirement:** The WebGL canvas must run its own `requestAnimationFrame` loop, fully decoupled from React's render cycle. 60 Hz physics data must not flow through `useState` or trigger React component re-renders.
- **Finding:** Verified and Hardened.
- **Evidence:**
  - In `src/lib/engine/useTerrainEngine.ts`, the simulation runs within a self-scheduling `requestAnimationFrame(frame)` loop completely isolated from React's commit phase.
  - Per-frame state updates (player positions, velocities, orientation matrices, chunk origins, camera bases) are mutated directly on local typed arrays and scalar records (`Float32Array(9)`, `cam.fly`, `player.grounded`).
  - WebGL uniform calls (`gl.uniform*`) are dirty-checked per-frame against previous scalar values, executing driver calls only when uniforms actually mutate.
  - React telemetry reporting (`setStats`) is explicitly throttled to a 200 ms interval (`statsAt = now; if (now - statsAt > 200)`), limiting React re-renders to a maximum of 5 Hz regardless of display refresh rate (60 Hz to 240 Hz).

---

### 2. Input Latency
- **Requirement:** Pointer-lock mouse-look and keyboard inputs (WASD, Sprint, Jump) should be sampled synchronously at the top of each physics tick without being delayed or batched through React scheduling.
- **Finding:** Verified and Hardened.
- **Evidence:**
  - Raw window event listeners (`keydown`, `keyup`) update an integer bitmask (`keyMask`) using bitwise OR/AND NOT operations.
  - Pointer lock `mousemove` events mutate `cam.yaw` and `cam.pitch` directly on the local camera record in native window coordinate space.
  - At the beginning of each animation frame, the physics tick decodes `keyMask` and camera angles synchronously before performing the semi-implicit Euler integration, guaranteeing minimal input-to-photon latency.

---

### 3. Test Organization
- **Requirement:** Assess whether `tests/terrainMath.test.ts` (31 suites / 72 tests / 37,853 assertions) should be split across multiple files or retained as a monolithic suite.
- **Finding:** Audited. Retained as a single, indexed suite with 31 cleanly sectioned `describe` blocks.
- **Evidence:**
  - Bun executes the entire 37,853-assertion test suite in **~420 ms** in a single process without module re-evaluation overhead.
  - Each `describe` block is strictly numbered (§1 to §31), mapping 1:1 with specification chapters (§4 to §33) and JSDoc tags in `src/lib/terrainMath.ts`.
  - Retaining a unified test file prevents cross-file helper fragmentation and ensures deterministic serial test execution in CI without parallel worker race conditions.

---

### 4. Continuous Integration (CI)
- **Requirement:** Ensure `bun test`, `bun run bench`, and `tsc --noEmit` run automatically in CI on every push and pull request.
- **Finding:** Verified and Operational.
- **Evidence:**
  - `.github/workflows/ci.yml` is active on `main` and feature branches.
  - CI steps execute:
    1. `bun install --frozen-lockfile`
    2. `bun x tsc --noEmit` (TypeScript strict mode gate)
    3. `bun test` (Full 37,853-assertion parity test suite)
    4. `bun run bench --cpu-only` (Physics throughput and determinism regression gate)
