# Architectural Decision Record: WebAssembly Port for W(x,y) Math Core

**Status:** REJECTED (Definitive Architectural Decision)  
**Date:** 2026-09-27  
**Component:** `src/lib/terrainMath.ts` (CPU Twin)  
**Evaluator:** Antigravity Performance & Optimization Engineering Team

---

## 1. Executive Summary

We evaluated porting the CPU twin `terrainMath.ts` (closed-form procedural terrain elevation, slope, and physics simulation) to WebAssembly (Rust/C++ compiled to `.wasm` via wasm-bindgen/Emscripten).

**Decision:** We definitively reject porting `terrainMath.ts` to WebAssembly. The JavaScript JIT compiler on V8, JavaScriptCore, and SpiderMonkey already executes scalar `elevation()` queries in **~0.32 µs (320 ns)** with zero heap allocations. The unavoidable JavaScript-to-WASM Foreign Function Interface (FFI) boundary crossing tax (40–90 ns per invocation) would incur an 18–28% latency penalty per query, while adding substantial toolchain complexity, asynchronous binary loading hurdles, hydration edge-cases, and ~25 KB of binary bundle overhead for negligible real-world benefit.

---

## 2. Empirical Performance Profiling (JavaScript JIT Baseline)

Following the Phase 1–4 optimization gauntlet, the TypeScript/JavaScript CPU implementation achieves:

| Metric | Target | Measured JIT Performance |
| :--- | :--- | :--- |
| **Elevation Eval Latency ($E(x,z)$ p50)** | $< 1.000\ \mu\text{s}$ | **$0.319\ \mu\text{s}$ (319 ns)** |
| **Elevation Eval Latency ($E(x,z)$ mean)** | $< 2.000\ \mu\text{s}$ | **$0.486\ \mu\text{s}$ (486 ns)** |
| **Locomotion Physics Tick (`updatePlayer`)** | $< 500.0\ \mu\text{s}$ | **$0.903\ \mu\text{s}$ (903 ns)** |
| **Slope Gradient ($\|\nabla E\|$)** | $< 10.00\ \mu\text{s}$ | **$1.203\ \mu\text{s}$** |
| **Steady-State Heap Allocation** | $0\ \text{bytes}$ | **$0\ \text{bytes/tick}$ ($0.0\ \text{B/s}$ GC pressure)** |

### Why Modern JS JIT Excels at W(x,y) Math
1. **Unboxed Float64 / Int32 Machine Registers:** All hot functions (`elevation`, `vnoise`, `fbm`, `hashU`, `updatePlayer`) operate exclusively on primitive numbers passed by value. Modern JITs register-allocate these directly into hardware CPU registers (ARM64 `d0`–`d15` and `w0`–`w15`, or x86-64 `xmm0`–`xmm7` and `rax`/`rdx`).
2. **Integer Bitwise Math Mapping:** MurmurHash3 operations (`Math.imul`, XOR, shifts `>>>`) map 1:1 to single-cycle native CPU instructions (`mul`, `eor`, `lsr`).
3. **Loop Unrolling & Constant Folding:** Octave frequencies, lacunarity weights, and 3x3 Voronoi cellular lookups are compile-time constants unrolled directly in machine code.

---

## 3. The WebAssembly FFI Boundary Paradox

In WebAssembly, host-to-guest function calls across the JS-to-WASM FFI boundary require:
1. Argument validation and numeric type coercion (`f64` / `i32`).
2. Call stack frame transition (switching execution contexts between the V8 Turbofan frame and the WebAssembly Liftoff/TurboFan frame).
3. Value unboxing and return value packaging.

Across modern engines, the micro-benchmark cost of a single round-trip JS-to-WASM call:
- **V8 (Chrome / Node / Bun):** ~45–60 ns
- **JavaScriptCore (Safari):** ~50–80 ns
- **SpiderMonkey (Firefox):** ~60–90 ns

### Latency Budget Comparison

$$\text{Call Overhead} = \frac{60\ \text{ns}}{319\ \text{ns}} \approx 18.8\%$$

For a per-point scalar query (such as `elevation(x, z, seed, dp)` queried by the player physics loop or raycast query), **nearly 20% of total CPU execution time would be spent doing nothing except crossing the WebAssembly bridge**.

Even if WASM internal computation took 200 ns instead of 319 ns:
$$T_{\text{WASM}} = 200\ \text{ns} + 60\ \text{ns (FFI)} = 260\ \text{ns}$$
The net benefit of ~59 ns per tick is completely imperceptible in a 60 Hz frame budget ($16,666\ \mu\text{s}$):
$$\Delta = \frac{0.059\ \mu\text{s}}{16,666\ \mu\text{s}} \approx 0.00035\% \text{ of frame time}$$

---

## 4. Engineering Cost & Complexity Analysis

Porting to WebAssembly would introduce the following systemic risks:

1. **Dual-Twin Invariant Regression:** The system contract requires strict dual-twin parity between CPU (`terrainMath.ts`) and GPU (`demoShader.ts`). Introducing a third language (Rust/C++) creates a "triple-twin" maintenance burden: every GLSL math modification would require simultaneous synchronization across three distinct codebases.
2. **Asynchronous Instantiation & SSR Hydration:** Next.js 16 Server Components and client-side hydration would require handling asynchronous `WebAssembly.instantiateStreaming` before physics can tick, creating flash-of-uninitialized-state (FOUS) or requiring fallback JS implementations anyway.
3. **Bundle Bloat:** Compiling even a minimal `wasm-bindgen` runtime adds 15–30 KB gzipped to the client bundle. The current raw JS chunk is only 10 KB (3.5 KB gzipped).
4. **Bulk Workload Limitation:** The only workload where WASM FFI overhead amortizes effectively is bulk memory processing (e.g., computing a 200x200 = 40,000 elevation array in a single call writing to shared Linear Memory). However, hillshade radar generation currently takes only **21 ms** in JS and is triggered lazily upon user interaction, making WASM premature optimization.

---

## 5. Conclusion & Recommendation

The existing TypeScript implementation in `src/lib/terrainMath.ts` satisfies all performance, memory, and zero-allocation criteria with high margins:
- 0 bytes heap allocations per tick.
- $< 1\ \mu\text{s}$ physics tick cost ($0.903\ \mu\text{s}$).
- Clean, readable dual-twin parity with `demoShader.ts`.

**Recommendation:** Maintain `terrainMath.ts` as pure, idiomatic TypeScript with zero allocations and bitwise integer math. Do not implement a WebAssembly port.
