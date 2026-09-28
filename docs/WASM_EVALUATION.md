# ADR: WebAssembly port of the CPU twin (`terrainMath.ts`)

**Status:** Not recommended now — revisit only on the triggers below.
**Date:** 2026-09-28 · supersedes the 2026-09-27 version, whose FFI and tick figures were not measured.
**Evidence:** `bun run bench --suites=cpu,alloc`, the play-session harness (`bench/play.ts`), and a
hand-assembled `(f64, f64) → f64` wasm module timed in both engines (Apple M1, Bun 1.3.14, Node 26.1).

## The question

Phase 5 asks whether `elevation()` / `gradient()` are a meaningful fraction of the frame budget on
low-end devices, and if so what a WASM port would buy against its cost.

## Measurements

| Quantity | Value | Source |
| :--- | ---: | :--- |
| `updatePlayer()` per 60 Hz tick (walk script, 55 % grounded) | **0.95 µs** (JSC) | `bench/cpu.ts`, median of 25 alternated runs |
| `elevation(dp = 0)` | 0.41 µs (JSC) · 0.55 µs (V8) | `bench/cpu.ts`, `bench/alloc.ts` |
| GPU frame time, Low / Medium / High | 34 / 90 / 218 ms at Phase 0; now 0.80 / 0.68 / 0.64 × that (interleaved A/B) | `bench/gpu` timer queries |
| CPU tick as a share of a 60 Hz frame | **0.006 %** | 0.95 µs / 16,667 µs |
| JS → wasm call overhead | **4.9 ns** (JSC) · **11.0 ns** (V8) | 20 M calls, median of 7 runs |
| V8 allocation in the tick | 3.6 KB/tick = 210 KiB/s | boxed doubles at non-inlined calls (`bench/alloc.ts`) |

The previous ADR's "45–90 ns per crossing" was off by ~10×. The crossing is 1–3 % of one
`elevation()` call and would be irrelevant if the whole tick (5 evaluations) moved to wasm.

## What a port would buy

- **Frame time: nothing measurable.** Even a 2× faster tick saves ~0.5 µs per frame; the binding
  constraint is the GPU by four orders of magnitude.
- **Allocation (V8 only): ~210 KiB/s less garbage.** f64 math inside wasm is never boxed, so the tick
  would allocate ~16 B (the boxed return) instead of 3.6 KB. At 210 KiB/s V8's young-generation
  scavenger runs every few seconds for well under a millisecond; after the Phase 4/8 telemetry fix the
  whole `/game` page allocates ~380 KiB/s. JSC and SpiderMonkey NaN-box doubles and allocate nothing
  here today.

## What it would cost

- **A third implementation of E(x,z).** The dual-twin contract (GLSL fp32 ↔ TS fp64, pinned by
  `tests/mathCore.test.ts` and the GPU parity probe) becomes a triple-twin; every numeric change lands in
  three languages.
- **Toolchain:** Rust + wasm-bindgen or AssemblyScript alongside Bun/Next/Turbopack, and a second test
  runner for the wasm build.
- **Startup:** asynchronous instantiation before the first physics tick (a JS fallback would keep the
  TS twin alive anyway).
- **Bundle:** a math-only wasm module with bindings is typically 10–40 KB gzip — comparable to or larger
  than the whole engine chunk today (78 KB raw / 23 KB gzip).

## Recommendation

Keep the TypeScript twin. Revisit if any of these becomes true:

1. A CPU consumer needs thousands of `elevation()` queries per frame (e.g. CPU collision for many
   entities, AI path costs over E, or an offline map bake) — then batch them through one wasm call
   writing to linear memory, where the crossing amortises to nothing.
2. Chrome GC pauses attributable to the tick show up in play-session traces (they do not today).
3. The CPU tick exceeds ~2 % of the frame budget on the target floor device.
