# TEST_INFRA — Continuous Procedural Terrain Engine Test Infrastructure

## 1. Overview & Architecture

This document defines the automated testing infrastructure for the procedural terrain WebGL2 browser game engine.

The test suite validates all system requirements:
- **R1: Strict Procedural Mathematics Invariance**: Procedural field mathematics $E(x,z)$, MurmurHash3 avalanche properties, dyadic frequencies $n/2^{20}$, C² quintic smoothing, Worley cellular distances, and Nyquist footprint weighting.
- **R2: Deterministic CPU Physics Twin & First-Person Controls**: Responsive WASD locomotion, sprint speed multiplier ($1.8\times$), upward jump momentum preservation, ground clamping with `GROUND_BIAS = 0.08m`, and slope slide-off trigger on steep ridges ($m > 1.5$).
- **R3: Raymarching & Soft Shadows**: Raymarching terrain intersection vs open-sky ray miss, penumbra soft shadow computation bounded in $[0, 1]$, and quality preset configurations.
- **R4: Engine Reliability & Code Quality**: Zero ESLint errors across all TypeScript and Next.js components, clean Next.js production builds.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Test Runner (bun test)                             │
│                       tests/terrainMath.test.ts                             │
└───────┬─────────────────┬───────────────────┬───────────────────┬───────────┘
        │                 │                   │                   │
        ▼                 ▼                   ▼                   ▼
┌───────────────┐ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Module 1    │ │   Module 2    │   │   Module 3    │   │   Module 4    │
│  MurmurHash3  │ │ Dyadic Freq.  │   │  Field E(x,z) │   │ Raymarching   │
│ & Determinism │ & Floating Orig.│   │  & Locomotion │   │ & Soft Shadows│
└───────────────┘ └───────────────┘   └───────────────┘   └───────────────┘
```

---

## 2. Test Execution

Run the complete test suite:
```bash
bun test
```

Run code quality and build verification:
```bash
npm run lint
npm run build
```
