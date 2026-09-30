# Project: W(x, y) Infinite Procedural Terrain WebGL2 Browser Engine

## Architecture
The engine renders an infinite, continuous procedural world $W(x, y)$ purely in the browser using a high-efficiency hardware-accelerated WebGL2 raymarching pipeline with an exact double-precision TypeScript physics and procedural math twin (`terrainMath.ts`). All desktop electron and Godot dependencies have been removed.

Measured performance (GPU timer queries, fixed fly-through, Apple M1 7-core GPU) is in
[`BASELINE.md`](BASELINE.md) (Phase 0) and [`docs/OPTIMIZATION_REPORT.md`](docs/OPTIMIZATION_REPORT.md)
(current): tens of milliseconds per frame at the preset resolutions, with dynamic resolution scaling
holding interactive frame rates. Reproduce with `bun run bench`.

```
                              [WebGL2 Browser Game /game]
                                           │
                  ┌────────────────────────┴────────────────────────┐
                  ▼                                                 ▼
    [GPU WebGL2 Raymarching Pipeline]                     [TypeScript Math Twin]
      (demoShader.ts / WebGL2 Canvas)                    (src/lib/terrainMath.ts)
                  │                                                 │
   - Chunk-Local fp32 Evaluation                     - Global Double Precision fp64
   - Lipschitz-Corrected Adaptive Stepping           - Strict MurmurHash3 uint32
   - Dyadic Frequencies (n / 2^20)                   - Zero Seam Parity Guarantee
   - OKLab 5-Biome PBR Shading                       - Ground Clamp + Jump Momentum
   - Soft Shadows with Inflated Footprint            - Slope Sliding Gate (||∇E|| > 1.5)
   - Dynamic Resolution Scaling                      - Player Controller Loop (§31)
                  │                                                 │
                  ▼                                                 ▼
         [Raymarched View]                                  [PlayerController]
     (DRS-scaled resolution)                              (WASD, Sprint, Jump,
                                                          Flight Mode, Mouse Look,
                                                          Respawn, Dynamic Settings)
```

## Feature Inventory
| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | Pure Browser WebGL2 Engine | Fullscreen procedural terrain raymarching at `/game` with zero desktop/engine dependencies | DONE |
| 2 | Strict Procedural Math Engine $E(x, z)$ | Exact $E(x, z)$ with dyadic frequencies ($n/2^{20}$), 9 macro + 5 ridged octaves, Voronoi valleys, crescent dunes, Zonai anomalies | DONE |
| 3 | TypeScript CPU Twin | Double-precision `terrainMath.ts` implementing MurmurHash3, C² quintic noise, Worley edge, and Lipschitz stepper | DONE |
| 4 | Lipschitz-Corrected Adaptive Raymarching | `demoShader.ts` GLSL ES 3.00 shader raymarching continuous $E(x, z)$ without geometry meshes | DONE |
| 5 | OKLab Biome Shading & Atmosphere | 5 OKLab biome RBF anchors, sun directional lighting, soft shadows, sky atmosphere, and water plane | DONE |
| 6 | Toroidal Floating Origin | Double-precision chunk origin rebasing at $8192\text{ m}$ ($2^{13}$) with dyadic frequencies preventing vertex jitter | DONE |
| 7 | First-Person Locomotion & Flight Controller | 60 Hz jump-preserving ground clamp, slope sliding ($>56.3^\circ$), WASD movement, sprint boost, fly mode toggle | DONE |
| 8 | In-Browser Engine Settings & Controls | Dynamic FOV adjustment (60°-90°), mouse look sensitivity, flight speed multiplier (1x-3x), seed randomizer, respawn hotkey (R) | DONE |
| 9 | Automated Test Suite | `tests/terrainMath.test.ts` validating MurmurHash3, dyadic frequencies, quintic interpolation, noise bounds, $E(x, z)$, and physics | DONE |
| 10| Launcher Script | `play.command` starts local server if needed and directly opens `http://localhost:3000/game` in the browser | DONE |
| 11| Third-Person Hero | `/game` starts in third person: a procedural green-tunic hero (no mesh, no texture) with run / sprint / sneak / jump / paraglider / climb / swim locomotion and a stamina wheel; **V** switches to first person and back, **F** to flight — see [`docs/HERO.md`](docs/HERO.md) | DONE |

## Code Layout
- `src/app/game/page.tsx`: Fullscreen browser game page, controls HUD, settings modal
- `src/lib/engine/useTerrainEngine.ts`: React binding for the engine (lazy-loads the engine core chunk)
- `src/lib/engine/engineCore.ts`: WebGL2 engine core — frame loop, camera modes (hero / walk / fly), uniforms, DRS
- `src/lib/hero/`: third-person hero — locomotion (`locomotion.ts`), procedural animation rig (`rig.ts`), orbit camera (`camera.ts`)
- `src/components/terrain/demoShader.ts`: WebGL2 / GLSL ES 3.00 procedural terrain raymarching shader
- `src/lib/terrainMath.ts`: TypeScript procedural mathematics, $E(x, z)$, climate, raymarcher, and player physics
- `src/lib/constants.ts`: Shared mathematical constants, dyadic frequencies, player physics parameters, quality presets
- `src/components/terrain/TerrainDemo.tsx`: Interactive WebGL2 canvas for the spec console
- `src/components/terrain/HillshadePanel.tsx`: 2D relief mapper driven by the CPU math engine
- `src/components/terrain/WorkedExample.tsx`: Worked numerical example table for §30 Lipschitz raymarching
- `src/components/FileBrowser.tsx`: Code browser serving engine source files via `/api/files`
- `src/app/api/files/route.ts`: Secure source file server for browser engine files in `src/`
- `play.command`: Double-clickable macOS launcher for opening the browser game
- `tests/`: automated test suites runnable via `bun test` (`terrainMath`, `mathCore`, `floatingOrigin`, `drs`, `statsStore`, `hero`)
