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
   - 15 Biome Regions (biomes.ts)                    - Ground Clamp + Jump Momentum
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
| 2 | Strict Procedural Math Engine $E(x, z)$ | Exact $E(x, z)$ with dyadic frequencies ($n/2^{20}$): fifteen biome regions from climate and geology, each with its own landform (karst towers, mesas, crescent dunes, volcano cones, marsh flats, ranges, coasts and cliffs), rivers, level lakes and waterfalls — see [`docs/LANDS.md`](docs/LANDS.md) | DONE |
| 3 | TypeScript CPU Twin | Double-precision `terrainMath.ts` implementing MurmurHash3, C² quintic noise, Worley edge, and Lipschitz stepper | DONE |
| 4 | Lipschitz-Corrected Adaptive Raymarching | `demoShader.ts` GLSL ES 3.00 shader raymarching continuous $E(x, z)$ without geometry meshes | DONE |
| 5 | Biome Shading & Atmosphere | Per-biome soil, grass, rock strata, sand, mud, snow with wind drifts, lava and ice; per-biome trees (17 flora zones) and air (humid haze, marsh fog, desert glare, volcanic ash); sun lighting, soft shadows, sky atmosphere; water that stands where the land holds it | DONE |
| 6 | Toroidal Floating Origin | Double-precision chunk origin rebasing at $8192\text{ m}$ ($2^{13}$) with dyadic frequencies preventing vertex jitter | DONE |
| 7 | First-Person Locomotion & Flight Controller | 60 Hz jump-preserving ground clamp, slope sliding ($>56.3^\circ$), WASD movement, sprint boost, fly mode toggle | DONE |
| 8 | In-Browser Engine Settings & Controls | Dynamic FOV adjustment (60°-90°), mouse look sensitivity, flight speed multiplier (1x-3x), seed randomizer, respawn hotkey (R) | DONE |
| 9 | Automated Test Suite | `tests/terrainMath.test.ts` validating MurmurHash3, dyadic frequencies, quintic interpolation, noise bounds, $E(x, z)$, and physics | DONE |
| 10| Launcher Script | `play.command` starts local server if needed and directly opens `http://localhost:3000/game` in the browser | DONE |
| 11| Third-Person Hero | `/game` starts in third person: a procedural green-tunic hero (no mesh, no texture) with run / sprint / sneak / jump / paraglider / climb / swim locomotion and a stamina wheel; **V** switches to first person and back, **F** to flight — see [`docs/HERO.md`](docs/HERO.md) | DONE |
| 12| Online Multiplayer | Up to five players share one world: a lobby before the game (a name and one of five heroes — the same hero in a green, red, blue, violet or amber dress, one per player), heroes played back smoothly from 15 Hz snapshots with their own shadows, the world's seed and time of day kept in step. `server.mjs` serves the game and the relay on one port; `play.command` (macOS) / `play.bat` (Windows) host it, with a public https link for friends far away when `cloudflared` is installed. Others are drawn by the WebGPU renderer (Chrome / Edge on Windows and macOS) | DONE |

## Code Layout
- `src/app/game/page.tsx`: Fullscreen browser game page, controls HUD, settings modal
- `src/lib/engine/useTerrainEngine.ts`: React binding for the engine (lazy-loads the engine core chunk)
- `src/lib/engine/engineCore.ts`: WebGL2 engine core — frame loop, camera modes (hero / walk / fly), uniforms, DRS
- `src/lib/hero/`: third-person hero — locomotion (`locomotion.ts`), procedural animation rig (`rig.ts`), orbit camera (`camera.ts`)
- `src/components/terrain/demoShader.ts`: WebGL2 / GLSL ES 3.00 procedural terrain raymarching shader
- `src/lib/terrainMath.ts`: TypeScript procedural mathematics, $E(x, z)$, the lands (regions, landforms, water), climate, raymarcher, and player physics
- `src/lib/hydro.ts`: the world's geography (ranges, massifs, tablelands, basins, volcanoes, coast) and its rivers and lakes — the drainage graph, river levels and profiles, valleys, and the node table the GPUs read
- `src/lib/erosion.ts`: the gully erosion filter (MPL-2.0, after Rune Skovbo Johansen) in TypeScript, WGSL and GLSL
- `src/lib/noise.ts`: the hash and value-noise primitives the world function is built from
- `src/lib/gpu/hydroBuffer.ts` / `src/lib/engine/hydroTexture.ts`: the river node table round the camera, for WebGPU (storage buffer) and WebGL2 (texture)
- `src/lib/biomes.ts`: the fifteen biomes — each one's landform, ground look and air (the tables every renderer is generated from)
- `src/components/terrain/landsGlsl.ts`: the lands' GLSL port for the WebGL2 renderer
- `src/lib/sky/air.ts`: the biome air round the camera, and the HUD's land name
- `src/lib/constants.ts`: Shared mathematical constants, dyadic frequencies, player physics parameters, quality presets
- `src/components/terrain/TerrainDemo.tsx`: Interactive WebGL2 canvas for the spec console
- `src/components/terrain/HillshadePanel.tsx`: 2D relief mapper driven by the CPU math engine
- `src/components/terrain/WorkedExample.tsx`: Worked numerical example table for §30 Lipschitz raymarching
- `src/components/FileBrowser.tsx`: Code browser serving engine source files via `/api/files`
- `src/app/api/files/route.ts`: Secure source file server for browser engine files in `src/`
- `play.command` / `play.bat`: double-click launchers (macOS / Windows) that host the multiplayer game server and open the game
- `server.mjs`: the game server — Next.js plus the multiplayer relay (WebSocket `/mp`) on one port; `--prod` builds when stale, `--tunnel` prints an https link for friends (cloudflared)
- `src/lib/net/`: multiplayer in the browser — the session and lobby state (`session.ts`), other players' heroes played back and packed for the renderer (`remoteHeroes.ts`)
- `src/lib/hero/characters.ts`: the five heroes a player can be (their dresses)
- `tests/`: automated test suites runnable via `bun test` (`terrainMath`, `hydro`, `mathCore`, `floatingOrigin`, `drs`, `statsStore`, `hero`, `forest`, …)
