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
| 8 | In-Browser Engine Settings & Controls | Dynamic FOV adjustment (60°-90°), mouse look sensitivity, flight speed multiplier (1x-3x), seed randomizer, respawn hotkey (R), rebindable keys for every control (Controls page, saved in the browser) and Key Look, turning the camera with keys (see [`docs/HERO.md`](docs/HERO.md)) | DONE |
| 9 | Automated Test Suite | `tests/terrainMath.test.ts` validating MurmurHash3, dyadic frequencies, quintic interpolation, noise bounds, $E(x, z)$, and physics | DONE |
| 10| Launcher Script | `play.command` starts local server if needed and directly opens `http://localhost:3000/game` in the browser | DONE |
| 11| Third-Person Hero | `/game` starts in third person: a procedural green-tunic hero (no mesh, no texture) with run / sprint / sneak / jump / paraglider / climb / swim locomotion and a stamina wheel; **V** switches to first person and back, **F** to flight — see [`docs/HERO.md`](docs/HERO.md) | DONE |
| 12| Online Multiplayer | Up to five players share one world: a lobby before the game (a name and one of five heroes — the same hero in a green, red, blue, violet or amber dress, one per player), heroes played back smoothly from 15 Hz snapshots with their own shadows, the world's seed and time of day kept in step. `server.mjs` serves the game and the relay on one port; `play.command` (macOS) / `play.bat` (Windows) host it, with a public https link for friends far away when `cloudflared` is installed. Others are drawn by the WebGPU renderer (Chrome / Edge on Windows and macOS) | DONE |
| 13| Combat (after *Tears of the Kingdom*) | Sword combos, a charged spin, jump / dash attacks, sneakstrikes; the shield's guard, Perfect Guard and lock-on; side hops and backflips into a Flurry Rush; the bow with bullet time; throwing weapons; shield surfing; Bokoblin camps with senses, turns, archers and drops; hearts, wearing weapons, a combat HUD, procedural sounds; in multiplayer, players fight each other with all of it (each browser settles what strikes its own hero) — see [`docs/COMBAT.md`](docs/COMBAT.md). Foes and effects are drawn by the WebGPU renderer | DONE |
| 14| Inventory (after *Breath of the Wild*) | A pause screen on **I**: tabs for the weapons, bows, shields and food carried, a grid and a details panel, to equip, put away, drop and eat; shields and bows are carried and wear out like weapons (a Wooden Shield, a Pot Lid, a Soldier's Bow…, each drawn on the hero), food is dropped by foes and eaten for hearts, and alone in the world everything stands still behind it — see [`docs/INVENTORY.md`](docs/INVENTORY.md) | DONE |
| 15| Ground, rocks and a natural land (after *Breath of the Wild*) | The terrain has textures, painted on the GPU at start-up (no image files): turf, soil, baked clay, sand, gravel, three kinds of rock and moss, laid on the ground by its biome and slope, the rock seen from the side it faces; roads, footpaths, bare patches and drifts of long grass among short lawns, grass that grows in tufts; eight kinds of rock from pebbles to landmark boulders scattered by the GPU (the boulders stop the hero and the arrows); and a land with no ruler-straight cliff: karst towers that are fluted and irregular, crags and ledges on every steep face, mesas, valley walls and region edges (the snow line too) that bay and meander, a shape of its own for every lake and pond — see [`docs/GROUND.md`](docs/GROUND.md). The textures, rocks and tufts are drawn by the WebGPU renderer (about 0.4 ms, +10 %, of a frame in play and more in the densest places: the Cost section of GROUND.md); the land's shape is in all three twins | DONE |
| 16| Map and HUD (after *Tears of the Kingdom*) | A round minimap that turns with the view, a temperature gauge (the world's climate: falls with height, warmest at 15:00) and a sound meter (how loud the hero is), the time of day, flat coral hearts and a green stamina wheel; **M** opens a full north-up map drawn from the world's own maths (biomes, relief, contours, water, snow) with a grid, pan and zoom; other players show on both, with their names, and a pointer to the edge towards those out of reach — see [`docs/MAP.md`](docs/MAP.md) | DONE |
| 17| The cloud world (inactive) | A second world of solid cloud (a rolling cumulus sea, drawn cloud lobes, banks of cloud under the islands, puff ladders up them, springy puffs) under the same sky islands, parked for later work: no ring leads there from the land; Engine Settings → World switches to it and back (fade, health, pouch, clock kept) for inspection. WebGPU only | PARKED |

## Code Layout
- `src/app/game/page.tsx`: Fullscreen browser game page, controls HUD, settings modal
- `src/lib/engine/useTerrainEngine.ts`: React binding for the engine (lazy-loads the engine core chunk)
- `src/lib/engine/engineCore.ts`: WebGL2 engine core — frame loop, camera modes (hero / walk / fly), uniforms, DRS
- `src/lib/engine/keybinds.ts` / `src/lib/engine/keybindStore.ts`: the rebindable controls — actions, default keys, binding rules, Key Look's presets; their saving in the browser and React hook (edited in `src/app/game/SettingsModal.tsx`); `src/lib/engine/keyLook.ts`: the camera turned by the look keys
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
- `src/lib/world.ts`: the two worlds (the land, the cloud world) as seeds; `src/lib/portal.ts`: the ring between them; `src/lib/skyworld/clouds.ts` / `puffs.ts`: the cloud world's ground and its stepping clouds (design: `docs/superpowers/specs/2026-10-22-cloud-world-portal-design.md`)
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
- `src/lib/combat/`: combat — the world (`world.ts`: the hero's fight, foes, projectiles, pickups, camps, pace), the swings (`moves.ts`), weapons (`weapons.ts`), Bokoblins and camps (`foes.ts`), arrows and throws (`projectiles.ts`), other players as rivals and the messages their fights trade (`pvp.ts`), what the renderer draws of it (`scene.ts`), sounds (`sfx.ts`), the HUD's channel (`hud.ts`)
- `src/lib/gpu/combat.wgsl.ts` / `src/lib/gpu/combatGpu.ts`: the Bokoblins' distance bodies, the parts and the sprites (WebGPU); `src/app/game/CombatHud.tsx`: hearts, equipment, crosshair, prompts, banners, markers
- `src/lib/combat/pouch.ts` / `src/app/game/InventoryScreen.tsx` / `src/app/game/ItemIcon.tsx`: the inventory — what the pouch lists and the commands it takes (equip, put away, drop, eat), the pause screen opened with I, the items' icons (shared with the combat HUD); the shields, bows and food themselves are `SHIELDS`, `BOWS` and `FOODS` in `combat/weapons.ts` (see [`docs/INVENTORY.md`](docs/INVENTORY.md))
- `src/lib/map/` / `src/app/game/MapHud.tsx` / `src/app/game/MapScreen.tsx` / `src/app/game/mapDraw.ts`: the map — the engine's channel (`mapHud.ts`), the temperature and noise gauges (`gauges.ts`), the tile painter and its workers (`tiles.ts`, `tilesWorker.ts`, `tileService.ts`), the corner of the screen (minimap, gauges, clock) and the full map opened with M (see [`docs/MAP.md`](docs/MAP.md))
- `src/lib/gpu/groundTex.wgsl.ts` / `src/lib/gpu/groundTex.ts` / `src/lib/gpu/groundDetail.wgsl.ts` / `src/lib/gpu/groundCover.wgsl.ts`: the ground's nine textures (painted on the GPU), how a pixel of ground is made of them, and the roads, bare patches and long grass baked into the clipmap (WebGPU; see [`docs/GROUND.md`](docs/GROUND.md))
- `src/lib/rocks/` / `src/lib/gpu/rocks.wgsl.ts` / `src/lib/gpu/rocks.ts`: the rocks — meshes (`models.ts`), tiers (`tiers.ts`), the scatter and draws, and `collision.ts`, the CPU twin of the boulders that `forest/collision.ts` hands to the hero
- `tests/`: automated test suites runnable via `bun test` (`terrainMath`, `hydro`, `mathCore`, `floatingOrigin`, `drs`, `statsStore`, `hero`, `forest`, `combat`, `inventory`, `landforms`, `rocks`, `ground`, …)
