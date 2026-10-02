# Experiment ledger — an infinite TotK-style world at 60 fps

The programme and its gates are in the approved plan (E0–E8). Every number here comes from
`bench/webgpu.ts` on the development machine, recorded as it was measured.

**Setup.** MacBook Air M1 (7-core GPU), headless Chrome 154, WebGPU on Dawn/Metal, WebGL2 on
ANGLE/Metal, run natively (never through `/usr/local/bin/timeout`, which forces Rosetta).
Internal resolution 1280×800, march budget 128 steps, 28 shadow steps (the High preset's
budgets). Frames: the frozen benchmark camera path (`bench/baseline/camera_path.json`),
10 frames (0, 6, 12, 18, 24, 30, 35, 41, 47, 53) unless noted, reps interleaved per frame.

**How cost is measured.** Dawn samples a render pass's *begin* timestamp at the start of its
vertex stage, which on the M1's tile-based GPU overlaps the previous pass's fragment work;
begin→end durations over-reported by up to 2× when passes were queued back to back (first
attempt: 283 ms p50 for work that costs 241 ms). Cost is therefore the gap between
consecutive render-pass **end** timestamps inside a batch that keeps the GPU busy
(`bench/gpu/webgpu-page.ts`): per sample, three identical render passes of one variant and
frame, cost = median gap. Clipmap generation cost is the frame time it adds.

**How fidelity is measured.** OKLab ΔE per pixel against the converged ground truth (the WebGL2
`truth` transform: tiny horizon clamp, 3,000-step march, 600-step shadows), same frames and
resolution. Reported: share of pixels above one JND (ΔE > 0.02) and clearly visible (> 0.05).

## Results

| Renderer | GPU ms p50 | p95 | vs truth: > JND | > 0.05 | mean ΔE |
| :--- | ---: | ---: | ---: | ---: | ---: |
| WebGL2 (production shader) | 249.7 | 281.4 | 1.59 % | 0.55 % | 0.0015 |
| E1 WGSL direct (same math per pixel) | 241.1 | 272.9 | 1.59 % | 0.55 % | 0.0015 |
| E2 cached ray-march (`clip512`) | 9.23 | 9.77 | 2.06 % | 0.70 % | 0.0021 |
| E3 raster clipmap (`raster128`) | **3.25** | **3.79** | 2.49 % | 0.69 % | 0.0020 |
| E3 raster clipmap, half density (`raster64x2`) | 2.45 | 2.77 | 3.26 % | 1.17 % | 0.0029 |

Paired ratios (geometric mean over frames, bootstrap 95 % CI):
WebGL2 / WGSL direct 1.027 [1.006, 1.048]; cached march / direct 0.039 [0.037, 0.042];
raster128 / cached march 0.340 [0.308, 0.371].

### E0 — WebGPU harness: passed

- Headless Chrome 154 exposes WebGPU on Metal without flags; `timestamp-query` and
  `float32-filterable` are available (adapter "Apple M1", metal-3). The harness runs with
  `--enable-webgpu-developer-features` for unquantized timestamps.
- `bench/gpu/runner.ts` serves any page (`pageJs`) and Chrome flags; `bench/gpu/webgpu-page.ts`
  is bundled with `Bun.build` and speaks the same /config, /blob, /done protocol as page.js.

### E1 — WGSL world function and clipmap generator: passed

- `src/lib/gpu/worldgen.wgsl.ts` ports E(x, z) and its helpers line for line; constants are
  interpolated from `src/lib/constants.ts`.
- Frames (`bun bench/webgpu.ts port`): WGSL direct vs the WebGL2 renderer, 10 frames at
  1280×800 — **99.94 % bit-identical pixels**, no pixel above JND.
- Elevation probes (the `bench/parity.ts` grid, 0–1,000 km, chunk seams): |E_WGSL − E_GLSL|
  ≤ 0.061 mm everywhere; both show the same error against the fp64 CPU twin (0.28 mm at the
  spawn, 113 mm at 1,000 km — the fp32-global precision recorded in `PLAYER.groundBias`).
- Generator (`bun bench/webgpu.ts gen`): 391 M lattice points/s. Full rebuild of 8 levels of
  512² (2.1 M points) adds **5.4 ms** to its frame (gate ≤ 50 ms); 7 levels of 1024²: 19 ms.
- Incremental updates while moving (60 frames at 60 Hz, 5 start points), generator pass p50:
  walk 5.4 m/s 0.033 ms, sprint 8.6 m/s 0.035 ms, fly 40 m/s 0.049 ms, 200 m/s 0.116 ms
  (gate ≤ 0.5 ms); frame time moving vs still differs by ≤ 0.2 ms at p50.

### E2 — cached ray-march: 26× faster at the same look

- Same march, bisection, shading and soft shadows as the WebGL2 shader, with every E query
  replaced by `heightAt()`: the finest clipmap level holding the point, bilinear, faded into
  the next level over the outer quarter (log2 distance) of its range.
- 241 → 9.23 ms p50 at 1280×800; fidelity 2.06 % of pixels above JND vs 1.59 % for direct
  evaluation (0.60 % above JND against the direct frames). Prefiltering the cache for twice
  the lattice spacing (`clip512-dp2`) changes nothing measurable.
- Where the 9.2 ms goes: soft shadows ~1.0 ms (9.24 → 8.28 ms without); the rest is the
  march (up to 128 samples, each a level choice and one or two bilinear fetches) and shading.

### E3 — raster geometry clipmap: chosen

- `src/lib/gpu/rasterRenderer.ts`: 8 rings of (2k)² cells over the same cache (ring r at
  0.5 m · 2^r · stride), ring 0 full, rings ≥ 1 with the finer ring's square cut out by one of
  four index buffers; odd vertices morph onto the coarser lattice near each ring's edge
  (CDLOD) with positions in integer lattice units, so rings meet vertex for vertex.
  Reversed-Z depth; water is one quad at the water level; sky and clouds fill the far plane.
- **3.25 ms p50, 3.79 ms p95** for the whole frame (terrain, water, sky, clouds, soft shadows),
  0.34× the cached march.
- First version shaded at the triangle surface: shadow rays started below the smoother
  cached surface in concave spots and self-shadowed (jagged dark streaks on far mountains,
  2.96 % above JND). Shading at the cached height under the mesh point fixed it (2.49 %).
- Remaining difference: sawtooth silhouettes where sharp crests (dune cusps, ridged-noise
  ridges) run diagonally across the triangulation; the per-pixel bilinear surface of the
  march has none. Candidates: centre-vertex cell split, a 0.25 m ring 0, or a smooth-abs crest
  in the world function (the math may change). Later addressed by choosing each cell's
  diagonal from the heights (see "Ridge crests" below).
- **Decision:** raster. It leaves ~10 ms of the 14 ms budget for vegetation, lighting, TAA
  and the hero, where the march would leave ~4.

### Hero on WebGPU

- `src/lib/gpu/hero.wgsl.ts` ports the hero SDF (body, trace, shadow, contact darkening,
  shading) from demoShader.ts. In the raster renderer it is a depth-writing pass limited to
  the scissor rectangle of the hero's bounding sphere; terrain and water take its shadow and
  contact darkening through two hooks in the shared `shadeSurface()`.
- `bun bench/webgpu.ts hero` (the hero framed on the camera path, 1280×720, 10 frames):
  1.75 % of pixels above JND against the WebGL2 hero program (the remaining difference is
  the terrain: raster vs march). 3.52 ms p50 / 4.50 ms p95 per frame with the hero.
- The WebGL2 frames show bright one-pixel fringes along ridge silhouettes where the march
  steps over the crest; the raster frames have none.

### E8-lite — the game on WebGPU: 60 fps

`/game?renderer=webgpu` draws with the raster renderer (`src/lib/gpu/engineGpu.ts`, its own
chunk); without the parameter, or without WebGPU, the WebGL2 path runs unchanged. The engine
loop (`engineCore.ts`) now draws through a small View interface with the two backends.

Production build, headless Chrome, 1280×720 CSS pixels at DPR 2, 15 s scripted play
(`bench/play.ts` with `renderer: 'webgpu'`; frames counted per `GPUQueue.submit`):

| Renderer, view, preset | Internal resolution | Frames/s | Dynamic resolution |
| :--- | :--- | ---: | :--- |
| WebGPU, third person, Medium | 1280×720 | **60.0** | never reduced |
| WebGPU, third person, High | 1920×1080 | **60.0** | never reduced |
| WebGPU, first person, Medium | 1280×720 | 60.0 | never reduced |
| WebGL2, third person, Medium | 925×520 → 448×252 | 20.8 | reduced 5 times |
| WebGL2, third person, High | 1632×918 → 672×378 | 8.8 | reduced 6 times |

JS allocation in third person: 1.2 MB/s on WebGPU vs 0.96 MB/s on WebGL2 (both dominated by
the same page-level functions). Found on the way: a preset change only resized the canvas
at the next dynamic-resolution step, which never comes at 60 fps; the loop now resizes on
the change.

### E4a — cached sun visibility

- The clipmap gains a second layer array: each lattice point's sun visibility, a 48-step
  soft-shadow march over the cache run once when the point is generated
  (`sunShader`, right after the heights). Shading samples it (`sunAt`) instead of marching
  per pixel; the hero keeps its per-pixel march (few pixels, contact matters).
- Third-person gallery (20 ground-level views, 1280×720, grass + MSAA): **6.55 → 3.80 ms p50,
  8.65 → 6.07 ms p95**. Shadows stay soft and in place (cliffs, hillsides, the hero's shadow).

### E5 — vegetation: grass and trees

- **Grass** (`grass.wgsl.ts`): a compute pass scatters one blade candidate per cell of
  world-anchored lattices around the camera — 5 rings of 256² cells, 0.125 m doubling per
  ring, so blades per screen area stay roughly constant out to ~250 m; rings cross-fade by
  growing / shrinking blades. Candidates off screen, under water, on steep ground or where
  the biome grows little (clumped by a 22 m patch field) are dropped; survivors are appended
  for one indirect draw of 7-vertex strips bent by lean, travelling wind gusts and the hero
  (blades part around the feet). Ground under and beyond the blades takes the grass's
  average colour so a field reads as one surface. 35–52 k blades per ground-level view.
- **Trees** (first version, since replaced by the forest below; `trees.wgsl.ts`, `treeMesh.ts`
  are gone): the same scatter on 6 m cells (4 rings, out to
  ~2.3 km): forests from moisture, temperature and a 700 m woods mask, lone trees in
  meadows, none on rock, sand, snow, steep ground or water. One unit mesh per detail level
  (6-sided trunk + icosphere canopy), shaped per tree in the vertex shader: lumpy broadleaf
  dome or tiered conifer, size, facing, wind sway. Outer rings draw fewer, wider canopy
  clusters so distant forests stay dense. ~0.9–1.1 k near and 1.9–3.3 k far trees per view.
- **Tree shadows** without a shadow map: the tree scatter also publishes ring 0's canopies,
  one sphere per 6 m cell, and every surface tests its sun ray against the 16 cells upwind.
- **4× MSAA** for thin blades: +1.1 ms p50 at 1280×720.
- Grass costs almost nothing net: blade fragments replace the terrain shading they cover.

### E7 — sky islands (replaced: see "The sky realm" below)

- `src/lib/skyIslands.ts` (CPU twin) and `src/lib/gpu/skyIslands.wgsl.ts`: at most one island
  per 2,048 m cell (32 % of cells), 110–320 m radius, top at 430–820 m: grassy relief over a
  jagged outline, then a cliff wall and an inverted mountain of layered rock with drip-stone
  spikes. Top and underside meet exactly at the rim (tested), so each island is closed.
- Drawn as a shared polar mesh instanced per island surface; the 16 nearest islands are
  re-picked every 200 m. Islands shadow everything below them analytically; grass and tree
  groves grow on their tops.
- The hero stands and runs on island tops (`sampleGround` prefers the top when above it) and
  falls — or glides — off the rim; the camera keeps its clearance over islands (tested).
- Height fog is now integrated along the view ray from the camera, so a view from an island
  is not fogged as if it were at sea level.

### The complete frame

Clean GPU (the user's Cartograph app held the GPU at 98–100 % for a while and inflated one
round of measurements 5–10×; those were discarded and re-run). Terrain + grass + trees +
sky islands + 4× MSAA + hero:

| Views | 1280×800 p50 | p95 | 1920×1080 p50 | p95 |
| :--- | ---: | ---: | ---: | ---: |
| 20 ground-level third-person views (bare raster terrain) | 3.03 | 4.75 | 4.43 | 7.01 |
| the same views, complete world | **7.15** | **9.70** | 9.24 | 13.10 |
| 9 sky-island views, complete world | 4.35 | 5.43 | | |

In game (production build, headless Chrome, 1280×720 CSS pixels at DPR 2, 20 s scripted play):

| Renderer, view, preset | Internal resolution | Frames/s | Dynamic resolution |
| :--- | :--- | ---: | :--- |
| WebGPU complete world, third person, Medium | 1280×720 | **60.0** | never reduced |
| WebGPU complete world, third person, High | 1920×1080 | **60.0** | never reduced |
| WebGPU complete world, first person, High | 1920×1080 | 60.0 | never reduced |
| WebGL2, third person, Medium | 1088×612 → 448×252 | 21.3 | reduced 6 times |

### E4b — first grade (for review)

The WebGPU renderer's output goes through a stylized grade before gamma: warm split toning
in the light, cooler lifted shadows, a filmic shoulder (ACES fit) and +7 % saturation; its sky
is a deeper blue overhead. The march variants keep the WebGL2 output transform, so the parity
measurements above still compare like with like. Cost: within measurement noise
(7.07 ms p50 / 9.40 ms p95 on the gallery after the grade, 7.15 / 9.70 before).

### In the game

`/game` now picks WebGPU when the browser has it (`renderer: 'auto'` in useTerrainEngine);
`?renderer=webgl2` forces the ray-march renderer and `?renderer=webgpu` the new one. The
landing page's demo stays on WebGL2. The header badge and the HUD show which renderer runs.
`bench/play.ts` defaults to `?renderer=webgl2`, so its sessions stay comparable with
revisions that predate WebGPU; `renderer: 'auto'` opens `/game` bare (verified: WebGPU, 60 fps
at 1920×1080 on High).

### E6 — temporal anti-aliasing and upscaling

- `src/lib/gpu/taa.ts`, `taa.wgsl.ts`: the scene renders at the internal resolution with a
  Halton (2, 3) sub-pixel jitter over 16 frames. The resolve rebuilds each pixel centre from
  the depth buffer, reprojects it into last frame's view (camera motion only), samples the
  history there (Catmull-Rom), clips it to the current 3 × 3 neighbourhood in YCoCg
  (mean ± 1.25 σ) and blends 10 % of the new frame in. A present pass scales to the canvas
  (Catmull-Rom) with contrast-adaptive sharpening. History resets on a resize or a jump
  of more than 40 m.
- Bug found on the way: reprojecting the *jittered* sample position re-read the history half
  a pixel off every frame, a random-walk blur. With the pixel centre, the soft variant went
  from 8.78 % to 2.06 % of pixels above JND.
- `bun bench/webgpu.ts aa`, 1280×800, 5 ground-level gallery views scored against a 2 × 2
  supersampled render (4× MSAA at twice the size, box-filtered); temporal stability from a
  slow pure yaw with time frozen (share of pixels whose luma jumps by more than 24 / 255
  between frames). The GPU was at 94 % from another app when this run started, so treat
  the costs as indicative:

| Variant | mean ΔE | > JND | pan flicker | GPU p50 | p95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| no AA | 0.0059 | 8.65 % | 1.683 % | 6.78 | 9.02 |
| 4× MSAA | 0.0025 | 3.39 % | 0.828 % | 8.23 | 11.17 |
| TAA, sharpened | 0.0074 | 12.40 % | 0.420 % | 7.89 | 9.46 |
| **TAA, soft** | **0.0026** | **2.06 %** | **0.162 %** | 7.89 | 10.72 |
| TAA upscaling from 0.67 × | 0.0099 | 16.65 % | 0.168 % | 6.12 | 7.24 |

- **Decision:** TAA replaces 4× MSAA in the game. It matches MSAA's still-image fidelity,
  flickers 5× less in motion, costs less, and upscales: the canvas stays at the display
  resolution and the preset × dynamic-resolution scale only sets the internal size.

### Water

- The water quad is blended (premultiplied alpha) over the lake bed, which is drawn as
  terrain: per-channel Beer-Lambert absorption by depth, ripple normals from three
  scrolling noise scales, a sky-gradient reflection with Fresnel and a small sharp sun
  glint, and shoreline foam bands. The bed takes a sand tint and caustics in the shallows.
- Swimming views: 6.71 ms p50 / 8.35 ms p95 at 1280×720 (complete world).

### E7 — waterfalls, ruins and shrines

- **Waterfalls** (replaced by the sky realm's, below) (`waterfalls.wgsl.ts`, `islandWaterfall` in `skyIslands.ts`): 60 % of sky
  islands pour a 12–24 m wide fall from a hashed rim point to the ground or water below: a
  camera-facing ribbon that widens as it drops, with scrolling streaks, and a mist
  billboard at the foot. Views of falls: 5.39 ms p50 / 5.76 ms p95 at 1280×720.
- **Ruins and shrines** (`src/lib/ruins.ts`, `ruins.wgsl.ts`): one candidate site per 1,024 m
  cell (30 % kept), on walkable ground above the water or on a sky island's top. A site is a
  stone platform bedded into the slope, pillars round its edge (a third broken), lintels over
  neighbouring intact pillars, steps, a broken back wall and, for 30 % of sites, a shrine: a
  dark monolith cut by pulsing teal lines (broken concentric squares). The boxes are built
  once per site on the CPU and shared by the renderer (instanced cubes pulled from a storage
  buffer, one draw) and the hero's collision (push-out from walls, stand on tops).
  Stone shading: coursed blocks with mortar joints, flagstones on top faces, per-block tints,
  moss in the joints, near the ground and under the top edges. Grass and trees stay out of
  the sites. 9 ruin views (approach, above, on the platform): 5.40 ms p50 / 7.46 ms p95
  at 1280×800.

### Ridge crests: triangulation chosen from the heights

- Each ring cell is now split along the diagonal whose midpoint lies nearer the true height
  at the cell's centre, read from the next finer clipmap level (ring 0, which has no finer
  level, keeps the higher midpoint). A compute pass (`ringTris`) writes one index buffer per
  ring whenever that ring moves. A cell that collapses into a coarser cell as its ring
  morphs takes the coarser ring's choice, so the morph still ends exactly on the coarser
  ring's triangles.
- 10 path frames: 0.01–0.24 % of pixels change above JND, all on silhouettes. There are no
  cracks, and diagonal crests follow triangle edges instead of notching.
- Cost at 1280×800 (terrain only): 1.68 → **1.47 ms p50** (paired ratio 0.919
  [0.867, 0.975]); the new triangles are also cheaper to draw. Re-triangulating all 8 rings
  every frame (the worst case, e.g. a teleport) leaves p50 unchanged and raises p95 by
  0.5 ms. In play, one ring moves every few frames.

### Far field: a 1024² cache (rejected)

- With 1024² levels each level reaches twice as far, so every point is shaded one level finer.
  On 7 path vistas it changes at most 1.3 % of pixels above JND (mean ΔE ≤ 0.0016). The
  differences are the level-blend seams moving outward, not new detail. Distant detail is
  limited by the world function's prefiltering for each level's footprint, and by the fog,
  not by the cache. It would cost 4× the memory (64 MB) and about 4× the full rebuild (E1:
  19 ms for 7 levels). The cache stays at 512².

### Collision in first person

- `updatePlayer` takes an optional ground over the terrain. The game passes
  `firstPersonGround` (`hero/locomotion.ts`): ruin walls push the player out, and box tops
  and sky-island tops hold them up, as for the hero (tested: stand on a platform, stopped
  by a pillar, land on an island). The ruin-site cache no longer allocates on a hit.

### Headed Chrome on the real display

Production build, `bench/play.ts` with `headed: true`: a visible Chrome 154 window on the
MacBook's display (DPR 2, vsync), canvas 2560×1440, High preset (1920×1080 internal,
upscaled by TAA), the complete world with ruins and adaptive diagonals:

| Session | Frames/s | Dynamic resolution |
| :--- | ---: | :--- |
| Third person, 20 s, headed | **60.0** | never reduced |
| First person, 15 s, headed | **60.0** | never reduced |
| Third person, 15 s, headless (same build) | 60.0 | never reduced |

### The forest: ten species by climate (replaces E5's trees)

The first trees were one icosphere on a stick, bent per tree in the vertex shader: the same
lollipop everywhere, blobs on sticks from a distance, spheres for shadows. All of it
(`trees.wgsl.ts`, `treeMesh.ts`, the canopy-sphere shadows, their buffers and bindings) was
removed and rebuilt:

- **Models** (`src/lib/forest/models.ts`, CPU, at start-up): per species three shapes, each
  grown as a skeleton (trunk and branch paths, leaf clumps, conifer tiers, palm fronds, cactus
  arms) and emitted at three levels of detail from that one skeleton, so a switch never changes
  the outline. Tapered tubes with parallel-transport frames for trunks and limbs (buttresses,
  root flare and cactus ribs shape them), lumpy displaced clumps whose normals lean outward
  from the crown so a crown shades as one soft volume, jagged drooping skirts for conifers,
  V-folded serrated fronds for palms. Per vertex: part, wind sway, baked occlusion and a
  per-clump colour variation. All 90 meshes: 37 k triangles, built in ~60 ms. An oak is
  1.9 k / 430 / 48 triangles at its three levels, a pine 328 / 135 / 53.
- **Where trees grow** (`src/lib/forest/placement.ts`, redesigned from scratch after the first
  version put ~50 trees/ha in every meadow and 86/ha in woods, trees in every direction).
  Open land is the default; trees come in features, each placed for how it reads in a view:
  - *Forests*: bodies of a few hundred metres with lobed edges (a domain-warped field over
    700, 260 and 110 m). They grow where the hills suit them: flanks and hollows (relief and
    slope over 84 m, so hummocks do not punch holes), damp ground, never crests or cliffs.
    Each has a fringe of smaller trees and bushes, glades inside, and one species per ~200 m
    stand.
  - *Groves*: clumps of 16–36 m in open land, most in hollows.
  - *Lone trees*: at most one per coarse cell (48 m in mild land), so they never crowd;
    big and open-grown, a little likelier on rises, where they stand against the sky.
  - *Shores*: a broken line just above the water (birches north, palms south, oases in
    dry land).
  - *Bushes*: sparse in the open, thicker along forest edges.
  - Groves and lone trees swing between open plain and parkland over ~2 km, so a journey
    alternates between the two; the noise offset puts the default spawn in parkland (a lone
    oak on the hill by the hero, groves on the hills round it).

  How much of each, and which species, comes from eight climate zones blended across
  their borders: temperate (oak, birch stands), autumn woods (maple), warm lowland (oak with
  jungle trees, palms by the water), jungle, savanna (acacias, no forests), desert (cactus
  patches), mountain (pine) and alpine (snowy pines and snags). The numbers live in one
  table; the placement shader (`forestGen`) is generated from it and mirrors the CPU twin
  line for line. Read back from the GPU, the two agree on 99.998–100 % of the 147,456 cells
  of a window, and lone trees stand at the same spot to the centimetre
  (`bun bench/webgpu.ts parity`). Survey of 18 × 18 km round the spawn (CPU twin):

| Zone | Share of land | Forest cover | Trees/ha | of which in the open (groves, lone) | Bushes/ha |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Temperate | 39 % | 22 % | 19.5 | 3.5 | 5.9 |
| Autumn woods | 21 % | 28 % | 23.1 | 2.8 | 6.1 |
| Warm lowland | 9 % | 19 % | 13.1 | 3.4 | 5.3 |
| Jungle | 4 % | 51 % | 48.2 | 1.7 | 6.8 |
| Savanna | 15 % | — | 1.6 | 1.5 | 4.1 |
| Desert | 3 % | — | 2.0 | 2.0 | 1.0 |
| Mountain | 8 % | 33 % | 27.1 | 1.2 | 3.6 |
| Alpine | 2 % | 9 % | 7.1 | 0.5 | 1.8 |

  Tests (CPU twin): forest cover in mild land between 12 % and 30 %, none on savanna or
  desert; forest bodies coherent (normalised neighbour coherence > 0.45 at 50 m); each
  zone's leading species; tree counts per hectare in range; lone trees never closer than
  0.29 of their grid; nothing in water or on cliffs; ruins bare; island tops planted.
  Judged by eye on 32 random views (24 at eye level, 8 from the air) and the spawn: most
  eye-level views hold a lone tree, a grove or a forest edge with open grass round them.
- **Collision** (`src/lib/forest/collision.ts`, used by the hero and first-person walking):
  the trees of the 3 × 3 lattice cells round the body, from the placement's CPU twin, with
  the shape and facing the GPU picks from the same cell hash and the scale it stores. Each
  trunk is a circle round its centre at hip height, wide enough for the drawn trunk from knee
  to shoulder (it follows the detailed mesh's rings, so buttress fins and a palm's lean are
  covered), standing on the ground or an island's top. The body is pushed out and its
  velocity into the trunk removed, so it slides round; crowns and bushes do not block, and
  above a trunk's top nothing does. Cells are cached (nothing is allocated on a hit): a hero
  tick costs 5.4 µs through a forest, 7.9 µs while entering new cells. Tests: every collider
  encloses its drawn trunk at knee, hip and shoulder; the hero running at a lone oak near the
  spawn stops at the bark and slides past it; first-person walking is stopped the same way
  and passes freely 5 m above a trunk's top; bushes never block.
- **Placement cache** (`forestGen`): a 6 m lattice in tiles of 8 × 8 cells, a window of 48 × 48
  tiles (±1.15 km) kept toroidally (`src/lib/forest/tiles.ts`). A cell is evaluated once, when
  its tile enters the window; nothing is re-evaluated per frame.
- **Culling** (`forestCull`, every frame, 147 k slots): distance, a stable 45 % thinning beyond
  420 m (the rest grow a little), the view frustum, and the level by distance to the crown,
  larger trees switching farther out. Trees within ±6 % of a switch are cross-faded by a
  complementary dither, in buckets of their own. One bucket per (species, shape, level) is one
  indexed indirect draw: 90 draws for the trees, 60 for their shadows.
- **Shadows**: a 2048² sun shadow map ±90 m around the camera. Casters within 32 m use the
  level-1 shape, farther ones the level-2 shape. Terrain, grass, water, ruins and the hero
  take one filtered tap; trees take it per vertex. A full canopy passes a fifth of the sun.

**Cost.** Three things cost far more than their size, each isolated with paired runs:

1. Sampling the shadow map in the same frame that renders it stalled the GPU for ~3 ms,
   whatever the map's size (1024² or 2048²) and whether a comparison sampler or a plain
   load was used. Sampling a texture not written that frame was free. The two maps are now
   double-buffered: each frame renders one and samples the other, finished the frame
   before, with that frame's box.
2. `discard` for the cross-fade in the one tree pipeline turned off the M1's hidden-surface
   removal for every tree (~1.3 ms). Fading trees now have their own buckets and pipeline
   (a WGSL `override`); the rest draw opaque.
3. Five filtered taps in the terrain's fragment shader cost ~2 ms; one costs ~0.3 ms.

Result, 1280×800, the game's world (`world-taa`) against the same world without trees
(`world-taa-bare`), paired by view, after the placement redesign:

| Views | With the forest, p50 | p95 | vs no trees (95 % CI) |
| :--- | ---: | ---: | :--- |
| 20 gallery views (the first trees: 7.93 / 9.94 ms) | 6.65 ms | 11.65 ms | × 1.03 [0.92, 1.15] |
| 32 random views (24 eye level, 8 aerial) | 5.37 ms | 7.58 ms | × 1.25 [1.11, 1.39] |

The forest's draw calls take their dynamic offsets from one preallocated array (a new array
per draw had cost ~0.4 MB/s of JS allocation, 150 draws a frame); in play the page now
allocates 1.3–1.5 MB/s, all of it in code that predates the forest.

The forest costs about a millisecond. It now draws a few dozen near and 350–2,500 far
trees per view in open country (a forest view: a few hundred near), and trees hide terrain
that is dearer to shade than they are. In game (production build, High, 2560×1440 canvas): 60.0 fps in a visible Chrome
window (third and first person) and headless, dynamic resolution never reduced.

### The day–night sky (the classic sky kept as an option)

The game now draws time. A day lasts 24 real minutes by default (12 or 48 in the settings);
the sun and the moon move, the sky is computed from the air, and the light on the land
follows both. The classic fixed sky stays one click away (Settings → Sky → Classic) and draws
exactly as before: every classic frame of the gallery sets is identical to the pixel to the
frame before this work (15 views, largest channel difference 0).

**What makes each hour scenic in this world.** The world is long sight lines over hills,
peaks to 1.3 km, sky islands, lakes and forests, under a hazy, stylised air. The cycle is
designed round what light does to land like that:

- *Relief from a moving low sun.* The sun keeps a real path: 30° latitude on an equinox that
  never ends, rising due east at 06:00, 60° high toward −z at noon, setting due west at 18:00.
  Mornings and evenings rake the relief, and mountain shadows sweep across the valleys as it
  moves.
- *Colour from the air, not from a palette.* Light reaches each point through the air toward
  that point's own horizon, so a low sun turns it gold, then orange, then red. Midday light is
  a near-neutral white, and the golden hour lasts while the sun is below ~15°.
- *The last light on high places.* The land's horizon dips with altitude over a small planet
  (R = 300 km). After a valley loses the sun, peaks and sky islands keep a red-gold glow for
  ~20 game minutes (alpenglow), and clouds at 1.5 km keep it longest.
- *A luminous twilight.* The sky is Earth's atmosphere (Rayleigh, Mie and ozone, after
  Hillaire 2020), so the orange horizon under the sun and the pink Belt of Venus over Earth's
  blue shadow opposite it appear on their own. Ozone keeps the blue hour blue.
- *A night worth looking up at.* Stars turn about a pole 30° above the north: about 7 % of
  sky cells hold one, twice as many along the Milky Way, and ~1 % are brilliant, with a
  twinkle near the horizon. The moon is a textured sphere lit by the sun, with maria and a
  large storybook disc; its 8-day month starts on a waxing gibbous rising in the east at the
  first sunset. Shooting stars cross now and then.
- *Weather that suits the hour.* Fair-weather clouds build through the afternoon so the sunset
  has something to light (their undersides glow pink and orange). They dissolve after dusk, so
  nights are clear for the stars and mornings fresh. Mist settles over the low land before
  dawn and burns off by ten.
- *Time where it is beautiful.* Time lingers through golden hour, sunset and blue hour
  (×0.55) and hurries through the deep night (×1.7). Of a 24-minute day, ~7 minutes are
  twilight, 6 night and 11 day.
- *Night that stays readable.* Moonlight is ~1/900 of sunlight, darker than the blue hour as
  it should be. The camera adapts partly to the light on the land (displayed brightness ∝
  light^0.22): a moonlit night shows at about a quarter of day and a moonless one at about a
  fifth, bluer and less saturated, as the eye's night vision is.

**How it is drawn.**

- **Shadows for any hour, baked once.** The sun and the moon run on the same great circle, so
  a direction on it is one angle. For every clipmap texel, a march in the path's tilted plane
  (out to 8 km, the land falling away over the small planet) finds the angle where light
  first clears the land (rising) and where it sinks behind it (setting). Two numbers per texel
  (rgba16float) serve the sun or the moon at any hour with no re-bake, and the shadows sweep
  continuously. Accuracy against a brute-force march of the CPU twin in the same plane
  (`bun bench/webgpu.ts horizons`: steps of 1 % of the distance; 3 cameras, 160 texels each):

| Clipmap level | Median error | 95th percentile | Within 1° (= 4 game minutes) |
| :--- | ---: | ---: | ---: |
| 0 (0.5 m) | 0.05–0.11° | 0.5–1.6° | 84–99.7 % |
| 3 (4 m) | 0.09–0.13° | 1.3–1.9° | 90–94 % |
| 6 (32 m) | 0.29–0.33° | 3.5–5.1° | 79–81 % |

  The first bake stepped 9 % of the distance and read heights through the level blend: at
  level 0 only 69 % of the peak camera's texels were within 1°, almost all of them *lower*
  than the truth (shadows cut short). Halving the step fixed that, and reading one level per
  step halved the fetches without losing accuracy. What remains at the coarse levels is
  mostly how heights are stored: a 32 m lattice read bilinearly shaves the crests that the
  exact CPU twin sees.
- **The sky.** Transmittance and multiple-scattering LUTs are computed once. Each frame a
  192 × 108 sky-view LUT (azimuth × elevation, dense at the horizon) is integrated for the
  sun and the moon together from the camera's altitude, and one probe pass averages it into
  the fill light (toward and away from the key light, overhead, off the ground), the
  horizon's colour as a cosine series round the key light's azimuth (the haze's colour), and
  the exposure.
- **The light rig** (`src/lib/sky/lightRig.ts`, CPU, 1 µs a frame, allocation-free). The key
  light is the sun until it is 7.5° down (no altitude sees it then) and the moon from 8° down,
  fading in over 3°, so the switch happens in the dark. Its colour is tabulated at 8 altitudes
  packed toward the ground, where the dip changes fastest. The rig also sets saturation,
  night vision, haze, mist, the clouds' rhythm and the stars.
- **One light API for every surface** (`sky.wgsl.ts`: `keyDir`, `keyCol(y)`, `skyAmb(n, w)`,
  `bounceCol`, `glow`…). Terrain, grass, trees, ruins, islands, waterfalls, water and the hero
  all call it. The classic implementation returns the classic constants, which is why the
  classic sky's frames are unchanged. The day–night one keeps each shader's tuned balance of
  sun and fill (their weights `w`) and changes the light's colour and level.
- **Art direction on top of the physics.** A colour-keeping soft clip keeps a sunset's pink on
  the clouds instead of burning to white at a dusk exposure. At twilight, when the camera
  exposes for lit peaks, the dome and haze (not the land's light) are lifted toward a luminous
  pastel. Ozone absorbs red at 0.64 of green, as measured (Hillaire's table has 0.35, which
  turns the blue hour violet).
- **In the game.** Settings → Sky offers *Day & night* or *Classic*. Switching builds the other
  renderer in the background while the current one keeps drawing (4.4 s for the first build
  of a sky in a dev session; ~0.25 s once Chrome has cached its pipelines). Also in Settings:
  the time of day (slider, with the phase: Dawn, Sunrise, Morning glow, Morning, Midday,
  Afternoon, Golden hour, Sunset, Dusk, Night), a pause and the day length. Holding **T** runs
  time an hour a second. A clock sits beside the engine badge. `?time=18:30` and `?sky=classic`
  start the page there. WebGL2 keeps the classic sky.

**Cost** (1280×800, day–night against the classic sky on the same views, paired, 95 % CI):

| Views | Hour | Day–night / classic |
| :--- | :--- | :--- |
| 20 gallery views | 15:30 | × 1.01 [0.93, 1.10] |
| 12 scenic views | 12:30 | × 1.03 [0.93, 1.13] |
| 12 scenic views | 18:09 | × 0.86 [0.79, 0.93] (fewer surfaces face the low sun) |
| 12 scenic views | 21:00 | × 1.25 [1.10, 1.45] (stars and the Milky Way, ~0.2 ms; the classic sky pass is nearly free) |

Two things cost far more than their size, as before:

1. The sky-view LUT and the probe, written by a compute pass and read by the render passes in
   the same frame: × 1.46 [1.37, 1.54] on the gallery. Double-buffered (each frame computes
   one set and samples the other, a frame old): × 1.10.
2. The terrain and grass fragment shaders sit at an occupancy cliff. Stubbing out any one of
   the haze colour, the fill, the key-light lookup, the horizon read or the grade removed
   most of the remaining cost, and those savings summed to more than twice the total. Taking
   per-frame values off per-pixel work fixed it: the haze's forward glow now uses the key
   light at the camera's altitude from the CPU instead of interpolating it per pixel, the
   probe results are read as a uniform instead of a storage buffer, and the horizon colour is
   a cosine series instead of a 16-entry ring indexed through `atan2`.

Clipmap generation (the frame time it adds):

| | Classic sun bake | Path horizons |
| :--- | ---: | ---: |
| Full rebuild (spawn, respawn, sky switch) | 16 ms | 67 ms |
| Walking and sprinting (5–9 m/s) | +0.0 ms | +0.0–0.1 ms |
| Flying (40 m/s) | +0.3 ms | +1.0 ms |
| Fast flight (200 m/s) | +0.6 ms | +2.0 ms |

In game (production build, High, 2560×1440 canvas, headed Chrome, third person): **60.0 fps**
at 15:30, 60.0 at 18:06, 59.9 at 21:30, 60.0 with the classic sky. Dynamic resolution was
never reduced. Tests: 16 more (`tests/sky.test.ts`) cover the sun's and the moon's paths, the
pacing of the clock, the air's transmittance, the reddening and alpenglow of the key light,
the sun-to-moon switch in the dark, the night's level and the morning mist.

### Tree surfaces: leaf cards, painted bark, needles and leaflets

Every tree surface was one flat colour broken by two octaves of value noise: crowns read as
faceted blobs beside the grass, trunks as plain tubes. The aim, from reference pictures of
Breath of the Wild's trees, was crowns that are masses of single leaves with leafy, broken
outlines and sky through the gaps, and pale, mossy, painterly trunks spreading into roots.
Nothing is a texture file: the leaves are painted at start-up and the rest is shader code,
fixed to the model so it sways with the tree.

- **Leaf sprays** (`src/lib/forest/leafAtlas.ts`): one 256² layer per leaf style, painted
  once at start-up in ~0.15 s, with its mipmaps:
  - long pointed leaves (oak), five-lobed leaves (maple), small toothed ovals (birch), broad
    leaves with heavy midribs (jungle), compound leaves of tiny leaflets (acacia) and small
    round leaves (bushes);
  - each layer is a spray of 13–52 leaves over a disc, pointing outward, deeper ones darker,
    each with its own tone, a darker base and midrib, and its own tilt;
  - texels hold coverage, tone and tilt, all premultiplied by coverage, so the mipmaps
    average correctly.
- **Leaf cards** (`models.ts`, `forest.wgsl.ts`): the near and middle levels of every
  broadleaf crown carry cards spread evenly over each clump. That is ~470 on a near oak, ~130
  at the middle level, ~45 on a bush. Cards buried deep inside another clump are dropped;
  those near the seams between clumps stay and cover them.
  - Each card is four vertices at one centre. The vertex shader turns the card to face the
    camera, then sizes and turns it per card.
  - Each leaf is lit by the crown's normal (out from its middle, so the crown shades as one
    soft volume), bent the way that leaf faces. Light comes through the leaves when backlit.
  - The crown under the cards is the foliage further in: the same sprays projected onto it
    three ways, darker and in its own shade, with deep gaps between the leaves.
  - Cut-outs are dithered against the spray's coverage, so TAA blends the edges. Far off,
    where the mipmaps blur a spray, it thins to its mean coverage instead of shrinking.
  - The middle level draws 28 % of the cards at 1.75 times the size, and they shrink away
    over the far half of that level, where one spans a few pixels. Cards also give way
    within 1.6 m of the camera, so a camera inside the leaves does not fill with them.
- **Painted bark** for oak, maple, acacia and bush stems:
  - long soft streaks down the trunk in three octaves, shallow furrows wandering along it,
    and large light and dark mottling;
  - moss in patches, most on whatever faces up and round the foot;
  - a pale mossy olive base on oaks.
  It replaced the first version's Voronoi plates, which were too hard and too dark next to
  the references.
- **Roots:** oak and maple trunks flare at the foot into four or five roots spreading over
  the ground. Near trunks have 12 sides (from 8) and close rings at the base, so the roots
  show. The trunk colliders follow the roots automatically.
- **Other species keep their own surfaces:**
  - pine: scaly plates;
  - birch: lenticels, dark scars and a black cracked foot;
  - palm: ring scars and fibres;
  - jungle: pale bark with lichen and rings;
  - dead trees: twisting grain and cracks;
  - cactus: grooves and areoles;
  - pine needle tufts: domes on a lattice drawn out down each skirt, with needle bundles
    close up;
  - palm fronds: leaflets with a pale midrib.
- **Relief** bends the normal by each pattern's analytic slopes along the surface
  coordinates, times the screen-space derivatives of those coordinates (Mikkelsen's surface
  gradient). The new coordinates are the turn round each tube, with a seam vertex closing
  every ring, and the metres along it. Screen-space differences of the pattern itself, the
  first attempt, step in 2 × 2 pixel blocks that TAA does not remove.
- **How it got here:** the first crowns were shading only, leaf clusters as small domes.
  They gave textured surfaces but smooth lump outlines and no gaps, unlike the references.
  The cards replace them; conifers keep the domes as needle tufts.
- **Fixed on the way:**
  - A pine skirt's underside shared the rim's up-facing normals and was lit like its top. It
    now faces down.
  - Foliage seen from inside a crown stays plain and shaded.

**Drawing cut-out cards cheaply on the M1.** `discard` turns off hidden-surface removal, and
cut-out cards overlap, so every layer of leaves was shaded and so was whatever lay behind
them. Drawing the cards after every opaque draw cost × 1.16 on the forest views. What brought
it down, measured at each step:
1. **Fewer, larger middle-level cards:** 28 % of them at 1.75 times the size, shrinking away
   over the far half of the level. Only near cards look up the tree shadow map, once a
   vertex. The aerial view over the autumn woods went from +3.7 ms to +0.5 ms.
2. **Two passes:** the cards' depth first, cut out and cheap; then their colour where that
   depth is equal, opaque and without `discard`, shaded once per pixel. Both passes place
   corners with the same function, and the position is `@invariant`, so their depths agree
   to the bit. Forest views: × 1.09.
3. **Depth first in the pass:** the cards' depth pass now runs before the terrain, so the
   terrain, grass, crowns and ruins skip whatever lies behind leaves. Forest views: × 1.07.
   The seam fix then added cards and brought it back to × 1.09.

**Cost.** The comparison variant is `world-cycle-plain`, the same world with plain trees
(`forest.detail: false`: no cards, no surface detail). All numbers are at 1920×1080, the
game's internal resolution on High.

The order of variants biases every paired comparison: whichever variant renders a view
first, and to a lesser degree second, pays for arriving there while the GPU's clocks ramp.
`--warmup=1` runs a copy of the first variant ahead of the rest, but that absorbs only part
of the effect. Each number below is therefore the geometric mean of both orders (each run in
brackets), plus the per-view change taken as each variant's fastest time over both runs.

| Views | Surfaces vs plain | Median view | Worst views |
| :--- | ---: | ---: | :--- |
| 20 gallery views (the game's typical views) | × 1.00 (0.99, 1.02) | ±0.0 ms | 9.8 → 11.4 ms |
| 50 close-ups (bark, crowns, under the canopy) | × 1.02 (1.00, 1.05) | +0.03 ms | under a maple 7.0 → 9.9 ms |
| 20 forest views (inside woods and from the air) | × 1.09 (1.09, 1.08) | +0.15 ms | autumn woods 6.2 → 10.0 ms, jungle shore 7.3 → 9.3 ms |

**Temporal stability** (`bun bench/webgpu.ts shimmer`: 6 forest views, 24 frames held still,
then a 32-frame slow pan with time frozen, 1280×800):
- flicker, the share of pixels changing by more than 24 of 255 between frames, is 0.22–0.95 %
  with the cards against 0.27–0.95 % plain, so the dithered cut-outs add none;
- the mean change per frame rose by 0.16–0.36 of 255, because the trees have more contrast
  to move.

**In game** (production build, High, 2560×1440 canvas, headed Chrome): 59.9 fps in third
person, 59.9 in first person, 60.0 in third person at 17:54. Dynamic resolution was never
reduced.

Tests, in `tests/forest.test.ts`:
- surface coordinates run round each trunk (the seam vertex repeats the first) and up it in
  metres, and frond vertices carry their side and their length;
- broadleaf crowns carry cards at the near levels only, each four corners at one centre;
- every leaf spray covers 20–60 % of its layer, is premultiplied, keeps the tile's edge
  empty, and its last mipmap is its mean coverage.

### The sky realm: archipelagos, sky biomes and water with a source (replaces E7's islands)

What was wrong with E7's islands, as reported in play:
- a waterfall poured from a bare point on the rim, with no lake or stream above it;
- it fell to whatever lay below, often a dry mountainside;
- island tops were the lowland's meadow and trees;
- islands were small (110–320 m) and low (tops at 430–820 m), so some floated inside the
  1.3 km peaks;
- they never connected.

The new realm lives in `src/lib/skyworld/` (the CPU twin, physics and tests). On the GPU it is
`gpu/skyRealm.wgsl.ts` (shaders) and `gpu/skyRealm.ts` (host).

**Layout** (`archipelago.ts`, all by hash from cell and seed, cached):
- **Archipelagos:** one candidate per 8,192 m cell (60 % kept), whole inside its cell.
  - a hub of 0.65–1.25 km;
  - 3–6 satellites across 170–460 m of open sky, most level enough to be joined by an arch;
  - 2–6 small isles farther out, at their own heights;
  - 5–12 floating rocks.
- **Lone isles:** one candidate per 2,048 m cell (22 %), never inside an archipelago.
- **Height:** every underside's tip stays above 1,400 m, so above the highest peak. Tops lie
  at 1.55–2.9 km, above the 1.5 km cloud deck.
- **Around the spawn (seed 1337):** 9 archipelagos and 14 lone isles within 12 km. The
  nearest stands 3.9 km straight ahead of the spawn's view (hash salts chosen for that).
  Building them on the CPU takes ~110 ms, once.

**Sky biomes** (`biomes.ts`). The ground's biomes follow latitude, altitude and rain. Up here
other things decide:
- the cloud deck: an island whose underside hangs into its mist drinks it;
- the thin cold air of height, and the wind on a small isle;
- the archipelago's trait (verdant, blossom, frost or wild), which runs through all its
  islands.

Five biomes, each with its own ground palette, grass (height, density, wild flowers), relief,
underside rock, water and trees:

| Biome | Where it grows | Look and trees | Water |
| :--- | :--- | :--- | :--- |
| Sky meadow | middling height and size | golden-green grass full of flowers; golden and blossoming trees | lakes |
| Mistwood | underside steeped in the cloud deck | moss-dark ground; giant teal oaks, tree ferns, silver birches | lakes in 95 % |
| Petal glade | blossom archipelagos | pink and pearl-white groves over lilac-flecked grass | lakes |
| Gale crags | small, high and exposed | tawny turf and bare crag; wind-stunted pines | rarely |
| Rime isles | the highest (above ~2.6 km) | snow; a summit; rime-frosted pines | frozen lakes, no stream |

The trees are the ground's tree shapes in eight sky palettes. Islands of one archipelago
share its trait, then split by their own height and size. Around the spawn, every biome
occurs (tested).

**Water with a source** (`hydrology.ts`). The cloud deck wets the islands, most of all those
hanging into it. A wet enough island keeps a lake in its lowest hollow, and the lake drains
by one stream to a notch in the rim:
- **Lake:** the best of a dozen candidate hollows well inside the rim. Its level sits 0.35 m
  under the lowest point of its shore (two rings of 48 samples), and a low levee covers what
  lies between the samples. It never spills: 360 bearings × 4 distances are tested.
- **Stream:** 16 meandering points from the lake's outlet to the lip. Its bed:
  - starts 0.6 m under the lake's level and never rises;
  - falls at least 0.4 %;
  - stays 0.8 m under the land across its whole corridor (±1.6 half-widths), not just its axis;
  - notches 1.2 m into the rim.

  The first version kept the bed under the axis only, and a stream along a slope ran above the
  land beside it; the test caught it.
- **Lip:** the best of 36 bearings round the rim. It keeps clear of the arches' footings and
  of any island beneath. It prefers shallow cuts and a big drop, and rejects any ravine deeper
  than three half-widths plus 4 m.
- **Fall:** water that drops hundreds of metres frays into spray and then mist, as the
  world's tallest falls do. Each fall has all turned to mist by its reach: at most half the
  drop to the land below, and at most 220 m + 40 × its width. Its mist trails on below,
  drifting downwind, with a faint rainbow when the sun is behind the viewer. **No fall reaches
  the ground**, so none lands on a dry mountain (tested), and every fall has a stream and a
  lake above it (tested).
- **Closed shell:** where the stream notches the rim, the underside follows it down, so top
  and underside still meet at the lip (tested).

**Arches** (`arches.ts`) are natural rock bridges from inside one island's rim to inside
another's:
- the deck is never steeper than 0.3 and is crowned with grass;
- it is 10–18 m wide mid-span and 2–2.6 times that where it roots into the islands;
- its rock is about 1/30 of the span thick in the middle and 1/12 at the ends, with a ragged
  keel of hanging spurs;
- it never crosses another island;
- `kind: 'arch'` leaves room for built bridges.

**Physics** (`realm.ts`):
- The hero stands on tops and arch decks, swims in sky lakes, wades in streams and falls off
  rims. Swimming needed a water level that depends on position: `HeroState.water`, the sea or
  a sky lake or stream.
- The camera keeps clear of sky water.
- First person stands on tops and decks.
- Tests: the hero runs to a hub's lake and swims, crosses an arch to a satellite without
  falling, and falls 300 m+ off a rim.
- `?spawn=sky` starts the game on the nearest hub, beside an arch's footing.
- Fly mode's ceiling was 1,150 m, under the WebGL2 ray-march's early-exit height. It hid the
  realm behind an invisible ceiling from the ground. On WebGPU it is now 3,500 m
  (`FLY_CEILING`, tested against the highest top). Checked in the production build: from the
  ground spawn, fly up (about 30 s), over the hub, F, and the hero lands on its top.

**Rendering:**
- **The realm buffer** is re-picked every 250 m, nearest archipelagos first. Floating rocks
  are uploaded only within 6 km. It holds up to 256 islands, 4,096 stream points and 96 arches.
- **Polar meshes** draw undersides always, and tops unless the sky rings hold all of them.
  Rings are bunched toward the rim (its cliff), at three levels of detail (8×32, 24×96,
  64×192) chosen by on-screen size.
- **Sky rings** draw the tops near a camera that is up among the islands: a geometry clipmap of
  six rings of 128² cells, 1 m cells first, out to ±2 km. Each vertex takes the top of its
  island and is cut along the outline by a clip distance (the signed distance inside it;
  `discard` if the device lacks the feature).
  - A compute pass caches each ring's heights, normals and clip distances. It remakes a ring
    only when that ring moves or the realm changes. Evaluated per vertex every frame instead,
    the rings cost × 1.39 on island views; cached, × 1.00.
- **Water:** lakes as a blended disc at the water line (frozen ones as opaque, cracked ice),
  and streams as a ribbon down the channel with foam streaks, white water in the rapids and at
  the lip.
- **Shadows:** the undersides and arches seen from the sun go into a 2,048² shadow map
  (±9 km, double-buffered like the trees'), replacing the analytic disc shadows.
- **The cloud deck veils what lies across it:** islands seen from the land sit partly behind
  the same clouds as the sky round them, and from an island the land lies under a sea of
  clouds.
- **Fog** looking up takes the colour of the sky behind it rather than the horizon's haze;
  islands had read as pale ghosts.
- **Surfaces:**
  - beyond the forest's drawing range an island's top is painted with its canopy;
  - undersides show strata, vertical weathering and a wet streak under the fall's lip;
  - a soft bounce from the cloud deck lights undersides above it.
- **Grass** grows on tops and arch decks in the biome's colours, height and flowers. It grows
  where the top is nearer the camera than the land below, never on bare crag or under snow.
- **The forest's second layer:** a second layer of placement slots holds the sky's trees, so
  the land under an island keeps its own trees. Before, every cell held one plant, island or
  land, and the ground under each island was bare.
  - The sky layer's GPU placement matches its CPU twin in 99.999–100 % of 147,456 cells per
    view (`bun bench/webgpu.ts parity --config=realm`).

**Bugs found on the way:**
- **NaN at the origin.** WGSL `atan2(0, 0)` is NaN on Metal, and Metal's `min` returns the
  other operand. A polar mesh's centre vertex got NaN relief, and the lake shore's steep curve
  replaced it: a 240 m spire, at coarse levels of detail only. The fix is an `atan2` that is 0
  at the origin.
- **Garbage streams.** When the stream points ran out of room, islands whose streams were
  left out were still drawn as streams, reading other islands' points.

**Cost** (1920×1080; each ratio the geometric mean of both variant orders, each order in
brackets; this session ran warm, so absolute times were up to 1.5× the earlier sessions'):

| Views | Realm vs the same world without it | Notes |
| :--- | ---: | :--- |
| 20 gallery views on the land (the game's typical views) | × 1.00 (0.95, 1.05) | no measurable cost |
| 15 views on and among the islands | × 1.13 (1.09, 1.17) | p50 8.2–8.6 ms; against empty sky, so the islands' own trees, grass and water |
| the sky rings, on those views (one order) | × 1.00 | × 1.39 before their cache |

In game (production build, High, 2560×1440 canvas): 60.0 fps in third and first person, on
the land and on the hub (`?spawn=sky`). Dynamic resolution was never reduced. JavaScript
allocation: +0.16 MB/s for the realm (2.06 against 1.91 MB/s without it); its per-frame
layout allocates nothing.

**Tests:**
- `tests/skyworld.test.ts`: 20 tests covering layout, closed shells, water that never spills
  or climbs, falls with sources that turn to mist above the land, walkable arches, biome
  logic, sky flora, and the hero on tops, in lakes and on arches.
- `tests/skyRealm.test.ts`: 6 tests covering the realm buffer against the WGSL structs, the
  clip-distance and discard variants, the packing round the spawn, and the meshes.

**Room for a sky civilization:**
- Each archipelago is a graph of islands joined by arches, with a trait.
- Each island carries its biome, its water and its flat dry ground, which `nearWater`, the
  slope and ρ already find.
- Ruins already take sites on island tops, clear of the water.
- Settlements, built bridges (a new `Arch` kind), and roads along the graph can hook in there
  without touching the terrain.

### Performance pass: the default preset past 120 fps, the picture unchanged

**How it was measured.**
- **In game.** Chrome on macOS holds `requestAnimationFrame` to the display's 60 Hz, even with
  vsync switched off. `bench/play.ts` `uncapped` therefore paces the engine by the GPU (two
  frames in flight).
  - Two production builds were compared: the code before this pass and after it.
  - Sessions alternated between the builds: 3 rounds × 8 s each.
  - Hero mode, 1440×790 CSS pixels at DPR 2, a 2880×1580 canvas.
- **Bench.** `bench/webgpu.ts stills --timing` ran the game's own variant (`game-med`,
  `game-high`: `GAME_VARIANT` at the presets' internal scale) at 2880×1580.
  - Every ratio is paired and taken in both orders, with 4–6 reps.
  - `--noshots=1` times without saving images.
  - `profile.ts` stamps every pass.
  - The `taa` mode times the TAA passes alone.
  - The scenes are the gallery, the `run` path (third person), the `realm` views and the `hub`
    (the hero on the sky hub, as `?spawn=sky`; `framed-hub.json`).
- **Picture.** The 20 gallery views at 2880×1580, converged (8 frames), compared with the same
  views rendered by the code before this pass.

**In game** (fps; the means of 3 alternated rounds, rounds in brackets):

| Preset | Scene | Before | After | Ratio |
| :--- | :--- | ---: | ---: | ---: |
| Medium (default) | land | 104 (108, 105, 100) | **145** (148, 149, 138) | × 1.39 |
| Medium | sky hub | 106 (104, 108, 106) | **147** (152, 146, 142) | × 1.39 |
| High | land | 67 (72, 66, 65) | **96** (102, 94, 93) | × 1.43 |
| High | sky hub | 82 (87, 80, 79) | **103** (101, 105, 103) | × 1.26 |

The display is 60 Hz, so the game shows 60 fps. These figures are what the GPU sustains: the
headroom, and what a faster display would show.

**What changed, and what each bought** (bench ratios of the whole frame, both orders):
- **The ground's colour from the clipmap.** Moisture, the vegetation clumps and the grass's two
  patch fields were six value noises per pixel per frame. They are now baked once per lattice
  point into a material layer, written by the sun/horizon pass, and sampled with the height and
  the sun layer in one level lookup (`groundCacheAt`).
  - Height and slope, which turn the colour sharply, stay per pixel, so near ground looks as it
    did.
  - Frame × 0.895 at Medium.
  - A first version cached the albedo itself. It was faster (× 0.86) but softened a cliff
    top's crisp edge, and was dropped.
- **The normal beyond 60–120 m from the sun layer's gradient.** This is a LOD improvement.
  Central differences over the lattice follow the land's smooth contours, where the cached
  surface's facets drew distant ridge crests, valley creases and snow edges as stair-steps.
  Near the camera the faceted normal is kept, as before. On the glide views these are the only
  differences above a JND.
- **Lookup tables as `const`, not `var<private>`.** Tint copied each private table into every
  invocation: about 50 tables of species, sky biomes and placement zones. The frame got
  × 0.927 at High (`run`).
- **TAA in half precision.** The colour arithmetic runs in f16; positions and the reprojection
  stay f32. TAA alone went from 2.09 to 1.64 ms at High (resolve 1.18 → 0.81, present
  0.90 → 0.82).
  - The scene target is now 10-bit (`rgb10a2unorm`, display-encoded values), half the bytes of
    16-bit floats; the history stays f16.
  - Fog's colour, the tone map and the gamma are in f16 too (about × 0.98).
- **Terrain rings culled by block.** Each ring's index buffer is laid out in 16 × 16-cell
  blocks, ordered round the ring by angle (`ringBlockOrder`). Only the blocks whose boxes meet
  the view are drawn, one or two runs a ring. The images are bit-identical; about 40 % of the
  terrain's 820 k triangles are left out.
- **The sky realm's shadow map and the sky LUT are made only when they would change.**
  - The shadow map is redrawn when the sun turns 0.05° or the camera moves 500 m. A day of
    12 real minutes then asks for one every 0.1 s.
  - A 64 × 64 caster mask, made with each map, lets every surface skip the realm's four shadow
    taps where nothing is in reach. They had run for every pixel within 9 km.
  - The sky LUT and its probe are remade when what they read moves past a tolerance.
- **Grass.** Rings 3 and 4 (63 m and more away) bend in two segments, not three (a LOD of the
  geometry; within a third of a pixel).
  - The sky's blades are cached per lattice cell (`bladeCache`, tagged with their cell). The
    scatter on the hub went from 1.68 to 0.32 ms, though it overlapped other passes, so the
    frame did not change.
  - One walk along an island's stream now serves the water, the top and the damp, where it
    took up to five.
- **The hero.**
  - It is traced only inside the box round its parts, and its scissor is that box on screen.
  - The belt and the sword, sheath and shield are skipped like the limbs where they cannot be
    nearest.
  - The trace returns the material it hit.
  - Contact darkening returns early 3 m away.
- **Work the picture cannot show is skipped.**
  - Behind opaque fog (optical depth > 5.82) the ground, the water and the islands' surfaces
    return the fog without being shaded.
  - The fog stops reading the sky dome where it does not blend it in (looking down).
  - Water skips its ripple noise past 300 m and its foam noise in deep water.

**The picture** (20 gallery views against the code before this pass):
- Mean ΔE 0.0022; 0.47 % of pixels above a JND on average, 1.28 % in the worst view.
- The differences are the smoother far ridges, the far grass's blade shapes, and rounding.
- Close-ups of the hero, the ground and the trees are unchanged.

**Dead ends** (measured, reverted):
- **TAA as compute kernels.** Workgroup tiles, 12 point loads where there were 9 bilinear taps.
  The image was identical but cost 3.08 against 1.50 ms. Fragment passes write tile memory;
  compute writes go out to memory.
- **The present sampling a 10-bit copy of the history.** No change (0.90 ms either way): it is
  not filter-bound.
- **Drawing the hero first.** No change on the `run` views. Kept, because whatever lies behind
  it is then never shaded.

**What the measurements taught:**
- **Per-pass "end minus previous end" gaps go wrong once compute is in the frame.** Compute
  overlaps the render passes, and the gaps made TAA look like 2.5–3 ms. Time a pass alone, and
  attribute the main pass by ablation (`STUB_*` constants, `skip`).
- **Compute that overlaps render passes hides its cost.** Making the grass scatter five times
  cheaper moved the frame by nothing.
- **The big fragment shaders are register-bound.** Stubbing fog out of every surface saved 9 %
  of the frame for a few dozen instructions, and the biome's remaining arithmetic 6.5 %.
  Fewer live values (f16, caching) pays more than fewer instructions.
- **Two variants cost very differently in one page.** With eight renderers at 2880×1580 the
  8 GB machine paged and every ratio was noise. Keep it to two or three renderers per session.

**Bench switches added** (the game never sets them): `stub` and `stubAll` (`STUB_*`,
`PER_PIXEL_GROUND`, `CACHED_GRAD`, `NO_BLADE_CACHE`), `skip` (main-pass draws), `privateTables`,
`taaF32`, `shadeF32`, `heroLate`, and `RealmSpec.stubGround`. Their pipelines fold the constants.

## Next

- High at 120 fps on this M1 Air: High is at 96–103 fps. What remains, each 4–12 % at High:
  - the hero (its trace could skip the limb groups its ray never comes near);
  - the terrain's per-pixel biome and shadows (register pressure: f16 through the lighting);
  - water, the tree shadow pass and the sky's clouds.
- A sky civilization on the archipelagos: settlements on the hubs' flat dry ground, built
  bridges as a new arch kind, roads along each archipelago's graph of islands and arches.
- A way up to the islands in play (updrafts or launch towers, as in Tears of the Kingdom);
  today it is fly mode, or `?spawn=sky`.
- Climbable island undersides and cliffs (the underside is not solid to the hero yet), and
  falls that feed a lower island's lake.
- Leafy shadows: the tree shadow map draws the crowns' cores, not their cards. Cut-out cards
  in the shadow pass would give dappled light under the trees; that pass has no colour to
  save, so its cost is the cards' depth alone, still to be measured.
- The far level's crowns are still smooth lumps beyond ~120 m, where the middle level's cards
  have shrunk away.
- Cloud shadows drifting over the land by day (a small texture of the cloud layer, once a frame).
- The horizon bake when flying fast (+2 ms at 200 m/s): bake coarse levels at half rate, or
  reuse the coarser level's far horizon with a height correction.
- Clouds with volume, and a bloom pass for the shrine lines, the sun and the moon.
- Erosion-like gullies and cliff bands in the world function (each needs its generation
  cost measured against the E1 budget).
- The hero rig update allocates about 0.25 MB/s. It predates this work but is the top JS
  allocation in third person.
