# The ground: textures, cover, rocks, and a land that does not look cut out

The land used to be smooth colour: a biome's soil, grass and rock tints, a slope's threshold between
them, and nothing under the eye's reach but the grass blades. This is what was added, after the look
of *Breath of the Wild*, and what was changed in the land's shape so that nothing in it reads as
cut with a ruler. No image file was added anywhere: every texture is painted on the GPU at start-up,
every mesh is built in TypeScript.

| | Where | What |
|---|---|---|
| Ground textures | `gpu/groundTex.wgsl.ts`, `gpu/groundTex.ts` | nine tileable layers: turf, soil, baked clay, sand, gravel, three rocks, moss |
| Their use | `gpu/groundDetail.wgsl.ts`, `gpu/terrain.wgsl.ts` `shadeGround` | what a pixel of ground is made of, from its biome and slope |
| Ground cover | `gpu/groundCover.wgsl.ts`, `biomes.ts` `GROUND_USE` | roads, footpaths, bare patches, long and short grass |
| Grass | `gpu/grass.wgsl.ts` | tufts with bare ground between them, short lawns and tall drifts |
| Rocks | `rocks/`, `gpu/rocks.wgsl.ts`, `gpu/rocks.ts` | pebbles to landmarks, scattered on the GPU, drawn as instanced meshes |
| Boulders' collision | `rocks/collision.ts` | the CPU twin of the big tiers: the hero and arrows meet what is drawn |
| The land's shape | `terrainMath.ts`, `hydro.ts` and their shader twins | karst towers, crags, mesas, valleys, lakes, region edges |

The WebGL2 fallback gets the land's new shape (its GLSL twin is generated from the same tables) and
not the textures, the rocks or the new grass: it ray-marches the world per pixel and has no
clipmap layers to bake them into.

## Ground textures

`groundTex.ts` paints one 512² layer per material into a texture array at start-up (a pipeline a
layer, so the nine compile side by side; then a shade pass that turns the painted heights into slopes
from the exact floats, not 8-bit steps; then the mips). A texel holds:

| channel | |
|---|---|
| r, g | the surface's slope across the layer, as a tangent-space normal's x and y (½ = flat), from the layer's own height in metres over its tile |
| b | its tone, a multiplier on the biome's colour (½ = ×1, 1 = ×2) |
| a | its cavity: 1 on open ground, down toward 0 in cracks and between pebbles |

The biome's colour (the clipmap's soil, grass and rock layers) stays the albedo: a texture only
moves it, tilts the shading normal and darkens the crevices, so a badlands' rust and a mountain's
grey use the same rock. Every layer tiles (all its noise is periodic over the tile) at a world scale
that divides the chunk, so chunk-local coordinates agree across a chunk boundary.

| layer | tile | relief | made of |
|---|---|---|---|
| turf | 4 m | 7 cm | tufts (domes with dark gaps), a slow mottling, faint combed strokes, seeds and leaves |
| soil | 4 m | 10 cm | crumbs, grit, pebbles of two sizes (lighter, with a dark rim), dark clods |
| crack | 8 m | 16 cm | baked clay: plates of two sizes, some split again, lifted rims, shallow cracks |
| sand | 8 m | 3 cm | wind ripples with a steep lee in drifts, a second fainter set across them |
| gravel | 4 m | 16 cm | packed pebbles each of its own tone, small stones between the large |
| rockA | 16 m | 1.5 m | fractured stone: blocks (tilted planes, ledged against their neighbours), a finer set of facets, cracks open along half the borders, edges worn bright, weathering streaks |
| rockB | 16 m | 1.1 m | strata: bands of every thickness with wavy edges that pinch out, hard ones standing as ledges, slanting joints that taper |
| rockC | 16 m | 0.8 m | dark volcanic stone: slabs, vesicles, a faint grain |
| moss | 2 m | 18 cm | soft cushions of two sizes |

`groundDetail.wgsl.ts` decides what a pixel of ground is. `shadeGround` gives it the biome's shares
of grass, soil, rock and sand (`groundLook`), and `groundDetail` taps the layers by them:

- **turf** over grass (where the cover is high), at its own tile and again at five times the tile,
  turned 37° (the two tones multiply), so the repeat does not show; its height also decides where
  grass gives way to bare earth, so the edge of a lawn, a road or a bare patch is ragged and not a
  smooth gradient;
- **soil**: loam, or what the dry country, a bare slope or a shore make of it (baked clay where the
  soil is the dry country's, sand where a desert's cover gives out, scree on a bare slope, pebbles at
  a shore); a second scale, turned, only where the soil is a large share of the ground;
- **rock** on the steep and in outcrops, seen from the side it stands facing (a biplanar projection:
  the two best-facing axes of the three, the strata always level, so a wall is never smeared), the
  biome's kind (fractured stone, strata where it bands its rock, dark slabs where its rock is black),
  with the fine chips of the main kind at a fifth of the scale close up, the lookups bent by a slow
  wave (100 – 350 m long: the tile is 16 m) so a tile's repeat does not stand in straight rows on a
  wall; moss on its upper faces in lush country;
- the colour, the shading normal and the ambient occlusion are moved by the detail (`K_TONE`,
  `K_CAV`); it fades out from `DETAIL_NEAR` (50 m: the mips have long averaged it away) and is not
  read at all from `DETAIL_FAR` (160 m), so the far ground costs no taps. Under a lawn the blades
  hide the ground, so a lawn's detail fades out over 4 – 11 m. The sampler is trilinear with 2×
  anisotropy; every tap is a `textureSampleGrad` with the pixel's own footprint, so a material's taps sit in
  per-pixel branches (no uniform control flow is needed) and are made only where the material is.

`bun bench/ground.ts` paints the atlas in headless Chrome and writes, per layer, a PNG of its tone,
its normal map and the surface lit and tiled 2 × 2 (a seam or an obvious repeat shows); it prints
each layer's mean tone, slope and cavity (the tone averages 1, so the biome's colour stays the
albedo; `groundDetail.wgsl.ts` `TONE_GAIN` is 1 / that mean).

## Ground cover: roads, patches, long and short grass

A lawn the same everywhere is a carpet. `groundUse` (baked per clipmap lattice point, with the
biome's `GROUND_USE` blended by the region weights) breaks it up the way a walked land is broken:

- a network of **dirt roads** (the borders of a 640 m Voronoi diagram, half kept, bent by two
  scales of noise so they wind, 3.4 m wide) and **footpaths** (150 m cells, a third of the borders,
  1.3 m wide); a road shows only where the lattice can resolve it, so it fades out as the texel
  grows toward its width;
- **bare patches** in the grass, 3 – 9 m across;
- grass that grows **long** in drifts (the blades two to four times the height of the short lawn
  beside it) and **short**, trampled grass round roads, footpaths and bare patches, sparse toward
  them: the edge of a road is thin short grass over soil, not a wall of blades.

The fields are baked into the channels the ground and the grass already read (cover, blade height,
the grass's dryness, the soil's colour), so the surfaces and the scatter needed no new bindings. How
much of this each biome has is `GROUND_USE` in `biomes.ts` (trail, bare, tall, rocks, each 0 … 1):
the plains and the savanna are crossed by roads, the jungle by none; the mountains are the rockiest.

Grass itself follows: in the near rings (`grass.wgsl.ts`) blades gather in **tufts** round roots of a
1 m lattice with bare ground between them, some roots sparse and some tall, the blades leaning away
from their root; a blade is within a third of its tuft's height; tall grass is thinner and short
grass broader. Out in the far rings the blades are single and the lawn is level. Reeds need a wet
land, not only tall blades.

## Rocks

Eight kinds of rock, each an icosphere pulled into an ellipsoid, cut by planes (flat faces and sharp
edges, the way stone breaks), roughened by two octaves of noise and sat on a flat base, in three
levels of detail that share the kind's planes (so a rock keeps its shape as it recedes): boulder,
crag, slab, tall (a standing stone), block, shard, stone (smooth, a river stone) and cluster (a
handful of pebbles in one mesh). They are built once in TypeScript (`rocks/models.ts`), drawn
flat-shaded with each vertex's normal softened a third of the way toward the smooth one, and carry
their height over the base (the dirt they sit in, the moss on top) and an occlusion from the noise's
hollows.

`rockScatter` (compute, a thread a lattice cell, `gpu/rocks.wgsl.ts`) places them on four world-
anchored lattices round the camera:

| tier | cell | drawn out to | size (m) | chance |
|---|---|---|---|---|
| pebbles | 0.5 m | 15 m | 0.07 – 0.24 | 0.05 (and four times that on bare ground) |
| rocks | 2 m | 62 m | 0.28 – 0.95 | 0.018 |
| boulders | 8 m | 250 m | 1.0 – 3.2 | 0.014 |
| landmarks | 32 m | 900 m | 3.2 – 8.5 | 0.005 |

A cell holds at most one rock, at a hashed place, so rocks stay put while the tiers follow the
camera. The chance is the tier's, times the biome's rockiness (`GROUND_USE.rocks`, blended by the
regions), times drifts of stone (a noise of 83 m, from 0.15 between drifts to 1.7 in them, so the rocks gather in patches with open ground between), times what the slope does (rockier on slopes,
none on a cliff), with none in the water, the ruins, or out of view or too small to see (under 2.5
pixels). A rock sits a little into the ground, tilted two thirds of the way toward the slope, and
sinks away over the last fifth of its tier's radius, so none pops. It is appended to the bucket of
its kind and level of detail (chosen by the pixels it spans), in a first set of buckets if it is too
small to cast a shadow (a pebble's is under a shadow-map texel; 0.7 m and up cast) and in a second
if not; only the second set is drawn into the trees' shadow map, and of it only the two finest
levels of detail.

A rock's surface is the terrain's own rock (`rockMix`, `rockFine`) at the same scale, over the biome's colour
at its place, but laid on the rock in its own axes (the vertex shader hands the fragment the rock's local
position and normal, offset by a window of the tile of its own), so the texture stays on the rock as it is
carried, thrown and rolled. It is blurred (a coarser mip) and toned down (about 0.4 of the cliff's tone, 0.4 of
its cavity, 0.35 of its normal tilt, the chips of `rockFine` at under half), so the fractures read as soft
marks and not cuts; over that goes soft weathering of its own (broad mottling, faint streaks down the rock's up,
`vnoise` on the local axes, gone with distance), after the look of the Breath of the Wild boulders. A boulder at
a cliff's foot is therefore of a piece with the cliff in colour and grain, not in its cracks; with the dirt it sits in at its foot (the soil colour there), moss on its upper faces in
lush country, and snow on them where snow lies.

**Solid rocks.** The tiers from the stones up are solid (`rocks/field.ts`): a cell's rock is made on the
CPU as the scatter makes it — the same cell hash, biome rockiness, drift, slope, water and ruin tests
in double precision, and the same transform (size, axes, tilt toward the slope, turn) — and its solid
is a column height field of its finest mesh, rasterised in world axes the first time something comes
near it. So the hero is stopped at the rock's drawn face, not by a circle, and stands on its top: a
face steeper than the hero can walk is a wall from 12 cm up (no stepping up a boulder's side), a low
ledge or a gentle slab is walked onto, and a jump lands on a boulder's top (a steep top slides the
hero off). Foes share the hero's locomotion and meet rocks the same way.

**The box round the hero is the CPU's.** In a box of cells round the hero (24 m a side for the
stones, 40 m for the boulders, 64 m for the landmarks, `ROCK_ZONE`), the GPU scatter leaves the cells
alone and draws the CPU's list instead (`packRocks`, `rockList`: a thread a rock into the same buckets,
so no extra draws). Near the hero what is drawn is exactly what is solid: the 3 % of rocks where the
GPU's chance (from the clipmap's baked rockiness and gradient) and the CPU's (exact heights) fall on
either side of the cell's random number no longer show without a solid. Farther out the scatter's
own rocks are drawn as before. A rock taken from its cell (lifted, pushed) leaves its cell empty for
the scatter too (up to 128 cells remembered, the oldest coming back), and lives on as a free rock —
carried, flying, rolling, or lying where it came to rest (48 kept, the oldest lying one giving way),
drawn from the same list. The arrows still meet the boulders as circles (`rocks/collision.ts`,
80 % of the widest radius, boulders and up), taken rocks gone and rested big ones added.
`bun bench/webgpu.ts rockparity` (the benchmark pages have no hero, so no box) still reads back the
scatter's instances against the CPU's circles. Only the WebGPU view draws rocks: on WebGL2 none
stand in the way (`setRocksOn`).

**Lifting, throwing, rolling** (`combat/stones.ts`, the acts in `combat/world.ts`, the poses in
`hero/rig.ts` `poseRock`; the controls are in [COMBAT.md](COMBAT.md)). By how far a rock reaches from
its middle:

| rock | reach | the hero | thrown / pushed |
|---|---|---|---|
| stone | up to 0.34 m | stoops, lifts it in one hand (0.5 s), carries it on its palm by the shoulder, at 85 % of its pace (sprint and jump allowed) | overhand, the shoulders turning through: 17 m/s, 8 damage |
| rock | up to 0.62 m | stoops, heaves it overhead in both hands (0.85 s), walks with it at 55 % (no sprint, no jump) | wound back over the head and heaved: 10.5 m/s, 18 damage, knocks a foe down |
| boulder | up to 1.7 m, 2.6 m tall | leans in with both hands and drives (0.8 s) | rolls off at 5.2 / √size m/s and on down the slopes; a blow of 4 × its speed × √size (at most 48) to every foe in its way |

Bigger rocks (broad slabs, the landmarks) do not move. A throw goes along the view; a foe locked onto,
or one within ±20° of the view and in range (26 m / 12 m), draws it to its chest, and a crosshair on
the ground just ahead gives the throw its own arc. A thrown rock tumbles, glances off a foe it
strikes, bounces once if it comes down hard, then rolls (rolling drag 0.22 g for stones, 0.12 g for
boulders, plus air), off ledges, round trunks and over stones much smaller than itself, and lies
still; in deep water it sinks. Hurt, climbing, swimming or gliding, the hero drops what it carries;
the action key puts it down ahead. In multiplayer the rocks are each player's own (a rock lifted is
still on its cell for the others, and their poses show the arms, not the rock).

## The shape of the land

Most of what read as cut with a ruler was a **contour of a smooth field**. A tableland's wall, a
mesa's tier, a region's edge and a valley's wall are each where a field of kilometres' wavelength
crosses a level, and over a few hundred metres a contour of such a field is a straight line. The
fixes bend, break and roughen those contours at the scales the eye reads them at (100 m – 1 km),
and give steep faces the ledges, ribs and knobs that stone has. Every change is in the CPU twin
(`terrainMath.ts`, `hydro.ts`), the WGSL (`worldgen.wgsl.ts`) and the GLSL (`landsGlsl.ts`), driven by the
same tables in `constants.ts`.

- **Karst towers** (`TOWER`, `karstTowers`): a tower was a circular cone, `H (1 − q^3.5)`, a cylinder
  with a flat top, every one the same shape. A tower is now an ellipse (stretched along a bearing
  that is the district's grain give or take), its outline wandering by one displacement of the
  point shared by every tower; its wall is fluted into buttresses and gullies (the noise read
  round a circle, so no seam; the first octave drifts with the height so gullies wander and
  join); it rises from a skirt through a ledged wall (strata tilted round it by the flutes' noise)
  to a domed crown that swells and is lumpy; and some lattice cells hold none. The fades to the
  mean (`LAND_MEANS.tower`, now 19.5 m: the old 21 was a stale figure, the old function's mean was
  38) take the finer octaves out as the footprint grows.
- **Crags** (`CRAG`, `cragDelta`): where the land under the gullies is steep (a rock face), three
  octaves of ridged noise (53, 26 and 11 m) deepen with the slope into buttresses, knobs and ribs,
  and the strata by height (24 m thick, their phase tilted by a slow noise) flatten the face into
  benches between risers. Nothing where the ground is walkable. A mesa's wall gets the knobs in its
  field. Each octave and the ledges fade out where the pixel's footprint could not resolve them.
- **Mesas** (`mesasE`): a tier's wall used to be a cliff of 30 – 50 m in one metre, drawn in
  rectangular steps by any clipmap (and a lowest apron stood 30 m up where the region began, a cliff
  along the whole of its edge): the wall is at least about five texels wide at every level of
  detail, the mesas fade in over the region's first quarter, rise over the whole of the region's
  edge, and their walls meander (a wander of the region weight, and of how much the field is
  lowered at the edge).
- **Regions' edges** (`LANDS.edge`): a region's fields are read in a plane shifted by two scales of
  noise (170 m at 300 m, 56 m at 150 m), so every biome edge, the snow line included, bays and spurs
  instead of running a smooth curve for kilometres.
- **Mountain crests** (`GEO.rangeBend`): the chains were bent by a 9 km warp and ran straight for
  kilometres in between; three finer bends (380 m, 130 m, 45 m) break them.
- **Tablelands** (`GEO.platWarp`): the outline's warp is larger (210 m and 60 m).
- **Valleys** (`HYDRO`): each side of a valley's floor keeps ¼ – 1 of its width on its own noise
  (never less than the channel and a margin: where a meander swings against the narrower side the
  water cuts the bank under the wall), the walls' run past the floor's rim is stretched by up to a
  third and shifted sideways by up to 110 m by noise at three scales (bays and spurs on a canyon's
  walls), steep reaches meander a little more (`meanderMin` 55 m), and the channel's width and the
  banks' slope vary along the reach.
- **Lakes and ponds**: a shore's lobes (three octaves) and its stretch along one axis are drawn from
  a hash of the lake, so no two shores are alike.
- **Dunes**: the crests are bent at a third scale.

`tests/landforms.test.ts` pins the constants the code relies on, determinism, bounds and that nothing
steps. GPU against CPU elevation (`bun bench/webgpu.ts landprobe --config=karst`, `walls`, `lakes`):
at most 6.5 cm over 64 × 64 probes round every view, none over 10 cm.

## Cost

It is not free. Measured on an M1 MacBook on AC power, hero and combat left out of both sides (they are other work), against
the last commit's renderer on this tree's land (so only the ground, grass and rocks differ), two ways:

**In play** (`bench/play.ts`: a production build in headless Chrome, WebGPU, uncapped so the GPU paces it, a 1280 × 633
canvas, the real frame loop; walking 15 s from each place, two rounds in alternating order): the frame costs **+0.4 ms on
average (+10 %)**, from +0.26 to +0.62 ms (+6 to +15 %) over ten biomes (frame times before → after, ms):

| plains | forest | jungle | desert | badlands | savanna | mountains | highland | tundra | taiga | spawn |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.60 → 3.98 | 4.05 → 4.37 | 3.84 → 4.24 | 3.36 → 3.85 | 4.17 → 4.43 | 4.01 → 4.63 | 4.35 → 4.81 | 4.14 → 4.56 | 3.48 → 3.82 | 3.95 → 4.37 | 3.97 → 4.29 |

(A second session over four of them read +0.5 to +0.85 ms: sessions on this machine differ by ± 0.2 ms.) Taking each part
out at four of the places, the ground's detail costs about 0.45 ms and the rocks about 0.35 – 0.45, together 0.66 (they
overlap); the new grass, the cover and the colours are within noise (about 0.05). At a 60 Hz display it is 4.3 of 16.7 ms.

**On curated views** (`bench/webgpu.ts stills --timing`, 15 ground-level and 41 aerial views chosen to show the ground,
the grass and the rocks): +1.0 ms at ground level (4.3 → 5.3 ms, +24 %) and +0.45 ms from the air (+13 %). That is the worst
case, not the play: the views are the densest places, and the harness draws three frames from one command buffer after
one `setFrame`, which makes the rocks' counters run on (up to three times the instances) and hides the hazards between
frames that a real loop has.

- **The rocks' cost is the draws, not the rocks.** In play the scatter alone is free (+0.05 ms) and the draws cost about
  0.3 ms, in views with 60 rocks drawn and in views with 400. In the bench a pass that places nothing but issues the same
  indirect draws costs the same, while plain draws, or indirect draws from a buffer no pass writes, cost nothing, and 8
  draws cost as much as 64. Double-buffering the arguments, resetting them inside the frame's commands, the
  `indirect-first-instance` feature, scattering at the end of the frame before (one frame stale) and keeping the
  arguments in the grass's own indirect buffer each left it where it was; the cause is not found.
- **The generator** (a full rebuild of the 3.15 M texels of the clipmap): the pass takes 335 ms against 258, +30 %, all of it the
  land's new terms (the same land under the old renderer takes 334 ms: the biome bake's cover costs nothing measurable);
  streaming while walking takes 0.59 ms a frame against 0.45 (+0.14), at 200 m/s 2.98 against 2.58.
- **Start-up**: the nine textures' paint pipelines compile beside the rest, +0.33 s of `compileMs` (669 → 1003 ms) on a warm
  shader cache.
- **The CPU's world function** (physics, collision, the server): 10.9 µs a point against 8.8 (+22 %), a physics tick 38 µs against
  30; `tests/terrainMath.test.ts` budgets are 14 and 50 µs now (they were 12 and 40).

What holds it down: the detail's taps are made only where a material is (per-pixel branches on `textureSampleGrad`), a lawn's
detail fades out over 4 – 11 m and every detail is gone by 160 m, anisotropy is 2, the rocks are culled before every fetch
and their detail is gated by their size, only the two finest levels of the rocks of 0.7 m and up cast shadows, and the grass
scatter drops what the camera cannot see and the gaps between tufts before it reads the biome layers (exactly the same
blades, about 0.1 ms off the frame in the bench, and a saving for the old pass too).

If it must come down: `GAME_VARIANT.rocks` (`engineGpu.ts`) drops the rocks and their pass, `DETAIL_NEAR` / `DETAIL_FAR`
(`groundDetail.wgsl.ts`) shorten the detail, the pipeline constant `STUB_DETAIL` turns it off (as the variants
`game-high-nodetail` and `-norocks` of `bench/webgpu.ts` do), `Rocks.density` thins the pebbles and stones.

## What is not done

- The WebGL2 fallback has none of the ground's textures, the rocks or the tufts.
- Rocks are climbed like the land's walls ([`HERO.md`](HERO.md#climbing): a low one is vaulted onto), but their
  solid is the top of the mesh as a column, so under a bulging boulder's widest point the hero stops a hand's
  breadth short of the undercut, and nothing hangs under an overhang.
  Past the box round the hero (24–64 m), about 3 % of the scatter's rocks have no solid — too far to be met.
- Making the rocks of the box round a new place costs about 36 ms once (the spawn, a respawn, a jump in fly
  mode); following the hero costs about 15 µs a frame.
- The look of the cliffs close up is the textures' alone (their relief is a normal map); the
  geometry's crags are 1 – 10 m.
