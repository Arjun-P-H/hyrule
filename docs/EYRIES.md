# The Eyries: technical plan

How the sky civilization of [SKY_CIVILIZATION.md](SKY_CIVILIZATION.md) is built, drawn, walked and
kept fast. It replaces the archipelagos of "The sky realm" in EXPERIMENTS.md and keeps the rest of
that work: islands, water with a source, the sky rings, shadows and far detail.

## 1. Goals and limits

- **The reference images, element by element** (§2): structured clusters of islands, architecture
  with interiors, water from aqueducts to underwater plants, a living ecosystem. Every element is
  tracked as a checklist.
- **Infinite and procedural.** Everything is a pure function of (cell, seed), built at run time.
  There are no asset files, as with the trees.
- **Frame budget**, measured on this M1 Air the way the performance pass was:
  - land views (no Eyrie near): unchanged;
  - Medium preset: ≥ 100 fps in a city;
  - High preset: ≥ 60 fps everywhere.
- **No stalls.** Eyries are built a few milliseconds a frame, as far cells are today. Nothing
  allocates per frame.
- **Walkable.** The hero stands on bridges, plazas, stairs and floors, is stopped by walls, and walks
  into buildings.

## 2. From the reference images to features

| Element in the images | Feature | Phase |
| :--- | :--- | :---: |
| dense clusters of islands, each different, inverted-spire undersides with many hanging spikes | Eyrie classes and their island layouts, underside lobes and spikes | 2 |
| gothic cathedral-castles, needle spires, round and square towers, scale-tiled roofs, finials | kit: tower bodies, roofs, spires, pinnacles, buttresses; cathedral and roost grammars | 1–2 |
| multi-arch viaducts and aqueducts between islands | vow-bridges with hanging piers, an aqueduct channel with flowing water | 2–3 |
| circular plaza with radial mosaic in a turquoise lagoon, rotunda with a blue dome, causeway | Mirrorwell's lagoon, islet, rotunda, mosaic floor | 3 |
| giant stone ring with bridges | Ringhold's Great Halo, runestones | 3 |
| crystal mountain with a beam of light | Lanternpeak's Vaelspire and the Thread | 3 |
| cherry trees (pink, white), yellow and red trees, purple trees | orchard palettes and blossom leaf sprays, ember maple, wisteria; grammar-planted groves and avenues | 2 |
| river flowing down a chain of islands, falls into pools | Petalfall's Stair of Waters: falls that land in pools | 3 |
| pagodas with upswept eaves, red gates, rope bridges | Petalfall kit: wind-house, gate-arch, rope bridge | 3 |
| giant tree whose roots bind islands | Elder Bough | 4 |
| volcanic islands, lava, dark citadel | Cinderforge: ember biome, lava glow, smoke, forges | 4 |
| snowy fortress, vortex in the sky | Rimewatch, the Gyre's Eye | 4 |
| ruined castles, broken towers, overgrown | Hushfall; ruined variants of every kit piece | 4 |
| airships with sails, docks | skyships on routes between Eyries, docks, cranes | 5 |
| balcony with balustrade, market stall, banners with a gold emblem, awnings | market terrace, stalls, banners with emblems | 3 |
| glass-domed greenhouse, observatory domes | greenhouse and observatory pieces | 3 |
| lanterns everywhere, lit windows at dusk and night | emissive lanterns and windows by the sky's clock, bloom, lantern light | 6 |
| clear turquoise water, underwater rocks, plants and light | lagoon tint, caustics on beds, waterweed, stones, lily pads, lotus, petals on water | 5 |
| fish, flocks, eagles, butterflies, fireflies, falling petals | koi, swifts, hawks, petalwings, lantern-flies, petals | 5 |
| sun rays, mist, cloud sea | existing sky and mist; bloom (phase 6) | 6 |
| interiors | interiors and props by building purpose; collision | 7 |
| a planet in the sky | not planned: the cycle sky's moon stays (a later option) | — |
| griffins with riders, the bird people | characters: later, as asked | — |

## 3. Systems

### 3.1 Eyries (CPU, `src/lib/skyworld/eyrie/`)

- **`classes.ts`**: the eleven classes, their weights, compatible features, palettes and parameter
  ranges.
- **`layout.ts`**: one candidate per 8,192 m cell. The class is picked by hash, and the class's
  arranger places the islands. Each arrangement is its own function, so no two classes share a
  silhouette:
  - radial wards, bowl and wells, helix, peak, chain, tree-bound, shards, ice, broken, harbour ring
    and low isles.
- **Islands stay `SkyIsland`**, with new fields that the WGSL twin reads:
  - **underside lobes:** 1–4 extra hanging spikes off-centre, so undersides read as clusters of
    inverted spires;
  - **underside style:** rock, crystal veins, roots, basalt or ice;
  - **water tint:** turquoise lagoons;
  - **lake size and placement** from the class (Mirrorwell's lagoon covers most of its bowl).
- **`settle/*.ts`**: the class's grammar. It plans districts (streets, plazas, lots) on each isle and
  fills them with structures:
  - cathedral, roosts, rotunda, walls, market, docks, cisterns, gardens;
  - vow-bridges between isles;
  - water features: aqueducts, fountains, spouts and falls;
  - lanterns, banners and planted trees;
  - fauna anchors: flocks, ponds and nests.

  Every placement checks the ground: footprint slope, distance to the rim, water, other footprints.
- **Built a few milliseconds a frame**, nearest first, and cached per seed, exactly as today's far
  cells.

### 3.2 Structures and the kit

- **A structure** is one culling group (a roost, a bridge span, a plaza) with:
  - a bounding sphere;
  - its piece instances (type, position, yaw, tilt, scale, palette, flags);
  - its interior pieces;
  - its colliders.
- **The kit** (`src/lib/skyworld/kit/`) is a procedural mesh library built once at start-up (like the
  tree models), with three levels of detail per piece. Mesh builders: box, prism, lathe (spires,
  domes, finials, columns, balusters, basins), extrusion along a path (arches, cornices, rails),
  stairs.

  Pieces come in two kinds:
  - **stretchable:** tower bodies, walls, roofs, podiums and decks. Their detail is painted by the
    shader in metric surface coordinates, so stretching keeps stone courses and windows true to size;
  - **fixed:** cornices, merlons, pinnacles, finials, balconies, arches, lanterns, statues and props.
    These are scaled uniformly.
- **Vertex (24 bytes):**
  - position f32×3;
  - normal snorm8×3, plus a uv-axis code;
  - material, occlusion, facade (window pattern) and animation codes (unorm8×4);
  - surface coordinates unorm16×2, in metres at unit scale.

### 3.3 Drawing structures (`src/lib/gpu/structures.ts`, `structures.wgsl.ts`)

- **Instances.** All piece instances of the Eyries in range sit in one storage buffer (32 bytes
  each). It is rewritten only when an Eyrie enters or leaves range.
- **Culling on the CPU, per structure** (a few thousand spheres: ~0.1 ms). Each visible structure
  takes one level of detail for all its pieces, so its parts never disagree. Small pieces (lanterns,
  props, banners) are dropped past their own distance. Interiors are drawn only near.
- **The draw list.** A counting sort puts visible instances into buckets of (piece, level). The list
  of instance indices is uploaded, and each non-empty bucket is one instanced indexed draw. That is
  typically 30–80 draws, and only the buckets in use.
- **Shading** is one fragment shader with materials:
  - ashlar stone, marble, scale tiles, timber, metal, glass, crystal, cloth with emblem, mosaic,
    woven nest, rope, lava;
  - painted facades: pointed-arch windows per floor that glow from dusk to dawn, flickering a little
    per window;
  - keystones that glow faintly blue at night.

  Lighting goes through the existing light API (key, sky, bounce, fog, display).
- **Shadows.**
  - Near: structures are drawn into the forest's sun shadow map (±90 m, 9 cm texels), which every
    surface already samples.
  - Far: their coarse level goes into the realm's shadow map with the undersides.
- **Ground.** Each settled isle gets a **district map**: an RGBA8 signed-distance texture, 512² over
  the isle's square, holding paving, footprints, gardens and channels.
  - The isle's ground shader paints cobbles and flagstones from it.
  - The grass scatter and the trees keep off paving and footprints.
  - The CPU keeps the same bytes for the hero and the trees' twin.

### 3.4 Water

- **Aqueduct channels** are ribbons on bridge decks and terraces, drawn as streams are, in world space.
- **Falls** have more sources: cisterns, aqueduct ends, spouts and lagoon notches. A fall that lands
  in a pool ends at its surface in a splash and mist. The realm's 64 falls become 256.
- **Lagoons** get the class's water tint. Beds under water get caustics, sand and pebbles.
- **Water life** is placed by hash from the lake's own record in the vertex shader, so it costs no
  CPU:
  - lily pads, lotus and drifting petals on the surface;
  - waterweed and stones underwater;
  - koi swimming in loops.

### 3.5 The living sky (`src/lib/gpu/fauna.wgsl.ts`)

These are stateless: positions are functions of time and a hash, so nothing is simulated on the CPU.
- **Flocks** (swifts round spires, hawks above, herons over lagoons) are low-poly birds with flapping
  wings, all in one draw. Each Eyrie lists its flocks.
- **Particles round the camera:** petals near blossom, petalwings by day, lantern-flies at dusk. Their
  density follows where the camera is (the CPU knows the biome and the hour).
- **Skyships** sail routes between Eyries' docks, their poses set per frame (a few dozen instances).
- **Moving pieces** are animated in the vertex shader: banners, awnings, wind-wheel sails, runestones,
  crystal pulse.
- **Effects:**
  - the Thread: an additive beam;
  - the Gyre's Eye: a turning spiral of cloud;
  - forge smoke;
  - the Halo's hum: rune glow.

### 3.6 Night light (phase 6)

- Emissive windows and lanterns are driven by the cycle sky's hour.
- **Lantern light:** a small top-down light map round the camera, splatted from nearby lanterns
  whenever they change. Walls and paving near a lantern sample it for warm light, so hundreds of
  lanterns cost one texture fetch.
- **Bloom** in the present pass: a 1/4-resolution bright pass and blur.

### 3.7 Collision (`src/lib/skyworld/structures.ts`)

- **Colliders** are boxes, ramps (a box with a sloped top), cylinders and tubes (hollow cylinders with
  door gaps), from each piece's definition and its instance's transform.
- **A grid of 16 m cells per Eyrie** indexes them.
- **The hero** stands on the highest floor within a step of their feet and is pushed out of walls,
  generalising the ruins' `collideRuins`. The camera keeps out of walls.

## 4. Budgets

| Item | Budget |
| :--- | :--- |
| Eyrie build (CPU, one Eyrie) | ≤ 25 ms in all, ≤ 3 ms a frame (built over several frames) |
| Kit build at start-up | ≤ 150 ms |
| Piece instances in range | ≤ 65,536 (2 MB) |
| Culling and draw list (CPU) | ≤ 0.3 ms a frame |
| Draws | ≤ 120 a frame for structures |
| Visible triangles in a city view | ≤ 1.5 M |
| Structures' GPU time, city view at Medium | ≤ 2.5 ms |
| District maps | ≤ 16 isles near the camera, 512² RGBA8 each (16 MB) |

## 5. Phases

Each phase ends with the tests green, the production build working, screenshots of every class it
touches, and a GPU timing against the phase before (paired, both orders, as in the performance pass).

1. **Kit and renderer.** The mesh builders and the first pieces (roost, roofs, spires, walls, hall,
   podium, stair, bridge span and pier, rotunda, dome, lantern, banner, statue). Structures culled and
   drawn with materials, painted windows and shadows. One test settlement on the spawn's hub.
2. **Eyries and the Crown.** Classes and layouts with new underside lobes. Settlement grammar for the
   Crown: cathedral, rotunda, roosts, walls, plazas, gardens, vow-bridges, district maps with grass
   and trees kept off. Orchard palettes. Collision on bridges, plazas, stairs and walls.
3. **More classes.** Mirrorwell, Ringhold, Lanternpeak, Petalfall, Galeharbour and Mistwells with
   their pieces. Aqueducts and falls into pools.
4. **The rest.** Elder Bough, Cinderforge, Rimewatch and Hushfall (ruined variants).
5. **Water and life.** Lagoon tint, caustics, water flora and koi, flocks, petals, butterflies,
   fireflies, skyships, animated pieces and effects.
6. **Night.** Window and lantern emission, the lantern light map, bloom.
7. **Interiors.** Inside pieces and props by purpose (roost, cathedral, rotunda, archive, market,
   cistern), with doors, floors and stairs walkable.
8. **Tuning and record.** Profiles, levels of detail, the gallery, the EXPERIMENTS.md entry.

### Where it stands

Measured and recorded in `docs/EXPERIMENTS.md` ("The Eyries").

| Phase | Done | Still to do |
| :--- | :--- | :--- |
| 1–2 | The kit (105 pieces), the instanced renderer, all 11 classes and their layouts, the Crown, district maps, colliders, building a slice a frame | — |
| 3–4 | Every class's grammar and landmarks; built and rope bridges; Hushfall's ruins; aqueducts down the Crown's, Mirrorwell's and Galeharbour's bridges (a cistern head at the top, a fall at every step down, a spout into a basin at the foot) | — |
| 5 | Lake life (pads, lotus, reeds, waterweed, koi); flocks; skyships under way; motes; the Thread, forge smoke, the Gyre's Eye | caustics on lake beds; petals on pools |
| 6 | Window, lantern and crystal emission; bloom; the lantern light map (`gpu/lanterns.ts`: a 256 m wrapping map round the camera, warm pools on walls and paving by night) | — |
| 7 | Every building of the catalogue with an inside, walkable, furnished and lit as an inside: roosts, hall-houses, the cathedral's nave and apse, the Moot Rotunda, the Rotunda of the Accord, observatories, greenhouses, cisterns, keeps, forges, markets, wind-houses, the archive spire (its ramp winding up the shaft, its cord racks between the turns) | interiors for the upper floors (the ramps end at hatches) |
| 8 | The record; the gallery views (`framed-eyrie*.json`) | an in-game A/B against a build of the code before the Eyries |

Two choices made on the way differ from the plan above:
- Water life is placed as structures by each island's grammar (`eyrie/lakes.ts`), not by hash in a
  shader: the pieces then cull, light and shadow as the rest do.
- Insides are the backs of the outer pieces. Inner walls are built facing out, so from inside they
  are back faces, and the shader finishes them as an inside. There is no separate interior pass.

## 6. Tests

- **Kit:** every mesh's indices are in range, its normals are unit, and its bounds hold its vertices.
  Triangle budgets per level. Colliders lie inside the bounds.
- **Layout:** every class is built and makes a different arrangement. Islands keep clear of each
  other and of the land. Bridges stay walkable (slope, nothing crossing another island).
- **Settlements:**
  - footprints lie on their island, clear of water and rim, and do not overlap;
  - foundations reach the ground;
  - every lantern stands on something;
  - every fall has a source and either lands in a pool or turns to mist above the clouds.
- **GPU twins:** the instance layout against the WGSL structs, and the islands' new underside terms
  against the CPU.
- **Hero:** crosses a vow-bridge, climbs a stair, stands on a plaza, is stopped by a wall, and walks
  into a roost.
- **Bench:** a gallery view of each class (day and dusk), and GPU time with and without structures.

## 7. Risks

| Risk | Mitigation |
| :--- | :--- |
| Draw count grows with piece types | only buckets in use are drawn; few, general pieces; props only near |
| Overdraw in city views | per-structure levels of detail, opaque only (no discard), ordered near to far by structure |
| The CPU build stalls frames | build in slices (≤ 3 ms a frame), nearest first; the kit once at start-up |
| Stretching shows | detail painted in metric coordinates; fixed pieces scaled uniformly |
| Walls without a proper GI look flat | baked occlusion per vertex, interior ambient by enclosure, near shadow map, lantern light |
| The CPU twin and the WGSL drift apart | parity tests for every new term, as for the islands and the trees |
