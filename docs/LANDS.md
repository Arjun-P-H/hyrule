# The lands: fifteen biomes, their landforms, water and air

The world below the sky islands is no longer five colour anchors over one heightfield. It is a
patchwork of fifteen biome **regions**. Each region shapes its own land (karst towers, mesas,
dunes, a volcano's cone, a marsh's flats), shares one drainage (rivers that run from the heights to lakes or the sea,
level lakes in basins, meadow ponds, waterfalls), grows its own ground cover and trees, and fills its own air (jungle
humidity, marsh fog, desert glare, volcanic ash). Regions blend smoothly at their borders, so the
lands merge instead of meeting at seams.

The HUD names the land under the camera (*Land:* row, `sky/air.ts` `landName`).

A view of each, rendered by the game's renderer at High: [`lands-contact-sheet.jpg`](lands-contact-sheet.jpg).
How the land is shaped (ranges, rivers, lakes, erosion, forest spacing, lava), and the research
behind it: [`TOTK_TERRAIN.md`](TOTK_TERRAIN.md).

| Biome | Where it forms | The land | Ground and trees | Air |
| :--- | :--- | :--- | :--- | :--- |
| Grassland plains | mild, dry lowland | low rolling hills, meandering rivers, ponds | open grass and flowers; copses of oak, a few lone trees | clear |
| Temperate forest | mild, moist lowland | rolling hills, streams, small lakes | moss and leaf litter; oak and birch woods with clearings, maples in autumn colours | a little mist |
| Tropical rainforest | hot and wet | low hills, many rivers, waterfalls | lush dark green; jungle trees and palms spread over most of the land, one in eight a giant | humid haze, mist |
| Jungle karst | hot, wet, karst geology (and hot wet plateaus) | steep limestone towers on a 300 m lattice, highlands | heavy undergrowth; jungle trees and palms spread to the tops of the towers | humid haze |
| Wetland marsh | wet lowland where the marsh field is high | flat lowland: shallow pools in the marsh's heart, hummocks round them, channels | mud and reeds standing in the water; birch and dead snags | fog lying low |
| Desert dunes | hot and arid | crescent dunes (240 m) with slip faces and ripples, open flats | sand; cacti, an oasis palm | warm glare |
| Canyon badlands | hot or mild dry rugged country, hot dry plateaus | tiered mesas and buttes, cliff walls, talus, canyon-walled rivers | banded rock strata (cream, rust, purple-grey); dead trees, acacia | warm haze |
| Savanna | hot, half-dry | broad flats and gentle swells | tall golden grass; lone acacias | warm haze |
| Volcanic fields | a volcano (a candidate in each 8 km cell, kept where it stands on land, off the ranges, in warm land) | a cone rising 380–520 m to its crater's rim, ash plains round it | black ash and rock, a molten lava lake in the crater and lava on its gentle rivers; snags and scorched pines in pairs | ash haze |
| Rocky mountains | the range crests and lone massifs | mountain countries a few km across, with peaks to ~0.9 km and passes, gullied slopes, gorges (no boulders or caves yet) | rock above the treeline, snow on the peaks; pine forests below | crisp |
| Alpine highland | cold to mild plateaus | high rolling tableland, tarns | alpine turf, rock outcrops, snow; sparse pines | crisp, some mist |
| Snowy tundra | cold lowland | gentle swells, frozen lakes | permanent snow, wind drifts, ice; stunted pines | crisp, bluish |
| Boreal taiga | cool lowland | low hills, many lakes | needle litter; pine and birch woods | some mist |
| Ancient forest | sacred groves (a candidate in each 8 km cell, where the land is mild and moist) | the land it sits on | deep moss; giant oaks and jungle trees up to 2.2× in mossy and golden palettes | thick mist |
| Lake district | cool or mild lowland where the lake field is high | lakes on its rivers, and basins holding lakes where rivers end | lush shores; pine, birch and oak | some mist |

The coast is a band, not a region: wherever the land meets the sea it ramps down to beaches, or
breaks off in 45–95 m cliffs where the cliff field is high, over a shelf that deepens to 126 m.

## Where each region forms

`terrainMath.ts` `regionsInto` (twinned in `worldgen.wgsl.ts` `landRegions` and
`landsGlsl.ts`). The inputs:

- **Sea-level temperature** T0 (°C): `CLIMATE.tBase` + `tAmp` × a 5.8 km noise field (plus a finer
  one). The air cools at 10.5 °C/km of height, so the snow line follows the land and the climate.
- **Moisture** M: a 3.2 km field, wetter toward the sea.
- **Geography** (`hydro.ts` `upliftAt`): the range crests and lone massifs (mountains), and the
  tablelands (plateaus).
- **Geology:** rugged country, lake districts, karst and marsh. Each is a low-frequency field put
  through a smoothstep (`LANDS.*`).

A smooth decision tree turns these into weights that sum to 1 (`LANDS.cold/cool/hot`,
`dry/hotDry/arid/wet`):

- **Lowland, by climate:**
  - cold gives tundra;
  - cool gives taiga, or the lake district where the lake field is high;
  - mild gives plains (or badlands where rugged) when dry, forest (or wetland in the marsh field)
    when moist;
  - hot gives desert when arid, savanna when half-dry, badlands where rugged, and rainforest, jungle
    (karst) or wetland when wet.
- **Plateaus:** alpine highland in the cold to mild, mesa country (badlands) in the hot and dry,
  karst highland (jungle) in the hot and wet.
- **Ranges:** rocky mountains.
- **Overlays:** volcanoes and sacred groves are placed by hash, one candidate of each per 8,192 m
  cell (the groves' lattice offset by half a cell). Each is kept to the middle of its cell, so only
  that cell can hold it. Both lattices lie in the warped plane, so an overlay's whole fade lies
  inside its cell. Volcanoes stand only on land, never on a range,
  and only where the sea-level temperature at their centre is at least 4 °C (TotK's stands in hot
  country); groves take only mild, moist land. Overlays replace the regions under them, fading out
  toward their rim.

Every frequency is dyadic (n / 2²⁰ per metre, `constants.ts` `F`), so chunk seams stay exact.

## The landforms

The big shapes (ranges, massifs, tablelands, basins, volcanoes, the coast) and the rivers' valleys
are shared geography (`hydro.ts`, see [`TOTK_TERRAIN.md`](TOTK_TERRAIN.md)). Each biome's
`LANDFORM` (`biomes.ts`) shapes the land between them, as a row of 10 parameters:
- hills and ridges (heights in metres);
- karst, mesa, dune and flat;
- the valleys' wall steepness `k`;
- `riverMin`: the drainage a river's head needs here for a reservoir (deserts keep only the big
  rivers);
- erosion: the gully filter's strength;
- pond: the chance of a meadow pond.

The rows blend by the region weights, so a forest running into mountains climbs into them, and
dunes thin out into the savanna. `landSurface` builds the height in steps:

1. **The uplift** U: the geography, warped.
2. **The detail:**
   - rolling hills and mountain ridges;
   - karst towers;
   - dunes (crescent ridges with 240 m spacing, bent and crossed, plus ripples);
   - mesas (tiers with cliff walls and talus);
   - gully erosion (`erosion.ts`, MPL-2.0) on hills and mountains: the gullies run down the land's
     own slope (its uplift, hills and ridges, sampled 128 m apart) and are laid out in the world, so
     they branch down each flank instead of lying side by side.

   Karst, mesas and dunes appear only where their region dominates. The detail fades out on
   valley floors, in lakes, in the crater and toward the coast. Beside rivers and marshes it may
   only rise: a hollow at their edge would spill the water or leave a river standing over it.
3. **The valleys:** the land is clamped between the river terraces and the valleys,
   `min(max(U + detail, F), R)`.
4. **Marshes, ponds and the crater:** marsh flats on flat lowland, with pools only in the marsh's
   heart and the hummocks round them raised over the pools' level; a pond's bowl in its meadow; the
   crater's rim.
5. **The coast:** ramp, cliffs, shelf.

## Water

`surfaceAt(x, z)` returns the ground and the water's height. Water stands only where something
holds it: a river channel, a lake, a marsh's pools, a pond, a lagoon, the sea. Elsewhere the table
lies `DRY_TABLE` (3 m) under the ground. `lastFlow()` gives the river's current there, and
`lastShore()` the level of the water actually beside a point, which shores and shore trees are judged
from.

- **Rivers** flow from lakes: a river shows only below one, and each river system starts at a
  reservoir (a lake of up to 840 m across at its head). They run in channels cut into their valley
  floors (14 m wide where they leave their lake, up to 180 m; 3 + √acc m deep), at their reach's
  level. A valley with no lake above it is dry: no channel and no falls, only its floor and walls,
  which rise from the floor gently (a swale). A river's water never stands over a bank. It is capped at the lowest surface
  out to its far bank (another reach's ground or water, a lake's shore), so tributaries come down to
  the river they meet and siblings join without walls of water.
- **Lakes:** a river widens into a lake at half the nodes on flat land below a lake (100–600 m
  across, 9.5–27 m deep, held level by its outlet, with land above its level all round it), and a
  sink holds a lake just under the lowest ground round it.
- **Ponds** lie in flat meadows away from rivers (median ~32 m across, 90 % under ~82 m; 4–8 m
  deep), judged on the land itself at their centre and round their rim. **Marsh pools**
  lie only in a marsh's heart, with the hummocks round them standing over the water.
- **The sea** fills only land below its level by the coast, so it never climbs a shore.
- **On screen** the water mesh runs level under a dry bank, just below its ground: the ground hides
  it, and the shoreline is where the two meet. (Before, water climbed each bank toward its dry
  vertices and drew white sheets down every steep shore.)
- **Lava** is the volcano's water: its crater lake, and its rivers where they run gently. It is
  drawn molten, with crust rafts drifting on it, and is solid underfoot.
- **Ice:** water freezes where it lies 80 m above the snow line (`CLIMATE.iceAbove`): tundra lakes,
  high tarns, the cold sea. Ice is walkable (`hero/locomotion.ts`) and drawn as pale blue,
  cracked, glossy ice, dusted with snow in patches.
- **The local water level** replaces the old global `W_LEVEL` everywhere it mattered:
  - swimming and the camera (`hero/`), and the current that carries a swimmer;
  - ruins' footings;
  - the sky realm's waterfalls and lakes;
  - grass and reeds;
  - the underwater view (`FRAME.camWater`, `frame.ts`), which every shader that asks "is the camera
    underwater?" now reads.
- **On the GPU:**
  - the clipmap's height texture holds ground, water and the current (rgba32float), and the water
    mesh reuses the ring vertices on its water channel;
  - the `ringTris` compute pass compacts each ring's wet cells into a list, drawn with one indirect
    draw per ring; this took the water from about 21 % of the frame to about 3 %;
  - the water's ripples ride the current (flow maps), with foam on rapids. The fragment shader finds
    falls from the mesh's own slope and draws white streaked water down each step between pools.

## The ground's look

The biome **bake pass** (`terrain.wgsl.ts` `biomeShader`, run after the sun pass on every clipmap
update) writes four layers per lattice point:
- `mat`: snow line, wetness or lava, a patch noise, flower bloom;
- `bioA`: soil colour and cover;
- `bioB`: grass colour and blade height;
- `bioC`: rock colour and strata.

The colours come from the region-blended `LOOK` table. The pass is separate because WebGPU's
default limit is four storage textures per stage.

The pixel shader's `groundLook` finishes the look:
- grass over soil by cover and slope;
- rock on the steep, banded in strata in canyon country;
- sand on low, gentle sea shores (and oases);
- darker damp ground by the water;
- snow above the snow line, broken into drifts and kept off cliffs;
- glowing lava where it spills from the crater.

Snow's albedo (`SNOW_ALBEDO`, 0.4–0.5) is below fresh snow's. The display is exposed for the green
lands, and brighter snow sat in the tone curve's shoulder: 84–99 % of a snowfield's pixels clipped
to white. Near the camera, snow is also shaded with wind-drift relief:
- drifts about 12 m long and sastrugi about 2.5 m long, lying across the wind;
- applied as a tilt of the normal, faded out as they shrink below a pixel.

Grass follows the same layers. Reeds stand in shallow water (tall blades). There is no grass on
snow, beach or lava, and flowers grow where the bloom layer is high.

## Trees

`forest/placement.ts` (twinned in `forest.wgsl.ts`) maps the region weights and the air temperature
onto 17 flora zones. Montane, alpine and treeline bands rise with height, but the dry, hot and
cold-country zones keep their own trees (a volcano its snags, up to the tree line). Snow on a tree
follows the ground's own snow line, the regions' frost included, so the tundra's pines are white and
a volcano's snags never are. Every zone has its species pairs for:
- forests (stands), groves and lone trees;
- shores;
- giants: a share of trees grown up to 1.25–2.2× (the scale range is now 0.55–3.0).

The ancient forest's trees take sky-flora palettes (mossy, golden; flag `F_PAL`).

The amounts are Tears of the Kingdom's, measured per region from its game data and matched region
by region ([`TOTK_TERRAIN.md`](TOTK_TERRAIN.md)). Trees stand in clumps: stands of 28–55 trees/ha
with trunks 8–10 m apart, and copses. A wooded region holds 5–8 trees/ha, with a tree in a third to
half of its 50 m blocks, and the jungle spreads about 2.6 trees over two thirds of its blocks. Woods
stay walkable: a stand fills at most a fifth of its 6 m cells and a copse's core at most two
fifths, and meadows open between them.

| Zone | Forest | Groves | Lone trees | Giants |
| :--- | :--- | :--- | :--- | :--- |
| temperate | oak, birch | oak, birch | oak | 5 % up to 1.35× |
| autumn | maple, oak | maple, birch | maple, oak | 4 % up to 1.3× |
| plains | (rare) oak, birch | oak, birch | oak | 3 % |
| savanna | none | acacia | acacia | 2 % |
| rainforest | jungle, palm | jungle, palm | jungle, palm | 12 % up to 1.6× |
| jungle | jungle, palm | jungle, palm | palm, jungle | 8 % up to 1.45× |
| wetland | birch, dead | birch, dead | dead, birch | none |
| desert | none | none (cactus) | cactus, dead | none |
| badlands | none | acacia, dead | dead, acacia | none |
| volcanic | none | dead, pine | dead, pine | none |
| mountain | pine, birch | pine, birch | pine, dead | 3 % |
| alpine | (sparse) pine | pine | pine, dead | none |
| tundra | (sparse) pine | pine, dead | dead, pine | none |
| taiga | pine, birch | pine, birch | pine, dead | 3 % |
| highland | pine, birch | pine, birch | pine, birch | none |
| sacred | oak, jungle | oak, birch | oak, jungle | 28 % up to 2.2×, palettes |
| lakes | pine, birch | birch, oak | oak, pine | 3 % |

## The air

`sky/air.ts` `biomeAir` blends each region's `AIR` (`biomes.ts`): a haze multiplier, mist lying in
the low ground, and a tint. It runs once a frame at the camera, and fades out between 150 and 900 m
over the ground. The day-night sky (`gpu/sky.ts`) multiplies its haze by it, adds the mist, and
tints the in-scatter; the mist's height is measured from the local water.

## Twins and invariants

- **One source:** the CPU twin (`terrainMath.ts`, fp64), the WGSL world function
  (`worldgen.wgsl.ts`) and the WebGL2 port (`landsGlsl.ts`) are all generated from the same tables:
  `biomes.ts` (`LANDFORM`, `LOOK`, `AIR`, `SNOW_ALBEDO`) and `constants.ts` (`F`, `GEO`, `HYDRO`,
  `CLIMATE`, `LANDS`). Changing a number changes all three. The rivers' node graph is computed on
  the CPU only; the GPU reads it from a table round the camera.
- **Measured (seed 1337):**
  - GPU vs CPU elevation: at most 33 mm over 64 × 64 probes at each of the 40 `totk` views, water
    levels equal where wet but for one shoreline cell whose ground lies within 0.3 mm of the sea
    (`bun bench/webgpu.ts landprobe --config=totk`);
  - GLSL vs CPU: 1.6 mm at 3 km, 6.1 mm at 12 km; chunk seams ≤ 0.5 mm (`bun bench.ts
    --suites=parity`);
  - forest placement, GPU vs CPU: 99.999–100 % of plants agree, none of another species, none
    moved (`parity --config=totk`).
- **Tests:**
  - `tests/terrainMath.test.ts`: the survey "The lands" checks that every biome occurs, with sea,
    inland water, ice and level lakes; also climate, regions, dunes, and the GLSL contract
    generated from `F`;
  - `tests/forest.test.ts`: zone cover and density (set from TotK's), species, walkable woods,
    sacred giants and palettes, the far canopy predicting the trees within their own spread, grids
    that are multiples of the cell, a ±16 km survey;
  - `tests/hydro.test.ts`: every river drains to the sea or a lake, never climbs, grows downstream
    and stands below its banks; lakes at sinks and lakes on rivers are level, with their shores
    above their water all round;
  - `tests/hero.test.ts`: water and ice under the hero.

## Cost

GPU time per frame at 1600×900 over the 40 `totk` views (every biome from the ground and from the
air, plus a river, a lake on a river, a lake at a sink, a waterfall, a beach and a cliffed coast, a
volcano and its crater, a pond and a marsh), fastest of 4 repetitions per view: Medium 1.5–6.6 ms
(median 2.9), High 2.4–5.5 ms (median 3.8). Single views vary by ±1–2 ms between runs. On the CPU
one world evaluation takes 7.4 µs, and the rivers' node table ~320 ms the first time and 32 ms per
re-centre (measured with the machine loaded). The clipmap's streaming costs 0.44 ms a frame walking and 2.2 ms at 200 m/s; a full
rebuild takes ~200 ms of generation and adds ~385 ms to its frame. The WebGL2 fallback, which
ray-marches the world per pixel, is much slower.

`bun bench/webgpu.ts stills --config=totk --frames=all --variants=game-med,game-high --timing=1
--noshots=1 --warmup=1 --perframe=1 --res=1600x900 --reps=4`

## Tuning

- A biome's shape, look or air: its row in `biomes.ts`.
- Where regions form and how large they are: `LANDS`, `CLIMATE` and `F` in `constants.ts` (keep
  frequencies dyadic: n / 1048576).
- Ranges, massifs, tablelands, basins and cliffs: `GEO`. Rivers, valleys, falls, lakes and ponds:
  `HYDRO`.
- Trees: `RULES` in `forest/placement.ts` (grids must stay multiples of the 6 m cell).
- The spawn (`SPAWN`) is in the open fields, facing up a gentle rise toward the Crown Eyrie 2.6 km
  ahead.
  `tests/eyries.test.ts` needs the Crown within 5 km of it.
