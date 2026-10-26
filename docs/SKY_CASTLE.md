# The sky castles (the land's model isles)

Colossal white-marble and gold gothic citadels float over the land, in two layouts (each cell picks one): one candidate every 6 km cell
(`SKY_MODELS` in `src/lib/constants.ts`, about one every 9 km), its plaza 2600–3000 m up, at a random turn, drawn out
to 40 km. WebGPU only (fly mode climbs to 4.6 km, over the spires). Contact sheets: `docs/sky-castle-contact-sheet.jpg`,
`docs/sky-castle-layouts.jpg` (the two side by side), `docs/sky-castle-colossal.jpg` (at full size, the hero at a portal),
`docs/sky-castle-details.jpg` (the pools, gardens, falls, ivy and textures).

**What the land lends them:** the trees and bushes are the land's forest's (pines for the cypresses, bushes in
moss and blossom round the fountains): a castle records plant spots, and the forest grows them as its sky layer
(`skyPlantAt` in `src/lib/gpu/modelIsles.wgsl.ts`, from the castles near the camera), with its own light, wind,
levels of detail and shadows. The ivy is cards of the land's leaf sprays, lit as its leaves. The pools and fountains
hold the land's water (its ripples, the sky in it, the bed through it); the waterfalls pouring from gold spouts round
each rim are the sky realm's falls and mist.

**The textures** are procedural, per material (each face carries one: `M_*` in `kit.py`), painted in the castle's own
axes and faded out with distance: veined marble laid in great blocks, ashlar stone, hammered gold, the domes' lapis
scales, banner cloth in folds, leaded stained glass, polished floors, the pools' lapis mosaic. The hanging rock is the sky
realm's islands' own: the same underside (skyworld/island.ts bottomAt: cliff, inverted mountain, drip spires, great
spikes) under the castle's outline, painted with its strata, flutes and mossy rim. No texture files.

Every layout is drawn at the sizes below and built 2.5 times over (`SCALE` in `scripts/sky-castle/castle.py`): a castle
is ~1.6 km across, its hanging rock ~0.9 km deep, the cathedral's spire ~1.1 km over its plaza. Only the stairs keep
the hero's size: every flight is cut into steps of ~0.45 m.

**The castle** is built procedurally in Blender from a gothic kit (`scripts/sky-castle/kit.py`: pointed-arch walls,
lancet windows with gold tracery, pinnacles, buttresses, onion and bell domes, balustrades, clustered columns,
banners, angels, cypresses, ivy, stalactites) laid out by `scripts/sky-castle/castle.py`:

- the island: a 630 m plaza (before the scale) on an inverted mountain of hanging rock and gold pendants, its rim an arcade with ivy;
- three terraces, grand stairs, a reflecting pool, a loggia round the edge with a gatehouse, ten rim towers;
- the cathedral: facade with a deep portal, rose windows and twin towers, transepts, an apse, the central spire
  (~450 m) and the golden halo ring behind it; four grand towers with onion domes; two-storey arcades;
  side courts with domed chapels and fountains; the palace behind the apse;
- interiors: the great hall (checkered marble, clustered columns and arcades, chandeliers, banners, the throne),
  the palace gallery, the grand towers' halls.

**The rotunda** (`scripts/sky-castle/rotunda.py`, the same island and kit): round terraces up to a great lapis dome
on two drums of lancets (~330 m), crowned by a horizontal gold halo round its lantern; four long galleries radiating
from it, each ending in a gabled front between domed turrets (the front one the deep portal); a ring of eight round
onion-domed towers, the four between the galleries joined by arcades; quarter gardens with fountains and chapels.
Inside: the round hall (a ring of columns and arcades, a gold-ribbed ceiling round an oculus, a great chandelier, a
gold armillary sphere) and the four galleries.

Rebuild them (≈ 2.5 min each; `public/models/<name>.bin`, ~16 MB gzipped each):

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/sky-castle/build.py -- sky-castle 7 cathedral
/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/sky-castle/build.py -- sky-rotunda 11 rotunda
```

`build.py` makes three levels of detail straight from the kit (not decimated), records the pinnacles and balusters
as instanced props, and bakes into every vertex with Cycles its ambient occlusion and the sky it sees (the
interiors go dark, lit by their glowing glass and candlelight). Floors are baked on a 1.25 m grid in layers; a layer
is marked blocked where something stands 1.1–1.6 m above it (clear of a stair's next tread), so walls, columns and
railings stop the hero; the podiums' walls break where their stairs arrive.
Every surface under a cell is kept (under a dome three dozen stack over the hall's floor).

**Another island from a .glb**: `scripts/bake-sky-model.py -- <in.glb> <name> [across_m]` (decimates, keeps its
texture), then add `<name>` to `SKY_MODELS.names`: each cell picks one of the names.

**The format** (`scripts/skymodel_pack.py`, read by `src/lib/skymodels.ts`, drawn by `src/lib/gpu/modelIsles.ts`):
parts (the exterior's three levels; the interior, drawn within 1.75 km), prototypes and their props in 128 m chunks,
the floors. The shader lights gold as metal (the sky in it, a glint) and gives the halls candlelight as a glow, so
it stays steady by night.

Not done: the isles cast no shadow on the land and take no sun shadows (the bake stands in); the hero passes through
the castles' trees and walks on the pools' beds (no swimming there).
