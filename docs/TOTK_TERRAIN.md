# A Tears of the Kingdom–style world: TotK measured, the audit, and the generator

The surface world below the sky islands is generated through a hierarchy of geographic scales:
mountain countries first, then the rivers they shed, the valleys and lakes those carve, and the
regions, forests and water that follow. The aim is the feel of
*The Legend of Zelda: Tears of the Kingdom*'s Hyrule at TotK's own scale. It copies none of its map
or assets, and it stays infinite, deterministic and seamless.

Every number below that describes TotK was **measured from TotK's own game data** (section 1), not
recalled from memory. Every number that describes this world was measured the same way on the
generator (section 2).

A view of every biome and water feature, rendered by the game's renderer at High:
[`lands-contact-sheet.jpg`](lands-contact-sheet.jpg). The biome reference (look, trees, air) is
[`LANDS.md`](LANDS.md).

## 1. TotK's land, measured from its game data

### Method and sources

- [TotK Object Map](https://objmap-totk.zeldamods.org) by the ZeldaMods community (open source:
  [objmap-totk](https://github.com/zeldamods/objmap-totk), [radar-totk](https://github.com/zeldamods/radar-totk)).
  - **Actors.** Its server (`radar-totk.zeldamods.org/objs/MainField/Surface?q=…`) lists every actor
    placed on the surface, with its position and region. Its game-file exports provide the rest.
  - **Field areas.** `ecosystem/Ground.json` holds the surface's 108 field areas as polygons.
  - **Climates.** `area_data.json` gives each area's climate, and `climate_data.json` gives each
    climate's temperatures, moisture and weather.
  - **Markers and map.** `map_summary/MainField/static.json` holds the named places, and the map
    texture comes as `maptex` tiles at up to 2 px per metre.
- **Units.** The internal unit is taken as the metre, as BotW modding uses it. The terrain format
  ([TSCB](https://zeldamods.org/wiki/TSCB)) scales its tiles by 500 and its heights over 0–800.
- **Heights are sampled, not read off a height map.** Elevations come from the 65,825 rocks, cliff
  pieces, korok stones, ruins and trees on the surface. They describe where things stand, which
  leans toward land that has things on it.
- **Water is read off the map texture.** Each map pixel is classed as water by its colour (blue at
  least red, green above red) at 2 m per pixel. The check: real water bodies hold no trees or rocks,
  and 134 of them hold none.

### Scale and the sea

- **The surface is 10 × 8 km, 80 km².** The field areas span x −5,000…5,000 and z −4,000…4,000.
- **Land is 73 km².** The sea covers about 6–7 km² (8 %), along the east and south coasts.
- **Sea level is at about 105** (internal height). The 429 beach palms stand no lower than 105.5–106.1,
  and beach rocks cluster just above that.
- **Coasts.** The game's internal place names count 14 beaches and 8 bays; it also has sea cliffs.
  Objects on the sea floor near shore lie 35–66 m below the sea's level.

### Relief (heights above the sea)

Heights in metres above the sea, from the object samples:

| Region | 10th percentile | Median | 90th percentile | Highest sample |
| :--- | ---: | ---: | ---: | ---: |
| Hyrule Field (Central Hyrule) | 12 | 29 | 105 | 312 |
| Lanayru Wetlands | 10 | 16 | 57 | 136 |
| Faron grassland | 6 | 23 | 86 | 196 |
| Gerudo Desert | 10 | 32 | 80 | 286 |
| West / East Necluda | 12 / 9 | 48 / 118 | 180 / 222 | 437 / 394 |
| Hyrule Hill | 95 | 150 | 260 | 396 |
| Akkala (plateau / outback) | 74 / 36 | 184 / 152 | 290 / 233 | 433 / 274 |
| Great Hyrule Forest | 104 | 175 | 182 | 388 |
| Tabantha Frontier | 31 | 141 | 305 | 429 |
| Lanayru (Zora's lands) | 39 | 143 | 371 | 538 |
| Hebra Mountains | 133 | 244 | 452 | 727 |
| Gerudo Highlands | 50 | 226 | 500 | 694 |
| Mount Lanayru | 166 | 297 | 421 | 632 |
| Eldin | 71 | 330 | 502 | 607 |
| Death Mountain | 522 | 528 | 817 | 866 |

So in TotK:
- **Fields lie 10–100 m above the sea** (Hyrule Field's median is about 30 m).
- **Plateaus and uplands stand 140–250 m.**
- **The great mountains top out at 630–730 m**, and Death Mountain at about 870 m. A community
  readout of the in-game coordinates puts its summit at
  [896](https://lr.ggtyler.dev/r/zelda/comments/146bbn9/totk_a_list_of_interesting_heightelevation_data),
  which is consistent once the uncertainty of where its highest sample sits is allowed for.

### Regions

- **Size.** TotK has 19 climates over its 108 surface field areas. Grouped as below, they form 17 patches
  (over 0.25 km²) on 73 km² of land: median 3.1 km², mean 4.3 km², the largest Hyrule Field at
  15 km². A region is about 1–4 km across.
- **Shares of the land** by climate family:

| Family | Share | Family | Share |
| :--- | ---: | :--- | ---: |
| plains, wetlands, Lake Hylia | 21 % | Gerudo Desert | 7 % |
| Necluda, Akkala's woods | 15 % | Gerudo canyon | 4 % |
| snowy mountains | 14 % | Gerudo plateau (baobabs) | 3 % |
| Eldin (volcanic) | 11 % | Lurelin coast, Faron jungle | 5 % |
| Tabantha (cold steppe) | 8 % | Zora conifers | 4 % |
| Great Hyrule Forest and north | 8 % | | |

- **Snow comes from the climate, not from height.** TotK's climates give temperature by altitude
  band, so snow belongs to the cold climates: Hebra, Mount Lanayru's frost and the Gerudo Highlands.
  Death Mountain (Eldin, 18–39 °C) carries none.

### Trees

TotK places 25,163 tree actors on the surface, of 130 kinds. Per hectare of land:

| Region | Trees/ha | Blocks with a tree | Trees per such block | Blocks of ≥ 6 trees | /ha in those | Nearest neighbour (median) | Mostly |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Hyrule Field | 5.5 | 32 % | 4.3 | 8 % | 40 | 9.6 m | broadleaf, low broadleaf |
| Hyrule Hill | 6.2 | 37 % | 4.2 | 9 % | 43 | — | broadleaf |
| West Necluda | 8.9 | 36 % | 6.2 | 12 % | 56 | 7.9 m | broadleaf |
| East Necluda | 4.1 | 27 % | 3.9 | 5 % | 46 | — | broadleaf |
| Akkala outback / plateau | 9.5 / 5.3 | 39 / 27 % | 6.1 / 5.0 | 16 / 8 % | 44 / 46 | 7.2 m | maple, low broadleaf |
| Great Hyrule Forest | 7.5 | 40 % | 4.7 | 12 % | 37 | 11.7 m | ghost trees, conifer |
| Zora's lands (two areas) | 4.3–8.3 | 31–48 % | 3.5–4.3 | 6–13 % | 37–38 | 7.5 m | conifer |
| Lanayru Wetlands | 5.5 | 47 % | 2.9 | 6 % | 32 | — | broadleaf |
| Faron (jungle) | 7.2–7.5 | 68–70 % | 2.7 | 2–4 % | 25–29 | 17 m | jungle palm, tropical |
| Hebra Mountains | 2.5 | 27 % | 2.3 | 2 % | 29 | 12.6 m | conifer, birch, snowberry |
| Mount Lanayru | 3.3 | 21 % | 3.9 | 6 % | 37 | 9.4 m | conifer |
| Tabantha Frontier | 2.1 | 12 % | 4.3 | 3 % | 46 | 7.8 m | conifer |
| Gerudo Highlands (frost) | 1.3 | 16 % | 2.0 | 1 % | 33 | 8.9 m | snowberry, conifer |
| Eldin | 0.6–0.9 | 7–11 % | 2.1 | 0.3 % | — | 12.8 m | burned, dead |
| Gerudo Desert | 0.8–1.0 | 16–17 % | 1.3–1.4 | 0 | — | 31–44 m | small cacti, palms |
| Gerudo canyon / plateau | 0.16 / 0.24 | 3.5 / 5.4 % | 1.1 | 0 | — | 45–60 m | baobab, dead |

"Blocks" are 50 m squares of land. TotK's land as a whole carries 3.4 trees/ha, with a tree in 23 %
of its blocks.

What this shows:
- **Woods are clumps.** Even the densest stands carry 30–56 trees/ha, with trunks 7–10 m from their
  nearest neighbour.
- **Hyrule Field is copses**, about 4 trees to each block that has any, not scattered singles.
- **Faron's jungle is no closed canopy.** Its large trees are spread evenly over two thirds of the land.

### Water

From the map texture:
- **Inland water is 4.8 km²**, about 6.5 % of the land. The sea component is 6.0 km².
- **Rivers.** There are 43 km of channel, 0.6 km per km² of land. Width along the channels:
  10th / 25th / 50th / 75th / 90th percentile **4 / 8 / 21 / 41 / 64 m**.
- **Named rivers**, at their widest within 120 m of their map markers:
  - Katura 43–104 m, Ronron 52–92 m and Gasema 101–140 m;
  - the Hylia ~137 m;
  - Yamari 32–80 m, Susono ~63 m, Floria 23–63 m and Jiu 20–53 m;
  - Kazuryu 40 m, Gera 37 m and Yarimo 16 m.
- **Lakes.** There are nine compact lakes of 2 ha or more: median 3.7 ha (about 216 m across). The
  largest is **Lake Hylia, about 0.3–0.5 km²** (its water reads as two bodies split by its bridge and
  islands, each 0.5–0.8 km across). The game's internal names count 59 surface lakes, most of them
  small and many of them widenings of rivers.
- **Ponds.** There are 181 under 2 ha, **2.5 per km² of land**: median 760 m² (about 31 m across),
  90 % under 4,550 m² (about 76 m across).
- **Named water.** Besides the lakes, the internal names count 11 ponds, 14 rivers and three falls
  (Floria, Minakkare and Pupukore Falls).
- **Depths are not in the data.** Objects on Lake Hylia's bed lie 15–27 m below the sea's level, so
  the largest lake is tens of metres deep. River and pond depths could not be measured, so the depths
  here (section 3) are this world's own choices.

### Design principles

From the developers' talk, as reported by GMTK's
[reconstruction](https://gmtk.substack.com/p/how-nintendo-solved-zeldas-open-world),
[80.lv](https://80.lv/articles/the-design-secrets-of-breath-of-the-wild/?amp=1) and
[Gigazine](https://gigazine.net/gsc_news/en/20171004-breath-of-the-wild-design-secret), and from
Game Developer's analysis of
["gravity"](https://gamedeveloper.com/design/breath-of-the-wild-open-world-analysis-gravity-to-go-forward):
- **The triangle rule.** Big mountains are landmarks, hills route the player, small rises hide what
  lies behind them.
- **Gravity.** A visible landmark always pulls the player on.
- **Readable at every distance.** Silhouettes far off, forest edges and rivers in the middle
  distance, grass and shoreline up close.

Generation techniques used:
- flow maps for moving water
  ([Vlachos, SIGGRAPH 2010](https://www.advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf));
- Rune Skovbo Johansen's
  [erosion filter](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html)
  (MPL-2.0, ported from its WGSL port in `bevy_erosion_filter`).

## 2. The audit: TotK against this world, before and after

The same measurements, run on this world. "Before" is the generator before this audit (the first TotK
pass); "after" is now, after the fourth pass (the landforms, below). Seeds 1337, 7 and 2024; ranges
span them.
- **Water** is read as TotK's map was (a wet mask at 2 m per pixel, bodies of 200 m² and more), over
  16 windows of 5 × 4 km: 288 km² of land, four times TotK's map.
- **Trees** are counted on 500 random 50 m blocks of dry land per region and seed, by the region
  that dominates each block, as TotK's were counted per field area.
- **Regions and heights** are sampled over 120 km squares.

| Quantity | TotK | Before | After |
| :--- | :--- | :--- | :--- |
| Region patches (median, mean, largest) | 3.1, 4.3, 15 km² | 4.5, 14, 274 km² | 2.1–2.7, 5.0–5.9, 41–67 km² |
| Patches per 73 km² of land | 17 | 5.2 | 12–15 |
| Plains + wetland + lakes share | 21 % | 17 % | 20–25 % |
| Forest (temperate, autumn) share | 15 % | 12 % | 13–17 % |
| Mountains + highland + tundra share | 22 % | 32 % | 20–28 % |
| Volcanic share | 11 % | 1.6 % | 5.6–6.8 % |
| Desert / badlands share | 7 / 4 % | 3.8 / 10 % | 6–8 / 5–7 % |
| Ancient (sacred) forest share | ~5 % | 1 % | 2–3 % |
| Field heights (median above the sea) | 16–48 m | 116 m | 60 m |
| Mountains (median, highest) | 244–297, 630–730 m | 635, 1,220 m | 312–327, 648–679 m |
| Plateaus (median) | 141–184 m | 413 m | 250 m |
| Volcano summit | ~870–896 m | 1,130 m | ~910 m |
| Snow on the volcano, or on its trees | none | yes | none |
| Temperate woods, trees/ha | 4.1–9.5 | 21–23 | 5.9–7.5 |
| Temperate woods: blocks holding a tree, trees in each | 27–39 %, 3.9–6.2 | — | 34–42 %, 4.3–4.9 |
| Conifer woods (taiga), trees/ha | 4.3–8.3 | 24–36 | 5.2–6.6 |
| Mountains, trees/ha | 2.5–3.3 | 6.5–21 | 2.9–4.0 |
| Plains: trees/ha, per occupied block | 5.5, 4.3 | 2.3–4.5, 1.3–2.7 | 5.0–5.9, 3.8–4.2 |
| Jungle and rainforest, trees/ha | 7.2–7.5 | 10–30 | 6.8–7.8 |
| Jungle: blocks holding a tree, dense blocks | 68–70 %, 2–4 % | — | 64–67 %, 4–6 % |
| Ancient forest, trees/ha | 7.5 (GHF) | 46–52 | 6.9–7.5 |
| Marsh (wetland), trees/ha | 5.5 | — | 5.9–7.4 |
| Desert / canyon / savanna, trees/ha | 0.8–1.0 / 0.16 / 0.24 | 2.1–3.3 / 0.2–0.7 / 1.5–2.7 | 0.7–1.0 / 0.29–0.38 / 0.42–0.51 |
| Volcanic, trees/ha (per occupied block) | 0.6–0.9 (2.1) | — | 0.73–0.98 (1.5–1.6) |
| Snowy steppe (tundra), trees/ha | 1.3–2.1 | 0.2 | 1.0–1.3 |
| Trunk spacing in woods (nearest neighbour) | 7.5–12.6 m | 6.0–6.4 m | 7.7–9.6 m |
| River width, 10/25/50/75/90th percentile | 4 / 8 / 21 / 41 / 64 m | 4 / 5 / 5–8 / 8–11 / 9–12 m | 16 / 24 / 40 / 56 / 80 m (wider by request) |
| Channel length per km² of land | 0.6 km | 0.27–0.55 km | 0.21 km (rivers only below lakes) |
| Inland water share of the land | ~6.5 % | 0.2–0.5 % | 4.0 % |
| Lakes: size | 160 m–0.8 km across (median ~216 m) | sinks only, up to ~1.9 km | 100–600 m on rivers, 320–840 m at their heads, ≤ 900 m at sinks (median 380–445 m) |
| Lakes per km² of land | a compact one per ~8 km², more on rivers | at sinks only, a few | one per 14–25 km² of the map |
| Ponds per km² of land | 2.5 | 0–0.6 | 2.2 |
| Pond size: median, 90th percentile (across) | 31 m, 76 m | 39–69 m, — | 32 m, 82 m |
| Sea share of the map | ~8 % | 2–8 % | 7.5 % |

### Bugs and gaps the audit found

The audit turned up eight faults beyond the miscalibrations in the table:

1. **Shore trees everywhere.** The shore band measured height above the water table, which on dry
   land always lies exactly 3 m down, inside the band. So "shore" palms and birches grew on 5–25 % of
   all cells: the palm-filled desert canyons. Shores now use the level of the water actually beside
   a point (`lastShore`: a river's bank, a lake, a pond, the coast, a marsh) on both CPU and GPU.
2. **A snow-capped volcano.** A volcano in a cold region wore snow; TotK's Death Mountain is hot.
   The volcanic look's frost of −40 °C raises its snow line out of reach.
3. **Plaster cliffs.** Temperate rock was 0.25–0.31 linear albedo, near-white on sunlit cliffs. It is
   now 0.13–0.2 warm grey, closer to TotK's rock.
4. **Rivers too narrow.** The width law is now TotK's measured distribution (below).
5. **Lakes too big and too rare.** Basin lakes reached ~1.9 km across, against TotK's ~0.8 km at most.
   Rivers also never widened into lakes, though most of TotK's lakes are on rivers.
6. **Ponds too few and too large.**
7. **Biomes far too large**, with mountains swallowing a quarter to half of the land.
8. **The world too high:** fields twice to four times higher than TotK's, mountains half again as
   tall.

A second, skeptical pass re-measured everything over far more land (16 windows, 288 km² of land, on
three seeds, where the first pass read four windows), looked at every view again, and found ten more:

9. **A seam round every ancient forest.** A grove's region was cut at a circle in the world plane
   while its fade was measured in the warped plane. At 68 of 81 groves its weight jumped from 1 to 0
   across that circle: a step in the land's shape, its look and its trees. The groves' lattice now
   lies in the warped plane, as the volcanoes' does, so each fades out inside its own cell (the
   steepest change is now 0.03 per metre).
10. **A seam round every volcano.** A volcano's region faded out to 1.05 of its radius but was cut at
    its radius, a 3 % step in its weight. Beside tall relief (karst towers, mesas) that made steps of
    the ground of up to 15 m. The fade now ends at the radius.
11. **Woods of singletons.** Lone trees put a tree in 57–63 % of the temperate woods' 50 m blocks,
    one or two in each, against TotK's 27–39 % with four to six. TotK's trees stand in clumps. Lone
    trees are now 2–4 times rarer, and stands and copses carry the trees.
12. **A jungle of thickets.** TotK's Faron spreads about 2.7 trees over 70 % of its blocks, with 2–4 %
    of them dense. Ours had 16–26 % dense blocks and 9–13.5 trees/ha. The jungle's stands now cover
    most of the land at a third of the fill. The marsh (10–11 trees/ha against TotK's 5.5) and the
    savanna were thinned too.
13. **Ponds without a tail.** The pond law was documented as skewed small but drew uniform radii, and
    TotK's larger ponds (its 90th percentile, 76 m across) were missing. Radii are now 14 + 50·u³ m.
14. **Rivers a quarter too narrow.** Pooled, the widths read 4 / 8 / 16 / 29 / 51 m against TotK's
    4 / 8 / 21 / 41 / 64 m; the first pass's four windows had happened to hold wide rivers.
15. **Half the lakes.** Lakes on rivers at 45 % of the eligible nodes gave one per ~21 km² of land.
    TotK has a compact lake per ~8 km², and more where rivers widen. Now 90 %.
16. **Volcanic land at half TotK's share, and volcanoes in snowfields.** Volcanic land was 3.7–5.1 %
    against Eldin's 11 % (the first pass's 6 % had been measured before a later change to the
    volcanoes' cells). Every 8 km cell now holds a candidate, with regions to the cell's limit. But
    then a tenth of the volcanoes rose out of snowfields, their flanks dotted with snowy pines, and
    the tallest summit reached 943 m against Death Mountain's ~870–896 m. TotK's volcano stands in
    hot country (Eldin, 18–39 °C) among burned trees. So volcanoes now need land of at least 4 °C at
    sea level, cones are lower (summits to ~900 m), the volcanic zone keeps its snags up to the tree
    line, and trees carry snow by the ground's own snow line, the regions' frost included.
17. **A marsh's rim switched on by a 50 m drop.** The rim was raised over a marsh's pools by
    subtracting 50 m outside a narrow band of the marsh's weight. That turned the twins' rounding
    into ground differences of up to 0.29 m beside a lake. The rim is now raised in proportion to
    that weight, and the GPU and the CPU agree to 33 mm.
18. **A stale parameter table.** Eleven rows of the table on the project page still showed values
    from before the TotK pass: lowland 28 + 120 + 34 m, crests 980 m, climate fields of 9.4 and
    5.2 km, volcano and grove cells of 16 km, a half-width river law. They now match the code, as
    do the code comments that still gave the climate fields as 16 and 8 km.

A third pass followed a player's screenshots of seven places, and fixed what they showed:

19. **Thin streams.** Every valley that drained enough land showed a river, so most rivers were
    streams a few metres wide. Rivers now show only below a lake. Each river system starts at a
    reservoir, a lake of up to 840 m across at its head, and a valley with no lake above it is dry.
    A river leaves its lake 14 m wide and widens to 180 m, in a channel 4 m deep at the lake and
    deeper downstream (3 + √acc m). That is wider than TotK's rivers, by the player's request, and
    there are fewer of them (0.21 km per km² of land against 0.6).
20. **Ponds in pits.** A pond's level was set from the land's broad shape, so in hill or mesa country
    a pond sank a shaft down to its water. Ponds are now judged on the land itself, at their centre
    and at three points round their rim, and lie only where it is flat.
21. **Shallow water.** Ponds are now 4–8 m deep (they were 1–2.5 m), lakes on rivers 9.5–27 m and
    reservoirs 12–35 m. The water absorbs more light per metre (red first), so deep water reads
    deep: a third of the light reaches a bed 8 m down.
22. **Lakes over falling land.** Some lakes stood over land that fell away outside their shore: walls
    of water. A lake now needs land above its level all round it, out to 1.28 radii, and clear of
    other valleys. The channel a shore is judged by is the nearest one that shows. Of 215 lakes, 24
    rays each, one ray meets a drop, and that is a waterfall at a lake's outlet.
23. **Sine-wave mountains.** There were two causes.
    - The gully filter's fade target (+1 on peaks) lifted every gentle crest by up to ~150 m but not
      its steep flanks. That stood walls along the ridges, which the gullies cut into teeth.
    - The gullies were laid out in the warped plane and ran down the smooth uplift alone, so over
      kilometres they lay side by side: a comb of even crests, each of the four octaves as steep as
      the mountains themselves.

    The fade target is gone. The gullies now run down the land itself (its uplift, hills and ridges,
    sampled 128 m apart) and are laid out in the world, so they branch down each flank. Re-measured,
    the mountains then held more woods than TotK's (3.5–5.1 trees/ha against 2.5–3.3), so their
    warm foot now begins at 22–26 °C: most mountain slopes carry montane pines, 2.9–4.0 trees/ha.
24. **Holes in the desert mountains.** The "anomaly" features (Zonai-style terraces, spires and
    spirals) cut pits and rings into canyon country. They are removed.
25. **Dry valleys as trenches.** A dry valley kept a channel's width with no depth, and its falls'
    steps. Where that channel reached past the valley's floor it stood a rock riser 3–4 m tall, and
    each fall drew a scarp across the valley: pale strips and lines that read as dry rivers. A dry
    valley now has neither. Every valley's walls also leave the floor at the floor's own slope and
    steepen over about 12 m (`HYDRO.wallRound`): a shallow valley is a grassy swale.

A fourth pass followed a player's screenshots of five more places: cliffs, karst towers and mountain
flanks that did not read as TotK's. It went deeper, not bigger: no new regions or features, the
existing landforms done properly, to a form brief ([`FORMS.md`](FORMS.md)) with one page per
landform. Before anything changed, each place was dumped from the CPU twin: the height and each of
its terms on its own (uplift, hills, ridges, karst, mesas, gullies, valleys, terraces), over
2 × 2 km at 0.5 m, with hillshade, slope, curvature, geomorphons and spectra. Each artifact was then
traced to the layer that made it: the height function (shape), the clipmap and its normals
(sampling), or the material and its light (surface). Seed 1337 (the default; the HUD does not show
the seed), at 12:00. Place 3's x was hidden by the minimap and was re-read as 1849.

26. **A comb of teeth in the jungle karst** (place 1, shape). The karst towers were round domes, and
    the river valleys' soft minimum (`RF.R`) clipped them where a valley crossed: the valleys cut
    80 m into land the karst had raised 83 m (118 m at most). Along the terrace bands that left
    scalloped cusps 150–300 m apart, and the gully comb (27) and the value-noise ridges (32) added
    finer teeth. The screenshot's cracked bands and finest teeth are not in the height at 0.5 m, and
    this build has no cracked-rock material, so they were not reproduced. Towers are now prisms
    (30), each under a ceiling that a river's floor puts over them: from 4 m back from the floor's
    edge it rises at the talus's 35° for 8 m, then at 72° over a tower's wall and cap (`KARST_BANK`,
    `KARST_TALUS`, `KARST_TOE`, `KARST_CEIL`; 34). A tower meets the water in a wall over a short
    apron, not a plate tilted by the valley's side. Walls whose contour is a comb fell from 12 % to
    0 %. Small wedges remain where a tower stands in a valley.
27. **Corrugated mountain flanks** (place 2, shape). The gully filter cut gentle land as hard as steep
    land, each octave adding about 0.25 to the slope, at an even spacing. On a smooth flank its
    stripes lay side by side as regular dark corrugations, and on the hill beside it as ripples.
    Gullies now fade in over 6–24° and out over 40–55° (`GULLY_SLOPE`), so walls stay walls and only
    aprons and flanks gully, and each cell jitters its stripes' spacing by ±40 %. At place 1 the
    gentle land's comb share fell from 38 % to 3 %.
28. **Stair-stepped banks and pond edges** (place 3, sampling). The height had no steps. The coarse
    clipmap rings left out every cell with one wet corner, so beyond the finest rings a shore
    stepped along the lattice, 8–32 m at a time. Every ring now draws water by one rule
    (`waterCellDrawn` in `rings.ts`, and its twin in `terrain.wgsl.ts`): a coarse cell with one wet
    corner is left out only where a dry corner lies more than an eighth of the cell's width under
    the water, where the sheet would dive into a cliff. The share of place 3's shore on the lattice
    fell from 24–29 % to 0 %, and a test fails on the old rule.
29. **A toothed rim over a melted sea cliff** (place 4, shape). The coast rose through a U-shaped
    ramp (`COAST_RAMP`), the land's full detail ran out over the rim, and the land blend (`_land`)
    softened the face. A cliffed coast now has a section (`SEA_CLIFF`): a wave-cut platform 1 m
    under the sea falling seaward at 3°, a talus apron of 34° up to 26 m out, an 80° wall from a
    toe 6 m in, and a rim rounded over 1.5 m. The rim's run (60° to 15°) fell from 12.75 m to
    0.5 m, and the wall stands at 80.6°.
30. **Karst towers as smooth domes with stretched rock** (place 5, shape and surface). Each tower was
    a radial falloff, H·(1 − q^3.5): a dome with no rim. Its rock was drawn in the ground's plane, so
    on its sides the texture stretched 2.54–7.45×. A tower is now a prism (`karstTowers`):
    - seven faces, turned and pushed in or out by the cell's hash, the edges between them rounded
      over 12 % of the width;
    - a cap that falls 12 % of its radius to the rim;
    - a wall of 74–81° down to a foot at 12–20 % of the height, and a talus apron of 32–37° below;
    - heights of 50–220 m, and a height over width of 0.7–1.5 at the wall's foot;
    - a 300 m lattice with a tower in 7 cells of 8: wide towers merge on a shared apron
      (*fengcong*), and narrow ones stand alone on the plain (*fenglin*);
    - laid out in the world, not in the domain warp's plane as before, where the warp stretched and
      sheared each prism (by up to about 2×, 34). So each tower stands where it did, moved by the
      warp's offset (a median of 478 m round place 5).

    The rim's run fell from 17.25 m to 2.25 m. Rock steeper than about 45° is now drawn by biplanar
    mapping on five planes, with world-space strata, vertical joints 13 and 21 m apart, and edge
    light and cavity dark from the curvature. Its worst stretch is 1.473×.
31. **Mesa walls and canyon walls smeared over 45–70°** (shape; not in the screenshots, found by the
    slope histograms). Each mesa tier was a smoothstep of its plan noise, and a canyon's wall was one
    slope of 45–50°. A tier is now laid out by the distance in from its contour: a caprock band of
    82° over a cliff of 75–79°, then a talus apron down to the bench of the tier below, each tier a
    step harder than the one under it. In mesa country a river's valley cuts flat-lying strata
    28 m ± 30 % thick, each layer's run shared among a cliff, a bench and a talus slope. The mesas'
    region weight is taken by a forward difference whose step grows with distance (max(1 m,
    distance / 32,768), `fdSpan`): at a fixed 1 m, fp32 noise put the GPU's ground 4.6 m off the
    CPU's at 1,000 km. In WGSL its two samples share one call in a loop: each call site inlines its
    own copy of the uplift and the regions, and side by side they made the world shader 1.6× the
    size it was before the pass (§4).
32. **Ridges of value noise.** The ridged octaves now come from simplex noise, each damped by the
    slope already built (`RIDGE_DAMP` 0.25), so the finer octaves add detail on crests and benches
    more than down flanks.
33. **The cost: gentler mountains.** Removing the comb and damping the ridges took about a third of
    the mountains' slope. Over 8 m the mean slope fell from 15.1° to 10.2°, and the share over 30°
    from 10.6 % to 5.7 %. Summits are 648–679 m on the three seeds (640–717 m before) and medians
    312–327 m, within TotK's relief, and fields stay at 55–60 m above the sea. Neither build has
    alpine walls or scree ([`FORMS.md`](FORMS.md) §4.5).
34. **A step at a tower's foot** (found in the golden frames: shape). The first build of this pass
    showed a dark band at the foot of the towers by a river (place 1's near frame). Profiled down the
    view line, the towers' term equalled the river's ceiling there: 1.5 m, then 4.4 m half a metre on.
    The ceiling rose at 72° from 4 m back from the floor's edge, so it cut the apron off in a sheer
    step a few metres tall. It was also steeper than designed. The valleys are cut in the domain
    warp's plane, and the distance to a floor's edge is measured there (`RF.xFloor`). On that apron
    the warp shrank the world's metres about 2× (|∇xFloor| ≈ 1.93), so 72° stood at about 80°. The
    ceiling now:
    - divides that distance by the warp's largest stretch, from its Jacobian in all three twins, so
      it stands in the world's metres;
    - rises at the talus's 35° for its first 8 m, then at 72°;
    - takes the steep part only as far as a tower covers the ground (`_kcov`), so an apron is laid
      back to the talus, never cut.

    Within 30 m of a floor's edge, 31 % of the towers' first 8 m stood over 60° and 28 % over 70°.
    Now 0.1 % and none do, and 86 % lie at 30–50°. Toes on a valley's side stand at 38–47°, steeper
    than the brief's 32–37°: the ceiling caps the towers' term, and the valley's side adds its own
    slope. A test fails on the old ceiling.
35. **Towers melted at their region's edge** (found in the golden frames: shape). Place 5's mid frame
    showed a grassy dome in the foreground: a tower 175 m tall where the jungle's weight was about
    0.65. The towers were scaled by their region's weight, which rises from 0 to 1 over a median of
    214 m (78–751 m, p10–p90), so at the edge every slope was multiplied by the weight and the whole
    tower tilted by the weight's gradient times its height. Across that tower the ground rose in a ramp, 133 m to 222 m
    over 200 m. The towers now stand whole, under a ceiling like a river's (34, `KARST_EDGE`):
    - it starts where the karst's share of the landform is 0.55 (the weight ½), and rises with the
      distance in from there (the share over its gradient's length): at the talus's 35° for 8 m,
      then at 72° where a tower covers the ground;
    - it lifts away by 90 m of distance over shares of 0.65–0.75, so it stands clear of every tower
      where the share's gradient is no longer taken;
    - the gradient is a forward difference over 128 m (`KARST_EDGE_STEP`, or the mesas' growing
      step (31) where that is longer).

    Over the mesas' 1 m step the walls broke the twins' parity. In fp32 the share carries a few
    millionths of noise, which put the GPU twins' gradient up to 0.9 % off at place 5. The distance
    in is the share over that gradient, up to 65 m there, so it moved by up to 0.6 m, and the 3 m/m
    wall moved its height three times that. The WGSL and GLSL twins stood 1.69 m and 1.70 m from the
    CPU twin's at place 5's mid frame, 0.26 m at its near frame and 30–32 mm at places 1 and 2: over
    the player's `GROUND_BIAS` (0.08 m). The two fp32 twins agreed within 40 mm, so the cause was
    precision, not the port. Over 128 m the noise is 128 times smaller. Perturbing the CPU twin's
    gradient by the same relative error predicts 13 mm at the worst point, against 1.71 m over 1 m.
    The walls' slopes hardly move: on the edge test, 15–60° goes from 4.3 % to 3.7 % and over 60°
    stays at 74 %. At 256 m the walls start to blur (8.0 % at 15–60°).

    The same tower is now cut in a wall: a plain at 114–120 m, then 170, 222 and 270 m at 25 m steps
    (2 m/m, 64°). Over 1 m it stood at 3 m/m. The 128 m step takes the share's gradient partway up
    the rise, where it is steeper than at the tower, so this wall lies back from 72°. On tower tops at the edge (the jungle's weight 0.35–0.75, the towers over 40 m), 62 % of
    the slopes over 0.5 m lay at 15–60° and 35 % over 60°; now 4 % and 74 %. The edge's slope modes
    were one smear (trough 0.45) and are separate (0.22). A test fails on the scaled towers. Far out,
    where the towers fade to their mean height, that mean is still scaled by the weight.

What the fourth pass measured, from the dumps and over 160 km squares ([`FORMS.md`](FORMS.md) §3–5):
- **Slope modes.** Tower, mesa and canyon land each now show a talus mode at 32–34° and a cliff mode
  at 75–80°, separate where they were one smear: the lowest bin between them is 0.11 / 0.18 / 0.14
  of the lower peak, against 0.75 / 1.48 / 8.47, and the 45–60° share is 1.3 / 2.0 / 3.0 %, against
  5.8 / 6.5 / 16.7 %. At tower land's edge they were a smear until 35 (0.45) and are now separate
  (0.22).
- **The comb**, along contours over four places and twelve drawn over 80 km: combed walls fell from
  12 % to 0 %, gentle land from 40 % to 24 %, flanks from 37 % to 28 %.
- **Geomorphons.** Spur and hollow cells, the teeth and corrugations, fell by a third to a half at
  places 1, 2 and 5. Footslopes and shoulders rose at place 5.
- **Spectra.** The power spectrum's exponent over 10–200 m rose from 3.54 to 4.92 (place 1), 4.30 to
  4.94 (place 2) and 4.82 to 5.13 (place 5): power moved from steps and rough noise into planes.
- **Stretch.** No face that looks up draws its rock stretched more than 1.473×.
- **Not set.** The tolerances for geomorphon shares and spectra need the reference DEMs, which were
  not fetched. The slope, comb, water and stretch tolerances are met.

## 3. The generator

```
world   continent (sea / coast: beaches, or cliffs ≤ 95 m: an 80° wall over talus and a wave-cut platform)
  │     uplift U: lowland swell · mountain countries with range crests · lone massifs · tablelands
  │               · lake basins · volcanoes
region  drainage graph on a 2,048 m node lattice (CPU, fp64, cached; GPU node table)
  │       each node → steepest lower neighbour (one diagonal per 2 × 2 block: rivers never cross)
  │       exact drainage, levels from the lowest land across each floor, reach profiles, lakes
  │     climate (5.8 km temperature, 3.2 km moisture) + geography → 15 biome regions
chunk   land = min( max(U + detail·masks, river terraces), river valleys, lake bowls )
  │       detail: hills, ridges, karst prisms, dunes, mesa tiers, canyon strata + gullies down
  │       the land's own slope, on flanks only; near rivers and marshes it only rises
  │       ponds in flat meadows, marsh pools in a marsh's heart, a crater's lava lake
pixel   biome look, flow-mapped water, rapids foam, waterfalls, ice, molten lava
```

The code: `src/lib/hydro.ts` (geography, rivers, lakes), `src/lib/terrainMath.ts` (the world
function), `src/lib/erosion.ts` (MPL-2.0), `src/lib/biomes.ts` (per-biome tables),
`src/lib/forest/placement.ts` (trees), and the twins `gpu/worldgen.wgsl.ts`, `gpu/forest.wgsl.ts`
and `components/terrain/landsGlsl.ts`.

### Geography (`GEO`, `LANDS` in `constants.ts`)

- **Lowland:** 6–90 m above the sea (`GEO.low`), plus each region's rolling hills.
- **Mountain countries:** range crests (`GEO.rangeH` 520 m) where a 12.5 km mask allows them. They
  form massifs a few kilometres across, like Hebra or the Gerudo Highlands, not continent-long
  chains. Summits reach about 880 m.
- **Massifs:** lone peaks of 340 m.
- **Tablelands:** 190 m behind cliff walls.
- **Basins:** bowls of 450–900 m radius, 14–34 m deep.
- **Volcanoes:** one candidate per 8 km cell, kept where it stands on land, off the ranges and in
  land of at least 4 °C at sea level: about one volcano per 150 km², in warm country as TotK's is.
  Each has a 1.9–2.29 km region and a cone rising 380–520 m to its crater's rim (summits to ~900 m
  above the sea, as Death Mountain's ~870–896 m), with a lava lake in its crater. Its ash is too hot
  for snow, and so are its trees: snags, a few pines, up to the tree line.
- **Ancient forests:** one candidate per 8 km cell, taking only mild, moist land. Like the volcanoes'
  lattice, theirs lies in the warped plane, so a grove fades out inside its own cell.
- **Coast:** beaches, or cliffs of 45–95 m (`SEA_CLIFF`). From the sea: a wave-cut platform 1 m
  under the sea, falling seaward at 3°; a talus apron of 34°, up to 26 m out; an 80° wall from a toe
  6 m in from the coastline; and a rim rounded over 1.5 m.

### Rivers (`HYDRO`)

- **Routing.** Each node drains to its steepest lower neighbour whose way crosses no ridge, as before.
  Diagonal moves are allowed only along the shorter diagonal of each 2 × 2 block of nodes, so no two
  reaches cross.
- **Levels follow the lowest land across each floor.** Node levels and reach profiles sample the
  uplift at the channel and at both edges of the valley floor. A river runs at the level of its
  floor's lowest land, never on an embankment.
- **Profiles have three stages:**
  1. the land alone (a running minimum, no fall in the first or last sixteenth: `nodePool`);
  2. kept below the valley of the reach leaving the node it joins, so a tributary comes down into
     the valley it meets;
  3. kept below the valleys of the other reaches flowing into that node, so of two siblings the
     higher comes down to the lower.
- **Rivers flow from lakes.** A reach shows only if a lake lies on it or anywhere upstream
  (`fedOf`). Each river system starts at a reservoir: at a node with no lake above it, where a river
  may show in its region (`riverMin`), on land flat enough (its reach falls under 6 %), off a
  volcano, the largest lake that fits, 180–840 m across. It must be held by land above its level
  all round and clear of other valleys, or it shrinks.
- **Width grows along the reach.** Full width is 14 + 20·(acc − 1)^0.9 m, at most 180 m, ±20 %,
  where acc is the drainage at that point (in nodes): the head's, plus the share of the reach run.
  That is 14 m where a river leaves its lake, ~34 m one node down, ~84 m at acc 5 and 180 m from
  acc ~12: wider than TotK's (4 / 8 / 21 / 41 / 64 m), by the player's request. Channel depth is
  3 + √acc m.
- **Dry valleys.** A valley with no lake above it carries no water, and has no channel and no falls:
  its floor (10 + 5·√acc m each side of its line) and its walls. Every valley's walls leave the
  floor at the floor's own slope and steepen to their region's slope over about 12 m
  (`HYDRO.wallRound`), so a shallow valley is a swale, not a trench.
- **The bank cap: no water over a bank.** A river's water may not stand above any other reach's
  surface out to its own far bank. That surface is the other reach's ground, or its water in its
  channel. The bound uses each surface's slope along the line across the channel, never lower than
  that surface's own river level, less its fall across the bank. Reaches meeting at a node are kin:
  their water runs continuously into each other.

### Lakes, ponds and marshes

- **Lakes on rivers**, as in TotK:
  - At half the nodes on flat land below a lake (the reach leaving them falls under 3.5 %), the river
    widens into a lobed lake 100–600 m across, skewed small. With the reservoirs, there is one lake
    per 14–25 km² of the map, a median 380–445 m across.
  - Its level is the river's where it leaves the lake: held by its outlet, so it never spills.
    Lakes form only where that outlet is within 0.5 m of the node's level, or the lake would sit far
    under the river flowing in.
  - Land must stand above its level all round it (out to 1.28 radii), and it must clear other
    valleys, or it would stand over falling land.
  - Its bowl is 9.5–27 m deep (6 m + 7 % of its radius; reservoirs 12–35 m), with a shore rising
    gently into valley walls.
  - The CPU decides each lake; the GPU reads radius and level from the node table.
- **Lakes at sinks:**
  - Their level is just under the lowest ground on a ring round the sink, about 900 m across.
  - The ring is judged on the gentlest valley walls of any region, so no region's valleys can
    breach it.
  - At most 25 m deep.
- **Ponds:**
  - One candidate per 256 m cell, of radius 14 + 50·u³ m for a uniform u: most small, a few large,
    so their water is a median ~32 m across (90 % under ~82 m), as TotK's. They are 4–8 m deep.
  - Only on flat land away from rivers and lakes, judged on the land itself (all its detail) at the
    pond's centre and at three points round its rim: judged on the broad shape alone, a pond in hill
    or mesa country sank a shaft to its water.
  - The pond's own bowl and rim then shape the ground round it, eased back to the land within half a
    radius past its rim (so even the largest stays inside its cell).
- **Marshes:**
  - On flat lowland, with pools 1 m under the smooth land, only in the marsh's heart.
  - Round it, the hummocks stand at least a little above the pools.
  - Next to a marsh or a river, terrain detail may only rise. A hollow at its edge would spill the
    pools or leave the river standing over it.

### Water on screen

- **The coast.** The sea fills only land below its level by the coast. The old blend raised the sea's
  surface up any shore where the coast faded out: the white sheets on sea cliffs.
- **The water mesh.** The water's triangles run level under a dry bank, held just below that bank's
  ground (`waterVertexH`). The ground then hides the water, and its edge is the true shoreline.
  Before, water climbed the bank toward each dry vertex, which drew white "waterfall" sheets on every
  steep shore.
- **Flow.** Ripples ride the current (flow maps), rapids foam, and swimmers drift with the current.
- **Lava** is a molten liquid with drifting crust rafts in craters.

### Forests (`RULES` in `forest/placement.ts`)

Each zone's forest threshold, core fill, groves and lone trees are set from TotK's measurements,
checked per region on 500 random 50 m blocks of dry land on each of three seeds:
- **Trees come in clumps, as TotK's do.** Stands fill 0.1–0.2 of the 6 m cells (28–55 trees/ha,
  trunks 8–10 m apart), groves are copses, and lone trees are few (a 42 m lattice shared by every
  zone, at 5–35 % a cell; 60 % in the jungle's thin woods). A wooded region then holds a tree in a
  third to half of its 50 m blocks, about three to five trees in each, as TotK's do. Before, lone
  trees put a tree in nearly two thirds of the blocks, one or two in each.
- **Plains:** copses of oak, a few lone trees between them.
- **Jungle and rainforest:** thin stands (0.034–0.04 of the cells) over most of the land, so about
  2.6 trees stand in two thirds of the blocks and few blocks are thickets, as in Faron.
- **Sparse zones:** deserts are scattered cacti, the canyons and the savanna nearly bare, volcanic
  ash pairs of snags, the snowy steppe small clumps of conifers.
- **Snow on trees** follows the ground's own snow line, each region's frost included, so the
  tundra's pines are white and a volcano's snags never are.

## 4. Measurements

Seed 1337 unless noted, MacBook Air M1, headless Chrome, WebGPU.

- **Parity:**
  - GPU vs CPU: heights within 46 mm over 64 × 64 probes at each of the 40 views (14 mm but for the
    marsh's hummocks; `landprobe --config=totk`), and within 5 mm at the player's seven places.
    Wet water levels are equal.
  - Tree placement agrees for 100 % of plants, none of another species, none moved
    (`parity --config=totk`).
  - WebGL2 vs CPU: 1.3 mm at 3 km, 3.4 mm at 12 km; chunk seams under 0.5 mm. (With the three gully
    samples inlined side by side, WebGL2 on an Apple GPU put the gullies' gradient metres off,
    though each sample alone was exact: they now run in one loop.)
  - After the fourth pass (`port --config=forms`, SwiftShader, before → after):
    - WGSL vs CPU at the five places, near, mid and far: at most 11.8 mm (place 2, near), against
      21.9 mm before. (The pass's first build had 22.8 mm there; the towers' new ceiling (2, 34)
      halved places 1 and 2 near.) The karst and flank views at mid range fell from 21.9 and
      19.7 mm to 1.8 and 1.6 mm. The sea cliff is now 3.9–5.5 mm, against 0.6–0.9 mm: its 80° wall
      turns fp32's rounding of x and z into height.
    - Away from them: 5.3 mm at 10 km (10.3 before), 52.6 mm at 500 km (44.1) and 172.7 mm at
      1,000 km (155.1), all on coarse levels. Seams are at most 5.9 mm (7.6).
    - Tree placement at places 1–5 near (`parity --config=forms`, frames 0, 3, 6, 9 and 12):
      99.999 % of cells agree. 4 of ~22,400 land plants are on one side only, none is of another
      species and none moved; sky plants agree for all.
    - GLSL 100 km or more from spawn was already metres off before the pass (72 m and 23 m on
      coarse levels at 100 and 250 km; 24 m at 1,000 km fine), and still is (65 m, 12.7 m and
      38 m). The pass did not look into why.
- **Walls.** In 2.6 million samples at 2 m (seed 1337), no water stands more than 0.5 m above
  adjacent dry ground. Round 215 lakes, 24 rays each, one ray meets a drop: a waterfall at an outlet.
- **River continuity.** Visible channels are dry along 0.02 % of their length, with no gap of 16 m or
  more (seed 1337).
- **Frame cost** at 1600×900, fastest of four runs per view, over the 40 views:
  - Medium 1.5–6.6 ms (median 2.9);
  - High 2.4–5.5 ms (median 3.8);
  - about unchanged.

  These are the third pass's numbers. The fourth pass did not re-measure them on an M1 (§5).
- **World shader size** (the fourth pass). SwiftShader, the software GPU these runs use, inlines
  every call: each call site becomes its own copy of the function. Inlined, `landSurface` in the
  world shader was:
  - 633k characters before the pass;
  - 1,020k with the mesas' two region-weight samples side by side. Each sample is a copy of the
    uplift and the regions, and `landDetail` is itself inlined three times, so there were six copies;
  - 859k with the two samples in one loop, the values unchanged.

  The forest's generator (`forestGen`) held six copies of `landSurface` and ten of the coarse
  ground. Inlined, it grew from 4,439k characters to 5,796k with the pass. Building it, SwiftShader
  ran Chrome's GPU process out of its ~16 GiB allocator pool: ~15–16.6 GiB reserved, then a failed
  request of ~0.9–1.2 GiB, and the device was lost. It was not short of the machine's memory, and
  compiling the pipelines one at a time (`--serial-compile`) did not help. Its ground samples now
  share call sites in loops (`plGround`, and one loop for the ten coarse samples), the same
  expressions as before, and placement still agrees with the CPU (above). Inlined it is 1,853k,
  and it builds in 55 s at a peak of ~6 GB. A hardware driver inlines less, but compile time and
  frame cost on an M1 were not re-measured.
- **Streaming** (the game's clipmap):
  - the generator costs 0.44 ms a frame walking and 2.2 ms at 200 m/s;
  - a full rebuild (spawn, teleport) takes ~200 ms of generation and adds ~385 ms to its frame;
  - the third pass's gullies cost 1–5 % more generation than the old ones (an A/B under the same
    load; the frame cost above was not re-measured, the render path being unchanged).
- **CPU:** one world evaluation takes 7.4 µs: points round a pond now share its cell's verdict. The
  river table takes ~320 ms the first time and 32 ms per re-centre (measured with the machine
  loaded). The WebGL2 world function costs 38 ns an evaluation, 8 % more than with the old gullies.
- **Tests:** 288 pass, including the hydrology suite:
  - drainage reaches the sea or a lake;
  - levels never rise downstream;
  - banks stand at the water;
  - basin lakes and lakes on rivers are level, and their shores stand above their water all round;
  - the GPU table matches the CPU.

  The fourth pass adds four detectors (`tests/forms.test.ts`). The first three fail on the build
  before the pass, the fourth on the pass's first build, and all pass now:
  - the comb: a spectral peak along contours on steep cells at place 1;
  - the stair-step: the share of the shore at place 3 that runs along the 8, 16 and 32 m cells'
    edges, where a cell left out cuts the water off (15 % or more by the old rule, under 2 % now);
  - the stretch: the biplanar rock's worst texel stretch over all slopes and aspects (under 1.5×);
  - the toe: the share of the towers' first 8 m by a river that stands over 60° (30 % on the
    pass's first build, under 2 % now; 34). The domes before the pass had gentle feet, so it passes
    on that build too.

## 5. What still differs, and what is not done

- **Fewer, wider rivers than TotK.** Rivers show only below a lake, as the player asked, so many
  valleys are dry: channels run 0.21 km per km² of land against TotK's 0.6. They are also wider
  (median 40 m against 21 m), by the same request. Inland water is 4.0 % of the land against TotK's
  ~6.5 %. Lakes come one per 14–25 km² of the map, against a compact lake per ~8 km² in TotK. The
  bank cap, the outlet rule and the lakes' containment keep lakes only where they are safe, which
  costs some.
- **Volcanic land is 5.6–6.8 % against Eldin's 11 %.** Volcanoes stand in warm land, off the ranges,
  one per ~150 km² against TotK's one in its 73 km² of land, and a volcano's region cannot grow past
  its 8 km cell.
- **Region shares are otherwise near TotK's.** Taiga is commoner, and the largest patches (mountain
  countries of 41–67 km²) exceed TotK's largest (Hyrule Field, 15 km²). This world is infinite, so it
  matches TotK's statistics, not its one map.
- **Nine species against TotK's 130 tree actors.** Oak, birch, maple, pine, jungle, palm, acacia, dead
  and cactus. TotK's ghost trees of the Lost Woods, baobabs, dragon's blood trees and snowberries
  have stand-ins (acacia for the baobab), not models.
- **Depths are this world's choices.** TotK's data holds no water depths. Its one measurable lake bed,
  Lake Hylia's, lies 15–27 m under the sea's level. Here ponds are 4–8 m deep, river channels 4 m
  at a lake's outlet and 6–9 m on big rivers, lakes on rivers 9.5–27 m, reservoirs 12–35 m, and
  lakes at sinks up to 25 m.
- **Hitches.** The first river table (~320 ms) and a full clipmap rebuild (~385 ms) are one-time
  costs that a worker and staggered levels would hide.
- **WebGL2 is slower.** The WebGL2 fallback ray-marches the full world per pixel, so it is correct
  but much slower than the WebGPU clipmap.
- **The fourth pass's rock is WebGPU only.** The biplanar rock, its strata and joints, and the
  curvature light are in the clipmap's shader. The WebGL2 fallback has the new shapes but none of
  that look.
- **Mountains are gentler, not more TotK-like.** The fourth pass took about a third of their slope
  (33). There are still no alpine walls, knife ridges or scree fans like Hebra's or the Dueling
  Peaks'. The ridge page in [`FORMS.md`](FORMS.md) §4.5 says what they would need.
- **The reference DEMs were not fetched.** No tolerance for geomorphon shares or spectra is set
  from USGS 3DEP, swissALTI3D, the EA's lidar or GLO-30, so those two measures show only which way
  the terrain moved ([`FORMS.md`](FORMS.md) §3 and §7). The repository holds no TotK captures
  either. The golden frames sit beside an empty reference slot.
- **A heightfield's limit.** At 10–50 m the heightfield reaches the brief's planes, benches and
  slope modes. It cannot carry wall detail finer than 2–6 m in height, and it cannot undercut: no
  foot notch on a tower, no sea notch, no caprock overhang. [`FORMS.md`](FORMS.md) §6 proposes the
  smallest cliff-mesh layer that would; it is not built.
- **Wedges in the karst.** Where a tower stands in a river's valley, small wedges of its prism are
  left between the ceiling and the valley's side (26).
- **Not measured here:** the frame cost on an M1 at Medium (`bun run bench`), and the blind A/B with
  five or more players. The golden frames and the A/B kit are ready, but the A/B needs people.
- **Not done:** caves, boulder fields, lava that harms, local waves where the hero swims.
