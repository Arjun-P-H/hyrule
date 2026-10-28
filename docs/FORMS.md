# The form brief: cliffs, towers, mesas, canyons and ridges

This brief says what the surface world's steep landforms should look like and how close they are.
It covers the five landforms the terrain's walls come from: the karst tower, the butte and mesa, the
canyon wall, the sea cliff and the mountain ridge. Each gets one page with:

- a cross-section with its slope angles;
- its height over its width;
- its cap, its wall and its talus apron;
- how it clusters;
- what the references say, what the generator does, and what was measured.

The measurements are taken on the fp64 CPU twin (`src/lib/terrainMath.ts`), which the WGSL and GLSL
twins follow to within the parity tests. "Before" is the build before this pass (commit `609bee2`).
"After" is this pass. The seed is 1337 throughout.
[`TOTK_TERRAIN.md`](TOTK_TERRAIN.md) §2, the fourth pass, gives the cause of each reported artifact
and what changed.

## 1. The form language

TotK's walls read as a few large, clean planes. The breaks of slope between them are crisp. Flat
benches lie between the walls, and fine detail sits only where the form needs it: the strata on a
wall, a talus apron at its foot, a rim. TotK's own landforms were looked at for proportion and look
only: Dueling Peaks, the spire at Rito Village, Gerudo Canyon, Tabantha's bridge canyon and Hebra's
peaks. Nothing of them is in this repository, and no asset was copied. The shapes and numbers below
come from real landforms, through the sources in §2.

That language makes four demands of a heightfield:

1. **Slopes fall into modes.** Each landform has a bench (under 15°), a talus apron at the angle of
   repose (32–37°) and a cliff (70–85°). There should be little between them: a slope histogram should
   show separate modes, not one smear over 45–70°.
2. **The breaks are where the form puts them.** A rim, a foot or a bench edge is a kink. It is rounded
   only as far as the pixel footprint needs to keep it from aliasing.
3. **No comb.** A wall's plan is a few long faces. No row of teeth runs along its contour.
4. **The rock shows no stretch.** No face draws its texture stretched more than 1.5×.

## 2. Sources

The reference elevation models are named here so the tolerances in §3 can be set from them. None
were reachable from the environment this pass was built in, so **no DEM was measured** (§3 says what
that leaves open). `bench/forms/dem.py` measures a DEM tile the same way as this world.

| Landform | Reference | Where to get it |
| :--- | :--- | :--- |
| Butte and mesa | Monument Valley, USGS 3DEP 1 m | [3DEP](https://www.usgs.gov/3d-elevation-program), via [the National Map downloader](https://apps.nationalmap.gov/downloader/) |
| Canyon wall | Canyonlands (Island in the Sky, White Rim), USGS 3DEP 1 m | as above |
| Mountain ridge | Alpine ridges and scree, swissALTI3D 0.5 m | [swisstopo](https://www.swisstopo.admin.ch/en/height-model-swissalti3d) |
| Sea cliff | Seven Sisters, Environment Agency 1 m lidar | [the EA survey portal](https://environment.data.gov.uk/survey) |
| Karst tower | Guilin and Yangshuo, Copernicus GLO-30 | [Copernicus](https://spacedata.copernicus.eu) or [OpenTopography](https://opentopography.org) |

Until those are measured, each page's reference describes the landform from published geology, as
rough figures. These are the figures the generator's parameters were chosen to.

## 3. Tolerances

| Measure | Tolerance | Basis | Status |
| :--- | :--- | :--- | :--- |
| Talus mode | 32–37° (the 5° bin peaks over 30–40°) | The angle of repose of coarse rock debris | **Met**: towers 32°, mesas 34°, canyons 34° |
| Cliff mode | 70–85° (the 5° bin peaks over 65–90°) | Tower walls, sandstone cliffs and chalk faces stand near vertical | **Met**: towers 77°; mesas and canyons peak in the 75–80° bin |
| Separate modes | The lowest 5° bin between the talus and cliff peaks is at most half the lower peak; both are local maxima | This brief, §1.1 | **Met**: troughs 0.10 / 0.18 / 0.14, against 0.75 / 1.48 / 8.47 before |
| The 45–60° gap | Under the talus share, and well under the before's | §1.1 | **Met**: 1.4 / 2.0 / 3.0 %, against 5.8 / 6.5 / 16.7 % |
| Tower height over width | 0.7–1.5 at the wall's foot | Guilin tower karst (§4.1) | **Met** by construction |
| Talus apron | 32–37° | The angle of repose | **Met** by construction on every landform |
| Comb | At place 1, under 15 % of contours on steep cells swing ≥ 20° at 32–160 m, and the 75th percentile is under 18° | §1.3 | **Met** (`tests/forms.test.ts`, which fails on the before) |
| Stair-stepped water | Under 2 % of the shore at place 3 lies on the cell lattice | Water edges follow the ground | **Met**: 24–29 % before, 0 % after |
| Rock stretch | Under 1.5× on every face that looks up | §1.4 | **Met**: 1.473 at worst (2.54–7.45× at place 5 before) |
| Geomorphon class shares | Within the spread of each landform's reference tiles | The DEMs | **Not set** |
| Power-spectrum exponent β, 10–200 m | Within the spread of each landform's reference tiles | The DEMs | **Not set** |

**The open rows.** The geomorphon and spectrum tolerances need the reference DEMs, and they were not
fetched. This pass sets no numbers for them. It reports the before and after values (§5), and §7
gives the procedure.

**Reading β.** This is for interpretation, not a tolerance. A radially averaged 2D power spectrum
falls as k^−β:

- a resolved step (a cliff seen as a jump) gives β ≈ 3;
- large planes meeting at crisp kinks give β ≈ 5;
- a fractal surface gives β = 2 + 2H, between 2 and 4.

A rise in β over 10–200 m means power moved out of steps and rough noise into clean planes. That is
the direction the form language asks for. The value still depends on what the window holds, which is
why the tolerance has to come from reference tiles of the same landform.

## 4. The landforms

Each page's measured slopes come from `bench/forms/slopes.ts`. It takes 12,000 random dry points
where the landform's regions weigh at least 0.6, over a 160 km square. Slopes are central differences
over 0.5 m at a 0.5 m footprint (the nearest clipmap level's). The breaks come from
`bench/forms/breaks.py` on the 2 × 2 km dumps.

### 4.1 Karst tower

```
                 cap: near flat, falls 12 % of its radius to the rim
              ._______________________.
             /                         \   rim rounded over the footprint (≥ 1 m)
            |                           |
            |       wall 74–81°         |   H = 50–220 m
            |                           |
           /|                           |\  foot at 12–20 % of H
      ____/  talus apron 32–37°           \____  plain
           |<-------- W = 2 Rw -------->|
                    H / W = 0.7–1.5
```

- **Reference.** In Guilin and Yangshuo, *fenglin* (peak forest) towers stand alone on an alluvial
  plain. *Fengcong* (peak cluster) towers rise from a shared rock base. Towers are roughly 50–200 m
  high, and their walls stand near vertical. An isolated tower is about as tall as it is wide or
  taller. Water undercuts the foot into notches and caves.
- **Generator** (`karstTowers`).
  - **Plan.** A prism on a seven-sided plan (`KARST_FACES`), its faces turned and pushed in or out
    (0.8–1.2×) by the cell's hash. The edges between faces are rounded over 12 % of the tower's width.
  - **Section.**
    - The cap falls at most 0.12 of its radius to the rim.
    - The wall stands at 74–81° down to a foot at 12–20 % of the height.
    - Below the foot, a talus apron of 32–37° runs out to the ground.
  - **Sizes.** Heights are 50–220 m, and H/W at the wall's foot is 0.7–1.5.
  - **Breaks.** They are rounded over the footprint (`ws = clamp(dp, 1, 24)`), never finer than a
    pixel can show.
  - **Near rivers.** Each tower stands under a ceiling that rises at 72° from 4 m back from the
    river's floor (`KARST_CEIL`, `KARST_BANK`), so it meets the water in a wall, not a plate tilted
    by the valley's side.
- **Clustering.** Towers sit on a 300 m lattice, and 7 cells in 8 hold one. Each centre lies in the
  middle 60 % of its cell. Each point takes the highest tower of the 3 × 3 cells around it. Towers
  wider than their spacing merge into a cluster on a shared apron, like *fengcong*. Narrower ones
  stand alone on the plain, like *fenglin*.
- **Measured.**
  - The wall's foot radius has a median of 61.6 m, and the cap's radius 35.6 m.
  - Face lengths, p25 / p50 / p75: 29.8 / 45.9 / 66.7 m. The form language's few large planes is
    about seven faces of tens of metres each.
  - Rim run (60° → 15°), median:

    | Place | Before | After |
    | :--- | :--- | :--- |
    | 1 | 13.0 m | 1.5 m |
    | 5 | 17.25 m | 2.25 m |

    The wall's median slope is 76.7° at place 1. Before, the radial falloff H·(1 − q^3.5) drew a dome
    with no rim.

| Tower land | Bench < 15° | Talus 28–40° | Gap 45–60° | Cliff > 60° | Modes | Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Before | 64.0 % | 6.1 % | 5.8 % | 12.0 % | 0°, 29°, 37°, 39°, 60°, 72° | smear (0.75) |
| After | 70.3 % | 11.3 % | 1.4 % | 8.3 % | 0°, 32°, 77° | separate (0.10) |

The 5° bins after: 54.3, 11.3, 4.6, 2.7, 2.4, 3.7, **5.0**, 4.0, 2.3, 0.5, 0.5, 0.4, 0.5, 0.7, 2.0,
**3.7**, 1.3, 0.0 %. That is a talus mode at 30–35° and a cliff mode at 75–80°, with 0.4–0.7 % per
bin between them.

### 4.2 Butte and mesa

```
                 caprock 82°, ¼ of the wall
             ._____________________.
             |                     |   cliff 75–79° (harder rock steeper)
             |    tier 60–80 m     |   wall 50–72 % of the tier
            /                       \
           /   talus apron 32–37°    \_____ bench: the cap of the tier below
     _____/                               \____ ...
     (up to three tiers, 210 m; each a step harder than the one under it)
```

- **Reference.** Monument Valley's buttes rise up to about 300 m over the valley floor. Each is
  layered:
  - a slope of Organ Rock Shale at the base;
  - a cliff of De Chelly Sandstone;
  - a resistant caprock (Moenkopi, and the Shinarump at the top).

  The usual rule: a mesa is wider than it is tall, a butte about as tall as it is wide, and a spire
  taller than wide. The cliffs retreat behind talus aprons, and isolated buttes and spires stand
  where the retreat has gone furthest.
- **Generator** (`MESA`, `mesasE`).
  - **Tiers.** Up to three, 80, 70 and 60 m high. Each stands where a plan noise (1.2 km, with a
    480 m outline) passes its threshold.
  - **Wall.** A band of caprock (82°, a quarter of the wall) over a cliff of 75–79°, which takes
    50–72 % of the tier. A talus apron of 32–37° runs from its foot down to the bench of the tier
    below.
  - **Layout.** The wall is laid out by the distance in from the tier's contour, along the plan's
    gradient, refined by a secant step, so walls keep their slopes however fast the noise changes.
    Before this pass, each tier was a smoothstep of the noise, and the wall smeared over 45–70°.
  - **Hardness.** A slow noise sets it, each tier a step (0.25) harder than the one below. Harder
    rock stands in steeper, taller walls over steeper talus.
  - **Buttes.** Toward the region's edge the thresholds rise and the tiers shrink, which leaves buttes
    out on the plain.
- **Clustering.** Mesas are tablelands in badlands regions, and buttes stand at the regions' edges.
- **Height over width.** For a mesa it is well under 1 by construction: a tier of 60–80 m on a plan
  noise of about 1.2 km. **Not measured** for the buttes at the edge.

| Mesa land | Bench | Talus | Gap | Cliff | Modes | Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Before | 74.5 % | 5.1 % | 6.5 % | 4.4 % | 0°, 49° | smear (1.48) |
| After | 75.2 % | 12.8 % | 2.0 % | 4.9 % | 0°, 34° | separate (0.18) |

The cliff's mode stands out in 5° bins (75–80°: 2.6 %, between 0.6 % on either side), but it falls
under the 4 % of peak that the 1° modes count. Mesa land is mostly cap and plain.

### 4.3 Canyon wall

```
     rim ______
               |  cliff 75–79° (40–75 % of a layer's rise; harder rock more)
               |____ bench 3°
                    \
                     \  talus 32–37°
                      \______
                             |  cliff
                             |____ bench
                                  \
                                   \____ valley floor        layers 28 m ± 30 %
```

- **Reference.** In Canyonlands, the walls below Island in the Sky are a staircase:
  - the Wingate Sandstone cliff, roughly 100 m;
  - the ledges and benches of the Kayenta above it;
  - the Chinle and Moenkopi slopes below;
  - the broad White Rim bench below those.

  Hard beds stand as cliffs, soft beds weather to talus slopes, and the top of each hard bed is a
  bench.
- **Generator** (`CANYON`, `canyonLayer`, `canyonRaise`).
  - **Layers.** Where mesas rule, a river valley's wall is cut through flat-lying strata, 28 m ± 30 %
    thick, each with a hashed hardness. They dip gently (12 m over 1.2 km).
  - **Profile.** Each layer's run is shared among a cliff (75–79°), a bench (3°) and a talus slope
    (32–37°), so the wall still climbs the layer over the same run. That pins the three slopes
    wherever the wall's own slope allows them.
  - **Banks.** The raise is never negative, so a bank only rises, and the river's water stays below
    it.
  - **Fade.** The strata fade in over 12 m past the valley floor's edge, and over wall slopes of
    14–24°.
  - **Prefilter.** `canyonRaise` is the exact box filter of the profile over the footprint.
- **Clustering.** Canyons follow the river valleys, only in mesa country (the mesas' region weight,
  through smoothstep 0.35–0.75).

| Canyon land (valley cut ≥ 10 m) | Bench | Talus | Gap | Cliff | Modes | Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Before | 55.4 % | 11.8 % | 16.7 % | 2.5 % | 0°, 7°, 24°, 44°, 46°, 50° | smear (8.47) |
| After | 63.9 % | 18.5 % | 3.0 % | 5.2 % | 0°, 5°, 34°, 77° | separate (0.14) |

The 5° mode is the benches. Before, a canyon wall was one slope of about 45–50°, which is the smear
the brief rules out.

### 4.4 Sea cliff

```
     land ________
                  )  rim rounded over 1.5 m (and the footprint)
                  |
                  |  wall 80°   (cliffs of 45–95 m)
                  |
                   \  talus apron 34°, up to 26 m out where the noise lets it
       ~~~~~~~~~~~~~\_________  wave-cut platform 1 m under the sea, falling seaward at 3°
                    |<-6 m->| wall's toe in from the coastline
```

- **Reference.** The Seven Sisters are near-vertical chalk cliffs, up to 77 m at Haven Brow. A
  wave-cut chalk platform lies at their foot, bared at low tide. The cliff line is cut into its
  summits by dry valleys that hang above the sea. The sea clears fallen chalk quickly, so talus at
  the foot is small and short-lived.
- **Generator** (`SEA_CLIFF`).
  - **Section** (along the distance in from the coastline):
    - a wave-cut platform 1 m under the sea, falling seaward at 3°;
    - a talus apron of 34°, reaching out up to 26 m where the noise allows;
    - a wall of 80° from a toe 6 m in;
    - a rim rounded over 1.5 m.
  - **Weight.** The cut follows the cliffed coast's weight over 0.05–0.45.
  - **Distance.** It is measured within 0.04 of the level in the coast field, from a 16 m finite
    difference, which keeps the GPU twins' wall within millimetres of the CPU twin's.
- **Clustering.** Along cliffed coasts (where the coast is not a beach). The rim follows the
  coastline's plan.
- **Height over width.** Not a body but a face. Its ratio is the face's height over the apron's
  width, 45–95 m over at most 26 m.
- **Measured (place 4).**
  - The wall's median slope is 80.6°.
  - The rim run, median, was 12.75 m and is now 0.5 m, a single cell (the floor of what the
    measurement can see).
  - Before, the coast rose through a U-shaped ramp, which read as a melted face under a toothed rim.

### 4.5 Mountain ridge

```
   alpine reference                         this world, after
          /\   arête                             __
         /  \  rock walls 45–70°               _/  \_
        /    \                               _/      \_   flanks mostly 0–20°
       /      \                            _/          \_
   ___/ scree  \___  30–38°            ___/              \___
```

- **Reference.** Alpine ridges in swissALTI3D 0.5 m: crisp arêtes, rock walls of 45–70° and more,
  scree aprons of 30–38° below them, and gullies cut into the walls and fanned out over the scree.
- **Generator** (`ridgesAt`, `erosion.ts`).
  - **Ridges.** Ridged octaves of simplex noise, not value noise (5 octaves, from about 8.2 km down
    to about 370 m). Each octave's amplitude is damped by the slope already built (`RIDGE_DAMP`), so
    the finer octaves add detail on the crests and benches more than on the flanks.
  - **Gullies.** They fade in over 6–24° and out over 40–55° (`GULLY_SLOPE`), so walls stay walls,
    and only the aprons and flanks are gullied.
- **Clustering.** Mountain countries are regions; within one, the ridges follow the noise's
  crestlines.

| Mountain land, 0.5 m | Bench | Talus | Gap | Cliff | Modes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Before | 56.5 % | 11.2 % | 1.6 % | 0.1 % | 6°, 8° |
| After | 79.7 % | 5.0 % | 1.0 % | 0.1 % | 4° |

- **Over 8 m** (central differences at 20,000 random points over 80 km, where mountains weigh at
  least 0.7): the mean slope fell from 15.1° to 10.2°, and the share over 30° from 10.6 % to 5.7 %.
- **Per term** (RMS slope over mountain land, before → after):

  | Term | Before | After |
  | :--- | :--- | :--- |
  | Ridges | 0.207 | 0.164 |
  | Gullies | 0.283 | 0.137 |
  | Lift | 0.126 | 0.085 |
  | Uplift | 0.167 | 0.167 (unchanged) |
  | Total | 0.360 | 0.270 |
- **Relief** (`bench/forms/relief.ts`, seed 1337): the median stays at 322–324 m. The summit is 648 m
  after and 701 m before, both within TotK's 630–730 m.

**Shortfall.** This page does **not** meet the reference, before or after:

- **The trade.** Removing the gully comb (the corrugations at place 2) and damping the ridges by
  slope took about a third of the mountains' slope. The flanks are cleaner but gentler.
- **The gap.** Neither build has the alpine walls (under 1 % of mountain land is steeper than 45°)
  or a scree mode.
- **Next step (not taken here).** Sharpen the crest (a ridge profile with a kink, not the square of a
  ridged noise), give the flanks planar 30–40° faces, and add rock bands that step them, as the
  canyon strata do. That needs the swissALTI3D tolerances to aim at.

## 5. The five places, before and after

The dumps (`bench/forms/dump.ts`) are 2 × 2 km at 0.5 m around each place reported. The analysis is
`bench/forms/analyze.py`: geomorphons on 2 m cells with 60 m lines of sight, and β over 10–200 m.
Place 3's x was hidden by the minimap. It was re-read as 1849, and the window spans ±1 km around it.

| Place | Landform | β before | β after | Slope class | Spur | Hollow | Ridge | Valley | Footslope | Shoulder | Flat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 (5767, −25258) | Jungle karst | 3.54 | 4.92 | 48.8 → 63.5 | 14.7 → 8.7 | 13.2 → 6.6 | 4.6 → 3.0 | 5.5 → 3.7 | 3.4 → 3.8 | 3.8 → 4.3 | 4.8 → 5.5 |
| 2 (5571.8, −25027.4) | Mountain flank | 4.30 | 4.94 | 48.8 → 60.7 | 12.3 → 7.4 | 11.2 → 5.4 | 4.8 → 3.4 | 5.6 → 4.0 | 4.1 → 4.7 | 4.9 → 5.5 | 7.1 → 8.0 |
| 3 (1849, −2753) | River banks | 3.61 | 3.61 | 81.9 → 86.5 | 6.9 → 4.5 | 5.6 → 3.6 | 3.1 → 3.1 | 1.7 → 1.6 | 0.1 → 0.0 | 0.2 → 0.2 | 0.0 → 0.0 |
| 4 (1205, −2644) | Sea cliff | 3.74 | 3.73 | 86.4 → 90.6 | 4.5 → 2.6 | 3.8 → 2.0 | 2.0 → 1.7 | 1.0 → 0.8 | 0.4 → 0.5 | 0.2 → 0.3 | 1.4 → 1.4 |
| 5 (−2585.4, −3520.1) | Karst towers | 4.82 | 5.13 | 50.1 → 64.3 | 16.4 → 11.7 | 15.5 → 9.1 | 6.5 → 4.3 | 8.6 → 5.0 | 0.8 → 2.3 | 0.4 → 1.6 | 0.1 → 0.7 |

Geomorphon shares are in %.

- **The spur–hollow pairs** (the teeth of a comb, and the corrugations on a flank) fell by a third to
  a half at places 1, 2 and 5.
- **Benches and breaks.** Footslopes and shoulders rose at place 5, and flats at places 1, 2 and 5.
- **β** rose at the places whose forms changed (1, 2 and 5), toward the planes-and-kinks end (§3).
  It held where they did not.

**The comb**, from `bench/forms/comb.ts`: contours traced 512 m from random points near places 1, 2,
4 and 5 and 12 places drawn over 80 km. The aspect's swing along each contour is taken apart by a
DFT. A trace is a comb when its strongest 32–160 m swing is at least 8° and holds 30 % of the power.

| Slope band | Comb share, before | Comb share, after | Median swing, before | Median swing, after |
| :--- | :--- | :--- | :--- | :--- |
| Gentle, < 12° | 40 % | 24 % | 19.5° | 11.7° |
| Flank, 20–40° | 37 % | 28 % | 15.2° | 9.5° |
| Steep, 40–60° | 29 % | 25 % | 11.6° | 8.7° |
| Wall, > 60° | 12 % | 0 % | 3.2° | 0.2° |

Place 1's gentle band fell from 38 % to 3 %, and place 5's walls from 31 % to 0 %.

`analyze.py` also has an older straight-line score (a spectral peak along a line), and it *rose* at
place 5: 0.73 → 2.07 dB at 42.7 m. That is the towers' faces, a median of 45.9 m long, and not a
comb: with the karst term taken out at place 1 it falls from 0.90 to 0.46. The contour detector is
the one the tests use.

## 6. What a heightfield can reach, and the smallest cliff mesh

**Camera and clipmap.** The camera has a 70° vertical field over 900 rows, which is about 643 px per
metre at 1 m. The clipmap's finest level is 0.5 m within ±32 m of the camera, and the next is 1 m.

- **The finest break.** The sharpest a break can be drawn is one cell: 32 px at 10 m (0.5 m cell),
  26 px at 25 m and 13 px at 50 m (1 m cells).
- **Rims.** They are rounded over the footprint so they do not alias. Their measured runs (0.5–2.25 m)
  span:

  | Place | At 10 m | At 25 m | At 50 m |
  | :--- | :--- | :--- | :--- |
  | 1 | 96 px | 39 px | 19 px |
  | 4 | 32 px | 13 px | 6 px |
  | 5 | 145 px | 58 px | 29 px |

  At 30–50 m and beyond, a rim reads as a crisp edge on a wall tens of metres high. At 10 m it reads
  as a rounded arris, and finer would alias.
- **Detail on the face.** At 77°, one 0.5 m cell climbs 2.1 m of wall, and a 1 m cell 4.2 m (3.0 m
  and 6.0 m at the sea cliff's 80°). The geometry carries nothing on a face finer than 2–6 m in
  height. Strata, joints and ledges are drawn by the surface: biplanar rock, world-space strata and
  vertical joints, and curvature edge light (WebGPU only; the WebGL2 demo raymarcher has none of it).
- **No overhangs.** A heightfield has no undercut: not the foot notches of a Guilin tower, not a sea
  notch, not a caprock overhang.

**Verdict.** At 10–50 m a pure heightfield reaches the brief's planes, slopes, benches and modes, and
its breaks are as crisp as the footprint allows. It cannot carry wall detail under about 2–6 m, and
it cannot undercut.

**The smallest layer that would** (proposed, **not built**): a wall-displacement band on each karst
tower and sea cliff within about 64–100 m of the camera.

- **Source.** Generated in the vertex shader from the same lattice hash and parameters as the
  heightfield's wall, so it stays infinite, deterministic and seamless.
- **What it adds.** Strata ledges, joint recesses and a 2–4 m foot notch.
- **The heightfield** stays authoritative for collision, water and everything farther away.
- **Budget.** About 7 faces × 32 × 64 quads, about 14k vertices per tower; at most 9 towers near
  enough is about 130k vertices.

## 7. Reproducing, and the pending DEM step

| What | Command |
| :--- | :--- |
| Dumps of E and each term (2 × 2 km at 0.5 m, float32) | `bun bench/forms/dump.ts` |
| Hillshade, slope, curvature, geomorphons, spectrum and comb score per place | `python3 -I bench/forms/analyze.py <dump dir>/<place>` |
| Breaks of slope | `python3 -I bench/forms/breaks.py <dump dir>/<place>` |
| Slope histograms by landform | `bun bench/forms/slopes.ts [--only=tower,mesa,canyon,ridge] [--root=<older checkout>]` |
| Contour comb | `bun bench/forms/comb.ts [--root=<older checkout>]` |
| Relief | `bun bench/forms/relief.ts` |
| Golden frames at the five places (near, mid and far, 12:00) | `bun bench/webgpu.ts stills --config=forms` |
| Before and after beside the TotK reference | `python3 -I bench/forms/sheet.py <before> <after> [--ref=<captures>] --out=<sheet.jpg>` |
| The blind A/B: a page for the players, then their answers | `python3 -I bench/forms/abkit.py build <before> <after> [--ref=<captures>] --out=<page.html>`, then `python3 -I bench/forms/abkit.py tally <answers.json> …` |

**To set the open tolerances**, run each reference tile through `dem.py`, which reports the same
slope shares, modes and test, geomorphons on 2 m cells and β:

```
python3 -I bench/forms/dem.py <tile.tif> [--crop=<col>,<row>,<size>] --json=<file>
```

GLO-30 is in degrees, so reproject it to metres first, for Guilin with
`gdalwarp -t_srs EPSG:32649 -tr 30 30`. `dem.py` also reads a dump directory, so a reference and a
place can be measured side by side. Take several tiles per landform. Each tolerance is the range
across them, and §5's after-values are checked against it.
