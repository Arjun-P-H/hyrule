# The cloud world and its portal: design

Date: 2026-10-22. Status: **parked (2026-10-26).** The sky islands are back over the land as before; the cloud
world stays in the code, inactive: no ring on the land, reached only by Engine Settings → World, for inspection.

## 1. Intent

The sky islands, the Eyries and the Aelari lore built on them leave the main world. The main world
becomes land only. The islands move to a second world, **the cloud world**, which you reach by stepping
through a portal on the land, and leave through its twin.

Under the islands, where the land used to be, lies a procedurally generated world of solid cloud: deep
valleys with towering cumulus stacks rising out of them. Staircases of floating puffs climb the stacks'
sides, so the hero can jump, climb and glide from the valley floors up to the islands.

The reason is to give the sky its own place to travel to, a destination with its own ground, instead of
a layer you reach by flying straight up from the land.

## 2. Goals, non-goals, anti-goals

**Goals**
- Two worlds in one running game: the **land** (today's terrain, with no islands) and the **clouds**
  (cloud ground, puffs and today's islands, Eyries and their contents).
- One portal on the land near the spawn, and its twin on a valley floor in the cloud world. Walking into
  either one moves the hero to the other with a short fade. Health, inventory, the clock and the
  multiplayer connection survive the trip.
- The cloud ground is walkable and climbable, and every island can be reached from it on foot: by
  hopping, climbing and bouncing up, and by gliding across.
- Multiplayer players see only the players who are in the same world as them.
- The cloud world is peaceful: no Bokoblin camps or other foes spawn there (the user's decision).

**Non-goals**
- Drifting or moving puffs (they stay still for now).
- See-through, volumetric 3D clouds with overhangs.
- More than one portal pair, or portals you build or place yourself.
- Saving progress across page reloads.
- A map of the cloud world. While you're there, the map screen and map HUD say "No map of the cloud
  world" instead of drawing land tiles. This is my call and the smallest option: a cloud-world map can
  come later.
- Drawing the cloud world on WebGL2.

**Anti-goals** (it fails if any of these happen, even if it technically works)
- Land things (sea, rivers, grass, trees, rocks, ground ruins) show up in the cloud world, or islands
  and their shadows show up over the land. One stale cache that leaks a world into the other counts.
- The trip drops the multiplayer connection or resets health or inventory.
- Getting up to the islands is a tedious slog of a thousand one-metre hops.
- Frame rate in either world falls noticeably below today's on the M1. Each world draws less than today
  does, so it should not.

## 3. Constraints

- **WebGPU only for the cloud world.** WebGL2 has never drawn the islands (`FLY_CEILING.webgl2`).
- **The hero's reach:** a jump peaks at about 1.2 m (`HERO.jumpVelocity` 7.2). The paraglider moves 7 m/s
  forward for 2.2 m/s of sink, about 3 : 1. Climbing uses stamina (`HERO.staminaClimbJump`). Any step
  up has to fit within those numbers.
- **Island heights stay as they are:** `SKY_REALM.floor` is 1400 m and `topMax` is 2900 m. The island
  generator, the Eyries and their colliders do not change.
- **Height ceiling:** `MAX_WORLD_H` (1300 m) is an early-exit ceiling that the ray-march, the clipmap
  bounds and projectiles all rely on. The cloud ground stays under it, at most 1280 m. Every island lies
  above `SKY_REALM.floor` (1400 m), so no stack can ever cut through an island.
  - Revised while planning: the first draft had stacks reaching about 2000 m.
- **CPU/GPU twins:** every surface is computed on the CPU (physics) and in WGSL (drawing). The two must
  agree, and the parity tests in `bench.ts` and the unit tests keep them honest.
- **Code quality rules:** no dead exports, the quality gate passes, and CPU world cost stays inside the
  14/50 µs budgets.
- **The working tree:** the branch `chore/cleanup-gauntlet` holds a large amount of uncommitted work in
  progress, including combat, inventory, the map and the ground. This work builds on top of it and must
  not overwrite it.

## 4. Approaches

1. **Live world switch (chosen).** The running engine swaps worlds in place. The WebGPU backend
   already builds a new renderer in the background and swaps it in when the sky mode changes
   (`setSky` in `engineGpu.ts`). The world switch uses the same path, keyed on world and sky
   together. The CPU side marks the hero's world, and every world-dependent cache is keyed by it.
   - It keeps all in-memory state and the network link.
   - It costs the most work, and stale caches are the main risk.
2. **Portal reloads the page** into `/game?world=clouds`, so the world is fixed once at start, the way
   `?spawn=sky` already works.
   - Much smaller.
   - It loses health and inventory (nothing saves them), drops multiplayer, and adds a loading pause.
   - Rejected for those losses.
3. **Volumetric clouds:** a raymarched density field with collision against it.
   - The fluffiest look.
   - It needs new collision and new climbing, and its cost on the M1 is a real risk.
   - Rejected. Soft shading on the height-map clouds gives the fluffy look instead.

## 5. The design (decided)

### 5.1 World identity
- There are two worlds, `'land'` and `'clouds'`. The engine holds the current world, the hero's world.
- The WebGPU renderer variant gets a `world` field, and `GAME_VARIANT` becomes a function of sky and
  world.
  - **Land variant:** `islands: false` (no realm, Eyries, birds, motes, lanterns, waterfalls or
    sky-island shadows). Everything else stays as it is today.
  - **Clouds variant:** the islands and their kin are on. The ground is the cloud surface. Sea and lakes,
    rivers, grass, forest, rocks and ground ruins are off.
- **CPU queries:** the realm's queries (`islandAt`, `skyGround`, `skyWater`, `skyFloorUnder`, the Eyrie
  colliders, the island ruins) answer "nothing" in the land world. The land's queries (`elevation`,
  `waterLevel`, forest trunks, rocks, ground ruins) answer with the cloud surface, or nothing, in the
  cloud world. These are the call sites in locomotion, the camera, projectiles, combat and spawning.
- **Caches:** every CPU cache that depends on the world is keyed by the world as well as the seed.
  These include forest tiles, rocks, ruins, archipelago cells, the hydro texture and combat (foes).
  A switch must never serve a stale entry.
- **How (decided while planning):** the cloud world's seed is the user's seed + 2³².
  - Every seed-keyed cache sees a different seed and keeps the two worlds apart without further work.
  - Hashing truncates seeds to 32 bits (`seed | 0`), and the GPU receives seeds as i32. So the islands
    come out exactly as they do today, and the CPU and GPU still agree.
  - Code asks `worldOf(seed)` which world a seed belongs to.

### 5.2 The cloud ground

> **Revised 2026-10-22 after playtesting, from the user's reference paintings.** The stacks, their puff stairs
> and their lobes are gone. The ground is now a rolling cumulus sea (880–1280 m, heaped |noise| domes at
> 1100/380/120/36 m), with drawn lobes on its crests (within 2 km) and banks of cloud under every island
> (within 6 km). Lone isles are 5× as common (60 % of 2 km cells), 30–240 m across, layered 1450–2850 m.
> Puffs are ladders only. The text below is the first design, kept for the record.

- A height map, `cloudSurface(x, z, seed)`, with a CPU version and a WGSL twin. It lives in the
  `skyworld` area.
- **Valley floors** sit around 450–650 m, with gentle billows.
- **Cumulus stacks** rise to 900–1280 m, below the lowest islands (1400 m). They have billowed sides and
  their own procedural spacing, which is independent of the islands. Their walls are steep enough to
  read as towers and to be climbed with today's climbing (the land's cliffs already work this way).
- **No stack in the portal's cell:** the cell holding `SPAWN` has no stack, so the cloud portal always
  stands on an open valley floor.
- **Fall rule:** there is no water and no void. The valley floor is solid, so a fall lands you lower down.
  Fall damage works as it does on the land.
- **Look:** white and soft. The lighting comes from the day-night sun, with soft wrap shading and a
  bright rim at grazing angles. There is no grass or texture detail, though a little shading noise is
  allowed. Before this is final, a mockup of the look will be shown to the user.
- **The sky's cloud layer:** it stays where it is today (`cloudPlane` at 1500 m). That is above every
  stack top and below most islands, the same arrangement as today.

### 5.3 Puffs and springy puffs
- **Puffs:** small, still, squashed-sphere cloud bodies that you can stand on. They are placed on the
  CPU, collided as part of the sky floor (`skyFloorUnder`), and drawn as instanced meshes the same way
  the ruins' boxes are.
- **Stack stairs:** a staircase winds up each stack's side, from the valley floor to the stack's top.
- **Island ladders:** each island except debris gets a ladder of puffs. It rises from the cloud ground
  just outside the island's rim up to the island's top, staying clear of the island's outline.
- **Steps:** a plain step rises ≤ 1 m and leaves a gap of ≤ 2.5 m that a running jump clears.
- **Springy puffs** launch the hero 10–20 m straight up on landing, and the next puff waits at the top of
  that lift. Stairs mix hops with bounces. Ladders are mostly bounces, so a climb of about 1 km takes
  minutes, not hundreds of hops.

### 5.4 The portal
- **On the land:** a glowing stone ring stands on flat ground near `SPAWN`. It is built as a ruin site
  made of shrine-stone boxes, so it is drawn and collided like the ruins, and grass and trees keep out
  of it.
- **In the cloud world:** its twin stands on a valley floor, near the foot of a stack whose staircase is
  easy to find.
- **Using it:** walking through the ring's opening starts the trip.
  1. The view fades out.
  2. The other world's renderer is built in the background, or reused if it is already built.
  3. The hero is placed just in front of the twin, facing away from it.
  4. The view fades in.
  5. A short cooldown stops the hero bouncing straight back through.
- **During the trip:** the clock, health, inventory and the network link carry on unchanged.
- **On WebGL2:** the ring is drawn, and walking into it shows "The cloud world needs WebGPU". Nothing
  else happens.
- **The URL option `?spawn=sky`** becomes "start in the cloud world, at the portal's twin".

### 5.5 Foes and the map in the cloud world
- **No foes.** `CombatWorld` places no camps in the cloud world. `campGround` would otherwise accept
  level cloud ground. Player-versus-player fighting between players in the cloud world still works as it
  does on the land.
- **The map:** see the non-goals. The map's tile worker only knows the land, so in the cloud world the
  map is replaced by the "No map of the cloud world" note.
- **Island content goes with the islands:** the ruins on island tops and the sky ships (`ships.ts`) move
  to the cloud world (my call).
- **WebGL2 portal fallback:** if drawing the ring there is costly, walking up to its spot shows the
  "needs WebGPU" hint instead (my call).

### 5.6 Multiplayer
- The hero snapshot carries the sender's world, as one extra field.
- Remote heroes are drawn and simulated only when their world matches yours.
- The server needs no world logic beyond passing the field through.

### 5.7 The lore and the docs
- [SKY_CIVILIZATION.md](../../SKY_CIVILIZATION.md) is retold so the Hollow Sky floats above a sea of
  cloud mountains in a world of its own, reached through the ring. The cloud deck becomes the "cloud
  sea" ground you walk on.
- [EYRIES.md](../../EYRIES.md), [MAP.md](../../MAP.md), [LANDS.md](../../LANDS.md), [HERO.md](../../HERO.md),
  `PROJECT.md` and the code comments that say the islands float over the land are updated to match.

## 6. Tests
- **Parity:** the CPU and WGSL versions of the cloud surface agree (the same style as today's terrain
  parity checks).
- **Reachability:** for a few seeds, follow each staircase. No step needs more than a jump or a springy
  puff's lift, and every stack's top lies within a glide of an island or another staircase.
- **World isolation:** in the land world, the realm queries return nothing at an island's (x, z). In the
  cloud world, the land queries return the cloud surface, and there are no trees, rocks, ruins or water.
- **Portal:** entering the ring switches worlds, places the hero at the twin, and keeps health and
  inventory. The cooldown stops an immediate return.
- **Multiplayer:** a remote hero in the other world is not shown. When it comes through the portal, it
  appears.
- **Existing tests** that place the hero on an island (skyworld, skyRealm, hero, ruins) are updated to
  run in the cloud world.

## 7. Left to the builder
- Exact noise, spacing and heights for stacks and valleys, within the ranges above.
- How puffs are placed along a stack, and how a springy puff looks and sounds.
- Whether the land variant's realm-free shader is a compile-time switch (preferred, as `islands`
  already is) or a runtime flag.
- The ring's model and glow, and the fade's length.
- Whether to keep both worlds' renderers alive after a first trip, so later trips are instant, or to
  destroy the old one. Choose on the M1's memory and frame time.

## 8. Risks
- **Stale world state leaking across the switch.** Mitigation: key caches by world, and run an isolation
  test for each cache.
- **Code that does arithmetic on the seed without truncating it** could make the 2³² seed tag break CPU/GPU
  agreement. Mitigation: a test checks that the islands for the cloud seed equal the islands for the user's
  seed, and the parity bench gets a clouds variant.
- **The renderer swap stalling a frame on the M1.** Mitigation: build behind the fade, and measure with
  the real-loop A/B (bench/play.ts on production builds).
