# Cloud World and Portal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the sky islands, the Eyries and their lore out of the land into a second world, the cloud
world. Its ground is walkable cumulus stacks, valleys and puff staircases. You reach it through a portal
ring near the spawn, and the running game swaps worlds live.

**Architecture:** A world is a seed. The cloud world's seed is `seed + 2**32`.
- Hashing truncates seeds to 32 bits (`seed | 0`), and the GPU gets seeds as i32. So the islands stay
  exactly as they are, while every seed-keyed cache keeps the two worlds apart.
- Generators gate on `worldOf(seed)`:
  - Islands exist only in the clouds.
  - Trees, rocks, camps and ground ruins exist only on the land.
  - `landSurface` returns the cloud height map in the clouds.
- On the GPU, a compile-time `CLOUD_WORLD` constant picks the surface. The WebGPU backend swaps
  renderers by `(sky, world)` through the existing `setSky` path.

**Tech Stack:** TypeScript, Next.js 16, WebGPU/WGSL, bun test, and headless Chrome over CDP for checking
the game.

**Spec:** `docs/superpowers/specs/2026-10-22-cloud-world-portal-design.md`

## Global Constraints

- The cloud ground stays ≤ 1280 m (under `MAX_WORLD_H` 1300). Valley floors are 450–650 m and stack tops
  900–1280 m. Islands are unchanged (`SKY_REALM.floor` 1400, `topMax` 2900).
- Puff steps: a plain step rises ≤ 1.0 m with a gap ≤ 2.5 m edge to edge. A springy puff lifts 10–20 m.
- The cloud world is peaceful: no camps or foes. PvP still works.
- The cloud world is WebGPU only. On WebGL2 the portal shows "The cloud world needs WebGPU".
- With no map of the cloud world, the map screen and map HUD read "No map of the cloud world".
- **No dead exports:** `bun run quality` passes. CPU world cost stays within the 14/50 µs budgets.
- **Do not commit.** The working tree holds the user's uncommitted work in progress in many of the
  files touched here (`engineCore.ts`, `locomotion.ts`, `page.tsx` and more). `git add <file>` would
  sweep it into a commit. Each task ends with its tests green, and the user commits at the end.
- Read `node_modules/next/dist/docs/` before touching `page.tsx` (`AGENTS.md`: this Next.js has breaking
  changes).
- The Bash tool is zsh: unquoted `$var` is not word-split.

## Review Focus

1. **Dying or respawning in the cloud world** brings you back at the cloud portal and keeps you in the
   cloud world. It never drops you on the land at the spawn. Pinned in Task 4 (pure `spawnPoint`) and
   Task 10.
2. **Changing the seed while in the cloud world** (multiplayer seed sync included) puts you in the new
   seed's cloud world at its portal, with no islands left over from the old seed. Pinned in Task 4.
3. **Falling or gliding off an island anywhere** lands on cloud. There's never NaN, a void or water.
   Pinned in Task 2.
4. **A player in the other world** is neither drawn nor hittable (PvP rivals come from `list()`), and
   shows up once they come through. Pinned in Task 8.
5. **Only a hero crossing the ring's opening** starts a trip. Brushing past its side, the fly camera, or
   standing in the ring after arriving does nothing. Pinned in Task 7.

---

### Task 1: World seeds and the gates

**Files:**
- Create: `src/lib/world.ts`, `tests/world.test.ts`
- Modify:
  - `src/lib/skyworld/archipelago.ts` (`archipelagoInCell`, `archipelagoStep`, `isleInCell`)
  - `src/lib/ruins.ts` (`buildSite`)
  - `src/lib/forest/placement.ts` (`plantInCell`, `forestAt`, `forestCanopy`)
  - `src/lib/rocks/field.ts` (the body generator behind `rockBodiesNear`/`rockCirclesNear`)
  - `src/lib/combat/foes.ts` (`campInCell`, `spawnCamp`)
- Modify tests: `tests/skyworld.test.ts`, `tests/skyRealm.test.ts`, `tests/eyries.test.ts`,
  `tests/ruins.test.ts`. Their island seeds become `worldSeed(SEED, 'clouds')`.

**Interfaces:**
- Produces, in `src/lib/world.ts`:
  - `export type World = 'land' | 'clouds'`
  - `export const CLOUD_TAG = 2 ** 32`
  - `export function worldSeed(seed: number, world: World): number`: seed, or seed + CLOUD_TAG.
  - `export function worldOf(seed: number): World`: 'clouds' iff seed ≥ CLOUD_TAG.

- [ ] **Step 1: Write the failing tests** in `tests/world.test.ts`

```ts
const L = 1337, C = worldSeed(1337, 'clouds');
test('worlds round-trip', () => {
  expect(worldOf(L)).toBe('land'); expect(worldOf(2 ** 32 - 1)).toBe('land'); expect(worldOf(C)).toBe('clouds');
});
test('islands do not depend on the tag: seed + 2^32 and seed + 2^33 lay out the same Eyries', () => {
  // find 3 cells with an archipelago under C; compare hub cx, cz, R and top of every island to the same cell under 1337 + 2 ** 33
});
test('the land has no islands; the clouds have no trees, rocks, camps or ground ruins', () => {
  // at a hub centre found under C: islandAt(x, z, L) === null and Number.isNaN(skyGround(x, 3000, z, L))
  // over a 40×40 grid of forest cells round SPAWN: plantInCell(ix, iz, C) === null
  // rockBodiesNear(SPAWN.x, SPAWN.z, C, 400, []).length === 0; campsNear(SPAWN.x, SPAWN.z, 2000, C, []).length === 0
  // every site of ruinSitesNear(hub.cx, hub.cz, 3000, C, 99) has islandAt(site.x, site.z, C) !== null
});
```

- [ ] **Step 2: Check they fail.** Run `bun test tests/world.test.ts`. Expected: FAIL, the module is not
  found.
- [ ] **Step 3: Implement `world.ts` and the gates.**
  - `archipelagoInCell`, `archipelagoStep` and `isleInCell` return null/true at once for land seeds.
  - `buildSite` returns null in the clouds when there's no island under it.
  - `plantInCell`, `forestAt` and `forestCanopy` return null in the clouds. So do the rock body generator
    and `campInCell`/`spawnCamp`.
  - Each gate is one line at the top: `if (worldOf(seed) === 'clouds') return null;`.
- [ ] **Step 4: Point the island tests at the cloud seed.** In the four existing test files, use
  `worldSeed(SEED, 'clouds')` wherever a test expects islands.
- [ ] **Step 5: Run everything.** `bun test`. Expected: all pass. The new truncation test proves nothing
  uses the seed untruncated. If it fails, normalise that arithmetic with `| 0`; don't special-case it.

### Task 2: The cloud surface (CPU)

**Files:**
- Create: `src/lib/skyworld/clouds.ts`, `tests/clouds.test.ts`
- Modify: `src/lib/constants.ts` (add `CLOUDS`), `src/lib/terrainMath.ts` (`landSurface`, plus the
  flow/ice/lava scratch it sets)

**Interfaces:**
- Consumes: `worldOf`, `CLOUD_TAG` (Task 1).
- Produces:
  - In `constants.ts`, `CLOUDS`:
    - `valley: [450, 650]`, `billowAmp: 25`
    - `stackCell: 640`, `stackProbability: 0.55`, `stackRadius: [60, 150]`, `stackTop: [900, 1280]`
    - `wallSharpness: 6`
    - `puffRise: 1.0`, `puffGap: 2.5`, `springLift: [10, 20]`
    - `albedo: [0.93, 0.95, 0.98]`
  - `export type Stack = { x: number; z: number; r: number; top: number }`
  - `export function stackInCell(ix: number, iz: number, seed: number): Stack | null`. The cell holding
    `SPAWN` never has one. The centre stays in the middle 60 % of the cell.
  - `export function cloudSurface(x: number, z: number, seed: number, dp: number): number`. This is the
    valley billow, plus each of the ≤ 4 neighbouring stacks' columns. A column is a plateau of `top`
    out to ~0.8 r, then falls as a steep `wallSharpness` sigmoid to the valley, with billow noise on its
    radius. It takes the max of valley and columns, then clamps to ≤ 1280.
  - `elevation`, `surfaceAt` and `waterLevel` return `cloudSurface` and −1e9 water for cloud seeds, with
    `lastFlow` set to 0, no ice and no lava.

- [ ] **Step 1: Write the failing tests** in `tests/clouds.test.ts`

```ts
const C = worldSeed(1337, 'clouds');
test('ground is finite, inside the ranges, never water (20 000 random points over 40 km)', () => {
  // e === e; CLOUDS.valley[0] - CLOUDS.billowAmp <= e <= 1280; waterLevel(x, z, C) < e
});
test('elevation is cloudSurface in the clouds', () => { /* 100 points: elevation(x, z, C, 0) === cloudSurface(x, z, C, 0) */ });
test('stacks: tall at the centre, walls steep enough to climb, none in the spawn cell', () => {
  // for 20 stacks: cloudSurface(st.x, st.z) >= st.top - 5; at radius st.r the gradient > 1.5
  // stackInCell(Math.floor(SPAWN.x / CLOUDS.stackCell), Math.floor(SPAWN.z / CLOUDS.stackCell), C) === null
});
test('the land is untouched', () => { /* elevation(x, z, 1337, 0) equals a value captured before the change at 5 fixed points */ });
```

- [ ] **Step 2: Check they fail.** `bun test tests/clouds.test.ts`. Expected: FAIL, the module is not
  found.
- [ ] **Step 3: Implement** `clouds.ts` (imports only `noise.ts`, `constants.ts` and `world.ts`), and the
  cloud branch at the top of `landSurface`.
- [ ] **Step 4: Run** `bun test`. Expected: all pass.
- [ ] **Step 5: Check the CPU budget.** `bun run quality` must show CPU world cost within budget.

### Task 3: The cloud world on the GPU

**Files:**
- Modify:
  - `src/lib/gpu/worldgen.wgsl.ts`: `WORLD_WGSL` becomes `worldWGSL(world: World)`, which emits
    `const CLOUD_WORLD: bool` plus `cloudSurface`/`stackInCell` WGSL twins. `landSurface` and `groundAt`
    branch on it.
  - `src/lib/gpu/terrain.wgsl.ts`: `terrainShader`, `rasterShader`, `genShader`, `sunShader`,
    `biomeShader` and `horizonShader` each take `world`. `PROBE_WGSL` becomes `probeWGSL(world)`.
  - `src/lib/gpu/clipmapCache.ts`: the constructor takes `world`.
  - `src/lib/gpu/frame.ts`: the raster variant gets `world?: World`, default 'land'.
  - `src/lib/gpu/rasterRenderer.ts`: passes `world` through.
  - `src/lib/gpu/engineGpu.ts`.
  - `bench/gpu/webgpu-page.ts`, `bench/parity.ts`, `bench.ts`: add a parity variant 'clouds' that probes
    with `worldSeed(seed, 'clouds')`.
- Test: `tests/raster.test.ts`

**Interfaces:**
- Consumes: `World` (Task 1), `CLOUDS` (Task 2).
- Produces, in `engineGpu.ts`:
  - `GAME_VARIANT` becomes `gameVariant(sky: SkyMode, world: World): TerrainVariant`.
    - Land: today's variant with `islands: false`.
    - Clouds: `islands: true`, `ruins: true`, `grass: null`, `forest: null`, `rocks: false`.
  - `GpuBackend` gains `world(): World` (the world being drawn) and `setWorld(w: World): void`. Both share
    `switchSky`'s build-then-swap loop, which is keyed on the `(sky, world)` pair.
  - Shading in the clouds variant: albedo `CLOUDS.albedo`, with no ground texture, detail, bed tint or
    wetness. Lighting, shadows and fog are unchanged.

- [ ] **Step 1: Write the failing test** in `tests/raster.test.ts`

```ts
test('the world is compiled in', () => {
  expect(worldWGSL('clouds')).toContain('const CLOUD_WORLD: bool = true;');
  expect(worldWGSL('land')).toContain('const CLOUD_WORLD: bool = false;');
  expect(gameVariant('cycle', 'land')).toMatchObject({ islands: false, world: 'land' });
  expect(gameVariant('cycle', 'clouds')).toMatchObject({ islands: true, forest: null, grass: null, rocks: false, world: 'clouds' });
});
```

- [ ] **Step 2: Check it fails.** `bun test tests/raster.test.ts`. Expected: FAIL, `worldWGSL` is not
  exported.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `bun test`. Expected: all pass.
- [ ] **Step 5: Check CPU/GPU parity.** Run `bun bench.ts parity` (it runs headless Chrome). Expected:
  the 'clouds' variant's fp32/fp64 error stays within the same bounds the land variants report.

### Task 4: The engine knows its world

**Files:**
- Create: `src/lib/portal.ts`, `tests/portal.test.ts`
- Modify:
  - `src/lib/engine/engineCore.ts`
  - `src/app/game/page.tsx` (`?spawn=sky` comment)
  - `src/app/game/MapScreen.tsx`, `src/app/game/MapHud.tsx`, `src/lib/map/mapHud.ts` (the map state
    gains `world`)
  - `src/lib/sky/air.ts`: `landName` returns 'The Hollow Sky' for cloud seeds.
  - `src/lib/skyworld/realm.ts`: delete `skySpawn`.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces, in `portal.ts`:
  - `export type Portal = { x: number; y: number; z: number; yaw: number }`. Its ground point is the
    centre of the ring's opening at its base, and `yaw` is the ring's normal.
  - `export function portalOf(world: World, seed: number): Portal`, where `seed` is the user's seed.
    - Land: the first level, dry spot 18–40 m to the right of `SPAWN`, searched like `spawnCamp`.
    - Clouds: `SPAWN` x/z on the cloud floor, facing the nearest stack.
  - `export function spawnPoint(world: World, seed: number): { x: number; y: number; z: number; yaw: number }`.
    - Land: today's land spawn.
    - Clouds: 4 m in front of the cloud portal, facing away from it.
- In `engineCore.ts`:
  - `let world: World` starts as 'clouds' when `spawnAt === 'sky'`, else 'land'.
  - `const seedNow = () => worldSeed(seedRef.current, world)` replaces every `seedRef.current` in
    `engineCore.ts`, except network seed sync.
  - `spawnFor`/`doRespawn` use `spawnPoint(world, seedRef.current)` and keep `world`.
  - On WebGPU at start, `gpu.setWorld(world)`.

- [ ] **Step 1: Write the failing tests** in `tests/portal.test.ts`

```ts
test('portals stand on their world\'s ground', () => {
  for (const s of [1337, 42, 999_999]) {
    const p = portalOf('land', s); expect(Math.abs(p.y - elevation(p.x, p.z, s, 0))).toBeLessThan(0.01);
    expect(elevation(p.x, p.z, s, 0)).toBeGreaterThan(waterLevel(p.x, p.z, s) + 1);
    const q = portalOf('clouds', s); expect(Math.abs(q.y - elevation(q.x, q.z, worldSeed(s, 'clouds'), 0))).toBeLessThan(0.01);
    expect(q.y).toBeLessThan(CLOUDS.valley[1] + CLOUDS.billowAmp);
  }
});
test('respawn in the clouds is at the cloud portal, for any seed', () => {
  const sp = spawnPoint('clouds', 42), q = portalOf('clouds', 42);
  expect(Math.hypot(sp.x - q.x, sp.z - q.z)).toBeCloseTo(4, 1);
});
```

- [ ] **Step 2: Check they fail.** `bun test tests/portal.test.ts`. Expected: FAIL, the module is not
  found.
- [ ] **Step 3: Implement** `portal.ts` and the engine changes. The map shows "No map of the cloud world"
  when `world === 'clouds'`.
- [ ] **Step 4: Run** `bun test` and `bun run quality`. Expected: all pass, with no dead export after
  `skySpawn` is deleted.
- [ ] **Step 5: Look at it.** Start `/game?renderer=webgpu&spawn=sky` in headless Chrome. The hero stands
  on white cloud, with stacks around and islands above. There's no grass or trees. Screenshot it for the
  user: this is the look check the spec asks for.

### Task 5: Puffs (placement, floor, bounce)

**Files:**
- Create: `src/lib/skyworld/puffs.ts`, `tests/puffs.test.ts`
- Modify:
  - `src/lib/skyworld/realm.ts`: `SkyFloor` gains `Puff`. `skyFloorUnder` and `floorHeight` handle
    puffs.
  - `src/lib/hero/locomotion.ts`: `HeroState.spring`. `sampleSurface` sets it from a springy puff
    underfoot, and `land()` bounces.

**Interfaces:**
- Consumes: `stackInCell`, `cloudSurface`, `CLOUDS` (Task 2), and `realmNear`, `outline`/`islandRho`
  (existing).
- Produces:
  - `export type Puff = { x: number; y: number; z: number; rx: number; ry: number; rz: number; spring: number }`.
    `y` is the centre, and `spring` is the lift in m (0: plain).
  - `export function puffTop(p: Puff, x: number, z: number): number`. This is the top of the ellipsoid,
    NaN outside it.
  - `export function puffsNear(x: number, z: number, range: number, seed: number, out: Puff[]): Puff[]`.
    It covers the stack stairs of `stackInCell` and the island ladders of `realmNear`. Both are cached
    per cell, like the ruins.
  - `export function puffChain(kind: 'stair' | 'ladder', ix: number, iz: number, seed: number): readonly Puff[]`.
    It returns ordered steps and is exported for the test.
  - The bounce on landing on a puff with `spring > 0`: an upward `vy` whose apex is `spring` above the puff,
    and the landing roll and squat are skipped. Start from `Math.sqrt(2 * PLAYER.gravity * spring)` and
    correct it for `HERO.apexGravity`/`apexBand` hang time. The bounce test is the arbiter.

- [ ] **Step 1: Write the failing tests** in `tests/puffs.test.ts`

```ts
const C = worldSeed(1337, 'clouds');
test('every chain is climbable step by step', () => {
  // for 20 stair chains and 20 ladder chains: for each consecutive (a, b):
  //   const rise = (b.y + b.ry) - (a.y + a.ry); expect(rise).toBeLessThanOrEqual(a.spring > 0 ? a.spring : CLOUDS.puffRise);
  //   edge gap = hypot(b.x-a.x, b.z-a.z) - max(a.rx, a.rz) - max(b.rx, b.rz); expect(gap).toBeLessThanOrEqual(CLOUDS.puffGap);
  //   every spring is within CLOUDS.springLift
  // a stair starts within one step of the valley floor and ends within one step of its stack's top
  // a ladder ends within one step of its island's top at the rim, and no puff lies inside an island's outline
});
test('a springy puff throws the hero up by its lift', () => {
  // find a puff with spring 15 ± 5; placeHero just above its top; run updateHero with no input until the apex;
  // expect(apex - top).toBeGreaterThan(p.spring - 1) and toBeLessThan(p.spring + 1)
});
test('puffs are floor: skyGround over a puff is its top', () => { /* skyGround(p.x, p.y + p.ry + 1, p.z, C) ≈ puffTop(p, p.x, p.z) */ });
```

- [ ] **Step 2: Check they fail.** `bun test tests/puffs.test.ts`. Expected: FAIL, the module is not
  found.
- [ ] **Step 3: Implement.**
  - A stair spirals around the stack's wall at radius r + 3. Every 5th step is springy, and the steps
    after a spring continue from its lift height.
  - A ladder rises beside the island's rim, 6 m outside its outline. It is all springs, with a plain rest
    puff every 4th step, and its last step stands level with the top at the rim.
- [ ] **Step 4: Run** `bun test` and `bun run quality`. Expected: all pass.

### Task 6: Drawing the puffs

**Files:**
- Create: `src/lib/gpu/puffs.wgsl.ts`
- Modify: `src/lib/gpu/rasterRenderer.ts`, `src/lib/gpu/terrain.wgsl.ts`, `src/lib/gpu/frame.ts`
  (`puffs?: boolean`, on in the clouds variant), `src/lib/gpu/engineGpu.ts`

**Interfaces:**
- Consumes: `puffsNear`, `Puff` (Task 5).
- Produces: `MAX_PUFFS = 2048`, `PUFF_BYTES = 32` (centre xyz chunk-local and spring; radii xyz and a
  hash), and `PUFFS_WGSL` with `vsPuff`/`fsPuff`.
  - Instanced icospheres (2 subdivisions), vertex-pulled from a storage buffer at the next free group-0
    binding. They follow `ruins.wgsl.ts`.
  - Shading: `CLOUDS.albedo` with soft wrap lighting. A springy puff gets a faint warm rim so players can
    tell it apart.
  - Puffs are re-picked every 100 m, like `uploadRuins`.

- [ ] **Step 1: Write the failing test** in `tests/raster.test.ts`: the clouds variant has `puffs: true`,
  the land variant has none, and `rasterShader({... puffs: true})` contains `fn vsPuff`.
- [ ] **Step 2: Check it fails.** `bun test tests/raster.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `bun test`. Expected: all pass.
- [ ] **Step 5: Look at it.** In headless Chrome at `?spawn=sky`, walk to the nearest stack. A stair of
  puffs winds up it, and springy puffs show their rim. Screenshot it.

### Task 7: The portal ring and the trip

**Files:**
- Modify:
  - `src/lib/portal.ts`: add the crossing test and the ring's boxes.
  - `src/lib/ruins.ts`: `ruinSitesNear`/`ruinSiteAt` include the world's portal site.
  - `src/lib/constants.ts`: add `PORTAL`.
  - `src/lib/engine/engineCore.ts`: the trip.
  - `src/app/game/page.tsx`: the fade overlay, passed in as `inputs.fade`.
  - `src/lib/engine/useTerrainEngine.ts`: the `fade` input.
- Test: `tests/portal.test.ts`

**Interfaces:**
- Consumes: `portalOf`, `spawnPoint` (Task 4), `GpuBackend.setWorld/world` (Task 3).
- Produces:
  - `PORTAL = { radius: 3.5, segments: 12, fade: 0.45, cooldown: 2.0 }`.
  - `export function portalSite(world: World, seed: number): RuinSite`. It is `shrine: true`, made of
    shrine-stone boxes standing in a vertical ring in the plane of `yaw`.
  - `export function portalCrossed(p: Portal, ax: number, ay: number, az: number, bx: number, by: number, bz: number): boolean`.
    It is true iff the segment a→b crosses the ring's plane inside a radius of `PORTAL.radius - 0.5`
    around its centre (base + radius up).
  - **The trip** in `engineCore.ts`, run in hero mode only:
    1. On `portalCrossed`, with no cooldown, call `gpu.setWorld(other)` and fade the overlay out over
       `PORTAL.fade` s.
    2. Once it's black and `gpu.world() === other`: set `world = other`, place the hero at
       `spawnPoint(other, seed)`, and call `combat.reset(seedNow())`.
    3. Fade back in and start the cooldown.
    - Health, pouch, clock and network are untouched.
    - On WebGL2 (`gpu === null`), call `say('The cloud world needs WebGPU')` at most once per cooldown.

- [ ] **Step 1: Write the failing tests** in `tests/portal.test.ts`

```ts
test('only a pass through the opening counts', () => {
  const p = portalOf('land', 1337), nx = Math.sin(p.yaw), nz = Math.cos(p.yaw), cy = p.y + PORTAL.radius;
  // through the middle, either way: true
  // beside it (offset 5 m along the ring's plane), above it (cy + 4), and a step that stops short of the plane: false
});
test('the ring is a ring of shrine stone with its opening clear', () => {
  const s = portalSite('land', 1337);
  expect(s.boxes.length).toBe(PORTAL.segments); expect(s.boxes.every((b) => b.material === 2)).toBe(true);
  // no box contains the point at the opening's centre
});
test('each world draws its own ring', () => {
  // ruinSitesNear(portal.x, portal.z, 50, seed, 99) includes the portal site in its world: land seed for land, worldSeed(seed, 'clouds') for the clouds
});
```

- [ ] **Step 2: Check they fail.** `bun test tests/portal.test.ts`. Expected: FAIL, `portalCrossed` is
  not exported.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `bun test`. Expected: all pass.

### Task 8: Multiplayer knows the world

**Files:**
- Modify: `src/lib/net/remoteHeroes.ts`, `src/lib/engine/engineCore.ts` (passes `world` to
  `encodeSnapshot` and `update`)
- Test: `tests/multiplayer.test.ts`

**Interfaces:**
- Produces:
  - Snapshot field `S.world = 41`, with `S.length = 42` (0: land, 1: clouds; missing means land).
  - `encodeSnapshot(..., world: World = 'land')` takes it as its last parameter.
  - `RemoteHeroes.update(link, now, dt, world: World)`: `list()` and `pack()` hold only peers whose latest
    snapshot is in `world`.
- `server.mjs` needs no change. It passes up to 48 numbers through.

- [ ] **Step 1: Write the failing test** in `tests/multiplayer.test.ts`: two peers, one sending
  world 1. After `update(..., 'land')`, `list()` has only the land peer. After the clouds peer's next
  snapshot says world 0, it is listed. A snapshot from an older build (length 41) counts as land.
- [ ] **Step 2: Check it fails.** `bun test tests/multiplayer.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `bun test`. Expected: all pass.

### Task 9: The lore and the docs

**Files:**
- Modify: `docs/SKY_CIVILIZATION.md`, `docs/EYRIES.md`, `docs/MAP.md`, `docs/LANDS.md`, `docs/HERO.md`,
  `PROJECT.md`, and the header comments of `constants.ts` (`SKY_REALM`, `FLY_CEILING`),
  `skyworld/realm.ts` and `engineGpu.ts`.

- [ ] **Step 1: Find the stale claims.** Run
  `grep -rn -iE "above the land|over the land|cloud deck|spawn=sky|from the land" docs src PROJECT.md`.
  Expected: a list of the places that claim the islands float over the land.
- [ ] **Step 2: Rewrite them.**
  - The Hollow Sky floats above a cloud world of its own, a "cloud sea" of stacks and valleys, reached
    through the ring near the spawn.
  - Keep the lore's voice. Add a short §1 paragraph on the ring as an Aelari "Gate of Breath".
  - `?spawn=sky` now means the cloud portal.
- [ ] **Step 3: Check nothing stale is left.** Run the grep again. Expected: only lines that are true
  now.
- [ ] **Step 4: Run** `bun run quality`. Expected: pass.

### Task 10: Check it in the game

- [ ] **Step 1: Run the test suite.** `bun test`, `bun run quality` and `bun run lint`. Expected: all
  pass.
- [ ] **Step 2: Walk a round trip in headless Chrome over CDP** (WebGPU).
  1. Spawn on the land. Read `window.__hyrule`: there's no island over the spawn.
  2. Walk through the ring.
  3. After the fade, the hero is at the cloud portal, with the same health and pouch.
  4. Climb a stair and bounce on a spring.
  5. Respawn: the hero is back at the cloud portal (Review Focus 1).
  6. Change the seed: the hero lands at the new cloud portal (Review Focus 2).
  7. Walk back through the ring to the land.
  - Screenshot each world for the user.
- [ ] **Step 3: Check multiplayer.** Run `server.mjs` with two headless Chromes through the lobby. One
  goes through the portal and the other stops seeing it. The first comes back and is seen again.
- [ ] **Step 4: Measure the frame-time cost.** Do a real-loop A/B with `bench/play.ts` on scratch
  production builds, against HEAD. Expected: each world's frame time ≤ HEAD's, within ±0.2 ms. Note the
  stall during a switch.
- [ ] **Step 5: Report to the user.** Show the screenshots, say what was verified and what wasn't, and
  list the files touched, ready for them to commit.
