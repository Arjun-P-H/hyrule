# Third-person hero

The `/game` page now starts in third person: a green-tunic hero modelled on the classic *Legend of
Zelda* look, with open-world Zelda locomotion (run, sprint, sneak, jump, paraglide, climb, swim, a
stamina wheel). **V** switches to the existing first-person controller and back; **F** still takes
off into free flight and lands in whichever view was last used on foot.

## Controls (third person)

| Key | Action |
| :--- | :--- |
| WASD / arrows | Run, relative to the camera; the hero turns to face the way it moves |
| Shift | Sprint (stamina); dash stroke while swimming |
| Space | Jump · in the air (≥ 2.5 m up): open / fold the paraglider · on a wall: climb jump |
| C | Sneak · on a wall: let go |
| Mouse (click to lock) | Orbit the camera · wheel: zoom 1.6 – 11 m |
| V | First person (and back) |
| F | Free flight (F again lands the hero where the camera is) |
| R | Respawn |

## How it is built

Nothing here is a mesh or a texture: the zero-mesh / zero-texture invariant holds.

- **Locomotion** — `src/lib/hero/locomotion.ts`. Same contract as the first-person `updatePlayer()`:
  deterministic, mutated in place, no allocation per tick, terrain queried through the CPU twin's
  `elevation()` on fp64 global coordinates (one sample plus two forward differences per tick). When
  grounded the root sits at `E + GROUND_BIAS`, exactly like the first-person eye minus its height, so
  switching views keeps one collision contract. States: ground (run / sprint / sneak / slide down
  ground steeper than 45°), air, glide, climb (push into ground steeper than `HERO.walkSlope`; hang,
  climb, climb-jump, let go), swim (water deeper than 1.3 m: the sea, a lake, a river's pool, from `surfaceAt`; frozen water is walked on, see [`LANDS.md`](LANDS.md)). Constants: `HERO` in
  `constants.ts`.
- **Animation** — `src/lib/hero/rig.ts`. An 18-joint skeleton posed every frame: a gait cycle whose
  stance feet move backwards at exactly the ground speed (so they stay planted), placed on the
  terrain plane at the root with two-bone IK for knees and elbows; air (tuck / fall / skydive),
  paraglider, climb and swim poses cross-faded by smoothed weights; lean into acceleration and turns,
  a landing squash, idle breathing. It then packs the joints, a torso and a head frame and a bounding
  sphere **camera-relative** into one `Float32Array` (fp64 differences before the fp32 store, so the
  body keeps sub-millimetre precision 1,000 km from the origin).
- **Camera** — `src/lib/hero/camera.ts`. Orbit on the mouse; the look-at point trails the hero by at
  most 0.6 m; terrain between the hero and the eye shortens the boom (snap in, ease out); the eye
  stays 0.45 m above the ground and the water; the view is re-aimed from the final eye to the target.
- **Rendering** — `heroMap()` in `demoShader.ts`: capsules, tapered capsules, a capped cone (tunic
  skirt), boxes, a dome shell (paraglider) around the joints, with material ids and procedural detail
  (eyes, belt buckle, shield rim / emblem / crest, boot and gauntlet splits, canopy stripes).
  `main()` sphere-traces the hero inside its bounding sphere **before** the terrain and caps the
  terrain march at the hit, so hero pixels skip most of the terrain march; the few samples between the
  last terrain step and the hero are resolved exactly (water plane, then one terrain sample). The hero
  receives the terrain's soft shadow, casts a soft shadow on terrain, water and itself (one query per
  pixel), and darkens the ground around its feet.
- **Two programs.** All hero code sits under `#ifdef HERO`. First person and fly draw with the plain
  `TERRAIN_FS` program — no hero code at all, so their speed and compile time are those of the terrain
  shader. Third person draws with `withHero(TERRAIN_FS)`, which the engine compiles without a status
  query (`KHR_parallel_shader_compile`) so it builds in the background; until it is ready the terrain
  program keeps drawing and the hero appears a moment after start. Each expensive function keeps one
  call site (`softShadow`, `heroShadow`, `heroTrace`; `elevation` its original five), because the GPU
  compiler inlines every call.

The extents that bound `heroMap()` around each joint (`HERO_SHAPE.*Reach`, `limbPad`) are mirrored
as `HERO_*` constants in the shader and pinned by `tests/hero.test.ts`; the CPU builds the bounding
sphere from them.

## Tests

`tests/hero.test.ts` (37 tests) runs everything on the real terrain around the spawn:

- **Locomotion** — spawn on the ground (or afloat); run speed and camera-relative heading; diagonals
  not faster; sneak speed with exact ground contact; one jump per press with a v²/2g apex; sprint
  drains one wheel in 1/`staminaSprint` s, then exhaustion (no sprint, slow jog) until the wheel is
  full again, recovery only after the rest delay; paraglider opens only on a mid-air press with
  clearance, sinks at `glideSink`, folds on a second press or a dry wheel, terminal velocity on long
  drops; climbing grabs walls steeper than `walkSlope`, moves at `climbSpeed` along the surface, hang
  and climb-jump costs, letting go, no re-grab while exhausted, climbs end where the wall eases;
  steep ground slides; swimming floats at the swim depth, dashes on stamina, and the hero walks from
  the lake bed into deep water and back out; bit-identical trajectories for identical input; fuzzed
  input stays finite and never below the ground, near the origin and 1,000 km out.
- **Rig** — bone lengths hold in every settled pose; stance feet stay planted while running (the
  core property of the gait); idle feet rest within 3 cm of the true surface on a 27–39° slope; the
  bounding sphere holds every joint with its primitive reach in every state; the packed frames are
  orthonormal with right = up × forward, the camera basis convention; camera-relative joints keep
  < 0.1 mm error 1,000 km from the origin.
- **Camera** — orbit distance and aim; never under the terrain or the water, boom never shorter than
  `minDistance` (480 orbit samples at four sites); a fast fall is trailed by at most `maxLag`.
- **Shader contract** — `HERO_*` reach constants mirror `HERO_SHAPE`; uniform layout matches
  `HERO_UNIFORM`; the hero is gated on `u_hero_bound.w`; `softShadow`, `heroShadow` and `heroTrace`
  keep one call site and `elevation` its five; the joint array is indexed by constants only; the
  terrain-only program contains no hero code once `#ifdef HERO` is preprocessed out; every benchmark
  shader transform (`render`, `instrument`, `probe`, `evalcost`, `truth`) still applies to both
  programs.

## Measured cost

Apple M1 (7-core GPU), headless Chrome 154 (ANGLE → Metal, native arm64), production build. GPU
figures come from `bun bench/hero.ts cost`: timer queries along the benchmark fly-through with the
hero posed where the third-person camera frames it (4.6 m down the view axis), all variants
interleaved frame by frame in one session, paired per-frame ratio with a 95 % bootstrap CI (the
harness's A/A spread is ±0.03). High runs 20-frame sessions (2 reps): 60-frame High sessions wedge
the ANGLE/Metal GPU process.

| Axis | Result |
| :--- | :--- |
| GPU, third person vs first person at the same view — running | Low 0.990 (0.965–1.017) · Medium 0.997 (0.981–1.014) · High 0.971 (0.934–1.014) |
| — paragliding (canopy open) | Low 1.001 (0.975–1.031) · Medium 1.015 (0.998–1.032) · High 1.006 (0.974–1.038) |
| Drawing the hero (same program, hero on vs off) | 0.995 – 1.042: its pixels skip most of the terrain march |
| First person vs the previous revision | Low 1.000 (0.974–1.028) · Medium 1.006 (0.992–1.021) · High 0.999 (0.966–1.032) |
| First-person frames vs the previous revision | 134 of 140 bit-identical; the other 6 differ in one pixel by 1/255 (ΔE_ok ≈ 1e-9) — float rounding of the single-call-site shading layout |
| Cold shader compile → first frame (fresh browser, median of 5) | previous 128 ms · terrain program 130 ms · hero program 145 ms, built in the background |
| CPU per frame (`bun bench/hero.ts cpu`, scripted 60 s walk) | JSC: first person 1.21 µs, third person 2.76 µs · V8: 1.69 µs, 3.67 µs — 0.02 % of a 60 Hz frame |
| V8 allocation per tick (Node, GC-free windows, positive control) | first person 3.8 KB, third person 5.4 KB — mostly boxed doubles inside `elevation()` |
| In game, 30 s scripted run on `/game`, Medium, 1280×720 (`bun run bench --suites=play`) | first person: 60.0 ticks/s, 28.7 frames/s, GPU p50 18.9 ms, 376 KiB/s allocated · third person: 59.9 ticks/s, 28.8 frames/s, GPU p50 16.8 ms, 996 KiB/s (0.7–1.1 MB/s across sessions); DRS 2 steps, no reversals in both. The views differ, so these GPU times are not a like-for-like comparison — the first rows are |
| Bundle, `/game` first-load JS (raw / gzip) | 535.5 / 163.6 → 538.5 / 164.9 KiB (HUD: stamina wheel, controls, view button) |
| Bundle, engine chunk (lazy, fetched in parallel with hydration) | 60.3 / 18.8 → 81.8 / 27.1 KiB |

Two findings from getting there:

- **A dynamic index into a uniform array cost 8 % everywhere.** One `u_hero_j[i]` in the material code
  made the whole hero program ~8 % slower even with the hero disabled (ANGLE/Metal); constant
  indices removed it completely (0.995 vs 1.074). The test suite now forbids dynamic indexing of the
  joint array.
- **Chrome allocates more than Node in third person:** ≈ +0.4–0.7 MB/s in game against the +90 KiB/s
  the Node microbenchmark predicts. The same code bundled by Bun (unminified) and run in Chrome shows
  it too — ≈ 6 KB per tick, 1.8 KB of it in `poseGround` alone — so it is V8's tiering in the page,
  not the production minifier: code called once per frame takes minutes to reach full optimization
  (Node's benchmark calls it 60,000+ times), and until then the small float helpers (`set`, `ik`, …)
  are not inlined into the large pose functions and their double arguments are boxed. It is
  short-lived garbage for the young-generation collector, not frame time (third person holds the
  same frame rate); JSC (Safari) does not box these doubles. Restructuring for V8's inliner is the
  lever if it ever matters; one attempt (a dispatch table) changed nothing measurable and was not
  kept.
