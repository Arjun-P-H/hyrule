# Third-person hero

The `/game` page now starts in third person: a green-tunic hero modelled on the classic *Legend of
Zelda* look, with open-world Zelda locomotion (run, sprint, sneak, jump, paraglide, climb, swim,
shield surf, a stamina wheel) and *Tears of the Kingdom* combat — the sword, the shield, the bow,
throws, dodges, Bokoblin camps — described in [`COMBAT.md`](COMBAT.md). **V** switches to the
existing first-person controller and back; **F** still takes off into free flight and lands in
whichever view was last used on foot.

## Controls (third person)

| Key | Action |
| :--- | :--- |
| WASD / arrows | Run, relative to the camera; the hero turns to face the way it moves |
| Shift | Sprint (stamina); dash stroke while swimming; skydiving: nosedive (let go, or pull S, to flare) |
| Space | Jump · in the air (≥ 2.5 m up): open / fold the paraglider · climbing: climb jump · climbing with S held: leap off |
| C | Sneak · climbing: slide down · on a treetop, stepping to its edge: climb down onto the trunk |
| Mouse (click to lock) | Orbit the camera · wheel: zoom 1.6 – 11 m |
| Left / right button, Q, G, E, X | Attack, shield, bow, throw, action (surf, pick up), next weapon: [`COMBAT.md`](COMBAT.md) |
| I | Inventory: equip weapons, shields and bows, eat food ([`INVENTORY.md`](INVENTORY.md)) |
| V | First person (and back) |
| F | Free flight (F again lands the hero where the camera is) |
| R | Respawn |

These are the default keys. **Engine Settings → Controls → Configure keys** (the sliders button, top right) lists
every control with two slots, a primary and an alternate key: click a slot, press the key wanted (Esc cancels,
Backspace empties the slot). A key does one thing, so one given to a control is taken from the control that had it,
and a control left with no key is shown in amber. The mouse buttons stay on attack and shield. The choice is kept in
the browser (`localStorage`, `hyrule.keybinds`); **Reset** puts the defaults back. The banner, the tooltips and
the settings name the keys as set. The Spec Console's demo always uses the defaults.

**Key Look** (Engine Settings, under Mouse Look; off by default) turns the camera with the keyboard in any view:
*On* gives the four arrow keys to Look up, down, left and right, taking them from the moves they were the
alternates of (WASD still moves); *Off* gives them back. Any other keys can be chosen for the Look controls on the
Controls page, and Key Look shows *On* while any key turns the camera. Up and down tilt the view as far as the
mouse can, two opposite keys cancel, and diagonals work. The turn starts slowly, for fine aim, and reaches full
speed after 0.4 s: 109°/s at the default Mouse Look, which scales it (Low 60°/s, High 188°/s).

## How it is built

Nothing here is a mesh or a texture: the zero-mesh / zero-texture invariant holds.

- **Locomotion** — `src/lib/hero/locomotion.ts`. Same contract as the first-person `updatePlayer()`:
  deterministic, mutated in place, no allocation per tick, terrain queried through the CPU twin's
  `elevation()` on fp64 global coordinates (one sample plus two forward differences per tick). When
  grounded the root sits at `E + GROUND_BIAS`, exactly like the first-person eye minus its height, so
  switching views keeps one collision contract. States: ground (run / sprint / sneak / slide down
  ground steeper than 45°), air, glide, climb (a wall, a rock or a tree: [Climbing](#climbing)), swim (water deeper than 1.3 m: the sea, a lake, a river's pool, from `surfaceAt`; frozen water is walked on, see [`LANDS.md`](LANDS.md)), surf (on the
  shield: gravity along the slope, little friction, the input turning the slide). Constants: `HERO`
  in `constants.ts`. In third person it runs inside the combat world (`combat/world.ts`), which
  steers it through `HeroInput`'s optional fields (a heading to face, a speed factor, no climbing);
  the Bokoblins walk on the same code.
- **Animation** — `src/lib/hero/rig.ts`. An 18-joint skeleton posed every frame: a gait cycle whose
  stance feet move backwards at exactly the ground speed (so they stay planted), placed on the
  terrain plane at the root with two-bone IK for knees and elbows; air (tuck / fall / skydive),
  paraglider, climb, swim and surf poses cross-faded by smoothed weights; lean into acceleration and
  turns, a landing squash, idle breathing; over them the combat layer (arms, gear, swings, guard,
  bow, throws, flinches, spins and flips: [`COMBAT.md`](COMBAT.md)). It then packs the joints, a
  torso and a head frame, a bounding sphere and the gear's frames **camera-relative** into one
  `Float32Array` (fp64 differences before the fp32 store, so the body keeps sub-millimetre
  precision 1,000 km from the origin).
- **Keys** — `src/lib/engine/keybinds.ts` (pure: the controls, their default keys, the rules of binding, saved
  bindings made safe to read) and `keybindStore.ts` (saving, the React hook). The engine holds the keys that are
  down and turns them into the actions polled each frame (a bit mask) or fired once (fly, view, respawn, HUD,
  sound, fast time); a key is the event's `code`, so WASD stays a cluster on any layout, and the left and right
  Shift and Ctrl count as one key each. Ctrl, once only a way down in flight, now also sneaks. The look keys are
  polled like the moves: `keyLook.ts` (pure, tested) turns the camera's yaw and pitch by the time they are held,
  before the view's basis is built, so first person, flight and the orbit camera all take it.
- **Camera** — `src/lib/hero/camera.ts`. Orbit on the mouse; the look-at point trails the hero by at
  most 0.6 m; terrain between the hero and the eye shortens the boom (snap in, ease out); the eye
  stays 0.45 m above the ground and the water; the view is re-aimed from the final eye to the target.
- **Rendering** — `heroMap()` in `demoShader.ts` (and `gpu/hero.wgsl.ts`): capsules, tapered
  capsules, a capped cone (tunic skirt), boxes, a dome shell (paraglider) around the joints, the
  gear (sword or club, shield, bow and arrow) in its own frames, with material ids and procedural
  detail (eyes, belt buckle, shield rim / emblem / crest, boot and gauntlet splits, canopy stripes).
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

## Climbing

Everything climbable is climbed one way, as in *Tears of the Kingdom*: the land's walls (ground steeper than
`HERO.walkSlope`, 45°), the sky islands' steep hills, rocks, and tree trunks. Push into one and the hero
grabs it. On it the stick works on the hero, not the camera: forward climbs (0.95 m/s along the surface),
back climbs down (1.1 m/s), and the sides go across it (round a trunk) at 0.8 m/s at the hips. The pace
eases in and out (`CLIMB.accel`, about 0.4 s to full speed or to rest), so starting, stopping and turning
the stick never jerk the body. A grab's slip and a climb jump move at once. Stamina is spent as before. Code: `climbSurface` (locomotion.ts), `climbGait` and `poseClimb` (rig.ts). Constants:
`CLIMB` in `constants.ts`.

| Situation | What happens |
| :--- | :--- |
| Running into a wall, a rock or a trunk | Grabs it where the hero stands and climbs from its foot, however fast the run |
| A ledge lower than the hands' reach (1.6 m), at least 0.2 m high and level on top | Vaulted straight onto (a low rock, a bank) |
| Jumping at one | Catches it where the jump meets it (the mid-air grab below) |
| Falling or gliding into one (≤ 14 m/s) | Grabs it, sliding v² / 36 m down it (0.05–0.5 m) over 0.18 s as the grip takes |
| Landing on ground too steep to stand on, pushing into it | Grabs it |
| Swimming into a steep bank or a rock | Climbs out of the water up it |
| A thin pole (< 0.35 m) | The hands reach round its sides (up to 110° round on the thinnest), the feet clamp it, the turn is held to 140°/s |
| Space | A climb jump of `HERO.climbJump` the way the stick points along the surface (up when it rests) |
| Space with S held | Leaps off: 5.5 m/s the way the stick points from the camera (straight out from the surface if that is into it), 2.4 m/s up |
| C held, or a dry wheel | Slides down at 3.8 m/s along the surface (Space pushes off). Running dry never drops the hero |
| The top in the hands' reach | The hero vaults onto it (a cubic Bézier over 0.65 s) and stands there. On a treetop it balances within 0.45 m of the axis; a jump or the glider leaves, and C with a step to the edge climbs down onto the trunk |
| A wall easing below `HERO.climbExitSlope` | The hero walks onto it, its feet where they were |
| The ground | Climbing down onto it steps off |

Trunks shorter than 2 m (stumps, saplings) and cacti are not climbed, and Bokoblins never climb.

**Walls and rocks.** The land's ground (in the cloud world, the cloud sea's: its steep heaps are climbed as cliffs are; or a sky island's top) and the rocks standing on it are treated as one
solid under one surface: `solidTop`, their highest top at a point (a rock's top is its own height grid,
`rockTopAt`). The hero holds a wall where that solid comes up to the hips' level, facing it, and moves along
that level's contour across the wall and from level to level up and down it. Every tick it finds the contact
afresh (a march, then a halving to about 2 mm), so the hold never drifts off the wall. Because the hold is at
the hips, on a wall leaning back the root stands in front of the feet's ground. Leaving the wall for the air
or the water, the hero first steps out to where its feet are. Climbing a wall costs about the same CPU as
running (≈ 0.2 ms a tick here): the wall's curvature and lean are measured again only once the hold has
moved 0.15 m, and the top and the ground are first checked with one sample each.

**The trunk.** A tree's trunk is not a fixed circle. `trunkColliders()` (forest/models.ts) gives each
species and shape a profile of 8 knots, the mesh's own rings at their widest, from knee height to the top.
`trunkSection()` (forest/collision.ts) interpolates the centre and radius at any height, so the climb
follows the taper and the lean (a 0.5 m trunk is often 0.07 m at its top). Climbing, the root sits on the
bark. Each tick moves its angle round the axis (ω = v / (R + clearance), the clearance 0.22–0.38 m by
radius) and its height, then puts it back on the bark from the cross-section there, so no error builds up
however long the hero circles (3 × 10⁻¹³ m measured). Near the ground, bumping into a trunk still uses the
conservative circle that covers flare, buttress fins and lean from knee to shoulder; higher up, the
cross-section is used.

**The animation** (rig.ts `climbGait`, `poseClimb`) is the same on every surface. Locomotion hands the rig
the surface round the hips as a cylinder that tapers and leans: a trunk's own cross-section, or a wall's
contour (its radius where it bulges toward the hero, a flat 30 m otherwise; its lean from the contours
0.4 m above and below). Snapshots carry the same numbers, so other players' rigs need nothing else.

- **Holds.** Four holds are world points on the surface, so a planted hand or foot stays put however the
  body moves past it. They are placed on the cylinder at their own height, with a slope above the hips (to
  where the hands hold) and one below (to where the feet do), since a trunk's profile bends. On the
  hero's own climb each new hold is then moved onto the surface as it really is there
  (`snapToClimbed`), since a boulder's knobbly side is no cylinder.
- **Diagonal pairs.** Left hand with right foot, right hand with left foot. A pair steps at the spec's
  MaxLimbStretch, a hand 0.52 m and a foot 0.56 m from its socket, to land 0.26 m past it, or once a hold
  trails round past the furthest the limb wraps.
- **Taking turns.** The pairs alternate: the one that did not step last goes at 70 % of due. The next pair
  may begin as a step comes down onto its hold (82 % through), while its own limbs still hold through their
  pre-load, so one pair is always on the bark.
- **Step phases.** Each step runs the spec's four phases: pre-load, peel-off (heel or wrist first), a
  swing round the bark, then contact with an 18 mm slip. Every step takes the spec's 0.28 s; at the slower pace there is more
  rest between steps, not slower steps.
- **Swings.** A swing is interpolated round the axis on the trunk's taper and lean, so it follows the
  bark's curve rather than cutting through it.
- **Body.** The pelvis sits at the spec's weighted centre: 0.35 of the feet, 0.15 of the hands, 0.5 of the
  root. It shifts 7.5 cm onto the support side and dips 3.5 cm through a step, and rolls toward the higher
  foot, with the spine counter-bent. Knees and elbows flare out on pole vectors (the frog stance), and the
  toes point at the axis.
- **The body follows its holds.** Spec strides at the climbing pace leave one side's hand high above while
  its foot is still far below: about 0.26 m over the hand's socket and 0.56 m under the foot's. No body held
  at the weighted centre spans that. So, as a climber does, the body goes to the holds. The chest (the spine
  pivoting on the pelvis) moves toward a hand out of reach and the pelvis toward a foot, each pulling the
  other a little, both hugging in toward the bark but never nearer than 0.12 m to it. This is relaxed 12
  times a frame (`reachHolds`), and climbing down the hands ride 0.25 m lower. On a wall leaning out
  over the hero the hands sit lower and the chest hugs closer, and the clearance is measured along the
  surface's normal, so an overhang does not push the body off it.
- **Smoothing.** The pelvis and chest follow their targets on critically damped springs (the reach's pull
  too), and a wall's normal is blended from tick to tick, so a body never pops when a hold is replaced or
  the contour turns; a hard reach pass after the springs keeps every planted limb on its hold.
- **Fast moves.** A climb jump, a slide or a grab's slip carry the limbs with the body.

Measured on three trunks (radius 0.51, 0.28 and 1.42 m), climbing up, round, diagonally and down over 6,147
settled frames (a terrain wall and a 2.6 m boulder give the same: no slides, no straight limbs, no limb in
the solid):

| Measure | Result |
| :--- | :--- |
| Mean stride (hold to hold): up · down · round the trunk | 0.67 · 0.63 · 0.44 m (round: as far as a hand wraps round a thin trunk) |
| Planted holds sliding more than 1 mm | none |
| Frames with an arm or leg pulled straight | none |
| Both pairs off the bark | never |
| Closest a hand or ankle joint gets to the bark (the trunk's real cross-section at its height) | 4.5 cm outside it |
| Body acceleration (second difference a frame), p95 · max: tree · wall · boulder | 9 · 39, 4 · 56, 8 · 53 mm (was 29 · 72, 26 · 763, 34 · 127) |

**The camera** draws back 0.45 × the trunk's radius while climbing (eased). **Other players** see the same
climb: a snapshot carries the surface's cylinder (centre, radius, top, taper and lean up and down) and the
vault's progress, appended after the bow (an older build's snapshot has none: held as a flat wall). This raises `server.mjs`'s
`MAX_SNAPSHOT` to 48 numbers, so an older relay drops the new snapshots: update it with the client.

**Adapted from the Godot spec.** The spec this follows was written for a physics engine:

- **No sensor rig.** The 14-cast sensor array is replaced by exact geometry: every trunk here is an
  analytic, profiled cylinder, and every wall and rock is the surface of one solid that can be sampled
  anywhere.
- **States.** Its ten states map onto the existing `climb` move: a trunk, a grip's slip, a surge, a vault,
  or a slide.
- **Stride.** The spec's thresholds, lead and step time are kept. Its rigid body pose could not reach the
  holds they make (a foot 0.56 m under its socket while the same side's hand is over its own), so the body
  follows the holds (above). The spec's sockets (hands 1.55 m, feet 0.45 m over the root) are 1.3 m and
  0.5 m here, centred on this 1.7 m hero's shoulders and hips.
- **Pole wrap.** It is held to what the arms reach (0.1 m behind a pole's front), which is 110° only on the
  thinnest poles.

Not built:

- **Branches.** Mantling onto, balancing along and jumping between horizontal branches: crowns and limbs
  have no colliders.
- **Overhangs and roots.** Trunks are vertical, and the land and rocks are height fields: no overhangs, no
  undercuts, no island undersides.
- **Ruins and Eyrie buildings.** Their walls stop the hero but are not climbed.
- **Collider morphing and foliage dithering.** There are no capsules or physics layers here.

Tests: `tests/hero.test.ts` (*climbing walls and rocks*, *climbing trees*, and the rig's checks on a trunk,
a pole, a wall and a boulder), `tests/multiplayer.test.ts`.

The extents that bound `heroMap()` around each joint (`HERO_SHAPE.*Reach`, `limbPad`) are mirrored
as `HERO_*` constants in the shader and pinned by `tests/hero.test.ts`; the CPU builds the bounding
sphere from them.

## Puffs

In the cloud world small still clouds stand in ladders up the islands (`skyworld/puffs.ts`): their tops
are floor (`realm.ts skyFloorUnder`). A plain puff is a jump up from the last; a springy one (a warm glow round its
edge) throws the hero straight back up on landing, 10–20 m (`locomotion.ts land`, `springSpeed`: the launch speed whose
top, with the jump's hang near the apex, is the lift), with no roll or squat.

## Skydiving

A fall with at least 8 m of air under the feet, once faster than 8.5 m/s, spreads into a skydive (a sub-state of
`'air'`: `HeroState.skydive`). The posture `dive` eases (8/s) between a belly-to-earth spread (0) and a head-first
nosedive (1, Shift held). Vertical drag is quadratic, k = g / Vt², toward 16.5 m/s spread and 42 m/s diving,
integrated semi-implicitly (`vy' = (vy − g·dt) / (1 + k|vy|·dt)`: steady at Vt exactly, stable at any dt). Letting go
of a dive faster than 20 m/s, or pulling back, flares for up to 0.38 s toward 8.5 m/s, braking at most 52 m/s². Across
the ground the stick steers at up to 14.5 m/s spread, 2.5 m/s tucked (turning to face it at 4.2/s, 1.5/s), and the spread body glides forward on its own
(0.28·|vy|·sin 2θ along its heading). Space still opens the paraglider. Constants: `SKYDIVE` in `constants.ts`.

The rig (`poseSkydive`) builds the body upright from control surfaces — spine arch and twist, differential arm sweep
and abduction (the outside arm out into the wind, the inside one tucked; in the nosedive the arms lie straight and
parallel along the sides), leg spread and scissor, knee and elbow
fold — then tips it about its middle (82° spread, 160° head first, less in a flare) and banks it into turns
(tan θ = ω·v / g × 1.35, ≤ 52°). Hands, forearms, calves and shoulders flutter at 14.5 and 23 Hz, scaled by the
dynamic pressure (v / 42)², never under half its full strength while skydiving (`FLUTTER_FLOOR`: ≈ 6 cm at the
hands, 3 cm at the feet in the spread; the thighs flutter too). Steering is read off the motion (yaw rate, speed ahead), so another player's hero,
played back, poses the same; the snapshot carries `skydive`, the flare and `dive`. In the nosedive the neck extends so the face looks
down the fall at the ground, the back stays gently arched and the knees bend, the shins trailing (`DIVE_*` in
`rig.ts`: the face is level with the fall when tilt + arch − neck = 90°). Within 0.42 s of the ground the
pose blends back to the upright fall, feet first.

Landing: the pelvis dips on an underdamped spring (ωn 16 rad/s, ζ 0.62, depth 0.024·impact, 8–48 cm). From
12 m/s (a 3.3 m drop: the spec's 8 m/s at 9.81, as the same height under the land's 22 m/s²) while moving, the hero
rolls on at 6.8 m/s for 0.55 s; from 33 m/s (a 25 m drop: a nosedive) it squats 0.68 s, braking to a stop.

The camera widens the field of view from 72° to 104° (as a factor on the setting) between 10 and 42 m/s, draws the
boom in to 2.15 / 3.1 of the zoom in a nosedive, and tips the view toward looking down the fall (−68°, −82° diving)
unless the mouse or the look keys moved it in the last 0.7 s.

Not built from the spec: wind volumes and updrafts (the world has none), the aerial dodge and corkscrew roll (no
dodge key in the air), the mid-air wall kick (every wall is climbable here). Mid-air trunk and wall catches are the
climb's own (`latchSpeed` 14 m/s still applies). Standard air control keeps its tuning; a jump's top hangs at 0.55 g
while |vy| < 1.8 m/s.

## Tests

`tests/hero.test.ts` (58 tests) runs everything on the real terrain around the spawn:

- **Locomotion** — spawn on the ground (or afloat); run speed and camera-relative heading; diagonals
  not faster; sneak speed with exact ground contact; one jump per press with a v²/2g apex; sprint
  drains one wheel in 1/`staminaSprint` s, then exhaustion (no sprint, slow jog) until the wheel is
  full again, recovery only after the rest delay; paraglider opens only on a mid-air press with
  clearance, sinks at `glideSink`, folds on a second press or a dry wheel, terminal velocity on long
  drops; climbing (walls, rocks and trees, [Climbing](#climbing)) grabs, climbs at `CLIMB.upSpeed` along the
  surface holding it within 2 cm, goes across it, slides down it, jumps and leaps off, vaults onto tops and
  low rocks, catches a wall falling past it, steps off at the bottom, never re-grabs while exhausted;
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
  `HERO_UNIFORM` (with the gear's `u_hero_gear[9]`); the hero is gated on `u_hero_bound.w`; `softShadow`, `heroShadow` and `heroTrace`
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
