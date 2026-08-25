# W(x,y) — Continuous World Engine (Godot 4)

A *Tears of the Kingdom*-style exploration terrain implemented as **pure
mathematics**: the world is a continuous 2D field `E(x, z)` evaluated by
per-pixel raymarching on the GPU and by a double-precision twin on the CPU.
**There is no terrain mesh.** Every pixel, every collision, every shadow and
every biome colour is derived deterministically from one `uint32` hash chain.

Target platform: **Apple Silicon (arm64), Godot 4 Forward+, Metal (TBDR),
SDFGI**.

---

## Quick start

```bash
# 1. Open this folder in Godot 4.3+ (Forward+ renderer — required for SDFGI).
# 2. Press F5. The GDScript physics twin runs out of the box.

# Optional — build the C++ CPU twin (the shipping path, ~10× faster queries):
cd gdextension
git clone -b 4.3 https://github.com/godotengine/godot-cpp godot-cpp
scons platform=macos arch=arm64 target=template_debug
scons platform=macos arch=arm64 target=template_release
```

### Controls

| Key | Action |
| --- | --- |
| `W A S D` | Move (camera-relative) |
| `Shift` | Sprint (×1.8) |
| `Space` | Jump — preserved by the contact clamp (§31) |
| `F` | Toggle walk / fly |
| `C` | Descend (fly mode) |
| Mouse drag | Look (click canvas to capture) |
| `Esc` | Release mouse |

---

## Project layout

```
godot-project/
├── project.godot                  Forward+ · MSAA off · FXAA on (TBDR)
├── terrain_cpu.gdextension        extension manifest (optional at runtime)
├── scenes/main.tscn               thin root → scripts/main.gd
├── scripts/
│   ├── main.gd                    scene assembly: SDFGI env, sun w/ cull mask,
│   │                              floating origin, terrain, player, HUD, props
│   ├── floating_origin.gd         §25 chunked rebasing (8192 = 2^13)
│   ├── terrain_renderer.gd        fullscreen-quad driver (render_priority −100)
│   ├── player_controller.gd       §31 movement on the CPU twin, §24 camera spring
│   └── terrain_math_fallback.gd   pure-GDScript twin (identical Dictionary API)
├── shaders/
│   └── terrain_raymarch.gdshader  the entire visual world (see header notes)
└── gdextension/
    ├── SConstruct                 arm64 build, -ffp-contract=off (parity flag)
    └── src/
        ├── terrain_cpu.h          the double-precision reference twin
        ├── terrain_cpu.cpp        TerrainCPU GDExtension class
        └── register_types.cpp     entry point
```

---

## The parity contract

Five implementations evaluate the **same field**, and all of them agree:

| Implementation | Precision | Hash math | Role |
| --- | --- | --- | --- |
| `gdextension/src/terrain_cpu.h` | `double` globals | `uint32_t` | reference twin, shipping physics |
| `scripts/terrain_math_fallback.gd` | `double` (GDScript floats are 64-bit) | int64 masked to 32 bits | zero-dependency fallback |
| `shaders/terrain_raymarch.gdshader` | `float` chunk-local + fp32-exact origin | GLSL `uint` | the visible world |
| web harness `src/lib/terrainMath.ts` | `double` | `Math.imul` | browser demo + worked example (§30) |
| web harness `demoShader.ts` (GLSL ES 3.0) | `float` chunk-local | GLSL `uint` | WebGL2 parity proof |

* **Hashing** is strict `uint32` integer math everywhere (MurmurHash3
  finalizer over `xi ⊕ (yi·0x27D4EB2F) ⊕ (seed·0x9E3779B9)`), so the noise
  lattice is bit-identical across CPU/GPU/GDScript/JS.
* **Frequencies** are dyadic rationals `n/2^20` and the floating-origin chunk
  is `2^13`, so `origin · f` is exactly representable in fp32 — noise cells
  align across chunk rebases with zero jitter (§25/§27).
* **Divergence budget**: the fp32 GPU vs fp64 CPU paths differ by < 1 mm on a
  700 m peak; `GROUND_BIAS = 0.08 m` absorbs it in the contact clamp (§31).
* **`-ffp-contract=off`** in SConstruct: FMA contraction differs between the
  CPU compiler and the Metal toolchain — leave the flag in place.

## Key engineering decisions

* **No mesh, no physics server** — the shader writes `DEPTH = clip.z/clip.w`
  (reversed-Z, inherited from Godot's Metal NDC) and `NORMAL` in view space,
  so ordinary meshes inter-occlude with the terrain *and* SDFGI voxelizes it.
* **Sun lighting is custom** (`EMISSION`) — the DirectionalLight's
  `light_cull_mask` excludes the terrain's layer to prevent double lighting;
  SDFGI indirect still reaches the surface through `ALBEDO`.
* **TBDR discipline** — `render_priority = -100`, `depth_draw_always`,
  MSAA off + FXAA, bounded marches whose per-step cost *falls* with distance
  (Nyquist weights switch octaves off).
* **Jump-preserving clamp** — the twin zeroes vertical velocity only when it
  is negative, so contact never kills an ascending jump (§31).

## Spec → file cross-reference

| Spec section | Where |
| --- | --- |
| §2 elevation composition | `terrain_cpu.h → elevation()`, shader `elevation()` |
| §4 hash | every `hash3()` |
| §5 fBm + Nyquist weights | `fbm()`, `nyq_w()` |
| §6 ridged mountains | `ridged()` (2f Nyquist) |
| §7/§8 valleys + water | `voronoi()` edge carve, `W_LEVEL` plane in march |
| §9 climate | `temperature()` / `moisture()` |
| §10/§11/§21 biome RBF + OKLab | shader `biomeColor()` |
| §12 anomalies | `anomaly_e()`, `query_anomaly()` |
| §13 regions + dunes | `region_field()`, `dunes_e()` |
| §14 domain warp | `domain_warp()` |
| §16 march + bisection + DEPTH | `march_ray()`, shader `fragment()` |
| §17 analytic normals | `gradient()` (ε = pixel footprint) |
| §18 soft shadows | `softShadow()` (4×Δp footprint) |
| §19–§22 fog/sky/clouds | `applyFog()`, `skyColor()`, cloud plane |
| §24 camera smoothing | `player_controller.gd` spring k_s=60 |
| §25 floating origin | `floating_origin.gd` |
| §26 march budgets | quality presets (64/96/128 steps) |
| §27 precision policy | parity contract above |
| §31 CPU physics twin | `update_player()` + `TerrainCPU` API |
| §33 GPU integration | shader header + `terrain_renderer.gd` |

## Validation checklist

1. **DEPTH interop** — walk up to the red proof boxes near spawn: they sit on
   the mathematical surface and depth-sort with it.
2. **SDFGI cascade test** — drop a red box at a cascade boundary and look for
   light leaking; the analytic DEPTH is continuous, so seams indicate cascade
   resolution, not the terrain.
3. **Floating origin** — walk > 4 km in one direction and watch the HUD chunk
   counter: no noise-lattice jitter, no fract() wobble at the rebase.
4. **Parity** — with the extension built, compare `TerrainCPU.get_elevation`
   against the fallback at the same coordinates; the delta is < 1 mm.
