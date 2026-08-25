##
##      terrain_math_fallback.gd — pure GDScript CPU twin (spec §27/§31)
##      ====================================================================
##      Identical API and identical numbers to the TerrainCPU GDExtension
##      (gdextension/src/terrain_cpu.h) and the WebGL2 harness
##      (src/lib/terrainMath.ts + demoShader.ts). GDScript floats are 64-bit
##      doubles, so this is a full-precision twin with ZERO dependencies —
##      the project runs before you compile the extension.
##
##      UINT32 PARITY: GDScript ints are 64-bit, so every product is reduced
##      mod 2^32 via a 16-bit split multiply (_mul32) that never overflows
##      int64. Hashes are therefore bit-identical to the C++/GLSL versions.
##
extends RefCounted

# ---- contract constants (MIRROR constants.ts / terrain_cpu.h) ---------------
const CHUNK_SIZE := 8192.0
const MAX_WORLD_H := 1300.0
const MAX_DIST := 6000.0

const F_MACRO := 512.0 / 1048576.0
const F_RIDGE := 341.0 / 1048576.0
const F_MASK := 128.0 / 1048576.0
const F_WARP := 655.0 / 1048576.0
const F_VALLEY := 1942.0 / 1048576.0
const F_CONT := 64.0 / 1048576.0
const F_MOIST := 512.0 / 1048576.0
const F_DUNE := 23.0 / 1048576.0

const OCT_MACRO := 9
const LAC_MACRO := 2.0
const GAIN_MACRO := 0.5
const H_BASE := 90.0
const OCT_RIDGE := 5
const LAC_RIDGE := 2.17
const GAIN_RIDGE := 0.48
const H_MOUNT := 880.0
const MASK_LO := 0.28
const MASK_HI := 0.60
const OCT_WARP := 2
const S_WARP := 620.0
const C_DEPTH := 48.0
const W_VALLEY := 0.42
const W_LEVEL := 5.0
const W_WIDTH := 0.18
const CONT_LO := 0.22
const CONT_HI := 0.44
const CONT_MIN := 0.2
const REGION_CELL := 8192.0
const REG_BLEND_LO := 0.04
const REG_BLEND_HI := 0.18
const REG_HILLS := 0.45
const REG_MTN := 1.3
const REG_DUNE := 0.85
const DUNE_H := 42.0
const ANOM_CELL := 2048.0
const P_ANOM := 0.06
const ANOM_R_IN := 380.0
const ANOM_R_OUT := 620.0
const TERRACE_STEP := 70.0
const TERRACE_H := 15.0
const SPIRE_H := 250.0
const SPIRE_R := 150.0
const SPIRAL_AMP := 16.0
const SPIRAL_TURNS := 6.0
const SPIRAL_RAD := 0.05
const T_BASE := 14.0
const L_LAT := 1.2e-5
const L_ALT := 0.02
const M_BASE := 0.5
const M_AMP := 0.40
const M_OCEAN := 0.35

const LAMBDA_STEP := 0.8
const DELTA_MIN := 0.05
const BISECT_STEPS := 6
const STEP_GROWTH := 0.0015
const START_T := 0.4

const PLAYER_HEIGHT := 1.7
const GROUND_BIAS := 0.08
const GRAVITY := 22.0
const WALK_SPEED := 6.0
const SPRINT_MUL := 1.8
const JUMP_V := 8.5
const FLY_SPEED := 40.0
const SLOPE_MAX := 1.5
const SLIDE_ACCEL := 14.0


# ===========================================================================
#  §4 — deterministic hashing (uint32, Murmur3 finalizer)
# ===========================================================================

## (a·b) mod 2^32 via 16-bit split — every intermediate < 2^33 (int64-safe).
static func _mul32(a: int, b: int) -> int:
        var ah := a >> 16
        var al := a & 0xFFFF
        var bh := b >> 16
        var bl := b & 0xFFFF
        var cross := (al * bh + ah * bl) & 0xFFFF  # only bits < 32 survive
        return ((al * bl) + (cross << 16)) & 0xFFFFFFFF


static func hash_u32(x: int) -> int:
        var h := x & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        h = _mul32(h, 0x85ebca6b)
        h = (h ^ (h >> 13)) & 0xFFFFFFFF
        h = _mul32(h, 0xc2b2ae35)
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        return h


static func hash3(xi: int, yi: int, seed: int) -> int:
        # xi may be negative: & 0xFFFFFFFF takes the two's-complement low bits —
        # exactly the uint32 pattern the C++/GLSL versions hash.
        var h := (xi & 0xFFFFFFFF) \
                ^ _mul32(yi & 0xFFFFFFFF, 0x27d4eb2f) \
                ^ _mul32(seed & 0xFFFFFFFF, 0x9e3779b9)
        return hash_u32(h)


static func hash01(xi: int, yi: int, seed: int) -> float:
        return hash3(xi, yi, seed) / 4294967295.0


static func cell_point(ix: int, iy: int, seed: int) -> Vector2:
        var h := hash3(ix, iy, seed)
        return Vector2(float(h & 0xFFF) / 4095.0, float((h >> 12) & 0xFFF) / 4095.0)


static func cell_type(ix: int, iy: int, seed: int) -> float:
        return float(hash3(ix, iy, seed) >> 24) / 255.0


# ===========================================================================
#  §5 — quintic value noise + Nyquist weights
# ===========================================================================

static func quintic(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


static func vnoise(x: float, y: float, seed: int) -> float:
        var ix := floorf(x)
        var iy := floorf(y)
        var fx := x - ix
        var fy := y - iy
        var ux := quintic(fx)
        var uy := quintic(fy)
        var xi := int(ix)
        var yi := int(iy)
        var a := hash01(xi, yi, seed)
        var b := hash01(xi + 1, yi, seed)
        var c := hash01(xi, yi + 1, seed)
        var d := hash01(xi + 1, yi + 1, seed)
        # identical bilinear expansion as GPU/C++ (NOT lerp(lerp)) — bit parity
        var v := a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
        return v * 2.0 - 1.0


static func nyq_w(f: float, dp: float) -> float:
        return clampf(2.0 - (f * dp) / 0.5, 0.0, 1.0)


static func fbm(x: float, y: float, seed: int, octaves: int,
                lacunarity: float, gain: float, dp: float, f0: float) -> float:
        var freq := 1.0
        var amp := 1.0
        var sum := 0.0
        var wsum := 0.0
        var asum := 0.0
        for i in octaves:
                var w := nyq_w(f0 * freq, dp)
                sum += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37)
                wsum += amp * w
                asum += amp
                freq *= lacunarity
                amp *= gain
        return sum * (asum / maxf(wsum, 1e-6))


static func ridged(x: float, y: float, seed: int, octaves: int,
                lacunarity: float, gain: float, dp: float, f0: float) -> float:
        var freq := 1.0
        var amp := 0.5
        var sum := 0.0
        var wsum := 0.0
        var asum := 0.0
        for i in octaves:
                var w := nyq_w(2.0 * f0 * freq, dp)  # ridged → 2f (§5n)
                var n := 1.0 - absf(vnoise(x * freq, y * freq, seed + i * 0x85eb))
                var r := n * n
                sum += amp * w * r
                wsum += amp * w
                asum += amp
                freq *= lacunarity
                amp *= gain
        return sum * (asum / maxf(wsum, 1e-6))


# ===========================================================================
#  §7/§12/§13 — cellular (Worley) fields
# ===========================================================================

static func voronoi(x: float, y: float, seed: int) -> Dictionary:
        var ipx := floorf(x)
        var ipy := floorf(y)
        var fx := x - ipx
        var fy := y - ipy
        var f1 := 8.0
        var f2 := 8.0
        var cx := 0.0
        var cy := 0.0
        var ctype := 0.0
        for j in range(-1, 2):
                for i in range(-1, 2):
                        var cxi := int(ipx) + i
                        var cyi := int(ipy) + j
                        var p := cell_point(cxi, cyi, seed)
                        var dx := float(i) + p.x - fx
                        var dy := float(j) + p.y - fy
                        var d := sqrt(dx * dx + dy * dy)
                        if d < f1:
                                f2 = f1
                                f1 = d
                                cx = float(cxi) + p.x
                                cy = float(cyi) + p.y
                                ctype = cell_type(cxi, cyi, seed)
                        elif d < f2:
                                f2 = d
        return {"f1": f1, "f2": f2, "edge": f2 - f1, "cx": cx, "cy": cy, "type": ctype}


static func worley_edge(x: float, y: float, seed: int) -> float:
        return voronoi(x, y, seed)["edge"]


# ===========================================================================
#  §14 / §12 / §13 — warp, anomalies, regions
# ===========================================================================

static func domain_warp(x: float, y: float, seed: int, dp: float) -> Vector2:
        var wx := x * F_WARP
        var wy := y * F_WARP
        var qx := fbm(wx, wy, seed + 11, OCT_WARP, 2.0, 0.5, dp, F_WARP)
        var qy := fbm(wx + 5.2, wy + 1.3, seed + 37, OCT_WARP, 2.0, 0.5, dp, F_WARP)
        return Vector2(x + S_WARP * qx, y + S_WARP * qy)


static func anomaly_e(x: float, y: float, seed: int) -> float:
        var v := voronoi(x / ANOM_CELL, y / ANOM_CELL, seed + 400)
        if v["type"] >= P_ANOM:
                return 0.0
        var cx: float = v["cx"] * ANOM_CELL
        var cy: float = v["cy"] * ANOM_CELL
        var dx := x - cx
        var dy := y - cy
        var r := sqrt(dx * dx + dy * dy)
        if r >= ANOM_R_OUT:
                return 0.0
        var theta := atan2(dy, dx)
        var fall := 1.0 - sstep(ANOM_R_IN, ANOM_R_OUT, r)
        var t := r / TERRACE_STEP
        var steps := (floorf(t) + sstep(0.3, 0.7, t - floorf(t))) * TERRACE_H
        var spire := SPIRE_H * exp(-(r / SPIRE_R) * (r / SPIRE_R))
        var spiral := SPIRAL_AMP * sin(SPIRAL_TURNS * theta + SPIRAL_RAD * r)
        return fall * (steps + spire + spiral)


static func region_field(x: float, y: float, seed: int) -> Dictionary:
        var v := voronoi(x / REGION_CELL, y / REGION_CELL, seed + 900)
        var w_region := sstep(REG_BLEND_LO, REG_BLEND_HI, v["edge"])
        var h: float = v["type"]
        var mount_factor := REG_HILLS if h < 0.33 else (REG_MTN if h < 0.66 else REG_DUNE)
        var dune_amount := sstep(0.66, 0.8, h) * w_region
        return {"wRegion": w_region, "mountFactor": mount_factor, "duneAmount": dune_amount}


static func dunes_e(x: float, y: float, seed: int, dp: float, amount: float) -> float:
        if amount <= 0.0:
                return 0.0
        var phase := x * F_DUNE * 6.28318530718 \
                + 3.0 * fbm(x * F_WARP, y * F_WARP, seed + 301, 2, 2.0, 0.5, dp, F_WARP)
        var c := 1.0 - absf(sin(phase))
        return DUNE_H * amount * c * c * c


static func sstep(e0: float, e1: float, x: float) -> float:
        var t := clampf((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)


# ===========================================================================
#  Master elevation field — spec §2 composition
# ===========================================================================

func get_elevation(x: float, z: float, seed: int) -> float:
        return elevation(x, z, seed, 0.0)


static func elevation(x: float, z: float, seed: int, dp: float) -> float:
        var wq := domain_warp(x, z, seed, dp)                       # §14
        var reg := region_field(x, z, seed)                         # §13 (raw)

        var cont := sstep(CONT_LO, CONT_HI, \
                0.5 + 0.5 * vnoise(x * F_CONT, z * F_CONT, seed + 801))  # §9

        var base := H_BASE * (0.5 + 0.5 * fbm(                      # §5
                wq.x * F_MACRO, wq.y * F_MACRO, seed + 1,
                OCT_MACRO, LAC_MACRO, GAIN_MACRO, dp, F_MACRO))

        var mask := sstep(MASK_LO, MASK_HI, \
                0.5 + 0.5 * vnoise(wq.x * F_MASK, wq.y * F_MASK, seed + 77))
        var ridge := ridged(wq.x * F_RIDGE, wq.y * F_RIDGE, seed + 5,  # §6
                OCT_RIDGE, LAC_RIDGE, GAIN_RIDGE, dp, F_RIDGE)
        var h_mount: float = H_MOUNT * (1.0 + (reg["mountFactor"] - 1.0) * reg["wRegion"])

        var de: float = voronoi(wq.x * F_VALLEY, wq.y * F_VALLEY, seed + 21)["edge"]
        var sv := 1.0 - sstep(0.0, W_VALLEY, de)                    # §7
        var valley := C_DEPTH * sv * sv

        var dune := dunes_e(x, z, seed, dp, reg["duneAmount"])      # §13 (raw)
        var anom := anomaly_e(x, z, seed)                           # §12 (raw)

        return base * (CONT_MIN + (1.0 - CONT_MIN) * cont) \
                + h_mount * mask * ridge \
                + dune \
                - valley \
                + anom


static func gradient(x: float, z: float, seed: int, eps: float) -> Vector2:
        var e := maxf(eps, 0.05)
        var gx := (elevation(x + e, z, seed, 0.0) - elevation(x - e, z, seed, 0.0)) / (2.0 * e)
        var gz := (elevation(x, z + e, seed, 0.0) - elevation(x, z - e, seed, 0.0)) / (2.0 * e)
        return Vector2(gx, gz)


# ===========================================================================
#  Public API — mirrors the TerrainCPU GDExtension exactly
# ===========================================================================

func get_gradient(x: float, z: float, seed: int) -> Vector2:
        return gradient(x, z, seed, 0.25)


func get_normal(x: float, z: float, seed: int) -> Vector3:
        var g := gradient(x, z, seed, 0.25)
        var len := sqrt(g.x * g.x + g.y * g.y + 1.0)
        return Vector3(-g.x / len, 1.0 / len, -g.y / len)


func get_climate(x: float, z: float, seed: int) -> Vector2:
        var e := elevation(x, z, seed, 0.0)
        var t := T_BASE - L_LAT * absf(z) - L_ALT * maxf(e, 0.0)
        var n := fbm(x * F_MOIST, z * F_MOIST, seed + 61, 2, 2.0, 0.5, 0.0, F_MOIST)
        var cont := sstep(CONT_LO, CONT_HI, \
                0.5 + 0.5 * vnoise(x * F_CONT, z * F_CONT, seed + 801))
        return Vector2(t, M_BASE + M_AMP * n + M_OCEAN * (1.0 - cont))


func query_anomaly(x: float, z: float, seed: int) -> Dictionary:
        var out := {"found": false}
        var cell_x := floorf(x / ANOM_CELL)
        var cell_z := floorf(z / ANOM_CELL)
        var best := 1e18
        var bx := 0.0
        var bz := 0.0
        var found := false
        for j in range(-2, 3):
                for i in range(-2, 3):
                        var ix := int(cell_x) + i
                        var iz := int(cell_z) + j
                        if cell_type(ix, iz, seed + 400) >= P_ANOM:
                                continue
                        var p := cell_point(ix, iz, seed + 400)
                        var cx := (float(ix) + p.x) * ANOM_CELL
                        var cz := (float(iz) + p.y) * ANOM_CELL
                        var d := sqrt((cx - x) * (cx - x) + (cz - z) * (cz - z))
                        if d < best:
                                best = d
                                bx = cx
                                bz = cz
                                found = true
        out["found"] = found
        if found:
                out["x"] = bx
                out["z"] = bz
                out["y"] = elevation(bx, bz, seed, 0.0)
                out["distance"] = best
        return out


func march_ray(from: Vector3, dir: Vector3, seed: int, steps: int,
                footprint: float, max_dist: float) -> Dictionary:
        var d := dir.normalized()
        var md := MAX_DIST if max_dist <= 0.0 else max_dist
        var t := START_T
        var t_prev := t
        var grad := 0.0
        for i in steps:
                var px := from.x + d.x * t
                var py := from.y + d.y * t
                var pz := from.z + d.z * t
                if py > MAX_WORLD_H and d.y > 0.0:
                        return {"hit": false, "t": t, "position": from + d * t}
                var dp := maxf(t * footprint, 1e-4)
                var e := elevation(px, pz, seed, dp)
                if py < e:
                        # bisection refinement (§16)
                        var lo := t_prev
                        var hi := t
                        for b in BISECT_STEPS:
                                var tm := 0.5 * (lo + hi)
                                var gap := (from.y + d.y * tm) \
                                        - elevation(from.x + d.x * tm, from.z + d.z * tm, seed, tm * footprint)
                                if gap < 0.0:
                                        hi = tm
                                else:
                                        lo = tm
                        var th := 0.5 * (lo + hi)
                        return {"hit": true, "t": th, "position": from + d * th}
                if i % 2 == 0:
                        var eps := maxf(dp, 0.15)
                        var ex := elevation(px + eps, pz, seed, dp)
                        var ez := elevation(px, pz + eps, seed, dp)
                        grad = sqrt(((ex - e) / eps) * ((ex - e) / eps) + ((ez - e) / eps) * ((ez - e) / eps))
                t_prev = t
                var step := maxf(DELTA_MIN, (py - e) / sqrt(grad * grad + 1.0) * LAMBDA_STEP)
                step *= 1.0 + t * STEP_GROWTH
                step = minf(step, t * 0.06 + 2.0)
                t += step
                if t > md:
                        break
        return {"hit": false, "t": t, "position": from + d * t}


## §31 movement loop verbatim — jump-preserving clamp (§31), GROUND_BIAS,
## slope slide-off gate. `state` and `input` are dictionaries of doubles.
func update_player(state: Dictionary, input: Dictionary, dt: float, seed: int) -> Dictionary:
        var speed := WALK_SPEED * (SPRINT_MUL if input.get("sprint", false) else 1.0)
        var yaw: float = input.get("yaw", 0.0)
        var fwd: float = input.get("forward", 0.0)
        var str: float = input.get("strafe", 0.0)
        var sin_y := sin(yaw)
        var cos_y := cos(yaw)
        var wish_x := (sin_y * fwd + cos_y * str) * speed
        var wish_z := (cos_y * fwd - sin_y * str) * speed

        var s := {
                "x": state.get("x", 0.0), "y": state.get("y", 0.0), "z": state.get("z", 0.0),
                "vx": state.get("vx", 0.0), "vy": state.get("vy", 0.0), "vz": state.get("vz", 0.0),
                "grounded": state.get("grounded", false),
        }

        var k := 1.0 - exp(-12.0 * dt)
        s["vx"] += (wish_x - s["vx"]) * k
        s["vz"] += (wish_z - s["vz"]) * k

        if input.get("jump", false) and s["grounded"]:
                s["vy"] = JUMP_V

        s["vy"] -= GRAVITY * dt                    # gravity ALWAYS integrates
        s["x"] += s["vx"] * dt
        s["y"] += s["vy"] * dt
        s["z"] += s["vz"] * dt

        var ground := elevation(s["x"], s["z"], seed, 0.0) + GROUND_BIAS + PLAYER_HEIGHT
        s["grounded"] = false
        if s["y"] < ground:
                s["y"] = ground
                if s["vy"] < 0.0:
                        s["vy"] = 0.0                       # stop FALLING only — preserve jumps
                s["grounded"] = true

        if s["grounded"]:
                var g := gradient(s["x"], s["z"], seed, 0.25)
                var m := sqrt(g.x * g.x + g.y * g.y)
                if m > SLOPE_MAX:
                        var inv := 1.0 / m
                        s["vx"] += -g.x * inv * (m - SLOPE_MAX) * SLIDE_ACCEL * dt
                        s["vz"] += -g.y * inv * (m - SLOPE_MAX) * SLIDE_ACCEL * dt

        return s


func get_constants() -> Dictionary:
        return {
                "chunk_size": CHUNK_SIZE, "max_world_h": MAX_WORLD_H, "max_dist": MAX_DIST,
                "player_height": PLAYER_HEIGHT, "ground_bias": GROUND_BIAS,
                "gravity": GRAVITY, "walk_speed": WALK_SPEED, "sprint_mul": SPRINT_MUL,
                "jump_velocity": JUMP_V, "slope_max": SLOPE_MAX,
                "fly_speed": FLY_SPEED,
                "lambda_step": LAMBDA_STEP, "delta_min": DELTA_MIN,
                "bisect_steps": BISECT_STEPS, "water_level": W_LEVEL,
        }
