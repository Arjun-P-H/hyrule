##
##      PlayerController — §31 movement loop on the CPU twin
##      ====================================================================
##      The GPU surface is pure math; the engine physics servers cannot see it,
##      so movement runs on the CPU twin (TerrainCPU GDExtension, or the pure
##      GDScript fallback — identical Dictionary API, identical numbers):
##
##        update_player(state, input, dt, seed)
##          state : {x, y, z, vx, vy, vz, grounded}   — GLOBAL doubles
##          input : {forward, strafe, jump, sprint, yaw}
##
##      JUMP-PRESERVING CLAMP (§31): the twin zeroes vertical velocity only when
##      it is NEGATIVE (falling) — contact never kills an ascending jump.
##      GROUND_BIAS = 0.08 absorbs the fp32(GPU)/fp64(CPU) field divergence.
##
##      Positions: `wx/wy/wz` are global doubles (GDScript floats are 64-bit).
##      The node itself is placed at (global − chunk_origin) so the scene graph
##      lives inside the fp32-sweet spot (§25). Camera follows through a
##      critically damped spring (§24: k_s = 60, k_d = 2√k_s).
##
class_name PlayerController
extends Node3D

const PLAYER_HEIGHT := 1.7
const CAMERA_SENSITIVITY := 0.0022
const SPRING_K := 60.0          # §24 position spring
const SPRING_D := 15.491933     # 2·√60 — critical damping

const SPAWN := Vector3(2016.0, 0.0, -2592.0)  # tuned vista (spec §30)
const SPAWN_YAW := 2.056

## Set by main.gd — TerrainCPU extension or GDScript fallback (same API).
var physics: Object
var seed_value: int = 1337

# global doubles
var wx := SPAWN.x
var wy := 0.0
var wz := SPAWN.z
var grounded := false

var yaw := SPAWN_YAW
var pitch := 0.10
var fly_mode := false

var state := {}
var rig: Node3D
var cam: Camera3D
var _rig_vel := Vector3.ZERO


func _ready() -> void:
        add_to_group(&"rebasable")

        # ---- visual: capsule (real mesh — proves DEPTH interop with the field)
        var capsule := MeshInstance3D.new()
        var cap := CapsuleMesh.new()
        cap.radius = 0.35
        cap.height = 1.7
        capsule.mesh = cap
        capsule.position = Vector3(0, 0.85, 0)
        var mat := StandardMaterial3D.new()
        mat.albedo_color = Color(0.85, 0.55, 0.25)
        mat.roughness = 0.6
        capsule.material_override = mat
        add_child(capsule)

        # ---- camera rig (§24 spring-smoothed, terrain-clamped)
        rig = Node3D.new()
        add_child(rig)  # moves with the player; offset by the spring below
        rig.set_as_top_level(true)  # do NOT inherit the player transform — the
        #                      spring drives it in world (chunk-local) space.
        rig.add_to_group(&"rebasable")  # top-level node: shift by −delta on rebase
        #                      (§25) or the camera would swoop 8192 m per crossing.
        cam = Camera3D.new()
        cam.fov = 70.0
        cam.near = 0.1
        cam.far = 16000.0
        cam.current = true
        rig.add_child(cam)

        # spawn on the field
        wy = _elevation() + PLAYER_HEIGHT + 0.5
        state = {
                "x": wx, "y": wy, "z": wz,
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "grounded": false,
        }
        position = _local_position()
        rig.position = position + Vector3(0, 1.55, 0)

        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _elevation() -> float:
        return physics.get_elevation(wx, wz, seed_value)


func _local_position() -> Vector3:
        var fo := get_tree().get_first_node_in_group(&"floating_origin") as FloatingOrigin
        if fo == null:
                return Vector3(wx, wy - PLAYER_HEIGHT, wz)
        return Vector3(wx - fo.origin.x, wy - PLAYER_HEIGHT, wz - fo.origin.z)


func _physics_process(delta: float) -> void:
        # ---- input (raw physical keys — no InputMap needed)
        var fwd := 0.0
        if Input.is_physical_key_pressed(KEY_W): fwd += 1.0
        if Input.is_physical_key_pressed(KEY_S): fwd -= 1.0
        var str := 0.0
        if Input.is_physical_key_pressed(KEY_D): str += 1.0
        if Input.is_physical_key_pressed(KEY_A): str -= 1.0
        var jump := Input.is_physical_key_pressed(KEY_SPACE)
        var sprint := Input.is_physical_key_pressed(KEY_SHIFT)

        if fly_mode:
                _fly(delta, fwd, str, jump, sprint)
        else:
                # ---- §31 CPU twin movement loop (jump-preserving ground clamp)
                state = physics.update_player(state, {
                        "forward": fwd,
                        "strafe": str,
                        "jump": jump,
                        "sprint": sprint,
                        "yaw": yaw,
                }, delta, seed_value)
                wx = state["x"]
                wy = state["y"]
                wz = state["z"]
                grounded = state["grounded"]

        # ---- floating origin rebase (§25) — keeps node coords fp32-clean
        var fo := get_tree().get_first_node_in_group(&"floating_origin") as FloatingOrigin
        if fo:
                fo.maybe_rebase(Vector3(wx - fo.origin.x, 0.0, wz - fo.origin.z))

        position = _local_position()
        _update_camera(delta)


func _fly(delta: float, fwd: float, str: float, up_key: bool, sprint: bool) -> void:
        # FLY_SPEED = 40 (§31); sprint ×2.5. The twin clamps us above the field.
        var speed := 40.0 * (2.5 if sprint else 1.0)
        var cy := cos(yaw)
        var sy := sin(yaw)
        var cp := cos(pitch)
        var sp := sin(pitch)
        var fx := sy * cp
        var fy := sp
        var fz := cy * cp
        var vx := (fx * fwd + cy * str) * speed
        var vy := fy * fwd * speed + (40.0 if up_key else 0.0) \
                - (40.0 if Input.is_physical_key_pressed(KEY_C) else 0.0)
        var vz := (fz * fwd - sy * str) * speed
        var k := 1.0 - exp(-8.0 * delta)
        state["vx"] += (vx - state["vx"]) * k
        state["vy"] += (vy - state["vy"]) * k
        state["vz"] += (vz - state["vz"]) * k
        wx += state["vx"] * delta
        wy += state["vy"] * delta
        wz += state["vz"] * delta
        var min_y := _elevation() + 1.9
        if wy < min_y:
                wy = min_y
                state["vy"] = maxf(state["vy"], 0.0)
        wy = minf(wy, 1150.0)  # stay under the cloud deck
        grounded = false


func _update_camera(delta: float) -> void:
        # §24: critically damped spring onto the head position.
        var target := position + Vector3(0, 1.55, 0)
        var acc := (target - rig.position) * SPRING_K - _rig_vel * SPRING_D
        _rig_vel += acc * delta
        rig.position += _rig_vel * delta

        # orbit: yaw/pitch around the head, 5.5 m back
        var dist := 5.5
        var cy := cos(yaw)
        var sy := sin(yaw)
        var cp := cos(pitch)
        var sp := sin(pitch)
        var off := Vector3(-sy * cp * dist, sp * dist + 0.35, -cy * cp * dist)
        var cam_pos := rig.position + off

        # terrain clamp — the twin and the shader agree to < 1 mm (§27)
        var gy := physics.get_elevation(cam_pos.x + _origin_x(), cam_pos.z + _origin_z(), seed_value)
        cam_pos.y = maxf(cam_pos.y, gy + 0.6)
        cam.global_position = cam_pos
        cam.look_at(rig.position + Vector3(0, 0.35, 0), Vector3.UP)


func _origin_x() -> float:
        var fo := get_tree().get_first_node_in_group(&"floating_origin") as FloatingOrigin
        return 0.0 if fo == null else fo.origin.x


func _origin_z() -> float:
        var fo := get_tree().get_first_node_in_group(&"floating_origin") as FloatingOrigin
        return 0.0 if fo == null else fo.origin.z


func _unhandled_input(event: InputEvent) -> void:
        if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
                yaw -= event.relative.x * CAMERA_SENSITIVITY
                pitch = clampf(pitch - event.relative.y * CAMERA_SENSITIVITY, -1.2, 1.35)
        elif event is InputEventMouseButton and event.pressed:
                if Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
                        Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
        elif event is InputEventKey and event.pressed:
                if event.keycode == KEY_ESCAPE:
                        Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
                elif event.keycode == KEY_F:
                        fly_mode = not fly_mode
                        if fly_mode:
                                state["vy"] = maxf(state["vy"], 0.0)
                        else:
                                # Sync the twin's position from the fly camera — otherwise the
                                # stale dictionary teleports the player back on the next step.
                                state["x"] = wx
                                state["y"] = wy
                                state["z"] = wz
                                state["vx"] = 0.0
                                state["vz"] = 0.0
