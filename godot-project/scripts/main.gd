##
##      main.gd — W(x,y) Continuous World Engine scene assembly
##      ====================================================================
##      Builds everything in code (no .tscn graph drift):
##        * WorldEnvironment — SDFGI + ACES (Forward+ required for SDFGI)
##        * DirectionalLight — sun, light_cull_mask excludes layer 2 (terrain is
##          custom-lit by the shader; SDFGI indirect still reaches it)
##        * FloatingOrigin  — §25 chunked rebasing
##        * TerrainRenderer — the fullscreen mathematical surface
##        * PlayerController — §31 CPU-twin movement
##        * HUD + proof props (real meshes that depth-test against the analytic
##          terrain — the §16 DEPTH interop demo)
##
extends Node3D

const SEED := 1337
const SUN_DIR := Vector3(0.550744, 0.420568, 0.720974)  # toward the sun

var physics: Object
var player: PlayerController
var hud_label: Label
var _hud_at := 0.0


func _ready() -> void:
        # ---- floating origin (§25) ------------------------------------------
        var fo := FloatingOrigin.new()
        fo.name = "FloatingOrigin"
        fo.add_to_group(&"floating_origin")
        add_child(fo)

        # ---- environment: SDFGI + ACES (Forward+ only) -----------------------
        var env := Environment.new()
        env.background_mode = Environment.BG_COLOR
        env.background_color = Color(0.02, 0.03, 0.04)
        env.sdfgi_enabled = true            # voxel GI reads our ALBEDO/NORMAL/DEPTH
        env.sdfgi_min_cell_size = 0.25
        env.tonemap_mode = Environment.TONE_MAPPER_ACES
        env.glow_enabled = true
        env.glow_intensity = 0.4
        # Ambient source is SDFGI (sky energy), NOT the engine ambient — the
        # shader already carries a half-strength analytic sky-ambient floor.
        var we := WorldEnvironment.new()
        we.environment = env
        add_child(we)

        # ---- sun (props + SDFGI energy; terrain excluded via cull mask) -------
        var sun := DirectionalLight3D.new()
        sun.name = "Sun"
        # Basis.looking_at(dir, up) points −Z at `dir`; we want the light to
        # travel along −SUN_DIR, so look at −SUN_DIR.
        sun.basis = Basis.looking_at(-SUN_DIR, Vector3.UP)
        sun.light_energy = 1.6
        sun.light_color = Color(1.0, 0.955, 0.885)
        sun.shadow_enabled = true           # shadows for real meshes near the player
        # CRITICAL (§31): exclude layer 2 — the terrain's direct sun term lives in
        # the shader's EMISSION; engine lighting would double-light the surface.
        sun.light_cull_mask = 4294967295 - 2
        add_child(sun)

        # ---- physics twin: GDExtension if built, GDScript fallback otherwise --
        if ClassDB.class_exists("TerrainCPU"):
                physics = ClassDB.instantiate("TerrainCPU")
                print("W(x,y): using TerrainCPU GDExtension (C++ twin)")
        else:
                physics = load("res://scripts/terrain_math_fallback.gd").new()
                print("W(x,y): TerrainCPU extension not built — using GDScript twin ",
                        "(run scons in gdextension/ for the shipping path)")

        # ---- the mathematical surface ----------------------------------------
        var terrain := TerrainRenderer.new()
        terrain.name = "Terrain"
        terrain.seed_value = SEED
        add_child(terrain)

        # ---- player ------------------------------------------------------------
        player = PlayerController.new()
        player.name = "Player"
        player.physics = physics
        player.seed_value = SEED
        add_child(player)

        # ---- proof props: real meshes inter-occluding with the analytic field --
        _spawn_proof_props()

        # ---- HUD ---------------------------------------------------------------
        var canvas := CanvasLayer.new()
        add_child(canvas)
        hud_label = Label.new()
        hud_label.position = Vector2(12, 10)
        hud_label.add_theme_font_size_override("font_size", 13)
        hud_label.add_theme_color_override("font_color", Color(0.92, 0.95, 0.9))
        canvas.add_child(hud_label)


func _spawn_proof_props() -> void:
        # Three markers around spawn. They are ordinary meshes on layer 1 (lit by
        # the sun, voxelized by SDFGI) that sit ON the mathematical surface and
        # depth-sort against it — proving the reversed-Z DEPTH write (§16).
        var spots := [Vector2(2016.0, -2592.0), Vector2(2076.0, -2602.0), Vector2(1998.0, -2650.0)]
        for i in spots.size():
                var box := MeshInstance3D.new()
                var bm := BoxMesh.new()
                bm.size = Vector3(1.2, 2.4, 1.2)
                box.mesh = bm
                var mat := StandardMaterial3D.new()
                mat.albedo_color = Color(0.9, 0.2, 0.15) if i == 0 else Color(0.95, 0.75, 0.2)
                mat.roughness = 0.5
                box.material_override = mat
                var e: float = physics.get_elevation(spots[i].x, spots[i].y, SEED)
                # globals → chunk-local (§25)
                var fo := get_tree().get_first_node_in_group(&"floating_origin") as FloatingOrigin
                box.position = Vector3(spots[i].x - fo.origin.x, e + 1.2, spots[i].y - fo.origin.z)
                box.add_to_group(&"rebasable")
                add_child(box)


func _process(delta: float) -> void:
        _hud_at += delta
        if _hud_at < 0.25 or hud_label == null:
                return
        _hud_at = 0.0
        var fo := get_tree().get_first_node_in_group(&"floating_origin") as FloatingOrigin
        hud_label.text = "W(x,y) seed %d  |  %s  |  pos (%.1f, %.1f, %.1f)  |  chunk (%d, %d)  |  %s" % [
                SEED,
                "fly" if player.fly_mode else ("grounded" if player.grounded else "airborne"),
                player.wx, player.wy, player.wz,
                int(fo.origin.x), int(fo.origin.z),
                "TerrainCPU GDExtension" if ClassDB.class_exists("TerrainCPU") else "GDScript twin",
        ]
