##
##	TerrainRenderer — fullscreen-quad driver for the raymarch shader
##	====================================================================
##	APPLE SILICON / TBDR NOTES (Metal, Forward+):
##	  * render_priority = -100: terrain enters the opaque bin FIRST, so
##	    per-tile fragments of later (real) geometry depth-test against the
##	    analytically written DEPTH for free.
##	  * custom_aabb is huge: the quad must survive frustum culling from any
##	    camera orientation (it is clip-space emitted via POSITION override).
##	  * layers = 2: the DirectionalLight's light_cull_mask excludes bit 2 —
##	    direct sun on the terrain comes from the shader's own EMISSION term;
##	    engine lighting would double-light the surface. SDFGI indirect is
##	    unaffected (voxelization reads ALBEDO/NORMAL, not the cull mask).
##	  * cast_shadow OFF: a fullscreen quad has no meaningful shadow silhouette.
##
##	Per-frame uniforms (cheap scalars — the matrices come from the engine's
##	own vertex-stage built-ins forwarded as varyings inside the shader).
##
class_name TerrainRenderer
extends MeshInstance3D

const SHADER := preload("res://shaders/terrain_raymarch.gdshader")

## Quality presets (spec §26 / §29) — march steps / shadow steps.
const PRESETS := {
	&"low": [64, 12],
	&"med": [96, 20],
	&"high": [128, 28],
}

@export var seed_value: int = 1337
@export_enum("low", "med", "high") var quality: String = "med"

var mat: ShaderMaterial
var _time := 0.0


func _ready() -> void:
	var quad := QuadMesh.new()
	quad.size = Vector2(2.0, 2.0)  # spans [-1, 1]² — the vertex stage emits
	#                                it in clip space via POSITION override.
	mesh = quad

	# Giant AABB: never frustum-cull the mathematical surface.
	custom_aabb = AABB(Vector3(-1e6, -1e6, -1e6), Vector3(2e6, 2e6, 2e6))

	render_priority = -100        # TBDR: first in the opaque bin
	layers = 2                    # excluded from the sun's cull mask
	cast_shadow = SHADOW_CASTING_SETTING_OFF

	mat = ShaderMaterial.new()
	mat.shader = SHADER
	material_override = mat


func _process(delta: float) -> void:
	_time += delta
	var cam := get_viewport().get_camera_3d()
	if cam == null:
		return
	var vp := get_viewport()
	var res_y := float(vp.size.y)
	var tan_half := tan(deg_to_rad(cam.fov * 0.5))

	mat.set_shader_parameter("u_time", _time)
	mat.set_shader_parameter("u_seed", seed_value)

	var preset: Array = PRESETS.get(StringName(quality), PRESETS[&"med"])
	mat.set_shader_parameter("u_march_steps", preset[0])
	mat.set_shader_parameter("u_shadow_steps", preset[1])

	# Nyquist pixel footprint driver (§5): Δp(t) = t · 2tan(FOV/2)/Res_y
	mat.set_shader_parameter("u_tan_half", tan_half)
	mat.set_shader_parameter("u_aspect", float(vp.size.x) / maxf(res_y, 1.0))
	mat.set_shader_parameter("u_footprint", 2.0 * tan_half / maxf(res_y, 1.0))

	# Floating origin (§25): multiples of 8192 — fp32-exact in the shader.
	var fo := get_tree().get_first_node_in_group(&"floating_origin")
	if fo is FloatingOrigin:
		mat.set_shader_parameter("u_chunk_origin",
			Vector2((fo as FloatingOrigin).origin.x, (fo as FloatingOrigin).origin.z))
