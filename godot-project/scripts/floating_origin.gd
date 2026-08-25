##
##	FloatingOrigin — chunked floating-origin manager (spec §25)
##	====================================================================
##	Chunk size 8192 = 2^13. Every hash lattice in the field uses dyadic
##	rational frequencies (n/2^20), so `chunk_origin · f` is EXACT in fp32 —
##	noise cells align across rebases with zero jitter. Nodes stay within
##	±4096 m of the origin, where fp32 spacing is < 0.5 mm.
##
##	Usage:
##	  - World-anchored nodes join the "rebasable" group; on rebase their
##	    global_position shifts by −delta (delta is a multiple of 8192 →
##	    the subtraction is exact in fp32).
##	  - The renderer reads `origin` every frame into u_chunk_origin.
##	  - Gameplay keeps GLOBAL doubles (GDScript floats are 64-bit) and
##	    derives local = global − origin only when touching nodes.
##
class_name FloatingOrigin
extends Node

signal rebased(delta: Vector3)

const CHUNK_SIZE := 8192.0   ## 2^13 (spec §25: 2^12..2^13)
const HALF_CHUNK := 4096.0

## Current chunk origin. x/z are multiples of CHUNK_SIZE (fp32-exact).
var origin := Vector3.ZERO


func world_to_local(p_world: Vector3) -> Vector3:
	return p_world - origin


func local_to_world(p_local: Vector3) -> Vector3:
	return p_local + origin


##	Rebase when `local_pos` (the player, usually) leaves the ±4096 m core.
##	Returns true if a rebase happened this call.
func maybe_rebase(local_pos: Vector3) -> bool:
	var delta := Vector3.ZERO
	if absf(local_pos.x) > HALF_CHUNK:
		delta.x = signf(local_pos.x) * CHUNK_SIZE
	if absf(local_pos.z) > HALF_CHUNK:
		delta.z = signf(local_pos.z) * CHUNK_SIZE
	if delta == Vector3.ZERO:
		return false

	origin += delta  # multiples of 8192 — exact in fp32 up to ±2^20 chunks

	# Shift every world-anchored node by −delta. Positions stay small and
	# exact; the terrain quad never moves (it is camera-relative math, not
	# world geometry — see terrain_raymarch.gdshader).
	for node in get_tree().get_nodes_in_group(&"rebasable"):
		node.global_position -= delta

	rebased.emit(delta)
	return true
