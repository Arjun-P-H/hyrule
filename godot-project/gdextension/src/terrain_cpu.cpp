/**
 * ============================================================================
 *  TerrainCPU — Godot 4 GDExtension class (spec §31 CPU twin wrapper)
 * ============================================================================
 *  Exposes the double-precision field in terrain_cpu.h to GDScript.
 *
 *  API (all positions are GLOBAL doubles — Godot Variant floats are 64-bit,
 *  so precision survives the round trip; node positions stay chunk-local):
 *
 *    get_elevation(x, z, seed) -> float
 *        Full-detail field E(x, z) in metres (dp = 0).
 *
 *    get_gradient(x, z, seed) -> Vector2
 *        (dE/dx, dE/dz) via central differences.
 *
 *    get_normal(x, z, seed) -> Vector3
 *        normalize(−dE/dx, 1, −dE/dz) — matches the shader's analytic normal.
 *
 *    get_climate(x, z, seed) -> Vector2
 *        (temperature °C, moisture 0..1) — §9.
 *
 *    query_anomaly(x, z, seed) -> Dictionary
 *        Nearest ACTIVE Zonai structure within a 5×5 cell scan (±2 cells of
 *        2048 m): { found, x, y, z, distance }.
 *
 *    march_ray(from, dir, seed, steps, footprint, max_dist) -> Dictionary
 *        Lipschitz march + bisection against E (gameplay raycast twin of the
 *        shader): { hit, t, position }.
 *
 *    update_player(state, input, dt, seed) -> Dictionary
 *        §31 movement loop verbatim (jump-preserving clamp, GROUND_BIAS,
 *        slope slide-off). state/input are Dictionaries of doubles/bools.
 *
 *  SDFGI / TBDR note: this class never touches the renderer. The GPU surface
 *  is drawn by shaders/terrain_raymarch.gdshader, which writes DEPTH/NORMAL
 *  per pixel; this twin exists because a purely mathematical surface is
 *  invisible to Godot's physics servers. Parity budget: < 1 mm (§27),
 *  absorbed by GROUND_BIAS.
 * ============================================================================
 */

#include "terrain_cpu.h"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

class TerrainCPU : public RefCounted {
    GDCLASS(TerrainCPU, RefCounted)

public:
    // ---- field queries ----------------------------------------------------
    double get_elevation(double x, double z, int64_t seed) const {
        return terrain::elevation(x, z, static_cast<int32_t>(seed), 0.0);
    }

    Vector2 get_gradient(double x, double z, int64_t seed) const {
        double gx, gz;
        terrain::gradient(x, z, static_cast<int32_t>(seed), 0.25, gx, gz);
        return Vector2(gx, gz);
    }

    Vector3 get_normal(double x, double z, int64_t seed) const {
        double gx, gz;
        terrain::gradient(x, z, static_cast<int32_t>(seed), 0.25, gx, gz);
        const double len = std::sqrt(gx * gx + gz * gz + 1.0);
        return Vector3(-gx / len, 1.0 / len, -gz / len);
    }

    Vector2 get_climate(double x, double z, int64_t seed) const {
        const auto s = static_cast<int32_t>(seed);
        const double e = terrain::elevation(x, z, s, 0.0);
        return Vector2(
            terrain::temperature(x, z, z, e, s),
            terrain::moisture(x, z, s));
    }

    // ---- §12 anomaly query (gameplay: shrine compass / map icons) ----------
    Dictionary query_anomaly(double x, double z, int64_t seed) const {
        const auto s = static_cast<int32_t>(seed);
        Dictionary out;
        // 5×5 cell scan at 2048 m — covers ±4 km around the query point.
        const double cell_x = std::floor(x / terrain::ANOM_CELL);
        const double cell_z = std::floor(z / terrain::ANOM_CELL);
        bool found = false;
        double best = 1e18, bx = 0, bz = 0;
        for (int j = -2; j <= 2; ++j) {
            for (int i = -2; i <= 2; ++i) {
                const auto ix = static_cast<int32_t>(cell_x + i);
                const auto iz = static_cast<int32_t>(cell_z + j);
                if (terrain::cell_type(ix, iz, s + 400) >= terrain::P_ANOM) continue;
                double px, py;
                terrain::cell_point(ix, iz, s + 400, px, py);
                const double cx = (ix + px) * terrain::ANOM_CELL;
                const double cz = (iz + py) * terrain::ANOM_CELL;
                const double d = std::sqrt((cx - x) * (cx - x) + (cz - z) * (cz - z));
                if (d < best) { best = d; bx = cx; bz = cz; found = true; }
            }
        }
        out["found"] = found;
        if (found) {
            out["x"] = bx;
            out["z"] = bz;
            out["y"] = terrain::elevation(bx, bz, s, 0.0);
            out["distance"] = best;
        }
        return out;
    }

    // ---- §16 gameplay raycast twin -----------------------------------------
    Dictionary march_ray(const Vector3 &from, const Vector3 &dir, int64_t seed,
                         int64_t steps, double footprint, double max_dist) const {
        // Vector3 is fp32 — fine for near-field gameplay rays (< 0.01 m error
        // inside the 8 km play disc; long-range ballistics should call the
        // double-based update loop instead).
        const Vector3 d = dir.normalized();
        const auto s = static_cast<int32_t>(seed);
        const double md = max_dist > 0.0 ? max_dist : terrain::MAX_DIST;
        const terrain::MarchResult r = terrain::march_ray(
            from.x, from.y, from.z, d.x, d.y, d.z, s,
            static_cast<int>(steps), footprint, md);
        Dictionary out;
        out["hit"] = r.hit;
        out["t"] = r.t;
        out["position"] = Vector3(
            static_cast<float>(from.x + d.x * r.t),
            static_cast<float>(from.y + d.y * r.t),
            static_cast<float>(from.z + d.z * r.t));
        return out;
    }

    // ---- §31 movement loop --------------------------------------------------
    Dictionary update_player(const Dictionary &state, const Dictionary &input,
                             double dt, int64_t seed) const {
        terrain::PlayerState ps{
            state.get("x", 0.0), state.get("y", 0.0), state.get("z", 0.0),
            state.get("vx", 0.0), state.get("vy", 0.0), state.get("vz", 0.0),
            state.get("grounded", false)};
        terrain::InputState is{
            input.get("forward", 0.0), input.get("strafe", 0.0),
            input.get("jump", false), input.get("sprint", false),
            input.get("yaw", 0.0)};
        terrain::update_player(ps, is, dt, static_cast<int32_t>(seed));

        Dictionary out;
        out["x"] = ps.x; out["y"] = ps.y; out["z"] = ps.z;
        out["vx"] = ps.vx; out["vy"] = ps.vy; out["vz"] = ps.vz;
        out["grounded"] = ps.grounded;
        return out;
    }

    // ---- contract constants for GDScript-side tuning/UI --------------------
    Dictionary get_constants() const {
        Dictionary c;
        c["chunk_size"] = terrain::CHUNK_SIZE;
        c["max_world_h"] = terrain::MAX_WORLD_H;
        c["max_dist"] = terrain::MAX_DIST;
        c["player_height"] = terrain::PLAYER_HEIGHT;
        c["ground_bias"] = terrain::GROUND_BIAS;
        c["gravity"] = terrain::GRAVITY;
        c["walk_speed"] = terrain::WALK_SPEED;
        c["sprint_mul"] = terrain::SPRINT_MUL;
        c["jump_velocity"] = terrain::JUMP_V;
        c["fly_speed"] = terrain::FLY_SPEED;
        c["slope_max"] = terrain::SLOPE_MAX;
        c["lambda_step"] = terrain::LAMBDA_STEP;
        c["delta_min"] = terrain::DELTA_MIN;
        c["bisect_steps"] = terrain::BISECT_STEPS;
        c["water_level"] = terrain::W_LEVEL;
        return c;
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("get_elevation", "x", "z", "seed"), &TerrainCPU::get_elevation);
        ClassDB::bind_method(D_METHOD("get_gradient", "x", "z", "seed"), &TerrainCPU::get_gradient);
        ClassDB::bind_method(D_METHOD("get_normal", "x", "z", "seed"), &TerrainCPU::get_normal);
        ClassDB::bind_method(D_METHOD("get_climate", "x", "z", "seed"), &TerrainCPU::get_climate);
        ClassDB::bind_method(D_METHOD("query_anomaly", "x", "z", "seed"), &TerrainCPU::query_anomaly);
        ClassDB::bind_method(D_METHOD("march_ray", "from", "dir", "seed", "steps", "footprint", "max_dist"), &TerrainCPU::march_ray);
        ClassDB::bind_method(D_METHOD("update_player", "state", "input", "dt", "seed"), &TerrainCPU::update_player);
        ClassDB::bind_method(D_METHOD("get_constants"), &TerrainCPU::get_constants);
    }
};

} // namespace godot
