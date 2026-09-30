/**
 * Waterfalls pouring off sky islands (placement: skyIslands.ts islandWaterfall).
 * ---------------------------------------------------------------------------
 * Each fall is a ribbon turned to face the camera about its vertical axis:
 * it leaves the rim in a short outward arc, widens as it falls, and is
 * streaked by foam scrolling down at falling speed; a mist cloud billows
 * where it lands. Both are blended (premultiplied) after the opaque world
 * and the water, test depth but do not write it.
 *
 * Needs, from the raster shader: u (Frame), vnoise, SUN, SUN_COL, SKY_TINT,
 * surfaceFog, display, islandShadowAt, project().
 */

export const MAX_FALLS = 16;
/** count vec4, then two vec4 per fall. */
export const FALLS_BYTES = 16 + MAX_FALLS * 32;
/** Segments along a fall (triangle-strip vertices = 2 × (segments + 1)). */
export const FALL_SEGMENTS = 48;

export const WATERFALLS_WGSL = /* wgsl */ `
struct FallsU {
  count: vec4u,
  a: array<vec4f, ${MAX_FALLS}>, // lip (chunk-local x, y, z), width
  b: array<vec4f, ${MAX_FALLS}>, // outward direction x, z; landing height; seed
};
@group(0) @binding(11) var<uniform> falls: FallsU;
const FALL_SEGS: u32 = ${FALL_SEGMENTS}u;

struct FOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) uv: vec2f,        // across (-1..1), metres fallen
  @location(2) @interpolate(flat) k: u32,
  @location(3) drop: f32,        // total height of the fall
};

// Centre of fall k after dropping d metres: the water leaves the lip at ~6 m/s.
fn fallCentre(k: u32, d: f32) -> vec3f {
  let A = falls.a[k];
  let B = falls.b[k];
  let out = 6.0 * sqrt(2.0 * d / 22.0);
  return vec3f(A.x + B.x * out, A.y - d, A.z + B.y * out);
}

@vertex
fn vsFall(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> FOut {
  let seg = vi / 2u;
  let side = f32(vi & 1u) * 2.0 - 1.0;
  let drop = max(falls.a[k].y - falls.b[k].z, 1.0);
  let f = f32(seg) / f32(FALL_SEGS);
  let d = drop * f * f; // denser segments near the lip, where the arc bends
  let c = fallCentre(k, d);
  let toCam = u.camPos - c;
  let right = normalize(vec3f(toCam.z, 0.0, -toCam.x) + vec3f(1e-4, 0.0, 0.0));
  let w = falls.a[k].w * (1.0 + 1.6 * sqrt(d / max(drop, 1.0)));
  let p = c + right * (side * 0.5 * w);
  var o: FOut;
  o.rel = p - u.camPos;
  o.pos = project(o.rel);
  o.uv = vec2f(side, d);
  o.k = k;
  o.drop = drop;
  return o;
}

@fragment
fn fsFall(i: FOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let sd = falls.b[i.k].w;
  let across = i.uv.x;
  let d = i.uv.y;
  // Streaks: noise stretched along the fall, scrolling down at falling speed.
  let speed = 18.0;
  let v = (d - u.time * speed) / 9.0;
  let s1 = 0.5 + 0.5 * vnoise(across * 5.0 + sd * 0.001, v, 1201);
  let s2 = 0.5 + 0.5 * vnoise(across * 13.0, v * 2.3 + 7.0, 1203);
  let streak = smoothstep(0.35, 0.8, 0.65 * s1 + 0.35 * s2);
  let edge = 1.0 - smoothstep(0.55, 1.0, abs(across));
  let fadeTop = smoothstep(0.0, 3.0, d);
  let fadeBottom = 1.0 - smoothstep(i.drop - 12.0, i.drop, d);
  let alpha = clamp((0.35 + 0.65 * streak) * edge * fadeTop * fadeBottom * 0.9, 0.0, 1.0);
  let p = u.camPos + i.rel;
  let sun = islandShadowAt(p, 0xffffffffu);
  let lit = SUN_COL * 0.3 * (0.45 + 0.55 * sun) + SKY_TINT * 0.75;
  let col = mix(vec3f(0.55, 0.7, 0.8), vec3f(0.95, 0.97, 1.0), streak) * lit;
  let shown = display(surfaceFog(col, dist, p.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}

// Mist where the fall lands: a camera-facing quad per fall, a soft billowing disc.
struct MOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) q: vec2f,
  @location(2) @interpolate(flat) k: u32,
};

@vertex
fn vsMist(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> MOut {
  let corners = array<vec2f, 4>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  let q = corners[vi];
  let drop = max(falls.a[k].y - falls.b[k].z, 1.0);
  let base = fallCentre(k, drop);
  let size = falls.a[k].w * 5.0;
  let c = base + vec3f(0.0, size * 0.35, 0.0);
  let toCam = normalize(u.camPos - c);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam) + vec3f(1e-4, 0.0, 0.0));
  let up = cross(toCam, right);
  let p = c + (right * q.x + up * q.y) * size;
  var o: MOut;
  o.rel = p - u.camPos;
  o.pos = project(o.rel);
  o.q = q;
  o.k = k;
  return o;
}

@fragment
fn fsMist(i: MOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let r = length(i.q);
  let t = u.time;
  let n = 0.5 + 0.5 * vnoise(i.q.x * 2.2 + t * 0.35, i.q.y * 2.2 - t * 0.6, 1211 + i32(i.k));
  let alpha = clamp((1.0 - smoothstep(0.25, 1.0, r + 0.25 * n)) * 0.55, 0.0, 1.0);
  let col = vec3f(0.9, 0.93, 0.96) * (SUN_COL * 0.28 + SKY_TINT * 0.7);
  let shown = display(surfaceFog(col, dist, u.camPos.y + i.rel.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}
`;
