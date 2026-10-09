/**
 * Ruins and shrines in WGSL: the stone boxes of src/lib/ruins.ts drawn as
 * instanced cubes (vertex pulling from a storage buffer: 6 faces × 2
 * triangles per box), and the site list the grass and tree scatters use to
 * keep out of them.
 *
 * Needs, from the raster shader: u (Frame), vnoise, the light API (sky.wgsl.ts),
 * sunVisibility, heroShadowAt, surfaceFog, display, underwater, project().
 */

export const MAX_RUIN_BOXES = 1024;
export const RUIN_BOX_BYTES = 32;
export const MAX_RUIN_SITES = 24;
export const RUIN_SITES_BYTES = 16 + MAX_RUIN_SITES * 16;

/** The site list (group 0, all stages): scatters skip candidates inside a site. */
export const RUIN_SITES_WGSL = /* wgsl */ `
struct RuinSitesU {
  count: vec4u,
  s: array<vec4f, ${MAX_RUIN_SITES}>, // centre x, z (chunk-local), radius, platform top
};
@group(0) @binding(13) var<uniform> ruinSites: RuinSitesU;
fn inRuinSite(xz: vec2f) -> bool {
  for (var k = 0u; k < ruinSites.count.x; k++) {
    let s = ruinSites.s[k];
    if (distance(xz, s.xy) < s.z) { return true; }
  }
  return false;
}
`;

export const RUINS_WGSL = /* wgsl */ `
struct RuinBoxG {
  c: vec4f, // centre (chunk-local), yaw
  h: vec4f, // half extents; material (0 stone, 1 mossy, 2 shrine) + a per-box hash in [0, 1) as the fraction
};
@group(0) @binding(12) var<storage, read> ruinBoxes: array<RuinBoxG>;

struct ROut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) lp: vec3f,  // box-local position (m)
  @location(3) nl: vec3f,  // box-local normal
  @location(4) @interpolate(flat) mat: vec2u, // material, per-box hash (16 bits)
  @location(5) edge: vec2f, // height above the box's bottom, depth below its top
};

@vertex
fn vsRuin(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ROut {
  let b = ruinBoxes[ii];
  let face = vi / 6u;
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let normals = array<vec3f, 6>(vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));
  let tangents = array<vec3f, 6>(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0));
  let st = corners[vi % 6u];
  let n = normals[face];
  let t1 = tangents[face];
  let t2 = cross(n, t1);
  let lp = (n + t1 * st.x + t2 * st.y) * b.h.xyz;
  // Site-local u (x) and v (z) to world, as ruins.ts at(): x += cs·u + sn·v, z += −sn·u + cs·v.
  let cy = cos(b.c.w);
  let sy = sin(b.c.w);
  let world = vec3f(b.c.x + cy * lp.x + sy * lp.z, b.c.y + lp.y, b.c.z - sy * lp.x + cy * lp.z);
  var o: ROut;
  o.rel = world - u.camPos;
  o.pos = project(o.rel);
  o.n = vec3f(cy * n.x + sy * n.z, n.y, -sy * n.x + cy * n.z);
  o.lp = lp;
  o.nl = n;
  o.mat = vec2u(u32(b.h.w), u32(fract(b.h.w) * 65535.0));
  o.edge = vec2f(lp.y + b.h.y, b.h.y - lp.y);
  return o;
}

fn ruinHash(c: vec2f, salt: u32) -> f32 {
  return f32(hash3(i32(c.x), i32(c.y), i32(salt)) & 0xffffu) / 65535.0;
}

@fragment
fn fsRuin(i: ROut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let n = normalize(i.n);
  let p = u.camPos + i.rel;
  let g = p.xz + u.origin;
  let an = abs(i.nl);
  let top = an.y > 0.5;
  // Face coordinates: along the face and up it, or across a top face.
  var fc = i.lp.xy;
  if (top) { fc = i.lp.xz; } else if (an.x > 0.5) { fc = i.lp.zy; }
  let aa = length(fwidth(fc));
  let boxH = i.mat.y;
  let mossN = 0.62 * vnoise(g.x * 0.35 + fc.x * 0.2, g.y * 0.35 + fc.y * 0.3, 1307)
    + 0.38 * vnoise(g.x * 1.4 + fc.x, g.y * 1.4 + fc.y * 1.3, 1309);
  var albedo: vec3f;
  var emissive = vec3f(0.0);
  var moss = 0.0;
  if (i.mat.x == 2u) {
    // Shrine stone: dark slate cut by glowing lines, concentric broken squares round each face's middle.
    albedo = vec3f(0.032, 0.036, 0.045) * (0.85 + 0.3 * (0.5 + 0.5 * mossN));
    let q = fc - vec2f(0.0, select(0.4, 0.0, top));
    let r = max(abs(q.x), abs(q.y) * 0.8);
    let ring = r / 0.75;
    let lineD = abs(fract(ring + 0.5) - 0.5) * 0.75;
    let k = floor(ring + 0.5);
    let ang = atan2(q.y, q.x) / 6.2831853 + 0.5;
    let gap = step(0.2, fract(ang * (2.0 + k) + ruinHash(vec2f(k, f32(boxH)), 1319u)));
    let line = (1.0 - smoothstep(0.012, 0.03 + aa, lineD)) * gap * step(0.5, k);
    let pulse = 0.7 + 0.3 * sin(u.time * 1.6 - k * 0.9);
    emissive = vec3f(0.1, 0.95, 0.8) * line * pulse;
  } else {
    // Stone blocks: courses 0.9 m high, staggered; flagstones 1.5 m across on top faces.
    var cell: vec2f;
    var dj: f32; // metres to the nearest joint
    if (top) {
      let qf = fc / 1.5 + 0.5;
      cell = floor(qf);
      let f = abs(fract(qf) - 0.5);
      dj = (0.5 - max(f.x, f.y)) * 1.5;
    } else {
      let course = floor(fc.y / 0.9 + 0.5);
      let qx = (fc.x + course * 0.6) / 1.7 + 0.5;
      cell = vec2f(floor(qx), course);
      dj = min((0.5 - abs(fract(qx) - 0.5)) * 1.7, (0.5 - abs(fract(fc.y / 0.9 + 0.5) - 0.5)) * 0.9);
    }
    let joint = 1.0 - smoothstep(0.015, 0.04 + aa, dj);
    let tint = ruinHash(cell, 1311u + boxH);
    let wear = 0.5 + 0.5 * vnoise(g.x * 0.9 + fc.x * 0.7, g.y * 0.9 + fc.y * 0.9, 1301);
    albedo = mix(vec3f(0.19, 0.176, 0.148), vec3f(0.285, 0.268, 0.228), wear) * (0.82 + 0.3 * tint);
    albedo *= 1.0 - 0.5 * joint * smoothstep(0.0, 0.06, dj + 0.05);
    // Moss: in patches and the joints of top faces; on the sides near the ground and under the top edge.
    if (top) {
      moss = max(smoothstep(0.42, 0.72, mossN), joint * smoothstep(0.15, 0.45, mossN) * 0.9);
    } else {
      let low = 1.0 - smoothstep(0.1, select(0.9, 2.2, i.mat.x == 1u), i.edge.x);
      let high = 1.0 - smoothstep(0.05, select(0.25, 0.7, i.mat.x == 1u), i.edge.y);
      moss = max(low, high) * smoothstep(0.38, 0.62, mossN);
    }
  }
  albedo = mix(albedo, vec3f(0.07, 0.115, 0.035) * (0.8 + 0.4 * mossN), moss);
  // Darker toward the base (ground contact), soft two-tone light with a warm bounce off the ground.
  let ao = mix(0.6, 1.0, smoothstep(0.0, 1.2, i.edge.x));
  var sun = sunVisibility(p, n * 0.1);
  if (sun > 0.0) { sun = min(sun, heroShadowAt(i.rel)); }
  let ramp = smoothstep(-0.1, 0.45, dot(n, keyDir()));
  let amb = skyAmb(n, 0.3 + 0.25 * max(n.y, 0.0)) + bounceCol(vec3f(0.07, 0.075, 0.04)) * (1.0 - abs(n.y));
  var col = albedo * (keyCol(p.y) * 0.6 * ramp * sun + amb) * ao + glow(emissive);
  col = surfaceFog(col, dist, p.y, rd);
  return display(col);
}
`;
