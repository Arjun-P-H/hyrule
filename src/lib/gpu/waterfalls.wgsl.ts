/**
 * Waterfalls off the sky islands (skyworld/hydrology.ts plans them): each is
 * a lake's stream pouring over a notch in the rim. Too high to land as a
 * column, it frays on the way down — streaks, then clumps of spray, then
 * nothing but drifting mist — and has all turned to mist by `reach` metres
 * down (at most half way to the land below), as the world's tallest falls
 * do. The mist trails on below it, drifting downwind and thinning; with the
 * sun at your back it holds a faint rainbow.
 * ---------------------------------------------------------------------------
 * The fall: a ribbon turned to face the camera about its vertical axis,
 * leaving the lip at the stream's speed and widening as it frays; foam
 * streaks scroll down it at falling speed. The mist: a column of soft
 * camera-facing billows per fall, larger and fainter the lower they hang.
 * Both are blended (premultiplied) after the opaque world and the water,
 * test depth but do not write it.
 *
 * Needs, from the raster shader: u (Frame), realm (skyRealm.wgsl.ts),
 * vnoise, the light API (sky.wgsl.ts), surfaceFog, display, skyShadowAt,
 * project().
 */

/** Sections down a fall (triangle-strip vertices = 2 × (sections + 1)). */
export const FALL_SEGMENTS = 48;
/** Mist billows per fall. */
export const MIST_PER_FALL = 6;

export const WATERFALLS_WGSL = /* wgsl */ `
const FALL_SEGS: u32 = ${FALL_SEGMENTS}u;
const FALL_G: f32 = 22.0;
const FALL_WIND: vec2f = vec2f(0.87, 0.49);

struct FOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) uv: vec2f,        // across (−1..1), metres fallen
  @location(2) @interpolate(flat) k: u32,
};

// Centre of fall k after dropping d metres: out over the lip at its speed, the spray drifting downwind.
fn fallCentre(k: u32, d: f32) -> vec3f {
  let F = realm.fall[k];
  let out = F.b.z * sqrt(2.0 * d / FALL_G);
  let drift = 0.035 * d * d / max(F.b.w, 1.0);
  return vec3f(F.a.x + F.b.x * out + FALL_WIND.x * drift, F.a.y - d, F.a.z + F.b.y * out + FALL_WIND.y * drift);
}

@vertex
fn vsFall(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> FOut {
  let seg = vi / 2u;
  let side = f32(vi & 1u) * 2.0 - 1.0;
  let reach = realm.fall[k].b.w;
  let f = f32(seg) / f32(FALL_SEGS);
  let d = reach * f * f; // denser near the lip, where the arc bends
  let c = fallCentre(k, d);
  let toCam = u.camPos - c;
  let right = normalize(vec3f(toCam.z, 0.0, -toCam.x) + vec3f(1e-4, 0.0, 0.0));
  let w = realm.fall[k].a.w * (1.0 + 1.6 * sqrt(d / reach));
  let p = c + right * (side * 0.5 * w);
  var o: FOut;
  o.rel = p - u.camPos;
  o.pos = project(o.rel);
  o.uv = vec2f(side, d);
  o.k = k;
  return o;
}

@fragment
fn fsFall(i: FOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let F = realm.fall[i.k];
  let sd = F.c.x;
  let across = i.uv.x;
  let d = i.uv.y;
  let reach = F.b.w;
  // Streaks scrolling down at falling speed.
  let v = (d - u.time * 18.0) / 9.0;
  let s1 = 0.5 + 0.5 * vnoise(across * 5.0 + sd * 0.001, v, 1201);
  let s2 = 0.5 + 0.5 * vnoise(across * 13.0, v * 2.3 + 7.0, 1203);
  let streak = smoothstep(0.35, 0.8, 0.65 * s1 + 0.35 * s2);
  // It frays: a column near the lip, clumps of spray lower down, gone by its reach.
  let whole = 1.0 - smoothstep(0.12 * reach, 0.75 * reach, d);
  let clumps = smoothstep(0.45, 0.8, 0.5 + 0.5 * vnoise(across * 2.2 + 3.0, (d - u.time * 12.0) / 14.0, 1205));
  let spray = clumps * 0.55 * (1.0 - whole) * (1.0 - smoothstep(0.65 * reach, reach, d));
  let edge = 1.0 - smoothstep(mix(0.55, 0.2, 1.0 - whole), 1.0, abs(across));
  let lip = smoothstep(0.0, 2.0, d);
  let alpha = clamp(((0.35 + 0.65 * streak) * whole + spray) * edge * lip * 0.9, 0.0, 1.0);
  if (alpha < 0.003) { discard; }
  let p = u.camPos + i.rel;
  let sun = skyShadowAt(p);
  let L = keyDir();
  // Light through the spray toward the sun.
  let through = pow(max(dot(rd, L), 0.0), 4.0) * 0.6;
  let lit = keyCol(p.y) * (0.3 + through) * (0.45 + 0.55 * sun) + skyAmb(vec3f(0.0, 1.0, 0.0), 0.75);
  let col = mix(vec3f(0.55, 0.7, 0.8), vec3f(0.95, 0.97, 1.0), max(streak, 1.0 - whole)) * lit;
  let shown = display(surfaceFog(col, dist, p.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}

// The mist a fall turns into: billows down its lower reach and below, drifting downwind.
struct MOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) q: vec2f,
  @location(2) @interpolate(flat) km: vec2u, // fall, billow
};

@vertex
fn vsMist(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let corners = array<vec2f, 4>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));
  let q = corners[vi];
  let k = ii / ${MIST_PER_FALL}u;
  let m = ii % ${MIST_PER_FALL}u;
  let F = realm.fall[k];
  let reach = F.b.w;
  let fm = f32(m);
  let d = reach * (0.35 + 0.22 * fm);
  let size = F.a.w * (2.5 + 1.5 * fm) + 0.06 * reach * fm;
  // Below its reach the mist keeps drifting downwind, and bobs slowly.
  let t = u.time * 0.05 + F.c.x * 0.001 + fm * 1.7;
  let c = fallCentre(k, min(d, reach)) + vec3f(FALL_WIND.x, 0.0, FALL_WIND.y) * (max(d - reach, 0.0) * 0.4 + 6.0 * sin(t))
        + vec3f(0.0, -max(d - reach, 0.0) + 4.0 * sin(t * 1.3), 0.0);
  let toCam = normalize(u.camPos - c);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam) + vec3f(1e-4, 0.0, 0.0));
  let up = cross(toCam, right);
  let p = c + (right * q.x + up * q.y) * size;
  var o: MOut;
  o.rel = p - u.camPos;
  o.pos = project(o.rel);
  o.q = q;
  o.km = vec2u(k, m);
  return o;
}

@fragment
fn fsMist(i: MOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  let r = length(i.q);
  let t = u.time;
  let fm = f32(i.km.y);
  let n = 0.5 + 0.5 * vnoise(i.q.x * 2.2 + t * 0.12 + fm * 3.1, i.q.y * 2.2 - t * 0.2, 1211 + i32(i.km.x));
  let alpha = clamp((1.0 - smoothstep(0.2, 1.0, r + 0.3 * n)) * 0.3 * (1.0 - fm / ${MIST_PER_FALL + 1}.0), 0.0, 1.0);
  if (alpha < 0.003) { discard; }
  let p = u.camPos + i.rel;
  let L = keyDir();
  var col = vec3f(0.9, 0.93, 0.96) * (keyCol(p.y) * 0.3 * (0.5 + 0.5 * skyShadowAt(p)) + skyAmb(vec3f(0.0, 1.0, 0.0), 0.7));
  // A rainbow 42° from the point opposite the sun (red outside, violet in).
  let a = acos(clamp(dot(rd, -L), -1.0, 1.0)) * 57.29578;
  let band = (a - 40.6) / 1.8;
  if (band > -0.3 && band < 1.3 && L.y > 0.0) {
    let hue = clamp(band, 0.0, 1.0);
    let bow = clamp(vec3f(1.5 - abs(hue - 1.0) * 3.0, 1.5 - abs(hue - 0.5) * 3.0, 1.5 - abs(hue) * 3.0), vec3f(0.0), vec3f(1.0));
    col += bow * keyCol(p.y) * 0.18 * (1.0 - smoothstep(0.0, 0.3, abs(band - 0.5) - 0.5));
  }
  let shown = display(surfaceFog(col, dist, p.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}
`;
