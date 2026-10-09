/**
 * Light and air the Eyries give off (SKY_CIVILIZATION.md §7, §9), in WGSL:
 * gpu/structures.ts lists them (skyworld/structures.ts Effect) round the
 * camera and draws them blended, after the realm's falls.
 * ---------------------------------------------------------------------------
 *   beam    the Vaelspire's Thread: a ribbon of pale light turned to the
 *           camera, rising 3.2 km from the crystals; pulses run up it. Far
 *           off it keeps at least ~0.7 px of core, its light spread thinner
 *           (so a distant Thread neither vanishes nor turns into a bar).
 *           Faint by day, bright from dusk (lampLevel). Added to the scene.
 *   smoke   a forge chimney's: FX_SMOKE_PUFFS soft puffs, each rising,
 *           growing, carried downwind and fading over FX_SMOKE_LIFE s, in
 *           turn (so the column is steady); lit by the key light and the
 *           sky, the lowest glowing with the fire at night.
 *   vortex  the Gyre's Eye over Rimewatch: FX_VORTEX_LAYERS discs stacked
 *           in a funnel, each a spiral of storm cloud wound round a clear
 *           eye, turning faster toward it, and round the eye its wall (a
 *           flared tube of cloud); edge-on a layer or the wall thickens as
 *           a slab would; lightning flickers in it, most at night.
 * Smoke and the vortex blend premultiplied (as the falls' mist); the beam's
 * alpha is 0, so the same blend adds it.
 *
 * Needs, from the raster shader: u (Frame), project(), display(), surfaceFog,
 * keyDir, keyCol, skyAmb, lampLevel, vnoise, hash3, rlReachFade.
 */

/** Bytes per effect in the GPU's list (struct Fx). */
export const FX_BYTES = 32;
/** Segments up a beam (a triangle strip of 2 (n + 1) vertices). */
export const FX_BEAM_SEG = 24;
/** Puffs per smoke column, and layers per vortex (each one instance of a 4-vertex strip). */
export const FX_SMOKE_PUFFS = 16;
export const FX_VORTEX_LAYERS = 6;
/** Segments round a vortex's eyewall (a triangle strip of 2 (n + 1) vertices). */
export const FX_EYEWALL_SEG = 48;

export function effectsWGSL(): string {
  return /* wgsl */ `
// ---- effects (structures.ts lists them, nearest kinds together) ---------------------------------------
struct Fx {
  p: vec4f, // position (chunk-local m), radius (beam: its core; smoke: a puff's; vortex: the Eye's)
  q: vec4f, // seed (0 … 1), rise speed (smoke, m/s), strength, unused
};
@group(1) @binding(42) var<storage, read> fxList: array<Fx>;

const FX_BEAM_SEG: u32 = ${FX_BEAM_SEG}u;
const FX_BEAM_H: f32 = 3200.0;
const FX_SMOKE_PUFFS: u32 = ${FX_SMOKE_PUFFS}u;
const FX_SMOKE_LIFE: f32 = 11.0;
const FX_VORTEX_LAYERS: u32 = ${FX_VORTEX_LAYERS}u;
const FX_EYEWALL_SEG: u32 = ${FX_EYEWALL_SEG}u;
/** A vortex's layers span ± this × its radius in height; its eyewall a little more. */
const FX_VORTEX_SPAN: f32 = 0.1;
const FX_WIND: vec2f = vec2f(0.87, 0.49);

struct FxOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,                     // camera-relative
  @location(1) q: vec2f,                       // smoke, vortex: the quad's −1 … 1; beam: across (core radii), up (0 … 1)
  @location(2) a: vec4f,                       // beam: light kept per core; smoke: age, puff; vortex: layer (0 … 1)
  @location(3) @interpolate(flat) id: u32,     // the effect
};

const FX_CORNERS = array<vec2f, 4>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0));

// The light of an additive effect seen through the haze: what the fog lets through of it.
fn fxThrough(col: vec3f, rel: vec3f) -> vec3f {
  let dist = max(length(rel), 1e-3);
  let rd = rel / dist;
  let p = u.camPos + rel;
  return (surfaceFog(col, dist, p.y, rd) - surfaceFog(vec3f(0.0), dist, p.y, rd)) * rlReachFade(rel);
}

// Added light as the display shows it: what it adds over black (the output's black level lifts
// black a little, which added over a whole quad would show as its square).
fn fxAdded(col: vec3f) -> vec3f { return max(display(col).rgb - display(vec3f(0.0)).rgb, vec3f(0.0)); }

@vertex
fn vsBeam(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxOut {
  let F = fxList[ii];
  let t = f32(vi >> 1u) / f32(FX_BEAM_SEG);
  let side = f32(vi & 1u) * 2.0 - 1.0;
  // (its segments closer together low down, where it is seen nearest)
  let rel = F.p.xyz + vec3f(0.0, FX_BEAM_H * t * t, 0.0) - u.camPos;
  let right = normalize(vec3f(-rel.z, 0.0, rel.x) + vec3f(1e-3, 0.0, 0.0));
  let core = F.p.w * (1.0 + 0.8 * t);
  let r = max(core, 0.7 * u.footprint * length(rel));
  var o: FxOut;
  o.rel = rel + right * side * 3.5 * r;
  o.pos = project(o.rel);
  o.q = vec2f(side * 3.5, t);
  o.a = vec4f(core / r, 0.0, 0.0, 0.0);
  o.id = ii;
  return o;
}

@fragment
fn fsBeam(i: FxOut) -> @location(0) vec4f {
  let F = fxList[i.id];
  let s = i.q.x;
  let t = i.q.y;
  let y = FX_BEAM_H * t * t;
  // (nothing at the ribbon's edges: the night's exposure would lift a faint tail into a band)
  let core = (exp(-5.0 * s * s) + 0.16 * exp(-1.6 * abs(s))) * (1.0 - smoothstep(2.4, 3.5, abs(s)));
  // brightest where it leaves the crystals, gone into the high sky; light running up it
  let along = smoothstep(0.0, 0.004, t) * (1.0 - smoothstep(0.12, 1.0, t));
  let pulse = 0.75 + 0.25 * sin(y * 0.021 - u.time * 2.6 + 6.28 * F.q.x);
  let I = core * along * pulse * i.a.x * F.q.z * (0.3 + 0.9 * lampLevel());
  // (kept within what the display shows as light: past it, its hue would turn)
  return vec4f(fxAdded(fxThrough(min(vec3f(0.8, 0.92, 1.0) * I, vec3f(1.1)), i.rel)), 0.0);
}

@vertex
fn vsSmoke(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxOut {
  let k = ii / FX_SMOKE_PUFFS;
  let m = ii % FX_SMOKE_PUFFS;
  let F = fxList[k];
  let q = FX_CORNERS[vi];
  let fm = f32(m);
  let age = fract(u.time / FX_SMOKE_LIFE + (fm + F.q.x) / f32(FX_SMOKE_PUFFS));
  let T = age * FX_SMOKE_LIFE;
  // rising ever slower as it cools, carried downwind more and more, wandering
  let rise = F.q.y * (1.0 - exp(-0.3 * T)) / 0.3;
  let wander = vec3f(sin(0.9 * T + 2.1 * fm), 0.0, cos(0.7 * T + 1.3 * fm)) * (0.4 + 0.5 * T);
  let drift = 0.6 * T + 0.09 * T * T;
  let c = F.p.xyz + vec3f(FX_WIND.x * drift, rise, FX_WIND.y * drift) + wander;
  let size = F.p.w * (0.75 + 0.42 * T);
  let toCam = normalize(u.camPos - c);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam) + vec3f(1e-4, 0.0, 0.0));
  let up = cross(toCam, right);
  // (each puff turned its own way, and turning slowly)
  let ang = 1.7 * fm + 0.12 * T;
  let qr = vec2f(q.x * cos(ang) - q.y * sin(ang), q.x * sin(ang) + q.y * cos(ang));
  var o: FxOut;
  o.rel = c + (right * qr.x + up * qr.y) * size - u.camPos;
  o.pos = project(o.rel);
  o.q = q;
  o.a = vec4f(age, fm, 0.0, 0.0);
  o.id = k;
  return o;
}

@fragment
fn fsSmoke(i: FxOut) -> @location(0) vec4f {
  let F = fxList[i.id];
  let age = i.a.x;
  let n = 0.5 + 0.5 * vnoise(i.q.x * 2.1 + i.a.y * 5.3, i.q.y * 2.1 + age * 2.5, 3301 + i32(i.a.y));
  let shape = 1.0 - smoothstep(0.2, 1.0, length(i.q) + 0.4 * (n - 0.5));
  let alpha = shape * smoothstep(0.0, 0.05, age) * (1.0 - age) * (1.0 - age) * 0.7 * F.q.z;
  if (alpha < 0.004) { discard; }
  let p = u.camPos + i.rel;
  let dist = length(i.rel);
  let L = keyDir();
  // lit from above and from the sun's side of the puff; the youngest glow with the fire at night
  let lit = 0.55 + 0.45 * clamp(dot(normalize(vec3f(i.q.x, i.q.y, 0.6)), normalize(vec3f(0.0, 0.7, 0.3) + L * 0.5)), 0.0, 1.0);
  // (soot drinks the light: darker still by night, but for the fire under it)
  let lamps = lampLevel();
  var col = vec3f(0.24, 0.23, 0.23) * (0.7 + 0.3 * n) * (1.0 - 0.6 * lamps) * (keyCol(p.y) * 0.4 * lit + skyAmb(vec3f(0.0, 1.0, 0.0), 0.8));
  col += vec3f(0.85, 0.3, 0.07) * (0.08 + 0.4 * lamps) * pow(1.0 - age, 2.5) * (1.0 - 0.5 * length(i.q));
  let shown = display(surfaceFog(col, dist, p.y, i.rel / dist));
  return vec4f(shown.rgb * alpha, alpha);
}

@vertex
fn vsVortex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxOut {
  let k = ii / FX_VORTEX_LAYERS;
  let m = ii % FX_VORTEX_LAYERS;
  let F = fxList[k];
  let q = FX_CORNERS[vi];
  let fm = f32(m) / f32(FX_VORTEX_LAYERS - 1u);
  // a funnel: the lower layers narrower
  let R = F.p.w * (0.5 + 0.5 * fm);
  var o: FxOut;
  o.rel = vec3f(F.p.x + q.x * R, F.p.y + (2.0 * fm - 1.0) * FX_VORTEX_SPAN * F.p.w, F.p.z + q.y * R) - u.camPos;
  o.pos = project(o.rel);
  o.q = q;
  o.a = vec4f(fm, 0.0, 0.0, 0.0);
  o.id = k;
  return o;
}

@fragment
fn fsVortex(i: FxOut) -> @location(0) vec4f {
  let F = fxList[i.id];
  let rho = length(i.q);
  if (rho > 1.0) { discard; }
  let fm = i.a.x;
  let t = u.time;
  // turning, faster toward the eye; wound in three arms
  let spin = t * (0.012 + 0.02 / max(rho, 0.2)) + 6.28 * F.q.x;
  let c = cos(spin);
  let s = sin(spin);
  let w = vec2f(i.q.x * c - i.q.y * s, i.q.x * s + i.q.y * c);
  let th = select(0.0, atan2(w.y, w.x), rho > 1e-4);
  let arm = 0.5 + 0.5 * sin(3.0 * th + 7.5 * log(max(rho, 0.02)));
  let n = 0.5 + 0.5 * vnoise(w.x * 7.0 + fm * 3.0, w.y * 7.0, 7711 + i32(fm * 8.0));
  let n2 = 0.5 + 0.5 * vnoise(w.x * 23.0, w.y * 23.0 + fm * 5.0, 7717);
  let cloud = smoothstep(0.35, 0.85, 0.55 * arm + 0.45 * n + 0.15 * (n2 - 0.5));
  let eye = smoothstep(0.1, 0.32, rho);
  let edge = 1.0 - smoothstep(0.6, 1.0, rho);
  let a0 = cloud * eye * edge * 0.42 * F.q.z;
  // edge-on, a layer is a slab: more cloud along a grazing line of sight (capped)
  let rd = normalize(i.rel);
  let alpha = 1.0 - pow(1.0 - a0, 1.0 / max(abs(rd.y), 0.22));
  if (alpha < 0.004) { discard; }
  return fxStorm(i.rel, i.id, fm, n2, cloud, i.q, alpha);
}

// Storm cloud, shown: lit from above, darker low down (fm 0 its bottom, 1 its top); lightning
// flickering through it now and then where it is thick (at q, its place across the Eye, −1 … 1).
fn fxStorm(rel: vec3f, id: u32, fm: f32, n2: f32, cloud: f32, q: vec2f, alpha: f32) -> vec4f {
  let t = u.time;
  let p = u.camPos + rel;
  var col = vec3f(0.66, 0.69, 0.76) * (0.55 + 0.45 * fm) * (0.75 + 0.25 * n2) * (keyCol(p.y) * 0.45 + skyAmb(vec3f(0.0, 1.0, 0.0), 0.9));
  let beat = floor(t * 5.0);
  let strike = f32(hash3(i32(beat), i32(id), 7723) & 1023u) / 1023.0;
  let spot = vec2f(f32(hash3(i32(beat), 1, 7727) & 1023u), f32(hash3(i32(beat), 2, 7727) & 1023u)) / 511.5 - 1.0;
  let flash = step(0.86, strike) * (1.0 - smoothstep(0.0, 0.35, length(q - spot * 0.7))) * fract(t * 5.0) * (1.0 - fract(t * 5.0)) * 4.0;
  col += vec3f(0.75, 0.82, 1.0) * flash * (0.35 + 2.5 * lampLevel()) * cloud;
  let dist = length(rel);
  let shown = display(surfaceFog(col, dist, p.y, rel / dist));
  return vec4f(shown.rgb * alpha, alpha);
}

@vertex
fn vsEyewall(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxOut {
  let F = fxList[ii];
  let a = 6.2831853 * f32(vi >> 1u) / f32(FX_EYEWALL_SEG);
  let h = f32(vi & 1u);
  // flaring as it rises, as the eye in the layers widens
  let r = F.p.w * mix(0.11, 0.24, h);
  var o: FxOut;
  o.rel = vec3f(F.p.x + cos(a) * r, F.p.y + (2.0 * h - 1.0) * 1.2 * FX_VORTEX_SPAN * F.p.w, F.p.z + sin(a) * r) - u.camPos;
  o.pos = project(o.rel);
  o.q = vec2f(cos(a), sin(a)) * mix(0.11, 0.24, h);
  o.a = vec4f(h, 0.0, 0.0, 0.0);
  o.id = ii;
  return o;
}

@fragment
fn fsEyewall(i: FxOut) -> @location(0) vec4f {
  let F = fxList[i.id];
  let h = i.a.x;
  let t = u.time;
  // the wall's cloud turning with the Eye (fastest here), streaked as it is drawn up round it
  let spin = t * 0.09 + 6.28 * F.q.x;
  let c = cos(spin);
  let s = sin(spin);
  let w = vec2f(i.q.x * c - i.q.y * s, i.q.x * s + i.q.y * c) / max(length(i.q), 1e-3);
  let n = 0.5 + 0.5 * vnoise(w.x * 3.0 + h * 1.5, w.y * 3.0 - h * 4.0, 7741);
  let n2 = 0.5 + 0.5 * vnoise(w.x * 11.0 - h * 3.0, w.y * 11.0 + h * 9.0, 7743);
  let cloud = smoothstep(0.3, 0.8, 0.7 * n + 0.3 * n2);
  let a0 = cloud * sin(3.14159 * h) * 0.38 * F.q.z;
  // seen along it (at the eye's edges), more cloud in the line of sight (capped)
  let rd = normalize(i.rel);
  let facing = abs(dot(rd, normalize(vec3f(i.q.x, 0.0, i.q.y))));
  let alpha = 1.0 - pow(1.0 - a0, 1.0 / max(facing, 0.25));
  if (alpha < 0.004) { discard; }
  return fxStorm(i.rel, i.id, h, n2, cloud, i.q, alpha);
}
`;
}
