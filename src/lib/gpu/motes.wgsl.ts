/**
 * Motes in WGSL: what drifts in the air round the camera (skyworld/motes.ts
 * says what, gpu/motes.ts draws it), MOTES_N instances a kind, each a quad.
 * ---------------------------------------------------------------------------
 * Every mote has a place in a box MOTE_BOX metres a side round the camera
 * (from a hash of its index) that drifts with time and wraps round the box,
 * so the field never ends and never pops as the camera moves; motes fade out
 * toward the box's faces. How many of a kind's MOTES_N show is its density.
 *   petals, leaves  tumbling as they fall, carried downwind, fluttering;
 *   snow            soft flakes, falling and wavering;
 *   embers          sparks rising and flickering out (added light);
 *   butterflies     wandering a metre or two over the ground, wings beating
 *                   (the quad folded along its middle), by day;
 *   fireflies       hovering low, wandering, blinking (added light), by dusk.
 * Petals, leaves, snow and butterflies blend premultiplied; embers and
 * fireflies add (alpha 0).
 *
 * Needs, from the raster shader: u (Frame), project(), display(), surfaceFog,
 * keyDir, keyCol, skyAmb, lampLevel, hash3; and effects.wgsl.ts's fxAdded.
 */

import { MOTE } from '../skyworld/motes';

/** Motes a kind may show (each 4 vertices), and the side of the box round the camera they live in (m). */
export const MOTES_N = 3072;
const MOTE_BOX = 44;
/** Bytes of the motes' uniform. */
export const MOTES_BYTES = 48;

export function motesWGSL(): string {
  return /* wgsl */ `
// ---- motes (motes.ts: the air round the camera) ------------------------------------------------------
struct Motes {
  kinds: vec4f, // the falling kind, the day's, the dusk's (MOTE_*), the ground's height (m)
  dens: vec4f,  // their densities (0 … 1), unused
  wind: vec4f,  // the wind (x, z; m/s), unused
};
@group(1) @binding(44) var<uniform> motes: Motes;

const MOTES_N: u32 = ${MOTES_N}u;
const MOTE_BOX: f32 = ${MOTE_BOX}.0;
const MOTE_PETALS: u32 = ${MOTE.petals}u;
const MOTE_SNOW: u32 = ${MOTE.snow}u;
const MOTE_EMBERS: u32 = ${MOTE.embers}u;
const MOTE_LEAVES: u32 = ${MOTE.leaves}u;
const MOTE_FIREFLIES: u32 = ${MOTE.fireflies}u;
const MOTE_BUTTERFLIES: u32 = ${MOTE.butterflies}u;
// the share of MOTES_N each kind shows at density 1 (none, petals, snow, embers, leaves, fireflies, butterflies)
const MOTE_SHARE = array<f32, 7>(0.0, 1.0, 1.0, 0.35, 0.5, 0.3, 0.07);

struct MoteOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) q: vec2f,                     // the quad's −1 … 1
  @location(2) @interpolate(flat) k: vec4u,  // kind, index, unused, unused
  @location(3) @interpolate(flat) a: vec4f,  // fade (distance and life), colour pick, blink, unused
};

fn moteH(i: u32, salt: i32) -> f32 { return f32(hash3(i32(i), salt, 6101) & 0xffffu) / 65535.0; }

@vertex
fn vsMote(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MoteOut {
  let slot = ii / MOTES_N;
  let i = ii % MOTES_N;
  let kind = u32(motes.kinds[slot] + 0.5);
  var o: MoteOut;
  o.rel = vec3f(0.0);
  o.q = vec2f(0.0);
  o.k = vec4u(kind, i, 0u, 0u);
  o.a = vec4f(0.0);
  var dens = motes.dens[slot];
  // (butterflies by day, fireflies from dusk)
  if (kind == MOTE_BUTTERFLIES) { dens *= 1.0 - lampLevel(); }
  if (kind == MOTE_FIREFLIES) { dens *= lampLevel(); }
  if (kind == 0u || f32(i) >= dens * MOTE_SHARE[min(kind, 6u)] * f32(MOTES_N)) {
    o.pos = vec4f(2.0, 2.0, 0.5, 1.0); // (outside the view: not drawn)
    return o;
  }
  let q = array<vec2f, 4>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0))[vi];
  let h = vec3f(moteH(i, 1), moteH(i, 2), moteH(i, 3));
  let h4 = moteH(i, 4);
  let t = u.time;
  let w = vec3f(motes.wind.x, 0.0, motes.wind.y);
  let c = u.camPos;
  let corner = c - vec3f(0.5 * MOTE_BOX);
  var p: vec3f;
  var size = 0.1;
  var life = 1.0;
  let fi = f32(i);
  if (kind == MOTE_FIREFLIES || kind == MOTE_BUTTERFLIES) {
    // wandering over the ground, the box wrapping round in x and z only
    let lift = select(0.6 + 2.2 * h.y, 0.3 + 3.2 * h.y, kind == MOTE_FIREFLIES);
    let wander = vec3f(sin(0.31 * t + 6.28 * h4) * 2.5 + sin(0.83 * t + fi) * 0.6, sin(0.9 * t + 3.0 * h.x) * 0.4, cos(0.27 * t + 6.28 * h.y) * 2.5 + cos(0.71 * t + fi) * 0.6);
    let base = h * MOTE_BOX + wander;
    p = vec3f(corner.x + (base.x - corner.x - MOTE_BOX * floor((base.x - corner.x) / MOTE_BOX)), motes.kinds.w + lift + wander.y,
              corner.z + (base.z - corner.z - MOTE_BOX * floor((base.z - corner.z) / MOTE_BOX)));
    size = select(0.12, 0.05, kind == MOTE_FIREFLIES);
  } else {
    var vel = w * 1.1 + vec3f(0.0, -0.45, 0.0);
    var flut = vec3f(sin(2.3 * t + fi) * 0.35, 0.0, cos(1.9 * t + 1.7 * fi) * 0.35);
    if (kind == MOTE_SNOW) { vel = w * 0.6 + vec3f(0.0, -0.9, 0.0); flut *= 0.6; size = 0.05; }
    if (kind == MOTE_LEAVES) { vel = w * 1.0 + vec3f(0.0, -0.8, 0.0); size = 0.12; }
    if (kind == MOTE_EMBERS) { vel = w * 0.5 + vec3f(0.0, 0.9, 0.0); flut *= 0.5; size = 0.06; }
    let base = h * MOTE_BOX + vel * t + flut;
    let d = base - corner;
    p = corner + d - MOTE_BOX * floor(d / MOTE_BOX);
    // (a spark's life: flickering out as it rises)
    if (kind == MOTE_EMBERS) { life = 1.0 - fract(0.15 * t + h4); }
  }
  let rel = p - c;
  let dist = length(rel);
  // fading toward the box's faces, and right at the eye (a flake on the lens would fill the view)
  let fade = (1.0 - smoothstep(0.32, 0.48, max(abs(rel.x), max(abs(rel.y), abs(rel.z))) / MOTE_BOX)) * life * smoothstep(0.4, 1.6, dist);
  // oriented: petals and leaves tumbling; butterflies flat, their wings beating; the rest facing the camera
  let toCam = -rel / max(dist, 1e-3);
  var right = normalize(cross(vec3f(0.0, 1.0, 0.0), toCam) + vec3f(1e-4, 0.0, 0.0));
  var up = cross(toCam, right);
  var off = right * q.x + up * q.y;
  if (kind == MOTE_PETALS || kind == MOTE_LEAVES) {
    let a = 1.7 * t + 6.28 * h.x;
    let b = 1.3 * t + 6.28 * h.y;
    let ax = vec3f(cos(a), sin(a) * cos(b), sin(a) * sin(b));
    let ay = normalize(cross(ax, vec3f(sin(b), cos(b), 0.3)));
    off = ax * q.x + ay * q.y * 0.7;
  } else if (kind == MOTE_BUTTERFLIES) {
    let head = 0.31 * t + 6.28 * h4;
    let fwd = vec3f(cos(head), 0.0, sin(head));
    let side = vec3f(-fwd.z, 0.0, fwd.x);
    let beat = 0.9 * sin(14.0 * t + 6.28 * h.x);
    off = side * q.x * cos(beat) + vec3f(0.0, abs(q.x) * sin(beat), 0.0) + fwd * q.y * 0.7;
  }
  o.rel = rel + off * size;
  o.pos = project(o.rel);
  o.q = q;
  o.a = vec4f(fade, h4, 6.28 * h.z, 0.0);
  return o;
}

@fragment
fn fsMote(i: MoteOut) -> @location(0) vec4f {
  let kind = i.k.x;
  let dist = length(i.rel);
  let rd = i.rel / max(dist, 1e-3);
  let p = u.camPos + i.rel;
  let r = length(i.q);
  let light = keyCol(p.y) * 0.6 + skyAmb(vec3f(0.0, 1.0, 0.0), 0.9);
  if (kind == MOTE_FIREFLIES || kind == MOTE_EMBERS) {
    // added light: a soft point that blinks (fireflies) or flickers (sparks); exactly nothing at the
    // quad's edge (the night's exposure would lift a faint tail into a square)
    let glow = max(exp(-4.0 * r * r) - 0.0183, 0.0) * (1.0 - smoothstep(0.7, 1.0, r));
    var col: vec3f;
    if (kind == MOTE_FIREFLIES) { col = vec3f(0.55, 0.9, 0.12) * 0.9 * pow(0.5 + 0.5 * sin(1.7 * u.time + i.a.z), 4.0); }
    else { col = vec3f(1.0, 0.45, 0.1) * (1.4 + 0.6 * sin(23.0 * u.time + i.a.z)); }
    let lit = col * glow * i.a.x;
    return vec4f(fxAdded(surfaceFog(lit, dist, p.y, rd) - surfaceFog(vec3f(0.0), dist, p.y, rd)), 0.0);
  }
  var col: vec3f;
  var alpha: f32;
  if (kind == MOTE_PETALS) {
    alpha = 1.0 - smoothstep(0.6, 1.0, length(i.q * vec2f(1.0, 1.35)));
    col = mix(vec3f(0.93, 0.42, 0.6), vec3f(0.98, 0.84, 0.9), i.a.y * i.a.y) * light;
  } else if (kind == MOTE_LEAVES) {
    alpha = 1.0 - smoothstep(0.55, 1.0, length(i.q * vec2f(1.0, 1.6)));
    col = mix(vec3f(0.25, 0.42, 0.1), vec3f(0.62, 0.48, 0.12), i.a.y) * light;
  } else if (kind == MOTE_SNOW) {
    alpha = (1.0 - smoothstep(0.2, 1.0, r)) * 0.85;
    col = vec3f(0.95, 0.96, 1.0) * light;
  } else {
    // a butterfly: two rounded wings, coloured by its kind, darker at their edges
    let wing = abs(i.q.x);
    alpha = 1.0 - smoothstep(0.75, 1.0, length(vec2f(wing - 0.45, i.q.y) * vec2f(1.6, 1.0)));
    alpha = max(alpha, 1.0 - smoothstep(0.05, 0.12, wing) * step(abs(i.q.y), 0.7));
    let pick = u32(i.a.y * 4.0);
    let wc = array<vec3f, 4>(vec3f(0.95, 0.55, 0.1), vec3f(0.25, 0.45, 0.95), vec3f(0.98, 0.9, 0.35), vec3f(0.95, 0.95, 0.92))[min(pick, 3u)];
    col = mix(wc, vec3f(0.05), smoothstep(0.55, 0.9, length(vec2f(wing - 0.45, i.q.y) * vec2f(1.6, 1.0)))) * light;
  }
  alpha *= i.a.x;
  if (alpha < 0.01) { discard; }
  let shown = display(surfaceFog(col, dist, p.y, rd));
  return vec4f(shown.rgb * alpha, alpha);
}
`;
}
