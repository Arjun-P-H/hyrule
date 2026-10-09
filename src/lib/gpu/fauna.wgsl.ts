/**
 * The living sky's birds in WGSL (skyworld/fauna.ts plans the flocks; gpu/fauna.ts draws them):
 * FLOCK_BIRDS instances a flock, each a bird of seven triangles made here from its vertex index.
 * ---------------------------------------------------------------------------
 * A bird flies round its flock's circle a little off it (its own place in
 * the flock from a hash: ahead or behind, in or out, up or down), wandering
 * slowly, heading along its way and banked into the turn. Its wings beat at
 * its species' rate (the elbow leading the tip) or hold still, a little
 * raised, while it glides; the gliders (hawks, gulls) glide most of the
 * time. Past its flock's count, or roosting by night, a bird is not drawn.
 * Shaded flat (body and wings each their species' colour), lit, fogged.
 *
 * Needs, from the raster shader: u (Frame), project(), display(), surfaceFog,
 * keyDir, keyCol, skyAmb, lampLevel, hash3, rlFogged, underwater, W_LEVEL.
 */

import { FLOCK_BIRDS, SPECIES } from '../skyworld/fauna';

/** Bytes per flock in the GPU's list (struct Flock). */
export const FLOCK_BYTES = 32;
/** Vertices per bird: seven triangles. */
export const BIRD_VERTS = 21;

// per species (skyworld/fauna.ts SPECIES): half-span scale (m), wing beats a second, how much it glides, wings swept back
const SP: Record<(typeof SPECIES)[number], { size: number; flap: number; glide: number; sweep: number; body: number[]; wing: number[] }> = {
  swift: { size: 0.55, flap: 8.0, glide: 0.35, sweep: 0.5, body: [0.07, 0.07, 0.09], wing: [0.1, 0.1, 0.13] },
  hawk: { size: 1.9, flap: 2.0, glide: 0.88, sweep: 0.12, body: [0.36, 0.25, 0.15], wing: [0.3, 0.21, 0.13] },
  heron: { size: 1.5, flap: 2.4, glide: 0.45, sweep: 0.05, body: [0.92, 0.92, 0.9], wing: [0.88, 0.89, 0.9] },
  crow: { size: 0.9, flap: 4.5, glide: 0.3, sweep: 0.12, body: [0.04, 0.04, 0.05], wing: [0.05, 0.05, 0.06] },
  gull: { size: 1.1, flap: 3.2, glide: 0.65, sweep: 0.25, body: [0.92, 0.93, 0.94], wing: [0.62, 0.64, 0.68] },
  dove: { size: 0.8, flap: 6.0, glide: 0.35, sweep: 0.15, body: [0.95, 0.95, 0.93], wing: [0.9, 0.9, 0.9] },
  snowhawk: { size: 1.7, flap: 2.2, glide: 0.85, sweep: 0.1, body: [0.85, 0.86, 0.88], wing: [0.7, 0.72, 0.76] },
};
const n = (x: number) => (Number.isInteger(x) ? `${x}.0` : `${x}`);
const table = (pick: (s: (typeof SP)[keyof typeof SP]) => number) => `array<f32, ${SPECIES.length}>(${SPECIES.map((k) => n(pick(SP[k]))).join(', ')})`;
const colours = (pick: (s: (typeof SP)[keyof typeof SP]) => number[]) =>
  `array<vec3f, ${SPECIES.length}>(${SPECIES.map((k) => `vec3f(${pick(SP[k]).map(n).join(', ')})`).join(', ')})`;

export function faunaWGSL(): string {
  return /* wgsl */ `
// ---- birds (fauna.ts lists the flocks round the camera) ------------------------------------------
struct Flock {
  p: vec4f, // its circle's centre (chunk-local m), radius
  q: vec4u, // speed (f32 bits, m/s, signed), count | species << 8 | flies by night << 15, seed, unused
};
@group(1) @binding(43) var<storage, read> flocks: array<Flock>;

const FLOCK_BIRDS: u32 = ${FLOCK_BIRDS}u;
const BIRD_SIZE = ${table((s) => s.size)};
const BIRD_FLAP = ${table((s) => s.flap)};
const BIRD_GLIDE = ${table((s) => s.glide)};
const BIRD_SWEEP = ${table((s) => s.sweep)};
const BIRD_BODY = ${colours((s) => s.body)};
const BIRD_WING = ${colours((s) => s.wing)};
// the corners of its seven triangles: the body's two halves, the tail, each wing's inner and outer part
const BIRD_CORNER = array<u32, 21>(0u, 1u, 2u, 0u, 2u, 3u, 4u, 5u, 6u, 7u, 9u, 8u, 9u, 10u, 8u, 7u, 8u, 9u, 9u, 8u, 10u);

struct BirdOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) n: vec3f,
  @location(2) @interpolate(flat) c: vec3f,
};

// A corner in the bird's own frame (x right, y up, z ahead; ~1 m long before its species' scale);
// the wings' elbow at ye, their tips at yt.
fn birdCorner(c: u32, side: f32, ye: f32, yt: f32, sweep: f32) -> vec3f {
  switch c {
    case 0u: { return vec3f(0.0, 0.0, 0.32); }
    case 1u: { return vec3f(-0.065, 0.0, 0.02); }
    case 2u: { return vec3f(0.0, 0.0, -0.28); }
    case 3u: { return vec3f(0.065, 0.0, 0.02); }
    case 4u: { return vec3f(0.0, 0.0, -0.22); }
    case 5u: { return vec3f(-0.11, 0.0, -0.46); }
    case 6u: { return vec3f(0.11, 0.0, -0.46); }
    case 7u: { return vec3f(0.05 * side, 0.0, 0.12); }
    case 8u: { return vec3f(0.05 * side, 0.0, -0.12); }
    case 9u: { return vec3f(0.42 * side, ye, -0.02 - 0.3 * sweep); }
    default: { return vec3f(0.85 * side, yt, -0.2 - 0.6 * sweep); }
  }
}

@vertex
fn vsBird(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BirdOut {
  let F = flocks[ii / FLOCK_BIRDS];
  let b = ii % FLOCK_BIRDS;
  let sp = (F.q.y >> 8u) & 15u;
  var o: BirdOut;
  o.c = vec3f(0.0);
  o.n = vec3f(0.0, 1.0, 0.0);
  o.rel = vec3f(0.0);
  if (b >= (F.q.y & 255u) || (((F.q.y >> 15u) & 1u) == 0u && lampLevel() > 0.6)) {
    o.pos = vec4f(2.0, 2.0, 0.5, 1.0); // (outside the view: not drawn)
    return o;
  }
  let fb = f32(b);
  let h = f32(hash3(i32(b), i32(F.q.z), 4409) & 0xffffu) / 65535.0;
  let h2 = f32(hash3(i32(b), i32(F.q.z), 4411) & 0xffffu) / 65535.0;
  let h3 = f32(hash3(i32(b), i32(F.q.z), 4413) & 0xffffu) / 65535.0;
  let t = u.time;
  let speed = bitcast<f32>(F.q.x);
  let R = F.p.w;
  // its place round the circle: the flock together, each a little ahead or behind, in or out, wandering
  let th = speed / R * t + f32(F.q.z & 1023u) * 0.0061 + (h - 0.5) * min(60.0 / R, 1.2) + 0.06 * sin(0.7 * t + fb);
  let rr = R + (h2 - 0.5) * min(16.0, 0.3 * R) + 4.0 * sin(0.31 * t + 2.0 * fb);
  let y = F.p.y + (h3 - 0.5) * 10.0 + 3.0 * sin(0.5 * t + 1.3 * fb) + 12.0 * sin(0.05 * t + f32(F.q.z & 255u));
  let at = vec3f(F.p.x + rr * cos(th), y, F.p.z + rr * sin(th));
  // heading along its way round, banked into the turn (toward the circle's middle)
  let dir = sign(speed);
  let fwd = vec3f(-sin(th), 0.0, cos(th)) * dir;
  let right0 = vec3f(fwd.z, 0.0, -fwd.x);
  let roll = clamp(1.5 * speed * speed / (9.8 * R), 0.08, 0.6);
  let up = vec3f(0.0, cos(roll), 0.0) - dir * right0 * sin(roll);
  let right = cross(up, fwd);
  // its wings: beating (the elbow leading the tip), or held a little raised in a glide — for its
  // species' share of the time (the gliders most of it)
  let g = 1.0 - 2.0 * BIRD_GLIDE[sp];
  let glide = smoothstep(g - 0.2, g + 0.2, sin(0.23 * t + 6.2831853 * h));
  let flap = 1.0 - glide;
  let ph = t * BIRD_FLAP[sp] * 6.2831853 + 6.2831853 * h2;
  let ye = 0.22 * sin(ph) * flap + 0.04 * glide;
  let yt = 0.5 * sin(ph - 0.8) * flap + 0.1 * glide;
  let side = select(1.0, -1.0, vi >= 9u && vi < 15u);
  let k = BIRD_SIZE[sp];
  let tri = vi / 3u;
  let p0 = birdCorner(BIRD_CORNER[tri * 3u], side, ye, yt, BIRD_SWEEP[sp]);
  let p1 = birdCorner(BIRD_CORNER[tri * 3u + 1u], side, ye, yt, BIRD_SWEEP[sp]);
  let p2 = birdCorner(BIRD_CORNER[tri * 3u + 2u], side, ye, yt, BIRD_SWEEP[sp]);
  let L = birdCorner(BIRD_CORNER[vi], side, ye, yt, BIRD_SWEEP[sp]) * k;
  let nl = normalize(cross(p1 - p0, p2 - p0) + vec3f(0.0, 1e-5, 0.0));
  o.rel = at + right * L.x + up * L.y + fwd * L.z - u.camPos;
  o.pos = project(o.rel);
  o.n = right * nl.x + up * nl.y + fwd * nl.z;
  o.c = select(BIRD_WING[sp], BIRD_BODY[sp], tri < 3u);
  return o;
}

@fragment
fn fsBird(i: BirdOut) -> @location(0) vec4f {
  let dist = length(i.rel);
  let rd = i.rel / dist;
  if (u.camPos.y < u.camWater) { return underwater(rd); }
  let fogged = rlFogged(i.rel);
  if (fogged.a >= 0.0) { return fogged; }
  var nn = normalize(i.n);
  if (dot(nn, rd) > 0.0) { nn = -nn; }
  let p = u.camPos + i.rel;
  // (light through the feathers on the far side, a little)
  let lit = max(dot(nn, keyDir()), 0.0) * 0.85 + 0.12;
  let col = i.c * (keyCol(p.y) * lit + skyAmb(nn, 1.0));
  return display(surfaceFog(col, dist, p.y, rd));
}
`;
}
