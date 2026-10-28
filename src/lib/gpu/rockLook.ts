/**
 * The rock's detail, drawn so no face stretches it (terrain.wgsl.ts groundLook). A height field's
 * texture laid on the ground plane alone smears down a cliff, 1/cos(slope) times (3.9× at 75°): the
 * melted faces of the karst towers and the sea cliffs. Instead the detail is drawn on five planes —
 * the ground and four upright ones, 45° apart round the compass — and each pixel takes the two that
 * face it most, weighted by how much more they face it than the third, so the weights turn smoothly
 * from one pair to the next. The worst stretch over every face that looks up is then 1.47× (at 43°
 * up, between two upright planes and the ground), where the ground plane's own is unbounded.
 *
 * This file is the WGSL's twin: the planes and weights the shader is built from, and the stretch the
 * tests measure (tests/forms.test.ts).
 */

const S = Math.SQRT1_2;
/** The planes' normals: the ground, then four upright ones. */
export const ROCK_AXES: readonly (readonly [number, number, number])[] = [
  [0, 1, 0],
  [1, 0, 0],
  [S, 0, S],
  [0, 0, 1],
  [-S, 0, S],
];
/** Each plane's own coordinates (u, v): the ground's x and z; an upright plane's run along it and height. */
export const ROCK_U: readonly (readonly [number, number, number])[] = [[1, 0, 0], [0, 0, -1], [-S, 0, S], [1, 0, 0], [-S, 0, -S]];
export const ROCK_V: readonly (readonly [number, number, number])[] = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]];
/** How sharply the weights favour the plane that faces the face most. */
export const ROCK_SHARP = 4;

/** The two planes a face of unit normal n takes (indices into ROCK_AXES) and their weights, summing to 1. */
export function rockWeights(n: readonly [number, number, number]): { i: [number, number]; w: [number, number]; c: [number, number] } {
  let c1 = -1, c2 = -1, c3 = -1, i1 = 0, i2 = 0;
  for (let k = 0; k < ROCK_AXES.length; k++) {
    const a = ROCK_AXES[k];
    const c = Math.abs(n[0] * a[0] + n[1] * a[1] + n[2] * a[2]);
    if (c > c1) { c3 = c2; c2 = c1; i2 = i1; c1 = c; i1 = k; } else if (c > c2) { c3 = c2; c2 = c; i2 = k; } else if (c > c3) { c3 = c; }
  }
  const b1 = Math.pow(c1 - c3, ROCK_SHARP), b2 = Math.pow(c2 - c3, ROCK_SHARP), sum = Math.max(b1 + b2, 1e-12);
  return { i: [i1, i2], w: [b1 / sum, b2 / sum], c: [c1, c2] };
}

/**
 * How far the detail stretches on a face of unit normal n: a plane's drawing laid on a face it meets
 * at cosine c is drawn out 1/c times across their shared line, so the blend's is the weighted mean.
 */
export function rockStretch(n: readonly [number, number, number]): number {
  const { w, c } = rockWeights(n);
  return w[0] / Math.max(c[0], 1e-9) + w[1] / Math.max(c[1], 1e-9);
}

const v3 = (a: readonly number[]) => `vec3f(${a.map((x) => (Number.isInteger(x) ? `${x}.0` : String(x))).join(', ')})`;

/** WGSL: the planes and rockDetail (needs vnoise). */
export const ROCK_WGSL = /* wgsl */ `
const ROCK_AX = array<vec3f, ${ROCK_AXES.length}>(${ROCK_AXES.map(v3).join(', ')});
const ROCK_U = array<vec3f, ${ROCK_AXES.length}>(${ROCK_U.map(v3).join(', ')});
const ROCK_V = array<vec3f, ${ROCK_AXES.length}>(${ROCK_V.map(v3).join(', ')});

// One plane's drawing at its coordinates uv (m): broad patches (7 m) and the cracks between blocks
// (a ridged 2 m octave), each faded to its mean as it shrinks under the footprint dp.
fn rockPlane(uv: vec2f, dp: f32, seed: i32) -> f32 {
  let k1 = 1.0 - smoothstep(1.75, 3.5, dp);
  if (k1 <= 0.0) { return 0.0; }
  var d = 0.55 * k1 * vnoise(uv.x / 7.0, uv.y / 7.0, seed);
  let k2 = 1.0 - smoothstep(0.5, 1.0, dp);
  if (k2 > 0.0) { d += 0.45 * k2 * (0.5 - abs(vnoise(uv.x / 1.9, uv.y / 1.9, seed + 1))) * 2.0; }
  return d;
}

// The rock's detail (about −1 … 1) at q (m) on a face of unit normal n, seen at footprint dp:
// rockLook.ts rockWeights, the two planes that face n most.
fn rockDetail(q: vec3f, n: vec3f, dp: f32) -> f32 {
  if (dp >= 3.5) { return 0.0; }
  var c1 = -1.0;
  var c2 = -1.0;
  var c3 = -1.0;
  var i1 = 0;
  var i2 = 0;
  for (var k = 0; k < ${ROCK_AXES.length}; k++) {
    let c = abs(dot(n, ROCK_AX[k]));
    if (c > c1) { c3 = c2; c2 = c1; i2 = i1; c1 = c; i1 = k; } else if (c > c2) { c3 = c2; c2 = c; i2 = k; } else if (c > c3) { c3 = c; }
  }
  let b1 = pow(c1 - c3, ${ROCK_SHARP}.0);
  let b2 = pow(c2 - c3, ${ROCK_SHARP}.0);
  let d1 = rockPlane(vec2f(dot(q, ROCK_U[i1]), dot(q, ROCK_V[i1])), dp, 4481 + 2 * i1);
  var d2 = 0.0;
  if (b2 > 1e-4 * b1) { d2 = rockPlane(vec2f(dot(q, ROCK_U[i2]), dot(q, ROCK_V[i2])), dp, 4481 + 2 * i2); }
  return (b1 * d1 + b2 * d2) / max(b1 + b2, 1e-12);
}
`;
