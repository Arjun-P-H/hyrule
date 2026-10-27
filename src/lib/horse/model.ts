/**
 * The horse's model (public/models/horse.bin, built by scripts/horse/build.py): its mesh, bound to 44 joints, and
 * its clips baked into skinning matrices (3×4, rows) a joint a frame, in the horse's own frame (metres, +Y up,
 * facing +Z, the origin on the ground under the girth).
 * ---------------------------------------------------------------------------
 * A pose is a palette of JOINT matrices (12 floats each): a clip's frame, two frames lerped between, two clips
 * cross-faded (lerping the matrices: a short blend between poses this alike stays rigid enough).
 */

const HORSE_URL = '/models/horse.bin';
export const HORSE_TEXTURE_URL = '/models/horse.jpg';
/** Bytes a vertex: position f32×3, normal f32×3, uv f32×2, joints u8×4, weights unorm8×4, tags unorm8×4 (x: mane and tail). */
export const HORSE_VERTEX_BYTES = 44;
/** Floats a joint's matrix. */
export const MAT = 12;

type HorseClip = { name: string; frames: number; loop: boolean; first: number; duration: number };

export type HorseModel = {
  vertices: Uint8Array;
  indices: Uint16Array | Uint32Array;
  joints: number;
  fps: number;
  clips: HorseClip[];
  clip: Record<string, HorseClip>;
  /** Every clip's frames, back to back: (frame, joint) → MAT floats. */
  frames: Float32Array;
  /** The rider's seat in the bind pose, and the joint that carries it. */
  saddle: [number, number, number];
  saddleJoint: number;
  bounds: [[number, number, number], [number, number, number]];
  /** Each joint's name, parent (−1: the hips) and place in the bind pose (for the turns the game adds: turnJoints). */
  names: string[];
  parents: number[];
  rest: Array<[number, number, number]>;
  /** The leaves' places in the bind pose by name (the colossus's fists, toes and crown; the horse has none). */
  tips?: Record<string, [number, number, number]>;
  /** A collision mesh (the colossus's): bind-pose positions, joints and weights (4 a vertex), triangles. */
  coll?: { pos: Float32Array; joints: Uint8Array; weights: Uint8Array; tris: Uint16Array };
};

export function parseHorse(buf: ArrayBuffer): HorseModel {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x31535248) throw new Error('horse.bin: not HRS1'); // 'HRS1'
  const n = dv.getUint32(4, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  let at = 8 + n;
  const vertices = new Uint8Array(buf, at, head.vertices * HORSE_VERTEX_BYTES);
  at += vertices.byteLength;
  const indices = head.index32 ? new Uint32Array(buf, at, head.indices) : new Uint16Array(buf, at, head.indices);
  at += indices.byteLength + (indices.byteLength & 2);
  const joints = head.joints.length;
  let first = 0;
  const clips: HorseClip[] = head.clips.map((c: { name: string; frames: number; loop: boolean }) => {
    const clip = { ...c, first, duration: (c.frames - 1) / head.fps };
    first += c.frames;
    return clip;
  });
  const frames = new Float32Array(buf, at, first * joints * MAT);
  at += frames.byteLength;
  let coll: HorseModel['coll'];
  if (head.coll) {
    const n = head.coll.vertices;
    const pos = new Float32Array(buf, at, n * 3);
    const jw = new Uint8Array(buf, at + n * 12, n * 8);
    const joints = new Uint8Array(n * 4), weights = new Uint8Array(n * 4);
    for (let v = 0; v < n; v++) for (let k = 0; k < 4; k++) { joints[4 * v + k] = jw[8 * v + k]; weights[4 * v + k] = jw[8 * v + 4 + k]; }
    coll = { pos, joints, weights, tris: new Uint16Array(buf, at + n * 20, head.coll.indices) };
  }
  return {
    vertices, indices, joints, fps: head.fps, clips, clip: Object.fromEntries(clips.map((c) => [c.name, c])), frames,
    saddle: head.saddle, saddleJoint: head.saddleJoint, bounds: head.bounds,
    names: head.joints, parents: head.parents, rest: head.rest, tips: head.tips, coll,
  };
}

const loading = new Map<string, Promise<HorseModel>>();

/** A model in this format (the horse's, the colossus's), fetched once (a failed fetch is retried on the next call). */
export function loadModel(url: string): Promise<HorseModel> {
  let p = loading.get(url);
  if (!p) {
    p = fetch(url)
      .then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.arrayBuffer(); })
      .then(parseHorse)
      .catch((e) => { loading.delete(url); throw e; });
    loading.set(url, p);
  }
  return p;
}

export const loadHorse = () => loadModel(HORSE_URL);

/** Where `t` seconds into a clip falls: the frame before, and how far to the next (a looping clip wraps, else holds). */
function frameAt(m: HorseModel, c: HorseClip, t: number): [number, number] {
  let f = t * m.fps;
  const last = c.frames - 1;
  if (c.loop) f = ((f % last) + last) % last;
  else f = Math.min(Math.max(f, 0), last);
  const i = Math.min(Math.floor(f), last - 1);
  return [c.first + i, f - i];
}

/**
 * A pose into out[o …]: clip `a` at `ta` seconds, cross-faded toward clip `b` at `tb` by `w` (0: all a).
 */
export function poseHorse(m: HorseModel, a: HorseClip, ta: number, b: HorseClip | null, tb: number, w: number, out: Float32Array, o = 0): void {
  const F = m.frames, J = m.joints * MAT;
  const [fa, ua] = frameAt(m, a, ta);
  const a0 = fa * J, a1 = a0 + J;
  if (!b || w <= 0) {
    for (let k = 0; k < J; k++) out[o + k] = F[a0 + k] + (F[a1 + k] - F[a0 + k]) * ua;
    return;
  }
  const [fb, ub] = frameAt(m, b, tb);
  const b0 = fb * J, b1 = b0 + J;
  for (let k = 0; k < J; k++) {
    const va = F[a0 + k] + (F[a1 + k] - F[a0 + k]) * ua;
    const vb = F[b0 + k] + (F[b1 + k] - F[b0 + k]) * ub;
    out[o + k] = va + (vb - va) * w;
  }
}

/** Joint `root` and every joint under it, parents first. */
export function subtree(m: HorseModel, root: number): number[] {
  const out = [root];
  for (let k = 0; k < out.length; k++) for (let j = 0; j < m.joints; j++) if (m.parents[j] === out[k]) out.push(j);
  return out;
}

const PIV = new Float64Array(3);

/**
 * The joints `js` (a subtree, its root first) of the pose at pose[o …] turned about where their root is in that
 * pose: by `yaw` (positive: toward the horse's right) then `pitch` (positive: nose down), in the horse's frame.
 */
export function turnJoints(m: HorseModel, pose: Float32Array, o: number, js: readonly number[], yaw: number, pitch: number): void {
  const r = m.rest[js[0]];
  jointPoint(pose, o, js[0], r[0], r[1], r[2], PIV);
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // R = Rx(pitch) · Ry(yaw), rows
  const r00 = cy, r01 = 0, r02 = sy;
  const r10 = sp * sy, r11 = cp, r12 = -sp * cy;
  const r20 = -cp * sy, r21 = sp, r22 = cp * cy;
  for (const j of js) {
    const q = o + j * MAT;
    for (let c = 0; c < 4; c++) {
      const piv = c === 3;
      const a0 = pose[q + c] - (piv ? PIV[0] : 0), a1 = pose[q + 4 + c] - (piv ? PIV[1] : 0), a2 = pose[q + 8 + c] - (piv ? PIV[2] : 0);
      pose[q + c] = r00 * a0 + r01 * a1 + r02 * a2 + (piv ? PIV[0] : 0);
      pose[q + 4 + c] = r10 * a0 + r11 * a1 + r12 * a2 + (piv ? PIV[1] : 0);
      pose[q + 8 + c] = r20 * a0 + r21 * a1 + r22 * a2 + (piv ? PIV[2] : 0);
    }
  }
}

/** Point (x, y, z) through joint j's matrix of the pose at pose[o …], into out. */
export function jointPoint(pose: Float32Array, o: number, j: number, x: number, y: number, z: number, out: Float64Array): void {
  const q = o + j * MAT;
  out[0] = pose[q] * x + pose[q + 1] * y + pose[q + 2] * z + pose[q + 3];
  out[1] = pose[q + 4] * x + pose[q + 5] * y + pose[q + 6] * z + pose[q + 7];
  out[2] = pose[q + 8] * x + pose[q + 9] * y + pose[q + 10] * z + pose[q + 11];
}
