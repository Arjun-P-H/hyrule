/**
 * The hero's ragdoll: thrown (a horse's blow, thrown off a horse), the body flies and tumbles limp, lands and
 * comes to rest; then the combat world hands it back to the rig, which gets up from the knockdown pose.
 * ---------------------------------------------------------------------------
 * Verlet: each of the rig's joints but the glider (rig.ts J) is a particle, world fp64; the bones keep their
 * lengths and a few braces keep the torso and the hips a body (LINKS, lengths taken from the pose it was thrown
 * from). The ground is the plane the caller samples under the pelvis each tick; a particle on it is held out of it
 * by its radius and rubs to a stop. At rest (every particle slower than REST_SPEED for REST_FOR s), or after
 * MAX_TIME s, `settled` is set.
 */

import { PLAYER } from '../constants';
import { J } from './rig';

/** Particles: the rig's joints 0..16 (the glider's handle, 17, is not part of the body). */
export const RAG_N = 17;

const LINKS: ReadonlyArray<readonly [number, number]> = [
  // bones
  [J.pelvis, J.chest], [J.chest, J.head],
  [J.chest, J.shoulderL], [J.shoulderL, J.elbowL], [J.elbowL, J.handL],
  [J.chest, J.shoulderR], [J.shoulderR, J.elbowR], [J.elbowR, J.handR],
  [J.pelvis, J.hipL], [J.hipL, J.kneeL], [J.kneeL, J.ankleL], [J.ankleL, J.toeL],
  [J.pelvis, J.hipR], [J.hipR, J.kneeR], [J.kneeR, J.ankleR], [J.ankleR, J.toeR],
  // braces: the shoulders and the hips held square, the neck on the shoulders, the feet on the shins
  [J.shoulderL, J.shoulderR], [J.hipL, J.hipR], [J.shoulderL, J.pelvis], [J.shoulderR, J.pelvis],
  [J.hipL, J.chest], [J.hipR, J.chest], [J.head, J.shoulderL], [J.head, J.shoulderR],
  [J.shoulderL, J.hipR], [J.shoulderR, J.hipL], [J.kneeL, J.toeL], [J.kneeR, J.toeR],
];
// how far each particle stands off the ground (m): the head and the body thick, the limbs thin
const RADIUS = [0.16, 0.16, 0.13, 0.08, 0.06, 0.05, 0.08, 0.06, 0.05, 0.09, 0.07, 0.06, 0.04, 0.09, 0.07, 0.06, 0.04];
// the upper body thrown a little harder than the legs: it tumbles head over heels
const KICK = [0, 0.35, 0.45, 0.35, 0.4, 0.45, 0.35, 0.4, 0.45, -0.1, -0.2, -0.25, -0.25, -0.1, -0.2, -0.25, -0.25];
const SUBSTEPS = 2;
const ITERATIONS = 8;
const DAMP = 0.998;      // air drag a substep
const FRICTION = 0.55;   // share of the sliding speed a particle on the ground loses a substep
const REST_SPEED = 0.45; // m/s
const REST_FOR = 0.35;
const MAX_TIME = 3.5;

export type GroundPlane = { x: number; z: number; y: number; gx: number; gz: number };

export class Ragdoll {
  /** Particle positions (world fp64), RAG_N × 3. */
  readonly p = new Float64Array(RAG_N * 3);
  private q = new Float64Array(RAG_N * 3);
  private len = new Float64Array(LINKS.length);
  time = 0;
  settled = false;
  private still = 0;

  /**
   * Thrown from the pose `local` (hero-local joints, rig.ts HeroRig.local) of a hero at (x, y, z) facing `yaw`
   * (y its root: PLAYER.groundBias over the feet), at (vx, vy, vz) m/s; `dt` the tick it starts on.
   */
  start(local: ArrayLike<number>, x: number, y: number, z: number, yaw: number, vx: number, vy: number, vz: number, dt: number): void {
    const sy = Math.sin(yaw), cy = Math.cos(yaw), by = y - PLAYER.groundBias;
    const p = this.p, q = this.q;
    for (let j = 0; j < RAG_N; j++) {
      const lx = local[3 * j], ly = local[3 * j + 1], lz = local[3 * j + 2];
      p[3 * j] = x + cy * lx + sy * lz;
      p[3 * j + 1] = by + ly;
      p[3 * j + 2] = z - sy * lx + cy * lz;
      const k = 1 + KICK[j];
      q[3 * j] = p[3 * j] - vx * k * dt;
      q[3 * j + 1] = p[3 * j + 1] - vy * dt;
      q[3 * j + 2] = p[3 * j + 2] - vz * k * dt;
    }
    LINKS.forEach(([a, b], i) => { this.len[i] = Math.hypot(p[3 * a] - p[3 * b], p[3 * a + 1] - p[3 * b + 1], p[3 * a + 2] - p[3 * b + 2]); });
    this.time = 0;
    this.settled = false;
    this.still = 0;
  }

  /** One tick on the ground plane `g`. */
  step(dt: number, g: GroundPlane): void {
    const p = this.p, q = this.q, h = dt / SUBSTEPS, gh = PLAYER.gravity * h * h;
    let fastest = 0;
    for (let s = 0; s < SUBSTEPS; s++) {
      for (let i = 0; i < RAG_N * 3; i += 3) {
        const vx = (p[i] - q[i]) * DAMP, vy = (p[i + 1] - q[i + 1]) * DAMP, vz = (p[i + 2] - q[i + 2]) * DAMP;
        q[i] = p[i]; q[i + 1] = p[i + 1]; q[i + 2] = p[i + 2];
        p[i] += vx; p[i + 1] += vy - gh; p[i + 2] += vz;
      }
      for (let it = 0; it < ITERATIONS; it++) {
        for (let k = 0; k < LINKS.length; k++) {
          const a = 3 * LINKS[k][0], b = 3 * LINKS[k][1];
          const dx = p[b] - p[a], dy = p[b + 1] - p[a + 1], dz = p[b + 2] - p[a + 2];
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-9;
          const c = 0.5 * (d - this.len[k]) / d;
          p[a] += dx * c; p[a + 1] += dy * c; p[a + 2] += dz * c;
          p[b] -= dx * c; p[b + 1] -= dy * c; p[b + 2] -= dz * c;
        }
        for (let j = 0; j < RAG_N; j++) {
          const i = 3 * j, floor = g.y + g.gx * (p[i] - g.x) + g.gz * (p[i + 2] - g.z) + RADIUS[j];
          if (p[i + 1] < floor) p[i + 1] = floor;
        }
      }
      // on the ground: rubbing to a stop (and no bounce)
      for (let j = 0; j < RAG_N; j++) {
        const i = 3 * j, floor = g.y + g.gx * (p[i] - g.x) + g.gz * (p[i + 2] - g.z) + RADIUS[j];
        if (p[i + 1] <= floor + 1e-3) {
          q[i] += (p[i] - q[i]) * FRICTION;
          q[i + 2] += (p[i + 2] - q[i + 2]) * FRICTION;
          if (q[i + 1] < p[i + 1]) q[i + 1] = p[i + 1];
        }
      }
    }
    for (let i = 0; i < RAG_N * 3; i += 3) fastest = Math.max(fastest, Math.hypot(p[i] - q[i], p[i + 1] - q[i + 1], p[i + 2] - q[i + 2]) / h);
    this.time += dt;
    this.still = fastest < REST_SPEED ? this.still + dt : 0;
    if (this.still > REST_FOR || this.time > MAX_TIME) this.settled = true;
  }

  /** Joint j's world position into out. */
  at(j: number, out: Float64Array): Float64Array {
    out[0] = this.p[3 * j]; out[1] = this.p[3 * j + 1]; out[2] = this.p[3 * j + 2];
    return out;
  }

  /** The pelvis' velocity (m/s, the last substep's). */
  velocity(dt: number, out: Float64Array): Float64Array {
    const i = 3 * J.pelvis, h = dt / SUBSTEPS;
    out[0] = (this.p[i] - this.q[i]) / h; out[1] = (this.p[i + 1] - this.q[i + 1]) / h; out[2] = (this.p[i + 2] - this.q[i + 2]) / h;
    return out;
  }
}
