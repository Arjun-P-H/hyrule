/**
 * Camera Spline for Hyrule Engine Benchmarks
 * ------------------------------------------
 * Representative trajectory sampling:
 * 1. Plains / Vista Ridge (Spawn: x=2016, z=-2592)
 * 2. High Mountain Summit / Alpine (x=4000, z=-4000)
 * 3. Cellular River Valley & Water Shoreline (x=-2000, z=-1700)
 * 4. Zonai Anomaly Ruins & Terrace Spire (x=-3300, z=3900)
 * 5. Arid Desert Dunes (x=6500, z=6000)
 *
 * The Catmull-Rom interpolation of the keyframe heights alone passes up to
 * 400 m *below* the terrain between keyframes (28 of 60 samples in the
 * original harness), which renders a cheap "inside the ground" frame and
 * skews every percentile. `generateCameraPath` lifts each sample to at least
 * `clearance` metres above the highest terrain within ±`probe` metres, using
 * the CPU twin. The result is frozen to bench/baseline/camera_path.json so the
 * path stays fixed even if the math core changes.
 */

import { elevation } from '../src/lib/terrainMath';
import { DEFAULT_SEED } from '../src/lib/constants';

export interface SplineKeyframe {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  label: string;
}

export const CAMERA_SPLINE_KEYFRAMES: SplineKeyframe[] = [
  { x: 2016.0, y: 75.8, z: -2592.0, yaw: 2.056, pitch: 0.10, label: 'Plains Vista (Spawn)' },
  { x: 4000.0, y: 560.0, z: -4000.0, yaw: 1.200, pitch: -0.05, label: 'Alpine Mountain Peak' },
  { x: -2000.0, y: 25.0, z: -1700.0, yaw: 0.800, pitch: -0.10, label: 'River Valley Shoreline' },
  { x: -3300.0, y: 550.0, z: 3900.0, yaw: 2.800, pitch: -0.15, label: 'Zonai Anomaly Terraces' },
  { x: 6500.0, y: 420.0, z: 6000.0, yaw: 3.500, pitch: 0.00, label: 'Arid Sand Dunes' },
  { x: 2016.0, y: 75.8, z: -2592.0, yaw: 2.056, pitch: 0.10, label: 'Plains Vista (Return)' },
];

function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    (2 * p1) +
    (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

export function sampleSpline(progress: number): { x: number; y: number; z: number; yaw: number; pitch: number } {
  const n = CAMERA_SPLINE_KEYFRAMES.length - 1;
  const clamped = Math.max(0, Math.min(progress, 1.0));
  const segmentFloat = clamped * n;
  const seg = Math.min(Math.floor(segmentFloat), n - 1);
  const t = segmentFloat - seg;

  const i0 = Math.max(0, seg - 1);
  const i1 = seg;
  const i2 = Math.min(n, seg + 1);
  const i3 = Math.min(n, seg + 2);

  const k0 = CAMERA_SPLINE_KEYFRAMES[i0];
  const k1 = CAMERA_SPLINE_KEYFRAMES[i1];
  const k2 = CAMERA_SPLINE_KEYFRAMES[i2];
  const k3 = CAMERA_SPLINE_KEYFRAMES[i3];

  return {
    x: catmullRom(k0.x, k1.x, k2.x, k3.x, t),
    y: catmullRom(k0.y, k1.y, k2.y, k3.y, t),
    z: catmullRom(k0.z, k1.z, k2.z, k3.z, t),
    yaw: catmullRom(k0.yaw, k1.yaw, k2.yaw, k3.yaw, t),
    pitch: catmullRom(k0.pitch, k1.pitch, k2.pitch, k3.pitch, t),
  };
}

export type CameraFrame = {
  frame: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  /** Nearest keyframe label, for per-scene reporting. */
  scene: string;
  /** Packed hero uniforms (src/lib/hero/rig.ts HERO_UNIFORM), camera-relative to this frame. */
  hero?: number[];
};

export function generateCameraPath(totalFrames = 60, clearance = 20, probe = 12): CameraFrame[] {
  const n = CAMERA_SPLINE_KEYFRAMES.length - 1;
  const frames: CameraFrame[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const progress = i / (totalFrames - 1);
    const s = sampleSpline(progress);
    let ground = -Infinity;
    for (let dx = -probe; dx <= probe; dx += probe) {
      for (let dz = -probe; dz <= probe; dz += probe) {
        ground = Math.max(ground, elevation(s.x + dx, s.z + dz, DEFAULT_SEED, 0));
      }
    }
    const scene = CAMERA_SPLINE_KEYFRAMES[Math.min(n, Math.round(progress * n))].label;
    frames.push({ frame: i, ...s, y: Math.max(s.y, ground + clearance), scene });
  }
  return frames;
}
