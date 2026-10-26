/**
 * Scissor rectangles for the traced bodies (the hero, other players, foes):
 * the pixels a camera-relative sphere, then a box, can cover on screen.
 */

import { CAMERA } from '../constants';

/**
 * The part of framebuffer rectangle `r` ([x, y, w, h], updated in place) the
 * box with corners (x0, y0, z0)–(x1, y1, z1) (camera-relative) can cover, from
 * its eight corners projected (sx, sy as for sphereScissor) and a pixel of
 * margin for the TAA's jitter; `r` as it is when a corner reaches the near
 * plane, null when nothing of it is left.
 */
export function cornersScissor(
  F: Float32Array, x0b: number, y0b: number, z0b: number, x1b: number, y1b: number, z1b: number,
  sx: number, sy: number, w: number, h: number, r: [number, number, number, number],
): [number, number, number, number] | null {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let k = 0; k < 8; k++) {
    const cx = k & 1 ? x1b : x0b, cy = k & 2 ? y1b : y0b, cz = k & 4 ? z1b : z0b;
    const vz = cx * F[8] + cy * F[9] + cz * F[10];
    if (vz <= CAMERA.near) return r;
    const px = ((cx * F[0] + cy * F[1] + cz * F[2]) / vz) * sx, py = ((cx * F[4] + cy * F[5] + cz * F[6]) / vz) * sy;
    if (px < x0) x0 = px; if (px > x1) x1 = px;
    if (py < y0) y0 = py; if (py > y1) y1 = py;
  }
  const px0 = Math.max(r[0], Math.floor((x0 * 0.5 + 0.5) * w) - 1);
  const px1 = Math.min(r[0] + r[2], Math.ceil((x1 * 0.5 + 0.5) * w) + 1);
  const py0 = Math.max(r[1], Math.floor((0.5 - y1 * 0.5) * h) - 1);
  const py1 = Math.min(r[1] + r[3], Math.ceil((0.5 - y0 * 0.5) * h) + 1);
  if (px1 <= px0 || py1 <= py0) return null;
  r[0] = px0; r[1] = py0; r[2] = px1 - px0; r[3] = py1 - py0;
  return r;
}

/**
 * Framebuffer rectangle [x, y, w, h] covering a camera-relative sphere, from
 * the frame's basis (F[0..2] right, F[4..6] up, F[8..10] forward) and the
 * projection scales sx = f/aspect, sy = f; null when it is off screen, the
 * whole framebuffer when the sphere reaches the near plane.
 */
export function sphereScissor(
  F: Float32Array, cx: number, cy: number, cz: number, r: number, sx: number, sy: number, w: number, h: number,
): [number, number, number, number] | null {
  const vx = cx * F[0] + cy * F[1] + cz * F[2];
  const vy = cx * F[4] + cy * F[5] + cz * F[6];
  const vz = cx * F[8] + cy * F[9] + cz * F[10];
  if (vz + r <= CAMERA.near) return null;
  if (vz - r <= CAMERA.near) return [0, 0, w, h];
  // Extremes of X/Z over the view-space box around the sphere (conservative).
  const x0 = Math.min((vx - r) / (vz - r), (vx - r) / (vz + r)) * sx;
  const x1 = Math.max((vx + r) / (vz - r), (vx + r) / (vz + r)) * sx;
  const y0 = Math.min((vy - r) / (vz - r), (vy - r) / (vz + r)) * sy;
  const y1 = Math.max((vy + r) / (vz - r), (vy + r) / (vz + r)) * sy;
  const px0 = Math.max(0, Math.floor((x0 * 0.5 + 0.5) * w));
  const px1 = Math.min(w, Math.ceil((x1 * 0.5 + 0.5) * w));
  const py0 = Math.max(0, Math.floor((0.5 - y1 * 0.5) * h));
  const py1 = Math.min(h, Math.ceil((0.5 - y0 * 0.5) * h));
  if (px1 <= px0 || py1 <= py0) return null;
  return [px0, py0, px1 - px0, py1 - py0];
}
