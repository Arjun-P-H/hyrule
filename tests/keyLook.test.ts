import { describe, it, expect } from 'bun:test';
import { KEY_LOOK, createKeyLook, stepKeyLook } from '../src/lib/engine/keyLook';
import { HELD } from '../src/lib/engine/keybinds';
import { CAMERA } from '../src/lib/constants';

const view = (yaw = 0, pitch = 0) => ({ yaw, pitch });

/** Hold the keys of `mask` for `seconds` at `hz` frames a second; the view and the look state carry on across calls. */
function hold(mask: number, seconds: number, opts: { hz?: number; sens?: number; v?: { yaw: number; pitch: number }; look?: ReturnType<typeof createKeyLook> } = {}) {
  const { hz = 60, sens = CAMERA.sensitivity, v = view(), look = createKeyLook() } = opts;
  const n = Math.round(seconds * hz);
  for (let i = 0; i < n; i++) stepKeyLook(look, mask, 1 / hz, sens, v);
  return { v, look };
}

/** The angle a key held for `t` seconds turns the view at the default Mouse Look: the speed ramp, integrated. */
function turned(t: number): number {
  const r = Math.min(t, KEY_LOOK.ramp);
  const ramped = KEY_LOOK.start * r + ((1 - KEY_LOOK.start) * r * r) / (2 * KEY_LOOK.ramp);
  return KEY_LOOK.rate * (ramped + Math.max(0, t - KEY_LOOK.ramp));
}

describe('Key Look: turning the camera with keys', () => {
  it('turns the view right and left, at the rate the ramp gives', () => {
    for (const t of [0.1, 0.4, 1, 2.5]) {
      expect(Math.abs(hold(HELD.lookRight, t).v.yaw - turned(t))).toBeLessThan(0.02);
      expect(Math.abs(hold(HELD.lookLeft, t).v.yaw + turned(t))).toBeLessThan(0.02);
    }
    // a second of it: about 90°, as a full turn (2π) takes under four seconds
    expect(turned(1)).toBeGreaterThan(1.4);
    expect(turned(1)).toBeLessThan(1.8);
  });

  it('tilts the view up and down, no further than the mouse can', () => {
    const up = hold(HELD.lookUp, 0.5).v;
    expect(up.pitch).toBeGreaterThan(0);
    expect(up.yaw).toBe(0);
    expect(hold(HELD.lookDown, 0.5).v.pitch).toBeLessThan(0);
    expect(hold(HELD.lookUp, 6).v.pitch).toBe(KEY_LOOK.pitchLimit);
    expect(hold(HELD.lookDown, 6).v.pitch).toBe(-KEY_LOOK.pitchLimit);
  });

  it('turns both ways at once for a diagonal, and two opposite keys cancel', () => {
    const d = hold(HELD.lookRight | HELD.lookUp, 0.5).v;
    expect(d.yaw).toBeGreaterThan(0);
    expect(d.pitch).toBeGreaterThan(0);
    expect(hold(HELD.lookLeft | HELD.lookRight, 1).v).toEqual({ yaw: 0, pitch: 0 });
    expect(hold(HELD.lookUp | HELD.lookDown, 1).v).toEqual({ yaw: 0, pitch: 0 });
  });

  it('follows the Mouse Look setting', () => {
    const low = hold(HELD.lookRight, 0.5, { sens: 0.0012 }).v.yaw;
    const med = hold(HELD.lookRight, 0.5, { sens: 0.0022 }).v.yaw;
    const high = hold(HELD.lookRight, 0.5, { sens: 0.0038 }).v.yaw;
    expect(low).toBeLessThan(med);
    expect(med).toBeLessThan(high);
    expect(high / low).toBeCloseTo(0.0038 / 0.0012, 6);
    expect(Math.abs(med - turned(0.5))).toBeLessThan(0.02); // (the default setting turns at the stated rate)
  });

  it('turns the same at any frame rate', () => {
    const at = (hz: number) => hold(HELD.lookRight, 1, { hz }).v.yaw;
    for (const hz of [30, 90, 144, 240]) expect(Math.abs(at(hz) - at(60))).toBeLessThan(0.04);
  });

  it('starts slowly, for fine aim, and reaches full speed', () => {
    const first = hold(HELD.lookRight, 0.1).v.yaw;
    expect(first).toBeLessThan(KEY_LOOK.rate * 0.1 * 0.6);
    const { v, look } = hold(HELD.lookRight, 1);
    const before = v.yaw;
    hold(HELD.lookRight, 0.1, { v, look });
    expect(v.yaw - before).toBeCloseTo(KEY_LOOK.rate * 0.1, 3);
  });

  it('starts slowly again after a key is let go, or the turn is reversed', () => {
    const dt = 1 / 60;
    const slow = KEY_LOOK.rate * KEY_LOOK.start * dt;
    const { v, look } = hold(HELD.lookRight, 1);
    // let go for one frame, then hold again
    stepKeyLook(look, 0, dt, CAMERA.sensitivity, v);
    let was = v.yaw;
    stepKeyLook(look, HELD.lookRight, dt, CAMERA.sensitivity, v);
    expect(v.yaw - was).toBeCloseTo(slow, 9);
    // a long turn to the right, then to the left without a pause
    const t = hold(HELD.lookRight, 1);
    was = t.v.yaw;
    stepKeyLook(t.look, HELD.lookLeft, dt, CAMERA.sensitivity, t.v);
    expect(t.v.yaw - was).toBeCloseTo(-slow, 9);
  });

  it('leaves the view alone when no look key is held, whatever else is', () => {
    expect(hold(0, 2, { v: view(1.25, -0.4) }).v).toEqual({ yaw: 1.25, pitch: -0.4 });
    expect(hold(HELD.forward | HELD.sprint | HELD.jump | HELD.attack, 2, { v: view(1.25, -0.4) }).v).toEqual({ yaw: 1.25, pitch: -0.4 });
  });

  it('turns without touching a tilt that is out of the mouse range, as another view sets it', () => {
    expect(hold(HELD.lookRight, 0.5, { v: view(0, 1.6) }).v.pitch).toBe(1.6);
    // and tilting brings it back within
    expect(hold(HELD.lookDown, 0.05, { v: view(0, 1.6) }).v.pitch).toBeLessThan(1.6);
    expect(hold(HELD.lookUp, 0.05, { v: view(0, 1.6) }).v.pitch).toBe(KEY_LOOK.pitchLimit);
  });
});
