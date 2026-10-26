import { describe, expect, test } from 'bun:test';
import { PORTAL } from '../src/lib/constants';
import { PortalTrip, TRIP_TIMEOUT } from '../src/lib/portalTrip';
import type { World } from '../src/lib/world';

/** A view whose other world's renderer is ready after `buildFrames` steps (Infinity: never; null world: WebGL2). */
function fakeView(start: World | null, buildFrames: number) {
  let drawn = start, wanted = start, frames = 0;
  return {
    world: () => drawn,
    setWorld(w: World) { if (drawn === null) return; wanted = w; frames = 0; },
    tick() { if (drawn !== null && wanted !== drawn && ++frames >= buildFrames) drawn = wanted; },
  };
}
const DT = 1 / 60;

describe('the trip through the portal', () => {
  test('fades out, arrives once the other world is drawn, fades back in', () => {
    const v = fakeView('land', 60), t = new PortalTrip();
    expect(t.step(DT, true, false, 'land', v)).toEqual({});
    let arrived: World | undefined, frames = 0;
    for (; frames < 600 && !arrived; frames++) { v.tick(); arrived = t.step(DT, false, false, 'land', v).arrive; }
    expect(arrived).toBe('clouds');
    expect(t.fade()).toBe(1);
    for (let k = 0; k < Math.ceil(PORTAL.fade / DT) + 1; k++) t.step(DT, false, false, 'clouds', v);
    expect(t.fade()).toBe(0);
    expect(t.busy()).toBe(false);
  });

  test('a swap before the fade-out ends arrives at once (never a frame of the new world at the old place)', () => {
    const v = fakeView('land', 1), t = new PortalTrip();
    t.step(DT, true, false, 'land', v);
    v.tick();
    expect(t.step(DT, false, false, 'land', v).arrive).toBe('clouds');
    expect(t.fade()).toBe(1);
  });

  test('a build that never comes gives up: back to the world it left, the view lit again', () => {
    const v = fakeView('land', Infinity), t = new PortalTrip();
    t.step(DT, true, false, 'land', v);
    let said: string | undefined;
    for (let k = 0; k < Math.ceil((TRIP_TIMEOUT + PORTAL.fade) / DT) + 10; k++) { v.tick(); const r = t.step(DT, false, false, 'land', v); said ??= r.say; }
    expect(said).toBeTruthy();
    expect(t.fade()).toBe(0);
    expect(t.busy()).toBe(false);
    expect(v.world()).toBe('land');
  });

  test('no crossing, no trip; just after one, the ring rests', () => {
    const v = fakeView('land', 1), t = new PortalTrip();
    expect(t.step(DT, false, true, 'land', v)).toEqual({});
    expect(t.busy()).toBe(false);
    t.step(DT, true, false, 'land', v); v.tick();
    t.step(DT, false, false, 'land', v); // arrived
    for (let k = 0; k < Math.ceil(PORTAL.fade / DT) + 1; k++) t.step(DT, false, false, 'clouds', v);
    expect(t.busy()).toBe(false);
    t.step(DT, true, false, 'clouds', v);
    expect(t.busy()).toBe(false); // (the cooldown)
  });

  test('on WebGL2 walking up to the ring says it needs WebGPU, once a cooldown', () => {
    const v = fakeView(null, 1), t = new PortalTrip();
    expect(t.step(DT, false, true, 'land', v).say).toContain('WebGPU');
    expect(t.step(DT, false, true, 'land', v).say).toBeUndefined();
    expect(t.busy()).toBe(false);
  });
});
