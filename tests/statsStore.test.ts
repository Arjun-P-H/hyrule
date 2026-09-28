import { describe, it, expect } from 'bun:test';
import { createStatsStore, selectMode, type EngineStats } from '../src/lib/engine/statsStore';

const snap = (mode: 'walk' | 'fly', fps: number): EngineStats => ({
  fps, x: 1, y: 2, z: 3, mode, grounded: mode === 'walk', originX: 0, originZ: 0, resPct: 50,
});

describe('§30 HUD telemetry store', () => {
  it('notifies subscribers on every publish and stops after unsubscribe', () => {
    const store = createStatsStore();
    let calls = 0;
    const off = store.subscribe(() => calls++);
    store.set(snap('fly', 30));
    store.set(snap('fly', 31));
    expect(calls).toBe(2);
    off();
    store.set(snap('walk', 32));
    expect(calls).toBe(2);
    expect(store.get()?.fps).toBe(32);
  });

  it('the mode slice is Object.is-stable across telemetry updates, so the page does not re-render', () => {
    const store = createStatsStore();
    expect(selectMode(store)).toBeNull();
    const seen: Array<string | null> = [];
    for (let i = 0; i < 50; i++) {
      store.set(snap(i < 25 ? 'fly' : 'walk', 20 + i));
      seen.push(selectMode(store));
    }
    let changes = 0;
    for (let i = 1; i < seen.length; i++) if (!Object.is(seen[i], seen[i - 1])) changes++;
    expect(changes).toBe(1); // 50 publishes, one real mode change
  });
});
