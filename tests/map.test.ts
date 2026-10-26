import { describe, it, expect } from 'bun:test';
import { DEFAULT_SEED } from '../src/lib/constants';
import { createHeroState } from '../src/lib/hero/locomotion';
import {
  TEMP_MAX, TEMP_MIN, ambientTemp, createNoiseMeter, heroNoise, landingLoudness, noisePulse, stepNoise, tempFraction, tempWord, walkNoise,
} from '../src/lib/map/gauges';
import { MAP_PEERS, createMapHud } from '../src/lib/map/mapHud';
import { MAX_LEVEL, TILE_PX, contourInterval, levelFor, levelMpp, levelSpan, renderTile, tilesAround } from '../src/lib/map/tiles';

describe('the temperature gauge', () => {
  it('falls with height and is warmest at 15:00, coldest at 03:00', () => {
    const low = ambientTemp(500, 10, 500, DEFAULT_SEED, 12);
    const high = ambientTemp(500, 900, 500, DEFAULT_SEED, 12);
    expect(high).toBeLessThan(low - 5);
    const afternoon = ambientTemp(500, 10, 500, DEFAULT_SEED, 15);
    const night = ambientTemp(500, 10, 500, DEFAULT_SEED, 3);
    expect(afternoon).toBeGreaterThan(night + 4);
    expect(ambientTemp(500, 10, 500, DEFAULT_SEED, 27)).toBeCloseTo(ambientTemp(500, 10, 500, DEFAULT_SEED, 3), 9);
  });

  it('keeps its needle on the arc, and has a word for every temperature', () => {
    expect(tempFraction(TEMP_MIN - 50)).toBe(0);
    expect(tempFraction(TEMP_MAX + 50)).toBe(1);
    expect(tempFraction((TEMP_MIN + TEMP_MAX) / 2)).toBeCloseTo(0.5, 9);
    expect(tempWord(-10)).toBe('Freezing');
    expect(tempWord(18)).toBe('Mild');
    expect(tempWord(40)).toBe('Scorching');
  });
});

describe('the sound meter', () => {
  const hero = () => createHeroState(0, 0, 0, DEFAULT_SEED);
  const moving = (set: (h: ReturnType<typeof hero>) => void) => {
    const h = hero();
    h.move = 'ground';
    h.vx = 5;
    set(h);
    return heroNoise(h);
  };

  it('is silent standing still and in the air, quiet sneaking, and loudest sprinting', () => {
    expect(heroNoise(hero())).toBe(0);
    const sneak = moving((h) => { h.sneaking = true; });
    const run = moving(() => {});
    const sprint = moving((h) => { h.sprinting = true; });
    expect(sneak).toBeGreaterThan(0);
    expect(sneak).toBeLessThan(run);
    expect(run).toBeLessThan(sprint);
    expect(moving((h) => { h.move = 'glide'; })).toBeLessThan(sneak);
    expect(walkNoise(0, true)).toBe(0);
    expect(walkNoise(5, true)).toBeGreaterThan(walkNoise(5, false));
  });

  it('rises quickly, falls slowly, and a spike dies away', () => {
    const m = createNoiseMeter();
    let shown = 0;
    for (let k = 0; k < 6; k++) shown = stepNoise(m, 1 / 60, 0.9); // 100 ms of sprinting
    expect(shown).toBeGreaterThan(0.5);
    const before = shown;
    for (let k = 0; k < 6; k++) shown = stepNoise(m, 1 / 60, 0);   // 100 ms of silence: still ringing
    expect(shown).toBeGreaterThan(before * 0.4);
    for (let k = 0; k < 600; k++) shown = stepNoise(m, 1 / 60, 0); // ten seconds
    expect(shown).toBe(0);
    noisePulse(m, 0.8);
    expect(stepNoise(m, 1 / 60, 0)).toBeGreaterThan(0.7);
    for (let k = 0; k < 120; k++) shown = stepNoise(m, 1 / 60, 0);
    expect(shown).toBe(0);
  });

  it('hears a hard landing, not a step down', () => {
    expect(landingLoudness(3)).toBe(0);
    expect(landingLoudness(11)).toBeGreaterThan(0.3);
    expect(landingLoudness(60)).toBe(1);
  });
});

describe('the map store', () => {
  it('starts with room for every other player', () => {
    const m = createMapHud();
    expect(m.peers.length).toBe(MAP_PEERS);
    expect(m.peerCount).toBe(0);
    expect(m.live).toBe(false);
  });
});

describe('the map tiles', () => {
  it('double their metres to the pixel with each level, and the level suits the view', () => {
    expect(levelMpp(0)).toBe(2);
    expect(levelMpp(3)).toBe(16);
    expect(levelSpan(1)).toBe(TILE_PX * 4);
    expect(levelFor(4)).toBe(1);
    expect(levelFor(0.1)).toBe(0);
    expect(levelFor(1e6)).toBe(MAX_LEVEL);
    expect(contourInterval(1)).toBeLessThan(contourInterval(6));
  });

  it('are those a view touches', () => {
    const span = levelSpan(1);
    const t = tilesAround(1, 0, 0, 10);
    expect(t).toEqual({ tx0: -1, tx1: 0, tz0: -1, tz1: 0 });
    const u = tilesAround(1, span * 2.5, -span * 0.5, 1);
    expect(u).toEqual({ tx0: 2, tx1: 2, tz0: -1, tz1: -1 });
  });

  it('are painted the same every time, opaque, with every pixel a colour', () => {
    const a = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    const b = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    renderTile(DEFAULT_SEED, 1, 2, -1, a);
    renderTile(DEFAULT_SEED, 1, 2, -1, b);
    expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);
    for (let i = 3; i < a.length; i += 4) expect(a[i]).toBe(255);
    // not one flat colour
    const seen = new Set<number>();
    for (let i = 0; i < a.length; i += 4) seen.add((a[i] << 16) | (a[i + 1] << 8) | a[i + 2]);
    expect(seen.size).toBeGreaterThan(40);
  });

  it('show water as blue and the land under another seed differently', () => {
    // far out at the coarsest level there is sea somewhere: blue where the blue beats the red
    const px = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    let blue = 0;
    for (const [tx, tz] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [2, 2], [-2, -2]]) {
      renderTile(DEFAULT_SEED, MAX_LEVEL, tx, tz, px);
      for (let i = 0; i < px.length; i += 4) if (px[i + 2] > px[i] + 40 && px[i + 2] > px[i + 1]) blue++;
    }
    expect(blue).toBeGreaterThan(0);
    const other = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    const base = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    renderTile(DEFAULT_SEED, 3, 0, 0, base);
    renderTile(DEFAULT_SEED + 17, 3, 0, 0, other);
    expect(Buffer.from(base).equals(Buffer.from(other))).toBe(false);
  });

  it('meet across a tile edge (the same ground is shaded alike on both sides)', () => {
    const left = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    const right = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
    renderTile(DEFAULT_SEED, 2, 0, 0, left);
    renderTile(DEFAULT_SEED, 2, 1, 0, right);
    // the step across the seam is no bigger than the steps within a tile, on average
    let seam = 0, inner = 0;
    for (let j = 0; j < TILE_PX; j++) {
      const l = (j * TILE_PX + TILE_PX - 1) * 4, r = (j * TILE_PX) * 4;
      seam += Math.abs(left[l] - right[r]) + Math.abs(left[l + 1] - right[r + 1]) + Math.abs(left[l + 2] - right[r + 2]);
      const a = (j * TILE_PX + 60) * 4, b = a + 4;
      inner += Math.abs(left[a] - left[b]) + Math.abs(left[a + 1] - left[b + 1]) + Math.abs(left[a + 2] - left[b + 2]);
    }
    expect(seam).toBeLessThan(inner * 3 + 60);
  });
});
