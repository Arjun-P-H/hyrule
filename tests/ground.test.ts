/**
 * The ground's textures and cover (gpu/groundTex.wgsl.ts, biomes.ts GROUND_USE): the constants every shader that reads
 * the layers relies on. A layer tiles at a world scale that divides the chunk, so chunk-local coordinates agree across a chunk
 * boundary; every layer has a tile and a relief; the cover's shares are shares.
 */

import { describe, expect, it } from 'bun:test';
import { BIOMES, GROUND_USE } from '../src/lib/biomes';
import { CHUNK_SIZE } from '../src/lib/constants';
import { GROUND_LAYERS, GROUND_MIPS, GROUND_RELIEF, GROUND_SIZE, GROUND_TILE } from '../src/lib/gpu/groundTex.wgsl';

describe('ground layers', () => {
  it('every layer has a tile (a power of two dividing the chunk) and a relief', () => {
    expect(new Set(GROUND_LAYERS).size).toBe(GROUND_LAYERS.length);
    for (const l of GROUND_LAYERS) {
      const tile = GROUND_TILE[l];
      expect(Number.isInteger(Math.log2(tile))).toBe(true);
      expect(CHUNK_SIZE % tile).toBe(0);
      expect(GROUND_RELIEF[l]).toBeGreaterThan(0);
      expect(GROUND_RELIEF[l]).toBeLessThan(tile / 4);
    }
  });

  it('the atlas is a power of two with a full mip chain', () => {
    expect(Number.isInteger(Math.log2(GROUND_SIZE))).toBe(true);
    expect(GROUND_MIPS).toBe(Math.log2(GROUND_SIZE) + 1);
  });
});

describe('ground use (cover, trails, bare patches, tall grass, rocks)', () => {
  it('every biome has all four shares, each in 0 … 1', () => {
    for (const b of BIOMES) {
      const u = GROUND_USE[b];
      for (const k of ['trail', 'bare', 'tall', 'rocks'] as const) {
        expect(u[k]).toBeGreaterThanOrEqual(0);
        expect(u[k]).toBeLessThanOrEqual(1);
      }
    }
  });

  it('roads cross the open country and not the jungle; rocks lie thickest in the mountains', () => {
    expect(GROUND_USE.plains.trail).toBeGreaterThan(GROUND_USE.jungle.trail);
    expect(GROUND_USE.savanna.trail).toBeGreaterThan(GROUND_USE.rainforest.trail);
    const rockiest = BIOMES.reduce((a, b) => (GROUND_USE[b].rocks > GROUND_USE[a].rocks ? b : a));
    expect(rockiest).toBe('mountains');
  });
});
