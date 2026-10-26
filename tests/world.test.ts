import { describe, expect, test } from 'bun:test';
import { FOE, SPAWN } from '../src/lib/constants';
import { campsNear } from '../src/lib/combat/foes';
import { plantInCell } from '../src/lib/forest/placement';
import { CELL } from '../src/lib/forest/species';
import { rockBodiesNear } from '../src/lib/rocks/field';
import { ruinSitesNear } from '../src/lib/ruins';
import { archipelagoInCell, type Archipelago } from '../src/lib/skyworld/archipelago';
import { islandAt, skyGround } from '../src/lib/skyworld/realm';
import { worldOf, worldSeed } from '../src/lib/world';

const L = 1337, C = worldSeed(1337, 'clouds');

/** The first n archipelagos round the origin under seed s, with their cells. */
function hubs(s: number, n: number): Array<{ ix: number; iz: number; a: Archipelago }> {
  const out: Array<{ ix: number; iz: number; a: Archipelago }> = [];
  for (let iz = -3; iz < 3 && out.length < n; iz++) for (let ix = -3; ix < 3 && out.length < n; ix++) {
    const a = archipelagoInCell(ix, iz, s);
    if (a) out.push({ ix, iz, a });
  }
  return out;
}

describe('worlds', () => {
  test('a seed says which world it belongs to', () => {
    expect(worldOf(L)).toBe('land');
    expect(worldOf(2 ** 32 - 1)).toBe('land');
    expect(worldOf(C)).toBe('clouds');
    expect(worldSeed(L, 'land')).toBe(L);
  });

  test('islands do not depend on the tag: seed + 2^32 and seed + 2^33 lay out the same Eyries', () => {
    const a = hubs(C, 3);
    expect(a.length).toBe(3);
    for (const { ix, iz, a: x } of a) {
      const y = archipelagoInCell(ix, iz, L + 2 ** 33)!;
      const shape = (r: Archipelago) => r.islands.map((i) => [i.cx, i.cz, i.R, i.y, i.seed | 0]);
      expect(shape(y)).toEqual(shape(x));
    }
  });

  test('the islands float over the land as ever (and over the inactive cloud world alike)', () => {
    const { a } = hubs(L, 1)[0];
    const hub = a.islands[0];
    expect(islandAt(hub.cx, hub.cz, L)).not.toBeNull();
    expect(Number.isNaN(skyGround(hub.cx, 3000, hub.cz, L))).toBe(false);
    expect(hubs(C, 1)[0].a.islands[0].cx).toBe(hub.cx);
  });

  test('the clouds have no trees, rocks, camps or ground ruins', () => {
    const cx = Math.floor(SPAWN.x / CELL), cz = Math.floor(SPAWN.z / CELL);
    for (let iz = -20; iz < 20; iz++) for (let ix = -20; ix < 20; ix++) expect(plantInCell(cx + ix, cz + iz, C)).toBeNull();
    const rocks = (seed: number) => {
      let n = 0;
      for (let k = 0; k < 64; k++) n += rockBodiesNear(SPAWN.x + 30 * k * Math.cos(k * 0.39), SPAWN.z + 30 * k * Math.sin(k * 0.39), seed, 0, []).length;
      return n;
    };
    expect(rocks(C)).toBe(0);
    expect(campsNear(SPAWN.x, SPAWN.z, 4 * FOE.campCell, C, []).length).toBe(0);
    const { a } = hubs(C, 1)[0];
    const sites = ruinSitesNear(a.cx, a.cz, 6000, C, 999);
    for (const s of sites) expect(islandAt(s.x, s.z, C)).not.toBeNull();
    // and the land still has them
    expect(rocks(L)).toBeGreaterThan(0);
    expect(ruinSitesNear(SPAWN.x, SPAWN.z, 6000, L, 999).length).toBeGreaterThan(0);
  });
});
