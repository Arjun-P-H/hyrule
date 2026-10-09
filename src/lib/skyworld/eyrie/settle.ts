/**
 * The Eyries' grammars: what the Aelari build on each island of an Eyrie, by
 * its class and each island's role (SKY_CIVILIZATION.md §7, §8, §11).
 * ---------------------------------------------------------------------------
 * A planner per island (planner.ts) keeps what is taken and places buildings
 * only where they fit. Every built bridge (bridges.ts) is raised first; the
 * islands then build round their bridgeheads, so streets run to them; then
 * the life in its lake (lakes.ts), then its district map (district.ts). The
 * Crown's grammars are crown.ts, every other class's grammars.ts.
 */

import type { Archipelago } from '../archipelago';
import { hashU } from '../../terrainMath';
import { rng } from '../kit/mesh';
import type { Structure } from '../structures';
import { bridgeStructures, type BridgeStyle } from './bridges';
import type { IslandRole } from './layouts';
import { buildDistrict, districtAdded } from './district';
import { crownIsle, perchIsle, WARD_TASKS, wardIsle } from './crown';
import { Planner, type Footing, type Grammar } from './planner';
import { CLASS_GRAMMARS } from './grammars';
import { lakeLife } from './lakes';

/** Each island role's grammar: the Crown's (crown.ts), every other class's (grammars.ts). */
const GRAMMARS: Partial<Record<IslandRole, Grammar>> = { crown: crownIsle, perch: perchIsle, ...CLASS_GRAMMARS };

/** The bridge style of each class (span lengths, how deep the piers hang). */
function bridgeStyle(a: Archipelago): BridgeStyle {
  const pointed = a.cls === 'crown' || a.cls === 'halo' || a.cls === 'rime' || a.cls === 'beacon' || a.cls === 'fallen';
  return {
    palette: a.palette, pointed, span: a.cls === 'crown' ? [20, 30] : [16, 24], gate: a.cls === 'crown', hang: a.cls === 'crown' ? [0.8, 1.4] : [0.5, 1.0],
    timber: a.cls === 'petal' || a.cls === 'wells',
    aqueduct: a.cls === 'crown' || a.cls === 'mirror' || a.cls === 'harbour',
  };
}

/**
 * Everything the Aelari have built on an Eyrie: its bridges, then each island by its role — a
 * generator, yielding every so often (a slice is a few ms at most); it returns the structures.
 */
export function* settleEyrie(a: Archipelago, seed: number): Generator<void, Structure[], void> {
  const out: Structure[] = [];
  const rnd = rng(hashU(a.ix * 7919 + a.iz * 104729 + seed * 31 + 77));
  const style = bridgeStyle(a);
  const feet: Footing[][] = a.islands.map(() => []);
  for (const A of a.arches) {
    if (A.kind !== 'bridge') continue;
    out.push(...bridgeStructures(A, style, rnd));
    feet[A.from].push({ x: A.ax, z: A.az, out: Math.atan2(A.dirX, A.dirZ), w: A.wMid });
    feet[A.to].push({ x: A.bx, z: A.bz, out: Math.atan2(-A.dirX, -A.dirZ), w: A.wMid });
  }
  yield;
  let ward = 0;
  for (let i = 0; i < a.islands.length; i++) {
    const isl = a.islands[i], role = a.roles[i];
    if (role === 'debris' || isl.kind === 'debris') continue;
    const pl = new Planner(isl, rng(hashU(isl.seed ^ 0x5eed)), a.palette);
    const g = role === 'ward' ? wardIsle(WARD_TASKS[ward++ % WARD_TASKS.length]) : GRAMMARS[role];
    if (g) yield* g(a, i, pl, feet[i], out);
    yield;
    // Life in its lake: pads and lotus, reeds, waterweed, koi.
    yield* lakeLife(isl, pl.rnd, a.palette, out);
    // Its district map: where the grass and trees keep off, where the ground is paved.
    if (pl.paving.length || pl.built.length) {
      // (the cathedral's podium and the like are boxes in their structures: their discs are in `built`)
      isl.district = yield* buildDistrict(isl, pl.paving, pl.built);
      districtAdded();
    }
  }
  return out;
}
