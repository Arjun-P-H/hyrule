import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'fs';
import { COLOSSUS } from '../src/lib/constants';
import { Colossi, COLOSSUS_JOINTS, COLOSSUS_STRIDE, createColossusFrame, EYE_HIT, ik, spotOf } from '../src/lib/colossus/colossi';
import { HORSE_VERTEX_BYTES, MAT, parseHorse } from '../src/lib/horse/model';
import { surfaceAt } from '../src/lib/terrainMath';
import { createHeroState } from '../src/lib/hero/locomotion';
import { createBodyHit } from '../src/lib/colossus/body';

const file = readFileSync(new URL('../public/models/colossus.bin', import.meta.url));
const model = parseHorse(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const SEED = 1337;
const ground = (x: number, z: number) => surfaceAt(x, z, SEED, [0, 0])[0];

/** The first colossus spot found near the origin for SEED. */
function aSpot() {
  for (let cz = 0; cz < 20; cz++) for (let cx = 0; cx < 20; cx++) {
    const s = spotOf(cx, cz, SEED);
    if (s) return s;
  }
  throw new Error('no spot');
}

function world(): Colossi {
  const W = new Colossi();
  W.setModel(model);
  W.reset(SEED);
  return W;
}

/** Joint j's rest place through the packed (camera at 0) skinning matrix of colossus k: where it is in the world. */
function jointAt(f: ReturnType<typeof createColossusFrame>, k: number, j: number): [number, number, number] {
  const r = model.rest[j], o = k * COLOSSUS_STRIDE + 4 + j * MAT, D = f.data;
  return [0, 1, 2].map((i) => D[o + 4 * i] * r[0] + D[o + 4 * i + 1] * r[1] + D[o + 4 * i + 2] * r[2] + D[o + 4 * i + 3]) as [number, number, number];
}

describe('colossus model (public/models/colossus.bin)', () => {
  test('a decimated mesh on a 17-joint skeleton, 30 m tall', () => {
    expect(model.joints).toBe(COLOSSUS_JOINTS);
    const n = model.vertices.byteLength / HORSE_VERTEX_BYTES;
    expect(n).toBeGreaterThan(20000);
    expect(model.indices.length / 3).toBeLessThanOrEqual(60000);
    expect(model.bounds[1][1]).toBeCloseTo(30, 0);
    for (const t of ['hand_l', 'hand_r', 'toe_l', 'toe_r', 'crown']) expect(model.tips?.[t]).toHaveLength(3);
    for (let v = 0; v < n; v += 97) {
      const j = model.vertices.subarray(v * HORSE_VERTEX_BYTES + 32, v * HORSE_VERTEX_BYTES + 36);
      for (const k of j) expect(k).toBeLessThan(COLOSSUS_JOINTS);
    }
  });
});

describe('colossi', () => {
  test('two-bone IK keeps the bones their lengths and reaches what it can', () => {
    const out = new Float64Array(3);
    for (const [t, ok] of [[[3, -8, 2], true], [[0, -30, 0], false], [[5, 2, 6], true]] as const) {
      ik([0, 0, 0], t, 6, 5, [0, 0, 1], out);
      expect(Math.hypot(out[0], out[1], out[2])).toBeCloseTo(6, 5);
      const end = Math.hypot(t[0] - out[0], t[1] - out[1], t[2] - out[2]);
      if (ok) expect(end).toBeCloseTo(5, 5);
      else expect(end).toBeGreaterThan(5); // (out of reach: straight at it)
      expect(out[2]).toBeGreaterThanOrEqual(-1e-9); // (bent toward the pole)
    }
  });

  test('spots are the same for a seed, and most cells have one', () => {
    let n = 0;
    for (let c = 0; c < 40; c++) {
      const a = spotOf(c, -c, SEED), b = spotOf(c, -c, SEED);
      expect(a).toEqual(b);
      if (a) n++;
    }
    expect(n).toBeGreaterThan(25); // (common, in every biome)
  });

  test('walks to the hero on its feet: a planted foot holds its place and its ankle is over it', () => {
    const W = world(), s = aSpot();
    const C = W.spawn(s.id, s.x, s.z, 0);
    const hx = s.x, hz = s.z + 70, hy = ground(hx, hz);
    const f = createColossusFrame();
    let steps = 0;
    const d0 = Math.hypot(hx - C.x, hz - C.z);
    const R = model.rest, AH = R[9][1];
    for (let k = 0; k < 60 * 25; k++) {
      const was = C.feet.map((F) => [F.x, F.z, F.u]);
      W.step(1 / 60, hx, hy, hz, false);
      steps += W.quakes.length;
      C.feet.forEach((F, i) => { if (was[i][2] < 0 && F.u < 0) { expect(F.x).toBe(was[i][0]); expect(F.z).toBe(was[i][1]); } });
      if (k % 60 === 0) {
        W.pack(0, 0, 0, f);
        for (const [i, j] of [[0, 9], [1, 16]]) {
          const F = C.feet[i];
          if (F.u >= 0) continue;
          const a = jointAt(f, W.list.indexOf(C), j); // (pack drew them nearest first: others may be about now)
          expect(Math.hypot(a[0] - F.x, a[1] - (F.y + AH), a[2] - F.z)).toBeLessThan(0.6);
        }
      }
      if (C.mind !== 'hunt') break;
    }
    expect(C.mind).not.toBe('idle');
    expect(steps).toBeGreaterThan(2);
    expect(Math.hypot(hx - C.x, hz - C.z)).toBeLessThan(d0 - 15);
    for (const v of f.data.subarray(0, COLOSSUS_STRIDE)) expect(Number.isFinite(v)).toBe(true);
  });

  test('a hero before it, away from its feet, is smashed or fallen on, never stomped; and is struck', () => {
    const W = world(), s = aSpot();
    const C = W.spawn(s.id, s.x, s.z, 0);
    const hx = s.x + 2, hz = s.z + 15, hy = ground(hx, hz);
    const minds = new Set<string>();
    let hits = 0;
    for (let k = 0; k < 60 * 60; k++) {
      W.step(1 / 60, hx, hy, hz, false);
      minds.add(C.mind);
      if (W.hit) { hits++; expect(W.hit.q).toBeGreaterThan(0); expect(W.hit.vy).toBeGreaterThan(0); }
    }
    expect(hits).toBeGreaterThan(1);
    expect(minds.has('smash') || minds.has('fall')).toBe(true);
    expect(minds.has('stomp')).toBe(false);
  });

  test('a hero at its foot is stomped', () => {
    const W = world(), s = aSpot();
    const C = W.spawn(s.id, s.x, s.z, 0);
    W.step(1 / 60, s.x, ground(s.x, s.z), s.z + 60, false);
    const F = C.feet[1];
    const hx = F.x + 3, hz = F.z + 3, hy = ground(hx, hz);
    let stomped = false;
    for (let k = 0; k < 60 * 10 && !stomped; k++) { W.step(1 / 60, hx, hy, hz, false); stomped = C.mind === 'stomp'; }
    expect(stomped).toBe(true);
  });

  test('its body is hurt, its eye more, its eye felled more still; slain, it vanishes for good', () => {
    const W = world(), s = aSpot();
    const C = W.spawn(s.id, s.x, s.z, 0);
    C.hitCap = 0;
    expect(W.hurt(C, 100)).toBeCloseTo(100 * COLOSSUS.body);
    expect(C.hp).toBeCloseTo(COLOSSUS.hp - 100 * COLOSSUS.body);
    C.hitCap = EYE_HIT;
    expect(W.hurt(C, 100)).toBeCloseTo(100 * COLOSSUS.eye);
    expect(C.mind).toBe('hunt');
    C.mind = 'down';
    const h = C.hp;
    W.hurt(C, 100);
    expect(C.hp).toBeCloseTo(h - 100 * COLOSSUS.eye * COLOSSUS.downed);
    W.hurt(C, 1e6);
    expect(C.mind as string).toBe('die');
    let gone = false;
    for (let k = 0; k < 60 * 15 && !gone; k++) { W.step(1 / 60, s.x + 500, 0, s.z, false); gone = W.vanished.length > 0; }
    expect(gone).toBe(true);
    expect(W.list.some((c) => c.id === s.id)).toBe(false);
    // (its spot never brings it back)
    for (let k = 0; k < 10; k++) W.step(0.5, s.x, ground(s.x, s.z), s.z, false);
    expect(W.list.some((c) => c.id === s.id)).toBe(false);
  });

  test('climbing it: grabbed pushing into a leg, up its stone riding its moves, shaken, thrown off when out of stamina', () => {
    const W = world(), s = aSpot();
    W.climbWalls = true; // (its walls climbed: off in the game for now)
    const C = W.spawn(s.id, s.x, s.z, 0);
    // (the hero ignored: it does not stomp it as it walks up)
    const F = C.feet[0];
    // the hero beside its left shin, facing it, pushing into it
    const hero = createHeroState(F.x - 4.2, F.z - 1, Math.PI / 2, SEED);
    hero.y = ground(hero.x, hero.z) + 0.08;
    hero.move = 'ground';
    let held = false;
    for (let k = 0; k < 60 && !held; k++) {
      W.step(1 / 60, hero.x, hero.y, hero.z, true);
      held = W.grab(hero, 1, 0, 1 / 60);
      if (!held) hero.x += 0.08;
    }
    expect(held).toBe(true);
    expect(hero.clinging).toBe(true);
    const y0 = hero.y;
    hero.stamina = 1;
    let shook = false, thrown = false, maxY = y0;
    for (let k = 0; k < 60 * 60 && !thrown; k++) {
      W.step(1 / 60, hero.x, hero.y, hero.z, true);
      // (up toward its body for 14 s, then hanging on through its shakes; standing on a ledge would rest)
      const up = k < 60 * 14;
      W.climb(hero, up ? 1 : 0, 0, Math.atan2(C.x - hero.x, C.z - hero.z), false, false, 12, 1 / 60);
      if (W.cling?.top) hero.stamina = Math.min(hero.stamina, 0.6); // (no resting: this is about the shakes)
      maxY = Math.max(maxY, hero.y);
      shook ||= C.shake > 0;
      // (never floating: its hands on the stone, or its feet standing on it)
      if (W.cling) {
        const top = W.cling.top, hand = createBodyHit();
        expect(C.body!.nearest(hero.x, top ? hero.y : hero.y + 0.9, hero.z, 1.0, hand)).toBe(true);
      }
      thrown = W.thrown !== null;
      expect(Number.isFinite(hero.x + hero.y + hero.z)).toBe(true);
    }
    expect(maxY).toBeGreaterThan(y0 + 6);   // it climbed
    expect(shook).toBe(true);               // and was shaken
    expect(C.mind === 'smash' || C.mind === 'stomp' || C.mind === 'fall').toBe(false);
    expect(thrown).toBe(true);              // and thrown off at last
    expect(hero.clinging).toBe(false);
    expect(W.cling).toBe(null);
  });

  test('holding on, Jump lets go; a stab far from the eye only clinks', () => {
    const W = world(), s = aSpot();
    W.climbWalls = true; // (its walls climbed: off in the game for now)
    const C = W.spawn(s.id, s.x, s.z, 0);
    const F = C.feet[1];
    const hero = createHeroState(F.x + 5, F.z - 1, -Math.PI / 2, SEED);
    hero.y = ground(hero.x, hero.z) + 3;
    hero.move = 'air';
    let held = false;
    for (let k = 0; k < 80 && !held; k++) { W.step(1 / 60, hero.x, hero.y, hero.z, true); held = W.grab(hero, 0, 0, 1 / 60); hero.x -= 0.06; }
    expect(held).toBe(true);
    W.climb(hero, 0, 0, 0, false, true, 12, 1 / 60);
    expect(W.stab?.took).toBe(0);
    expect(C.hp).toBe(COLOSSUS.hp);
    W.climb(hero, 0, 0, 0, true, false, 12, 1 / 60);
    expect(hero.clinging).toBe(false);
    expect(hero.move).toBe('air');
  });

  test('clinging beside its face, a stab goes into the eye', () => {
    const W = world(), s = aSpot();
    W.climbWalls = true; // (its walls climbed: off in the game for now)
    const C = W.spawn(s.id, s.x, s.z, 0);
    W.step(1 / 60, s.x, 0, s.z, true);
    // (in the air just before its eye)
    const e = C.eye;
    const hero = createHeroState(e[0], e[2] + 1.6, Math.PI, SEED);
    hero.y = e[1] - 1.2;
    hero.move = 'air';
    let held = false;
    for (let k = 0; k < 80 && !held; k++) { W.step(1 / 60, hero.x, hero.y, hero.z, true); held = W.grab(hero, 0, 0, 1 / 60); hero.z -= 0.05; }
    expect(held).toBe(true);
    W.climb(hero, 0, 0, 0, false, true, 12, 1 / 60);
    expect(W.stab?.took).toBeCloseTo(12 * COLOSSUS.stab * COLOSSUS.eye);
    expect(C.hp).toBeCloseTo(COLOSSUS.hp - 12 * COLOSSUS.stab * COLOSSUS.eye);
  });

  test('climbing off: walking into it never climbs it; landing on top, the hero stands on it as it moves, and falls off its edge', () => {
    const W = world(), s = aSpot();
    const C = W.spawn(s.id, s.x, s.z, 0);
    W.step(1 / 60, s.x, 0, s.z, true);
    // pushing into its shin: no hold
    const F = C.feet[0];
    const walker = createHeroState(F.x - 2.5, F.z, Math.PI / 2, SEED);
    walker.y = ground(walker.x, walker.z) + 0.08; walker.move = 'ground';
    for (let k = 0; k < 60; k++) { W.step(1 / 60, walker.x, walker.y, walker.z, true); expect(W.grab(walker, 1, 0, 1 / 60)).toBe(false); walker.x += 0.05; }
    // dropped onto its shoulder from above: it lands and stands
    const hand = createBodyHit();
    let held = false;
    const hero = createHeroState(C.x - 6, C.z, 0, SEED);
    expect(C.body!.ray(hero.x, C.y + 40, hero.z, hero.x, C.y, hero.z, hand)).toBeGreaterThan(0);
    hero.y = hand.y + 3; hero.move = 'air'; hero.vy = -2;
    for (let k = 0; k < 120 && !held; k++) {
      W.step(1 / 60, hero.x, hero.y, hero.z, true);
      held = W.grab(hero, 0, 0, 1 / 60);
      if (!held) { hero.vy -= 18 / 60; hero.y += hero.vy / 60; }
    }
    expect(held).toBe(true);
    expect(W.cling!.top).toBe(true);
    expect(hero.move as string).toBe('ground');
    // it walks; the hero stays standing on its stone
    for (let k = 0; k < 60 * 4; k++) {
      W.step(1 / 60, s.x + 100, 0, s.z, false);
      W.climb(hero, 0, 0, 0, false, false, 12, 1 / 60);
      if (!W.cling) break;
      expect(C.body!.nearest(hero.x, hero.y, hero.z, 0.6, hand)).toBe(true);
    }
    // walking off it: it falls
    let off = false;
    for (let k = 0; k < 60 * 20 && !off; k++) {
      W.step(1 / 60, s.x, 0, s.z, true);
      W.climb(hero, 1, 0, -Math.PI / 2, false, false, 12, 1 / 60);
      off = !W.cling;
    }
    expect(off).toBe(true);
    expect(hero.move).toBe('air');
  });

  test('multiplayer: the thinker\'s state drives the other player\'s puppet; the puppet\'s blow on the eye counts there', () => {
    const A = world(), B = world(), s = aSpot();
    B.authority = false;
    const C = A.spawn(s.id, s.x, s.z, 0);
    const hx = s.x, hz = s.z + 40, hy = ground(hx, hz);
    for (let k = 0; k < 60 * 6; k++) {
      A.step(1 / 60, hx, hy, hz, false);
      B.step(1 / 60, hx, hy, hz, false);
      if (k % 6 === 0) B.receive(A.encode(C));
    }
    const P = B.list.find((c) => c.id === s.id)!;
    expect(P).toBeDefined();
    expect(P.puppet).toBe(true);
    expect(P.mind).toBe(C.mind);
    expect(Math.hypot(P.x - C.x, P.z - C.z)).toBeLessThan(1);
    P.hitCap = EYE_HIT;
    const took = B.hurt(P, 50);
    expect(took).toBeGreaterThan(0);
    expect(B.sent).toHaveLength(1);
    const [kind, id, amount] = B.sent[0];
    expect(kind).toBe(5);
    A.struckBy(id, amount);
    expect(C.hp).toBeCloseTo(COLOSSUS.hp - took);
    // (a puppet never comes into being by itself)
    const B2 = world();
    B2.authority = false;
    B2.step(1 / 60, s.x, ground(s.x, s.z), s.z, false);
    expect(B2.list).toHaveLength(0);
  });

  test('comes into being near the hero at its spot, the hero is pushed out of its legs', () => {
    const W = world(), s = aSpot();
    W.step(1 / 60, s.x + 200, ground(s.x + 200, s.z), s.z, false);
    const C = W.list.find((c) => Math.hypot(c.x - s.x, c.z - s.z) < 1);
    expect(C).toBeDefined();
    const F = C!.feet[0];
    const hero = { x: F.x + 0.3, y: F.y, z: F.z };
    const d0 = Math.hypot(hero.x - F.x, hero.z - F.z);
    expect(W.collideHero(hero)).toBe(true);
    for (let k = 0; k < 5; k++) W.collideHero(hero);
    expect(W.collideHero(hero)).toBe(false); // (out of the stone now)
    expect(Math.hypot(hero.x - F.x, hero.z - F.z)).toBeGreaterThan(d0 + 0.8);
  });
});
