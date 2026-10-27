import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'fs';
import { HORSE, PLAYER } from '../src/lib/constants';
import { createHeroState } from '../src/lib/hero/locomotion';
import { grassland, herdsNear, type WildHorse } from '../src/lib/horse/herds';
import { Horses, createHorseFrame, createHorseInput, HORSE_JOINTS, HORSE_STRIDE, MAX_DRAWN, type Horse } from '../src/lib/horse/horses';
import { HORSE_VERTEX_BYTES, MAT, jointPoint, parseHorse, poseHorse, subtree, turnJoints } from '../src/lib/horse/model';
import { trunksNear } from '../src/lib/forest/collision';
import { Ragdoll } from '../src/lib/hero/ragdoll';
import { createHeroRig, updateHeroRig, J } from '../src/lib/hero/rig';
import { CombatWorld, createCombatInput } from '../src/lib/combat/world';

const file = readFileSync(new URL('../public/models/horse.bin', import.meta.url));
const model = parseHorse(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const SEED = 1337;

/** The nearest real tree (a trunk, not a boulder) to (x, z), searched outward. */
function treeNear(x: number, z: number) {
  for (let r = 6; r < 1500; r += 12) for (let a = 0; a < 16; a++) {
    const t = trunksNear(x + Math.cos(a * 0.3927) * r, z + Math.sin(a * 0.3927) * r, SEED, [], false).find((q) => q.top > 2);
    if (t) return t;
  }
  return null;
}

/** A herd near the origin for SEED (the first found). */
function aHerd(): WildHorse[] {
  for (let r = 0; r < 20000; r += 400) {
    const L = herdsNear(r, r * 0.5, SEED, 600, []);
    if (L.length >= 2) return L.filter((w) => w.herd === L[0].herd);
  }
  throw new Error('no herd');
}

describe('horse model (public/models/horse.bin)', () => {
  test('mesh, joints and clips', () => {
    expect(model.joints).toBe(HORSE_JOINTS);
    expect(model.vertices.byteLength / HORSE_VERTEX_BYTES).toBeGreaterThan(5000);
    for (const c of ['Idle', 'Walk', 'Trot', 'Run', 'Eating', 'Sleep', 'lay_to_idle', 'Rear', 'Kick', 'Head_But']) expect(model.clip[c]).toBeDefined();
    // every joint index a vertex names is a joint
    let hair = 0;
    for (let v = 0; v < model.vertices.length; v += HORSE_VERTEX_BYTES) {
      for (let k = 0; k < 4; k++) expect(model.vertices[v + 32 + k]).toBeLessThan(HORSE_JOINTS);
      if (model.vertices[v + 40] > 127) hair++;
    }
    // the mane and the tail tagged: a good share of the mesh, not all of it
    expect(hair / (model.vertices.length / HORSE_VERTEX_BYTES)).toBeGreaterThan(0.08);
    expect(hair / (model.vertices.length / HORSE_VERTEX_BYTES)).toBeLessThan(0.5);
  });

  test("the bind pose is Idle's first frame: its matrices are the identity", () => {
    const P = new Float32Array(HORSE_JOINTS * MAT);
    poseHorse(model, model.clip.Idle, 0, null, 0, 0, P);
    for (let j = 0; j < HORSE_JOINTS; j++) {
      const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
      for (let k = 0; k < MAT; k++) expect(Math.abs(P[j * MAT + k] - I[k])).toBeLessThan(1e-4);
    }
  });

  test('stands on the ground, facing +Z, the saddle at a horse’s back', () => {
    const [lo, hi] = model.bounds;
    expect(Math.abs(lo[1])).toBeLessThan(0.05);         // hooves on y = 0
    expect(hi[1]).toBeGreaterThan(1.7);                  // ears ~1.9 m
    expect(hi[2]).toBeGreaterThan(-lo[2] - 0.3);         // the head reaches +Z (the tail hangs at −Z)
    expect(model.saddle[1]).toBeGreaterThan(1.25);
    expect(model.saddle[1]).toBeLessThan(1.5);
  });

  test('a gallop moves the legs, the saddle stays on the back', () => {
    const A = new Float32Array(HORSE_JOINTS * MAT), out = new Float64Array(3);
    let lo = Infinity, hi = -Infinity;
    for (let t = 0; t < model.clip.Run.duration; t += 0.02) {
      poseHorse(model, model.clip.Run, t, null, 0, 0, A);
      jointPoint(A, 0, model.saddleJoint, ...model.saddle, out);
      lo = Math.min(lo, out[1]); hi = Math.max(hi, out[1]);
    }
    expect(hi - lo).toBeGreaterThan(0.02);
    expect(hi - lo).toBeLessThan(0.5);
  });
});

describe('herds', () => {
  test('the same herds every time, only on dry grassland', () => {
    const a = herdsNear(0, 0, SEED, 3000, []).map((w) => ({ ...w }));
    const b = herdsNear(0, 0, SEED, 3000, []);
    expect(b).toEqual(a);
    for (const w of a) expect(grassland(w.ax, w.az, SEED)).toBeGreaterThan(0.05);
    expect(new Set(a.map((w) => w.id)).size).toBe(a.length);
  });
});

describe('a horse’s mind and riding', () => {
  function world() {
    const herd = aHerd();
    const H = new Horses();
    H.setModel(model);
    H.reset(SEED);
    const w = herd[0];
    const hero = createHeroState(w.x + 60, w.z, 0, SEED);
    const inp = createHorseInput();
    const tick = (n: number, dt = 1 / 60) => { for (let k = 0; k < n; k++) H.step(dt, hero, inp, 12); };
    return { H, hero, inp, tick, herd };
  }
  const near = (H: Horses, id: number) => H.list.find((h) => h.id === id)!;

  test('the herd comes into being round the hero, grazing', () => {
    const { H, tick, herd } = world();
    tick(1);
    for (const w of herd) expect(near(H, w.id)).toBeDefined();
  });

  test('running up: the herd bolts; sneaking up: it lets the hero mount', () => {
    const { H, hero, inp, tick, herd } = world();
    tick(1);
    const h0 = near(H, herd[0].id);
    // running past it within bolt range
    hero.x = h0.x + 6; hero.z = h0.z; hero.vx = -5;
    tick(2);
    expect(H.list.filter((h) => h.herd === h0.herd).every((h) => h.mind === 'flee')).toBe(true);

    const b = world();
    b.tick(1);
    const h1 = near(b.H, b.herd[0].id);
    b.hero.x = h1.x + 1.5; b.hero.z = h1.z; b.hero.y = h1.y + PLAYER.groundBias; b.hero.vx = 0; b.hero.sneaking = true;
    b.tick(1);
    expect(h1.mind).not.toBe('flee');
    b.inp.action = true; b.tick(1);
    expect(b.H.riding).toBe(h1);
    expect(b.hero.riding).toBe(true);
    expect(h1.mind).toBe('struggle');
    void inp;
  });

  test('taming: soothed, the horse is the hero’s; without stamina, the hero is thrown', () => {
    const mountWild = (stamina: number) => {
      const b = world();
      b.tick(1);
      const h = near(b.H, b.herd[0].id);
      h.temper = 1;
      b.hero.x = h.x + 1; b.hero.z = h.z; b.hero.y = h.y + PLAYER.groundBias; b.hero.sneaking = true; b.hero.stamina = stamina;
      b.inp.action = true; b.tick(1);
      return { ...b, h };
    };
    const a = mountWild(1);
    for (let k = 0; k < 400 && a.H.tameLeft === a.H.tameLeft; k++) { a.inp.action = k % 6 < 3; a.tick(1); }
    expect(a.h.owned).toBe(true);
    expect(a.H.riding).toBe(a.h);

    const b = mountWild(0.2);
    b.inp.action = false;
    b.tick(120);
    expect(b.H.riding).toBe(null);
    expect(b.hero.riding).toBe(false);
    expect(b.h.owned).toBe(false);
  });

  test('ridden: forward walks, each spur a gait up to the gallop, action dismounts', () => {
    const b = world();
    b.tick(1);
    const h: Horse = near(b.H, b.herd[0].id);
    h.owned = true; h.spurs = 4;
    b.hero.x = h.x + 1; b.hero.z = h.z; b.hero.y = h.y + PLAYER.groundBias;
    b.inp.action = true; b.tick(1); b.inp.action = false;
    expect(h.mind).toBe('ridden');
    b.inp.camYaw = h.yaw; b.inp.forward = 1;
    b.tick(120);
    b.H.seat(b.hero);
    expect(b.H.gait).toBe(1);
    if (!h.blocked) expect(h.speed).toBeCloseTo(HORSE.gaitSpeed[1], 0);
    for (let k = 0; k < 3; k++) { b.inp.sprint = true; b.tick(1); b.inp.sprint = false; b.tick(10); }
    expect(b.H.gait === 4 || h.blocked).toBe(true);
    expect(b.H.spurs).toBeLessThan(h.spurs);
    // the hero sits on the saddle, facing the horse's way
    b.H.seat(b.hero);
    expect(b.hero.yaw).toBeCloseTo(h.yaw, 5);
    expect(b.hero.y - PLAYER.groundBias - h.y).toBeGreaterThan(0.3);
    b.inp.forward = 0; b.inp.action = true; b.tick(1);
    expect(b.H.riding).toBe(null);
    expect(b.hero.riding).toBe(false);
  });

  test('a horse ridden at a tree stops at its trunk', () => {
    const b = world();
    b.tick(1);
    const h: Horse = near(b.H, b.herd[0].id);
    // the nearest real trunk (not a boulder) to the herd
    let t = null;
    t = treeNear(h.x, h.z);
    expect(t).not.toBe(null);
    h.owned = true; h.spurs = 4;
    h.x = t!.x - 5; h.z = t!.z; h.gx = NaN; h.yaw = Math.PI / 2; // facing +x, at the trunk
    b.hero.x = h.x + 1; b.hero.z = h.z; b.hero.y = h.y + PLAYER.groundBias;
    b.inp.action = true; b.tick(1); b.inp.action = false;
    b.inp.camYaw = Math.PI / 2; b.inp.forward = 1;
    b.inp.sprint = true; b.tick(1); b.inp.sprint = false;
    let closest = Infinity;
    for (let k = 0; k < 120; k++) {
      b.tick(1);
      for (const off of [0.85, 0, -0.85]) closest = Math.min(closest, Math.hypot(h.x + Math.sin(h.yaw) * off - t!.x, h.z + Math.cos(h.yaw) * off - t!.z) - t!.r);
    }
    expect(closest).toBeGreaterThan(0.3);
  });

  test('pack: the nearest horses, each its joints camera-relative', () => {
    const b = world();
    b.tick(1);
    const f = b.H.pack(b.hero.x, b.hero.y, b.hero.z, createHorseFrame());
    expect(f.count).toBeGreaterThan(0);
    const h = b.H.list[0];
    // a joint's translation lands near the horse (within its size)
    const o = 4 + 3;
    expect(Math.hypot(f.data[o] - (h.x - b.hero.x), f.data[o + 8] - (h.z - b.hero.z))).toBeLessThan(3);
    expect(f.data.length).toBe(MAX_DRAWN * HORSE_STRIDE);
  });
});

describe('thrown: the ragdoll, the horse’s body, getting up', () => {
  test('a body flung at 12 m/s flies, lands and comes to rest on the ground, its bones whole', () => {
    const rig = createHeroRig();
    const hero = createHeroState(0, 0, 0, SEED);
    updateHeroRig(rig, hero, 1 / 60);
    const R = new Ragdoll();
    R.start(rig.local, hero.x, hero.y, hero.z, hero.yaw, 0, 6, 12, 1 / 60);
    const ground = hero.y - PLAYER.groundBias;
    const plane = { x: hero.x, z: hero.z, y: ground, gx: 0, gz: 0 };
    const bone = () => Math.hypot(R.p[3 * J.kneeL] - R.p[3 * J.hipL], R.p[3 * J.kneeL + 1] - R.p[3 * J.hipL + 1], R.p[3 * J.kneeL + 2] - R.p[3 * J.hipL + 2]);
    const thigh = bone();
    let top = -Infinity;
    for (let k = 0; k < 300 && !R.settled; k++) { R.step(1 / 60, plane); top = Math.max(top, R.p[3 * J.pelvis + 1]); }
    expect(R.settled).toBe(true);
    expect(R.time).toBeLessThan(3.5);
    expect(top - ground).toBeGreaterThan(1.5);              // it flew
    expect(R.p[3 * J.pelvis + 2] - hero.z).toBeGreaterThan(3); // along the throw
    for (let j = 0; j < 17; j++) expect(R.p[3 * j + 1]).toBeGreaterThan(ground);
    expect(R.p[3 * J.head + 1] - ground).toBeLessThan(0.5);  // lying down
    expect(Math.abs(bone() - thigh) / thigh).toBeLessThan(0.08);
  });

  function herdWorld() {
    const herd = aHerd();
    const H = new Horses();
    H.setModel(model);
    H.reset(SEED);
    const w = herd[0];
    const hero = createHeroState(w.x + 60, w.z, 0, SEED);
    H.step(1 / 60, hero, createHorseInput(), 12);
    return { H, hero, h: H.list.find((x) => x.id === w.id)! };
  }

  test('the hero is pushed out of a standing horse, and flung by a galloping one', () => {
    const { H, hero, h } = herdWorld();
    const out = new Float64Array(3);
    h.speed = 0;
    hero.x = h.x + 0.2; hero.z = h.z; hero.y = h.y + PLAYER.groundBias; hero.move = 'ground';
    expect(H.collideHero(hero, out)).toBe(false);
    expect(Math.hypot(hero.x - h.x, hero.z - h.z)).toBeGreaterThan(0.7);
    h.speed = 11;
    hero.x = h.x + Math.sin(h.yaw) * 0.9; hero.z = h.z + Math.cos(h.yaw) * 0.9;
    expect(H.collideHero(hero, out)).toBe(true);
    expect(out[0] * Math.sin(h.yaw) + out[2] * Math.cos(h.yaw)).toBeGreaterThan(5); // along the horse's way
    expect(out[1]).toBeGreaterThan(3);
  });

  test('thrown in the combat world: limp, on the ground, then up and the hero’s own again', () => {
    const hero = createHeroState(0, 0, 0, SEED);
    const world = new CombatWorld(hero, SEED);
    const rig = createHeroRig();
    const inp = createCombatInput();
    updateHeroRig(rig, hero, 1 / 60, world.pose);
    world.throwHero(0, 5, 10, rig.local);
    expect(world.act).toBe('thrown');
    const acts = new Set<string>();
    for (let k = 0; k < 60 * 6 && world.act !== 'idle'; k++) {
      world.step(1 / 60, inp);
      updateHeroRig(rig, hero, 1 / 60, world.pose);
      acts.add(world.act);
      // the body is drawn where the ragdoll is
      if (world.act === 'thrown') expect(Math.abs(rig.local[3 * J.pelvis] ** 2 + rig.local[3 * J.pelvis + 2] ** 2)).toBeLessThan(0.01);
    }
    expect([...acts]).toContain('rising');
    expect(world.act).toBe('idle');
    expect(hero.move).toBe('ground');
    expect(hero.z).toBeGreaterThan(2);
    // standing again
    expect(rig.local[3 * J.head + 1]).toBeGreaterThan(1.3);
  });
});

describe('a horse at peace, and in the way of things', () => {
  test('the head turns on the neck: the head swings to the side, the neck’s base and the body stay', () => {
    const P = new Float32Array(HORSE_JOINTS * MAT), a = new Float64Array(3), b = new Float64Array(3);
    poseHorse(model, model.clip.Idle, 0, null, 0, 0, P);
    const neck = subtree(model, model.names.indexOf('spine_5')), head = model.names.indexOf('head'), hips = model.names.indexOf('hips');
    expect(neck).toContain(head);
    expect(neck).not.toContain(hips);
    const at = (j: number, out: Float64Array) => jointPoint(P, 0, j, ...model.rest[j], out);
    at(head, a);
    const base = new Float64Array(3); at(neck[0], base);
    turnJoints(model, P, 0, neck, 0.5, 0);
    at(head, b);
    expect(b[0] - a[0]).toBeGreaterThan(0.12);                // toward the right
    expect(Math.hypot(b[0] - base[0], b[2] - base[2])).toBeCloseTo(Math.hypot(a[0] - base[0], a[2] - base[2]), 4);
    at(neck[0], a);
    expect(Math.hypot(a[0] - base[0], a[1] - base[1], a[2] - base[2])).toBeLessThan(1e-4);
  });

  test('grazing: eating with the head moving, looking up and round, a few steps, and eating again', () => {
    const H = new Horses();
    H.setModel(model);
    H.reset(SEED);
    const w = aHerd()[0];
    const hero = createHeroState(w.x + 150, w.z, 0, SEED);
    const inp = createHorseInput();
    H.step(1 / 60, hero, inp, 12);
    const h = H.list.find((x) => x.id === w.id)!;
    const subs = new Set<string>(), clips = new Set<string>();
    let heads = 0;
    const x0 = h.x, z0 = h.z;
    for (let k = 0; k < 60 * 60; k++) {
      if (h.mind !== 'graze') { h.mind = 'graze'; h.timer = 99; }
      H.step(1 / 60, hero, inp, 12);
      subs.add(h.sub); clips.add(h.clip);
      heads = Math.max(heads, Math.abs(h.head));
    }
    expect([...subs].sort()).toEqual(['eat', 'look', 'step']);
    expect(clips.has('Eating') && clips.has('Idle')).toBe(true);
    expect(heads).toBeGreaterThan(0.15);
    expect(Math.hypot(h.x - x0, h.z - z0)).toBeGreaterThan(1); // it moved on to fresh grass
  });

  test('a horse walking into a tree turns away and takes another way, not pushing on into it', () => {
    const H = new Horses();
    H.setModel(model);
    H.reset(SEED);
    const w = aHerd()[0];
    const hero = createHeroState(w.x + 150, w.z, 0, SEED);
    const inp = createHorseInput();
    H.step(1 / 60, hero, inp, 12);
    const h = H.list.find((x) => x.id === w.id)!;
    let t = null;
    t = treeNear(w.x, w.z);
    h.x = t!.x - 4; h.z = t!.z + 0.1; h.gx = NaN; h.yaw = Math.PI / 2;
    h.mind = 'wander'; h.timer = 60; h.tx = t!.x + 8; h.tz = t!.z;
    hero.x = h.x + 150; hero.z = h.z;
    let detoured = false, far = 0;
    for (let k = 0; k < 60 * 25 && h.mind === 'wander'; k++) {
      H.step(1 / 60, hero, inp, 12);
      if (h.avoid > 0) detoured = true;
      far = Math.max(far, Math.hypot(h.x - t!.x, h.z - t!.z));
      expect(Math.hypot(h.x - t!.x, h.z - t!.z)).toBeGreaterThan(t!.r + 0.3);
    }
    expect(detoured).toBe(true);
    expect(far).toBeGreaterThan(8); // gone on its way, not stuck at the trunk
  });
});

describe('a horse’s health: struck, bolting, slain, vanishing', () => {
  test('the hero’s sword strikes a beast before it, once a swing', () => {
    const hero = createHeroState(0, 0, 0, SEED);
    const world = new CombatWorld(hero, SEED);
    const beast = { x: hero.x, y: hero.y - PLAYER.groundBias, z: hero.z + 1.7, yaw: Math.PI / 2, hp: 64, hitSwing: -1, rockHit: null };
    const blows: number[] = [];
    world.beasts = [beast];
    world.onBeast = (_b, dmg) => blows.push(dmg);
    const inp = createCombatInput();
    inp.camYaw = 0;
    for (let k = 0; k < 90; k++) { inp.attack = k < 3 || (k > 40 && k < 43); world.step(1 / 60, inp); }
    expect(blows.length).toBeGreaterThanOrEqual(1);
    expect(blows.length).toBeLessThanOrEqual(3);
    expect(blows[0]).toBeGreaterThan(0);
    // one behind the hero is not struck
    const back = { ...beast, z: hero.z - 1.7, hitSwing: -1 };
    world.beasts = [back];
    blows.length = 0;
    for (let k = 0; k < 40; k++) { inp.attack = k < 3; world.step(1 / 60, inp); }
    expect(blows.length).toBe(0);
  });

  test('hurt, it bolts with its herd; slain, it dies, vanishes, and does not come back', () => {
    const H = new Horses();
    H.setModel(model);
    H.reset(SEED);
    const herd = aHerd();
    const hero = createHeroState(herd[0].x + 60, herd[0].z, 0, SEED);
    const inp = createHorseInput();
    H.step(1 / 60, hero, inp, 12);
    const h = H.list.find((x) => x.id === herd[0].id)!;
    H.hurt(h, 10, hero.x, hero.z);
    expect(h.hp).toBe(HORSE.hp - 10);
    expect(h.mind).toBe('flee');
    expect(H.list.filter((x) => x.herd === h.herd).every((x) => x.mind === 'flee')).toBe(true);
    H.hurt(h, 1000, hero.x, hero.z);
    expect(h.hp).toBe(0);
    expect(h.mind).toBe('die');
    expect(h.clip).toBe('Death');
    let vanished = false, faded = 0;
    for (let k = 0; k < 60 * 4 && H.list.includes(h); k++) {
      H.step(1 / 60, hero, inp, 12);
      if (H.vanished.includes(h)) vanished = true;
      if (h.mind === 'vanish' && H.list.includes(h)) {
        const f = H.pack(h.x, h.y, h.z, createHorseFrame());
        expect(f.fading).toBe(1);
        const o = (f.count - 1) * HORSE_STRIDE;
        expect(f.u32[o]).toBe(h.id);           // the vanishing one drawn last
        faded = Math.max(faded, f.data[o + 2]);
      }
    }
    expect(vanished).toBe(true);
    expect(faded).toBeGreaterThan(0.8);
    expect(H.list.includes(h)).toBe(false);
    // the hero walks off and back: the herd is listed again, without it
    hero.x += 2000; H.step(5, hero, inp, 12);
    hero.x -= 2000; H.step(5, hero, inp, 12);
    expect(H.list.some((x) => x.id === h.id)).toBe(false);
    expect(H.list.some((x) => x.herd === h.herd)).toBe(true);
  });
});
