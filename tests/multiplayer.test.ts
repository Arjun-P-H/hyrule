import { describe, it, expect } from 'bun:test';
import { COMBAT, DEFAULT_SEED, SPAWN } from '../src/lib/constants';
import { CHARACTERS } from '../src/lib/hero/characters';
import { createHeroState, type HeroState } from '../src/lib/hero/locomotion';
import { GEAR, HERO_UNIFORM, J, createCombatPose, createHeroRig, packHeroUniforms, updateHeroRig } from '../src/lib/hero/rig';
import { MOVE } from '../src/lib/combat/moves';
import { BOW_MODEL, MODEL, SHIELD_MODEL } from '../src/lib/combat/weapons';
import { HERO_BYTES, HERO_OTHERS, OTHERS, packHeroOthers, packHeroWgsl } from '../src/lib/gpu/hero.wgsl';
import { RemoteHeroes, SEND_INTERVAL, encodeSnapshot } from '../src/lib/net/remoteHeroes';
import type { ClockState, NetLink } from '../src/lib/net/link';

/**
 * Multiplayer: other players' heroes played back from their snapshots
 * (interpolated, extrapolated, teleports, departures), packed in their
 * dresses; the dress and the other players' shadow bodies in the WGSL block.
 */

/** A session stand-in: the players in the world and the snapshots that arrived. */
function fakeLink(peers: Map<string, number>) {
  const inbox: Array<[string, number, number[]]> = [];
  const link: NetLink = {
    inWorld: () => true,
    peers: () => peers,
    drain: (take) => { for (const [id, at, d] of inbox) take(id, at, d); inbox.length = 0; },
    sendState: () => {},
    drainCombat: () => {},
    sendCombat: () => {},
    peerName: (id) => id,
    takeClock: (): ClockState | null => null,
    sendClock: () => {},
    keepsClock: () => false,
  };
  return { link, inbox };
}

/** A hero running along +x at `speed` m/s from the spawn: its snapshot at sender time t (ms). */
function runner(speed: number) {
  const s = createHeroState(SPAWN.x, SPAWN.z, Math.PI / 2, DEFAULT_SEED);
  const x0 = s.x;
  return (t: number) => {
    s.x = x0 + (speed * t) / 1000;
    s.vx = speed;
    s.yaw = Math.PI / 2;
    return encodeSnapshot(s, t, []);
  };
}

/** The world x of the only hero drawn (its packed pelvis joint, the camera at the spawn: heroes farther than
 *  the draw range are not drawn). */
const packedX = (r: RemoteHeroes) => r.pack(SPAWN.x, 0, SPAWN.z, 4)[0][3 * J.pelvis] + SPAWN.x;

describe('remote heroes', () => {
  it('play back 120 ms late, interpolated between snapshots, whatever the network delay', () => {
    const snap = runner(4);
    const { link, inbox } = fakeLink(new Map([['a', 2]]));
    const r = new RemoteHeroes();
    const latency = 80; // ms, sender clock to local clock
    const s0 = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED).x;
    let local = 0;
    for (let t = 0; t <= 2000; t += SEND_INTERVAL) {
      inbox.push(['a', t + latency, snap(t)]);
      // frames at 60 Hz until the next snapshot is sent
      for (; local < t + latency + SEND_INTERVAL; local += 1000 / 60) {
        r.update(link, local, 1 / 60);
        if (t < 500) continue;
        const shown = local - latency - 120; // the sender's time on screen
        const x = packedX(r);
        const want = s0 + (4 * shown) / 1000;
        // the rig's pelvis sits within a few cm of the root across the gait
        expect(Math.abs(x - want)).toBeLessThan(0.25);
      }
    }
  });

  it('keep moving a moment past the newest snapshot, then hold', () => {
    const snap = runner(5);
    const { link, inbox } = fakeLink(new Map([['a', 0]]));
    const r = new RemoteHeroes();
    for (let t = 0; t <= 1000; t += SEND_INTERVAL) inbox.push(['a', t, snap(t)]);
    const last = snap(Math.floor(1000 / SEND_INTERVAL) * SEND_INTERVAL);
    r.update(link, 1120 + 100, 1 / 60); // 100 ms past the newest
    const a = packedX(r);
    r.update(link, 1120 + 2000, 1 / 60); // long gone: held 250 ms past it
    const b = packedX(r);
    expect(a).toBeGreaterThan(last[1] + 0.3);
    // (the same pose either side: the pelvis's lean ahead of the root cancels)
    expect(Math.abs(b - a - 5 * (0.25 - 0.1))).toBeLessThan(0.1);
  });

  it('jump across a teleport instead of sliding through the world', () => {
    const { link, inbox } = fakeLink(new Map([['a', 0]]));
    const r = new RemoteHeroes();
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    inbox.push(['a', 0, encodeSnapshot(s, 0, [])]);
    const far = { ...s, x: s.x + 500 } as HeroState;
    inbox.push(['a', 100, encodeSnapshot(far, 100, [])]);
    r.update(link, 50 + 120, 1 / 60); // halfway between the two
    expect(Math.abs(packedX(r) - far.x)).toBeLessThan(0.5);
  });

  it('forget players who left, ignore broken snapshots and players not in the world', () => {
    const peers = new Map([['a', 0], ['b', 1]]);
    const { link, inbox } = fakeLink(peers);
    const r = new RemoteHeroes();
    const snap = runner(0);
    inbox.push(['a', 0, snap(0)], ['b', 0, snap(0)], ['ghost', 0, snap(0)]);
    inbox.push(['a', 10, [10, NaN, 0, 0]], ['b', 10, snap(10).slice(0, 5)]);
    r.update(link, 200, 1 / 60);
    expect(r.pack(SPAWN.x, 0, SPAWN.z, 4).length).toBe(2);
    peers.delete('b');
    r.update(link, 216, 1 / 60);
    expect(r.pack(SPAWN.x, 0, SPAWN.z, 4).length).toBe(1);
  });

  it('carry the combat pose: a swing is played back with the sword in hand, its progress in between; old snapshots are at peace', () => {
    const { link, inbox } = fakeLink(new Map([['a', 0], ['b', 1]]));
    const r = new RemoteHeroes();
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    const pose = { ...createCombatPose(), armed: true, act: 'swing' as const, move: MOVE.slashR, u: 0.2 };
    inbox.push(['a', 0, encodeSnapshot(s, 0, [], pose)]);
    inbox.push(['a', 100, encodeSnapshot(s, 100, [], { ...pose, u: 0.45 })]);
    // a snapshot of an older build: no pose at all
    inbox.push(['b', 0, encodeSnapshot({ ...s, x: s.x + 3 } as HeroState, 0, []).slice(0, 15)]);
    for (let t = 0; t <= 170; t += 1000 / 60) r.update(link, t, 1 / 60);
    const [a, b] = r.pack(SPAWN.x, 0, SPAWN.z, 4);
    // (halfway between the two snapshots, 120 ms late)
    const g = HERO_UNIFORM.gear + GEAR.grip;
    expect(a[g + 3]).toBe(MODEL.sword);
    const hand = 3 * J.handR;
    expect(Math.hypot(a[g] - a[hand], a[g + 1] - a[hand + 1], a[g + 2] - a[hand + 2])).toBeLessThan(1e-4);
    // b: its sword sheathed (the hilt over the shoulder, far from the hand), its shield on the back
    expect(b[g + 3]).toBe(MODEL.sword);
    expect(b[HERO_UNIFORM.gear + GEAR.flat + 3]).toBe(1);
    expect(b[HERO_UNIFORM.gear + GEAR.shield + 3]).toBe(1);
  });

  it('carry the hero\'s hearts, and are listed for the combat world with the clock they show (older snapshots: all hearts)', () => {
    const { link, inbox } = fakeLink(new Map([['a', 0], ['b', 1], ['c', 2]]));
    const r = new RemoteHeroes();
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    inbox.push(['a', 1000, encodeSnapshot(s, 0, [], createCombatPose(), 7.25, 6)]);
    inbox.push(['b', 1000, encodeSnapshot(s, 0, [], { ...createCombatPose(), act: 'guard' }).slice(0, 25)]); // (with a pose, no hearts)
    inbox.push(['c', 1000, encodeSnapshot(s, 0, []).slice(0, 15)]);
    r.update(link, 1200, 1 / 60);
    const [a, b, c] = ['a', 'b', 'c'].map((id) => r.list().find((p) => p.id === id)!);
    expect([a.health, a.hearts]).toEqual([7.25, 6]);
    expect([b.health, b.hearts]).toEqual([COMBAT.hearts * 4, COMBAT.hearts]);
    expect([c.health, c.hearts]).toEqual([COMBAT.hearts * 4, COMBAT.hearts]);
    // (the shield up still shows on a pose without hearts)
    expect(b.pose.act).toBe('guard');
    // the sender's clock shown: 1000 ms of delivery, 120 of playback
    expect(a.clock).toBeCloseTo(1200 - 1000 - 120, 6);
    expect(r.list().length).toBe(3);
  });

  it('carry the surface a hero climbs, so it is posed hugging the same bark (older snapshots: a flat wall)', () => {
    const { link, inbox } = fakeLink(new Map([['a', 0], ['b', 1]]));
    const r = new RemoteHeroes();
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    // (on the bark of a trunk 0.6 m round, its axis 0.6 m ahead, 3 m up a 9 m trunk, a third of the way through a vault)
    const climbing = { ...s, move: 'climb', climbX: s.x, climbZ: s.z + 0.6, climbR: 0.6, climbTop: s.y + 9, mantle: 0.33 } as HeroState;
    inbox.push(['a', 0, encodeSnapshot(climbing, 0, [])]);
    inbox.push(['b', 0, encodeSnapshot(climbing, 0, []).slice(0, 29)]); // (a build before trees were climbed)
    for (let t = 0; t <= 400; t += 1000 / 60) r.update(link, t, 1 / 60);
    const [a, b] = ['a', 'b'].map((id) => r.list().find((p) => p.id === id)!);
    expect([a.body.climbX, a.body.climbZ, a.body.climbR, a.body.climbTop, a.body.mantle])
      .toEqual([+climbing.climbX.toFixed(3), +climbing.climbZ.toFixed(3), 0.6, +climbing.climbTop.toFixed(3), 0.33]);
    expect(a.body.move).toBe('climb');
    expect(a.rig.treeR).toBeCloseTo(0.6, 6);
    expect(b.body.climbR).toBe(0);
    expect(b.rig.treeR).toBe(30); // (no surface sent: held as a flat wall)
    for (const v of [...a.rig.local, ...b.rig.local]) expect(Number.isFinite(v)).toBe(true);
  });

  it('carry the shield and the bow: their models are played back (older snapshots: the Hylian Shield and the Traveler\'s Bow)', () => {
    const { link, inbox } = fakeLink(new Map([['a', 0], ['b', 1], ['c', 2]]));
    const r = new RemoteHeroes();
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    const aiming = { ...createCombatPose(), armed: true, act: 'aim' as const, draw: 0.5, shieldModel: SHIELD_MODEL.lid, bowModel: BOW_MODEL.soldier };
    inbox.push(['a', 0, encodeSnapshot(s, 0, [], aiming)]);
    inbox.push(['b', 0, encodeSnapshot(s, 0, [], { ...createCombatPose(), shield: false })]);
    inbox.push(['c', 0, encodeSnapshot(s, 0, [], createCombatPose()).slice(0, 27)]); // (a build before the models)
    for (let t = 0; t <= 400; t += 1000 / 60) r.update(link, t, 1 / 60);
    const [a, b, c] = ['a', 'b', 'c'].map((id) => r.list().find((p) => p.id === id)!);
    expect([a.pose.shield, a.pose.shieldModel, a.pose.bowModel]).toEqual([true, SHIELD_MODEL.lid, BOW_MODEL.soldier]);
    expect(b.pose.shield).toBe(false);
    expect([c.pose.shield, c.pose.shieldModel, c.pose.bowModel]).toEqual([true, SHIELD_MODEL.hylian, BOW_MODEL.traveler]);
    // packed for the renderer: the models ride in the gear frames' w (the bow only while drawn)
    const [pa, pb] = r.pack(SPAWN.x, 0, SPAWN.z, 4);
    expect(pa[HERO_UNIFORM.gear + GEAR.shield + 3]).toBe(SHIELD_MODEL.lid);
    expect(pa[HERO_UNIFORM.gear + GEAR.bow + 3]).toBe(BOW_MODEL.soldier);
    expect(pb[HERO_UNIFORM.gear + GEAR.shield + 3]).toBe(0);
    expect(pb[HERO_UNIFORM.gear + GEAR.bow + 3]).toBe(0);
  });

  it('pack the nearest, at most `max`, each in its dress', () => {
    const peers = new Map(CHARACTERS.map((c) => [`p${c.id}`, c.id] as [string, number]));
    const { link, inbox } = fakeLink(peers);
    const r = new RemoteHeroes();
    for (const c of CHARACTERS) {
      const s = createHeroState(SPAWN.x + 10 * c.id, SPAWN.z, 0, DEFAULT_SEED);
      inbox.push([`p${c.id}`, 0, encodeSnapshot(s, 0, [])]);
    }
    r.update(link, 200, 1 / 60);
    // (the camera at the ground's height there: distances are in three dimensions, and at height 0 the order of heroes ten metres apart
    // was the order of their ground's heights, whatever the land at the spawn is)
    const camY = createHeroState(SPAWN.x + 40, SPAWN.z, 0, DEFAULT_SEED).y;
    const out = r.pack(SPAWN.x + 40, camY, SPAWN.z, 3); // nearest: characters 4, 3, 2
    expect(out.length).toBe(3);
    const dresses = out.map((p) => [...p.subarray(HERO_UNIFORM.dress, HERO_UNIFORM.dress + 3)].map((v) => v.toFixed(4)).join());
    for (const id of [4, 3, 2]) expect(dresses).toContain(CHARACTERS[id].dress.map((v) => v.toFixed(4)).join());
  });
});

describe('remote heroes in two worlds', () => {
  it('only the players in this world are drawn and fought; one who comes through the portal shows up', () => {
    const { link, inbox } = fakeLink(new Map([['land', 0], ['sky', 1]]));
    const r = new RemoteHeroes();
    const h = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    let t = 0;
    const send = (id: string, world: 'land' | 'clouds') => inbox.push([id, t, encodeSnapshot(h, t, [], undefined, undefined, undefined, world)]);
    send('land', 'land'); send('sky', 'clouds');
    r.update(link, t, 1 / 60, 'land');
    expect(r.list().map((p) => p.id)).toEqual(['land']);
    expect(r.pack(SPAWN.x, 0, SPAWN.z, 4).length).toBe(1);
    r.update(link, t, 1 / 60, 'clouds');
    expect(r.list().map((p) => p.id)).toEqual(['sky']);
    // the sky player comes back through the portal
    t += SEND_INTERVAL;
    send('sky', 'land');
    r.update(link, t, 1 / 60, 'land');
    expect(r.list().map((p) => p.id).sort()).toEqual(['land', 'sky']);
  });

  it('a snapshot from an older build (no world in it) is on the land', () => {
    const { link, inbox } = fakeLink(new Map([['old', 0]]));
    const r = new RemoteHeroes();
    const d = encodeSnapshot(createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED), 0, []).slice(0, 41);
    inbox.push(['old', 0, d]);
    r.update(link, 0, 1 / 60, 'land');
    expect(r.list().length).toBe(1);
    r.update(link, 0, 1 / 60, 'clouds');
    expect(r.list().length).toBe(0);
  });
});

describe('hero block: dress and other players', () => {
  const posed = (x: number, dress = CHARACTERS[1].dress) => {
    const s = createHeroState(SPAWN.x + x, SPAWN.z, 0, DEFAULT_SEED);
    const rig = createHeroRig();
    updateHeroRig(rig, s, 1 / 60);
    const out = new Float32Array(HERO_UNIFORM.length);
    packHeroUniforms(rig, s, SPAWN.x, s.y, SPAWN.z, out, dress);
    return out;
  };

  it('packs the dress (classic green by default) into the WGSL block', () => {
    const block = new Float32Array(HERO_BYTES / 4);
    packHeroWgsl(posed(0), block);
    expect([...block.subarray(112, 115)]).toEqual([...new Float32Array(CHARACTERS[1].dress)]);
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, DEFAULT_SEED);
    const plain = new Float32Array(HERO_UNIFORM.length);
    packHeroUniforms(createHeroRig(), s, 0, 0, 0, plain);
    expect([...plain.subarray(HERO_UNIFORM.dress, HERO_UNIFORM.dress + 3)]).toEqual([...new Float32Array(CHARACTERS[0].dress)]);
  });

  it('lists the other players (pelvis, head, between the ankles), then the foes, and ends the list', () => {
    const block = new Float32Array(HERO_BYTES / 4).fill(7);
    const a = posed(3), b = posed(-3), gone = posed(9);
    gone[HERO_UNIFORM.bound + 3] = 0; // not drawn: skipped
    const foes = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    packHeroOthers([a, gone, b], block, foes, 1);
    for (const [k, src] of [[0, a], [1, b]] as const) {
      const o = HERO_OTHERS + 12 * k;
      expect(block[o + 3]).toBe(1);
      for (let c = 0; c < 3; c++) {
        expect(block[o + c]).toBeCloseTo(src[3 * J.pelvis + c], 5);
        expect(block[o + 4 + c]).toBeCloseTo(src[3 * J.head + c], 5);
        expect(block[o + 8 + c]).toBeCloseTo(0.5 * (src[3 * J.ankleL + c] + src[3 * J.ankleR + c]), 5);
      }
    }
    // (a foe's body after the players: its pelvis, head and feet as given)
    expect([...block.subarray(HERO_OTHERS + 24, HERO_OTHERS + 27)]).toEqual([1, 2, 3]);
    expect([...block.subarray(HERO_OTHERS + 28, HERO_OTHERS + 31)]).toEqual([4, 5, 6]);
    expect([...block.subarray(HERO_OTHERS + 32, HERO_OTHERS + 35)]).toEqual([7, 8, 9]);
    expect(block[HERO_OTHERS + 36 + 3]).toBe(0);
    packHeroOthers(null, block);
    expect(block[HERO_OTHERS + 3]).toBe(0);
    expect(HERO_BYTES / 4).toBe(HERO_OTHERS + 12 * OTHERS);
  });
});
