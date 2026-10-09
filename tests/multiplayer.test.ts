import { describe, it, expect } from 'bun:test';
import { DEFAULT_SEED, SPAWN } from '../src/lib/constants';
import { CHARACTERS } from '../src/lib/hero/characters';
import { createHeroState, type HeroState } from '../src/lib/hero/locomotion';
import { HERO_UNIFORM, J, createHeroRig, packHeroUniforms, updateHeroRig } from '../src/lib/hero/rig';
import { HERO_BYTES, packHeroOthers, packHeroWgsl } from '../src/lib/gpu/hero.wgsl';
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

  it('pack the nearest, at most `max`, each in its dress', () => {
    const peers = new Map(CHARACTERS.map((c) => [`p${c.id}`, c.id] as [string, number]));
    const { link, inbox } = fakeLink(peers);
    const r = new RemoteHeroes();
    for (const c of CHARACTERS) {
      const s = createHeroState(SPAWN.x + 10 * c.id, SPAWN.z, 0, DEFAULT_SEED);
      inbox.push([`p${c.id}`, 0, encodeSnapshot(s, 0, [])]);
    }
    r.update(link, 200, 1 / 60);
    const out = r.pack(SPAWN.x + 40, 0, SPAWN.z, 3); // nearest: characters 4, 3, 2
    expect(out.length).toBe(3);
    const dresses = out.map((p) => [...p.subarray(HERO_UNIFORM.dress, HERO_UNIFORM.dress + 3)].map((v) => v.toFixed(4)).join());
    for (const id of [4, 3, 2]) expect(dresses).toContain(CHARACTERS[id].dress.map((v) => v.toFixed(4)).join());
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

  it('lists the other players (pelvis, head, between the ankles) and ends the list', () => {
    const block = new Float32Array(HERO_BYTES / 4).fill(7);
    const a = posed(3), b = posed(-3), gone = posed(9);
    gone[HERO_UNIFORM.bound + 3] = 0; // not drawn: skipped
    packHeroOthers([a, gone, b], block);
    for (const [k, src] of [[0, a], [1, b]] as const) {
      const o = 116 + 12 * k;
      expect(block[o + 3]).toBe(1);
      for (let c = 0; c < 3; c++) {
        expect(block[o + c]).toBeCloseTo(src[3 * J.pelvis + c], 5);
        expect(block[o + 4 + c]).toBeCloseTo(src[3 * J.head + c], 5);
        expect(block[o + 8 + c]).toBeCloseTo(0.5 * (src[3 * J.ankleL + c] + src[3 * J.ankleR + c]), 5);
      }
    }
    expect(block[116 + 24 + 3]).toBe(0);
    packHeroOthers(null, block);
    expect(block[116 + 3]).toBe(0);
    expect(HERO_BYTES / 4).toBe(116 + 48);
  });
});
