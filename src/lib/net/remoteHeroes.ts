/**
 * Other players' heroes, as the engine draws them: each player's snapshots
 * (sent ~15 times a second) are played back 120 ms late, interpolated between
 * the two around that time (extrapolated briefly when one is late), into a
 * HeroState that drives a HeroRig of its own — the same animation as the
 * player's own hero — and packed camera-relative like it, in the player's
 * dress. Allocation-free per frame once every player has been seen.
 */

import { characterOf } from '../hero/characters';
import { createHeroState, type HeroMove, type HeroState } from '../hero/locomotion';
import { createHeroRig, packHeroUniforms, updateHeroRig, HERO_UNIFORM, type HeroRig } from '../hero/rig';
import type { NetLink } from './link';

const MOVES: readonly HeroMove[] = ['ground', 'air', 'glide', 'climb', 'swim'];

/** Snapshot layout: the sender's clock (ms), root, velocity, facing, locomotion state and flags, the ground under it. */
const S = { t: 0, x: 1, y: 2, z: 3, vx: 4, vy: 5, vz: 6, yaw: 7, move: 8, flags: 9, landings: 10, impact: 11, ground: 12, gx: 13, gz: 14, length: 15 } as const;
const SPRINT = 1, SNEAK = 2, SLIDE = 4, TIRED = 8;

/** Milliseconds between the snapshots a player sends. */
export const SEND_INTERVAL = 1000 / 15;
const DELAY = 120;        // ms the playback trails the newest snapshot: a send interval and the network's jitter
const EXTRAPOLATE = 250;  // ms a late player keeps moving along its last velocity
const HISTORY = 24;       // snapshots kept per player
const TELEPORT = 30;      // m between two snapshots: a respawn or a seed change, not a run
const DRAW_RANGE = 3000;  // m: farther heroes are under a pixel

const round = (v: number, k: number) => Math.round(v * k) / k;
const wrap = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

/** The player's hero as a snapshot (in `out`), at time t (performance.now(), ms). */
export function encodeSnapshot(s: HeroState, t: number, out: number[]): number[] {
  out.length = S.length;
  out[S.t] = Math.round(t);
  out[S.x] = round(s.x, 1000); out[S.y] = round(s.y, 1000); out[S.z] = round(s.z, 1000);
  out[S.vx] = round(s.vx, 100); out[S.vy] = round(s.vy, 100); out[S.vz] = round(s.vz, 100);
  out[S.yaw] = round(s.yaw, 1000);
  out[S.move] = Math.max(0, MOVES.indexOf(s.move));
  out[S.flags] = (s.sprinting ? SPRINT : 0) | (s.sneaking ? SNEAK : 0) | (s.sliding ? SLIDE : 0) | (s.exhausted ? TIRED : 0);
  out[S.landings] = s.landings;
  out[S.impact] = round(s.impact, 100);
  out[S.ground] = round(s.ground, 1000);
  out[S.gx] = round(s.gx, 1000); out[S.gz] = round(s.gz, 1000);
  return out;
}

type Peer = {
  char: number;
  snaps: number[][];   // oldest first
  offset: number;      // local ms − sender ms: the least seen (the fastest delivery), let creep up
  state: HeroState;
  rig: HeroRig;
  packed: Float32Array;
  dist2: number;
};

export class RemoteHeroes {
  private peers = new Map<string, Peer>();
  private order: Peer[] = [];
  private out: Float32Array[] = [];

  /** Take the snapshots that arrived, forget players who left, and pose every hero for time `now`. */
  update(link: NetLink, now: number, dt: number): void {
    const ids = link.peers();
    for (const id of this.peers.keys()) if (!ids.has(id)) this.peers.delete(id);
    link.drain((id, at, d) => {
      const char = ids.get(id);
      if (char === undefined || d.length < S.length || !d.every(Number.isFinite)) return;
      let p = this.peers.get(id);
      if (!p) {
        const state = createHeroState(d[S.x], d[S.z], d[S.yaw], 0);
        p = { char, snaps: [], offset: at - d[S.t], state, rig: createHeroRig(), packed: new Float32Array(HERO_UNIFORM.length), dist2: 0 };
        this.peers.set(id, p);
      }
      p.char = char;
      const last = p.snaps[p.snaps.length - 1];
      if (last && d[S.t] <= last[S.t]) return; // (one socket, in order: only a sender's clock jumping back)
      p.offset = Math.min(at - d[S.t], p.offset + 1);
      p.snaps.push(d.slice(0, S.length));
      if (p.snaps.length > HISTORY) p.snaps.shift();
    });
    for (const p of this.peers.values()) {
      if (!p.snaps.length) continue;
      this.sample(p, now - p.offset - DELAY);
      updateHeroRig(p.rig, p.state, dt);
    }
  }

  /** The player's state at sender time t: between the snapshots around it, or past the newest. */
  private sample(p: Peer, t: number): void {
    const a = p.snaps;
    const n = a.length;
    const s = p.state;
    if (t >= a[n - 1][S.t] || n === 1) {
      const e = Math.min(Math.max(t - a[n - 1][S.t], 0), EXTRAPOLATE) / 1000;
      this.take(s, a[n - 1], a[n - 1], 0);
      s.x += s.vx * e; s.y += s.vy * e; s.z += s.vz * e;
      return;
    }
    if (t <= a[0][S.t]) {
      this.take(s, a[0], a[0], 0);
      return;
    }
    let i = n - 1;
    while (i > 1 && a[i - 1][S.t] > t) i--;
    const s0 = a[i - 1], s1 = a[i];
    const dx = s1[S.x] - s0[S.x], dy = s1[S.y] - s0[S.y], dz = s1[S.z] - s0[S.z];
    if (dx * dx + dy * dy + dz * dz > TELEPORT * TELEPORT) {
      this.take(s, s1, s1, 0);
      p.rig.snap = true;
      return;
    }
    this.take(s, s0, s1, (t - s0[S.t]) / (s1[S.t] - s0[S.t]));
  }

  private take(s: HeroState, a: number[], b: number[], f: number): void {
    const l = (k: number) => a[k] + (b[k] - a[k]) * f;
    s.x = l(S.x); s.y = l(S.y); s.z = l(S.z);
    s.vx = l(S.vx); s.vy = l(S.vy); s.vz = l(S.vz);
    s.yaw = wrap(a[S.yaw] + wrap(b[S.yaw] - a[S.yaw]) * f);
    s.ground = l(S.ground); s.gx = l(S.gx); s.gz = l(S.gz);
    const d = f < 0.5 ? a : b;
    s.move = MOVES[d[S.move]] ?? 'ground';
    s.sprinting = (d[S.flags] & SPRINT) !== 0;
    s.sneaking = (d[S.flags] & SNEAK) !== 0;
    s.sliding = (d[S.flags] & SLIDE) !== 0;
    s.exhausted = (d[S.flags] & TIRED) !== 0;
    s.landings = a[S.landings];
    s.impact = a[S.impact];
  }

  /** Up to `max` heroes nearest the camera, packed camera-relative (rig.ts HERO_UNIFORM) in their dresses. */
  pack(camX: number, camY: number, camZ: number, max: number): readonly Float32Array[] {
    const order = this.order;
    order.length = 0;
    for (const p of this.peers.values()) {
      if (!p.snaps.length) continue;
      const dx = p.state.x - camX, dy = p.state.y - camY, dz = p.state.z - camZ;
      p.dist2 = dx * dx + dy * dy + dz * dz;
      if (p.dist2 < DRAW_RANGE * DRAW_RANGE) order.push(p);
    }
    if (order.length > max) {
      order.sort((a, b) => a.dist2 - b.dist2);
      order.length = max;
    }
    const out = this.out;
    out.length = 0;
    for (const p of order) {
      packHeroUniforms(p.rig, p.state, camX, camY, camZ, p.packed, characterOf(p.char).dress);
      out.push(p.packed);
    }
    return out;
  }
}
