/**
 * Other players' heroes, as the engine draws them: each player's snapshots
 * (sent ~15 times a second) are played back 120 ms late, interpolated between
 * the two around that time (extrapolated briefly when one is late), into a
 * HeroState that drives a HeroRig of its own — the same animation as the
 * player's own hero — and packed camera-relative like it, in the player's
 * dress. Allocation-free per frame once every player has been seen.
 *
 * A snapshot also carries the hero's combat pose (rig.ts CombatPose: the
 * weapon, its swing and how far through it, the shield, the bow's draw) and
 * its hearts, so other players see each other fight. Played back, each hero
 * is a rival to the combat world (combat/pvp.ts): what the shield locks onto,
 * what blows and arrows meet, timed by the clock played back. Foes are each
 * player's own.
 */

import type { World } from '../world';
import { COMBAT } from '../constants';
import type { Rival } from '../combat/pvp';
import { characterOf } from '../hero/characters';
import { createHeroState, type HeroMove, type HeroState } from '../hero/locomotion';
import { createCombatPose, createHeroRig, packHeroUniforms, updateHeroRig, HERO_UNIFORM, type CombatPose, type HeroRig, type PoseAct } from '../hero/rig';
import type { NetLink } from './link';

const MOVES: readonly HeroMove[] = ['ground', 'air', 'glide', 'climb', 'swim', 'surf'];
// (new acts go at the end: an older client reads one it does not know as 'none')
const ACTS: readonly PoseAct[] = ['none', 'swing', 'charge', 'guard', 'aim', 'throw', 'hurt', 'stun', 'down', 'dead', 'lift', 'carry', 'hurl', 'push'];

/**
 * Snapshot layout: the sender's clock (ms), root, velocity, facing, locomotion state and flags, the ground
 * under it; then the combat pose (from `pose`: a snapshot without it, from an older build, is at peace),
 * then its hearts left (quarters) and heart containers (without them: all there), then the models of its
 * shield (0: none carried) and bow (without them: the Hylian Shield and the Traveler's Bow), then the trunk
 * it climbs (its centre, radius and top; radius 0: none, as from an older build), how far through a vault
 * onto its top, and the trunk's taper and lean above and below the hips; then a skydive's posture (without it:
 * spread); then its world (0 the land, 1 the cloud world: world.ts; without it, the land).
 */
const S = {
  t: 0, x: 1, y: 2, z: 3, vx: 4, vy: 5, vz: 6, yaw: 7, move: 8, flags: 9, landings: 10, impact: 11, ground: 12, gx: 13, gz: 14,
  pose: 15, weapon: 15, act: 16, cmove: 17, cu: 18, pitch: 19, draw: 20, glow: 21, spin: 22, flip: 23, roll: 24,
  health: 25, hearts: 26, shield: 27, bow: 28, climbX: 29, climbZ: 30, climbR: 31, climbTop: 32, mantle: 33, taper: 34, leanX: 35, leanZ: 36, taperDown: 37, leanDownX: 38, leanDownZ: 39, dive: 40, world: 41, length: 42,
} as const;
const SPRINT = 1, SNEAK = 2, SLIDE = 4, TIRED = 8, SURFING = 16, ARMED = 32, SHIELD = 64, SKYDIVE = 128, FLARE = 256, ROLL = 512;

/** Milliseconds between the snapshots a player sends. */
export const SEND_INTERVAL = 1000 / 15;
const DELAY = 120;        // ms the playback trails the newest snapshot: a send interval and the network's jitter
const EXTRAPOLATE = 250;  // ms a late player keeps moving along its last velocity
const HISTORY = 24;       // snapshots kept per player
const TELEPORT = 30;      // m between two snapshots: a respawn or a seed change, not a run
const DRAW_RANGE = 3000;  // m: farther heroes are under a pixel

const round = (v: number, k: number) => Math.round(v * k) / k;
const wrap = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

const AT_PEACE = createCombatPose();

/** The player's hero as a snapshot (in `out`), at time t (performance.now(), ms), with its combat pose, hearts and world. */
export function encodeSnapshot(s: HeroState, t: number, out: number[], pose: CombatPose = AT_PEACE,
  health: number = COMBAT.hearts * 4, hearts: number = COMBAT.hearts, world: World = 'land'): number[] {
  out.length = S.length;
  out[S.t] = Math.round(t);
  out[S.x] = round(s.x, 1000); out[S.y] = round(s.y, 1000); out[S.z] = round(s.z, 1000);
  out[S.vx] = round(s.vx, 100); out[S.vy] = round(s.vy, 100); out[S.vz] = round(s.vz, 100);
  out[S.yaw] = round(s.yaw, 1000);
  out[S.move] = Math.max(0, MOVES.indexOf(s.move));
  out[S.flags] = (s.sprinting ? SPRINT : 0) | (s.sneaking ? SNEAK : 0) | (s.sliding ? SLIDE : 0) | (s.exhausted ? TIRED : 0)
    | (s.surfing ? SURFING : 0) | (pose.armed ? ARMED : 0) | (pose.shield ? SHIELD : 0)
    | (s.skydive ? SKYDIVE : 0) | (s.flare > 0 ? FLARE : 0) | (s.rolling ? ROLL : 0);
  out[S.landings] = s.landings;
  out[S.impact] = round(s.impact, 100);
  out[S.ground] = round(s.ground, 1000);
  out[S.gx] = round(s.gx, 1000); out[S.gz] = round(s.gz, 1000);
  out[S.weapon] = pose.weapon;
  out[S.act] = Math.max(0, ACTS.indexOf(pose.act));
  out[S.cmove] = pose.move;
  out[S.cu] = round(pose.u, 1000);
  out[S.pitch] = round(pose.pitch, 1000);
  out[S.draw] = round(pose.draw, 100);
  out[S.glow] = round(pose.glow, 100);
  out[S.spin] = round(pose.spin, 1000);
  out[S.flip] = round(pose.flip, 1000);
  out[S.roll] = round(pose.roll, 1000);
  out[S.health] = round(health, 100);
  out[S.hearts] = hearts;
  out[S.shield] = pose.shield ? pose.shieldModel : 0;
  out[S.bow] = pose.bowModel;
  out[S.climbX] = round(s.climbX, 1000); out[S.climbZ] = round(s.climbZ, 1000);
  out[S.climbR] = round(s.climbR, 1000); out[S.climbTop] = round(s.climbTop, 1000);
  out[S.mantle] = round(s.mantle, 1000);
  out[S.taper] = round(s.climbTaper, 1000); out[S.leanX] = round(s.climbLeanX, 1000); out[S.leanZ] = round(s.climbLeanZ, 1000);
  out[S.taperDown] = round(s.climbTaperDown, 1000); out[S.leanDownX] = round(s.climbLeanDownX, 1000); out[S.leanDownZ] = round(s.climbLeanDownZ, 1000);
  out[S.dive] = round(s.dive, 100);
  out[S.world] = world === 'clouds' ? 1 : 0;
  return out;
}

/** Another player's hero as played back here (a rival to the combat world). */
export type RemoteHero = Rival & {
  char: number;
  snaps: number[][];   // oldest first
  offset: number;      // local ms − sender ms: the least seen (the fastest delivery), let creep up
  readonly body: HeroState;
  readonly rig: HeroRig;
  packed: Float32Array;
  dist2: number;
};

export class RemoteHeroes {
  private peers = new Map<string, RemoteHero>();
  private order: RemoteHero[] = [];
  private seen: RemoteHero[] = [];
  private out: Float32Array[] = [];
  // the world played in here (0 the land, 1 the clouds): players in the other are neither drawn nor fought
  private here = 0;

  /** The other players seen so far (a snapshot from each), as of the last update. */
  list(): readonly RemoteHero[] {
    return this.seen;
  }

  /** Take the snapshots that arrived, forget players who left, and pose every hero in `world` for time `now`. */
  update(link: NetLink, now: number, dt: number, world: World = 'land'): void {
    this.here = world === 'clouds' ? 1 : 0;
    const ids = link.peers();
    for (const id of this.peers.keys()) if (!ids.has(id)) this.peers.delete(id);
    link.drain((id, at, d) => {
      const char = ids.get(id);
      if (char === undefined || d.length < S.pose || !d.every(Number.isFinite)) return;
      let p = this.peers.get(id);
      if (!p) {
        const body = createHeroState(d[S.x], d[S.z], d[S.yaw], 0);
        p = {
          id, char, snaps: [], offset: at - d[S.t], body, pose: createCombatPose(), rig: createHeroRig(),
          packed: new Float32Array(HERO_UNIFORM.length), dist2: 0,
          health: COMBAT.hearts * 4, hearts: COMBAT.hearts, clock: -Infinity, hitSwing: -1, hitShot: -1,
        };
        this.peers.set(id, p);
      }
      p.char = char;
      const last = p.snaps[p.snaps.length - 1];
      if (last && d[S.t] <= last[S.t]) return; // (one socket, in order: only a sender's clock jumping back)
      p.offset = Math.min(at - d[S.t], p.offset + 1);
      const snap = d.slice(0, S.length);
      // (from an older build: without a pose, at peace with the sword and shield on the back; without
      // its hearts, all there)
      for (let k = d.length; k < S.length; k++) {
        snap[k] = k === S.weapon ? AT_PEACE.weapon : k === S.draw ? -1 : k === S.health ? COMBAT.hearts * 4 : k === S.hearts ? COMBAT.hearts
          : k === S.shield ? AT_PEACE.shieldModel : k === S.bow ? AT_PEACE.bowModel : 0;
      }
      if (d.length <= S.pose) snap[S.flags] |= SHIELD;
      p.snaps.push(snap);
      if (p.snaps.length > HISTORY) p.snaps.shift();
    });
    const seen = this.seen;
    seen.length = 0;
    for (const p of this.peers.values()) {
      if (!this.inHere(p)) continue;
      this.sample(p, now - p.offset - DELAY);
      updateHeroRig(p.rig, p.body, dt, p.pose);
      seen.push(p);
    }
  }

  /** Has the player been seen, its newest snapshot in this world? */
  private inHere(p: RemoteHero): boolean {
    return p.snaps.length > 0 && p.snaps[p.snaps.length - 1][S.world] === this.here;
  }

  /** The player's state at sender time t: between the snapshots around it, or past the newest. */
  private sample(p: RemoteHero, t: number): void {
    const a = p.snaps;
    const n = a.length;
    const s = p.body;
    p.clock = t;
    if (t >= a[n - 1][S.t] || n === 1) {
      const e = Math.min(Math.max(t - a[n - 1][S.t], 0), EXTRAPOLATE) / 1000;
      this.take(p, a[n - 1], a[n - 1], 0);
      s.x += s.vx * e; s.y += s.vy * e; s.z += s.vz * e;
      return;
    }
    if (t <= a[0][S.t]) {
      this.take(p, a[0], a[0], 0);
      return;
    }
    let i = n - 1;
    while (i > 1 && a[i - 1][S.t] > t) i--;
    const s0 = a[i - 1], s1 = a[i];
    const dx = s1[S.x] - s0[S.x], dy = s1[S.y] - s0[S.y], dz = s1[S.z] - s0[S.z];
    if (dx * dx + dy * dy + dz * dz > TELEPORT * TELEPORT) {
      this.take(p, s1, s1, 0);
      p.rig.snap = true;
      return;
    }
    this.take(p, s0, s1, (t - s0[S.t]) / (s1[S.t] - s0[S.t]));
  }

  private take(p: RemoteHero, a: number[], b: number[], f: number): void {
    const s = p.body;
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
    s.surfing = (d[S.flags] & SURFING) !== 0;
    s.landings = a[S.landings];
    s.impact = a[S.impact];
    s.rolling = (a[S.flags] & ROLL) !== 0;
    s.skydive = (d[S.flags] & SKYDIVE) !== 0;
    s.flare = (d[S.flags] & FLARE) !== 0 ? 1 : 0;
    s.dive = l(S.dive);
    // (one trunk at a time: the nearer snapshot's; a vault's progress in between)
    s.climbX = d[S.climbX]; s.climbZ = d[S.climbZ]; s.climbR = d[S.climbR]; s.climbTop = d[S.climbTop];
    s.climbTaper = d[S.taper]; s.climbLeanX = d[S.leanX]; s.climbLeanZ = d[S.leanZ];
    s.climbTaperDown = d[S.taperDown]; s.climbLeanDownX = d[S.leanDownX]; s.climbLeanDownZ = d[S.leanDownZ];
    s.mantle = a[S.mantle] > 0 && b[S.mantle] > 0 ? l(S.mantle) : d[S.mantle];
    // The pose: the nearer snapshot's act; within one swing (or act) its progress in between.
    const c = p.pose;
    c.weapon = d[S.weapon];
    c.armed = (d[S.flags] & ARMED) !== 0;
    c.shield = (d[S.flags] & SHIELD) !== 0;
    c.shieldModel = d[S.shield] > 0 ? d[S.shield] : AT_PEACE.shieldModel;
    c.bowModel = d[S.bow];
    c.act = ACTS[d[S.act]] ?? 'none';
    c.move = d[S.cmove];
    const same = a[S.act] === b[S.act] && a[S.cmove] === b[S.cmove] && b[S.cu] >= a[S.cu];
    c.u = same ? l(S.cu) : d[S.cu];
    c.pitch = l(S.pitch);
    c.draw = d[S.draw] < 0 ? d[S.draw] : same ? l(S.draw) : d[S.draw];
    c.glow = l(S.glow);
    c.spin = same ? l(S.spin) : d[S.spin];
    c.flip = same ? l(S.flip) : d[S.flip];
    c.roll = l(S.roll);
    p.health = d[S.health];
    p.hearts = d[S.hearts];
  }

  /** Up to `max` heroes nearest the camera, packed camera-relative (rig.ts HERO_UNIFORM) in their dresses. */
  pack(camX: number, camY: number, camZ: number, max: number): readonly Float32Array[] {
    const order = this.order;
    order.length = 0;
    for (const p of this.peers.values()) {
      if (!this.inHere(p)) continue;
      const dx = p.body.x - camX, dy = p.body.y - camY, dz = p.body.z - camZ;
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
      packHeroUniforms(p.rig, p.body, camX, camY, camZ, p.packed, characterOf(p.char).dress);
      out.push(p.packed);
    }
    return out;
  }
}
