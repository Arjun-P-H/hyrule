/**
 * The horses round the hero, after Breath of the Wild: each one's own mind, riding, taming, the whistle.
 * ---------------------------------------------------------------------------
 * Wild horses (herds.ts) come into being within HORSE.spawnRange of the hero and go beyond despawnRange; the
 * hero's own horses (tamed) stay where they were left. A horse's mind:
 *   graze / idle / wander   at peace round its herd's anchor (Eating, Idle, Walk)
 *   alert                   it noticed the hero: head up, facing it (closer than HORSE.notice; less sneaking)
 *   flee                    it bolted (the hero came closer than HORSE.bolt, or a herd mate bolted): galloping away
 *   lie / sleep / rise      lying down at night (lay_to_idle backwards), asleep, getting up (forwards)
 *   come                    called by the whistle: trotting, then galloping to the hero (the hero's own horses)
 *   struggle                a wild horse just mounted: rearing and bucking until tamed or the rider is thrown
 *   ridden                  under the hero
 *   die / vanish            killed (its health gone: the hero's blows, arrows, throws): the death clip, then gone in a
 *                           dissolve (the renderer) and a drift of motes (the combat scene's 'vanish'); killed, it stays
 *                           gone (dead, by id)
 * Riding: the keys steer it the camera's way (a sharp turn back brakes, and backs it up from a stand); the gait
 * goes up a step with each spur (Sprint: a spur spent, they come back with rest), to a gallop that eases to a
 * canter; Jump leaps from a trot or faster; it refuses a drop, a wall too steep and deep water. Action mounts
 * (from close by; a wild horse must be reached without it bolting: sneak), soothes a struggling horse, dismounts.
 *
 * The ground under a horse is sampled (terrainMath surfaceAt, ~10 µs) only once it has moved a little, so a herd at
 * peace costs nothing; its pitch follows the slope along the way it went (two samples apart). The way ahead is
 * looked at five times a second. A moving horse's body (three circles, chest to quarters) is pushed out of the
 * trees and boulders (forest/collision.ts); running into one stops it.
 */

import { HORSE } from '../constants';
import { surfaceAt } from '../terrainMath';
import { trunksNear, type Trunk } from '../forest/collision';
import { PLAYER } from '../constants';
import type { HeroState } from '../hero/locomotion';
import { herdsNear, type WildHorse } from './herds';
import { MAT, jointPoint, poseHorse, subtree, turnJoints, type HorseModel } from './model';

type HorseMind = 'graze' | 'idle' | 'wander' | 'alert' | 'flee' | 'lie' | 'sleep' | 'rise' | 'come' | 'struggle' | 'ridden' | 'die' | 'vanish';

export type Horse = WildHorse & {
  y: number; pitch: number;
  speed: number;              // along its heading (m/s; < 0 backing)
  vy: number; air: boolean;   // a leap
  mind: HorseMind;
  timer: number;              // s left in this mind
  tx: number; tz: number;     // where it is going (wander, flee, come)
  owned: boolean;
  bond: number;               // 0..1: grows as it is ridden
  // the ground, sampled where it was last (gx, gz); the slope along the way it went; s until the way ahead is looked at again, and what was seen
  ground: number; water: number; gx: number; gz: number; slope: number;
  look: number; blocked: boolean;
  // animation: the clip and its time, the clip faded out and its time, the fade's weight left on it
  clip: string; t: number; rate: number;
  prev: string; pt: number; fade: number; fadeFor: number;
  // grazing's phases (eat, look up and round, a few steps to fresh grass) and s left in this one
  sub: 'eat' | 'look' | 'step'; subT: number;
  // the head turned off the clip (rad, toward the right) and where it is turning to
  head: number; headTo: number;
  // a detour round something solid: s left on it, and the heading it takes
  avoid: number; avoidYaw: number;
  spin: number;               // how fast it is turning (rad/s)
  hp: number;                 // health (HORSE.hp full; 0: dying)
  hitSwing: number;           // the hero's swing that last struck it (one blow a swing)
  rockHit: object | null;     // the thrown rock that last struck it (one blow a throw)
  flash: number;              // s left of the flash of a blow
};

type HorseInput = {
  forward: number; strafe: number; camYaw: number;
  sprint: boolean; jump: boolean; sneak: boolean;
  action: boolean; whistle: boolean;
};

/**
 * One frame of horses for the renderer (gpu/horses.ts): HORSE_STRIDE floats each, camera-relative; the last `fading`
 * of the `count` are vanishing (drawn dissolving).
 */
export type HorseFrame = { data: Float32Array; u32: Uint32Array; count: number; fading: number };

/** Horses drawn at most (the nearest). */
export const MAX_DRAWN = 24;
/** The joints the renderer expects (scripts/horse/build.py's deforming joints). */
export const HORSE_JOINTS = 44;
/** Floats a horse: a header vec4 (its id and coat seed as u32, how far it has vanished 0..1, a blow's flash 0..1), then each
 *  joint's 3×4 matrix (camera-relative world). */
export const HORSE_STRIDE = 4 + HORSE_JOINTS * MAT;

const TAU = Math.PI * 2;
const FADE = 0.25;
// clips' natural speeds (m/s): the playback rate follows the ground speed
const NATURAL: Record<string, number> = { Walk: 1.9, Trot: 4.5, Run: 11 };

const wrap = (a: number) => a - TAU * Math.round(a / TAU);
const approach = (rate: number, dt: number) => 1 - Math.exp(-rate * dt);
const rand = Math.random;

export function createHorseInput(): HorseInput {
  return { forward: 0, strafe: 0, camYaw: 0, sprint: false, jump: false, sneak: false, action: false, whistle: false };
}

export function createHorseFrame(): HorseFrame {
  const data = new Float32Array(MAX_DRAWN * HORSE_STRIDE);
  return { data, u32: new Uint32Array(data.buffer), count: 0, fading: 0 };
}

export class Horses {
  model: HorseModel | null = null;
  list: Horse[] = [];
  seed = NaN;
  /** The horse under the hero, if any. */
  riding: Horse | null = null;
  gait = 0;
  spurs = 0;
  /** s since a spur was last used, s of gallop left. */
  private sinceSpur = 0;
  private gallopLeft = 0;
  private gaitHold = 0;
  /** s left of the struggle (taming); NaN: not struggling. */
  tameLeft = NaN;
  /** Set on a frame something happened the engine says: 'tamed', 'thrown', 'mounted', 'dismounted', 'whistle'. */
  event: string | null = null;
  /** The Action press was taken this frame (the combat world must not also act on it). */
  tookAction = false;
  /** Horses that began to vanish this frame (the engine puffs their motes). */
  vanished: Horse[] = [];
  /** The horses killed (by id): they do not come back. */
  private dead = new Set<number>();
  private held = { action: false, sprint: false, jump: false, whistle: false };
  private wild: WildHorse[] = [];
  private listedAt = [NaN, NaN];
  private listedT = 0;
  private time = 0;
  /** s the hero still counts as sneaking (a step off a bump is not a jump out of hiding). */
  private sneakT = 0;
  private pose: Float32Array;
  /** The neck and all on it (the head turns it: Horse.head). */
  private neck: number[] = [];
  private P = new Float64Array(3);

  constructor() {
    this.pose = new Float32Array(HORSE_JOINTS * MAT);
  }

  setModel(m: HorseModel): void {
    if (m.joints !== HORSE_JOINTS) throw new Error(`horse.bin: ${m.joints} joints, expected ${HORSE_JOINTS}`);
    this.model = m;
    this.neck = subtree(m, m.names.indexOf('spine_5'));
  }

  /** A new world or seed: no horses (the next step lists them afresh). */
  reset(seed: number): void {
    this.seed = seed;
    this.list.length = 0;
    this.riding = null;
    this.tameLeft = NaN;
    this.listedAt[0] = NaN;
    this.dead.clear();
  }

  /** A blow of `dmg` from (fromX, fromZ): it bolts (with its herd), or, its health gone, dies. */
  hurt(H: Horse, dmg: number, fromX: number, fromZ: number): void {
    if (H.hp <= 0 || H === this.riding) return;
    H.hp = Math.max(0, H.hp - dmg);
    H.flash = FLASH;
    if (H.hp > 0) { this.bolt(H, fromX, fromZ); return; }
    H.mind = 'die';
    H.timer = HORSE.dieFor;
    H.avoid = 0;
    this.play(H, 'Death', 1, 0.15);
    this.dead.add(H.id);
  }

  /** The horse the hero could mount now (in reach, at peace or only alert), or null. */
  mountable(h: HeroState): Horse | null {
    if (this.riding || h.move !== 'ground') return null;
    let best: Horse | null = null, bd: number = HORSE.mountReach;
    for (const H of this.list) {
      if (H.mind === 'flee' || H.mind === 'sleep' || H.mind === 'lie' || H.mind === 'rise' || H.air || H.hp <= 0) continue;
      const d = Math.hypot(H.x - h.x, H.z - h.z);
      if (d < bd && Math.abs(H.y - (h.y - PLAYER.groundBias)) < 1.5) { bd = d; best = H; }
    }
    return best;
  }

  /** What the HUD's prompt says about horses, or null. */
  prompt(h: HeroState): string | null {
    if (this.riding) {
      if (this.tameLeft === this.tameLeft) return 'E|Soothe (press repeatedly)';
      return `E|Dismount  ·  Shift|Spur ${'●'.repeat(Math.floor(this.spurs))}${'○'.repeat(this.riding.spurs - Math.floor(this.spurs))}`;
    }
    const m = this.mountable(h);
    return m ? `E|${m.owned ? 'Mount' : 'Mount (wild)'}` : null;
  }

  /**
   * One tick: list the herds round the hero, run every horse's mind, carry the rider. Call before the hero's own
   * step (combat world); while riding, the hero's locomotion stands aside (HeroState.riding) and seat() places it.
   */
  step(dt: number, h: HeroState, inp: HorseInput, hour: number): void {
    this.event = null;
    this.tookAction = false;
    this.vanished.length = 0;
    this.time += dt;
    const press = (k: 'action' | 'sprint' | 'jump' | 'whistle', v: boolean) => { const p = v && !this.held[k]; this.held[k] = v; return p; };
    const act = press('action', inp.action), spur = press('sprint', inp.sprint), jump = press('jump', inp.jump), whistle = press('whistle', inp.whistle);
    if (!this.model) return;
    // (placed elsewhere meanwhile, a respawn: off the horse)
    if (this.riding && !h.riding) { this.riding.mind = 'idle'; this.riding.timer = 2; this.riding = null; this.tameLeft = NaN; }
    this.listHerds(h);
    const night = hour >= HORSE.nightFrom || hour < HORSE.nightTo;

    // mount, soothe, dismount
    if (act) {
      if (this.riding) {
        this.tookAction = true;
        if (this.tameLeft === this.tameLeft) this.tameLeft -= HORSE.soothe;
        else this.dismount(h, false);
      } else {
        const m = this.mountable(h);
        if (m) { this.tookAction = true; this.mount(h, m); }
      }
    }
    if (whistle) this.whistle(h);

    const hx = h.x, hz = h.z, hv = Math.hypot(h.vx, h.vz);
    this.sneakT = h.sneaking ? 0.5 : Math.max(0, this.sneakT - dt);
    const sneaking = this.sneakT > 0 && hv < 3;
    const senses = sneaking ? 0.35 : h.sprinting ? 1.4 : hv < 0.3 ? 0.7 : 1;
    for (const H of this.list) {
      if (H === this.riding) { this.ride(H, h, inp, spur, jump, dt); continue; }
      const dx = H.x - hx, dz = H.z - hz, d = Math.hypot(dx, dz);
      if (d > HORSE.despawnRange * 1.5 && H.mind !== 'come') continue; // (the hero's own, far away: left standing)
      this.think(H, d, dx, dz, senses, sneaking, night, h, dt);
      this.move(H, dt);
    }
    for (const H of this.list) this.animate(H, dt);
    // the vanished are gone
    if (this.list.some((H) => H.mind === 'vanish' && H.timer <= 0)) this.list = this.list.filter((H) => H.mind !== 'vanish' || H.timer > 0);
  }

  /** While riding: the hero on the saddle (after the hero's own step), facing the way the horse does. */
  seat(h: HeroState): void {
    const H = this.riding;
    if (!H || !this.model) return;
    const m = this.model, P = this.P;
    poseHorse(m, m.clip[H.clip], H.t, H.fade > 0 ? m.clip[H.prev] : null, H.pt, H.fade, this.pose);
    jointPoint(this.pose, 0, m.saddleJoint, m.saddle[0], m.saddle[1], m.saddle[2], P);
    const sy = Math.sin(H.yaw), cy = Math.cos(H.yaw), sp = Math.sin(H.pitch), cp = Math.cos(H.pitch);
    // (the horse's frame: its right, up and forward turned by its pitch and yaw; see pack)
    const x = cy * P[0] - sy * sp * P[1] + sy * cp * P[2];
    const y = cp * P[1] + sp * P[2];
    const z = -sy * P[0] - cy * sp * P[1] + cy * cp * P[2];
    h.x = H.x + x; h.z = H.z + z;
    h.y = H.y + y + PLAYER.groundBias - RIDE_PELVIS + HORSE.seatDrop;
    h.yaw = H.yaw;
    h.vx = Math.sin(H.yaw) * H.speed; h.vz = Math.cos(H.yaw) * H.speed; h.vy = 0;
    h.ground = H.ground; h.water = H.water;
  }

  /**
   * The hero on foot against the horses' bodies (after its own step): pushed out of a horse standing or walking,
   * and the throw (m/s, into out; true) when a horse faster than HORSE.trample runs into it.
   */
  collideHero(h: HeroState, out: Float64Array): boolean {
    if (this.riding || (h.move !== 'ground' && h.move !== 'air')) return false;
    const feet = h.y - PLAYER.groundBias;
    for (const H of this.list) {
      if (H.hp <= 0 || Math.abs(H.x - h.x) > 4 || Math.abs(H.z - h.z) > 4 || feet < H.y - 0.4 || feet > H.y + 1.5) continue;
      const fx = Math.sin(H.yaw), fz = Math.cos(H.yaw);
      for (const k of BODY) {
        const cx = H.x + fx * k, cz = H.z + fz * k;
        let dx = h.x - cx, dz = h.z - cz, d = Math.hypot(dx, dz);
        const R = BODY_R + HERO_R;
        if (d >= R) continue;
        if (d < 1e-6) { dx = fz; dz = -fx; d = 1; } // (dead centre: out to its side)
        dx /= d; dz /= d;
        const v = Math.abs(H.speed);
        if (v > HORSE.trample) {
          // run down: flung along the horse's way and off its body, up off the ground
          out[0] = fx * H.speed * 0.8 + dx * (2 + 0.25 * v);
          out[1] = 3 + 0.3 * v;
          out[2] = fz * H.speed * 0.8 + dz * (2 + 0.25 * v);
          H.speed *= 0.6;
          return true;
        }
        h.x += dx * (R - d); h.z += dz * (R - d);
        // (no speed into the horse: the hero slides along it)
        const into = h.vx * dx + h.vz * dz;
        if (into < 0) { h.vx -= into * dx; h.vz -= into * dz; }
      }
    }
    return false;
  }

  /** The hero thrown off or stepping down: beside the horse, on its left. */
  dismount(h: HeroState, thrown: boolean): void {
    const H = this.riding;
    if (!H) return;
    this.riding = null;
    h.riding = false;
    this.tameLeft = NaN;
    const sy = Math.sin(H.yaw), cy = Math.cos(H.yaw);
    h.x = H.x - cy * 1.1; h.z = H.z + sy * 1.1;
    h.y = H.y + PLAYER.groundBias + 0.6;
    h.move = 'air';
    h.vx = -cy * (thrown ? 4 : 1.2) + Math.sin(H.yaw) * H.speed * 0.5;
    h.vz = sy * (thrown ? 4 : 1.2) + Math.cos(H.yaw) * H.speed * 0.5;
    h.vy = thrown ? 5 : 2.5;
    H.speed = Math.min(H.speed, HORSE.gaitSpeed[1]);
    if (thrown) this.bolt(H, h.x, h.z);
    else { H.mind = 'idle'; H.timer = 3 + 4 * rand(); }
    this.event = thrown ? 'thrown' : 'dismounted';
  }

  private mount(h: HeroState, H: Horse): void {
    this.riding = H;
    h.riding = true;
    h.sprinting = h.sneaking = h.surfing = false;
    H.mind = H.owned ? 'ridden' : 'struggle';
    H.speed = 0;
    this.gait = 0;
    this.gallopLeft = 0;
    this.spurs = H.spurs;
    this.sinceSpur = 99;
    this.tameLeft = H.owned ? NaN : HORSE.tameTime + HORSE.tameTemper * H.temper;
    if (!H.owned) this.play(H, 'Rear', 1);
    this.event = 'mounted';
  }

  private whistle(h: HeroState): void {
    this.event = 'whistle';
    let best: Horse | null = null, bd: number = HORSE.whistleRange;
    for (const H of this.list) {
      if (!H.owned || H === this.riding || H.hp <= 0) continue;
      const d = Math.hypot(H.x - h.x, H.z - h.z);
      if (d < bd) { bd = d; best = H; }
    }
    if (!best) return;
    if (best.mind === 'sleep' || best.mind === 'lie') { best.mind = 'rise'; best.timer = 0; this.play(best, 'lay_to_idle', 1); }
    else { best.mind = 'come'; best.timer = 30; }
  }

  /** Every horse of H's herd near it bolts away from (fx, fz). */
  private bolt(H: Horse, fx: number, fz: number): void {
    for (const O of this.list) {
      if (O === this.riding || O.hp <= 0 || (O !== H && (O.owned || O.herd !== H.herd || Math.hypot(O.x - H.x, O.z - H.z) > 40))) continue;
      const a = Math.atan2(O.x - fx, O.z - fz) + (rand() - 0.5) * 0.6;
      const [lo, hi] = HORSE.fleeFor;
      O.mind = 'flee';
      O.timer = lo + (hi - lo) * rand();
      O.tx = O.x + Math.sin(a) * 200; O.tz = O.z + Math.cos(a) * 200;
    }
  }

  private think(H: Horse, d: number, dx: number, dz: number, senses: number, sneaking: boolean, night: boolean, h: HeroState, dt: number): void {
    H.timer -= dt;
    // (sneaking, the hero can come right up to a horse: it only looks up)
    const notice = HORSE.notice * senses, bolt = sneaking ? 0 : HORSE.bolt * senses;
    const wild = !H.owned;
    switch (H.mind) {
      case 'graze': case 'idle': case 'wander': {
        if (wild && d < bolt) { this.bolt(H, h.x, h.z); return; }
        if (wild && d < notice) { H.mind = 'alert'; H.timer = 2 + rand(); return; }
        if (night) { H.mind = 'lie'; H.timer = this.model!.clip.lay_to_idle.duration; this.play(H, 'lay_to_idle', -1); return; }
        if (H.mind === 'wander' && Math.hypot(H.tx - H.x, H.tz - H.z) < 1.5) H.timer = 0;
        if (H.mind === 'graze') this.grazing(H, dt);
        // (standing about, it looks round now and then)
        else if (H.mind === 'idle' && rand() < dt * 0.5) H.headTo = (rand() - 0.5) * 1.1;
        if (H.timer > 0) return;
        const r = rand();
        if (r < 0.45) { H.mind = 'graze'; H.timer = 10 + 14 * rand(); H.sub = 'eat'; H.subT = 3 + 4 * rand(); H.rate = 0.8 + 0.4 * rand(); }
        else if (r < 0.7) { H.mind = 'idle'; H.timer = 3 + 5 * rand(); }
        else {
          const a = rand() * TAU, w = HORSE.wander * Math.sqrt(rand());
          H.mind = 'wander'; H.timer = 15;
          H.tx = H.ax + Math.cos(a) * w; H.tz = H.az + Math.sin(a) * w;
        }
        return;
      }
      case 'alert': {
        if (d < bolt) { this.bolt(H, h.x, h.z); return; }
        // head up, turning to face the hero
        H.yaw = wrap(H.yaw + wrap(Math.atan2(-dx, -dz) - H.yaw) * approach(2.5, dt));
        if (H.timer <= 0 || d > notice * 1.3) { H.mind = 'idle'; H.timer = 1 + 2 * rand(); }
        return;
      }
      case 'flee':
        if (H.timer <= 0) {
          H.mind = 'idle'; H.timer = 2 + 3 * rand();
          // (a wild herd settles where it fled to)
          if (wild) { H.ax = H.x; H.az = H.z; }
        }
        return;
      case 'lie':
        if (H.timer <= 0) { H.mind = 'sleep'; this.play(H, 'Sleep', 1); }
        return;
      case 'sleep':
        if (wild && d < bolt * 0.6) { H.mind = 'rise'; H.timer = this.model!.clip.lay_to_idle.duration; this.play(H, 'lay_to_idle', 1); return; }
        if (!night) { H.mind = 'rise'; H.timer = this.model!.clip.lay_to_idle.duration; this.play(H, 'lay_to_idle', 1); }
        return;
      case 'rise':
        if (H.timer <= 0) {
          if (wild && d < notice) this.bolt(H, h.x, h.z);
          else { H.mind = 'idle'; H.timer = 2; }
        }
        return;
      case 'die':
        if (H.timer <= 0) { H.mind = 'vanish'; H.timer = HORSE.vanishFor; this.vanished.push(H); }
        return;
      case 'vanish':
        return;
      case 'come':
        H.tx = h.x; H.tz = h.z;
        if (d < 3.5 || H.timer <= 0) { H.mind = 'idle'; H.timer = 4 + 4 * rand(); H.ax = H.x; H.az = H.z; }
        return;
      default:
        return;
    }
  }

  /**
   * Grazing in phases: eating (the head down, cropping side to side) for a few seconds, then the head up looking
   * round (once or twice) or a few slow steps to fresh grass, and eating again.
   */
  private grazing(H: Horse, dt: number): void {
    if ((H.subT -= dt) > 0) {
      if (H.sub === 'look' && rand() < dt * 0.7) H.headTo = (rand() - 0.5) * 1.3;
      return;
    }
    if (H.sub !== 'eat') { H.sub = 'eat'; H.subT = 3 + 5 * rand(); H.rate = 0.8 + 0.4 * rand(); return; }
    if (rand() < 0.55) {
      H.sub = 'look'; H.subT = 1.8 + 2.4 * rand(); H.headTo = (rand() - 0.5) * 1.3;
    } else {
      const a = H.yaw + (rand() - 0.5) * 1.4;
      H.sub = 'step'; H.subT = 1.2 + 1.5 * rand();
      H.tx = H.x + Math.sin(a) * 3; H.tz = H.z + Math.cos(a) * 3;
    }
  }

  /**
   * Something solid in the way (a trunk, a boulder, a drop, a slope too steep, deep water): a detour, turning away
   * from it (side −1 left, 1 right) and going that way a moment before heading for where it was going again.
   */
  private detour(H: Horse, side: number): void {
    // (still on one: keep turning the same way, further)
    if (H.avoid > 0) side = Math.sign(wrap(H.avoidYaw - H.yaw)) || side;
    H.avoidYaw = wrap(H.yaw + side * (1.1 + 0.8 * rand()));
    H.avoid = 1.3 + 1.0 * rand();
    H.speed *= 0.3;
    H.look = 0; // (the way ahead looked at again at once)
    if (H.mind === 'wander' || (H.mind === 'graze' && H.sub === 'step')) {
      // a new spot that way
      const w = H.mind === 'wander' ? 8 + 8 * rand() : 3;
      H.tx = H.x + Math.sin(H.avoidYaw) * w; H.tz = H.z + Math.cos(H.avoidYaw) * w;
    }
  }

  /** A horse not ridden: toward its target at its mind's pace, turning as it goes; on the ground. */
  private move(H: Horse, dt: number): void {
    let want = 0;
    if (H.mind === 'wander') want = HORSE.gaitSpeed[1];
    else if (H.mind === 'graze' && H.sub === 'step') want = 0.6;
    else if (H.mind === 'flee') want = H.timer > 2 ? HORSE.fleeSpeed * H.gallop : HORSE.gaitSpeed[2];
    else if (H.mind === 'come') want = Math.hypot(H.tx - H.x, H.tz - H.z) > 20 ? HORSE.gaitSpeed[4] * H.gallop : HORSE.gaitSpeed[2];
    const yaw0 = H.yaw;
    if (want > 0) {
      let a = Math.atan2(H.tx - H.x, H.tz - H.z);
      // on a detour: its heading instead, at no more than a trot (a walk at least)
      if (H.avoid > 0) { H.avoid -= dt; a = H.avoidYaw; want = Math.max(Math.min(want, HORSE.gaitSpeed[2]), HORSE.gaitSpeed[1]); }
      const turn = HORSE.gaitTurn[want > 6 ? 3 : 1] * 1.4;
      H.yaw = wrap(H.yaw + Math.max(-turn * dt, Math.min(turn * dt, wrap(a - H.yaw))));
      // (a drop, a wall or deep water ahead: it stops going that way and turns off; still facing it, it turns on)
      if (this.blocked(H, dt)) { want = 0; this.detour(H, H.id & 1 ? 1 : -1); }
    } else H.avoid = 0;
    H.spin = wrap(H.yaw - yaw0) / Math.max(dt, 1e-4);
    H.speed += (want - H.speed) * approach(want > H.speed ? 1.6 : 3.5, dt);
    if (Math.abs(H.speed) < 0.02 && want === 0) H.speed = 0;
    this.advance(H, dt);
  }

  /** The ridden horse: the rider's keys, spurs and leaps, the struggle while being tamed. */
  private ride(H: Horse, h: HeroState, inp: HorseInput, spur: boolean, jump: boolean, dt: number): void {
    if (this.tameLeft === this.tameLeft) {
      // the struggle: rearing and bucking in place, the rider's stamina draining until tamed or thrown
      H.speed = 0;
      this.tameLeft -= dt;
      h.stamina = Math.max(0, h.stamina - HORSE.tameDrain * dt);
      h.rest = 0;
      const c = this.model!.clip[H.clip];
      if (H.t >= c.duration - 0.05) this.play(H, ['Rear', 'Kick', 'Head_But'][Math.floor(rand() * 3)], 1);
      if (this.tameLeft <= 0) {
        this.tameLeft = NaN;
        H.owned = true; H.bond = 0.2; H.mind = 'ridden';
        this.event = 'tamed';
      } else if (h.stamina <= 0) {
        h.exhausted = true;
        this.dismount(h, true);
      }
      this.advance(H, dt);
      return;
    }
    H.mind = 'ridden';
    H.bond = Math.min(1, H.bond + dt / 600);
    // stamina comes back on horseback
    h.rest += dt;
    h.stamina = Math.min(1, h.stamina + 0.34 * dt);
    if (h.stamina >= 1) h.exhausted = false;
    // spurs come back with rest
    this.sinceSpur += dt;
    if (this.spurs < H.spurs && this.sinceSpur > HORSE.spurRest) {
      this.spurs = Math.min(H.spurs, this.spurs + dt / HORSE.spurRegen);
    }
    // the way the keys ask (camera-relative), and how far it is off the horse's heading
    let mag = 0, a = H.yaw;
    if (inp.forward !== 0 || inp.strafe !== 0) {
      const sn = Math.sin(inp.camYaw), cs = Math.cos(inp.camYaw);
      const wx = sn * inp.forward + cs * inp.strafe, wz = cs * inp.forward - sn * inp.strafe;
      mag = Math.min(1, Math.hypot(wx, wz));
      a = Math.atan2(wx, wz);
    }
    const off = wrap(a - H.yaw);
    const back = mag > 0.3 && Math.abs(off) > 2.4;
    this.gaitHold -= dt;
    if (back) {
      // a sharp turn back: braking a gait at a time; standing, backing up
      if (this.gait > 0 && this.gaitHold <= 0) { this.gait--; this.gaitHold = 0.35; }
    } else if (mag > 0.3) {
      if (this.gait === 0) this.gait = 1;
      const turn = HORSE.gaitTurn[this.gait];
      H.yaw = wrap(H.yaw + Math.max(-turn * dt, Math.min(turn * dt, off * 3 * dt * turn)));
    } else if (this.gait > 0 && this.gaitHold <= 0) { this.gait--; this.gaitHold = 0.6; }
    if (spur && this.spurs >= 1) {
      this.spurs = Math.floor(this.spurs) - 1;
      this.sinceSpur = 0;
      this.gait = Math.min(4, Math.max(this.gait, 1) + 1);
      if (this.gait === 4) this.gallopLeft = HORSE.gallopFor;
      this.gaitHold = 0.6;
    }
    if (this.gait === 4 && (this.gallopLeft -= dt) <= 0) this.gait = 3;
    let want = HORSE.gaitSpeed[this.gait] * (this.gait === 4 ? H.gallop : 1);
    if (back && this.gait === 0) want = -HORSE.backSpeed;
    if (jump && !H.air && this.gait >= 2) { H.air = true; H.vy = HORSE.jumpSpeed; }
    if (want > 0 && !H.air && this.blocked(H, dt)) { want = 0; this.gait = 0; }
    H.speed += (want - H.speed) * approach(want > H.speed ? HORSE.accel : HORSE.brake, dt) * (H.air ? 0.2 : 1);
    if (Math.abs(H.speed) < 0.02 && want === 0) H.speed = 0;
    this.advance(H, dt);
  }

  /** A drop, a wall too steep or deep water a few metres ahead (looked at every LOOK_EVERY s). */
  private blocked(H: Horse, dt: number): boolean {
    if ((H.look -= dt) > 0) return H.blocked;
    H.look = LOOK_EVERY;
    const look = 2.5 + Math.max(0, H.speed) * 0.35;
    const x = H.x + Math.sin(H.yaw) * look, z = H.z + Math.cos(H.yaw) * look;
    const s = surfaceAt(x, z, this.seed, SURF);
    H.blocked = s[0] < H.ground - HORSE.cliff || (s[0] - H.ground) / look > HORSE.steep || s[1] - s[0] > HORSE.deep;
    return H.blocked;
  }

  /** Along its heading by its speed; the ground under it (resampled once it moved), its pitch, a leap's arc. */
  private advance(H: Horse, dt: number): void {
    H.x += Math.sin(H.yaw) * H.speed * dt;
    H.z += Math.cos(H.yaw) * H.speed * dt;
    if (H.speed !== 0) this.collide(H);
    const moved = (H.x - H.gx) * Math.sin(H.yaw) + (H.z - H.gz) * Math.cos(H.yaw);
    if (Math.hypot(H.x - H.gx, H.z - H.gz) > RESAMPLE) {
      const s = surfaceAt(H.x, H.z, this.seed, SURF);
      // the slope along its heading, from the ground where it was to where it is
      if (Math.abs(moved) > 0.5 * RESAMPLE) H.slope = Math.max(-1.5, Math.min(1.5, (s[0] - H.ground) / moved));
      H.ground = s[0]; H.water = s[1];
      H.gx = H.x; H.gz = H.z;
    }
    // (standing in water it is waded through: the hooves stay on the bed)
    if (H.air) {
      H.vy -= HORSE.gravity * dt;
      H.y += H.vy * dt;
      if (H.y <= H.ground && H.vy < 0) { H.y = H.ground; H.air = false; H.vy = 0; }
    } else H.y += (H.ground - H.y) * approach(18, dt);
    H.pitch += (Math.atan(H.slope) * 0.8 - H.pitch) * approach(8, dt);
  }

  /** Out of the trunks and boulders round it (the body: circles at the chest, the girth and the quarters); a hit stops it. */
  private collide(H: Horse): void {
    const fx = Math.sin(H.yaw), fz = Math.cos(H.yaw);
    let hit = false, side = 1;
    for (const t of trunksNear(H.x, H.z, this.seed, TRUNKS)) {
      if (t.y + t.top < H.y + 0.3 || t.y > H.y + 1.2) continue; // (low enough to step over, or standing above it)
      for (const k of BODY) {
        const cx = H.x + fx * k, cz = H.z + fz * k;
        const dx = cx - t.x, dz = cz - t.z, d = Math.hypot(dx, dz), R = t.r + BODY_R;
        if (d >= R || d < 1e-6) continue;
        H.x += (dx / d) * (R - d); H.z += (dz / d) * (R - d);
        // (head on: it stops; a glancing touch slides it along)
        if (Math.abs(dx * fx + dz * fz) / d > 0.35 && Math.sign(k || 1) === Math.sign(H.speed)) {
          hit = true;
          // (the trunk on its right: away to the left, and the other way round)
          side = (t.x - H.x) * fz - (t.z - H.z) * fx > 0 ? -1 : 1;
        }
      }
    }
    if (hit) {
      if (H === this.riding) { H.speed = 0; this.gait = 0; this.gallopLeft = 0; }
      else this.detour(H, side);
    }
  }

  /** The clip a horse's mind and pace call for (a fade from the last), and its time. */
  private animate(H: Horse, dt: number): void {
    const m = this.model!;
    let want = H.clip, rate = H.rate;
    const v = Math.abs(H.speed);
    switch (H.mind) {
      case 'graze':
        if (H.sub === 'eat') { want = 'Eating'; rate = H.rate; }
        else if (H.sub === 'look' || v < 0.15) { want = 'Idle'; rate = 1; }
        else { want = 'Walk'; rate = Math.max(0.6, v / NATURAL.Walk); }
        break;
      case 'idle': case 'alert': want = v > 0.3 ? 'Walk' : 'Idle'; rate = 1; break;
      case 'lie': case 'rise': case 'sleep': case 'struggle': case 'die': case 'vanish': break; // (set by play)
      default:
        // (turning on the spot, it steps round rather than sliding)
        want = v < 0.25 ? (Math.abs(H.spin) > 0.4 && H !== this.riding ? 'Walk' : 'Idle') : v < 3.0 ? 'Walk' : v < 6.5 ? 'Trot' : 'Run';
        rate = want === 'Idle' ? 1 : v < 0.25 ? 0.6 : Math.max(0.6, Math.min(1.5, v / NATURAL[want])) * Math.sign(H.speed || 1);
    }
    // (grazing, the head goes down and comes up slowly)
    if (want !== H.clip) this.play(H, want, rate, H.mind === 'graze' ? 0.7 : FADE);
    H.rate = rate;
    H.t += dt * H.rate;
    const c = m.clip[H.clip];
    if (!c.loop) H.t = Math.max(0, Math.min(c.duration, H.t));
    if (H.fade > 0) { H.pt += dt; H.fade = Math.max(0, H.fade - dt / H.fadeFor); }
    H.flash = Math.max(0, H.flash - dt);
    // the head off the clip: cropping side to side while eating, looking round with it up, else straight
    const to = H.mind === 'graze' && H.sub === 'eat' ? 0.2 * Math.sin(this.time * 0.9 + (H.id & 255))
      : (H.mind === 'graze' && H.sub === 'look') || H.mind === 'idle' ? H.headTo : 0;
    H.head += (to - H.head) * approach(H.mind === 'graze' && H.sub === 'eat' ? 6 : 2.2, dt);
  }

  /** Start clip `name` (rate −1: backwards from its end), fading from the one playing over `fade` s. */
  private play(H: Horse, name: string, rate: number, fade = FADE): void {
    if (H.clip !== name) { H.prev = H.clip; H.pt = H.t; H.fade = 1; H.fadeFor = fade; }
    H.clip = name;
    H.rate = rate;
    H.t = rate < 0 && !this.model!.clip[name].loop ? this.model!.clip[name].duration : 0;
  }

  /** List the herds again when the hero has moved (or now and then); wild horses come and go with range. */
  private listHerds(h: HeroState): void {
    if (Math.hypot(h.x - this.listedAt[0], h.z - this.listedAt[1]) < 40 && this.time - this.listedT < 4) return;
    this.listedAt[0] = h.x; this.listedAt[1] = h.z; this.listedT = this.time;
    // the wild ones beyond range go (the hero's own stay)
    this.list = this.list.filter((H) => H.owned || H === this.riding || Math.hypot(H.x - h.x, H.z - h.z) < HORSE.despawnRange);
    const have = new Set(this.list.map((H) => H.id));
    for (const w of herdsNear(h.x, h.z, this.seed, HORSE.spawnRange, this.wild)) {
      if (have.has(w.id) || this.dead.has(w.id) || this.list.length >= HORSE.maxLive) continue;
      if (Math.hypot(w.x - h.x, w.z - h.z) > HORSE.spawnRange) continue;
      this.list.push(this.born(w));
    }
  }

  private born(w: WildHorse): Horse {
    const s = surfaceAt(w.x, w.z, this.seed, SURF);
    return {
      ...w, y: s[0], pitch: 0, speed: 0, vy: 0, air: false,
      mind: 'graze', timer: 2 + 10 * rand(), tx: w.x, tz: w.z, owned: false, bond: 0,
      ground: s[0], water: s[1], gx: w.x, gz: w.z, slope: 0, look: 0, blocked: false,
      clip: 'Eating', t: rand() * 2, rate: 1, prev: 'Eating', pt: 0, fade: 0, fadeFor: FADE,
      sub: 'eat', subT: 2 + 5 * rand(), head: 0, headTo: 0, avoid: 0, avoidYaw: 0, spin: 0,
      hp: HORSE.hp, hitSwing: -1, rockHit: null, flash: 0,
    };
  }

  /**
   * The nearest MAX_DRAWN horses for the renderer, camera-relative to (cx, cy, cz): each its coat and its joints'
   * matrices (the pose's, carried by the horse's pitch, yaw and place).
   */
  pack(cx: number, cy: number, cz: number, out: HorseFrame): HorseFrame {
    out.count = 0;
    out.fading = 0;
    const m = this.model;
    if (!m) return out;
    const L = this.list;
    if (L.length > MAX_DRAWN) L.sort((a, b) => Math.hypot(a.x - cx, a.z - cz) - Math.hypot(b.x - cx, b.z - cz));
    const n = Math.min(L.length, MAX_DRAWN), S = this.pose, D = out.data, U = out.u32;
    // (the vanishing last: they are drawn apart, dissolving)
    const order = DRAW_ORDER;
    order.length = 0;
    for (let k = 0; k < n; k++) if (L[k].mind !== 'vanish') order.push(L[k]);
    for (let k = 0; k < n; k++) if (L[k].mind === 'vanish') { order.push(L[k]); out.fading++; }
    for (let k = 0; k < n; k++) {
      const H = order[k];
      poseHorse(m, m.clip[H.clip], H.t, H.fade > 0 ? m.clip[H.prev] : null, H.pt, H.fade, S);
      if (Math.abs(H.head) > 1e-3) turnJoints(m, S, 0, this.neck, H.head, 0);
      const o = k * HORSE_STRIDE;
      U[o] = H.id; U[o + 1] = H.coat;
      D[o + 2] = H.mind === 'vanish' ? Math.min(1, 1 - H.timer / HORSE.vanishFor) : 0;
      D[o + 3] = H.flash / FLASH;
      // the horse's frame in the world: right (cy, 0, −sy); up and forward turned by the pitch, then the yaw
      const sy = Math.sin(H.yaw), cyw = Math.cos(H.yaw), sp = Math.sin(H.pitch), cp = Math.cos(H.pitch);
      const r00 = cyw, r01 = -sy * sp, r02 = sy * cp;
      const r10 = 0, r11 = cp, r12 = sp;
      const r20 = -sy, r21 = -cyw * sp, r22 = cyw * cp;
      const tx = H.x - cx, ty = H.y - cy, tz = H.z - cz;
      for (let j = 0; j < HORSE_JOINTS; j++) {
        const s = j * MAT, d = o + 4 + j * MAT;
        for (let c = 0; c < 4; c++) {
          const a0 = S[s + c], a1 = S[s + 4 + c], a2 = S[s + 8 + c];
          D[d + c] = r00 * a0 + r01 * a1 + r02 * a2;
          D[d + 4 + c] = r10 * a0 + r11 * a1 + r12 * a2;
          D[d + 8 + c] = r20 * a0 + r21 * a1 + r22 * a2;
        }
        D[d + 3] += tx; D[d + 7] += ty; D[d + 11] += tz;
      }
      out.count++;
    }
    return out;
  }
}

/** The hero's pelvis over its root in the riding pose (rig.ts poseRide). */
const RIDE_PELVIS = 0.9;

const SURF: [number, number] = [0, 0];
const DRAW_ORDER: Horse[] = [];
/** s a blow's flash lasts. */
const FLASH = 0.15;
const TRUNKS: Trunk[] = [];
/** The body against the trees: circles BODY_R across at these places along it (m from the girth). */
const BODY = [0.85, 0, -0.85];
const BODY_R = 0.42;
/** The hero's body against a horse's (m). */
const HERO_R = 0.35;
/** The ground under a horse is sampled again once it moved this far (m); the way ahead every LOOK_EVERY s. */
const RESAMPLE = 0.4;
const LOOK_EVERY = 0.2;
