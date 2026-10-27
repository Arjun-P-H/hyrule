/**
 * Combat as the renderer draws it (gpu/frame.ts CombatFrame), rebuilt every
 * frame camera-relative from the world (combat/world.ts), and the effects that
 * live a moment, kept here: sparks off a blow, a Perfect Guard's ring and
 * flash, smoke where a foe vanishes, dust, splashes, a slash's trail, the
 * glint of a charged blade, the camps' fires.
 * ---------------------------------------------------------------------------
 * Foes are packed in their unit space (combat.wgsl.ts packFoeWgsl) with their
 * shadow bodies; arrows, thrown and dropped weapons, hearts and the camps'
 * logs and stones, shields, bows and food lying about are parts (lit boxes); everything that glows, smokes or
 * streaks is a sprite. Allocation-free per frame: the lists are made once.
 */

import { HERO_SHAPE, PLAYER } from '../constants';
import { GEAR, HERO_UNIFORM, J, type HeroRig } from '../hero/rig';
import type { HeroState } from '../hero/locomotion';
import type { CombatFrame } from '../gpu/frame';
import { FOE_STRIDE, PART_BYTES, SPRITE, SPRITE_BYTES, packFoeWgsl } from '../gpu/combat.wgsl';
import { MAX_PARTS, MAX_SPRITES } from '../gpu/combatGpu';
import { packHeroUniforms } from '../hero/rig';
import { FOE_KINDS, foeJoint, type Camp } from './foes';
import { MOVES } from './moves';
import { BOWS, BOW_MODEL, MODEL, SHIELDS, SHIELD_MODEL, WEAPON, WEAPONS } from './weapons';
import type { CombatWorld } from './world';
import { FOE } from '../constants';

type Particle = {
  live: boolean;
  kind: number;          // SPRITE kind
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  gravity: number;
  drag: number;
  age: number; life: number;
  size0: number; size1: number;
  r: number; g: number; b: number; a: number; // a 0: adds light
  spin: number;
};

const MAX_PARTICLES = 384;
/** A bow lying down: its limb points along the bow (u) and out to the side (v), m. */
const BOW_U = [-0.58, -0.3, 0, 0.3, 0.58], BOW_V = [0, 0.12, 0.16, 0.12, 0];
const TRAIL = 8;
const P3 = new Float64Array(3);

/** Random numbers for the effects (they need not be deterministic, but repeatable is kinder to tests). */
let fxSeed = 1;
function rnd(): number {
  fxSeed = (fxSeed * 16807) % 2147483647;
  return (fxSeed - 1) / 2147483646;
}

export class CombatScene {
  readonly frame: CombatFrame;
  private particles: Particle[] = [];
  private foeSrc = new Float32Array(HERO_UNIFORM.length);
  // the slash's trail: the blade's base and tip (world) the last TRAIL frames, newest first; their ages
  private trail = new Float64Array(TRAIL * 6);
  private trailAge = new Float64Array(TRAIL).fill(Infinity);
  private camps: Camp[] = [];
  private time = 0;

  constructor() {
    this.frame = {
      foes: new Float32Array((FOE.max * FOE_STRIDE) / 4), nFoes: 0,
      shadows: new Float32Array(FOE.max * 9), nShadows: 0,
      parts: new Float32Array((MAX_PARTS * PART_BYTES) / 4), nParts: 0,
      sprites: new Float32Array((MAX_SPRITES * SPRITE_BYTES) / 4), nSprites: 0,
    };
    for (let k = 0; k < MAX_PARTICLES; k++) {
      this.particles.push({ live: false, kind: 0, x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, gravity: 0, drag: 0, age: 0, life: 1, size0: 0.1, size1: 0.1, r: 1, g: 1, b: 1, a: 0, spin: 0 });
    }
  }

  private emit(kind: number, x: number, y: number, z: number, vx: number, vy: number, vz: number, life: number,
    size0: number, size1: number, r: number, g: number, b: number, a: number, gravity = 0, drag = 0): void {
    let p: Particle | null = null;
    for (const q of this.particles) if (!q.live) { p = q; break; }
    if (!p) return;
    p.live = true; p.kind = kind; p.x = x; p.y = y; p.z = z; p.vx = vx; p.vy = vy; p.vz = vz;
    p.life = life; p.age = 0; p.size0 = size0; p.size1 = size1; p.r = r; p.g = g; p.b = b; p.a = a;
    p.gravity = gravity; p.drag = drag; p.spin = rnd() * 6.28;
  }

  private sparks(x: number, y: number, z: number, n: number, speed: number, r: number, g: number, b: number): void {
    for (let k = 0; k < n; k++) {
      const u = rnd() * 2 - 1, th = rnd() * 6.283, s = Math.sqrt(1 - u * u);
      const v = speed * (0.5 + rnd());
      this.emit(SPRITE.streak, x, y, z, s * Math.cos(th) * v, Math.abs(u) * v * 0.8 + 1, s * Math.sin(th) * v, 0.16 + 0.16 * rnd(), 0.012, 0.004, r, g, b, 0, 9, 1.5);
    }
  }

  private smoke(x: number, y: number, z: number, n: number, size: number, r: number, g: number, b: number, a: number, rise: number, life: number): void {
    for (let k = 0; k < n; k++) {
      const th = rnd() * 6.283, d = rnd() * size * 0.6;
      this.emit(SPRITE.puff, x + Math.cos(th) * d, y + rnd() * size * 0.4, z + Math.sin(th) * d,
        Math.cos(th) * 0.6, rise * (0.6 + 0.6 * rnd()), Math.sin(th) * 0.6, life * (0.7 + 0.5 * rnd()), size * 0.5, size * (1.1 + 0.4 * rnd()), r, g, b, a, -0.3, 0.8);
    }
  }

  /** Effects for what happened this tick (world.events), aging and moving those alive; the blade's trail. */
  update(w: CombatWorld, rig: HeroRig, dt: number): void {
    this.time += dt;
    for (let k = 0; k < w.eventCount; k++) {
      const e = w.events[k];
      switch (e.type) {
        case 'hit': case 'sneak':
          this.sparks(e.x, e.y, e.z, 9, 5, 4, 3.2, 1.8);
          this.emit(SPRITE.star, e.x, e.y, e.z, 0, 0, 0, 0.13, 0.32, 0.5, 3.5, 3.2, 2.4, 0);
          break;
        case 'crit':
          this.sparks(e.x, e.y, e.z, 14, 6, 4, 3, 1.5);
          this.emit(SPRITE.star, e.x, e.y, e.z, 0, 0, 0, 0.2, 0.5, 0.8, 4, 3.5, 2, 0);
          this.emit(SPRITE.ring, e.x, e.y, e.z, 0, 0, 0, 0.25, 0.15, 0.9, 2.5, 2, 1, 0);
          break;
        case 'parry': case 'parried':
          this.sparks(e.x, e.y, e.z, 16, 6, 2.2, 3, 4.5);
          this.emit(SPRITE.ring, e.x, e.y, e.z, 0, 0, 0, 0.3, 0.2, 1.6, 1.6, 2.6, 4.5, 0);
          this.emit(SPRITE.star, e.x, e.y, e.z, 0, 0, 0, 0.22, 0.6, 0.9, 3, 4, 6, 0);
          break;
        case 'block':
          this.sparks(e.x, e.y, e.z, 7, 3.5, 3, 2.2, 1.2);
          break;
        case 'hurt':
          this.sparks(e.x, e.y, e.z, 8, 3, 3, 0.4, 0.3);
          break;
        case 'vanish': {
          // a slain beast going: pale motes drifting up off the length of its body, a faint haze
          const fx = Math.sin(e.a), fz = Math.cos(e.a);
          for (let k = 0; k < 16; k++) {
            const t = (rnd() * 2 - 1) * 1.05, j = (rnd() - 0.5) * 0.5;
            this.emit(SPRITE.glow, e.x + fx * t + fz * j, e.y + (rnd() - 0.35) * 0.9, e.z + fz * t - fx * j,
              (rnd() - 0.5) * 0.4, 0.6 + 0.8 * rnd(), (rnd() - 0.5) * 0.4, 1.0 + 0.8 * rnd(), 0.06, 0.015, 0.95, 0.9, 0.7, 0.8, -0.4, 0.6);
          }
          for (const t of [-0.8, 0, 0.8]) this.smoke(e.x + fx * t, e.y - 0.3, e.z + fz * t, 2, 0.7, 0.82, 0.8, 0.74, 0.18, 0.6, 1.2);
          break;
        }
        case 'quake': {
          // dust thrown up round something colossal meeting the ground, rolling outward
          const n = Math.min(16, 4 + Math.round(e.a));
          for (let k = 0; k < n; k++) {
            const th = rnd() * 6.283, d = rnd() * e.a * 0.4, cx = Math.cos(th), sz = Math.sin(th);
            this.emit(SPRITE.puff, e.x + cx * d, e.y + 0.4, e.z + sz * d, cx * e.a * 0.4, 0.6 + 1.2 * rnd(), sz * e.a * 0.4,
              1.8 + rnd(), e.a * 0.22, e.a * (0.55 + 0.25 * rnd()), 0.43, 0.38, 0.3, 0.62, -0.25, 0.9);
          }
          break;
        }
        case 'poof':
          this.smoke(e.x, e.y, e.z, 10, 0.9, 0.09, 0.045, 0.13, 0.85, 1.4, 1.3);
          this.emit(SPRITE.glow, e.x, e.y, e.z, 0, 0, 0, 0.35, 0.6, 1.4, 1.2, 0.35, 1.6, 0);
          break;
        case 'down': case 'thud': case 'surf':
          this.smoke(e.x, e.y - (e.type === 'down' ? 0.8 : 0), e.z, e.type === 'thud' ? 2 : 4, e.type === 'thud' ? 0.25 : 0.55, 0.36, 0.31, 0.24, 0.55, 0.6, 0.7);
          break;
        case 'splash':
          for (let i = 0; i < 10; i++) {
            const th = rnd() * 6.283;
            this.emit(SPRITE.streak, e.x, e.y, e.z, Math.cos(th) * 1.5, 3 + 2 * rnd(), Math.sin(th) * 1.5, 0.45, 0.02, 0.01, 0.9, 0.95, 1, 0.8, 9.8, 0.5);
          }
          this.emit(SPRITE.ring, e.x, e.y + 0.02, e.z, 0, 0, 0, 0.5, 0.1, 0.8, 0.8, 0.9, 1, 0.6);
          break;
        case 'dodge':
          if (e.a === 2) this.emit(SPRITE.ring, e.x, e.y, e.z, 0, 0, 0, 0.45, 0.4, 2.2, 0.6, 2.4, 3.2, 0);
          else this.smoke(e.x, e.y - 0.4, e.z, 3, 0.4, 0.36, 0.31, 0.24, 0.45, 0.4, 0.5);
          break;
        case 'flurry':
          this.emit(SPRITE.ring, e.x, e.y, e.z, 0, 0, 0, 0.4, 0.3, 2.6, 1.2, 2.6, 4, 0);
          break;
        case 'charged':
          this.emit(SPRITE.star, e.x, e.y + 0.3, e.z, 0, 0, 0, 0.3, 0.3, 0.7, 2, 3, 5, 0);
          break;
        case 'pickup': case 'return':
          this.emit(SPRITE.star, e.x, e.y + 0.3, e.z, 0, 0.8, 0, 0.35, 0.2, 0.45, 2.5, 2.4, 1.6, 0);
          break;
        case 'eat':
          this.emit(SPRITE.star, e.x, e.y + 0.3, e.z, 0, 0.9, 0, 0.45, 0.2, 0.6, 3, 0.5, 0.6, 0);
          this.emit(SPRITE.glow, e.x, e.y, e.z, 0, 0.5, 0, 0.5, 0.2, 0.8, 2.2, 0.35, 0.4, 0);
          break;
        case 'break':
          this.sparks(e.x, e.y, e.z, 12, 4, 2.2, 1.8, 1.2);
          this.smoke(e.x, e.y, e.z, 4, 0.35, 0.3, 0.26, 0.2, 0.6, 0.5, 0.6);
          break;
      }
    }
    // the camps' fires: embers now and then
    w.awakeCamps(this.camps);
    for (const c of this.camps) {
      if (rnd() < dt * 3) this.emit(SPRITE.streak, c.x + (rnd() - 0.5) * 0.4, c.y + 0.5, c.z + (rnd() - 0.5) * 0.4, (rnd() - 0.5) * 0.6, 1.5 + rnd(), (rnd() - 0.5) * 0.6, 1.0, 0.012, 0.004, 4, 1.6, 0.4, 0, -0.5, 0.3);
    }
    // a charged blade's glint
    const s = w.hero, g = rig.gear;
    if (w.pose.glow > 0.05 && g[GEAR.grip + 3] > 0 && rnd() < dt * 20 * w.pose.glow) {
      this.heroPoint(s, rig, g[GEAR.grip] + g[GEAR.axis] * 0.9, g[GEAR.grip + 1] + g[GEAR.axis + 1] * 0.9, g[GEAR.grip + 2] + g[GEAR.axis + 2] * 0.9);
      this.emit(SPRITE.glow, P3[0], P3[1], P3[2], 0, 0.3, 0, 0.25, 0.08, 0.16, 0.8, 1.6, 3, 0);
    }
    for (const p of this.particles) {
      if (!p.live) continue;
      p.age += dt;
      if (p.age >= p.life) { p.live = false; continue; }
      const k = Math.exp(-p.drag * dt);
      p.vx *= k; p.vy = p.vy * k - p.gravity * dt; p.vz *= k;
      p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
    }
    // the slash's trail: the blade's base and tip while a swing strikes (a little either side)
    for (let i = 0; i < TRAIL; i++) this.trailAge[i] += dt;
    const m = w.act === 'swing' || w.act === 'flurry' ? MOVES[w.move] : null;
    if (m && w.u > m.hit[0] - 0.12 && w.u < m.hit[1] + 0.08 && g[GEAR.grip + 3] > 0) {
      this.trail.copyWithin(6, 0, (TRAIL - 1) * 6);
      this.trailAge.copyWithin(1, 0, TRAIL - 1);
      const len = g[GEAR.grip + 3] === MODEL.sword ? HERO_SHAPE.swordLen : HERO_SHAPE.clubLen;
      this.heroPoint(s, rig, g[GEAR.grip] + g[GEAR.axis] * 0.25, g[GEAR.grip + 1] + g[GEAR.axis + 1] * 0.25, g[GEAR.grip + 2] + g[GEAR.axis + 2] * 0.25);
      this.trail[0] = P3[0]; this.trail[1] = P3[1]; this.trail[2] = P3[2];
      this.heroPoint(s, rig, g[GEAR.grip] + g[GEAR.axis] * len, g[GEAR.grip + 1] + g[GEAR.axis + 1] * len, g[GEAR.grip + 2] + g[GEAR.axis + 2] * len);
      this.trail[3] = P3[0]; this.trail[4] = P3[1]; this.trail[5] = P3[2];
      this.trailAge[0] = 0;
    }
  }

  /** A hero-local point of the rig to the world (into P3). */
  private heroPoint(s: HeroState, rig: HeroRig, x: number, y: number, z: number): void {
    const yaw = s.yaw + rig.spin, sy = Math.sin(yaw), cy = Math.cos(yaw);
    P3[0] = s.x + cy * x + sy * z;
    P3[1] = s.y - PLAYER.groundBias + y;
    P3[2] = s.z - sy * x + cy * z;
  }

  // ---- packing ----
  private nParts = 0;
  private nSprites = 0;
  private cx = 0; private cy = 0; private cz = 0;

  /** A lit box: centre (world), half-extents along axes a (unit), b (unit), a × b, colour, glow. */
  private box(x: number, y: number, z: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number,
    ha: number, hb: number, hc: number, r: number, g: number, b: number, glow = 0): void {
    if (this.nParts >= MAX_PARTS) return;
    const f = this.frame.parts, o = this.nParts++ * (PART_BYTES / 4);
    const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
    f[o] = x - this.cx; f[o + 1] = y - this.cy; f[o + 2] = z - this.cz; f[o + 3] = glow;
    f[o + 4] = ax * ha; f[o + 5] = ay * ha; f[o + 6] = az * ha; f[o + 7] = r;
    f[o + 8] = bx * hb; f[o + 9] = by * hb; f[o + 10] = bz * hb; f[o + 11] = g;
    f[o + 12] = cx * hc; f[o + 13] = cy * hc; f[o + 14] = cz * hc; f[o + 15] = b;
  }

  private sprite(kind: number, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, size: number,
    r: number, g: number, b: number, a: number, c3 = 0, d3 = 0, x2 = 0, y2 = 0, z2 = 0, x3 = 0, y3 = 0, z3 = 0): void {
    if (this.nSprites >= MAX_SPRITES) return;
    const f = this.frame.sprites, o = this.nSprites++ * (SPRITE_BYTES / 4);
    f[o] = x0 - this.cx; f[o + 1] = y0 - this.cy; f[o + 2] = z0 - this.cz; f[o + 3] = kind;
    f[o + 4] = x1 - this.cx; f[o + 5] = y1 - this.cy; f[o + 6] = z1 - this.cz; f[o + 7] = size;
    f[o + 8] = x2 - this.cx; f[o + 9] = y2 - this.cy; f[o + 10] = z2 - this.cz; f[o + 11] = c3;
    f[o + 12] = x3 - this.cx; f[o + 13] = y3 - this.cy; f[o + 14] = z3 - this.cz; f[o + 15] = d3;
    f[o + 16] = r; f[o + 17] = g; f[o + 18] = b; f[o + 19] = a;
  }

  /** An arrow along unit (dx, dy, dz) with its tip at (x, y, z). */
  private arrow(x: number, y: number, z: number, dx: number, dy: number, dz: number, hostile: boolean): void {
    // a side axis ⟂ the arrow
    let sx = -dz, sz = dx;
    const sy = 0;
    let l = Math.hypot(sx, sz);
    if (l < 1e-4) { sx = 1; sz = 0; l = 1; }
    sx /= l; sz /= l;
    const L = HERO_SHAPE.arrowLen;
    this.box(x - dx * L * 0.5, y - dy * L * 0.5, z - dz * L * 0.5, dx, dy, dz, sx, sy, sz, L * 0.47, 0.006, 0.006, 0.3, 0.21, 0.11);
    this.box(x - dx * 0.03, y - dy * 0.03, z - dz * 0.03, dx, dy, dz, sx, sy, sz, 0.035, 0.012, 0.004, 0.5, 0.53, 0.58);
    const fr = hostile ? 0.08 : 0.36, fg = hostile ? 0.08 : 0.05, fb = hostile ? 0.09 : 0.03;
    this.box(x - dx * (L - 0.08), y - dy * (L - 0.08), z - dz * (L - 0.08), dx, dy, dz, sx, sy, sz, 0.055, 0.02, 0.002, fr, fg, fb);
    // (the second vane, turned a quarter)
    const ux = dy * sz - dz * sy, uy = dz * sx - dx * sz, uz = dx * sy - dy * sx;
    this.box(x - dx * (L - 0.08), y - dy * (L - 0.08), z - dz * (L - 0.08), dx, dy, dz, ux, uy, uz, 0.055, 0.02, 0.002, fr, fg, fb);
  }

  /** A weapon (kind: WEAPONS index) along unit axis a from its grip at (x, y, z), flat facing unit b. */
  private weapon(kind: number, x: number, y: number, z: number, ax: number, ay: number, az: number, bx: number, by: number, bz: number, glow: number): void {
    const model = WEAPONS[kind].model;
    if (model === MODEL.sword) {
      this.box(x + ax * 0.52, y + ay * 0.52, z + az * 0.52, ax, ay, az, bx, by, bz, 0.42, 0.004, 0.024, 0.5, 0.53, 0.58, glow);
      this.box(x, y, z, ax, ay, az, bx, by, bz, 0.09, 0.016, 0.016, 0.045, 0.032, 0.19);
      this.box(x + ax * 0.085, y + ay * 0.085, z + az * 0.085, ax, ay, az, bx, by, bz, 0.016, 0.016, 0.09, 0.045, 0.032, 0.19);
    } else {
      const bone = model === MODEL.bone;
      this.box(x + ax * 0.4, y + ay * 0.4, z + az * 0.4, ax, ay, az, bx, by, bz, 0.28, 0.05, 0.05, bone ? 0.5 : 0.17, bone ? 0.46 : 0.095, bone ? 0.37 : 0.045, glow);
      this.box(x, y, z, ax, ay, az, bx, by, bz, 0.12, 0.022, 0.022, 0.095, 0.045, 0.018);
    }
  }

  /** A flat bar from (x0, z0) to (x1, z1) at height y: half width hw, half thickness hh. */
  private bar(x0: number, y: number, z0: number, x1: number, z1: number, hw: number, hh: number, r: number, g: number, b: number): void {
    const dx = x1 - x0, dz = z1 - z0, l = Math.hypot(dx, dz) || 1;
    this.box((x0 + x1) / 2, y, (z0 + z1) / 2, dx / l, 0, dz / l, 0, 1, 0, l / 2, hh, hw, r, g, b);
  }

  /** A shield (SHIELDS index) lying face up at (x, y, z), turned by yaw. */
  private shield(kind: number, x: number, y: number, z: number, yaw: number): void {
    const m = SHIELDS[kind].model;
    if (m === SHIELD_MODEL.wooden || m === SHIELD_MODEL.traveler || m === SHIELD_MODEL.lid) {
      // a disc: four bars across it, an eighth of a turn apart
      const r = m === SHIELD_MODEL.lid ? 0.22 : m === SHIELD_MODEL.wooden ? 0.27 : 0.28;
      const cr = m === SHIELD_MODEL.wooden ? 0.2 : m === SHIELD_MODEL.traveler ? 0.15 : 0.36;
      const cg = m === SHIELD_MODEL.wooden ? 0.1 : m === SHIELD_MODEL.traveler ? 0.072 : 0.38;
      const cb = m === SHIELD_MODEL.wooden ? 0.04 : m === SHIELD_MODEL.traveler ? 0.03 : 0.41;
      for (let k = 0; k < 4; k++) {
        const a = yaw + k * 0.7854;
        this.box(x, y + 0.02 + 0.002 * k, z, Math.cos(a), 0, Math.sin(a), 0, 1, 0, r, 0.016, r * 0.42, cr, cg, cb);
      }
      if (m !== SHIELD_MODEL.wooden) this.box(x, y + 0.045, z, 1, 0, 0, 0, 1, 0, 0.055, 0.02, 0.055, 0.5, 0.53, 0.58);
      return;
    }
    // a kite: a body, and a square turned to a point below it
    const sy = Math.sin(yaw), cy = Math.cos(yaw);
    const boko = m === SHIELD_MODEL.boko;
    const half = boko ? 0.17 : 0.19;
    const r = boko ? 0.30 : 0.014, g = boko ? 0.25 : 0.038, b = boko ? 0.16 : 0.17;
    this.box(x + sy * 0.06, y + 0.02, z + cy * 0.06, cy, 0, -sy, 0, 1, 0, half, 0.016, 0.12, r, g, b);
    const t = -yaw + 0.7854;
    this.box(x - sy * 0.09, y + 0.02, z - cy * 0.09, Math.cos(t), 0, Math.sin(t), 0, 1, 0, half * 0.72, 0.016, half * 0.72, r, g, b);
    if (boko) this.box(x, y + 0.04, z, sy, 0, cy, 0, 1, 0, 0.3, 0.012, 0.016, 0.5, 0.46, 0.37);
    else this.box(x + sy * 0.05, y + 0.04, z + cy * 0.05, cy, 0, -sy, 0, 1, 0, 0.045, 0.012, 0.04, 0.45, 0.3, 0.05, 0.2);
  }

  /** A bow (BOWS index) lying on its side at (x, y, z), turned by yaw: the limbs bent, the string straight. */
  private bow(kind: number, x: number, y: number, z: number, yaw: number): void {
    const m = BOWS[kind].model;
    const fx = Math.sin(yaw), fz = Math.cos(yaw), rx = Math.cos(yaw), rz = -Math.sin(yaw);
    const r = m === BOW_MODEL.soldier ? 0.04 : m === BOW_MODEL.boko ? 0.19 : 0.11;
    const g = m === BOW_MODEL.soldier ? 0.022 : m === BOW_MODEL.boko ? 0.12 : 0.05;
    const b = m === BOW_MODEL.soldier ? 0.016 : m === BOW_MODEL.boko ? 0.065 : 0.022;
    for (let i = 0; i < 4; i++) {
      this.bar(x + BOW_U[i] * fx + BOW_V[i] * rx, y + 0.03, z + BOW_U[i] * fz + BOW_V[i] * rz,
        x + BOW_U[i + 1] * fx + BOW_V[i + 1] * rx, z + BOW_U[i + 1] * fz + BOW_V[i + 1] * rz, 0.016, 0.014, r, g, b);
    }
    this.bar(x + BOW_U[0] * fx + BOW_V[0] * rx, y + 0.026, z + BOW_U[0] * fz + BOW_V[0] * rz,
      x + BOW_U[4] * fx + BOW_V[4] * rx, z + BOW_U[4] * fz + BOW_V[4] * rz, 0.004, 0.004, 0.55, 0.52, 0.45);
    // (the grip's wrap)
    const gx = x + BOW_V[2] * rx, gz = z + BOW_V[2] * rz;
    this.bar(gx - fx * 0.07, y + 0.034, gz - fz * 0.07, gx + fx * 0.07, gz + fz * 0.07, 0.024, 0.016, 0.07, 0.035, 0.014);
  }

  /** Food (FOODS index) lying at (x, y, z). */
  private food(kind: number, x: number, y: number, z: number, yaw: number): void {
    const sy = Math.sin(yaw), cy = Math.cos(yaw);
    switch (kind) {
      case 1: // Hylian Shroom: a cream stem, a blue cap
        this.box(x, y + 0.06, z, 1, 0, 0, 0, 1, 0, 0.022, 0.06, 0.022, 0.6, 0.55, 0.42);
        this.box(x, y + 0.14, z, cy, 0, -sy, 0, 1, 0, 0.085, 0.035, 0.085, 0.12, 0.25, 0.5);
        break;
      case 2: case 3: { // raw meat (prime: bigger, deeper red, a glint of gold) with a bone
        const prime = kind === 3;
        this.box(x, y + 0.06, z, cy, 0, -sy, 0, 1, 0, prime ? 0.13 : 0.1, 0.055, prime ? 0.1 : 0.075, prime ? 0.55 : 0.6, prime ? 0.04 : 0.09, prime ? 0.06 : 0.12, prime ? 0.12 : 0);
        this.box(x + sy * 0.11, y + 0.07, z + cy * 0.11, sy, 0, cy, 0, 1, 0, 0.07, 0.014, 0.014, 0.6, 0.55, 0.45);
        break;
      }
      default: // Apple: a red body, a stem and a leaf
        this.box(x, y + 0.085, z, cy, 0, -sy, 0, 1, 0, 0.085, 0.08, 0.085, 0.75, 0.07, 0.05);
        this.box(x, y + 0.19, z, 1, 0, 0, 0, 1, 0, 0.008, 0.028, 0.008, 0.12, 0.07, 0.03);
        this.box(x + sy * 0.03, y + 0.2, z + cy * 0.03, sy, 0, cy, 0, 1, 0, 0.035, 0.004, 0.02, 0.1, 0.35, 0.08);
    }
  }

  /** This frame's combat for a camera at (cx, cy, cz). */
  pack(w: CombatWorld, cx: number, cy: number, cz: number): CombatFrame {
    this.cx = cx; this.cy = cy; this.cz = cz;
    this.nParts = 0; this.nSprites = 0;
    const fr = this.frame;
    // foes: their blocks and shadow bodies, nearest first (all of them fit)
    let n = 0;
    for (const f of w.foes) {
      if (!f.live || n >= FOE.max) continue;
      const kind = FOE_KINDS[f.kind];
      const s = f.body;
      packHeroUniforms(f.rig, s, cx, cy, cz, this.foeSrc, kind.skin);
      const flash = f.flash > 0 ? 1 : 0;
      packFoeWgsl(this.foeSrc, s.x - cx, s.y - PLAYER.groundBias - cy, s.z - cz, kind.scale, flash, f.kind, fr.foes, (n * FOE_STRIDE) / 4);
      const so = 9 * n;
      foeJoint(f, J.pelvis, P3); fr.shadows[so] = P3[0] - cx; fr.shadows[so + 1] = P3[1] - cy; fr.shadows[so + 2] = P3[2] - cz;
      foeJoint(f, J.head, P3); fr.shadows[so + 3] = P3[0] - cx; fr.shadows[so + 4] = P3[1] - cy; fr.shadows[so + 5] = P3[2] - cz;
      foeJoint(f, J.ankleL, P3);
      const ax = P3[0], ay = P3[1], az = P3[2];
      foeJoint(f, J.ankleR, P3);
      fr.shadows[so + 6] = 0.5 * (ax + P3[0]) - cx; fr.shadows[so + 7] = 0.5 * (ay + P3[1]) - cy; fr.shadows[so + 8] = 0.5 * (az + P3[2]) - cz;
      n++;
    }
    fr.nFoes = n;
    fr.nShadows = n;

    // what flies and what is stuck
    for (const p of w.shots) {
      if (!p.live) continue;
      if (p.arrow) {
        const sp = Math.hypot(p.vx, p.vy, p.vz);
        const dx = p.stuck ? p.dx : p.vx / (sp || 1), dy = p.stuck ? p.dy : p.vy / (sp || 1), dz = p.stuck ? p.dz : p.vz / (sp || 1);
        // (stuck: its head buried a little)
        const bury = p.stuck ? 0.12 : 0;
        this.arrow(p.x + dx * bury, p.y + dy * bury, p.z + dz * bury, dx, dy, dz, !p.friendly);
        if (!p.stuck) this.sprite(SPRITE.streak, p.x - p.vx * 0.05, p.y - p.vy * 0.05, p.z - p.vz * 0.05, p.x, p.y, p.z, 0.01, 0.4, 0.4, 0.35, 0);
      } else {
        // a weapon tumbling end over end about the axis ⟂ its flight
        const h = Math.hypot(p.vx, p.vz) || 1;
        const fx = p.vx / h, fz = p.vz / h;
        const c = Math.cos(p.spin), sn = Math.sin(p.spin);
        const ax = fx * c, ay = sn, az = fz * c;
        this.weapon(p.weapon, p.x - ax * 0.4, p.y - ay * 0.4, p.z - az * 0.4, ax, ay, az, -fz, 0, fx, 0);
      }
    }
    // what lies on the ground: a glint over each
    for (const p of w.pickups) {
      if (!p.live) continue;
      const cyw = Math.cos(p.yaw), syw = Math.sin(p.yaw);
      if (p.kind === 'weapon') {
        this.weapon(p.weapon, p.x - syw * 0.35, p.y + 0.03, p.z - cyw * 0.35, syw, 0, cyw, 0, 1, 0, p.weapon === WEAPON.master ? 0.15 : 0);
      } else if (p.kind === 'arrows') {
        for (let k = -1; k <= 1; k++) this.arrow(p.x + syw * 0.4 + cyw * k * 0.03, p.y + 0.02 + 0.01 * k * k, p.z + cyw * 0.4 - syw * k * 0.03, syw, 0, cyw, false);
      } else if (p.kind === 'shield') {
        this.shield(p.weapon, p.x, p.y, p.z, p.yaw);
      } else if (p.kind === 'bow') {
        this.bow(p.weapon, p.x, p.y, p.z, p.yaw);
      } else if (p.kind === 'food') {
        this.food(p.weapon, p.x, p.y, p.z, p.yaw);
      } else {
        const bob = 0.25 + 0.06 * Math.sin(this.time * 3 + p.yaw);
        this.box(p.x, p.y + bob, p.z, 0.7071, 0.7071, 0, 0, 0, 1, 0.09, 0.03, 0.09, 0.6, 0.02, 0.04, 0.5);
      }
      const tw = 0.5 + 0.5 * Math.sin(this.time * 4 + p.x);
      this.sprite(SPRITE.star, p.x, p.y + 0.35 + 0.05 * tw, p.z, 0, 0, 0, 0.12 + 0.08 * tw, 1.4, 1.3, 0.9, 0);
    }
    // the camps' fires: logs and a ring of stones, flames and their glow
    for (const c of this.camps) {
      for (let k = 0; k < 4; k++) {
        const a = k * 1.571 + 0.4;
        const dx = Math.cos(a), dz = Math.sin(a);
        // (leaning in to the middle)
        this.box(c.x + dx * 0.22, c.y + 0.14, c.z + dz * 0.22, -dx * 0.85, 0.53, -dz * 0.85, -dz, 0, dx, 0.3, 0.055, 0.055, 0.12, 0.07, 0.035);
      }
      for (let k = 0; k < 9; k++) {
        const a = (k / 9) * 6.283;
        this.box(c.x + Math.cos(a) * 0.62, c.y + 0.06, c.z + Math.sin(a) * 0.62, Math.cos(a), 0, Math.sin(a), 0, 1, 0, 0.09, 0.08, 0.11, 0.2, 0.19, 0.18);
      }
      for (let k = 0; k < 3; k++) {
        const fl = 0.85 + 0.15 * Math.sin(this.time * (7 + k) + k * 2);
        this.sprite(SPRITE.flame, c.x + (k - 1) * 0.1, c.y + 0.38 * fl, c.z + (k % 2) * 0.08, 0, 0, 0, 0.36 * fl - 0.04 * k, 3.2, 1.25, 0.28, 0, k * 1.7);
      }
      this.sprite(SPRITE.glow, c.x, c.y + 0.35, c.z, 0, 0, 0, 1.3, 0.9, 0.35, 0.08, 0);
    }
    // the slash's trail: a band between each two samples, brightest at the newest
    for (let i = 0; i + 1 < TRAIL; i++) {
      const a0 = this.trailAge[i], a1 = this.trailAge[i + 1];
      if (a1 > 0.11) break;
      const t = this.trail, o = 6 * i, q = 6 * (i + 1);
      const f0 = 1 - a0 / 0.11, f1 = Math.max(0, 1 - a1 / 0.11);
      this.sprite(SPRITE.trail, t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], t[o + 5], 0, 1.0, 1.15, 1.4, 0, f0 * 0.85, f1 * 0.85, t[q], t[q + 1], t[q + 2], t[q + 3], t[q + 4], t[q + 5]);
    }
    // the effects alive
    for (const p of this.particles) {
      if (!p.live) continue;
      const k = p.age / p.life;
      const size = p.size0 + (p.size1 - p.size0) * k;
      const fade = p.kind === SPRITE.puff ? (1 - k) * (1 - k) : 1 - k * k;
      if (p.kind === SPRITE.streak) {
        const l = 0.035;
        this.sprite(SPRITE.streak, p.x - p.vx * l, p.y - p.vy * l, p.z - p.vz * l, p.x, p.y, p.z, size, p.r * fade, p.g * fade, p.b * fade, p.a > 0 ? p.a * fade : 0);
      } else if (p.a > 0) {
        this.sprite(p.kind, p.x, p.y, p.z, 0, 0, 0, size, p.r, p.g, p.b, p.a * fade, p.spin, k);
      } else {
        this.sprite(p.kind, p.x, p.y, p.z, 0, 0, 0, size, p.r * fade, p.g * fade, p.b * fade, 0, p.spin, k);
      }
    }
    fr.nParts = this.nParts;
    fr.nSprites = this.nSprites;
    return fr;
  }
}

/** The packed hero's sword grip, for tests (camera-relative). */
export function packedGrip(src: Float32Array): [number, number, number] {
  return [src[HERO_UNIFORM.gear + GEAR.grip], src[HERO_UNIFORM.gear + GEAR.grip + 1], src[HERO_UNIFORM.gear + GEAR.grip + 2]];
}
