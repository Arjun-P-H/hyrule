/**
 * Third-person hero locomotion: run, sprint, sneak, jump, paraglide, climb,
 * swim and a stamina wheel, after the open-world Zelda games.
 * ---------------------------------------------------------------------------
 * Same contract as updatePlayer() (terrainMath.ts §31): deterministic, the
 * state is mutated in place (no allocation per tick), terrain queries go
 * through the CPU twin's elevation() on fp64 global coordinates — one ground
 * sample plus two forward differences per tick. The root (x, y, z) sits
 * between the feet; when grounded y = E + PLAYER.groundBias like the first-
 * person eye, so switching views keeps the same collision contract. Up in
 * the sky realm (skyworld/realm.ts) an island's top or an arch's deck is the
 * ground instead, and a sky lake or stream the water.
 */

import { elevation, type PlayerState } from '../terrainMath';
import { floorHeight, skyFloorUnder, skyGround, skyWater } from '../skyworld/realm';
import { ruinSiteAt, toBoxLocal } from '../ruins';
import { trunksNear, type Trunk } from '../forest/collision';
import { HERO, PLAYER, VALLEY } from '../constants';

export type HeroMove = 'ground' | 'air' | 'glide' | 'climb' | 'swim';

export type HeroState = {
  x: number; y: number; z: number;  // root, global fp64
  vx: number; vy: number; vz: number;
  yaw: number;                      // facing: forward = (sin yaw, 0, cos yaw), as the camera
  move: HeroMove;
  stamina: number;                  // 0..1, one wheel
  exhausted: boolean;               // ran dry: no stamina actions until the wheel is full again
  rest: number;                     // seconds since stamina was last spent
  sprinting: boolean;
  sneaking: boolean;
  sliding: boolean;
  jumpHeld: boolean;                // jumps, the glider and climb jumps fire on the press
  surge: number;                    // metres of climb-jump surge left
  ground: number;                   // E under the root at the last sample
  gx: number; gz: number;           // ∇E there (forward differences)
  water: number;                    // the water's surface there: the lakes and sea, a sky lake or stream, or −∞ (dry)
  impact: number;                   // downward speed at the last landing (m/s)
  landings: number;                 // counts landings so the rig reacts to each once
};

export type HeroInput = {
  forward: number; strafe: number;  // −1..1, camera-relative
  sprint: boolean;
  jump: boolean;
  sneak: boolean;
  camYaw: number;
};

export type HeroAction =
  | 'idle' | 'walk' | 'run' | 'sprint' | 'sneak' | 'tired' | 'slide'
  | 'jump' | 'fall' | 'glide' | 'climb' | 'hang' | 'swim' | 'dash';

const TAU = Math.PI * 2;
const GRAD_H = 0.25; // m — forward-difference step of the ground gradient

function wrapAngle(a: number): number {
  return a - TAU * Math.round(a / TAU);
}

/** Fraction of the remaining gap an exponential approach at `rate` covers in dt. */
function approach(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}

function turnToward(s: HeroState, targetYaw: number, rate: number, dt: number): void {
  s.yaw = wrapAngle(s.yaw + wrapAngle(targetYaw - s.yaw) * approach(rate, dt));
}

const HERO_RADIUS = 0.35; // m — body radius against ruin walls and tree trunks
const STEP_UP = 0.5;      // m — ruin ledges this high are stepped onto
const BODY_HEIGHT = 1.7;

type Body = { x: number; z: number; vx: number; vz: number };

/**
 * Ruins: push a body whose feet are at `feetY` out of any stone box it
 * overlaps (and drop the velocity into that face, so it slides along walls);
 * return the highest box top it stands over within a step's reach, or −∞.
 */
function collideRuins(s: Body, feetY: number, seed: number): number {
  const site = ruinSiteAt(s.x, s.z, seed);
  if (!site) return -Infinity;
  let floor = -Infinity;
  for (const b of site.boxes) {
    const [u, v] = toBoxLocal(b, s.x, s.z);
    const top = b.y + b.hy, bottom = b.y - b.hy;
    if (Math.abs(u) <= b.hx && Math.abs(v) <= b.hz && feetY >= top - STEP_UP) {
      floor = Math.max(floor, top);
      continue;
    }
    if (feetY >= top - STEP_UP || feetY + BODY_HEIGHT <= bottom) continue;
    const du = u - Math.max(-b.hx, Math.min(b.hx, u));
    const dv = v - Math.max(-b.hz, Math.min(b.hz, v));
    const dist = Math.sqrt(du * du + dv * dv);
    if (dist >= HERO_RADIUS) continue;
    let nu: number, nv: number, push: number;
    if (dist > 1e-6) {
      nu = du / dist; nv = dv / dist; push = HERO_RADIUS - dist;
    } else {
      // Centre inside the footprint: leave by the nearest face.
      const pu = b.hx - Math.abs(u), pv = b.hz - Math.abs(v);
      if (pu < pv) { nu = u < 0 ? -1 : 1; nv = 0; push = pu + HERO_RADIUS; } else { nu = 0; nv = v < 0 ? -1 : 1; push = pv + HERO_RADIUS; }
    }
    const cs = Math.cos(b.yaw), sn = Math.sin(b.yaw);
    const wx = cs * nu + sn * nv, wz = -sn * nu + cs * nv;
    s.x += wx * push;
    s.z += wz * push;
    const vn = s.vx * wx + s.vz * wz;
    if (vn < 0) { s.vx -= vn * wx; s.vz -= vn * wz; }
  }
  return floor;
}

const nearTrunks: Trunk[] = [];

/**
 * Trees: push a body whose feet are at `feetY` out of any trunk it overlaps
 * between feet and head (and drop the velocity into the trunk, so it slides
 * round it). Crowns and bushes do not block.
 */
function collideTrees(s: Body, feetY: number, seed: number): void {
  for (const t of trunksNear(s.x, s.z, seed, nearTrunks)) {
    if (feetY > t.y + t.top || feetY + BODY_HEIGHT < t.y - 0.5) continue;
    const dx = s.x - t.x, dz = s.z - t.z;
    const d = Math.sqrt(dx * dx + dz * dz), min = HERO_RADIUS + t.r;
    if (d >= min) continue;
    const nx = d > 1e-6 ? dx / d : 1, nz = d > 1e-6 ? dz / d : 0;
    s.x += nx * (min - d);
    s.z += nz * (min - d);
    const vn = s.vx * nx + s.vz * nz;
    if (vn < 0) { s.vx -= vn * nx; s.vz -= vn * nz; }
  }
}

/**
 * First person (updatePlayer's extra ground): tree trunks and ruins, with
 * their walls, and the top of a sky island under the feet.
 */
export function firstPersonGround(p: PlayerState, feetY: number, seed: number): number {
  collideTrees(p, feetY, seed);
  const box = collideRuins(p, feetY, seed);
  const sky = skyGround(p.x, feetY, p.z, seed);
  return Math.max(box, sky === sky ? sky : -Infinity);
}

function sampleGround(s: HeroState, seed: number): void {
  collideTrees(s, s.y, seed);
  const boxTop = collideRuins(s, s.y, seed);
  sampleSurface(s, seed);
  if (boxTop > s.ground) {
    s.ground = boxTop;
    s.gx = 0;
    s.gz = 0;
  }
}

function sampleSurface(s: HeroState, seed: number): void {
  const f = skyFloorUnder(s.x, s.y, s.z, seed);
  if (f) {
    const t = floorHeight(f, s.x, s.z);
    // (past an arch's edge or an island's rim the floor ends: no slope across it)
    const tx = floorHeight(f, s.x + GRAD_H, s.z), tz = floorHeight(f, s.x, s.z + GRAD_H);
    s.gx = tx === tx ? (tx - t) / GRAD_H : 0;
    s.gz = tz === tz ? (tz - t) / GRAD_H : 0;
    s.ground = t;
    const w = skyWater(s.x, s.y, s.z, seed);
    s.water = w === w ? w : -Infinity;
    return;
  }
  const e = elevation(s.x, s.z, seed, 0);
  s.gx = (elevation(s.x + GRAD_H, s.z, seed, 0) - e) / GRAD_H;
  s.gz = (elevation(s.x, s.z + GRAD_H, seed, 0) - e) / GRAD_H;
  s.ground = e;
  s.water = VALLEY.wLevel;
}

function deepWater(s: HeroState): boolean {
  return s.ground < s.water - HERO.swimDepth;
}

/** Pushing into ground too steep to run up, with stamina to spend: grab it. */
function wantsClimb(s: HeroState, wx: number, wz: number, mag: number, fresh: boolean): boolean {
  const m = Math.sqrt(s.gx * s.gx + s.gz * s.gz);
  return fresh && mag > 0 && m > HERO.walkSlope && wx * s.gx + wz * s.gz > 0.5 * m;
}

function land(s: HeroState, wx: number, wz: number, mag: number, fresh: boolean): void {
  s.impact = Math.max(0, -s.vy);
  s.landings++;
  s.y = s.ground + PLAYER.groundBias;
  s.vy = 0;
  s.move = wantsClimb(s, wx, wz, mag, fresh) ? 'climb' : 'ground';
}

/**
 * Put the hero at a root position with a velocity (spawn, view switch, drop
 * from fly mode) and pick the state the terrain allows there.
 */
export function placeHero(
  s: HeroState, x: number, y: number, z: number,
  vx: number, vy: number, vz: number, yaw: number, seed: number,
): void {
  s.x = x; s.y = y; s.z = z; s.vx = vx; s.vy = vy; s.vz = vz;
  s.yaw = wrapAngle(yaw);
  s.surge = 0;
  s.jumpHeld = true; // a Space still held from the previous mode must not fire
  sampleGround(s, seed); // islands and ruin ledges count from this height
  const floor = s.ground + PLAYER.groundBias;
  if (deepWater(s) && y < s.water) {
    s.move = 'swim';
    s.y = Math.max(y, s.water - HERO.swimFloat - 0.3);
  } else if (y <= floor + 0.05) {
    s.move = 'ground';
    s.y = floor;
    s.vy = 0;
  } else {
    s.move = 'air';
    s.y = y;
  }
}

export function createHeroState(x: number, z: number, yaw: number, seed: number): HeroState {
  const s: HeroState = {
    x, y: 0, z, vx: 0, vy: 0, vz: 0, yaw: 0, move: 'ground',
    stamina: 1, exhausted: false, rest: 0,
    sprinting: false, sneaking: false, sliding: false, jumpHeld: false,
    surge: 0, ground: 0, gx: 0, gz: 0, water: VALLEY.wLevel, impact: 0, landings: 0,
  };
  placeHero(s, x, -Infinity, z, 0, 0, 0, yaw, seed);
  s.jumpHeld = false;
  return s;
}

/** One tick of hero locomotion (any dt; the engine caps it at 50 ms). */
export function updateHero(s: HeroState, input: HeroInput, dt: number, seed: number): void {
  // Camera-relative wish direction, unit length (diagonals are not faster).
  let wx = 0, wz = 0, mag = 0;
  if (input.forward !== 0 || input.strafe !== 0) {
    const sn = Math.sin(input.camYaw), cs = Math.cos(input.camYaw);
    wx = sn * input.forward + cs * input.strafe;
    wz = cs * input.forward - sn * input.strafe;
    const l = Math.sqrt(wx * wx + wz * wz);
    mag = Math.min(1, l);
    wx /= l; wz /= l;
  }
  const jumpPressed = input.jump && !s.jumpHeld;
  s.jumpHeld = input.jump;
  const fresh = !s.exhausted && s.stamina > 0; // stamina actions allowed
  let spend = 0;
  s.sprinting = false;
  s.sliding = false;
  s.sneaking = false;

  switch (s.move) {
    case 'ground': {
      if (wantsClimb(s, wx, wz, mag, fresh)) {
        s.move = 'climb';
        s.vx = 0; s.vz = 0; s.vy = 0;
        break;
      }
      s.sneaking = input.sneak;
      let speed: number = s.exhausted ? HERO.exhaustedSpeed : s.sneaking ? HERO.sneakSpeed : HERO.runSpeed;
      if (input.sprint && fresh && !s.sneaking && mag > 0) {
        speed = HERO.sprintSpeed;
        s.sprinting = true;
        spend += HERO.staminaSprint * dt;
      }
      if (s.ground < s.water - 0.3) speed *= HERO.wadeFactor;
      const k = approach(HERO.accel, dt);
      s.vx += (wx * speed * mag - s.vx) * k;
      s.vz += (wz * speed * mag - s.vz) * k;
      const m = Math.sqrt(s.gx * s.gx + s.gz * s.gz);
      if (m > HERO.walkSlope) {
        // Too steep to stand on: slide down the fall line (∇E points uphill).
        s.sliding = true;
        const f = HERO.slideAccel * dt * Math.min(1, (m - HERO.walkSlope) * 2) / m;
        s.vx -= s.gx * f;
        s.vz -= s.gz * f;
      }
      if (mag > 0) turnToward(s, Math.atan2(wx, wz), HERO.turnRate, dt);
      const prevY = s.y;
      s.x += s.vx * dt;
      s.z += s.vz * dt;
      sampleGround(s, seed);
      const floor = s.ground + PLAYER.groundBias;
      if (jumpPressed) {
        s.vy = HERO.jumpVelocity;
        s.y = Math.max(prevY, floor);
        s.move = 'air';
      } else if (deepWater(s)) {
        s.move = 'swim';
        s.vy = 0;
      } else if (prevY - floor > 0.05 + Math.sqrt(s.vx * s.vx + s.vz * s.vz) * dt * HERO.fallSlope) {
        s.move = 'air'; // the ground fell away faster than a slope can be followed
        s.vy = 0;
      } else {
        s.y = floor;
        s.vy = 0;
      }
      break;
    }

    case 'air': {
      if (mag > 0) {
        const k = approach(HERO.airAccel, dt);
        s.vx += (wx * HERO.runSpeed * mag - s.vx) * k;
        s.vz += (wz * HERO.runSpeed * mag - s.vz) * k;
        turnToward(s, Math.atan2(wx, wz), HERO.turnRate * 0.5, dt);
      }
      s.vy = Math.max(s.vy - PLAYER.gravity * dt, -HERO.terminalVelocity);
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.z += s.vz * dt;
      sampleGround(s, seed);
      if (jumpPressed && fresh && s.y - s.ground > HERO.glideMinHeight) {
        s.move = 'glide';
      } else if (deepWater(s) && s.y < s.water - HERO.swimFloat) {
        s.move = 'swim';
        s.vy *= 0.3;
      } else if (s.y <= s.ground + PLAYER.groundBias) {
        land(s, wx, wz, mag, fresh);
      }
      break;
    }

    case 'glide': {
      spend += HERO.staminaGlide * dt;
      if (jumpPressed || !fresh) s.move = 'air'; // fold the glider (also when the wheel runs dry)
      const tx = mag > 0 ? wx * HERO.glideSpeed * mag : Math.sin(s.yaw) * HERO.glideDrift;
      const tz = mag > 0 ? wz * HERO.glideSpeed * mag : Math.cos(s.yaw) * HERO.glideDrift;
      const k = approach(HERO.glideAccel, dt);
      s.vx += (tx - s.vx) * k;
      s.vz += (tz - s.vz) * k;
      s.vy += (-HERO.glideSink - s.vy) * approach(3.0, dt);
      if (mag > 0) turnToward(s, Math.atan2(wx, wz), HERO.turnRate * 0.35, dt);
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.z += s.vz * dt;
      sampleGround(s, seed);
      if (deepWater(s) && s.y < s.water - HERO.swimFloat) {
        s.move = 'swim';
        s.vy *= 0.3;
      } else if (s.y <= s.ground + PLAYER.groundBias) {
        land(s, wx, wz, mag, fresh);
      }
      break;
    }

    case 'climb': {
      const m = Math.sqrt(s.gx * s.gx + s.gz * s.gz);
      if (m < HERO.climbExitSlope) {
        // The wall eased off (top reached, or climbed down to walkable ground).
        s.move = 'ground';
        s.vx = Math.sin(s.yaw) * 1.5;
        s.vz = Math.cos(s.yaw) * 1.5;
        s.vy = 0;
        s.surge = 0;
        break;
      }
      const ux = s.gx / m, uz = s.gz / m; // uphill
      const rx = uz, rz = -ux;            // hero's right while facing the wall
      if (!fresh || input.sneak) {
        // Let go: push off the wall and fall.
        s.move = 'air';
        s.vx = -ux * 1.5; s.vz = -uz * 1.5; s.vy = 0;
        s.surge = 0;
        break;
      }
      if (jumpPressed && s.surge <= 0) {
        s.surge = HERO.climbJump;
        spend += HERO.staminaClimbJump;
      }
      const up = (wx * ux + wz * uz) * mag;
      const side = (wx * rx + wz * rz) * mag;
      let along = up * HERO.climbSpeed; // along the fall line, m/s
      if (s.surge > 0) {
        along = HERO.climbJumpSpeed;
        s.surge -= along * dt;
      }
      spend += (along !== 0 || side !== 0 ? HERO.staminaClimb : HERO.staminaHang) * dt;
      const inv = 1 / Math.sqrt(1 + m * m); // horizontal share of motion along the fall line
      s.vx = ux * along * inv + rx * side * HERO.climbSpeed;
      s.vz = uz * along * inv + rz * side * HERO.climbSpeed;
      s.vy = along * m * inv;
      turnToward(s, Math.atan2(ux, uz), HERO.turnRate, dt);
      s.x += s.vx * dt;
      s.z += s.vz * dt;
      sampleGround(s, seed);
      s.y = s.ground + PLAYER.groundBias;
      if (deepWater(s)) {
        s.move = 'swim';
        s.vy = 0;
        s.surge = 0;
      }
      break;
    }

    case 'swim': {
      const dash = input.sprint && fresh && mag > 0;
      const speed = dash ? HERO.swimDashSpeed : s.exhausted ? HERO.swimSpeed * 0.6 : HERO.swimSpeed;
      if (dash) {
        s.sprinting = true;
        spend += HERO.staminaSwimDash * dt;
      }
      const k = approach(3.0, dt);
      s.vx += (wx * speed * mag - s.vx) * k;
      s.vz += (wz * speed * mag - s.vz) * k;
      if (mag > 0) turnToward(s, Math.atan2(wx, wz), HERO.turnRate * 0.6, dt);
      // Buoyancy: a damped spring settles the root at the float depth.
      s.vy += ((s.water - HERO.swimFloat - s.y) * 12 - s.vy * 5) * dt;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.z += s.vz * dt;
      sampleGround(s, seed);
      if (s.ground > s.water - HERO.swimDepth + 0.1) {
        // Shallow enough to stand: wade out.
        s.move = 'ground';
        s.y = Math.max(s.y, s.ground + PLAYER.groundBias);
        s.vy = 0;
      }
      break;
    }
  }

  if (spend > 0) {
    s.stamina -= spend;
    s.rest = 0;
    if (s.stamina <= 0) {
      s.stamina = 0;
      s.exhausted = true;
    }
  } else {
    s.rest += dt;
    // No recovery while hanging on a wall or under the glider.
    if (s.move !== 'climb' && s.move !== 'glide' && s.rest > HERO.staminaRegenDelay) {
      const rate = s.exhausted ? HERO.staminaRegenExhausted
        : s.move === 'swim' ? HERO.staminaRegenSwim : HERO.staminaRegen;
      s.stamina = Math.min(1, s.stamina + rate * dt);
      if (s.stamina >= 1) s.exhausted = false;
    }
  }
}

/** What the hero is doing, for the HUD and the animation rig. */
export function heroAction(s: HeroState): HeroAction {
  const hv = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
  switch (s.move) {
    case 'ground':
      if (s.sliding) return 'slide';
      if (s.sprinting) return 'sprint';
      if (hv < 0.3) return 'idle';
      if (s.sneaking) return 'sneak';
      if (s.exhausted) return 'tired';
      return hv < 3.0 ? 'walk' : 'run';
    case 'air':
      return s.vy > 0 ? 'jump' : 'fall';
    case 'glide':
      return 'glide';
    case 'climb':
      return hv > 0.05 || s.vy !== 0 ? 'climb' : 'hang';
    case 'swim':
      return s.sprinting ? 'dash' : 'swim';
  }
}
