import { describe, it, expect } from 'bun:test';
import { elevation, slope, waterLevel } from '../src/lib/terrainMath';
import { DEFAULT_SEED, HERO, HERO_CAMERA, HERO_SHAPE, PLAYER, SPAWN, CLIMB, SKYDIVE } from '../src/lib/constants';
import {
  createHeroState, heroAction, placeHero, climbClearance, updateHero, type HeroInput, type HeroState,
} from '../src/lib/hero/locomotion';
import { trunkSection, trunksNear, type Trunk } from '../src/lib/forest/collision';
import { rockBodiesNear, rockTopAt, type RockBody } from '../src/lib/rocks/field';
import { createHeroRig, updateHeroRig, packHeroUniforms, J, GEAR, HERO_JOINTS, HERO_UNIFORM, type HeroRig } from '../src/lib/hero/rig';
import { createHeroCamera, updateHeroCamera } from '../src/lib/hero/camera';
import { TERRAIN_FS, minifyGlsl, withHero } from '../src/components/terrain/demoShader';
import { HERO_BOX, HERO_BYTES, packHeroWgsl } from '../src/lib/gpu/hero.wgsl';
import { transformShader } from '../bench/gpu/runner';
import { plantInCell } from '../src/lib/forest/placement';
import { CELL } from '../src/lib/forest/species';

/**
 * Third-person hero: locomotion (run / sprint / jump / glide / climb / swim /
 * stamina), the procedural rig, the orbit camera, and the contract with the
 * shader's heroMap(). Everything runs on the real terrain around the spawn.
 */

const SEED = DEFAULT_SEED;
const DT = 1 / 60;
const BIAS = PLAYER.groundBias;

function findSpot(pred: (x: number, z: number) => boolean, step = 3, radius = 2500): [number, number] {
  for (let r = 0; r <= radius; r += step) {
    for (let a = 0; a < 64; a++) {
      const x = SPAWN.x + r * Math.cos((a / 64) * Math.PI * 2);
      const z = SPAWN.z + r * Math.sin((a / 64) * Math.PI * 2);
      if (pred(x, z)) return [x, z];
      if (r === 0) break;
    }
  }
  throw new Error('no spot found');
}

function gradAt(x: number, z: number): [number, number] {
  const e = elevation(x, z, SEED, 0);
  return [(elevation(x + 0.25, z, SEED, 0) - e) / 0.25, (elevation(x, z + 0.25, SEED, 0) - e) / 0.25];
}

/** Heading whose next `len` metres stay flattest. */
function flattestYaw(x: number, z: number, len = 20): number {
  let best = 0, bestM = Infinity;
  for (let k = 0; k < 24; k++) {
    const yaw = (k / 24) * Math.PI * 2;
    let m = 0;
    for (let d = 1; d <= len; d++) m = Math.max(m, slope(x + Math.sin(yaw) * d, z + Math.cos(yaw) * d, SEED));
    if (m < bestM) { bestM = m; best = yaw; }
  }
  return best;
}

/** No tree (bushes aside) within r metres of (x, z). */
function clearOfTrees(x: number, z: number, r: number): boolean {
  for (let iz = Math.floor((z - r) / CELL); iz <= Math.floor((z + r) / CELL); iz++) {
    for (let ix = Math.floor((x - r) / CELL); ix <= Math.floor((x + r) / CELL); ix++) {
      const p = plantInCell(ix, iz, SEED);
      if (p && p.species !== 'bush' && Math.hypot(p.x - x, p.z - z) < r) return false;
    }
  }
  return true;
}
// (level open ground for a run: flat at the spot and 12 m round it, dry, no tree within 30 m)
const FLAT = findSpot((x, z) => slope(x, z, SEED) < 0.06 && elevation(x, z, SEED, 0) > waterLevel(x, z, SEED) + 2.9
  && [0, 1, 2, 3, 4, 5, 6, 7].every((k) => slope(x + 12 * Math.cos(k * Math.PI / 4), z + 12 * Math.sin(k * Math.PI / 4), SEED) < 0.08)
  && clearOfTrees(x, z, 30), 4);
const FLAT_YAW = flattestYaw(FLAT[0], FLAT[1], 60); // (sprinting runs ~45 m)
const DEEP = findSpot((x, z) => elevation(x, z, SEED, 0) < waterLevel(x, z, SEED) - 3, 2);
const WALL = findSpot((x, z) => {
  const [gx, gz] = gradAt(x, z);
  const m = Math.hypot(gx, gz);
  if (m < 1.35 || m > 5) return false; // (a wall, not a sheer step)
  // (steep from 2 m below the spot to 6 m above it: room to climb, jump and push off)
  for (let k = -2; k <= 6; k++) if (k !== 0 && slope(x + (gx / m) * k, z + (gz / m) * k, SEED) < 1.2) return false;
  return true;
}, 2);
const WALL_UP = (() => { const [gx, gz] = gradAt(WALL[0], WALL[1]); return Math.atan2(gx, gz); })();

function input(over: Partial<HeroInput> = {}): HeroInput {
  return { forward: 0, strafe: 0, sprint: false, jump: false, sneak: false, camYaw: 0, ...over };
}

function run(s: HeroState, seconds: number, inp: HeroInput | ((t: number) => HeroInput), each?: (t: number) => void): void {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) {
    updateHero(s, typeof inp === 'function' ? inp(i * DT) : inp, DT, SEED);
    each?.(i * DT);
  }
}

const hspeed = (s: HeroState) => Math.hypot(s.vx, s.vz);

describe('Hero locomotion — ground', () => {
  it('spawns standing on the terrain (same GROUND_BIAS contract as the first-person eye)', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    expect(s.move).toBe('ground');
    expect(s.y).toBeCloseTo(elevation(FLAT[0], FLAT[1], SEED, 0) + BIAS, 9);
    expect(heroAction(s)).toBe('idle');
    const w = createHeroState(DEEP[0], DEEP[1], 0, SEED);
    expect(w.move).toBe('swim');
  });

  it('runs at HERO.runSpeed in the camera-relative direction and turns to face it', () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW + 2.0, SEED);
    run(s, 1.2, input({ forward: 1, camYaw: FLAT_YAW }));
    expect(s.move).toBe('ground');
    expect(hspeed(s)).toBeGreaterThan(HERO.runSpeed * 0.98);
    expect(hspeed(s)).toBeLessThan(HERO.runSpeed * 1.02);
    expect(Math.atan2(s.vx, s.vz)).toBeCloseTo(FLAT_YAW - (FLAT_YAW > Math.PI ? 2 * Math.PI : 0), 2);
    expect(Math.cos(s.yaw - FLAT_YAW)).toBeGreaterThan(0.999);
    expect(heroAction(s)).toBe('run');
  });

  it('diagonal input is not faster than straight input', () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    run(s, 1.2, input({ forward: 1, strafe: 1, camYaw: FLAT_YAW - Math.PI / 4 }));
    expect(hspeed(s)).toBeLessThan(HERO.runSpeed * 1.02);
    expect(hspeed(s)).toBeGreaterThan(HERO.runSpeed * 0.95);
  });

  it('sneaking is slow and crouched; ground contact holds on walkable slopes', () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    let maxGap = 0;
    run(s, 1.5, input({ forward: 1, sneak: true, camYaw: FLAT_YAW }), () => {
      if (s.move === 'ground') maxGap = Math.max(maxGap, Math.abs(s.y - (elevation(s.x, s.z, SEED, 0) + BIAS)));
    });
    expect(hspeed(s)).toBeCloseTo(HERO.sneakSpeed, 1);
    expect(heroAction(s)).toBe('sneak');
    expect(maxGap).toBeLessThan(1e-9);
  });

  it('jumps on the press (≈ v²/2g apex), never auto-repeats while Space is held', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    const y0 = s.y;
    let apex = y0, jumps = 0, wasAir = false;
    run(s, 2.0, input({ jump: true }), () => {
      apex = Math.max(apex, s.y);
      if (s.move === 'air' && !wasAir) jumps++;
      wasAir = s.move === 'air';
    });
    // (v²/2g, plus the hang at the top: |vy| < apexBand falls at apexGravity · g)
    const ideal = (HERO.jumpVelocity ** 2 + HERO.apexBand ** 2 * (1 / HERO.apexGravity - 1)) / (2 * PLAYER.gravity);
    expect(apex - y0).toBeGreaterThan(ideal * 0.93);
    expect(apex - y0).toBeLessThan(ideal * 1.02);
    expect(jumps).toBe(1);
    expect(s.move).toBe('ground');
    expect(s.landings).toBe(1);
    expect(s.impact).toBeGreaterThan(HERO.jumpVelocity * 0.9);
  });
});

describe('Hero locomotion — stamina', () => {
  it('sprint drains the wheel at HERO.staminaSprint per second, then the hero is exhausted', () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    const sprint = input({ forward: 1, sprint: true, camYaw: FLAT_YAW });
    run(s, 1.0, sprint);
    expect(s.stamina).toBeCloseTo(1 - HERO.staminaSprint * 1.0, 2);
    expect(heroAction(s)).toBe('sprint');
    let t = 1.0;
    while (!s.exhausted && t < 10) { updateHero(s, sprint, DT, SEED); t += DT; }
    expect(t).toBeCloseTo(1 / HERO.staminaSprint, 1); // one wheel lasts 1/staminaSprint seconds
    expect(s.stamina).toBe(0);
    // exhausted: no sprinting, a slow jog
    run(s, 1.0, sprint);
    expect(s.sprinting).toBe(false);
    expect(heroAction(s)).toBe('tired');
    expect(hspeed(s)).toBeLessThan(HERO.exhaustedSpeed * 1.01);
  });

  it('recovers only after the rest delay; exhaustion clears when the wheel is full', () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    s.stamina = 0.5;
    run(s, HERO.staminaRegenDelay * 0.9, input());
    expect(s.stamina).toBe(0.5);
    run(s, 1.0, input());
    expect(s.stamina).toBeGreaterThan(0.5 + HERO.staminaRegen * 0.8);
    s.stamina = 0; s.exhausted = true;
    run(s, 1 / HERO.staminaRegenExhausted + HERO.staminaRegenDelay + 0.2, input());
    expect(s.stamina).toBe(1);
    expect(s.exhausted).toBe(false);
  });
});

describe('Hero locomotion — skydiving', () => {
  const drop = (h = 1500) => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    placeHero(s, s.x, s.ground + h, s.z, 0, 0, 0, FLAT_YAW, SEED);
    s.jumpHeld = false;
    return s;
  };

  it('spreads only once high and falling fast; a short hop never does', () => {
    const s = drop();
    run(s, 0.2, input());
    expect(s.skydive).toBe(false); // (still slower than enterSpeed)
    run(s, 0.3, input());
    expect(s.skydive).toBe(true);
    expect(heroAction(s)).toBe('skydive');
    const low = drop(SKYDIVE.minAltitude - 2);
    run(low, 3, input());
    expect(low.landings).toBe(1);
    expect(low.skydive).toBe(false);
  });

  it('holding sprint tucks into a nosedive at its terminal speed; letting go flares, braking no harder than flareDecel', () => {
    const s = drop(3000);
    run(s, 9, input({ sprint: true }));
    expect(heroAction(s)).toBe('dive');
    expect(s.dive).toBeGreaterThan(0.999);
    expect(-s.vy).toBeCloseTo(SKYDIVE.diveTerminal, 0);
    let flared = false, hardest = 0, prev = s.vy;
    run(s, 2, input(), () => {
      flared ||= s.flare > 0;
      hardest = Math.max(hardest, (s.vy - prev) / DT);
      prev = s.vy;
    });
    expect(flared).toBe(true);
    expect(hardest).toBeLessThan(SKYDIVE.flareDecel + 1e-6);
    expect(-s.vy).toBeCloseTo(SKYDIVE.glideTerminal, 0);
  });

  it('steers faster spread than tucked, and the spread body glides forward on its own', () => {
    const a = drop(), b = drop();
    run(a, 3, input({ forward: 1, camYaw: FLAT_YAW }));
    run(b, 3, input({ forward: 1, sprint: true, camYaw: FLAT_YAW }));
    expect(hspeed(a)).toBeGreaterThan(SKYDIVE.glideMaxXZ);
    expect(hspeed(b)).toBeLessThan(SKYDIVE.diveMaxXZ + 0.5);
    const c = drop();
    run(c, 3, input());
    expect(hspeed(c)).toBeGreaterThan(0.5); // (lift off the spread body's pitch, along its heading)
  });

  it('lands rolling on from a spread skydive and squatting from a nosedive, then runs again', () => {
    const s = drop(60);
    run(s, 10, input({ forward: 1, camYaw: FLAT_YAW }), () => { if (s.landings === 1 && s.recover > 0) expect(heroAction(s)).toBe('roll'); });
    expect(s.landings).toBe(1);
    expect(s.impact).toBeGreaterThan(SKYDIVE.rollMin);
    expect(s.impact).toBeLessThan(SKYDIVE.squatMin);
    expect(s.recover).toBe(0);
    expect(s.move === 'ground' || s.move === 'swim' || s.move === 'climb').toBe(true);
    const d = drop(400);
    let squatted = false;
    run(d, 20, input({ sprint: true }), () => { squatted ||= heroAction(d) === 'squat'; });
    expect(d.impact).toBeGreaterThan(SKYDIVE.squatMin);
    expect(squatted).toBe(true);
  });

  it('the rig spreads belly-down, comes upright to land, and the camera widens with speed', () => {
    const s = drop(3000), r = createHeroRig(), c = createHeroCamera();
    run(s, 4, input(), () => { updateHeroRig(r, s, DT); updateHeroCamera(c, s, 0, -0.3, DT, SEED); });
    // (belly-down: the head out ahead of the pelvis, not over it)
    const o = r.local, up = o[3 * J.head + 1] - o[3 * J.pelvis + 1], ahead = o[3 * J.head + 2] - o[3 * J.pelvis + 2];
    expect(Math.abs(up)).toBeLessThan(ahead);
    expect(c.fov).toBeGreaterThan(1.03);
    run(s, 8, input({ sprint: true }), () => { updateHeroRig(r, s, DT); updateHeroCamera(c, s, 0, -0.3, DT, SEED); });
    expect(o[3 * J.head + 1]).toBeLessThan(o[3 * J.pelvis + 1]); // head first
    expect(c.fov).toBeGreaterThan(1.35);
    // (coming in to land: upright in the plain fall pose — weights[1] — before the feet touch)
    const l = drop(60), lr = createHeroRig();
    let upright = 0;
    run(l, 10, input(), () => {
      updateHeroRig(lr, l, DT);
      if (l.move === 'air' && l.y - BIAS - l.ground < 1.5 && upright === 0) upright = lr.weights[1];
    });
    expect(upright).toBeGreaterThan(0.6);
  });
});

describe('Hero rig — skydiving frames', () => {
  it('the torso and head turn smoothly into and out of a nosedive (no chest-for-back flip past level)', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED), r = createHeroRig();
    placeHero(s, s.x, s.ground + 3000, s.z, 0, 0, 0, 0, SEED);
    s.jumpHeld = false;
    let prev: number[] | null = null, worst = 1, shieldFront = -Infinity, armOut = 0, faceDown = 1;
    const o = r.local, g = r.gear, P3 = (j: number, k: number) => o[3 * j + k];
    for (let i = 0; i < 600; i++) {
      updateHero(s, input({ sprint: i > 240 && i < 480 }), DT, SEED);
      updateHeroRig(r, s, DT);
      const f = [...r.frames];
      if (prev) for (const k of [6, 15]) worst = Math.min(worst, f[k] * prev[k] + f[k + 1] * prev[k + 1] + f[k + 2] * prev[k + 2]);
      prev = f;
      // the shield on the back stays behind the chest, whichever way the body is tipped
      let d = 0;
      for (let k = 0; k < 3; k++) d += (g[GEAR.shield + k] - P3(J.chest, k)) * f[6 + k];
      shieldFront = Math.max(shieldFront, d);
      // deep in the dive the hands lie along the sides: close to the spine's line, not out in the wind
      if (i > 420 && i < 480) {
        faceDown = Math.min(faceDown, -f[16]); // (the head's forward: looking down the fall at the ground)
        const ax = [0, 1, 2].map((k) => P3(J.chest, k) - P3(J.pelvis, k)), al = Math.hypot(...ax);
        for (const h of [J.handL, J.handR]) {
          const v = [0, 1, 2].map((k) => P3(h, k) - P3(J.pelvis, k)), t = (v[0] * ax[0] + v[1] * ax[1] + v[2] * ax[2]) / (al * al);
          armOut = Math.max(armOut, Math.hypot(...v.map((x, k) => x - ax[k] * t)));
        }
      }
    }
    expect(worst).toBeGreaterThan(0.95);
    expect(shieldFront).toBeLessThan(0);
    expect(armOut).toBeLessThan(0.3);
    expect(faceDown).toBeGreaterThan(0.9);
  });
});

describe('Hero locomotion — paraglider', () => {
  const high = () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    placeHero(s, s.x, s.ground + 40, s.z, 0, 0, 0, FLAT_YAW, SEED);
    s.jumpHeld = false;
    return s;
  };

  it('opens on a mid-air press, sinks at ≈ glideSink, flies at glideSpeed, drains glide stamina', () => {
    const s = high();
    expect(s.move).toBe('air');
    run(s, 0.3, input());
    updateHero(s, input({ jump: true }), DT, SEED);
    expect(s.move).toBe('glide');
    run(s, 3.0, input({ forward: 1, camYaw: FLAT_YAW }));
    expect(s.move).toBe('glide');
    expect(s.vy).toBeCloseTo(-HERO.glideSink, 1);
    expect(hspeed(s)).toBeGreaterThan(HERO.glideSpeed * 0.95);
    expect(s.stamina).toBeCloseTo(1 - HERO.staminaGlide * (3.0 + DT), 2);
    expect(heroAction(s)).toBe('glide');
  });

  it('a second press folds it; it cannot open close to the ground; landing ends the glide', () => {
    const s = high();
    run(s, 0.2, input());
    updateHero(s, input({ jump: true }), DT, SEED);
    updateHero(s, input(), DT, SEED);
    updateHero(s, input({ jump: true }), DT, SEED);
    expect(s.move).toBe('air');
    const low = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    placeHero(low, low.x, low.ground + BIAS + 1.0, low.z, 0, 0, 0, 0, SEED);
    low.jumpHeld = false;
    updateHero(low, input({ jump: true }), DT, SEED);
    expect(low.move).toBe('air');
    const g = high();
    updateHero(g, input({ jump: true }), DT, SEED);
    run(g, 30, input());
    expect(g.move === 'ground' || g.move === 'swim' || g.move === 'climb').toBe(true);
  });

  it('a long drop (from fly mode) skydives at its terminal velocity and can still open the glider', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    placeHero(s, s.x, s.ground + 900, s.z, 0, 0, 0, 0, SEED);
    s.jumpHeld = false;
    let fastest = 0;
    run(s, 6, input(), () => { fastest = Math.max(fastest, -s.vy); });
    expect(s.skydive).toBe(true);
    expect(fastest).toBeLessThan(SKYDIVE.glideTerminal + 1e-6);
    expect(-s.vy).toBeCloseTo(SKYDIVE.glideTerminal, 2);
    updateHero(s, input({ jump: true }), DT, SEED);
    expect(s.move).toBe('glide');
    run(s, 3, input());
    expect(s.vy).toBeGreaterThan(-HERO.glideSink - 0.5);
  });

  it('folds when the wheel runs dry', () => {
    const s = high();
    s.stamina = 0.01;
    updateHero(s, input({ jump: true }), DT, SEED);
    expect(s.move).toBe('glide');
    run(s, 0.5, input());
    expect(s.exhausted).toBe(true);
    expect(s.move).toBe('air');
  });
});

// ---- climbing walls and rocks ----------------------------------------------------

const ROCKS_NEAR: RockBody[] = [];
/** The top of the solid at (x, z): the land, or a rock standing there (as the climb sees it). */
function solidAt(x: number, z: number): number {
  rockBodiesNear(x, z, SEED, 3, ROCKS_NEAR);
  let top = elevation(x, z, SEED, 0);
  for (const b of ROCKS_NEAR) { const r = rockTopAt(b, x, z); if (r > top) top = r; }
  return top;
}
/** A rock standing still on dry, gentle ground near the spawn, `lo`..`hi` m over the land, at least `wide` across. */
function findRock(lo: number, hi: number, wide: number): RockBody {
  const near: RockBody[] = [];
  for (let d = 0; d < 3000; d += 20) {
    for (let a = 0; a < 16; a++) {
      for (const b of rockBodiesNear(SPAWN.x + d * Math.cos(a * 0.39), SPAWN.z + d * Math.sin(a * 0.39), SEED, 0, near)) {
        const h = rockTopAt(b, b.x, b.z) - elevation(b.x, b.z, SEED, 0);
        if (b.state !== 'ground' || h < lo || h > hi || b.reach < wide || slope(b.x, b.z, SEED) > 0.2
          || elevation(b.x, b.z, SEED, 0) < waterLevel(b.x, b.z, SEED) + 2 || !clearOfTrees(b.x, b.z, b.reach + 4)) continue;
        return b;
      }
    }
  }
  throw new Error('no rock found');
}
const BOULDER = findRock(2.4, 6, 0.9);
const LOW_ROCK = findRock(0.6, 0.85, 0.6);

describe('Hero locomotion — climbing walls and rocks', () => {
  const atWall = () => createHeroState(WALL[0] - Math.sin(WALL_UP) * 0.3, WALL[1] - Math.cos(WALL_UP) * 0.3, WALL_UP, SEED);
  /** The hold's distance from the wall: where the solid's top stands at the root against the hips' height. */
  const offWall = (s: HeroState) => Math.abs(solidAt(s.x, s.z) - (s.y - BIAS + 0.9));

  it('pushing into a wall grabs it and climbs it at upSpeed along the surface, held where the hips meet it, facing it', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    expect(s.move).toBe('climb');
    expect(s.wall).toBe(true);
    let worst = 0;
    // (the speed eases in over the first half second: measured once it is up to pace)
    run(s, 0.6, () => input({ forward: 1, camYaw: s.yaw }), () => { worst = Math.max(worst, offWall(s)); });
    const y0 = s.y;
    run(s, 1.0, () => input({ forward: 1, camYaw: s.yaw }), () => { worst = Math.max(worst, offWall(s)); });
    expect(s.move).toBe('climb');
    // (a metre along a wall leaning in wallInv per metre up rises 1 / √(1 + wallInv²))
    expect((s.y - y0) * Math.sqrt(1 + s.wallInv * s.wallInv)).toBeCloseTo(CLIMB.upSpeed, 1);
    expect(worst).toBeLessThan(0.02);
    expect(Math.cos(s.yaw - WALL_UP)).toBeGreaterThan(0.95);
    expect(s.stamina).toBeCloseTo(1 - HERO.staminaClimb * 1.7, 1);
    expect(heroAction(s)).toBe('climb');
  });

  it('starting and stopping, the climb eases in and out: no jolt at either end', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    let worst = 0, last = NaN;
    const step = () => { const v = s.vy; if (last === last) worst = Math.max(worst, Math.abs(v - last)); last = v; };
    run(s, 1.0, () => input({ forward: 1, camYaw: s.yaw }), step);
    run(s, 1.0, () => input({ camYaw: s.yaw }), step);
    // (the speed changes by at most accel · upSpeed · dt a tick: no step from rest to full pace)
    expect(worst).toBeLessThan(CLIMB.accel * CLIMB.upSpeed * DT * 1.01);
    expect(heroAction(s)).toBe('hang');
  });

  it('across a wall along its level: the height kept, the hold on it all the way', () => {
    const s = atWall();
    climbFor(s, 1.0, { camYaw: WALL_UP });
    run(s, 0.8, () => input({ camYaw: s.yaw })); // (settled from the climb up)
    const y0 = s.y, x0 = s.x, z0 = s.z;
    let worst = 0;
    run(s, 1.0, () => input({ strafe: 1, camYaw: s.yaw }), () => { worst = Math.max(worst, offWall(s)); });
    expect(s.move).toBe('climb');
    expect(Math.abs(s.y - y0)).toBeLessThan(0.02);
    expect(worst).toBeLessThan(0.02);
    expect(Math.hypot(s.x - x0, s.z - z0)).toBeGreaterThan(0.6 * CLIMB.wrapSpeed);
  });

  it('hanging drains the hang rate; the sneak key or a dry wheel slides down the wall at slideSpeed, never drops', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    run(s, 0.5, () => input({ forward: 1, camYaw: s.yaw }));
    const st = s.stamina;
    run(s, 1.0, input());
    expect(s.move).toBe('climb');
    expect(heroAction(s)).toBe('hang');
    expect(st - s.stamina).toBeCloseTo(HERO.staminaHang, 2);
    const seen = new Set<string>();
    run(s, 0.6, () => input({ sneak: true, camYaw: s.yaw }), () => seen.add(s.move)); // (up to speed)
    const y0 = s.y;
    run(s, 0.3, () => input({ sneak: true, camYaw: s.yaw }), () => seen.add(s.move));
    // (along the surface at slideSpeed: down it at slideSpeed / √(1 + wallInv²))
    expect(((y0 - s.y) / 0.3) * Math.sqrt(1 + s.wallInv * s.wallInv)).toBeCloseTo(CLIMB.slideSpeed, 0);
    expect(seen.has('air')).toBe(false);
    const d = atWall();
    run(d, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    run(d, 0.6, () => input({ forward: 1, camYaw: d.yaw }));
    d.stamina = 0.001;
    seen.clear();
    run(d, 2.0, () => input({ forward: 1, camYaw: d.yaw }), () => seen.add(d.move));
    expect(d.exhausted).toBe(true);
    expect(d.sliding).toBe(true);
    expect(seen.has('air')).toBe(false);
    // exhausted heroes cannot grab a wall
    const e = atWall();
    e.stamina = 0; e.exhausted = true;
    run(e, 0.5, input({ forward: 1, camYaw: WALL_UP }));
    expect(e.move).not.toBe('climb');
  });

  it('a climb jump surges HERO.climbJump metres up the wall; with the stick pulled back the jump leaps off it', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    s.jumpHeld = false;
    const st = s.stamina, x0 = s.x, y0 = s.y, z0 = s.z;
    run(s, 0.5, (t) => input({ jump: t < 0.05, camYaw: s.yaw }));
    expect(Math.hypot(s.x - x0, s.y - y0, s.z - z0)).toBeGreaterThan(HERO.climbJump * 0.9);
    expect(st - s.stamina).toBeGreaterThan(HERO.staminaClimbJump);
    s.jumpHeld = false;
    updateHero(s, input({ forward: -1, jump: true, camYaw: s.yaw }), DT, SEED);
    expect(s.move).toBe('air');
    expect(s.wall).toBe(false);
    expect(-(s.vx * Math.sin(WALL_UP) + s.vz * Math.cos(WALL_UP))).toBeGreaterThan(0.9 * CLIMB.leapSpeed);
  });

  it('easing ground ends a climb; steep ground without a grab slides downhill', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    s.move = 'climb';
    updateHero(s, input(), DT, SEED);
    expect(s.move as string).toBe('ground');
    const w = atWall();
    const e0 = w.ground;
    run(w, 0.5, input({ forward: -1, camYaw: WALL_UP })); // pushing away from the wall
    expect(w.sliding || w.move === 'air').toBe(true);
    expect(w.ground).toBeLessThan(e0);
  });

  it('falling past a wall, pushing into it, catches it', () => {
    const f = createHeroState(WALL[0] - Math.sin(WALL_UP) * 1.2, WALL[1] - Math.cos(WALL_UP) * 1.2, WALL_UP, SEED);
    placeHero(f, f.x, f.y + 4, f.z, 0, 0, 0, WALL_UP, SEED);
    f.jumpHeld = false;
    let caught = false;
    run(f, 1.2, input({ forward: 1, camYaw: WALL_UP }), () => { if (f.move === 'climb') caught = true; });
    expect(caught).toBe(true);
    expect(f.move).toBe('climb');
    expect(offWall(f)).toBeLessThan(0.02);
  });

  it('runs into a boulder, climbs its face and vaults onto its top', () => {
    const ang = 0.8, yaw = ang + Math.PI;
    const s = createHeroState(BOULDER.x + Math.sin(ang) * (BOULDER.reach + 2.5), BOULDER.z + Math.cos(ang) * (BOULDER.reach + 2.5), yaw, SEED);
    let climbed = false, vaulted = false, worst = 0, top = -Infinity;
    run(s, 6, () => input({ forward: s.move === 'ground' && vaulted ? 0 : 1, camYaw: yaw }), () => {
      if (s.move === 'climb' && s.wall && s.mantle === 0) { climbed = true; worst = Math.max(worst, offWall(s)); }
      if (s.mantle > 0) vaulted = true;
      if (vaulted && s.move === 'ground') top = Math.max(top, s.y - BIAS);
    });
    expect(climbed).toBe(true);
    expect(vaulted).toBe(true);
    expect(worst).toBeLessThan(0.02);
    expect(s.move).toBe('ground');
    // (standing on the rock: its top under the feet, well over the land)
    expect(Math.abs(top - rockTopAt(BOULDER, s.x, s.z))).toBeLessThan(0.05);
    expect(top - elevation(s.x, s.z, SEED, 0)).toBeGreaterThan(1.5);
  });

  it('climbing down a boulder steps off onto the ground at its foot', () => {
    const ang = 2.4, yaw = ang + Math.PI;
    const s = createHeroState(BOULDER.x + Math.sin(ang) * (BOULDER.reach + 1.5), BOULDER.z + Math.cos(ang) * (BOULDER.reach + 1.5), yaw, SEED);
    let up = 0;
    run(s, 1.6, () => input({ forward: s.wall && s.y - BIAS - elevation(s.x, s.z, SEED, 0) > -0.3 ? 0 : 1, camYaw: s.wall ? s.yaw : yaw }), () => { if (s.wall) up++; });
    expect(s.move).toBe('climb');
    expect(s.mantle).toBe(0);
    expect(up).toBeGreaterThan(10);
    run(s, 4.5, () => input({ forward: -1, camYaw: s.yaw }));
    expect(s.move).toBe('ground');
    expect(s.wall).toBe(false);
    expect(Math.abs(s.y - (s.ground + BIAS))).toBeLessThan(1e-9);
    // (at its foot: on the land, not up on the rock)
    expect(Math.abs(s.ground - elevation(s.x, s.z, SEED, 0))).toBeLessThan(0.3);
  });

  it('a low rock is vaulted straight onto, without climbing', () => {
    const ang = 1.1, yaw = ang + Math.PI;
    const s = createHeroState(LOW_ROCK.x + Math.sin(ang) * (LOW_ROCK.reach + 2), LOW_ROCK.z + Math.cos(ang) * (LOW_ROCK.reach + 2), yaw, SEED);
    let walled = false, vaulted = false, onTop = false;
    run(s, 2, () => input({ forward: vaulted && s.move === 'ground' ? 0 : 1, camYaw: yaw }), () => {
      if (s.wall) walled = true;
      if (s.mantle > 0) vaulted = true;
      if (vaulted && s.move === 'ground' && s.y - BIAS > elevation(s.x, s.z, SEED, 0) + 0.4) onTop = true;
    });
    expect(walled).toBe(false);
    expect(vaulted).toBe(true);
    expect(onTop).toBe(true);
  });
});

// ---- climbing trees ------------------------------------------------------------

/** A tree near the spawn on dry, gentle ground with room all round, as `pred` asks. */
function findTrunk(pred: (t: Trunk) => boolean): Trunk {
  const near: Trunk[] = [];
  for (let d = 0; d < 2500; d += 12) {
    for (let a = 0; a < 24; a++) {
      const x = SPAWN.x + d * Math.cos((a / 24) * Math.PI * 2), z = SPAWN.z + d * Math.sin((a / 24) * Math.PI * 2);
      for (const t of trunksNear(x, z, SEED, near, false)) {
        if (!t.prof || !pred(t)) continue;
        const ring = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => [t.x + (t.r + 1.5) * Math.cos(k * Math.PI / 4), t.z + (t.r + 1.5) * Math.sin(k * Math.PI / 4)]);
        if (ring.some(([px, pz]) => slope(px, pz, SEED) > 0.25 || elevation(px, pz, SEED, 0) < waterLevel(px, pz, SEED) + 2)) continue;
        // (no other trunk within reach of the climb)
        const others = trunksNear(t.x, t.z, SEED, [], false).filter((o) => o.x !== t.x || o.z !== t.z);
        if (others.some((o) => Math.hypot(o.x - t.x, o.z - t.z) < t.r + o.r + 3)) continue;
        return t;
      }
    }
  }
  throw new Error('no tree found');
}
const TREE = findTrunk((t) => t.top > 6 && t.top < 12 && t.r > 0.4 && t.r < 0.8);
const POLE = findTrunk((t) => t.top > 4 && t.r < 0.3);
const SECTION = new Float64Array(3);

/** A hero standing against `t`, facing its axis from the bearing `ang` (rad from +z), and the yaw facing it. */
function atTrunk(t: Trunk, ang: number): [HeroState, number] {
  const s = createHeroState(t.x + Math.sin(ang) * (t.r + 0.4), t.z + Math.cos(ang) * (t.r + 0.4), ang + Math.PI, SEED);
  return [s, ang + Math.PI];
}
/** Distance of the root off the bark of the trunk it climbs (0 on it). */
function offBark(s: HeroState, t: Trunk): number {
  trunkSection(t, s.y - BIAS - t.y + 0.9, SECTION);
  return Math.abs(Math.hypot(s.x - SECTION[0], s.z - SECTION[1]) - SECTION[2]);
}
/** Climb `t` from the ground for `seconds` with the stick toward it (the camera following the hero). */
function climbFor(s: HeroState, seconds: number, over: Partial<HeroInput> = {}): void {
  run(s, seconds, () => input({ forward: 1, camYaw: s.yaw, ...over }));
}

describe('Hero locomotion — climbing trees', () => {
  it('pushing into a trunk grabs it, climbs at upSpeed and holds the bark exactly, facing the axis', () => {
    const [s, yaw] = atTrunk(TREE, 0.4);
    run(s, 0.3, input({ forward: 1, camYaw: yaw }));
    expect(s.move).toBe('climb');
    expect(s.tree).not.toBeNull();
    const y0 = s.y;
    let worst = 0;
    run(s, 1.0, () => input({ forward: 1, camYaw: s.yaw }), () => { worst = Math.max(worst, offBark(s, TREE)); });
    expect(s.move).toBe('climb');
    expect((s.y - y0) / 1.0).toBeCloseTo(CLIMB.upSpeed, 1);
    expect(worst).toBeLessThan(1e-9);
    trunkSection(TREE, s.y - BIAS - TREE.y + 0.9, SECTION);
    expect(Math.cos(s.yaw - Math.atan2(SECTION[0] - s.x, SECTION[1] - s.z))).toBeGreaterThan(0.999999);
    expect(s.stamina).toBeCloseTo(1 - HERO.staminaClimb * 1.0, 1);
    expect(heroAction(s)).toBe('climb');
  });

  it('circles the trunk at wrapSpeed round the hips, with no drift off the bark however long', () => {
    const [s, yaw] = atTrunk(TREE, 1.1);
    run(s, 0.6, input({ forward: 1, camYaw: yaw }));
    run(s, 1.5, () => input({ strafe: 1, camYaw: s.yaw })); // (settled from the climb up, up to the wrap's pace)
    const y0 = s.y;
    s.stamina = 1;
    let turned = 0, worst = 0, last = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
    run(s, 8, () => input({ strafe: 1, camYaw: s.yaw }), () => {
      const a = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
      turned += Math.atan2(Math.sin(a - last), Math.cos(a - last));
      last = a;
      worst = Math.max(worst, offBark(s, TREE));
      s.stamina = 1; // (eight seconds of climbing: keep the wheel full)
    });
    expect(s.move).toBe('climb');
    expect(worst).toBeLessThan(1e-9);
    expect(s.y).toBeCloseTo(y0, 9);
    // (to the hero's right is clockwise from above)
    const ideal = (CLIMB.wrapSpeed / (s.climbR + climbClearance(s.climbR))) * 8;
    expect(-turned / ideal).toBeGreaterThan(0.97);
    expect(-turned / ideal).toBeLessThan(1.03);
  });

  it('round a thin pole it goes round at the wrap speed at the hips, never faster than wrapMax', () => {
    const [s, yaw] = atTrunk(POLE, 2.0);
    run(s, 0.6, input({ forward: 1, camYaw: yaw }));
    expect(s.move).toBe('climb');
    expect(s.climbR).toBeLessThan(CLIMB.poleWrap);
    run(s, 0.6, () => input({ strafe: -1, camYaw: s.yaw })); // (up to pace)
    const a0 = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
    run(s, 0.5, () => input({ strafe: -1, camYaw: s.yaw }));
    const a1 = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
    const rate = Math.abs(Math.atan2(Math.sin(a1 - a0), Math.cos(a1 - a0))) / 0.5;
    const ideal = Math.min(CLIMB.wrapMax, CLIMB.wrapSpeed / (s.climbR + climbClearance(s.climbR)));
    expect(rate).toBeLessThanOrEqual(CLIMB.wrapMax * 1.0001);
    expect(rate / ideal).toBeGreaterThan(0.95);
    expect(rate / ideal).toBeLessThan(1.05);
  });

  it('a run (or sprint) into a trunk grabs it at its foot and climbs from there; a climb jump surges up the bark', () => {
    for (const sprint of [false, true]) {
      const [s, yaw] = atTrunk(TREE, 2.6);
      placeHero(s, TREE.x + Math.sin(2.6) * (TREE.r + 6), s.y, TREE.z + Math.cos(2.6) * (TREE.r + 6), 0, 0, 0, yaw, SEED);
      let grabbed = NaN, ground = NaN, first = NaN;
      run(s, 2.0, input({ forward: 1, sprint, camYaw: yaw }), () => {
        if (s.move === 'climb' && grabbed !== grabbed) { grabbed = s.y; ground = s.ground; }
        // (the first ticks on the bark: climbing at the climb's pace, no boost)
        if (s.move === 'climb' && first !== first && s.y > grabbed) first = s.y - grabbed;
      });
      expect(s.move).toBe('climb');
      expect(grabbed - (ground + BIAS)).toBeLessThan(0.05); // (its feet on the ground under it)
      expect(first).toBeLessThan(CLIMB.upSpeed * DT * 1.01);
    }
    const [w] = atTrunk(TREE, 2.6);
    climbFor(w, 0.9);
    // a jump: HERO.climbJump metres up for its cost
    w.jumpHeld = false;
    const y0 = w.y, st = w.stamina;
    run(w, 0.5, (t) => input({ jump: t < 0.05, camYaw: w.yaw }));
    expect(w.y - y0).toBeGreaterThan(HERO.climbJump * 0.95);
    expect(st - w.stamina).toBeGreaterThan(HERO.staminaClimbJump);
  });

  it('a jump with the stick pulled back leaps off the trunk the way the stick points, and does not grab it again', () => {
    const [s] = atTrunk(TREE, 3.3);
    climbFor(s, 1.0);
    s.jumpHeld = false;
    const out = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
    // (the camera behind the hero, looking at the trunk: back is away from it, a little to the left)
    updateHero(s, input({ forward: -1, strafe: -0.3, jump: true, camYaw: s.yaw }), DT, SEED);
    expect(s.move).toBe('air');
    expect(s.tree).toBeNull();
    expect(Math.hypot(s.vx, s.vz)).toBeCloseTo(CLIMB.leapSpeed, 6);
    expect(Math.cos(Math.atan2(s.vx, s.vz) - out)).toBeGreaterThan(0.9);
    expect(s.vy).toBeCloseTo(CLIMB.leapUp, 6);
    run(s, 0.6, input({ forward: -1, camYaw: out + Math.PI }));
    expect(s.move).not.toBe('climb');
  });

  it('on the trunk the stick works on the hero, not the camera: forward climbs wherever the camera looks', () => {
    const [s] = atTrunk(TREE, 0.5);
    climbFor(s, 0.6);
    const y0 = s.y, a0 = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
    // (the camera off to the side, then behind the trunk: forward still climbs straight up)
    run(s, 1.0, (t) => input({ forward: 1, camYaw: t < 0.5 ? s.yaw + 1.4 : s.yaw + Math.PI }));
    expect((s.y - y0) / 1.0).toBeCloseTo(CLIMB.upSpeed, 1);
    expect(Math.abs(Math.atan2(s.x - s.climbX, s.z - s.climbZ) - a0)).toBeLessThan(1e-9);
  });

  it('the sneak key slides down the bark to the ground; a dry wheel slides too, never drops', () => {
    const [s] = atTrunk(TREE, 4.0);
    climbFor(s, 4.0);
    run(s, 0.7, () => input({ sneak: true, camYaw: s.yaw })); // (up to speed)
    const y0 = s.y;
    run(s, 0.15, () => input({ sneak: true, camYaw: s.yaw }));
    expect(s.move).toBe('climb');
    expect((y0 - s.y) / 0.15 / CLIMB.slideSpeed).toBeGreaterThan(0.97);
    expect((y0 - s.y) / 0.15 / CLIMB.slideSpeed).toBeLessThan(1.001);
    run(s, 2.0, () => input({ sneak: true, camYaw: s.yaw }));
    expect(s.move).toBe('ground');
    const [d] = atTrunk(TREE, 4.6);
    climbFor(d, 2.0);
    d.stamina = 0.001;
    const seen = new Set<string>();
    run(d, 2.5, () => input({ forward: 1, camYaw: d.yaw }), () => seen.add(d.move));
    expect(d.exhausted).toBe(true);
    expect(seen.has('air')).toBe(false);
    expect(d.move).toBe('ground');
  });

  it('jumping at a trunk catches it where the jump meets the bark, well up from its foot', () => {
    const [s] = atTrunk(TREE, 1.4);
    // (a running jump at the trunk from 3 m out, the stick toward it)
    const out = 1.4, yaw = out + Math.PI;
    placeHero(s, TREE.x + Math.sin(out) * (TREE.r + 3), s.y, TREE.z + Math.cos(out) * (TREE.r + 3), 0, 0, 0, yaw, SEED);
    run(s, 0.4, input({ forward: 1, camYaw: yaw }));
    s.jumpHeld = false;
    let caught = NaN;
    run(s, 1.0, (t) => input({ forward: 1, jump: t < 0.05, camYaw: yaw }), () => { if (s.move === 'climb' && caught !== caught) caught = s.y - s.ground; });
    expect(s.move).toBe('climb');
    expect(caught).toBeGreaterThan(0.5);
  });

  it('falling onto a trunk grabs it, sliding v²/2a down the bark (held to latchSlip) as the grip takes', () => {
    const ang = 5.2, h = TREE.top - 3;
    // (well below the top: beside the trunk's cross-section there, drifting into it)
    trunkSection(TREE, h + 0.9, SECTION);
    const s = createHeroState(SECTION[0] + Math.sin(ang) * (SECTION[2] + 0.45), SECTION[1] + Math.cos(ang) * (SECTION[2] + 0.45), ang + Math.PI, SEED);
    placeHero(s, s.x, TREE.y + h + BIAS, s.z, -Math.sin(ang) * 2, -6, -Math.cos(ang) * 2, ang + Math.PI, SEED);
    let at = NaN, v = 0;
    run(s, 0.6, () => input({ camYaw: s.yaw }), () => {
      if (s.move === 'climb' && at !== at) at = s.y;
      if (s.move !== 'climb') v = -s.vy;
    });
    expect(s.move).toBe('climb');
    const [lo, hi] = CLIMB.latchSlip;
    const slip = Math.min(hi, Math.max(lo, (v * v) / (2 * CLIMB.latchDecel)));
    expect(at - s.y).toBeCloseTo(slip, 2);
    expect(s.grip).toBe(0);
    expect(offBark(s, TREE)).toBeLessThan(1e-9);
  });

  it('at the top the hero vaults onto it and balances there; sneaking off its edge hangs on the bark below', () => {
    const [p] = atTrunk(TREE, 0.2);
    let vaulted = false;
    // (pushing on all the while: on top it balances at the edge instead of walking off)
    run(p, 12, () => input({ forward: 1, camYaw: p.yaw }), () => { p.stamina = 1; if (p.mantle > 0) vaulted = true; });
    expect(vaulted).toBe(true);
    expect(p.move).toBe('ground');
    expect(p.perch).not.toBeNull(); // (perched on this tree's top)
    expect(p.y).toBeCloseTo(TREE.y + TREE.top + BIAS, 6);
    trunkSection(TREE, TREE.top, SECTION);
    expect(Math.hypot(p.x - SECTION[0], p.z - SECTION[1])).toBeLessThan(CLIMB.perch);
    // sneaking toward the edge: over it, hanging on the bark just below the top
    run(p, 1.0, () => input({ forward: 1, sneak: true, camYaw: p.yaw }), () => { if (p.move === 'climb') p.stamina = 1; });
    expect(p.move).toBe('climb');
    expect(p.tree).not.toBeNull();
    expect(p.y).toBeLessThan(TREE.y + TREE.top - 1.5);
    expect(offBark(p, TREE)).toBeLessThan(1e-9);
  });

  it('climbing down steps off onto the ground', () => {
    const [c] = atTrunk(TREE, 1.9);
    climbFor(c, 1.5);
    run(c, 3, () => input({ forward: -1, camYaw: c.yaw }));
    expect(c.move).toBe('ground');
    expect(c.tree).toBeNull();
    expect(Math.abs(c.y - (c.ground + BIAS))).toBeLessThan(1e-9);
  });

  it('foes never grab trees', () => {
    const [s, yaw] = atTrunk(TREE, 0.9);
    run(s, 0.5, input({ forward: 1, camYaw: yaw, noClimb: true }));
    expect(s.move).toBe('ground');
  });
});

describe('Hero locomotion — swimming', () => {
  it('floats at the swim depth, swims at swimSpeed, dashes on stamina', () => {
    const s = createHeroState(DEEP[0], DEEP[1], 0, SEED);
    run(s, 3.0, input());
    expect(s.move).toBe('swim');
    expect(s.y).toBeCloseTo(waterLevel(s.x, s.z, SEED) - HERO.swimFloat, 1);
    const yaw = Math.atan2(DEEP[0] - SPAWN.x, DEEP[1] - SPAWN.z); // keep heading into the lake
    run(s, 0.8, input({ forward: 1, camYaw: yaw }));
    if (s.move === 'swim') expect(hspeed(s)).toBeCloseTo(HERO.swimSpeed, 0);
    const t = createHeroState(DEEP[0], DEEP[1], 0, SEED);
    run(t, 0.5, input({ forward: 1, sprint: true, camYaw: yaw }));
    expect(t.stamina).toBeLessThan(1 - HERO.staminaSwimDash * 0.4);
    expect(heroAction(t)).toBe('dash');
  });

  it('walks into deep water from the shore and wades back out', () => {
    // march from the lake bed towards the spawn (dry land) and back
    const s = createHeroState(DEEP[0], DEEP[1], 0, SEED);
    const toShore = Math.atan2(SPAWN.x - DEEP[0], SPAWN.z - DEEP[1]);
    const seen = new Set<string>();
    run(s, 40, input({ forward: 1, camYaw: toShore }), () => seen.add(s.move));
    expect(seen.has('swim')).toBe(true);
    expect(seen.has('ground')).toBe(true);
  });
});

describe('Hero locomotion — determinism and robustness', () => {
  const script = (t: number): HeroInput => input({
    forward: Math.sin(t * 1.3) > -0.3 ? 1 : 0,
    strafe: Math.sin(t * 0.7) > 0.6 ? 1 : Math.sin(t * 0.7) < -0.6 ? -1 : 0,
    sprint: Math.sin(t * 0.37) > 0.2,
    jump: Math.sin(t * 2.9) > 0.9,
    sneak: Math.sin(t * 0.23) > 0.85,
    camYaw: t * 0.21,
  });

  it('identical inputs give bit-identical trajectories', () => {
    const a = createHeroState(SPAWN.x, SPAWN.z, SPAWN.yaw, SEED);
    const b = createHeroState(SPAWN.x, SPAWN.z, SPAWN.yaw, SEED);
    run(a, 12, script);
    run(b, 12, script);
    for (const k of Object.keys(a) as Array<keyof HeroState>) expect(Object.is(a[k], b[k])).toBe(true);
  });

  it('stays finite and on or above the ground under fuzzed input, near and ±1,000 km out', () => {
    for (const [x, z] of [[SPAWN.x, SPAWN.z], [WALL[0], WALL[1]], [DEEP[0], DEEP[1]], [1_000_003.7, -999_871.2], [-731_000.5, 402_123.25]]) {
      const s = createHeroState(x, z, 0, SEED);
      let worst = 0;
      run(s, 20, script, () => {
        for (const v of [s.x, s.y, s.z, s.vx, s.vy, s.vz, s.yaw, s.stamina]) expect(Number.isFinite(v)).toBe(true);
        expect(s.stamina).toBeGreaterThanOrEqual(0);
        expect(s.stamina).toBeLessThanOrEqual(1);
        // (on a wall the root is its hold, where the wall meets the hips' level — in a slope leaning back,
        // over the feet's ground: held to the wall instead; a vault moves off it)
        if (s.move === 'climb' && s.wall) {
          if (s.mantle === 0) expect(Math.abs(solidAt(s.x, s.z) - (s.y - BIAS + 0.9))).toBeLessThan(0.05);
        } else if (s.move !== 'swim' && !(s.move === 'climb' && s.mantle > 0)) {
          worst = Math.min(worst, s.y - (elevation(s.x, s.z, SEED, 0) + BIAS));
        }
      });
      expect(worst).toBeGreaterThan(-1e-6);
    }
  });
});

/** Joint position in world space (fp64) from the rig's hero-local frame. */
function worldJoint(r: HeroRig, s: HeroState, j: number): [number, number, number] {
  const sy = Math.sin(s.yaw), cy = Math.cos(s.yaw);
  const x = r.local[3 * j], y = r.local[3 * j + 1], z = r.local[3 * j + 2];
  return [s.x + cy * x + sy * z, s.y - BIAS + y, s.z - sy * x + cy * z];
}
const dist = (r: HeroRig, a: number, b: number) =>
  Math.hypot(r.local[3 * a] - r.local[3 * b], r.local[3 * a + 1] - r.local[3 * b + 1], r.local[3 * a + 2] - r.local[3 * b + 2]);

describe('Hero rig (procedural animation)', () => {
  it('keeps bone lengths in every settled pose', () => {
    const cases: Array<[string, () => HeroState]> = [
      ['ground', () => createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED)],
      ['air', () => { const s = createHeroState(FLAT[0], FLAT[1], 0, SEED); placeHero(s, s.x, s.ground + 20, s.z, 0, -3, 0, 0, SEED); return s; }],
      ['glide', () => { const s = createHeroState(FLAT[0], FLAT[1], 0, SEED); placeHero(s, s.x, s.ground + 20, s.z, 0, 0, 0, 0, SEED); s.move = 'glide'; return s; }],
      ['climb', () => { const s = createHeroState(WALL[0], WALL[1], WALL_UP, SEED); s.move = 'climb'; return s; }],
      ['swim', () => createHeroState(DEEP[0], DEEP[1], 0, SEED)],
    ];
    for (const [name, make] of cases) {
      const s = make();
      const r = createHeroRig();
      for (let i = 0; i < 90; i++) updateHeroRig(r, s, DT);
      expect(r.weights[['ground', 'air', 'glide', 'climb', 'swim'].indexOf(name)]).toBeGreaterThan(0.999);
      for (const v of r.local) expect(Number.isFinite(v)).toBe(true);
      for (const [a, b, len] of [
        [J.hipL, J.kneeL, HERO_SHAPE.thigh], [J.kneeL, J.ankleL, HERO_SHAPE.shin],
        [J.hipR, J.kneeR, HERO_SHAPE.thigh], [J.kneeR, J.ankleR, HERO_SHAPE.shin],
        [J.shoulderL, J.elbowL, HERO_SHAPE.upperArm], [J.elbowL, J.handL, HERO_SHAPE.forearm],
        [J.shoulderR, J.elbowR, HERO_SHAPE.upperArm], [J.elbowR, J.handR, HERO_SHAPE.forearm],
        [J.pelvis, J.chest, HERO_SHAPE.spine],
      ] as const) {
        expect(Math.abs(dist(r, a, b) - len)).toBeLessThan(name === 'ground' ? 0.01 : 2e-3);
      }
    }
  });

  it('stance feet stay planted while running (the gait moves them back at ground speed)', () => {
    const s = createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED);
    const r = createHeroRig();
    const inp = input({ forward: 1, camYaw: FLAT_YAW });
    for (let i = 0; i < 90; i++) { updateHero(s, inp, DT, SEED); updateHeroRig(r, s, DT); }
    // Track the left ankle through one stance: horizontal drift while it is down.
    let drift = 0, samples = 0;
    let prev: [number, number, number] | null = null;
    for (let i = 0; i < 60; i++) {
      updateHero(s, inp, DT, SEED);
      updateHeroRig(r, s, DT);
      const a = worldJoint(r, s, J.ankleL);
      const ground = elevation(a[0], a[2], SEED, 0);
      const down = a[1] - ground < HERO_SHAPE.ankleH + 0.02;
      if (down && prev) { drift += Math.hypot(a[0] - prev[0], a[2] - prev[2]); samples++; }
      prev = down ? a : null;
    }
    expect(samples).toBeGreaterThan(5);
    // the hero covers ≈ 9 cm per frame; a planted foot moves ≲ 1 cm per frame
    expect(drift / samples).toBeLessThan(0.015);
  });

  it('on a trunk: spec-length strides, planted holds put in the world, one diagonal pair always on the bark, limbs in reach and off it', () => {
    const ENDS = [J.handL, J.handR, J.ankleL, J.ankleR];
    const ROOTS = [J.shoulderL, J.shoulderR, J.hipL, J.hipR];
    const LEN = [0, 1, 2, 3].map((i) => (i < 2 ? HERO_SHAPE.upperArm + HERO_SHAPE.forearm : HERO_SHAPE.thigh + HERO_SHAPE.shin));
    for (const t of [TREE, POLE]) {
      const [s, yaw] = atTrunk(t, 0.7);
      const r = createHeroRig();
      run(s, 0.5, input({ forward: 1, camYaw: yaw }), () => updateHeroRig(r, s, DT));
      expect(s.move).toBe('climb');
      // up, round to the right, round to the left and down, hanging between
      const script = (k: number) => k < 90 ? input({ forward: 1, camYaw: s.yaw }) : k < 200 ? input({ strafe: 1, camYaw: s.yaw })
        : k < 240 ? input({ camYaw: s.yaw }) : k < 330 ? input({ strafe: -1, forward: -0.6, camYaw: s.yaw }) : input({ forward: -1, camYaw: s.yaw });
      let prev: number[][] | null = null, prevPlanted: boolean[] = [];
      let slid = 0, planted = 0, bothOff = 0, steps = 0, straight = 0, samples = 0, worstBark = Infinity, worstBone = 0;
      const strides: number[] = [], wasU = [1, 1, 1, 1];
      for (let k = 0; k < 380 && s.move === 'climb'; k++) {
        s.stamina = 1;
        updateHero(s, script(k), DT, SEED);
        updateHeroRig(r, s, DT);
        if (s.move !== 'climb' || r.weights[3] < 0.999) { prev = null; continue; }
        const at = ENDS.map((j) => worldJoint(r, s, j));
        // (HeroRig.limbs: per limb 7 values, the step's progress last; off the bark from the peel to the contact)
        const u = [0, 1, 2, 3].map((i) => r.limbs[7 * i + 6]);
        const isPlanted = u.map((v) => v >= 1);
        const off = u.map((v) => v >= 0.18 && v < 0.85);
        if (off[0] && off[1]) bothOff++; // (left and right hand are in different pairs)
        samples++;
        if ([0, 1, 2, 3].some((i) => dist(r, ROOTS[i], ENDS[i]) / LEN[i] > 0.999)) straight++;
        for (let i = 0; i < 4; i++) {
          if (!isPlanted[i]) steps++;
          // (a hand's step climbing straight up: from the hold it left to the one it took — limbs[7i..] start, [7i+3..] hold)
          if (i < 2 && k < 90 && isPlanted[i] && wasU[i] < 1) {
            const g = r.limbs, b = 7 * i;
            strides.push(Math.hypot(g[b + 3] - g[b], g[b + 4] - g[b + 1], g[b + 5] - g[b + 2]));
          }
          wasU[i] = u[i];
          if (prev && isPlanted[i] && prevPlanted[i]) {
            planted++;
            if (Math.hypot(at[i][0] - prev[i][0], at[i][1] - prev[i][1], at[i][2] - prev[i][2]) > 1e-3) slid++;
          }
          // (against the trunk's real cross-section at the hold's height)
          trunkSection(t, at[i][1] - t.y, SECTION);
          worstBark = Math.min(worstBark, Math.hypot(at[i][0] - SECTION[0], at[i][2] - SECTION[1]) - SECTION[2]);
        }
        for (const [a, b, len] of [[J.hipL, J.kneeL, HERO_SHAPE.thigh], [J.kneeL, J.ankleL, HERO_SHAPE.shin],
          [J.shoulderR, J.elbowR, HERO_SHAPE.upperArm], [J.elbowR, J.handR, HERO_SHAPE.forearm], [J.pelvis, J.chest, HERO_SHAPE.spine]] as const) {
          worstBone = Math.max(worstBone, Math.abs(dist(r, a, b) - len));
        }
        prev = at; prevPlanted = isPlanted;
      }
      expect(planted).toBeGreaterThan(400);
      expect(steps).toBeGreaterThan(100);
      // The spec's strides: a hand steps once 0.52 m from its socket to 0.26 m past it (the body moving on
      // through the step); none slides once planted, one pair always holds, the body follows its holds so no
      // arm or leg is pulled straight, and every hand and ankle stays outside the bark.
      const mean = strides.reduce((a, b) => a + b, 0) / strides.length;
      expect(strides.length).toBeGreaterThan(2);
      expect(mean).toBeGreaterThan(CLIMB.handStretch + 0.5 * CLIMB.lead);
      expect(slid / planted).toBeLessThan(0.002);
      expect(bothOff).toBe(0);
      expect(straight / samples).toBeLessThan(0.005);
      expect(worstBark).toBeGreaterThan(0);
      expect(worstBone).toBeLessThan(2e-3);
    }
  });

  it('on a wall and a boulder: the same gait — holds put in the world, one pair always on the surface, limbs in reach and out of the solid', () => {
    const ENDS = [J.handL, J.handR, J.ankleL, J.ankleR];
    const ROOTS = [J.shoulderL, J.shoulderR, J.hipL, J.hipR];
    const LEN = [0, 1, 2, 3].map((i) => (i < 2 ? HERO_SHAPE.upperArm + HERO_SHAPE.forearm : HERO_SHAPE.thigh + HERO_SHAPE.shin));
    const cases: Array<[HeroState, number]> = [
      [createHeroState(WALL[0] - Math.sin(WALL_UP) * 0.3, WALL[1] - Math.cos(WALL_UP) * 0.3, WALL_UP, SEED), WALL_UP],
      [createHeroState(BOULDER.x + Math.sin(2.3) * (BOULDER.reach + 1), BOULDER.z + Math.cos(2.3) * (BOULDER.reach + 1), 2.3 + Math.PI, SEED), 2.3 + Math.PI],
    ];
    for (const [s, yaw] of cases) {
      const r = createHeroRig();
      // up, across each way, hanging, down (on the boulder briefly up: its top is near)
      const script = (k: number) => k < 20 ? input({ forward: 1, camYaw: yaw }) : k < 80 ? input({ strafe: 1, camYaw: s.yaw })
        : k < 140 ? input({ strafe: -1, camYaw: s.yaw }) : k < 170 ? input({ camYaw: s.yaw }) : input({ forward: -1, camYaw: s.yaw });
      let prev: number[][] | null = null, prevPlanted: boolean[] = [];
      let slid = 0, planted = 0, bothOff = 0, straight = 0, samples = 0, inside = 0;
      for (let k = 0; k < 220; k++) {
        s.stamina = 1;
        updateHero(s, script(k), DT, SEED);
        updateHeroRig(r, s, DT);
        if (s.move !== 'climb' || s.mantle > 0 || r.weights[3] < 0.999) { prev = null; continue; }
        const at = ENDS.map((j) => worldJoint(r, s, j));
        const u = [0, 1, 2, 3].map((i) => r.limbs[7 * i + 6]);
        const isPlanted = u.map((v) => v >= 1);
        if (u[0] >= 0.18 && u[0] < 0.85 && u[1] >= 0.18 && u[1] < 0.85) bothOff++;
        samples++;
        if ([0, 1, 2, 3].some((i) => dist(r, ROOTS[i], ENDS[i]) / LEN[i] > 0.999)) straight++;
        for (let i = 0; i < 4; i++) {
          if (prev && isPlanted[i] && prevPlanted[i]) {
            planted++;
            if (Math.hypot(at[i][0] - prev[i][0], at[i][1] - prev[i][1], at[i][2] - prev[i][2]) > 1e-3) slid++;
          }
          if (solidAt(at[i][0], at[i][2]) > at[i][1] + 0.02) inside++;
        }
        prev = at; prevPlanted = isPlanted;
      }
      expect(samples).toBeGreaterThan(80);
      expect(planted).toBeGreaterThan(150);
      expect(slid).toBe(0);
      expect(bothOff).toBe(0);
      expect(straight / samples).toBeLessThan(0.005);
      expect(inside).toBe(0);
    }
  });

  it('climbing moves the body smoothly: starts, stops and steps sway it, never jolt it', () => {
    const [s, yaw] = atTrunk(TREE, 1.7);
    const r = createHeroRig();
    run(s, 0.5, input({ forward: 1, camYaw: yaw }), () => updateHeroRig(r, s, DT));
    // (up, a stop, across, a stop, across and up, down: the stick on and off)
    const script = (k: number) => input({
      forward: k < 90 ? 1 : k < 230 ? 0 : k < 310 ? 0.5 : -1, strafe: k >= 120 && k < 200 ? 1 : k >= 230 && k < 310 ? -1 : 0, camYaw: s.yaw,
    });
    const hist: number[][][] = [], acc: number[] = [];
    for (let k = 0; k < 380; k++) {
      s.stamina = 1;
      updateHero(s, script(k), DT, SEED);
      updateHeroRig(r, s, DT);
      if (s.move !== 'climb' || r.weights[3] < 0.999) { hist.length = 0; continue; }
      hist.push([J.pelvis, J.chest, J.head].map((j) => worldJoint(r, s, j)));
      if (hist.length < 3) continue;
      const [a, b, c] = hist.slice(-3);
      for (let i = 0; i < 3; i++) acc.push(Math.hypot(a[i][0] - 2 * b[i][0] + c[i][0], a[i][1] - 2 * b[i][1] + c[i][1], a[i][2] - 2 * b[i][2] + c[i][2]));
    }
    acc.sort((x, y) => x - y);
    expect(acc.length).toBeGreaterThan(900);
    // (the body's acceleration in the world, m per frame²: a sway's, not a jolt's — before the springs and the
    // eased pace, p95 0.029 and peaks of 0.07 on a tree climb like this one)
    expect(acc[Math.floor(0.95 * (acc.length - 1))]).toBeLessThan(0.015);
    expect(acc[acc.length - 1]).toBeLessThan(0.06);
  });

  it('round a thin pole the hands reach far round its sides; vaulting onto a top keeps the bones whole', () => {
    /** How far round the trunk's axis the hands hold (rad off the root's bearing), the hero hanging still. */
    const handsRound = (t: Trunk): number => {
      const [s, yaw] = atTrunk(t, 2.2);
      const r = createHeroRig();
      run(s, 0.4, input({ forward: 1, camYaw: yaw }), () => updateHeroRig(r, s, DT));
      run(s, 1.0, input({ camYaw: s.yaw }), () => updateHeroRig(r, s, DT));
      expect(s.move).toBe('climb');
      const ang = Math.atan2(s.x - s.climbX, s.z - s.climbZ);
      let least = Infinity;
      for (const j of [J.handL, J.handR]) {
        const [x, , z] = worldJoint(r, s, j);
        const a = Math.atan2(x - s.climbX, z - s.climbZ);
        least = Math.min(least, Math.abs(Math.atan2(Math.sin(a - ang), Math.cos(a - ang))));
      }
      return least;
    };
    // (on a wide trunk ≈ 0.2 m to either side; round a pole most of the way to its sides, hugging it)
    expect(handsRound(POLE)).toBeGreaterThan(0.9);
    const [v] = atTrunk(TREE, 0.2);
    const rv = createHeroRig();
    let vaulting = 0, up = false;
    run(v, 12, () => input({ forward: up ? 0 : 1, camYaw: v.yaw }), () => {
      v.stamina = 1;
      updateHeroRig(rv, v, DT);
      if (v.move === 'ground' && v.y > TREE.y + TREE.top - 0.5) up = true;
      if (v.mantle <= 0) return;
      vaulting++;
      for (const x of rv.local) expect(Number.isFinite(x)).toBe(true);
      for (const [a, b, len] of [[J.hipR, J.kneeR, HERO_SHAPE.thigh], [J.elbowL, J.handL, HERO_SHAPE.forearm]] as const) {
        expect(Math.abs(dist(rv, a, b) - len)).toBeLessThan(2e-3);
      }
    });
    expect(vaulting).toBeGreaterThan(30);
  });

  it('idle feet rest on the terrain even on a slope (≤ 3 cm off the true surface)', () => {
    // (dry land: a hero on a slope under water swims, and its feet rest on nothing)
    const slopeSpot = findSpot((x, z) => { const m = slope(x, z, SEED); return m > 0.5 && m < 0.8 && elevation(x, z, SEED, 0) > waterLevel(x, z, SEED) + 2.9; }, 3);
    for (const yaw of [0, 1.3, 2.9, 4.4]) {
      const s = createHeroState(slopeSpot[0], slopeSpot[1], yaw, SEED);
      const r = createHeroRig();
      for (let i = 0; i < 60; i++) updateHeroRig(r, s, DT);
      for (const j of [J.ankleL, J.ankleR]) {
        const a = worldJoint(r, s, j);
        expect(Math.abs(a[1] - HERO_SHAPE.ankleH - elevation(a[0], a[2], SEED, 0))).toBeLessThan(0.03);
      }
    }
  });

  it('the bounding sphere holds every joint with its primitive reach, in every state', () => {
    const s = createHeroState(SPAWN.x, SPAWN.z, 0, SEED);
    const r = createHeroRig();
    let checked = 0;
    run(s, 16, (t) => input({ forward: 1, sprint: t % 4 < 1, jump: t % 3 < 0.1, camYaw: t * 0.3 }), () => {
      updateHeroRig(r, s, DT);
      if (Math.floor(s.x * 7) % 5 === 0 && s.move === 'air') { s.move = 'glide'; } // exercise the canopy too
      const [cx, cy, cz, rad] = r.bound;
      for (let j = 0; j < HERO_JOINTS; j++) {
        if (j === J.glider && r.glide === 0) continue;
        const pad = j === J.head ? HERO_SHAPE.headReach : j === J.pelvis || j === J.chest ? HERO_SHAPE.backReach
          : j === J.glider ? HERO_SHAPE.gliderReach * r.glide : HERO_SHAPE.limbPad;
        const d = Math.hypot(r.local[3 * j] - cx, r.local[3 * j + 1] - cy, r.local[3 * j + 2] - cz);
        expect(d + pad).toBeLessThanOrEqual(rad + 1e-9);
        checked++;
      }
    });
    expect(checked).toBeGreaterThan(900 * 17);
  });

  it('packs orthonormal frames and camera-relative joints with sub-millimetre precision 1,000 km out', () => {
    for (const [x, z] of [[FLAT[0], FLAT[1]], [1_000_000.37, -1_000_000.61]]) {
      const s = createHeroState(x, z, 0.7, SEED);
      const r = createHeroRig();
      run(s, 0.8, input({ forward: 1, camYaw: 0.7 }), () => updateHeroRig(r, s, DT));
      const cam = [x - 3.1, s.y + 1.9, z - 2.7] as const;
      const out = new Float32Array(HERO_UNIFORM.length);
      packHeroUniforms(r, s, cam[0], cam[1], cam[2], out);
      for (let j = 0; j < HERO_JOINTS; j++) {
        const w = worldJoint(r, s, j);
        for (let c = 0; c < 3; c++) expect(Math.abs(out[3 * j + c] - (w[c] - cam[c]))).toBeLessThan(1e-4);
      }
      for (const base of [HERO_UNIFORM.torso, HERO_UNIFORM.head]) {
        const col = (c: number) => [out[base + 3 * c], out[base + 3 * c + 1], out[base + 3 * c + 2]];
        const [rt, up, fw] = [col(0), col(1), col(2)];
        const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for (const v of [rt, up, fw]) expect(dot(v, v)).toBeCloseTo(1, 5);
        expect(dot(rt, up)).toBeCloseTo(0, 5);
        expect(dot(up, fw)).toBeCloseTo(0, 5);
        expect(dot(rt, fw)).toBeCloseTo(0, 5);
        // right = up × forward (the engine's left-handed x-right, y-up, z-forward basis)
        expect(dot(rt, [up[1] * fw[2] - up[2] * fw[1], up[2] * fw[0] - up[0] * fw[2], up[0] * fw[1] - up[1] * fw[0]])).toBeCloseTo(1, 5);
      }
      expect(out[HERO_UNIFORM.bound + 3]).toBeGreaterThan(0.8);
    }
  });
});

describe('Hero WGSL packing (hero.wgsl.ts)', () => {
  it("the parts' box holds every joint padded by its reach, inside the bounding sphere's cube", () => {
    const poses: Array<() => HeroState> = [
      () => createHeroState(FLAT[0], FLAT[1], FLAT_YAW, SEED),
      () => { const s = createHeroState(FLAT[0], FLAT[1], 0, SEED); placeHero(s, s.x, s.ground + 20, s.z, 0, 0, 0, 0, SEED); s.move = 'glide'; return s; },
      () => { const s = createHeroState(WALL[0], WALL[1], WALL_UP, SEED); s.move = 'climb'; return s; },
      () => createHeroState(DEEP[0], DEEP[1], 0, SEED),
    ];
    for (const make of poses) {
      const s = make();
      const r = createHeroRig();
      for (let i = 0; i < 90; i++) updateHeroRig(r, s, DT);
      const src = new Float32Array(HERO_UNIFORM.length);
      packHeroUniforms(r, s, s.x + 3, s.y + 2, s.z - 4, src);
      const out = new Float32Array(HERO_BYTES / 4);
      packHeroWgsl(src, out);
      const glide = src[HERO_UNIFORM.misc];
      const b = HERO_UNIFORM.bound;
      for (let j = 0; j < (glide > 0 ? HERO_JOINTS : HERO_JOINTS - 1); j++) {
        const pad = j === J.head ? HERO_SHAPE.headReach : j === J.pelvis || j === J.chest ? HERO_SHAPE.backReach
          : j === J.glider ? HERO_SHAPE.gliderReach * glide : HERO_SHAPE.limbPad;
        for (let c = 0; c < 3; c++) {
          const x = src[3 * j + c];
          expect(x - pad).toBeGreaterThanOrEqual(out[HERO_BOX + c] - 1e-5);
          expect(x + pad).toBeLessThanOrEqual(out[HERO_BOX + 4 + c] + 1e-5);
        }
      }
      for (let c = 0; c < 3; c++) {
        expect(out[HERO_BOX + c]).toBeGreaterThanOrEqual(src[b + c] - src[b + 3] - 1e-4);
        expect(out[HERO_BOX + 4 + c]).toBeLessThanOrEqual(src[b + c] + src[b + 3] + 1e-4);
      }
    }
  });
});

describe('Hero third-person camera', () => {
  it('orbits at the requested distance and aims at the look-at point when unobstructed', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    const c = createHeroCamera();
    for (const [yaw, pitch] of [[0, -0.24], [2.0, -0.6], [4.0, -0.9]]) {
      c.snap = true;
      updateHeroCamera(c, s, yaw, pitch, DT, SEED);
      const tx = s.x, ty = s.y - BIAS + HERO_CAMERA.targetH, tz = s.z;
      const d = Math.hypot(c.x - tx, c.y - ty, c.z - tz);
      if (c.boom === HERO_CAMERA.distance && Math.abs(c.pitch - pitch) < 1e-9) expect(d).toBeCloseTo(HERO_CAMERA.distance, 6);
      const fx = Math.sin(c.yaw) * Math.cos(c.pitch), fy = Math.sin(c.pitch), fz = Math.cos(c.yaw) * Math.cos(c.pitch);
      expect((fx * (tx - c.x) + fy * (ty - c.y) + fz * (tz - c.z)) / d).toBeCloseTo(1, 9);
    }
  });

  it('never goes under the terrain or the water, and the boom never collapses below minDistance', () => {
    let n = 0;
    for (const spot of [WALL, FLAT, DEEP, [SPAWN.x, SPAWN.z] as [number, number]]) {
      const s = createHeroState(spot[0], spot[1], 0, SEED);
      const c = createHeroCamera();
      for (let k = 0; k < 120; k++) {
        const yaw = k * 0.37, pitch = HERO_CAMERA.pitchMin + ((k * 0.61) % (HERO_CAMERA.pitchMax - HERO_CAMERA.pitchMin));
        updateHeroCamera(c, s, yaw, pitch, DT, SEED);
        expect(c.y).toBeGreaterThanOrEqual(elevation(c.x, c.z, SEED, 0) + HERO_CAMERA.clearance - 1e-9);
        expect(c.y).toBeGreaterThanOrEqual(waterLevel(c.x, c.z, SEED) + HERO_CAMERA.clearance - 1e-9);
        expect(c.boom).toBeGreaterThanOrEqual(HERO_CAMERA.minDistance - 1e-9);
        n++;
      }
    }
    expect(n).toBe(480);
  });

  it('trails a fast fall by at most maxLag, so the hero stays in frame', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    placeHero(s, s.x, s.ground + 80, s.z, 0, 0, 0, 0, SEED);
    const c = createHeroCamera();
    let worst = 0;
    run(s, 2.2, input({ sprint: true }), () => { // (a nosedive: the fastest fall)
      updateHeroCamera(c, s, 0, -0.2, DT, SEED);
      worst = Math.max(worst, Math.abs(c.ty - (s.y - BIAS + HERO_CAMERA.targetH)));
    });
    expect(s.vy).toBeLessThan(-30);
    expect(worst).toBeLessThanOrEqual(HERO_CAMERA.maxLag + 1e-9);
  });
});

describe('Hero ↔ shader contract (demoShader.ts heroMap)', () => {
  const glsl = new Map<string, number>();
  for (const m of TERRAIN_FS.matchAll(/const\s+(?:float|int)\s+([^;]+);/g)) {
    for (const decl of m[1].split(',')) {
      const [name, expr] = decl.split('=').map((x) => x.trim());
      if (name && expr && /^[\d.eE+-]+$/.test(expr)) glsl.set(name, Number(expr));
    }
  }
  it.each([
    ['HERO_LIMB_PAD', HERO_SHAPE.limbPad], ['HERO_HEAD_REACH', HERO_SHAPE.headReach],
    ['HERO_BACK_REACH', HERO_SHAPE.backReach], ['HERO_GLIDER_REACH', HERO_SHAPE.gliderReach],
  ] as const)('%s mirrors HERO_SHAPE', (name, value) => {
    expect(glsl.get(name)).toBe(value);
  });

  it('uniform layout: 18 joints, two mat3 frames, bound, misc and dress vec4s, the gear', () => {
    expect(TERRAIN_FS).toContain(`uniform vec3  u_hero_j[${HERO_JOINTS}];`);
    expect<number>(HERO_UNIFORM.torso).toBe(3 * HERO_JOINTS);
    expect(HERO_UNIFORM.head - HERO_UNIFORM.torso).toBe(9);
    expect(HERO_UNIFORM.bound - HERO_UNIFORM.head).toBe(9);
    expect(HERO_UNIFORM.misc - HERO_UNIFORM.bound).toBe(4);
    expect(HERO_UNIFORM.dress - HERO_UNIFORM.misc).toBe(4);
    expect(HERO_UNIFORM.gear - HERO_UNIFORM.dress).toBe(4);
    expect(HERO_UNIFORM.length - HERO_UNIFORM.gear).toBe(GEAR.length);
    expect(TERRAIN_FS).toContain('uniform vec3  u_hero_dress;');
    expect(TERRAIN_FS).toContain(`uniform vec4  u_hero_gear[${GEAR.length / 4}];`);
    expect(TERRAIN_FS).toContain(`u_hero_j[${J.glider}]`);
  });

  it('the hero is gated on the bound radius, so first person runs the unchanged terrain path', () => {
    expect(TERRAIN_FS).toContain('float heroT = u_hero_bound.w > 0.0 ? heroTrace(rd) : -1.0;');
    expect(TERRAIN_FS).toContain('float tMax = heroT > 0.0 ? heroT : MAXDIST;');
    // without a hero the march ends exactly where it used to (t > MAXDIST)
    expect(TERRAIN_FS).toContain('if (tMax >= MAXDIST || tPrev >= tMax) break;');
  });

  it('each expensive function keeps a single call site (inlining sets compile time and registers)', () => {
    const body = TERRAIN_FS.slice(TERRAIN_FS.indexOf('void main() {'));
    const calls = (name: string) => (body.match(new RegExp(`\\b${name}\\(`, 'g')) ?? []).length;
    expect(calls('softShadow')).toBe(1);
    expect(calls('heroShadow')).toBe(1);
    expect(calls('heroTrace')).toBe(1);
    // the march loop's three samples + bisection, and the one water-plane probe
    expect(calls('elevation')).toBe(5);
  });

  it('the joint array is only ever indexed by constants (a dynamic index cost the program 8 %)', () => {
    const uses = [...minifyGlsl(TERRAIN_FS).matchAll(/u_hero_j\[([^\]]+)\]/g)].map((m) => m[1]);
    expect(uses.length).toBeGreaterThan(40);
    for (const idx of uses) expect(idx === String(HERO_JOINTS) || /^\d+$/.test(idx)).toBe(true);
  });

  it('first person and fly compile no hero code: everything hero-related sits under #ifdef HERO', () => {
    // Minimal preprocessor for the one define (the blocks are not nested).
    const preprocess = (src: string, hero: boolean) => {
      const out: string[] = [];
      let keep = true;
      for (const line of src.split('\n')) {
        const t = line.trim();
        if (t === '#ifdef HERO') { keep = hero; continue; }
        if (t === '#else' && !keep === hero) { keep = !hero; continue; }
        if (t === '#else') { keep = !keep; continue; }
        if (t.startsWith('#endif')) { keep = true; continue; }
        if (keep) out.push(line);
      }
      return out.join('\n');
    };
    expect(TERRAIN_FS).not.toContain('#define HERO');
    expect(withHero(TERRAIN_FS).split('\n')[1]).toBe('#define HERO 1');
    const terrain = minifyGlsl(preprocess(TERRAIN_FS, false)); // code only, no comments
    for (const fn of ['heroMap', 'heroTrace', 'heroShadow', 'heroShade', 'heroNormal', 'heroContactAO', 'heroAlbedo']) {
      expect(terrain).not.toContain(`${fn}(`);
    }
    expect(terrain).toContain('const float heroT = -1.0;');
    const hero = minifyGlsl(preprocess(TERRAIN_FS, true));
    for (const fn of ['heroTrace(rd)', 'heroShadow(', 'heroShade(', 'heroNormal(', 'heroContactAO(']) expect(hero).toContain(fn);
  });

  it('every benchmark shader transform still applies to the extended shader', () => {
    for (const fs of [TERRAIN_FS, withHero(TERRAIN_FS)]) {
      for (const kind of ['render', 'instrument', 'probe', 'evalcost', 'truth'] as const) {
        expect(transformShader(fs, kind).length).toBeGreaterThan(fs.length * 0.9);
      }
    }
  });
});
