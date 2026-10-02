import { describe, it, expect } from 'bun:test';
import { elevation, slope } from '../src/lib/terrainMath';
import { DEFAULT_SEED, HERO, HERO_CAMERA, HERO_SHAPE, PLAYER, SPAWN, VALLEY } from '../src/lib/constants';
import {
  createHeroState, heroAction, placeHero, updateHero, type HeroInput, type HeroState,
} from '../src/lib/hero/locomotion';
import { createHeroRig, updateHeroRig, packHeroUniforms, J, HERO_JOINTS, HERO_UNIFORM, type HeroRig } from '../src/lib/hero/rig';
import { createHeroCamera, updateHeroCamera } from '../src/lib/hero/camera';
import { TERRAIN_FS, minifyGlsl, withHero } from '../src/components/terrain/demoShader';
import { HERO_BOX, HERO_BYTES, packHeroWgsl } from '../src/lib/gpu/hero.wgsl';
import { transformShader } from '../bench/gpu/runner';

/**
 * Third-person hero: locomotion (run / sprint / jump / glide / climb / swim /
 * stamina), the procedural rig, the orbit camera, and the contract with the
 * shader's heroMap(). Everything runs on the real terrain around the spawn.
 */

const SEED = DEFAULT_SEED;
const DT = 1 / 60;
const BIAS = PLAYER.groundBias;

function findSpot(pred: (x: number, z: number) => boolean, step = 3, radius = 700): [number, number] {
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

const FLAT = findSpot((x, z) => slope(x, z, SEED) < 0.08 && elevation(x, z, SEED, 0) > VALLEY.wLevel + 3, 4);
const FLAT_YAW = flattestYaw(FLAT[0], FLAT[1]);
const DEEP = findSpot((x, z) => elevation(x, z, SEED, 0) < VALLEY.wLevel - 3, 2);
const WALL = findSpot((x, z) => {
  const [gx, gz] = gradAt(x, z);
  const m = Math.hypot(gx, gz);
  if (m < 1.35) return false;
  for (let k = 1; k <= 4; k++) if (slope(x + (gx / m) * k, z + (gz / m) * k, SEED) < 1.2) return false;
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
    const ideal = (HERO.jumpVelocity * HERO.jumpVelocity) / (2 * PLAYER.gravity);
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

  it('a long drop (from fly mode) settles at terminal velocity and can still open the glider', () => {
    const s = createHeroState(FLAT[0], FLAT[1], 0, SEED);
    placeHero(s, s.x, s.ground + 900, s.z, 0, 0, 0, 0, SEED);
    s.jumpHeld = false;
    let fastest = 0;
    run(s, 6, input(), () => { fastest = Math.max(fastest, -s.vy); });
    expect(fastest).toBe(HERO.terminalVelocity);
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

describe('Hero locomotion — climbing', () => {
  const atWall = () => createHeroState(WALL[0] - Math.sin(WALL_UP) * 0.3, WALL[1] - Math.cos(WALL_UP) * 0.3, WALL_UP, SEED);

  it('pushing into ground steeper than walkSlope grabs it and climbs uphill at ≈ climbSpeed', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    expect(s.move).toBe('climb');
    const e0 = s.ground;
    const x0 = s.x, z0 = s.z, y0 = s.y;
    run(s, 1.0, input({ forward: 1, camYaw: WALL_UP }));
    expect(s.move).toBe('climb');
    expect(s.ground).toBeGreaterThan(e0 + 0.5);
    const surface = Math.hypot(s.x - x0, s.y - y0, s.z - z0);
    expect(surface).toBeGreaterThan(HERO.climbSpeed * 0.8);
    expect(surface).toBeLessThan(HERO.climbSpeed * 1.15);
    expect(s.stamina).toBeCloseTo(1 - HERO.staminaClimb * 1.1, 1);
    expect(heroAction(s)).toBe('climb');
  });

  it('hanging still drains the hang rate; the sneak key lets go; a dry wheel drops the hero', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    const st = s.stamina;
    run(s, 1.0, input());
    expect(s.move).toBe('climb');
    expect(heroAction(s)).toBe('hang');
    expect(st - s.stamina).toBeCloseTo(HERO.staminaHang, 2);
    updateHero(s, input({ sneak: true }), DT, SEED);
    expect(s.move).toBe('air');
    const d = atWall();
    run(d, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    d.stamina = 0.001;
    run(d, 0.2, input({ forward: 1, camYaw: WALL_UP }));
    expect(d.exhausted).toBe(true);
    expect(d.move).not.toBe('climb');
    // exhausted heroes cannot grab the wall again
    run(d, 1.5, input({ forward: 1, camYaw: WALL_UP }));
    expect(d.move).not.toBe('climb');
  });

  it('a climb jump surges HERO.climbJump metres up the wall for its stamina cost', () => {
    const s = atWall();
    run(s, 0.1, input({ forward: 1, camYaw: WALL_UP }));
    s.jumpHeld = false;
    const st = s.stamina, x0 = s.x, y0 = s.y, z0 = s.z;
    run(s, 0.5, (t) => input({ jump: t < 0.05 }));
    expect(Math.hypot(s.x - x0, s.y - y0, s.z - z0)).toBeGreaterThan(HERO.climbJump * 0.9);
    expect(st - s.stamina).toBeGreaterThan(HERO.staminaClimbJump);
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
});

describe('Hero locomotion — swimming', () => {
  it('floats at the swim depth, swims at swimSpeed, dashes on stamina', () => {
    const s = createHeroState(DEEP[0], DEEP[1], 0, SEED);
    run(s, 3.0, input());
    expect(s.move).toBe('swim');
    expect(s.y).toBeCloseTo(VALLEY.wLevel - HERO.swimFloat, 1);
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
        if (s.move !== 'swim') worst = Math.min(worst, s.y - (elevation(s.x, s.z, SEED, 0) + BIAS));
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

  it('idle feet rest on the terrain even on a slope (≤ 3 cm off the true surface)', () => {
    const slopeSpot = findSpot((x, z) => { const m = slope(x, z, SEED); return m > 0.5 && m < 0.8; }, 3);
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
        expect(c.y).toBeGreaterThanOrEqual(VALLEY.wLevel + HERO_CAMERA.clearance - 1e-9);
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
    run(s, 2.2, input(), () => {
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

  it('uniform layout: 18 joints, two mat3 frames, bound and misc vec4s', () => {
    expect(TERRAIN_FS).toContain(`uniform vec3  u_hero_j[${HERO_JOINTS}];`);
    expect<number>(HERO_UNIFORM.torso).toBe(3 * HERO_JOINTS);
    expect(HERO_UNIFORM.head - HERO_UNIFORM.torso).toBe(9);
    expect(HERO_UNIFORM.bound - HERO_UNIFORM.head).toBe(9);
    expect(HERO_UNIFORM.misc - HERO_UNIFORM.bound).toBe(4);
    expect(HERO_UNIFORM.length - HERO_UNIFORM.misc).toBe(4);
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
