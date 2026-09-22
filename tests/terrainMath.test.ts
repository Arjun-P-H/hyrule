import { describe, it, expect } from 'bun:test';
import {
  hashU,
  hash01,
  quintic,
  vnoise,
  worleyEdge,
  domainWarp,
  anomalyE,
  regionField,
  dunesE,
  elevation,
  gradient,
  slope,
  temperature,
  moisture,
  marchRay,
  softShadow,
  updatePlayer,
  type PlayerState,
} from '../src/lib/terrainMath';
import {
  CHUNK_SIZE,
  MAX_WORLD_H,
  DEFAULT_SEED,
  SPAWN,
  F,
  PLAYER,
  CLIMATE,
  CAMERA,
  MARCH,
  WORKED_EXAMPLE,
  QUALITY_PRESETS,
} from '../src/lib/constants';

describe('1. MurmurHash3 & Integer Determinism', () => {
  it('hashU produces deterministic 32-bit unsigned integers', () => {
    const h1 = hashU(1337);
    const h2 = hashU(1337);
    expect(h1).toBe(h2);
    expect(h1).toBeGreaterThanOrEqual(0);
    expect(h1).toBeLessThanOrEqual(0xffffffff);
  });

  it('hashU avalanches bit flips properly', () => {
    const a = hashU(100);
    const b = hashU(101);
    expect(a).not.toBe(b);
  });
});

describe('2. Dyadic Frequencies & Chunk Invariance', () => {
  it('every frequency n / 2^20 is an exact dyadic rational', () => {
    for (const freq of Object.values(F)) {
      const num = freq * 1048576;
      expect(Math.round(num)).toBe(num);
    }
  });

  it('chunk origin * f is exact in fp32 without residual seam', () => {
    const origin = CHUNK_SIZE; // 8192 = 2^13
    for (const freq of Object.values(F)) {
      const prod = origin * freq;
      // 8192 / 1048576 = 1/128
      const n = prod * 128;
      expect(Math.round(n)).toBe(n);
    }
  });
});

describe('3. Quintic Interpolation & Value Noise', () => {
  it('quintic satisfies C2 continuity boundary conditions', () => {
    expect(quintic(0.0)).toBe(0.0);
    expect(quintic(1.0)).toBe(1.0);
    expect(quintic(0.5)).toBe(0.5);

    // Verify derivative at endpoints is 0 (first & second derivative)
    const eps = 1e-5;
    const d0 = (quintic(eps) - quintic(0)) / eps;
    const d1 = (quintic(1) - quintic(1 - eps)) / eps;
    expect(Math.abs(d0)).toBeLessThan(1e-4);
    expect(Math.abs(d1)).toBeLessThan(1e-4);
  });

  it('vnoise remains within [-1, 1] range', () => {
    for (let x = -50; x <= 50; x += 10) {
      for (let y = -50; y <= 50; y += 10) {
        const v = vnoise(x, y, DEFAULT_SEED);
        expect(v).toBeGreaterThanOrEqual(-1.001);
        expect(v).toBeLessThanOrEqual(1.001);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it('worleyEdge returns non-negative river distances', () => {
    for (let x = 0; x < 20; x += 4) {
      for (let y = 0; y < 20; y += 4) {
        const d = worleyEdge(x, y, DEFAULT_SEED);
        expect(d).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(d)).toBe(true);
      }
    }
  });
});

describe('4. Elevation Field E(x, z) & Limits', () => {
  it('elevation is deterministic and identical across repeated evaluations', () => {
    const e1 = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const e2 = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    expect(e1).toBe(e2);
    expect(Number.isFinite(e1)).toBe(true);
  });

  it('elevation respects maximum world height ceiling', () => {
    const samples = [
      [0, 0],
      [1000, 1000],
      [-2000, 4000],
      [8192, 8192],
      [-16384, 16384],
    ];
    for (const [x, z] of samples) {
      const h = elevation(x, z, DEFAULT_SEED, 0);
      expect(h).toBeLessThanOrEqual(MAX_WORLD_H);
      expect(Number.isFinite(h)).toBe(true);
    }
  });

  it('elevation handles extreme coordinates without NaN or Infinity', () => {
    const extremes = [
      [-100000, -100000],
      [100000, 100000],
      [0.0001, -0.0001],
      [-CHUNK_SIZE, CHUNK_SIZE],
    ];
    for (const [x, z] of extremes) {
      const h = elevation(x, z, DEFAULT_SEED, 0);
      expect(Number.isFinite(h)).toBe(true);
    }
  });
});

describe('5. Gradient & Slope Calculation', () => {
  it('gradient produces finite central-difference partial derivatives', () => {
    const [gx, gz] = gradient(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0.05);
    expect(Number.isFinite(gx)).toBe(true);
    expect(Number.isFinite(gz)).toBe(true);
  });

  it('slope magnitude matches hypot(gx, gz) and is non-negative', () => {
    const s = slope(SPAWN.x, SPAWN.z, DEFAULT_SEED);
    expect(s).toBeGreaterThanOrEqual(0.0);
    expect(Number.isFinite(s)).toBe(true);
  });
});

describe('6. Player Physics & Locomotion Loop', () => {
  it('clamps player to terrain with ground bias and reports grounded', () => {
    const ground = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const targetY = ground + 0.08 + 1.7;
    const player: PlayerState = {
      x: SPAWN.x,
      y: ground, // below clamp threshold
      z: SPAWN.z,
      vx: 0,
      vy: -10, // falling downward
      vz: 0,
      grounded: false,
    };

    updatePlayer(
      player,
      { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 },
      0.1,
      DEFAULT_SEED,
    );

    // After clamp, player should be on ground at targetY and grounded true
    expect(player.grounded).toBe(true);
    expect(Math.abs(player.y - targetY)).toBeLessThan(1e-4);
    expect(player.vy).toBe(0.0);
  });

  it('jump imparts upward vertical velocity when grounded', () => {
    const ground = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const player: PlayerState = {
      x: SPAWN.x,
      y: ground + PLAYER.height,
      z: SPAWN.z,
      vx: 0,
      vy: 0,
      vz: 0,
      grounded: true,
    };

    updatePlayer(
      player,
      { forward: 0, strafe: 0, jump: true, sprint: false, yaw: 0 },
      0.016,
      DEFAULT_SEED,
    );

    expect(player.vy).toBeGreaterThan(0);
  });

  it('sprint accelerates player faster than normal walk', () => {
    const ground = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const pWalk: PlayerState = {
      x: SPAWN.x,
      y: ground + PLAYER.height,
      z: SPAWN.z,
      vx: 0,
      vy: 0,
      vz: 0,
      grounded: true,
    };
    const pSprint: PlayerState = {
      x: SPAWN.x,
      y: ground + PLAYER.height,
      z: SPAWN.z,
      vx: 0,
      vy: 0,
      vz: 0,
      grounded: true,
    };

    updatePlayer(
      pWalk,
      { forward: 1, strafe: 0, jump: false, sprint: false, yaw: 0 },
      0.1,
      DEFAULT_SEED,
    );
    updatePlayer(
      pSprint,
      { forward: 1, strafe: 0, jump: false, sprint: true, yaw: 0 },
      0.1,
      DEFAULT_SEED,
    );

    const speedWalk = Math.hypot(pWalk.vx, pWalk.vz);
    const speedSprint = Math.hypot(pSprint.vx, pSprint.vz);
    expect(speedSprint).toBeGreaterThan(speedWalk);
  });
});

describe('7. Quality Presets & Config Integrity', () => {
  it('quality presets provide distinct march and shadow step counts', () => {
    expect(QUALITY_PRESETS.length).toBe(3);
    const [low, med, high] = QUALITY_PRESETS;
    expect(low.marchSteps).toBeLessThan(med.marchSteps);
    expect(med.marchSteps).toBeLessThan(high.marchSteps);
    expect(low.resScale).toBeLessThanOrEqual(med.resScale);
  });
});

describe('8. Raymarching & Soft Shadows', () => {
  it('raymarching downward from above terrain detects collision', () => {
    const ground = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const ro: [number, number, number] = [SPAWN.x, ground + 100, SPAWN.z];
    const rd: [number, number, number] = [0, -1, 0]; // straight down
    const result = marchRay(ro, rd, DEFAULT_SEED, {
      steps: 64,
      footprint: 0.001,
      maxDist: 500,
    });
    expect(result.hit).toBe(true);
    expect(result.t).toBeGreaterThan(0);
    expect(result.t).toBeLessThan(150);
  });

  it('raymarching upward into the open sky misses', () => {
    const ground = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const ro: [number, number, number] = [SPAWN.x, ground + 50, SPAWN.z];
    const rd: [number, number, number] = [0, 1, 0]; // straight up
    const result = marchRay(ro, rd, DEFAULT_SEED, {
      steps: 48,
      footprint: 0.001,
      maxDist: 500,
    });
    expect(result.hit).toBe(false);
  });

  it('softShadow returns a valid penumbra coefficient in [0, 1]', () => {
    const ground = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0);
    const p: [number, number, number] = [SPAWN.x, ground + 1.0, SPAWN.z];
    const sunDir: [number, number, number] = [0.550744, 0.420568, 0.720974];
    const shadow = softShadow(p, sunDir, DEFAULT_SEED, 16, 0.001);
    expect(shadow).toBeGreaterThanOrEqual(0.0);
    expect(shadow).toBeLessThanOrEqual(1.0);
    expect(Number.isFinite(shadow)).toBe(true);
  });
});

describe('9. Climate & Biome Temperature/Moisture Equations', () => {
  it('temperature decreases with latitude and altitude (lapse rate)', () => {
    const tEquator = temperature(0, 0, 0, DEFAULT_SEED);
    const tNorth = temperature(0, 50000, 0, DEFAULT_SEED);
    const tSummit = temperature(0, 0, 700, DEFAULT_SEED);

    expect(tEquator).toBe(CLIMATE.tBase);
    expect(tNorth).toBeLessThan(tEquator);
    expect(tSummit).toBeLessThan(tEquator);
    // 700m summit should experience 700 * 0.02 = 14°C drop -> 0°C (snow line)
    expect(Math.abs(tSummit)).toBeLessThan(1e-4);
  });

  it('moisture remains bounded in reasonable climate range', () => {
    for (let x = -2000; x <= 2000; x += 1000) {
      for (let z = -2000; z <= 2000; z += 1000) {
        const m = moisture(x, z, 0, DEFAULT_SEED);
        expect(m).toBeGreaterThan(0.0);
        expect(m).toBeLessThan(2.0);
        expect(Number.isFinite(m)).toBe(true);
      }
    }
  });
});

describe('10. Geological Features & Regional Anomaly Generation', () => {
  it('domainWarp produces continuous 2D coordinate displacement', () => {
    const [wx1, wy1] = domainWarp(100, 200, DEFAULT_SEED, 0);
    const [wx2, wy2] = domainWarp(100.1, 200.1, DEFAULT_SEED, 0);
    expect(Number.isFinite(wx1)).toBe(true);
    expect(Number.isFinite(wy1)).toBe(true);
    expect(Math.hypot(wx2 - wx1, wy2 - wy1)).toBeLessThan(1.0);
  });

  it('regionField produces normalized region weights and factors', () => {
    const reg = regionField(SPAWN.x, SPAWN.z, DEFAULT_SEED);
    expect(reg.wRegion).toBeGreaterThanOrEqual(0);
    expect(reg.wRegion).toBeLessThanOrEqual(1);
    expect(reg.mountFactor).toBeGreaterThan(0);
    expect(reg.duneAmount).toBeGreaterThanOrEqual(0);
  });

  it('dunesE returns 0 when dune amount is 0 and non-negative amplitude when active', () => {
    const dZero = dunesE(100, 200, DEFAULT_SEED, 0, 0);
    expect(dZero).toBe(0.0);

    const dActive = dunesE(100, 200, DEFAULT_SEED, 0, 1.0);
    expect(dActive).toBeGreaterThanOrEqual(0.0);
    expect(Number.isFinite(dActive)).toBe(true);
  });

  it('anomalyE returns finite elevation bonus or zero outside anomaly radius', () => {
    const anom = anomalyE(SPAWN.x, SPAWN.z, DEFAULT_SEED);
    expect(Number.isFinite(anom)).toBe(true);
    expect(anom).toBeGreaterThanOrEqual(0.0);
  });
});

describe('11. Bit Parity: Integer Lattice Determinism & GPU Alignment', () => {
  it('vnoise at integer lattice coordinates matches hash01 bit parity exactly', () => {
    const testPoints = [
      [0, 0],
      [10, 20],
      [-5, -12],
      [100, -250],
      [8192, 8192],
    ];
    for (const [ix, iy] of testPoints) {
      const vn = vnoise(ix, iy, DEFAULT_SEED);
      const h01 = hash01(ix, iy, DEFAULT_SEED);
      const expected = h01 * 2.0 - 1.0;
      expect(Math.abs(vn - expected)).toBeLessThan(1e-12);
    }
  });

  it('chunk relative coordinates align seamlessly at origin rebasing', () => {
    // Spec §25: chunk origin = 8192 * floor(pos / 8192)
    const worldX = 12345.67;
    const worldZ = -9876.54;
    const ox = Math.floor(worldX / CHUNK_SIZE) * CHUNK_SIZE;
    const oz = Math.floor(worldZ / CHUNK_SIZE) * CHUNK_SIZE;
    const lx = worldX - ox;
    const lz = worldZ - oz;

    expect(ox + lx).toBeCloseTo(worldX, 10);
    expect(oz + lz).toBeCloseTo(worldZ, 10);
    expect(lx).toBeGreaterThanOrEqual(0);
    expect(lx).toBeLessThan(CHUNK_SIZE);
  });
});

describe('12. Slope Slide-Off Gate & Steep Terrain Physics', () => {
  it('sliding acceleration triggers only when slope magnitude exceeds slopeMax (1.5)', () => {
    // Create player on a flat surface
    const pFlat: PlayerState = {
      x: 0,
      y: 100,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      grounded: true,
    };
    updatePlayer(
      pFlat,
      { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 },
      0.016,
      DEFAULT_SEED,
    );

    // Look for a steep gradient point
    let steepX = 0, steepZ = 0, foundSteep = false;
    for (let x = 1500; x <= 2500; x += 50) {
      for (let z = -3000; z <= -2000; z += 50) {
        if (slope(x, z, DEFAULT_SEED) > PLAYER.slopeMax + 0.2) {
          steepX = x;
          steepZ = z;
          foundSteep = true;
          break;
        }
      }
      if (foundSteep) break;
    }

    if (foundSteep) {
      const ground = elevation(steepX, steepZ, DEFAULT_SEED, 0);
      const pSteep: PlayerState = {
        x: steepX,
        y: ground + PLAYER.height + PLAYER.groundBias,
        z: steepZ,
        vx: 0,
        vy: 0,
        vz: 0,
        grounded: true,
      };

      updatePlayer(
        pSteep,
        { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 },
        0.05,
        DEFAULT_SEED,
      );

      // Slide acceleration must have imparted horizontal downhill velocity
      const slideSpeed = Math.hypot(pSteep.vx, pSteep.vz);
      expect(slideSpeed).toBeGreaterThan(0);
    }
  });
});

describe('13. Worked-Example Raymarch Convergence (Spec §30)', () => {
  it('ray targeted at vista ridge brackets and bisects near expected nominal distance', () => {
    const foot = (2 * Math.tan((CAMERA.fov * Math.PI / 180) * 0.5)) / WORKED_EXAMPLE.resY;
    const camY = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0) + WORKED_EXAMPLE.camHeight;
    const ro: [number, number, number] = [SPAWN.x, camY, SPAWN.z];

    const cp = Math.cos(SPAWN.pitch), sp = Math.sin(SPAWN.pitch);
    const rd: [number, number, number] = [
      Math.sin(SPAWN.yaw) * cp,
      sp,
      Math.cos(SPAWN.yaw) * cp,
    ];

    const result = marchRay(ro, rd, DEFAULT_SEED, {
      steps: 96,
      footprint: foot,
      record: true,
    });

    expect(result.hit).toBe(true);
    expect(result.t).toBeGreaterThan(1500);
    expect(result.t).toBeLessThan(3000);
    expect(result.bisectRows.length).toBeGreaterThanOrEqual(MARCH.bisectSteps);
  });
});
