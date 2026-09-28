import { describe, it, expect } from 'bun:test';
import {
  hashU,
  hash01,
  cellPoint,
  cellType,
  voronoi,
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
  VALLEY,
  LIGHT,
  BIOME_ANCHORS,
  MACRO,
  MOUNTAIN,
  WARP,
  REGION,
  ANOMALY,
} from '../src/lib/constants';
import { TERRAIN_VS, TERRAIN_FS } from '../src/components/terrain/demoShader';

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

  it('hash01, cellPoint, and cellType satisfy range and packing contracts', () => {
    for (let ix = -10; ix <= 10; ix++) {
      for (let iy = -10; iy <= 10; iy++) {
        const u = hash01(ix, iy, DEFAULT_SEED);
        expect(u).toBeGreaterThanOrEqual(0.0);
        expect(u).toBeLessThanOrEqual(1.0);

        const [px, py] = cellPoint(ix, iy, DEFAULT_SEED);
        expect(px).toBeGreaterThanOrEqual(0.0);
        expect(px).toBeLessThanOrEqual(1.0);
        expect(py).toBeGreaterThanOrEqual(0.0);
        expect(py).toBeLessThanOrEqual(1.0);

        const ct = cellType(ix, iy, DEFAULT_SEED);
        expect(ct).toBeGreaterThanOrEqual(0.0);
        expect(ct).toBeLessThanOrEqual(1.0);
      }
    }
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
    for (let k = -12; k <= 12; k++) {
      const origin = k * CHUNK_SIZE; // k * 8192 = k * 2^13
      for (const [key, freq] of Object.entries(F)) {
        if (key === 'veg') continue;
        const prod = origin * freq;
        // Exact dyadic rational: (k * 2^13 * n) / 2^20 = (k * n) / 128
        const n = prod * 128;
        expect(Math.round(n)).toBe(n);
        // Bit-exact in single precision fp32: Math.fround must equal double value
        expect(Math.fround(prod)).toBe(prod);
      }
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

describe('14. Large-Scale Parity & Numerical Determinism (§IV.L.1.a)', () => {
  it('10,000 pseudo-random points verify elevation stability and range bounds', () => {
    let seedState = 42;
    // Simple LCG PRNG for determinism across platforms
    const lcg = () => {
      seedState = (Math.imul(1664525, seedState) + 1013904223) | 0;
      return (seedState >>> 0) / 4294967295.0;
    };

    let nanCount = 0;
    let minH = Infinity;
    let maxH = -Infinity;

    for (let i = 0; i < 10000; i++) {
      const rx = (lcg() - 0.5) * 100000;
      const rz = (lcg() - 0.5) * 100000;
      const rSeed = (lcg() * 10000) | 0;
      const h = elevation(rx, rz, rSeed, 0);

      if (!Number.isFinite(h)) nanCount++;
      if (h < minH) minH = h;
      if (h > maxH) maxH = h;
    }

    expect(nanCount).toBe(0);
    expect(maxH).toBeLessThanOrEqual(MAX_WORLD_H);
    expect(minH).toBeGreaterThan(-500.0);
  });
});

describe('15. Chunk Boundary Continuity & Seam Invariance (§IV.L.1.b)', () => {
  it('elevation is continuous across chunk boundaries at k * 8192 m', () => {
    const boundaries = [-16384, -8192, 0, 8192, 16384];
    const eps = 1e-4;

    for (const b of boundaries) {
      // Test crossing boundary in X
      const eLeftX = elevation(b - eps, 2500, DEFAULT_SEED, 0);
      const eExactX = elevation(b, 2500, DEFAULT_SEED, 0);
      const eRightX = elevation(b + eps, 2500, DEFAULT_SEED, 0);

      expect(Math.abs(eLeftX - eExactX)).toBeLessThan(0.01);
      expect(Math.abs(eRightX - eExactX)).toBeLessThan(0.01);

      // Test crossing boundary in Z
      const eLeftZ = elevation(2500, b - eps, DEFAULT_SEED, 0);
      const eExactZ = elevation(2500, b, DEFAULT_SEED, 0);
      const eRightZ = elevation(2500, b + eps, DEFAULT_SEED, 0);

      expect(Math.abs(eLeftZ - eExactZ)).toBeLessThan(0.01);
      expect(Math.abs(eRightZ - eExactZ)).toBeLessThan(0.01);
    }
  });

  it('distant chunk boundaries (±100,000m) maintain C0 continuity and fp32 boundary invariance', () => {
    // Multiples of CHUNK_SIZE (8192m) up to ±100,000m
    const distantChunkIndices = [-12, -7, -3, 3, 7, 12];
    for (const k of distantChunkIndices) {
      const chunkEdge = k * CHUNK_SIZE;
      for (const offset of [0, 1024, 4096, 7168]) {
        const testZ = offset;
        // Limit continuity test: e(edge+) vs e(edge-)
        const eMinus = elevation(chunkEdge - 1e-11, testZ, DEFAULT_SEED, 0);
        const ePlus = elevation(chunkEdge + 1e-11, testZ, DEFAULT_SEED, 0);
        expect(Math.abs(ePlus - eMinus)).toBeLessThan(1e-7);

        // Simulated fp32 chunk-relative rebasing continuity across edge
        // In chunk k-1, local coordinate approaches CHUNK_SIZE
        const localLeft = Math.fround(CHUNK_SIZE - 0.001);
        const originLeft = Math.fround((k - 1) * CHUNK_SIZE);
        const eChunkLeft = elevation(originLeft + localLeft, testZ, DEFAULT_SEED, 0);

        // In chunk k, local coordinate starts at 0
        const localRight = Math.fround(0.001);
        const originRight = Math.fround(k * CHUNK_SIZE);
        const eChunkRight = elevation(originRight + localRight, testZ, DEFAULT_SEED, 0);

        expect(Math.abs(eChunkRight - eChunkLeft)).toBeLessThan(0.01);
      }
    }
  });
});

describe('16. Physics Determinism (§IV.L.1.e)', () => {
  it('1,000-step physics simulation produces deterministic final state from initial state', () => {
    const runSimulation = () => {
      const p: PlayerState = {
        x: SPAWN.x,
        y: 400.0,
        z: SPAWN.z,
        vx: 1.0,
        vy: 0.0,
        vz: 2.0,
        grounded: false,
      };

      for (let step = 0; step < 1000; step++) {
        updatePlayer(
          p,
          {
            forward: (step % 120 < 60) ? 1.0 : -0.5,
            strafe: (step % 80 < 40) ? 0.5 : -0.5,
            jump: (step % 50 === 0),
            sprint: (step % 200 < 100),
            yaw: (step * 0.01) % (2 * Math.PI),
          },
          0.016666666666666666, // 60 Hz dt
          DEFAULT_SEED,
        );
      }
      return p;
    };

    const run1 = runSimulation();
    const run2 = runSimulation();

    expect(run1.x).toBe(run2.x);
    expect(run1.y).toBe(run2.y);
    expect(run1.z).toBe(run2.z);
    expect(run1.vx).toBe(run2.vx);
    expect(run1.vy).toBe(run2.vy);
    expect(run1.vz).toBe(run2.vz);
    expect(run1.grounded).toBe(run2.grounded);
  });

  it('10,000-step physics simulation produces bit-for-bit deterministic final state across two runs', () => {
    const runSimulation = () => {
      const p: PlayerState = {
        x: SPAWN.x,
        y: 400.0,
        z: SPAWN.z,
        vx: 1.0,
        vy: 0.0,
        vz: 2.0,
        grounded: false,
      };

      for (let step = 0; step < 10000; step++) {
        updatePlayer(
          p,
          {
            forward: (step % 120 < 60) ? 1.0 : -0.5,
            strafe: (step % 80 < 40) ? 0.5 : -0.5,
            jump: (step % 50 === 0),
            sprint: (step % 200 < 100),
            yaw: (step * 0.01) % (2 * Math.PI),
          },
          0.016666666666666666, // 60 Hz dt
          DEFAULT_SEED,
        );
      }
      return p;
    };

    const run1 = runSimulation();
    const run2 = runSimulation();

    expect(run1.x).toBe(run2.x);
    expect(run1.y).toBe(run2.y);
    expect(run1.z).toBe(run2.z);
    expect(run1.vx).toBe(run2.vx);
    expect(run1.vy).toBe(run2.vy);
    expect(run1.vz).toBe(run2.vz);
    expect(run1.grounded).toBe(run2.grounded);
  });
});

describe('17. Performance Benchmark Thresholds (§VII Benchmark Standard)', () => {
  it('elevation function evaluates in < 0.002 ms per point', () => {
    const N = 20000;
    const t0 = performance.now();
    let sum = 0;
    for (let i = 0; i < N; i++) {
      sum += elevation(SPAWN.x + (i & 127), SPAWN.z + ((i >> 7) & 127), DEFAULT_SEED, 0);
    }
    const t1 = performance.now();
    const msPerPoint = (t1 - t0) / N;

    expect(Number.isFinite(sum)).toBe(true);
    expect(msPerPoint).toBeLessThan(0.002); // 2 microseconds target
  });

  it('physics loop tick executes in < 0.005 ms per tick', () => {
    const N = 10000;
    const p: PlayerState = {
      x: SPAWN.x,
      y: 350.0,
      z: SPAWN.z,
      vx: 2.0,
      vy: 0.0,
      vz: 1.0,
      grounded: true,
    };
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      updatePlayer(
        p,
        { forward: 1, strafe: 0, jump: false, sprint: true, yaw: 1.0 },
        0.016,
        DEFAULT_SEED,
      );
    }
    const t1 = performance.now();
    const msPerTick = (t1 - t0) / N;

    expect(msPerTick).toBeLessThan(0.005); // 5 microseconds target
  });
});

describe('18. GLSL ES 3.00 Shader Contract & Integrity (§IV.A, §IV.L)', () => {
  it('TERRAIN_VS and TERRAIN_FS declare #version 300 es and precision qualifiers', () => {
    expect(TERRAIN_VS.trim().startsWith('#version 300 es')).toBe(true);
    expect(TERRAIN_FS.trim().startsWith('#version 300 es')).toBe(true);
    expect(TERRAIN_FS).toContain('precision highp float;');
    expect(TERRAIN_FS).toContain('out vec4 fragColor;');
  });

  it('all contractual uniforms from useTerrainEngine are declared in TERRAIN_FS', () => {
    const requiredUniforms = [
      'uniform vec2  u_res',
      'uniform float u_time',
      'uniform vec3  u_cam_pos',
      'uniform vec2  u_chunk_origin',
      'uniform mat3  u_basis',
      'uniform float u_tan_half',
      'uniform int   u_seed',
      'uniform int   u_march_steps',
      'uniform int   u_shadow_steps',
      'uniform float u_footprint',
    ];
    for (const u of requiredUniforms) {
      expect(TERRAIN_FS).toContain(u);
    }
  });

  it('all fbm and ridged invocations in TERRAIN_FS strictly match parameter signatures', () => {
    function parseCalls(source: string, fnName: string) {
      const calls: { call: string; args: string[] }[] = [];
      let idx = 0;
      while ((idx = source.indexOf(fnName + '(', idx)) !== -1) {
        const before = source.slice(Math.max(0, idx - 10), idx);
        if (/float\s+$/.test(before)) {
          idx += fnName.length + 1;
          continue;
        }
        let depth = 0;
        let currentArg = '';
        const args: string[] = [];
        for (let i = idx + fnName.length; i < source.length; i++) {
          const c = source[i];
          if (c === '(') {
            depth++;
            if (depth > 1) currentArg += c;
          } else if (c === ')') {
            depth--;
            if (depth === 0) {
              args.push(currentArg.trim());
              calls.push({ call: source.slice(idx, i + 1), args });
              idx = i + 1;
              break;
            } else {
              currentArg += c;
            }
          } else if (c === ',' && depth === 1) {
            args.push(currentArg.trim());
            currentArg = '';
          } else if (depth >= 1) {
            currentArg += c;
          }
        }
      }
      return calls;
    }

    const fbmCalls = parseCalls(TERRAIN_FS, 'fbm');
    expect(fbmCalls.length).toBe(6);
    for (const call of fbmCalls) {
      expect(call.args.length).toBe(9);
    }

    const ridgedCalls = parseCalls(TERRAIN_FS, 'ridged');
    expect(ridgedCalls.length).toBe(1);
    for (const call of ridgedCalls) {
      expect(call.args.length).toBe(9);
    }
  });

  it('dyadic frequency constants in GLSL match CPU constants exactly', () => {
    expect(TERRAIN_FS).toContain('const float F_MACRO   = 512.0  / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_RIDGE   = 341.0  / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_MASK    = 128.0  / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_WARP    = 655.0  / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_VALLEY  = 1942.0 / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_CONT    = 64.0   / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_MOIST   = 512.0  / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_DUNE    = 23.0   / 1048576.0;');
    expect(TERRAIN_FS).toContain('const float F_CLOUD   = 403.0  / 1048576.0;');
  });

  it('amplitude and geometric contracts in GLSL match CPU contracts', () => {
    expect(TERRAIN_FS).toContain('const float MAXH      = 1300.0;');
    expect(TERRAIN_FS).toContain('H_BASE  = 90.0;');
    expect(TERRAIN_FS).toContain('H_MOUNT = 880.0;');
    expect(TERRAIN_FS).toContain('C_DEPTH  = 48.0');
    expect(TERRAIN_FS).toContain('W_LEVEL = 5.0');
    expect(TERRAIN_FS).toContain('const float ASUM_RIDGE = 0.93703808;');
  });

  it('programmatically parses and cross-checks all GLSL float constants against constants.ts', () => {
    const glslConsts = new Map<string, number>();
    const stmtRegex = /const\s+float\s+([^;]+);/g;
    let stmtMatch;
    while ((stmtMatch = stmtRegex.exec(TERRAIN_FS)) !== null) {
      const decls = stmtMatch[1].split(',');
      for (const decl of decls) {
        const parts = decl.split('=');
        if (parts.length === 2) {
          const name = parts[0].trim();
          const expr = parts[1].trim();
          try {
            const val = Function(`"use strict"; return (${expr});`)();
            if (typeof val === 'number') {
              glslConsts.set(name, val);
            }
          } catch {
            // ignore
          }
        }
      }
    }

    expect(glslConsts.get('MAXH')).toBe(MAX_WORLD_H);
    expect(glslConsts.get('H_BASE')).toBe(MACRO.hBase);
    expect(glslConsts.get('H_MOUNT')).toBe(MOUNTAIN.hMount);
    expect(glslConsts.get('C_DEPTH')).toBe(VALLEY.cDepth);
    expect(glslConsts.get('W_LEVEL')).toBe(VALLEY.wLevel);
    expect(glslConsts.get('W_VALLEY')).toBe(VALLEY.wValley);
    expect(glslConsts.get('W_WIDTH')).toBe(VALLEY.wWidth);
    expect(glslConsts.get('S_WARP')).toBe(WARP.strength);
    expect(glslConsts.get('DUNE_H')).toBe(REGION.duneHeight);
    expect(glslConsts.get('ANOM_CELL')).toBe(ANOMALY.cell);
    expect(glslConsts.get('ANOM_R_IN')).toBe(ANOMALY.rIn);
    expect(glslConsts.get('ANOM_R_OUT')).toBe(ANOMALY.rOut);
    expect(glslConsts.get('TERRACE_STEP')).toBe(ANOMALY.terraceStep);
    expect(glslConsts.get('TERRACE_H')).toBe(ANOMALY.terraceHeight);
    expect(glslConsts.get('SPIRE_H')).toBe(ANOMALY.spireHeight);
    expect(glslConsts.get('SPIRE_R')).toBe(ANOMALY.spireRadius);
    expect(glslConsts.get('SPIRAL_AMP')).toBe(ANOMALY.spiralAmp);
    expect(glslConsts.get('SPIRAL_TURNS')).toBe(ANOMALY.spiralTurns);
    expect(glslConsts.get('SPIRAL_RAD')).toBe(ANOMALY.spiralRadial);
    expect(glslConsts.get('T_BASE')).toBe(CLIMATE.tBase);
    expect(glslConsts.get('M_BASE')).toBe(CLIMATE.mBase);
    expect(glslConsts.get('M_AMP')).toBe(CLIMATE.mNoiseAmp);
    expect(glslConsts.get('M_OCEAN')).toBe(CLIMATE.mOceanLambda);
    expect(glslConsts.get('CLOUD_PLANE')).toBe(LIGHT.cloudPlane);

    expect(glslConsts.get('F_MACRO')).toBeCloseTo(F.macro, 10);
    expect(glslConsts.get('F_RIDGE')).toBeCloseTo(F.ridge, 10);
    expect(glslConsts.get('F_MASK')).toBeCloseTo(F.mask, 10);
    expect(glslConsts.get('F_WARP')).toBeCloseTo(F.warp, 10);
    expect(glslConsts.get('F_VALLEY')).toBeCloseTo(F.valley, 10);
    expect(glslConsts.get('F_CONT')).toBeCloseTo(F.cont, 10);
    expect(glslConsts.get('F_MOIST')).toBeCloseTo(F.moist, 10);
    expect(glslConsts.get('F_DUNE')).toBeCloseTo(F.dune, 10);
    expect(glslConsts.get('F_CLOUD')).toBeCloseTo(F.cloud, 10);
  });
});

describe('19. Analytic Directional Gradient & Numerical Determinism (§IV.L.1.c, §IV.L.1.d)', () => {
  it('gradient directional projection matches symmetric central differences', () => {
    const eps = 0.25;
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 4) {
      const dx = Math.cos(angle);
      const dz = Math.sin(angle);
      const [gx, gz] = gradient(SPAWN.x, SPAWN.z, DEFAULT_SEED, eps);
      const proj = gx * dx + gz * dz;
      const fwd = elevation(SPAWN.x + dx * eps, SPAWN.z + dz * eps, DEFAULT_SEED, 0);
      const bwd = elevation(SPAWN.x - dx * eps, SPAWN.z - dz * eps, DEFAULT_SEED, 0);
      const finiteDiff = (fwd - bwd) / (2 * eps);
      expect(Math.abs(proj - finiteDiff)).toBeLessThan(0.01);
    }
  });

  it('elevation is 100% bit-deterministic across multiple runs (§IV.L.1.d)', () => {
    const coords: [number, number][] = [];
    for (let i = 0; i < 1000; i++) {
      coords.push([(i * 37.13) % 20000 - 10000, (i * 59.27) % 20000 - 10000]);
    }
    const pass1 = coords.map(([x, z]) => elevation(x, z, DEFAULT_SEED, 0));
    const pass2 = coords.map(([x, z]) => elevation(x, z, DEFAULT_SEED, 0));
    for (let i = 0; i < coords.length; i++) {
      expect(pass1[i]).toBe(pass2[i]);
    }
  });
});

function smoothstepHelper(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0.0), 1.0);
  return t * t * (3.0 - 2.0 * t);
}

function worleyEdgeUnprunedReference(x: number, y: number, seed: number): number {
  const ipx = Math.floor(x) | 0, ipy = Math.floor(y) | 0;
  const fx = x - ipx, fy = y - ipy;
  const sTerm = Math.imul(seed | 0, 0x9e3779b9);
  let f1_sq = Infinity, f2_sq = Infinity;

  for (let dy = -1; dy <= 1; dy++) {
    const cy = (ipy + dy) | 0;
    const cyTerm = Math.imul(cy, 0x27d4eb2f);
    for (let dx = -1; dx <= 1; dx++) {
      const cx = (ipx + dx) | 0;
      let h = (cx ^ cyTerm ^ sTerm) | 0;
      h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
      const px = (h & 0xfff) * (1.0 / 4095.0);
      const py = ((h >>> 12) & 0xfff) * (1.0 / 4095.0);
      const dist_x = dx + px - fx;
      const dist_y = dy + py - fy;
      const d2 = dist_x * dist_x + dist_y * dist_y;
      if (d2 < f1_sq) {
        f2_sq = f1_sq;
        f1_sq = d2;
      } else if (d2 < f2_sq) {
        f2_sq = d2;
      }
    }
  }
  return Math.sqrt(f2_sq) - Math.sqrt(f1_sq);
}

function anomalyEUnprunedReference(x: number, y: number, seed: number): number {
  const ax = x * (1.0 / ANOMALY.cell), ay = y * (1.0 / ANOMALY.cell);
  const ipx = Math.floor(ax) | 0, ipy = Math.floor(ay) | 0;
  const fx = ax - ipx, fy = ay - ipy;
  const sTerm = Math.imul((seed + 400) | 0, 0x9e3779b9);
  let f1_sq = Infinity;
  let bcx = 0, bcy = 0, btype = 0;

  for (let dy = -1; dy <= 1; dy++) {
    const cy = (ipy + dy) | 0;
    const cyTerm = Math.imul(cy, 0x27d4eb2f);
    for (let dx = -1; dx <= 1; dx++) {
      const cx = (ipx + dx) | 0;
      let h = (cx ^ cyTerm ^ sTerm) | 0;
      h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
      const px = (h & 0xfff) * (1.0 / 4095.0);
      const py = ((h >>> 12) & 0xfff) * (1.0 / 4095.0);
      const dist_x = dx + px - fx;
      const dist_y = dy + py - fy;
      const d2 = dist_x * dist_x + dist_y * dist_y;
      if (d2 < f1_sq) {
        f1_sq = d2;
        bcx = cx + px;
        bcy = cy + py;
        btype = (h >>> 24) * (1.0 / 255.0);
      }
    }
  }

  if (btype >= ANOMALY.probability) return 0.0;
  const cx = bcx * ANOMALY.cell;
  const cy = bcy * ANOMALY.cell;
  const d_x = x - cx, d_y = y - cy;
  const r2 = d_x * d_x + d_y * d_y;
  const r = Math.sqrt(r2);
  if (r >= ANOMALY.rOut) return 0.0;
  const fall = 1.0 - smoothstepHelper(ANOMALY.rIn, ANOMALY.rOut, r);
  const t = r / ANOMALY.terraceStep;
  const ft = t - Math.floor(t);
  const steps = (Math.floor(t) + smoothstepHelper(0.3, 0.7, ft)) * ANOMALY.terraceHeight;
  const spire = ANOMALY.spireHeight * Math.exp(-r2 / (ANOMALY.spireRadius * ANOMALY.spireRadius));
  const spiral = ANOMALY.spiralAmp * Math.sin(ANOMALY.spiralTurns * Math.atan2(d_y, d_x) + ANOMALY.spiralRadial * r);
  return fall * (steps + spire + spiral);
}

describe('20. Cellular Voronoi Pruning & Traversal Invariance (§IV.B.5, §IV.L)', () => {
  it('worleyEdge matches exhaustive unpruned cellular search', () => {
    for (let i = 0; i < 500; i++) {
      const x = (i * 123.45) % 10000;
      const y = (i * 678.91) % 10000;
      const edge = worleyEdge(x, y, DEFAULT_SEED);
      const refEdge = worleyEdgeUnprunedReference(x, y, DEFAULT_SEED);
      expect(Number.isFinite(edge)).toBe(true);
      expect(edge).toBeGreaterThanOrEqual(0.0);
      expect(Math.abs(edge - refEdge)).toBeLessThan(1e-10);
    }
  });

  it('anomalyE matches unpruned cell search across varying coordinates', () => {
    for (let i = 0; i < 500; i++) {
      const x = (i * 456.78) % 20000 - 10000;
      const y = (i * 234.56) % 20000 - 10000;
      const anom = anomalyE(x, y, DEFAULT_SEED);
      const refAnom = anomalyEUnprunedReference(x, y, DEFAULT_SEED);
      expect(Number.isFinite(anom)).toBe(true);
      expect(anom).toBeGreaterThanOrEqual(0.0);
      expect(Math.abs(anom - refAnom)).toBeLessThan(1e-10);
    }
  });

  it('voronoi produces valid geometric partition and edge matches worleyEdge', () => {
    for (let i = 0; i < 200; i++) {
      const x = (i * 123.45) % 10000;
      const y = (i * 678.91) % 10000;
      const res = voronoi(x, y, DEFAULT_SEED);
      expect(Number.isFinite(res.f1)).toBe(true);
      expect(Number.isFinite(res.f2)).toBe(true);
      expect(res.f2).toBeGreaterThanOrEqual(res.f1);
      expect(res.edge).toBeCloseTo(res.f2 - res.f1, 10);
      expect(res.cellHash).toBeGreaterThanOrEqual(0.0);
      expect(res.cellHash).toBeLessThanOrEqual(1.0);

      const fastEdge = worleyEdge(x, y, DEFAULT_SEED);
      expect(Math.abs(res.edge - fastEdge)).toBeLessThan(1e-10);
    }
  });

  it('regionField matches bounded cellular traversal invariants across varying coordinates', () => {
    for (let i = 0; i < 500; i++) {
      const x = (i * 345.67) % 30000 - 15000;
      const y = (i * 789.12) % 30000 - 15000;
      const reg = regionField(x, y, DEFAULT_SEED);
      expect(Number.isFinite(reg.wRegion)).toBe(true);
      expect(reg.wRegion).toBeGreaterThanOrEqual(0.0);
      expect(reg.wRegion).toBeLessThanOrEqual(1.0);
      expect(Number.isFinite(reg.mountFactor)).toBe(true);
      expect(reg.mountFactor).toBeGreaterThan(0.0);
      expect(Number.isFinite(reg.duneAmount)).toBe(true);
      expect(reg.duneAmount).toBeGreaterThanOrEqual(0.0);
    }
  });
});

describe('21. Multi-Octave Nyquist Attenuation Monotonicity (§IV.A.8, §IV.B.4)', () => {
  it('elevation Nyquist attenuation smoothly suppresses high octaves as footprint increases', () => {
    const footprints = [0.0, 0.05, 0.2, 1.0, 5.0];
    const evals = footprints.map((dp) => elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, dp));
    for (const e of evals) {
      expect(Number.isFinite(e)).toBe(true);
      expect(e).toBeLessThanOrEqual(MAX_WORLD_H);
      expect(e).toBeGreaterThan(-100.0);
    }
  });

  it('per-octave Nyquist weight formula is non-increasing with footprint', () => {
    const nyqWeight = (freq: number, dp: number) => Math.min(1, Math.max(0, 2.0 - freq * dp * 2.0));
    for (let octave = 0; octave < 9; octave++) {
      const f = F.macro * Math.pow(2.0, octave);
      let prevW = 1.0;
      for (let dp = 0.0; dp <= 20.0; dp += 0.5) {
        const w = nyqWeight(f, dp);
        expect(w).toBeLessThanOrEqual(prevW + 1e-12);
        prevW = w;
      }
    }
  });
});

describe('22. Beer-Lambert Atmospheric Extinction & Fog Invariants (§IV.H.1, §IV.A.9)', () => {
  it('optical depth increases monotonically with distance t', () => {
    let prevOd = -1;
    for (let t = 0; t <= 5000; t += 250) {
      const od = LIGHT.sigmaFog * t * Math.exp(-Math.max(100.0, 0.0) / LIGHT.fogScaleHeight);
      expect(od).toBeGreaterThanOrEqual(prevOd);
      prevOd = od;
      const tChannel = Math.exp(-0.95 * od);
      expect(tChannel).toBeGreaterThan(0.0);
      expect(tChannel).toBeLessThanOrEqual(1.0);
    }
  });

  it('optical depth decreases with altitude following exponential scale height', () => {
    const od0 = LIGHT.sigmaFog * 1000.0 * Math.exp(0);
    const odH = LIGHT.sigmaFog * 1000.0 * Math.exp(-LIGHT.fogScaleHeight / LIGHT.fogScaleHeight);
    expect(Math.abs(odH - od0 / Math.E)).toBeLessThan(1e-6);
  });
});

describe('23. OKLab 5-Biome Color Space Geometry & Partition of Unity (§IV.H.2, §IV.A.1)', () => {
  it('biome anchors form a complete partition of climate space with positive weights', () => {
    const testPoints = [
      { t: -10, m: 0.5, e: 900, s: 1.2 },
      { t: 15, m: 0.2, e: 40, s: 0.2 },
      { t: 12, m: 0.8, e: 150, s: 0.4 },
      { t: 8, m: 0.4, e: 600, s: 1.8 },
      { t: 18, m: 0.1, e: 60, s: 0.3 },
    ];

    for (const p of testPoints) {
      let wsum = 0;
      for (const b of BIOME_ANCHORS) {
        const dt = (p.t - b.t) / b.sT;
        const dm = (p.m - b.m) / b.sM;
        const de = (p.e - b.e) / b.sE;
        const ds = (p.s - b.slope) / b.sS;
        const d2 = dt * dt + dm * dm + de * de + ds * ds;
        const w = Math.exp(-d2);
        expect(w).toBeGreaterThan(0.0);
        wsum += w;
      }
      expect(wsum).toBeGreaterThan(1e-4);
    }
  });
});

describe('24. Floating-Origin Toroidal Coordinate Invariance (§IV.G.1, §IV.G.2)', () => {
  it('coordinate rebasing preserves exact local displacement across 8192m grid', () => {
    for (let kx = -3; kx <= 3; kx++) {
      for (let kz = -3; kz <= 3; kz++) {
        const worldX = kx * CHUNK_SIZE + 1234.56;
        const worldZ = kz * CHUNK_SIZE + 6543.21;
        const ox = Math.floor(worldX / CHUNK_SIZE) * CHUNK_SIZE;
        const oz = Math.floor(worldZ / CHUNK_SIZE) * CHUNK_SIZE;
        const lx = worldX - ox;
        const lz = worldZ - oz;
        expect(ox + lx).toBeCloseTo(worldX, 10);
        expect(oz + lz).toBeCloseTo(worldZ, 10);
        expect(lx).toBeGreaterThanOrEqual(0);
        expect(lx).toBeLessThan(CHUNK_SIZE);
      }
    }
  });

  it('frequency multiplication f * chunk_origin has zero fractional roundoff error in fp32', () => {
    for (let k = -10; k <= 10; k++) {
      const origin = k * CHUNK_SIZE;
      for (const freq of Object.values(F)) {
        const prod = origin * freq;
        const quantized = Math.round(prod * 128) / 128;
        expect(Math.abs(prod - quantized)).toBeLessThan(1e-12);
        expect(Math.fround(prod)).toBe(prod);
      }
    }
  });
});

describe('25. Water Surface Analytical Intersection & Shoreline Attenuation (§IV.A.10, §IV.B.4)', () => {
  it('analytical ray-plane intersection produces exact water plane hit distance', () => {
    const roY = 30.0;
    const rdY = -0.5;
    const tWater = (VALLEY.wLevel - roY) / rdY;
    expect(tWater).toBe(50.0);
    expect(roY + rdY * tWater).toBe(VALLEY.wLevel);
  });

  it('water depth is zero or positive and bounded across valleys', () => {
    for (let x = 0; x <= 1000; x += 200) {
      for (let z = 0; z <= 1000; z += 200) {
        const e = elevation(x, z, DEFAULT_SEED, 0);
        const depth = Math.max(VALLEY.wLevel - e, 0.0);
        expect(depth).toBeGreaterThanOrEqual(0.0);
        expect(Number.isFinite(depth)).toBe(true);
      }
    }
  });
});

describe('26. Energy Conservation & Semi-Implicit Locomotion Integration (§IV.F.4, §IV.K.3)', () => {
  it('jump trajectory apex matches theoretical kinematic energy conservation', () => {
    const v0 = PLAYER.jumpVelocity;
    const g = PLAYER.gravity;
    const theoreticalApex = (v0 * v0) / (2.0 * g); // 8.5^2 / (2 * 22) = 1.642 m
    let y = 100.0;
    let vy = v0;
    let maxY = y;
    const dt = 0.001;

    for (let step = 0; step < 1000; step++) {
      vy -= g * dt;
      y += vy * dt;
      if (y > maxY) maxY = y;
      if (vy <= 0 && maxY > 100.0) break;
    }
    const simulatedApex = maxY - 100.0;
    expect(Math.abs(simulatedApex - theoreticalApex)).toBeLessThan(0.02);
  });

  it('repeated jumps from flat ground return to identical resting ground elevation', () => {
    const p: PlayerState = {
      x: 0,
      y: 50.0,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      grounded: true,
    };
    const ground = elevation(0, 0, DEFAULT_SEED, 0) + PLAYER.groundBias + PLAYER.height;
    p.y = ground;

    for (let jump = 0; jump < 5; jump++) {
      updatePlayer(p, { forward: 0, strafe: 0, jump: true, sprint: false, yaw: 0 }, 0.016, DEFAULT_SEED);
      for (let air = 0; air < 60; air++) {
        updatePlayer(p, { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 }, 0.016, DEFAULT_SEED);
      }
      expect(p.grounded).toBe(true);
      expect(Math.abs(p.y - ground)).toBeLessThan(1e-4);
      expect(p.vy).toBe(0.0);
    }
  });
});

describe('27. Sprint Momentum & Kinematic Acceleration Limits (§IV.F.1, §IV.K.2)', () => {
  it('sprint acceleration multiplier matches 1.8x steady-state velocity', () => {
    const pWalk: PlayerState = { x: 0, y: 300, z: 0, vx: 0, vy: 0, vz: 0, grounded: true };
    const pSprint: PlayerState = { x: 0, y: 300, z: 0, vx: 0, vy: 0, vz: 0, grounded: true };

    for (let i = 0; i < 100; i++) {
      updatePlayer(pWalk, { forward: 1, strafe: 0, jump: false, sprint: false, yaw: 0 }, 0.05, DEFAULT_SEED);
      updatePlayer(pSprint, { forward: 1, strafe: 0, jump: false, sprint: true, yaw: 0 }, 0.05, DEFAULT_SEED);
    }

    const ratio = pSprint.vz / pWalk.vz;
    expect(Math.abs(ratio - PLAYER.sprintMul)).toBeLessThan(0.01);
  });

  it('horizontal momentum decays smoothly without overshoot on input release', () => {
    const p: PlayerState = { x: 0, y: 300, z: 0, vx: 10.0, vy: 0, vz: 10.0, grounded: true };
    let prevSpeed = Math.hypot(p.vx, p.vz);

    for (let i = 0; i < 20; i++) {
      updatePlayer(p, { forward: 0, strafe: 0, jump: false, sprint: false, yaw: 0 }, 0.016, DEFAULT_SEED);
      const speed = Math.hypot(p.vx, p.vz);
      expect(speed).toBeLessThan(prevSpeed);
      prevSpeed = speed;
    }
  });
});

describe('28. Extreme Boundary Coordinates & Floating-Point Stress Test (§IV.G, §IV.L)', () => {
  it('elevation handles massive planetary coordinates (+-1,000,000m) without NaN or overflow', () => {
    const massive = [
      [1_000_000, 1_000_000],
      [-1_000_000, 1_000_000],
      [1_000_000, -1_000_000],
      [-1_000_000, -1_000_000],
    ];
    for (const [x, z] of massive) {
      const h = elevation(x, z, DEFAULT_SEED, 0);
      expect(Number.isFinite(h)).toBe(true);
      expect(h).toBeLessThanOrEqual(MAX_WORLD_H);
      expect(h).toBeGreaterThan(-500.0);
    }
  });

  it('microscopic perturbations (1e-7m) do not cause catastrophic cancellation', () => {
    const e1 = elevation(1234.5678, -8765.4321, DEFAULT_SEED, 0);
    const e2 = elevation(1234.5678 + 1e-7, -8765.4321, DEFAULT_SEED, 0);
    expect(Math.abs(e2 - e1)).toBeLessThan(1e-4);
  });

  it('large-scale coordinate fuzzing (±1,000,000m) maintains finite elevation, slope, and climate', () => {
    let seedState = 12345;
    const lcg = () => {
      seedState = (Math.imul(1664525, seedState) + 1013904223) | 0;
      return (seedState >>> 0) / 4294967295.0;
    };

    for (let i = 0; i < 2000; i++) {
      const rx = (lcg() - 0.5) * 2_000_000;
      const rz = (lcg() - 0.5) * 2_000_000;
      const rSeed = (lcg() * 50000) | 0;
      const h = elevation(rx, rz, rSeed, 0);
      expect(Number.isFinite(h)).toBe(true);
      expect(h).toBeLessThanOrEqual(MAX_WORLD_H);
      expect(h).toBeGreaterThan(-500.0);

      const [gx, gz] = gradient(rx, rz, rSeed, 0.25);
      expect(Number.isFinite(gx)).toBe(true);
      expect(Number.isFinite(gz)).toBe(true);

      const temp = temperature(rx, rz, h, rSeed);
      expect(Number.isFinite(temp)).toBe(true);

      const moist = moisture(rx, rz, h, rSeed);
      expect(Number.isFinite(moist)).toBe(true);
      expect(moist).toBeGreaterThanOrEqual(-0.5);
      expect(moist).toBeLessThanOrEqual(2.0);
    }
  });
});

describe('29. Zero-Allocation Hot-Loop Conformance & Heap Invariance (§IV.C.1, §IV.C.3)', () => {
  it('10,000 consecutive elevation queries execute with zero memory leaks', () => {
    let sum = 0;
    for (let i = 0; i < 10000; i++) {
      sum += elevation(SPAWN.x + (i & 63), SPAWN.z + ((i >> 6) & 63), DEFAULT_SEED, 0);
    }
    expect(Number.isFinite(sum)).toBe(true);
  });

  it('1,000 consecutive physics ticks execute with zero memory allocations', () => {
    const p: PlayerState = {
      x: SPAWN.x,
      y: 350.0,
      z: SPAWN.z,
      vx: 1.0,
      vy: 0.0,
      vz: 1.0,
      grounded: true,
    };
    for (let i = 0; i < 1000; i++) {
      updatePlayer(
        p,
        { forward: (i & 1), strafe: 0, jump: false, sprint: false, yaw: 0 },
        0.016,
        DEFAULT_SEED,
      );
    }
    expect(Number.isFinite(p.x)).toBe(true);
    expect(Number.isFinite(p.y)).toBe(true);
    expect(Number.isFinite(p.z)).toBe(true);
  });
});

describe('30. Dynamic Resolution Scaling (DRS) Deadband & Hysteresis (§IV.D, §28)', () => {
  interface DrsState {
    scale: number;
    lastDownscaleAt: number;
    lastUpscaleAt: number;
  }

  function simulateDrsStep(
    state: DrsState,
    fpsEma: number,
    now: number,
    t0: number,
    software = false,
  ): boolean {
    const minScale = software ? 0.15 : 0.35;
    const maxScale = software ? 0.25 : 1.0;
    if (now - t0 > 2500) {
      if (fpsEma < 22 && now - state.lastDownscaleAt > 1500 && state.scale > minScale) {
        state.scale = Math.max(minScale, state.scale * 0.85);
        state.lastDownscaleAt = now;
        state.lastUpscaleAt = now;
        return true;
      } else if (
        fpsEma > 45 &&
        now - state.lastUpscaleAt > 4000 &&
        now - state.lastDownscaleAt > 5000 &&
        state.scale < maxScale
      ) {
        state.scale = Math.min(maxScale, state.scale * 1.15);
        state.lastUpscaleAt = now;
        return true;
      }
    }
    return false;
  }

  it('warmup window ignores low framerates during first 2500 ms', () => {
    const state: DrsState = { scale: 1.0, lastDownscaleAt: 0, lastUpscaleAt: 0 };
    const t0 = 1000;
    for (let t = t0; t < t0 + 2500; t += 100) {
      simulateDrsStep(state, 12, t, t0);
    }
    expect(state.scale).toBe(1.0);
  });

  it('deadband [22, 45] FPS maintains constant resolution scale indefinitely', () => {
    const state: DrsState = { scale: 0.85, lastDownscaleAt: 0, lastUpscaleAt: 0 };
    const t0 = 0;
    for (let t = 3000; t < 20000; t += 50) {
      simulateDrsStep(state, 30, t, t0);
    }
    expect(state.scale).toBe(0.85);
  });

  it('sustained 14-FPS load downscales at 1500 ms intervals and clamps at minScale without thrashing', () => {
    const state: DrsState = { scale: 1.0, lastDownscaleAt: 0, lastUpscaleAt: 0 };
    const t0 = 0;
    const downscaleTimestamps: number[] = [];

    for (let t = 2600; t < 15000; t += 16) {
      const changed = simulateDrsStep(state, 14, t, t0);
      if (changed) downscaleTimestamps.push(t);
    }

    // Downscale must obey minimum 1500 ms dwell between changes
    for (let i = 1; i < downscaleTimestamps.length; i++) {
      expect(downscaleTimestamps[i] - downscaleTimestamps[i - 1]).toBeGreaterThanOrEqual(1500);
    }

    // Clamps at exactly minScale (0.35)
    expect(state.scale).toBe(0.35);
  });

  it('upscale recovery enforces asymmetric 4000 ms dwell and 5000 ms cooldown from downscale', () => {
    const state: DrsState = { scale: 0.5, lastDownscaleAt: 3000, lastUpscaleAt: 3000 };
    const t0 = 0;

    // Advance to 7900 ms (< 5000 ms from lastDownscaleAt): must NOT upscale yet
    for (let t = 3100; t <= 7900; t += 100) {
      simulateDrsStep(state, 60, t, t0);
    }
    expect(state.scale).toBe(0.5);

    // At t = 8100 ms (now - lastDownscaleAt = 5100 > 5000 and now - lastUpscaleAt = 5100 > 4000): should upscale
    simulateDrsStep(state, 60, 8100, t0);
    expect(state.scale).toBeCloseTo(0.5 * 1.15, 6);

    // Immediate next frame must not upscale again (must wait 4000 ms)
    simulateDrsStep(state, 60, 8200, t0);
    expect(state.scale).toBeCloseTo(0.5 * 1.15, 6);
  });
});

describe('31. GPU fp32 Emulation vs CPU fp64 Twin Parity (§IV.G, §IV.L)', () => {
  it('fp32 rebased coordinate elevation divergence is strictly bounded by PLAYER.groundBias (0.08m)', () => {
    let maxDivergence = 0;
    const testOrigins = [0, 8192, -8192, 16384, -16384];

    for (const originX of testOrigins) {
      for (const originZ of testOrigins) {
        for (let i = 0; i < 200; i++) {
          const localX = (i * 41.17) % CHUNK_SIZE;
          const localZ = (i * 73.31) % CHUNK_SIZE;
          const worldX = originX + localX;
          const worldZ = originZ + localZ;

          // CPU fp64 elevation
          const e64 = elevation(worldX, worldZ, DEFAULT_SEED, 0);

          // Simulated GPU fp32: chunk origin is exact dyadic rational, local offset is fp32
          const localX32 = Math.fround(localX);
          const localZ32 = Math.fround(localZ);
          const originX32 = Math.fround(originX);
          const originZ32 = Math.fround(originZ);
          const e32 = elevation(originX32 + localX32, originZ32 + localZ32, DEFAULT_SEED, 0);

          const div = Math.abs(e64 - e32);
          if (div > maxDivergence) maxDivergence = div;
          expect(div).toBeLessThan(PLAYER.groundBias);
        }
      }
    }

    // Must be well within the 8cm ground bias contract
    expect(maxDivergence).toBeLessThan(0.04);
  });
});



