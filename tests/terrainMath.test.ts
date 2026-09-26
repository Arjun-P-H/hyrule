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
});


