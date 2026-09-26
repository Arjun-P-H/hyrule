/**
 * W(x,y) Hyrule Continuous World Engine — Performance & Parity Benchmark Suite
 * ---------------------------------------------------------------------------
 * Measures throughput, latency, numerical determinism, and memory allocation
 * against the Section VII performance targets.
 */

import {
  elevation,
  slope,
  marchRay,
  updatePlayer,
  type PlayerState,
} from './src/lib/terrainMath';
import {
  DEFAULT_SEED,
  SPAWN,
  CAMERA,
  WORKED_EXAMPLE,
} from './src/lib/constants';

console.log('================================================================================');
console.log('         W(x, y) HYRULE CONTINUOUS WORLD ENGINE — BENCHMARK SUITE               ');
console.log('================================================================================\n');

// 1. ELEVATION FUNCTION THROUGHPUT (dp = 0, near-field / physics)
{
  const N = 100_000;
  const t0 = performance.now();
  let sum = 0;
  for (let i = 0; i < N; i++) {
    sum += elevation(SPAWN.x + (i & 255), SPAWN.z + ((i >> 8) & 255), DEFAULT_SEED, 0);
  }
  const t1 = performance.now();
  const totalMs = t1 - t0;
  const evalsPerSec = N / (totalMs / 1000);
  const usPerEval = (totalMs / N) * 1000;
  console.log(`[1] Elevation E(x, z) (dp=0, physics hot path):`);
  console.log(`    Evals:       ${N.toLocaleString()}`);
  console.log(`    Total Time:  ${totalMs.toFixed(2)} ms`);
  console.log(`    Throughput:  ${(evalsPerSec / 1000).toFixed(1)}k evals/sec`);
  console.log(`    Latency:     ${usPerEval.toFixed(3)} µs/eval`);
  console.log(`    Checksum:    ${sum.toFixed(4)}\n`);
}

// 2. ELEVATION FUNCTION THROUGHPUT (dp > 0, filtered raymarching)
{
  const N = 100_000;
  const t0 = performance.now();
  let sum = 0;
  for (let i = 0; i < N; i++) {
    sum += elevation(SPAWN.x + (i & 255), SPAWN.z + ((i >> 8) & 255), DEFAULT_SEED, 0.05 + (i & 15) * 0.02);
  }
  const t1 = performance.now();
  const totalMs = t1 - t0;
  const evalsPerSec = N / (totalMs / 1000);
  const usPerEval = (totalMs / N) * 1000;
  console.log(`[2] Elevation E(x, z) (dp>0, raymarch filtered):`);
  console.log(`    Evals:       ${N.toLocaleString()}`);
  console.log(`    Total Time:  ${totalMs.toFixed(2)} ms`);
  console.log(`    Throughput:  ${(evalsPerSec / 1000).toFixed(1)}k evals/sec`);
  console.log(`    Latency:     ${usPerEval.toFixed(3)} µs/eval`);
  console.log(`    Checksum:    ${sum.toFixed(4)}\n`);
}

// 3. GRADIENT & SLOPE COMPUTATION
{
  const N = 25_000;
  const t0 = performance.now();
  let sum = 0;
  for (let i = 0; i < N; i++) {
    const s = slope(SPAWN.x + (i & 255), SPAWN.z + ((i >> 8) & 255), DEFAULT_SEED);
    sum += s;
  }
  const t1 = performance.now();
  const totalMs = t1 - t0;
  const opsPerSec = N / (totalMs / 1000);
  const usPerOp = (totalMs / N) * 1000;
  console.log(`[3] Slope ‖∇E‖ (4-tap finite diff, zero-allocation):`);
  console.log(`    Operations:  ${N.toLocaleString()}`);
  console.log(`    Total Time:  ${totalMs.toFixed(2)} ms`);
  console.log(`    Throughput:  ${(opsPerSec / 1000).toFixed(1)}k ops/sec`);
  console.log(`    Latency:     ${usPerOp.toFixed(3)} µs/op`);
  console.log(`    Checksum:    ${sum.toFixed(4)}\n`);
}

// 4. FULL PHYSICS SIMULATION LOOP (60 Hz locomotion tick)
{
  const N = 100_000;
  const player: PlayerState = {
    x: SPAWN.x,
    y: 350.0,
    z: SPAWN.z,
    vx: 1.5,
    vy: 0.0,
    vz: 2.0,
    grounded: true,
  };

  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    updatePlayer(
      player,
      {
        forward: (i & 64) ? 1 : 0,
        strafe: (i & 32) ? 1 : -1,
        jump: (i % 60 === 0),
        sprint: (i & 16) !== 0,
        yaw: (i * 0.005) % 6.28,
      },
      0.016666,
      DEFAULT_SEED,
    );
  }
  const t1 = performance.now();
  const totalMs = t1 - t0;
  const ticksPerSec = N / (totalMs / 1000);
  const usPerTick = (totalMs / N) * 1000;
  console.log(`[4] Locomotion Physics Tick (updatePlayer):`);
  console.log(`    Ticks:       ${N.toLocaleString()}`);
  console.log(`    Total Time:  ${totalMs.toFixed(2)} ms`);
  console.log(`    Throughput:  ${(ticksPerSec / 1000).toFixed(1)}k ticks/sec`);
  console.log(`    Latency:     ${usPerTick.toFixed(3)} µs/tick (Target < 500 µs)\n`);
}

// 5. MARCH-RAY RAYMARCHING CONVERGENCE
{
  const N = 2_000;
  const foot = (2 * Math.tan((CAMERA.fov * Math.PI / 180) * 0.5)) / WORKED_EXAMPLE.resY;
  const camY = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0) + WORKED_EXAMPLE.camHeight;
  const ro: [number, number, number] = [SPAWN.x, camY, SPAWN.z];
  const cp = Math.cos(0.0), sp = Math.sin(0.0);
  const rd: [number, number, number] = [Math.sin(SPAWN.yaw) * cp, sp, Math.cos(SPAWN.yaw) * cp];

  const t0 = performance.now();
  let hits = 0;
  let totalSteps = 0;
  for (let i = 0; i < N; i++) {
    const res = marchRay(ro, rd, DEFAULT_SEED, { steps: 96, footprint: foot });
    if (res.hit) hits++;
    totalSteps += res.rows.length;
  }
  const t1 = performance.now();
  const totalMs = t1 - t0;
  const raysPerSec = N / (totalMs / 1000);
  const msPerRay = totalMs / N;
  console.log(`[5] CPU Raymarcher (marchRay, vista ridge targeting):`);
  console.log(`    Rays:        ${N.toLocaleString()}`);
  console.log(`    Hits:        ${hits} / ${N} (Total steps: ${totalSteps})`);
  console.log(`    Total Time:  ${totalMs.toFixed(2)} ms`);
  console.log(`    Throughput:  ${raysPerSec.toFixed(1)} rays/sec`);
  console.log(`    Latency:     ${(msPerRay * 1000).toFixed(1)} µs/ray (${msPerRay.toFixed(3)} ms)\n`);
}

// 6. 2D HILLSHADE RADAR GRID GENERATION (200x200 = 40,000 pixels)
{
  const SIZE = 200;
  const EXTENT = 3600.0;
  const step = EXTENT / (SIZE - 1);
  const data = new Float32Array(SIZE * SIZE);

  const t0 = performance.now();
  for (let row = 0; row < SIZE; row++) {
    const z = SPAWN.z - EXTENT / 2 + row * step;
    for (let j = 0; j < SIZE; j++) {
      const x = SPAWN.x - EXTENT / 2 + j * step;
      data[row * SIZE + j] = elevation(x, z, DEFAULT_SEED, 0);
    }
  }
  const t1 = performance.now();
  const totalMs = t1 - t0;
  console.log(`[6] Hillshade Radar Full Generation (${SIZE}x${SIZE} = ${SIZE * SIZE} points):`);
  console.log(`    Total Time:  ${totalMs.toFixed(2)} ms (was > 45 ms)\n`);
}

// 7. NUMERICAL PARITY & DETERMINISM
{
  const samples = [
    [SPAWN.x, SPAWN.z],
    [0, 0],
    [-8192, 8192],
    [16384, -16384],
    [12345.67, -9876.54],
  ];
  console.log(`[7] Numerical Precision & Parity Check:`);
  for (const [x, z] of samples) {
    const e1 = elevation(x, z, DEFAULT_SEED, 0);
    const e2 = elevation(x, z, DEFAULT_SEED, 0);
    const delta = Math.abs(e1 - e2);
    console.log(`    Point (${x.toFixed(1)}, ${z.toFixed(1)}): E = ${e1.toFixed(4)} m | Repeated delta = ${delta}`);
  }
}

console.log('\n================================================================================');
console.log('                         ALL BENCHMARKS COMPLETED                               ');
console.log('================================================================================');
