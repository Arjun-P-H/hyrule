import { elevation, voronoi, regionField } from "../src/lib/terrainMath.ts";
const SEED = 1337;
console.log("E(-1696,3936, dp=0) =", elevation(-1696, 3936, SEED, 0).toFixed(1));
console.log("E(-1696,3936, dp=5.7) =", elevation(-1696, 3936, SEED, 5.7).toFixed(1));
console.log("E(-1717,3946, dp=0) =", elevation(-1717, 3946, SEED, 0).toFixed(1));
for (let dx = -60; dx <= 20; dx += 10) {
  const e = elevation(-1696 + dx, 3936, SEED, 0);
  const reg = regionField(-1696 + dx, 3936, SEED);
  const v = voronoi((-1696 + dx) / 2048, 3936 / 2048, SEED + 400);
  console.log(`x=${-1696+dx} E=${e.toFixed(1)} region(factor=${reg.mountFactor.toFixed(2)}, w=${reg.wRegion.toFixed(2)}) anomHash=${v.cellHash.toFixed(3)}`);
}
