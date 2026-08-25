import { elevation } from "../src/lib/terrainMath.ts";
const SEED = 1337;
const ro = [224, 75.0, 3040];
const rd = [-0.8824, 0.2278, 0.4118];
for (let t = 0; t <= 2600; t += 100) {
  const px = ro[0] + rd[0]*t, py = ro[1] + rd[1]*t, pz = ro[2] + rd[2]*t;
  const e = elevation(px, pz, SEED, Math.max(t*0.002597, 1e-4));
  console.log(`t=${String(t).padStart(5)} P=(${px.toFixed(0)}, ${py.toFixed(0)}, ${pz.toFixed(0)}) E=${e.toFixed(1)} gap=${(py-e).toFixed(1)}`);
}
