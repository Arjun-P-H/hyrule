import { elevation, marchRay } from "../src/lib/terrainMath.ts";
const SEED = 1337;
const cam: [number,number,number] = [2016, 75.8, -2592];
const footprint = 2*Math.tan(35*Math.PI/180)/540;

const targets: [number,number,number][] = [
  [-2336, 637, 3936], [-1440, 622, 3936], [-3104, 654, 3936], [-4000, 649, 1376],
  [-2720, 642, 3936], [4913, 300, -1673], [1351, 200, -71],
];
for (const tg of targets) {
  const d = [tg[0]-cam[0], tg[1]-cam[1], tg[2]-cam[2]];
  const L = Math.hypot(d[0], d[1], d[2]);
  const rd: [number,number,number] = [d[0]/L, d[1]/L, d[2]/L];
  const res = marchRay(cam, rd, SEED, { steps: 128, footprint, record: true });
  const hitP = res.hit ? [cam[0]+rd[0]*res.t, cam[1]+rd[1]*res.t, cam[2]+rd[2]*res.t] : null;
  const e = hitP ? elevation(hitP[0], hitP[2], SEED, res.t*footprint) : NaN;
  console.log(`target (${tg}) dist=${L.toFixed(0)} -> hit=${res.hit} t*=${res.t.toFixed(0)} at (${hitP?.map(v=>v.toFixed(0)).join(",")}) E=${e.toFixed(0)} steps_recorded=${res.rows.length}`);
  if (res.hit && res.t > 2500) {
    // show rows near the hit
    const rows = res.rows.slice(-6);
    for (const r of rows) {
      console.log(`  i=${r.i} t=${r.t.toFixed(0)} dp=${r.dp.toFixed(2)} |g|=${r.grad.toFixed(2)} step=${r.step.toFixed(1)} wM9=${r.wMacro[8].toFixed(2)} wM8=${r.wMacro[7].toFixed(2)} wVeg=${Math.max(0, Math.min(1, 2 - 14*r.dp/0.5)).toFixed(2)}`);
    }
    console.log("  bisect:", res.bisectRows.map(b=>`${b.t.toFixed(1)}(${b.gap>=0?"+":"-"}${Math.abs(b.gap).toFixed(2)})`).join(" "));
  }
}
