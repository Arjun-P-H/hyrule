/* Scratch tuning script — not part of the deliverable. Lives in .scratch/. */
import { elevation, slope, voronoi, marchRay } from "../src/lib/terrainMath.ts";

const SEED = 1337;

// 1) World statistics + find tall peaks & water within a 8km window
let peaks: {x:number,z:number,e:number}[] = [];
let waters: {x:number,z:number,e:number}[] = [];
for (let z = -4000; z <= 4000; z += 128) {
  for (let x = -4000; x <= 4000; x += 128) {
    const e = elevation(x, z, SEED, 0);
    if (e > 480) peaks.push({x, z, e});
    if (e < 3) waters.push({x, z, e});
  }
}
peaks.sort((a,b)=>b.e-a.e);
console.log("peaks found:", peaks.length, "waters:", waters.length);
console.log("top peaks:", peaks.slice(0,8).map(p=>`(${p.x},${p.z}) ${p.e.toFixed(0)}m`).join("  "));

// anomaly cells in window
const anomalies: {x:number,z:number}[] = [];
for (let cz = -3; cz <= 3; cz++) for (let cx = -3; cx <= 3; cx++) {
  const v = voronoi(cx, cz, SEED + 400);
  if (v.cellHash < 0.06) anomalies.push({x: v.centerX*2048, z: v.centerY*2048});
}
console.log("anomaly centers:", anomalies.map(a=>`(${a.x.toFixed(0)},${a.z.toFixed(0)})`).join("  "));

// 2) Find a spawn: elevation 40..160, near (600..2200m) a >480m peak, near (<900m) water, flat
type Sp = {x:number,z:number,score:number,d:number};
const best: Sp[] = [];
for (let z = -4000; z <= 4000; z += 64) {
  for (let x = -4000; x <= 4000; x += 64) {
    const e = elevation(x, z, SEED, 0);
    if (e < 40 || e > 170) continue;
    let dPeak = 1e9, peak = peaks[0];
    for (const p of peaks.slice(0, 40)) {
      const d = Math.hypot(p.x - x, p.z - z);
      if (d < dPeak) { dPeak = d; peak = p; }
    }
    if (!peak || dPeak > 2200 || dPeak < 600) continue;
    let dWater = 1e9;
    for (const w of waters) {
      const d = Math.hypot(w.x - x, w.z - z);
      if (d < dWater) dWater = d;
    }
    if (dWater > 900) continue;
    const s = slope(x, z, SEED);
    if (s > 0.8) continue;
    best.push({ x, z, score: dPeak*0.3 + dWater - s*500, d: dPeak });
  }
}
best.sort((a,b)=>a.score-b.score);
console.log("best spawns (near peak+water, flat):");
for (const b of best.slice(0, 6)) {
  console.log(`  spawn (${b.x},${b.z}) E=${elevation(b.x,b.z,SEED,0).toFixed(1)} slope=${slope(b.x,b.z,SEED).toFixed(2)} dPeak=${b.d.toFixed(0)}`);
}

// 3) Worked-example ray: from best spawn toward its nearest peak
const sp = best[0] ?? { x: 620, z: -380 };
let peak = peaks[0], dBest = 1e9;
for (const p of peaks.slice(0, 40)) {
  const d = Math.hypot(p.x - sp.x, p.z - sp.z);
  if (d < dBest) { dBest = d; peak = p; }
}
const camY = elevation(sp.x, sp.z, SEED, 0) + 30;
console.log(`\nworked example: cam=(${sp.x}, ${camY.toFixed(1)}, ${sp.z}) target peak=(${peak.x},${peak.z}) h=${peak.e.toFixed(0)} dist=${dBest.toFixed(0)}`);
const dir = [peak.x - sp.x, peak.e - 25 - camY, peak.z - sp.z];
const L = Math.hypot(dir[0], dir[1], dir[2]);
const rd: [number,number,number] = [dir[0]/L, dir[1]/L, dir[2]/L];
console.log("rd =", rd.map(v=>v.toFixed(4)).join(", "));
const res = marchRay([sp.x, camY, sp.z], rd, SEED, { steps: 128, footprint: 2*Math.tan(35*Math.PI/180)/540, record: true });
console.log("hit:", res.hit, "t*=", res.t.toFixed(2));
if (res.hit) {
  for (const r of res.rows.slice(0, 12)) {
    console.log(`i=${String(r.i).padStart(2)} t=${r.t.toFixed(1).padStart(7)} P=(${r.px.toFixed(0)},${r.py.toFixed(0)},${r.pz.toFixed(0)}) E=${r.e.toFixed(1)} gap=${r.gap.toFixed(1)} dp=${r.dp.toFixed(2)} |g|=${r.grad.toFixed(2)} step=${r.step.toFixed(1)} wM=[${r.wMacro.map(w=>w.toFixed(2)).join(",")}] wR=[${r.wRidge.map(w=>w.toFixed(2)).join(",")}]`);
  }
  console.log("bisect:", res.bisectRows.map(b=>`${b.t.toFixed(2)}(${b.gap>=0?"+":"-"}${Math.abs(b.gap).toFixed(2)})`).join(" -> "));
}
