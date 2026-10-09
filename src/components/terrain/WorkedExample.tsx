'use client';

/**
 * Worked numerical example (spec §30) — the Lipschitz march, instrumented.
 * ---------------------------------------------------------------------------
 * The camera stands WORKED_EXAMPLE.camHeight above the tuned spawn, aimed by
 * a pitch search toward the vista ridge so the hit lands near the nominal
 * t* from the spec. marchRay() records every step: watch the gap (P.y − E)
 * shrink, the Nyquist weights w collapse as the pixel footprint Δp grows,
 * and the bisection polish the bracket to sub-metre precision.
 */

import { useState } from 'react';
import { elevation, marchRay } from '@/lib/terrainMath';
import { DEFAULT_SEED, SPAWN, CAMERA, WORKED_EXAMPLE } from '@/lib/constants';

type Data = {
  tStar: number;
  pitch: number;
  stepsUsed: number;
  rows: ReturnType<typeof marchRay>['rows'];
  bisect: { i: number; t: number; gap: number }[];
  hit: boolean;
};


/** One-shot deterministic computation — the §30 march with recording on. */
function compute(): Data {
  const foot = (2 * Math.tan((CAMERA.fov * Math.PI / 180) * 0.5)) / WORKED_EXAMPLE.resY;
  const camY = elevation(SPAWN.x, SPAWN.z, DEFAULT_SEED, 0) + WORKED_EXAMPLE.camHeight;
  const ro: [number, number, number] = [SPAWN.x, camY, SPAWN.z];
  const cp = Math.cos(WORKED_EXAMPLE.pitch), sp = Math.sin(WORKED_EXAMPLE.pitch);
  const rd: [number, number, number] = [Math.sin(WORKED_EXAMPLE.yaw) * cp, sp, Math.cos(WORKED_EXAMPLE.yaw) * cp];

  const rec = marchRay(ro, rd, DEFAULT_SEED, {
    steps: 96, footprint: foot, record: true,
  });
  return {
    tStar: rec.t,
    pitch: WORKED_EXAMPLE.pitch,
    stepsUsed: rec.rows.length,
    rows: rec.rows,
    bisect: rec.bisectRows,
    hit: rec.hit,
  };
}

export default function WorkedExample() {
  // Deterministic derived data — computed once through lazy state init.
  const [data] = useState<Data>(compute);

  const fmt = (v: number, d = 1) => v.toFixed(d);

  return (
    <div className="rounded-xl border border-stone-800 bg-stone-950 p-4">
      <div className="grid grid-cols-2 gap-3 font-mono text-[11px] text-stone-300 sm:grid-cols-4">
        <div className="rounded-lg bg-stone-900/70 p-2">
          <div className="text-stone-500">camera height</div>
          <div className="text-emerald-300">{WORKED_EXAMPLE.camHeight} m above E</div>
        </div>
        <div className="rounded-lg bg-stone-900/70 p-2">
          <div className="text-stone-500">aim pitch (searched)</div>
          <div className="text-emerald-300">{fmt(data.pitch * 180 / Math.PI, 2)}°</div>
        </div>
        <div className="rounded-lg bg-stone-900/70 p-2">
          <div className="text-stone-500">march samples</div>
          <div className="text-emerald-300">{data.stepsUsed} + {data.bisect.length} bisect</div>
        </div>
        <div className="rounded-lg bg-stone-900/70 p-2">
          <div className="text-stone-500">t* (hit distance)</div>
          <div className="text-amber-300">{fmt(data.tStar)} m</div>
        </div>
      </div>

      <div className="mt-3 max-h-80 overflow-y-auto rounded-lg border border-stone-800">
        <table className="w-full border-collapse font-mono text-[10.5px]">
          <thead className="sticky top-0 bg-stone-900 text-stone-400">
            <tr>
              <th className="px-2 py-1.5 text-left font-normal">i</th>
              <th className="px-2 py-1.5 text-right font-normal">t (m)</th>
              <th className="px-2 py-1.5 text-right font-normal">P.y − E</th>
              <th className="px-2 py-1.5 text-right font-normal">Δp(t)</th>
              <th className="px-2 py-1.5 text-right font-normal">‖∇E‖</th>
              <th className="px-2 py-1.5 text-right font-normal">Δt</th>
              <th className="px-2 py-1.5 text-right font-normal">w₃</th>
              <th className="px-2 py-1.5 text-right font-normal">w₆</th>
              <th className="px-2 py-1.5 text-right font-normal">w₉</th>
              <th className="px-2 py-1.5 text-right font-normal">wr₂</th>
              <th className="px-2 py-1.5 text-right font-normal">wr₅</th>
            </tr>
          </thead>
          <tbody className="text-stone-300">
            {data.rows.map((r) => (
              <tr key={r.i} className="border-t border-stone-800/60">
                <td className="px-2 py-1 text-stone-500">{r.i}</td>
                <td className="px-2 py-1 text-right">{fmt(r.t)}</td>
                <td className={`px-2 py-1 text-right ${r.gap < 50 ? 'text-amber-300' : ''}`}>
                  {fmt(r.gap)}
                </td>
                <td className="px-2 py-1 text-right text-stone-500">{fmt(r.dp, 2)}</td>
                <td className="px-2 py-1 text-right text-stone-500">{fmt(r.grad, 2)}</td>
                <td className="px-2 py-1 text-right">{fmt(r.step)}</td>
                <td className="px-2 py-1 text-right text-stone-500">{r.wMacro[2].toFixed(2)}</td>
                <td className="px-2 py-1 text-right text-stone-500">{r.wMacro[5].toFixed(2)}</td>
                <td className="px-2 py-1 text-right text-stone-500">{r.wMacro[8].toFixed(2)}</td>
                <td className="px-2 py-1 text-right text-stone-500">{r.wRidge[1].toFixed(2)}</td>
                <td className="px-2 py-1 text-right text-stone-500">{r.wRidge[4].toFixed(2)}</td>
              </tr>
            ))}
            {data.bisect.map((b) => (
              <tr key={`b${b.i}`} className="border-t border-emerald-900/50 bg-emerald-950/20">
                <td className="px-2 py-1 text-emerald-500">b{b.i}</td>
                <td className="px-2 py-1 text-right text-emerald-300">{fmt(b.t, 3)}</td>
                <td className="px-2 py-1 text-right text-emerald-300">{fmt(b.gap, 3)}</td>
                <td colSpan={8} className="px-2 py-1 text-stone-500">bisection refinement (§16)</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 font-mono text-[10px] leading-relaxed text-stone-500">
        w₃/w₆/w₉ = Nyquist weights of macro-fBm octaves 3/6/9 · wr = ridged
        (2f) weights · Δp = t·2tan(FOV/2)/Res_y with Res_y = {WORKED_EXAMPLE.resY}
        (half-res target). Steps shrink near the ridge as ‖∇E‖ feeds the
        Lipschitz denominator; the horizon clamp caps each step at 6% of t.
      </p>
    </div>
  );
}
