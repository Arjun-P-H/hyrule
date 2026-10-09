/**
 * `bun run quality` — the Phase 6 code-quality gate, suitable for CI.
 * Fails (exit 1) on any TypeScript strict error, any circular dependency in
 * src/, any failing test, or any dead export in src/ that is neither a Next.js
 * entry point nor listed in JUSTIFIED below with its reason.
 */

import { runQuality } from './quality';

/** Exports kept on purpose although nothing imports them (ts-prune findings). */
const JUSTIFIED: Record<string, string> = {
  'src/lib/terrainMath.ts - hash3': '§4 public hash API; the shader mirrors it and tests pin its bit pattern',
  'src/lib/terrainMath.ts - MACRO_AMP': 'octave table mirrored by the shader (tests/floatingOrigin.test.ts compares the FREQ tables)',
  'src/lib/terrainMath.ts - RIDGE_AMP': 'octave table mirrored by the shader',
  'src/lib/terrainMath.ts - WARP_AMP': 'octave table mirrored by the shader',
  'src/lib/terrainMath.ts - MarchResult': 'return type of the exported marchRay()',
  'src/components/ui/badge.tsx - badgeVariants': 'shadcn/ui component contract (variant helper exported with the component)',
  'src/components/ui/button.tsx - buttonVariants': 'shadcn/ui component contract (variant helper exported with the component)',
  'src/lib/engine/useTerrainEngine.ts - TerrainEngineOptions': 'parameter type of the exported useTerrainEngine()',
  'src/lib/hero/camera.ts - HeroCamera': 'return type of the exported createHeroCamera()',
  'src/lib/hero/locomotion.ts - HeroMove': 'type of the exported HeroState.move',
  'src/lib/engine/useTerrainEngine.ts - RendererChoice': 'type of the exported TerrainEngineOptions.renderer',
};

const q = await runQuality();
const key = (finding: string) => finding.replace(/:\d+ - /, ' - ').replace(/ \(used in module\)$/, '');
const unjustified = q.deadExports.findings.filter(
  (f) => !/^src\/app\/.*(page|layout|route)\.tsx?:\d+ - (default|metadata|GET)$/.test(f) && !(key(f) in JUSTIFIED),
);

const failures: string[] = [];
if (q.tscErrors > 0) failures.push(`${q.tscErrors} TypeScript strict-mode errors`);
if (q.madgeCircular > 0) failures.push(`${q.madgeCircular} circular dependencies: ${JSON.stringify(q.madgeCycles)}`);
if (!(q.tests.fail === 0 && q.tests.pass > 0)) failures.push(`tests: ${q.tests.pass} pass / ${q.tests.fail} fail${q.tests.failing.length ? `:\n  ${q.tests.failing.join('\n  ')}` : ''}`);
if (unjustified.length > 0) failures.push(`unjustified dead exports:\n  ${unjustified.join('\n  ')}`);

console.log(
  `quality: tsc ${q.tscErrors} errors · madge ${q.madgeCircular} cycles · dead exports ${q.deadExports.findings.length} ` +
    `(${q.deadExports.findings.length - unjustified.length} justified) · tests ${q.tests.pass}/${q.tests.pass + q.tests.fail} ` +
    `(${q.tests.expects} expects) · coverage ${q.coverage.funcs}% funcs / ${q.coverage.lines}% lines`,
);
if (failures.length) {
  console.error(`quality gate FAILED:\n- ${failures.join('\n- ')}`);
  process.exit(1);
}
console.log('quality gate passed');
