/**
 * Code-quality axis: TypeScript strict errors, circular dependencies (madge),
 * unused exports (ts-prune), test count / pass rate, and line/function
 * coverage (bun test --coverage). Tools run via bunx so CI needs no extra
 * setup beyond network access to the npm registry.
 */

import { $ } from 'bun';
import path from 'node:path';

const ROOT = path.resolve(import.meta.dir, '..');

export type QualityReport = {
  tscErrors: number;
  madgeCircular: number;
  madgeCycles: string[][];
  /** src/ only. unused = imported nowhere; internalOnly = exported but used only in its own module. */
  deadExports: { unused: number; unusedJustified: number; internalOnly: number; findings: string[] };
  tests: { pass: number; fail: number; expects: number; files: number };
  coverage: { funcs: number; lines: number; perFile: Record<string, { funcs: number; lines: number }> };
};

// Exports consumed by a framework or tool rather than by our own imports.
const FRAMEWORK: RegExp[] = [
  /^src\/app\/.*\/?(page|layout|route|not-found|error|loading)\.tsx?:\d+ - (default|metadata|GET|POST|dynamic|revalidate)$/,
  /^next\.config\.ts:\d+ - default$/,
];

export async function runQuality(): Promise<QualityReport> {
  const tsc = await $`bunx tsc --noEmit -p tsconfig.json`.cwd(ROOT).nothrow().text();
  const tscErrors = (tsc.match(/error TS\d+/g) ?? []).length;

  const madgeOut = await $`bunx madge --circular --extensions ts,tsx --ts-config tsconfig.json --json src`.cwd(ROOT).nothrow().text();
  let madgeCycles: string[][] = [];
  try { madgeCycles = JSON.parse(madgeOut.slice(madgeOut.indexOf('['))); } catch { madgeCycles = [['<madge parse error>']]; }

  const prune = await $`bunx ts-prune -p tsconfig.json`.cwd(ROOT).nothrow().text();
  const findings = prune.split('\n').map((l) => l.trim()).filter((l) => l.startsWith('src/'));
  const internalOnly = findings.filter((l) => l.endsWith('(used in module)'));
  const unused = findings.filter((l) => !l.endsWith('(used in module)'));
  const unusedJustified = unused.filter((l) => FRAMEWORK.some((r) => r.test(l)));

  // Pass/fail from a plain run: coverage instrumentation slows code enough to
  // trip the wall-clock threshold tests, so the instrumented run only supplies
  // the coverage table.
  const test = await $`bun test`.cwd(ROOT).nothrow();
  const text = test.stdout.toString() + test.stderr.toString();
  const num = (re: RegExp) => Number(text.match(re)?.[1] ?? 'NaN');
  const cov = await $`bun test --coverage`.cwd(ROOT).nothrow();
  const covText = cov.stdout.toString() + cov.stderr.toString();
  const perFile: QualityReport['coverage']['perFile'] = {};
  let funcs = NaN, lines = NaN;
  for (const row of covText.split('\n')) {
    const m = row.match(/^\s*(\S.*?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|/);
    if (!m) continue;
    if (m[1] === 'All files') { funcs = Number(m[2]); lines = Number(m[3]); }
    else perFile[m[1]] = { funcs: Number(m[2]), lines: Number(m[3]) };
  }

  return {
    tscErrors,
    madgeCircular: madgeCycles.length,
    madgeCycles,
    deadExports: { unused: unused.length, unusedJustified: unusedJustified.length, internalOnly: internalOnly.length, findings },
    tests: {
      pass: num(/(\d+) pass/),
      fail: num(/(\d+) fail/),
      expects: num(/(\d+) expect\(\) calls/),
      files: num(/across (\d+) files?/),
    },
    coverage: { funcs, lines, perFile },
  };
}
