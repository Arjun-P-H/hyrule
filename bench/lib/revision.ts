/**
 * Load the shader / CPU-twin sources as they exist at any git revision, so a
 * benchmark can A/B the working tree against a pinned baseline in the same
 * process (CPU) or the same browser session (GPU). `WORKTREE` means the
 * current files on disk.
 */

import { $ } from 'bun';
import { mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';

export const WORKTREE = 'WORKTREE';
const ROOT = path.resolve(import.meta.dir, '../..');
const CACHE = path.join(ROOT, 'bench/out/.cache');

export async function resolveRev(rev: string): Promise<string> {
  if (rev === WORKTREE) return WORKTREE;
  return (await $`git -C ${ROOT} rev-parse --short=7 ${rev}`.text()).trim();
}

/**
 * Short hash of HEAD plus a `-dirty` suffix when engine sources (src/) differ
 * from it. Harness-only edits (bench/, tests/) do not change what is measured.
 */
export async function describeWorktree(): Promise<string> {
  const head = (await $`git -C ${ROOT} rev-parse --short=7 HEAD`.text()).trim();
  const dirty = (await $`git -C ${ROOT} status --porcelain -- src`.text()).trim();
  return dirty ? `${head}-dirty` : head;
}

async function materialize(rev: string, files: string[]): Promise<string> {
  const short = await resolveRev(rev);
  const dir = path.join(CACHE, `rev-${short}`);
  for (const f of files) {
    const dest = path.join(dir, f);
    if (existsSync(dest)) continue;
    mkdirSync(path.dirname(dest), { recursive: true });
    const text = await $`git -C ${ROOT} show ${short}:${f}`.text();
    await Bun.write(dest, text);
  }
  return dir;
}

export type ShaderSource = { rev: string; vs: string; fs: string };

export async function loadShader(rev: string): Promise<ShaderSource> {
  if (rev === WORKTREE) {
    const mod = await import(path.join(ROOT, 'src/components/terrain/demoShader.ts'));
    return { rev: await describeWorktree(), vs: mod.TERRAIN_VS, fs: mod.TERRAIN_FS };
  }
  const dir = await materialize(rev, ['src/components/terrain/demoShader.ts']);
  const mod = await import(path.join(dir, 'src/components/terrain/demoShader.ts'));
  return { rev: await resolveRev(rev), vs: mod.TERRAIN_VS, fs: mod.TERRAIN_FS };
}

export type TerrainMathModule = typeof import('../../src/lib/terrainMath');

export async function loadTerrainMath(rev: string): Promise<{ rev: string; mod: TerrainMathModule }> {
  if (rev === WORKTREE) {
    const mod = (await import(path.join(ROOT, 'src/lib/terrainMath.ts'))) as TerrainMathModule;
    return { rev: await describeWorktree(), mod };
  }
  const dir = await materialize(rev, ['src/lib/terrainMath.ts', 'src/lib/constants.ts']);
  const mod = (await import(path.join(dir, 'src/lib/terrainMath.ts'))) as TerrainMathModule;
  return { rev: await resolveRev(rev), mod };
}

export type ConstantsModule = typeof import('../../src/lib/constants');

/** constants.ts as of `rev` (e.g. to render a baseline with the preset budgets it shipped). */
export async function loadConstants(rev: string): Promise<ConstantsModule> {
  if (rev === WORKTREE) return (await import(path.join(ROOT, 'src/lib/constants.ts'))) as ConstantsModule;
  const dir = await materialize(rev, ['src/lib/terrainMath.ts', 'src/lib/constants.ts']);
  return (await import(path.join(dir, 'src/lib/constants.ts'))) as ConstantsModule;
}
