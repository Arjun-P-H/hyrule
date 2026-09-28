/**
 * Production-build helpers: build a revision (the working tree, or any git
 * revision in a temporary worktree outside the repo), run its standalone Next
 * server on a free port, and tear it down. Used by the bundle and
 * play-session harnesses so a pinned baseline and the candidate can be
 * measured back to back under the same thermal conditions.
 */

import { $ } from 'bun';
import path from 'node:path';
import { existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';

const ROOT = path.resolve(import.meta.dir, '../..');

export async function buildProduction(root = ROOT): Promise<{ ms: number }> {
  const t0 = performance.now();
  await $`bun run build`.cwd(root).quiet();
  return { ms: performance.now() - t0 };
}

/**
 * Check out `rev` into a temporary worktree and build it. Turbopack rejects a
 * node_modules symlink that leaves the project root, so the worktree gets its
 * own install (Bun clones from its global cache on APFS: fast, no copy).
 */
export async function buildRevision(rev: string): Promise<{ root: string; ms: number }> {
  const short = (await $`git -C ${ROOT} rev-parse --short=7 ${rev}`.text()).trim();
  const root = path.join(tmpdir(), `hyrule-build-${short}`);
  if (existsSync(root)) {
    await $`git -C ${ROOT} worktree remove --force ${root}`.nothrow().quiet();
    rmSync(root, { recursive: true, force: true });
  }
  await $`git -C ${ROOT} worktree add --detach ${root} ${short}`.quiet();
  await $`bun install --frozen-lockfile`.cwd(root).quiet();
  const { ms } = await buildProduction(root);
  return { root, ms };
}

export async function removeRevisionBuild(root: string): Promise<void> {
  await $`git -C ${ROOT} worktree remove --force ${root}`.nothrow().quiet();
}

export async function withProductionServer<T>(fn: (baseUrl: string) => Promise<T>, root = ROOT): Promise<T> {
  const serverJs = path.join(root, '.next/standalone/server.js');
  if (!existsSync(serverJs)) throw new Error(`no standalone build in ${root} — build it first`);
  const port = 3100 + Math.floor(Math.random() * 800);
  const proc = Bun.spawn(['bun', serverJs], {
    cwd: path.join(root, '.next/standalone'),
    env: { ...process.env, PORT: String(port), HOSTNAME: '127.0.0.1', NODE_ENV: 'production' },
    stdout: 'ignore',
    stderr: 'ignore',
  });
  const base = `http://127.0.0.1:${port}`;
  try {
    for (let i = 0; i < 100; i++) {
      try {
        if ((await fetch(base + '/')).ok) break;
      } catch { /* not up yet */ }
      await Bun.sleep(100);
    }
    return await fn(base);
  } finally {
    proc.kill();
    await proc.exited;
  }
}
