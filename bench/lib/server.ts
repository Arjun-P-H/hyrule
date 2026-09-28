/**
 * Production-build helpers: build once, run the standalone Next server on a
 * free port, and tear it down. Used by the bundle and play-session harnesses.
 */

import { $ } from 'bun';
import path from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = path.resolve(import.meta.dir, '../..');

export async function buildProduction(): Promise<{ ms: number }> {
  const t0 = performance.now();
  await $`bun run build`.cwd(ROOT).quiet();
  return { ms: performance.now() - t0 };
}

export async function withProductionServer<T>(fn: (baseUrl: string) => Promise<T>): Promise<T> {
  const serverJs = path.join(ROOT, '.next/standalone/server.js');
  if (!existsSync(serverJs)) throw new Error('no standalone build — run buildProduction() first');
  const port = 3100 + Math.floor(Math.random() * 800);
  const proc = Bun.spawn(['bun', serverJs], {
    cwd: path.join(ROOT, '.next/standalone'),
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
