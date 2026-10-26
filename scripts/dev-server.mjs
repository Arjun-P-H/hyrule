/**
 * The dev server the preview runs: server.mjs (the game at /game and the multiplayer relay, one port) with a
 * Turbopack cache that cannot hang it.
 *
 * The cache lives in .next-dev (-2, -3 for a second and third server at once: `next dev` locks its directory),
 * never inside .next, which `next build` empties (that is what once pulled a running server's cache out from under
 * it: "Failed to restore task data", a missing .sst file). It is written while the server runs, so a server killed
 * in mid-write leaves it half-written. A marker file says the last run did not end cleanly: the cache is then
 * cleared first (a cold start, about 20 s) rather than trusted.
 *
 *   node scripts/dev-server.mjs [--port=N]     (PORT too; the port to start looking from, default 3100)
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const portArg = process.argv.find((a) => a.startsWith('--port='))?.slice(7);
const port = portArg || process.env.PORT || '3100';

/** Whether a running process holds the file (`next dev` keeps its lock file open). */
function held(file) {
  if (!existsSync(file)) return false;
  const r = spawnSync('lsof', ['-t', file], { encoding: 'utf8' });
  return r.status === 0 && r.stdout.trim() !== '';
}

const dir = ['.next-dev', '.next-dev-2', '.next-dev-3'].find((d) => !held(join(DIR, d, 'dev', 'lock')));
if (!dir) {
  console.error('dev-server: .next-dev, -2 and -3 are all locked by running servers.');
  process.exit(1);
}
const abs = join(DIR, dir);
const marker = join(abs, '.running');
if (existsSync(marker)) {
  console.error(`dev-server: the last server on ${dir} did not exit cleanly: clearing its cache`);
  rmSync(abs, { recursive: true, force: true });
}
mkdirSync(abs, { recursive: true });
writeFileSync(marker, String(process.pid));

const child = spawn(process.execPath, [join(DIR, 'server.mjs'), `--port=${port}`], {
  cwd: DIR, stdio: 'inherit', env: { ...process.env, NEXT_DIST_DIR: dir },
});
// A stop (the preview's, Ctrl+C) goes on to the server; the marker goes only once it has finished writing its cache.
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => child.kill(sig));
child.on('exit', (code, signal) => {
  rmSync(marker, { force: true });
  process.exit(code ?? (signal ? 0 : 1));
});
