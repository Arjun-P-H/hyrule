/**
 * The game server for multiplayer: the Next.js app (the game at /game) and the
 * multiplayer relay (a WebSocket at /mp) on one port, so one link — on the
 * LAN, or through a tunnel for friends far away — serves both. Plain Node.js
 * (18+), the same on macOS, Windows and Linux:
 *
 *   node server.mjs                          development (hot reload)
 *   node server.mjs --prod                   production (builds first when the build is stale)
 *   node server.mjs --prod --tunnel --open   + a public https link (cloudflared), + opens the browser
 *   (--port=N or PORT=N: the port to start looking from; default 3000)
 *
 * The relay keeps one world: up to five players, one per character
 * (src/lib/hero/characters.ts). Each browser simulates its own hero and sends
 * its state ~15 times a second; the relay passes it to the others, and keeps
 * the world's seed and clock (the first player's) for those who join later.
 * Players fight each other too: their blows and shots, and what came of them,
 * are passed on as they are (src/lib/combat/pvp.ts) — each browser settles
 * what strikes its own hero.
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { networkInterfaces } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

const DIR = dirname(fileURLToPath(import.meta.url));
const args = new Set(process.argv.slice(2));
const portArg = Number(process.argv.find((a) => a.startsWith('--port='))?.slice(7));
const prod = args.has('--prod') || process.env.NODE_ENV === 'production';
// (set before Next loads, and here rather than on the command line, which Windows' shell can't do)
process.env.NODE_ENV = prod ? 'production' : 'development';

const MAX_PLAYERS = 5;   // one per character
const CHARACTERS = 5;
const MAX_SNAPSHOT = 48; // numbers in one state message (remoteHeroes.ts S.length, with room to grow)
const MAX_COMBAT = 32;   // … in one combat message (a blow, a shot, what came of one, a colossus's state)
const MP_PATH = '/mp';

// ---- production build ------------------------------------------------------

/** Newest modification time under p (skipping what the build itself writes). */
function newest(p) {
  if (!existsSync(p)) return 0;
  const st = statSync(p);
  if (!st.isDirectory()) return st.mtimeMs;
  let t = 0;
  for (const name of readdirSync(p)) {
    if (name === 'node_modules' || name === '.next') continue;
    t = Math.max(t, newest(join(p, name)));
  }
  return t;
}

function buildIfStale() {
  const id = join(DIR, '.next', 'BUILD_ID');
  const built = existsSync(id) ? statSync(id).mtimeMs : 0;
  const sources = Math.max(...['src', 'public', 'next.config.ts', 'package.json', 'postcss.config.mjs', 'tsconfig.json']
    .map((p) => newest(join(DIR, p))));
  if (built > sources) return;
  console.log(built ? '[build] Sources changed since the last build: building the game…' : '[build] Building the game (first run, a minute or two)…');
  const r = spawnSync(process.execPath, [join(DIR, 'node_modules', 'next', 'dist', 'bin', 'next'), 'build'], { cwd: DIR, stdio: 'inherit' });
  if (r.status !== 0) {
    console.error('[build] The build failed (see above).');
    process.exit(1);
  }
}

// ---- the world -----------------------------------------------------------

/** @typedef {{ id: string, name: string, char: number, joined: boolean, seq: number, alive: boolean }} Client */
/** @type {Map<import('ws').WebSocket, Client>} */
const clients = new Map();
const world = { seed: /** @type {number | null} */ (null), clock: /** @type {number[] | null} */ (null) };
let nextId = 1;
let nextSeq = 1;
/** The public link (the tunnel's), once known: the lobby offers it as the invitation. */
let share = null;

const joined = () => [...clients.values()].filter((c) => c.joined).sort((a, b) => a.seq - b.seq);
/** The players in the world, in the order they joined (the first keeps the clock). */
const roster = () => joined().map(({ id, name, char }) => ({ id, name, char }));

function send(ws, msg) {
  if (ws.readyState === ws.OPEN) ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
}
/** To every connection (lobbies too), or with `inWorld` only to players in the world; never to `except`. */
function broadcast(msg, { except = null, inWorld = false } = {}) {
  const text = JSON.stringify(msg);
  for (const [ws, c] of clients) {
    if (ws !== except && (!inWorld || c.joined)) send(ws, text);
  }
}

const finiteList = (d, max) => Array.isArray(d) && d.length > 0 && d.length <= max && d.every((v) => typeof v === 'number' && Number.isFinite(v));
const validSeed = (s) => Number.isInteger(s) && s >= 0 && s < 4294967296;
// [days, paused 0/1, real minutes per game day]
const validClock = (c) => finiteList(c, 3) && c.length === 3 && c[0] >= 0 && c[2] > 0;

function cleanName(v, id) {
  const name = String(v ?? '').replace(/[\u0000-\u001f\u007f<>]/g, '').trim().slice(0, 16);
  return name || `Player ${id}`;
}

function onJoin(ws, c, m) {
  const char = m.char;
  if (!Number.isInteger(char) || char < 0 || char >= CHARACTERS) return send(ws, { t: 'denied', reason: 'bad' });
  const others = joined().filter((o) => o !== c);
  if (others.some((o) => o.char === char)) return send(ws, { t: 'denied', reason: 'taken' });
  if (!c.joined && others.length >= MAX_PLAYERS) return send(ws, { t: 'denied', reason: 'full' });
  // The first player into an empty world brings its seed and clock.
  if (others.length === 0) world.clock = validClock(m.clock) ? m.clock : null;
  if ((others.length === 0 || world.seed === null) && validSeed(m.seed)) world.seed = m.seed;
  c.name = cleanName(m.name, c.id);
  c.char = char;
  if (!c.joined) c.seq = nextSeq++;
  c.joined = true;
  send(ws, { t: 'joined', id: c.id, seed: world.seed, clock: world.clock });
  broadcast({ t: 'players', players: roster() });
  console.log(`[mp] ${c.name} is in the world (${joined().length}/${MAX_PLAYERS})`);
}

function onMessage(ws, c, raw, isBinary) {
  if (isBinary) return;
  let m;
  try { m = JSON.parse(raw.toString()); } catch { return; }
  if (!m || typeof m !== 'object' || typeof m.t !== 'string') return;
  switch (m.t) {
    case 'join':
      onJoin(ws, c, m);
      break;
    case 's': // hero state: passed on as it is
      if (c.joined && finiteList(m.d, MAX_SNAPSHOT)) broadcast({ t: 's', id: c.id, d: m.d }, { except: ws, inWorld: true });
      break;
    case 'c': // combat: to one player (`to`), or to everyone else in the world
      if (!c.joined || !finiteList(m.d, MAX_COMBAT)) break;
      if (typeof m.to === 'string') {
        for (const [other, o] of clients) {
          if (o.id === m.to && o.joined && other !== ws) send(other, { t: 'c', id: c.id, d: m.d });
        }
      } else broadcast({ t: 'c', id: c.id, d: m.d }, { except: ws, inWorld: true });
      break;
    case 'seed':
      if (!c.joined || !validSeed(m.seed) || m.seed === world.seed) break;
      world.seed = m.seed;
      broadcast({ t: 'seed', seed: m.seed, by: c.name }, { except: ws });
      console.log(`[mp] ${c.name} changed the world's seed to ${m.seed}`);
      break;
    case 'clock':
      if (!c.joined || !validClock(m.c)) break;
      world.clock = m.c;
      broadcast({ t: 'clock', c: m.c }, { except: ws, inWorld: true });
      break;
  }
}

// ---- start -----------------------------------------------------------------

function lanAddresses() {
  const out = [];
  for (const list of Object.values(networkInterfaces())) {
    for (const a of list ?? []) if (a.family === 'IPv4' && !a.internal) out.push(a.address);
  }
  return out;
}

function openBrowser(url) {
  const [cmd, cmdArgs] = process.platform === 'darwin' ? ['open', [url]]
    : process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]] // ('': start's window title)
      : ['xdg-open', [url]];
  spawn(cmd, cmdArgs, { stdio: 'ignore', detached: true }).on('error', () => {}).unref();
}

/** A public https link to this port (Cloudflare quick tunnel: free, no account), when cloudflared is installed. */
function startTunnel(port) {
  const proc = spawn('cloudflared', ['tunnel', '--no-autoupdate', '--url', `http://localhost:${port}`], { stdio: ['ignore', 'pipe', 'pipe'] });
  let shown = false;
  const scan = (buf) => {
    const url = !shown && /https:\/\/[a-z0-9-]+\.trycloudflare\.com/.exec(buf.toString())?.[0];
    if (!url) return;
    shown = true;
    share = `${url}/game`;
    broadcast({ t: 'share', url: share });
    console.log('\n  ┌──────────────────────────────────────────────────────────────────────');
    console.log('  │  Friends anywhere (Windows, macOS, Linux — Chrome or Edge): send them');
    console.log(`  │    ${url}/game`);
    console.log('  └──────────────────────────────────────────────────────────────────────\n');
  };
  proc.stdout.on('data', scan);
  proc.stderr.on('data', scan);
  proc.on('error', () => {
    console.log('\n[tunnel] To play with friends outside your network, install cloudflared (free, no account) and start again:');
    console.log('           macOS:   brew install cloudflared');
    console.log('           Windows: winget install --id Cloudflare.cloudflared');
    console.log('         (any https tunnel to this port works too: the game needs https for WebGPU)\n');
  });
  // (the tunnel goes with the server, however it is stopped)
  process.on('exit', () => proc.kill());
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => process.exit(0));
  return proc;
}

/** Listen on the first free port from `port` on. */
function listen(server, port, tries = 10) {
  return new Promise((resolve, reject) => {
    const onError = (e) => {
      server.off('listening', onListening);
      if (e.code === 'EADDRINUSE' && tries > 1) {
        console.log(`[server] Port ${port} is busy, trying ${port + 1}…`);
        resolve(listen(server, port + 1, tries - 1));
      } else reject(e);
    };
    const onListening = () => {
      server.off('error', onError);
      resolve(port);
    };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port);
  });
}

if (prod) buildIfStale();

// The port first (the first free one from PORT, default 3000), then Next.js on it.
let handle = null;
const server = createServer((req, res) => {
  // (plain GET on the relay's path: tells the launchers this server is up already)
  if (req.url === MP_PATH) {
    res.statusCode = 426;
    res.setHeader('Upgrade', 'websocket');
    return res.end('W(x,y) multiplayer relay');
  }
  if (handle) return handle(req, res);
  res.statusCode = 503;
  res.setHeader('Retry-After', '2');
  res.end('The game server is starting…');
});
const port = await listen(server, portArg || Number(process.env.PORT) || 3000);
const { default: next } = await import('next');
const app = next({ dev: !prod, dir: DIR, hostname: 'localhost', port, httpServer: server });
await app.prepare();
handle = app.getRequestHandler();

const wss = new WebSocketServer({ noServer: true, maxPayload: 4096 });

// Our path only; Next.js attaches its own listener for its hot reload and leaves paths it has no page for alone.
server.on('upgrade', (req, socket, head) => {
  if (new URL(req.url ?? '/', 'http://x').pathname !== MP_PATH) return;
  wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
});

wss.on('connection', (ws) => {
  const c = { id: String(nextId++), name: '', char: -1, joined: false, seq: 0, alive: true };
  clients.set(ws, c);
  send(ws, { t: 'hello', id: c.id, players: roster(), seed: world.seed, max: MAX_PLAYERS, share });
  ws.on('pong', () => { c.alive = true; });
  ws.on('message', (raw, isBinary) => onMessage(ws, c, raw, isBinary));
  ws.on('close', () => {
    clients.delete(ws);
    if (!c.joined) return;
    broadcast({ t: 'players', players: roster() });
    console.log(`[mp] ${c.name} left (${joined().length}/${MAX_PLAYERS})`);
    // An empty world forgets its seed and clock: the next first player brings theirs.
    if (joined().length === 0) { world.seed = null; world.clock = null; }
  });
});

// Drop connections that stopped answering (a closed laptop, a lost network).
const heartbeat = setInterval(() => {
  for (const [ws, c] of clients) {
    if (!c.alive) { ws.terminate(); continue; }
    c.alive = false;
    ws.ping();
  }
}, 15000);
wss.on('close', () => clearInterval(heartbeat));

const local = `http://localhost:${port}/game`;
console.log(`\n  W(x,y) Hyrule — multiplayer ${prod ? '' : '(development) '}server running`);
console.log(`    This computer:  ${local}`);
for (const ip of lanAddresses()) console.log(`    Same network:   http://${ip}:${port}/game   (WebGPU needs https or localhost: use the tunnel link)`);
console.log('');
if (args.has('--tunnel')) startTunnel(port);
if (args.has('--open')) openBrowser(local);
