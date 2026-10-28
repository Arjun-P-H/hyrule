/**
 * Chrome launcher (headless, or a visible window with `headed`) + a minimal
 * Chrome DevTools Protocol client.
 * The GPU path runs through ANGLE/Metal on macOS (the same backend the game
 * uses in desktop Chrome); `software: true` switches to SwiftShader, the
 * CPU rasterizer that stands in for the "no usable GPU" hardware floor.
 */

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

export const CHROME_PATH =
  process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

/** More switches for the platform (CHROME_ARGS, space-separated): on Linux, say, `--no-sandbox --use-angle=swiftshader`. */
const ENV_ARGS = (process.env.CHROME_ARGS ?? '').split(' ').filter(Boolean);

export type ChromeOptions = {
  url: string;
  software?: boolean;
  width?: number;
  height?: number;
  remoteDebugging?: boolean;
  /** A visible window on the real display (vsync, compositor) instead of headless. */
  headed?: boolean;
  extraArgs?: string[];
};

export type ChromeHandle = {
  proc: ReturnType<typeof Bun.spawn>;
  wsEndpoint: string | null;
  close: () => Promise<void>;
};

export async function launchChrome(opts: ChromeOptions): Promise<ChromeHandle> {
  const profile = mkdtempSync(path.join(tmpdir(), 'hyrule-bench-'));
  const args = [
    ...(opts.headed ? [] : ['--headless=new']),
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    '--ignore-gpu-blocklist',
    '--enable-webgl',
    '--use-gl=angle',
    opts.software ? '--use-angle=swiftshader' : '--use-angle=metal',
    ...(opts.software ? ['--enable-unsafe-swiftshader'] : []),
    `--window-size=${opts.width ?? 1280},${opts.height ?? 720}`,
    ...(opts.remoteDebugging ? ['--remote-debugging-port=0'] : []),
    ...(opts.extraArgs ?? []),
    ...ENV_ARGS,
    opts.url,
  ];
  const proc = Bun.spawn([CHROME_PATH, ...args], { stdout: 'ignore', stderr: 'pipe' });

  let wsEndpoint: string | null = null;
  if (opts.remoteDebugging) {
    const reader = proc.stderr.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    const deadline = Date.now() + 20_000;
    while (!wsEndpoint && Date.now() < deadline) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value);
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) wsEndpoint = m[1];
    }
    reader.releaseLock();
    if (!wsEndpoint) throw new Error('Chrome did not report a DevTools endpoint');
  }

  return {
    proc,
    wsEndpoint,
    close: async () => {
      proc.kill();
      await proc.exited;
      try { rmSync(profile, { recursive: true, force: true }); } catch { /* profile busy */ }
    },
  };
}

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

/** One WebSocket session to a single CDP target (browser or page). */
export class CDPSession {
  private ws: WebSocket;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Map<string, Array<(params: unknown) => void>>();

  private constructor(ws: WebSocket) {
    this.ws = ws;
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(String(ev.data));
      if (typeof msg.id === 'number') {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        if (msg.error) p.reject(new Error(`${msg.error.message} (${msg.error.code})`));
        else p.resolve(msg.result);
      } else if (msg.method) {
        for (const fn of this.listeners.get(msg.method) ?? []) fn(msg.params);
      }
    });
  }

  static connect(url: string): Promise<CDPSession> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.addEventListener('open', () => resolve(new CDPSession(ws)));
      ws.addEventListener('error', () => reject(new Error(`CDP connect failed: ${url}`)));
    });
  }

  send<T = Record<string, unknown>>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    });
  }

  on(method: string, fn: (params: unknown) => void): void {
    const list = this.listeners.get(method) ?? [];
    list.push(fn);
    this.listeners.set(method, list);
  }

  close(): void {
    this.ws.close();
  }
}

/** Connect to the first page target of a Chrome launched with remoteDebugging. */
export async function connectToPage(wsEndpoint: string): Promise<CDPSession> {
  const httpBase = wsEndpoint.replace(/^ws:\/\//, 'http://').replace(/\/devtools\/browser\/.*$/, '');
  for (let i = 0; i < 50; i++) {
    const targets = (await (await fetch(`${httpBase}/json/list`)).json()) as Array<{
      type: string;
      webSocketDebuggerUrl?: string;
    }>;
    const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
    if (page?.webSocketDebuggerUrl) return CDPSession.connect(page.webSocketDebuggerUrl);
    await Bun.sleep(100);
  }
  throw new Error('no CDP page target found');
}
