/**
 * The browser's side of multiplayer: a WebSocket to the game server's relay
 * (server.mjs, path /mp) — who is in the world and which character each
 * plays (for the lobby), entering the world as a character, and the stream
 * of hero snapshots, the blows and shots players trade (combat/pvp.ts), and
 * clock / seed changes the engine and the page trade through it (link.ts).
 * Reconnects on its own and enters the world again.
 * Without the server (plain `next dev`) it stays offline: one player.
 */

import { useSyncExternalStore } from 'react';
import type { ClockState, NetLink } from './link';

type PlayerInfo = { id: string; name: string; char: number };
type NetStatus = 'connecting' | 'online' | 'offline';
/** Why entering the world was refused: the character is someone else's, five are in already, or a bad request. */
type DenyReason = 'taken' | 'full' | 'bad';

export type NetView = {
  status: NetStatus;
  selfId: string | null;
  inWorld: boolean;
  /** The players in the world, in the order they joined. */
  players: readonly PlayerInfo[];
  max: number;
  denied: DenyReason | null;
  /** A link friends far away can open (the server's tunnel), when there is one. */
  share: string | null;
  /** The latest arrival, departure or world change, for a moment on screen. */
  notice: { text: string; at: number } | null;
};

const INBOX_MAX = 512;     // snapshots held for the engine (it drains them every frame; a hidden tab doesn't)
const SEND_BACKLOG = 65536; // bytes queued on the socket before snapshots are dropped
const CONNECT_TIMEOUT = 6000; // ms: plain `next dev` never answers the relay's path, it leaves the socket hanging

function parseClock(c: unknown): ClockState | null {
  if (!Array.isArray(c) || c.length !== 3 || !c.every((v) => typeof v === 'number' && Number.isFinite(v))) return null;
  return { days: c[0], paused: c[1] !== 0, dayMinutes: c[2] };
}

const clockMsg = (c: ClockState) => [c.days, c.paused ? 1 : 0, c.dayMinutes];

export class NetSession implements NetLink {
  private ws: WebSocket | null = null;
  private view: NetView = { status: 'connecting', selfId: null, inWorld: false, players: [], max: 5, denied: null, share: null, notice: null };
  private listeners = new Set<() => void>();
  private inbox: Array<{ id: string; at: number; d: number[] }> = [];
  private combatInbox: Array<{ id: string; at: number; d: number[] }> = [];
  private clockIn: ClockState | null = null;
  private peerChars = new Map<string, number>();
  private wish: { name: string; char: number } | null = null;
  private seed = 0;
  private clockOut: ClockState | null = null;
  private attempts = 0;
  private everOnline = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = true;
  private seedListener: ((seed: number) => void) | null = null;

  /** Hear the world's seed: on entering the world, and when another player changes it. */
  listenSeed(fn: (seed: number) => void): () => void {
    this.seedListener = fn;
    return () => { if (this.seedListener === fn) this.seedListener = null; };
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close();
  }

  // React (useSyncExternalStore)
  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => { this.listeners.delete(fn); };
  };
  getView = (): NetView => this.view;

  private set(patch: Partial<NetView>): void {
    this.view = { ...this.view, ...patch };
    for (const fn of this.listeners) fn();
  }

  private connect(): void {
    const url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/mp`;
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      this.retry();
      return;
    }
    this.ws = ws;
    const timeout = setTimeout(() => { if (ws.readyState === WebSocket.CONNECTING) ws.close(); }, CONNECT_TIMEOUT);
    ws.onopen = () => { clearTimeout(timeout); this.attempts = 0; this.everOnline = true; };
    ws.onmessage = (e) => this.onMessage(e.data);
    ws.onclose = () => {
      clearTimeout(timeout);
      if (this.ws !== ws) return;
      this.ws = null;
      this.peerChars = new Map();
      this.inbox.length = 0;
      this.combatInbox.length = 0;
      // Lost after being online: reconnecting (the world is entered again); never reached: offline.
      this.set({ status: this.everOnline ? 'connecting' : 'offline', selfId: null, inWorld: false, players: [] });
      this.retry();
    };
  }

  private retry(): void {
    // (never reached: no relay here — plain `next dev` — so a few tries, then play alone)
    if (this.stopped || (!this.everOnline && this.attempts >= 3)) return;
    const delay = Math.min(10000, 1000 * 2 ** this.attempts++);
    this.timer = setTimeout(() => { this.timer = null; if (!this.stopped) this.connect(); }, delay);
  }

  private send(msg: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  private onMessage(raw: unknown): void {
    if (typeof raw !== 'string') return;
    let m: Record<string, unknown>;
    try { m = JSON.parse(raw); } catch { return; }
    switch (m?.t) {
      case 'hello':
        this.set({
          status: 'online', selfId: String(m.id), players: asPlayers(m.players),
          max: typeof m.max === 'number' ? m.max : 5, share: typeof m.share === 'string' ? m.share : null,
        });
        this.rosterChanged();
        this.sendJoin(); // back into the world after a reconnect
        break;
      case 'share':
        this.set({ share: typeof m.url === 'string' ? m.url : null });
        break;
      case 'players': {
        const before = this.view.players;
        const after = asPlayers(m.players);
        const came = after.find((p) => p.id !== this.view.selfId && !before.some((b) => b.id === p.id));
        const went = before.find((b) => !after.some((p) => p.id === b.id));
        this.set({
          players: after,
          ...(this.view.inWorld && (came || went) ? { notice: { text: came ? `${came.name} joined the world` : `${went!.name} left`, at: Date.now() } } : {}),
        });
        this.rosterChanged();
        break;
      }
      case 'joined': {
        this.set({ inWorld: true, denied: null });
        if (typeof m.seed === 'number') this.seedListener?.(m.seed);
        const c = parseClock(m.clock);
        if (c) this.clockIn = c;
        break;
      }
      case 'denied':
        this.wish = null;
        this.set({ inWorld: false, denied: m.reason === 'taken' || m.reason === 'full' ? m.reason : 'bad' });
        break;
      case 's':
        if (this.view.inWorld && typeof m.id === 'string' && Array.isArray(m.d)) {
          if (this.inbox.length >= INBOX_MAX) this.inbox.splice(0, INBOX_MAX / 2);
          this.inbox.push({ id: m.id, at: performance.now(), d: m.d as number[] });
        }
        break;
      case 'c': // a blow, a shot, or what came of one (combat/pvp.ts)
        if (this.view.inWorld && typeof m.id === 'string' && Array.isArray(m.d)) {
          if (this.combatInbox.length >= INBOX_MAX) this.combatInbox.splice(0, INBOX_MAX / 2);
          this.combatInbox.push({ id: m.id, at: performance.now(), d: m.d as number[] });
        }
        break;
      case 'seed':
        if (this.view.inWorld && typeof m.seed === 'number') {
          this.seed = m.seed;
          const by = typeof m.by === 'string' ? m.by : null;
          this.seedListener?.(m.seed);
          this.set({ notice: { text: `${by ?? 'Someone'} changed the world (seed ${m.seed})`, at: Date.now() } });
        }
        break;
      case 'clock': {
        const c = parseClock(m.c);
        if (c && this.view.inWorld) this.clockIn = c;
        break;
      }
    }
  }

  private rosterChanged(): void {
    const self = this.view.selfId;
    this.peerChars = new Map(this.view.players.filter((p) => p.id !== self).map((p) => [p.id, p.char]));
  }

  private sendJoin(): void {
    const w = this.wish;
    if (!w) return;
    this.send({ t: 'join', name: w.name, char: w.char, seed: this.seed, clock: this.clockOut ? clockMsg(this.clockOut) : null });
  }

  /**
   * Enter the world as character `char` (and again after any reconnect, until refused). An empty
   * world takes this player's seed and clock; otherwise the world's come back (onSeed, takeClock).
   */
  join(name: string, char: number, seed: number, clock: ClockState | null): void {
    this.wish = { name, char };
    this.seed = seed;
    if (clock) this.clockOut = clock;
    this.set({ denied: null });
    this.sendJoin();
  }

  /** This player changed the world's seed: everyone's world changes with it. */
  sendSeed(seed: number): void {
    this.seed = seed;
    if (this.view.inWorld) this.send({ t: 'seed', seed });
  }

  // ---- NetLink (the engine's frame loop) ----
  inWorld(): boolean {
    return this.view.inWorld;
  }
  peers(): ReadonlyMap<string, number> {
    return this.peerChars;
  }
  drain(take: (id: string, at: number, snap: readonly number[]) => void): void {
    for (const m of this.inbox) take(m.id, m.at, m.d);
    this.inbox.length = 0;
  }
  sendState(snap: readonly number[]): void {
    const ws = this.ws;
    if (!this.view.inWorld || !ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > SEND_BACKLOG) return;
    ws.send(JSON.stringify({ t: 's', d: snap }));
  }
  drainCombat(take: (id: string, at: number, msg: readonly number[]) => void): void {
    for (const m of this.combatInbox) take(m.id, m.at, m.d);
    this.combatInbox.length = 0;
  }
  sendCombat(msg: readonly number[], to: string | null): void {
    // (not dropped for a backlog as snapshots are: the next snapshot replaces a lost one, a lost blow is gone)
    if (this.view.inWorld) this.send(to === null ? { t: 'c', d: msg } : { t: 'c', to, d: msg });
  }
  peerName(id: string): string {
    return this.view.players.find((p) => p.id === id)?.name ?? 'Someone';
  }
  takeClock(): ClockState | null {
    const c = this.clockIn;
    this.clockIn = null;
    return c;
  }
  sendClock(c: ClockState): void {
    this.clockOut = c;
    if (this.view.inWorld) this.send({ t: 'clock', c: clockMsg(c) });
  }
  keepsClock(): boolean {
    return this.view.inWorld && this.view.players[0]?.id === this.view.selfId;
  }
}

function asPlayers(v: unknown): PlayerInfo[] {
  if (!Array.isArray(v)) return [];
  return v.filter((p): p is PlayerInfo =>
    !!p && typeof p.id === 'string' && typeof p.name === 'string' && Number.isInteger(p.char));
}

export function useNetView(s: NetSession): NetView {
  return useSyncExternalStore(s.subscribe, s.getView, s.getView);
}
