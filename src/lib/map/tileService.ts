/**
 * The map's tiles as the screen uses them: asked for by every frame that draws the map, painted by a pair
 * of workers (tilesWorker.ts), kept as canvases in a least-recently-used cache.
 * ---------------------------------------------------------------------------
 * A frame calls `begin()`, then `get()` for each tile it wants: a cached tile is returned at once, a tile
 * not yet painted is queued (the nearest to the focus first; coarse tiles ahead of fine, they cover more)
 * and `null` is returned. A request not renewed for a few frames is dropped, so a fast pan does not paint
 * what it has left behind. Without Workers (or if one cannot start) the tiles are painted here, one at a
 * time between frames.
 */

import { TILE_PX, renderTile } from './tiles';
import type { TileReply, TileRequest } from './tilesWorker';

type Entry = { key: string; level: number; tx: number; tz: number; seen: number; dist: number };

const CACHE_MAX = 260;
const STALE_FRAMES = 12;
const WORKERS = 2;

const keyOf = (level: number, tx: number, tz: number) => `${level}:${tx}:${tz}`;

export class TileService {
  private seed = Number.NaN;
  private cache = new Map<string, HTMLCanvasElement>();
  private queued = new Map<string, Entry>();
  private flying = new Map<number, { key: string; seed: number }>();
  private workers: Array<{ w: Worker; busy: boolean }> = [];
  private nextId = 1;
  private frame = 0;
  private fx = 0;
  private fz = 0;
  private inline: ReturnType<typeof setTimeout> | null = null;
  private broken = false;
  /** Counts every tile that arrives: a view that stopped changing redraws when this moves. */
  arrived = 0;

  /** The world the tiles belong to: another seed empties the cache. */
  setSeed(seed: number): void {
    if (seed === this.seed) return;
    this.seed = seed;
    this.cache.clear();
    this.queued.clear();
    this.arrived++;
  }

  /** A new frame, looking at (x, z): the tiles asked for from here on are the ones wanted now. */
  begin(x: number, z: number): void {
    this.frame++;
    this.fx = x;
    this.fz = z;
  }

  /** The tile's canvas, or null while it is being painted (it is asked for from now on). */
  get(level: number, tx: number, tz: number, span: number, boost = 0): HTMLCanvasElement | null {
    const key = keyOf(level, tx, tz);
    const hit = this.cache.get(key);
    if (hit) {
      this.cache.delete(key);
      this.cache.set(key, hit);
      return hit;
    }
    const cx = (tx + 0.5) * span, cz = (tz + 0.5) * span;
    const dist = Math.hypot(cx - this.fx, cz - this.fz) / span - boost;
    const e = this.queued.get(key);
    if (e) { e.seen = this.frame; e.dist = dist; }
    else this.queued.set(key, { key, level, tx, tz, seen: this.frame, dist });
    this.pump();
    return null;
  }

  /** The tile's canvas if painted (without asking for it, and without touching the cache's order). */
  peek(level: number, tx: number, tz: number): HTMLCanvasElement | null {
    return this.cache.get(keyOf(level, tx, tz)) ?? null;
  }

  dispose(): void {
    for (const w of this.workers) w.w.terminate();
    this.workers = [];
    if (this.inline !== null) clearTimeout(this.inline);
    this.inline = null;
    this.queued.clear();
    this.flying.clear();
    this.cache.clear();
  }

  private take(): Entry | null {
    let best: Entry | null = null;
    for (const e of this.queued.values()) {
      if (this.frame - e.seen > STALE_FRAMES) { this.queued.delete(e.key); continue; }
      if (!best || e.dist < best.dist) best = e;
    }
    if (best) this.queued.delete(best.key);
    return best;
  }

  private ensureWorkers(): void {
    if (this.workers.length > 0 || this.broken || typeof Worker === 'undefined') return;
    try {
      for (let k = 0; k < WORKERS; k++) {
        const w = new Worker(new URL('./tilesWorker.ts', import.meta.url), { type: 'module' });
        const slot = { w, busy: false };
        w.onmessage = (e: MessageEvent<TileReply>) => {
          slot.busy = false;
          this.arrive(e.data.id, new Uint8ClampedArray(e.data.px));
          this.pump();
        };
        w.onerror = () => {
          slot.busy = false;
          this.broken = true;
          for (const s of this.workers) s.w.terminate();
          this.workers = [];
          this.flying.clear();
          this.pump();
        };
        this.workers.push(slot);
      }
    } catch {
      this.broken = true;
      for (const s of this.workers) s.w.terminate();
      this.workers = [];
    }
  }

  private pump(): void {
    this.ensureWorkers();
    if (this.workers.length === 0) {
      // (no workers: one tile at a time, between frames)
      if (this.inline === null && this.queued.size > 0) {
        this.inline = setTimeout(() => {
          this.inline = null;
          const e = this.take();
          if (!e) return;
          const px = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
          renderTile(this.seed, e.level, e.tx, e.tz, px);
          this.put(e.key, this.seed, px);
          this.pump();
        }, 30);
      }
      return;
    }
    for (const slot of this.workers) {
      if (slot.busy) continue;
      const e = this.take();
      if (!e) return;
      const id = this.nextId++;
      this.flying.set(id, { key: e.key, seed: this.seed });
      slot.busy = true;
      const req: TileRequest = { id, seed: this.seed, level: e.level, tx: e.tx, tz: e.tz };
      slot.w.postMessage(req);
    }
  }

  private arrive(id: number, px: Uint8ClampedArray): void {
    const f = this.flying.get(id);
    this.flying.delete(id);
    if (f) this.put(f.key, f.seed, px);
  }

  private put(key: string, seed: number, px: Uint8ClampedArray): void {
    if (seed !== this.seed) return; // (painted for a world that has since changed)
    const c = document.createElement('canvas');
    c.width = c.height = TILE_PX;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.putImageData(new ImageData(px as Uint8ClampedArray<ArrayBuffer>, TILE_PX, TILE_PX), 0, 0);
    this.cache.set(key, c);
    this.arrived++;
    while (this.cache.size > CACHE_MAX) {
      const oldest = this.cache.keys().next().value;
      if (oldest === undefined) break;
      this.cache.delete(oldest);
    }
  }
}

let shared: TileService | null = null;

/** The one tile service of the page (the minimap and the full map share its tiles). */
export function tileService(): TileService {
  shared ??= new TileService();
  return shared;
}
