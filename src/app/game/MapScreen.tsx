'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Crosshair, Minus, Plus, X } from 'lucide-react';
import { characterOf } from '@/lib/hero/characters';
import { NO_CLOUD_MAP, type MapHudState } from '@/lib/map/mapHud';
import { tileService } from '@/lib/map/tileService';
import { keyHint, type Bindings } from '@/lib/engine/keybinds';
import { MAP_GROUND, drawArrow, drawEdgePointer, drawPeer, drawTiles } from './mapDraw';

/**
 * The full map (the map key): the lands under north-up, the hero's arrow, and every other player in the
 * world with their name — where they are, or a pointer to the edge towards them. Drag to pan, scroll to
 * zoom, double-click (or C) to come back to the hero; click a player in the list to follow them. The
 * world carries on behind it (alone in it, it stands still, as for the inventory). Drawn in its own
 * animation frames from the engine's channel (mapHud.ts).
 */

const MIN_MPP = 1;
const MAX_MPP = 600;
const START_MPP = 10;
const NICE = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];

/** The first of the round distances at least `m` metres. */
const niceAtLeast = (m: number) => NICE.find((n) => n >= m) ?? NICE[NICE.length - 1];
const kmText = (m: number) => (m >= 1000 ? `${m / 1000} km` : `${m} m`);

type Row = { id: string; name: string; colour: string; dist: number; self: boolean };

type Follow = { kind: 'self' } | { kind: 'peer'; id: string } | null;

export default function MapScreen({
  mapHud, selfName, selfChar, onClose, bindings,
}: {
  mapHud: MapHudState;
  selfName: string;
  selfChar: number;
  onClose: () => void;
  bindings: Bindings;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLSpanElement>(null);
  // the view: its centre (m; z is north), metres to the pixel, and who the centre follows
  const view = useRef<{ cx: number; cz: number; mpp: number; follow: Follow }>({ cx: mapHud.x, cz: mapHud.z, mpp: START_MPP, follow: { kind: 'self' } });
  const [rows, setRows] = useState<Row[]>([]);
  const [following, setFollowing] = useState<string>('self');
  const [land, setLand] = useState(mapHud.land);
  const selfColour = characterOf(selfChar).swatch;

  const setFollow = useCallback((f: Follow) => {
    view.current.follow = f;
    setFollowing(f === null ? '' : f.kind === 'self' ? 'self' : f.id);
  }, []);
  const zoomBy = useCallback((factor: number, sx?: number, sy?: number) => {
    const v = view.current, c = canvasRef.current;
    if (!c) return;
    const w = c.clientWidth, h = c.clientHeight;
    const next = Math.min(MAX_MPP, Math.max(MIN_MPP, v.mpp * factor));
    if (sx !== undefined && sy !== undefined && v.follow === null) {
      // (the world point under the cursor stays under it)
      v.cx += (sx - w / 2) * (v.mpp - next);
      v.cz -= (sy - h / 2) * (v.mpp - next);
    }
    v.mpp = next;
  }, []);

  // the list of players, and the land's name: a few times a second
  useEffect(() => {
    const read = () => {
      const m = mapHud;
      const list: Row[] = [{ id: 'self', name: selfName || 'You', colour: selfColour, dist: 0, self: true }];
      for (let k = 0; k < m.peerCount; k++) {
        const p = m.peers[k];
        list.push({ id: p.id, name: p.name, colour: characterOf(p.char).swatch, dist: Math.hypot(p.x - m.x, p.z - m.z), self: false });
      }
      setRows(list);
      setLand(m.land);
    };
    read();
    const t = setInterval(read, 300);
    return () => clearInterval(t);
  }, [mapHud, selfName, selfColour]);

  // the keys the engine leaves alone while the map is open
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey) return;
      if (e.code === 'Escape') { e.preventDefault(); onClose(); }
      else if (e.code === 'KeyC') setFollow({ kind: 'self' });
      else if (e.code === 'Equal' || e.code === 'NumpadAdd') zoomBy(1 / 1.4);
      else if (e.code === 'Minus' || e.code === 'NumpadSubtract') zoomBy(1.4);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, setFollow, zoomBy]);

  // drawing
  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !wrap || !ctx) return;
    const svc = tileService();
    let dpr = 1, w = 0, h = 0;
    const fit = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = wrap.clientWidth;
      h = wrap.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(wrap);
    let raf = 0;
    const label = (text: string, x: number, y: number, align: CanvasTextAlign = 'center') => {
      ctx.font = '600 11.5px ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.lineJoin = 'round';
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = 'rgba(8,14,16,0.85)';
      ctx.strokeText(text, x, y);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, x, y);
    };
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const m = mapHud, v = view.current;
      if (document.hidden || w === 0) return;
      // following: the centre goes where the hero (or the player) is
      if (v.follow) {
        if (v.follow.kind === 'self') { v.cx = m.x; v.cz = m.z; }
        else {
          const id = v.follow.id;
          let found = false;
          for (let k = 0; k < m.peerCount; k++) if (m.peers[k].id === id) { v.cx = m.peers[k].x; v.cz = m.peers[k].z; found = true; }
          if (!found) { v.follow = { kind: 'self' }; setFollowing('self'); }
        }
      }
      const clouds = m.world === 'clouds';
      if (!clouds) {
        svc.setSeed(m.seed);
        svc.begin(v.cx, v.cz);
      }
      const { cx, cz, mpp } = v;
      const sx = (x: number) => w / 2 + (x - cx) / mpp;
      const sy = (z: number) => h / 2 - (z - cz) / mpp;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = MAP_GROUND;
      ctx.fillRect(0, 0, w, h);
      ctx.save();
      ctx.translate(w / 2, h / 2);
      if (!clouds) drawTiles(ctx, svc, cx, cz, mpp, Math.hypot(w, h) / 2);
      ctx.restore();
      if (clouds) {
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.font = '20px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(NO_CLOUD_MAP, w / 2, h / 2 - 40);
        ctx.restore();
      }

      // the grid, in round distances, labelled on the edges
      const step = niceAtLeast(110 * mpp);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.beginPath();
      const gx0 = Math.ceil((cx - (w / 2) * mpp) / step) * step, gx1 = cx + (w / 2) * mpp;
      for (let gx = gx0; gx <= gx1; gx += step) { const x = Math.round(sx(gx)) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, h); }
      const gz0 = Math.ceil((cz - (h / 2) * mpp) / step) * step, gz1 = cz + (h / 2) * mpp;
      for (let gz = gz0; gz <= gz1; gz += step) { const y = Math.round(sy(gz)) + 0.5; ctx.moveTo(0, y); ctx.lineTo(w, y); }
      ctx.stroke();
      ctx.font = '500 10px ui-sans-serif, system-ui, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';
      for (let gx = gx0; gx <= gx1; gx += step) ctx.fillText(kmText(gx), sx(gx) + 4, 6);
      ctx.textBaseline = 'bottom';
      for (let gz = gz0; gz <= gz1; gz += step) ctx.fillText(kmText(gz), 6, sy(gz) - 3);

      // the other players, then the hero over them
      const pad = 30;
      for (let k = 0; k < m.peerCount; k++) {
        const p = m.peers[k];
        const px = sx(p.x), py = sy(p.z), colour = characterOf(p.char).swatch;
        if (px > pad && px < w - pad && py > pad && py < h - pad) {
          drawPeer(ctx, px, py, p.yaw, colour, 7);
          label(p.name, px, py - 20);
        } else {
          // past the edge: a pointer on it, towards them, with the name and the distance
          const dx = px - w / 2, dy = py - h / 2;
          const t = Math.min((w / 2 - pad) / Math.abs(dx || 1e-6), (h / 2 - pad) / Math.abs(dy || 1e-6));
          const ex = w / 2 + dx * t, ey = h / 2 + dy * t;
          drawEdgePointer(ctx, ex, ey, Math.atan2(dx, -dy), colour, 9);
          const d = Math.hypot(p.x - cx, p.z - cz);
          const text = `${p.name} · ${d >= 1000 ? `${(d / 1000).toFixed(1)} km` : `${Math.round(d)} m`}`;
          ctx.font = '600 11.5px ui-sans-serif, system-ui, sans-serif';
          const tw = ctx.measureText(text).width;
          const lx = Math.min(w - tw / 2 - 8, Math.max(tw / 2 + 8, ex - (dx / Math.hypot(dx, dy)) * (tw / 2 + 18)));
          const ly = Math.min(h - 14, Math.max(14, ey - (dy / Math.hypot(dx, dy)) * 22));
          label(text, lx, ly);
        }
      }
      const hx = sx(m.x), hy = sy(m.z);
      if (hx > -20 && hx < w + 20 && hy > -20 && hy < h + 20) {
        drawArrow(ctx, hx, hy, m.facing, 11);
        label(selfName || 'You', hx, hy - 24);
      } else {
        const dx = hx - w / 2, dy = hy - h / 2;
        const t = Math.min((w / 2 - pad) / Math.abs(dx || 1e-6), (h / 2 - pad) / Math.abs(dy || 1e-6));
        drawEdgePointer(ctx, w / 2 + dx * t, h / 2 + dy * t, Math.atan2(dx, -dy), '#ffd84a', 9);
      }

      // the scale
      const bar = niceAtLeast(70 * mpp);
      const bw = bar / mpp, bx = 20, by = h - 22;
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = 'rgba(8,14,16,0.8)';
      ctx.beginPath(); ctx.moveTo(bx, by - 5); ctx.lineTo(bx, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by - 5); ctx.stroke();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
      label(kmText(bar), bx + bw / 2, by - 12);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [mapHud, selfName]);

  // pan, zoom, recentre
  const drag = useRef<{ x: number; y: number } | null>(null);
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const v = view.current, rect = e.currentTarget.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    if (cursorRef.current) {
      const x = v.cx + (sx - rect.width / 2) * v.mpp, z = v.cz - (sy - rect.height / 2) * v.mpp;
      cursorRef.current.textContent = `${Math.round(x)} · ${Math.round(z)}`;
    }
    const d = drag.current;
    if (!d) return;
    if (v.follow !== null) setFollow(null);
    v.cx -= (e.clientX - d.x) * v.mpp;
    v.cz += (e.clientY - d.y) * v.mpp;
    drag.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    zoomBy(Math.exp(e.deltaY * 0.0014), e.clientX - rect.left, e.clientY - rect.top);
  };

  const mapKey = keyHint(bindings, 'map');
  const btn = 'flex size-8 items-center justify-center rounded-lg bg-[#0b1214]/80 text-white/80 ring-1 ring-white/15 backdrop-blur transition-colors hover:bg-[#16222a] hover:text-white';
  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/55 p-3 backdrop-blur-sm sm:p-8" data-hud="map" role="dialog" aria-label="Map">
      <div ref={wrapRef} className="relative h-full w-full max-w-[1500px] overflow-hidden rounded-2xl bg-[#142022] shadow-[0_20px_70px_rgba(0,0,0,0.7)] ring-1 ring-white/20">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 block cursor-grab touch-none active:cursor-grabbing"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
          onDoubleClick={() => setFollow({ kind: 'self' })}
        />

        <div className="pointer-events-none absolute left-4 top-4 flex flex-col gap-0.5 text-white drop-shadow-[0_1px_3px_rgba(0,0,0,0.9)]">
          <span className="text-lg font-semibold tracking-wide">Map</span>
          <span className="text-xs text-white/75">{land}</span>
        </div>

        <div className="absolute right-4 top-4 flex items-center gap-1.5">
          <button type="button" className={btn} onClick={() => zoomBy(1 / 1.5)} aria-label="Zoom in"><Plus className="size-4" /></button>
          <button type="button" className={btn} onClick={() => zoomBy(1.5)} aria-label="Zoom out"><Minus className="size-4" /></button>
          <button type="button" className={btn} onClick={() => setFollow({ kind: 'self' })} aria-label="Centre on me" title="Centre on me (C)"><Crosshair className="size-4" /></button>
          <button type="button" className={btn} onClick={onClose} aria-label="Close the map"><X className="size-4" /></button>
        </div>

        {/* who is where */}
        <div className="absolute right-4 top-16 flex max-h-[60%] w-52 flex-col gap-0.5 overflow-y-auto rounded-xl bg-[#0b1214]/80 p-1.5 text-xs text-white ring-1 ring-white/15 backdrop-blur" data-hud="players">
          {rows.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => setFollow(r.self ? { kind: 'self' } : { kind: 'peer', id: r.id })}
              className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-white/10 ${following === r.id ? 'bg-white/15' : ''}`}
            >
              <span className="size-2.5 shrink-0 rounded-full ring-1 ring-white/80" style={{ backgroundColor: r.colour }} />
              <span className="min-w-0 flex-1 truncate">{r.name}</span>
              <span className="tabular-nums text-white/55">{r.self ? 'you' : r.dist >= 1000 ? `${(r.dist / 1000).toFixed(1)} km` : `${Math.round(r.dist)} m`}</span>
            </button>
          ))}
          {rows.length === 1 && <span className="px-2 py-1 text-white/50">No one else is here yet.</span>}
        </div>

        <div className="pointer-events-none absolute bottom-4 right-4 flex items-center gap-3 text-[11px] text-white/70 drop-shadow-[0_1px_2px_rgba(0,0,0,0.9)]">
          <span ref={cursorRef} className="tabular-nums" />
          <span>drag to pan · scroll to zoom · double-click to centre{mapKey ? ` · ${mapKey} or Esc to close` : ' · Esc to close'}</span>
        </div>
        <span className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 text-xs font-bold tracking-widest text-white/60 drop-shadow-[0_1px_2px_rgba(0,0,0,0.9)]">N</span>
      </div>
    </div>
  );
}
