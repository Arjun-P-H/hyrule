'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Moon, Sun, Sunrise, Sunset } from 'lucide-react';
import { useEngineStats, type StatsStore } from '@/lib/engine/statsStore';
import { characterOf } from '@/lib/hero/characters';
import { tempFraction, tempWord } from '@/lib/map/gauges';
import { NO_CLOUD_MAP, type MapHudState } from '@/lib/map/mapHud';
import { tileService } from '@/lib/map/tileService';
import { clockText5, dayPhase } from '@/lib/sky/clock';
import { MAP_GROUND, drawArrow, drawEdgePointer, drawPeer, drawTiles } from './mapDraw';

/**
 * The corner of the screen the map lives in, after Tears of the Kingdom, kept flat and quiet: a round
 * minimap that turns with the view (the hero's arrow always up, a pale cone for what the camera sees, a
 * small N and the hero's coordinates along the rim), the temperature gauge and the sound meter beside it,
 * the time of day under them in steps of five minutes. Other players are on the minimap: a dot where they
 * are in reach, a pointer on the rim towards them where they are not. The map is drawn in its own animation
 * frames from the engine's channel (mapHud.ts), not by React. Everything is sized by the screen's width, as
 * a console's HUD is by its screen's.
 */

const MINI_RADIUS = 360; // metres from the hero to the minimap's rim

type Sizes = { mini: number; gauge: number };

function sizesFor(width: number): Sizes {
  const mini = Math.round(Math.min(224, Math.max(128, width * 0.125)));
  return { mini, gauge: Math.round(mini * 0.225) };
}

function useHudSizes(): Sizes {
  const [sizes, setSizes] = useState(() => sizesFor(typeof window === 'undefined' ? 1440 : window.innerWidth));
  useEffect(() => {
    const onResize = () => setSizes((prev) => {
      const next = sizesFor(window.innerWidth);
      return next.mini === prev.mini ? prev : next;
    });
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return sizes;
}

/** Piecewise-linear colour along the temperature scale (cold → hot): pale ice blue to amber. */
const TEMP_STOPS: ReadonlyArray<readonly [number, readonly [number, number, number]]> = [
  [0, [128, 214, 250]], [0.3, [176, 232, 252]], [0.5, [238, 236, 220]], [0.72, [255, 196, 96]], [1, [255, 138, 42]],
];
function tempColour(t: number): string {
  for (let k = 1; k < TEMP_STOPS.length; k++) {
    const [t1, c1] = TEMP_STOPS[k];
    if (t <= t1) {
      const [t0, c0] = TEMP_STOPS[k - 1];
      const u = (t - t0) / (t1 - t0);
      return `rgb(${c0.map((v, i) => Math.round(v + (c1[i] - v) * u)).join(',')})`;
    }
  }
  const last = TEMP_STOPS[TEMP_STOPS.length - 1][1];
  return `rgb(${last.join(',')})`;
}

/** The dark, translucent disc the gauges sit on. */
const DISC = 'relative rounded-full bg-[#222b30]/70 ring-1 ring-white/[0.13] backdrop-blur-[2px]';

/** A coordinate as the rim prints it: a sign for the south and west, four digits at least. */
const coord = (n: number) => `${n < 0 ? '-' : ''}${String(Math.abs(Math.round(n))).padStart(4, '0')}`;

// ---- the minimap ----------------------------------------------------------------------------

const Minimap = React.memo(function Minimap({ mapHud, size }: { mapHud: MapHudState; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const svc = tileService();
    const R = size / 2;
    const mpp = (2 * MINI_RADIUS) / size;
    const small = size * 0.045; // the arrow, the dots: they grow with the map
    let raf = 0, sig = '';
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const m = mapHud;
      if (!m.live || document.hidden) return;
      // (nothing moved, nothing arrived: the picture is as it was)
      let s = `${svc.arrived}|${m.seed}|${m.world}|${m.x.toFixed(1)}|${m.y.toFixed(0)}|${m.z.toFixed(1)}|${m.heading.toFixed(3)}|${m.facing.toFixed(3)}|${m.peerCount}`;
      for (let k = 0; k < m.peerCount; k++) s += `|${m.peers[k].x.toFixed(1)},${m.peers[k].z.toFixed(1)},${m.peers[k].yaw.toFixed(2)}`;
      if (s === sig) return;
      sig = s;

      const clouds = m.world === 'clouds';
      if (!clouds) {
        svc.setSeed(m.seed);
        svc.begin(m.x, m.z);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      ctx.save();
      ctx.beginPath();
      ctx.arc(R, R, R - 0.5, 0, Math.PI * 2);
      ctx.clip();
      ctx.fillStyle = MAP_GROUND;
      ctx.fillRect(0, 0, size, size);
      // the ground, turned so that the view faces up
      ctx.save();
      ctx.translate(R, R);
      ctx.rotate(-m.heading);
      if (!clouds) drawTiles(ctx, svc, m.x, m.z, mpp, R * 1.02);
      ctx.restore();
      if (clouds) {
        ctx.save();
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.font = `${Math.round(size * 0.065)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(NO_CLOUD_MAP, R, R + size * 0.2);
        ctx.restore();
      }
      // muted: a slate wash over the colours, so that the markers are what the eye finds
      ctx.fillStyle = 'rgba(32,40,45,0.5)';
      ctx.fillRect(0, 0, size, size);
      // what the camera sees: a pale cone fading out towards the rim
      const cone = ctx.createRadialGradient(R, R, 0, R, R, R);
      cone.addColorStop(0, 'rgba(255,255,255,0.2)');
      cone.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.beginPath();
      ctx.moveTo(R, R);
      ctx.arc(R, R, R, -Math.PI / 2 - Math.PI / 4, -Math.PI / 2 + Math.PI / 4);
      ctx.closePath();
      ctx.fillStyle = cone;
      ctx.fill();

      // markers, upright: world → map (z is north) → turned by the heading
      const ch = Math.cos(m.heading), sh = Math.sin(m.heading);
      const place = (x: number, z: number): [number, number] => {
        const sx = (x - m.x) / mpp, sy = -(z - m.z) / mpp;
        return [sx * ch + sy * sh, -sx * sh + sy * ch];
      };
      for (let k = 0; k < m.peerCount; k++) {
        const p = m.peers[k];
        const [px, py] = place(p.x, p.z);
        const d = Math.hypot(px, py), colour = characterOf(p.char).swatch;
        if (d <= R - small * 2.4) drawPeer(ctx, R + px, R + py, p.yaw - m.heading, colour, small * 0.62);
        else drawEdgePointer(ctx, R + (px / d) * (R - small * 1.7), R + (py / d) * (R - small * 1.7), Math.atan2(px, -py), colour, small * 0.85);
      }
      drawArrow(ctx, R, R, m.facing - m.heading, small * 1.05);

      // north, and the hero's x, z and height along the lower rim, each tilted to the curve
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const fontPx = Math.round(size * 0.064);
      ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
      const [nx, ny] = place(m.x, m.z + mpp * (R - fontPx * 1.1));
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.fillText('N', R + nx, R + ny);
      ctx.font = `500 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
      const label = [coord(m.x), coord(m.z), coord(m.y)];
      for (let k = 0; k < 3; k++) {
        const a = ((k - 1) * 33 * Math.PI) / 180; // from straight down
        ctx.save();
        ctx.translate(R + Math.sin(a) * (R - fontPx * 1.15), R + Math.cos(a) * (R - fontPx * 1.15));
        ctx.rotate(-a);
        ctx.fillStyle = 'rgba(255,255,255,0.78)';
        ctx.fillText(label[k], 0, 0);
        ctx.restore();
      }
      ctx.restore();

      ctx.beginPath();
      ctx.arc(R, R, R - 0.7, 0, Math.PI * 2);
      ctx.lineWidth = 1.3;
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.stroke();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mapHud, size]);
  return <canvas ref={ref} style={{ width: size, height: size }} className="rounded-full" aria-hidden />;
});

// ---- the temperature gauge --------------------------------------------------------------------

const SEGMENTS = 9;
const SWEEP = 270; // degrees of arc, open at the bottom
const GAP = 5;     // degrees between segments

const TempGauge = React.memo(function TempGauge({ temp, size }: { temp: number; size: number }) {
  const f = tempFraction(temp);
  const c = size / 2;
  const r = size * 0.35;
  const seg = (SWEEP - GAP * (SEGMENTS - 1)) / SEGMENTS;
  const at = (deg: number) => {
    const a = (deg * Math.PI) / 180;
    return [c + Math.sin(a) * r, c - Math.cos(a) * r] as const;
  };
  const arcs = [];
  for (let k = 0; k < SEGMENTS; k++) {
    const a0 = -SWEEP / 2 + k * (seg + GAP), a1 = a0 + seg;
    const mid = (k + 0.5) / SEGMENTS;
    const [x0, y0] = at(a0), [x1, y1] = at(a1);
    arcs.push(
      <path
        key={k}
        d={`M${x0.toFixed(2)} ${y0.toFixed(2)}A${r} ${r} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`}
        fill="none" stroke={mid <= f + 0.5 / SEGMENTS ? tempColour(mid) : 'rgba(255,255,255,0.16)'} strokeWidth={size * 0.115}
      />,
    );
  }
  const needle = -SWEEP / 2 + SWEEP * f;
  return (
    <div className={DISC} style={{ width: size, height: size }} title={`${tempWord(temp)}: ${temp.toFixed(0)} °C`} data-hud="temperature">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        {arcs}
        <g transform={`rotate(${needle} ${c} ${c})`}>
          <line x1={c} y1={c} x2={c} y2={c - size * 0.2} stroke="#fff" strokeWidth={size * 0.055} strokeLinecap="round" />
        </g>
        <circle cx={c} cy={c} r={size * 0.062} fill="#fff" />
        <text x={c} y={size * 0.9} textAnchor="middle" fontSize={size * 0.2} fontWeight="700" fill="#fff" fontFamily="ui-sans-serif, system-ui, sans-serif">°C</text>
      </svg>
    </div>
  );
});

// ---- the sound meter --------------------------------------------------------------------------

const COLUMNS = 26;

const SoundMeter = React.memo(function SoundMeter({ mapHud, size }: { mapHud: MapHudState; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const trace = new Float32Array(COLUMNS);
    let raf = 0, pushed = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (document.hidden) return;
      const n = mapHud.noise;
      // a new column of the trace about every 40 ms: as loud as the hero is, with the jitter of a voice on a scope,
      // and a faint tremor even at rest
      if (now - pushed > 40) {
        pushed = now;
        trace.copyWithin(0, 1);
        trace[COLUMNS - 1] = (0.05 + n * 0.95) * (Math.random() * 2 - 1) * (n < 0.02 ? 0.35 : 1);
      }
      // quiet: violet; loud: warm pink
      const hot = Math.min(1, Math.max(0, (n - 0.3) / 0.6));
      const colour = `rgb(${Math.round(168 + 87 * hot)},${Math.round(130 - 22 * hot)},${Math.round(236 - 100 * hot)})`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      const c = size / 2, w = size * 0.8, x0 = (size - w) / 2;
      ctx.beginPath();
      for (let k = 0; k < COLUMNS; k++) {
        const x = x0 + (w * k) / (COLUMNS - 1);
        // (the ends fade into the disc)
        const edge = Math.sin((Math.PI * k) / (COLUMNS - 1));
        const y = c + trace[k] * edge * size * 0.36;
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.lineWidth = Math.max(1.2, size * 0.04);
      ctx.strokeStyle = colour;
      ctx.stroke();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mapHud, size]);
  return (
    <div className={DISC} style={{ width: size, height: size }} title="Sound: how much noise you make" data-hud="sound">
      <canvas ref={ref} style={{ width: size, height: size }} aria-hidden />
    </div>
  );
});

// ---- the time of day ------------------------------------------------------------------------------

const PHASE_ICONS = [Moon, Sunrise, Sun, Sunset] as const;
function phaseIndex(hour: number): number {
  switch (dayPhase(hour)) {
    case 'Night': return 0;
    case 'Dawn': case 'Sunrise': case 'Morning glow': return 1;
    case 'Morning': case 'Midday': case 'Afternoon': return 2;
    default: return 3;
  }
}

const TimeOfDay = React.memo(function TimeOfDay({ hour, size }: { hour: number; size: number }) {
  const now = phaseIndex(hour);
  const icon = size * 0.3;
  return (
    <div className="flex flex-col items-start" style={{ gap: size * 0.1 }}>
      <span className="pl-0.5 font-normal leading-none tracking-wide tabular-nums text-white/95" style={{ fontSize: size * 0.33 }} data-hud="clock">
        {clockText5(hour)}
      </span>
      <div className="flex items-center rounded-full bg-[#222b30]/70 ring-1 ring-white/[0.13] backdrop-blur-[2px]" style={{ gap: size * 0.2, padding: `${size * 0.19}px ${size * 0.26}px` }}>
        {PHASE_ICONS.map((Icon, k) => (
          <span key={k} className="relative flex">
            {k === now && <span className="absolute left-1/2 -translate-x-1/2 border-x-transparent border-t-white/90" style={{ top: -size * 0.15, borderWidth: `${size * 0.07}px ${size * 0.06}px 0` }} />}
            <Icon className={k === now ? 'text-white' : 'text-white/30'} style={{ width: icon, height: icon }} strokeWidth={k === now ? 2.2 : 1.7} />
          </span>
        ))}
      </div>
    </div>
  );
});

// ---- the cluster ----------------------------------------------------------------------------------

export const MapHud = React.memo(function MapHud({ statsStore, mapHud, onOpen }: { statsStore: StatsStore; mapHud: MapHudState; onOpen: () => void }) {
  const stats = useEngineStats(statsStore);
  const { mini, gauge } = useHudSizes();
  return (
    <div className="pointer-events-none absolute bottom-4 right-4 z-20 flex select-none items-end font-sans" style={{ gap: mini * 0.07 }}>
      <div className="flex flex-col items-end" style={{ gap: gauge * 0.22 }}>
        <TempGauge temp={mapHud.temp} size={gauge} />
        <SoundMeter mapHud={mapHud} size={gauge} />
        <div className="mt-1 self-stretch">
          <TimeOfDay hour={Math.floor((stats?.hour ?? 12) * 12) / 12} size={gauge} />
        </div>
      </div>
      <button
        type="button"
        onClick={onOpen}
        title="Open the map"
        aria-label="Open the map"
        data-hud="minimap"
        className="pointer-events-auto cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-white/70"
      >
        <Minimap mapHud={mapHud} size={mini} />
      </button>
    </div>
  );
});
