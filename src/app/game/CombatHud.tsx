'use client';

import React, { useEffect, useRef } from 'react';
import { useCombatHud, type CombatHudStore, type HudMarker } from '@/lib/combat/hud';
import { ItemIcon } from './ItemIcon';

/**
 * The combat HUD (third person): hearts filled by quarters, the weapon, shield
 * and bow in hand with what each has left, the arrows and the food kept, the crosshair while drawing, a
 * prompt for the action at hand, a banner for a Perfect Guard or a Flurry Rush,
 * a vignette while the world is slowed, and Game Over. Its state comes from
 * the engine on change (combat/hud.ts); the markers over the foes (lock-on,
 * their hearts, "!" and "?") are moved every animation frame without React.
 */

// One heart, filled by quarters (0–4), as TotK draws them: small and squat, one flat light coral-red, no outline
// and no gradient; the empty part a faint ghost of it.
const HEART_PATH = 'M13 20.6C12.6 20.3 1.2 13.2 1.2 6.9 1.2 3.6 3.7 1.2 6.9 1.2c2.4 0 4.6 1.4 6.1 3.6 1.5-2.2 3.7-3.6 6.1-3.6 3.2 0 5.7 2.4 5.7 5.7 0 6.3-11.4 13.4-11.8 13.7z';
const HEART_RED = '#f0705f';
function Heart({ q }: { q: number }) {
  const id = React.useId();
  // (quarters fill clockwise from the top left)
  const fill = q >= 4 ? 'M0 0h26v22H0z' : q === 3 ? 'M0 0h26v22H13V11H0z' : q === 2 ? 'M0 0h26v11H0z' : q === 1 ? 'M0 0h13v11H0z' : '';
  return (
    <svg viewBox="0 0 26 22" style={{ width: '1.3em', height: '1.1em' }} aria-hidden>
      <defs>
        <clipPath id={id}><path d={fill || 'M0 0h0v0H0z'} /></clipPath>
      </defs>
      <path d={HEART_PATH} fill="#fff" fillOpacity="0.2" />
      {q > 0 && <path d={HEART_PATH} fill={HEART_RED} clipPath={`url(#${id})`} />}
    </svg>
  );
}

/** A row of hearts: `health` quarters out of `hearts` containers (ten to a row), sized with the screen like the rest of the HUD. */
export function HeartRow({ health, hearts }: { health: number; hearts: number }) {
  const row: number[] = [];
  for (let k = 0; k < hearts; k++) row.push(Math.max(0, Math.min(4, health - 4 * k)));
  return (
    <div
      className="flex flex-wrap items-center"
      style={{ fontSize: 'clamp(10px, 1vw, 18px)', gap: '0.34em', maxWidth: '16.4em' }}
      aria-label={`${health / 4} of ${hearts} hearts`}
      data-hud="hearts"
    >
      {row.map((q, k) => <Heart key={k} q={q} />)}
    </div>
  );
}

export const Hearts = React.memo(function Hearts({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  if (!s) return null;
  return <HeartRow health={s.health} hearts={s.hearts} />;
});

/** What an item has left, as a thin bar (nothing for what never wears out). */
function WearBar({ wear }: { wear: number }) {
  if (wear >= 1) return null;
  return (
    <span className="mt-0.5 h-[3px] w-24 rounded-full bg-white/15">
      <span className={`block h-[3px] rounded-full ${wear < 0.25 ? 'bg-[#ff5a36]' : 'bg-white/85'}`} style={{ width: `${wear * 100}%` }} />
    </span>
  );
}

const ROW = 'flex items-center gap-2 rounded-xl bg-[#0b1214]/60 px-2.5 py-1.5 ring-1 ring-white/10 backdrop-blur-sm';

const Equipment = React.memo(function Equipment({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  if (!s) return null;
  return (
    <div className="pointer-events-none absolute bottom-[17.5rem] right-4 z-20 flex flex-col items-end gap-1.5 font-sans text-[11px] text-white">
      <div className={ROW} data-hud="weapon">
        <span className={s.weapon ? '' : 'opacity-30'}><ItemIcon cat="weapon" kind={Math.max(0, s.weaponKind)} /></span>
        <div className="flex flex-col">
          <span className="font-semibold">{s.weapon ?? 'No weapon'}</span>
          {s.weapon && <WearBar wear={s.wear} />}
        </div>
        {s.weapons > 1 && <span className="text-stone-400">X · {s.weapons}</span>}
      </div>
      <div className={ROW} data-hud="shield">
        <span className={s.shield ? '' : 'opacity-30'}><ItemIcon cat="shield" kind={Math.max(0, s.shieldKind)} /></span>
        <div className="flex flex-col">
          <span className="font-semibold">{s.shield ?? 'No shield'}</span>
          {s.shield && <WearBar wear={s.shieldWear} />}
        </div>
      </div>
      <div className={ROW} data-hud="bow">
        <span className={s.bow ? '' : 'opacity-30'}><ItemIcon cat="bow" kind={Math.max(0, s.bowKind)} /></span>
        <div className="flex flex-col">
          <span className={s.arrows === 0 || !s.bow ? 'text-red-400' : ''}>{s.bow ? `× ${s.arrows}` : 'No bow'}</span>
          {s.bow && <WearBar wear={s.bowWear} />}
        </div>
      </div>
      {s.food > 0 && (
        <div className={ROW} data-hud="food">
          <ItemIcon cat="food" kind={0} />
          <span>× {s.food}</span>
        </div>
      )}
    </div>
  );
});

const Crosshair = React.memo(function Crosshair({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  if (!s?.aiming) return null;
  const r = 13, c = 2 * Math.PI * r;
  return (
    <>
      <svg className="pointer-events-none absolute left-1/2 top-1/2 z-20 -translate-x-1/2 -translate-y-1/2" width="40" height="40" viewBox="0 0 40 40" aria-hidden>
        <circle cx="20" cy="20" r="2" fill="white" />
        <circle cx="20" cy="20" r={r} fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="2" />
        <circle cx="20" cy="20" r={r} fill="none" stroke={s.draw >= 1 ? '#fde68a' : 'white'} strokeWidth="2" strokeDasharray={c}
          strokeDashoffset={c * (1 - s.draw)} transform="rotate(-90 20 20)" style={{ transition: 'stroke-dashoffset 60ms linear' }} />
      </svg>
      {/* the arrows left, small, beside the target */}
      <span
        data-hud="arrows"
        className={`pointer-events-none absolute left-[calc(50%+25px)] top-1/2 z-20 -translate-y-1/2 text-[10px] font-medium leading-none tabular-nums ${s.arrows === 0 ? 'text-[#ff8f80]' : 'text-white/85'}`}
      >
        {s.arrows}
      </span>
    </>
  );
});

const Prompt = React.memo(function Prompt({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  if (!s?.prompt) return null;
  const [key, text] = s.prompt.split('|');
  return (
    <div className={`pointer-events-none absolute bottom-24 left-1/2 z-20 -translate-x-1/2 flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-xs backdrop-blur ${s.rush ? 'border-cyan-300/80 bg-cyan-950/70 text-cyan-100 animate-pulse' : 'border-stone-600/70 bg-stone-950/75 text-stone-100'}`}>
      <kbd className="rounded bg-stone-100 px-1.5 py-0.5 text-[10px] font-bold text-stone-900">{key}</kbd>
      <span>{text}</span>
    </div>
  );
});

const Banner = React.memo(function Banner({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  if (!s?.banner || s.dead) return null;
  return (
    <div key={s.bannerId} className="pointer-events-none absolute left-1/2 top-[30%] z-20 -translate-x-1/2 animate-in fade-in zoom-in-95 duration-200">
      <span className="font-serif text-2xl tracking-wide text-amber-50 drop-shadow-[0_2px_6px_rgba(0,0,0,0.85)]">{s.banner}</span>
    </div>
  );
});

const Vignette = React.memo(function Vignette({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  const on = !!s?.slowed && !s.dead;
  return (
    <div aria-hidden className={`pointer-events-none absolute inset-0 z-10 transition-opacity duration-300 ${on ? 'opacity-100' : 'opacity-0'}`}
      style={{ background: 'radial-gradient(ellipse at center, transparent 55%, rgba(60,140,200,0.32) 100%)' }} />
  );
});

const GameOver = React.memo(function GameOver({ store }: { store: CombatHudStore }) {
  const s = useCombatHud(store);
  if (!s?.dead) return null;
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-red-950/40 animate-in fade-in duration-700">
      <span className="font-serif text-5xl tracking-[0.2em] text-red-100 drop-shadow-[0_3px_10px_rgba(0,0,0,0.9)]">GAME OVER</span>
      {s.downBy && <span className="font-serif text-xl tracking-wide text-red-50/90 drop-shadow-[0_2px_6px_rgba(0,0,0,0.9)]">Felled by {s.downBy}</span>}
    </div>
  );
});

// ---- markers over the foes: moved every animation frame, no React re-render ----
const MARKERS = 24;

function markerHtml(kind: HudMarker['kind']): string {
  switch (kind) {
    case 'lock':
      return '<svg width="34" height="34" viewBox="0 0 34 34" style="position:absolute;left:-17px;top:-17px"><path d="M17 2l5 8H12z" fill="#ff6a3d"/><path d="M17 32l5-8H12z" fill="#ff6a3d"/><circle cx="17" cy="17" r="9" fill="none" stroke="#ffb199" stroke-width="1.5" stroke-dasharray="4 3"/></svg>';
    case 'bar':
      return '<div style="position:absolute;left:-28px;top:-3px;width:56px;height:6px;border-radius:3px;background:rgba(15,10,10,0.7);box-shadow:0 0 0 1px rgba(255,255,255,0.35)"><div data-fill style="height:6px;border-radius:3px;background:#e3263b"></div></div>';
    case 'alert':
      return '<div style="position:absolute;left:-10px;top:-30px;font:900 30px/30px Georgia,serif;color:#ff3b30;-webkit-text-stroke:1.5px #fff;text-shadow:0 2px 4px rgba(0,0,0,.7)">!</div>';
    default:
      return '<div data-fill style="position:absolute;left:-9px;top:-26px;font:900 24px/26px Georgia,serif;color:#ffd23f;-webkit-text-stroke:1.2px #fff;text-shadow:0 2px 4px rgba(0,0,0,.7)">?</div>';
  }
}

const Markers = React.memo(function Markers({ store }: { store: CombatHudStore }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const pool: HTMLDivElement[] = [];
    const kinds: string[] = [];
    for (let k = 0; k < MARKERS; k++) {
      const el = document.createElement('div');
      el.style.cssText = 'position:absolute;left:0;top:0;display:none;will-change:transform';
      root.appendChild(el);
      pool.push(el);
      kinds.push('');
    }
    let raf = 0, seen = -1;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const mk = store.markers;
      if (mk.version === seen) return;
      seen = mk.version;
      for (let i = 0; i < MARKERS; i++) {
        const el = pool[i];
        if (i >= mk.count) { if (el.style.display !== 'none') el.style.display = 'none'; continue; }
        const m = mk.list[i];
        if (kinds[i] !== m.kind) { kinds[i] = m.kind; el.innerHTML = markerHtml(m.kind); }
        el.style.display = 'block';
        el.style.transform = `translate(${m.x.toFixed(1)}px, ${m.y.toFixed(1)}px) scale(${(0.75 + 0.35 * m.near).toFixed(3)})`;
        if (m.kind === 'bar') {
          const fill = el.querySelector<HTMLElement>('[data-fill]');
          if (fill) fill.style.width = `${Math.max(0, Math.min(1, m.value)) * 100}%`;
        } else if (m.kind === 'notice') {
          const q = el.querySelector<HTMLElement>('[data-fill]');
          if (q) q.style.opacity = (0.35 + 0.65 * m.value).toFixed(2);
        }
      }
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      for (const el of pool) el.remove();
    };
  }, [store]);
  return <div ref={ref} aria-hidden className="pointer-events-none absolute inset-0 z-10 overflow-hidden" />;
});

/**
 * Everything of the combat HUD but the hearts (which sit with the page's top-left panel). What the fight
 * is read by — the slowed-time vignette, the foes' markers, the bow's crosshair, the banners, Game Over —
 * is always shown; the prompts and the gear (the weapon, shield, bow and food in hand) only with the
 * whole HUD (`full`: the + key).
 */
export function CombatHud({ store, show, full }: { store: CombatHudStore; show: boolean; full: boolean }) {
  if (!show) return null;
  return (
    <>
      <Vignette store={store} />
      <Markers store={store} />
      <Crosshair store={store} />
      <Banner store={store} />
      {full && <Prompt store={store} />}
      {full && <Equipment store={store} />}
      <GameOver store={store} />
    </>
  );
}
