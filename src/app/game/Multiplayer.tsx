'use client';

import React, { useEffect, useState } from 'react';
import { Check, Copy, Globe, Loader2, Users, WifiOff } from 'lucide-react';
import { CHARACTERS, characterOf } from '@/lib/hero/characters';
import type { NetView } from '@/lib/net/session';
import type { RenderApi } from '@/lib/engine/statsStore';

const SKIN = '#e8b48c', HAIR = '#dba443', LEATHER = '#6e4220', CREAM = '#ebdfc4', GOLD = '#e3b53b';
const SHIELD = '#28408f', SILVER = '#c9ccd2', HILT = '#2c3a8e', EYE = '#1c3350';

/** The hero, front on, in a dress: the same figure the world draws (cap, tunic, belt, boots, shield). */
function HeroFigure({ color, className }: { color: string; className?: string }) {
  return (
    <svg viewBox="0 0 100 150" className={className} aria-hidden>
      <ellipse cx="50" cy="143" rx="24" ry="4.5" fill="#000" opacity="0.35" />
      {/* shield on the back, sword over the shoulder */}
      <path d="M61 60 h24 v25 q0 15 -12 21 q-12 -6 -12 -21 z" fill={SHIELD} stroke={SILVER} strokeWidth="2.5" />
      <rect x="25" y="40" width="6" height="22" rx="2" transform="rotate(-32 28 51)" fill={HILT} />
      <rect x="21" y="52" width="14" height="4" rx="1.5" transform="rotate(-32 28 54)" fill={GOLD} />
      {/* legs, boots */}
      <rect x="38" y="102" width="9.5" height="26" rx="3" fill={CREAM} />
      <rect x="52.5" y="102" width="9.5" height="26" rx="3" fill={CREAM} />
      <path d="M37 122 h11.5 v16 h-15 q-1.5 -3 3.5 -5 z" fill={LEATHER} />
      <path d="M51.5 122 h11.5 v11 q5 2 3.5 5 h-15 z" fill={LEATHER} />
      {/* arms: sleeves, gauntlets, hands */}
      <rect x="22" y="64" width="10.5" height="31" rx="5" fill={CREAM} />
      <rect x="67.5" y="64" width="10.5" height="31" rx="5" fill={CREAM} />
      <rect x="21.5" y="82" width="11.5" height="13" rx="3" fill={LEATHER} />
      <rect x="67" y="82" width="11.5" height="13" rx="3" fill={LEATHER} />
      <circle cx="27.2" cy="99" r="5" fill={SKIN} />
      <circle cx="72.8" cy="99" r="5" fill={SKIN} />
      {/* tunic and its short sleeves */}
      <path d="M36 58 Q50 54 64 58 L77 66 L74 77 L68 74 L73 113 Q50 120 27 113 L32 74 L26 77 L23 66 Z" fill={color} />
      <path d="M36 58 Q50 54 64 58 L62 63 Q50 60 38 63 Z" fill="#000" opacity="0.18" />
      {/* belt and buckle */}
      <path d="M30.5 86 Q50 90.5 69.5 86 L70 93 Q50 97.5 30 93 Z" fill={LEATHER} />
      <rect x="45.5" y="86.5" width="9" height="8.5" rx="1.5" fill={GOLD} />
      <rect x="48" y="89.2" width="4" height="3.2" fill={LEATHER} />
      {/* neck, ears, head */}
      <rect x="45" y="52" width="10" height="9" fill={SKIN} />
      <path d="M35 42 L19 33 L34 50 Z" fill={SKIN} />
      <path d="M65 42 L81 33 L66 50 Z" fill={SKIN} />
      <circle cx="50" cy="43" r="16" fill={SKIN} />
      {/* hair */}
      <path d="M34 41 Q35 29 50 29.5 Q65 29 66 41 Q61 35.5 57 38.5 Q53 34 49 38.5 Q45 34.5 41 39 Q37 37 35 45 Z" fill={HAIR} />
      <path d="M34.5 40 L33.5 55 L38 49 Z" fill={HAIR} />
      <path d="M65.5 40 L66.5 55 L62 49 Z" fill={HAIR} />
      {/* eyes */}
      <ellipse cx="44" cy="46.5" rx="2.1" ry="2.9" fill={EYE} />
      <ellipse cx="56" cy="46.5" rx="2.1" ry="2.9" fill={EYE} />
      {/* cap: over the hair, its tip down the back */}
      <path d="M33 38 Q33.5 19 50 18.5 Q66 18.5 68 34 Q80 46 92 71 Q78 61 66 44 Q50 31 33 38 Z" fill={color} />
      <path d="M33 38 Q33.5 19 50 18.5 Q66 18.5 68 34 Q80 46 92 71 Q78 61 66 44 Q50 31 33 38 Z" fill="#000" opacity="0.14" />
    </svg>
  );
}

const DENIED: Record<NonNullable<NetView['denied']>, string> = {
  taken: 'Someone just took that hero — pick another.',
  full: 'This world is full (five players).',
  bad: 'The server refused that choice — pick again.',
};

/** Copy a link, showing a tick for a moment. */
function CopyLink({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard?.writeText(url).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }, () => {});
      }}
      className="flex min-w-0 items-center gap-1.5 rounded-md border border-stone-700 bg-stone-900/80 px-2 py-1 font-mono text-[11px] text-stone-300 hover:border-emerald-600 hover:text-emerald-300"
    >
      {copied ? <Check className="size-3.5 shrink-0 text-emerald-400" /> : <Copy className="size-3.5 shrink-0" />}
      <span className="truncate">{url}</span>
    </button>
  );
}

/** The link to give friends: the server's tunnel, else this page's own address unless it is this computer's. */
function inviteLink(view: NetView): string | null {
  if (view.share) return view.share;
  if (typeof window === 'undefined') return null;
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  return local ? null : `${location.origin}/game`;
}

/**
 * Before the game: a name and one of the five heroes (each player in a world
 * wears a different dress), over the world already running behind it — the
 * hero standing there wears the dress picked.
 */
export function Lobby({
  view, name, setName, character, setCharacter, onEnter, api,
}: {
  view: NetView;
  name: string;
  setName: (v: string) => void;
  character: number;
  setCharacter: (c: number) => void;
  onEnter: () => void;
  api: RenderApi | null;
}) {
  const online = view.status === 'online';
  const takenBy = new Map(view.players.filter((p) => p.id !== view.selfId).map((p) => [p.char, p.name]));
  const full = online && !view.inWorld && view.players.length >= view.max;
  const canEnter = !takenBy.has(character) && !full;
  const invite = inviteLink(view);

  return (
    <div className="absolute inset-0 z-40 flex flex-col justify-between bg-gradient-to-b from-stone-950/70 via-stone-950/10 to-stone-950/90 p-4 pointer-events-auto">
      <div className="mx-auto mt-4 text-center">
        <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-wide text-amber-100 drop-shadow-lg">Choose your hero</h1>
        <p className="mt-2 font-mono text-xs text-stone-300">
          {view.status === 'connecting' && (
            <span className="inline-flex items-center gap-1.5"><Loader2 className="size-3.5 animate-spin" /> Connecting to the world…</span>
          )}
          {online && (
            <span className="inline-flex items-center gap-1.5 text-emerald-300">
              <Globe className="size-3.5" /> Online · {view.players.length} of {view.max} in this world
              {view.players.length > 0 && <span className="text-stone-400">({view.players.map((p) => p.name).join(', ')})</span>}
            </span>
          )}
          {view.status === 'offline' && (
            <span className="inline-flex items-center gap-1.5 text-amber-300">
              <WifiOff className="size-3.5" /> No multiplayer server here — you will play alone (start play.command / play.bat to play with friends)
            </span>
          )}
        </p>
      </div>

      <div className="mx-auto w-full max-w-3xl rounded-2xl border border-stone-700/80 bg-stone-950/85 p-4 shadow-2xl backdrop-blur-md">
        <div className="grid grid-cols-5 gap-2 sm:gap-3">
          {CHARACTERS.map((c) => {
            const owner = takenBy.get(c.id);
            const selected = c.id === character;
            return (
              <button
                key={c.id}
                type="button"
                disabled={owner !== undefined}
                onClick={() => setCharacter(c.id)}
                aria-pressed={selected}
                className={`group relative flex flex-col items-center rounded-xl border-2 px-1 pb-2 pt-1 transition ${
                  selected ? 'bg-stone-800/90' : 'border-stone-800 bg-stone-900/70 hover:border-stone-600'
                } disabled:cursor-not-allowed disabled:opacity-40`}
                style={selected ? { borderColor: c.swatch, boxShadow: `0 0 18px ${c.swatch}66` } : undefined}
              >
                <HeroFigure color={c.swatch} className="h-24 w-auto sm:h-32 transition-transform group-enabled:group-hover:-translate-y-0.5" />
                <span className="mt-1 font-mono text-xs font-bold" style={{ color: c.swatch }}>{c.name}</span>
                <span className="h-4 truncate font-mono text-[10px] text-stone-400 max-w-full">
                  {owner !== undefined ? `played by ${owner}` : selected ? 'you' : ''}
                </span>
              </button>
            );
          })}
        </div>

        <form
          className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-end"
          onSubmit={(e) => { e.preventDefault(); if (canEnter) onEnter(); }}
        >
          <label className="flex flex-1 flex-col gap-1 font-mono text-[11px] text-stone-400">
            Your name
            <input
              value={name}
              onChange={(e) => setName(e.target.value.slice(0, 16))}
              maxLength={16}
              placeholder={`${characterOf(character).name} hero`}
              className="rounded-md border border-stone-700 bg-stone-900 px-3 py-2 text-sm text-stone-100 outline-none focus:border-emerald-500"
            />
          </label>
          <button
            type="submit"
            disabled={!canEnter}
            className="rounded-md px-6 py-2.5 font-mono text-sm font-bold text-black shadow-lg transition disabled:cursor-not-allowed disabled:opacity-40"
            style={{ backgroundColor: characterOf(character).swatch }}
          >
            Enter the world as {characterOf(character).name}
          </button>
        </form>

        {(view.denied || full) && (
          <p className="mt-2 font-mono text-xs text-red-400">{full ? DENIED.full : DENIED[view.denied!]}</p>
        )}
        {api === 'webgl2' && (
          <p className="mt-2 font-mono text-[11px] text-amber-300">
            This browser draws with WebGL2, which shows only your own hero. To see your friends, use Chrome or Edge on the https link.
          </p>
        )}
        {online && (
          <div className="mt-3 flex flex-wrap items-center gap-2 font-mono text-[11px] text-stone-400">
            <Users className="size-3.5" />
            {invite ? <>Invite friends: <CopyLink url={invite} /></> : 'Invite friends with the https link the server window shows (play.command / play.bat).'}
          </div>
        )}
      </div>
    </div>
  );
}

/** In game: who is in the world (each in their dress), arrivals and departures, the invitation. */
export const PlayersPanel = React.memo(function PlayersPanel({ view, api }: { view: NetView; api: RenderApi | null }) {
  const [shownAt, setShownAt] = useState(0);
  const notice = view.notice;
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setShownAt(notice.at), 5000);
    return () => clearTimeout(t);
  }, [notice]);
  if (view.status === 'offline' && !view.inWorld) return null;
  const invite = inviteLink(view);
  return (
    <div className="flex w-[220px] flex-col gap-1.5 rounded-lg border border-stone-800/80 bg-stone-950/85 p-2.5 font-mono text-[11px] text-stone-300 shadow-2xl backdrop-blur-md">
      <div className="flex items-center justify-between border-b border-stone-800 pb-1">
        <span className="flex items-center gap-1.5 font-bold text-stone-200"><Users className="size-3.5" /> Players</span>
        <span className={view.status === 'online' ? 'text-emerald-400' : 'text-amber-300'}>
          {view.status === 'online' ? `${view.players.length}/${view.max}` : 'reconnecting…'}
        </span>
      </div>
      {view.players.map((p) => (
        <div key={p.id} className="flex items-center gap-2">
          <span className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: characterOf(p.char).swatch }} />
          <span className="truncate">{p.name}</span>
          {p.id === view.selfId && <span className="text-stone-500">(you)</span>}
        </div>
      ))}
      {notice && notice.at !== shownAt && <div className="text-amber-200">{notice.text}</div>}
      {api === 'webgl2' && <div className="text-amber-300">WebGL2: friends aren&apos;t drawn — use Chrome/Edge over https.</div>}
      {invite && view.status === 'online' && <CopyLink url={invite} />}
    </div>
  );
});
