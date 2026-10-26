'use client';

import React from 'react';

/**
 * The icons of the weapons, shields, bows and food (combat/weapons.ts tables, by index): small
 * drawings on a 48-unit square, shared by the inventory screen and the combat HUD.
 */

export type IconCat = 'weapon' | 'shield' | 'bow' | 'food';

const WOOD = '#8a5a2b', WOOD_D = '#4a2f14', BONE = '#e8e0cc', BONE_D = '#8b8269', STEEL = '#b8bec9';

function Weapon({ kind }: { kind: number }) {
  if (kind === 0) {
    // the Master Sword: a slim blade, a blue guard, a gold pommel
    return (
      <g>
        <path d="M6 42l8-8" stroke="#2a3270" strokeWidth="3.6" strokeLinecap="round" />
        <path d="M11.3 25.3l11.4 11.4" stroke="#3f4fbf" strokeWidth="3.4" strokeLinecap="round" />
        <path d="M17.8 27.6 38.6 6.8l3.9-1.3-1.3 3.9-20.8 20.8z" fill="#eef3f9" stroke="#8ea2bd" strokeWidth="0.7" strokeLinejoin="round" />
        <path d="M19.5 28.5 40 8" stroke="#b9c7da" strokeWidth="0.8" />
        <circle cx="17" cy="31" r="1.5" fill="#e3b53b" />
        <circle cx="5.5" cy="42.5" r="2.4" fill="#d9a836" />
      </g>
    );
  }
  const bone = kind === 3;
  const head = bone ? BONE : WOOD;
  const edge = bone ? BONE_D : WOOD_D;
  return (
    <g>
      <path d="M8 41 26 23" stroke={bone ? '#cfc6ad' : '#6b4423'} strokeWidth="4.5" strokeLinecap="round" />
      <path d="M24 25 38 11" stroke={edge} strokeWidth="12.4" strokeLinecap="round" />
      <path d="M24 25 38 11" stroke={head} strokeWidth="10.6" strokeLinecap="round" />
      <path d="M26 22 36 12" stroke={bone ? '#fbf7ea' : '#a87442'} strokeWidth="2.4" strokeLinecap="round" />
      {kind === 2 && [[36, 7, -0.6], [42, 14, 0.4], [30, 20, 2.2], [40, 20, 1.2], [29, 11, -1.8]].map(([x, y, a], i) => (
        <path key={i} d="M-2.4 0 0 -6.4 2.4 0z" fill="#cfd3da" stroke="#6d7581" strokeWidth="0.6" transform={`translate(${x} ${y}) rotate(${(a * 180) / Math.PI})`} />
      ))}
      {bone && <><circle cx="40.5" cy="8.5" r="3.4" fill={head} stroke={edge} strokeWidth="1" /><circle cx="22.5" cy="26.5" r="3.4" fill={head} stroke={edge} strokeWidth="1" /></>}
    </g>
  );
}

function Shield({ kind }: { kind: number }) {
  const clip = React.useId();
  switch (kind) {
    case 1: // Wooden Shield
      return (
        <g>
          <defs><clipPath id={clip}><circle cx="24" cy="24" r="18" /></clipPath></defs>
          <circle cx="24" cy="24" r="18" fill="#a06f3c" />
          <g clipPath={`url(#${clip})`} stroke="#6b4423" strokeWidth="1.3">
            <path d="M12 0v48M18 0v48M24 0v48M30 0v48M36 0v48" />
            <path d="M0 17h48M0 31h48" stroke="#7b8088" strokeWidth="2.6" />
          </g>
          <circle cx="24" cy="24" r="18" fill="none" stroke="#5a3a1a" strokeWidth="2.2" />
        </g>
      );
    case 2: // Traveler's Shield
      return (
        <g>
          <circle cx="24" cy="24" r="18" fill="#7a5232" stroke={STEEL} strokeWidth="2.6" />
          <circle cx="24" cy="24" r="11.5" fill="none" stroke="#5b3a20" strokeWidth="1.5" />
          <circle cx="24" cy="24" r="5" fill="#c3c9d3" stroke="#6f7783" strokeWidth="1.2" />
          {[0, 1, 2, 3, 4, 5].map((i) => <circle key={i} cx={24 + 14.6 * Math.cos(i * 1.047)} cy={24 + 14.6 * Math.sin(i * 1.047)} r="1.1" fill={STEEL} />)}
        </g>
      );
    case 3: // Pot Lid
      return (
        <g>
          <circle cx="24" cy="26" r="16" fill="#aab1bb" stroke="#6d7581" strokeWidth="2" />
          <circle cx="24" cy="26" r="11" fill="none" stroke="#8a919c" strokeWidth="1.5" />
          <path d="M16 20c2-3 5-5 8-5" stroke="#dfe3e8" strokeWidth="1.6" fill="none" strokeLinecap="round" />
          <path d="M20.5 22.5h7a1.6 1.6 0 0 1 1.6 1.6v1.2a1.6 1.6 0 0 1-1.6 1.6h-7a1.6 1.6 0 0 1-1.6-1.6v-1.2a1.6 1.6 0 0 1 1.6-1.6z" fill="#d5dae1" stroke="#6d7581" strokeWidth="1" />
          <path d="M24 10V8" stroke="#6d7581" strokeWidth="2" strokeLinecap="round" />
        </g>
      );
    case 4: // Boko Shield
      return (
        <g>
          <path d="M24 5l14 6v12c0 9-6 16-14 20-8-4-14-11-14-20V11z" fill="#cdbf9e" stroke="#7b6a47" strokeWidth="2" strokeLinejoin="round" />
          <path d="M13 16h22M12 24h24M15 32h18" stroke="#9c8a63" strokeWidth="1.6" />
          <path d="M24 8v32" stroke="#efe8d3" strokeWidth="3" strokeLinecap="round" />
        </g>
      );
    default: // Hylian Shield
      return (
        <g>
          <path d="M24 4l16 5v14c0 11-7 18-16 22C15 41 8 34 8 23V9z" fill="#2d4592" stroke="#cdd5e6" strokeWidth="2.4" strokeLinejoin="round" />
          <path d="M24 12l4.4 7.4h-8.8zM19.6 20l4.4 7.4h-8.8zM28.4 20l4.4 7.4h-8.8z" fill="#e3b53b" />
          <path d="M12 31c5 7 9 9 12 10 3-1 7-3 12-10" stroke="#c43a3a" strokeWidth="2" fill="none" strokeLinecap="round" />
        </g>
      );
  }
}

function Bow({ kind }: { kind: number }) {
  const wood = kind === 1 ? '#7a5a3a' : kind === 2 ? '#5c4630' : '#c7925a';
  const width = kind === 2 ? 4.4 : kind === 1 ? 4.2 : 3.6;
  const string = kind === 1 ? '#c9b48a' : '#efe6d2';
  return (
    <g>
      <path d="M13 6C37 11 43 30 33 43" fill="none" stroke={wood} strokeWidth={width} strokeLinecap="round" />
      <path d="M13 6 33 43" stroke={string} strokeWidth="1.2" />
      <path d="M35 19.5l1.5 5" stroke={kind === 2 ? '#e3b53b' : '#4a2f14'} strokeWidth="5" strokeLinecap="round" />
      {kind === 1 && <><circle cx="13" cy="6" r="2.4" fill={BONE} /><circle cx="33" cy="43" r="2.4" fill={BONE} /></>}
      {kind === 2 && <><circle cx="13" cy="6" r="2.4" fill={STEEL} /><circle cx="33" cy="43" r="2.4" fill={STEEL} /></>}
    </g>
  );
}

function Food({ kind }: { kind: number }) {
  switch (kind) {
    case 1: // Hylian Shroom
      return (
        <g>
          <path d="M17 25h14l-2 15c0 2-2 3-5 3s-5-1-5-3z" fill="#efe6d2" stroke="#b8ac8e" strokeWidth="1" />
          <path d="M6 26C6 15 14 7 24 7s18 8 18 19z" fill="#6d9fd1" stroke="#3d6a9a" strokeWidth="1.4" />
          <circle cx="16" cy="17" r="2.6" fill="#e8f1fb" /><circle cx="27" cy="13" r="2.2" fill="#e8f1fb" /><circle cx="33" cy="20" r="2.6" fill="#e8f1fb" /><circle cx="22" cy="21" r="1.7" fill="#e8f1fb" />
        </g>
      );
    case 2: // Raw Meat
    case 3: { // Raw Prime Meat
      const prime = kind === 3;
      return (
        <g>
          <path d="M34 27l8 8" stroke="#efe6d2" strokeWidth="3.6" strokeLinecap="round" />
          <circle cx="43.4" cy="36.6" r="2.6" fill="#efe6d2" /><circle cx="40.6" cy="39.4" r="2.6" fill="#efe6d2" />
          <path d="M8 27C5 18 11 9 23 9c10 0 18 5 18 13 0 10-9 17-18 17-7 0-13-4-15-12z" fill={prime ? '#a8243a' : '#c4465a'} stroke={prime ? '#f2d27a' : '#7d2a3a'} strokeWidth={prime ? 1.8 : 1.2} />
          <path d="M14 18c4 1 8 0 11-3M17 27c5 1 10-1 14-5M25 31c3 0 6-2 8-4" stroke="#f3bcc4" strokeWidth="1.5" fill="none" strokeLinecap="round" />
        </g>
      );
    }
    default: // Apple
      return (
        <g>
          <path d="M24 14c-6-4-15-1-15 10 0 9 6 17 11 17 2 0 3-1 4-1s2 1 4 1c5 0 11-8 11-17 0-11-9-14-15-10z" fill="#d6352c" stroke="#8f1f1a" strokeWidth="1.2" />
          <path d="M15 22c1-4 4-6 7-6" stroke="#f08a80" strokeWidth="2" fill="none" strokeLinecap="round" />
          <path d="M24 14c0-4 1-7 4-9" stroke="#5a3a1a" strokeWidth="2.2" fill="none" strokeLinecap="round" />
          <path d="M26 8c3-3 8-3 10-1-2 3-7 4-10 1z" fill="#4fa04a" />
        </g>
      );
  }
}

/** The icon of item `kind` (its table index) of a category. */
export function ItemIcon({ cat, kind, size = 20 }: { cat: IconCat; kind: number; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden className="shrink-0">
      {cat === 'weapon' ? <Weapon kind={kind} /> : cat === 'shield' ? <Shield kind={kind} /> : cat === 'bow' ? <Bow kind={kind} /> : <Food kind={kind} />}
    </svg>
  );
}
