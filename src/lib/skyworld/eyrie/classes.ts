/**
 * The Eyrie classes (SKY_CIVILIZATION.md §7): what each is for, how rare it
 * is, how high it floats, which palette it builds in, whether its islands are
 * joined by built bridges or by natural arches — and its name.
 */

import { PALETTE } from '../kit/palettes';

export const EYRIE_CLASSES = ['crown', 'mirror', 'halo', 'beacon', 'petal', 'bough', 'ember', 'rime', 'fallen', 'harbour', 'wells'] as const;
export type EyrieClass = (typeof EYRIE_CLASSES)[number];

type ClassInfo = {
  /** Its epithet (the class's name for players). */
  title: string;
  /** Share of Eyries of this class (weights, normalised over those whose height fits). */
  weight: number;
  /** Base height of its main island's top (m). */
  top: readonly [number, number];
  palette: number;
  /** Built bridges between its islands, or (wild and fallen places) natural arches. */
  bridges: 'built' | 'natural' | 'none';
};

export const CLASS_INFO: Record<EyrieClass, ClassInfo> = {
  crown: { title: 'the Crown of Wings', weight: 0.15, top: [2100, 2450], palette: PALETTE.crown, bridges: 'built' },
  mirror: { title: 'the Mirror of the Accord', weight: 0.1, top: [1950, 2350], palette: PALETTE.mirror, bridges: 'built' },
  halo: { title: 'the Singing Ring', weight: 0.08, top: [2050, 2450], palette: PALETTE.halo, bridges: 'built' },
  beacon: { title: 'the Lanternpeak', weight: 0.08, top: [2250, 2600], palette: PALETTE.beacon, bridges: 'built' },
  petal: { title: 'the Falling Petal', weight: 0.12, top: [1950, 2350], palette: PALETTE.petal, bridges: 'built' },
  bough: { title: 'the First Nest', weight: 0.06, top: [1900, 2300], palette: PALETTE.bough, bridges: 'natural' },
  ember: { title: 'the Ember Anvil', weight: 0.07, top: [1800, 2200], palette: PALETTE.ember, bridges: 'built' },
  rime: { title: 'the Ice Watch', weight: 0.07, top: [2620, 2820], palette: PALETTE.rime, bridges: 'built' },
  fallen: { title: 'the Fallen Eyrie', weight: 0.1, top: [2000, 2400], palette: PALETTE.fallen, bridges: 'none' },
  harbour: { title: 'the Windmarket', weight: 0.1, top: [2000, 2350], palette: PALETTE.harbour, bridges: 'built' },
  wells: { title: 'the Breath Wells', weight: 0.07, top: [1640, 1860], palette: PALETTE.wells, bridges: 'natural' },
};

/** The class of an Eyrie from a [0, 1) draw, by weight. */
export function pickClass(u: number): EyrieClass {
  let total = 0;
  for (const c of EYRIE_CLASSES) total += CLASS_INFO[c].weight;
  let acc = 0;
  for (const c of EYRIE_CLASSES) {
    acc += CLASS_INFO[c].weight / total;
    if (u < acc) return c;
  }
  return 'crown';
}

// Aelari names: soft vowels, trills, a hyphen between two words of a few syllables (§4.5).
const ONSET = ['s', 'v', 'r', 'th', 'k', 'l', 'y', 'd', 'm', 'n', 'ae', 'i', 'o', 'u', 'sh', 'vh', 'rh', 'ess', 'al', 'or'];
const NUCLEUS = ['ae', 'a', 'e', 'i', 'o', 'u', 'ia', 'ei', 'ou', 'y'];
const CODA = ['', '', 'l', 'r', 'n', 's', 'th', 'sh', 'rr', 'n', 'ra', 'la', 'na', 'ri', 'th'];

/** An Eyrie's proper name from a stream of [0, 1) numbers (e.g. "Sael-Vhara"). */
export function eyrieName(rnd: () => number): string {
  const word = (syl: number) => {
    let w = '';
    for (let k = 0; k < syl; k++) {
      const pick = <T,>(a: readonly T[]) => a[Math.floor(rnd() * a.length)];
      w += (k === 0 || rnd() < 0.6 ? pick(ONSET) : '') + pick(NUCLEUS) + (k === syl - 1 || rnd() < 0.3 ? pick(CODA) : '');
    }
    return w.charAt(0).toUpperCase() + w.slice(1);
  };
  const a = word(1 + Math.floor(rnd() * 2));
  return rnd() < 0.55 ? `${a}-${word(1 + Math.floor(rnd() * 2))}` : `${a} ${word(1 + Math.floor(rnd() * 2))}`;
}
