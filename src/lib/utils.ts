import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Exhaustiveness guard: a `switch` over a union that misses a case fails to compile here. */
export function assertNever(value: never): never {
  throw new Error(`Unhandled variant: ${String(value)}`);
}

/** Thin scrollbar utility classes. */
export const THIN_SCROLL =
  '[scrollbar-width:thin] [scrollbar-color:#44403c_transparent] ' +
  '[&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent ' +
  '[&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-stone-700';

/** Parse and normalize user string input into a valid uint32 seed. */
export function parseSeedInput(v: string): number | null {
  const n = Math.floor(Number(v));
  if (v.trim() !== '' && Number.isFinite(n)) {
    return ((n % 4294967296) + 4294967296) % 4294967296;
  }
  return null;
}

/** Generate a random positive integer seed. */
export function generateRandomSeed(): number {
  return 1 + Math.floor(Math.random() * 999_999);
}

/** Linear RGB → sRGB hex color. */
export function linearToHex(rgb: readonly number[]): string {
  const enc = (c: number) => {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  };
  return `#${rgb.slice(0, 3).map(enc).map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}
