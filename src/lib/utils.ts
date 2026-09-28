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

/** OKLab → sRGB hex color conversion. */
export function oklabToHex(L: number, a: number, b: number): string {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  const enc = (c: number) => {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  };
  return `#${lin.map(enc).map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}
