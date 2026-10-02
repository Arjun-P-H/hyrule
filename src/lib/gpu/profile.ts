/**
 * Per-pass GPU timing for the benches.
 * ---------------------------------------------------------------------------
 * Every pass the renderers begin asks `stamped(label)` for its descriptor's
 * timestamp writes. Nothing is installed in the game (each pass then costs a
 * null check); a bench installs a stamp function (setPassStamp) that hands
 * out query slots under each label, and reads the costs back as differences
 * between consecutive pass ends (bench/gpu/webgpu-page.ts).
 */

type PassStamp = (label: string) => GPURenderPassTimestampWrites | undefined;

let stamp: PassStamp | null = null;

const NONE: { timestampWrites?: GPURenderPassTimestampWrites } = Object.freeze({});

/** Install (or with null remove) the stamp function every pass asks. */
export function setPassStamp(f: PassStamp | null): void {
  stamp = f;
}

/** The timestamp writes of a pass labelled `label`, to spread into its descriptor (none unless a bench installed a stamp). */
export function stamped(label: string): { timestampWrites?: GPURenderPassTimestampWrites } {
  if (!stamp) return NONE;
  const w = stamp(label);
  return w ? { timestampWrites: w } : NONE;
}
