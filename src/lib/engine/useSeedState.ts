import { useState, useCallback } from 'react';
import { DEFAULT_SEED } from '@/lib/constants';
import { generateRandomSeed, parseSeedInput } from '@/lib/utils';

/**
 * Unified custom hook for terrain seed input, parsing, and randomization.
 * Consolidates duplicated seed state across the spec console and fullscreen game.
 */
export function useSeedState(initialSeed: number = DEFAULT_SEED) {
  const [seedText, setSeedText] = useState<string>(String(initialSeed));
  const [seed, setSeed] = useState<number>(initialSeed);

  const onSeedChange = useCallback((v: string) => {
    setSeedText(v);
    const parsed = parseSeedInput(v);
    if (parsed !== null) {
      setSeed(parsed);
    }
  }, []);

  const randomizeSeed = useCallback(() => {
    const n = generateRandomSeed();
    setSeedText(String(n));
    setSeed(n);
  }, []);

  return {
    seed,
    seedText,
    onSeedChange,
    randomizeSeed,
    setSeed,
    setSeedText,
  };
}
