/**
 * The heroes a player can be: one body, five dresses. `dress` is the tunic's
 * linear albedo (the cap is drawn a shade darker), read by both renderers
 * through the packed hero uniforms (rig.ts HERO_UNIFORM.dress); `swatch` is
 * its sRGB look in sunlight, for the lobby and the player list.
 */
type Character = {
  id: number;
  name: string;
  dress: readonly [number, number, number];
  swatch: string;
};

export const CHARACTERS: readonly Character[] = [
  { id: 0, name: 'Verdant', dress: [0.028, 0.115, 0.022], swatch: '#3e8a33' },
  { id: 1, name: 'Crimson', dress: [0.3, 0.022, 0.018], swatch: '#c4362c' },
  { id: 2, name: 'Azure', dress: [0.018, 0.06, 0.3], swatch: '#2f62d6' },
  { id: 3, name: 'Violet', dress: [0.13, 0.026, 0.22], swatch: '#8a3fb5' },
  { id: 4, name: 'Amber', dress: [0.4, 0.2, 0.012], swatch: '#e09a1c' },
];

/** The classic green tunic: the dress when no character was chosen. */
export const DEFAULT_DRESS = CHARACTERS[0].dress;

export function characterOf(id: number): Character {
  return CHARACTERS[id] ?? CHARACTERS[0];
}
