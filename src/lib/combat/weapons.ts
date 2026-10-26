/**
 * The weapons: what the hero (and a Bokoblin) can hold, and how the
 * renderers draw each (its model, rig.ts gear). The hero starts with the
 * Master Sword, the Hylian Shield and a bow; Bokoblins drop their clubs,
 * which the hero can pick up and wear out (TotK durability: a club breaks
 * after so many blows; the sword never does).
 */

/** The models the hero shaders draw in a hand or on the back (rig gear: the weapon's w). */
export const MODEL = { none: 0, sword: 1, club: 2, spiked: 3, bone: 4 } as const;

export type WeaponKind = {
  readonly name: string;
  readonly model: number;
  readonly damage: number;     // hit points a plain swing deals
  readonly durability: number; // blows before it breaks (Infinity: never)
  readonly pace: number;       // swing speed × (heavier clubs swing slower)
  readonly info: string;       // the pouch's description
};

/** Indices into WEAPONS. */
export const WEAPON = { master: 0, club: 1, spiked: 2, bone: 3 } as const;

export const WEAPONS: readonly WeaponKind[] = [
  { name: 'Master Sword', model: MODEL.sword, damage: 12, durability: Infinity, pace: 1, info: 'The blade that seals the darkness. It never wears out.' },
  { name: 'Boko Club', model: MODEL.club, damage: 7, durability: 14, pace: 0.92, info: 'A crude club of wood. It swings fast and breaks soon.' },
  { name: 'Spiked Boko Club', model: MODEL.spiked, damage: 11, durability: 18, pace: 0.9, info: 'A club studded with spikes. It bites harder than plain wood.' },
  { name: 'Dragonbone Boko Club', model: MODEL.bone, damage: 17, durability: 22, pace: 0.86, info: 'A heavy club of bone. The hardest hitting of the clubs, and the slowest.' },
];

/** One weapon carried: its kind and the blows it has left. */
export type Carried = { kind: number; wear: number };

/** The grip-to-tip length of a model (constants.ts HERO_SHAPE), for reach and the renderers' bounds. */
export function modelLength(model: number, sword: number, club: number): number {
  return model === MODEL.sword ? sword : model === MODEL.none ? 0 : club;
}

// ---------------------------------------------------------------------------------------------
// The rest of the inventory: shields, bows and food. The hero starts with entry 0 of the first two
// (the Hylian Shield and the Traveler's Bow, which never wear out, like the Master Sword) and
// three Apples; the others are found, dropped by Bokoblins or won.

/** The shield models the hero shaders draw on the arm or the back (rig gear: the shield's w; 0 hides it). */
export const SHIELD_MODEL = { none: 0, hylian: 1, wooden: 2, traveler: 3, lid: 4, boko: 5 } as const;

/** The bow models the hero shaders draw (rig gear: the bow's w; 0 hides it). */
export const BOW_MODEL = { none: 0, traveler: 1, boko: 2, soldier: 3 } as const;

type ShieldKind = {
  readonly name: string;
  readonly model: number;
  readonly durability: number; // blows it guards before it breaks (Infinity: never)
  readonly push: number;       // × the push a guarded blow gives the hero (lower holds firmer)
  readonly info: string;
};

type BowKind = {
  readonly name: string;
  readonly model: number;
  readonly damage: number;     // × an arrow's damage
  readonly durability: number; // arrows it looses before it breaks (Infinity: never)
  readonly info: string;
};

type FoodKind = {
  readonly name: string;
  readonly heal: number;       // quarters of a heart it restores
  readonly info: string;
};

/** Indices into SHIELDS, BOWS and FOODS. */
export const SHIELD = { hylian: 0, wooden: 1, traveler: 2, lid: 3, boko: 4 } as const;
export const BOW = { traveler: 0, boko: 1, soldier: 2 } as const;
export const FOOD = { apple: 0, shroom: 1, meat: 2, prime: 3 } as const;

export const SHIELDS: readonly ShieldKind[] = [
  { name: 'Hylian Shield', model: SHIELD_MODEL.hylian, durability: Infinity, push: 0.7, info: 'The hero\'s own shield. It never wears out and holds firm against a blow.' },
  { name: 'Wooden Shield', model: SHIELD_MODEL.wooden, durability: 16, push: 1, info: 'Planks bound with iron. Light, but it splinters after a few blows.' },
  { name: 'Traveler\'s Shield', model: SHIELD_MODEL.traveler, durability: 24, push: 0.85, info: 'A round shield made for a long road. Sturdy enough.' },
  { name: 'Pot Lid', model: SHIELD_MODEL.lid, durability: 10, push: 1.4, info: 'Anything will do in a pinch. It dents fast, and a blow shoves you back.' },
  { name: 'Boko Shield', model: SHIELD_MODEL.boko, durability: 14, push: 1.15, info: 'A Bokoblin\'s shield of bone and hide.' },
];

export const BOWS: readonly BowKind[] = [
  { name: 'Traveler\'s Bow', model: BOW_MODEL.traveler, damage: 1, durability: Infinity, info: 'The hero\'s own bow. It never wears out.' },
  { name: 'Boko Bow', model: BOW_MODEL.boko, damage: 0.7, durability: 18, info: 'A Bokoblin\'s bow. Weak, and soon spent.' },
  { name: 'Soldier\'s Bow', model: BOW_MODEL.soldier, damage: 1.4, durability: 30, info: 'A soldier\'s bow. It shoots harder, and wears out.' },
];

export const FOODS: readonly FoodKind[] = [
  { name: 'Apple', heal: 2, info: 'Crisp and sweet, fresh off the branch.' },
  { name: 'Hylian Shroom', heal: 2, info: 'A mushroom of the forest floor. Earthy, but it keeps you going.' },
  { name: 'Raw Meat', heal: 4, info: 'Raw and a little gamey, but it fills the belly.' },
  { name: 'Raw Prime Meat', heal: 8, info: 'A fine cut, even raw.' },
];

/** What an event's `a` holds for an item: its category's base plus its index (a weapon: its WEAPONS index itself). */
export const ITEM_CODE = { weapon: 0, shield: 100, bow: 200, food: 300 } as const;

/** The name of the item an event's code (ITEM_CODE + index) names. */
export function itemName(code: number): string {
  return code >= ITEM_CODE.food ? FOODS[code - ITEM_CODE.food].name
    : code >= ITEM_CODE.bow ? BOWS[code - ITEM_CODE.bow].name
      : code >= ITEM_CODE.shield ? SHIELDS[code - ITEM_CODE.shield].name
        : WEAPONS[code].name;
}
