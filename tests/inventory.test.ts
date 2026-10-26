import { describe, it, expect } from 'bun:test';
import { COMBAT, DEFAULT_SEED, SPAWN } from '../src/lib/constants';
import { elevation, slope, waterLevel } from '../src/lib/terrainMath';
import { createHeroState } from '../src/lib/hero/locomotion';
import { CombatWorld, createCombatInput, type CombatInput } from '../src/lib/combat/world';
import { MOVE, MOVES } from '../src/lib/combat/moves';
import { FOE_KINDS, type Foe } from '../src/lib/combat/foes';
import { readPouch } from '../src/lib/combat/pouch';
import { BOW, BOWS, BOW_MODEL, FOOD, FOODS, ITEM_CODE, SHIELD, SHIELDS, SHIELD_MODEL, WEAPON, WEAPONS, itemName } from '../src/lib/combat/weapons';

const SEED = DEFAULT_SEED;
const DT = 1 / 60;

function findSpot(pred: (x: number, z: number) => boolean, step = 3, radius = 2500): [number, number] {
  for (let r = 0; r <= radius; r += step) {
    for (let a = 0; a < 64; a++) {
      const x = SPAWN.x + r * Math.cos((a / 64) * Math.PI * 2);
      const z = SPAWN.z + r * Math.sin((a / 64) * Math.PI * 2);
      if (pred(x, z)) return [x, z];
      if (r === 0) break;
    }
  }
  throw new Error('no spot found');
}

// level, dry, open ground (as tests/combat.test.ts's)
const FLAT = findSpot((x, z) => slope(x, z, SEED) < 0.06 && elevation(x, z, SEED, 0) > waterLevel(x, z, SEED) + 2.9
  && [0, 1, 2, 3, 4, 5, 6, 7].every((k) => slope(x + 12 * Math.cos(k * Math.PI / 4), z + 12 * Math.sin(k * Math.PI / 4), SEED) < 0.08), 4);

// a gentle slope down toward +x all along the glide, dry and open: where a surfer pushed off toward +x keeps gliding for the
// four seconds the shield's wear takes (on dead flat ground, or one that rises again, the friction stops it just short of them)
const SLOPED = findSpot((x, z) => {
  for (let k = 0; k < 8; k++) {
    const down = (elevation(x + 5 * k, z, SEED, 0) - elevation(x + 5 * k + 5, z, SEED, 0)) / 5;
    if (down < 0.045 || down > 0.12 || elevation(x + 5 * k, z, SEED, 0) < waterLevel(x + 5 * k, z, SEED) + 2.9) return false;
  }
  return slope(x, z, SEED) < 0.14 && [0, 1, 2, 3, 4, 5, 6, 7].every((k) => slope(x + 12 * Math.cos(k * Math.PI / 4), z + 12 * Math.sin(k * Math.PI / 4), SEED) < 0.16);
}, 4);

/** A world at FLAT (or `at`), the hero facing +x, no camps. */
function arena(yaw = Math.PI / 2, at: [number, number] = FLAT) {
  const hero = createHeroState(at[0], at[1], yaw, SEED);
  const w = new CombatWorld(hero, SEED);
  w.campsOn = false;
  const inp = createCombatInput();
  look(inp, yaw, -0.05, hero.x, hero.y, hero.z);
  return { w, hero, inp };
}

function look(inp: CombatInput, yaw: number, pitch: number, x: number, y: number, z: number): void {
  inp.camYaw = yaw;
  inp.camPitch = pitch;
  inp.eyeX = x - Math.sin(yaw) * 4; inp.eyeY = y + 1.8; inp.eyeZ = z - Math.cos(yaw) * 4;
  inp.dirX = Math.sin(yaw) * Math.cos(pitch); inp.dirY = Math.sin(pitch); inp.dirZ = Math.cos(yaw) * Math.cos(pitch);
}

function run(w: CombatWorld, inp: CombatInput, seconds: number, each?: (t: number) => void): void {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) {
    each?.(i * DT);
    w.step(DT, inp);
  }
}

const events = (w: CombatWorld) => w.events.slice(0, w.eventCount).map((e) => e.type);

function foeAt(w: CombatWorld, kind: number, d: number, yaw: number): Foe {
  const s = w.hero;
  const x = s.x + Math.sin(yaw) * d, z = s.z + Math.cos(yaw) * d;
  const f = w.spawnFoe(kind, x, z, Math.atan2(s.x - x, s.z - z))!;
  expect(f).not.toBeNull();
  return f;
}

/** A foe ahead in the middle of a swing, its blow due in a few ticks. */
function swingingFoe(w: CombatWorld, d: number, yaw: number, kind = 0): Foe {
  const f = foeAt(w, kind, d, yaw);
  f.roused = true; f.aware = 1;
  f.mind = 'attack'; f.t = 0;
  f.move = MOVE.smash;
  f.u = MOVES[MOVE.smash].hit[0] - 0.03;
  f.struck = false;
  return f;
}

/** Lay an item at the hero's feet. */
function lay(w: CombatWorld, kind: 'weapon' | 'shield' | 'bow' | 'food', weapon: number, wear: number, slot = 0): void {
  const p = w.pickups[slot];
  p.live = true; p.kind = kind; p.weapon = weapon; p.wear = wear; p.count = 1; p.age = 0;
  p.x = w.hero.x + 0.5; p.z = w.hero.z; p.y = w.hero.y; p.vy = 0;
}

describe('inventory tables', () => {
  it('starts with the Hylian Shield and the Traveler\'s Bow, neither of which wears out', () => {
    expect(SHIELD.hylian).toBe(0);
    expect(BOW.traveler).toBe(0);
    expect(SHIELDS[SHIELD.hylian].durability).toBe(Infinity);
    expect(BOWS[BOW.traveler].durability).toBe(Infinity);
  });

  it('gives every shield and bow its own model, none of them 0 (which hides the gear)', () => {
    const shields = SHIELDS.map((s) => s.model);
    const bows = BOWS.map((b) => b.model);
    expect(new Set(shields).size).toBe(shields.length);
    expect(new Set(bows).size).toBe(bows.length);
    expect(shields.every((m) => m >= 1)).toBe(true);
    expect(bows.every((m) => m >= 1)).toBe(true);
    expect(shields.sort()).toEqual(Object.values(SHIELD_MODEL).filter((m) => m >= 1).sort());
    expect(bows.sort()).toEqual(Object.values(BOW_MODEL).filter((m) => m >= 1).sort());
  });

  it('names its entries by the index constants', () => {
    expect(SHIELDS[SHIELD.wooden].name).toBe('Wooden Shield');
    expect(SHIELDS[SHIELD.lid].name).toBe('Pot Lid');
    expect(SHIELDS[SHIELD.boko].name).toBe('Boko Shield');
    expect(BOWS[BOW.boko].name).toBe('Boko Bow');
    expect(BOWS[BOW.soldier].name).toBe('Soldier\'s Bow');
    expect(FOODS[FOOD.apple].name).toBe('Apple');
    expect(FOODS[FOOD.prime].name).toBe('Raw Prime Meat');
    expect(SHIELDS.length).toBe(Object.keys(SHIELD).length);
    expect(BOWS.length).toBe(Object.keys(BOW).length);
    expect(FOODS.length).toBe(Object.keys(FOOD).length);
  });

  it('heals food in whole quarters, none of it more than the hero\'s hearts hold', () => {
    for (const f of FOODS) {
      expect(Number.isInteger(f.heal)).toBe(true);
      expect(f.heal).toBeGreaterThan(0);
      expect(f.heal).toBeLessThanOrEqual(COMBAT.hearts * 4);
    }
  });

  it('wears out everything but the hero\'s own gear, and a bow shoots no harder than a soldier\'s', () => {
    for (const s of SHIELDS.slice(1)) expect(Number.isFinite(s.durability)).toBe(true);
    for (const b of BOWS.slice(1)) expect(Number.isFinite(b.durability)).toBe(true);
    expect(Math.max(...BOWS.map((b) => b.damage))).toBe(BOWS[BOW.soldier].damage);
  });
});

describe('the pouch — a start', () => {
  it('begins with the Master Sword, the Hylian Shield and the Traveler\'s Bow in hand, 30 arrows and three Apples', () => {
    const { w } = arena();
    expect(w.carried()?.kind).toBe(WEAPON.master);
    expect(w.shieldOn).toBe(0);
    expect(w.shields[0].kind).toBe(SHIELD.hylian);
    expect(w.bowOn).toBe(0);
    expect(w.bows[0].kind).toBe(BOW.traveler);
    expect(w.arrows).toBe(COMBAT.arrows);
    expect(w.food[FOOD.apple]).toBe(COMBAT.startApples);
    expect(w.food.filter((n, i) => i !== FOOD.apple && n > 0).length).toBe(0);
  });

  it('reads out as the inventory screen lists it', () => {
    const { w } = arena();
    const s = readPouch(w);
    expect(s.weapons.map((i) => i.name)).toEqual(['Master Sword']);
    expect(s.weapons[0]).toMatchObject({ equipped: true, keeps: true, wear: -1, max: -1 });
    expect(s.shields[0]).toMatchObject({ name: 'Hylian Shield', equipped: true });
    expect(s.bows[0]).toMatchObject({ name: 'Traveler\'s Bow', equipped: true });
    expect(s.food).toEqual([{ kind: FOOD.apple, name: 'Apple', info: FOODS[FOOD.apple].info, count: COMBAT.startApples, heal: FOODS[FOOD.apple].heal }]);
    expect(s.health).toBe(COMBAT.hearts * 4);
    expect(s.slots).toEqual({ weapon: COMBAT.weaponSlots, bow: COMBAT.bowSlots, shield: COMBAT.shieldSlots });
  });

  it('names an item by its event code', () => {
    expect(itemName(WEAPON.club)).toBe(WEAPONS[WEAPON.club].name);
    expect(itemName(ITEM_CODE.shield + SHIELD.lid)).toBe('Pot Lid');
    expect(itemName(ITEM_CODE.bow + BOW.soldier)).toBe('Soldier\'s Bow');
    expect(itemName(ITEM_CODE.food + FOOD.prime)).toBe('Raw Prime Meat');
  });
});

describe('the pouch — eating', () => {
  it('eats an Apple: its hearts come back, one fewer is kept', () => {
    const { w } = arena();
    w.health = 8;
    w.eventCount = 0;
    expect(w.eat(FOOD.apple)).toBe(true);
    expect(w.health).toBe(8 + FOODS[FOOD.apple].heal);
    expect(w.food[FOOD.apple]).toBe(COMBAT.startApples - 1);
    expect(events(w)).toEqual(['eat']);
    expect(w.events[0].a).toBe(ITEM_CODE.food + FOOD.apple);
  });

  it('never heals past the hearts, and is refused (nothing eaten) with the hearts full or no food kept', () => {
    const { w } = arena();
    expect(w.eat(FOOD.apple)).toBe(false);
    expect(w.food[FOOD.apple]).toBe(COMBAT.startApples);
    w.health = w.hearts * 4 - 1;
    expect(w.eat(FOOD.apple)).toBe(true);
    expect(w.health).toBe(w.hearts * 4);
    w.health = 4;
    expect(w.eat(FOOD.meat)).toBe(false);
    expect(w.eat(99)).toBe(false);
    expect(w.eat(-1)).toBe(false);
    w.food[FOOD.prime] = 2;
    expect(w.eat(FOOD.prime)).toBe(true);
    expect(w.health).toBe(4 + FOODS[FOOD.prime].heal);
    expect(w.food[FOOD.prime]).toBe(1);
  });

  it('cannot eat once down', () => {
    const { w, inp } = arena();
    w.health = 1;
    swingingFoe(w, 1.8, -Math.PI / 2, 2);
    run(w, inp, 0.5);
    expect(w.act).toBe('dead');
    expect(w.eat(FOOD.apple)).toBe(false);
  });
});

describe('the pouch — equipping', () => {
  it('equips a weapon from the pouch, and puts it away: the hands are empty', () => {
    const { w, inp } = arena();
    lay(w, 'weapon', WEAPON.club, WEAPONS[WEAPON.club].durability);
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(w.carried()?.kind).toBe(WEAPON.club);
    expect(w.equip('weapon', 0)).toBe(true);
    expect(w.carried()?.kind).toBe(WEAPON.master);
    expect(w.unequip('weapon')).toBe(true);
    expect(w.carried()).toBeNull();
    expect(w.unequip('weapon')).toBe(false);
    expect(w.equip('weapon', 7)).toBe(false);
    w.refreshPose(inp);
    expect(w.pose.weapon).toBe(0);
  });

  it('picks up a shield and a bow with the action key; they join the pouch, and go on only if none was', () => {
    const { w, inp } = arena();
    lay(w, 'shield', SHIELD.wooden, SHIELDS[SHIELD.wooden].durability);
    w.eventCount = 0;
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(w.shields.map((c) => c.kind)).toEqual([SHIELD.hylian, SHIELD.wooden]);
    expect(w.shieldOn).toBe(0);
    expect(events(w)).toContain('pickup');
    expect(w.events.find((e) => e.type === 'pickup')!.a).toBe(ITEM_CODE.shield + SHIELD.wooden);
    expect(w.pickups[0].live).toBe(false);
    lay(w, 'bow', BOW.soldier, BOWS[BOW.soldier].durability);
    w.step(DT, inp);
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(w.bows.map((c) => c.kind)).toEqual([BOW.traveler, BOW.soldier]);
    expect(w.bowOn).toBe(0);
    // the pouch changes what is in hand
    expect(w.equip('shield', 1)).toBe(true);
    expect(w.shieldOn).toBe(1);
    expect(w.equip('bow', 1)).toBe(true);
    expect(w.bowOn).toBe(1);
    w.step(DT, inp);
    expect(w.pose.shield).toBe(true);
    expect(w.pose.shieldModel).toBe(SHIELD_MODEL.wooden);
    expect(w.pose.bowModel).toBe(BOW_MODEL.soldier);
  });

  it('turns down a shield with every slot full, and says so', () => {
    const { w, inp } = arena();
    while (w.shields.length < COMBAT.shieldSlots) w.shields.push({ kind: SHIELD.lid, wear: 5 });
    lay(w, 'shield', SHIELD.boko, 14);
    expect(w.canTake(0)).toBe(false);
    w.eventCount = 0;
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(w.pickups[0].live).toBe(true);
    expect(w.shields.length).toBe(COMBAT.shieldSlots);
    const e = w.events.find((x) => x.type === 'empty');
    expect(e?.a).toBe(3);
  });

  it('drops an item in front of the hero, to be picked up again; the hero\'s own gear is kept', () => {
    const { w, hero, inp } = arena();
    w.weapons.push({ kind: WEAPON.spiked, wear: 9 });
    expect(w.equip('weapon', 1)).toBe(true);
    expect(w.discard('weapon', 0)).toBe(false); // (the Master Sword)
    expect(w.discard('shield', 0)).toBe(false);
    expect(w.discard('bow', 0)).toBe(false);
    expect(w.discard('weapon', 1)).toBe(true);
    expect(w.weapons.length).toBe(1);
    expect(w.carried()?.kind).toBe(WEAPON.master); // (the sword takes the dropped one's place)
    const lying = w.pickups.find((p) => p.live && p.kind === 'weapon' && p.weapon === WEAPON.spiked)!;
    expect(lying.wear).toBe(9);
    expect(Math.hypot(lying.x - hero.x, lying.z - hero.z)).toBeLessThan(1.2);
    run(w, inp, 1.5);
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(w.weapons.some((c) => c.kind === WEAPON.spiked && c.wear === 9)).toBe(true);
    expect(w.discard('weapon', 5)).toBe(false);
  });

  it('keeps what is in hand straight as items leave the lists before it', () => {
    const { w } = arena();
    w.shields.push({ kind: SHIELD.wooden, wear: 8 }, { kind: SHIELD.lid, wear: 4 }, { kind: SHIELD.boko, wear: 6 });
    expect(w.equip('shield', 3)).toBe(true);
    expect(w.discard('shield', 1)).toBe(true);
    expect(w.shields[w.shieldOn].kind).toBe(SHIELD.boko); // (the one in hand moved up the list)
    expect(w.discard('shield', w.shieldOn)).toBe(true);
    expect(w.shields[w.shieldOn].kind).toBe(SHIELD.hylian); // (back to the hero's own)
  });
});

describe('the pouch — no shield, no bow', () => {
  it('raises no guard without a shield (a dud is heard); with one back it guards again', () => {
    const { w, inp } = arena();
    expect(w.unequip('shield')).toBe(true);
    w.eventCount = 0;
    inp.guard = true;
    w.step(DT, inp);
    expect(w.act).toBe('idle');
    expect(w.events.find((e) => e.type === 'empty')?.a).toBe(1);
    run(w, inp, 0.3);
    expect(w.act).toBe('idle');
    expect(w.pose.shield).toBe(false);
    inp.guard = false;
    w.step(DT, inp);
    expect(w.equip('shield', 0)).toBe(true);
    inp.guard = true;
    run(w, inp, 0.1);
    expect(w.act).toBe('guard');
  });

  it('draws no bow without one (a dud is heard), and lowers a bow that is put away mid-draw', () => {
    const { w, inp } = arena();
    expect(w.unequip('bow')).toBe(true);
    w.eventCount = 0;
    inp.aim = true;
    w.step(DT, inp);
    expect(w.act).toBe('idle');
    expect(w.events.find((e) => e.type === 'empty')?.a).toBe(2);
    inp.aim = false;
    w.step(DT, inp);
    expect(w.equip('bow', 0)).toBe(true);
    inp.aim = true;
    run(w, inp, 0.2);
    expect(w.act).toBe('aim');
    expect(w.unequip('bow')).toBe(true);
    run(w, inp, 0.1);
    expect(w.act).toBe('idle');
    expect(w.pose.bowModel).toBe(0);
  });

  it('a blow from the front lands when the shield is put away', () => {
    const { w, inp } = arena();
    w.unequip('shield');
    inp.guard = true;
    run(w, inp, 0.3);
    const f = swingingFoe(w, 1.8, Math.PI / 2);
    run(w, inp, 0.15);
    expect(f.mind).not.toBe('stun');
    expect(w.health).toBe(w.hearts * 4 - FOE_KINDS[0].damage);
  });
});

describe('the pouch — wear', () => {
  it('a shield wears with each blow it guards and breaks; the Hylian Shield is back on the arm', () => {
    const { w, inp } = arena();
    w.shields.push({ kind: SHIELD.lid, wear: 2 });
    w.equip('shield', 1);
    inp.guard = true;
    run(w, inp, 0.5);
    expect(w.act).toBe('guard');
    const seen: string[] = [];
    let broke = -1;
    for (let k = 0; k < 3; k++) {
      swingingFoe(w, 1.8, Math.PI / 2);
      run(w, inp, 0.2, () => { seen.push(...events(w)); if (events(w).includes('break')) broke = w.events.find((e) => e.type === 'break')!.a; });
      run(w, inp, 1.2);
      if (w.shields.length === 1) break;
    }
    expect(w.health).toBe(w.hearts * 4);
    expect(seen).toContain('block');
    expect(broke).toBe(ITEM_CODE.shield + SHIELD.lid);
    expect(w.shields.map((c) => c.kind)).toEqual([SHIELD.hylian]);
    expect(w.shieldOn).toBe(0);
  });

  it('a Perfect Guard costs the shield nothing', () => {
    const { w, inp } = arena();
    w.shields.push({ kind: SHIELD.wooden, wear: 3 });
    w.equip('shield', 1);
    swingingFoe(w, 1.8, Math.PI / 2);
    inp.guard = true;
    run(w, inp, 0.15);
    expect(w.shields[1].wear).toBe(3);
  });

  it('a firmer shield holds the hero steadier against a blow', () => {
    const slid = (kind: number) => {
      const { w, hero, inp } = arena();
      w.shields.push({ kind, wear: 50 });
      w.equip('shield', 1);
      inp.guard = true;
      run(w, inp, 0.5);
      const x0 = hero.x, z0 = hero.z;
      swingingFoe(w, 1.8, Math.PI / 2);
      let far = 0;
      run(w, inp, 0.3, () => { far = Math.max(far, Math.hypot(hero.x - x0, hero.z - z0)); });
      return far;
    };
    expect(slid(SHIELD.lid)).toBeGreaterThan(slid(SHIELD.traveler));
  });

  it('surfing wears the shield underfoot', () => {
    const { w, hero, inp } = arena(Math.PI / 2, SLOPED);
    w.shields.push({ kind: SHIELD.wooden, wear: 4 });
    w.equip('shield', 1);
    inp.guard = true;
    run(w, inp, 0.4);
    inp.action = true;
    w.step(DT, inp);
    inp.action = false;
    expect(hero.surfing).toBe(true);
    run(w, inp, COMBAT.surfWear + 0.3);
    expect(w.shields[1].wear).toBeLessThan(4);
  });

  it('a bow wears with each arrow loosed, shoots as hard as it is made to, and breaks', () => {
    const { w, inp } = arena();
    w.bows.push({ kind: BOW.soldier, wear: 2 });
    w.equip('bow', 1);
    w.arrows = 10;
    inp.aim = true;
    run(w, inp, COMBAT.nockTime + COMBAT.drawTime + 0.1);
    expect(w.draw).toBe(1);
    inp.attack = true;
    w.step(DT, inp);
    inp.attack = false;
    const shot = w.shots.find((p) => p.live && p.arrow)!;
    expect(shot.damage).toBeCloseTo(COMBAT.arrowDamage * BOWS[BOW.soldier].damage, 6);
    expect(w.bows[1].wear).toBe(1);
    expect(w.arrows).toBe(9);
    run(w, inp, COMBAT.nockTime + COMBAT.drawTime + 0.1);
    const seen: string[] = [];
    inp.attack = true;
    w.step(DT, inp);
    seen.push(...events(w));
    inp.attack = false;
    expect(seen).toContain('break');
    expect(w.events.find((e) => e.type === 'break')!.a).toBe(ITEM_CODE.bow + BOW.soldier);
    expect(w.bows.map((c) => c.kind)).toEqual([BOW.traveler]);
    expect(w.bowOn).toBe(0);
    expect(w.arrows).toBe(8);
  });
});

describe('the pouch — food and gear from the world', () => {
  it('collects food lying near, and stores it; full stacks leave it lying', () => {
    const { w, inp } = arena();
    lay(w, 'food', FOOD.meat, 0);
    w.eventCount = 0;
    w.step(DT, inp);
    expect(w.food[FOOD.meat]).toBe(1);
    expect(w.pickups[0].live).toBe(false);
    expect(w.events.find((e) => e.type === 'pickup')!.a).toBe(ITEM_CODE.food + FOOD.meat);
    w.food[FOOD.shroom] = COMBAT.maxFood;
    lay(w, 'food', FOOD.shroom, 0);
    w.step(DT, inp);
    expect(w.food[FOOD.shroom]).toBe(COMBAT.maxFood);
    expect(w.pickups[0].live).toBe(true);
  });

  it('food, shields and bows lying stay a while and are not taken without the action key', () => {
    const { w, inp } = arena();
    lay(w, 'shield', SHIELD.lid, 10);
    lay(w, 'bow', BOW.boko, 18, 1);
    run(w, inp, 1);
    expect(w.pickups[0].live && w.pickups[1].live).toBe(true);
    expect(w.nearestPickup()).toBeGreaterThanOrEqual(0);
    run(w, inp, 100);
    expect(w.pickups[0].live && w.pickups[1].live).toBe(true);
    run(w, inp, 150);
    expect(w.pickups[0].live || w.pickups[1].live).toBe(false);
  });

  it('beaten foes drop some of it: a shield now and then, an archer\'s bow, meals', () => {
    const food = new Set<number>(), shields = new Set<number>(), bows = new Set<number>();
    for (let seed = 0; seed < 90; seed++) {
      const { w, inp } = arena();
      w.swingId = seed * 11;
      const kind = seed % 3;
      const f = foeAt(w, kind, 6, Math.PI / 2);
      f.live = true; f.mind = 'dead'; f.t = 5;
      if (seed % 2) f.archer = true;
      w.time = seed * 0.37;
      w.step(DT, inp);
      for (const p of w.pickups) {
        if (!p.live) continue;
        if (p.kind === 'food') food.add(p.weapon);
        if (p.kind === 'shield') { shields.add(p.weapon); expect(p.wear).toBe(SHIELDS[p.weapon].durability); }
        if (p.kind === 'bow') { bows.add(p.weapon); expect(p.wear).toBe(BOWS[p.weapon].durability); }
      }
    }
    expect(food.size).toBeGreaterThanOrEqual(2);
    expect(shields.size).toBeGreaterThanOrEqual(1);
    expect(bows.size).toBeGreaterThanOrEqual(1);
    for (const k of FOE_KINDS) { expect(FOODS[k.food[0]]).toBeDefined(); expect(SHIELDS[k.shield]).toBeDefined(); expect(BOWS[k.bow]).toBeDefined(); }
  });

  it('a camp cleared leaves its spoils by the fire', () => {
    const { w, inp } = arena();
    const f = foeAt(w, 0, 5, Math.PI / 2);
    f.camp = 4242;
    w.camps.set(4242, { camp: { key: 4242, x: f.body.x, y: f.body.y, z: f.body.z, kinds: [0], archer: -1 }, awake: true, cleared: false, clearedAt: 0, slots: [f.slot] });
    f.mind = 'dead'; f.t = 5;
    w.step(DT, inp);
    const spoils = w.pickups.filter((p) => p.live && (p.kind === 'shield' || p.kind === 'bow'));
    expect(spoils.length).toBeGreaterThanOrEqual(1);
    expect(w.camps.get(4242)!.cleared).toBe(true);
  });
});

describe('the pouch — a kit to try it with', () => {
  it('stock() fills the pouch with one of everything and puts the hero at three hearts, with food to eat', () => {
    const { w } = arena();
    w.stock();
    expect(w.weapons.map((c) => c.kind).sort()).toEqual(WEAPONS.map((_, k) => k));
    expect(w.shields.map((c) => c.kind).sort()).toEqual(SHIELDS.map((_, k) => k));
    expect(w.bows.map((c) => c.kind).sort()).toEqual(BOWS.map((_, k) => k));
    expect(w.food.every((n) => n >= 5)).toBe(true);
    expect(w.health).toBe(12);
    // (the hero's own are still the ones in hand, and nothing is added twice)
    expect([w.carried()?.kind, w.shieldOn, w.bowOn]).toEqual([WEAPON.master, 0, 0]);
    w.stock();
    expect(w.weapons.length).toBe(WEAPONS.length);
    expect(w.shields.length).toBe(SHIELDS.length);
    expect(w.shields.length).toBeLessThanOrEqual(COMBAT.shieldSlots);
    expect(w.bows.length).toBeLessThanOrEqual(COMBAT.bowSlots);
    expect(w.eat(FOOD.apple)).toBe(true);
  });
});

describe('the pouch — behind the screen', () => {
  it('opening it lets go of a raised shield, a drawn bow and a charging spin', () => {
    const a = arena();
    a.inp.guard = true;
    run(a.w, a.inp, 0.3);
    expect(a.w.act).toBe('guard');
    a.w.cancelActs();
    expect(a.w.act).toBe('idle');
    const b = arena();
    b.inp.aim = true;
    run(b.w, b.inp, 0.7);
    expect(b.w.act).toBe('aim');
    b.w.cancelActs();
    expect(b.w.act).toBe('idle');
    expect(b.w.draw).toBe(0);
    const c = arena();
    c.inp.attack = true;
    run(c.w, c.inp, 1.0);
    expect(c.w.act).toBe('charge');
    c.w.cancelActs();
    expect(c.w.act).toBe('idle');
    expect(c.w.charge).toBe(0);
  });

  it('a pose can be taken without a step: the hero shows what is carried', () => {
    const { w, inp } = arena();
    w.step(DT, inp);
    w.equip('shield', 0);
    w.unequip('shield');
    w.refreshPose(inp);
    expect(w.pose.shield).toBe(false);
    w.equip('shield', 0);
    w.refreshPose(inp);
    expect(w.pose.shield).toBe(true);
    expect(w.pose.shieldModel).toBe(SHIELD_MODEL.hylian);
  });

  it('is deterministic for the same input stream, pouch calls and all', () => {
    const play = () => {
      const { w, inp } = arena();
      w.weapons.push({ kind: WEAPON.bone, wear: 22 });
      w.shields.push({ kind: SHIELD.traveler, wear: 24 });
      const foe = foeAt(w, 1, 3, Math.PI / 2);
      foe.roused = true; foe.mind = 'chase';
      const out: number[] = [];
      run(w, inp, 6, (t) => {
        inp.attack = Math.round(t / DT) % 20 < 2;
        inp.guard = t > 3;
        if (Math.round(t / DT) === 100) w.equip('shield', 1);
        if (Math.round(t / DT) === 200) { w.equip('weapon', 1); w.health = 6; w.eat(FOOD.apple); }
        if (Math.round(t / DT) % 60 === 0) out.push(w.health, w.hero.x, w.hero.z, w.shields[w.shieldOn]?.wear ?? -1);
      });
      return out;
    };
    expect(play()).toEqual(play());
  });
});
