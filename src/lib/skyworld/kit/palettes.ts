/**
 * The Eyries' palettes: the colours each class builds in (SKY_CIVILIZATION.md
 * §7), linear RGB. An instance names its palette; the shader
 * (structures.wgsl.ts) reads its materials' colours from it.
 */

type V3 = readonly [number, number, number];

type Palette = {
  name: string;
  stone: V3;     // ashlar walls
  trim: V3;      // dressed stone: cornices, quoins, mouldings
  marble: V3;    // columns, statues
  roof: V3;      // scale tiles
  roof2: V3;     // second roof colour (accents)
  dome: V3;      // dome sheathing (copper, verdigris, turquoise)
  timber: V3;
  metal: V3;     // gilding
  iron: V3;
  bronze: V3;
  cloth: V3;     // banners and awnings
  cloth2: V3;    // their second colour (stripes, borders)
  emblem: V3;    // the House emblem on the cloth
  plaster: V3;
  lacquer: V3;
  glass: V3;     // tint of the glass
  crystal: V3;   // skystone crystal
  lamp: V3;      // sunstone light (emission, scene units at night)
  window: V3;    // a lit window
  mosaic: readonly [V3, V3, V3];
  paving: V3;
  /** How much moss (0–1) and weathering (0–1) the stone carries. */
  moss: number;
  wear: number;
  /** Stone course height and block length (m). */
  course: number;
  block: number;
};

export const PALETTES: readonly Palette[] = [
  {
    name: 'crown', // white limestone, slate-blue scale tiles, verdigris domes, gold, night-blue and gold
    stone: [0.56, 0.52, 0.44], trim: [0.68, 0.64, 0.56], marble: [0.72, 0.7, 0.65],
    roof: [0.075, 0.13, 0.24], roof2: [0.1, 0.32, 0.3], dome: [0.09, 0.34, 0.31],
    timber: [0.12, 0.07, 0.035], metal: [0.85, 0.58, 0.18], iron: [0.03, 0.03, 0.035], bronze: [0.3, 0.19, 0.08],
    cloth: [0.014, 0.025, 0.1], cloth2: [0.65, 0.42, 0.1], emblem: [0.82, 0.58, 0.16],
    plaster: [0.7, 0.66, 0.58], lacquer: [0.5, 0.06, 0.03],
    glass: [0.2, 0.32, 0.38], crystal: [0.35, 0.6, 1.0],
    lamp: [1.0, 0.6, 0.24], window: [1.0, 0.56, 0.22],
    mosaic: [[0.5, 0.47, 0.4], [0.12, 0.17, 0.26], [0.45, 0.32, 0.13]], paving: [0.42, 0.39, 0.33],
    moss: 0.25, wear: 0.35, course: 0.56, block: 1.15,
  },
  {
    name: 'mirror', // white marble, turquoise domes, bronze, sea green
    stone: [0.62, 0.6, 0.55], trim: [0.74, 0.72, 0.67], marble: [0.8, 0.79, 0.75],
    roof: [0.04, 0.3, 0.3], roof2: [0.05, 0.42, 0.42], dome: [0.04, 0.45, 0.44],
    timber: [0.13, 0.08, 0.04], metal: [0.78, 0.55, 0.22], iron: [0.03, 0.035, 0.035], bronze: [0.34, 0.22, 0.1],
    cloth: [0.02, 0.2, 0.17], cloth2: [0.7, 0.68, 0.6], emblem: [0.8, 0.62, 0.25],
    plaster: [0.74, 0.72, 0.66], lacquer: [0.45, 0.08, 0.05],
    glass: [0.18, 0.36, 0.38], crystal: [0.3, 0.75, 0.9],
    lamp: [1.0, 0.68, 0.32], window: [1.0, 0.62, 0.3],
    mosaic: [[0.75, 0.73, 0.68], [0.04, 0.32, 0.34], [0.55, 0.42, 0.18]], paving: [0.5, 0.48, 0.43],
    moss: 0.15, wear: 0.2, course: 0.6, block: 1.4,
  },
  {
    name: 'halo', // weathered grey-white stone, moss, dark slate, blue runes
    stone: [0.4, 0.4, 0.37], trim: [0.5, 0.5, 0.46], marble: [0.58, 0.58, 0.55],
    roof: [0.05, 0.07, 0.1], roof2: [0.08, 0.16, 0.2], dome: [0.06, 0.2, 0.22],
    timber: [0.1, 0.07, 0.04], metal: [0.6, 0.5, 0.3], iron: [0.03, 0.03, 0.035], bronze: [0.25, 0.18, 0.09],
    cloth: [0.55, 0.55, 0.52], cloth2: [0.08, 0.2, 0.35], emblem: [0.2, 0.45, 0.75],
    plaster: [0.6, 0.6, 0.56], lacquer: [0.4, 0.06, 0.04],
    glass: [0.15, 0.25, 0.32], crystal: [0.3, 0.6, 1.0],
    lamp: [0.75, 0.85, 1.0], window: [0.85, 0.75, 0.55],
    mosaic: [[0.55, 0.55, 0.5], [0.12, 0.22, 0.3], [0.35, 0.33, 0.28]], paving: [0.36, 0.36, 0.33],
    moss: 0.7, wear: 0.65, course: 0.7, block: 1.3,
  },
  {
    name: 'beacon', // pale stone, violet crystal, brass, blue-grey roofs
    stone: [0.5, 0.5, 0.5], trim: [0.62, 0.62, 0.62], marble: [0.7, 0.7, 0.72],
    roof: [0.1, 0.12, 0.2], roof2: [0.35, 0.28, 0.12], dome: [0.42, 0.3, 0.12],
    timber: [0.1, 0.07, 0.05], metal: [0.75, 0.55, 0.25], iron: [0.03, 0.03, 0.04], bronze: [0.45, 0.3, 0.12],
    cloth: [0.12, 0.08, 0.28], cloth2: [0.6, 0.6, 0.65], emblem: [0.7, 0.62, 0.9],
    plaster: [0.66, 0.66, 0.66], lacquer: [0.35, 0.08, 0.2],
    glass: [0.2, 0.25, 0.4], crystal: [0.55, 0.45, 1.0],
    lamp: [0.85, 0.75, 1.0], window: [1.0, 0.7, 0.4],
    mosaic: [[0.62, 0.62, 0.64], [0.2, 0.15, 0.4], [0.5, 0.42, 0.2]], paving: [0.42, 0.42, 0.43],
    moss: 0.1, wear: 0.3, course: 0.5, block: 1.0,
  },
  {
    name: 'petal', // white plaster, dark timber, grey-green tiles, vermilion
    stone: [0.42, 0.4, 0.36], trim: [0.55, 0.52, 0.46], marble: [0.65, 0.63, 0.58],
    roof: [0.06, 0.09, 0.08], roof2: [0.12, 0.2, 0.17], dome: [0.1, 0.22, 0.18],
    timber: [0.07, 0.035, 0.02], metal: [0.8, 0.55, 0.2], iron: [0.025, 0.025, 0.03], bronze: [0.3, 0.18, 0.08],
    cloth: [0.55, 0.06, 0.04], cloth2: [0.75, 0.72, 0.65], emblem: [0.85, 0.65, 0.2],
    plaster: [0.74, 0.7, 0.62], lacquer: [0.62, 0.08, 0.035],
    glass: [0.2, 0.25, 0.25], crystal: [0.5, 0.6, 1.0],
    lamp: [1.0, 0.5, 0.25], window: [1.0, 0.6, 0.3],
    mosaic: [[0.6, 0.56, 0.5], [0.45, 0.08, 0.06], [0.15, 0.25, 0.18]], paving: [0.4, 0.37, 0.32],
    moss: 0.4, wear: 0.3, course: 0.45, block: 0.9,
  },
  {
    name: 'bough', // living wood, moss, weave, warm light
    stone: [0.3, 0.27, 0.22], trim: [0.4, 0.36, 0.3], marble: [0.55, 0.52, 0.46],
    roof: [0.08, 0.12, 0.05], roof2: [0.25, 0.18, 0.08], dome: [0.2, 0.25, 0.1],
    timber: [0.16, 0.1, 0.055], metal: [0.7, 0.5, 0.2], iron: [0.04, 0.035, 0.03], bronze: [0.3, 0.2, 0.08],
    cloth: [0.35, 0.3, 0.12], cloth2: [0.15, 0.3, 0.1], emblem: [0.85, 0.75, 0.4],
    plaster: [0.55, 0.5, 0.4], lacquer: [0.4, 0.12, 0.04],
    glass: [0.25, 0.3, 0.2], crystal: [0.4, 0.8, 0.6],
    lamp: [1.0, 0.65, 0.3], window: [1.0, 0.62, 0.3],
    mosaic: [[0.45, 0.4, 0.3], [0.2, 0.3, 0.12], [0.5, 0.35, 0.15]], paving: [0.3, 0.27, 0.22],
    moss: 0.8, wear: 0.4, course: 0.5, block: 1.0,
  },
  {
    name: 'ember', // black basalt, iron, bronze, ember orange
    stone: [0.07, 0.065, 0.065], trim: [0.12, 0.11, 0.1], marble: [0.2, 0.18, 0.17],
    roof: [0.04, 0.035, 0.035], roof2: [0.3, 0.12, 0.04], dome: [0.32, 0.16, 0.06],
    timber: [0.06, 0.035, 0.02], metal: [0.7, 0.42, 0.15], iron: [0.03, 0.028, 0.028], bronze: [0.4, 0.22, 0.08],
    cloth: [0.45, 0.05, 0.02], cloth2: [0.08, 0.07, 0.07], emblem: [0.95, 0.5, 0.1],
    plaster: [0.3, 0.27, 0.25], lacquer: [0.5, 0.08, 0.03],
    glass: [0.3, 0.15, 0.08], crystal: [1.0, 0.45, 0.2],
    lamp: [1.0, 0.45, 0.15], window: [1.0, 0.42, 0.12],
    mosaic: [[0.15, 0.13, 0.12], [0.5, 0.15, 0.04], [0.3, 0.25, 0.2]], paving: [0.1, 0.09, 0.085],
    moss: 0.0, wear: 0.5, course: 0.6, block: 1.2,
  },
  {
    name: 'rime', // blue-grey stone, ice, silver roofs, white banners
    stone: [0.36, 0.4, 0.45], trim: [0.48, 0.52, 0.58], marble: [0.62, 0.66, 0.72],
    roof: [0.3, 0.34, 0.4], roof2: [0.5, 0.55, 0.62], dome: [0.45, 0.5, 0.58],
    timber: [0.1, 0.08, 0.07], metal: [0.6, 0.62, 0.66], iron: [0.03, 0.035, 0.04], bronze: [0.3, 0.25, 0.2],
    cloth: [0.7, 0.72, 0.75], cloth2: [0.1, 0.15, 0.3], emblem: [0.15, 0.25, 0.5],
    plaster: [0.6, 0.64, 0.7], lacquer: [0.3, 0.1, 0.1],
    glass: [0.3, 0.4, 0.5], crystal: [0.5, 0.75, 1.0],
    lamp: [1.0, 0.7, 0.4], window: [1.0, 0.62, 0.3],
    mosaic: [[0.5, 0.55, 0.6], [0.15, 0.22, 0.4], [0.7, 0.72, 0.78]], paving: [0.38, 0.42, 0.46],
    moss: 0.0, wear: 0.25, course: 0.65, block: 1.3,
  },
  {
    name: 'fallen', // weathered stone, heavy moss, faded roofs
    stone: [0.36, 0.34, 0.3], trim: [0.44, 0.42, 0.37], marble: [0.5, 0.49, 0.45],
    roof: [0.06, 0.08, 0.1], roof2: [0.08, 0.2, 0.18], dome: [0.08, 0.22, 0.19],
    timber: [0.08, 0.06, 0.04], metal: [0.45, 0.38, 0.2], iron: [0.035, 0.03, 0.03], bronze: [0.2, 0.18, 0.12],
    cloth: [0.05, 0.06, 0.1], cloth2: [0.35, 0.3, 0.2], emblem: [0.45, 0.38, 0.2],
    plaster: [0.5, 0.48, 0.44], lacquer: [0.3, 0.08, 0.05],
    glass: [0.12, 0.15, 0.15], crystal: [0.3, 0.55, 0.9],
    lamp: [0.6, 0.85, 0.7], window: [0.4, 0.5, 0.45],
    mosaic: [[0.45, 0.43, 0.38], [0.12, 0.18, 0.22], [0.35, 0.3, 0.18]], paving: [0.32, 0.31, 0.28],
    moss: 0.95, wear: 0.85, course: 0.56, block: 1.15,
  },
  {
    name: 'harbour', // sandstone, terracotta, striped canvas
    stone: [0.52, 0.4, 0.27], trim: [0.6, 0.5, 0.36], marble: [0.68, 0.62, 0.52],
    roof: [0.36, 0.1, 0.04], roof2: [0.42, 0.24, 0.08], dome: [0.4, 0.22, 0.08],
    timber: [0.14, 0.085, 0.045], metal: [0.8, 0.55, 0.2], iron: [0.03, 0.03, 0.03], bronze: [0.35, 0.22, 0.09],
    cloth: [0.6, 0.18, 0.04], cloth2: [0.75, 0.7, 0.6], emblem: [0.08, 0.1, 0.2],
    plaster: [0.7, 0.62, 0.5], lacquer: [0.5, 0.1, 0.04],
    glass: [0.2, 0.28, 0.3], crystal: [0.4, 0.6, 1.0],
    lamp: [1.0, 0.6, 0.26], window: [1.0, 0.58, 0.25],
    mosaic: [[0.6, 0.5, 0.38], [0.36, 0.12, 0.05], [0.12, 0.2, 0.3]], paving: [0.45, 0.36, 0.26],
    moss: 0.1, wear: 0.4, course: 0.5, block: 1.0,
  },
  {
    name: 'wells', // grey stone, weathered timber, canvas
    stone: [0.32, 0.32, 0.3], trim: [0.42, 0.42, 0.4], marble: [0.55, 0.55, 0.52],
    roof: [0.08, 0.09, 0.08], roof2: [0.15, 0.13, 0.1], dome: [0.12, 0.2, 0.18],
    timber: [0.11, 0.08, 0.055], metal: [0.55, 0.45, 0.25], iron: [0.03, 0.03, 0.03], bronze: [0.25, 0.2, 0.12],
    cloth: [0.6, 0.58, 0.5], cloth2: [0.2, 0.3, 0.35], emblem: [0.15, 0.3, 0.4],
    plaster: [0.55, 0.55, 0.52], lacquer: [0.35, 0.1, 0.05],
    glass: [0.2, 0.28, 0.3], crystal: [0.35, 0.6, 0.9],
    lamp: [1.0, 0.65, 0.3], window: [1.0, 0.6, 0.28],
    mosaic: [[0.45, 0.45, 0.42], [0.15, 0.25, 0.28], [0.3, 0.27, 0.2]], paving: [0.34, 0.34, 0.32],
    moss: 0.6, wear: 0.5, course: 0.5, block: 1.0,
  },
];

export const PALETTE: Record<string, number> = Object.fromEntries(PALETTES.map((p, i) => [p.name, i]));
