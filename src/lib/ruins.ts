/**
 * Ruins and shrines (Zonai-style), placed by hash — CPU side: the boxes are
 * built here once per site and shared by the renderer (GPU instances) and
 * the hero's collision.
 * ---------------------------------------------------------------------------
 * One candidate site per RUINS.cell square, at a hashed spot away from its
 * border; kept on walkable ground above the water, or on a sky island's top
 * (well inside its rim and clear of its lake and stream).
 * A site is a stone platform bedded into the slope, pillars round its edge
 * (some broken), lintels over neighbouring intact pillars, a flight of steps,
 * low wall fragments and, for shrines, a dark monolith with glowing lines.
 * Every box of a site shares its yaw.
 */

import { RUINS } from './constants';
import { elevation, hash01, waterLevel } from './terrainMath';
import { islandAt } from './skyworld/realm';
import { islandRho, nearWater, topAt } from './skyworld/island';

/** Materials: weathered stone, mossy stone, shrine stone (glowing lines). */
type RuinMaterial = 0 | 1 | 2;

type RuinBox = {
  x: number; y: number; z: number;    // centre, world metres
  hx: number; hy: number; hz: number; // half extents (site-local axes)
  yaw: number;
  material: RuinMaterial;
};

export type RuinSite = {
  x: number; z: number;               // centre
  radius: number;                     // keeps trees and grass out
  top: number;                        // platform top
  shrine: boolean;
  boxes: RuinBox[];
};

/** Deterministic stream of [0, 1) values for one site. */
function stream(ix: number, iz: number, seed: number): () => number {
  let k = 0;
  return () => hash01(ix, iz, seed + 13001 + 97 * k++);
}

function buildSite(ix: number, iz: number, seed: number): RuinSite | null {
  if (hash01(ix, iz, seed + 13000) >= RUINS.probability) return null;
  const r = stream(ix, iz, seed);
  const x = (ix + 0.25 + 0.5 * r()) * RUINS.cell;
  const z = (iz + 0.25 + 0.5 * r()) * RUINS.cell;
  const yaw = r() * Math.PI * 0.5;
  const w = 12 + 12 * r(), d = 10 + 10 * r();
  const cs = Math.cos(yaw), sn = Math.sin(yaw);
  // Site-local (u along width, v along depth) → world.
  const at = (u: number, v: number) => [x + cs * u + sn * v, z - sn * u + cs * v] as const;

  // Ground: a sky island's top when the site is well inside one (and clear of its water), else the terrain.
  const isl = islandAt(x, z, seed);
  if (isl && (isl.kind === 'debris' || islandRho(isl, x, z) >= 0.6 || nearWater(isl, x, z, 30))) return null;
  const onIsland = !!isl;
  const ground = (px: number, pz: number) => (onIsland ? topAt(isl!, px, pz) : elevation(px, pz, seed, 0));
  const corners = [at(-w / 2, -d / 2), at(w / 2, -d / 2), at(-w / 2, d / 2), at(w / 2, d / 2), at(0, 0)];
  const hs = corners.map(([px, pz]) => ground(px, pz));
  const gMin = Math.min(...hs), gMax = Math.max(...hs);
  // (on the land, clear of its water: the sea, a lake, a river at its own level)
  if (gMax - gMin > RUINS.maxRelief || (!onIsland && corners.some(([px, pz]) => ground(px, pz) < waterLevel(px, pz, seed) + 1))) return null;

  const boxes: RuinBox[] = [];
  const box = (u: number, v: number, y: number, hx: number, hy: number, hz: number, material: RuinMaterial) => {
    const [bx, bz] = at(u, v);
    boxes.push({ x: bx, y, z: bz, hx, hy, hz, yaw, material });
  };
  const base = gMin - 0.8;
  const top = gMax + 0.45;
  box(0, 0, (base + top) / 2, w / 2, (top - base) / 2, d / 2, 0);

  // Pillars at the corners and along the long sides.
  const along = Math.max(1, Math.round(w / 7));
  const pillarAt: Array<[number, number]> = [];
  for (let i = 0; i <= along; i++) {
    const u = -w / 2 + 1.1 + (i / along) * (w - 2.2);
    pillarAt.push([u, -d / 2 + 1.1], [u, d / 2 - 1.1]);
  }
  const heights: number[] = [];
  for (const [u, v] of pillarAt) {
    const broken = r() < 0.35;
    const h = broken ? 1.2 + 2.8 * r() : 6 + 3 * r();
    heights.push(broken ? -h : h);
    box(u, v, top + h / 2, 0.6, h / 2, 0.6, r() < 0.4 ? 1 : 0);
  }
  // Lintels over neighbouring intact pillars on each long side.
  for (let side = 0; side < 2; side++) {
    for (let i = 0; i < along; i++) {
      const a = heights[2 * i + side], b = heights[2 * (i + 1) + side];
      if (a <= 0 || b <= 0) continue;
      const [ua, v] = pillarAt[2 * i + side], [ub] = pillarAt[2 * (i + 1) + side];
      const h = Math.min(a, b);
      box((ua + ub) / 2, v, top + h + 0.45, (ub - ua) / 2 + 0.7, 0.45, 0.7, 0);
    }
  }
  // Steps down the front, from the platform top toward the ground.
  const steps = Math.max(2, Math.round((top - gMin) / 0.35));
  for (let i = 1; i <= Math.min(steps, 5); i++) {
    const stepTop = top - i * 0.35;
    box(0, -d / 2 - 0.6 * i + 0.3, (stepTop + base) / 2, 2.2, (stepTop - base) / 2, 0.3, 0);
  }
  // A low broken wall across the back.
  if (r() < 0.7) {
    const len = w * (0.3 + 0.4 * r());
    box(-w / 2 + len / 2 + 1, d / 2 - 0.4, top + 0.7, len / 2, 0.7, 0.35, 1);
  }
  const shrine = r() < RUINS.shrineShare;
  if (shrine) box(0, 0, top + 2.8, 1.8, 2.8, 1.8, 2);
  return { x, z, radius: Math.hypot(w, d) / 2 + 2, top, shrine, boxes };
}

// Sites are pure functions of (cell, seed); recent ones are kept in a
// direct-mapped cache (no allocation on a hit: the hero asks every tick).
type Slot = { ix: number; iz: number; seed: number; site: RuinSite | null };
const slots: Array<Slot | undefined> = new Array(256);
function siteInCell(ix: number, iz: number, seed: number): RuinSite | null {
  const k = (Math.imul(ix, 73856093) ^ Math.imul(iz, 19349663) ^ Math.imul(seed, 83492791)) & 255;
  const hit = slots[k];
  if (hit && hit.ix === ix && hit.iz === iz && hit.seed === seed) return hit.site;
  const site = buildSite(ix, iz, seed);
  slots[k] = { ix, iz, seed, site };
  return site;
}

/** Ruin sites whose centre lies within `range` of (x, z), nearest first, at most `max`. */
export function ruinSitesNear(x: number, z: number, range: number, seed: number, max: number): RuinSite[] {
  const out: Array<{ s: RuinSite; d: number }> = [];
  const c0x = Math.floor((x - range) / RUINS.cell), c1x = Math.floor((x + range) / RUINS.cell);
  const c0z = Math.floor((z - range) / RUINS.cell), c1z = Math.floor((z + range) / RUINS.cell);
  for (let iz = c0z; iz <= c1z; iz++) {
    for (let ix = c0x; ix <= c1x; ix++) {
      const s = siteInCell(ix, iz, seed);
      if (!s) continue;
      const d = Math.hypot(s.x - x, s.z - z);
      if (d <= range) out.push({ s, d });
    }
  }
  out.sort((a, b) => a.d - b.d);
  return out.slice(0, max).map((o) => o.s);
}

/** The site whose footprint may touch (x, z) (a site never leaves the middle half of its cell). */
export function ruinSiteAt(x: number, z: number, seed: number): RuinSite | null {
  const s = siteInCell(Math.floor(x / RUINS.cell), Math.floor(z / RUINS.cell), seed);
  return s && Math.hypot(x - s.x, z - s.z) <= s.radius + 1 ? s : null;
}

/** (x, z) in a box's local frame (u along its width axis, v along its depth axis). */
export function toBoxLocal(b: RuinBox, x: number, z: number): [number, number] {
  const dx = x - b.x, dz = z - b.z;
  const cs = Math.cos(b.yaw), sn = Math.sin(b.yaw);
  return [cs * dx - sn * dz, sn * dx + cs * dz];
}
