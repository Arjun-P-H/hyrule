/**
 * Canvas drawing the minimap and the full map share: the tiles under a view, the hero's arrow, another
 * player's marker. A view is a centre (x, z in the world; z is north) and a scale (metres to the pixel);
 * the caller has put the canvas's origin where the centre goes and turned it, if the map turns.
 */

import { MAX_LEVEL, TILE_PX, levelFor, levelMpp, levelSpan, tilesAround } from '@/lib/map/tiles';
import type { TileService } from '@/lib/map/tileService';

/** The ground the tiles have not covered yet. */
export const MAP_GROUND = '#142022';

/**
 * Draw the tiles over `reach` pixels round (cx, cz) at `mpp` metres to the pixel. A tile still being painted
 * shows a coarser one over it, if one is there.
 */
export function drawTiles(ctx: CanvasRenderingContext2D, svc: TileService, cx: number, cz: number, mpp: number, reach: number): void {
  const level = levelFor(mpp);
  const span = levelSpan(level);
  const k = levelMpp(level) / mpp;
  const size = TILE_PX * k;
  const { tx0, tx1, tz0, tz1 } = tilesAround(level, cx, cz, reach * mpp);
  const coarse = Math.min(MAX_LEVEL, level + 2);
  ctx.imageSmoothingEnabled = k < 1 || k > 1.0001;
  for (let tz = tz0; tz <= tz1; tz++) {
    for (let tx = tx0; tx <= tx1; tx++) {
      const dx = (tx * span - cx) / mpp;
      const dy = -((tz + 1) * span - cz) / mpp;
      const tile = svc.get(level, tx, tz, span);
      if (tile) {
        ctx.drawImage(tile, dx, dy, size + 0.6, size + 0.6);
        continue;
      }
      // not painted yet: ask for a coarser tile over it (cheap, and it covers its neighbours too), and show the finest there is
      if (coarse > level) {
        const f = 2 ** (coarse - level);
        svc.get(coarse, Math.floor(tx / f), Math.floor(tz / f), levelSpan(coarse), 4);
      }
      for (let up = 1; up <= MAX_LEVEL - level; up++) {
        const f = 2 ** up;
        const parent = svc.peek(level + up, Math.floor(tx / f), Math.floor(tz / f));
        if (!parent) continue;
        const ix = tx - f * Math.floor(tx / f), iz = tz - f * Math.floor(tz / f);
        const part = TILE_PX / f;
        ctx.drawImage(parent, ix * part, (f - 1 - iz) * part, part, part, dx, dy, size + 0.6, size + 0.6);
        break;
      }
    }
  }
}

/** The hero's arrow at (x, y), pointing `angle` radians clockwise from up. */
export function drawArrow(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, size: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(0, -size);
  ctx.lineTo(size * 0.72, size * 0.82);
  ctx.lineTo(0, size * 0.36);
  ctx.lineTo(-size * 0.72, size * 0.82);
  ctx.closePath();
  ctx.fillStyle = '#f0ea3a';
  ctx.fill();
  ctx.lineJoin = 'round';
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(30,30,0,0.5)';
  ctx.stroke();
  ctx.restore();
}

/** Another player: a dot in their colour with a white ring and a nub for the way they face (`angle`, clockwise from up). */
export function drawPeer(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, colour: string, size: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(0, -size * 1.75);
  ctx.lineTo(size * 0.6, -size * 0.7);
  ctx.lineTo(-size * 0.6, -size * 0.7);
  ctx.closePath();
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(0, 0, size, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.lineWidth = Math.max(1.5, size * 0.28);
  ctx.strokeStyle = '#fff';
  ctx.stroke();
  ctx.restore();
}

/** A pointer on the map's edge for a player past it: at (x, y), `angle` clockwise from up is outward. */
export function drawEdgePointer(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, colour: string, size: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(0, -size);
  ctx.lineTo(size * 0.95, size * 0.7);
  ctx.lineTo(0, size * 0.25);
  ctx.lineTo(-size * 0.95, size * 0.7);
  ctx.closePath();
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.lineJoin = 'round';
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = '#fff';
  ctx.stroke();
  ctx.restore();
}
