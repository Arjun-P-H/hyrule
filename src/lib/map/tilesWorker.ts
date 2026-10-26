/**
 * The map's tile painter, off the main thread: a request (tile, seed) in, the tile's pixels out.
 * (tileService.ts starts it; a tile's maths is tiles.ts renderTile.)
 */

import { TILE_PX, renderTile } from './tiles';

export type TileRequest = { id: number; seed: number; level: number; tx: number; tz: number };
export type TileReply = { id: number; px: ArrayBuffer };

type WorkerScope = {
  onmessage: ((e: MessageEvent<TileRequest>) => void) | null;
  postMessage(message: TileReply, transfer: Transferable[]): void;
};

const scope = self as unknown as WorkerScope;

scope.onmessage = (e) => {
  const { id, seed, level, tx, tz } = e.data;
  const px = new Uint8ClampedArray(TILE_PX * TILE_PX * 4);
  renderTile(seed, level, tx, tz, px);
  scope.postMessage({ id, px: px.buffer as ArrayBuffer }, [px.buffer]);
};
