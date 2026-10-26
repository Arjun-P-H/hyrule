/**
 * What the map HUD shows (src/app/game/MapHud.tsx, MapScreen.tsx), out of the engine's loop: where the hero
 * is and which way the view faces, where the other players are, the temperature round the hero and how much
 * noise it makes. One mutable object the engine writes every frame and the HUD's own animation frames read
 * (the map turns and slides with the hero: no React re-render, as for the foes' markers in combat/hud.ts).
 */

import type { CamMode } from '@/lib/engine/statsStore';
import type { World } from '@/lib/world';

/** What the maps show in the cloud world, in place of the land's tiles. */
export const NO_CLOUD_MAP = 'No map of the cloud world';

/** The most players the map keeps (the server's limit is five, the other four are peers). */
export const MAP_PEERS = 8;

/** Another player on the map. */
type MapPeer = { id: string; name: string; char: number; x: number; y: number; z: number; yaw: number };

export type MapHudState = {
  /** The world's seed: the tiles drawn belong to it. */
  seed: number;
  /** The world the hero is in: the cloud world has no map (the maps say so). */
  world: World;
  /** The hero (the camera, in the first-person and flying views) in the world: metres, z is north. */
  x: number;
  y: number;
  z: number;
  /** The way the view faces (rad, 0 is north, clockwise): the minimap turns to it. */
  heading: number;
  /** The way the hero faces (rad). */
  facing: number;
  mode: CamMode;
  /** The air temperature round the hero (°C), refreshed about five times a second. */
  temp: number;
  /** How much noise the hero makes (0 silent … 1 loud), every frame. */
  noise: number;
  /** The land under the hero (sky/air.ts landName). */
  land: string;
  /** Whether the engine has written anything yet. */
  live: boolean;
  /** The other players in the world: the first `peerCount` of `peers`. */
  peers: MapPeer[];
  peerCount: number;
};

export function createMapHud(): MapHudState {
  const peers: MapPeer[] = [];
  for (let k = 0; k < MAP_PEERS; k++) peers.push({ id: '', name: '', char: 0, x: 0, y: 0, z: 0, yaw: 0 });
  return {
    seed: 0, world: 'land', x: 0, y: 0, z: 0, heading: 0, facing: 0, mode: 'hero', temp: 15, noise: 0, land: '', live: false,
    peers, peerCount: 0,
  };
}
