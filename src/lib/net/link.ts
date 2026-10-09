/**
 * What the engine's frame loop sees of a multiplayer session (session.ts):
 * other players' hero snapshots to draw, a way to send its own, and the
 * shared clock. Types only, so the engine chunk and the page share no code.
 */

/** The sky clock as sent between players: game days, stopped or not, real minutes per game day. */
export type ClockState = { days: number; paused: boolean; dayMinutes: number };

export type NetLink = {
  /** In the world (joined): snapshots and clocks are sent only then. */
  inWorld(): boolean;
  /** The other players in the world: id → character (src/lib/hero/characters.ts). */
  peers(): ReadonlyMap<string, number>;
  /** Hand over the snapshots received since the last call: player, local arrival time (ms), snapshot. */
  drain(take: (id: string, at: number, snap: readonly number[]) => void): void;
  /** Send this player's snapshot (remoteHeroes.ts encodeSnapshot). */
  sendState(snap: readonly number[]): void;
  /** A clock another player set, taken once (null: none since the last call). */
  takeClock(): ClockState | null;
  sendClock(c: ClockState): void;
  /** This player keeps the world's clock (the first in the world): it sends it now and then. */
  keepsClock(): boolean;
};
