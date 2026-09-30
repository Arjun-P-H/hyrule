/**
 * The window of placement tiles kept around the camera (pure; the GPU side is
 * forest.ts). Tiles are squares of TILE_CELLS × TILE_CELLS lattice cells;
 * the window is `side` × `side` tiles centred on the camera's tile, stored
 * toroidally: tile (tx, tz) lives in slot (tz mod side) · side + (tx mod side),
 * so when the camera crosses a tile edge only the row or column that entered
 * the window is generated, into the slots of the one that left.
 */

export const TILE_CELLS = 8;

/** A tile to generate: absolute tile coordinates and the slot it goes to. */
type TileJob = [tx: number, tz: number, slot: number];

const mod = (i: number, n: number) => ((i % n) + n) % n;

export class TileWindow {
  readonly side: number;
  private cx = NaN;
  private cz = NaN;
  private seed = NaN;

  constructor(side: number) {
    if (side < 2 || side % 2 !== 0) throw new Error(`tile window side must be even: ${side}`);
    this.side = side;
  }

  /** First tile of the window along an axis, for the camera's tile c. */
  private first(c: number): number {
    return c - this.side / 2;
  }

  slot(tx: number, tz: number): number {
    return mod(tz, this.side) * this.side + mod(tx, this.side);
  }

  /** Forget everything (the next update regenerates the whole window). */
  invalidate(): void {
    this.cx = NaN;
  }

  /** Move the window to the camera's tile; returns the tiles that entered it. */
  update(tx: number, tz: number, seed: number): TileJob[] {
    const n = this.side;
    const jobs: TileJob[] = [];
    const full = !Number.isFinite(this.cx) || seed !== this.seed || Math.abs(tx - this.cx) >= n || Math.abs(tz - this.cz) >= n;
    if (!full && tx === this.cx && tz === this.cz) return jobs;
    const x0 = this.first(tx), z0 = this.first(tz);
    const ox0 = this.first(this.cx), oz0 = this.first(this.cz);
    for (let z = z0; z < z0 + n; z++) {
      for (let x = x0; x < x0 + n; x++) {
        const inOld = !full && x >= ox0 && x < ox0 + n && z >= oz0 && z < oz0 + n;
        if (!inOld) jobs.push([x, z, this.slot(x, z)]);
      }
    }
    this.cx = tx;
    this.cz = tz;
    this.seed = seed;
    return jobs;
  }
}
