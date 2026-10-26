/**
 * A trip through the portal (portal.ts), stepped every frame whatever the
 * view's mode: crossing the ring asks the view for the other world and fades
 * out; the moment the other world is drawn the hero arrives (the fade cut to
 * black, so no frame shows the new world from the old place) and the view
 * fades back in. A build that never comes gives up and fades back in where it
 * started. On WebGL2 (the view draws only the land) the ring says so.
 */

import { PORTAL } from './constants';
import type { World } from './world';

/** What the trip needs of the view: the world drawn (null: only the land, ever), and a request for another. */
type TripView = { world(): World | null; setWorld(w: World): void };

/** Seconds a trip waits for the other world before it gives up. */
export const TRIP_TIMEOUT = 8;

export class PortalTrip {
  private to: World | null = null;
  private from: World = 'land';
  private t = 0;
  private arrived = false;
  private cool = 0;

  /** A trip under way (fading out, waiting, or fading in). */
  busy(): boolean {
    return this.to !== null;
  }

  /** How dark the view is: 0 lit … 1 black. */
  fade(): number {
    if (this.to === null) return 0;
    return this.arrived ? Math.max(0, 1 - this.t / PORTAL.fade) : Math.min(1, this.t / PORTAL.fade);
  }

  /**
   * One frame: `crossed` the hero went through the ring this frame, `near` it stands by it, in `world`.
   * Returns the world arrived in (the caller moves the hero there) or a line to show.
   */
  step(dt: number, crossed: boolean, near: boolean, world: World, view: TripView): { arrive?: World; say?: string } {
    this.cool = Math.max(0, this.cool - dt);
    if (this.to === null) {
      if (this.cool > 0) return {};
      if (view.world() === null) {
        if (!crossed && !near) return {};
        this.cool = PORTAL.cooldown;
        return { say: 'The cloud world needs WebGPU' };
      }
      if (!crossed) return {};
      this.to = world === 'land' ? 'clouds' : 'land';
      this.from = world;
      this.t = 0;
      this.arrived = false;
      view.setWorld(this.to);
      return {};
    }
    this.t += dt;
    if (!this.arrived) {
      if (view.world() === this.to) {
        this.arrived = true;
        this.t = 0;
        return { arrive: this.to };
      }
      if (this.t > TRIP_TIMEOUT) {
        view.setWorld(this.from);
        this.to = this.from;
        this.arrived = true;
        this.t = 0;
        return { say: 'The way through is closed' };
      }
      return {};
    }
    if (this.t >= PORTAL.fade) {
      this.to = null;
      this.cool = PORTAL.cooldown;
    }
    return {};
  }
}
