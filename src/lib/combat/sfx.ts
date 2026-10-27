/**
 * Combat's sounds, made on the fly with Web Audio — no sound files: a swing's
 * whoosh (a sweep of filtered noise), a blow's thump and crack, the shield's
 * clank and a Perfect Guard's bright ring, the bow's creak and twang, an
 * arrow's thud, a Bokoblin's alarm, the smoke it vanishes in, a pickup's
 * chime, a tick as gear is equipped and the crunch of a meal. Each is placed by where it happened: quieter with distance, panned
 * left or right of the view. Audio starts with the first click (browsers
 * allow it only after a gesture); `muted` silences it.
 */

import type { CombatEvent } from './world';

type Ctx = AudioContext;

export class CombatSfx {
  private ctx: Ctx | null = null;
  private out: GainNode | null = null;
  private noise: AudioBuffer | null = null;
  muted = false;
  volume = 0.5;

  /** Start or resume audio (from a user gesture). */
  wake(): void {
    if (typeof window === 'undefined') return;
    try {
      if (!this.ctx) {
        const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!AC) return;
        this.ctx = new AC();
        this.out = this.ctx.createGain();
        this.out.gain.value = this.volume;
        // (a gentle limiter, so a crowd of blows does not clip)
        const comp = this.ctx.createDynamicsCompressor();
        comp.threshold.value = -14; comp.ratio.value = 6;
        this.out.connect(comp).connect(this.ctx.destination);
        const n = this.ctx.sampleRate;
        this.noise = this.ctx.createBuffer(1, n, n);
        const d = this.noise.getChannelData(0);
        let s = 12345;
        for (let i = 0; i < n; i++) { s = (s * 16807) % 2147483647; d[i] = (s / 1073741823.5) - 1; }
      }
      if (this.ctx.state === 'suspended') void this.ctx.resume();
    } catch { /* no audio */ }
  }

  dispose(): void {
    void this.ctx?.close().catch(() => {});
    this.ctx = null;
  }

  /** Play the sound of event e for a listener at (x, y, z) facing yaw. */
  play(e: CombatEvent, lx: number, ly: number, lz: number, yaw: number): void {
    const c = this.ctx;
    if (!c || !this.out || this.muted || c.state !== 'running') return;
    const dx = e.x - lx, dy = e.y - ly, dz = e.z - lz;
    const d = Math.hypot(dx, dy, dz);
    const near = e.type === 'quake' ? Math.min(1, e.a / 10) / (1 + d / 60) : 1 / (1 + d / 7); // (a colossus is heard from afar)
    if (near < 0.04) return;
    const pan = d > 0.5 ? Math.max(-0.8, Math.min(0.8, (dx * Math.cos(yaw) - dz * Math.sin(yaw)) / d)) : 0;
    const bus = c.createStereoPanner();
    bus.pan.value = pan;
    const g = c.createGain();
    g.gain.value = near;
    g.connect(bus).connect(this.out);
    const t = c.currentTime + 0.005;
    switch (e.type) {
      case 'swing': this.whoosh(g, t, 0.14, 700, 2600, 0.32); break;
      case 'spin': this.whoosh(g, t, 0.45, 500, 2200, 0.4); break;
      case 'throw': this.whoosh(g, t, 0.3, 400, 1600, 0.35); break;
      case 'hit': this.thump(g, t, 150, 0.5); this.burst(g, t, 0.05, 'lowpass', 1800, 0.35); break;
      case 'sneak': case 'crit':
        this.thump(g, t, 150, 0.6); this.burst(g, t, 0.05, 'lowpass', 2200, 0.4); this.tone(g, t + 0.02, 'triangle', 1480, 1480, 0.3, 0.18);
        break;
      case 'block': this.clank(g, t, [520, 763, 1131], 0.18, 0.22); this.burst(g, t, 0.04, 'highpass', 2000, 0.25); break;
      case 'parry': case 'parried':
        this.clank(g, t, [1318, 1975, 2637, 3520], 0.9, 0.22); this.burst(g, t, 0.03, 'highpass', 3000, 0.35);
        break;
      case 'defeat': [523, 659, 784].forEach((f, i) => this.tone(g, t + i * 0.09, 'triangle', f, f, 0.22, 0.16)); break;
      case 'dodge': if (e.a === 2) { this.whoosh(g, t, 0.35, 300, 1200, 0.3); this.tone(g, t, 'sine', 880, 1760, 0.5, 0.15); } else this.whoosh(g, t, 0.2, 300, 900, 0.18); break;
      case 'flurry': this.whoosh(g, t, 0.6, 200, 3000, 0.35); this.tone(g, t, 'sine', 220, 110, 1.2, 0.18); break;
      case 'fire': this.tone(g, t, 'triangle', 240, 170, 0.16, 0.35); this.burst(g, t, 0.025, 'highpass', 2500, 0.2); break;
      case 'draw': this.burst(g, t, 0.22, 'bandpass', 320, 0.14); break;
      case 'thud': this.burst(g, t, 0.07, 'lowpass', 700, e.a ? 0.45 : 0.3); this.thump(g, t, 110, 0.18); break;
      case 'splash': this.burst(g, t, 0.35, 'bandpass', 900, 0.35); break;
      case 'alert': this.tone(g, t, 'square', 660, 660, 0.07, 0.12); this.tone(g, t + 0.08, 'square', 990, 990, 0.11, 0.12); break;
      case 'down': this.thump(g, t, 90, 0.45); break;
      case 'quake': this.tone(g, t, 'sine', 55, 28, 0.9, 0.9); this.burst(g, t, 0.7, 'lowpass', 260, 0.7); this.thump(g, t, 70, 0.6); break;
      case 'poof': this.whoosh(g, t, 0.55, 2200, 180, 0.35); break;
      case 'vanish': this.whoosh(g, t, 0.9, 400, 2400, 0.22); [784, 1175, 1568].forEach((f, i) => this.tone(g, t + 0.1 + i * 0.12, 'sine', f, f, 0.5, 0.08)); break;
      case 'pickup': [880, 1109, 1319].forEach((f, i) => this.tone(g, t + i * 0.06, 'triangle', f, f, 0.12, 0.16)); break;
      case 'return': [659, 988, 1319, 1976].forEach((f, i) => this.tone(g, t + i * 0.07, 'sine', f, f, 0.25, 0.14)); break;
      case 'hurt': this.thump(g, t, 120, 0.6); this.burst(g, t, 0.08, 'lowpass', 900, 0.4); break;
      case 'die': this.tone(g, t, 'triangle', 440, 110, 1.1, 0.3); break;
      case 'break': this.burst(g, t, 0.12, 'highpass', 1500, 0.5); this.clank(g, t, [430, 610], 0.2, 0.2); break;
      case 'surf': this.burst(g, t, 0.25, 'lowpass', 900, 0.25); break;
      case 'charged': this.tone(g, t, 'sine', 1320, 2640, 0.35, 0.16); break;
      case 'swap': case 'empty': this.burst(g, t, 0.03, 'bandpass', e.type === 'swap' ? 2400 : 600, 0.25); break;
      case 'equip': this.burst(g, t, 0.04, 'bandpass', 1800, 0.22); this.tone(g, t, 'triangle', 700, 1000, 0.08, 0.1); break;
      case 'eat':
        this.burst(g, t, 0.05, 'bandpass', 1200, 0.35); this.burst(g, t + 0.09, 0.05, 'bandpass', 900, 0.3);
        this.tone(g, t + 0.2, 'sine', 523, 784, 0.25, 0.1);
        break;
    }
  }

  private src(): AudioBufferSourceNode {
    const s = this.ctx!.createBufferSource();
    s.buffer = this.noise;
    s.loop = true;
    return s;
  }

  /** Filtered noise sweeping from f0 to f1 Hz over dur s. */
  private whoosh(to: AudioNode, t: number, dur: number, f0: number, f1: number, peak: number): void {
    const c = this.ctx!, s = this.src(), f = c.createBiquadFilter(), g = c.createGain();
    f.type = 'bandpass'; f.Q.value = 1.4;
    f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + dur);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + dur * 0.3); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    s.connect(f).connect(g).connect(to);
    s.start(t, Math.random()); s.stop(t + dur + 0.02);
  }

  /** A short burst of filtered noise. */
  private burst(to: AudioNode, t: number, dur: number, type: BiquadFilterType, freq: number, peak: number): void {
    const c = this.ctx!, s = this.src(), f = c.createBiquadFilter(), g = c.createGain();
    f.type = type; f.frequency.value = freq;
    g.gain.setValueAtTime(peak, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    s.connect(f).connect(g).connect(to);
    s.start(t, Math.random()); s.stop(t + dur + 0.02);
  }

  /** A pitched blip gliding from f0 to f1 Hz. */
  private tone(to: AudioNode, t: number, type: OscillatorType, f0: number, f1: number, dur: number, peak: number): void {
    const c = this.ctx!, o = c.createOscillator(), g = c.createGain();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    if (f1 !== f0) o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + 0.008); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(to);
    o.start(t); o.stop(t + dur + 0.02);
  }

  /** A body blow: a low sine dropping fast. */
  private thump(to: AudioNode, t: number, f: number, peak: number): void {
    this.tone(to, t, 'sine', f, f * 0.4, 0.16, peak);
  }

  /** Metal: inharmonic partials ringing down. */
  private clank(to: AudioNode, t: number, partials: number[], dur: number, peak: number): void {
    for (let i = 0; i < partials.length; i++) this.tone(to, t, i ? 'sine' : 'triangle', partials[i], partials[i] * 0.995, dur * (1 - i * 0.15), peak / (1 + i * 0.6));
  }
}
