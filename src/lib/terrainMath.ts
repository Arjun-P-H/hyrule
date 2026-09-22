/**
 * W(x,y) — CPU physics twin, TypeScript reference implementation (double precision).
 * ---------------------------------------------------------------------------
 * This is the exact math of the GPU raymarcher (demoShader.ts /
 * terrain_raymarch.gdshader), evaluated on the CPU:
 *   - walk-mode collision in the WebGL demo (spec §31 movement loop),
 *   - the hillshade relief map on the hero panel,
 *   - the live worked numerical example (spec §30).
 *
 * PARITY NOTES
 *  - Hashing is strict uint32 integer math (spec §27): Math.imul + >>> 0.
 *  - GPU evaluates the same functions in fp32 on chunk-local coordinates.
 *    Divergence vs this double-precision twin is ≤ ~1e-6 of amplitude
 *    (< 1 mm on a 700 m peak) and is absorbed by GROUND_BIAS = 0.08 (spec §31).
 *  - Field lookup convention: y-up world, terrain field is E(p.x, p.z);
 *    the spec's math z (elevation) maps to engine y.
 */

import {
  F, MACRO, MOUNTAIN, WARP, VALLEY, CONTINENT, REGION, ANOMALY,
  CLIMATE, MARCH, MAX_WORLD_H, MAX_DIST, PLAYER,
} from "./constants";

/* ------------------------------------------------------------------ */
/* §4 — Seed & deterministic hashing (uint32, Murmur3 finalizer)       */
/* ------------------------------------------------------------------ */

/** MurmurHash3 finalizer — avalanche. Input/output are uint32 bit patterns. */
export function hashU(x: number): number {
  let h = x | 0;
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * H(x_i, y_i, s) = finalize(x_i ⊕ (y_i · 0x27D4EB2F) ⊕ (s · 0x9E3779B9))
 * xi/yi are integer lattice coords (may be negative — two's-complement wrap).
 */
export function hash3(xi: number, yi: number, seed: number): number {
  const h =
    (xi | 0) ^
    Math.imul(yi | 0, 0x27d4eb2f) ^
    Math.imul(seed | 0, 0x9e3779b9);
  return hashU(h);
}

/** Uniform in [0,1] from an integer lattice cell. */
export function hash01(xi: number, yi: number, seed: number): number {
  return hash3(xi, yi, seed) / 4294967295.0;
}

/** Feature-point coords in [0,1]² packed from the two 12-bit halves of one hash. */
export function cellPoint(ix: number, iy: number, seed: number): [number, number] {
  const h = hash3(ix, iy, seed);
  return [(h & 0xfff) / 4095.0, ((h >>> 12) & 0xfff) / 4095.0];
}

/** Cell type byte packed from the top 8 bits of the same hash (0..1). */
export function cellType(ix: number, iy: number, seed: number): number {
  return (hash3(ix, iy, seed) >>> 24) / 255.0;
}

/* ------------------------------------------------------------------ */
/* §5 — Quintic value noise (C² continuous) & Nyquist weights          */
/* ------------------------------------------------------------------ */

export function quintic(t: number): number {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

export function vnoise(x: number, y: number, seed: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
  const uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

  const ux0 = ix | 0;
  const ux1 = (ux0 + 1) | 0;
  const uy0 = Math.imul(iy | 0, 0x27d4eb2f);
  const uy1 = Math.imul((iy | 0) + 1, 0x27d4eb2f);
  const sTerm = Math.imul(seed | 0, 0x9e3779b9);

  const h0 = (uy0 ^ sTerm) >>> 0;
  const h1 = (uy1 ^ sTerm) >>> 0;

  const a = hashU((ux0 ^ h0) >>> 0) / 4294967295.0;
  const b = hashU((ux1 ^ h0) >>> 0) / 4294967295.0;
  const c = hashU((ux0 ^ h1) >>> 0) / 4294967295.0;
  const d = hashU((ux1 ^ h1) >>> 0) / 4294967295.0;

  const v = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
  return v * 2.0 - 1.0;
}

/**
 * Nyquist octave weight (spec §5): fade out between 0.5 and 1 cycle/pixel.
 * Δp(t) = t · 2tan(FOV/2) / Res_y is the world-space pixel footprint.
 */
export function nyqW(freq: number, dp: number): number {
  return Math.min(1, Math.max(0, 2.0 - (freq * dp) / 0.5));
}

/**
 * fBm with Nyquist attenuation + amplitude renormalization (spec §5).
 * `px, py` are pre-scaled coordinates (p · f0); `f0` is the base world
 * frequency so the Nyquist weight uses the ABSOLUTE octave frequency
 * f_i = f0 · lac^i against the metre-space footprint Δp.
 */
export function fbm(
  x: number, y: number, seed: number,
  octaves: number, lacunarity: number, gain: number, dp: number,
  f0 = 1.0,
): number {
  let freq = 1.0, amp = 1.0, sum = 0.0, wsum = 0.0, asum = 0.0;
  for (let i = 0; i < octaves; i++) {
    const w = nyqW(f0 * freq, dp);
    if (w > 0.0) {
      sum += amp * w * vnoise(x * freq, y * freq, seed + i * 0x9e37);
      wsum += amp * w;
    }
    asum += amp;
    freq *= lacunarity;
    amp *= gain;
  }
  return sum * (asum / Math.max(wsum, 1e-6));
}

/**
 * Ridged multifractal (spec §6): (1 − |N|)² octaves; ridged features double
 * the effective bandwidth, so the Nyquist weight uses 2f.
 */
export function ridged(
  x: number, y: number, seed: number,
  octaves: number, lacunarity: number, gain: number, dp: number,
  f0 = 1.0,
): number {
  let freq = 1.0, amp = 0.5, sum = 0.0, wsum = 0.0, asum = 0.0;
  for (let i = 0; i < octaves; i++) {
    const w = nyqW(2.0 * f0 * freq, dp);
    if (w > 0.0) {
      const n = 1.0 - Math.abs(vnoise(x * freq, y * freq, seed + i * 0x85eb));
      const r = n * n;
      sum += amp * w * r;
      wsum += amp * w;
    }
    asum += amp;
    freq *= lacunarity;
    amp *= gain;
  }
  return sum * (asum / Math.max(wsum, 1e-6));
}

/* ------------------------------------------------------------------ */
/* §7 / §12 / §13 — Cellular (Worley) fields                           */
/* ------------------------------------------------------------------ */

export type VoronoiResult = {
  f1: number;          // nearest feature distance (cell units)
  f2: number;          // second nearest
  edge: number;        // F2 − F1 (spec §7 edge distance)
  centerX: number;     // nearest feature point, input-space coords
  centerY: number;
  cellHash: number;    // type byte (top 8 bits) of the owning cell
};

/**
 * 3×3 Worley scan. ONE hash per cell packs the feature point (two 12-bit
 * halves) and the cell type byte (top 8 bits) — see worklog, GPU budget.
 */
export function voronoi(x: number, y: number, seed: number): VoronoiResult {
  const ipx = Math.floor(x), ipy = Math.floor(y);
  const fx = x - ipx, fy = y - ipy;
  let f1 = 8.0, f2 = 8.0;
  let bcx = 0, bcy = 0, btype = 0;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const cx = ipx + i, cy = ipy + j;
      const [px, py] = cellPoint(cx, cy, seed);
      const dx = i + px - fx;
      const dy = j + py - fy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < f1) {
        f2 = f1; f1 = d;
        bcx = cx + px; bcy = cy + py;
        btype = cellType(cx, cy, seed);
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  return { f1, f2, edge: f2 - f1, centerX: bcx, centerY: bcy, cellHash: btype };
}

/** Spec §7: D_edge = F2 − F1 only. */
export function worleyEdge(x: number, y: number, seed: number): number {
  return voronoi(x, y, seed).edge;
}

/* ------------------------------------------------------------------ */
/* §14 — Domain warping                                                */
/* ------------------------------------------------------------------ */

export function domainWarp(
  x: number, y: number, seed: number, dp: number,
): [number, number] {
  const wx = x * F.warp, wy = y * F.warp;
  const qx = fbm(wx, wy, seed + 11, WARP.octaves, 2.0, 0.5, dp, F.warp);
  const qy = fbm(wx + 5.2, wy + 1.3, seed + 37, WARP.octaves, 2.0, 0.5, dp, F.warp);
  return [x + WARP.strength * qx, y + WARP.strength * qy];
}

/* ------------------------------------------------------------------ */
/* §12 — Geological anomalies (Zonai-style terraces, spire, spiral)    */
/* ------------------------------------------------------------------ */

export function anomalyE(x: number, y: number, seed: number): number {
  const v = voronoi(x / ANOMALY.cell, y / ANOMALY.cell, seed + 400);
  if (v.cellHash >= ANOMALY.probability) return 0.0;
  // center back in input-space metres
  const cx = v.centerX * ANOMALY.cell;
  const cy = v.centerY * ANOMALY.cell;
  const dx = x - cx, dy = y - cy;
  const r = Math.sqrt(dx * dx + dy * dy);
  if (r >= ANOMALY.rOut) return 0.0;
  const theta = Math.atan2(dy, dx);
  const fall = 1.0 - smoothstep(ANOMALY.rIn, ANOMALY.rOut, r);
  const t = r / ANOMALY.terraceStep;
  const steps = (Math.floor(t) + smoothstep(0.3, 0.7, t - Math.floor(t))) * ANOMALY.terraceHeight;
  const spire = ANOMALY.spireHeight * Math.exp(-((r / ANOMALY.spireRadius) ** 2));
  const spiral = ANOMALY.spiralAmp * Math.sin(ANOMALY.spiralTurns * theta + ANOMALY.spiralRadial * r);
  return fall * (steps + spire + spiral);
}

/* ------------------------------------------------------------------ */
/* §13 — Regional mathematical rules                                   */
/* ------------------------------------------------------------------ */

export type RegionInfo = {
  wRegion: number;      // 0 at cell borders → 1 in cell interior
  mountFactor: number;  // H_MOUNT multiplier for this region
  duneAmount: number;   // 0..1 crescent-dune blend
};

export function regionField(x: number, y: number, seed: number): RegionInfo {
  const v = voronoi(x / REGION.cell, y / REGION.cell, seed + 900);
  const wRegion = smoothstep(REGION.blendLo, REGION.blendHi, v.edge);
  const h = v.cellHash;
  const mountFactor = h < 0.33 ? REGION.hillsFactor
    : h < 0.66 ? REGION.mountainFactor
    : REGION.duneFactor;
  const duneAmount = smoothstep(0.66, 0.8, h) * wRegion;
  return { wRegion, mountFactor, duneAmount };
}

/** Crescent (barchan) dunes — trigonometric region rule f₁ (spec §13). */
export function dunesE(x: number, y: number, seed: number, dp: number, amount: number): number {
  if (amount <= 0.0) return 0.0;
  const phase = x * F.dune * 2.0 * Math.PI +
    3.0 * fbm(x * F.warp, y * F.warp, seed + 301, 2, 2.0, 0.5, dp, F.warp);
  const c = 1.0 - Math.abs(Math.sin(phase));
  const shaped = c * c * c;
  return REGION.duneHeight * amount * shaped;
}

/* ------------------------------------------------------------------ */
/* Master elevation field — spec §2 environment composition            */
/* ------------------------------------------------------------------ */

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/**
 * E(x, z) — terrain elevation in metres. `dp` is the Nyquist pixel footprint
 * (spec §5): pass ~0 for full-detail CPU evaluation, or the screen footprint
 * to mirror the GPU's filtered result. All lookups are chunk-local (spec §25).
 */
export function elevation(x: number, z: number, seed: number, dp: number): number {
  // §14 — warp the domain (fluid geological folding)
  const [wx, wy] = domainWarp(x, z, seed, dp);

  // §13 — regional rules on the raw (un-warped) plane: massive structures
  const reg = regionField(x, z, seed);

  // §9 support — continent field (base amplitude + ocean-moisture proxy)
  const cont = smoothstep(
    CONTINENT.lo, CONTINENT.hi,
    0.5 + 0.5 * vnoise(x * F.cont, z * F.cont, seed + 801),
  );

  // §5 — macro fBm hills
  const base = MACRO.hBase * (0.5 + 0.5 * fbm(wx * F.macro, wy * F.macro, seed + 1,
    MACRO.octaves, MACRO.lacunarity, MACRO.gain, dp, F.macro));

  // §6 — ridged mountains gated by a macro mask
  const maskVal = 0.5 + 0.5 * vnoise(wx * F.mask, wy * F.mask, seed + 77);
  const mask = smoothstep(MOUNTAIN.maskLo, MOUNTAIN.maskHi, maskVal);
  const hMount = MOUNTAIN.hMount * (1 + (reg.mountFactor - 1) * reg.wRegion);
  
  let ridge = 0.0;
  if (mask > 0.0) {
    ridge = ridged(wx * F.ridge, wy * F.ridge, seed + 5,
      MOUNTAIN.octaves, MOUNTAIN.lacunarity, MOUNTAIN.gain, dp, F.ridge);
  }

  // §7 — cellular valleys carve the network of rivers
  const de = worleyEdge(wx * F.valley, wy * F.valley, seed + 21);
  const valley = VALLEY.cDepth * (1.0 - smoothstep(0, VALLEY.wValley, de)) ** 2;

  // §13 — dune rule + §12 — anomalies (raw plane, rigid Zonai geometry)
  const dune = dunesE(x, z, seed, dp, reg.duneAmount);
  const anom = anomalyE(x, z, seed);

  return base * (CONTINENT.minFactor + (1 - CONTINENT.minFactor) * cont)
    + hMount * mask * ridge
    + dune
    - valley
    + anom;
}

/** Central-difference gradient at the pixel-footprint scale (spec §17). */
export function gradient(
  x: number, z: number, seed: number, eps: number,
): [number, number] {
  const e = Math.max(eps, 0.05);
  const gx = (elevation(x + e, z, seed, 0) - elevation(x - e, z, seed, 0)) / (2 * e);
  const gz = (elevation(x, z + e, seed, 0) - elevation(x, z - e, seed, 0)) / (2 * e);
  return [gx, gz];
}

/** Slope magnitude ‖∇E‖ — the Lipschitz factor + slide gate input. */
export function slope(x: number, z: number, seed: number): number {
  const [gx, gz] = gradient(x, z, seed, 0.25);
  return Math.hypot(gx, gz);
}

/* ------------------------------------------------------------------ */
/* §9 — Climate fields                                                 */
/* ------------------------------------------------------------------ */

export function temperature(
  x: number, z: number, e: number, _seed = 0, globalY?: number,
): number {
  const lat = globalY !== undefined ? globalY : z;
  return CLIMATE.tBase
    - CLIMATE.lambdaLat * Math.abs(lat)
    - CLIMATE.lambdaAlt * Math.max(0, e);
}

export function moisture(x: number, z: number, e: number, seed: number): number {
  const n = fbm(x * F.moist, z * F.moist, seed + 61, 2, 2.0, 0.5, 0, F.moist);
  const cont = smoothstep(CONTINENT.lo, CONTINENT.hi,
    0.5 + 0.5 * vnoise(x * F.cont, z * F.cont, seed + 801));
  // λ_ocean·exp(−D_ocean/σ) approximated by the continent field itself
  // (smooth, deterministic, identical CPU/GPU — see worklog).
  return CLIMATE.mBase + CLIMATE.mNoiseAmp * n + CLIMATE.mOceanLambda * (1 - cont);
}

/* ------------------------------------------------------------------ */
/* §16 — Lipschitz-corrected raymarch + bisection refinement           */
/* ------------------------------------------------------------------ */

export type MarchRow = {
  i: number;
  t: number;
  px: number; py: number; pz: number;
  e: number;
  gap: number;        // P.y − E
  dp: number;         // pixel footprint Δp(t)
  grad: number;       // ‖∇E‖
  step: number;       // Δt applied
  wMacro: number[];   // Nyquist weights of macro octaves at this step
  wRidge: number[];   // Nyquist weights (2f) of ridged octaves
};

export type MarchResult = {
  hit: boolean;
  t: number;
  rows: MarchRow[];
  bisectRows: { i: number; t: number; gap: number }[];
};

function octaveWeights(
  f0: number, octaves: number, lac: number, dp: number, doubled: boolean,
): number[] {
  const out: number[] = [];
  let f = f0;
  for (let i = 0; i < octaves; i++) {
    out.push(nyqW(doubled ? 2 * f : f, dp));
    f *= lac;
  }
  return out;
}

/**
 * March one ray against E. Records every step so the worked example (§30)
 * can show the Nyquist weights collapsing and the Lipschitz denominator
 * shrinking the steps near cliffs.
 */
export function marchRay(
  ro: [number, number, number],
  rd: [number, number, number],
  seed: number,
  opts: {
    steps: number;
    footprint: number;     // 2·tan(FOV/2)/Res_y
    maxDist?: number;
    record?: boolean;
  },
): MarchResult {
  const maxDist = opts.maxDist ?? MAX_DIST;
  const rows: MarchRow[] = [];
  let t: number = MARCH.startT;
  let tPrev: number = t;
  let grad = 0;

  for (let i = 0; i < opts.steps; i++) {
    const px = ro[0] + rd[0] * t;
    const py = ro[1] + rd[1] * t;
    const pz = ro[2] + rd[2] * t;

    if (py > MAX_WORLD_H && rd[1] > 0) return { hit: false, t, rows, bisectRows: [] }; // §26 early exit

    const dp = Math.max(t * opts.footprint, 1e-4);
    const e = elevation(px, pz, seed, dp);

    if (opts.record && rows.length < 40) {
      rows.push({
        i, t, px, py, pz, e, gap: py - e, dp, grad,
        step: t - tPrev,
        wMacro: octaveWeights(F.macro, MACRO.octaves, MACRO.lacunarity, dp, false),
        wRidge: octaveWeights(F.ridge, MOUNTAIN.octaves, MOUNTAIN.lacunarity, dp, true),
      });
    }

    if (py < e) {
      // crossed the surface — bisect over [tPrev, t] (spec §16)
      const bisectRows: { i: number; t: number; gap: number }[] = [];
      let lo: number = tPrev, hi: number = t;
      for (let b = 0; b < MARCH.bisectSteps; b++) {
        const tm = 0.5 * (lo + hi);
        const mx = ro[0] + rd[0] * tm;
        const my = ro[1] + rd[1] * tm;
        const mz = ro[2] + rd[2] * tm;
        const gap = my - elevation(mx, mz, seed, tm * opts.footprint);
        bisectRows.push({ i: b, t: tm, gap });
        if (gap < 0) hi = tm; else lo = tm;
      }
      return { hit: true, t: 0.5 * (lo + hi), rows, bisectRows };
    }

    // Lipschitz factor ‖∇E‖ via forward differences at the pixel-footprint
    // scale, refreshed every 2nd sample (bisection catches overshoot; the
    // horizon clamp below bounds the *stale* window — see §16/§34.1).
    if (i % 2 === 0) {
      const eps = Math.max(dp, 0.15);
      const ex = elevation(px + eps, pz, seed, dp);
      const ez = elevation(px, pz + eps, seed, dp);
      grad = Math.hypot((ex - e) / eps, (ez - e) / eps);
    }

    tPrev = t;
    let step = Math.max(
      MARCH.deltaMin,
      (py - e) / Math.sqrt(grad * grad + 1.0) * MARCH.lambdaStep,
    );
    step *= 1.0 + t * MARCH.stepGrowth; // §26 adaptive growth
    // Horizon clamp: no single step may exceed ~6% of distance travelled —
    // bounds tunneling through unseen cliffs when the gradient is stale.
    step = Math.min(step, t * 0.06 + 2.0);
    t += step;

    if (t > maxDist) return { hit: false, t, rows, bisectRows: [] };
  }
  return { hit: false, t, rows, bisectRows: [] };
}

/** §18 — soft shadow: march toward the sun with a 4× inflated footprint. */
export function softShadow(
  p: [number, number, number],
  sunDir: [number, number, number],
  seed: number,
  steps: number,
  footprint: number,
): number {
  let res = 1.0;
  let t = 0.6;
  for (let i = 0; i < steps; i++) {
    const qx = p[0] + sunDir[0] * t;
    const qy = p[1] + sunDir[1] * t;
    const qz = p[2] + sunDir[2] * t;
    if (qy > MAX_WORLD_H) break;
    const h = qy - elevation(qx, qz, seed, 4.0 * t * footprint); // §18: 4×Δp
    res = Math.min(res, 10.0 * h / t); // penumbra kernel
    if (res < 0.005) break;
    t += clamp(h * 0.8, 0.5, 60.0) + t * 0.04;
  }
  return clamp(res, 0, 1);
}

/* ------------------------------------------------------------------ */
/* §31 — CPU physics twin movement loop (jump-preserving)              */
/* ------------------------------------------------------------------ */

export type PlayerState = {
  x: number; y: number; z: number;   // global, double precision
  vx: number; vy: number; vz: number;
  grounded: boolean;
};

export type InputState = {
  forward: number; strafe: number;   // −1..1
  jump: boolean;
  sprint: boolean;
  yaw: number;                        // camera yaw for wish direction
};

/**
 * §31 loop verbatim: gravity always integrates; the ground clamp only ever
 * zeroes NEGATIVE vertical velocity so jumps are never killed; the slope
 * gate slides the player off steep warped ridges. Global frame, evaluated
 * through chunk-local coordinates (CHUNK_SIZE lattice alignment).
 */
export function updatePlayer(
  s: PlayerState, input: InputState, dt: number, seed: number,
): void {
  const speed = PLAYER.walkSpeed * (input.sprint ? PLAYER.sprintMul : 1.0);
  const sin = Math.sin(input.yaw), cos = Math.cos(input.yaw);
  // forward = (sin yaw, 0, cos yaw) — matches demo camera basis
  const wishX = (sin * input.forward + cos * input.strafe) * speed;
  const wishZ = (cos * input.forward - sin * input.strafe) * speed;

  // horizontal velocity: exponential approach (feels like critically damped)
  const k = 1 - Math.exp(-12 * dt);
  s.vx += (wishX - s.vx) * k;
  s.vz += (wishZ - s.vz) * k;

  if (input.jump && s.grounded) s.vy = PLAYER.jumpVelocity;

  s.vy -= PLAYER.gravity * dt;              // GRAVITY
  s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt;

  // Global-frame evaluation (spec §25/§27): the CPU twin computes in double
  // precision on GLOBAL coordinates; the GPU computes (local + chunk_origin)
  // in fp32 — the dyadic frequency table keeps that sum exact, and the
  // residual divergence is absorbed by GROUND_BIAS below.
  const groundZ = elevation(s.x, s.z, seed, 0) + PLAYER.groundBias + PLAYER.height;

  s.grounded = false;
  if (s.y < groundZ) {
    s.y = groundZ;
    if (s.vy < 0.0) s.vy = 0.0;            // stop FALLING only — preserve jumps
    s.grounded = true;
  }

  // Slope gate: slide off steep warped ridges (spec §31 insert)
  if (s.grounded) {
    const [gx, gz] = gradient(s.x, s.z, seed, 0.25);
    const m = Math.hypot(gx, gz);
    if (m > PLAYER.slopeMax) {
      const inv = 1 / m;
      s.vx += -gx * inv * (m - PLAYER.slopeMax) * PLAYER.slideAccel * dt;
      s.vz += -gz * inv * (m - PLAYER.slopeMax) * PLAYER.slideAccel * dt;
    }
  }
}
