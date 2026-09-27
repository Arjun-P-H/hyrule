'use client';

/**
 * W(x,y) Continuous World Engine — "/" spec console.
 * ---------------------------------------------------------------------------
 * Dark engineering console that composes the pre-built live components:
 *   - TerrainDemo    (WebGL2 raymarcher, §30 Stage-0 GPU parity harness)
 *   - HillshadePanel (relief map from the CPU twin)
 *   - WorkedExample  (instrumented §30 march table)
 *   - FileBrowser    (the browser world engine sources via /api/files)
 * plus the distilled spec (formula cards), the §29 parameter table and the
 * §27/§31 parity contract. Seed + quality preset state lives here only —
 * TerrainDemo owns all interaction internally.
 */

import { Fragment, useEffect, useRef, useState } from 'react';
import {
  ArrowDownToLine,
  CloudFog,
  Compass,
  Cpu,
  Dices,
  FileCode2,
  Gauge,
  Hash,
  Layers,
  Mountain,
  Move,
  PersonStanding,
  Radar,
  Sigma,
  Sun,
  TreePine,
  Waves,
  type LucideIcon,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import dynamic from 'next/dynamic';
import Link from 'next/link';
const HillshadePanel = dynamic(() => import('@/components/terrain/HillshadePanel'), {
  ssr: false,
  loading: () => (
    <div className="h-64 rounded-xl border border-stone-800 bg-stone-950/50 animate-pulse flex items-center justify-center">
      <span className="font-mono text-xs text-stone-500">Loading radar relief map...</span>
    </div>
  ),
});

const TerrainDemo = dynamic(() => import('@/components/terrain/TerrainDemo'), {
  ssr: false,
  loading: () => (
    <div className="relative w-full h-[46vh] min-h-[320px] sm:h-[56vh] rounded-xl border border-stone-800 bg-stone-950 flex flex-col items-center justify-center gap-2">
      <div className="font-mono text-xs text-stone-400 animate-pulse flex items-center gap-2">
        <span className="size-2 rounded-full bg-emerald-500"></span>
        Loading WebGL2 Procedural Terrain Engine...
      </div>
      <span className="font-mono text-[10px] text-stone-600">Initializing shader pipeline</span>
    </div>
  ),
});

const WorkedExample = dynamic(() => import('@/components/terrain/WorkedExample'), {
  ssr: false,
  loading: () => (
    <div className="h-64 rounded-xl border border-stone-800 bg-stone-950/50 animate-pulse flex items-center justify-center">
      <span className="font-mono text-xs text-stone-500">Loading instrumented march table...</span>
    </div>
  ),
});

const FileBrowser = dynamic(() => import('@/components/FileBrowser'), {
  ssr: false,
  loading: () => (
    <div className="h-96 rounded-xl border border-stone-800 bg-stone-900/30 animate-pulse flex items-center justify-center">
      <span className="font-mono text-xs text-stone-500">Loading engine sources...</span>
    </div>
  ),
});
import {
  BIOME_ANCHORS,
  CHUNK_SIZE,
  DEFAULT_SEED,
  PARAM_TABLE,
  PLAYER,
  QUALITY_PRESETS,
  type ParamRow,
  type PresetId,
} from '@/lib/constants';
import { useSeedState } from '@/lib/engine/useSeedState';
import {
  THIN_SCROLL,
  oklabToHex,
} from '@/lib/utils';

/* -- small local helpers --------------------------------------------------- */

function FadeIn({
  children,
  className = '',
  id,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { rootMargin: '-60px' }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <section
      id={id}
      ref={ref}
      className={`transition-all duration-500 ease-out ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3.5'
      } ${className}`}
    >
      {children}
    </section>
  );
}

function LazyMount({
  children,
  fallback,
  rootMargin = '350px',
}: {
  children: React.ReactNode;
  fallback: React.ReactNode;
  rootMargin?: string;
}) {
  const [mounted, setMounted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (mounted) return;
    const el = ref.current;
    if (!el) return;

    if (typeof IntersectionObserver === 'undefined') {
      const id = requestAnimationFrame(() => setMounted(true));
      return () => cancelAnimationFrame(id);
    }

    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setMounted(true);
          obs.disconnect();
        }
      },
      { rootMargin }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [mounted, rootMargin]);

  return <div ref={ref}>{mounted ? children : fallback}</div>;
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="max-w-3xl">
      <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-emerald-400">
        {eyebrow}
      </div>
      <h2 className="mt-2 text-2xl font-semibold tracking-tight text-stone-100 sm:text-3xl">
        {title}
      </h2>
      {description && (
        <p className="mt-3 text-sm leading-relaxed text-stone-400">{description}</p>
      )}
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-stone-800 bg-stone-900/40 p-3">
      <div className="font-mono text-[10px] uppercase tracking-wider text-stone-500">
        {label}
      </div>
      <div className="mt-1 font-mono text-xs text-emerald-300 sm:text-sm">{value}</div>
      {sub && <div className="mt-1 font-mono text-[10px] text-stone-500">{sub}</div>}
    </div>
  );
}

/* -- content data ----------------------------------------------------------- */

const FORMULAS: { icon: LucideIcon; tag: string; title: string; formula: string; note: string }[] = [
  {
    icon: Hash,
    tag: '§4',
    title: 'Hash',
    formula: 'h = fmix32(x ⊕ (y·0x27D4EB2F) ⊕ (s·0x9E3779B9))',
    note: 'Strict uint32 everywhere: identical lattice on CPU (uint32_t), GPU (GLSL uint), GDScript (masked int64) and JS (Math.imul).',
  },
  {
    icon: Sigma,
    tag: '§5',
    title: 'Value noise',
    formula: 'S(t) = 6t⁵ − 15t⁴ + 10t³',
    note: 'C²-continuous quintic; bilinear corners hashed to [−1,1].',
  },
  {
    icon: Radar,
    tag: '§5',
    title: 'Nyquist weighting',
    formula: 'Δp(t) = t·2tan(FOV/2)/Res_y\nwᵢ = clamp(2 − fᵢ·Δp/0.5, 0, 1)',
    note: 'Ridged octaves use 2f; fBm renormalised ×(Σa/Σa·w). No sparkle at vista range: octaves fade exactly when the tile renderer stops resolving them.',
  },
  {
    icon: Mountain,
    tag: '§6',
    title: 'Ridged mountains',
    formula: '(1 − |N|)² multifractal',
    note: 'Lacunarity 2.17, gated by a 1/8192 macro mask, domain-warped (§14, ±620 m).',
  },
  {
    icon: Waves,
    tag: '§7/§8',
    title: 'Valleys & water',
    formula: 'F₂ − F₁ Worley edge carve',
    note: '48 m carve depth; water plane at 5 m with 0.18 edge-space shore softness.',
  },
  {
    icon: Compass,
    tag: '§16',
    title: 'March',
    formula: 'tᵢ₊₁ = tᵢ + max(0.05, (P.y−E)/√(‖∇E‖²+1) · 0.8)',
    note: '6-step bisection; horizon clamp caps every step at ≤ 6%·t.',
  },
  {
    icon: Sun,
    tag: '§17/§18',
    title: 'Normals & shadows',
    formula: 'n = (−∇E, 1) / ‖·‖,  ε = Δp',
    note: 'Shadow rays sample a 4×Δp field — free penumbra.',
  },
  {
    icon: CloudFog,
    tag: '§19',
    title: 'Atmosphere',
    formula: 'T = exp(−σ·t·exp(−h/900)),  σ = 9e-4',
    note: 'Beer–Lambert hides the 6 km march cutoff.',
  },
  {
    icon: TreePine,
    tag: '§10/§11/§21',
    title: 'Biomes',
    formula: 'wₖ = exp(−Σⱼ((xⱼ−cⱼₖ)/σⱼₖ)²)',
    note: 'RBF over (T, M, E, slope); colours blended in OKLab.',
  },
  {
    icon: Layers,
    tag: '§16/§33',
    title: 'Depth',
    formula: 'DEPTH = (P·V·p).z / (P·V·p).w',
    note: 'Reversed-Z depth buffer mapping for sub-millimeter precision at any viewing distance.',
  },
];

const PARITY_ROWS: { impl: string; precision: string; hash: string; role: string }[] = [
  {
    impl: 'src/lib/terrainMath.ts',
    precision: 'double (fp64 globals)',
    hash: 'Math.imul (strict uint32)',
    role: 'CPU mathematical twin & physics loop',
  },
  {
    impl: 'src/components/terrain/demoShader.ts',
    precision: 'fp32 chunk-local + fp32-exact origin',
    hash: 'GLSL uint',
    role: 'WebGL2 high-framerate raymarcher',
  },
  {
    impl: 'src/app/game/page.tsx',
    precision: 'fp64 physics / fp32 visual',
    hash: 'strict uint32',
    role: 'standalone fullscreen browser game',
  },
  {
    impl: 'src/components/terrain/TerrainDemo.tsx',
    precision: 'fp32 chunk-local',
    hash: 'GLSL uint',
    role: 'interactive console canvas',
  },
  {
    impl: 'src/components/terrain/HillshadePanel.tsx',
    precision: 'double globals',
    hash: 'Math.imul',
    role: '2D relief and elevation mapper',
  },
];

const TBDR_CARDS: { icon: LucideIcon; title: string; note: string }[] = [
  {
    icon: Layers,
    title: 'Fullscreen Triangle Pipeline',
    note: 'Zero vertex buffer allocation; gl_VertexID generates clip-space triangle directly on the GPU without mesh overhead.',
  },
  {
    icon: Gauge,
    title: 'WebGLSync Frame Pacing',
    note: 'GPU fence syncing (clientWaitSync / fenceSync) prevents command queue congestion and maintains stable 120-240 FPS.',
  },
  {
    icon: Move,
    title: 'Toroidal Origin Rebasing',
    note: 'Double-precision chunk origin rebasing at 8192 m (2^13) combined with dyadic frequencies (n/2^20) eliminates fp32 jitter.',
  },
  {
    icon: ArrowDownToLine,
    title: 'Adaptive Dynamic Resolution',
    note: 'Frametime EMA auto-adjusts render target resolution scale to sustain target framerate on any GPU or software rasterizer.',
  },
  {
    icon: Sun,
    title: 'Central-Difference Normals',
    note: 'Analytic 4-tap central differences at the pixel footprint provide Nyquist-consistent shading in a single fragment pass.',
  },
  {
    icon: CloudFog,
    title: 'OKLab & Beer-Lambert Atmosphere',
    note: '5-biome RBF blending in OKLab with Beer-Lambert extinction fog hiding the 6 km cutoff distance seamlessly.',
  },
];

/** PARAM_TABLE grouped by `group`, preserving declaration order. */
const PARAM_GROUPS: [string, ParamRow[]][] = (() => {
  const map = new Map<string, ParamRow[]>();
  for (const row of PARAM_TABLE) {
    const arr = map.get(row.group);
    if (arr) arr.push(row);
    else map.set(row.group, [row]);
  }
  return Array.from(map.entries());
})();

/* -- page -------------------------------------------------------------------- */

export default function Home() {
  const { seed, seedText, onSeedChange, randomizeSeed } = useSeedState(DEFAULT_SEED);
  const [preset, setPreset] = useState<PresetId>('med');

  const engineChips = [
    { k: 'chunk', v: `${CHUNK_SIZE} (2^${Math.round(Math.log2(CHUNK_SIZE))})` },
    { k: 'walk', v: `${PLAYER.walkSpeed} m/s ×${PLAYER.sprintMul}` },
    { k: 'ground bias', v: `${PLAYER.groundBias} m` },
    { k: 'gravity', v: `${PLAYER.gravity} m/s²` },
  ];

  return (
    <div className="flex min-h-screen flex-col bg-stone-950 text-stone-200 antialiased selection:bg-emerald-500/30">
      {/* 1 — sticky header ------------------------------------------------- */}
      <header className="sticky top-0 z-40 border-b border-stone-800 bg-stone-950/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-3 px-4 sm:px-6">
          <a href="#top" className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-10 shrink-0 items-center justify-center rounded-md bg-emerald-500 font-mono text-[10px] font-bold tracking-tighter text-stone-950">
              W(x,y)
            </span>
            <span className="flex min-w-0 flex-col leading-tight">
              <span className="truncate text-sm font-semibold tracking-tight text-stone-100">
                Continuous World Engine
              </span>
              <span className="hidden font-mono text-[10px] text-stone-500 sm:block">
                procedural terrain · pure math · zero assets
              </span>
            </span>
          </a>
          <nav className="hidden items-center gap-6 md:flex">
            {[
              ['#demo', 'Demo'],
              ['#math', 'Math'],
              ['#parity', 'Parity'],
              ['#files', 'Files'],
            ].map(([href, label]) => (
              <a
                key={href}
                href={href}
                className="font-mono text-xs uppercase tracking-wider text-stone-400 transition-colors hover:text-emerald-300"
              >
                {label}
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <a
              href="/game"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-500 hover:bg-emerald-400 text-stone-950 font-mono text-xs font-bold transition-all shadow-lg hover:shadow-emerald-500/20 active:scale-95"
            >
              🎮 Fullscreen Game
            </a>
            <Badge
              variant="outline"
              className="border-stone-700 bg-stone-900/60 font-mono text-[10px] text-stone-300"
            >
              WebGL2 · 120-240 FPS · Pure Math
            </Badge>
          </div>
        </div>
      </header>

      {/* 2 — hero ----------------------------------------------------------- */}
      <section id="top" className="relative overflow-hidden border-b border-stone-800/60">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage:
              'radial-gradient(ellipse 90% 55% at 50% -12%, rgba(16,185,129,0.09), transparent 70%)',
          }}
        />
        <div className="relative mx-auto grid max-w-7xl gap-10 px-4 py-14 sm:px-6 lg:grid-cols-2 lg:gap-12 lg:py-20">
          <div>
            <Badge
              variant="outline"
              className="border-emerald-800/60 bg-emerald-950/40 font-mono text-[10px] text-emerald-300"
            >
              Pure math · zero mesh vertices
            </Badge>
            <h1 className="mt-5 text-3xl font-semibold leading-[1.12] tracking-tight text-stone-100 sm:text-4xl lg:text-[2.6rem]">
              Tears of the Kingdom terrain, generated by{' '}
              <span className="text-emerald-300">mathematics alone</span>.
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-relaxed text-stone-400 sm:text-base">
              The world is a continuous 2D field{' '}
              <code className="rounded bg-stone-900 px-1 py-0.5 font-mono text-xs text-emerald-300">
                E(x,z)
              </code>{' '}
              evaluated per-pixel by Lipschitz-corrected raymarching — there is no
              terrain mesh. Every pixel, collision and shadow derives
              deterministically from one uint32 hash chain, and a
              double-precision CPU twin drives physics with jump-preserving
              contact. Five implementations of the same equations agree to under
              a millimetre.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <Link
                href="/game"
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-stone-950 font-mono text-xs font-bold transition-all shadow-lg hover:shadow-emerald-500/25 active:scale-95"
              >
                🎮 Play Fullscreen in Browser
              </Link>
              <a
                href="#demo"
                className="flex items-center gap-1.5 px-4 py-2.5 rounded-lg border border-stone-800 bg-stone-900/60 hover:bg-stone-800 text-stone-300 font-mono text-xs transition-colors"
              >
                Interactive Canvas ↓
              </a>
            </div>
            <div className="mt-8 grid grid-cols-2 gap-3">
              <Stat label="geometry" value="0 mesh vertices" sub="no LOD · no streaming" />
              <Stat label="determinism" value="uint32 hash lattice" sub="Murmur3 finalizer" />
              <Stat label="interpolant" value="C² quintic S(t)" sub="6t⁵−15t⁴+10t³" />
              <Stat
                label="solver"
                value="λ=0.8 Lipschitz stepper"
                sub="Δt = (P.y−E)/√(‖∇E‖²+1)"
              />
            </div>
          </div>

          <Card className="gap-4 self-start border-stone-800 bg-stone-900/30 p-5">
            <CardHeader className="gap-1 px-0">
              <CardTitle className="text-sm font-medium tracking-tight text-stone-200">
                Relief map — computed live by the CPU twin
              </CardTitle>
              <CardDescription className="font-mono text-[11px] text-stone-500">
                seed {seed} · spawn (2016, −2592) · ±1.8 km
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0">
              <HillshadePanel seed={seed} />
              <p className="mt-3 font-mono text-[10px] leading-relaxed text-stone-500">
                E(x,z) sampled at 200² points around the spawn, shaded with the §17
                analytic normal against the contract sun; water plane at 5 m.
                Computed row-by-row across animation frames — no jank.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 3 — live demo ------------------------------------------------------- */}
      <FadeIn id="demo" className="scroll-mt-20 border-b border-stone-800/60">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§30 · WebGL2 terrain engine"
            title="The procedural world, live in your browser"
            description="This canvas runs the procedural world field in real-time WebGL2 (fp32 chunk-local evaluation, floating origin at 2^13). Walk mode is driven by the TypeScript engine running the §31 movement loop — jump-preserving clamp, GROUND_BIAS parity absorber, slope slide-off. Click the canvas first, then fly."
          />

          <div className="mt-8 rounded-xl border border-stone-800 bg-stone-900/30 p-4 sm:p-5">
            {/* controls around TerrainDemo — state lives here only */}
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <label className="flex flex-col gap-1.5">
                <span className="font-mono text-[10px] uppercase tracking-wider text-stone-500">
                  seed (uint32)
                </span>
                <span className="flex gap-2">
                  <Input
                    type="number"
                    min={0}
                    max={2147483647}
                    value={seedText}
                    onChange={(e) => onSeedChange(e.target.value)}
                    aria-label="Terrain seed"
                    className="h-9 w-40 border-stone-700 bg-stone-950 font-mono text-sm text-stone-200 focus-visible:ring-emerald-500/30"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={randomizeSeed}
                    aria-label="Random seed"
                    title="random seed"
                    className="h-9 w-9 border-stone-700 bg-stone-950 text-stone-300 hover:bg-stone-800 hover:text-emerald-300"
                  >
                    <Dices className="size-4" />
                  </Button>
                </span>
              </label>

              <div className="flex flex-col gap-1.5">
                <span className="font-mono text-[10px] uppercase tracking-wider text-stone-500">
                  quality preset
                </span>
                <TooltipProvider delayDuration={150}>
                  <span className="flex flex-wrap gap-2">
                    {QUALITY_PRESETS.map((p) => {
                      const activePreset = preset === p.id;
                      return (
                        <Tooltip key={p.id}>
                          <TooltipTrigger asChild>
                            <Button
                              variant="outline"
                              onClick={() => setPreset(p.id)}
                              aria-pressed={activePreset}
                              className={`h-9 min-w-16 font-mono text-xs ${
                                activePreset
                                  ? 'border-emerald-500 bg-emerald-500 text-stone-950 hover:bg-emerald-400'
                                  : 'border-stone-700 bg-stone-950 text-stone-300 hover:bg-stone-800 hover:text-emerald-300'
                              }`}
                            >
                              {p.label.toUpperCase()}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent className="border border-stone-700 bg-stone-900 font-mono text-[11px] text-stone-200">
                            {p.resScale}× res · {p.marchSteps} march ·{' '}
                            {p.shadowSteps} shadow — {p.note}
                          </TooltipContent>
                        </Tooltip>
                      );
                    })}
                  </span>
                </TooltipProvider>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {engineChips.map((c) => (
                <span
                  key={c.k}
                  className="rounded border border-stone-800 bg-stone-950 px-2 py-1 font-mono text-[10px] text-stone-400"
                >
                  <span className="text-stone-600">{c.k}</span> {c.v}
                </span>
              ))}
            </div>

            <div className="mt-4">
              <LazyMount
                fallback={
                  <div className="relative w-full h-[46vh] min-h-[320px] sm:h-[56vh] rounded-xl border border-stone-800 bg-stone-950 flex flex-col items-center justify-center gap-2">
                    <div className="font-mono text-xs text-stone-400 animate-pulse flex items-center gap-2">
                      <span className="size-2 rounded-full bg-emerald-500"></span>
                      Loading WebGL2 Procedural Terrain Engine...
                    </div>
                    <span className="font-mono text-[10px] text-stone-600">Initializing shader pipeline</span>
                  </div>
                }
              >
                <TerrainDemo seed={seed} preset={preset} />
              </LazyMount>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              {[
                {
                  icon: Move,
                  title: 'Camera',
                  text: 'Drag = look · WASD = move · Shift = sprint',
                },
                {
                  icon: PersonStanding,
                  title: 'Traversal',
                  text: 'F = walk/fly · Space = jump · C = descend',
                },
                {
                  icon: Cpu,
                  title: 'Physics parity',
                  text: "Walk mode = §31 CPU twin physics; fly mode clamps against the twin's elevation",
                },
              ].map((h) => (
                <div
                  key={h.title}
                  className="flex gap-3 rounded-lg border border-stone-800 bg-stone-900/30 p-3"
                >
                  <h.icon className="mt-0.5 size-4 shrink-0 text-emerald-400" />
                  <div className="min-w-0">
                    <div className="font-mono text-[10px] uppercase tracking-wider text-stone-500">
                      {h.title}
                    </div>
                    <div className="mt-1 text-xs leading-relaxed text-stone-300">
                      {h.text}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </FadeIn>

      {/* 4 — worked example --------------------------------------------------- */}
      <FadeIn className="border-b border-stone-800/60">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§30 · instrumented march"
            title="Worked numerical example — the march, instrumented (§30)"
            description="One camera ray from the tuned spawn, followed step by step through the Lipschitz stepper: per-step pixel footprint Δp(t), Nyquist weights, gradient denominators and the 6-step bisection that pins the hit distance t*."
          />
          {/* overflow-x-auto contains the instrumented table's min-content
              width on narrow viewports (WorkedExample is off-limits to edit) */}
          <div className="mt-8 overflow-x-auto">
            <LazyMount
              fallback={
                <div className="h-64 rounded-xl border border-stone-800 bg-stone-950/50 animate-pulse flex items-center justify-center">
                  <span className="font-mono text-xs text-stone-500">Loading instrumented march table...</span>
                </div>
              }
            >
              <WorkedExample />
            </LazyMount>
          </div>
        </div>
      </FadeIn>

      {/* 5 — math --------------------------------------------------------------- */}
      <FadeIn id="math" className="scroll-mt-20 border-b border-stone-800/60">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§4 – §21 · the spec, one card per subsystem"
            title="The specification, distilled"
            description="Every constant below traces to the §29 parameter table and is shared across the browser engine — TypeScript CPU math twin, WebGL2 raymarcher, and interactive game controller."
          />

          <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {FORMULAS.map((f) => (
              <Card key={f.title} className="gap-3 border-stone-800 bg-stone-900/30 p-5">
                <div className="flex items-center gap-2">
                  <f.icon className="size-4 text-emerald-400" />
                  <span className="font-mono text-[10px] uppercase tracking-wider text-stone-500">
                    {f.tag} · {f.title}
                  </span>
                </div>
                <div className="whitespace-pre-line break-words font-mono text-xs leading-relaxed text-emerald-300">
                  {f.formula}
                </div>
                <p className="text-xs leading-relaxed text-stone-400">{f.note}</p>
              </Card>
            ))}
          </div>

          <Card className="mt-4 gap-4 border-stone-800 bg-stone-900/30 p-5">
            <CardHeader className="gap-1 px-0">
              <CardTitle className="flex items-center gap-2 text-sm font-medium tracking-tight text-stone-200">
                <TreePine className="size-4 text-emerald-400" />
                Biome RBF anchors — OKLab space
              </CardTitle>
              <CardDescription className="font-mono text-[11px] text-stone-500">
                wₖ = exp(−Σ((xⱼ−cⱼₖ)/σⱼₖ)²) over (T, M, E, slope) — centres
                below; colours blend in OKLab
              </CardDescription>
            </CardHeader>
            <CardContent className="min-w-0 px-0">
              <div className={`overflow-x-auto ${THIN_SCROLL}`}>
                <table className="w-full min-w-[560px] border-collapse font-mono text-[11px]">
                  <thead>
                    <tr className="border-b border-stone-800 text-left text-stone-500">
                      <th className="px-3 py-2 font-normal">biome</th>
                      <th className="px-3 py-2 text-right font-normal">L</th>
                      <th className="px-3 py-2 text-right font-normal">a</th>
                      <th className="px-3 py-2 text-right font-normal">b</th>
                      <th className="px-3 py-2 text-right font-normal">T (°C)</th>
                      <th className="px-3 py-2 text-right font-normal">M</th>
                      <th className="px-3 py-2 text-right font-normal">E (m)</th>
                      <th className="px-3 py-2 text-right font-normal">slope</th>
                    </tr>
                  </thead>
                  <tbody className="text-stone-300">
                    {BIOME_ANCHORS.map((b) => (
                      <tr key={b.name} className="border-b border-stone-800/60">
                        <td className="px-3 py-2">
                          <span className="flex items-center gap-2">
                            <span
                              aria-hidden
                              className="inline-block size-3 shrink-0 rounded-sm border border-stone-700"
                              style={{ backgroundColor: oklabToHex(...b.oklab) }}
                            />
                            {b.name}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right text-stone-400">
                          {b.oklab[0].toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right text-stone-400">
                          {b.oklab[1].toFixed(3)}
                        </td>
                        <td className="px-3 py-2 text-right text-stone-400">
                          {b.oklab[2].toFixed(3)}
                        </td>
                        <td className="px-3 py-2 text-right text-amber-300">{b.t}</td>
                        <td className="px-3 py-2 text-right text-amber-300">
                          {b.m.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right text-amber-300">{b.e}</td>
                        <td className="px-3 py-2 text-right text-amber-300">
                          {b.slope.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </FadeIn>

      {/* 6 — parameter table ------------------------------------------------- */}
      <FadeIn className="border-b border-stone-800/60">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§29 · auditable constants"
            title="Parameter table (§29) — CPU/GPU auditable"
            description={`${PARAM_TABLE.length} constants across ${PARAM_GROUPS.length} groups. The cpu and gpu columns name the exact symbol in the TypeScript twin and the shader, so every value is checkable against the sources in the file browser below.`}
          />
          <div
            className={`mt-8 max-h-[480px] overflow-auto rounded-xl border border-stone-800 bg-stone-950 ${THIN_SCROLL}`}
          >
            <table className="w-full min-w-[760px] border-collapse text-left">
              <thead className="sticky top-0 z-10 bg-stone-900 text-xs text-stone-400">
                <tr>
                  <th className="px-3 py-2.5 font-normal">symbol</th>
                  <th className="px-3 py-2.5 font-normal">name</th>
                  <th className="px-3 py-2.5 font-normal">value</th>
                  <th className="px-3 py-2.5 font-normal">cpu</th>
                  <th className="px-3 py-2.5 font-normal">gpu</th>
                  <th className="px-3 py-2.5 font-normal">spec</th>
                </tr>
              </thead>
              <tbody>
                {PARAM_GROUPS.map(([group, rows]) => (
                  <Fragment key={group}>
                    <tr className="bg-stone-900/60">
                      <td
                        colSpan={6}
                        className="border-y border-stone-800 px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-emerald-500"
                      >
                        {group}
                      </td>
                    </tr>
                    {rows.map((r) => (
                      <tr
                        key={`${group}-${r.symbol}-${r.name}`}
                        className="border-b border-stone-800/50 hover:bg-stone-900/30"
                      >
                        <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-emerald-300">
                          {r.symbol}
                        </td>
                        <td className="px-3 py-2 text-xs text-stone-300">{r.name}</td>
                        <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-amber-300">
                          {r.value}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] text-stone-500">
                          {r.cpu}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] text-stone-500">
                          {r.gpu}
                        </td>
                        <td className="px-3 py-2">
                          <Badge
                            variant="outline"
                            className="border-stone-700 bg-transparent font-mono text-[10px] text-stone-400"
                          >
                            {r.spec}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </FadeIn>

      {/* 7 — parity + TBDR/SDFGI ------------------------------------------------ */}
      <FadeIn id="parity" className="scroll-mt-20 border-b border-stone-800/60">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§27 · §31 · mathematical invariance"
            title="Parity contract & precision architecture"
            description="The procedural equations run across the browser stack — TypeScript CPU math twin, WebGL2 shader pipeline, and interactive game runner. The contract: double globals, fp32 chunk-locals, strict uint32 hashing."
          />

          <div className="mt-8 grid gap-4 lg:grid-cols-5">
            <Card className="min-w-0 gap-4 border-stone-800 bg-stone-900/30 p-5 lg:col-span-3">
              <CardHeader className="gap-1 px-0">
                <CardTitle className="flex items-center gap-2 text-sm font-medium tracking-tight text-stone-200">
                  <Cpu className="size-4 text-emerald-400" />
                  The parity table
                </CardTitle>
                <CardDescription className="font-mono text-[11px] text-stone-500">
                  implementation / precision / hash math / role
                </CardDescription>
              </CardHeader>
              <CardContent className="min-w-0 px-0">
                <div className={`overflow-x-auto ${THIN_SCROLL}`}>
                  <table className="w-full min-w-[560px] border-collapse text-left">
                    <thead>
                      <tr className="border-b border-stone-800 text-xs text-stone-500">
                        <th className="px-3 py-2 font-normal">implementation</th>
                        <th className="px-3 py-2 font-normal">precision</th>
                        <th className="px-3 py-2 font-normal">hash math</th>
                        <th className="px-3 py-2 font-normal">role</th>
                      </tr>
                    </thead>
                    <tbody>
                      {PARITY_ROWS.map((r) => (
                        <tr key={r.impl} className="border-b border-stone-800/60">
                          <td className="px-3 py-2.5">
                            <span className="flex items-center gap-2">
                              <FileCode2 className="size-3.5 shrink-0 text-emerald-400" />
                              <span className="font-mono text-xs text-emerald-300">
                                {r.impl}
                              </span>
                            </span>
                          </td>
                          <td className="px-3 py-2.5 text-xs text-stone-300">
                            {r.precision}
                          </td>
                          <td className="px-3 py-2.5 font-mono text-xs text-amber-300">
                            {r.hash}
                          </td>
                          <td className="px-3 py-2.5 text-xs text-stone-400">
                            {r.role}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card className="min-w-0 gap-3 border-amber-900/40 bg-amber-950/20 p-5 lg:col-span-2">
              <div className="flex items-center gap-2">
                <Gauge className="size-4 text-amber-400" />
                <span className="font-mono text-[10px] uppercase tracking-wider text-amber-500/90">
                  divergence budget
                </span>
              </div>
              <p className="text-sm leading-relaxed text-stone-300">
                fp32 GPU vs fp64 CPU differ{' '}
                <span className="font-mono text-amber-300">&lt; 1 mm</span> on a{' '}
                <span className="font-mono">700 m</span> peak — absorbed by{' '}
                <span className="font-mono text-amber-300">GROUND_BIAS = 0.08 m</span>.
              </p>
              <p className="text-sm leading-relaxed text-stone-400">
                Frequencies are dyadic{' '}
                <span className="font-mono">n/2^20</span> with chunk{' '}
                <span className="font-mono">2^13</span> so{' '}
                <span className="font-mono">origin·f</span> is exact in fp32: zero
                lattice jitter at rebasing.
              </p>
            </Card>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {TBDR_CARDS.map((c) => (
              <Card key={c.title} className="gap-2.5 border-stone-800 bg-stone-900/30 p-4">
                <div className="flex items-center gap-2">
                  <c.icon className="size-4 shrink-0 text-emerald-400" />
                  <span className="font-mono text-xs text-emerald-300">{c.title}</span>
                </div>
                <p className="text-xs leading-relaxed text-stone-400">{c.note}</p>
              </Card>
            ))}
          </div>
        </div>
      </FadeIn>

      {/* 8 — files --------------------------------------------------------------- */}
      <FadeIn id="files" className="scroll-mt-20">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§34 · deliverable"
            title="Browser Engine Sources"
            description="Every file below is real and served straight from src/ — the WebGL2 raymarch shader, the TypeScript procedural mathematics engine, and the first-person locomotion controller."
          />
          <div className="mt-8">
            <LazyMount
              fallback={
                <div className="h-96 rounded-xl border border-stone-800 bg-stone-900/30 animate-pulse flex items-center justify-center">
                  <span className="font-mono text-xs text-stone-500">Loading engine sources...</span>
                </div>
              }
            >
              <FileBrowser />
            </LazyMount>
          </div>
        </div>
      </FadeIn>

      {/* 9 — footer (mt-auto keeps it pinned to the bottom) ------------------- */}
      <footer className="mt-auto border-t border-stone-800 bg-stone-950">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 font-mono text-[10px] leading-relaxed text-stone-500 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <span>
            W(x,y) Continuous World Engine — WebGL2 Browser Engine · 120-240 FPS · seed 1337
          </span>
          <span className="text-stone-600">
            CPU twin: fp64 · GPU: fp32 + dyadic origin · GROUND_BIAS 0.08 m
          </span>
        </div>
      </footer>
    </div>
  );
}
