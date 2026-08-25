'use client';

/**
 * W(x,y) Continuous World Engine — "/" spec console.
 * ---------------------------------------------------------------------------
 * Dark engineering console that composes the pre-built live components:
 *   - TerrainDemo    (WebGL2 raymarcher, §30 Stage-0 GPU parity harness)
 *   - HillshadePanel (relief map from the CPU twin)
 *   - WorkedExample  (instrumented §30 march table)
 *   - FileBrowser    (the real godot-project/ sources via /api/files)
 * plus the distilled spec (formula cards), the §29 parameter table and the
 * §27/§31 parity contract. Seed + quality preset state lives here only —
 * TerrainDemo owns all interaction internally.
 */

import { Fragment, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  ArrowDownToLine,
  Boxes,
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
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import TerrainDemo from '@/components/terrain/TerrainDemo';
import HillshadePanel from '@/components/terrain/HillshadePanel';
import WorkedExample from '@/components/terrain/WorkedExample';
import FileBrowser from '@/components/FileBrowser';
import {
  BIOME_ANCHORS,
  CHUNK_SIZE,
  DEFAULT_SEED,
  PARAM_TABLE,
  PLAYER,
  QUALITY_PRESETS,
  type ParamRow,
} from '@/lib/constants';

type PresetId = 'low' | 'med' | 'high';

/** Thin scrollbar — Tailwind arbitrary values only (globals.css untouched). */
const THIN_SCROLL =
  '[scrollbar-width:thin] [scrollbar-color:#44403c_transparent] ' +
  '[&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent ' +
  '[&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-stone-700';

/* -- small local helpers --------------------------------------------------- */

function FadeIn({
  children,
  className,
  id,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <motion.section
      id={id}
      className={className}
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
    >
      {children}
    </motion.section>
  );
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

/** OKLab → sRGB hex, for the biome anchor swatches. */
function oklabToHex(L: number, a: number, b: number): string {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  const enc = (c: number) => {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  };
  return `#${lin.map(enc).map((v) => v.toString(16).padStart(2, '0')).join('')}`;
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
    note: "Inherits Godot's reversed-Z Metal NDC, so real meshes inter-occlude with pure math.",
  },
];

const PARITY_ROWS: { impl: string; precision: string; hash: string; role: string }[] = [
  {
    impl: 'gdextension/src/terrain_cpu.h',
    precision: 'double globals',
    hash: 'uint32_t',
    role: 'reference twin, shipping physics',
  },
  {
    impl: 'scripts/terrain_math_fallback.gd',
    precision: 'double (GDScript floats are 64-bit)',
    hash: 'int64 masked to 32 bits',
    role: 'zero-dependency fallback',
  },
  {
    impl: 'shaders/terrain_raymarch.gdshader',
    precision: 'fp32 chunk-local + fp32-exact origin',
    hash: 'GLSL uint',
    role: 'the visible world',
  },
  {
    impl: 'src/lib/terrainMath.ts',
    precision: 'double',
    hash: 'Math.imul',
    role: 'browser twin + worked example',
  },
  {
    impl: 'demoShader.ts (GLSL ES 3.0)',
    precision: 'fp32 chunk-local',
    hash: 'GLSL uint',
    role: 'WebGL2 parity proof',
  },
];

const TBDR_CARDS: { icon: LucideIcon; title: string; note: string }[] = [
  {
    icon: Layers,
    title: 'render_priority = −100',
    note: 'Terrain first in the opaque bin; later per-tile fragments depth-test against analytic depth for free.',
  },
  {
    icon: ArrowDownToLine,
    title: 'depth_draw_always + DEPTH every pixel',
    note: 'A miss writes 0.0 = reversed-Z far.',
  },
  {
    icon: Gauge,
    title: 'MSAA 0 + FXAA',
    note: 'The Nyquist-weighted field IS the anti-aliaser; MSAA would multiply tile bandwidth for nothing.',
  },
  {
    icon: Sun,
    title: 'Light cull-mask',
    note: "The sun's engine light excludes the terrain layer; direct sun lives in EMISSION, SDFGI indirect still reaches ALBEDO.",
  },
  {
    icon: CloudFog,
    title: 'Sky DEPTH = 0',
    note: 'Outside every SDFGI cascade: the sky is never voxelized.',
  },
  {
    icon: Boxes,
    title: 'Red-box validation',
    note: 'Drop a red box at a cascade boundary to verify no light leaking.',
  },
];

/* -- page -------------------------------------------------------------------- */

export default function Home() {
  const [seedText, setSeedText] = useState(String(DEFAULT_SEED));
  const [seed, setSeed] = useState<number>(DEFAULT_SEED);
  const [preset, setPreset] = useState<PresetId>('med');

  const onSeedChange = (v: string) => {
    setSeedText(v);
    const n = Math.floor(Number(v));
    if (v.trim() !== '' && Number.isFinite(n)) {
      setSeed(((n % 4294967296) + 4294967296) % 4294967296);
    }
  };

  const randomizeSeed = () => {
    const n = 1 + Math.floor(Math.random() * 999_999);
    setSeedText(String(n));
    setSeed(n);
  };

  /** PARAM_TABLE grouped by `group`, preserving declaration order. */
  const paramGroups = useMemo(() => {
    const map = new Map<string, ParamRow[]>();
    for (const row of PARAM_TABLE) {
      const arr = map.get(row.group);
      if (arr) arr.push(row);
      else map.set(row.group, [row]);
    }
    return Array.from(map.entries());
  }, []);

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
          <Badge
            variant="outline"
            className="border-stone-700 bg-stone-900/60 font-mono text-[10px] text-stone-300"
          >
            Godot 4 · Metal · SDFGI
          </Badge>
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
                seed 1337 · spawn (2016, −2592) · ±1.8 km
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0">
              <HillshadePanel />
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
            eyebrow="§30 · stage-0 GPU parity"
            title="The same math, live in your browser"
            description="This canvas runs the exact field as the Godot Metal shader (WebGL2 port, fp32 chunk-local evaluation, floating origin at 2^13). Walk mode is driven by the TypeScript CPU twin running the §31 movement loop — jump-preserving clamp, GROUND_BIAS parity absorber, slope slide-off. Click the canvas first, then fly."
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
              <TerrainDemo seed={seed} preset={preset} />
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
            <WorkedExample />
          </div>
        </div>
      </FadeIn>

      {/* 5 — math --------------------------------------------------------------- */}
      <FadeIn id="math" className="scroll-mt-20 border-b border-stone-800/60">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <SectionHeading
            eyebrow="§4 – §21 · the spec, one card per subsystem"
            title="The specification, distilled"
            description="Every constant below traces to the §29 parameter table and to a named symbol in each of the five implementations — CPU twin, GDScript fallback, Godot shader, browser twin and WebGL2 parity proof."
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
            description={`${PARAM_TABLE.length} constants across ${paramGroups.length} groups. The cpu and gpu columns name the exact symbol in the TypeScript twin and the shader, so every value is checkable against the sources in the file browser below.`}
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
                {paramGroups.map(([group, rows]) => (
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
            eyebrow="§27 · §31 · five implementations, one field"
            title="Parity contract & Apple-Silicon constraints"
            description="The same equations exist five times — C++ twin, GDScript fallback, Godot/Metal shader, browser twin and the WebGL2 proof on this page. The contract: double globals, fp32 chunk-locals, strict uint32 hashing."
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
            title="The complete Godot 4 project"
            description="Every file below is real and served straight from godot-project/ — the Metal-class raymarch shader, the GDScripts, the C++ GDExtension CPU twin and the SConstruct build. This is the exact code the parity table audits."
          />
          <div className="mt-8">
            <FileBrowser />
          </div>
        </div>
      </FadeIn>

      {/* 9 — footer (mt-auto keeps it pinned to the bottom) ------------------- */}
      <footer className="mt-auto border-t border-stone-800 bg-stone-950">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 font-mono text-[10px] leading-relaxed text-stone-500 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <span>
            W(x,y) Continuous World Engine — generated for Apple Silicon · Godot 4
            Forward+ · Metal TBDR · SDFGI · seed 1337
          </span>
          <span className="text-stone-600">
            CPU twin: fp64 · GPU: fp32 + dyadic origin · GROUND_BIAS 0.08 m
          </span>
        </div>
      </footer>
    </div>
  );
}
