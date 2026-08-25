'use client';

/**
 * FileBrowser — client-side code browser for the real Godot 4 project.
 * ---------------------------------------------------------------------------
 * - GET /api/files            → { root, tree: [{ path, size, lines }] }
 * - GET /api/files?path=<rel> → { path, content, lines, size }
 *
 * Left pane: scrollable file list (max-h-96, thin custom scrollbar built from
 * Tailwind arbitrary values only — globals.css is off-limits).
 * Right pane: header (path, line/size badges, copy button) + line-numbered
 * <pre> with a very light comment tokenizer: HTML is escaped FIRST, then
 * comment tails are wrapped in a span (no syntax-highlighting dependency).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Copy, FileCode2, FolderOpen } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

type FileEntry = { path: string; size: number; lines: number };
type FileContent = { path: string; content: string; lines: number; size: number };

/** Thin scrollbar — Tailwind arbitrary values only (globals.css untouched). */
const THIN_SCROLL =
  '[scrollbar-width:thin] [scrollbar-color:#44403c_transparent] ' +
  '[&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent ' +
  '[&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-stone-700';

/** Extension → icon colour (amber = C/C++, emerald = shader/TS, …). */
function extColor(path: string): string {
  if (/\.(h|cpp)$/i.test(path)) return 'text-amber-400';
  if (/\.(gdshader|ts|tsx)$/i.test(path)) return 'text-emerald-400';
  if (/\.gd$/i.test(path)) return 'text-stone-400';
  if (/\.md$/i.test(path)) return 'text-stone-300';
  return 'text-stone-500';
}

function fmtKB(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} kB`;
}

/* -- very light, safe tokenizer: escape first, then wrap comment tails ---- */

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Comment marker per language family (markdown headings stay untouched). */
function commentRegex(path: string): RegExp | null {
  if (/\.(h|cpp|ts|tsx|js|mjs|gdshader|glsl)$/i.test(path)) return /\/\/.*$/;
  if (/\.gd$/i.test(path)) return /#.*$/;
  if (/\.(lua|sql)$/i.test(path)) return /--.*$/;
  return null;
}

function highlightLine(line: string, path: string): string {
  const esc = escapeHtml(line);
  const re = commentRegex(path);
  if (!re) return esc;
  const m = esc.match(re);
  if (!m || m.index === undefined) return esc;
  const i = m.index;
  return (
    esc.slice(0, i) +
    '<span class="text-stone-600">' +
    esc.slice(i) +
    '</span>'
  );
}

/* ------------------------------------------------------------------------- */

function ListSkeleton() {
  return (
    <div className="space-y-2 p-3">
      {Array.from({ length: 9 }).map((_, i) => (
        <div
          key={i}
          className="h-8 animate-pulse rounded bg-stone-800/70"
          style={{ width: `${62 + ((i * 29) % 36)}%` }}
        />
      ))}
    </div>
  );
}

function CodeSkeleton() {
  return (
    <div className="space-y-2.5 p-4">
      {Array.from({ length: 12 }).map((_, i) => (
        <div
          key={i}
          className="h-3 animate-pulse rounded bg-stone-800/70"
          style={{ width: `${34 + ((i * 17) % 62)}%` }}
        />
      ))}
    </div>
  );
}

export default function FileBrowser() {
  const [tree, setTree] = useState<FileEntry[] | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [active, setActive] = useState<FileContent | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [contentError, setContentError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const cache = useRef(new Map<string, FileContent>());
  const reqToken = useRef(0);

  const open = useCallback(async (path: string) => {
    setSelected(path);
    setContentError(null);
    const hit = cache.current.get(path);
    if (hit) {
      setActive(hit);
      return;
    }
    const tok = ++reqToken.current;
    setLoadingContent(true);
    try {
      const res = await fetch(`/api/files?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as FileContent;
      if (tok !== reqToken.current) return; // stale response — drop it
      cache.current.set(path, data);
      setActive(data);
    } catch (e) {
      if (tok === reqToken.current) {
        setContentError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (tok === reqToken.current) setLoadingContent(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/files');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { root: string; tree: FileEntry[] };
        if (cancelled) return;
        setTree(data.tree);
        const first =
          data.tree.find((f) => f.path === 'README.md') ?? data.tree[0];
        if (first) void open(first.path);
      } catch (e) {
        if (!cancelled) setTreeError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  const copy = async () => {
    if (!active) return;
    try {
      await navigator.clipboard.writeText(active.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard API unavailable — silently ignore */
    }
  };

  const highlighted = useMemo(() => {
    if (!active) return [];
    return active.content.split('\n').map((l) => highlightLine(l, active.path));
  }, [active]);

  if (treeError) {
    return (
      <Card className="border-stone-800 bg-stone-900/30 p-6">
        <p className="font-mono text-sm text-red-300">
          failed to list godot-project/ — {treeError}
        </p>
        <p className="mt-2 font-mono text-xs text-stone-500">
          GET /api/files did not answer with the file tree.
        </p>
      </Card>
    );
  }

  return (
    <Card className="gap-0 overflow-hidden border-stone-800 bg-stone-900/30 p-0">
      <div className="grid lg:grid-cols-[280px_1fr]">
        {/* LEFT — file tree ------------------------------------------------ */}
        <div className="border-b border-stone-800 lg:border-b-0 lg:border-r">
          <div className="flex items-center justify-between gap-2 border-b border-stone-800/70 px-3 py-2">
            <span className="flex min-w-0 items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-stone-500">
              <FolderOpen className="size-3.5 shrink-0 text-amber-400" />
              <span className="truncate">godot-project</span>
            </span>
            {tree && (
              <span className="shrink-0 font-mono text-[10px] text-stone-600">
                {tree.length} files
              </span>
            )}
          </div>

          <div className={`max-h-96 overflow-y-auto ${THIN_SCROLL}`}>
            {!tree ? (
              <ListSkeleton />
            ) : tree.length === 0 ? (
              <p className="p-4 font-mono text-xs text-stone-500">
                no files found
              </p>
            ) : (
              tree.map((f) => {
                const isSel = selected === f.path;
                return (
                  <button
                    key={f.path}
                    type="button"
                    onClick={() => void open(f.path)}
                    aria-current={isSel ? 'true' : undefined}
                    className={`flex min-h-9 w-full items-center gap-2 border-l-2 px-3 py-2 text-left transition-colors ${
                      isSel
                        ? 'border-emerald-500 bg-emerald-950/40'
                        : 'border-transparent hover:bg-stone-800/40'
                    }`}
                  >
                    <FileCode2
                      className={`size-3.5 shrink-0 ${extColor(f.path)}`}
                    />
                    <span className="min-w-0 flex-1">
                      <span
                        className={`block truncate text-xs ${
                          isSel ? 'text-emerald-200' : 'text-stone-300'
                        }`}
                        title={f.path}
                      >
                        {f.path}
                      </span>
                      <span className="block font-mono text-[10px] text-stone-500">
                        {fmtKB(f.size)} · {f.lines} lines
                      </span>
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* RIGHT — viewer -------------------------------------------------- */}
        <div className="min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-800/70 px-3 py-2">
            <span
              className="min-w-0 truncate font-mono text-[11px] text-emerald-300"
              title={selected ?? undefined}
            >
              {selected ?? '—'}
            </span>
            <div className="flex shrink-0 items-center gap-1.5">
              {active && (
                <>
                  <Badge
                    variant="outline"
                    className="border-stone-700 bg-transparent font-mono text-[10px] text-stone-400"
                  >
                    {active.lines} lines
                  </Badge>
                  <Badge
                    variant="outline"
                    className="border-stone-700 bg-transparent font-mono text-[10px] text-stone-400"
                  >
                    {fmtKB(active.size)}
                  </Badge>
                </>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() => void copy()}
                disabled={!active}
                className="h-7 gap-1.5 border-stone-700 bg-transparent px-2 font-mono text-[10px] text-stone-300 hover:bg-stone-800 hover:text-emerald-300"
              >
                {copied ? (
                  <Check className="size-3 text-emerald-400" />
                ) : (
                  <Copy className="size-3" />
                )}
                {copied ? 'copied' : 'copy'}
              </Button>
            </div>
          </div>

          <div
            className={`max-h-[560px] overflow-auto bg-stone-950 ${THIN_SCROLL}`}
          >
            {contentError ? (
              <div className="p-6">
                <p className="font-mono text-xs text-red-300">
                  failed to load {selected} — {contentError}
                </p>
              </div>
            ) : !active ? (
              <CodeSkeleton />
            ) : (
              <pre
                className="w-max min-w-full py-3 font-mono text-[11px] leading-relaxed text-stone-300"
                style={{ tabSize: 4 }}
              >
                {highlighted.map((html, i) => (
                  <span key={i} className="flex">
                    <span className="sticky left-0 z-10 w-12 shrink-0 select-none border-r border-stone-800/70 bg-stone-950 pr-2 text-right text-stone-600">
                      {i + 1}
                    </span>
                    <span
                      className="whitespace-pre pl-3 pr-4"
                      dangerouslySetInnerHTML={{
                        __html: html.length > 0 ? html : '\u00A0',
                      }}
                    />
                  </span>
                ))}
              </pre>
            )}
          </div>

          {loadingContent && active && (
            <div className="border-t border-stone-800/70 px-3 py-1.5 font-mono text-[10px] text-stone-500">
              loading {selected}…
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
