import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Code-browser backend — serves the browser world engine sources.
 * GET /api/files            → { root, tree: [{ path, size, lines }] }
 * GET /api/files?path=...   → { path, content, lines, size }
 *
 * The root is locked to <project>/src (path-traversal safe).
 */

const ROOT = path.join(process.cwd(), "src");

const SKIP_DIRS = new Set([".next", "node_modules", ".git", ".DS_Store"]);

function countLines(buf: Buffer): number {
  if (buf.length === 0) return 0;
  let lines = 1;
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] === 10) lines++;
  }
  return lines;
}

async function walk(dir: string, base = ""): Promise<string[]> {
  const out: string[] = [];
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    if (e.name.startsWith(".") || SKIP_DIRS.has(e.name)) continue;
    const rel = base ? `${base}/${e.name}` : e.name;
    if (e.isDirectory()) {
      out.push(...(await walk(path.join(dir, e.name), rel)));
    } else {
      out.push(rel);
    }
  }
  return out;
}

let cachedTree: { path: string; size: number; lines: number }[] | null = null;
let lastCacheTime = 0;

export async function GET(request: Request) {
  const url = new URL(request.url);
  const want = url.searchParams.get("path");

  try {
    if (!want) {
      const now = Date.now();
      if (cachedTree && (process.env.NODE_ENV === "production" || now - lastCacheTime < 60000)) {
        return NextResponse.json(
          { root: "src", tree: cachedTree },
          {
            headers: {
              "Cache-Control": "public, max-age=60, stale-while-revalidate=300",
            },
          }
        );
      }

      const files = (await walk(ROOT)).sort();
      const tree = await Promise.all(
        files.map(async (f) => {
          const fullPath = path.join(ROOT, f);
          const [stat, buf] = await Promise.all([
            fs.stat(fullPath),
            fs.readFile(fullPath),
          ]);
          return {
            path: f,
            size: stat.size,
            lines: countLines(buf),
          };
        }),
      );
      cachedTree = tree;
      lastCacheTime = now;
      return NextResponse.json(
        { root: "src", tree },
        {
          headers: {
            "Cache-Control": "public, max-age=60, stale-while-revalidate=300",
          },
        }
      );
    }

    // sanitize: resolve inside ROOT only
    const target = path.resolve(ROOT, want);
    if (!target.startsWith(ROOT + path.sep) && target !== ROOT) {
      return NextResponse.json({ error: "invalid path" }, { status: 400 });
    }
    const content = await fs.readFile(target, "utf-8");
    const stat = await fs.stat(target);
    return NextResponse.json(
      {
        path: want,
        content,
        lines: content.split("\n").length,
        size: stat.size,
      },
      {
        headers: {
          "Cache-Control": "public, max-age=60, stale-while-revalidate=300",
        },
      }
    );
  } catch {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
}
