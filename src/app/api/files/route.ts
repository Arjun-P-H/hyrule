import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Code-browser backend — serves the real Godot project sources.
 * GET /api/files            → { root, tree: [{ path, size, lines }] }
 * GET /api/files?path=...   → { path, content, lines, size }
 *
 * The root is locked to <project>/godot-project (path-traversal safe).
 */

const ROOT = path.join(process.cwd(), "godot-project");

const SKIP_DIRS = new Set([".godot", "godot-cpp", "bin", ".DS_Store"]);

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

export async function GET(request: Request) {
  const url = new URL(request.url);
  const want = url.searchParams.get("path");

  try {
    if (!want) {
      const files = (await walk(ROOT)).sort();
      const tree = await Promise.all(
        files.map(async (f) => {
          const stat = await fs.stat(path.join(ROOT, f));
          const content = await fs.readFile(path.join(ROOT, f), "utf-8");
          return {
            path: f,
            size: stat.size,
            lines: content.split("\n").length,
          };
        }),
      );
      return NextResponse.json({ root: "godot-project", tree });
    }

    // sanitize: resolve inside ROOT only
    const target = path.resolve(ROOT, want);
    if (!target.startsWith(ROOT + path.sep) && target !== ROOT) {
      return NextResponse.json({ error: "invalid path" }, { status: 400 });
    }
    const content = await fs.readFile(target, "utf-8");
    const stat = await fs.stat(target);
    return NextResponse.json({
      path: want,
      content,
      lines: content.split("\n").length,
      size: stat.size,
    });
  } catch {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }
}
