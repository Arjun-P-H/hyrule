/**
 * First-load JavaScript per route, measured from what headless Chrome actually
 * downloads from the production standalone server (gzip on), not from build
 * manifests. `@next/bundle-analyzer` is wired in next.config.ts but only works
 * with webpack builds; Next 16 builds with Turbopack by default, so it never
 * runs. "First load" = scripts referenced by the server HTML; "lazy" = scripts
 * fetched after hydration (next/dynamic chunks) within the settle window.
 */

import { launchChrome, connectToPage } from './lib/chrome';

export type RouteBundle = {
  route: string;
  firstLoad: { files: number; rawBytes: number; gzipBytes: number };
  lazy: { files: number; rawBytes: number; gzipBytes: number };
  largest: Array<{ file: string; rawBytes: number; gzipBytes: number; firstLoad: boolean }>;
};

type ResEntry = { name: string; initiatorType: string; transferSize: number; encodedBodySize: number; decodedBodySize: number };

export async function measureRoutes(baseUrl: string, routes: string[]): Promise<RouteBundle[]> {
  const out: RouteBundle[] = [];
  for (const route of routes) {
    const html = await (await fetch(baseUrl + route)).text();
    const firstLoadSrcs = new Set(
      [...html.matchAll(/<script[^>]+src="([^"]+\.js[^"]*)"/g)].map((m) => new URL(m[1], baseUrl).pathname),
    );
    const chrome = await launchChrome({ url: 'about:blank', remoteDebugging: true });
    try {
      const cdp = await connectToPage(chrome.wsEndpoint!);
      await cdp.send('Network.enable');
      await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
      await cdp.send('Page.enable');
      const loaded = new Promise<void>((r) => cdp.on('Page.loadEventFired', () => r()));
      await cdp.send('Page.navigate', { url: baseUrl + route });
      await loaded;
      await Bun.sleep(2500); // let next/dynamic chunks settle
      const { result } = await cdp.send<{ result: { value: ResEntry[] } }>('Runtime.evaluate', {
        expression: `performance.getEntriesByType('resource').map(e => ({ name: e.name, initiatorType: e.initiatorType, transferSize: e.transferSize, encodedBodySize: e.encodedBodySize, decodedBodySize: e.decodedBodySize }))`,
        returnByValue: true,
      });
      cdp.close();
      const js = result.value.filter((e) => new URL(e.name).pathname.endsWith('.js'));
      const agg = { first: { files: 0, rawBytes: 0, gzipBytes: 0 }, lazy: { files: 0, rawBytes: 0, gzipBytes: 0 } };
      const largest: RouteBundle['largest'] = [];
      for (const e of js) {
        const p = new URL(e.name).pathname;
        const isFirst = firstLoadSrcs.has(p);
        const bucket = isFirst ? agg.first : agg.lazy;
        bucket.files++;
        bucket.rawBytes += e.decodedBodySize;
        bucket.gzipBytes += e.encodedBodySize;
        largest.push({ file: p.split('/').pop()!, rawBytes: e.decodedBodySize, gzipBytes: e.encodedBodySize, firstLoad: isFirst });
      }
      largest.sort((a, b) => b.rawBytes - a.rawBytes);
      out.push({ route, firstLoad: agg.first, lazy: agg.lazy, largest: largest.slice(0, 6) });
    } finally {
      await chrome.close();
    }
  }
  return out;
}
