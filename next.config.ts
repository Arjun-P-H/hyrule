import type { NextConfig } from "next";
import path from "path";
import withBundleAnalyzer from "@next/bundle-analyzer";

const nextConfig: NextConfig = {
  // `next dev` locks its build directory, and `next build` empties .next: a dev server's directory (and its Turbopack
  // cache) must not be inside it. scripts/dev-server.sh sets NEXT_DIST_DIR (.next-dev, -2, -3: servers side by side);
  // a bare `next dev` takes .next-dev too, and only a production build uses .next.
  distDir: process.env.NEXT_DIST_DIR || (process.env.NODE_ENV === "production" ? ".next" : ".next-dev"),
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname),
  reactStrictMode: true,
  poweredByHeader: false,
  compress: true,
  compiler: {
    removeConsole: process.env.NODE_ENV === "production" ? { exclude: ["error", "warn"] } : false,
  },
  experimental: {
    optimizePackageImports: ["lucide-react", "@radix-ui/react-tooltip"],
  },
};

export default withBundleAnalyzer({
  enabled: process.env.ANALYZE === "true",
  openAnalyzer: false,
})(nextConfig);

