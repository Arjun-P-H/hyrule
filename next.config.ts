import type { NextConfig } from "next";
import path from "path";
import withBundleAnalyzer from "@next/bundle-analyzer";

const nextConfig: NextConfig = {
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

