import path from 'path';
import type { NextConfig } from "next";
import type { Configuration, Resolve } from 'webpack';

const nextConfig: NextConfig = {
  /* config options here */
  // Set turbopack root to this frontend directory to avoid workspace root inference
  turbopack: {
    root: __dirname,
  },
  // Add webpack resolve aliases to force module resolution to local node_modules
  webpack: (config: Configuration): Configuration => {
    // Ensure `resolve` has the correct type for IDE support and avoid implicit `any`
    config.resolve = (config.resolve ?? {}) as Resolve;
    config.resolve.alias = {
      ...(config.resolve.alias ?? {}),
      'lucide-react': path.resolve(__dirname, 'node_modules', 'lucide-react'),
      'tailwindcss': path.resolve(__dirname, 'node_modules', 'tailwindcss'),
    };
    return config;
  },
};

export default nextConfig;
