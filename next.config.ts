import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  allowedDevOrigins: ['127.0.0.1', 'localhost'],
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL;
    // On Vercel without an external backend, do not rewrite to localhost or vercel itself
    if (process.env.VERCEL && (!backendUrl || backendUrl.includes('localhost') || backendUrl.includes('vercel.app'))) {
      return [];
    }
    const target = backendUrl || 'http://127.0.0.1:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${target}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
