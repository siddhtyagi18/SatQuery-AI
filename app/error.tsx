'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { AlertTriangle, RefreshCw, Home, ShieldAlert } from 'lucide-react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log telemetry/diagnostic error to console
    console.error('[SatQuery-AI Telemetry Crash Protected]', error);
  }, [error]);

  const handleResetCache = () => {
    try {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('satquery_analyses_store');
        localStorage.removeItem('satquery_uploads_store');
      }
    } catch {
      // Ignore storage errors
    }
    window.location.href = '/analysis/new';
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ background: 'var(--bg-space, #070b12)' }}>
      <div
        className="w-full max-w-lg p-8 rounded-xl relative overflow-hidden"
        style={{
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(239, 68, 68, 0.35)',
          boxShadow: '0 0 35px rgba(239, 68, 68, 0.15)',
        }}
      >
        {/* Glow accent */}
        <div
          className="absolute -top-24 -right-24 w-48 h-48 rounded-full blur-3xl pointer-events-none"
          style={{ background: 'rgba(239, 68, 68, 0.2)' }}
        />

        <div className="flex items-center gap-3 mb-6">
          <div
            className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid rgba(239, 68, 68, 0.4)',
            }}
          >
            <ShieldAlert className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono tracking-widest uppercase px-2 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-800/40">
                SYSTEM SAFEGUARD ENGAGED
              </span>
            </div>
            <h1 className="text-xl font-bold text-slate-100 tracking-wide mt-1">
              SatQuery Mission Telemetry Interrupted
            </h1>
          </div>
        </div>

        <p className="text-sm text-slate-400 mb-6 leading-relaxed">
          An unexpected interface exception occurred during client raster synthesis.
          Mission control has protected active orbital parameters and isolated the state.
        </p>

        {error.message && (
          <div
            className="p-3.5 rounded-lg mb-6 text-xs font-mono text-slate-300 break-words"
            style={{
              background: 'rgba(0, 0, 0, 0.5)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
            }}
          >
            <span className="text-red-400 font-semibold">SIG: </span>
            {error.message.length > 180 ? `${error.message.slice(0, 180)}...` : error.message}
          </div>
        )}

        <div className="flex flex-col sm:flex-row items-center gap-3">
          <button
            onClick={() => reset()}
            className="w-full sm:w-auto flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium text-xs tracking-wider uppercase transition-all"
            style={{
              background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
              color: '#ffffff',
              boxShadow: '0 0 16px rgba(14, 165, 233, 0.35)',
            }}
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Resume Pipeline
          </button>

          <button
            onClick={handleResetCache}
            className="w-full sm:w-auto flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium text-xs tracking-wider uppercase transition-all bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700/60"
          >
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            Clear & Reset Session
          </button>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-500 font-mono">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 hover:text-cyan-400 transition-colors"
          >
            <Home className="w-3.5 h-3.5" />
            Return to Mission Terminal
          </Link>
          <span>SatQuery v2.4 (Resilient)</span>
        </div>
      </div>
    </div>
  );
}
