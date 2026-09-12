'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { RotateCw, ArrowLeft, Satellite, AlertOctagon } from 'lucide-react';

export default function AnalysisErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[SatQuery-AI Analysis Subsystem Error]', error);
  }, [error]);

  return (
    <div className="min-h-[70vh] flex items-center justify-center p-6">
      <div
        className="w-full max-w-lg p-7 rounded-xl relative overflow-hidden text-center"
        style={{
          background: 'rgba(11, 19, 36, 0.88)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(56, 189, 248, 0.25)',
          boxShadow: '0 0 30px rgba(14, 165, 233, 0.12)',
        }}
      >
        <div
          className="w-14 h-14 mx-auto mb-4 rounded-xl flex items-center justify-center"
          style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
          }}
        >
          <Satellite className="w-7 h-7 text-red-400 animate-pulse" />
        </div>

        <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono tracking-wider uppercase mb-3 bg-red-950/50 text-red-400 border border-red-800/30">
          <AlertOctagon className="w-3 h-3" />
          Analysis Session Synchronized Safe Mode
        </div>

        <h2 className="text-lg font-bold text-slate-100 mb-2">
          Unable to Synthesize Target Orbital Pass
        </h2>

        <p className="text-xs text-slate-400 mb-6 max-w-sm mx-auto leading-relaxed">
          The requested analysis scene experienced a transient client data mismatch.
          You can re-trigger pipeline synchronization or launch a new analysis mission.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <button
            onClick={() => reset()}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all"
            style={{
              background: 'linear-gradient(135deg, #0284c7, #0369a1)',
              color: '#ffffff',
              boxShadow: '0 0 15px rgba(2, 132, 199, 0.3)',
            }}
          >
            <RotateCw className="w-3.5 h-3.5" />
            Retry Orbital Sync
          </button>

          <Link
            href="/analysis/new"
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            New Analysis
          </Link>
        </div>
      </div>
    </div>
  );
}
