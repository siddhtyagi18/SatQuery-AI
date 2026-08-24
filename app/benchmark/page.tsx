// app/benchmark/page.tsx
// Benchmark & Evaluation Dashboard — conforming strictly to zero fabricated numbers.
'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { BenchmarkMetric } from '@/lib/types/analysis';
import { BenchmarkMetrics } from '@/components/BenchmarkMetrics';
import { Skeleton } from '@/components/ui/LoadingSkeletonPanel';
import { Award, RefreshCw } from 'lucide-react';

export default function BenchmarkPage() {
  const [metrics, setMetrics] = useState<BenchmarkMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getBenchmarkMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4 border-b border-[var(--border-hairline)] pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Award className="w-5 h-5 text-[var(--accent-signal)]" />
            <h1 className="text-xl md:text-2xl font-bold font-heading text-[var(--text-primary)]">
              Benchmark & Validation Protocol
            </h1>
          </div>
          <p className="text-xs text-[var(--text-muted)] font-mono">
            Standardised accuracy, visual grounding IoU, and bi-temporal change metrics across remote-sensing benchmarks.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="panel p-5 flex flex-col gap-3">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ))}
        </div>
      ) : (
        <BenchmarkMetrics metrics={metrics} />
      )}
    </div>
  );
}
