// components/ui/StatusBadge.tsx
// Status badge — one shape language (rounded-sm 4px), color is ONLY variant.
// Green = success/completed, Amber = processing/in-progress, Cyan = queued/mock
// Red = error/failed, Neutral = pending/planned
'use client';

import { cn } from '@/lib/utils';
import type { AnalysisStatus, StepStatus } from '@/lib/types/analysis';

type BadgeVariant = AnalysisStatus | StepStatus | 'available' | 'mock' | 'planned' | 'operational';

type BadgeClass =
  | 'badge-green'
  | 'badge-amber'
  | 'badge-cyan'
  | 'badge-red'
  | 'badge-neutral';

const VARIANT_CLASS: Record<string, BadgeClass> = {
  completed:   'badge-green',
  done:        'badge-green',
  available:   'badge-green',
  operational: 'badge-green',
  processing:  'badge-amber',
  in_progress: 'badge-amber',
  queued:      'badge-cyan',
  mock:        'badge-cyan',
  pending:     'badge-neutral',
  planned:     'badge-neutral',
  failed:      'badge-red',
  error:       'badge-red',
};

const VARIANT_LABELS: Partial<Record<string, string>> = {
  in_progress: 'Processing',
  single_image: 'Single Image',
  bi_temporal: 'Bi-Temporal',
  optical_sar: 'Optical+SAR',
};

const DOT_VARIANTS = new Set(['processing', 'in_progress']);

interface StatusBadgeProps {
  status: string;
  label?: string;
  className?: string;
  showDot?: boolean;
}

export function StatusBadge({ status, label, className, showDot }: StatusBadgeProps) {
  const cls = VARIANT_CLASS[status] ?? 'badge-neutral';
  const displayLabel = label ?? VARIANT_LABELS[status] ?? status.charAt(0).toUpperCase() + status.slice(1);
  const shouldPulse = showDot !== false && DOT_VARIANTS.has(status);

  return (
    <span className={cn('badge', cls, className)}>
      {shouldPulse && (
        <span className="relative flex h-1.5 w-1.5 -ml-0.5">
          <span className="animate-pulse-dot absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      )}
      {displayLabel}
    </span>
  );
}
