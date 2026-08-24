// components/ui/ErrorState.tsx
// Error panel — renders a clear failure state with reason and optional retry action.
'use client';

import { AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ErrorStateProps {
  title?: string;
  reason?: string | null;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  title = 'Analysis Failed',
  reason,
  onRetry,
  className,
}: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col gap-4 p-6 rounded-md',
        className
      )}
      style={{
        background: 'rgba(255,92,92,0.05)',
        border: '1px solid rgba(255,92,92,0.25)',
      }}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle
          className="w-5 h-5 mt-0.5 flex-shrink-0"
          style={{ color: 'var(--accent-danger)' }}
          aria-hidden="true"
        />
        <div className="flex flex-col gap-2 flex-1">
          <h3
            className="text-sm font-semibold"
            style={{ color: 'var(--accent-danger)', fontFamily: 'var(--font-heading)' }}
          >
            {title}
          </h3>
          {reason && (
            <p
              className="text-xs leading-relaxed font-mono"
              style={{ color: 'var(--text-muted)' }}
            >
              {reason}
            </p>
          )}
        </div>
      </div>

      {onRetry && (
        <button
          onClick={onRetry}
          className="self-start px-4 py-1.5 rounded text-xs font-mono font-medium uppercase tracking-wider transition-all hover:opacity-90 focus-visible:ring-1"
          style={{
            background: 'rgba(255,92,92,0.12)',
            border: '1px solid rgba(255,92,92,0.3)',
            color: 'var(--accent-danger)',
          }}
        >
          Retry Analysis
        </button>
      )}
    </div>
  );
}
