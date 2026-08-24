// components/ui/EmptyState.tsx
// Designed empty state — never looks like a blank screen.
'use client';

import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon: Icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-4 py-16 px-8 text-center',
        className
      )}
    >
      {/* Icon with subtle radial glow */}
      <div className="relative flex items-center justify-center">
        <div
          className="absolute inset-0 rounded-full blur-xl"
          style={{ background: 'radial-gradient(circle, rgba(62,208,255,0.08) 0%, transparent 70%)' }}
          aria-hidden="true"
        />
        <div
          className="relative w-16 h-16 rounded-lg flex items-center justify-center"
          style={{
            background: 'var(--bg-panel-elevated)',
            border: '1px solid var(--border-hairline)',
          }}
        >
          <Icon
            className="w-7 h-7"
            style={{ color: 'var(--accent-signal)', opacity: 0.7 }}
            strokeWidth={1.5}
            aria-hidden="true"
          />
        </div>
      </div>

      <div className="flex flex-col gap-1.5 max-w-sm">
        <h3
          className="text-base font-semibold"
          style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}
        >
          {title}
        </h3>
        {description && (
          <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            {description}
          </p>
        )}
      </div>

      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
