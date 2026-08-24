// components/ui/ModeBadge.tsx
// Mode badge — one shape language (rounded-sm 4px), domain color only.
// Cyan = Single Image, Magenta = Bi-Temporal, Amber = Optical+SAR
'use client';

import { cn } from '@/lib/utils';
import type { AnalysisMode } from '@/lib/types/analysis';
import { Layers, GitCompare, Radar } from 'lucide-react';

type DomainColor = 'cyan' | 'magenta' | 'amber';

const MODE_CONFIG: Record<AnalysisMode, { label: string; icon: React.ElementType; domain: DomainColor }> = {
  single_image: {
    label: 'Single Image',
    icon: Layers,
    domain: 'cyan',
  },
  bi_temporal: {
    label: 'Bi-Temporal',
    icon: GitCompare,
    domain: 'magenta',
  },
  optical_sar: {
    label: 'Optical+SAR',
    icon: Radar,
    domain: 'amber',
  },
};

const BADGE_DOMAIN_CLASS: Record<DomainColor, string> = {
  cyan:    'badge-cyan',
  magenta: 'badge-magenta',
  amber:   'badge-amber',
};

interface ModeBadgeProps {
  mode: AnalysisMode;
  className?: string;
  showIcon?: boolean;
}

export function ModeBadge({ mode, className, showIcon = true }: ModeBadgeProps) {
  const config = MODE_CONFIG[mode];
  const Icon = config.icon;
  return (
    <span
      className={cn(
        'badge',
        BADGE_DOMAIN_CLASS[config.domain],
        className
      )}
    >
      {showIcon && <Icon className="w-2.5 h-2.5" strokeWidth={2.25} />}
      {config.label}
    </span>
  );
}
