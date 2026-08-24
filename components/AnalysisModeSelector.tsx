// components/AnalysisModeSelector.tsx
// Three large selectable cards — mode selector for New Analysis flow.
// STRONG SELECTED STATE: 2px border + top accent bar + tint + glow.
// Unselected cards recede (muted border + faded icon) for maximum contrast.
'use client';

import { cn } from '@/lib/utils';
import type { AnalysisMode } from '@/lib/types/analysis';
import { Layers, GitCompare, Radar } from 'lucide-react';

type DomainColor = 'cyan' | 'magenta' | 'amber';

interface ModeDef {
  value: AnalysisMode;
  icon: React.ElementType;
  label: string;
  description: string;
  tasks: string[];
  domain: DomainColor;
}

const MODES: ModeDef[] = [
  {
    value: 'single_image',
    icon: Layers,
    label: 'Single Image',
    description: 'Analyse a single optical, multispectral, or SAR image with natural language queries.',
    tasks: ['Visual Q&A', 'Image Captioning', 'Object Grounding'],
    domain: 'cyan',
  },
  {
    value: 'bi_temporal',
    icon: GitCompare,
    label: 'Bi-Temporal Pair',
    description: 'Compare a before/after image pair to detect and analyse land-use changes over time.',
    tasks: ['Change Detection', 'Change VQA', 'Change Description'],
    domain: 'magenta',
  },
  {
    value: 'optical_sar',
    icon: Radar,
    label: 'Optical + SAR',
    description: 'Fuse co-registered optical and SAR imagery for cross-modal analysis and validation.',
    tasks: ['Cross-modal Fusion', 'SAR Feature Analysis', 'Multi-modal VQA'],
    domain: 'amber',
  },
];

const GLOW_CLASS: Record<DomainColor, string> = {
  cyan:    'panel-selected-cyan',
  magenta: 'panel-selected-magenta',
  amber:   'panel-selected-amber',
};

const TAG_CLASS: Record<DomainColor, string> = {
  cyan:    'badge-cyan',
  magenta: 'badge-magenta',
  amber:   'badge-amber',
};

interface AnalysisModeSelectorProps {
  value: AnalysisMode | null;
  onChange: (mode: AnalysisMode) => void;
  disabled?: boolean;
}

export function AnalysisModeSelector({ value, onChange, disabled }: AnalysisModeSelectorProps) {
  return (
    <fieldset className="border-0 p-0 m-0 flex flex-col gap-3" disabled={disabled} aria-label="Select analysis mode">
      <legend className="hud-label">Analysis Mode</legend>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {MODES.map((mode) => {
          const Icon = mode.icon;
          const isSelected = value === mode.value;
          const domainVar = `var(--${mode.domain})`;

          return (
            <button
              key={mode.value}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(mode.value)}
              disabled={disabled}
              className={cn(
                'relative flex flex-col gap-4 p-5 text-left rounded transition-all duration-200 overflow-hidden focus-visible:outline-none',
                disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
                isSelected ? GLOW_CLASS[mode.domain] : '',
              )}
              style={{
                background: isSelected
                  ? `color-mix(in srgb, ${domainVar} 5%, var(--surface-2))`
                  : 'var(--surface-1)',
                border: isSelected
                  ? `2px solid ${domainVar}`
                  : '1px solid var(--border-hairline)',
                opacity: isSelected ? 1 : 0.82,
              }}
            >
              {/* SELECTED: Filled top accent bar (3-4px strip along top edge) */}
              {isSelected && (
                <div
                  className="top-accent-bar"
                  data-domain={mode.domain}
                  aria-hidden
                />
              )}

              {/* Hex icon container: SELECTED = filled/glowing, UNSELECTED = outline/muted */}
              <div
                className={cn(
                  'icon-hex',
                  isSelected ? '' : 'icon-hex-outline',
                )}
                data-domain={mode.domain}
                style={{
                  opacity: isSelected ? 1 : 0.7,
                }}
                aria-hidden
              >
                <Icon
                  className="w-5 h-5"
                  strokeWidth={1.8}
                  style={{
                    color: domainVar,
                    filter: isSelected ? `drop-shadow(0 0 4px ${domainVar})` : 'none',
                  }}
                />
              </div>

              {/* Text: SELECTED = heading in domain color, UNSELECTED = neutral heading */}
              <div className="flex flex-col gap-2">
                <span
                  className="text-base font-semibold"
                  style={{
                    fontFamily: 'var(--font-heading)',
                    color: isSelected ? domainVar : 'var(--text-primary)',
                    letterSpacing: isSelected ? '-0.01em' : '0',
                  }}
                >
                  {mode.label}
                </span>
                <span
                  className="text-xs leading-relaxed"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {mode.description}
                </span>
              </div>

              {/* Task chips: same badge class regardless, opacity shifts */}
              <div className="flex flex-wrap gap-1.5 mt-auto pt-1">
                {mode.tasks.map((task) => (
                  <span
                    key={task}
                    className={cn('badge', TAG_CLASS[mode.domain])}
                    style={{ opacity: isSelected ? 1 : 0.55 }}
                  >
                    {task}
                  </span>
                ))}
              </div>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
