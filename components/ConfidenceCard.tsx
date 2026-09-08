// components/ConfidenceCard.tsx
// Displays overall confidence with visual gauge and sub-task breakdown.
'use client';

import { ConfidenceGauge } from '@/components/ui/ConfidenceGauge';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { ShieldCheck, Info } from 'lucide-react';
import type { TaskType } from '@/lib/types/analysis';

interface BreakdownItem {
  label: string;
  score: number;
}

interface ConfidenceCardProps {
  score: number | null;
  detectedTasks?: TaskType[];
  breakdown?: BreakdownItem[];
  isMock?: boolean;
  className?: string;
}

export function ConfidenceCard({ score, detectedTasks, breakdown, isMock, className }: ConfidenceCardProps) {
  // Only display sub-claim breakdown if genuinely provided by calibrated specialist
  const items: BreakdownItem[] = breakdown ?? [];
  const isUncalibrated = score == null;

  const getConfidenceTier = (s: number | null) => {
    if (s != null) {
      if (s >= 0.85) {
        return {
          text: `High Confidence (${Math.round(s * 100)}%)`,
          color: 'var(--accent-success)',
          subtext: 'Calibrated model uncertainty estimate from authentic specialist execution.',
        };
      }
      if (s >= 0.70) {
        return {
          text: `Moderate Confidence (${Math.round(s * 100)}%)`,
          color: 'var(--accent-warning)',
          subtext: 'Calibrated model uncertainty estimate from authentic specialist execution.',
        };
      }
      return {
        text: `Low Confidence (${Math.round(s * 100)}%)`,
        color: 'var(--accent-danger)',
        subtext: 'Calibrated model uncertainty estimate indicates elevated prediction variance.',
      };
    }

    return {
      text: 'N/A — Uncalibrated',
      color: 'var(--accent-signal)',
      subtext: 'Authentic specialist model does not emit calibrated confidence scores; confidence remains null to maintain scientific integrity.',
    };
  };

  const tier = getConfidenceTier(score);

  return (
    <CornerFrame label="CONFIDENCE ASSESSMENT" className={className} domain={isUncalibrated ? 'amber' : 'green'}>
      <div className="panel p-5 flex flex-col gap-4 transition-all duration-300 hover:border-[var(--green)]/40 hover:shadow-lg">
        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <span className="hud-label">Overall Model Confidence</span>
            <span className="text-base font-semibold" style={{ color: tier.color, fontFamily: 'var(--font-heading)' }}>
              {tier.text}
            </span>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed max-w-[240px]" style={{ fontFamily: 'var(--font-body)' }}>
              {tier.subtext}
            </p>
          </div>
          {isUncalibrated ? (
            <div className="flex flex-col items-center justify-center p-3 rounded bg-[var(--surface-2)] border border-amber-500/30 text-center min-w-[90px]">
              <span className="text-xs font-mono font-bold text-amber-400">NULL</span>
              <span className="text-[0.6rem] font-mono text-[var(--text-faint)] uppercase tracking-wider">Uncalibrated</span>
            </div>
          ) : (
            <ConfidenceGauge score={score} size="md" />
          )}
        </div>

        {items.length > 0 && (
          <div className="flex flex-col gap-2.5 pt-3 border-t border-[var(--border-hairline)]">
            <span className="hud-label">Sub-claim Breakdown</span>
            <div className="flex flex-col gap-2.5">
              {items.map((item, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[var(--text-muted)] font-normal">{item.label}</span>
                    <span className="text-[var(--text-primary)] font-semibold">{(item.score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-[var(--bg-panel-elevated)] overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${item.score * 100}%`,
                        background: item.score >= 0.8 ? 'var(--accent-success)' : item.score >= 0.65 ? 'var(--accent-warning)' : 'var(--accent-danger)',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </CornerFrame>
  );
}
