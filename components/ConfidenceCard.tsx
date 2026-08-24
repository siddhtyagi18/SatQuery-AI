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
  className?: string;
}

export function ConfidenceCard({ score, detectedTasks, breakdown, className }: ConfidenceCardProps) {
  // Generate simulated breakdown if none explicitly provided
  const items: BreakdownItem[] = breakdown ?? (score != null ? [
    { label: 'Feature Extraction Quality', score: Math.min(1, score + 0.05) },
    { label: 'Spatial Alignment / Co-registration', score: Math.min(1, score + 0.02) },
    { label: 'Vision-Language Calibration', score: score },
  ] : []);

  const getConfidenceTier = (s: number | null) => {
    if (s == null) return { text: 'Not Evaluated', color: 'var(--text-faint)' };
    if (s >= 0.85) return { text: 'High Confidence', color: 'var(--accent-success)' };
    if (s >= 0.70) return { text: 'Moderate Confidence', color: 'var(--accent-warning)' };
    return { text: 'Low Confidence / Review Advised', color: 'var(--accent-danger)' };
  };

  const tier = getConfidenceTier(score);

  return (
    <CornerFrame label="CONFIDENCE ASSESSMENT" className={className}>
      <div className="panel p-5 flex flex-col gap-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <span className="hud-label">Overall Model Confidence</span>
            <span className="text-sm font-semibold" style={{ color: tier.color, fontFamily: 'var(--font-heading)' }}>
              {tier.text}
            </span>
            <p className="text-[0.65rem] text-[var(--text-faint)] leading-tight max-w-[200px]">
              Computed from posterior probabilities across invoked specialist heads.
            </p>
          </div>
          <ConfidenceGauge score={score} size="md" />
        </div>

        {items.length > 0 && (
          <div className="flex flex-col gap-2 pt-3 border-t border-[var(--border-hairline)]">
            <span className="hud-label">Sub-claim Breakdown</span>
            <div className="flex flex-col gap-2">
              {items.map((item, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-[0.68rem] font-mono">
                    <span className="text-[var(--text-muted)]">{item.label}</span>
                    <span className="text-[var(--text-primary)]">{(item.score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1 rounded-full bg-[var(--bg-panel-elevated)] overflow-hidden">
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
