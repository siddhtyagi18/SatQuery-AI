// components/ConfidenceCard.tsx
// Displays overall confidence with visual gauge and sub-task breakdown.
'use client';

import { useMemo } from 'react';
import { ConfidenceGauge } from '@/components/ui/ConfidenceGauge';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { ShieldCheck, Activity } from 'lucide-react';
import type { TaskType } from '@/lib/types/analysis';

export interface BreakdownItem {
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
  const isCalibrated = score != null && !isNaN(score) && score > 0;

  const effectiveScore = useMemo(() => {
    if (isCalibrated) {
      return Math.min(1, Math.max(0, score!));
    }
    return null;
  }, [score, isCalibrated]);

  const effectiveBreakdown = useMemo((): BreakdownItem[] => {
    if (breakdown && breakdown.length > 0) {
      return breakdown;
    }
    return [];
  }, [breakdown]);

  const getConfidenceTier = (s: number | null) => {
    if (s == null) {
      return {
        text: 'N/A — Uncalibrated',
        color: 'var(--text-muted)',
        subtext: 'Model confidence is not calibrated for this analysis mode (confidence = null). Operational outputs reflect topological and feature measurements without probabilistic calibration.',
      };
    }
    if (s >= 0.85) {
      return {
        text: `High Confidence (${Math.round(s * 100)}%)`,
        color: 'var(--accent-success)',
        subtext: 'Calibrated certainty aggregated across sensor feature extraction, spatial alignment, and specialist inference.',
      };
    }
    if (s >= 0.70) {
      return {
        text: `Moderate Confidence (${Math.round(s * 100)}%)`,
        color: 'var(--accent-warning)',
        subtext: 'Calibrated certainty indicates acceptable agreement with minor observational variance.',
      };
    }
    return {
      text: `Low Confidence (${Math.round(s * 100)}%)`,
      color: 'var(--accent-danger)',
      subtext: 'Elevated prediction variance detected across specialist feature heads.',
    };
  };

  const tier = getConfidenceTier(effectiveScore);

  return (
    <CornerFrame label="CONFIDENCE ASSESSMENT" className={className} domain="green">
      <div className="panel p-5 flex flex-col gap-4 transition-all duration-300 hover:border-[var(--green)]/40 hover:shadow-lg">
        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="hud-label">Model Confidence</span>
              {isCalibrated ? (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[0.6rem] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 uppercase tracking-wider">
                  <ShieldCheck className="w-3 h-3 text-emerald-400" />
                  CALIBRATED
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[0.6rem] font-mono bg-zinc-800/80 text-zinc-400 border border-zinc-700/60 uppercase tracking-wider">
                  <Activity className="w-3 h-3 text-zinc-400" />
                  UNCALIBRATED
                </span>
              )}
            </div>
            <span className="text-base font-semibold font-mono" style={{ color: tier.color }}>
              {tier.text}
            </span>
            <p className="text-xs text-[var(--text-muted)] leading-relaxed max-w-[240px]" style={{ fontFamily: 'var(--font-body)' }}>
              {tier.subtext}
            </p>
          </div>
          <ConfidenceGauge score={effectiveScore} size="md" />
        </div>

        {effectiveBreakdown.length > 0 ? (
          <div className="flex flex-col gap-2.5 pt-3 border-t border-[var(--border-hairline)]">
            <div className="flex items-center justify-between">
              <span className="hud-label">Diagnostic Signals</span>
              <span className="text-[0.65rem] font-mono text-[var(--text-faint)] flex items-center gap-1">
                <Activity className="w-2.5 h-2.5 text-zinc-400" />
                Uncalibrated Heuristics
              </span>
            </div>
            <div className="flex flex-col gap-2.5">
              {effectiveBreakdown.map((item, i) => (
                <div key={i} className="flex flex-col gap-1.5 p-1 rounded transition-colors hover:bg-[var(--surface-2)]/50">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[var(--text-muted)] font-normal">{item.label}</span>
                    <span className="text-[var(--text-primary)] font-semibold font-mono">{(item.score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-[var(--surface-2)] overflow-hidden border border-[var(--border-subtle)]">
                    <div
                      className="h-full rounded-full transition-all duration-700 ease-out"
                      style={{
                        width: `${item.score * 100}%`,
                        background:
                          item.score >= 0.85
                            ? 'linear-gradient(90deg, #10B981, #34D399)'
                            : item.score >= 0.70
                            ? 'linear-gradient(90deg, #F59E0B, #FBBF24)'
                            : 'linear-gradient(90deg, #EF4444, #F87171)',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : !isCalibrated ? (
          <div className="pt-2 border-t border-[var(--border-hairline)] text-[0.68rem] font-mono text-[var(--text-faint)]">
            ℹ Probabilistic sub-claim breakdown unavailable for uncalibrated models.
          </div>
        ) : null}
      </div>
    </CornerFrame>
  );
}
