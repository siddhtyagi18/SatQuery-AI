// components/AnalysisSummary.tsx
// Prominent, well-typeset AI analysis summary answer.
'use client';

import { CornerFrame } from '@/components/ui/CornerFrame';
import { Sparkles, Terminal, FileText, CheckCircle2 } from 'lucide-react';
import type { TaskType } from '@/lib/types/analysis';

interface AnalysisSummaryProps {
  answerText: string | null;
  detectedTasks?: TaskType[];
  createdAt?: string;
  className?: string;
}

export function AnalysisSummary({ answerText, detectedTasks, createdAt, className }: AnalysisSummaryProps) {
  if (!answerText) {
    return (
      <CornerFrame label="ANALYSIS SYNTHESIS" className={className}>
        <div className="panel p-6 flex flex-col gap-3 items-center justify-center min-h-[140px] text-center">
          <Terminal className="w-6 h-6 text-[var(--text-faint)]" />
          <span className="text-xs font-mono text-[var(--text-muted)]">
            Awaiting execution synthesis...
          </span>
        </div>
      </CornerFrame>
    );
  }

  // Format paragraphs nicely
  const paragraphs = answerText.split('\n\n').filter(Boolean);

  return (
    <CornerFrame label="ANALYSIS SYNTHESIS" className={className}>
      <div className="panel p-6 flex flex-col gap-4">
        <div className="flex items-center justify-between gap-2 border-b border-[var(--border-hairline)] pb-3">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded flex items-center justify-center bg-[var(--accent-signal)]/10 border border-[var(--accent-signal)]/20">
              <Sparkles className="w-3.5 h-3.5 text-[var(--accent-signal)]" />
            </span>
            <h2 className="text-base font-semibold text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
              Executive Remote Sensing Report
            </h2>
          </div>

          {detectedTasks && detectedTasks.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              {detectedTasks.map((t) => {
                const isChangeTask = t.toLowerCase().includes('change') || t === 'change_detection';
                return (
                  <span
                    key={t}
                    className={`badge ${isChangeTask ? 'badge-magenta' : 'badge-cyan'}`}
                  >
                    {t.replace('_', ' ')}
                  </span>
                );
              })}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3 text-sm leading-relaxed text-[var(--text-primary)]">
          {paragraphs.map((para, idx) => (
            <p key={idx} className="tracking-wide">
              {para}
            </p>
          ))}
        </div>

        <div className="flex items-center justify-between text-[0.65rem] font-mono text-[var(--text-faint)] pt-2 border-t border-[var(--border-hairline)]">
          <div className="flex items-center gap-1.5 text-[var(--accent-success)]">
            <CheckCircle2 className="w-3 h-3" />
            <span>Verified by Specialist Ensemble</span>
          </div>
          {createdAt && (
            <span>Generated: {new Date(createdAt).toLocaleString('en-IN', { hour12: false })}</span>
          )}
        </div>
      </div>
    </CornerFrame>
  );
}
