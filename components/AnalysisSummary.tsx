// components/AnalysisSummary.tsx
// Prominent, well-typeset AI analysis summary answer with rich section
// formatting, callout cards, copy-to-clipboard, and refined hover states.
'use client';

import { useState } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import {
  Sparkles,
  Terminal,
  FileText,
  CheckCircle2,
  Copy,
  Check,
  Layers,
  Info,
  ShieldCheck,
} from 'lucide-react';
import type { TaskType } from '@/lib/types/analysis';

interface AnalysisSummaryProps {
  answerText: string | null;
  detectedTasks?: TaskType[];
  createdAt?: string;
  className?: string;
}

/** Helper to parse inline markdown bold (**text**) and italics (*text*) */
function formatInlineText(text: string) {
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      const inner = part.slice(2, -2);
      return (
        <strong key={i} className="font-semibold text-[var(--text-primary)]">
          {inner}
        </strong>
      );
    }
    if (part.startsWith('*') && part.endsWith('*')) {
      const inner = part.slice(1, -1);
      return (
        <em key={i} className="italic text-[var(--text-muted)] font-mono text-xs">
          {inner}
        </em>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function AnalysisSummary({
  answerText,
  detectedTasks,
  createdAt,
  className,
}: AnalysisSummaryProps) {
  const [copied, setCopied] = useState(false);

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

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(answerText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback if clipboard permissions are restricted
    }
  };

  // Split raw text into paragraphs
  const rawParagraphs = answerText.split(/\n\s*\n/).filter(Boolean);

  return (
    <CornerFrame label="ANALYSIS SYNTHESIS" className={className}>
      <div className="panel p-6 flex flex-col gap-5 transition-all duration-300 hover:border-[var(--cyan)]/35 hover:shadow-lg">
        {/* Header toolbar */}
        <div className="flex items-center justify-between gap-3 border-b border-[var(--border-hairline)] pb-3.5 flex-wrap">
          <div className="flex items-center gap-2.5">
            <span className="w-7 h-7 rounded flex items-center justify-center bg-[var(--cyan)]/15 border border-[var(--cyan)]/30 shadow-[0_0_12px_rgba(56,189,248,0.2)]">
              <Sparkles className="w-4 h-4 text-[var(--cyan)]" />
            </span>
            <div>
              <h2
                className="text-base font-semibold text-[var(--text-primary)] leading-tight"
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                Executive Remote Sensing Report
              </h2>
              <span className="hud-label" style={{ fontSize: '0.58rem' }}>
                Multi-Sensor Specialist Synthesis
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Detected task badges */}
            {detectedTasks && detectedTasks.length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                {detectedTasks.map((t) => {
                  const isChangeTask =
                    t.toLowerCase().includes('change') || t === 'change_detection';
                  return (
                    <span
                      key={t}
                      className={`badge ${isChangeTask ? 'badge-magenta' : 'badge-cyan'}`}
                    >
                      {t.replace(/_/g, ' ')}
                    </span>
                  );
                })}
              </div>
            )}

            {/* Copy report button */}
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-[var(--surface-2)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--cyan)]/40 transition-colors"
              title="Copy Full Report to Clipboard"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-[var(--green)]" />
                  <span className="text-[var(--green)] font-medium">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Structured Body Content */}
        <div className="flex flex-col gap-4 text-[14.5px] leading-relaxed text-[var(--text-primary)] font-body">
          {rawParagraphs.map((para, idx) => {
            const trimmed = para.trim();

            // 1. Compatibility & Domain Gate notice banner
            if (
              trimmed.startsWith('[CHANGE DETECTION NOTICE:') ||
              trimmed.startsWith('[COMPATIBILITY NOTICE:') ||
              trimmed.startsWith('[UNSUPPORTED')
            ) {
              return (
                <div
                  key={idx}
                  className="p-4 rounded bg-[rgba(255,176,32,0.08)] border border-[rgba(255,176,32,0.35)] flex flex-col gap-2 text-xs font-mono text-[var(--text-primary)] leading-relaxed shadow-sm"
                >
                  <div className="flex items-center gap-2 text-[var(--accent-warning)] font-semibold text-xs tracking-wider uppercase">
                    <Info className="w-4 h-4 flex-shrink-0" />
                    <span>Specialist Domain Compatibility Notice</span>
                  </div>
                  <div className="whitespace-pre-line text-sm text-[var(--text-primary)] font-body">
                    {formatInlineText(trimmed)}
                  </div>
                </div>
              );
            }

            // 1b. Mock or Execution Provenance header banner
            if (trimmed.startsWith('[MOCK') || trimmed.startsWith('[AUTO') || trimmed.startsWith('[REAL')) {
              return (
                <div
                  key={idx}
                  className="px-3.5 py-2.5 rounded bg-[var(--surface-2)] border border-[var(--border-hairline)] flex items-start gap-2.5 text-xs font-mono text-[var(--text-muted)] leading-relaxed"
                >
                  <ShieldCheck className="w-4 h-4 text-[var(--cyan)] flex-shrink-0 mt-0.5" />
                  <div>
                    {formatInlineText(trimmed)}
                  </div>
                </div>
              );
            }

            // 2. Section Headers (e.g. ## Change Detection Report...)
            if (trimmed.startsWith('## ') || trimmed.startsWith('### ')) {
              const title = trimmed.replace(/^#+\s*/, '');
              const isChange = title.toLowerCase().includes('change');
              return (
                <div
                  key={idx}
                  className="flex items-center gap-2.5 pt-2 pb-1 border-b border-[var(--border-hairline)]"
                >
                  <span
                    className={`w-5 h-5 rounded flex items-center justify-center ${
                      isChange ? 'bg-[var(--magenta)]/15 text-[var(--magenta)]' : 'bg-[var(--cyan)]/15 text-[var(--cyan)]'
                    }`}
                  >
                    <Layers className="w-3 h-3" />
                  </span>
                  <h3
                    className="text-sm font-semibold tracking-tight text-[var(--text-primary)] font-heading"
                  >
                    {title}
                  </h3>
                </div>
              );
            }

            // 3. Bullet list block (contains • or - )
            if (trimmed.includes('•') || trimmed.startsWith('- ')) {
              const lines = trimmed.split('\n');
              return (
                <div key={idx} className="flex flex-col gap-2">
                  {lines.map((line, lIdx) => {
                    const lineTrimmed = line.trim();
                    if (lineTrimmed.startsWith('•') || lineTrimmed.startsWith('-')) {
                      const itemText = lineTrimmed.replace(/^[•\-]\s*/, '');
                      return (
                        <div
                          key={lIdx}
                          className="flex items-start gap-2.5 pl-1 py-1 rounded transition-colors hover:bg-[var(--surface-2)]/60"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--cyan)] mt-2 flex-shrink-0 shadow-[0_0_6px_var(--cyan)]" />
                          <p className="text-[14px] text-[var(--text-primary)] leading-relaxed">
                            {formatInlineText(itemText)}
                          </p>
                        </div>
                      );
                    }
                    return (
                      <p key={lIdx} className="font-normal text-[var(--text-primary)] leading-relaxed">
                        {formatInlineText(lineTrimmed)}
                      </p>
                    );
                  })}
                </div>
              );
            }

            // 4. Highlighted Callouts (e.g. **Model details:** or **Interpretation:**)
            if (trimmed.startsWith('**Model details:**') || trimmed.startsWith('**Interpretation:**') || trimmed.startsWith('**Note:**')) {
              return (
                <div
                  key={idx}
                  className="p-3.5 rounded bg-[var(--surface-2)] border-l-2 border-l-[var(--cyan)] border-y border-r border-[var(--border-hairline)] flex items-start gap-2.5 text-xs font-mono text-[var(--text-muted)] leading-relaxed"
                >
                  <Info className="w-4 h-4 text-[var(--cyan)] flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    {formatInlineText(trimmed)}
                  </div>
                </div>
              );
            }

            // 5. Verification Footnote (e.g. *Analysis performed by...*)
            if (trimmed.startsWith('*Analysis performed') || trimmed.startsWith('^Analysis')) {
              return (
                <div
                  key={idx}
                  className="pt-2 text-xs font-mono text-[var(--text-faint)] italic flex items-center gap-1.5"
                >
                  <span>{trimmed.replace(/^[*^]+|[*^]+$/g, '')}</span>
                </div>
              );
            }

            // 6. Regular narrative paragraph
            return (
              <p
                key={idx}
                className="font-normal leading-relaxed text-[var(--text-primary)]"
              >
                {formatInlineText(trimmed)}
              </p>
            );
          })}
        </div>

        {/* Footer Provenance */}
        <div className="flex items-center justify-between text-xs font-mono text-[var(--text-muted)] pt-3.5 border-t border-[var(--border-hairline)] flex-wrap gap-2">
          <div className="flex items-center gap-1.5 text-[var(--green)] font-medium">
            <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" />
            <span>Verified by Specialist Ensemble</span>
          </div>
          {createdAt && (
            <span className="text-[var(--text-faint)]">
              Generated: {new Date(createdAt).toLocaleString('en-IN', { hour12: false })}
            </span>
          )}
        </div>
      </div>
    </CornerFrame>
  );
}
