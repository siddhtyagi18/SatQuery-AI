// components/AgentExecutionTrace.tsx
// THE centerpiece component — vertical timeline stepper showing the live
// execution trace of an analysis pipeline. Drives real async state,
// not a CSS animation pretending to be data.
'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import type { ExecutionTrace, ExecutionStep } from '@/lib/types/analysis';
import { CheckCircle2, XCircle, Loader2, Circle, ChevronDown, ChevronUp, Clock } from 'lucide-react';
import { CornerFrame } from '@/components/ui/CornerFrame';

// --- Step icon ---
function StepIcon({ status }: { status: ExecutionStep['status'] }) {
  if (status === 'done')
    return (
      <CheckCircle2
        className="w-4 h-4 flex-shrink-0 animate-[trace-complete-pop_0.35s_ease-out]"
        style={{
          color: 'var(--accent-success)',
          filter: 'drop-shadow(0 0 4px rgba(61,220,132,0.5))',
        }}
        aria-label="Completed"
      />
    );
  if (status === 'error')
    return (
      <XCircle
        className="w-4 h-4 flex-shrink-0"
        style={{
          color: 'var(--accent-danger)',
          filter: 'drop-shadow(0 0 4px rgba(255,92,92,0.5))',
        }}
        aria-label="Error"
      />
    );
  if (status === 'in_progress')
    return (
      <span className="w-4 h-4 flex-shrink-0 relative flex items-center justify-center" aria-label="In progress">
        <span
          className="absolute inset-0 rounded-full animate-ping opacity-60"
          style={{ background: 'var(--accent-warning)' }}
        />
        <span
          className="relative w-2.5 h-2.5 rounded-full shadow-[0_0_8px_var(--accent-warning)]"
          style={{ background: 'var(--accent-warning)' }}
        />
      </span>
    );
  return (
    <span className="w-4 h-4 flex-shrink-0 flex items-center justify-center" aria-label="Pending">
      <span className="w-2 h-2 rounded-full border border-[var(--text-faint)] opacity-35" />
    </span>
  );
}

// --- Single step row ---
function StepRow({ step, isLast }: { step: ExecutionStep; isLast: boolean }) {
  const isActive = step.status === 'in_progress';
  const isDone = step.status === 'done';
  const isError = step.status === 'error';
  const isPending = step.status === 'pending';

  const timeLabel = step.completedAt
    ? new Date(step.completedAt).toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : step.startedAt
    ? new Date(step.startedAt).toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : null;

  return (
    <li
      className={cn(
        'flex gap-3 animate-trace-step transition-all duration-300',
        isActive && 'bg-[var(--surface-2)]/70 border border-[var(--amber)]/30 p-2.5 rounded -mx-2 shadow-[0_0_16px_rgba(255,176,32,0.08)]',
        isPending ? 'opacity-40' : 'opacity-100'
      )}
      aria-current={isActive ? 'step' : undefined}
    >
      {/* Left: timestamp + connecting line */}
      <div className="flex flex-col items-center gap-1 pt-0.5">
        <StepIcon status={step.status} />
        {!isLast && (
          <div
            className={cn('w-px flex-1 min-h-[20px] transition-colors duration-300')}
            style={{
              background: isActive
                ? 'linear-gradient(180deg, var(--accent-warning), var(--border-hairline))'
                : isDone || isError
                ? 'var(--border-hairline)'
                : 'var(--border-subtle)',
            }}
            aria-hidden="true"
          />
        )}
      </div>

      {/* Right: content */}
      <div className="flex flex-col gap-1 pb-4 flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <span
            className={cn(
              'text-xs font-semibold flex items-center gap-1.5 transition-colors',
              isActive && 'tracking-wide'
            )}
            style={{
              fontFamily: 'var(--font-heading)',
              color: isActive ? 'var(--accent-warning)'
                : isError ? 'var(--accent-danger)'
                : isDone ? 'var(--text-primary)'
                : 'var(--text-faint)',
              textShadow: isActive ? '0 0 10px rgba(255,176,32,0.3)' : undefined,
            }}
          >
            {step.title}
            {isActive && (
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent-warning)] animate-ping" />
            )}
          </span>
          {timeLabel && (
            <span
              className="text-[0.6rem] font-mono flex items-center gap-1 flex-shrink-0"
              style={{ color: 'var(--text-faint)' }}
            >
              <Clock className="w-2.5 h-2.5" aria-hidden="true" />
              {timeLabel}
            </span>
          )}
        </div>

        {step.detail && !isPending && (
          <p
            className="text-[0.7rem] font-mono leading-relaxed break-all"
            style={{ color: isActive ? 'var(--text-primary)' : isError ? 'rgba(255,92,92,0.85)' : 'var(--text-muted)' }}
          >
            {step.detail}
          </p>
        )}

        {/* Key-value meta */}
        {step.meta && Object.keys(step.meta).length > 0 && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1">
            {Object.entries(step.meta).map(([k, v]) => (
              <span key={k} className="text-[0.6rem] font-mono" style={{ color: 'var(--text-faint)' }}>
                <span style={{ color: 'var(--accent-signal)', opacity: 0.8 }}>{k}</span>
                =
                <span style={{ color: 'var(--text-muted)' }}>{String(v)}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

interface AgentExecutionTraceProps {
  trace: ExecutionTrace;
  defaultExpanded?: boolean;
  className?: string;
}

export function AgentExecutionTrace({ trace, defaultExpanded = false, className }: AgentExecutionTraceProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  const doneCount = trace.steps.filter((s) => s.status === 'done' || s.status === 'error').length;
  const total = trace.steps.length;
  const currentStep = trace.steps.find((s) => s.status === 'in_progress');
  const hasError = trace.steps.some((s) => s.status === 'error');

  const summaryColor = hasError ? 'var(--accent-danger)'
    : trace.overallStatus === 'completed' ? 'var(--accent-success)'
    : trace.overallStatus === 'processing' ? 'var(--accent-warning)'
    : 'var(--text-muted)';

  return (
    <CornerFrame
      className={cn('rounded-md overflow-hidden', className)}
      label="EXECUTION TRACE"
      domain="cyan"
    >
      <div
        className="flex flex-col"
        style={{ background: 'var(--bg-panel)', border: '1px solid var(--border-hairline)' }}
      >
        {/* Collapsed summary bar — always visible */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center justify-between gap-3 px-4 py-3 w-full text-left transition-colors hover:bg-[var(--bg-panel-hover)]"
          aria-expanded={expanded}
          aria-controls="trace-steps"
        >
          <div className="flex items-center gap-2 min-w-0">
            {trace.overallStatus === 'processing' ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin flex-shrink-0" style={{ color: 'var(--accent-warning)' }} aria-hidden="true" />
            ) : (
              <span className="w-3.5 h-3.5 flex items-center justify-center flex-shrink-0">
                {hasError ? (
                  <XCircle className="w-3.5 h-3.5" style={{ color: 'var(--accent-danger)' }} aria-hidden="true" />
                ) : (
                  <CheckCircle2 className="w-3.5 h-3.5" style={{ color: summaryColor }} aria-hidden="true" />
                )}
              </span>
            )}

            <span className="text-[0.7rem] font-mono" style={{ color: summaryColor }}>
              {doneCount}/{total} steps
              {currentStep ? ` · ${currentStep.title}…` : trace.overallStatus === 'completed' ? ' · Complete' : hasError ? ' · Failed' : ''}
            </span>

            {trace.totalElapsedMs != null && (
              <span className="text-[0.6rem] font-mono" style={{ color: 'var(--text-faint)' }}>
                {(trace.totalElapsedMs / 1000).toFixed(2)}s
              </span>
            )}
          </div>

          <span style={{ color: 'var(--text-faint)' }} aria-hidden="true">
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </span>
        </button>

        {/* Progress bar */}
        <div className="h-[2px] w-full bg-[var(--border-subtle)] overflow-hidden" aria-hidden="true">
          <div
            className="h-full transition-all duration-500"
            style={{
              width: `${(doneCount / total) * 100}%`,
              background: hasError
                ? 'var(--accent-danger)'
                : trace.overallStatus === 'completed'
                ? 'var(--accent-success)'
                : 'var(--accent-warning)',
              boxShadow: hasError
                ? '0 0 8px var(--accent-danger)'
                : trace.overallStatus === 'completed'
                ? '0 0 8px var(--accent-success)'
                : '0 0 8px var(--accent-warning)',
            }}
          />
        </div>

        {/* Expanded steps */}
        {expanded && (
          <div id="trace-steps" className="px-4 pt-4">
            <ol className="flex flex-col" aria-label="Analysis execution steps">
              {trace.steps.map((step, i) => (
                <StepRow key={step.id} step={step} isLast={i === trace.steps.length - 1} />
              ))}
            </ol>
          </div>
        )}
      </div>
    </CornerFrame>
  );
}
