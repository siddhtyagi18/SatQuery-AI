// components/ChangeStatsPanel.tsx
// Compact "Change Statistics" panel for bi-temporal analysis results.
// Reads scalar change stats from executionTrace.steps (step-6 meta) —
// injected by the backend when the real change detection service runs.
// Displays execution provenance: model_checkpoint vs cpu_classical.
// Renders nothing when stats are not available (graceful absent for mock/older results).

import type { ExecutionTrace } from '@/lib/types/analysis';

interface ChangeStatsPanelProps {
  trace: ExecutionTrace;
}

// Severity → design-system colour token
const SEVERITY_STYLES: Record<string, { label: string; color: string; bg: string }> = {
  low:      { label: 'LOW',      color: 'var(--accent-success)',  bg: 'rgba(61,220,132,0.10)'  },
  moderate: { label: 'MODERATE', color: 'var(--accent-warning)',  bg: 'rgba(255,176,32,0.10)'  },
  high:     { label: 'HIGH',     color: 'var(--accent-danger)',   bg: 'rgba(255,92,92,0.10)'   },
};

// Execution mode badge styles
const MODE_BADGE: Record<string, { label: string; color: string; bg: string; border: string }> = {
  model_checkpoint: {
    label: 'SIAMESE U-NET MODEL',
    color: '#a78bfa',
    bg:    'rgba(167,139,250,0.12)',
    border:'rgba(167,139,250,0.35)',
  },
  cpu_classical: {
    label: 'CPU CLASSICAL BASELINE',
    color: 'var(--accent-signal)',
    bg:    'rgba(56,189,248,0.10)',
    border:'rgba(56,189,248,0.30)',
  },
};

export function ChangeStatsPanel({ trace }: ChangeStatsPanelProps) {
  // Extract step-6 meta from the execution trace
  const step6 = trace.steps.find((s) => s.id === 'step-6');
  const meta = step6?.meta;

  // Only render if the real change detection stats are present
  if (
    !meta ||
    typeof meta.changed_pixel_pct !== 'number' ||
    typeof meta.unchanged_pixel_pct !== 'number'
  ) {
    return null;
  }

  const changedPct   = meta.changed_pixel_pct as number;
  const unchangedPct = meta.unchanged_pixel_pct as number;
  const changedCount = meta.changed_pixel_count as number | undefined;
  const totalCount   = meta.total_pixel_count as number | undefined;
  const severity     = typeof meta.severity === 'string' ? meta.severity : null;
  const imageSize    = typeof meta.image_size_str === 'string' ? meta.image_size_str : null;
  const thresholdRaw = typeof meta.threshold_raw_255 === 'number' ? meta.threshold_raw_255 : null;

  // Execution provenance
  const execMode    = typeof meta.execution_mode === 'string' ? meta.execution_mode : null;
  const ckptPath    = typeof meta.checkpoint_path === 'string' ? meta.checkpoint_path : null;

  const severityStyle = severity ? (SEVERITY_STYLES[severity] ?? null) : null;
  const modeBadge     = execMode ? (MODE_BADGE[execMode] ?? null) : null;

  const fmtPct = (v: number) => `${v.toFixed(1)}%`;
  const fmtNum = (v: number) => v.toLocaleString();

  // Build the algorithm description based on what actually ran
  let algorithmFootnote: string;
  if (execMode === 'model_checkpoint') {
    algorithmFootnote = 'Inference: Siamese U-Net trained on LEVIR-CD (256×256 overlapping tile sliding, averaged probabilities)';
    if (ckptPath) algorithmFootnote += ` · checkpoint: ${ckptPath.split(/[\\/]/).pop()}`;
  } else if (execMode === 'cpu_classical') {
    algorithmFootnote = 'Algorithm: grayscale absolute pixel difference';
    if (thresholdRaw != null) algorithmFootnote += ` (threshold ${thresholdRaw}/255)`;
    algorithmFootnote += ' · No trained model used · Severity labels are heuristic only';
  } else {
    // Older result without execution_mode in meta
    algorithmFootnote = 'Algorithm: grayscale absolute difference';
    if (thresholdRaw != null) algorithmFootnote += ` (threshold ${thresholdRaw}/255)`;
    if (imageSize) algorithmFootnote += ` · Image: ${imageSize} px`;
  }

  if (imageSize && execMode !== 'cpu_classical') {
    algorithmFootnote += ` · Image: ${imageSize} px`;
  }

  return (
    <div
      className="panel p-4 flex flex-col gap-3"
      style={{ border: '1px solid rgba(192,132,252,0.22)' }}
      aria-label="Change Statistics"
    >
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="hud-label">Change Statistics</span>
        <div className="flex items-center gap-2">
          {/* Execution mode badge */}
          {modeBadge && (
            <span
              className="text-[0.6rem] font-mono font-bold px-2 py-0.5 rounded"
              style={{
                color: modeBadge.color,
                background: modeBadge.bg,
                border: `1px solid ${modeBadge.border}`,
              }}
              title={
                execMode === 'model_checkpoint'
                  ? `Results produced by trained SiameseUNet model checkpoint${ckptPath ? ': ' + ckptPath : ''}`
                  : 'Results produced by CPU classical pixel-difference algorithm (no trained model)'
              }
            >
              {modeBadge.label}
            </span>
          )}
          {/* Severity badge */}
          {severityStyle && (
            <span
              className="text-[0.6rem] font-mono font-bold px-2 py-0.5 rounded"
              style={{ color: severityStyle.color, background: severityStyle.bg }}
            >
              SEVERITY: {severityStyle.label}
            </span>
          )}
        </div>
      </div>

      {/* Main bar — visual proportion */}
      <div
        className="w-full rounded overflow-hidden"
        style={{ height: 6, background: 'var(--surface-2)' }}
        aria-label={`${fmtPct(changedPct)} changed`}
      >
        <div
          className="h-full rounded"
          style={{
            width: `${Math.min(changedPct, 100)}%`,
            background: 'var(--accent-danger)',
            transition: 'width 0.4s ease',
          }}
        />
      </div>

      {/* Stat rows */}
      <div className="grid grid-cols-2 gap-2">
        <StatRow
          label="Changed Area"
          value={fmtPct(changedPct)}
          accent="var(--accent-danger)"
        />
        <StatRow
          label="Unchanged Area"
          value={fmtPct(unchangedPct)}
          accent="var(--accent-success)"
        />
        {changedCount != null && (
          <StatRow
            label="Changed Pixels"
            value={fmtNum(changedCount)}
            accent="var(--text-muted)"
          />
        )}
        {totalCount != null && (
          <StatRow
            label="Total Pixels"
            value={fmtNum(totalCount)}
            accent="var(--text-muted)"
          />
        )}
      </div>

      {/* Footer footnote */}
      <p
        className="text-[0.6rem] font-mono leading-relaxed"
        style={{ color: 'var(--text-faint)' }}
      >
        {algorithmFootnote}
      </p>
    </div>
  );
}

// ---- sub-component ----

function StatRow({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div
      className="flex flex-col gap-0.5 p-2 rounded"
      style={{ background: 'var(--surface-2)', border: '1px solid var(--border-hairline)' }}
    >
      <span
        className="text-[0.58rem] font-mono uppercase tracking-wider"
        style={{ color: 'var(--text-faint)' }}
      >
        {label}
      </span>
      <span
        className="text-sm font-bold font-mono"
        style={{ color: accent }}
      >
        {value}
      </span>
    </div>
  );
}
