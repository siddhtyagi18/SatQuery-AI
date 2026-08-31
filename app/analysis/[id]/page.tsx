// app/analysis/[id]/page.tsx
// Analysis Results page — second highest priority.
// Two-column layout on desktop: Visual evidence + synthesis on the left, sticky execution trace on right rail.
'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { api } from '@/lib/api';
import type { AnalysisResult } from '@/lib/types/analysis';
import { ModeBadge } from '@/components/ui/ModeBadge';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { ErrorState } from '@/components/ui/ErrorState';
import { AnalysisResultSkeleton } from '@/components/ui/LoadingSkeletonPanel';
import { AnalysisSummary } from '@/components/AnalysisSummary';
import { ConfidenceCard } from '@/components/ConfidenceCard';
import { AgentExecutionTrace } from '@/components/AgentExecutionTrace';
import { SatelliteViewer } from '@/components/SatelliteViewer';
import { GroundingOverlay } from '@/components/GroundingOverlay';
import { BeforeAfterViewer } from '@/components/BeforeAfterViewer';
import { ChangeMapViewer } from '@/components/ChangeMapViewer';
import { OpticalSarViewer } from '@/components/OpticalSarViewer';
import { ChangeStatsPanel } from '@/components/ChangeStatsPanel';
import { Download, RotateCcw, ArrowLeft, Cpu, Clock } from 'lucide-react';
import { toast } from 'sonner';

const MODE_DOMAIN: Record<AnalysisResult['mode'], 'cyan' | 'magenta' | 'amber'> = {
  single_image: 'cyan',
  bi_temporal: 'magenta',
  optical_sar: 'amber',
};

export default function AnalysisResultPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const id = params?.id as string;

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let isMounted = true;

    api.getAnalysis(id)
      .then((data) => {
        if (isMounted) {
          setResult(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Analysis not found');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [id]);

  const handleDownloadReport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_${result.id}_report.json`;
    a.click();
    toast.success('Downloaded complete analysis telemetry report (JSON)');
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto flex flex-col gap-12">
        <AnalysisResultSkeleton />
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-3xl mx-auto py-12">
        <ErrorState
          title="Analysis Not Found"
          reason={error ?? 'The requested analysis record does not exist.'}
          onRetry={() => router.push('/analysis/new')}
        />
      </div>
    );
  }

  if (result.status === 'failed') {
    return (
      <div className="max-w-4xl mx-auto flex flex-col gap-12 py-6">
        <button
          onClick={() => router.push('/analysis/history')}
          className="flex items-center gap-1.5 text-xs font-mono text-[var(--text-muted)] hover:text-[var(--text-primary)]"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to History
        </button>

        <ErrorState
          title="Mission Pipeline Execution Terminated"
          reason={result.errorReason ?? 'Unknown telemetry failure'}
          onRetry={() => router.push('/analysis/new')}
        />

        <div className="flex flex-col gap-3">
          <span className="hud-label">Failure Trace Diagnosis</span>
          <AgentExecutionTrace trace={result.executionTrace} defaultExpanded={true} />
        </div>
      </div>
    );
  }

  const headerDomain = MODE_DOMAIN[result.mode];

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-12 pb-16 animate-fade-in-up">
      {/* Top Header Card — active analysis surface → bracket framed */}
      <CornerFrame
        label="MISSION ANALYSIS TELEMETRY"
        domain={headerDomain}
        bracketSize={14}
        intensity="normal"
      >
        <div className="panel p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <button
                onClick={() => router.push('/analysis/history')}
                className="p-1.5 rounded hover:bg-[var(--bg-panel-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors border border-[var(--border-hairline)]"
                title="Back to History"
              >
                <ArrowLeft className="w-4 h-4" />
              </button>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold text-[var(--accent-signal)]">
                  {result.id.toUpperCase()}
                </span>
                <ModeBadge mode={result.mode} />
                <StatusBadge status={result.status} />
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => router.push('/analysis/new')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Re-run Query
              </button>

              <button
                onClick={handleDownloadReport}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono bg-[var(--accent-signal)]/15 border border-[var(--accent-signal)]/40 text-[var(--accent-signal)] hover:bg-[var(--accent-signal)]/25 transition-colors font-medium"
              >
                <Download className="w-3.5 h-3.5" />
                Export Report
              </button>
            </div>
          </div>

          {/* User Query Banner — Level-2 elevated surface within the framed header */}
          <div
            className="p-3.5 rounded flex flex-col gap-1"
            style={{
              background: 'var(--surface-2)',
              border: '1px solid var(--border-hairline)',
            }}
          >
            <span className="hud-label">Evaluated User Query</span>
            <p className="text-sm font-medium text-[var(--text-primary)] font-heading">
              &ldquo;{result.query}&rdquo;
            </p>
          </div>
        </div>
      </CornerFrame>

      {/* Main Two-Column Layout — 48px gap between columns */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 items-start">
        {/* Left Column (2 spans): Synthesis + Visual Evidence — gap-12 between major blocks */}
        <div className="lg:col-span-2 flex flex-col gap-12">
          {/* Executive Summary */}
          <AnalysisSummary
            answerText={result.answerText}
            detectedTasks={result.detectedTasks}
            createdAt={result.createdAt}
          />

          {/* Mode-Specific Visual Viewers */}
          {result.mode === 'single_image' && (
            <div className="flex flex-col gap-2">
              <span className="hud-label">Visual Sensor Evidence & Grounding Annotations</span>
              <SatelliteViewer
                imageUrl={result.images[0]?.previewUrl ?? '/demo/optical_sample.jpg'}
                crs={result.images[0]?.metadata.crs}
                resolution={`${result.images[0]?.metadata.gsdMeters ?? 5.8}m`}
                title="Single Scene Spatial Grounding"
              >
                {result.boundingBoxes && result.boundingBoxes.length > 0 && (
                  <GroundingOverlay boxes={result.boundingBoxes} />
                )}
              </SatelliteViewer>
            </div>
          )}

          {result.mode === 'bi_temporal' && (
            <div className="flex flex-col gap-12">
              {/* Slider & Dual View */}
              <div className="flex flex-col gap-2">
                <span className="hud-label">Bi-Temporal Visual Baseline Swipe</span>
                <BeforeAfterViewer
                  beforeUrl={result.images[0]?.previewUrl ?? '/demo/optical_before.jpg'}
                  afterUrl={result.images[1]?.previewUrl ?? '/demo/optical_after.jpg'}
                  beforeDate={result.images[0]?.metadata.acquisitionDate ?? 'T1 (Jan 2022)'}
                  afterDate={result.images[1]?.metadata.acquisitionDate ?? 'T2 (Jan 2024)'}
                />
              </div>

              {/* Change Detection Heatmap */}
              <div className="flex flex-col gap-2">
                <span className="hud-label">Classified Change Matrix</span>
                <ChangeMapViewer
                  baseImageUrl={result.images[1]?.previewUrl ?? '/demo/optical_after.jpg'}
                  changeMaskUrl={result.changeMap?.overlayUrl ?? '/demo/change_mask.png'}
                  legend={result.changeMap?.legend}
                />
              </div>

              {/* Real Change Statistics Panel */}
              <ChangeStatsPanel trace={result.executionTrace} />
            </div>
          )}

          {result.mode === 'optical_sar' && (
            <div className="flex flex-col gap-2">
              <span className="hud-label">Multimodal Cross-Sensor Fusion</span>
              <OpticalSarViewer
                opticalUrl={result.images[0]?.previewUrl ?? '/demo/optical_sample.jpg'}
                sarUrl={result.images[1]?.previewUrl ?? '/demo/sar_sample.jpg'}
              />
            </div>
          )}
        </div>

        {/* Right Rail (1 span): Confidence + Sticky Execution Trace & Tool Invocations */}
        <div className="flex flex-col gap-12 lg:sticky lg:top-20">
          {/* Confidence Score Card */}
          <ConfidenceCard
            score={result.confidence}
            detectedTasks={result.detectedTasks}
          />

          {/* Specialist Models Invoked Panel — active agent surface → cyan bracket frame */}
          {result.toolInvocations && result.toolInvocations.length > 0 && (
            <CornerFrame label="INVOKED SPECIALIST AGENTS" domain="cyan">
              <div className="panel p-4 flex flex-col gap-3">
                {result.toolInvocations.map((tool) => (
                  <div
                    key={tool.toolId}
                    className="p-3 rounded flex flex-col gap-2"
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border-hairline)',
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Cpu className="w-3.5 h-3.5 text-[var(--accent-signal)]" />
                        <span className="text-xs font-semibold font-heading text-[var(--text-primary)]">
                          {tool.toolName}
                        </span>
                      </div>
                      <span className="text-[0.6rem] font-mono text-[var(--text-faint)]">
                        v{tool.version}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[0.65rem] font-mono text-[var(--text-muted)]">
                      <span>Task: <strong className="text-[var(--text-primary)]">{tool.taskType}</strong></span>
                      {tool.processingTimeMs != null && (
                        <span className="flex items-center gap-1 text-[var(--accent-success)]">
                          <Clock className="w-2.5 h-2.5" />
                          {(tool.processingTimeMs / 1000).toFixed(2)}s
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CornerFrame>
          )}

          {/* Full Agent Execution Trace — centerpiece panel already internally framed + elevated */}
          <AgentExecutionTrace trace={result.executionTrace} defaultExpanded={true} />
        </div>
      </div>
    </div>
  );
}
