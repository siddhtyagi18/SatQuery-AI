// app/analysis/[id]/page.tsx
// Analysis Results page — second highest priority.
// Two-column layout on desktop: Visual evidence + synthesis on the left, sticky execution trace on right rail.
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { api } from '@/lib/api';
import type { AnalysisResult, SpatialAction } from '@/lib/types/analysis';
import { ModeBadge } from '@/components/ui/ModeBadge';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { ErrorState } from '@/components/ui/ErrorState';
import { AnalysisResultSkeleton } from '@/components/ui/LoadingSkeletonPanel';
import { AnalysisSummary } from '@/components/AnalysisSummary';
import { MultilingualSummaryPanel } from '@/components/MultilingualSummaryPanel';
import { SmartInsightCards } from '@/components/SmartInsightCards';
import { FollowUpPanel } from '@/components/FollowUpPanel';
import { ConfidenceCard } from '@/components/ConfidenceCard';
import { AgentExecutionTrace } from '@/components/AgentExecutionTrace';
import { SatelliteViewer } from '@/components/SatelliteViewer';
import { GroundingOverlay } from '@/components/GroundingOverlay';
import { BeforeAfterViewer } from '@/components/BeforeAfterViewer';
import { ChangeMapViewer } from '@/components/ChangeMapViewer';
import { OpticalSarViewer } from '@/components/OpticalSarViewer';
import { ChangeStatsPanel } from '@/components/ChangeStatsPanel';
import { GeoSpatialChangeAnalytics } from '@/components/GeoSpatialChangeAnalytics';
import { RoiInvestigationPanel } from '@/components/RoiInvestigationPanel';
import { MissionReportCard } from '@/components/MissionReportCard';
import { ROIBounds, ROIAnalysisResponse } from '@/lib/types/analysis';
import { Download, RotateCcw, ArrowLeft, Cpu, Clock, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';
import { supabaseAnalysisService, SUPABASE_PERSISTENCE_ENABLED } from '@/lib/supabase/services';
import { getCurrentUserId } from '@/lib/authService';

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

  // Phase 2: Region of Interest (ROI) State
  const [selectedRoi, setSelectedRoi] = useState<ROIBounds | null>(null);
  const [roiData, setRoiData] = useState<ROIAnalysisResponse | null>(null);
  const [roiLoading, setRoiLoading] = useState(false);

  const handleSelectRoi = async (bounds: ROIBounds) => {
    setSelectedRoi(bounds);
    setRoiLoading(true);
    try {
      const res = await fetch(`/api/analysis/${id}/roi-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          x1: bounds.x1,
          y1: bounds.y1,
          x2: bounds.x2,
          y2: bounds.y2,
          is_normalized: true,
          run_vqa: false,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setRoiData(data);
      }
    } catch (err) {
      console.error('Failed to calculate ROI change analytics:', err);
    } finally {
      setRoiLoading(false);
    }
  };

  const handleClearRoi = () => {
    setSelectedRoi(null);
    setRoiData(null);
  };

  // Spatial highlighting & viewer focus link for contextual follow-up
  const [mapHighlighted, setMapHighlighted] = useState(false);
  const visualViewerRef = useRef<HTMLDivElement>(null);

  const handleSpatialAction = useCallback((action: SpatialAction) => {
    if (action && action.action !== 'none') {
      visualViewerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setMapHighlighted(true);
      if (action.note) {
        toast.info(action.note);
      }
      setTimeout(() => setMapHighlighted(false), 2600);
    }
  }, []);

  // Derived: execution mode of the change detection step (from trace step-6 meta)
  const changeExecMode: string | null = (() => {
    if (!result) return null;
    const step6 = result.executionTrace?.steps?.find((s) => s.id === 'step-6');
    const em = step6?.meta?.execution_mode;
    return typeof em === 'string' ? em : null;
  })();

  // Human-readable algorithm label for ChangeMapViewer toolbar
  const changeAlgorithmLabel = (() => {
    if (changeExecMode === 'model_checkpoint') return 'Siamese U-Net Model (LEVIR-CD trained)';
    if (changeExecMode === 'cpu_classical')    return 'Grayscale Absolute Difference (CPU Classical)';
    return 'Pixel-level Change Mask';
  })();

  useEffect(() => {
    if (!id) return;
    let isMounted = true;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const fetchResult = async () => {
      try {
        const data = await api.getAnalysis(id);
        if (!isMounted) return;
        setResult(data);
        setLoading(false);

        // If analysis is still in progress, continue polling live until terminal state
        if (data.status === 'queued' || data.status === 'processing' || (data.status as string) === 'running') {
          timeoutId = setTimeout(fetchResult, 1000);
        }
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.message || 'Analysis not found');
        setLoading(false);
      }
    };

    fetchResult();

    return () => {
      isMounted = false;
      if (timeoutId) clearTimeout(timeoutId);
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

    if (SUPABASE_PERSISTENCE_ENABLED) {
      (async () => {
        try {
          const uid = await getCurrentUserId();
          if (!uid) return;
          await supabaseAnalysisService.uploadResultBlob({
            userId: uid,
            analysisId: result.id,
            blobName: `SatQuery_${result.id}_report.json`,
            blob,
          });
        } catch (err) {
          console.warn('[analysis-detail] report archive failed (silent):', err);
        }
      })();
    }
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
  const anyReal = result.toolInvocations?.some((t) => t.executionMode === 'real');
  const isMock = result.isMock !== undefined ? result.isMock : !anyReal;

  return (
    <div className="max-w-7xl mx-auto flex flex-col gap-7 pb-16 animate-fade-in-up">
      {/* Top Header Card — active analysis surface → bracket framed */}
      <CornerFrame
        label="MISSION ANALYSIS TELEMETRY"
        domain={headerDomain}
        bracketSize={14}
        intensity="normal"
      >
        <div className="panel p-5 flex flex-col gap-4 transition-all duration-300 hover:border-[var(--cyan)]/35">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <button
                onClick={() => router.push('/analysis/history')}
                className="p-1.5 rounded hover:bg-[var(--bg-panel-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all hover:scale-105 active:scale-95 border border-[var(--border-hairline)]"
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
                {isMock ? (
                  <span
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-bold uppercase tracking-wider bg-amber-500/15 border border-amber-500/40 text-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.15)]"
                    title="Phase 1 Mock Execution: Synthetic placeholder output; no authentic model weights were invoked."
                  >
                    <AlertTriangle className="w-3.5 h-3.5" />
                    MOCK — Phase 1
                  </span>
                ) : (
                  <span
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-bold uppercase tracking-wider bg-emerald-500/15 border border-emerald-500/40 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.15)]"
                    title="Real ML Specialist Execution: Output produced by authentic model inference on imagery."
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    REAL ML SPECIALIST
                  </span>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => router.push('/analysis/new')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)] transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Re-run Query
              </button>

              <button
                onClick={handleDownloadReport}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono bg-[var(--accent-signal)]/15 border border-[var(--accent-signal)]/40 text-[var(--accent-signal)] hover:bg-[var(--accent-signal)]/25 transition-all hover:scale-[1.02] active:scale-[0.98] font-medium"
              >
                <Download className="w-3.5 h-3.5" />
                Export Report
              </button>
            </div>
          </div>

          {/* User Query Banner — Level-2 elevated surface within the framed header */}
          <div
            className="p-3.5 rounded flex flex-col gap-1 transition-colors hover:border-[var(--border-strong)]"
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

      {/* Disaster Assessment Banner (shown only when disaster mode) */}
      {result.analysisMissionMode === 'disaster_assessment' && result.disasterType && (
        <div
          className="p-5 rounded-lg border-2 border-[var(--amber)]/50 animate-fade-in-up"
          style={{
            background: 'color-mix(in srgb, var(--amber) 6%, var(--bg-base))',
            boxShadow: '0 0 20px rgba(251,191,36,0.12)',
          }}
        >
          <div className="flex items-center gap-3 mb-3">
            <span className="font-mono text-[0.65rem] font-bold tracking-widest uppercase px-2.5 py-1 rounded bg-[var(--amber)]/15 border border-[var(--amber)]/40 text-[var(--amber)]">
              DISASTER ASSESSMENT
            </span>
          </div>
          <div className="flex flex-col gap-1.5 mb-4">
            <div className="flex items-center gap-2">
              <span className="text-[0.7rem] font-mono text-[var(--text-muted)] uppercase">Selected Mission:</span>
              <span className="text-base font-bold text-[var(--amber)] font-heading">
                {result.disasterType === 'flood' && '🌊 '}
                {result.disasterType === 'earthquake' && '🏚️ '}
                {result.disasterType === 'wildfire' && '🔥 '}
                {result.disasterType === 'cyclone' && '🌀 '}
                {result.disasterType === 'landslide' && '⛰️ '}
                {result.disasterType.charAt(0).toUpperCase() + result.disasterType.slice(1)}
              </span>
            </div>
          </div>

          {/* Disaster Assessment Context */}
          <div className="p-3.5 rounded bg-[var(--surface-2)] border border-[var(--border-hairline)]">
            <span className="hud-label text-[var(--amber)] mb-2 block">DISASTER ASSESSMENT CONTEXT</span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
              <div>
                <span className="text-[var(--text-muted)]">Selected disaster: </span>
                <span className="text-[var(--text-primary)] font-semibold">
                  {result.disasterType.charAt(0).toUpperCase() + result.disasterType.slice(1)}
                </span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Analysis basis: </span>
                <span className="text-[var(--text-primary)] font-semibold">Bi-temporal change detection</span>
              </div>
            </div>
            <p className="text-xs text-[var(--text-muted)] mt-2.5 leading-relaxed">
              {result.disasterType === 'flood' &&
                'Flood assessment context selected. Detected visual change regions are highlighted for operator investigation.'}
              {result.disasterType === 'earthquake' &&
                'Earthquake assessment context selected. Detected structural/visual change regions are highlighted for operator investigation.'}
              {result.disasterType === 'wildfire' &&
                'Wildfire assessment context selected. Detected change regions are highlighted for investigation.'}
              {result.disasterType === 'cyclone' &&
                'Cyclone assessment context selected. Detected change regions are highlighted for investigation.'}
              {result.disasterType === 'landslide' &&
                'Landslide assessment context selected. Detected change regions are highlighted for investigation.'}
            </p>
          </div>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-7 items-start">
        {/* Left Column (2 spans): Synthesis + Visual Evidence */}
        <div className="lg:col-span-2 flex flex-col gap-7">
          {/* Smart Insights & Telemetry Cards */}
          <SmartInsightCards result={result} />

          {/* Bilingual Summary + TTS Playback */}
          <MultilingualSummaryPanel
            summaries={result.multilingualSummaries}
            fallbackAnswer={result.answerText}
          />

          {/* Executive Summary */}
          <AnalysisSummary
            answerText={result.answerText}
            detectedTasks={result.detectedTasks}
            createdAt={result.createdAt}
            isMock={isMock}
          />

          {/* Mode-Specific Visual Viewers */}
          <div
            ref={visualViewerRef}
            className={cn(
              'flex flex-col gap-7 transition-all duration-500 rounded-lg',
              mapHighlighted && 'ring-2 ring-[var(--cyan)] shadow-[0_0_24px_rgba(34,211,238,0.35)]'
            )}
          >
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
              <div className="flex flex-col gap-7">
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
                  <span className="hud-label">Change Detection Output</span>
                  <ChangeMapViewer
                    baseImageUrl={result.images[1]?.previewUrl ?? '/demo/optical_after.jpg'}
                    changeMaskUrl={result.changeMap?.overlayUrl ?? '/demo/change_mask.png'}
                    legend={result.changeMap?.legend}
                    algorithmLabel={changeAlgorithmLabel}
                    roi={selectedRoi}
                    onSelectRoi={handleSelectRoi}
                    onClearRoi={handleClearRoi}
                  />
                </div>

                {/* Real Change Statistics Panel */}
                <ChangeStatsPanel trace={result.executionTrace} />

                {/* Geo-Spatial Change Analytics */}
                <GeoSpatialChangeAnalytics
                  analytics={result.changeMap?.analytics}
                  trace={result.executionTrace}
                />

                {/* Phase 2: Interactive Region of Interest (ROI) Investigation */}
                <RoiInvestigationPanel
                  analysisId={result.id}
                  roi={selectedRoi}
                  roiData={roiData}
                  loading={roiLoading}
                  onClearRoi={handleClearRoi}
                />

                {/* Phase 3: Automated AI Mission Report / Evidence Report */}
                <MissionReportCard
                  analysisId={result.id}
                  roiData={roiData}
                />
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

          {/* Contextual Follow-up Questions Console */}
          <FollowUpPanel
            result={result}
            onSpatialAction={handleSpatialAction}
          />
        </div>

        {/* Right Rail (1 span): Confidence + Sticky Execution Trace & Tool Invocations */}
        <div className="flex flex-col gap-7 lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto pr-1">
          {/* Confidence Score Card */}
          <ConfidenceCard
            score={result.confidence}
            detectedTasks={result.detectedTasks}
            isMock={isMock}
          />

          {/* Specialist Models Invoked Panel — active agent surface → cyan bracket frame */}
          {result.toolInvocations && result.toolInvocations.length > 0 && (
            <CornerFrame label="INVOKED SPECIALIST AGENTS" domain="cyan">
              <div className="panel p-4 flex flex-col gap-3">
                {result.toolInvocations.map((tool) => (
                  <div
                    key={tool.toolId}
                    className="p-3 rounded flex flex-col gap-2 transition-all duration-200 hover:-translate-y-0.5 hover:border-[var(--cyan)]/35 hover:shadow-sm"
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
