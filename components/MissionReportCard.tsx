// components/MissionReportCard.tsx
// Evidence-backed AI Mission Report generator and PDF viewer.
// Assembles existing analysis results without rerunning ML models.
'use client';

import { useState } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import {
  FileText,
  Download,
  Eye,
  CheckCircle2,
  AlertCircle,
  RotateCcw,
  X,
  ExternalLink,
  Shield,
  Layers,
  Crosshair,
  Sparkles,
  Info,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { MissionReportResponse, ROIAnalysisResponse } from '@/lib/types/analysis';

interface MissionReportCardProps {
  analysisId: string;
  roiData?: ROIAnalysisResponse | null;
  className?: string;
}

export function MissionReportCard({
  analysisId,
  roiData,
  className,
}: MissionReportCardProps) {
  const [generating, setGenerating] = useState(false);
  const [reportResult, setReportResult] = useState<MissionReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isViewing, setIsViewing] = useState(false);

  const handleGenerateReport = async () => {
    if (!analysisId) return;
    setGenerating(true);
    setError(null);

    try {
      const res = await fetch(`/api/analysis/${analysisId}/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          roi: roiData || null,
          generate_pdf: true,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Report compilation failed (${res.status})`);
      }

      const data: MissionReportResponse = await res.json();
      setReportResult(data);
    } catch (err: any) {
      setError(err.message || 'Mission report compilation failed.');
    } finally {
      setGenerating(false);
    }
  };

  const report = reportResult?.report;

  return (
    <>
      <CornerFrame
        label="MISSION EVIDENCE REPORT"
        domain="cyan"
        className={cn('w-full', className)}
      >
        <div className="panel p-5 flex flex-col gap-4 bg-[var(--bg-base)] transition-all duration-300 hover:border-[var(--cyan)]/35 hover:shadow-lg">
          {/* Header Strip */}
          <div className="flex items-center justify-between flex-wrap gap-4 border-b border-[var(--border-hairline)] pb-3">
            <div className="flex items-center gap-2.5">
              <FileText className="w-4 h-4 text-[var(--cyan)]" />
              <div>
                <h3 className="font-mono text-sm font-bold text-[var(--text-primary)]">
                  Automated Evidence-Backed Mission Report
                </h3>
                <p className="font-mono text-[0.68rem] text-[var(--text-muted)]">
                  Consolidate telemetry, spatial analytics, ROI results, and model provenance into a verified PDF
                </p>
              </div>
            </div>

            {reportResult && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[0.65rem] font-mono bg-emerald-950/40 border border-emerald-500/40 text-emerald-400">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                <span>Report Ready</span>
              </div>
            )}
          </div>

          {/* Action & Status Row */}
          {!reportResult && !generating && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-hairline)]">
              <div className="flex items-start gap-2.5 max-w-xl">
                <Shield className="w-4 h-4 text-[var(--cyan)] mt-0.5 flex-shrink-0" />
                <p className="font-mono text-xs text-[var(--text-muted)] leading-relaxed">
                  Generate an immutable mission artifact that documents data context, Siamese U-Net metrics (threshold 0.70), spatial density, local hotspots, ROI metrics, and uncalibrated confidence safeguards.
                </p>
              </div>

              <button
                type="button"
                onClick={handleGenerateReport}
                className="flex items-center gap-2 px-4 py-2 rounded text-xs font-mono font-bold bg-[var(--cyan)] text-black hover:bg-[var(--cyan)]/90 shadow-[0_0_12px_rgba(62,208,255,0.3)] transition-all whitespace-nowrap"
              >
                <FileText className="w-3.5 h-3.5" />
                Generate Mission Report
              </button>
            </div>
          )}

          {generating && (
            <div className="flex items-center gap-3 p-4 rounded-lg bg-[var(--bg-panel)] border border-[var(--cyan)]/30 text-xs font-mono text-[var(--cyan)]">
              <div className="w-4 h-4 rounded-full border-2 border-[var(--cyan)] border-t-transparent animate-spin" />
              <span>Generating report...</span>
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-950/30 border border-red-500/40 rounded flex items-center justify-between gap-2 text-xs font-mono text-red-300">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                <span>{error}</span>
              </div>
              <button
                type="button"
                onClick={handleGenerateReport}
                className="text-[0.65rem] text-[var(--text-muted)] hover:text-white underline"
              >
                Retry
              </button>
            </div>
          )}

          {reportResult && report && (
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-lg bg-[var(--bg-panel-elevated)] border border-[var(--cyan)]/40 animate-fade-in-up">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-bold text-[var(--text-primary)]">
                    Mission Report #{report.analysis_id.slice(0, 8)}
                  </span>
                  <span className="text-[0.65rem] font-mono text-[var(--text-muted)]">
                    ({report.created_at.slice(0, 10)})
                  </span>
                </div>
                <span className="font-mono text-[0.7rem] text-[var(--text-muted)]">
                  Identified {report.change_detection.changed_pixel_pct?.toFixed(2) ?? 'N/A'}% changed pixels ({report.change_detection.changed_pixel_count?.toLocaleString()} px)
                </span>
              </div>

              <div className="flex items-center gap-2.5 flex-wrap">
                <button
                  type="button"
                  onClick={() => setIsViewing(true)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono font-medium bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[var(--text-primary)] hover:border-[var(--cyan)] transition-all"
                >
                  <Eye className="w-3.5 h-3.5 text-[var(--cyan)]" />
                  View Report
                </button>

                {reportResult.pdf_url && (
                  <a
                    href={reportResult.pdf_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded text-xs font-mono font-bold bg-[var(--cyan)] text-black hover:bg-[var(--cyan)]/90 shadow-[0_0_12px_rgba(62,208,255,0.3)] transition-all"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download PDF
                  </a>
                )}

                <button
                  type="button"
                  onClick={handleGenerateReport}
                  className="p-1.5 rounded text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel)] transition-all"
                  title="Re-generate report with latest ROI selection"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      </CornerFrame>

      {/* Interactive Report View Modal */}
      {isViewing && report && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
          <div className="bg-[var(--bg-base)] border border-[var(--cyan)]/50 rounded-lg max-w-4xl w-full max-h-[90vh] flex flex-col shadow-2xl animate-fade-in-up">
            {/* Modal Header */}
            <div className="p-4 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-[var(--cyan)]" />
                <span className="font-mono text-sm font-bold text-[var(--text-primary)]">
                  SATQUERY-AI MISSION REPORT • {report.analysis_id}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {reportResult?.pdf_url && (
                  <a
                    href={reportResult.pdf_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-[var(--cyan)] text-black font-bold hover:bg-[var(--cyan)]/90 transition-all"
                  >
                    <Download className="w-3 h-3" />
                    PDF
                  </a>
                )}
                <button
                  type="button"
                  onClick={() => setIsViewing(false)}
                  className="p-1 rounded text-[var(--text-muted)] hover:text-white transition-all"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Modal Scrollable Content */}
            <div className="p-6 overflow-y-auto flex flex-col gap-6 text-xs font-mono">
              {/* 1. Executive Summary */}
              <div className="p-4 rounded bg-[var(--bg-panel)] border border-[var(--cyan)]/30 flex flex-col gap-1.5">
                <span className="hud-label text-[var(--cyan)]">1. Executive Summary</span>
                <p className="text-sm text-[var(--text-primary)] leading-relaxed">
                  {report.executive_summary}
                </p>
              </div>

              {/* 2. Quantitative Change Detection */}
              <div className="flex flex-col gap-2">
                <span className="hud-label">2. Quantitative Change Detection</span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)]">
                    <span className="text-[0.62rem] text-[var(--text-muted)] uppercase block">Total Pixels</span>
                    <span className="text-sm font-bold text-[var(--text-primary)]">
                      {report.change_detection.total_pixel_count?.toLocaleString()}
                    </span>
                  </div>
                  <div className="p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)]">
                    <span className="text-[0.62rem] text-[var(--text-muted)] uppercase block">Changed Pixels</span>
                    <span className="text-sm font-bold text-[var(--accent-change)]">
                      {report.change_detection.changed_pixel_count?.toLocaleString()} ({report.change_detection.changed_pixel_pct?.toFixed(2)}%)
                    </span>
                  </div>
                  <div className="p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)]">
                    <span className="text-[0.62rem] text-[var(--text-muted)] uppercase block">Detector Threshold</span>
                    <span className="text-sm font-bold text-[var(--text-primary)]">
                      {report.change_detection.threshold_used}
                    </span>
                  </div>
                  <div className="p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)]">
                    <span className="text-[0.62rem] text-[var(--text-muted)] uppercase block">Confidence</span>
                    <span className="text-sm font-bold text-[var(--text-faint)]">
                      {report.change_detection.confidence_label}
                    </span>
                  </div>
                </div>
              </div>

              {/* 3. Geo-Spatial Analytics */}
              <div className="flex flex-col gap-2">
                <span className="hud-label">3. Geo-Spatial Analytics</span>
                <div className="p-3.5 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)] grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <span className="text-[0.62rem] text-[var(--text-muted)] block">Spatial Resolution (GSD)</span>
                    <span className="text-xs font-bold text-[var(--text-primary)]">
                      {report.geospatial_analytics.resolution || 'Not available'}
                    </span>
                  </div>
                  <div>
                    <span className="text-[0.62rem] text-[var(--text-muted)] block">Coordinate System (CRS)</span>
                    <span className="text-xs font-bold text-[var(--text-primary)]">
                      {report.geospatial_analytics.crs || 'Not available'}
                    </span>
                  </div>
                  <div>
                    <span className="text-[0.62rem] text-[var(--text-muted)] block">Physical Changed Area</span>
                    <span className="text-xs font-bold text-[var(--text-primary)]">
                      {report.geospatial_analytics.physical_area?.available
                        ? `${report.geospatial_analytics.physical_area.changed_area_hectares} ha`
                        : 'Unavailable (No GSD)'}
                    </span>
                  </div>
                </div>
              </div>

              {/* 4. ROI Investigation */}
              <div className="flex flex-col gap-2">
                <span className="hud-label">4. Region of Interest (ROI) Investigation</span>
                {report.roi_investigation.performed && report.roi_investigation.details ? (
                  <div className="p-3.5 rounded bg-[var(--bg-panel-elevated)] border border-emerald-500/40 flex flex-col gap-2">
                    <span className="font-bold text-emerald-400">ROI Inspected:</span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      <div>
                        <span className="text-[0.6rem] text-[var(--text-muted)] block">Dimensions</span>
                        <span>{report.roi_investigation.details.roi?.pixel_coordinates?.width} × {report.roi_investigation.details.roi?.pixel_coordinates?.height} px</span>
                      </div>
                      <div>
                        <span className="text-[0.6rem] text-[var(--text-muted)] block">ROI Change</span>
                        <span className="text-emerald-400 font-bold">{report.roi_investigation.details.statistics?.changed_percentage?.toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-[0.6rem] text-[var(--text-muted)] block">Scene Baseline</span>
                        <span>{report.roi_investigation.details.global_comparison?.global_changed_percentage?.toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-[0.6rem] text-[var(--text-muted)] block">Relative Delta</span>
                        <span>{report.roi_investigation.details.global_comparison?.difference_percentage > 0 ? '+' : ''}{report.roi_investigation.details.global_comparison?.difference_percentage?.toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[var(--text-muted)]">
                    ROI investigation was not performed.
                  </div>
                )}
              </div>

              {/* 5. AI Interpretation (Change VQA) */}
              <div className="flex flex-col gap-2">
                <span className="hud-label">5. Vision-Language AI Interpretation</span>
                {report.ai_interpretation.executed && report.ai_interpretation.answer ? (
                  <div className="p-3.5 rounded bg-[var(--bg-panel)] border border-[var(--cyan)]/40 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-[var(--cyan)]">VLM Explanation</span>
                      <span className="text-[0.65rem] text-[var(--text-faint)]">Confidence: N/A — Uncalibrated</span>
                    </div>
                    <p className="text-[var(--text-primary)] leading-relaxed whitespace-pre-line">
                      {report.ai_interpretation.answer}
                    </p>
                  </div>
                ) : (
                  <div className="p-3 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[var(--text-muted)]">
                    AI interpretation was not requested for this analysis.
                  </div>
                )}
              </div>

              {/* 6. Truthful Limitations */}
              <div className="p-3.5 rounded bg-amber-950/20 border border-amber-500/30 flex flex-col gap-2">
                <span className="hud-label text-amber-400 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5" />
                  6. Truthful Disclaimers & Limitations
                </span>
                <ul className="list-disc list-inside space-y-1 text-[var(--text-muted)]">
                  {report.limitations.map((lim: string, idx: number) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
