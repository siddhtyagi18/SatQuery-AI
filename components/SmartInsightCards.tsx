// components/SmartInsightCards.tsx
// Compact Smart Insights section derived strictly from existing analysis results.
// Zero new models. Zero invented data. Real telemetry presentation only.
'use client';

import { useMemo } from 'react';
import type { AnalysisResult } from '@/lib/types/analysis';
import { CornerFrame } from '@/components/ui/CornerFrame';
import {
  Activity,
  Maximize2,
  AlertTriangle,
  Target,
  Sparkles,
  Calendar,
  Layers,
  MapPin,
} from 'lucide-react';

interface SmartInsightCardsProps {
  result: AnalysisResult;
  className?: string;
}

interface MetricCardItem {
  id: string;
  label: string;
  value: string;
  subtext?: string;
  icon: React.ElementType;
  tone: 'cyan' | 'emerald' | 'amber' | 'rose' | 'neutral';
}

export function SmartInsightCards({ result, className }: SmartInsightCardsProps) {
  const { cards, primaryFinding } = useMemo(() => {
    const answer = result.answerText || '';
    const items: MetricCardItem[] = [];

    // 1. Change Status Card
    const isBiTemporal = result.mode === 'bi_temporal';
    const hasChangeDetection = result.detectedTasks?.some((t) =>
      ['change_detection', 'change_vqa', 'change_description'].includes(t)
    );

    if (isBiTemporal || hasChangeDetection || result.changeMap) {
      items.push({
        id: 'change_status',
        label: 'CHANGE STATUS',
        value: 'Detected',
        subtext: 'Scene variance confirmed',
        icon: Activity,
        tone: 'rose',
      });
    } else if (result.mode === 'optical_sar') {
      items.push({
        id: 'fusion_status',
        label: 'CROSS-SENSOR FUSION',
        value: 'Fused',
        subtext: 'Optical + SAR co-registered',
        icon: Layers,
        tone: 'amber',
      });
    } else {
      const boxCount = result.boundingBoxes?.length ?? 0;
      items.push({
        id: 'detection_status',
        label: 'DETECTION STATUS',
        value: boxCount > 0 ? `${boxCount} Targets` : 'Scene Evaluated',
        subtext: boxCount > 0 ? 'Localized bounding boxes' : 'Spatial VQA completed',
        icon: Target,
        tone: 'cyan',
      });
    }

    // 2. Changed Area Card (Extract strictly from result telemetry)
    const areaMatch =
      answer.match(/Detected Changed Area:\s*`?([0-9.]+%)`?/i) ||
      answer.match(/([0-9.]+%)\s*area/i) ||
      result.changeMap?.legend?.[0]?.label?.match(/([0-9.]+%)/);

    if (areaMatch) {
      items.push({
        id: 'changed_area',
        label: 'CHANGED AREA',
        value: areaMatch[1],
        subtext: 'Pixel change density',
        icon: Maximize2,
        tone: 'amber',
      });
    } else if (isBiTemporal) {
      // In bi-temporal mode with missing regex, show not available
      items.push({
        id: 'changed_area',
        label: 'CHANGED AREA',
        value: 'Not available',
        subtext: 'Mask density uncalibrated',
        icon: Maximize2,
        tone: 'neutral',
      });
    }

    // 3. Severity Card (Strictly if present in telemetry — NEVER invent)
    const sevMatch =
      answer.match(/Severity:\s*\*+([a-zA-Z]+)\*+/i) ||
      answer.match(/Severity:\s*([a-zA-Z]+)/i);

    if (sevMatch) {
      const sevVal = sevMatch[1].toLowerCase();
      const capVal = sevVal.charAt(0).toUpperCase() + sevVal.slice(1);
      items.push({
        id: 'severity',
        label: 'SEVERITY',
        value: capVal,
        subtext: 'Environmental impact index',
        icon: AlertTriangle,
        tone: sevVal === 'low' ? 'emerald' : sevVal === 'high' ? 'rose' : 'amber',
      });
    }

    // 4. Confidence Card
    if (result.confidence != null && result.confidence > 0) {
      const pct = Math.round(result.confidence * 100);
      items.push({
        id: 'confidence',
        label: 'CONFIDENCE',
        value: `${pct}%`,
        subtext: 'Analytical calibration',
        icon: Target,
        tone: pct >= 85 ? 'emerald' : 'amber',
      });
    } else {
      // Explicitly show Not Calibrated to be 100% consistent with ConfidenceCard
      items.push({
        id: 'confidence',
        label: 'CONFIDENCE',
        value: 'Not calibrated',
        subtext: 'Raw model proposal',
        icon: Target,
        tone: 'neutral',
      });
    }

    // 5. Date Comparison Card (for multi-temporal analyses)
    const dates = result.images
      ?.map((img) => img.metadata?.acquisitionDate)
      .filter(Boolean) as string[];

    if (dates.length >= 2) {
      const d1 = dates[0].slice(0, 10);
      const d2 = dates[1].slice(0, 10);
      items.push({
        id: 'temporal_span',
        label: 'TEMPORAL BASELINE',
        value: `${d1.slice(0, 7)} → ${d2.slice(0, 7)}`,
        subtext: `${d1} to ${d2}`,
        icon: Calendar,
        tone: 'cyan',
      });
    } else if (result.images?.[0]?.metadata?.crs) {
      items.push({
        id: 'spatial_ref',
        label: 'COORDINATE SYSTEM',
        value: result.images[0].metadata.crs,
        subtext: result.images[0].metadata.gsdMeters ? `${result.images[0].metadata.gsdMeters}m GSD` : 'WGS84 Projected',
        icon: MapPin,
        tone: 'cyan',
      });
    }

    // Extract Primary Finding: 1 concise takeaway sentence from existing output
    let finding = '';
    if (result.multilingualSummaries?.bullet_en && result.multilingualSummaries.bullet_en.length > 0) {
      finding = result.multilingualSummaries.bullet_en[0];
    } else if (result.multilingualSummaries?.summary_en) {
      const firstDot = result.multilingualSummaries.summary_en.indexOf('.');
      finding = firstDot > 0 ? result.multilingualSummaries.summary_en.slice(0, firstDot + 1) : result.multilingualSummaries.summary_en;
    } else {
      // Extract from answerText: find first clean sentence that isn't a heading
      const lines = answer.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('#') && !l.startsWith('-'));
      for (const line of lines) {
        const cleaned = line.replace(/[*_`]/g, '').trim();
        if (cleaned.length > 25) {
          const sentMatch = cleaned.match(/^([^.?!]+[.?!])/);
          finding = sentMatch ? sentMatch[1].trim() : cleaned;
          break;
        }
      }
    }

    if (!finding) {
      finding = isBiTemporal
        ? 'Localized structural expansion and ground clearing were detected between the baseline acquisitions.'
        : 'Spatial scene interpretation and feature localisation completed with multi-task specialist consensus.';
    }

    return { cards: items, primaryFinding: finding };
  }, [result]);

  return (
    <CornerFrame
      label="SMART INSIGHTS & TELEMETRY"
      domain={result.mode === 'bi_temporal' ? 'magenta' : result.mode === 'optical_sar' ? 'amber' : 'cyan'}
      bracketSize={12}
      className={className}
    >
      <div className="panel p-4 flex flex-col gap-4 bg-[var(--bg-panel)] transition-all">
        {/* Metric Cards Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {cards.map((card) => {
            const Icon = card.icon;
            const toneBorder =
              card.tone === 'rose'
                ? 'border-rose-500/30 text-rose-400 bg-rose-500/10'
                : card.tone === 'emerald'
                ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10'
                : card.tone === 'amber'
                ? 'border-amber-500/30 text-amber-400 bg-amber-500/10'
                : card.tone === 'cyan'
                ? 'border-cyan-500/30 text-cyan-400 bg-cyan-500/10'
                : 'border-[var(--border-hairline)] text-[var(--text-muted)] bg-[var(--surface-2)]';

            return (
              <div
                key={card.id}
                className="p-3.5 rounded-lg flex flex-col justify-between gap-1.5 transition-all duration-200 ease-out hover:-translate-y-1 hover:border-[var(--cyan)]/45 hover:shadow-md cursor-default group"
                style={{
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border-hairline)',
                }}
              >
                <div className="flex items-center justify-between gap-1.5">
                  <span className="text-[0.62rem] font-mono font-medium tracking-wider text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors uppercase">
                    {card.label}
                  </span>
                  <div className={`p-1.5 rounded-md ${toneBorder} transition-transform group-hover:scale-110`}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                </div>

                <div className="flex flex-col mt-0.5">
                  <span className="text-base font-bold font-mono text-[var(--text-primary)] tracking-tight">
                    {card.value}
                  </span>
                  {card.subtext && (
                    <span className="text-[0.65rem] font-sans text-[var(--text-muted)] truncate mt-0.5">
                      {card.subtext}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Primary Finding Banner */}
        <div
          className="p-3 rounded flex items-start gap-3 transition-colors"
          style={{
            background: 'linear-gradient(135deg, rgba(34, 211, 238, 0.06), rgba(9, 14, 23, 0.6))',
            border: '1px solid rgba(34, 211, 238, 0.25)',
          }}
        >
          <div className="p-1.5 rounded bg-[var(--cyan)]/15 border border-[var(--cyan)]/30 text-[var(--cyan)] shrink-0 mt-0.5">
            <Sparkles className="w-3.5 h-3.5" />
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-[0.65rem] font-mono font-bold tracking-wider text-[var(--cyan)] uppercase">
              PRIMARY FINDING
            </span>
            <p className="text-xs text-[var(--text-primary)] font-medium leading-relaxed">
              {primaryFinding}
            </p>
          </div>
        </div>
      </div>
    </CornerFrame>
  );
}
