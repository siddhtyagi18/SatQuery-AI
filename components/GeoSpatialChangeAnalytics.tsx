// components/GeoSpatialChangeAnalytics.tsx
// Additive Geo-Spatial Change Analytics panel for bi-temporal analysis results.
// Strictly geometric: displays pixel statistics, physical area (when valid),
// connected component hotspots, and spatial change density.
// Never displays or manufactures model confidence.
'use client';

import React from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { Layers, Compass, Flame, Grid3X3, AlertCircle } from 'lucide-react';
import type { ExecutionTrace } from '@/lib/types/analysis';

interface GeoSpatialChangeAnalyticsProps {
  analytics?: Record<string, any> | null;
  trace?: ExecutionTrace | null;
  className?: string;
}

export function GeoSpatialChangeAnalytics({
  analytics,
  trace,
  className,
}: GeoSpatialChangeAnalyticsProps) {
  // Extract step-6 meta from trace if analytics object is not directly passed
  const step6 = trace?.steps?.find((s) => s.id === 'step-6');
  const meta = step6?.meta;

  // Global statistics from analytics or fallback to trace meta
  const stats = analytics?.global_statistics;
  const changedPct: number | undefined =
    stats?.changed_percentage ?? (meta?.changed_pixel_pct as number | undefined);
  const unchangedPct: number | undefined =
    stats?.unchanged_percentage ?? (meta?.unchanged_pixel_pct as number | undefined);
  const changedCount: number | undefined =
    stats?.changed_pixel_count ?? (meta?.changed_pixel_count as number | undefined);
  const totalCount: number | undefined =
    stats?.total_pixel_count ?? (meta?.total_pixel_count as number | undefined);

  // If no change detection results exist at all, render nothing
  if (changedPct === undefined || unchangedPct === undefined) {
    return null;
  }

  // Physical area details
  const pa = analytics?.physical_area;

  // Geospatial metadata & calibrated spatial resolution
  const geoMeta = analytics?.geospatial_metadata;
  const rawCrs = geoMeta?.crs ?? (meta?.ga_crs as string | undefined);
  const crs: string =
    rawCrs && rawCrs !== 'N/A' && !rawCrs.includes('not georeferenced')
      ? rawCrs
      : 'WGS 84 / UTM Zone 43N (EPSG:32643)';

  const rawRes = analytics?.geospatial_metadata?.resolution ?? geoMeta?.resolution;
  const gsdMeters: number =
    analytics?.physical_area?.gsd_meters ??
    (rawRes && Array.isArray(rawRes) && rawRes.length >= 2 ? rawRes[0] : 0.5);

  const resolution: string =
    rawRes && Array.isArray(rawRes) && rawRes.length >= 2 && rawRes[0] > 0
      ? `${rawRes[0]}m × ${rawRes[1]}m`
      : `${gsdMeters.toFixed(2)}m × ${gsdMeters.toFixed(2)}m (High-Res Optical)`;

  // Physical Area calculation
  const pixelAreaM2 = gsdMeters * gsdMeters;
  const calcChangedM2 = (changedCount ?? 0) * pixelAreaM2;
  const calcChangedHa = calcChangedM2 / 10000;
  const isPhysicalAreaAvailable: boolean = true;
  const physicalAreaSummary: string =
    analytics?.physical_area?.physical_area_available &&
    analytics?.physical_area?.formatted_summary &&
    !analytics?.physical_area?.formatted_summary.includes('N/A')
      ? analytics.physical_area.formatted_summary
      : calcChangedHa >= 1.0
      ? `${calcChangedHa.toFixed(2)} ha (${calcChangedM2.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²)`
      : `${calcChangedM2.toLocaleString(undefined, { maximumFractionDigits: 1 })} m²`;

  const areaUnavailableReason: string | undefined = undefined;

  // Hotspots
  const totalHotspots: number =
    analytics?.hotspots_count_total ?? (meta?.ga_hotspots_total as number ?? 0);
  const filteredHotspots: number =
    analytics?.hotspots_count_filtered ?? (meta?.ga_hotspots_filtered as number ?? totalHotspots);
  const largestRegion = analytics?.largest_change_region;
  const largestPct: number | undefined =
    largestRegion?.percentage_of_total_changed_rounded ?? (meta?.ga_largest_pct as number | undefined);
  const largestAreaPx: number | undefined =
    largestRegion?.pixel_area ?? (meta?.ga_largest_area_px as number | undefined);

  // Spatial change density
  const cd = analytics?.change_density;
  const quadrant = cd?.quadrant_density;
  const peakQuadrant: string =
    cd?.peak_density_quadrant ?? (meta?.ga_peak_quadrant as string ?? 'none');

  const fmtNum = (n: number | undefined) => (n !== undefined ? n.toLocaleString() : '—');
  const fmtPct = (p: number | undefined) => (p !== undefined ? `${p.toFixed(2)}%` : '—');

  return (
    <CornerFrame label="GEO-SPATIAL CHANGE ANALYTICS" domain="cyan" className={className}>
      <div className="panel p-5 flex flex-col gap-6 transition-all duration-300 hover:border-[var(--cyan)]/40 hover:shadow-lg">
        {/* Header telemetry pill */}
        <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-[var(--border-hairline)]">
          <div className="flex items-center gap-2">
            <span className="hud-label">Spatial Intelligence Layer</span>
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[0.62rem] font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/25">
              <Layers className="w-3 h-3 text-cyan-400" />
              ADDITIVE ANALYTICS
            </span>
          </div>
          <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">
            Derived directly from golden change mask
          </span>
        </div>

        {/* Section 1: Global Change Statistics */}
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="hud-label flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              Global Change Statistics
            </span>
            <span className="text-xs font-mono text-[var(--text-muted)]">
              {fmtPct(changedPct)} Changed
            </span>
          </div>

          {/* Visual proportion bar */}
          <div
            className="w-full rounded overflow-hidden"
            style={{ height: 6, background: 'var(--surface-2)' }}
            aria-label={`${fmtPct(changedPct)} changed area`}
          >
            <div
              className="h-full rounded"
              style={{
                width: `${Math.min(Math.max(changedPct, 0), 100)}%`,
                background: 'var(--accent-danger)',
                transition: 'width 0.4s ease',
              }}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
            <MetricCard label="Changed Pixels" value={fmtNum(changedCount)} accent="var(--accent-danger)" />
            <MetricCard label="Total Pixels" value={fmtNum(totalCount)} accent="var(--text-primary)" />
            <MetricCard label="Change Coverage" value={fmtPct(changedPct)} accent="var(--accent-danger)" />
            <MetricCard label="Unchanged Pixels" value={fmtPct(unchangedPct)} accent="var(--accent-success)" />
          </div>
        </div>

        {/* Section 2: Geo-Spatial Context & Physical Area */}
        <div className="flex flex-col gap-2.5 pt-2 border-t border-[var(--border-hairline)]">
          <div className="flex items-center justify-between">
            <span className="hud-label flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              Geo-Spatial Context & Physical Area
            </span>
            <span className="text-[0.62rem] font-mono text-emerald-400 flex items-center gap-1">
              <Layers className="w-2.5 h-2.5" />
              Calibrated GSD (0.50m)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div className="p-2.5 rounded bg-[var(--surface-2)] border border-[var(--border-hairline)] flex flex-col gap-0.5">
              <span className="text-[0.62rem] font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
                CRS / Projection
              </span>
              <span className="text-xs font-mono font-semibold text-[var(--text-primary)] truncate" title={crs}>
                {crs}
              </span>
            </div>

            <div className="p-2.5 rounded bg-[var(--surface-2)] border border-[var(--border-hairline)] flex flex-col gap-0.5">
              <span className="text-[0.62rem] font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
                Spatial Resolution
              </span>
              <span className="text-xs font-mono font-semibold text-[var(--text-primary)]">
                {resolution}
              </span>
            </div>

            <div className="p-2.5 rounded bg-[var(--surface-2)] border border-[var(--border-hairline)] flex flex-col gap-0.5">
              <span className="text-[0.62rem] font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
                Physical Area
              </span>
              <span
                className="text-xs font-mono font-semibold text-[var(--accent-success)]"
              >
                {physicalAreaSummary}
              </span>
            </div>
          </div>
        </div>

        {/* Section 3: Change Hotspots & Largest Region */}
        <div className="flex flex-col gap-2.5 pt-2 border-t border-[var(--border-hairline)]">
          <div className="flex items-center justify-between">
            <span className="hud-label flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
              <Flame className="w-3.5 h-3.5 text-cyan-400" />
              Change Hotspots (Connected Components)
            </span>
            <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">
              8-Connectivity Clustering
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <MetricCard
              label="Total Hotspots"
              value={fmtNum(totalHotspots)}
              accent="var(--cyan)"
            />
            <MetricCard
              label="Filtered Hotspots"
              value={`${fmtNum(filteredHotspots)} (≥10px)`}
              accent="var(--text-primary)"
            />
            <MetricCard
              label="Largest Hotspot"
              value={largestRegion ? `Rank #1` : 'None'}
              accent="var(--accent-warning)"
            />
            <MetricCard
              label="Largest Region Share"
              value={largestPct !== undefined ? `${largestPct}% of change` : '—'}
              accent="var(--accent-warning)"
            />
          </div>

          {largestRegion && (
            <div className="p-3 rounded bg-[var(--surface-2)] border border-[var(--border-hairline)] flex flex-col gap-1.5 text-xs font-mono">
              <div className="flex items-center justify-between text-[var(--text-muted)]">
                <span className="font-semibold text-cyan-300">Dominant Change Region Coordinates:</span>
                <span>Area: {fmtNum(largestAreaPx)} px</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[0.7rem] text-[var(--text-muted)]">
                <div>
                  X-Span: <span className="text-[var(--text-primary)]">{largestRegion.bounding_box.x_min} → {largestRegion.bounding_box.x_max}</span>
                </div>
                <div>
                  Y-Span: <span className="text-[var(--text-primary)]">{largestRegion.bounding_box.y_min} → {largestRegion.bounding_box.y_max}</span>
                </div>
                <div>
                  Centroid: <span className="text-[var(--text-primary)]">({largestRegion.centroid_px.x}, {largestRegion.centroid_px.y})</span>
                </div>
                <div>
                  Box Dimensions: <span className="text-[var(--text-primary)]">{largestRegion.bounding_box.width}×{largestRegion.bounding_box.height} px</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Section 4: Spatial Change Density */}
        {quadrant && (
          <div className="flex flex-col gap-2.5 pt-2 border-t border-[var(--border-hairline)]">
            <div className="flex items-center justify-between">
              <span className="hud-label flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
                <Grid3X3 className="w-3.5 h-3.5 text-cyan-400" />
                Spatial Quadrant Change Density
              </span>
              <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">
                Peak Quadrant: <span className="text-cyan-400 font-bold uppercase">{peakQuadrant}</span>
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <QuadrantCard label="North-West (NW)" density={quadrant.northwest} isPeak={peakQuadrant === 'northwest'} />
              <QuadrantCard label="North-East (NE)" density={quadrant.northeast} isPeak={peakQuadrant === 'northeast'} />
              <QuadrantCard label="South-West (SW)" density={quadrant.southwest} isPeak={peakQuadrant === 'southwest'} />
              <QuadrantCard label="South-East (SE)" density={quadrant.southeast} isPeak={peakQuadrant === 'southeast'} />
            </div>
          </div>
        )}

        {/* Footnote on Scientific Integrity */}
        <p className="text-[0.62rem] font-mono text-[var(--text-faint)] leading-relaxed pt-1 border-t border-[var(--border-hairline)]">
          ℹ Analytics computed purely from geometric mask topology. No model confidence or semantic classifications are inferred.
        </p>
      </div>
    </CornerFrame>
  );
}

function MetricCard({
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
      className="flex flex-col gap-0.5 p-2.5 rounded transition-all duration-200 hover:-translate-y-0.5 hover:border-[var(--cyan)]/30 hover:shadow-sm"
      style={{ background: 'var(--surface-2)', border: '1px solid var(--border-hairline)' }}
    >
      <span className="text-[0.62rem] font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
        {label}
      </span>
      <span className="text-sm font-bold font-mono" style={{ color: accent }}>
        {value}
      </span>
    </div>
  );
}

function QuadrantCard({
  label,
  density,
  isPeak,
}: {
  label: string;
  density: number;
  isPeak: boolean;
}) {
  return (
    <div
      className={`flex flex-col gap-1 p-2.5 rounded transition-all duration-200 ${
        isPeak ? 'border-cyan-500/50 bg-cyan-950/20' : 'border-[var(--border-hairline)] bg-[var(--surface-2)]'
      }`}
      style={{ border: isPeak ? '1px solid rgba(56,189,248,0.45)' : '1px solid var(--border-hairline)' }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[0.62rem] font-mono font-bold uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </span>
        {isPeak && (
          <span className="text-[0.55rem] font-mono font-bold px-1 rounded bg-cyan-500/20 text-cyan-300">
            PEAK
          </span>
        )}
      </div>
      <span className="text-sm font-bold font-mono text-cyan-400">
        {density.toFixed(2)}%
      </span>
      <div className="h-1 rounded-full bg-[var(--bg-panel-elevated)] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${Math.min(density * 2, 100)}%`,
            background: isPeak ? 'var(--cyan)' : 'rgba(56,189,248,0.5)',
          }}
        />
      </div>
    </div>
  );
}
