// components/ChangeMapViewer.tsx
// Change detection mask overlay with opacity control, toggle, and class legend.
'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { Eye, EyeOff, Layers, Sliders, Crosshair, X, RotateCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ROIBounds } from '@/lib/types/analysis';

interface LegendItem {
  label: string;
  color: string;
}

interface ChangeMapViewerProps {
  baseImageUrl?: string | null;
  changeMaskUrl?: string | null;
  legend?: LegendItem[];
  className?: string;
  algorithmLabel?: string;  // displayed in toolbar, e.g. "Siamese U-Net Model" or "Grayscale Abs-Diff"
  roi?: ROIBounds | null;
  onSelectRoi?: (bounds: ROIBounds) => void;
  onClearRoi?: () => void;
}

const DEFAULT_LEGEND: LegendItem[] = [
  { label: 'New Urban / Built-up', color: '#FF5C5C' },
  { label: 'Vegetation Loss', color: '#FFB020' },
  { label: 'New Infrastructure', color: '#3ED0FF' },
  { label: 'No Change', color: '#1A2340' },
];

export function ChangeMapViewer({
  baseImageUrl,
  changeMaskUrl,
  legend = DEFAULT_LEGEND,
  className,
  algorithmLabel,
  roi = null,
  onSelectRoi,
  onClearRoi,
}: ChangeMapViewerProps) {
  const [showMask, setShowMask] = useState(true);
  const [opacity, setOpacity] = useState(0.75);
  const [baseError, setBaseError] = useState(false);
  const [maskError, setMaskError] = useState(false);
  const [isSelectMode, setIsSelectMode] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);


  useEffect(() => {
    setBaseError(false);
  }, [baseImageUrl]);

  useEffect(() => {
    setMaskError(false);
  }, [changeMaskUrl]);

  const getNormalizedPoint = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return null;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return { x, y };
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isSelectMode) return;
      e.preventDefault();
      const pt = getNormalizedPoint(e);
      if (!pt) return;
      setIsDragging(true);
      setDragStart(pt);
      setDragCurrent(pt);
    },
    [isSelectMode, getNormalizedPoint]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isDragging || !dragStart) return;
      const pt = getNormalizedPoint(e);
      if (!pt) return;
      setDragCurrent(pt);
    },
    [isDragging, dragStart, getNormalizedPoint]
  );

  const handleMouseUp = useCallback(() => {
    if (!isDragging || !dragStart || !dragCurrent) {
      setIsDragging(false);
      return;
    }
    setIsDragging(false);
    const x1 = Math.min(dragStart.x, dragCurrent.x);
    const x2 = Math.max(dragStart.x, dragCurrent.x);
    const y1 = Math.min(dragStart.y, dragCurrent.y);
    const y2 = Math.max(dragStart.y, dragCurrent.y);

    if (x2 - x1 > 0.01 && y2 - y1 > 0.01) {
      onSelectRoi?.({ x1, y1, x2, y2 });
    }
    setDragStart(null);
    setDragCurrent(null);
  }, [isDragging, dragStart, dragCurrent, onSelectRoi]);

  // Active ROI rectangle for rendering (draft or committed)
  const activeBox =
    isDragging && dragStart && dragCurrent
      ? {
          x1: Math.min(dragStart.x, dragCurrent.x),
          y1: Math.min(dragStart.y, dragCurrent.y),
          x2: Math.max(dragStart.x, dragCurrent.x),
          y2: Math.max(dragStart.y, dragCurrent.y),
        }
      : roi;

  return (
    <CornerFrame label="CHANGE DETECTION HEATMAP" domain="magenta" className={cn('w-full', className)}>
      <div className="panel overflow-hidden flex flex-col bg-[var(--bg-base)] transition-all duration-300 hover:border-[var(--magenta)]/35 hover:shadow-lg">
        {/* Controls Toolbar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex-wrap gap-2 text-xs">
          <div className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-[var(--accent-change)]" />
            <span className="font-mono text-[0.68rem] text-[var(--text-primary)] font-medium">
              {algorithmLabel ?? 'Pixel-level Change Mask'}
            </span>
            {roi && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[0.62rem] font-mono bg-[var(--cyan)]/15 border border-[var(--cyan)]/40 text-[var(--cyan)]">
                ROI Active
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            {/* ROI Selector Mode Button */}
            <button
              type="button"
              onClick={() => setIsSelectMode(!isSelectMode)}
              className={cn(
                'flex items-center gap-1.5 px-2.5 py-1 rounded text-[0.65rem] font-mono font-medium transition-all shadow-sm',
                isSelectMode
                  ? 'bg-[var(--cyan)] text-black border border-[var(--cyan)] shadow-[0_0_10px_rgba(62,208,255,0.4)]'
                  : 'bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-primary)] hover:border-[var(--cyan)]/50'
              )}
              title={isSelectMode ? 'Click and drag on the map to define an ROI' : 'Enable ROI selection mode'}
            >
              <Crosshair className="w-3 h-3" />
              {isSelectMode ? 'Drawing ROI (Drag on Map)' : 'Select Region'}
            </button>

            {/* Clear ROI Button */}
            {roi && (
              <button
                type="button"
                onClick={() => {
                  onClearRoi?.();
                  setIsSelectMode(false);
                }}
                className="flex items-center gap-1 px-2 py-1 rounded text-[0.65rem] font-mono text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] transition-all"
                title="Clear selected Region of Interest"
              >
                <RotateCcw className="w-3 h-3" />
                Clear ROI
              </button>
            )}

            {/* Opacity slider */}
            <div className="flex items-center gap-2">
              <Sliders className="w-3 h-3 text-[var(--text-muted)]" />
              <span className="text-[0.65rem] font-mono text-[var(--text-muted)]">Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-16 h-1 bg-[var(--bg-panel-elevated)] rounded appearance-none accent-[var(--accent-change)] cursor-pointer"
                disabled={!showMask}
                aria-label="Change mask opacity"
              />
              <span className="text-[0.65rem] font-mono text-[var(--text-faint)] w-8">
                {(opacity * 100).toFixed(0)}%
              </span>
            </div>

            {/* Mask Visibility Toggle */}
            <button
              type="button"
              onClick={() => setShowMask(!showMask)}
              className={cn(
                'flex items-center gap-1.5 px-2 py-1 rounded text-[0.65rem] font-mono transition-all',
                showMask
                  ? 'bg-[var(--accent-change)]/15 border border-[var(--accent-change)]/40 text-[var(--accent-change)]'
                  : 'bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-muted)]'
              )}
            >
              {showMask ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              {showMask ? 'Mask Visible' : 'Mask Hidden'}
            </button>
          </div>
        </div>

        {/* Viewport with Mask overlay & Interactive ROI Selection */}
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className={cn(
            'relative w-full h-[360px] bg-[#050b14] overflow-hidden flex items-center justify-center select-none',
            isSelectMode ? 'cursor-crosshair' : 'cursor-default'
          )}
        >
          {/* Telemetry Scan Line */}
          <div className="viewer-scan-line opacity-60 pointer-events-none" aria-hidden="true" />

          {/* Base Layer */}
          {baseImageUrl && !baseError ? (
            <img
              src={baseImageUrl}
              alt="Base imagery"
              className="w-full h-full object-cover pointer-events-none"
              onError={() => setBaseError(true)}
            />
          ) : (
            <div className="w-full h-full bg-[#081320] flex items-center justify-center pointer-events-none">
              <span className="text-xs font-mono text-[var(--text-faint)]">
                Base Satellite Scene (Optical Composite)
              </span>
            </div>
          )}

          {/* Change Mask Overlay */}
          {showMask && (
            <div
              className="absolute inset-0 transition-opacity duration-150 pointer-events-none"
              style={{ opacity }}
            >
              {changeMaskUrl && !maskError ? (
                <img
                  src={changeMaskUrl}
                  alt="Change detection mask"
                  className="w-full h-full object-cover"
                  onError={() => setMaskError(true)}
                />
              ) : (
                /* Simulated raster heatmap with CSS radial gradients */
                <div
                  className="w-full h-full"
                  style={{
                    background: `
                      radial-gradient(ellipse at 70% 30%, rgba(255,92,92,0.85) 0%, rgba(255,92,92,0.4) 25%, transparent 60%),
                      radial-gradient(ellipse at 35% 65%, rgba(255,176,32,0.8) 0%, rgba(255,176,32,0.3) 30%, transparent 65%),
                      radial-gradient(circle at 85% 75%, rgba(62,208,255,0.75) 0%, transparent 40%)
                    `,
                    mixBlendMode: 'screen',
                  }}
                />
              )}
            </div>
          )}

          {/* Visual ROI Selection Bounding Box */}
          {activeBox && (
            <div
              className="absolute pointer-events-none border-2 border-dashed border-[var(--cyan)] bg-[var(--cyan)]/15 shadow-[0_0_15px_rgba(62,208,255,0.45)] transition-all duration-75 z-20"
              style={{
                left: `${activeBox.x1 * 100}%`,
                top: `${activeBox.y1 * 100}%`,
                width: `${(activeBox.x2 - activeBox.x1) * 100}%`,
                height: `${(activeBox.y2 - activeBox.y1) * 100}%`,
              }}
            >
              {/* Corner accent reticles */}
              <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-[var(--cyan)]" />
              <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-[var(--cyan)]" />
              <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-[var(--cyan)]" />
              <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-[var(--cyan)]" />

              {/* Tag header badge */}
              <div className="absolute -top-6 left-0 px-1.5 py-0.5 rounded bg-black/85 border border-[var(--cyan)]/60 text-[0.6rem] font-mono font-bold text-[var(--cyan)] flex items-center gap-1 shadow">
                <Crosshair className="w-2.5 h-2.5" />
                <span>ROI TARGET</span>
              </div>
            </div>
          )}
        </div>


        {/* Semantic Legend Strip */}
        <div className="px-4 py-3 bg-[var(--bg-panel)] border-t border-[var(--border-hairline)] flex flex-col gap-2">
          <span className="hud-label">Change Classification Legend</span>
          <div className="flex flex-wrap gap-4">
            {legend.map((item, i) => (
              <div key={i} className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: item.color, border: '1px solid rgba(255,255,255,0.2)' }}
                />
                <span className="text-[0.68rem] font-mono text-[var(--text-primary)]">
                  {item.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </CornerFrame>
  );
}
