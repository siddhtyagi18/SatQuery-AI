// components/ChangeMapViewer.tsx
// Change detection mask overlay with opacity control, toggle, and class legend.
'use client';

import { useState } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { Eye, EyeOff, Layers, Sliders } from 'lucide-react';
import { cn } from '@/lib/utils';

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
}: ChangeMapViewerProps) {
  const [showMask, setShowMask] = useState(true);
  const [opacity, setOpacity] = useState(0.75);

  return (
    <CornerFrame label="CHANGE DETECTION HEATMAP" domain="magenta" className={cn('w-full', className)}>
      <div className="panel overflow-hidden flex flex-col bg-[var(--bg-base)]">
        {/* Controls Toolbar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex-wrap gap-2 text-xs">
          <div className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-[var(--accent-change)]" />
            <span className="font-mono text-[0.68rem] text-[var(--text-primary)] font-medium">
              {algorithmLabel ?? 'Pixel-level Change Mask'}
            </span>
          </div>

          <div className="flex items-center gap-4">
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
                className="w-20 h-1 bg-[var(--bg-panel-elevated)] rounded appearance-none accent-[var(--accent-change)] cursor-pointer"
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

        {/* Viewport with Mask overlay */}
        <div className="relative w-full h-[360px] bg-[#050b14] overflow-hidden flex items-center justify-center">
          {/* Telemetry Scan Line */}
          <div className="viewer-scan-line opacity-60" aria-hidden="true" />

          {/* Base Layer */}
          {baseImageUrl ? (
            <img src={baseImageUrl} alt="Base imagery" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full bg-[#081320] flex items-center justify-center">
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
              {changeMaskUrl ? (
                <img src={changeMaskUrl} alt="Change detection mask" className="w-full h-full object-cover" />
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
