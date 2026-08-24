// components/OpticalSarViewer.tsx
// Cross-modal Optical + SAR viewer with blend control and dual-view sync.
'use client';

import { useState } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { Radar, Eye, SlidersHorizontal, SunMedium } from 'lucide-react';
import { cn } from '@/lib/utils';

interface OpticalSarViewerProps {
  opticalUrl?: string | null;
  sarUrl?: string | null;
  className?: string;
}

export function OpticalSarViewer({ opticalUrl, sarUrl, className }: OpticalSarViewerProps) {
  const [viewMode, setViewMode] = useState<'sideBySide' | 'blend'>('sideBySide');
  const [sarBlend, setSarBlend] = useState(0.5);

  return (
    <CornerFrame label="CROSS-MODAL OPTICAL + SAR FUSION" domain="amber" className={cn('w-full', className)}>
      <div className="panel overflow-hidden flex flex-col bg-[var(--bg-base)]">
        {/* Controls Toolbar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex-wrap gap-2 text-xs">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[var(--accent-signal)]">
              <SunMedium className="w-3.5 h-3.5" />
              <span className="font-mono text-[0.68rem]">Optical (LISS-IV)</span>
            </div>
            <span className="text-[var(--text-faint)] font-mono">✕</span>
            <div className="flex items-center gap-1.5 text-[var(--accent-warning)]">
              <Radar className="w-3.5 h-3.5" />
              <span className="font-mono text-[0.68rem]">SAR (RISAT-1A VV)</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {viewMode === 'blend' && (
              <div className="flex items-center gap-2">
                <span className="text-[0.65rem] font-mono text-[var(--text-muted)]">SAR Weight:</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={sarBlend}
                  onChange={(e) => setSarBlend(parseFloat(e.target.value))}
                  className="w-20 h-1 bg-[var(--bg-panel-elevated)] rounded appearance-none accent-[var(--accent-warning)] cursor-pointer"
                  aria-label="SAR fusion weight"
                />
                <span className="text-[0.65rem] font-mono text-[var(--accent-warning)] w-8">
                  {(sarBlend * 100).toFixed(0)}%
                </span>
              </div>
            )}

            <div className="flex items-center bg-[var(--bg-panel-elevated)] p-0.5 rounded border border-[var(--border-hairline)]">
              <button
                type="button"
                onClick={() => setViewMode('sideBySide')}
                className={cn(
                  'px-2 py-1 rounded text-[0.65rem] font-mono transition-colors',
                  viewMode === 'sideBySide'
                    ? 'bg-[var(--accent-signal)] text-[#05070D] font-medium'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                )}
              >
                Dual Channel
              </button>
              <button
                type="button"
                onClick={() => setViewMode('blend')}
                className={cn(
                  'px-2 py-1 rounded text-[0.65rem] font-mono transition-colors',
                  viewMode === 'blend'
                    ? 'bg-[var(--accent-warning)] text-[#05070D] font-medium'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                )}
              >
                Alpha Fusion Blend
              </button>
            </div>
          </div>
        </div>

        {/* Viewport Content */}
        {viewMode === 'sideBySide' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[var(--border-hairline)] h-[380px]">
            {/* Optical Channel */}
            <div className="relative bg-[#071322] flex flex-col items-center justify-center overflow-hidden">
              {opticalUrl ? (
                <img src={opticalUrl} alt="Optical Imagery" className="w-full h-full object-cover" />
              ) : (
                <div className="flex flex-col items-center gap-2 p-6 text-center">
                  <SunMedium className="w-8 h-8 text-[var(--accent-signal)] opacity-50" />
                  <span className="text-xs font-mono text-[var(--text-muted)]">Optical / RGB Sensor Feed</span>
                  <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">Surface Reflectance Domain</span>
                </div>
              )}
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-[0.6rem] font-mono bg-black/70 border border-[var(--accent-signal)]/40 text-[var(--accent-signal)]">
                Optical · 4-Band MSI
              </span>
            </div>

            {/* SAR Channel */}
            <div className="relative bg-[#14120a] flex flex-col items-center justify-center overflow-hidden">
              {sarUrl ? (
                <img src={sarUrl} alt="SAR Imagery" className="w-full h-full object-cover grayscale contrast-125" />
              ) : (
                <div className="flex flex-col items-center gap-2 p-6 text-center">
                  <Radar className="w-8 h-8 text-[var(--accent-warning)] opacity-50" />
                  <span className="text-xs font-mono text-[var(--text-muted)]">SAR C-band VV Backscatter</span>
                  <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">Dielectric & Roughness Domain</span>
                </div>
              )}
              <span className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-[0.6rem] font-mono bg-black/70 border border-[var(--accent-warning)]/40 text-[var(--accent-warning)]">
                SAR · C-Band VV (σ⁰ dB)
              </span>
            </div>
          </div>
        ) : (
          /* Fusion Blend View */
          <div className="relative w-full h-[380px] bg-[#071322] overflow-hidden flex items-center justify-center">
            {/* Optical Base */}
            <div className="absolute inset-0">
              {opticalUrl ? (
                <img src={opticalUrl} alt="Optical Base" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full bg-[#071322] flex items-center justify-center">
                  <span className="text-xs font-mono text-[var(--accent-signal)]">Optical Base</span>
                </div>
              )}
            </div>

            {/* SAR Blended Top Layer */}
            <div className="absolute inset-0" style={{ opacity: sarBlend, mixBlendMode: 'screen' }}>
              {sarUrl ? (
                <img src={sarUrl} alt="SAR Overlay" className="w-full h-full object-cover grayscale contrast-150" />
              ) : (
                <div
                  className="w-full h-full"
                  style={{
                    backgroundImage: 'repeating-linear-gradient(45deg, rgba(255,176,32,0.15) 0, rgba(255,176,32,0.15) 2px, transparent 0, transparent 8px)',
                  }}
                />
              )}
            </div>

            <div className="absolute top-3 left-3 px-2 py-1 rounded text-[0.65rem] font-mono bg-black/70 border border-[var(--border-hairline)] text-[var(--text-primary)]">
              Cross-Modal Alpha Stack: Optical {(100 - sarBlend * 100).toFixed(0)}% + SAR {(sarBlend * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>
    </CornerFrame>
  );
}
