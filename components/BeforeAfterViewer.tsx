// components/BeforeAfterViewer.tsx
// Bi-temporal comparison viewer with interactive slider/swipe or side-by-side mode.
'use client';

import { useState, useRef, useCallback } from 'react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { Split, Columns, Calendar, MoveHorizontal } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BeforeAfterViewerProps {
  beforeUrl?: string | null;
  afterUrl?: string | null;
  beforeDate?: string | null;
  afterDate?: string | null;
  className?: string;
}

export function BeforeAfterViewer({
  beforeUrl,
  afterUrl,
  beforeDate = 'T1 (Jan 2022)',
  afterDate = 'T2 (Jan 2024)',
  className,
}: BeforeAfterViewerProps) {
  const [mode, setMode] = useState<'slider' | 'sideBySide'>('slider');
  const [sliderPos, setSliderPos] = useState(50); // percentage
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isDragging && e.type !== 'pointerdown') return;
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      setSliderPos((x / rect.width) * 100);
    },
    [isDragging]
  );

  return (
    <CornerFrame label="BI-TEMPORAL VISUAL COMPARISON" domain="magenta" className={cn('w-full', className)}>
      <div className="panel overflow-hidden flex flex-col bg-[var(--bg-base)]">
        {/* Header toolbar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] text-xs">
          <div className="flex items-center gap-4 text-[0.68rem] font-mono">
            <div className="flex items-center gap-1.5 text-[var(--accent-signal)]">
              <Calendar className="w-3.5 h-3.5" />
              <span>Before: {beforeDate}</span>
            </div>
            <div className="flex items-center gap-1.5 text-[var(--accent-change)]">
              <Calendar className="w-3.5 h-3.5" />
              <span>After: {afterDate}</span>
            </div>
          </div>

          <div className="flex items-center bg-[var(--bg-panel-elevated)] p-0.5 rounded border border-[var(--border-hairline)]">
            <button
              type="button"
              onClick={() => setMode('slider')}
              className={cn(
                'flex items-center gap-1 px-2 py-1 rounded text-[0.65rem] font-mono font-medium transition-colors',
                mode === 'slider'
                  ? 'bg-[var(--accent-signal)] text-[#05070D]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              )}
            >
              <Split className="w-3 h-3" />
              Slider Swipe
            </button>
            <button
              type="button"
              onClick={() => setMode('sideBySide')}
              className={cn(
                'flex items-center gap-1 px-2 py-1 rounded text-[0.65rem] font-mono font-medium transition-colors',
                mode === 'sideBySide'
                  ? 'bg-[var(--accent-signal)] text-[#05070D]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              )}
            >
              <Columns className="w-3 h-3" />
              Side-by-Side
            </button>
          </div>
        </div>

        {/* Comparison Content */}
        {mode === 'slider' ? (
          <div
            ref={containerRef}
            onPointerDown={(e) => {
              setIsDragging(true);
              handlePointerMove(e);
            }}
            onPointerMove={handlePointerMove}
            onPointerUp={() => setIsDragging(false)}
            onPointerLeave={() => setIsDragging(false)}
            className="relative w-full h-[400px] select-none cursor-ew-resize overflow-hidden bg-[#0a101d]"
          >
            {/* After Image (Background) */}
            <div className="absolute inset-0 flex items-center justify-center">
              {afterUrl ? (
                <img src={afterUrl} alt="After state" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full bg-gradient-to-tr from-[#160d2b] to-[#251347] flex items-center justify-center">
                  <span className="font-mono text-xs text-[var(--accent-change)] opacity-70">
                    T2 Acquisition (Urban Expansion Layer)
                  </span>
                </div>
              )}
            </div>

            {/* Before Image (Clipped Overlay) */}
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ width: `${sliderPos}%` }}
            >
              <div className="absolute top-0 left-0 w-full h-full min-w-[100%] min-h-full">
                {beforeUrl ? (
                  <img
                    src={beforeUrl}
                    alt="Before state"
                    className="w-full h-full object-cover"
                    style={{ width: containerRef.current?.clientWidth ?? '100%' }}
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-tr from-[#091b29] to-[#0d3447] flex items-center justify-center">
                    <span className="font-mono text-xs text-[var(--accent-signal)] opacity-70">
                      T1 Baseline Acquisition (Agricultural / Vegetation Layer)
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Divider Line */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-[var(--accent-signal)] shadow-[0_0_12px_rgba(62,208,255,0.9)] z-20 pointer-events-none"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-[var(--surface-1)] border-2 border-[var(--accent-signal)] flex items-center justify-center shadow-[0_0_16px_rgba(62,208,255,0.6)] transition-transform duration-150">
                <MoveHorizontal className="w-3.5 h-3.5 text-[var(--accent-signal)]" />
              </div>
            </div>

            {/* Corner Indicators */}
            <span className="absolute bottom-3 left-3 px-2.5 py-1 rounded text-[0.6rem] font-mono bg-black/70 backdrop-blur-sm border border-[var(--accent-signal)]/40 text-[var(--accent-signal)] z-10 shadow-lg">
              ◄ {beforeDate}
            </span>
            <span className="absolute bottom-3 right-3 px-2.5 py-1 rounded text-[0.6rem] font-mono bg-black/70 backdrop-blur-sm border border-[var(--accent-change)]/40 text-[var(--accent-change)] z-10 shadow-lg">
              {afterDate} ►
            </span>
          </div>
        ) : (
          /* Side by side */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[var(--border-hairline)] h-[400px]">
            <div className="relative h-full bg-[#091b29] overflow-hidden flex items-center justify-center">
              {beforeUrl ? (
                <img src={beforeUrl} alt="Before" className="w-full h-full object-cover" />
              ) : (
                <span className="font-mono text-xs text-[var(--accent-signal)]">
                  {beforeDate} Baseline
                </span>
              )}
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-[0.6rem] font-mono bg-black/70 text-[var(--accent-signal)]">
                {beforeDate}
              </span>
            </div>

            <div className="relative h-full bg-[#160d2b] overflow-hidden flex items-center justify-center">
              {afterUrl ? (
                <img src={afterUrl} alt="After" className="w-full h-full object-cover" />
              ) : (
                <span className="font-mono text-xs text-[var(--accent-change)]">
                  {afterDate} Target
                </span>
              )}
              <span className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-[0.6rem] font-mono bg-black/70 text-[var(--accent-change)]">
                {afterDate}
              </span>
            </div>
          </div>
        )}
      </div>
    </CornerFrame>
  );
}
