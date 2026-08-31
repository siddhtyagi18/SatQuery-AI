// components/SatelliteViewer.tsx
// Lightweight custom SatelliteViewer with pan/zoom via CSS transforms.
// HUD overlay with zoom level, coordinates, reset button, and reticle corner frame.
'use client';

import { useState, useRef, useCallback } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Maximize2, Crosshair } from 'lucide-react';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { cn } from '@/lib/utils';

interface SatelliteViewerProps {
  imageUrl?: string | null;
  altText?: string;
  crs?: string | null;
  resolution?: string | number | null;
  children?: React.ReactNode;
  className?: string;
  title?: string;
}

export function SatelliteViewer({
  imageUrl,
  altText = 'Satellite Imagery View',
  crs = 'EPSG:32643',
  resolution = '5.8m',
  children,
  className,
  title = 'Optical Imagery Layer',
}: SatelliteViewerProps) {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomDelta = e.deltaY < 0 ? 0.15 : -0.15;
    setScale((prev) => Math.min(Math.max(0.5, prev + zoomDelta), 4));
  };

  const zoomIn = () => setScale((prev) => Math.min(prev + 0.25, 4));
  const zoomOut = () => setScale((prev) => Math.max(prev - 0.25, 0.5));
  const resetView = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  return (
    <CornerFrame label="SATELLITE VIEWPORT" domain="cyan" className={cn('w-full', className)}>
      <div className="panel overflow-hidden relative flex flex-col min-h-[420px] bg-[var(--bg-base)]">
        {/* HUD Top Bar */}
        <div className="flex items-center justify-between px-3.5 py-2 bg-[var(--bg-panel)]/90 border-b border-[var(--border-hairline)] z-20 text-[0.65rem] font-mono">
          <div className="flex items-center gap-2">
            <Crosshair className="w-3.5 h-3.5 text-[var(--accent-signal)]" />
            <span className="text-[var(--text-primary)] font-medium">{title}</span>
          </div>

          <div className="flex items-center gap-3 text-[var(--text-muted)]">
            <span>CRS: <strong className="text-[var(--accent-signal)]">{crs ?? 'WGS84 / UTM'}</strong></span>
            <span>GSD: <strong className="text-[var(--accent-signal)]">{resolution ?? '—'}</strong></span>
            <span>Zoom: <strong className="text-[var(--accent-signal)]">{(scale * 100).toFixed(0)}%</strong></span>
          </div>
        </div>

        {/* Viewport Area */}
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          className={cn(
            'relative flex-1 w-full h-[400px] overflow-hidden select-none flex items-center justify-center',
            isDragging ? 'cursor-grabbing' : 'cursor-grab'
          )}
        >
          {/* Subtle Grid lines */}
          <div
            className="absolute inset-0 pointer-events-none opacity-20"
            style={{
              backgroundImage: 'linear-gradient(to right, rgba(62,208,255,0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(62,208,255,0.15) 1px, transparent 1px)',
              backgroundSize: '40px 40px',
            }}
          />

          {/* Telemetry Scan Line */}
          <div className="viewer-scan-line" aria-hidden="true" />

          {/* Transform Container */}
          <div
            className="relative transition-transform duration-100 ease-out"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transformOrigin: 'center center',
            }}
          >
            {imageUrl ? (
              <img
                src={imageUrl}
                alt={altText}
                className="max-w-none w-[600px] h-[400px] object-cover rounded shadow-2xl pointer-events-none transition-opacity duration-300"
                draggable={false}
              />
            ) : (
              <div className="w-[600px] h-[400px] bg-gradient-to-br from-[#0c1829] via-[#092237] to-[#04101e] rounded flex flex-col items-center justify-center p-6 border border-[var(--border-hairline)] text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-[radial-gradient(#3ED0FF_1px,transparent_1px)] [background-size:16px_16px] opacity-10" />
                <div className="w-16 h-16 rounded-full border border-[var(--accent-signal)]/40 flex items-center justify-center mb-3 animate-pulse">
                  <Crosshair className="w-8 h-8 text-[var(--accent-signal)]" />
                </div>
                <span className="text-sm font-semibold text-[var(--text-primary)] font-heading">
                  High-Resolution Multispectral Tile
                </span>
                <span className="text-xs text-[var(--text-muted)] font-mono mt-1">
                  Co-registered Sensor Stream · 8192 × 8192 px
                </span>
              </div>
            )}

            {/* Overlays slot (e.g. Grounding boxes) */}
            {children}
          </div>
        </div>

        {/* HUD Controls Bottom Floating Toolbar */}
        <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-[var(--bg-panel-elevated)]/90 backdrop-blur-md border border-[var(--border-hairline)] p-1 rounded-md z-20 shadow-lg">
          <button
            type="button"
            onClick={zoomIn}
            className="p-1.5 rounded hover:bg-[var(--bg-panel-hover)] text-[var(--text-primary)] transition-colors"
            title="Zoom In"
            aria-label="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={zoomOut}
            className="p-1.5 rounded hover:bg-[var(--bg-panel-hover)] text-[var(--text-primary)] transition-colors"
            title="Zoom Out"
            aria-label="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={resetView}
            className="p-1.5 rounded hover:bg-[var(--bg-panel-hover)] text-[var(--text-primary)] transition-colors"
            title="Reset View"
            aria-label="Reset viewport view"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </CornerFrame>
  );
}
