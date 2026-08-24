// components/ImageMetadata.tsx
// HUD-style metadata readout panel for an uploaded image.
'use client';

import { cn, formatBytes } from '@/lib/utils';
import type { UploadedImage } from '@/lib/types/analysis';
import { MonoField } from '@/components/ui/MonoField';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Skeleton } from '@/components/ui/LoadingSkeletonPanel';
import { Cpu } from 'lucide-react';

const MODALITY_COLORS: Record<string, string> = {
  optical: 'bg-[#3ED0FF]/10 text-[#3ED0FF] border-[#3ED0FF]/25',
  sar: 'bg-[#FFB020]/10 text-[#FFB020] border-[#FFB020]/25',
  multispectral: 'bg-[#C084FC]/10 text-[#C084FC] border-[#C084FC]/25',
  unknown: 'bg-[#4A5270]/20 text-[#8B93A7] border-[#4A5270]/30',
};

interface ImageMetadataProps {
  image: UploadedImage | null;
  loading?: boolean;
  label?: string;
  className?: string;
}

export function ImageMetadata({ image, loading, label, className }: ImageMetadataProps) {
  if (loading) {
    return (
      <div className={cn('panel p-4 flex flex-col gap-3', className)}>
        <Skeleton className="h-3 w-32" />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="flex flex-col gap-1">
              <Skeleton className="h-2 w-16" />
              <Skeleton className="h-3 w-24" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!image) return null;

  const { metadata } = image;
  const modalityColor = MODALITY_COLORS[metadata.modality] ?? MODALITY_COLORS.unknown;

  return (
    <div
      className={cn('flex flex-col gap-3 p-4 rounded-md', className)}
      style={{ background: 'var(--bg-panel)', border: '1px solid var(--border-hairline)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--accent-signal)', opacity: 0.7 }} aria-hidden="true" />
          <span className="hud-label">{label ?? 'Image Metadata'}</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'inline-flex items-center gap-1 px-2 py-0.5 rounded text-[0.6rem] font-mono uppercase tracking-wide border',
              modalityColor
            )}
          >
            {metadata.modality}
          </span>
          {metadata.modalityDetectionConfidence != null && (
            <span className="text-[0.6rem] font-mono" style={{ color: 'var(--text-faint)' }}>
              {Math.round(metadata.modalityDetectionConfidence * 100)}% conf.
            </span>
          )}
        </div>
      </div>

      {/* Grid of HUD fields */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
        <MonoField label="Format" value={metadata.fileFormat} />
        <MonoField label="File Size" value={formatBytes(metadata.fileSizeBytes)} />
        <MonoField
          label="Dimensions"
          value={metadata.widthPx != null && metadata.heightPx != null ? `${metadata.widthPx} × ${metadata.heightPx}` : null}
          unavailableText="Not available for this format"
        />
        <MonoField label="Band Count" value={metadata.bandCount} />
        <MonoField label="CRS / EPSG" value={metadata.crs} />
        <MonoField label="GSD" value={metadata.gsdMeters} unit="m/px" />
        <MonoField
          label="Acquisition Date"
          value={metadata.acquisitionDate}
          unavailableText="Not extractable"
        />
        <MonoField label="File Name" value={metadata.fileName} className="col-span-2" />
      </div>

      {/* Note for plain PNG/JPEG */}
      {(metadata.fileFormat === 'PNG' || metadata.fileFormat === 'JPEG') && (
        <p className="text-[0.6rem] font-mono leading-relaxed" style={{ color: 'var(--text-faint)' }}>
          ⓘ Geospatial metadata (CRS, GSD, band count, acquisition date) is not available for PNG/JPEG inputs.
          Upload a GeoTIFF to enable full metadata extraction.
        </p>
      )}
    </div>
  );
}
