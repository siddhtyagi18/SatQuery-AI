// components/ImageUploader.tsx
// Drag-and-drop image uploader — Mode-aware slot rendering with clear role labeling and "Choose Image" button.
'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { cn, formatBytes } from '@/lib/utils';
import type { AnalysisMode, UploadedImage } from '@/lib/types/analysis';
import {
  X,
  FileImage,
  AlertCircle,
  CheckCircle2,
  Loader2,
  UploadCloud,
} from 'lucide-react';

const MODE_SLOTS: Record<
  AnalysisMode,
  {
    role: UploadedImage['role'];
    label: string;
    badge: string;
    hint: string;
    domain: 'cyan' | 'magenta' | 'amber';
  }[]
> = {
  single_image: [
    {
      role: 'single',
      label: 'Single Satellite Scene',
      badge: 'OPTICAL / SAR PAYLOAD',
      hint: 'High-resolution GeoTIFF (EPSG geospatial) or PNG / JPEG image',
      domain: 'cyan',
    },
  ],
  bi_temporal: [
    {
      role: 'before',
      label: 'Image 1 · Earlier Acquisition (T1)',
      badge: 'PRE-CHANGE BASELINE',
      hint: 'Earlier temporal satellite tile (GeoTIFF, PNG, JPG)',
      domain: 'magenta',
    },
    {
      role: 'after',
      label: 'Image 2 · Later Acquisition (T2)',
      badge: 'POST-CHANGE TARGET',
      hint: 'Later temporal satellite tile (GeoTIFF, PNG, JPG)',
      domain: 'magenta',
    },
  ],
  optical_sar: [
    {
      role: 'optical',
      label: 'Sensor 1 · Optical / Multispectral',
      badge: 'REFLECTANCE SPECTRUM',
      hint: 'Optical or multispectral GeoTIFF / PNG (RGB / NIR bands)',
      domain: 'cyan',
    },
    {
      role: 'sar',
      label: 'Sensor 2 · Synthetic Aperture Radar',
      badge: 'MICROWAVE BACKSCATTER',
      hint: 'SAR (C/L/X-band) GeoTIFF or PNG radar amplitude tile',
      domain: 'amber',
    },
  ],
};

const ACCEPTED_TYPES = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
const MAX_SIZE_MB = 500;

interface UploadSlotProps {
  slot: {
    role: UploadedImage['role'];
    label: string;
    badge: string;
    hint: string;
    domain: 'cyan' | 'magenta' | 'amber';
  };
  uploaded: UploadedImage | null;
  uploading: boolean;
  error: string | null;
  onFile: (file: File, role: UploadedImage['role']) => void;
  onRemove: () => void;
  disabled?: boolean;
}

function UploadSlot({
  slot,
  uploaded,
  uploading,
  error,
  onFile,
  onRemove,
  disabled,
}: UploadSlotProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [previewError, setPreviewError] = useState(false);

  useEffect(() => {
    setPreviewError(false);
  }, [uploaded?.previewUrl]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file, slot.role);
    },
    [disabled, onFile, slot.role]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFile(file, slot.role);
    e.target.value = '';
  };

  const domainVar = `var(--${slot.domain})`;
  const hasContent = !!uploaded || uploading;

  return (
    <div className="flex flex-col gap-2.5">
      {/* Slot Header Label & Badge */}
      <div className="flex items-center justify-between">
        <span className="font-heading font-semibold text-sm text-[var(--text-primary)]">
          {slot.label}
        </span>
        <span
          className="font-mono text-[0.6rem] tracking-wider uppercase px-2 py-0.5 rounded font-semibold"
          style={{
            background: `color-mix(in srgb, ${domainVar} 12%, transparent)`,
            border: `1px solid color-mix(in srgb, ${domainVar} 30%, transparent)`,
            color: domainVar,
          }}
        >
          {slot.badge}
        </span>
      </div>

      {/* Main Drag and Drop Box */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !hasContent && !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label={`Upload ${slot.label}`}
        onKeyDown={(e) =>
          e.key === 'Enter' && !hasContent && !disabled && inputRef.current?.click()
        }
        className={cn(
          'relative flex flex-col items-center justify-center rounded-xl transition-all duration-200 min-h-[190px] p-6 text-center border overflow-hidden',
          !hasContent && !disabled ? 'cursor-pointer hover:border-[var(--text-muted)]' : '',
          dragging ? 'border-dashed' : 'border-solid'
        )}
        style={{
          background: dragging
            ? `color-mix(in srgb, ${domainVar} 8%, var(--surface-2))`
            : 'var(--surface-1)',
          borderColor: dragging
            ? domainVar
            : error
            ? 'var(--red)'
            : 'var(--border-hairline)',
          boxShadow: '0 4px 20px -2px rgba(0,0,0,0.25)',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleFileChange}
          className="sr-only"
          aria-label={`Choose ${slot.label} file`}
          disabled={disabled}
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-3 py-4">
            <Loader2 className="w-8 h-8 animate-spin" style={{ color: domainVar }} />
            <span className="font-mono text-xs text-[var(--text-muted)]">
              Uploading &amp; Extracting Metadata…
            </span>
          </div>
        ) : uploaded ? (
          <div className="w-full h-full relative flex flex-col items-center justify-center">
            {/* Image Preview Thumbnail */}
            {uploaded.previewUrl && !previewError ? (
              <div className="w-full relative rounded-lg overflow-hidden border border-[var(--border-hairline)] max-h-[220px]">
                <img
                  src={uploaded.previewUrl}
                  alt={`Preview of ${uploaded.metadata.fileName}`}
                  className="w-full h-full object-cover"
                  onError={() => setPreviewError(true)}
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 py-6">
                <FileImage className="w-10 h-10" style={{ color: domainVar, opacity: 0.8 }} />
                <span className="font-mono text-xs text-[var(--text-primary)] font-medium">
                  {uploaded.metadata.fileName}
                </span>
                <span className="text-[0.65rem] text-[var(--text-muted)]">
                  GeoTIFF Multiband Tile (Preview generated after pipeline ingest)
                </span>
              </div>
            )}

            {/* Bottom Info Bar */}
            <div className="w-full mt-3 flex items-center justify-between gap-2 px-3 py-2 rounded-lg bg-[var(--surface-2)] border border-[var(--border-hairline)] text-xs font-mono">
              <div className="flex items-center gap-1.5 min-w-0">
                <CheckCircle2 className="w-3.5 h-3.5 text-[var(--green)] flex-shrink-0" />
                <span className="truncate text-[var(--text-primary)]">
                  {uploaded.metadata.fileName}
                </span>
              </div>
              <span className="text-[var(--text-faint)] flex-shrink-0 text-[0.65rem]">
                {formatBytes(uploaded.metadata.fileSizeBytes)}
              </span>
            </div>

            {/* Remove Button */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onRemove();
              }}
              className="absolute top-2 right-2 w-7 h-7 rounded-full flex items-center justify-center bg-[var(--surface-0)]/90 border border-[var(--red)]/40 hover:bg-[var(--red)]/20 transition-colors shadow-md cursor-pointer"
              aria-label={`Remove ${slot.label}`}
              title="Remove this image"
            >
              <X className="w-3.5 h-3.5 text-[var(--red)]" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center transition-transform"
              style={{
                background: `color-mix(in srgb, ${domainVar} 12%, transparent)`,
                border: `1px solid color-mix(in srgb, ${domainVar} 25%, transparent)`,
              }}
            >
              <UploadCloud className="w-6 h-6" style={{ color: domainVar }} />
            </div>

            <div className="flex flex-col gap-1">
              <span className="text-sm font-semibold text-[var(--text-primary)] font-heading">
                Drag &amp; drop satellite image here
              </span>
              <span className="text-xs text-[var(--text-muted)] max-w-xs leading-relaxed">
                {slot.hint}
              </span>
            </div>

            {/* Normal "Choose Image" / "Upload Image" Button */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
              disabled={disabled}
              className="mt-1 inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all shadow-sm cursor-pointer"
              style={{
                background: 'var(--surface-2)',
                border: `1px solid color-mix(in srgb, ${domainVar} 40%, transparent)`,
                color: domainVar,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = `color-mix(in srgb, ${domainVar} 15%, var(--surface-2))`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'var(--surface-2)';
              }}
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Choose Image File</span>
            </button>

            <span className="text-[0.65rem] font-mono text-[var(--text-faint)] mt-1">
              GeoTIFF · PNG · JPG (Up to {MAX_SIZE_MB}MB)
            </span>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-2 p-2 rounded bg-[var(--red)]/10 border border-[var(--red)]/30 text-xs text-[var(--red)]">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

interface ImageUploaderProps {
  mode: AnalysisMode;
  uploads: Partial<Record<UploadedImage['role'], UploadedImage>>;
  uploading: Partial<Record<UploadedImage['role'], boolean>>;
  errors: Partial<Record<UploadedImage['role'], string>>;
  onFile: (file: File, role: UploadedImage['role']) => void;
  onRemove: (role: UploadedImage['role']) => void;
  disabled?: boolean;
}

export function ImageUploader({
  mode,
  uploads,
  uploading,
  errors,
  onFile,
  onRemove,
  disabled,
}: ImageUploaderProps) {
  const slots = MODE_SLOTS[mode] ?? MODE_SLOTS.single_image;

  return (
    <div className="flex flex-col gap-4">
      <div
        className={cn(
          'grid gap-6',
          slots.length === 1 ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-2'
        )}
      >
        {slots.map((slot) => (
          <UploadSlot
            key={slot.role}
            slot={slot}
            uploaded={uploads[slot.role] ?? null}
            uploading={uploading[slot.role] ?? false}
            error={errors[slot.role] ?? null}
            onFile={onFile}
            onRemove={() => onRemove(slot.role)}
            disabled={disabled}
          />
        ))}
      </div>
    </div>
  );
}
