// components/ImageUploader.tsx
// Drag-and-drop image uploader — Mode-aware slot rendering.
// "Sensor Intake" aesthetic: CornerFrame reticle brackets,
// satellite/downlink glyph, animated dashed border ONLY on drag,
// mono CRS/format spec readout.
'use client';

import { useRef, useState, useCallback } from 'react';
import { cn, formatBytes } from '@/lib/utils';
import type { AnalysisMode, UploadedImage } from '@/lib/types/analysis';
import {
  X, FileImage, AlertCircle, CheckCircle2, Loader2,
  Satellite, Download,
} from 'lucide-react';
import { CornerFrame } from '@/components/ui/CornerFrame';

const MODE_SLOTS: Record<AnalysisMode, {
  role: UploadedImage['role'];
  label: string;
  hint: string;
  domain: 'cyan' | 'magenta' | 'amber';
}[]> = {
  single_image: [
    { role: 'single', label: 'Satellite Image', hint: 'GeoTIFF (primary) or PNG/JPEG (demo)', domain: 'cyan' },
  ],
  bi_temporal: [
    { role: 'before', label: 'Before (T1)', hint: 'Earlier acquisition — GeoTIFF or PNG/JPEG', domain: 'magenta' },
    { role: 'after',  label: 'After (T2)',  hint: 'Later acquisition — GeoTIFF or PNG/JPEG',   domain: 'magenta' },
  ],
  optical_sar: [
    { role: 'optical', label: 'Optical / MSI', hint: 'Optical or multispectral GeoTIFF or PNG/JPEG', domain: 'cyan' },
    { role: 'sar',     label: 'SAR Image',     hint: 'SAR (C/L/X-band) GeoTIFF or PNG/JPEG',        domain: 'amber' },
  ],
};

const ACCEPTED_TYPES = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
const MAX_SIZE_MB = 500;

interface UploadSlotProps {
  slot: { role: UploadedImage['role']; label: string; hint: string; domain: 'cyan' | 'magenta' | 'amber' };
  uploaded: UploadedImage | null;
  uploading: boolean;
  error: string | null;
  onFile: (file: File, role: UploadedImage['role']) => void;
  onRemove: () => void;
  disabled?: boolean;
}

function UploadSlot({ slot, uploaded, uploading, error, onFile, onRemove, disabled }: UploadSlotProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file, slot.role);
  }, [disabled, onFile, slot.role]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFile(file, slot.role);
    e.target.value = '';
  };

  const isGeoTiff = uploaded?.metadata.fileFormat === 'GeoTIFF' || uploaded?.metadata.fileFormat === 'TIFF';
  const hasContent = !!uploaded || uploading;

  return (
    <div className="flex flex-col gap-2">
      <span className="hud-label">{slot.label}</span>

      <CornerFrame
        domain={slot.domain}
        intensity={dragging ? 'strong' : 'normal'}
        bracketSize={14}
      >
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => !hasContent && !disabled && inputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label={`Upload ${slot.label}`}
          onKeyDown={(e) => e.key === 'Enter' && !hasContent && !disabled && inputRef.current?.click()}
          className={cn(
            'relative flex flex-col items-center justify-center rounded transition-all duration-200 min-h-[168px]',
            !hasContent && !disabled ? 'cursor-pointer' : '',
            dragging ? 'dropzone-active' : '',
          )}
          style={{
            background: dragging
              ? 'color-mix(in srgb, var(--cyan) 5%, var(--surface-2))'
              : 'var(--surface-1)',
            border: dragging
              ? '1px solid color-mix(in srgb, var(--cyan) 60%, transparent)'
              : error
                ? '1px solid var(--red)'
                : '1px solid var(--border-hairline)',
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
            <div className="flex flex-col items-center gap-2 p-6">
              <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--cyan)' }} />
              <span className="hud-label">Uploading…</span>
            </div>
          ) : uploaded ? (
            <div className="w-full h-full relative">
              {/* Preview or placeholder */}
              {uploaded.previewUrl ? (
                <img
                  src={uploaded.previewUrl}
                  alt={`Preview of ${uploaded.metadata.fileName}`}
                  className="w-full h-full object-cover rounded"
                  style={{ maxHeight: '200px' }}
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-3 p-6 h-full min-h-[160px]">
                  <FileImage className="w-8 h-8" style={{ color: 'var(--cyan)', opacity: 0.6 }} />
                  <div className="flex flex-col items-center gap-1 text-center">
                    <span className="text-xs font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                      {uploaded.metadata.fileName}
                    </span>
                    <span className="hud-label" style={{ fontSize: '0.58rem' }}>
                      Preview unavailable for GeoTIFF — metadata extracted below
                    </span>
                  </div>
                </div>
              )}

              {/* Overlay strip: filename + status */}
              <div
                className="absolute bottom-0 left-0 right-0 flex items-center justify-between gap-2 px-3 py-2 rounded-b"
                style={{
                  background: 'color-mix(in srgb, var(--surface-0) 85%, transparent)',
                  backdropFilter: 'blur(4px)',
                  borderTop: '1px solid var(--border-hairline)',
                }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <CheckCircle2
                    className="w-3.5 h-3.5 flex-shrink-0"
                    style={{ color: 'var(--green)' }}
                    aria-hidden
                  />
                  <span
                    className="text-[0.65rem] font-mono truncate"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {uploaded.metadata.fileName}
                  </span>
                </div>
                <span
                  className="hud-label flex-shrink-0"
                  style={{ fontSize: '0.58rem', color: 'var(--text-faint)' }}
                >
                  {formatBytes(uploaded.metadata.fileSizeBytes)}
                </span>
              </div>

              {/* Remove button */}
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onRemove(); }}
                className="absolute top-2 right-2 w-6 h-6 rounded flex items-center justify-center transition-colors"
                style={{
                  background: 'color-mix(in srgb, var(--surface-0) 80%, transparent)',
                  border: '1px solid color-mix(in srgb, var(--red) 30%, transparent)',
                  backdropFilter: 'blur(4px)',
                }}
                aria-label={`Remove ${slot.label}`}
              >
                <X className="w-3.5 h-3.5" style={{ color: 'var(--red)' }} />
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 p-6 text-center">
              {/* Satellite/downlink glyph — replaces generic upload cloud */}
              <div
                className="w-11 h-11 rounded flex items-center justify-center"
                style={{
                  background: dragging
                    ? 'color-mix(in srgb, var(--cyan) 15%, transparent)'
                    : 'color-mix(in srgb, var(--cyan) 7%, transparent)',
                  border: `1px solid color-mix(in srgb, var(--cyan) ${dragging ? 35 : 18}%, transparent)`,
                }}
                aria-hidden
              >
                <Satellite
                  className="w-5 h-5"
                  strokeWidth={1.8}
                  style={{
                    color: 'var(--cyan)',
                    opacity: dragging ? 1 : 0.8,
                    transform: dragging ? 'translateY(-1px)' : 'none',
                  }}
                />
              </div>

              <div className="flex flex-col gap-1">
                <span
                  className="text-xs font-medium"
                  style={{ color: 'var(--text-primary)' }}
                >
                  Drop payload here or{' '}
                  <span style={{ color: 'var(--cyan)', fontWeight: 600 }}>browse</span>
                </span>
                <span
                  className="text-[0.65rem] leading-snug"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {slot.hint}
                </span>
              </div>

              {/* Spec readout line — mono-font CRS/format info as if spec sheet */}
              <div
                className="mt-1 px-2 py-1 rounded flex items-center gap-2"
                style={{
                  background: 'var(--surface-2)',
                  border: '1px dashed var(--border-hairline)',
                }}
              >
                <Download className="w-3 h-3" style={{ color: 'var(--text-faint)' }} />
                <span
                  className="font-mono"
                  style={{
                    fontSize: '0.58rem',
                    color: 'var(--text-faint)',
                    letterSpacing: '0.04em',
                    whiteSpace: 'nowrap',
                  }}
                >
                  GeoTIFF (EPSG) · PNG/JPEG · MAX {MAX_SIZE_MB}MB
                </span>
              </div>
            </div>
          )}
        </div>
      </CornerFrame>

      {/* Error */}
      {error && (
        <div
          className="flex items-center gap-1.5 badge badge-red"
          role="alert"
        >
          <AlertCircle className="w-3 h-3 flex-shrink-0" aria-hidden />
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

export function ImageUploader({ mode, uploads, uploading, errors, onFile, onRemove, disabled }: ImageUploaderProps) {
  const slots = MODE_SLOTS[mode];

  return (
    <div className="flex flex-col gap-3">
      <span className="hud-label">Image Input · Sensor Downlink Intake</span>
      <div className={cn('grid gap-4', slots.length === 1 ? 'grid-cols-1 max-w-md' : 'grid-cols-1 sm:grid-cols-2')}>
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
