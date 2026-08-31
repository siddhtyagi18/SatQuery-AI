// app/analysis/new/page.tsx
// New Analysis submission flow — highest priority page.
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { AnalysisModeSelector } from '@/components/AnalysisModeSelector';
import { ImageUploader } from '@/components/ImageUploader';
import { ImageMetadata } from '@/components/ImageMetadata';
import { QueryInput } from '@/components/QueryInput';
import { AgentExecutionTrace } from '@/components/AgentExecutionTrace';
import { api } from '@/lib/api';
import type { AnalysisMode, ExecutionTrace, UploadedImage } from '@/lib/types/analysis';
import { Play, BookmarkCheck, AlertCircle } from 'lucide-react';
import { singleImageResult, biTemporalResult, opticalSarResult } from '@/lib/api/mock/fixtures';
import { API_MODE } from '@/lib/config';
import { toast } from 'sonner';

export default function NewAnalysisPage() {
  const router = useRouter();

  const [mode, setMode] = useState<AnalysisMode>('single_image');
  const [query, setQuery] = useState('');
  const [uploads, setUploads] = useState<Partial<Record<UploadedImage['role'], UploadedImage>>>({});
  const [uploading, setUploading] = useState<Partial<Record<UploadedImage['role'], boolean>>>({});
  const [errors, setErrors] = useState<Partial<Record<UploadedImage['role'], string>>>({});

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTrace, setActiveTrace] = useState<ExecutionTrace | null>(null);

  const handleModeChange = (newMode: AnalysisMode) => {
    setMode(newMode);
    setUploads({});
    setErrors({});
  };

  const handleFileUpload = async (file: File, role: UploadedImage['role']) => {
    if (file.size > 500 * 1024 * 1024) {
      setErrors((prev) => ({ ...prev, [role]: 'File exceeds maximum limit of 500 MB.' }));
      return;
    }

    setUploading((prev) => ({ ...prev, [role]: true }));
    setErrors((prev) => ({ ...prev, [role]: undefined }));

    try {
      const uploaded = await api.uploadImage(file, role);
      setUploads((prev) => ({ ...prev, [role]: uploaded }));
      toast.success(`Loaded ${uploaded.metadata.fileName}`);
    } catch (err: any) {
      setErrors((prev) => ({ ...prev, [role]: err.message || 'Failed to upload image' }));
      toast.error('Image upload failed');
    } finally {
      setUploading((prev) => ({ ...prev, [role]: false }));
    }
  };

  const handleRemove = (role: UploadedImage['role']) => {
    setUploads((prev) => {
      const next = { ...prev };
      delete next[role];
      return next;
    });
  };

  const handleLoadDemoExample = () => {
    if (API_MODE === 'live') {
      // In live mode, images must be uploaded by the user.
      // We only pre-fill the query text as a helpful demo starting point.
      if (mode === 'single_image') {
        setQuery(singleImageResult.query);
      } else if (mode === 'bi_temporal') {
        setQuery(biTemporalResult.query);
      } else {
        setQuery(opticalSarResult.query);
      }
      toast.info(`Demo query loaded — please upload your satellite imagery to run analysis.`);
    } else {
      // Mock mode: use pre-loaded fixture images so analysis runs immediately
      if (mode === 'single_image') {
        const img = singleImageResult.images[0];
        setUploads({ single: img });
        setQuery(singleImageResult.query);
      } else if (mode === 'bi_temporal') {
        setUploads({
          before: biTemporalResult.images[0],
          after: biTemporalResult.images[1],
        });
        setQuery(biTemporalResult.query);
      } else {
        setUploads({
          optical: opticalSarResult.images[0],
          sar: opticalSarResult.images[1],
        });
        setQuery(opticalSarResult.query);
      }
      toast.info(`Loaded pre-configured ${mode.replace('_', ' ')} demonstration suite`);
    }
  };

  const hasRequiredImages = () => {
    if (mode === 'single_image') return !!uploads.single;
    if (mode === 'bi_temporal') return !!uploads.before && !!uploads.after;
    if (mode === 'optical_sar') return !!uploads.optical && !!uploads.sar;
    return false;
  };

  const canSubmit = hasRequiredImages() && query.trim().length > 3 && !isSubmitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;

    setIsSubmitting(true);
    let unsubscribe: (() => void) | null = null;

    try {
      const imageIds = Object.values(uploads)
        .filter(Boolean)
        .map((img) => img!.id);

      const { analysisId } = await api.submitAnalysis({
        mode,
        query: query.trim(),
        imageIds,
      });

      let redirected = false;
      unsubscribe = api.streamExecutionTrace(analysisId, (trace) => {
        setActiveTrace(trace);
        if (!redirected && (trace.overallStatus === 'completed' || trace.overallStatus === 'failed')) {
          redirected = true;
          unsubscribe?.();
          setTimeout(() => {
            router.push(`/analysis/${analysisId}`);
          }, 800);
        }
      });
    } catch (err: any) {
      toast.error(err.message || 'Submission failed');
      setIsSubmitting(false);
      unsubscribe?.();
    }
  };

  return (
    <div className="page-shell relative flex flex-col max-w-5xl mx-auto pb-12 animate-fade-in-up" style={{ gap: '48px' }}>
      {/* ============================================================
         HEADER BAR
         ============================================================ */}
      <div
        className="flex items-center justify-between flex-wrap gap-4 pb-5"
        style={{ borderBottom: '1px solid var(--border-hairline)' }}
      >
        <div className="flex flex-col gap-1.5">
          <h1 className="text-display">Initiate Remote Sensing Analysis</h1>
          <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            Configure multi-modal input payloads and prompt the ISRO specialist ensemble.
          </p>
        </div>

        <button
          type="button"
          onClick={handleLoadDemoExample}
          disabled={isSubmitting}
          className="badge badge-cyan cursor-pointer hover:opacity-90 transition-opacity"
          style={{ padding: '6px 12px', fontSize: '0.75rem' }}
        >
          <BookmarkCheck className="w-3.5 h-3.5" />
          Load Demo Preset
        </button>
      </div>

      {/* ============================================================
         STEP 1: ANALYSIS MODE
         ============================================================ */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <span
            className="badge badge-cyan"
          >
            STEP 01
          </span>
          <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
            Select Analysis Domain &amp; Workflow
          </h2>
        </div>
        <AnalysisModeSelector
          value={mode}
          onChange={handleModeChange}
          disabled={isSubmitting}
        />
      </section>

      {/* ============================================================
         STEP 2: IMAGE UPLOAD
         ============================================================ */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <span
            className={cn(
              'badge',
              mode === 'single_image'
                ? 'badge-cyan'
                : mode === 'bi_temporal'
                  ? 'badge-magenta'
                  : 'badge-amber'
            )}
          >
            STEP 02
          </span>
          <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
            Ingest Sensor Payloads (Downlink Imagery)
          </h2>
        </div>
        <ImageUploader
          mode={mode}
          uploads={uploads}
          uploading={uploading}
          errors={errors}
          onFile={handleFileUpload}
          onRemove={handleRemove}
          disabled={isSubmitting}
        />
      </section>

      {/* ============================================================
         STEP 3: METADATA HUD PANELS
         ============================================================ */}
      {Object.values(uploads).some(Boolean) && (
        <section className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <span
              className="badge badge-green"
            >
              STEP 03
            </span>
            <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
              Extracted Telemetry &amp; Format Metadata
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(uploads).map(([role, img]) => (
              img ? (
                <ImageMetadata
                  key={role}
                  image={img}
                  label={`${role.toUpperCase()} Input Metadata`}
                />
              ) : null
            ))}
          </div>
        </section>
      )}

      {/* ============================================================
         STEP 4: NATURAL LANGUAGE QUERY
         ============================================================ */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <span
            className="badge badge-cyan"
          >
            STEP 04
          </span>
          <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
            Issue Natural-Language Inquiry to Agent Ensemble
          </h2>
        </div>
        <QueryInput
          value={query}
          onChange={setQuery}
          mode={mode}
          onSubmit={handleSubmit}
          disabled={isSubmitting}
          canSubmit={canSubmit}
          isSubmitting={isSubmitting}
        />
      </section>

      {/* ============================================================
         VALIDATION MESSAGE (DISABLED STATE EXPLANATION)
         ============================================================ */}
      {!canSubmit && !isSubmitting && (
        <div
          className="panel p-4 flex items-start gap-2.5"
          style={{
            background: 'var(--surface-2)',
          }}
        >
          <AlertCircle
            className="w-4 h-4 flex-shrink-0 mt-0.5"
            style={{ color: 'var(--amber)' }}
          />
          <span
            className="text-sm leading-relaxed"
            style={{
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-body)',
              fontWeight: 400,
            }}
          >
            {!hasRequiredImages()
              ? `Please provide all required image payloads for ${mode.replace('_', ' ')} mode.`
              : 'Please enter a natural language question or select a suggestion.'}
          </span>
        </div>
      )}

      {/* ============================================================
         SUBMIT CTA
         ============================================================ */}
      <section className="flex items-center justify-end gap-3 pt-1">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="btn-primary"
        >
          <Play className="w-4 h-4 fill-current" />
          {isSubmitting ? 'Executing Specialist Pipeline…' : 'Run SatQuery Pipeline'}
        </button>
      </section>

      {/* ============================================================
         LIVE EXECUTION TRACE (DURING SUBMISSION)
         ============================================================ */}
      {isSubmitting && activeTrace && (
        <section
          className="flex flex-col gap-3 pt-4 animate-fade-in-up"
          style={{ borderTop: '1px solid var(--border-hairline)' }}
        >
          <div className="flex items-center gap-3">
            <span
              className="badge badge-amber"
              style={{ padding: '2px 8px', fontSize: '0.58rem' }}
            >
              LIVE
            </span>
            <span className="hud-label" style={{ color: 'var(--amber)' }}>
              ACTIVE MULTI-AGENT ORCHESTRATION STREAM
            </span>
          </div>
          <AgentExecutionTrace trace={activeTrace} defaultExpanded={true} />
        </section>
      )}
    </div>
  );
}

// Local cn helper (one-off since we're inside a page module)
function cn(...c: (string | false | undefined)[]): string {
  return c.filter(Boolean).join(' ');
}
