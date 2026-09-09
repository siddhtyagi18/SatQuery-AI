// app/analysis/new/page.tsx
// New Analysis submission flow — supporting Single Image, Bi-Temporal, and Optical+SAR.
'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { AnalysisModeSelector } from '@/components/AnalysisModeSelector';
import { ImageUploader } from '@/components/ImageUploader';
import { ImageMetadata } from '@/components/ImageMetadata';
import { QueryInput, type QueryLanguage } from '@/components/QueryInput';
import { AgentExecutionTrace } from '@/components/AgentExecutionTrace';
import { api } from '@/lib/api';
import type { AnalysisMode, AnalysisMissionMode, DisasterType, ExecutionTrace, UploadedImage } from '@/lib/types/analysis';
import { Play, BookmarkCheck, AlertCircle, ShieldAlert } from 'lucide-react';
import { singleImageResult, biTemporalResult, opticalSarResult } from '@/lib/api/mock/fixtures';
import { API_MODE } from '@/lib/config';
import { toast } from 'sonner';

const WORKFLOW_HEADERS: Record<
  AnalysisMode,
  {
    title: string;
    badge: string;
    description: string;
    domain: 'cyan' | 'magenta' | 'amber';
  }
> = {
  single_image: {
    title: 'Single Image Analysis',
    badge: 'VQA & SPATIAL GROUNDING',
    description:
      'Query an individual optical, multispectral, or SAR satellite scene with natural language. Perform visual Q&A, scene captioning, and bounding-box spatial target grounding.',
    domain: 'cyan',
  },
  bi_temporal: {
    title: 'Bi-Temporal Pair Analysis',
    badge: 'CHANGE DETECTION PIPELINE',
    description:
      'Compare multi-temporal baseline image pairs (T1 and T2) with trained Siamese U-Net models to detect building construction, urban sprawl, and water level changes.',
    domain: 'magenta',
  },
  optical_sar: {
    title: 'Optical + SAR Analysis',
    badge: 'CROSS-MODAL SENSOR FUSION',
    description:
      'Fuse co-registered optical reflectance with Synthetic Aperture Radar (SAR) microwave dielectric backscatter for all-weather feature discovery and verification.',
    domain: 'amber',
  },
};

function NewAnalysisContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const initialMode = (searchParams.get('mode') as AnalysisMode) || 'single_image';
  const [mode, setMode] = useState<AnalysisMode>(
    ['single_image', 'bi_temporal', 'optical_sar'].includes(initialMode)
      ? initialMode
      : 'single_image'
  );

  useEffect(() => {
    const qMode = searchParams.get('mode') as AnalysisMode;
    if (qMode && ['single_image', 'bi_temporal', 'optical_sar'].includes(qMode)) {
      setMode(qMode);
    }
  }, [searchParams]);

  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState<QueryLanguage>('en');
  const [uploads, setUploads] = useState<Partial<Record<UploadedImage['role'], UploadedImage>>>({});
  const [uploading, setUploading] = useState<Partial<Record<UploadedImage['role'], boolean>>>({});
  const [errors, setErrors] = useState<Partial<Record<UploadedImage['role'], string>>>({});

  // Disaster Assessment Mode state
  const [analysisMissionMode, setAnalysisMissionMode] = useState<AnalysisMissionMode>('general_change');
  const [disasterType, setDisasterType] = useState<DisasterType | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTrace, setActiveTrace] = useState<ExecutionTrace | null>(null);

  const handleModeChange = (newMode: AnalysisMode) => {
    setMode(newMode);
    setUploads({});
    setErrors({});
    // When switching to non-bi_temporal mode, reset disaster state
    if (newMode !== 'bi_temporal') {
      setAnalysisMissionMode('general_change');
      setDisasterType(null);
    }
  };

  const handleMissionModeChange = (mm: AnalysisMissionMode) => {
    setAnalysisMissionMode(mm);
    if (mm === 'disaster_assessment') {
      // Force bi-temporal mode for disaster assessment
      if (mode !== 'bi_temporal') {
        setMode('bi_temporal');
        setUploads({});
        setErrors({});
      }
    } else {
      setDisasterType(null);
    }
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
      if (mode === 'single_image') {
        setQuery(singleImageResult.query);
      } else if (mode === 'bi_temporal') {
        setQuery(biTemporalResult.query);
      } else {
        setQuery(opticalSarResult.query);
      }
      toast.info(`Demo query loaded — please upload your satellite imagery to run analysis.`);
    } else {
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
        language,
        analysisMissionMode,
        disasterType: analysisMissionMode === 'disaster_assessment' ? disasterType : undefined,
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

  const currentHeader = WORKFLOW_HEADERS[mode] ?? WORKFLOW_HEADERS.single_image;
  const domainVar = `var(--${currentHeader.domain})`;

  return (
    <div className="page-shell max-w-5xl mx-auto pb-16 animate-fade-in-up flex flex-col gap-10">
      {/* Dynamic Workflow Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-[var(--border-hairline)]">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span
              className="font-mono text-[0.62rem] font-bold tracking-widest uppercase px-2.5 py-0.5 rounded"
              style={{
                background: `color-mix(in srgb, ${domainVar} 14%, transparent)`,
                border: `1px solid color-mix(in srgb, ${domainVar} 30%, transparent)`,
                color: domainVar,
              }}
            >
              {currentHeader.badge}
            </span>
            <span className="text-xs text-[var(--text-faint)] font-mono">
              Ground Ingest Console
            </span>
          </div>
          <h1 className="text-display-lg text-[var(--text-primary)]">
            {currentHeader.title}
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1.5 max-w-2xl leading-relaxed">
            {currentHeader.description}
          </p>
        </div>

        <button
          type="button"
          onClick={handleLoadDemoExample}
          disabled={isSubmitting}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-mono font-medium border border-[var(--cyan)]/30 bg-[var(--cyan)]/10 text-[var(--cyan)] hover:bg-[var(--cyan)]/20 transition-all cursor-pointer self-start sm:self-center shadow-sm"
        >
          <BookmarkCheck className="w-4 h-4" />
          <span>Load Demo Preset</span>
        </button>
      </div>

      {/* STEP 1: Analysis Mode Selector */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="badge badge-cyan">STEP 01</span>
          <h2 className="text-sm font-semibold uppercase tracking-wider font-heading text-[var(--text-primary)]">
            Select Analysis Workflow
          </h2>
        </div>
        <AnalysisModeSelector
          value={mode}
          onChange={handleModeChange}
          disabled={isSubmitting || analysisMissionMode === 'disaster_assessment'}
        />
      </section>

      {/* STEP 1.5: Mission Mode — General vs Disaster Assessment */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="badge badge-amber">MISSION</span>
          <h2 className="text-sm font-semibold uppercase tracking-wider font-heading text-[var(--text-primary)]">
            Select Mission Context
          </h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* General Change Analysis */}
          <button
            type="button"
            onClick={() => handleMissionModeChange('general_change')}
            disabled={isSubmitting}
            className={`relative flex flex-col gap-2 p-4 text-left rounded transition-all duration-200 cursor-pointer ${
              analysisMissionMode === 'general_change'
                ? 'panel-selected-cyan'
                : ''
            }`}
            style={{
              background: analysisMissionMode === 'general_change'
                ? 'color-mix(in srgb, var(--cyan) 5%, var(--surface-2))'
                : 'var(--surface-1)',
              border: analysisMissionMode === 'general_change'
                ? '2px solid var(--cyan)'
                : '1px solid var(--border-hairline)',
              opacity: analysisMissionMode === 'general_change' ? 1 : 0.82,
            }}
          >
            {analysisMissionMode === 'general_change' && (
              <div className="top-accent-bar" data-domain="cyan" aria-hidden />
            )}
            <span className="text-base font-semibold" style={{ fontFamily: 'var(--font-heading)', color: analysisMissionMode === 'general_change' ? 'var(--cyan)' : 'var(--text-primary)' }}>
              General Change Analysis
            </span>
            <span className="text-sm text-[var(--text-muted)] leading-relaxed">
              Standard bi-temporal change detection for general land-use, construction, or environmental monitoring.
            </span>
          </button>

          {/* Disaster Assessment */}
          <button
            type="button"
            onClick={() => handleMissionModeChange('disaster_assessment')}
            disabled={isSubmitting}
            className={`relative flex flex-col gap-2 p-4 text-left rounded transition-all duration-200 cursor-pointer ${
              analysisMissionMode === 'disaster_assessment'
                ? 'panel-selected-amber'
                : ''
            }`}
            style={{
              background: analysisMissionMode === 'disaster_assessment'
                ? 'color-mix(in srgb, var(--amber) 5%, var(--surface-2))'
                : 'var(--surface-1)',
              border: analysisMissionMode === 'disaster_assessment'
                ? '2px solid var(--amber)'
                : '1px solid var(--border-hairline)',
              opacity: analysisMissionMode === 'disaster_assessment' ? 1 : 0.82,
            }}
          >
            {analysisMissionMode === 'disaster_assessment' && (
              <div className="top-accent-bar" data-domain="amber" aria-hidden />
            )}
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-5 h-5" style={{ color: analysisMissionMode === 'disaster_assessment' ? 'var(--amber)' : 'var(--text-muted)' }} />
              <span className="text-base font-semibold" style={{ fontFamily: 'var(--font-heading)', color: analysisMissionMode === 'disaster_assessment' ? 'var(--amber)' : 'var(--text-primary)' }}>
                Disaster Assessment
              </span>
            </div>
            <span className="text-sm text-[var(--text-muted)] leading-relaxed">
              Analyze before/after imagery under a disaster context. Change regions highlighted for operator investigation.
            </span>
          </button>
        </div>

        {/* Disaster Type Selector */}
        {analysisMissionMode === 'disaster_assessment' && (
          <div className="flex flex-col gap-2 animate-fade-in-up">
            <span className="hud-label text-[var(--amber)]">Select Disaster Type</span>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
              {([
                { value: 'flood' as DisasterType, icon: '🌊', label: 'Flood' },
                { value: 'earthquake' as DisasterType, icon: '🏚️', label: 'Earthquake' },
                { value: 'wildfire' as DisasterType, icon: '🔥', label: 'Wildfire' },
                { value: 'cyclone' as DisasterType, icon: '🌀', label: 'Cyclone' },
                { value: 'landslide' as DisasterType, icon: '⛰️', label: 'Landslide' },
              ]).map((d) => (
                <button
                  key={d.value}
                  type="button"
                  onClick={() => setDisasterType(d.value)}
                  disabled={isSubmitting}
                  className={`flex items-center gap-2 px-3 py-2.5 rounded text-sm font-medium transition-all cursor-pointer ${
                    disasterType === d.value ? 'ring-2 ring-[var(--amber)] shadow-[0_0_12px_rgba(251,191,36,0.25)]' : ''
                  }`}
                  style={{
                    background: disasterType === d.value
                      ? 'color-mix(in srgb, var(--amber) 12%, var(--surface-2))'
                      : 'var(--surface-1)',
                    border: disasterType === d.value
                      ? '2px solid var(--amber)'
                      : '1px solid var(--border-hairline)',
                    color: disasterType === d.value ? 'var(--amber)' : 'var(--text-primary)',
                  }}
                >
                  <span className="text-lg">{d.icon}</span>
                  <span>{d.label}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* STEP 2: Image Ingest Area */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span
            className="font-mono text-[0.65rem] font-bold px-2 py-0.5 rounded"
            style={{
              background: `color-mix(in srgb, ${domainVar} 15%, transparent)`,
              color: domainVar,
              border: `1px solid color-mix(in srgb, ${domainVar} 35%, transparent)`,
            }}
          >
            STEP 02
          </span>
          <h2 className="text-sm font-semibold uppercase tracking-wider font-heading text-[var(--text-primary)]">
            Upload Imagery ({mode === 'single_image' ? '1 Tile' : '2 Tiles Required'})
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

      {/* STEP 3: Extracted Metadata (if uploaded) */}
      {Object.values(uploads).some(Boolean) && (
        <section className="flex flex-col gap-3 animate-fade-in-up">
          <div className="flex items-center gap-2">
            <span className="badge badge-green">STEP 03</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider font-heading text-[var(--text-primary)]">
              Extracted Sensor Metadata &amp; Coordinate System
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(uploads).map(([role, img]) =>
              img ? (
                <ImageMetadata
                  key={role}
                  image={img}
                  label={`${role.toUpperCase()} Input Telemetry`}
                />
              ) : null
            )}
          </div>
        </section>
      )}

      {/* STEP 4: Natural Language Inquiry Console (ChatGPT-style) */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="badge badge-cyan">STEP 04</span>
          <h2 className="text-sm font-semibold uppercase tracking-wider font-heading text-[var(--text-primary)]">
            Ask Your Analysis Question
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
          language={language}
          onLanguageChange={setLanguage}
        />
      </section>

      {/* Validation Message Box if not ready */}
      {!canSubmit && !isSubmitting && (
        <div
          className="p-4 rounded-xl border border-[var(--border-hairline)] flex items-start gap-3 text-xs"
          style={{ background: 'var(--surface-1)' }}
        >
          <AlertCircle className="w-4 h-4 text-[var(--amber)] flex-shrink-0 mt-0.5" />
          <div className="flex flex-col gap-0.5">
            <span className="font-semibold text-[var(--text-primary)]">
              Prerequisites for Analysis
            </span>
            <span className="text-[var(--text-muted)] leading-relaxed">
              {!hasRequiredImages()
                ? `Please provide all required image payloads for ${mode.replace('_', ' ')} mode.`
                : 'Please type a specific inquiry question or choose one of the suggested questions above.'}
            </span>
          </div>
        </div>
      )}

      {/* Submit CTA */}
      <section className="flex items-center justify-end gap-3 pt-2">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="btn-primary flex items-center gap-2 shadow-xl cursor-pointer"
          style={{ padding: '12px 28px', fontSize: '0.84rem' }}
        >
          <Play className="w-4 h-4 fill-current" />
          <span>{isSubmitting ? 'Executing Specialist Pipeline…' : 'Run SatQuery Pipeline'}</span>
        </button>
      </section>

      {/* Live Execution Trace Stream during pipeline execution */}
      {isSubmitting && activeTrace && (
        <section
          className="flex flex-col gap-3 pt-6 animate-fade-in-up border-t border-[var(--border-hairline)]"
        >
          <div className="flex items-center gap-2">
            <span className="badge badge-amber animate-pulse">LIVE ORCHESTRATION</span>
            <span className="hud-label text-[var(--amber)]">
              Active Multi-Agent Execution Trace Stream
            </span>
          </div>
          <AgentExecutionTrace trace={activeTrace} defaultExpanded={true} />
        </section>
      )}
    </div>
  );
}

export default function NewAnalysisPage() {
  return (
    <Suspense
      fallback={
        <div className="page-shell max-w-5xl mx-auto py-12 flex items-center justify-center">
          <span className="font-mono text-xs text-[var(--text-faint)] tracking-widest uppercase animate-pulse">
            Loading Mission Console…
          </span>
        </div>
      }
    >
      <NewAnalysisContent />
    </Suspense>
  );
}
