// app/page.tsx
// Dashboard — Mission Control Ground Station overview.
'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import type { AnalysisResult, ToolDefinition } from '@/lib/types/analysis';
import { ModeBadge } from '@/components/ui/ModeBadge';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { CornerFrame } from '@/components/ui/CornerFrame';
import { EmptyState } from '@/components/ui/EmptyState';
import { DashboardSkeleton } from '@/components/ui/LoadingSkeletonPanel';
import {
  Satellite,
  PlusCircle,
  Layers,
  GitCompare,
  Radar,
  ArrowRight,
  Clock,
} from 'lucide-react';
import { formatRelativeTime } from '@/lib/utils';

const WORKFLOWS = [
  {
    key: 'single',
    href: '/analysis/new',
    label: 'Single Image Analysis',
    description: 'VQA, captioning, and bounding-box spatial grounding on individual optical or SAR tiles.',
    domain: 'cyan' as const,
    Icon: Layers,
    cta: 'Initialize Workspace',
    tags: ['Visual Q&A', 'Image Caption', 'Spatial Grounding'],
  },
  {
    key: 'bi',
    href: '/analysis/new',
    label: 'Bi-Temporal Change Detection',
    description: 'Compare multi-year baseline image pairs to detect urban sprawl, deforestation, and water retreat.',
    domain: 'magenta' as const,
    Icon: GitCompare,
    cta: 'Initialize Workspace',
    tags: ['Change Detection', 'Change VQA', 'Land-Use Delta'],
  },
  {
    key: 'sar',
    href: '/analysis/new',
    label: 'Optical + SAR Cross-Modal Fusion',
    description: 'Reconcile optical reflection with radar dielectric backscatter for all-weather feature discovery.',
    domain: 'amber' as const,
    Icon: Radar,
    cta: 'Initialize Workspace',
    tags: ['Cross-Modal Fusion', 'SAR Feature', 'Multi-Modal VQA'],
  },
];

const DOMAIN_HOVER: Record<'cyan' | 'magenta' | 'amber', string> = {
  cyan:    'panel-hover-glow-cyan',
  magenta: 'panel-hover-glow-magenta',
  amber:   'panel-hover-glow-amber',
};

const TAG_DOMAIN: Record<'cyan' | 'magenta' | 'amber', string> = {
  cyan:    'badge-cyan',
  magenta: 'badge-magenta',
  amber:   'badge-amber',
};

export default function DashboardPage() {
  const router = useRouter();
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.listAnalysisHistory({ pageSize: 5 }),
      api.listTools(),
    ])
      .then(([histData, toolData]) => {
        setHistory(histData.items);
        setTools(toolData);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return <DashboardSkeleton />;
  }

  const totalAnalyses = history.length;
  const completedCount = history.filter((h) => h.status === 'completed').length;

  return (
    <div className="page-shell relative flex flex-col gap-12 max-w-7xl mx-auto pb-12 animate-fade-in-up">
      {/* Ambient radial accent glow behind hero */}
      <div className="hero-glow" aria-hidden />

      {/* ============================================================
         HERO BAND — "Mission Control Directive"
         Corner-framed, display-size heading, mono sub-copy.
         ============================================================ */}
      <CornerFrame
        domain="cyan"
        label="ISRO MISSION CONTROL DIRECTIVE"
        bracketSize={16}
        intensity="strong"
      >
        <div
          className="panel p-6 md:p-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-8 relative overflow-hidden"
          style={{ background: 'linear-gradient(135deg, var(--surface-1) 0%, var(--surface-2) 50%, var(--surface-1) 100%)' }}
        >
          <div className="starfield" aria-hidden />
          <div className="nebula-glow cyan" aria-hidden />
          <div className="nebula-glow magenta" aria-hidden />
          <div className="nebula-glow amber" aria-hidden />
          <div className="shooting-star" style={{ top: '14%', left: '62%' }} aria-hidden />
          <div className="shooting-star s2" aria-hidden />
          <div className="shooting-star s3" aria-hidden />

          <div className="flex flex-col gap-4 max-w-2xl z-10">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="badge badge-cyan">AI GROUND STATION v0.1</span>
              <span className="badge badge-green radar-wrap">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-pulse-dot absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
                </span>
                TELEMETRY ACTIVE
              </span>
            </div>

            <h1 className="text-display-lg">
              SatQuery-AI Intelligence Console
            </h1>

            <p className="text-sm text-[var(--text-muted)] leading-relaxed max-w-xl">
              Interactive multimodal vision-language assistant for ISRO remote-sensing missions.
              Ask natural-language questions over high-resolution optical, SAR, and multispectral payloads.
            </p>
          </div>

          <div className="relative z-10 flex-shrink-0 self-start md:self-center">
            <div className="orbit-wrap" style={{ top: '50%', left: '50%' }}>
              <span className="orbit-dot" aria-hidden />
              <span className="orbit-dot orbit-magenta" aria-hidden />
              <span className="orbit-dot orbit-amber" aria-hidden />
            </div>
            <Link
              href="/analysis/new"
              className="btn-primary shadow-xl"
            >
              <PlusCircle className="w-4 h-4" />
              New Analysis
            </Link>
          </div>
        </div>
      </CornerFrame>

      {/* ============================================================
         STAT TILES — Telemetry counter aesthetic
         Mono uppercase hud-label + large tabular-mono numerals
         Domain-colored accent bar on left (Level 1 elevation)
         ============================================================ */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Tile 1: Analyses Executed — CYAN domain */}
        <div className="panel p-5 flex flex-col gap-2 relative overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 w-0.5"
            style={{ background: 'linear-gradient(180deg, var(--cyan), transparent)' }}
            aria-hidden
          />
          <div className="stat-sweep" aria-hidden />
          <span className="hud-label">Analyses Executed</span>
          <span className="hud-counter">{totalAnalyses}</span>
          <div className="h-px w-10" style={{ background: 'var(--cyan)', opacity: 0.4 }} />
          <span className="text-[0.65rem] font-mono" style={{ color: 'var(--text-faint)' }}>
            {completedCount} validated successfully
          </span>
        </div>

        {/* Tile 2: Specialist Ensemble — MAGENTA domain (bi-temporal/change) */}
        <div className="panel p-5 flex flex-col gap-2 relative overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 w-0.5"
            style={{ background: 'linear-gradient(180deg, var(--magenta), transparent)' }}
            aria-hidden
          />
          <div className="stat-sweep" style={{ animationDelay: '-1.2s' }} aria-hidden />
          <span className="hud-label">Specialist Ensemble</span>
          <span className="hud-counter">{tools.length}</span>
          <div className="h-px w-10" style={{ background: 'var(--magenta)', opacity: 0.4 }} />
          <span className="text-[0.65rem] font-mono" style={{ color: 'var(--text-faint)' }}>
            Active neural agents online
          </span>
        </div>

        {/* Tile 3: Workflows Ready — AMBER domain (SAR/fusion) */}
        <div className="panel p-5 flex flex-col gap-2 relative overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 w-0.5"
            style={{ background: 'linear-gradient(180deg, var(--amber), transparent)' }}
            aria-hidden
          />
          <div className="stat-sweep" style={{ animationDelay: '-2.4s' }} aria-hidden />
          <span className="hud-label">Workflows Ready</span>
          <span className="hud-counter">3</span>
          <div className="h-px w-10" style={{ background: 'var(--amber)', opacity: 0.4 }} />
          <span className="text-[0.65rem] font-mono" style={{ color: 'var(--text-faint)' }}>
            Single · Bi-Temporal · Optical+SAR
          </span>
        </div>

        {/* Tile 4: System Readiness — GREEN domain */}
        <div className="panel p-5 flex flex-col gap-2 relative overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 w-0.5"
            style={{ background: 'linear-gradient(180deg, var(--green), transparent)' }}
            aria-hidden
          />
          <div className="stat-sweep" style={{ animationDelay: '-3.6s' }} aria-hidden />
          <span className="hud-label">System Readiness</span>
          <div className="pt-1">
            <StatusBadge status="operational" label="OPERATIONAL" />
          </div>
          <div className="h-px w-10" style={{ background: 'var(--green)', opacity: 0.4 }} />
          <span className="text-[0.65rem] font-mono" style={{ color: 'var(--text-faint)' }}>
            All ground pipelines nominal
          </span>
        </div>
      </div>

      {/* ============================================================
         WORKFLOW SELECTION CARDS
         Level 1 panels, hexagon icon frames,
         domain-color glow on hover, normalize capability badges.
         ============================================================ */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <span className="hud-label">Select Mission Analysis Workflow</span>
          <span className="hud-label" style={{ color: 'var(--text-faint)' }}>
            Multi-modal Sensor Dispatch
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {WORKFLOWS.map((w, idx) => {
            const IconComp = w.Icon;
            const floatClass = idx === 0 ? 'animate-float' : idx === 1 ? 'animate-float-d1' : 'animate-float-d2';
            return (
              <Link
                key={w.key}
                href={w.href}
                className={`panel p-5 flex flex-col justify-between gap-4 transition-all duration-200 group relative overflow-hidden ${DOMAIN_HOVER[w.domain]} ${floatClass}`}
              >
                {/* Subtle top accent tint on hover */}
                <div
                  className="absolute top-0 inset-x-0 h-0.5 scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"
                  style={{ background: `linear-gradient(90deg, transparent, var(--${w.domain}), transparent)` }}
                  aria-hidden
                />

                <div className="flex flex-col gap-4">
                  {/* Hexagonal icon frame with inner domain glow */}
                  <div className={`icon-hex icon-hex-outline group-hover:icon-hex`} data-domain={w.domain}>
                    <IconComp
                      className="w-5 h-5"
                      strokeWidth={1.8}
                      style={{ color: `var(--${w.domain})` }}
                    />
                  </div>

                  <div className="flex flex-col gap-2">
                    <h3
                      className="text-base font-semibold transition-colors"
                      style={{ color: 'var(--text-primary)' }}
                      onMouseEnter={(e) => (e.currentTarget.style.color = `var(--${w.domain})`)}
                      onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-primary)')}
                    >
                      {w.label}
                    </h3>
                    <p
                      className="text-xs leading-relaxed"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      {w.description}
                    </p>
                  </div>

                  {/* Capability tags */}
                  <div className="flex flex-wrap gap-1.5">
                    {w.tags.map((t) => (
                      <span
                        key={t}
                        className={`badge ${TAG_DOMAIN[w.domain]}`}
                        style={{ opacity: 0.72 }}
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>

                <div
                  className="flex items-center justify-between pt-3 mt-auto"
                  style={{ borderTop: '1px solid var(--border-hairline)' }}
                >
                  <span
                    className="text-xs font-mono animate-arrow-hover"
                    style={{ color: `var(--${w.domain})`, fontWeight: 600, letterSpacing: '0.06em' }}
                  >
                    {w.cta}
                  </span>
                  <ArrowRight
                    className="w-3.5 h-3.5 arrow-slide"
                    style={{ color: `var(--${w.domain})` }}
                  />
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* ============================================================
         RECENT ANALYSES LOG
         ============================================================ */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <span className="hud-label">Recent Satellite Telemetry Inquiries</span>
          <Link
            href="/analysis/history"
            className="flex items-center gap-1 badge badge-cyan"
          >
            <span>View Full Archive</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        {history.length === 0 ? (
          <div className="panel p-8">
            <EmptyState
              icon={Satellite}
              title="No Analyses In History"
              description="No recent queries found. Launch your first satellite vision-language analysis."
              action={
                <Link
                  href="/analysis/new"
                  className="btn-primary"
                  style={{ padding: '8px 16px', fontSize: '0.65rem' }}
                >
                  Start New Analysis
                </Link>
              }
            />
          </div>
        ) : (
          <div className="panel overflow-hidden">
            <div className="divide-y divide-[var(--border-subtle)]">
              {history.map((item, idx) => {
                const driftClass = `animate-satellite d${(idx % 5) + 1}`;
                return (
                <Link
                  key={item.id}
                  href={`/analysis/${item.id}`}
                  className="flex items-center justify-between p-4 hover:bg-[var(--surface-2)] transition-colors gap-4 group"
                >
                  <div className="flex items-center gap-3.5 min-w-0">
                    <div
                      className="w-10 h-10 rounded flex items-center justify-center flex-shrink-0"
                      style={{
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border-hairline)',
                      }}
                    >
                      <Satellite className={`w-4 h-4 ${driftClass}`} style={{ color: 'var(--cyan)' }} />
                    </div>

                    <div className="flex flex-col min-w-0 gap-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span
                          className="font-mono text-xs font-semibold"
                          style={{ color: 'var(--text-primary)' }}
                        >
                          {item.id}
                        </span>
                        <ModeBadge mode={item.mode} />
                        <StatusBadge status={item.status} />
                      </div>

                      <p
                        className="text-xs truncate max-w-md md:max-w-xl"
                        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-heading)' }}
                      >
                        &ldquo;{item.query}&rdquo;
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-shrink-0">
                    {item.confidence != null && (
                      <span
                        className="hidden sm:inline font-mono text-xs"
                        style={{ color: 'var(--green)', fontWeight: 600 }}
                      >
                        {(item.confidence * 100).toFixed(0)}% conf
                      </span>
                    )}
                    <span className="flex items-center gap-1 hud-label">
                      <Clock className="w-2.5 h-2.5" />
                      {formatRelativeTime(item.createdAt)}
                    </span>
                    <ArrowRight
                      className="w-3.5 h-3.5 transition-colors group-hover:text-[var(--cyan)]"
                      style={{ color: 'var(--text-faint)' }}
                    />
                  </div>
                </Link>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
