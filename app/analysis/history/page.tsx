// app/analysis/history/page.tsx
// Filterable and searchable Analysis History archive table.
'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import type { AnalysisMode, AnalysisResult, AnalysisStatus } from '@/lib/types/analysis';
import { ModeBadge } from '@/components/ui/ModeBadge';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { EmptyState } from '@/components/ui/EmptyState';
import { HistoryTableSkeleton } from '@/components/ui/LoadingSkeletonPanel';
import {
  Search,
  Filter,
  Trash2,
  ExternalLink,
  RotateCcw,
  History,
  Satellite,
  SlidersHorizontal,
} from 'lucide-react';
import { formatTimestamp, formatRelativeTime } from '@/lib/utils';
import { toast } from 'sonner';

export default function HistoryPage() {
  const router = useRouter();

  const [items, setItems] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);

  // Filters
  const [selectedMode, setSelectedMode] = useState<AnalysisMode | 'all'>('all');
  const [selectedStatus, setSelectedStatus] = useState<AnalysisStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [minConfidence, setMinConfidence] = useState<number>(0);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const filters: any = {};
      if (selectedMode !== 'all') filters.mode = selectedMode;
      if (selectedStatus !== 'all') filters.status = selectedStatus;
      if (minConfidence > 0) filters.minConfidence = minConfidence / 100;

      const res = await api.listAnalysisHistory(filters);
      setItems(res.items);
    } catch (err) {
      toast.error('Failed to load history archive');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [selectedMode, selectedStatus, minConfidence]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await api.deleteAnalysis(id);
      setItems((prev) => prev.filter((item) => item.id !== id));
      toast.success(`Removed analysis ${id} from local cache`);
    } catch (err) {
      toast.error('Failed to delete analysis');
    }
  };

  const handleClearAll = () => {
    items.forEach((item) => api.deleteAnalysis(item.id));
    setItems([]);
    toast.info('Cleared history archive (Empty State Active)');
  };

  // Local text search
  const filteredItems = items.filter((item) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.query.toLowerCase().includes(q) ||
      item.id.toLowerCase().includes(q) ||
      (item.answerText && item.answerText.toLowerCase().includes(q))
    );
  });

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4 border-b border-[var(--border-hairline)] pb-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold font-heading text-[var(--text-primary)]">
            Analysis Telemetry Archive
          </h1>
          <p className="text-xs text-[var(--text-muted)] font-mono mt-0.5">
            Historical log of all vision-language inference queries and mission reports.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {items.length > 0 && (
            <button
              onClick={handleClearAll}
              className="px-3 py-1.5 rounded text-xs font-mono text-[var(--text-muted)] hover:text-[var(--accent-danger)] border border-[var(--border-hairline)] transition-colors"
            >
              Clear Demo Cache
            </button>
          )}

          <Link
            href="/analysis/new"
            className="flex items-center gap-1.5 px-4 py-2 rounded text-xs font-mono font-bold bg-[var(--accent-signal)] text-[#05070D] transition-colors shadow-sm"
          >
            <span>+ New Analysis</span>
          </Link>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="panel p-4 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 bg-[var(--bg-panel)]">
        {/* Search input */}
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-faint)]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search queries, keywords, or analysis IDs…"
            className="w-full pl-9 pr-3 py-2 rounded text-xs font-mono bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-primary)] placeholder:text-[var(--text-faint)] focus:outline-none focus:border-[var(--accent-signal)]"
          />
        </div>

        {/* Mode filter */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[0.65rem] font-mono text-[var(--text-faint)]">Mode:</span>
          <select
            value={selectedMode}
            onChange={(e) => setSelectedMode(e.target.value as any)}
            className="px-2.5 py-1.5 rounded text-xs font-mono bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-primary)] focus:outline-none"
          >
            <option value="all">All Modes</option>
            <option value="single_image">Single Image</option>
            <option value="bi_temporal">Bi-Temporal</option>
            <option value="optical_sar">Optical + SAR</option>
          </select>

          {/* Status filter */}
          <span className="text-[0.65rem] font-mono text-[var(--text-faint)] ml-2">Status:</span>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value as any)}
            className="px-2.5 py-1.5 rounded text-xs font-mono bg-[var(--bg-panel-elevated)] border border-[var(--border-hairline)] text-[var(--text-primary)] focus:outline-none"
          >
            <option value="all">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Table Content */}
      {loading ? (
        <HistoryTableSkeleton />
      ) : filteredItems.length === 0 ? (
        <div className="panel p-8">
          <EmptyState
            icon={History}
            title="No Matching Analysis Records"
            description={
              items.length === 0
                ? 'No historical runs found. Launch a new analysis to populate the archive.'
                : 'No records matched your active filter criteria. Try adjusting filters or search query.'
            }
            action={
              <Link
                href="/analysis/new"
                className="px-4 py-2 rounded font-mono text-xs bg-[var(--accent-signal)] text-[#05070D] font-bold"
              >
                Launch New Analysis
              </Link>
            }
          />
        </div>
      ) : (
        <div className="panel overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[var(--border-hairline)] bg-[var(--bg-panel-elevated)] font-mono text-[0.65rem] text-[var(--text-muted)]">
                <th className="p-3.5">ID / Mode</th>
                <th className="p-3.5">Inquiry Query</th>
                <th className="p-3.5">Status</th>
                <th className="p-3.5">Confidence</th>
                <th className="p-3.5">Timestamp</th>
                <th className="p-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)] font-mono text-xs">
              {filteredItems.map((item) => (
                <tr
                  key={item.id}
                  onClick={() => router.push(`/analysis/${item.id}`)}
                  className="hover:bg-[var(--bg-panel-hover)] transition-colors cursor-pointer"
                >
                  {/* ID + Mode */}
                  <td className="p-3.5 whitespace-nowrap">
                    <div className="flex flex-col gap-1">
                      <span className="font-bold text-[var(--accent-signal)]">
                        {item.id}
                      </span>
                      <ModeBadge mode={item.mode} showIcon={false} />
                    </div>
                  </td>

                  {/* Query */}
                  <td className="p-3.5 font-sans font-medium text-[var(--text-primary)] max-w-xs md:max-w-md truncate">
                    &ldquo;{item.query}&rdquo;
                  </td>

                  {/* Status */}
                  <td className="p-3.5 whitespace-nowrap">
                    <StatusBadge status={item.status} />
                  </td>

                  {/* Confidence */}
                  <td className="p-3.5 whitespace-nowrap">
                    {item.confidence != null ? (
                      <span className="font-bold text-[var(--accent-success)]">
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-[var(--text-faint)]">—</span>
                    )}
                  </td>

                  {/* Timestamp */}
                  <td className="p-3.5 whitespace-nowrap text-[var(--text-faint)] text-[0.68rem]">
                    {formatTimestamp(item.createdAt)}
                  </td>

                  {/* Actions */}
                  <td className="p-3.5 whitespace-nowrap text-right" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center justify-end gap-1.5">
                      <Link
                        href={`/analysis/${item.id}`}
                        className="p-1.5 rounded hover:bg-[var(--bg-panel-elevated)] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                        title="View Full Report"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </Link>

                      <button
                        type="button"
                        onClick={(e) => handleDelete(e, item.id)}
                        className="p-1.5 rounded hover:bg-red-500/10 text-[var(--text-muted)] hover:text-[var(--accent-danger)]"
                        title="Delete Record"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
